"""The M7 base-model bake-off from the command line (research_agent.eval.bakeoff; docs/m7-计划.md §4,
docs/m7-底座对比预登记.md).

    .venv/Scripts/python.exe scripts/bakeoff.py plan --candidate c2 [--config configs/bakeoff-m7.json]
    .venv/Scripts/python.exe scripts/bakeoff.py smoke --candidate c2 [--steps S0,S2] [MACHINE]
    .venv/Scripts/python.exe scripts/bakeoff.py pilot --candidate c2 [--stop-step 274] [MACHINE]
    .venv/Scripts/python.exe scripts/bakeoff.py evaluate --candidate c2 [--stop-step 274] [--rerun] [MACHINE]
    .venv/Scripts/python.exe scripts/bakeoff.py gate --candidate c2
    .venv/Scripts/python.exe scripts/bakeoff.py all --candidate c2 [MACHINE]
    .venv/Scripts/python.exe scripts/bakeoff.py aggregate
    .venv/Scripts/python.exe scripts/bakeoff.py fidelity --candidate c2 [--stop-step 548] [MACHINE]
    MACHINE: [--llama-server PATH] [--llama-cpp-src PATH] [--port 8080] [--no-wait]

plan prints every command of the candidate's smoke, pilot, evaluation and fidelity and runs nothing. smoke
runs the seven smoke steps (or the ones --steps lists), each into runs/bakeoff/m7/<candidate>/smoke.json;
pilot trains the candidate to the stop step after the preflight and exports the checkpoint, resuming a run
from its last saved checkpoint; evaluate runs every variant and b0 on the checkpoint's GGUF and scores them,
keeping the records that stand for a run already (--rerun runs every one again); gate writes the
candidate's gate.json and rebuilds the aggregate results file from every gate present; all goes on where an
earlier run of it stopped: it runs the smoke steps still to pass (and those that use their output), then
the pilot, the evaluation and the gate, and stops at the first failure that counts (a smoke step failed or
blocked, other than the ones the candidate runs for the record only; the pilot; the evaluation), the gate
written all the same so the aggregate sees the candidate; aggregate rebuilds results/bakeoff-m7.json alone,
applying the preregistered rule (research_agent.eval.bakeoff: provisional while a candidate is pending or
incomplete). --stop-step beyond the preregistered one is the extension (docs/m7-底座对比预登记.md 胜者 1): the
pilot resumes the run to it, the evaluation names its runs dev-<step>-<variant>, and the gate reads them as
the extension. fidelity is the winner's trained-adapter check, report only (docs/m7-底座对比预登记.md 修订 1;
research_agent.eval.bakeoff, The trained adapter's fidelity): the pilot's checkpoint, the extension's once the
run was extended (--stop-step names the stop step or the extension step, no other), its GGUF exported first
when missing, its top-1 agreement between the training stack and llama.cpp over S6's sequences beside the
agreement without an adapter, into runs/bakeoff/m7/<candidate>/fidelity-trained-<step>.json; it writes no
smoke record, gate or aggregate.

The machine's llama-server binary and llama.cpp source tree (the converter and gguf-py) come from
--llama-server and --llama-cpp-src or the LLAMA_SERVER and LLAMA_CPP_SRC variables, as for the other
scripts; smoke, pilot, evaluate, all and fidelity need both. --no-wait makes a preflight that does not hold
fail at once instead of polling for up to twelve hours. Run from the repository root: every path in the
records is repository-relative.

Exit 0 when the command did its work (a gate or an aggregate is written whatever it says, and a fidelity
record whatever agreement it holds); 1 when a smoke step failed or is blocked, the pilot or the evaluation
failed, the trained adapter's fidelity could not be compared (its record says why), or when all ends with a
gate that is not scored (ineligible, or incomplete); 2 on a usage error (an unknown candidate, the reference's
key, a malformed configuration, a missing machine path, a fidelity step the preregistration does not name,
argparse's exit); 3 when the preflight did not hold within the wait limit.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from research_agent.eval.bakeoff import (
    CONFIG,
    SMOKE_FILE,
    SRC_VAR,
    BakeoffError,
    MachineProbes,
    PreflightTimeout,
    Tools,
    aggregate,
    candidate_gate,
    default_fidelity_backends,
    gguf_tensor_count,
    load_config,
    plan_lines,
    read_json,
    rel,
    run_evaluate,
    run_pilot,
    run_script,
    run_smoke,
    run_trained_fidelity,
    smoke_plan,
    smoke_status,
    start_script,
    tool_round_trip,
)
from research_agent.family.core import load_family
from research_agent.llm.local import LocalClient
from research_agent.llm.server import SERVER_VAR, llama_server

MACHINE = ("smoke", "pilot", "evaluate", "all", "fidelity")  # the commands that need the machine's tools
WAIT_LIMIT = 12 * 3600.0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="The M7 base-model bake-off: smoke, pilot, evaluation, gate and "
                                             "the aggregate decision.")
    sub = ap.add_subparsers(dest="command", required=True)

    def command(name: str, *, candidate: bool, machine: bool,
                stop_step: bool = False) -> argparse.ArgumentParser:
        parser = sub.add_parser(name)
        parser.add_argument("--config", default=str(CONFIG), help=f"the configuration (default {CONFIG})")
        if candidate:
            parser.add_argument("--candidate", required=True, help="the candidate's key, e.g. c2")
        if stop_step:
            parser.add_argument("--stop-step", type=int, default=None,
                                help="the checkpoint step (default: the schedule's; the extension step "
                                     "resumes the run)")
        if machine:
            parser.add_argument("--llama-server", default=None,
                                help=f"llama-server executable (default: ${SERVER_VAR})")
            parser.add_argument("--llama-cpp-src", default=None,
                                help=f"llama.cpp source tree (default: ${SRC_VAR})")
            parser.add_argument("--port", type=int, default=8080)
            parser.add_argument("--no-wait", action="store_true",
                                help="fail at once when the preflight does not hold, instead of polling")
        return parser

    command("plan", candidate=True, machine=False)
    command("smoke", candidate=True, machine=True).add_argument(
        "--steps", default=None, help="the smoke steps to run, comma-separated (default: all seven)")
    command("pilot", candidate=True, machine=True, stop_step=True)
    command("evaluate", candidate=True, machine=True, stop_step=True).add_argument(
        "--rerun", action="store_true",
        help="run every variant and b0 again, also those whose records stand for the run already")
    command("gate", candidate=True, machine=False)
    command("all", candidate=True, machine=True)
    command("aggregate", candidate=False, machine=False)
    command("fidelity", candidate=True, machine=True).add_argument(
        "--stop-step", type=int, default=None,
        help="the checkpoint whose trained adapter is compared: the schedule's stop step or its extension "
             "step (default: the extension step once the run was extended, else the stop step)")
    return ap


def build_tools(args: argparse.Namespace, *, needs_machine: bool = False) -> Tools:
    """The Tools of this machine; BakeoffError when a command that needs the machine lacks a path. The
    scripts run under this interpreter, named relative to the repository root (the project's .venv) so that
    the commands the records keep hold no machine path."""
    try:
        server = llama_server(getattr(args, "llama_server", None))
    except LookupError as exc:
        if needs_machine:
            raise BakeoffError(str(exc)) from None
        server = None
    src = getattr(args, "llama_cpp_src", None) or os.environ.get(SRC_VAR)
    if needs_machine and not src:
        raise BakeoffError(f"no llama.cpp source tree given: pass --llama-cpp-src or set {SRC_VAR}")
    src_path = Path(src) if src else None
    return Tools(run=run_script, start=start_script, popen=subprocess.Popen, client=LocalClient,
                 load_family=lambda folder, name: load_family(folder, name=name), probes=MachineProbes(),
                 count_gguf_tensors=lambda path: gguf_tensor_count(path, src_path),
                 fidelity_backends=default_fidelity_backends, round_trip=tool_round_trip, sleep=time.sleep,
                 clock=time.monotonic, python=rel(sys.executable), llama_server=server,
                 llama_cpp_src=src_path,
                 wait_limit=0.0 if getattr(args, "no_wait", False) else WAIT_LIMIT,
                 port=getattr(args, "port", 8080))


def main(argv: list[str] | None = None, *, tools: Tools | None = None) -> int:
    """tools stands in for the machine's in tests."""
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    try:
        config = load_config(args.config)
        tools = tools or build_tools(args, needs_machine=args.command in MACHINE)
        candidate = config.candidate(args.candidate) if args.command != "aggregate" else None
    except (BakeoffError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    try:
        if args.command == "plan":
            for line in plan_lines(config, candidate, tools):
                print(line)
            return 0
        if args.command == "aggregate":
            report_aggregate(config)
            return 0
        if args.command == "gate":
            report_gate(config, candidate)
            return 0
        if args.command == "smoke":
            steps = None if args.steps is None else [s.strip() for s in args.steps.split(",") if s.strip()]
            smoke = run_smoke(config, candidate, tools, steps=steps)
            return 0 if not smoke["failed"] and not smoke["blocked"] else 1
        if args.command == "pilot":
            return 0 if run_pilot(config, candidate, tools, stop_step=args.stop_step)["passed"] else 1
        if args.command == "evaluate":
            evaluated = run_evaluate(config, candidate, tools, stop_step=args.stop_step, rerun=args.rerun)
            return 0 if evaluated["passed"] else 1
        if args.command == "fidelity":
            checked = run_trained_fidelity(config, candidate, tools, stop_step=args.stop_step)
            return 0 if checked["error"] is None else 1
        return run_all(config, candidate, tools)
    except BakeoffError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    except PreflightTimeout as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 3


def report_gate(config, candidate) -> dict:
    """Write the candidate's gate and rebuild the aggregate, both reported in one line each; the gate."""
    gate = candidate_gate(config, candidate)
    why = gate["reasons"] or gate["missing"]
    where = (config.folder(candidate.key) / "gate.json").as_posix()
    print(f"{candidate.key}: P {gate['P']}, FT3 {gate['ft3_hours_estimate']} h, {gate['status']}"
          + (f": {'; '.join(why)}" if why else "") + f"; gate {where}")
    report_aggregate(config)
    return gate


def report_aggregate(config) -> None:
    result = aggregate(config)
    decision = result["decision"]
    if decision["status"] == "provisional":
        outcome = f"leader {decision['leader']}, waiting for {', '.join(decision['waiting'])}"
    else:
        outcome = f"winner {decision['winner']}"
    print(f"{config.results.as_posix()}: {decision['status']}, {outcome}")


def run_all(config, candidate, tools: Tools) -> int:
    """The smoke steps still to pass (smoke_plan), the pilot, the evaluation and the gate in turn, stopping at
    the first failure that counts (the gate written anyway): a smoke step that failed or is blocked stops it
    unless the candidate runs it for the record only, as smoke_status reads the steps."""
    steps = smoke_plan(read_json(config.folder(candidate.key) / SMOKE_FILE), candidate)
    smoke = run_smoke(config, candidate, tools, steps=steps)
    _, reasons, missing = smoke_status(smoke, candidate)
    if reasons or missing:
        print(f"stopping: {'; '.join(reasons + missing)}")
        report_gate(config, candidate)
        return 1
    pilot = run_pilot(config, candidate, tools)
    if not pilot["passed"]:
        print(f"stopping: the pilot failed: {pilot['error']}")
        report_gate(config, candidate)
        return 1
    evaluated = run_evaluate(config, candidate, tools)
    if not evaluated["passed"]:
        print(f"stopping: the evaluation failed: {evaluated['error']}")
        report_gate(config, candidate)
        return 1
    return 0 if report_gate(config, candidate)["status"] == "scored" else 1


if __name__ == "__main__":
    sys.exit(main())

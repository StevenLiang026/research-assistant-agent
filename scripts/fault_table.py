"""Build the M4 fault table (docs/plan.md, 故障注入) from the reports of several end-to-end runs: per system
and fault condition, how the tasks fared under injected faults.

    .venv/Scripts/python.exe scripts/fault_table.py runs/eval/e2e-ft-f0/report.json \\
        runs/eval/e2e-ft-f20/report.json runs/eval/e2e-ft-f50/report.json [...] \\
        [--out results/faults-m4.json] [--force]

Inputs. The report.json of each run of scripts/run_e2e.py (a run folder is taken for its report.json), with
its timing sidecar timing.json beside it. Each run's manifest names its system ("system") and its fault
condition ("fault_injection": rate and persistent; a manifest without it is a run at rate 0, which run_e2e
writes as it wrote before M4). Every input must be a different (system, rate, persistent), and all must share
the tasks file's SHA-256 and the seed, since the systems are compared task by task on the same faults
(net.faults: a task meets the same fault on its k-th request for a URL in every system). A record of a run
with faults must carry "faults" (research_agent.eval.e2e, Fault injection); one of a run at rate 0 has none
and met none.

Per system and condition ("rows", sorted by system, rate, persistent), from the records alone:

    tasks, successes, success_rate      the records, those that succeeded and their share
    tasks_with_faults                   tasks that met at least one fault (a delay included)
    recovered, recovery_rate            of those, the tasks that succeeded and their share (null without any)
    empty_result_tasks                  tasks a tool answered EMPTY_RESULTS or NOT_FOUND (record tool_errors;
                                        an injected empty_results gives one, so does an unknown id the model
                                        asked for, so the rate-0 row is the baseline)
    fabricated_ids_after_empty          of those, the tasks that then failed the cited_ids check (every arXiv
                                        id of the final text must have appeared in a tool output of the turn)
    retry_storms, retry_storm_tasks     tasks with more than STORM_CALLS calls to one tool whose previous
                                        call of that tool in the task met an injected failure (FAILURE_KINDS:
                                        a fault that leaves the call without its content; a delay answers,
                                        and an html_404 answers with the abstract). A call's position among
                                        the task's calls of its tool is the fault's "call", and the calls of
                                        a tool are the record's tool_calls count. The agent loop already
                                        refuses a call that ran twice (agent/loop.py, MAX_RUNS_PER_CALL; such
                                        a call is answered SKIPPED, not by a fault), so a storm shows across
                                        arguments
    by_kind                             per fault of net.faults.FAULTS: "met" (times the fault was met),
                                        "tasks" (tasks that met it) and "succeeded" (of those, the tasks
                                        that still succeeded)
    delay                               from the timing sidecars' arXiv tool calls ("tools"): the fetch
                                        calls (get_paper, fetch_fulltext) that met a delay and nothing else
                                        and those that met no fault, the mean wall time of each in
                                        milliseconds and the difference, the extra latency per delayed fetch
                                        call (null when either group is empty). The search meets no delay
                                        (net.faults)

Writes --out (default results/faults-m4.json): JSON, two-space indent, LF, no time, so the same runs give the
same bytes: the tasks file's SHA-256, the seed, the storm threshold, every input (the report and the timing
sidecar, each with its SHA-256, as POSIX paths relative to the current folder when inside it: run it from the
repository root; the system and the condition) in the order given, and the rows. An existing --out is kept
unless --force is given. Nothing is registered in results/numbers.json.

Exit 0 when written; 1 on malformed input (a report or sidecar that is missing or not JSON, a manifest
without a system, a record of a run with faults that lacks "faults"); 2 when the inputs do not belong
together (one system and condition twice, other tasks or another seed) or on a usage error; 3 when --out
exists and --force was not given. Nothing is written unless the exit is 0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from research_agent.net.faults import FAULTS

OUT = Path("results/faults-m4.json")
REPORT_FILE = "report.json"
TIMING_FILE = "timing.json"
STORM_CALLS = 3  # more calls than this right after an injected failure of the same tool is a storm
FAILURE_KINDS = ("timeout", "rate_limited", "server_error", "empty_results", "truncated_atom")
FETCH_TOOLS = ("get_paper", "fetch_fulltext")
EMPTY_CODES = ("EMPTY_RESULTS", "NOT_FOUND")


class Malformed(ValueError):
    """An input that is missing or not of its kind: exit 1."""


class Unrelated(ValueError):
    """Inputs that do not belong together: exit 2."""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build the M4 fault table from end-to-end run reports.")
    ap.add_argument("reports", nargs="+", type=Path, help="each run's report.json (or its run folder)")
    ap.add_argument("--out", type=Path, default=OUT, help=f"default {OUT.as_posix()}")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    return ap


def rel(path: str | Path) -> str:
    """path relative to the current folder in POSIX form when it lies inside it, else absolute (POSIX)."""
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return absolute.as_posix()


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Malformed(f"{rel(path)}: {exc}") from None
    if not isinstance(data, dict):
        raise Malformed(f"{rel(path)}: not a JSON object")
    return data


def condition(rate: float, persistent: bool) -> str:
    """The condition's label: rate=<rate> with ",persistent" for a persistent one."""
    return f"rate={rate:g}" + (",persistent" if persistent else "")


def load_run(path: Path) -> dict:
    """One input: its report and sidecar read and checked, with its system, condition and SHA-256s."""
    report_path = path / REPORT_FILE if path.is_dir() else path
    timing_path = report_path.with_name(TIMING_FILE)
    report, timing = read_json(report_path), read_json(timing_path)
    manifest, records = report.get("manifest"), report.get("tasks")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("system"), str):
        raise Malformed(f"{rel(report_path)}: the manifest names no system")
    if not isinstance(records, list) or not all(isinstance(r, dict) and "id" in r for r in records):
        raise Malformed(f"{rel(report_path)}: tasks must be a list of records")
    injection = manifest.get("fault_injection")
    rate = float(injection["rate"]) if injection else 0.0
    persistent = bool(injection["persistent"]) if injection else False
    if injection and any(not isinstance(r.get("faults"), list) for r in records):
        raise Malformed(f"{rel(report_path)}: a run with faults whose record lacks \"faults\"")
    timed = timing.get("tasks")
    if not isinstance(timed, list):
        raise Malformed(f"{rel(timing_path)}: tasks must be a list")
    return {"report_path": report_path, "timing_path": timing_path, "system": manifest["system"],
            "rate": rate, "persistent": persistent, "seed": manifest.get("seed"),
            "tasks_sha256": manifest.get("tasks_sha256"), "records": records, "timing": timed}


def storm(record: dict) -> bool:
    """Whether the task made more than STORM_CALLS calls to one tool right after that tool's previous call
    met an injected failure (module docstring)."""
    for tool, calls in (record.get("tool_calls") or {}).items():
        failed = {f["call"] for f in record.get("faults") or []
                  if f["tool"] == tool and f["kind"] in FAILURE_KINDS and f["call"] is not None}
        if sum(position - 1 in failed for position in range(1, calls)) > STORM_CALLS:
            return True
    return False


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def delay_row(timing: list[dict]) -> dict:
    """The delay cell (module docstring) from a run's timed arXiv tool calls."""
    calls = [c for task in timing for c in task.get("tools") or [] if c.get("tool") in FETCH_TOOLS]
    delayed = [c["wall_ms"] for c in calls if c.get("faults") == ["delay"]]
    clean = [c["wall_ms"] for c in calls if c.get("faults") == []]
    mean_delayed, mean_clean = _mean(delayed), _mean(clean)
    extra = None if mean_delayed is None or mean_clean is None else round(mean_delayed - mean_clean, 1)
    return {"delayed_fetch_calls": len(delayed), "clean_fetch_calls": len(clean),
            "mean_ms_delayed": None if mean_delayed is None else round(mean_delayed, 1),
            "mean_ms_clean": None if mean_clean is None else round(mean_clean, 1), "extra_ms_per_call": extra}


def row(run: dict) -> dict:
    """One system and condition (module docstring)."""
    records = run["records"]
    faulted = [r for r in records if r.get("faults")]
    empty = [r for r in records if any((r.get("tool_errors") or {}).get(code) for code in EMPTY_CODES)]
    fabricated = [r for r in empty if (r.get("checks") or {}).get("cited_ids", {}).get("ok") is False]
    storms = sorted(r["id"] for r in records if storm(r))
    by_kind = {}
    for kind in FAULTS:
        met = [r for r in records for f in r.get("faults") or [] if f["kind"] == kind]
        tasks = {id(r): r for r in met}.values()
        by_kind[kind] = {"met": len(met), "tasks": len(tasks),
                         "succeeded": sum(bool(r["success"]) for r in tasks)}
    successes = sum(bool(r["success"]) for r in records)
    recovered = sum(bool(r["success"]) for r in faulted)
    return {"system": run["system"], "condition": condition(run["rate"], run["persistent"]),
            "rate": run["rate"], "persistent": run["persistent"], "tasks": len(records),
            "successes": successes,
            "success_rate": successes / len(records) if records else None,
            "tasks_with_faults": len(faulted), "recovered": recovered,
            "recovery_rate": recovered / len(faulted) if faulted else None,
            "empty_result_tasks": len(empty), "fabricated_ids_after_empty": len(fabricated),
            "retry_storms": len(storms), "retry_storm_tasks": storms, "by_kind": by_kind,
            "delay": delay_row(run["timing"])}


def build(paths: list[Path]) -> dict:
    """The table over the runs at paths; Malformed or Unrelated naming what is wrong."""
    runs = [load_run(path) for path in paths]
    seen: dict[tuple, str] = {}
    for run in runs:
        key = (run["system"], run["rate"], run["persistent"])
        if key in seen:
            raise Unrelated(f"{rel(run['report_path'])}: system {run['system']} at "
                            f"{condition(run['rate'], run['persistent'])} is given twice (also {seen[key]})")
        seen[key] = rel(run["report_path"])
    for field, name in (("tasks_sha256", "tasks"), ("seed", "seed")):
        values = {json.dumps(run[field]) for run in runs}
        if len(values) > 1:
            raise Unrelated(f"the runs differ in their {name} ({field}: {', '.join(sorted(values))}); the "
                            "table compares systems on the same tasks and faults")
    inputs = [{"report": rel(run["report_path"]), "report_sha256": sha256_of(run["report_path"]),
               "timing": rel(run["timing_path"]), "timing_sha256": sha256_of(run["timing_path"]),
               "system": run["system"], "condition": condition(run["rate"], run["persistent"])}
              for run in runs]
    rows = [row(run) for run in sorted(runs, key=lambda r: (r["system"], r["rate"], r["persistent"]))]
    return {"tasks_sha256": runs[0]["tasks_sha256"], "seed": runs[0]["seed"], "storm_calls": STORM_CALLS,
            "failure_kinds": list(FAILURE_KINDS), "inputs": inputs, "rows": rows}


def percent(rate: float | None) -> str:
    return "-" if rate is None else f"{100 * rate:.1f}%"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    if args.out.exists() and not args.force:
        print(f"FAIL: {args.out} exists; --force overwrites it", file=sys.stderr)
        return 3
    try:
        table = build(args.reports)
    except Malformed as exc:
        print(f"FAIL: malformed input: {exc}", file=sys.stderr)
        return 1
    except Unrelated as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                        newline="\n")
    print(f"{rel(args.out)}: {len(table['rows'])} row(s)")
    for r in table["rows"]:
        print(f"  {r['system']:<4} {r['condition']:<22} n {r['tasks']:>3}"
              f"  success {percent(r['success_rate'])}"
              f"  faulted {r['tasks_with_faults']:>3}  recovery {percent(r['recovery_rate'])}"
              f"  fabricated {r['fabricated_ids_after_empty']}  storms {r['retry_storms']}"
              f"  delay +{r['delay']['extra_ms_per_call']} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""BFCL single-turn regression of M3 (phase 3 of the M3 plan): B0, FT and optionally B4 on BFCL's single-turn
Python categories, each served by a llama-server of its own, scored by BFCL, summed up by our own aggregate.

    .venv/Scripts/python.exe scripts/run_bfcl.py [--systems b0,ft[,b4]] [--categories C1,C2,...] \\
        [--ft-adapter GGUF] [--b4-adapter GGUF] [--drop-env PATTERN ...] [--out results/bfcl-m3.json] \\
        [--raw runs/bfcl/m3] [--bfcl-python .venv-bfcl/Scripts/python.exe] [--llama-server PATH] \\
        [--model models/Qwen3-1.7B-Q8_0.gguf] [--tokenizer models/Qwen3-1.7B] [--port 8080] \\
        [--timeout 120] [--dry-run] [-- extra llama-server arguments]

Systems. b0 is the base model on a server without an adapter; ft and b4 are the base model on a server that
loads an adapter GGUF (--lora <gguf> --lora-init-without-apply). BFCL cannot send llama.cpp's per-request
"lora" field, and llama.cpp b11349 applies a loaded adapter at its global scale 1.0 to every request without
one (spike S1b, docs/m0-验证.md), so each system has a server of its own, one after the other, in the order
b0, ft, b4. The adapter is the checkpoint chosen on the development set: the "chosen" GGUF of
results/checkpoint-selection-<system>.json (scripts/select_checkpoint.py), whose SHA-256 must still be the one
the selection recorded, unless --ft-adapter or --b4-adapter names a GGUF, which is then taken as it is. B0 and
FT are required (--systems must name both), B4 is optional.

Servers. research_agent.llm.server.LocalServer with the command of the M1 baseline (docs/m1-进度.md): -ngl 99,
-c 12288 (the longest BFCL prompt, 6,182 tokens, plus BFCL's 4,096-token reply budget), --jinja, bound to
127.0.0.1, llama-server's cache of past prompts off (--cache-ram 0) and -np 1: BFCL submits up to 100 requests
at once, and with one slot an answer does not depend on what it was batched with. Unlike M1, the prompt cache
is off too (--no-cache-prompt): BFCL sends no cache_prompt field, and by default the slot would reuse the
prefix a prompt shares with the request before it (999 of the 1,000 requests in the M1 B0 server log did),
which changes a greedy token now and then (docs/m2-进度.md, measured 2026-10-15). Arguments after "--" are
appended to every server's command, and recorded. The server's /props must report one slot of 12,288 tokens.
The b0 server must list no adapter. An adapter server must list exactly its system's GGUF, and it is probed
before BFCL starts: a one-token completion of a fixed prompt at lora scale 1, at scale 0 and without the
field, in that order; the answer without the field (its text and log-probabilities) must equal scale 1's and
differ from scale 0's, so BFCL's requests are known to get the adapter (a llama.cpp that honoured
--lora-init-without-apply would otherwise have the FT run measure the base model). Every server is then
probed for the cache: the same prompt twice as BFCL sends a request, without cache_prompt and lora fields;
neither answer may report prompt tokens taken from the cache (usage.prompt_tokens_details.cached_tokens,
which b11349 reports: over 1,040 for every request of the 2026-10-15 runs that reused prefixes, 0 for every
one that did not), so BFCL's prompts are known to be evaluated from scratch. The server's output goes to
<raw>/<system>/server.log, and the server is stopped once BFCL has generated, before the evaluation.

BFCL runs in its own environment (--bfcl-python, the .venv-bfcl interpreter: docs/m0-验证.md, S5), through
scripts/bfcl_nothink.py (BFCL's Qwen FC handler plus the empty thinking block): model
"Qwen/Qwen3-1.7B-FC-nothink" on the b0 server and "research-agent/Qwen3-1.7B-FT-FC-nothink" on an adapter
server, b4's too (each system has a BFCL folder of its own), the categories (default: BFCL's "python" group,
the 11 single-turn categories of the M1 baseline), --skip-server-setup and --local-model-path <tokenizer>, at
BFCL's default temperature 0.001, as in M1. The environment adjustments apply to BFCL's processes only,
never to this one or to any global setting (the M0 method): BFCL_PROJECT_ROOT (the system's raw folder as an
absolute path: BFCL writes result/ and score/ there), REMOTE_OPENAI_BASE_URL (the server's /v1), PYTHONUTF8=1
(BFCL prints emoji), HF_HUB_OFFLINE=1 (the tokenizer comes from the local folder), TQDM_DISABLE=1, and without
the inherited variables whose names match a --drop-env pattern (a regular expression, searched in any case):
those that would make BFCL's HTTP clients fail on the local server (S5). The patterns come from the command
line, as they did for the M1 run, and are not recorded. Before generating, a
check in that environment fetches the server's /v1/models with requests, as BFCL's readiness check does, and
through the OpenAI SDK (OpenAI(...).models.list()), which sends BFCL's requests over the HTTP transport the
installed SDK chooses: BFCL itself would wait for ever on a server it cannot reach.

Checks of BFCL's output. BFCL records a failed request as the answer "Error during inference: ..." and still
exits 0 (the M1 baseline lost four categories to HTTP 503s that way), so after generation every category must
have its result file and no such answer (exit 4) before anything is evaluated. BFCL also keeps the answers it
finds in its folder and generates only the missing ones, which would mix two runs: a system's raw folder that
already holds a file is refused before anything starts, and a failed run's folders stay for inspection, to be
moved away before the next run; a server that could not be started at all leaves nothing to inspect, and its
empty log goes, with the system's raw folder when that is then empty. The evaluation runs without
--partial-eval (BFCL then refuses a result file that misses questions) and skips a category without a result
file, so every category needs its score file, whose header must agree with itself (accuracy = correct_count /
total_count), with its list of failed questions and with the result file's answers; the systems must have the
same n in every category.

The aggregate, our own single-turn score: the unweighted mean of the per-category accuracies over the
categories run, each category's n recorded beside it. BFCL's Overall counts every category not run (Java,
JavaScript, multi-turn, agentic) as 0, and its latency and cost columns include queueing time
(docs/m1-进度.md): neither is used.

Writes <out> (results/bfcl-m3.json by default): JSON, two-space indent, LF, repository-relative POSIX paths
and no time, so the same answers give the same bytes: the categories, bfcl_eval's version, the aggregate rule,
the base model, the tokenizer folder, the variables set for BFCL, the git commit and whether the tree was
dirty, and per system the BFCL model, the server command (the binary named llama-server), the server's build,
the adapter {gguf, sha256, selection} (null for b0), the raw folder, the generate and evaluate commands, per
category {n, correct, accuracy} and the aggregate {categories, n, accuracy}. Nothing is registered in
results/numbers.json (the phase-4 report does that). The raw outputs stay under <raw>/<system>/ (runs/ is not
tracked): BFCL's result/ and score/, server.log, generate.log and evaluate.log.

--dry-run makes the same checks (the systems and categories, the binary, the base model, the tokenizer folder,
the BFCL interpreter, the adapters and their SHA-256, clean raw folders, paths inside the current folder) and
prints every command with the environment BFCL would get; it starts nothing and writes nothing. Run from the
repository root: a path outside the current folder is refused, since the record could hold it only as a
machine path.

Exit 0 when the results are written (or, with --dry-run, listed); 1 when BFCL's output is incomplete or
inconsistent (a category without its result or score file, a file that is not JSON lines, counts that
disagree, systems whose n differ); 2
on a usage error (a system list without b0 and ft, an unknown category, a --drop-env pattern that is not a
regular expression, no llama-server binary, base model, tokenizer folder or BFCL interpreter, a BFCL
interpreter that cannot be started for the version or the reach check, a llama-server binary that cannot be
run at all (cannot_run: not there, not permitted, not a program of this system), an interpreter without
bfcl_eval, an adapter that is missing or changed since its selection, a raw folder holding an earlier run, a
path outside the current folder, a busy port, BFCL's environment unable to reach the server); 3 when a server
could not be started for another reason, did not become ready, did not report one slot of 12,288 tokens,
listed other adapters than its system's, did not apply its adapter to a request without a lora field, or took
prompt tokens from its cache for a request without a cache_prompt field (or did not report how many); 4 when
a BFCL step exited non-zero, could not be started, or its generation recorded inference errors. Every failure
is a FAIL line, an OSError from starting a process included.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import httpx

from research_agent.eval.lock import sha256_of
from research_agent.llm.local import LocalClient, parse_logprobs
from research_agent.llm.server import (
    HOST,
    MODEL,
    SERVER_VAR,
    LocalServer,
    NotReady,
    PortBusy,
    llama_server,
    same_path,
    server_command,
)
from research_agent.train.manifest import REPO, git_commit, git_dirty

SCRIPTS = Path(__file__).resolve().parent
NOTHINK = SCRIPTS / "bfcl_nothink.py"  # runs in the BFCL environment
BFCL_PYTHON = Path(".venv-bfcl/Scripts/python.exe") if os.name == "nt" else Path(".venv-bfcl/bin/python")
TOKENIZER = "models/Qwen3-1.7B"
OUT = Path("results/bfcl-m3.json")
RAW = Path("runs/bfcl/m3")
SELECTION = "results/checkpoint-selection-{}.json"
# CreateProcess image errors that Windows reports as EINVAL: an application for another CPU (216), an
# unrecognised or truncated image (11, 191, 192), a corrupt file (1392); 193 maps to ENOEXEC already
WINDOWS_NOT_A_PROGRAM = frozenset({11, 191, 192, 193, 216, 1392})
SYSTEMS = ("b0", "ft", "b4")  # in the order they run
REQUIRED = ("b0", "ft")
ADAPTER_SYSTEMS = ("ft", "b4")
# The names scripts/bfcl_nothink.py registers, one per kind of server: without an adapter, with one.
BFCL_MODELS = {"b0": "Qwen/Qwen3-1.7B-FC-nothink", "ft": "research-agent/Qwen3-1.7B-FT-FC-nothink",
               "b4": "research-agent/Qwen3-1.7B-FT-FC-nothink"}
# BFCL's "python" group (bfcl_eval.constants.category_mapping), the 11 categories of the M1 baseline
CATEGORIES = ("simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance",
              "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple", "live_irrelevance",
              "live_relevance")
CONTEXT = 12288  # the longest prompt, 6,182 tokens (bfcl_nothink.py self-check), plus 4,096 reply tokens
SLOTS = 1
BFCL_ENV = {"PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1", "TQDM_DISABLE": "1"}
VERSION_CODE = "import importlib.metadata as m; print(m.version('bfcl_eval'))"
# BFCL's readiness check (requests, in its OSS handler) and its requests (the OpenAI SDK, over the transport
# the SDK chooses), each with the environment's connection settings, against the server's /v1 given as the
# argument; no retries, so an unreachable server fails within the timeout
REACH_CODE = ("import sys, requests; from openai import OpenAI; url = sys.argv[1]; "
              "requests.get(url + '/models', timeout=30).raise_for_status(); "
              "OpenAI(base_url=url, api_key='EMPTY', timeout=30, max_retries=0).models.list()")
REACH_TIMEOUT = 120.0
INFERENCE_ERROR = "Error during inference"  # BFCL's answer for a failed request (multi_threaded_inference)
PROBE_PROMPT = ("<|im_start|>user\nFind this week's papers on speculative decoding.<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n")
AGGREGATE_RULE = ("The unweighted mean of the per-category accuracies (correct / n) over the categories run, "
                  "each category counting once whatever its n, which is recorded beside it. BFCL's own "
                  "Overall is not used: it counts every category not run (Java, JavaScript, multi-turn, "
                  "agentic) as 0.")
LOG_TAIL = 20


class UsageError(ValueError):
    """An input this script cannot run with (exit 2)."""


class OutputError(ValueError):
    """BFCL's output is incomplete or inconsistent (exit 1)."""


@dataclass(frozen=True)
class Adapter:
    gguf: Path
    sha256: str
    selection: Path | None  # the selection file that chose it; None when --<system>-adapter named it


@dataclass(frozen=True)
class Plan:
    """What a run does for one system: its server, BFCL's two steps and their environment settings."""

    system: str
    model: str  # BFCL's model name
    adapter: Adapter | None
    root: Path  # the system's raw folder, BFCL_PROJECT_ROOT
    serve: list[str]  # the server command as run
    record: list[str]  # the same with the binary named llama-server, as recorded
    generate: list[str]
    evaluate: list[str]
    settings: dict[str, str]  # the variables set for BFCL's processes


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="BFCL single-turn regression: B0, FT and optionally B4, each on "
                                             "a llama-server of its own.")
    ap.add_argument("--systems", default=",".join(REQUIRED),
                    help="comma-separated: b0 and ft, optionally b4 (default b0,ft)")
    ap.add_argument("--categories", default=",".join(CATEGORIES),
                    help="comma-separated single-turn categories (default: BFCL's python group, the 11 of "
                         "the M1 baseline)")
    for system in ADAPTER_SYSTEMS:
        ap.add_argument(f"--{system}-adapter", type=Path, default=None,
                        help=f"the {system} adapter GGUF (default: the chosen one of "
                             f"{SELECTION.format(system)})")
    ap.add_argument("--drop-env", action="append", default=None, metavar="PATTERN",
                    help="drop the inherited variables whose names match PATTERN (a regular expression, any "
                         "case) from BFCL's processes only: those that keep its HTTP clients from the local "
                         "server; repeatable")
    ap.add_argument("--out", type=Path, default=OUT, help=f"the results (default {OUT.as_posix()})")
    ap.add_argument("--raw", type=Path, default=RAW,
                    help=f"the folder of the systems' raw outputs (default {RAW.as_posix()})")
    ap.add_argument("--bfcl-python", type=Path, default=BFCL_PYTHON,
                    help=f"the BFCL environment's interpreter (default {BFCL_PYTHON.as_posix()})")
    ap.add_argument("--llama-server", default=None,
                    help=f"llama-server executable (default: ${SERVER_VAR}; one of the two is required)")
    ap.add_argument("--model", default=MODEL, help=f"base model GGUF (default {MODEL})")
    ap.add_argument("--tokenizer", default=TOKENIZER,
                    help=f"the model folder BFCL counts prompt tokens with (default {TOKENIZER})")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for each server")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and the environment; start nothing, write nothing")
    ap.add_argument("server_args", nargs="*", help="extra llama-server arguments, after --")
    return ap


def rel(path: str | Path) -> str:
    """path relative to the current folder in POSIX form when it lies inside it, else absolute (POSIX)."""
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return absolute.as_posix()


def inside(path: str | Path) -> bool:
    return not Path(rel(path)).is_absolute()


def chosen(value: str, allowed: tuple[str, ...], what: str) -> list[str]:
    """The comma-separated names of value in the order of allowed; UsageError for an unknown one or none."""
    names = [name.strip() for name in value.split(",") if name.strip()]
    unknown = [name for name in names if name not in allowed]
    if unknown or not names:
        raise UsageError(f"{what} {unknown or value!r}: choose from {', '.join(allowed)}")
    return [name for name in allowed if name in names]


def child_env(base: Mapping[str, str], drop: Iterable[str], settings: Mapping[str, str]) -> dict[str, str]:
    """The environment of a BFCL process: base without the variables whose names match a pattern of drop
    (searched, in any case), with settings added; base is left as it is."""
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in drop]
    env = {name: value for name, value in base.items() if not any(p.search(name) for p in patterns)}
    env.update(settings)
    return env


def chosen_adapter(system: str, explicit: Path | None) -> Adapter:
    """The GGUF named on the command line, else the one results/checkpoint-selection-<system>.json chose,
    whose SHA-256 must still match the selection's; UsageError when it is missing or has changed."""
    if explicit is not None:
        if not explicit.is_file():
            raise UsageError(f"{system}: the adapter {rel(explicit)} is not here (--{system}-adapter)")
        return Adapter(explicit, sha256_of(explicit), None)
    selection = Path(SELECTION.format(system))
    if not selection.is_file():
        raise UsageError(f"{system}: no adapter: {selection.as_posix()} is not here "
                         f"(scripts/select_checkpoint.py runs/train/{system} writes it); or name the GGUF "
                         f"with --{system}-adapter")
    try:
        choice = json.loads(selection.read_text(encoding="utf-8"))["chosen"]
        gguf, recorded = Path(choice["gguf"]), choice["sha256"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise UsageError(f"{selection.as_posix()} names no chosen GGUF with its SHA-256 "
                         f"({type(exc).__name__}: {exc})") from None
    if not gguf.is_file():
        raise UsageError(f"{system}: {selection.as_posix()} chose {gguf.as_posix()}, which is not here")
    sha256 = sha256_of(gguf)
    if sha256 != recorded:
        raise UsageError(f"{system}: {gguf.as_posix()} has changed since {selection.as_posix()} chose it "
                         f"(sha256 {sha256}, the selection recorded {recorded})")
    return Adapter(gguf, sha256, selection)


def bfcl_command(python: Path, step: str, model: str, categories: list[str], tokenizer: str) -> list[str]:
    cmd = [os.fspath(python), rel(NOTHINK), step, "--model", model, "--test-category", ",".join(categories)]
    if step == "generate":
        cmd += ["--skip-server-setup", "--local-model-path", tokenizer]
    return cmd


def as_recorded(cmd: list[str]) -> list[str]:
    """A BFCL command as the results record it: the interpreter as a repository-relative POSIX path."""
    return [rel(cmd[0]), *cmd[1:]]


def server_extra(args: argparse.Namespace) -> list[str]:
    """What follows llama.server's own arguments in every server's command: one slot and the prompt cache off
    (module docstring), then the arguments given after "--"."""
    return ["-np", str(SLOTS), "--no-cache-prompt", *args.server_args]


def prepare(args: argparse.Namespace) -> tuple[list[Plan], Path, list[str]]:
    """The plan of every system, the llama-server binary and the categories, after the checks of the module
    docstring that need nothing started; UsageError otherwise."""
    systems = chosen(args.systems, SYSTEMS, "--systems")
    missing = [system for system in REQUIRED if system not in systems]
    if missing:
        raise UsageError(f"--systems {args.systems!r} leaves out {', '.join(missing)}: B0 and FT are both "
                         "required, B4 is optional")
    categories = chosen(args.categories, CATEGORIES, "--categories")
    for pattern in args.drop_env or []:
        try:
            re.compile(pattern)
        except re.error as exc:
            raise UsageError(f"--drop-env {pattern!r} is not a regular expression: {exc}") from None
    for system in ADAPTER_SYSTEMS:
        if getattr(args, f"{system}_adapter") is not None and system not in systems:
            raise UsageError(f"--{system}-adapter names an adapter for {system}, which --systems does not "
                             "run")
    try:
        binary = llama_server(args.llama_server)
    except LookupError as exc:
        raise UsageError(str(exc)) from None
    if not binary.is_file():
        raise UsageError(f"llama-server not found: {binary} (--llama-server or {SERVER_VAR})")
    if not Path(args.model).is_file():
        raise UsageError(f"the base model {args.model} is not here (--model)")
    if not Path(args.tokenizer).is_dir():
        raise UsageError(f"the tokenizer folder {args.tokenizer} is not here (--tokenizer)")
    if not args.bfcl_python.is_file():
        raise UsageError(f"the BFCL interpreter {rel(args.bfcl_python)} is not here (--bfcl-python; the "
                         ".venv-bfcl environment of docs/m0-验证.md, S5)")
    adapters = {system: chosen_adapter(system, getattr(args, f"{system}_adapter"))
                for system in systems if system in ADAPTER_SYSTEMS}
    paths = [args.model, args.tokenizer, args.bfcl_python, args.raw, args.out, NOTHINK,
             *(adapter.gguf for adapter in adapters.values())]
    outside = [rel(path) for path in paths if not inside(path)]
    if outside:
        raise UsageError(f"run from the repository root: {outside} lie outside the current folder, and the "
                         "results hold repository-relative paths only")
    url = f"http://{HOST}:{args.port}/v1"
    extra = server_extra(args)
    plans = []
    for system in systems:
        root = args.raw / system
        earlier = sorted(path for path in root.rglob("*") if path.is_file()) if root.is_dir() else []
        if earlier:
            raise UsageError(f"{rel(root)} holds files of an earlier run ({rel(earlier[0])} and "
                             f"{len(earlier) - 1} more): BFCL would keep their answers and generate only the "
                             "missing ones; move the folder away first")
        adapter = adapters.get(system)
        gguf = None if adapter is None else rel(adapter.gguf)
        model = BFCL_MODELS[system]
        plans.append(Plan(
            system=system, model=model, adapter=adapter, root=root,
            serve=server_command(binary, model=args.model, adapter=gguf, port=args.port, ctx_size=CONTEXT,
                                 extra=extra),
            record=server_command("llama-server", model=rel(args.model), adapter=gguf, port=args.port,
                                  ctx_size=CONTEXT, extra=extra),
            generate=bfcl_command(args.bfcl_python, "generate", model, categories, rel(args.tokenizer)),
            evaluate=bfcl_command(args.bfcl_python, "evaluate", model, categories, rel(args.tokenizer)),
            settings={"BFCL_PROJECT_ROOT": os.path.abspath(root), "REMOTE_OPENAI_BASE_URL": url, **BFCL_ENV}))
    return plans, binary, categories


def describe(plans: list[Plan], categories: list[str], dropped: int) -> None:
    fixed = ", ".join(f"{name}={value}" for name, value in BFCL_ENV.items())
    print(f"BFCL single-turn regression: systems {', '.join(p.system for p in plans)}; {len(categories)} "
          f"categories: {', '.join(categories)}")
    print(f"BFCL's processes only: BFCL_PROJECT_ROOT (the system's raw folder), "
          f"REMOTE_OPENAI_BASE_URL={plans[0].settings['REMOTE_OPENAI_BASE_URL']}, {fixed}; {dropped} "
          f"inherited variables dropped (--drop-env)")
    for p in plans:
        if p.adapter is None:
            adapter = "no adapter"
        else:
            source = "named on the command line" if p.adapter.selection is None else rel(p.adapter.selection)
            adapter = f"adapter {rel(p.adapter.gguf)} sha256 {p.adapter.sha256} ({source})"
        print(f"{p.system}: BFCL model {p.model}, {adapter}, raw {rel(p.root)}")
        print(f"  server:   {subprocess.list2cmdline(p.serve)}")
        print(f"  env:      BFCL_PROJECT_ROOT={p.settings['BFCL_PROJECT_ROOT']}")
        print(f"  generate: {subprocess.list2cmdline(p.generate)}")
        print(f"  evaluate: {subprocess.list2cmdline(p.evaluate)}", flush=True)


def first_token(client: LocalClient, scale: float | None) -> tuple:
    """The probe's one-token completion at a lora scale, or without the field (None): text, tokens and
    log-probabilities. The request bypasses LocalClient.complete, which refuses one without a scale."""
    payload: dict = {"prompt": PROBE_PROMPT, "max_tokens": 1, "temperature": 0.0, "logprobs": 5,
                     "cache_prompt": False}
    if scale is not None:
        payload["lora"] = [{"id": 0, "scale": scale}]
    response = client.http.post("/v1/completions", json=payload)
    response.raise_for_status()
    choice = response.json()["choices"][0]
    return choice.get("text"), *parse_logprobs(choice.get("logprobs"))


def applies_adapter_by_default(client: LocalClient) -> bool:
    """Whether a request without a "lora" field gets the adapter at scale 1 (the probe of the module
    docstring); the request without the field comes last, right after one at scale 0."""
    on, off, default = (first_token(client, scale) for scale in (1.0, 0.0, None))
    return default == on and default != off


def cached_tokens(client: LocalClient) -> list:
    """The cache probe of the module docstring: the prompt tokens the server reports as taken from its cache
    (usage.prompt_tokens_details.cached_tokens, None when it reports none) for the probe prompt sent twice the
    way BFCL sends a request, without cache_prompt and lora fields."""
    counts = []
    for _ in range(2):
        response = client.http.post("/v1/completions",
                                    json={"prompt": PROBE_PROMPT, "max_tokens": 1, "temperature": 0.0})
        response.raise_for_status()
        usage = response.json().get("usage") or {}
        counts.append((usage.get("prompt_tokens_details") or {}).get("cached_tokens"))
    return counts


def server_problem(p: Plan, server: LocalServer) -> str | None:
    """Why the started server cannot serve p (module docstring), or None."""
    props = server.props or {}
    slots, context = props.get("total_slots"), (props.get("default_generation_settings") or {}).get("n_ctx")
    if (slots, context) != (SLOTS, CONTEXT):
        return f"the server reports {slots} slots of {context} tokens (/props), not {SLOTS} of {CONTEXT}"
    listed = [a.get("path") if isinstance(a, dict) else a for a in server.adapters]
    if p.adapter is None and listed:
        return f"the server lists the adapters {listed}, and the base model needs none"
    if p.adapter is not None:
        if len(listed) != 1 or not isinstance(listed[0], str) or not same_path(listed[0], p.adapter.gguf):
            return f"the server lists the adapters {listed}, not exactly {rel(p.adapter.gguf)}"
        if not applies_adapter_by_default(server.client):
            return ("a completion without a lora field does not match the one at lora scale 1, or matches "
                    "scale 0 too: BFCL's requests would not measure the adapter")
    cached = cached_tokens(server.client)
    if cached != [0, 0]:
        return (f"the same prompt sent twice without a cache_prompt field, as BFCL sends its requests, came "
                f"back with {cached} prompt tokens taken from the cache (usage.prompt_tokens_details."
                "cached_tokens), not [0, 0]: BFCL's answers would depend on the request before them")
    return None


def bfcl_file(root: Path, kind: str, model: str, category: str) -> Path:
    """BFCL's result or score file (kind) of a category: <kind>/<model, "/" as "_">/<live or non_live>/
    BFCL_v4_<category>_<kind>.json (bfcl_eval.utils)."""
    group = "live" if "live" in category else "non_live"
    return root / kind / model.replace("/", "_") / group / f"BFCL_v4_{category}_{kind}.json"


def json_lines(path: Path) -> list[dict]:
    """The JSON objects of a BFCL file, one per line; OutputError for a line that is not one. Lines end at a
    line feed only (a CR before it is JSON whitespace): BFCL writes text unescaped, U+2028, U+2029 and U+0085
    included, at which str.splitlines() would also cut."""
    try:
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    except ValueError as exc:  # JSONDecodeError, UnicodeDecodeError
        raise OutputError(f"{rel(path)} is not JSON lines: {exc}") from None
    if not all(isinstance(line, dict) for line in lines):
        raise OutputError(f"{rel(path)} holds a line that is not a JSON object")
    return lines


def answers(root: Path, model: str, categories: list[str]) -> dict[str, list[dict]]:
    """Every category's entries of BFCL's result files; OutputError for a category without its file."""
    found = {}
    for category in categories:
        path = bfcl_file(root, "result", model, category)
        if not path.is_file():
            raise OutputError(f"{rel(path)} is missing: BFCL generated no answer for {category}")
        found[category] = json_lines(path)
    return found


def inference_errors(found: Mapping[str, list[dict]]) -> dict[str, int]:
    """Per category, the answers that are BFCL's record of a failed request (only categories with some)."""
    counts = {category: sum(1 for entry in entries if isinstance(entry.get("result"), str)
                            and entry["result"].startswith(INFERENCE_ERROR))
              for category, entries in found.items()}
    return {category: n for category, n in counts.items() if n}


def scores(root: Path, model: str, found: Mapping[str, list[dict]]) -> dict[str, dict]:
    """{n, correct, accuracy} per category from BFCL's score files, checked against themselves and the result
    files (module docstring); OutputError otherwise."""
    scored = {}
    for category, entries in found.items():
        path = bfcl_file(root, "score", model, category)
        if not path.is_file():
            raise OutputError(f"{rel(path)} is missing: BFCL scored no answer of {category}")
        header, *failed = json_lines(path) or [{}]
        n, correct = header.get("total_count"), header.get("correct_count")
        valid = type(n) is int and type(correct) is int and 0 <= correct <= n and n > 0
        if not valid or header.get("accuracy") != correct / n or len(failed) != n - correct \
                or n != len(entries):
            raise OutputError(f"{rel(path)}: the header {header} does not agree with its {len(failed)} "
                              f"failed questions or with the {len(entries)} answers of the result file "
                              f"({category})")
        scored[category] = {"n": n, "correct": correct, "accuracy": correct / n}
    return scored


def aggregate(categories: Mapping[str, Mapping]) -> dict:
    """Our single-turn score (AGGREGATE_RULE): the unweighted mean of the category accuracies, with the number
    of categories and of questions."""
    rates = [Fraction(row["correct"], row["n"]) for row in categories.values()]
    return {"categories": len(rates), "n": sum(row["n"] for row in categories.values()),
            "accuracy": float(sum(rates) / len(rates))}


def bfcl_step(run, cmd: list[str], env: dict[str, str], log: Path) -> int:
    """One BFCL step with its output (stdout and stderr) in log; its exit code. OSError when its interpreter
    cannot be started."""
    with log.open("wb") as out:
        return run(cmd, env=env, stdout=out, stderr=subprocess.STDOUT).returncode


def cannot_run(exc: OSError) -> bool:
    """Whether a program failed to start because its file cannot be run at all: not there, not permitted, or
    not an executable of this system (ENOEXEC, to which Windows maps "not a valid Win32 application", 193;
    Windows maps the image errors of WINDOWS_NOT_A_PROGRAM to EINVAL, so they are told by their code)."""
    return (isinstance(exc, (FileNotFoundError, PermissionError)) or exc.errno == errno.ENOEXEC
            or getattr(exc, "winerror", None) in WINDOWS_NOT_A_PROGRAM)


def discard_unstarted(root: Path, log: Path) -> None:
    """Remove what a server that could not be started leaves: its empty log, and then the system's raw
    folder if nothing else is in it, so that the next run is not refused for an earlier one that ran
    nothing."""
    if log.is_file() and log.stat().st_size == 0:
        log.unlink()
    if root.is_dir() and not any(root.iterdir()):
        root.rmdir()


def show_tail(log: Path) -> None:
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines() if log.is_file() else []
    for line in lines[-LOG_TAIL:]:
        print(f"  | {line}")


def evaluate_system(p: Plan, *, args: argparse.Namespace, binary: Path, categories: list[str], popen, run,
                    client: Callable[[str], LocalClient]) -> tuple[int, dict | None]:
    """Serve, generate, stop, evaluate and check one system (module docstring): (0, its row) or (exit code,
    None)."""
    p.root.mkdir(parents=True, exist_ok=True)
    env = child_env(os.environ, args.drop_env or [], p.settings)
    url, server_log = p.settings["REMOTE_OPENAI_BASE_URL"], p.root / "server.log"
    generate_log, evaluate_log = p.root / "generate.log", p.root / "evaluate.log"
    served = "the base model" if p.adapter is None else rel(p.adapter.gguf)
    print(f"{p.system}: serving {served} (server log {rel(server_log)})", flush=True)
    try:
        with LocalServer(binary, model=args.model, adapter=None if p.adapter is None else rel(p.adapter.gguf),
                         port=args.port, ctx_size=CONTEXT, extra=server_extra(args), timeout=args.timeout,
                         log=server_log, popen=popen, client=client) as server:
            try:
                problem = server_problem(p, server)
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                problem = f"probing the server failed: {type(exc).__name__}: {exc}"
            if problem:
                print(f"FAIL: {p.system}: {problem}")
                return 3, None
            build = (server.props or {}).get("build_info")
            try:
                reach = run([p.generate[0], "-c", REACH_CODE, url], env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=REACH_TIMEOUT)
                reached, why = reach.returncode == 0, (reach.stderr or "").strip().splitlines()[-1:]
            except subprocess.TimeoutExpired:
                reached, why = False, [f"no answer within {REACH_TIMEOUT:g} s"]
            except OSError as exc:
                print(f"FAIL: {p.system}: BFCL's interpreter {rel(p.generate[0])} cannot be started for the "
                      f"reach check ({exc}; --bfcl-python)")
                return 2, None
            if not reached:
                print(f"FAIL: {p.system}: BFCL's environment cannot reach the server at {url} "
                      f"({' '.join(why) or 'no error text'}): drop the inherited variables that keep it from "
                      "the server with --drop-env PATTERN (docs/m0-验证.md, S5); BFCL's own readiness check "
                      "would wait for ever")
                return 2, None
            print(f"{p.system}: BFCL generate (log {rel(generate_log)})", flush=True)
            try:
                code = bfcl_step(run, p.generate, env, generate_log)
            except OSError as exc:
                print(f"FAIL: {p.system}: BFCL generate cannot be started ({exc}; {rel(p.generate[0])})")
                return 4, None
    except PortBusy as exc:
        print(f"FAIL: {exc}")
        return 2, None
    except NotReady as exc:
        print(f"FAIL: {p.system}: {exc} (server log {rel(server_log)})")
        return 3, None
    except OSError as exc:  # starting llama-server: the reach check and BFCL's steps report their own
        discard_unstarted(p.root, server_log)
        unusable = cannot_run(exc)
        print(f"FAIL: {p.system}: llama-server could not be started ({exc})"
              + (f": check --llama-server or {SERVER_VAR}" if unusable else ""))
        return (2 if unusable else 3), None
    if code != 0:
        print(f"FAIL: {p.system}: BFCL generate exited with {code} (log {rel(generate_log)})")
        show_tail(generate_log)
        return 4, None
    try:
        found = answers(p.root, p.model, categories)
    except OutputError as exc:
        print(f"FAIL: {p.system}: {exc} (log {rel(generate_log)})")
        return 1, None
    errors = inference_errors(found)
    if errors:
        counts = ", ".join(f"{category} {n}" for category, n in errors.items())
        print(f"FAIL: {p.system}: BFCL recorded {sum(errors.values())} inference errors ({counts}): requests "
              f"the server did not answer, which would count as wrong answers (log {rel(generate_log)})")
        return 4, None
    print(f"{p.system}: BFCL evaluate (log {rel(evaluate_log)})", flush=True)
    try:
        code = bfcl_step(run, p.evaluate, env, evaluate_log)
    except OSError as exc:
        print(f"FAIL: {p.system}: BFCL evaluate cannot be started ({exc}; {rel(p.evaluate[0])})")
        return 4, None
    if code != 0:
        print(f"FAIL: {p.system}: BFCL evaluate exited with {code} (log {rel(evaluate_log)})")
        show_tail(evaluate_log)
        return 4, None
    try:
        scored = scores(p.root, p.model, found)
    except OutputError as exc:
        print(f"FAIL: {p.system}: {exc} (log {rel(evaluate_log)})")
        return 1, None
    row = {
        "bfcl_model": p.model, "server_command": p.record, "server_build": build,
        "adapter": None if p.adapter is None else {
            "gguf": rel(p.adapter.gguf), "sha256": p.adapter.sha256,
            "selection": None if p.adapter.selection is None else rel(p.adapter.selection)},
        "raw": rel(p.root), "generate": as_recorded(p.generate), "evaluate": as_recorded(p.evaluate),
        "categories": scored, "aggregate": aggregate(scored),
    }
    print(f"{p.system}: {100 * row['aggregate']['accuracy']:.2f}% (mean over {len(scored)} categories)",
          flush=True)
    return 0, row


def summary(rows: dict[str, dict], categories: list[str]) -> None:
    width = max(len(category) for category in categories) + 2
    print("category".ljust(width) + "".join(system.rjust(18) for system in rows))
    for category in categories:
        cells = (f"{100 * row['categories'][category]['accuracy']:.2f}% ({row['categories'][category]['n']})"
                 for row in rows.values())
        print(category.ljust(width) + "".join(cell.rjust(18) for cell in cells))
    print("mean".ljust(width) + "".join(f"{100 * row['aggregate']['accuracy']:.2f}%".rjust(18)
                                        for row in rows.values()))


def main(argv: list[str] | None = None, *, popen=subprocess.Popen, run=subprocess.run,
         client: Callable[[str], LocalClient] = LocalClient) -> int:
    """popen, run and client stand in for subprocess.Popen (llama-server), subprocess.run (BFCL's processes)
    and LocalClient in tests."""
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    try:
        plans, binary, categories = prepare(args)
    except UsageError as exc:
        print(f"FAIL: {exc}")
        return 2
    drop = args.drop_env or []
    describe(plans, categories, len(os.environ) - len(child_env(os.environ, drop, {})))
    if args.dry_run:
        return 0
    python = plans[0].generate[0]
    try:
        done = run([python, "-c", VERSION_CODE], env=child_env(os.environ, drop, BFCL_ENV),
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
    except OSError as exc:
        print(f"FAIL: {rel(python)} cannot be started ({exc}): is it the BFCL environment's interpreter "
              "(--bfcl-python)?")
        return 2
    version = (done.stdout or "").strip() if done.returncode == 0 else ""
    if not version:
        why = " ".join((done.stderr or "").strip().splitlines()[-1:]) or "no error text"
        print(f"FAIL: {rel(python)} cannot tell the version of bfcl_eval ({why}): is it the BFCL environment "
              "(--bfcl-python)?")
        return 2
    rows: dict[str, dict] = {}
    for p in plans:
        code, row = evaluate_system(p, args=args, binary=binary, categories=categories, popen=popen, run=run,
                                    client=client)
        if code:
            return code
        rows[p.system] = row
    first, *others = rows.items()
    for system, row in others:
        differ = [category for category in categories
                  if row["categories"][category]["n"] != first[1]["categories"][category]["n"]]
        if differ:
            print(f"FAIL: {system} and {first[0]} differ in the number of questions of {', '.join(differ)}: "
                  "the runs did not see the same BFCL data")
            return 1
    results = {
        "benchmark": "BFCL v4, single-turn categories, Qwen FC handler in non-thinking mode "
                     "(scripts/bfcl_nothink.py)",
        "bfcl_eval": version, "categories": categories, "aggregate_rule": AGGREGATE_RULE,
        "base_model": rel(args.model), "tokenizer": rel(args.tokenizer),
        "bfcl_environment": {"BFCL_PROJECT_ROOT": "the system's raw folder, as an absolute path",
                             "REMOTE_OPENAI_BASE_URL": plans[0].settings["REMOTE_OPENAI_BASE_URL"],
                             **BFCL_ENV},
        "git_commit": git_commit(REPO), "git_dirty": git_dirty(REPO),
        "systems": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                        newline="\n")
    summary(rows, categories)
    print(f"results {rel(args.out)}; raw outputs under {rel(args.raw)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

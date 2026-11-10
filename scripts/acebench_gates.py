"""Gate G4, the settings check and H3's statistic of M8's one ACEBench run (docs/m8-计划.md WP5), against
B0's M7 run, whose raw answers it reuses.

    .venv/Scripts/python.exe scripts/acebench_gates.py --result results/acebench-m8.json \\
        --b0-raw runs/acebench/m7/b0 [--b0-result results/acebench-m7.json] [--system ft] \\
        [--decision results/decision-m8.json] [--b0-base-sha256 HEX] [--out results/gates-acebench-m8.json]
    .venv/Scripts/python.exe scripts/acebench_gates.py --result results/acebench-m7.json \\
        --b0-raw runs/acebench/m7/b0 --allow-no-decision --out <elsewhere>

The run. --result is the run's result file (scripts/run_acebench.py in think mode) and --system the system of
it that is gated (ft, the default: FT5 in M8; the M7 file holds b0 and ft). B0 is the b0 system of
--b0-result (results/acebench-m7.json by default), and --b0-raw must be its raw folder (runs/acebench/m7/b0):
M8 runs B0 no more (docs/m8-计划.md, 方法约束), so its answers are M7's, and the run must have been made with
B0's settings.

The decision. The gated system must run on the checkpoint WP4 chose: --decision (DECISION,
results/decision-m8.json, scripts/decide_m8.py's, by default) must decide GO, and the adapter the run records
(its GGUF and SHA-256, as scripts/run_acebench.py hashed it) must be the one GGUF its server command loads
(--lora) and have the chosen checkpoint's SHA-256; otherwise the run is refused (exit 2). --allow-no-decision
reads no decision, and the file records that it was given: it is for the M7 reproduction check alone (FT3's
M7 run, which no M8 decision chose), so it is refused unless --result is B0's own result file (--b0-result),
and with --decision.

The settings (settings_of), compared field by field between the run's system and B0: the llama.cpp build of
the server; the SHA-256 of the base GGUF (a result file's "base_model_sha256" when it records one; otherwise
the run's file at its "base_model" path is hashed, and B0's is --b0-base-sha256, whose default B0_BASE_SHA256
is the SHA-256 the bake-off recorded when the GGUF was made, runs/bakeoff/m7/c5/smoke.json S0:
results/acebench-m7.json names the file, models/Nanbeige4.1-3B-Q8_0.gguf, and records no SHA-256); the
ACEBench clone's commit, its patch's SHA-256 and its data check; the context size (the server's -c and the
budget's); greedy decoding with its temperature, top_p and seed; the reasoning budget (the budget's and the
server's --reasoning-budget) and the reply; ACEBench's max_tokens, temperature, top_p and threads; the
server's slots, prompt cache and reasoning format; the mode, the family, the languages, the groups and
whether it was a smoke run; and the whole server command once the adapter's part is taken out
(without_adapter: every --lora with its path, and --lora-init-without-apply, which llm.server.server_command
adds for an adapter), so that no other flag (--jinja, -ngl, --cache-ram, the port, the binary, their order)
differs. Any difference refuses the run (exit 2), each field named with both values, and for the server
command each option that differs: B0's answers could not stand for this run's B0.

G4 (docs/m7-预登记.md, as M7 computed it with the archived acebench_g4.py, which reproduced docs/m4-进度.md's
counts), over the raw answers (<raw>/<language>/result/data_<category>_result.json: one answer per line, its
"result"; lines end at a line feed, as eval.acebench reads them), pooled over both languages:
  (a) the share of the normal group's answers in ACEBench's format (run_acebench.acebench_format: the first
      outermost [...] is a Python list of calls): the run's at least B0's less 0.05;
  (b) the share of the special group's answers, read as text (str()), holding one of the three phrases
      ACEBench's special questions ask for (PHRASES, case-sensitive substrings): the run's at least 0.5 x
      B0's.
The shares are compared exactly (as fractions); the archived script compared floats, and both read M7's
counts alike.

H3 (docs/m6-计划.md §5; descriptive in M8, which is no opening, docs/m8-计划.md WP5): the run's raw folder
against B0's, read answer by answer (eval.acebench.per_item) and paired by stratum and answer id
(pair_strata), under eval.metrics.stratified_noninferiority_p with margin 0.02, seed 42 and 10,000 resamples
(MARGIN, SEED, RESAMPLES: M7's configuration, results/test-config-m7.json), recorded as scripts/run_test.py
records an ACEBench comparison.

Writes --out (results/gates-acebench-m8.json by default, a name research_agent.eval.report does not read as an
ACEBench result; JSON, two-space indent, LF), never over an existing
file: its presence refuses the run before anything is read. It holds the inputs (paths and SHA-256), the
decision (its path and SHA-256, its chosen checkpoint and the run's adapter; null with --allow-no-decision)
and whether --allow-no-decision was given, the settings compared, G4's counts per language and pooled with
both parts, H3, and both systems' aggregates as the result files record them. Prints the decision, those
aggregates, the counts, both parts of G4 and H3: aggregates only.

Exit 0 when the settings are B0's and G4 passes, 1 when G4 fails (the file is written), 2 on a usage error
(--allow-no-decision outside the M7 reproduction check, or with --decision), an existing --out, a missing or
malformed input, a --b0-raw that is not B0's raw folder, a decision that is not GO for the run's adapter, or
a setting that differs.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import re
import sys
from fractions import Fraction
from pathlib import Path

from research_agent.eval.acebench import AceItemsError, PerItem, pair_strata, per_item
from research_agent.eval.lock import sha256_of
from research_agent.eval.metrics import stratified_noninferiority_p

SCRIPTS = Path(__file__).resolve().parent


def _load_run_acebench():
    """scripts/run_acebench.py as a module of its own name (scripts/ is no package): its acebench_format reads
    an answer as ACEBench's parser does."""
    spec = importlib.util.spec_from_file_location("_run_acebench_shared", SCRIPTS / "run_acebench.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_acebench = _load_run_acebench()
acebench_format = run_acebench.acebench_format

OUT = Path("results/gates-acebench-m8.json")  # not acebench-*.json, which eval.report reads as runs
B0_RESULT = Path("results/acebench-m7.json")
DECISION = Path("results/decision-m8.json")  # scripts/decide_m8.py's (module docstring: the decision)
B0, SYSTEM = "b0", "ft"
# the adapter's part of a server command (llm.server.server_command): the flag with the GGUF after it, and the
# flag alone; the rest of the command must be B0's (module docstring: the settings)
ADAPTER_PAIR, ADAPTER_FLAG = "--lora", "--lora-init-without-apply"
COMMAND = "server command (without the adapter)"
OPTION = re.compile(r"--?[A-Za-z]")  # a token that starts an option, not a negative number
# the SHA-256 of models/Nanbeige4.1-3B-Q8_0.gguf when the bake-off made it (runs/bakeoff/m7/c5/smoke.json,
# S0), the file results/acebench-m7.json names as its base model without recording a SHA-256
B0_BASE_SHA256 = "c3396369730abb6c140bf249a89978464257e8e723841be75f7695dd5acdb0d6"
PHRASES = ("Missing necessary parameters", "There is incorrect value", "the limitations of the function")
MARGIN, SEED, RESAMPLES = 0.02, 42, 10000  # H3 as M7 computed it (results/test-config-m7.json)
PARSE_DROP = Fraction(1, 20)  # G4 (a): at most 5 points below B0's parse share
PHRASE_SHARE = Fraction(1, 2)  # G4 (b): at least half B0's phrase share
COUNT_KEYS = ("normal", "parses", "special", "phrase")
G4_RULE = ("G4 (docs/m7-预登记.md, as the archived acebench_g4.py computed it): pooled over both languages, "
           "(a) the share of the normal group's answers in ACEBench's format (their first outermost [...] a "
           "Python list of calls) at least B0's less 0.05, and (b) the share of the special group's answers "
           "holding one of ACEBench's three phrases (case-sensitive) at least 0.5 x B0's; compared exactly.")
ABOUT = ("M8's ACEBench gates (docs/m8-计划.md WP5), by scripts/acebench_gates.py: the run's adapter against "
         "WP4's decision, its settings and server command against B0's M7 run, gate G4 as M7 computed it and "
         "H3's statistic against B0's M7 raw answers, descriptive.")
USAGE, G4_FAILED = 2, 1


class InputError(ValueError):
    """An input that is missing, malformed or not the one it must be; the message names it."""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="G4, the settings check and H3 of an ACEBench run against B0's "
                                             "M7 run.")
    ap.add_argument("--result", type=Path, required=True,
                    help="the run's result file (results/acebench-m8.json)")
    ap.add_argument("--system", default=SYSTEM, help=f"the system of --result to gate (default {SYSTEM})")
    ap.add_argument("--b0-raw", type=Path, required=True, help="B0's raw folder (runs/acebench/m7/b0)")
    ap.add_argument("--b0-result", type=Path, default=B0_RESULT,
                    help="the result file of B0's run, its settings and groups (default "
                         f"{B0_RESULT.as_posix()})")
    ap.add_argument("--b0-base-sha256", default=B0_BASE_SHA256, metavar="HEX",
                    help="the SHA-256 of B0's base GGUF when --b0-result records none (default: the "
                         "bake-off's record of models/Nanbeige4.1-3B-Q8_0.gguf)")
    ap.add_argument("--decision", type=Path, default=None,
                    help=f"WP4's decision, GO for the run's adapter (default {DECISION.as_posix()})")
    ap.add_argument("--allow-no-decision", action="store_true",
                    help="read no decision, recorded in --out: for the M7 reproduction check alone (--result "
                         "is --b0-result)")
    ap.add_argument("--out", type=Path, default=OUT, help=f"the gates' file (default {OUT.as_posix()})")
    return ap


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(f"{path.as_posix()} cannot be read: {exc.strerror or exc}") from None
    except ValueError as exc:
        raise InputError(f"{path.as_posix()} is not JSON: {exc}") from None
    if not isinstance(value, dict):
        raise InputError(f"{path.as_posix()} is not a JSON object")
    return value


def system_of(result: dict, name: str, path: Path) -> dict:
    """The record of a system in a result file, with its raw folder; InputError when there is none."""
    systems = result.get("systems")
    record = systems.get(name) if isinstance(systems, dict) else None
    if not isinstance(record, dict) or not isinstance(record.get("raw"), str):
        held = sorted(systems) if isinstance(systems, dict) else []
        raise InputError(f"{path.as_posix()} holds no system {name!r} with a raw folder (it holds {held})")
    return record


def _get(data: object, *keys: str) -> object:
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def command_value(command: object, flag: str) -> object:
    """The value after flag in a server command; None when the flag is not there."""
    if not isinstance(command, list) or flag not in command[:-1]:
        return None
    return command[command.index(flag) + 1]


def without_adapter(command: object) -> object:
    """A server command without the adapter's part (module docstring: the settings): every ADAPTER_PAIR with
    the token after it, and ADAPTER_FLAG; the value as it is when it is no list."""
    if not isinstance(command, list):
        return command
    out, tokens = [], iter(command)
    for token in tokens:
        if token == ADAPTER_PAIR:
            next(tokens, None)  # its GGUF
        elif token != ADAPTER_FLAG:
            out.append(token)
    return out


def options(command: list) -> dict[object, list[tuple]]:
    """A server command's options after the binary, by flag: the values after each occurrence (the tokens up
    to the next option, OPTION); values before any option go under None."""
    out: dict[object, list[list]] = {}
    current = None
    for token in command[1:]:
        if isinstance(token, str) and OPTION.match(token):
            current = token
            out.setdefault(token, []).append([])
        else:
            out.setdefault(current, [[]])[-1].append(token)
    return {flag: [tuple(values) for values in occurrences] for flag, occurrences in out.items()}


def _values(occurrences: list[tuple]) -> str:
    return " | ".join(" ".join(map(str, values)) for values in occurrences)


def command_differences(b0: object, run: object, system: str) -> list[str]:
    """How the run's server command without the adapter differs from B0's, a line per option that differs
    (and the binary); [] when the two are equal."""
    if b0 == run:
        return []
    if not (isinstance(b0, list) and isinstance(run, list) and b0 and run):
        return [f"{COMMAND}: {b0!r} (B0) vs {run!r} ({system})"]
    lines = [] if b0[0] == run[0] else [f"{COMMAND}: the binary {b0[0]!r} (B0) vs {run[0]!r} ({system})"]
    ours, theirs = options(b0), options(run)
    for flag in [*ours, *(flag for flag in theirs if flag not in ours)]:
        if flag not in theirs:
            tail = _values(ours[flag]).strip()
            lines.append(f"{COMMAND}: {flag}{' ' + tail if tail else ''} in B0's, not in {system}'s")
        elif flag not in ours:
            tail = _values(theirs[flag]).strip()
            lines.append(f"{COMMAND}: {flag}{' ' + tail if tail else ''} in {system}'s, not in B0's")
        elif ours[flag] != theirs[flag]:
            lines.append(f"{COMMAND}: {flag} {_values(ours[flag])!r} (B0) vs {_values(theirs[flag])!r} "
                         f"({system})")
    if not lines:
        lines.append(f"{COMMAND}: the same options in another order: {b0!r} (B0) vs {run!r} ({system})")
    return lines


def file_sha256(path: str | Path) -> str:
    try:
        return sha256_of(path)
    except OSError as exc:
        raise InputError(f"the base GGUF {Path(path).as_posix()} cannot be hashed: "
                         f"{exc.strerror or exc}") from None


def base_sha256(result: dict, *, given: str | None) -> tuple[str, str]:
    """The SHA-256 of the base GGUF a result file was run on and where it comes from (module docstring: the
    settings): the file's own record, else given (B0's --b0-base-sha256), else the file at its base_model
    path, hashed now."""
    recorded = result.get("base_model_sha256")
    if isinstance(recorded, str) and recorded:
        return recorded, "recorded by the result file"
    if given is not None:
        return given, "--b0-base-sha256"
    path = result.get("base_model")
    if not isinstance(path, str) or not path:
        raise InputError("the result file names no base_model and records no base_model_sha256")
    return file_sha256(path), f"{path} hashed"


def settings_of(result: dict, system: str, base: str) -> dict:
    """The settings of a system's run that must be B0's, by name (module docstring: the settings)."""
    record = result["systems"][system]
    command = record.get("server_command")
    return {"llama.cpp build": record.get("server_build"),
            "base GGUF SHA-256": base,
            "ACEBench clone commit": _get(result, "acebench", "commit"),
            "ACEBench patch SHA-256": _get(result, "acebench", "patch", "sha256"),
            "ACEBench data": _get(result, "acebench", "data", "identical_to"),
            "context size (-c)": command_value(command, "-c"),
            "context size (budget)": _get(result, "budget", "context"),
            "greedy": _get(result, "decoding", "greedy"),
            "decoding temperature": _get(result, "decoding", "temperature"),
            "decoding top_p": _get(result, "decoding", "top_p"),
            "decoding seed": _get(result, "decoding", "seed"),
            "reasoning budget (budget)": _get(result, "budget", "reasoning"),
            "reasoning budget (server)": command_value(command, "--reasoning-budget"),
            "reply tokens": _get(result, "budget", "reply"),
            "ACEBench max_tokens": _get(result, "generation", "max_tokens"),
            "ACEBench temperature": _get(result, "generation", "temperature"),
            "ACEBench top_p": _get(result, "generation", "top_p"),
            "ACEBench threads": _get(result, "generation", "num_threads"),
            "slots (-np)": command_value(command, "-np"),
            "prompt cache off": isinstance(command, list) and "--no-cache-prompt" in command,
            "reasoning format": command_value(command, "--reasoning-format"),
            "mode": result.get("mode"),
            "family": _get(result, "family", "name"),
            "languages": result.get("languages"),
            "groups": result.get("groups"),
            "smoke": result.get("smoke"),
            COMMAND: without_adapter(command)}


def settings_differences(run: dict, b0: dict, system: str) -> list[str]:
    """A line per setting that differs, the server command's per option (command_differences)."""
    lines = []
    for name, value in run.items():
        if name == COMMAND:
            lines += command_differences(b0[name], value, system)
        elif value != b0[name]:
            lines.append(f"{name}: {b0[name]!r} (B0) vs {value!r} ({system})")
    return lines


def same_file(path: Path, other: Path) -> bool:
    """Whether two paths name one file (resolved, case folded where the file system folds it)."""
    if os.path.normcase(os.path.realpath(path)) == os.path.normcase(os.path.realpath(other)):
        return True
    try:
        return os.path.samefile(path, other)
    except OSError:  # one of the two does not exist
        return False


def decision_record(path: Path, record: dict, system: str) -> dict:
    """The decision as --out records it, when it is GO for the adapter the run's system ran on (module
    docstring: the decision); InputError naming what differs otherwise."""
    decision = read_json(path)
    verdict, chosen = decision.get("decision"), decision.get("chosen")
    if verdict != "GO":
        raise InputError(f"{posix(path)} decides {verdict!r}, not GO: WP4 chose no checkpoint, and M8 gates "
                         "no ACEBench run then (docs/m8-计划.md WP4)")
    if not isinstance(chosen, dict) or not isinstance(chosen.get("sha256"), str):
        raise InputError(f"{posix(path)} decides GO without the chosen checkpoint's SHA-256")
    adapter = record.get("adapter")
    if not isinstance(adapter, dict) or not isinstance(adapter.get("sha256"), str):
        raise InputError(f"{system} ran with no adapter, where the checkpoint WP4 chose "
                         f"({chosen.get('step')}, sha256 {chosen['sha256']})")
    command = record.get("server_command")
    loaded = ([value for flag, value in itertools.pairwise(command) if flag == ADAPTER_PAIR]
              if isinstance(command, list) else None)
    if loaded != [adapter.get("gguf")]:
        raise InputError(f"adapter GGUF: {system}'s server command loads {loaded!r}, where the adapter the "
                         f"run records ({adapter.get('gguf')!r}) alone")
    if adapter["sha256"] != chosen["sha256"]:
        raise InputError(f"adapter SHA-256: {chosen['sha256']} (WP4's checkpoint {chosen.get('step')}, "
                         f"{posix(path)}) vs {adapter['sha256']} ({system}'s {adapter.get('gguf')})")
    return {"path": posix(path), "sha256": sha256_of(path), "decision": verdict, "chosen": chosen,
            "adapter": {"gguf": adapter.get("gguf"), "sha256": adapter["sha256"]}}


def answers(raw: str | Path, language: str, category: str) -> list:
    """The results of a category's answers, in order (module docstring: G4); InputError for a file that is
    missing or whose lines are not answers."""
    path = Path(raw) / language / "result" / f"data_{category}_result.json"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"{path.as_posix()} cannot be read: {exc.strerror or exc}") from None
    results = []
    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise InputError(f"{path.as_posix()}: line {number} is not JSON: {exc}") from None
        if not isinstance(row, dict) or "result" not in row:
            raise InputError(f"{path.as_posix()}: line {number} is no answer (an object with a result)")
        results.append(row["result"])
    return results


def g4_counts(raw: str | Path, groups: dict, languages: list[str]) -> dict:
    """Per language and pooled: the normal group's answers and those in ACEBench's format, the special group's
    answers and those holding a phrase (module docstring: G4)."""
    if not all(isinstance(groups.get(group), list) for group in ("normal", "special")):
        raise InputError("the result file's groups hold no list of normal and of special categories")
    out = {}
    for language in languages:
        normal = [a for category in groups["normal"] for a in answers(raw, language, category)]
        special = [str(a) for category in groups["special"] for a in answers(raw, language, category)]
        out[language] = {"normal": len(normal), "parses": sum(1 for a in normal if acebench_format(a)),
                         "special": len(special),
                         "phrase": sum(1 for a in special if any(p in a for p in PHRASES))}
    out["pooled"] = {key: sum(out[language][key] for language in languages) for key in COUNT_KEYS}
    return out


def g4(run: dict, b0: dict, system: str) -> dict:
    """Both parts of G4 over the pooled counts of the run and of B0, compared exactly."""
    pooled = {system: run["pooled"], B0: b0["pooled"]}
    for name, counts in pooled.items():
        if not counts["normal"] or not counts["special"]:
            raise InputError(f"{name}'s answers hold no normal or no special answer: G4 cannot be read")
    parse = {name: Fraction(c["parses"], c["normal"]) for name, c in pooled.items()}
    phrase = {name: Fraction(c["phrase"], c["special"]) for name, c in pooled.items()}
    a_bound, b_bound = parse[B0] - PARSE_DROP, PHRASE_SHARE * phrase[B0]
    a = {system: float(parse[system]), B0: float(parse[B0]), "threshold": float(a_bound),
         "passed": parse[system] >= a_bound}
    b = {system: float(phrase[system]), B0: float(phrase[B0]), "threshold": float(b_bound),
         "passed": phrase[system] >= b_bound}
    return {"rule": G4_RULE, "counts": {system: run, B0: b0}, "a": a, "b": b,
            "passed": a["passed"] and b["passed"]}


def h3_statistic(raw: str | Path, b0_raw: str | Path) -> dict:
    """H3's statistic of the run's raw folder against B0's, recorded as scripts/run_test.py records an
    ACEBench comparison (module docstring: H3)."""
    a, b = per_item(raw), per_item(b0_raw)
    strata_a, strata_b = pair_strata(a, b)
    stats = stratified_noninferiority_p(strata_a, strata_b, MARGIN, seed=SEED, n=RESAMPLES)

    def side(run: PerItem, strata: dict) -> dict:
        return {"path": run.path, "sha256": run.sha256, "files": len(run.files), "items": run.items,
                "mean": sum(sum(s) / len(s) for s in strata.values()) / len(strata)}

    per_stratum = [{"language": language, "category": category, "n": len(strata_a[key]),
                    "correct_a": sum(strata_a[key]), "correct_b": sum(strata_b[key]),
                    "later_candidates_a": a.headers[key]["later_candidates"],
                    "later_candidates_b": b.headers[key]["later_candidates"]}
                   for key in strata_a for language, category in [key]]
    return {"a": side(a, strata_a), "b": side(b, strata_b), "margin": stats["margin"],
            "strata": stats["strata"], "items": stats["items"], "diff": stats["diff"],
            "paired_ci": {key: stats[key] for key in ("diff", "mean", "low", "high", "n", "seed")},
            "p_ni": stats["p_ni"], "below": stats["below"], "per_stratum": per_stratum}


def posix(path: str | Path) -> str:
    return Path(path).as_posix()


def write_json(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def counts_lines(name: str, record: dict, counts: dict, languages: list[str]) -> list[str]:
    aggregate = record.get("aggregate") or {}
    accuracy = aggregate.get("accuracy")
    lines = [f"{name}: aggregate {accuracy:.4f} over {aggregate.get('categories')} categories, "
             f"{aggregate.get('n')} answers" if isinstance(accuracy, float)
             else f"{name}: no aggregate recorded"]
    for language in languages:
        c = counts[language]
        lines.append(f"  {language}: normal parses {c['parses']}/{c['normal']}, special phrase "
                     f"{c['phrase']}/{c['special']}")
    p = counts["pooled"]
    lines.append(f"  pooled: normal parses {p['parses']}/{p['normal']} = {p['parses'] / p['normal']:.4f}, "
                 f"special phrase {p['phrase']}/{p['special']} = {p['phrase'] / p['special']:.4f}")
    return lines


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    if args.out.exists():
        print(f"FAIL: {posix(args.out)} exists: an ACEBench run is gated once, and nothing was read")
        return USAGE
    if args.system == B0:
        print(f"FAIL: --system {B0} is B0 itself: the gates compare another system with it")
        return USAGE
    if args.allow_no_decision and args.decision is not None:
        print("FAIL: --allow-no-decision reads no decision, and --decision names one: give one of the two")
        return USAGE
    if args.allow_no_decision and not same_file(args.result, args.b0_result):
        print(f"FAIL: --allow-no-decision is for the M7 reproduction check alone, which gates a system of "
              f"B0's own result file: --result {posix(args.result)} is not --b0-result "
              f"{posix(args.b0_result)}")
        return USAGE
    decision_path = args.decision or DECISION
    try:
        result, b0_result = read_json(args.result), read_json(args.b0_result)
        record = system_of(result, args.system, args.result)
        b0_record = system_of(b0_result, B0, args.b0_result)
        if posix(b0_record["raw"]) != posix(args.b0_raw):
            raise InputError(f"--b0-raw {posix(args.b0_raw)} is not the raw folder of B0's run in "
                             f"{posix(args.b0_result)} ({b0_record['raw']})")
        decision = None if args.allow_no_decision else decision_record(decision_path, record, args.system)
        base, base_source = base_sha256(result, given=None)
        b0_base, b0_source = base_sha256(b0_result, given=args.b0_base_sha256)
        run_settings = settings_of(result, args.system, base)
        b0_settings = settings_of(b0_result, B0, b0_base)
    except InputError as exc:
        print(f"FAIL: {exc}")
        return USAGE
    differences = settings_differences(run_settings, b0_settings, args.system)
    if differences:
        print(f"FAIL: {args.system} of {posix(args.result)} was not run with the settings of B0's run in "
              f"{posix(args.b0_result)}, so B0's answers cannot stand for its B0:")
        for line in differences:
            print(f"  {line}")
        return USAGE
    try:
        if not isinstance(result.get("groups"), dict) or not isinstance(result.get("languages"), list):
            raise InputError(f"{posix(args.result)} records no groups or languages")
        counts = g4_counts(record["raw"], result["groups"], result["languages"])
        b0_counts = g4_counts(args.b0_raw, b0_result["groups"], b0_result["languages"])
        gate = g4(counts, b0_counts, args.system)
        h3 = h3_statistic(record["raw"], args.b0_raw)
    except (InputError, AceItemsError) as exc:
        print(f"FAIL: {exc}")
        return USAGE
    except ValueError as exc:  # pair_strata: the two runs hold other strata or answers
        print(f"FAIL: H3 cannot pair the runs: {exc}")
        return USAGE
    out = {"about": ABOUT, "result": {"path": posix(args.result), "sha256": sha256_of(args.result)},
           "system": args.system,
           "b0": {"result": {"path": posix(args.b0_result), "sha256": sha256_of(args.b0_result)},
                  "raw": posix(args.b0_raw)},
           "decision": decision, "allow_no_decision": args.allow_no_decision,
           "settings": {"equal": True, "fields": run_settings,
                        "base_sha256_from": {args.system: base_source, B0: b0_source}},
           "g4": gate, "h3": h3,
           "aggregates": {args.system: record.get("aggregate"), B0: b0_record.get("aggregate")}}
    write_json(out, args.out)
    print(f"ACEBench gates: {args.system} of {posix(args.result)} against B0 ({posix(args.b0_result)}, "
          f"{posix(args.b0_raw)})")
    if decision is None:
        print("decision: none read (--allow-no-decision, the M7 reproduction check)")
    else:
        print(f"decision: GO for checkpoint {decision['chosen'].get('step')} ({posix(decision_path)}); the "
              f"run's adapter {decision['adapter']['gguf']} has its SHA-256 {decision['adapter']['sha256']}")
    print(f"settings: B0's ({len(run_settings)} fields; llama.cpp {run_settings['llama.cpp build']}, base "
          f"GGUF {base[:12]}..., clone {str(run_settings['ACEBench clone commit'])[:7]}, -c "
          f"{run_settings['context size (-c)']}, greedy {run_settings['greedy']}, R "
          f"{run_settings['reasoning budget (budget)']})")
    for name, rec, c, languages in ((args.system, record, counts, result["languages"]),
                                    (B0, b0_record, b0_counts, b0_result["languages"])):
        print("\n".join(counts_lines(name, rec, c, languages)))
    a, b = gate["a"], gate["b"]
    print(f"G4 (a): {args.system} {a[args.system]:.4f} >= {B0} {a[B0]:.4f} - 0.05 = {a['threshold']:.4f}: "
          f"{'pass' if a['passed'] else 'FAIL'}")
    print(f"G4 (b): {args.system} {b[args.system]:.4f} >= 0.5 x {B0} {b[B0]:.4f} = {b['threshold']:.4f}: "
          f"{'pass' if b['passed'] else 'FAIL'}")
    ci = h3["paired_ci"]
    print(f"H3: {args.system} - {B0} = {h3['diff'] * 100:+.2f} points over {h3['strata']} strata "
          f"({h3['items']} answers), paired 95% interval [{ci['low'] * 100:+.2f}, {ci['high'] * 100:+.2f}], "
          f"margin {h3['margin'] * 100:g} points: p_NI {h3['p_ni']:.4f} ({RESAMPLES} resamples, seed {SEED}; "
          "descriptive)")
    print(f"wrote {posix(args.out)}")
    return 0 if gate["passed"] else G4_FAILED


if __name__ == "__main__":
    sys.exit(main())

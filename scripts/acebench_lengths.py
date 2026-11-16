"""The generated lengths of M7's ACEBench run (docs/m11-计划.md, 1. 报告补项: ACEBench 生成长度): per
system and language, the tokens llama-server generated for each of ACEBench's requests, read from the server
logs scripts/run_acebench.py kept. Aggregates only: counts, percentiles, maxima and the SHA-256 of the files
read.

    .venv/Scripts/python.exe scripts/acebench_lengths.py [--force]

What is measured. ACEBench's raw answers keep no reasoning (scripts/run_acebench.py: generate.py keeps the
answer it read from message.content, the server having parsed the reasoning out into reasoning_content), so
the one length left of a request is G, the tokens the server generated for it: the reasoning, the end tag
that closes it (the server adds the tag itself when the reasoning budget closes the reasoning) and the
answer, together, the token that ended the generation included. The log does not split G into reasoning
and answer. For every request it serves, task N of its log, llama-server (b11349-fb4b2737a on both servers,
as the source records; verbosity 3) writes three lines:

    ... | task N | prompt eval time = ... ms /   P tokens (...)
    ... | task N |        eval time = ... ms /   G tokens (...)
    ... | task N | stop processing: n_tokens = T, truncated = X

P the prompt tokens it evaluated, G the tokens it sampled, T those its slot held when the request ended:
the prompt and every sampled token but the last, which ended the request and was never evaluated, so
T = P + G - 1 when the whole prompt was evaluated (the run's servers kept no prompt cache: one slot,
--no-cache-prompt); X is 0 unless the prompt was cut to fit the context. Its other lines (the slot chosen,
the task started, progress, the total time, graphs reused, warnings) are not read.

The requests of a log, in the order the server took them, one at a time on its one slot. Before ACEBench
starts, scripts/run_acebench.py probes every server (server_problem): on a server with an adapter the
adapter probe, three one-token completions (scripts/run_bfcl.py, applies_adapter_by_default: lora scale 1,
scale 0, no lora field); then the cache probe, two one-token completions (cached_tokens), all five of
run_bfcl's PROBE_PROMPT; then in think mode the thinking probe, one chat completion with the run's
max_tokens (think_probe). These requests (probes: 3 on b0's server, 6 on ft's) are left out. ACEBench's
requests follow, one per answer (generate.py, --num-threads 1), the languages one after the other in the
order of the source's "languages", as many in a language as its categories have answers (the sum of their
n).

The source, results/acebench-m7.json (scripts/run_acebench.py in think mode): the languages; the reasoning
budget R and max_tokens, budget.reasoning and budget.max_tokens (Setup.budget_record), which every system's
server command must carry as --reasoning-budget R and every generate command as --max-tokens max_tokens,
and generation.max_tokens must equal; per system its raw folder, whose server.log is read, whether its
server loaded an adapter (and so had the adapter probe), and its answers per language.

Fail closed. The run exits 1 with a FAIL line naming the file (and the task) and writes nothing when the
source or a log is missing or unreadable; the source is not JSON or not a think-mode result as above (a
budget a recorded command does not carry, a raw folder that is not a repository-relative path, a category
without its answers); a task of a log lacks one of its three lines or has a line twice (the line that
starts it included); n_tokens is not P + G - 1; truncated is not 0; G is above max_tokens; a log holds
other than the probes and one request per answer; or the probes are not what the code sends: a one-token
probe that generated another number of tokens, one-token probes of different prompt lengths (all five send
the same prompt), a thinking probe of one token (a probe that passed returned a reasoning and an answer).

Writes results/lengths-acebench-m7.json, JSON with two-space indent and LF, repository-relative paths and
no time, never over an existing file unless --force is given: kind "acebench-lengths"; the definition
(DEFINITION, one paragraph); the source and its SHA-256; the budget {reasoning, max_tokens, from}; the
languages; and per system (in the source's order) the log and its SHA-256, the probes left out, the
requests of the log (probes included) and per language and over both ("all") a block: n, the requests;
tokens_p50, tokens_p95 (linear interpolation between closest ranks: research_agent.eval.resources'
percentile, as scripts/thinking_lengths.py takes them) and tokens_max of G; over_budget, the requests with
G above R, an upper bound on the reasonings the budget closed (one closed at R tokens has the end tag after
them, so its G is above R; a shorter reasoning with a long answer may be too); at_max_tokens, those with G
equal to max_tokens, the limit at which the server stops a generation. Prints the same blocks. Run it from
the repository root: the paths are relative to it.

Exit 0 when the file is written; 1 when an input fails a check above (nothing is written); 2 on a usage
error: an argument other than --force, or the file exists and --force was not given.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from research_agent.eval.resources import percentile

SOURCE = Path("results/acebench-m7.json")  # scripts/run_acebench.py's THINK_OUT
OUT = Path("results/lengths-acebench-m7.json")
LOG = "server.log"  # in each system's raw folder (scripts/run_acebench.py, evaluate_system)
KIND = "acebench-lengths"
WRITTEN, REFUSED, USAGE = 0, 1, 2
# The probes scripts/run_acebench.py sends a server before ACEBench's requests (server_problem), in this
# order: the adapter probe on a server with an adapter (scripts/run_bfcl.py, applies_adapter_by_default:
# lora scale 1, scale 0, no lora field), the cache probe (cached_tokens), each a one-token completion of
# run_bfcl's PROBE_PROMPT; then in think mode the thinking probe (think_probe), a chat completion.
ADAPTER_PROBE, CACHE_PROBE, THINK_PROBE = 3, 2, 1
ONE_TOKEN = 1  # the max_tokens of a one-token probe
ALL = "all"  # the block over every language
ROW_KEYS = ("log", "log_sha256", "probes", "requests", ALL)  # no language may be named so
DEFINITION = (
    "G is the number of tokens llama-server generated for one of ACEBench's requests, as the 'eval time' "
    "line of its task in the system's server log counts them: every token the server sampled for the "
    "request, which is the reasoning, the end tag that closes it (the server adds the tag itself when the "
    "reasoning budget closes the reasoning) and the answer, together, the token that ended the generation "
    "included. ACEBench's raw answers keep no reasoning, so G is the only length of a request left, and the "
    "log does not split it into reasoning and answer. Each request is a task of the log with a 'prompt eval "
    "time' line (P tokens), an 'eval time' line (G) and a 'stop processing: n_tokens = T, truncated = 0' "
    "line, T = P + G - 1. The probes scripts/run_acebench.py sends a server before ACEBench starts are left "
    "out: on a server with an adapter the adapter probe's three one-token completions, on every server the "
    "cache probe's two and the thinking probe's chat completion. ACEBench's requests follow, one per answer, "
    "the languages one after the other in the order of 'languages', as many in a language as the source "
    "counts answers in its categories. tokens_p50 and tokens_p95 interpolate linearly between closest ranks. "
    "over_budget counts the requests whose G is above the reasoning budget, an upper bound on the reasonings "
    "the budget closed: a reasoning closed at the budget has the end tag after its budget's tokens, so its G "
    "is above the budget, and a shorter reasoning followed by a long answer is counted too. at_max_tokens "
    "counts the requests whose G is max_tokens, the limit at which the server stops a generation.")
BUDGET_FROM = (f"{SOURCE.as_posix()} budget.reasoning and budget.max_tokens (scripts/run_acebench.py, "
               "Setup.budget_record), which every system's server command carries as --reasoning-budget and "
               "every generate command as --max-tokens")

_TASK = r"\| task (?P<task>\d+) \| "
# a request's lines (module docstring), by the name a message gives them; the one that starts it may be absent
STARTED = "processing task"
LINES = {
    STARTED: re.compile(_TASK + r"processing task\b"),
    "prompt eval time": re.compile(_TASK + r"prompt eval time = *[0-9.]+ ms / *(?P<value>\d+) tokens"),
    "eval time": re.compile(_TASK + r" *eval time = *[0-9.]+ ms / *(?P<value>\d+) tokens"),
    "stop processing": re.compile(_TASK + r"stop processing: n_tokens = (?P<value>\d+), "
                                          r"truncated = (?P<truncated>-?\d+)"),
}
REQUIRED = ("prompt eval time", "eval time", "stop processing")


class Refused(Exception):
    """An input that fails a check of the module docstring: exit 1, nothing written. The message names the
    file and, for a log, the task."""


@dataclass(frozen=True)
class Task:
    """One request of a server log: its task number, P, G, T and its truncated flag."""

    number: int
    prompt: int
    generated: int
    tokens: int
    truncated: int


@dataclass(frozen=True)
class System:
    """A system of the source: its name, its log (repository-relative), the probes its server had and its
    answers per language, in the source's order of the languages."""

    name: str
    log: PurePosixPath
    probes: int
    answers: dict[str, int]


@dataclass(frozen=True)
class Source:
    """What the script takes from the source (module docstring), with the SHA-256 of its bytes."""

    sha256: str
    languages: list[str]
    reasoning: int
    max_tokens: int
    systems: list[System]


def probes(adapter: bool) -> int:
    """The requests scripts/run_acebench.py sends a think-mode server before ACEBench's: the adapter probe on
    a server with an adapter, the cache probe and the thinking probe."""
    return (ADAPTER_PROBE if adapter else 0) + CACHE_PROBE + THINK_PROBE


def _listed(names: list[str]) -> str:
    quoted = [f"'{name}'" for name in names]
    return quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def read_tasks(text: str, log: str) -> list[Task]:
    """The tasks of a server log in the order they first appear, the order the server took them on its one
    slot, each with its three lines; Refused for a task that lacks one of them or has a line twice (the line
    that starts it included), or whose lines do not add up: n_tokens other than P + G - 1, truncated not 0."""
    seen: dict[int, dict[str, re.Match]] = {}
    for line in text.splitlines():
        if "| task " not in line:
            continue
        for name, pattern in LINES.items():
            match = pattern.search(line)
            if match is None:
                continue
            number = int(match["task"])
            lines = seen.setdefault(number, {})
            if name in lines:
                raise Refused(f"{log}: task {number} has two '{name}' lines")
            lines[name] = match
            break
    tasks = []
    for number, lines in seen.items():
        missing = [name for name in REQUIRED if name not in lines]
        if missing:
            plural = "s" if len(missing) > 1 else ""
            raise Refused(f"{log}: task {number} has no {_listed(missing)} line{plural}")
        stop = lines["stop processing"]
        task = Task(number=number, prompt=int(lines["prompt eval time"]["value"]),
                    generated=int(lines["eval time"]["value"]), tokens=int(stop["value"]),
                    truncated=int(stop["truncated"]))
        if task.tokens != task.prompt + task.generated - 1:
            raise Refused(f"{log}: task {number} ended with n_tokens = {task.tokens}, not P + G - 1 = "
                          f"{task.prompt} + {task.generated} - 1 = {task.prompt + task.generated - 1}: its "
                          "prompt was not evaluated whole, or its counts do not belong together")
        if task.truncated != 0:
            raise Refused(f"{log}: task {number} ended with truncated = {task.truncated}: its prompt was cut "
                          "to fit the context")
        tasks.append(task)
    return tasks


def check_probes(probe_tasks: list[Task], log: str) -> None:
    """The probes are what scripts/run_acebench.py sends (module docstring): every one but the last a
    one-token completion, all of one prompt, and the last the thinking probe, which a server that passed it
    answered with a reasoning and an answer; Refused otherwise."""
    *one_token, think = probe_tasks
    for index, task in enumerate(one_token, 1):
        if task.generated != ONE_TOKEN:
            raise Refused(f"{log}: probe {index} (task {task.number}), a one-token completion, generated "
                          f"{task.generated} tokens: the first {len(probe_tasks)} requests are not the "
                          "probes scripts/run_acebench.py sends")
    prompts = sorted({task.prompt for task in one_token})
    if len(prompts) > 1:
        raise Refused(f"{log}: the one-token probes' prompts are of {', '.join(map(str, prompts))} tokens, "
                      "where all of them send run_bfcl's PROBE_PROMPT")
    if think.generated <= ONE_TOKEN:
        raise Refused(f"{log}: the thinking probe (task {think.number}) generated {think.generated} token, "
                      "where a probe that passed returned a reasoning and an answer")


def _positive(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def option(command: object, flag: str) -> object:
    """The value after the last flag of a recorded command; None when it has none."""
    if not isinstance(command, list):
        return None
    at = [i for i, part in enumerate(command[:-1]) if part == flag]
    return command[at[-1] + 1] if at else None


def _relative(path: object) -> bool:
    """Whether path is a repository-relative POSIX path, as scripts/run_acebench.py records a raw folder."""
    if not isinstance(path, str) or not path or "\\" in path or ":" in path:
        return False
    posix = PurePosixPath(path)
    return not posix.is_absolute() and ".." not in posix.parts


def _languages(value: object) -> bool:
    return (isinstance(value, list) and bool(value) and all(isinstance(item, str) for item in value)
            and len(set(value)) == len(value) and not set(value) & set(ROW_KEYS))


def _answers(categories: object) -> bool:
    return (isinstance(categories, dict) and bool(categories)
            and all(isinstance(cell, dict) and _positive(cell.get("n")) for cell in categories.values()))


def read_source(root: Path) -> Source:
    """The source under root (module docstring); Refused when it is missing, not JSON or not of its kind."""
    name = SOURCE.as_posix()
    try:
        data = (root / SOURCE).read_bytes()
    except OSError:
        raise Refused(f"{name} is missing or unreadable: run it from the repository root") from None
    try:
        results = json.loads(data.decode("utf-8"))
    except ValueError:  # not UTF-8 or not JSON
        raise Refused(f"{name} is not JSON") from None
    if not isinstance(results, dict) or results.get("mode") != "think":
        mode = results.get("mode") if isinstance(results, dict) else None
        raise Refused(f"{name} is not a think-mode run of scripts/run_acebench.py (mode {mode!r}): its "
                      "requests had no reasoning budget")
    languages = results.get("languages")
    if not _languages(languages):
        raise Refused(f"{name}: languages {languages!r} are not a list of distinct languages")
    budget = results.get("budget")
    reasoning, max_tokens = ((budget.get("reasoning"), budget.get("max_tokens")) if isinstance(budget, dict)
                             else (None, None))
    if not (_positive(reasoning) and _positive(max_tokens) and reasoning < max_tokens):
        raise Refused(f"{name}: budget {budget!r} has no reasoning budget below its max_tokens")
    generation = results.get("generation")
    generated = generation.get("max_tokens") if isinstance(generation, dict) else None
    if generated != max_tokens:
        raise Refused(f"{name}: generation.max_tokens {generated}, not budget.max_tokens {max_tokens}")
    rows = results.get("systems")
    if not isinstance(rows, dict) or not rows:
        raise Refused(f"{name} holds no systems")
    systems = []
    for system, row in rows.items():
        at = f"{name}: system {system}"
        if not isinstance(row, dict):
            raise Refused(f"{at} is not a system's record")
        raw, adapter = row.get("raw"), row.get("adapter")
        if not _relative(raw):
            raise Refused(f"{at}: its raw folder {raw!r} is not a repository-relative path")
        if adapter is not None and not isinstance(adapter, dict):
            raise Refused(f"{at}: its adapter {adapter!r} is neither null nor a record")
        given = option(row.get("server_command"), "--reasoning-budget")
        if given != str(reasoning):
            raise Refused(f"{at}: its server command carries --reasoning-budget {given}, not "
                          f"budget.reasoning {reasoning}")
        blocks = row.get("languages")
        if not isinstance(blocks, dict) or set(blocks) != set(languages):
            raise Refused(f"{at}: its languages are not {', '.join(languages)}")
        answers = {}
        for language in languages:
            block = blocks[language] if isinstance(blocks[language], dict) else {}
            given = option(block.get("generate"), "--max-tokens")
            if given != str(max_tokens):
                raise Refused(f"{at}, {language}: its generate command carries --max-tokens {given}, not "
                              f"budget.max_tokens {max_tokens}")
            categories = block.get("categories")
            if not _answers(categories):
                raise Refused(f"{at}, {language}: its categories do not each count their answers (n above 0)")
            answers[language] = sum(cell["n"] for cell in categories.values())
        systems.append(System(name=system, log=PurePosixPath(raw) / LOG, probes=probes(adapter is not None),
                              answers=answers))
    return Source(sha256=hashlib.sha256(data).hexdigest(), languages=list(languages), reasoning=reasoning,
                  max_tokens=max_tokens, systems=systems)


def block(generated: list[int], source: Source) -> dict:
    """The lengths of some requests (module docstring): n, the percentiles and maximum of G, over_budget and
    at_max_tokens."""
    return {"n": len(generated), "tokens_p50": percentile(generated, 50),
            "tokens_p95": percentile(generated, 95), "tokens_max": max(generated),
            "over_budget": sum(g > source.reasoning for g in generated),
            "at_max_tokens": sum(g == source.max_tokens for g in generated)}


def system_lengths(root: Path, system: System, source: Source) -> dict:
    """A system's row (module docstring) from its log under root; Refused when the log fails a check."""
    log = system.log.as_posix()
    try:
        data = (root / system.log).read_bytes()
    except OSError:
        raise Refused(f"{log} is missing or unreadable (the server log in {system.name}'s raw folder)") \
            from None
    tasks = read_tasks(data.decode("utf-8", errors="replace"), log)
    over = next((task for task in tasks if task.generated > source.max_tokens), None)
    if over is not None:
        raise Refused(f"{log}: task {over.number} generated {over.generated} tokens, more than max_tokens "
                      f"{source.max_tokens}")
    expected = system.probes + sum(system.answers.values())
    if len(tasks) != expected:
        answers = " + ".join(f"{n} {language}" for language, n in system.answers.items())
        raise Refused(f"{log} holds {len(tasks)} requests, not {system.probes} probes + {answers} answers = "
                      f"{expected}")
    check_probes(tasks[:system.probes], log)
    generated = [task.generated for task in tasks[system.probes:]]
    row = {"log": log, "log_sha256": hashlib.sha256(data).hexdigest(), "probes": system.probes,
           "requests": len(tasks)}
    start = 0
    for language, n in system.answers.items():
        row[language] = block(generated[start:start + n], source)
        start += n
    row[ALL] = block(generated, source)
    return row


def summarize(root: Path = Path(".")) -> dict:
    """The file's content (module docstring) from the source and the logs under root, the repository root;
    Refused when an input fails a check. Reads, never writes."""
    source = read_source(root)
    systems = {system.name: system_lengths(root, system, source) for system in source.systems}
    return {"kind": KIND, "definition": DEFINITION, "source": SOURCE.as_posix(),
            "source_sha256": source.sha256,
            "budget": {"reasoning": source.reasoning, "max_tokens": source.max_tokens, "from": BUDGET_FROM},
            "languages": source.languages, "systems": systems}


def _share(count: int, n: int) -> str:
    return f"{count} ({100 * count / n:.1f}%)"


def show(data: dict) -> None:
    """The blocks on the screen: aggregates only."""
    budget = data["budget"]
    print(f"{data['source']}: reasoning budget {budget['reasoning']}, max_tokens {budget['max_tokens']}; "
          f"languages {', '.join(data['languages'])}")
    for system, row in data["systems"].items():
        print(f"{system}: {row['log']}, {row['requests']} requests, {row['probes']} probes left out")
        for part in (*data["languages"], ALL):
            cell = row[part]
            print(f"  {part:<4} n {cell['n']}, p50 {cell['tokens_p50']:.0f}, p95 {cell['tokens_p95']:.0f}, "
                  f"max {cell['tokens_max']}, over_budget {_share(cell['over_budget'], cell['n'])}, "
                  f"at_max_tokens {_share(cell['at_max_tokens'], cell['n'])}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="The tokens llama-server generated for each of ACEBench's "
                                             "requests in M7's run, per system and language, from its server "
                                             "logs.")
    ap.add_argument("--force", action="store_true", help=f"overwrite an existing {OUT.as_posix()}")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if OUT.exists() and not args.force:
        print(f"FAIL: {OUT.as_posix()} exists; --force overwrites it", file=sys.stderr)
        return USAGE
    try:
        data = summarize(Path("."))
    except Refused as exc:
        print(f"FAIL: {exc}; nothing is written", file=sys.stderr)
        return REFUSED
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    show(data)
    print(f"{OUT.as_posix()}: written")
    return WRITTEN


if __name__ == "__main__":
    sys.exit(main())

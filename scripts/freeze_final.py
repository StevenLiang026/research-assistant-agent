"""Freeze M7's final local system on the development sets (docs/m7-本地提升计划.md WP5; docs/m7-预登记.md,
revision 2): the file scripts/run_test.py's preflight checks and its dry run holds final and finalg against
(results/final-m7.json). This script writes it; nobody writes it by hand.

    .venv/Scripts/python.exe scripts/freeze_final.py --config results/test-config-m7.json \\
        [--runs <development set> <greedy run> <run of sample 0> ... <run of sample K-1>]... [--out FILE]

The system. The configuration's final system (scripts/run_test.py, A system's options), checked as run_test
checks a configuration but for systems.final.frozen.sha256, which this script's output decides
(run_test.check_config's freezing): adapters.final's GGUF over its base model and tokenizer folder (whose
family is loaded), its generation with systems.final's decoding and votes (run_test.system_generation; K
votes, 0 without a vote), the vote mode, the runtime rules and the glossary. finalg is its greedy self: the
same adapter, greedy, no wrapper.

The runs. Per development set of the configuration's test sets (run_test.dev_sets_of: data/eval/dev.jsonl,
then data/eval/v2/dev-v2.jsonl), the runner records (scripts/run_eval.py) of final's candidates over it: the
greedy candidate's run and, when final votes, one run per sample in sample order, the k-th the run with
--seed base_seed + k (agent.vote.voted_settings). --runs gives them per set; without it they are
DEFAULT_RUNS, FT3's at step 1635 (the greedy runs and seeds 0 to 4; the first K samples when K is less than
5). A final without votes is its greedy candidate with the wrappers (systems.final.decoding greedy), and only
the greedy run is read; a final that samples one answer is not frozen here.

The checks (exit 2, naming every disagreement): every run's manifest (<stem>.manifest.json beside it) records
the configuration's adapter (one adapter, its SHA-256) applied (lora scale 1), its base model
(server_props.model_path), the family of its tokenizer folder (family.record()), the generation of its
candidate (the greedy candidate's for the greedy run, sample k's for the k-th sample run, as
GenerationSettings.record() gives them), the prompt and the tool schemas of this code (PROMPT_HASH,
SCHEMA_HASH), the set's SHA-256 with every item run (items_run and items_total the set's size, no limit), no
wrapper of its own (neither vote_mode nor rules, which are applied here) and the configured glossary or none;
every run holds exactly one record of every item of its set and nothing else; and the sets share no item. The
messages name files and fields and count items, never an item (scripts/decide_m7.py prints them as they are).

The records. Per item: each candidate's record with the runtime rules applied over the item's messages
(agent.rules.ruled_record, when systems.final.rules), then the vote in the vote mode
(agent.vote.voted_record), an item the greedy run never asked the model (context_overflow, budget_stop,
skipped) keeping the greedy record: the records scripts/vote_offline.py makes from the same runs, which the
online final system gives for the same answers. finalg's records are the greedy runs' as they are.

The output (JSON, two-space indent, LF; --out, by default systems.final.frozen.path): run_test.frozen_identity
of the configuration (the adapter, the base model with the SHA-256 of its file, the tokenizer folder, the
family, final's generation with the votes, the votes, the vote mode, the rules, the glossary, the development
sets with their SHA-256), the scorers (eval.metrics.SCORERS), the runs read (their paths and SHA-256), and per
system, final and finalg (with its generation), the successes per scorer and per item, in the sets' order
(dev, then dev-v2): its id, its success under each scorer and run_test.output_hash of its record. The totals
are printed.

Exit 0 when written; 1 on a malformed configuration or a run that is not JSONL objects; 2 on a usage error, a
missing file and every disagreement above; 4 when a set is test material; 5 when a set does not match its
lock. Nothing is written but on exit 0.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from research_agent.agent.prompts import PROMPT_HASH
from research_agent.agent.rules import ruled_record
from research_agent.agent.types import BUDGET_STOP, CONTEXT_OVERFLOW
from research_agent.agent.vote import voted_record, voted_settings
from research_agent.canonical import canonical_json
from research_agent.eval.lock import (
    DevSet,
    JsonlError,
    LockError,
    RefusedTestSet,
    lock_required,
    read_jsonl,
    refuse_test_material,
    refuse_test_sets,
    sha256_of,
)
from research_agent.eval.metrics import SCORERS, score_items
from research_agent.eval.runner import SKIPPED
from research_agent.family.core import LoadedFamily, load_family
from research_agent.llm.generation import GenerationSettings
from research_agent.llm.server import same_path
from research_agent.tools.schemas import SCHEMA_HASH

REPO = Path(__file__).resolve().parents[1]


def _load_run_test():
    """scripts/run_test.py as a module of its own name (scripts/ is no package): the configuration's checks,
    the final system's generation and identity, and output_hash, defined there once."""
    spec = importlib.util.spec_from_file_location("_run_test_shared", REPO / "scripts" / "run_test.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_test = _load_run_test()
FINAL, FINALG = run_test.FINAL, run_test.FINALG
NOT_ASKED = (CONTEXT_OVERFLOW, BUDGET_STOP, SKIPPED)  # an item the greedy run never asked the model
# FT3's runs at step 1635 per development set (module docstring: The runs): the greedy run, then seeds 0 to 4
DEFAULT_RUNS = {
    "data/eval/dev.jsonl": ("runs/m7/dev-ft3-1635-greedy.jsonl",
                            "runs/eval/select-ft3/dev/checkpoint-1635.s0.jsonl",
                            "runs/eval/select-ft3/dev/checkpoint-1635.s1.jsonl",
                            *(f"runs/m7/voting/dev-ft3-1635-s{k}.jsonl" for k in (2, 3, 4))),
    "data/eval/v2/dev-v2.jsonl": ("runs/m7/dev-v2-ft3-1635-greedy.jsonl",
                                  "runs/eval/select-ft3/dev-v2/checkpoint-1635.s0.jsonl",
                                  "runs/eval/select-ft3/dev-v2/checkpoint-1635.s1.jsonl",
                                  *(f"runs/m7/voting/dev-v2-ft3-1635-s{k}.jsonl" for k in (2, 3, 4))),
}
ABOUT = ("M7's final local system frozen on the development sets by scripts/freeze_final.py "
         "(docs/m7-本地提升计划.md WP5): the configuration it was frozen from (adapter, base, tokenizer, "
         "family, generation, votes, vote_mode, rules, glossary, dev_sets), the runs its records were made "
         "from, and per system and development item the success under each scorer and the output hash "
         "(scripts/run_test.py output_hash). scripts/run_test.py's preflight checks it against the "
         "configuration and its dry run holds final's successes and finalg's outputs against it.")


class Disagreement(RuntimeError):
    """What exit 2 names: a run, a manifest or a usage that disagrees with the configuration."""


class Malformed(RuntimeError):
    """A run that is not JSONL objects (exit 1)."""


class Refused(RuntimeError):
    """A set that is test material (exit 4) or does not match its lock (exit 5): the exit code with it."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = code


@dataclasses.dataclass(frozen=True)
class FinalSystem:
    """The final system of a configuration (module docstring: The system): the configuration, the family of
    the final server's tokenizer folder, the adapter's generation (settings), final's (with its votes) and
    its candidates' (the greedy candidate's and every sample's)."""

    config: dict
    family: LoadedFamily
    settings: GenerationSettings
    final: GenerationSettings
    greedy: GenerationSettings
    samples: tuple[GenerationSettings, ...]

    @property
    def options(self) -> dict:
        return run_test.options_of(self.config, FINAL)

    @property
    def votes(self) -> int:
        return self.final.votes

    @property
    def vote_mode(self) -> str:
        return self.options.get(run_test.VOTE_MODE, run_test.VOTE_FINAL)

    @property
    def rules(self) -> bool:
        return bool(self.options.get(run_test.RULED_KEY))


def final_system(config: dict, load: Callable[[str], LoadedFamily] = load_family) -> FinalSystem:
    """The configuration's final system (module docstring: The system); Disagreement when the configuration
    runs no final system or its final samples one answer, run_test.Missing when the family cannot be
    loaded."""
    if FINAL not in config["systems"]:
        raise Disagreement(f"the configuration runs no {FINAL} system: there is nothing to freeze")
    generated = run_test.generations(config, load)[run_test.SERVER_OF[FINAL]]
    final = run_test.system_generation(config, FINAL, generated.settings)
    if final.votes:
        greedy, samples = voted_settings(final)
    elif final.samples:
        raise Disagreement(f"{FINAL} samples one answer per step without a vote (systems.{FINAL} has neither "
                           "votes nor decoding greedy): this script freezes a voted or a greedy final")
    else:
        greedy, samples = final, []
    return FinalSystem(config, generated.family, generated.settings, final, greedy, tuple(samples))


def open_set(path: Path) -> list[dict]:
    """A development set's items, lock-checked when it has a lock (as scripts/vote_offline.py opens them);
    Refused for test material (4, no item named) and a lock that does not match (5), Malformed for a file
    that is not JSONL objects."""
    roots = (Path.cwd(), REPO)
    try:
        refuse_test_sets(path, roots)
        items = DevSet(path).open() if lock_required(path, roots) else read_jsonl(path)
        refuse_test_material(items, path)
    except RefusedTestSet:
        raise Refused(f"{path.as_posix()} is test material: only scripts/run_test.py opens it", 4) from None
    except LockError as exc:
        raise Refused(str(exc), 5) from None
    except JsonlError as exc:
        raise Malformed(f"{path.as_posix()} is not JSONL objects: {exc}") from None
    except OSError as exc:
        raise Disagreement(f"{path.as_posix()} cannot be read: {type(exc).__name__}") from None
    unusable = sum(not (isinstance(item.get("id"), str) and isinstance(item.get("messages"), list))
                   for item in items)
    if unusable:
        raise Malformed(f"{path.as_posix()}: {unusable} item(s) without a string id or without messages")
    return items


def check_distinct(sets: dict[str, list[dict]]) -> None:
    """Disagreement, by a count, unless every item of the sets has an id of its own."""
    ids = [item["id"] for items in sets.values() for item in items]
    if len(set(ids)) != len(ids):
        raise Disagreement(f"the sets {', '.join(sets)} repeat {len(ids) - len(set(ids))} item id(s)")


def manifest_path(run: Path) -> Path:
    return run.with_name(run.stem + ".manifest.json")


def manifest_problems(system: FinalSystem, manifest: dict, candidate: GenerationSettings, items: Path,
                      n: int) -> list[str]:
    """What a run's manifest records otherwise than the configuration gives it (module docstring: The checks),
    each naming the field."""
    config = system.config
    adapter = config["adapters"][run_test.SERVER_OF[FINAL]]
    base = run_test.base_model(config, run_test.SERVER_OF[FINAL])
    problems = []
    adapters = manifest.get("adapters")
    if not (isinstance(adapters, list) and len(adapters) == 1 and isinstance(adapters[0], dict)
            and adapters[0].get("sha256") == adapter["sha256"]):
        problems.append(f"adapters {canonical_json(adapters)}, not the one adapter with SHA-256 "
                        f"{adapter['sha256']}")
    if manifest.get("lora_scale") != 1.0:
        problems.append(f"lora_scale {manifest.get('lora_scale')!r}, not 1: the adapter not applied")
    props = manifest.get("server_props") if isinstance(manifest.get("server_props"), dict) else {}
    model_path = props.get("model_path")
    if not isinstance(model_path, str) or not same_path(model_path, base):
        problems.append(f"server_props.model_path {model_path!r}, not the base model {base}")
    family = system.family.record()
    if manifest.get("family") != family:
        problems.append(f"family {canonical_json(manifest.get('family'))}, not {canonical_json(family)}")
    if manifest.get("generation") != candidate.record():
        problems.append(f"generation {canonical_json(manifest.get('generation'))}, not "
                        f"{canonical_json(candidate.record())}, its candidate's")
    for key, value in (("prompt_hash", PROMPT_HASH), ("schema_hash", SCHEMA_HASH)):
        if manifest.get(key) != value:
            problems.append(f"{key} {manifest.get(key)!r}, not this code's {value}")
    digest = sha256_of(items)
    if manifest.get("items_sha256") != digest:
        problems.append(f"items_sha256 {manifest.get('items_sha256')!r}, not that of {items.as_posix()} "
                        f"({digest})")
    if (manifest.get("items_run"), manifest.get("items_total"), manifest.get("limit")) != (n, n, None):
        problems.append(f"items_run {manifest.get('items_run')!r} of items_total "
                        f"{manifest.get('items_total')!r} (limit {manifest.get('limit')!r}), not every one "
                        f"of the {n} items")
    for key in (run_test.VOTE_MODE, run_test.RULED_KEY):
        if key in manifest:
            problems.append(f"{key} {canonical_json(manifest[key])}: a run with a wrapper of its own, which "
                            "this script applies to a candidate's plain records")
    glossary = system.options.get(run_test.GLOSSARY)
    noted = manifest.get(run_test.GLOSSARY)
    if glossary is None and noted is not None:
        problems.append(f"glossary {canonical_json(noted)}, while {FINAL} notes none")
    if glossary is not None and (not isinstance(noted, dict) or noted.get("sha256") != glossary["sha256"]):
        problems.append(f"glossary {canonical_json(noted)}, not {glossary['path']} with SHA-256 "
                        f"{glossary['sha256']}")
    return problems


def run_records(path: Path, items: list[dict]) -> list[dict]:
    """The run's record of every item, in the items' order; Disagreement, by counts alone, unless it holds
    exactly one record of every item and nothing else; Malformed for a file that is not JSONL objects or a
    record whose id is not a string (counted, never shown)."""
    try:
        records = read_jsonl(path)
    except JsonlError as exc:
        raise Malformed(f"{path.as_posix()} is not JSONL objects: {exc}") from None
    except OSError as exc:
        raise Disagreement(f"{path.as_posix()} cannot be read: {type(exc).__name__}") from None
    unnamed = sum(not isinstance(record.get("id"), str) for record in records)
    if unnamed:
        raise Malformed(f"{path.as_posix()}: {unnamed} record(s) whose id is not a string")
    found: dict[object, list[dict]] = {}
    for record in records:
        found.setdefault(record.get("id"), []).append(record)
    ids = {item["id"] for item in items}
    missing = sum(item["id"] not in found for item in items)
    repeated = sum(len(found.get(item["id"], [])) > 1 for item in items)
    other = sum(len(rows) for key, rows in found.items() if key not in ids)
    if missing or repeated or other:
        raise Disagreement(f"{path.as_posix()} holds no record of {missing} of the {len(items)} items, more "
                           f"than one of {repeated}, and {other} record(s) of no item: every item is run "
                           "once")
    return [found[item["id"]][0] for item in items]


def check_counts(system: FinalSystem, runs: dict[str, list[str]]) -> None:
    """Disagreement unless every set has a greedy run and one run per sample of the vote."""
    for path, paths in runs.items():
        if len(paths) != 1 + system.votes:
            raise Disagreement(f"{path}: {len(paths) - 1} sample runs, not the {system.votes} of the vote (a "
                               "greedy run and one run per sample)")


def candidate_runs(system: FinalSystem, items: Path, rows: list[dict],
                   runs: Sequence[str]) -> list[list[dict]]:
    """The records of the greedy run and of every sample run over a set's items, rows (module docstring: The
    runs), their manifests checked; Disagreement naming every problem."""
    problems = []
    for run, candidate in zip(runs, (system.greedy, *system.samples), strict=True):
        manifest = manifest_path(Path(run))
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
        except FileNotFoundError:
            problems.append(f"{manifest.as_posix()} is missing")
            continue
        except (OSError, ValueError) as exc:
            problems.append(f"{manifest.as_posix()} cannot be read: {type(exc).__name__}")
            continue
        if not isinstance(data, dict):
            problems.append(f"{manifest.as_posix()} is not a JSON object")
            continue
        problems += [f"{manifest.as_posix()}: {problem}" for problem in
                     manifest_problems(system, data, candidate, items, len(rows))]
    if problems:
        raise Disagreement("; ".join(problems))
    return [run_records(Path(run), rows) for run in runs]


def final_records(items: list[dict], greedy: list[dict], samples: Sequence[list[dict]], mode: str,
                  rules: bool) -> list[dict]:
    """The records the final system gives (module docstring: The records), in the items' order: each
    candidate's record ruled when rules, then the vote over the samples in mode; an item the greedy run never
    asked the model, or one without samples, keeps the greedy record."""
    out = []
    for position, item in enumerate(items):
        history = item["messages"]
        given = [greedy[position], *(run[position] for run in samples)]
        if rules:
            given = [ruled_record(record, history) for record in given]
        if not samples or given[0].get("status") in NOT_ASKED:
            out.append(given[0])
        else:
            out.append(voted_record(given[0], given[1:], history, mode))
    return out


def entries(records: list[dict], items: list[dict]) -> list[dict]:
    """Per record, in the items' order: its id, its success under each scorer and its output hash."""
    scores = {scorer: score_items(records, items, scorer) for scorer in SCORERS}
    return [{"id": record["id"],
             "success": {scorer: bool(scores[scorer][k]["success"]) for scorer in SCORERS},
             "output_sha256": run_test.output_hash(record)} for k, record in enumerate(records)]


def totals(rows: list[dict]) -> dict[str, int]:
    return {scorer: sum(row["success"][scorer] for row in rows) for scorer in SCORERS}


def runs_by_set(config: dict, given: list[list[str]] | None, votes: int) -> dict[str, list[str]]:
    """The runs per development set of the configuration's test sets: those --runs gives, else DEFAULT_RUNS
    (their greedy run and first votes samples); Disagreement for a set without runs or runs of another set."""
    sets = [dev.as_posix() for dev in run_test.dev_sets_of(config)]
    if given is None:
        return {path: list(DEFAULT_RUNS.get(path, ()))[:1 + votes] for path in sets}
    runs: dict[str, list[str]] = {}
    for path, *paths in given:
        if path not in sets or path in runs:
            raise Disagreement(f"--runs {path}: not one of the development sets {', '.join(sets)}, or given "
                               "twice")
        runs[path] = paths
    missing = [path for path in sets if path not in runs]
    if missing:
        raise Disagreement(f"no runs of {', '.join(missing)} (--runs <set> <greedy run> <sample runs>...)")
    return {path: runs[path] for path in sets}


def freeze(system: FinalSystem, runs: dict[str, list[str]]) -> dict:
    """The frozen file's content (module docstring: The output)."""
    check_counts(system, runs)
    sets = {path: open_set(Path(path)) for path in runs}
    check_distinct(sets)
    final_rows, greedy_rows, read = [], [], []
    for path, paths in runs.items():
        records = candidate_runs(system, Path(path), sets[path], paths)
        voted = final_records(sets[path], records[0], records[1:], system.vote_mode, system.rules)
        final_rows += entries(voted, sets[path])
        greedy_rows += entries(records[0], sets[path])
        read.append({"set": path, "greedy": {"path": paths[0], "sha256": sha256_of(paths[0])},
                     "samples": [{"path": run, "sha256": sha256_of(run)} for run in paths[1:]]})
    identity = run_test.frozen_identity(system.config, system.family, system.settings)
    return {"about": ABOUT, **identity, "scorers": list(SCORERS), "runs": read,
            FINAL: {"successes": totals(final_rows), "items": final_rows},
            FINALG: {"generation": system.greedy.record(), "successes": totals(greedy_rows),
                     "items": greedy_rows}}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Freeze M7's final local system on the development sets.")
    ap.add_argument("--config", type=Path, required=True,
                    help="the configuration, results/test-config-m7.json")
    ap.add_argument("--runs", nargs="+", action="append", default=None, metavar="PATH",
                    help="a development set, then its greedy run and its sample runs in sample order "
                         "(repeated per set; default: FT3's at step 1635)")
    ap.add_argument("--out", type=Path, default=None,
                    help="the frozen file (default: systems.final.frozen.path)")
    return ap


def main(argv: list[str] | None = None, *, load_family: Callable[[str], LoadedFamily] = load_family) -> int:
    """load_family stands in for research_agent.family.load_family in tests."""
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    try:
        config = run_test.load_config(args.config, freezing=True)
        system = final_system(config, load_family)
        runs = runs_by_set(config, args.runs, system.votes)
        frozen = freeze(system, runs)
    except run_test.ConfigError as exc:
        print(f"FAIL: {exc}")
        return 1
    except Malformed as exc:
        print(f"FAIL: {exc}")
        return 1
    except (Disagreement, run_test.Missing) as exc:
        print(f"FAIL: {exc}")
        return 2
    except Refused as exc:
        print(f"FAIL: {exc}")
        return exc.code
    except OSError as exc:  # the base model or a glossary to hash
        print(f"FAIL: {exc.filename or exc}: {type(exc).__name__}")
        return 2
    out = args.out or Path(run_test.options_of(config, FINAL)[run_test.FROZEN]["path"])
    run_test.write_json(frozen, out)
    n = len(frozen[FINAL]["items"])
    for name in (FINAL, FINALG):
        counts = frozen[name]["successes"]
        print(f"{name}: " + ", ".join(f"{scorer} {count}" for scorer, count in counts.items()) + f" of {n}")
    how = (f"a vote of {system.votes} samples in the {system.vote_mode} mode" if system.votes else "greedy")
    print(f"{FINAL} is {how}{' with the runtime rules' if system.rules else ''}; {FINALG} greedy, no wrapper")
    print(f"wrote {out.as_posix()} (SHA-256 {sha256_of(out)}): record it as systems.{FINAL}.frozen.sha256")
    return 0


if __name__ == "__main__":
    sys.exit(main())

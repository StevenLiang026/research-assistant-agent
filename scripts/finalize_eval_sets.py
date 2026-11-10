"""Assemble the development and test sets from the reviewed drafted items and the handwritten ones.

    uv run --no-sync python scripts/finalize_eval_sets.py
        [--runs runs/m2/synth/dev runs/m2/synth/test] [--pool-ids data/eval/review/pool-ids.jsonl]
        [--verdicts data/eval/review/verdicts.jsonl] [--handwritten data/eval/handwritten.jsonl]
        [--handwritten-verdicts data/eval/review/handwritten-verdicts.jsonl] [--out-dir data/eval]

research_agent.eval.sets states the rule. The drafted items are drafted again from the generator runs
(eval.items.draft_items, validated requests only, as tools/eval-review/build_pool.py drafted the review pool),
so a set holds exactly what the runs produced. Writes <out-dir>/dev.jsonl and test.jsonl (canonical JSON
lines, eval.lock.write_jsonl) and <out-dir>/sets-manifest.json: per split the item counts, drafted and
handwritten, the handwritten share and the table of every stratum; the SHA-256 of every input (the pool ids,
both verdict files, the handwritten items and specs, the counts the handwritten jobs worked from, every run's
manifest.json), of the review script tools/eval-review/review_items.py and of the end-to-end task files;
the review model, its batches and the verdict counts by category; and who wrote the handwritten items. Every
item keeps its source field (source.request for a drafted item, source.handwritten for a handwritten one),
so a system can be scored on the two kinds separately: the gold of a drafted item is the teacher's own
turn, which favours the teacher (B2). The output is deterministic: no timestamps.

Lock the sets afterwards with scripts/lock_set.py. Exit 0 when written; 1 when the sets cannot be assembled
(eval.sets.SetError: a pool id the runs do not draft, an id seen twice, a short stratum) or an input is
malformed; 2 when an input file is missing. Nothing is written on exit 1 or 2.

The second family (--family v2; M7, docs/m7-计划.md WP-I 5), from the repository root:

    .venv/Scripts/python.exe scripts/finalize_eval_sets.py --family v2
        [--runs runs/m7/synth/v2 runs/m7/synth/v2b] [--review-dir data/eval/v2/review]
        [--handwritten data/eval/v2/handwritten-v2.jsonl]
        [--handwritten-specs data/eval/v2/handwritten-specs-v2.jsonl] [--out-dir data/eval/v2]

writes <out-dir>/test-v2.jsonl, dev-v2.jsonl and dev-v2-ext.jsonl by eval.sets.assemble_v2 (its module
docstring states the rule; tools/eval-review-v2/README.md the review's) and <out-dir>/sets-manifest-v2.json.
The drafted items are drafted again from every run (validated requests only, as
tools/eval-review-v2/build_pool.py drafted the review pool from both runs); an id drafted by two runs is
refused, every pool id must be drafted, and only pool items enter a set. The review is read from
--review-dir under the names its tools wrote there: pool-ids.jsonl (every row with its batch),
verdicts.jsonl (the first review of every pool item, none of another id), the second reviews in
verdicts-double/ (one file per batch reviewed twice, batch-NNN.jsonl, judging exactly the items
pool-ids.jsonl assigns to batch NNN), likewise for the handwritten items handwritten-verdicts.jsonl,
handwritten-verdicts-double/ and handwritten-batches/ (the review's batch files, read for their ids: every
handwritten item in exactly one), handwritten-needed.json (tools/eval-review-v2/needed.py's count, which
every stratum of the three sets must reproduce: its kept, taken, needed and short counts) and
instructions.md (the reviewers' standard, recorded when it is there). An item reviewed twice counts as kept
only when both reviews keep it (eval.sets.combined_verdicts), the rule needed.py counted with. The manifest
has v1's shape, per set instead of per split: per set its split, items, quota, drafted, handwritten,
handwritten share, short (quota - items), SHA-256 and the table of every stratum (v1's fields and short);
the SHA-256 of every input (each run's manifest.json and index.jsonl, the review's files, the handwritten
items and specs); the reviewers, the review's batches, the verdict counts by category of the first and the
second reviews and what the both-keep rule kept; and who wrote the handwritten items. The end-to-end tasks
of v2 are locked and recorded by their own build (data/eval/v2/e2e-build.md) and are not read. A set that
has a lock beside it is written only with the bytes its lock records (the same inputs give the same bytes),
so a locked set is never written over: exit 3, and nothing is written. Exit 0, 1 and 2 as above; in v2 a run
the drafter refuses (eval.items.draft_items: a HarnessError, or a transcript its index names that is not
there) is malformed input too, exit 1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

from research_agent.agent.types import HarnessError
from research_agent.canonical import canonical_json
from research_agent.eval.items import draft_items
from research_agent.eval.lock import (
    HANDWRITTEN_ITEMS_V2,
    HANDWRITTEN_SPECS_V2,
    V2_DIR,
    JsonlError,
    LockError,
    lock_path,
    read_jsonl,
    read_lock,
    sha256_of,
    write_jsonl,
)
from research_agent.eval.sets import (
    DEV_QUOTAS_V2,
    DRAFTED_SHARE,
    KEEP,
    TEST_V2,
    SetError,
    assemble,
    assemble_v2,
    combined_verdicts,
    verdict_counts,
)

RUNS = (Path("runs/m2/synth/dev"), Path("runs/m2/synth/test"))
POOL_IDS = Path("data/eval/review/pool-ids.jsonl")
VERDICTS = Path("data/eval/review/verdicts.jsonl")
HANDWRITTEN = Path("data/eval/handwritten.jsonl")
HANDWRITTEN_VERDICTS = Path("data/eval/review/handwritten-verdicts.jsonl")
HANDWRITTEN_SPECS = Path("data/eval/handwritten-specs.jsonl")
HANDWRITTEN_NEEDED = Path("data/eval/review/handwritten-needed.json")
REVIEW_SCRIPT = Path("tools/eval-review/review_items.py")
E2E_TASKS = ("e2e-dev.jsonl", "e2e-test.jsonl")
MANIFEST = "sets-manifest.json"
HANDWRITTEN_AUTHOR = (
    "Written from scratch by Claude Code (Opus 5.5) in headless jobs on 2026-10-15, as specs in "
    "data/eval/handwritten-specs.jsonl built into items by scripts/build_handwritten_items.py; not drafted "
    "from teacher trajectories, and not written by the project owner. docs/plan.md asks for at least 30% of "
    "the test set written from scratch by the owner; that share is met by these items instead, the owner "
    "having decided on 2026-10-13 that Claude Code writes the owner's parts and that the division of labour "
    "is described as it is.")
RULE = ("per split and (type, language) stratum: the drafted items the review kept, in review-pool order, up "
        f"to the stratum's quota at share {DRAFTED_SHARE}; the rest of its full quota from the handwritten "
        "items the review kept, in id order; a verdict other than keep never enters a set")
# The second family (module docstring): its runs, review folder and the files the review's tools wrote there,
# its handwritten files and its output folder and manifest
FAMILIES = ("v1", "v2")
V1_DEFAULTS = {"runs": list(RUNS), "pool_ids": POOL_IDS, "verdicts": VERDICTS, "handwritten": HANDWRITTEN,
               "handwritten_verdicts": HANDWRITTEN_VERDICTS, "out_dir": Path("data/eval")}
V1_ONLY = ("pool_ids", "verdicts", "handwritten_verdicts")  # v2 reads them from --review-dir
V2_ONLY = ("review_dir", "handwritten_specs")
RUNS_V2 = (Path("runs/m7/synth/v2"), Path("runs/m7/synth/v2b"))
REVIEW_V2 = V2_DIR / "review"
REVIEW_FILES = {"pool_ids": "pool-ids.jsonl", "verdicts": "verdicts.jsonl",
                "verdicts_double": "verdicts-double", "handwritten_verdicts": "handwritten-verdicts.jsonl",
                "handwritten_verdicts_double": "handwritten-verdicts-double",
                "handwritten_batches": "handwritten-batches", "needed": "handwritten-needed.json",
                "instructions": "instructions.md"}
REVIEW_FOLDERS = ("verdicts_double", "handwritten_verdicts_double", "handwritten_batches")
MANIFEST_V2 = "sets-manifest-v2.json"
BATCH = re.compile(r"\d{3}")  # a pool row's batch, as build_pool.py numbers them
HANDWRITTEN_AUTHOR_V2 = (
    "Written from scratch by Claude Code in Workflow subagents on 2026-10-26, "
    "as specs in data/eval/v2/handwritten-specs-v2.jsonl built into items by "
    "scripts/build_handwritten_items.py --family v2; not drafted from teacher trajectories, and not written "
    "by the project owner. docs/plan.md asks for at least 30% of the test set written from scratch by the "
    "owner; that share is met by these items instead, the owner having decided on 2026-10-13 that Claude "
    "Code writes the owner's parts and that the division of labour is described as it is.")
REVIEWERS_V2 = (
    "Workflow subagents of Claude Code, one per batch file, each judging its batch alone by "
    "data/eval/v2/review/instructions.md (v1's RUBRIC and CATEGORIES verbatim, with the agent's tool "
    "list); the batches listed under batches.drafted_double and batches.handwritten_double were reviewed a "
    "second time by another subagent, independently (tools/eval-review-v2/README.md, "
    "data/eval/v2/handwritten-build.md)")
RULE_V2 = (
    "test-v2, per (type, language) stratum of the test split: the drafted items the review kept, in "
    f"review-pool order, up to the stratum's quota (QUOTAS_TEST) at share {DRAFTED_SHARE}; the rest of its "
    "full quota from the handwritten items the review kept, in id order. dev-v2 and dev-v2-ext, per stratum "
    "of the dev split, without handwritten items: the drafted items the review kept, in review-pool order, "
    "first to dev-v2 up to its quota (QUOTAS_DEV scaled to 140), then to dev-v2-ext up to its own "
    "(QUOTAS_DEV scaled to 70), so a stratum short of kept items leaves dev-v2-ext short first. An item "
    "reviewed twice is kept only when both reviews keep it; a verdict other than keep never enters a set")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble data/eval/dev.jsonl and test.jsonl from the "
                                                 "reviewed drafted and handwritten items; with --family v2, "
                                                 "data/eval/v2/test-v2.jsonl, dev-v2.jsonl and "
                                                 "dev-v2-ext.jsonl.")
    parser.add_argument("--family", choices=FAMILIES, default="v1",
                        help="the family of sets to assemble (default v1)")
    parser.add_argument("--runs", type=Path, nargs="+",
                        help="generator run folders (v1: runs/m2/synth/dev and test; v2: runs/m7/synth/v2 "
                             "and v2b)")
    parser.add_argument("--pool-ids", type=Path, help=f"v1 only (default {POOL_IDS.as_posix()})")
    parser.add_argument("--verdicts", type=Path, help=f"v1 only (default {VERDICTS.as_posix()})")
    parser.add_argument("--handwritten", type=Path,
                        help=f"default {HANDWRITTEN.as_posix()}; v2: {HANDWRITTEN_ITEMS_V2.as_posix()}")
    parser.add_argument("--handwritten-verdicts", type=Path,
                        help=f"v1 only (default {HANDWRITTEN_VERDICTS.as_posix()})")
    parser.add_argument("--review-dir", type=Path,
                        help=f"v2 only: the review's folder (default {REVIEW_V2.as_posix()})")
    parser.add_argument("--handwritten-specs", type=Path,
                        help="v2 only: the specs whose SHA-256 the manifest records (default "
                             f"{HANDWRITTEN_SPECS_V2.as_posix()})")
    parser.add_argument("--out-dir", type=Path, help=f"default data/eval; v2: {V2_DIR.as_posix()}")
    return parser


def by_id(rows: list[dict]) -> dict[str, str]:
    return {row["id"]: row.get("verdict") for row in rows}


def digest(path: Path) -> dict:
    return {"path": path.as_posix(), "sha256": sha256_of(path)}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    given = [f"--{key.replace('_', '-')}" for key in (*V1_ONLY, *V2_ONLY) if getattr(args, key) is not None]
    if args.family == "v2":
        wrong = [name for name in given if name[2:].replace("-", "_") in V1_ONLY]
        if wrong:
            parser.error(f"{', '.join(wrong)}: v2 reads the review from --review-dir")
        return main_v2(args)
    wrong = [name for name in given if name[2:].replace("-", "_") in V2_ONLY]
    if wrong:
        parser.error(f"{', '.join(wrong)} go with --family v2")
    for key, value in V1_DEFAULTS.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    required = [args.pool_ids, args.verdicts, args.handwritten, args.handwritten_verdicts,
                *(run / "index.jsonl" for run in args.runs), *(run / "manifest.json" for run in args.runs)]
    missing = [p.as_posix() for p in required if not p.is_file()]
    if missing:
        print(f"missing input: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        pool_rows = read_jsonl(args.pool_ids)
        verdict_rows, handwritten_rows = read_jsonl(args.verdicts), read_jsonl(args.handwritten_verdicts)
        handwritten = read_jsonl(args.handwritten)
        drafted: dict[str, dict] = {}
        for run in args.runs:
            for item in draft_items(run, splits=("dev", "test"), include_unvalidated=False):
                drafted[item["id"]] = item
        sets, table = assemble(pool_rows, by_id(verdict_rows), drafted, handwritten, by_id(handwritten_rows))
    except (SetError, JsonlError, KeyError, ValueError, TypeError) as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    out_dir = args.out_dir
    for split, items in sets.items():
        write_jsonl(items, out_dir / f"{split}.jsonl")
    inputs = {"pool_ids": digest(args.pool_ids), "verdicts": digest(args.verdicts),
              "handwritten": digest(args.handwritten),
              "handwritten_verdicts": digest(args.handwritten_verdicts),
              "runs": [digest(run / "manifest.json") for run in args.runs]}
    for name, path in (("handwritten_specs", HANDWRITTEN_SPECS), ("handwritten_needed", HANDWRITTEN_NEEDED)):
        if path.is_file():
            inputs[name] = digest(path)
    review_rows = verdict_rows + handwritten_rows
    manifest = {
        "rule": RULE,
        "splits": {split: {"items": len(items),
                           "drafted": sum(1 for i in items if not (i.get("source") or {}).get("handwritten")),
                           "handwritten": sum(1 for i in items if (i.get("source") or {}).get("handwritten")),
                           "handwritten_share": round(sum(1 for i in items if (i.get("source") or {}).get(
                               "handwritten")) / len(items), 4) if items else None,
                           "sha256": sha256_of(out_dir / f"{split}.jsonl"), "strata": table[split]}
                   for split, items in sets.items()},
        "inputs": inputs,
        "review": {"models": sorted({str(r.get("model")) for r in review_rows}),
                   "batches": sorted({str(r.get("batch")) for r in review_rows}),
                   "script": digest(REVIEW_SCRIPT) if REVIEW_SCRIPT.is_file() else None,
                   "drafted": verdict_counts(verdict_rows), "handwritten": verdict_counts(handwritten_rows)},
        "handwritten_author": HANDWRITTEN_AUTHOR,
        "e2e_tasks": {name: sha256_of(out_dir / name) for name in E2E_TASKS if (out_dir / name).is_file()},
    }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                                    encoding="utf-8", newline="\n")
    for split, info in manifest["splits"].items():
        print(f"{split}: {info['items']} items, drafted {info['drafted']}, handwritten {info['handwritten']} "
              f"({100 * info['handwritten_share']:.1f}%), sha256 {info['sha256'][:16]}")
    print(f"manifest {(out_dir / MANIFEST).as_posix()}")
    return 0


# ---- the second family (v2) ------------------------------------------------------------------------------


def is_handwritten(item: dict) -> bool:
    return bool((item.get("source") or {}).get("handwritten"))


def draft_runs(runs: list[Path]) -> dict[str, dict]:
    """id -> item over every run's drafts (validated requests only); SetError for an id two runs draft."""
    drafted: dict[str, dict] = {}
    twice: set[str] = set()
    for run in runs:
        for item in draft_items(run, splits=("dev", "test"), include_unvalidated=False):
            if item["id"] in drafted:
                twice.add(item["id"])
            drafted[item["id"]] = item
    if twice:
        raise SetError(f"{len(twice)} items are drafted by more than one run, e.g. {sorted(twice)[:5]}")
    return drafted


def check_reviewed(first: list[dict], ids: set[str], what: str, where: str) -> None:
    """SetError unless the first reviews judge exactly the items ids names (what they are, where they are
    listed)."""
    judged = {row.get("id") for row in first}
    outside = sorted(str(i) for i in judged - ids)
    if outside:
        raise SetError(f"{len(outside)} first reviews judge ids outside the {what}, e.g. {outside[:5]}")
    unreviewed = sorted(ids - judged)
    if unreviewed:
        raise SetError(f"{len(unreviewed)} {what} have no first review, e.g. {unreviewed[:5]} ({where})")


def second_reviews(folder: Path, batches: dict[str, Counter], listing: Path) -> tuple[list[Path], list[dict]]:
    """The files of folder (one per batch reviewed twice, by name) and their review lines in name order;
    SetError for a file that names no batch of listing or does not judge exactly that batch's items."""
    files = sorted(folder.glob("*.jsonl"))
    rows: list[dict] = []
    for path in files:
        if path.stem not in batches:
            raise SetError(f"{path.as_posix()} names no batch of {listing.as_posix()}")
        lines = read_jsonl(path)
        if Counter(row.get("id") for row in lines) != batches[path.stem]:
            raise SetError(f"{path.as_posix()} does not judge exactly the items {listing.name} assigns to "
                           f"{path.stem}")
        rows += lines
    return files, rows


def pool_batches(rows: list[dict], listing: Path) -> dict[str, Counter]:
    """batch-NNN -> the ids pool-ids.jsonl assigns to it; SetError for a row without a batch."""
    batches: dict[str, Counter] = {}
    for row in rows:
        batch = row.get("batch")
        if not isinstance(batch, str) or BATCH.fullmatch(batch) is None:
            raise SetError(f"{listing.as_posix()}: pool row {row.get('id')} has no batch")
        batches.setdefault(f"batch-{batch}", Counter())[row["id"]] += 1
    return dict(sorted(batches.items()))


def handwritten_batches(folder: Path, ids: set[str]) -> tuple[list[Path], dict[str, Counter]]:
    """The review's batch files of the handwritten items and batch name -> the ids of each; SetError unless
    they hold every handwritten item exactly once and nothing else."""
    files = sorted(folder.glob("*.jsonl"))
    batches = {path.stem: Counter(row.get("id") for row in read_jsonl(path)) for path in files}
    held = sum(batches.values(), Counter())
    if set(held) != ids or any(n != 1 for n in held.values()):
        raise SetError(f"the handwritten batches do not hold every handwritten item exactly once "
                       f"({folder.as_posix()}: {sum(held.values())} lines over {len(held)} ids for "
                       f"{len(ids)} items)")
    return files, batches


def check_needed(table: dict[str, dict], counted: object, path: Path) -> None:
    """SetError unless every stratum of the sets gives what needed.py counted for it (path): per test
    stratum full, drafted target, kept, taken and needed; per dev stratum kept and, per development set, its
    quota, taken and short; and the count has no other stratum."""
    strata = counted.get("strata") if isinstance(counted, dict) else None
    if not isinstance(strata, dict):
        raise SetError(f"{path.as_posix()} has no strata")
    mine = {f"test/{key}": {"full": row["full"], "drafted_target": row["target"], "kept": row["drafted_kept"],
                            "taken": row["drafted"], "needed": row["handwritten"]}
            for key, row in table[TEST_V2].items()}
    first = next(iter(DEV_QUOTAS_V2))
    for key, row in table[first].items():
        mine[f"dev/{key}"] = {"kept": row["drafted_kept"]}
        for name in DEV_QUOTAS_V2:
            cell = table[name][key]
            mine[f"dev/{key}"][name] = {"quota": cell["full"], "taken": cell["drafted"],
                                        "short": cell["short"]}
    for key in sorted(set(mine) | set(strata)):
        if mine.get(key) != strata.get(key):
            raise SetError(f"{key}: the sets give {mine.get(key)} where {path.name} counts {strata.get(key)}")


def review_stats(first: list[dict], second: list[dict], kept: dict[str, str]) -> dict:
    """What the review judged and the both-keep rule kept: items, reviewed twice, agreeing (both reviews
    gave the same verdict), kept."""
    verdict = {row["id"]: row["verdict"] for row in first}
    return {"items": len(first), "reviewed_twice": len(second),
            "agreeing": sum(1 for row in second if row["verdict"] == verdict[row["id"]]),
            "kept": sum(1 for v in kept.values() if v == KEEP)}


def set_record(split: str, items: list[dict], strata: dict, path: Path) -> dict:
    handwritten = sum(1 for i in items if is_handwritten(i))
    quota = sum(row["full"] for row in strata.values())
    return {"split": split, "items": len(items), "quota": quota, "drafted": len(items) - handwritten,
            "handwritten": handwritten,
            "handwritten_share": round(handwritten / len(items), 4) if items else None,
            "short": quota - len(items), "sha256": sha256_of(path), "strata": strata}


def main_v2(args: argparse.Namespace) -> int:
    """The second family's build (module docstring); returns main's exit code."""
    runs = list(args.runs) if args.runs is not None else list(RUNS_V2)
    review = args.review_dir if args.review_dir is not None else REVIEW_V2
    handwritten_path = args.handwritten if args.handwritten is not None else HANDWRITTEN_ITEMS_V2
    specs = args.handwritten_specs if args.handwritten_specs is not None else HANDWRITTEN_SPECS_V2
    out_dir = args.out_dir if args.out_dir is not None else V2_DIR
    paths = {key: review / name for key, name in REVIEW_FILES.items()}
    files = [paths[key] for key in ("pool_ids", "verdicts", "handwritten_verdicts", "needed")]
    files += [handwritten_path, *(run / "index.jsonl" for run in runs),
              *(run / "manifest.json" for run in runs)]
    missing = [p.as_posix() for p in files if not p.is_file()]
    missing += [paths[key].as_posix() for key in REVIEW_FOLDERS if not paths[key].is_dir()]
    if missing:
        print(f"missing input: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        pool_rows = read_jsonl(paths["pool_ids"])
        first, hw_first = read_jsonl(paths["verdicts"]), read_jsonl(paths["handwritten_verdicts"])
        handwritten = read_jsonl(handwritten_path)
        counted = json.loads(paths["needed"].read_text(encoding="utf-8"))
        drafted = draft_runs(runs)
        batches = pool_batches(pool_rows, paths["pool_ids"])
        check_reviewed(first, {row["id"] for row in pool_rows}, "pool items", paths["pool_ids"].as_posix())
        double_files, second = second_reviews(paths["verdicts_double"], batches, paths["pool_ids"])
        hw_ids = {item["id"] for item in handwritten}
        hw_batch_files, hw_batches = handwritten_batches(paths["handwritten_batches"], hw_ids)
        check_reviewed(hw_first, hw_ids, "handwritten items", handwritten_path.as_posix())
        hw_double_files, hw_second = second_reviews(paths["handwritten_verdicts_double"], hw_batches,
                                                    paths["handwritten_batches"])
        verdicts, hw_verdicts = combined_verdicts(first, second), combined_verdicts(hw_first, hw_second)
        sets, table = assemble_v2(pool_rows, verdicts, drafted, handwritten, hw_verdicts)
        check_needed(table, counted, paths["needed"])
    except (SetError, JsonlError, HarnessError, OSError, KeyError, ValueError, TypeError) as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    texts = {name: "".join(canonical_json(item) + "\n" for item in items) for name, items in sets.items()}
    for name, text in texts.items():  # a locked set is written only with the bytes its lock records
        path = out_dir / f"{name}.jsonl"
        if not lock_path(path).is_file():
            continue
        new = hashlib.sha256(text.encode("utf-8")).hexdigest()
        try:
            recorded = read_lock(path)
        except LockError as exc:
            recorded = f"none readable ({exc})"
        if recorded != new:
            print(f"{path.as_posix()} is locked by {lock_path(path).name} with SHA-256 {recorded}, and this "
                  f"build gives {new}: a locked set is not written over; nothing written", file=sys.stderr)
            return 3
    for name, items in sets.items():
        write_jsonl(items, out_dir / f"{name}.jsonl")
    split_of = {TEST_V2: "test", **dict.fromkeys(DEV_QUOTAS_V2, "dev")}
    inputs = {"pool_ids": digest(paths["pool_ids"]), "verdicts": digest(paths["verdicts"]),
              "verdicts_double": [digest(p) for p in double_files], "handwritten": digest(handwritten_path),
              "handwritten_verdicts": digest(paths["handwritten_verdicts"]),
              "handwritten_verdicts_double": [digest(p) for p in hw_double_files],
              "handwritten_batches": [digest(p) for p in hw_batch_files],
              "handwritten_needed": digest(paths["needed"]),
              "runs": [digest(run / "manifest.json") for run in runs],
              "run_indexes": [digest(run / "index.jsonl") for run in runs]}
    if specs.is_file():
        inputs["handwritten_specs"] = digest(specs)
    manifest = {
        "family": "v2",
        "rule": RULE_V2,
        "sets": {name: set_record(split_of[name], items, table[name], out_dir / f"{name}.jsonl")
                 for name, items in sets.items()},
        "inputs": inputs,
        "review": {"reviewers": REVIEWERS_V2,
                   "instructions": digest(paths["instructions"]) if paths["instructions"].is_file() else None,
                   "batches": {"drafted": list(batches), "drafted_double": [p.stem for p in double_files],
                               "handwritten": list(hw_batches),
                               "handwritten_double": [p.stem for p in hw_double_files]},
                   "drafted": verdict_counts(first), "drafted_double": verdict_counts(second),
                   "handwritten": verdict_counts(hw_first), "handwritten_double": verdict_counts(hw_second),
                   "kept": {"drafted": review_stats(first, second, verdicts),
                            "handwritten": review_stats(hw_first, hw_second, hw_verdicts)}},
        "handwritten_author": HANDWRITTEN_AUTHOR_V2,
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (out_dir / MANIFEST_V2).write_text(text, encoding="utf-8", newline="\n")
    for name, info in manifest["sets"].items():
        share = 100 * (info["handwritten_share"] or 0)
        print(f"{name}: {info['items']} items of {info['quota']}, drafted {info['drafted']}, handwritten "
              f"{info['handwritten']} ({share:.1f}%), short {info['short']}, sha256 {info['sha256'][:16]}")
    kept = manifest["review"]["kept"]
    print("review: " + "; ".join(f"{kind} {k['items']} items ({k['reviewed_twice']} reviewed twice, "
                                 f"{k['agreeing']} agreeing), {k['kept']} kept" for kind, k in kept.items()))
    print(f"manifest {(out_dir / MANIFEST_V2).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

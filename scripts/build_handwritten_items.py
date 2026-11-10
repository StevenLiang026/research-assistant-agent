"""Build the hand-written evaluation items from their specs (research_agent.eval.handwritten explains both).

    uv run --no-sync python scripts/build_handwritten_items.py [--specs data/eval/handwritten-specs.jsonl]
        [--out data/eval/handwritten.jsonl] [--needed <handwritten-needed.json>] [--split dev|test]

Reads the specs, plays each one's prior steps through the agent loop over the recorded corpus (the replay
cassettes, the owner's parse_atom, extract_section, check_call and decide, a note store seeded for the item
id as the generator seeds one), runs every check of the module docstring and, when all pass, writes the
items as canonical JSON lines (sorted keys, LF), in spec order, so two runs over the same specs write
byte-identical files. --split keeps the specs of one split only. --needed names the review's
handwritten-needed.json ({"strata": {"<split>/<type>/<lang>": {"needed": n, ...}}}): every stratum of a
split the specs cover must then hold n + 1 + n // 5 specs (the spares replace items a later review drops).
Prints one line per problem and, on success, the counts by split, type and language.

The second family (--family v2; M7, docs/m7-计划.md WP-I 4), from the repository root:

    .venv/Scripts/python.exe scripts/build_handwritten_items.py --family v2 [--specs FILE [FILE ...]]
        [--out FILE] [--needed data/eval/v2/review/handwritten-needed.json]
    .venv/Scripts/python.exe scripts/build_handwritten_items.py --family v2 --batch <batch file>
    .venv/Scripts/python.exe scripts/build_handwritten_items.py --family v2 --needed <needed file>
        --write-plan data/eval/v2/handwritten-specs/plan.json
    .venv/Scripts/python.exe scripts/build_handwritten_items.py --family v2 --specs FILE [FILE ...] --replay

builds the test split's items from data/eval/v2/handwritten-specs-v2.jsonl, or from the --specs files read in
order as one list, into data/eval/v2/handwritten-v2.jsonl (the defaults of v2 lie under the repository root,
wherever the script runs), by the module's checks over a wider world: check 10 against the requests of every
request file (REQUESTS_V2) and, read once per run through eval.lock.RetiredSet, which returns similarities
alone and logs the read (data/eval/handwritten-specs-retired-reads.jsonl, data/eval/test-retired-reads.jsonl),
against v1's hand-written requests and v1's single-step test items; and every id a test id from hw-test-165
on. --batch checks one writer's batch (data/eval/v2/handwritten-writing.md) and writes nothing: every check
above, its entry in the plan beside it (exactly its ids, every spec of its stratum), and the plan's other
batches present (no id of theirs, no request within HANDWRITTEN_LIMIT of one of theirs; a batch that cannot
be read, which another writer may be writing, is named on stderr and passed over). --write-plan writes the
writers' plan of --needed (eval.handwritten.writing_plan, the batches beside the plan) and stops. --replay
prints for each spec the history its prior steps lead to, one ASCII JSON line ({"id", "messages" after the
system prompt, "seen_ids"}, or {"id", "error"}; every non-ASCII character escaped, so the lines decode on any
console), and checks nothing else: it reads no retired file.

Exit 0 when the items were written (--batch: every check passed; --write-plan: the plan was written;
--replay: every spec replayed); 1 when any spec fails a check (each problem is printed with its item id and
nothing is written); 2 when the specs file or the --needed file does not exist, a batch has no plan beside it
or is none of its batches, or v1's retired files refuse the read (eval.lock.RetiredSet: not retired, or not
matching their lock or the SHA-256 the assembly recorded).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from research_agent.eval.handwritten import (
    Retired,
    SpecError,
    batch_problems,
    build_items,
    load_world,
    numbering_problems,
    peer_problems,
    read_specs,
    replayed,
    retired_v1,
    stratum_problems,
    writing_plan,
)
from research_agent.eval.items import summarize
from research_agent.eval.lock import (
    HANDWRITTEN_ITEMS_V2,
    HANDWRITTEN_SPECS_V2,
    LockError,
    sha256_of,
    write_jsonl,
)

SPECS = "data/eval/handwritten-specs.jsonl"
OUT = "data/eval/handwritten.jsonl"
CORPUS_MANIFEST = "data/corpus/manifest.json"
CASSETTES = "data/cassettes"
REQUESTS = "data/synth/requests.jsonl"
REQUESTS_MANIFEST = "data/synth/requests-manifest.json"
# The second family (v2, M7): its default paths lie under the repository root.
REPO = Path(__file__).resolve().parents[1]
REQUESTS_V2 = ("data/synth/requests.jsonl", "data/synth/requests-m6.jsonl", "data/synth/requests-v2.jsonl",
               "data/synth/requests-v2b.jsonl", "data/synth/requests-m7-train.jsonl")
PLAN_NAME = "plan.json"  # the writers' plan, beside their batches (data/eval/v2/handwritten-specs/)
WRITING = "data/eval/v2/handwritten-writing.md"  # the writers' instructions
CHECKER = ".venv/Scripts/python.exe scripts/build_handwritten_items.py --family v2 --batch {file}"
PLAN_RULE = ("per test stratum of the needed file with n hand-written items needed: n + 1 + n // 5 specs "
             "(the spares replace items the review drops), ids from hw-test-165 on in the needed file's "
             "stratum order; batches of at most 25 specs, one stratum per batch, a larger stratum cut into "
             "batches of near-equal size; a batch is checked with the checker command, {file} its out path")
WORLD_KEYS = ("corpus_manifest", "cassettes", "requests", "requests_manifest")


def family_defaults(family: str) -> dict:
    """A family's default paths: v1's relative to the folder the script runs in (the repository root), as they
    always were; v2's under the repository root."""
    if family == "v1":
        return {"specs": SPECS, "out": OUT, "corpus_manifest": CORPUS_MANIFEST, "cassettes": CASSETTES,
                "requests": REQUESTS, "requests_manifest": REQUESTS_MANIFEST}
    return {"specs": REPO / HANDWRITTEN_SPECS_V2, "out": REPO / HANDWRITTEN_ITEMS_V2,
            "corpus_manifest": REPO / CORPUS_MANIFEST, "cassettes": REPO / CASSETTES,
            "requests": [REPO / path for path in REQUESTS_V2], "requests_manifest": REPO / REQUESTS_MANIFEST}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build the hand-written evaluation items from their specs.")
    ap.add_argument("--family", choices=("v1", "v2"), default="v1",
                    help="v1 (the default): the M2 build; v2: the second family's (M7), see the docstring")
    ap.add_argument("--specs", nargs="+", default=None,
                    help=f"the specs, one file or several read in order (default {SPECS}; v2 "
                         f"{HANDWRITTEN_SPECS_V2.as_posix()})")
    ap.add_argument("--out", default=None,
                    help=f"where the items go (default {OUT}; v2 {HANDWRITTEN_ITEMS_V2.as_posix()})")
    ap.add_argument("--needed", default=None, help="handwritten-needed.json: check the count per stratum")
    ap.add_argument("--split", choices=("dev", "test"), default=None, help="build one split only")
    ap.add_argument("--batch", default=None,
                    help="v2: check a writer's batch file against the plan beside it; nothing is written")
    ap.add_argument("--write-plan", default=None, help="v2: write the writers' plan of --needed to this path")
    ap.add_argument("--replay", action="store_true",
                    help="print the history each spec's prior steps lead to, and check nothing else")
    ap.add_argument("--corpus-manifest", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--cassettes", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--requests", nargs="+", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--requests-manifest", default=None, help=argparse.SUPPRESS)
    return ap


# ---- the writers' plan and their batches (v2) -----------------------------------------------------------


def _shown(path: str | Path) -> str:
    """path relative to the repository root when it lies under it, else absolute; with forward slashes."""
    resolved = Path(path).resolve()
    return (resolved.relative_to(REPO) if resolved.is_relative_to(REPO) else resolved).as_posix()


def make_plan(needed_path: str | Path, plan_path: str | Path) -> dict:
    """The writers' plan of the needed file (eval.handwritten.writing_plan), its batches beside plan_path,
    with the rule, the needed file's SHA-256, the writers' instructions and the checker command."""
    needed = json.loads(Path(needed_path).read_text(encoding="utf-8"))
    plan = writing_plan(needed, folder=_shown(Path(plan_path).parent))
    return {"family": "v2", "rule": PLAN_RULE,
            "needed": {"path": _shown(needed_path), "sha256": sha256_of(needed_path)},
            "writing": WRITING, "checker": CHECKER, **plan}


def render_plan(plan: dict) -> str:
    return json.dumps(plan, ensure_ascii=False, indent=1) + "\n"


def find_batch(path: Path) -> tuple[dict, dict] | str:
    """(the plan beside a writer's batch file, the batch's entry in it), or why there is none."""
    plan_path = path.parent / PLAN_NAME
    if not plan_path.is_file():
        return f"no plan at {plan_path}"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    for batch in plan.get("batches") or []:
        if Path(batch["out"]).name == path.name:
            return plan, batch
    return f"{path.name} is not a batch of {plan_path}"


def read_peers(plan: dict, path: Path) -> tuple[list[tuple[str, dict]], list[str]]:
    """The specs of the plan's other batches present beside path, as (file name, spec) pairs, and a warning
    for each that cannot be read (another writer may be writing it)."""
    peers: list[tuple[str, dict]] = []
    warnings = []
    for batch in plan["batches"]:
        other = path.parent / Path(batch["out"]).name
        if other.name == path.name or not other.is_file():
            continue
        try:
            peers += [(other.name, spec) for spec in read_specs(other)]
        except (SpecError, OSError, UnicodeDecodeError) as exc:
            warnings.append(f"{other.name} cannot be read ({exc}); passed over")
    return peers, warnings


def replay_specs(specs: list[dict], world: object) -> int:
    """--replay: per spec one JSON line, its history (after the system prompt) and seen ids, or the error. The
    lines are ASCII (json.dumps' escapes for every other character): a console or pipe in a legacy code page
    (cp1252 when the output is redirected on Windows) would otherwise get bytes no JSON reader decodes, since
    the recorded papers hold accented names."""
    failed = 0
    for spec in specs:
        try:
            history, seen = replayed(spec, world)
        except SpecError as exc:
            print(json.dumps({"id": spec.get("id"), "error": str(exc)}, ensure_ascii=True))
            failed += 1
            continue
        print(json.dumps({"id": spec["id"], "messages": history[1:], "seen_ids": sorted(seen)},
                         ensure_ascii=True))
    return 1 if failed else 0


def write_plan(needed: str, out: str) -> int:
    if not Path(needed).is_file():
        print(f"no needed file at {needed}", file=sys.stderr)
        return 2
    plan = make_plan(needed, out)
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_plan(plan), encoding="utf-8", newline="\n")
    totals = plan["totals"]
    print(f"wrote the plan of {totals['specs']} specs in {totals['batches']} batches to {out}")
    for batch in plan["batches"]:
        print(f"  {batch['batch']}  {batch['stratum']:18} {batch['count']:3}  {batch['first_id']} to "
              f"{batch['last_id']}")
    return 0


# ---- the script ----------------------------------------------------------------------------------------


def main(argv: list[str] | None = None, *, retired: Sequence[Retired] | None = None) -> int:
    """The script; retired, when given, stands in for v1's retired files in the v2 checks (the tests')."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.family == "v1" and (args.batch is not None or args.write_plan is not None):
        parser.error("--batch and --write-plan are the second family's: add --family v2")
    if args.batch is not None and (args.specs or args.needed or args.write_plan):
        parser.error("--batch checks one batch file: no --specs, --needed or --write-plan with it")
    if args.write_plan is not None and args.needed is None:
        parser.error("--write-plan needs --needed")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")  # a cp1252 console must not fail on Chinese text
    if args.write_plan is not None:
        return write_plan(args.needed, args.write_plan)
    defaults, v2 = family_defaults(args.family), args.family == "v2"
    given = [args.batch] if args.batch is not None else args.specs or [defaults["specs"]]
    specs_paths = [Path(p) for p in given]
    for specs_path in specs_paths:
        if not specs_path.is_file():
            print(f"no specs file at {specs_path}", file=sys.stderr)
            return 2
    plan = batch = None
    if args.batch is not None:
        found = find_batch(specs_paths[0])
        if isinstance(found, str):
            print(found, file=sys.stderr)
            return 2
        plan, batch = found
    needed = None
    if args.needed is not None:
        if not Path(args.needed).is_file():
            print(f"no needed file at {args.needed}", file=sys.stderr)
            return 2
        needed = json.loads(Path(args.needed).read_text(encoding="utf-8"))
    specs: list[dict] = []
    for specs_path in specs_paths:
        try:
            specs += read_specs(specs_path)
        except SpecError as exc:
            print(f"{specs_path}: {exc}", file=sys.stderr)
            return 1
    if args.split is not None:
        specs = [s for s in specs if s.get("split") == args.split]
    world_args = {key: getattr(args, key) or defaults[key] for key in WORLD_KEYS}
    if v2 and not args.replay:
        world_args["retired"] = tuple(retired if retired is not None else retired_v1(REPO))
    world = load_world(**world_args)
    if args.replay:
        return replay_specs(specs, world)
    try:
        items, problems = build_items(specs, world)
    except LockError as exc:
        print(f"FAIL: the retired read was refused: {exc}", file=sys.stderr)
        return 2
    if v2:
        problems += numbering_problems(specs)
    if needed is not None:
        problems += stratum_problems(specs, needed)
    if batch is not None:
        peers, warnings = read_peers(plan, specs_paths[0])
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)
        problems += batch_problems(specs, batch) + peer_problems(specs, peers)
    for problem in problems:
        print(problem)
    if batch is not None:
        if problems:
            print(f"{len(problems)} problem(s) in {batch['batch']}; nothing written", file=sys.stderr)
            return 1
        print(f"OK: {batch['batch']} ({batch['stratum']}): {len(specs)} specs pass every check; nothing "
              "written")
        return 0
    if problems:
        print(f"{len(problems)} problem(s); nothing written", file=sys.stderr)
        return 1
    out = args.out or defaults["out"]
    write_jsonl(items, out)
    print(f"wrote {len(items)} items to {out}")
    for split, types in summarize(items).items():
        for type_, langs in types.items():
            print(f"  {split} {type_:9} " + "  ".join(f"{lang} {n}" for lang, n in langs.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The vote and the runtime rules over saved runs (docs/m7-本地提升计划.md WP2, WP3): the records a voted or
ruled system gives online (research_agent.agent.vote, research_agent.agent.rules), made offline from the
records of its candidates' runs, by the same functions.

    .venv/Scripts/python.exe scripts/vote_offline.py <items.jsonl> --greedy <greedy run.jsonl> \\
        [--sample <run with base seed 0>.jsonl --sample <base seed 1> ...] [--vote-mode final|reproduction] \\
        [--rules] [--limit N] [--run-copy DIR] (--out runs/m7/voted/<name>.jsonl | --aggregate-only)

Inputs. The items (a set, data/eval/dev.jsonl or data/eval/v2/dev-v2.jsonl, or items of it; a locked set is
lock-checked, test material is refused) and the runner records (scripts/run_eval.py) of the system's
candidates over them:
the greedy run (--greedy; FT3's is runs/m7/<set>-ft3-1635-greedy.jsonl) and, to vote, one run per sample in
sample order (--sample, K times): sample k is the run with --seed k (agent.vote, Candidates; FT3's seeds 0 and
1 are runs/eval/select-ft3/<set>/checkpoint-1635.s0.jsonl and .s1.jsonl, seeds 2 to 4
runs/m7/voting/<set>-ft3-1635-s2.jsonl to s4). Every run must hold exactly one record of every item (of the
first N with --limit N), found by its id in whatever order; it may hold records of other items too.

The records. For every item, each candidate's record with the runtime rules applied over the item's messages
(--rules; agent.rules.ruled_record: the output rewritten, "rules" added), then, with K samples, the vote in
--vote-mode (final, the default, or reproduction: the 10-29 analysis' keys; agent.vote.voted_record: the
representative's record, the usage of every answered request summed, "votes"); without samples the greedy
record with the rules, the offline run_eval --greedy --rules. A candidate whose record is not ok (an error
record: its model raised, a HarnessError included) is a failed candidate, which the vote goes on without and
lists under its "errors" (agent.vote, Failed candidates). An item the greedy run never asked the model
(context_overflow, budget_stop, skipped) keeps the greedy record as it is, as the runner records it online.
So the records are byte for byte those the online system writes for the same answers: run_eval --votes K
--rules against the server the runs were made on (one slot), whose candidates send the runs' requests.

Totals (printed; the manifest's "totals"): the items; per scorer (eval.metrics.SCORERS, default and
normalized) the successes of the voted records, of the greedy run and of every sample run (with the rules,
when on), and the class mean, the analysis' vote score: per item, the mean success of the chosen class's
samples (the greedy record's when no sample voted); the vote's tallies (unanimous: every sample voted alike;
a tie decided by the greedy answer's class, or by the smallest sample index; no sample voted; not asked; the
chosen key not the greedy answer's; with_errors: the votes that list failed candidates, and
failed_candidates: how many they list); and per rule the candidate answers it changed. Nothing per item.

Outputs. The records at --out (JSONL, one per item in item order, as scripts/score.py scores a run), with the
manifest <out stem>.manifest.json beside them: the items file, its SHA-256, whether its lock was checked, the
number of items run, every run's path and SHA-256, K ("votes"), the vote mode (with samples), the rules'
version (with --rules), the statuses, the totals, the date and git. --aggregate-only prints the totals alone,
the last line as JSON, writes nothing and names no item in any message: malformed input is reported by how
many items or records are wrong, not which (without it, the first is named). The held-out sets are read that
way only (the owner's rule: their records give totals alone). Without --aggregate-only a held-out set is
refused before anything is read, found by its resolved path (HELD_OUT: data/eval/dev-ext.jsonl and
data/eval/v2/dev-v2-ext.jsonl, under the repository or its run copy, held_out_roots: the folder --run-copy
names, else the one the environment variable RAA_RUN_COPY names, no default;
eval.lock.named_set, so a link to one too), by its name (dev-ext.jsonl, dev-v2-ext.jsonl, wherever it is) or
by its content, so a copy under another name too: a file with the SHA-256 of one of those files, which the
script hashes as it runs, only one whose size is the items file's (a file of another size cannot have its
SHA-256), and prints nothing about. Under --aggregate-only no held-out set is looked at.

Exit 0 when the records or the totals are made; 1 on malformed input (an items file or a run that is not
JSONL objects, items without distinct string ids, a run without exactly one record of an item); 2 on a usage
error (nothing to do: no sample and no --rules; --vote-mode without samples; neither or both of --out and
--aggregate-only); 4 when the items are test material, or a held-out set without --aggregate-only, and 5 when
they do not match their lock (nothing is read or written).
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from collections import Counter
from pathlib import Path

from research_agent.agent.rules import RULES, RULES_VERSION, ruled_record
from research_agent.agent.types import BUDGET_STOP, CONTEXT_OVERFLOW
from research_agent.agent.vote import FINAL, GREEDY_CANDIDATE, VOTE_MODES, voted_record
from research_agent.canonical import canonical_json
from research_agent.eval.lock import (
    EVAL_DIR,
    TEST_SPLIT,
    V2_DIR,
    DevSet,
    JsonlError,
    LockError,
    RefusedTestSet,
    find_test_material,
    lock_required,
    named_set,
    read_jsonl,
    refuse_test_material,
    refuse_test_sets,
    sha256_of,
    write_jsonl,
)
from research_agent.eval.metrics import SCORERS, score_items
from research_agent.eval.runner import SKIPPED
from research_agent.train.manifest import REPO, git_commit, git_dirty

# The held-out sets, read for their totals alone, under the repository and, when one is named, its run copy,
# the checkout the GPU jobs run from (module docstring: Outputs; held_out_roots).
HELD_OUT = (EVAL_DIR / "dev-ext.jsonl", V2_DIR / "dev-v2-ext.jsonl")
RUN_COPY_ENV = "RAA_RUN_COPY"  # names the run copy when --run-copy does not; there is no default run copy
HELD_OUT_ROOTS = (REPO,)
NOT_ASKED = (CONTEXT_OVERFLOW, BUDGET_STOP, SKIPPED)  # statuses of an item the model was never asked
TALLIES = ("unanimous", "tie_to_greedy", "tie_to_first", "no_sample_voted", "not_asked", "chosen_not_greedy",
           "with_errors", "failed_candidates")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="The vote and the runtime rules over saved runs, offline.")
    ap.add_argument("items", type=Path, help="the items the runs are over (JSONL)")
    ap.add_argument("--greedy", type=Path, required=True, help="the greedy candidate's run (runner records)")
    ap.add_argument("--sample", type=Path, action="append", default=[], metavar="RUN",
                    help="a sample's run, in sample order: the k-th is the run with --seed k")
    ap.add_argument("--vote-mode", choices=VOTE_MODES, default=None,
                    help=f"the vote's answer keys: {FINAL} (the default) or the 10-29 analysis' reproduction")
    ap.add_argument("--rules", action="store_true",
                    help="apply the runtime rules to every candidate's answer")
    ap.add_argument("--limit", type=int, default=None, metavar="N", help="only the first N items")
    ap.add_argument("--out", type=Path, default=None, help="the voted records (JSONL)")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="print the totals alone and write nothing (the held-out sets)")
    ap.add_argument("--run-copy", type=Path, default=None, metavar="DIR",
                    help=f"the run copy, a second checkout whose held-out sets are found as well (default: "
                         f"the folder the environment variable {RUN_COPY_ENV} names; none when it is unset)")
    return ap


def local_date() -> str:
    return datetime.datetime.now(datetime.UTC).astimezone().date().isoformat()


def manifest_path(out: Path) -> Path:
    return out.with_name(out.stem + ".manifest.json")


def held_out_roots(run_copy: Path | None = None) -> tuple[Path, ...]:
    """The roots the held-out sets are looked for under: HELD_OUT_ROOTS (the repository) and the run copy when
    one is named: run_copy (--run-copy), else the folder the environment variable RAA_RUN_COPY names. Without
    either, HELD_OUT_ROOTS alone."""
    if run_copy is None:
        named = os.environ.get(RUN_COPY_ENV, "").strip()
        run_copy = Path(named) if named else None
    return HELD_OUT_ROOTS if run_copy is None else (*HELD_OUT_ROOTS, Path(run_copy))


def held_out_by(path: Path, roots: tuple[Path, ...] | None = None) -> str | None:
    """How path is a held-out set under roots (default held_out_roots()): "path", "name" or "content" (module
    docstring: Outputs); None when it is none. Prints nothing; a held-out file is read only when its size is
    path's."""
    roots = held_out_roots() if roots is None else roots
    if named_set(path, HELD_OUT, roots) is not None:
        return "path"
    if path.name in {name.name for name in HELD_OUT}:
        return "name"
    try:
        size = path.stat().st_size
    except OSError:  # no such file: nothing to compare, and reading the items will say so
        return None
    digest = None
    for root in roots:
        for name in HELD_OUT:
            held = Path(root) / name
            try:
                if not held.is_file() or held.stat().st_size != size:  # another size, another SHA-256
                    continue
            except OSError:
                continue
            digest = digest or sha256_of(path)
            if sha256_of(held) == digest:
                return "content"
    return None


def refuse_items(items: list[dict], source: Path, *, aggregate_only: bool) -> None:
    """eval.lock.refuse_test_material, whose message names the first item of the test split; under
    --aggregate-only RefusedTestSet with their count alone (module docstring: Outputs)."""
    found = find_test_material(items)
    if found and aggregate_only:
        raise RefusedTestSet(f"{source.as_posix()} holds test material ({len(found)} item(s) with split "
                             f"{TEST_SPLIT!r} or an id of the test split); only the phase-4 test runner "
                             "(scripts/run_test.py) evaluates it")
    refuse_test_material(items, source)


def check_ids(items: list[dict], *, aggregate_only: bool) -> None:
    """ValueError unless every item's id is a distinct non-empty string, the runs' records being found by it;
    the message names the first bad id, or under --aggregate-only counts them (module docstring: Outputs)."""
    seen: set[str] = set()
    bad = []
    for item in items:
        rid = item.get("id")
        if isinstance(rid, str) and rid and rid not in seen:
            seen.add(rid)
        else:
            bad.append(rid)
    if bad:
        raise ValueError(f"{len(bad)} item(s) without an id of their own (a distinct non-empty string)"
                         + ("" if aggregate_only else f", the first {bad[0]!r}"))


def well_formed(record: dict) -> bool:
    """Whether a run record can be ruled, voted and scored: one that is not ok says what it is by its
    status; an ok one holds an output object whose tool_calls, when there are any, are a list of calls, each
    an object whose function is an object with a string name and string arguments (the runner's records)."""
    if record.get("status") != "ok":
        return True
    output = record.get("output")
    if not isinstance(output, dict):
        return False
    calls = output.get("tool_calls", [])
    return isinstance(calls, list) and all(
        isinstance(call, dict) and isinstance(call.get("function"), dict)
        and isinstance(call["function"].get("name"), str)
        and isinstance(call["function"].get("arguments"), str) for call in calls)


def run_records(path: Path, items: list[dict], *, aggregate_only: bool = False) -> list[dict]:
    """The run's record of every item, found by its id, in the items' order; ValueError when the run holds
    no record of an item or more than one, or a record of the wrong shape (well_formed), the message naming
    the first such item, or under --aggregate-only counting them alone (module docstring: Outputs). A record
    without a string id is no item's."""
    found: dict[str, list[dict]] = {}
    for record in read_jsonl(path):
        if isinstance(record.get("id"), str):
            found.setdefault(record["id"], []).append(record)
    bad = [item["id"] for item in items if len(found.get(item["id"], [])) != 1]
    if bad:
        raise ValueError(f"{path.as_posix()} does not hold exactly one record of {len(bad)} of the "
                         f"{len(items)} items" + ("" if aggregate_only else f" (the first: {bad[0]})"))
    chosen = [found[item["id"]][0] for item in items]
    shapeless = [item["id"] for item, record in zip(items, chosen, strict=True) if not well_formed(record)]
    if shapeless:
        raise ValueError(f"{path.as_posix()} holds {len(shapeless)} record(s) of the wrong shape (an ok "
                         "record needs an output object, its tool_calls a list of calls)"
                         + ("" if aggregate_only else f" (the first: {shapeless[0]})"))
    return chosen


def voted_records(items: list[dict], greedy: list[dict], samples: list[list[dict]], mode: str,
                  rules: bool) -> tuple[list[dict], list[list[dict]]]:
    """(the output records, every candidate's records as they voted: the greedy run's, then each sample
    run's, with the rules applied when on); module docstring: The records."""
    candidates: list[list[dict]] = [[] for _ in range(1 + len(samples))]
    out = []
    for position, item in enumerate(items):
        history = item["messages"]
        given = [greedy[position], *(run[position] for run in samples)]
        if rules:
            given = [ruled_record(record, history) for record in given]
        for run, record in zip(candidates, given, strict=True):
            run.append(record)
        if not samples or given[0].get("status") in NOT_ASKED:
            out.append(given[0])
        else:
            out.append(voted_record(given[0], given[1:], history, mode))
    return out, candidates


def successes(records: list[dict], items: list[dict]) -> dict[str, list[bool]]:
    """Per scorer, every record's success, in item order."""
    return {scorer: [bool(score["success"]) for score in score_items(records, items, scorer)]
            for scorer in SCORERS}


def totals_of(items: list[dict], out: list[dict], candidates: list[list[dict]], *, rules: bool) -> dict:
    """The totals of the module docstring: aggregates alone, nothing per item."""
    voted = successes(out, items)
    by_run = [successes(run, items) for run in candidates]
    totals: dict = {"items": len(items),
                    "statuses": dict(sorted(Counter(record["status"] for record in out).items())),
                    "voted": {scorer: sum(voted[scorer]) for scorer in SCORERS},
                    "greedy": {scorer: sum(by_run[0][scorer]) for scorer in SCORERS}}
    if len(candidates) > 1:
        totals["samples"] = {scorer: [sum(run[scorer]) for run in by_run[1:]] for scorer in SCORERS}
        totals["class_mean"] = {scorer: round(sum(class_mean(record, position, by_run, scorer)
                                                  for position, record in enumerate(out)), 6)
                                for scorer in SCORERS}
        totals["votes"] = tallies(out)
    if rules:
        changed = Counter(change["rule"] for run in candidates for record in run
                          for change in record.get("rules") or [])
        totals["rules"] = {name: changed[name] for name in RULES}
    return totals


def class_mean(record: dict, position: int, by_run: list[dict], scorer: str) -> float:
    """The mean success of the chosen class's samples for one item, the greedy record's when no sample voted
    or the item was not asked (module docstring: Totals)."""
    votes = record.get("votes")
    if votes is None or votes["representative"] == GREEDY_CANDIDATE:
        return float(by_run[0][scorer][position])
    members = [k for k, key in enumerate(votes["samples"]) if key == votes["chosen"]]
    return sum(by_run[1 + k][scorer][position] for k in members) / len(members)


def tallies(out: list[dict]) -> dict:
    """How the votes went (module docstring: Totals)."""
    count = Counter()
    for record in out:
        votes = record.get("votes")
        if votes is None:
            count["not_asked"] += 1
            continue
        if votes.get("errors"):
            count["with_errors"] += 1
            count["failed_candidates"] += len(votes["errors"])
        if votes["chosen"] is None:
            count["no_sample_voted"] += 1
            continue
        top = max(votes["counts"].values())
        tied = [key for key, n in votes["counts"].items() if n == top]
        if len(tied) > 1:
            count["tie_to_greedy" if votes["chosen"] == votes["greedy"] else "tie_to_first"] += 1
        if top == votes["k"]:
            count["unanimous"] += 1
        if votes["chosen"] != votes["greedy"]:
            count["chosen_not_greedy"] += 1
    return {name: count[name] for name in TALLIES}


def print_totals(totals: dict) -> None:
    """The totals, a line each, then all of them as one JSON line."""
    def scored(name: str, values: dict) -> None:
        print(f"  {name:<11}" + "  ".join(f"{scorer} {values[scorer]}" for scorer in SCORERS))

    print(f"  items      {totals['items']} ({', '.join(f'{k} {v}' for k, v in totals['statuses'].items())})")
    scored("voted", totals["voted"])
    if "class_mean" in totals:
        scored("class mean", totals["class_mean"])
    scored("greedy", totals["greedy"])
    if "samples" in totals:
        scored("samples", totals["samples"])
        print("  votes      " + ", ".join(f"{k} {v}" for k, v in totals["votes"].items()))
    if "rules" in totals:
        print("  rules      " + ", ".join(f"{k} {v}" for k, v in totals["rules"].items())
              + " (candidate answers changed)")
    print(canonical_json(totals))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    k = len(args.sample)
    if not k and not args.rules:
        parser.error("nothing to do: give --sample runs to vote over, or --rules")
    if args.vote_mode is not None and not k:
        parser.error("--vote-mode chooses the keys of a vote: it needs --sample runs")
    if args.aggregate_only and args.out is not None:
        parser.error("--aggregate-only prints the totals alone and writes nothing: no --out with it")
    if not args.aggregate_only and args.out is None:
        parser.error("--out names the records to write; --aggregate-only prints the totals alone")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    mode = args.vote_mode or FINAL
    how = None if args.aggregate_only else held_out_by(args.items, held_out_roots(args.run_copy))
    if how is not None:
        print(f"FAIL: {args.items.as_posix()} is a held-out set (by its {how}): its records are read for "
              "their totals alone (--aggregate-only)", file=sys.stderr)
        return 4
    roots = (Path.cwd(), REPO)
    try:
        refuse_test_sets(args.items, roots)
        locked = lock_required(args.items, roots)
        items = DevSet(args.items).open() if locked else read_jsonl(args.items)
        refuse_items(items, args.items, aggregate_only=args.aggregate_only)
        check_ids(items, aggregate_only=args.aggregate_only)
    except RefusedTestSet as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 4
    except LockError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 5
    except (JsonlError, OSError, ValueError) as exc:
        print(f"FAIL: malformed items: {exc}", file=sys.stderr)
        return 1
    selected = items if args.limit is None else items[:args.limit]
    try:
        greedy = run_records(args.greedy, selected, aggregate_only=args.aggregate_only)
        samples = [run_records(path, selected, aggregate_only=args.aggregate_only) for path in args.sample]
    except (JsonlError, OSError, ValueError) as exc:
        print(f"FAIL: malformed runs: {exc}", file=sys.stderr)
        return 1
    out, candidates = voted_records(selected, greedy, samples, mode, args.rules)
    totals = totals_of(selected, out, candidates, rules=args.rules)
    what = (f"{k} sample run(s), vote mode {mode}" if k else "no sample run") + (", the rules" if args.rules
                                                                                else "")
    print(f"{args.items.as_posix()}: {len(selected)} items; greedy {args.greedy.as_posix()}; {what}")
    if args.aggregate_only:
        print_totals(totals)
        return 0
    write_jsonl(out, args.out)
    manifest = {"items": args.items.as_posix(), "items_sha256": sha256_of(args.items), "items_locked": locked,
                "items_total": len(items), "items_run": len(selected), "limit": args.limit,
                "greedy": {"path": args.greedy.as_posix(), "sha256": sha256_of(args.greedy)},
                "samples": [{"path": path.as_posix(), "sha256": sha256_of(path)} for path in args.sample],
                "votes": k, **({"vote_mode": mode} if k else {}),
                **({"rules": {"version": RULES_VERSION, "names": list(RULES)}} if args.rules else {}),
                "statuses": totals["statuses"], "totals": totals, "date": local_date(),
                "git_commit": git_commit(REPO), "git_dirty": git_dirty(REPO)}
    path = manifest_path(args.out)
    path.write_text(canonical_json(manifest) + "\n", encoding="utf-8", newline="\n")
    print_totals(totals)
    print(f"{args.out}: {len(out)} records; manifest {path}")
    print(f"score: python scripts/score.py {args.out.as_posix()} --items {args.items.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

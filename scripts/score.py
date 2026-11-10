"""Score an evaluation run: the runner's records against the items (research_agent.eval.metrics).

    .venv/Scripts/python.exe scripts/score.py <results.jsonl> --items <items.jsonl>
        [--out <summary.json>] [--numbers] [--subset drafted|handwritten] [--scorer default|normalized]

--subset scores only the items drafted from teacher trajectories (source.request) or only the handwritten
ones (source.handwritten), with the records of those items; the records of the other items are left out.
The gold of a drafted item is the teacher's own turn, so the teacher (B2) is read on the handwritten subset.

--scorer (M7, docs/m7-本地提升计划.md WP1) names the scorer (research_agent.eval.metrics, Scorers): default,
today's, or normalized, which adds S1 to S3 (a date_to on the prompt's today left out, a query's hyphens read
as spaces, a count the request does not give at the tool's default or the gold value). A file the normalized
scorer made records it as "scorer": "normalized", its first key; a default one has no such key and is byte
for byte what it was before --scorer. The default --out gets -<scorer> after the stem for a scorer other than
the default, after the subset's (runs/eval/scores/<stem>-handwritten-normalized.json). Before anything is read
or written, an --out the scorer may not take is refused (exit 2): a registered result whose name is another
scorer's (results/dev-<name>-normalized.json for the default scorer: research_agent.eval.report.scored_name,
results/ of the current folder or of the repository root), and a file, wherever it is, that another scorer
made (one without "scorer" is the default scorer's), so that a summary never replaces one of another scorer.
A file of the same scorer, or one that is no JSON object, is written over as before.

Reads the records written by research_agent.eval.runner (one per item, any order) and the items they were run
on, scores every record (metrics.score_item) and writes one JSON file, {"summary": metrics.summarize(...),
"scores": [per-item scores in the records' order]}, with a fixed key order and LF line ends, so two runs over
the same files write byte-identical output. --out defaults to runs/eval/scores/<stem of results.jsonl>.json
under the repository root (the folder holding pyproject.toml), whatever the current folder: runs/ is ignored
by git, and results/ is reserved for the summaries whose numbers are registered. A given --out is taken as
written (relative to the current folder). It prints the headline rates.

--numbers then rebuilds results/numbers.json (relative to the current folder) with scripts/build_report.py's
build (research_agent.eval.report), from every registered result present in results/, the summary just
written among them. --out must then be results/dev-<name>.json, the name of a registered development
summary, whose numbers are eval.dev-<name>.<metric> (the name with every character outside [A-Za-z0-9_-]
replaced by "-"):

    n                 number of items                      e.g. "150"
    success           overall single-item success          e.g. "87.3%"
    success.<type>    success per item type (call, ...)     only types present
    success.<lang>    success per language (zh, en)         only languages present
    json_valid, tool_choice, exact_args, query_strict, false_call, think_leak, clarify   rates, "12.0%"
    param_f1          macro per-parameter F1                e.g. "0.912"

Rates are percentages with one decimal, F1 three decimals, as README.md would quote them; a metric with no
applicable item is left out. Scoring again under the same name updates its numbers, and every other
registered result is read again as it is now; numbers.json holds nothing else. Without --out
results/dev-<name>.json, --numbers is a usage error and nothing is written. git must not ignore a result
either (write_numbers refuses a source that a clean checkout would not have). A summary the normalized
scorer made gives its numbers as eval.dev-<name>-normalized.<metric> (eval.dev-<name>-normalized.* once
when <name> ends in -normalized already), never a default summary's keys; a default summary under a name
ending in -normalized is refused before anything is written (exit 2, --scorer above).

Which items: as in scripts/run_eval.py, test material is refused before anything is read (exit 4), since only
the phase-4 test runner scores it: a locked test set (data/eval/test.jsonl, data/eval/e2e-test.jsonl, however
the path is spelled, or a copy with a test set's digest) and the files of the test set's handwritten items
with their gold turns (data/eval/handwritten.jsonl, data/eval/handwritten-specs.jsonl; by location); and,
once read, before anything is scored, any items file that holds an item of split "test" or with an id of the
test split (eval.lock.refuse_test_material: a subset of a test set, a copy with other line ends).

Exit 0 when scored; 1 on malformed input (a file that is not JSONL objects, a record or item of the wrong
shape, an unknown item type or status); 2 when the records' ids and the items' ids differ (missing, extra or
repeated), or on a usage error (argparse's exit, --numbers without --out results/dev-<name>.json and an --out
the scorer may not take included; nothing is written); 3
when --numbers could not rebuild the numbers (a registered result that is not of its kind, two that give the
same key, a source git ignores; the summary file is written all the same); 4 when the items are test material
(nothing is written).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research_agent.eval.lock import (
    JsonlError,
    RefusedTestSet,
    read_jsonl,
    refuse_test_material,
    refuse_test_sets,
)
from research_agent.eval.metrics import DEFAULT_SCORER, SCORERS, IdMismatch, score_items, summarize
from research_agent.eval.numbers import NUMBERS
from research_agent.eval.report import SUBSETS, ReportError, in_subset, kind_of, percent, scored_name, write

DEV_SUMMARY = "dev"  # the kind of results/dev-<name>.json in research_agent.eval.report's registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score runner records against evaluation items and write "
                                                 "the summary with the per-item scores.")
    parser.add_argument("results", type=Path, help="the runner's records, JSONL")
    parser.add_argument("--items", type=Path, required=True, help="the items the records were run on, JSONL")
    parser.add_argument("--out", type=Path,
                        help="the summary file (default: runs/eval/scores/<results stem>.json under the "
                             "repository root)")
    parser.add_argument("--numbers", action="store_true",
                        help="rebuild results/numbers.json from the registered results, this summary among "
                             "them (keys eval.dev-<name>.*); needs --out results/dev-<name>.json")
    parser.add_argument("--subset", choices=SUBSETS,
                        help="score only the drafted items (source.request) or only the handwritten ones "
                             "(source.handwritten); the default --out gets -<subset> after the stem")
    parser.add_argument("--scorer", choices=SCORERS, default=DEFAULT_SCORER,
                        help="the scorer (research_agent.eval.metrics: default, or normalized with S1-S3); "
                             "another than the default is recorded in the file, and the default --out gets "
                             "-<scorer> after the stem")
    return parser


def numbered(out: Path | None) -> bool:
    """Whether out is results/dev-<name>.json (relative to the current folder), a registered development
    summary: the only summary --numbers rebuilds the numbers with."""
    if out is None or out.resolve().parent != NUMBERS.parent.resolve():
        return False
    kind = kind_of(out.name)
    return kind is not None and kind.key == DEV_SUMMARY


def find_root(start: Path) -> Path:
    """The nearest folder at or above start that holds pyproject.toml: the repository root."""
    for folder in (start, *start.parents):
        if (folder / "pyproject.toml").is_file():
            return folder
    raise FileNotFoundError(f"no pyproject.toml at or above {start}")


ROOT = find_root(Path(__file__).resolve().parent)


def refusal(out: Path, scorer: str) -> str | None:
    """Why the summary the scorer makes may not go to out (module docstring: --scorer), None when it may: out
    is a registered result of a kind a scorer makes (results/<file> of the current folder or of the repository
    root; research_agent.eval.report, Scorers) whose name is another scorer's, or out holds a file another
    scorer made (one without "scorer" is the default scorer's; a file that is no JSON object records none)."""
    registered = out.resolve().parent in {NUMBERS.parent.resolve(), (ROOT / NUMBERS.parent).resolve()}
    kind = kind_of(out.name) if registered else None
    if kind is not None and kind.scored:
        try:
            scored_name(kind.name(kind.pattern.fullmatch(out.name)), {"scorer": scorer})
        except ValueError as exc:
            return f"--out {out}: {exc}; a registered summary goes under a name of its own scorer's"
    try:
        held = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else None
    except (OSError, ValueError):  # a UnicodeDecodeError is one
        held = None
    made = held.get("scorer", DEFAULT_SCORER) if isinstance(held, dict) else scorer
    if made != scorer:
        return f"--out {out} holds a summary the {made} scorer made, and this one is the {scorer} scorer's"
    return None


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        # a console whose code page lacks the Chinese error-class names prints "?" instead of stopping the
        # script after the summary is written and before --numbers registers it
        sys.stdout.reconfigure(errors="replace")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.numbers and not numbered(args.out):
        parser.error(f"--numbers needs --out {NUMBERS.parent.as_posix()}/dev-<name>.json: the numbers are "
                     "rebuilt from the registered results only (research_agent.eval.report), and a "
                     "development summary is registered under that name")
    suffix = f"-{args.subset}" if args.subset else ""
    if args.scorer != DEFAULT_SCORER:
        suffix += f"-{args.scorer}"
    out = args.out or ROOT / "runs" / "eval" / "scores" / f"{args.results.stem}{suffix}.json"
    refused = refusal(out, args.scorer)
    if refused is not None:
        print(f"FAIL: {refused}", file=sys.stderr)
        return 2
    try:
        refuse_test_sets(args.items, (Path.cwd(), ROOT))
        results = read_jsonl(args.results)
        items = read_jsonl(args.items)
        refuse_test_material(items, args.items)
        if args.subset:
            items = [item for item in items if in_subset(item, args.subset)]
            kept = {item.get("id") for item in items}
            results = [record for record in results if record.get("id") in kept]
        scores = score_items(results, items, scorer=args.scorer)
        summary = summarize(scores, items)
    except RefusedTestSet as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 4
    except IdMismatch as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    except (JsonlError, OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        print(f"FAIL: malformed input: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    scored = {} if args.scorer == DEFAULT_SCORER else {"scorer": args.scorer}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({**scored, "summary": summary, "scores": scores}, ensure_ascii=False, indent=2)
                   + "\n", encoding="utf-8", newline="\n")
    rate = summary["success_rate"]
    shown = f" (scorer {args.scorer})" if scored else ""
    print(f"{out}: {summary['n']} items, success {percent(rate) if rate is not None else '-'}{shown}")
    for type_, cell in summary["by_type"].items():
        if cell["n"]:
            print(f"  {type_:<10} n {cell['n']:>4}  success {percent(cell['success_rate'])}")
    classes = ", ".join(f"{k} {v}" for k, v in summary["error_classes"].items() if v)
    print(f"  errors: {classes or 'none'}")
    if args.numbers:
        try:
            report = write(Path("."))
        except (ReportError, ValueError, OSError) as exc:
            print(f"numbers not rebuilt: {exc}", file=sys.stderr)
            return 3
        print(f"{NUMBERS.as_posix()}: {len(report['numbers'])} numbers, {len(report['sources'])} source(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

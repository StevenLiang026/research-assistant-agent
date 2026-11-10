"""Draft the evaluation sets from synthetic-generator runs (research_agent.eval.items explains the items).

    .venv/Scripts/python.exe scripts/build_eval_items.py runs/m2/synth/<run> [runs/m2/synth/<run> ...]
        [--out-dir data/eval] [--seed 42] [--share 0.7] [--include-unvalidated | --validated-only]
        [--allow-placeholders]

Reads <run>/manifest.json first: a run generated with --placeholders (config.placeholders true; the
generator's stand-ins took the place of the owner's check_call, decide and extract_section) is for debugging
only and never enters the evaluation sets (docs/modules/接口.md), so it is refused unless
--allow-placeholders is given, in which case it is drafted as usual and one stdout line notes that it is a
placeholder run. Then it reads <run>/index.jsonl and the transcripts it names, drafts one item per accepted
step of the dev and test requests, draws the share of the plan's quotas (QUOTAS_DEV, QUOTAS_TEST) per (type,
language) stratum and writes <out-dir>/dev-draft.jsonl and <out-dir>/test-draft.jsonl as canonical JSON
lines, so two runs over the same input write byte-identical files. Several run directories may be given (the
full generation wrote dev and test into runs of their own): every one is checked first (index, manifest,
placeholder guard), the candidates are the union of their drafts (item ids stay unique because request ids
are; a repeated id is malformed input) and the rest is as for one run. It prints how many index records were
drafted and why the others were left out, per run and in total, the candidates by split, type and language
and, per stratum, the quota, what was available, what was taken and the shortfall. A shortfall is not an
error: the remaining share is written by hand anyway, and the pool grows with every generator run.
Requests without a verdict (the owner's check_trajectory not run yet) are drafted unless --validated-only is
given, and only when their turn ended answered or as a clarification: a turn that stopped as a dead loop, at
the step limit, over budget, on an invalid output or in a context overflow is left out, and the "left out"
line counts it under that status.

Exit 1 on malformed input: a run directory without manifest.json (the generator writes one with every run) or
with a manifest that is not a JSON object or lacks a true-or-false config.placeholders, an index line that is
not a JSON object, a transcript that is missing, has keys Transcript does not have or was recorded under
another tool schema, a history that does not fit its steps, repeated item ids. Exit 2 when a run directory
has no index.jsonl. Exit 3 when a run was generated with --placeholders and --allow-placeholders was not
given; nothing is written.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from research_agent.agent.types import HarnessError
from research_agent.data.synth import MANIFEST_FILE
from research_agent.eval.items import (
    DRAFT_FILES,
    INDEX_FILE,
    LANGS,
    QUOTAS_DEV,
    QUOTAS_TEST,
    SKIP_REASONS,
    draft_items,
    scaled,
    select,
    skip_reason,
    summarize,
)
from research_agent.eval.lock import JsonlError, read_jsonl, write_jsonl

QUOTAS = {"dev": QUOTAS_DEV, "test": QUOTAS_TEST}
MALFORMED = (JsonlError, FileNotFoundError, TypeError, HarnessError, ValueError)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draft dev-draft.jsonl and test-draft.jsonl from a generator run: one evaluation item "
                    "per accepted step, a stratified draw of the plan's quotas. Shortfalls are reported, not "
                    "failed on.")
    parser.add_argument("run_dirs", nargs="+", type=Path,
                        help="generator runs, e.g. runs/m2/synth/<run>, each holding index.jsonl and "
                             "transcripts/; their drafts are pooled")
    parser.add_argument("--out-dir", type=Path, default=Path("data/eval"),
                        help="where the two draft files go (default: data/eval)")
    parser.add_argument("--seed", type=int, default=42, help="seed of the stratified draw (default: 42)")
    parser.add_argument("--share", type=float, default=0.7,
                        help="the share of each quota the drafter fills; the rest is written by hand "
                             "(default: 0.7)")
    parser.add_argument("--include-unvalidated", dest="include_unvalidated", action="store_true",
                        default=True, help="draft requests without a verdict too (the default)")
    parser.add_argument("--validated-only", dest="include_unvalidated", action="store_false",
                        help="draft only requests whose verdict accepted them")
    parser.add_argument("--allow-placeholders", action="store_true",
                        help="draft a run generated with --placeholders anyway (its drafts are for debugging "
                             "only and never enter the evaluation sets); without it such a run is refused "
                             "with exit 3")
    return parser


def placeholder_run(run_dir: Path) -> bool:
    """Whether the run's manifest records config.placeholders true (research_agent.data.synth writes it).

    FileNotFoundError naming the manifest when the run has none; ValueError when the manifest is not JSON;
    TypeError when it is not a JSON object or its config.placeholders is not true or false.
    """
    path = run_dir / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: the generator writes it with every run")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise TypeError(f"{path} is not a JSON object")
    config = manifest.get("config")
    flag = config.get("placeholders") if isinstance(config, dict) else None
    if not isinstance(flag, bool):
        raise TypeError(f"{path}: config.placeholders must be true or false, not {flag!r:.80}")
    return flag


def index_line(rows: list[dict], *, splits: tuple[str, ...], include_unvalidated: bool,
               run_dir: Path | None = None) -> str:
    """How many index records were drafted and how many were left out for each reason, of one run (named)
    or of all of them (run_dir None)."""
    reasons = Counter(skip_reason(row, splits=splits, include_unvalidated=include_unvalidated)
                      for row in rows)
    skipped = ", ".join(f"{reason} {reasons[reason]}" for reason in SKIP_REASONS if reasons[reason])
    label = "index" if run_dir is None else f"index {run_dir}"
    return f"{label}: {len(rows)} requests, {reasons[None]} drafted; left out: {skipped or 'none'}"


def summary_lines(summary: dict) -> list[str]:
    """One line per split and type with the count per language and the total, after a header."""
    langs = list(LANGS)
    for types in summary.values():
        for counts in types.values():
            langs += [lang for lang in counts if lang not in langs]
    lines = [f"{'split':<6} {'type':<10}" + "".join(f" {lang:>6}" for lang in langs) + f" {'total':>6}"]
    for split, types in summary.items():
        for type_, counts in types.items():
            cells = "".join(f" {counts.get(lang, 0):>6}" for lang in langs)
            lines.append(f"{split:<6} {type_:<10}{cells} {sum(counts.values()):>6}")
    return lines


def report_lines(split: str, report: dict) -> list[str]:
    """One line per stratum: quota, available, taken and the shortfall, marked when there is one."""
    lines = []
    for type_, per_lang in report.items():
        for lang, counts in per_lang.items():
            mark = "  <- shortfall" if counts["shortfall"] else ""
            lines.append(f"{split:<6} {type_:<10} {lang:<3} quota {counts['quota']:>3}  available "
                         f"{counts['available']:>3}  taken {counts['taken']:>3}  shortfall "
                         f"{counts['shortfall']:>3}{mark}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    for run_dir in args.run_dirs:
        if not (run_dir / INDEX_FILE).is_file():
            print(f"{run_dir} has no {INDEX_FILE}", file=sys.stderr)
            return 2
    try:
        placeholders = [run_dir for run_dir in args.run_dirs if placeholder_run(run_dir)]
        if placeholders and not args.allow_placeholders:
            for run_dir in placeholders:
                print(f"{run_dir}: this run was generated with --placeholders, so its drafts are for "
                      "debugging only; nothing written (--allow-placeholders drafts them anyway)",
                      file=sys.stderr)
            return 3
        for run_dir in placeholders:
            print(f"note: {run_dir} is a placeholder run (generated with --placeholders); its drafts "
                  "are for debugging only")
        options = {"splits": tuple(QUOTAS), "include_unvalidated": args.include_unvalidated}
        rows_by_run = {run_dir: read_jsonl(run_dir / INDEX_FILE) for run_dir in args.run_dirs}
        items = [item for run_dir in args.run_dirs for item in draft_items(run_dir, **options)]
        source = (str(args.run_dirs[0]) if len(args.run_dirs) == 1
                  else f"{len(args.run_dirs)} runs ({', '.join(str(d) for d in args.run_dirs)})")
        print(f"candidates: {len(items)} items drafted from {source}")
        if len(args.run_dirs) > 1:
            for run_dir, rows in rows_by_run.items():
                print(index_line(rows, run_dir=run_dir, **options))
        print(index_line([row for rows in rows_by_run.values() for row in rows], **options))
        for line in summary_lines(summarize(items)):
            print(line)
        for split, quotas in QUOTAS.items():
            pool = [item for item in items if item["split"] == split]
            chosen, report = select(pool, scaled(quotas, args.share), seed=args.seed)
            path = args.out_dir / DRAFT_FILES[split]
            write_jsonl(chosen, path)
            print()
            for line in report_lines(split, report):
                print(line)
            short = sum(c["shortfall"] for per_lang in report.values() for c in per_lang.values())
            print(f"{path}: {len(chosen)} items written (share {args.share} of {sum(quotas.values())}, "
                  f"seed {args.seed}); total shortfall {short}")
    except MALFORMED as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Check training files for 13-gram overlap with the BFCL and ACEBench questions (research_agent.data.overlap
explains the grams and which benchmark files and fields are read).

    uv run --no-sync python scripts/check_benchmark_overlap.py data/raw/public/*.jsonl
        data/raw/synth/synth.jsonl
        [--out data/synth/benchmark-overlap.json]
        [--bfcl-dir <the bfcl_eval/data folder>]
        [--acebench-dir runs/third_party/ACEBench/data_all]

Without --bfcl-dir the bfcl_eval/data folder is looked up under .venv-bfcl in either virtualenv layout
(Windows: Lib/site-packages; Linux and macOS: lib/python3.X/site-packages), with the names as the file
system has them; more than one match is an error (exit 2), so pass --bfcl-dir then. Glob patterns are
expanded here (PowerShell expands none). For every training file the report lists the
records whose user messages share a 13-gram with a benchmark question: the record's index in the file (as
research_agent.train.samples.load_records numbers them, from 0), its "id" when it has one, and the matching
benchmark entries (benchmark, file, entry id, one shared gram), with the file's SHA-256 and the counts; the
benchmark files read are listed with their SHA-256, entry and question counts. scripts/train.py skips the
listed records when the report exists (it refuses a report whose file hashes no longer match). One line per
training file is printed. Overlap is reported, not failed on.

Exit 0 on success, overlaps or not; 1 on a training file that is malformed or missing; 2 when a benchmark
folder is absent or holds none of the files the check reads (both lie outside the repository's tracked files:
the BFCL environment and runs/third_party/ACEBench), or no training file matches a pattern. Nothing is
written on exit 1 or 2.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

from research_agent.data.overlap import BenchmarkIndex, BenchmarkMissing, N, check_file
from research_agent.train.samples import RecordError

OUT = Path("data/synth/benchmark-overlap.json")
BFCL_VENV = Path(".venv-bfcl")
# the bfcl_eval data folder inside a virtualenv: the Windows layout, then the POSIX one
BFCL_LAYOUTS = ("Lib/site-packages/bfcl_eval/data", "lib/python3*/site-packages/bfcl_eval/data")
ACEBENCH_DIR = Path("runs/third_party/ACEBench/data_all")


def find_bfcl_dirs(venv: Path = BFCL_VENV) -> list[Path]:
    """The bfcl_eval data folders under venv in either layout (BFCL_LAYOUTS), as the file system names them,
    sorted; [] when there is none."""
    found = {path for pattern in BFCL_LAYOUTS for path in venv.glob(pattern) if path.is_dir()}
    return sorted(found)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="13-gram overlap of training files with BFCL and ACEBench.")
    parser.add_argument("data", nargs="+", help="training files (JSONL or JSON); glob patterns are expanded")
    parser.add_argument("--out", type=Path, default=OUT, help=f"the report (default: {OUT.as_posix()})")
    parser.add_argument("--bfcl-dir", type=Path, default=None,
                        help=f"the bfcl_eval data folder (default: found under {BFCL_VENV.as_posix()} in the "
                             "Windows or the POSIX virtualenv layout)")
    parser.add_argument("--acebench-dir", type=Path, default=ACEBENCH_DIR,
                        help=f"ACEBench's data_all folder (default: {ACEBENCH_DIR.as_posix()})")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files: list[Path] = []
    for pattern in args.data:
        matches = sorted(glob.glob(pattern)) if glob.has_magic(pattern) else [pattern]
        if not matches:
            print(f"no training file matches {pattern!r}", file=sys.stderr)
            return 2
        files += [Path(m) for m in matches]
    bfcl_dir = args.bfcl_dir
    if bfcl_dir is None:
        found = find_bfcl_dirs()
        if len(found) > 1:
            print(f"several bfcl_eval data folders under {BFCL_VENV.as_posix()}: "
                  f"{', '.join(p.as_posix() for p in found)}; pass --bfcl-dir", file=sys.stderr)
            return 2
        # none found: the Windows layout's path, which BenchmarkIndex reports as missing below
        bfcl_dir = found[0] if found else BFCL_VENV / BFCL_LAYOUTS[0]
    try:
        index = BenchmarkIndex.build(bfcl_dir, args.acebench_dir)
    except BenchmarkMissing as exc:
        print(f"benchmarks missing: {exc}", file=sys.stderr)
        return 2
    try:
        reports = [check_file(path, index) for path in files]
    except (RecordError, OSError, ValueError) as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    report = {"n": N, "method": "of normalize_text(text).lower(), word 13-grams over the Latin-script words "
                                "(runs of non-space, non-CJK characters) and character 13-grams over the "
                                "non-space characters for the windows holding a CJK character; user messages "
                                "of each record against user turns of each benchmark question, a shared gram "
                                "of either kind being an overlap",
              "benchmarks": index.files, "benchmark_grams": len(index.grams), "files": reports,
              "total": {"records": sum(r["records"] for r in reports),
                        "overlapping": sum(r["overlapping"] for r in reports)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                        newline="\n")
    by_benchmark = {}
    for f in index.files:
        by_benchmark.setdefault(f["benchmark"], [0, 0, 0])
        by_benchmark[f["benchmark"]][0] += 1
        by_benchmark[f["benchmark"]][1] += f["entries"]
        by_benchmark[f["benchmark"]][2] += f["questions"]
    for name, (count, entries, questions) in by_benchmark.items():
        print(f"benchmark {name}: {count} files, {entries} entries, {questions} user turns")
    print(f"benchmark grams: {len(index.grams)}")
    for r in reports:
        hit_by = {}
        for rec in r["records_overlapping"]:
            for b in {m["benchmark"] for m in rec["matches"]}:
                hit_by[b] = hit_by.get(b, 0) + 1
        detail = ", ".join(f"{b} {n}" for b, n in sorted(hit_by.items()))
        print(f"{r['path']}: {r['overlapping']} of {r['records']} records overlap" +
              (f" ({detail})" if detail else ""))
    print(f"total: {report['total']['overlapping']} of {report['total']['records']} records overlap; "
          f"report {args.out.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

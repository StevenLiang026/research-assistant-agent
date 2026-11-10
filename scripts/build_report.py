"""Build results/numbers.json, the numbers README.md quotes, from the registered result files: the only writer
of that file (research_agent.eval.report; docs/plan.md, 数字对账).

    .venv/Scripts/python.exe scripts/build_report.py [--check]

Run from the repository root. Every registered result present in results/ is read (research_agent.eval.report
lists the file names and the keys each kind gives: the development summaries of scripts/score.py, the
phase-4 results of scripts/run_test.py, the checkpoint selections, BFCL, the backend comparisons, M4's
routing configuration, fault table and ACEBench results, and M8's checkpoint decision), and
results/numbers.json is written with the SHA-256 of each (research_agent.eval.numbers.write_numbers, which
refuses a source git ignores, since a clean checkout would not have it). The numbers depend on those files
alone, so rebuilding from the same files gives the same bytes; scripts/score.py --numbers runs the same build
after writing its summary. README.md's marker blocks are then checked by scripts/check_readme_numbers.py (the
pre-commit hook).

--check builds the same numbers without writing anything and compares them with results/numbers.json byte for
byte: a result changed or added since the last build shows as a difference.

Exit 0 when the numbers are written (with --check: when results/numbers.json is up to date); 1 with --check
when results/numbers.json is missing or differs from a rebuild; 2 on a usage error; 3 when the numbers cannot
be built (a registered file that is not valid JSON or not of its kind, two files that give the same key, a
source git ignores); nothing is written then.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from research_agent.eval.numbers import NUMBERS
from research_agent.eval.report import ReportError, write


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build results/numbers.json from the registered result files.")
    ap.add_argument("--check", action="store_true",
                    help="compare a rebuild with results/numbers.json instead of writing it")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    shown = NUMBERS.as_posix()
    try:
        if args.check:
            with tempfile.TemporaryDirectory() as folder:
                fresh = Path(folder) / NUMBERS.name
                report = write(Path("."), fresh)
                rebuilt = fresh.read_bytes()
        else:
            report = write(Path("."))
    except (ReportError, ValueError, OSError) as exc:
        print(f"FAIL: {exc}")
        return 3
    counts = f"{len(report['numbers'])} numbers from {len(report['sources'])} result file(s)"
    if not args.check:
        print(f"{shown}: {counts}")
        return 0
    if not NUMBERS.is_file():
        print(f"{shown} is missing; a build gives {counts}: run scripts/build_report.py")
        return 1
    if NUMBERS.read_bytes() != rebuilt:
        print(f"{shown} differs from a rebuild ({counts}): run scripts/build_report.py")
        return 1
    print(f"{shown} is up to date: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

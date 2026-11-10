"""The README number check of the public copy: every result number README.md quotes is the report's.

    python scripts/check_public_numbers.py [--readme README.md] [--numbers results/numbers.json] \\
        [--provenance docs/provenance.md]

The public repository is an export of a private one (docs/provenance.md). It holds the report,
results/numbers.json, and the aggregate result files; the per-item results, the results of the test-set
openings and the test sets' access logs stay in the private repository. So this is the private check
(scripts/check_readme_numbers.py, research_agent.eval.numbers) without what is not here:

- every marker block <!-- num:KEY -->VALUE<!-- /num --> of README.md outside code is well formed, and VALUE is
  the report's numbers[KEY], compared as strings (research_agent.eval.numbers.markers and unclosed);
- the counts of test-set openings, test_set.opens and test_set_v2.opens, which the private check counts in the
  access logs, must equal the counts docs/provenance.md records in the same marker form;
- every source of the report that is here still has the SHA-256 the report recorded; a source that is not here
  is one the export leaves out, and is counted, not failed (the export checks, when it builds this copy, that
  every absent source is one its rules leave out).

Prints one line per problem, then a summary line; exit 1 when there is a problem, 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research_agent.eval.lock import sha256_of
from research_agent.eval.numbers import TEST_SET_OPENS, TEST_SET_V2_OPENS, markers, unclosed

README = Path("README.md")
NUMBERS = Path("results/numbers.json")
PROVENANCE = Path("docs/provenance.md")
OPENS = (TEST_SET_OPENS, TEST_SET_V2_OPENS)


def check(readme: Path = README, numbers: Path = NUMBERS, provenance: Path = PROVENANCE,
          root: Path = Path(".")) -> tuple[list[str], dict[str, int]]:
    """(problems, counts): one line per problem, and how many markers, opens, present and absent sources."""
    root = Path(root)
    text = (root / readme).read_text(encoding="utf-8")
    found = markers(text)
    problems = [f"{readme.as_posix()}:{line}: malformed marker block" for line in unclosed(text)]
    counts = {"markers": 0, "opens": 0, "sources_present": 0, "sources_absent": 0}

    recorded: dict[str, str] = {}
    if (root / provenance).exists():
        recorded = {key: value for key, value, _ in markers((root / provenance).read_text(encoding="utf-8"))
                    if key in OPENS}
    for key, value, line in found:
        if key not in OPENS:
            continue
        counts["opens"] += 1
        if key not in recorded:
            problems.append(f"{readme.as_posix()}:{line}: {key} is not recorded in {provenance.as_posix()}")
        elif value != recorded[key]:
            problems.append(f"{readme.as_posix()}:{line}: {key} is {value!r}, {provenance.as_posix()} "
                            f"records {recorded[key]!r}")

    quoted = [(key, value, line) for key, value, line in found if key not in OPENS]
    report_path = root / numbers
    if not report_path.exists():
        if quoted:
            problems.append(f"{readme.as_posix()}: {len(quoted)} marker(s) but no report at "
                            f"{numbers.as_posix()}")
        return problems, counts
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return problems + [f"{numbers.as_posix()}: not valid JSON ({exc})"], counts
    values = report.get("numbers", {})
    for key, value, line in quoted:
        counts["markers"] += 1
        if key not in values:
            problems.append(f"{readme.as_posix()}:{line}: {key} is not in {numbers.as_posix()}")
        elif value != values[key]:
            problems.append(f"{readme.as_posix()}:{line}: {key} is {value!r}, the report says "
                            f"{values[key]!r}")
    for source, digest in report.get("sources", {}).items():
        path = root / source
        if not path.exists():
            counts["sources_absent"] += 1
            continue
        counts["sources_present"] += 1
        if sha256_of(path) != digest:
            problems.append(f"{numbers.as_posix()}: source {source} changed since the report was built")
    return problems, counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check README.md's result numbers against the report.")
    ap.add_argument("--readme", type=Path, default=README)
    ap.add_argument("--numbers", type=Path, default=NUMBERS)
    ap.add_argument("--provenance", type=Path, default=PROVENANCE)
    args = ap.parse_args(argv)
    problems, counts = check(args.readme, args.numbers, args.provenance)
    for problem in problems:
        print(problem)
    print(f"{counts['markers']} number(s) and {counts['opens']} opening count(s) checked; "
          f"{counts['sources_present']} source(s) of the report here, hash-checked; "
          f"{counts['sources_absent']} kept in the private repository; {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

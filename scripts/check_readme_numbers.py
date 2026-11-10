"""Pre-commit hook: every result number README.md quotes must be the report's (research_agent.eval.numbers).

Usage: python scripts/check_readme_numbers.py [--readme README.md] [--numbers results/numbers.json]
       [--access-log data/eval/test-access.jsonl]
Run from the repository root (pre-commit does): the report, its sources and the access log must also be in
git's index, staged counts. Prints one line per problem; exit 1 if there is any.
"""

from __future__ import annotations

import argparse
import sys

from research_agent.eval.lock import TEST_ACCESS_LOG
from research_agent.eval.numbers import NUMBERS, README, check, git_tracked


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check README.md's result numbers against the report.")
    ap.add_argument("--readme", default=str(README))
    ap.add_argument("--numbers", default=str(NUMBERS))
    ap.add_argument("--access-log", default=str(TEST_ACCESS_LOG))
    args = ap.parse_args(argv)
    problems = check(args.readme, args.numbers, args.access_log, tracked=git_tracked)
    for problem in problems:
        print(problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

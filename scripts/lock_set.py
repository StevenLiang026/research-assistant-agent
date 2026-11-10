"""Lock an evaluation set: write <file>.sha256 and print its digest in the sha256sum format.

    .venv/Scripts/python.exe scripts/lock_set.py data/eval/test.jsonl           # write, print "<hex>  <name>"
    .venv/Scripts/python.exe scripts/lock_set.py data/eval/test.jsonl --check   # verify; exit 1 on mismatch

The locked sets live in data/eval and are committed with their locks and the test set's access log:
data/eval/dev.jsonl with dev.jsonl.sha256, data/eval/test.jsonl with test.jsonl.sha256, and
data/eval/test-access.jsonl (research_agent.eval.lock describes the lock and the log). The lock is the
two-space text form, so `sha256sum -c test.jsonl.sha256` run inside data/eval checks it as well.

An existing lock that no longer matches the file is kept unless --force is given: a test set edited after
its lock is exactly what the lock is there to make visible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from research_agent.eval.lock import LockError, check_lock, lock_path, read_lock, sha256_of, write_lock


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write or check the SHA-256 lock of an evaluation set. The locked sets are "
                    "data/eval/dev.jsonl and data/eval/test.jsonl, committed with their .sha256 locks and "
                    "the test set's access log data/eval/test-access.jsonl.")
    parser.add_argument("path", type=Path,
                        help="the set (JSONL), e.g. data/eval/test.jsonl; the lock is written next to it")
    parser.add_argument("--check", action="store_true", help="verify the existing lock instead of writing")
    parser.add_argument("--force", action="store_true", help="replace a lock that no longer matches the file")
    args = parser.parse_args(argv)
    if not args.path.is_file():
        parser.error(f"{args.path} is not a file")
    if args.check:
        try:
            digest = check_lock(args.path)
        except LockError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
        print(f"{digest}  {args.path.name}")
        return 0
    lock = lock_path(args.path)
    if lock.exists() and not args.force:
        try:
            recorded = read_lock(args.path)
        except LockError:
            recorded = None
        if recorded != sha256_of(args.path):
            print(f"{lock.name} exists and does not match {args.path.name}; pass --force to replace it",
                  file=sys.stderr)
            return 1
    print(f"{write_lock(args.path)}  {args.path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

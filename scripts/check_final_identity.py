"""Check that a run was M7's final system, unchanged (docs/m9-计划.md WP3, WP4 and 验证): the identity the
run recorded, held against the frozen record.

    .venv/Scripts/python.exe scripts/check_final_identity.py <report.json | session.json>

The file is either the report.json of scripts/run_e2e.py (its manifest: the built system's info with the
prompt and schema hashes, and server_props) or the session.json of a run of scripts/ask.py (info and props).
The frozen system is loaded and checked first (research_agent.live.frozen.load_frozen: the code, the frozen
record, the session configuration, the GGUFs' SHA-256); then identity_problems holds the run's info and server
props against it. Either kind must also have run on a clean working tree (its git_dirty false:
live.frozen.tree_problems), and an end-to-end report must have measured its steps against final's training
length (TRAINING_LENGTH, 5120): its training_length, a context check and a count of the steps over it.

Prints OK or every problem. Exit 0 when the run was the frozen system; 1 otherwise: a problem, a frozen system
that does not hold, or a file that is neither kind or cannot be read.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from research_agent.live.frozen import (
    TRAINING_LENGTH,
    FrozenMismatch,
    FrozenSystem,
    identity_problems,
    load_frozen,
    tree_problems,
)

REPORT, SESSION = "run_e2e report", "ask session"


def recorded(data: object) -> tuple[str, dict, dict | None, list[str]]:
    """(kind, info, server props, the problems of its own) of a run's file (module docstring); ValueError for
    a file that is neither kind."""
    if isinstance(data, dict) and isinstance(data.get("manifest"), dict):
        manifest = data["manifest"]
        problems = tree_problems(manifest.get("git_dirty"))
        if manifest.get("training_length") != TRAINING_LENGTH:
            problems.append(f"training_length: the report measured against "
                            f"{manifest.get('training_length')!r}, final's training length is "
                            f"{TRAINING_LENGTH}")
        if manifest.get("context_check") is None or manifest.get("steps_over_training_length") is None:
            problems.append("steps_over_training_length: the run had no context check, so no step was "
                            "measured against the training length")
        return REPORT, manifest, manifest.get("server_props"), problems
    if isinstance(data, dict) and isinstance(data.get("info"), dict):
        return SESSION, data["info"], data.get("props"), tree_problems(data.get("git_dirty"))
    raise ValueError("neither a report of scripts/run_e2e.py (manifest) nor a session.json of scripts/ask.py "
                     "(info)")


def main(argv: list[str] | None = None, *, load: Callable[[], FrozenSystem] = load_frozen) -> int:
    """load stands in for load_frozen in tests."""
    ap = argparse.ArgumentParser(description="Check that a run was M7's final system, unchanged.")
    ap.add_argument("run", type=Path, help="a report.json of scripts/run_e2e.py or a session.json of "
                                           "scripts/ask.py")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    try:
        kind, info, props, problems = recorded(json.loads(args.run.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        print(f"FAIL: {args.run.as_posix()}: {exc}")
        return 1
    try:
        frozen = load()
    except FrozenMismatch as exc:
        print(f"FAIL: the frozen system does not hold: {'; '.join(exc.problems)}")
        return 1
    problems = identity_problems(info, frozen, props) + problems
    if problems:
        print(f"FAIL: {args.run.as_posix()} ({kind}) is not the frozen final system "
              f"({frozen.frozen_path}): {len(problems)} problem(s)")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"OK: {args.run.as_posix()} ({kind}) ran the frozen final system ({frozen.frozen_path}, SHA-256 "
          f"{frozen.frozen_sha256})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Compare two generator runs the way scripts/m1_smoke.py's neutral() compares a paid run with its cached
rerun: identical apart from the money fields, and the second run free.

    .venv/Scripts/python.exe scripts/compare_synth_runs.py runs/m2/synth/<run_a> runs/m2/synth/<run_b>

The index.jsonl rows are matched by id with MONEY_FIELDS dropped and the transcript path reduced to its file
name; transcripts/<id>.json are compared as JSON with the same fields dropped at the top level; of
manifest.json the top-level keys that differ are listed (calls, cost_rmb and the run-specific parts of config
are expected to). Prints one line per request id in sorted order (the status, whether the row and the
transcript are identical, each run's paid calls and cost), the totals of both runs, the differing manifest
keys and a final VERDICT line: PASS when every row and every transcript is identical and run B made no paid
call, else FAIL. A row on one side only, a transcript file that is gone or not JSON, and a transcript named
on one side only all count as not identical; two rows that name no transcript (harness error rows) have
nothing to compare there and count as identical.

Exit 0 on PASS, 1 on FAIL, 2 when either folder has no index.jsonl (or one that is not JSON lines).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research_agent.data.synth import INDEX_FILE, MANIFEST_FILE
from research_agent.eval.lock import JsonlError, read_jsonl

# What a paid run and its cached rerun may differ in and nothing else: the four fields scripts/m1_smoke.py's
# neutral() zeroes on a Transcript (paid_calls, cache_hits, local_calls, cost_rmb) plus the index row's share
# of the rewrite. Keep this tuple and that call in step.
MONEY_FIELDS = ("paid_calls", "cache_hits", "local_calls", "cost_rmb", "rewrite_cost_rmb")
OK, NONE, MISSING, UNREADABLE = "ok", "none", "missing", "unreadable"  # what one side of a comparison holds
Side = tuple[str, object]  # (OK, the value) or (one of the other tags, None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two generator runs apart from money: PASS when every index row and transcript "
                    "is identical and run B made no paid call.")
    parser.add_argument("run_a", type=Path, help="the first run folder (the paid one), holding index.jsonl")
    parser.add_argument("run_b", type=Path, help="the second run folder (the rerun), holding index.jsonl")
    return parser


# ---- the neutral forms ----------------------------------------------------------------------------------


def neutral_row(row: dict) -> dict:
    """The index row without the money fields, its transcript path reduced to the file name."""
    neutral = {key: value for key, value in row.items() if key not in MONEY_FIELDS}
    if isinstance(neutral.get("transcript"), str):
        neutral["transcript"] = Path(neutral["transcript"]).name
    return neutral


def neutral_transcript(data: object) -> object:
    """The transcript JSON without the money fields at the top level (anything but an object is kept)."""
    if not isinstance(data, dict):
        return data
    return {key: value for key, value in data.items() if key not in MONEY_FIELDS}


def cost_of(row: dict) -> float:
    """What the request cost: the trajectory plus its rewrite, as research_agent.data.synth sums them."""
    return float(row.get("cost_rmb") or 0.0) + float(row.get("rewrite_cost_rmb") or 0.0)


# ---- the sides of a comparison --------------------------------------------------------------------------


def json_side(path: Path) -> Side:
    """(OK, the JSON) when path holds JSON, (MISSING, None) without a file, (UNREADABLE, None) otherwise."""
    if not path.is_file():
        return (MISSING, None)
    try:
        return (OK, json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return (UNREADABLE, None)


def row_side(rows: dict[str, dict], rid: str) -> Side:
    return (OK, neutral_row(rows[rid])) if rid in rows else (MISSING, None)


def transcript_side(run: Path, rows: dict[str, dict], rid: str) -> Side:
    """The neutral transcript the run's row names; NONE when the row names no transcript."""
    if rid not in rows:
        return (MISSING, None)
    name = rows[rid].get("transcript")
    if not isinstance(name, str) or not name:
        return (NONE, None)
    tag, data = json_side(run / name)
    return (tag, neutral_transcript(data) if tag == OK else None)


def manifest_side(run: Path) -> Side:
    """The run's manifest as a dict; a manifest that is not a JSON object is UNREADABLE."""
    tag, data = json_side(run / MANIFEST_FILE)
    return (tag, data) if tag != OK or isinstance(data, dict) else (UNREADABLE, None)


def compared(a: Side, b: Side) -> tuple[bool, str]:
    """Whether the two sides are identical, and the word printed for them: same, differs, none (neither side
    names one), or which side lacks one and why (missing, none, unreadable)."""
    if a[0] == b[0] == OK:
        return (a[1] == b[1], "same" if a[1] == b[1] else "differs")
    if a[0] == b[0] == NONE:
        return (True, NONE)
    return (False, ", ".join(f"{tag} in {side}" for side, (tag, _) in (("A", a), ("B", b)) if tag != OK))


# ---- the report -----------------------------------------------------------------------------------------


def rows_by_id(run: Path) -> dict[str, dict]:
    """The run's index rows keyed by id, in file order (a repeated id keeps its last row)."""
    return {str(row.get("id")): row for row in read_jsonl(run / INDEX_FILE)}


def money(row: dict | None) -> str:
    return "-" if row is None else f"paid {row.get('paid_calls') or 0} cost {cost_of(row):.4f} RMB"


def request_line(rid: str, rows_a: dict[str, dict], rows_b: dict[str, dict], row_word: str,
                 transcript_word: str) -> str:
    statuses = [str(rows[rid].get("status")) for rows in (rows_a, rows_b) if rid in rows]
    status = statuses[0] if len(set(statuses)) == 1 else " / ".join(statuses)
    return (f"{rid:<12} {status:<16} row {row_word:<14} transcript {transcript_word:<14} "
            f"A {money(rows_a.get(rid))}  B {money(rows_b.get(rid))}")


def totals_line(rows_a: dict[str, dict], rows_b: dict[str, dict]) -> str:
    parts = []
    for side, rows in (("A", rows_a), ("B", rows_b)):
        paid = sum(int(row.get("paid_calls") or 0) for row in rows.values())
        cached = sum(int(row.get("cache_hits") or 0) for row in rows.values())
        cost = sum(cost_of(row) for row in rows.values())
        parts.append(f"{side} {len(rows)} requests, paid {paid}, cached {cached}, cost {cost:.4f} RMB")
    return "totals: " + "; ".join(parts)


def manifest_line(run_a: Path, run_b: Path) -> str:
    """The top-level manifest keys that differ between the runs, or which run has no usable manifest."""
    sides = [manifest_side(run) for run in (run_a, run_b)]
    lacking = [f"{tag} in {side}" for side, (tag, _) in zip("AB", sides, strict=True) if tag != OK]
    if lacking:
        return "manifest: " + ", ".join(lacking)
    a, b = (data if isinstance(data, dict) else {} for _, data in sides)
    differing = sorted(key for key in a.keys() | b.keys() if key not in a or key not in b or a[key] != b[key])
    return f"manifest: keys that differ: {', '.join(differing) or 'none'}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows: list[dict[str, dict]] = []
    for run in (args.run_a, args.run_b):
        if not (run / INDEX_FILE).is_file():
            print(f"{run} has no {INDEX_FILE}", file=sys.stderr)
            return 2
        try:
            rows.append(rows_by_id(run))
        except (JsonlError, OSError) as exc:
            print(f"{run / INDEX_FILE} is not JSON lines of objects: {exc}", file=sys.stderr)
            return 2
    rows_a, rows_b = rows
    ids = sorted(rows_a.keys() | rows_b.keys())
    rows_same = transcripts_same = 0
    for rid in ids:
        row_same, row_word = compared(row_side(rows_a, rid), row_side(rows_b, rid))
        transcript_same, transcript_word = compared(transcript_side(args.run_a, rows_a, rid),
                                                    transcript_side(args.run_b, rows_b, rid))
        rows_same += row_same
        transcripts_same += transcript_same
        print(request_line(rid, rows_a, rows_b, row_word, transcript_word))
    print(totals_line(rows_a, rows_b))
    print(manifest_line(args.run_a, args.run_b))
    paid_b = sum(int(row.get("paid_calls") or 0) for row in rows_b.values())
    passed = rows_same == len(ids) and transcripts_same == len(ids) and paid_b == 0
    print(f"VERDICT: {'PASS' if passed else 'FAIL'}: rows {rows_same}/{len(ids)} identical, transcripts "
          f"{transcripts_same}/{len(ids)} identical, run B paid {paid_b} calls")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

"""Assemble the development extension set: the reviewed dev items beyond the dev set's quotas
(docs/m6-计划.md §3.2 B2, §4 WP1).

    uv run --no-sync python scripts/finalize_dev_ext.py [--runs runs/m2/synth/dev]
        [--pool-ids data/eval/review/pool-ids.jsonl] [--verdicts data/eval/review/verdicts.jsonl]
        [--handwritten data/eval/handwritten.jsonl]
        [--handwritten-verdicts data/eval/review/handwritten-verdicts.jsonl] [--dev data/eval/dev.jsonl]
        [--out data/eval/dev-ext.jsonl] [--manifest data/eval/dev-ext-manifest.json] [--force]

Why. The development set has 150 items: one item is 0.67 percentage points, and a proportion's standard
error is about 3.9 points, too coarse for the decisions of the M6 round. The review kept more dev items
than the quotas admitted (data/eval/sets-manifest.json: drafted_kept over drafted, handwritten_kept over
handwritten, per stratum). This script gathers them into a second development file, used only to VALIDATE
what was decided on data/eval/dev.jsonl, never to select anything.

The rule. The dev set is assembled again from the same records with the same rule (eval.sets.assemble over
the dev pool rows, the review's verdicts, the items the dev generator run drafts, and the handwritten dev
items; QUOTAS at eval.sets.DRAFTED_SHARE), and it must reproduce --dev item for item, so that the extension
is defined relative to the dev set these records made. The extension is then, per (type, language) stratum:
the drafted dev items the review kept that the dev set did not take, in the review pool's order, and the
handwritten dev items the review kept that it did not take, in id order; ordered as the dev set is
(eval.sets.sort_key). An item keeps the fields and the split ("dev") of a dev item, so every scorer and
runner reads it as one. The handwritten file holds the test split's items too: it is read through
dev_rows, which keeps the dev rows alone before anything else sees it, the test generator run is never
opened, and eval.lock.refuse_test_material checks the written items.

Writes --out (canonical JSON lines, eval.lock.write_jsonl) with its SHA-256 lock beside it
(eval.lock.write_lock: a file with a lock beside it is lock-checked by the runners, as the dev set is), and
--manifest: the rule, the counts (items, drafted, handwritten; per stratum the extension's drafted and
handwritten counts next to the dev set's and the review's, which must agree: drafted = drafted_kept −
dev_drafted, handwritten = handwritten_kept − dev_handwritten), the inputs' SHA-256, the dev set's SHA-256
and the output's SHA-256 and size. Deterministic: no timestamps.

Exit 0 when written; 1 when the sets cannot be assembled (eval.sets.SetError), an input is malformed, or the
records do not reproduce the dev set; 2 when an input is missing or --out exists without --force. Nothing is
written on exit 1 or 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research_agent.eval.items import QUOTAS_DEV, draft_items
from research_agent.eval.lock import (
    JsonlError,
    read_jsonl,
    refuse_test_material,
    sha256_of,
    write_jsonl,
    write_lock,
)
from research_agent.eval.sets import KEEP, SetError, assemble, sort_key

RUNS = (Path("runs/m2/synth/dev"),)
POOL_IDS = Path("data/eval/review/pool-ids.jsonl")
VERDICTS = Path("data/eval/review/verdicts.jsonl")
HANDWRITTEN = Path("data/eval/handwritten.jsonl")
HANDWRITTEN_VERDICTS = Path("data/eval/review/handwritten-verdicts.jsonl")
DEV = Path("data/eval/dev.jsonl")
OUT = Path("data/eval/dev-ext.jsonl")
MANIFEST = Path("data/eval/dev-ext-manifest.json")
QUOTAS = {"dev": QUOTAS_DEV}
RULE = ("the reviewed dev items beyond the dev set's quotas: per (type, language) stratum, the drafted dev "
        "items the review kept that the dev set did not take, in review-pool order, and the handwritten dev "
        "items the review kept that it did not take, in id order; the dev set is assembled again from the "
        "same records first and must reproduce data/eval/dev.jsonl; used for validation only, never for "
        "selection")


def dev_rows(rows: list[dict]) -> list[dict]:
    """The dev split's rows alone: the handwritten file holds the test split's items too."""
    return [row for row in rows if row.get("split") == "dev"]


def by_id(rows: list[dict]) -> dict[str, str]:
    return {row["id"]: row.get("verdict") for row in rows}


def digest(path: Path) -> dict:
    return {"path": path.as_posix(), "sha256": sha256_of(path)}


def extension(pool_rows: list[dict], verdicts: dict[str, str], drafted: dict[str, dict],
              handwritten: list[dict], handwritten_verdicts: dict[str, str],
              dev: list[dict]) -> tuple[list[dict], dict]:
    """The extension items and the per-stratum table (module docstring); SetError when the records do not
    reproduce dev."""
    sets, table = assemble(pool_rows, verdicts, drafted, handwritten, handwritten_verdicts, quotas=QUOTAS)
    made = [item["id"] for item in sets["dev"]]
    given = [item["id"] for item in dev]
    if made != given:
        raise SetError(f"the records assemble a dev set of {len(made)} items that is not the given dev set "
                       f"of {len(given)}: the extension would not be defined relative to it")
    taken = set(made)
    rows = sorted(pool_rows, key=lambda r: (r["type"], r["lang"], r["rank"]))
    items = [drafted[r["id"]] for r in rows
             if verdicts.get(r["id"]) == KEEP and r["id"] not in taken]
    items += [h for h in sorted(handwritten, key=lambda h: h["id"])
              if handwritten_verdicts.get(h["id"]) == KEEP and h["id"] not in taken]
    items.sort(key=sort_key)
    strata = {}
    for name, counts in table["dev"].items():
        typ, lang = name.split("/")
        mine = [i for i in items if (i["type"], i["lang"]) == (typ, lang)]
        strata[name] = {
            "drafted": sum(1 for i in mine if not (i.get("source") or {}).get("handwritten")),
            "handwritten": sum(1 for i in mine if (i.get("source") or {}).get("handwritten")),
            "dev_drafted": counts["drafted"], "dev_handwritten": counts["handwritten"],
            "drafted_kept": counts["drafted_kept"], "handwritten_kept": counts["handwritten_kept"]}
        if (strata[name]["drafted"] != counts["drafted_kept"] - counts["drafted"]
                or strata[name]["handwritten"] != counts["handwritten_kept"] - counts["handwritten"]):
            raise SetError(f"stratum {name}: the extension's counts do not match the review's table")
    return items, strata


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Assemble data/eval/dev-ext.jsonl, the reviewed dev items "
                                             "beyond the dev set's quotas (validation only).")
    ap.add_argument("--runs", type=Path, nargs="+", default=list(RUNS), help="the dev generator run(s)")
    ap.add_argument("--pool-ids", type=Path, default=POOL_IDS)
    ap.add_argument("--verdicts", type=Path, default=VERDICTS)
    ap.add_argument("--handwritten", type=Path, default=HANDWRITTEN)
    ap.add_argument("--handwritten-verdicts", type=Path, default=HANDWRITTEN_VERDICTS)
    ap.add_argument("--dev", type=Path, default=DEV, help="the dev set the records must reproduce")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    required = [args.pool_ids, args.verdicts, args.handwritten, args.handwritten_verdicts, args.dev,
                *(run / "index.jsonl" for run in args.runs), *(run / "manifest.json" for run in args.runs)]
    missing = [p.as_posix() for p in required if not p.is_file()]
    if missing:
        print(f"missing input: {', '.join(missing)}", file=sys.stderr)
        return 2
    if args.out.exists() and not args.force:
        print(f"FAIL: {args.out.as_posix()} exists; --force overwrites it", file=sys.stderr)
        return 2
    try:
        pool_rows = dev_rows(read_jsonl(args.pool_ids))
        verdicts = by_id(dev_rows(read_jsonl(args.verdicts)))
        handwritten = dev_rows(read_jsonl(args.handwritten))
        handwritten_verdicts = by_id(dev_rows(read_jsonl(args.handwritten_verdicts)))
        dev = read_jsonl(args.dev)
        drafted: dict[str, dict] = {}
        for run in args.runs:
            for item in draft_items(run, splits=("dev",), include_unvalidated=False):
                drafted[item["id"]] = item
        items, strata = extension(pool_rows, verdicts, drafted, handwritten, handwritten_verdicts, dev)
        refuse_test_material(items, "the development extension set")
    except (SetError, JsonlError, KeyError, ValueError, TypeError) as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    write_jsonl(items, args.out)
    write_lock(args.out)
    manifest = {
        "rule": RULE,
        "items": len(items),
        "drafted": sum(1 for i in items if not (i.get("source") or {}).get("handwritten")),
        "handwritten": sum(1 for i in items if (i.get("source") or {}).get("handwritten")),
        "strata": strata,
        "dev": digest(args.dev),
        "inputs": {"pool_ids": digest(args.pool_ids), "verdicts": digest(args.verdicts),
                   "handwritten": digest(args.handwritten),
                   "handwritten_verdicts": digest(args.handwritten_verdicts),
                   "runs": [digest(run / "manifest.json") for run in args.runs]},
        "output": {"path": args.out.as_posix(), "sha256": sha256_of(args.out),
                   "bytes": args.out.stat().st_size},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
    print(f"{args.out.as_posix()}: {len(items)} items, drafted {manifest['drafted']}, handwritten "
          f"{manifest['handwritten']}, sha256 {manifest['output']['sha256'][:16]}; manifest "
          f"{args.manifest.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

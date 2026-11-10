"""Calibrate refusal development set B against ACEBench's failure pattern (docs/m8-计划.md WP2, 校准),
before FT5 is trained: on B0's and FT3's runs over dev set B, the set must show what M7's ACEBench run
showed.

    .venv/Scripts/python.exe scripts/calibrate_special_dev.py --b0 <b0 records.jsonl> \\
        --ft <ft records.jsonl> --items data/eval/special-dev-b.jsonl \\
        [--out results/special-dev-calibration-m8.json] [--force]

The runs are scripts/run_special_dev.py's records with their manifests beside them (<stem>.manifest.json),
scored again here against the items (research_agent.eval.special_dev.score_records, with the sentences of
the whole set; an item the scorer fails on is scored as failed, and counted).

The conditions, written before the runs (docs/m8-计划.md WP2):
- K1 (K1_RULE): B0's lenient special rate (the incorrect, missing and irrelevant items of both languages
  pooled) is at least 5 points above FT's;
- K2 (K2_RULE): FT's share of the incorrect items it answers by calling with the invalid value or with the
  missing-parameter sentence in place of the item's own (带非法值照调或错用缺参数) is above B0's: an item
  counts when its answer holds a call (special_dev.find_calls, loose calls included) that keeps an invalid
  value (called_with_invalid), or when a missing case's sentence of the set stands in it and the item's own
  sentence does not. The broad count, any call or any other case's sentence in place of the item's own (the
  failures called and wrong_case), is recorded beside it and decides nothing;
- K3 (K3_RULE): FT alters more verbatim values than B0 on the valid items (zh_altered).
Each is compared exactly (fractions of the counts). PG1 of the checkpoint selection follows K1 (PG1_RULE):
when K1 holds, dev set B's lenient special rate of a checkpoint must be at least B0's plus 5 points; when it
does not, its strict special rate must be at least 0.75.

Checks (exit 2 unless said otherwise; nothing is written): the set is refused as scripts/run_special_dev.py
refuses one (test material or a held-out set: exit 4) and must be dev set B, locked (its lock must be there
and match: exit 5) and of split "b" alone; each run's manifest must be there and name its system (b0, ft),
the set's SHA-256, think mode, greedy decoding and the reasoning budget R = 1,280, and the two runs the same
generation settings and the same family; each run must hold one record of every item. A record that is not
ok (an error, a context overflow) is scored as no answer, and counted.

Writes --out (default results/special-dev-calibration-m8.json; JSON, two-space indent, LF; refused when it
exists unless --force): the rules, the set (path, SHA-256, lock, items), the runs (path, SHA-256, manifest,
family, adapters, error records, scoring errors), each condition at the top of the file ("K1", "K2", "K3",
each with its "holds"), "pg1_fallback" (true when K1 does not hold, so PG1 reads the strict special rate:
what scripts/decide_m8.py reads), the PG1 basis ("pg1") and both summaries. Prints whether each condition
holds and the PG1 basis.

Exit 0 written (whether the conditions hold or not); 2 bad inputs or an existing output without --force; 4
test material or a held-out set; 5 a set without its lock or not matching it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

from research_agent.eval import special_dev as sd
from research_agent.eval.lock import JsonlError, LockError, RefusedTestSet, read_jsonl, sha256_of

SCRIPTS = Path(__file__).resolve().parent
OUT = Path("results/special-dev-calibration-m8.json")
SYSTEMS = ("b0", "ft")
REASONING_BUDGET = 1280  # R of the runs (docs/m8-计划.md, 方法约束)
K1_POINTS = Fraction(5, 100)
PG1_STRICT = 0.75
K1_RULE = ("B0's lenient special rate (incorrect, missing and irrelevant items, both languages pooled) minus "
           "FT's is at least 5 percentage points")
K2_RULE = ("FT's share of the incorrect items answered by calling with the invalid value (a call found, "
           "loose ones included, that keeps an invalid value: called_with_invalid) or with the missing "
           "case's sentence in place of the item's own (a missing sentence of the set stands, the item's "
           "sentence does not) is above B0's; the broad count (failures called and wrong_case: any call, any "
           "other case's sentence) is recorded beside it and decides nothing")
K3_RULE = "FT's count of altered verbatim values on the valid items (zh_altered) is above B0's"
PG1_RULE = ("PG1 when K1 holds: dev set B's lenient special rate of a checkpoint is at least B0's plus 5 "
            f"points; when K1 does not hold: its strict special rate is at least {PG1_STRICT}")


def _load_runner():
    """scripts/run_special_dev.py as a module of its own name (scripts/ is no package), for its guards of a
    set and the path of a run's manifest."""
    spec = importlib.util.spec_from_file_location("_run_special_dev_shared", SCRIPTS / "run_special_dev.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


class InputError(ValueError):
    pass


def k1(b0: dict, ft: dict) -> dict:
    """K1 (K1_RULE) of the two summaries."""
    rates = {}
    for name, summary in (("b0", b0), ("ft", ft)):
        special = summary["special"]
        if not special["n"]:
            raise InputError("the set has no special item")
        rates[name] = Fraction(special["lenient"], special["n"])
    difference = rates["b0"] - rates["ft"]
    return {"b0": float(rates["b0"]), "ft": float(rates["ft"]), "difference": float(difference),
            "threshold": float(K1_POINTS), "holds": difference >= K1_POINTS}


def called_with_invalid(score: dict) -> bool:
    """Whether an incorrect item's answer holds a call that keeps an invalid value (K2_RULE)."""
    return score["call_notation"] is not None and score["called_with_invalid"] is True


def missing_sentence(score: dict) -> bool:
    """Whether an incorrect item's answer holds the missing case's sentence in place of its own (K2_RULE)."""
    return "missing" in (score["sentences"] or []) and score["sentence"] is False


def k2(b0: list[dict], ft: list[dict]) -> dict:
    """K2 (K2_RULE) of the two runs' scores, item by item, with the broad count beside it."""
    out, shares = {}, {}
    for name, scores in (("b0", b0), ("ft", ft)):
        rows = [score for score in scores if score["kind"] == "incorrect"]
        if not rows:
            raise InputError("the set has no incorrect item")
        count = sum(called_with_invalid(row) or missing_sentence(row) for row in rows)
        broad = {failure: sum(row["failure"] == failure for row in rows)
                 for failure in ("called", "wrong_case")}
        shares[name] = Fraction(count, len(rows))
        out[name] = {"n": len(rows), "count": count, "share": float(shares[name]),
                     "called_with_invalid": sum(called_with_invalid(row) for row in rows),
                     "missing_sentence": sum(missing_sentence(row) for row in rows),
                     "broad": {"count": sum(broad.values()), "share": sum(broad.values()) / len(rows),
                               **broad}}
    return {**out, "holds": shares["ft"] > shares["b0"]}


def k3(b0: dict, ft: dict) -> dict:
    """K3 (K3_RULE) of the two summaries."""
    b0_altered, ft_altered = b0["valid"]["zh_altered"], ft["valid"]["zh_altered"]
    return {"b0": b0_altered, "ft": ft_altered, "holds": ft_altered > b0_altered}


def pg1(b0: dict, k1_holds: bool) -> dict:
    """The PG1 basis that K1 decides (PG1_RULE), with B0's lenient special rate that the lenient basis adds 5
    points to."""
    rate = b0["special"]["lenient"] / b0["special"]["n"]
    if k1_holds:
        return {"basis": "lenient", "b0_lenient_special": rate,
                "rule": f"lenient special rate >= {rate:.4f} + {float(K1_POINTS)}"}
    return {"basis": "strict", "b0_lenient_special": rate, "rule": f"strict special rate >= {PG1_STRICT}"}


def load_set(path: Path) -> list[dict]:
    """Dev set B after the checks of the module docstring (RefusedTestSet, LockError, InputError)."""
    try:
        items, locked = runner.load_items(path)
    except runner.MalformedItems as exc:
        raise InputError(f"malformed items: {exc}") from None
    if not locked:
        raise LockError(f"{path.name} has no lock: dev set B is calibrated only as it was locked")
    splits = sorted({item["split"] for item in items})
    if splits != ["b"]:
        raise InputError(f"{path} holds items of split {splits}; the calibration is on dev set B alone")
    return items


def load_run(path: Path, system: str, items_sha256: str) -> tuple[list[dict], dict]:
    """A run's records and its manifest, checked against its system and the set (module docstring)."""
    manifest_file = runner.manifest_path(path)
    try:
        records = read_jsonl(path)
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, JsonlError) as exc:
        raise InputError(f"{system}: {exc}") from None
    if not isinstance(manifest, dict):
        raise InputError(f"{manifest_file} is not a JSON object")
    generation = manifest.get("generation") or {}
    decoding = generation.get("decoding") or {}
    problems = []
    if manifest.get("system") != system:
        problems.append(f"it is a run of {manifest.get('system')!r}")
    if manifest.get("items_sha256") != items_sha256:
        problems.append("it was run on another set (items_sha256)")
    if generation.get("mode") != "think" or decoding.get("temperature") != 0.0:
        problems.append("it did not think greedily")
    if generation.get("reasoning_budget") != REASONING_BUDGET:
        problems.append(f"its reasoning budget is {generation.get('reasoning_budget')}, not "
                        f"{REASONING_BUDGET}")
    if problems:
        raise InputError(f"--{system} {path}: {'; '.join(problems)} ({manifest_file.name})")
    return records, manifest


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Calibrate dev set B against ACEBench's failure pattern.")
    p.add_argument("--b0", type=Path, required=True,
                   help="B0's records on dev set B (scripts/run_special_dev.py)")
    p.add_argument("--ft", type=Path, required=True, help="FT's records on dev set B")
    p.add_argument("--items", type=Path, required=True, help="dev set B")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--force", action="store_true", help="overwrite an existing output")
    args = p.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    if args.out.exists() and not args.force:
        print(f"FAIL: {args.out} exists; pass --force to overwrite", file=sys.stderr)
        return 2
    try:
        items = load_set(args.items)
        digest = sha256_of(args.items)
        runs = {system: load_run(path, system, digest) for system, path in (("b0", args.b0), ("ft", args.ft))}
        (_, b0_manifest), (_, ft_manifest) = runs["b0"], runs["ft"]
        if b0_manifest.get("generation") != ft_manifest.get("generation"):
            raise InputError("the two runs were generated with other settings (manifest generation)")
        families = {(manifest.get("family") or {}).get("name") for manifest in (b0_manifest, ft_manifest)}
        if len(families) != 1:
            raise InputError(f"the two runs are of other families: {sorted(map(str, families))}")
        try:
            scores = {system: sd.score_records(records, items) for system, (records, _) in runs.items()}
            summaries = {system: sd.summarize(scores[system], items) for system in scores}
        except ValueError as exc:
            raise InputError(str(exc)) from None
        checks = {"K1": k1(summaries["b0"], summaries["ft"]), "K2": k2(scores["b0"], scores["ft"]),
                  "K3": k3(summaries["b0"], summaries["ft"])}
    except RefusedTestSet as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 4
    except LockError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 5
    except InputError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    basis = pg1(summaries["b0"], checks["K1"]["holds"])
    errors = {system: sum(record.get("status") != "ok" for record in records)
              for system, (records, _) in runs.items()}
    result = {
        "calibration": "dev set B against ACEBench's failure pattern (docs/m8-计划.md WP2, 校准)",
        "rules": {"K1": K1_RULE, "K2": K2_RULE, "K3": K3_RULE, "PG1": PG1_RULE, "scoring": sd.RULES},
        "items": {"path": args.items.as_posix(), "sha256": digest, "locked": True, "n": len(items)},
        "runs": {system: {"path": path.as_posix(), "sha256": sha256_of(path),
                          "manifest": runner.manifest_path(path).as_posix(),
                          "family": (runs[system][1].get("family") or {}).get("name"),
                          "adapters": runs[system][1].get("adapters"), "errors": errors[system],
                          "scoring_errors": summaries[system]["scoring_errors"]}
                 for system, path in (("b0", args.b0), ("ft", args.ft))},
        **checks,
        "pg1_fallback": not checks["K1"]["holds"],
        "pg1": basis,
        "summaries": summaries,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                        newline="\n")
    for system in SYSTEMS:
        if errors[system]:
            print(f"{system}: {errors[system]} error record(s) scored as no answer")
        if summaries[system]["scoring_errors"]:
            print(f"{system}: {summaries[system]['scoring_errors']} item(s) the scorer failed on, scored as "
                  "failed")

    def verdict(check: dict) -> str:
        return "holds" if check["holds"] else "does not hold"

    one, two, three = checks["K1"], checks["K2"], checks["K3"]
    print(f"K1 lenient special: b0 {100 * one['b0']:.1f}%, ft {100 * one['ft']:.1f}%, difference "
          f"{100 * one['difference']:.1f} points (at least 5): {verdict(one)}")
    print(f"K2 incorrect items called with the invalid value or answered with the missing sentence: b0 "
          f"{two['b0']['count']}/{two['b0']['n']}, ft {two['ft']['count']}/{two['ft']['n']}: {verdict(two)} "
          f"(broad count, any call or other case's sentence: b0 {two['b0']['broad']['count']}, ft "
          f"{two['ft']['broad']['count']})")
    print(f"K3 verbatim values altered: b0 {three['b0']}, ft {three['ft']}: {verdict(three)}")
    print(f"PG1: {basis['basis']} ({basis['rule']})")
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

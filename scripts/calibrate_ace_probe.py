"""Calibrate the ACEBench-style development probe against ACEBench (docs/m6-计划.md §3.4 D3, WP4): the
probe is usable only when it orders B0, FT and B4 the way ACEBench does, B0 > B4 ≈ FT, on the parsable
rate and on the sentence hit rate.

    uv run --no-sync python scripts/calibrate_ace_probe.py results/ace-probe-b0.json \\
        results/ace-probe-ft.json results/ace-probe-b4.json \\
        --acebench results/acebench-m4.json results/acebench-m4-b4.json \\
        [--out results/ace-probe-calibration.json] [--margin 0.0]

Rates. Probe (scripts/run_ace_probe.py results, one per system, named by manifest.system; all three on the
same probe SHA-256): kinds.normal.parsable_rate and kinds.incomplete.sentence_rate. ACEBench (the
scripts/run_acebench.py results, their systems merged): parsable rate = the answers in ACEBench's format
(format[category].acebench_format) over all answers of the normal categories (groups.normal), both
languages pooled; sentence rate = correct / n of special_incomplete, both languages pooled (the special
category the probe's incomplete items stand in for).

The ordering (ORDERING_RULE), per rate: B0 above both fine-tuned systems by more than --margin, and FT and B4
closer to each other than the nearer of them is to B0 (|FT - B4| < B0 - max(FT, B4)). It must hold on both
rates for ACEBench (the reference; if it does not, there is no ordering to match) and for the probe.

Writes --out (JSON, two-space indent, LF) with the rates, the checks and the verdict. Exit 0 "probe usable";
1 "probe unusable" (the plan's fallback: run ACEBench in full for every candidate); 2 on bad inputs (not the
three systems once each, different probes, a system missing from the ACEBench results, malformed files).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SYSTEMS = ("b0", "ft", "b4")
RATES = ("parsable_rate", "sentence_rate")
OUT = Path("results/ace-probe-calibration.json")
ORDERING_RULE = (
    "per rate: B0 - max(FT, B4) > margin, and |FT - B4| < B0 - max(FT, B4) (FT and B4 closer to "
    "each other than the nearer of them is to B0); required on both rates for ACEBench and for "
    "the probe"
)


class InputError(ValueError):
    pass


def ordering(rates: dict[str, float], margin: float = 0.0) -> dict:
    """The ordering check of one rate (ORDERING_RULE)."""
    b0, ft, b4 = (rates[s] for s in SYSTEMS)
    gap = b0 - max(ft, b4)
    above = gap > margin
    close = abs(ft - b4) < gap
    return {
        "rates": {s: rates[s] for s in SYSTEMS},
        "b0_gap": gap,
        "ft_b4_distance": abs(ft - b4),
        "b0_above": above,
        "ft_b4_close": close,
        "holds": above and close,
    }


def probe_rates(results: list[dict]) -> tuple[dict, str]:
    """{rate: {system: value}} of the three probe results and their common probe SHA-256."""
    by_system: dict[str, dict] = {}
    for r in results:
        try:
            system = r["manifest"]["system"]
        except (KeyError, TypeError):
            raise InputError("a probe result without manifest.system") from None
        if system in by_system:
            raise InputError(f"two probe results for {system}")
        by_system[system] = r
    if sorted(by_system) != sorted(SYSTEMS):
        raise InputError(f"probe results for {sorted(by_system)}, need {', '.join(SYSTEMS)} once each")
    shas = {r["manifest"]["probe"]["sha256"] for r in by_system.values()}
    if len(shas) != 1:
        raise InputError("the probe results were measured on different probes (probe SHA-256 differ)")
    try:
        rates = {
            "parsable_rate": {s: float(by_system[s]["kinds"]["normal"]["parsable_rate"]) for s in SYSTEMS},
            "sentence_rate": {
                s: float(by_system[s]["kinds"]["incomplete"]["sentence_rate"]) for s in SYSTEMS
            },
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise InputError(f"a probe result without its rates ({exc})") from None
    return rates, shas.pop()


def acebench_rates(results: list[dict]) -> dict:
    """{rate: {system: value}} of the merged ACEBench results (module docstring)."""
    systems: dict[str, dict] = {}
    groups = None
    for r in results:
        groups = groups or r.get("groups")
        systems.update(r.get("systems") or {})
    missing = [s for s in SYSTEMS if s not in systems]
    if missing or not groups:
        raise InputError(f"the ACEBench results lack {missing or 'their groups'}")
    out: dict[str, dict] = {"parsable_rate": {}, "sentence_rate": {}}
    try:
        for s in SYSTEMS:
            fmt = n = correct = total = 0
            for lang in systems[s]["languages"].values():
                for category in groups["normal"]:
                    fmt += lang["format"][category]["acebench_format"]
                    n += lang["format"][category]["n"]
                inc = lang["categories"]["special_incomplete"]
                correct, total = correct + inc["correct"], total + inc["n"]
            out["parsable_rate"][s] = fmt / n
            out["sentence_rate"][s] = correct / total
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        raise InputError(f"malformed ACEBench result ({type(exc).__name__}: {exc})") from None
    return out


def read_json(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InputError(f"{path}: {exc}") from None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("probe_results", nargs=3, help="the run_ace_probe.py results of b0, ft and b4")
    p.add_argument("--acebench", nargs="+", required=True, help="the run_acebench.py results")
    p.add_argument("--out", type=Path, default=OUT)
    p.add_argument("--margin", type=float, default=0.0, help="B0's least lead over FT and B4 (fraction)")
    args = p.parse_args(argv)
    try:
        probe, sha = probe_rates([read_json(f) for f in args.probe_results])
        ace = acebench_rates([read_json(f) for f in args.acebench])
    except InputError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    checks = {
        f"{source}:{rate}": ordering(rates[rate], args.margin)
        for source, rates in (("acebench", ace), ("probe", probe))
        for rate in RATES
    }
    usable = all(c["holds"] for c in checks.values())
    failed = [k for k, c in checks.items() if not c["holds"]]
    result = {
        "rule": ORDERING_RULE,
        "margin": args.margin,
        "probe_sha256": sha,
        "inputs": {"probe": list(args.probe_results), "acebench": list(args.acebench)},
        "probe": probe,
        "acebench": ace,
        "checks": checks,
        "usable": usable,
        "verdict": "probe usable" if usable else f"probe unusable: the ordering fails on {', '.join(failed)}",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    for key, c in checks.items():
        rates = ", ".join(f"{s} {100 * c['rates'][s]:.1f}%" for s in SYSTEMS)
        print(f"{key}: {rates}: {'holds' if c['holds'] else 'FAILS'}")
    print(result["verdict"])
    return 0 if usable else 1


if __name__ == "__main__":
    sys.exit(main())

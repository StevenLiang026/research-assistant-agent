"""The oracle of the routing curve's cost row (docs/plan.md, 路由曲线: success against the API share and
the cost per 100 items, 含随机与理想路由), offline over the records the frozen routing file was made from;
it calls no model.

    .venv/Scripts/python.exe scripts/route_cost_oracle.py [--routing results/routing-m4.json] \\
        [--out results/cost-oracle-m4.json] [--force]

Why a file of its own. scripts/route_curve.py froze results/routing-m4.json, whose SHA-256 eval.systems and
the M5 test configuration check, and its oracle curve is per number of routed items, without costs. The
oracle at a cost needs every item's cost, so this script computes it from the same records and writes it
beside, and the routing file stays as it is. scripts/figs/route_curve_tex.py draws it in the figure's cost
row.

Inputs. The routing file's FT and B2 development records and scores (its "inputs" ft_records, ft_scores,
b2_records and b2_scores, paths relative to the repository root), each checked against the SHA-256 the
routing file recorded. The rows are route_curve.py's own (build_rows, at the repository's price of the
routing file's cloud model). As a check that they are the rows the routing file was made from, FT's and B2's
success and the cost of routing every item (its scan's route_everything point) must come out exactly as the
routing file has them: route_curve.py computed them with the same arithmetic over the same rows in the same
order, so any difference, however small, means other rows or another price. Test material is refused
(eval.lock.refuse_test_material over the records' items).

The oracle at a cost. Routing an item helps only when FT fails it and B2 passes it (a gain); on any other
item it leaves the success as it is or lowers it, and it costs B2's price either way. So the most items right
at a cost of at most C per 100 items are FT's successes plus the most gains whose costs add up to at most C,
and taking the gains cheapest first fits the most of them: point j of the curve (j = 0 to the number of
gains) routes the j cheapest gains, at a cost per 100 items of 100 x their summed cost / n, with FT's
successes + j items right. Between two points the success is the earlier point's, and past the last one more
money buys no more success. The routing file's oracle differs: at a number k of routed items it must route k
of them, losses included, and so ends at B2's success; this one spends only on what helps.

Output (--out, default results/cost-oracle-m4.json; scripts/build_report.py reads it as cost-oracle-<name>):
JSON with two-space indent, UTF-8, LF line ends: the rule, the routing file's path and SHA-256, the four
inputs with their paths and SHA-256, the cloud model, n, FT's successes, the number of gains, the cost per
100 items of routing every item (b2_cost_per_100_rmb), and the curve, each point with routed (gains routed),
cost_per_100_rmb, successes and success. An existing file is kept unless --force is given.

Exit 0 when written; 1 on malformed input (a routing file or records that are not route_curve.py's, no
items); 2 when the inputs are not those the routing file was made from (another SHA-256, other successes or
another cost of routing every item), or on a usage error; 3 when --out exists and --force was not given; 4
when the inputs are test material. Nothing is written unless the exit is 0.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from research_agent.eval.lock import JsonlError, RefusedTestSet, refuse_test_material, sha256_of
from research_agent.llm.budget import load_prices

REPO = Path(__file__).resolve().parents[1]
ROUTING = Path("results/routing-m4.json")
OUT = Path("results/cost-oracle-m4.json")
INPUTS = ("ft_records", "ft_scores", "b2_records", "b2_scores")
RULE = ("At a cost of at most C per 100 items, the most items right: FT's successes plus the items FT fails "
        "and B2 passes, routed cheapest first, as many as C pays for; no other item is routed "
        "(scripts/route_cost_oracle.py).")


def _load_route_curve():
    """scripts/route_curve.py as a module of its own name (scripts/ is no package), for its rows."""
    spec = importlib.util.spec_from_file_location("_route_curve_shared", REPO / "scripts" / "route_curve.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


route_curve = _load_route_curve()
InputError = route_curve.InputError


def cost_oracle(rows: list[dict]) -> list[dict]:
    """The oracle at a cost over route_curve.py's rows (module docstring): one point per number of gains
    routed, the cheapest first."""
    n = len(rows)
    successes = sum(row["ft"] for row in rows)
    gains = sorted(row["cost"] for row in rows if not row["ft"] and row["b2"])
    points, spent = [], 0.0
    for routed in range(len(gains) + 1):
        if routed:
            spent += gains[routed - 1]
        points.append({"routed": routed, "cost_per_100_rmb": 100 * spent / n, "successes": successes + routed,
                       "success": (successes + routed) / n})
    return points


def _rows(routing: dict, root: Path) -> tuple[list[dict], dict[str, dict]]:
    """route_curve.py's rows from the routing file's inputs, and those inputs, each checked against the
    SHA-256 it recorded (module docstring, Inputs)."""
    inputs = {}
    for name in INPUTS:
        entry = routing["inputs"][name]
        digest = sha256_of(root / entry["path"])
        if digest != entry["sha256"]:
            raise InputError(f"{entry['path']} has changed since the routing file was frozen: its SHA-256 is "
                             f"{digest}, the routing file recorded {entry['sha256']}")
        inputs[name] = {"path": entry["path"], "sha256": digest}
    ft_records = route_curve._records(root / inputs["ft_records"]["path"])
    b2_records = route_curve._records(root / inputs["b2_records"]["path"])
    refuse_test_material([{**(r.get("item") or {}), "id": r["id"]} for r in ft_records + b2_records],
                         "route_cost_oracle's records")
    rows = route_curve.build_rows(ft_records, route_curve._scores(root / inputs["ft_scores"]["path"]),
                                  b2_records, route_curve._scores(root / inputs["b2_scores"]["path"]),
                                  load_prices()[routing["cloud_model"]])
    if not rows:
        raise JsonlError("the records hold no items")
    return rows, inputs


def _check(rows: list[dict], routing: dict) -> float:
    """That the rows give the routing file's FT and B2 success and cost of routing every item; that cost."""
    n = len(rows)
    ends = [point["cost_per_100_rmb"] for point in routing["curves"]["routing"]
            if point.get("label") == "route_everything"]
    if len(ends) != 1:
        raise JsonlError("the routing file's scan has no single route_everything point")
    mine = {"FT's success": sum(r["ft"] for r in rows) / n, "B2's success": sum(r["b2"] for r in rows) / n,
            "the cost of routing every item": 100 * sum(r["cost"] for r in rows) / n}
    theirs = {"FT's success": routing["ft_success"], "B2's success": routing["b2_success"],
              "the cost of routing every item": ends[0]}
    for what, value in mine.items():
        if value != theirs[what]:  # exactly: the same arithmetic over the same rows (module docstring)
            raise InputError(f"{what} over the inputs is {value!r}, the routing file's {theirs[what]!r}: "
                             "these are not the rows it was made from")
    return mine["the cost of routing every item"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="The routing curve's oracle at a cost, beside the frozen "
                                             "routing file.")
    ap.add_argument("--routing", type=Path, default=ROUTING,
                    help=f"scripts/route_curve.py's routing file (default {ROUTING.as_posix()})")
    ap.add_argument("--out", type=Path, default=OUT, help=f"the result (default {OUT.as_posix()})")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    return ap


def main(argv: list[str] | None = None, *, root: Path = REPO) -> int:
    """root is the repository the routing file's input paths are relative to (a test's folder in tests)."""
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    if args.out.exists() and not args.force:
        print(f"FAIL: {args.out.as_posix()} exists; --force overwrites it", file=sys.stderr)
        return 3
    try:
        routing = route_curve._json(args.routing)
        if not isinstance(routing, dict):
            raise JsonlError(f"{args.routing.as_posix()} is not a routing file of scripts/route_curve.py")
        rows, inputs = _rows(routing, root)
        everything = _check(rows, routing)
    except RefusedTestSet as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 4
    except InputError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    except (JsonlError, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f"FAIL: malformed input: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    def relative(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            return resolved.as_posix()

    curve = cost_oracle(rows)
    result = {"rule": RULE, "routing": {"path": relative(args.routing), "sha256": sha256_of(args.routing)},
              "inputs": inputs, "cloud_model": routing["cloud_model"], "n": len(rows),
              "ft_successes": curve[0]["successes"], "gains": len(curve) - 1,
              "b2_cost_per_100_rmb": everything, "curve": curve}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                        newline="\n")
    best = curve[-1]
    print(f"{result['gains']} gains: the oracle reaches {best['success']:.4f} at "
          f"{best['cost_per_100_rmb']:.4f} per 100 items (routing every item costs {everything:.4f}); "
          f"written {args.out.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

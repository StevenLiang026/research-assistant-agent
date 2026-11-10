"""Evaluate a frozen routing configuration's policy on other sets' recorded signals (docs/m6-计划.md §3.3 C3:
the development extension set validates what was chosen on the development set; docs/m7-计划.md §5 and WP-K:
dev-ext and dev-v2-ext jointly), offline: it reads saved runs, calls no model and chooses nothing.

    .venv/Scripts/python.exe scripts/route_validate.py --routing results/routing-m6.json \\
        --ft runs/m6/dev-ext-ft2-signals.jsonl --ft-scores runs/eval/scores/dev-ext-ft2-signals.json \\
        --b2 runs/m6/dev-ext-b2.jsonl --b2-scores runs/eval/scores/dev-ext-b2.json \\
        --set data/eval/dev-ext.jsonl --out results/routing-m6.json [--force]
    .venv/Scripts/python.exe scripts/route_validate.py --routing results/routing-m7.json \\
        --ft runs/m7/dev-ext-ft3-signals.jsonl --ft-scores runs/m7/scores/dev-ext-ft3-signals.json \\
        --b2 runs/m6/dev-ext-b2.jsonl --b2-scores runs/eval/scores/dev-ext-b2.json \\
        --set data/eval/dev-ext.jsonl \\
        --ft runs/m7/dev-v2-ext-ft3-signals.jsonl --ft-scores runs/m7/scores/dev-v2-ext-ft3-signals.json \\
        --b2 runs/m7/dev-v2-ext-b2.jsonl --b2-scores runs/m7/scores/dev-v2-ext-b2.json \\
        --set data/eval/v2/dev-v2-ext.jsonl --out results/routing-m7.json

Inputs. The routing file scripts/route_curve.py wrote (its signal, threshold, rules and chosen policy, the
adapter and the cloud model it names); per set, FT's records over the set made with --signals and B2's
records over it, each with its scores (scripts/score.py), and beside each records file its manifest
(<stem>.manifest.json): FT's must list exactly one adapter, the routing file's, and record the generation the
routing file was chosen on (its "generation", as scripts/run_eval.py records it, equal to the routing file's
"ft_generation"; neither one for a routing file chosen on runs without a generation, as M4 to M6), B2's model
must be the routing file's cloud model, and both must name the set (--set) by its SHA-256 as it is now. The
rows are route_curve.py's (build_rows: the same successes, signals, hard reasons, calls, forced closes and
list-price costs). Test material is refused (eval.lock.refuse_test_material over the records' items).

Several sets (M7: dev-ext and dev-v2-ext jointly). The five options --ft, --ft-scores, --b2, --b2-scores and
--set are given once per set, the k-th of each naming the k-th set's inputs, or it is a usage error. Every set
is checked as above on its own, and no item id may be in two sets; the rows are the sets' rows in their order,
and the block is over all of them.

The block. The frozen policy's point on the rows, as route_curve.py's threshold_point computes it with the
routing file's rules and threshold: n, policy, signal, threshold, rules, routed, api_share, successes,
success, cost_per_100_rmb (the routed items' B2 answers at the list price), random_success at the same
share, and ft_success and b2_success on the rows; for a routing file of the target objective (M7) also the
cost cap on the rows, half of what routing every one of them to B2 costs per 100 items
("cost_cap_per_100_rmb", route_curve.cost_cap), and whether the point's cost is within it ("within_cap";
docs/m7-计划.md G3 asks it of dev-ext and dev-v2-ext as of the development sets); then, for one set, its path
and SHA-256 ("set",
"set_sha256") and the inputs with their SHA-256, as before; for several, "sets", one entry per set in order
with its path, SHA-256, number of items and inputs; then the adapter and the cloud model. Nothing is chosen
here: the point is the routing file's, evaluated where it was not chosen.

Output (--out). When --out is the routing file itself, the block is added to it under "validation".<name>,
the set's stem (dev-ext), or the stems of several joined by "+" (dev-ext+dev-v2-ext), and every other field is
left as it is; otherwise a file of its own, {"routing": {path, sha256 of the routing file}, "validation":
{<name>: block}}. An existing block of the same name, or an existing separate file, is kept unless --force is
given. JSON with two-space indent, UTF-8, LF line ends.

Exit 0 when written; 1 on malformed input (a file that is not JSON or JSONL, a routing file or records of the
wrong shape, a generation in either that is not an object); 2 when the inputs do not belong together (a
routing file without a threshold, other item ids, a manifest of another system or set, another adapter,
generation or cloud model, sets holding the same item) or on a usage error (not every input given once per
set); 3 when the block or the file exists and --force was not given; 4 when the inputs are test material.
Nothing is written unless the exit is 0.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from research_agent.agent.route import SIGNALS, RouteRules
from research_agent.eval.lock import (
    JsonlError,
    RefusedTestSet,
    refuse_test_material,
    refuse_test_sets,
    sha256_of,
)
from research_agent.llm.budget import load_prices

REPO = Path(__file__).resolve().parents[1]
POINT_KEYS = ("routed", "api_share", "successes", "success", "cost_per_100_rmb", "random_success")
INPUT_OPTIONS = ("--ft", "--ft-scores", "--b2", "--b2-scores", "--set")  # given once per set
JOIN = "+"  # the block of several sets is named after their stems joined by it


def _load_route_curve():
    """scripts/route_curve.py as a module of its own name (scripts/ is no package): its rows and points."""
    spec = importlib.util.spec_from_file_location("_route_curve_validate",
                                                  REPO / "scripts" / "route_curve.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


route_curve = _load_route_curve()
InputError = route_curve.InputError


def frozen_policy(config: dict) -> tuple[str, float, RouteRules | None, str | None]:
    """The routing file's signal, threshold, rules and chosen policy; InputError when it has no threshold."""
    if config.get("threshold") is None or config.get("rule_met") is False:
        raise InputError("the routing file has no threshold (its rule was not met): nothing to validate")
    signal = config.get("signal")
    if signal not in SIGNALS:
        raise InputError(f"the routing file's signal {signal!r} is not one of {list(SIGNALS)}")
    rules = config.get("rules")
    try:
        rules = None if rules is None else RouteRules.from_json(rules)
    except ValueError as exc:
        raise InputError(f"the routing file's {exc}") from None
    return signal, config["threshold"], rules, config.get("chosen_policy")


def validation_block(rows: list[dict], *, signal: str, threshold: float, rules: RouteRules | None,
                     policy: str | None) -> dict:
    """The frozen policy's point on the rows (module docstring: the block, its first part)."""
    point = route_curve.threshold_point(rows, signal, threshold, rules)
    ft_success, b2_success = route_curve._rates(rows)
    return {"n": len(rows), "policy": policy, "signal": signal, "threshold": threshold,
            "rules": None if rules is None else rules.to_json(), **{k: point[k] for k in POINT_KEYS},
            "ft_success": ft_success, "b2_success": b2_success}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate a frozen routing configuration on other sets.")
    ap.add_argument("--routing", type=Path, required=True, help="scripts/route_curve.py's routing file")
    ap.add_argument("--ft", type=Path, action="append", required=True,
                    help="FT's records over the set, made with --signals (once per set)")
    ap.add_argument("--ft-scores", type=Path, action="append", required=True,
                    help="scripts/score.py's output for --ft (once per set)")
    ap.add_argument("--b2", type=Path, action="append", required=True,
                    help="B2's records over the set (once per set)")
    ap.add_argument("--b2-scores", type=Path, action="append", required=True,
                    help="scripts/score.py's output for --b2 (once per set)")
    ap.add_argument("--set", type=Path, action="append", required=True,
                    help="the set the records are over (dev-ext; once per set)")
    ap.add_argument("--out", type=Path, required=True,
                    help="the routing file itself (the block is added to it) or a file of its own")
    ap.add_argument("--force", action="store_true", help="replace an existing block or file")
    return ap


def _beside(records: Path, suffix: str) -> Path:
    return records.with_name(records.stem + suffix)


@dataclass(frozen=True)
class SetInputs:
    """One set and the records over it (module docstring: inputs)."""

    set: Path
    ft: Path
    ft_scores: Path
    b2: Path
    b2_scores: Path

    def files(self) -> dict[str, Path]:
        """The six inputs by the names the block records them under, in its order."""
        return {"ft_records": self.ft, "ft_scores": self.ft_scores,
                "ft_manifest": _beside(self.ft, ".manifest.json"),
                "b2_records": self.b2, "b2_scores": self.b2_scores,
                "b2_manifest": _beside(self.b2, ".manifest.json")}


def input_sets(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[SetInputs]:
    """The sets the options name, in their order; a usage error unless every input is given once per set."""
    given = {"--ft": args.ft, "--ft-scores": args.ft_scores, "--b2": args.b2, "--b2-scores": args.b2_scores,
             "--set": args.set}
    count = len(args.set)
    uneven = [f"{option} {len(values)}" for option, values in given.items() if len(values) != count]
    if uneven:
        parser.error(f"give {', '.join(INPUT_OPTIONS)} once per set: --set names {count} set(s), but "
                     f"{'; '.join(uneven)}")
    return [SetInputs(*paths) for paths in zip(args.set, args.ft, args.ft_scores, args.b2, args.b2_scores,
                                               strict=True)]


def same_generation(generation: dict | None, chosen: dict | None, where: Path) -> None:
    """InputError unless FT's run over the set where (its manifest's generation, None without one) is in the
    generation the routing file was chosen on (chosen: its ft_generation, None without one)."""
    if generation == chosen:
        return
    text = route_curve.generation_text
    if generation is None:
        raise InputError(f"FT's run over {where.as_posix()} recorded no generation, while the routing file "
                         f"was chosen on FT's runs with {text(chosen)} (ft_generation)")
    if chosen is None:
        raise InputError(f"FT's run over {where.as_posix()} was generated with {text(generation)}, while the "
                         "routing file records no ft_generation: it was chosen on FT's runs without one")
    raise InputError(f"FT's run over {where.as_posix()} was generated with {text(generation)}, the routing "
                     f"file was chosen on FT's runs with {text(chosen)} (ft_generation)")


def read_set(inputs: SetInputs, routing: dict, root: Path) -> tuple[list[dict], str, str, dict[str, str]]:
    """One set's rows, its SHA-256, the cloud model and the SHA-256 of every input, once its inputs are
    checked on their own and against the routing file (module docstring: inputs); the exceptions main turns
    into exit codes."""
    files = inputs.files()
    refuse_test_sets(inputs.set, (Path.cwd(), root))
    ft_records, b2_records = route_curve._records(inputs.ft), route_curve._records(inputs.b2)
    refuse_test_material([{**(r.get("item") or {}), "id": r["id"]} for r in ft_records + b2_records],
                         "route_validate's records")
    set_digest = sha256_of(inputs.set)
    ft_manifest = route_curve._manifest(files["ft_manifest"], "ft", set_digest, inputs.set)
    b2_manifest = route_curve._manifest(files["b2_manifest"], "b2", set_digest, inputs.set)
    adapters = ft_manifest.get("adapters")
    if not isinstance(adapters, list) or len(adapters) != 1 or not adapters[0].get("sha256"):
        raise InputError(f"{files['ft_manifest'].as_posix()} must list exactly one adapter with its "
                         f"SHA-256, got {adapters!r}")
    if adapters[0]["sha256"] != routing["adapter_sha256"]:
        raise InputError(f"FT's run used the adapter {adapters[0]['sha256']}, the routing file was "
                         f"chosen with {routing['adapter_sha256']}")
    same_generation(route_curve._generation(ft_manifest, files["ft_manifest"]),
                    routing.get(route_curve.FT_GENERATION), inputs.set)
    cloud_model = b2_manifest.get("model")
    if cloud_model != routing.get("cloud_model"):
        raise InputError(f"B2's run used {cloud_model!r}, the routing file routes to "
                         f"{routing.get('cloud_model')!r}")
    rows = route_curve.build_rows(ft_records, route_curve._scores(inputs.ft_scores), b2_records,
                                  route_curve._scores(inputs.b2_scores), load_prices()[cloud_model])
    return rows, set_digest, cloud_model, {name: sha256_of(path) for name, path in files.items()}


def main(argv: list[str] | None = None, *, root: Path = REPO) -> int:
    """root is the repository whose relative paths the block records (a test's folder in tests)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    sets = input_sets(args, parser)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    try:
        routing = route_curve._json(args.routing)
        if (not isinstance(routing, dict) or "signal" not in routing or "adapter_sha256" not in routing
                or not isinstance(routing.get(route_curve.FT_GENERATION, {}), dict)):
            raise JsonlError(f"{args.routing.as_posix()} is not a routing file of scripts/route_curve.py")
        signal, threshold, rules, policy = frozen_policy(routing)
        read = [read_set(inputs, routing, root) for inputs in sets]
        held: dict[str, Path] = {}
        for inputs, (rows, *_) in zip(sets, read, strict=True):
            for row in rows:
                if row["id"] in held:
                    raise InputError(f"the sets {held[row['id']].as_posix()} and {inputs.set.as_posix()} "
                                     f"both hold the item {row['id']!r}: the sets must be disjoint")
                held[row["id"]] = inputs.set
        rows = [row for set_rows, *_ in read for row in set_rows]
    except RefusedTestSet as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 4
    except InputError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    except (JsonlError, OSError, ValueError, KeyError, AttributeError) as exc:
        print(f"FAIL: malformed input: {exc}", file=sys.stderr)
        return 1
    if not rows:
        print("FAIL: malformed input: no items", file=sys.stderr)
        return 1

    def relative(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            return resolved.as_posix()

    def inputs_of(inputs: SetInputs, digests: dict[str, str]) -> dict:
        return {name: {"path": relative(path), "sha256": digests[name]}
                for name, path in inputs.files().items()}

    block = validation_block(rows, signal=signal, threshold=threshold, rules=rules, policy=policy)
    if routing.get("objective") == route_curve.TARGET:  # the cap on these rows (module docstring: the block)
        ft_success, b2_success = route_curve._rates(rows)
        everything = route_curve._point(rows, [True] * len(rows), route_curve.EVERYTHING, None, ft_success,
                                        b2_success)
        cap = route_curve.cost_cap([everything])
        block.update({"cost_cap_per_100_rmb": cap, "within_cap": route_curve.within(block, cap)})
    if len(sets) == 1:
        (_, set_digest, cloud_model, digests), = read
        block.update({"set": relative(sets[0].set), "set_sha256": set_digest,
                      "inputs": inputs_of(sets[0], digests)})
    else:
        cloud_model = read[0][2]
        block["sets"] = [{"path": relative(inputs.set), "sha256": set_digest, "n": len(set_rows),
                          "inputs": inputs_of(inputs, digests)}
                         for inputs, (set_rows, set_digest, _, digests) in zip(sets, read, strict=True)]
    block.update({"adapter_sha256": routing["adapter_sha256"], "cloud_model": cloud_model})
    name = JOIN.join(inputs.set.stem for inputs in sets)
    if args.out.resolve() == args.routing.resolve():
        existing = routing.get("validation")
        if existing is not None and not isinstance(existing, dict):
            print(f"FAIL: malformed input: {args.routing.as_posix()} has a validation field that is not an "
                  "object", file=sys.stderr)
            return 1
        if existing and name in existing and not args.force:
            print(f"FAIL: {args.routing} already has a validation block for {name}; --force replaces it",
                  file=sys.stderr)
            return 3
        data = {**routing, "validation": {**(existing or {}), name: block}}
    else:
        if args.out.exists() and not args.force:
            print(f"FAIL: {args.out} exists; --force overwrites it", file=sys.stderr)
            return 3
        data = {"routing": {"path": relative(args.routing), "sha256": sha256_of(args.routing)},
                "validation": {name: block}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                        newline="\n")
    print(f"{policy or 'the frozen policy'} at threshold {threshold:.6g} on {name} ({block['n']} items): "
          f"routed {block['routed']}, success {block['success']:.4f} (FT {block['ft_success']:.4f}, "
          f"B2 {block['b2_success']:.4f}), {block['cost_per_100_rmb']:.4f} RMB per 100 items; "
          f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

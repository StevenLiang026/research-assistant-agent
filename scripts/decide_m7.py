"""Decide, once, whether M7's final local system keeps its vote and its runtime rules (docs/m7-本地提升计划.md
WP8, steps 1 and 2, for system A; docs/m7-预登记.md, revision 2), on the held-out sets and from aggregates
alone.

    .venv/Scripts/python.exe scripts/decide_m7.py --config results/test-config-m7.json \\
        --runs data/eval/dev-ext.jsonl <greedy run> <run of sample 0> ... <run of sample K-1> \\
        --runs data/eval/v2/dev-v2-ext.jsonl <greedy run> <run of sample 0> ...

The system. The configuration's final system as scripts/freeze_final.py reads it (its frozen file need not
exist yet: run_test.check_config's freezing): its adapter, base model and family, its generation with K
votes (systems.final.votes, required: the first decision is the vote's) and the vote mode; without the
glossary's note (system A's).

The runs. Per held-out set (HELD_OUT_SETS: data/eval/dev-ext.jsonl, then data/eval/v2/dev-v2-ext.jsonl; both
required), the final adapter's greedy run and its K sample runs in sample order, the k-th the run with
--seed k, checked as scripts/freeze_final.py checks the development runs: their manifests against the
configuration, every item run once, no item in both sets.

The aggregates, over both sets together, under the normalized scorer (the primary one) and the default one
beside it (eval.metrics.SCORERS): greedy, the greedy records' successes; greedy_rules, with the runtime rules
(agent.rules.ruled_record); vote, the vote over the samples without the rules (agent.vote.voted_record in the
vote mode); vote_rules, with the rules applied to every candidate first (the final system's records,
freeze_final.final_records); rules_fixes and rules_breaks, the greedy records the rules turn from wrong to
right, and from right to wrong.

The decisions (revision 2), under the normalized scorer: step 1, the vote is kept unless vote_rules is below
greedy_rules; step 2, the rules are kept unless rules_fixes is below rules_breaks. The final system is then
the records of the configuration they choose (FINAL_RECORDS: vote_rules, vote, greedy_rules or greedy),
counted per subset (eval.report.SUBSETS: drafted and handwritten items, in_subset) under both scorers: the
held-out parts of revision 2's prediction.

Aggregates only (the owner's rule: the held-out records give totals alone). The script prints the number of
items per set, the totals, the two decisions and the final system's counts per subset, and writes OUT,
results/decision-m7.json in this repository (JSON, two-space indent, LF): the configuration's path and
SHA-256, the system, the sets and the runs (the files' paths and SHA-256), the scorers, the totals, the
decisions and the final system's counts; never an item's id or anything of a single item, in an error
either (a run is refused by its counts; an error the scorer raises is not shown). The held-out sets are
scored once:
the decision has this one place, never another and never relative to the working folder, it is never
overwritten, and its presence refuses the run before anything is read.

Exit 0 when decided and written; 1 on a malformed configuration or run; 2 on a usage error, an existing
decision, a missing file and every disagreement of a manifest or a run; 4 when a set is test material; 5 when
a set does not match its lock. Nothing is written but on exit 0.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path

from research_agent.agent.rules import ruled_record
from research_agent.eval.lock import sha256_of
from research_agent.eval.metrics import NORMALIZED_SCORER, SCORERS, score_items
from research_agent.eval.report import SUBSETS, in_subset
from research_agent.family.core import LoadedFamily, load_family

REPO = Path(__file__).resolve().parents[1]


def _load_freeze_final():
    """scripts/freeze_final.py as a module of its own name (scripts/ is no package): the final system, the
    checks of its runs and its records, as the frozen file is made."""
    spec = importlib.util.spec_from_file_location("_freeze_final_shared",
                                                  REPO / "scripts" / "freeze_final.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


freeze_final = _load_freeze_final()
run_test = freeze_final.run_test
FINAL = run_test.FINAL
HELD_OUT_SETS = ("data/eval/dev-ext.jsonl", "data/eval/v2/dev-v2-ext.jsonl")  # in this order
OUT = REPO / "results" / "decision-m7.json"  # the decision's one place (module docstring)
PRIMARY = NORMALIZED_SCORER  # the decisions' scorer (revision 2)
RECORDS = ("greedy", "greedy_rules", "vote", "vote_rules")
COUNTS = (*RECORDS, "rules_fixes", "rules_breaks")
# the records of the final system, by (vote kept, rules kept)
FINAL_RECORDS = {(True, True): "vote_rules", (True, False): "vote", (False, True): "greedy_rules",
                 (False, False): "greedy"}
LABELS = {"greedy": "greedy", "greedy_rules": "greedy + rules", "vote": "vote", "vote_rules": "vote + rules",
          "rules_fixes": "rules fix", "rules_breaks": "rules break"}
VOTE_RULE = ("step 1: the vote is kept unless, under the normalized scorer over both held-out sets, the vote "
             "with the rules gets fewer items right than the greedy answers with the rules")
RULES_RULE = ("step 2: the rules are kept unless, under the normalized scorer over both held-out sets, they "
              "turn fewer greedy answers from wrong to right than from right to wrong")
ABOUT = ("WP8 steps 1 and 2 for system A (docs/m7-本地提升计划.md; docs/m7-预登记.md, revision 2), "
         "decided once on the held-out sets by scripts/decide_m7.py from aggregates alone: whether M7's "
         "final system keeps its vote and its runtime rules.")


def held_out_runs(given: list[list[str]] | None) -> dict[str, list[str]]:
    """The runs per held-out set, in HELD_OUT_SETS' order; Disagreement for another set or a set without
    runs."""
    runs: dict[str, list[str]] = {}
    for path, *paths in given or []:
        if path not in HELD_OUT_SETS:
            raise freeze_final.Disagreement(f"--runs {path}: {path} is not one of the held-out sets "
                                            f"({', '.join(HELD_OUT_SETS)})")
        if path in runs:
            raise freeze_final.Disagreement(f"--runs {path} is given twice")
        runs[path] = paths
    missing = [path for path in HELD_OUT_SETS if path not in runs]
    if missing:
        raise freeze_final.Disagreement(f"no runs of {', '.join(missing)} (--runs <held-out set> <greedy "
                                        "run> <sample runs>...)")
    return {path: runs[path] for path in HELD_OUT_SETS}


def counted(items: list[dict], greedy: list[dict], samples: list[list[dict]], mode: str) -> dict:
    """Per scorer the counts of COUNTS over one set and, under "subsets", each record kind's right answers per
    subset (module docstring: The aggregates)."""
    records = {"greedy": greedy,
               "greedy_rules": [ruled_record(record, item["messages"]) for record, item in
                                zip(greedy, items, strict=True)],
               "vote": freeze_final.final_records(items, greedy, samples, mode, rules=False),
               "vote_rules": freeze_final.final_records(items, greedy, samples, mode, rules=True)}
    out = {}
    for scorer in SCORERS:
        right = {name: [bool(score["success"]) for score in score_items(rows, items, scorer)]
                 for name, rows in records.items()}
        pairs = list(zip(right["greedy"], right["greedy_rules"], strict=True))
        out[scorer] = {**{name: sum(values) for name, values in right.items()},
                       "rules_fixes": sum(not before and after for before, after in pairs),
                       "rules_breaks": sum(before and not after for before, after in pairs),
                       "subsets": {name: {subset: sum(ok for ok, item in zip(values, items, strict=True)
                                                      if in_subset(item, subset)) for subset in SUBSETS}
                                   for name, values in right.items()}}
    out["n"] = {subset: sum(in_subset(item, subset) for item in items) for subset in SUBSETS}
    return out


def decide(totals: dict) -> dict:
    """The two decisions of revision 2 over the totals (module docstring: The decisions)."""
    primary = totals[PRIMARY]
    return {"vote": {"rule": VOTE_RULE, "kept": not primary["vote_rules"] < primary["greedy_rules"],
                     "vote_rules": primary["vote_rules"], "greedy_rules": primary["greedy_rules"]},
            "rules": {"rule": RULES_RULE, "kept": not primary["rules_fixes"] < primary["rules_breaks"],
                      "fixes": primary["rules_fixes"], "breaks": primary["rules_breaks"]}}


def final_counts(decisions: dict, per_set: list[dict]) -> dict:
    """The final system the decisions choose, its record kind, and its right answers per subset over both sets
    under every scorer (module docstring: The decisions)."""
    vote, rules = decisions["vote"]["kept"], decisions["rules"]["kept"]
    kind = FINAL_RECORDS[(vote, rules)]
    return {"vote": vote, "rules": rules, "records": kind,
            "n": {subset: sum(counts["n"][subset] for counts in per_set) for subset in SUBSETS},
            "right": {scorer: {subset: sum(counts[scorer]["subsets"][kind][subset] for counts in per_set)
                               for subset in SUBSETS} for scorer in SCORERS}}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Decide once, on the held-out sets, whether M7's final system "
                                             "keeps its vote and its runtime rules.")
    ap.add_argument("--config", type=Path, required=True,
                    help="the configuration, results/test-config-m7.json")
    ap.add_argument("--runs", nargs="+", action="append", default=None, metavar="PATH",
                    help="a held-out set, then the final adapter's greedy run and its sample runs in sample "
                         "order (once per held-out set)")
    return ap


def main(argv: list[str] | None = None, *, load_family: Callable[[str], LoadedFamily] = load_family) -> int:
    """load_family stands in for research_agent.family.load_family in tests."""
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    if OUT.exists():
        print(f"FAIL: {OUT.as_posix()} exists: the held-out sets are scored once "
              "(docs/m7-本地提升计划.md WP8), and nothing was read")
        return 2
    try:
        config = run_test.load_config(args.config, freezing=True)
        options = run_test.options_of(config, FINAL)
        if FINAL in config["systems"] and not options.get(run_test.VOTES):
            raise freeze_final.Disagreement(f"systems.{FINAL} has no votes: the first decision is whether "
                                            "its vote stays")
        if options.get(run_test.GLOSSARY) is not None:
            raise freeze_final.Disagreement(f"systems.{FINAL} notes a glossary: the decision is system A's, "
                                            "which notes none")
        system = freeze_final.final_system(config, load_family)
        runs = held_out_runs(args.runs)
        freeze_final.check_counts(system, runs)
        sets = {path: freeze_final.open_set(Path(path)) for path in runs}
        freeze_final.check_distinct(sets)
        records = {path: freeze_final.candidate_runs(system, Path(path), sets[path], paths)
                   for path, paths in runs.items()}
    except (run_test.ConfigError, freeze_final.Malformed) as exc:
        print(f"FAIL: {exc}")
        return 1
    except (freeze_final.Disagreement, run_test.Missing) as exc:
        print(f"FAIL: {exc}")
        return 2
    except freeze_final.Refused as exc:
        print(f"FAIL: {exc}")
        return exc.code
    except OSError as exc:
        print(f"FAIL: {exc.filename or 'a file'} cannot be read: {type(exc).__name__}")
        return 2
    try:
        per_set = [counted(sets[path], rows[0], rows[1:], system.vote_mode) for path, rows in records.items()]
    except Exception:  # noqa: BLE001  deliberate: its text could name an item, which this script never prints
        print("FAIL: the records cannot be scored against the items (the error is not shown: it could name "
              "an item)")
        return 1
    totals = {"items": sum(len(rows) for rows in sets.values()),
              **{scorer: {name: sum(counts[scorer][name] for counts in per_set) for name in COUNTS}
                 for scorer in SCORERS}}
    decisions = decide(totals)
    final = final_counts(decisions, per_set)
    adapter = config["adapters"][run_test.SERVER_OF[FINAL]]
    decision = {"about": ABOUT, "config": {"path": args.config.as_posix(), "sha256": sha256_of(args.config)},
                "system": {"adapter": {"path": adapter["gguf"], "sha256": adapter["sha256"]},
                           "base": run_test.base_model(config, run_test.SERVER_OF[FINAL]),
                           "tokenizer": run_test.adapter_tokenizer(config, run_test.SERVER_OF[FINAL]),
                           "family": system.family.record(), "generation": system.final.record(),
                           "votes": system.votes, "vote_mode": system.vote_mode},
                "sets": [{"path": path, "sha256": sha256_of(path), "n": len(rows)}
                         for path, rows in sets.items()],
                "runs": [{"set": path, "greedy": {"path": paths[0], "sha256": sha256_of(paths[0])},
                          "samples": [{"path": run, "sha256": sha256_of(run)} for run in paths[1:]]}
                         for path, paths in runs.items()],
                "scorers": list(SCORERS), "scorer": PRIMARY, "totals": totals, "decisions": decisions,
                "final": final}
    run_test.write_json(decision, OUT)
    print("held-out sets: " + ", ".join(f"{path} {len(rows)} items" for path, rows in sets.items())
          + f" ({totals['items']} in all); a vote of {system.votes} samples in the {system.vote_mode} mode")
    print(f"  {'':<16}" + "".join(f"{scorer:>12}" for scorer in SCORERS))
    for name in COUNTS:
        print(f"  {LABELS[name]:<16}" + "".join(f"{totals[scorer][name]:>12}" for scorer in SCORERS))
    vote, rules = decisions["vote"], decisions["rules"]
    print(f"step 1, the vote: {'kept' if vote['kept'] else 'dropped'} (vote + rules {vote['vote_rules']}, "
          f"greedy + rules {vote['greedy_rules']}; {PRIMARY})")
    print(f"step 2, the rules: {'kept' if rules['kept'] else 'dropped'} (fixes {rules['fixes']}, breaks "
          f"{rules['breaks']} on the greedy records; {PRIMARY})")
    for scorer in (PRIMARY, *(s for s in SCORERS if s != PRIMARY)):
        parts = [f"{subset} {final['right'][scorer][subset]}/{final['n'][subset]}" for subset in SUBSETS]
        print(f"the final system ({LABELS[final['records']]}): {', '.join(parts)} ({scorer})")
    print(f"wrote {OUT.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

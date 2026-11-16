"""scripts/build_report.py and research_agent.eval.report: results/numbers.json rebuilt from the registered
result files alone (docs: the M3 plan, phase 4). The repository's own numbers come back byte for byte, every
kind of result gives its keys (M5's routed system and its comparisons against M3's records among them), what
does not apply is left out, files no pattern names are not read, a malformed file and two files giving the
same key are refused with nothing written, and the same files give the same bytes."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
from support.fakes import call, turn

from research_agent.canonical import canonical_json
from research_agent.eval import bakeoff as bakeoff_module
from research_agent.eval import report
from research_agent.eval.e2e import CHECKS
from research_agent.eval.e2e import summarize as summarize_tasks
from research_agent.eval.metrics import (
    ERROR_CLASSES,
    compare_systems,
    holm,
    score_items,
    summarize,
    wilson_interval,
)
from research_agent.eval.numbers import NUMBERS, check

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


build_report = load_script("build_report")

SYSTEM = {"role": "system", "content": "You are a research assistant."}
USER = {"role": "user", "content": "找 2026 年 9 月以来 speculative decoding 的论文。"}
SEARCH = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-01")
WRONG = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-08-01")


def item(id_: str, type_: str, *calls, lang: str = "zh", handwritten: bool = False) -> dict:
    source = {"handwritten": True} if handwritten else {"request": id_.rsplit("-s", 1)[0], "step": 1}
    expected = {"content": "", "tool_calls": [{"name": c.name, "arguments": c.arguments,
                                               "arguments_json": c.arguments_json} for c in calls]}
    return {"id": id_, "messages": [SYSTEM, USER], "expected": expected, "type": type_, "lang": lang,
            "family": f"f/{type_}", "split": "dev",
            "slots": {"topic": "speculative decoding", "date_from": "2026-09-01"}, "source": source}


ITEMS = [item("r1-s1", "call", SEARCH), item("r2-s1", "call", SEARCH, lang="en"),
         item("hw-dev-001", "no_call", handwritten=True)]


def scores(right: set[str]) -> list[dict]:
    """The scores of a run that answers the items in right as expected and the others wrongly."""
    records = []
    for it in ITEMS:
        if it["type"] == "call":
            output = turn(SEARCH if it["id"] in right else WRONG)
        else:
            output = turn(content="好的。") if it["id"] in right else turn(SEARCH)
        records.append({"id": it["id"], "status": "ok", "item": {}, "output": output.to_api()})
    return score_items(records, ITEMS)


def single_step(right: set[str], *, subsets: bool = False) -> dict:
    """A summary file: scripts/score.py's form, or run_test's with the drafted and handwritten subsets."""
    s = scores(right)
    data = {"summary": summarize(s, ITEMS), "scores": s}
    if subsets:
        data["subsets"] = {}
        for subset in report.SUBSETS:
            chosen = [i for i in ITEMS if report.in_subset(i, subset)]
            ids = {i["id"] for i in chosen}
            data["subsets"][subset] = summarize([x for x in s if x["id"] in ids], chosen)
    return data


def task(id_: str, intent: str, lang: str, success: bool) -> dict:
    """An end-to-end task record (eval.e2e.task_record's keys); a failed one fails the tools check only."""
    return {"id": id_, "intent": intent, "lang": lang, "split": "test", "status": "answered",
            "success": success, "checks": {c: {"ok": success or c != "tools"} for c in CHECKS}, "steps": 3,
            "tool_calls": {"get_paper": 1}, "tool_errors": {} if success else {"NOT_FOUND": 1},
            "tokens": {"prompt_tokens": 900, "completion_tokens": 40}, "cost_rmb": 0.0, "paid_calls": 0,
            "cache_hits": 0, "local_calls": 3, "transcript": f"transcripts/{id_}.json", "error": None}


def comparisons() -> dict:
    a, b = scores({"r1-s1", "r2-s1", "hw-dev-001"}), scores({"r1-s1"})
    tested = compare_systems(a, b, ITEMS, n=200)
    p = [tested["mcnemar"]["p"], 1.0]
    adjusted = holm(p)
    deferred = {"name": "ftr-vs-b2", "a": "ft+r", "b": "b2", "deferred": "FT+R is built in M4", "p": 1.0,
                "p_holm": adjusted[1]}
    return {"comparisons": [{"name": "ft-vs-b0", "a": "ft", "b": "b0", **tested, "p": p[0],
                             "p_holm": adjusted[0]}, deferred]}


RESOURCES = {
    "systems": {
        "ft": {"single_step": {"basis": "calls", "n": 3, "p50_ms": 812.4, "p95_ms": 1490.2,
                               "server": {"n": 3, "p50_ms": 700.0, "p95_ms": 1388.6}},
               "e2e_step": {"basis": "steps", "n": 4, "p50_ms": 905.0, "p95_ms": 2004.9}, "cost": None},
        "b2": {"single_step": {"basis": "paid calls", "n": 2, "p50_ms": 3100.0, "p95_ms": 5200.4},
               "e2e_step": {"basis": "steps whose calls were all paid", "n": 0, "p50_ms": None,
                            "p95_ms": None},
               "cost": {"paid_rmb": 0.0123, "list_rmb": 0.98765, "ledger_rmb": 0.0123,
                        "per_task_fen": 0.0625,
                        "single_step": {"turns": 3, "paid": 2, "cached": 1, "paid_rmb": 0.0082,
                                        "list_rmb": 0.0123},
                        "e2e": {"tasks": 2, "paid_calls": 1, "cache_hits": 0, "paid_rmb": 0.0041,
                                "list_rmb": 0.0041}}},
    },
    "servers": {"ft": {"systems": ["ft", "b0", "b1"], "baseline_mib": 512.0, "peak_mib": 4612.0,
                       "vram_mib": 4100.0, "samples": 10, "power_mean_w": 85.24, "power_peak_w": 120.0,
                       "energy_wh": 2.25}},
}
SELECTION = {"run": "runs/train/ft", "system": "ft",
             "checkpoints": [{"step": 209, "n": 150, "successes": 70, "rate": 70 / 150},
                             {"step": 418, "n": 150, "successes": 78, "rate": 0.52}],
             "chosen": {"step": 418, "gguf": "runs/train/ft/gguf/checkpoint-418-f16.gguf",
                        "sha256": "0" * 64, "n": 150, "successes": 78, "rate": 0.52}}
BFCL = {"systems": {"b0": {"categories": {"simple_python": {"n": 400, "correct": 301,
                                                            "accuracy": 301 / 400}},
                           "aggregate": {"categories": 1, "n": 400, "accuracy": 301 / 400}}}}
BACKEND = {"n": 150, "runs": {"hf": {"success_rate": 0.54}, "llama.cpp": {"success_rate": 0.52}},
           "agreement": {"agree": 141, "n": 150, "rate": 0.94},
           "success_difference": {"diff": 0.02, "low": -0.0133, "high": 0.0533}, "mcnemar": {"p": 0.42}}


# scripts/route_cost_oracle.py's oracle at a cost: 39 gains, all of them routed at 0.133 per 100 items
COST_ORACLE = {"rule": "the most items right at a cost", "routing": {"path": "results/routing-m4.json",
                                                                  "sha256": "0" * 64},
               "inputs": {}, "cloud_model": "deepseek-flash", "n": 150, "ft_successes": 99, "gains": 39,
               "b2_cost_per_100_rmb": 0.462996,
               "curve": [{"routed": j, "cost_per_100_rmb": 0.13297333 * j / 39, "successes": 99 + j,
                          "success": (99 + j) / 150} for j in range(40)]}


def routing(*, met: bool = True) -> dict:
    """A routing configuration as scripts/route_curve.py writes it; without met, the rule is not met: no
    threshold, no chosen point, no handwritten point and an AUROC over no pair."""
    point = {"label": "threshold", "threshold": 0.689404839052033, "routed": 94, "api_share": 94 / 150,
             "successes": 132, "success": 0.88, "cost_per_100_rmb": 0.322808, "random_success": 0.8104}
    hand = {"label": "threshold", "threshold": 0.689404839052033, "routed": 31, "api_share": 31 / 47,
            "successes": 34, "success": 34 / 47, "cost_per_100_rmb": 0.2467, "random_success": 0.63739}
    # the scans' last points, which every routing file has: routing every item, B2's alone
    everything = {"label": "route_everything", "threshold": None, "routed": 150, "api_share": 1.0,
                  "successes": 135, "success": 0.9, "cost_per_100_rmb": 0.462996, "random_success": 0.9}
    everything_hand = {"label": "route_everything", "threshold": None, "routed": 47, "api_share": 1.0,
                       "successes": 32, "success": 32 / 47, "cost_per_100_rmb": 0.376851,
                       "random_success": 32 / 47}
    oracle = [{"routed": 0, "api_share": 0.0, "successes": 99, "success": 0.66},
              {"routed": 94, "api_share": 94 / 150, "successes": 149, "success": 149 / 150}]
    return {"rule_met": met, "signal": "min_decision_prob", "threshold": point["threshold"] if met else None,
            "n": 150, "ft_success": 0.66, "b2_success": 0.9,
            "auroc": {"min_decision_prob": 0.6735987324222619, "mean_logprob": 0.49356 if met else None},
            "signal_order": ["min_decision_prob", "mean_logprob"], "chosen_point": point if met else None,
            "curves": {"routing": [everything], "oracle": oracle},
            "handwritten": {"n": 47, "ft_success": 26 / 47, "b2_success": 32 / 47, "scan": [everything_hand],
                            "at_threshold": hand if met else None},
            "latency_ms": {"local_per_call": {"n": 150, "p50_ms": 855.05, "p95_ms": 3499.975},
                           "cloud_per_paid_call": {"n": 10, "p50_ms": 741.4, "p95_ms": 901.69}}}


def fault_row(system: str, rate: float, persistent: bool, successes: int, faulted: int, recovered: int,
              extra_ms: float | None = None) -> dict:
    """A row of scripts/fault_table.py's table of 20 tasks; extra_ms, the delay's, when fetch calls were
    delayed."""
    delay = {"delayed_fetch_calls": 0 if extra_ms is None else 5, "clean_fetch_calls": 40,
             "mean_ms_delayed": None if extra_ms is None else 50.0 + extra_ms,
             "mean_ms_clean": 50.0, "extra_ms_per_call": extra_ms}
    return {"system": system, "condition": f"rate={rate:g}" + (",persistent" if persistent else ""),
            "rate": rate, "persistent": persistent, "tasks": 20, "successes": successes,
            "success_rate": successes / 20, "tasks_with_faults": faulted, "recovered": recovered,
            "recovery_rate": recovered / faulted if faulted else None, "empty_result_tasks": 1,
            "fabricated_ids_after_empty": 2, "retry_storms": 5 if persistent else 0, "retry_storm_tasks": [],
            "by_kind": {}, "delay": delay}


FAULTS = {"seed": 42, "storm_calls": 3, "inputs": [],
          "rows": [fault_row("ft", 0.0, False, 9, 0, 0), fault_row("ft", 0.2, False, 8, 11, 1, 4996.8),
                   fault_row("ft", 0.5, True, 5, 12, 0, 5067.9)]}


def ace_mean(cells: list[dict]) -> dict:
    return {"categories": len(cells), "n": sum(c["n"] for c in cells),
            "accuracy": sum(c["accuracy"] for c in cells) / len(cells)}


def ace_language(categories: dict[str, tuple[int, int]], tool_calls: int, errors: int,
                 summary: dict | None = None) -> dict:
    """A language's block of scripts/run_acebench.py's results: categories {name: (correct, n)}, grouped by
    the prefix of their names as ACEBench's category.py groups them."""
    scored = {c: {"n": n, "correct": right, "accuracy": right / n} for c, (right, n) in categories.items()}
    groups = {g: ace_mean([cell for c, cell in scored.items() if c.startswith(f"{g}_")])
              for g in ("normal", "special") if any(c.startswith(f"{g}_") for c in scored)}
    return {"generate": [], "evaluate": [], "categories": scored,
            "aggregate": ace_mean(list(scored.values())),
            "groups": groups, "format": {c: {"n": n, "tool_call": tool_calls, "acebench_format": right + 1}
                                         for c, (right, n) in categories.items()},
            "inference_errors": {c: errors for c in categories}, "acebench_summary": summary}


# ACEBench's summary row as eval_main.py writes it (object_short and object_deep swapped, as upstream does).
ACE_SUMMARY = {"Model": "local-qwen3-1.7b-base", "object_short": 0.54, "object_deep": 0.46,
               "normal_summary": 0.469, "special_summary": 0.193, "agent_summary": 0, "Summary": 0.323}


ACEBENCH = {"smoke": False, "languages": ["zh", "en"],
            "groups": {"normal": ["normal_single_turn_single_function", "normal_atom_bool"],
                       "special": ["special_incomplete"]}, "systems": {"b0": {
    "acebench_model": "local-qwen3-1.7b-base", "adapter": None, "raw": "runs/acebench/m4/b0",
    "aggregate": {"categories": 3, "n": 250, "accuracy": (0.37 + 0.4 + 0.45) / 3},
    "languages": {"zh": ace_language({"normal_single_turn_single_function": (37, 100),
                                      "special_incomplete": (20, 50)}, 0, 0, ACE_SUMMARY),
                  "en": ace_language({"normal_single_turn_single_function": (45, 100)}, 3, 2)}}}}


# ---- M5: FT+R on the test sets against B2's M3 records (scripts/run_test.py, Session) ----------------------


def m5_single_step(*, routing: bool = True) -> dict:
    """test-m5-ftr.json: a routed system's single-step file (Session.single_step) with its routing block
    (Session.routing): 2 of 3 items routed, one of them by a hard trigger."""
    data = single_step({"r1-s1", "hw-dev-001"}, subsets=True)
    if routing:
        data["routing"] = {"n": 3, "routed": 2, "api_share": 2 / 3, "hard_triggers": {"invalid_args": 1},
                           "reference_success": {"ft": 2 / 3, "b2": 1.0},
                           "random_success": (1 / 3) * (2 / 3) + (2 / 3) * 1.0,
                           "agreement": {"local": {"reference": "ft", "n": 1, "equal": 1},
                                         "routed": {"reference": "b2", "n": 2, "equal": 1}}}
    return data


def m5_e2e(*, routing: bool = True) -> dict:
    """test-m5-e2e-ftr.json: a routed system's end-to-end file (Session.end_to_end) with its routing block
    (Session.e2e_routing)."""
    tasks = [task("e2e-test-01", "lookup_paper", "en", True), task("e2e-test-02", "chit_chat", "zh", False)]
    data = {"system": "ftr", "summary": summarize_tasks(tasks), "tasks": tasks}
    if routing:
        data["routing"] = {"tasks": 2, "tasks_routed": 1, "local_calls": 5, "cloud_calls": 2,
                           "local_steps": 4, "cloud_steps": 2}
    return data


def m5_comparisons() -> dict:
    """test-m5-comparisons.json (Session.comparisons): a comparison recorded from M3's file, FT+R against B2
    tested with a non-inferiority margin, and the reference systems rescored (one with differing items)."""
    tested = {"name": "ftr-vs-b2", "a": "ftr", "b": "b2", "n": 3, "success_a": 2 / 3, "success_b": 2 / 3,
              "diff": 0.0, "paired_ci": {"diff": 0.0, "mean": 0.0, "low": -0.0123, "high": 0.0333, "n": 200,
                                         "seed": 42},
              "cluster_ci": {"diff": 0.0, "mean": 0.0, "low": -0.0456, "high": 0.05, "n": 200, "seed": 42,
                             "clusters": 2},
              "mcnemar": {"a_only": 1, "b_only": 1, "discordant": 2, "p": 1.0}, "p": 1.0,
              "noninferiority": {"margin": 0.02, "paired_low": -0.0123, "paired_above": True,
                                 "cluster_low": -0.0456, "cluster_above": False}, "p_holm": 1.0}
    recorded = {"name": "ft-vs-b0", "a": "ft", "b": "b0", "recorded": "results/test-m3-comparisons.json",
                "p": 0.125, "p_holm": 0.25}
    return {"milestone": "M5", "seed": 42, "resamples": 200, "comparisons": [recorded, tested],
            "reference": {"milestone": "M3", "systems": {
                "b2": {"records": "runs/test-m3/open-1/b2.jsonl", "sha256": "0" * 64},
                "ft": {"records": "runs/test-m3/open-1/ft.jsonl", "sha256": "1" * 64}}},
            "rescored": {"b2": {"equal": True, "differ": []},
                         "ft": {"equal": False, "differ": ["r2-s1", "hw-dev-001"]}}}


def m5_resources() -> dict:
    """test-m5-resources.json (Session.resources, Session.routed_resources): ftr's single step split into its
    local, routed and paid routed calls, and its cost over the routed records and the end-to-end tasks."""
    single = {"local": {"n": 2, "p50_ms": 250.4, "p95_ms": 251.6,
                        "server": {"n": 2, "p50_ms": 100.2, "p95_ms": 110.7}},
              "routed": {"n": 2, "p50_ms": 1250.4, "p95_ms": 1890.2},
              "routed_paid": {"n": 1, "p50_ms": 1250.4, "p95_ms": 1250.4}}
    cost = {"paid_rmb": 0.0096, "list_rmb": None, "per_task_fen": 0.24,
            "single_step": {"turns": 2, "paid": 1, "cached": 1, "paid_rmb": 0.0048, "list_rmb": 0.0096},
            "e2e": {"tasks": 2, "paid_calls": 2, "cache_hits": 0, "paid_rmb": 0.0048, "list_rmb": None},
            "ledger_rmb": 0.0096}
    return {"milestone": "M5", "systems": {"ftr": {
        "server": "ft", "single_step": single,
        "e2e_step": {"basis": "steps", "n": 4, "p50_ms": 260.0, "p95_ms": 1410.3}, "cost": cost}},
        "servers": {"ft": {"systems": ["ftr"], "baseline_mib": 512.0, "peak_mib": 4600.0, "vram_mib": 4088.0,
                           "samples": 2, "interval_ms": 200, "power_mean_w": 100.0, "power_peak_w": 120.0,
                           "energy_wh": 0.0111}}}


# ---- M6: FT2 and FT2+R2 against the records of M3 and M5 (scripts/run_test.py, docs/m6-计划.md §5) ------


def interval_of(diff: float, low: float, high: float, *, clusters: int | None = None) -> dict:
    out = {"diff": diff, "mean": diff, "low": low, "high": high, "n": 200, "seed": 42}
    return out if clusters is None else {**out, "clusters": clusters}


def m6_single_step() -> dict:
    """test-m6-ft2r.json: FT2+R2's single-step file with its routing block against FT2's own records and
    B2's M3 records."""
    data = single_step({"r1-s1", "hw-dev-001"}, subsets=True)
    data["routing"] = {"n": 3, "routed": 1, "api_share": 1 / 3, "hard_triggers": {},
                       "reference_success": {"ft2": 1 / 3, "b2": 1.0}, "random_success": 5 / 9,
                       "agreement": {"local": {"reference": "ft2", "n": 2, "equal": 2},
                                     "routed": {"reference": "b2", "n": 1, "equal": 1}}}
    return data


def m6_e2e() -> dict:
    """test-m6-e2e-ft2r.json: the end-to-end file of a routed system under stepwise routing (the routing
    block names its mode)."""
    data = m5_e2e()
    data["system"], data["e2e_routing"] = "ft2r", "stepwise"
    data["routing"] = {"tasks": 2, "tasks_routed": 1, "local_calls": 3, "cloud_calls": 1, "local_steps": 3,
                       "cloud_steps": 1}
    return data


def m6_comparisons() -> dict:
    """test-m6-comparisons.json: the three hypotheses (McNemar; a one-sided paired bootstrap; a stratified
    one over ACEBench's per-answer successes), Holm over their three p-values, a descriptive comparison
    outside the family and the three reference systems rescored."""
    mcnemar_row = {"name": "ft2-vs-ft", "a": "ft2", "b": "ft", "test": "mcnemar", "n": 3, "success_a": 1.0,
                   "success_b": 2 / 3, "diff": 1 / 3, "paired_ci": interval_of(1 / 3, 0.0, 2 / 3),
                   "cluster_ci": interval_of(1 / 3, 0.0, 1.0, clusters=2),
                   "mcnemar": {"a_only": 1, "b_only": 0, "discordant": 1, "p": 1.0}, "p": 1.0, "p_holm": 1.0}
    ni_row = {"name": "ft2r-vs-b2", "a": "ft2r", "b": "b2", "test": "noninferiority", "n": 3,
              "success_a": 2 / 3, "success_b": 2 / 3, "diff": 0.0,
              "paired_ci": interval_of(0.0, -0.0123, 0.0333),
              "cluster_ci": interval_of(0.0, -0.0456, 0.05, clusters=2),
              "mcnemar": {"a_only": 1, "b_only": 1, "discordant": 2, "p": 1.0},
              "noninferiority": {"margin": 0.02, "p_ni": 0.0362, "below": 7, "cluster_p_ni": 0.05,
                                 "cluster_below": 10},
              "p": 0.0362, "p_holm": 0.0724}
    per_stratum = [{"language": "zh", "category": "normal_atom_bool", "n": 50, "correct_a": 40,
                    "correct_b": 36, "later_candidates_a": 0, "later_candidates_b": 0}]
    ace_row = {"name": "ft2-vs-b0-acebench", "test": "acebench", "a": "runs/acebench/m6/ft",
               "b": "runs/acebench/m4/b0",
               "acebench": {"a": {"path": "runs/acebench/m6/ft", "sha256": "2" * 64, "files": 60,
                                  "items": 1940, "mean": 0.4612},
                            "b": {"path": "runs/acebench/m4/b0", "sha256": "3" * 64, "files": 60,
                                  "items": 1940, "mean": 0.4536},
                            "margin": 0.02, "strata": 30, "items": 1940, "diff": 0.0076,
                            "paired_ci": interval_of(0.0076, -0.0149, 0.0301), "p_ni": 0.012, "below": 120,
                            "per_stratum": per_stratum},
               "p": 0.012, "p_holm": 0.036}
    described = {"name": "ft2r-vs-ftr", "a": "ft2r", "b": "ftr", "descriptive": True, "n": 3,
                 "success_a": 2 / 3, "success_b": 1.0, "diff": -1 / 3,
                 "paired_ci": interval_of(-1 / 3, -2 / 3, 0.0),
                 "cluster_ci": interval_of(-1 / 3, -1.0, 0.0, clusters=2),
                 "mcnemar": {"a_only": 0, "b_only": 1, "discordant": 1}}  # the counts; no p, not even nested
    return {"milestone": "M6", "seed": 42, "resamples": 200,
            "comparisons": [mcnemar_row, ni_row, ace_row, described],
            "reference": {"milestone": "M3", "systems": {
                "ft": {"milestone": "M3", "records": "runs/test-m3/open-1/ft.jsonl", "sha256": "1" * 64},
                "b2": {"milestone": "M3", "records": "runs/test-m3/open-1/b2.jsonl", "sha256": "0" * 64},
                "ftr": {"milestone": "M5", "records": "runs/test-m5/open-2/ftr.jsonl", "sha256": "4" * 64}}},
            "rescored": {"ft": {"equal": True, "differ": []}, "b2": {"equal": True, "differ": []},
                         "ftr": {"equal": False, "differ": ["r2-s1"]}}}


def m6_resources() -> dict:
    """test-m6-resources.json: ft2 and ft2r on the server with the ft2 adapter."""
    data = m5_resources()
    routed = data["systems"].pop("ftr")
    data["milestone"] = "M6"
    data["systems"] = {"ft2": {"server": "ft2", "single_step": RESOURCES["systems"]["ft"]["single_step"],
                               "e2e_step": RESOURCES["systems"]["ft"]["e2e_step"], "cost": None},
                       "ft2r": {**routed, "server": "ft2"}}
    data["servers"] = {"ft2": {**data["servers"]["ft"], "systems": ["ft2", "ft2r"], "vram_mib": 2934.0}}
    return data


# ---- M8: the checkpoint decision (scripts/decide_m8.py; docs/m8-预登记.md, 选点) -------------------------

M8_GATES = ("PG1", "PG2", "PG3", "PG4")


def dev_b_counts(strict: int, lenient: int, exact: int, refusals: int, zh: int) -> dict:
    """A dev set B run's counts as scripts/decide_m8.py records them (its records scored again): 150 special
    items and 90 valid ones."""
    return {"special_n": 150, "strict": strict, "lenient": lenient, "valid_n": 90, "exact": exact,
            "false_refusals": refusals, "zh_altered": zh}


def m8_candidate(step: int, s0: int, s1: int, counts: dict | None = None,
                 passed: tuple[bool, ...] = ()) -> dict:
    """A candidate row of results/decision-m8.json: its two-seed sums with the runtime rules and without them
    (6 successes fewer a seed), and when it was run on dev set B, its counts and the verdicts of PG1 to PG4
    (decide_m8's gate values beside each verdict are not read: the counts give them)."""
    row = {"step": step, "gguf": f"runs/train/ft5/gguf/checkpoint-{step}-f16.gguf", "sha256": "5" * 64,
           "s0": s0, "s0_without_rules": s0 - 6, "s1": s1, "s1_without_rules": s1 - 6, "sum": s0 + s1,
           "sum_without_rules": s0 + s1 - 12, "special_dev": counts, "gates": None, "passes": None}
    if counts is not None:
        row["gates"] = {gate: {"measure": "a measure", "passed": ok} for gate, ok in zip(M8_GATES, passed,
                                                                                     strict=True)}
        row["passes"] = all(passed)
    return row


def m8_decision(*, go: bool = False) -> dict:
    """results/decision-m8.json as scripts/decide_m8.py writes it, the parts the report reads and a few
    others. NO-GO: both candidates run on dev set B, none passing PG1 to PG4 (232 ranked first: a tie in the
    sum goes to the later step). GO: 232 passes and MG holds, so 116, ranked after it, was never run on dev
    set B."""
    first = (m8_candidate(232, 234, 234, dev_b_counts(110, 120, 71, 10, 0), (True, True, True, True)) if go
             else m8_candidate(232, 234, 234, dev_b_counts(76, 94, 71, 10, 0), (False, False, True, True)))
    second = (m8_candidate(116, 230, 238) if go
              else m8_candidate(116, 230, 238, dev_b_counts(63, 85, 67, 12, 1), (False, False, True, False)))
    return {"about": "M8's choice of the FT5 checkpoint", "rules": {"MG": "sum >= FT3's - 10"},
            "scorer": "normalized",
            "selection": {"path": "results/checkpoint-selection-ft5.json", "sha256": "0" * 64,
                          "run": "runs/train/ft5"},
            "special_dev": {}, "pg1_fallback": True,
            "b0": {"special_dev": dev_b_counts(26, 103, 58, 11, 0)},
            "baseline": {"path": "results/checkpoint-selection-ft3.json", "sha256": "1" * 64, "step": 1635,
                         "gguf": "runs/train/ft3/gguf/checkpoint-1635-f16.gguf", "gguf_sha256": "2" * 64,
                         "s0": 238, "s0_without_rules": 230, "s1": 237, "s1_without_rules": 232, "sum": 475,
                         "sum_without_rules": 462, "special_dev": dev_b_counts(49, 103, 64, 11, 0)},
            "candidates": [second, first], "ranking": [232, 116], "walk": [232] if go else [232, 116],
            "chosen": ({"step": 232, "gguf": first["gguf"], "sha256": first["sha256"], "sum": 468} if go
                       else None),
            "mg": {"sum": 468, "ft3_sum": 475, "threshold": 465, "passed": True} if go else None,
            "decision": "GO" if go else "NO-GO",
            "reason": "the first to pass, MG met" if go else "no checkpoint passes PG1-PG4"}


# ---- M9: the counts of the inference wrappers, a development run's lengths, the lengths summary -----------

# A voted and ruled system's turns (runner records, or the attempts of end-to-end transcripts) as agent.vote
# and agent.rules leave them: a unanimous vote; a 3-2 vote whose chosen class is not the greedy answer's, with
# a failed candidate, and transient_retry changing the answer; a vote no sample voted on (two failed
# candidates, given as their number) whose answer both rules changed; and a turn with neither (a harness
# error).
VOTED_TURNS = [
    {"votes": {"k": 5, "mode": "final", "counts": {"a": 5}, "chosen": "a", "greedy": "a", "errors": []},
     "rules": []},
    {"votes": {"k": 5, "mode": "final", "counts": {"a": 3, "b": 2}, "chosen": "a", "greedy": "b",
               "errors": [{"error_type": "HarnessError"}]},
     "rules": [{"rule": "transient_retry", "before": {}, "after": {}}]},
    {"votes": {"k": 5, "mode": "final", "counts": {}, "chosen": None, "greedy": "c", "errors": 2},
     "rules": [{"rule": "id_repair", "before": {}, "after": {}},
               {"rule": "transient_retry", "before": {}, "after": {}}]},
    {"status": "harness_error"},
]
# What scripts/run_test.py's wrapper_counts makes of VOTED_TURNS (a test below checks it against the writer),
# and of two end-to-end attempts no sample voted on, unruled: no agreement and no rules block.
WRAPPER_BLOCKS = {"votes": {"steps": 3, "unanimous": 1, "agreement": 0.8, "chosen_not_greedy": 1,
                            "no_sample_voted": 1, "candidate_errors": 3},
                  "rules": {"steps": 3, "fired": {"transient_retry": 2, "id_repair": 1}}}
UNVOTED_ATTEMPTS = [{"votes": {"k": 5, "mode": "final", "counts": {}, "chosen": None, "greedy": None,
                               "errors": []}}] * 2
E2E_WRAPPER_BLOCKS = {"votes": {"steps": 2, "unanimous": 0, "agreement": None, "chosen_not_greedy": 0,
                                "no_sample_voted": 2, "candidate_errors": 0}}


def m7_final() -> dict:
    """test-m7-final.json: a voted and ruled system's single-step file with its wrapper counts (a copy of
    them: the cases that break a file change it)."""
    return {**single_step({"r1-s1", "hw-dev-001"}, subsets=True), **copy.deepcopy(WRAPPER_BLOCKS)}


def m7_e2e_final() -> dict:
    """test-m7-e2e-final.json: its end-to-end file, counted over its transcripts' attempts."""
    return {**m5_e2e(routing=False), "system": "final", **copy.deepcopy(E2E_WRAPPER_BLOCKS)}


FAILED_CANDIDATES = {"HarnessError": 2, "ReadTimeout": 1}  # a voted run's summary: failed candidates by type


def dev_e2e_final(*, over: int | None = 1, failed: dict | None = FAILED_CANDIDATES) -> dict:
    """dev-e2e-final.json, run_e2e's report of a voted local system: its manifest counts the local steps over
    the training length (over None: run without the context check), its summary the failed vote candidates by
    error type (failed None: a run without --votes, no such key); its two tasks made 3 and 5 local calls."""
    tasks = [task("e2e-dev-01", "lookup_paper", "en", True), task("e2e-dev-02", "chit_chat", "zh", False)]
    tasks[1]["local_calls"] = 5
    summary = summarize_tasks(tasks)
    if failed is not None:
        summary["candidate_errors"] = dict(failed)
    return {"manifest": {"system": "final", "training_length": 5120, "steps_over_training_length": over},
            "summary": summary, "tasks": tasks}


def thinking(n: int, forced: int, *values: float | None) -> dict:
    """A part's block of a lengths file: n turns, the five token figures (THINKING_TOKENS' order), forced."""
    return {"n": n, **dict(zip(report.THINKING_TOKENS, values, strict=True)), "forced": forced}


def lengths_file() -> dict:
    """results/lengths-m7.json as scripts/thinking_lengths.py writes it, the parts the report reads and a few
    others: final's thinking on its single-step records and end-to-end attempts with its steps over 5,120
    tokens, and FT2's steps over 4,096 (it keeps no thinking record)."""
    return {"milestone": "M7", "inputs": {"final": {"sha256": "6" * 64, "files": 31}},
            "systems": {"final": {"single_step": thinking(300, 12, 812.4, 1279.95, 1280, 41.0, 97.4),
                                  "e2e": thinking(1200, 30, 640.0, 1280, 1280, 52.4, 180.6),
                                  "e2e_length": {"training_length": 5120, "calls": 412, "over": 3,
                                                 "max": 6011}},
                        "ft2": {"e2e_length": {"training_length": 4096, "calls": 380, "over": 0,
                                               "max": 3998}}}}


def results(root: Path, files: dict[str, object]) -> Path:
    """root/results with the given files, JSON as the writers write it (or the text of a str)."""
    folder = root / "results"
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2) + "\n"
        (folder / name).write_text(text, encoding="utf-8", newline="\n")
    return folder


def every_kind() -> dict[str, object]:
    dev_tasks = [task("e2e-dev-01", "lookup_paper", "en", True)]
    test_tasks = [task("e2e-test-01", "lookup_paper", "en", True),
                  task("e2e-test-02", "chit_chat", "zh", False)]
    return {"dev-b0.json": single_step({"r1-s1"}),
            "dev-e2e-ft.json": {"manifest": {"system": "ft"}, "summary": summarize_tasks(dev_tasks),
                                "tasks": []},
            "test-m3-ft.json": single_step({"r1-s1", "r2-s1", "hw-dev-001"}, subsets=True),
            "test-m3-b0.json": single_step({"r1-s1"}),  # a wrong value and an unwanted call
            "test-m3-e2e-ft.json": {"summary": summarize_tasks(test_tasks)},
            "test-m3-comparisons.json": comparisons(), "test-m3-resources.json": RESOURCES,
            "checkpoint-selection-ft.json": SELECTION, "bfcl-m3.json": BFCL,
            "backend-comparison-ft-scale1.json": BACKEND, "routing-m4.json": routing(),
            "cost-oracle-m4.json": COST_ORACLE,
            "faults-m4.json": FAULTS, "acebench-m4.json": ACEBENCH,
            "test-m5-ftr.json": m5_single_step(), "test-m5-e2e-ftr.json": m5_e2e(),
            "test-m5-comparisons.json": m5_comparisons(), "test-m5-resources.json": m5_resources(),
            "test-m6-ft2.json": single_step({"r1-s1", "r2-s1", "hw-dev-001"}, subsets=True),
            "test-m6-ft2r.json": m6_single_step(), "test-m6-e2e-ft2r.json": m6_e2e(),
            "test-m6-comparisons.json": m6_comparisons(), "test-m6-resources.json": m6_resources(),
            "decision-m8.json": m8_decision(),
            "test-m7-final.json": m7_final(), "test-m7-e2e-final.json": m7_e2e_final(),
            "dev-e2e-final.json": dev_e2e_final(), "lengths-m7.json": lengths_file()}


# the access logs (data/eval) and the per-item results the report is built from (results/dev-*.json and
# others): data the public copy leaves out (tests/conftest.py)
@pytest.mark.needs_data("data/eval", "results/dev-ft.json")
def test_the_repositorys_numbers_are_rebuilt_byte_for_byte_from_its_results(tmp_path):
    out = tmp_path / "numbers.json"
    report.write(REPO, out, ignored=lambda path: False)
    assert out.read_bytes() == (REPO / NUMBERS).read_bytes()


EXPECTED = {
    "eval.dev-b0.n": "3", "eval.dev-b0.success": "33.3%", "eval.dev-b0.success.call": "50.0%",
    "eval.test-m3-ft.success": "100.0%", "eval.test-m3-ft-drafted.n": "2",
    "eval.test-m3-ft.errors.wrong_value": "0", "eval.test-m3-ft.errors.unclassified": "0",
    "eval.test-m3-b0.errors.wrong_tool": "0", "eval.test-m3-b0.errors.missing_param": "0",
    "eval.test-m3-b0.errors.extra_param": "0", "eval.test-m3-b0.errors.wrong_value": "1",
    "eval.test-m3-b0.errors.fabricated_id": "0", "eval.test-m3-b0.errors.unwanted_call": "1",
    "eval.test-m3-b0.errors.no_clarify": "0", "eval.test-m3-b0.errors.bad_recovery": "0",
    "eval.test-m3-b0.errors.retry_storm": "0", "eval.test-m3-b0.errors.broken_json": "0",
    "eval.test-m3-b0.errors.think_leak": "0", "eval.test-m3-b0.errors.unclassified": "0",
    "eval.test-m3-ft-handwritten.success": "100.0%",
    "eval.test-m3-ft-handwritten.success.no_call": "100.0%",
    "e2e.dev-ft.n": "1", "e2e.dev-ft.success": "100.0%",
    "e2e.test-m3-ft.n": "2", "e2e.test-m3-ft.success": "50.0%",
    "e2e.test-m3-ft.success.lookup_paper": "100.0%", "e2e.test-m3-ft.success.zh": "0.0%",
    "e2e.test-m3-ft.tools": "50.0%", "e2e.test-m3-ft.evidence": "100.0%",
    "e2e.test-m3-ft.steps_mean": "3.0", "e2e.test-m3-ft.tool_errors": "1",
    "compare.test-m3.ft-vs-b0.n": "3", "compare.test-m3.ft-vs-b0.success_a": "100.0%",
    "compare.test-m3.ft-vs-b0.success_b": "33.3%", "compare.test-m3.ft-vs-b0.diff": "+66.7",
    "compare.test-m3.ft-vs-b0.a_only": "2", "compare.test-m3.ft-vs-b0.b_only": "0",
    "compare.test-m3.ft-vs-b0.p": "0.500", "compare.test-m3.ft-vs-b0.p_holm": "1.000",
    "compare.test-m3.ft-vs-b0.clusters": "2",  # the two drafted items share a family; the handwritten one
    "compare.test-m3.ftr-vs-b2.p": "1.000", "compare.test-m3.ftr-vs-b2.p_holm": "1.000",
    "resources.test-m3.ft.single_step.p50_ms": "812", "resources.test-m3.ft.single_step.p95_ms": "1490",
    "resources.test-m3.ft.single_step.n": "3", "resources.test-m3.ft.single_step.server_p95_ms": "1389",
    "resources.test-m3.ft.e2e_step.p50_ms": "905", "resources.test-m3.b2.e2e_step.n": "0",
    "resources.test-m3.b2.cost.paid_rmb": "0.012", "resources.test-m3.b2.cost.list_rmb": "0.988",
    "resources.test-m3.b2.cost.per_task_fen": "0.06",
    "resources.test-m3.server.ft.vram_mib": "4100", "resources.test-m3.server.ft.baseline_mib": "512",
    "resources.test-m3.server.ft.power_mean_w": "85.2", "resources.test-m3.server.ft.energy_wh": "2.2",
    "checkpoint.ft.step": "418", "checkpoint.ft.dev_success": "52.0%", "checkpoint.ft.candidates": "2",
    "bfcl.m3.b0.aggregate": "75.25%", "bfcl.m3.b0.simple_python": "75.25%", "bfcl.m3.b0.n": "400",
    "backend.ft-scale1.agreement": "94.0%", "backend.ft-scale1.diff": "+2.0",
    "backend.ft-scale1.ci": "[-1.3, +5.3]", "backend.ft-scale1.p": "0.420",
    "cost_oracle.m4.gains": "39", "cost_oracle.m4.best_success": "92.0%",
    "cost_oracle.m4.best_cost_per_100_rmb": "0.133",
    "routing.m4.signal": "min_decision_prob", "routing.m4.threshold": "0.689",
    "routing.m4.auroc.min_decision_prob": "0.674", "routing.m4.auroc.mean_logprob": "0.494",
    "routing.m4.n": "150", "routing.m4.ft_success": "66.0%", "routing.m4.b2_success": "90.0%",
    "routing.m4.routed": "94", "routing.m4.api_share": "62.7%", "routing.m4.success": "88.0%",
    "routing.m4.random_success": "81.0%", "routing.m4.oracle_success": "99.3%",
    "routing.m4.cost_per_100_rmb": "0.323", "routing.m4.b2_cost_per_100_rmb": "0.463",
    "routing.m4.rule_met": "yes",
    "routing.m4.handwritten.n": "47", "routing.m4.handwritten.ft_success": "55.3%",
    "routing.m4.handwritten.b2_success": "68.1%", "routing.m4.handwritten.routed": "31",
    "routing.m4.handwritten.api_share": "66.0%", "routing.m4.handwritten.success": "72.3%",
    "routing.m4.handwritten.random_success": "63.7%", "routing.m4.handwritten.cost_per_100_rmb": "0.247",
    "routing.m4.handwritten.b2_cost_per_100_rmb": "0.377",
    "routing.m4.latency.local_per_call.n": "150", "routing.m4.latency.local_per_call.p50_ms": "855",
    "routing.m4.latency.local_per_call.p95_ms": "3500", "routing.m4.latency.cloud_per_paid_call.n": "10",
    "routing.m4.latency.cloud_per_paid_call.p50_ms": "741",
    "routing.m4.latency.cloud_per_paid_call.p95_ms": "902",
    "faults.m4.ft.r0.tasks": "20", "faults.m4.ft.r0.successes": "9", "faults.m4.ft.r0.success": "45.0%",
    "faults.m4.ft.r0.tasks_with_faults": "0", "faults.m4.ft.r0.recovered": "0",
    "faults.m4.ft.r0.fabricated_ids_after_empty": "2", "faults.m4.ft.r0.retry_storms": "0",
    "faults.m4.ft.r0.2.success": "40.0%", "faults.m4.ft.r0.2.tasks_with_faults": "11",
    "faults.m4.ft.r0.2.recovered": "1", "faults.m4.ft.r0.2.recovery": "9.1%",
    "faults.m4.ft.r0.2.delay_extra_ms": "4997",
    "faults.m4.ft.r0.5p.success": "25.0%", "faults.m4.ft.r0.5p.recovery": "0.0%",
    "faults.m4.ft.r0.5p.retry_storms": "5", "faults.m4.ft.r0.5p.delay_extra_ms": "5068",
    "acebench.m4.b0.aggregate": "40.67%", "acebench.m4.b0.n": "250", "acebench.m4.b0.categories": "3",
    "acebench.m4.b0.zh.aggregate": "38.50%", "acebench.m4.b0.zh.normal_single_turn_single_function": "37.00%",
    "acebench.m4.b0.zh.special_incomplete": "40.00%",
    "acebench.m4.b0.zh.format.normal_single_turn_single_function.tool_call": "0",
    "acebench.m4.b0.zh.format.normal_single_turn_single_function.acebench_format": "38",
    "acebench.m4.b0.zh.format.normal_single_turn_single_function.n": "100",
    "acebench.m4.b0.en.aggregate": "45.00%",
    "acebench.m4.b0.en.format.normal_single_turn_single_function.tool_call": "3",
    "acebench.m4.b0.en.inference_errors": "2",
    "acebench.m4.b0.zh.normal": "37.00%", "acebench.m4.b0.zh.special": "40.00%",
    "acebench.m4.b0.en.normal": "45.00%",
    "acebench.m4.b0.zh.format.normal.tool_call": "0", "acebench.m4.b0.zh.format.normal.acebench_format": "38",
    "acebench.m4.b0.zh.format.normal.n": "100", "acebench.m4.b0.zh.format.special.tool_call": "0",
    "acebench.m4.b0.zh.format.special.acebench_format": "21", "acebench.m4.b0.zh.format.special.n": "50",
    "acebench.m4.b0.en.format.normal.tool_call": "3", "acebench.m4.b0.en.format.normal.n": "100",
    "acebench.m4.b0.zh.acebench_summary.normal": "46.90%",
    "acebench.m4.b0.zh.acebench_summary.special": "19.30%",
    # M5
    "eval.test-m5-ftr.n": "3", "eval.test-m5-ftr.success": "66.7%",
    "eval.test-m5-ftr-drafted.success": "50.0%",
    "eval.test-m5-ftr.routing.n": "3", "eval.test-m5-ftr.routing.routed": "2",
    "eval.test-m5-ftr.routing.api_share": "66.7%", "eval.test-m5-ftr.routing.random_success": "88.9%",
    "eval.test-m5-ftr.routing.hard": "1", "eval.test-m5-ftr.routing.hard.invalid_args": "1",
    "eval.test-m5-ftr.routing.reference_success.ft": "66.7%",
    "eval.test-m5-ftr.routing.reference_success.b2": "100.0%",
    "eval.test-m5-ftr.routing.agreement.local.n": "1",
    "eval.test-m5-ftr.routing.agreement.local.equal": "1",
    "eval.test-m5-ftr.routing.agreement.routed.n": "2",
    "eval.test-m5-ftr.routing.agreement.routed.equal": "1",
    "e2e.test-m5-ftr.n": "2", "e2e.test-m5-ftr.success": "50.0%",
    "e2e.test-m5-ftr.routing.tasks": "2", "e2e.test-m5-ftr.routing.tasks_routed": "1",
    "e2e.test-m5-ftr.routing.local_calls": "5", "e2e.test-m5-ftr.routing.cloud_calls": "2",
    "e2e.test-m5-ftr.routing.local_steps": "4", "e2e.test-m5-ftr.routing.cloud_steps": "2",
    "compare.test-m5.ft-vs-b0.p": "0.125", "compare.test-m5.ft-vs-b0.p_holm": "0.250",
    "compare.test-m5.ftr-vs-b2.n": "3", "compare.test-m5.ftr-vs-b2.diff": "+0.0",
    "compare.test-m5.ftr-vs-b2.ci": "[-1.2, +3.3]", "compare.test-m5.ftr-vs-b2.cluster_ci": "[-4.6, +5.0]",
    "compare.test-m5.ftr-vs-b2.p": "1.000", "compare.test-m5.ftr-vs-b2.p_holm": "1.000",
    "compare.test-m5.ftr-vs-b2.noninferiority.margin": "2.0",
    "compare.test-m5.ftr-vs-b2.noninferiority.paired_low": "-1.2",
    "compare.test-m5.ftr-vs-b2.noninferiority.paired_above": "yes",
    "compare.test-m5.ftr-vs-b2.noninferiority.cluster_low": "-4.6",
    "compare.test-m5.ftr-vs-b2.noninferiority.cluster_above": "no",
    "compare.test-m5.rescored.b2.equal": "yes", "compare.test-m5.rescored.b2.differ": "0",
    "compare.test-m5.rescored.ft.equal": "no", "compare.test-m5.rescored.ft.differ": "2",
    "resources.test-m5.ftr.single_step.local.n": "2", "resources.test-m5.ftr.single_step.local.p50_ms": "250",
    "resources.test-m5.ftr.single_step.local.p95_ms": "252",
    "resources.test-m5.ftr.single_step.local.server_p50_ms": "100",
    "resources.test-m5.ftr.single_step.local.server_p95_ms": "111",
    "resources.test-m5.ftr.single_step.routed.n": "2",
    "resources.test-m5.ftr.single_step.routed.p50_ms": "1250",
    "resources.test-m5.ftr.single_step.routed.p95_ms": "1890",
    "resources.test-m5.ftr.single_step.routed_paid.n": "1",
    "resources.test-m5.ftr.single_step.routed_paid.p50_ms": "1250",
    "resources.test-m5.ftr.single_step.routed_paid.p95_ms": "1250",
    "resources.test-m5.ftr.e2e_step.n": "4", "resources.test-m5.ftr.e2e_step.p50_ms": "260",
    "resources.test-m5.ftr.e2e_step.p95_ms": "1410",
    "resources.test-m5.ftr.cost.paid_rmb": "0.010", "resources.test-m5.ftr.cost.ledger_rmb": "0.010",
    "resources.test-m5.ftr.cost.per_task_fen": "0.24",
    "resources.test-m5.ftr.cost.single_step.turns": "2", "resources.test-m5.ftr.cost.single_step.paid": "1",
    "resources.test-m5.ftr.cost.single_step.cached": "1",
    "resources.test-m5.ftr.cost.single_step.paid_rmb": "0.005",
    "resources.test-m5.ftr.cost.single_step.list_rmb": "0.010",
    "resources.test-m5.ftr.cost.e2e.tasks": "2", "resources.test-m5.ftr.cost.e2e.paid_calls": "2",
    "resources.test-m5.ftr.cost.e2e.cache_hits": "0", "resources.test-m5.ftr.cost.e2e.paid_rmb": "0.005",
    "resources.test-m5.server.ft.vram_mib": "4088",
    # M6
    "eval.test-m6-ft2.success": "100.0%", "eval.test-m6-ft2-handwritten.n": "1",
    "eval.test-m6-ft2r.success": "66.7%", "eval.test-m6-ft2r.routing.routed": "1",
    "eval.test-m6-ft2r.routing.reference_success.ft2": "33.3%",
    "eval.test-m6-ft2r.routing.agreement.local.equal": "2",
    "e2e.test-m6-ft2r.routing.mode": "stepwise", "e2e.test-m6-ft2r.routing.cloud_steps": "1",
    "compare.test-m6.ft2-vs-ft.p": "1.000", "compare.test-m6.ft2-vs-ft.p_holm": "1.000",
    "compare.test-m6.ft2-vs-ft.diff": "+33.3", "compare.test-m6.ft2-vs-ft.a_only": "1",
    "compare.test-m6.ft2r-vs-b2.p_ni": "0.036", "compare.test-m6.ft2r-vs-b2.p_holm": "0.072",
    "compare.test-m6.ft2r-vs-b2.cluster_p_ni": "0.050", "compare.test-m6.ft2r-vs-b2.margin": "2.0",
    "compare.test-m6.ft2r-vs-b2.n": "3", "compare.test-m6.ft2r-vs-b2.diff": "+0.0",
    "compare.test-m6.ft2r-vs-b2.ci": "[-1.2, +3.3]", "compare.test-m6.ft2r-vs-b2.cluster_ci": "[-4.6, +5.0]",
    "compare.test-m6.ft2r-vs-b2.clusters": "2", "compare.test-m6.ft2r-vs-b2.a_only": "1",
    "compare.test-m6.ft2r-vs-b2.success_a": "66.7%",
    "compare.test-m6.ft2-vs-b0-acebench.p_ni": "0.012", "compare.test-m6.ft2-vs-b0-acebench.p_holm": "0.036",
    "compare.test-m6.ft2-vs-b0-acebench.diff": "+0.76", "compare.test-m6.ft2-vs-b0-acebench.margin": "2.0",
    "compare.test-m6.ft2-vs-b0-acebench.ci": "[-1.49, +3.01]",
    "compare.test-m6.ft2-vs-b0-acebench.strata": "30", "compare.test-m6.ft2-vs-b0-acebench.n": "1940",
    "compare.test-m6.ft2-vs-b0-acebench.mean_a": "46.12%",
    "compare.test-m6.ft2-vs-b0-acebench.mean_b": "45.36%",
    "compare.test-m6.ft2r-vs-ftr.diff": "-33.3", "compare.test-m6.ft2r-vs-ftr.ci": "[-66.7, +0.0]",
    "compare.test-m6.ft2r-vs-ftr.b_only": "1",
    "compare.test-m6.rescored.ftr.equal": "no", "compare.test-m6.rescored.ftr.differ": "1",
    "compare.test-m6.rescored.ft.equal": "yes",
    "resources.test-m6.ft2.single_step.p50_ms": "812", "resources.test-m6.ft2r.single_step.local.n": "2",
    "resources.test-m6.ft2r.cost.single_step.cached": "1", "resources.test-m6.server.ft2.vram_mib": "2934",
    # M8's checkpoint decision
    "decision.m8.decision": "NO-GO", "decision.m8.ft3.sum": "475", "decision.m8.b0.lenient_rate": "68.7%",
    "decision.m8.ft5-232.rank": "1", "decision.m8.ft5-232.strict_rate": "50.7%",
    "decision.m8.ft5-116.pg4": "no",
    # M9: the inference wrappers' counts, a development run's lengths, the lengths summary
    "eval.test-m7-final.votes.steps": "3", "eval.test-m7-final.votes.agreement": "80.0%",
    "eval.test-m7-final.rules.fired.transient_retry": "2", "eval.test-m7-final.success": "66.7%",
    "e2e.test-m7-final.votes.no_sample_voted": "2", "e2e.test-m7-final.n": "2",
    "e2e.dev-final.training_length": "5120", "e2e.dev-final.steps_over_training_length": "1",
    "e2e.dev-final.local_calls": "8", "e2e.dev-final.candidate_errors": "3",
    "lengths.m7.final.single_step.reasoning_p50": "812", "lengths.m7.final.e2e_length.over": "3",
    "lengths.m7.ft2.e2e_length.training_length": "4096",
}


def test_every_kind_of_result_gives_its_keys_in_its_format(tmp_path):
    results(tmp_path, every_kind())
    numbers, sources = report.build(tmp_path)
    assert [p.as_posix() for p in sources] == sorted(f"results/{name}" for name in every_kind())
    assert {key: numbers.get(key) for key in EXPECTED} == EXPECTED
    assert "resources.test-m3.b2.e2e_step.p50_ms" not in numbers  # no paid step: no percentile
    assert "compare.test-m3.ftr-vs-b2.n" not in numbers  # deferred: its p only
    low, high = numbers["compare.test-m3.ft-vs-b0.ci"][1:-1].split(", ")
    assert float(low) <= 66.7 <= float(high)
    assert "faults.m4.ft.r0.recovery" not in numbers  # no task met a fault: no recovery rate
    assert "faults.m4.ft.r0.delay_extra_ms" not in numbers  # no delayed fetch call
    assert "acebench.m4.b0.zh.inference_errors" not in numbers  # none: left out
    assert "acebench.m4.b0.en.special" not in numbers  # no special category run in English
    assert "acebench.m4.b0.en.format.special.n" not in numbers
    assert "acebench.m4.b0.en.acebench_summary.normal" not in numbers  # no summary row
    assert not [key for key in numbers if key.startswith("acebench.m4.b0.zh.acebench_summary.object")]
    # M5: a recorded comparison gives its p only; the routed system's parts replace a single latency record;
    # its list price is null and its end-to-end list price is not taken; M3's numbers are as they were
    assert not [key for key in numbers if key.startswith("compare.test-m5.ft-vs-b0.")
                and key not in ("compare.test-m5.ft-vs-b0.p", "compare.test-m5.ft-vs-b0.p_holm")]
    assert "resources.test-m5.ftr.single_step.n" not in numbers
    assert "resources.test-m5.ftr.cost.list_rmb" not in numbers
    assert "resources.test-m5.ftr.cost.e2e.list_rmb" not in numbers
    assert "resources.test-m5.ftr.single_step.routed.server_p50_ms" not in numbers
    assert not [key for key in numbers if key.startswith(("resources.test-m3.b2.cost.single_step",
                                                          "resources.test-m3.b2.cost.e2e"))]
    assert not [key for key in numbers if ".routing." in key and "m5" not in key and "m6" not in key]
    # M6: a non-inferiority comparison gives its one-sided p, not McNemar's; the ACEBench comparison has
    # no clusters; a descriptive comparison has no p at all, since it is outside the Holm family
    assert "compare.test-m6.ft2r-vs-b2.p" not in numbers
    assert not [key for key in numbers if key.startswith("compare.test-m6.ft2r-vs-b2.noninferiority")]
    for key in ("cluster_ci", "clusters", "a_only", "success_a", "p"):
        assert f"compare.test-m6.ft2-vs-b0-acebench.{key}" not in numbers
    for key in ("p", "p_holm", "p_ni"):
        assert f"compare.test-m6.ft2r-vs-ftr.{key}" not in numbers
    assert "e2e.test-m5-ftr.routing.mode" not in numbers  # M5 ran sticky, before the mode was recorded


def test_what_does_not_apply_to_an_m5_file_is_left_out(tmp_path):
    single = m5_single_step()
    single["routing"].update(api_share=None, random_success=None, hard_triggers={})
    single["routing"]["reference_success"]["b2"] = None
    single["routing"]["agreement"]["routed"]["equal"] = None  # b2's own records were not at hand
    resources = m5_resources()
    resources["systems"]["ftr"]["single_step"]["routed_paid"] = {"n": 0, "p50_ms": None, "p95_ms": None}
    resources["systems"]["ftr"]["single_step"]["local"].pop("server")
    resources["systems"]["ftr"]["cost"]["single_step"]["list_rmb"] = None
    comparisons = m5_comparisons()
    del comparisons["rescored"]  # a dry run's
    del comparisons["comparisons"][1]["noninferiority"]  # no margin configured
    dev_e2e = m5_e2e()  # a development run's end-to-end report of ftr: its routing block is not registered
    results(tmp_path, {"test-m5-ftr.json": single, "test-m5-resources.json": resources,
                       "test-m5-comparisons.json": comparisons, "dev-e2e-ftr.json": dev_e2e,
                       "test-m5-e2e-ftr.json": m5_e2e(routing=False), "test-m6-ftr.json": m5_single_step(
                           routing=False)})
    numbers, _ = report.build(tmp_path)
    at = "eval.test-m5-ftr.routing"
    assert numbers[f"{at}.hard"] == "0" and numbers[f"{at}.agreement.routed.n"] == "2"
    assert numbers[f"{at}.reference_success.ft"] == "66.7%"
    for key in ("api_share", "random_success", "reference_success.b2", "agreement.routed.equal"):
        assert f"{at}.{key}" not in numbers
    assert not [key for key in numbers if key.startswith(f"{at}.hard.")]
    assert numbers["resources.test-m5.ftr.single_step.routed_paid.n"] == "0"
    for key in ("routed_paid.p50_ms", "routed_paid.p95_ms", "local.server_p50_ms", "local.server_p95_ms"):
        assert f"resources.test-m5.ftr.single_step.{key}" not in numbers
    assert "resources.test-m5.ftr.cost.single_step.list_rmb" not in numbers
    assert numbers["resources.test-m5.ftr.cost.single_step.paid_rmb"] == "0.005"
    assert numbers["compare.test-m5.ftr-vs-b2.ci"] == "[-1.2, +3.3]"
    assert not [key for key in numbers if ".rescored." in key or ".noninferiority." in key]
    assert numbers["e2e.test-m5-ftr.n"] == "2" and numbers["e2e.dev-ftr.n"] == "2"
    assert not [key for key in numbers if key.startswith(("e2e.test-m5-ftr.routing", "e2e.dev-ftr.routing",
                                                          "eval.test-m6-ftr.routing"))]


def test_a_routing_configuration_whose_rule_is_not_met_gives_no_chosen_point(tmp_path):
    results(tmp_path, {"routing-m4.json": routing(met=False)})
    numbers, _ = report.build(tmp_path)
    assert numbers["routing.m4.rule_met"] == "no"
    assert numbers["routing.m4.signal"] == "min_decision_prob"
    assert numbers["routing.m4.auroc.min_decision_prob"] == "0.674"
    assert numbers["routing.m4.handwritten.ft_success"] == "55.3%"
    # routing every item costs what it costs whether or not a threshold met the rule
    assert numbers["routing.m4.b2_cost_per_100_rmb"] == "0.463"
    assert numbers["routing.m4.handwritten.b2_cost_per_100_rmb"] == "0.377"
    for key in ("threshold", "routed", "api_share", "success", "random_success", "oracle_success",
                "cost_per_100_rmb", "auroc.mean_logprob", "handwritten.routed", "handwritten.success",
                "handwritten.cost_per_100_rmb"):
        assert f"routing.m4.{key}" not in numbers


def test_an_oracle_at_a_cost_without_gains_gives_fts_success_at_no_cost(tmp_path):
    data = {**COST_ORACLE, "gains": 0, "curve": COST_ORACLE["curve"][:1]}
    results(tmp_path, {"cost-oracle-m4.json": data})
    numbers, _ = report.build(tmp_path)
    assert (numbers["cost_oracle.m4.gains"], numbers["cost_oracle.m4.best_success"],
            numbers["cost_oracle.m4.best_cost_per_100_rmb"]) == ("0", "66.0%", "0.000")


@pytest.mark.parametrize("ends", [0, 2])
def test_a_scan_without_a_single_route_everything_point_gives_no_cost_of_routing_every_item(tmp_path, ends):
    # route_curve.py writes exactly one; with none or two the cost of routing every item is not known, and
    # the key is left out (a README marker on it then fails the README check)
    data = routing()
    for scan in (data["curves"]["routing"], data["handwritten"]["scan"]):
        scan[:] = scan * ends
    results(tmp_path, {"routing-m4.json": data})
    numbers, _ = report.build(tmp_path)
    assert "routing.m4.b2_cost_per_100_rmb" not in numbers
    assert "routing.m4.handwritten.b2_cost_per_100_rmb" not in numbers
    assert numbers["routing.m4.cost_per_100_rmb"] == "0.323"  # the chosen point's own cost stays


def test_the_fault_conditions_are_named_from_rate_and_persistent_whatever_the_label(tmp_path):
    row = fault_row("b2", 0.5, False, 7, 13, 1)
    row["condition"] = "something else"  # the label is not read
    results(tmp_path, {"faults-m4.json": {"rows": [row]}})
    numbers, _ = report.build(tmp_path)
    assert numbers["faults.m4.b2.r0.5.recovery"] == "7.7%"
    assert all(key.startswith("faults.m4.b2.r0.5.") for key in numbers)


def test_files_no_pattern_names_are_not_read_and_not_sources(tmp_path):
    results(tmp_path, {"dev-b0.json": single_step({"r1-s1"}), "test-config-m3.json": "not json: never read",
                       "test-config-m3.example.json": "{", "other.json": "{broken", "notes.txt": "x",
                       "numbers.json": "{}"})
    (tmp_path / "results" / "sub").mkdir()
    (tmp_path / "results" / "sub" / "dev-b1.json").write_text("{broken", encoding="utf-8")
    numbers, sources = report.build(tmp_path)
    assert [p.as_posix() for p in sources] == ["results/dev-b0.json"]
    assert numbers and all(key.startswith("eval.dev-b0.") for key in numbers)


@pytest.mark.parametrize("name, data, why", [
    ("dev-b0.json", {"summary": {}}, "results/dev-b0.json is not a single-step summary"),
    ("bfcl-m3.json", "not json", "results/bfcl-m3.json is not valid JSON"),
    ("test-m3-comparisons.json", {"comparisons": [{"name": "ft vs b0", "deferred": "x", "p": 0.1,
                                                    "p_holm": 0.1}]},
     "results/test-m3-comparisons.json is not a comparisons file"),
    ("routing-m4.json", {"rule_met": True, "signal": "min_decision_prob"},
     "results/routing-m4.json is not a routing configuration"),
    ("routing-m4.json", {**routing(), "rule_met": "yes"},
     "results/routing-m4.json is not a routing configuration"),
    ("routing-m4.json", {**routing(), "curves": {"oracle": []}},  # no oracle point at 94 routed
     "results/routing-m4.json is not a routing configuration"),
    ("faults-m4.json", {"rows": [{"system": "ft", "rate": 0.2}]},
     "results/faults-m4.json is not a fault table"),
    ("faults-m4.json", {"rows": [{**fault_row("ft", 0.2, False, 8, 11, 1), "rate": "0.2"}]},
     "results/faults-m4.json is not a fault table"),
    ("faults-m4.json", {"rows": [fault_row("ft", 0.2, False, 8, 11, 1),
                                 fault_row("ft", 0.2, False, 7, 9, 0)]},
     "results/faults-m4.json is not a fault table"),  # the same system and condition twice
    ("acebench-m4.json", {"systems": {"b0": {"aggregate": {"n": 1}}}},
     "results/acebench-m4.json is not an ACEBench result"),
    # M5
    ("test-m5-ftr.json", {**m5_single_step(), "routing": {"n": 3, "routed": 2}},
     "results/test-m5-ftr.json is not a single-step summary"),
    ("test-m5-e2e-ftr.json", {**m5_e2e(), "routing": {"tasks": 2}},
     "results/test-m5-e2e-ftr.json is not an end-to-end summary"),
    ("test-m5-comparisons.json", {"comparisons": [{"name": "ft-vs-b0", "recorded": "results/x.json",
                                                    "p": 0.125}]},  # no p_holm
     "results/test-m5-comparisons.json is not a comparisons file"),
    ("test-m5-comparisons.json", {"comparisons": [{**m5_comparisons()["comparisons"][1], "noninferiority": {
        "margin": 0.02, "paired_low": -0.0123, "paired_above": "yes", "cluster_low": -0.0456,
        "cluster_above": False}}]},
     "results/test-m5-comparisons.json is not a comparisons file"),
    ("test-m5-comparisons.json", {**m5_comparisons(), "rescored": {"b2": {"equal": "true", "differ": []}}},
     "results/test-m5-comparisons.json is not a comparisons file"),
    ("test-m5-comparisons.json", {**m5_comparisons(), "rescored": {"b2": {"equal": False, "differ": 2}}},
     "results/test-m5-comparisons.json is not a comparisons file"),
    ("test-m5-resources.json", {"servers": {}, "systems": {"ftr": {  # routed_paid missing
        "single_step": {key: part for key, part in m5_resources()["systems"]["ftr"]["single_step"].items()
                        if key != "routed_paid"}, "e2e_step": None, "cost": None}}},
     "results/test-m5-resources.json is not a resources file"),
    ("test-m5-resources.json", {"servers": {}, "systems": {"ftr": {
        **m5_resources()["systems"]["ftr"],
        "cost": {"paid_rmb": 0.0096, "list_rmb": None, "ledger_rmb": 0.0096, "per_task_fen": 0.24}}}},
     # a routed system's cost without its single_step and e2e blocks
     "results/test-m5-resources.json is not a resources file"),
    # gains that are not the curve's points after the first, or not a count: README would quote a number the
    # curve does not have
    ("cost-oracle-m4.json", {**COST_ORACLE, "gains": 38},
     "results/cost-oracle-m4.json is not an oracle at a cost"),
    ("cost-oracle-m4.json", {**COST_ORACLE, "gains": True, "curve": COST_ORACLE["curve"][:2]},
     "results/cost-oracle-m4.json is not an oracle at a cost"),
    # M6: a non-inferiority row without its one-sided p, an ACEBench row without its interval, a test kind
    # the report does not know, and a descriptive row carrying a p after all
    ("test-m6-comparisons.json", {"comparisons": [{**m6_comparisons()["comparisons"][1], "noninferiority": {
        "margin": 0.02}}]}, "results/test-m6-comparisons.json is not a comparisons file"),
    ("test-m6-comparisons.json", {"comparisons": [{**m6_comparisons()["comparisons"][2], "acebench": {
        k: v for k, v in m6_comparisons()["comparisons"][2]["acebench"].items() if k != "paired_ci"}}]},
     "results/test-m6-comparisons.json is not a comparisons file"),
    ("test-m6-comparisons.json", {"comparisons": [{**m6_comparisons()["comparisons"][3], "mcnemar": {
        **m6_comparisons()["comparisons"][3]["mcnemar"], "p": 1.0}}]},  # a p nested in a descriptive row
     "results/test-m6-comparisons.json is not a comparisons file"),
    ("test-m6-comparisons.json", {"comparisons": [{**m6_comparisons()["comparisons"][0],
                                                    "test": "wilcoxon"}]},
     "results/test-m6-comparisons.json is not a comparisons file"),
    ("test-m6-comparisons.json", {"comparisons": [{**m6_comparisons()["comparisons"][3], "p": 0.5,
                                                    "p_holm": 0.5}]},
     "results/test-m6-comparisons.json is not a comparisons file"),
])
def test_a_registered_file_that_is_not_of_its_kind_is_refused_naming_it(tmp_path, name, data, why):
    results(tmp_path, {name: data})
    with pytest.raises(report.ReportError) as caught:
        report.build(tmp_path)
    assert why in str(caught.value)


def test_two_files_that_give_the_same_key_are_refused(tmp_path):
    # a dev summary with subsets gives eval.dev-b0-drafted.*, and so does the drafted subset's own file
    results(tmp_path, {"dev-b0.json": single_step({"r1-s1"}, subsets=True),
                       "dev-b0-drafted.json": single_step({"r1-s1"})})
    with pytest.raises(report.ReportError, match="eval.dev-b0-drafted.n .* results/dev-b0-drafted.json"):
        report.build(tmp_path)


def test_the_same_files_give_the_same_bytes_whatever_order_they_were_written_in(tmp_path):
    files = every_kind()
    for name, order in (("a", list(files)), ("b", list(reversed(files)))):
        results(tmp_path / name, {key: files[key] for key in order})
        report.write(tmp_path / name, ignored=lambda path: False)
    first, second = ((tmp_path / name / NUMBERS).read_bytes() for name in ("a", "b"))
    assert first == second and b"\r\n" not in first


# ---- the command line ------------------------------------------------------------------------------------


def test_build_report_writes_numbers_json_and_check_tells_a_stale_one(tmp_path, monkeypatch, capsys):
    results(tmp_path, every_kind())
    monkeypatch.chdir(tmp_path)  # no repository here: git ignores nothing
    assert build_report.main(["--check"]) == 1  # no numbers.json yet
    assert "results/numbers.json" in capsys.readouterr().out
    assert build_report.main([]) == 0
    written = (tmp_path / NUMBERS).read_bytes()
    assert build_report.main(["--check"]) == 0
    results(tmp_path, {"dev-b0.json": single_step({"r1-s1", "r2-s1"})})  # the dev run was scored again
    assert build_report.main(["--check"]) == 1
    assert (tmp_path / NUMBERS).read_bytes() == written  # --check writes nothing
    assert build_report.main([]) == 0
    rebuilt = json.loads((tmp_path / NUMBERS).read_text(encoding="utf-8"))
    assert rebuilt["numbers"]["eval.dev-b0.success"] == "66.7%"
    assert build_report.main(["--check"]) == 0


def test_build_report_exits_3_on_a_malformed_registered_file_and_writes_nothing(tmp_path, monkeypatch,
                                                                                capsys):
    results(tmp_path, every_kind())
    monkeypatch.chdir(tmp_path)
    assert build_report.main([]) == 0
    written = (tmp_path / NUMBERS).read_bytes()
    results(tmp_path, {"test-m3-resources.json": {"systems": "not an object"}})
    assert build_report.main([]) == 3
    assert "results/test-m3-resources.json" in capsys.readouterr().out
    assert (tmp_path / NUMBERS).read_bytes() == written


def test_readme_markers_quote_the_rebuilt_numbers(tmp_path, monkeypatch):
    results(tmp_path, every_kind())
    monkeypatch.chdir(tmp_path)
    assert build_report.main([]) == 0
    readme = ("FT: <!-- num:eval.test-m3-ft.success -->100.0%<!-- /num -->, B2 paid "
              "<!-- num:resources.test-m3.b2.cost.paid_rmb -->0.012<!-- /num --> RMB\n")
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    assert check(root=tmp_path) == []


def test_error_classes_are_given_for_test_runs_only_under_ascii_names():
    numbers = report.error_numbers({"error_classes": {"选错工具": 2, "值或格式错": 0, "budget_stop": 1},
                                    "unclassified_failures": 3}, "test-m9-ft")
    assert numbers == {"eval.test-m9-ft.errors.wrong_tool": "2", "eval.test-m9-ft.errors.wrong_value": "0",
                       "eval.test-m9-ft.errors.budget_stop": "1", "eval.test-m9-ft.errors.unclassified": "3"}
    assert set(report.ERROR_KEYS) == set(ERROR_CLASSES)
    assert len(set(report.ERROR_KEYS.values())) == len(ERROR_CLASSES)


def test_development_runs_give_no_error_classes(tmp_path):
    results(tmp_path, {"dev-b0.json": single_step({"r1-s1"})})
    numbers, _ = report.build(tmp_path)
    assert not [key for key in numbers if ".errors." in key]


def test_routing_numbers_carry_the_objective_the_policy_and_the_validation_block():
    # the M6 file names its objective and chosen policy (scripts/route_curve.py --objective); the M4 file has
    # neither. scripts/route_validate.py adds the frozen point's results on another set under "validation"
    block = {"n": 132, "policy": "P1", "signal": "min_decision_prob", "threshold": 0.4303716972742964,
             "rules": {"text_exempt": True, "param_triggers": []}, "routed": 15, "api_share": 15 / 132,
             "successes": 108, "success": 108 / 132, "cost_per_100_rmb": 0.09076060606060607,
             "random_success": 0.7790977961432507, "ft_success": 100 / 132, "b2_success": 125 / 132,
             "set": "data/eval/dev-ext.jsonl", "set_sha256": "ab" * 32, "inputs": {},
             "adapter_sha256": "cd" * 32, "cloud_model": "deepseek-flash"}
    data = {**routing(), "objective": "cost", "chosen_policy": "P1", "validation": {"dev-ext": block}}
    numbers = report.routing_numbers(data, "m6")
    assert numbers["routing.m6.objective"] == "cost" and numbers["routing.m6.policy"] == "P1"
    seen = "routing.m6.validation.dev-ext"
    assert {k: v for k, v in numbers.items() if ".validation." in k} == {
        f"{seen}.n": "132", f"{seen}.policy": "P1", f"{seen}.threshold": "0.430", f"{seen}.routed": "15",
        f"{seen}.api_share": "11.4%", f"{seen}.success": "81.8%", f"{seen}.random_success": "77.9%",
        f"{seen}.cost_per_100_rmb": "0.091", f"{seen}.ft_success": "75.8%", f"{seen}.b2_success": "94.7%"}
    # a validation of the M4 file (the threshold alone) names no policy
    alone = {**routing(), "validation": {"dev-ext": {**block, "policy": None}}}
    unnamed = report.routing_numbers(alone, "m4")
    assert "routing.m4.validation.dev-ext.policy" not in unnamed
    assert unnamed["routing.m4.validation.dev-ext.success"] == "81.8%"
    old = report.routing_numbers(routing(), "m4")
    assert not [k for k in old if ".validation." in k or k.endswith((".policy", ".objective"))]
    with pytest.raises(TypeError):
        report.routing_numbers({**data, "validation": {"dev-ext": {**block, "n": "132"}}}, "m6")


# ---- M7 WP1: results another scorer made (eval.metrics, Scorers; docs/m7-本地提升计划.md) -----------------

FINAL = "test-m7-final"  # the name of a test runner's single-step file of the system "final"


def normalized(data: dict) -> dict:
    """data as the normalized scorer's: "scorer": "normalized" first, as scripts/score.py writes it."""
    return {"scorer": "normalized", **data}


def test_a_result_the_normalized_scorer_made_gives_its_numbers_under_keys_of_its_own(tmp_path):
    """A single-step summary (scripts/score.py), a test runner's single-step file and a checkpoint selection
    that record "scorer": "normalized" give their numbers under <name>-normalized (once, when the name ends so
    already), never under a default result's keys; a single-step file may also carry the normalized scorer's
    summaries beside its own under "scorers". The kinds no scorer makes ignore the name's ending."""
    both = {**single_step({"r1-s1"}, subsets=True),
            "scorers": {"normalized": single_step({"r1-s1", "r2-s1"}, subsets=True)}}
    results(tmp_path, {"dev-b0.json": single_step({"r1-s1"}),
                       "dev-b0-normalized.json": normalized(single_step({"r1-s1", "r2-s1", "hw-dev-001"})),
                       "dev-b1.json": normalized(single_step({"r1-s1", "r2-s1"})),
                       f"{FINAL}.json": both,
                       "checkpoint-selection-ft3.json": SELECTION,
                       "checkpoint-selection-ft4.json": normalized(SELECTION),
                       "bfcl-m3-normalized.json": BFCL})
    numbers, _ = report.build(tmp_path)
    assert numbers["eval.dev-b0.success"] == "33.3%" and numbers["eval.dev-b0-normalized.success"] == "100.0%"
    assert numbers["eval.dev-b1-normalized.success"] == "66.7%" and numbers["eval.dev-b1-normalized.n"] == "3"
    assert numbers[f"eval.{FINAL}.success"] == "33.3%"
    assert numbers[f"eval.{FINAL}.errors.wrong_value"] == "1"
    assert numbers[f"eval.{FINAL}-drafted.success"] == "50.0%"
    assert numbers[f"eval.{FINAL}-normalized.success"] == "66.7%"
    assert numbers[f"eval.{FINAL}-normalized-drafted.success"] == "100.0%"
    assert numbers[f"eval.{FINAL}-normalized-handwritten.success"] == "0.0%"
    assert numbers[f"eval.{FINAL}-normalized.errors.wrong_value"] == "0"
    assert numbers[f"eval.{FINAL}-normalized.errors.unwanted_call"] == "1"
    assert numbers["checkpoint.ft3.step"] == numbers["checkpoint.ft4-normalized.step"] == "418"
    assert numbers["bfcl.m3-normalized.b0.aggregate"] == "75.25%"
    assert not [key for key in numbers if key.startswith(("eval.dev-b1.", "checkpoint.ft4."))
                or "normalized-normalized" in key]


def test_two_results_that_give_the_same_normalized_keys_are_refused(tmp_path):
    results(tmp_path, {"dev-b1.json": normalized(single_step({"r1-s1"})),
                       "dev-b1-normalized.json": normalized(single_step({"r1-s1"}))})
    with pytest.raises(report.ReportError,
                       match="eval.dev-b1-normalized.n comes from both results/dev-b1-normalized.json and "
                             "results/dev-b1.json"):
        report.build(tmp_path)


@pytest.mark.parametrize("name, data, what", [
    ("dev-b0-normalized.json", single_step({"r1-s1"}), "a single-step summary"),
    ("dev-b0.json", {"scorer": "lenient", **single_step({"r1-s1"})}, "a single-step summary"),
    ("dev-b0.json", {"scorer": None, **single_step({"r1-s1"})}, "a single-step summary"),
    (f"{FINAL}.json", {**single_step({"r1-s1"}), "scorers": {"default": single_step({"r1-s1"})}},
     "a single-step summary"),
    (f"{FINAL}.json", {**single_step({"r1-s1"}), "scorers": {"lenient": single_step({"r1-s1"})}},
     "a single-step summary"),
    ("checkpoint-selection-ft4-normalized.json", SELECTION, "a checkpoint selection"),
], ids=["named-normalized-made-default", "unknown-scorer", "no-scorer-name", "nested-default",
        "nested-unknown", "selection-named-normalized"])
def test_a_scorer_that_is_unknown_or_not_the_names_is_refused(tmp_path, name, data, what):
    results(tmp_path, {name: data})
    with pytest.raises(report.ReportError) as caught:
        report.build(tmp_path)
    assert f"results/{name} is not {what}" in str(caught.value) and "scorer" in str(caught.value)


# ---- M8: the checkpoint decision, read (scripts/decide_m8.py; docs/m8-预登记.md, 选点) -------------------


def test_m8s_decision_gives_every_dev_set_b_count_the_sums_the_gates_and_the_verdict(tmp_path):
    """results/decision-m8.json gives decision.m8.*: B0's and FT3's dev set B counts and rates, FT3's sums,
    every candidate's rank and sums under <run>-<step>, its counts, gates and verdict once it was run on dev
    set B; the chosen checkpoint and MG only when there is one. decision-m7.json, decide_m7's file of another
    shape, is not read."""
    results(tmp_path, {"decision-m8.json": m8_decision(), "decision-m7.json": "{not this kind"})
    numbers, sources = report.build(tmp_path)
    assert [p.as_posix() for p in sources] == ["results/decision-m8.json"]
    at = "decision.m8"
    assert {key: value for key, value in numbers.items() if not key.startswith(f"{at}.ft5-")} == {
        f"{at}.decision": "NO-GO", f"{at}.pg1_fallback": "yes", f"{at}.candidates": "2",
        f"{at}.b0.special_n": "150", f"{at}.b0.strict": "26", f"{at}.b0.lenient": "103",
        f"{at}.b0.valid_n": "90", f"{at}.b0.exact": "58", f"{at}.b0.false_refusals": "11",
        f"{at}.b0.zh_altered": "0",
        f"{at}.b0.strict_rate": "17.3%", f"{at}.b0.lenient_rate": "68.7%", f"{at}.b0.exact_rate": "64.4%",
        f"{at}.ft3.step": "1635", f"{at}.ft3.s0": "238", f"{at}.ft3.s1": "237", f"{at}.ft3.sum": "475",
        f"{at}.ft3.s0_without_rules": "230", f"{at}.ft3.s1_without_rules": "232",
        f"{at}.ft3.sum_without_rules": "462",
        f"{at}.ft3.special_n": "150", f"{at}.ft3.strict": "49", f"{at}.ft3.lenient": "103",
        f"{at}.ft3.valid_n": "90", f"{at}.ft3.exact": "64", f"{at}.ft3.false_refusals": "11",
        f"{at}.ft3.zh_altered": "0",
        f"{at}.ft3.strict_rate": "32.7%", f"{at}.ft3.lenient_rate": "68.7%", f"{at}.ft3.exact_rate": "71.1%"}
    first, second = f"{at}.ft5-232", f"{at}.ft5-116"
    assert {key: numbers[key] for key in numbers if key.startswith(f"{first}.")} == {
        f"{first}.rank": "1", f"{first}.s0": "234", f"{first}.s1": "234", f"{first}.sum": "468",
        f"{first}.s0_without_rules": "228", f"{first}.s1_without_rules": "228",
        f"{first}.sum_without_rules": "456",
        f"{first}.special_n": "150", f"{first}.strict": "76", f"{first}.lenient": "94",
        f"{first}.valid_n": "90", f"{first}.exact": "71", f"{first}.false_refusals": "10",
        f"{first}.zh_altered": "0",
        f"{first}.strict_rate": "50.7%", f"{first}.lenient_rate": "62.7%", f"{first}.exact_rate": "78.9%",
        f"{first}.pg1": "no", f"{first}.pg2": "no", f"{first}.pg3": "yes", f"{first}.pg4": "yes",
        f"{first}.passes": "no"}
    assert (numbers[f"{second}.rank"], numbers[f"{second}.sum"], numbers[f"{second}.zh_altered"],
            numbers[f"{second}.pg4"], numbers[f"{second}.passes"]) == ("2", "468", "1", "no", "no")


def test_a_go_decision_gives_the_chosen_checkpoint_and_mg_and_no_counts_for_a_checkpoint_never_run(tmp_path):
    results(tmp_path, {"decision-m8.json": m8_decision(go=True)})
    numbers, _ = report.build(tmp_path)
    at = "decision.m8"
    assert (numbers[f"{at}.decision"], numbers[f"{at}.chosen"]) == ("GO", "232")
    assert {key: numbers[key] for key in numbers if key.startswith(f"{at}.mg.")} == {
        f"{at}.mg.sum": "468", f"{at}.mg.ft3_sum": "475", f"{at}.mg.threshold": "465",
        f"{at}.mg.passed": "yes"}
    assert numbers[f"{at}.ft5-232.passes"] == "yes" and numbers[f"{at}.ft5-232.strict_rate"] == "73.3%"
    # 116 is ranked after the chosen checkpoint and was never run on dev set B: its sums and rank alone
    assert sorted(key for key in numbers if key.startswith(f"{at}.ft5-116.")) == sorted(
        f"{at}.ft5-116.{key}" for key in ("rank", "s0", "s1", "sum", "s0_without_rules", "s1_without_rules",
                                          "sum_without_rules"))


def m8_broken(change) -> dict:
    data = m8_decision()
    change(data)
    return data


@pytest.mark.parametrize("data", [
    m8_broken(lambda d: d.update(decision="MAYBE")),
    m8_broken(lambda d: d.update(pg1_fallback="yes")),
    m8_broken(lambda d: d["b0"]["special_dev"].update(strict=True)),  # a count that is not a whole number
    m8_broken(lambda d: d["baseline"].update(sum=475.0)),
    m8_broken(lambda d: d["candidates"][0]["special_dev"].update(exact="67")),
    m8_broken(lambda d: d["candidates"][0].update(step="116")),
    m8_broken(lambda d: d.update(ranking=[232])),  # the ranking does not hold every candidate once
    m8_broken(lambda d: d.update(ranking=[232, 232, 116])),
    m8_broken(lambda d: d["candidates"][1]["gates"]["PG2"].update(passed=None)),
    m8_broken(lambda d: d["candidates"][1]["special_dev"].update(special_n=0)),  # a rate over no item
], ids=["decision", "fallback", "bool-count", "float-sum", "text-count", "text-step", "short-ranking",
        "ranking-twice", "gate-verdict", "no-special-items"])
def test_a_decision_that_is_not_of_its_shape_is_refused(tmp_path, data):
    results(tmp_path, {"decision-m8.json": data})
    with pytest.raises(report.ReportError, match="results/decision-m8.json is not a checkpoint decision"):
        report.build(tmp_path)


# ---- M9: the inference wrappers' counts, a development run's lengths, the lengths summary (docs/m9-计划.md,
# WP4) ------------------------------------------------------------------------------------------------------


def test_the_wrapper_blocks_of_the_fixtures_are_what_the_test_runner_writes():
    run_test = load_script("run_test")
    assert run_test.wrapper_counts(VOTED_TURNS) == WRAPPER_BLOCKS
    assert run_test.wrapper_counts(UNVOTED_ATTEMPTS) == E2E_WRAPPER_BLOCKS
    assert run_test.wrapper_counts([{"status": "ok"}]) == {}  # neither voted nor ruled: no block


def test_a_voted_and_ruled_systems_test_files_give_their_votes_and_rules_and_no_other_file_does(tmp_path):
    """The single-step file counts over its records, the end-to-end file over its transcripts' attempts; a
    vote no sample voted on has no agreement, and an unruled run no rules block, so neither key is given."""
    results(tmp_path, every_kind())
    numbers, _ = report.build(tmp_path)
    at, e2e = "eval.test-m7-final", "e2e.test-m7-final"
    assert {key: value for key, value in numbers.items() if ".votes." in key or ".rules." in key} == {
        f"{at}.votes.steps": "3", f"{at}.votes.unanimous": "1", f"{at}.votes.agreement": "80.0%",
        f"{at}.votes.chosen_not_greedy": "1", f"{at}.votes.no_sample_voted": "1",
        f"{at}.votes.candidate_errors": "3", f"{at}.rules.steps": "3",
        f"{at}.rules.fired.transient_retry": "2", f"{at}.rules.fired.id_repair": "1",
        f"{e2e}.votes.steps": "2", f"{e2e}.votes.unanimous": "0", f"{e2e}.votes.chosen_not_greedy": "0",
        f"{e2e}.votes.no_sample_voted": "2", f"{e2e}.votes.candidate_errors": "0"}
    assert report.wrapper_numbers({"summary": {}}, "eval.test-m7-ft2") == {}


def test_a_development_runs_report_gives_its_steps_over_the_training_length_and_failed_candidates(tmp_path):
    """Only a manifest that counted the steps (a count, 0 included) gives training_length, the count and the
    local calls; a run without the context check (null) and a report from before M7 give none of them, and
    only a voted run's summary gives its failed candidates, in all and per error type."""
    before_m7 = {"manifest": {"system": "ft"}, "tasks": [],
                 "summary": summarize_tasks([task("e2e-dev-01", "lookup_paper", "en", True)])}
    results(tmp_path, {"dev-e2e-final.json": dev_e2e_final(),
                       "dev-e2e-final0.json": dev_e2e_final(over=0, failed={}),
                       "dev-e2e-final-nocheck.json": dev_e2e_final(over=None, failed=None),
                       "dev-e2e-ft.json": before_m7})
    numbers, _ = report.build(tmp_path)
    extras = ("training_length", "steps_over_training_length", "local_calls", "candidate_errors")

    def extra(name: str) -> dict[str, str]:
        return {key: value for key, value in numbers.items()
                if key.startswith(f"e2e.{name}.") and key.split(".", 2)[2].startswith(extras)}

    assert extra("dev-final") == {
        "e2e.dev-final.training_length": "5120", "e2e.dev-final.steps_over_training_length": "1",
        "e2e.dev-final.local_calls": "8", "e2e.dev-final.candidate_errors": "3",
        "e2e.dev-final.candidate_errors.HarnessError": "2", "e2e.dev-final.candidate_errors.ReadTimeout": "1"}
    assert extra("dev-final0") == {
        "e2e.dev-final0.training_length": "5120", "e2e.dev-final0.steps_over_training_length": "0",
        "e2e.dev-final0.local_calls": "8", "e2e.dev-final0.candidate_errors": "0"}
    assert extra("dev-final-nocheck") == {} and extra("dev-ft") == {}
    assert (numbers["e2e.dev-final.n"], numbers["e2e.dev-final.success"]) == ("2", "50.0%")
    assert (numbers["e2e.dev-final-nocheck.n"], numbers["e2e.dev-ft.n"]) == ("2", "1")  # their summaries'


def test_a_lengths_summary_gives_each_systems_thinking_lengths_and_steps_over_the_training_length(tmp_path):
    """Per system the parts it has (FT2 keeps no thinking record), percentiles and maxima over no turn and the
    rate of no turn left out, the longest of no call too; the file's other fields are not read."""
    data = lengths_file()
    data["systems"]["finalg"] = {"single_step": thinking(0, 0, None, None, None, None, None), "e2e": None,
                                 "e2e_length": {"training_length": 5120, "calls": 0, "over": 0, "max": None}}
    results(tmp_path, {"lengths-m7.json": data})
    numbers, sources = report.build(tmp_path)
    assert [p.as_posix() for p in sources] == ["results/lengths-m7.json"]
    final, ft2, finalg = "lengths.m7.final", "lengths.m7.ft2", "lengths.m7.finalg"
    assert numbers == {
        f"{final}.single_step.n": "300", f"{final}.single_step.reasoning_p50": "812",
        f"{final}.single_step.reasoning_p95": "1280", f"{final}.single_step.reasoning_max": "1280",
        f"{final}.single_step.answer_p50": "41", f"{final}.single_step.answer_p95": "97",
        f"{final}.single_step.forced": "12", f"{final}.single_step.forced_rate": "4.0%",
        f"{final}.e2e.n": "1200", f"{final}.e2e.reasoning_p50": "640", f"{final}.e2e.reasoning_p95": "1280",
        f"{final}.e2e.reasoning_max": "1280", f"{final}.e2e.answer_p50": "52",
        f"{final}.e2e.answer_p95": "181", f"{final}.e2e.forced": "30", f"{final}.e2e.forced_rate": "2.5%",
        f"{final}.e2e_length.training_length": "5120", f"{final}.e2e_length.calls": "412",
        f"{final}.e2e_length.over": "3", f"{final}.e2e_length.max": "6011",
        f"{ft2}.e2e_length.training_length": "4096", f"{ft2}.e2e_length.calls": "380",
        f"{ft2}.e2e_length.over": "0", f"{ft2}.e2e_length.max": "3998",
        f"{finalg}.single_step.n": "0", f"{finalg}.single_step.forced": "0",
        f"{finalg}.e2e_length.training_length": "5120", f"{finalg}.e2e_length.calls": "0",
        f"{finalg}.e2e_length.over": "0"}


def test_lengths_summaries_are_registered_by_milestone_and_take_no_other_kinds_file():
    assert report.kind_of("lengths-m7.json").key == report.kind_of("lengths-m12.json").key == "lengths"
    for name in ("lengths-final.json", "lengths-m7-final.json", "lengths.json", "lengths-m7.jsonl"):
        assert report.kind_of(name) is None
    assert report.kind_of("dev-lengths-m7.json").key == "dev"  # a development summary's name, as before
    assert [kind.key for kind in report.REGISTRY].count("lengths") == 1


def m9_broken(make, change) -> dict:
    data = make()
    change(data)
    return data


@pytest.mark.parametrize("name, data, what", [
    ("test-m7-final.json", m9_broken(m7_final, lambda d: d["votes"].update(steps=3.0)),
     "a single-step summary"),
    ("test-m7-final.json", m9_broken(m7_final, lambda d: d["votes"].update(unanimous=True)),
     "a single-step summary"),
    ("test-m7-final.json", m9_broken(m7_final, lambda d: d["votes"].pop("no_sample_voted")),
     "a single-step summary"),
    ("test-m7-final.json", m9_broken(m7_final, lambda d: d["votes"].update(agreement="0.8")),
     "a single-step summary"),
    ("test-m7-final.json", m9_broken(m7_final, lambda d: d["votes"].update(agreement=1.2)),
     "a single-step summary"),
    ("test-m7-final.json", m9_broken(m7_final, lambda d: d["rules"]["fired"].update(id_repair="1")),
     "a single-step summary"),
    ("test-m7-final.json", m9_broken(m7_final, lambda d: d["rules"].update(fired=[2, 1])),
     "a single-step summary"),
    ("test-m7-e2e-final.json", m9_broken(m7_e2e_final, lambda d: d["votes"].update(candidate_errors=None)),
     "an end-to-end summary"),
    ("dev-e2e-final.json", m9_broken(dev_e2e_final, lambda d: d["manifest"].update(
        steps_over_training_length="1")), "an end-to-end summary"),
    ("dev-e2e-final.json", m9_broken(dev_e2e_final, lambda d: d["manifest"].pop("training_length")),
     "an end-to-end summary"),
    ("dev-e2e-final.json", m9_broken(dev_e2e_final, lambda d: d["tasks"][0].update(local_calls=None)),
     "an end-to-end summary"),
    ("dev-e2e-final.json", m9_broken(dev_e2e_final, lambda d: d["summary"].update(
        candidate_errors={"HarnessError": 1.0})), "an end-to-end summary"),
    ("dev-e2e-final.json", m9_broken(dev_e2e_final, lambda d: d["summary"].update(candidate_errors=[1])),
     "an end-to-end summary"),
    ("lengths-m7.json", {"systems": {"final": {"e2e": None}}}, "a lengths summary"),
    ("lengths-m7.json", {"systems": [lengths_file()["systems"]["final"]]}, "a lengths summary"),
    ("lengths-m7.json", m9_broken(lengths_file, lambda d: d["systems"]["final"]["single_step"].update(
        forced=301)), "a lengths summary"),
    ("lengths-m7.json", m9_broken(lengths_file, lambda d: d["systems"]["ft2"]["e2e_length"].update(over=381)),
     "a lengths summary"),
    ("lengths-m7.json", m9_broken(lengths_file, lambda d: d["systems"]["final"]["e2e"].update(n=1200.0)),
     "a lengths summary"),
    ("lengths-m7.json", m9_broken(lengths_file, lambda d: d["systems"]["final"]["e2e"].update(
        reasoning_p50="640")), "a lengths summary"),
    ("lengths-m7.json", m9_broken(lengths_file, lambda d: d["systems"]["final"]["e2e"].pop("answer_p95")),
     "a lengths summary"),
    ("lengths-m7.json", m9_broken(lengths_file, lambda d: d["systems"]["ft2"]["e2e_length"].update(max=True)),
     "a lengths summary"),
], ids=["float-steps", "bool-unanimous", "no-undecided-count", "text-agreement", "agreement-above-one",
        "text-fired", "fired-list", "null-candidate-errors", "text-over", "no-training-length",
        "null-local-calls", "float-candidate-error", "candidate-error-list", "no-block", "systems-list",
        "forced-above-n", "over-above-calls", "float-n", "text-percentile", "no-answer-p95", "bool-max"])
def test_an_m9_result_that_is_not_of_its_shape_is_refused_naming_it(tmp_path, name, data, what):
    results(tmp_path, {name: data})
    with pytest.raises(report.ReportError) as caught:
        report.build(tmp_path)
    assert f"results/{name} is not {what}" in str(caught.value)


# ---- M11: the Wilson intervals, B2's turns per thinking mode, ACEBench per stratum, the bake-off
# (docs/m11-计划.md, 1) -------------------------------------------------------------------------------------


@pytest.mark.parametrize("successes, n, shown", [
    (270, 300, "[86.1%, 92.9%]"), (135, 150, "[84.2%, 93.8%]"),  # docs/m7-预登记.md's two worked examples
    (263, 300, "[83.5%, 90.9%]"), (252, 300, "[79.4%, 87.7%]"), (122, 150, "[74.3%, 86.8%]"),
    (0, 300, "[0.0%, 1.3%]"), (300, 300, "[98.7%, 100.0%]"),
    (0, 1, "[0.0%, 79.3%]"), (1, 1, "[20.7%, 100.0%]")])
def test_a_wilson_interval_is_quoted_as_two_percentages_with_one_decimal(successes, n, shown):
    assert report.rate_interval(*wilson_interval(successes, n)) == shown


def test_the_preregistered_z_gives_the_same_strings_and_the_ends_of_the_scale_are_exact():
    for successes, n, shown in ((270, 300, "[86.1%, 92.9%]"), (135, 150, "[84.2%, 93.8%]")):
        assert report.rate_interval(*wilson_interval(successes, n, z=1.959964)) == shown
    assert wilson_interval(0, 300)[0] == 0.0 and wilson_interval(300, 300)[1] == 1.0


@pytest.mark.parametrize("successes, n, error", [
    (5, 0, ValueError), (-1, 300, ValueError), (301, 300, ValueError),
    (True, 300, TypeError), (2.0, 300, TypeError), ("3", 300, TypeError)])
def test_counts_that_give_no_interval_are_refused(successes, n, error):
    with pytest.raises(error):
        report.rate_interval(*wilson_interval(successes, n))


@pytest.mark.parametrize("low, high", [(0.5, 0.4), (-0.01, 0.5), (0.5, 1.01), (math.nan, 0.5)])
def test_an_interval_that_is_not_one_of_a_rate_is_refused(low, high):
    with pytest.raises(ValueError):
        report.rate_interval(low, high)


B2_MODES = {"non-thinking": 2, "thinking": 4}  # scripts/run_test.py's "modes": a cloud model's turns per mode


def m7_b2() -> dict:
    """test-m7-b2.json: a cloud system's single-step file with its turns per thinking mode (a copy)."""
    return {**single_step({"r1-s1", "r2-s1", "hw-dev-001"}, subsets=True), "modes": dict(B2_MODES)}


def m7_e2e_b2() -> dict:
    """test-m7-e2e-b2.json: its end-to-end file, with the turns per mode of its end-to-end model."""
    return {**m5_e2e(routing=False), "system": "b2", "modes": {"non-thinking": 2, "thinking": 2}}


def test_a_test_runs_success_rate_gives_its_wilson_interval_under_its_own_and_each_scorers_name(tmp_path):
    """The single-step file's summary and the summary of each block under another scorer give success_wilson
    beside success, as does the end-to-end file's summary, over the successes the rate and n give back; the
    subsets, a development run's files and a file of another kind give none."""
    every = single_step({"r1-s1", "r2-s1", "hw-dev-001"}, subsets=True)
    results(tmp_path, {"test-m7-final.json": {**m7_final(), "scorers": {"normalized": every}},
                       "test-m7-e2e-final.json": m7_e2e_final(), "dev-b0.json": single_step({"r1-s1"}),
                       "dev-e2e-final.json": dev_e2e_final(), "lengths-m7.json": lengths_file()})
    numbers, _ = report.build(tmp_path)
    assert {key: value for key, value in numbers.items() if "wilson" in key} == {
        "eval.test-m7-final.success_wilson": "[20.8%, 93.9%]",  # 2 of 3
        "eval.test-m7-final-normalized.success_wilson": "[43.9%, 100.0%]",  # 3 of 3
        "e2e.test-m7-final.success_wilson": "[9.5%, 90.5%]"}  # 1 of 2
    assert (numbers["eval.test-m7-final.success"], numbers["eval.test-m7-final-normalized.success"],
            numbers["e2e.test-m7-final.success"]) == ("66.7%", "100.0%", "50.0%")


def test_a_rate_written_to_ten_decimals_still_gives_back_its_successes(tmp_path):
    data = m7_final()
    data["summary"]["success_rate"] = 0.6666666667  # 2 of 3, 3.3e-11 off
    results(tmp_path, {"test-m7-final.json": data})
    numbers, _ = report.build(tmp_path)
    assert numbers["eval.test-m7-final.success_wilson"] == "[20.8%, 93.9%]"


def test_no_item_or_no_rate_gives_no_wilson_interval(tmp_path):
    unrated, e2e_unrated = m7_final(), m7_e2e_final()
    unrated["summary"]["success_rate"] = e2e_unrated["summary"]["success_rate"] = None
    results(tmp_path, {"test-m7-final.json": {**m7_final(), "summary": summarize([], [])},
                       "test-m7-e2e-final.json": {**m7_e2e_final(), "summary": summarize_tasks([])},
                       "test-m7-finalg.json": unrated, "test-m7-e2e-finalg.json": e2e_unrated})
    numbers, _ = report.build(tmp_path)
    assert (numbers["eval.test-m7-final.n"], numbers["e2e.test-m7-final.n"]) == ("0", "0")
    assert (numbers["eval.test-m7-finalg.n"], numbers["e2e.test-m7-finalg.n"]) == ("3", "2")
    assert not [key for key in numbers if "wilson" in key]


def with_scorer_rate(data: dict) -> dict:
    """data with a block under the normalized scorer whose rate is 1.5 successes out of 3."""
    block = single_step({"r1-s1"}, subsets=True)
    block["summary"]["success_rate"] = 0.5
    return {**data, "scorers": {"normalized": block}}


@pytest.mark.parametrize("name, data, what", [
    ("test-m7-final.json", m9_broken(m7_final, lambda d: d["summary"].update(success_rate=0.5)),
     "a single-step summary"),  # 1.5 successes out of 3
    ("test-m7-final.json", with_scorer_rate(m7_final()), "a single-step summary"),
    ("test-m7-e2e-final.json", m9_broken(m7_e2e_final, lambda d: d["summary"].update(n=3, success_rate=0.5)),
     "an end-to-end summary"),
    ("test-m7-e2e-final.json", m9_broken(m7_e2e_final, lambda d: d["summary"].update(n=2.0)),
     "an end-to-end summary"),  # e2e_numbers quotes any n; the interval needs a count of items
], ids=["half-a-success", "half-a-success-under-a-scorer", "half-a-success-end-to-end", "float-n"])
def test_a_rate_and_n_that_give_back_no_whole_number_of_successes_are_refused(tmp_path, name, data, what):
    results(tmp_path, {name: data})
    with pytest.raises(report.ReportError) as caught:
        report.build(tmp_path)
    assert f"results/{name} is not {what}" in str(caught.value)
    assert "whole number of successes" in str(caught.value) or "is not a count" in str(caught.value)


@pytest.mark.parametrize("summary, error", [
    ({"n": 3, "success_rate": 0.5}, report.ReportError), ({"n": 3, "success_rate": "0.667"}, TypeError),
    ({"n": 3, "success_rate": True}, TypeError), ({"n": "3", "success_rate": 1.0}, TypeError),
    ({"n": 3, "success_rate": math.nan}, ValueError), ({"n": 3, "success_rate": math.inf}, ValueError),
    ({"n": 5, "success_rate": 1.2}, ValueError),  # 6 successes out of 5
], ids=["half-a-success", "text-rate", "bool-rate", "text-n", "nan-rate", "infinite-rate", "rate-above-one"])
def test_the_interval_needs_a_count_of_items_and_a_rate_that_gives_back_a_count_of_successes(summary, error):
    with pytest.raises(error):
        report.wilson_numbers(summary, "eval.test-m7-final")


def test_the_tolerance_of_a_rate_that_gives_back_its_successes_is_on_the_rate_not_on_the_count():
    """270 of 300 (docs/m7-预登记.md's example): a rate 5e-10 off is 1.5e-7 of a success off, and still gives
    back its 270 successes; one 2e-9 off gives back none."""
    assert report.wilson_numbers({"n": 300, "success_rate": 0.9 + 5e-10}, "eval.test-m7-final") == {
        "eval.test-m7-final.success_wilson": "[86.1%, 92.9%]"}
    with pytest.raises(report.ReportError, match="no whole number of successes"):
        report.wilson_numbers({"n": 300, "success_rate": 0.9 + 2e-9}, "eval.test-m7-final")


def test_the_turns_per_thinking_mode_of_a_test_runs_files_are_given_under_the_files_name(tmp_path):
    """B2's turns per thinking mode (scripts/run_test.py, "modes": llm.cloud's thinking and non-thinking), in
    its single-step and end-to-end files, once per file, not per scorer block; a file without "modes" gives
    none."""
    b2 = {**m7_b2(), "scorers": {"normalized": single_step({"r1-s1"}, subsets=True)}}
    results(tmp_path, {"test-m7-b2.json": b2, "test-m7-e2e-b2.json": m7_e2e_b2(),
                       "test-m7-final.json": m7_final(), "test-m7-e2e-final.json": m7_e2e_final()})
    numbers, _ = report.build(tmp_path)
    assert {key: value for key, value in numbers.items() if ".modes." in key} == {
        "eval.test-m7-b2.modes.non-thinking": "2", "eval.test-m7-b2.modes.thinking": "4",
        "e2e.test-m7-b2.modes.non-thinking": "2", "e2e.test-m7-b2.modes.thinking": "2"}


@pytest.mark.parametrize("modes", [{"thinking": -1}, {"thinking": True}, {"thinking": 4.0}, {"thinking": "4"},
                                   {"thinking": None}, [4, 2]],
                         ids=["negative", "bool", "float", "text", "null", "list"])
@pytest.mark.parametrize("name, make, what", [("test-m7-b2.json", m7_b2, "a single-step summary"),
                                              ("test-m7-e2e-b2.json", m7_e2e_b2, "an end-to-end summary")],
                         ids=["single-step", "end-to-end"])
def test_turns_per_mode_that_are_not_counts_are_refused(tmp_path, modes, name, make, what):
    results(tmp_path, {name: {**make(), "modes": modes}})
    with pytest.raises(report.ReportError) as caught:
        report.build(tmp_path)
    assert f"results/{name} is not {what}" in str(caught.value)


# ACEBench's strata of two systems, {language: {category: (correct, n)}}: ft and b0 alike in English, ft
# lower on one Chinese stratum and higher on two
ACE_PAIR_GROUPS = {"normal": ["normal_single_turn_single_function", "normal_atom_bool"],
                   "special": ["special_incomplete"]}
FT_STRATA = {"zh": {"normal_single_turn_single_function": (40, 100), "normal_atom_bool": (1, 3),
                    "special_incomplete": (25, 50)},
             "en": {"normal_single_turn_single_function": (45, 100)}}
B0_STRATA = {"zh": {"normal_single_turn_single_function": (37, 100), "normal_atom_bool": (2, 3),
                    "special_incomplete": (20, 50)},
             "en": {"normal_single_turn_single_function": (45, 100)}}


def ace_system(strata: dict[str, dict[str, tuple[int, int]]]) -> dict:
    """A system's row of scripts/run_acebench.py's results from its strata: each language's block
    (ace_language) and the mean over every (language, category) pair."""
    languages = {language: ace_language(categories, 0, 0) for language, categories in strata.items()}
    cells = [cell for block in languages.values() for cell in block["categories"].values()]
    return {"acebench_model": "local-model", "adapter": None, "raw": "runs/acebench/pair",
            "aggregate": ace_mean(cells), "languages": languages}


def ace_pair(ft: dict | None = None, b0: dict | None = None) -> dict:
    """An ACEBench result that holds ft and b0, as results/acebench-m7.json does."""
    return {"smoke": False, "languages": ["zh", "en"], "groups": ACE_PAIR_GROUPS,
            "systems": {"b0": ace_system(b0 or B0_STRATA), "ft": ace_system(ft or FT_STRATA)}}


def test_ft_against_b0_gives_each_strata_difference_their_means_and_the_strata_each_is_higher_on(tmp_path):
    results(tmp_path, {"acebench-pair.json": ace_pair()})
    numbers, _ = report.build(tmp_path)
    at = "acebench.pair.ft-vs-b0"
    assert {key: value for key, value in numbers.items() if key.startswith(f"{at}.")} == {
        f"{at}.zh.normal_atom_bool": "-33.33",  # 1/3 - 2/3
        f"{at}.zh.normal_single_turn_single_function": "+3.00", f"{at}.zh.special_incomplete": "+10.00",
        f"{at}.zh.normal": "-15.17",  # (-1/3 + 3/100) / 2
        f"{at}.zh.special": "+10.00", f"{at}.zh.aggregate": "-6.78",  # (-1/3 + 3/100 + 1/10) / 3
        f"{at}.en.normal_single_turn_single_function": "+0.00", f"{at}.en.normal": "+0.00",
        f"{at}.en.aggregate": "+0.00",  # no special category run in English: no group mean there
        f"{at}.aggregate": "-5.08",  # the mean over the four strata
        f"{at}.strata": "4", f"{at}.ft_higher": "2", f"{at}.b0_higher": "1", f"{at}.equal": "1"}
    # the mean of the strata's differences is the difference of the systems' means over the strata
    assert (numbers["acebench.pair.ft.aggregate"], numbers["acebench.pair.b0.aggregate"]) == (
        "42.08%", "47.17%")


def test_a_file_without_both_ft_and_b0_gives_no_differences(tmp_path):
    ft_alone = ace_pair()
    del ft_alone["systems"]["b0"]
    results(tmp_path, {"acebench-ft.json": ft_alone, "acebench-m4.json": ACEBENCH})  # ACEBENCH: b0 alone
    numbers, _ = report.build(tmp_path)
    assert (numbers["acebench.ft.ft.aggregate"], numbers["acebench.m4.b0.aggregate"]) == ("42.08%", "40.67%")
    assert not [key for key in numbers if "-vs-" in key]


def ace_pair_with(change) -> dict:
    data = ace_pair()
    change(data["systems"])
    return data


@pytest.mark.parametrize("data", [
    ace_pair(ft={**FT_STRATA, "zh": {**FT_STRATA["zh"], "normal_atom_bool": (1, 4)}}),  # 4 answers, b0 3
    ace_pair(ft={**FT_STRATA, "zh": {key: cell for key, cell in FT_STRATA["zh"].items()
                                     if key != "special_incomplete"}}),
    ace_pair(b0={"zh": B0_STRATA["zh"]}),  # English run by ft alone
    ace_pair(ft={**FT_STRATA, "en": {"normal_single_turn_single_function": (101, 100)}}),
    ace_pair_with(lambda systems: systems["ft"]["languages"]["zh"]["categories"]["normal_atom_bool"].update(
        accuracy=0.34)),  # an accuracy that is not its correct over its n
    ace_pair_with(lambda systems: systems["b0"]["languages"]["en"]["categories"][
        "normal_single_turn_single_function"].update(n=100.0)),
], ids=["different-n", "different-categories", "different-languages", "more-correct-than-answers",
        "accuracy-not-the-counts", "float-n"])
def test_ft_and_b0_that_do_not_hold_the_same_strata_with_the_same_answers_are_refused(tmp_path, data):
    results(tmp_path, {"acebench-pair.json": data})
    with pytest.raises(report.ReportError, match="results/acebench-pair.json is not an ACEBench result"):
        report.build(tmp_path)


# The keys results/acebench-m7.json gave before M11 (results/numbers.json as the commit of the M11 plan holds
# it): how many, and the SHA-256 of them with their values as canonical JSON. M11 adds its ft-vs-b0 keys and
# changes none of these (docs/m11-计划.md: numbers.json only grows, its old keys and values unchanged).
M7_ACEBENCH_BEFORE_M11 = (290, "248e2a1a1aa623db24a5f57cfa4ec5afbbe1d03e38fd25d927ab1d41e5018d5b")


def test_m7s_mean_difference_over_the_strata_is_h3s_difference(tmp_path):
    """results/acebench-m7.json (committed) gives acebench.m7.ft-vs-b0.aggregate equal to H3's difference in
    results/numbers.json, compare.test-m7.final-vs-b0-acebench.diff (scripts/run_test.py,
    Session.acebench_comparison: the mean over the 30 (language, category) strata of FT3's per-answer success
    rate minus B0's, which reproduce the counts of the score files the results file holds). Beside its 41
    ft-vs-b0 keys it gives the keys it gave before M11, unchanged (M7_ACEBENCH_BEFORE_M11), and numbers.json
    holds every key it gives as the build gives it, and no other under acebench.m7."""
    text = (REPO / "results" / "acebench-m7.json").read_text(encoding="utf-8")
    results(tmp_path, {"acebench-m7.json": text})
    numbers, _ = report.build(tmp_path)
    recorded = json.loads((REPO / NUMBERS).read_text(encoding="utf-8"))["numbers"]
    h3, at = "compare.test-m7.final-vs-b0-acebench", "acebench.m7.ft-vs-b0"
    assert numbers[f"{at}.aggregate"] == recorded[f"{h3}.diff"] == "-3.96"
    assert numbers[f"{at}.strata"] == recorded[f"{h3}.strata"] == "30"
    assert sum(int(numbers[f"{at}.{key}"]) for key in ("ft_higher", "b0_higher", "equal")) == 30
    # M11's: the 30 strata, the normal and special groups and the aggregate of each of the two languages, the
    # aggregate over every stratum and the four counts
    new = {key for key in numbers if key.startswith(f"{at}.")}
    assert len(new) == 30 + 2 * 3 + 1 + 4 == 41
    before = {key: value for key, value in numbers.items() if key not in new}
    digest = hashlib.sha256(canonical_json(before).encode("utf-8")).hexdigest()
    assert (len(before), digest) == M7_ACEBENCH_BEFORE_M11
    assert {key: value for key, value in recorded.items() if key.startswith("acebench.m7.")} == numbers


def bakeoff_candidate(key: str, name: str, thinking: str, mode: str, p: float, handwritten: float,
                      hours: float, *, greedy: int | None = None, extension: dict | None = None) -> dict:
    """A scored candidate of results/bakeoff-m7.json as eval.bakeoff.aggregate writes it: its gate's entry
    with the gate's name, smoke status, path and SHA-256."""
    return {"candidate": key, "status": "scored", "P": p, "eligible": True, "reasons": [], "missing": [],
            "thinking": thinking, "mode": mode, "handwritten": handwritten,
            "ft3_hours_estimate": round(hours, 1), "ft3_hours_unrounded": hours, "no_think_greedy": greedy,
            "extension": extension, "name": name, "smoke": {f"S{step}": "passed" for step in range(7)},
            "gate": f"runs/bakeoff/m7/{key}/gate.json", "gate_sha256": "0" * 64}


def variant_scores(successes: int, handwritten: int) -> dict:
    """A variant's scores over the 150 development items (eval.bakeoff.summarize_scores)."""
    return {"successes": successes, "n": 150, "handwritten": handwritten, "drafted": successes - handwritten,
            "by_type": {"call": "40/60"}}


def bakeoff_file() -> dict:
    """results/bakeoff-m7.json as scripts/bakeoff.py aggregate writes it, the parts the report reads and a few
    others, with six of its entries: the reference C0, C1 (a family that does not think), C2 (extended to
    step 548), C3 (a hybrid family, with its nothink greedy run), C5, the winner, and C6, skipped by a cut
    line."""
    extension = {"stop_step": 548, "P": 115.66666666666667, "handwritten": 37.333333333333336,
                 "no_think_greedy": None, "primary": ["think-s0", "think-s1", "think-s2"],
                 "variants": {"think-s0": variant_scores(113, 36), "think-s1": variant_scores(120, 40),
                              "think-s2": variant_scores(114, 36), "think-greedy": variant_scores(119, 38)}}
    reference = {"status": "reference", "candidate": "c0", "name": "Qwen3-4B", "model": "Qwen/Qwen3-4B",
                 "thinking": "hybrid", "mode": "no_think", "P": 107, "handwritten": 32, "eligible": True,
                 "reasons": [], "missing": [], "tokens_per_epoch": 15548908,
                 "tokens_per_s_median_100_to_stop": 445.5, "ft3_hours_estimate": 28.1,
                 "ft3_hours_unrounded": 28.115621149769293, "no_think_greedy": None, "extension": None,
                 "score": "runs/m7/g2-dev-ft3pilot.score.json", "source": "runs/m7/g2.json",
                 "train": "runs/train/ft3-pilot", "files": {}, "drafted": 75, "by_type": {"call": "47/60"}}
    candidates = {
        "c0": reference,
        "c1": bakeoff_candidate("c1", "Qwen3-4B-Instruct-2507", "none", "no_think", 102, 31,
                                32.59873783587509),
        "c2": bakeoff_candidate("c2", "Qwen3-4B-Thinking-2507", "only", "think", 111.33333333333333, 39,
                                31.83878364347357, extension=extension),
        "c3": bakeoff_candidate("c3", "Qwen3.5-4B", "hybrid", "think", 109.33333333333333, 36.666666666666664,
                                40.41300902969084, greedy=114),
        "c5": bakeoff_candidate("c5", "Nanbeige4.1-3B", "only", "think", 115.33333333333333, 36,
                                27.662558654055232),
        "c6": {"status": "skipped", "candidate": "c6", "name": "MiniCPM5-2B", "thinking": "hybrid",
               "reason": "cut line 2 (docs/m7-计划.md 砍线): wave 4 more than a day behind on 2026-10-26",
               "gate": "runs/bakeoff/m7/c6/gate.json"}}
    return {"config": "configs/bakeoff-m7.json", "config_sha256": "1" * 64,
            "preregistration": "docs/m7-底座对比预登记.md", "preregistration_sha256": "2" * 64,
            "rule": {"tie": 3, "hybrid_margin": 5, "ft3_hours": 60.0, "extend_step": 548,
                     "smoke_steps": [f"S{step}" for step in range(7)]},
            "candidates": candidates,
            "decision": {"status": "decided", "winner": "c5", "basis": "P", "mode": "think",
                         "ranking": ["c5", "c2", "c3", "c0", "c1"], "extend": None, "runner_up": "c2",
                         "margin": 4.0, "note": None, "waiting": []},
            "complete": True}


def test_a_bake_off_gives_each_candidates_scores_the_extension_the_decision_and_the_rule(tmp_path):
    """Per candidate its status, and unless it was skipped, the numbers of its entry that are not null (a
    whole score as it is, a mean over three seeds with two decimals, the FT3 estimate with one) and its rank;
    a candidate extended to step 548 also the extension's numbers and its variants' scores; the decision, the
    rule's numbers, whether the file is complete and how many candidates it has besides the reference and
    skipped. The file's other fields (paths, hashes, smoke records, reasons) are not read."""
    results(tmp_path, {"bakeoff-m7.json": bakeoff_file()})
    numbers, sources = report.build(tmp_path)
    assert [p.as_posix() for p in sources] == ["results/bakeoff-m7.json"]
    at = "bakeoff.m7"
    scores = (("think-s0", ("113", "150", "36")), ("think-s1", ("120", "150", "40")),
              ("think-s2", ("114", "150", "36")), ("think-greedy", ("119", "150", "38")))
    variants = {f"{at}.c2.extension.{name}.{key}": value for name, values in scores
                for key, value in zip(("successes", "n", "handwritten"), values, strict=True)}
    assert numbers == {
        f"{at}.complete": "yes", f"{at}.candidates": "5", f"{at}.skipped": "1",
        f"{at}.rule.tie": "3", f"{at}.rule.hybrid_margin": "5", f"{at}.rule.ft3_hours": "60.0",
        f"{at}.rule.extend_step": "548",
        f"{at}.decision.status": "decided", f"{at}.decision.winner": "c5", f"{at}.decision.basis": "P",
        f"{at}.decision.mode": "think", f"{at}.decision.runner_up": "c2", f"{at}.decision.margin": "4.00",
        f"{at}.c0.status": "reference", f"{at}.c0.P": "107", f"{at}.c0.handwritten": "32",
        f"{at}.c0.eligible": "yes", f"{at}.c0.mode": "no_think", f"{at}.c0.thinking": "hybrid",
        f"{at}.c0.ft3_hours": "28.1", f"{at}.c0.rank": "4",
        f"{at}.c1.status": "scored", f"{at}.c1.P": "102", f"{at}.c1.handwritten": "31",
        f"{at}.c1.eligible": "yes", f"{at}.c1.mode": "no_think", f"{at}.c1.thinking": "none",
        f"{at}.c1.ft3_hours": "32.6", f"{at}.c1.rank": "5",
        f"{at}.c2.status": "scored", f"{at}.c2.P": "111.33", f"{at}.c2.handwritten": "39",
        f"{at}.c2.eligible": "yes", f"{at}.c2.mode": "think", f"{at}.c2.thinking": "only",
        f"{at}.c2.ft3_hours": "31.8", f"{at}.c2.rank": "2", f"{at}.c2.extension.stop_step": "548",
        f"{at}.c2.extension.P": "115.67", f"{at}.c2.extension.handwritten": "37.33", **variants,
        f"{at}.c3.status": "scored", f"{at}.c3.P": "109.33", f"{at}.c3.handwritten": "36.67",
        f"{at}.c3.eligible": "yes", f"{at}.c3.mode": "think", f"{at}.c3.thinking": "hybrid",
        f"{at}.c3.no_think_greedy": "114", f"{at}.c3.ft3_hours": "40.4", f"{at}.c3.rank": "3",
        f"{at}.c5.status": "scored", f"{at}.c5.P": "115.33", f"{at}.c5.handwritten": "36",
        f"{at}.c5.eligible": "yes", f"{at}.c5.mode": "think", f"{at}.c5.thinking": "only",
        f"{at}.c5.ft3_hours": "27.7", f"{at}.c5.rank": "1",
        f"{at}.c6.status": "skipped"}


def test_a_bake_off_still_waiting_names_no_winner_and_a_candidate_without_a_gate_only_its_status(tmp_path):
    data = bakeoff_file()
    data["candidates"]["c4"] = {"status": "pending", "candidate": "c4", "name": "Nanbeige4-3B-Thinking-2511",
                                "thinking": "only", "gate": "runs/bakeoff/m7/c4/gate.json"}
    incomplete = bakeoff_candidate("c8", "granite-4.2-3b", "hybrid", "think", 105.66666666666667, 35,
                                   42.7099788510101)
    data["candidates"]["c8"] = {**incomplete, "status": "incomplete", "eligible": None,
                                "missing": ["no scores for nothink"], "ft3_hours_estimate": None,
                                "ft3_hours_unrounded": None}
    data["decision"] = {"status": "provisional", "standing": "decided", "winner": None, "leader": "c5",
                        "basis": "P", "mode": "think", "ranking": ["c5", "c2", "c3", "c0", "c8", "c1"],
                        "extend": None, "runner_up": "c2", "margin": 4.0, "note": None,
                        "waiting": ["c4", "c8"]}
    data["complete"] = False
    results(tmp_path, {"bakeoff-m7.json": data})
    numbers, _ = report.build(tmp_path)
    at = "bakeoff.m7"
    assert (numbers[f"{at}.complete"], numbers[f"{at}.decision.status"], numbers[f"{at}.candidates"]) == (
        "no", "provisional", "7")
    assert f"{at}.decision.winner" not in numbers and numbers[f"{at}.decision.runner_up"] == "c2"
    assert [key for key in numbers if key.startswith(f"{at}.c4.")] == [f"{at}.c4.status"]
    assert {key: value for key, value in numbers.items() if key.startswith(f"{at}.c8.")} == {
        f"{at}.c8.status": "incomplete", f"{at}.c8.P": "105.67", f"{at}.c8.handwritten": "35",
        f"{at}.c8.mode": "think", f"{at}.c8.thinking": "hybrid", f"{at}.c8.rank": "5"}
    assert numbers[f"{at}.c1.rank"] == "6"


def bakeoff_broken(change) -> dict:
    data = bakeoff_file()
    change(data)
    return data


def extension_of(data: dict) -> dict:
    return data["candidates"]["c2"]["extension"]


@pytest.mark.parametrize("data", [
    bakeoff_broken(lambda d: d["candidates"]["c1"].update(status="withdrawn")),
    bakeoff_broken(lambda d: d["decision"].update(status="maybe")),
    bakeoff_broken(lambda d: d["candidates"]["c1"].update(P="102")),
    bakeoff_broken(lambda d: d["candidates"]["c3"].update(handwritten=True)),
    bakeoff_broken(lambda d: d["candidates"]["c1"].update(eligible="yes")),
    bakeoff_broken(lambda d: d["candidates"]["c1"].update(mode="fast")),
    bakeoff_broken(lambda d: d["candidates"]["c1"].update(thinking="sometimes")),
    bakeoff_broken(lambda d: d["candidates"]["c3"].update(no_think_greedy=114.5)),
    bakeoff_broken(lambda d: d["candidates"]["c1"].update(ft3_hours_estimate="32.6")),
    bakeoff_broken(lambda d: d["candidates"]["c1"].pop("ft3_hours_estimate")),
    bakeoff_broken(lambda d: d["candidates"]["c1"].update(candidate="c9")),  # filed under another key
    bakeoff_broken(lambda d: d["candidates"]["c6"].update(P=100)),  # a cut line skips before any score
    bakeoff_broken(lambda d: extension_of(d).update(stop_step="548")),
    bakeoff_broken(lambda d: extension_of(d)["variants"]["think-s0"].update(successes=151)),
    bakeoff_broken(lambda d: extension_of(d)["variants"]["think-s0"].update(handwritten=114)),
    bakeoff_broken(lambda d: extension_of(d).update(variants=[])),
    bakeoff_broken(lambda d: d["decision"].update(ranking=["c5", "c2", "c9"])),  # not one of the file's
    bakeoff_broken(lambda d: d["decision"].update(ranking=["c5", "c2", "c2"])),
    bakeoff_broken(lambda d: d["decision"].update(ranking=["c5", "c2", "c6"])),  # a skipped one ranked
    bakeoff_broken(lambda d: d["decision"].update(winner="c6")),  # a winner the ranking does not hold
    bakeoff_broken(lambda d: d["decision"].update(mode="fast")),
    bakeoff_broken(lambda d: d["decision"].update(margin="4.0")),
    bakeoff_broken(lambda d: d["rule"].update(tie="3")),
    bakeoff_broken(lambda d: d["rule"].update(smoke_steps="S0-S6")),
    bakeoff_broken(lambda d: d.update(complete="yes")),
    bakeoff_broken(lambda d: d.update(candidates=list(d["candidates"].values()))),
], ids=["candidate-status", "decision-status", "text-P", "bool-handwritten", "text-eligible",
        "candidate-mode", "thinking-kind", "float-greedy", "text-hours", "no-hours", "filed-elsewhere",
        "skipped-with-P", "text-stop-step", "successes-above-n", "handwritten-above-successes",
        "variants-list", "ranked-unknown", "ranked-twice", "ranked-skipped", "winner-not-ranked",
        "decision-mode", "text-margin", "text-tie", "text-smoke-steps", "text-complete", "candidates-list"])
def test_a_bake_off_that_is_not_of_its_shape_is_refused(tmp_path, data):
    results(tmp_path, {"bakeoff-m7.json": data})
    with pytest.raises(report.ReportError, match="results/bakeoff-m7.json is not a bake-off aggregate"):
        report.build(tmp_path)


def test_the_m7_bake_off_gives_the_scores_and_the_decision_of_its_committed_file(tmp_path):
    """results/bakeoff-m7.json (committed): C5 wins on P, 4.00 ahead of C2, in think mode; every candidate
    that took part has its rank and P, C2 its extension to step 548; the three that a cut line skipped give
    their status alone."""
    text = (REPO / "results" / "bakeoff-m7.json").read_text(encoding="utf-8")
    results(tmp_path, {"bakeoff-m7.json": text})
    numbers, _ = report.build(tmp_path)
    at = "bakeoff.m7"
    assert (numbers[f"{at}.complete"], numbers[f"{at}.candidates"], numbers[f"{at}.skipped"]) == (
        "yes", "9", "3")
    decision = ("status", "winner", "basis", "mode", "runner_up", "margin")
    assert {key: numbers[f"{at}.decision.{key}"] for key in decision} == {
        "status": "decided", "winner": "c5", "basis": "P", "mode": "think", "runner_up": "c2",
        "margin": "4.00"}
    ranked = ("c5", "c2", "c3", "c0", "c4", "c8", "c1")
    assert [numbers[f"{at}.{key}.rank"] for key in ranked] == [str(rank) for rank in range(1, 8)]
    assert [numbers[f"{at}.{key}.P"] for key in ranked] == ["115.33", "111.33", "109.33", "107", "106.33",
                                                            "105.67", "102"]
    assert (numbers[f"{at}.c2.extension.P"], numbers[f"{at}.c2.extension.handwritten"],
            numbers[f"{at}.c2.extension.think-s1.successes"]) == ("115.67", "37.33", "120")
    assert (numbers[f"{at}.c5.ft3_hours"], numbers[f"{at}.c3.no_think_greedy"],
            numbers[f"{at}.rule.tie"]) == ("27.7", "114", "3")
    for skipped in ("c6", "c7", "c9"):
        assert {key: value for key, value in numbers.items() if key.startswith(f"{at}.{skipped}.")} == {
            f"{at}.{skipped}.status": "skipped"}


def test_bake_off_aggregates_are_registered_by_milestone_and_no_other_file_changes_its_kind():
    assert report.kind_of("bakeoff-m7.json").key == report.kind_of("bakeoff-m12.json").key == "bakeoff"
    assert [kind.key for kind in report.REGISTRY].count("bakeoff") == 1
    bakeoff = next(kind for kind in report.REGISTRY if kind.key == "bakeoff")
    for name in ("bakeoff-descriptive-m7.json", "bakeoff-test.json", "bakeoff.json", "bakeoff-m7.jsonl",
                 "lengths-acebench-m7.json", *every_kind()):
        assert not bakeoff.pattern.fullmatch(name)
    assert report.kind_of("dev-bakeoff-m7.json").key == "dev"  # a development summary's name, as before


# ---- M11: the bake-off's description and ACEBench's generated lengths (docs/m11-计划.md, 1) ----------------


def ablation_run(successes: int, handwritten: int, by_type: dict[str, tuple[int, int]] | None = None) -> dict:
    """A run of an ablation as eval.bakeoff.descriptive keeps it (RUN): its counts over the 150 development
    items and its per-type counts, {"call": (successes, n)}, or none."""
    cells = None if by_type is None else {kind: {"successes": a, "n": b} for kind, (a, b) in by_type.items()}
    return {"successes": successes, "n": 150, "handwritten": handwritten, "drafted": successes - handwritten,
            "by_type": cells}


def bakeoff_descriptive_file() -> dict:
    """results/bakeoff-descriptive-m7.json as scripts/bakeoff.py descriptive writes it, with two item types:
    C0's model trained to think (three seeds, a greedy nothink run whose gate kept no per-type counts, and the
    untrained base's run in think mode), C2's model trained not to think (no base run), and the trained
    adapter of the winner, C5, checked against the training stack."""
    c0 = {"candidate": "c0", "model": "Qwen/Qwen3-4B", "revision": "1cfa9a720891", "mode": "think",
          "gate": "runs/bakeoff/m7/c0-think/gate.json", "gate_sha256": "3" * 64, "stop_step": 274,
          "primary": ["think-s0", "think-s1", "think-s2"], "P": 99.66666666666667,
          "handwritten": 36.333333333333336, "drafted": 63.333333333333336,
          "variants": {"think-s0": ablation_run(102, 34, {"call": (95, 130), "clarify": (7, 20)}),
                       "think-s1": ablation_run(96, 38, {"call": (91, 130), "clarify": (5, 20)}),
                       "think-s2": ablation_run(101, 37, {"call": (92, 130), "clarify": (9, 20)}),
                       "nothink": ablation_run(88, 27)},
          "b0": {**ablation_run(84, 30, {"call": (78, 130), "clarify": (6, 20)}), "mode": "think",
                 "manifest": "runs/bakeoff/m7/c0-think/dev-b0.manifest.json", "manifest_sha256": "4" * 64}}
    c2 = {"candidate": "c2", "model": "Qwen/Qwen3-4B-Thinking-2507",
          "revision": "768f209d9ea81521153ed38c47d515654e938aea", "mode": "no_think",
          "gate": "runs/bakeoff/m7/c2-nothink/gate.json", "gate_sha256": "5" * 64, "stop_step": 274,
          "primary": ["nothink"], "P": 70, "handwritten": 25, "drafted": 45,
          "variants": {"nothink": ablation_run(70, 25, {"call": (62, 130), "clarify": (8, 20)})}, "b0": None}
    return {"kind": "bakeoff-descriptive", "milestone": "m7", "config": "configs/bakeoff-m7.json",
            "config_sha256": "1" * 64, "aggregate": "results/bakeoff-m7.json", "aggregate_sha256": "2" * 64,
            "ablations": {"c0-think": c0, "c2-nothink": c2},
            "fidelity": {"candidate": "c5", "record": "runs/bakeoff/m7/c5/fidelity-trained-274.json",
                         "record_sha256": "6" * 64, "trained_step": 274,
                         "settings": {"items": 20, "positions": 16, "mode": "no_think"}, "items": 20,
                         "positions": 320, "agreeing": 316, "base_agreeing": 315, "changed_positions": 20}}


def run_keys(prefix: str, successes: int, handwritten: int, **types: tuple[int, int]) -> dict[str, str]:
    """The numbers of an ablation's run over 150 items under prefix, and of its types (successes, n)."""
    numbers = {f"{prefix}.successes": str(successes), f"{prefix}.n": "150",
               f"{prefix}.handwritten": str(handwritten), f"{prefix}.drafted": str(successes - handwritten)}
    for kind, (a, b) in types.items():
        numbers.update({f"{prefix}.{kind}.successes": str(a), f"{prefix}.{kind}.n": str(b)})
    return numbers


def test_a_bake_off_description_gives_each_ablations_runs_and_the_winners_fidelity(tmp_path):
    """Per ablation its candidate, model, revision, mode and stop step, its P and means as a bake-off's
    scores (two decimals unless whole), and each run's counts: every variant of its gate, the untrained
    base's run when it has one, and per item type when the gate kept them. The winner's trained adapter: its
    step and counts, and three rates of its positions with one decimal. Paths, hashes and settings are not
    read."""
    results(tmp_path, {"bakeoff-descriptive-m7.json": bakeoff_descriptive_file()})
    numbers, sources = report.build(tmp_path)
    assert [p.as_posix() for p in sources] == ["results/bakeoff-descriptive-m7.json"]
    c0, c2, c5 = "bakeoff.m7.ablation.c0-think", "bakeoff.m7.ablation.c2-nothink", "bakeoff.m7.fidelity.c5"
    assert numbers == {
        f"{c0}.candidate": "c0", f"{c0}.model": "Qwen/Qwen3-4B", f"{c0}.revision": "1cfa9a720891",
        f"{c0}.mode": "think", f"{c0}.stop_step": "274", f"{c0}.P": "99.67", f"{c0}.handwritten": "36.33",
        f"{c0}.drafted": "63.33",
        **run_keys(f"{c0}.think-s0", 102, 34, call=(95, 130), clarify=(7, 20)),
        **run_keys(f"{c0}.think-s1", 96, 38, call=(91, 130), clarify=(5, 20)),
        **run_keys(f"{c0}.think-s2", 101, 37, call=(92, 130), clarify=(9, 20)),
        **run_keys(f"{c0}.nothink", 88, 27), **run_keys(f"{c0}.b0", 84, 30, call=(78, 130), clarify=(6, 20)),
        f"{c2}.candidate": "c2", f"{c2}.model": "Qwen/Qwen3-4B-Thinking-2507",
        f"{c2}.revision": "768f209d9ea81521153ed38c47d515654e938aea", f"{c2}.mode": "no_think",
        f"{c2}.stop_step": "274", f"{c2}.P": "70", f"{c2}.handwritten": "25", f"{c2}.drafted": "45",
        **run_keys(f"{c2}.nothink", 70, 25, call=(62, 130), clarify=(8, 20)),
        f"{c5}.trained_step": "274", f"{c5}.items": "20", f"{c5}.positions": "320", f"{c5}.agreeing": "316",
        f"{c5}.base_agreeing": "315", f"{c5}.changed_positions": "20", f"{c5}.agreement": "98.8%",
        f"{c5}.base_agreement": "98.4%", f"{c5}.changed": "6.2%"}


def test_a_fidelity_over_no_position_gives_its_counts_and_no_rate(tmp_path):
    data = bakeoff_descriptive_file()
    data["fidelity"].update(items=0, positions=0, agreeing=0, base_agreeing=0, changed_positions=0)
    results(tmp_path, {"bakeoff-descriptive-m7.json": data})
    numbers, _ = report.build(tmp_path)
    at = "bakeoff.m7.fidelity.c5"
    assert {key: value for key, value in numbers.items() if key.startswith(f"{at}.")} == {
        f"{at}.trained_step": "274", f"{at}.items": "0", f"{at}.positions": "0", f"{at}.agreeing": "0",
        f"{at}.base_agreeing": "0", f"{at}.changed_positions": "0"}


def descriptive_broken(change) -> dict:
    data = bakeoff_descriptive_file()
    change(data)
    return data


def c0_think(data: dict) -> dict:
    return data["ablations"]["c0-think"]


def c0_variant(data: dict, variant: str) -> dict:
    return data["ablations"]["c0-think"]["variants"][variant]


@pytest.mark.parametrize("data, why", [
    (descriptive_broken(lambda d: d.update(kind="bakeoff")), "kind 'bakeoff' is not 'bakeoff-descriptive'"),
    (descriptive_broken(lambda d: d.update(milestone="m8")), "the milestone 'm8' is not the file name's, m7"),
    (descriptive_broken(lambda d: d.update(ablations=list(d["ablations"].values()))),
     "is not an object of ablations by name"),
    (descriptive_broken(lambda d: d.update(ablations={})), "no ablation"),
    (descriptive_broken(lambda d: c0_think(d).update(mode="fast")), "c0-think.mode 'fast' is not one of"),
    (descriptive_broken(lambda d: c0_think(d).update(model=None)), "c0-think.model None is not a name"),
    (descriptive_broken(lambda d: c0_think(d).update(revision="")), "c0-think.revision '' is not a name"),
    (descriptive_broken(lambda d: c0_think(d).update(stop_step="274")), "'274' is not a count"),
    (descriptive_broken(lambda d: c0_think(d).update(P=99.0)), "c0-think.P 99.0 is not the mean successes"),
    (descriptive_broken(lambda d: c0_think(d).update(handwritten=36)),
     "c0-think.handwritten 36 is not the mean handwritten of think-s0, think-s1, think-s2"),
    (descriptive_broken(lambda d: c0_think(d).update(drafted="63.33")), "drafted '63.33' is not a number"),
    (descriptive_broken(lambda d: c0_think(d).update(primary=["think-s0", "think-s3"])),
     "the primary variants ['think-s0', 'think-s3'] are not distinct variants it holds"),
    (descriptive_broken(lambda d: c0_think(d).update(primary=[])), "the primary variants [] are not"),
    (descriptive_broken(lambda d: c0_think(d).update(primary=["think-s0", "think-s0"])),
     "the primary variants ['think-s0', 'think-s0'] are not"),
    (descriptive_broken(lambda d: c0_think(d).update(primary="think-s0")),
     "the primary variants 'think-s0' are not"),
    (descriptive_broken(lambda d: c0_think(d).update(variants={})), "c0-think: no variant"),
    (descriptive_broken(lambda d: c0_think(d).update(variants=list(c0_think(d)["variants"].values()))),
     "is not an object of runs per variant"),
    (descriptive_broken(lambda d: c0_variant(d, "nothink").update(successes=151, drafted=124)),
     "nothink.successes: 151 is more than the 150 it is out of"),
    (descriptive_broken(lambda d: c0_variant(d, "nothink").update(drafted=60)),
     "27 handwritten and 60 drafted successes do not make its 88"),
    (descriptive_broken(lambda d: c0_variant(d, "nothink").update(handwritten=-1, drafted=89)),
     "nothink.handwritten: -1 is not a count"),
    (descriptive_broken(lambda d: c0_variant(d, "nothink").update(successes=88.0)), "88.0 is not a count"),
    (descriptive_broken(lambda d: c0_variant(d, "think-s0")["by_type"]["call"].update(successes=94)),
     "think-s0: its item types add up to 101/150, not 102/150"),
    (descriptive_broken(lambda d: c0_variant(d, "think-s0").update(by_type={
        "call": {"successes": 82, "n": 80}, "clarify": {"successes": 20, "n": 70}})),  # they add up
     "think-s0.call.successes: 82 is more than the 80 it is out of"),
    (descriptive_broken(lambda d: c0_variant(d, "think-s0").update(by_type=["95/130", "7/20"])),
     "is not an object of counts per item type"),
    (descriptive_broken(lambda d: c0_variant(d, "think-s0").pop("by_type")), "KeyError: 'by_type'"),
    # a primary seed scored on 130 of the 150 items, its counts consistent and P still the mean of the seeds
    (descriptive_broken(lambda d: c0_variant(d, "think-s1").update(n=130, by_type={
        "call": {"successes": 91, "n": 110}, "clarify": {"successes": 5, "n": 20}})),
     ("c0-think: its runs are not over one number of items: {'think-s0': 150, 'think-s1': 130, "
      "'think-s2': 150, 'nothink': 150, 'b0': 150}")),
    (descriptive_broken(lambda d: c0_think(d)["b0"].update(n=130, by_type={
        "call": {"successes": 78, "n": 110}, "clarify": {"successes": 6, "n": 20}})),
     "c0-think: its runs are not over one number of items"),
    (descriptive_broken(lambda d: c0_think(d)["b0"].update(mode="no_think")),
     "c0-think.b0 ran in the mode 'no_think', not the ablation's, think"),
    (descriptive_broken(lambda d: c0_think(d)["variants"].update(b0=ablation_run(84, 30))),
     "c0-think.b0.successes is given twice"),  # a variant named as the base's run
    (descriptive_broken(lambda d: d["fidelity"].update(agreeing=321)),
     "c5.agreeing: 321 is more than the 320 it is out of"),
    (descriptive_broken(lambda d: d["fidelity"].update(changed_positions="20")), "'20' is not a count"),
    (descriptive_broken(lambda d: d["fidelity"].update(trained_step=274.0)), "274.0 is not a count"),
    (descriptive_broken(lambda d: d["fidelity"].update(candidate=None)),
     "fidelity.candidate None is not a name"),
    (descriptive_broken(lambda d: d.pop("fidelity")), "KeyError: 'fidelity'"),
], ids=["other-kind", "other-milestone", "ablations-list", "no-ablation", "ablation-mode", "no-model",
        "empty-revision", "text-stop-step", "P-not-the-mean", "handwritten-not-the-mean", "text-drafted",
        "primary-unknown", "no-primary", "primary-twice", "primary-text", "no-variant", "variants-list",
        "successes-above-n", "split-not-successes", "negative-handwritten", "float-successes",
        "types-not-the-run", "type-above-its-n", "by-type-list", "no-by-type", "seed-over-fewer-items",
        "base-over-fewer-items", "base-in-another-mode",
        "variant-named-b0", "agreeing-above-positions", "text-changed", "float-trained-step",
        "no-fidelity-candidate", "no-fidelity"])
def test_a_bake_off_description_that_is_not_of_its_shape_is_refused_for_its_fault(tmp_path, data, why):
    results(tmp_path, {"bakeoff-descriptive-m7.json": data})
    with pytest.raises(report.ReportError) as caught:
        report.build(tmp_path)
    assert "results/bakeoff-descriptive-m7.json is not a bake-off description" in str(caught.value)
    assert why in str(caught.value)


def test_m7s_bake_off_description_gives_the_ablations_and_the_fidelity_of_its_committed_file(tmp_path):
    """results/bakeoff-descriptive-m7.json beside the aggregate it describes, results/bakeoff-m7.json (both
    committed; docs/m7-进度.md): C0's model trained to think scores P 99.67 over its seeds and loses
    clarify when it does not think (0/20); C2's model trained not to think scores 70 and fails every no_call
    item (0/25); the winner's adapter at its trained step agrees with the training stack on 316 of 320
    positions. The two files' keys do not meet: together they give what each gives alone."""
    texts = {name: (REPO / "results" / name).read_text(encoding="utf-8")
             for name in ("bakeoff-m7.json", "bakeoff-descriptive-m7.json")}
    alone = {}
    for name, text in texts.items():
        results(tmp_path / name, {name: text})
        alone[name] = report.build(tmp_path / name)[0]
    results(tmp_path / "both", texts)
    numbers, _ = report.build(tmp_path / "both")
    assert numbers == {**alone["bakeoff-m7.json"], **alone["bakeoff-descriptive-m7.json"]}
    assert len(numbers) == len(alone["bakeoff-m7.json"]) + len(alone["bakeoff-descriptive-m7.json"])
    assert all(key.startswith(("bakeoff.m7.ablation.", "bakeoff.m7.fidelity."))
               for key in alone["bakeoff-descriptive-m7.json"])
    c0, c2 = "bakeoff.m7.ablation.c0-think", "bakeoff.m7.ablation.c2-nothink"
    ablation = ("candidate", "model", "mode", "stop_step", "P", "handwritten", "drafted")
    assert {key: numbers[f"{c0}.{key}"] for key in ablation} == {
        "candidate": "c0", "model": "Qwen/Qwen3-4B", "mode": "think", "stop_step": "274", "P": "99.67",
        "handwritten": "36.33", "drafted": "63.33"}
    runs = ("think-s0", "think-s1", "think-s2", "nothink", "b0")
    assert [numbers[f"{c0}.{run}.successes"] for run in runs] == ["102", "96", "101", "88", "84"]
    assert (numbers[f"{c0}.nothink.clarify.successes"], numbers[f"{c0}.nothink.clarify.n"]) == ("0", "20")
    assert {key: numbers[f"{c2}.{key}"] for key in ablation} == {
        "candidate": "c2", "model": "Qwen/Qwen3-4B-Thinking-2507", "mode": "no_think", "stop_step": "274",
        "P": "70", "handwritten": "25", "drafted": "45"}
    assert (numbers[f"{c2}.nothink.no_call.successes"], numbers[f"{c2}.nothink.no_call.n"]) == ("0", "25")
    assert not any(key.startswith(f"{c2}.b0.") for key in numbers)  # no base run beside C2's ablation
    winner = numbers["bakeoff.m7.decision.winner"]
    fidelity = ("trained_step", "items", "positions", "agreeing", "base_agreeing", "changed_positions",
                "agreement", "base_agreement", "changed")
    assert {key: numbers[f"bakeoff.m7.fidelity.{winner}.{key}"] for key in fidelity} == {
        "trained_step": "274", "items": "20", "positions": "320", "agreeing": "316", "base_agreeing": "315",
        "changed_positions": "20", "agreement": "98.8%", "base_agreement": "98.4%", "changed": "6.2%"}


def test_the_report_reads_the_description_by_the_names_eval_bakeoff_writes():
    assert report.BAKEOFF_DESCRIPTIVE_KIND == bakeoff_module.DESCRIPTIVE_KIND
    assert report.FIDELITY_COUNTS == bakeoff_module.FIDELITY_COUNTS
    assert {mode for _, mode in bakeoff_module.ABLATIONS.values()} <= set(report.BAKEOFF_MODES)
    assert report.kind_of(bakeoff_module.DESCRIPTIVE.name).key == "bakeoff-descriptive"


def ace_lengths_block(n: int, p50: float | None, p95: float | None, top: int | None, over: int,
                      at_max: int) -> dict:
    """A block of an ACEBench lengths summary: n requests, the percentiles and the maximum of their tokens,
    those above the reasoning budget and those at max_tokens."""
    return {"n": n, "tokens_p50": p50, "tokens_p95": p95, "tokens_max": top, "over_budget": over,
            "at_max_tokens": at_max}


def ace_lengths_file() -> dict:
    """results/lengths-acebench-m7.json as scripts/acebench_lengths.py writes it, smaller: 60 Chinese and 40
    English answers per system, B0's server probed three times (cache and thinking probes), FT's six (the
    adapter probe's three besides)."""
    return {"kind": "acebench-lengths", "definition": "G is the number of tokens llama-server generated.",
            "source": "results/acebench-m7.json", "source_sha256": "7" * 64,
            "budget": {"reasoning": 1280, "max_tokens": 2480, "from": "results/acebench-m7.json budget"},
            "languages": ["zh", "en"],
            "systems": {"b0": {"log": "runs/acebench/m7/b0/server.log", "log_sha256": "8" * 64, "probes": 3,
                               "requests": 103, "zh": ace_lengths_block(60, 1333.0, 1606.0, 2480, 44, 1),
                               "en": ace_lengths_block(40, 1321.0, 1524.4, 2310, 24, 0),
                               "all": ace_lengths_block(100, 1327.0, 1568.05, 2480, 68, 1)},
                        "ft": {"log": "runs/acebench/m7/ft/server.log", "log_sha256": "9" * 64, "probes": 6,
                               "requests": 106, "zh": ace_lengths_block(60, 318.0, 1378.0, 2480, 15, 2),
                               "en": ace_lengths_block(40, 364.0, 1388.8, 1999, 9, 0),
                               "all": ace_lengths_block(100, 343.0, 1386.0, 2480, 24, 2)}}}


ACE_LENGTH_KEYS = ("n", "tokens_p50", "tokens_p95", "tokens_max", "over_budget", "over_budget_rate",
                   "at_max_tokens", "at_max_tokens_rate")  # a part's keys


def part_keys(prefix: str, *values: str | None) -> dict[str, str]:
    """A part's numbers under prefix, in ACE_LENGTH_KEYS' order; None for one left out."""
    return {f"{prefix}.{key}": value for key, value in zip(ACE_LENGTH_KEYS, values, strict=True)
            if value is not None}


def test_acebench_lengths_give_each_systems_tokens_per_language_and_over_both(tmp_path):
    """The budget; per system its probes and requests, and per language and over both (all) the requests,
    the percentiles and maximum of their tokens as whole numbers, and those above the budget and those at
    the limit as counts and as rates of the requests. A part with no request gives no percentile, maximum or
    rate. The definition, paths and hashes are not read."""
    data = ace_lengths_file()
    data["systems"]["b4"] = {"log": "runs/acebench/m7/b4/server.log", "log_sha256": "a" * 64, "probes": 2,
                             "requests": 12, "zh": ace_lengths_block(10, 500.0, 900.0, 1000, 1, 0),
                             "en": ace_lengths_block(0, None, None, None, 0, 0),
                             "all": ace_lengths_block(10, 500.0, 900.0, 1000, 1, 0)}
    results(tmp_path, {"lengths-acebench-m7.json": data})
    numbers, sources = report.build(tmp_path)
    assert [p.as_posix() for p in sources] == ["results/lengths-acebench-m7.json"]
    at = "lengths.acebench-m7"
    assert numbers == {
        f"{at}.budget.reasoning": "1280", f"{at}.budget.max_tokens": "2480",
        f"{at}.b0.probes": "3", f"{at}.b0.requests": "103",
        **part_keys(f"{at}.b0.zh", "60", "1333", "1606", "2480", "44", "73.3%", "1", "1.7%"),
        **part_keys(f"{at}.b0.en", "40", "1321", "1524", "2310", "24", "60.0%", "0", "0.0%"),
        **part_keys(f"{at}.b0.all", "100", "1327", "1568", "2480", "68", "68.0%", "1", "1.0%"),
        f"{at}.ft.probes": "6", f"{at}.ft.requests": "106",
        **part_keys(f"{at}.ft.zh", "60", "318", "1378", "2480", "15", "25.0%", "2", "3.3%"),
        **part_keys(f"{at}.ft.en", "40", "364", "1389", "1999", "9", "22.5%", "0", "0.0%"),
        **part_keys(f"{at}.ft.all", "100", "343", "1386", "2480", "24", "24.0%", "2", "2.0%"),
        f"{at}.b4.probes": "2", f"{at}.b4.requests": "12",
        **part_keys(f"{at}.b4.zh", "10", "500", "900", "1000", "1", "10.0%", "0", "0.0%"),
        **part_keys(f"{at}.b4.en", "0", None, None, None, "0", None, "0", None),
        **part_keys(f"{at}.b4.all", "10", "500", "900", "1000", "1", "10.0%", "0", "0.0%")}


def lengths_broken(change) -> dict:
    data = ace_lengths_file()
    change(data)
    return data


def b0_lengths(data: dict) -> dict:
    return data["systems"]["b0"]


@pytest.mark.parametrize("data, why", [
    (lengths_broken(lambda d: d.update(kind="lengths")), "kind 'lengths' is not 'acebench-lengths'"),
    (lengths_broken(lambda d: d.update(languages="zh,en")), "languages 'zh,en' is not a list of names"),
    (lengths_broken(lambda d: d.update(languages=[])), "no language"),
    (lengths_broken(lambda d: d.update(systems={})), "no system"),
    (lengths_broken(lambda d: d.update(systems=list(d["systems"].values()))), "AttributeError"),
    (lengths_broken(lambda d: d.update(languages=["zh", "en", "all"])), "b0.all.n is given twice"),
    (lengths_broken(lambda d: d.update(languages=["zh", "en", "fr"])), "KeyError: 'fr'"),
    (lengths_broken(lambda d: b0_lengths(d)["zh"].update(over_budget=61)),
     "b0.zh.over_budget: 61 is more than the 60 it is out of"),
    (lengths_broken(lambda d: d["systems"]["ft"]["en"].update(at_max_tokens=41)),
     "ft.en.at_max_tokens: 41 is more than the 40 it is out of"),
    (lengths_broken(lambda d: (b0_lengths(d)["zh"].update(over_budget=-1),
                               b0_lengths(d)["all"].update(over_budget=23))),  # the sum holds
     "b0.zh.over_budget: -1 is not a count"),
    (lengths_broken(lambda d: b0_lengths(d)["all"].update(n=101)), "b0.all.n is 101, not its languages' 100"),
    (lengths_broken(lambda d: b0_lengths(d)["all"].update(over_budget=67)),
     "b0.all.over_budget is 67, not its languages' 68"),
    (lengths_broken(lambda d: b0_lengths(d)["all"].update(at_max_tokens=0)),
     "b0.all.at_max_tokens is 0, not its languages' 1"),
    (lengths_broken(lambda d: b0_lengths(d)["all"].update(tokens_max=2310)),
     "b0.all.tokens_max is 2310, not its languages' longest, 2480"),
    (lengths_broken(lambda d: d["systems"]["ft"].update(requests=105)),
     "ft.requests: 105 is not its 6 probes and 100 answers"),
    (lengths_broken(lambda d: b0_lengths(d)["zh"].update(tokens_p50="1333")),
     "'1333' is not a number of tokens"),
    (lengths_broken(lambda d: b0_lengths(d)["en"].update(tokens_max=True)), "True is not a number of tokens"),
    (lengths_broken(lambda d: b0_lengths(d)["zh"].update(n=60.0)), "60.0 is not a count"),
    (lengths_broken(lambda d: b0_lengths(d)["zh"].pop("tokens_p95")), "KeyError: 'tokens_p95'"),
    (lengths_broken(lambda d: b0_lengths(d).pop("probes")), "KeyError: 'probes'"),
    (lengths_broken(lambda d: d["budget"].update(reasoning="1280")), "'1280' is not a count"),
], ids=["other-kind", "languages-text", "no-language", "no-system", "systems-list", "language-named-all",
        "language-without-block", "over-budget-above-n", "at-max-above-n", "negative-over-budget",
        "all-n-not-the-sum", "all-over-budget-not-the-sum", "all-at-max-not-the-sum",
        "all-max-not-the-longest", "requests-not-probes-and-answers", "text-percentile", "bool-max",
        "float-n", "no-p95", "no-probes", "text-budget"])
def test_acebench_lengths_that_are_not_of_their_shape_are_refused_for_their_fault(tmp_path, data, why):
    results(tmp_path, {"lengths-acebench-m7.json": data})
    with pytest.raises(report.ReportError) as caught:
        report.build(tmp_path)
    assert "results/lengths-acebench-m7.json is not an ACEBench lengths summary" in str(caught.value)
    assert why in str(caught.value)


def test_m7s_acebench_lengths_give_the_generated_tokens_of_its_committed_file(tmp_path):
    """results/lengths-acebench-m7.json beside the ACEBench result it measures, results/acebench-m7.json (both
    committed): under a reasoning budget of 1,280 tokens B0 generated a median of 1,327 tokens per request and
    more than the budget on 66.9% of its 1,940 requests, FT 343 and 24.2%; each reached the limit of 2,480 on
    fewer than 1% of them. Each system's requests over both languages are its answers in the ACEBench
    result."""
    texts = {name: (REPO / "results" / name).read_text(encoding="utf-8")
             for name in ("acebench-m7.json", "lengths-acebench-m7.json")}
    results(tmp_path, texts)
    numbers, _ = report.build(tmp_path)
    at = "lengths.acebench-m7"
    assert (numbers[f"{at}.budget.reasoning"], numbers[f"{at}.budget.max_tokens"]) == ("1280", "2480")
    assert {system: (numbers[f"{at}.{system}.probes"], numbers[f"{at}.{system}.requests"])
            for system in ("b0", "ft")} == {"b0": ("3", "1943"), "ft": ("6", "1946")}
    parts = {system: {part: [numbers[f"{at}.{system}.{part}.{key}"] for key in ACE_LENGTH_KEYS]
                      for part in ("zh", "en", "all")} for system in ("b0", "ft")}
    assert parts == {"b0": {"zh": ["967", "1333", "1606", "2480", "717", "74.1%", "4", "0.4%"],
                            "en": ["973", "1321", "1524", "2480", "581", "59.7%", "6", "0.6%"],
                            "all": ["1940", "1327", "1568", "2480", "1298", "66.9%", "10", "0.5%"]},
                     "ft": {"zh": ["967", "318", "1378", "2480", "234", "24.2%", "8", "0.8%"],
                            "en": ["973", "364", "1389", "2480", "236", "24.3%", "5", "0.5%"],
                            "all": ["1940", "343", "1386", "2480", "470", "24.2%", "13", "0.7%"]}}
    assert len([key for key in numbers if key.startswith(f"{at}.")]) == 2 + 2 * (2 + 3 * len(ACE_LENGTH_KEYS))
    assert [numbers[f"acebench.m7.{system}.n"] for system in ("b0", "ft")] == [
        numbers[f"{at}.{system}.all.n"] for system in ("b0", "ft")]


M11_KINDS = ("bakeoff", "bakeoff-descriptive", "acebench-lengths")  # the kinds M11 registered


def test_m11s_result_files_each_have_their_kind_and_the_earlier_ones_keep_theirs():
    assert {name: report.kind_of(name).key for name in (
        "bakeoff-m7.json", "bakeoff-descriptive-m7.json", "lengths-acebench-m7.json", "lengths-m7.json",
        "acebench-m7.json")} == {
        "bakeoff-m7.json": "bakeoff", "bakeoff-descriptive-m7.json": "bakeoff-descriptive",
        "lengths-acebench-m7.json": "acebench-lengths", "lengths-m7.json": "lengths",
        "acebench-m7.json": "acebench"}
    keys = [kind.key for kind in report.REGISTRY]
    assert all(keys.count(key) == 1 for key in M11_KINDS)
    assert report.kind_of("bakeoff-descriptive-m12.json").key == "bakeoff-descriptive"
    assert report.kind_of("lengths-acebench-m12.json").key == "acebench-lengths"
    for name in ("bakeoff-descriptive.json", "bakeoff-descriptive-final.json", "bakeoff-descriptive-m7.jsonl",
                 "lengths-acebench.json", "lengths-acebench-m7-b0.json", "lengths-acebench-final.json"):
        assert report.kind_of(name) is None, name
    assert report.kind_of("acebench-lengths-m7.json").key == "acebench"  # an ACEBench result's, as before
    assert report.kind_of("dev-lengths-acebench-m7.json").key == "dev"  # a development summary's, as before


def test_every_current_result_is_named_by_one_pattern_at_most_and_keeps_its_kind():
    """Every JSON file in results/ but the test runner's (test-*) is named by at most one pattern, but for a
    development run's end-to-end report, dev-e2e-<name>.json, which dev-<name>.json's pattern names too and
    REGISTRY's order gives its own kind (as since M3); and none has another kind than the patterns before
    M11's gave it: a file they named keeps its kind, and one they did not is named by one of M11's or by
    none."""
    names = sorted(path.name for path in (REPO / "results").glob("*.json")
                   if not path.name.startswith("test-"))
    assert "numbers.json" in names  # the folder was listed
    earlier = [kind for kind in report.REGISTRY if kind.key not in M11_KINDS]
    for name in names:
        matching = [kind.key for kind in report.REGISTRY if kind.pattern.fullmatch(name)]
        assert len(matching) <= 1 or (name.startswith("dev-e2e-") and matching == ["dev-e2e", "dev"]), (
            name, matching)
        before = next((kind.key for kind in earlier if kind.pattern.fullmatch(name)), None)
        now = report.kind_of(name)
        if before is not None:
            assert now is not None and now.key == before, name
        else:
            assert now is None or now.key in M11_KINDS, name

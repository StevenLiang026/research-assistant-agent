"""scripts/build_report.py and research_agent.eval.report: results/numbers.json rebuilt from the registered
result files alone (docs: the M3 plan, phase 4). The repository's own numbers come back byte for byte, every
kind of result gives its keys (M5's routed system and its comparisons against M3's records among them), what
does not apply is left out, files no pattern names are not read, a malformed file and two files giving the
same key are refused with nothing written, and the same files give the same bytes."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from support.fakes import call, turn

from research_agent.eval import report
from research_agent.eval.e2e import CHECKS
from research_agent.eval.e2e import summarize as summarize_tasks
from research_agent.eval.metrics import ERROR_CLASSES, compare_systems, holm, score_items, summarize
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

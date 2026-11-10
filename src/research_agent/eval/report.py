"""The numbers of results/numbers.json and the result files they come from (docs/plan.md, 数字对账; the M3
plan, phase 4).

scripts/build_report.py is the only writer of results/numbers.json, and scripts/score.py --numbers runs the
same build after writing its summary. build reads every registered result file present in results/
(REGISTRY: a file's kind follows from its name, the first pattern that matches it), turns each into
README-ready numbers, and write hands them, with the SHA-256 of every file read, to
eval.numbers.write_numbers. The numbers depend on those files alone and are written sorted, so the same files
give the same bytes, whatever was registered before. A registered file that is not valid JSON, is not of its
kind or gives a key that is not KEY-shaped, and two files that would give the same key, are a ReportError,
and nothing is written. A file no pattern names is not read: the test runner's configurations
(test-config-*.json), numbers.json itself, anything in a subfolder.

    results/<file>                    written by                        keys
    dev-e2e-<name>.json               scripts/run_e2e.py (report.json)  e2e.dev-<name>.*
    dev-<name>.json                   scripts/score.py                  eval.dev-<name>.*
    test-<m>-comparisons.json         scripts/run_test.py               compare.test-<m>.<comparison>.*
    test-<m>-resources.json           scripts/run_test.py               resources.test-<m>.*
    test-<m>-e2e-<system>.json        scripts/run_test.py               e2e.test-<m>-<system>[.routing].*
    test-<m>-<system>.json            scripts/run_test.py               eval.test-<m>-<system>[-<subset>].*,
                                                                        eval.test-<m>-<system>.routing.*
    checkpoint-selection-<run>.json   scripts/select_checkpoint.py      checkpoint.<run>.*
    bfcl-<name>.json                  scripts/run_bfcl.py               bfcl.<name>.<system>.*
    backend-comparison-<name>.json    scripts/compare_backends.py       backend.<name>.*
    routing-<name>.json               scripts/route_curve.py            routing.<name>.*
    cost-oracle-<name>.json           scripts/route_cost_oracle.py      cost_oracle.<name>.*
    faults-<name>.json                scripts/fault_table.py            faults.<name>.<system>.<condition>.*
    acebench-<name>.json              scripts/run_acebench.py           acebench.<name>.<system>.*
    decision-m8.json                  scripts/decide_m8.py              decision.m8.*

<m> is a milestone in lower case (m3) and <system> a system name of letters and digits; a development run's
end-to-end report is run_e2e's report.json kept under its name. In a key, a part taken from a file name or a
field has every character outside [A-Za-z0-9_-] replaced by "-" (as scripts/score.py named its keys); a
comparison's name is taken as it is, and one that does not fit a key is refused. A fault table's <condition>
is not its row's label ("rate=0.5,persistent", which does not fit a key) but comes from the row's rate and
persistent fields: r<rate> with "p" for a persistent one (r0, r0.2, r0.5, r0.5p).

Scorers (M7, docs/m7-本地提升计划.md WP1; eval.metrics, Scorers). A scorer makes the files of the scored
kinds (Kind.scored): the single-step summaries, dev-<name>.json and test-<m>-<system>.json, and the checkpoint
selections. A file that records no "scorer", or "scorer": "default", is the default scorer's, as every file
before M7 is, and gives the keys above. A file another scorer made gives them under its name with -<scorer>
after it (results/dev-ft3.json with "scorer": "normalized": eval.dev-ft3-normalized.*), once when the name
ends so already (results/dev-ft3-normalized.json gives the same keys), so it never gives a default result's
keys, and two files that would give the same keys are refused as any two are (scored_name). A file named for
another scorer than its own (a default summary in results/dev-ft3-normalized.json) and a scorer eval.metrics
does not have are refused. A single-step file may also carry its items' summaries under other scorers beside
its own, "scorers": {"normalized": {"summary", "subsets"}}, which give the single-step numbers (in a test
runner's file the error classes too) under <name>-normalized (scorer_blocks); the default scorer, an unknown
one and a block that would give the file's own keys are refused there. To the other kinds "scorer" and a
name's ending mean nothing.

Formats. A VALUE is the string README.md quotes verbatim: a rate is a percentage with one decimal ("87.3%"),
a benchmark's accuracies (BFCL's, ACEBench's) with two ("71.23%", as scripts/run_bfcl.py and
scripts/run_acebench.py print them); a difference of two rates is in percentage points with its sign ("+12.3")
and its interval "[+5.1, +19.4]", a non-inferiority margin in percentage points without a sign ("2.0"); a
p-value has three decimals ("0.012"), "<0.001" below 0.001; an F1, a
threshold and an AUROC three decimals ("0.689", "0.674"); milliseconds and MiB are whole numbers; watts,
watt-hours and a mean step count have one decimal; RMB three decimals, fen per task two; counts are as they
are; whether a rule was met is "yes" or "no", and a name is as it is. A number that does not apply (a rate
over no item, a percentile over no value) is left out.

Keys per kind:

    single step     the summary of eval.metrics.summarize (scripts/score.py's headline numbers): n, success,
                    success.<type>, success.<lang>, json_valid, tool_choice, exact_args, query_strict,
                    false_call, think_leak, clarify, param_f1; a file with "subsets" (the test runner's) also
                    gives each subset's under <name>-<subset> (SUBSETS); the test runner's file of a routed
                    system (ftr, M5) also gives its "routing" block (scripts/run_test.py, Session.routing) as
                    routing.{n, routed, api_share, random_success, hard (the hard triggers in all),
                    hard.<kind> (each kind's), reference_success.<system>, agreement.local.{n, equal},
                    agreement.routed.{n, equal}}: n, routed, the hard triggers and the agreements are counts,
                    the others rates; every test runner's file also gives its failures per error class
                    (eval.metrics.summarize's error_classes, the plan's eleven under ERROR_KEYS' names and a
                    record status other than ok under its own) as errors.<class>, and errors.unclassified
    end to end      the summary of eval.e2e.summarize: n, success, success.<intent>, success.<lang>, the rate
                    of every check (eval.e2e.CHECKS) under its name, steps_mean, tool_errors; the test
                    runner's file of a routed system also gives its "routing" block (Session.e2e_routing), all
                    counts: routing.{tasks, tasks_routed, local_calls, cloud_calls, local_steps, cloud_steps},
                    and routing.mode (sticky or stepwise) when the file names its e2e_routing (M6) (a
                    development run's report is not read for one)
    comparisons     per comparison tested by McNemar (no "test" key, or "mcnemar"): n, success_a, success_b,
                    diff, ci (the paired bootstrap's), cluster_ci, clusters, a_only, b_only, p (McNemar's)
                    and p_holm, and with a non-inferiority bound noninferiority.{margin, paired_low,
                    cluster_low (in points), paired_above, cluster_above (yes or no)}; a comparison deferred
                    to a later milestone, or recorded from an earlier milestone's comparisons file, only p
                    and p_holm; an M6 non-inferiority comparison ("test": "noninferiority"): p_ni (the
                    one-sided paired bootstrap's), p_holm, cluster_p_ni, margin (in points, no sign), n,
                    success_a, success_b, diff, ci, cluster_ci, clusters, a_only, b_only (McNemar's p is
                    not given); the ACEBench comparison ("test": "acebench"): p_ni (the stratified
                    bootstrap's), p_holm, margin, strata, n (the answers per system), mean_a, mean_b (the
                    stratum-mean accuracies, two decimals), diff and ci (in points of the stratum-mean
                    accuracy, two decimals); a descriptive comparison ("descriptive": true, outside the
                    Holm family): the McNemar row's numbers without p and p_holm, which it must not carry; a
                    file with "rescored" (a session with reference systems, M5) also gives
                    rescored.<system>.{equal (yes or no), differ (how many ids differ)}
    resources       <system>.single_step.{n, p50_ms, p95_ms, server_p50_ms, server_p95_ms},
                    <system>.e2e_step.{n, p50_ms, p95_ms}, <system>.cost.{paid_rmb, list_rmb, ledger_rmb,
                    per_task_fen} (a cloud system), server.<server>.{vram_mib, baseline_mib, peak_mib,
                    power_mean_w, energy_wh}; a routed system, whose single step holds local, routed and
                    routed_paid (ROUTED_PARTS; Session.routed_resources), gives
                    <system>.single_step.<part>.{n, p50_ms, p95_ms} for each part (and the local part's
                    server_p50_ms and server_p95_ms) instead of <system>.single_step.*, and besides its cost
                    keys <system>.cost.single_step.{turns, paid, cached, paid_rmb, list_rmb} and
                    <system>.cost.e2e.{tasks, paid_calls, cache_hits, paid_rmb} (a cloud system's blocks of
                    those names are not taken)
    checkpoint      the chosen checkpoint's step, n, successes and dev_success, and candidates (how many were
                    compared)
    BFCL            <system>.aggregate, <system>.n, <system>.categories and <system>.<category>
    backend         n, agreement, success_hf, success_llama_cpp, diff, ci, p
    routing         signal (the chosen signal's name), threshold, auroc.<signal> (each signal's), n,
                    ft_success, b2_success, rule_met; the chosen point's routed, api_share, success,
                    random_success and cost_per_100_rmb, and oracle_success (the oracle curve's success at
                    the same number routed), none of them when the rule is not met; b2_cost_per_100_rmb
                    (routing every item: the scan's route_everything point, left out when the scan has not
                    exactly one); handwritten.{n, ft_success,
                    b2_success, b2_cost_per_100_rmb} and the chosen threshold's point on the handwritten
                    items, handwritten.{routed, api_share, success, random_success, cost_per_100_rmb};
                    latency.local_per_call.{n, p50_ms, p95_ms}, latency.cloud_per_paid_call.{n, p50_ms,
                    p95_ms}
    cost-oracle     gains (the items FT fails and B2 passes), and the curve's last point, where every gain is
                    routed: best_success and best_cost_per_100_rmb
    faults          per row: tasks, successes, success, tasks_with_faults, recovered, recovery,
                    fabricated_ids_after_empty, retry_storms, and delay_extra_ms (the delay's extra
                    milliseconds per fetch call) when the row has delayed fetch calls; a system and condition
                    given twice is refused
    ACEBench        <system>.aggregate, <system>.n, <system>.categories (over both languages), and per
                    language <system>.<lang>.aggregate, <system>.<lang>.<category> and
                    <system>.<lang>.<group> (the accuracies; a group, normal or special, of the categories
                    run), <system>.<lang>.format.<category>.{tool_call, acebench_format, n} and the same
                    summed over a group's categories run, <system>.<lang>.format.<group>.{...},
                    <system>.<lang>.inference_errors (over the categories) when not 0, and when ACEBench
                    wrote its summary row, its normal_summary and special_summary as
                    <system>.<lang>.acebench_summary.{normal, special} (its other columns are not taken:
                    its overall summary counts the agent categories, not run, as 0, and its object_short
                    and object_deep columns hold each other's accuracy)
    decision        M8's checkpoint decision (docs/m8-预登记.md, 选点; decision-m7.json, the file of
                    scripts/decide_m7.py, is of another shape and not read): decision (GO or NO-GO),
                    pg1_fallback (yes or no), candidates (how many), and when a checkpoint was chosen, chosen
                    (its step) and mg.{sum, ft3_sum, threshold, passed}; per system, b0, ft3 (the baseline,
                    decide_m8's label for FT3's step) and <run>-<step> for every candidate (<run> the name of
                    the selection's run folder: ft5-348), the dev set B counts special_n, strict, lenient,
                    valid_n, exact, false_refusals and zh_altered (the records scored again) and the rates
                    strict_rate, lenient_rate (over special_n) and exact_rate (over valid_n), for a candidate
                    only when it was run on dev set B, and then with its gates pg1 to pg4 and passes (yes or
                    no); ft3.step; for ft3 and every candidate the two-seed sums s0, s1, sum,
                    s0_without_rules, s1_without_rules and sum_without_rules; per candidate its rank (1: the
                    ranking's first)
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .metrics import (
    BAD_RECOVERY,
    BROKEN_JSON,
    DEFAULT_SCORER,
    EXTRA_PARAM,
    FABRICATED_ID,
    MISSING_PARAM,
    NO_CLARIFY,
    RETRY_STORM,
    SCORERS,
    THINK_LEAK,
    UNWANTED_CALL,
    WRONG_TOOL,
    WRONG_VALUE,
)
from .numbers import KEY, NUMBERS, write_numbers

RESULTS = NUMBERS.parent  # results/, relative to the repository root
SUBSETS = ("drafted", "handwritten")  # the single-step items by origin (in_subset)
NUMBER_RATES = (("json_valid", "json_valid_rate"), ("tool_choice", "tool_choice_accuracy"),
                ("exact_args", "exact_args_rate"), ("query_strict", "query_strict_rate"),
                ("false_call", "false_call_rate"), ("think_leak", "think_leak_rate"),
                ("clarify", "clarify_accuracy"))
_KEY = re.compile(KEY)
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


class ReportError(ValueError):
    """A registered result that cannot be turned into numbers, or two results that give the same key."""


def in_subset(item: dict, subset: str) -> bool:
    """Whether a single-step item belongs to the subset: handwritten when its source says so, drafted
    otherwise."""
    handwritten = bool((item.get("source") or {}).get("handwritten"))
    return handwritten if subset == "handwritten" else not handwritten


def label(text: str) -> str:
    """text as a part of a key: every character outside [A-Za-z0-9_-] replaced by "-"."""
    return _UNSAFE.sub("-", text)


def percent(rate: float) -> str:
    return f"{100 * rate:.1f}%"


def points(diff: float) -> str:
    """A difference of two rates in percentage points, with its sign."""
    return f"{100 * diff:+.1f}"


def interval(low: float, high: float) -> str:
    return f"[{points(low)}, {points(high)}]"


def p_value(p: float) -> str:
    if not 0 <= p <= 1:
        raise ValueError(f"{p!r} is not a p-value")
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def whole(value: float) -> str:
    return f"{value:.0f}"


def headline_numbers(summary: dict, name: str) -> dict[str, str]:
    """The README-ready numbers of a single-step summary under eval.<name>.* (module docstring)."""
    prefix = f"eval.{label(name)}"
    numbers = {f"{prefix}.n": str(summary["n"])}
    if summary["success_rate"] is not None:
        numbers[f"{prefix}.success"] = percent(summary["success_rate"])
    for group in ("by_type", "by_lang"):
        for key, cell in summary[group].items():
            if cell["n"] and cell["success_rate"] is not None:
                numbers[f"{prefix}.success.{label(key)}"] = percent(cell["success_rate"])
    for key, field in NUMBER_RATES:
        if summary[field] is not None:
            numbers[f"{prefix}.{key}"] = percent(summary[field])
    if summary["macro_param_f1"] is not None:
        numbers[f"{prefix}.param_f1"] = f"{summary['macro_param_f1']:.3f}"
    return numbers


def single_step_numbers(data: dict, name: str) -> dict[str, str]:
    numbers = headline_numbers(data["summary"], name)
    for subset in SUBSETS:
        summary = (data.get("subsets") or {}).get(subset)
        if summary is not None:
            numbers.update(headline_numbers(summary, f"{name}-{subset}"))
    for scored, block in scorer_blocks(data, name):
        numbers.update(single_step_numbers(block, scored))
    return numbers


def scored_name(name: str, data: object) -> str:
    """The name the keys of a result of a scored kind use (module docstring: Scorers): its own for a result
    the default scorer made, a file that records no "scorer" among them; for another scorer, the name with
    -<scorer> after it, unless the name ends so already. ValueError for a scorer eval.metrics does not have,
    and for a name ending in -<scorer> of a scorer other than the file's."""
    scorer = data.get("scorer", DEFAULT_SCORER) if isinstance(data, dict) else DEFAULT_SCORER
    if not isinstance(scorer, str) or scorer not in SCORERS:
        raise ValueError(f"scorer {scorer!r} is not one of {list(SCORERS)}")
    named = [s for s in SCORERS if s != DEFAULT_SCORER and label(name).endswith(f"-{s}")]
    if named and named[0] != scorer:
        raise ValueError(f"the name {name} is the {named[0]} scorer's, but the {scorer} scorer made the file")
    return name if scorer == DEFAULT_SCORER or named else f"{name}-{scorer}"


def scorer_blocks(data: dict, name: str) -> list[tuple[str, dict]]:
    """The results of a single-step file under other scorers, "scorers": {scorer: {summary, subsets}}, each
    with the name its keys use (scored_name); ValueError for the default scorer, an unknown one, and a block
    whose keys would be the file's own."""
    blocks = []
    for scorer, block in (data.get("scorers") or {}).items():
        if scorer == DEFAULT_SCORER or scorer not in SCORERS:
            raise ValueError(f"scorers holds {scorer!r}: only a scorer other than the default, one of "
                             f"{list(SCORERS)}")
        scored = scored_name(name, {"scorer": scorer})
        if scored == name:
            raise ValueError(f"the {scorer} scorer's block under scorers would give the file's own keys "
                             f"({name})")
        blocks.append((scored, block))
    return blocks


# The plan's eleven error classes (eval.metrics, 误差分类法) under ASCII names: label() would turn the Chinese
# ones into dashes alike.
ERROR_KEYS = {WRONG_TOOL: "wrong_tool", MISSING_PARAM: "missing_param", EXTRA_PARAM: "extra_param",
              WRONG_VALUE: "wrong_value", FABRICATED_ID: "fabricated_id", UNWANTED_CALL: "unwanted_call",
              NO_CLARIFY: "no_clarify", BAD_RECOVERY: "bad_recovery", RETRY_STORM: "retry_storm",
              BROKEN_JSON: "broken_json", THINK_LEAK: "think_leak"}


def error_numbers(summary: dict, name: str) -> dict[str, str]:
    """A test run's failures per error class under eval.<name>.errors.*: summarize's error_classes, the plan's
    eleven under ERROR_KEYS' names and a record status other than ok under its own, and the unclassified
    failures."""
    prefix = f"eval.{label(name)}.errors"
    numbers: dict[str, str] = {}
    for kind, count in summary["error_classes"].items():
        _put(numbers, f"{prefix}.{ERROR_KEYS.get(kind) or label(kind)}", str(count))
    _put(numbers, f"{prefix}.unclassified", str(summary["unclassified_failures"]))
    return numbers


def run_test_single_step_numbers(data: dict, name: str) -> dict[str, str]:
    """A test runner's single-step file: single_step_numbers and its failures per error class (error_numbers),
    those of its blocks under other scorers (scorer_blocks) too, and for a routed system its "routing" block
    (scripts/run_test.py, Session.routing) under eval.<name>.routing.*."""
    numbers = {**single_step_numbers(data, name), **error_numbers(data["summary"], name)}
    for scored, block in scorer_blocks(data, name):
        numbers.update(error_numbers(block["summary"], scored))
    routing = data.get("routing")
    if routing is None:
        return numbers
    prefix = f"eval.{label(name)}.routing"
    numbers.update({f"{prefix}.n": str(routing["n"]), f"{prefix}.routed": str(routing["routed"])})
    for key in ("api_share", "random_success"):
        if routing[key] is not None:
            numbers[f"{prefix}.{key}"] = percent(routing[key])
    hard = routing["hard_triggers"]
    numbers[f"{prefix}.hard"] = str(sum(hard.values()))
    for kind, count in hard.items():
        _put(numbers, f"{prefix}.hard.{label(kind)}", str(count))
    for system, success in routing["reference_success"].items():
        if success is not None:
            _put(numbers, f"{prefix}.reference_success.{label(system)}", percent(success))
    for part in ("local", "routed"):
        agreement = routing["agreement"][part]
        numbers[f"{prefix}.agreement.{part}.n"] = str(agreement["n"])
        if agreement["equal"] is not None:
            numbers[f"{prefix}.agreement.{part}.equal"] = str(agreement["equal"])
    return numbers


E2E_ROUTING = ("tasks", "tasks_routed", "local_calls", "cloud_calls", "local_steps", "cloud_steps")


def run_test_e2e_numbers(data: dict, name: str) -> dict[str, str]:
    """A test runner's end-to-end file: e2e_numbers, and for a routed system its "routing" block
    (scripts/run_test.py, Session.e2e_routing) under e2e.<name>.routing.*, with the file's e2e_routing as
    routing.mode when it names one."""
    numbers = e2e_numbers(data, name)
    routing = data.get("routing")
    if routing is not None:
        numbers.update({f"e2e.{label(name)}.routing.{key}": str(routing[key]) for key in E2E_ROUTING})
        mode = data.get("e2e_routing")
        if mode is not None:
            if not isinstance(mode, str):
                raise TypeError(f"e2e_routing {mode!r} is not a name")
            numbers[f"e2e.{label(name)}.routing.mode"] = mode
    return numbers


def e2e_numbers(data: dict, name: str) -> dict[str, str]:
    summary = data["summary"]
    prefix = f"e2e.{label(name)}"
    numbers = {f"{prefix}.n": str(summary["n"])}
    if summary["success_rate"] is not None:
        numbers[f"{prefix}.success"] = percent(summary["success_rate"])
    for group in ("by_intent", "by_lang"):
        for key, cell in summary[group].items():
            if cell["n"] and cell["success_rate"] is not None:
                numbers[f"{prefix}.success.{label(key)}"] = percent(cell["success_rate"])
    for check, rate in summary["by_check"].items():
        if rate is not None:
            numbers[f"{prefix}.{label(check)}"] = percent(rate)
    if summary["steps_mean"] is not None:
        numbers[f"{prefix}.steps_mean"] = f"{summary['steps_mean']:.1f}"
    numbers[f"{prefix}.tool_errors"] = str(summary["tool_errors"])
    return numbers


def yes_no(value: object) -> str:
    """Whether a rule was met: "yes" or "no"; TypeError for anything but true or false."""
    if not isinstance(value, bool):
        raise TypeError(f"{value!r} is not true or false")
    return "yes" if value else "no"


def _noninferiority(prefix: str, bound: dict) -> dict[str, str]:
    """A comparison's descriptive non-inferiority bound (scripts/run_test.py, noninferiority): the margin in
    percentage points without a sign, each interval's low end and whether it lies above -margin."""
    if isinstance(bound["margin"], bool) or not isinstance(bound["margin"], int | float):
        raise TypeError(f"margin {bound['margin']!r} is not a number")
    numbers = {f"{prefix}.margin": f"{100 * bound['margin']:.1f}"}
    for which in ("paired", "cluster"):
        numbers[f"{prefix}.{which}_low"] = points(bound[f"{which}_low"])
        numbers[f"{prefix}.{which}_above"] = yes_no(bound[f"{which}_above"])
    return numbers


def _paired_numbers(prefix: str, row: dict) -> dict[str, str]:
    """The numbers of a comparison eval.metrics.compare_systems made: the rates, the difference, both
    intervals, the clusters and McNemar's discordant counts."""
    paired, clustered, test = row["paired_ci"], row["cluster_ci"], row["mcnemar"]
    return {f"{prefix}.n": str(row["n"]), f"{prefix}.success_a": percent(row["success_a"]),
            f"{prefix}.success_b": percent(row["success_b"]), f"{prefix}.diff": points(row["diff"]),
            f"{prefix}.ci": interval(paired["low"], paired["high"]),
            f"{prefix}.cluster_ci": interval(clustered["low"], clustered["high"]),
            f"{prefix}.clusters": str(clustered["clusters"]),
            f"{prefix}.a_only": str(test["a_only"]), f"{prefix}.b_only": str(test["b_only"])}


def margin_points(margin: object) -> str:
    """A non-inferiority margin in percentage points without a sign."""
    if isinstance(margin, bool) or not isinstance(margin, int | float):
        raise TypeError(f"margin {margin!r} is not a number")
    return f"{100 * margin:.1f}"


def benchmark_points(diff: float) -> str:
    """A difference of benchmark accuracies in percentage points, with its sign and two decimals."""
    return f"{100 * diff:+.2f}"


def _acebench_comparison(prefix: str, row: dict) -> dict[str, str]:
    """The M6 ACEBench comparison (scripts/run_test.py, Session.acebench_comparison): the stratified
    bootstrap's one-sided p, the margin, the strata, the answers and the stratum-mean accuracies."""
    block = row["acebench"]
    paired = block["paired_ci"]
    return {f"{prefix}.p_ni": p_value(block["p_ni"]), f"{prefix}.p_holm": p_value(row["p_holm"]),
            f"{prefix}.margin": margin_points(block["margin"]), f"{prefix}.strata": str(block["strata"]),
            f"{prefix}.n": str(block["items"]), f"{prefix}.mean_a": benchmark_accuracy(block["a"]["mean"]),
            f"{prefix}.mean_b": benchmark_accuracy(block["b"]["mean"]),
            f"{prefix}.diff": benchmark_points(block["diff"]),
            f"{prefix}.ci": f"[{benchmark_points(paired['low'])}, {benchmark_points(paired['high'])}]"}


def comparison_numbers(data: dict, name: str) -> dict[str, str]:
    numbers: dict[str, str] = {}
    for row in data["comparisons"]:
        prefix = f"compare.{label(name)}.{row['name']}"
        if row.get("deferred") or row.get("recorded"):
            numbers[f"{prefix}.p"] = p_value(row["p"])
            numbers[f"{prefix}.p_holm"] = p_value(row["p_holm"])
            continue
        if row.get("descriptive"):
            if "p" in row or "p_holm" in row or "p" in (row.get("mcnemar") or {}):
                raise ValueError(f"{row['name']} is descriptive, outside the Holm family, and carries a p")
            numbers.update(_paired_numbers(prefix, row))
            continue
        test = row.get("test", "mcnemar")
        if test == "mcnemar":
            numbers[f"{prefix}.p"] = p_value(row["p"])
            numbers[f"{prefix}.p_holm"] = p_value(row["p_holm"])
            numbers.update(_paired_numbers(prefix, row))
            if row.get("noninferiority") is not None:
                numbers.update(_noninferiority(f"{prefix}.noninferiority", row["noninferiority"]))
        elif test == "noninferiority":
            bound = row["noninferiority"]
            numbers.update({f"{prefix}.p_ni": p_value(bound["p_ni"]),
                            f"{prefix}.p_holm": p_value(row["p_holm"]),
                            f"{prefix}.cluster_p_ni": p_value(bound["cluster_p_ni"]),
                            f"{prefix}.margin": margin_points(bound["margin"])})
            numbers.update(_paired_numbers(prefix, row))
        elif test == "acebench":
            numbers.update(_acebench_comparison(prefix, row))
        else:
            raise ValueError(f"{row['name']}: unknown test {test!r}")
    for system, check in (data.get("rescored") or {}).items():
        if not isinstance(check["differ"], list):
            raise TypeError(f"rescored.{system}.differ {check['differ']!r} is not a list of ids")
        at = f"compare.{label(name)}.rescored.{label(system)}"
        _put(numbers, f"{at}.equal", yes_no(check["equal"]))
        _put(numbers, f"{at}.differ", str(len(check["differ"])))
    return numbers


def _latency(prefix: str, row: dict) -> dict[str, str]:
    numbers = {f"{prefix}.n": str(row["n"])}
    for q in ("p50_ms", "p95_ms"):
        if row.get(q) is not None:
            numbers[f"{prefix}.{q}"] = whole(row[q])
    server = row.get("server")
    if server is not None:
        for q in ("p50_ms", "p95_ms"):
            if server.get(q) is not None:
                numbers[f"{prefix}.server_{q}"] = whole(server[q])
    return numbers


ROUTED_PARTS = ("local", "routed", "routed_paid")  # a routed system's single step (Session.routed_resources)
# a routed system's cost blocks: (block, its counts, its RMB amounts taken when not null)
ROUTED_COST = (("single_step", ("turns", "paid", "cached"), ("paid_rmb", "list_rmb")),
               ("e2e", ("tasks", "paid_calls", "cache_hits"), ("paid_rmb",)))


def rmb(value: float) -> str:
    return f"{value:.3f}"


def resource_numbers(data: dict, name: str) -> dict[str, str]:
    prefix = f"resources.{label(name)}"
    numbers: dict[str, str] = {}
    for system, row in data["systems"].items():
        at = f"{prefix}.{label(system)}"
        single = row.get("single_step")
        routed = single is not None and all(part in single for part in ROUTED_PARTS)
        if routed:
            for part in ROUTED_PARTS:
                numbers.update(_latency(f"{at}.single_step.{part}", single[part]))
        elif single is not None:
            numbers.update(_latency(f"{at}.single_step", single))
        if row.get("e2e_step") is not None:
            numbers.update(_latency(f"{at}.e2e_step", row["e2e_step"]))
        cost = row.get("cost")
        if cost is not None:
            for key in ("paid_rmb", "list_rmb", "ledger_rmb"):
                if cost.get(key) is not None:
                    numbers[f"{at}.cost.{key}"] = rmb(cost[key])
            if cost.get("per_task_fen") is not None:
                numbers[f"{at}.cost.per_task_fen"] = f"{cost['per_task_fen']:.2f}"
            if routed:  # a cloud system's own blocks of the same names are not taken (M3's numbers)
                for part, counts, paid in ROUTED_COST:
                    block = cost[part]
                    numbers.update({f"{at}.cost.{part}.{key}": str(block[key]) for key in counts})
                    for key in paid:
                        if block[key] is not None:
                            numbers[f"{at}.cost.{part}.{key}"] = rmb(block[key])
    for server, row in data["servers"].items():
        at = f"{prefix}.server.{label(server)}"
        for key in ("vram_mib", "baseline_mib", "peak_mib"):
            if row.get(key) is not None:
                numbers[f"{at}.{key}"] = whole(row[key])
        for key in ("power_mean_w", "energy_wh"):
            if row.get(key) is not None:
                numbers[f"{at}.{key}"] = f"{row[key]:.1f}"
    return numbers


def checkpoint_numbers(data: dict, name: str) -> dict[str, str]:
    prefix = f"checkpoint.{label(name)}"
    chosen = data["chosen"]
    return {f"{prefix}.step": str(chosen["step"]), f"{prefix}.n": str(chosen["n"]),
            f"{prefix}.successes": str(chosen["successes"]),
            f"{prefix}.dev_success": percent(chosen["successes"] / chosen["n"]),
            f"{prefix}.candidates": str(len(data["checkpoints"]))}


def bfcl_numbers(data: dict, name: str) -> dict[str, str]:
    numbers = {}
    for system, row in data["systems"].items():
        prefix = f"bfcl.{label(name)}.{label(system)}"
        aggregate = row["aggregate"]
        numbers.update({f"{prefix}.aggregate": benchmark_accuracy(aggregate["accuracy"]),
                        f"{prefix}.n": str(aggregate["n"]),
                        f"{prefix}.categories": str(aggregate["categories"])})
        for category, cell in row["categories"].items():
            numbers[f"{prefix}.{label(category)}"] = benchmark_accuracy(cell["accuracy"])
    return numbers


def _put(numbers: dict[str, str], key: str, value: str) -> None:
    """numbers[key] = value; ValueError for a key the file already gave."""
    if key in numbers:
        raise ValueError(f"{key} is given twice")
    numbers[key] = value


def third(value: float) -> str:
    """A threshold or an AUROC: three decimals."""
    return f"{value:.3f}"


def _point(prefix: str, point: dict) -> dict[str, str]:
    """A routing curve's point: routed (a count), api_share, success and random_success (rates)."""
    numbers = {f"{prefix}.routed": str(point["routed"])}
    for key in ("api_share", "success", "random_success"):
        numbers[f"{prefix}.{key}"] = percent(point[key])
    return numbers


def _everything(prefix: str, scan: list[dict]) -> dict[str, str]:
    """b2_cost_per_100_rmb, the cost per 100 items of routing every item (B2's alone), from the scan's
    route_everything point; nothing when the scan has no single such point."""
    ends = [point["cost_per_100_rmb"] for point in scan if point.get("label") == "route_everything"]
    return {f"{prefix}.b2_cost_per_100_rmb": f"{ends[0]:.3f}"} if len(ends) == 1 else {}


def routing_numbers(data: dict, name: str) -> dict[str, str]:
    prefix = f"routing.{label(name)}"
    if not isinstance(data["rule_met"], bool) or not isinstance(data["signal"], str):
        raise TypeError("rule_met is not true or false, or signal is not a name")
    numbers = {f"{prefix}.signal": data["signal"], f"{prefix}.rule_met": "yes" if data["rule_met"] else "no",
               f"{prefix}.n": str(data["n"])}
    for key in ("ft_success", "b2_success"):
        numbers[f"{prefix}.{key}"] = percent(data[key])
    for signal, auroc in data["auroc"].items():
        if auroc is not None:
            numbers[f"{prefix}.auroc.{label(signal)}"] = third(auroc)
    if data["threshold"] is not None:
        numbers[f"{prefix}.threshold"] = third(data["threshold"])
    chosen = data["chosen_point"]
    if chosen is not None:
        oracle = {point["routed"]: point for point in data["curves"]["oracle"]}
        numbers.update(_point(prefix, chosen))
        numbers[f"{prefix}.oracle_success"] = percent(oracle[chosen["routed"]]["success"])
        numbers[f"{prefix}.cost_per_100_rmb"] = f"{chosen['cost_per_100_rmb']:.3f}"
    numbers.update(_everything(prefix, data["curves"]["routing"]))
    hand = data["handwritten"]
    numbers[f"{prefix}.handwritten.n"] = str(hand["n"])
    for key in ("ft_success", "b2_success"):
        if hand[key] is not None:
            numbers[f"{prefix}.handwritten.{key}"] = percent(hand[key])
    if hand["at_threshold"] is not None:
        numbers.update(_point(f"{prefix}.handwritten", hand["at_threshold"]))
        numbers[f"{prefix}.handwritten.cost_per_100_rmb"] = f"{hand['at_threshold']['cost_per_100_rmb']:.3f}"
    numbers.update(_everything(f"{prefix}.handwritten", hand["scan"]))
    for part in ("local_per_call", "cloud_per_paid_call"):
        numbers.update(_latency(f"{prefix}.latency.{part}", data["latency_ms"][part]))
    # the M6 file (scripts/route_curve.py --objective) names its objective and chosen policy; the M4 file has
    # neither. scripts/route_validate.py adds the frozen point's results on another set under "validation"
    for key in ("objective", "chosen_policy"):
        if data.get(key) is not None:
            numbers[f"{prefix}.{'policy' if key == 'chosen_policy' else key}"] = str(data[key])
    for set_name, block in (data.get("validation") or {}).items():
        count, threshold = block["n"], block["threshold"]
        if any(isinstance(v, bool) for v in (count, threshold)) or not isinstance(count, int) \
                or not isinstance(threshold, (int, float)):
            raise TypeError(f"validation.{set_name}: n is not a count or threshold is not a number")
        seen = f"{prefix}.validation.{label(set_name)}"
        numbers[f"{seen}.n"] = str(block["n"])
        if block.get("policy") is not None:
            numbers[f"{seen}.policy"] = str(block["policy"])
        numbers[f"{seen}.threshold"] = third(block["threshold"])
        numbers.update(_point(seen, block))
        numbers[f"{seen}.cost_per_100_rmb"] = f"{block['cost_per_100_rmb']:.3f}"
        for key in ("ft_success", "b2_success"):
            numbers[f"{seen}.{key}"] = percent(block[key])
    return numbers


def cost_oracle_numbers(data: dict, name: str) -> dict[str, str]:
    """scripts/route_cost_oracle.py's oracle at a cost: its gains, and the success and cost of its last point,
    where every gain is routed."""
    prefix = f"cost_oracle.{label(name)}"
    gains, best = data["gains"], data["curve"][-1]
    if isinstance(gains, bool) or not isinstance(gains, int) or gains != len(data["curve"]) - 1:
        raise TypeError(f"gains {gains!r} is not the number of the curve's points after the first")
    return {f"{prefix}.gains": str(gains), f"{prefix}.best_success": percent(best["success"]),
            f"{prefix}.best_cost_per_100_rmb": f"{best['cost_per_100_rmb']:.3f}"}


def fault_condition(rate: float, persistent: bool) -> str:
    """A fault table row's condition as a part of a key: r<rate>, with "p" for a persistent one (r0.5p)."""
    if isinstance(rate, bool) or not isinstance(rate, int | float) or not isinstance(persistent, bool):
        raise TypeError(f"rate {rate!r} is not a number or persistent {persistent!r} is not true or false")
    return f"r{rate:g}" + ("p" if persistent else "")


def fault_numbers(data: dict, name: str) -> dict[str, str]:
    numbers: dict[str, str] = {}
    for row in data["rows"]:
        condition = fault_condition(row["rate"], row["persistent"])
        prefix = f"faults.{label(name)}.{label(row['system'])}.{condition}"
        if f"{prefix}.tasks" in numbers:
            raise ValueError(f"{prefix} is given twice")
        for key in ("tasks", "successes", "tasks_with_faults", "recovered", "fabricated_ids_after_empty",
                    "retry_storms"):
            numbers[f"{prefix}.{key}"] = str(row[key])
        for key, field in (("success", "success_rate"), ("recovery", "recovery_rate")):
            if row[field] is not None:
                numbers[f"{prefix}.{key}"] = percent(row[field])
        delay = row["delay"]
        if delay["delayed_fetch_calls"] and delay["extra_ms_per_call"] is not None:
            numbers[f"{prefix}.delay_extra_ms"] = whole(delay["extra_ms_per_call"])
    return numbers


def benchmark_accuracy(accuracy: float) -> str:
    """A benchmark's accuracy as scripts/run_bfcl.py and scripts/run_acebench.py print it: two decimals."""
    return f"{100 * accuracy:.2f}%"


def acebench_numbers(data: dict, name: str) -> dict[str, str]:
    numbers: dict[str, str] = {}
    for system, row in data["systems"].items():
        prefix = f"acebench.{label(name)}.{label(system)}"
        aggregate = row["aggregate"]
        _put(numbers, f"{prefix}.aggregate", benchmark_accuracy(aggregate["accuracy"]))
        _put(numbers, f"{prefix}.n", str(aggregate["n"]))
        _put(numbers, f"{prefix}.categories", str(aggregate["categories"]))
        for language, block in row["languages"].items():
            at = f"{prefix}.{label(language)}"
            _put(numbers, f"{at}.aggregate", benchmark_accuracy(block["aggregate"]["accuracy"]))
            for category, cell in block["categories"].items():
                _put(numbers, f"{at}.{label(category)}", benchmark_accuracy(cell["accuracy"]))
            for group, cell in block["groups"].items():
                _put(numbers, f"{at}.{label(group)}", benchmark_accuracy(cell["accuracy"]))
            for category, counts in block["format"].items():
                for key in ("tool_call", "acebench_format", "n"):
                    _put(numbers, f"{at}.format.{label(category)}.{key}", str(counts[key]))
            for group, categories in data["groups"].items():
                run = [block["format"][category] for category in categories if category in block["format"]]
                for key in ("tool_call", "acebench_format", "n") if run else ():
                    _put(numbers, f"{at}.format.{label(group)}.{key}", str(sum(cell[key] for cell in run)))
            errors = sum(block["inference_errors"].values())
            if errors:
                _put(numbers, f"{at}.inference_errors", str(errors))
            summary = block["acebench_summary"]
            if summary is not None:
                for group in ("normal", "special"):
                    _put(numbers, f"{at}.acebench_summary.{group}",
                         benchmark_accuracy(summary[f"{group}_summary"]))
    return numbers


def backend_numbers(data: dict, name: str) -> dict[str, str]:
    prefix = f"backend.{label(name)}"
    runs, difference = data["runs"], data["success_difference"]
    return {f"{prefix}.n": str(data["n"]), f"{prefix}.agreement": percent(data["agreement"]["rate"]),
            f"{prefix}.success_hf": percent(runs["hf"]["success_rate"]),
            f"{prefix}.success_llama_cpp": percent(runs["llama.cpp"]["success_rate"]),
            f"{prefix}.diff": points(difference["diff"]),
            f"{prefix}.ci": interval(difference["low"], difference["high"]),
            f"{prefix}.p": p_value(data["mcnemar"]["p"])}


DEV_B_COUNTS = ("special_n", "strict", "lenient", "valid_n", "exact", "false_refusals", "zh_altered")
DEV_B_RATES = (("strict", "special_n"), ("lenient", "special_n"), ("exact", "valid_n"))  # (count, over)
TWO_SEED_SUMS = ("s0", "s1", "sum", "s0_without_rules", "s1_without_rules", "sum_without_rules")
DECISION_GATES = ("PG1", "PG2", "PG3", "PG4")
DECISIONS = ("GO", "NO-GO")


def count_text(value: object) -> str:
    """A count as README quotes it; TypeError for anything but a whole number (true and false included)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{value!r} is not a count")
    return str(value)


def _dev_set_b(prefix: str, counts: dict) -> dict[str, str]:
    """A system's dev set B counts as scripts/decide_m8.py recorded them (its records scored again), and
    their rates."""
    numbers = {f"{prefix}.{key}": count_text(counts[key]) for key in DEV_B_COUNTS}
    for key, total in DEV_B_RATES:
        numbers[f"{prefix}.{key}_rate"] = percent(counts[key] / counts[total])
    return numbers


def decision_numbers(data: dict, name: str) -> dict[str, str]:
    """M8's checkpoint decision (scripts/decide_m8.py) under decision.<name>.* (module docstring); ValueError
    for a verdict other than GO and NO-GO, and for a ranking that does not hold every candidate once."""
    prefix = f"decision.{label(name)}"
    if data["decision"] not in DECISIONS:
        raise ValueError(f"decision {data['decision']!r} is not one of {list(DECISIONS)}")
    candidates, ranking = data["candidates"], data["ranking"]
    steps = [row["step"] for row in candidates]
    for step in steps:
        count_text(step)
    if len(set(steps)) != len(steps) or sorted(ranking) != sorted(steps):
        raise ValueError(f"the ranking {ranking} does not hold every candidate's step {steps} once")
    numbers = {f"{prefix}.decision": data["decision"], f"{prefix}.pg1_fallback": yes_no(data["pg1_fallback"]),
               f"{prefix}.candidates": str(len(candidates))}
    numbers.update(_dev_set_b(f"{prefix}.b0", data["b0"]["special_dev"]))
    baseline = data["baseline"]
    numbers[f"{prefix}.ft3.step"] = count_text(baseline["step"])
    numbers.update({f"{prefix}.ft3.{key}": count_text(baseline[key]) for key in TWO_SEED_SUMS})
    numbers.update(_dev_set_b(f"{prefix}.ft3", baseline["special_dev"]))
    run = label(Path(data["selection"]["run"]).name)
    for row in candidates:
        at = f"{prefix}.{run}-{row['step']}"
        numbers[f"{at}.rank"] = str(ranking.index(row["step"]) + 1)
        numbers.update({f"{at}.{key}": count_text(row[key]) for key in TWO_SEED_SUMS})
        if row["special_dev"] is not None:  # run on dev set B
            numbers.update(_dev_set_b(at, row["special_dev"]))
            for gate in DECISION_GATES:
                numbers[f"{at}.{gate.lower()}"] = yes_no(row["gates"][gate]["passed"])
            numbers[f"{at}.passes"] = yes_no(row["passes"])
    if data["chosen"] is not None:
        numbers[f"{prefix}.chosen"] = count_text(data["chosen"]["step"])
    mg = data["mg"]
    if mg is not None:
        numbers.update({f"{prefix}.mg.{key}": count_text(mg[key]) for key in ("sum", "ft3_sum", "threshold")})
        numbers[f"{prefix}.mg.passed"] = yes_no(mg["passed"])
    return numbers


@dataclass(frozen=True)
class Kind:
    """A kind of registered result: what it is (for messages), the pattern of its file name, its numbers
    and the name its keys use, taken from the match; scored when a scorer makes its files (module docstring:
    Scorers), so that the name its keys use follows the file's scorer (scored_name)."""

    key: str
    what: str
    pattern: re.Pattern
    numbers: Callable[[dict, str], dict[str, str]]
    name: Callable[[re.Match], str]
    scored: bool = False


def _test_name(match: re.Match) -> str:
    return "-".join(("test", *match.groups()))


REGISTRY = (
    Kind("dev-e2e", "an end-to-end summary", re.compile(r"dev-e2e-(.+)\.json"), e2e_numbers,
         lambda m: f"dev-{m.group(1)}"),
    Kind("dev", "a single-step summary", re.compile(r"(dev-.+)\.json"), single_step_numbers,
         lambda m: m.group(1), scored=True),
    Kind("comparisons", "a comparisons file", re.compile(r"test-(m\d+)-comparisons\.json"),
         comparison_numbers, _test_name),
    Kind("resources", "a resources file", re.compile(r"test-(m\d+)-resources\.json"), resource_numbers,
         _test_name),
    Kind("test-e2e", "an end-to-end summary", re.compile(r"test-(m\d+)-e2e-([a-z0-9]+)\.json"),
         run_test_e2e_numbers, lambda m: f"test-{m.group(1)}-{m.group(2)}"),
    Kind("test", "a single-step summary", re.compile(r"test-(m\d+)-([a-z0-9]+)\.json"),
         run_test_single_step_numbers, _test_name, scored=True),
    Kind("checkpoint", "a checkpoint selection", re.compile(r"checkpoint-selection-(.+)\.json"),
         checkpoint_numbers, lambda m: m.group(1), scored=True),
    Kind("bfcl", "a BFCL result", re.compile(r"bfcl-(.+)\.json"), bfcl_numbers, lambda m: m.group(1)),
    Kind("backend", "a backend comparison", re.compile(r"backend-comparison-(.+)\.json"), backend_numbers,
         lambda m: m.group(1)),
    Kind("routing", "a routing configuration", re.compile(r"routing-(.+)\.json"), routing_numbers,
         lambda m: m.group(1)),
    Kind("cost-oracle", "an oracle at a cost", re.compile(r"cost-oracle-(.+)\.json"), cost_oracle_numbers,
         lambda m: m.group(1)),
    Kind("faults", "a fault table", re.compile(r"faults-(.+)\.json"), fault_numbers, lambda m: m.group(1)),
    Kind("acebench", "an ACEBench result", re.compile(r"acebench-(.+)\.json"), acebench_numbers,
         lambda m: m.group(1)),
    Kind("decision", "a checkpoint decision", re.compile(r"decision-(m8)\.json"), decision_numbers,
         lambda m: m.group(1)),
)


def kind_of(name: str) -> Kind | None:
    """The kind of a file in results/ by its name (REGISTRY, the first pattern that matches); None when no
    pattern names it."""
    return next((kind for kind in REGISTRY if kind.pattern.fullmatch(name)), None)


def registered(root: str | Path = ".") -> list[tuple[Path, Kind]]:
    """The registered result files in root/results, as paths relative to root, in name order, with their
    kind."""
    folder = Path(root) / RESULTS
    names = sorted(path.name for path in folder.glob("*.json")) if folder.is_dir() else []
    return [(RESULTS / name, kind) for name in names if (kind := kind_of(name)) is not None]


def _numbers_of(kind: Kind, data: object, name: str) -> dict[str, str]:
    """kind.numbers(data, name), checked: string values under KEY-shaped keys."""
    numbers = kind.numbers(data, name)
    for key, value in numbers.items():
        if not _KEY.fullmatch(key):
            raise ValueError(f"{key!r} is not a key ({KEY})")
        if not isinstance(value, str):
            raise TypeError(f"{key}: {value!r} is not a string")
    return numbers


def build(root: str | Path = ".") -> tuple[dict[str, str], list[Path]]:
    """The numbers of every registered result under root/results and the files they came from, relative to
    root, in name order (module docstring); ReportError for a file that cannot be read as its kind and for a
    key two files give."""
    root = Path(root)
    numbers: dict[str, str] = {}
    origin: dict[str, str] = {}
    sources = []
    for relative, kind in registered(root):
        shown = relative.as_posix()
        try:
            data = json.loads((root / relative).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            raise ReportError(f"{shown} cannot be read: {exc}") from None
        except ValueError as exc:
            raise ReportError(f"{shown} is not valid JSON: {exc}") from None
        try:
            name = kind.name(kind.pattern.fullmatch(relative.name))
            found = _numbers_of(kind, data, scored_name(name, data) if kind.scored else name)
        except (KeyError, TypeError, ValueError, AttributeError, IndexError, ZeroDivisionError) as exc:
            raise ReportError(f"{shown} is not {kind.what}: {type(exc).__name__}: {exc}") from None
        for key, value in found.items():
            if key in numbers:
                raise ReportError(f"{key} comes from both {origin[key]} and {shown}")
            numbers[key], origin[key] = value, shown
        sources.append(relative)
    return numbers, sources


def write(root: str | Path = ".", path: str | Path = NUMBERS, *,
          ignored: Callable[[str], bool] | None = None) -> dict:
    """Build the numbers (build) and write them to root/path with the SHA-256 of every source
    (eval.numbers.write_numbers, which refuses a source git ignores: ValueError); returns what was written."""
    numbers, sources = build(root)
    return write_numbers(numbers, sources, path, root=root, ignored=ignored)

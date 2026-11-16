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
    lengths-<m>.json                  scripts/thinking_lengths.py       lengths.<m>.<system>.*
    lengths-acebench-<m>.json         scripts/acebench_lengths.py       lengths.acebench-<m>.*
    bakeoff-<m>.json                  scripts/bakeoff.py aggregate      bakeoff.<m>.*
    bakeoff-descriptive-<m>.json      scripts/bakeoff.py descriptive    bakeoff.<m>.ablation.*,
                                                                        bakeoff.<m>.fidelity.*

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
and the interval of a rate (a Wilson interval) has both its ends so ("[83.5%, 90.9%]"); a benchmark's
accuracies (BFCL's, ACEBench's) have two decimals ("71.23%", as scripts/run_bfcl.py and
scripts/run_acebench.py print them); a difference of two rates is in percentage points with its sign ("+12.3")
and its interval "[+5.1, +19.4]", a difference of benchmark accuracies with two decimals ("-3.96"), a
non-inferiority margin in percentage points without a sign ("2.0"); a
p-value has three decimals ("0.012"), "<0.001" below 0.001; an F1, a
threshold and an AUROC three decimals ("0.689", "0.674"); milliseconds, MiB and tokens (their percentiles
included) are whole numbers; watts,
watt-hours, a mean step count and hours have one decimal; RMB three decimals, fen per task two; a bake-off's
score (successes, or their mean over a family's three seeds) is a whole number as it is or has two decimals
("115.33"), as has its decision's margin ("4.00"); counts are as they
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
                    development run's report is not read for one); a development run's report also gives,
                    when its manifest counted the local steps longer than the training length (M7;
                    scripts/run_e2e.py, StepLengths; a null count, a run without the context check, gives
                    none of the three), training_length, steps_over_training_length and local_calls (the
                    local model calls, summed over its task records), and when it voted (--votes), its failed
                    vote candidates as candidate_errors (in all) and candidate_errors.<type> (each error
                    type's), all counts
    votes, rules    a test runner's file, single step or end to end, of a voted or ruled system (M7's final;
                    scripts/run_test.py, wrapper_counts, descriptive) also gives its "votes" and "rules"
                    blocks under eval.<name>.* and e2e.<name>.*: votes.{steps, unanimous, chosen_not_greedy,
                    no_sample_voted, candidate_errors}, counts, and votes.agreement (the mean share of the
                    samples in the chosen class) as a rate, left out when no sample voted; rules.steps and,
                    per rule of the block (agent.rules.RULES), rules.fired.<rule>, the answers it changed
    Wilson          every test runner's file, single step or end to end, also gives success_wilson (M11), the
                    Wilson 95% interval of its success rate (rate_interval of eval.metrics.wilson_interval;
                    docs/m7-预登记.md, 成功判据: 区间), over the successes its summary's success_rate and n
                    give back, k = round(success_rate * n), whose rate k / n must be within 1e-9 of
                    success_rate (a tolerance on the rate, n * 1e-9 on the count; a rate no whole number of
                    successes gives is refused); a single-step file for its own summary and for each block
                    under another scorer (eval.<name>[-<scorer>].success_wilson, beside success); left out
                    over no item
    modes           a test runner's file, single step or end to end, that holds "modes" (scripts/run_test.py;
                    B2's: the turns its cloud model took per thinking mode, llm.cloud's thinking and
                    non-thinking) also gives them as counts under eval.<name>.modes.<mode> and
                    e2e.<name>.modes.<mode>, once per file whatever its scorer blocks
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
                    and object_deep columns hold each other's accuracy); a file that holds both ft and b0
                    (M11) also gives FT's accuracy minus B0's per stratum, correct / n of each exactly, in
                    points: ft-vs-b0.<lang>.<category>, ft-vs-b0.<lang>.<group> and ft-vs-b0.<lang>.aggregate
                    (the mean of those over a group's categories run and over the language's),
                    ft-vs-b0.aggregate (over every (language, category) stratum: H3's difference, the mean
                    over the strata that scripts/run_test.py's ACEBench comparison takes) and the counts
                    ft-vs-b0.{strata, ft_higher, b0_higher, equal}; two systems that did not run the same
                    strata with the same n, or a stratum whose accuracy is not its correct / n, are refused
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
    lengths         the thinking lengths of a test session's systems and their steps over the training
                    length (M7's preregistered description; scripts/thinking_lengths.py, aggregates only),
                    from {"systems": {<system>: {"single_step": {...}, "e2e": {...}, "e2e_length": {...}}}},
                    each block's fields named as its keys: for each of the two parts a system has, single_step
                    (its single-step records) and e2e (every attempt of its end-to-end transcripts),
                    <system>.<part>.{n (the turns with a thinking record), reasoning_p50, reasoning_p95,
                    reasoning_max, answer_p50, answer_p95 (tokens; null over no turn, then left out), forced
                    (the reasonings closed by force)} and <system>.<part>.forced_rate (forced over n, computed
                    here; left out for n 0), and from its e2e_length block
                    <system>.e2e_length.{training_length (in tokens), calls (the local calls measured), over
                    (those longer than the training length), max (the longest, in tokens; null without a
                    call, then left out)}, all counts; a system with none of the three blocks is refused, and
                    so is a count larger than the turns or calls it is out of
    ACE lengths     the tokens llama-server generated for each request of an ACEBench run (M11;
                    scripts/acebench_lengths.py, from its servers' logs; aggregates only): the reasoning, the
                    tag that closes it and the answer together, which the logs do not split. Under
                    lengths.acebench-<m>.*: budget.{reasoning, max_tokens} (the run's reasoning budget and the
                    limit at which a server stops a generation, in tokens); per system <system>.probes (the
                    probe requests left out) and <system>.requests (every request of its log, the probes
                    included), and for each language of the file and for all of them together (all)
                    <system>.<part>.{n (the requests), tokens_p50, tokens_p95, tokens_max (tokens; null over
                    no request, then left out), over_budget (those that generated more tokens than the
                    reasoning budget), at_max_tokens (those that reached the limit)}, counts, and
                    <system>.<part>.over_budget_rate and at_max_tokens_rate (over n, computed here; left out
                    for n 0). Another kind, languages that are not a list of names, no language or system, a
                    negative count, one larger than the requests it is out of, an all block that is not its
                    languages together (n and the two counts summed, tokens_max the longest) and requests
                    that are not the probes and the answers are refused
    bake-off       M7's base-model bake-off (eval.bakeoff.aggregate; docs/m7-底座对比预登记.md): complete
                    (yes or no), candidates (how many besides the reference C0, the skipped included) and
                    skipped; per candidate <c>.status (reference, scored, incomplete, ineligible, pending or
                    skipped), and for one that is neither pending nor skipped the fields of its entry that
                    are not null: <c>.P (the primary score), <c>.handwritten, <c>.eligible (yes or no),
                    <c>.mode, <c>.thinking, <c>.no_think_greedy, <c>.ft3_hours (the FT3 estimate as its gate
                    shows it), <c>.rank (1: the decision's ranking's first) and, extended to the extension
                    step, <c>.extension.{stop_step, P, handwritten, no_think_greedy} and per variant
                    <c>.extension.<variant>.{successes, n, handwritten}; decision.{status, winner, basis,
                    mode, runner_up, margin}, those not null, and rule.<field> for each number of the rule
                    (the names of its smoke steps are not taken); the paths, hashes, smoke records and
                    reasons are not read. A status, mode or thinking kind the aggregate does not write, a
                    candidate filed under another key, a ranking that names one twice, one the file does not
                    have or one pending or skipped, and a winner or runner-up it does not rank are refused
    description     what M7 ran beside its bake-off, report only (M11; scripts/bakeoff.py descriptive,
                    eval.bakeoff.descriptive), under bakeoff.<m>.* beside the aggregate's keys. Per ablation
                    <a> (its folder's name, c0-think): ablation.<a>.{candidate (the key of the candidate whose
                    model it trained), model, revision (names), mode (think or no_think), stop_step}, and its
                    scores as a bake-off's, ablation.<a>.{P (the mean successes over its primary variants),
                    handwritten, drafted (their means on the handwritten and on the drafted items)}; per run,
                    each variant of its gate and the untrained base's b0 when it has one,
                    ablation.<a>.<run>.{successes, n, handwritten, drafted} and, when the gate counted them
                    per item type, ablation.<a>.<run>.<type>.{successes, n}, all counts. The winner <c>'s
                    trained adapter's fidelity at the step trained (eval.bakeoff.trained_step):
                    fidelity.<c>.{trained_step, items, positions, agreeing (the positions where llama.cpp's
                    top-1 token with the adapter is the training stack's), base_agreeing (the same without
                    the adapter), changed_positions (those where the adapter changed the training stack's
                    top-1 token)}, counts, and agreement, base_agreement and changed, those three over the
                    positions as rates (left out over no position). The paths, hashes and settings are not
                    read. Another kind, another milestone than the file name's, no ablation, a mode the
                    bake-off does not write, primary variants that are none, named twice or not the
                    ablation's, a P or mean that is not the mean over them, a negative count, a run with more
                    successes than items or whose handwritten and drafted successes do not make its
                    successes, per-type counts that do not add up to the run's, runs of an ablation (its
                    base's included) over different numbers of items, a base run in another mode than the
                    ablation's, and a count larger than the positions it is out of are refused
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
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
    wilson_interval,
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


def rate_interval(low: float, high: float) -> str:
    """An interval of a rate, such as a Wilson interval (eval.metrics.wilson_interval): its two ends as
    percentages with one decimal, "[83.5%, 90.9%]", as docs/m7-预登记.md writes them; ValueError unless
    0 <= low <= high <= 1."""
    if not 0 <= low <= high <= 1:
        raise ValueError(f"[{low!r}, {high!r}] is not an interval of a rate")
    return f"[{percent(low)}, {percent(high)}]"


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


# A voted system's "votes" block (scripts/run_test.py, wrapper_counts): its counts; "agreement" is a rate.
VOTE_COUNTS = ("steps", "unanimous", "chosen_not_greedy", "no_sample_voted", "candidate_errors")


def wrapper_numbers(data: dict, prefix: str) -> dict[str, str]:
    """The "votes" and "rules" blocks of a voted or ruled system's test runner file (scripts/run_test.py,
    wrapper_counts: over its records, or over the attempts of its transcripts) under <prefix>.votes.* and
    <prefix>.rules.* (module docstring); nothing for a file without them. TypeError for a count that is not a
    whole number and an agreement that is not a number, ValueError for an agreement outside [0, 1]."""
    numbers: dict[str, str] = {}
    votes = data.get("votes")
    if votes is not None:
        numbers.update({f"{prefix}.votes.{key}": count_text(votes[key]) for key in VOTE_COUNTS})
        agreement = votes["agreement"]
        if agreement is not None:
            if isinstance(agreement, bool) or not isinstance(agreement, int | float):
                raise TypeError(f"votes.agreement {agreement!r} is not a number")
            if not 0 <= agreement <= 1:
                raise ValueError(f"votes.agreement {agreement!r} is not a share")
            numbers[f"{prefix}.votes.agreement"] = percent(agreement)
    rules = data.get("rules")
    if rules is not None:
        numbers[f"{prefix}.rules.steps"] = count_text(rules["steps"])
        for rule, fired in rules["fired"].items():
            _put(numbers, f"{prefix}.rules.fired.{label(rule)}", count_text(fired))
    return numbers


def wilson_numbers(summary: dict, prefix: str) -> dict[str, str]:
    """<prefix>.success_wilson: the Wilson 95% interval of a test run's success rate (rate_interval of
    eval.metrics.wilson_interval; docs/m7-预登记.md, 成功判据: 区间) over the successes that the summary's
    success_rate and n give back, round(success_rate * n); nothing over no item (n 0) or without a rate.
    ReportError when those successes over n are more than 1e-9 off the rate (no whole number of successes
    gives it), TypeError for an n that is not a whole number and a rate that is not a number, ValueError for
    a rate that is not finite or outside [0, 1] (eval.metrics.wilson_interval: successes outside 0..n)."""
    n, rate = summary["n"], summary["success_rate"]
    count_text(n)
    if n == 0 or rate is None:
        return {}
    if isinstance(rate, bool) or not isinstance(rate, int | float):
        raise TypeError(f"success_rate {rate!r} is not a rate")
    if not math.isfinite(rate):
        raise ValueError(f"success_rate {rate!r} is not a rate")
    successes = round(rate * n)
    if abs(successes / n - rate) > 1e-9:
        raise ReportError(f"the success rate {rate!r} over {n} items is no whole number of successes")
    return {f"{prefix}.success_wilson": rate_interval(*wilson_interval(successes, n))}


def mode_numbers(data: dict, prefix: str) -> dict[str, str]:
    """A test runner's turns per thinking mode (scripts/run_test.py, "modes": the turns its cloud model took
    in each mode, llm.cloud's thinking and non-thinking; B2's, which docs/m7-预登记.md reports) under
    <prefix>.modes.<mode>, from the file's top level, once per file; nothing for a file without them.
    TypeError for "modes" that is not an object and a count that is not a whole number, ValueError for a
    negative one."""
    modes = data.get("modes")
    if modes is None:
        return {}
    if not isinstance(modes, dict):
        raise TypeError(f"modes {modes!r} is not an object of counts per mode")
    numbers: dict[str, str] = {}
    for mode, count in modes.items():
        text = count_text(count)
        if count < 0:
            raise ValueError(f"modes.{mode}: {count} is not a count")
        _put(numbers, f"{prefix}.modes.{label(mode)}", text)
    return numbers


def run_test_single_step_numbers(data: dict, name: str) -> dict[str, str]:
    """A test runner's single-step file: single_step_numbers, its failures per error class (error_numbers) and
    the Wilson interval of its success rate (wilson_numbers), those of its blocks under other scorers
    (scorer_blocks) too, a voted or ruled system's votes and rules (wrapper_numbers) and its turns per
    thinking mode (mode_numbers) under eval.<name>.*, and for a routed system its "routing" block
    (scripts/run_test.py, Session.routing) under eval.<name>.routing.*."""
    at = f"eval.{label(name)}"
    numbers = {**single_step_numbers(data, name), **error_numbers(data["summary"], name),
               **wilson_numbers(data["summary"], at)}
    for scored, block in scorer_blocks(data, name):
        numbers.update(error_numbers(block["summary"], scored))
        numbers.update(wilson_numbers(block["summary"], f"eval.{label(scored)}"))
    numbers.update(wrapper_numbers(data, at))
    numbers.update(mode_numbers(data, at))
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
    """A test runner's end-to-end file: e2e_numbers, the Wilson interval of its success rate (wilson_numbers),
    a voted or ruled system's votes and rules (wrapper_numbers, counted over its transcripts' attempts) and
    its turns per thinking mode (mode_numbers) under e2e.<name>.*, and for a routed system its "routing"
    block (scripts/run_test.py, Session.e2e_routing) under e2e.<name>.routing.*, with the file's e2e_routing
    as routing.mode when it names one."""
    at = f"e2e.{label(name)}"
    numbers = e2e_numbers(data, name)
    numbers.update(wilson_numbers(data["summary"], at))
    numbers.update(wrapper_numbers(data, at))
    numbers.update(mode_numbers(data, at))
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


def dev_e2e_numbers(data: dict, name: str) -> dict[str, str]:
    """A development run's end-to-end report (scripts/run_e2e.py's report.json): e2e_numbers, and when its
    manifest counted the local steps longer than the training length, training_length,
    steps_over_training_length and local_calls (summed over the task records); when it voted, its failed vote
    candidates in all and per error type (module docstring). TypeError for a count that is not a whole
    number."""
    numbers = e2e_numbers(data, name)
    prefix = f"e2e.{label(name)}"
    manifest = data.get("manifest") or {}
    over = manifest.get("steps_over_training_length")
    if over is not None:  # null: run without the context check, so nothing was measured
        calls = [task["local_calls"] for task in data["tasks"]]
        for count in calls:
            count_text(count)
        numbers.update({f"{prefix}.training_length": count_text(manifest["training_length"]),
                        f"{prefix}.steps_over_training_length": count_text(over),
                        f"{prefix}.local_calls": str(sum(calls))})
    failed = data["summary"].get("candidate_errors")
    if failed is not None:
        by_type = {kind: count_text(count) for kind, count in failed.items()}
        numbers[f"{prefix}.candidate_errors"] = str(sum(failed.values()))
        for kind, count in by_type.items():
            _put(numbers, f"{prefix}.candidate_errors.{label(kind)}", count)
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


def benchmark_points(diff: float | Fraction) -> str:
    """A difference of benchmark accuracies in percentage points, with its sign and two decimals (an exact
    Fraction rounded as exactly)."""
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
    for key, value in acebench_differences(data, name).items():
        _put(numbers, key, value)
    return numbers


ACE_PAIR = ("ft", "b0")  # the fine-tuned system and its untrained base, compared stratum by stratum


def _stratum_accuracy(cell: dict, where: str) -> Fraction:
    """A stratum's accuracy exactly, correct / n; TypeError for counts that are not whole numbers, ValueError
    for n 0, correct outside 0..n and an accuracy that is not correct / n (the one the system's keys
    quote)."""
    correct, n = cell["correct"], cell["n"]
    count_text(correct)
    count_text(n)
    if n < 1 or not 0 <= correct <= n:
        raise ValueError(f"{where}: {correct} correct out of {n} answers")
    if abs(cell["accuracy"] - correct / n) > 1e-12:
        raise ValueError(f"{where}: accuracy {cell['accuracy']!r} is not {correct}/{n}")
    return Fraction(correct, n)


def _mean(values: list[Fraction]) -> Fraction:
    return sum(values, Fraction(0)) / len(values)


def acebench_differences(data: dict, name: str) -> dict[str, str]:
    """FT's accuracy minus B0's on each (language, category) stratum of an ACEBench file that holds both
    (ACE_PAIR; module docstring: ACEBench), under acebench.<name>.ft-vs-b0.*: exact differences of correct / n
    in points, and their means over each group's categories run in a language, over the language and over
    every stratum, the last H3's difference (scripts/run_test.py, Session.acebench_comparison: the mean over
    the strata of the strata's differences of success rates, eval.metrics.stratified_noninferiority_p's
    diff), with how many strata each system is the higher on; nothing for a file without both. ValueError
    for two systems that did not run the same languages, the same categories or the same n in a stratum, and
    for no stratum at all."""
    systems = data["systems"]
    if not all(system in systems for system in ACE_PAIR):
        return {}
    ft, b0 = (systems[system]["languages"] for system in ACE_PAIR)
    if set(ft) != set(b0):
        raise ValueError(f"ft and b0 ran different languages: {sorted(ft)} and {sorted(b0)}")
    prefix = f"acebench.{label(name)}.ft-vs-b0"
    numbers: dict[str, str] = {}
    every: list[Fraction] = []
    for language, block in ft.items():
        mine, theirs = block["categories"], b0[language]["categories"]
        if set(mine) != set(theirs):
            raise ValueError(f"{language}: ft and b0 ran different categories, ft alone "
                             f"{sorted(set(mine) - set(theirs))}, b0 alone {sorted(set(theirs) - set(mine))}")
        at = f"{prefix}.{label(language)}"
        diffs: dict[str, Fraction] = {}
        for category, cell in mine.items():
            where = f"{language}.{category}"
            if cell["n"] != theirs[category]["n"]:
                raise ValueError(f"{where}: ft has {cell['n']} answers and b0 {theirs[category]['n']}")
            diffs[category] = (_stratum_accuracy(cell, f"ft {where}")
                               - _stratum_accuracy(theirs[category], f"b0 {where}"))
            _put(numbers, f"{at}.{label(category)}", benchmark_points(diffs[category]))
        if not diffs:
            raise ValueError(f"{language}: no category run")
        for group, categories in data["groups"].items():
            run = [diffs[category] for category in categories if category in diffs]
            if run:
                _put(numbers, f"{at}.{label(group)}", benchmark_points(_mean(run)))
        _put(numbers, f"{at}.aggregate", benchmark_points(_mean(list(diffs.values()))))
        every.extend(diffs.values())
    if not every:
        raise ValueError("ft and b0 ran no language")
    _put(numbers, f"{prefix}.aggregate", benchmark_points(_mean(every)))
    counts = {"strata": len(every), "ft_higher": sum(diff > 0 for diff in every),
              "b0_higher": sum(diff < 0 for diff in every), "equal": sum(diff == 0 for diff in every)}
    for key, count in counts.items():
        _put(numbers, f"{prefix}.{key}", str(count))
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


# A lengths file's blocks per system (scripts/thinking_lengths.py; module docstring): the turns with a
# thinking record of its single-step records and of its end-to-end attempts, and its local calls measured end
# to end.
LENGTH_PARTS = ("single_step", "e2e")
THINKING_TOKENS = ("reasoning_p50", "reasoning_p95", "reasoning_max", "answer_p50", "answer_p95")
E2E_LENGTH = "e2e_length"


def token_text(value: object) -> str:
    """A number of tokens, or a percentile of them, as README quotes it: a whole number; TypeError for
    anything but a number (true and false included)."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{value!r} is not a number of tokens")
    return whole(value)


def _out_of(count: object, total: object, what: str) -> None:
    """ValueError when count is larger than total, the turns or calls it is out of; TypeError when either is
    not a whole number (count_text)."""
    count_text(count)
    count_text(total)
    if count > total:
        raise ValueError(f"{what}: {count} is more than the {total} it is out of")


def _thinking(prefix: str, block: dict) -> dict[str, str]:
    """A part's thinking lengths: its turns, the token percentiles and maxima that apply, the reasonings
    closed by force and their rate over the turns."""
    n, forced = block["n"], block["forced"]
    _out_of(forced, n, f"{prefix}.forced")
    numbers = {f"{prefix}.n": str(n), f"{prefix}.forced": str(forced)}
    for key in THINKING_TOKENS:
        if block[key] is not None:
            numbers[f"{prefix}.{key}"] = token_text(block[key])
    if n:
        numbers[f"{prefix}.forced_rate"] = percent(forced / n)
    return numbers


def lengths_numbers(data: dict, name: str) -> dict[str, str]:
    """A test session's thinking lengths and steps over the training length (scripts/thinking_lengths.py)
    under lengths.<name>.<system>.* (module docstring); ValueError for a system with none of the blocks and
    for a count larger than the turns or calls it is out of."""
    numbers: dict[str, str] = {}
    for system, row in data["systems"].items():
        prefix = f"lengths.{label(name)}.{label(system)}"
        blocks = (*LENGTH_PARTS, E2E_LENGTH)
        if all(row.get(block) is None for block in blocks):
            raise ValueError(f"{system} has none of the blocks {', '.join(blocks)}")
        for part in LENGTH_PARTS:
            if row.get(part) is not None:
                for key, value in _thinking(f"{prefix}.{part}", row[part]).items():
                    _put(numbers, key, value)
        length = row.get(E2E_LENGTH)
        if length is not None:
            at = f"{prefix}.{E2E_LENGTH}"
            _out_of(length["over"], length["calls"], f"{at}.over")
            for key in ("training_length", "calls", "over"):
                _put(numbers, f"{at}.{key}", count_text(length[key]))
            if length["max"] is not None:
                _put(numbers, f"{at}.max", token_text(length["max"]))
    return numbers


def _count(value: object, what: str) -> str:
    """A count as README quotes it (count_text); TypeError for anything but a whole number, ValueError for one
    below 0."""
    text = count_text(value)
    if value < 0:
        raise ValueError(f"{what}: {value} is not a count")
    return text


# An ACEBench run's generated lengths (scripts/acebench_lengths.py, M11; module docstring: ACE lengths): the
# kind the script writes (its KIND), its block over every language (its ALL), a block's figures in tokens and
# its counts out of the block's requests.
ACE_LENGTHS_KIND = "acebench-lengths"
ACE_ALL = "all"
ACE_TOKENS = ("tokens_p50", "tokens_p95", "tokens_max")
ACE_COUNTS = ("over_budget", "at_max_tokens")


def _ace_lengths_block(prefix: str, block: dict) -> dict[str, str]:
    """A block of requests under prefix: how many, the percentiles and the maximum of their tokens that apply,
    those above the reasoning budget and those at the limit, and those two over the requests (none over no
    request); ValueError for a count larger than the requests or below 0."""
    n = block["n"]
    numbers = {f"{prefix}.n": _count(n, f"{prefix}.n")}
    for key in ACE_TOKENS:
        if block[key] is not None:
            numbers[f"{prefix}.{key}"] = token_text(block[key])
    for key in ACE_COUNTS:
        numbers[f"{prefix}.{key}"] = _count(block[key], f"{prefix}.{key}")
        _out_of(block[key], n, f"{prefix}.{key}")
        if n:
            numbers[f"{prefix}.{key}_rate"] = percent(block[key] / n)
    return numbers


def acebench_lengths_numbers(data: dict, name: str) -> dict[str, str]:
    """The tokens an ACEBench run's servers generated per request (scripts/acebench_lengths.py) under
    lengths.<name>.* (<name> acebench-<m>; module docstring: ACE lengths). ValueError for another kind, no
    language or no system, a count below 0 or larger than the requests it is out of, an all block that is not
    the languages' blocks together (n and the two counts summed, tokens_max the longest) and requests of a
    log that are not its probes and its answers; TypeError for languages that are not a list of names and for
    a count or a number of tokens that is none."""
    if data["kind"] != ACE_LENGTHS_KIND:
        raise ValueError(f"kind {data['kind']!r} is not {ACE_LENGTHS_KIND!r}")
    languages, systems = data["languages"], data["systems"]
    if not isinstance(languages, list) or not all(isinstance(language, str) for language in languages):
        raise TypeError(f"languages {languages!r} is not a list of names")
    if not languages:
        raise ValueError("no language: nothing was measured")
    if not systems:
        raise ValueError("no system: nothing was measured")
    prefix = f"lengths.{label(name)}"
    numbers = {f"{prefix}.budget.{key}": _count(data["budget"][key], f"budget.{key}")
               for key in ("reasoning", "max_tokens")}
    for system, row in systems.items():
        at = f"{prefix}.{label(system)}"
        for key in ("probes", "requests"):
            _put(numbers, f"{at}.{key}", _count(row[key], f"{at}.{key}"))
        for part in (*languages, ACE_ALL):  # a language named all gives that block's keys twice: refused
            for key, value in _ace_lengths_block(f"{at}.{label(part)}", row[part]).items():
                _put(numbers, key, value)
        together, alone = row[ACE_ALL], [row[language] for language in languages]
        for key in ("n", *ACE_COUNTS):
            summed = sum(block[key] for block in alone)
            if together[key] != summed:
                raise ValueError(f"{at}.{ACE_ALL}.{key} is {together[key]}, not its languages' {summed}")
        maxima = [block["tokens_max"] for block in alone if block["tokens_max"] is not None]
        longest = max(maxima, default=None)
        if together["tokens_max"] != longest:
            raise ValueError(f"{at}.{ACE_ALL}.tokens_max is {together['tokens_max']!r}, not its languages' "
                             f"longest, {longest!r}")
        if row["requests"] != row["probes"] + together["n"]:
            raise ValueError(f"{at}.requests: {row['requests']} is not its {row['probes']} probes and "
                             f"{together['n']} answers")
    return numbers


# M7's base-model bake-off (scripts/bakeoff.py aggregate, eval.bakeoff.aggregate; module docstring: bake-off):
# the statuses of its entries (the reference C0; a candidate's gate, scored, incomplete or ineligible; one
# with no gate yet; one a cut line skipped), the two that give their status alone, the decision's statuses,
# the modes a winner answers in (family.core.Mode) and the thinking kinds of a family (eval.bakeoff.THINKING).
REFERENCE, SKIPPED = "reference", "skipped"
BAKEOFF_STATUSES = (REFERENCE, "scored", "incomplete", "ineligible", "pending", SKIPPED)
BAKEOFF_STATUS_ONLY = ("pending", SKIPPED)
BAKEOFF_DECISIONS = ("decided", "extend", "undecided", "none-eligible", "provisional")
BAKEOFF_MODES = ("think", "no_think")
BAKEOFF_THINKING = ("none", "only", "hybrid")
VARIANT_COUNTS = ("successes", "n", "handwritten")  # a variant's scores the report takes


def _number(value: object, what: str) -> float:
    """value, a finite number; TypeError for anything else (true and false included), ValueError for NaN and
    the infinities."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{what} {value!r} is not a number")
    if not math.isfinite(value):
        raise ValueError(f"{what} {value!r} is not a finite number")
    return value


def score_text(value: object) -> str:
    """A bake-off score, a number of successes or their mean over a family's three seeds (P, handwritten), as
    README quotes it: a whole number as it is ("107"), any other with two decimals ("115.33"); TypeError for
    anything but a number, ValueError for one that is not finite."""
    value = _number(value, "a score")
    if isinstance(value, int):
        return str(value)
    return f"{value:.0f}" if value.is_integer() else f"{value:.2f}"


def hours_text(value: object) -> str:
    """Hours, an FT3 estimate or the rule's limit on it: one decimal ("28.1")."""
    return f"{_number(value, 'hours'):.1f}"


def _one_of(value: object, known: tuple[str, ...], what: str) -> str:
    """value, one of the names known; ValueError for any other."""
    if not isinstance(value, str) or value not in known:
        raise ValueError(f"{what} {value!r} is not one of {list(known)}")
    return value


def _bakeoff_extension(prefix: str, extension: dict) -> dict[str, str]:
    """A candidate's extension to the extension step (eval.bakeoff.candidate_gate): the step, P and the
    handwritten mean there, a hybrid family's nothink greedy successes when not null, and each variant's
    successes, items and handwritten successes; ValueError for a variant with more successes than items or
    more handwritten successes than successes, and for no variant."""
    numbers = {f"{prefix}.stop_step": count_text(extension["stop_step"]),
               f"{prefix}.P": score_text(extension["P"]),
               f"{prefix}.handwritten": score_text(extension["handwritten"])}
    if extension["no_think_greedy"] is not None:
        numbers[f"{prefix}.no_think_greedy"] = count_text(extension["no_think_greedy"])
    variants = extension["variants"]
    if not isinstance(variants, dict):
        raise TypeError(f"variants {variants!r} is not an object of scores per variant")
    if not variants:
        raise ValueError("an extension without a variant's scores")
    for variant, scores in variants.items():
        at = f"{prefix}.{label(variant)}"
        _out_of(scores["successes"], scores["n"], f"{at}.successes")
        _out_of(scores["handwritten"], scores["successes"], f"{at}.handwritten")
        for key in VARIANT_COUNTS:
            _put(numbers, f"{at}.{key}", count_text(scores[key]))
    return numbers


def _bakeoff_candidate(prefix: str, entry: dict, rank: int | None) -> dict[str, str]:
    """A candidate's numbers: its status, and unless it has no gate or was skipped, the fields of its entry
    that are not null (P, handwritten, eligible, mode, thinking, no_think_greedy, the FT3 estimate as
    ft3_hours), its rank and its extension; ValueError for a status, mode or thinking kind the aggregate does
    not write, and for a candidate with no gate or skipped that holds a P or is ranked."""
    status = _one_of(entry["status"], BAKEOFF_STATUSES, "status")
    numbers = {f"{prefix}.status": status}
    if status in BAKEOFF_STATUS_ONLY:
        if entry.get("P") is not None or rank is not None:
            raise ValueError(f"a {status} candidate holds a P or is ranked")
        return numbers
    for key in ("P", "handwritten"):
        if entry[key] is not None:
            numbers[f"{prefix}.{key}"] = score_text(entry[key])
    if entry["eligible"] is not None:
        numbers[f"{prefix}.eligible"] = yes_no(entry["eligible"])
    if entry["mode"] is not None:
        numbers[f"{prefix}.mode"] = _one_of(entry["mode"], BAKEOFF_MODES, "mode")
    if entry["thinking"] is not None:
        numbers[f"{prefix}.thinking"] = _one_of(entry["thinking"], BAKEOFF_THINKING, "thinking")
    if entry["no_think_greedy"] is not None:
        numbers[f"{prefix}.no_think_greedy"] = count_text(entry["no_think_greedy"])
    if entry["ft3_hours_estimate"] is not None:
        numbers[f"{prefix}.ft3_hours"] = hours_text(entry["ft3_hours_estimate"])
    if rank is not None:
        numbers[f"{prefix}.rank"] = str(rank)
    if entry["extension"] is not None:
        numbers.update(_bakeoff_extension(f"{prefix}.extension", entry["extension"]))
    return numbers


def bakeoff_numbers(data: dict, name: str) -> dict[str, str]:
    """M7's base-model bake-off (scripts/bakeoff.py aggregate) under bakeoff.<name>.* (module docstring:
    bake-off). ValueError for a status, mode or thinking kind the aggregate does not write, a candidate filed
    under another key than its own, a ranking that names a candidate twice, one the file does not have or one
    with no gate or skipped, and a winner or runner-up the ranking does not hold; TypeError for a number,
    a name or a list that is none."""
    prefix = f"bakeoff.{label(name)}"
    candidates, decision = data["candidates"], data["decision"]
    if not isinstance(candidates, dict) or not isinstance(decision, dict):
        raise TypeError("candidates and decision must be objects")
    ranking = decision["ranking"]
    if not isinstance(ranking, list):
        raise TypeError(f"the ranking {ranking!r} is not a list")
    if any(key not in candidates for key in ranking) or len(set(ranking)) != len(ranking):
        raise ValueError(f"the ranking {ranking} does not name distinct candidates of the file")
    numbers = {f"{prefix}.complete": yes_no(data["complete"])}
    for key, entry in candidates.items():
        if entry["candidate"] != key:
            raise ValueError(f"candidates.{key} is the entry of {entry['candidate']!r}")
        rank = ranking.index(key) + 1 if key in ranking else None
        for found, value in _bakeoff_candidate(f"{prefix}.{label(key)}", entry, rank).items():
            _put(numbers, found, value)
    statuses = [entry["status"] for entry in candidates.values()]
    _put(numbers, f"{prefix}.candidates", str(sum(status != REFERENCE for status in statuses)))
    _put(numbers, f"{prefix}.skipped", str(statuses.count(SKIPPED)))
    at = f"{prefix}.decision"
    _put(numbers, f"{at}.status", _one_of(decision["status"], BAKEOFF_DECISIONS, "decision.status"))
    for key in ("winner", "runner_up"):
        if decision[key] is not None:
            if decision[key] not in ranking:
                raise ValueError(f"decision.{key} {decision[key]!r} is not a ranked candidate")
            _put(numbers, f"{at}.{key}", decision[key])
    if decision["basis"] is not None:
        if not isinstance(decision["basis"], str):
            raise TypeError(f"decision.basis {decision['basis']!r} is not a name")
        _put(numbers, f"{at}.basis", decision["basis"])
    if decision["mode"] is not None:
        _put(numbers, f"{at}.mode", _one_of(decision["mode"], BAKEOFF_MODES, "decision.mode"))
    if decision["margin"] is not None:
        _put(numbers, f"{at}.margin", f"{_number(decision['margin'], 'decision.margin'):.2f}")
    for field, value in data["rule"].items():
        if field == "smoke_steps":  # the names of the smoke steps, not a number
            if not isinstance(value, list) or not all(isinstance(step, str) for step in value):
                raise TypeError(f"rule.smoke_steps {value!r} is not a list of step names")
            continue
        text = hours_text(value) if field == "ft3_hours" else score_text(value)
        _put(numbers, f"{prefix}.rule.{label(field)}", text)
    return numbers


# What M7 ran beside its bake-off (scripts/bakeoff.py descriptive, eval.bakeoff.descriptive, M11; module
# docstring: description): the kind it writes (eval.bakeoff.DESCRIPTIVE_KIND), the counts of a run, the name
# of the untrained base's run beside a gate's variants, the counts of the fidelity record it keeps
# (eval.bakeoff.FIDELITY_COUNTS) and the rates of the positions the report gives, (rate, count).
BAKEOFF_DESCRIPTIVE_KIND = "bakeoff-descriptive"
RUN_COUNTS = ("successes", "n", "handwritten", "drafted")
BASE_RUN = "b0"
FIDELITY_COUNTS = ("items", "positions", "agreeing", "base_agreeing", "changed_positions")
FIDELITY_RATES = (("agreement", "agreeing"), ("base_agreement", "base_agreeing"),
                  ("changed", "changed_positions"))


def _name(value: object, what: str) -> str:
    """value, a name; TypeError for anything else, the empty string included."""
    if not isinstance(value, str) or not value:
        raise TypeError(f"{what} {value!r} is not a name")
    return value


def _ablation_run(prefix: str, run: dict) -> dict[str, str]:
    """A run of an ablation (eval.bakeoff.descriptive: RUN) under prefix: its successes over its items, its
    successes on the handwritten and on the drafted items and, when its gate counted them per item type, each
    type's successes and items; ValueError for a count below 0, more successes than items, handwritten and
    drafted successes that do not make its successes, a type with more successes than items and types that
    do not add up to the run; TypeError for per-type counts that are not an object."""
    numbers = {f"{prefix}.{key}": _count(run[key], f"{prefix}.{key}") for key in RUN_COUNTS}
    successes, n = run["successes"], run["n"]
    _out_of(successes, n, f"{prefix}.successes")
    if run["handwritten"] + run["drafted"] != successes:
        raise ValueError(f"{prefix}: {run['handwritten']} handwritten and {run['drafted']} drafted successes "
                         f"do not make its {successes}")
    by_type = run["by_type"]
    if by_type is None:
        return numbers
    if not isinstance(by_type, dict):
        raise TypeError(f"{prefix}.by_type {by_type!r} is not an object of counts per item type")
    for kind, cell in by_type.items():
        at = f"{prefix}.{label(kind)}"
        for key in ("successes", "n"):
            _put(numbers, f"{at}.{key}", _count(cell[key], f"{at}.{key}"))
        _out_of(cell["successes"], cell["n"], f"{at}.successes")
    total = (sum(cell["successes"] for cell in by_type.values()), sum(cell["n"] for cell in by_type.values()))
    if total != (successes, n):
        raise ValueError(f"{prefix}: its item types add up to {total[0]}/{total[1]}, not {successes}/{n}")
    return numbers


def _ablation(prefix: str, ablation: dict) -> dict[str, str]:
    """An ablation (eval.bakeoff.descriptive: ABLATION) under prefix: the key of the candidate whose model it
    trained, that model's id and revision, the mode it was trained and scored in, the step it was trained to,
    its scores as a bake-off's (score_text: P and its handwritten and drafted means) and each run's counts
    (_ablation_run): its gate's variants and the untrained base's run when it has one, under BASE_RUN.
    ValueError for a mode the bake-off does not write, a base run in another mode than the ablation's or with
    a variant's name, no variant, runs (the base's included) over different numbers of items (P is a mean of
    runs over the same development items, and the base's run is set beside them), primary variants that are
    none, named twice or not the ablation's, and a P or mean that is not the mean over them; TypeError for a
    name that is none and variants that are not an object."""
    numbers = {f"{prefix}.{key}": _name(ablation[key], f"{prefix}.{key}")
               for key in ("candidate", "model", "revision")}
    mode = _one_of(ablation["mode"], BAKEOFF_MODES, f"{prefix}.mode")
    numbers[f"{prefix}.mode"] = mode
    numbers[f"{prefix}.stop_step"] = _count(ablation["stop_step"], f"{prefix}.stop_step")
    variants, base = ablation["variants"], ablation[BASE_RUN]
    if not isinstance(variants, dict):
        raise TypeError(f"{prefix}.variants {variants!r} is not an object of runs per variant")
    if not variants:
        raise ValueError(f"{prefix}: no variant")
    runs = list(variants.items())
    if base is not None:
        if base["mode"] != mode:
            raise ValueError(f"{prefix}.{BASE_RUN} ran in the mode {base['mode']!r}, not the ablation's, "
                             f"{mode}")
        runs.append((BASE_RUN, base))  # a variant of that name gives the same keys: refused by _put
    for run, scores in runs:
        for key, value in _ablation_run(f"{prefix}.{label(run)}", scores).items():
            _put(numbers, key, value)
    sizes = {run: scores["n"] for run, scores in runs}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"{prefix}: its runs are not over one number of items: {sizes}")
    primary = ablation["primary"]
    if (not isinstance(primary, list) or not primary
            or not all(isinstance(variant, str) and variant in variants for variant in primary)
            or len(set(primary)) != len(primary)):
        raise ValueError(f"{prefix}: the primary variants {primary!r} are not distinct variants it holds")
    for key, field in (("P", "successes"), ("handwritten", "handwritten"), ("drafted", "drafted")):
        mean = sum(variants[variant][field] for variant in primary) / len(primary)
        if abs(_number(ablation[key], f"{prefix}.{key}") - mean) > 1e-9:
            raise ValueError(f"{prefix}.{key} {ablation[key]!r} is not the mean {field} of "
                             f"{', '.join(primary)}, {mean!r}")
        numbers[f"{prefix}.{key}"] = score_text(ablation[key])
    return numbers


def _fidelity(prefix: str, fidelity: dict) -> dict[str, str]:
    """The winner's trained adapter's fidelity (eval.bakeoff.descriptive: FIDELITY) under <prefix>.<winner>:
    the step trained and the record's counts (FIDELITY_COUNTS), and the agreeing, base-agreeing and changed
    positions as rates of the positions (FIDELITY_RATES; none over no position); ValueError for a count below
    0 or larger than the positions, TypeError for a winner that is not a name."""
    at = f"{prefix}.{label(_name(fidelity['candidate'], 'fidelity.candidate'))}"
    numbers = {f"{at}.trained_step": _count(fidelity["trained_step"], f"{at}.trained_step")}
    numbers.update({f"{at}.{key}": _count(fidelity[key], f"{at}.{key}") for key in FIDELITY_COUNTS})
    positions = fidelity["positions"]
    for rate, count in FIDELITY_RATES:
        _out_of(fidelity[count], positions, f"{at}.{count}")
        if positions:
            numbers[f"{at}.{rate}"] = percent(fidelity[count] / positions)
    return numbers


def bakeoff_descriptive_numbers(data: dict, name: str) -> dict[str, str]:
    """What M7 ran beside its bake-off (scripts/bakeoff.py descriptive) under bakeoff.<name>.ablation.* and
    bakeoff.<name>.fidelity.* (module docstring: description). ValueError for another kind, another milestone
    than the file name's, no ablation, and what _ablation and _fidelity refuse; TypeError for ablations that
    are not an object."""
    if data["kind"] != BAKEOFF_DESCRIPTIVE_KIND:
        raise ValueError(f"kind {data['kind']!r} is not {BAKEOFF_DESCRIPTIVE_KIND!r}")
    if data["milestone"] != name:
        raise ValueError(f"the milestone {data['milestone']!r} is not the file name's, {name}")
    ablations = data["ablations"]
    if not isinstance(ablations, dict):
        raise TypeError(f"ablations {ablations!r} is not an object of ablations by name")
    if not ablations:
        raise ValueError("no ablation")
    prefix = f"bakeoff.{label(name)}"
    numbers: dict[str, str] = {}
    for ablation, entry in ablations.items():
        for key, value in _ablation(f"{prefix}.ablation.{label(ablation)}", entry).items():
            _put(numbers, key, value)
    for key, value in _fidelity(f"{prefix}.fidelity", data["fidelity"]).items():
        _put(numbers, key, value)
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
    Kind("dev-e2e", "an end-to-end summary", re.compile(r"dev-e2e-(.+)\.json"), dev_e2e_numbers,
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
    Kind("lengths", "a lengths summary", re.compile(r"lengths-(m\d+)\.json"), lengths_numbers,
         lambda m: m.group(1)),
    # M11's three, last. No pattern above names a lengths-acebench- or a bakeoff- file (lengths-<m>.json's <m>
    # is m and digits, and acebench-<name>.json's name begins with acebench-), and the three name each other's
    # files neither (bakeoff-<m>.json's <m> is m and digits too), so no file's kind changes
    Kind("acebench-lengths", "an ACEBench lengths summary", re.compile(r"lengths-acebench-(m\d+)\.json"),
         acebench_lengths_numbers, lambda m: f"acebench-{m.group(1)}"),
    Kind("bakeoff", "a bake-off aggregate", re.compile(r"bakeoff-(m\d+)\.json"), bakeoff_numbers,
         lambda m: m.group(1)),
    Kind("bakeoff-descriptive", "a bake-off description", re.compile(r"bakeoff-descriptive-(m\d+)\.json"),
         bakeoff_descriptive_numbers, lambda m: m.group(1)),
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

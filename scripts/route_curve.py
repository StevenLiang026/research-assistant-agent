"""Choose and freeze FT+R's routing on the development set (docs/plan.md, 置信度路由 and 路由曲线;
docs/m6-计划.md §3.3, the policies and the cost objective; docs/m7-计划.md WP-K, the target objective over
several development sets), offline: it reads saved runs and calls no model.

    .venv/Scripts/python.exe scripts/route_curve.py --ft runs/eval/dev-ft-signals.jsonl \\
        --ft-scores runs/eval/scores/dev-ft-signals.json [--b2 runs/eval/dev-b2-m3.jsonl] \\
        --b2-scores runs/eval/scores/dev-b2-m3.json [--dev data/eval/dev.jsonl] \\
        [--objective cost|share|target] [--target T] [--cloud b2] [--out results/routing-m4.json] [--force]
    .venv/Scripts/python.exe scripts/route_curve.py \\
        --ft runs/m7/dev-ft3-signals.jsonl --ft-scores runs/m7/scores/dev-ft3-signals.json \\
        --b2 runs/eval/dev-b2-m3.jsonl --b2-scores runs/eval/scores/dev-b2-m3.json \\
        --dev data/eval/dev.jsonl \\
        --ft runs/m7/dev-v2-ft3-signals.jsonl --ft-scores runs/m7/scores/dev-v2-ft3-signals.json \\
        --b2 runs/m7/dev-v2-b2.jsonl --b2-scores runs/m7/scores/dev-v2-b2.json \\
        --dev data/eval/v2/dev-v2.jsonl \\
        --objective target --target 0.92 --cloud b2 --out results/routing-m7.json

Inputs, per development set. FT's development records made by scripts/run_eval.py --system ft --signals
(every ok record carries "signals": min_decision_prob, mean_logprob, hard_reason, and "output", the turn with
its tool calls; a turn of a thinking generation also its "thinking", whose "forced" says the reasoning hit its
budget and was closed by force) and B2's development records, each with its per-item scores (scripts/score.py
over the same items), and beside each records file its manifest (<stem>.manifest.json) and timing sidecar
(<stem>.timing.json). Both manifests must name the system (ft, b2) and the SHA-256 of the development set as
it is now (--dev); FT's must list exactly one adapter, whose SHA-256 the configuration freezes, and B2's model
is the cloud model (deepseek-flash, eval.systems.CLOUD_MODELS["b2"]: the cloud system is b2, --cloud, which
names no other). FT's manifest holds FT's generation when its run was made on the family path ("generation",
an object: the mode, the decoding, the reasoning budget, the base seed..., as scripts/run_eval.py records
llm.generation.GenerationSettings.record()), which the configuration freezes too; a run of M3 to M6 holds
none. The four id lists must be the same items. Test material is refused (eval.lock.refuse_test_material over
the records' items).

Several development sets (M7: dev and dev-v2 jointly, docs/m7-计划.md §5 开发集口径). The five options --ft,
--ft-scores, --b2, --b2-scores and --dev are given once per set, the k-th of each naming the k-th set's
inputs (with one set, --b2 and --dev may be left out for their defaults; with several, each is given once per
set, or it is a usage error). Every set is checked as above on its own; then FT's runs must have used one
adapter and one generation (every set's FT manifest the same, or none of them one: a mix is refused), and no
item id may be in two sets. The rows are the sets' rows in their order, and everything below is computed over
all of them as over one set.

Per item: FT's and B2's success from the scores; FT's signals and hard reason from its record (an FT record
that is not ok has none: the runner recorded an error or an overflow, and a routed system would record the
same, so such an item is never routed by a threshold); FT's tool calls from its output, the assistant turn
the runner records for every ok item (agent.route.api_calls, what the parameter triggers and the text
exemption read; an ok record without it is malformed, not a text answer); whether its reasoning was closed by
force (the record's thinking.forced, what the budget trigger reads; false without a thinking record); B2's
list-price cost of the item (llm.budget.cost_rmb with the repository's price of the cloud model over the
record's usage, whether it was paid or came from the cache); whether the item is handwritten (its source,
eval.report.in_subset).

1. AUROC of each signal for predicting "FT's step is correct" (success), over the items with an ok FT record
   and a value of that signal: the share of (correct, incorrect) pairs where the correct item's value is
   higher, ties counting one half. The higher AUROC chooses the signal; a tie goes to min_decision_prob (the
   preregistered order, agent.route.SIGNALS).
2. Policies (POLICIES, docs/m6-计划.md §3.3 C1-C3), each a set of rules agent.route.route_reason applies
   before the threshold (RouteRules): P0, the threshold alone (no rules: M4's routing); P1, the text
   exemption (a turn without any tool call stays local); P5, the text exemption plus the parameter triggers
   (search_notes.top_k other than 3, list_notes.limit other than 10: the non-default values FT writes out of
   habit, PARAM_TRIGGERS). The hard triggers fire under every policy. Under the target objective (M7) each
   policy also carries the budget trigger (M7_POLICIES, docs/m7-计划.md §2 路由信号: a step whose reasoning
   was closed by force is routed like a hard trigger), so P0's rules are the budget trigger alone; under share
   and cost the policies are M6's, without it, and a forced step is routed by the threshold as any other.
3. Scan, per policy. Thresholds: every distinct value of the chosen signal, ascending, the same for every
   policy. At threshold t an item is routed when its FT record is ok and route_reason under the policy gives
   a reason: a hard trigger, the budget trigger, a parameter trigger, or (unless the turn is exempt) the
   signal missing or below t (the rule RoutedModel applies online). FT+R's success on an item is B2's when
   routed, else FT's; the API share is routed items over all items; the cost per 100 items is 100 x the mean
   over items of B2's list-price cost when routed, else 0. Two more points, "route_nothing" (FT, share 0) and
   "route_everything" (B2 on every item, share 1), are every curve's endpoints; they are not thresholds.
   Every point also carries the random-routing success at its share, exactly (1 - s) x FT's success +
   s x B2's success.
4. The objective (--objective, RULES). cost, the M6 preregistered rule: of the (policy, threshold) points
   whose success is at least B2's success minus 0.02 (compared to 12 decimals), the one with the lowest cost
   per 100 items; ties go to fewer routed items, then the lower threshold, then the policy listed first.
   share, the M4 rule: over P0 alone, the threshold with the smallest API share whose success is at least
   B2's minus 0.02; ties go to the higher success, then the lower threshold. target (M7, --target T in
   (0, 1]): of the (policy, threshold) points whose success is at least T (compared to 12 decimals), the one
   with the lowest cost per 100 items, whether or not it is within the cost cap; ties go to the higher
   success, then fewer routed items, then the lower threshold, then the policy listed first. When no point
   reaches T, the point with the highest success whose cost per 100 items is within the cost cap (at most
   CAP_SHARE, half, of route_everything's: the owner's cap, half of giving every item to the cloud,
   docs/m7-计划.md), ties going to the lower cost, then fewer routed items, then the lower threshold, then
   the policy listed first, and the configuration says the target was not reached. When no point meets the
   objective (no point within the margin; under target, none reaching T and none within the cap), the
   configuration says so: "rule_met" false and "threshold" null, which eval.systems refuses to run. Each
   policy's own best point under the objective is recorded beside its curve (under target with whether it
   reaches T).
5. The oracle curve, at every number k = 0..n of routed items: the k routed first are the items FT fails and
   B2 passes, then those where both agree, then those FT passes and B2 fails; it is never below a scanned
   point with the same number routed.
6. The same scan over the handwritten items alone under the chosen policy, with the chosen threshold's point
   on them (descriptive).
7. Latency p50 and p95 (eval.resources.latency): local per call over FT's timing sidecars (every call that
   came back), cloud per paid call over B2's.

Output (--out, default results/routing-m4.json): JSON with two-space indent, UTF-8, LF line ends: the rule and
the objective (under target the target, whether the chosen point reaches it, the cost cap per 100 items and
whether the chosen point is within it, in place of the margin), whether the rule was met, the chosen signal,
threshold, policy ("chosen_policy") and rules ("rules", agent.route.RouteRules' JSON form; null for P0 under
share and cost, which eval.systems reads as before), "adapter_sha256", then, when FT's manifests record a
generation, "ft_generation" (theirs, as they record it: what scripts/run_test.py holds against the generation
its routed system runs with, and scripts/route_validate.py against the validated runs'), "cloud_model" (with
the development sets' SHA-256, the fields eval.systems' ftr freezes), the development set: for one set
"dev_sha256", its path ("dev_set") and every input with its path relative to the repository and its SHA-256
("inputs"), as before; for several, "dev_sets", one entry per set in order with its path, SHA-256, number of
items and inputs; then n, FT's and B2's success, both AUROCs, the chosen point (with its policy), "policies"
(each policy's label, rules, curve and its own best point under the objective, "chosen", and under target
"target_reached"), the curves ("routing", the chosen policy's scan, so that the chosen point lies on it as the
figures and the report expect, and P0's when no point was chosen; "oracle"), the handwritten scan (under the
chosen policy too, named in "handwritten.policy") and the latency. Under the share objective the chosen policy
is P0 and the file reads as M4's; the share and cost objectives over one set write exactly the bytes they
wrote before M7, and every objective, over FT's runs without a generation, the bytes it wrote before
"ft_generation" (results/routing-m4.json and routing-m6.json among them). An existing file is kept unless
--force is given. scripts/route_validate.py evaluates the frozen policy on other sets.

Exit 0 when written (the rule met or not); 1 on malformed input: a file that is not JSON or JSONL, records or
scores of the wrong shape, an ok FT record without signals (rerun FT with --signals) or without its output,
the assistant turn, a generation in FT's manifest that is not an object; 2 when the inputs do not belong
together (other item ids, a manifest of another system or development set, not exactly one adapter in FT's
manifest, another cloud model, sets run with other adapters or generations, one without a generation among
them, or holding the same item) or on a usage error (--target without the target objective or that objective
without it, a target outside (0, 1], another cloud system, not every input given once per set); 3 when --out
exists and --force was not given; 4 when the inputs are test material. Nothing is written unless the exit is
0.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from research_agent.agent.route import SIGNALS, RouteConfig, RouteRules, api_calls, route_reason
from research_agent.canonical import canonical_json
from research_agent.eval.lock import (
    JsonlError,
    RefusedTestSet,
    read_jsonl,
    refuse_test_material,
    refuse_test_sets,
    sha256_of,
)
from research_agent.eval.report import in_subset
from research_agent.eval.resources import latency
from research_agent.eval.systems import CLOUD_MODELS, DEV_SET, ROUTING_CONFIG
from research_agent.llm.budget import cost_rmb, load_prices

REPO = Path(__file__).resolve().parents[1]
MARGIN = 0.02
SHARE, COST, TARGET = "share", "cost", "target"
OBJECTIVES = (SHARE, COST, TARGET)
CAP_SHARE = 0.5  # the cost cap: half of what routing every item to the cloud costs (docs/m7-计划.md)
CLOUD_SYSTEM = "b2"  # the cloud system a routed step goes to (--cloud): deepseek-flash
B2_RECORDS = Path("runs/eval/dev-b2-m3.jsonl")  # --b2's default with a single set
RULES = {
    SHARE: ("Of the thresholds of the chosen signal (an item is routed when a hard trigger fires or its "
            "signal is below the threshold), the one with the smallest API share (routed items over all "
            "items) whose development success is at least B2's minus 0.02; ties go to the higher success, "
            "then the lower threshold (docs/plan.md, 置信度路由)."),
    COST: ("Of the policies (P0 the threshold alone, P1 the text exemption, P5 the text exemption with "
           "the non-default parameter triggers) and the thresholds of the chosen signal, the (policy, "
           "threshold) with the lowest list-price cost per 100 items whose development success is at "
           "least B2's minus 0.02; ties go to fewer routed items, then the lower threshold, then the "
           "policy listed first (docs/m6-计划.md §3.3 C3)."),
    TARGET: ("Of the policies (P0 the threshold alone, P1 the text exemption, P5 the text exemption with the "
             "non-default parameter triggers, each with the budget trigger: a step whose reasoning was "
             "closed by force is routed) and the thresholds of the chosen signal, the (policy, threshold) "
             "with the lowest list-price cost per 100 items whose development success is at least the "
             "target; ties go to the higher success, then fewer routed items, then the lower threshold, then "
             "the policy listed first. When no point reaches the target, the one with the highest success "
             "whose cost per 100 items is within the cap, half of routing every item to B2; ties go to the "
             "lower cost, then fewer routed items, then the lower threshold, then the policy listed first, "
             "and the target is recorded as not reached (docs/m7-计划.md WP-K)."),
}
RULE = RULES[SHARE]  # the M4 rule, as results/routing-m4.json records it
NOTHING, THRESHOLD, EVERYTHING = "route_nothing", "threshold", "route_everything"
# The non-default parameter values (docs/m6-计划.md §3.3 C2): the allowed value is the tool's default
# (tools.schemas.DEFAULTS), written out so the preregistered rule reads the same whatever the schema says.
PARAM_TRIGGERS = ({"tool": "search_notes", "param": "top_k", "allowed": [3]},
                  {"tool": "list_notes", "param": "limit", "allowed": [10]})
INPUT_OPTIONS = ("--ft", "--ft-scores", "--b2", "--b2-scores", "--dev")  # given once per development set
GENERATION = "generation"  # FT's generation in its manifest (scripts/run_eval.py)
FT_GENERATION = "ft_generation"  # and in the configuration (module docstring: output)


@dataclass(frozen=True)
class Policy:
    """A candidate policy: its name, what it does and the rules route_reason applies (None: none)."""

    name: str
    label: str
    rules: RouteRules | None

    def rules_json(self) -> dict | None:
        return None if self.rules is None else self.rules.to_json()


POLICIES = {p.name: p for p in (
    Policy("P0", "threshold only", None),
    Policy("P1", "text exemption", RouteRules.from_json({"text_exempt": True, "param_triggers": []})),
    Policy("P5", "text exemption + non-default parameter triggers",
           RouteRules.from_json({"text_exempt": True, "param_triggers": list(PARAM_TRIGGERS)})),
)}
# The target objective's policies (module docstring, 2): M6's, each with the budget trigger.
M7_POLICIES = {name: Policy(name, f"{policy.label} + budget trigger",
                            dataclasses.replace(policy.rules or RouteRules(), budget_forced=True))
               for name, policy in POLICIES.items()}


def policies_of(objective: str) -> dict[str, Policy]:
    """The policies the objective chooses among: M7_POLICIES under target, M6's POLICIES otherwise."""
    return M7_POLICIES if objective == TARGET else POLICIES


class InputError(ValueError):
    """Inputs that do not belong together (exit 2)."""


# ---- the computation --------------------------------------------------------------------------------------


def auroc(values: Sequence[float], labels: Sequence[bool]) -> float | None:
    """P(value of a positive > value of a negative), ties counting one half; None without both classes."""
    positives = [v for v, y in zip(values, labels, strict=True) if y]
    negatives = [v for v, y in zip(values, labels, strict=True) if not y]
    if not positives or not negatives:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def aurocs(rows: list[dict]) -> dict[str, float | None]:
    """Each signal's AUROC for FT's success over the items with an ok FT record and a value of it."""
    out = {}
    for signal in SIGNALS:
        kept = [r for r in rows if r["routable"] and r["signals"].get(signal) is not None]
        out[signal] = auroc([r["signals"][signal] for r in kept], [r["ft"] for r in kept])
    return out


def choose_signal(scores: dict[str, float | None]) -> str:
    """The signal with the higher AUROC; a tie (or no AUROC at all) goes to the first of SIGNALS."""
    best = SIGNALS[0]
    for signal in SIGNALS[1:]:
        if scores.get(signal) is not None and (scores.get(best) is None or scores[signal] > scores[best]):
            best = signal
    return best


def is_routed(row: dict, signal: str, threshold: float, rules: RouteRules | None = None) -> bool:
    """Whether a threshold routes the item under the rules: as RoutedModel would (agent.route.route_reason
    over the row's signals, hard reason, calls and whether its reasoning was closed by force)."""
    if not row["routable"]:
        return False
    signals = {**row["signals"], "hard_reason": row["hard"]}
    config = RouteConfig(signal=signal, threshold=threshold, rules=rules)
    return route_reason(signals, config, row.get("calls"), forced=bool(row.get("forced"))) is not None


def _point(rows: list[dict], routed: list[bool], label: str, threshold: float | None,
           ft_success: float, b2_success: float) -> dict:
    n = len(rows)
    successes = sum(r["b2"] if go else r["ft"] for r, go in zip(rows, routed, strict=True))
    count = sum(routed)
    share = count / n if n else 0.0
    return {"label": label, "threshold": threshold, "routed": count, "api_share": share,
            "successes": successes, "success": successes / n if n else None,
            "cost_per_100_rmb": 100 * sum(r["cost"] for r, go in zip(rows, routed, strict=True) if go) / n
            if n else 0.0,
            "random_success": (1 - share) * ft_success + share * b2_success if n else None}


def _rates(rows: list[dict]) -> tuple[float, float]:
    n = len(rows)
    return (sum(r["ft"] for r in rows) / n, sum(r["b2"] for r in rows) / n) if n else (0.0, 0.0)


def threshold_point(rows: list[dict], signal: str, threshold: float,
                    rules: RouteRules | None = None) -> dict:
    ft_success, b2_success = _rates(rows)
    routed = [is_routed(r, signal, threshold, rules) for r in rows]
    return _point(rows, routed, THRESHOLD, threshold, ft_success, b2_success)


def scan(rows: list[dict], signal: str, rules: RouteRules | None = None) -> list[dict]:
    """route_nothing, every distinct value of the signal as a threshold (ascending), route_everything, under
    the rules (None: the threshold alone)."""
    ft_success, b2_success = _rates(rows)
    values = sorted({r["signals"][signal] for r in rows
                     if r["routable"] and r["signals"].get(signal) is not None})
    points = [_point(rows, [False] * len(rows), NOTHING, None, ft_success, b2_success)]
    points += [threshold_point(rows, signal, t, rules) for t in values]
    points.append(_point(rows, [True] * len(rows), EVERYTHING, None, ft_success, b2_success))
    return points


def policy_scan(rows: list[dict], signal: str,
                policies: dict[str, Policy] | None = None) -> dict[str, dict]:
    """Every policy's scan (module docstring, 2 and 3): {name: {"label", "rules", "curve"}}, in the order of
    policies (POLICIES unless given)."""
    policies = POLICIES if policies is None else policies
    return {name: {"label": policy.label, "rules": policy.rules_json(),
                   "curve": scan(rows, signal, policy.rules)}
            for name, policy in policies.items()}


def oracle(rows: list[dict]) -> list[dict]:
    """The best success at every number of routed items (module docstring, 5)."""
    n = len(rows)
    gain = sum(1 for r in rows if not r["ft"] and r["b2"])
    loss = sum(1 for r in rows if r["ft"] and not r["b2"])
    base = sum(r["ft"] for r in rows)
    out = []
    for k in range(n + 1):
        successes = base + min(k, gain) - max(0, k - (n - loss))
        out.append({"routed": k, "api_share": k / n if n else 0.0, "successes": successes,
                    "success": successes / n if n else None})
    return out


def _meeting(points: list[dict], b2_success: float, margin: float) -> list[dict]:
    return [p for p in points if p["label"] == THRESHOLD
            and round(p["success"] - (b2_success - margin), 12) >= 0]


def choose(points: list[dict], b2_success: float, margin: float = MARGIN) -> dict | None:
    """The M4 rule over the threshold points (RULES[SHARE]); None when none meets it."""
    meeting = _meeting(points, b2_success, margin)
    if not meeting:
        return None
    return min(meeting, key=lambda p: (p["api_share"], -p["success"], p["threshold"]))


def best_point(points: list[dict], b2_success: float, objective: str, margin: float = MARGIN) -> dict | None:
    """One policy's best threshold point under the share or the cost objective (module docstring, 4); None
    when no point of the curve meets the margin. The target objective's is target_choice's."""
    if objective == SHARE:
        return choose(points, b2_success, margin)
    if objective != COST:
        raise ValueError(f"unknown objective {objective!r} for a margin; one of {[SHARE, COST]} "
                         "(target_choice chooses under target)")
    meeting = _meeting(points, b2_success, margin)
    if not meeting:
        return None
    return min(meeting, key=lambda p: (p["cost_per_100_rmb"], p["routed"], p["threshold"]))


def choose_policy(policies: dict[str, dict], b2_success: float, objective: str,
                  margin: float = MARGIN) -> tuple[str, dict] | None:
    """The (policy name, point) the objective chooses over the policies' curves (module docstring, 4): under
    share, P0's best alone (the M4 rule); under cost, the cheapest best over every policy, ties to fewer
    routed, the lower threshold, then the policy listed first. None when no point meets the margin."""
    names = [p for p in policies if p == "P0"] if objective == SHARE else list(policies)
    candidates = []
    for index, name in enumerate(names):
        best = best_point(policies[name]["curve"], b2_success, objective, margin)
        if best is not None:
            candidates.append((index, name, best))
    if not candidates:
        return None
    if objective == SHARE:
        _, name, best = candidates[0]
        return name, best
    _, name, best = min(candidates, key=lambda c: (c[2]["cost_per_100_rmb"], c[2]["routed"],
                                                   c[2]["threshold"], c[0]))
    return name, best


def cost_cap(curve: list[dict]) -> float:
    """The cost cap per 100 items (module docstring, 4): CAP_SHARE of the curve's route_everything point's
    cost, B2's list-price cost of every item."""
    everything = [p for p in curve if p["label"] == EVERYTHING]
    if len(everything) != 1:
        raise ValueError(f"a scan has one {EVERYTHING} point, this one {len(everything)}")
    return CAP_SHARE * everything[0]["cost_per_100_rmb"]


def reaches(point: dict, target: float) -> bool:
    """Whether the point's success is at least the target, compared to 12 decimals."""
    return round(point["success"] - target, 12) >= 0


def within(point: dict, cap: float) -> bool:
    """Whether the point's cost per 100 items is at most the cap, compared to 12 decimals."""
    return round(cap - point["cost_per_100_rmb"], 12) >= 0


def target_choice(curves: dict[str, list[dict]], target: float,
                  cap: float) -> tuple[str, dict, bool] | None:
    """The target objective's (policy name, point, whether it reaches the target) over the curves, by policy
    name in their order (module docstring, 4): the cheapest threshold point that reaches the target, else the
    most successful one within the cap; None when no threshold point reaches the target and none is within
    the cap."""
    candidates = [(index, name, point) for index, (name, curve) in enumerate(curves.items())
                  for point in curve if point["label"] == THRESHOLD]
    reaching = [c for c in candidates if reaches(c[2], target)]
    if reaching:
        _, name, point = min(reaching, key=lambda c: (c[2]["cost_per_100_rmb"], -c[2]["success"],
                                                      c[2]["routed"], c[2]["threshold"], c[0]))
        return name, point, True
    affordable = [c for c in candidates if within(c[2], cap)]
    if not affordable:
        return None
    _, name, point = min(affordable, key=lambda c: (-c[2]["success"], c[2]["cost_per_100_rmb"],
                                                    c[2]["routed"], c[2]["threshold"], c[0]))
    return name, point, False


# ---- the inputs -------------------------------------------------------------------------------------------


def _json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise JsonlError(f"{path.as_posix()} is not JSON: {exc}") from None


def _scores(path: Path) -> dict[str, bool]:
    data = _json(path)
    scores = data.get("scores") if isinstance(data, dict) else None
    if not isinstance(scores, list) or not all(isinstance(s, dict) and isinstance(s.get("id"), str)
                                               and isinstance(s.get("success"), bool) for s in scores):
        raise JsonlError(f"{path.as_posix()}: expected scripts/score.py's output, scores with id and success")
    ids = [s["id"] for s in scores]
    if len(set(ids)) != len(ids):
        raise InputError(f"{path.as_posix()}: repeated ids")
    return {s["id"]: s["success"] for s in scores}


def _records(path: Path) -> list[dict]:
    records = read_jsonl(path)
    if not all(isinstance(r.get("id"), str) and isinstance(r.get("item", {}), dict) for r in records):
        raise JsonlError(f"{path.as_posix()}: expected the runner's records, each with an id")
    ids = [r["id"] for r in records]
    if len(set(ids)) != len(ids):
        raise InputError(f"{path.as_posix()}: repeated ids")
    return records


def _calls(path: Path) -> list[dict]:
    data = _json(path)
    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise JsonlError(f"{path.as_posix()}: expected a timing sidecar with items")
    return [c for item in items for c in item.get("calls") or [] if isinstance(c, dict)]


def _manifest(path: Path, system: str, dev_digest: str, dev: Path | None = None) -> dict:
    """The manifest of a run of system over the set whose SHA-256 is dev_digest (dev, when given, names the
    set in the refusal); InputError for another system's or another set's."""
    manifest = _json(path)
    if not isinstance(manifest, dict) or manifest.get("system") != system:
        raise InputError(f"{path.as_posix()} is not the manifest of a {system} run")
    if manifest.get("items_sha256") != dev_digest:
        where = "the development set" if dev is None else f"the set {dev.as_posix()}"
        raise InputError(f"{path.as_posix()}: the run's items have SHA-256 {manifest.get('items_sha256')}, "
                         f"while {where} has SHA-256 {dev_digest} now")
    return manifest


def _generation(manifest: dict, path: Path) -> dict | None:
    """FT's generation as its run's manifest records it (module docstring: inputs), None for a run without
    one; JsonlError for one that is not an object."""
    generation = manifest.get(GENERATION)
    if generation is not None and not isinstance(generation, dict):
        raise JsonlError(f"{path.as_posix()}: the generation must be an object, as scripts/run_eval.py "
                         f"records it, not {generation!r:.200}")
    return generation


def generation_text(generation: dict | None) -> str:
    """A generation as a refusal names it: its canonical JSON, or "none"."""
    return "none" if generation is None else canonical_json(generation)


def _forced(record: dict) -> bool:
    """Whether the record's turn had its reasoning closed by force (its thinking record's forced)."""
    thinking = record.get("thinking")
    return isinstance(thinking, dict) and thinking.get("forced") is True


def build_rows(ft_records: list[dict], ft_scores: dict[str, bool], b2_records: list[dict],
               b2_scores: dict[str, bool], price: object) -> list[dict]:
    """One row per item, in FT's record order (module docstring)."""
    ids = [r["id"] for r in ft_records]
    if not (set(ids) == set(ft_scores) == {r["id"] for r in b2_records} == set(b2_scores)):
        raise InputError("the ids of FT's records, FT's scores, B2's records and B2's scores differ")
    b2_by_id = {r["id"]: r for r in b2_records}
    rows = []
    for record in ft_records:
        ok = record.get("status") == "ok"
        signals = record.get("signals")
        if ok and (not isinstance(signals, dict) or not {*SIGNALS, "hard_reason"} <= set(signals)):
            raise JsonlError(f"FT record {record['id']!r} has no signals: run FT with scripts/run_eval.py "
                             "--signals")
        output = record.get("output")
        if ok and (not isinstance(output, dict) or output.get("role") != "assistant"):
            raise JsonlError(f"FT record {record['id']!r} has no output, the assistant turn the runner "
                             "records for every ok item: not a record of scripts/run_eval.py")
        b2 = b2_by_id[record["id"]]
        usage = b2.get("usage") if b2.get("status") == "ok" else None
        rows.append({"id": record["id"], "ft": ft_scores[record["id"]], "b2": b2_scores[record["id"]],
                     "routable": ok, "signals": {s: (signals or {}).get(s) for s in SIGNALS},
                     "hard": (signals or {}).get("hard_reason"),
                     "calls": api_calls(output) if ok else [],
                     "forced": ok and _forced(record),
                     "cost": cost_rmb(price, usage) if isinstance(usage, dict) else 0.0,
                     "handwritten": in_subset(record.get("item") or {}, "handwritten")})
    return rows


def _beside(records: Path, suffix: str) -> Path:
    return records.with_name(records.stem + suffix)


@dataclass(frozen=True)
class DevInputs:
    """One development set and the inputs over it (module docstring: inputs): the records and scores of FT
    and B2, and beside each records file its manifest and timing sidecar."""

    dev: Path
    ft: Path
    ft_scores: Path
    b2: Path
    b2_scores: Path

    def files(self) -> dict[str, Path]:
        """The eight inputs by the names the configuration records them under, in its order."""
        return {"ft_records": self.ft, "ft_scores": self.ft_scores,
                "ft_manifest": _beside(self.ft, ".manifest.json"),
                "ft_timing": _beside(self.ft, ".timing.json"),
                "b2_records": self.b2, "b2_scores": self.b2_scores,
                "b2_manifest": _beside(self.b2, ".manifest.json"),
                "b2_timing": _beside(self.b2, ".timing.json")}


@dataclass(frozen=True)
class DevRun:
    """One development set as read and checked: its SHA-256, FT's adapter, the cloud model, the rows, the
    timed calls of both runs, the SHA-256 of every input and FT's generation (None without one)."""

    inputs: DevInputs
    digest: str
    adapter: str
    cloud_model: str
    rows: list[dict]
    ft_calls: list[dict]
    b2_calls: list[dict]
    digests: dict[str, str]
    generation: dict | None = None


def read_set(inputs: DevInputs, root: Path) -> DevRun:
    """One development set's inputs read and checked on their own (module docstring: inputs); the exceptions
    main turns into exit codes."""
    files = inputs.files()
    refuse_test_sets(inputs.dev, (Path.cwd(), root))
    ft_records, b2_records = _records(inputs.ft), _records(inputs.b2)
    refuse_test_material([{**(r.get("item") or {}), "id": r["id"]} for r in ft_records + b2_records],
                         "route_curve's records")
    dev_digest = sha256_of(inputs.dev)
    ft_manifest = _manifest(files["ft_manifest"], "ft", dev_digest, inputs.dev)
    b2_manifest = _manifest(files["b2_manifest"], "b2", dev_digest, inputs.dev)
    adapters = ft_manifest.get("adapters")
    if not isinstance(adapters, list) or len(adapters) != 1 or not adapters[0].get("sha256"):
        raise InputError(f"{files['ft_manifest'].as_posix()} must list exactly one adapter with its "
                         f"SHA-256, got {adapters!r}")
    generation = _generation(ft_manifest, files["ft_manifest"])
    cloud_model = b2_manifest.get("model")
    if cloud_model != CLOUD_MODELS[CLOUD_SYSTEM]:
        raise InputError(f"B2's run used {cloud_model!r}, not {CLOUD_MODELS[CLOUD_SYSTEM]}")
    rows = build_rows(ft_records, _scores(inputs.ft_scores), b2_records, _scores(inputs.b2_scores),
                      load_prices()[cloud_model])
    ft_calls, b2_calls = _calls(files["ft_timing"]), _calls(files["b2_timing"])
    digests = {name: sha256_of(path) for name, path in files.items()}
    return DevRun(inputs, dev_digest, adapters[0]["sha256"], cloud_model, rows, ft_calls, b2_calls, digests,
                  generation)


def joined_generation(runs: list[DevRun]) -> dict | None:
    """FT's generation over every set (module docstring: several development sets): the one every set's FT
    manifest records, None when none records one; InputError when the sets' runs were generated otherwise, one
    without a generation beside one with it among them."""
    first = runs[0].generation
    if any(run.generation != first for run in runs[1:]):
        held = "; ".join(f"{run.inputs.dev.as_posix()} {generation_text(run.generation)}" for run in runs)
        raise InputError(f"FT's runs over the development sets were not generated alike (the generation "
                         f"their manifests record): {held}")
    return first


def joined_rows(runs: list[DevRun]) -> list[dict]:
    """The rows of every set, in the sets' order, once the sets belong together: FT's runs used one adapter
    and no item is in two sets; InputError otherwise."""
    if len({run.adapter for run in runs}) > 1:
        used = ", ".join(f"{run.inputs.dev.as_posix()} {run.adapter}" for run in runs)
        raise InputError(f"FT's runs over the development sets used different adapters: {used}")
    held: dict[str, Path] = {}
    rows = []
    for run in runs:
        for row in run.rows:
            if row["id"] in held:
                raise InputError(f"the development sets {held[row['id']].as_posix()} and "
                                 f"{run.inputs.dev.as_posix()} both hold the item {row['id']!r}: the sets "
                                 "must be disjoint")
            held[row["id"]] = run.inputs.dev
        rows += run.rows
    return rows


# ---- the script -------------------------------------------------------------------------------------------


def target_rate(text: str) -> float:
    """--target: a success rate above 0 and at most 1."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a success rate in (0, 1], got {text!r}") from None
    if not math.isfinite(value) or not 0 < value <= 1:
        raise argparse.ArgumentTypeError(f"expected a success rate in (0, 1], got {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Choose and freeze FT+R's routing on the development set.")
    ap.add_argument("--ft", type=Path, action="append", required=True,
                    help="FT's development records made with --signals (once per development set)")
    ap.add_argument("--ft-scores", type=Path, action="append", required=True,
                    help="scripts/score.py's output for --ft (once per development set)")
    ap.add_argument("--b2", type=Path, action="append", default=None,
                    help=f"B2's development records (once per development set; default "
                         f"{B2_RECORDS.as_posix()} with a single set)")
    ap.add_argument("--b2-scores", type=Path, action="append", required=True,
                    help="scripts/score.py's output for --b2 (once per development set)")
    ap.add_argument("--dev", type=Path, action="append", default=None,
                    help=f"the development set (once per set; default {DEV_SET} with a single set)")
    ap.add_argument("--objective", choices=OBJECTIVES, default=COST,
                    help="cost (the M6 rule over the policies, the default), share (the M4 rule over the "
                         "threshold alone) or target (M7: the cheapest point reaching --target)")
    ap.add_argument("--target", type=target_rate, default=None, metavar="T",
                    help="the target objective's development success, in (0, 1]")
    ap.add_argument("--cloud", choices=(CLOUD_SYSTEM,), default=CLOUD_SYSTEM,
                    help=f"the cloud system a routed step goes to: {CLOUD_SYSTEM} "
                         f"({CLOUD_MODELS[CLOUD_SYSTEM]}), the only one")
    ap.add_argument("--out", type=Path, default=ROUTING_CONFIG, help=f"default {ROUTING_CONFIG}")
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    return ap


def input_sets(args: argparse.Namespace, parser: argparse.ArgumentParser) -> list[DevInputs]:
    """The development sets the options name, in their order (module docstring: several development sets);
    a usage error unless every input is given once per set."""
    count = len(args.ft)
    b2 = args.b2 if args.b2 is not None else [B2_RECORDS] if count == 1 else []
    dev = args.dev if args.dev is not None else [DEV_SET] if count == 1 else []
    given = {"--ft": args.ft, "--ft-scores": args.ft_scores, "--b2": b2, "--b2-scores": args.b2_scores,
             "--dev": dev}
    uneven = [f"{option} {len(values)}" for option, values in given.items() if len(values) != count]
    if uneven:
        parser.error(f"give {', '.join(INPUT_OPTIONS)} once per development set: --ft names {count} "
                     f"set(s), but {'; '.join(uneven)}")
    return [DevInputs(*paths) for paths in zip(dev, args.ft, args.ft_scores, b2, args.b2_scores, strict=True)]


def main(argv: list[str] | None = None, *, root: Path = REPO) -> int:
    """root is the repository whose relative paths the configuration records (a test's folder in tests)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.objective == TARGET and args.target is None:
        parser.error("--objective target needs --target T, the development success to reach")
    if args.target is not None and args.objective != TARGET:
        parser.error(f"--target is the target objective's; the {args.objective} objective has none")
    sets = input_sets(args, parser)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    if args.out.exists() and not args.force:
        print(f"FAIL: {args.out} exists; --force overwrites it", file=sys.stderr)
        return 3
    try:
        runs = [read_set(inputs, root) for inputs in sets]
        rows = joined_rows(runs)
        generation = joined_generation(runs)
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

    objective, target = args.objective, args.target
    scores = aurocs(rows)
    signal = choose_signal(scores)
    ft_success, b2_success = _rates(rows)
    candidates = policies_of(objective)
    policies = policy_scan(rows, signal, candidates)
    cap = reached = None
    if objective == TARGET:
        cap = cost_cap(policies[next(iter(policies))]["curve"])  # the curves share their endpoints
        for name, policy in policies.items():
            own = target_choice({name: policy["curve"]}, target, cap)
            policy["chosen"], policy["target_reached"] = (None, False) if own is None else own[1:]
        choice = target_choice({name: policy["curve"] for name, policy in policies.items()}, target, cap)
        chosen_policy, chosen, reached = (None, None, False) if choice is None else choice
    else:
        for policy in policies.values():
            policy["chosen"] = best_point(policy["curve"], b2_success, objective)
        choice = choose_policy(policies, b2_success, objective)
        chosen_policy, chosen = (None, None) if choice is None else choice
    threshold = None if chosen is None else chosen["threshold"]
    rules = None if chosen_policy is None else candidates[chosen_policy].rules
    hand_rows = [r for r in rows if r["handwritten"]]
    handwritten = {"n": len(hand_rows), "ft_success": _rates(hand_rows)[0] if hand_rows else None,
                   "b2_success": _rates(hand_rows)[1] if hand_rows else None,
                   "scan": scan(hand_rows, signal, rules) if hand_rows else [],
                   "at_threshold": threshold_point(hand_rows, signal, threshold, rules)
                   if hand_rows and threshold is not None else None,
                   "policy": chosen_policy}

    def relative(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            return resolved.as_posix()

    def inputs_of(run: DevRun) -> dict:
        return {name: {"path": relative(path), "sha256": run.digests[name]}
                for name, path in run.inputs.files().items()}

    single = runs[0] if len(runs) == 1 else None
    config: dict = {"rule": RULES[objective], "objective": objective}
    if objective == TARGET:
        config.update({"target": target, "target_reached": reached, "cost_cap_per_100_rmb": cap,
                       "within_cap": None if chosen is None else within(chosen, cap)})
    else:
        config["margin"] = MARGIN
    config.update({"rule_met": chosen is not None, "signal": signal, "threshold": threshold,
                   "chosen_policy": chosen_policy, "rules": None if rules is None else rules.to_json()})
    if single is not None:
        config["dev_sha256"] = single.digest
    config["adapter_sha256"] = runs[0].adapter
    if generation is not None:  # FT's runs on the family path (module docstring: output)
        config[FT_GENERATION] = generation
    config.update({"cloud_model": runs[0].cloud_model, "local_system": "ft", "cloud_system": CLOUD_SYSTEM})
    if single is not None:
        config.update({"dev_set": relative(single.inputs.dev), "inputs": inputs_of(single)})
    else:
        config["dev_sets"] = [{"path": relative(run.inputs.dev), "sha256": run.digest, "n": len(run.rows),
                               "inputs": inputs_of(run)} for run in runs]
    config.update({
        "n": len(rows), "ft_success": ft_success, "b2_success": b2_success,
        "auroc": scores, "signal_order": list(SIGNALS),
        "chosen_point": None if chosen is None else {**chosen, "policy": chosen_policy},
        "policies": policies,
        # the chosen policy's scan, so that the chosen point lies on it (P0's, the threshold alone, when no
        # point was chosen): what the figures and the report read as the routing curve
        "curves": {"routing": policies[chosen_policy or "P0"]["curve"], "oracle": oracle(rows)},
        "handwritten": handwritten,
        "latency_ms": {
            "local_per_call": latency(c["wall_ms"] for run in runs for c in run.ft_calls
                                      if c.get("error") is None),
            "cloud_per_paid_call": latency(c["wall_ms"] for run in runs for c in run.b2_calls
                                           if c.get("error") is None and c.get("paid") is True)},
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                        newline="\n")
    report(config, runs, args.out)
    return 0


def report(config: dict, runs: list[DevRun], out: Path) -> None:
    """What main prints once the configuration is written."""
    objective, policies, chosen = config["objective"], config["policies"], config["chosen_point"]
    if len(runs) > 1:
        print(f"{len(runs)} development sets, {config['n']} items: "
              + ", ".join(f"{run.inputs.dev.as_posix()} {len(run.rows)}" for run in runs))
    print(f"AUROC {', '.join(f'{k} {v}' for k, v in config['auroc'].items())}; chosen {config['signal']}; "
          f"objective {objective}")
    target = config.get("target")
    if objective == TARGET:
        print(f"target {target:g}; cost cap {config['cost_cap_per_100_rmb']:.4f} RMB per 100 items (half of "
              f"routing every item to {CLOUD_SYSTEM})")
    for name, policy in policies.items():
        best = policy["chosen"]
        if best is None:
            outcome = ("no point reaches the target or is within the cap" if objective == TARGET
                       else "no threshold meets the margin")
        else:
            outcome = (f"best threshold {best['threshold']:.6g}, routed {best['routed']}, "
                       f"success {best['success']:.4f}, {best['cost_per_100_rmb']:.4f} RMB per 100 items")
            if objective == TARGET:
                outcome += "; target reached" if policy["target_reached"] else "; target not reached"
        print(f"  {name} ({policy['label']}): {outcome}")
    if chosen is None:
        if objective == TARGET:
            print(f"the rule is not met: no point reaches the target {target:g} and none is within the cap; "
                  f"{out} records it with no threshold")
        else:
            print(f"the rule is not met: no point meets success >= B2 {config['b2_success']:.4f} - {MARGIN}; "
                  f"{out} records it with no threshold")
        return
    line = (f"chosen {config['chosen_policy']} at threshold {config['threshold']:.6g}: API share "
            f"{chosen['api_share']:.4f}, success {chosen['success']:.4f} (FT {config['ft_success']:.4f}, "
            f"B2 {config['b2_success']:.4f}), {chosen['cost_per_100_rmb']:.4f} RMB per 100 items")
    if objective == TARGET:
        line += ("; target reached" if config["target_reached"]
                 else "; target not reached: the highest success within the cap")
        line += "" if config["within_cap"] else "; over the cost cap"
    print(f"{line}; written {out}")


if __name__ == "__main__":
    sys.exit(main())

"""scripts/route_curve.py: the AUROC of the two signals, the threshold scan with the random and oracle curves,
the preregistered rule and the frozen configuration it writes, offline over fabricated records; M6's policies
and cost objective; M7's target objective with its cost cap and budget trigger over several development sets
at once; and, where this machine has the recorded development runs, results/routing-m4.json and
results/routing-m6.json rebuilt byte for byte."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from research_agent.agent.route import RouteRules, api_calls
from research_agent.eval.lock import sha256_of
from research_agent.llm.budget import cost_rmb, load_prices
from research_agent.tools.schemas import DEFAULTS

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


route_curve = load_script("route_curve")
MDP, ML = "min_decision_prob", "mean_logprob"

# Six development items: FT right on a, b, e; B2 right on a, c, d, e; f has a hard trigger; e and f are
# handwritten. FT 3/6, B2 4/6, so the rule needs at least 4 successes (4/6 >= 4/6 - 0.02).
IDS = ["a", "b", "c", "d", "e", "f"]
FT_OK = [True, True, False, False, True, False]
B2_OK = [True, False, True, True, True, False]
MDPS = [0.9, 0.8, 0.3, 0.5, 0.7, 0.2]
MLS = [-0.1, -0.5, -0.2, -0.3, -0.4, -0.6]
HARD = [None, None, None, None, None, "unknown_tool: 'search_web' is not among the step's tools"]
HANDWRITTEN = [False, False, False, False, True, True]
USAGE = [{"prompt_tokens": 1000 * (i + 1), "completion_tokens": 100 * (i + 1)} for i in range(6)]
PRICE = load_prices()["deepseek-flash"]
COSTS = [cost_rmb(PRICE, u) for u in USAGE]


def rows() -> list[dict]:
    return [{"id": i, "ft": f, "b2": b, "routable": True, "signals": {MDP: m, ML: lg}, "hard": h, "cost": c,
             "handwritten": w}
            for i, f, b, m, lg, h, c, w in zip(IDS, FT_OK, B2_OK, MDPS, MLS, HARD, COSTS, HANDWRITTEN,
                                               strict=True)]


# ---- AUROC ----------------------------------------------------------------------------------------------


def test_auroc_by_hand_with_a_tie_counting_one_half():
    # pairs (positive, negative): 0.9>0.8, 0.9>0.3, 0.8=0.8 (1/2), 0.8>0.3: 3.5 of 4
    assert route_curve.auroc([0.9, 0.8, 0.8, 0.3], [True, True, False, False]) == 0.875
    assert route_curve.auroc([0.1, 0.2], [True, False]) == 0.0
    assert route_curve.auroc([0.1, 0.2], [True, True]) is None  # no negative: undefined


def test_the_higher_auroc_is_chosen_and_a_tie_goes_to_min_decision_prob():
    assert route_curve.choose_signal({MDP: 0.6, ML: 0.7}) == ML
    assert route_curve.choose_signal({MDP: 0.7, ML: 0.7}) == MDP
    assert route_curve.choose_signal({MDP: None, ML: 0.7}) == ML
    aurocs = route_curve.aurocs(rows())
    assert aurocs == {MDP: 1.0, ML: pytest.approx(5 / 9)}


# ---- the scan and the curves ----------------------------------------------------------------------------


def test_the_scan_routes_hard_triggers_and_signals_below_the_threshold():
    points = {p["threshold"]: p for p in route_curve.scan(rows(), MDP) if p["label"] == "threshold"}
    assert sorted(points) == sorted(MDPS)
    assert [points[t]["routed"] for t in sorted(points)] == [1, 1, 2, 3, 4, 5]
    assert [points[t]["successes"] for t in sorted(points)] == [3, 3, 4, 5, 5, 4]
    assert points[0.5]["api_share"] == pytest.approx(2 / 6)
    assert points[0.5]["cost_per_100_rmb"] == pytest.approx(100 * (COSTS[2] + COSTS[5]) / 6)


def test_the_curves_endpoints_are_fts_and_b2s_success():
    points = route_curve.scan(rows(), MDP)
    first, last = points[0], points[-1]
    assert (first["label"], first["api_share"], first["success"]) == ("route_nothing", 0.0, 3 / 6)
    assert (last["label"], last["api_share"], last["success"]) == ("route_everything", 1.0, 4 / 6)
    assert last["cost_per_100_rmb"] == pytest.approx(100 * sum(COSTS) / 6)
    assert first["cost_per_100_rmb"] == 0.0


def test_the_random_curve_is_linear_between_ft_and_b2():
    for point in route_curve.scan(rows(), MDP):
        s = point["api_share"]
        assert point["random_success"] == pytest.approx((1 - s) * 3 / 6 + s * 4 / 6)


def test_the_oracle_routes_ft_failures_b2_fixes_first_and_is_never_below_a_scanned_point():
    oracle = route_curve.oracle(rows())
    assert [p["successes"] for p in oracle] == [3, 4, 5, 5, 5, 5, 4]
    by_routed = {p["routed"]: p["success"] for p in oracle}
    for point in route_curve.scan(rows(), MDP):
        assert by_routed[point["routed"]] >= point["success"]


def test_the_rule_takes_the_smallest_share_within_two_points_of_b2():
    points = route_curve.scan(rows(), MDP)
    chosen = route_curve.choose(points, b2_success=4 / 6)
    assert chosen["threshold"] == 0.5 and chosen["successes"] == 4


def test_ties_go_to_the_higher_success_then_the_lower_threshold():
    points = [{"label": "threshold", "threshold": 0.4, "api_share": 0.5, "success": 0.70, "successes": 7},
              {"label": "threshold", "threshold": 0.3, "api_share": 0.5, "success": 0.70, "successes": 7},
              {"label": "threshold", "threshold": 0.2, "api_share": 0.5, "success": 0.60, "successes": 6},
              {"label": "route_everything", "threshold": None, "api_share": 1.0, "success": 0.9,
               "successes": 9}]
    assert route_curve.choose(points, b2_success=0.6)["threshold"] == 0.3
    assert route_curve.choose(points, b2_success=0.9) is None  # only routing everything would meet it


def test_a_non_ok_ft_record_is_never_routed_by_a_threshold():
    data = rows()
    data[2]["routable"] = False  # c: the local model raised, the runner recorded an error
    data[2]["signals"] = {MDP: None, ML: None}
    points = {p["threshold"]: p for p in route_curve.scan(data, MDP) if p["label"] == "threshold"}
    assert 0.3 not in points and points[0.5]["routed"] == 1  # only f's hard trigger


def test_a_missing_signal_value_counts_as_below_every_threshold():
    data = rows()
    data[0]["signals"] = {MDP: None, ML: -0.1}
    points = [p for p in route_curve.scan(data, MDP) if p["label"] == "threshold"]
    assert all(p["routed"] >= 2 for p in points)  # a and f at every threshold


# ---- the script ------------------------------------------------------------------------------------------


def write_jsonl(path: Path, rows_: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_), encoding="utf-8",
                    newline="\n")
    return path


def write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return path


def output(calls: list[tuple[str, dict]]) -> dict:
    """A recorded turn in API form (the runner's "output" of an ok record): the calls, or a plain answer
    without any."""
    return {"role": "assistant", "content": "" if calls else "好的", "tool_calls": [
        {"id": f"local_{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}
        for i, (name, args) in enumerate(calls)]}


# Every one of the six turns calls a tool, none is a text answer and none gives a parameter the triggers read,
# so the M6 rules change nothing on them: every policy's scan is the threshold's. f's call is the tool its
# hard reason names.
CALLS = [[("search_arxiv", {"query": "x"})], [("get_paper", {"arxiv_id": "2609.01234"})],
         [("search_arxiv", {"query": "y"})], [("get_paper", {"arxiv_id": "2609.05678"})],
         [("search_arxiv", {"query": "z"})], [("search_web", {"query": "x"})]]


def fixture(root: Path, *, signals: bool = True) -> dict[str, Path]:
    """The inputs as the repository would hold them, under root: every ok FT record with its output."""
    (root / "data" / "eval").mkdir(parents=True)
    (root / "runs" / "eval").mkdir(parents=True)
    dev = write_jsonl(root / "data" / "eval" / "dev.jsonl", [{"id": i, "split": "dev"} for i in IDS])
    digest = sha256_of(dev)
    ev = root / "runs" / "eval"

    def item(i: int) -> dict:
        source = {"handwritten": "hw"} if HANDWRITTEN[i] else {"request": IDS[i]}
        return {"split": "dev", "source": source, "type": "call"}

    ft_records = []
    for i, rid in enumerate(IDS):
        record = {"id": rid, "status": "ok", "item": item(i), "model": "qwen3-1.7b:lora=1", "paid": False,
                  "output": output(CALLS[i])}
        if signals:
            record["signals"] = {MDP: MDPS[i], ML: MLS[i], "hard_reason": HARD[i]}
        ft_records.append(record)
    paths = {"dev": dev,
             "ft": write_jsonl(ev / "dev-ft-signals.jsonl", ft_records),
             "ft_scores": write_json(ev / "ft-scores.json", {"scores": [
                 {"id": rid, "success": ok} for rid, ok in zip(IDS, FT_OK, strict=True)]}),
             "b2": write_jsonl(ev / "dev-b2-m3.jsonl", [
                 {"id": rid, "status": "ok", "item": item(i), "usage": USAGE[i], "paid": i == 0,
                  "cached": i != 0, "model": "deepseek-flash"} for i, rid in enumerate(IDS)]),
             "b2_scores": write_json(ev / "b2-scores.json", {"scores": [
                 {"id": rid, "success": ok} for rid, ok in zip(IDS, B2_OK, strict=True)]})}
    write_json(ev / "dev-ft-signals.manifest.json", {"system": "ft", "items_sha256": digest,
                                                     "adapters": [{"id": 0, "path": "x.gguf",
                                                                   "sha256": "ad" * 32}]})
    write_json(ev / "dev-b2-m3.manifest.json", {"system": "b2", "items_sha256": digest,
                                                "model": "deepseek-flash"})
    write_json(ev / "dev-ft-signals.timing.json", {"items": [
        {"id": rid, "calls": [{"wall_ms": 100.0 * (i + 1), "error": None, "paid": False, "cached": False}]}
        for i, rid in enumerate(IDS)]})
    write_json(ev / "dev-b2-m3.timing.json", {"items": [
        {"id": rid, "calls": [{"wall_ms": 1000.0 + i, "error": None, "paid": i < 2, "cached": i >= 2}]}
        for i, rid in enumerate(IDS)]})
    return paths


def argv(paths: dict[str, Path], out: Path, *extra: str) -> list[str]:
    return ["--ft", str(paths["ft"]), "--ft-scores", str(paths["ft_scores"]), "--b2", str(paths["b2"]),
            "--b2-scores", str(paths["b2_scores"]), "--dev", str(paths["dev"]), "--out", str(out), *extra]


def test_the_script_writes_the_frozen_configuration_with_its_inputs_and_curves(tmp_path):
    paths = fixture(tmp_path)
    out = tmp_path / "results" / "routing-m4.json"
    assert route_curve.main(argv(paths, out), root=tmp_path) == 0
    raw = out.read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"}\n") and raw.startswith(b'{\n  "')
    config = json.loads(raw)
    assert (config["signal"], config["threshold"], config["rule_met"]) == (MDP, 0.5, True)
    assert config["dev_sha256"] == sha256_of(paths["dev"]) and config["adapter_sha256"] == "ad" * 32
    assert config["cloud_model"] == "deepseek-flash"
    assert config["auroc"] == {MDP: 1.0, ML: pytest.approx(5 / 9)}
    assert config["dev_set"] == "data/eval/dev.jsonl"
    inputs = config["inputs"]
    assert inputs["ft_records"] == {"path": "runs/eval/dev-ft-signals.jsonl",
                                    "sha256": sha256_of(paths["ft"])}
    assert set(inputs) == {"ft_records", "ft_scores", "ft_manifest", "ft_timing", "b2_records", "b2_scores",
                           "b2_manifest", "b2_timing"}
    assert all(not Path(v["path"]).is_absolute() for v in inputs.values())
    assert (config["ft_success"], config["b2_success"]) == (0.5, pytest.approx(4 / 6))
    curves = config["curves"]
    assert curves["routing"][0]["label"] == "route_nothing" and curves["routing"][-1]["success"] == 4 / 6
    assert [p["successes"] for p in curves["oracle"]] == [3, 4, 5, 5, 5, 5, 4]
    hand = config["handwritten"]
    assert hand["n"] == 2 and hand["at_threshold"]["routed"] == 1  # f's hard trigger; e stays local
    latency = config["latency_ms"]
    assert latency["local_per_call"]["n"] == 6 and latency["local_per_call"]["p50_ms"] == 350.0
    assert latency["cloud_per_paid_call"]["n"] == 2 and latency["cloud_per_paid_call"]["p50_ms"] == 1000.5


def test_an_existing_configuration_is_kept_unless_forced(tmp_path):
    paths = fixture(tmp_path)
    out = tmp_path / "routing-m4.json"
    out.write_text("{}\n", encoding="utf-8")
    assert route_curve.main(argv(paths, out), root=tmp_path) == 3
    assert out.read_text(encoding="utf-8") == "{}\n"
    assert route_curve.main(argv(paths, out, "--force"), root=tmp_path) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["threshold"] == 0.5


def test_records_made_without_signals_are_malformed_input(tmp_path, capsys):
    paths = fixture(tmp_path, signals=False)
    assert route_curve.main(argv(paths, tmp_path / "o.json"), root=tmp_path) == 1
    assert "--signals" in capsys.readouterr().err
    assert not (tmp_path / "o.json").exists()


def test_inputs_over_other_items_or_another_dev_set_are_refused(tmp_path, capsys):
    paths = fixture(tmp_path)
    scores = json.loads(paths["b2_scores"].read_text(encoding="utf-8"))
    write_json(paths["b2_scores"], {"scores": scores["scores"][:-1]})
    assert route_curve.main(argv(paths, tmp_path / "o.json"), root=tmp_path) == 2
    assert "ids" in capsys.readouterr().err
    paths = fixture(tmp_path / "second")
    write_jsonl(paths["dev"], [{"id": "z", "split": "dev"}])  # the dev set changed since the runs
    assert route_curve.main(argv(paths, tmp_path / "o.json"), root=tmp_path / "second") == 2
    assert "SHA-256" in capsys.readouterr().err
    assert not (tmp_path / "o.json").exists()


def test_a_rule_that_no_threshold_meets_is_written_as_not_met(tmp_path, capsys):
    paths = fixture(tmp_path)
    # B2 6/6, FT wrong on a, the item with the highest signal, which no threshold routes: every threshold
    # stays at most 5/6, below 6/6 - 0.02
    write_json(paths["b2_scores"], {"scores": [{"id": rid, "success": True} for rid in IDS]})
    write_json(paths["ft_scores"], {"scores": [{"id": rid, "success": ok and rid != "a"}
                                               for rid, ok in zip(IDS, FT_OK, strict=True)]})
    out = tmp_path / "o.json"
    assert route_curve.main(argv(paths, out), root=tmp_path) == 0
    config = json.loads(out.read_text(encoding="utf-8"))
    assert config["rule_met"] is False and config["threshold"] is None
    assert "no threshold meets" in capsys.readouterr().out


def test_test_material_is_refused(tmp_path):
    paths = fixture(tmp_path)
    records = [json.loads(line) for line in paths["ft"].read_text(encoding="utf-8").splitlines()]
    records[0]["item"]["split"] = "test"
    write_jsonl(paths["ft"], records)
    assert route_curve.main(argv(paths, tmp_path / "o.json"), root=tmp_path) == 4


# ---- M6: the policies, the cost objective and the rules (docs/m6-计划.md §3.3 C1-C3) --------------------

# Eight items. FT right on a, b, e, g (4/8); B2 right on a, c, d, e, f, h (6/8), so a point meets the margin
# with 6 successes (6/8 >= 6/8 - 0.02) and not with 5. b and e are plain text answers; a's search_notes keeps
# the default top_k, c's asks for 5; d's list_notes asks for 20; f names a tool the step has not (its hard
# trigger); g and h call tools without a parameter the triggers read. B2's usage is prompt tokens alone at
# 2 RMB per million, so 500 tokens cost 0.001.
P_IDS = list("abcdefgh")
P_FT = [True, True, False, False, True, False, True, False]
P_B2 = [True, False, True, True, True, True, False, True]
P_MDP = [0.9, 0.8, 0.95, 0.85, 0.3, 0.2, 0.6, 0.7]
P_HARD = [None, None, None, None, None, HARD[5], None, None]
P_COST = [0.010, 0.002, 0.001, 0.004, 0.008, 0.016, 0.020, 0.003]
P_ML = [-1.0, -1.0, -0.1, -0.1, -1.0, -0.1, -1.0, -0.1]  # AUROC 0 for FT's success: mdp is chosen
P_HAND = [False, False, True, True, True, False, False, False]  # c, d and e are handwritten
P_CALLS = [[("search_notes", {"query": "x", "top_k": 3})], [], [("search_notes", {"query": "x", "top_k": 5})],
           [("list_notes", {"limit": 20})], [], [("search_web", {"query": "x"})],
           [("get_paper", {"arxiv_id": "2609.01234"})], [("search_arxiv", {"query": "x"})]]
P_THRESHOLDS = sorted(P_MDP)
P1_JSON = {"text_exempt": True, "param_triggers": []}
P5_JSON = {"text_exempt": True, "param_triggers": [
    {"tool": "search_notes", "param": "top_k", "allowed": [3]},
    {"tool": "list_notes", "param": "limit", "allowed": [10]}]}
P1_RULES, P5_RULES = RouteRules.from_json(P1_JSON), RouteRules.from_json(P5_JSON)


def policy_rows() -> list[dict]:
    return [{"id": i, "ft": f, "b2": b, "routable": True, "signals": {MDP: m, ML: lg}, "hard": h,
             "cost": c, "handwritten": w, "calls": api_calls(output(k))}
            for i, f, b, m, lg, h, c, w, k in zip(P_IDS, P_FT, P_B2, P_MDP, P_ML, P_HARD, P_COST, P_HAND,
                                                  P_CALLS, strict=True)]


def per_100(cost: float) -> float:
    return 100 * cost / len(P_IDS)


def test_the_policies_are_the_threshold_alone_the_text_exemption_and_the_exemption_with_the_triggers():
    assert list(route_curve.POLICIES) == ["P0", "P1", "P5"]
    assert route_curve.POLICIES["P0"].rules is None
    assert route_curve.POLICIES["P1"].rules == P1_RULES
    assert route_curve.POLICIES["P5"].rules == P5_RULES
    for trigger in P5_RULES.param_triggers:  # what does not trigger is the tool's default value
        assert list(trigger.allowed) == [DEFAULTS[trigger.tool][trigger.param]]


def test_the_exemption_keeps_text_answers_local_and_the_triggers_route_non_default_values():
    rows_ = policy_rows()

    def routed(rules, threshold):
        return [r["id"] for r in rows_ if route_curve.is_routed(r, MDP, threshold, rules)]

    assert routed(None, 0.9) == ["b", "d", "e", "f", "g", "h"]
    assert routed(P1_RULES, 0.9) == ["d", "f", "g", "h"]
    assert routed(P5_RULES, 0.2) == ["c", "d", "f"]
    assert routed(P5_RULES, 0.9) == ["c", "d", "f", "g", "h"]


def test_the_policy_scan_by_hand():
    policies = route_curve.policy_scan(policy_rows(), MDP)
    assert list(policies) == ["P0", "P1", "P5"]
    curves = {name: [p for p in policies[name]["curve"] if p["label"] == "threshold"] for name in policies}
    assert all([p["threshold"] for p in curve] == P_THRESHOLDS for curve in curves.values())
    assert [p["successes"] for p in curves["P0"]] == [5, 5, 5, 4, 5, 4, 5, 5]
    assert [p["routed"] for p in curves["P0"]] == [1, 1, 2, 3, 4, 5, 6, 7]
    assert [p["successes"] for p in curves["P1"]] == [5, 5, 5, 4, 5, 5, 6, 6]
    assert [p["routed"] for p in curves["P1"]] == [1, 1, 1, 2, 3, 3, 4, 5]
    assert [p["successes"] for p in curves["P5"]] == [7, 7, 7, 6, 7, 7, 7, 7]
    assert [p["routed"] for p in curves["P5"]] == [3, 3, 3, 4, 5, 5, 5, 6]
    assert curves["P5"][0]["cost_per_100_rmb"] == pytest.approx(per_100(0.021))  # c, d, f
    assert curves["P1"][6]["cost_per_100_rmb"] == pytest.approx(per_100(0.043))  # d, f, g, h at 0.9
    assert policies["P0"]["rules"] is None and policies["P5"]["rules"] == P5_JSON
    for name in policies:  # the endpoints are every policy's: FT alone and B2 alone
        first, last = policies[name]["curve"][0], policies[name]["curve"][-1]
        assert (first["label"], first["successes"]) == ("route_nothing", 4)
        assert (last["label"], last["successes"]) == ("route_everything", 6)


def test_the_cost_objective_takes_the_cheapest_point_within_the_margin_over_every_policy():
    policies = route_curve.policy_scan(policy_rows(), MDP)
    name, point = route_curve.choose_policy(policies, b2_success=6 / 8, objective="cost")
    assert name == "P5" and point["threshold"] == 0.2 and point["routed"] == 3 and point["successes"] == 7
    assert point["cost_per_100_rmb"] == pytest.approx(per_100(0.021))
    # each policy's own best under the objective: P0 never meets the margin, P1 does at 0.9
    assert route_curve.best_point(policies["P0"]["curve"], 6 / 8, "cost") is None
    assert route_curve.best_point(policies["P1"]["curve"], 6 / 8, "cost")["threshold"] == 0.9
    assert route_curve.best_point(policies["P5"]["curve"], 6 / 8, "cost") == point
    # the share objective is the old rule over the threshold alone: not met on these items
    assert route_curve.choose_policy(policies, b2_success=6 / 8, objective="share") is None


def test_cost_ties_go_to_fewer_routed_then_the_lower_threshold_then_the_policy_listed_first():
    def point(threshold, routed, cost, success=0.9):
        return {"label": "threshold", "threshold": threshold, "routed": routed, "api_share": routed / 10,
                "successes": round(success * 10), "success": success, "cost_per_100_rmb": cost}

    def chosen(p0: list[dict], p1: list[dict], **kw) -> tuple[str, float, int]:
        policies = {"P0": {"curve": p0}, "P1": {"curve": p1}}
        name, best = route_curve.choose_policy(policies, b2_success=0.9, objective="cost", **kw)
        return name, best["threshold"], best["routed"]

    # Each pair of fixtures differs in one key only, so each wrong order of the keys fails one assertion.
    # Within one policy (best_point, over hand-made points: a real scan routes more as the threshold
    # rises): at one cost, fewer routed items beat the lower threshold; at one count, the lower threshold.
    best = route_curve.best_point([point(0.5, 2, 0.2), point(0.2, 3, 0.2), point(0.4, 2, 0.2)], 0.9, "cost")
    assert (best["threshold"], best["routed"]) == (0.4, 2)
    # Across the policies: the cost first, whatever the count, the threshold and the order of the policies
    assert chosen([point(0.2, 2, 0.2)], [point(0.5, 5, 0.1)]) == ("P1", 0.5, 5)
    # at one cost, fewer routed items, though at the higher threshold and on the policy listed second
    assert chosen([point(0.2, 3, 0.2)], [point(0.5, 2, 0.2)]) == ("P1", 0.5, 2)
    # at one cost and count, the lower threshold, though on the policy listed second
    assert chosen([point(0.5, 2, 0.2)], [point(0.2, 2, 0.2)]) == ("P1", 0.2, 2)
    # at one cost, count and threshold, the policy listed first
    assert chosen([point(0.2, 2, 0.2)], [point(0.2, 2, 0.2)]) == ("P0", 0.2, 2)
    # the margin: a cheaper point below it is left out until the margin lets it in
    p0 = [point(0.4, 3, 0.2), point(0.3, 2, 0.2), point(0.2, 2, 0.2)]
    p1 = [point(0.2, 2, 0.2), point(0.1, 1, 0.3), point(0.05, 1, 0.1, success=0.5)]
    assert chosen(p0, p1) == ("P0", 0.2, 2)
    assert chosen(p0, p1, margin=0.5) == ("P1", 0.05, 1)


def thinking(forced: bool) -> dict:
    """The thinking record of a local step in think mode (agent.types.AssistantTurn.thinking), as the runner
    records it beside the signals: a reasoning closed by force takes a second request for the answer (two
    phases), one closed by itself here answers within the first."""
    return {"mode": "think", "reasoning_tokens": 8, "answer_tokens": 5, "forced": forced,
            "phases": 2 if forced else 1, "seeds": None}


def thought(rid: str, *, forced: tuple[str, ...] = (), think: bool = False) -> dict:
    """What a run in think mode adds to the FT record of item rid, its reasoning and its thinking record: the
    reasoning of an item forced names closed by force; with think (the whole run thought, as M7's thinking
    family does on every step) every other item's closed by itself; nothing otherwise."""
    if rid in forced:
        return {"reasoning": "The budget ran out.", "thinking": thinking(True)}
    if think:
        return {"reasoning": "Thought it through.", "thinking": thinking(False)}
    return {}


def policy_fixture(root: Path, *, forced: tuple[str, ...] = (), think: bool = False) -> dict[str, Path]:
    """The eight items as the repository would hold them, FT's records with their outputs, under root; the
    records of the items forced names were made in think mode, their reasoning closed by force, and with think
    every other record was made in think mode too, its reasoning closed by itself (thought)."""
    (root / "data" / "eval").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "m6").mkdir(parents=True, exist_ok=True)
    dev = write_jsonl(root / "data" / "eval" / "dev.jsonl", [{"id": i, "split": "dev"} for i in P_IDS])
    digest = sha256_of(dev)
    run = root / "runs" / "m6"
    def item(i: int) -> dict:
        source = {"handwritten": "hw"} if P_HAND[i] else {"request": "r"}
        return {"split": "dev", "source": source, "type": "call"}

    paths = {"dev": dev,
             "ft": write_jsonl(run / "dev-ft2-signals.jsonl", [
                 {"id": rid, "status": "ok", "item": item(i), "model": "qwen3-1.7b:lora=1", "paid": False,
                  "output": output(P_CALLS[i]),
                  "signals": {MDP: P_MDP[i], ML: P_ML[i], "hard_reason": P_HARD[i]},
                  **thought(rid, forced=forced, think=think)}
                 for i, rid in enumerate(P_IDS)]),
             "ft_scores": write_json(run / "ft2-scores.json", {"scores": [
                 {"id": rid, "success": ok} for rid, ok in zip(P_IDS, P_FT, strict=True)]}),
             "b2": write_jsonl(run / "dev-b2.jsonl", [
                 {"id": rid, "status": "ok", "item": item(i), "model": "deepseek-flash", "paid": False,
                  "cached": True,
                  "usage": {"prompt_tokens": round(P_COST[i] * 1e6 / 2), "completion_tokens": 0}}
                 for i, rid in enumerate(P_IDS)]),
             "b2_scores": write_json(run / "b2-scores.json", {"scores": [
                 {"id": rid, "success": ok} for rid, ok in zip(P_IDS, P_B2, strict=True)]})}
    write_json(run / "dev-ft2-signals.manifest.json", {
        "system": "ft", "items_sha256": digest,
        "adapters": [{"id": 0, "path": "ft2.gguf", "sha256": "f2" * 32}]})
    write_json(run / "dev-b2.manifest.json", {"system": "b2", "items_sha256": digest,
                                              "model": "deepseek-flash"})
    write_json(run / "dev-ft2-signals.timing.json", {"items": [
        {"id": rid, "calls": [{"wall_ms": 100.0, "error": None, "paid": False, "cached": False}]}
        for rid in P_IDS]})
    write_json(run / "dev-b2.timing.json", {"items": [
        {"id": rid, "calls": [{"wall_ms": 1000.0, "error": None, "paid": False, "cached": True}]}
        for rid in P_IDS]})
    return paths


def test_the_script_writes_the_chosen_policy_its_rules_and_every_policys_curve(tmp_path):
    paths = policy_fixture(tmp_path)
    out = tmp_path / "results" / "routing-m6.json"
    assert route_curve.main(argv(paths, out), root=tmp_path) == 0
    config = json.loads(out.read_text(encoding="utf-8"))
    assert (config["objective"], config["chosen_policy"], config["rule_met"]) == ("cost", "P5", True)
    assert config["rule"] == route_curve.RULES["cost"] and config["margin"] == 0.02
    assert (config["threshold"], config["rules"], config["signal"]) == (0.2, P5_JSON, MDP)
    chosen = config["chosen_point"]
    assert (chosen["policy"], chosen["routed"], chosen["successes"]) == ("P5", 3, 7)
    assert chosen["cost_per_100_rmb"] == pytest.approx(per_100(0.021))
    policies = config["policies"]
    assert list(policies) == ["P0", "P1", "P5"] and policies["P0"]["chosen"] is None
    assert policies["P1"]["chosen"]["threshold"] == 0.9 and policies["P1"]["rules"] == P1_JSON
    assert policies["P5"]["chosen"] == {k: v for k, v in chosen.items() if k != "policy"}
    assert all(policies[name]["label"] for name in policies)
    # the routing curve is the chosen policy's, so the chosen point lies on it (the figures mark it there)
    assert config["curves"]["routing"] == policies["P5"]["curve"] != policies["P0"]["curve"]
    assert {k: v for k, v in chosen.items() if k != "policy"} in config["curves"]["routing"]
    assert [p["successes"] for p in config["curves"]["oracle"]][:3] == [4, 5, 6]
    assert config["adapter_sha256"] == "f2" * 32 and config["dev_sha256"] == sha256_of(paths["dev"])
    assert config["inputs"]["ft_records"]["path"] == "runs/m6/dev-ft2-signals.jsonl"
    # the handwritten items (c, d, e) under the chosen policy's rules: the triggers route c and d at every
    # threshold and e, a text answer, stays local; the threshold alone would route none of them at 0.2
    hand = config["handwritten"]
    assert (hand["n"], hand["policy"], hand["b2_success"]) == (3, "P5", 1.0)
    assert hand["ft_success"] == pytest.approx(1 / 3)
    assert [p["threshold"] for p in hand["scan"]] == [None, 0.3, 0.85, 0.95, None]
    assert [(p["routed"], p["successes"]) for p in hand["scan"]] == [(0, 1), (2, 3), (2, 3), (2, 3), (3, 3)]
    at = hand["at_threshold"]
    assert (at["threshold"], at["routed"], at["successes"]) == (0.2, 2, 3)
    assert at["cost_per_100_rmb"] == pytest.approx(100 * (0.001 + 0.004) / 3)
    assert (config["ft_success"], config["b2_success"]) == (0.5, 0.75)
    assert config["cloud_model"] == "deepseek-flash" and config["dev_set"] == "data/eval/dev.jsonl"


def test_the_share_objective_keeps_the_old_rule_and_the_old_choice(tmp_path):
    paths = fixture(tmp_path)  # the six items of the M4 tests
    out = tmp_path / "routing-m4.json"
    assert route_curve.main(argv(paths, out, "--objective", "share"), root=tmp_path) == 0
    config = json.loads(out.read_text(encoding="utf-8"))
    assert config["rule"] == route_curve.RULES["share"] and config["objective"] == "share"
    assert (config["chosen_policy"], config["rules"], config["threshold"]) == ("P0", None, 0.5)
    assert config["chosen_point"]["policy"] == "P0" and config["chosen_point"]["successes"] == 4
    assert config["policies"]["P0"]["chosen"]["threshold"] == 0.5
    assert config["curves"]["routing"] == config["policies"]["P0"]["curve"]
    assert config["handwritten"]["policy"] == "P0" and config["handwritten"]["at_threshold"]["routed"] == 1
    # on the eight items the old rule is not met, while the cost objective chooses P5 (above)
    paths = policy_fixture(tmp_path / "eight")
    out = tmp_path / "share.json"
    assert route_curve.main(argv(paths, out, "--objective", "share"), root=tmp_path / "eight") == 0
    config = json.loads(out.read_text(encoding="utf-8"))
    assert (config["rule_met"], config["threshold"], config["chosen_policy"]) == (False, None, None)
    assert config["rules"] is None and config["handwritten"]["policy"] is None
    assert config["policies"]["P5"]["chosen"]["threshold"] == 0.2  # recorded, not chosen
    assert config["curves"]["routing"] == config["policies"]["P0"]["curve"]  # nothing chosen: P0's scan


def test_an_ok_record_without_its_output_is_malformed_input(tmp_path, capsys):
    paths = policy_fixture(tmp_path)
    records = [json.loads(line) for line in paths["ft"].read_text(encoding="utf-8").splitlines()]
    del records[2]["output"]  # c, whose call the trigger routes: without it, c would pass as a text turn
    write_jsonl(paths["ft"], records)
    assert route_curve.main(argv(paths, tmp_path / "o.json"), root=tmp_path) == 1
    assert "'c'" in capsys.readouterr().err and not (tmp_path / "o.json").exists()
    records[2]["output"] = {"role": "user", "content": "x"}  # not an assistant turn either
    write_jsonl(paths["ft"], records)
    assert route_curve.main(argv(paths, tmp_path / "o.json"), root=tmp_path) == 1
    b2 = [{"id": rid, "status": "ok", "item": {}, "usage": {"prompt_tokens": 1}} for rid in P_IDS]
    with pytest.raises(route_curve.JsonlError, match="'c'.*output"):
        route_curve.build_rows(records, dict(zip(P_IDS, P_FT, strict=True)), b2,
                               dict(zip(P_IDS, P_B2, strict=True)), PRICE)


def test_an_unknown_objective_is_a_usage_error(tmp_path):
    paths = fixture(tmp_path)
    with pytest.raises(SystemExit):
        route_curve.main(argv(paths, tmp_path / "o.json", "--objective", "accuracy"), root=tmp_path)


FROZEN = REPO / "results" / "routing-m4.json"
M4_THRESHOLD = 0.689404839052033


def recorded_inputs() -> dict[str, Path] | None:
    """The development records results/routing-m4.json was made from, when this machine has them (runs/ is
    not tracked); None otherwise."""
    if not FROZEN.is_file():
        return None
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    inputs = {name: REPO / entry["path"] for name, entry in frozen["inputs"].items()}
    paths = {"dev": REPO / frozen["dev_set"], "ft": inputs["ft_records"], "ft_scores": inputs["ft_scores"],
             "b2": inputs["b2_records"], "b2_scores": inputs["b2_scores"]}
    if not all(p.is_file() for p in [*inputs.values(), *paths.values()]):
        return None
    return paths


def test_on_the_recorded_development_signals_the_policies_reproduce_the_m6_plans_numbers(tmp_path):
    paths = recorded_inputs()
    if paths is None:
        pytest.skip("the development records of results/routing-m4.json are local only (runs/ is untracked)")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    assert all(sha256_of(REPO / entry["path"]) == entry["sha256"] for entry in frozen["inputs"].values())
    out = tmp_path / "routing.json"
    assert route_curve.main(argv(paths, out), root=REPO) == 0
    config = json.loads(out.read_text(encoding="utf-8"))

    def at(name: str, threshold: float) -> dict:
        return next(p for p in config["policies"][name]["curve"] if p["threshold"] == threshold)

    # docs/m6-计划.md §2.3: the frozen point, then P1 and P5 at the same threshold
    p0, p1, p5 = (at(name, M4_THRESHOLD) for name in ("P0", "P1", "P5"))
    assert (p0["routed"], p0["successes"], round(p0["cost_per_100_rmb"], 3)) == (94, 132, 0.323)
    assert (p1["routed"], p1["success"], round(p1["cost_per_100_rmb"], 3)) == (76, 0.88, 0.242)
    assert (p5["routed"], p5["successes"], round(p5["cost_per_100_rmb"], 3)) == (83, 137, 0.258)
    assert round(p5["success"], 3) == 0.913
    # the cost objective: P5 at 88.0% for about 0.181 per 100 items, 52 items routed
    chosen = config["chosen_point"]
    assert config["chosen_policy"] == "P5" and chosen["policy"] == "P5" and chosen["success"] == 0.88
    assert (chosen["routed"], round(chosen["cost_per_100_rmb"], 3)) == (52, 0.181)
    assert round(chosen["threshold"], 3) == 0.488 and config["threshold"] == chosen["threshold"]
    assert config["rules"] == P5_JSON and config["policies"]["P1"]["chosen"]["threshold"] == M4_THRESHOLD
    assert (config["adapter_sha256"], config["dev_sha256"]) == (frozen["adapter_sha256"],
                                                                 frozen["dev_sha256"])
    # the routing curve is the chosen policy's, with the chosen point on it; the handwritten block is under
    # its rules: 47 items, 15 routed, 31 right, about 0.118 per 100 items (the threshold alone: 12, 30)
    assert config["curves"]["routing"] == config["policies"]["P5"]["curve"]
    assert {k: v for k, v in chosen.items() if k != "policy"} in config["curves"]["routing"]
    hand = config["handwritten"]
    assert (hand["n"], hand["policy"], hand["at_threshold"]["threshold"]) == (47, "P5", chosen["threshold"])
    at = hand["at_threshold"]
    assert (at["routed"], at["successes"], round(at["cost_per_100_rmb"], 3)) == (15, 31, 0.118)


def test_on_the_recorded_development_signals_the_share_objective_reproduces_the_frozen_m4_file(tmp_path):
    paths = recorded_inputs()
    if paths is None:
        pytest.skip("the development records of results/routing-m4.json are local only (runs/ is untracked)")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    out = tmp_path / "routing.json"
    assert route_curve.main(argv(paths, out, "--objective", "share"), root=REPO) == 0
    config = json.loads(out.read_text(encoding="utf-8"))
    assert (config["threshold"], config["chosen_policy"]) == (frozen["threshold"], "P0")
    assert config["rules"] is None
    assert {k: v for k, v in config["chosen_point"].items() if k != "policy"} == frozen["chosen_point"]
    assert config["curves"] == frozen["curves"] and config["auroc"] == frozen["auroc"]
    assert {k: v for k, v in config["handwritten"].items() if k != "policy"} == frozen["handwritten"]
    assert config["latency_ms"] == frozen["latency_ms"] and config["inputs"] == frozen["inputs"]
    assert config["rule"] == frozen["rule"] and config["margin"] == frozen["margin"]


def test_a_point_exactly_at_b2_minus_the_margin_meets_the_rule():
    # the real frozen point sits on this boundary (88.0% = 90.0% − 2 pp); the comparison is to 12 decimals
    def point(threshold, routed, cost, success):
        return {"label": "threshold", "threshold": threshold, "routed": routed, "api_share": routed / 50,
                "successes": round(success * 50), "success": success, "cost_per_100_rmb": cost}

    on = point(0.5, 10, 0.2, 44 / 50)
    assert route_curve.best_point([on], 45 / 50, "cost") == on
    assert route_curve.best_point([point(0.5, 10, 0.2, 43.9 / 50)], 45 / 50, "cost") is None
    assert route_curve.best_point([on], 45 / 50, "share") == on


def test_on_the_recorded_development_signals_the_share_objective_is_the_m4_file_byte_for_byte(tmp_path):
    # the share objective's output without the keys M6 added to every objective (the objective, the chosen
    # policy, its rules, the policies' block and the policy beside the chosen and the handwritten point)
    paths = recorded_inputs()
    if paths is None:
        pytest.skip("the development records of results/routing-m4.json are local only (runs/ is untracked)")
    out = tmp_path / "routing.json"
    assert route_curve.main(argv(paths, out, "--objective", "share"), root=REPO) == 0
    config = json.loads(out.read_text(encoding="utf-8"))
    for key in ("objective", "chosen_policy", "rules", "policies"):
        del config[key]
    del config["chosen_point"]["policy"], config["handwritten"]["policy"]
    text = json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    assert text.encode("utf-8") == FROZEN.read_bytes()


# ---- M7: the target objective, the budget trigger and several development sets (docs/m7-计划.md WP-K) ---

P0_BUDGET = {"text_exempt": False, "param_triggers": [], "budget_forced": True}
P1_BUDGET = {**P1_JSON, "budget_forced": True}
P5_BUDGET = {**P5_JSON, "budget_forced": True}


def hand_point(threshold: float | None, routed: int, cost: float, success: float) -> dict:
    return {"label": "threshold", "threshold": threshold, "routed": routed, "api_share": routed / 10,
            "successes": round(success * 10), "success": success, "cost_per_100_rmb": cost}


def test_the_target_objective_takes_the_cheapest_point_reaching_the_target():
    choose = route_curve.target_choice
    cheap, dear = hand_point(0.2, 2, 0.2, 0.9), hand_point(0.5, 5, 0.3, 1.0)
    # the cheapest point at or above the target, however far above it a dearer one goes, on any policy
    assert choose({"P0": [dear], "P1": [cheap]}, 0.9, 1.0) == ("P1", cheap, True)
    # a point below the target is left out however cheap; the target is compared to 12 decimals
    assert choose({"P0": [hand_point(0.1, 1, 0.01, 0.9 - 1e-9), dear]}, 0.9, 1.0) == ("P0", dear, True)
    assert choose({"P0": [hand_point(0.1, 1, 0.01, 0.9 - 1e-13), dear]}, 0.9, 1.0)[1]["threshold"] == 0.1
    # a point that reaches the target is taken over the cap too: the configuration says whether it is within
    over = hand_point(0.3, 4, 0.6, 0.95)
    assert choose({"P0": [over]}, 0.9, 0.4) == ("P0", over, True)
    # the endpoints are no candidates: routing every item is no threshold
    assert choose({"P0": [{**dear, "label": "route_everything", "threshold": None}]}, 0.9, 1.0) is None


def test_target_ties_go_to_the_higher_success_then_fewer_routed_then_the_lower_threshold_then_the_policy():
    def chosen(curves: dict) -> tuple[str, float, int]:
        name, point, reached = route_curve.target_choice(curves, 0.9, 1.0)
        assert reached
        return name, point["threshold"], point["routed"]

    # each pair of points differs in one key only, so each wrong order of the keys fails one assertion: at
    # one cost the higher success, though it routes more and at a higher threshold
    assert chosen({"P0": [hand_point(0.2, 2, 0.2, 0.9), hand_point(0.5, 5, 0.2, 1.0)]}) == ("P0", 0.5, 5)
    # at one cost and success, fewer routed items, though at a higher threshold
    assert chosen({"P0": [hand_point(0.2, 3, 0.2, 0.9), hand_point(0.5, 2, 0.2, 0.9)]}) == ("P0", 0.5, 2)
    # at one cost, success and count, the lower threshold, though on the policy listed second
    assert chosen({"P0": [hand_point(0.5, 2, 0.2, 0.9)], "P1": [hand_point(0.2, 2, 0.2, 0.9)]}) == (
        "P1", 0.2, 2)
    # all of them equal: the policy listed first
    assert chosen({"P0": [hand_point(0.2, 2, 0.2, 0.9)], "P1": [hand_point(0.2, 2, 0.2, 0.9)]}) == (
        "P0", 0.2, 2)


def test_without_a_point_reaching_the_target_the_highest_success_within_the_cap_is_taken_unreached():
    choose = route_curve.target_choice
    over = hand_point(0.6, 6, 0.45, 0.88)  # the best success, over the cap
    best, cheaper = hand_point(0.4, 3, 0.3, 0.85), hand_point(0.2, 1, 0.1, 0.80)
    assert choose({"P0": [cheaper, over], "P1": [best]}, 0.9, 0.4) == ("P1", best, False)
    at_cap = hand_point(0.6, 6, 0.4, 0.88)  # a point at the cap is within it
    assert choose({"P0": [cheaper, at_cap]}, 0.9, 0.4) == ("P0", at_cap, False)
    # ties: the lower cost, though it routes more; then fewer routed, though at a higher threshold; then the
    # lower threshold, though on the policy listed second; then the policy listed first
    assert choose({"P0": [hand_point(0.2, 2, 0.3, 0.85), hand_point(0.4, 3, 0.2, 0.85)]}, 0.9, 0.4)[1][
        "threshold"] == 0.4
    assert choose({"P0": [hand_point(0.2, 3, 0.2, 0.85), hand_point(0.5, 2, 0.2, 0.85)]}, 0.9, 0.4)[1][
        "threshold"] == 0.5
    assert choose({"P0": [hand_point(0.5, 2, 0.2, 0.85)], "P1": [hand_point(0.2, 2, 0.2, 0.85)]}, 0.9,
                  0.4)[0] == "P1"
    assert choose({"P0": [best], "P1": [dict(best)]}, 0.9, 0.4)[0] == "P0"
    # nothing reaches the target and nothing is within the cap: no choice
    assert choose({"P0": [over]}, 0.9, 0.4) is None


def test_the_cost_cap_is_half_of_routing_every_item_to_b2():
    assert route_curve.CAP_SHARE == 0.5
    scan = route_curve.scan(policy_rows(), MDP)
    assert route_curve.cost_cap(scan) == pytest.approx(0.5 * per_100(sum(P_COST)))
    assert route_curve.cost_cap(scan) == 0.5 * scan[-1]["cost_per_100_rmb"]


def test_under_the_target_objective_every_policy_carries_the_budget_trigger_and_the_others_keep_theirs():
    m7 = route_curve.policies_of("target")
    assert list(m7) == ["P0", "P1", "P5"]
    assert [m7[name].rules.to_json() for name in m7] == [P0_BUDGET, P1_BUDGET, P5_BUDGET]
    assert all(m7[name].label.startswith(route_curve.POLICIES[name].label) for name in m7)
    for objective in ("share", "cost"):
        assert route_curve.policies_of(objective) is route_curve.POLICIES


def test_a_step_closed_by_force_is_routed_under_the_target_objectives_policies_only(tmp_path):
    plain = policy_fixture(tmp_path / "plain")
    forced = policy_fixture(tmp_path / "forced", forced=("h",))  # h: FT wrong, B2 right, a call at 0.7
    configs = {}
    for name, paths in (("plain", plain), ("forced", forced)):
        out = tmp_path / f"{name}-cost.json"
        assert route_curve.main(argv(paths, out), root=tmp_path / name) == 0
        configs[name] = json.loads(out.read_text(encoding="utf-8"))
    # the M6 policies have no budget trigger: the step closed by force changes nothing under the cost
    # objective
    assert configs["forced"]["policies"] == configs["plain"]["policies"]
    assert configs["forced"]["chosen_point"] == configs["plain"]["chosen_point"]
    out = tmp_path / "forced-target.json"
    assert route_curve.main(argv(forced, out, "--objective", "target", "--target", "1.0"),
                            root=tmp_path / "forced") == 0
    config = json.loads(out.read_text(encoding="utf-8"))
    curves = {name: [p for p in config["policies"][name]["curve"] if p["label"] == "threshold"]
              for name in config["policies"]}
    # h is routed at every threshold of every policy now: one more routed and one more right below 0.8
    assert [p["routed"] for p in curves["P0"]] == [2, 2, 3, 4, 4, 5, 6, 7]
    assert [p["successes"] for p in curves["P0"]] == [6, 6, 6, 5, 5, 4, 5, 5]
    assert [p["successes"] for p in curves["P1"]] == [6, 6, 6, 5, 5, 5, 6, 6]
    assert [p["successes"] for p in curves["P5"]] == [8, 8, 8, 7, 7, 7, 7, 7]
    assert [config["policies"][name]["rules"] for name in curves] == [P0_BUDGET, P1_BUDGET, P5_BUDGET]
    assert (config["chosen_policy"], config["threshold"], config["target_reached"]) == ("P5", 0.2, True)
    chosen = config["chosen_point"]
    assert (chosen["routed"], chosen["successes"], chosen["policy"]) == (4, 8, "P5")
    assert chosen["cost_per_100_rmb"] == pytest.approx(per_100(0.024))  # c, d, f and h
    assert config["rules"] == P5_BUDGET and config["within_cap"] is True


def test_a_step_whose_reasoning_closed_by_itself_is_not_routed_by_the_budget_trigger(tmp_path):
    # M7's thinking family gives every ok step a thinking record; only a reasoning closed by force is the
    # budget trigger's. Every record made in think mode, h's closed by force and the seven others' by
    # themselves, routes as h's record alone does.
    alone = policy_fixture(tmp_path / "alone", forced=("h",))
    every = policy_fixture(tmp_path / "every", forced=("h",), think=True)
    records = [json.loads(line) for line in every["ft"].read_text(encoding="utf-8").splitlines()]
    assert [r["thinking"]["forced"] for r in records] == [False] * 7 + [True]
    inputs = route_curve.DevInputs(every["dev"], every["ft"], every["ft_scores"], every["b2"],
                                   every["b2_scores"])
    rows = {r["id"]: r for r in route_curve.read_set(inputs, tmp_path / "every").rows}
    assert [rows[rid]["forced"] for rid in P_IDS] == [False] * 7 + [True]
    # a, a sure call (0.9) with the default top_k, stays local under every target policy at 0.2; h (0.7)
    # is routed by the budget trigger alone
    for policy in route_curve.policies_of("target").values():
        assert not route_curve.is_routed(rows["a"], MDP, 0.2, policy.rules)
        assert route_curve.is_routed(rows["h"], MDP, 0.2, policy.rules)
    configs = {}
    for name, paths in (("alone", alone), ("every", every)):
        out = tmp_path / f"{name}.json"
        assert route_curve.main(argv(paths, out, "--objective", "target", "--target", "1.0"),
                                root=tmp_path / name) == 0
        config = json.loads(out.read_text(encoding="utf-8"))
        configs[name] = {k: v for k, v in config.items() if k != "inputs"}  # FT's records differ
    curve = [p for p in configs["every"]["policies"]["P0"]["curve"] if p["label"] == "threshold"]
    assert [p["routed"] for p in curve] == [2, 2, 3, 4, 4, 5, 6, 7]  # h, then the threshold's
    assert configs["every"] == configs["alone"]


def test_the_script_writes_the_target_its_cap_and_whether_the_chosen_point_reached_it(tmp_path, capsys):
    paths = policy_fixture(tmp_path)
    out = tmp_path / "results" / "routing-m7.json"
    # 8/8: no point reaches it; the best success within the cap is P5's 7 at 0.2 (P5's 7 at 0.8 is over it)
    assert route_curve.main(argv(paths, out, "--objective", "target", "--target", "1.0"), root=tmp_path) == 0
    config = json.loads(out.read_text(encoding="utf-8"))
    assert list(config)[:8] == ["rule", "objective", "target", "target_reached", "cost_cap_per_100_rmb",
                                "within_cap", "rule_met", "signal"]
    assert config["rule"] == route_curve.RULES["target"] and "margin" not in config
    assert (config["objective"], config["target"], config["target_reached"]) == ("target", 1.0, False)
    assert config["cost_cap_per_100_rmb"] == pytest.approx(0.5 * per_100(sum(P_COST)))
    assert (config["rule_met"], config["within_cap"]) == (True, True)
    assert (config["chosen_policy"], config["threshold"], config["rules"]) == ("P5", 0.2, P5_BUDGET)
    chosen = config["chosen_point"]
    assert (chosen["routed"], chosen["successes"]) == (3, 7)
    assert chosen["cost_per_100_rmb"] == pytest.approx(per_100(0.021))
    policies = config["policies"]
    assert [policies[name]["target_reached"] for name in policies] == [False, False, False]
    assert policies["P5"]["chosen"] == {k: v for k, v in chosen.items() if k != "policy"}
    assert config["curves"]["routing"] == policies["P5"]["curve"]
    assert config["handwritten"]["policy"] == "P5" and config["dev_set"] == "data/eval/dev.jsonl"
    assert "target not reached" in capsys.readouterr().out
    # 6/8: reached by P5 at 0.2 for 0.021 (and by P1 at 0.9 for 0.043), within the cap
    out = tmp_path / "reached.json"
    assert route_curve.main(argv(paths, out, "--objective", "target", "--target", "0.75"), root=tmp_path) == 0
    config = json.loads(out.read_text(encoding="utf-8"))
    assert (config["target_reached"], config["chosen_policy"], config["threshold"]) == (True, "P5", 0.2)
    assert [config["policies"][name]["target_reached"] for name in config["policies"]] == [False, True, True]
    assert "target reached" in capsys.readouterr().out
    # 4/8: every point reaches it; the cheapest (f alone, 0.016) ties on P0 and P1 at 0.2: P0, listed first
    out = tmp_path / "low.json"
    assert route_curve.main(argv(paths, out, "--objective", "target", "--target", "0.5"), root=tmp_path) == 0
    config = json.loads(out.read_text(encoding="utf-8"))
    assert (config["chosen_policy"], config["threshold"], config["rules"]) == ("P0", 0.2, P0_BUDGET)
    assert (config["chosen_point"]["routed"], config["chosen_point"]["successes"]) == (1, 5)


def test_a_target_no_point_reaches_or_affords_is_written_as_not_met(tmp_path, capsys):
    paths = fixture(tmp_path)  # the six items of the M4 tests
    records = [json.loads(line) for line in paths["ft"].read_text(encoding="utf-8").splitlines()]
    for record in records:  # every step has a hard trigger: every threshold routes every item, at full cost
        record["signals"]["hard_reason"] = HARD[5]
    write_jsonl(paths["ft"], records)
    out = tmp_path / "o.json"
    assert route_curve.main(argv(paths, out, "--objective", "target", "--target", "0.9"), root=tmp_path) == 0
    config = json.loads(out.read_text(encoding="utf-8"))
    assert (config["rule_met"], config["threshold"], config["chosen_policy"]) == (False, None, None)
    assert (config["target_reached"], config["within_cap"], config["chosen_point"]) == (False, None, None)
    assert "no point reaches the target 0.9 and none is within the cap" in capsys.readouterr().out


@pytest.mark.parametrize("extra", [["--objective", "target"], ["--target", "0.9"],
                                   ["--objective", "cost", "--target", "0.9"],
                                   ["--objective", "target", "--target", "0"],
                                   ["--objective", "target", "--target", "1.5"],
                                   ["--objective", "target", "--target", "most"],
                                   ["--cloud", "b3"]])
def test_a_target_without_its_objective_or_out_of_range_or_another_cloud_is_a_usage_error(tmp_path, extra):
    paths = policy_fixture(tmp_path)
    with pytest.raises(SystemExit) as stopped:
        route_curve.main(argv(paths, tmp_path / "o.json", *extra), root=tmp_path)
    assert stopped.value.code == 2 and not (tmp_path / "o.json").exists()


def test_the_cloud_system_is_b2_and_naming_it_changes_nothing(tmp_path):
    paths = policy_fixture(tmp_path)
    plain, named = tmp_path / "plain.json", tmp_path / "named.json"
    assert route_curve.main(argv(paths, plain), root=tmp_path) == 0
    assert route_curve.main(argv(paths, named, "--cloud", "b2"), root=tmp_path) == 0
    assert plain.read_bytes() == named.read_bytes()


# A second development set, data/eval/v2/dev-v2.jsonl: w1 a sure call FT and B2 get right; w2 an unsure call
# FT gets wrong and B2 right; w3 a sure text answer, B2 wrong; w4 a list_notes asking for 20 notes (P5's
# trigger), B2 right. Over the twelve items FT is right on 6, B2 on 9.
W_IDS = ["w1", "w2", "w3", "w4"]
W_FT, W_B2 = [True, False, True, False], [True, True, False, True]
W_MDP, W_COST = [0.4, 0.25, 0.99, 0.65], [0.002, 0.006, 0.003, 0.005]
W_CALLS = [[("get_paper", {"arxiv_id": "2609.07777"})], [("search_arxiv", {"query": "y"})], [],
           [("list_notes", {"limit": 20})]]


def v2_fixture(root: Path, *, ids: list[str] = W_IDS, adapter: str = "f2" * 32) -> dict[str, Path]:
    """The second set and FT's and B2's records over it, under root beside the first set's."""
    (root / "data" / "eval" / "v2").mkdir(parents=True, exist_ok=True)
    run = root / "runs" / "m7"
    run.mkdir(parents=True, exist_ok=True)
    dev = write_jsonl(root / "data" / "eval" / "v2" / "dev-v2.jsonl",
                      [{"id": i, "split": "dev"} for i in ids])
    digest = sha256_of(dev)
    item = {"split": "dev", "source": {"request": "r"}, "type": "call"}
    paths = {"dev": dev,
             "ft": write_jsonl(run / "dev-v2-ft3-signals.jsonl", [
                 {"id": rid, "status": "ok", "item": item, "model": "qwen3-4b:lora=1", "paid": False,
                  "output": output(W_CALLS[i]), "signals": {MDP: W_MDP[i], ML: -0.5, "hard_reason": None}}
                 for i, rid in enumerate(ids)]),
             "ft_scores": write_json(run / "ft3-scores.json", {"scores": [
                 {"id": rid, "success": ok} for rid, ok in zip(ids, W_FT, strict=True)]}),
             "b2": write_jsonl(run / "dev-v2-b2.jsonl", [
                 {"id": rid, "status": "ok", "item": item, "model": "deepseek-flash", "paid": i == 0,
                  "cached": i != 0,
                  "usage": {"prompt_tokens": round(W_COST[i] * 1e6 / 2), "completion_tokens": 0}}
                 for i, rid in enumerate(ids)]),
             "b2_scores": write_json(run / "b2-scores.json", {"scores": [
                 {"id": rid, "success": ok} for rid, ok in zip(ids, W_B2, strict=True)]})}
    write_json(run / "dev-v2-ft3-signals.manifest.json", {
        "system": "ft", "items_sha256": digest,
        "adapters": [{"id": 0, "path": "ft3.gguf", "sha256": adapter}]})
    write_json(run / "dev-v2-b2.manifest.json", {"system": "b2", "items_sha256": digest,
                                                 "model": "deepseek-flash"})
    write_json(run / "dev-v2-ft3-signals.timing.json", {"items": [
        {"id": rid, "calls": [{"wall_ms": 200.0, "error": None, "paid": False, "cached": False}]}
        for rid in ids]})
    write_json(run / "dev-v2-b2.timing.json", {"items": [
        {"id": rid, "calls": [{"wall_ms": 1500.0, "error": None, "paid": i == 0, "cached": i != 0}]}
        for i, rid in enumerate(ids)]})
    return paths


def joint_argv(sets: list[dict[str, Path]], out: Path, *extra: str) -> list[str]:
    """Every set's five inputs in its turn (the repeated options pair up by their order), then the rest."""
    words = []
    for paths in sets:
        words += ["--ft", str(paths["ft"]), "--ft-scores", str(paths["ft_scores"]), "--b2", str(paths["b2"]),
                  "--b2-scores", str(paths["b2_scores"]), "--dev", str(paths["dev"])]
    return [*words, "--out", str(out), *extra]


def b2_manifest(paths: dict[str, Path]) -> Path:
    """The manifest beside B2's records."""
    return paths["b2"].with_name(paths["b2"].stem + ".manifest.json")


def rewrite(path: Path, **fields) -> None:
    """The JSON file at path with fields set."""
    write_json(path, {**json.loads(path.read_text(encoding="utf-8")), **fields})


def earlier_version(root: Path, ids: list[str]) -> str:
    """The SHA-256 of a set as it was before an edit that kept its items' ids (a file of its own under
    root)."""
    return sha256_of(write_jsonl(root / "earlier.jsonl", [{"id": i, "split": "dev", "draft": True}
                                                           for i in ids]))


def test_several_development_sets_are_joined_and_each_named_with_its_inputs(tmp_path, capsys):
    first, second = policy_fixture(tmp_path), v2_fixture(tmp_path)
    out = tmp_path / "results" / "routing-m7.json"
    assert route_curve.main(joint_argv([first, second], out, "--objective", "target", "--target", "0.9",
                                       "--cloud", "b2"), root=tmp_path) == 0
    config = json.loads(out.read_text(encoding="utf-8"))
    assert not {"dev_sha256", "dev_set", "inputs"} & set(config)  # one set's fields: the sets are named below
    names = ["ft_records", "ft_scores", "ft_manifest", "ft_timing", "b2_records", "b2_scores", "b2_manifest",
             "b2_timing"]
    assert [(s["path"], s["sha256"], s["n"], list(s["inputs"])) for s in config["dev_sets"]] == [
        ("data/eval/dev.jsonl", sha256_of(first["dev"]), 8, names),
        ("data/eval/v2/dev-v2.jsonl", sha256_of(second["dev"]), 4, names)]
    assert config["dev_sets"][1]["inputs"]["ft_records"] == {"path": "runs/m7/dev-v2-ft3-signals.jsonl",
                                                             "sha256": sha256_of(second["ft"])}
    assert (config["n"], config["ft_success"], config["b2_success"]) == (12, 0.5, 0.75)
    assert config["auroc"] == {MDP: pytest.approx(21 / 36), ML: pytest.approx(2 / 36)}
    assert (config["adapter_sha256"], config["cloud_model"]) == ("f2" * 32, "deepseek-flash")
    # 11/12 at least: P5 at 0.3 (c, d, f, w2 and w4 routed) for 0.032, the cheapest point reaching it
    assert (config["chosen_policy"], config["threshold"], config["target_reached"]) == ("P5", 0.3, True)
    chosen = config["chosen_point"]
    assert (chosen["routed"], chosen["successes"]) == (5, 11)
    assert chosen["cost_per_100_rmb"] == pytest.approx(100 * 0.032 / 12)
    assert config["cost_cap_per_100_rmb"] == pytest.approx(0.5 * 100 * 0.080 / 12)
    assert config["within_cap"] is True
    # each policy's own: P0 never reaches 11 (its best within the cap, 8 at 0.3); P1 only at 0.99, over
    # the cap
    policies = config["policies"]
    assert [(policies[n]["target_reached"], policies[n]["chosen"]["threshold"]) for n in policies] == [
        (False, 0.3), (True, 0.99), (True, 0.3)]
    # the oracle and the handwritten items (c, d and e, all of the first set) are over the twelve items
    assert len(config["curves"]["oracle"]) == 13
    hand = config["handwritten"]
    assert (hand["n"], hand["at_threshold"]["routed"], hand["at_threshold"]["successes"]) == (3, 2, 3)
    latency = config["latency_ms"]
    assert latency["local_per_call"]["n"] == 12 and latency["cloud_per_paid_call"]["n"] == 1
    assert latency["cloud_per_paid_call"]["p50_ms"] == 1500.0
    assert "2 development sets, 12 items" in capsys.readouterr().out
    # the first set alone, as before: its own fields, no list of sets
    out = tmp_path / "one.json"
    assert route_curve.main(argv(first, out), root=tmp_path) == 0
    one = json.loads(out.read_text(encoding="utf-8"))
    assert "dev_sets" not in one and one["dev_sha256"] == sha256_of(first["dev"]) and one["n"] == 8


def test_the_sets_must_have_their_own_items_one_adapter_and_their_own_manifests(tmp_path, capsys):
    first = policy_fixture(tmp_path / "shared")
    second = v2_fixture(tmp_path / "shared", ids=["w1", "w2", "w3", "a"])  # a is the first set's too
    out = tmp_path / "o.json"
    assert route_curve.main(joint_argv([first, second], out), root=tmp_path / "shared") == 2
    assert "both" in (err := capsys.readouterr().err) and "'a'" in err
    first = policy_fixture(tmp_path / "adapters")
    second = v2_fixture(tmp_path / "adapters", adapter="ab" * 32)
    assert route_curve.main(joint_argv([first, second], out), root=tmp_path / "adapters") == 2
    assert "adapter" in capsys.readouterr().err
    first, second = policy_fixture(tmp_path / "changed"), v2_fixture(tmp_path / "changed")
    write_jsonl(second["dev"], [{"id": "z", "split": "dev"}])  # the second set changed since its runs
    assert route_curve.main(joint_argv([first, second], out), root=tmp_path / "changed") == 2
    assert "dev-v2.jsonl" in (err := capsys.readouterr().err) and "SHA-256" in err
    first, second = policy_fixture(tmp_path / "test"), v2_fixture(tmp_path / "test")
    records = [json.loads(line) for line in second["ft"].read_text(encoding="utf-8").splitlines()]
    records[0]["item"] = {**records[0]["item"], "split": "test"}
    write_jsonl(second["ft"], records)
    assert route_curve.main(joint_argv([first, second], out), root=tmp_path / "test") == 4
    assert not out.exists()


def set_inputs(tmp_path: Path, sets: int) -> tuple[list[dict[str, Path]], Path, list[str]]:
    """The eight items' inputs, alone or with the second set's after them: (the sets' inputs, --out, the
    arguments of a run under the target objective)."""
    first = policy_fixture(tmp_path)
    paths = [first] if sets == 1 else [first, v2_fixture(tmp_path)]
    out = tmp_path / "o.json"
    words = argv(first, out) if sets == 1 else joint_argv(paths, out)
    return paths, out, [*words, "--objective", "target", "--target", "0.9"]


@pytest.mark.parametrize("sets", [1, 2])
def test_a_b2_run_over_another_version_of_its_set_is_refused_beside_fts_over_this_one(tmp_path, capsys,
                                                                                       sets):
    # FT's manifest names the set as it is now; B2's alone names it as it was before an edit that kept its
    # ids, on the only set or on the second of two: B2's answers are not over these items
    paths, out, words = set_inputs(tmp_path, sets)
    last = paths[-1]
    ids = [json.loads(line)["id"] for line in last["dev"].read_text(encoding="utf-8").splitlines()]
    rewrite(b2_manifest(last), items_sha256=earlier_version(tmp_path, ids))
    assert route_curve.main(words, root=tmp_path) == 2
    err = capsys.readouterr().err
    assert f"{b2_manifest(last).as_posix()}: the run's items have SHA-256" in err
    assert f"the set {last['dev'].as_posix()} has SHA-256 {sha256_of(last['dev'])} now" in err
    assert not out.exists()


@pytest.mark.parametrize("sets", [1, 2])
def test_a_b2_run_of_another_cloud_model_is_refused(tmp_path, capsys, sets):
    # the cloud system is b2, deepseek-flash: B2's manifest naming b3's model (deepseek-v4-pro), on the only
    # set or on the second of two, is no run of the cloud system
    paths, out, words = set_inputs(tmp_path, sets)
    other = route_curve.CLOUD_MODELS["b3"]
    rewrite(b2_manifest(paths[-1]), model=other)
    assert route_curve.main(words, root=tmp_path) == 2
    assert f"B2's run used {other!r}, not deepseek-flash" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize("drop", ["--ft-scores", "--b2", "--b2-scores", "--dev"])
def test_each_input_must_be_given_once_per_set_when_there_are_several(tmp_path, drop):
    first, second = policy_fixture(tmp_path), v2_fixture(tmp_path)
    words = joint_argv([first, second], tmp_path / "o.json")
    at = len(words) - 1 - words[::-1].index(drop)  # the second set's
    with pytest.raises(SystemExit) as stopped:
        route_curve.main([*words[:at], *words[at + 2:]], root=tmp_path)
    assert stopped.value.code == 2 and not (tmp_path / "o.json").exists()


def test_a_target_reached_only_over_the_cap_is_chosen_and_recorded_as_over_it(tmp_path, capsys):
    # the six items of the M4 tests: 5 of 6 only by routing c, d and f (0.7 and above), which costs more than
    # half of routing all six; every call is a call without a parameter the triggers read, so the three
    # policies' curves are one and the tie goes to P0
    paths = fixture(tmp_path)
    out = tmp_path / "o.json"
    assert route_curve.main(argv(paths, out, "--objective", "target", "--target", "0.8"), root=tmp_path) == 0
    config = json.loads(out.read_text(encoding="utf-8"))
    assert (config["rule_met"], config["target_reached"], config["within_cap"]) == (True, True, False)
    assert (config["chosen_policy"], config["threshold"], config["rules"]) == ("P0", 0.7, P0_BUDGET)
    chosen = config["chosen_point"]
    assert (chosen["routed"], chosen["successes"]) == (3, 5)
    assert chosen["cost_per_100_rmb"] == pytest.approx(100 * (COSTS[2] + COSTS[3] + COSTS[5]) / 6)
    assert config["cost_cap_per_100_rmb"] == pytest.approx(0.5 * 100 * sum(COSTS) / 6)
    assert config["cost_cap_per_100_rmb"] < chosen["cost_per_100_rmb"]
    out_text = capsys.readouterr().out
    assert "target reached" in out_text and "over the cost cap" in out_text


M6_KEYS = ["rule", "objective", "margin", "rule_met", "signal", "threshold", "chosen_policy", "rules",
           "dev_sha256", "adapter_sha256", "cloud_model", "local_system", "cloud_system", "dev_set", "inputs",
           "n", "ft_success", "b2_success", "auroc", "signal_order", "chosen_point", "policies", "curves",
           "handwritten", "latency_ms"]


def test_the_share_and_cost_objectives_write_the_m6_fields_and_nothing_of_m7(tmp_path):
    # a step closed by force (h) changes nothing under M6's policies, which have no budget trigger
    plain, forced = policy_fixture(tmp_path / "plain"), policy_fixture(tmp_path / "forced", forced=("h",))
    for objective in ("share", "cost"):
        written = []
        for name, paths in (("plain", plain), ("forced", forced)):
            out = tmp_path / f"{name}-{objective}.json"
            assert route_curve.main(argv(paths, out, "--objective", objective), root=tmp_path / name) == 0
            config = json.loads(out.read_text(encoding="utf-8"))
            assert list(config) == M6_KEYS, objective
            keys = [list(p) for p in config["policies"].values()]
            assert keys == [["label", "rules", "curve", "chosen"]] * 3
            assert "budget" not in json.dumps(config["policies"])
            assert config["rules"] in (None, P1_JSON, P5_JSON)
            written.append({k: v for k, v in config.items() if k not in ("inputs", "dev_sha256")})
        assert written[0] == written[1]


FROZEN_M6 = REPO / "results" / "routing-m6.json"


def recorded_m6_inputs() -> dict[str, Path] | None:
    """The development records results/routing-m6.json was chosen from, when this machine has them (runs/ is
    not tracked); None otherwise."""
    if not FROZEN_M6.is_file():
        return None
    frozen = json.loads(FROZEN_M6.read_text(encoding="utf-8"))
    inputs = {name: REPO / entry["path"] for name, entry in frozen["inputs"].items()}
    paths = {"dev": REPO / frozen["dev_set"], "ft": inputs["ft_records"], "ft_scores": inputs["ft_scores"],
             "b2": inputs["b2_records"], "b2_scores": inputs["b2_scores"]}
    if not all(p.is_file() for p in [*inputs.values(), *paths.values()]):
        return None
    return paths


def test_on_the_recorded_development_signals_the_cost_objective_is_the_frozen_m6_file_byte_for_byte(tmp_path):
    paths = recorded_m6_inputs()
    if paths is None:
        pytest.skip("the development records of results/routing-m6.json are local only (runs/ is untracked)")
    out = tmp_path / "routing.json"
    assert route_curve.main(argv(paths, out), root=REPO) == 0  # cost is the default objective
    frozen = json.loads(FROZEN_M6.read_text(encoding="utf-8"))
    del frozen["validation"]  # scripts/route_validate.py's block, added to the file afterwards
    assert out.read_bytes() == (json.dumps(frozen, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


# ---- M7: FT's generation, frozen with the routing (docs/m7-计划.md §2 解码, WP-K and WP-M) -----------------

# FT's generation as scripts/run_eval.py records it in a manifest on the family path (llm.generation.
# GenerationSettings.record(), its keys sorted as canonical JSON writes them): think mode decoded greedily,
# and the family's sampling with base seed 0. A run on the legacy path, as every run of M3 to M6, records
# none.
THINK_GREEDY = {"answer_tokens": 512, "base_seed": 0,
                "decoding": {"min_p": 0.0, "presence_penalty": 0.0, "temperature": 0.0, "top_k": 0,
                             "top_p": 1.0},
                "mode": "think", "reasoning_budget": 1280, "sample": 0}
THINK_SAMPLED = {**THINK_GREEDY, "decoding": {"min_p": 0.0, "presence_penalty": 0.0, "temperature": 0.6,
                                              "top_k": 20, "top_p": 0.95}}


def ft_manifest(paths: dict[str, Path]) -> Path:
    """The manifest beside FT's records."""
    return paths["ft"].with_name(paths["ft"].stem + ".manifest.json")


def target_words(sets: list[dict[str, Path]], out: Path) -> list[str]:
    return joint_argv(sets, out, "--objective", "target", "--target", "0.9")


def test_fts_generation_is_frozen_after_its_adapter_and_changes_nothing_else(tmp_path):
    first, second = policy_fixture(tmp_path), v2_fixture(tmp_path)
    for paths in (first, second):
        rewrite(ft_manifest(paths), generation=THINK_GREEDY)
    out = tmp_path / "results" / "routing-m7.json"
    assert route_curve.main(target_words([first, second], out), root=tmp_path) == 0
    config = json.loads(out.read_text(encoding="utf-8"))
    keys = list(config)
    assert config["ft_generation"] == THINK_GREEDY
    assert keys[keys.index("adapter_sha256") + 1] == "ft_generation"
    one = tmp_path / "one.json"
    assert route_curve.main(argv(first, one), root=tmp_path) == 0  # one set, the cost objective
    assert json.loads(one.read_text(encoding="utf-8"))["ft_generation"] == THINK_GREEDY
    # the same runs without it: no field, and the same choice, curves and inputs but for the manifests
    for paths in (first, second):
        manifest = json.loads(ft_manifest(paths).read_text(encoding="utf-8"))
        del manifest["generation"]
        write_json(ft_manifest(paths), manifest)
    plain = tmp_path / "plain.json"
    assert route_curve.main(target_words([first, second], plain), root=tmp_path) == 0
    without = json.loads(plain.read_text(encoding="utf-8"))
    assert "ft_generation" not in without
    assert {k: v for k, v in config.items() if k not in ("ft_generation", "dev_sets")} == {
        k: v for k, v in without.items() if k != "dev_sets"}


def test_fts_runs_over_the_sets_must_share_one_generation_or_record_none(tmp_path, capsys):
    first, second = policy_fixture(tmp_path), v2_fixture(tmp_path)
    out = tmp_path / "o.json"
    rewrite(ft_manifest(second), generation=THINK_GREEDY)  # the first set's run recorded none: a mix
    assert route_curve.main(target_words([first, second], out), root=tmp_path) == 2
    err = capsys.readouterr().err
    assert "generation" in err and f"{first['dev'].as_posix()} none" in err
    rewrite(ft_manifest(first), generation=THINK_GREEDY)  # the other way round: a later set's recorded none
    manifest = json.loads(ft_manifest(second).read_text(encoding="utf-8"))
    del manifest["generation"]
    write_json(ft_manifest(second), manifest)
    assert route_curve.main(target_words([first, second], out), root=tmp_path) == 2
    err = capsys.readouterr().err
    assert "generation" in err and f"{second['dev'].as_posix()} none" in err
    rewrite(ft_manifest(second), generation=THINK_GREEDY)
    rewrite(ft_manifest(first), generation=THINK_SAMPLED)  # two generations
    assert route_curve.main(target_words([first, second], out), root=tmp_path) == 2
    assert '"temperature":0.6' in capsys.readouterr().err
    rewrite(ft_manifest(first), generation="think")  # not the object run_eval records
    assert route_curve.main(target_words([first, second], out), root=tmp_path) == 1
    assert "generation" in capsys.readouterr().err and not out.exists()


# Computed with scripts/route_curve.py and research_agent of commit 64250c0 (before FT's generation was
# frozen) on the fixtures above, without a generation in any manifest: the SHA-256 of what it writes over the
# eight items under the cost objective, over the six under the share objective and over both sets under the
# target objective
CURVE_GOLDEN = {"cost": "2cb7c20d932ff72d79b914fdd893a1b6ea6e3811a396464518cd574d9a7ca4e7",
                "share": "48240352231ba3161a07bf97955b4496d308e03a65db3812b90ed98acc040578",
                "target": "13275d0b749f9f1975f1452ba5e0b3ff7b6b598ae0d653fb61e9822ebeb29169"}


def test_without_fts_generation_every_objective_writes_byte_for_byte_what_it_wrote_before(tmp_path):
    runs = {"cost": (tmp_path / "cost", lambda root, out: argv(policy_fixture(root), out)),
            "share": (tmp_path / "share", lambda root, out: argv(fixture(root), out, "--objective", "share")),
            "target": (tmp_path / "target",
                       lambda root, out: target_words([policy_fixture(root), v2_fixture(root)], out))}
    got = {}
    for name, (root, words) in runs.items():
        out = root / "results" / "routing.json"
        assert route_curve.main(words(root, out), root=root) == 0
        got[name] = sha256_of(out)
    assert got == CURVE_GOLDEN

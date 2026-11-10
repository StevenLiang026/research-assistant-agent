"""scripts/route_validate.py: a frozen routing configuration's policy evaluated on another set's recorded
signals (the development extension set, docs/m6-计划.md §3.3 C3), or on several sets jointly (M7: dev-ext and
dev-v2-ext, with the cost cap of a target configuration), written as a validation block without re-choosing
anything, offline over fabricated records; and results/routing-m6.json rebuilt byte for byte where this
machine has its recorded runs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_route_curve import (
    FROZEN_M6,
    P5_JSON,
    REPO,
    THINK_GREEDY,
    THINK_SAMPLED,
    argv,
    b2_manifest,
    earlier_version,
    ft_manifest,
    load_script,
    output,
    policy_fixture,
    recorded_m6_inputs,
    rewrite,
    route_curve,
    thought,
    write_json,
    write_jsonl,
)

from research_agent.eval.lock import sha256_of

route_validate = load_script("route_validate")
MDP, ML = "min_decision_prob", "mean_logprob"

# The other set: x1 is a confident text answer (never routed under the frozen P5), x2 asks search_notes for
# 5 notes (the trigger routes it, B2 is right), x3 is an unsure call (the threshold routes it, B2 is wrong
# too), x4 a confident list_notes with the default limit (local, FT right). FT 2/4, B2 2/4, FT+R 3/4.
V_IDS = ["x1", "x2", "x3", "x4"]
V_FT = [True, False, False, True]
V_B2 = [True, True, False, False]
V_MDP = [0.9, 0.5, 0.1, 0.95]
V_COST = [0.002, 0.004, 0.008, 0.016]
V_CALLS = [[], [("search_notes", {"query": "x", "top_k": 5})], [("get_paper", {"arxiv_id": "2609.01234"})],
           [("list_notes", {"limit": 10})]]
EXPECTED = {"n": 4, "policy": "P5", "signal": MDP, "threshold": 0.2, "rules": P5_JSON, "routed": 2,
            "api_share": 0.5, "successes": 3, "success": 0.75, "ft_success": 0.5, "b2_success": 0.5,
            "random_success": 0.5, "adapter_sha256": "f2" * 32, "cloud_model": "deepseek-flash"}


def validation_fixture(root: Path, *, adapter: str = "f2" * 32, split: str = "dev", think: bool = False,
                       forced: tuple[str, ...] = ()) -> dict[str, Path]:
    """The second set and FT's and B2's records over it, under root beside the frozen file's inputs; FT's
    records in think mode as test_route_curve.thought makes them (think, forced)."""
    (root / "data" / "eval").mkdir(parents=True, exist_ok=True)
    (root / "runs" / "m6").mkdir(parents=True, exist_ok=True)
    ext = write_jsonl(root / "data" / "eval" / "dev-ext.jsonl", [{"id": i, "split": split} for i in V_IDS])
    digest = sha256_of(ext)
    run = root / "runs" / "m6"
    item = {"split": split, "source": {"request": "r"}, "type": "call"}
    paths = {"set": ext,
             "ft": write_jsonl(run / "dev-ext-ft2-signals.jsonl", [
                 {"id": rid, "status": "ok", "item": item, "model": "qwen3-1.7b:lora=1", "paid": False,
                  "output": output(V_CALLS[i]), "signals": {MDP: V_MDP[i], ML: -0.1, "hard_reason": None},
                  **thought(rid, forced=forced, think=think)}
                 for i, rid in enumerate(V_IDS)]),
             "ft_scores": write_json(run / "ext-ft2-scores.json", {"scores": [
                 {"id": rid, "success": ok} for rid, ok in zip(V_IDS, V_FT, strict=True)]}),
             "b2": write_jsonl(run / "dev-ext-b2.jsonl", [
                 {"id": rid, "status": "ok", "item": item, "model": "deepseek-flash", "paid": False,
                  "cached": True,
                  "usage": {"prompt_tokens": round(V_COST[i] * 1e6 / 2), "completion_tokens": 0}}
                 for i, rid in enumerate(V_IDS)]),
             "b2_scores": write_json(run / "ext-b2-scores.json", {"scores": [
                 {"id": rid, "success": ok} for rid, ok in zip(V_IDS, V_B2, strict=True)]})}
    write_json(run / "dev-ext-ft2-signals.manifest.json", {
        "system": "ft", "items_sha256": digest,
        "adapters": [{"id": 0, "path": "ft2.gguf", "sha256": adapter}]})
    write_json(run / "dev-ext-b2.manifest.json", {"system": "b2", "items_sha256": digest,
                                                  "model": "deepseek-flash"})
    return paths


def frozen(root: Path) -> Path:
    """A frozen configuration over the eight items: P5 at 0.2 under the cost objective."""
    out = root / "results" / "routing-m6.json"
    assert route_curve.main(argv(policy_fixture(root), out), root=root) == 0
    return out


def vargs(routing: Path, paths: dict[str, Path], out: Path, *extra: str) -> list[str]:
    return ["--routing", str(routing), "--ft", str(paths["ft"]), "--ft-scores", str(paths["ft_scores"]),
            "--b2", str(paths["b2"]), "--b2-scores", str(paths["b2_scores"]), "--set", str(paths["set"]),
            "--out", str(out), *extra]


def test_the_block_evaluates_the_frozen_policy_on_the_other_set_without_re_choosing(tmp_path):
    routing = frozen(tmp_path)
    before = json.loads(routing.read_text(encoding="utf-8"))
    paths = validation_fixture(tmp_path)
    assert route_validate.main(vargs(routing, paths, routing), root=tmp_path) == 0
    after = json.loads(routing.read_text(encoding="utf-8"))
    block = after.pop("validation")["dev-ext"]
    assert after == before  # nothing else changed: the choice stands
    assert {k: block[k] for k in EXPECTED} == EXPECTED
    assert block["cost_per_100_rmb"] == pytest.approx(100 * (0.004 + 0.008) / 4)
    assert block["set"] == "data/eval/dev-ext.jsonl" and block["set_sha256"] == sha256_of(paths["set"])
    assert block["inputs"]["ft_records"] == {"path": "runs/m6/dev-ext-ft2-signals.jsonl",
                                             "sha256": sha256_of(paths["ft"])}
    assert set(block["inputs"]) == {"ft_records", "ft_scores", "ft_manifest", "b2_records", "b2_scores",
                                    "b2_manifest"}
    raw = routing.read_bytes()
    assert b"\r\n" not in raw and raw.endswith(b"}\n")


def test_a_separate_output_names_the_routing_file_it_validates(tmp_path):
    routing = frozen(tmp_path)
    paths = validation_fixture(tmp_path)
    out = tmp_path / "results" / "routing-m6-dev-ext.json"
    assert route_validate.main(vargs(routing, paths, out), root=tmp_path) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["routing"] == {"path": "results/routing-m6.json", "sha256": sha256_of(routing)}
    assert data["validation"]["dev-ext"]["successes"] == 3
    assert "validation" not in json.loads(routing.read_text(encoding="utf-8"))


def test_an_existing_block_or_file_is_kept_unless_forced(tmp_path, capsys):
    routing = frozen(tmp_path)
    paths = validation_fixture(tmp_path)
    assert route_validate.main(vargs(routing, paths, routing), root=tmp_path) == 0
    assert route_validate.main(vargs(routing, paths, routing), root=tmp_path) == 3
    assert "--force" in capsys.readouterr().err
    assert route_validate.main(vargs(routing, paths, routing, "--force"), root=tmp_path) == 0
    assert list(json.loads(routing.read_text(encoding="utf-8"))["validation"]) == ["dev-ext"]
    out = tmp_path / "o.json"
    out.write_text("{}\n", encoding="utf-8")
    assert route_validate.main(vargs(routing, paths, out), root=tmp_path) == 3
    assert out.read_text(encoding="utf-8") == "{}\n"


def test_records_of_another_adapter_or_another_set_are_refused(tmp_path, capsys):
    routing = frozen(tmp_path)
    paths = validation_fixture(tmp_path, adapter="ab" * 32)
    assert route_validate.main(vargs(routing, paths, routing), root=tmp_path) == 2
    assert "adapter" in capsys.readouterr().err
    paths = validation_fixture(tmp_path)
    write_jsonl(paths["set"], [{"id": "z", "split": "dev"}])  # the set changed since the runs
    assert route_validate.main(vargs(routing, paths, routing), root=tmp_path) == 2
    assert "SHA-256" in capsys.readouterr().err
    assert "validation" not in json.loads(routing.read_text(encoding="utf-8"))


def test_an_ok_record_without_its_output_is_malformed_input(tmp_path, capsys):
    routing = frozen(tmp_path)
    paths = validation_fixture(tmp_path)
    records = [json.loads(line) for line in paths["ft"].read_text(encoding="utf-8").splitlines()]
    del records[1]["output"]  # x2, whose call the trigger routes: without it, x2 would pass as a text turn
    write_jsonl(paths["ft"], records)
    assert route_validate.main(vargs(routing, paths, routing), root=tmp_path) == 1
    assert "'x2'" in capsys.readouterr().err
    assert "validation" not in json.loads(routing.read_text(encoding="utf-8"))


def test_a_configuration_whose_rule_was_not_met_has_nothing_to_validate(tmp_path, capsys):
    paths = policy_fixture(tmp_path)
    routing = tmp_path / "results" / "routing.json"
    assert route_curve.main(argv(paths, routing, "--objective", "share"), root=tmp_path) == 0  # not met
    assert route_validate.main(vargs(routing, validation_fixture(tmp_path), routing), root=tmp_path) == 2
    assert "threshold" in capsys.readouterr().err


def test_test_material_is_refused(tmp_path):
    routing = frozen(tmp_path)
    paths = validation_fixture(tmp_path, split="test")
    assert route_validate.main(vargs(routing, paths, routing), root=tmp_path) == 4
    assert "validation" not in json.loads(routing.read_text(encoding="utf-8"))


def test_validations_of_two_sets_are_both_kept_in_the_routing_file(tmp_path):
    # dev and dev-ext each get a block; the second validation must not drop the first
    routing = frozen(tmp_path)
    paths = validation_fixture(tmp_path)
    assert route_validate.main(vargs(routing, paths, routing), root=tmp_path) == 0
    second = paths["set"].with_name("dev-ext2.jsonl")
    second.write_bytes(paths["set"].read_bytes())  # the same items under another name: the manifests fit
    assert route_validate.main(vargs(routing, {**paths, "set": second}, routing), root=tmp_path) == 0
    blocks = json.loads(routing.read_text(encoding="utf-8"))["validation"]
    assert set(blocks) == {"dev-ext", "dev-ext2"}
    assert blocks["dev-ext"]["success"] == blocks["dev-ext2"]["success"]


# ---- M7: several sets validated jointly, dev-ext with dev-v2-ext (docs/m7-计划.md §5, WP-K) -------------

# The second extension set, data/eval/v2/dev-v2-ext.jsonl: y1 a confident call FT and B2 get right (local);
# y2 a list_notes asking for 20 notes (P5's trigger routes it, B2 is right); y3 an unsure call (the threshold
# routes it, B2 is wrong too). With the first set's four: FT 3/7, B2 4/7, the frozen P5 at 0.2 5/7, 4 routed.
Y_IDS = ["y1", "y2", "y3"]
Y_FT, Y_B2 = [True, False, False], [True, True, False]
Y_MDP, Y_COST = [0.9, 0.6, 0.1], [0.002, 0.006, 0.004]
Y_CALLS = [[("get_paper", {"arxiv_id": "2609.07777"})], [("list_notes", {"limit": 20})],
           [("search_arxiv", {"query": "y"})]]
SIX = ["ft_records", "ft_scores", "ft_manifest", "b2_records", "b2_scores", "b2_manifest"]


def v2_validation_fixture(root: Path, *, ids: list[str] = Y_IDS, adapter: str = "f2" * 32,
                          split: str = "dev", think: bool = False) -> dict[str, Path]:
    """The second extension set and FT's and B2's records over it, under root beside the first's; with think
    FT's run was made in think mode, every reasoning closed by itself (test_route_curve.thought)."""
    (root / "data" / "eval" / "v2").mkdir(parents=True, exist_ok=True)
    run = root / "runs" / "m7"
    run.mkdir(parents=True, exist_ok=True)
    ext = write_jsonl(root / "data" / "eval" / "v2" / "dev-v2-ext.jsonl",
                      [{"id": i, "split": "dev"} for i in ids])
    digest = sha256_of(ext)
    item = {"split": split, "source": {"request": "r"}, "type": "call"}
    paths = {"set": ext,
             "ft": write_jsonl(run / "dev-v2-ext-ft3-signals.jsonl", [
                 {"id": rid, "status": "ok", "item": item, "model": "qwen3-4b:lora=1", "paid": False,
                  "output": output(Y_CALLS[i]), "signals": {MDP: Y_MDP[i], ML: -0.2, "hard_reason": None},
                  **thought(rid, think=think)}
                 for i, rid in enumerate(ids)]),
             "ft_scores": write_json(run / "ext-v2-ft3-scores.json", {"scores": [
                 {"id": rid, "success": ok} for rid, ok in zip(ids, Y_FT, strict=True)]}),
             "b2": write_jsonl(run / "dev-v2-ext-b2.jsonl", [
                 {"id": rid, "status": "ok", "item": item, "model": "deepseek-flash", "paid": False,
                  "cached": True,
                  "usage": {"prompt_tokens": round(Y_COST[i] * 1e6 / 2), "completion_tokens": 0}}
                 for i, rid in enumerate(ids)]),
             "b2_scores": write_json(run / "ext-v2-b2-scores.json", {"scores": [
                 {"id": rid, "success": ok} for rid, ok in zip(ids, Y_B2, strict=True)]})}
    write_json(run / "dev-v2-ext-ft3-signals.manifest.json", {
        "system": "ft", "items_sha256": digest,
        "adapters": [{"id": 0, "path": "ft3.gguf", "sha256": adapter}]})
    write_json(run / "dev-v2-ext-b2.manifest.json", {"system": "b2", "items_sha256": digest,
                                                     "model": "deepseek-flash"})
    return paths


def joint_vargs(routing: Path, sets: list[dict[str, Path]], out: Path, *extra: str) -> list[str]:
    """Every set's five inputs in its turn (the repeated options pair up by their order), then the rest."""
    words = ["--routing", str(routing)]
    for paths in sets:
        words += ["--ft", str(paths["ft"]), "--ft-scores", str(paths["ft_scores"]), "--b2", str(paths["b2"]),
                  "--b2-scores", str(paths["b2_scores"]), "--set", str(paths["set"])]
    return [*words, "--out", str(out), *extra]


def test_two_sets_are_validated_jointly_in_one_block_named_after_both(tmp_path, capsys):
    routing = frozen(tmp_path)
    before = json.loads(routing.read_text(encoding="utf-8"))
    first, second = validation_fixture(tmp_path), v2_validation_fixture(tmp_path)
    assert route_validate.main(joint_vargs(routing, [first, second], routing), root=tmp_path) == 0
    after = json.loads(routing.read_text(encoding="utf-8"))
    blocks = after.pop("validation")
    assert after == before and list(blocks) == ["dev-ext+dev-v2-ext"]
    block = blocks["dev-ext+dev-v2-ext"]
    assert list(block) == ["n", "policy", "signal", "threshold", "rules", "routed", "api_share", "successes",
                           "success", "cost_per_100_rmb", "random_success", "ft_success", "b2_success",
                           "sets", "adapter_sha256", "cloud_model"]
    assert (block["n"], block["policy"], block["threshold"], block["rules"]) == (7, "P5", 0.2, P5_JSON)
    assert (block["routed"], block["successes"]) == (4, 5)
    assert block["success"] == pytest.approx(5 / 7) and block["api_share"] == pytest.approx(4 / 7)
    assert (block["ft_success"], block["b2_success"]) == (pytest.approx(3 / 7), pytest.approx(4 / 7))
    assert block["cost_per_100_rmb"] == pytest.approx(100 * (0.004 + 0.008 + 0.006 + 0.004) / 7)
    assert block["random_success"] == pytest.approx(3 / 7 * 3 / 7 + 4 / 7 * 4 / 7)
    assert [(s["path"], s["sha256"], s["n"], list(s["inputs"])) for s in block["sets"]] == [
        ("data/eval/dev-ext.jsonl", sha256_of(first["set"]), 4, SIX),
        ("data/eval/v2/dev-v2-ext.jsonl", sha256_of(second["set"]), 3, SIX)]
    assert block["sets"][1]["inputs"]["ft_records"] == {"path": "runs/m7/dev-v2-ext-ft3-signals.jsonl",
                                                        "sha256": sha256_of(second["ft"])}
    assert (block["adapter_sha256"], block["cloud_model"]) == ("f2" * 32, "deepseek-flash")
    assert "on dev-ext+dev-v2-ext (7 items)" in capsys.readouterr().out
    # one set alone is validated as before, under its own name, beside the joint block
    assert route_validate.main(vargs(routing, first, routing), root=tmp_path) == 0
    blocks = json.loads(routing.read_text(encoding="utf-8"))["validation"]
    assert list(blocks) == ["dev-ext+dev-v2-ext", "dev-ext"] and "sets" not in blocks["dev-ext"]
    assert blocks["dev-ext"]["set"] == "data/eval/dev-ext.jsonl" and blocks["dev-ext"]["n"] == 4
    # a file of its own holds the joint block too
    out = tmp_path / "results" / "routing-m7-ext.json"
    assert route_validate.main(joint_vargs(routing, [first, second], out), root=tmp_path) == 0
    assert list(json.loads(out.read_text(encoding="utf-8"))["validation"]) == ["dev-ext+dev-v2-ext"]


def test_jointly_validated_sets_must_hold_their_own_items_over_the_frozen_adapter(tmp_path, capsys):
    routing = frozen(tmp_path)
    first = validation_fixture(tmp_path)
    shared = v2_validation_fixture(tmp_path, ids=["y1", "y2", "x3"])  # x3 is the first set's too
    assert route_validate.main(joint_vargs(routing, [first, shared], routing), root=tmp_path) == 2
    assert "both" in (err := capsys.readouterr().err) and "'x3'" in err
    other = v2_validation_fixture(tmp_path, adapter="ab" * 32)
    assert route_validate.main(joint_vargs(routing, [first, other], routing), root=tmp_path) == 2
    assert "adapter" in capsys.readouterr().err
    test = v2_validation_fixture(tmp_path, split="test")
    assert route_validate.main(joint_vargs(routing, [first, test], routing), root=tmp_path) == 4
    assert "validation" not in json.loads(routing.read_text(encoding="utf-8"))


@pytest.mark.parametrize("sets", [1, 2])
def test_a_b2_run_over_another_version_of_its_set_or_of_another_cloud_model_is_refused(tmp_path, capsys,
                                                                                       sets):
    # FT's records over the set are the right ones; B2's manifest alone is not, on the only set or on the
    # second of two: it names the set as it was before an edit that kept its ids, or a model the routing
    # file does not route to (b3's)
    routing = frozen(tmp_path)
    first = validation_fixture(tmp_path)
    paths = [first] if sets == 1 else [first, v2_validation_fixture(tmp_path)]
    last, manifest = paths[-1], b2_manifest(paths[-1])
    recorded = json.loads(manifest.read_text(encoding="utf-8"))
    ids = [json.loads(line)["id"] for line in last["set"].read_text(encoding="utf-8").splitlines()]
    words = vargs(routing, first, routing) if sets == 1 else joint_vargs(routing, paths, routing)
    rewrite(manifest, items_sha256=earlier_version(tmp_path, ids))
    assert route_validate.main(words, root=tmp_path) == 2
    err = capsys.readouterr().err
    assert f"{manifest.as_posix()}: the run's items have SHA-256" in err
    assert f"the set {last['set'].as_posix()} has SHA-256 {sha256_of(last['set'])} now" in err
    write_json(manifest, {**recorded, "model": route_curve.CLOUD_MODELS["b3"]})
    assert route_validate.main(words, root=tmp_path) == 2
    assert ("B2's run used 'deepseek-v4-pro', the routing file routes to 'deepseek-flash'"
            in capsys.readouterr().err)
    assert "validation" not in json.loads(routing.read_text(encoding="utf-8"))


@pytest.mark.parametrize("drop", ["--ft", "--ft-scores", "--b2", "--b2-scores", "--set"])
def test_each_input_is_given_once_per_validated_set(tmp_path, drop):
    routing = frozen(tmp_path)
    words = joint_vargs(routing, [validation_fixture(tmp_path), v2_validation_fixture(tmp_path)], routing)
    at = len(words) - 1 - words[::-1].index(drop)  # the second set's
    with pytest.raises(SystemExit) as stopped:
        route_validate.main([*words[:at], *words[at + 2:]], root=tmp_path)
    assert stopped.value.code == 2 and "validation" not in json.loads(routing.read_text(encoding="utf-8"))


def test_a_target_files_block_says_whether_its_cost_is_within_the_cap_on_the_validated_sets(tmp_path):
    # docs/m7-计划.md G3: on dev-ext and dev-v2-ext too the cost must be within half of routing every item
    routing = tmp_path / "results" / "routing-m7.json"
    words = argv(policy_fixture(tmp_path), routing, "--objective", "target", "--target", "0.75")
    assert route_curve.main(words, root=tmp_path) == 0  # P5 with the budget trigger, at 0.2
    # FT's runs over both sets were made in think mode, every reasoning closed by itself: the budget trigger
    # routes none of the seven
    first, second = validation_fixture(tmp_path, think=True), v2_validation_fixture(tmp_path, think=True)
    assert route_validate.main(joint_vargs(routing, [first, second], routing), root=tmp_path) == 0
    block = json.loads(routing.read_text(encoding="utf-8"))["validation"]["dev-ext+dev-v2-ext"]
    assert (block["routed"], block["successes"], block["rules"]["budget_forced"]) == (4, 5, True)
    cap = 0.5 * 100 * (sum(V_COST) + sum(Y_COST)) / 7  # 0.300: x2, x3, y2 and y3 cost 0.314 per 100
    assert block["cost_cap_per_100_rmb"] == pytest.approx(cap) and block["within_cap"] is False
    assert list(block)[13:15] == ["cost_cap_per_100_rmb", "within_cap"]  # after b2_success
    assert route_validate.main(vargs(routing, first, routing), root=tmp_path) == 0
    alone = json.loads(routing.read_text(encoding="utf-8"))["validation"]["dev-ext"]
    assert alone["cost_cap_per_100_rmb"] == pytest.approx(0.5 * 100 * sum(V_COST) / 4)
    assert alone["within_cap"] is True  # x2 and x3: 0.300 of 0.375
    # a reasoning closed by force is the budget trigger's, however sure the step: x4 (0.95, a list_notes with
    # the default limit, FT right and B2 wrong) is routed too
    forced = validation_fixture(tmp_path, think=True, forced=("x4",))
    out = tmp_path / "forced.json"
    assert route_validate.main(joint_vargs(routing, [forced, second], out), root=tmp_path) == 0
    block = json.loads(out.read_text(encoding="utf-8"))["validation"]["dev-ext+dev-v2-ext"]
    assert (block["routed"], block["successes"]) == (5, 4)
    assert block["cost_per_100_rmb"] == pytest.approx(100 * (0.004 + 0.008 + 0.016 + 0.006 + 0.004) / 7)
    # the cost objective's block is as M6 wrote it, without a cap (results/routing-m6.json)
    m6 = frozen(tmp_path / "m6")
    assert route_validate.main(vargs(m6, validation_fixture(tmp_path / "m6"), m6), root=tmp_path / "m6") == 0
    assert "within_cap" not in json.loads(m6.read_text(encoding="utf-8"))["validation"]["dev-ext"]


def test_on_the_recorded_signals_the_frozen_m6_file_is_rebuilt_byte_for_byte(tmp_path):
    # results/routing-m6.json: scripts/route_curve.py under the cost objective, then this script's block over
    # dev-ext added to the same file
    paths = recorded_m6_inputs()
    frozen_m6 = json.loads(FROZEN_M6.read_text(encoding="utf-8")) if FROZEN_M6.is_file() else {}
    block = (frozen_m6.get("validation") or {}).get("dev-ext") or {}
    ext = {name: REPO / entry["path"] for name, entry in (block.get("inputs") or {}).items()}
    if paths is None or not ext or not all(p.is_file() for p in [*ext.values(), REPO / block["set"]]):
        pytest.skip("the development records of results/routing-m6.json are local only (runs/ is untracked)")
    out = tmp_path / "routing-m6.json"
    assert route_curve.main(argv(paths, out), root=REPO) == 0
    ext_paths = {"set": REPO / block["set"], "ft": ext["ft_records"], "ft_scores": ext["ft_scores"],
                 "b2": ext["b2_records"], "b2_scores": ext["b2_scores"]}
    assert route_validate.main(vargs(out, ext_paths, out), root=REPO) == 0
    assert out.read_bytes() == FROZEN_M6.read_bytes()


# ---- M7: FT's generation, the routing file's (docs/m7-计划.md §2 解码, WP-K) -------------------------------


def frozen_with(root: Path, generation: dict) -> Path:
    """A frozen configuration over the eight items (P5 at 0.2, the cost objective), chosen on FT's runs of the
    given generation: its ft_generation."""
    paths = policy_fixture(root)
    rewrite(ft_manifest(paths), generation=generation)
    out = root / "results" / "routing-m7.json"
    assert route_curve.main(argv(paths, out), root=root) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["ft_generation"] == generation
    return out


def test_the_validated_runs_must_be_fts_in_the_routing_files_generation(tmp_path, capsys):
    routing = frozen_with(tmp_path, THINK_GREEDY)
    paths = validation_fixture(tmp_path)
    assert route_validate.main(vargs(routing, paths, routing), root=tmp_path) == 2  # FT's run recorded none
    assert "recorded no generation" in (err := capsys.readouterr().err) and "ft_generation" in err
    rewrite(ft_manifest(paths), generation=THINK_SAMPLED)  # another generation
    assert route_validate.main(vargs(routing, paths, routing), root=tmp_path) == 2
    assert '"temperature":0.6' in capsys.readouterr().err
    assert "validation" not in json.loads(routing.read_text(encoding="utf-8"))
    rewrite(ft_manifest(paths), generation=THINK_GREEDY)
    assert route_validate.main(vargs(routing, paths, routing), root=tmp_path) == 0
    # jointly: the second set's run without a generation is refused beside the first's
    second = v2_validation_fixture(tmp_path)
    out = tmp_path / "joint.json"
    assert route_validate.main(joint_vargs(routing, [paths, second], out), root=tmp_path) == 2
    assert second["set"].as_posix() in capsys.readouterr().err and not out.exists()
    rewrite(ft_manifest(second), generation=THINK_GREEDY)
    assert route_validate.main(joint_vargs(routing, [paths, second], out), root=tmp_path) == 0


def test_a_routing_file_without_fts_generation_validates_only_runs_without_one(tmp_path, capsys):
    routing = frozen(tmp_path)  # chosen on runs of the legacy path, as M4 to M6
    paths = validation_fixture(tmp_path)
    rewrite(ft_manifest(paths), generation=THINK_GREEDY)
    assert route_validate.main(vargs(routing, paths, routing), root=tmp_path) == 2
    assert "records no ft_generation" in capsys.readouterr().err
    data = json.loads(routing.read_text(encoding="utf-8"))
    write_json(routing, {**data, "ft_generation": "think"})  # not the object scripts/route_curve.py writes
    assert route_validate.main(vargs(routing, paths, routing), root=tmp_path) == 1


# Computed with scripts/route_validate.py and route_curve.py and research_agent of commit 64250c0 (before FT's
# generation was frozen) on the fixtures above: the SHA-256 of the routing file of frozen() after the block of
# dev-ext and the joint block of dev-ext and dev-v2-ext were added to it
VALIDATE_GOLDEN = "48171a195f60361eb4702b17ece337bc2b7be55c78ea805d6bdf781b0fc56c94"


def test_without_fts_generation_the_blocks_are_written_byte_for_byte_as_before(tmp_path):
    routing = frozen(tmp_path)
    first, second = validation_fixture(tmp_path), v2_validation_fixture(tmp_path)
    assert route_validate.main(vargs(routing, first, routing), root=tmp_path) == 0
    assert route_validate.main(joint_vargs(routing, [first, second], routing), root=tmp_path) == 0
    assert sha256_of(routing) == VALIDATE_GOLDEN

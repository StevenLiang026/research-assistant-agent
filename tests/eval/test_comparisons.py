"""The comparison statistics of M3 (research_agent.eval.metrics; the M3 plan, default 2): Holm's step-down
adjustment over the preregistered comparisons, the clusters of the cluster bootstrap (a drafted item's
template family, every handwritten item alone), and compare_systems, which joins two systems' scores to the
items by id and runs the paired bootstrap, the cluster bootstrap and the exact McNemar test on them."""

from __future__ import annotations

import itertools
import math
import random
import re
from pathlib import Path

import pytest

from research_agent.eval.lock import DevSet
from research_agent.eval.metrics import (
    IdMismatch,
    bootstrap_ci,
    cluster_bootstrap_ci,
    cluster_keys,
    cluster_noninferiority_p,
    compare_systems,
    holm,
    mcnemar,
    noninferiority_p,
    stratified_noninferiority_p,
)

REPO = Path(__file__).resolve().parents[2]


def exactly(values: list[float]):
    """Hand-computed values, up to the rounding of one float multiplication."""
    return pytest.approx(values, rel=1e-12, abs=0)


def drafted(id_: str, family: object = "read_section/en/5") -> dict:
    """A drafted item's fields that the comparison reads (eval.items: source = {"request", "step"})."""
    request, step = id_.rsplit("-s", 1)
    return {"id": id_, "type": "call", "lang": "en", "family": family, "split": "dev",
            "source": {"request": request, "step": int(step)}}


def handwritten(id_: str) -> dict:
    """A handwritten item's fields that the comparison reads (eval.handwritten: a family per stratum)."""
    return {"id": id_, "type": "call", "lang": "zh", "family": "handwritten/call/zh", "split": "dev",
            "source": {"handwritten": True}}


# ---- holm ------------------------------------------------------------------------------------------------

def test_holm_multiplies_the_kth_smallest_p_by_m_minus_k_plus_1_keeping_the_input_order():
    # sorted: 0.005 x 4 = 0.02, 0.01 x 3 = 0.03, 0.03 x 2 = 0.06, 0.04 x 1 = 0.04, raised to 0.06
    assert holm([0.01, 0.04, 0.03, 0.005]) == exactly([0.03, 0.06, 0.06, 0.02])
    assert holm([0.005, 0.01, 0.03, 0.04]) == exactly([0.02, 0.03, 0.06, 0.06])


def test_tied_p_values_get_the_same_adjusted_value_in_any_order():
    # sorted: 0.02 x 3 = 0.06, 0.02 x 2 = 0.04, raised to 0.06, 0.5 x 1
    assert holm([0.02, 0.5, 0.02]) == exactly([0.06, 0.5, 0.06])
    assert holm([0.5, 0.02, 0.02]) == exactly([0.5, 0.06, 0.06])
    assert holm([0.2, 0.2, 0.2, 0.2]) == exactly([0.8, 0.8, 0.8, 0.8])


def test_a_p_value_of_1_stays_1_and_every_adjusted_value_is_capped_at_1():
    # M3 enters the fourth preregistered comparison, FT+R against B2, at p = 1 (only M4 has FT+R):
    # sorted 0.001 x 4, 0.02 x 3, 0.04 x 2, 1 x 1; more conservative than over the three alone
    assert holm([0.001, 0.02, 0.04, 1.0]) == exactly([0.004, 0.06, 0.08, 1.0])
    assert holm([0.001, 0.02, 0.04]) == exactly([0.003, 0.04, 0.04])
    assert holm([0.01, 1.0, 0.02, 1]) == exactly([0.04, 1.0, 0.06, 1.0])
    assert holm([0.6, 0.7]) == [1.0, 1.0]  # 0.6 x 2 = 1.2, capped
    assert holm([0.0, 0.3]) == exactly([0.0, 0.3])


def test_a_single_p_value_is_left_as_it_is_and_no_p_values_give_none():
    assert holm([0.03]) == [0.03]
    assert holm([1]) == [1.0] and isinstance(holm([1])[0], float)
    assert holm([]) == []


def test_holm_permutes_its_answer_with_its_input_and_keeps_the_order_of_the_raw_p_values():
    """Over random lists with ties: a permuted input gives the permuted answer, bit for bit, and the adjusted
    values lie between the raw value and 1 and are ordered as the raw ones (equal where they are equal)."""
    rng = random.Random(42)
    grid = [0.0, 0.001, 0.01, 0.0125, 0.02, 0.05, 0.25, 0.5, 1.0]
    for _ in range(300):
        p = [rng.choice(grid) for _ in range(rng.randint(1, 7))]
        adjusted = holm(p)
        order = list(range(len(p)))
        rng.shuffle(order)
        assert holm([p[k] for k in order]) == [adjusted[k] for k in order]
        for i, j in itertools.product(range(len(p)), repeat=2):
            assert p[i] <= adjusted[i] <= 1
            if p[i] <= p[j]:
                assert adjusted[i] <= adjusted[j]


@pytest.mark.parametrize("bad", [-0.01, 1.01, math.nan, math.inf, True, "0.05", None])
def test_holm_refuses_an_entry_that_is_no_p_value_by_its_position(bad):
    with pytest.raises(ValueError, match=r"p_values\[1\]"):
        holm([0.01, bad, 0.02])


# ---- cluster keys ----------------------------------------------------------------------------------------

@pytest.mark.needs_data("data/eval/dev.jsonl", "data/eval/dev.jsonl.sha256")  # left out of the public copy
def test_a_drafted_dev_item_clusters_by_its_template_family_and_a_handwritten_one_alone():
    items = DevSet(REPO / "data" / "eval" / "dev.jsonl").open()
    keys = cluster_keys(items)
    assert len(keys) == len(items) == 150
    for item, key in zip(items, keys, strict=True):
        if re.fullmatch(r"hw-dev-\d{3}", item["id"]):
            assert key == ("item", item["id"]), item["id"]
        else:
            assert re.fullmatch(r"req-dev-\d{4}-s\d+", item["id"]), item["id"]
            assert key == ("family", item["family"]), item["id"]
    hand = [item for item in items if item["id"].startswith("hw-dev-")]
    assert len(hand) == 47 and len({item["family"] for item in hand}) == 12  # 12 strata, no shared template
    assert len(set(keys)) == 32 + 47  # the drafted items' 32 template families and 47 single items


def test_cluster_keys_give_one_key_per_item_in_the_items_order():
    items = [drafted("req-dev-0001-s1", "find/zh/1"), handwritten("hw-dev-001"),
             drafted("req-dev-0002-s3", "find/zh/1"), handwritten("hw-dev-002"),
             drafted("req-dev-0003-s1", 7)]
    assert cluster_keys(items) == [("family", "find/zh/1"), ("item", "hw-dev-001"), ("family", "find/zh/1"),
                                   ("item", "hw-dev-002"), ("family", 7)]
    assert cluster_keys([]) == []


def test_a_family_and_a_handwritten_id_spelled_alike_are_different_clusters():
    keys = cluster_keys([drafted("req-dev-0001-s1", family="hw-dev-001"), handwritten("hw-dev-001")])
    assert keys[0] != keys[1]
    assert len(set(cluster_keys([drafted("req-dev-0001-s1", family=7),
                                 drafted("req-dev-0002-s1", family="7")]))) == 2


@pytest.mark.parametrize("source", [None, "req-dev-0001", {}, {"step": 1}, {"handwritten": False},
                                    {"handwritten": "true"}, {"request": None, "step": 1},
                                    {"request": "req-dev-0001", "step": 1, "handwritten": True}])
def test_an_item_whose_kind_cannot_be_determined_is_refused_by_its_id(source):
    """Drafted means source.request (a string), handwritten source.handwritten true; neither or both is
    refused, so no item is clustered by a guess."""
    with pytest.raises(ValueError, match=r"req-dev-0001-s1\b"):
        cluster_keys([handwritten("hw-dev-001"), {**drafted("req-dev-0001-s1"), "source": source}])
    without = {k: v for k, v in drafted("req-dev-0001-s1").items() if k != "source"}
    with pytest.raises(ValueError, match=r"req-dev-0001-s1\b"):
        cluster_keys([without])


@pytest.mark.parametrize("family", [None, "", True, ["find/zh/1"], 1.5])
def test_a_drafted_item_without_a_template_family_is_refused_by_its_id(family):
    with pytest.raises(ValueError, match=r"req-dev-0001-s1\b"):
        cluster_keys([{**drafted("req-dev-0001-s1"), "family": family}])
    without = {k: v for k, v in drafted("req-dev-0001-s1").items() if k != "family"}
    with pytest.raises(ValueError, match=r"req-dev-0001-s1\b"):
        cluster_keys([without])


def test_an_item_without_an_id_or_no_item_at_all_is_refused_by_its_position():
    with pytest.raises(ValueError, match="#2"):
        cluster_keys([handwritten("hw-dev-001"), {**handwritten("hw-dev-002"), "id": None}])
    with pytest.raises(ValueError, match="#3"):
        cluster_keys([handwritten("hw-dev-001"), handwritten("hw-dev-002"), "req-dev-0001-s1"])


# ---- compare_systems -------------------------------------------------------------------------------------

# 40 drafted items in 8 template families of 5, then 20 handwritten ones: 8 + 20 clusters
ITEMS = ([drafted(f"req-dev-{k // 5 + 1:04d}-s{k % 5 + 1}", family=f"find/zh/{k // 5}") for k in range(40)]
         + [handwritten(f"hw-dev-{k:03d}") for k in range(1, 21)])
A_OK = [k % 4 != 0 for k in range(60)]  # 45 of 60
B_OK = [k % 3 == 0 for k in range(60)]  # 20 of 60; b alone gets k % 12 == 0 right (5), a alone 30 others


def scores(successes: list[bool]) -> list[dict]:
    """Scores of ITEMS as score_items gives them, reduced to what compare_systems reads."""
    return [{"id": item["id"], "status": "ok", "success": ok}
            for item, ok in zip(ITEMS, successes, strict=True)]


def test_compare_systems_reports_both_rates_their_difference_and_the_three_tests_of_metrics():
    result = compare_systems(scores(A_OK), scores(B_OK), ITEMS, seed=7, n=2000)
    assert list(result) == ["n", "success_a", "success_b", "diff", "paired_ci", "cluster_ci", "mcnemar"]
    assert (result["n"], result["success_a"], result["success_b"]) == (60, 45 / 60, 20 / 60)
    assert result["diff"] == 25 / 60 == result["paired_ci"]["diff"] == result["cluster_ci"]["diff"]
    assert result["paired_ci"] == bootstrap_ci(A_OK, B_OK, seed=7, n=2000)
    assert result["cluster_ci"] == cluster_bootstrap_ci(A_OK, B_OK, cluster_keys(ITEMS), seed=7, n=2000)
    assert result["cluster_ci"]["clusters"] == 8 + 20
    assert result["mcnemar"] == mcnemar(A_OK, B_OK)
    assert (result["mcnemar"]["a_only"], result["mcnemar"]["b_only"]) == (30, 5)


def test_compare_systems_defaults_to_10000_resamples_with_seed_42_and_answers_the_same_every_time():
    first = compare_systems(scores(A_OK), scores(B_OK), ITEMS)
    assert first == compare_systems(scores(A_OK), scores(B_OK), ITEMS)
    for interval in (first["paired_ci"], first["cluster_ci"]):
        assert (interval["n"], interval["seed"]) == (10000, 42)
        assert interval["low"] < first["diff"] < interval["high"]
    assert first["paired_ci"] == bootstrap_ci(A_OK, B_OK, seed=42, n=10000)


def test_scores_are_matched_to_the_items_by_id_in_any_order_and_paired_in_the_items_order():
    a, b = scores(A_OK), scores(B_OK)
    shuffled = a[:]
    random.Random(1).shuffle(shuffled)
    expected = compare_systems(a, b, ITEMS, seed=3, n=500)
    assert compare_systems(shuffled, b[::-1], ITEMS, seed=3, n=500) == expected
    reordered = compare_systems(a, b, ITEMS[::-1], seed=3, n=500)
    assert reordered["paired_ci"] == bootstrap_ci(A_OK[::-1], B_OK[::-1], seed=3, n=500)


@pytest.mark.parametrize("which", ["a", "b"])
def test_a_score_list_without_exactly_one_score_per_item_raises_id_mismatch_naming_it(which):
    good = {"a": scores(A_OK), "b": scores(B_OK)}
    listed = good[which]
    for bad in (listed[1:],                                           # an item without a score
                [*listed, {"id": "req-dev-0099-s1", "success": True}],  # a score without an item
                [*listed[:-1], listed[0]],                            # a repeated id in place of another
                [*listed, listed[0]]):                                # a repeated id beside all the others
        given = {**good, which: bad}
        with pytest.raises(IdMismatch, match=f"scores_{which}"):
            compare_systems(given["a"], given["b"], ITEMS, seed=1, n=10)


@pytest.mark.parametrize("which", ["a", "b"])
def test_a_malformed_score_list_is_refused_naming_it_not_with_a_python_error(which):
    good = {"a": scores(A_OK), "b": scores(B_OK)}
    listed = good[which]
    malformed = ([*listed[:-1], "req-dev-0001-s1"],               # a score that is not an object
                 [*listed[:-1], {**listed[-1], "id": ["x"]}])     # an id that is not a string
    for bad, error in zip(malformed, (ValueError, IdMismatch), strict=True):
        given = {**good, which: bad}
        with pytest.raises(error, match=f"scores_{which}"):
            compare_systems(given["a"], given["b"], ITEMS, seed=1, n=10)


def test_an_item_that_no_score_list_covers_raises_id_mismatch_naming_it():
    with pytest.raises(IdMismatch, match="hw-dev-099"):
        compare_systems(scores(A_OK), scores(B_OK), [*ITEMS, handwritten("hw-dev-099")], seed=1, n=10)


def test_items_without_distinct_string_ids_are_refused_as_malformed_not_as_a_mismatch():
    for items in ([*ITEMS, handwritten("hw-dev-001")], [*ITEMS[:-1], {**ITEMS[-1], "id": 20}]):
        with pytest.raises(ValueError, match="distinct strings") as caught:
            compare_systems(scores(A_OK), scores(B_OK), items, seed=1, n=10)
        assert not isinstance(caught.value, IdMismatch)


@pytest.mark.parametrize("success", [None, 1, "true", "missing"])
def test_a_score_whose_success_is_not_a_bool_is_refused_by_its_id(success):
    def spoilt(score: dict) -> dict:
        if score["id"] != "hw-dev-003":
            return score
        return {k: v for k, v in score.items() if k != "success"} if success == "missing" else {
            **score, "success": success}

    with pytest.raises(ValueError, match="hw-dev-003"):
        compare_systems(scores(A_OK), [spoilt(s) for s in scores(B_OK)], ITEMS, seed=1, n=10)


def test_compare_systems_refuses_an_item_whose_kind_cannot_be_determined_by_its_id():
    items = [*ITEMS[:-1], {**ITEMS[-1], "source": {}}]
    with pytest.raises(ValueError, match="hw-dev-020"):
        compare_systems(scores(A_OK), scores(B_OK), items, seed=1, n=10)


def test_no_items_are_refused():
    with pytest.raises(ValueError, match="no items"):
        compare_systems([], [], [])


# ---- M6: the one-sided non-inferiority bootstraps (docs/m6-计划.md §5, H2 and H3) ------------------------

N_NI = 1000
# 200 items only a gets right and 220 only b: a's rate is 0.02 below b's, exactly the margin
AT_MARGIN_A = [k < 200 or k >= 420 for k in range(N_NI)]
AT_MARGIN_B = [k >= 200 for k in range(N_NI)]


def test_noninferiority_p_is_0_far_above_the_margin_and_about_half_at_minus_the_margin():
    far = noninferiority_p([True] * N_NI, [k % 2 == 0 for k in range(N_NI)], 0.02, seed=42, n=10000)
    assert far["p_ni"] == 0.0 and far["below"] == 0 and far["diff"] == 0.5
    assert (far["margin"], far["n"], far["seed"]) == (0.02, 10000, 42)
    at = noninferiority_p(AT_MARGIN_A, AT_MARGIN_B, 0.02, seed=42, n=10000)
    assert at["diff"] == pytest.approx(-0.02)
    assert abs(at["p_ni"] - 0.5) < 0.03 and at["below"] == round(at["p_ni"] * 10000)
    # a 0.02 above b: the mirror image, a small p; a far below b (0.5 under it): every draw counts
    assert noninferiority_p(AT_MARGIN_B, AT_MARGIN_A, 0.02, seed=42, n=10000)["p_ni"] < 0.05
    below = noninferiority_p([k % 2 == 0 for k in range(N_NI)], [True] * N_NI, 0.02, seed=42, n=10000)
    assert below["p_ni"] == 1.0 and below["below"] == 10000 and below["diff"] == -0.5


def test_noninferiority_p_is_deterministic_and_0_on_identical_inputs():
    first = noninferiority_p(AT_MARGIN_A, AT_MARGIN_B, 0.02, seed=5, n=300)
    assert first == noninferiority_p(AT_MARGIN_A, AT_MARGIN_B, 0.02, seed=5, n=300)
    same = noninferiority_p(AT_MARGIN_A, AT_MARGIN_A, 0.02, seed=5, n=300)
    assert same == noninferiority_p(AT_MARGIN_A, AT_MARGIN_A, 0.02, seed=5, n=300)
    assert (same["p_ni"], same["below"], same["diff"]) == (0.0, 0, 0.0)
    # a draw at or below minus the margin counts: with a margin of 0 every draw of identical inputs does
    assert noninferiority_p(AT_MARGIN_A, AT_MARGIN_A, 0.0, seed=5, n=300)["p_ni"] == 1.0
    with pytest.raises(ValueError):
        noninferiority_p([True], [True, False], 0.02, seed=5, n=10)
    with pytest.raises(ValueError):
        noninferiority_p([], [], 0.02, seed=5, n=10)


def test_the_one_sided_p_counts_the_draws_of_the_paired_interval_below_minus_the_margin():
    # the same seed draws the same resamples: a low end above -margin means fewer than 2.5% of the draws
    # lie below it, and the count is a plain fraction of the draws
    interval = bootstrap_ci(A_OK, B_OK, seed=7, n=2000)
    p = noninferiority_p(A_OK, B_OK, 0.02, seed=7, n=2000)
    assert interval["low"] > -0.02 and p["p_ni"] < 0.025
    assert p["p_ni"] == p["below"] / 2000


def test_the_cluster_version_draws_whole_families_and_is_deterministic():
    families = cluster_keys(ITEMS)
    got = cluster_noninferiority_p(A_OK, B_OK, families, 0.02, seed=7, n=2000)
    assert got == cluster_noninferiority_p(A_OK, B_OK, families, 0.02, seed=7, n=2000)
    assert got["p_ni"] == 0.0 and got["clusters"] == 28 and got["diff"] == 25 / 60
    assert cluster_noninferiority_p(B_OK, A_OK, families, 0.02, seed=7, n=2000)["p_ni"] == 1.0
    with pytest.raises(ValueError, match="family"):
        cluster_noninferiority_p(A_OK, B_OK, families[:5], 0.02, seed=7, n=10)


def test_stratified_noninferiority_p_matches_a_hand_computation_on_two_strata():
    # Stratum x: two items, a right on both, b on the first only, so the paired differences are [1, 0] and
    # their resampled mean is 0, 0.5 or 1 with probabilities 1/4, 1/2, 1/4. Stratum y: one item, which a
    # fails and b passes: a difference of -1 whatever the draw (a stratum with one item does not crash).
    # The mean over the two strata is -0.5, -0.25 or 0 with those probabilities, observed (0.5 - 1) / 2.
    strata_a = {"x": [True, True], "y": [False]}
    strata_b = {"x": [True, False], "y": [True]}
    got = stratified_noninferiority_p(strata_a, strata_b, 0.3, seed=42, n=10000)
    assert got["diff"] == pytest.approx(-0.25) and (got["strata"], got["items"]) == (2, 3)
    assert abs(got["p_ni"] - 0.25) < 0.02  # P(mean <= -0.3) = P(-0.5) = 1/4
    assert got["below"] == round(got["p_ni"] * 10000) and (got["n"], got["seed"]) == (10000, 42)
    assert got["low"] <= got["diff"] <= got["high"] and got["low"] == -0.5 and got["high"] == 0.0
    tighter = stratified_noninferiority_p(strata_a, strata_b, 0.25, seed=42, n=10000)
    assert abs(tighter["p_ni"] - 0.75) < 0.02  # P(mean <= -0.25) = 1/4 + 1/2
    assert got == stratified_noninferiority_p(strata_a, strata_b, 0.3, seed=42, n=10000)


def test_stratified_noninferiority_p_is_0_far_above_the_margin_and_about_half_at_minus_the_margin():
    far = stratified_noninferiority_p({"x": [True] * 50, "y": [True] * 50}, {"x": [False] * 50,
                                                                              "y": [True] * 50},
                                      0.02, seed=42, n=5000)
    assert far["p_ni"] == 0.0 and far["diff"] == 0.5
    at = stratified_noninferiority_p({"x": AT_MARGIN_A, "y": AT_MARGIN_A},
                                     {"x": AT_MARGIN_B, "y": AT_MARGIN_B}, 0.02, seed=42, n=5000)
    assert at["diff"] == pytest.approx(-0.02) and abs(at["p_ni"] - 0.5) < 0.04
    same = stratified_noninferiority_p({"x": AT_MARGIN_A}, {"x": AT_MARGIN_A}, 0.02, seed=42, n=500)
    assert same["p_ni"] == 0.0 and same["diff"] == 0.0


@pytest.mark.parametrize("strata_a, strata_b, message", [
    ({"x": [True]}, {"y": [True]}, "strata"),
    ({"x": [True, False]}, {"x": [True]}, "x"),
    ({"x": []}, {"x": []}, "x"),
    ({}, {}, "no strata"),
])
def test_strata_that_do_not_pair_are_refused(strata_a, strata_b, message):
    with pytest.raises(ValueError, match=message):
        stratified_noninferiority_p(strata_a, strata_b, 0.02, seed=1, n=10)


def test_compare_systems_with_a_margin_adds_the_one_sided_p_values_of_both_bootstraps():
    result = compare_systems(scores(A_OK), scores(B_OK), ITEMS, seed=7, n=2000, margin=0.02)
    assert list(result)[-1] == "noninferiority"
    bound = result["noninferiority"]
    paired = noninferiority_p(A_OK, B_OK, 0.02, seed=7, n=2000)
    clustered = cluster_noninferiority_p(A_OK, B_OK, cluster_keys(ITEMS), 0.02, seed=7, n=2000)
    assert bound == {"margin": 0.02, "p_ni": paired["p_ni"], "below": paired["below"],
                     "cluster_p_ni": clustered["p_ni"], "cluster_below": clustered["below"]}
    without = compare_systems(scores(A_OK), scores(B_OK), ITEMS, seed=7, n=2000)
    assert "noninferiority" not in without
    assert {k: v for k, v in result.items() if k != "noninferiority"} == without

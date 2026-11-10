"""research_agent.eval.acebench: the per-answer successes of an ACEBench run read back from the raw folder
scripts/run_acebench.py copies (runs/acebench/<run>/<system>/<language>/{result,score}/), the (language,
category) strata H3's stratified bootstrap resamples within, and the pairing of two runs by answer id. On
fabricated folders, and on the repository's M4 folders when they are here (the counts of
results/acebench-m4.json come back)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_agent.eval.acebench import (
    MULTI_CANDIDATE,
    AceItemsError,
    pair_strata,
    per_item,
    stratum_label,
)

REPO = Path(__file__).resolve().parents[2]
ITEMS = {("zh", "normal_atom_bool"): 4, ("zh", "special_incomplete"): 2, ("en", "normal_atom_bool"): 3}
RIGHT = {("zh", "normal_atom_bool"): {0, 1, 3}, ("zh", "special_incomplete"): set(),
         ("en", "normal_atom_bool"): {0, 1, 2}}


def ids_of(category: str, n: int) -> list[str]:
    return [f"{category}_{i}" for i in range(n)]


def write_stratum(folder: Path, language: str, category: str, ids: list[str], wrong: list[str], *,
                  correct: int | None = None, extra: list[dict] = (), result: bool = True) -> None:
    """One category's result and score files as ACEBench writes them: one answer per question, a header
    then one line per wrong answer (no final newline); extra lines are appended to the score file."""
    base = folder / language
    if result:
        lines = "".join(json.dumps({"id": i, "result": "[f(x=1)]"}) + "\n" for i in ids)
        base.joinpath("result").mkdir(parents=True, exist_ok=True)
        (base / "result" / f"data_{category}_result.json").write_text(lines, encoding="utf-8")
    right = len(ids) - len(wrong) if correct is None else correct
    header = {"accuracy": round(right / len(ids), 3) if ids else 0, "correct_count": right,
              "total_count": len(ids)}
    rows = [header, *({"id": i, "valid": False, "error": ["x"], "error_type": "y"} for i in wrong), *extra]
    base.joinpath("score").mkdir(parents=True, exist_ok=True)
    (base / "score" / f"data_{category}_score.json").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")


def folder(root: Path, name: str = "b0", right: dict | None = None, **changes) -> Path:
    """A raw folder of ITEMS' strata, the answers of RIGHT (or right) correct; changes override one
    stratum's write_stratum keywords, keyed "<language>/<category>"."""
    right = RIGHT if right is None else right
    base = root / name
    for (language, category), n in ITEMS.items():
        ids = ids_of(category, n)
        wrong = [i for k, i in enumerate(ids) if k not in right[(language, category)]]
        write_stratum(base, language, category, ids, wrong, **changes.get(f"{language}/{category}", {}))
    return base


def test_per_item_reads_every_language_and_category_with_one_success_per_answer(tmp_path):
    got = per_item(folder(tmp_path))
    assert list(got.strata) == [("en", "normal_atom_bool"), ("zh", "normal_atom_bool"),
                                ("zh", "special_incomplete")]  # sorted
    assert got.strata[("zh", "normal_atom_bool")] == {"normal_atom_bool_0": True, "normal_atom_bool_1": True,
                                                      "normal_atom_bool_2": False,
                                                      "normal_atom_bool_3": True}
    assert got.strata[("zh", "special_incomplete")] == {"special_incomplete_0": False,
                                                        "special_incomplete_1": False}
    assert got.headers[("zh", "normal_atom_bool")] == {"n": 4, "correct": 3, "later_candidates": 0}
    assert got.headers[("zh", "special_incomplete")] == {"n": 2, "correct": 0, "later_candidates": 0}
    assert got.items == 9 and got.path == (tmp_path / "b0").as_posix()
    assert [name for name, _ in got.files] == sorted(name for name, _ in got.files) and len(got.files) == 6
    assert all(len(digest) == 64 for _, digest in got.files) and len(got.sha256) == 64
    assert stratum_label(("zh", "normal_atom_bool")) == "zh/normal_atom_bool"


def test_the_digest_follows_the_files_read_whatever_the_folder_is_called(tmp_path):
    first, second = per_item(folder(tmp_path, "one")), per_item(folder(tmp_path, "two"))
    assert first.sha256 == second.sha256 and first.files == second.files
    changed = folder(tmp_path, "three", right={**RIGHT, ("en", "normal_atom_bool"): {0, 1}})
    other = per_item(changed)
    assert other.sha256 != first.sha256
    assert [name for name, _ in other.files] == [name for name, _ in first.files]


def test_an_extra_wrong_line_of_a_question_with_several_candidate_answers_counts_it_right(tmp_path):
    # ACEBench writes a wrong line for a question that failed its first candidate answer and passed a later
    # one, and counts it correct: the header says one more correct answer than the wrong lines leave
    ids = ids_of("normal_atom_bool", 25)
    later = "normal_atom_bool_21"  # zh normal_atom_bool's one question with two candidate answers
    assert later in MULTI_CANDIDATE[("zh", "normal_atom_bool")]
    base = tmp_path / "ft"
    write_stratum(base, "zh", "normal_atom_bool", ids, ["normal_atom_bool_3", later], correct=24)
    got = per_item(base)
    successes = got.strata[("zh", "normal_atom_bool")]
    assert successes[later] and not successes["normal_atom_bool_3"] and sum(successes.values()) == 24
    assert got.headers[("zh", "normal_atom_bool")] == {"n": 25, "correct": 24, "later_candidates": 1}
    # the same extra line for a question with one candidate answer cannot be right: refused
    write_stratum(base, "zh", "normal_atom_bool", ids, ["normal_atom_bool_3", "normal_atom_bool_4"],
                  correct=24)
    with pytest.raises(AceItemsError, match="normal_atom_bool_score.json"):
        per_item(base)


@pytest.mark.parametrize("changes, message", [
    ({"zh/normal_atom_bool": {"result": False}}, "data_normal_atom_bool_result.json is missing"),
    ({"zh/normal_atom_bool": {"correct": 5}}, "data_normal_atom_bool_score.json"),  # correct above n
    ({"zh/normal_atom_bool": {"correct": 2}}, "data_normal_atom_bool_score.json"),  # fewer wrong lines
    ({"zh/normal_atom_bool": {"extra": [{"id": "normal_atom_bool_9", "valid": False}]}},
     "normal_atom_bool_9"),  # a wrong id that is no answer
    ({"zh/normal_atom_bool": {"extra": [{"id": "normal_atom_bool_2", "valid": False}]}},
     "data_normal_atom_bool_score.json"),  # a wrong id twice
    ({"zh/normal_atom_bool": {"extra": [{"id": "normal_atom_bool_1", "valid": True}]}},
     "data_normal_atom_bool_score.json"),  # a line that is not a wrong answer
    ({"zh/normal_atom_bool": {"ids": ["normal_atom_bool_0", "normal_atom_bool_0"], "wrong": []}},
     "data_normal_atom_bool_result.json"),  # repeated answer ids
])
def test_a_folder_whose_files_do_not_give_one_success_per_answer_is_refused_naming_the_file(tmp_path,
                                                                                           changes,
                                                                                           message):
    base = tmp_path / "b0"
    for (language, category), n in ITEMS.items():
        ids = ids_of(category, n)
        wrong = [i for k, i in enumerate(ids) if k not in RIGHT[(language, category)]]
        options = dict(changes.get(f"{language}/{category}", {}))
        ids, wrong = options.pop("ids", ids), options.pop("wrong", wrong)
        write_stratum(base, language, category, ids, wrong, **options)
    with pytest.raises(AceItemsError, match=message):
        per_item(base)


def test_a_folder_without_a_language_or_a_score_file_is_refused(tmp_path):
    with pytest.raises(AceItemsError, match="no language"):
        per_item(tmp_path / "nothing")
    (tmp_path / "empty" / "zh" / "score").mkdir(parents=True)
    with pytest.raises(AceItemsError, match="no score file"):
        per_item(tmp_path / "empty")
    broken = folder(tmp_path, "broken")
    (broken / "en" / "score" / "data_normal_atom_bool_score.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(AceItemsError, match="data_normal_atom_bool_score.json"):
        per_item(broken)


def test_pair_strata_pairs_the_answers_by_id_in_the_order_of_the_first_and_refuses_other_runs(tmp_path):
    b0 = per_item(folder(tmp_path, "b0"))
    ft = per_item(folder(tmp_path, "ft", right={("zh", "normal_atom_bool"): {0, 1, 2, 3},
                                                 ("zh", "special_incomplete"): {1},
                                                 ("en", "normal_atom_bool"): {2}}))
    strata_a, strata_b = pair_strata(ft, b0)
    assert list(strata_a) == list(strata_b) == list(b0.strata)
    assert strata_a[("zh", "special_incomplete")] == [False, True]
    assert strata_b[("zh", "special_incomplete")] == [False, False]
    assert strata_a[("en", "normal_atom_bool")] == [False, False, True]
    assert strata_b[("en", "normal_atom_bool")] == [True, True, True]
    fewer = {k: v for k, v in ITEMS.items() if k[0] == "zh"}
    base = tmp_path / "fewer"
    for (language, category), n in fewer.items():
        write_stratum(base, language, category, ids_of(category, n), [])
    with pytest.raises(ValueError, match="same .*strata"):
        pair_strata(per_item(base), b0)
    other = tmp_path / "other"
    for (language, category), n in ITEMS.items():
        ids = ids_of(category, n)
        if (language, category) == ("en", "normal_atom_bool"):
            ids = ["normal_atom_bool_7", *ids[1:]]
        write_stratum(other, language, category, ids, [])
    with pytest.raises(ValueError, match="en/normal_atom_bool"):
        pair_strata(per_item(other), b0)


@pytest.mark.parametrize("system", ["b0", "ft", "b4"])
def test_the_m4_per_answer_successes_reproduce_the_counts_of_the_results_file(system):
    raw, results = REPO / "runs" / "acebench" / "m4" / system, REPO / "results" / "acebench-m4.json"
    if not raw.is_dir() or not results.is_file():
        pytest.skip("the M4 ACEBench run is not here")
    recorded = json.loads(results.read_text(encoding="utf-8"))["systems"]
    if system not in recorded:  # b4 ran into a results file of its own (results/acebench-m4-b4.json)
        other = REPO / "results" / f"acebench-m4-{system}.json"
        if not other.is_file():
            pytest.skip(f"no ACEBench results of {system} here")
        recorded = json.loads(other.read_text(encoding="utf-8"))["systems"]
    got = per_item(raw)
    assert len(got.strata) == 30 and got.items == 1940
    for (language, category), successes in got.strata.items():
        cell = recorded[system]["languages"][language]["categories"][category]
        assert (len(successes), sum(successes.values())) == (cell["n"], cell["correct"]), (language, category)
        assert got.headers[(language, category)]["later_candidates"] == 0
    assert got.sha256 == per_item(raw).sha256

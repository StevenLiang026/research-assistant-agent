"""The second family's single-step sets (v2, M7: docs/m7-计划.md WP-I 5).

research_agent.eval.sets' combined_verdicts, fill_in_order and assemble_v2 on hand-made pools;
scripts/finalize_eval_sets.py --family v2 at the real quotas on fabricated inputs, its refusals and its lock
guard; v1's mode, byte for byte what the script wrote before v2 (a fixture); and the contract of the files
the v2 build wrote: data/eval/v2/dev-v2.jsonl and dev-v2-ext.jsonl, and data/eval/v2/test-v2.jsonl only with
--test-material (tests/conftest.py). The contract: sizes and strata as the review's count
(data/eval/v2/review/handwritten-needed.json) gives them, every item of its set's split, every id unique and
apart from v1's sets (v1's test ids by ids alone: its test set is never opened), every item one the review
kept, taken in the review pool's order, no item near a training request, the locks, the guards that keep
test-v2 from every runner but run_test, and run_test's v2 development sets. No check prints an item's text."""

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import conftest
import pytest

from research_agent.agent.types import HarnessError
from research_agent.canonical import canonical_json
from research_agent.data.requests import TRAIN_VS_EVAL, mask_text, shingles
from research_agent.eval.handwritten import TEMPLATE_LIMIT
from research_agent.eval.items import LANGS, QUOTAS_DEV, QUOTAS_TEST, TYPES, scaled
from research_agent.eval.lock import (
    DEV_SET,
    E2E_TEST_ACCESS_LOG,
    E2E_TEST_ACCESS_LOG_V2,
    TEST_ACCESS_LOG,
    TEST_ACCESS_LOG_V2,
    TEST_COMPLETIONS,
    TEST_COMPLETIONS_V2,
    DevSet,
    RefusedTestSet,
    check_lock,
    find_test_material,
    read_jsonl,
    read_lock,
    refuse_test_sets,
    sha256_of,
    write_lock,
)
from research_agent.eval.sets import (
    DEV_QUOTAS_V2,
    TEST_V2,
    SetError,
    assemble,
    assemble_v2,
    combined_verdicts,
    fill_in_order,
    per_language,
    sort_key,
    verdict_counts,
)

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


finalize = load_script("finalize_eval_sets")


def item(iid: str, type_: str, lang: str, split: str = "dev", handwritten: bool = False) -> dict:
    source = {"handwritten": True} if handwritten else {"request": iid.rsplit("-s", 1)[0], "step": 1}
    return {"id": iid, "split": split, "type": type_, "lang": lang,
            "messages": [{"role": "user", "content": iid}], "source": source,
            "expected": {"content": "", "tool_calls": []}}


def review(iid: str, verdict: str) -> dict:
    """A review line in the v2 reviewers' format (data/eval/v2/review/instructions.md)."""
    return {"id": iid, "verdict": verdict, "category": "ok" if verdict == "keep" else "ambiguous",
            "confidence": "medium", "reason": "A made-up reason."}


def write_rows(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8",
                    newline="\n")
    return path


# ---- the library: combined_verdicts --------------------------------------------------------------------


def test_an_item_reviewed_twice_is_kept_only_when_both_reviews_keep_it():
    first = [review("a", "keep"), review("b", "keep"), review("c", "drop"), review("d", "keep"),
             review("e", "drop")]
    second = [review("a", "keep"), review("b", "drop"), review("c", "keep")]
    assert combined_verdicts(first, second) == {"a": "keep", "b": "drop", "c": "drop", "d": "keep",
                                                "e": "drop"}
    assert combined_verdicts(first) == {"a": "keep", "b": "keep", "c": "drop", "d": "keep", "e": "drop"}
    # a verdict other than keep or drop is kept as it is: it never enters a set either
    assert combined_verdicts([review("a", "keep")], [{"id": "a", "verdict": "review_failed"}]) == {
        "a": "review_failed"}


@pytest.mark.parametrize(("first", "second", "message"), [
    ([{"id": "a"}], [], "first review line 1 has no id or no verdict"),
    ([review("a", "keep"), {"verdict": "keep"}], [], "first review line 2 has no id or no verdict"),
    ([review("a", "keep"), review("a", "drop")], [], "the first reviews judge a twice"),
    ([review("a", "keep")], [{"id": "a", "verdict": None}], "second review line 1 has no id or no verdict"),
    ([review("a", "keep")], [review("b", "keep")], "a second review judges b, which no first review judges"),
    ([review("a", "keep")], [review("a", "keep"), review("a", "drop")], "the second reviews judge a twice"),
], ids=["no-verdict", "no-id", "first-twice", "second-no-verdict", "second-only", "second-twice"])
def test_reviews_that_do_not_fit_together_are_refused(first, second, message):
    with pytest.raises(SetError, match=message):
        combined_verdicts(first, second)


# ---- the library: fill_in_order and assemble_v2 --------------------------------------------------------

# Two sets of the dev split, filled in this order: "a" (call zh 2 / en 2, no_call zh 1 / en 1), then "b"
# (call zh 1 / en 1, no_call zh 1 / en 1).
SMALL = {"a": {"call": 4, "no_call": 2}, "b": {"call": 2, "no_call": 2}}
# (split, type, lang, rank, id, verdict); listed out of rank order on purpose
POOL = [("dev", "call", "zh", 5, "d-05-s1", "keep"), ("dev", "call", "zh", 2, "d-02-s1", "drop"),
        ("dev", "call", "zh", 1, "d-01-s1", "keep"), ("dev", "call", "zh", 4, "d-04-s1", "keep"),
        ("dev", "call", "zh", 3, "d-03-s1", "keep"), ("dev", "call", "en", 2, "d-07-s1", "keep"),
        ("dev", "call", "en", 1, "d-06-s1", "keep"), ("dev", "no_call", "zh", 1, "d-08-s1", "keep"),
        ("dev", "no_call", "en", 1, "d-09-s1", "review_failed"), ("test", "call", "zh", 1, "t-01-s1", "keep")]
POOL_ROWS = [{"split": s, "type": t, "lang": lang, "rank": r, "id": i} for s, t, lang, r, i, _ in POOL]
POOL_VERDICTS = {i: v for *_, i, v in POOL}
POOL_DRAFTED = {i: item(i, t, lang, s) for s, t, lang, _, i, _ in POOL}


def record(full: int, kept: int, taken: int, *, target: int | None = None, handwritten_kept: int = 0,
           handwritten: int = 0) -> dict:
    return {"full": full, "target": full if target is None else target, "drafted_kept": kept,
            "drafted": taken, "handwritten_kept": handwritten_kept, "handwritten": handwritten,
            "short": full - taken - handwritten}


def test_the_sets_of_a_split_take_its_kept_drafted_items_in_pool_order_one_after_the_other():
    sets, table = fill_in_order(POOL_ROWS, POOL_VERDICTS, POOL_DRAFTED, split="dev", quotas=SMALL)
    # call/zh keeps ranks 1, 3, 4 and 5: "a" takes 1 and 3, "b" takes 4, rank 5 is beyond both quotas
    assert [i["id"] for i in sets["a"]] == ["d-01-s1", "d-03-s1", "d-06-s1", "d-07-s1", "d-08-s1"]
    assert [i["id"] for i in sets["b"]] == ["d-04-s1"]
    assert table["a"] == {"call/zh": record(2, 4, 2), "call/en": record(2, 2, 2),
                          "no_call/zh": record(1, 1, 1), "no_call/en": record(1, 0, 0)}
    assert table["b"] == {"call/zh": record(1, 4, 1), "call/en": record(1, 2, 0),
                          "no_call/zh": record(1, 1, 0), "no_call/en": record(1, 0, 0)}
    for items in sets.values():
        assert items == sorted(items, key=sort_key)
    # a verdict other than keep never enters a set, and the other split's rows are not read into it
    taken = {i["id"] for items in sets.values() for i in items}
    assert not taken & {"d-02-s1", "d-09-s1", "t-01-s1"}


def test_fill_in_order_reads_the_pool_rows_once_whatever_iterable_they_come_in():
    sets, _ = fill_in_order(iter(POOL_ROWS), POOL_VERDICTS, POOL_DRAFTED, split="dev", quotas=SMALL)
    assert [i["id"] for i in sets["b"]] == ["d-04-s1"]


@pytest.mark.parametrize(("change", "message"), [
    ({"drafted": {k: v for k, v in POOL_DRAFTED.items() if k != "d-03-s1"}}, "d-03-s1 is not drafted"),
    ({"drafted": {**POOL_DRAFTED, "d-03-s1": item("d-03-s1", "call", "en")}},
     "d-03-s1 is a dev/call/en item"),
    ({"rows": [*POOL_ROWS, {"split": "dev", "type": "call", "lang": "zh", "rank": 6, "id": "d-01-s1"}]},
     "pool ids seen twice: \\['d-01-s1'\\]"),
    ({"quotas": {"a": {"call": 4}, "b": {"call": 2, "no_call": 2}}}, "the sets of dev have different types"),
], ids=["undrafted", "stratum-mismatch", "repeated", "types-differ"])
def test_fill_in_order_refuses_inputs_that_do_not_fit(change, message):
    args = {"rows": POOL_ROWS, "drafted": POOL_DRAFTED, "quotas": SMALL, **change}
    with pytest.raises(SetError, match=message):
        fill_in_order(args["rows"], POOL_VERDICTS, args["drafted"], split="dev", quotas=args["quotas"])


def test_assemble_v2_fills_the_test_set_as_v1_did_and_the_dev_sets_in_order():
    # the test split: call zh 2 / en 2 at share 0.7 (target zh 2, en 1); call/en drops its first item
    rows = [*POOL_ROWS, {"split": "test", "type": "call", "lang": "zh", "rank": 2, "id": "t-02-s1"},
            {"split": "test", "type": "call", "lang": "en", "rank": 1, "id": "t-03-s1"},
            {"split": "test", "type": "call", "lang": "en", "rank": 2, "id": "t-04-s1"}]
    verdicts = {**POOL_VERDICTS, "t-02-s1": "keep", "t-03-s1": "drop", "t-04-s1": "keep"}
    drafted = {**POOL_DRAFTED, "t-02-s1": item("t-02-s1", "call", "zh", "test"),
               "t-03-s1": item("t-03-s1", "call", "en", "test"),
               "t-04-s1": item("t-04-s1", "call", "en", "test")}
    handwritten = [item(iid, "call", lang, "test", True)
                   for iid, lang in (("hw-test-903", "en"), ("hw-test-901", "zh"), ("hw-test-902", "en"),
                                     ("hw-test-904", "en"))]
    hw_verdicts = {"hw-test-901": "keep", "hw-test-902": "drop", "hw-test-903": "keep", "hw-test-904": "keep"}
    sets, table = assemble_v2(rows, verdicts, drafted, handwritten, hw_verdicts, test_quotas={"call": 4},
                              dev_quotas=SMALL)
    assert list(sets) == [TEST_V2, "a", "b"] and list(table) == list(sets)
    assert [i["id"] for i in sets[TEST_V2]] == ["t-01-s1", "t-02-s1", "hw-test-903", "t-04-s1"]
    assert table[TEST_V2] == {
        "call/zh": record(2, 2, 2, target=2, handwritten_kept=1, handwritten=0),
        "call/en": record(2, 1, 1, target=1, handwritten_kept=2, handwritten=1)}
    assert [i["id"] for i in sets["b"]] == ["d-04-s1"] and table["b"]["call/en"]["short"] == 1
    # the dev sets hold no handwritten item: a handwritten dev item is refused, as v1 refused an unknown split
    with pytest.raises(SetError, match="handwritten items of no known split"):
        assemble_v2(rows, verdicts, drafted, [*handwritten, item("hw-dev-001", "call", "zh", "dev", True)],
                    hw_verdicts, test_quotas={"call": 4}, dev_quotas=SMALL)


def test_the_v2_quotas_are_v1s_test_quotas_and_v1s_dev_quotas_scaled_to_140_and_to_70():
    assert TEST_V2 == "test-v2"
    assert DEV_QUOTAS_V2 == {
        "dev-v2": {"call": 56, "next_step": 14, "no_call": 23, "clarify": 19, "recover": 19, "parallel": 9},
        "dev-v2-ext": {"call": 28, "next_step": 7, "no_call": 12, "clarify": 9, "recover": 9, "parallel": 5}}
    assert sum(QUOTAS_TEST.values()) == 300


# ---- the script, --family v2, on fabricated inputs at the real quotas ------------------------------------

TEST_FULL, TEST_TARGET = per_language(QUOTAS_TEST), per_language(scaled(QUOTAS_TEST, 0.7))
DEV_FULL = {"dev-v2": per_language(scaled(QUOTAS_DEV, Fraction(140, 150))),
            "dev-v2-ext": per_language(scaled(QUOTAS_DEV, Fraction(70, 150)))}
SHORT_DEV = ("recover", "zh")  # the review keeps one item fewer than dev-v2's quota there
DROPPED_HW = ("no_call", "en")  # the review drops the lowest handwritten id there


def fabricate(root: Path) -> dict:
    """The inputs of a v2 build at the real quotas under root.

    Dev: per stratum dev-v2's plus dev-v2-ext's quota of kept drafted items and one dropped after them, but
    SHORT_DEV keeps one item fewer than dev-v2's quota. Test: per stratum its drafted target of kept items and
    one dropped after them; the first item of call/zh and of call/en sits in batch 001, reviewed twice: the
    second review drops call/zh's (so that stratum takes one drafted item fewer) and keeps call/en's, as it
    keeps the first dev call/en item, batch 001's third. Handwritten: per test stratum what it needs plus one
    spare, all kept, and in DROPPED_HW one more, the lowest id, dropped; the first two handwritten items sit
    in batch 001, reviewed twice and kept twice. Run "a" drafts the dev items and the test call items, run "b"
    the rest. The review's count (handwritten-needed.json) is worked out here, independently of the script."""
    numbers = itertools.count(1)
    rows, first, second, runs = [], [], [], {"a": [], "b": []}
    expect: dict[str, list[str]] = {TEST_V2: [], "dev-v2": [], "dev-v2-ext": []}
    strata: dict[str, dict] = {}

    def drafted(split: str, type_: str, lang: str, rank: int, verdict: str, batch: str, run: str) -> str:
        iid = f"req-{split}-{next(numbers):04d}-s1"
        runs[run].append(item(iid, type_, lang, split))
        rows.append({"split": split, "type": type_, "lang": lang, "rank": rank, "id": iid, "batch": batch})
        first.append(review(iid, verdict))
        return iid

    for (type_, lang), q1 in DEV_FULL["dev-v2"].items():
        q2 = DEV_FULL["dev-v2-ext"][type_, lang]
        size = q1 if (type_, lang) == SHORT_DEV else q1 + q2 + 1
        ids = [drafted("dev", type_, lang, rank, "keep" if rank < size else "drop",
                       "001" if (type_, lang, rank) == ("call", "en", 1) else "002", "a")
               for rank in range(1, size + 1)]
        if (type_, lang) == ("call", "en"):
            second.append(review(ids[0], "keep"))
        kept = ids[:size - 1]
        expect["dev-v2"] += kept[:q1]
        expect["dev-v2-ext"] += kept[q1:q1 + q2]
        taken = min(len(kept), q1)
        strata[f"dev/{type_}/{lang}"] = {
            "kept": len(kept), "dev-v2": {"quota": q1, "taken": taken, "short": q1 - taken},
            "dev-v2-ext": {"quota": q2, "taken": min(len(kept) - taken, q2),
                           "short": q2 - min(len(kept) - taken, q2)}}
    handwritten, hw_first, hw_batches = [], [], {"001": [], "002": []}
    hw_numbers = itertools.count(165)
    for (type_, lang), target in TEST_TARGET.items():
        run = "a" if type_ == "call" else "b"
        ids = [drafted("test", type_, lang, rank, "keep" if rank <= target else "drop",
                       "001" if type_ == "call" and rank == 1 else "002", run)
               for rank in range(1, target + 2)]
        kept = ids[:target]
        if (type_, lang) == ("call", "zh"):
            second.append(review(ids[0], "drop"))
            kept = ids[1:target]
        elif (type_, lang) == ("call", "en"):
            second.append(review(ids[0], "keep"))
        expect[TEST_V2] += kept
        needed = TEST_FULL[type_, lang] - len(kept)
        kept_hw = []
        for n in range(needed + 1 + ((type_, lang) == DROPPED_HW)):
            hid = f"hw-test-{next(hw_numbers):03d}"
            handwritten.append(item(hid, type_, lang, "test", True))
            dropped = (type_, lang) == DROPPED_HW and n == 0
            hw_first.append(review(hid, "drop" if dropped else "keep"))
            hw_batches["001" if len(handwritten) <= 2 else "002"].append(hid)
            if not dropped:
                kept_hw.append(hid)
        expect[TEST_V2] += kept_hw[:needed]
        strata[f"test/{type_}/{lang}"] = {"full": TEST_FULL[type_, lang], "drafted_target": target,
                                          "kept": len(kept), "taken": len(kept), "needed": needed}
    review_dir = root / "review"
    write_rows(review_dir / "pool-ids.jsonl", rows)
    write_rows(review_dir / "verdicts.jsonl", first)
    write_rows(review_dir / "verdicts-double" / "batch-001.jsonl", second)
    write_rows(review_dir / "handwritten-verdicts.jsonl", hw_first)
    write_rows(review_dir / "handwritten-verdicts-double" / "batch-001.jsonl",
               [review(hid, "keep") for hid in hw_batches["001"]])
    for batch, hids in hw_batches.items():
        write_rows(review_dir / "handwritten-batches" / f"batch-{batch}.jsonl",
                   [{"id": hid, "type": "x", "lang": "y"} for hid in hids])
    (review_dir / "handwritten-needed.json").write_text(json.dumps({"strata": strata}, indent=1) + "\n",
                                                        encoding="utf-8", newline="\n")
    (review_dir / "instructions.md").write_text("# Review instructions\n", encoding="utf-8", newline="\n")
    write_rows(root / "handwritten-v2.jsonl", handwritten)
    (root / "handwritten-specs-v2.jsonl").write_text("{}\n", encoding="utf-8", newline="\n")
    for name in runs:
        write_rows(root / "runs" / name / "index.jsonl", [])
        (root / "runs" / name / "manifest.json").write_text(f'{{"run": "{name}"}}\n', encoding="utf-8")
    argv = ["--family", "v2", "--runs", str(root / "runs" / "a"), str(root / "runs" / "b"),
            "--review-dir", str(review_dir), "--handwritten", str(root / "handwritten-v2.jsonl"),
            "--handwritten-specs", str(root / "handwritten-specs-v2.jsonl"), "--out-dir", str(root / "out")]
    return {"argv": argv, "runs": runs, "expect": expect, "review": review_dir, "rows": rows, "first": first,
            "second": second, "handwritten": handwritten, "hw_first": hw_first, "out": root / "out"}


@pytest.fixture
def v2(tmp_path, monkeypatch) -> dict:
    """fabricate's inputs, with draft_items replaced by the fabricated drafts of each run (the generator runs
    are not needed); every call's arguments are recorded."""
    inputs = fabricate(tmp_path)
    calls = []

    def draft(run, **kwargs):
        calls.append((Path(run).name, kwargs))
        return list(inputs["runs"][Path(run).name])

    monkeypatch.setattr(finalize, "draft_items", draft)
    inputs["calls"] = calls
    return inputs


OUTPUTS = ("test-v2.jsonl", "dev-v2.jsonl", "dev-v2-ext.jsonl", "sets-manifest-v2.json")


def outputs(out: Path) -> dict[str, bytes | None]:
    return {name: (out / name).read_bytes() if (out / name).exists() else None for name in OUTPUTS}


def test_v2_writes_the_three_sets_and_their_manifest_deterministically(v2, capsys):
    assert finalize.main(v2["argv"]) == 0
    stdout = capsys.readouterr().out
    first = outputs(v2["out"])
    assert all(first.values())
    by_id = {i["id"]: i for items in v2["runs"].values() for i in items}
    by_id |= {h["id"]: h for h in v2["handwritten"]}
    for name in (TEST_V2, "dev-v2", "dev-v2-ext"):
        lines = first[f"{name}.jsonl"].decode("utf-8").splitlines()
        expected = sorted((by_id[x] for x in v2["expect"][name]), key=sort_key)
        assert lines == [canonical_json(i) for i in expected]
    # both runs drafted, validated requests only, as the review pool was drafted
    assert v2["calls"] == [("a", {"splits": ("dev", "test"), "include_unvalidated": False}),
                           ("b", {"splits": ("dev", "test"), "include_unvalidated": False})]
    assert finalize.main(v2["argv"]) == 0
    assert outputs(v2["out"]) == first
    manifest = json.loads(first["sets-manifest-v2.json"])
    sets = manifest["sets"]
    assert manifest["family"] == "v2" and list(sets) == ["dev-v2", "dev-v2-ext", TEST_V2]
    summary = {name: {k: v for k, v in info.items() if k != "strata"} for name, info in sets.items()}
    out = v2["out"]
    assert summary == {
        TEST_V2: {"split": "test", "items": 300, "quota": 300, "drafted": 209, "handwritten": 91,
                  "handwritten_share": 0.3033, "short": 0, "sha256": sha256_of(out / "test-v2.jsonl")},
        "dev-v2": {"split": "dev", "items": 139, "quota": 140, "drafted": 139, "handwritten": 0,
                   "handwritten_share": 0.0, "short": 1, "sha256": sha256_of(out / "dev-v2.jsonl")},
        "dev-v2-ext": {"split": "dev", "items": 65, "quota": 70, "drafted": 65, "handwritten": 0,
                       "handwritten_share": 0.0, "short": 5, "sha256": sha256_of(out / "dev-v2-ext.jsonl")}}
    test_strata = sets[TEST_V2]["strata"]
    assert test_strata["call/zh"] == record(60, 41, 41, target=42, handwritten_kept=20, handwritten=19)
    assert test_strata["call/en"] == record(60, 42, 42, target=42, handwritten_kept=19, handwritten=18)
    assert test_strata["no_call/en"] == record(25, 17, 17, target=17, handwritten_kept=9, handwritten=8)
    assert sets["dev-v2"]["strata"]["recover/zh"] == record(10, 9, 9)
    assert sets["dev-v2-ext"]["strata"]["recover/zh"] == record(5, 9, 0)
    assert sets["dev-v2-ext"]["strata"]["call/zh"] == record(14, 42, 14)
    for name, info in sets.items():
        assert set(info["strata"]) == {f"{t}/{lang}" for t in TYPES for lang in LANGS}
        assert sum(s["drafted"] for s in info["strata"].values()) == info["drafted"]
        assert sum(s["short"] for s in info["strata"].values()) == info["short"]
    review_dir = v2["review"]
    assert manifest["inputs"]["pool_ids"] == {"path": (review_dir / "pool-ids.jsonl").as_posix(),
                                              "sha256": sha256_of(review_dir / "pool-ids.jsonl")}
    assert [e["path"] for e in manifest["inputs"]["verdicts_double"]] == [
        (review_dir / "verdicts-double" / "batch-001.jsonl").as_posix()]
    assert [e["path"] for e in manifest["inputs"]["handwritten_batches"]] == [
        (review_dir / "handwritten-batches" / f"batch-{n}.jsonl").as_posix() for n in ("001", "002")]
    assert [e["path"] for e in manifest["inputs"]["runs"]] == [
        (out.parent / "runs" / n / "manifest.json").as_posix() for n in ("a", "b")]
    assert [e["path"] for e in manifest["inputs"]["run_indexes"]] == [
        (out.parent / "runs" / n / "index.jsonl").as_posix() for n in ("a", "b")]
    assert set(manifest["inputs"]) == {
        "pool_ids", "verdicts", "verdicts_double", "handwritten", "handwritten_verdicts",
        "handwritten_verdicts_double", "handwritten_batches", "handwritten_specs", "handwritten_needed",
        "runs", "run_indexes"}
    assert manifest["review"]["batches"] == {"drafted": ["batch-001", "batch-002"],
                                             "drafted_double": ["batch-001"],
                                             "handwritten": ["batch-001", "batch-002"],
                                             "handwritten_double": ["batch-001"]}
    assert manifest["review"]["kept"] == {
        "drafted": {"items": len(v2["rows"]), "reviewed_twice": 3, "agreeing": 2,
                    "kept": sum(r["verdict"] == "keep" for r in v2["first"]) - 1},
        "handwritten": {"items": len(v2["handwritten"]), "reviewed_twice": 2, "agreeing": 2,
                        "kept": len(v2["handwritten"]) - 1}}
    assert manifest["review"]["drafted"] == verdict_counts(v2["first"])
    assert manifest["review"]["drafted_double"] == verdict_counts(v2["second"])
    assert manifest["review"]["handwritten"] == verdict_counts(v2["hw_first"])
    assert manifest["review"]["instructions"] == {"path": (review_dir / "instructions.md").as_posix(),
                                                  "sha256": sha256_of(review_dir / "instructions.md")}
    assert "Workflow subagents" in manifest["review"]["reviewers"]
    assert "not written by the project owner" in manifest["handwritten_author"]
    assert "reviewed twice is kept only when both reviews keep it" in manifest["rule"]
    assert "e2e_tasks" not in manifest  # the v2 end-to-end tasks have their own lock and build record
    shas = {name: info["sha256"][:16] for name, info in summary.items()}
    assert stdout.splitlines()[:3] == [
        f"test-v2: 300 items of 300, drafted 209, handwritten 91 (30.3%), short 0, sha256 {shas[TEST_V2]}",
        f"dev-v2: 139 items of 140, drafted 139, handwritten 0 (0.0%), short 1, sha256 {shas['dev-v2']}",
        f"dev-v2-ext: 65 items of 70, drafted 65, handwritten 0 (0.0%), short 5, sha256 {shas['dev-v2-ext']}"]


def test_v2_writes_a_locked_set_again_only_with_the_bytes_its_lock_records(v2, capsys):
    assert finalize.main(v2["argv"]) == 0
    digest = write_lock(v2["out"] / "test-v2.jsonl")
    assert finalize.main(v2["argv"]) == 0  # the same inputs give the same bytes: the lock still holds
    assert check_lock(v2["out"] / "test-v2.jsonl") == digest
    before = outputs(v2["out"])
    capsys.readouterr()
    # the review now drops a test item it kept, and the count follows: test-v2 would change under its lock
    flipped_id = next(i for i in v2["expect"][TEST_V2] if i.startswith("req-test-"))
    flipped = [review(r["id"], "drop") if r["id"] == flipped_id else r for r in v2["first"]]
    write_rows(v2["review"] / "verdicts.jsonl", flipped)
    strata = json.loads((v2["review"] / "handwritten-needed.json").read_text(encoding="utf-8"))
    flipped_row = next(r for r in v2["rows"] if r["id"] == flipped_id)
    entry = strata["strata"][f"test/{flipped_row['type']}/{flipped_row['lang']}"]
    entry.update(kept=entry["kept"] - 1, taken=entry["taken"] - 1, needed=entry["needed"] + 1)
    (v2["review"] / "handwritten-needed.json").write_text(json.dumps(strata) + "\n", encoding="utf-8")
    assert finalize.main(v2["argv"]) == 3
    err = capsys.readouterr().err
    assert "test-v2.jsonl" in err and "test-v2.jsonl.sha256" in err and "nothing written" in err
    assert outputs(v2["out"]) == before
    (v2["out"] / "test-v2.jsonl.sha256").unlink()  # without its lock the set is written like any other file
    assert finalize.main(v2["argv"]) == 0 and outputs(v2["out"])["test-v2.jsonl"] != before["test-v2.jsonl"]


def mutate(v2: dict, change: str) -> None:
    """One defect in the fabricated inputs (test_v2_refuses_inputs_that_do_not_fit names them)."""
    review_dir, rows = v2["review"], v2["rows"]
    double = review_dir / "verdicts-double"
    test_id = next(r["id"] for r in rows if r["split"] == "test" and r["batch"] == "002")
    if change == "undrafted":
        for run in v2["runs"]:
            v2["runs"][run] = [i for i in v2["runs"][run] if i["id"] != test_id]
    elif change == "drafted-twice":
        v2["runs"]["b"].append(v2["runs"]["a"][0])
    elif change == "verdict-outside-pool":
        write_rows(review_dir / "verdicts.jsonl", [*v2["first"], review("req-test-9999-s1", "keep")])
    elif change == "unreviewed":
        write_rows(review_dir / "verdicts.jsonl", [r for r in v2["first"] if r["id"] != test_id])
    elif change == "double-outside-batch":
        write_rows(double / "batch-001.jsonl", [*v2["second"], review(test_id, "keep")])
    elif change == "double-of-no-batch":
        (double / "batch-001.jsonl").rename(double / "batch-009.jsonl")
    elif change == "row-without-batch":
        write_rows(review_dir / "pool-ids.jsonl",
                   [{k: v for k, v in r.items() if k != "batch"} for r in rows])
    elif change == "hw-verdict-outside":
        write_rows(review_dir / "handwritten-verdicts.jsonl",
                   [*v2["hw_first"], review("hw-test-999", "keep")])
    elif change == "hw-unreviewed":
        write_rows(review_dir / "handwritten-verdicts.jsonl", v2["hw_first"][:-1])
    elif change == "hw-batches":
        path = review_dir / "handwritten-batches" / "batch-002.jsonl"
        write_rows(path, read_jsonl(path)[:-1])
    elif change == "needed":
        path = review_dir / "handwritten-needed.json"
        counted = json.loads(path.read_text(encoding="utf-8"))
        counted["strata"]["dev/call/zh"]["kept"] += 1
        path.write_text(json.dumps(counted) + "\n", encoding="utf-8")
    elif change == "short":  # the last stratum, test parallel/en, keeps one handwritten item too few
        last_two = {r["id"] for r in v2["hw_first"][-2:]}
        write_rows(review_dir / "handwritten-verdicts.jsonl",
                   [review(r["id"], "drop") if r["id"] in last_two else r for r in v2["hw_first"]])
    else:
        raise AssertionError(change)


@pytest.mark.parametrize(("change", "message"), [
    ("undrafted", "is not drafted by the generator runs"),
    ("drafted-twice", "drafted by more than one run"),
    ("verdict-outside-pool", "judge ids outside the pool items, e.g. ['req-test-9999-s1']"),
    ("unreviewed", "pool items have no first review"),
    ("double-outside-batch", "batch-001.jsonl does not judge exactly the items"),
    ("double-of-no-batch", "batch-009.jsonl names no batch of"),
    ("row-without-batch", "has no batch"),
    ("hw-verdict-outside", "judge ids outside the handwritten items, e.g. ['hw-test-999']"),
    ("hw-unreviewed", "handwritten items have no first review"),
    ("hw-batches", "the handwritten batches do not hold every handwritten item exactly once"),
    ("needed", "dev/call/zh: the sets give"),
    ("short", "test/parallel/en needs 3 handwritten items the review kept and has 2"),
])
def test_v2_refuses_inputs_that_do_not_fit_with_exit_1_writing_nothing(v2, change, message, capsys):
    mutate(v2, change)
    assert finalize.main(v2["argv"]) == 1
    assert message in capsys.readouterr().err
    assert not v2["out"].exists()


@pytest.mark.parametrize(("error", "message"), [
    (HarnessError("t.json was recorded under tool schema 0"), "FAIL: HarnessError: t.json was recorded"),
    (FileNotFoundError("transcripts/t.json"), "FAIL: FileNotFoundError: transcripts/t.json")])
def test_v2_exits_1_when_a_run_cannot_be_drafted_writing_nothing(v2, monkeypatch, capsys, error, message):
    """A run the drafter refuses (another tool schema, a history that does not fit its steps) or one whose
    index names a missing transcript is malformed input: exit 1 with the reason, not a traceback."""

    def refuse(run, **kwargs):
        raise error

    monkeypatch.setattr(finalize, "draft_items", refuse)
    assert finalize.main(v2["argv"]) == 1
    assert capsys.readouterr().err.startswith(message) and not v2["out"].exists()


@pytest.mark.parametrize("missing", ["review/handwritten-needed.json", "review/verdicts-double",
                                     "review/handwritten-batches", "handwritten-v2.jsonl",
                                     "runs/b/index.jsonl"])
def test_v2_exits_2_on_a_missing_input_writing_nothing(v2, missing, capsys):
    path = v2["out"].parent / missing
    if path.is_dir():
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
    else:
        path.unlink()
    assert finalize.main(v2["argv"]) == 2
    assert "missing input" in capsys.readouterr().err and not v2["out"].exists()


@pytest.mark.parametrize("argv", [["--review-dir", "r"], ["--handwritten-specs", "s.jsonl"],
                                  ["--family", "v2", "--pool-ids", "p.jsonl"],
                                  ["--family", "v2", "--verdicts", "v.jsonl"],
                                  ["--family", "v2", "--handwritten-verdicts", "h.jsonl"],
                                  ["--family", "v3"]])
def test_the_options_of_one_family_are_a_usage_error_in_the_other(argv, capsys):
    with pytest.raises(SystemExit) as stopped:
        finalize.main(argv)
    assert stopped.value.code == 2


# ---- v1's mode, byte for byte as before v2 ----------------------------------------------------------------


def v1_fixture(root: Path) -> tuple[list[str], dict[str, list[dict]]]:
    """v1's inputs at the real quotas under root, the current folder, named by relative paths so that the
    manifest's paths and the printed lines do not depend on the temporary folder: per split and stratum
    target + 1 drafted items (the last one dropped) and full - target + 1 handwritten ones, all kept but the
    first of test recover/en, which is dropped (so that stratum takes all the others); the review's count, the
    handwritten specs and the review script where the script looks for them; and an end-to-end development
    file in the output folder. Every file is written with LF line endings, so that the inputs' SHA-256s the
    manifest records are the same on every platform. Returns the arguments and each run's drafted items."""
    drafted: dict[str, list[dict]] = {"dev": [], "test": []}
    rows, verdicts, handwritten, hw_verdicts = [], [], [], []
    for split, quotas in (("dev", QUOTAS_DEV), ("test", QUOTAS_TEST)):
        full, target = per_language(quotas), per_language(scaled(quotas, 0.7))
        for (type_, lang), size in full.items():
            for n in range(target[type_, lang] + 1):
                d = item(f"req-{split}-{type_}-{lang}-{n:03d}-s1", type_, lang, split)
                drafted[split].append(d)
                rows.append({"split": split, "type": type_, "lang": lang, "rank": n + 1, "id": d["id"]})
                verdicts.append({"id": d["id"], "verdict": "keep" if n < target[type_, lang] else "drop",
                                 "category": "ok" if n < target[type_, lang] else "ambiguous",
                                 "model": "claude-opus-5-5", "batch": "b1"})
            extra = (split, type_, lang) == ("test", "recover", "en")
            for n in range(size - target[type_, lang] + 1 + extra):
                h = item(f"hw-{split}-{type_}-{lang}-{n:03d}", type_, lang, split, handwritten=True)
                handwritten.append(h)
                hw_verdicts.append({"id": h["id"], "verdict": "drop" if extra and n == 0 else "keep",
                                    "category": "other" if extra and n == 0 else "ok",
                                    "model": "claude-opus-5-5", "batch": "b2"})
    write_rows(root / "review" / "pool-ids.jsonl", rows)
    write_rows(root / "review" / "verdicts.jsonl", verdicts)
    write_rows(root / "hand" / "handwritten.jsonl", handwritten)
    write_rows(root / "review" / "handwritten-verdicts.jsonl", hw_verdicts)
    write_rows(root / "data" / "eval" / "handwritten-specs.jsonl", [{"id": "hw-dev-x"}])
    (root / "data" / "eval" / "review").mkdir(parents=True)
    (root / "data" / "eval" / "review" / "handwritten-needed.json").write_text('{"strata": {}}\n',
                                                                               encoding="utf-8", newline="\n")
    (root / "tools" / "eval-review").mkdir(parents=True)
    (root / "tools" / "eval-review" / "review_items.py").write_text("RUBRIC = 'x'\n", encoding="utf-8",
                                                                    newline="\n")
    for name in ("dev", "test"):
        write_rows(root / "runs" / name / "index.jsonl", [])
        (root / "runs" / name / "manifest.json").write_text(f'{{"run": "{name}"}}\n', encoding="utf-8",
                                                            newline="\n")
    write_rows(root / "out" / "e2e-dev.jsonl", [{"id": "e2e-dev-01"}])
    argv = ["--pool-ids", "review/pool-ids.jsonl", "--verdicts", "review/verdicts.jsonl",
            "--handwritten", "hand/handwritten.jsonl",
            "--handwritten-verdicts", "review/handwritten-verdicts.jsonl",
            "--runs", "runs/dev", "runs/test", "--out-dir", "out"]
    return argv, drafted


# Computed with scripts/finalize_eval_sets.py and research_agent of commit d80cb3a (before v2) on v1_fixture
# (its LF inputs): the SHA-256 of the three files it writes, and what it prints on success, on a short stratum
# (exit 1) and on a missing input (exit 2).
V1_GOLDEN = {
    "dev.jsonl": "09177c71b03e049325c9a2e2438652395916e5cfb5a2e942cd5e64fbd50a7273",
    "test.jsonl": "c1bf0ea18a58b491c2d219e9821e7bde20acbca89b44f895b2091a32563012fb",
    "sets-manifest.json": "0c3f992558a768d22849fc22a934d8ceb11e756d74405336baee2538411e8ae0",
    "stdout": "dev: 150 items, drafted 105, handwritten 45 (30.0%), sha256 09177c71b03e0493\n"
              "test: 300 items, drafted 210, handwritten 90 (30.0%), sha256 c1bf0ea18a58b491\n"
              "manifest out/sets-manifest.json\n",
    "short": "FAIL: SetError: test/parallel/zh needs 3 handwritten items the review kept and has 0\n",
    "missing": "missing input: review/verdicts.jsonl\n",
}


def test_v1_mode_writes_and_prints_byte_for_byte_what_it_did_before_v2(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    argv, drafted = v1_fixture(tmp_path)
    monkeypatch.setattr(finalize, "draft_items", lambda run, **kwargs: drafted[Path(run).name])
    assert finalize.main(argv) == 0
    got = {name: sha256_of(tmp_path / "out" / name)
           for name in ("dev.jsonl", "test.jsonl", "sets-manifest.json")}
    got["stdout"] = capsys.readouterr().out
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "dev.jsonl", "e2e-dev.jsonl", "sets-manifest.json", "test.jsonl"]
    manifest = json.loads((tmp_path / "out" / "sets-manifest.json").read_text(encoding="utf-8"))
    # the fixture reaches every branch of the manifest: the optional inputs, the review script, the e2e file
    assert {"handwritten_specs", "handwritten_needed"} <= set(manifest["inputs"])
    assert manifest["review"]["script"] and list(manifest["e2e_tasks"]) == ["e2e-dev.jsonl"]
    verdicts = tmp_path / "review" / "handwritten-verdicts.jsonl"
    rows = read_jsonl(verdicts)
    write_rows(verdicts, [{**r, "verdict": "drop"} if r["id"].startswith("hw-test-parallel-zh") else r
                          for r in rows])
    assert finalize.main(argv) == 1
    got["short"] = capsys.readouterr().err
    (tmp_path / "review" / "verdicts.jsonl").unlink()
    assert finalize.main(argv) == 2
    got["missing"] = capsys.readouterr().err
    assert got == V1_GOLDEN


# ---- the files the v2 build wrote: data/eval/v2 (test-v2 with --test-material) -----------------------------

V2_DIR = REPO / "data" / "eval" / "v2"
REVIEW = V2_DIR / "review"
FILES = {TEST_V2: V2_DIR / "test-v2.jsonl", "dev-v2": V2_DIR / "dev-v2.jsonl",
         "dev-v2-ext": V2_DIR / "dev-v2-ext.jsonl"}
MANIFEST_V2 = V2_DIR / "sets-manifest-v2.json"
HANDWRITTEN_V2 = V2_DIR / "handwritten-v2.jsonl"
SPLIT = {TEST_V2: "test", "dev-v2": "dev", "dev-v2-ext": "dev"}
SIZES = {TEST_V2: 300, "dev-v2": 126, "dev-v2-ext": 23}
DRAFTED = {TEST_V2: 192, "dev-v2": 126, "dev-v2-ext": 23}
QUOTA = {TEST_V2: 300, "dev-v2": 140, "dev-v2-ext": 70}
DEV_SETS = ("dev-v2", "dev-v2-ext")
# test-v2 is locked test material: its checks run only with --test-material (tests/conftest.py); the
# development sets, their locks, the manifest and the review are data the public copy leaves out
# (tests/conftest.py, needs_data: all of it is in data/eval/v2)
NEEDS_V2 = pytest.mark.needs_data("data/eval/v2")
NEEDS_V1_AND_V2 = pytest.mark.needs_data("data/eval", "data/eval/v2")  # v1's sets and review records too
EVERY_SET = [pytest.param(TEST_V2, marks=pytest.mark.test_material),
             *(pytest.param(name, marks=NEEDS_V2) for name in DEV_SETS)]
SYNTH = REPO / "data" / "synth"
TRAINING = [SYNTH / name for name in ("requests.jsonl", "requests-m6.jsonl", "requests-m7-train.jsonl")]
GENERATED = [SYNTH / name for name in ("requests.jsonl", "requests-m6.jsonl", "requests-v2.jsonl",
                                       "requests-v2b.jsonl", "requests-m7-train.jsonl")]
V2_REQUESTS = [SYNTH / "requests-v2.jsonl", SYNTH / "requests-v2b.jsonl"]
LOGS = (TEST_ACCESS_LOG, E2E_TEST_ACCESS_LOG, TEST_COMPLETIONS, TEST_ACCESS_LOG_V2, E2E_TEST_ACCESS_LOG_V2,
        TEST_COMPLETIONS_V2)


def opened(name: str) -> list[dict]:
    """A set's items, lock-checked: the development sets through DevSet, test-v2 by its lock and bytes (only
    in a --test-material run; never through LockedSet, whose open would go to its access log)."""
    if name in DEV_SETS:
        return DevSet(FILES[name]).open()
    check_lock(FILES[name])
    return read_jsonl(FILES[name])


def logs() -> dict[str, bytes | None]:
    """Both families' access logs and completion markers, as bytes (None where there is none yet)."""
    return {p.as_posix(): (REPO / p).read_bytes() if (REPO / p).exists() else None for p in LOGS}


def counted() -> dict:
    return json.loads((REVIEW / "handwritten-needed.json").read_text(encoding="utf-8"))["strata"]


def both_keep(first: Path, second: Path) -> dict[str, bool]:
    """id -> kept: every review of it keeps it (the first reviews in first, the second ones in the files of
    the folder second); worked out here, apart from eval.sets."""
    verdicts: dict[str, list[str]] = defaultdict(list)
    for path in (first, *sorted(second.glob("*.jsonl"))):
        for row in read_jsonl(path):
            verdicts[row["id"]].append(row["verdict"])
    return {i: all(v == "keep" for v in vs) for i, vs in verdicts.items()}


def stratum(i: dict) -> str:
    return f"{i['type']}/{i['lang']}"


def is_handwritten(i: dict) -> bool:
    return bool((i.get("source") or {}).get("handwritten"))


@pytest.mark.parametrize("name", EVERY_SET)
def test_each_set_has_its_size_its_split_and_the_strata_the_review_counted(name):
    items = opened(name)
    lines = FILES[name].read_text(encoding="utf-8").splitlines()
    assert lines == [canonical_json(i) for i in items]  # canonical lines, as v1's sets
    assert items == sorted(items, key=sort_key)
    assert len(items) == SIZES[name] and {i["split"] for i in items} == {SPLIT[name]}
    assert sum(not is_handwritten(i) for i in items) == DRAFTED[name]
    strata = counted()
    full = per_language(QUOTAS_TEST if name == TEST_V2 else DEV_QUOTAS_V2[name])
    drafted = Counter(stratum(i) for i in items if not is_handwritten(i))
    handwritten = Counter(stratum(i) for i in items if is_handwritten(i))
    manifest = json.loads(MANIFEST_V2.read_text(encoding="utf-8"))["sets"][name]
    for type_ in TYPES:
        for lang in LANGS:
            key, entry = f"{type_}/{lang}", strata[f"{SPLIT[name]}/{type_}/{lang}"]
            if name == TEST_V2:  # every stratum full: the drafted items taken, then the handwritten ones
                assert (drafted[key], handwritten[key]) == (entry["taken"], entry["needed"]), key
                assert drafted[key] + handwritten[key] == full[type_, lang] == entry["full"], key
            else:
                assert (drafted[key], handwritten[key]) == (entry[name]["taken"], 0), key
                assert full[type_, lang] == entry[name]["quota"], key
            assert (manifest["strata"][key]["drafted"], manifest["strata"][key]["handwritten"]) == (
                drafted[key], handwritten[key]), key
    assert (manifest["items"], manifest["drafted"], manifest["quota"], manifest["short"]) == (
        SIZES[name], DRAFTED[name], QUOTA[name], QUOTA[name] - SIZES[name])
    assert manifest["sha256"] == read_lock(FILES[name]) == sha256_of(FILES[name])
    if name in DEV_SETS:
        assert find_test_material(items) == []  # nothing a runner would refuse as test material
    else:
        assert find_test_material(items) == [i["id"] for i in items]  # every item, by its split and its id


@pytest.fixture(scope="module")
def v1_ids() -> dict[str, set[str]]:
    """The ids of v1's sets by ids alone; v1's test set is never opened. Its dev and test sets are assembled
    again from v1's review records with eval.sets.assemble on stubs (ids and strata, no item text), from the
    very files data/eval/sets-manifest.json records: the stubs must give the strata the manifest records for
    both splits and exactly the ids of data/eval/dev.jsonl. dev-ext's ids come from its file; "review" holds
    every id v1's review judged, drafted and handwritten, a superset of all three sets."""
    v1 = REPO / "data" / "eval"
    manifest = json.loads((v1 / "sets-manifest.json").read_text(encoding="utf-8"))
    for key in ("pool_ids", "verdicts", "handwritten_verdicts"):
        assert sha256_of(REPO / manifest["inputs"][key]["path"]) == manifest["inputs"][key]["sha256"], key
    pool = read_jsonl(v1 / "review" / "pool-ids.jsonl")
    hand = read_jsonl(v1 / "review" / "handwritten-verdicts.jsonl")

    def stub(row: dict) -> dict:
        return {k: row[k] for k in ("id", "split", "type", "lang")}

    verdicts = {r["id"]: r["verdict"] for r in read_jsonl(v1 / "review" / "verdicts.jsonl")}
    sets, table = assemble(pool, verdicts, {r["id"]: stub(r) for r in pool}, [stub(r) for r in hand],
                           {r["id"]: r["verdict"] for r in hand})
    assert table == {split: manifest["splits"][split]["strata"] for split in ("dev", "test")}
    dev = {i["id"] for i in DevSet(REPO / DEV_SET).open()}
    assert {i["id"] for i in sets["dev"]} == dev and len(sets["test"]) == manifest["splits"]["test"]["items"]
    return {"dev": dev, "dev-ext": {i["id"] for i in DevSet(v1 / "dev-ext.jsonl").open()},
            "test": {i["id"] for i in sets["test"]},
            "review": {r["id"] for r in pool} | {r["id"] for r in hand}}


@pytest.mark.parametrize("names", [pytest.param(DEV_SETS, marks=NEEDS_V1_AND_V2),
                                   pytest.param((TEST_V2, *DEV_SETS), marks=pytest.mark.test_material)],
                         ids=["dev-sets", "all-sets"])
def test_ids_are_unique_across_the_sets_and_apart_from_v1s_sets(names, v1_ids):
    ids = [i["id"] for name in names for i in opened(name)]
    assert len(ids) == len(set(ids)) == sum(SIZES[name] for name in names)
    for v1_set, v1 in v1_ids.items():
        assert not set(ids) & v1, v1_set


@pytest.mark.parametrize("name", EVERY_SET)
def test_every_item_is_one_the_review_kept_taken_in_the_review_pools_order(name):
    """The drafted items: pool items of the set's split and stratum that every review kept, the stratum's
    first kept ranks (test-v2 up to its drafted target, dev-v2 up to its quota, dev-v2-ext the next ones up to
    its own). The handwritten items (test-v2 alone): items of handwritten-v2.jsonl as they are there, kept by
    every review, the stratum's lowest kept ids."""
    items = opened(name)
    rows = {r["id"]: r for r in read_jsonl(REVIEW / "pool-ids.jsonl")}
    kept = both_keep(REVIEW / "verdicts.jsonl", REVIEW / "verdicts-double")
    ranked: dict[str, list[str]] = defaultdict(list)
    for row in sorted(rows.values(), key=lambda r: r["rank"]):
        if row["split"] == SPLIT[name] and kept[row["id"]]:
            ranked[f"{row['type']}/{row['lang']}"].append(row["id"])
    drafted: dict[str, list[str]] = defaultdict(list)
    for i in items:
        if is_handwritten(i):
            continue
        row = rows[i["id"]]
        assert (row["split"], row["type"], row["lang"]) == (i["split"], i["type"], i["lang"]), i["id"]
        assert kept[i["id"]] and i["source"]["request"] == i["id"].rsplit("-s", 1)[0], i["id"]
        drafted[stratum(i)].append(i["id"])
    strata = counted()
    keys = [f"{type_}/{lang}" for type_ in TYPES for lang in LANGS]
    for key in keys:
        if name == TEST_V2:
            start, size = 0, strata[f"test/{key}"]["drafted_target"]
        else:
            size = strata[f"dev/{key}"][name]["quota"]
            start = 0 if name == "dev-v2" else strata[f"dev/{key}"]["dev-v2"]["quota"]
        assert sorted(drafted[key], key=ranked[key].index) == ranked[key][start:start + size], key
    handwritten = [i for i in items if is_handwritten(i)]
    if name in DEV_SETS:
        assert handwritten == []
        return
    written = {h["id"]: h for h in read_jsonl(HANDWRITTEN_V2)}
    hw_kept = both_keep(REVIEW / "handwritten-verdicts.jsonl", REVIEW / "handwritten-verdicts-double")
    assert set(hw_kept) == set(written)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for h in handwritten:
        assert h == written[h["id"]] and hw_kept[h["id"]], h["id"]
        by_stratum[stratum(h)].append(h["id"])
    for key in keys:
        lowest = sorted(i for i, h in written.items() if stratum(h) == key and hw_kept[i])
        assert sorted(by_stratum[key]) == lowest[:strata[f"test/{key}"]["needed"]], key


@pytest.fixture(scope="module")
def drafted_v2() -> dict[str, dict]:
    """id -> item as the drafter writes it from both generator runs (validated requests only). runs/ is not
    committed, so the check is skipped where the runs are absent."""
    runs = [REPO / run for run in finalize.RUNS_V2]
    if not all(run.is_dir() for run in runs):
        pytest.skip("the generator runs runs/m7/synth/v2 and v2b are not here")
    return finalize.draft_runs(runs)


@pytest.mark.parametrize("name", EVERY_SET)
def test_every_drafted_item_is_the_drafters_item_unchanged(name, drafted_v2):
    """A drafted item edited after the build (and its set locked again) fails here; only ids are reported."""
    changed = [i["id"] for i in opened(name) if not is_handwritten(i) and i != drafted_v2[i["id"]]]
    assert changed == []


class Shingled:
    """Masked 5-gram shingle sets of request texts with an inverted index: largest(text) is the largest
    Jaccard of text against any of them (data.requests: jaccard over shingles(mask_text(...)), exact)."""

    def __init__(self, texts: list[str]) -> None:
        self.sizes: list[int] = []
        self.postings: dict[str, list[int]] = defaultdict(list)
        for n, text in enumerate(texts):
            grams = shingles(mask_text(text))
            self.sizes.append(len(grams))
            for gram in grams:
                self.postings[gram].append(n)

    def largest(self, text: str) -> float:
        grams = shingles(mask_text(text))
        shared: Counter[int] = Counter()
        for gram in grams:
            shared.update(self.postings.get(gram, ()))
        return max((n / (len(grams) + self.sizes[i] - n) for i, n in shared.items()), default=0.0)


@pytest.fixture(scope="module")
def requests_index() -> dict:
    training = [r["text"] for path in TRAINING for r in read_jsonl(path) if r["split"] == "train"]
    generated = [r["text"] for path in GENERATED for r in read_jsonl(path)]
    v2_requests = {r["id"]: r for path in V2_REQUESTS for r in read_jsonl(path)}
    assert len(training) > 3000 and len(generated) > len(training)
    return {"training": Shingled(training), "generated": Shingled(generated), "v2": v2_requests}


def user_request(i: dict) -> str:
    """An item's text: its user's request, the first user message of its history (eval.lock reads a
    single-step item's text the same way)."""
    return next(m["content"] for m in i["messages"] if m.get("role") == "user")


@pytest.mark.needs_data("data/synth")  # the training and generated requests
@pytest.mark.parametrize("name", EVERY_SET)
def test_no_item_is_near_a_training_request(name, requests_index):
    """The leak rule of data.requests (masked 5-gram Jaccard) on the assembled sets: every item's request as
    the model sees it, and a drafted item's request as it was generated (its record in requests-v2.jsonl or
    requests-v2b.jsonl, of the item's split), stay below TRAIN_VS_EVAL (0.6) against every training request,
    so below NEAR_DUP (0.8) as well; a handwritten item's request stays below TEMPLATE_LIMIT (0.6) against
    every generated request of every split, check 10 of eval.handwritten."""
    training, generated, v2_requests = (requests_index[k] for k in ("training", "generated", "v2"))
    for i in opened(name):
        assert training.largest(user_request(i)) < TRAIN_VS_EVAL, i["id"]
        if is_handwritten(i):
            assert generated.largest(user_request(i)) < TEMPLATE_LIMIT, i["id"]
            continue
        request = v2_requests[i["source"]["request"]]
        assert request["split"] == i["split"] and training.largest(request["text"]) < TRAIN_VS_EVAL, i["id"]


@NEEDS_V2
def test_the_manifest_records_the_review_and_its_inputs_as_they_are_now():
    manifest = json.loads(MANIFEST_V2.read_text(encoding="utf-8"))
    inputs = manifest["inputs"]
    for key in ("pool_ids", "verdicts", "handwritten_verdicts", "handwritten_needed"):
        assert sha256_of(REPO / inputs[key]["path"]) == inputs[key]["sha256"], key
    for entry in (*inputs["verdicts_double"], *inputs["handwritten_verdicts_double"]):
        assert sha256_of(REPO / entry["path"]) == entry["sha256"], entry["path"]
    assert [Path(e["path"]).name for e in inputs["verdicts_double"]] == [
        f"batch-{n:03d}.jsonl" for n in (1, 11, 21, 31, 41, 44)]
    assert manifest["review"]["kept"] == {
        "drafted": {"items": 829, "reviewed_twice": 108, "agreeing": 102, "kept": 373},
        "handwritten": {"items": 137, "reviewed_twice": 18, "agreeing": 18, "kept": 135}}
    assert manifest["review"]["batches"]["drafted"] == [f"batch-{n:03d}" for n in range(1, 49)]
    assert manifest["review"]["batches"]["handwritten"] == [f"batch-{n:03d}" for n in range(1, 9)]
    for name, path in FILES.items():
        assert manifest["sets"][name]["sha256"] == read_lock(path), name  # the lock, not the set, is read
    counts = {name: (s["items"], s["drafted"], s["handwritten"], s["short"])
              for name, s in manifest["sets"].items()}
    assert counts == {TEST_V2: (300, 192, 108, 0), "dev-v2": (126, 126, 0, 14), "dev-v2-ext": (23, 23, 0, 47)}


@pytest.mark.test_material
def test_the_manifest_records_the_handwritten_files_as_they_are_now():
    inputs = json.loads(MANIFEST_V2.read_text(encoding="utf-8"))["inputs"]
    for key in ("handwritten", "handwritten_specs"):
        assert sha256_of(REPO / inputs[key]["path"]) == inputs[key]["sha256"], key
    for entry in inputs["handwritten_batches"]:
        assert sha256_of(REPO / entry["path"]) == entry["sha256"], entry["path"]


# ---- the guards: test-v2 for run_test alone; run_test's v2 development sets --------------------------------


@pytest.fixture
def forgive():
    """Take the refusals this test provokes on purpose off the guard's record, as
    tests/test_test_material_guard.py does."""
    before = len(conftest.REFUSED)
    yield
    del conftest.REFUSED[before:]


def test_test_v2_is_refused_to_every_runner_but_run_test_without_being_opened(tmp_path, monkeypatch, capsys,
                                                                             forgive):
    before = logs()
    monkeypatch.chdir(REPO)
    for spelling in ("data/eval/v2/test-v2.jsonl", "data/eval/v2/../v2/test-v2.jsonl", FILES[TEST_V2]):
        with pytest.raises(RefusedTestSet, match=r"is the locked test set .*data/eval/v2/test-v2\.jsonl"):
            refuse_test_sets(spelling, [Path.cwd(), REPO])
    run_eval = load_script("run_eval")
    out = tmp_path / "r.jsonl"
    assert run_eval.main(["data/eval/v2/test-v2.jsonl", "--system", "replay", "--out", str(out),
                          "--no-tokenizer"]) == 4
    assert "locked test set" in capsys.readouterr().err and not out.exists()
    # a test that is not marked test_material cannot open it either (tests/conftest.py)
    with pytest.raises(conftest.MaterialOpenRefused):
        open(FILES[TEST_V2], "rb").close()
    assert logs() == before


@pytest.mark.test_material
def test_a_copy_of_test_v2_is_refused_by_its_bytes(tmp_path):
    copy = tmp_path / "copy.jsonl"
    copy.write_bytes(FILES[TEST_V2].read_bytes())
    with pytest.raises(RefusedTestSet, match="SHA-256 of the locked test set"):
        refuse_test_sets(copy, [REPO])


@pytest.mark.needs_data("data/eval/dev.jsonl", "data/eval/dev.jsonl.sha256", "data/eval/v2")
def test_run_tests_v2_family_finds_its_development_sets_dev_and_dev_v2(monkeypatch):
    run_test = load_script("run_test")
    monkeypatch.chdir(REPO)
    before = logs()
    dev_sets = run_test.FAMILY_DEV_SETS["v2"]
    assert dev_sets == (Path("data/eval/dev.jsonl"), Path("data/eval/v2/dev-v2.jsonl"))
    assert all(path.is_file() for path in dev_sets)  # the preflight's check (run_test.prerequisites)
    material = run_test.development_material(dev_sets)  # each set lock-checked through DevSet
    assert material.sets["single_step"]["sets"] == [
        {"path": "data/eval/dev.jsonl", "sha256": read_lock(REPO / "data/eval/dev.jsonl")},
        {"path": "data/eval/v2/dev-v2.jsonl", "sha256": read_lock(FILES["dev-v2"])}]
    assert len(material.full) == 150 + SIZES["dev-v2"]
    assert material.opens == {"single_step": None, "e2e": None}
    assert logs() == before

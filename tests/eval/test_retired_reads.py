"""The retired reads of v1's single-step test items and of its hand-written specs
(eval.lock.RetiredSet.single_step and RetiredSet.handwritten_specs; docs/m7-计划.md WP-I 4): one similarity
per text and nothing else, every read logged in a reads log of its own, no access log written. A folder of
fixtures laid out as the repository is stands in for v1's files, which a test run may not open
(tests/conftest.py)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_agent.canonical import canonical_json
from research_agent.data.requests import jaccard, mask_text, shingles
from research_agent.eval import lock as lock_module
from research_agent.eval.lock import (
    E2E_TEST_ACCESS_LOG,
    E2E_TEST_ACCESS_LOG_V2,
    E2E_TEST_RETIRED_READS,
    EVAL_DIR,
    HANDWRITTEN_SETS,
    HANDWRITTEN_SPECS,
    HANDWRITTEN_SPECS_RETIRED_READS,
    SETS_MANIFEST,
    TEST_ACCESS_LOG,
    TEST_ACCESS_LOG_V2,
    TEST_RETIRED_READS,
    TEST_SET,
    TEST_SETS,
    LockError,
    RefusedTestSet,
    RetiredSet,
    check_recorded,
    read_jsonl,
    refuse_test_sets,
    sha256_of,
    write_jsonl,
    write_lock,
)

REPO = Path(__file__).resolve().parents[2]
SYSTEM = {"role": "system", "content": "你是科研助理,今天是 2026-10-10。按需调用工具,参数必须符合工具说明。"}
GET_PAPER = {"id": "c1", "type": "function",
             "function": {"name": "get_paper", "arguments": '{"arxiv_id": "2609.05022"}'}}
# v1-style single-step test items: the history holds the system prompt, the user's request and, for a later
# step, the calls and tool outputs before it; only the request counts.
ITEMS = [
    {"id": "req-test-0007-s1", "split": "test", "type": "call", "lang": "en",
     "family": "find_recent_papers/en", "slots": {"topic": "tokenizer", "count": 3},
     "messages": [SYSTEM, {"role": "user", "content": "Find three tokenizer papers posted since 2026-08-01 "
                                                      "and rank them by relevance for me."}],
     "expected": {"content": "", "tool_calls": []}, "source": {"request": "req-test-0007", "step": 1}},
    {"id": "req-test-0012-s2", "split": "test", "type": "next_step", "lang": "zh",
     "family": "read_section/zh", "slots": {"arxiv_id": "2609.05022", "section": "experiments"},
     "messages": [SYSTEM, {"role": "user",
                           "content": "帮我看看 2609.05022 的实验部分,作者到底用了哪些数据集?"},
                  {"role": "assistant", "content": "", "tool_calls": [GET_PAPER]},
                  {"role": "tool", "tool_call_id": "c1",
                   "content": '{"title": "Depth schedules for looped transformers on algorithmic data"}'}],
     "expected": {"content": "The abstract describes depth schedules for looped transformers on algorithmic "
                             "data at some length.", "tool_calls": []},
     "source": {"request": "req-test-0012", "step": 2}},
    {"id": "hw-test-001", "split": "test", "type": "no_call", "lang": "en",
     "family": "handwritten/no_call/en", "slots": {},
     "messages": [SYSTEM, {"role": "user",
                           "content": "That is all for today, thanks a lot and see you soon!"}],
     "expected": {"content": "You are welcome; see you soon!", "tool_calls": []},
     "source": {"handwritten": True}},
]
# v1-style hand-written specs of both splits: only the request counts.
SPECS = [
    {"id": "hw-dev-001", "split": "dev", "lang": "en", "type": "call",
     "request": "Look on arXiv for recent work about mixture of experts routing, please.",
     "slots": {"topic": "mixture of experts", "topic_id": "mixture-of-experts"}, "prior": [],
     "expected": {"tool_calls": [{"name": "search_arxiv",
                                  "arguments": {"query": "mixture of experts routing"}}]},
     "rationale": "The request names its topic and nothing else, so one search on it is the single first "
                  "step."},
    {"id": "hw-test-002", "split": "test", "lang": "zh", "type": "no_call",
     "request": "今天的讨论就到这里,谢谢你一直帮我整理论文!", "slots": {}, "prior": [],
     "expected": {"content": "不客气,随时找我。"},
     "rationale": "Small talk asks for no tool, so a short friendly reply is the single fitting answer."},
]
SPENT = [{"milestone": m, "purpose": "p", "sha256": "0" * 64, "ts": "2026-10-17T00:00:00+00:00"}
         for m in ("M3", "M5", "M6")]
STAMPS = ["2026-10-26T08:00:00+00:00", "2026-10-26T09:00:00+00:00", "2026-10-26T10:00:00+00:00",
          "2026-10-26T11:00:00+00:00"]
REAL_ACCESS_LOGS = (TEST_ACCESS_LOG, E2E_TEST_ACCESS_LOG, TEST_ACCESS_LOG_V2, E2E_TEST_ACCESS_LOG_V2)


def clock(*stamps: str):
    """now() stand-in: the given timestamps in order."""
    it = iter(stamps)
    return lambda: next(it)


def write_v1(root: Path, items: list[dict] = ITEMS, specs: list[dict] = SPECS) -> Path:
    """v1's retired files under root, laid out as in the repository: the single-step test set with its lock
    and an access log holding its three opens, and the hand-written specs with the assembly's manifest
    recording their SHA-256 (they have no lock of their own)."""
    (root / EVAL_DIR).mkdir(parents=True, exist_ok=True)
    write_jsonl(items, root / TEST_SET)
    write_lock(root / TEST_SET)
    write_jsonl(SPENT, root / TEST_ACCESS_LOG)
    write_jsonl(specs, root / HANDWRITTEN_SPECS)
    record = {"path": HANDWRITTEN_SPECS.as_posix(), "sha256": sha256_of(root / HANDWRITTEN_SPECS)}
    (root / SETS_MANIFEST).write_text(json.dumps({"inputs": {"handwritten_specs": record}}, indent=1) + "\n",
                                      encoding="utf-8")
    return root


@pytest.fixture
def root(tmp_path):
    return write_v1(tmp_path)


def largest(text: str, others: list[str]) -> float:
    """The contract tests' measure: the largest masked 5-gram Jaccard of text against others."""
    mine = shingles(mask_text(text))
    return max(jaccard(mine, shingles(mask_text(other))) for other in others)


def line_count(path: Path) -> int | None:
    return len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else None


REQUESTS_OF_ITEMS = [item["messages"][1]["content"] for item in ITEMS]
REQUESTS_OF_SPECS = [spec["request"] for spec in SPECS]


# ---- what a read returns -------------------------------------------------------------------------------


def test_a_single_step_read_scores_each_text_against_the_items_user_requests_alone(root):
    texts = ["Find four tokenizer papers posted since 2026-09-15 and rank them by relevance for me!",
             REQUESTS_OF_ITEMS[1],
             ITEMS[1]["expected"]["content"],  # in an item, but no request: not scored as one
             ITEMS[1]["messages"][3]["content"],  # a tool output
             SYSTEM["content"],  # the system prompt
             ""]
    retired = RetiredSet.single_step(root, now=clock(*STAMPS))
    scores = retired.similarities(texts)
    assert type(scores) is list and all(type(s) is float for s in scores)
    assert scores == [largest(text, REQUESTS_OF_ITEMS) for text in texts]
    assert scores[0] >= 0.6 and scores[1] == 1.0
    assert all(score < 0.5 for score in scores[2:]), scores
    held = repr(vars(retired))  # nothing of the set stays on the object
    assert not [text for text in REQUESTS_OF_ITEMS if text[:12] in held]


def test_only_the_first_user_message_of_an_item_is_its_request(tmp_path):
    """A v1 item holds one user message, the request; a later one (a follow-up in a history) would not
    count."""
    later = {"role": "user", "content": "Also list my notes tagged survey from 2026-09-01, five of them."}
    item = {**ITEMS[0], "messages": [*ITEMS[0]["messages"], {"role": "assistant", "content": "Done."}, later]}
    root = write_v1(tmp_path, items=[item])
    retired = RetiredSet.single_step(root, now=clock(*STAMPS))
    scores = retired.similarities([REQUESTS_OF_ITEMS[0], later["content"]])
    assert scores == [1.0, largest(later["content"], REQUESTS_OF_ITEMS[:1])] and scores[1] < 0.3


def test_a_specs_read_scores_each_text_against_the_requests_of_both_splits_alone(root):
    texts = [REQUESTS_OF_SPECS[0], REQUESTS_OF_SPECS[1], SPECS[0]["rationale"],
             "Look on arXiv for recent papers about mixture of experts routing, please!"]
    scores = RetiredSet.handwritten_specs(root, now=clock(*STAMPS)).similarities(texts)
    assert scores == [largest(text, REQUESTS_OF_SPECS) for text in texts]
    assert scores[:2] == [1.0, 1.0]  # a dev spec's request counts as a test spec's does
    assert scores[2] < 0.3 and 0.6 <= scores[3] < 1.0, scores


def test_each_read_logs_one_line_in_its_own_reads_log_and_never_a_text_or_a_field(root):
    asked = ["Find tokenizer papers since 2026-08-01, three of them.", "See you on Monday, and thanks!"]
    RetiredSet.single_step(root, now=clock(*STAMPS)).similarities(asked)
    specs = RetiredSet.handwritten_specs(root, now=clock(*STAMPS[1:]))
    specs.similarities(asked)
    specs.similarities(asked[:1])
    test_digest, specs_digest = sha256_of(root / TEST_SET), sha256_of(root / HANDWRITTEN_SPECS)
    assert read_jsonl(root / TEST_RETIRED_READS) == [
        {"lock_ok": True, "sha256": test_digest, "texts": 2, "ts": STAMPS[0]}]
    assert read_jsonl(root / HANDWRITTEN_SPECS_RETIRED_READS) == [
        {"lock_ok": True, "sha256": specs_digest, "texts": 2, "ts": STAMPS[1]},
        {"lock_ok": True, "sha256": specs_digest, "texts": 1, "ts": STAMPS[2]}]
    for log in (TEST_RETIRED_READS, HANDWRITTEN_SPECS_RETIRED_READS):
        raw = (root / log).read_text(encoding="utf-8")
        assert raw == "".join(canonical_json(entry) + "\n" for entry in read_jsonl(root / log))
        values = {str(v) for row in [*ITEMS, *SPECS] for k, v in row.items()
                  if k not in ("slots", "messages")}
        values |= {m["content"] for item in ITEMS for m in item["messages"]} | set(asked)
        assert not [value for value in values if len(value) >= 5 and value in raw]
    assert not (root / E2E_TEST_RETIRED_READS).exists()  # each retired file logs in its own reads log


def test_no_read_writes_an_access_log_and_the_real_ones_keep_their_line_counts(root):
    access = root / TEST_ACCESS_LOG
    before = access.read_bytes()
    real_before = {log: line_count(REPO / log) for log in REAL_ACCESS_LOGS}
    stamps = clock(*STAMPS, *STAMPS)
    for _ in range(4):  # more reads than a test set has opens
        RetiredSet.single_step(root, now=stamps).similarities(["any text at all"])
    for _ in range(4):
        RetiredSet.handwritten_specs(root, now=stamps).similarities(["any text at all"])
    assert access.read_bytes() == before
    assert sorted(p.name for p in (root / EVAL_DIR).iterdir()) == sorted(
        [TEST_SET.name, TEST_SET.name + ".sha256", TEST_ACCESS_LOG.name, HANDWRITTEN_SPECS.name,
         SETS_MANIFEST.name, TEST_RETIRED_READS.name, HANDWRITTEN_SPECS_RETIRED_READS.name])
    assert {log: line_count(REPO / log) for log in REAL_ACCESS_LOGS} == real_before


# ---- refusals --------------------------------------------------------------------------------------------


def test_a_file_whose_set_is_not_retired_is_refused_and_nothing_is_logged(root):
    write_jsonl(SPENT[:2], root / TEST_ACCESS_LOG)
    for retired in (RetiredSet.single_step(root), RetiredSet.handwritten_specs(root)):
        with pytest.raises(LockError, match="not retired"):
            retired.similarities(["a text"])
    assert not (root / TEST_RETIRED_READS).exists() and not (root / HANDWRITTEN_SPECS_RETIRED_READS).exists()


def test_a_changed_test_set_or_specs_file_is_refused_and_nothing_is_logged(root):
    for path, words in ((root / TEST_SET, "does not match its lock"),
                        (root / HANDWRITTEN_SPECS, "does not match the SHA-256 sets-manifest.json records")):
        original = path.read_bytes()
        path.write_bytes(original + b"\n")
        single = path.name == TEST_SET.name
        retired = RetiredSet.single_step(root) if single else RetiredSet.handwritten_specs(root)
        with pytest.raises(LockError, match=words):
            retired.similarities(["a text"])
        path.write_bytes(original)
    assert not (root / TEST_RETIRED_READS).exists() and not (root / HANDWRITTEN_SPECS_RETIRED_READS).exists()


def test_the_specs_are_checked_against_the_digest_the_assembly_recorded(root):
    specs, manifest = root / HANDWRITTEN_SPECS, root / SETS_MANIFEST
    digest = sha256_of(specs)
    assert check_recorded(specs, manifest=manifest, key="handwritten_specs") == digest
    cases = [({"inputs": {}}, "records no SHA-256"),
             ({"inputs": {"handwritten_specs": {"path": "data/eval/handwritten.jsonl", "sha256": digest}}},
              "names handwritten.jsonl"),
             ({"inputs": {"handwritten_specs": {"path": HANDWRITTEN_SPECS.as_posix(), "sha256": "abc"}}},
              "records no SHA-256")]
    for changed, words in cases:
        manifest.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(LockError, match=words):
            check_recorded(specs, manifest=manifest, key="handwritten_specs")
        with pytest.raises(LockError, match=words):
            RetiredSet.handwritten_specs(root).similarities(["a text"])
    manifest.unlink()
    with pytest.raises(LockError, match="records no SHA-256"):
        RetiredSet.handwritten_specs(root).similarities(["a text"])
    write_lock(specs)  # a lock beside the specs does not stand in for the assembly's record
    with pytest.raises(LockError, match="records no SHA-256"):
        RetiredSet.handwritten_specs(root).similarities(["a text"])
    assert not (root / HANDWRITTEN_SPECS_RETIRED_READS).exists()


def test_an_entry_without_a_text_is_named_by_its_position_alone(tmp_path):
    no_request = {**ITEMS[1], "messages": [SYSTEM, ITEMS[1]["messages"][2], ITEMS[1]["messages"][3]]}
    root = write_v1(tmp_path, items=[ITEMS[0], no_request], specs=[{**SPECS[0], "request": " "}, SPECS[1]])
    with pytest.raises(LockError, match=r"test\.jsonl: item 2 has no text") as item_refusal:
        RetiredSet.single_step(root).similarities(["a text"])
    with pytest.raises(LockError, match=r"handwritten-specs\.jsonl: spec 1 has no text") as spec_refusal:
        RetiredSet.handwritten_specs(root).similarities(["a text"])
    for refusal in (item_refusal, spec_refusal):
        assert "req-test" not in str(refusal.value) and "hw-" not in str(refusal.value)
    assert not (root / TEST_RETIRED_READS).exists() and not (root / HANDWRITTEN_SPECS_RETIRED_READS).exists()


def test_an_end_to_end_task_without_a_text_is_still_named_a_task(tmp_path):
    tasks = tmp_path / "e2e-test.jsonl"
    write_jsonl([{"id": "e2e-test-01", "text": "Find tokenizer papers."}, {"id": "e2e-test-02", "text": ""}],
                tasks)
    write_lock(tasks)
    write_jsonl(SPENT, tmp_path / "e2e-test-access.jsonl")
    retired = RetiredSet(tasks, tmp_path / "e2e-test-access.jsonl", tmp_path / "reads.jsonl")
    with pytest.raises(LockError, match=r"e2e-test\.jsonl: task 2 has no text"):
        retired.similarities(["a text"])


def test_the_texts_must_be_strings_and_a_refused_call_reads_nothing(root):
    for retired in (RetiredSet.single_step(root), RetiredSet.handwritten_specs(root)):
        for bad in ("one bare string", ["fine", 3], [None]):
            with pytest.raises(TypeError, match="texts"):
                retired.similarities(bad)
    assert not (root / TEST_RETIRED_READS).exists() and not (root / HANDWRITTEN_SPECS_RETIRED_READS).exists()
    assert RetiredSet.single_step(root, now=clock(*STAMPS)).similarities([]) == []
    assert len(read_jsonl(root / TEST_RETIRED_READS)) == 1  # an empty read is logged too


# ---- the layout ------------------------------------------------------------------------------------------


def test_the_two_files_their_logs_and_the_end_to_end_defaults():
    single = RetiredSet.single_step(REPO)
    assert (single.path, single.access_log, single.reads_log) == (
        REPO / TEST_SET, REPO / TEST_ACCESS_LOG, REPO / TEST_RETIRED_READS)
    specs = RetiredSet.handwritten_specs(REPO)
    assert (specs.path, specs.access_log, specs.reads_log) == (
        REPO / HANDWRITTEN_SPECS, REPO / TEST_ACCESS_LOG, REPO / HANDWRITTEN_SPECS_RETIRED_READS)
    relative = RetiredSet.single_step()
    assert relative.path == TEST_SET  # relative to the repository root, where the scripts run
    assert TEST_RETIRED_READS == EVAL_DIR / "test-retired-reads.jsonl"
    assert HANDWRITTEN_SPECS_RETIRED_READS == EVAL_DIR / "handwritten-specs-retired-reads.jsonl"
    assert SETS_MANIFEST == EVAL_DIR / "sets-manifest.json"
    default = RetiredSet()
    assert (default.path, default.access_log, default.reads_log) == (
        lock_module.E2E_TEST_SET, E2E_TEST_ACCESS_LOG, E2E_TEST_RETIRED_READS)
    logs = {TEST_RETIRED_READS, HANDWRITTEN_SPECS_RETIRED_READS}
    assert not logs & {*TEST_SETS, *HANDWRITTEN_SETS}  # the reads logs hold no test material
    for name in ("data/eval/test-retired-reads.jsonl", "data/eval/handwritten-specs-retired-reads.jsonl",
                 "RetiredSet.single_step", "RetiredSet.handwritten_specs", "sets-manifest.json"):
        assert name in lock_module.__doc__, name


def test_the_guards_still_refuse_both_files_to_every_runner():
    for path in (TEST_SET, HANDWRITTEN_SPECS):
        with pytest.raises(RefusedTestSet):
            refuse_test_sets(REPO / path, [REPO])

"""End-to-end tasks (research_agent.eval.e2e and scripts/run_e2e.py) offline: scripted models over a hand-made
corpus, the scoring of every expectation, and the contract of data/eval/e2e-{dev,test}.jsonl and of the v2
test tasks data/eval/v2/e2e-test-v2.jsonl (the test files' checks only with --test-material:
tests/conftest.py)."""

from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest
from support.fakes import ScriptedModel, call, turn
from support.strict_deepseek import Answer, StrictDeepSeek

from research_agent.agent.models import CloudModel
from research_agent.arxiv import parse_atom
from research_agent.data.requests import NEAR_DUP, TRAIN_VS_EVAL, jaccard, mask_text, shingles
from research_agent.data.synth import OwnerFunctions, SeedPool
from research_agent.eval import e2e
from research_agent.eval.e2e import (
    CHECKS,
    E2E_INTENTS,
    Topic,
    World,
    check_expect,
    check_tasks,
    corpus_topics,
    load_tasks,
    run_tasks,
    score_task,
    summarize,
)
from research_agent.eval.lock import E2E_TEST_ACCESS_LOG, E2E_TEST_SET, RetiredSet
from research_agent.llm.budget import Budget
from research_agent.llm.cache import ResponseCache
from research_agent.llm.cloud import CloudClient
from research_agent.net.fetch import Fetched, Fetcher, ReplayMiss
from research_agent.policy import decide
from research_agent.tools.adapters import fulltext_url, paper_url
from research_agent.tools.corpus_search import CorpusIndex, corpus_ids, load_papers
from research_agent.validation import check_call

REPO = Path(__file__).resolve().parents[2]
TODAY = "2026-10-10"
DATA = REPO / "data" / "eval"
CORPUS_MANIFEST = REPO / "data" / "corpus" / "manifest.json"
REQUESTS_MANIFEST = REPO / "data" / "synth" / "requests-manifest.json"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_e2e = load_script("run_e2e")


# ---- a hand-made world ---------------------------------------------------------------------------------


def paper(arxiv_id: str, title: str, abstract: str, published: str) -> dict:
    return {"arxiv_id": arxiv_id, "title": title, "authors": ["A. Author"], "published": published,
            "abstract": abstract, "categories": ["cs.CL"]}


ID1, ID2, ID3 = "2609.00001", "2609.00002", "2608.00003"
CORPUS = [
    paper(ID1, "Adaptive Draft Length for Speculative Decoding",
          "Speculative decoding drafts tokens with a small model. The target model verifies every drafted "
          "token. Acceptance rates decide how long the next draft is.", "2026-09-15"),
    paper(ID2, "Tree Verification at Scale",
          "Tree-structured verification checks many drafted tokens at once. It serves large batches.",
          "2026-09-20"),
    paper(ID3, "KV Cache Compression for Long Context",
          "The key-value cache of long prompts is compressed by evicting entries. Quality barely drops.",
          "2026-08-05"),
]
TOPICS = {"speculative-decoding": Topic(frozenset({ID1}), ("speculative decoding",)),
          "kv-cache": Topic(frozenset({ID3}), ("kv cache",))}


class Pages:
    """The replay fetcher: each paper's metadata as a JSON feed and a full-text page."""

    def get(self, url: str) -> Fetched:
        for p in CORPUS:
            if url == paper_url(p["arxiv_id"]):
                return Fetched(url=url, final_url=url, status=200, body=json.dumps([p]).encode(),
                               content_type="")
            if url == fulltext_url(p["arxiv_id"]):
                return Fetched(url=url, final_url=url, status=200, body=b"<html><p>Method text.</p></html>",
                               content_type="")
        raise ReplayMiss(f"no cassette for {url}")


def parse_feed(xml: str) -> list[dict] | None:
    try:
        return json.loads(xml)
    except json.JSONDecodeError:
        return None


def world(seed_pool: SeedPool | None = None) -> World:
    owner = OwnerFunctions(check_call, decide, parse_feed, lambda html, section, n: "Method text.")
    return World(owner, Pages(), CorpusIndex(CORPUS), seed_pool, TOPICS)


EXPECT = {"status": "answered", "must_write_note": False, "note_about": None,
          "must_cite_ids_within": "tool_outputs", "expected_tools_any": ["get_paper"]}


def task(tid: str, intent: str, text: str, slots: dict | None = None, **expect) -> dict:
    return {"id": tid, "intent": intent, "scenario": e2e.SCENARIO_OF.get(intent, "normal"), "lang": "en",
            "split": "dev", "family": f"e2e/{intent}", "text": text, "slots": slots or {},
            "expect": {**EXPECT, **expect}}


NOTE_TASK = task("t-note", "write_note", "Find speculative decoding papers since 2026-09-01 and write a "
                 "scaffold note on the best one.",
                 {"topic": "speculative decoding", "date_from": "2026-09-01", "mode": "scaffold"},
                 must_write_note=True, note_about={"topic_id": "speculative-decoding"},
                 expected_tools_any=["write_note"])
LOOKUP_TASK = task("t-lookup", "lookup_paper", f"What is {ID2} about?", {"arxiv_id": ID2})
SEARCH = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-01")
GET1, GET2 = call("get_paper", "c2", arxiv_id=ID1), call("get_paper", "c2", arxiv_id=ID2)
NOTE = call("write_note", "c3", arxiv_id=ID1, mode="scaffold",
            evidence='"The target model verifies every drafted token."')


def note_turns(answer: str = f"I wrote a scaffold note on {ID1}.") -> list:
    return [turn(SEARCH), turn(GET1), turn(NOTE), turn(content=answer)]


def run(tasks: list[dict], turns: list, tmp_path: Path, **options) -> list[dict]:
    return run_tasks(tasks, model=ScriptedModel(turns), world=options.pop("world", None) or world(),
                     today=TODAY, out=tmp_path, log=lambda line: None, **options)


# ---- running two tiny tasks ----------------------------------------------------------------------------


def test_a_correct_run_passes_and_a_fabricated_id_fails(tmp_path):
    turns = [*note_turns(), turn(GET2), turn(content=f"{ID2} builds on 2609.99999 for tree verification.")]
    passed, failed = run([NOTE_TASK, LOOKUP_TASK], turns, tmp_path)
    assert passed["success"] and all(passed["checks"][c]["ok"] for c in CHECKS)
    evidence = passed["checks"]["evidence"]
    assert (evidence["calls"], evidence["fragments"], evidence["stored"]["not_in_source"]) == (1, 1, [])
    assert evidence["stored"]["sentences"] > 0 and passed["checks"]["note"]["written"] == [ID1]
    assert not failed["success"]
    assert failed["checks"]["cited_ids"] == {"ok": False, "cited": [ID2, "2609.99999"],
                                             "unseen": ["2609.99999"]}
    assert any(p.startswith("ids:") and "2609.99999" in p for p in failed["checks"]["trajectory"]["problems"])
    assert (tmp_path / "transcripts" / "t-note.json").is_file() and passed["steps"] == 4
    assert passed["tool_calls"] == {"search_arxiv": 1, "get_paper": 1, "write_note": 1}
    assert passed["tokens"] == {"prompt_tokens": 400, "completion_tokens": 40}
    # no time in a record: the wall times go to the timing sidecar (eval.timing), so the records stay the same
    # bytes whatever the clock said
    assert "wall_s" not in passed and "wall_s" not in failed


def test_the_summary_counts_success_per_check_intent_and_language(tmp_path):
    turns = [*note_turns(), turn(GET2), turn(content="It verifies trees of drafts, see 2609.99999.")]
    summary = summarize(run([NOTE_TASK, LOOKUP_TASK], turns, tmp_path))
    assert summary["n"] == 2 and summary["success_rate"] == 0.5
    assert summary["by_check"]["cited_ids"] == 0.5 and summary["by_check"]["note"] == 1.0
    assert summary["by_intent"]["lookup_paper"] == {"n": 1, "success_rate": 0.0}
    assert summary["by_lang"] == {"en": {"n": 2, "success_rate": 0.5}}
    assert summary["statuses"] == {"answered": 2} and summary["tool_errors"] == 0
    assert "wall_s" not in summary  # the timing sidecar's, as in the records


# ---- each expectation ----------------------------------------------------------------------------------


def test_the_status_must_be_the_expected_one(tmp_path):
    clarify = task("t-clar", "needs_clarification", "Summarize the paper we talked about.",
                   status="clarification", expected_tools_any=["ask_clarification"])
    (record,) = run([clarify], [turn(content="Which paper do you mean?")], tmp_path)
    assert record["checks"]["status"] == {"ok": False, "got": "answered", "want": "clarification"}
    assert not record["checks"]["tools"]["ok"]
    asked = call("ask_clarification", "c1", question="Which paper?", missing=["paper_id"])
    (record,) = run([clarify], [turn(asked)], tmp_path / "again")
    assert record["success"], record["checks"]


def test_small_talk_must_make_no_call(tmp_path):
    chat = task("t-chat", "chit_chat", "Thanks, see you tomorrow.", expected_tools_any=[])
    (quiet,) = run([chat], [turn(content="See you!")], tmp_path / "a")
    assert quiet["success"]
    (busy,) = run([chat], [turn(call("list_notes", "c1")), turn(content="See you!")], tmp_path / "b")
    assert not busy["checks"]["tools"]["ok"] and busy["checks"]["tools"]["called"] == ["list_notes"]


def test_one_of_the_expected_tools_is_enough(tmp_path):
    compare = task("t-cmp", "compare_papers", f"Compare {ID1} and {ID2}.", {"arxiv_ids": [ID1, ID2]},
                   expected_tools_any=["get_paper", "fetch_fulltext"])
    fetch = call("fetch_fulltext", "c1", arxiv_id=ID1, section="method")
    (record,) = run([compare], [turn(fetch), turn(content=f"{ID1} uses a method text.")], tmp_path)
    assert record["checks"]["tools"]["ok"]


def test_a_required_note_must_be_on_the_named_paper(tmp_path):
    named = {**NOTE_TASK, "id": "t-named", "expect": {**NOTE_TASK["expect"], "note_about": {"arxiv_id": ID2}}}
    (record,) = run([named], note_turns(), tmp_path)
    assert record["checks"]["note"] == {"ok": False, "written": [ID1], "about": {"arxiv_id": ID2}}


def test_a_topic_note_may_be_on_a_paper_outside_the_list_that_names_the_topic(tmp_path):
    topics = {"speculative-decoding": Topic(frozenset(), ("speculative decoding",))}
    (record,) = run([NOTE_TASK], note_turns(), tmp_path,
                    world=World(**{**world().__dict__, "topics": topics}))
    assert record["checks"]["note"]["ok"]
    unrelated = {"speculative-decoding": Topic(frozenset(), ("mixture of experts",))}
    (record,) = run([NOTE_TASK], note_turns(), tmp_path / "b",
                    world=World(**{**world().__dict__, "topics": unrelated}))
    assert not record["checks"]["note"]["ok"]


def test_a_missing_note_fails_the_note_check(tmp_path):
    (record,) = run([NOTE_TASK], [turn(SEARCH), turn(content=f"The best one is {ID1}.")], tmp_path)
    assert record["checks"]["note"] == {"ok": False, "written": [],
                                        "about": {"topic_id": "speculative-decoding"}}


def note_with(evidence: object, mode: str = "scaffold") -> dict:
    return call("write_note", "c3", arxiv_id=ID1, mode=mode, evidence=evidence)


@pytest.mark.parametrize("evidence", [
    "Drafts are verified by the target.",  # a paraphrase
    '"The method doubles throughput on every benchmark."',  # invented
    '"The target model checks every drafted token."',  # one word changed inside the quotation
    '"目标模型会验证每一个草稿词元。"',  # a translation
    '"Tree-structured verification checks many drafted tokens at once."',  # another paper's abstract
], ids=["paraphrase", "invented", "changed-word", "translated", "other-paper"])
def test_evidence_the_model_did_not_quote_from_its_papers_retrieved_text_fails(evidence, tmp_path):
    both = call("search_arxiv", "c1", query="speculative decoding drafted tokens", date_from="2026-09-01")
    turns = [turn(both), turn(GET1), turn(note_with(evidence)), turn(content=f"Noted {ID1}.")]
    (record,) = run([NOTE_TASK], turns, tmp_path)
    transcript = json.loads((tmp_path / "transcripts" / "t-note.json").read_text(encoding="utf-8"))
    hits = json.loads(next(m["content"] for m in transcript["messages"] if m["role"] == "tool"))
    assert {hit["arxiv_id"] for hit in hits} == {ID1, ID2}  # the other paper's text was retrieved too
    assert not record["checks"]["evidence"]["ok"] and not record["success"]
    assert record["checks"]["evidence"]["not_retrieved"][0].startswith("c3: ")


@pytest.mark.parametrize("evidence", [
    '"The target model verifies every drafted token."',
    ('Abstract: "Speculative decoding drafts tokens with a small model." (first sentence)\n'
     '- "The target model verifies every drafted token"'),  # labels outside the quotation marks, no full stop
    "Speculative decoding drafts tokens with a small model. The target model verifies every drafted token.",
    '"Speculative decoding drafts tokens ... decide how long the next draft is."',  # an elision
    "“the target model verifies every drafted token”",  # curly quotes, first letter lower-cased
], ids=["quoted", "labelled", "unquoted-sentences", "elided", "curly"])
def test_evidence_quoted_from_the_retrieved_abstract_passes_whatever_the_typography(evidence, tmp_path):
    turns = [turn(SEARCH), turn(GET1), turn(note_with(evidence)), turn(content=f"Noted {ID1}.")]
    (record,) = run([NOTE_TASK], turns, tmp_path)
    assert record["checks"]["evidence"]["ok"] and record["success"], record["checks"]["evidence"]


def test_a_draft_note_is_not_checked_and_the_stored_notes_evidence_is_information_only(tmp_path):
    draft = call("write_note", "c3", arxiv_id=ID1, mode="draft", problem="p", method="m",
                 evidence="Drafts get checked.", limitations="l")
    (record,) = run([NOTE_TASK], [turn(SEARCH), turn(GET1), turn(draft), turn(content=f"Noted {ID1}.")],
                    tmp_path)
    assert record["checks"]["evidence"]["ok"] and record["checks"]["evidence"]["calls"] == 0
    run([NOTE_TASK], note_turns(), tmp_path / "again")
    note = next((tmp_path / "again" / "notes" / "t-note").glob("*.md"))
    note.write_text(note.read_text(encoding="utf-8").replace(
        "The target model verifies every drafted token.", "The method doubles throughput."), encoding="utf-8")
    transcript = json.loads((tmp_path / "again" / "transcripts" / "t-note.json").read_text(encoding="utf-8"))
    scored = score_task(NOTE_TASK, transcript, notes_dir=tmp_path / "again" / "notes" / "t-note",
                        papers=CorpusIndex(CORPUS).paper, topics=TOPICS)
    assert scored["success"] and scored["checks"]["evidence"]["ok"]
    assert scored["checks"]["evidence"]["stored"]["not_in_source"][0].endswith("doubles throughput.")


def test_evidence_tokens_ignore_typography_and_latex_but_keep_every_word_number_and_sign():
    from research_agent.eval.e2e import evidence_fragments, evidence_tokens, unsupported_fragments

    source = "Accuracy rises by 55.1\\% at 5.5$\\times$ lower cost with $\\alpha$-scaled heads; it’s robust."
    assert unsupported_fragments('"accuracy rises by 55.1% at 5.5x lower cost with α-scaled heads."',
                                 [source]) == []
    assert unsupported_fragments('"Accuracy rises by 55% at 5.5x lower cost"', [source]) == [
        "Accuracy rises by 55% at 5.5x lower cost"]
    typeset = evidence_tokens("It’s “fine” — really, 1,000 times.")
    assert typeset == evidence_tokens("It's fine - really 1000 times")
    assert evidence_fragments('The "draft" model [sic] is verified.') == ['The "draft" model', "is verified."]
    assert unsupported_fragments("...", [source]) == ["(nothing quoted)"]
    assert unsupported_fragments(None, [source]) == ["(nothing quoted)"]


@pytest.mark.parametrize("quote", [
    '"Our method reaches 125 BLEU, an improvement of 31 points"',  # decimal points dropped
    '"Our method reaches 1.25 BLEU, an improvement of 3.1 points"',  # a decimal point moved
    '"an improvement of 3.1 points, with a 3% error"',  # the minus sign dropped
    '"with a -3% error measured on an other test set"',  # letters align, words do not ("another")
], ids=["decimals-dropped", "decimal-moved", "sign-dropped", "word-boundary"])
def test_a_changed_number_sign_or_word_boundary_is_not_a_quotation(quote):
    from research_agent.eval.e2e import unsupported_fragments

    source = "Our method reaches 12.5 BLEU, an improvement of 3.1 points, with a −3% error measured on " \
             "another test set; Qwen3-14B follows it."
    assert unsupported_fragments(quote, [source]) != []
    assert unsupported_fragments('"with a -3% error measured on another test set; Qwen3 14B follows it"',
                                 [source]) == []  # the minus sign's two forms, and a hyphen between tokens


@pytest.mark.parametrize(("quote", "source"), [
    ('"We set $\\Delta$ to 0.1 for all experiments here"', "We set Δ to 0.1 for all experiments here"),
    ('"the margin $\\varepsilon$ stays fixed across every run"', "the margin ε stays fixed across every run"),
    ('"the margin ε stays fixed across every run"', "the margin $\\varepsilon$ stays fixed across every run"),
])
def test_capital_and_variant_greek_commands_read_as_their_letters(quote, source):
    from research_agent.eval.e2e import unsupported_fragments

    assert unsupported_fragments(quote, [source]) == []


@pytest.mark.parametrize(("source", "quote"), [
    ("the error was p < 0.05 in all runs", "the error was p > 0.05 in all runs"),
    ("accuracy was ≥ 90 percent overall", "accuracy was ≤ 90 percent overall"),
    ("accuracy was >= 90 percent overall", "accuracy was <= 90 percent overall"),
    ("we require $x \\leq 5$ always here", "we require $x \\geq 5$ always here"),
    ("where $a \\in S$ holds true here", "where $a \\notin S$ holds true here"),
    ("accuracy $\\uparrow$ on all splits", "accuracy $\\downarrow$ on all splits"),
    ("a drop of 5% in recall here", "a drop of 5 in recall here"),
    ("scaled by 10^-3 per step taken", "scaled by 10^3 per step taken"),
    ("this holds ¬ A for every case", "this holds A for every case"),
], ids=["lt-gt", "le-ge-unicode", "le-ge-ascii", "le-ge-latex", "in-notin", "arrows", "percent",
        "exponent-sign", "negation"])
def test_a_flipped_relation_sign_or_symbol_is_not_a_quotation(source, quote):
    from research_agent.eval.e2e import unsupported_fragments

    assert unsupported_fragments(f'"{quote}"', [source]) != []


@pytest.mark.parametrize(("source", "quote"), [
    ("image size 224x224 pixels used here", "image size 224×224 pixels used here"),
    ("image size $224 \\times 224$ pixels here", "image size 224×224 pixels here"),
    ("a change of –3 points overall here", "a change of -3 points overall here"),
    ("about 1 234 567 samples seen in total", "about 1234567 samples seen in total"),
    ("inter-\nnational standards apply here", "international standards apply here"),
    ("the length $\\ell$ of tokens per sample", "the length ℓ of tokens per sample"),
    ("we require $x \\leq 5$ always here", "we require x ≤ 5 always here"),
], ids=["times-sign", "latex-times", "en-dash-minus", "thin-space-thousands", "line-break-hyphen", "ell",
        "latex-relation"])
def test_a_typographic_variant_is_still_a_quotation(source, quote):
    from research_agent.eval.e2e import unsupported_fragments

    assert unsupported_fragments(f'"{quote}"', [source]) == []


def test_a_sign_after_a_symbol_stays_with_its_number_and_the_ascii_arrow_is_the_arrow():
    from research_agent.eval.e2e import unsupported_fragments

    source = "The correlation was r=-0.42 (p<0.01) and a step A -> B was taken; x<-1 held throughout."
    assert unsupported_fragments('"The correlation was r=0.42 (p<0.01)"', [source]) != []  # sign dropped
    assert unsupported_fragments('"The correlation was r = -0.42 (p < 0.01)"', [source]) == []
    assert unsupported_fragments('"and a step A → B was taken"', [source]) == []
    assert unsupported_fragments('"x < -1 held throughout"', [source]) == []  # "<-" is not an arrow
    assert unsupported_fragments('"x < 1 held throughout"', [source]) != []


def test_the_date_constraint_of_the_slots_is_checked_through_check_trajectory(tmp_path):
    undated = call("search_arxiv", "c1", query="speculative decoding")
    (record,) = run([NOTE_TASK], [turn(undated), turn(GET1), turn(NOTE), turn(content=f"Noted {ID1}.")],
                    tmp_path)
    assert not record["checks"]["trajectory"]["ok"]
    assert any("date_from" in p for p in record["checks"]["trajectory"]["problems"])


def test_seeded_notes_are_there_for_a_list_notes_task(tmp_path):
    pool = SeedPool(topics={"speculative-decoding": (ID1, ID2), "kv-cache": (ID3,)},
                    names={"speculative decoding": "speculative-decoding"}, splits=None)
    listing = task("t-list", "list_notes", "List my recent notes.", {"count": 5},
                   expected_tools_any=["list_notes", "search_notes"])
    (record,) = run([listing], [turn(call("list_notes", "c1", limit=5)), turn(content="You have notes.")],
                    tmp_path, world=world(pool))
    transcript = json.loads((tmp_path / "transcripts" / "t-list.json").read_text(encoding="utf-8"))
    listed = json.loads(next(m for m in transcript["messages"] if m["role"] == "tool")["content"])
    assert record["success"] and len(listed) == 3


def test_a_budget_stop_ends_the_run(tmp_path):
    from research_agent.llm.budget import BudgetExceeded

    records = run([LOOKUP_TASK, NOTE_TASK], [BudgetExceeded("cap")], tmp_path)
    assert [r["status"] for r in records] == ["budget_stop"]


# ---- the task files' format ----------------------------------------------------------------------------


def test_expect_is_checked_for_shape():
    assert check_expect(EXPECT) == []
    assert "expect lacks status" in check_expect({k: v for k, v in EXPECT.items() if k != "status"})
    assert check_expect({**EXPECT, "expected_tools_any": ["browse"]})
    assert check_expect({**EXPECT, "must_write_note": True})
    assert check_expect({**EXPECT, "status": "clarification"})
    assert check_expect({**EXPECT, "must_cite_ids_within": "anything"})


def test_load_tasks_refuses_an_intent_outside_the_list_and_a_wrong_scenario(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({**LOOKUP_TASK, "intent": "summarize_paper"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="intent"):
        load_tasks(path)
    path.write_text(json.dumps({**LOOKUP_TASK, "scenario": "no_call"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="scenario"):
        load_tasks(path)


def test_check_tasks_makes_load_tasks_checks_on_tasks_already_read(tmp_path):
    """The end-to-end test tasks stay in memory once opened (scripts/run_test.py): check_tasks makes
    load_tasks' checks on them and gives what load_tasks gives for the same records in a file."""
    path = tmp_path / "t.jsonl"
    path.write_text("".join(json.dumps(t) + "\n" for t in (NOTE_TASK, LOOKUP_TASK)), encoding="utf-8")
    assert check_tasks([NOTE_TASK, LOOKUP_TASK], "opened") == load_tasks(path)
    lacking = {k: v for k, v in EXPECT.items() if k != "status"}
    for bad, message in (({**LOOKUP_TASK, "intent": "summarize_paper"}, "opened: task t-lookup: intent"),
                         ({**LOOKUP_TASK, "split": "train"}, "opened: task t-lookup: split must be dev"),
                         ({**LOOKUP_TASK, "text": ""}, "opened: record 2: text must be a non-empty string"),
                         ({**LOOKUP_TASK, "expect": lacking}, "opened: task t-lookup: expect lacks status")):
        with pytest.raises(ValueError, match=message):
            check_tasks([NOTE_TASK, bad], "opened")


# ---- the task files: data/eval/e2e-dev.jsonl and e2e-test.jsonl (with --test-material) -------------

FILES = {"dev": DATA / "e2e-dev.jsonl", "test": DATA / "e2e-test.jsonl"}
SIZES = {"dev": 20, "test": 60}
# The test file is locked test material: its checks run only with --test-material (tests/conftest.py). They
# held when the file was locked, and the lock (the hash of its bytes, checked before every open) keeps them.
def splits(*more: str) -> list:
    """The two splits as parameters. The test file is test material (above); the dev file, and the further
    paths a test reads (more), are data the public copy leaves out, so dev is marked needs_data
    (tests/conftest.py)."""
    return [pytest.param("dev", marks=pytest.mark.needs_data("data/eval/e2e-dev.jsonl", *more)),
            pytest.param("test", marks=pytest.mark.test_material)]



_ID = re.compile(r"(?<![\d.])\d{4}\.\d{4,5}(?![\d])")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def manifest() -> dict:
    return json.loads(REQUESTS_MANIFEST.read_text(encoding="utf-8"))


def split_ids(split: str) -> set[str]:
    topics = corpus_topics(CORPUS_MANIFEST)
    return {a for t in manifest()["topics"][split] for a in topics[t].papers}


def task_ids(t: dict) -> set[str]:
    slots = t["slots"]
    named = [slots["arxiv_id"]] if "arxiv_id" in slots else list(slots.get("arxiv_ids", []))
    return set(named) | set(_ID.findall(t["text"]))


@pytest.mark.parametrize("split", splits())
def test_the_task_file_has_its_size_half_chinese_half_english_and_every_intent(split):
    tasks = load_tasks(FILES[split])
    assert len(tasks) == SIZES[split] and {t["split"] for t in tasks} == {split}
    assert sum(t["lang"] == "zh" for t in tasks) == SIZES[split] // 2
    assert {t["intent"] for t in tasks} == set(E2E_INTENTS)
    assert all(t["id"].startswith(f"e2e-{split}-") for t in tasks)


@pytest.mark.parametrize("split", splits("data/synth"))  # the requests' manifest
def test_tasks_use_only_their_splits_topics_and_paper_ids(split):
    allowed_topics = set(manifest()["topics"][split])
    allowed_ids = split_ids(split)
    other = {s for s in ("train", "dev", "test") if s != split}
    other_ids = set().union(*(split_ids(s) for s in other))
    for t in load_tasks(FILES[split]):
        if "topic_id" in t["slots"]:
            assert t["slots"]["topic_id"] in allowed_topics, t["id"]
        assert task_ids(t) <= allowed_ids, (t["id"], task_ids(t) - allowed_ids)
        assert not task_ids(t) & other_ids, t["id"]
        about = t["expect"]["note_about"] or {}
        assert about.get("topic_id", next(iter(allowed_topics))) in allowed_topics, t["id"]
        assert about.get("arxiv_id", next(iter(allowed_ids))) in allowed_ids, t["id"]


@pytest.mark.parametrize("split", splits("data/synth"))  # the requests' manifest
def test_every_date_lies_in_the_recording_window_and_every_id_is_in_the_corpus(split):
    start, end = manifest()["window"]
    corpus = set(corpus_ids(CORPUS_MANIFEST))
    for t in load_tasks(FILES[split]):
        dates = set(_DATE.findall(t["text"])) | {v for k, v in t["slots"].items() if k in
                                                 ("date_from", "date_to", "since")}
        assert all(start <= d <= end <= TODAY for d in dates), t["id"]
        assert task_ids(t) <= corpus, t["id"]
        if "date_from" in t["slots"] and "date_to" in t["slots"]:
            assert t["slots"]["date_from"] <= t["slots"]["date_to"], t["id"]


@pytest.fixture(scope="module")
def recorded_index() -> CorpusIndex:
    fetcher = Fetcher("replay", REPO / "data" / "cassettes")
    return CorpusIndex(load_papers(fetcher, corpus_ids(CORPUS_MANIFEST), parse_atom))


SLACK = 2  # recorded hits beyond the requested count, so one extra filter or synonym does not starve a task


@pytest.mark.parametrize("split", splits("data/cassettes"))
def test_every_topic_search_of_a_task_leaves_two_recorded_hits_beyond_its_count(split, recorded_index):
    for t in load_tasks(FILES[split]):
        slots = t["slots"]
        if "topic" not in slots or t["intent"] == "search_notes":
            continue
        args = {"query": slots["topic"],
                **{k: copy.deepcopy(slots[k]) for k in ("date_from", "date_to", "categories") if k in slots}}
        assert len(recorded_index.search(args)) >= slots.get("count", 1) + SLACK, t["id"]


@pytest.mark.parametrize("split", splits("data/synth"))  # the training requests
def test_no_task_resembles_a_train_request_or_another_task(split):
    """The leak rule of data.requests (masked 5-gram Jaccard): every task of the split below TRAIN_VS_EVAL
    against every train request, and below NEAR_DUP against every other task of the two files (dev's run
    compares dev with dev, test's compares test with test and with dev)."""
    train = [(r["id"], shingles(mask_text(r["text"])))
             for r in map(json.loads, (REPO / "data" / "synth" / "requests.jsonl").read_text(
                 encoding="utf-8").splitlines()) if r["split"] == "train"]
    tasks = {t["id"]: shingles(mask_text(t["text"])) for t in load_tasks(FILES[split])}
    dev = {} if split == "dev" else {t["id"]: shingles(mask_text(t["text"]))
                                     for t in load_tasks(FILES["dev"])}
    assert len(train) > 1000 and len(tasks) == SIZES[split]
    for tid, grams in tasks.items():
        closest, rid = max((jaccard(grams, other), rid) for rid, other in train)
        assert closest < TRAIN_VS_EVAL, (tid, rid, round(closest, 3))
    for (a, grams_a), (b, grams_b) in [*itertools.combinations(tasks.items(), 2),
                                       *itertools.product(tasks.items(), dev.items())]:
        assert jaccard(grams_a, grams_b) < NEAR_DUP, (a, b)


@pytest.mark.parametrize("split", splits("data/cassettes"))
def test_every_named_paper_is_in_the_recorded_metadata(split, recorded_index):
    for t in load_tasks(FILES[split]):
        assert all(recorded_index.paper(a) is not None for a in task_ids(t)), t["id"]


# ---- v2: data/eval/v2/e2e-test-v2.jsonl (docs/m7-计划.md WP-I 6; with --test-material) -----------------
# The same conditions on the v2 test tasks, plus v2's: the drawn counts and slot records
# (scripts/draw_e2e_slots.py), every training request of M6 and M7 too, and the retired v1 tasks, read only
# through eval.lock.RetiredSet (similarities alone; the tests log their reads in a log of their own).
# scripts/check_e2e_tasks.py makes the same checks and the writing rules' on a draft or the whole file.

V2_FILE = DATA / "v2" / "e2e-test-v2.jsonl"
V2_SLOTS = DATA / "v2" / "e2e-slots-v2.jsonl"
V2_SIZE = 150
V2_TRAIN = [REPO / "data" / "synth" / name
            for name in ("requests.jsonl", "requests-m6.jsonl", "requests-m7-train.jsonl")]


def v2_slots() -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, V2_SLOTS.read_text(encoding="utf-8").splitlines())}


def retired_v1(reads: Path) -> RetiredSet:
    """v1's end-to-end tasks, retired since M6, with a reads log of the test's own."""
    return RetiredSet(REPO / E2E_TEST_SET, REPO / E2E_TEST_ACCESS_LOG, reads)


def access_lines() -> int:
    return len((REPO / E2E_TEST_ACCESS_LOG).read_text(encoding="utf-8").splitlines())


@pytest.mark.test_material
def test_v2_has_150_tasks_half_chinese_half_english_and_the_drawn_intent_counts():
    tasks, slots = load_tasks(V2_FILE), v2_slots()
    assert len(tasks) == len(slots) == V2_SIZE and {t["split"] for t in tasks} == {"test"}
    assert sum(t["lang"] == "zh" for t in tasks) == V2_SIZE // 2
    assert Counter(t["intent"] for t in tasks) == Counter(r["intent"] for r in slots.values())
    assert {t["intent"] for t in tasks} == set(E2E_INTENTS)
    assert sorted(t["id"] for t in tasks) == sorted(slots)
    assert all(t["id"].startswith("e2e-test-") for t in tasks)


@pytest.mark.test_material
def test_v2_tasks_are_their_slot_records_with_a_text():
    slots = v2_slots()
    for t in load_tasks(V2_FILE):
        assert {k: v for k, v in t.items() if k != "text"} == slots[t["id"]], t["id"]


@pytest.mark.test_material
def test_v2_tasks_use_only_test_topics_and_ids_dates_in_the_window_and_recorded_papers(recorded_index):
    allowed_topics, allowed_ids = set(manifest()["topics"]["test"]), split_ids("test")
    other_ids = split_ids("train") | split_ids("dev")
    start, end = manifest()["window"]
    corpus = set(corpus_ids(CORPUS_MANIFEST))
    for t in load_tasks(V2_FILE):
        if "topic_id" in t["slots"]:
            assert t["slots"]["topic_id"] in allowed_topics, t["id"]
        assert task_ids(t) <= allowed_ids and not task_ids(t) & other_ids, t["id"]
        about = t["expect"]["note_about"] or {}
        assert about.get("topic_id", next(iter(allowed_topics))) in allowed_topics, t["id"]
        assert about.get("arxiv_id", next(iter(allowed_ids))) in allowed_ids, t["id"]
        dates = set(_DATE.findall(t["text"])) | {v for k, v in t["slots"].items() if k in
                                                 ("date_from", "date_to", "since")}
        assert all(start <= d <= end <= TODAY for d in dates), t["id"]
        if "date_from" in t["slots"] and "date_to" in t["slots"]:
            assert t["slots"]["date_from"] <= t["slots"]["date_to"], t["id"]
        assert task_ids(t) <= corpus, t["id"]
        assert all(recorded_index.paper(a) is not None for a in task_ids(t)), t["id"]


@pytest.mark.test_material
def test_every_v2_topic_search_leaves_two_recorded_hits_beyond_its_count(recorded_index):
    for t in load_tasks(V2_FILE):
        slots = t["slots"]
        if "topic" not in slots or t["intent"] == "search_notes":
            continue
        args = {"query": slots["topic"],
                **{k: copy.deepcopy(slots[k]) for k in ("date_from", "date_to", "categories") if k in slots}}
        assert len(recorded_index.search(args)) >= slots.get("count", 1) + SLACK, t["id"]


@pytest.mark.test_material
def test_no_v2_task_resembles_a_training_request_another_task_or_a_dev_task():
    """The leak rule over every training request so far (the train split of requests.jsonl, the M6 requests
    and M7's new training requests): below TRAIN_VS_EVAL; below NEAR_DUP between any two v2 tasks and
    against every dev task."""
    train = [(r["id"], shingles(mask_text(r["text"]))) for path in V2_TRAIN
             for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if r["split"] == "train"]
    tasks = {t["id"]: shingles(mask_text(t["text"])) for t in load_tasks(V2_FILE)}
    dev = {t["id"]: shingles(mask_text(t["text"])) for t in load_tasks(FILES["dev"])}
    assert len(train) > 3000 and len(tasks) == V2_SIZE
    for tid, grams in tasks.items():
        closest, rid = max((jaccard(grams, other), rid) for rid, other in train)
        assert closest < TRAIN_VS_EVAL, (tid, rid, round(closest, 3))
    for (a, grams_a), (b, grams_b) in [*itertools.combinations(tasks.items(), 2),
                                       *itertools.product(tasks.items(), dev.items())]:
        assert jaccard(grams_a, grams_b) < NEAR_DUP, (a, b)


@pytest.mark.test_material
def test_no_v2_task_resembles_a_retired_v1_task(tmp_path):
    before = access_lines()
    scores = retired_v1(tmp_path / "reads.jsonl").similarities([t["text"] for t in load_tasks(V2_FILE)])
    assert len(scores) == V2_SIZE and all(type(s) is float for s in scores)
    assert max(scores) < TRAIN_VS_EVAL, [round(s, 3) for s in scores if s >= TRAIN_VS_EVAL]
    assert access_lines() == before


@pytest.mark.test_material
def test_a_retired_read_of_the_real_v1_tasks_returns_floats_and_leaves_their_access_log_as_it_is(tmp_path):
    before = access_lines()
    assert before == 3  # retired: all three opens used (M3, M5, M6)
    texts = ["Find tokenizer papers from August 2026.", ""]
    scores = retired_v1(tmp_path / "reads.jsonl").similarities(texts)
    assert len(scores) == 2 and all(type(s) is float and 0.0 <= s <= 1.0 for s in scores)
    assert access_lines() == before
    (entry,) = map(json.loads, (tmp_path / "reads.jsonl").read_text(encoding="utf-8").splitlines())
    assert set(entry) == {"lock_ok", "sha256", "texts", "ts"} and entry["texts"] == 2


@pytest.mark.test_material
def test_the_checker_finds_nothing_in_the_v2_file(tmp_path):
    checker = load_script("check_e2e_tasks")
    inputs = checker.load_inputs(retired=retired_v1(tmp_path / "reads.jsonl"))
    tasks = [json.loads(line) for line in V2_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = checker.check(tasks, inputs)
    assert report.ok, report.lines()  # ids and reason codes, never a text


# ---- the script ----------------------------------------------------------------------------------------


def test_run_e2e_writes_the_report_and_prints_the_summary(tmp_path, capsys):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("".join(json.dumps(t) + "\n" for t in (NOTE_TASK, LOOKUP_TASK)), encoding="utf-8")
    turns = [*note_turns(), turn(GET2), turn(content=f"{ID2} is about tree verification.")]
    out = tmp_path / "e2e-scripted"
    code = run_e2e.main([str(tasks), "--system", "b2", "--out", str(out), "--no-tokenizer"],
                        model=ScriptedModel(turns), world=world())
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert code == 0 and report["summary"]["success_rate"] == 1.0 and len(report["tasks"]) == 2
    assert report["manifest"]["system"] == "b2" and report["manifest"]["tasks_sha256"]
    assert "success 100.0%" in capsys.readouterr().out


def test_run_e2e_has_no_replay_system():
    with pytest.raises(SystemExit):
        run_e2e.build_parser().parse_args(["t.jsonl", "--system", "replay", "--out", "x"])


def test_run_e2e_counts_a_cloud_models_turns_per_mode_and_a_scripted_run_has_none(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps(LOOKUP_TASK) + "\n", encoding="utf-8")
    answer = f"{ID2} is about tree verification."
    # an empty reasoning cannot be passed back, so the step after it runs with thinking off
    server = StrictDeepSeek([Answer(calls=(("call_00_g", "get_paper", {"arxiv_id": ID2}),), reasoning=""),
                             Answer(answer)])
    cache, budget = ResponseCache(":memory:"), Budget(":memory:", stage="M3", stage_cap_rmb=1.0)
    argv = [str(tasks), "--system", "b2", "--no-tokenizer", "--out"]
    try:
        model = CloudModel(CloudClient("deepseek-flash", cache=cache, budget=budget, client=server.sdk()))
        code = run_e2e.main([*argv, str(tmp_path / "cloud")], model=model, world=world())
    finally:
        cache.close()
        budget.close()
    report = json.loads((tmp_path / "cloud" / "report.json").read_text(encoding="utf-8"))
    assert code == 0 and [t["status"] for t in report["tasks"]] == ["answered"], report["tasks"]
    assert report["manifest"]["modes"] == {"non-thinking": 1, "thinking": 1} and server.refused == 0
    scripted = ScriptedModel([turn(GET2), turn(content=answer)])
    assert run_e2e.main([*argv, str(tmp_path / "scripted")], model=scripted, world=world()) == 0
    scripted_report = json.loads((tmp_path / "scripted" / "report.json").read_text(encoding="utf-8"))
    assert "modes" not in scripted_report["manifest"]

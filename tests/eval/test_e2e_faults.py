"""Fault injection in the end-to-end run (docs/plan.md, 故障注入; M4): run_tasks with a FaultSetting records
every fault a task meets in its record and its transcript, and two systems meet the same faults."""

from __future__ import annotations

import json
import re

from support.fakes import ScriptedModel, call, turn

from research_agent.data.synth import FaultSetting, OwnerFunctions
from research_agent.eval.e2e import Topic, World, run_tasks
from research_agent.net.faults import FAULTS, applicable, draw
from research_agent.net.fetch import Fetched, ReplayMiss, canonical_url
from research_agent.policy import decide
from research_agent.tools.adapters import fulltext_url, paper_url
from research_agent.tools.corpus_search import CorpusIndex
from research_agent.validation import check_call

TODAY = "2026-10-10"
ID1, ID2 = "2609.00001", "2609.00002"
CORPUS = [{"arxiv_id": ID1, "title": "Adaptive Draft Length for Speculative Decoding",
           "authors": ["A. Author"], "published": "2026-09-15",
           "abstract": "Speculative decoding drafts tokens with a small model.", "categories": ["cs.CL"]},
          {"arxiv_id": ID2, "title": "Tree Verification at Scale", "authors": ["A. Author"],
           "published": "2026-09-20", "categories": ["cs.CL"],
           "abstract": "Tree-structured verification checks drafted tokens at once."}]
FEED_END = "</feed>"
_ENTRY = re.compile(r"<entry>(.*?)</entry>", re.DOTALL)


def feed(paper: dict) -> str:
    """An Atom-shaped feed the fault injector can empty or cut: each entry holds a paper as JSON."""
    return ("<feed><opensearch:totalResults>1</opensearch:totalResults>"
            f"<entry>{json.dumps(paper)}</entry>{FEED_END}")


def parse_feed(xml: str) -> list[dict] | None:
    """The scripted parse_atom: None for a feed that breaks off, else the papers of its entries."""
    if not xml.rstrip().endswith(FEED_END):
        return None
    return [json.loads(entry) for entry in _ENTRY.findall(xml)]


class Pages:
    """The replay fetcher: each paper's feed and full-text page."""

    def get(self, url: str) -> Fetched:
        for p in CORPUS:
            if url == paper_url(p["arxiv_id"]):
                return Fetched(url=url, final_url=url, status=200, body=feed(p).encode(), content_type="")
            if url == fulltext_url(p["arxiv_id"]):
                return Fetched(url=url, final_url=url, status=200, body=b"<html><p>Method text.</p></html>",
                               content_type="")
        raise ReplayMiss(f"no cassette for {url}")


def world() -> World:
    owner = OwnerFunctions(check_call, decide, parse_feed, lambda html, section, n: "Method text.")
    return World(owner, Pages(), CorpusIndex(CORPUS), None,
                 {"speculative-decoding": Topic(frozenset({ID1}), ("speculative decoding",))})


LOOKUP = {"id": "t-lookup", "intent": "lookup_paper", "scenario": "normal", "lang": "en", "split": "dev",
          "family": "e2e/lookup_paper", "text": f"What is {ID2} about?", "slots": {"arxiv_id": ID2},
          "expect": {"status": "answered", "must_write_note": False, "note_about": None,
                     "must_cite_ids_within": "tool_outputs", "expected_tools_any": ["get_paper"]}}
ANSWER = f"{ID2} is about tree verification."
FAULT_KEYS = ["tool", "key", "kind", "ordinal", "step", "call", "call_id"]
NO_SLEEP = FaultSetting(rate=1.0, sleep=lambda seconds: None)


def quiet(**options) -> dict:
    return {"world": world(), "today": TODAY, "log": lambda line: None, **options}


def test_without_a_fault_setting_a_record_has_no_faults_key(tmp_path):
    model = ScriptedModel([turn(call("get_paper", "c1", arxiv_id=ID2)), turn(content=ANSWER)])
    [record] = run_tasks([LOOKUP], model=model, out=tmp_path, **quiet())
    assert "faults" not in record
    transcript = json.loads((tmp_path / record["transcript"]).read_text(encoding="utf-8"))
    assert all("faults" not in step for step in transcript["steps"])


def test_every_fault_a_task_meets_is_in_its_record_and_in_the_step_of_its_transcript(tmp_path):
    turns = [turn(call("search_arxiv", "c1", query="tree verification")),
             turn(call("get_paper", "c2", arxiv_id=ID2), call("fetch_fulltext", "c3", arxiv_id=ID2)),
             turn(call("get_paper", "c4", arxiv_id=ID2)), turn(content=ANSWER)]
    [record] = run_tasks([LOOKUP], model=ScriptedModel(turns), out=tmp_path, seed=3,
                         **quiet(faults=NO_SLEEP))
    faults = record["faults"]
    assert faults and all(list(f) == FAULT_KEYS for f in faults)
    assert all(f["kind"] in FAULTS for f in faults)
    by_call = {f["call_id"]: f for f in faults}  # rate 1: every tool request met a fault
    assert {"c1", "c2", "c3", "c4"} <= set(by_call)
    assert (by_call["c1"]["tool"], by_call["c1"]["step"], by_call["c1"]["call"]) == ("search_arxiv", 1, 0)
    assert by_call["c1"]["key"].startswith("search:")
    paper = canonical_url(paper_url(ID2))
    assert (by_call["c2"]["tool"], by_call["c2"]["key"], by_call["c2"]["ordinal"]) == ("get_paper", paper, 0)
    assert (by_call["c2"]["step"], by_call["c2"]["call"]) == (2, 0)
    assert (by_call["c4"]["step"], by_call["c4"]["call"], by_call["c4"]["key"]) == (3, 1, paper)
    assert by_call["c4"]["ordinal"] == 1 + sum(f["key"] == paper and f["call_id"] == "c3" for f in faults)
    assert by_call["c3"]["tool"] == "fetch_fulltext"
    transcript = json.loads((tmp_path / record["transcript"]).read_text(encoding="utf-8"))
    in_steps = [f for step in transcript["steps"] for f in step.get("faults", [])]
    assert in_steps == faults
    assert all(f["step"] == step["index"] for step in transcript["steps"] for f in step.get("faults", []))


def test_a_task_that_met_no_fault_under_a_setting_has_an_empty_faults_list(tmp_path):
    model = ScriptedModel([turn(call("get_paper", "c1", arxiv_id=ID2)), turn(content=ANSWER)])
    [record] = run_tasks([LOOKUP], model=model, out=tmp_path, **quiet(faults=FaultSetting(rate=0.0)))
    assert record["faults"] == [] and record["success"] is True


def test_a_harness_error_keeps_the_faults_met_before_it_without_a_step(tmp_path):
    model = ScriptedModel([turn(call("get_paper", "c1", arxiv_id=ID2)), RuntimeError("the server went away")])
    [record] = run_tasks([LOOKUP], model=model, out=tmp_path, seed=3, **quiet(faults=NO_SLEEP))
    assert record["status"] == "harness_error" and record["faults"]
    assert all(f["tool"] == "get_paper" and (f["step"], f["call"], f["call_id"]) == (None, None, None)
               for f in record["faults"])


def test_run_tasks_hands_the_tool_times_of_every_task_to_on_tools(tmp_path):
    heard = []
    model = ScriptedModel([turn(call("get_paper", "c1", arxiv_id=ID2)), turn(content=ANSWER)])
    run_tasks([LOOKUP], model=model, out=tmp_path, seed=3,
              **quiet(faults=NO_SLEEP, on_tools=lambda record, tools: heard.append((record["id"], tools))))
    [(task, [timed])] = heard
    assert task == "t-lookup" and list(timed) == ["tool", "step", "call", "wall_ms", "faults"]
    assert (timed["tool"], timed["step"], timed["call"]) == ("get_paper", 1, 0) and timed["faults"]
    assert isinstance(timed["wall_ms"], float)


def seed_with_fault_then_none(task: str, url: str, rate: float) -> int:
    """A seed under which the task's first request for url meets a fault and its second does not."""
    for seed in range(1000):
        if draw(seed, task, url, 0)[0] < rate <= draw(seed, task, url, 1)[0]:
            return seed
    raise AssertionError("no such seed")


def test_two_systems_meet_the_same_faults_on_their_kth_request_for_the_same_url(tmp_path):
    """The per-item pairing (FT, B0, B2): what a task meets on its k-th request for a URL depends on the seed,
    the task and k alone, not on the model, its other calls or their order."""
    paper = canonical_url(paper_url(ID2))
    seed = seed_with_fault_then_none(LOOKUP["id"], paper, 0.5)
    get = call("get_paper", "g", arxiv_id=ID2)
    first = ScriptedModel([turn(get), turn(call("get_paper", "g2", arxiv_id=ID2)), turn(content=ANSWER)],
                          name="ft")
    second = ScriptedModel([turn(call("search_arxiv", "s", query="tree verification")),
                            turn(call("fetch_fulltext", "f", arxiv_id=ID1)),
                            turn(call("get_paper", "x", arxiv_id=ID2)),
                            turn(call("fetch_fulltext", "y", arxiv_id=ID1, section="method")),
                            turn(call("get_paper", "z", arxiv_id=ID2)), turn(content=ANSWER)], name="b0")
    setting = FaultSetting(rate=0.5, sleep=lambda seconds: None)
    records = [run_tasks([LOOKUP], model=model, out=tmp_path / model.name, seed=seed,
                         **quiet(faults=setting))[0] for model in (first, second)]
    met = [[(f["ordinal"], f["kind"]) for f in record["faults"] if f["key"] == paper] for record in records]
    choices = applicable(paper_url(ID2), 200)
    expected = choices[draw(seed, LOOKUP["id"], paper, 0)[1] % len(choices)]
    assert met[0] == met[1] == [(0, expected)]  # the first request met it, the second did not


def test_records_and_transcripts_are_the_same_for_the_same_seed_and_answers(tmp_path):
    outs = []
    for name in ("one", "two"):
        model = ScriptedModel([turn(call("get_paper", "c1", arxiv_id=ID2)), turn(content=ANSWER)])
        [record] = run_tasks([LOOKUP], model=model, out=tmp_path / name, seed=11,
                             **quiet(faults=FaultSetting(rate=0.5, persistent=True, sleep=lambda s: None)))
        outs.append((record, (tmp_path / name / record["transcript"]).read_bytes()))
    assert outs[0] == outs[1]


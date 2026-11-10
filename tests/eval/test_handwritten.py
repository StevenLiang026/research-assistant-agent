"""Hand-written evaluation items (research_agent.eval.handwritten) over the small corpus of the generator's
tests: three papers, speculative decoding as the dev topic and the KV cache as a train topic."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from data.test_synth import CORPUS, ID1, ID2, ID3, TODAY, Pages, parse_feed

from research_agent import policy, validation
from research_agent.agent.loop import RELAX_HINT
from research_agent.agent.prompts import system_prompt
from research_agent.canonical import canonical_json
from research_agent.data.synth import OwnerFunctions, SeedPool
from research_agent.eval import handwritten
from research_agent.eval.handwritten import (
    SpecError,
    World,
    build_item,
    build_items,
    spares,
    stratum_problems,
)
from research_agent.eval.items import item_type
from research_agent.tools.corpus_search import CorpusIndex

REPO = Path(__file__).resolve().parents[2]
REQUEST = "Look on arXiv for work about speculative decoding, please."
TOPIC = {"topic": "speculative decoding", "topic_id": "speculative-decoding"}


def world(references: tuple[str, ...] = ()) -> World:
    pool = SeedPool(topics={"speculative-decoding": (ID1, ID2), "kv-cache": (ID3,)},
                    names={"speculative-decoding": "speculative-decoding", "speculative decoding":
                           "speculative-decoding", "投机解码": "speculative-decoding", "kv-cache": "kv-cache",
                           "kv cache": "kv-cache"},
                    splits={"train": ("kv-cache",), "dev": ("speculative-decoding",), "test": ()})
    owner = OwnerFunctions(validation.check_call, policy.decide, parse_feed,
                           lambda html, section, max_chars: "Method text.")
    return World(index=CorpusIndex(CORPUS), fetcher=Pages(), owner=owner, pool=pool,
                 window=("2026-07-01", TODAY), references=references)


def call(name: str, **arguments) -> dict:
    return {"name": name, "arguments": arguments}


def spec(type_: str = "call", *, request: str = REQUEST, slots: dict | None = None, prior: list | None = None,
         expected: dict | None = None, lang: str = "en", split: str = "dev", sid: str | None = None) -> dict:
    return {"id": sid or f"hw-{split}-001", "split": split, "lang": lang, "type": type_, "request": request,
            "slots": dict(TOPIC) if slots is None else slots, "prior": prior or [],
            "expected": expected or {"tool_calls": [call("search_arxiv", query="speculative decoding")]},
            "rationale": "One sentence."}


def search(**arguments) -> dict:
    return call("search_arxiv", query="speculative decoding", **arguments)


def build(s: dict, w: World | None = None) -> dict:
    return build_item(s, w or world())


def refusal(s: dict, w: World | None = None) -> str:
    with pytest.raises(SpecError) as caught:
        build(s, w)
    assert str(caught.value).startswith(s["id"])
    return str(caught.value)


# ---- building each type --------------------------------------------------------------------------


def test_a_call_item_has_the_system_prompt_and_the_request_as_its_history_and_the_drafted_shape():
    item = build(spec())
    assert item["messages"] == [{"role": "system", "content": system_prompt(TODAY)},
                                {"role": "user", "content": REQUEST}]
    assert item["expected"] == {"content": "", "tool_calls": [
        {"name": "search_arxiv", "arguments": {"query": "speculative decoding"},
         "arguments_json": '{"query": "speculative decoding"}'}]}
    assert (item["type"], item["lang"], item["split"], item["family"]) == ("call", "en", "dev",
                                                                          "handwritten/call/en")
    assert item["source"] == {"handwritten": True}
    assert item["slots"] == TOPIC


def test_a_next_step_item_holds_the_real_tool_output_of_its_prior_step_byte_for_byte():
    s = spec("next_step", request="Search speculative decoding and open the abstract of the top result.",
             prior=[[call("search_arxiv", query="speculative decoding")]],
             expected={"tool_calls": [call("get_paper", arxiv_id=ID1)]})
    item = build(s)
    assistant, tool = item["messages"][2:]
    assert assistant == {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function",
         "function": {"name": "search_arxiv", "arguments": '{"query": "speculative decoding"}'}}]}
    outcome = CorpusIndex(CORPUS).tool()({"query": "speculative decoding"})
    assert tool == {"role": "tool", "tool_call_id": "c1", "content": canonical_json(outcome.content)}
    assert json.loads(tool["content"])[0]["arxiv_id"] == ID1
    assert item["type"] == "next_step"


def test_a_write_note_on_a_paper_retrieved_before_is_a_next_step():
    s = spec("next_step", request=f"Check the abstract of {ID1}, then write a scaffold note on it.",
             slots={"arxiv_id": ID1, "mode": "scaffold"}, prior=[[call("get_paper", arxiv_id=ID1)]],
             expected={"tool_calls": [call("write_note", arxiv_id=ID1, mode="scaffold")]})
    assert build(s)["expected"]["tool_calls"][0]["arguments"] == {"arxiv_id": ID1, "mode": "scaffold"}


def test_a_final_answer_after_the_note_listing_is_a_next_step_without_calls():
    s = spec("next_step", request="Which papers do my notes tagged speculative-decoding cover?",
             slots={**TOPIC, "tag": "speculative-decoding"},
             prior=[[call("list_notes", tag="speculative-decoding")]], expected={"content": "TBD"})
    item = build(s)
    listed = json.loads(item["messages"][-1]["content"])
    assert {n["arxiv_id"] for n in listed} == {ID1, ID2}  # the store was seeded for this item id
    answer = " and ".join(n["arxiv_id"] for n in listed)
    item = build({**s, "expected": {"content": f"They cover {answer}."}})
    assert item["expected"] == {"content": f"They cover {answer}.", "tool_calls": []}


def test_a_no_call_item_expects_the_answer_only():
    s = spec("no_call", request="Thanks, that was all for today!", slots={},
             expected={"content": "You're welcome."})
    item = build(s)
    assert item["expected"] == {"content": "You're welcome.", "tool_calls": []}
    assert item["type"] == "no_call"


def test_a_clarify_item_expects_ask_clarification_with_the_missing_slots():
    s = spec("clarify", request="Summarize the paper for me.", slots={"missing": ["paper_id"]},
             expected={"tool_calls": [call("ask_clarification", question="Which paper?",
                                           missing=["paper_id"])]})
    assert build(s)["type"] == "clarify"


@pytest.mark.parametrize(("name", "arguments", "code", "content"), [
    ("get_paper", {"arxiv_id": ID1}, "RATE_LIMITED", {"error": "RATE_LIMITED"}),
    ("fetch_fulltext", {"arxiv_id": ID1, "section": "method"}, "UPSTREAM_ERROR", {"error": "UPSTREAM_ERROR"}),
    ("search_arxiv", {"query": "speculative decoding"}, "UPSTREAM_ERROR",
     {"error": "UPSTREAM_ERROR", "items": []}),
])
def test_a_recover_item_shows_the_injected_error_as_the_injectors_shape_it_and_expects_the_retry(
        name, arguments, code, content):
    request = f"Get {ID1} for me, the method section if you fetch text, on speculative decoding."
    s = spec("recover", request=request, slots={**TOPIC, "arxiv_id": ID1, "section": "method"},
             prior=[[{**call(name, **arguments), "inject": code}]],
             expected={"tool_calls": [call(name, **arguments)]})
    item = build(s)
    assert json.loads(item["messages"][-1]["content"]) == content
    assert item["type"] == "recover"


def test_an_empty_dated_search_gets_the_loops_relax_hint_and_expects_the_search_without_dates():
    s = spec("recover", request="Speculative decoding since 2026-10-01; if nothing, drop the date.",
             slots={**TOPIC, "date_from": "2026-10-01"},
             prior=[[{**search(date_from="2026-10-01"),
                      "inject": "EMPTY_RESULTS"}]])
    item = build(s)
    assert json.loads(item["messages"][-1]["content"]) == {"error": "EMPTY_RESULTS", "hint": RELAX_HINT,
                                                           "items": []}


def test_a_parallel_item_expects_two_independent_calls():
    s = spec("parallel", request=f"Get the abstracts of {ID1} and {ID2} at once.",
             slots={"arxiv_ids": [ID1, ID2]},
             expected={"tool_calls": [call("get_paper", arxiv_id=ID1), call("get_paper", arxiv_id=ID2)]})
    assert build(s)["type"] == "parallel"


def test_the_type_matches_what_the_drafters_rules_assign():
    item = build(spec())
    output = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": c["name"], "arguments": c["arguments_json"]}}
        for c in item["expected"]["tool_calls"]]}
    assert item_type(item["messages"], output, 1) == item["type"]


def test_building_twice_gives_identical_items():
    specs = [spec(), spec("no_call", request="Thanks!", slots={}, expected={"content": "Welcome."},
                          sid="hw-dev-002")]
    first, problems = build_items(specs, world())
    second, _ = build_items(copy.deepcopy(specs), world())
    assert problems == []
    assert [canonical_json(i) for i in first] == [canonical_json(i) for i in second]


# ---- refusals ------------------------------------------------------------------------------------


def test_a_spec_whose_type_disagrees_with_the_type_rules_is_refused():
    message = refusal(spec("call", request="Thanks!", slots={}, expected={"content": "Welcome."}))
    assert "the type rules give 'no_call'" in message


def test_a_write_note_on_a_paper_never_shown_is_refused_by_check_call():
    s = spec("next_step", request=f"Check {ID1}, then write a scaffold note on {ID2}.",
             slots={"mode": "scaffold"}, prior=[[call("get_paper", arxiv_id=ID1)]],
             expected={"tool_calls": [call("write_note", arxiv_id=ID2, mode="scaffold")]})
    assert "did not appear in any tool output" in refusal(s)


def test_a_test_spec_on_a_train_topic_is_refused():
    s = spec(split="test", request="Search arXiv for KV cache work.",
             slots={"topic": "KV cache", "topic_id": "kv-cache"},
             expected={"tool_calls": [call("search_arxiv", query="KV cache")]})
    message = refusal(s)
    assert "not a test topic" in message
    assert "names topics of another split" in message


def test_a_dev_spec_naming_a_paper_of_another_split_is_refused():
    s = spec(request=f"Pull up {ID3}.", slots={"arxiv_id": ID3},
             expected={"tool_calls": [call("get_paper", arxiv_id=ID3)]})
    assert f"ids outside the dev split's papers: ['{ID3}']" in refusal(s)


def test_a_date_outside_the_recording_window_is_refused():
    s = spec(request="Speculative decoding since 2026-06-01.", slots={**TOPIC, "date_from": "2026-06-01"},
             expected={"tool_calls": [search(date_from="2026-06-01")]})
    assert "dates outside the window" in refusal(s)


def test_a_request_too_close_to_a_generated_request_is_refused():
    _, problems = build_items([spec()], world(references=(REQUEST.replace("please", "thanks"),)))
    assert problems and problems[0].startswith("hw-dev-001: request too close to a generated request")


def test_two_handwritten_requests_too_close_to_each_other_are_refused():
    twin = spec(sid="hw-dev-002", request=REQUEST.replace(".", "!"))
    _, problems = build_items([spec(), twin], world())
    assert problems == [problems[0]] and problems[0].startswith("hw-dev-002: request too close to hw-dev-001")


def test_a_repeated_id_is_refused():
    _, problems = build_items([spec(), spec(request="Find speculative decoding papers on arXiv for me.")],
                              world())
    assert "hw-dev-001: id used 2 times" in problems


def test_an_answer_citing_an_id_absent_from_the_history_is_refused_by_the_scoring():
    s = spec("no_call", request="Thanks!", slots={}, expected={"content": f"See {ID2}."})
    assert "does not score as a success" in refusal(s)


def test_an_argument_the_request_does_not_ask_for_is_refused():
    s = spec(expected={"tool_calls": [search(categories=["cs.CL"])]})
    assert "categories" in refusal(s)


def test_a_default_the_request_does_not_ask_for_is_refused():
    s = spec(expected={"tool_calls": [search(sort_by="relevance")]})
    assert "sort_by" in refusal(s)


def test_a_query_without_required_words_is_refused():
    s = spec(slots={}, expected={"tool_calls": [call("search_arxiv", query="speculative decoding")]})
    assert "has no required words" in refusal(s)


def test_a_recover_item_that_does_not_retry_the_failed_call_is_refused():
    s = spec("recover", request=f"Get {ID1} and {ID2}.", slots={"arxiv_ids": [ID1, ID2]},
             prior=[[{**call("get_paper", arxiv_id=ID1), "inject": "RATE_LIMITED"}]],
             expected={"tool_calls": [call("get_paper", arxiv_id=ID2)]})
    assert "retries the failed one" in refusal(s)


def test_a_relaxed_search_that_keeps_its_date_is_refused():
    s = spec("recover", request="Speculative decoding since 2026-10-01; if nothing, drop the date.",
             slots={**TOPIC, "date_from": "2026-10-01"},
             prior=[[{**search(date_from="2026-10-01"),
                      "inject": "EMPTY_RESULTS"}]],
             expected={"tool_calls": [search(date_from="2026-10-01")]})
    assert "without its date filter" in refusal(s)


def test_a_fault_injected_outside_a_recover_item_is_refused():
    s = spec("next_step", prior=[[{**call("search_arxiv", query="speculative decoding"),
                                   "inject": "RATE_LIMITED"}]],
             expected={"tool_calls": [call("get_paper", arxiv_id=ID1)]})
    assert "only a recover item injects a fault" in refusal(s)


def test_a_prior_step_the_executor_refuses_is_refused():
    s = spec("next_step", prior=[[call("write_note", arxiv_id=ID1)]], expected={"content": "Done."})
    assert "refused by the executor" in refusal(s)


def test_a_clarify_item_must_list_the_slots_it_says_are_missing():
    s = spec("clarify", request="Summarize the paper for me.", slots={"missing": ["paper_id"]},
             expected={"tool_calls": [call("ask_clarification", question="Which?", missing=["topic"])]})
    assert "missing must equal" in refusal(s)


@pytest.mark.parametrize(("change", "words"), [
    ({"lang": "zh"}, "hold Chinese characters"),
    ({"id": "hw-test-001"}, "the id's"),
    ({"rationale": ""}, "rationale must be"),
    ({"expected": {"content": "x", "tool_calls": []}}, "expected must be"),
])
def test_a_malformed_spec_is_refused_before_anything_runs(change, words):
    assert words in refusal({**spec(), **change})


# ---- strata and the script -----------------------------------------------------------------------


def test_each_stratum_must_hold_its_needed_items_plus_the_spares():
    assert [spares(n) for n in (2, 4, 5, 9, 26)] == [1, 1, 2, 2, 6]
    needed = {"strata": {"dev/call/en": {"needed": 1}, "dev/no_call/en": {"needed": 0},
                         "test/call/en": {"needed": 4}}}
    specs = [spec(), spec(sid="hw-dev-002"), spec("no_call", sid="hw-dev-003")]
    assert stratum_problems(specs, needed) == []
    assert stratum_problems(specs[:1], needed) == [
        "stratum dev/call/en: 1 specs, 2 wanted (1 needed + 1 spares)",
        "stratum dev/no_call/en: 0 specs, 1 wanted (0 needed + 1 spares)"]


def load_script():
    path = REPO / "scripts" / "build_handwritten_items.py"
    module_spec = importlib.util.spec_from_file_location("script_build_handwritten_items", path)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def test_the_script_exits_2_without_a_specs_file(tmp_path):
    assert load_script().main(["--specs", str(tmp_path / "none.jsonl")]) == 2


def test_the_script_exits_1_and_writes_nothing_when_a_spec_fails(tmp_path, monkeypatch, capsys):
    script = load_script()
    monkeypatch.setattr(script, "load_world", lambda **kwargs: world())
    specs = tmp_path / "specs.jsonl"
    specs.write_text(json.dumps(spec("call", request="Thanks!", slots={}, expected={"content": "x"})) + "\n",
                     encoding="utf-8")
    out = tmp_path / "items.jsonl"
    assert script.main(["--specs", str(specs), "--out", str(out)]) == 1
    assert not out.exists()
    assert capsys.readouterr().out.startswith("hw-dev-001: ")


def test_the_script_writes_canonical_lines_and_the_counts(tmp_path, monkeypatch, capsys):
    script = load_script()
    monkeypatch.setattr(script, "load_world", lambda **kwargs: world())
    specs = tmp_path / "specs.jsonl"
    specs.write_text(json.dumps(spec()) + "\n", encoding="utf-8")
    out = tmp_path / "items.jsonl"
    assert script.main(["--specs", str(specs), "--out", str(out)]) == 0
    item = build(spec())
    assert out.read_text(encoding="utf-8") == canonical_json(item) + "\n"
    assert "dev call      zh 0  en 1" in capsys.readouterr().out


def test_the_module_docstring_lists_ten_checks():
    assert "10. Wording" in handwritten.__doc__

"""agent/types.py: the one argument normalisation behind parse_arguments, call_key and the search's draw key
(net/faults.py, search_key), and the call keys the loop's dead-loop detection relies on."""

from __future__ import annotations

import dataclasses

from support.fakes import call

from research_agent.agent import types
from research_agent.agent.types import AssistantTurn, ToolCall, call_key, normalize_arguments, parse_arguments


def test_normalize_arguments_turns_integral_floats_into_ints_wherever_they_sit():
    value = {"max_results": 3.0, "nested": [2.0, 1.5, {"deep": 4.0}], "flag": True, "text": "3.0", "n": 7}
    normalized = normalize_arguments(value)
    assert normalized == {"max_results": 3, "nested": [2, 1.5, {"deep": 4}], "flag": True, "text": "3.0",
                          "n": 7}
    assert type(normalized["max_results"]) is int and type(normalized["nested"][0]) is int
    assert type(normalized["nested"][2]["deep"]) is int and type(normalized["nested"][1]) is float
    assert normalized["flag"] is True  # a bool is not a float, so it is left alone
    assert normalize_arguments(5.0) == 5 and type(normalize_arguments(5.0)) is int
    assert normalize_arguments(2.5) == 2.5 and normalize_arguments("x") == "x"
    assert normalize_arguments(None) is None
    assert normalize_arguments([]) == [] and normalize_arguments({}) == {}


def test_normalize_arguments_returns_a_new_structure_and_leaves_its_input_untouched():
    value = {"max_results": 3.0, "nested": [2.0]}
    normalized = normalize_arguments(value)
    assert normalized is not value and normalized["nested"] is not value["nested"]
    assert value == {"max_results": 3.0, "nested": [2.0]} and type(value["max_results"]) is float


def test_the_former_private_name_stays_an_alias_of_normalize_arguments():
    assert types._integral_floats_to_int is normalize_arguments


def test_parse_arguments_normalises_through_the_same_helper():
    text = '{"max_results": 3.0, "nested": [2.0, 1.5]}'
    assert parse_arguments(text) == {"max_results": 3, "nested": [2, 1.5]}
    assert parse_arguments(text) == normalize_arguments({"max_results": 3.0, "nested": [2.0, 1.5]})
    assert parse_arguments("[1.0]") is None and parse_arguments("{") is None  # not a JSON object
    assert parse_arguments("") == {}


def test_call_keys_are_unchanged_for_the_known_cases():
    # pinned strings: the loop's dead-loop detection (agent/loop.py) and the fault tables key by these
    search = call_key(call("search_arxiv", query="x"))
    assert search == 'search_arxiv:{"max_results":5,"query":"x","sort_by":"relevance"}'
    assert call_key(call("search_arxiv", query="x", max_results=5, sort_by="relevance")) == search  # defaults
    reordered = call("search_arxiv", raw='{"sort_by": "relevance", "query": "x", "max_results": 5}')
    assert call_key(reordered) == search  # key order
    assert call_key(call("search_arxiv", raw='{"query": "x", "max_results": 5.0}')) == search  # 5.0 is 5
    assert call_key(call("search_arxiv", query="y")) != search
    pinned = [
        (call("get_paper", arxiv_id="2609.01234"), 'get_paper:{"arxiv_id":"2609.01234"}'),
        (call("fetch_fulltext", arxiv_id="2609.01234"),
         'fetch_fulltext:{"arxiv_id":"2609.01234","max_chars":1500,"section":"introduction"}'),
        (call("write_note", arxiv_id="2609.01234"),
         'write_note:{"arxiv_id":"2609.01234","mode":"scaffold"}'),
        (call("ask_clarification", question="which year?"), 'ask_clarification:{"question":"which year?"}'),
        (call("no_such_tool", a=1), 'no_such_tool:{"a":1}'),  # unknown tool: no defaults to fill
    ]
    for tool_call, key in pinned:
        assert call_key(tool_call) == key
    bad = call("get_paper", raw='{"arxiv_id": ')
    assert bad.arguments is None and call_key(bad) == 'get_paper:{"arxiv_id": '  # unparsable: the raw text


def test_call_key_normalises_arguments_a_caller_built_by_hand():
    # a ToolCall whose arguments did not pass through parse_arguments still keys 3.0 as 3
    by_hand = ToolCall(id="c1", name="search_arxiv", arguments_json='{"query": "x", "max_results": 3.0}',
                       arguments={"query": "x", "max_results": 3.0})
    assert call_key(by_hand) == call_key(call("search_arxiv", query="x", max_results=3))
    assert call_key(by_hand) == 'search_arxiv:{"max_results":3,"query":"x","sort_by":"relevance"}'


# ---- M7: the reasoning and the thinking of a local turn (docs/m7-家族层设计.md §7) ------------------------


def test_reasoning_and_thinking_stay_off_the_api_form_and_out_of_equality():
    plain = AssistantTurn(content="x", tool_calls=[], model="m")
    thinking = {"mode": "think", "reasoning_tokens": 3, "answer_tokens": 1, "forced": False, "phases": 1,
                "seeds": [7]}
    thought = AssistantTurn(content="x", tool_calls=[], model="m", reasoning="because", thinking=thinking)
    assert plain.reasoning is None and plain.thinking is None
    assert thought.to_api() == plain.to_api() == {"role": "assistant", "content": "x"}
    assert thought == plain  # like the signals and the route: histories and records are as they were
    assert dataclasses.replace(thought, route={"answered_by": "local"}).reasoning == "because"
    # appended after the fields before them (M7 WP2 and WP3 added votes and rules after these two)
    assert [f.name for f in dataclasses.fields(AssistantTurn)][-4:-2] == ["reasoning", "thinking"]


# ---- M7 WP2 and WP3: the vote and the rules' changes of a turn (docs/m7-本地提升计划.md) -------------------


def test_votes_and_rules_stay_off_the_api_form_and_out_of_equality_and_default_to_none():
    plain = AssistantTurn(content="x", tool_calls=[], model="m")
    marked = dataclasses.replace(plain, votes={"k": 5, "chosen": "TEXT"}, rules=[])
    assert plain.votes is None and plain.rules is None
    assert marked.to_api() == plain.to_api() and marked == plain
    assert [f.name for f in dataclasses.fields(AssistantTurn)][-2:] == ["votes", "rules"]

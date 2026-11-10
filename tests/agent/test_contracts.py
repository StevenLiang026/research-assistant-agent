import json

from support.fakes import SAMPLE_MESSAGES, call

from research_agent.agent.prompts import PROMPT_HASH, system_prompt
from research_agent.agent.types import ToolCall, ToolResult, api_to_hf, call_key, parse_arguments
from research_agent.canonical import canonical_json
from research_agent.llm.cache import request_key
from research_agent.tools import schemas


def test_seven_tools_with_closed_argument_sets():
    assert schemas.TOOL_NAMES == ["search_arxiv", "get_paper", "fetch_fulltext", "search_notes", "write_note",
                                  "list_notes", "ask_clarification"]
    for tool in schemas.TOOLS:
        assert tool["function"]["parameters"]["additionalProperties"] is False


def test_model_view_drops_only_validation_keywords():
    text = canonical_json(schemas.MODEL_TOOLS)
    for keyword in ("additionalProperties", "minLength", "maxLength", "uniqueItems", '"default"'):
        assert keyword not in text
    assert [t["function"]["name"] for t in schemas.MODEL_TOOLS] == schemas.TOOL_NAMES
    search = schemas.MODEL_TOOLS[0]["function"]["parameters"]["properties"]
    assert search["max_results"]["maximum"] == 5  # caps stay visible to the model


def test_caps_are_in_the_schema():
    props = {t["function"]["name"]: t["function"]["parameters"]["properties"] for t in schemas.TOOLS}
    assert props["search_arxiv"]["max_results"]["maximum"] == schemas.MAX_SEARCH_RESULTS == 5
    assert props["fetch_fulltext"]["max_chars"]["maximum"] == schemas.MAX_FULLTEXT_CHARS == 1500
    assert "open_terms" in props["write_note"]  # 待解术语, missing from the spike prototype


def test_defaults_and_schema_hash():
    assert schemas.DEFAULTS["search_arxiv"] == {"max_results": 5, "sort_by": "relevance"}
    assert schemas.DEFAULTS["write_note"] == {"mode": "scaffold"}
    assert len(schemas.SCHEMA_HASH) == 16 and schemas.SCHEMA_HASH == schemas.SCHEMA_HASH.lower()


def test_canonical_json_is_shared_with_the_response_cache():
    payload = {"b": 1, "a": "推测"}
    assert canonical_json(payload) == '{"a":"推测","b":1}'
    import hashlib
    assert request_key(payload) == hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def test_call_key_fills_defaults_and_ignores_key_order():
    explicit = call("search_arxiv", query="x", max_results=5, sort_by="relevance")
    implicit = call("search_arxiv", raw='{"query": "x"}')
    assert call_key(explicit) == call_key(implicit)
    assert call_key(call("search_arxiv", query="y")) != call_key(implicit)


def test_bad_json_arguments_are_kept_raw():
    bad = call("get_paper", raw='{"arxiv_id": ')
    assert bad.arguments is None and call_key(bad) == 'get_paper:{"arxiv_id": '
    assert parse_arguments("[1, 2]") is None  # valid JSON but not an object


def test_api_round_trip_and_hf_conversion():
    api = call("get_paper", "c9", arxiv_id="2609.01234").to_api()
    assert ToolCall.from_api(api) == call("get_paper", "c9", arxiv_id="2609.01234")
    hf = api_to_hf(SAMPLE_MESSAGES)
    assert hf[2]["tool_calls"][0]["function"]["arguments"] == {"query": "speculative decoding",
                                                                 "date_from": "2026-09-01"}
    assert "tool_call_id" not in hf[3] and hf[3]["role"] == "tool"


def test_tool_result_error_envelope():
    ok = ToolResult("c1", "search_arxiv", [{"arxiv_id": "2609.01234"}])
    assert json.loads(ok.to_api()["content"]) == [{"arxiv_id": "2609.01234"}]
    empty = ToolResult("c1", "search_arxiv", [], error=schemas.EMPTY_RESULTS)
    assert json.loads(empty.to_api()["content"]) == {"error": "EMPTY_RESULTS", "items": []}


def test_integral_floats_become_ints_so_call_keys_agree():
    parsed = parse_arguments('{"max_results": 3.0, "nested": [2.0, 1.5]}')
    assert parsed == {"max_results": 3, "nested": [2, 1.5]}
    assert call_key(call("search_arxiv", raw='{"query": "x", "max_results": 5.0}')) == call_key(
        call("search_arxiv", query="x"))


def test_unparsable_arguments_stay_raw_for_the_template():
    bad = '{"arxiv_id": '
    message = {"role": "assistant", "content": "", "tool_calls": [call("get_paper", "c1", raw=bad).to_api()]}
    hf = api_to_hf([message])
    assert hf[0]["tool_calls"][0]["function"]["arguments"] == bad


def test_every_event_has_its_allowed_actions():
    assert set(schemas.ALLOWED_ACTIONS) == set(schemas.EVENTS)
    for actions in schemas.ALLOWED_ACTIONS.values():
        assert set(actions) <= set(schemas.ACTIONS)
    assert set(schemas.PER_STEP_EVENTS) <= set(schemas.EVENTS)


def test_system_prompt_takes_today_and_is_versioned():
    assert "2026-10-04" in system_prompt("2026-10-04") and len(PROMPT_HASH) == 16

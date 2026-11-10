"""Runtime rules (research_agent.agent.rules; docs/m7-本地提升计划.md WP3): transient_retry and id_repair as
the 10-29 analysis defined them (errors_rules.py), the run limit the loop sets, and RuledModel, which applies
them to a model's answer over the whole history the runner or the loop holds, not the context it sends."""

from __future__ import annotations

import json

from support.fakes import ScriptedModel, table_check, table_decide, table_tools, turn

from research_agent.agent.context import build_context
from research_agent.agent.executor import ToolExecutor
from research_agent.agent.loop import asking, current_history, run_turn
from research_agent.agent.prompts import system_prompt
from research_agent.agent.rules import (
    ID_REPAIR,
    RULES,
    RULES_VERSION,
    TRANSIENT_RETRY,
    RuledModel,
    apply_rules,
    call_runs,
    failed_calls,
    history_ids,
    id_repair,
    ruled_record,
    transient_retry,
)
from research_agent.agent.types import ToolCall, ToolOutcome
from research_agent.canonical import canonical_json
from research_agent.eval.runner import run_items

TODAY = "2026-10-10"
SYSTEM = {"role": "system", "content": system_prompt(TODAY)}
USER = {"role": "user", "content": "Find hallucination papers."}
ID1, ID2 = "2609.00001", "2609.00002"
HITS = [{"arxiv_id": ID1, "title": "Hallucination survey", "snippet": "Extends 2608.05678 to tool use."},
        {"arxiv_id": ID2, "title": "Tree Verification at Scale", "snippet": "Tree verification."}]


def api_call(call_id: str, name: str, arguments: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def tool(call_id: str, body: object) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": canonical_json(body)}


def answer(*calls: dict, content: str = "") -> dict:
    """An assistant message in API form (AssistantTurn.to_api's)."""
    message: dict = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = list(calls)
    return message


def rewritten(*calls: tuple[str, dict], content: str = "") -> dict:
    """What a rule writes (errors_rules.make_output): every call with the id pp_<k> and its arguments as
    json.dumps(arguments, ensure_ascii=False) writes them."""
    return {"role": "assistant", "content": content,
            "tool_calls": [api_call(f"pp_{k}", name, json.dumps(args, ensure_ascii=False))
                           for k, (name, args) in enumerate(calls)]}


SEARCH = api_call("local_0", "search_arxiv", '{"query":"hallucination"}')
GET = api_call("local_1", "get_paper", '{"arxiv_id":"2609.00002"}')
OTHER_SEARCH = api_call("local_0", "search_arxiv", '{"query":"LLM hallucination"}')
FAILED_SEARCH = [SYSTEM, USER, answer(SEARCH), tool("local_0", {"error": "RATE_LIMITED"})]


# ---- transient_retry ---------------------------------------------------------------------------------------


def test_after_a_transient_error_the_answer_becomes_exactly_the_failed_call_again():
    after = transient_retry(FAILED_SEARCH, answer(OTHER_SEARCH))
    assert after == rewritten(("search_arxiv", {"query": "hallucination"}))
    assert failed_calls(FAILED_SEARCH) == [SEARCH]


def test_a_text_answer_after_a_transient_error_becomes_the_retry_with_empty_text():
    after = transient_retry(FAILED_SEARCH, answer(content="The search failed; try again later."))
    assert after == rewritten(("search_arxiv", {"query": "hallucination"}))


def test_in_a_mixed_round_only_the_failed_calls_are_sent_again():
    history = [SYSTEM, USER, answer(SEARCH, GET), tool("local_0", {"error": "UPSTREAM_ERROR"}),
               tool("local_1", HITS[1])]
    after = transient_retry(history, answer(SEARCH, GET))
    assert after == rewritten(("search_arxiv", {"query": "hallucination"}))
    both_failed = [SYSTEM, USER, answer(SEARCH, GET), tool("local_0", {"error": "UPSTREAM_ERROR"}),
                   tool("local_1", {"error": "RATE_LIMITED"})]
    assert transient_retry(both_failed, answer(OTHER_SEARCH)) == rewritten(
        ("search_arxiv", {"query": "hallucination"}), ("get_paper", {"arxiv_id": ID2}))


def test_an_answer_that_already_is_the_retry_is_left_as_it_is():
    # the same calls with the same arguments, whatever their ids, spacing and text: no change at all
    same = answer(api_call("x", "search_arxiv", '{"query": "hallucination"}'), content="Retrying.")
    assert transient_retry(FAILED_SEARCH, same) is None


def test_other_errors_and_a_history_that_does_not_end_with_tool_results_do_not_retry():
    for error in ("EMPTY_RESULTS", "NOT_FOUND", "SKIPPED"):
        history = [SYSTEM, USER, answer(SEARCH), tool("local_0", {"error": error})]
        assert transient_retry(history, answer(OTHER_SEARCH)) is None, error
    assert transient_retry([SYSTEM, USER], answer(OTHER_SEARCH)) is None
    answered = [*FAILED_SEARCH, answer(content="Sorry."), {"role": "user", "content": "Try again."}]
    assert failed_calls(answered) == [] and transient_retry(answered, answer(OTHER_SEARCH)) is None


def test_a_failed_call_that_already_ran_twice_in_the_turn_is_not_retried():
    # the loop answers a third run of a call SKIPPED (agent.loop.MAX_RUNS_PER_CALL): a retry would not run;
    # the call is the same by call_key, its default filled in and its spelling aside
    twice = [*FAILED_SEARCH, answer(api_call("local_0", "search_arxiv", '{"max_results": 5, "query": '
                                                                        '"hallucination"}')),
             tool("local_0", {"error": "RATE_LIMITED"})]
    assert sum(call_runs(twice).values()) == 2 and transient_retry(twice, answer(OTHER_SEARCH)) is None
    # a call answered SKIPPED did not run, and an earlier turn's runs do not count (the loop's are per turn)
    skipped = [*FAILED_SEARCH, answer(SEARCH), tool("local_0", {"error": "SKIPPED", "hint": "..."}),
               answer(SEARCH), tool("local_0", {"error": "RATE_LIMITED"})]
    assert sum(call_runs(skipped).values()) == 2
    earlier_turn = [*twice, answer(content="The search keeps failing."),
                    {"role": "user", "content": "Please try once more."}, answer(SEARCH),
                    tool("local_0", {"error": "RATE_LIMITED"})]
    assert sum(call_runs(earlier_turn).values()) == 1
    assert transient_retry(earlier_turn, answer(OTHER_SEARCH)) == rewritten(
        ("search_arxiv", {"query": "hallucination"}))


def test_one_failed_call_at_the_limit_stops_the_retry_of_the_whole_step():
    at_limit = [*FAILED_SEARCH, answer(SEARCH, GET), tool("local_0", {"error": "RATE_LIMITED"}),
                tool("local_1", {"error": "RATE_LIMITED"})]
    assert call_runs(at_limit)[canonical_key("search_arxiv", {"query": "hallucination"})] == 2
    assert transient_retry(at_limit, answer(OTHER_SEARCH)) is None


def canonical_key(name: str, args: dict) -> str:
    from research_agent.agent.types import call_key

    return call_key(ToolCall(id="k", name=name, arguments_json=json.dumps(args), arguments=args))


# ---- id_repair ---------------------------------------------------------------------------------------------

SEARCHED = [SYSTEM, USER, answer(SEARCH), tool("local_0", HITS)]


def test_an_id_the_history_never_showed_becomes_the_one_shown_id_with_its_sequence_part():
    after = id_repair(SEARCHED, answer(api_call("local_0", "get_paper", '{"arxiv_id":"2509.00002"}')))
    assert after == rewritten(("get_paper", {"arxiv_id": ID2}))
    # an id named only inside a tool output's text counts too (metrics.arxiv_ids over the whole history)
    assert "2608.05678" in history_ids(SEARCHED)
    after = id_repair(SEARCHED, answer(api_call("local_0", "fetch_fulltext",
                                                '{"arxiv_id":"2508.05678","section":"method"}')))
    assert after == rewritten(("fetch_fulltext", {"arxiv_id": "2608.05678", "section": "method"}))


def test_an_id_whose_sequence_part_several_shown_ids_share_is_left_as_it_is():
    history = [*SEARCHED[:3], tool("local_0", [*HITS, {"arxiv_id": "2608.00002", "title": "Other"}])]
    assert id_repair(history, answer(api_call("local_0", "get_paper", '{"arxiv_id":"2509.00002"}'))) is None


def test_a_shown_id_a_malformed_id_and_an_id_outside_arxiv_id_are_left_as_they_are():
    for arguments in ('{"arxiv_id":"2609.00002"}', '{"arxiv_id":"arXiv:2609.00002v2"}',
                      '{"arxiv_id":"not an id"}'):
        assert id_repair(SEARCHED, answer(api_call("local_0", "get_paper", arguments))) is None, arguments
    note = api_call("local_0", "write_note", '{"arxiv_id":"2609.00001","evidence":"see 2509.00002"}')
    assert id_repair(SEARCHED, answer(note)) is None
    assert id_repair(SEARCHED, answer(content="2509.00002 is about trees.")) is None


def test_a_repaired_answer_keeps_its_text_and_writes_every_call_as_the_analysis_does():
    two = answer(api_call("local_0", "get_paper", '{"arxiv_id":"2509.00002"}'),
                 api_call("local_1", "get_paper", '{"arxiv_id":"2609.00001"}'), content="Two papers.")
    assert id_repair(SEARCHED, two) == rewritten(("get_paper", {"arxiv_id": ID2}),
                                                 ("get_paper", {"arxiv_id": ID1}), content="Two papers.")


def test_a_call_whose_arguments_do_not_parse_keeps_its_text_beside_a_repaired_call():
    broken = api_call("local_1", "get_paper", '{"arxiv_id": ')
    after = id_repair(SEARCHED, answer(api_call("local_0", "get_paper", '{"arxiv_id":"2509.00002"}'), broken))
    assert after["tool_calls"][1] == api_call("pp_1", "get_paper", '{"arxiv_id": ')


# ---- the rules together, and offline -----------------------------------------------------------------------


def test_the_rules_apply_in_their_order_and_every_change_is_recorded():
    assert RULES == (TRANSIENT_RETRY, ID_REPAIR) and isinstance(RULES_VERSION, str) and RULES_VERSION
    before = answer(OTHER_SEARCH)
    output, changes = apply_rules(FAILED_SEARCH, before)
    after = rewritten(("search_arxiv", {"query": "hallucination"}))
    assert output == after and changes == [{"rule": TRANSIENT_RETRY, "before": before, "after": after}]
    unchanged = answer(api_call("local_0", "get_paper", '{"arxiv_id":"2609.00002"}'))
    assert apply_rules(SEARCHED, unchanged) == (unchanged, [])
    assert apply_rules(SEARCHED, before, ()) == (before, [])


def test_a_record_offline_gets_the_rules_output_and_its_changes_and_a_failed_record_stays():
    record = {"id": "q-1", "status": "ok", "item": {}, "output": answer(OTHER_SEARCH), "usage": {},
              "paid": False, "cached": False, "model": "m"}
    ruled = ruled_record(record, FAILED_SEARCH)
    assert ruled["output"] == rewritten(("search_arxiv", {"query": "hallucination"}))
    assert ruled["rules"][0]["rule"] == TRANSIENT_RETRY and record["output"] == answer(OTHER_SEARCH)
    assert {k: v for k, v in ruled.items() if k not in ("output", "rules")} == {
        k: v for k, v in record.items() if k != "output"}
    assert ruled_record(record, SEARCHED) == {**record, "rules": []}
    failed = {"id": "q-1", "status": "error", "item": {}, "error": "boom", "error_type": "HarnessError"}
    assert ruled_record(failed, FAILED_SEARCH) == failed


# ---- RuledModel --------------------------------------------------------------------------------------------


def local_turn(*calls: ToolCall, content: str = ""):
    t = turn(*calls, content=content, model="m:lora=1", paid=False)
    t.signals = {"min_decision_prob": 0.4, "mean_logprob": -0.2, "hard_reason": None}
    t.reasoning, t.thinking = "Think.", {"mode": "think", "forced": False}
    return t


def as_call(api: dict) -> ToolCall:
    return ToolCall.from_api(api)


def test_ruled_model_rewrites_the_answer_over_the_whole_history_and_keeps_the_signals():
    # the older search output's snippet names 2608.05678; the context the model is sent cuts it away
    history = [*SEARCHED, answer(GET), tool("local_1", HITS[1])]
    assert "2608.05678" not in json.dumps(build_context(history))
    cite = api_call("local_0", "get_paper", '{"arxiv_id":"2508.05678"}')
    inner = ScriptedModel([local_turn(as_call(cite))])
    model = RuledModel(inner)
    with asking(history):
        got = model.next_turn(build_context(history), [])
    assert inner.contexts == [build_context(history)]
    assert got.to_api() == rewritten(("get_paper", {"arxiv_id": "2608.05678"}))
    assert [c["rule"] for c in got.rules] == [ID_REPAIR]
    assert got.signals["min_decision_prob"] == 0.4 and got.reasoning == "Think." and got.model == "m:lora=1"
    assert model.name == "fake+rules"
    # outside a run the model judges the context it was given: there the id has no match
    alone = RuledModel(ScriptedModel([local_turn(as_call(api_call("local_0", "get_paper",
                                                                  '{"arxiv_id":"2508.05678"}')))]))
    assert current_history() is None
    plain = alone.next_turn(build_context(history), [])
    assert plain.rules == [] and plain.tool_calls[0].arguments == {"arxiv_id": "2508.05678"}


def test_a_ruled_answer_without_a_change_is_the_answer_itself_with_empty_rules():
    original = local_turn(as_call(GET))
    got = RuledModel(ScriptedModel([original])).next_turn(SEARCHED, [])
    assert got == original and got.rules == [] and got.tool_calls == original.tool_calls


def test_the_runner_records_the_rules_on_the_items_whole_history():
    item = {"id": "q-1", "messages": FAILED_SEARCH, "expected": {}, "type": "recover"}
    inner = ScriptedModel([local_turn(as_call(OTHER_SEARCH))])
    (record,) = run_items([item], RuledModel(inner))
    assert record["output"] == rewritten(("search_arxiv", {"query": "hallucination"}))
    assert record["rules"] == [{"rule": TRANSIENT_RETRY, "before": answer(OTHER_SEARCH),
                                "after": record["output"]}]
    assert inner.contexts == [build_context(FAILED_SEARCH)]


def test_in_the_loop_the_rewritten_call_is_the_one_checked_run_and_recorded():
    # the model asks for an id the search never showed; the check refuses it, so without the rule the step
    # would be repaired; with it, the rewritten call passes the check and runs, and the attempt records it
    wrong = '{"arxiv_id": "2509.00002"}'
    check = table_check({f"get_paper:{canonical_json({'arxiv_id': '2509.00002'})}": ["unknown id"]})
    tools, seen = table_tools({"search_arxiv": [ToolOutcome(HITS)], "get_paper": [ToolOutcome(HITS[1])]})
    executor = ToolExecutor(tools, check, today=TODAY)
    inner = ScriptedModel([local_turn(as_call(SEARCH)),
                           local_turn(as_call(api_call("local_0", "get_paper", wrong))),
                           local_turn(content="It is about tree verification.")])
    histories: list[list[dict]] = []

    class Spy:
        name = "spy"

        def next_turn(self, messages, tools):
            histories.append(json.loads(json.dumps(current_history())))
            return inner.next_turn(messages, tools)

    transcript = run_turn(USER["content"], model=RuledModel(Spy()), executor=executor,
                          decide=table_decide(), today=TODAY)
    assert transcript.status == "answered" and seen["get_paper"] == [{"arxiv_id": ID2}]
    attempts = [a for step in transcript.steps for a in step["attempts"]]
    assert attempts[0]["rules"] == [] and [c["rule"] for c in attempts[1]["rules"]] == [ID_REPAIR]
    assert attempts[1]["tool_calls"] == rewritten(("get_paper", {"arxiv_id": ID2}))["tool_calls"]
    assert attempts[1]["issues"] == []
    # the rules saw the loop's whole history: the transcript's messages up to each step
    assert histories == [transcript.messages[:2], transcript.messages[:4], transcript.messages[:6]]

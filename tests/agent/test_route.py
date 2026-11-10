"""Confidence routing (research_agent.agent.route): the token signals of a local step, its hard triggers, and
RoutedModel, which hands a task to the cloud model once the local model is unsure and keeps it there until the
next task begins. Offline, with fabricated token streams and scripted models."""

from __future__ import annotations

import json
import math

import pytest
from support.fakes import ScriptedModel, table_decide, turn

from research_agent.agent.executor import ToolExecutor
from research_agent.agent.loop import run_turn
from research_agent.agent.models import LocalModel
from research_agent.agent.prompts import system_prompt
from research_agent.agent.route import (
    BUDGET_FORCED,
    MEAN_LOGPROB,
    MIN_DECISION_PROB,
    SIGNALS,
    ParamTrigger,
    RouteConfig,
    RoutedModel,
    RouteRules,
    api_calls,
    completion_signals,
    context_facts,
    decision_tokens,
    hard_trigger,
    route_reason,
    step_signals,
)
from research_agent.agent.types import ANSWERED, AssistantTurn, ToolCall
from research_agent.eval.runner import run_items
from research_agent.family.toolformats import CallSpan
from research_agent.family.toolformats.qwen_json import QWEN_JSON
from research_agent.llm.local import Completion
from research_agent.tools.schemas import MODEL_TOOLS, TOOL_NAMES

TODAY = "2026-10-10"
SYSTEM = {"role": "system", "content": system_prompt(TODAY)}
NAMES = [t["function"]["name"] for t in MODEL_TOOLS]

# A call with two arguments, one Chinese, token by token as Qwen3's tokenizer might split it.
CALL = '<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "幻觉", "max_results": 5}}\n</tool_call>'
CALL_TOKENS = ["<tool_call>", "\n", '{"name": "', "search", "_arxiv", '", "arguments": {"query": "', "幻",
               "觉", '", "max_results": ', "5", "}}\n", "</tool_call>"]
CALL_LPS = [-0.01, -0.02, -0.03, -0.2, -0.05, -0.04, -0.7, -0.1, -0.02, -1.2, -0.01, -0.01]
# decision tokens: the first token, both tokens of the name, the first token of each argument value
CALL_DECISIONS = [0, 3, 4, 6, 9]

ANSWER = "你好,请问需要找什么论文?"
ANSWER_TOKENS = ["你好", ",", "请问", "需要", "找", "什么", "论文", "?"]
ANSWER_LPS = [-0.3, -0.1, -0.2, -0.05, -0.4, -0.1, -0.02, -0.01]


def utf8(tokens: list[str]) -> list[list[int]]:
    return [list(t.encode("utf-8")) for t in tokens]


def signals(text: str, tokens: list[str], lps: list[float], token_bytes=None, **kwargs) -> dict:
    options = {"tool_names": NAMES, "today": TODAY, "seen_ids": []} | kwargs
    return step_signals(text, tokens, lps, token_bytes, **options)


# ---- decision tokens ------------------------------------------------------------------------------------


def test_a_calls_decision_tokens_are_the_first_token_the_name_and_each_arguments_first_value_token():
    assert "".join(CALL_TOKENS) == CALL
    assert decision_tokens(CALL, CALL_TOKENS) == CALL_DECISIONS
    assert decision_tokens(CALL, CALL_TOKENS, utf8(CALL_TOKENS)) == CALL_DECISIONS  # through the bytes too


def test_a_plain_answers_only_decision_token_is_the_first_one():
    assert decision_tokens(ANSWER, ANSWER_TOKENS) == [0]
    got = signals(ANSWER, ANSWER_TOKENS, ANSWER_LPS)
    assert got == {MIN_DECISION_PROB: pytest.approx(math.exp(-0.3)),
                   MEAN_LOGPROB: pytest.approx(sum(ANSWER_LPS) / len(ANSWER_LPS)), "hard_reason": None}


def test_min_decision_prob_is_over_the_decision_tokens_and_mean_logprob_over_all_tokens():
    got = signals(CALL, CALL_TOKENS, CALL_LPS)
    assert got[MIN_DECISION_PROB] == pytest.approx(math.exp(-1.2))  # max_results' value "5"
    assert got[MEAN_LOGPROB] == pytest.approx(sum(CALL_LPS) / len(CALL_LPS))
    assert got["hard_reason"] is None


def test_several_arguments_each_give_their_first_value_token_and_a_string_value_starts_after_its_quote():
    text = ('<tool_call>\n{"name": "get_paper", "arguments": {"arxiv_id": "2609.01234", "full": true, '
            '"n": [1, 2]}}\n</tool_call>')
    tokens = ["<tool_call>\n", '{"name": "', "get", "_paper", '", "arguments": {"arxiv_id": ', '"', "2609",
              ".01234", '", "full": ', "true", ', "n": ', "[", "1, 2]}}\n</tool_call>"]
    assert "".join(tokens) == text
    # the opening quote's token is not the value's first character: "2609" is; [ is the list's first one
    assert decision_tokens(text, tokens) == [0, 2, 3, 6, 9, 11]


def test_a_chinese_value_whose_first_character_is_split_across_two_tokens_maps_to_the_first_of_them():
    text = '<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "幻觉"}}\n</tool_call>'
    head = '<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "'
    first = "幻".encode()  # three bytes, split 2 + 1 the way a byte-level BPE can split them
    token_bytes = [list(head.encode()), list(first[:2]), list(first[2:] + "觉".encode()),
                   list(b'"}}\n</tool_call>')]
    tokens = [head, "�", "�觉", '"}}\n</tool_call>']  # what llama-server shows as the token text
    assert b"".join(bytes(b) for b in token_bytes).decode() == text
    assert decision_tokens(text, tokens, token_bytes) == [0, 1]
    got = signals(text, tokens, [-0.01, -0.9, -0.3, -0.01], token_bytes)
    assert got[MIN_DECISION_PROB] == pytest.approx(math.exp(-0.9))


def test_without_bytes_positions_come_from_the_token_strings():
    tokens = ["<tool_call>\n", '{"name": "search_arxiv", "arguments": {"query": "', "幻觉",
              '"}}\n</tool_call>']
    text = "".join(tokens)
    assert decision_tokens(text, tokens) == [0, 1, 2]


def test_no_tokens_give_no_signal_values():
    got = signals(ANSWER, [], [])
    assert got == {MIN_DECISION_PROB: None, MEAN_LOGPROB: None, "hard_reason": None}


# ---- hard triggers --------------------------------------------------------------------------------------


def hard(text: str, **kwargs) -> str | None:
    options = {"tool_names": NAMES, "today": TODAY, "seen_ids": []} | kwargs
    return hard_trigger(text, **options)


def block(name: str, arguments: object) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False) + \
        "\n</tool_call>"


def test_a_plain_answer_and_a_valid_call_have_no_hard_trigger():
    assert hard(ANSWER) is None
    assert hard(CALL) is None


def test_a_block_the_parser_cannot_read_as_a_call_is_a_parse_failure():
    reason = hard('<tool_call>\n{"name": "search_arxiv", "arguments": {"query": }\n</tool_call>')
    assert reason.startswith("parse_failure")
    assert hard("<tool_call>\n[1, 2]\n</tool_call>").startswith("parse_failure")


def test_text_arguments_are_repaired_and_still_invalid_ones_fire():
    assert hard(block("search_arxiv", json.dumps({"query": "KV cache"}))) is None  # repaired into an object
    assert hard(block("search_arxiv", "query=KV cache")).startswith("invalid_after_repair")


def test_a_tool_that_is_not_among_the_steps_tools_fires():
    assert hard(block("search_web", {"query": "x"})).startswith("unknown_tool")
    assert "search_web" in hard(block("search_web", {"query": "x"}))
    # a real tool the step was not offered
    assert hard(CALL, tool_names=["get_paper"]).startswith("unknown_tool")


def test_arguments_the_executors_own_validation_refuses_fire():
    reason = hard(block("search_arxiv", {"max_results": 5}))  # query is required
    assert reason.startswith("invalid_args") and "query" in reason
    late = block("search_arxiv", {"query": "x", "date_from": "2026-12-01"})  # after today
    assert hard(late).startswith("invalid_args")
    assert hard(late, today="2027-01-01") is None
    note = {"arxiv_id": "2609.01234", "mode": "scaffold"}
    assert "did not appear" in (hard(block("write_note", note)) or "")  # the id must have been seen
    assert "did not appear" not in (hard(block("write_note", note), seen_ids=["2609.01234"]) or "")


def test_too_many_calls_in_one_step_fire_as_the_executor_refuses_them():
    many = "\n".join(block("get_paper", {"arxiv_id": f"2609.0123{i}"}) for i in range(5))
    assert hard(many).startswith("invalid_args")


def test_the_facts_of_a_context_are_its_date_and_the_ids_its_tool_outputs_showed():
    messages = [SYSTEM, {"role": "user", "content": "x"},
                {"role": "assistant", "content": "", "tool_calls": []},
                {"role": "tool", "tool_call_id": "c", "content": json.dumps(
                    {"items": [{"arxiv_id": "2609.01234"}, {"arxiv_id": "2609.00001"}]})},
                {"role": "tool", "tool_call_id": "d", "content": "not json"}]
    assert context_facts(messages) == (TODAY, ["2609.00001", "2609.01234"])
    assert context_facts([{"role": "user", "content": "x"}]) == (None, [])


def test_signals_are_a_pure_function_of_the_completion_data():
    first = signals(CALL, CALL_TOKENS, CALL_LPS, utf8(CALL_TOKENS))
    again = signals(CALL, list(CALL_TOKENS), list(CALL_LPS), utf8(CALL_TOKENS))
    assert first == again and json.loads(json.dumps(first)) == first  # JSON round-trips
    assert set(SIGNALS) == {MIN_DECISION_PROB, MEAN_LOGPROB} and SIGNALS[0] == MIN_DECISION_PROB


# ---- LocalModel keeps the signals off the API form ------------------------------------------------------


class TokenClient:
    """LocalClient stand-in: scripted (text, tokens, logprobs) completions in order, requests counted."""

    def __init__(self, completions: list[tuple[str, list[str], list[float]]]) -> None:
        self.completions, self.requests = list(completions), 0

    def complete(self, prompt: str, *, lora_scale: float | None = None,
                 stop: list[str] | None = None) -> Completion:
        self.requests += 1
        text, tokens, lps = self.completions.pop(0)
        return Completion(text=text, tokens=tokens, logprobs=lps, token_bytes=utf8(tokens),
                          raw={"usage": {"prompt_tokens": 10, "completion_tokens": len(tokens)}})


def local_model(*completions) -> LocalModel:
    return LocalModel(TokenClient(list(completions)), lambda m, t: "PROMPT", lora_scale=1.0,
                      name="qwen3-1.7b:lora=1")


def test_a_local_turn_carries_its_signals_outside_the_api_form():
    model = local_model((CALL, CALL_TOKENS, CALL_LPS))
    got = model.next_turn([SYSTEM, {"role": "user", "content": "找幻觉的论文"}], MODEL_TOOLS)
    assert got.signals == signals(CALL, CALL_TOKENS, CALL_LPS, utf8(CALL_TOKENS))
    assert "signals" not in json.dumps(got.to_api())
    bare = AssistantTurn(content=got.content, tool_calls=got.tool_calls, model=got.model, usage=got.usage)
    assert got == bare  # the signals take no part in a turn's equality either


# ---- RoutedModel ----------------------------------------------------------------------------------------


CONFIDENT = (ANSWER, ANSWER_TOKENS, ANSWER_LPS)  # min decision prob exp(-0.3) = 0.74
UNSURE = (ANSWER, ANSWER_TOKENS, [-2.0] + ANSWER_LPS[1:])  # exp(-2.0) = 0.135
UNKNOWN_TOOL = (block("search_web", {"query": "x"}), ["<tool_call>"], [-0.001])
CONFIG = RouteConfig(signal=MIN_DECISION_PROB, threshold=0.5)
CONTEXT = [SYSTEM, {"role": "user", "content": "你好"}]


class PricedCloud(ScriptedModel):
    """A cloud stand-in that prices its paid turns at 0.01."""

    def cost(self, turn: AssistantTurn) -> float:
        return 0.01 if turn.paid else 0.0


def cloud_answers(*contents: str, paid: bool = True, cached: bool = False) -> PricedCloud:
    return PricedCloud([turn(content=c, model="deepseek-flash", paid=paid, cached=cached, mode="thinking")
                        for c in contents], name="deepseek-flash")


def test_a_confident_local_step_stands_and_the_cloud_is_not_asked():
    local, cloud = local_model(CONFIDENT), cloud_answers()
    routed = RoutedModel(local, cloud, CONFIG)
    got = routed.next_turn(CONTEXT, MODEL_TOOLS)
    assert got.content == ANSWER and got.model == "qwen3-1.7b:lora=1" and not got.paid
    assert got.route == {"answered_by": "local", MIN_DECISION_PROB: pytest.approx(math.exp(-0.3)),
                         MEAN_LOGPROB: pytest.approx(sum(ANSWER_LPS) / len(ANSWER_LPS)),
                         "hard_reason": None, "reason": None}
    assert cloud.contexts == [] and routed.cost(got) == 0.0
    assert "route" not in json.dumps(got.to_api())


def test_a_signal_below_the_threshold_routes_this_step_and_every_later_step_of_the_task_to_the_cloud():
    local, cloud = local_model(UNSURE), cloud_answers("云端第一步", "云端第二步")
    routed = RoutedModel(local, cloud, CONFIG)
    first = routed.next_turn(CONTEXT, MODEL_TOOLS)
    assert first.content == "云端第一步" and first.paid and first.mode == "thinking"
    assert first.route["answered_by"] == "cloud" and first.route[MIN_DECISION_PROB] == pytest.approx(
        math.exp(-2.0))
    assert first.route["hard_reason"] is None and MIN_DECISION_PROB in first.route["reason"]
    second = routed.next_turn(CONTEXT, MODEL_TOOLS)  # the local model is not asked again
    assert second.content == "云端第二步" and local.client.requests == 1
    assert second.route["answered_by"] == "cloud" and second.route[MIN_DECISION_PROB] is None
    assert routed.cost(first) == routed.cost(second) == 0.01  # cost, paid and cached are the cloud's


def test_a_hard_trigger_routes_whatever_the_signal():
    local, cloud = local_model(UNKNOWN_TOOL), cloud_answers("云端", cached=True, paid=False)
    got = RoutedModel(local, cloud, CONFIG).next_turn(CONTEXT, MODEL_TOOLS)
    assert got.route["answered_by"] == "cloud" and got.route["hard_reason"].startswith("unknown_tool")
    assert got.route["reason"] == got.route["hard_reason"]
    assert got.cached and not got.paid


def test_the_mean_logprob_signal_is_compared_when_configured():
    config = RouteConfig(signal=MEAN_LOGPROB, threshold=-0.1)  # the answer's mean is -0.1475
    got = RoutedModel(local_model(CONFIDENT), cloud_answers("云端"), config).next_turn(CONTEXT, MODEL_TOOLS)
    assert got.route["answered_by"] == "cloud" and MEAN_LOGPROB in got.route["reason"]
    with pytest.raises(ValueError, match="signal"):
        RouteConfig(signal="entropy", threshold=0.5)


def test_two_consecutive_items_both_start_on_the_local_model():
    local, cloud = local_model(UNSURE, CONFIDENT), cloud_answers("云端")
    routed = RoutedModel(local, cloud, CONFIG)
    items = [{"id": "a", "messages": CONTEXT}, {"id": "b", "messages": CONTEXT}]
    records = run_items(items, routed)
    assert [r["route"]["answered_by"] for r in records] == ["cloud", "local"]
    assert [r["model"] for r in records] == ["deepseek-flash", "qwen3-1.7b:lora=1"]
    assert local.client.requests == 2


def test_two_consecutive_tasks_of_the_loop_both_start_on_the_local_model():
    local, cloud = local_model(UNSURE, CONFIDENT), cloud_answers("云端")
    routed = RoutedModel(local, cloud, CONFIG)
    transcripts = []
    for request in ("第一个任务", "第二个任务"):
        executor = ToolExecutor({}, lambda *a: [], today=TODAY)
        transcripts.append(run_turn(request, model=routed, executor=executor, decide=table_decide(),
                                    today=TODAY, cost_of=routed.cost))
    assert [t.status for t in transcripts] == [ANSWERED, ANSWERED]
    assert [t.final_text for t in transcripts] == ["云端", ANSWER]
    assert (transcripts[0].paid_calls, transcripts[0].local_calls, transcripts[0].cost_rmb) == (1, 0, 0.01)
    assert (transcripts[1].paid_calls, transcripts[1].local_calls, transcripts[1].cost_rmb) == (0, 1, 0.0)
    assert local.client.requests == 2


def test_models_without_the_hook_run_in_the_loop_as_before():
    model = ScriptedModel([turn(content="好的")])
    assert not hasattr(model, "begin_task")
    executor = ToolExecutor({}, lambda *a: [], today=TODAY)
    transcript = run_turn("x", model=model, executor=executor, decide=table_decide(), today=TODAY)
    assert transcript.status == ANSWERED


def test_the_routed_models_name_names_both_models():
    routed = RoutedModel(local_model(), cloud_answers(), CONFIG)
    assert routed.name == "qwen3-1.7b:lora=1>deepseek-flash"


def test_the_tool_names_of_the_step_are_the_ones_given():
    assert set(NAMES) <= set(TOOL_NAMES)


# ---- M6 rules: the text exemption and the parameter triggers (docs/m6-计划.md §3.3 C1, C2) ---------------


TRIGGERS = (ParamTrigger("search_notes", "top_k", (3,)), ParamTrigger("list_notes", "limit", (10,)))
RULES = RouteRules(text_exempt=True, param_triggers=TRIGGERS)
RULES_JSON = {"text_exempt": True, "param_triggers": [
    {"tool": "search_notes", "param": "top_k", "allowed": [3]},
    {"tool": "list_notes", "param": "limit", "allowed": [10]}]}
RULED = RouteConfig(signal=MIN_DECISION_PROB, threshold=0.5, rules=RULES)
UNSURE_SIGNALS = {MIN_DECISION_PROB: 0.1, MEAN_LOGPROB: -1.0, "hard_reason": None}
SURE_SIGNALS = {MIN_DECISION_PROB: 0.9, MEAN_LOGPROB: -0.1, "hard_reason": None}
HARD = "unknown_tool: 'search_web' is not among the step's tools"


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(id="local_0", name=name, arguments_json=json.dumps(arguments), arguments=arguments)


def test_a_text_turn_is_never_routed_by_the_threshold_when_exempt():
    assert route_reason(UNSURE_SIGNALS, RULED, []) is None
    assert route_reason(UNSURE_SIGNALS, CONFIG, []).startswith(MIN_DECISION_PROB)  # no rules: as before
    off = RouteConfig(signal=MIN_DECISION_PROB, threshold=0.5, rules=RouteRules())  # the exemption off
    assert route_reason(UNSURE_SIGNALS, off, []).startswith(MIN_DECISION_PROB)
    # a text turn without the signal's value stays local too: the exemption reads no token
    assert route_reason({**UNSURE_SIGNALS, MIN_DECISION_PROB: None}, RULED, []) is None


def test_the_hard_trigger_still_comes_first():
    habit = [call("search_notes", query="x", top_k=5)]
    assert route_reason({**SURE_SIGNALS, "hard_reason": HARD}, RULED, habit) == HARD
    assert route_reason({**UNSURE_SIGNALS, "hard_reason": HARD}, RULED, []) == HARD  # before the exemption
    assert route_reason(None, RULED, []) == "no signals"


def test_the_default_value_or_a_missing_parameter_does_not_trigger():
    assert route_reason(SURE_SIGNALS, RULED, [call("search_notes", query="x", top_k=3)]) is None
    assert route_reason(SURE_SIGNALS, RULED, [call("search_notes", query="x")]) is None
    assert route_reason(SURE_SIGNALS, RULED, [call("list_notes")]) is None
    assert route_reason(SURE_SIGNALS, RULED, [call("search_arxiv", query="x", max_results=5)]) is None
    # below the threshold such a call is routed by the signal, as before
    reason = route_reason(UNSURE_SIGNALS, RULED, [call("search_notes", query="x", top_k=3)])
    assert reason.startswith(MIN_DECISION_PROB)


def test_a_non_default_value_triggers_with_a_reason_naming_tool_parameter_and_value():
    reason = route_reason(SURE_SIGNALS, RULED, [call("search_notes", query="x", top_k=5)])
    assert reason.startswith("param_trigger") and "search_notes" in reason and "top_k" in reason
    assert "5" in reason and "3" in reason  # the value and what is allowed
    second = route_reason(SURE_SIGNALS, RULED, [call("get_paper", arxiv_id="2609.01234"),
                                                call("list_notes", limit=20)])
    assert second.startswith("param_trigger") and "list_notes" in second and "limit" in second
    assert "20" in second
    assert route_reason(SURE_SIGNALS, RULED, [call("list_notes", limit=3)]).startswith("param_trigger")
    # a call whose arguments are not an object is the hard trigger's business, not the rule's
    broken = ToolCall(id="local_0", name="search_notes", arguments_json="query=x", arguments=None)
    assert route_reason(SURE_SIGNALS, RULED, [broken]) is None


def test_a_configuration_without_rules_behaves_as_before():
    assert CONFIG.rules is None and RouteConfig(signal=MIN_DECISION_PROB, threshold=0.5) == CONFIG
    habit = [call("search_notes", query="x", top_k=5)]
    assert route_reason(SURE_SIGNALS, CONFIG, habit) is None  # the value is not read
    assert route_reason(UNSURE_SIGNALS, CONFIG, habit) == route_reason(UNSURE_SIGNALS, CONFIG)
    assert route_reason(UNSURE_SIGNALS, CONFIG).startswith(MIN_DECISION_PROB)


def test_rules_need_the_steps_calls_and_the_rules_object():
    with pytest.raises(ValueError, match="calls"):
        route_reason(SURE_SIGNALS, RULED)
    with pytest.raises(ValueError, match="rules"):
        RouteConfig(signal=MIN_DECISION_PROB, threshold=0.5, rules=RULES_JSON)  # the JSON form, not rules


def test_rules_round_trip_through_their_json_form_and_malformed_ones_are_refused():
    assert RULES.to_json() == RULES_JSON
    assert RouteRules.from_json(RULES_JSON) == RULES
    assert RouteRules.from_json({}) == RouteRules() == RouteRules(text_exempt=False, param_triggers=())
    assert RouteRules.from_json(RouteRules().to_json()) == RouteRules()
    trigger = {"tool": "search_notes", "param": "top_k", "allowed": [3]}
    for bad, message in [
            ([], "object"), ({"text_exempt": "yes"}, "text_exempt"), ({"x": 1}, "unknown"),
            ({"param_triggers": {}}, "param_triggers"),
            ({"param_triggers": [{"tool": "search_notes", "param": "top_k"}]}, "allowed"),
            ({"param_triggers": [{**trigger, "tool": ""}]}, "tool"),
            ({"param_triggers": [{**trigger, "param": 3}]}, "param"),
            ({"param_triggers": [{**trigger, "allowed": 3}]}, "allowed"),
            ({"param_triggers": [{**trigger, "allowed": [[3]]}]}, "allowed"),
            ({"param_triggers": [{**trigger, "x": 1}]}, "unknown")]:
        with pytest.raises(ValueError, match=message):
            RouteRules.from_json(bad)


def test_a_recorded_turns_calls_are_read_as_the_model_wrote_them():
    message = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "local_0", "type": "function",
         "function": {"name": "search_notes", "arguments": '{"query": "x", "top_k": 5.0}'}}]}
    (got,) = api_calls(message)
    assert (got.name, got.arguments) == ("search_notes", {"query": "x", "top_k": 5})  # 5.0 reads as 5
    assert api_calls({"role": "assistant", "content": "你好"}) == [] and api_calls(None) == []
    assert route_reason(SURE_SIGNALS, RULED, api_calls(message)).startswith("param_trigger")


NOTES_HABIT = block("search_notes", {"query": "KV cache", "top_k": 5})
NOTES_DEFAULT = block("search_notes", {"query": "KV cache", "top_k": 3})
HABIT = (NOTES_HABIT, [NOTES_HABIT], [-0.001])  # one token: min decision prob exp(-0.001), well above 0.5
DEFAULT = (NOTES_DEFAULT, [NOTES_DEFAULT], [-0.001])


def test_the_routed_model_keeps_an_unsure_text_answer_local_under_the_text_exemption():
    local, cloud = local_model(UNSURE), cloud_answers("云端")
    got = RoutedModel(local, cloud, RULED).next_turn(CONTEXT, MODEL_TOOLS)
    assert got.content == ANSWER and got.route["answered_by"] == "local" and got.route["reason"] is None
    assert got.route[MIN_DECISION_PROB] == pytest.approx(math.exp(-2.0))  # the signal is still recorded
    assert cloud.contexts == []


def test_the_routed_model_routes_a_confident_call_with_a_non_default_parameter():
    local, cloud = local_model(HABIT, DEFAULT), cloud_answers("云端")
    routed = RoutedModel(local, cloud, RULED)
    first = routed.next_turn(CONTEXT, MODEL_TOOLS)
    assert first.content == "云端" and first.route["answered_by"] == "cloud"
    assert first.route["reason"].startswith("param_trigger") and "top_k" in first.route["reason"]
    assert first.route["hard_reason"] is None
    assert first.route[MIN_DECISION_PROB] == pytest.approx(math.exp(-0.001))
    routed.begin_task()
    second = routed.next_turn(CONTEXT, MODEL_TOOLS)
    assert second.route["answered_by"] == "local" and second.route["reason"] is None
    assert second.tool_calls[0].arguments == {"query": "KV cache", "top_k": 3}
    # the same two steps without the rules: the first is confident, so it stands
    plain = RoutedModel(local_model(HABIT), cloud_answers("云端"), CONFIG).next_turn(CONTEXT, MODEL_TOOLS)
    assert plain.route["answered_by"] == "local"


def test_a_parameter_trigger_is_named_before_the_threshold_and_compares_as_json_values():
    # the reason names the trigger even when the signal is below the threshold too (the routed set is the
    # same either way); true is not 1, 3 and 3.0 are the same number
    reason = route_reason(UNSURE_SIGNALS, RULED, [call("search_notes", query="x", top_k=5)])
    assert reason.startswith("param_trigger")
    one = ParamTrigger("t", "p", (1,))
    assert one.reason(call("t", p=True)) is not None
    assert one.reason(call("t", p=1.0)) is None


# ---- M6: stepwise routing end to end (docs/m6-计划.md §3.3 C7) --------------------------------------------

UNSURE_CALL = (CALL, CALL_TOKENS, [-2.0] + CALL_LPS[1:])  # a valid call whose first token is improbable


def test_stepwise_routing_decides_every_step_on_its_own_and_never_routes_a_final_text_answer():
    local = local_model(UNSURE_CALL, UNSURE, UNSURE_CALL, CONFIDENT)
    cloud = cloud_answers("云端第一步", "云端第三步")
    routed = RoutedModel(local, cloud, CONFIG, stepwise=True)  # no rules: the threshold alone
    assert routed.stepwise and not RoutedModel(local, cloud, CONFIG).stepwise
    first = routed.next_turn(CONTEXT, MODEL_TOOLS)
    assert first.content == "云端第一步" and first.route["answered_by"] == "cloud"
    assert first.route["reason"].startswith(MIN_DECISION_PROB)
    second = routed.next_turn(CONTEXT, MODEL_TOOLS)  # the local model is asked again: an unsure text answer
    assert second.content == ANSWER and second.route["answered_by"] == "local"
    assert second.route["reason"] is None and second.route[MIN_DECISION_PROB] == pytest.approx(math.exp(-2.0))
    third = routed.next_turn(CONTEXT, MODEL_TOOLS)  # an unsure call again: routed again, on its own
    assert third.content == "云端第三步" and third.route["answered_by"] == "cloud"
    fourth = routed.next_turn(CONTEXT, MODEL_TOOLS)
    assert fourth.route["answered_by"] == "local" and fourth.route["reason"] is None
    assert local.client.requests == 4 and len(cloud.contexts) == 2
    assert routed.cost(first) == routed.cost(third) == 0.01 and routed.cost(second) == 0.0


def test_sticky_routing_is_the_default_and_keeps_a_task_in_the_cloud_once_routed():
    local, cloud = local_model(UNSURE_CALL, UNSURE), cloud_answers("云端第一步", "云端第二步")
    routed = RoutedModel(local, cloud, CONFIG)
    assert routed.next_turn(CONTEXT, MODEL_TOOLS).route["answered_by"] == "cloud"
    second = routed.next_turn(CONTEXT, MODEL_TOOLS)
    assert second.content == "云端第二步" and second.route["reason"] == "routed earlier in this task"
    assert local.client.requests == 1


def test_a_hard_trigger_still_routes_a_step_under_stepwise_routing():
    local, cloud = local_model(UNKNOWN_TOOL, UNKNOWN_TOOL), cloud_answers("云端", "云端")
    routed = RoutedModel(local, cloud, CONFIG, stepwise=True)
    for _ in range(2):
        got = routed.next_turn(CONTEXT, MODEL_TOOLS)
        assert got.route["answered_by"] == "cloud" and got.route["reason"].startswith("unknown_tool")
    assert local.client.requests == 2 and len(cloud.contexts) == 2


def test_stepwise_routing_follows_the_rules_when_the_configuration_has_them():
    local, cloud = local_model(HABIT, UNSURE, DEFAULT), cloud_answers("云端")
    routed = RoutedModel(local, cloud, RULED, stepwise=True)
    assert routed.next_turn(CONTEXT, MODEL_TOOLS).route["reason"].startswith("param_trigger")
    assert routed.next_turn(CONTEXT, MODEL_TOOLS).route["answered_by"] == "local"  # the text exemption
    assert routed.next_turn(CONTEXT, MODEL_TOOLS).route["answered_by"] == "local"  # a confident call
    assert len(cloud.contexts) == 1


# ---- M7: the injected locate, the answer span and the budget trigger (docs/m7-家族层设计.md §8) ------------

PYTHONIC = "search_arxiv(幻觉)"  # a call as another format might write it: the name, then the value
PYTHONIC_TOKENS = ["search", "_arxiv", "(", "幻觉", ")"]
PYTHONIC_LPS = [-0.1, -0.2, -0.3, -0.9, -0.1]


def pythonic_locate(text: str) -> list[CallSpan]:
    return [CallSpan(parsed=True, name=(0, 12), values=(13,))]


def pythonic_parse(text: str) -> list[dict]:
    return [{"name": "search_arxiv", "arguments": {"query": "幻觉"}}]


def test_the_decision_tokens_and_the_hard_trigger_read_the_calls_through_the_injected_locate():
    assert decision_tokens(PYTHONIC, PYTHONIC_TOKENS, locate=pythonic_locate) == [0, 1, 3]
    assert decision_tokens(PYTHONIC, PYTHONIC_TOKENS) == [0]  # the Qwen3 JSON reader sees no block
    got = step_signals(PYTHONIC, PYTHONIC_TOKENS, PYTHONIC_LPS, tool_names=NAMES, today=TODAY, seen_ids=[],
                       parse=pythonic_parse, locate=pythonic_locate)
    assert got[MIN_DECISION_PROB] == pytest.approx(math.exp(-0.9)) and got["hard_reason"] is None
    unread = hard_trigger(PYTHONIC, tool_names=NAMES, today=TODAY, seen_ids=[], parse=pythonic_parse,
                          locate=lambda text: [CallSpan(parsed=False)])
    assert unread.startswith("parse_failure") and "block 1" in unread
    # step_signals and completion_signals hand the injected locate on to the hard trigger too
    def no_call(text: str) -> list[CallSpan]:
        return [CallSpan(parsed=False)]

    unreadable = [step_signals(PYTHONIC, PYTHONIC_TOKENS, PYTHONIC_LPS, tool_names=NAMES, today=TODAY,
                               seen_ids=[], parse=pythonic_parse, locate=no_call),
                  completion_signals(Completion(text=PYTHONIC, tokens=PYTHONIC_TOKENS, logprobs=PYTHONIC_LPS),
                                     MODEL_TOOLS, CONTEXT, parse=pythonic_parse, locate=no_call)]
    for unread_signals in unreadable:
        assert unread_signals["hard_reason"] == unread
        assert unread_signals[MIN_DECISION_PROB] == pytest.approx(math.exp(-0.1))  # the first token alone
    # the default is the Qwen3 JSON reader: the module's own cases read as before
    assert decision_tokens(CALL, CALL_TOKENS, locate=QWEN_JSON.locate) == CALL_DECISIONS
    explicit = signals(CALL, CALL_TOKENS, CALL_LPS, locate=QWEN_JSON.locate)
    assert explicit == signals(CALL, CALL_TOKENS, CALL_LPS)


def test_completion_signals_are_over_the_completion_given_the_answer_span_alone():
    # the reasoning's tokens are not in the answer completion a THINK generation gives (llm.generation), so
    # a doubtful reasoning before a sure answer does not lower the answer's signals
    answer = Completion(text=CALL, tokens=list(CALL_TOKENS), logprobs=list(CALL_LPS),
                        token_bytes=utf8(CALL_TOKENS))
    got = completion_signals(answer, MODEL_TOOLS, [SYSTEM, {"role": "user", "content": "找幻觉的论文"}])
    assert got == signals(CALL, CALL_TOKENS, CALL_LPS, utf8(CALL_TOKENS))
    whole = Completion(text="Doubt.\n</think>\n\n" + CALL, tokens=["Doubt.", "\n</think>\n\n", *CALL_TOKENS],
                       logprobs=[-6.0, -0.01, *CALL_LPS])
    assert completion_signals(whole, MODEL_TOOLS, CONTEXT)[MIN_DECISION_PROB] == pytest.approx(math.exp(-6.0))
    located = completion_signals(Completion(text=PYTHONIC, tokens=PYTHONIC_TOKENS, logprobs=PYTHONIC_LPS),
                                 MODEL_TOOLS, CONTEXT, parse=pythonic_parse, locate=pythonic_locate)
    assert located[MIN_DECISION_PROB] == pytest.approx(math.exp(-0.9))


BUDGET = RouteConfig(signal=MIN_DECISION_PROB, threshold=0.5, rules=RouteRules(budget_forced=True))


def test_the_budget_trigger_routes_a_forced_close_only_when_the_rules_turn_it_on():
    assert BUDGET_FORCED == "budget_forced"
    reason = route_reason(SURE_SIGNALS, BUDGET, [], forced=True)
    assert reason.startswith("budget_forced: ") and "budget" in reason
    assert route_reason(SURE_SIGNALS, BUDGET, [], forced=False) is None
    assert route_reason(SURE_SIGNALS, BUDGET, []) is None  # a step without a generation was never forced
    # the frozen configurations keep their meaning: without the field, or without rules, a forced close is
    # routed by its signals alone
    assert route_reason(SURE_SIGNALS, RULED, [], forced=True) is None
    assert route_reason(SURE_SIGNALS, CONFIG, [], forced=True) is None
    assert route_reason(UNSURE_SIGNALS, CONFIG, [], forced=True).startswith(MIN_DECISION_PROB)
    # the hard reason still comes first; the trigger comes before the parameter triggers and the exemption
    assert route_reason({**SURE_SIGNALS, "hard_reason": HARD}, BUDGET, [], forced=True) == HARD
    assert route_reason(None, BUDGET, [], forced=True) == "no signals"
    both = RouteConfig(signal=MIN_DECISION_PROB, threshold=0.5,
                       rules=RouteRules(text_exempt=True, param_triggers=TRIGGERS, budget_forced=True))
    assert route_reason(UNSURE_SIGNALS, both, [], forced=True).startswith("budget_forced")
    habit = [call("search_notes", query="x", top_k=5)]
    assert route_reason(SURE_SIGNALS, both, habit, forced=True).startswith("budget_forced")
    assert route_reason(SURE_SIGNALS, both, habit, forced=False).startswith("param_trigger")


def test_the_budget_trigger_round_trips_through_the_rules_json_and_stays_out_of_it_when_off():
    rules = RouteRules(budget_forced=True)
    assert rules.to_json() == {"text_exempt": False, "param_triggers": [], "budget_forced": True}
    assert RouteRules.from_json({"budget_forced": True}) == rules
    assert RouteRules.from_json(rules.to_json()) == rules
    assert RouteRules().to_json() == {"text_exempt": False, "param_triggers": []}  # the frozen files' form
    assert RouteRules.from_json({"text_exempt": True}).budget_forced is False
    with pytest.raises(ValueError, match="budget_forced"):
        RouteRules.from_json({"budget_forced": "yes"})
    with pytest.raises(ValueError, match="budget_forced"):
        RouteRules(budget_forced=1)


def thinking_local(*turns: AssistantTurn, forced: bool) -> ScriptedModel:
    """A local stand-in whose turns carry sure signals and a thinking record closed by force or not."""
    for item in turns:
        item.signals = dict(SURE_SIGNALS)
        item.thinking = {"mode": "think", "reasoning_tokens": 64, "answer_tokens": 3, "forced": forced,
                         "phases": 2 if forced else 1, "seeds": [1, 2] if forced else [1]}
    return ScriptedModel(list(turns), name="qwen3-4b:lora=1")


def test_the_routed_model_routes_a_forced_step_under_the_trigger_even_a_final_text_under_stepwise_routing():
    forced = thinking_local(turn(content=ANSWER), turn(content=ANSWER), forced=True)
    routed = RoutedModel(forced, cloud_answers("云端", "云端"), BUDGET, stepwise=True)
    got = routed.next_turn(CONTEXT, MODEL_TOOLS)
    assert got.route["answered_by"] == "cloud" and got.route["reason"].startswith("budget_forced")
    assert got.route["hard_reason"] is None and got.route[MIN_DECISION_PROB] == 0.9
    assert routed.next_turn(CONTEXT, MODEL_TOOLS).route["answered_by"] == "cloud"  # decided and routed again
    closed = thinking_local(turn(content=ANSWER), forced=False)
    stays = RoutedModel(closed, cloud_answers("云端"), BUDGET, stepwise=True).next_turn(CONTEXT, MODEL_TOOLS)
    assert stays.route["answered_by"] == "local" and stays.route["reason"] is None
    off = thinking_local(turn(content=ANSWER), forced=True)
    kept = RoutedModel(off, cloud_answers("云端"), RULED).next_turn(CONTEXT, MODEL_TOOLS)  # the frozen rules
    assert kept.route["answered_by"] == "local" and kept.content == ANSWER

"""DeepSeek's thinking mode in CloudClient and CloudModel (research_agent.llm.cloud and agent.models) against
a strict fake endpoint (support.strict_deepseek): a request with tools that does not pass back the reasoning
of every earlier assistant turn is refused with HTTP 400, as the dev B2 run of 2026-10-15 was refused for its
10 handwritten items with scripted earlier steps.

The order under test: a request cached under its plain key (model, messages, tools and parameters, exactly as
the client sent every request before thinking was handled) is served as it is; on a miss, a history whose
earlier assistant turns all have known reasoning goes out in thinking mode with that reasoning attached on the
wire only; any other history goes out with thinking turned off, under a cache key of its own.
"""

from __future__ import annotations

import copy
from collections import Counter

import openai
import pytest
from support.fakes import HITS, PAPER, table_check, table_decide, table_tools
from support.strict_deepseek import (
    MISSING_REASONING,
    Answer,
    StrictDeepSeek,
    completion,
    sent_reasoning,
    thinks,
)

from research_agent.agent.context import build_context
from research_agent.agent.executor import ToolExecutor
from research_agent.agent.loop import run_turn
from research_agent.agent.models import CloudModel
from research_agent.agent.types import ANSWERED, AssistantTurn, ToolOutcome, Transcript
from research_agent.canonical import canonical_json
from research_agent.eval.runner import OK, run_items
from research_agent.llm.budget import Budget, BudgetExceeded, Price
from research_agent.llm.cache import ResponseCache, request_key
from research_agent.llm.cloud import CloudClient
from research_agent.tools.schemas import MODEL_TOOLS

MODEL = "deepseek-flash"
PRICE = Price(input_cache_miss=2.0, output=8.0)
TODAY = "2026-10-04"
THINKING, NON_THINKING = "thinking", "non-thinking"  # the modes a cloud turn records
OFF = {"thinking": {"type": "disabled"}}  # DeepSeek's documented switch, sent through extra_body
SYSTEM = {"role": "system", "content": "你是科研助理。"}
USER = {"role": "user", "content": "帮我找 2026 年 9 月关于 speculative decoding 的论文"}
SCRIPTED = {"id": "c1", "type": "function",
            "function": {"name": "search_arxiv", "arguments": '{"query": "speculative decoding"}'}}
# A handwritten item's history: its earlier step was scripted (call id c1), so no reasoning exists for it.
HANDWRITTEN = [SYSTEM, USER, {"role": "assistant", "content": "", "tool_calls": [SCRIPTED]},
               {"role": "tool", "tool_call_id": "c1", "content": canonical_json(HITS)}]
SEARCH = ("call_00_s", "search_arxiv", {"query": "speculative decoding", "date_from": "2026-09-01"})
GET = ("call_00_g", "get_paper", {"arxiv_id": "2609.01234"})
ANSWER = "讲自适应草稿长度。"
DEEPSEEK_400 = ("Error code: 400 - {'error': {'message': '" + MISSING_REASONING + "', 'type': "
                "'invalid_request_error', 'param': None, 'code': 'invalid_request_error'}}")


def plain(messages: list[dict]) -> dict:
    """A request as the client sent every request before thinking was handled: what its plain key hashes."""
    return {"model": MODEL, "messages": messages, "tools": MODEL_TOOLS, "temperature": 0.0}


def no_retry(seconds: float) -> None:
    raise AssertionError("the strict endpoint never answers with an error worth a retry")


@pytest.fixture
def clients(tmp_path):
    """make(sdk, cap=1.0) -> a CloudClient for deepseek-flash; all clients of a test share one cache file and
    one ledger file, as the runs of one machine share runs/llm/."""
    opened = []

    def make(sdk: object, cap: float = 1.0) -> CloudClient:
        cache = ResponseCache(tmp_path / "cache.sqlite")
        budget = Budget(tmp_path / "budget.sqlite", stage="M3", stage_cap_rmb=cap, prices={MODEL: PRICE})
        opened.extend([cache, budget])
        return CloudClient(MODEL, cache=cache, budget=budget, client=sdk, sleep=no_retry)

    yield make
    for resource in opened:
        resource.close()


def search_turn(model: CloudModel, hits: list | None = None) -> Transcript:
    """A request run as the generator runs one, over search_arxiv and get_paper stand-ins."""
    tools, _ = table_tools({"search_arxiv": [ToolOutcome(HITS if hits is None else hits)],
                            "get_paper": [ToolOutcome(PAPER)]})
    return run_turn("找推测解码的论文", model=model, executor=ToolExecutor(tools, table_check(), today=TODAY),
                    decide=table_decide(), today=TODAY, cost_of=model.cost)


def keys_of(client: CloudClient) -> tuple[set[str], list[str]]:
    """The keys of the cached answers, and the request keys of the ledger's rows in the order charged."""
    cached = {row[0] for row in client.cache.db.execute("SELECT key FROM responses")}
    charged = [row[0] for row in client.budget.db.execute("SELECT request_key FROM spend ORDER BY rowid")]
    return cached, charged


def without_reasoning(body: dict) -> list[dict]:
    return [{k: v for k, v in m.items() if k != "reasoning_content"} for m in body["messages"]]


def test_the_strict_endpoint_refuses_a_tools_request_that_sends_no_reasoning_back():
    """What the client sent before this change, sent straight through the SDK, meets the dev B2 run's error;
    the same history passes with the reasoning sent back, or with thinking off."""
    server = StrictDeepSeek([Answer(ANSWER), Answer("同上。")])
    sdk = server.sdk()
    with pytest.raises(openai.BadRequestError) as info:
        sdk.chat.completions.create(**plain(HANDWRITTEN))
    assert info.value.status_code == 400 and str(info.value) == DEEPSEEK_400
    back = [{**m, "reasoning_content": "先检索。"} if m["role"] == "assistant" else m for m in HANDWRITTEN]
    passed = sdk.chat.completions.create(**{**plain(HANDWRITTEN), "messages": back})
    assert passed.choices[0].message.content == ANSWER
    off = sdk.chat.completions.create(**plain(HANDWRITTEN), extra_body=OFF).model_dump(exclude_none=True)
    assert off["choices"][0]["message"] == {"role": "assistant", "content": "同上。"}  # and no reasoning
    assert [thinks(body) for body in server.bodies] == [True, True, False] and server.refused == 1


def test_a_request_cached_under_the_old_key_is_served_without_an_http_call(clients):
    server = StrictDeepSeek()
    client = clients(server.sdk())
    stored = completion(Answer(calls=(GET,), reasoning="读第一篇。"), thinking=True)  # as the M2 cache has it
    client.cache.put(request_key(plain(HANDWRITTEN)), MODEL, stored)
    # the earlier turn's reasoning is unknown, yet the answer cached under the plain key comes back as it is
    assert client.chat(copy.deepcopy(HANDWRITTEN), tools=MODEL_TOOLS, reasoning_of=lambda m: None,
                       temperature=0.0) == stored
    model = CloudModel(client)
    got = model.next_turn(build_context(HANDWRITTEN), MODEL_TOOLS)
    assert server.bodies == [] and (client.calls, client.cache_hits, client.budget.spent()) == (0, 2, 0.0)
    assert (got.paid, got.cached, got.mode) == (False, True, THINKING)
    assert model.reasoning == {canonical_json(got.to_api()): "读第一篇。"}  # learnt from the cached answer


def test_a_multi_step_loop_completes_against_the_strict_endpoint(clients):
    server = StrictDeepSeek([Answer(calls=(SEARCH,), reasoning="先检索。"),
                             Answer(calls=(GET,), reasoning="再读第一篇。"),
                             Answer(ANSWER, reasoning="可以回答了。")])
    client = clients(server.sdk())
    model = CloudModel(client)
    t = search_turn(model)
    assert (t.status, t.final_text, t.paid_calls, server.refused) == (ANSWERED, ANSWER, 3, 0)
    # every request passes back, on each earlier assistant message, the reasoning the endpoint gave with it
    sent = [sent_reasoning(body) for body in server.bodies]
    assert sent == [[], ["先检索。"], ["先检索。", "再读第一篇。"]]
    assert not any("thinking" in body for body in server.bodies)
    assert model.modes == Counter({THINKING: 3})
    # on the wire only: the history and the transcript carry no reasoning, and every answer is cached and
    # charged under the plain key of the request as the loop built it
    contexts = [build_context(t.messages[:n]) for n in (2, 4, 6)]
    assert [without_reasoning(body) for body in server.bodies] == contexts
    assert "reasoning_content" not in t.to_json()
    keys = [request_key(plain(context)) for context in contexts]
    assert keys_of(client) == (set(keys), keys)


def test_the_reasoning_goes_on_copies_and_the_callers_messages_stay_as_they_were(clients):
    server = StrictDeepSeek([Answer(calls=(GET,))])
    client = clients(server.sdk())
    messages = copy.deepcopy(HANDWRITTEN)
    before, assistant = copy.deepcopy(messages), messages[2]
    client.chat(messages, tools=MODEL_TOOLS, reasoning_of=lambda m: "先检索。", temperature=0.0)
    assert messages == before and messages[2] is assistant and "reasoning_content" not in assistant
    assert sent_reasoning(server.bodies[0]) == ["先检索。"] and client.last_mode == THINKING
    assert keys_of(client) == ({request_key(plain(before))}, [request_key(plain(before))])


def test_a_handwritten_history_runs_with_thinking_off_under_a_key_of_its_own(clients):
    item = {"id": "hw-dev-023", "messages": HANDWRITTEN,
            "expected": {"tool_calls": [{"name": "get_paper", "arguments": {"arxiv_id": "2609.01234"}}]}}
    server = StrictDeepSeek([Answer(calls=(GET,))])
    client = clients(server.sdk())
    model = CloudModel(client)
    [record] = run_items([item], model)
    assert record["status"] == OK, record.get("error")
    assert (record["mode"], record["paid"], server.refused) == (NON_THINKING, True, 0)
    [body] = server.bodies
    assert body["thinking"] == OFF["thinking"] and sent_reasoning(body) == [None]  # not even an empty one
    assert model.reasoning == {}  # an answer without thinking has no reasoning to learn
    request = plain(build_context(HANDWRITTEN))
    off_key = request_key({**request, "extra_body": OFF})  # the key composition, pinned
    assert off_key != request_key(request) and keys_of(client) == ({off_key}, [off_key])
    # a rerun is free: the answer comes from the thinking-off key, labelled as it was given
    rerun = StrictDeepSeek()
    [again] = run_items([item], CloudModel(clients(rerun.sdk())))
    assert rerun.bodies == [] and again["output"] == record["output"]
    assert (again["paid"], again["cached"], again["mode"]) == (False, True, NON_THINKING)


def test_an_answer_is_served_only_in_the_mode_it_was_given_in(clients):
    """An answer given without thinking is never served for a request whose reasoning is known; one given in
    thinking mode keeps that label whoever is served it."""
    first = StrictDeepSeek([Answer(calls=(SEARCH,), reasoning="先检索。"), Answer("思考后的回答。")])
    knowing = CloudModel(clients(first.sdk()))
    step = knowing.next_turn([SYSTEM, USER], MODEL_TOOLS)
    history = [SYSTEM, USER, step.to_api(),
               {"role": "tool", "tool_call_id": "call_00_s", "content": canonical_json(HITS)}]
    blind_server = StrictDeepSeek([Answer("关闭思考时的回答。")])
    blind = CloudModel(clients(blind_server.sdk()))  # another instance: it never saw that turn's reasoning
    off = blind.next_turn(history, MODEL_TOOLS)
    assert (off.content, off.mode, off.paid) == ("关闭思考时的回答。", NON_THINKING, True)
    on = knowing.next_turn(history, MODEL_TOOLS)
    assert (on.content, on.mode, on.paid) == ("思考后的回答。", THINKING, True)
    assert sent_reasoning(first.bodies[1]) == ["先检索。"] and "thinking" not in first.bodies[1]
    # the thinking answer, now under the plain key, is served from then on, still labelled thinking
    again = blind.next_turn(history, MODEL_TOOLS)
    assert (again.content, again.mode, again.cached) == ("思考后的回答。", THINKING, True)
    assert len(blind_server.bodies) == 1


def test_an_empty_reasoning_counts_as_unknown_and_is_never_sent(clients):
    server = StrictDeepSeek([Answer(calls=(SEARCH,), reasoning=""), Answer(ANSWER)])
    model = CloudModel(clients(server.sdk()))
    t = search_turn(model)
    assert (t.status, t.final_text, server.refused) == (ANSWERED, ANSWER, 0)
    assert "thinking" not in server.bodies[0] and server.bodies[1]["thinking"] == OFF["thinking"]
    assert sent_reasoning(server.bodies[1]) == [None]
    assert model.reasoning == {} and model.modes == Counter({THINKING: 1, NON_THINKING: 1})


def test_a_known_reasoning_before_an_unknown_one_does_not_keep_thinking_on(clients):
    """Rule 3 weighs every earlier turn, not only the first: the second step's reasoning came back empty, so
    the third request goes out with thinking off although the first step's reasoning is known."""
    server = StrictDeepSeek([Answer(calls=(SEARCH,), reasoning="先检索。"),
                             Answer(calls=(GET,), reasoning=""), Answer(ANSWER)])
    model = CloudModel(clients(server.sdk()))
    t = search_turn(model)
    assert (t.status, t.final_text, server.refused) == (ANSWERED, ANSWER, 0)
    assert [thinks(body) for body in server.bodies] == [True, True, False]
    assert [sent_reasoning(body) for body in server.bodies] == [[], ["先检索。"], [None, None]]
    assert model.modes == Counter({THINKING: 2, NON_THINKING: 1})


def test_a_known_reasoning_after_an_unknown_one_does_not_turn_thinking_on(clients):
    """Rule 3 weighs every earlier turn, not only the latest: a handwritten item's scripted step, then a step
    served from the cache under its plain key with its reasoning (as the M2 cache holds such steps), so the
    next request's latest earlier turn has a known reasoning and its first one none; it goes out with thinking
    off."""
    server = StrictDeepSeek([Answer(ANSWER)])
    client = clients(server.sdk())
    stored = completion(Answer(calls=(GET,), reasoning="读第一篇。"), thinking=True)
    client.cache.put(request_key(plain(HANDWRITTEN)), MODEL, stored)
    model = CloudModel(client)
    step = model.next_turn(build_context(HANDWRITTEN), MODEL_TOOLS)
    assert (step.cached, step.mode, model.reasoning_of(step.to_api())) == (True, THINKING, "读第一篇。")
    history = [*HANDWRITTEN, step.to_api(),
               {"role": "tool", "tool_call_id": GET[0], "content": canonical_json(PAPER)}]
    answer = model.next_turn(build_context(history), MODEL_TOOLS)
    assert (answer.content, answer.mode, answer.paid, server.refused) == (ANSWER, NON_THINKING, True, 0)
    [body] = server.bodies
    assert not thinks(body) and sent_reasoning(body) == [None, None]


def test_reasoning_learnt_from_a_cache_hit_is_passed_back(clients):
    first = StrictDeepSeek([Answer(calls=(SEARCH,), reasoning="先检索。"), Answer("找到两篇。")])
    search_turn(CloudModel(clients(first.sdk())))
    second = StrictDeepSeek([Answer("只找到一篇。")])
    fresh = CloudModel(clients(second.sdk()))  # another instance over the same cache: it knows nothing yet
    t = search_turn(fresh, hits=HITS[:1])  # step 1 comes from the cache; step 2 has another tool result
    assert (t.status, t.final_text, t.cache_hits, t.paid_calls) == (ANSWERED, "只找到一篇。", 1, 1)
    [body] = second.bodies
    assert sent_reasoning(body) == ["先检索。"] and "thinking" not in body


def test_a_repair_request_passes_back_the_reasoning_of_the_rejected_step(clients):
    garbled = ("call_00_x", "get_paper", '{"arxiv_id": ')
    server = StrictDeepSeek([Answer(calls=(garbled,), reasoning="读这篇。"),
                             Answer(calls=(GET,), reasoning="补全参数。"), Answer(ANSWER)])
    model = CloudModel(clients(server.sdk()))
    tools, _ = table_tools({"get_paper": [ToolOutcome(PAPER)]})
    t = run_turn("2609.01234 讲什么?", model=model, executor=ToolExecutor(tools, table_check(), today=TODAY),
                 decide=table_decide({("bad_json", 1): "repair"}), today=TODAY)
    assert (t.status, t.final_text, server.refused) == (ANSWERED, ANSWER, 0)
    # the repair request carries the rejected step with its reasoning; the next request only the accepted one
    assert [sent_reasoning(body) for body in server.bodies] == [[], ["读这篇。"], ["补全参数。"]]
    assert not any("thinking" in body for body in server.bodies)


def test_a_request_without_earlier_assistant_turns_is_sent_exactly_as_before(clients):
    server = StrictDeepSeek([Answer(calls=(SEARCH,)), Answer("改写后的请求。")])
    client = clients(server.sdk())

    def never(message: dict) -> str:
        raise AssertionError("there is no earlier turn to look up")

    client.chat([SYSTEM, USER], tools=MODEL_TOOLS, reasoning_of=never, temperature=0.0)
    assert client.last_mode == THINKING
    rewrite = {"model": MODEL, "messages": [{"role": "system", "content": "改写这条请求。"}, USER],
               "temperature": 0.7}
    client.chat(rewrite["messages"], temperature=0.7)  # the generator's rewriter: no tools, no reasoning_of
    assert server.bodies == [plain([SYSTEM, USER]), rewrite] and client.last_mode == THINKING
    keys = [request_key(plain([SYSTEM, USER])), request_key(rewrite)]
    assert keys_of(client) == (set(keys), keys)


def test_the_cap_is_checked_after_both_lookups_and_before_the_sdk(clients):
    client = clients(object(), cap=0.0)  # nothing may be paid; object() has no chat attribute to reach
    stored = completion(Answer("关闭思考时的回答。"), thinking=False)
    client.cache.put(request_key({**plain(HANDWRITTEN), "extra_body": OFF}), MODEL, stored)
    assert client.chat(HANDWRITTEN, tools=MODEL_TOOLS, temperature=0.0) == stored
    assert client.last_mode == NON_THINKING
    with pytest.raises(BudgetExceeded):  # the reasoning is known: the answer without thinking is not served
        client.chat(HANDWRITTEN, tools=MODEL_TOOLS, reasoning_of=lambda m: "先检索。", temperature=0.0)
    with pytest.raises(BudgetExceeded):  # a thinking request that misses
        client.chat(HANDWRITTEN[:2], tools=MODEL_TOOLS, temperature=0.0)
    unknown = [*HANDWRITTEN[:3], {"role": "tool", "tool_call_id": "c1", "content": "[]"}]
    with pytest.raises(BudgetExceeded):  # a thinking-off request that misses
        client.chat(unknown, tools=MODEL_TOOLS, temperature=0.0)
    assert client.calls == 0 and client.cache_hits == 1


def test_a_caller_cannot_choose_the_mode_through_extra_body(clients):
    """The client decides the mode: a caller's extra_body with a "thinking" key would be labelled thinking and
    cached under the plain key even when it turns thinking off, so it is refused before the cache, the ledger
    or the SDK is touched."""
    client = clients(object(), cap=0.0)  # nothing may be paid; object() has no chat attribute to reach
    for switch in (OFF, {"thinking": {"type": "enabled"}}):
        with pytest.raises(ValueError, match="extra_body.*thinking"):
            client.chat([SYSTEM, USER], tools=MODEL_TOOLS, temperature=0.0, extra_body=switch)
    assert (client.calls, client.cache_hits, client.last_mode) == (0, 0, None)
    assert keys_of(client) == (set(), [])


def test_last_mode_is_the_mode_of_the_latest_answer_obtained_even_when_the_ledger_stops_after_it(clients):
    """A paid answer is cached and charged before the ledger raises BudgetExceeded, and last_mode keeps its
    mode; a call the ledger refuses before any answer leaves None."""
    server = StrictDeepSeek([Answer(calls=(SEARCH,), reasoning="先检索。")])
    client = clients(server.sdk(), cap=0.001)  # one answer costs 0.0024 RMB at PRICE: it passes the cap
    with pytest.raises(BudgetExceeded):
        client.chat([SYSTEM, USER], tools=MODEL_TOOLS, temperature=0.0)
    assert (client.calls, client.last_mode) == (1, THINKING)
    with pytest.raises(BudgetExceeded):  # refused before the SDK is touched: this call has no answer
        client.chat([SYSTEM, {"role": "user", "content": "谢谢"}], tools=MODEL_TOOLS, temperature=0.0)
    assert client.last_mode is None and len(server.bodies) == 1


def test_the_reasoning_of_the_latest_answer_of_a_form_is_kept(clients):
    """Two answers with the same API form: the later reasoning wins, so in the loop the reasoning sent back
    with a turn is always the one the model gave with that very turn."""
    server = StrictDeepSeek([Answer("好的。", reasoning="第一次。"), Answer("好的。", reasoning="第二次。")])
    model = CloudModel(clients(server.sdk()))
    first = model.next_turn([SYSTEM, {"role": "user", "content": "谢谢"}], MODEL_TOOLS)
    model.next_turn([SYSTEM, {"role": "user", "content": "多谢"}], MODEL_TOOLS)
    assert model.reasoning == {canonical_json(first.to_api()): "第二次。"}


def test_a_turn_keeps_its_mode_out_of_its_api_form():
    from research_agent.llm.cloud import NON_THINKING as non_thinking
    from research_agent.llm.cloud import THINKING as thinking
    from research_agent.llm.cloud import THINKING_OFF

    assert (thinking, non_thinking, THINKING_OFF) == (THINKING, NON_THINKING, OFF)
    local = AssistantTurn(ANSWER, [], "qwen3-1.7b:lora=1")  # positional, as eval/handwritten.py builds turns
    assert local.mode is None
    cloud = AssistantTurn(ANSWER, [], MODEL, mode=THINKING)
    assert cloud.to_api() == local.to_api() == {"role": "assistant", "content": ANSWER}

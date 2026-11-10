import copy
import dataclasses
import json
import math

import pytest
from support.fakes import HITS, PAPER, SAMPLE_MESSAGES, call, table_check, table_decide, table_tools
from support.family_fakes import TURN_END_ID
from support.llama_fakes import ID_BASE, FakeLlama, FakeToolFormat, Scripted, fake_loaded_family, scripted

from research_agent.agent.executor import ToolExecutor
from research_agent.agent.loop import REFUSED_STEP_HINT, run_turn
from research_agent.agent.models import STOP, CloudModel, LocalModel
from research_agent.agent.route import MEAN_LOGPROB, MIN_DECISION_PROB, completion_signals
from research_agent.agent.types import ANSWERED, AssistantTurn, HarnessError, ToolCall, ToolOutcome, api_to_hf
from research_agent.family.core import GREEDY, Decoding, Mode
from research_agent.family.toolformats import CallSpan
from research_agent.llm.budget import Budget, BudgetExceeded, Price, cost_rmb
from research_agent.llm.cache import ResponseCache
from research_agent.llm.cloud import CloudClient
from research_agent.llm.generation import GenerationSettings, request_seed
from research_agent.llm.local import Completion
from research_agent.tools.schemas import MODEL_TOOLS

TODAY = "2026-10-04"
FLASH = Price(input_cache_miss=2.0, output=8.0)
USAGE = {"prompt_tokens": 1200, "completion_tokens": 40, "total_tokens": 1240}
LOCAL_USAGE = {"prompt_tokens": 1050, "completion_tokens": 25, "total_tokens": 1075}
MESSAGES = SAMPLE_MESSAGES[:2]
TEXT = '<tool_call>\n{"name": "get_paper", "arguments": {"arxiv_id": "2609.01234"}}\n</tool_call>'
GARBLED = '<tool_call>\n{"name": "get_paper", "arguments": {"arxiv_id": </tool_call>'
NOT_JSON = "arguments are not valid JSON: Expecting value at character 13"  # the executor on '{"arxiv_id": '


class FakeResponse:
    def __init__(self, data: dict) -> None:
        self.data = data

    def model_dump(self, **_) -> dict:
        return copy.deepcopy(self.data)


class FakeOpenAI:
    """Stands in for openai.OpenAI: scripted response dicts in order, every payload recorded."""

    def __init__(self, script: list[dict]) -> None:
        self.script, self.payloads = list(script), []
        self.chat = type("Chat", (), {})()
        self.chat.completions = self

    def create(self, **payload) -> FakeResponse:
        self.payloads.append(copy.deepcopy(payload))
        return FakeResponse(self.script.pop(0))


def response(*calls: ToolCall, content: str | None = "", model: str | None = "flash-2026-09") -> dict:
    """A chat-completions response as CloudClient returns it (model_dump without None values)."""
    message: dict = {"role": "assistant"}
    if content is not None:
        message["content"] = content
    if calls:
        message["tool_calls"] = [{**c.to_api(), "index": i} for i, c in enumerate(calls)]
    data = {"id": "chatcmpl-1", "object": "chat.completion", "usage": USAGE,
            "choices": [{"index": 0, "finish_reason": "tool_calls" if calls else "stop", "message": message}]}
    if model is not None:
        data["model"] = model
    return data


@pytest.fixture
def make_cloud():
    """CloudClient for model "flash" over a scripted SDK, an in-memory cache and an in-memory ledger."""
    opened = []

    def make(script: list[dict], cap: float = 1.0) -> tuple[CloudClient, FakeOpenAI]:
        fake = FakeOpenAI(script)
        cache, budget = ResponseCache(":memory:"), Budget(":memory:", stage="M1", stage_cap_rmb=cap,
                                                          prices={"flash": FLASH})
        opened.extend([cache, budget])
        return CloudClient("flash", cache=cache, budget=budget, client=fake), fake

    yield make
    for resource in opened:
        resource.close()


class FakeLocalClient:
    """Stands in for LocalClient: scripted completion texts in order, every request recorded."""

    def __init__(self, texts: list[str]) -> None:
        self.texts, self.requests = list(texts), []

    def complete(self, prompt: str, *, lora_scale: float | None = None,
                 stop: list[str] | None = None) -> Completion:
        self.requests.append({"prompt": prompt, "lora_scale": lora_scale, "stop": stop})
        return Completion(text=self.texts.pop(0), raw={"usage": LOCAL_USAGE})


def recording_render():
    """render stand-in: records (messages, tools) and returns a numbered prompt."""
    seen = []

    def render(messages: list[dict], tools: list[dict] | None = None) -> str:
        seen.append((copy.deepcopy(messages), tools))
        return f"PROMPT {len(seen)}"

    render.seen = seen
    return render


def table_parse(table: dict[str, list[dict]]):
    """parse_tool_calls stand-in: scripted calls per completion text, [] for any other text."""

    def parse_tool_calls(text: str) -> list[dict]:
        return copy.deepcopy(table.get(text, []))

    return parse_tool_calls


# CloudModel


def test_cloud_turn_converts_the_response_message(make_cloud):
    search = call("search_arxiv", "call_0", query="speculative decoding")
    client, fake = make_cloud([response(search)])
    model = CloudModel(client)
    got = model.next_turn(MESSAGES, MODEL_TOOLS)
    assert got == AssistantTurn(content="", tool_calls=[search], model="flash-2026-09", usage=USAGE,
                                paid=True, mode="thinking")
    assert fake.payloads == [{"model": "flash", "messages": MESSAGES, "tools": MODEL_TOOLS,
                              "temperature": 0.0}]
    assert model.name == "flash"  # the price-list name; the turn keeps the name the server reported


def test_a_cached_turn_is_not_paid_and_costs_nothing(make_cloud):
    client, fake = make_cloud([response(content="你好")])
    model = CloudModel(client)
    first, second = model.next_turn(MESSAGES, MODEL_TOOLS), model.next_turn(MESSAGES, MODEL_TOOLS)
    assert (first.paid, second.paid) == (True, False) and first.content == second.content == "你好"
    assert (first.cached, second.cached) == (False, True)
    assert model.cost(first) == pytest.approx(cost_rmb(FLASH, USAGE))
    assert client.budget.spent("M1") == pytest.approx(model.cost(first))
    assert model.cost(second) == 0.0 and len(fake.payloads) == 1


def test_temperature_is_sent_and_a_missing_content_or_model_name_falls_back(make_cloud):
    client, fake = make_cloud([response(call("get_paper", "call_0", arxiv_id="2609.01234"), content=None,
                                        model=None)])
    got = CloudModel(client, temperature=0.7).next_turn(MESSAGES, MODEL_TOOLS)
    assert fake.payloads[0]["temperature"] == 0.7
    assert (got.content, got.model) == ("", "flash")


def without_ids(data: dict, ids: list[str | None]) -> dict:
    """The response with its call ids replaced; None drops the id altogether."""
    for api_call, call_id in zip(data["choices"][0]["message"]["tool_calls"], ids, strict=True):
        if call_id is None:
            del api_call["id"]
        else:
            api_call["id"] = call_id
    return data


def test_calls_without_an_id_or_with_a_repeated_one_get_an_id_of_their_own(make_cloud):
    calls = [call("get_paper", "x", arxiv_id=f"2609.0000{i}") for i in range(6)]
    data = without_ids(response(*calls), [None, "dup", "dup", "", "call_5", ""])
    client, _ = make_cloud([data])
    got = CloudModel(client).next_turn(MESSAGES, MODEL_TOOLS).tool_calls
    # real ids stay; the others become call_<index>, made distinct from a real id that already has the name
    assert [c.id for c in got] == ["call_0", "dup", "call_2", "call_3", "call_5", "call_5_"]
    assert [c.arguments for c in got] == [{"arxiv_id": f"2609.0000{i}"} for i in range(6)]


def test_a_repair_request_answers_calls_without_ids_one_by_one(make_cloud):
    valid = call("get_paper", "x", arxiv_id="2609.01234")
    garbled = call("get_paper", "x", raw='{"arxiv_id": ')
    client, fake = make_cloud([without_ids(response(valid, garbled), [None, None]),
                               response(content="讲自适应草稿长度。")])
    tools, _ = table_tools({"get_paper": [ToolOutcome(PAPER)]})
    t = run_turn("2609.01234 讲什么?", model=CloudModel(client), today=TODAY,
                 executor=ToolExecutor(tools, table_check(), today=TODAY),
                 decide=table_decide({("bad_json", 1): "repair"}))
    assert t.status == ANSWERED
    side = fake.payloads[1]["messages"][2:]  # the repair request: the rejected step and one reply per call
    assert [c["id"] for c in side[0]["tool_calls"]] == ["call_0", "call_1"]
    assert [(m["tool_call_id"], json.loads(m["content"])) for m in side[1:]] == [
        ("call_0", {"error": "SKIPPED", "hint": REFUSED_STEP_HINT}),
        ("call_1", {"error": "INVALID_JSON", "details": [NOT_JSON]})]  # its own error, not merged with call_0
    assert t.steps[0]["attempts"][0]["issues"] == [{"call_id": "call_1", "event": "bad_json",
                                                    "messages": [NOT_JSON]}]


def test_budget_exceeded_propagates_before_any_request(make_cloud):
    client, fake = make_cloud([response(content="never sent")], cap=0.0)
    with pytest.raises(BudgetExceeded):
        CloudModel(client).next_turn(MESSAGES, MODEL_TOOLS)
    assert fake.payloads == []


# LocalModel


def test_local_turn_renders_hf_messages_and_reads_calls_with_the_injected_parser():
    parsed = [{"name": "get_paper", "arguments": {"arxiv_id": "2609.01234"}},
              {"name": "search_arxiv", "arguments": {"query": "推测解码", "max_results": 3}}]
    client, render = FakeLocalClient([TEXT]), recording_render()
    model = LocalModel(client, render, table_parse({TEXT: parsed}), lora_scale=1.0)
    got = model.next_turn(SAMPLE_MESSAGES, MODEL_TOOLS)
    assert render.seen == [(api_to_hf(SAMPLE_MESSAGES), MODEL_TOOLS)]
    assert STOP == ["<|im_end|>"]
    assert client.requests == [{"prompt": "PROMPT 1", "lora_scale": 1.0, "stop": ["<|im_end|>"]}]
    assert got.tool_calls == [
        ToolCall("local_0", "get_paper", '{"arxiv_id":"2609.01234"}', {"arxiv_id": "2609.01234"}),
        ToolCall("local_1", "search_arxiv", '{"max_results":3,"query":"推测解码"}',
                 {"max_results": 3, "query": "推测解码"})]
    assert (got.content, got.paid, got.cached, got.usage) == ("", False, False, LOCAL_USAGE)
    assert got.model == model.name == "local:lora=1" and model.cost(got) == 0.0


def test_text_without_calls_is_the_answer_and_ids_restart_every_step():
    answer = "找到两篇相关论文。"
    parse = table_parse({TEXT: [{"name": "get_paper", "arguments": {"arxiv_id": "2609.01234"}}]})
    model = LocalModel(FakeLocalClient([TEXT, answer, TEXT]), recording_render(), parse, lora_scale=1.0)
    first, second, third = [model.next_turn(MESSAGES, MODEL_TOOLS) for _ in range(3)]
    assert [c.id for c in first.tool_calls] == [c.id for c in third.tool_calls] == ["local_0"]
    assert (second.content, second.tool_calls) == (answer, [])


def test_arguments_that_are_not_a_json_object_reach_the_executor_as_bad_json():
    parsed = [{"name": "get_paper", "arguments": '{"arxiv_id": '},  # text kept as the model wrote it
              {"name": "list_notes"},  # no arguments at all
              {"name": "get_paper", "arguments": ["2609.01234"]}]
    model = LocalModel(FakeLocalClient([TEXT]), recording_render(), table_parse({TEXT: parsed}),
                       lora_scale=1.0)
    calls = model.next_turn(MESSAGES, MODEL_TOOLS).tool_calls
    assert [(c.arguments_json, c.arguments) for c in calls] == [
        ('{"arxiv_id": ', None), ("{}", {}), ('["2609.01234"]', None)]
    issues = ToolExecutor({}, table_check(), today=TODAY).check_step(calls)
    assert [(i.call_id, i.event) for i in issues] == [("local_0", "bad_json"), ("local_2", "bad_json")]


def test_arguments_given_as_object_text_become_canonical_json():
    parsed = [{"name": "search_arxiv", "arguments": '{"query": "推测解码", "max_results": 3}'}]
    model = LocalModel(FakeLocalClient([TEXT]), recording_render(), table_parse({TEXT: parsed}),
                       lora_scale=1.0)
    assert model.next_turn(MESSAGES, MODEL_TOOLS).tool_calls == [
        ToolCall("local_0", "search_arxiv", '{"max_results":3,"query":"推测解码"}',
                 {"max_results": 3, "query": "推测解码"})]


def test_arguments_that_json_cannot_hold_are_a_harness_error():
    parsed = [{"name": "search_arxiv", "arguments": {"query": "x", "categories": {"cs.CL", "cs.LG"}}}]
    model = LocalModel(FakeLocalClient([TEXT]), recording_render(), table_parse({TEXT: parsed}),
                       lora_scale=1.0)
    with pytest.raises(HarnessError, match="JSON cannot hold") as info:
        model.next_turn(MESSAGES, MODEL_TOOLS)
    assert isinstance(info.value.__cause__, TypeError)


def test_a_failing_parser_is_a_harness_error():
    def unfinished(text: str) -> list[dict]:
        raise NotImplementedError

    model = LocalModel(FakeLocalClient([TEXT]), recording_render(), unfinished, lora_scale=1.0)
    with pytest.raises(HarnessError) as info:
        model.next_turn(MESSAGES, [])
    assert isinstance(info.value.__cause__, NotImplementedError)


@pytest.mark.parametrize("returned", [None, {"name": "get_paper"}, ["get_paper"]])
def test_a_parser_that_returns_no_list_of_dicts_is_a_harness_error(returned):
    model = LocalModel(FakeLocalClient([TEXT]), recording_render(), lambda text: returned, lora_scale=1.0)
    with pytest.raises(HarnessError, match="list of dicts"):
        model.next_turn(MESSAGES, [])


def test_the_base_model_scale_is_sent_and_names_tell_the_variants_apart():
    client = FakeLocalClient(["你好"])
    base = LocalModel(client, recording_render(), table_parse({}), lora_scale=0.0)
    base.next_turn(MESSAGES, [])
    assert client.requests[0]["lora_scale"] == 0.0 and base.name == "local:lora=0"
    named = LocalModel(client, recording_render(), table_parse({}), lora_scale=1.0, name="qwen3-1.7b+lora")
    assert named.name == "qwen3-1.7b+lora"


# adapters in a turn


def test_a_rerun_of_a_turn_is_free_and_gives_the_same_record(make_cloud):
    search = call("search_arxiv", "call_0", query="speculative decoding")
    paper = call("get_paper", "call_1", arxiv_id="2609.01234")
    client, fake = make_cloud([response(search), response(paper), response(content="讲自适应草稿长度。")])
    model = CloudModel(client)
    tools, _ = table_tools({"search_arxiv": [ToolOutcome(HITS)], "get_paper": [ToolOutcome(PAPER)]})
    executor = ToolExecutor(tools, table_check(), today=TODAY)
    first, second = [run_turn("找推测解码的论文", model=model, executor=executor, decide=table_decide(),
                              today=TODAY, cost_of=model.cost) for _ in range(2)]
    assert first.status == second.status == ANSWERED and len(fake.payloads) == 3
    assert (first.paid_calls, first.cache_hits, second.paid_calls, second.cache_hits) == (3, 0, 0, 3)
    assert first.cost_rmb == pytest.approx(client.budget.spent("M1")) and second.cost_rmb == 0.0
    # apart from the accounting, the cached rerun's record is byte for byte the first one
    accounts = {"paid_calls": first.paid_calls, "cache_hits": first.cache_hits, "cost_rmb": first.cost_rmb}
    assert dataclasses.replace(second, **accounts).to_json() == first.to_json()


def test_a_local_step_that_does_not_parse_is_handed_off_to_the_cloud_model(make_cloud):
    garbled = '<tool_call>\n{"name": "get_paper", "arguments": {"arxiv_id": </tool_call>'
    parse = table_parse({garbled: [{"name": "get_paper", "arguments": '{"arxiv_id": '}]})
    local = LocalModel(FakeLocalClient([garbled]), recording_render(), parse, lora_scale=1.0)
    client, fake = make_cloud([response(call("get_paper", "call_0", arxiv_id="2609.01234")),
                               response(content="这篇讲自适应草稿长度。")])
    cloud = CloudModel(client)
    tools, _ = table_tools({"get_paper": [ToolOutcome(PAPER)]})
    t = run_turn("2609.01234 讲什么?", model=local, executor=ToolExecutor(tools, table_check(), today=TODAY),
                 decide=table_decide({("bad_json", 1): "handoff"}), today=TODAY, handoff_model=cloud,
                 cost_of=cloud.cost)
    assert (t.status, t.final_text) == (ANSWERED, "这篇讲自适应草稿长度。")
    assert fake.payloads[0]["messages"] == t.messages[:2]  # the original context: system prompt and request
    assert [s["model"] for s in t.steps] == ["flash", "flash"]
    assert [a["model"] for a in t.steps[0]["attempts"]] == ["local:lora=1", "flash-2026-09"]
    assert (t.paid_calls, t.cache_hits, t.local_calls) == (2, 0, 1)  # the local attempt is no cache hit
    assert t.cost_rmb == pytest.approx(client.budget.spent("M1"))
    assert t.usage == {"prompt_tokens": 1050 + 2 * 1200, "completion_tokens": 25 + 2 * 40}


def test_a_local_repair_request_shows_the_rejected_arguments_as_the_model_wrote_them():
    parse = table_parse({GARBLED: [{"name": "get_paper", "arguments": '{"arxiv_id": '}],
                         TEXT: [{"name": "get_paper", "arguments": {"arxiv_id": "2609.01234"}}]})
    render = recording_render()
    client = FakeLocalClient([GARBLED, TEXT, "这篇讲自适应草稿长度。"])
    local = LocalModel(client, render, parse, lora_scale=1.0)
    tools, _ = table_tools({"get_paper": [ToolOutcome(PAPER)]})
    t = run_turn("2609.01234 讲什么?", model=local, executor=ToolExecutor(tools, table_check(), today=TODAY),
                 decide=table_decide({("bad_json", 1): "repair"}), today=TODAY)
    assert (t.status, t.final_text) == (ANSWERED, "这篇讲自适应草稿长度。")
    rejected, error = render.seen[1][0][2:]  # the repair request: system prompt, request, then these two
    assert rejected["tool_calls"] == [{"type": "function",
                                       "function": {"name": "get_paper", "arguments": '{"arxiv_id": '}}]
    assert json.loads(error["content"]) == {"error": "INVALID_JSON", "details": [NOT_JSON]}
    assert (t.paid_calls, t.cache_hits, t.local_calls) == (0, 0, 3)  # three completions, none cached


# ---- M7: LocalModel.for_family, the family path (docs/m7-家族层设计.md §7) --------------------------------

SAMPLING = Decoding(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
GREEDY_NO_THINK = GenerationSettings(mode=Mode.NO_THINK, decoding=GREEDY)
ADAPTER = {"id": 0, "path": "ft.gguf", "scale": 1.0}


def family_model(tmp_path, script: list[Scripted], settings: GenerationSettings, *, lora_scale: float = 1.0):
    """LocalModel.for_family over a fake hybrid family and a scripted server; (model, family, server)."""
    family = fake_loaded_family(tmp_path / "model", "hybrid")
    fake = FakeLlama(script, adapters=[ADAPTER] if lora_scale else [])
    model = LocalModel.for_family(fake.client(), family, settings, lora_scale=lora_scale,
                                  name=f"fake:lora={lora_scale:g}")
    return model, family, fake


def test_a_family_turn_sends_the_prompt_ids_of_the_familys_render_and_reads_the_calls_with_its_format(
        tmp_path):
    model, family, fake = family_model(tmp_path, [scripted(TEXT)], GREEDY_NO_THINK)
    got = model.next_turn(SAMPLE_MESSAGES, MODEL_TOOLS)
    (request,) = fake.requests
    hf = family.to_template_messages(SAMPLE_MESSAGES)
    assert request["prompt"] == family.encode_prompt(family.prompt(hf, MODEL_TOOLS, Mode.NO_THINK))
    assert request["lora"] == [{"id": 0, "scale": 1.0}] and request["stop"] == ["<|im_end|>"]
    assert got.tool_calls == [ToolCall("local_0", "get_paper", '{"arxiv_id":"2609.01234"}',
                                       {"arxiv_id": "2609.01234"})]
    assert (got.content, got.paid, got.cached, got.model) == ("", False, False, "fake:lora=1")
    assert got.usage == {"prompt_tokens": len(request["prompt"]), "completion_tokens": 1,
                         "total_tokens": len(request["prompt"]) + 1}
    assert got.reasoning is None and got.thinking is None  # a greedy no_think turn records nothing new
    assert got.signals == completion_signals(Completion(text=TEXT, tokens=[TEXT], logprobs=[-0.1],
                                                        token_bytes=[list(TEXT.encode())]),
                                             MODEL_TOOLS, SAMPLE_MESSAGES)
    assert isinstance(model, LocalModel) and model.lora_scale == 1.0 and model.family is family
    assert model.settings is GREEDY_NO_THINK and model.cost(got) == 0.0


def test_a_family_turn_in_think_mode_carries_its_reasoning_and_thinking_outside_the_api_form(tmp_path):
    settings = GenerationSettings(mode=Mode.THINK, decoding=SAMPLING, reasoning_budget=16, base_seed=2)
    model, family, fake = family_model(tmp_path, [Scripted(("Need", " the paper.", "\n</think>\n\n", TEXT))],
                                       settings)
    got = model.next_turn(SAMPLE_MESSAGES, MODEL_TOOLS)
    ids = fake.requests[0]["prompt"]
    hf = family.to_template_messages(SAMPLE_MESSAGES)
    assert ids == family.encode_prompt(family.prompt(hf, MODEL_TOOLS, Mode.THINK))
    assert fake.requests[0]["max_tokens"] == 16 and fake.requests[0]["seed"] == request_seed(2, ids, 1, 0)
    assert got.reasoning == "Need the paper." and got.content == ""
    assert [c.name for c in got.tool_calls] == ["get_paper"]
    assert got.thinking == {"mode": "think", "reasoning_tokens": 3, "answer_tokens": 1, "forced": False,
                            "phases": 1, "seeds": [request_seed(2, ids, 1, 0)]}
    assert "reasoning" not in json.dumps(got.to_api()) and "thinking" not in json.dumps(got.to_api())
    bare = AssistantTurn(content=got.content, tool_calls=got.tool_calls, model=got.model, usage=got.usage)
    assert got == bare


def test_a_text_answer_after_the_reasoning_is_the_answer_alone(tmp_path):
    model, _, _ = family_model(tmp_path, [Scripted(("Think", "\n</think>\n\n", "找到", "两篇。"))],
                               GenerationSettings(mode=Mode.THINK, decoding=GREEDY), lora_scale=0.0)
    got = model.next_turn(MESSAGES, MODEL_TOOLS)
    assert (got.content, got.tool_calls, got.reasoning) == ("找到两篇。", [], "Think")
    assert got.thinking["answer_tokens"] == 2 and got.thinking["seeds"] is None  # greedy: no seed was sent


def test_the_signals_of_a_family_turn_are_over_the_answer_tokens_alone(tmp_path):
    doubtful = Scripted(("Unsure", " thought", "\n</think>\n\n", TEXT), logprobs=(-5.0, -6.0, -0.01, -0.02))
    model, _, _ = family_model(tmp_path, [doubtful], GenerationSettings(mode=Mode.THINK, decoding=SAMPLING))
    got = model.next_turn(SAMPLE_MESSAGES, MODEL_TOOLS)
    assert got.signals[MIN_DECISION_PROB] == pytest.approx(math.exp(-0.02))
    assert got.signals[MEAN_LOGPROB] == pytest.approx(-0.02) and got.signals["hard_reason"] is None


def test_a_forced_close_is_recorded_and_a_sampled_no_think_turn_records_its_seed(tmp_path):
    model, _, fake = family_model(tmp_path, [Scripted(("endless",), "length"), scripted("好的。")],
                                  GenerationSettings(mode=Mode.THINK, decoding=SAMPLING, reasoning_budget=1))
    got = model.next_turn(MESSAGES, MODEL_TOOLS)
    assert got.content == "好的。" and got.reasoning == "endless"
    assert got.thinking == {"mode": "think", "reasoning_tokens": 1, "answer_tokens": 1, "forced": True,
                            "phases": 2, "seeds": [r["seed"] for r in fake.requests]}
    sampled = GenerationSettings(mode=Mode.NO_THINK, decoding=SAMPLING, base_seed=5)
    model, _, fake = family_model(tmp_path / "again", [scripted("好的。")], sampled)
    got = model.next_turn(MESSAGES, MODEL_TOOLS)
    assert got.reasoning is None and got.thinking == {
        "mode": "no_think", "reasoning_tokens": 0, "answer_tokens": 1, "forced": False, "phases": 1,
        "seeds": [fake.requests[0]["seed"]]}


ZHONG = "中".encode()  # three bytes: the fake server lists a character split across two tokens once


def step_length(family, fake, got) -> int:
    """A step's sequence as scripts/run_e2e.py's StepLengths sums it from the turn: the prompt's ids, the
    reasoning tokens, a forced close's ids and the answer tokens."""
    close = len(family.encode_close()) if got.thinking["forced"] else 0
    return len(fake.requests[0]["prompt"]) + got.thinking["reasoning_tokens"] + close + got.thinking[
        "answer_tokens"]


def test_the_thinking_record_counts_every_answer_token_generated_the_ones_the_server_did_not_list_too(
        tmp_path):
    # b11349 leaves out of its logprob list a token after which the text ends inside a UTF-8 character
    # (llm.generation, Token ids): an answer holding a character split across tokens lists fewer tokens than
    # the model generated. The record counts the generated ones (Generation.answer_tokens), so a step's
    # length, which scripts/run_e2e.py sums from the record, is the last request's prompt and every token it
    # generated
    settings = GenerationSettings(mode=Mode.THINK, decoding=SAMPLING, reasoning_budget=16)
    closed = Scripted(("Need it.", "\n</think>\n\n", ZHONG[:2], ZHONG[2:], "文", ""), "stop",
                      ids=(*range(ID_BASE, ID_BASE + 5), TURN_END_ID))
    model, family, fake = family_model(tmp_path, [closed], settings)
    got = model.next_turn(MESSAGES, MODEL_TOOLS)
    assert (got.content, got.reasoning) == ("中文", "Need it.")
    assert got.thinking["reasoning_tokens"] == 2 and got.thinking["answer_tokens"] == 4  # 3 of them listed
    assert step_length(family, fake, got) == len(fake.requests[-1]["prompt"]) + got.usage["completion_tokens"]
    # closed by force: the answer is every token the second request generated, as its usage counts them
    forced = GenerationSettings(mode=Mode.THINK, decoding=SAMPLING, reasoning_budget=1)
    model, family, fake = family_model(tmp_path / "forced", [Scripted(("endless",), "length"),
                                                             Scripted((ZHONG[:2], ZHONG[2:], "。"))], forced)
    got = model.next_turn(MESSAGES, MODEL_TOOLS)
    assert got.content == "中。" and got.thinking["forced"] and got.thinking["answer_tokens"] == 3
    assert step_length(family, fake, got) == len(fake.requests[-1]["prompt"]) + 3
    # an answer the budget cut, continued: the first request's tokens after the reasoning and the second's
    cut = GenerationSettings(mode=Mode.THINK, decoding=SAMPLING, reasoning_budget=3)
    model, family, fake = family_model(tmp_path / "continued",
                                       [Scripted(("T", "\n</think>\n\n", "A"), "length"),
                                        Scripted((ZHONG[:2], ZHONG[2:]))], cut)
    got = model.next_turn(MESSAGES, MODEL_TOOLS)
    assert got.content == "A中" and got.thinking["phases"] == 2 and got.thinking["answer_tokens"] == 3
    assert step_length(family, fake, got) == len(fake.requests[-1]["prompt"]) + 2
    # a sampled NO_THINK turn: its one request's count
    sampled = GenerationSettings(mode=Mode.NO_THINK, decoding=SAMPLING)
    split = Scripted((ZHONG[:2], ZHONG[2:], "。"))
    model, family, fake = family_model(tmp_path / "no_think", [split], sampled)
    got = model.next_turn(MESSAGES, MODEL_TOOLS)
    assert got.content == "中。" and got.thinking["answer_tokens"] == 3 == got.usage["completion_tokens"]
    assert step_length(family, fake, got) == len(fake.requests[-1]["prompt"]) + 3


def test_the_family_path_reads_text_arguments_and_broken_blocks_as_the_legacy_path_does(tmp_path):
    broken = '<tool_call>\n{"name": "get_paper", "arguments": {"arxiv_id": </tool_call>'
    model, _, _ = family_model(tmp_path, [scripted(broken)], GREEDY_NO_THINK)
    (bad,) = model.next_turn(MESSAGES, MODEL_TOOLS).tool_calls
    assert bad.name == "get_paper" and bad.arguments is None  # the bad_json path, as before
    assert bad.arguments_json == '{"name": "get_paper", "arguments": {"arxiv_id":'
    assert ToolExecutor({}, table_check(), today=TODAY).check_step([bad])[0].event == "bad_json"
    assert ID_BASE == 50_000  # the fake's ids are the server's, never the family tokenizer's


# ---- M7: which text the family's format reads, and that the signals follow its locate --------------------

GREEDY_THINK = GenerationSettings(mode=Mode.THINK, decoding=GREEDY)
TEXT_BODY = TEXT[len("<tool_call>"):]  # the call's block after its opening tag


def format_model(tmp_path, script: list[Scripted], settings: GenerationSettings,
                 fmt: FakeToolFormat) -> tuple[LocalModel, FakeLlama]:
    """LocalModel.for_family over a fake hybrid family whose tool format is fmt (support/llama_fakes.py), the
    texts load_family's probe handed the format forgotten; (model, server)."""
    family = fake_loaded_family(tmp_path / "model", "hybrid", tool_format=fmt)
    fmt.parsed.clear()
    fmt.located.clear()
    fake = FakeLlama(script)
    return LocalModel.for_family(fake.client(), family, settings, lora_scale=0.0), fake


def test_the_text_after_the_close_is_parsed_and_reported_while_the_signals_read_aligned_text(tmp_path):
    # the first answer token straddles the close's newlines: the turn's calls and content come from the
    # format's parse of the text after the close, the signals (its locate, for the decision tokens and for
    # the hard trigger, and the hard trigger's own parse) from the completion's text, which starts where
    # that token does
    fmt = FakeToolFormat()
    model, _ = format_model(tmp_path, [Scripted(("Think", "\n</think>", "\n\n好的。"))], GREEDY_THINK, fmt)
    got = model.next_turn(MESSAGES, MODEL_TOOLS)
    assert (got.content, got.tool_calls, got.reasoning) == ("好的。", [], "Think")
    assert fmt.parsed == ["好的。", "\n\n好的。"] and fmt.located == ["\n\n好的。"] * 2
    assert got.thinking["answer_tokens"] == 1
    fmt = FakeToolFormat()
    straddling = Scripted(("Need it.", "\n</think>", "\n\n<tool_call>", TEXT_BODY))
    model, _ = format_model(tmp_path / "call", [straddling], GREEDY_THINK, fmt)
    got = model.next_turn(MESSAGES, MODEL_TOOLS)
    assert [c.name for c in got.tool_calls] == ["get_paper"] and got.content == ""
    assert fmt.parsed == [TEXT, "\n\n" + TEXT] and fmt.located == ["\n\n" + TEXT] * 2


def test_a_forced_answer_is_the_second_completion_as_it_came_for_parser_content_and_signals(tmp_path):
    forced = GenerationSettings(mode=Mode.THINK, decoding=GREEDY, reasoning_budget=1)
    fmt = FakeToolFormat()
    model, _ = format_model(tmp_path, [Scripted(("Cut",), "length"), Scripted(("好的", "。"))], forced, fmt)
    got = model.next_turn(MESSAGES, MODEL_TOOLS)
    assert (got.content, got.tool_calls, got.reasoning) == ("好的。", [], "Cut")
    assert got.thinking["forced"] and got.thinking["answer_tokens"] == 2
    assert fmt.parsed == ["好的。"] * 2 and fmt.located == ["好的。"] * 2  # the turn's, the signals'
    fmt = FakeToolFormat()
    model, _ = format_model(tmp_path / "call", [Scripted(("Cut",), "length"), scripted(TEXT)], forced, fmt)
    got = model.next_turn(MESSAGES, MODEL_TOOLS)
    assert [c.name for c in got.tool_calls] == ["get_paper"] and got.content == ""
    assert fmt.parsed == [TEXT] * 2 and fmt.located == [TEXT] * 2


def test_the_signals_of_a_family_turn_read_the_calls_through_the_familys_locate(tmp_path):
    unread = FakeToolFormat(locate=lambda text: [CallSpan(parsed=False)])
    model, _ = format_model(tmp_path, [scripted(TEXT)], GREEDY_NO_THINK, unread)
    got = model.next_turn(MESSAGES, MODEL_TOOLS)
    assert [c.name for c in got.tool_calls] == ["get_paper"]  # the format's parse reads the call...
    assert got.signals["hard_reason"].startswith("parse_failure")  # ...its locate does not: it is followed
    head = FakeToolFormat(locate=lambda text: [CallSpan(parsed=True, name=(0, 1), values=())])
    doubtful = Scripted(("<tool_call>\n", TEXT_BODY[1:]), logprobs=(-0.1, -3.0))
    model, _ = format_model(tmp_path / "head", [doubtful], GREEDY_NO_THINK, head)
    got = model.next_turn(MESSAGES, MODEL_TOOLS)
    assert got.signals["hard_reason"] is None
    # the format's locate names the first token alone; Qwen3's reader would take the name's token, exp(-3.0)
    assert got.signals[MIN_DECISION_PROB] == pytest.approx(math.exp(-0.1))

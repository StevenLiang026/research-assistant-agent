import copy
import json

import pytest
from support.fakes import SAMPLE_MESSAGES, ScriptedModel, call, turn

from research_agent.agent.context import build_context
from research_agent.agent.models import LocalModel
from research_agent.agent.types import BUDGET_STOP, CONTEXT_OVERFLOW, HarnessError
from research_agent.canonical import canonical_json
from research_agent.eval.lock import LockedSet, RefusedTestSet, write_jsonl, write_lock
from research_agent.eval.runner import (
    ERROR,
    OK,
    RESULT_STATUSES,
    SKIPPED,
    read_results,
    run_items,
    write_results,
)
from research_agent.eval.systems import ReplayModel
from research_agent.llm.budget import BudgetExceeded
from research_agent.llm.local import Completion
from research_agent.tools.schemas import CONTEXT_TOKENS, MODEL_TOOLS, REPLY_RESERVE_TOKENS

ITEMS = [
    {"id": "t001", "messages": SAMPLE_MESSAGES[:2], "expected": {"tool": "search_arxiv"}, "lang": "zh"},
    {"id": "t002", "messages": SAMPLE_MESSAGES, "expected": {"tool": "fetch_fulltext"}, "lang": "zh"},
    {"id": "t003", "messages": [{"role": "user", "content": "Save a note."}],
     "expected": {"tool": "ask_clarification"}, "lang": "en"},
]
SEARCH = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-01")


def answers(*contents: str) -> ScriptedModel:
    return ScriptedModel([turn(content=c) for c in contents])


def test_test_material_runs_only_as_the_items_lockedset_open_returned(tmp_path):
    test_items = [{**item, "id": f"req-test-000{n}-s1", "split": "test"} for n, item in enumerate(ITEMS, 1)]
    model = answers("a", "b", "c")
    with pytest.raises(RefusedTestSet, match="only the phase-4 test runner"):
        run_items(test_items, model)
    assert model.contexts == []  # refused before any model call
    path = tmp_path / "test.jsonl"
    write_jsonl(test_items, path)
    write_lock(path)
    opened = LockedSet(path, tmp_path / "test-access.jsonl").open("M3", "the phase-4 run")
    assert [r["status"] for r in run_items(opened, model)] == [OK, OK, OK]


def test_ok_records_the_turn_as_it_came_back():
    first = turn(SEARCH, usage={"prompt_tokens": 1100, "completion_tokens": 30})
    second = turn(content="读完了。", paid=False, cached=True, model="flash-2026-09",
                  usage={"prompt_tokens": 1500, "completion_tokens": 12})
    model = ScriptedModel([first, second])
    results = run_items(ITEMS[:2], model)
    assert results == [
        {"id": "t001", "status": OK, "item": {"expected": {"tool": "search_arxiv"}, "lang": "zh"},
         "output": {"role": "assistant", "content": "", "tool_calls": [SEARCH.to_api()]},
         "usage": {"prompt_tokens": 1100, "completion_tokens": 30}, "paid": True, "cached": False,
         "model": "fake"},
        {"id": "t002", "status": OK, "item": {"expected": {"tool": "fetch_fulltext"}, "lang": "zh"},
         "output": {"role": "assistant", "content": "读完了。"},
         "usage": {"prompt_tokens": 1500, "completion_tokens": 12}, "paid": False, "cached": True,
         "model": "flash-2026-09"},
    ]
    # The model sees what the loop would show it: older tool outputs compressed, the latest step whole.
    assert model.contexts == [build_context(SAMPLE_MESSAGES[:2]), build_context(SAMPLE_MESSAGES)]
    assert model.contexts[1][3]["content"] != SAMPLE_MESSAGES[3]["content"]
    assert model.contexts[1][5:] == SAMPLE_MESSAGES[5:]


def test_context_overflow_is_recorded_without_a_model_call():
    seen = []

    def count_tokens(messages: list[dict], tools: list[dict]) -> int:
        seen.append((len(messages), tools))
        return CONTEXT_TOKENS - REPLY_RESERVE_TOKENS + (1 if len(messages) > 2 else 0)  # t002 is one over

    model = answers("ok", "never asked")
    results = run_items(ITEMS[:2], model, count_tokens=count_tokens)
    assert [r["status"] for r in results] == [OK, CONTEXT_OVERFLOW]
    assert results[1] == {"id": "t002", "status": CONTEXT_OVERFLOW,
                          "item": {"expected": {"tool": "fetch_fulltext"}, "lang": "zh"}}
    assert len(model.contexts) == 1 and len(model.turns) == 1
    assert seen == [(2, MODEL_TOOLS), (7, MODEL_TOOLS)]  # measured with the tools the model is given


def test_a_failing_item_does_not_stop_the_run():
    model = ScriptedModel([HarnessError("parse_tool_calls failed: ValueError('x')"), turn(content="ok"),
                           KeyError("choices")])
    results = run_items(ITEMS, model)
    assert [r["status"] for r in results] == [ERROR, OK, ERROR]
    assert results[0] == {"id": "t001", "status": ERROR,
                          "item": {"expected": {"tool": "search_arxiv"}, "lang": "zh"},
                          "error": "parse_tool_calls failed: ValueError('x')", "error_type": "HarnessError"}
    assert (results[2]["error"], results[2]["error_type"]) == ("'choices'", "KeyError")
    assert len(model.contexts) == 3


def test_budget_stop_marks_the_item_and_skips_the_rest():
    model = ScriptedModel([turn(content="ok"), BudgetExceeded("stage M3: spent 30.0000 of 30.0 RMB"),
                           turn(content="never asked")])
    results = run_items(ITEMS, model)
    assert [(r["id"], r["status"]) for r in results] == [("t001", OK), ("t002", BUDGET_STOP),
                                                         ("t003", SKIPPED)]
    assert results[1] == {"id": "t002", "status": BUDGET_STOP,
                          "item": {"expected": {"tool": "fetch_fulltext"}, "lang": "zh"}}
    assert results[2] == {"id": "t003", "status": SKIPPED,
                          "item": {"expected": {"tool": "ask_clarification"}, "lang": "en"}}
    assert len(model.contexts) == 2 and len(model.turns) == 1  # nothing asked after the stop
    assert {r["status"] for r in results} <= set(RESULT_STATUSES)


def test_results_file_round_trips_and_two_runs_are_byte_identical(tmp_path):
    def run(path):
        model = ScriptedModel([turn(SEARCH), HarnessError("replay miss"), turn(content="请给出论文编号")])
        results = run_items(ITEMS, model)
        write_results(results, path)
        return results

    paths = [tmp_path / "a" / "results.jsonl", tmp_path / "b" / "results.jsonl"]
    first, second = run(paths[0]), run(paths[1])
    assert first == second and paths[0].read_bytes() == paths[1].read_bytes()
    assert read_results(paths[0]) == first
    raw = paths[0].read_bytes()
    lines = raw.split(b"\n")
    assert lines[-1] == b"" and b"\r" not in raw
    assert [json.loads(line) for line in lines[:-1]] == first
    assert lines[0] == canonical_json(first[0]).encode("utf-8")  # sorted keys, no spaces, UTF-8 as is
    assert "请给出论文编号".encode() in raw


def test_items_are_not_modified_and_records_do_not_share_them():
    items = copy.deepcopy(ITEMS)
    results = run_items(items, answers("a", "b", "c"))
    assert items == ITEMS
    results[0]["item"]["expected"]["tool"] = "changed"
    assert items[0]["expected"]["tool"] == "search_arxiv"


def test_ids_are_checked_before_any_model_call():
    model = answers("a")
    with pytest.raises(ValueError, match="distinct; repeated: \\['t001'\\]"):
        run_items([ITEMS[0], {**ITEMS[1], "id": "t001"}], model)
    with pytest.raises(ValueError, match="non-empty string id"):
        run_items([{**ITEMS[0], "id": 7}], model)
    with pytest.raises(ValueError, match="non-empty string id"):
        run_items([{"messages": ITEMS[0]["messages"]}], model)
    assert model.contexts == []


def test_a_history_that_does_not_end_with_the_user_or_a_tool_is_an_error():
    model = answers("ok")
    ends_with_call = {"id": "t009", "messages": SAMPLE_MESSAGES[:3]}
    results = run_items([ends_with_call, {"id": "t010", "messages": []}, ITEMS[0]], model)
    assert [r["status"] for r in results] == [ERROR, ERROR, OK]
    assert results[0]["error"] == "the history must end with a user or tool message, not 'assistant'"
    assert results[1]["error"] == "the item has no messages"
    assert len(model.contexts) == 1


def test_no_items_give_no_records():
    assert run_items([], answers()) == []


def test_on_item_hears_every_record_as_it_is_made_the_budget_stop_and_the_skipped_included():
    events: list[str] = []

    class Logged(ScriptedModel):
        def next_turn(self, messages: list[dict], tools: list[dict]):
            events.append("model")
            return super().next_turn(messages, tools)

    model = Logged([turn(content="ok"), BudgetExceeded("stage M3: spent 30.0000 of 30.0 RMB")])
    heard: list[dict] = []

    def on_item(record: dict) -> None:
        heard.append(record)
        events.append(record["id"])

    results = run_items(ITEMS, model, on_item=on_item)
    assert heard == results and [r["status"] for r in heard] == [OK, BUDGET_STOP, SKIPPED]
    assert events == ["model", "t001", "model", "t002", "t003"]  # each record before the next item runs


def test_a_cloud_turns_mode_is_recorded_and_a_turn_without_one_records_none():
    model = ScriptedModel([turn(SEARCH, mode="thinking"), turn(content="读完了。", mode="non-thinking"),
                           turn(content="请给出论文编号")])
    results = run_items(ITEMS, model)
    assert [r["status"] for r in results] == [OK, OK, OK]
    assert [r.get("mode", "absent") for r in results] == ["thinking", "non-thinking", "absent"]


# The bytes the runner wrote at c166518, before turns had a mode: local and replay records stay as they were.
LOCAL_RECORD = (
    b'{"cached":false,"id":"t001","item":{"expected":{"tool":"search_arxiv"},"lang":"zh"},'
    b'"model":"qwen3-1.7b:lora=1","output":{"content":"","role":"assistant","tool_calls":[{"function":'
    b'{"arguments":"{\\"max_results\\":3,\\"query\\":\\"speculative decoding\\"}","name":"search_arxiv"},'
    b'"id":"local_0","type":"function"}]},"paid":false,"status":"ok",'
    b'"usage":{"completion_tokens":25,"prompt_tokens":1050}}\n')
REPLAY_RECORD = (
    b'{"cached":false,"id":"r001","item":{"expected":{"content":"","tool_calls":[{"arguments":'
    b'{"query":"speculative decoding"},"name":"search_arxiv"}]}},"model":"replay","output":{"content":"",'
    b'"role":"assistant","tool_calls":[{"function":{"arguments":"{\\"query\\":\\"speculative decoding\\"}",'
    b'"name":"search_arxiv"},"id":"replay_0","type":"function"}]},"paid":false,"status":"ok","usage":{}}\n')


class OneCompletion:
    """LocalClient stand-in: one fixed completion for every prompt."""

    def complete(self, prompt: str, *, lora_scale: float | None = None,
                 stop: list[str] | None = None) -> Completion:
        return Completion(text="<tool_call>", raw={"usage": {"prompt_tokens": 1050, "completion_tokens": 25}})


def test_local_and_replay_records_are_byte_for_byte_those_written_before_turns_had_a_mode(tmp_path):
    parsed = [{"name": "search_arxiv", "arguments": {"query": "speculative decoding", "max_results": 3}}]
    local = LocalModel(OneCompletion(), lambda messages, tools: "PROMPT", lambda text: parsed, lora_scale=1.0,
                       name="qwen3-1.7b:lora=1")
    write_results(run_items([ITEMS[0]], local), tmp_path / "local.jsonl")
    expected_call = {"name": "search_arxiv", "arguments": {"query": "speculative decoding"}}
    replay_item = {"id": "r001", "messages": SAMPLE_MESSAGES[:2],
                   "expected": {"content": "", "tool_calls": [expected_call]}}
    write_results(run_items([replay_item], ReplayModel([replay_item])), tmp_path / "replay.jsonl")
    assert (tmp_path / "local.jsonl").read_bytes() == LOCAL_RECORD
    assert (tmp_path / "replay.jsonl").read_bytes() == REPLAY_RECORD


# ---- M4: signals, routes and the begin-task hook ---------------------------------------------------------


class SignalledModel:
    """A model whose turns carry signals and, when given, a route; begin_task calls are counted."""

    name = "signalled"

    def __init__(self, route: dict | None = None) -> None:
        self.route, self.begun = route, 0

    def begin_task(self) -> None:
        self.begun += 1

    def next_turn(self, messages: list[dict], tools: list[dict]):
        answer = turn(content="好的", paid=False, model=self.name)
        answer.signals = {"min_decision_prob": 0.5, "mean_logprob": -0.25, "hard_reason": None}
        answer.route = self.route
        return answer


def test_signals_enter_the_records_only_when_asked_for():
    plain = run_items(ITEMS[:2], SignalledModel())
    assert all("signals" not in r for r in plain)
    asked = run_items(ITEMS[:2], SignalledModel(), signals=True)
    assert [r["signals"] for r in asked] == [{"min_decision_prob": 0.5, "mean_logprob": -0.25,
                                             "hard_reason": None}] * 2
    errors = run_items([{"id": "x", "messages": []}], SignalledModel(), signals=True)
    assert errors[0]["status"] == ERROR and "signals" not in errors[0]
    cloud = run_items(ITEMS[:1], ScriptedModel([turn(content="hi")]), signals=True)
    assert "signals" not in cloud[0]  # a turn without signals records none


def test_a_routed_turn_records_its_route_and_begin_task_runs_before_every_item():
    route = {"answered_by": "local", "min_decision_prob": 0.5, "mean_logprob": -0.25, "hard_reason": None,
             "reason": None}
    model = SignalledModel(route)
    records = run_items(ITEMS, model)
    assert [r["route"] for r in records] == [route] * 3 and model.begun == 3
    assert all("route" not in r for r in run_items(ITEMS[:1], ScriptedModel([turn(content="hi")])))


# ---- M7: the reasoning and the thinking of a local turn (docs/m7-家族层设计.md §7) ------------------------


def test_reasoning_and_thinking_enter_a_record_only_when_the_turn_has_them():
    thinking = {"mode": "think", "reasoning_tokens": 3, "answer_tokens": 2, "forced": True, "phases": 2,
                "seeds": [1, 2]}
    thought, empty, plain = turn(content="好的"), turn(content="好的"), turn(content="好的")
    thought.reasoning, thought.thinking = "because", thinking
    empty.reasoning, empty.thinking = "", {**thinking, "reasoning_tokens": 0}  # closed at once: still its own
    records = run_items(ITEMS, ScriptedModel([thought, empty, plain]))
    assert records[0]["reasoning"] == "because" and records[0]["thinking"] == thinking
    assert records[1]["reasoning"] == "" and records[1]["thinking"]["reasoning_tokens"] == 0
    assert "reasoning" not in records[2] and "thinking" not in records[2]  # as they were
    assert {k for k in records[0] if k not in records[2]} == {"reasoning", "thinking"}

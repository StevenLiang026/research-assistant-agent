"""research_agent.eval.timing: the wall times of the runners' model calls, items and task steps, which go to a
sidecar file so that the records themselves stay deterministic."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from support.fakes import ScriptedModel, turn

from research_agent.agent.types import HarnessError
from research_agent.eval.runner import run_items
from research_agent.eval.timing import ItemTimer, TaskTimer, TimedModel

CONTEXT = [{"role": "user", "content": "hi"}]


def ticking(step: float):
    """A clock that moves step seconds forward every time it is read."""
    now = [0.0]

    def clock() -> float:
        now[0] += step
        return now[0]

    return clock


def test_a_timed_model_answers_as_its_model_and_times_every_call_a_failed_one_included():
    answer = turn(content="a", paid=False, cached=True)
    inner = ScriptedModel([answer, HarnessError("parse_tool_calls failed")], name="qwen3-1.7b:lora=1")
    inner.cost = lambda t: 0.5
    inner.modes = {"thinking": 1}
    reported = iter([{"prompt_ms": 3.0}, None])
    timed = TimedModel(inner, clock=ticking(0.25), server_timings=lambda: next(reported))
    assert timed.next_turn(CONTEXT, []) is answer
    with pytest.raises(HarnessError, match="parse_tool_calls failed"):
        timed.next_turn(CONTEXT, [])
    assert timed.name == "qwen3-1.7b:lora=1" and timed.cost(answer) == 0.5 and timed.modes == {"thinking": 1}
    assert inner.contexts == [CONTEXT, CONTEXT]
    assert timed.calls == [
        {"wall_ms": 250.0, "error": None, "paid": False, "cached": True, "server": {"prompt_ms": 3.0}},
        {"wall_ms": 250.0, "error": "HarnessError", "paid": None, "cached": None, "server": None},
    ]
    plain = TimedModel(ScriptedModel([]))
    assert getattr(plain, "cost", None) is None  # still no cost: run_request prices such a model at 0.0


def test_the_item_timer_gives_each_record_its_wall_time_and_the_calls_made_for_it():
    clock = ticking(1.0)
    timed = TimedModel(ScriptedModel([turn(content="a")]), clock=clock)
    timer = ItemTimer(timed, clock=clock)  # read at 1 s
    items = [{"id": "a", "messages": CONTEXT}, {"id": "b", "messages": []}]  # b: an error without a call
    records = run_items(items, timed, on_item=timer)
    assert [r["status"] for r in records] == ["ok", "error"]
    assert timer.items == [  # the call runs from 2 s to 3 s; the records are made at 4 s and 5 s
        {"id": "a", "status": "ok", "wall_ms": 3000.0,
         "calls": [{"wall_ms": 1000.0, "error": None, "paid": True, "cached": False, "server": None}]},
        {"id": "b", "status": "error", "wall_ms": 1000.0, "calls": []},
    ]


def test_the_task_timer_times_each_task_and_puts_each_call_in_the_step_whose_attempt_it_was():
    clock = ticking(0.5)
    timed = TimedModel(ScriptedModel([turn(content="1"), turn(content="2"), turn(content="3"),
                                      HarnessError("duplicate ids"), turn(content="4")]), clock=clock)
    timer = TaskTimer(timed, clock=clock)  # read at 0.5 s
    for _ in range(3):  # 1.0 to 1.5 s, 2.0 to 2.5 s, 3.0 to 3.5 s
        timed.next_turn(CONTEXT, [])
    two_steps = SimpleNamespace(steps=[{"index": 1, "attempts": [{}, {}]}, {"index": 2, "attempts": [{}]}])
    timer({"id": "t1", "status": "answered"}, two_steps)  # at 4.0 s: the records carry no time of their own
    with pytest.raises(HarnessError):
        timed.next_turn(CONTEXT, [])
    timer({"id": "t2", "status": "harness_error"}, SimpleNamespace(steps=[{"index": 1, "attempts": []}]))
    timed.next_turn(CONTEXT, [])
    timer({"id": "t3", "status": "harness_error"}, None)  # run_request raised: no transcript
    t1, t2, t3 = timer.tasks
    ok = {"wall_ms": 500.0, "error": None, "paid": True, "cached": False, "server": None}
    assert t1 == {"id": "t1", "status": "answered", "wall_ms": 3500.0,
                  "steps": [{"index": 1, "calls": 2, "wall_ms": 1000.0},
                            {"index": 2, "calls": 1, "wall_ms": 500.0}],
                  "calls": [{"step": 1, **ok}, {"step": 1, **ok}, {"step": 2, **ok}]}
    # a call without an attempt (it raised) belongs to the last step, the one that was running
    assert t2["steps"] == [{"index": 1, "calls": 1, "wall_ms": 500.0}] and t2["wall_ms"] == 1500.0
    assert t2["calls"] == [{"step": 1, **ok, "error": "HarnessError", "paid": None, "cached": None}]
    assert t3 == {"id": "t3", "status": "harness_error", "wall_ms": 1500.0, "steps": [],
                  "calls": [{"step": None, **ok}]}

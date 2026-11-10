"""Wall times of an evaluation run, kept beside its records in a timing sidecar, so that the records stay
deterministic: two runs over the same items with the same answers write the same bytes however long each call
took (docs: the M3 plan, phase 3; phase 4 reports latency from these files).

TimedModel wraps the model a runner asks and times every next_turn call with a monotonic clock
(time.perf_counter unless a test injects one), a call that raises included. Per call: the wall time in
milliseconds, the exception's class name or None, whether the turn was paid or served from the cache (None for
a call that raised; cloud latency is reported over paid calls only), and llama.cpp's own timings of the
completion when the model runs on llama-server (server_timings, LocalClient.last_timings: prompt and
generation milliseconds and token counts), else None. Everything else is the wrapped model's (name, cost,
modes...), and a model without a cost stays without one, so the loop and the runners see the same model and
the records and transcripts hold the same turns.

ItemTimer (scripts/run_eval.py, run_items' on_item hook) gives every record its item's wall time, from the
previous record to this one (building and measuring the context included), and the calls made for it.
TaskTimer (scripts/run_e2e.py, run_tasks' on_task hook) gives every end-to-end task its wall time, from the
previous task's record (the first task: from the timer's creation) to this one (seeding the note store, the
run, saving the transcript and scoring included; the report itself holds no time), and its calls, each placed
in the step of the transcript whose attempt it was: the loop records one attempt per call that came back, in
call order, and a call that raised has no attempt and belongs to the step that was running, the last one;
without a transcript (run_request raised) a call has no step. A step's wall time is the sum of its calls': the
model's time, not the replayed tools'.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from ..agent.loop import Model
from ..agent.types import AssistantTurn
from ..canonical import canonical_json

Clock = Callable[[], float]


def _no_timings() -> dict | None:
    return None


class TimedModel:
    """The wrapped model, with every next_turn call timed into calls (module docstring)."""

    def __init__(self, model: Model, *, clock: Clock = time.perf_counter,
                 server_timings: Callable[[], dict | None] = _no_timings) -> None:
        self.model, self.clock, self.server_timings = model, clock, server_timings
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return self.model.name

    def next_turn(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        started = self.clock()
        turn: AssistantTurn | None = None
        error = None
        try:
            turn = self.model.next_turn(messages, tools)
            return turn
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            self.calls.append({"wall_ms": round((self.clock() - started) * 1000, 1), "error": error,
                               "paid": None if turn is None else turn.paid,
                               "cached": None if turn is None else turn.cached,
                               "server": self.server_timings()})

    def __getattr__(self, name: str):
        if name == "model":  # not set yet: no recursion through self.model
            raise AttributeError(name)
        return getattr(self.model, name)


class ItemTimer:
    """run_items' on_item hook: one {"id", "status", "wall_ms", "calls"} per record, in items."""

    def __init__(self, model: TimedModel, *, clock: Clock = time.perf_counter) -> None:
        self.model, self.clock = model, clock
        self.items: list[dict] = []
        self._since, self._seen = clock(), len(model.calls)

    def __call__(self, record: dict) -> None:
        now = self.clock()
        calls = self.model.calls[self._seen:]
        self._seen = len(self.model.calls)
        self.items.append({"id": record["id"], "status": record["status"],
                           "wall_ms": round((now - self._since) * 1000, 1), "calls": calls})
        self._since = now


class TaskTimer:
    """run_tasks' on_task hook: one {"id", "status", "wall_ms", "steps", "calls"} per task record, in tasks;
    steps are {"index", "calls", "wall_ms"} and every call carries its step's index (module docstring)."""

    def __init__(self, model: TimedModel, *, clock: Clock = time.perf_counter) -> None:
        self.model, self.clock = model, clock
        self.tasks: list[dict] = []
        self._since, self._seen = clock(), len(model.calls)

    def __call__(self, record: dict, transcript: object | None) -> None:
        now = self.clock()
        calls = self.model.calls[self._seen:]
        self._seen = len(self.model.calls)
        steps = list(getattr(transcript, "steps", None) or [])
        placed: list[dict] = []
        timed_steps = []
        position = 0
        for number, step in enumerate(steps):
            count = len(step.get("attempts") or [])
            if number == len(steps) - 1:  # a call that raised has no attempt: it was the last step's
                count = max(len(calls) - position, 0)
            mine = calls[position:position + count]
            position += len(mine)
            placed += [{"step": step["index"], **call} for call in mine]
            timed_steps.append({"index": step["index"], "calls": len(mine),
                                "wall_ms": round(sum(call["wall_ms"] for call in mine), 1)})
        placed += [{"step": None, **call} for call in calls[position:]]  # no transcript, no step
        self.tasks.append({"id": record["id"], "status": record["status"],
                           "wall_ms": round((now - self._since) * 1000, 1), "steps": timed_steps,
                           "calls": placed})
        self._since = now


def write_timing(timing: dict, path: str | Path) -> None:
    """The sidecar as one canonical JSON line (sorted keys, UTF-8, LF); parent folders are created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(timing) + "\n", encoding="utf-8", newline="\n")

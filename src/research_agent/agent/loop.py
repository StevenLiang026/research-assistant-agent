"""One turn of the agent: at most 8 steps of model request, step check and tool execution.

Every model request gets build_context(history): the latest step whole, older tool outputs cut to a few
keys. With a token counter, every request is measured before it is sent, repair and handoff requests
included, and one over the budget ends the turn as context_overflow. The executor checks a step as a whole
before any of its calls runs. Each failure point asks the owner's decide(event, times) (run_turn says how
times counts), and the loop carries out each answer in schemas.ALLOWED_ACTIONS[event] as given:

- invalid step (bad_json, unknown_tool, invalid_args; the event of its first issue is decided): "repair"
  asks the same model again with the rejected step and its errors appended to the context; "handoff" asks
  the handoff model with the original context and keeps it for the rest of the turn; "stop" ends the turn as
  invalid_output, and so does a handoff with no handoff model to go to, or any answer once the step has
  used up its model calls. Rejected steps are kept in the step record but never enter the history, which is
  what the training samples are cut from (a trajectory with an invalid call would be refused). An attempt
  of a turn that carries reasoning and a thinking record (a local model's in a thinking mode,
  agent.types.AssistantTurn) keeps both under "reasoning" and "thinking"; the attempts of other turns are
  as they were.
- repeat_call (the same call, by call_key, already ran twice this turn): "stop" ends the turn as dead_loop;
  "tell_user" answers the call SKIPPED with a hint, without running it, and the turn goes on.
- empty_results: "relax" adds a hint to relax the search once; "tell_user" a hint to tell the user.

ask_clarification ends the turn with its question; the other calls of its step are not run. However a turn
ends, every call of an accepted step gets exactly one tool message, in call order, so the history stays
well-formed. BudgetExceeded ends the turn as budget_stop. A HarnessError (a replay miss, a broken adapter, a
failing owner function, a decide answer outside the event's ALLOWED_ACTIONS) ends it as harness_error; its
message goes to the step record, never to the model.

The whole history (M7, docs/m7-本地提升计划.md WP2, WP3). A model is sent the context build_context cut from
the history, but the inference wrappers that judge an answer by its conversation (agent.rules.RuledModel,
agent.vote.VotedModel) read the history itself: while a model answers, current_history() is the history the
request was built from, the loop's accepted history here (the transcript's messages so far, for a repair
request too, whose rejected exchange never enters it) and the item's messages in the single-step runner
(eval.runner). asking(history) sets it for one request; outside a request it is None. An attempt whose turn
carries a vote or the rules' changes (agent.types.AssistantTurn.votes, .rules) keeps them under "votes" and
"rules"; the attempts of other turns are as they were.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

from ..llm.budget import BudgetExceeded
from ..tools.schemas import (
    ALLOWED_ACTIONS,
    EMPTY_RESULTS,
    INVALID_ARGUMENTS,
    INVALID_JSON,
    MODEL_TOOLS,
    PER_STEP_EVENTS,
    SCHEMA_HASH,
    SKIPPED,
    UNKNOWN_TOOL,
)
from .context import TokenCounter, build_context, fits
from .executor import ToolExecutor
from .prompts import PROMPT_HASH, PROMPT_VERSION, system_prompt
from .types import (
    ANSWERED,
    BUDGET_STOP,
    CLARIFICATION,
    CONTEXT_OVERFLOW,
    DEAD_LOOP,
    HARNESS_ERROR,
    INVALID_OUTPUT,
    STEP_LIMIT,
    AssistantTurn,
    Decide,
    HarnessError,
    StepIssue,
    ToolCall,
    ToolResult,
    Transcript,
    call_key,
)

ASK = "ask_clarification"
MAX_RUNS_PER_CALL = 2  # a call that already ran this often in the turn is a repeat_call

# Hints the model reads in tool results (in Chinese, like the system prompt).
RELAX_HINT = "放宽一次检索条件(例如去掉日期或分类限制)再试一次"
TELL_USER_HINT = "告诉用户没有找到,不要编造论文编号"
REPEAT_HINT = "同一调用本轮已执行过两次,不再执行;请换用其他参数或工具,或直接回答"
CLARIFYING_HINT = "与 ask_clarification 同一步发出,未执行"
REFUSED_STEP_HINT = "本步有调用不合法,整步都未执行;改正后重新给出这一步的全部调用"
STOPPED_HINT = "本轮已结束,未执行"

_EMPTY_RESULTS_HINTS = {"relax": RELAX_HINT, "tell_user": TELL_USER_HINT}  # decide's answer -> hint
_ISSUE_ERRORS = {"bad_json": INVALID_JSON, "unknown_tool": UNKNOWN_TOOL, "invalid_args": INVALID_ARGUMENTS}

Status = tuple[str, str | None]  # (status, final_text)


class Model(Protocol):
    """What the loop needs from a model adapter: agent.models, or a scripted stand-in in tests.

    Every call of a turn needs an id of its own, never empty: tool messages and step issues name calls by id.
    """

    @property
    def name(self) -> str: ...

    def next_turn(self, messages: list[dict], tools: list[dict]) -> AssistantTurn: ...


_HISTORY: ContextVar[list[dict] | None] = ContextVar("research_agent_history", default=None)


def current_history() -> list[dict] | None:
    """The whole history of the request a model is answering (module docstring: The whole history), None
    outside a request. Readers must not change it."""
    return _HISTORY.get()


@contextmanager
def asking(history: list[dict]) -> Iterator[None]:
    """current_history() is history while the block runs, and what it was before afterwards."""
    token = _HISTORY.set(history)
    try:
        yield
    finally:
        _HISTORY.reset(token)


def run_turn(request: str, *, model: Model, executor: ToolExecutor, decide: Decide, today: str,
             tools: list[dict] = MODEL_TOOLS, max_steps: int = 8, max_model_calls_per_step: int = 3,
             count_tokens: TokenCounter | None = None, handoff_model: Model | None = None,
             cost_of: Callable[[AssistantTurn], float] | None = None) -> Transcript:
    """Run one request to its end. It starts with the begin_task hook of the model and the handoff model, for
    those that have one (begin_task).

    How decide(event, times) counts: for the step-check events (schemas.PER_STEP_EVENTS: bad_json,
    unknown_tool, invalid_args) times restarts at 1 in every step and counts the rejected attempts of that
    step, whichever of the three events each one had, so "repair once, then hand off" holds for every step;
    for empty_results and repeat_call it counts that event over the whole turn ("relax once, then tell the
    user").

    BudgetExceeded and HarnessError end the turn with their status instead of propagating; any other
    exception (an API error that outlasted its retries, a bug) propagates. Every model turn counts once in
    the accounts: as a paid call, a cache hit (AssistantTurn.cached) or, when it is neither, a local call.
    cost_of prices one model turn, e.g. CloudModel.cost; without it cost_rmb stays 0.0. When the paid call
    that reached the cap raises BudgetExceeded, that call is cached and in the ledger but not in this
    transcript's accounts. today must be the executor's today, so the system prompt and the owner's
    check_call work with the same date.
    """
    if executor.today != today:
        raise ValueError(f"today is {today!r} but the executor checks calls against {executor.today!r}")
    begin_task(model, handoff_model)
    executor.reset()
    run = _Run(model=model, executor=executor, decide=decide, tools=tools,
               max_model_calls_per_step=max_model_calls_per_step, count_tokens=count_tokens,
               handoff_model=handoff_model, cost_of=cost_of)
    run.messages = [{"role": "system", "content": system_prompt(today)},
                    {"role": "user", "content": request}]
    status, final_text = run.run(max_steps)
    return Transcript(request=request, today=today, status=status, final_text=final_text,
                      messages=run.messages, steps=run.steps, schema_hash=SCHEMA_HASH,
                      prompt_version=PROMPT_VERSION, prompt_hash=PROMPT_HASH, paid_calls=run.paid_calls,
                      cache_hits=run.cache_hits, local_calls=run.local_calls, cost_rmb=run.cost_rmb,
                      usage=run.usage)


def begin_task(*models: Model | None) -> None:
    """Tell every model that has the optional begin_task hook (agent.route.RoutedModel: a routed task stays in
    the cloud until the next task begins) that a new task starts; each model once, models without it as they
    are."""
    told: list[object] = []
    for model in models:
        hook = getattr(model, "begin_task", None) if model is not None else None
        if hook is not None and not any(model is m for m in told):
            told.append(model)
            hook()


def run_batch(requests: Iterable[str], **kwargs) -> list[Transcript]:
    """run_turn on each request in order, with the same keyword arguments.

    A budget_stop ends the whole batch: the spending cap was reached.
    """
    transcripts = []
    for request in requests:
        transcripts.append(run_turn(request, **kwargs))
        if transcripts[-1].status == BUDGET_STOP:
            break
    return transcripts


def save_transcript(transcript: Transcript, path: str | Path) -> None:
    """transcript.to_json() as UTF-8 with LF line ends and a final newline; parent folders are created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(transcript.to_json() + "\n", encoding="utf-8", newline="\n")


def load_transcript(path: str | Path) -> Transcript:
    return Transcript(**json.loads(Path(path).read_text(encoding="utf-8")))


class _ContextOverflow(Exception):
    """A request over the token budget: the turn ends as context_overflow and the request is never sent."""


class _Run:
    """The state of one turn while it runs."""

    def __init__(self, *, model: Model, executor: ToolExecutor, decide: Decide, tools: list[dict],
                 max_model_calls_per_step: int, count_tokens: TokenCounter | None,
                 handoff_model: Model | None, cost_of: Callable[[AssistantTurn], float] | None) -> None:
        self.model, self.executor, self.decide, self.tools = model, executor, decide, tools
        self.max_model_calls, self.count_tokens = max_model_calls_per_step, count_tokens
        self.handoff_model, self.cost_of = handoff_model, cost_of
        self.messages: list[dict] = []  # the accepted history, API format
        self.steps: list[dict] = []
        self.event_times: dict[str, int] = {}  # turn-wide event -> times decide was asked about it
        self.runs: dict[str, int] = {}  # call_key -> times the call ran in this turn
        self.paid_calls = self.cache_hits = self.local_calls = 0
        self.cost_rmb = 0.0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def run(self, max_steps: int) -> Status:
        for index in range(1, max_steps + 1):
            step = {"index": index, "model": self.model.name, "attempts": [], "executed": [],
                    "decisions": []}
            self.steps.append(step)
            try:
                accepted = self._ask(step, build_context(self.messages))
            except BudgetExceeded:
                return BUDGET_STOP, None
            except _ContextOverflow:
                return CONTEXT_OVERFLOW, None
            except HarnessError as exc:
                step["harness_error"] = str(exc)
                return HARNESS_ERROR, None
            if accepted is None:
                return INVALID_OUTPUT, None
            self.messages.append(accepted.to_api())
            if not accepted.tool_calls:
                return ANSWERED, accepted.content
            ended = self._answer_calls(step, accepted.tool_calls)
            if ended is not None:
                return ended
        return STEP_LIMIT, None

    def _ask(self, step: dict, context: list[dict]) -> AssistantTurn | None:
        """The step the model settles on, or None when it stays invalid.

        Every attempt goes to the step record; a repair request is built on the side, so the history stays
        as it is. decide hears about every rejected attempt, the last one included, but once the step has
        used up its model calls no action can bring another answer.
        """
        request = context
        for number in range(1, self.max_model_calls + 1):
            turn = self._send(request)
            ids = [c.id for c in turn.tool_calls]
            if not all(ids) or len(set(ids)) != len(ids):  # the adapters guarantee this; a breach is a bug
                raise HarnessError(f"model {turn.model!r} returned empty or duplicate tool call ids: {ids}")
            attempt = {"model": turn.model, "content": turn.content,
                       "tool_calls": [c.to_api() for c in turn.tool_calls], "issues": None, "action": None}
            if turn.reasoning is not None:  # a thinking turn's (agent.types.AssistantTurn); others' stay
                attempt["reasoning"] = turn.reasoning
            if turn.thinking is not None:
                attempt["thinking"] = turn.thinking
            if turn.votes is not None:  # a voted and a ruled turn's (module docstring: The whole history)
                attempt["votes"] = turn.votes
            if turn.rules is not None:
                attempt["rules"] = turn.rules
            step["attempts"].append(attempt)
            issues = self.executor.check_step(turn.tool_calls)
            attempt["issues"] = [asdict(issue) for issue in issues]
            if not issues:
                return turn
            stray = sorted({i.event for i in issues} - set(PER_STEP_EVENTS))
            if stray:  # check_step only reports these three events; anything else is an executor bug
                raise HarnessError(f"check_step reported events outside {PER_STEP_EVENTS}: {stray}")
            action = self._decide(step, issues[0].event, issues[0].call_id)
            attempt["action"] = action
            if number == self.max_model_calls or action == "stop":
                return None  # "stop", or no model call left in this step, whatever the action
            if action == "repair":
                request = context + _rejected_exchange(turn, issues)
            elif self.handoff_model is None or self.model is self.handoff_model:
                return None  # a handoff with nowhere to go
            else:
                self.model = self.handoff_model  # for the rest of the turn
                step["model"] = self.model.name
                request = context
        return None  # max_model_calls_per_step is 0

    def _send(self, request: list[dict]) -> AssistantTurn:
        """The one way a request reaches a model: measured first when there is a token counter, and answered
        with the accepted history as current_history() (module docstring: The whole history)."""
        if self.count_tokens is not None and not fits(request, self.tools, self.count_tokens):
            raise _ContextOverflow
        with asking(self.messages):
            turn = self.model.next_turn(request, self.tools)
        self._account(turn)
        return turn

    def _answer_calls(self, step: dict, calls: list[ToolCall]) -> Status | None:
        """One tool message per call, in call order; the status when the turn ends with this step."""
        ask = next((c for c in calls if c.name == ASK), None)
        if ask is not None:
            question = (ask.arguments or {}).get("question", "")
            for c in calls:
                if c is ask:
                    self._reply(step, ToolResult(c.id, c.name, {"question": question, "status": "asked"}))
                else:
                    self._reply(step, _skipped(c, CLARIFYING_HINT))
            return CLARIFICATION, question
        for i, c in enumerate(calls):
            try:
                stop = self._run_call(step, c)
            except HarnessError as exc:
                step["harness_error"] = str(exc)
                for rest in calls[i:]:  # the failed call included: its output is never shown
                    self._reply(step, _skipped(rest, STOPPED_HINT))
                return HARNESS_ERROR, None
            if stop:
                for rest in calls[i + 1:]:
                    self._reply(step, _skipped(rest, STOPPED_HINT))
                return DEAD_LOOP, None
        return None

    def _run_call(self, step: dict, call: ToolCall) -> bool:
        """Run one accepted call, or answer it SKIPPED when it repeats; True when decide stops the turn.

        The reply comes last, so a HarnessError never leaves this call answered twice.
        """
        key = call_key(call)
        if self.runs.get(key, 0) >= MAX_RUNS_PER_CALL:
            stop = self._decide(step, "repeat_call", call.id) == "stop"  # else "tell_user": the turn goes on
            self._reply(step, _skipped(call, REPEAT_HINT))
            return stop
        result = self.executor.run(call)
        self.runs[key] = self.runs.get(key, 0) + 1
        if result.error == EMPTY_RESULTS:
            hint = _EMPTY_RESULTS_HINTS[self._decide(step, "empty_results", call.id)]
            result = ToolResult(result.call_id, result.name, _with_hint(result.content, hint), result.error)
        self._reply(step, result)
        return False

    def _decide(self, step: dict, event: str, call_id: str) -> str:
        """The owner's decide(event, times), recorded in the step; run_turn says how times counts.

        Like check_call in the executor, an exception or an answer outside ALLOWED_ACTIONS[event] is a
        HarnessError: a broken policy must not pass for a failure of the model, and the loop never carries
        out an answer other than the one decide gave.
        """
        allowed = ALLOWED_ACTIONS[event]
        if event in PER_STEP_EVENTS:  # the rejected attempts of this step, whichever event each had
            times = 1 + sum(d["event"] in PER_STEP_EVENTS for d in step["decisions"])
        else:
            times = self.event_times[event] = self.event_times.get(event, 0) + 1
        try:
            action = self.decide(event, times)
        except HarnessError:
            raise
        except Exception as exc:
            raise HarnessError(f"decide({event!r}, {times}) failed: {exc!r}") from exc
        if action not in allowed:
            raise HarnessError(f"decide({event!r}, {times}) must return one of {', '.join(allowed)}, "
                               f"got {action!r}")
        step["decisions"].append({"call_id": call_id, "event": event, "times": times, "action": action})
        return action

    def _reply(self, step: dict, result: ToolResult) -> None:
        self.messages.append(result.to_api())
        step["executed"].append({"call_id": result.call_id, "name": result.name, "error": result.error})

    def _account(self, turn: AssistantTurn) -> None:
        """Every model call counts once, rejected attempts included, with its cost and tokens.

        It is a paid call, a cache hit, or a local call when it is neither paid nor cached.
        """
        if turn.paid:
            self.paid_calls += 1
        elif turn.cached:
            self.cache_hits += 1
        else:
            self.local_calls += 1
        if self.cost_of is not None:
            self.cost_rmb += self.cost_of(turn)
        for key in self.usage:
            self.usage[key] += int(turn.usage.get(key) or 0)


def _skipped(call: ToolCall, hint: str) -> ToolResult:
    return ToolResult(call.id, call.name, {"hint": hint}, SKIPPED)


def _with_hint(content: dict | list, hint: str) -> dict:
    """content with a hint for the model; a list goes under "items", as in ToolResult's error envelope."""
    return {**content, "hint": hint} if isinstance(content, dict) else {"items": content, "hint": hint}


def _rejected_exchange(turn: AssistantTurn, issues: list[StepIssue]) -> list[dict]:
    """The rejected step and one tool message per call: its errors, or SKIPPED for a call that was fine.

    An issue without a call id concerns the whole step (too many calls) and applies to every call.
    """
    whole_step = [issue for issue in issues if not issue.call_id]
    exchange = [turn.to_api()]
    for c in turn.tool_calls:
        own = whole_step + [issue for issue in issues if issue.call_id == c.id]
        if own:
            details = [message for issue in own for message in issue.messages]
            error = ToolResult(c.id, c.name, {"details": details}, _issue_error(own[0].event))
            exchange.append(error.to_api())
        else:
            exchange.append(_skipped(c, REFUSED_STEP_HINT).to_api())
    return exchange


def _issue_error(event: str) -> str:
    """The error code the model sees for a step-check event."""
    if event not in _ISSUE_ERRORS:
        raise HarnessError(f"the executor reported {event!r}, which is not a step-check event")
    return _ISSUE_ERRORS[event]

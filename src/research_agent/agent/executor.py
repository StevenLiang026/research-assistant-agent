"""Tool executor: checks a whole step before any of it runs, then runs the calls and collects the turn's ids.

Checking first means a step whose last call is invalid is refused before its first call waits out a 15 s rate
limit. The collected ids are the "seen ids" the owner's check_call compares against (write_note must name a
paper that appeared in this turn's tool outputs), so a tool puts an arxiv_id in its content only for a paper
that exists. A result holds its content as the model reads it, canonical JSON parsed back, and the ids are
collected from that same value, so an id counts exactly when the model was shown it. A missing adapter, an
exception in a tool or in check_call and a malformed outcome are HarnessErrors: framework failures, never
shown to the model.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator

from ..canonical import canonical_json
from ..tools.schemas import ERROR_CODES, MAX_CALLS_PER_STEP, TOOL_NAMES
from .types import CheckCall, HarnessError, StepIssue, ToolCall, ToolFn, ToolOutcome, ToolResult


class ToolExecutor:
    def __init__(self, tools: dict[str, ToolFn], check_call: CheckCall, *, today: str) -> None:
        self.tools, self.check_call, self.today = tools, check_call, today
        self._seen: set[str] = set()

    @property
    def seen_ids(self) -> list[str]:
        """Paper ids that appeared in this turn's tool outputs, sorted; a new list on every access."""
        return sorted(self._seen)

    def reset(self) -> None:
        """Forget the seen ids: a new turn starts."""
        self._seen.clear()

    def check_step(self, calls: list[ToolCall]) -> list[StepIssue]:
        """Why the step must be refused, in call order; [] when every call may run. Nothing is executed here.

        More than MAX_CALLS_PER_STEP calls is one issue for the whole step: the model has to choose which
        calls to keep before the single calls are worth checking.
        """
        if len(calls) > MAX_CALLS_PER_STEP:
            message = f"{len(calls)} tool calls in one step; at most {MAX_CALLS_PER_STEP} are allowed"
            return [StepIssue("", "invalid_args", [message])]
        issues = []
        for call in calls:
            if call.name not in TOOL_NAMES:
                message = f"unknown tool {call.name!r}; available tools: {', '.join(TOOL_NAMES)}"
                issues.append(StepIssue(call.id, "unknown_tool", [message]))
            elif call.arguments is None:
                issues.append(StepIssue(call.id, "bad_json", [_json_problem(call.arguments_json)]))
            else:
                errors = self._check_args(call)
                if errors:
                    issues.append(StepIssue(call.id, "invalid_args", errors))
        return issues

    def run(self, call: ToolCall) -> ToolResult:
        """Run one accepted call and record the paper ids in its output.

        The result's content is the outcome's content as the model reads it: through canonical JSON and back,
        so a tuple is a list, a number key a string, and nothing the tool keeps can change the result later.
        The paper ids are collected from that same value.
        """
        tool = self.tools.get(call.name)
        if tool is None:
            raise HarnessError(f"no adapter for tool {call.name!r}")
        if call.arguments is None:
            raise HarnessError(f"{call.name} call {call.id!r} has unparsed arguments")
        try:
            outcome = tool(copy.deepcopy(call.arguments))  # a copy: the recorded call never changes
            content = _content_as_read(call.name, outcome)
        except HarnessError:
            raise
        except Exception as exc:
            raise HarnessError(f"tool {call.name!r} failed on call {call.id!r}: {exc!r}") from exc
        self._seen.update(_paper_ids(content))
        return ToolResult(call.id, call.name, content, outcome.error)

    def _check_args(self, call: ToolCall) -> list[str]:
        """The owner's check_call on a copy of the arguments, with the ids seen before this step."""
        try:
            errors = self.check_call(call.name, copy.deepcopy(call.arguments), self.seen_ids, self.today)
        except HarnessError:
            raise
        except Exception as exc:
            raise HarnessError(f"check_call failed on {call.name} call {call.id!r}: {exc!r}") from exc
        if not isinstance(errors, list) or not all(isinstance(e, str) for e in errors):
            raise HarnessError(f"check_call must return a list of strings, got {errors!r}")
        return errors


def _json_problem(arguments_json: str) -> str:
    """Why the arguments did not parse, in words the model can act on when it repairs the call."""
    try:
        json.loads(arguments_json or "{}")
    except json.JSONDecodeError as exc:
        return f"arguments are not valid JSON: {exc.msg} at character {exc.pos}"
    return "arguments must be a JSON object"


def _content_as_read(name: str, outcome: object) -> dict | list:
    """The outcome's content as the model reads it, once the outcome is what a tool must return."""
    if not isinstance(outcome, ToolOutcome):
        raise HarnessError(f"tool {name!r} returned {type(outcome).__name__}, not a ToolOutcome")
    if not isinstance(outcome.content, (dict, list)):
        kind = type(outcome.content).__name__
        raise HarnessError(f"tool {name!r} returned {kind} content, not a dict or list")
    if outcome.error is not None and outcome.error not in ERROR_CODES:
        raise HarnessError(f"tool {name!r} returned the unknown error code {outcome.error!r}")
    return json.loads(canonical_json(outcome.content))  # TypeError on sets and other values JSON cannot hold


def _paper_ids(node: object) -> Iterator[str]:
    """Every non-empty string stored under an "arxiv_id" key, at any depth of a value parsed from JSON."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "arxiv_id" and isinstance(value, str) and value:
                yield value
            else:
                yield from _paper_ids(value)
    elif isinstance(node, list):
        for value in node:
            yield from _paper_ids(value)

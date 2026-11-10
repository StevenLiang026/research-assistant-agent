"""Evaluation runner: one model turn per item, recorded for the owner's scorer. Nothing is scored here.

An item is {"id": str, "messages": [...], ...}: messages is a history in API format that ends with a user or
tool message, as the loop's history does whenever it asks the model for the next step, so build_context
(research_agent.agent.context) gives the model exactly the context the loop would. The runner asks for that
one next turn and records it as it came back, without running any tool call. The owner's metrics module (M2)
compares the recorded turn with the expected answer, which the item's other fields carry; every record passes
them through under "item".

Statuses: ok (the turn is in "output", API format, with its usage, whether it was paid or cached and, for a
cloud turn, the DeepSeek thinking mode it ran in under "mode"; the records of other models have no mode),
context_overflow (the context does not fit, measured with count_tokens as the loop does; the model is not
asked), error (the model raised, HarnessError included; the message is recorded and the run goes on),
budget_stop (a spending cap, raised as BudgetExceeded: that item ends the run) and skipped (every item after
the budget stop). An ok record of a routed model (ftr) also carries the turn's "route" (agent.route), with
signals one of a local model its "signals", and one of a local model in a thinking mode its "reasoning" and
"thinking" (agent.types.AssistantTurn; the records of other turns are as they were). The records carry no
timestamps, so two runs over the same items with the same model answers write byte-identical result files.
Test material runs only as the OpenedTestSet a LockedSet.open returned (research_agent.eval.lock, who opens
what).

The M7 inference wrappers (docs/m7-本地提升计划.md WP2, WP3). The model answers with the item's messages as
the whole history (agent.loop.asking: the rules and the vote judge an answer by them, not by the context it
was sent), and an ok record whose turn carries a vote or the rules' changes (AssistantTurn.votes, .rules)
keeps them under "votes" and "rules". An error record keeps the vote its exception carries ("votes": a voted
step whose chosen answer is a failed one, agent.vote.VotedModel). The records of other models stay as they
were. An error record's "error" and "error_type" are error_fields of the exception, the one function a vote
also makes a failed candidate's record with, so that the failed candidates its votes list (agent.vote) read
alike online, from the exception, and offline, from this record.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from pathlib import Path

from ..agent.context import TokenCounter, build_context, fits
from ..agent.loop import Model, asking, begin_task
from ..agent.types import BUDGET_STOP, CONTEXT_OVERFLOW
from ..llm.budget import BudgetExceeded
from ..tools.schemas import MODEL_TOOLS
from .lock import OpenedTestSet, read_jsonl, refuse_test_material, write_jsonl

OK = "ok"
ERROR = "error"
SKIPPED = "skipped"
RESULT_STATUSES = [OK, CONTEXT_OVERFLOW, ERROR, BUDGET_STOP, SKIPPED]
LAST_ROLES = ("user", "tool")  # the roles a history may end with before the model's next turn
_ITEM_KEYS = ("id", "messages")  # everything else is passed through under "item"


def run_items(items: Iterable[dict], model: Model, *, tools: list[dict] = MODEL_TOOLS,
              count_tokens: TokenCounter | None = None,
              on_item: Callable[[dict], None] | None = None, signals: bool = False) -> list[dict]:
    """One record per item, in item order; the module docstring lists the statuses.

    The ids must be distinct non-empty strings, which is checked before any model call, and so is test
    material: RefusedTestSet unless the items are the OpenedTestSet of a LockedSet.open (eval.lock, who opens
    what). An item without messages, or whose last message is not from the user or a tool, gets an error
    record without a model call. The items are not modified. on_item, when given, is called with every record
    as soon as it is made, before the next item runs (the budget stop and the skipped ones included): the
    timing sidecar's hook.

    Every item is a task of its own: the model's begin_task hook, when it has one (agent.route.RoutedModel),
    is called before every item, so a routed item never sends the next one to the cloud. signals adds to every
    ok record whose turn carries them the local step's signals ("signals": min_decision_prob, mean_logprob,
    hard_reason; agent.route); without it the records are as they were.
    """
    items = items if isinstance(items, OpenedTestSet) else list(items)
    refuse_test_material(items, "run_items' input")
    _check_ids(items)
    results: list[dict] = []

    def add(record: dict) -> None:
        results.append(record)
        if on_item is not None:
            on_item(record)

    for index, item in enumerate(items):
        begin_task(model)
        try:
            record = _run_item(item, model, tools, count_tokens, signals)
        except BudgetExceeded:
            add(_record(item, BUDGET_STOP))
            for rest in items[index + 1:]:
                add(_record(rest, SKIPPED))
            break
        add(record)
    return results


def write_results(results: list[dict], path: str | Path) -> None:
    """results as JSONL, one canonical JSON line each; parent folders are created."""
    write_jsonl(results, path)


def read_results(path: str | Path) -> list[dict]:
    return read_jsonl(path)


def _run_item(item: dict, model: Model, tools: list[dict], count_tokens: TokenCounter | None,
              signals: bool = False) -> dict:
    """The record of one item; BudgetExceeded propagates, every other exception becomes an error record."""
    try:
        messages = _messages(item)
        context = build_context(messages)
        if count_tokens is not None and not fits(context, tools, count_tokens):
            return _record(item, CONTEXT_OVERFLOW)
        with asking(messages):  # the whole history (module docstring: The M7 inference wrappers)
            turn = model.next_turn(context, tools)
    except BudgetExceeded:
        raise
    except Exception as exc:  # noqa: BLE001  deliberate: one item's failure is recorded, the run goes on
        votes = getattr(exc, "votes", None)  # a voted step whose chosen answer failed (agent.vote)
        return _record(item, ERROR, **error_fields(exc), **({} if votes is None else {"votes": votes}))
    extra: dict = {} if turn.mode is None else {"mode": turn.mode}  # none for local models: as they were
    if turn.route is not None:  # a routed model's turns only (ftr)
        extra["route"] = turn.route
    if signals and turn.signals is not None:
        extra["signals"] = turn.signals
    if turn.reasoning is not None:  # a local turn in a thinking mode: other records stay as they were
        extra["reasoning"] = turn.reasoning
    if turn.thinking is not None:
        extra["thinking"] = turn.thinking
    if turn.rules is not None:  # a ruled model's (agent.rules), a voted one's (agent.vote)
        extra["rules"] = turn.rules
    if turn.votes is not None:
        extra["votes"] = turn.votes
    return _record(item, OK, output=turn.to_api(), usage=dict(turn.usage), paid=turn.paid,
                   cached=turn.cached, model=turn.model, **extra)


def error_fields(exc: BaseException) -> dict:
    """What an error record says of the exception a model raised: {"error": its message (str), "error_type":
    its class's name} (module docstring)."""
    return {"error": str(exc), "error_type": type(exc).__name__}


def _messages(item: dict) -> list[dict]:
    messages = item.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("the item has no messages")
    role = messages[-1].get("role")
    if role not in LAST_ROLES:
        raise ValueError(f"the history must end with a user or tool message, not {role!r}")
    return messages


def _record(item: dict, status: str, **fields: object) -> dict:
    """{"id", "status", "item": the item's other fields (copied)} plus the status's own fields."""
    extra = {k: copy.deepcopy(v) for k, v in item.items() if k not in _ITEM_KEYS}
    return {"id": item["id"], "status": status, "item": extra, **fields}


def _check_ids(items: list[dict]) -> None:
    ids = [item.get("id") for item in items]
    if not all(isinstance(i, str) and i for i in ids):
        raise ValueError("every item needs a non-empty string id")
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    if repeated:
        raise ValueError(f"item ids must be distinct; repeated: {repeated}")

"""Draft single-step evaluation items from the synthetic generator's transcripts (docs/plan.md, 评测).

An item asks the model for one step of a recorded turn. Its "messages" are the full API history before that
step, the system message first and a user or tool message last, uncompressed: the runner
(research_agent.eval.runner) builds the context from it exactly as the loop would. Its "expected" is the step
the teacher model took and the loop accepted, for the owner's metrics module (M2) to score: the assistant text
and the tool calls, each with its arguments parsed as the loop parses them (parse_arguments, so 3.0 reads as
3) and verbatim, for strict matching. One item is drafted per accepted assistant step of a transcript, except
a step accepted after a repair: the model answered a repair request that the history does not hold
(agent.context.step_contexts), so no item can show what it saw. Items are named <request id>-s<step number>,
the step number counting every step of the turn, skipped ones included, so an item always points back into
Transcript.steps; "source" repeats the request id and the step number, and "lang", "family", "split" and
"slots" are the request's.

Each item has a type, decided by the first matching rule: a call to ask_clarification gives "clarify"; a
tool message of the step before whose error code is RATE_LIMITED, UPSTREAM_ERROR or EMPTY_RESULTS gives
"recover" (NO_FULLTEXT, NOT_FOUND and SKIPPED are answered outputs, not failures to recover from); two or more
calls give "parallel"; the first step gives "call" with one call and "no_call" with none; every other step is
"next_step", the plan's "看过工具输出后的下一步". The plan's quotas are QUOTAS_TEST, 300 items as 120 call,
30 next_step, 50 no_call, 40 clarify, 40 recover and 20 parallel, and QUOTAS_DEV, the same proportions at 150
(60, 15, 25, 20, 20, 10). The drafter fills a share of each quota (70 % by default, the --share of
scripts/build_eval_items.py, scaled here) and the rest is written by hand, so that the sets hold cases
no generator template produces. select draws that share per (type, language) stratum, half of each quota per
language, reproducibly from a seed, and reports per stratum the quota, what was available, what was taken and
the shortfall. A shortfall is expected while the request pool is small and is not an error.

The input is a generator run directory: index.jsonl, one record per request (the request's keys, its split,
status and verdict, and the path of its transcript), and transcripts/<id>.json as agent.loop.save_transcript
wrote them. draft_items takes the requests of the wanted splits that have a transcript, whose turn did not end
as a harness error, and whose verdict (the owner's check_trajectory) accepted them, or is still missing when
include_unvalidated is set, the only way to draft before that module exists. A request without a verdict is
taken only when its turn ended answered or as a clarification (DRAFTABLE_STATUSES): the steps of a turn that
stopped as a dead loop, at the step limit, over budget, on an invalid output or in a context overflow were
accepted by the loop one by one, but the turn as a whole went wrong somewhere, and without a verdict nothing
says which steps are worth imitating. An accepted verdict keeps the turn whatever its status: the validator
judged it. A transcript recorded under another tool schema is refused: the items would show the model calls
against tools it is not given. The CLI writes data/eval/dev-draft.jsonl and test-draft.jsonl; the sets the
owner finishes from them are locked by scripts/lock_set.py (research_agent.eval.lock).
"""

from __future__ import annotations

import copy
import json
import math
import random
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path

from ..agent.loop import ASK, load_transcript
from ..agent.types import (
    ANSWERED,
    BUDGET_STOP,
    CLARIFICATION,
    CONTEXT_OVERFLOW,
    DEAD_LOOP,
    HARNESS_ERROR,
    INVALID_OUTPUT,
    STEP_LIMIT,
    HarnessError,
    ToolCall,
    Transcript,
)
from ..tools.schemas import EMPTY_RESULTS, RATE_LIMITED, SCHEMA_HASH, UPSTREAM_ERROR
from ..train.samples import accepted_after_repair
from .lock import read_jsonl
from .runner import LAST_ROLES

CALL = "call"
NEXT_STEP = "next_step"
NO_CALL = "no_call"
CLARIFY = "clarify"
RECOVER = "recover"
PARALLEL = "parallel"
TYPES = (CALL, NEXT_STEP, NO_CALL, CLARIFY, RECOVER, PARALLEL)  # in the plan's order
LANGS = ("zh", "en")
SPLITS = ("dev", "test")
RECOVER_CODES = (RATE_LIMITED, UPSTREAM_ERROR, EMPTY_RESULTS)  # the failures a "recover" step follows
QUOTAS_TEST = {CALL: 120, NEXT_STEP: 30, NO_CALL: 50, CLARIFY: 40, RECOVER: 40, PARALLEL: 20}  # 300
QUOTAS_DEV = {CALL: 60, NEXT_STEP: 15, NO_CALL: 25, CLARIFY: 20, RECOVER: 20, PARALLEL: 10}  # 150: half
INDEX_FILE = "index.jsonl"
DRAFT_FILES = {"dev": "dev-draft.jsonl", "test": "test-draft.jsonl"}  # under the CLI's --out-dir
DRAFTABLE_STATUSES = (ANSWERED, CLARIFICATION)  # the turn ends a request without a verdict may have
# Why an index record is left out, in the order the CLI's tally lists them. A request without a verdict whose
# turn ended another way is left out under a reason named after its status, UNKNOWN_STATUS for a status the
# loop never records, so that skip_reason always answers with one of SKIP_REASONS and the tally adds up.
OTHER_SPLIT = "other split"
NO_TRANSCRIPT = "no transcript"
HARNESS_FAILED = "harness error"
STOPPED = {DEAD_LOOP: "dead loop", STEP_LIMIT: "step limit", BUDGET_STOP: "budget stop",
           INVALID_OUTPUT: "invalid output", CONTEXT_OVERFLOW: "context overflow"}  # in STATUSES order
UNKNOWN_STATUS = "unknown status"
REJECTED = "rejected"
NOT_VALIDATED = "not validated"
SKIP_REASONS = (OTHER_SPLIT, NO_TRANSCRIPT, HARNESS_FAILED, *STOPPED.values(), UNKNOWN_STATUS, REJECTED,
                NOT_VALIDATED)


def items_from_transcript(transcript: Transcript, request: dict) -> list[dict]:
    """One item per accepted assistant step, in step order; the module docstring gives the item's shape.

    The k-th assistant message of the history is the accepted attempt of steps[k-1], which must be the
    same message (content and calls); a step accepted after a repair gives no item. HarnessError when the
    history and the steps do not fit together (the history not starting with the system message, more
    assistant messages than steps, a step before the last without its message: only the last step of a turn
    may end without an accepted message), when the history before an assistant message does not end with a
    user or tool message, when the arguments of an accepted call are not a JSON object, or when the request
    lacks its id, lang or split.
    """
    request_id = _required(request, "id")
    messages = transcript.messages
    positions = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    _check_pairing(messages, positions, transcript.steps, request_id)
    slots = request.get("slots")
    items = []
    for k, i in enumerate(positions, 1):
        if accepted_after_repair(transcript.steps[k - 1]):
            continue
        history, message = messages[:i], messages[i]
        last = history[-1].get("role")
        if last not in LAST_ROLES:
            raise HarnessError(f"request {request_id}, step {k}: the history before the step ends with a "
                               f"{last!r} message, not a user or tool message")
        calls = [_expected_call(c, request_id, k) for c in message.get("tool_calls") or []]
        items.append({
            "id": f"{request_id}-s{k}",
            "messages": copy.deepcopy(history),
            "expected": {"content": message.get("content") or "", "tool_calls": calls},
            "type": item_type(history, message, k),
            "lang": _required(request, "lang"),
            "family": request.get("family"),
            "split": _required(request, "split"),
            "slots": {} if slots is None else copy.deepcopy(slots),
            "source": {"request": request_id, "step": k},
        })
    return items


def item_type(history: list[dict], message: dict, step: int) -> str:
    """The type of the assistant message produced after history as step number step (1-based).

    The rules, first match wins: a call named ask_clarification is "clarify"; a tool message of the step
    before (the tool messages at the end of history, back to its last assistant message) whose error code is
    one of RECOVER_CODES makes it "recover"; two or more calls are "parallel"; the first step is "call" with
    one call and "no_call" without; anything else is "next_step".
    """
    names = [(c.get("function") or {}).get("name") for c in message.get("tool_calls") or []]
    if ASK in names:
        return CLARIFY
    if any(_error_code(m) in RECOVER_CODES for m in _previous_tool_messages(history)):
        return RECOVER
    if len(names) >= 2:
        return PARALLEL
    if step == 1:
        return CALL if names else NO_CALL
    return NEXT_STEP


def skip_reason(row: dict, *, splits: Sequence[str] = SPLITS, include_unvalidated: bool = True) -> str | None:
    """Why the drafter leaves an index record out (one of SKIP_REASONS), or None when it uses it.

    A record is used when its split is in splits, it names a transcript (a request whose generation failed
    has none), its turn did not end as a harness error, and its verdict accepted it, or there is no verdict
    yet, include_unvalidated is set and the turn ended with one of DRAFTABLE_STATUSES. The first failing
    condition, in that order, is the reason: under include_unvalidated=False an unvalidated record is "not
    validated" whatever its status, and a verdict decides a validated record whatever its status.
    """
    if row.get("split") not in splits:
        return OTHER_SPLIT
    if not isinstance(row.get("transcript"), str) or not row["transcript"]:
        return NO_TRANSCRIPT
    if row.get("status") == HARNESS_ERROR:
        return HARNESS_FAILED
    verdict = row.get("verdict")
    if verdict is None:
        if not include_unvalidated:
            return NOT_VALIDATED
        status = row.get("status")
        if status in DRAFTABLE_STATUSES:
            return None
        return STOPPED.get(status, UNKNOWN_STATUS) if isinstance(status, str) else UNKNOWN_STATUS
    return None if isinstance(verdict, dict) and verdict.get("accepted") else REJECTED


def draftable(row: dict, *, splits: Sequence[str] = SPLITS, include_unvalidated: bool = True) -> bool:
    """True for an index record the drafter uses; skip_reason says why one is left out."""
    return skip_reason(row, splits=splits, include_unvalidated=include_unvalidated) is None


def draft_items(run_dir: str | Path, *, splits: Sequence[str] = SPLITS,
                include_unvalidated: bool = True) -> list[dict]:
    """The items of every draftable request of the run (draftable says which), in index order.

    Malformed input raises: JsonlError for an index line that is not a JSON object, FileNotFoundError for a
    missing index or transcript, TypeError for a transcript with keys Transcript does not have, HarnessError
    for one recorded under another tool schema or whose history and steps do not fit together.
    """
    run_dir = Path(run_dir)
    items: list[dict] = []
    for row in read_jsonl(run_dir / INDEX_FILE):
        if not draftable(row, splits=splits, include_unvalidated=include_unvalidated):
            continue
        transcript = load_transcript(run_dir / row["transcript"])
        if transcript.schema_hash != SCHEMA_HASH:
            raise HarnessError(f"{row['transcript']} was recorded under tool schema "
                               f"{transcript.schema_hash}; the current schema is {SCHEMA_HASH}")
        items.extend(items_from_transcript(transcript, row))
    return items


def summarize(items: list[dict]) -> dict:
    """Counts by split, type and language: {split: {type: {lang: count}}}.

    Every split of the items is present with every TYPES x LANGS cell, zeros included; a type or language
    outside those lists is appended after them, so nothing is lost in the table.
    """
    tally = Counter((item["split"], item["type"], item["lang"]) for item in items)
    types = TYPES + tuple(sorted({t for _, t, _ in tally} - set(TYPES)))
    langs = LANGS + tuple(sorted({lang for _, _, lang in tally} - set(LANGS)))
    return {split: {t: {lang: tally[split, t, lang] for lang in langs} for t in types}
            for split in sorted({s for s, _, _ in tally})}


def select(items: list[dict], quotas: dict[str, int], *, seed: int,
           langs: Sequence[str] = LANGS) -> tuple[list[dict], dict]:
    """A stratified draw: (the drawn items, the report of every (type, language) stratum).

    Each type's quota is split evenly over langs, the remainder going to the languages listed first (an odd
    quota over two languages gives the first one more). A stratum draws without replacement from its
    candidates sorted by id, with a random.Random of its own seeded from seed and the stratum's name, so a
    draw is reproducible and the draw of one stratum stays the same when another stratum's candidates change.
    The drawn items come grouped by stratum, in quotas and langs order, sorted by id within a stratum. The
    report is {type: {lang: {"quota", "available", "taken", "shortfall"}}}. Items of a type outside quotas or
    a language outside langs are left out. ValueError when ids repeat (the runner refuses such a set).
    """
    if not langs:
        raise ValueError("langs must name at least one language")
    repeated = sorted(i for i, n in Counter(item["id"] for item in items).items() if n > 1)
    if repeated:
        raise ValueError(f"item ids must be distinct; repeated: {repeated}")
    chosen: list[dict] = []
    report: dict[str, dict[str, dict[str, int]]] = {}
    for type_, quota in quotas.items():
        report[type_] = {}
        for lang, share in zip(langs, _split_evenly(quota, len(langs)), strict=True):
            candidates = sorted((it for it in items if it["type"] == type_ and it["lang"] == lang),
                                key=lambda it: it["id"])
            rng = random.Random(f"{seed}\x1f{type_}\x1f{lang}")
            taken = sorted(rng.sample(candidates, min(share, len(candidates))), key=lambda it: it["id"])
            chosen.extend(taken)
            report[type_][lang] = {"quota": share, "available": len(candidates), "taken": len(taken),
                                   "shortfall": share - len(taken)}
    return chosen, report


def scaled(quotas: dict[str, int], share: float) -> dict[str, int]:
    """quotas at a share of their size: integers whose total is the total times share, rounded half up.

    Each quota is taken at its share and floored; the units the floors lack to reach that total go to the
    quotas with the largest fractional parts, ties to the one listed first. Without ties this is rounding
    half up: scaled(QUOTAS_TEST, 0.7) is 84, 21, 35, 28, 28, 14 (210). With them the total still holds:
    scaled(QUOTAS_DEV, 0.7) has 10.5 and 17.5 to round and a total of 105 to reach, so next_step gets 11
    and no_call 17. share must be in [0, 1]; a float is read as the decimal it prints as, so 0.7 is 7/10.
    """
    ratio = Fraction(str(share))
    if not 0 <= ratio <= 1:
        raise ValueError(f"share must be in [0, 1], not {share!r}")
    raw = {name: Fraction(quota) * ratio for name, quota in quotas.items()}
    floors = {name: math.floor(value) for name, value in raw.items()}
    target = math.floor(sum(raw.values(), Fraction(0)) + Fraction(1, 2))
    extra = target - sum(floors.values())
    by_remainder = sorted(quotas, key=lambda name: floors[name] - raw[name])  # stable: ties keep their order
    raised = set(by_remainder[:extra])
    return {name: floors[name] + (1 if name in raised else 0) for name in quotas}


def _required(request: dict, key: str) -> object:
    value = request.get(key)
    if value is None or value == "":
        raise HarnessError(f"the request record has no {key!r}: {sorted(request)}")
    return value


def _check_pairing(messages: list[dict], positions: list[int], steps: list[dict], request_id: str) -> None:
    """The k-th assistant message must be the accepted attempt of steps[k-1]; only the last step may lack
    one, as a turn that ends without an accepted step leaves it (budget stop, invalid output, overflow)."""
    if not messages or messages[0].get("role") != "system":
        raise HarnessError(f"request {request_id}: the history does not start with the system message")
    n = len(positions)
    if n > len(steps):
        raise HarnessError(f"request {request_id}: the history has {n} assistant messages but the transcript "
                           f"only {len(steps)} steps")
    if n < len(steps) - 1:
        raise HarnessError(f"request {request_id}: the transcript has {len(steps)} steps but the history "
                           f"only {n} assistant messages; only the last step may end without an accepted "
                           "message")
    for k, i in enumerate(positions, 1):
        attempt = _accepted_attempt(steps[k - 1])
        message = messages[i]
        if attempt is None or (attempt.get("content") or "") != (message.get("content") or "") or \
                (attempt.get("tool_calls") or []) != (message.get("tool_calls") or []):
            raise HarnessError(f"request {request_id}: assistant message {k} is not the accepted attempt of "
                               f"step {k}")
    if n == len(steps) - 1 and _accepted_attempt(steps[-1]) is not None:
        raise HarnessError(f"request {request_id}: step {len(steps)} was accepted but the history holds no "
                           "message for it")


def _accepted_attempt(step: dict) -> dict | None:
    """The step's last attempt when the executor accepted it (no issues), else None."""
    attempts = step.get("attempts") or []
    if attempts and attempts[-1].get("issues") == []:
        return attempts[-1]
    return None


def _expected_call(call: dict, request_id: str, step: int) -> dict:
    parsed = ToolCall.from_api(call)
    if parsed.arguments is None:
        raise HarnessError(f"request {request_id}, step {step}: the arguments of the accepted call "
                           f"{parsed.name!r} are not a JSON object: {parsed.arguments_json!r}")
    return {"name": parsed.name, "arguments": parsed.arguments, "arguments_json": parsed.arguments_json}


def _previous_tool_messages(history: list[dict]) -> list[dict]:
    """The tool messages at the end of history: the results of the step before the next assistant message."""
    found = []
    for message in reversed(history):
        if message.get("role") != "tool":
            break
        found.append(message)
    return found


def _error_code(message: dict) -> str | None:
    """The "error" of a tool message's JSON object content; None for success, non-JSON or non-object text."""
    content = message.get("content")
    if not isinstance(content, str):
        return None
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return None
    code = value.get("error") if isinstance(value, dict) else None
    return code if isinstance(code, str) else None


def _split_evenly(quota: int, parts: int) -> list[int]:
    """quota over parts, the remainder one unit each to the first parts: 5 over 2 is [3, 2]."""
    base, remainder = divmod(quota, parts)
    return [base + (1 if index < remainder else 0) for index in range(parts)]

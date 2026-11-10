"""Trajectory validation: whether a synthetic trajectory may enter the training set (docs/plan.md, 微调数据
第 5 条; docs/modules/接口.md, 轨迹校验).

check_trajectory(transcript, request) takes a transcript as Transcript.to_json() reads back and its request
record, and returns the problems, sorted; [] means the trajectory is accepted. It is pure (neither input is
changed) and never raises: a key the checks need that is missing or has the wrong type is reported as a
problem, because the generator treats an exception as a harness error.

Steps are numbered as the loop numbers them: the n-th assistant message of the history is step n, and a
tool message belongs to the step of the assistant message before it. Every problem starts with the name of
its check, followed by " step <n>" when it concerns one step, then ": " and the detail.

1. calls: every call of every assistant message has arguments that parse to a JSON object (parse_arguments)
   and validation.check_call(name, args, seen_ids, today) returns []. seen_ids are the canonical ids that
   appeared in the tool outputs before that step, collected as the executor collects them (every non-empty
   string under an "arxiv_id" key, at any depth), sorted and distinct; an unknown tool is check_call's
   "tool: unknown tool <name>".
       "calls step <n>: <tool> call <id>: arguments are not a JSON object"
       "calls step <n>: <tool> call <id>: <check_call message>"
2. execution: the status is answered or clarification, and no tool message carries an error code. For a
   recover request the codes in synth.RECOVER_CODES are allowed and at least one of them must occur;
   NO_FULLTEXT is the designed fall-back to the abstract (the tool still delivers text) and is never a
   failure, nor does it count as a recovery; NOT_FOUND and SKIPPED are failures in every scenario.
       "execution: status <status> is not answered or clarification"
       "execution step <n>: <tool> call <id> returned <code>"
       "execution step <n>: <tool> call <id> has content that is not JSON"
       "execution: recover scenario but no tool output carries EMPTY_RESULTS, RATE_LIMITED or UPSTREAM_ERROR"
3. ids: every arXiv id in the final text (candidates matched by pattern on the NFKC text and kept when
   normalize_arxiv_id accepts them) appeared in a tool output of this turn; an id the user typed counts only
   once a tool returned it. An answered turn must have a final text.
       "ids: final text cites <id>, which did not appear in any tool output this turn"
       "ids: final_text is missing on an answered turn"
4. constraints: for each of date_from, date_to and categories that the request's slots carry, every
   search_arxiv call carries the same value (dates compared after normalize_date, categories as sets), and
   every hit those searches returned has a published date inside the slots' window; when the slots carry a
   topic (a string or a list of strings), at least one search query contains it (query_contains_all).
       "constraints: slot <key> is not <a YYYY-MM-DD date | a list of strings | a string>: <value>"
       "constraints step <n>: search_arxiv call <id> has <key> <value or none>, the request asks for <value>"
       "constraints step <n>: search_arxiv call <id> returned <aid> published <date>, before date_from <d>"
       "constraints step <n>: search_arxiv call <id> returned <aid> published <date>, after date_to <d>"
       "constraints step <n>: search_arxiv call <id> returned a hit without a YYYY-MM-DD published date"
   (<aid> is the hit's arxiv_id.)
       "constraints: no search_arxiv query contains the topic <topic>" (only when the turn made at least
       one search_arxiv call with parseable arguments: a request that names its ids needs no search)
5. fields: every write_note call carries the fields its mode requires (NOTE_FIELDS, from the schema's
   description of the two modes: scaffold is the evidence sentences, draft adds the four parts) as non-empty
   strings, and every list parameter whose schema has minItems that many items; a no_call request makes no
   tool call, and a clarify request ends as clarification with an ask_clarification call.
       "fields step <n>: write_note call <id> in <mode> mode needs a non-empty <field>"
       "fields step <n>: write_note call <id> needs at least <k> item(s) in <field>"
       "fields: no_call scenario but the turn made <k> tool call(s)"
       "fields: clarify scenario but the turn ended as <status>"
       "fields: clarify scenario but no ask_clarification call"

Structural problems come first and stop the checks, since nothing after them could be read reliably:
       "transcript: is not a dict" / "request: is not a dict"
       "transcript: missing key <key>" / "request: missing key <key>"
       "transcript: <key> is not <type>" / "request: <key> is not <type>"
       "transcript: message <i> is malformed: <why>"
       "transcript: <tool> call <id> of step <n> has no tool message"

Should a check itself raise on an input the rules above did not foresee, the trajectory is rejected with a
single problem instead of the exception propagating:
       "check_trajectory: the checks failed on this input: <error>"   (<error> is the exception's repr)
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator

from .agent.types import ANSWERED, CLARIFICATION, parse_arguments
from .data.synth import RECOVER_CODES
from .normalize import normalize_arxiv_id, normalize_date, normalize_text, query_contains_all
from .tools import schemas
from .validation import check_call

ASK = "ask_clarification"
SEARCH = "search_arxiv"
NOTE = "write_note"
# The fields each write_note mode requires besides arxiv_id. The schema states them in words ("scaffold:
# evidence sentences only; draft: also the four parts"); the four parts are problem, method, evidence and
# limitations, the string properties of write_note.
NOTE_FIELDS = {"scaffold": ("evidence",), "draft": ("problem", "method", "evidence", "limitations")}
CONSTRAINT_DATES = ("date_from", "date_to")

TRANSCRIPT_KEYS = {"today": str, "status": str, "final_text": (str, type(None)), "messages": list}
REQUEST_KEYS = {"scenario": str, "slots": dict}
_TYPE_NAMES = {str: "a string", list: "a list", dict: "a dict", (str, type(None)): "a string or null"}

# An arXiv id candidate: 4 digits, a dot, 4 or 5 digits, an optional version, not inside a longer number.
_ID_CANDIDATE = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?![\d])")


def check_trajectory(transcript: dict, request: dict) -> list[str]:
    """The problems of a trajectory, sorted; [] when it is accepted. The module docstring is the contract."""
    try:
        structure = _structure_problems(transcript, request)
        if structure:
            return sorted(structure)
        steps, structure = _read_steps(transcript["messages"])
        if structure:
            return sorted(structure)
        problems = (_check_calls(steps, transcript["today"])
                    + _check_execution(steps, transcript["status"], request["scenario"])
                    + _check_ids(steps, transcript["status"], transcript["final_text"])
                    + _check_constraints(steps, request["slots"])
                    + _check_fields(steps, transcript["status"], request["scenario"]))
    except Exception as exc:  # noqa: BLE001  deliberate: a bug here rejects the trajectory, never raises
        return [f"check_trajectory: the checks failed on this input: {exc!r}"]
    return sorted(problems)


# ---- reading the transcript -------------------------------------------------------------------------------


def _structure_problems(transcript: object, request: object) -> list[str]:
    problems = []
    records = (("transcript", transcript, TRANSCRIPT_KEYS), ("request", request, REQUEST_KEYS))
    for label, record, keys in records:
        if not isinstance(record, dict):
            problems.append(f"{label}: is not a dict")
            continue
        for key, kind in keys.items():
            if key not in record:
                problems.append(f"{label}: missing key {key}")
            elif not isinstance(record[key], kind):
                problems.append(f"{label}: {key} is not {_TYPE_NAMES[kind]}")
    return problems


class _Call:
    """One tool call of an accepted step, with its tool message's body once it has one."""

    def __init__(self, step: int, call_id: str, name: str, args: dict | None) -> None:
        self.step, self.id, self.name, self.args = step, call_id, name, args
        self.answered = False
        self.body: object = None  # the tool message's content parsed from JSON
        self.parsed = True  # False when the tool message's content is not JSON


class _Step:
    def __init__(self, number: int, seen_before: list[str]) -> None:
        self.number, self.seen_before = number, seen_before
        self.calls: list[_Call] = []


def _read_steps(messages: list) -> tuple[list[_Step], list[str]]:
    """The accepted steps of the history, or the structural problems that keep it from being read."""
    steps: list[_Step] = []
    by_id: dict[str, _Call] = {}
    seen: set[str] = set()
    problems = []
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            problems.append(f"transcript: message {i} is malformed: not a dict")
            continue
        role = message.get("role")
        if role == "assistant":
            step = _Step(len(steps) + 1, sorted(seen))
            steps.append(step)
            raw_calls = message.get("tool_calls") or []
            if not isinstance(raw_calls, list):
                problems.append(f"transcript: message {i} is malformed: tool_calls is not a list")
                continue
            for raw in raw_calls:
                call = _read_call(raw, step.number)
                if call is None:
                    problems.append(f"transcript: message {i} is malformed: a tool call without an id and a "
                                    "function name")
                    continue
                step.calls.append(call)
                by_id[call.id] = call
        elif role == "tool":
            call = by_id.get(message.get("tool_call_id"))
            content = message.get("content")
            if call is None or call.answered:
                problems.append(f"transcript: message {i} is malformed: a tool message that answers no "
                                "pending call")
                continue
            call.answered = True
            try:
                call.body = json.loads(content) if isinstance(content, str) else None
                call.parsed = isinstance(content, str)
            except json.JSONDecodeError:
                call.parsed = False
            seen.update(_paper_ids(call.body))
    problems += [f"transcript: {call.name} call {call.id} of step {call.step} has no tool message"
                 for call in by_id.values() if not call.answered]
    return steps, problems


def _read_call(raw: object, step: int) -> _Call | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("function"), dict):
        return None
    fn = raw["function"]
    call_id, name = raw.get("id"), fn.get("name")
    if not isinstance(call_id, str) or not call_id or not isinstance(name, str):
        return None
    arguments = fn.get("arguments")
    if isinstance(arguments, str):
        args = parse_arguments(arguments)
    else:  # some servers return a dict; ToolCall.from_api re-serialises it the same way
        args = None
        if isinstance(arguments, dict):
            args = parse_arguments(json.dumps(arguments, ensure_ascii=False))
    return _Call(step, call_id, name, args)


def _paper_ids(node: object) -> Iterator[str]:
    """Every non-empty string stored under an "arxiv_id" key, at any depth: what the executor collects as the
    turn's seen ids from the same canonical content."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "arxiv_id" and isinstance(value, str) and value:
                yield value
            else:
                yield from _paper_ids(value)
    elif isinstance(node, list):
        for value in node:
            yield from _paper_ids(value)


def _all_calls(steps: list[_Step]) -> Iterator[_Call]:
    for step in steps:
        yield from step.calls


def _error_code(call: _Call) -> str | None:
    return call.body.get("error") if isinstance(call.body, dict) else None


# ---- the five checks ------------------------------------------------------------------------------------


def _check_calls(steps: list[_Step], today: str) -> list[str]:
    problems = []
    for step in steps:
        for call in step.calls:
            where = f"calls step {step.number}: {call.name} call {call.id}"
            if call.args is None:
                problems.append(f"{where}: arguments are not a JSON object")
                continue
            for error in check_call(call.name, call.args, list(step.seen_before), today):
                problems.append(f"{where}: {error}")
    return problems


def _check_execution(steps: list[_Step], status: str, scenario: str) -> list[str]:
    problems = []
    if status not in (ANSWERED, CLARIFICATION):
        problems.append(f"execution: status {status} is not answered or clarification")
    allowed = RECOVER_CODES if scenario == "recover" else ()
    recovered = False
    for call in _all_calls(steps):
        where = f"execution step {call.step}: {call.name} call {call.id}"
        if not call.parsed:
            problems.append(f"{where} has content that is not JSON")
            continue
        code = _error_code(call)
        if code is None or code == schemas.NO_FULLTEXT:
            continue
        if code in allowed:
            recovered = True
        else:
            problems.append(f"{where} returned {code}")
    if scenario == "recover" and not recovered:
        codes = f"{', '.join(RECOVER_CODES[:-1])} or {RECOVER_CODES[-1]}"
        problems.append(f"execution: recover scenario but no tool output carries {codes}")
    return problems


def _check_ids(steps: list[_Step], status: str, final_text: str | None) -> list[str]:
    if final_text is None:
        return ["ids: final_text is missing on an answered turn"] if status == ANSWERED else []
    seen = {i for call in _all_calls(steps) for i in _paper_ids(call.body)}
    cited = {normalize_arxiv_id(m.group(1)) for m in _ID_CANDIDATE.finditer(normalize_text(final_text))}
    return [f"ids: final text cites {i}, which did not appear in any tool output this turn"
            for i in sorted(cited - {None} - seen)]


def _check_constraints(steps: list[_Step], slots: dict) -> list[str]:
    problems = []
    wanted: dict[str, object] = {}
    for key in CONSTRAINT_DATES:
        if key in slots:
            value = normalize_date(slots[key]) if isinstance(slots[key], str) else None
            if value is None:
                problems.append(f"constraints: slot {key} is not a YYYY-MM-DD date: {slots[key]}")
            else:
                wanted[key] = value
    if "categories" in slots:
        value = slots["categories"]
        if isinstance(value, list) and all(isinstance(c, str) for c in value):
            wanted["categories"] = sorted(set(value))
        else:
            problems.append(f"constraints: slot categories is not a list of strings: {value}")
    topics = _topics(slots.get("topic"))
    if "topic" in slots and topics is None:
        problems.append(f"constraints: slot topic is not a string: {slots['topic']}")

    searches = [c for c in _all_calls(steps) if c.name == SEARCH and c.args is not None]
    for call in searches:
        where = f"constraints step {call.step}: {SEARCH} call {call.id}"
        for key, want in wanted.items():
            got = _canonical_constraint(key, call.args.get(key))
            if got != want:
                shown = "none" if call.args.get(key) is None else _show(call.args[key])
                problems.append(f"{where} has {key} {shown}, the request asks for {_show(want)}")
        start, end = wanted.get("date_from"), wanted.get("date_to")
        if start is None and end is None:
            continue
        for hit in _hits(call.body):
            published, aid = hit.get("published"), hit.get("arxiv_id")
            day = normalize_date(published) if isinstance(published, str) else None
            if day is None:
                problems.append(f"{where} returned a hit without a YYYY-MM-DD published date")
            elif start is not None and day < start:
                problems.append(f"{where} returned {aid} published {day}, before date_from {start}")
            elif end is not None and day > end:
                problems.append(f"{where} returned {aid} published {day}, after date_to {end}")
    if topics and searches:  # a request that names its ids needs no search, so the topic binds only searches
        queries = [c.args.get("query") for c in searches]
        if not any(isinstance(q, str) and query_contains_all(q, topics) for q in queries):
            problems.append(f"constraints: no {SEARCH} query contains the topic {' '.join(topics)}")
    return problems


def _topics(value: object) -> list[str] | None:
    """The topic slot as the list of terms a query must contain; None when it is neither a string nor a list
    of strings (and when it is absent)."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(t, str) for t in value):
        return list(value)
    return None


def _canonical_constraint(key: str, value: object) -> object:
    if key == "categories":
        strings = isinstance(value, list) and all(isinstance(c, str) for c in value)
        return sorted(set(value)) if strings else None
    return normalize_date(value) if isinstance(value, str) else None


def _show(value: object) -> str:
    return json.dumps(value, ensure_ascii=False) if isinstance(value, list) else str(value)


def _hits(body: object) -> list[dict]:
    """The hits of a search output: the list itself, or the "items" of an error or hint envelope."""
    items = body.get("items") if isinstance(body, dict) else body
    return [h for h in items if isinstance(h, dict)] if isinstance(items, list) else []


def _check_fields(steps: list[_Step], status: str, scenario: str) -> list[str]:
    problems = []
    properties = _note_properties()
    calls = list(_all_calls(steps))
    for call in calls:
        if call.name != NOTE or call.args is None:
            continue
        mode = call.args.get("mode", schemas.DEFAULTS[NOTE]["mode"])
        where = f"fields step {call.step}: {NOTE} call {call.id}"
        # An unknown mode, or one that is not a string (a list would not even hash), is a calls problem:
        # check_call rejects it against the schema's enum, so the fields check skips it.
        required = NOTE_FIELDS.get(mode, ()) if isinstance(mode, str) else ()
        for field in required:
            value = call.args.get(field)
            if not isinstance(value, str) or not value.strip():
                problems.append(f"{where} in {mode} mode needs a non-empty {field}")
        # No write_note list parameter has minItems in the current schema, so this branch never fires today;
        # it stays so that a schema that adds one is checked without a code change.
        for field, spec in properties.items():
            least = spec.get("minItems")
            if spec.get("type") == "array" and least and field in call.args:
                value = call.args[field]
                if not isinstance(value, list) or len(value) < least:
                    problems.append(f"{where} needs at least {least} item(s) in {field}")
    if scenario == "no_call" and calls:
        problems.append(f"fields: no_call scenario but the turn made {len(calls)} tool call(s)")
    if scenario == "clarify":
        if status != CLARIFICATION:
            problems.append(f"fields: clarify scenario but the turn ended as {status}")
        if not any(c.name == ASK for c in calls):
            problems.append(f"fields: clarify scenario but no {ASK} call")
    return problems


def _note_properties() -> dict:
    for entry in schemas.TOOLS:
        if entry["function"]["name"] == NOTE:
            return entry["function"]["parameters"]["properties"]
    return {}

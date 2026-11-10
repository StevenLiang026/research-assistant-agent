"""Argument validation for the seven tools: shape first, meaning second.

Module 02. The executor calls check_call on every tool call of a step before any call of that step runs; a
non-empty answer makes the step an invalid_args issue and its messages go back to the model in the repair
request. tests/test_validation.py is the specification; the lecture is docs/modules/02-Schema与校验.md.
"""

from __future__ import annotations

from jsonschema import Draft202012Validator

from .normalize import normalize_arxiv_id, normalize_date
from .tools import schemas

DATE_PARAMS = ("date_from", "date_to", "since")


def _schema_of(tool: str) -> dict:
    """The full parameter schema of a known tool (the "parameters" of its entry in schemas.TOOLS)."""
    for entry in schemas.TOOLS:
        if entry["function"]["name"] == tool:
            return entry["function"]["parameters"]
    raise KeyError(tool)


def check_call(tool: str, args: dict, seen_ids: list[str], today: str) -> list[str]:
    """Why the call must be refused, sorted, one message per problem; [] when the call may run.

    Every message starts with the parameter it concerns (or "arguments" for a problem with the whole object)
    followed by ": ". Pure: args is never changed, and the same inputs always give the same list.

    1. tool not in schemas.TOOL_NAMES: ["tool: unknown tool <tool>"], nothing else.
    2. Shape: args is validated against the tool's full schema (the "parameters" of its entry in
       schemas.TOOLS) with jsonschema's Draft202012Validator: required keys, types, enums, minimum and
       maximum, minLength and maxLength, uniqueItems, no additional properties. One message per schema
       error, "<first path element or arguments>: <e.message>". When there is any shape error, only the
       shape errors are returned: the meaning of a malformed call is not examined.
    3. Meaning, only when the shape is fine:
       - an "arxiv_id" must already be canonical, normalize_arxiv_id(value) == value, otherwise
         "arxiv_id: expected the canonical form YYMM.NNNNN, got <value>";
       - for write_note the id must also be in seen_ids (the ids that appeared in this turn's tool outputs,
         sorted, canonical), otherwise "arxiv_id: <value> did not appear in any tool output this turn";
         get_paper and fetch_fulltext have no such rule, which is how an id the user typed enters the turn;
       - "date_from", "date_to" and "since" must be canonical, normalize_date(value) == value, otherwise
         "<name>: expected the canonical form YYYY-MM-DD, got <value>"; a canonical date after today is
         "<name>: <value> is after today <today>"; when date_from and date_to are both canonical and
         date_from > date_to: "date_from: <date_from> is after date_to <date_to>".

    today is a canonical ISO date given by the caller (the loop's today), never read from the clock.
    """
    if tool not in schemas.TOOL_NAMES:
        return [f"tool: unknown tool {tool}"]
    errors = []
    for error in Draft202012Validator(_schema_of(tool)).iter_errors(args):
        path = list(error.absolute_path)
        name = path[0] if path else "arguments"
        errors.append(f"{name}: {error.message}")
    if errors:
        return sorted(errors)
    if "arxiv_id" in args:
        value = args["arxiv_id"]
        if normalize_arxiv_id(value) != value:
            errors.append(f"arxiv_id: expected the canonical form YYMM.NNNNN, got {value}")
        elif tool == "write_note" and value not in seen_ids:
            errors.append(f"arxiv_id: {value} did not appear in any tool output this turn")
    for name in DATE_PARAMS:
        if name in args:
            value = args[name]
            if normalize_date(value) != value:
                errors.append(f"{name}: expected the canonical form YYYY-MM-DD, got {value}")
            elif value > today:
                errors.append(f"{name}: {value} is after today {today}")
    if "date_from" in args and "date_to" in args:
        start, end = args["date_from"], args["date_to"]
        if normalize_date(start) == start and normalize_date(end) == end and start > end:
            errors.append(f"date_from: {start} is after date_to {end}")
    return sorted(errors)

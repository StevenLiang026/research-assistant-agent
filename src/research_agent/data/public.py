"""Public function-calling datasets, each in its own encoding, converted to one sample format.

A sample is {"messages", "tools", "source", "source_id"}: messages in API format (agent.types: system, user,
assistant and tool roles only; assistant tool calls carry their arguments as a JSON object string and an id
unique within the sample; every tool message answers, by tool_call_id, a call of the step before it), tools
as OpenAI function schemas, source one of SOURCES and source_id the row index (or config/index) the sample
came from. research_agent.train.samples renders such a sample with the model's own chat template, so the
source's system prompt, which carried the tool list and a format instruction in the source's own notation,
is dropped: the tools field and the template replace it.

One pure converter per source turns one record into (sample, None) or (None, reason); convert_many runs a
converter over records and counts the reasons. Records are converted as they come, apart from these
normalisations, each measured on the Hub revisions of 2026-10-09: consecutive assistant turns of a source
become one assistant message (glaive-v2 says "Sure, let me check." and calls in the next turn); trailing
user and tool messages are cut (they give no training step); whitespace around texts is stripped. A record
is refused, never patched, when its tool list is missing, a call cannot be parsed, a call names a tool not in
the list, the tool results do not answer the calls one to one, a result has no call, or a user or assistant
turn is empty (an assistant turn with neither text nor a call would teach the model to answer with the
terminator alone). Dedup, splitting and leakage audits are the owner's module: a pool may hold exact
duplicates.

The hermes converter parses the dataset's own <tool_call> notation: bodies that are Python reprs rather
than JSON, bodies wrapped in literal backslash-n pairs, and rows that keep the tool name inside the
arguments object. The owner's parse_tool_calls, which reads the model's output at inference, is a separate
module written without consulting this file.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable

SOURCES = ["toolace", "glaive_v2", "hermes", "glaive_zh", "deepexi"]
ROLES = ["system", "user", "assistant", "tool"]

# Reasons a record gives no sample (the keys of convert_many's Counter).
BAD_RECORD = "record has an unexpected shape"
NO_TOOLS = "tool list missing or unparseable"
BAD_CALL = "tool call unparseable"
UNKNOWN_TOOL = "call names a tool not in the list"
BAD_ARGUMENTS = "arguments are not one JSON object"
RESULTS_MISMATCH = "tool results do not answer the calls one to one"
ORPHAN_RESULT = "tool result without a call"
UNANSWERED_CALL = "call without a result before the next message"
NO_ASSISTANT = "no assistant message"
NOT_USER_FIRST = "first message is not from the user"
EMPTY_TURN = "a user or assistant turn is empty"
INVALID_SAMPLE = "sample violates the format"

Converted = tuple[dict | None, str | None]  # (sample, None) or (None, reason)
Converter = Callable[[dict, str], Converted]  # converter(record, source_id)
Call = tuple[str, dict]  # (tool name, arguments)

_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TOOL_RESPONSE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)
_TOOLS_BLOCK = re.compile(r"<tools>(.*?)</tools>", re.DOTALL)
_GLAIVE_SEGMENT = re.compile(r"^(USER|ASSISTANT|FUNCTION RESPONSE): ?", re.MULTILINE)
_GLAIVE_END = "<|endoftext|>"
_GLAIVE_CALL = "<functioncall>"
_GLAIVE_QUOTED = re.compile(r"^\{\s*\"name\"\s*:\s*\"([^\"]*)\"\s*,\s*\"arguments\"\s*:\s*'(.*)'\s*\}$",
                            re.DOTALL)
_GLAIVE_NO_TOOLS = "no access to external functions"
_DECODER = json.JSONDecoder()
_TOOLACE_TYPES = {"dict": "object", "int": "integer", "float": "number"}


# ---- the sample format -----------------------------------------------------------------------------------

class _Conversation:
    """Builds the messages of one sample; call ids are call_0, call_1, ... in order of appearance.

    Each method returns None, or the reason the record must be refused. Calls are pending until a tool
    message answers them, in call order; a user or assistant message while calls are pending is refused.
    """

    def __init__(self) -> None:
        self.messages: list[dict] = []
        self.pending: list[str] = []
        self._count = 0

    def user(self, text: str) -> str | None:
        if self.pending:
            return UNANSWERED_CALL
        last = self.messages[-1] if self.messages else None
        if last is not None and last["role"] == "user":
            last["content"] = _join(last["content"], text)
        else:
            self.messages.append({"role": "user", "content": text.strip()})
        return None

    def assistant(self, content: str, calls: list[Call]) -> str | None:
        last = self.messages[-1] if self.messages else None
        if last is not None and last["role"] == "assistant":  # consecutive turns of the source: one message
            message = last
            message["content"] = _join(message["content"], content)
        elif self.pending:
            return UNANSWERED_CALL
        else:
            message = {"role": "assistant", "content": content.strip()}
            self.messages.append(message)
        for name, arguments in calls:
            try:
                raw = json.dumps(arguments, ensure_ascii=False)
            except (TypeError, ValueError):
                return BAD_CALL
            call_id = f"call_{self._count}"
            self._count += 1
            message.setdefault("tool_calls", []).append(
                {"id": call_id, "type": "function", "function": {"name": name, "arguments": raw}})
            self.pending.append(call_id)
        return None

    def tool(self, content: str) -> str | None:
        if not self.pending:
            return ORPHAN_RESULT
        self.messages.append({"role": "tool", "tool_call_id": self.pending.pop(0),
                              "content": content.strip()})
        return None

    def finish(self, source: str, source_id: str, tools: list[dict]) -> Converted:
        """The sample, with trailing user and tool messages cut; refused without an assistant message or
        with an empty user or assistant turn."""
        while self.messages and self.messages[-1]["role"] != "assistant":
            self.messages.pop()
        if not self.messages:
            return None, NO_ASSISTANT
        if self.messages[0]["role"] != "user":
            return None, NOT_USER_FIRST
        if any(_is_empty_turn(m) for m in self.messages):
            return None, EMPTY_TURN
        return {"messages": self.messages, "tools": tools, "source": source, "source_id": source_id}, None


def _join(first: str, second: str) -> str:
    first, second = first.strip(), second.strip()
    return f"{first}\n{second}" if first and second else first or second


def _is_empty_turn(message: dict) -> bool:
    """A user message without text, or an assistant message with neither text nor a tool call."""
    if message["role"] == "user":
        return not message["content"].strip()
    return message["role"] == "assistant" and not message["content"].strip() and not message.get("tool_calls")


def check_sample(sample: object) -> list[str]:
    """Why sample is not in the format, as messages; [] when it is.

    Checked: the four keys; roles; contents are strings; user messages have text and assistant messages
    text or a call; call ids unique within the sample and arguments a JSON object string; every tool
    message answers, once, a call of the assistant message right before its run of tool messages; an
    assistant message exists; the first message is from the user, after an optional system message; tools
    is a list of OpenAI function schemas.
    """
    errors: list[str] = []
    if not isinstance(sample, dict):
        return ["sample must be a dict"]
    messages, tools = sample.get("messages"), sample.get("tools")
    if not isinstance(sample.get("source"), str) or not isinstance(sample.get("source_id"), str):
        errors.append("source and source_id must be strings")
    if not isinstance(messages, list) or not messages:
        return errors + ["messages must be a non-empty list"]
    ids: set[str] = set()
    pending: set[str] = set()
    for i, m in enumerate(messages):
        role = m.get("role") if isinstance(m, dict) else None
        if role not in ROLES:
            errors.append(f"message {i}: unknown role {role!r}")
            continue
        content = m.get("content")
        text = content.strip() if isinstance(content, str) else None
        if text is None:
            errors.append(f"message {i}: content must be a string")
        elif role == "user" and not text:
            errors.append(f"message {i}: user message is empty")
        if role == "tool":
            call_id = m.get("tool_call_id")
            if call_id not in pending:
                errors.append(f"message {i}: tool_call_id {call_id!r} answers no call of the step before it")
            else:
                pending.discard(call_id)
            continue
        if pending:
            errors.append(f"message {i}: calls {sorted(pending)} were not answered")
        pending = set()
        if role != "assistant":
            continue
        calls = m.get("tool_calls") or []
        if not isinstance(calls, list):
            errors.append(f"message {i}: tool_calls must be a list")
            calls = []
        for j, call in enumerate(calls):
            fn = call.get("function") if isinstance(call, dict) else None
            call_id = call.get("id") if isinstance(call, dict) else None
            if not isinstance(call_id, str) or not call_id or call_id in ids:
                errors.append(f"message {i}: call {j} id {call_id!r} is missing or not unique")
            else:
                ids.add(call_id)
                pending.add(call_id)
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
                errors.append(f"message {i}: call {j} has no tool name")
            if not isinstance(fn, dict) or not _is_object_json(fn.get("arguments")):
                errors.append(f"message {i}: call {j} arguments are not a JSON object string")
        if text == "" and not calls:
            errors.append(f"message {i}: assistant message has neither text nor a tool call")
    if not any(isinstance(m, dict) and m.get("role") == "assistant" for m in messages):
        errors.append("no assistant message")
    start = 1 if isinstance(messages[0], dict) and messages[0].get("role") == "system" else 0
    first = messages[start] if start < len(messages) else None
    if not isinstance(first, dict) or first.get("role") != "user":
        errors.append("the first message (after an optional system message) must be from the user")
    if not isinstance(tools, list):
        errors.append("tools must be a list")
    else:
        for j, tool in enumerate(tools):
            if _function_schema(tool) != tool:
                errors.append(f"tool {j} is not an OpenAI function schema")
    return errors


def _is_object_json(text: object) -> bool:
    if not isinstance(text, str):
        return False
    try:
        return isinstance(json.loads(text), dict)
    except ValueError:
        return False


def has_tool_call(sample: dict) -> bool:
    return any(m.get("tool_calls") for m in sample["messages"] if m.get("role") == "assistant")


def convert_many(records: Iterable[dict], converter: Converter, *,
                 ids: Iterable[str] | None = None) -> tuple[list[dict], Counter]:
    """converter over every record: the samples in record order, and how many records each reason refused.

    ids are the source_ids, the row indices as strings by default. A sample the converter returns that fails
    check_sample is counted under INVALID_SAMPLE (a converter bug, not a data problem).
    """
    samples: list[dict] = []
    reasons: Counter = Counter()
    pairs = ((str(i), r) for i, r in enumerate(records)) if ids is None else zip(ids, records, strict=True)
    for source_id, record in pairs:
        sample, reason = converter(record, source_id)
        if sample is None:
            reasons[reason or BAD_RECORD] += 1
        elif check_sample(sample):
            reasons[INVALID_SAMPLE] += 1
        else:
            samples.append(sample)
    return samples, reasons


# ---- shared parsing --------------------------------------------------------------------------------------

def _literal(text: str) -> tuple[object, bool]:
    """A JSON value or a Python literal and True; (None, False) when text is neither."""
    try:
        return json.loads(text), True
    except ValueError:
        pass
    try:
        return ast.literal_eval(text), True
    except (ValueError, SyntaxError, TypeError, RecursionError):
        return None, False


def _json_object(text: str) -> dict | None:
    try:
        value = json.loads(text)
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _json_objects_at_line_starts(text: str) -> list[dict]:
    """The JSON objects that start at the beginning of a line, in order: how glaive and Deepexi list tools."""
    objects, pos = [], 0
    while True:
        pos = text.find("{", pos)
        if pos < 0:
            return objects
        if pos == 0 or text[pos - 1] == "\n":
            try:
                value, end = _DECODER.raw_decode(text, pos)
            except ValueError:
                pos += 1
                continue
            if isinstance(value, dict):
                objects.append(value)
            pos = end
        else:
            pos += 1


def _first_json_list(text: str) -> list | None:
    """The first JSON list holding an object that parses from a "[" in text, or None (ToolACE's tool list)."""
    pos = text.find("[")
    while pos >= 0:
        try:
            value, _ = _DECODER.raw_decode(text, pos)
        except ValueError:
            value = None
        if isinstance(value, list) and any(isinstance(item, dict) for item in value):
            return value
        pos = text.find("[", pos + 1)
    return None


def _tools_block(system: str) -> list | None:
    """The first <tools> block of a hermes system prompt that holds a JSON list; None when there is none."""
    for block in _TOOLS_BLOCK.finditer(system):
        try:
            value = json.loads(block.group(1))
        except ValueError:
            continue
        if isinstance(value, list):
            return value
    return None


def _function_schema(obj: object) -> dict | None:
    """obj as {"type": "function", "function": {...}}, from that form or a bare function dict; else None."""
    if not isinstance(obj, dict):
        return None
    fn = obj["function"] if isinstance(obj.get("function"), dict) else obj
    if not isinstance(fn.get("name"), str) or not fn["name"]:
        return None
    if set(obj) == {"type", "function"} and obj.get("type") == "function" and fn is obj["function"]:
        return obj
    return {"type": "function", "function": fn}


def _tool_names(tools: list[dict]) -> set[str]:
    return {t["function"]["name"] for t in tools}


def _strip_literal_newlines(text: str) -> str:
    """text without surrounding whitespace and the literal backslash-n pairs some hermes rows wrap it in."""
    text = text.strip()
    while text.startswith("\\n"):
        text = text[2:].lstrip()
    while text.endswith("\\n"):
        text = text[:-2].rstrip()
    return text


def _name_and_arguments(value: object) -> Call | None:
    """(name, arguments) from a {"name", "arguments"} object; else None.

    The arguments must be an object, given as such or as JSON text; a missing or null value is {}.
    """
    if not isinstance(value, dict) or not isinstance(value.get("name"), str) or not value["name"]:
        return None
    arguments = value.get("arguments")
    if arguments is None:
        arguments = {}
    elif isinstance(arguments, str):
        arguments = _json_object(arguments)
    return (value["name"], arguments) if isinstance(arguments, dict) else None


# ---- ToolACE: "[f(a=1, b='x'), g()]" -----------------------------------------------------------------------

def parse_call_string(text: str) -> list[Call] | None:
    """The calls in a ToolACE call string, or None when it does not have that shape.

    The string is "[name(key=value, ...), name2(...)]". Names are free text (spaces, dots, apostrophes,
    parentheses and commas occur), so a comma separates two calls only after a ")", and each call's argument
    list is the one that starts at the last top-level "(" and ends at the call's final ")". Values are JSON
    (true, null, nested objects) or Python literals (True, None, single quotes); a value that is neither, a
    positional argument or any other notation gives None.
    """
    text = text.strip()
    if not (text.startswith("[") and text.endswith("]")):
        return None
    calls: list[Call] = []
    for segment in _split_calls(text[1:-1]):
        name, arguments_text = _split_call(segment)
        if name is None:
            return None
        arguments: dict = {}
        for part in _split_top(arguments_text, quotes_at_top=True):
            key, sep, value_text = part.partition("=")
            key = key.strip()
            if not sep or not key:
                return None
            value, ok = _literal(value_text.strip())
            if not ok:
                return None
            arguments[key] = value
        calls.append((name, arguments))
    return calls or None


def _split_calls(text: str) -> list[str]:
    """A call list split at the commas between calls.

    A piece that does not end with ")" is the part of a tool name before a comma ("Get Walk, Transit, and
    Bike Score": 30 distinct ToolACE names hold one), so it is joined to the next piece.
    """
    calls: list[str] = []
    for piece in _split_top(text, quotes_at_top=False):
        if calls and not calls[-1].rstrip().endswith(")"):
            calls[-1] = f"{calls[-1]},{piece}"
        else:
            calls.append(piece)
    return calls


def _split_top(text: str, *, quotes_at_top: bool) -> list[str]:
    """text split at the commas outside brackets and quotes; blank pieces dropped.

    Quotes delimit strings inside brackets always, and at bracket depth 0 only when quotes_at_top: a call
    list is split with it off, because an apostrophe at depth 0 belongs to a name ("Get Agent's Listings"),
    while an argument list is split with it on, because there a quote at depth 0 starts a string value.
    """
    parts, depth, quote, start, i = [], 0, None, 0, 0
    while i < len(text):
        c = text[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = None
        elif c in "\"'" and (depth > 0 or quotes_at_top):
            quote = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
        i += 1
    parts.append(text[start:])
    return [p for p in parts if p.strip()]


def _split_call(segment: str) -> tuple[str | None, str]:
    """(name, argument text) of one "name(arguments)" segment; (None, "") when it has no such shape."""
    segment = segment.strip()
    depth, quote, open_at, close_at, i = 0, None, None, None, 0
    while i < len(segment):
        c = segment[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == quote:
                quote = None
        elif c in "\"'" and depth > 0:
            quote = c
        elif c in "([{":
            if depth == 0 and c == "(":
                open_at = i
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0 and c == ")":
                close_at = i
        i += 1
    if depth != 0 or quote or open_at is None or close_at != len(segment) - 1:
        return None, ""
    name = segment[:open_at].strip()
    return (name, segment[open_at + 1:close_at]) if name else (None, "")


def _toolace_schema(tool: dict) -> dict:
    """A ToolACE function as an OpenAI function schema: its own type names mapped to JSON Schema's."""
    parameters = tool.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        parameters = {"type": "object", "properties": {}}
    parameters = _map_types({"type": "object", **parameters})
    function = {"name": tool["name"], "description": tool.get("description") or "", "parameters": parameters}
    return {"type": "function", "function": function}


def _map_types(node: object) -> object:
    """node with each "type" that is a ToolACE type name (dict, int, float) mapped to JSON Schema's."""
    if isinstance(node, dict):
        return {k: _TOOLACE_TYPES.get(v, v) if k == "type" and isinstance(v, str) else _map_types(v)
                for k, v in node.items()}
    if isinstance(node, list):
        return [_map_types(v) for v in node]
    return node


def _toolace_named_schema(tool: dict) -> dict:
    """A tool in ToolACE's other notation as a function schema.

    That notation is {"tool_name", "definition", "arguments": [{"parameter_name", "type", "description",
    ...}], "required", "results"}: the arguments become the properties (a nested "properties" list the same
    way), required is kept when it is a list, and the results are not part of a function schema.
    """
    parameters: dict = {"type": "object", "properties": _named_properties(tool.get("arguments"))}
    if isinstance(tool.get("required"), list):
        parameters["required"] = [name for name in tool["required"] if isinstance(name, str)]
    function = {"name": tool["tool_name"], "description": tool.get("definition") or "",
                "parameters": _map_types(parameters)}
    return {"type": "function", "function": function}


def _named_properties(arguments: object) -> dict:
    """A list of {"parameter_name", ...} objects as a JSON Schema properties map; else {}."""
    properties: dict = {}
    for argument in arguments if isinstance(arguments, list) else []:
        if not isinstance(argument, dict) or not isinstance(argument.get("parameter_name"), str):
            continue
        prop = {k: v for k, v in argument.items() if k != "parameter_name"}
        if isinstance(prop.get("properties"), list):
            prop["properties"] = _named_properties(prop["properties"])
        properties[argument["parameter_name"]] = prop
    return properties


def _toolace_tools(system: str) -> list[dict]:
    """The function schemas of a ToolACE system prompt: its JSON list of {"name", ...} objects, or, when
    there is none, the {"tool_name", ...} objects listed one per line; [] when neither is there."""
    listed = _first_json_list(system) or []
    tools = [_toolace_schema(t) for t in listed if isinstance(t, dict) and isinstance(t.get("name"), str)]
    if tools:
        return tools
    return [_toolace_named_schema(t) for t in _json_objects_at_line_starts(system)
            if isinstance(t.get("tool_name"), str) and t["tool_name"]]


def convert_toolace(record: dict, source_id: str) -> Converted:
    """Team-ACE/ToolACE: {"system": prompt holding the tool list, "conversations": [{"from", "value"}]}.

    Roles are user, assistant and tool. The tool list is the prompt's JSON list of {"name", "description",
    "parameters"} objects. Rows 800-1299 of revision 6bda777 rephrase the prompt and use notations of their
    own for the tools and the calls: the 70 rows that list the tools as one {"tool_name", "definition",
    "arguments"} object per line are read, and 39 of them convert (no-call answers, the "None of the
    functions provided can be used" examples the plan wants) while the other 31 answer with calls in those
    notations and are refused as BAD_CALL; the remaining 430 rows (HTML tables, XML, Markdown lists, LaTeX)
    are refused as NO_TOOLS, as are the 248 rows elsewhere whose list is "[]". An assistant value that opens
    with a bracket is a call string and becomes tool calls; any other assistant value is text. A tool value
    is a JSON list with one {"name", "results"} object per call of the step, in call order; each object
    becomes one tool message, as JSON.
    """
    system, conversations = record.get("system"), record.get("conversations")
    if not isinstance(system, str) or not isinstance(conversations, list):
        return None, BAD_RECORD
    tools = _toolace_tools(system)
    if not tools:
        return None, NO_TOOLS
    names = _tool_names(tools)
    conversation = _Conversation()
    for message in conversations:
        role, value = message.get("from"), message.get("value")
        if not isinstance(value, str):
            return None, BAD_RECORD
        if role == "user":
            refused = conversation.user(value)
        elif role == "assistant":
            text = value.strip()
            if text[:1] in ("[", "{"):  # a call string, or one in a notation the dataset's noise rows use
                calls = parse_call_string(text)
                if calls is None:
                    return None, BAD_CALL
                if any(name not in names for name, _ in calls):
                    return None, UNKNOWN_TOOL
                refused = conversation.assistant("", calls)
            else:
                refused = conversation.assistant(text, [])
        elif role == "tool":
            try:
                results = json.loads(value)
            except ValueError:
                return None, RESULTS_MISMATCH
            if not isinstance(results, list) or len(results) != len(conversation.pending):
                return None, RESULTS_MISMATCH if conversation.pending else ORPHAN_RESULT
            refused = None
            for result in results:
                refused = refused or conversation.tool(json.dumps(result, ensure_ascii=False))
        else:
            return None, BAD_RECORD
        if refused:
            return None, refused
    return conversation.finish("toolace", source_id, tools)


# ---- glaive-function-calling-v2: "USER: ... ASSISTANT: <functioncall> {...} <|endoftext|> FUNCTION RESPONSE:

def _glaive_call(payload: str) -> Call | None:
    """The call in a <functioncall> payload: a JSON object, or {"name": "...", "arguments": '<json>'}."""
    call = _name_and_arguments(_json_object(payload))
    if call is not None:
        return call
    match = _GLAIVE_QUOTED.match(payload)
    if match is None:
        return None
    inner = match.group(2)
    # Some rows escape apostrophes as \' inside the quoted JSON, which is a Python escape, not a JSON one.
    arguments = _json_object(inner) or _json_object(inner.replace("\\'", "'"))
    return (match.group(1), arguments) if arguments is not None else None


def convert_glaive_v2(record: dict, source_id: str) -> Converted:
    """glaiveai/glaive-function-calling-v2: {"system": "SYSTEM: ...\\n{json}\\n\\n{json}\\n", "chat": "..."}.

    The chat is a sequence of "USER:", "ASSISTANT:" and "FUNCTION RESPONSE:" segments, each ended by
    <|endoftext|> after the assistant; an assistant segment may be "<functioncall> {payload}", possibly
    after a line of text. A system prompt "with no access to external functions" gives an empty tool list.
    """
    system, chat = record.get("system"), record.get("chat")
    if not isinstance(system, str) or not isinstance(chat, str):
        return None, BAD_RECORD
    tools = [schema for obj in _json_objects_at_line_starts(system) if (schema := _function_schema(obj))]
    if not tools and _GLAIVE_NO_TOOLS not in system:
        return None, NO_TOOLS
    names = _tool_names(tools)
    parts = _GLAIVE_SEGMENT.split(chat)
    if parts[0].strip():
        return None, BAD_RECORD
    conversation = _Conversation()
    for marker, text in zip(parts[1::2], parts[2::2], strict=True):
        text = text.replace(_GLAIVE_END, "").strip()
        if marker == "USER":
            refused = conversation.user(text)
        elif marker == "FUNCTION RESPONSE":
            refused = conversation.tool(text)
        else:
            content, found, payload = text.partition(_GLAIVE_CALL)
            calls: list[Call] = []
            if found:
                call = _glaive_call(payload.strip())
                if call is None:
                    return None, BAD_CALL
                if call[0] not in names:
                    return None, UNKNOWN_TOOL
                calls = [call]
            refused = conversation.assistant(content, calls)
        if refused:
            return None, refused
    return conversation.finish("glaive_v2", source_id, tools)


# ---- hermes-function-calling-v1: <tool_call> and <tool_response> XML tags -------------------------------

def _hermes_tools(record: dict) -> list[dict] | None:
    """The record's tools: its "tools" field (JSON text; "null" means none), else the system <tools> block.

    [] is a plain chat (no field or "null", and no readable block); None means the field is unreadable.
    """
    raw = record.get("tools")
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            return None
    else:
        value = raw
    if value is None:
        value = _tools_block(_system_value(record) or "")
        if value is None:
            return []
    if not isinstance(value, list):
        return None
    tools = [schema for obj in value if (schema := _function_schema(obj))]
    return tools if tools or not value else None


def _system_value(record: dict) -> str | None:
    for message in record.get("conversations") or []:
        if isinstance(message, dict) and message.get("from") == "system":
            value = message.get("value")
            if isinstance(value, str):
                return value
    return None


def _hermes_call(body: str) -> Call | None:
    """The call in a <tool_call> body: {"name", "arguments"}, or the variant {"arguments": {..., "name"}}.

    The variant (793 of the 1,893 func_calling rows at revision dae3e1d) keeps the tool name as the last key
    of the arguments object; it is lifted out, and the tool list still has to know the name.
    """
    value, ok = _literal(_strip_literal_newlines(body))
    if not ok:
        return None
    if isinstance(value, dict) and "name" not in value and isinstance(value.get("arguments"), dict):
        arguments = value["arguments"]
        if isinstance(arguments.get("name"), str):
            rest = {k: v for k, v in arguments.items() if k != "name"}
            value = {"name": arguments["name"], "arguments": rest}
    return _name_and_arguments(value)


def convert_hermes(record: dict, source_id: str) -> Converted:
    """NousResearch/hermes-function-calling-v1 (func_calling and glaive_func_calling configs).

    Sharegpt roles: system (skipped), human, gpt and tool. A gpt value holds zero or more
    <tool_call>{json}</tool_call> blocks, parallel calls being several blocks; text outside the blocks is
    the content. A tool value holds one <tool_response>...</tool_response> block per call of the step, in
    call order, each becoming one tool message with the block's text. Some rows wrap block bodies in literal
    backslash-n pairs, some bodies are Python reprs and some keep the name inside the arguments; all are read.
    """
    conversations = record.get("conversations")
    if not isinstance(conversations, list):
        return None, BAD_RECORD
    tools = _hermes_tools(record)
    if tools is None:
        return None, NO_TOOLS
    names = _tool_names(tools)
    conversation = _Conversation()
    for message in conversations:
        role, value = message.get("from"), message.get("value")
        if not isinstance(value, str):
            return None, BAD_RECORD
        if role == "system":
            continue
        if role == "human":
            refused = conversation.user(value)
        elif role == "gpt":
            calls = []
            for body in _TOOL_CALL.findall(value):
                call = _hermes_call(body)
                if call is None:
                    return None, BAD_CALL
                if call[0] not in names:
                    return None, UNKNOWN_TOOL
                calls.append(call)
            refused = conversation.assistant(_TOOL_CALL.sub("", value), calls)
        elif role == "tool":
            blocks = _TOOL_RESPONSE.findall(value) or [value]
            if len(blocks) != len(conversation.pending):
                return None, RESULTS_MISMATCH if conversation.pending else ORPHAN_RESULT
            refused = None
            for block in blocks:
                refused = refused or conversation.tool(_strip_literal_newlines(block))
        else:
            return None, BAD_RECORD
        if refused:
            return None, refused
    return conversation.finish("hermes", source_id, tools)


# ---- glaive_toolcall_zh: sharegpt with function_call and observation roles ------------------------------

def convert_glaive_zh(record: dict, source_id: str) -> Converted:
    """llamafactory/glaive_toolcall_zh: {"conversations": [{"from", "value"}], "tools": "<json list>"}.

    Roles are human, gpt, function_call (a {"name", "arguments"} JSON object) and observation (the result
    text of the call before it). "[]" as tools is a plain chat.
    """
    conversations, raw_tools = record.get("conversations"), record.get("tools")
    if not isinstance(conversations, list) or not isinstance(raw_tools, str):
        return None, BAD_RECORD
    try:
        listed = json.loads(raw_tools or "[]")
    except ValueError:
        return None, NO_TOOLS
    if not isinstance(listed, list):
        return None, NO_TOOLS
    tools = [schema for obj in listed if (schema := _function_schema(obj))]
    if listed and not tools:
        return None, NO_TOOLS
    names = _tool_names(tools)
    conversation = _Conversation()
    for message in conversations:
        role, value = message.get("from"), message.get("value")
        if not isinstance(value, str):
            return None, BAD_RECORD
        if role == "human":
            refused = conversation.user(value)
        elif role == "gpt":
            refused = conversation.assistant(value, [])
        elif role == "function_call":
            call = _name_and_arguments(_json_object(value))
            if call is None:
                return None, BAD_CALL
            if call[0] not in names:
                return None, UNKNOWN_TOOL
            refused = conversation.assistant("", [call])
        elif role == "observation":
            refused = conversation.tool(value)
        else:
            return None, BAD_RECORD
        if refused:
            return None, refused
    return conversation.finish("glaive_zh", source_id, tools)


# ---- Deepexi/function-calling-small: one call per row, tools in a custom notation -------------------------

def _deepexi_schema(fn: dict) -> dict:
    """{"function", "description", "arguments": [{"name", "type", "description"}]} as a function schema."""
    properties: dict = {}
    for argument in fn.get("arguments") or []:
        if not isinstance(argument, dict) or not isinstance(argument.get("name"), str):
            continue
        prop = {}
        if isinstance(argument.get("type"), str):
            prop["type"] = argument["type"]
        if isinstance(argument.get("description"), str):
            prop["description"] = argument["description"]
        properties[argument["name"]] = prop
    function = {"name": fn["function"], "description": fn.get("description") or "",
                "parameters": {"type": "object", "properties": properties}}
    return {"type": "function", "function": function}


def _deepexi_arguments(value: object) -> dict | None:
    """The arguments object: null is {}, a list of objects with distinct keys is their union; else None."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        return None
    merged: dict = {}
    for item in value:
        if any(key in merged for key in item):
            return None
        merged.update(item)
    return merged


def convert_deepexi(record: dict, source_id: str) -> Converted:
    """Deepexi/function-calling-small: {"systemPrompt", "userPrompt", "assistantResponse"} (CSV columns).

    The system prompt lists functions as JSON objects, one per line, in the dataset's own notation; the
    user prompt is wrapped in quotes and spaces, which are removed; the response is a JSON object
    {"function", "arguments"} where arguments is a list holding the argument object (sometimes split into
    several objects, sometimes null). Every sample is one user message and one call.
    """
    system, user, response = (record.get(k) for k in ("systemPrompt", "userPrompt", "assistantResponse"))
    if not all(isinstance(v, str) for v in (system, user, response)):
        return None, BAD_RECORD
    tools = [_deepexi_schema(fn) for fn in _json_objects_at_line_starts(system)
             if isinstance(fn.get("function"), str) and fn["function"]]
    if not tools:
        return None, NO_TOOLS
    value = _json_object(response)
    if value is None or not isinstance(value.get("function"), str) or not value["function"]:
        return None, BAD_CALL
    arguments = _deepexi_arguments(value.get("arguments"))
    if arguments is None:
        return None, BAD_ARGUMENTS
    if value["function"] not in _tool_names(tools):
        return None, UNKNOWN_TOOL
    text = user.strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        text = text[1:-1]
    conversation = _Conversation()
    refused = conversation.user(text) or conversation.assistant("", [(value["function"], arguments)])
    if refused:
        return None, refused
    return conversation.finish("deepexi", source_id, tools)


CONVERTERS: dict[str, Converter] = {"toolace": convert_toolace, "glaive_v2": convert_glaive_v2,
                                    "hermes": convert_hermes, "glaive_zh": convert_glaive_zh,
                                    "deepexi": convert_deepexi}

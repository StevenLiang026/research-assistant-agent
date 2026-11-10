"""Tool calls in a Qwen3 completion (docs/modules/接口.md, 本地模型接入时(M3)).

parse_tool_calls(text) reads every <tool_call>...</tool_call> block of the text the local model generated,
in order, and returns one {"name", "arguments"} dict per block; text outside the blocks is ignored (the
model's prose is read separately by agent.models.LocalModel, which keeps the text as the answer only when
there is no call). Qwen3 writes a call as

    <tool_call>
    {"name": "search_arxiv", "arguments": {"query": "speculative decoding"}}
    </tool_call>

Rules, per block:

- The block's JSON is read with a JSON decoder from the first non-blank character after <tool_call>, so braces
  and even a "</tool_call>" inside a string value never end the call early. The JSON value must be followed by
  nothing but whitespace before </tool_call>, or before the end of the text: an unterminated last block
  (generation stopped at a stop string, or ran out of tokens right after the JSON) is parsed when its JSON is
  complete.
- A JSON object with a string "name" gives {"name": name, "arguments": arguments}: arguments is the
  "arguments" object, {} when the key is absent or null, and passed on as it is when it is a string (some
  models write the arguments as JSON text; LocalModel parses text arguments).
- Anything else (JSON that does not parse, is not an object, has no string name, or has arguments that are
  neither an object nor text) gives {"name": <the "name" string the text shows at the top level of its
  object, else "">, "arguments": <the block's text, stripped>}, so the framework records bad_json for it (or
  unknown_tool for an empty name) and can show the model what it wrote; a "name" inside the arguments is
  never taken for the call's. Such a block ends at the next </tool_call>, else at the end of the text.

Never raises on any string; text without a block gives [].
"""

from __future__ import annotations

import json
import re

OPEN = "<tool_call>"
CLOSE = "</tool_call>"
_DECODER = json.JSONDecoder()
_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
_NAME_VALUE = re.compile(r'\s*:\s*"((?:[^"\\]|\\.)*)"')


def parse_tool_calls(text: str) -> list[dict]:
    """The calls of text's <tool_call> blocks, in order; see the module docstring."""
    calls: list[dict] = []
    position = 0
    while (start := text.find(OPEN, position)) >= 0:
        body = start + len(OPEN)
        decoded = _decode(text, body)
        if decoded is not None:
            value, position = decoded
            call = _call(value)
            if call is not None:
                calls.append(call)
                continue
        close = text.find(CLOSE, body)
        end = len(text) if close < 0 else close
        raw = text[body:end].strip()
        calls.append({"name": _best_name(raw), "arguments": raw})
        position = len(text) if close < 0 else close + len(CLOSE)
    return calls


def _decode(text: str, body: int) -> tuple[object, int] | None:
    """The JSON value starting at the first non-blank character from body and where reading goes on: after
    the closing tag, or at the end of the text; None when no JSON value stands there alone."""
    begin = _skip_blank(text, body)
    try:
        value, end = _DECODER.raw_decode(text, begin)
    except json.JSONDecodeError:
        return None
    after = _skip_blank(text, end)
    if after == len(text):
        return value, after
    if text.startswith(CLOSE, after):
        return value, after + len(CLOSE)
    return None


def _skip_blank(text: str, index: int) -> int:
    while index < len(text) and text[index].isspace():
        index += 1
    return index


def _call(value: object) -> dict | None:
    """{"name", "arguments"} for a well-formed call object, None for anything else."""
    if not isinstance(value, dict) or not isinstance(value.get("name"), str):
        return None
    arguments = value.get("arguments")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, (dict, str)):
        return None
    return {"name": value["name"], "arguments": arguments}


def _best_name(raw: str) -> str:
    """The string value of a "name" key at the top level of the raw text's object (one brace deep, read
    outside strings), unescaped; "" when there is none. A "name" inside the arguments is not the call's."""
    depth, index = 0, 0
    while index < len(raw):
        char = raw[index]
        if char == '"':
            string = _STRING.match(raw, index)
            if string is None:  # an unterminated string: nothing after it can be read reliably
                return ""
            index = string.end()
            if depth == 1 and string.group() == '"name"':
                value = _NAME_VALUE.match(raw, index)
                if value is not None:
                    return _unescape(value.group(1))
            continue
        if char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
        index += 1
    return ""


def _unescape(text: str) -> str:
    try:
        return json.loads(f'"{text}"')
    except json.JSONDecodeError:
        return text

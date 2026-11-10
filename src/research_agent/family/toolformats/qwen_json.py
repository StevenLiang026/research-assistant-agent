"""qwen_json: Qwen3's tool calls, one JSON object per block (docs/m7-家族层设计.md §4).

    <tool_call>
    {"name": "search_arxiv", "arguments": {"query": "speculative decoding"}}
    </tool_call>

parse is research_agent.toolcalls.parse_tool_calls itself, so a broken block gives what it gives today
(the best name and the block's text: the bad_json path); the tools are not needed, JSON carries its own
types.

locate reads the blocks with the logic of agent/route.py's _blocks, _read_call and _object_members, copied
here unchanged: phase 1 only adds files, so route.py keeps its copy until WP-D has it call this one (§4),
and tests/family/test_toolformats.py shows that the two give the same spans on every case of the parser's
and the router's tests, on every prefix of each and on seeded corruptions of them.

render writes calls as the Qwen3 chat templates write the calls of an assistant message without text: per
call '<tool_call>\\n{"name": "' + name + '", "arguments": ' + arguments + '}\\n</tool_call>', the name as it
is, the arguments as the template's tojson writes them (json.dumps with ensure_ascii off and the default
separators) or verbatim when they are a string, the blocks separated by "\\n". tests/family/test_golden.py
checks it against the real templates.
"""

from __future__ import annotations

import json

from ...toolcalls import CLOSE, OPEN, parse_tool_calls
from . import CallSpan

_DECODER = json.JSONDecoder()
_JSON_SPACE = " \t\n\r"


class QwenJson:
    """The qwen_json ToolFormat (module docstring)."""

    name = "qwen_json"
    open_tag = OPEN

    def render(self, calls: list[dict]) -> str:
        return "\n".join(_render(call) for call in calls)

    def parse(self, text: str, tools: list[dict] | None) -> list[dict]:
        return parse_tool_calls(text)

    def locate(self, text: str) -> list[CallSpan]:
        """The text's blocks in the parser's order: after a call that parsed, reading goes on after it (so a
        <tool_call> inside one of its strings is not a block), after one that did not, at its </tool_call>."""
        spans = []
        position = 0
        while (start := text.find(OPEN, position)) >= 0:
            body = start + len(OPEN)
            read = _read_call(text, body)
            if read is not None:
                span, position = read
                spans.append(span)
                continue
            close = text.find(CLOSE, body)
            position = len(text) if close < 0 else close + len(CLOSE)
            spans.append(CallSpan(parsed=False))
        return spans


QWEN_JSON = QwenJson()


def _render(call: dict) -> str:
    arguments = call["arguments"]
    text = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    return f'{OPEN}\n{{"name": "{call["name"]}", "arguments": {text}}}\n{CLOSE}'


def _read_call(text: str, body: int) -> tuple[CallSpan, int] | None:
    """The block starting at body and where reading goes on, when its JSON is a call object followed by
    nothing but blanks before </tool_call> or the end of the text (toolcalls' rule); None otherwise."""
    begin = body
    while begin < len(text) and text[begin].isspace():
        begin += 1
    read = _object_members(text, begin)
    if read is None:
        return None
    members, end = read
    after = end
    while after < len(text) and text[after].isspace():
        after += 1
    if after == len(text):
        position = after
    elif text.startswith(CLOSE, after):
        position = after + len(CLOSE)
    else:
        return None
    name = members.get("name")
    if name is None or not isinstance(name[2], str):
        return None
    arguments = members.get("arguments")
    value = None if arguments is None else arguments[2]
    if value is not None and not isinstance(value, dict | str):
        return None
    values: tuple[int, ...] = ()
    if isinstance(value, dict):
        inner = _object_members(text, arguments[0])
        if inner is not None:
            values = tuple(start + 1 if text[start] == '"' else start for start, _, _ in inner[0].values())
    return CallSpan(parsed=True, name=(name[0] + 1, name[1] - 1), values=values), position


def _object_members(text: str, begin: int) -> tuple[dict[str, tuple[int, int, object]], int] | None:
    """The JSON object at begin as {key: (value start, value end, value)} and the object's end; None when no
    JSON object starts there. Of a repeated key the last one counts, as for the JSON decoder."""
    try:
        whole, end = _DECODER.raw_decode(text, begin)
    except json.JSONDecodeError:
        return None
    if not isinstance(whole, dict):
        return None
    members: dict[str, tuple[int, int, object]] = {}
    index = _skip_space(text, begin + 1)
    while text[index] != "}":
        key, index = _DECODER.raw_decode(text, index)
        index = _skip_space(text, _skip_space(text, index) + 1)  # past the colon
        value, value_end = _DECODER.raw_decode(text, index)
        members[key] = (index, value_end, value)
        index = _skip_space(text, value_end)
        if text[index] == ",":
            index = _skip_space(text, index + 1)
    return members, end


def _skip_space(text: str, index: int) -> int:
    while index < len(text) and text[index] in _JSON_SPACE:
        index += 1
    return index

"""qwen_xml: Qwen3-Coder's XML tool calls, the format of Qwen3.5-4B's and granite-4.2-3b's chat templates
(docs/m7-家族层设计.md §4; the strings below are the templates' own, probed on 2026-10-19 and held against
them by tests/family/test_golden.py).

    <tool_call>
    <function=search_arxiv>
    <parameter=query>
    speculative decoding
    </parameter>
    <parameter=max_results>
    5
    </parameter>
    </function>
    </tool_call>

A value sits on lines of its own between <parameter=K> and </parameter>: a string as it is, a number, a
boolean or null as Python prints it (5, 1.5, True, None: the templates' "| string"), an array or an object
as JSON (the templates' "| tojson"). Blocks of one step are joined with one newline (Qwen3.5 writes them so;
Granite writes "\\n" after each block, which its completion keeps before the terminator). A call whose
arguments are text has the text where the parameters go.

parse reads every <tool_call> block in order; a block runs to the first </tool_call> after it, or to the
end of the text (an unterminated last block: generation stopped there), so unlike a JSON string no value
can hold that tag, and a block that is broken cannot swallow the one after it. A block is a call when,
blanks aside, it holds exactly <function=NAME> (NAME non-empty, without blanks or angle brackets), zero or
more <parameter=K> ... </parameter> (K alike; the value is everything up to the first </parameter>, so a
value cannot hold that tag either) and </function>. Its values are typed by the step's tool schemas
(coerce, parameter_schema). Any other block gives the best name (the <function=NAME> it shows, else "")
with its inner text, stripped.

locate gives the span of NAME and the offset of each value's first character: the character after the
newline that follows <parameter=K> when the value starts with one (the value's own text, where the model's
decision begins), else the first character after the tag.
"""

from __future__ import annotations

import json
import re

from . import CallSpan, ReadBlock, coerce, skip_blanks

OPEN, CLOSE = "<tool_call>", "</tool_call>"
FUNCTION_OPEN, FUNCTION_CLOSE = "<function=", "</function>"
PARAMETER_OPEN, PARAMETER_CLOSE = "<parameter=", "</parameter>"
_NAME = re.compile(r"[^\s<>]+")  # what fills <function=...> and <parameter=...>
_BEST_NAME = re.compile(r"<function=([^\s<>]+)>")


class QwenXml:
    """The qwen_xml ToolFormat (module docstring)."""

    name = "qwen_xml"
    open_tag = OPEN

    def render(self, calls: list[dict]) -> str:
        return "\n".join(_render(call) for call in calls)

    def parse(self, text: str, tools: list[dict] | None) -> list[dict]:
        return [block.call(tools, coerce) for block in _blocks(text)]

    def locate(self, text: str) -> list[CallSpan]:
        return [block.span() for block in _blocks(text)]


QWEN_XML = QwenXml()


def value_text(value: object) -> str:
    """A value as the templates write it (module docstring): JSON for an array or an object, Python's text
    for anything else."""
    return json.dumps(value, ensure_ascii=False) if isinstance(value, dict | list) else str(value)


def _render(call: dict) -> str:
    arguments = call["arguments"]
    if isinstance(arguments, str):
        body = arguments + "\n"
    else:
        body = "".join(f"{PARAMETER_OPEN}{key}>\n{value_text(value)}\n{PARAMETER_CLOSE}\n"
                       for key, value in arguments.items())
    return f"{OPEN}\n{FUNCTION_OPEN}{call['name']}>\n{body}{FUNCTION_CLOSE}\n{CLOSE}"


def _blocks(text: str) -> list[ReadBlock]:
    blocks = []
    position = 0
    while (start := text.find(OPEN, position)) >= 0:
        body = start + len(OPEN)
        close = text.find(CLOSE, body)
        end = len(text) if close < 0 else close
        block = _read(text, body, end)
        if block is None:
            raw = text[body:end].strip()
            found = _BEST_NAME.search(raw)
            block = ReadBlock(parsed=False, name=found.group(1) if found else "", raw=raw)
        blocks.append(block)
        position = len(text) if close < 0 else close + len(CLOSE)
    return blocks


def _read(text: str, body: int, end: int) -> ReadBlock | None:
    """The call of the block whose inner text is text[body:end] (module docstring); None when the block is
    not a call."""
    index = skip_blanks(text, body, end)
    name = _tag_value(text, index, FUNCTION_OPEN, end)
    if name is None:
        return None
    name_start = index + len(FUNCTION_OPEN)
    index = name_start + len(name) + 1
    values: list[tuple[str, str, int]] = []
    while True:
        index = skip_blanks(text, index, end)
        if text.startswith(FUNCTION_CLOSE, index) and index + len(FUNCTION_CLOSE) <= end:
            index += len(FUNCTION_CLOSE)
            break
        key = _tag_value(text, index, PARAMETER_OPEN, end)
        if key is None:
            return None
        value_start = index + len(PARAMETER_OPEN) + len(key) + 1
        value_end = text.find(PARAMETER_CLOSE, value_start, end)
        if value_end < 0:
            return None
        raw = text[value_start:value_end]
        values.append((key, raw, value_start + 1 if raw.startswith("\n") else value_start))
        index = value_end + len(PARAMETER_CLOSE)
    if skip_blanks(text, index, end) != end:
        return None
    return ReadBlock(parsed=True, name=name, name_span=(name_start, name_start + len(name)),
                     values=tuple(values))


def _tag_value(text: str, index: int, opening: str, end: int) -> str | None:
    """The NAME of an opening tag "<function=NAME>" or "<parameter=NAME>" standing at index and closed
    before end, or None."""
    if not text.startswith(opening, index):
        return None
    found = _NAME.match(text, index + len(opening), end)
    if found is None or found.end() >= end or text[found.end()] != ">":
        return None
    return found.group()

"""glm_kv: GLM-4.5-style key-value tool calls, the format of Spark-X2.5-4B's chat template
(docs/m7-家族层设计.md §4; the strings below are the template's own, probed on 2026-10-19 and held against it
by tests/family/test_golden.py). One line in the template's output:

    <tool_call>search_arxiv<arg_key>query</arg_key><arg_value>speculative decoding</arg_value>
    <arg_key>max_results</arg_key><arg_value>5</arg_value></tool_call>

Nothing separates the name, the pairs or the blocks of one step: a string value is written as it is and
every other value as JSON (the template's "v if v is string else v | tojson"). A call whose arguments are
text has the text right after the name.

parse reads every <tool_call> block in order; a block runs to the first </tool_call> after it, or to the
end of the text (an unterminated last block: generation stopped there), so unlike a JSON string no value
can hold that tag, and a block that is broken cannot swallow the one after it. A block is a call when it
holds the name (non-empty, without blanks or angle brackets, blanks around it allowed) and then pairs of
<arg_key>K</arg_key> (K alike) and <arg_value>V</arg_value> (V everything up to the first </arg_value>, so
a value cannot hold that tag either), blanks between the tags allowed, and nothing else. Values are typed
by the step's tool schemas (coerce without separating newlines). Any other block gives the best name (the
text before the first "<" when it is a name, else "") with the block's inner text, stripped.

locate gives the span of the name and the offset of each value's first character, the one after
<arg_value>.
"""

from __future__ import annotations

import json
import re

from . import CallSpan, ReadBlock, coerce, skip_blanks

OPEN, CLOSE = "<tool_call>", "</tool_call>"
KEY_OPEN, KEY_CLOSE = "<arg_key>", "</arg_key>"
VALUE_OPEN, VALUE_CLOSE = "<arg_value>", "</arg_value>"
_NAME = re.compile(r"[^\s<>]+")


class GlmKv:
    """The glm_kv ToolFormat (module docstring)."""

    name = "glm_kv"
    open_tag = OPEN

    def render(self, calls: list[dict]) -> str:
        return "".join(_render(call) for call in calls)

    def parse(self, text: str, tools: list[dict] | None) -> list[dict]:
        return [block.call(tools, _typed) for block in _blocks(text)]

    def locate(self, text: str) -> list[CallSpan]:
        return [block.span() for block in _blocks(text)]


GLM_KV = GlmKv()


def value_text(value: object) -> str:
    """A value as the template writes it (module docstring): a string as it is, anything else as JSON."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _typed(raw: str, schema: dict | None) -> object:
    return coerce(raw, schema, separated=False)


def _render(call: dict) -> str:
    arguments = call["arguments"]
    body = arguments if isinstance(arguments, str) else "".join(
        f"{KEY_OPEN}{key}{KEY_CLOSE}{VALUE_OPEN}{value_text(value)}{VALUE_CLOSE}"
        for key, value in arguments.items())
    return f"{OPEN}{call['name']}{body}{CLOSE}"


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
            head = raw.split("<", 1)[0].strip()
            block = ReadBlock(parsed=False, name=head if _NAME.fullmatch(head) else "", raw=raw)
        blocks.append(block)
        position = len(text) if close < 0 else close + len(CLOSE)
    return blocks


def _read(text: str, body: int, end: int) -> ReadBlock | None:
    """The call of the block whose inner text is text[body:end] (module docstring); None when the block is
    not a call."""
    index = skip_blanks(text, body, end)
    found = _NAME.match(text, index, end)
    if found is None:
        return None
    name, name_span = found.group(), (found.start(), found.end())
    index = found.end()
    values: list[tuple[str, str, int]] = []
    while (index := skip_blanks(text, index, end)) < end:
        if not text.startswith(KEY_OPEN, index):
            return None
        key_end = text.find(KEY_CLOSE, index + len(KEY_OPEN), end)
        if key_end < 0:
            return None
        key = text[index + len(KEY_OPEN):key_end].strip()
        if not _NAME.fullmatch(key):
            return None
        index = skip_blanks(text, key_end + len(KEY_CLOSE), end)
        if not text.startswith(VALUE_OPEN, index):
            return None
        value_start = index + len(VALUE_OPEN)
        value_end = text.find(VALUE_CLOSE, value_start, end)
        if value_end < 0:
            return None
        values.append((key, text[value_start:value_end], value_start))
        index = value_end + len(VALUE_CLOSE)
    return ReadBlock(parsed=True, name=name, name_span=name_span, values=tuple(values))

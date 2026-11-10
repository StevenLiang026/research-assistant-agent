"""minicpm_xml: MiniCPM5's XML tool calls, without any wrapper around the function element
(docs/m7-家族层设计.md §4; the strings below are MiniCPM5-2B's template's own, probed on 2026-10-19 and held
against it by tests/family/test_golden.py). One line in the template's output:

    <function name="search_arxiv"><param name="query">speculative decoding</param>
    <param name="max_results">5</param></function>

A value sits between its tags directly, with no separating newlines. A string holding "<", "&" or a
newline is wrapped as <![CDATA[...]]>; every other value is written as Python prints it, the template's
"{{ param_value }}": numbers as they are, True, False and None, and arrays and objects as Python literals
(['cs.CL'], {'a': 1}), not JSON. Blocks of one step are joined with one newline. A call whose arguments
are text has the text between its function tags.

parse reads every <function name="..."> block in order; a block runs to the first </function> after it,
or to the end of the text, so no value can hold that tag (a CDATA section included), and a block that is
broken cannot swallow the one after it. A block is a call when, blanks aside, it holds
<function name="NAME"> (NAME non-empty, without quotes, blanks or angle brackets), zero or more
<param name="K">...</param> (a CDATA value runs to the first "]]>" that "</param>" follows; a plain value
to the first </param>) and </function>; a block cut short is not one. Values are typed by the step's tool
schemas (coerce without separating newlines); where a schema admits an array or an object, or there is no
schema, a value that is no JSON is read as a Python literal when ast.literal_eval reads a list or a
dictionary in it that JSON holds (json_value: no set, tuple, bytes or infinite float inside, no key that is
not a string), since that is how the template writes them; otherwise its text stays, for the executor to
refuse. Any other block gives the best name (the quoted name after <function name=", else "") with the
block's inner text, stripped.

locate gives the span of NAME and the offset of each value's first character (after <![CDATA[ for a CDATA
value).
"""

from __future__ import annotations

import ast
import re

from . import CallSpan, ReadBlock, coerce, json_value, skip_blanks

OPEN, CLOSE = '<function name="', "</function>"
PARAM_OPEN, PARAM_CLOSE = '<param name="', "</param>"
CDATA_OPEN, CDATA_CLOSE = "<![CDATA[", "]]>"
_CDATA_END = re.compile(r"\]\]>\s*</param>")
_NAME = re.compile(r'[^\s"<>]+')
_BEST_NAME = re.compile(r'^([^\s"<>]*)"')
_CONTAINERS = frozenset({"array", "object", "list", "dict"})


class MinicpmXml:
    """The minicpm_xml ToolFormat (module docstring)."""

    name = "minicpm_xml"
    open_tag = OPEN

    def render(self, calls: list[dict]) -> str:
        return "\n".join(_render(call) for call in calls)

    def parse(self, text: str, tools: list[dict] | None) -> list[dict]:
        return [block.call(tools, typed) for block in _blocks(text)]

    def locate(self, text: str) -> list[CallSpan]:
        return [block.span() for block in _blocks(text)]


MINICPM_XML = MinicpmXml()


def value_text(value: object) -> str:
    """A value as the template writes it (module docstring): CDATA around a string that needs it, Python's
    text for everything else."""
    if isinstance(value, str):
        return f"{CDATA_OPEN}{value}{CDATA_CLOSE}" if any(c in value for c in "<&\n") else value
    return str(value)


def typed(raw: str, schema: dict | None) -> object:
    """raw as the type its schema names, a Python-literal list or dictionary read where the schema admits
    one (module docstring)."""
    value = coerce(raw, schema, separated=False)
    if not isinstance(value, str):
        return value
    kinds = (schema or {}).get("type")
    admits = kinds is None or bool(_CONTAINERS & set([kinds] if isinstance(kinds, str) else kinds))
    if not admits:
        return value
    try:
        literal = ast.literal_eval(value.strip())
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return value
    return literal if isinstance(literal, list | dict) and json_value(literal) else value


def _render(call: dict) -> str:
    arguments = call["arguments"]
    body = arguments if isinstance(arguments, str) else "".join(
        f"{PARAM_OPEN}{key}\">{value_text(value)}{PARAM_CLOSE}" for key, value in arguments.items())
    return f"{OPEN}{call['name']}\">{body}{CLOSE}"


def _blocks(text: str) -> list[ReadBlock]:
    blocks = []
    position = 0
    while (start := text.find(OPEN, position)) >= 0:
        body = start + len(OPEN)
        close = text.find(CLOSE, body)
        end = len(text) if close < 0 else close
        block = _read(text, body, end) if close >= 0 else None  # a block cut short is not a call
        if block is None:
            raw = text[body:end].strip()
            found = _BEST_NAME.match(raw)
            block = ReadBlock(parsed=False, name=found.group(1) if found else "", raw=raw)
        blocks.append(block)
        position = len(text) if close < 0 else close + len(CLOSE)
    return blocks


def _read(text: str, body: int, end: int) -> ReadBlock | None:
    """The call of the block whose name starts at body and whose </function> stands at end (module
    docstring); None when the block is not a call."""
    name = _quoted(text, body, end)
    if name is None:
        return None
    index = body + len(name) + 2
    values: list[tuple[str, str, int]] = []
    while (index := skip_blanks(text, index, end)) < end:
        if not text.startswith(PARAM_OPEN, index):
            return None
        key = _quoted(text, index + len(PARAM_OPEN), end)
        if key is None:
            return None
        value_start = index + len(PARAM_OPEN) + len(key) + 2
        if text.startswith(CDATA_OPEN, value_start):
            found = _CDATA_END.search(text, value_start + len(CDATA_OPEN), end)
            if found is None:
                return None
            content_start = value_start + len(CDATA_OPEN)
            values.append((key, text[content_start:found.start()], content_start))
            index = found.end()
        else:
            value_end = text.find(PARAM_CLOSE, value_start, end)
            if value_end < 0:
                return None
            values.append((key, text[value_start:value_end], value_start))
            index = value_end + len(PARAM_CLOSE)
    return ReadBlock(parsed=True, name=name, name_span=(body, body + len(name)), values=tuple(values))


def _quoted(text: str, index: int, end: int) -> str | None:
    """The NAME of a tag whose attribute value starts at index, NAME">: a non-empty name followed by the
    closing quote and the tag's bracket before end; None otherwise."""
    found = _NAME.match(text, index, end)
    if found is None or found.end() + 2 > end or not text.startswith('">', found.end()):
        return None
    return found.group()

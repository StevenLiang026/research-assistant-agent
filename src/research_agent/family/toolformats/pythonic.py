"""pythonic: LFM2.5's Python-style tool calls, every call of a step in one list (docs/m7-家族层设计.md §4;
the strings below are LFM2.5-2.6B's template's own, probed on 2026-10-19 and held against it by
tests/family/test_golden.py). One line in the template's output:

    <|tool_call_start|>[search_arxiv(query='speculative decoding', max_results=5),
    get_paper(arxiv_id='2609.01234')]<|tool_call_end|>

A string is single-quoted with backslash, quote, newline and carriage return escaped (the template's
format_arg_value); an array or an object is JSON (its "tojson"); any other value is written as Python
prints it (5, 1.5, True, None). A call whose arguments are text has the text between its parentheses.

parse reads every <|tool_call_start|> block in order; the text up to <|tool_call_end|> (or to the end of the
text, for an unterminated last block: generation stopped there) must be one Python list expression of
calls, read with ast.parse alone, never run: each element a call of a plain name with keyword arguments
only (no positional argument, no **mapping), each value a literal that ast.literal_eval reads and JSON
holds (a string, a number, True, False, None, a list, a dictionary with string keys: json_value) or, failing
that, JSON text, typed by the step's tool schema (coerce), which is how the template's true, false and null
inside an array or an object come back. Any other block (a syntax error, a comprehension, a bare name, an
attribute, a positional argument, a value that is an expression or a name, a literal JSON cannot hold
such as a set, a tuple, bytes or an infinite float) is one call that cannot be read: the best name (the
first name a parenthesis follows at the start of the list, else "") with the block's inner text,
stripped. Nothing in a block is ever evaluated beyond literals.

locate gives the span of each name and the offset of each value's first character, the one after the
quotes (and the prefix, u or r) of a string, in characters of the answer text: ast gives a line, counted as
Python ends lines (at \\n, \\r\\n or \\r, not at the other characters str.splitlines ends one at), and a
column in UTF-8 bytes, which are converted.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass

from . import CallSpan, broken, coerce, json_value, parameter_schema

OPEN, CLOSE = "<|tool_call_start|>", "<|tool_call_end|>"
_BEST_NAME = re.compile(r"\s*\[?\s*([A-Za-z_]\w*)\s*\(")
_ESCAPES = (("\\", "\\\\"), ("'", "\\'"), ("\n", "\\n"), ("\r", "\\r"))  # the template's replacements
_STRING_OPENING = re.compile(r"[rRuU]?(?:'''|\"\"\"|'|\")")  # what precedes a string's first character
_LINE = re.compile(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+")  # a line, as Python's tokenizer ends one


class Pythonic:
    """The pythonic ToolFormat (module docstring)."""

    name = "pythonic"
    open_tag = OPEN

    def render(self, calls: list[dict]) -> str:
        if not calls:
            return ""
        return f"{OPEN}[{', '.join(_render(call) for call in calls)}]{CLOSE}"

    def parse(self, text: str, tools: list[dict] | None) -> list[dict]:
        calls = []
        for block in _blocks(text):
            if block.calls is None:
                calls.append(broken(block.name, block.raw))
                continue
            for call in block.calls:
                arguments = {}
                for key, value, segment, _ in call.values:
                    if value is _NOT_LITERAL:
                        value = coerce(segment, parameter_schema(tools, call.name, key), separated=False)
                    arguments[key] = value
                calls.append({"name": call.name, "arguments": arguments})
        return calls

    def locate(self, text: str) -> list[CallSpan]:
        spans = []
        for block in _blocks(text):
            if block.calls is None:
                spans.append(CallSpan(parsed=False))
                continue
            for call in block.calls:
                spans.append(CallSpan(parsed=True, name=call.name_span,
                                      values=tuple(offset for _, _, _, offset in call.values)))
        return spans


PYTHONIC = Pythonic()
_NOT_LITERAL = object()  # a value ast.literal_eval does not read: its source text is read instead


def value_text(value: object) -> str:
    """A value as the template writes it (module docstring)."""
    if isinstance(value, str):
        for char, escaped in _ESCAPES:
            value = value.replace(char, escaped)
        return f"'{value}'"
    if isinstance(value, dict | list):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _render(call: dict) -> str:
    arguments = call["arguments"]
    body = arguments if isinstance(arguments, str) else ", ".join(
        f"{key}={value_text(value)}" for key, value in arguments.items())
    return f"{call['name']}({body})"


@dataclass(frozen=True)
class _Call:
    """One call of a block: its name and span, and (key, value or _NOT_LITERAL, source text, offset of the
    value's first character) per keyword argument."""

    name: str
    name_span: tuple[int, int]
    values: list[tuple[str, object, str, int]]


@dataclass(frozen=True)
class _Block:
    """One <|tool_call_start|> block: its calls, or None with the best name and the inner text when it is not
    a list of calls."""

    calls: list[_Call] | None
    name: str = ""
    raw: str = ""


def _blocks(text: str) -> list[_Block]:
    blocks = []
    position = 0
    while (start := text.find(OPEN, position)) >= 0:
        body = start + len(OPEN)
        close = text.find(CLOSE, body)
        end = len(text) if close < 0 else close
        inner = text[body:end]
        calls = _read(inner, body)
        if calls is not None:
            blocks.append(_Block(calls=calls))
        else:
            found = _BEST_NAME.match(inner)
            blocks.append(_Block(calls=None, name=found.group(1) if found else "", raw=inner.strip()))
        position = len(text) if close < 0 else close + len(CLOSE)
    return blocks


def _read(inner: str, base: int) -> list[_Call] | None:
    """The calls of a block's inner text, whose first character is at base in the answer (module
    docstring); None when it is not a list of calls."""
    lead = len(inner) - len(inner.lstrip())  # ast refuses leading blanks in an expression
    source = inner[lead:]
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError, MemoryError, RecursionError):  # ValueError: a null byte in the text
        return None
    if not isinstance(tree.body, ast.List) or not tree.body.elts:
        return None
    offsets = _Offsets(source, base + lead)
    calls = []
    for element in tree.body.elts:
        if not (isinstance(element, ast.Call) and isinstance(element.func, ast.Name) and not element.args
                and all(keyword.arg is not None for keyword in element.keywords)):
            return None
        values = []
        for keyword in element.keywords:
            node = keyword.value
            segment = ast.get_source_segment(source, node) or ""
            try:
                value = ast.literal_eval(node)
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                if not _is_json(segment):
                    return None
                value = _NOT_LITERAL
            else:
                if not json_value(value):
                    return None
            start = offsets.at(node.lineno, node.col_offset)
            opening = _STRING_OPENING.match(segment) if isinstance(value, str) else None
            values.append((keyword.arg, value, segment, start + (opening.end() if opening else 0)))
        name_start = offsets.at(element.func.lineno, element.func.col_offset)
        calls.append(_Call(name=element.func.id, name_span=(name_start, name_start + len(element.func.id)),
                           values=values))
    return calls


def _is_json(text: str) -> bool:
    """Whether text is JSON text of a value JSON holds (NaN and the infinities, which json.loads reads, are
    not)."""
    try:
        value = json.loads(text)
    except ValueError:
        return False
    return json_value(value)


class _Offsets:
    """Character offsets in the answer of ast positions (a line from 1, ended as Python ends lines, and a
    UTF-8 byte column) of a source whose first character is at base."""

    def __init__(self, source: str, base: int) -> None:
        self.lines = _LINE.findall(source) or [""]
        self.starts = [base]
        for line in self.lines[:-1]:
            self.starts.append(self.starts[-1] + len(line))

    def at(self, lineno: int, col_offset: int) -> int:
        line = self.lines[lineno - 1]
        return self.starts[lineno - 1] + len(line.encode("utf-8")[:col_offset].decode("utf-8", "ignore"))

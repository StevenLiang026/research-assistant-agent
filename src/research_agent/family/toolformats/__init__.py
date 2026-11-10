"""Tool-call formats (docs/m7-家族层设计.md §4): how a family's model writes its tool calls, and how they
are read.

A format renders calls (for b1's demonstrations and the tests; training targets come from the chat template
itself), parses an answer into research_agent.toolcalls' contract ({"name", "arguments": dict | str}; a
block it cannot read as a call gives {"name": the best name it can find, "arguments": the block's text},
which the framework records as bad_json), and locates each call's tool name and argument values in the text
(CallSpan), where the routing signals read their decision tokens (agent.route). Five formats: qwen_json
(Qwen3's JSON calls), qwen_xml (Qwen3-Coder's XML: Qwen3.5, Granite 4.2), minicpm_xml (MiniCPM5),
glm_kv (Spark-X2.5's GLM-4.5 key-value blocks) and pythonic (LFM2.5's Python-style list of calls).

A block a format cannot read as a call is {"name": the best name, "arguments": the block's inner text,
stripped}, as qwen_json gives it today; and every format renders a call whose arguments are such text (a
call the executor refused, shown back in a repair exchange) with the text where its arguments go, as the
Qwen3 templates print it verbatim where the JSON object goes. A format renders calls exactly as its family's
chat template renders the calls of an assistant message (tests/family/test_golden.py holds them against the
real templates), so b1's demonstrations and the inline policy of research_agent.family.core are the
template's text.

coerce is for the formats that write argument values as bare text (the XML-style and key-value ones): each
value becomes the type its JSON schema names, so their calls reach the executor as qwen_json's do. With
separated (the default) the value's text first loses one separating newline at either end (qwen_xml puts a
value on lines of its own); without it the text is the value's own (minicpm_xml and glm_kv write a value
between its tags directly). Then, by the schema's "type":

- "string": that text, as it is;
- "integer" / "number": a JSON number literal, blanks around it ignored. An integer may be written with an
  integral fraction or exponent (5.0, 1e2), which JSON Schema counts as an integer, and comes back an int.
  NaN, the infinities and a literal too large for a float are not numbers;
- "boolean": true, false, True or False; "null": null or None (blanks around them ignored);
- "array" / "object": JSON text of a list / of an object (json.loads);
- a list of types: each in its order, the first that holds the value wins. The aliases dict, list, float,
  int, bool and str stand for object, array, number, integer, boolean and string;
- no type (no schema, or one without "type"): the value JSON reads in the text when that is not a string
  (a number, a boolean, null, a list or an object), else the text;
- nothing holds the value, or the type is one this table does not know: the text, which the executor
  refuses as invalid_args.

Only "type" is read: enums, bounds and the like stay the executor's business. parameter_schema finds the
schema of a parameter among a step's tools (the OpenAI shape, {"type": "function", "function": {...}}, or
a bare function object), None when the tool, the parameter or the tools are absent: the value is then read
without a type.

Every value a format gives is one JSON holds (json_value): research_agent.agent.models refuses arguments
that canonical JSON cannot write as a harness error, which would end a run on a model's output. coerce
gives nothing else; the formats that read Python literals (minicpm_xml, pythonic) check what
ast.literal_eval gives, which can also be a set, a tuple, bytes, a complex number, an ellipsis, an infinite
float or a dictionary with keys that are not strings.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class CallSpan:
    """One call of an answer: parsed when the format reads it as a call; then the character span
    [start, end) of the tool name in the answer and the offset of the first character of each argument
    value, in the order the call writes them (for a quoted string, the first character after the quote)."""

    parsed: bool
    name: tuple[int, int] | None = None
    values: tuple[int, ...] = ()


@runtime_checkable
class ToolFormat(Protocol):
    """A family's tool-call format (module docstring)."""

    name: str
    open_tag: str  # the text that opens a call

    def render(self, calls: list[dict]) -> str:
        """The calls ({"name", "arguments"}) as the family's chat template writes them in an assistant
        turn."""

    def parse(self, text: str, tools: list[dict] | None) -> list[dict]:
        """The calls of an answer, in order, in research_agent.toolcalls' contract; tools (the step's tool
        schemas) type the values of the formats that write them as text (coerce)."""

    def locate(self, text: str) -> list[CallSpan]:
        """Every call of an answer, in parse's order, with where its name and its values are."""


_FAILED = object()  # a conversion that does not hold the value
_ALIASES = {"dict": "object", "list": "array", "float": "number", "int": "integer", "bool": "boolean",
            "str": "string"}
_NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")  # JSON's number grammar
_BOOLEANS = {"true": True, "True": True, "false": False, "False": False}
_NULLS = ("null", "None")


def coerce(text: str, schema: dict | None, *, separated: bool = True) -> object:
    """text as the type schema names (module docstring)."""
    value = text.removeprefix("\n").removesuffix("\n") if separated else text
    kind = (schema or {}).get("type")
    if kind is None:
        parsed = _json(value)
        return value if parsed is _FAILED or isinstance(parsed, str) else parsed
    for name in [kind] if isinstance(kind, str) else kind:
        converted = _CONVERSIONS.get(_ALIASES.get(name, name), _unknown)(value)
        if converted is not _FAILED:
            return converted
    return value


def parameter_schema(tools: list[dict] | None, name: str, key: str) -> dict | None:
    """The schema of parameter key of the tool named name among tools (module docstring), or None."""
    for tool in tools or ():
        function = tool.get("function", tool) if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name") == name:
            properties = (function.get("parameters") or {}).get("properties")
            schema = properties.get(key) if isinstance(properties, dict) else None
            return schema if isinstance(schema, dict) else None
    return None


def json_value(value: object) -> bool:
    """Whether value is one JSON holds, as json.loads gives them (module docstring): None, a boolean, an
    integer, a finite float, a string, a list of such values or a dictionary of string keys to them."""
    if value is None or isinstance(value, bool | int | str):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(json_value(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and json_value(item) for key, item in value.items())
    return False


def skip_blanks(text: str, index: int, end: int | None = None) -> int:
    """The first index from index, before end (the text's end when None), at which text holds something
    other than whitespace, or end: the formats let a model put blanks between the tags of a call."""
    end = len(text) if end is None else end
    while index < end and text[index].isspace():
        index += 1
    return index


def broken(name: str, raw: str) -> dict:
    """The call a block that cannot be read gives (module docstring): the best name and the block's text."""
    return {"name": name, "arguments": raw}


Reader = Callable[[str, dict | None], object]  # a value's text and schema -> the value (coerce's shape)


@dataclass(frozen=True)
class ReadBlock:
    """One block of a tag-delimited format (qwen_xml, minicpm_xml, glm_kv) as read: a call (parsed) with its
    name, the name's character span and (key, the value's text, the offset of the value's first character)
    per argument in the call's order; or not a call, with the best name and the block's inner text."""

    parsed: bool
    name: str
    name_span: tuple[int, int] | None = None
    values: tuple[tuple[str, str, int], ...] = ()
    raw: str = ""

    def call(self, tools: list[dict] | None, read: Reader) -> dict:
        """The block in the parser's contract: its values read by read with their schemas among tools, or
        broken (module docstring)."""
        if not self.parsed:
            return broken(self.name, self.raw)
        return {"name": self.name, "arguments": {key: read(text, parameter_schema(tools, self.name, key))
                                                 for key, text, _ in self.values}}

    def span(self) -> CallSpan:
        if not self.parsed:
            return CallSpan(parsed=False)
        return CallSpan(parsed=True, name=self.name_span,
                        values=tuple(offset for _, _, offset in self.values))


def _json(text: str) -> object:
    """The JSON value of text, or _FAILED: NaN, the infinities and an overflowing number are not JSON
    values."""
    try:
        return json.loads(text, parse_constant=_not_json, parse_float=_finite)
    except ValueError:  # json.JSONDecodeError is one, and so are the hooks' refusals
        return _FAILED


def _not_json(constant: str) -> object:
    raise ValueError(f"{constant} is not JSON")


def _finite(literal: str) -> float:
    number = float(literal)
    if not math.isfinite(number):
        raise ValueError(f"{literal} does not fit a float")
    return number


def _number(text: str) -> object:
    literal = text.strip()
    return _json(literal) if _NUMBER.fullmatch(literal) else _FAILED


def _integer(text: str) -> object:
    number = _number(text)
    if isinstance(number, float):
        return int(number) if number.is_integer() else _FAILED
    return number


def _of_type(kind: type) -> Callable[[str], object]:
    def convert(text: str) -> object:
        value = _json(text)
        return value if isinstance(value, kind) else _FAILED

    return convert


def _unknown(text: str) -> object:
    return _FAILED


_CONVERSIONS: dict[str, Callable[[str], object]] = {
    "string": lambda text: text,
    "integer": _integer,
    "number": _number,
    "boolean": lambda text: _BOOLEANS.get(text.strip(), _FAILED),
    "null": lambda text: None if text.strip() in _NULLS else _FAILED,
    "array": _of_type(list),
    "object": _of_type(dict),
}

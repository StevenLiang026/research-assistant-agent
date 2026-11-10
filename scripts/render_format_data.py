"""Render unused public records as format-mix training data (docs/m6-计划.md §3.4 D1, WP3).

Why. FT lost ACEBench's normal group to surface syntax: all 13,331 training samples carry the tool list
through the chat template (<tools> ... <tool_call>), none as text in the system prompt with a format
instruction, which is how ACEBench and many users ask (§2.5). These samples carry exactly that: the
functions and an instruction written into the system message, the gold calls of the record serialised in the
instructed text format, and no tools field for the template to render.

Records. research_agent.data.public_pool reads the five public sources back from the local Hugging Face
cache at the revisions of data/public/manifest.json, converts them with research_agent.data.public's
converters and leaves out every source id drawn into data/raw/public/<name>.jsonl. A record is eligible
when it has a tool call, no assistant message holds both text and calls (glaive-v2's "Sure, let me check."
followed by a call: the instruction says to answer with the calls only), and every called name is a dotted
Python identifier and every argument key a Python identifier that is not a keyword (so that each record can
take any of the three formats; ToolACE names with spaces are left out). Per source a fixed quota of the
1,500 (QUOTAS, scaled for --count) is drawn with random.Random(f"{seed}/{source}"): first
ceil(OBJECT_SHARE x quota) records with an object argument (a call argument whose value is an object, or a
list holding one), then the others; a source short of candidates passes its shortfall to the next source
in public.SOURCES order. The run fails unless at least half of the records have an object argument.
Parallel calls stay one assistant step with several calls.

Rendering. The system message lists each function's JSON schema (the OpenAI function object: name,
description, parameters) on one line, then the format instruction, in Chinese or English (two wordings per
language, chosen per record). The instruction's example uses the placeholder function_name, never a real
function. Every assistant message with calls becomes an assistant message whose content is the calls in the
record's format; tool messages keep their result text and answer by position (the model wrote text, there
is no call id); other messages are copied. The written sample keeps "tools": [] (present and empty, because
research_agent.train.samples.load_records reads a record without the key as an agent history and would
compress its tool outputs; an empty list makes the chat template add no <tools> block) and adds "format"
and "instruction_lang", which the trainer ignores. Formats, preregistered, a third of the records each,
balanced within each (language, object argument) stratum:

- python: one call per line, f(k="v", n=3), strings in double quotes, objects and lists as Python literals,
  no brackets around the calls;
- json: one JSON array [{"name": ..., "arguments": {...}}];
- yaml: a block list, "- name: f" then "  arguments:" and the arguments as a block mapping; strings are
  double-quoted (YAML's double-quoted scalars, which JSON escapes are a subset of), keys plain when they are
  identifiers and not YAML 1.1 booleans or null.

ACEBench's notation [ApiName(key='value')] (a bracket opening a call list of Python calls) is never
produced. Each format has a parser here (parse_calls) that inverts its serialiser; the run checks that every
written record parses back to the record's calls.

Instruction language: records of glaive_zh and deepexi (Chinese dialogues) get a Chinese instruction; enough
records of the English sources get one too for half of the file to be Chinese; the dialogue is not changed.

Length: with --model pointing at a model folder, a record is a candidate only when all twelve renderings
(three formats, two languages, two wordings) fit schemas.CONTEXT_TOKENS with the model's chat template
(thinking off), since the trainer drops an over-length step instead of truncating it.

Written: --out, one JSON object per line (json.dumps, UTF-8 without escapes, LF), the same bytes for the
same seed and cache; --manifest, the counts per format, source, language and stratum, the object-argument
share, the excluded source ids, the output's SHA-256 and size. An existing output is overwritten only with
--force.

    uv run --no-sync python scripts/render_format_data.py --out data/raw/public/format-mix.jsonl \\
        --manifest data/public/format-mix-manifest.json [--count 1500] [--seed 42] [--force]

Exit 0 written; 2 usage (an output exists without --force); 3 the data do not meet the rules (too few
records, object share under half, a record that does not round-trip).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import keyword
import math
import random
import re
import sys
from collections.abc import Callable
from pathlib import Path

from research_agent.agent.types import api_to_hf
from research_agent.data import public_pool
from research_agent.data.public import ROLES, SOURCES, has_tool_call
from research_agent.tools.schemas import CONTEXT_TOKENS

SEED = 42
COUNT = 1500
MODEL = "models/Qwen3-1.7B"
FORMATS = ("python", "json", "yaml")
LANGS = ("zh", "en")
VARIANTS = 2
ZH_SOURCES = ("glaive_zh", "deepexi")  # Chinese dialogues: always a Chinese instruction
QUOTAS = {"toolace": 500, "glaive_v2": 350, "hermes": 250, "glaive_zh": 62, "deepexi": 338}  # of COUNT
OBJECT_SHARE = 0.6  # per source, of its quota, when it has the candidates; the file needs >= 0.5
MIN_OBJECT_SHARE = 0.5
FORMAT_DESCRIPTIONS = {
    "python": 'one call per line as f(k="v", n=3), objects and lists as Python literals, no outer brackets',
    "json": 'one JSON array [{"name": ..., "arguments": {...}}]',
    "yaml": "a YAML block list: - name: f / arguments: block mapping, strings double-quoted",
}
TOOLS_TEXT = (
    "each function's JSON schema (the OpenAI function object: name, description, parameters) on "
    "one line of the system message, followed by the format instruction"
)
OBJECT_DEFINITION = "a call argument whose value is an object, or a list holding an object"

# Reasons a record is not eligible.
NO_CALL = "no tool call"
TEXT_WITH_CALL = "an assistant message holds text and calls"
NOT_PYTHON_NAME = "a called name or an argument key is not a Python identifier"

Call = tuple[str, dict]

# ---- the instruction (own wording; the examples use placeholders, never a listed function) ------------

_PREAMBLE = {
    "en": (
        "You have access to the functions below, one JSON schema per line:\n{tools}",
        "The following functions are available. Each line describes one of them as a JSON schema.\n{tools}",
    ),
    "zh": (
        "你可以调用下列函数,每行是一个函数的 JSON schema:\n{tools}",
        "以下是可用的函数,每行用 JSON schema 描述一个函数。\n{tools}",
    ),
}
_FORMAT_RULE = {
    "en": {
        "python": (
            "To call functions, reply with the calls only, one per line, written as Python function "
            'calls with keyword arguments, for example function_name(param_1="text", param_2=10). '
            "Write objects and lists as Python literals and put nothing around the calls."
        ),
        "json": (
            "To call functions, reply with the calls only, as one JSON array of objects with the keys "
            '"name" and "arguments", for example '
            '[{"name": "function_name", "arguments": {"param_1": "text"}}].'
        ),
        "yaml": (
            "To call functions, reply with the calls only, as a YAML list in which every item has a name "
            "and an arguments mapping, for example:\n- name: function_name\n  arguments:\n"
            '    param_1: "text"'
        ),
    },
    "zh": {
        "python": (
            "需要调用函数时,只输出调用本身,每行一个,写成带关键字参数的 Python 函数调用,例如 "
            'function_name(param_1="text", param_2=10)。对象和列表写成 Python 字面量,调用前后不加'
            "任何内容。"
        ),
        "json": (
            '需要调用函数时,只输出调用本身,写成一个 JSON 数组,数组里每个对象含 "name" 和 '
            '"arguments" 两个键,例如 [{"name": "function_name", "arguments": '
            '{"param_1": "text"}}]。'
        ),
        "yaml": (
            "需要调用函数时,只输出调用本身,写成 YAML 列表,每一项含 name 和 arguments 映射,例如:\n"
            '- name: function_name\n  arguments:\n    param_1: "text"'
        ),
    },
}
_CLOSING = {
    "en": (
        (
            "The names in the example are placeholders: use the functions and parameters listed above. "
            "When no call is needed, or once the results are back, answer in plain text."
        ),
        (
            "Replace the placeholder names with a listed function and its parameters. Answer in plain "
            "text when no function needs to be called, and after the function results arrive."
        ),
    ),
    "zh": (
        (
            "示例中的名称只是占位符,请换成上面列出的函数和参数。不需要调用函数时,或拿到函数结果之后,"
            "直接用自然语言回答。"
        ),
        "请把占位名称换成列表中的函数及其参数。无需调用函数时以及收到函数结果之后,用自然语言作答。",
    ),
}


def tools_text(tools: list[dict]) -> str:
    """One line per function: its OpenAI function object as JSON, in the record's key order."""
    return "\n".join(json.dumps(t["function"], ensure_ascii=False) for t in tools)


def instruction(tools: list[dict], fmt: str, lang: str, variant: int) -> str:
    """The system message: the functions as text, the format rule, the closing (wording variant 0 or 1)."""
    return "\n\n".join(
        [
            _PREAMBLE[lang][variant].format(tools=tools_text(tools)),
            _FORMAT_RULE[lang][fmt],
            _CLOSING[lang][variant],
        ]
    )


# ---- serialisers and parsers --------------------------------------------------------------------------


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, (str, bool, int, float)):
        return not isinstance(value, float) or math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_json_value(v) for k, v in value.items())
    return False


def _py(value: object) -> str:
    """value as a Python literal: strings in double quotes (JSON escapes are Python escapes),
    True/False/None."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "[" + ", ".join(_py(v) for v in value) + "]"
    if isinstance(value, dict):
        return (
            "{" + ", ".join(f"{json.dumps(k, ensure_ascii=False)}: {_py(v)}" for k, v in value.items()) + "}"
        )
    raise TypeError(f"not a JSON value: {value!r}")


def _dotted(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _dotted(node.value)
        return None if head is None else f"{head}.{node.attr}"
    return None


def _parse_python(text: str) -> list[Call]:
    try:
        tree = ast.parse(text.strip(), mode="exec")
    except SyntaxError as exc:
        raise ValueError(f"not Python calls: {exc.msg}") from None
    calls: list[Call] = []
    for stmt in tree.body:
        call = stmt.value if isinstance(stmt, ast.Expr) else None
        if not isinstance(call, ast.Call) or call.args:
            raise ValueError("each line must be one call with keyword arguments only")
        name = _dotted(call.func)
        if name is None:
            raise ValueError("a call must name a function")
        arguments: dict = {}
        for kw in call.keywords:
            if kw.arg is None or kw.arg in arguments:
                raise ValueError("arguments must be distinct keywords")
            try:
                value = ast.literal_eval(kw.value)
            except (ValueError, SyntaxError, TypeError) as exc:
                raise ValueError(f"argument {kw.arg} is not a literal") from exc
            if not _is_json_value(value):
                raise ValueError(f"argument {kw.arg} is not a JSON value")
            arguments[kw.arg] = value
        calls.append((name, arguments))
    if not calls:
        raise ValueError("no call")
    return calls


def _parse_json(text: str) -> list[Call]:
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"not JSON: {exc}") from None
    if not isinstance(value, list) or not value:
        raise ValueError("not a non-empty JSON array")
    calls = []
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"name", "arguments"}
            or not isinstance(item["name"], str)
            or not item["name"]
            or not isinstance(item["arguments"], dict)
        ):
            raise ValueError('each item must be {"name": ..., "arguments": {...}}')
        calls.append((item["name"], item["arguments"]))
    return calls


# YAML: the emitter writes, and the parser reads, one subset: block mappings and sequences, scalars as JSON
# (double-quoted strings, numbers, true/false/null, {} and []), plain keys only when unambiguous.
_PLAIN_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_YAML_WORDS = {"true", "false", "null", "yes", "no", "on", "off", "y", "n"}
_DECODER = json.JSONDecoder()


def _ykey(key: str) -> str:
    plain = _PLAIN_KEY.fullmatch(key) and key.lower() not in _YAML_WORDS
    return key if plain else json.dumps(key, ensure_ascii=False)


def _yscalar(value: object) -> str:
    if isinstance(value, float):
        text = repr(value)
        return text.replace("e", ".0e") if "e" in text and "." not in text else text  # 1e+20: YAML 1.1 str
    return json.dumps(value, ensure_ascii=False)  # None, bools, ints, strings, {} and []


def _ymap(mapping: dict, indent: int) -> list[str]:
    pad, lines = " " * indent, []
    for key, value in mapping.items():
        if isinstance(value, dict) and value:
            lines += [f"{pad}{_ykey(key)}:"] + _ymap(value, indent + 2)
        elif isinstance(value, list) and value:
            lines += [f"{pad}{_ykey(key)}:"] + _yseq(value, indent + 2)
        else:
            lines.append(f"{pad}{_ykey(key)}: {_yscalar(value)}")
    return lines


def _yseq(items: list, indent: int) -> list[str]:
    pad, lines = " " * indent, []
    for item in items:
        if isinstance(item, dict) and item:
            block = _ymap(item, indent + 2)
            lines += [f"{pad}- {block[0].lstrip()}"] + block[1:]
        elif isinstance(item, list) and item:
            lines += [f"{pad}-"] + _yseq(item, indent + 2)
        else:
            lines.append(f"{pad}- {_yscalar(item)}")
    return lines


def _serialize_yaml(calls: list[Call]) -> str:
    lines = []
    for name, arguments in calls:
        lines.append(f"- name: {_ykey(name)}")
        lines += ["  arguments:"] + _ymap(arguments, 4) if arguments else ["  arguments: {}"]
    return "\n".join(lines)


def _split_key(content: str) -> tuple[str, str] | None:
    """(key, rest after the colon) of a mapping entry line; None when the line is not one."""
    if content.startswith('"'):
        try:
            key, end = _DECODER.raw_decode(content)
        except ValueError:
            return None
        tail = content[end:]
        if not isinstance(key, str) or not tail.startswith(":") or tail[1:2] not in ("", " "):
            return None
        return key, tail[1:].strip()
    match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_.]*):(?: (.*))?", content)
    return (match.group(1), (match.group(2) or "").strip()) if match else None


def _no_constant(name: str) -> object:
    raise ValueError(f"{name} is not written")


def _yscalar_parse(text: str) -> object:
    """A JSON scalar (NaN and Infinity refused), {} or [], or a plain identifier-like string (a name)."""
    try:
        value = json.loads(text, parse_constant=_no_constant)
    except ValueError:
        if _PLAIN_KEY.fullmatch(text) and text.lower() not in _YAML_WORDS:
            return text
        raise ValueError(f"not a scalar of the subset: {text[:40]!r}") from None
    if isinstance(value, (dict, list)) and value:
        raise ValueError("flow collections other than {} and [] are not written")
    return value


def _is_item(content: str) -> bool:
    return content == "-" or content.startswith("- ")


def _yblock(lines: list[tuple[int, str]], i: int, indent: int) -> tuple[object, int]:
    return (_yseq_parse if _is_item(lines[i][1]) else _ymap_parse)(lines, i, indent)


def _child(lines: list[tuple[int, str]], i: int, indent: int) -> tuple[object, int]:
    if i < len(lines) and lines[i][0] > indent:
        return _yblock(lines, i, lines[i][0])
    raise ValueError("an empty value must be followed by a deeper block")


def _yseq_parse(lines: list[tuple[int, str]], i: int, indent: int) -> tuple[list, int]:
    out = []
    while i < len(lines) and lines[i][0] == indent and _is_item(lines[i][1]):
        rest = lines[i][1][1:].strip()
        if not rest:
            value, i = _child(lines, i + 1, indent)
        elif _split_key(rest) is not None:
            lines[i] = (indent + 2, rest)
            value, i = _ymap_parse(lines, i, indent + 2)
        else:
            value, i = _yscalar_parse(rest), i + 1
        out.append(value)
    return out, i


def _ymap_parse(lines: list[tuple[int, str]], i: int, indent: int) -> tuple[dict, int]:
    out: dict = {}
    while i < len(lines) and lines[i][0] == indent and not _is_item(lines[i][1]):
        entry = _split_key(lines[i][1])
        if entry is None or entry[0] in out:
            raise ValueError(f"not a mapping entry: {lines[i][1][:40]!r}")
        key, rest = entry
        if rest:
            out[key], i = _yscalar_parse(rest), i + 1
        else:
            out[key], i = _child(lines, i + 1, indent)
    return out, i


def _parse_yaml(text: str) -> list[Call]:
    lines = []
    for raw in text.strip("\n").split("\n"):
        if not raw.strip():
            continue
        body = raw.lstrip(" ")
        if body.startswith("\t"):
            raise ValueError("tabs in indentation")
        lines.append((len(raw) - len(body), body.rstrip()))
    if not lines or lines[0][0] != 0 or not _is_item(lines[0][1]):
        raise ValueError("not a YAML block list")
    value, i = _yseq_parse(lines, 0, 0)
    if i != len(lines):
        raise ValueError(f"unexpected line: {lines[i][1][:40]!r}")
    calls = []
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"name", "arguments"}
            or not isinstance(item["name"], str)
            or not item["name"]
            or not isinstance(item["arguments"], dict)
        ):
            raise ValueError("each item must hold a name and an arguments mapping")
        calls.append((item["name"], item["arguments"]))
    return calls


def serialize_calls(calls: list[Call], fmt: str) -> str:
    """The calls of one assistant step as text in format fmt (FORMATS)."""
    if fmt == "python":
        return "\n".join(
            f"{name}(" + ", ".join(f"{k}={_py(v)}" for k, v in args.items()) + ")" for name, args in calls
        )
    if fmt == "json":
        return json.dumps([{"name": name, "arguments": args} for name, args in calls], ensure_ascii=False)
    if fmt == "yaml":
        return _serialize_yaml(calls)
    raise ValueError(f"unknown format {fmt!r}")


def parse_calls(text: str, fmt: str) -> list[Call]:
    """The calls in text written in format fmt; ValueError when text is not that format."""
    parser = {"python": _parse_python, "json": _parse_json, "yaml": _parse_yaml}.get(fmt)
    if parser is None:
        raise ValueError(f"unknown format {fmt!r}")
    return parser(text)


# ---- records ------------------------------------------------------------------------------------------


def _calls(message: dict) -> list[Call]:
    return [
        (c["function"]["name"], json.loads(c["function"]["arguments"]))
        for c in message.get("tool_calls") or []
    ]


def _python_name(name: str) -> bool:
    return all(part.isidentifier() and not keyword.iskeyword(part) for part in name.split("."))


def ineligible(sample: dict) -> str | None:
    """Why the sample cannot be rendered in every format; None when it can."""
    if not has_tool_call(sample):
        return NO_CALL
    for message in sample["messages"]:
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        if message["content"].strip():
            return TEXT_WITH_CALL
        for name, arguments in _calls(message):
            if not _python_name(name) or not all(_python_name(k) and "." not in k for k in arguments):
                return NOT_PYTHON_NAME
    return None


def has_object_argument(sample: dict) -> bool:
    """True when some call has an argument whose value is an object or a list holding one."""
    for message in sample["messages"]:
        for _, arguments in _calls(message):
            for value in arguments.values():
                if isinstance(value, dict) or (
                    isinstance(value, list) and any(isinstance(v, dict) for v in value)
                ):
                    return True
    return False


def render_sample(sample: dict, fmt: str, lang: str, variant: int = 0) -> dict:
    """The sample with its functions and the format instruction in a system message, calls as text."""
    if sample["messages"][0]["role"] == "system":
        raise ValueError("the converted sample already has a system message")
    messages = [{"role": "system", "content": instruction(sample["tools"], fmt, lang, variant)}]
    for message in sample["messages"]:
        if message["role"] == "assistant" and message.get("tool_calls"):
            messages.append({"role": "assistant", "content": serialize_calls(_calls(message), fmt)})
        else:
            messages.append({"role": message["role"], "content": message["content"]})
    return {
        "messages": messages,
        "tools": [],
        "source": sample["source"],
        "source_id": sample["source_id"],
        "format": fmt,
        "instruction_lang": lang,
    }


def unrender(rendered: dict) -> dict:
    """The rendered sample with every assistant message that parses in its format turned back into calls
    (ids call_0, call_1, ...); used to check the round trip and to count object arguments."""
    messages, count = [], 0
    for message in rendered["messages"][1:]:
        if message["role"] == "assistant":
            try:
                calls = parse_calls(message["content"], rendered["format"])
            except ValueError:
                calls = []
            if calls:
                tool_calls = []
                for name, arguments in calls:
                    tool_calls.append(
                        {
                            "id": f"call_{count}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(arguments, ensure_ascii=False),
                            },
                        }
                    )
                    count += 1
                messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
                continue
        messages.append(dict(message))
    return {
        "messages": messages,
        "tools": [],
        "source": rendered["source"],
        "source_id": rendered["source_id"],
    }


_ACEBENCH_CALL_LIST = re.compile(r"\[\w+\(")


def check_rendered(sample: dict) -> list[str]:
    """Why a rendered sample is not in this file's format, as messages; [] when it is."""
    errors = []
    if sample.get("tools") != []:
        errors.append("tools must be an empty list")
    if sample.get("format") not in FORMATS or sample.get("instruction_lang") not in LANGS:
        errors.append("format or instruction_lang missing or unknown")
    messages = sample.get("messages") or []
    if len(messages) < 3 or messages[0].get("role") != "system" or messages[1].get("role") != "user":
        errors.append("messages must open with the system message and a user message")
    if messages and messages[-1].get("role") != "assistant":
        errors.append("the last message must be from the assistant")
    calls_before = 0  # calls of the latest assistant message still without a result
    for i, m in enumerate(messages):
        if m.get("role") not in ROLES or not isinstance(m.get("content"), str):
            errors.append(f"message {i}: bad role or content")
            continue
        if "tool_calls" in m or "tool_call_id" in m:
            errors.append(f"message {i}: tool_calls and tool_call_id are not written")
        if m["role"] in ("system", "assistant") and _ACEBENCH_CALL_LIST.search(m["content"]):
            errors.append(f"message {i}: holds ACEBench's call-list notation")
        if m["role"] == "assistant":
            try:
                calls_before = len(parse_calls(m["content"], sample.get("format", "")))
            except ValueError:
                calls_before = 0
            if not m["content"].strip():
                errors.append(f"message {i}: empty assistant message")
        elif m["role"] == "tool":
            if calls_before <= 0:
                errors.append(f"message {i}: tool message without a call before it")
            calls_before -= 1
        else:
            calls_before = 0
    if not any(m.get("role") == "assistant" and "tool_calls" not in m for m in messages):
        errors.append("no assistant message")
    return errors


# ---- selection ----------------------------------------------------------------------------------------


def quotas(count: int) -> dict[str, int]:
    """QUOTAS scaled to count (rounded down), the remainder to toolace."""
    total = sum(QUOTAS.values())
    out = {name: QUOTAS[name] * count // total for name in QUOTAS}
    out["toolace"] += count - sum(out.values())
    return out


def _parallel(sample: dict) -> bool:
    return any(len(m.get("tool_calls") or []) > 1 for m in sample["messages"])


def select(
    pools: dict[str, list[dict]],
    wanted: dict[str, int],
    seed: int,
    *,
    object_share: float = OBJECT_SHARE,
    fits: Callable[[dict], bool] | None = None,
) -> tuple[list[dict], dict]:
    """The chosen records, in source order (public.SOURCES) and pool order, and an account per source.

    Per source, eligible records are split into those with an object argument and the rest, each shuffled
    with random.Random(f"{seed}/{source}"); ceil(object_share x quota) come from the first list, the rest of
    the quota from the second (the first list fills in when the second runs out). A record fits() rejects
    is skipped and counted. A source's shortfall goes to the next sources in order, object records first.
    """
    order = [name for name in SOURCES if name in pools] + [n for n in pools if n not in SOURCES]
    queues: dict[str, tuple[list[dict], list[dict]]] = {}
    account: dict[str, dict] = {}
    chosen: dict[str, list[dict]] = {}

    def take(name: str, queue: list[dict], n: int) -> int:
        got = 0
        while queue and got < n:
            s = queue.pop(0)
            if fits is not None and not fits(s):
                account[name]["too_long"] += 1
                continue
            chosen[name].append(s)
            got += 1
        return got

    shortfall = 0
    for name in order:
        reasons: dict[str, int] = {}
        eligible = []
        for s in pools[name]:
            reason = ineligible(s)
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
            else:
                eligible.append(s)
        objects = [s for s in eligible if has_object_argument(s)]
        rest = [s for s in eligible if not has_object_argument(s)]
        rng = random.Random(f"{seed}/{name}")
        rng.shuffle(objects)
        rng.shuffle(rest)
        queues[name] = (objects, rest)
        quota = wanted.get(name, 0)
        account[name] = {
            "candidates": len(pools[name]),
            "ineligible": dict(sorted(reasons.items())),
            "eligible": len(eligible),
            "object_candidates": len(objects),
            "quota": quota,
            "too_long": 0,
        }
        chosen[name] = []
        got = take(name, objects, math.ceil(object_share * quota))
        got += take(name, rest, quota - got)
        got += take(name, objects, quota - got)
        shortfall += quota - got
    for name in order:
        if shortfall <= 0:
            break
        objects, rest = queues[name]
        shortfall -= take(name, objects, shortfall)
        shortfall -= take(name, rest, shortfall)
    index = {(name, id(s)): i for name in order for i, s in enumerate(pools[name])}
    picks = []
    for name in order:
        records = sorted(chosen[name], key=lambda s, n=name: index[(n, id(s))])
        account[name].update(
            {
                "picked": len(records),
                "object": sum(has_object_argument(s) for s in records),
                "parallel": sum(_parallel(s) for s in records),
            }
        )
        picks += records
    return picks, {"per_source": account, "shortfall": max(shortfall, 0)}


def assign(picks: list[dict], seed: int, *, zh_sources: tuple[str, ...] = ZH_SOURCES) -> list[dict]:
    """The picks rendered: a language, a format and a wording each, all drawn with random.Random(seed).

    ZH_SOURCES records are Chinese; then English-source records, shuffled, become Chinese until half the
    file is. Formats go round in FORMATS order over the strata (language, object argument), each stratum's
    records shuffled, so each format gets a third and every stratum is split as evenly as it can be.
    """
    rng = random.Random(f"{seed}/assign")
    langs = ["zh" if s["source"] in zh_sources else "en" for s in picks]
    others = [i for i, lang in enumerate(langs) if lang == "en"]
    rng.shuffle(others)
    for i in others[: max(0, len(picks) // 2 - langs.count("zh"))]:
        langs[i] = "zh"
    formats: list[str | None] = [None] * len(picks)
    turn = 0
    for lang in LANGS:
        for obj in (True, False):
            stratum = [i for i, s in enumerate(picks) if langs[i] == lang and has_object_argument(s) == obj]
            rng.shuffle(stratum)
            for i in stratum:
                formats[i] = FORMATS[turn % len(FORMATS)]
                turn += 1
    variants = [rng.randrange(VARIANTS) for _ in picks]
    return [render_sample(s, formats[i], langs[i], variants[i]) for i, s in enumerate(picks)]


# ---- CLI ----------------------------------------------------------------------------------------------


def load_pools(manifest: str | Path, raw_dir: str | Path) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """The unused records of every source from the Hugging Face cache, and their accounts (with revision)."""
    entries = public_pool.load_manifest(manifest)["sources"]
    pools, accounts = {}, {}
    for name in SOURCES:
        pools[name], accounts[name] = public_pool.load_unused(name, entries[name], raw_dir)
        accounts[name]["revision"] = entries[name]["revision"]
    return pools, accounts


def make_fits(model: str | Path) -> Callable[[dict], bool] | None:
    """fits(sample): all renderings fit CONTEXT_TOKENS with the model's template; None without the folder."""
    if not Path(model).is_dir():
        return None
    from transformers import AutoTokenizer  # heavy import, only for the length filter

    tokenizer = AutoTokenizer.from_pretrained(model)

    def fits(sample: dict) -> bool:
        texts = [
            tokenizer.apply_chat_template(
                api_to_hf(render_sample(sample, fmt, lang, v)["messages"]),
                tools=None,
                tokenize=False,
                enable_thinking=False,
            )
            for fmt in FORMATS
            for lang in LANGS
            for v in range(VARIANTS)
        ]
        lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
        return max(lengths) <= CONTEXT_TOKENS

    return fits


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def build_manifest(
    args: argparse.Namespace,
    rendered: list[dict],
    picks: list[dict],
    account: dict,
    accounts: dict[str, dict],
    data: bytes,
    fits_on: bool,
) -> dict:
    n = len(rendered)
    objects = sum(has_object_argument(s) for s in picks)
    per_source = {}
    for name, entry in account["per_source"].items():
        per_source[name] = {**accounts.get(name, {}), **entry}
    by_lang_format = {
        lang: {
            fmt: sum(1 for r in rendered if r["instruction_lang"] == lang and r["format"] == fmt)
            for fmt in FORMATS
        }
        for lang in LANGS
    }
    call_steps = [len(m.get("tool_calls")) for s in picks for m in s["messages"] if m.get("tool_calls")]
    return {
        "script": "scripts/render_format_data.py",
        "seed": args.seed,
        "count": args.count,
        "source_manifest": repo_relative(Path(args.source_manifest)),
        "tools_text": TOOLS_TEXT,
        "instruction_variants_per_language": VARIANTS,
        "formats": {
            fmt: {
                "description": FORMAT_DESCRIPTIONS[fmt],
                "records": sum(r["format"] == fmt for r in rendered),
            }
            for fmt in FORMATS
        },
        "languages": {lang: sum(r["instruction_lang"] == lang for r in rendered) for lang in ("en", "zh")},
        "format_by_language": by_lang_format,
        "object_argument": {
            "definition": OBJECT_DEFINITION,
            "records": objects,
            "share": round(objects / n, 4) if n else None,
            "target_share_per_source": OBJECT_SHARE,
            "minimum_share": MIN_OBJECT_SHARE,
        },
        "call_steps": len(call_steps),
        "parallel_steps": sum(1 for c in call_steps if c > 1),
        "calls": sum(call_steps),
        "multi_step_records": sum(
            1 for s in picks if sum(1 for m in s["messages"] if m.get("tool_calls")) > 1
        ),
        "per_source": per_source,
        "shortfall": account["shortfall"],
        "excluded_source_ids": {
            "per_source": {k: v.get("drawn", 0) for k, v in accounts.items()},
            "total": sum(v.get("drawn", 0) for v in accounts.values()),
        },
        "length_filter": {
            "model": repo_relative(Path(args.model)) if fits_on else None,
            "context_tokens": CONTEXT_TOKENS,
        },
        "output": {
            "path": repo_relative(Path(args.out)),
            "records": n,
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the JSONL file to write")
    ap.add_argument("--manifest", required=True, help="the manifest to write")
    ap.add_argument("--count", type=int, default=COUNT)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    ap.add_argument("--model", default=MODEL, help="model folder for the length filter (skipped when absent)")
    ap.add_argument("--source-manifest", default=str(public_pool.MANIFEST))
    ap.add_argument("--raw-dir", default=str(public_pool.RAW_DIR), help="the drawn files to exclude")
    args = ap.parse_args(argv)
    out, manifest_path = Path(args.out), Path(args.manifest)
    existing = [p for p in (out, manifest_path) if p.exists()]
    if existing and not args.force:
        print(f"{', '.join(map(str, existing))} exists; pass --force to overwrite", file=sys.stderr)
        return 2
    pools, accounts = load_pools(args.source_manifest, args.raw_dir)
    fits = make_fits(args.model)
    if fits is None:
        print(f"no model folder at {args.model}: no length filter", flush=True)
    picks, account = select(pools, quotas(args.count), args.seed, fits=fits)
    if len(picks) != args.count:
        print(f"only {len(picks)} eligible records for {args.count}", file=sys.stderr)
        return 3
    objects = sum(has_object_argument(s) for s in picks)
    if objects < MIN_OBJECT_SHARE * len(picks):
        print(f"object-argument share {objects}/{len(picks)} is under {MIN_OBJECT_SHARE}", file=sys.stderr)
        return 3
    rendered = assign(picks, args.seed)
    for original, sample in zip(picks, rendered, strict=True):
        errors = check_rendered(sample)
        back = [_calls(m) for m in unrender(sample)["messages"] if m.get("tool_calls")]
        if errors or back != [_calls(m) for m in original["messages"] if m.get("tool_calls")]:
            print(
                f"{sample['source']}/{sample['source_id']}: {errors or 'calls do not round-trip'}",
                file=sys.stderr,
            )
            return 3
    data = "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in rendered).encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    manifest = build_manifest(args, rendered, picks, account, accounts, data, fits is not None)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"{out}: {len(rendered)} records; formats {[manifest['formats'][f]['records'] for f in FORMATS]}; "
        f"languages {manifest['languages']}; object share {manifest['object_argument']['share']}; "
        f"per source { ({k: v['picked'] for k, v in manifest['per_source'].items()}) }"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

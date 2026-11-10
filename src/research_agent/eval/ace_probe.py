"""The ACEBench-style development probe of M6 (docs/m6-计划.md §3.4 D3, WP4): its prompt, its answer parsers
and its scorer, shared by scripts/build_ace_probe.py, scripts/run_ace_probe.py and
scripts/calibrate_ace_probe.py.

An item is {id, kind, lang, system, user, tools, expected, source, source_id, messages}. kind is "normal" (the
request is to be answered with tool calls) or "incomplete" (a required parameter's value is missing from the
request, and the answer is the fixed sentence naming it); lang is "zh" or "en", the language of the system
text; system is the function list as text plus the instruction (system_prompt); tools are the schemas, for
scoring and checks only (the model never gets a tools field); expected is {"calls": [{name, arguments}, ...]}
for normal items and {"function", "missing": [names]} for incomplete ones; messages repeats system and user
as the two API messages that are sent, so that research_agent.train.samples.load_records (and with it
scripts/check_benchmark_overlap.py) reads the probe as public samples.

The prompt (own wording, never ACEBench's). The system text lists each function's OpenAI function object
as JSON, one per line, then asks for one fixed format, a JSON list of call objects
[{"name": ..., "arguments": {...}}] (FORMAT), and, when a fitting function lacks a required value, for the
fixed English sentence SENTENCE_TEMPLATE in both languages. Its examples use placeholders (PLACEHOLDERS),
never a listed function. The sentence differs from ACEBench's ("Missing necessary parameters ...") and from
the training sentence of scripts/replay_base.py, so the probe measures whether a model follows a sentence it
is given rather than one it was trained on, as ACEBench does.

Scoring (SCORING_RULE). A normal answer is parsable when, stripped and with at most one surrounding Markdown
code fence removed, it is a JSON array of one or more objects with exactly the keys "name" (a non-empty
string) and "arguments" (an object), every number finite; it is exact when it is parsable and its calls equal
the gold calls as a multiset: names equal, arguments equal after the repository's normalisation
(research_agent.eval.metrics: the tool schema's defaults filled in on both sides, then normalize_value per
parameter, so strings are NFKC-normalised with whitespace collapsed and case kept, integral floats are ints,
and parameters named categories, tags or missing compare as sets). An incomplete answer hits the sentence
when the sentence form appears in it (fixed words case-insensitive, any whitespace, the function name and
parameter list possibly quoted, no placeholder), and hits the parameters when, besides, the named function is
the expected one and the named parameters equal the missing ones as a set; called counts incomplete answers
that are parsable calls (the call the instruction says not to make).

Failure shapes (SHAPE_RULE), counted over every answer: kwargs, an object written with key=value inside
braces ({key=value}); tool_call, the trained <tool_call> tag present; placeholder, a placeholder of the
instruction copied; runaway, an answer longer than RUNAWAY_CHARS characters.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter

from ..canonical import canonical_json
from .metrics import normalize_value

KINDS = ("normal", "incomplete")
LANGS = ("zh", "en")
FORMAT = 'one JSON array of call objects [{"name": ..., "arguments": {...}}]'
SENTENCE = "Unable to call {function} because the request does not give the required parameter(s): {params}."
SENTENCE_TEMPLATE = SENTENCE.format(function="<function_name>", params="<parameter_names>")
PLACEHOLDERS = ("example_function", "example_parameter", "<function_name>", "<parameter_names>")
RUNAWAY_CHARS = 2500
SHAPES = ("kwargs", "tool_call", "placeholder", "runaway")
SCORING_RULE = (
    "normal: parsable when the answer, stripped and with at most one surrounding Markdown code fence "
    "removed, "
    'is a JSON array of one or more objects with exactly the keys "name" (non-empty string) and "arguments" '
    "(object), numbers finite; exact when parsable and the calls equal the gold calls as a multiset, names "
    "equal and arguments equal after the schema defaults are filled in on both sides and "
    "research_agent.eval.metrics.normalize_value is applied per parameter. incomplete: sentence when the "
    "fixed sentence form appears in the answer (fixed words case-insensitive, whitespace free, no "
    "placeholder); parameter when, besides, the function is the expected one and the named parameters equal "
    "the missing ones as a set; called when the answer is parsable calls. Rates are counts / n of the kind."
)
SHAPE_RULE = (
    "Per answer: kwargs, braces holding key=value ({key=value}); tool_call, '<tool_call>' present; "
    f"placeholder, one of {', '.join(PLACEHOLDERS)} present; runaway, more than {RUNAWAY_CHARS} characters."
)

_PREAMBLE = {
    "en": "You are given the functions below. Each line is the JSON schema of one function.\n{tools}",
    "zh": "下面给出可用的函数,每行是一个函数的 JSON schema。\n{tools}",
}
_EXAMPLE = '[{"name": "example_function", "arguments": {"example_parameter": "value"}}]'
_RULES = {
    "en": (
        "How to answer:\n"
        "1. If the functions can carry out the request, reply only with a JSON array of the calls to make. "
        'Every element is an object with exactly two keys, "name" (the function name) and "arguments" (an '
        "object mapping parameter names to values). Put all calls in the one array. Example: {example}\n"
        "2. If a function fits the request but the user has not supplied a value for one of its required "
        "parameters, do not call it and do not make the value up. Reply with exactly this sentence, putting "
        "in the function name and the missing parameter names (separated by commas):\n{sentence}"
    ),
    "zh": (
        "回答方式:\n"
        "1. 如果这些函数能完成用户的请求,只回复一个 JSON 数组,列出要进行的调用。"
        '数组的每个元素是只含两个键的对象:"name"(函数名)和 "arguments"'
        "(参数名到参数值的对象)。所有调用放在同一个数组里。示例:{example}\n"
        "2. 如果某个函数适合该请求,但用户没有给出它的某个必需参数的值,"
        "不要调用,也不要编造参数值,而是原样回复下面这句英文,"
        "填入函数名和缺少的参数名(用逗号分隔):\n{sentence}"
    ),
}

# ACEBench's own notation and sentences, which no probe text may hold.
_ACEBENCH = (
    ("ACEBench's call list [ApiName(...)]", re.compile(r"\[\s*\w+\(")),
    ("ACEBench's missing-parameter sentence", re.compile(r"missing necessary parameters", re.IGNORECASE)),
    ("ACEBench's irrelevant sentence", re.compile(r"limitations of the function", re.IGNORECASE)),
    ("ACEBench's placeholder ApiName", re.compile(r"\bApiName\b")),
)

Call = tuple[str, dict]


def tools_text(tools: list[dict]) -> str:
    """One line per function: its OpenAI function object as JSON, in the record's key order."""
    return "\n".join(json.dumps(t["function"], ensure_ascii=False) for t in tools)


def _fill(text: str, **values: str) -> str:
    for key, value in values.items():
        head, _, tail = text.partition("{" + key + "}")
        text = head + value + tail
    return text


def system_prompt(tools: list[dict], lang: str) -> str:
    """The system message of an item: the functions as text, then the format and sentence rules."""
    return "\n\n".join(
        [
            _fill(_PREAMBLE[lang], tools=tools_text(tools)),
            _fill(_RULES[lang], example=_EXAMPLE, sentence=SENTENCE_TEMPLATE),
        ]
    )


def acebench_markers(text: str) -> list[str]:
    """The ACEBench notations or sentences that text holds (names); [] when none."""
    return [name for name, pattern in _ACEBENCH if pattern.search(text)]


# ---- parsing ----------------------------------------------------------------------------------------------

_FENCE = re.compile(r"```[A-Za-z]*[ \t]*\n?(.*?)\n?[ \t]*```", re.DOTALL)


def _no_constant(name: str) -> object:
    raise ValueError(f"{name} is not JSON")


def parse_calls(text: str) -> list[Call] | None:
    """The calls of an answer in the probe's format (module docstring); None when it is not that format."""
    body = text.strip()
    fenced = _FENCE.fullmatch(body)
    if fenced:
        body = fenced.group(1).strip()
    try:
        value = json.loads(body, parse_constant=_no_constant)
    except ValueError:
        return None
    if not isinstance(value, list) or not value:
        return None
    calls = []
    for item in value:
        if (
            not isinstance(item, dict)
            or set(item) != {"name", "arguments"}
            or not isinstance(item["name"], str)
            or not item["name"].strip()
            or not isinstance(item["arguments"], dict)
        ):
            return None
        calls.append((item["name"], item["arguments"]))
    return calls


def _words(text: str) -> str:
    return r"\s+".join(re.escape(word) for word in text.split())


_HEAD, _, _REST = SENTENCE.partition("{function}")
_MIDDLE, _, _TAIL = _REST.partition("{params}")
_SENTENCE = re.compile(
    _words(_HEAD) + r"\s*(?P<function>[^\s:]+?)\s*" + _words(_MIDDLE) + r"\s*(?P<params>[^\n]*)",
    re.IGNORECASE,
)
_QUOTES = "\"'`“”‘’「」[]()（）"


def parse_sentence(text: str) -> tuple[str, list[str]] | None:
    """(function, parameter names) of the first sentence of the probe's form in text; None when there is none,
    or when it names a placeholder."""
    match = _SENTENCE.search(text)
    if match is None:
        return None
    function = match.group("function").strip(_QUOTES)
    rest = match.group("params").strip()
    rest = rest.rstrip(_QUOTES + " ")
    rest = rest[:-1] if rest.endswith((".", "。")) else rest
    params = [p.strip().strip(_QUOTES).strip() for p in re.split(r"\s*(?:,|，|、|;|\band\b)\s*", rest)]
    params = [p for p in params if p]
    if not function or not params or any(re.search(r"\s", p) for p in params):
        return None
    if any(placeholder in function or placeholder in rest for placeholder in PLACEHOLDERS):
        return None
    return function, params


_KWARGS = re.compile(r"\{\s*['\"]?[A-Za-z_][\w.]*['\"]?\s*=(?!=)")


def shapes(text: str) -> dict[str, bool]:
    """The failure shapes of an answer (SHAPE_RULE)."""
    return {
        "kwargs": bool(_KWARGS.search(text)),
        "tool_call": "<tool_call>" in text,
        "placeholder": any(p in text for p in PLACEHOLDERS),
        "runaway": len(text) > RUNAWAY_CHARS,
    }


# ---- scoring ----------------------------------------------------------------------------------------------


def _schema(tools: list[dict], name: str) -> dict:
    for tool in tools:
        if tool["function"]["name"] == name:
            return tool["function"].get("parameters") or {}
    return {}


def normalize_arguments(arguments: dict, schema: dict) -> dict:
    """The schema's defaults filled in, then normalize_value per parameter (module docstring)."""
    defaults = {
        k: v["default"]
        for k, v in (schema.get("properties") or {}).items()
        if isinstance(v, dict) and "default" in v
    }
    return {k: normalize_value(k, v) for k, v in {**defaults, **arguments}.items()}


def _key(call: Call, tools: list[dict]) -> str:
    name, arguments = call
    return canonical_json([name, normalize_arguments(arguments, _schema(tools, name))])


def same_calls(got: list[Call], want: list[Call], tools: list[dict]) -> bool:
    """Whether the two call lists are equal as multisets after normalisation."""
    return Counter(_key(c, tools) for c in got) == Counter(_key(c, tools) for c in want)


def score_answer(item: dict, text: str) -> dict:
    """The score of one answer: id, kind, lang, chars, parsable and exact (normal items; else None), sentence,
    parameter and called (incomplete items; else None), shapes."""
    calls = parse_calls(text)
    score: dict = {
        "id": item["id"],
        "kind": item["kind"],
        "lang": item["lang"],
        "chars": len(text),
        "parsable": None,
        "exact": None,
        "sentence": None,
        "parameter": None,
        "called": None,
    }
    if item["kind"] == "normal":
        want = [(c["name"], c["arguments"]) for c in item["expected"]["calls"]]
        score["parsable"] = calls is not None
        score["exact"] = calls is not None and same_calls(calls, want, item["tools"])
    else:
        found = parse_sentence(text)
        expected = item["expected"]
        score["sentence"] = found is not None
        score["parameter"] = (
            found is not None
            and found[0] == expected["function"]
            and set(found[1]) == set(expected["missing"])
        )
        score["called"] = calls is not None
    score["shapes"] = shapes(text)
    return score


def _rate(count: int, n: int) -> float | None:
    return count / n if n else None


def _kind_summary(scores: list[dict], kind: str) -> dict:
    rows = [s for s in scores if s["kind"] == kind]
    n = len(rows)
    if kind == "normal":
        parsable = sum(bool(s["parsable"]) for s in rows)
        exact = sum(bool(s["exact"]) for s in rows)
        return {
            "n": n,
            "parsable": parsable,
            "parsable_rate": _rate(parsable, n),
            "exact": exact,
            "exact_rate": _rate(exact, n),
            "exact_given_parsable": _rate(exact, parsable),
        }
    sentence = sum(bool(s["sentence"]) for s in rows)
    parameter = sum(bool(s["parameter"]) for s in rows)
    called = sum(bool(s["called"]) for s in rows)
    return {
        "n": n,
        "sentence": sentence,
        "sentence_rate": _rate(sentence, n),
        "parameter": parameter,
        "parameter_rate": _rate(parameter, n),
        "called": called,
        "called_rate": _rate(called, n),
    }


def _shape_counts(rows: list[dict]) -> dict:
    return {"n": len(rows), **{shape: sum(s["shapes"][shape] for s in rows) for shape in SHAPES}}


def summarize(scores: list[dict]) -> dict:
    """{"n", "kinds": {kind: rates}, "languages": {lang: {kind: rates}},
    "shapes": {kind: {lang or "all": counts}}}."""
    return {
        "n": len(scores),
        "kinds": {kind: _kind_summary(scores, kind) for kind in KINDS},
        "languages": {
            lang: {kind: _kind_summary([s for s in scores if s["lang"] == lang], kind) for kind in KINDS}
            for lang in LANGS
        },
        "shapes": {
            kind: {
                **{
                    lang: _shape_counts([s for s in scores if s["kind"] == kind and s["lang"] == lang])
                    for lang in LANGS
                },
                "all": _shape_counts([s for s in scores if s["kind"] == kind]),
            }
            for kind in KINDS
        },
    }


# ---- item check -------------------------------------------------------------------------------------------


def _finite(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    return True


def check_item(item: dict) -> list[str]:
    """Why an item is not a well-formed probe item, as messages; [] when it is."""
    errors = []
    kind, lang = item.get("kind"), item.get("lang")
    if kind not in KINDS or lang not in LANGS:
        return ["kind or lang missing or unknown"]
    for key in ("id", "system", "user"):
        if not isinstance(item.get(key), str) or not item[key].strip():
            errors.append(f"{key} missing or empty")
    tools = item.get("tools")
    if not isinstance(tools, list) or not tools:
        return [*errors, "tools missing"]
    if item.get("messages") != [
        {"role": "system", "content": item.get("system")},
        {"role": "user", "content": item.get("user")},
    ]:
        errors.append("messages must be the system and the user message")
    for key in ("system", "user"):
        for marker in acebench_markers(str(item.get(key, ""))):
            errors.append(f"{key} holds {marker}")
    names = {t["function"]["name"] for t in tools}
    expected = item.get("expected") or {}
    if kind == "normal":
        calls = expected.get("calls")
        if not isinstance(calls, list) or not calls:
            errors.append("expected.calls missing")
        else:
            for call in calls:
                if call.get("name") not in names or not isinstance(call.get("arguments"), dict):
                    errors.append(f"expected call {call.get('name')!r} is not a listed function")
                elif not _finite(call["arguments"]):
                    errors.append(f"expected call {call['name']!r} has a non-finite number")
    else:
        function, missing = expected.get("function"), expected.get("missing")
        if function not in names:
            errors.append(f"expected function {function!r} is not listed")
        elif not isinstance(missing, list) or not missing:
            errors.append("expected.missing missing")
        else:
            required = set(_schema(tools, function).get("required") or [])
            if not set(missing) <= required:
                errors.append(f"missing {missing} are not all required parameters of {function}")
    return errors

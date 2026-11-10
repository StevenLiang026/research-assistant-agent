"""The refusal development sets of M8 (docs/m8-计划.md WP2, 拒答开发集与校准): their items, the scorer of an
answer (strict and lenient, its failure mode), and the summary of a run; shared by
scripts/run_special_dev.py, which runs a system over a set, and scripts/calibrate_special_dev.py, which checks
dev set B against ACEBench's failure pattern on B0 and FT3. Nothing here sends a request or reads ACEBench.

Items. A set is JSONL, one item per line (scripts/build_special_mix.py writes data/eval/special-dev-a.jsonl
and the locked special-dev-b.jsonl):

    {"id": "sdev-b-0001", "split": "a" | "b", "lang": "zh" | "en", "kind": <KINDS>,
     "messages": [{"role": "system", "content": <the rules and the function list as text>},
                  {"role": "user", "content": ...}, ...],
     "call_format": "python" | "json" | "yaml", "family": <the sentence family of the item's prompt>,
     "expected": <below>, "source": {"dataset": ..., "record": ...}}

The messages are sent as they are, without a tools field: the functions are in the system message, as
ACEBench gives them, and the last message is the user's. kind is one of the special kinds, incorrect (a value
is present but invalid), missing (a required parameter has no value) and irrelevant (no listed function
fits), each answered with the sentence the system message prescribes for that case; priority (an invalid
value and a missing parameter together), answered with the sentence of the case the prompt orders first,
reported and never gated; or one of the valid kinds, valid_verbatim, valid_multiturn and valid_nearmiss,
answered with the calls written in call_format, the notation of scripts/render_format_data.py's serialisers.
expected is

    {"type": "refusal", "case": "incorrect" | "missing" | "irrelevant", "fixed": [the sentence's fixed
     words, in order], "values": [each invalid value, verbatim], "params": [each parameter to name: the
     invalid ones, as many as the values and in their order, or exactly the missing top-level ones],
     "api": <the function, for missing>}
    {"type": "call", "calls": [{"name": ..., "arguments": {...}}, ...], "verbatim": [values that must
     appear unchanged]}

check_item says why an item is malformed: a field missing or unknown, a kind and a case that disagree, a
case without what it names, an incorrect case with not as many values as parameters, a sentence whose fixed
words do not stand in order on one line of the system message or whose slots cannot be told apart
(SLOT_RULE), a filler that could not be told from a placeholder of that sentence, expected calls that cannot
be written in the item's notation or that hold no verbatim value, and ACEBench's text anywhere in the
messages (acebench_text: ace_probe.acebench_markers, and the incorrect-value sentence and the history heading
that M8's rules name).

Scoring (RULES). A refusal item is scored strict (the item's sentence alone, its slots filled right:
STRICT_RULE) and lenient (ACEBench's substring rule with the item's sentence: LENIENT_RULE), with the first
failure that applies (FAILURE_RULE); a valid item by its exact call (CALL_RULE), whether it was refused (an
answer without a call, whatever its wording: FALSE_REFUSAL_RULE) and whether a verbatim value was altered.
A call is found in the three notations, in a few more, and loosely (CALL_RULE), so that a call written in
any common way counts as one, the calls before the cut of a greedy loop that the answer budget ends among
them; an answer the parsers cannot read (nested deeper than they go, an integer too long to write) holds no
call where they fail, and nothing it holds stops the scorer. Which sentences a set holds is read from its
refusal items, family by family and case by case, every placeholder style a family is written in kept
(sentence_families): a wrong case is told by them, and the form of a false refusal by them and by the
training families of research_agent.data.special_mix (training_sentences). score_items and score_records
score with the sentences of the whole set and keep a failure of the scorer on one item to that item (status
"scoring_error"); score_item alone knows the item's own sentence only.

Sentences (SENTENCE_RULE, NAME_RULE, SLOT_RULE). An answer holds a sentence when the sentence's fixed words
stand in it in order. Of the placements, the scorer takes the shortest, preferring one whose words neither
cut into an ASCII word (a dotted name such as date.name, a number such as 2.5) nor overlap a value or a name
the sentence is to be filled with, and one on a single line, so that the text between the fixed words, the
slots, holds what the model filled in. The template is the sentence as the system message writes it
(sentence_template): its slots are its placeholders, and so are the bracketed or capitalised tokens right
before its first and right after its last fixed word, the template's edge slots (PLACEHOLDER_RULE); each
slot's placeholder tells what it holds (slot_roles).

Summary (summarize): per group, the special kinds pooled ("special"), priority, and the valid kinds pooled
("valid"), overall, per kind ("kinds") and per language ("languages"): the strict and lenient counts and
rates and the failure counts of the refusal groups; the exact calls, the false refusals (in all, and by
their form), the zh alterations (among the answers whose calls could be read and whose items hold verbatim
values) and the failure counts of the valid ones; the scoring errors of each group and of the run.
"""

from __future__ import annotations

import ast
import bisect
import functools
import importlib.util
import itertools
import json
import re
import sys
import unicodedata
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ..normalize import normalize_text
from ..toolcalls import OPEN as TOOL_CALL_OPEN
from ..toolcalls import parse_tool_calls
from .ace_probe import acebench_markers, same_calls

SPLITS = ("a", "b")
LANGS = ("zh", "en")
CASES = ("incorrect", "missing", "irrelevant")
SPECIAL_KINDS = CASES  # one kind per case, answered with that case's sentence: the gated refusal kinds
PRIORITY = "priority"  # an invalid value and a missing parameter together: reported, never gated
PRIORITY_CASES = ("incorrect", "missing")
VALID_KINDS = ("valid_verbatim", "valid_multiturn", "valid_nearmiss")
REFUSAL_KINDS = (*SPECIAL_KINDS, PRIORITY)
KINDS = (*REFUSAL_KINDS, *VALID_KINDS)
FORMATS = ("python", "json", "yaml")  # scripts/render_format_data.py's notations
NOTATIONS = (*FORMATS, "python_list", "tool_call", "loose")  # where a call can be found (CALL_RULE)
ROLES = ("system", "user", "assistant", "tool")
# case -> the role of the slot its placeholder marks, the role of the other slot (SLOT_RULE)
SLOT_ROLES = {"incorrect": ("value", "param"), "missing": ("api", "params"), "irrelevant": ()}
REFUSAL_FAILURES = ("no_answer", "empty", "called", "wrong_case", "no_sentence", "placeholder_left",
                    "wrong_values", "wrong_params", "extra_text", "scoring_error")
CALL_FAILURES = ("no_answer", "empty", "refused", "with_text", "other_notation", "zh_altered", "wrong_call",
                 "no_call", "scoring_error")
REFUSAL_FORMS = ("sentence", "other_text", "empty")  # how a valid item was refused (FALSE_REFUSAL_RULE)
SEPARATOR = ", "  # between the values or the names of one slot, as research_agent.data.special_mix fills them
SENTENCE_MARKS = ".。!！?？"  # one of them may end an answer or not (STRICT_RULE)
OPENING = "([{<（【「『《〈＜〔"  # the brackets a placeholder may be written in
CLOSING = ")]}>）】」』》〉＞〕"
MAX_ORDERS = 5  # the most values or names of one slot read in every order; more, in the expected one only
MAX_VALUES = 10_000  # the most values (objects, lists and scalars) read in the calls of one answer
MAX_CALL_CHARS = 4000  # the longest Python call the loose pass reads
MAX_CALL_DEPTH = 64  # the deepest its brackets nest
# The deepest a YAML block may nest (_yaml_depth) to be read by PyYAML: its C loader overflows the stack,
# which ends the process, at about 4,000 levels (2026-11-02, PyYAML 6.0.3); no call nests so deep.
MAX_YAML_DEPTH = 256

SENTENCE_RULE = (
    "An answer holds a sentence when the sentence's fixed words (expected.fixed) stand in it in order, each "
    "after the one before, as plain case-sensitive substrings. Of the placements, the shortest is taken, "
    "preferring, in this order, one whose words neither cut into an ASCII word (its letters, digits and "
    "underscores, and a dot between two of them: date.name, 2.5) nor overlap an occurrence of a value or a "
    "name the sentence is to be filled with and that lies on one line, then one of the first two kinds on "
    "several lines, then one whose words merely avoid the fillers, then any. Its slots are the "
    "text between consecutive fixed words, and its edge slots, when the template has them, the text on the "
    "line before the first fixed word (from the last full stop) and after the last one (to the first full "
    "stop); the rest of the answer, besides the fixed words and those full stops, is outside the sentence. "
    "A set's sentences are read from its refusal items by family and case, every fixed-word list of a "
    "family kept (a family written in several placeholder styles has one per style), and no fixed-word list "
    "may stand for two cases."
)
NAME_RULE = (
    "A parameter or a function is named when its name stands on its own: no ASCII letter, digit, underscore "
    "or dot right before it, and neither such a character nor a dot followed by one right after it (address "
    "in address.city is not named). A value appears when it is a plain substring."
)
PLACEHOLDER_RULE = (
    "The template is the shortest placement of the sentence's fixed words on one line of the system "
    "message, preferring one whose words do not cut into an ASCII word. Its slots, stripped, are its "
    "placeholders, and so are its edge placeholders: a token right before its first or right after its last "
    "fixed word, quotes and blanks aside, written as (x), <x>, {x}, [x], {{x}}, <<x>>, ${x}, $x, %x%, __x__, "
    "a full-width bracket pair or a word in capitals. An answer leaves a placeholder when it holds verbatim, "
    "outside the values and names it is filled with, an edge placeholder or a slot that shows itself as one "
    "(it holds a punctuation mark or a symbol, or it is written in capitals), or when a slot that is a plain "
    "word fills the same slot of the answer's sentence alone (so a plain-word placeholder that is also the "
    "filler of its own slot reads as left: the items do not say which filler goes to which slot)."
)
SLOT_RULE = (
    "Every slot of a template, its edge slots included, holds one filler, told by its placeholder: in an "
    "incorrect sentence the slot whose placeholder names a value ('value' in any letter case, or 值) holds "
    "the invalid values and the other slot their parameters; in a missing sentence the slot whose "
    "placeholder names a function ('func', 'api' as a word, 函数 or 接口) holds the function (api) and the "
    "other slot the missing parameters; an irrelevant sentence has no slot. Several values or names in one "
    "slot are joined by ', ', as research_agent.data.special_mix writes its answers."
)
CALL_RULE = (
    "Exact call (valid items): the answer, stripped and with at most one surrounding Markdown code fence "
    "removed, parses whole in the item's call_format with scripts/render_format_data.py's parser, and its "
    "calls equal the expected ones as a multiset after normalisation (research_agent.eval.ace_probe."
    "same_calls; the items carry no tool schema, so no default is filled in; calls it cannot compare are not "
    "the expected ones). A call is found in an answer when the answer so read parses in any of the three "
    "notations or as a Python list of calls [f(k=v), ...] (each call read from its own text), when it holds "
    "a <tool_call> block, or when a fenced block, a line, a JSON array or a YAML list (from a line '- name:' "
    "to the end of its block) inside it parses, the YAML list whole or else item by item (the calls of the "
    "items that parse: a greedy loop of calls that the answer budget cuts leaves its last item unfinished); "
    "otherwise loosely (the notation 'loose'): a JSON object with a name and arguments or parameters (an "
    "object, or the JSON text of one), or an array holding such objects, anywhere in the answer, fenced or "
    "not, a function definition (its parameters a schema with properties) aside; a Python call with at least "
    "one literal argument, positional or keyword, of one of the item's functions (the names its system "
    "message gives functions, by \"name\" or by a ### heading, and those its expectation names), anywhere; a "
    "YAML block from a line '- name:' or 'name:' to the end of its block (its indented lines, and its list "
    "items, or its keys up to the next 'name:') holding such objects, read whole or else item by item, with "
    "PyYAML's safe loader when the block nests no deeper than MAX_YAML_DEPTH levels (deeper, PyYAML's C "
    "loader ends the process, and no call nests so deep), dates left as text, or without PyYAML with the "
    "subset after its bare scalars are quoted. What a parser cannot read (nested deeper than it goes, an "
    "integer too long to write) holds no call for it. Every block, line and \"name\" key is read once, a "
    "YAML block twice at most, so the search is linear in the answer."
)
STRICT_RULE = (
    "Refusal items (the special kinds and priority), by expected.case: no call is found (CALL_RULE), and the "
    "answer, its surrounding whitespace aside and one sentence mark at its end (. ! ? or the full-width "
    "。！？) in place of the sentence's own final mark or none, is exactly the item's sentence with every "
    "slot filled (SLOT_RULE): the fixed words in order and around them only the slots, whitespace where a "
    "slot meets a fixed word aside, each slot holding its filler bare or in its placeholder's own brackets "
    "when the fixed words leave them out. Incorrect: the value slot holds exactly the expected values and "
    "the parameter slot exactly the expected parameters, both in one order that keeps each value with its "
    "parameter; missing: the function slot holds the api and the parameter slot exactly the expected "
    "top-level parameters, in any order; irrelevant: the sentence alone. So no other text, no second "
    "sentence and no leftover placeholder. Anything outside the sentence (SENTENCE_RULE) but whitespace, a "
    "mark or a symbol as much as a word, is counted as extra_text too, the one sentence mark the answer may "
    "end with aside. Where the sentence is looked for, the answer's end is read the same way."
)
LENIENT_RULE = (
    "Refusal items: ACEBench's substring rule (its eval_main.py, special_eval, which looks for a phrase and "
    "then for the values or the names, never for brackets) with the item's sentence: its fixed words, each "
    "stripped of the bracket characters at its ends that belong to the placeholders beside it (closing ones "
    "at its start, opening ones at its end), the last one of the sentence's final mark too, a word left "
    "empty skipped and the whitespace collapsed (lenient_words), stand in the answer, its whitespace "
    "collapsed, in order; and incorrect: every expected value stands anywhere in the answer; missing: the "
    "function (api) and every expected parameter; irrelevant: nothing more. Plain case-sensitive substrings, "
    "as ACEBench's; a call or a placeholder in the answer does not matter."
)
FAILURE_RULE = (
    "Refusal items, the first that applies, none when strict: no_answer (the run recorded none), empty, "
    "called (a call is found, a loose one included), wrong_case (the item's sentence does not stand in the "
    "answer, another case's sentence of the set does: of any family, in any of its styles), no_sentence, "
    "placeholder_left, wrong_values (incorrect: the value slot does not hold the values), wrong_params (the "
    "parameter slot does not hold the parameters, kept with their values; missing: or the function slot "
    "not the api), extra_text (the sentence is filled right, but the answer is not that sentence alone: "
    "other text, a mark or a symbol, a second sentence). Valid items, none when the call is exact: "
    "no_answer, empty, zh_altered or wrong_call (the answer parses whole in call_format), with_text (a call "
    "in call_format inside other text), other_notation (a call in another notation, a loose one included), "
    "refused (no call, and a refusal sentence of a known family stands in the answer: FALSE_REFUSAL_RULE), "
    "no_call (no call, other text). Either kind: scoring_error, the scorer failed on the item, which then "
    "counts as neither strict nor lenient, nor exact nor refused. zh_altered, for a valid item with verbatim "
    "values whose answer holds a call (a loose one included): some verbatim value is inside no argument "
    "value of the calls, both read with research_agent.normalize.normalize_text (the first MAX_VALUES values "
    "of the calls read; the same containment tells called_with_invalid)."
)
FALSE_REFUSAL_RULE = (
    "A valid item is refused (false_refusal) when its answer holds no call (CALL_RULE, loose calls "
    "included), whatever its wording. Its form (refusal_form): 'sentence' when a refusal sentence of a known "
    "family stands in it (one of the set's, or of research_agent.data.special_mix.FAMILIES in any "
    "placeholder style: training_sentences), 'empty' when it is empty or blank, 'other_text' otherwise. An "
    "answer the run did not record (no_answer) is no refusal. A summary's false_refusals counts every form, "
    "false_refusals_by each."
)
RULES = {"sentence": SENTENCE_RULE, "names": NAME_RULE, "placeholders": PLACEHOLDER_RULE, "slots": SLOT_RULE,
         "calls": CALL_RULE, "strict": STRICT_RULE, "lenient": LENIENT_RULE, "failures": FAILURE_RULE,
         "false_refusals": FALSE_REFUSAL_RULE}

Call = tuple[str, dict]
Span = tuple[int, int]
Families = Mapping[tuple[str, str], frozenset[tuple[str, ...]]]  # (family, case) -> its fixed-word lists

# ACEBench's text that M8's rules name besides ace_probe's markers: its incorrect-value sentence and the
# heading of its conversation histories (docs/m8-计划.md WP1, 禁止 ACEBench 文本).
_ACEBENCH = (
    ("ACEBench's incorrect-value sentence", re.compile(r"there\s+is\s+incorrect\s+value", re.IGNORECASE)),
    ("ACEBench's history heading", re.compile(r"conversation\s+history\s*1\s*\.\.\s*t", re.IGNORECASE)),
    ("ACEBench's history heading", re.compile(r"对话历史\s*1\s*\.\.\s*t")),
)


def acebench_text(text: str) -> list[str]:
    """The ACEBench notations and sentences text holds, by name ([] when none): ace_probe.acebench_markers'
    and the incorrect-value sentence and the history heading of M8's rules."""
    return [*acebench_markers(text), *(name for name, pattern in _ACEBENCH if pattern.search(text))]


# ---- the notations ----------------------------------------------------------------------------------------

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "render_format_data.py"


@functools.cache
def notations():
    """scripts/render_format_data.py, whose serialisers and parsers define the three call notations.
    scripts/ is no package: the file is loaded on first use under a private name, as scripts/run_acebench.py
    loads scripts/run_bfcl.py."""
    spec = importlib.util.spec_from_file_location("_special_dev_render_format_data", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_FENCE = re.compile(r"```[A-Za-z]*[ \t]*\n?(.*?)\n?[ \t]*```", re.DOTALL)  # ace_probe's code fence
_JSON = json.JSONDecoder()


def _parsed(text: str, fmt: str) -> list[Call] | None:
    """text, whole, as the calls of the notation fmt; None when it is not that notation, a nesting deeper
    than the parsers go included (CPython's parser stops with a MemoryError at some hundreds of levels, the
    recursive readers with a RecursionError). The parsers compile Python, which warns of an invalid escape in
    a string: the answer's text, not this program's, so the warning is not shown."""
    if not text.strip():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            return notations().parse_calls(text, fmt)
        except (ValueError, SyntaxError, RecursionError, MemoryError):
            return None


def _unfenced(text: str) -> str:
    body = text.strip()
    fenced = _FENCE.fullmatch(body)
    return fenced.group(1).strip() if fenced else body


def parse_answer(text: str, fmt: str) -> list[Call] | None:
    """The calls of an answer written in the notation fmt (CALL_RULE): the answer stripped, at most one
    surrounding Markdown code fence removed, parsed whole; None when it is not that notation."""
    return _parsed(_unfenced(text), fmt)


def _sources(source: str, nodes: Sequence[ast.expr]) -> list[str]:
    """The text of each node of the parsed source, as ast.get_source_segment gives it (the parser's
    positions count the bytes of UTF-8 lines), the source split into lines once."""
    lines = source.encode("utf-8").splitlines(keepends=True)  # at \n, \r\n and \r, as the parser splits
    out = []
    for node in nodes:
        first, last = node.lineno - 1, node.end_lineno - 1
        if first == last:
            text = lines[first][node.col_offset:node.end_col_offset]
        else:
            text = b"".join([lines[first][node.col_offset:], *lines[first + 1:last],
                             lines[last][:node.end_col_offset]])
        out.append(text.decode("utf-8"))
    return out


def _python_list(text: str) -> list[Call] | None:
    """A Python list or tuple of calls with keyword arguments, [f(k=v), ...]: no notation of the prompts,
    but calls all the same, each read from its own text (written back, an integer too long to write in
    decimal or a value nested deeper than the writer goes would stop it); None otherwise."""
    source = text.strip()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            node = ast.parse(source, mode="eval").body
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return None
    if not isinstance(node, ast.List | ast.Tuple) or not node.elts:
        return None
    if not all(isinstance(element, ast.Call) for element in node.elts):
        return None
    # each call in parentheses: one the list's brackets continued over several lines stays one statement
    return _parsed("\n".join(f"({call})" for call in _sources(source, node.elts)), "python")


def _tool_calls(text: str) -> list[Call] | None:
    """The trained <tool_call> blocks of the text (research_agent.toolcalls), arguments that are not an
    object kept under the key ""; None without a block, or when a block cannot be read (JSON nested deeper
    than the reader goes, an integer too long to read)."""
    if TOOL_CALL_OPEN not in text:
        return None
    try:
        parsed = parse_tool_calls(text)
    except (ValueError, RecursionError, MemoryError):
        return None
    calls = []
    for call in parsed:
        arguments = call.get("arguments")
        calls.append((str(call.get("name") or ""),
                      arguments if isinstance(arguments, dict) else {"": arguments}))
    return calls or None


def _whole(text: str, first: str | None = None) -> tuple[str, list[Call]] | None:
    """(notation, calls) of a text that is calls as a whole, at most one surrounding fence removed: in a
    notation (first, when given, before the others), or as a Python list of calls."""
    body = _unfenced(text)
    order = FORMATS if first is None else (first, *(fmt for fmt in FORMATS if fmt != first))
    for fmt in order:
        calls = _parsed(body, fmt)
        if calls:
            return fmt, calls
    calls = _python_list(body)
    return ("python_list", calls) if calls else None


_LIST_START = re.compile(r"- name:(?:[ \t]|$)")
_MAP_START = re.compile(r"name:(?:[ \t]|$)")
_TOP_KEY = re.compile(r"""(?:[A-Za-z_][\w.\-]*|"[^"\n]*"|'[^'\n]*')[ \t]*:(?:[ \t]|$)""")


def _yaml_blocks(text: str, *, mappings: bool) -> Iterator[str]:
    """The YAML blocks of a text (CALL_RULE): from a line '- name:' (or, with mappings, 'name:') to the end
    of its block, the lines after it that are indented, the list items of a list, the keys of a mapping up
    to its next 'name:' (a mapping is one call, and the next 'name:' starts the next); a blank line or any
    other line ends it. Each line is read once."""
    lines = text.split("\n")
    index = 0
    while index < len(lines):
        listed = _LIST_START.match(lines[index]) is not None
        if not listed and not (mappings and _MAP_START.match(lines[index])):
            index += 1
            continue
        end = index + 1
        while end < len(lines) and lines[end].strip() and (
                lines[end][0] in " \t"
                or (listed and (lines[end] == "-" or lines[end].startswith("- ")))
                or (not listed and _TOP_KEY.match(lines[end]) is not None
                    and _MAP_START.match(lines[end]) is None)):
            end += 1
        yield "\n".join(lines[index:end])
        index = end


def _yaml_items(block: str) -> list[str]:
    """The items of a YAML block: of a list, each item from its line '- ' (or '-') to the next one; a
    mapping is its one item."""
    if not block.startswith("-"):
        return [block]
    items: list[list[str]] = []
    for line in block.split("\n"):
        if not items or line == "-" or line.startswith("- "):
            items.append([])
        items[-1].append(line)
    return ["\n".join(item) for item in items]


def _block_calls(block: str, read: Callable[[str], list[Call] | None]) -> list[Call] | None:
    """The calls of a YAML block (CALL_RULE), read whole or, when it does not read so, item by item, the
    calls of the items that read kept in order: a greedy loop of calls that the answer budget cuts leaves
    its last item unfinished, and the items before it are calls all the same. Each line is read twice at
    most."""
    calls = read(block)
    if calls:
        return calls
    items = _yaml_items(block)
    if len(items) < 2:
        return None
    return [call for item in items for call in read(item) or ()] or None


def _yaml_subset(block: str) -> list[Call] | None:
    return _parsed(block, "yaml")


def _json_list(value: list) -> list[Call] | None:
    """A JSON array read from an answer as the calls of the notation json (scripts/render_format_data.py);
    None when it is not, or when it cannot be written back (nested deeper than the writer goes)."""
    try:
        text = json.dumps(value, ensure_ascii=False)
    except (ValueError, RecursionError, MemoryError):
        return None
    return _parsed(text, "json")


def _embedded(text: str) -> tuple[str, list[Call]] | None:
    """(notation, calls) of a call written inside other text: a fenced block, a line, a JSON array, a YAML
    list starting at a line '- name:' (read item by item when it does not read whole: _block_calls); None
    when there is none. Each block, line and array is read once, a YAML list twice at most."""
    for block in _FENCE.finditer(text):
        found = _whole(block.group(1))
        if found is not None:
            return found
    for line in text.split("\n"):
        if "(" in line:
            calls = _parsed(line, "python")
            if calls:
                return "python", calls
            calls = _python_list(line)
            if calls:
                return "python_list", calls
    last = text.rfind("]")  # an array that opens after the last closing bracket never closes: none is read
    at = text.find("[", 0, max(last, 0))
    while at >= 0:
        try:
            value, end = _JSON.raw_decode(text, at)
        except (ValueError, RecursionError):
            at = text.find("[", at + 1, last)
            continue
        if isinstance(value, list) and value:
            calls = _json_list(value)
            if calls:
                return "json", calls
        at = text.find("[", end, last)  # an array read whole: the arrays inside it were read with it
    for block in _yaml_blocks(text, mappings=False):
        calls = _block_calls(block, _yaml_subset)
        if calls:
            return "yaml", calls
    return None


# ---- the loose calls (CALL_RULE) --------------------------------------------------------------------------

_NAME_KEY = re.compile(r'"name"\s*:')


def _call_object(value: object) -> Call | None:
    """A call written as an object: {"name": ..., "arguments" or "parameters": an object, or the JSON text
    of one}; None for anything else, a function definition (its parameters a schema) included."""
    if not isinstance(value, dict) or not isinstance(value.get("name"), str) or not value["name"].strip():
        return None
    key = "arguments" if "arguments" in value else "parameters"
    arguments = value.get(key)
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, RecursionError):
            return None
    if not isinstance(arguments, dict):
        return None
    if key == "parameters" and isinstance(arguments.get("properties"), dict):
        return None  # a function definition: its parameters are a JSON schema
    return value["name"], arguments


def _call_objects(value: object) -> list[Call] | None:
    """The calls of a value read from JSON or YAML: a call object, or the call objects of a list."""
    found = [_call_object(item) for item in value] if isinstance(value, list) else [_call_object(value)]
    calls = [call for call in found if call is not None]
    return calls or None


def _json_starts(text: str, at: int) -> Iterator[int]:
    """Where a JSON value holding the key "name" at `at` may start: the three braces nearest before it, the
    innermost first, each after the bracket of an array that opens right before it."""
    end = at
    for _ in range(3):
        brace = text.rfind("{", 0, end)
        if brace < 0:
            return
        before = brace - 1
        while before >= 0 and text[before] in " \t\r\n":
            before -= 1
        if before >= 0 and text[before] == "[":
            yield before
        yield brace
        end = brace


def _loose_json(text: str) -> list[Call] | None:
    """The calls of the first JSON value holding a "name" key that is a call object or holds some; a value
    that opens after the last brace or bracket that could close it is not read (it cannot close)."""
    last = {"{": text.rfind("}"), "[": text.rfind("]")}
    for match in _NAME_KEY.finditer(text):
        for start in _json_starts(text, match.start()):
            if start > last[text[start]]:
                continue
            try:
                value, end = _JSON.raw_decode(text, start)
            except (ValueError, RecursionError):
                continue
            calls = _call_objects(value) if end > match.start() else None
            if calls:
                return calls
    return None


def _python_name(name: str) -> bool:
    return bool(name) and all(part.isidentifier() for part in name.split("."))


def _dotted(node: ast.expr) -> str | None:
    """The name a call's function is written as (f, or a.b.f); None for anything else."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    return ".".join([node.id, *reversed(parts)]) if isinstance(node, ast.Name) else None


_CALL_CHARS = re.compile(r"[\"'\\()\[\]{}]")  # what decides where a Python call closes


def _closing(text: str, start: int) -> int | None:
    """The index of the parenthesis that closes the one at start, string literals skipped; None when another
    bracket closes first, the brackets nest deeper than MAX_CALL_DEPTH or none closes within MAX_CALL_CHARS
    characters."""
    depth, quote, skip = 0, None, -1
    for match in _CALL_CHARS.finditer(text, start, min(len(text), start + MAX_CALL_CHARS)):
        index, char = match.start(), match.group()
        if index == skip:  # the character a backslash escapes
            continue
        if quote is not None:
            if char == "\\":
                skip = index + 1
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in "([{":
            depth += 1
            if depth > MAX_CALL_DEPTH:
                return None
        elif char in ")]}":
            depth -= 1
            if depth == 0:
                return index if char == ")" else None
    return None


def _python_call(source: str, name: str) -> Call | None:
    """source as a call of the function name with at least one argument, every argument a literal; the
    positional ones kept under their positions ("0", "1", ...). None otherwise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        try:
            node = ast.parse(source, mode="eval").body
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return None
    if not isinstance(node, ast.Call) or _dotted(node.func) != name or not (node.args or node.keywords):
        return None
    given = [*((str(number), arg) for number, arg in enumerate(node.args)),
             *((keyword.arg, keyword.value) for keyword in node.keywords)]
    arguments: dict = {}
    for key, value in given:
        if key is None:  # f(**mapping)
            return None
        try:
            arguments[key] = ast.literal_eval(value)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return None
    return name, arguments


def _loose_python(text: str, functions: Iterable[str]) -> list[Call] | None:
    """The Python calls of the functions anywhere in the text, in the text's order."""
    found = []
    for name in sorted({name for name in functions if _python_name(name)}):
        for match in re.finditer(r"(?<![A-Za-z0-9_.])" + re.escape(name) + r"\s*\(", text):
            close = _closing(text, match.end() - 1)
            call = None if close is None else _python_call(text[match.start():close + 1], name)
            if call is not None:
                found.append((match.start(), call))
    return [call for _, call in sorted(found, key=lambda pair: pair[0])] or None


@functools.cache
def _yaml_load() -> Callable[[str], object] | None:
    """PyYAML's safe loading without its dates (a date stays the text a call gives it), as a function of a
    text; None without PyYAML."""
    try:
        import yaml
    except ImportError:
        return None

    class Loader(getattr(yaml, "CSafeLoader", yaml.SafeLoader)):
        pass

    Loader.yaml_implicit_resolvers = {
        first: [(tag, pattern) for tag, pattern in resolvers if tag != "tag:yaml.org,2002:timestamp"]
        for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()}

    def load(text: str) -> object:
        return yaml.load(text, Loader=Loader)

    return load


_BARE_ENTRY = re.compile(r"(?P<head>[ \t]*(?:-[ \t]+)?(?:[A-Za-z_][A-Za-z0-9_.]*|\"[^\"\n]*\")[ \t]*:[ \t]+)"
                         r"(?P<value>\S.*?)[ \t]*")
_BARE_ITEM = re.compile(r"(?P<head>[ \t]*-[ \t]+)(?P<value>[^\s:][^:]*?)[ \t]*")


def _quoted(line: str) -> str:
    """A line of a YAML block with its bare scalar, after a key or after an item's dash, written as a JSON
    string, so that the subset of scripts/render_format_data.py reads it."""
    match = _BARE_ENTRY.fullmatch(line) or _BARE_ITEM.fullmatch(line)
    if match is None:
        return line
    try:
        json.loads(match.group("value"))
    except (ValueError, RecursionError):  # no JSON scalar, or JSON nested deeper than the reader goes
        return match.group("head") + json.dumps(match.group("value"), ensure_ascii=False)
    return line


def _subset_yaml(block: str) -> list[Call] | None:
    """A YAML block's calls without PyYAML: its bare scalars quoted, a mapping made the one item of a list,
    an item's parameters read as its arguments, then the subset of scripts/render_format_data.py."""
    lines = [_quoted(line) for line in block.split("\n")]
    if not lines[0].startswith("-"):
        lines = ["- " + lines[0], *("  " + line for line in lines[1:])]
    return _parsed("\n".join(re.sub(r"^  parameters:", "  arguments:", line) for line in lines), "yaml")


_FLOW_BRACKET = re.compile(r"[\[\]{}]")


def _yaml_depth(block: str) -> int:
    """How deep a YAML block's collections nest, give or take a level: the most leading spaces, tabs, dashes
    and question marks of a line (its indentation and the compact items '- - -' it opens) plus the most flow
    brackets ([ and {) open at once, quotes not read."""
    prefix = max(len(line) - len(line.lstrip(" \t-?")) for line in block.split("\n"))
    depth = deepest = 0
    for bracket in _FLOW_BRACKET.finditer(block):
        depth = depth + 1 if bracket.group() in "[{" else max(depth - 1, 0)
        deepest = max(deepest, depth)
    return prefix + deepest


def _loose_yaml(block: str) -> list[Call] | None:
    """A YAML block's calls (CALL_RULE), read with PyYAML when it is installed and the block nests no deeper
    than MAX_YAML_DEPTH (_yaml_depth), with the subset otherwise; None when it holds none."""
    load = _yaml_load()
    if load is None:
        return _subset_yaml(block)
    if _yaml_depth(block) > MAX_YAML_DEPTH:
        return None  # PyYAML's C loader would overflow the stack, and no call nests so deep
    try:
        value = load(block)
    except Exception:  # noqa: BLE001  deliberate: any text a model wrote, read as YAML; none is a call
        return None
    return _call_objects(value)


def _loose(text: str, functions: Iterable[str]) -> list[Call] | None:
    """The calls only the loose pass finds (CALL_RULE): JSON, then Python, then YAML (each block read whole or
    item by item: _block_calls)."""
    calls = _loose_json(text) or _loose_python(text, functions)
    if calls:
        return calls
    for block in _yaml_blocks(text, mappings=True):
        calls = _block_calls(block, _loose_yaml)
        if calls:
            return calls
    return None


def find_calls(text: str, fmt: str, functions: Iterable[str] = ()) -> tuple[str, list[Call]] | None:
    """(notation, calls) of the calls an answer holds (CALL_RULE), the item's notation fmt tried first and
    the loose shapes last, the Python calls of the functions given (item_functions) among them; None when it
    holds none. The notation is one of NOTATIONS."""
    found = _whole(text, fmt)
    if found is not None:
        return found
    calls = _tool_calls(text)
    if calls:
        return "tool_call", calls
    found = _embedded(text)
    if found is not None:
        return found
    calls = _loose(text, functions)
    return ("loose", calls) if calls else None


_DEFINED = re.compile(r"""["']name["']\s*:\s*["']([^"'\\\n]+)["']""")
_HEADING = re.compile(r"^###[ \t]+(.+?)[ \t]*$", re.MULTILINE)


def item_functions(item: dict) -> frozenset[str]:
    """The functions an item shows or expects (CALL_RULE): every name its system message gives a function
    ("name": ... in JSON or a Python literal, a ### heading in Markdown), the function a missing case names
    and the names of the expected calls."""
    system = _system(item) or ""
    names = {*_DEFINED.findall(system), *_HEADING.findall(system)}
    expected = item.get("expected")
    if isinstance(expected, dict):
        if isinstance(expected.get("api"), str):
            names.add(expected["api"])
        names.update(call["name"] for call in expected.get("calls") or []
                     if isinstance(call, dict) and isinstance(call.get("name"), str))
    names.discard("")
    return frozenset(names)


def _leaves(value: object) -> Iterator[str]:
    """Every scalar inside a value, as text, in order: a string as it is, a JSON scalar as JSON writes it
    (an integer too long to write in decimal gives no text), anything else as str writes it; however deep
    the value, and the first MAX_VALUES values alone (objects and lists counted), so that a value that holds
    itself (a YAML alias inside its own anchor) or that a few aliases multiply ends all the same."""
    stack = [value]
    for _ in range(MAX_VALUES):
        if not stack:
            return
        item = stack.pop()
        if isinstance(item, dict):
            stack.extend(reversed(list(item.values())))
        elif isinstance(item, list | tuple | set | frozenset):
            stack.extend(reversed(list(item)))
        elif isinstance(item, str):
            yield item
        elif item is None or isinstance(item, bool | int | float):
            try:
                text = json.dumps(item)
            except ValueError:  # more digits than Python writes (sys.get_int_max_str_digits)
                continue
            yield text
        else:
            yield str(item)


def _kept(value: str, calls: Sequence[Call]) -> bool:
    """Whether value appears unchanged inside an argument value of the calls, both read with
    normalize_text (_leaves: the first MAX_VALUES values of the arguments)."""
    wanted = normalize_text(value)
    return any(wanted in normalize_text(leaf) for leaf in _leaves([arguments for _, arguments in calls]))


# ---- sentences --------------------------------------------------------------------------------------------

_WORD_CHAR = re.compile(r"[A-Za-z0-9_]")


def _wordy(char: str) -> bool:
    return bool(_WORD_CHAR.fullmatch(char))


def _in_word(text: str, index: int) -> bool:
    """Whether text[index] belongs to an ASCII word: an ASCII letter, digit or underscore, or a dot between
    two of them (the dots of date.name and of 2.5: NAME_RULE)."""
    char = text[index]
    return _wordy(char) or (char == "." and 0 < index < len(text) - 1
                            and _wordy(text[index - 1]) and _wordy(text[index + 1]))


def _bounded(text: str, start: int, end: int) -> bool:
    """Whether text[start:end] does not cut into an ASCII word (_in_word): where its first or last character
    belongs to one, so does not the character next to it."""
    return not ((start > 0 and _in_word(text, start) and _in_word(text, start - 1))
                or (end < len(text) and _in_word(text, end - 1) and _in_word(text, end)))


def _overlapping(spans: Sequence[Span]) -> Callable[[Span], bool]:
    """Whether a span overlaps one of spans, by bisection over them in order."""
    ordered = sorted(spans)
    starts = [start for start, _ in ordered]
    ends = list(itertools.accumulate((end for _, end in ordered), max))  # the furthest end so far

    def overlaps(span: Span) -> bool:
        before = bisect.bisect_left(starts, span[1])  # the spans that start before span ends
        return before > 0 and ends[before - 1] > span[0]

    return overlaps


def _find_all(text: str, needle: str, start: int = 0) -> Iterator[int]:
    at = text.find(needle, start)
    while at >= 0:
        yield at
        at = text.find(needle, at + 1)


def _chain(text: str, words: Sequence[str], *, avoid: Sequence[Span] = (), bounded: bool = False,
           one_line: bool = False) -> list[Span] | None:
    """The spans of words placed in text in order, each after the one before (SENTENCE_RULE): the shortest
    placement whose words cut into no ASCII word (with bounded), overlap none of avoid and lie on one line
    (with one_line), each word after the first at its first such place; None when there is none."""
    words = [word for word in words if word]
    if not words:
        return None
    overlaps = _overlapping(avoid)

    def allowed(span: Span) -> bool:
        return (not bounded or _bounded(text, *span)) and not overlaps(span)

    def following(word: str, position: int) -> Span | None:
        for at in _find_all(text, word, position):
            if one_line and "\n" in text[position:at]:
                return None
            span = (at, at + len(word))
            if allowed(span):
                return span
        return None

    best = None
    for at in _find_all(text, words[0]):
        spans = [(at, at + len(words[0]))]
        if not allowed(spans[0]):
            continue
        for word in words[1:]:
            span = following(word, spans[-1][1])
            if span is None:
                break
            spans.append(span)
        else:
            if best is None or spans[-1][1] - spans[0][0] < best[-1][1] - best[0][0]:
                best = spans
    return best


def sentence_spans(text: str, words: Sequence[str], fillers: Sequence[Span] = ()) -> list[Span] | None:
    """The spans of the fixed words where the sentence of these words stands in an answer (SENTENCE_RULE),
    fillers being the spans of the values and names it is to be filled with; None when it does not stand
    there."""
    for options in ({"bounded": True, "one_line": True}, {"bounded": True}, {}):
        spans = _chain(text, words, avoid=fillers, **options)
        if spans is not None:
            return spans
    return _chain(text, words)


def _present(text: str, words: Sequence[str]) -> bool:
    """Whether words stand in text in order, each after the one before (read greedily: the earliest place
    of each word leaves the most room to the next); False for no word."""
    at, seen = 0, False
    for word in words:
        if not word:
            continue
        found = text.find(word, at)
        if found < 0:
            return False
        at, seen = found + len(word), True
    return seen


def _ending(text: str, words: Sequence[str]) -> str:
    """text with its end read as the end of the sentence of these words (STRICT_RULE: one sentence mark may
    end an answer or not): when the last word ends with a sentence mark, the text's own final mark, if any,
    is replaced by it, or it is added; trailing whitespace goes. Every other character keeps its place."""
    last = words[-1] if words else ""
    if not last or last[-1] not in SENTENCE_MARKS:
        return text
    body = text.rstrip()
    if body and body[-1] in SENTENCE_MARKS:
        body = body[:-1]
    return body + last[-1]


def _stands(text: str, words: Sequence[str]) -> bool:
    """Whether the sentence of these words stands in the text, its end read as the sentence's (_ending)."""
    return _present(_ending(text, words), words)


def _sentence_cases(text: str, families: Families) -> list[str]:
    """The cases whose sentence, of any family given in any of its fixed-word lists, stands in the text."""
    return [case for case in CASES
            if any(key[1] == case and any(_stands(text, words) for words in variants)
                   for key, variants in families.items())]


@functools.cache
def training_sentences() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """(case, fixed words) of every sentence family of research_agent.data.special_mix in every placeholder
    style (FALSE_REFUSAL_RULE): the sentences a model trained on special-mix may refuse with, beyond those a
    development set holds. special_mix builds the training data and is imported here alone, on first use."""
    from ..data import special_mix

    found: dict[tuple[str, tuple[str, ...]], None] = {}
    for family in special_mix.FAMILIES.values():
        for style in special_mix.STYLES:
            found.setdefault((family.case, tuple(special_mix.fixed_parts(family, style, family.lang))), None)
    return tuple(found)


def _refusal_sentence(text: str, families: Families) -> bool:
    """Whether a refusal sentence of a known family stands in the text: one of families', or of
    training_sentences."""
    return (any(_stands(text, words) for variants in families.values() for words in variants)
            or any(_stands(text, words) for _, words in training_sentences()))


@functools.cache
def _name(name: str) -> re.Pattern:
    """A parameter or function name standing on its own (NAME_RULE)."""
    return re.compile(r"(?<![A-Za-z0-9_.])" + re.escape(name) + r"(?![A-Za-z0-9_]|\.[A-Za-z0-9_])")


def _name_spans(text: str, name: str) -> list[Span]:
    return [match.span() for match in _name(name).finditer(text)]


def _value_spans(text: str, value: str) -> list[Span]:
    return [(at, at + len(value)) for at in _find_all(text, value)]


_TOKENS = re.compile(r"[^\W_]+")  # runs of letters and digits, in any script


def lenient_words(words: Sequence[str]) -> list[str]:
    """The fixed words as ACEBench's rule reads them (LENIENT_RULE): each stripped of the closing brackets at
    its start and of the opening brackets at its end, which belong to the placeholders beside it, the last
    one of the sentence's final mark too, the whitespace collapsed; a word left empty dropped."""
    out = []
    for number, word in enumerate(words):
        text = word.strip().lstrip(CLOSING).rstrip(OPENING).strip()
        if number == len(words) - 1 and text[-1:] and text[-1] in SENTENCE_MARKS:
            text = text[:-1]
        text = " ".join(text.split())
        if text:
            out.append(text)
    return out


def lenient_holds(answer: str, words: Sequence[str], needed: Sequence[str]) -> bool:
    """LENIENT_RULE: the lenient words of a sentence stand in the answer, its whitespace collapsed, in order,
    and every needed value or name stands anywhere in it."""
    segments = lenient_words(words)
    return (bool(segments) and _present(" ".join(answer.split()), segments)
            and all(text in answer for text in needed))


# ---- the template -----------------------------------------------------------------------------------------

_MARKED = (r"\{\{[^{}\n]{1,40}\}\}|<<[^<>\n]{1,40}>>|\$\{[^{}\n]{1,40}\}|%[^%\s]{1,40}%|__[^_\s]{1,40}__"
           r"|[(<{\[（【「《〈＜][^()<>{}\[\]（）【】「」《》〈〉＜＞\n]{1,40}[)>}\]）】」》〉＞]"
           r"|\$[A-Za-z_][A-Za-z0-9_]*|(?<![A-Za-z0-9_])[A-Z][A-Z0-9_]+(?![A-Za-z0-9_])")
_LEADING = re.compile(f"(?:{_MARKED})$")
_TRAILING = re.compile(f"^(?:{_MARKED})")
_CAPITALS = re.compile(r"[A-Z][A-Z0-9_]+")
_EDGE = " \t\"'`“”‘’"  # what may stand between an edge placeholder and the sentence's quotes
_STOP = re.compile(r"[。！？!?]|\.(?=\s|$)")  # where a sentence ends
_VALUE_SLOT = re.compile(r"value|值", re.IGNORECASE)  # a placeholder of a value (SLOT_RULE)
_FUNCTION_SLOT = re.compile(r"func|(?<![a-z])api(?![a-z])|函数|接口", re.IGNORECASE)  # of a function


def _marked(token: str) -> bool:
    """Whether a placeholder shows itself as one: it holds a punctuation mark or a symbol (a bracket, $, %,
    _), or it is written in capitals."""
    return bool(_CAPITALS.fullmatch(token)) or any(unicodedata.category(c)[0] in "PS" for c in token)


@dataclass(frozen=True)
class Template:
    """A refusal item's sentence as its system message writes it (PLACEHOLDER_RULE): the text between
    consecutive fixed words (slots, stripped), and the placeholders right before the first and right after
    the last fixed word (leading, trailing), None where there is none."""

    slots: tuple[str, ...]
    leading: str | None
    trailing: str | None

    @property
    def placeholders(self) -> tuple[str, ...]:
        """The placeholders an answer must not hold anywhere: the slots that show themselves as
        placeholders, and the edge ones."""
        found = [slot for slot in self.slots if slot and _marked(slot)]
        found += [edge for edge in (self.leading, self.trailing) if edge]
        return tuple(dict.fromkeys(found))

    @property
    def every_slot(self) -> tuple[str, ...]:
        """The placeholders of all its slots in the sentence's order: the leading edge, the slots between
        the fixed words, the trailing edge."""
        return (*((self.leading,) if self.leading else ()), *self.slots,
                *((self.trailing,) if self.trailing else ()))


def slot_roles(template: Template, case: str) -> tuple[str, ...] | None:
    """The role of every slot of the template (Template.every_slot), in order (SLOT_RULE); None when its
    slots cannot be told apart so: other than two for an incorrect or a missing case, other than exactly one
    of them marked as the value's or the function's, any for an irrelevant case."""
    slots = template.every_slot
    if case not in SLOT_ROLES or len(slots) != len(SLOT_ROLES[case]):
        return None
    if not slots:
        return ()
    marked, other = SLOT_ROLES[case]
    mark = _VALUE_SLOT if case == "incorrect" else _FUNCTION_SLOT
    hits = [mark.search(slot) is not None for slot in slots]
    if hits.count(True) != 1:
        return None
    return tuple(marked if hit else other for hit in hits)


def _line(text: str, start: int, end: int) -> Span:
    """The start of the line holding text[start] and the end of the line holding text[end - 1]."""
    last = text.find("\n", end)
    return text.rfind("\n", 0, start) + 1, len(text) if last < 0 else last


def _system(item: dict) -> str | None:
    messages = item.get("messages")
    first = messages[0] if isinstance(messages, list) and messages else None
    content = first.get("content") if isinstance(first, dict) and first.get("role") == "system" else None
    return content if isinstance(content, str) else None


def _fixed(expected: object) -> list[str]:
    fixed = expected.get("fixed") if isinstance(expected, dict) else None
    return [word for word in fixed if isinstance(word, str) and word] if isinstance(fixed, list) else []


def sentence_template(item: dict) -> Template | None:
    """The template of a refusal item's sentence in its system message (PLACEHOLDER_RULE); None when the
    fixed words do not stand there in order on one line."""
    system, words = _system(item), _fixed(item.get("expected"))
    if system is None or not words:
        return None
    spans = _chain(system, words, bounded=True, one_line=True) or _chain(system, words, one_line=True)
    if spans is None:
        return None
    first, last = _line(system, spans[0][0], spans[-1][1])
    leading = _LEADING.search(system[first:spans[0][0]].rstrip(_EDGE))
    trailing = _TRAILING.search(system[spans[-1][1]:last].lstrip(_EDGE))
    return Template(slots=tuple(system[a[1]:b[0]].strip() for a, b in itertools.pairwise(spans)),
                    leading=leading.group() if leading else None,
                    trailing=trailing.group() if trailing else None)


def _placeholder_left(answer: str, template: Template, spans: list[Span], fillers: Sequence[Span]) -> bool:
    """Whether the answer, whose sentence stands at spans, leaves a placeholder of the template; an
    occurrence inside a value or a name it was filled with (fillers, their spans) is no placeholder."""
    for placeholder in template.placeholders:
        for at in _find_all(answer, placeholder):
            if not any(start <= at and at + len(placeholder) <= end for start, end in fillers):
                return True
    return any(slot and not _marked(slot) and answer[a[1]:b[0]].strip() == slot
               for slot, (a, b) in zip(template.slots, itertools.pairwise(spans), strict=True))


def _masked(text: str, spans: Sequence[Span]) -> str:
    chars = list(text)
    for start, end in spans:
        chars[start:end] = " " * (end - start)
    return "".join(chars)


def _regions(answer: str, spans: list[Span], template: Template | None,
             fillers: Sequence[Span]) -> tuple[list[str], list[str]]:
    """(slots, outside) of an answer whose sentence stands at spans (SENTENCE_RULE): the text of every slot
    in the sentence's order (Template.every_slot, the edge slots when the template has them) and the rest
    besides the fixed words and the full stops that end the edge slots. A full stop inside a filler
    (fillers: the spans of the values and names) ends no edge slot."""
    masked = _masked(answer, fillers)
    start, end = spans[0][0], spans[-1][1]
    first, last = _line(answer, start, end)
    slots: list[str] = []
    if template is not None and template.leading:
        stops = list(_STOP.finditer(masked, first, start))
        begin = stops[-1].end() if stops else first
        slots.append(answer[begin:start])
        outside = [answer[:begin]]
    else:
        outside = [answer[:start]]
    slots += [answer[a[1]:b[0]] for a, b in itertools.pairwise(spans)]
    if template is not None and template.trailing:
        stop = _STOP.search(masked, end, last)
        slots.append(answer[end:stop.start() if stop else last])
        outside.append(answer[stop.end() if stop else last:])
    else:
        outside.append(answer[end:])
    return slots, outside


# ---- the slots filled (STRICT_RULE) -----------------------------------------------------------------------


def _wrapping(placeholder: str) -> tuple[str, str] | None:
    """The brackets a placeholder is written in ((x), <x>, {{x}}, 【x】 ...): its opening and its closing
    run; None when it has none."""
    opening = len(placeholder) - len(placeholder.lstrip(OPENING))
    closing = len(placeholder) - len(placeholder.rstrip(CLOSING))
    if opening and closing and opening + closing < len(placeholder):
        return placeholder[:opening], placeholder[len(placeholder) - closing:]
    return None


def _fillings(case: str, values: Sequence[str], params: Sequence[str], api: str) -> list[dict[str, str]]:
    """Every right filling of a case's slots, by role (STRICT_RULE): incorrect, the values and the
    parameters each joined by SEPARATOR in one order that keeps each value with its parameter; missing,
    the api and the parameters in any order (in the expected order alone beyond MAX_ORDERS of them)."""
    if case == "incorrect":
        if not values or len(values) != len(params):
            return []
        orders = (itertools.permutations(range(len(values))) if len(values) <= MAX_ORDERS
                  else [range(len(values))])
        return [{"value": SEPARATOR.join(values[i] for i in order),
                 "param": SEPARATOR.join(params[i] for i in order)} for order in map(tuple, orders)]
    if case == "missing":
        if not params or not api:
            return []
        orders = itertools.permutations(params) if len(params) <= MAX_ORDERS else [params]
        return [{"api": api, "params": SEPARATOR.join(order)} for order in orders]
    return [{}]


_MARK = "[" + re.escape(SENTENCE_MARKS) + "]"


def _filled(words: Sequence[str], template: Template, roles: Sequence[str],
            filling: Mapping[str, str]) -> str:
    """The regular expression of the sentence of these fixed words with its slots filled so (STRICT_RULE),
    whitespace allowed where a slot meets a fixed word and one sentence mark allowed at the end."""
    placeholders = template.every_slot

    def slot(index: int) -> str:
        filler = filling[roles[index]]
        options = [re.escape(filler)]
        wrapping = _wrapping(placeholders[index])
        if wrapping is not None:
            options.append(re.escape(wrapping[0] + filler + wrapping[1]))
        return "(?:" + "|".join(options) + ")"

    pieces, index = [], 0
    if template.leading:
        pieces.append(slot(index))
        index += 1
    for number, word in enumerate(words):
        if number:
            pieces.append(slot(index))
            index += 1
        if number == len(words) - 1 and not template.trailing and word[-1] in SENTENCE_MARKS:
            word = word[:-1]  # the sentence's own final mark: one may end the answer or not
        if word:
            pieces.append(re.escape(word))
    if template.trailing:
        pieces.append(slot(index))
    return r"\s*" + r"\s*".join(pieces) + r"\s*" + _MARK + r"?\s*"


def _content(text: str, placeholder: str) -> str:
    """What a slot of an answer holds: its text stripped, taken out of its placeholder's brackets when it
    keeps them."""
    content = text.strip()
    wrapping = _wrapping(placeholder)
    if (wrapping is not None and len(content) > len(wrapping[0]) + len(wrapping[1])
            and content.startswith(wrapping[0]) and content.endswith(wrapping[1])):
        content = content[len(wrapping[0]):len(content) - len(wrapping[1])]
    return content


def _misfilled(case: str, contents: Mapping[str, str] | None,
               fillings: Sequence[Mapping[str, str]]) -> str | None:
    """wrong_values or wrong_params when the slots do not hold what they should (FAILURE_RULE); None when
    they do. contents: what each role's slot holds (None when the slots could not be read)."""
    if case == "irrelevant":
        return None
    if contents is None or not fillings:
        return "wrong_values" if case == "incorrect" else "wrong_params"
    if case == "incorrect":
        held = [filling for filling in fillings if contents["value"] == filling["value"]]
        if not held:
            return "wrong_values"
        return None if any(contents["param"] == filling["param"] for filling in held) else "wrong_params"
    right = any(contents["api"] == filling["api"] and contents["params"] == filling["params"]
                for filling in fillings)
    return None if right else "wrong_params"


# ---- scoring ----------------------------------------------------------------------------------------------


def sentence_families(items: Iterable[dict]) -> dict[tuple[str, str], frozenset[tuple[str, ...]]]:
    """{(family, case): every fixed-word list the refusal items give that family for that case}
    (SENTENCE_RULE): a family written in several placeholder styles gives a case one list per style.
    ValueError when one list stands for two cases, of one family or of two: an answer holding those words
    could not tell its case."""
    found: dict[tuple[str, str], set[tuple[str, ...]]] = {}
    owners: dict[tuple[str, ...], tuple[str, str, object]] = {}  # fixed words -> (family, case, item id)
    for item in items:
        expected = item.get("expected")
        if not isinstance(expected, dict) or expected.get("type") != "refusal":
            continue
        family, case, fixed = item.get("family"), expected.get("case"), tuple(_fixed(expected))
        if not isinstance(family, str) or case not in CASES or not fixed:
            continue
        owner = owners.setdefault(fixed, (family, case, item.get("id")))
        if owner[1] != case:
            raise ValueError(f"the fixed words {list(fixed)} stand for the {owner[1]} case of {owner[0]!r} "
                             f"(item {owner[2]}) and the {case} case of {family!r} (item {item.get('id')})")
        found.setdefault((family, case), set()).add(fixed)
    return {key: frozenset(variants) for key, variants in found.items()}


def _extra_text(outside: Sequence[str], words: Sequence[str], template: Template | None) -> bool:
    """Whether anything stands outside an answer's sentence (_regions; STRICT_RULE): any character but
    whitespace, a mark or a symbol as much as a word, the one sentence mark an answer may end with aside
    where it stands outside the sentence (no trailing slot ends at it, the last fixed word has no final
    mark)."""
    rest = list(outside)
    if rest and not (template is not None and template.trailing) and words[-1][-1] not in SENTENCE_MARKS:
        end = rest[-1].strip()
        if len(end) == 1 and end in SENTENCE_MARKS:
            rest[-1] = ""
    return any(part.strip() for part in rest)


def _refusal(answer: str | None, item: dict, families: Families) -> dict:
    """The fields of a refusal item's score (STRICT_RULE, LENIENT_RULE, FAILURE_RULE)."""
    expected = item["expected"]
    case = expected["case"]
    words = _fixed(expected)
    values = [v for v in expected.get("values") or [] if v] if case == "incorrect" else []
    params = [p for p in expected.get("params") or [] if p] if case != "irrelevant" else []
    api = expected.get("api") if case == "missing" and expected.get("api") else ""
    names = [*params, *([api] if api else [])]
    out: dict = {"case": case}
    if answer is None or not answer.strip():
        return {**out, "strict": False, "lenient": False,
                "failure": "no_answer" if answer is None else "empty"}
    # ACEBench checks an incorrect case's values alone, a missing case's function and parameters
    out["lenient"] = lenient_holds(answer, words, values if case == "incorrect" else names)
    found = find_calls(answer, item["call_format"], item_functions(item))
    placed = _ending(answer, words)  # where the sentence is looked for: its final mark read as its own
    fillers = [*(span for value in values for span in _value_spans(placed, value)),
               *(span for name in names for span in _name_spans(placed, name))]
    spans = sentence_spans(placed, words, fillers)
    out.update(call_notation=found[0] if found else None, sentence=spans is not None,
               sentences=_sentence_cases(answer, families))
    if case == "incorrect" and found:
        out["called_with_invalid"] = any(_kept(value, found[1]) for value in values)
    if spans is None:
        other = set(out["sentences"]) - {case}
        failure = "called" if found else "wrong_case" if other else "no_sentence"
        return {**out, "strict": False, "failure": failure}
    template = sentence_template(item)
    roles = slot_roles(template, case) if template is not None else None
    fillings = _fillings(case, values, params, api)
    strict = (found is None and roles is not None
              and any(re.fullmatch(_filled(words, template, roles, filling), answer) for filling in fillings))
    placeholder = template is not None and _placeholder_left(placed, template, spans, fillers)
    slots, outside = _regions(placed, spans, template, fillers)
    out.update(placeholder_left=placeholder, extra_text=_extra_text(outside, words, template))
    if strict:
        failure = None
    elif found:
        failure = "called"
    elif placeholder:
        failure = "placeholder_left"
    else:
        contents = None
        if roles is not None and len(slots) == len(roles):
            contents = {role: _content(text, slot) for role, text, slot
                        in zip(roles, slots, template.every_slot, strict=True)}
        failure = _misfilled(case, contents, fillings) or "extra_text"
    return {**out, "strict": strict, "failure": failure}


def _same(got: list[Call], want: list[Call], tools: list[dict]) -> bool:
    """ace_probe.same_calls; False for calls it cannot compare (an integer too long to write, a value nested
    deeper than its normalisation goes), which are not the expected ones."""
    try:
        return same_calls(got, want, tools)
    except (ValueError, TypeError, RecursionError, MemoryError):
        return False


def _call(answer: str | None, item: dict, families: Families) -> dict:
    """The fields of a valid item's score (CALL_RULE, FALSE_REFUSAL_RULE, FAILURE_RULE)."""
    expected = item["expected"]
    fmt = item["call_format"]
    want = [(call["name"], call["arguments"]) for call in expected["calls"]]
    verbatim = [value for value in expected.get("verbatim") or [] if value]
    out: dict = {"verbatim": len(verbatim)}
    if answer is None:
        return {**out, "exact_call": False, "false_refusal": False, "failure": "no_answer"}
    if not answer.strip():
        return {**out, "exact_call": False, "false_refusal": True, "refusal_form": "empty",
                "failure": "empty"}
    calls = parse_answer(answer, fmt)
    exact = calls is not None and _same(calls, want, item.get("tools") or [])
    found = (fmt, calls) if calls is not None else find_calls(answer, fmt, item_functions(item))
    form = None
    if found is None:
        form = "sentence" if _refusal_sentence(answer, families) else "other_text"
    altered = None if not verbatim or found is None else not all(_kept(v, found[1]) for v in verbatim)
    if exact:
        failure = None
    elif calls is not None:
        failure = "zh_altered" if altered else "wrong_call"
    elif found is not None:
        failure = "with_text" if found[0] == fmt else "other_notation"
    else:
        failure = "refused" if form == "sentence" else "no_call"
    return {**out, "exact_call": exact, "false_refusal": found is None, "refusal_form": form,
            "zh_altered": altered, "call_notation": found[0] if found else None,
            "sentences": _sentence_cases(answer, families), "failure": failure}


def _base(answer_text: str | None, item: dict) -> dict:
    """A score's fields before scoring, each that does not apply None."""
    return {"id": item.get("id"), "kind": item.get("kind"), "lang": item.get("lang"),
            "family": item.get("family"), "status": "ok", "case": None,
            "chars": 0 if answer_text is None else len(answer_text), "call_notation": None, "sentences": [],
            "strict": None, "lenient": None, "sentence": None, "placeholder_left": None, "extra_text": None,
            "called_with_invalid": None, "exact_call": None, "false_refusal": None, "refusal_form": None,
            "zh_altered": None, "verbatim": 0, "failure": None}


def score_item(answer_text: str | None, item: dict, *, families: Families | None = None) -> dict:
    """The score of one answer (None: the run recorded none) to an item: id, kind, lang, family, status
    ("ok"), case (the refusal's), chars, call_notation (where a call was found, None without one), sentences
    (the cases whose sentence of a family known stands in the answer), strict, lenient, sentence (the item's
    own sentence stands), placeholder_left, extra_text and called_with_invalid (refusal items), exact_call,
    false_refusal (a bool for every valid item), refusal_form, zh_altered and verbatim (the number of
    verbatim values; valid items), failure (FAILURE_RULE); a field that does not apply is None. families are
    the sentences known (sentence_families of the set; default: the item's own), which tell a wrong case and
    the form of a false refusal."""
    families = sentence_families([item]) if families is None else families
    score = _base(answer_text, item)
    score.update(_refusal(answer_text, item, families) if item["kind"] in REFUSAL_KINDS
                 else _call(answer_text, item, families))
    return score


def _failed(answer_text: str | None, item: dict, exc: Exception) -> dict:
    """The score of an item the scorer failed on (FAILURE_RULE: scoring_error), with the error."""
    score = _base(answer_text, item)
    if item.get("kind") in REFUSAL_KINDS:
        expected = item.get("expected")
        score.update(case=expected.get("case") if isinstance(expected, dict) else None, strict=False,
                     lenient=False)
    else:
        score.update(exact_call=False, false_refusal=False)
    score.update(status="scoring_error", error=f"{type(exc).__name__}: {exc}", failure="scoring_error")
    return score


def score_items(answers: Sequence[str | None], items: Sequence[dict]) -> list[dict]:
    """The score of every answer, in item order, with the sentences of the whole set (sentence_families). A
    failure of the scorer on one item is that item's score (status "scoring_error", with the error), and
    the others are scored all the same."""
    families = sentence_families(items)
    scores = []
    for answer, item in zip(answers, items, strict=True):
        try:
            scores.append(score_item(answer, item, families=families))
        except Exception as exc:  # noqa: BLE001  deliberate: one item's failure is recorded, the rest scored
            scores.append(_failed(answer, item, exc))
    return scores


def score_records(records: Iterable[dict], items: Sequence[dict]) -> list[dict]:
    """The scores of a run's records (scripts/run_special_dev.py) over the items, in item order: one record
    per item, found by its id, an ok record's answer scored, any other record as no answer. ValueError when
    an item has no record or several, or a record names no item."""
    ids = {item["id"] for item in items}
    by_id: dict[str, dict] = {}
    for record in records:
        rid = record.get("id")
        if rid not in ids:
            raise ValueError(f"a record of {rid!r}, which names no item of the set")
        if rid in by_id:
            raise ValueError(f"two records of {rid}")
        by_id[rid] = record
    missing = [item["id"] for item in items if item["id"] not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} item(s) without a record, the first {missing[0]}")
    answers = []
    for item in items:
        record = by_id[item["id"]]
        answer = record.get("answer")
        answers.append(answer if record.get("status") == "ok" and isinstance(answer, str) else None)
    return score_items(answers, items)


# ---- the summary ------------------------------------------------------------------------------------------


def _rate(count: int, n: int) -> float | None:
    return count / n if n else None


def _scoring_errors(rows: list[dict]) -> int:
    return sum(row.get("status") == "scoring_error" for row in rows)


def _refusal_stats(rows: list[dict]) -> dict:
    n = len(rows)
    strict = sum(row["strict"] is True for row in rows)
    lenient = sum(row["lenient"] is True for row in rows)
    failures = Counter(row["failure"] for row in rows)
    return {"n": n, "strict": strict, "strict_rate": _rate(strict, n), "lenient": lenient,
            "lenient_rate": _rate(lenient, n), "extra_text": sum(row["extra_text"] is True for row in rows),
            "called_with_invalid": sum(row["called_with_invalid"] is True for row in rows),
            "scoring_errors": _scoring_errors(rows),
            "failures": {mode: failures[mode] for mode in REFUSAL_FAILURES}}


def _call_stats(rows: list[dict]) -> dict:
    n = len(rows)
    exact = sum(row["exact_call"] is True for row in rows)
    failures = Counter(row["failure"] for row in rows)
    refused = [row for row in rows if row["false_refusal"] is True]
    return {"n": n, "exact": exact, "exact_rate": _rate(exact, n), "false_refusals": len(refused),
            "false_refusals_by": {form: sum(row.get("refusal_form") == form for row in refused)
                                  for form in REFUSAL_FORMS},
            "zh_altered": sum(row["zh_altered"] is True for row in rows),
            "zh_checked": sum(row["zh_altered"] is not None for row in rows),
            "verbatim_items": sum(row["verbatim"] > 0 for row in rows),
            "scoring_errors": _scoring_errors(rows),
            "failures": {mode: failures[mode] for mode in CALL_FAILURES}}


def _groups(rows: list[dict]) -> dict:
    """The groups of rows: special, priority, valid and each kind."""
    def of(kinds: Sequence[str]) -> list[dict]:
        return [row for row in rows if row["kind"] in kinds]

    return {"special": _refusal_stats(of(SPECIAL_KINDS)), "priority": _refusal_stats(of((PRIORITY,))),
            "valid": _call_stats(of(VALID_KINDS)),
            "kinds": {kind: (_refusal_stats if kind in REFUSAL_KINDS else _call_stats)(of((kind,)))
                      for kind in KINDS}}


def summarize(scores: Sequence[dict], items: Sequence[dict]) -> dict:
    """{"n", "scoring_errors", "special", "priority", "valid", "kinds": {kind: ...}, "languages": {lang:
    {"special", "priority", "valid", "kinds"}}} of a run's scores (module docstring: Summary). ValueError
    unless there is one score per item."""
    by_id = {score["id"]: score for score in scores}
    ids = [item["id"] for item in items]
    if len(by_id) != len(scores) or len(set(ids)) != len(ids) or set(by_id) != set(ids):
        raise ValueError(f"one score per item is needed: {len(scores)} scores, {len(items)} items")
    rows = [by_id[rid] for rid in ids]
    return {"n": len(rows), "scoring_errors": _scoring_errors(rows), **_groups(rows),
            "languages": {lang: _groups([row for row in rows if row["lang"] == lang]) for lang in LANGS}}


def _share(count: int, n: int) -> str:
    return f"{count}/{n} ({100 * count / n:.1f}%)" if n else f"{count}/0"


def _failures(stats: dict) -> str:
    return ", ".join(f"{mode} {count}" for mode, count in stats["failures"].items() if count) or "none"


def summary_lines(summary: dict) -> list[str]:
    """The summary as lines to print: the special kinds pooled, each kind and language, priority, the valid
    kinds pooled, each kind and language, and the scoring errors when there are any."""
    def refusal(label: str, stats: dict) -> str:
        return (f"{label}: strict {_share(stats['strict'], stats['n'])}, lenient "
                f"{_share(stats['lenient'], stats['n'])}; failures: {_failures(stats)}")

    def valid(label: str, stats: dict) -> str:
        by = stats["false_refusals_by"]
        return (f"{label}: exact {_share(stats['exact'], stats['n'])}, false refusals "
                f"{stats['false_refusals']} (sentence {by['sentence']}, other text {by['other_text']}, empty "
                f"{by['empty']}), zh altered {stats['zh_altered']} of {stats['zh_checked']}; failures: "
                f"{_failures(stats)}")

    lines = [refusal("special", summary["special"])]
    lines += [refusal(f"  {kind} {lang}", summary["languages"][lang]["kinds"][kind])
              for kind in SPECIAL_KINDS for lang in LANGS]
    lines.append(refusal("priority", summary["priority"]))
    lines.append(valid("valid", summary["valid"]))
    lines += [valid(f"  {kind} {lang}", summary["languages"][lang]["kinds"][kind])
              for kind in VALID_KINDS for lang in LANGS]
    if summary.get("scoring_errors"):
        lines.append(f"scoring errors: {summary['scoring_errors']} (scored as failed)")
    return lines


# ---- the item check ---------------------------------------------------------------------------------------


def _texts(value: object, *, empty: bool = False) -> bool:
    """Whether value is a list of non-empty strings (empty: the list may be empty)."""
    return (isinstance(value, list) and (empty or bool(value))
            and all(isinstance(text, str) and text.strip() for text in value))


def _check_messages(messages: object) -> list[str]:
    if not isinstance(messages, list) or len(messages) < 2:
        return ["messages must be the system message, then the conversation up to the user's request"]
    errors = []
    for number, message in enumerate(messages):
        if (not isinstance(message, dict) or message.get("role") not in ROLES
                or not isinstance(message.get("content"), str)):
            errors.append(f"message {number} needs a role ({', '.join(ROLES)}) and a text content")
            continue
        errors += [f"message {number} holds {marker}" for marker in acebench_text(message["content"])]
    if errors:
        return errors
    if messages[0]["role"] != "system" or not messages[0]["content"].strip():
        errors.append("the first message must be the system message, with the rules and the functions")
    if any(message["role"] == "system" for message in messages[1:]):
        errors.append("only the first message is the system's")
    if messages[-1]["role"] != "user":
        errors.append("the last message must be the user's")
    return errors


def _letters(text: str) -> str:
    return "".join(_TOKENS.findall(text))


def _check_refusal(item: dict, expected: dict) -> list[str]:
    errors = []
    kind, case = item["kind"], expected.get("case")
    if expected.get("type") != "refusal":
        errors.append('a refusal item expects {"type": "refusal", ...}')
    if case not in CASES:
        return [*errors, f"expected.case is {case!r}, not one of {', '.join(CASES)}"]
    if kind in SPECIAL_KINDS and case != kind:
        errors.append(f"kind {kind} expects the {kind} sentence, not the {case} one")
    if kind == PRIORITY and case not in PRIORITY_CASES:
        errors.append(f"a priority item expects the incorrect or the missing sentence, not the {case} one")
    if not _texts(expected.get("fixed")):
        errors.append("expected.fixed must be the sentence's fixed words, non-empty strings")
    for key in ("values", "params"):
        if not _texts(expected.get(key), empty=True):
            errors.append(f"expected.{key} must be a list of non-empty strings")
    api = expected.get("api")
    if api is not None and not isinstance(api, str):
        errors.append("expected.api must be a string")
    if errors:
        return errors
    if case == "incorrect" and not (expected["values"] and expected["params"]):
        errors.append("an incorrect case names its invalid values and their parameters")
    elif case == "incorrect" and len(expected["values"]) != len(expected["params"]):
        errors.append("an incorrect case names each invalid value with its parameter: as many values as "
                      "parameters")
    if case == "missing" and not (expected["params"] and isinstance(api, str) and api.strip()):
        errors.append("a missing case names the function (api) and its missing parameters")
    template = sentence_template(item)
    if template is None:
        errors.append("the fixed words do not stand in order on one line of the system message")
        return errors
    if slot_roles(template, case) is None:
        errors.append(f"the sentence's slots {list(template.every_slot)} cannot be told apart (SLOT_RULE): "
                      "an incorrect sentence has one slot named for a value and one other, a missing one one "
                      "named for a function and one other, an irrelevant one none")
    # A placeholder searched for anywhere in an answer (outside the fillers) must not be written like a
    # filler; a plain-word one, read in its own slot alone, can only be confused with the filler of that slot,
    # which the item does not name.
    fillers = [*expected["values"], *expected["params"], *([api] if case == "missing" and api else [])]
    for placeholder in template.placeholders:
        for filler in fillers:
            if _letters(placeholder) and _letters(placeholder) == _letters(filler):
                errors.append(f"the filler {filler!r} cannot be told from the placeholder {placeholder!r}")
    return errors


def _check_call(item: dict, expected: dict) -> list[str]:
    errors = []
    if expected.get("type") != "call":
        errors.append('a valid item expects {"type": "call", ...}')
    calls = expected.get("calls")
    if (not isinstance(calls, list) or not calls
            or not all(isinstance(call, dict) and set(call) == {"name", "arguments"}
                       and isinstance(call["name"], str) and call["name"].strip()
                       and isinstance(call["arguments"], dict) for call in calls)):
        return [*errors, "expected.calls must be one or more {name, arguments}"]
    want = [(call["name"], call["arguments"]) for call in calls]
    verbatim = expected.get("verbatim")
    if not _texts(verbatim, empty=True):
        errors.append("expected.verbatim must be a list of non-empty strings")
    else:
        errors += [f"the verbatim value {value!r} is in no argument of the expected calls"
                   for value in verbatim if not _kept(value, want)]
    fmt = item["call_format"]
    module = notations()
    try:
        back = module.parse_calls(module.serialize_calls(want, fmt), fmt)
    except (ValueError, TypeError) as exc:
        errors.append(f"the expected calls cannot be written in {fmt}: {exc}")
    else:
        if back != want:
            errors.append(f"the expected calls do not read back from {fmt}")
    return errors


def check_item(item: object) -> list[str]:
    """Why an item is not a well-formed item of a refusal development set (module docstring: Items), as
    messages; [] when it is."""
    if not isinstance(item, dict):
        return ["the item is not an object"]
    errors = []
    if not isinstance(item.get("id"), str) or not item["id"].strip():
        errors.append("id missing or empty")
    for key, allowed in (("split", SPLITS), ("lang", LANGS), ("kind", KINDS), ("call_format", FORMATS)):
        if item.get(key) not in allowed:
            errors.append(f"{key} is {item.get(key)!r}, not one of {', '.join(allowed)}")
    if not isinstance(item.get("family"), str) or not item["family"].strip():
        errors.append("family missing or empty")
    source = item.get("source")
    record = source.get("record") if isinstance(source, dict) else None
    if (not isinstance(source, dict) or not isinstance(source.get("dataset"), str) or not source["dataset"]
            or not isinstance(record, str | int) or isinstance(record, bool)):
        errors.append('source must be {"dataset": ..., "record": ...}')
    errors += _check_messages(item.get("messages"))
    if errors:
        return errors
    expected = item.get("expected")
    if not isinstance(expected, dict):
        return ["expected missing"]
    return (_check_refusal if item["kind"] in REFUSAL_KINDS else _check_call)(item, expected)

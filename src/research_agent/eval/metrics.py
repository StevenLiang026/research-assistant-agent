"""Evaluation metrics: score the runner's records against the items, summarise, compare two systems.

Module "指标" and "bootstrap 与 McNemar" (docs/plan.md, 评测 and 统计). score_item(result, item) checks one
record of research_agent.eval.runner against its item (docs/modules/接口.md, 评测题); summarize aggregates the
scores; bootstrap_ci, cluster_bootstrap_ci and mcnemar compare the per-item successes of two systems on the
same items, compare_systems runs the three on two systems' scores, and holm adjusts the p-values of the
preregistered comparisons (Comparing two systems, at the end).

Normalisation (plan: 归一规则). Before any comparison both sides of a call get the tool's schema defaults
filled in (DEFAULTS), so a parameter left out and the same parameter set to its default are the same call.
Strings are NFKC-normalised with whitespace collapsed (normalize_text); arxiv_id becomes the canonical
"YYMM.NNNNN" (normalize_arxiv_id), date_from, date_to and since ISO dates (normalize_date), each falling back
to the normalised text when it does not parse, so a malformed value still differs from a good one; categories,
tags and missing compare as sets; integral floats are ints (normalize_arguments). Where the tool's schema says
uniqueItems (search_arxiv.categories), a list with a repeated element is malformed: it never equals a list
without one, so ["cs.CL", "cs.LG", "cs.CL"] against ["cs.CL", "cs.LG"] is 值或格式错; tags and missing have
no uniqueItems and a repeat there is harmless (the plan: 分类与标签按集合比). A "query" is free text and
is judged by required words: slots["required_words"] when the item gives them (a list of strings, for
hand-written items), otherwise the slot topic's terms (slots["topic"], a string or a list) that the expected
query itself contains; the model's query must contain every one of them (query_contains_all). When no required
word applies (no topic, or a teacher query that rephrased it), the query must equal the expected one after
normalize_text and casefold(). The strict literal match, normalize_text equality without casefold, is reported
separately as query_strict and is not part of success. Other free text is not compared at all (COMPARED
lists what is): ask_clarification's question, and write_note's four parts (problem, method, evidence,
limitations), new_terms and open_terms, whose gold is the teacher's paraphrase, which a note that quotes the
paper never equals (docs/m3-进度.md). So of ask_clarification only "missing" and of write_note only arxiv_id
and mode enter the checks, the error classes and the per-parameter F1; ids_grounded still scans the free text.
What write_note does not compare must still pass the executor's gate, the tool schema that
validation.check_call validates every call against before it runs (SHAPE_CHECKED): a parameter the tool does
not have, or free text of a type the schema refuses (evidence that is not a string, new_terms that is not a
list of strings), makes a called write_note match no expected call, as 多参 and 值或格式错 (3 and 4 below).
arxiv_id and mode keep the M2 normalisation that the compared parameters of every tool get: a versioned or
prefixed arXiv id ("2609.01234v2", "arXiv:2609.01234") or a mode with spaces around it (" scaffold ") still
matches, though the executor would refuse each of them. So the gate checks only the parameters that are not
compared. ask_clarification's question has no such check.

Checks of one item; True or False where the check applies to the item's type, None where it does not:

- json_valid: every call's arguments parse to a JSON object (true when there is no call). All types.
- no_think_leak: the output's text carries no thinking tag (<think> or </think>, any case). All types;
  summarize reports 1 minus its rate as the think-leak rate.
- tool_choice: the set of called tool names equals the expected set (empty for no_call, {ask_clarification}
  for clarify). All types.
- arguments: every expected call has its own called counterpart (a one-to-one matching) with the same name and
  matching normalised arguments (free text aside, but a write_note must pass the executor's gate); extra calls
  are allowed here. call, next_step, parallel, recover.
- exact_args: arguments, and as many calls as expected: the calls equal the expected ones as a multiset, so
  parallel calls count with their multiplicity. call, next_step, parallel, recover.
- query_strict: exact_args with every query compared literally (normalize_text, case kept). Only items whose
  expected calls have a query; reported, not part of success.
- ids_grounded: every arXiv id in the output (text and call arguments) occurs in some message of the item's
  history (the plan's 编号无编造). All types.
- no_false_call: no call at all. no_call only; summarize reports 1 minus its rate as the false-call rate.
- clarify: ask_clarification was called and, when the expected call names "missing", with the same set.
  clarify only.
- recover: exact_args (the step after a failure is judged like any call). recover only.

success is True when every check that applies to the type passes, query_strict aside. TYPE_CHECKS lists them.

error_class is None for a success; otherwise one class of ERROR_CLASSES, the plan's eleven in the plan's
order (docs/plan.md, 分工: 误差分类法). Precedence (PRECEDENCE): the two structural classes, 10 and 11, are
decided first, since a broken output is not judged on its content; then 1 to 9 in the plan's order, the first
that applies. Classes 1 to 4 diagnose an attempt at the expected calls, so they do not apply to an output that
only repeats calls of the previous step: that is class 9 unless one of 5 to 8 applies first.

1. 选错工具: calls were expected and the called names differ from the expected ones as a multiset (no call at
   all included), except a clarify item without ask_clarification (7) and a recover item without any call (8).
2. 缺参: the names match; in the best pairing a called call lacks a parameter the expected one has.
3. 多参: ... a called call has a parameter the expected one does not, or a write_note one the tool does not
   have.
4. 值或格式错: ... a value differs after normalisation, or a write_note has free text of a type its schema
   refuses. Free text (Normalisation: COMPARED) is not compared otherwise, so a free-text argument that is
   missing, extra or different makes none of 2 to 4. A made-up arXiv id inside a compared argument (get_paper
   on an id the history never showed) is a value that differs, so it lands here, before 5: the plan's
   fabricated-id measurement (编号无编造) is the ids_grounded rate, which counts every made-up id, not the
   count of class 5. One inside free text (a question, a note's evidence) is no value that differs and lands
   in 5 at the earliest.
5. 编号编造: ids_grounded fails (in practice: a made-up id in the text, or in a call that is not an attempt).
6. 不该调却调: no call was expected and the output calls a tool.
7. 该追问没追问: a clarify item whose output does not call ask_clarification.
8. 恢复不当: a recover item failing otherwise.
9. 重试风暴: the output's calls all repeat calls of the previous assistant step (call_key: defaults filled).
10. JSON 损坏: a call's arguments are not a JSON object (json_valid fails).
11. 思考块泄漏: the output's text carries a thinking tag (no_think_leak fails).

A record whose status is not ok (context_overflow, error, budget_stop, skipped) has no output: the checks on
the output (json_valid, no_think_leak, ids_grounded, no_false_call) are None, not applicable, and drop out of
their rates and of summarize's n_per_check (so a no_call item that overflowed is no false call; with no other
no_call item the false-call rate is None); the other applying checks are False, success is False and the
status is the error_class. When nothing above applies to a failure, error_class is None.

Per-parameter precision, recall and F1 count (tool, parameter, normalised value) triples: called calls are
paired with expected calls of the same name so that the most parameters agree (query by the required-words
rule), each parameter of a paired call agreeing with the expected one is a true positive, the others and every
parameter of an unpaired called call are false positives, every parameter of an expected call without an
agreeing counterpart a false negative. A parameter neither side set explicitly (both at the schema default) is
left out of the counts (plan: 取默认值的参数视同未填), so defaults do not inflate the F1; one side setting it,
even to the default, counts it, and the values compare with the default filled in as above. score_item keeps
the counts per "tool.parameter"; summarize sums them over the items and reports the macro F1 over the
parameters that occur. Free text is not counted (Normalisation): the only keys of ask_clarification and
write_note are ask_clarification.missing, write_note.arxiv_id and write_note.mode, whatever else a call of
either sets, a parameter the tool does not have included.

Scorers (M7, docs/m7-本地提升计划.md WP1: the owner's decision of 10-29, the normalized scorer as the main
score and the default one reported beside it). score_item and score_items take scorer, one of SCORERS:
DEFAULT_SCORER ("default"), everything above, the scorer of every score before M7, or NORMALIZED_SCORER
("normalized"), which adds three normalisations, applied to the gold calls and the called ones alike and to
every comparison: the checks, the pairing, the error classes and the per-parameter counts, so a success has
no error class and the counts follow the same comparison under either scorer. The score has the same keys
under both; scripts/score.py records the scorer of a file the normalized scorer made.

- S1, a date_to on today restricts nothing: today is the date the item's first system message gives as
  "今天是 {today}" (agent.prompts; today_of). A call of DATE_TO_TOOLS (search_arxiv, the one tool whose
  schema has a date_to) whose date_to is that day after normalize_date has it dropped before anything else,
  on either side, as if the call had left it out: the key's presence counts in the matching, so a gold
  date_to on today and an answer without one match, and an answer's date_to on another day against a gold
  one on today is a parameter too many (多参). Without a today (no system message, no such text, no real
  date), S1 does not apply to the item.
- S2, a query's hyphens are spaces: the query comparison reads both queries, the slots' required words and
  topic terms, and the expected query the terms are looked for in, with every hyphen as a space
  (normalize.fold_hyphens: after NFKC, U+002D, U+2010, U+2011, U+2012 and U+2013); so do the containment test
  and the equality fallback. query_strict stays literal.
- S3, a count the request does not give is the tool's: for search_notes.top_k, list_notes.limit and
  fetch_fulltext.max_chars (COUNT_PARAMS) a called value equal to the schema default is accepted beside the
  gold value, when the item's slots have no "count" and the gold value is a whole number the request text
  does not write (request_text: the item's user messages; numbers_in: in digits, with a comma or a space
  between groups of three too (1,200, 1 200), in Chinese numerals, the financial ones too (伍, 壹仟贰佰), or
  in English number words, with a multiplier too (2千, 1.5千, 1万, 1千2百, 1.2k, 3w, a dozen, 2 thousand,
  1.2 thousand, 2 dozen, 3 hundred), wherever they stand, a date's or an id's digits included, which can only
  keep S3 from applying). The owner's quantity words (10-30) count too: a couple and a pair are 2, 一打 12,
  半打 and half a dozen 6, never 12; a vague word (a few, few, several, some, 几, 几个, 一些, 若干) is no
  count, so S3 still applies (一些 gives no 1). Any third value is wrong as before, and a gold value at the
  default accepts the default alone. search_arxiv.max_results is not among them.

Comparing two systems (docs/plan.md, 统计; the M3 plan, default 2). compare_systems joins each system's
scores to the items by id and compares their successes item by item: the paired bootstrap of the difference in
success rate (10,000 resamples, seed 42), the cluster bootstrap as its robustness check, and the exact McNemar
test. The clusters are cluster_keys': a drafted item's template family, and every handwritten item alone,
since handwritten items share no template. holm adjusts the McNemar p-values of the four preregistered primary
comparisons together (FT against B0, B1 and B4, FT+R against B2); M3 enters the last at p = 1, since only M4
has FT+R. The intervals are not adjusted for multiplicity, and the results per category and end to end are
described, not tested.

Non-inferiority (M6, docs/m6-计划.md §5, H2 and H3). noninferiority_p is the one-sided paired bootstrap: with
d the paired item differences and Δ*_b the mean of the b-th resample of them (the items drawn with
replacement, the same draws bootstrap_ci makes from the same seed), p_NI = #{b : Δ*_b <= -margin} / B, the
share of the draws at or below minus the margin, which is small when the interval's low end lies above it.
cluster_noninferiority_p is the same over the cluster bootstrap's draws (descriptive).
stratified_noninferiority_p is the stratified version for ACEBench's (language, category) strata: within
every stratum the items are drawn with replacement, the stratum's difference of success rates is taken on
the draw, and Δ* is the mean of those differences over the strata (the README's aggregate is the mean over
strata); p_NI as above.
compare_systems adds both item-level p-values under "noninferiority" when given a margin. H1 takes
McNemar's p, H2 and H3 p_NI, and holm adjusts the three together.
"""

from __future__ import annotations

import itertools
import math
import random
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction

from jsonschema import Draft202012Validator

from ..agent.types import ToolCall, call_key
from ..canonical import canonical_json
from ..normalize import fold_hyphens, normalize_arxiv_id, normalize_date, normalize_text, query_contains_all
from ..tools.schemas import DEFAULTS, TOOLS
from .items import CALL, CLARIFY, LANGS, NEXT_STEP, NO_CALL, PARALLEL, RECOVER, TYPES
from .lock import refuse_test_material
from .runner import OK, RESULT_STATUSES

ASK = "ask_clarification"

WRONG_TOOL = "选错工具"
MISSING_PARAM = "缺参"
EXTRA_PARAM = "多参"
WRONG_VALUE = "值或格式错"
FABRICATED_ID = "编号编造"
UNWANTED_CALL = "不该调却调"
NO_CLARIFY = "该追问没追问"
BAD_RECOVERY = "恢复不当"
RETRY_STORM = "重试风暴"
BROKEN_JSON = "JSON 损坏"
THINK_LEAK = "思考块泄漏"
# The plan's order (docs/plan.md, 误差分类法), the order summarize reports them in.
ERROR_CLASSES = (WRONG_TOOL, MISSING_PARAM, EXTRA_PARAM, WRONG_VALUE, FABRICATED_ID, UNWANTED_CALL,
                 NO_CLARIFY, BAD_RECOVERY, RETRY_STORM, BROKEN_JSON, THINK_LEAK)
# The order of precedence: the structural classes first, then the plan's order.
PRECEDENCE = (BROKEN_JSON, THINK_LEAK, WRONG_TOOL, MISSING_PARAM, EXTRA_PARAM, WRONG_VALUE, FABRICATED_ID,
              UNWANTED_CALL, NO_CLARIFY, BAD_RECOVERY, RETRY_STORM)

CHECKS = ("json_valid", "no_think_leak", "tool_choice", "arguments", "exact_args", "query_strict",
          "ids_grounded", "no_false_call", "clarify", "recover")
OUTPUT_CHECKS = ("json_valid", "no_think_leak", "ids_grounded", "no_false_call")  # None without an output
_CALL_CHECKS = ("json_valid", "no_think_leak", "tool_choice", "arguments", "exact_args", "ids_grounded")
TYPE_CHECKS = {  # the checks that decide success, per item type
    CALL: _CALL_CHECKS,
    NEXT_STEP: _CALL_CHECKS,
    PARALLEL: _CALL_CHECKS,
    RECOVER: (*_CALL_CHECKS, "recover"),
    NO_CALL: ("json_valid", "no_think_leak", "tool_choice", "no_false_call", "ids_grounded"),
    CLARIFY: ("json_valid", "no_think_leak", "tool_choice", "clarify", "ids_grounded"),
}

# (tool, parameter) pairs whose schema says uniqueItems: a repeated element there is malformed.
UNIQUE_ITEMS = frozenset((t["function"]["name"], k) for t in TOOLS
                         for k, v in t["function"]["parameters"]["properties"].items()
                         if v.get("uniqueItems"))
_THINK_TAG = re.compile(r"</?think\s*>", re.IGNORECASE)

ID_PARAMS = frozenset({"arxiv_id"})
DATE_PARAMS = frozenset({"date_from", "date_to", "since"})
SET_PARAMS = frozenset({"categories", "tags", "missing"})
QUERY_PARAMS = frozenset({"query"})
_MAX_PERMUTED = 7  # larger same-name groups are paired in order instead of searched
# A new-style arXiv id with an optional version, not inside a longer number; normalize_arxiv_id filters
# the rest.
_ARXIV_ID = re.compile(r"(?<![\d.])(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)")

# The scorers (module docstring: Scorers): today's, and the normalized one, which adds S1 to S3.
DEFAULT_SCORER = "default"
NORMALIZED_SCORER = "normalized"
SCORERS = (DEFAULT_SCORER, NORMALIZED_SCORER)
# S1: the tools whose schema has a date_to (search_arxiv alone).
DATE_TO_TOOLS = frozenset(t["function"]["name"] for t in TOOLS
                          if "date_to" in t["function"]["parameters"]["properties"])
# S3: the counts a request that gives none leaves to the tool; search_arxiv.max_results is not one of them.
COUNT_PARAMS = frozenset({("search_notes", "top_k"), ("list_notes", "limit"),
                          ("fetch_fulltext", "max_chars")})
# agent.prompts: "你是科研助理,今天是 {today}。...": the date itself, read after normalize_text, in the
# forms normalize_date reads (2026-10-10, 2026/10/10, 2026.10.10, 2026年10月10日, 20261010), spaces
# allowed between the parts
_TODAY = re.compile(r"今天是\s*(\d{4}\s*[-/.年]\s*\d{1,2}\s*[-/.月]\s*\d{1,2}(?!\d)\s*日?|\d{8}(?!\d))")
# numbers_in: digit runs ("1,200" with its separators), runs of Chinese numerals, English number words.
_DIGITS = re.compile(r"\d+(?:,\d{3}(?!\d))*")
# ... and digit groups with a space as the thousands separator ("1 200", "1 200 000"), the first group of one
# to three digits and every other of three
_SPACED = re.compile(r"(?<!\d)\d{1,3}(?: \d{3}(?!\d))+")
_ZH_CHARS = "零〇一二两三四五六七八九十百千万"
# the financial numerals (壹 to 玖, 拾, 佰, 仟) as the ordinary ones they stand for
_FINANCIAL = str.maketrans("壹贰叁肆伍陆柒捌玖拾佰仟", "一二三四五六七八九十百千")
_ZH_NUMERAL = re.compile(f"[{_ZH_CHARS}]+")
_ZH_DIGITS = {digit: value for value, digits in enumerate(("零〇", "一", "二两", "三", "四", "五", "六", "七",
                                                           "八", "九"))
              for digit in digits}
_ZH_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10000}
# numbers_in, a number with a multiplier: ASCII numbers (a decimal point allowed) mixed with Chinese
# numerals and units, a space allowed between two parts ("2千", "1.5千", "1千2百", "2 千"); a number
# followed by k or w and no other letter ("1.2k", "3K", "1w")
_MIXED_PART = f"[0-9]+(?:\\.[0-9]+)?|[{_ZH_CHARS}]"
_MIXED_RUN = re.compile(f"(?:{_MIXED_PART})(?: ?(?:{_MIXED_PART}))*")
_MIXED_PARTS = re.compile(_MIXED_PART)
_MULTIPLIED = re.compile(r"([0-9]+(?:\.[0-9]+)?) *([kw])(?![a-z])", re.IGNORECASE)
_MULTIPLIERS = {"k": 1000, "w": 10000}
# ... and a number followed by English multiplier words, each singular or plural with a space before it or not
# ("2 thousand", "1.2 thousand", "3hundred", "2 dozens", "2 hundred thousand"): several such terms in a row
# add up, "and" allowed between them, and so does a number without a multiplier after them ("1 thousand 2
# hundred" 1200, "2 thousand and 50" 2050)
_EN_SCALES = {"hundred": 100, "thousand": 1000, "million": 1000000, "dozen": 12}
# a number before a multiplier word: digit groups with a comma or a space as the thousands separator ("1,200
# thousand", "1 200 thousand"), or digits with a decimal point; a space or a hyphen before the word or nothing
# ("2-thousand")
_NUMBER = r"(?:[0-9]{1,3}(?:[, ][0-9]{3}(?![0-9]))+|[0-9]+(?:\.[0-9]+)?)"
_EN_TERM = f"{_NUMBER}(?:[ -]?(?:{'|'.join(_EN_SCALES)})s?(?![a-z]))+"
_EN_TERMS = re.compile(f"{_EN_TERM}(?: (?:and )?{_EN_TERM})*(?: (?:and )?{_NUMBER}(?![0-9a-z]))?")
_EN_TERM_PARTS = re.compile(f"({_NUMBER})|({'|'.join(_EN_SCALES)})")
# the owner's quantity words (10-30): "a couple", "couple of", "a pair" and "pair of" are 2, 一打 12, 半打 and
# half a dozen ("half-dozen", "a half dozen") 6; no reader goes on to read half a dozen's dozen as 12, nor the
# 一 of 一些, the one vague word of the owner's that holds a numeral (the others, "a few", "few", "several",
# "some", 几, 几个 and 若干, hold none): a vague word is no count
_HALF_DOZEN = r"(?<![a-z])half(?:[ -]a)?[ -]dozen(?![a-z])"
_QUANTITIES = ((re.compile(r"(?<![a-z])(?:a couple|couple of|a pair|pair of)(?![a-z])"), 2),
               (re.compile("一打"), 12), (re.compile("半打"), 6), (re.compile(_HALF_DOZEN), 6))
# a vague word with a unit stays vague, so the unit is no count either: several dozen, a few hundred, some
# thousand; 几十, 十几 and 二十几, 几百, 数十, 上百, 成千
_VAGUE_EN = r"(?<![a-z])(?:several|a few|few|some)[ -](?:dozen|hundred|thousand|million)s?(?![a-z])"
_VAGUE_ZH = "[一二两三四五六七八九]?十几|几[十百千万]|数[十百千万]|[上成][百千万]"
_NO_COUNT = re.compile(f"{_HALF_DOZEN}|一些|{_VAGUE_EN}|{_VAGUE_ZH}")
_EN_WORD = re.compile(r"[a-z]+")
_EN_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
            "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_EN_TENS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_EN_NUMBERS = {**{word: n for n, word in enumerate(_EN_ONES)},
               **{word: 10 * n for n, word in enumerate(_EN_TENS, 2)}}
_EN_VALUES = {**_EN_NUMBERS, "hundred": 100, "thousand": 1000, "dozen": 12}


# ---- normalisation ---------------------------------------------------------------------------------------

def normalize_value(param: str, value: object, *, unique: bool = False) -> object:
    """value in the form compared for parameter param (module docstring: Normalisation); unique when the
    schema says uniqueItems, so that a list with a repeated element stays distinct from every set."""
    if isinstance(value, str):
        if param in ID_PARAMS:
            return normalize_arxiv_id(value) or normalize_text(value)
        if param in DATE_PARAMS:
            return normalize_date(value) or normalize_text(value)
        return normalize_text(value)
    if isinstance(value, list):
        values = [normalize_value("", v) for v in value]
        if param in SET_PARAMS:
            keys = [canonical_json(v) for v in values]
            if unique and len(set(keys)) < len(keys):
                return ("repeated", tuple(sorted(keys)))
            return ("set", tuple(sorted(set(keys))))
        return values
    if isinstance(value, dict):
        return {k: normalize_value(k, v) for k, v in value.items()}
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def normalize_call_arguments(name: str, arguments: dict | None) -> dict | None:
    """The tool's defaults filled in, then every value normalised; None stays None (not a JSON object)."""
    if arguments is None:
        return None
    return {k: normalize_value(k, v, unique=(name, k) in UNIQUE_ITEMS)
            for k, v in {**DEFAULTS.get(name, {}), **arguments}.items()}


def required_words(slots: dict, expected_query: str, *, fold: bool = False) -> list[str]:
    """The words a query must contain: slots["required_words"], else the topic terms the expected query
    has. fold (S2, module docstring: Scorers): every word, and the expected query the topic terms are looked
    for in, with its hyphens read as spaces (fold_hyphens)."""
    given = slots.get("required_words")
    if isinstance(given, list) and all(isinstance(w, str) for w in given):
        return [fold_hyphens(w) for w in given] if fold else list(given)
    topic = slots.get("topic")
    if isinstance(topic, str):
        terms = [topic]
    elif isinstance(topic, list):
        terms = [t for t in topic if isinstance(t, str)]
    else:
        terms = []
    if fold:
        terms, expected_query = [fold_hyphens(t) for t in terms], fold_hyphens(expected_query)
    return [t for t in terms if t.strip() and query_contains_all(expected_query, [t])]


def query_matches(got: str, want: str, slots: dict, *, strict: bool = False, fold: bool = False) -> bool:
    """Whether the query got matches the expected query want (module docstring: Normalisation); strict: the
    literal match of query_strict, which fold leaves alone; fold (S2): the hyphens of both queries and of the
    required words read as spaces before the containment test and the equality fallback."""
    if strict:
        return normalize_text(got) == normalize_text(want)
    if fold:
        got, want = fold_hyphens(got), fold_hyphens(want)
    words = required_words(slots, want, fold=fold)
    if words:
        return query_contains_all(got, words)
    return normalize_text(got).casefold() == normalize_text(want).casefold()


def _value_matches(name: str, param: str, got: object, want: object, scoring: _Scoring, strict: bool) -> bool:
    """Whether a value got of parameter param of a call of tool name matches the expected value want, under
    the item's scoring (S2 for a query, S3 for a count)."""
    if param in QUERY_PARAMS and isinstance(got, str) and isinstance(want, str):
        return query_matches(got, want, scoring.slots, strict=strict, fold=scoring.fold)
    return got == want or (scoring.takes_default(name, param, want) and got == DEFAULTS[name][param])


# ---- scorers ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class _Scoring:
    """How one item's calls compare (module docstring: Scorers): the item's slots, and under the normalized
    scorer today's date for S1 (None when the item gives none), fold for S2 and, for S3, whether the slots
    leave the count open (they name no count) and the numbers the request text writes."""

    slots: dict
    today: str | None = None
    fold: bool = False
    open_count: bool = False
    numbers: frozenset[int] = frozenset()

    def takes_default(self, name: str, param: str, want: object) -> bool:
        """S3: whether parameter param of a call of tool name may also be at the tool's schema default where
        the expected value is want: a count of COUNT_PARAMS that the request did not give, the slots naming no
        count and want being a whole number the request text does not write."""
        return (self.open_count and (name, param) in COUNT_PARAMS and isinstance(want, int)
                and not isinstance(want, bool) and want not in self.numbers)


def _check_scorer(scorer: object) -> None:
    if not isinstance(scorer, str) or scorer not in SCORERS:
        raise ValueError(f"scorer {scorer!r} is not one of {list(SCORERS)}")


def _scoring(item: dict, scorer: str) -> _Scoring:
    """The item's _Scoring under scorer; ValueError for a scorer that is not one of SCORERS."""
    _check_scorer(scorer)
    slots = item.get("slots") or {}
    if scorer == DEFAULT_SCORER:
        return _Scoring(slots)
    return _Scoring(slots, today_of(item), fold=True, open_count="count" not in slots,
                    numbers=numbers_in(request_text(item)))


def today_of(item: dict) -> str | None:
    """S1's today: the date the item's first system message gives as "今天是 {today}" (agent.prompts), as
    normalize_date writes it; None without a system message, without that text or without a real date in
    it. The date itself is read (_TODAY, after normalize_text), whatever follows it: "今天是 2026-10-10." and
    "今天是 2026 年 10 月 10 日" give 2026-10-10 too."""
    messages = [m for m in item.get("messages") or [] if isinstance(m, dict)]
    system = next((m for m in messages if m.get("role") == "system"), None)
    content = system.get("content") if system is not None else None
    found = _TODAY.search(normalize_text(content)) if isinstance(content, str) else None
    return normalize_date("".join(found.group(1).split())) if found else None


def request_text(item: dict) -> str:
    """The request text S3 reads: the item's user messages, one per line."""
    return "\n".join(m["content"] for m in item.get("messages") or [] if isinstance(m, dict)
                     and m.get("role") == "user" and isinstance(m.get("content"), str))


def numbers_in(text: str) -> frozenset[int]:
    """The whole numbers text writes (S3), read after normalize_text and casefold: digit runs ("1,200" also as
    1200, and "1 200", digit groups with a space as the thousands separator, also as 1200), runs of Chinese
    numerals (三 3, 二十 20, 一千二百 and 一千二 1200, 一五〇〇 1500), the financial ones read besides as the
    ordinary ones they stand for (伍 5, 壹仟贰佰 1200), and runs of English number words ("twenty-five" 25,
    "twelve hundred" 1200, "one hundred and twenty" 120, "a dozen" 12, "two dozen" 24): each run's value, and
    each digit run, numeral digit and number word alone as well. Numbers with a multiplier are read besides:
    ASCII numbers mixed with Chinese units (2千 2000, 1.5千 1500, 1万 10000, 1千2百 1200, 2 千 2000), a number
    followed by k or w (1.2k 1200, 3K 3000, 1.2w 12000), and a number followed by English multiplier words
    (2 thousand 2000, 1.2 thousand 1200, 2 dozen 24, 3 hundred 300, 2 thousands 2000; several terms in a row
    added up, 1 thousand 2 hundred 1200); a value that is not a whole number gives the two whole numbers
    around it. A number counts wherever it stands, a date's or an arXiv id's digits included: one more number
    found can only keep S3 from applying. The owner's quantity words (10-30) are counts too: "a couple",
    "couple of", "a pair" and "pair of" 2, 一打 12, 半打 and "half a dozen" 6. By the owner's rule a vague
    word is no count and half a dozen no 12: the readers do not see half a dozen ("half-dozen", "a half
    dozen") nor 一些 (the one vague word of the owner's that holds a numeral), so they find neither the 12 of
    the one nor the 1 of the other, as they did before, nor a number either was part of (二一些 gives no 21),
    nor one across either (the text on each side is read alone). In a text without them, every number found
    before the financial numerals, the spaced groups and the English multipliers were read is still found."""
    text = normalize_text(text).casefold()
    found = {count for pattern, count in _QUANTITIES if pattern.search(text)}
    for part in _NO_COUNT.split(text):  # the readers read no number in the words the rule takes out or across
        _read_numbers(part, found)
    return frozenset(found)


def _read_numbers(text: str, found: set[int]) -> None:
    """numbers_in's readers over text, normalised and casefolded, adding the whole numbers they find."""
    for match in _DIGITS.finditer(text):
        found.add(int(match.group().replace(",", "")))
        found.update(int(part) for part in match.group().split(","))
    for match in _SPACED.finditer(text):
        found.add(int(match.group().replace(" ", "")))
    for chinese in dict.fromkeys((text, text.translate(_FINANCIAL))):  # the financial numerals as well
        _read_chinese(chinese, found)
    for match in _MULTIPLIED.finditer(text):
        _add_whole(found, Fraction(match.group(1)) * _MULTIPLIERS[match.group(2)])
    for match in _EN_TERMS.finditer(text):
        _add_terms(found, match.group())
    words = _EN_WORD.findall(text)
    run: list[str] = []
    for k, word in enumerate([*words, ""]):  # "" ends the last run
        if word in _EN_VALUES:
            run.append(word)
            found.add(_EN_VALUES[word])
        elif word == "and" and run and words[k + 1:k + 2] and words[k + 1] in _EN_VALUES:
            continue  # "one hundred and twenty"
        elif run:
            found.add(_en_number(run))
            run = []


def _read_chinese(text: str, found: set[int]) -> None:
    """The numbers of text's Chinese numerals among the whole numbers found (numbers_in): each run of numerals
    with its digits alone, and ASCII numbers mixed with Chinese units."""
    for match in _ZH_NUMERAL.finditer(text):
        found.add(_zh_number(match.group()))
        found.update(_ZH_DIGITS[c] for c in match.group() if c in _ZH_DIGITS)
    for match in _MIXED_RUN.finditer(text):
        parts = _MIXED_PARTS.findall(match.group())
        if any(part[0] in "0123456789" for part in parts) and any(part in _ZH_UNITS for part in parts):
            _add_whole(found, _with_units(parts))


def _add_terms(found: set[int], run: str) -> None:
    """A run of numbers with English multiplier words (_EN_TERMS) among the whole numbers found: each term
    with a multiplier, its number times every multiplier word after it, and the run, the sum of its terms."""
    terms: list[list[Fraction]] = []  # each term's number, then its multipliers
    for number, scale in _EN_TERM_PARTS.findall(run):
        if number:
            terms.append([Fraction(number.replace(",", "").replace(" ", ""))])  # "1,200", "1 200": 1200
        else:
            terms[-1].append(Fraction(_EN_SCALES[scale]))
    for term in terms:
        if len(term) > 1:
            _add_whole(found, math.prod(term))
    _add_whole(found, sum(math.prod(term) for term in terms))


def _zh_number(run: str) -> int:
    """A run of Chinese numerals as a number: without a unit (十, 百, 千, 万) its digits one by one
    (一五〇〇 1500); else the digits times their units (_with_units: 一千零五 1005, 二十 20, 十五 15, 一千二
    1200, 一万二 12000)."""
    if not any(c in _ZH_UNITS for c in run):
        return int("".join(str(_ZH_DIGITS[c]) for c in run))
    return int(_with_units(list(run)))


def _with_units(parts: list[str]) -> Fraction:
    """Numbers and Chinese units as one number: each number (a Chinese numeral digit, or ASCII digits with a
    decimal point allowed) times the unit after it (十, 百, 千, 万; 1 where a unit has no number before it),
    万 multiplying the whole section before it (一万二千 12000, 1万2千 12000, 3千万 30000000), and a last
    digit of one character right after a unit taking the next unit down (一千二 1200, 1千2 1200; 2千50 is
    2050)."""
    total = section = digit = Fraction(0)
    for part in parts:
        if part == "万":
            total += (section + digit or 1) * _ZH_UNITS[part]
            section = digit = Fraction(0)
        elif part in _ZH_UNITS:
            section += (digit or 1) * _ZH_UNITS[part]
            digit = Fraction(0)
        else:
            digit = Fraction(_ZH_DIGITS[part]) if part in _ZH_DIGITS else Fraction(part)
    if len(parts) > 1 and parts[-2] in _ZH_UNITS and len(parts[-1]) == 1 and parts[-1] not in _ZH_UNITS:
        digit *= _ZH_UNITS[parts[-2]] // 10
    return total + section + digit


def _add_whole(found: set[int], value: Fraction) -> None:
    """value among the whole numbers found, or the two whole numbers around it when it is not one."""
    if value.denominator == 1:
        found.add(int(value))
    else:
        found.update((math.floor(value), math.ceil(value)))


def _en_number(words: list[str]) -> int:
    """A run of English number words as a number: "twenty five" 25, "twelve hundred" 1200, "one thousand five
    hundred" 1500, "two dozen" 24."""
    total = current = 0
    for word in words:
        if word == "hundred":
            current = (current or 1) * 100
        elif word == "dozen":
            current = (current or 1) * 12
        elif word == "thousand":
            total += (current or 1) * 1000
            current = 0
        else:
            current += _EN_NUMBERS[word]
    return total + current


# ---- calls -----------------------------------------------------------------------------------------------

# Tools whose other parameters are free text, never compared (module docstring, Normalisation).
COMPARED = {ASK: frozenset({"missing"}), "write_note": frozenset({"arxiv_id", "mode"})}
# The tools of COMPARED whose uncompared parameters must still pass the executor's gate (module docstring,
# Normalisation): a validator per parameter of the tool's schema, which validation.check_call applies whole.
SHAPE_CHECKED = {t["function"]["name"]: {k: Draft202012Validator(v)
                                         for k, v in t["function"]["parameters"]["properties"].items()}
                 for t in TOOLS if t["function"]["name"] == "write_note"}


class _Call:
    """A call with its compared arguments, normalised (None when they are not a JSON object): every parameter,
    except for the tools in COMPARED, which keep only the parameters listed there, so that their free text is
    left out of the matching, the error classes and the F1. explicit: the parameters the call set itself,
    before the defaults were filled in. For a tool in SHAPE_CHECKED, what the executor's gate would refuse
    among the rest: unknown, the parameters the tool does not have, and malformed, the uncompared ones whose
    value the tool's schema refuses (both empty for every other tool). today (S1, module docstring: Scorers;
    None under the default scorer): a date_to on that day, in a call of DATE_TO_TOOLS, is dropped before
    anything else, as if the call had left it out."""

    def __init__(self, name: str, arguments: dict | None, today: str | None = None) -> None:
        self.name = name
        if today is not None and arguments is not None and name in DATE_TO_TOOLS \
                and _on_day(arguments.get("date_to"), today):
            arguments = {k: v for k, v in arguments.items() if k != "date_to"}
        self.explicit = frozenset(arguments or ())
        args = normalize_call_arguments(name, arguments)
        keys = COMPARED.get(name)
        self.args = args if args is None or keys is None else {k: v for k, v in args.items() if k in keys}
        self.unknown = self.malformed = frozenset()
        schema = SHAPE_CHECKED.get(name)
        if schema is not None and arguments is not None:
            self.unknown = frozenset(k for k in arguments if k not in schema)
            self.malformed = frozenset(k for k, v in arguments.items() if k in schema
                                       and k not in COMPARED[name] and not schema[k].is_valid(v))


def _on_day(value: object, today: str) -> bool:
    """Whether a date_to value is the day today (S1), compared after normalize_date."""
    return isinstance(value, str) and normalize_date(value) == today


def _output_calls(output: dict) -> list[ToolCall]:
    calls = output.get("tool_calls") or []
    if not isinstance(calls, list) or not all(isinstance(c, dict) for c in calls):
        raise ValueError("output.tool_calls must be a list of objects")
    return [ToolCall.from_api(c) for c in calls]


def _expected_calls(item: dict, today: str | None = None) -> list[_Call]:
    expected = item.get("expected")
    calls = expected.get("tool_calls") if isinstance(expected, dict) else None
    if not isinstance(calls, list) or not all(isinstance(c, dict) and isinstance(c.get("name"), str)
                                              and isinstance(c.get("arguments"), dict) for c in calls):
        raise ValueError(f"item {item.get('id')!r}: expected.tool_calls must be a list of "
                         f"{{name, arguments}}")
    return [_Call(c["name"], c["arguments"], today) for c in calls]


def _args_match(got: _Call, want: _Call, scoring: _Scoring, strict: bool = False) -> bool:
    """Whether the called call got matches want: the same name, compared arguments that match, and nothing the
    executor's gate would refuse (SHAPE_CHECKED)."""
    if got.name != want.name or got.args is None or want.args is None or got.unknown or got.malformed:
        return False
    g, w = got.args, want.args
    return g.keys() == w.keys() and all(_value_matches(want.name, k, g[k], w[k], scoring, strict) for k in w)


def _matching_size(got: list[_Call], want: list[_Call], ok) -> int:
    """Size of a maximum one-to-one matching between want and got under ok(got, want) (Kuhn's algorithm)."""
    owner: dict[int, int] = {}

    def augment(w: int, seen: set[int]) -> bool:
        for g in range(len(got)):
            if g not in seen and ok(got[g], want[w]):
                seen.add(g)
                if g not in owner or augment(owner[g], seen):
                    owner[g] = w
                    return True
        return False

    return sum(augment(w, set()) for w in range(len(want)))


def _agreeing(got: _Call, want: _Call, scoring: _Scoring) -> int:
    if got.args is None or want.args is None:
        return 0
    return sum(1 for k, v in want.args.items()
               if k in got.args and _value_matches(want.name, k, got.args[k], v, scoring, False))


def _best_pairs(got: list[_Call], want: list[_Call], scoring: _Scoring) -> tuple[list[tuple[_Call, _Call]],
                                                                                 list[_Call], list[_Call]]:
    """Same-name pairs agreeing on the most parameters; then the unpaired called and expected calls."""
    pairs, lone_got, lone_want = [], [], []
    for name in dict.fromkeys([c.name for c in want] + [c.name for c in got]):
        g = [c for c in got if c.name == name]
        w = [c for c in want if c.name == name]
        small, large, got_small = (g, w, True) if len(g) <= len(w) else (w, g, False)
        if len(large) <= _MAX_PERMUTED:
            options = itertools.permutations(large, len(small))
        else:
            options = iter([tuple(large[:len(small)])])
        best, best_score = None, -1
        for option in options:
            chosen = [(s, o) if got_small else (o, s) for s, o in zip(small, option, strict=True)]
            score = sum(_agreeing(a, b, scoring) for a, b in chosen)
            if score > best_score:
                best, best_score = chosen, score
        pairs += best or []
        used_got = {id(a) for a, _ in best or []}
        used_want = {id(b) for _, b in best or []}
        lone_got += [c for c in g if id(c) not in used_got]
        lone_want += [c for c in w if id(c) not in used_want]
    return pairs, lone_got, lone_want


def _param_counts(got: list[_Call], want: list[_Call], scoring: _Scoring) -> dict[str, list[int]]:
    """{"tool.parameter": [tp, fp, fn]} over the item, sorted by key; a parameter that neither call of a pair
    (or a lone call itself) set explicitly is not counted."""
    counts: dict[str, list[int]] = {}

    def add(name: str, param: str, slot: int) -> None:
        counts.setdefault(f"{name}.{param}", [0, 0, 0])[slot] += 1

    pairs, lone_got, lone_want = _best_pairs(got, want, scoring)
    for g, w in pairs:
        g_args, w_args = g.args or {}, w.args or {}
        explicit = g.explicit | w.explicit
        for k, v in g_args.items():
            if k in explicit:
                agree = k in w_args and _value_matches(w.name, k, v, w_args[k], scoring, False)
                add(g.name, k, 0 if agree else 1)
        for k, v in w_args.items():
            if k in explicit and not (k in g_args
                                      and _value_matches(w.name, k, g_args[k], v, scoring, False)):
                add(w.name, k, 2)
    for g in lone_got:
        for k in g.args or {}:
            if k in g.explicit:
                add(g.name, k, 1)
    for w in lone_want:
        for k in w.args or {}:
            if k in w.explicit:
                add(w.name, k, 2)
    return dict(sorted(counts.items()))


def prf(tp: int, fp: int, fn: int) -> dict:
    """precision, recall and F1; each None when its denominator is 0."""
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


# ---- ids -------------------------------------------------------------------------------------------------

def arxiv_ids(text: str) -> set[str]:
    """The canonical arXiv ids that occur in text."""
    return {i for i in (normalize_arxiv_id(m.group(1)) for m in _ARXIV_ID.finditer(text)) if i}


def _message_text(message: dict) -> str:
    parts = [message.get("content") if isinstance(message.get("content"), str) else ""]
    for c in message.get("tool_calls") or []:
        arguments = (c.get("function") or {}).get("arguments") if isinstance(c, dict) else None
        parts.append(arguments if isinstance(arguments, str) else canonical_json(arguments or {}))
    return "\n".join(parts)


def _previous_call_keys(messages: list[dict]) -> set[str]:
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            return {call_key(ToolCall.from_api(c)) for c in message["tool_calls"]}
    return set()


# ---- scoring ---------------------------------------------------------------------------------------------

def _check_item(item: dict) -> None:
    if not isinstance(item.get("messages"), list):
        raise ValueError(f"item {item.get('id')!r} has no messages list")  # noqa: TRY004  malformed input
    if item.get("type") not in TYPE_CHECKS:
        raise ValueError(f"item {item.get('id')!r}: type {item.get('type')!r} is not one of {list(TYPES)}")


def _base(result: dict, item: dict) -> dict:
    return {"id": item.get("id"), "type": item["type"], "lang": item.get("lang"),
            "family": item.get("family"), "status": result.get("status")}


def score_item(result: dict, item: dict, scorer: str = DEFAULT_SCORER) -> dict:
    """The checks of one runner record against its item (module docstring), with success and error_class,
    under scorer: DEFAULT_SCORER, or NORMALIZED_SCORER with S1 to S3 (module docstring: Scorers). The score
    has the same keys under either.

    ValueError when the record or the item is malformed (no messages, an unknown type, an ok record without an
    output object, expected calls without name and arguments), and for a scorer that is not one of SCORERS.
    """
    _check_item(item)
    scoring = _scoring(item, scorer)
    applies = TYPE_CHECKS[item["type"]]
    want = _expected_calls(item, scoring.today)
    has_query = any(w.args and any(k in QUERY_PARAMS for k in w.args) for w in want)
    status = result.get("status")
    if status not in RESULT_STATUSES:
        raise ValueError(f"result {result.get('id')!r}: unknown status {status!r}")
    if status != OK:  # no output: the output checks do not apply, the others fail
        failing = {*applies, *(("query_strict",) if has_query and "exact_args" in applies else ())}
        checks = {c: (False if c in failing and c not in OUTPUT_CHECKS else None) for c in CHECKS}
        counts = _param_counts([], want, scoring)
        return {**_base(result, item), **checks, "success": False, "error_class": status,
                "params": _sum_counts(counts), "param_counts": counts}
    output = result.get("output")
    if not isinstance(output, dict):
        raise ValueError(f"result {result.get('id')!r}: status ok without an output object")  # noqa: TRY004
    raw = _output_calls(output)
    got = [_Call(c.name, c.arguments, scoring.today) for c in raw]
    names_got, names_want = Counter(c.name for c in got), Counter(c.name for c in want)

    history = "\n".join(_message_text(m) for m in item["messages"] if isinstance(m, dict))
    produced = arxiv_ids(_message_text(output))
    values = {
        "json_valid": all(c.arguments is not None for c in raw),
        "no_think_leak": not _THINK_TAG.search(output.get("content") or ""),
        "tool_choice": set(names_got) == set(names_want),
        "arguments": _matching_size(got, want, lambda g, w: _args_match(g, w, scoring)) == len(want),
        "ids_grounded": produced <= arxiv_ids(history),
        "no_false_call": not got,
        "clarify": ASK in names_got and all(_args_match(g, w, scoring) for w in want if w.name == ASK
                                           and "missing" in (w.args or {}) for g in got if g.name == ASK),
    }
    values["exact_args"] = values["arguments"] and len(got) == len(want)
    values["recover"] = values["exact_args"]
    strict = _matching_size(got, want, lambda g, w: _args_match(g, w, scoring, strict=True)) == len(want)
    values["query_strict"] = strict and len(got) == len(want)

    checks = {}
    for check in CHECKS:
        if check == "query_strict":
            checks[check] = values[check] if has_query and "exact_args" in applies else None
        else:
            checks[check] = values[check] if check in applies else None
    success = all(checks[c] for c in applies)

    counts = _param_counts(got, want, scoring)
    score = {**_base(result, item), **checks, "success": success, "error_class": None,
             "params": _sum_counts(counts), "param_counts": counts}
    if not success:
        repeat = bool(raw) and all(call_key(c) in _previous_call_keys(item["messages"]) for c in raw)
        score["error_class"] = _error_class(item["type"], got, want, names_got, names_want, scoring,
                                            checks, values, repeat)
    return score


def _error_class(type_: str, got: list[_Call], want: list[_Call], names_got: Counter, names_want: Counter,
                 scoring: _Scoring, checks: dict, values: dict, repeat: bool) -> str | None:
    if not values["json_valid"]:
        return BROKEN_JSON
    if not values["no_think_leak"]:
        return THINK_LEAK
    attempt = not repeat
    if attempt and want and names_got != names_want \
            and not (type_ == CLARIFY and ASK not in names_got) and not (type_ == RECOVER and not got):
        return WRONG_TOOL
    if attempt and want and names_got == names_want:
        pairs, _, _ = _best_pairs(got, want, scoring)
        missing = extra = wrong = False
        for g, w in pairs:
            if g.args is None or w.args is None:
                wrong = True
                continue
            g_args, w_args = g.args, w.args
            missing |= bool(w_args.keys() - g_args.keys())
            extra |= bool(g_args.keys() - w_args.keys() or g.unknown)  # a parameter the tool does not have
            wrong |= bool(g.malformed) or any(not _value_matches(w.name, k, g_args[k], v, scoring, False)
                                              for k, v in w_args.items() if k in g_args)
        for flag, name in ((missing, MISSING_PARAM), (extra, EXTRA_PARAM), (wrong, WRONG_VALUE)):
            if flag:
                return name
    if not values["ids_grounded"]:
        return FABRICATED_ID
    if not want and got:
        return UNWANTED_CALL
    if type_ == CLARIFY and ASK not in names_got:
        return NO_CLARIFY
    if type_ == RECOVER:
        return BAD_RECOVERY
    if repeat:
        return RETRY_STORM
    return None


def _sum_counts(counts: dict[str, list[int]]) -> dict:
    tp, fp, fn = (sum(c[k] for c in counts.values()) for k in range(3))
    return prf(tp, fp, fn)


def score_items(results: list[dict], items: list[dict], scorer: str = DEFAULT_SCORER) -> list[dict]:
    """score_item for each record under scorer (module docstring: Scorers), joined to its item by id;
    ValueError (IdMismatch) when the ids differ and for a scorer that is not one of SCORERS, and
    RefusedTestSet for test material other than the OpenedTestSet of a LockedSet.open (eval.lock, who opens
    what)."""
    refuse_test_material(items, "score_items' items")
    _check_scorer(scorer)
    by_id = {}
    for item in items:
        if not isinstance(item.get("id"), str) or item["id"] in by_id:
            raise ValueError(f"item ids must be distinct strings; bad or repeated: {item.get('id')!r}")
        by_id[item["id"]] = item
    result_ids = [r.get("id") for r in results]
    if len(set(result_ids)) != len(result_ids) or set(result_ids) != set(by_id):
        missing = sorted(set(by_id) - set(result_ids))[:5]
        unknown = sorted(str(i) for i in set(result_ids) - set(by_id))[:5]
        raise IdMismatch(f"results and items differ: no result for {missing}, no item for {unknown}, "
                         f"{len(result_ids) - len(set(result_ids))} repeated result id(s)")
    return [score_item(r, by_id[r["id"]], scorer) for r in results]


class IdMismatch(ValueError):
    """The records and the items do not have the same ids."""


# ---- summary ---------------------------------------------------------------------------------------------

def _rate(values: list[bool | None]) -> tuple[float | None, int]:
    applying = [v for v in values if v is not None]
    return (sum(applying) / len(applying) if applying else None), len(applying)


def _groups(keys: list[str | None], known: Sequence[str]) -> list[str]:
    seen = {k for k in keys if k is not None}
    return [k for k in known] + sorted(seen - set(known))


def summarize(scores: list[dict], items: list[dict]) -> dict:
    """Aggregate rates over the scores, with n per group; keys in a fixed order (module docstring)."""
    by_id = {item["id"]: item for item in items}
    if len(scores) != len(items) or any(s["id"] not in by_id for s in scores):
        raise IdMismatch("the scores and the items do not have the same ids")
    types = [by_id[s["id"]]["type"] for s in scores]
    langs = [by_id[s["id"]].get("lang") for s in scores]

    def group(keys: list, known: Sequence[str]) -> dict:
        out = {}
        for key in _groups(keys, known):
            chosen = [s["success"] for s, k in zip(scores, keys, strict=True) if k == key]
            rate, n = _rate(chosen)
            out[key] = {"n": n, "success_rate": rate}
        return out

    summary: dict = {"n": len(scores), "n_ok": sum(s["status"] == OK for s in scores)}
    summary["success_rate"] = _rate([s["success"] for s in scores])[0]
    summary["by_type"] = group(types, TYPES)
    summary["by_lang"] = group(langs, LANGS)
    rates, n_per_check = {}, {}
    for label, check in (("json_valid_rate", "json_valid"), ("tool_choice_accuracy", "tool_choice"),
                         ("argument_match_rate", "arguments"), ("exact_args_rate", "exact_args"),
                         ("query_strict_rate", "query_strict"), ("ids_grounded_rate", "ids_grounded"),
                         ("clarify_accuracy", "clarify"), ("recover_rate", "recover")):
        rates[label], n_per_check[check] = _rate([s.get(check) for s in scores])
    no_false, n_per_check["no_false_call"] = _rate([s.get("no_false_call") for s in scores])
    rates["false_call_rate"] = None if no_false is None else 1 - no_false
    no_leak, n_per_check["no_think_leak"] = _rate([s.get("no_think_leak") for s in scores])
    rates["think_leak_rate"] = None if no_leak is None else 1 - no_leak
    summary.update(rates)

    totals: dict[str, list[int]] = {}
    for s in scores:
        for key, counts in s["param_counts"].items():
            total = totals.setdefault(key, [0, 0, 0])
            for k in range(3):
                total[k] += counts[k]
    per_param = {key: prf(*totals[key])["f1"] for key in sorted(totals)}
    f1s = [f for f in per_param.values() if f is not None]
    summary["macro_param_f1"] = sum(f1s) / len(f1s) if f1s else None
    summary["param_f1"] = per_param
    summary["n_per_check"] = n_per_check
    found = Counter(s["error_class"] for s in scores if s["error_class"] is not None)
    order = list(ERROR_CLASSES) + [st for st in RESULT_STATUSES if st != OK]
    classes = {c: found.get(c, 0) for c in ERROR_CLASSES}
    classes.update({c: found[c] for c in order[len(ERROR_CLASSES):] if found.get(c)})
    classes.update({c: found[c] for c in sorted(set(found) - set(order))})
    summary["error_classes"] = classes
    summary["unclassified_failures"] = sum(not s["success"] and s["error_class"] is None for s in scores)
    return summary


# ---- comparing two systems -------------------------------------------------------------------------------

def _paired(successes_a: Sequence[bool], successes_b: Sequence[bool]) -> list[int]:
    if len(successes_a) != len(successes_b) or not successes_a:
        raise ValueError("two non-empty success lists of the same length are needed")
    return [int(bool(a)) - int(bool(b)) for a, b in zip(successes_a, successes_b, strict=True)]


def _percentile(ordered: list[float], q: float) -> float:
    """Linear interpolation between closest ranks (numpy's default); ordered is sorted."""
    position = (len(ordered) - 1) * q / 100
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _interval(diff: float, draws: list[float], seed: int, n: int) -> dict:
    draws.sort()
    return {"diff": diff, "mean": sum(draws) / n, "low": _percentile(draws, 2.5),
            "high": _percentile(draws, 97.5), "n": n, "seed": seed}


def _paired_draws(d: list[int], seed: int, n: int) -> list[float]:
    """n means of the paired differences d over items drawn with replacement, from random.Random(seed)."""
    rng = random.Random(seed)
    m = len(d)
    return [sum(rng.choices(d, k=m)) / m for _ in range(n)]


def bootstrap_ci(successes_a: Sequence[bool], successes_b: Sequence[bool], seed: int, n: int = 10000) -> dict:
    """Paired bootstrap of rate(a) - rate(b): items drawn with replacement, n times, from random.Random(seed).

    {"diff": the observed difference, "mean": the mean of the draws, "low"/"high": their 2.5 and 97.5
    percentiles, "n", "seed"}. The same seed gives the same interval.
    """
    d = _paired(successes_a, successes_b)
    return _interval(sum(d) / len(d), _paired_draws(d, seed, n), seed, n)


def _clusters(d: list[int], families: Sequence[object]) -> list[tuple[int, int]]:
    """(sum of the differences, items) per family, in the order of the families' canonical JSON."""
    if len(families) != len(d):
        raise ValueError("one family per item is needed")
    sums: dict[str, list[int]] = {}
    for family, value in zip(families, d, strict=True):
        cell = sums.setdefault(canonical_json(family), [0, 0])
        cell[0] += value
        cell[1] += 1
    return [tuple(sums[k]) for k in sorted(sums)]


def _cluster_draws(clusters: list[tuple[int, int]], seed: int, n: int) -> list[float]:
    """n differences over whole families drawn with replacement, from random.Random(seed)."""
    rng = random.Random(seed)
    draws = []
    for _ in range(n):
        chosen = rng.choices(clusters, k=len(clusters))
        draws.append(sum(c[0] for c in chosen) / sum(c[1] for c in chosen))
    return draws


def cluster_bootstrap_ci(successes_a: Sequence[bool], successes_b: Sequence[bool], families: Sequence[object],
                         seed: int, n: int = 10000) -> dict:
    """bootstrap_ci with whole families (template families) drawn instead of items; each draw's difference is
    over all the items of the drawn families. Same keys plus "clusters"."""
    d = _paired(successes_a, successes_b)
    clusters = _clusters(d, families)
    draws = _cluster_draws(clusters, seed, n)
    return {**_interval(sum(d) / len(d), draws, seed, n), "clusters": len(clusters)}


def _below(draws: list[float], margin: float) -> int:
    return sum(draw <= -margin for draw in draws)


def _check_margin(margin: object) -> float:
    if isinstance(margin, bool) or not isinstance(margin, int | float) or not math.isfinite(margin):
        raise ValueError(f"the margin must be a finite number, got {margin!r}")
    return float(margin)


def noninferiority_p(successes_a: Sequence[bool], successes_b: Sequence[bool], margin: float, seed: int,
                     n: int = 10000) -> dict:
    """The one-sided paired bootstrap p-value of non-inferiority (module docstring: Non-inferiority).

    H0: rate(a) - rate(b) <= -margin. With d the paired differences a_i - b_i of the items and Δ*_b the mean
    of d over the b-th resample of the items (drawn with replacement, as bootstrap_ci draws them from the
    same seed), p_NI = #{b : Δ*_b <= -margin} / n: the share of the n draws at or below minus the margin.
    {"margin", "diff": the observed difference, "p_ni", "below": the draws counted, "n", "seed"}. The same
    seed gives the same value; identical inputs give 0 for a positive margin.
    """
    margin = _check_margin(margin)
    d = _paired(successes_a, successes_b)
    below = _below(_paired_draws(d, seed, n), margin)
    return {"margin": margin, "diff": sum(d) / len(d), "p_ni": below / n, "below": below, "n": n,
            "seed": seed}


def cluster_noninferiority_p(successes_a: Sequence[bool], successes_b: Sequence[bool],
                             families: Sequence[object], margin: float, seed: int, n: int = 10000) -> dict:
    """noninferiority_p over the cluster bootstrap's draws (cluster_bootstrap_ci: whole families drawn with
    replacement): the descriptive cluster version. noninferiority_p's keys plus "clusters"."""
    margin = _check_margin(margin)
    d = _paired(successes_a, successes_b)
    clusters = _clusters(d, families)
    below = _below(_cluster_draws(clusters, seed, n), margin)
    return {"margin": margin, "diff": sum(d) / len(d), "p_ni": below / n, "below": below, "n": n,
            "seed": seed, "clusters": len(clusters)}


def stratified_noninferiority_p(strata_a: Mapping[object, Sequence[bool]],
                                strata_b: Mapping[object, Sequence[bool]], margin: float, seed: int,
                                n: int = 10000) -> dict:
    """The one-sided stratified paired bootstrap p-value of non-inferiority (module docstring:
    Non-inferiority; docs/m6-计划.md §5, H3).

    strata_a and strata_b map each stratum (ACEBench's (language, category)) to the successes of the same
    items in the same order, paired by position. The effect is the mean over the strata of the stratum's
    difference of success rates, rate_a - rate_b, as the README's ACEBench aggregate is the mean over the
    strata. Each of the n resamples draws, within every stratum, as many items as it holds with replacement
    (the same items for both systems), takes the stratum's difference on the draw and averages it over the
    strata into Δ*_b; p_NI = #{b : Δ*_b <= -margin} / n. The draws come from random.Random(seed), the strata
    in strata_a's order. {"margin", "diff": the observed mean difference, "mean": the mean of the draws,
    "low"/"high": their 2.5 and 97.5 percentiles, "p_ni", "below", "strata", "items", "n", "seed"}.
    ValueError for no strata, strata that are not the same on both sides, and a stratum that is empty or of
    different lengths on the two sides. A stratum with one item is resampled like any other (it always
    gives its own difference).
    """
    margin = _check_margin(margin)
    if not strata_a:
        raise ValueError("no strata to compare")
    if set(strata_a) != set(strata_b):
        raise ValueError("the two sides do not hold the same strata")
    cells = []
    for key, successes in strata_a.items():
        other = strata_b[key]
        if not successes or len(successes) != len(other):
            raise ValueError(f"stratum {key!r}: two non-empty success lists of the same length are needed, "
                             f"got {len(successes)} and {len(other)}")
        cells.append(_paired(successes, other))
    rng = random.Random(seed)
    diff = sum(sum(d) / len(d) for d in cells) / len(cells)
    draws = []
    for _ in range(n):
        total = 0.0
        for d in cells:
            total += sum(rng.choices(d, k=len(d))) / len(d)
        draws.append(total / len(cells))
    below = _below(draws, margin)
    interval = _interval(diff, draws, seed, n)
    return {"margin": margin, "diff": diff, "mean": interval["mean"], "low": interval["low"],
            "high": interval["high"], "p_ni": below / n, "below": below, "strata": len(cells),
            "items": sum(len(d) for d in cells), "n": n, "seed": seed}


def mcnemar(successes_a: Sequence[bool], successes_b: Sequence[bool]) -> dict:
    """Exact two-sided McNemar test: under H0 the discordant pairs split as Binomial(b + c, 1/2).

    p = min(1, 2 * sum_{k <= min(b, c)} C(b + c, k) / 2^(b + c)), with b the items only a gets right and c the
    items only b gets right; p = 1 when there is no discordant pair.
    """
    _paired(successes_a, successes_b)
    only_a = sum(bool(a) and not bool(b) for a, b in zip(successes_a, successes_b, strict=True))
    only_b = sum(bool(b) and not bool(a) for a, b in zip(successes_a, successes_b, strict=True))
    total = only_a + only_b
    tail = Fraction(sum(math.comb(total, k) for k in range(min(only_a, only_b) + 1)), 2 ** total)
    return {"a_only": only_a, "b_only": only_b, "discordant": total, "p": float(min(Fraction(1), 2 * tail))}


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm's step-down adjustment of the p-values of m comparisons, returned in their input order.

    The k-th smallest p (k = 1 .. m) becomes min(1, (m - k + 1) p), raised to the largest adjusted value of
    a smaller p, so the adjusted values are ordered as the raw ones and none falls below its raw value. Ties
    get the same adjusted value in any order: the sort is stable, and the running maximum carries the value
    of a tie's first member, which has the largest factor, to the others. ValueError names the first entry
    that is no p-value: a bool, anything but an int or a float, NaN, or a number outside [0, 1].
    """
    values = []
    for k, p in enumerate(p_values):
        if isinstance(p, bool) or not isinstance(p, int | float) or not 0 <= p <= 1:
            raise ValueError(f"p_values[{k}] is not a p-value in [0, 1]: {p!r}")
        values.append(float(p))
    m = len(values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, k in enumerate(sorted(range(m), key=lambda k: values[k])):
        running = max(running, min(1.0, (m - rank) * values[k]))
        adjusted[k] = running
    return adjusted


def cluster_keys(items: Sequence[dict]) -> list[tuple[str, str | int]]:
    """The cluster of each item for cluster_bootstrap_ci, in the items' order (the M3 plan, default 2).

    A drafted item (source.request, a string: eval.items) clusters with the other items of its template
    family, ("family", family); a handwritten item (source.handwritten true: eval.handwritten) is a cluster of
    its own, ("item", id), since handwritten items share no template: their family,
    "handwritten/<type>/<lang>", only names their stratum. The tag keeps a family and an item id apart where
    they are spelled alike. ValueError, naming the item by its id (by its position, #1 for the first, when it
    has none), for an item whose kind cannot be determined (no source object, or neither or both of the two
    marks), a drafted item without a template family (a non-empty string or an int) and a handwritten item
    without an id.
    """
    keys: list[tuple[str, str | int]] = []
    for number, item in enumerate(items, 1):
        if not isinstance(item, dict):
            raise ValueError(f"#{number} is not an item: {item!r}")  # noqa: TRY004  malformed input
        name = item["id"] if isinstance(item.get("id"), str) else f"#{number}"
        source = item.get("source")
        drafted = isinstance(source, dict) and isinstance(source.get("request"), str)
        handwritten = isinstance(source, dict) and source.get("handwritten") is True
        if drafted == handwritten:
            raise ValueError(f"item {name}: cannot tell whether it is drafted (source.request) or "
                             f"handwritten (source.handwritten true); source {source!r}")
        if handwritten:
            if not isinstance(item.get("id"), str):
                raise ValueError(f"handwritten item {name} has no id to be its own cluster")
            keys.append(("item", item["id"]))
            continue
        family = item.get("family")
        if isinstance(family, bool) or not isinstance(family, str | int) or family == "":
            raise ValueError(f"drafted item {name} has no template family: {family!r}")
        keys.append(("family", family))
    return keys


def _successes(scores: Sequence[dict], ids: list[str], name: str) -> list[bool]:
    """The success of each id's score, in the order of ids; IdMismatch, naming the list, unless scores hold
    exactly one score per id (a score whose id is not a string is a mismatch); ValueError for a score that is
    not an object."""
    for position, score in enumerate(scores):
        if not isinstance(score, dict):
            raise ValueError(f"{name}: entry {position} is a {type(score).__name__}, not a score object")  # noqa: TRY004
        if not isinstance(score.get("id"), str):
            raise IdMismatch(f"{name}: entry {position} has the id {score.get('id')!r}, not a string")
    by_id = {s.get("id"): s for s in scores}
    if len(by_id) != len(scores) or by_id.keys() != set(ids):
        missing = sorted(set(ids) - by_id.keys())[:5]
        unknown = sorted(str(i) for i in by_id.keys() - set(ids))[:5]
        raise IdMismatch(f"{name} and the items differ: no score for {missing}, no item for {unknown}, "
                         f"{len(scores) - len(by_id)} repeated score id(s)")
    successes = []
    for i in ids:
        success = by_id[i].get("success")
        if not isinstance(success, bool):
            raise ValueError(f"{name}: the score of {i} has success {success!r}, not a bool")  # noqa: TRY004
        successes.append(success)
    return successes


def compare_systems(scores_a: Sequence[dict], scores_b: Sequence[dict], items: Sequence[dict], *,
                    seed: int = 42, n: int = 10000, margin: float | None = None) -> dict:
    """Systems a and b compared on the same items, from the scores score_items gave each (module docstring:
    Comparing two systems).

    Each list of scores is joined to the items by id, and the successes are paired in the items' order:
    {"n": the number of items, "success_a" and "success_b": the success rates, "diff": a's minus b's, as the
    bootstraps compute it, "paired_ci": bootstrap_ci, "cluster_ci": cluster_bootstrap_ci over
    cluster_keys(items), "mcnemar": mcnemar}; both bootstraps draw n resamples from seed, and the "n" of each
    interval is that count. With a margin, "noninferiority": {"margin", "p_ni", "below"} from
    noninferiority_p and {"cluster_p_ni", "cluster_below"} from cluster_noninferiority_p, over the same
    draws as the intervals (module docstring: Non-inferiority). McNemar's p is not adjusted here: holm
    adjusts the p-values of the preregistered comparisons together. ValueError for no items, items without
    distinct string ids, an item cluster_keys refuses and a score whose success is not a bool; IdMismatch,
    naming scores_a or scores_b, when a list does not hold exactly one score per item.
    """
    if not items:
        raise ValueError("no items to compare")
    ids = [item.get("id") if isinstance(item, dict) else None for item in items]
    seen: set[str] = set()
    for i in ids:
        if not isinstance(i, str) or i in seen:
            raise ValueError(f"item ids must be distinct strings; bad or repeated: {i!r}")
        seen.add(i)
    keys = cluster_keys(items)
    a, b = _successes(scores_a, ids, "scores_a"), _successes(scores_b, ids, "scores_b")
    paired = bootstrap_ci(a, b, seed, n)
    out = {"n": len(ids), "success_a": sum(a) / len(ids), "success_b": sum(b) / len(ids),
           "diff": paired["diff"], "paired_ci": paired,
           "cluster_ci": cluster_bootstrap_ci(a, b, keys, seed, n), "mcnemar": mcnemar(a, b)}
    if margin is not None:
        one_sided = noninferiority_p(a, b, margin, seed, n)
        clustered = cluster_noninferiority_p(a, b, keys, margin, seed, n)
        out["noninferiority"] = {"margin": one_sided["margin"], "p_ni": one_sided["p_ni"],
                                 "below": one_sided["below"], "cluster_p_ni": clustered["p_ni"],
                                 "cluster_below": clustered["below"]}
    return out

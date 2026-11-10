"""M8's special-mix: think-mode training samples and the refusal development sets A and B, built mechanically
from unused public records (docs/m8-计划.md WP1, WP2); scripts/build_special_mix.py draws the records and
writes the files.

Why. FT3 lost ACEBench's special group to its training data: no sample shows a request whose value is present
but invalid answered with a refusal, the self-replay covers only missing and irrelevant requests with two
sentences of its own, and the public samples that copy values verbatim all train no-think (docs/m8-计划.md,
诊断). This module builds every kind of sample the plan's WP1 table lists, in think mode, with the prompt
varied along several axes so that the model learns to read the rules it is given rather than one wording.

The kinds (KINDS; PLAN holds the per-language counts):
- incorrect: a value the request gives breaks its parameter's definition; the answer is the wrong-value
  sentence with the value and the parameter filled in. Half of them break it obviously, a quarter give the
  value in another format (only where the definition names a format), a quarter give a vague value (VAGUE);
  a tenth have two wrong values (the training items of revision 1: about a fifth, TRAIN_TWO_VALUES; at least
  a fifth put the wrong value on an optional parameter, OPTIONAL_SHARE; half of the format items take a format
  only the description implies, DESCRIBED_SHARE).
- missing: a required value is cut from the request; the answer is the missing-value sentence with the
  parameter(s) and the function. A quarter lack two values, a fifth lack a top-level object (its whole value
  is cut and only the top-level name is expected), some lack an optional value too, which is not named.
- irrelevant: no listed function serves the request (ToolACE's "no function" answers, glaive's refusals, and
  deepexi requests paired with another record's unrelated functions; when the refusals run out, English
  requests paired likewise, sharing no content word with the functions: shares_content_word); the answer is
  the out-of-scope sentence.
- priority: a wrong value and a missing value at once; the prompt says which is checked first (ORDERS) and
  the answer is that case's sentence only.
- ask: a missing value where the prompt says to ask through an ask function (ASK_FUNCTIONS); the answer is
  that call, so that "missing -> ask tool", which FT3's agent samples teach, keeps a place.
- valid_verbatim, valid_multiturn, valid_nearmiss: requests to be answered with the call; values are copied
  as written (Chinese values stay Chinese); a later turn changes a value or the topic and the latest value
  counts; a value close to a bound is still valid (a leap day, the maximum, an exact option).

Construction rules. A value is changed only when it is a top-level scalar or an item of a top-level array of
the one call of the record's first step, its gold value passes the constraint inferred for it, and its text
occurs exactly once in the request (occurrences: case-insensitive, an ASCII letter or digit at either end must
not continue into the text). Constraints (infer_constraint) come from the schema's keywords first (type, enum,
pattern, format, minimum/maximum, items.enum) and then from the description's common phrasings (a date or time
format such as YYYY-MM-DD or 格式为yyyy-MM-dd HH:mm:ss, ISO 8601, e-mail, URL, a three-letter code, a ZIP
code, a closed list such as 取值:A、B、C or "one of 'x', 'y'", a range such as 范围 1~100 or "between 1 and
6"). deepexi's schemas declare no required parameters, so its called function is shown with the gold call's
keys as required (REQUIRED_FROM_CALL); the other sources keep their own lists. A wrong value, a follow-up and
a topic switch are written in the request's own language (request_lang: read from its text, the record's
source only when the text writes no word), whatever the prompt's.

Gold calls fully grounded. A record makes no item when its gold call does not fit its schema (typed) or
when its request does not fully give the call (call_problem, for every builder, the value banks of the
multi-turn items included): a null, an empty string, list or object anywhere; a leaf the request does not
hold verbatim, in its own case (a number and a YYYY-MM-DD date may differ in trivial form, but a number inside
a name, the 1 of "Formula 1", gives nothing; an ASCII value only as whole words; a boolean needs a word of its
parameter's name); a value naming its own parameter (planName="planName"); a value given
once that fills two parameters; a Chinese value ending in a generic noun (LABEL_ENDS: 号码, 链接, 地址, 名称,
文件, 角色 ...) that the request does not introduce with 为, 是, 叫, a colon or quotes; a value naming its
parameter at any depth, as deepexi answers questions about an API: "实例ID是什么？" with
InstanceId="实例ID" (names_parameter), or made of its own description's words (names_own_parameter). The
hand check of the development sets and of a training sample added (rounds 1 to 4; the tests hold the
cases): a placeholder ("xxxxx", "example", 新的描述信息), calls of booleans alone, a value found only inside
another value (objectType "表" of "表C"), a value pointing at something (这段数据, "near me"), words in an
identifier parameter (WorksId="数据门户", movie_id="Inception"), a value the request names as another
parameter's thing (slot_mismatch) and a quoted value no call uses (unused_quote). A near miss moves a date,
a time or a number to its bound unless that would break the request (a date or a time beside another of its
kind, AM after a moved time, a number whose noun would no longer agree) or the request restates the old
value in other words; an enum near miss keeps the exact option the request gives (swapped options left
requests saying the opposite) and is dropped when the request names another option of the enum too, as
written, inflected or in other words (names_other_option: 反向 for Reverse beside Forward). An English
article before a replaced value agrees with it. A wrong value fits where the value stood (wrong_value_fits):
no word sticks to a currency or percent sign ("$a dozen or so", "some%"), and a vague amount replaces only a
number counting what follows it (40 hours, 每页30条; never 版本号为2的). A changed value of a multi-turn item
comes from the records of the same function (the build script's value banks).

Cuts. A missing value goes with its frame where that reads fluently (cut_with_frames): English, a
preposition with or without an article (and the participle or adjective it completes: based on X, released
in X, according to X) or a naming word before it when the value ends its noun phrase, a number's unit,
currency sign and "off" around it, "between X and Y" whole; Chinese, a clause that only labels it (the
parameter's label, never a clause holding the request's verb), its label (实例ID为X的; never reaching back
over the request's verb), a naming verb (名为X的), a preposition (在X的, 从X, 对X的), these three with a
postposition or an enumeration mark after them (工作空间ID为X下的, 项目名称为X、), 为X, X的, and a noun
before the cut that the text after it repeats (no 用户用户). Whether the result reads fluently is a heuristic
(reads_fluently, which compares with the request before the cut: no doubled word, no postposition opening
the request or after a verb, no sentence of one to three words, no participle, copula, hyphen or currency
sign left without its partner; and no word left dangling at the cut: a preposition, 包含, an approximating
word, a verb without the object or the amount the cut took, "and burned [500].", "a recipe that uses
[chicken]."); the hand check of this fix round's review added the cases rounds 1 to 6 found (the tests hold
them); a builder takes the first fluent cut among
its choices and says so in variant["cut"] ("frame", else "bare"), and no cut leaves a number still given in
words ("the third round": still_numbered). A top-level object is cut only whole: its span from the first leaf
to the last, with the braces, quotes, keys and joining words around it (and its parameter's own name, when
the cut reads fluently only so), and nothing of it may remain (object_span, object_left).

Near-duplicate requests. Two requests are near-duplicates (near_duplicate) when their normalised texts are
equal (NFKC, lower case, punctuation dropped, spaces collapsed), when their content signatures are equal
(English: the words outside a small stop and politeness list; Chinese: the text without punctuation, spaces
and particles), or when their tokens are at least 4/5 alike (Jaccard; English content words, Chinese
character pairs). An item's requests are its final user text and the original request of each record it
used (request_texts); RequestIndex finds near-duplicates without comparing every pair.

Templates. Two requests share a template (same_template) when their template tokens (template_tokens: the
English content words without names, numbers, identifiers and quoted values; the Chinese character pairs
within the runs of CJK characters) are at least half alike, or when the one's, three or more, are all among
the other's; TemplateIndex finds them by token. An item's template is read from its records' requests
(Core.originals), never from the history wrapping, follow-ups or call notation this module adds. The build
holds at most two development items of a template and keeps a development item's template out of the
training items of its kind.

Prompt variety. A system prompt (system_prompt) holds the functions in one of four renderings (LIST_STYLES),
the call notation (one of render_format_data's python, json or yaml serialisers), for the refusal kinds the
three prescribed sentences with their rules, and sometimes a rule to copy values verbatim. Each case has 36
sentence families per language (FAMILIES): 30 for training and 3 each for development sets A and B, which
no training sample uses and with which no training family shares a fixed part, a 3-word or a 4-character
Chinese window, nor a skeleton (every slot read as one token: a 3-token window holding a slot and two words,
or a token similarity of 0.5 or more; the review of revision 1 found near copies such as a B sentence with
one word added); a zh family's sentence is English or Chinese. A sentence shows its placeholders in a style
(STYLES: the six of BASE_STYLES, and for training only the fifteen of EXTRA_STYLES); the answer replaces the
placeholder words only and keeps the brackets. A training prompt may write a family in any style; a
development set writes each of its families in one style (DEV_STYLES), so that there a family gives each case
one sentence. A multi-turn request is either real messages or the history packed into one user message
(pack_history).
Answers are the bare sentence or the bare call; the explanation lives in a training sample's reasoning, the
check-first skeleton of reasoning_text (English, every value quoted verbatim); a development item has none.

Revision 1 (docs/m8-预登记.md, 修订 1: FT5's step 441 met the data-revision trigger R1 on dev set A) changes
the training items alone; the development path (families, styles, draws) is the locked sets'. The training
families are 30 per case and language, covering every sentence shape (family_shape: the filler or the name
first; the sentence starting with a slot, ending with one or neither) in both of a zh prompt's sentence
languages; a training refusal answer may be written in a cell (Cell, cell_deck: a style and a shape) that the
build deals out, and otherwise in a base style, EXTRA_SHARE of the time an extra one. The reasoning checks
before it decides: the functions, the earlier turns, the function that serves the request, every parameter
that matters (given or not, valid or not against its definition, the description quoted where it gives
the definition), one decision (the order of the checks only when a wrong and a missing value occur
together), then the sentence as the prompt writes it, its placeholders mapped to their fillers, the answer
and a check (reasoning_problems checks the form, and that no required parameter goes "not given" outside the
kinds that cut one, LACKING_KINDS). The builders take the revision's training constructions: a wrong value
on an optional parameter or in a format only the description implies, in a request that gives every
required value (build_incorrect: optional, described, complete), the cut of a given parameter (build_missing
keys: a missing twin), a pairing of a cut request (build_irrelevant user: an irrelevant twin), a refusal in
context whose earlier function could not serve the latest request (with_context); canned_request marks the
records' own refusals the build caps, and domain_words the near-domain partners it tries first.

The ACEBench ban. No text built here may hold ACEBench's prescribed sentences, its placeholder ApiName (as a
substring of any case, inside an identifier too: holds_apiname), its history header or its call-list notation
(banned_markers), or share a 5-word English or an 8-character CJK window with a template of the clone's
model_inference/prompt_en.py and prompt_zh.py (TemplateWindows; the templates are read with ast, never run).
The build script also leaves out every record that uses an ACEBench function name (acebench_function_names),
holds ApiName, or shares a 13-gram with ACEBench.

Formats. A training record is research_agent.data.public's sample with the functions in the system text
("tools": [], as scripts/replay_base.py writes them) plus the kind, the family and the prompt's choices
(training_record); its reasoning row binds by the record's name and turn (data.reasoning). A development item
is the JSON object dev_item writes and check_dev_item checks (docs/m8-计划.md WP2: the DATA and DEVEVAL
packages share it; a missing or priority item's source also says how its value was cut, "frame" or "bare"),
and dev_set_problems checks a set of them as a whole.
"""

from __future__ import annotations

import ast
import calendar
import copy
import datetime as dt
import functools
import importlib.util
import itertools
import json
import keyword
import pprint
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..normalize import normalize_text

REPO = Path(__file__).resolve().parents[3]


def _script(name: str):
    """scripts/<name>.py as a module of its own name (scripts/ is no package), loaded once."""
    key = f"_special_mix_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, REPO / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


_rfd = _script("render_format_data")  # the call serialisers and parsers of the format-mix data
serialize_calls, parse_calls = _rfd.serialize_calls, _rfd.parse_calls

KINDS = ("incorrect", "missing", "irrelevant", "priority", "ask", "valid_verbatim", "valid_multiturn",
         "valid_nearmiss")
SPECIAL_KINDS = KINDS[:5]
VALID_KINDS = KINDS[5:]
DEV_KINDS = tuple(kind for kind in KINDS if kind != "ask")
CASES = ("incorrect", "missing", "irrelevant")
LANGS = ("zh", "en")
SPLITS = ("train", "a", "b")
LIST_STYLES = ("json_lines", "json_array", "python_repr", "markdown")
CALL_FORMATS = tuple(_rfd.FORMATS)
# the six placeholder styles of the development sets (DEV_STYLES), and the training-only ones (revision 1):
# two more word sets for each of the six bracket shapes and three more bracket shapes with words of their own
BASE_STYLES = ("angle", "brace", "square", "paren", "caps", "corner")
EXTRA_STYLES = ("angle_words", "angle_caps", "brace_words", "brace_caps", "square_words", "square_snake",
                "paren_snake", "paren_caps", "caps_alt", "caps_short", "corner_alt", "corner_words",
                "lenticular", "double_brace", "dollar")
STYLES = BASE_STYLES + EXTRA_STYLES
EXTRA_SHARE = 0.2  # of the training prompts, an EXTRA_STYLES placeholder style
ORDERS = ("incorrect_first", "missing_first")
CATEGORIES = ("obvious", "format", "vague")
SUBKINDS = ("single", "two", "object", "optional")
PLAN = {  # docs/m8-计划.md WP1, the training counts per kind and language
    "incorrect": {"zh": 300, "en": 200},
    "missing": {"zh": 220, "en": 180},
    "irrelevant": {"zh": 110, "en": 90},
    "priority": {"zh": 50, "en": 30},
    "ask": {"zh": 30, "en": 30},
    "valid_verbatim": {"zh": 250, "en": 100},
    "valid_multiturn": {"zh": 130, "en": 70},
    "valid_nearmiss": {"zh": 90, "en": 60},
}
DEV_PLAN = {  # docs/m8-计划.md WP2: B's 240 gated items plus 10 + 10 priority items; A about 150
    "a": {"incorrect": {"zh": 15, "en": 15}, "missing": {"zh": 15, "en": 15},
          "irrelevant": {"zh": 15, "en": 15}, "priority": {"zh": 6, "en": 6},
          "valid_verbatim": {"zh": 10, "en": 6}, "valid_multiturn": {"zh": 10, "en": 6},
          "valid_nearmiss": {"zh": 10, "en": 6}},
    "b": {"incorrect": {"zh": 25, "en": 25}, "missing": {"zh": 25, "en": 25},
          "irrelevant": {"zh": 25, "en": 25}, "priority": {"zh": 10, "en": 10},
          "valid_verbatim": {"zh": 18, "en": 12}, "valid_multiturn": {"zh": 18, "en": 12},
          "valid_nearmiss": {"zh": 18, "en": 12}},
}
INCORRECT_MIX = (("obvious", 0.5), ("format", 0.25), ("vague", 0.25))  # and TWO_VALUES of them two values
TWO_VALUES = 0.1
MISSING_MIX = (("single", 0.4), ("two", 0.25), ("object", 0.2), ("optional", 0.15))
# Revision 1 of the training items' mix (docs/m8-预登记.md, 修订 1); the development sets keep the above:
# of the training incorrect slots, two wrong values: 21 % scheduled, so that about the analysis's 20 % of the
# items have two after the few slots that find no such record fall back to one
TRAIN_TWO_VALUES = 0.21
# of the training incorrect slots, the (first) wrong value on an optional parameter: 22 % scheduled, so that
# at least 20 % of the items have one after the few slots that find no such record fall back
OPTIONAL_SHARE = 0.22
DESCRIBED_SHARE = 0.5  # of the training format items, a format that the description alone implies
PACKED_SHARE = 0.6  # of the multi-turn items, the history packed into one user message
REQUIRED_FROM_CALL = ("deepexi",)  # sources whose schemas declare no required parameters
MAX_DISTRACTORS = 3  # functions shown besides the called one(s), at most
MIN_REST = 10  # characters a cut request keeps at least
MAX_REASONING_TOKENS = 900  # approx_tokens of a training reasoning, at most (FT5 thinks within 1,280 tokens)
TEMPLATE_MODEL = "special_mix-skeleton-2026-11-03"  # the sidecar rows' model: the reasoning skeleton
WORD_WINDOW, CHAR_WINDOW = 5, 8

Call = tuple[str, dict]

# ---- sentence families ---------------------------------------------------------------------------------

# (case, lang) -> (training, set A, set B); slots: {value} {param} (incorrect), {params} {api} (missing)
_FAMILY_TEXTS: dict[tuple[str, str], tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {
    ("incorrect", "en"): (
        ("Passing {value} as {param} would break the call, so nothing is sent.",
         "Halted: {param} would receive the disallowed {value}.",
         "{value} falls outside what {param} permits.",
         "Nothing was called; {param} came with the improper entry {value}.",
         "An entry like {value} has no place in {param}.",
         "The request sets {param} to {value}, which the function does not accept.",
         "Please correct {param}: {value} does not match its definition.",
         "An error sits in {param}, caused by the entry {value}.",
         "{param} received {value}, which is not a valid value.",
         "The call stops here, as {value} violates {param}.",
         "{value} does not fit the definition of {param}.",
         "{value} goes against how {param} is defined.",
         "{value} was rejected as a value of {param}.",
         "{value}, given as {param}, is not accepted.",
         "The entry {value} cannot serve as {param} here.",
         "Putting {value} into {param} conflicts with the schema, so the call is skipped.",
         "Got {value} for {param}, but that value is not allowed.",
         "Validation rejects {value} under {param} in this request.",
         "That entry, {value}, is rejected by {param}.",
         "Refusing {value} as an entry for {param}.",
         "Spotted something unsupported, {value}, in {param}.",
         "{param} refuses an entry such as {value}.",
         "{param} got {value}, which its definition rules out.",
         "{param} is set to {value}, an invalid choice.",
         "{param} holds {value} now, and the schema forbids that.",
         "In {param} I found {value}, which no rule allows.",
         "Checking {param}: the given {value} fails.",
         "Setting {param} to {value} is forbidden, so no call follows.",
         "Not a valid entry for {param}: {value}.",
         "The definition of {param} rules out {value}."),
        ("The supplied {param} value {value} breaks the function's rules.",
         "Bad value {value} in {param}; please fix it.",
         "I cannot use {value} as {param}."),
        ("{param} does not allow the value {value}.",
         "Wrong value for {param}: {value} is outside what the function allows.",
         "Unable to call: the {param} value {value} is not permitted."),
    ),
    ("incorrect", "zh"): (
        ("Sending {value} to {param} cannot work; please revise the request.",
         "No call is made with the faulty {value} in {param}.",
         "Please fix {param}, which currently reads {value}.",
         "{param} has an unusable value: {value}.",
         "Revisit {param}; {value} falls short of its rules.",
         "Under {param} the entry {value} fails validation, so I stop.",
         "检查 {param} 时发现 {value} 不符合规则,调用已停止。",
         "{param} 这里填了 {value},不满足它的约束。",
         "无法调用:{param} 被赋值为 {value},有误。",
         "请更正 {param}:给出的 {value} 无效。",
         "{value} will not pass the checks of {param}.",
         "{value} gets turned away by {param}.",
         "{value} 填进 {param} 不合规定,无法调用。",
         "{value} 不符合 {param} 的定义。",
         "{value} 对 {param} 来说是无效值。",
         "An invalid {value} was sent for {param} just now.",
         "Having {value} in {param} makes this request unusable.",
         "请求里的 {value} 放在 {param} 中不合法,请修改。",
         "输入的 {value} 与 {param} 的要求不符。",
         "This call stops because of {value} in {param}.",
         "Not accepted: {value} as the value of {param}.",
         "值 {value} 不适合参数 {param}。",
         "检查发现 {value} 不能赋给 {param}。",
         "Something is off with {param}: it was given {value}.",
         "A rule check on {param} failed for {value}.",
         "参数 {param} 不接受这个值:{value}。",
         "发现非法取值,{param} 收到了 {value}。",
         "{param} was filled with {value}, which is not allowed.",
         "{param} 不接受 {value} 这个值。",
         "{param} 填入的 {value} 违反了参数定义。"),
        ("Unacceptable value {value} supplied for {param}.",
         "{value} cannot be used for {param}.",
         "{value} 不是 {param} 的有效取值。"),
        ("The input {value} is invalid for {param}.",
         "Incorrect input in {param}: {value}.",
         "{param} 的值 {value} 无法接受。"),
    ),
    ("missing", "en"): (
        ("{api} depends on {params}, and the request leaves that out.",
         "Sending this request on to {api} first requires a value for {params}.",
         "Cannot call {api} without {params}.",
         "Please provide {params} so that {api} can be called.",
         "Nothing goes to {api} until the user adds {params}.",
         "The call to {api} is blocked: {params} not given.",
         "{params} is a precondition for running {api}.",
         "Holding off on {api}, since the request omits {params}.",
         "{api} has to wait; the request never specifies {params}.",
         "Unable to run {api} because {params} is not specified.",
         "Please supply {params} before I can call {api}.",
         "Values for {params} are required by {api}.",
         "Without {params} there is no way to run {api}.",
         "Give {params} so the request can go to {api}.",
         "First tell me {params}; then I can use {api}.",
         "{params} still has to be given for {api}.",
         "{params} is not in the request, yet {api} needs it.",
         "{params} must come first; {api} depends on it.",
         "{params}: required by {api} and absent.",
         "Asking for {params}: {api} cannot proceed without them.",
         "The value of {params} is needed for {api} to work.",
         "Need {params} first, then {api} can be used.",
         "Without {params}, a call to {api} would fail.",
         "Calling {api} needs {params} from you first.",
         "For {api} to do its job, {params} must be stated too.",
         "This {api} request comes up short on {params} at the moment.",
         "{api} stays paused while {params} remains unknown.",
         "{api} cannot be filled in: {params} was never mentioned.",
         "{api}: please add {params} and ask again.",
         "The call to {api} is incomplete without {params}."),
        ("To use {api} I still need {params}.",
         "Incomplete request for {api}: {params} missing.",
         "{api} cannot start until {params} is given."),
        ("Missing for {api}: {params}.",
         "Before {api} can run, please give {params}.",
         "{api} requires {params}, which were not stated."),
    ),
    ("missing", "zh"): (
        ("The request leaves out {params}, required by {api}.",
         "The {api} call needs {params} first.",
         "Required by {api} but not given: {params}.",
         "Provide {params} to call {api}.",
         "No call to {api}: {params} not supplied.",
         "{params} must be known before anything is sent to {api}.",
         "要让 {api} 运行,还得补充 {params}。",
         "{api} 还需要 {params} 才能调用。",
         "请补充 {params},否则无法调用 {api}。",
         "缺少 {params},暂不能调用 {api}。",
         "Value needed: {params}, so {api} must wait.",
         "Supply {params} and then {api} will run.",
         "{params} is still open, so no {api} call yet.",
         "{params} has not been given to {api}.",
         "{api} will not run without {params}.",
         "{api} lacks the values {params}.",
         "To run {api} the request must add {params} first.",
         "In this request, {api} is told nothing about {params}.",
         "Missing {params}, so nothing goes to {api}.",
         "补齐 {params} 以后,{api} 才可以调用。",
         "因为少了 {params},这次对 {api} 的调用只能作罢。",
         "眼下 {params} 空缺,因此 {api} 不会被调用。",
         "{params} 还没有给出,{api} 暂不能调用。",
         "{params} 还不明确,{api} 只好暂停。",
         "{api} 需要的 {params} 没有出现在请求中。",
         "{api} 还等着 {params} 的值,先不调用。",
         "调用 {api} 之前,还需提供 {params} 的取值。",
         "要执行 {api},请先补充 {params} 再试。",
         "函数 {api} 还差这些信息:{params}。",
         "无法运行 {api},缺的是 {params}。"),
        ("Still missing for {api}: {params}.",
         "{api} is waiting for {params}.",
         "{api} 的必填参数 {params} 未提供。"),
        ("More input needed before calling {api}: {params}.",
         "Missing values {params} prevent calling {api}.",
         "尚未给出 {params},{api} 无法执行。"),
    ),
    ("irrelevant", "en"): (
        ("None of the available functions can handle this request.",
         "This request lies past what these tools can do.",
         "None of the functions in this list is a fit.",
         "I have no function for this request.",
         "What is provided here leaves this task uncovered.",
         "Out of scope: no matching function.",
         "This cannot be done with the functions given.",
         "No suitable tool is available for this.",
         "The task falls outside the functions provided.",
         "There is no function here that serves this request.",
         "Nothing in this toolset can do what you ask.",
         "Your request falls outside every tool provided.",
         "No handler for that exists among these tools.",
         "I cannot help with this using the given tools.",
         "The available functions are unrelated to this request.",
         "This asks for something no function here offers.",
         "None of the tools I have covers this.",
         "There is no matching tool for this task.",
         "This request cannot be served by these functions.",
         "The toolset has nothing for this kind of request.",
         "This has nothing to do with the functions shown.",
         "That is a task none of my functions perform.",
         "Out of reach for the tools I was given.",
         "These functions serve other purposes, not this one.",
         "Unrelated request: no function serves it.",
         "I see no function that could do this.",
         "The provided tools are for something else entirely.",
         "Not possible with the functions at hand.",
         "Nothing here is built for this request.",
         "Every listed tool misses the point of this request."),
        ("No available function matches what is asked.",
         "The functions on offer cannot do this.",
         "Request not supported by any listed function."),
        ("None of these tools applies to the request.",
         "This is beyond the listed functions.",
         "No function in the list can carry this out."),
    ),
    ("irrelevant", "zh"): (
        ("The functions in this prompt cannot help with this request.",
         "Nothing in the function list fits.",
         "This request has no matching function.",
         "Unsupported request: no function applies.",
         "The available tools do not handle this.",
         "No tool on the list serves this purpose.",
         "现有函数都无法处理此项请求。",
         "列表中没有适合该请求的函数。",
         "该请求超出了所给函数的范围。",
         "没有可用的函数能完成这项请求。",
         "The tools offered here do not match this request.",
         "There is no way to handle this with what was supplied.",
         "This lies outside the scope of the tools.",
         "None of my functions fit what you need.",
         "I have nothing that does this.",
         "Request unrelated to the functions available.",
         "The function set cannot answer this request.",
         "No match between this request and the tools.",
         "This needs a tool that is not available.",
         "提供的工具都与这个需求无关。",
         "手头的函数做不了这件事。",
         "没有任何工具能满足此需求。",
         "这项需求与给定的函数不相关。",
         "以上函数均无法完成该任务。",
         "现有工具不支持这类操作。",
         "这超出了这些工具的能力范围。",
         "我没有能处理这一请求的函数。",
         "该需求不属于任何函数的功能。",
         "可选的工具里没有对应的函数。",
         "无法用所给工具完成该请求。"),
        ("This is not something the given functions do.",
         "No tool here fits the request.",
         "所列函数均不适用于此请求。"),
        ("The request is not covered by the functions.",
         "Functions listed: none suitable.",
         "这个请求不在可用函数的处理范围内。"),
    ),
}


@dataclass(frozen=True)
class Family:
    """A prescribed sentence: its id (<case>-<lang>-t01..t30, -a1..a3, -b1..b3), case, prompt language, split
    and template with the slots {value}, {param}, {params} and {api}."""

    id: str
    case: str
    lang: str
    split: str
    template: str


def _families() -> dict[str, Family]:
    out: dict[str, Family] = {}
    for (case, lang), (train, a, b) in _FAMILY_TEXTS.items():
        for split, texts in (("train", train), ("a", a), ("b", b)):
            for number, text in enumerate(texts, 1):
                suffix = f"t{number:02d}" if split == "train" else f"{split}{number}"
                family = Family(f"{case}-{lang}-{suffix}", case, lang, split, text)
                out[family.id] = family
    return out


FAMILIES = _families()


def families(case: str, lang: str, split: str) -> list[Family]:
    """The families of a case, prompt language and split, in id order."""
    return [f for f in FAMILIES.values() if (f.case, f.lang, f.split) == (case, lang, split)]


# A sentence's shape (revision 1): which slot comes first, the filler (the wrong value, the missing names) or
# the name (the parameter, the function), and whether the sentence starts with a slot, ends with one (before
# its final mark) or neither. FT5 got 3 of dev set A's 27 items right whose sentence starts with a slot, and
# no zh training answer started with the value slot.
SLOT_ORDERS = ("filler_first", "name_first")
POSITIONS = ("leading", "inner", "trailing")
_FILLER_SLOTS = ("value", "params")
_TRAILING_SLOT = re.compile(r"\{(?:value|param|params|api)\}[.。!！?？]?$")


def family_shape(family: Family) -> tuple[str, str] | None:
    """(slot order, position) of a family's sentence; None for one without slots (the irrelevant case)."""
    slots = _SLOT.findall(family.template)
    if not slots:
        return None
    text = family.template.strip()
    position = "leading" if _SLOT.match(text) else "trailing" if _TRAILING_SLOT.search(text) else "inner"
    return ("filler_first" if slots[0] in _FILLER_SLOTS else "name_first"), position


@dataclass(frozen=True)
class Cell:
    """What a training refusal answer is written in (revision 1): a placeholder style and the shape (slot
    order, position) of the family of the case it answers; the build deals the cells out of a deck
    (cell_deck), so that every style and shape is trained on in each language."""

    style: str
    order: str
    position: str


def _cells(styles: tuple[str, ...]) -> list[Cell]:
    return [Cell(style, order, position) for style in styles for order in SLOT_ORDERS
            for position in POSITIONS]


BASE_CELLS = _cells(BASE_STYLES)
EXTRA_CELLS = _cells(EXTRA_STYLES)


def cell_deck(rng: random.Random, rounds: int) -> list[Cell]:
    """The cells of rounds rounds, each every one of BASE_CELLS once and EXTRA_SHARE of its cells from
    EXTRA_CELLS (taken in turn from a shuffled cycle of them), shuffled within its round: after any whole
    number of rounds every base cell has come up as often as every other."""
    per_round = round(len(BASE_CELLS) * EXTRA_SHARE / (1 - EXTRA_SHARE))
    extras = list(EXTRA_CELLS)
    rng.shuffle(extras)
    deck: list[Cell] = []
    for number in range(rounds):
        chunk = [*BASE_CELLS, *(extras[(number * per_round + i) % len(extras)] for i in range(per_round))]
        rng.shuffle(chunk)
        deck += chunk
    return deck


STYLE_BRACKETS = {"angle": ("<", ">"), "brace": ("{", "}"), "square": ("[", "]"), "paren": ("(", ")"),
                  "caps": ("", ""), "corner": ("「", "」"),
                  "angle_words": ("<", ">"), "angle_caps": ("<", ">"), "brace_words": ("{", "}"),
                  "brace_caps": ("{", "}"), "square_words": ("[", "]"), "square_snake": ("[", "]"),
                  "paren_snake": ("(", ")"), "paren_caps": ("(", ")"), "caps_alt": ("", ""),
                  "caps_short": ("", ""), "corner_alt": ("「", "」"), "corner_words": ("「", "」"),
                  "lenticular": ("【", "】"), "double_brace": ("{{", "}}"), "dollar": ("${", "}")}
# (split, lang) -> the style of the development families numbered 1, 2, 3: a development prompt takes the
# incorrect and the missing family of one number and writes them in its style, so within a set a family keeps
# one sentence per case (research_agent.eval.special_dev reads a set's sentences family by family); the two
# languages of a set share the six base styles out (BASE_STYLES; the extra ones are for training only)
DEV_STYLES = {("a", "zh"): ("paren", "caps", "corner"), ("a", "en"): ("angle", "brace", "square"),
              ("b", "zh"): ("angle", "brace", "square"), ("b", "en"): ("paren", "caps", "corner")}
def _words(value: str, param: str, params: str, api: str) -> dict[str, str]:
    return {"value": value, "param": param, "params": params, "api": api}


# style -> the words inside its brackets, the same in both prompt languages or per language
_STYLE_WORDS = {
    "angle": _words("value", "parameter", "parameters", "function"),
    "brace": _words("bad_value", "param_name", "param_names", "function_name"),
    "square": _words("VALUE", "PARAM", "PARAMS", "FUNC"),
    "paren": _words("given value", "parameter name", "parameter names", "function name"),
    "caps": _words("VALUE_GIVEN", "PARAM_NAME", "MISSING_PARAMS", "FUNCTION_NAME"),
    "corner": {"en": _words("value", "param", "params", "function"),
               "zh": _words("错误值", "参数名", "缺失参数", "函数名")},
    "angle_words": _words("offending value", "parameter key", "absent parameters", "tool name"),
    "angle_caps": _words("BAD_INPUT", "KEY", "KEYS", "TOOL"),
    "brace_words": _words("invalid input", "argument", "arguments", "tool"),
    "brace_caps": _words("BAD", "ARG", "ARGS", "FN"),
    "square_words": _words("wrong value", "field name", "missing fields", "function"),
    "square_snake": _words("bad_val", "arg_name", "arg_names", "func_name"),
    "paren_snake": _words("input_value", "param_key", "missing_keys", "tool_id"),
    "paren_caps": _words("BAD_VALUE", "PARAM", "PARAMS", "API"),
    "caps_alt": _words("WRONG_VALUE", "PARAMETER", "MISSING_KEYS", "TOOL_NAME"),
    "caps_short": _words("BADVAL", "PKEY", "PKEYS", "FUNC_ID"),
    "corner_alt": {"en": _words("wrong value", "parameter", "parameters", "function name"),
                   "zh": _words("无效值", "参数", "缺少的参数", "接口名")},
    "corner_words": {"en": _words("the value", "the key", "the keys", "the tool"),
                     "zh": _words("所给值", "参数键", "缺失项", "工具名")},
    "lenticular": {"en": _words("value given", "parameter", "missing parameters", "function"),
                   "zh": _words("错误的值", "参数名称", "缺失的参数", "函数名称")},
    "double_brace": _words("value", "param", "params", "function"),
    "dollar": _words("given_value", "param_id", "absent_params", "function_id"),
}
_SLOT = re.compile(r"\{(value|param|params|api)\}")


def placeholders(style: str, lang: str) -> dict[str, str]:
    """slot -> the placeholder as a prompt shows it in that style (the word inside its brackets)."""
    words = _STYLE_WORDS[style]
    words = words.get(lang, words)
    opening, closing = STYLE_BRACKETS[style]
    return {slot: f"{opening}{word}{closing}" for slot, word in words.items()}


def prompt_sentence(family: Family, style: str, lang: str) -> str:
    """The sentence as the system prompt prescribes it, its placeholders in the style."""
    marks = placeholders(style, lang)
    return _SLOT.sub(lambda m: marks[m.group(1)], family.template)


def fill_sentence(family: Family, style: str, lang: str, *, value: str | None = None,
                  param: str | None = None, params: str | None = None, api: str | None = None) -> str:
    """The answer: the prescribed sentence with each placeholder word replaced and its brackets kept."""
    opening, closing = STYLE_BRACKETS[style]
    given = {"value": value, "param": param, "params": params, "api": api}

    def put(match: re.Match) -> str:
        text = given[match.group(1)]
        if text is None:
            raise ValueError(f"{family.id} needs a {match.group(1)}")
        return f"{opening}{text}{closing}"

    return _SLOT.sub(put, family.template)


def fixed_parts(family: Family, style: str, lang: str) -> list[str]:
    """The text of the prescribed sentence between its placeholder words, in order, each stripped (the
    brackets belong to it); an answer holds every part, in this order."""
    opening, closing = STYLE_BRACKETS[style]
    pieces = _SLOT.split(family.template)
    literals = pieces[0::2]
    parts = []
    for i, literal in enumerate(literals):
        text = (closing if i > 0 else "") + literal + (opening if i < len(literals) - 1 else "")
        if text.strip():
            parts.append(text.strip())
    return parts


def first_case(order: str, present: Iterable[str]) -> str:
    """The case whose sentence answers a request with the cases present: the one the order checks first."""
    present = set(present)
    first, second = ("incorrect", "missing") if order == "incorrect_first" else ("missing", "incorrect")
    for case in (first, second, "irrelevant"):
        if case in present:
            return case
    raise ValueError(f"no case among {sorted(present)}")


# ---- function lists ---------------------------------------------------------------------------------------

_MD = {"en": {"params": "Parameters:", "none": "(no parameters)", "required": "required",
              "optional": "optional", "one_of": "one of", "items": "items"},
       "zh": {"params": "参数:", "none": "(无参数)", "required": "必填", "optional": "可选",
              "one_of": "取值之一", "items": "元素"}}
_MD_SKIP = ("type", "description", "properties", "required", "enum", "items")


def _md_keywords(prop: dict, lang: str) -> list[str]:
    words = []
    if isinstance(prop.get("enum"), list):
        options = ", ".join(json.dumps(v, ensure_ascii=False) for v in prop["enum"])
        words.append(f"{_MD[lang]['one_of']}: {options}")
    for key, value in prop.items():
        if key not in _MD_SKIP:
            words.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    items = prop.get("items")
    if isinstance(items, dict):
        inner = [str(items["type"])] if isinstance(items.get("type"), str) else []
        inner += _md_keywords(items, lang)
        if inner:
            words.append(f"{_MD[lang]['items']}: " + ", ".join(inner))
    return words


def _md_properties(properties: dict, required: list, lang: str, indent: str) -> list[str]:
    lines = []
    for name, prop in properties.items():
        prop = prop if isinstance(prop, dict) else {}
        kind = prop.get("type") if isinstance(prop.get("type"), str) else "any"
        flag = _MD[lang]["required"] if name in required else _MD[lang]["optional"]
        inside = "; ".join([f"{kind}, {flag}", *_md_keywords(prop, lang)])
        description = " ".join(str(prop.get("description") or "").split())
        lines.append(f"{indent}- `{name}` ({inside})" + (f": {description}" if description else ""))
        if isinstance(prop.get("properties"), dict) and prop["properties"]:
            sub_required = prop.get("required") if isinstance(prop.get("required"), list) else []
            lines += _md_properties(prop["properties"], sub_required, lang, indent + "  ")
    return lines


def _markdown(function: dict, lang: str) -> str:
    parameters = function.get("parameters") if isinstance(function.get("parameters"), dict) else {}
    properties = parameters.get("properties") if isinstance(parameters.get("properties"), dict) else {}
    required = parameters.get("required") if isinstance(parameters.get("required"), list) else []
    lines = [f"### {function['name']}", " ".join(str(function.get("description") or "").split()),
             _MD[lang]["params"]]
    lines += _md_properties(properties, required, lang, "") or [_MD[lang]["none"]]
    return "\n".join(line for line in lines if line)


def render_functions(tools: list[dict], style: str, lang: str) -> str:
    """The functions (OpenAI tool objects) as a system prompt lists them: one JSON object per line, one JSON
    array, a Python literal or Markdown sections (LIST_STYLES)."""
    functions = [t["function"] for t in tools]
    if style == "json_lines":
        return "\n".join(json.dumps(f, ensure_ascii=False) for f in functions)
    if style == "json_array":
        return json.dumps(functions, ensure_ascii=False, indent=2)
    if style == "python_repr":
        return pprint.pformat(functions, width=110, sort_dicts=False)
    if style == "markdown":
        return "\n\n".join(_markdown(f, lang) for f in functions)
    raise ValueError(f"unknown list style {style!r}")


# ---- the system prompt (own wording, checked against the ACEBench templates by the tests) ------------------

_PREAMBLES = {
    "en": {
        "json_lines": (
            "You can call the functions below. Each line holds one function as a JSON object.\n{tools}",
            "Function definitions follow, one JSON object per line:\n{tools}"),
        "json_array": ("These functions are available, given as one JSON array:\n{tools}",
                       "Here is the function list (a JSON array of definitions):\n{tools}"),
        "python_repr": ("The available functions, written as a Python list of dictionaries:\n{tools}",
                        "Function catalogue (Python literal):\n{tools}"),
        "markdown": (
            "Available functions:\n\n{tools}",
            "You have these tools; each section describes one function and its parameters.\n\n{tools}"),
    },
    "zh": {
        "json_lines": ("你可以调用下面的函数,每行是一个函数的 JSON 对象。\n{tools}",
                       "函数定义如下,每行一个 JSON 对象:\n{tools}"),
        "json_array": ("可用函数如下,以一个 JSON 数组给出:\n{tools}", "函数列表(JSON 数组):\n{tools}"),
        "python_repr": ("可用函数,写成 Python 的字典列表:\n{tools}", "函数清单(Python 字面量):\n{tools}"),
        "markdown": ("可用函数:\n\n{tools}", "你可以使用以下工具,每一节说明一个函数及其参数。\n\n{tools}"),
    },
}
_CALL_RULES = {
    "en": {
        "python": (
            ('When a call is needed, answer with the call alone, one per line, as a Python function call '
             'with keyword arguments, e.g. tool_name(arg_a="text", arg_b=3).'),
            ('To call a function, write only the call in Python syntax with keyword arguments, such as '
             'tool_name(arg_a="text"); put each call on its own line.')),
        "json": (
            ('When a call is needed, answer with a JSON array only, each element {"name": ..., '
             '"arguments": {...}}, e.g. [{"name": "tool_name", "arguments": {"arg_a": "text"}}].'),
            ('Calls are written as one JSON array of objects with the keys "name" and "arguments", for '
             'instance [{"name": "tool_name", "arguments": {"arg_a": "text"}}]; reply with that array '
             'alone.')),
        "yaml": (
            ('When a call is needed, answer only with a YAML list; each item has name and arguments, e.g.\n'
             '- name: tool_name\n  arguments:\n    arg_a: "text"'),
            ('Write calls as a YAML block list (one item per call, with a name and an arguments mapping), '
             'like:\n- name: tool_name\n  arguments:\n    arg_a: "text"\nReply with the list only.')),
    },
    "zh": {
        "python": (
            ('如需调用函数,只写出调用,每行一个,用 Python 关键字参数的形式,'
             '例如 tool_name(arg_a="text", arg_b=3)。'),
            ('调用函数时请只输出 Python 形式的调用语句(关键字参数),'
             '如 tool_name(arg_a="text"),多个调用分行写。')),
        "json": (
            ('如需调用函数,只输出一个 JSON 数组,每个元素形如 {"name": ..., "arguments": {...}},例如 '
             '[{"name": "tool_name", "arguments": {"arg_a": "text"}}]。'),
            ('调用写成 JSON 数组,元素含 "name" 与 "arguments" 两个键,如 '
             '[{"name": "tool_name", "arguments": {"arg_a": "text"}}];只回复这个数组。')),
        "yaml": (
            ('如需调用函数,只输出 YAML 列表,每项包含 name 和 arguments,例如:\n'
             '- name: tool_name\n  arguments:\n    arg_a: "text"'),
            ('调用用 YAML 块列表表示(每个调用一项,含 name 与 arguments 映射),如:\n'
             '- name: tool_name\n  arguments:\n    arg_a: "text"\n只回复该列表。')),
    },
}
_VERBATIM_RULES = {
    "en": (("Copy argument values exactly as the user wrote them; do not translate or reword them unless a "
            "parameter's definition asks for a specific format."),
           ("Keep the user's own wording for values (no translation, no rewording) unless a definition "
            "demands a particular format.")),
    "zh": ("参数值按用户的原话照抄,不要翻译或改写,除非参数定义要求特定格式。",
           "取值沿用用户的说法(不翻译、不改写),参数定义另有格式要求的除外。"),
}
# refusal blocks: (header, incorrect, missing, irrelevant, ask, closing); {n} numbers a line, {s} is the
# sentence, {ask} {fk} {mk} the ask function and its two parameters
_BLOCKS = {
    "en": (
        ("Do not call a function in the cases below; reply with the matching sentence instead.",
         ("{n}. A value the user gave does not fit its parameter's definition (wrong type, not an allowed "
          "option, wrong format or out of range): {s}"),
         "{n}. A required parameter has no value in the request (do not guess one): {s}",
         "{n}. None of the functions can serve the request: {s}",
         ("{n}. A required parameter has no value in the request: do not guess it; call {ask} with {fk} set "
          "to the function you would call and {mk} set to the list of missing parameter names."),
         "Reply with the sentence alone."),
        ("Some requests must not lead to a call.",
         "If a value supplied for a parameter breaks that parameter's definition, answer exactly: {s}",
         "If the request leaves a required parameter without a value, answer exactly: {s}",
         "If the request has nothing to do with the listed functions, answer exactly: {s}",
         ("If the request leaves a required parameter without a value, ask for it through {ask}: call it "
          "with {fk} (the function that needs the value) and {mk} (the missing names, as a list)."),
         "Nothing else goes into such a reply."),
        ("Answer rules for problem requests:",
         "- wrong value (its type, allowed values, format or range not respected) -> {s}",
         "- required value absent -> {s}",
         "- no suitable function -> {s}",
         "- required value absent -> call {ask} with {fk} = the function name and {mk} = the missing names",
         "Give the sentence without any explanation."),
    ),
    "zh": (
        ("以下情况不要调用函数,改为回复对应的句子:",
         "{n}. 用户给出的某个值不符合该参数的定义(类型不对、不在可选范围内、格式不对或超出范围):{s}",
         "{n}. 请求里没有给出某个必填参数的值(不要自行猜测):{s}",
         "{n}. 所列函数都处理不了该请求:{s}",
         ("{n}. 请求里没有给出某个必填参数的值:不要猜测,调用 {ask},"
          "{fk} 填要调用的函数名,{mk} 填缺失参数名组成的列表。"),
         "只回复这句话。"),
        ("有些请求不应调用函数。",
         "若用户为某个参数提供的值违反了它的定义,原样回复:{s}",
         "若请求没有给出某个必填参数的值,原样回复:{s}",
         "若请求与所列函数都无关,原样回复:{s}",
         ("若请求没有给出某个必填参数的值,通过 {ask} 询问:调用它,"
          "{fk} 为需要该值的函数,{mk} 为缺失参数名的列表。"),
         "这类回复里不写别的内容。"),
        ("问题请求的回复规则:",
         "- 值有误(类型、可选值、格式或范围不符)→ {s}",
         "- 缺少必填值 → {s}",
         "- 没有合适的函数 → {s}",
         "- 缺少必填值 → 调用 {ask},{fk} 为函数名,{mk} 为缺失参数名列表",
         "只给出这句话,不要附加解释。"),
    ),
}
_ORDER_SENTENCES = {
    "en": {"incorrect_first": (("When a request has both a wrong value and a missing required value, check "
                                "the wrong value first and give only its sentence."),
                               ("Wrong values take precedence over missing ones: if both occur, use only the "
                                "wrong-value sentence.")),
           "missing_first": (("When a request has both a wrong value and a missing required value, check the "
                              "missing value first and give only its sentence."),
                             ("Missing values take precedence over wrong ones: if both occur, use only the "
                              "missing-value sentence."))},
    "zh": {"incorrect_first": ("若同时存在错误的值和缺失的必填值,先检查错误的值,只回复它对应的句子。",
                               "错误值优先于缺失值:两者都出现时,只用错误值的句子。"),
           "missing_first": ("若同时存在错误的值和缺失的必填值,先检查缺失的值,只回复它对应的句子。",
                             "缺失值优先于错误值:两者都出现时,只用缺失值的句子。")},
}
_FILL_ROLES = {  # slot -> (what replaces it in the first wording, the noun of the second wording)
    "en": {"value": ("the wrong value as the user wrote it", "value"),
           "param": ("its parameter name", "parameter name"),
           "params": ("the missing parameter names", "missing parameter names"),
           "api": ("the function name", "function name")},
    "zh": {"value": ("用户写的错误值", "值"), "param": ("对应的参数名", "参数名"),
           "params": ("没给出值的参数名", "没给出值的参数名"), "api": ("函数名", "函数名")},
}


def _join_en(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _fill_sentence(slots: list[str], marks: dict[str, str], lang: str, variant: int) -> str:
    """How the placeholders are filled in (two wordings per language), for the slots the sentences use."""
    roles = _FILL_ROLES[lang]
    if lang == "en":
        if variant == 0:
            told = _join_en([f"{marks[s]} with {roles[s][0]}" for s in slots])
            return (f"In these sentences, replace {told}, separating several entries by commas; keep every "
                    "other character, brackets included.")
        return (f"Fill in {_join_en([marks[s] for s in slots])} with the actual "
                f"{_join_en([roles[s][1] for s in slots])} (several entries separated by commas); everything "
                "else, brackets included, stays as written.")
    if variant == 0:
        told = ",".join(f"{marks[s]} 换成{roles[s][0]}" for s in slots)
        return f"句中的 {told};多个时用逗号分隔,其余字符(包括括号)保持不变。"
    return (f"把 {'、'.join(marks[s] for s in slots)} 填成实际的{'、'.join(roles[s][1] for s in slots)},"
            "多个时用逗号分隔,其他部分(括号也算)原样保留。")


@dataclass(frozen=True)
class AskFunction:
    """A function the prompt names for asking the user (the ask kind): its name and its two parameters."""

    name: str
    function_key: str
    missing_key: str

    def tool(self, lang: str) -> dict:
        texts = _ASK_TEXTS[lang]
        return {"type": "function", "function": {
            "name": self.name, "description": texts[0],
            "parameters": {"type": "object", "properties": {
                self.function_key: {"type": "string", "description": texts[1]},
                self.missing_key: {"type": "array", "items": {"type": "string"}, "description": texts[2]}},
                "required": [self.function_key, self.missing_key]}}}


ASK_FUNCTIONS = (AskFunction("ask_for_missing_values", "function", "missing"),
                 AskFunction("request_user_details", "target", "fields"),
                 AskFunction("collect_missing_inputs", "tool_name", "needed"),
                 AskFunction("follow_up_with_user", "for_function", "missing_params"))
_ASK_TEXTS = {"en": ("Ask the user for values a call still needs.", "The function that needs the values.",
                     "Which parameters are still without a value."),
              "zh": ("向用户询问调用还缺少的值。", "需要这些值的函数名。", "缺失参数的名称。")}


@dataclass(frozen=True)
class Rules:
    """The refusal rules a prompt holds: a family per case it states (the ask kind states no missing case),
    the placeholder style, the order of the checks (None when the prompt states none) and the wording
    choices."""

    families: dict
    style: str
    order: str | None
    block: int
    order_wording: int
    fill_wording: int


@dataclass(frozen=True)
class Wording:
    """The wording choices of a system prompt other than the rules' (indices into the variants)."""

    preamble: int = 0
    call_rule: int = 0
    verbatim: int = 0


def refusal_block(rules: Rules, lang: str, ask: AskFunction | None = None) -> str:
    """The rules part of a system prompt: one line per case (the ask rule in place of the missing one), the
    order of the checks, how the placeholders are filled in, the closing."""
    header, *lines, closing = _BLOCKS[lang][rules.block]
    by_case = dict(zip(("incorrect", "missing", "irrelevant", "ask"), lines, strict=True))
    out, number = [header], 0
    for case in ("incorrect", "missing", "irrelevant"):
        if case == "missing" and ask is not None:
            number += 1
            out.append(by_case["ask"].format(n=number, ask=ask.name, fk=ask.function_key, mk=ask.missing_key))
            continue
        family = rules.families.get(case)
        if family is None:
            continue
        number += 1
        out.append(by_case[case].format(n=number, s=prompt_sentence(family, rules.style, lang)))
    if rules.order is not None:
        out.append(_ORDER_SENTENCES[lang][rules.order][rules.order_wording])
    slots = [slot for case in ("incorrect", "missing") if case in rules.families
             for slot in (("value", "param") if case == "incorrect" else ("params", "api"))]
    if slots:
        out.append(_fill_sentence(slots, placeholders(rules.style, lang), lang, rules.fill_wording))
    out.append(closing)
    return "\n".join(out)


DEFAULT_WORDING = Wording()


def system_prompt(tools: list[dict], lang: str, *, list_style: str, call_format: str,
                  rules: Rules | None = None, ask: AskFunction | None = None, verbatim_rule: bool = False,
                  wording: Wording = DEFAULT_WORDING) -> str:
    """The system message: the functions, the call notation, the verbatim rule when asked for, the rules."""
    preamble = _PREAMBLES[lang][list_style][wording.preamble]
    head, _, tail = preamble.partition("{tools}")
    parts = [head + render_functions(tools, list_style, lang) + tail,
             _CALL_RULES[lang][call_format][wording.call_rule]]
    if verbatim_rule:
        parts.append(_VERBATIM_RULES[lang][wording.verbatim])
    if rules is not None:
        parts.append(refusal_block(rules, lang, ask))
    return "\n\n".join(parts)


# ---- multi-turn requests -----------------------------------------------------------------------------------

_WRAPPERS = {  # (header, user line, assistant line, footer); {n} numbers a line, {t} is its text
    "en": (("Chat so far (oldest first):", "[user] {t}", "[assistant] {t}", None),
           ("Here is our exchange up to now.", "Me: {t}", "You: {t}", "Please handle my last message."),
           ("Earlier turns:", "{n}. user said: {t}", "{n}. assistant replied: {t}", None)),
    "zh": (("到目前为止的聊天(从早到晚):", "[用户] {t}", "[助手] {t}", None),
           ("下面是我们之前的交流。", "我:{t}", "你:{t}", "请处理我最后一条消息。"),
           ("此前各轮:", "{n}. 用户说:{t}", "{n}. 助手答:{t}", None)),
}
FOLLOW_UPS = {"en": ("Sorry, I meant {new} rather than {old}.",
                     "Actually, please use {new} instead of {old}.",
                     "Correction: change {old} to {new}, everything else stays.",
                     "Wait, not {old} - make that {new}."),
              "zh": ("不好意思,不是{old},应该是{new}。", "改一下:把{old}换成{new},其他不变。",
                     "更正一下,{old}写错了,是{new}。", "等等,我说的不是{old},是{new}。")}
SWITCH_PREFIXES = {"en": ("", "Another request: ", "Something else: ", "Next: "),
                   "zh": ("", "另一个请求:", "还有一件事:", "接下来:")}


def pack_history(turns: list[tuple[str, str]], lang: str, wrapper: int) -> str:
    """A conversation (role, text) packed into one user message in one of three wrappings per language."""
    header, user, assistant, footer = _WRAPPERS[lang][wrapper]
    lines = [header] + [(user if role == "user" else assistant).format(n=n, t=text)
                        for n, (role, text) in enumerate(turns, 1)]
    if footer:
        lines.append(footer)
    return "\n".join(lines)


# ---- ACEBench text ban -----------------------------------------------------------------------------------

_APINAME = re.compile(r"apiname", re.IGNORECASE)  # a substring: inside an identifier (ApiNameKeyword) too
BANNED = (
    ("ACEBench's wrong-value sentence", re.compile(r"there is incorrect value", re.IGNORECASE)),
    ("ACEBench's missing-parameter sentence", re.compile(r"missing necessary parameters", re.IGNORECASE)),
    ("ACEBench's irrelevant sentence", re.compile(r"limitations of the function", re.IGNORECASE)),
    ("ACEBench's placeholder ApiName", _APINAME),
    ("ACEBench's history header", re.compile(r"conversation history 1\s*\.\.\s*t", re.IGNORECASE)),
    ("ACEBench's Chinese history header", re.compile(r"对话历史\s*1\s*\.\.\s*t")),
    ("ACEBench's call list [ApiName(...)]", re.compile(r"\[\s*\w+\(")),
)
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")
_FIELD = re.compile(r"\{[A-Za-z_]+\}")


def banned_markers(text: str) -> list[str]:
    """The ACEBench sentences, names and notations text holds (BANNED), by name; [] when none."""
    return [name for name, pattern in BANNED if pattern.search(text)]


def holds_apiname(text: str) -> bool:
    """Whether text holds ACEBench's placeholder ApiName as a substring of any case, inside an identifier such
    as deepexi's ApiNameKeyword too (the build leaves out every record whose functions, request or gold call
    hold it)."""
    return bool(_APINAME.search(text))


def word_windows(text: str, n: int = WORD_WINDOW) -> set[str]:
    """The n-word windows of a text: normalize_text, lower case, CJK characters as separators, words the runs
    of ASCII letters and digits."""
    words = re.findall(r"[a-z0-9]+", _CJK.sub(" ", normalize_text(text).lower()))
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def char_windows(text: str, n: int = CHAR_WINDOW) -> set[str]:
    """The n-character windows of a text holding a CJK character: normalize_text, lower case, no
    whitespace."""
    chars = "".join(normalize_text(text).lower().split())
    return {chars[i:i + n] for i in range(len(chars) - n + 1) if _CJK.search(chars[i:i + n])}


@dataclass(frozen=True)
class TemplateWindows:
    """The 5-word and 8-character windows of ACEBench's prompt templates; a format field ({time}, {function},
    ...) splits a template, since the text put there in ACEBench's prompts is not the template's."""

    words: frozenset[str]
    chars: frozenset[str]

    @classmethod
    def from_texts(cls, texts: Iterable[str]) -> TemplateWindows:
        segments = [segment for text in texts for segment in _FIELD.split(text)]
        return cls(frozenset(w for s in segments for w in word_windows(s)),
                   frozenset(c for s in segments for c in char_windows(s)))

    @classmethod
    def from_files(cls, paths: Iterable[str | Path]) -> TemplateWindows:
        return cls.from_texts([text for path in paths for text in prompt_templates(path)])

    def violations(self, text: str) -> list[str]:
        """The windows text shares with the templates, sorted; [] when none."""
        return sorted(word_windows(text) & self.words) + sorted(char_windows(text) & self.chars)


def prompt_templates(path: str | Path) -> list[str]:
    """The string constants a prompt module assigns at its top level, in order, read with ast (never run)."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    texts = []
    for node in tree.body:
        value = node.value if isinstance(node, (ast.Assign, ast.AnnAssign)) else None
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            texts.append(value.value)
    return texts


def _acebench_rows(data_dir: str | Path, patterns: tuple[str, ...]) -> Iterator[dict]:
    for lang in ("data_en", "data_zh"):
        for pattern in patterns:
            for path in sorted((Path(data_dir) / lang).glob(pattern)):
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        yield json.loads(line)


ACEBENCH_PATTERNS = ("data_normal_*.json", "data_special_*.json", "data_agent_*.json")


def acebench_files(data_dir: str | Path) -> list[Path]:
    """ACEBench's data files whose functions are read (normal, special and agent, both languages), sorted."""
    return sorted(p for lang in ("data_en", "data_zh") for pattern in ACEBENCH_PATTERNS
                  for p in (Path(data_dir) / lang).glob(pattern))


def acebench_functions(data_dir: str | Path) -> list[dict]:
    """Every function object of ACEBench's normal, special and agent data, both languages, in file order."""
    return [fn for row in _acebench_rows(data_dir, ACEBENCH_PATTERNS) for fn in row.get("function") or []
            if isinstance(fn, dict)]


def acebench_function_names(data_dir: str | Path) -> set[str]:
    """The names of ACEBench's functions (normal, special and agent data, both languages)."""
    return {fn["name"] for fn in acebench_functions(data_dir) if isinstance(fn.get("name"), str)}


def descriptions(node: object) -> list[str]:
    """The description texts of a function object or schema, recursively (the function's and every
    parameter's)."""
    out: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "description" and isinstance(value, str) and value.strip():
                out.append(value)
            else:
                out += descriptions(value)
    elif isinstance(node, list):
        for value in node:
            out += descriptions(value)
    return out


# ---- near-duplicate requests ------------------------------------------------------------------------------

# Jaccard's threshold, as a fraction: two requests whose tokens are at least 4/5 alike are near-duplicates
JACCARD = (4, 5)
# English words that say nothing of what is asked: articles, prepositions, pronouns, auxiliaries, contraction
# tails, politeness and the phrasings of asking
_STOP_EN = frozenset((
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "each", "every", "all", "both",
    "in", "on", "at", "by", "for", "from", "with", "without", "about", "into", "onto", "over", "under", "of",
    "to", "as", "per", "via", "between", "among", "than",
    "and", "or", "but", "if", "so", "then", "also", "too", "just", "very", "really",
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "you", "your", "yours", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their", "there", "here",
    "is", "are", "was", "were", "be", "been", "being", "am", "do", "does", "did", "done", "have", "has",
    "had", "having", "will", "would", "shall", "should", "can", "could", "may", "might", "must", "let",
    "lets",
    "s", "m", "t", "d", "ll", "re", "ve",
    "please", "kindly", "thanks", "thank", "hi", "hello", "hey", "sorry", "ok", "okay", "help", "need",
    "needs", "want", "wants", "like", "know", "tell"))
# Chinese particles and politeness, taken out of a request's signature: the multi-character ones first
_PARTICLES_ZH = ("麻烦", "帮我", "帮忙", "请", "您", "的", "了", "吗", "呢", "吧", "啊", "呀", "哦", "嘛")


@functools.cache
def _marks_to_spaces() -> dict[int, str]:
    """str.translate's table that turns every punctuation mark and symbol (Unicode categories P and S) into a
    space, made once."""
    return {code: " " for code in range(sys.maxunicode + 1) if unicodedata.category(chr(code))[0] in "PS"}


def _flat_text(text: str) -> str:
    """NFKC, lower case, every punctuation mark and symbol a space, whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFKC", text).lower().translate(_marks_to_spaces()).split())


@dataclass(frozen=True)
class RequestKey:
    """What near_duplicate compares of a request: its normalised text (_flat_text), its content signature (a
    Chinese request's text without spaces and particles; an English request's words outside _STOP_EN, as a
    set) and the tokens Jaccard counts (the signature's character pairs; the content words)."""

    norm: str
    signature: str | frozenset
    tokens: frozenset


def request_key(text: str) -> RequestKey:
    """The key of a request (RequestKey); a text holding a CJK character is read as Chinese."""
    norm = _flat_text(text)
    if _CJK.search(norm):
        signature = "".join(norm.split())
        for particle in _PARTICLES_ZH:
            signature = signature.replace(particle, "")
        return RequestKey(norm, signature, frozenset(signature[i:i + 2] for i in range(len(signature) - 1)))
    words = frozenset(word for word in norm.split() if word not in _STOP_EN)
    return RequestKey(norm, words, words)


def _alike(a: frozenset, b: frozenset) -> bool:
    """Jaccard similarity of two non-empty sets at least JACCARD, counted exactly."""
    shared = len(a & b)
    return JACCARD[1] * shared >= JACCARD[0] * (len(a) + len(b) - shared)


def _near(x: RequestKey, y: RequestKey) -> str | None:
    if x.norm == y.norm:
        return "exact"
    if type(x.signature) is type(y.signature) and x.signature == y.signature:
        return "signature"
    if x.tokens and y.tokens and _alike(x.tokens, y.tokens):
        return "jaccard"
    return None


def near_duplicate(a: str, b: str) -> str | None:
    """Which test finds two requests near-duplicates, in this order: "exact" (the same normalised text),
    "signature" (the same content signature), "jaccard" (tokens at least JACCARD alike); None when none
    does."""
    return _near(request_key(a), request_key(b))


class RequestIndex:
    """Requests under ids, for near_duplicate lookups without comparing every pair: the normalised texts and
    the signatures in dictionaries, the tokens in buckets by prefix. Two token sets at least JACCARD alike
    share a token among the first len - ceil(JACCARD * len) + 1 tokens of each, in any fixed order of the
    tokens (the prefix filter); rank orders them, rarest first, so that the buckets a lookup reads stay
    short. matches finds exactly what near_duplicate finds."""

    def __init__(self, rank: Mapping[str, int] | None = None) -> None:
        self.rank = rank or {}
        self.norms: dict[str, set] = defaultdict(set)
        self.signatures: dict[tuple, set] = defaultdict(set)
        self.buckets: dict[str, list[int]] = defaultdict(list)
        self.entries: list[tuple[object, frozenset]] = []

    def _prefix(self, tokens: frozenset) -> list[str]:
        n = len(tokens)
        need = -(-JACCARD[0] * n // JACCARD[1])  # ceil(JACCARD * n)
        return sorted(tokens, key=lambda t: (self.rank.get(t, 0), t))[:n - need + 1]

    def add(self, key_id: object, texts: Iterable[str]) -> None:
        """Index the texts of one request under its id."""
        for text in texts:
            key = request_key(text)
            self.norms[key.norm].add(key_id)
            self.signatures[type(key.signature).__name__, key.signature].add(key_id)
            if key.tokens:
                self.entries.append((key_id, key.tokens))
                for token in self._prefix(key.tokens):
                    self.buckets[token].append(len(self.entries) - 1)

    def matches(self, texts: Iterable[str]) -> dict:
        """id -> the test (near_duplicate) of every indexed request that a text near-duplicates; the first
        test found per id, texts in order and the tests in near_duplicate's order."""
        found: dict = {}
        for text in texts:
            key = request_key(text)
            for key_id in sorted(self.norms.get(key.norm, ()), key=repr):
                found.setdefault(key_id, "exact")
            for key_id in sorted(self.signatures.get((type(key.signature).__name__, key.signature), ()),
                                 key=repr):
                found.setdefault(key_id, "signature")
            if not key.tokens:
                continue
            seen: set[int] = set()
            for token in self._prefix(key.tokens):
                for entry in self.buckets.get(token, ()):
                    if entry in seen:
                        continue
                    seen.add(entry)
                    key_id, tokens = self.entries[entry]
                    if key_id not in found and _alike(key.tokens, tokens):
                        found[key_id] = "jaccard"
        return found


def token_rank(texts: Iterable[str]) -> dict[str, int]:
    """How many of the texts hold each token (request_key's tokens): RequestIndex's rank."""
    counts: Counter = Counter()
    for text in texts:
        counts.update(request_key(text).tokens)
    return dict(counts)


# ---- templates (the fix round's review, F1) ------------------------------------------------------------

# Two requests share a template when their template tokens are at least TEMPLATE_JACCARD alike, or when the
# tokens of one, TEMPLATE_CONTAINED or more of them, are all among the other's: "order a pizza from the
# nearest pizzeria" and "... restaurant" (0.6, under D1's 4/5), a flight from New York to London and one to
# Paris (names out). The build holds at most DEV_TEMPLATE_LIMIT development items of a template and keeps a
# development item's template out of the training items of its kind (scripts/build_special_mix.py).
TEMPLATE_JACCARD = (1, 2)
TEMPLATE_CONTAINED = 3
_QUESTION_EN = frozenset(("what", "how", "where", "when", "which", "who", "whom", "whose", "why"))
_SENTENCE_START = re.compile(r"(?:^|[.!?:;\n])[\s\"'“‘(\[]*$")  # what stands before a sentence's first word
_ENTITY_CHUNK = re.compile(r"\d|@|://")  # a chunk naming one thing: a number, a date, an identifier, an URL
_CJK_RUN = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]+")


@functools.lru_cache(maxsize=1 << 18)
def template_tokens(text: str) -> frozenset[str]:
    """What the requests of one template share whatever they name: the text after NFKC without its quoted
    spans (_QUOTED), read as Chinese when it holds a CJK character: the character pairs within each run of CJK
    characters, the particles (_PARTICLES_ZH) splitting the runs as numbers, identifiers and other ASCII do;
    else the English words outside _STOP_EN and the question words, leaving out each whitespace-separated
    chunk holding a digit, an @ or :// (numbers, dates, identifiers, addresses) and each name (a word
    capitalised inside a sentence, or one in capitals)."""
    body = _QUOTED.sub(" ", unicodedata.normalize("NFKC", text))
    if _CJK.search(body):
        for particle in _PARTICLES_ZH:
            body = body.replace(particle, " ")
        return frozenset(run[i:i + 2] for run in _CJK_RUN.findall(body) for i in range(len(run) - 1))
    tokens = set()
    for chunk in re.finditer(r"\S+", body):
        if _ENTITY_CHUNK.search(chunk.group(0)):
            continue
        for found in re.finditer(r"[A-Za-z]+", chunk.group(0)):
            word = found.group(0)
            at = chunk.start() + found.start()
            if (word[0].isupper() and not _SENTENCE_START.search(body, 0, at)) or \
                    (len(word) > 1 and word.isupper()):
                continue
            if word.lower() not in _STOP_EN and word.lower() not in _QUESTION_EN:
                tokens.add(word.lower())
    return frozenset(tokens)


def same_template(a: frozenset, b: frozenset) -> str | None:
    """Which test finds two requests' template tokens (template_tokens) one template: "jaccard" (at least
    TEMPLATE_JACCARD alike), "contains" (the smaller set, TEMPLATE_CONTAINED tokens or more, all among the
    other's); None when neither does or a set is empty."""
    if not a or not b:
        return None
    shared = len(a & b)
    if TEMPLATE_JACCARD[1] * shared >= TEMPLATE_JACCARD[0] * (len(a) + len(b) - shared):
        return "jaccard"
    if shared == min(len(a), len(b)) >= TEMPLATE_CONTAINED:
        return "contains"
    return None


class TemplateIndex:
    """Requests under ids, each with a kind, for same_template lookups: their template tokens, and postings
    from each token to the requests holding it, over all kinds and per kind (two requests of one template
    share a token). matches finds exactly what same_template finds; what one text matches is kept until the
    index grows (the training draw asks about the same requests for many kinds and slots)."""

    def __init__(self) -> None:
        self.entries: list[tuple[object, str, frozenset]] = []
        self.postings: dict[tuple[str | None, str], list[int]] = defaultdict(list)
        self._known: dict[tuple[str, str | None], list[tuple[object, str]]] = {}

    def add(self, key_id: object, kind: str, texts: Iterable[str]) -> None:
        """Index the texts of one request under its id and kind."""
        for text in texts:
            tokens = template_tokens(text)
            if tokens:
                self.entries.append((key_id, kind, tokens))
                for token in tokens:
                    self.postings[None, token].append(len(self.entries) - 1)
                    self.postings[kind, token].append(len(self.entries) - 1)
                self._known.clear()

    def _matched(self, text: str, kind: str | None) -> list[tuple[object, str]]:
        """(id, test) of the indexed requests of the kind (any when None) one text shares a template with, in
        the order of the text's sorted tokens and of the postings."""
        if (text, kind) not in self._known:
            tokens = template_tokens(text)
            seen: set[int] = set()
            out = []
            for token in sorted(tokens):
                for entry in self.postings.get((kind, token), ()):
                    if entry not in seen:
                        seen.add(entry)
                        test = same_template(tokens, self.entries[entry][2])
                        if test:
                            out.append((self.entries[entry][0], test))
            self._known[text, kind] = out
        return self._known[text, kind]

    def matches(self, texts: Iterable[str], kind: str | None = None) -> dict:
        """id -> the test (same_template) of every indexed request of the kind (any kind when None) that
        shares a template with a text; the first test found per id, texts in order, tokens sorted."""
        found: dict = {}
        for text in texts:
            for key_id, test in self._matched(text, kind):
                found.setdefault(key_id, test)
        return found


# ---- record helpers (copied from scripts/build_ace_probe.py and scripts/replay_base.py; the tests check
# that the copies agree) ---------------------------------------------------------------------------------

_ASCII_WORD = re.compile(r"[A-Za-z0-9_]")
_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’", "「": "」", "《": "》", "`": "`"}
_ASKS = re.compile(
    r"\b(?:please|could you|can you|would you)\s+(?:provide|tell|specify|share|let me know)", re.IGNORECASE)
_ASKS_ZH = re.compile(r"请提供|请告诉|请问您|您能提供|能否提供|可以提供|需要知道|请补充")
IRRELEVANT_GOLD = re.compile(
    r"\bnone of the (?:\w+ ){0,2}(?:functions|tools|apis)\b"
    r"|\b(?:given|provided|available|listed|above) (?:functions|tools|apis) (?:do|does|can|could|are) ?not\b"
    r"|\bno (?:suitable|relevant|appropriate|available|applicable) (?:function|tool|api)s?\b",
    re.IGNORECASE)
_PARAMETER_TALK = re.compile(
    r"\b(?:lacks?|missing|insufficient|(?:required|necessary) (?:argument|parameter)s?)\b", re.IGNORECASE)
_REFUSES = re.compile(
    r"\b(?:I'm sorry|I am sorry|I apologi[sz]e|unfortunately|I'm unable|I am unable|I can't|I cannot|"
    r"I'm not able|I am not able|not able to|don't have the (?:capability|ability|functionality)|"
    r"capabilit(?:y|ies)|limited to|outside|beyond)\b"
    r"|无法|不能|抱歉|很遗憾|只能|仅限|超出|没有.{0,6}(?:功能|能力)", re.IGNORECASE)


def first_step(sample: dict) -> tuple[str, list[Call]] | None:
    """(user text, calls) of a record that opens with a user request answered by tool calls alone."""
    messages = sample["messages"]
    if len(messages) < 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return None
    answer = messages[1]
    if not answer.get("tool_calls") or answer["content"].strip():
        return None
    calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in answer["tool_calls"]]
    return messages[0]["content"], calls


def _norm(text: str) -> str:
    return normalize_text(text).casefold()


def _number_text(value: float) -> str:
    return str(int(value)) if isinstance(value, int) or float(value).is_integer() else repr(value)


def value_text(value: object) -> str | None:
    """How a call value is looked for in a request: a string as it is, a number as a whole number when
    integral; None for any other value."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return _number_text(value)
    return value if isinstance(value, str) else None


@functools.lru_cache(maxsize=1 << 16)
def _occurrence_pattern(value: str, ignore_case: bool = True) -> re.Pattern:
    pattern = re.escape(value)
    if _ASCII_WORD.match(value[:1]):
        pattern = r"(?<![A-Za-z0-9_])" + pattern
    if _ASCII_WORD.match(value[-1:]):
        pattern += r"(?![A-Za-z0-9_])"
    return re.compile(pattern, re.IGNORECASE if ignore_case else 0)


def occurrences(value: str, text: str) -> list[re.Match]:
    """The case-insensitive occurrences of value in text; an ASCII word character at an end of value must not
    continue into the text."""
    return list(_occurrence_pattern(value).finditer(text))


def _leaves(value: object) -> list[object]:
    if isinstance(value, dict):
        return [leaf for v in value.values() for leaf in _leaves(v)]
    if isinstance(value, list):
        return [leaf for v in value for leaf in _leaves(v)]
    return [value]


def _leaf_in(leaf: object, text: str) -> bool:
    if leaf is None or isinstance(leaf, bool):
        return True
    if isinstance(leaf, (int, float)):
        return bool(occurrences(_number_text(leaf), text))
    if isinstance(leaf, str):
        return _norm(leaf) in _norm(text)
    return False


def grounded(user: str, calls: list[Call]) -> bool:
    """Every string and number of the calls occurs in the user text."""
    return all(_leaf_in(leaf, user) for _, arguments in calls for leaf in _leaves(arguments))


def _cut(text: str, match: re.Match) -> str:
    start, end = match.span()
    if start > 0 and end < len(text) and _QUOTE_PAIRS.get(text[start - 1]) == text[end]:
        start, end = start - 1, end + 1
    rest = text[:start] + text[end:]
    rest = re.sub(r"[ \t]{2,}", " ", rest)
    rest = re.sub(r"[ \t]+([,.?!;:，。？！；：])", r"\1", rest)
    return rest.strip()


def is_irrelevant_gold(text: str) -> bool:
    """A gold answer saying that no listed function applies (and not that parameters are missing)."""
    return bool(IRRELEVANT_GOLD.search(text)) and not _PARAMETER_TALK.search(text)


def replace_once(text: str, old: str, new: str) -> str | None:
    """text with the one occurrence of old replaced by new; None unless old occurs exactly once."""
    found = occurrences(old, text)
    if len(found) != 1:
        return None
    start, end = found[0].span()
    return text[:start] + new + text[end:]


def cut_values(text: str, values: list[str]) -> str | None:
    """text with each value cut in turn (with quotes around it, and the spaces it leaves); None unless every
    value occurs exactly once when its turn comes."""
    for value in values:
        found = occurrences(value, text)
        if len(found) != 1:
            return None
        text = _cut(text, found[0])
    return text


def approx_tokens(text: str) -> int:
    """A rough token count for the length cap: a CJK character is one, three other characters are one."""
    cjk = len(_CJK.findall(text))
    return cjk + (len(text) - cjk + 2) // 3


# ---- constraints -----------------------------------------------------------------------------------------

_TYPES = {"integer": "integer", "int": "integer", "long": "integer", "number": "number", "float": "number",
          "double": "number", "string": "string", "str": "string"}
_SCHEMA_FORMATS = {"date": "YYYY-MM-DD", "date-time": "ISO-DATETIME", "datetime": "ISO-DATETIME",
                   "time": "TIME", "email": "EMAIL", "uri": "URL", "url": "URL"}
_SPEC_IN_TEXT = re.compile(r"(?i)y{4}[-/.]?m{2}[-/.]?d{2}(?:(?:'?t'?|\s+)h{2}:m{2}(?::s{2})?)?(?:'?z'?)?"
                           r"|(?<![A-Za-z])h{2}:m{2}(?::s{2})?(?![A-Za-z])")
_DESCRIBED = (  # description phrasings of named formats, after the date and time specs
    ("ISO-8601", re.compile(r"ISO\s?8601", re.IGNORECASE)),
    ("EMAIL", re.compile(r"\be-?mail\b|邮箱|电子邮件", re.IGNORECASE)),
    ("URL", re.compile(r"\bURLs?\b|网址|链接地址", re.IGNORECASE)),
    ("CODE3", re.compile(r"\b(?:3|three)[- ]letter\b|ISO\s?4217|\bIATA\b|alpha-3", re.IGNORECASE)),
    ("ZIP", re.compile(r"\bzip ?code\b|\bpostal code\b", re.IGNORECASE)),
)
_FORMAT_REGEX = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+"),
    "URL": re.compile(r"https?://[^\s/$.?#][^\s]*"),
    "CODE3": re.compile(r"[A-Z]{3}"),
    "ZIP": re.compile(r"\d{5}(?:-\d{4})?"),
}
_ISO = re.compile(r"(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})"
                  r"(?:T(?P<h>\d{2}):(?P<mi>\d{2})(?::(?P<s>\d{2})(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?")
_TIME = re.compile(r"(?P<h>\d{2}):(?P<mi>\d{2})(?::(?P<s>\d{2}))?")
_HEDGE = re.compile(r"typically|usually|recommend|suggest|建议|一般|通常", re.IGNORECASE)
_NUM = r"(-?\d+(?:\.\d+)?)(?!,\d{3}(?!\d))"  # a number, never the head of 1,000
_RANGES = (  # (pattern, which bounds its groups give)
    (re.compile(r"范围(?:为|是)?\s*[:：]?\s*\[?" + _NUM + r"\s*(?:~|～|-|—|至|到)\s*" + _NUM),
     ("min", "max")),
    (re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"), ("min", "max")),
    (re.compile(r"between\s+" + _NUM + r"\s+and\s+" + _NUM, re.IGNORECASE), ("min", "max")),
    (re.compile(r"\(\s*" + _NUM + r"\s*-\s*" + _NUM + r"\s*\)"), ("min", "max")),
    (re.compile(r"最大值?(?:为|是)?\s*[:：]?\s*" + _NUM), ("max",)),
    (re.compile(r"(?:最多|不(?:能)?超过|不大于)\s*(?:为|是)?\s*" + _NUM), ("max",)),
    (re.compile(r"\b(?:maximum|max)(?:\s+value)?(?:\s+is|\s+of)?\s*[:=]?\s*" + _NUM, re.IGNORECASE),
     ("max",)),
    (re.compile(r"\b(?:at most|up to|no more than|not more than|cannot exceed|must not exceed)\s+" + _NUM,
                re.IGNORECASE), ("max",)),
    (re.compile(r"最小值?(?:为|是)?\s*[:：]?\s*" + _NUM), ("min",)),
    (re.compile(r"(?:不(?:能)?小于|至少)\s*(?:为|是)?\s*" + _NUM), ("min",)),
    (re.compile(r"\b(?:minimum|min)(?:\s+value)?(?:\s+is|\s+of)?\s*[:=]?\s*" + _NUM, re.IGNORECASE),
     ("min",)),
    (re.compile(r"\b(?:at least|no less than|not less than)\s+" + _NUM, re.IGNORECASE), ("min",)),
)
_CLAUSE = re.compile(r"[。;；!?\n，]|[.,](?!\d)")  # clause ends; 0.90 and 1,000 stay whole
_ENUM_ZH = re.compile(r"(?:取值|可选值|枚举值)(?:为|如下|包括|有)?\s*[:：]")
_ENUM_EN = re.compile(r"\b(?:one of|valid values(?: are)?|possible values(?: are)?|allowed values(?: are)?|"
                      r"accepted values(?: are)?|options are|must be either|can be either|either)\b\s*:?\s*",
                      re.IGNORECASE)
_OPEN_LIST = re.compile(r"\betc\b|e\.g\.|such as|for example|\blike\b|等", re.IGNORECASE)
_BULLET = re.compile(r"^\s*[-*•]\s*\**`?([A-Za-z0-9_.\-]+)`?\**\s*[:：]")
_TOKEN = re.compile(r"[A-Za-z0-9_.\-]+|[一-鿿]{1,10}")


@dataclass(frozen=True)
class Constraint:
    """What a parameter's definition allows, as far as it can be read (infer_constraint): its type, its
    options, a pattern, a named format (a date or time spec such as YYYY-MM-DD HH:MM, or ISO-8601,
    ISO-DATETIME, TIME, EMAIL, URL, CODE3, ZIP) and bounds; sources names where each part came from
    (schema:<keyword> or description:<rule>)."""

    type: str | None = None
    enum: tuple = ()
    pattern: str | None = None
    fmt: str | None = None
    minimum: float | None = None
    maximum: float | None = None
    min_exclusive: bool = False
    max_exclusive: bool = False
    sources: tuple[str, ...] = ()

    @property
    def mentions_format(self) -> bool:
        return self.fmt is not None or self.pattern is not None

    @property
    def usable(self) -> bool:
        return bool(self.type in ("integer", "number") or self.enum or self.pattern or self.fmt
                    or self.minimum is not None or self.maximum is not None)


def _spec(found: str) -> str:
    """A date or time spec as found in a description, normalised: no quotes, upper case, single spaces."""
    return " ".join(found.replace("'", "").upper().split())


def _description_format(text: str) -> tuple[str, str] | None:
    found = _SPEC_IN_TEXT.search(text)
    if found:
        return _spec(found.group(0)), "description:spec"
    for name, pattern in _DESCRIBED:
        if pattern.search(text):
            return name, f"description:{name.lower()}"
    return None


def _clean_token(token: str) -> str:
    return token.strip().strip("*`'\"“”‘’「」").strip()


def _description_enum(text: str) -> tuple[str, ...] | None:
    """A closed list of options the description gives (取值:A、B / bullets / "one of 'x', 'y'"); None when it
    gives none or an open one (etc., 等, e.g.)."""
    found = _ENUM_ZH.search(text)
    if found:
        rest = text[found.end():]
        bullets = []
        for line in rest.lstrip("\n").split("\n"):
            if not line.strip():
                continue
            match = _BULLET.match(line)
            if match is None:
                break
            bullets.append(match.group(1))
        if len(bullets) >= 2:
            return tuple(dict.fromkeys(bullets))
        segment = re.split(r"[。\n;；]", rest, maxsplit=1)[0]
        if _OPEN_LIST.search(segment):
            return None
        tokens = [_clean_token(t) for t in re.split(r"[、,，/]|或", segment)]
        if len(tokens) >= 2 and all(t and _TOKEN.fullmatch(t) for t in tokens):
            return tuple(dict.fromkeys(tokens))
        return None
    found = _ENUM_EN.search(text)
    if found:
        segment = re.split(r"[.;\n](?:\s|$)", text[found.end():], maxsplit=1)[0]
        if _OPEN_LIST.search(segment):
            return None
        quoted = re.findall(r"""['"`]([^'"`]{1,40})['"`]""", segment)
        rest = re.sub(r"""['"`][^'"`]{1,40}['"`]""", " ", segment)
        if len(quoted) >= 2 and re.fullmatch(r"[\s,/]*(?:(?:or|and)[\s,/]*)*", rest):
            return tuple(dict.fromkeys(quoted))
    return None


def _number(text: str) -> int | float:
    value = float(text)
    return int(value) if value.is_integer() else value


def _description_bounds(text: str) -> tuple[float | None, float | None]:
    """(minimum, maximum) a description states, each None when it states none; a hedged clause (typically
    ..., 建议 ...) and a minimum above the maximum give none."""
    lowest = highest = None
    for sentence in _CLAUSE.split(text):
        if _HEDGE.search(sentence):
            continue
        for pattern, gives in _RANGES:
            match = pattern.search(sentence)
            if match is None:
                continue
            values = dict(zip(gives, (_number(g) for g in match.groups()), strict=True))
            lowest = values.get("min", lowest) if lowest is None else lowest
            highest = values.get("max", highest) if highest is None else highest
    if lowest is not None and highest is not None and lowest > highest:
        return None, None
    return lowest, highest


def _schema_type(prop: dict) -> str | None:
    kind = prop.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), None)
    return _TYPES.get(kind) if isinstance(kind, str) else None


def infer_constraint(prop: dict) -> Constraint | None:
    """The constraint of a scalar parameter (module docstring): schema keywords first, then the description's
    phrasings; None for a boolean, an object or an array, and when nothing usable is read."""
    if not isinstance(prop, dict):
        return None
    raw = prop.get("type")
    if isinstance(raw, str) and raw.lower() in ("boolean", "bool", "object", "array", "dict", "list"):
        return None
    kind = _schema_type(prop)
    sources = ["schema:type"] if kind in ("integer", "number") else []
    description = prop.get("description") if isinstance(prop.get("description"), str) else ""
    enum: tuple = ()
    if isinstance(prop.get("enum"), list) and prop["enum"] and all(
            isinstance(v, (str, int, float)) and not isinstance(v, bool) for v in prop["enum"]):
        enum = tuple(prop["enum"])
        sources.append("schema:enum")
    elif description and kind not in ("integer", "number"):
        listed = _description_enum(description)
        if listed:
            enum = listed
            sources.append("description:enum")
    pattern = None
    if isinstance(prop.get("pattern"), str):
        try:
            re.compile(prop["pattern"])
            pattern = prop["pattern"]
            sources.append("schema:pattern")
        except re.error:
            pattern = None
    fmt = None
    if isinstance(prop.get("format"), str) and prop["format"].lower() in _SCHEMA_FORMATS:
        fmt = _SCHEMA_FORMATS[prop["format"].lower()]
        sources.append("schema:format")
    elif description and kind not in ("integer", "number") and not enum:
        found = _description_format(description)
        if found:
            fmt, source = found
            sources.append(source)
    minimum = maximum = None
    min_exclusive = max_exclusive = False
    for key, exclusive_key in (("minimum", "exclusiveMinimum"), ("maximum", "exclusiveMaximum")):
        value = prop.get(key)
        exclusive = prop.get(exclusive_key)
        if isinstance(exclusive, (int, float)) and not isinstance(exclusive, bool):
            value, flag = exclusive, True
        else:
            flag = exclusive is True
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if key == "minimum":
                minimum, min_exclusive = _number(str(value)), flag
            else:
                maximum, max_exclusive = _number(str(value)), flag
            sources.append(f"schema:{key}")
    if minimum is None and maximum is None and description and kind in ("integer", "number"):
        minimum, maximum = _description_bounds(description)
        if minimum is not None or maximum is not None:
            sources.append("description:range")
    constraint = Constraint(kind if kind in ("integer", "number") else None, enum, pattern, fmt, minimum,
                            maximum, min_exclusive, max_exclusive, tuple(sources))
    return constraint if constraint.usable else None


def item_constraint(prop: dict) -> Constraint | None:
    """The constraint of an array parameter's items (items keywords, else the array's own description
    list)."""
    if not isinstance(prop, dict) or not isinstance(prop.get("items"), dict):
        return None
    found = infer_constraint(prop["items"])
    if found is None and isinstance(prop.get("description"), str):
        listed = _description_enum(prop["description"])
        if listed:
            return Constraint(enum=listed, sources=("description:enum",))
    return found


def _spec_regex(spec: str) -> re.Pattern | None:
    if "YYYY" not in spec and "HH" not in spec:
        return None
    pattern, hour = "", False
    for token in re.findall(r"YYYY|MM|DD|HH|SS|.", spec):
        if token == "YYYY":
            pattern += r"(?P<y>\d{4})"
        elif token == "MM":
            pattern += r"(?P<mi>\d{2})" if hour else r"(?P<mo>\d{2})"
        elif token == "DD":
            pattern += r"(?P<d>\d{2})"
        elif token == "HH":
            pattern, hour = pattern + r"(?P<h>\d{2})", True
        elif token == "SS":
            pattern += r"(?P<s>\d{2})"
        else:
            pattern += re.escape(token)
    return re.compile(pattern)


def _valid(parts: dict) -> dict | None:
    """The date and time parts as integers when they make a real date and time; None otherwise."""
    out = {k: int(v) for k, v in parts.items() if v is not None}
    if "y" in out and not (1 <= out["y"] and 1 <= out.get("mo", 1) <= 12 and 1 <= out.get("d", 1)
                           <= calendar.monthrange(out["y"], out.get("mo", 1))[1]):
        return None
    if not (0 <= out.get("h", 0) <= 23 and 0 <= out.get("mi", 0) <= 59 and 0 <= out.get("s", 0) <= 59):
        return None
    return out


def date_parts(fmt: str, text: str) -> dict | None:
    """The date and time parts (y, mo, d, h, mi, s) of text in a date or time format; None when text is not in
    it or not a real date or time."""
    if not isinstance(text, str):
        return None
    if fmt in ("ISO-DATETIME", "ISO-8601"):
        match = _ISO.fullmatch(text)
        if match is None or (fmt == "ISO-DATETIME" and match.group("h") is None):
            return None
        return _valid(match.groupdict())
    regex = _TIME if fmt == "TIME" else _spec_regex(fmt)
    match = regex.fullmatch(text) if regex else None
    return _valid(match.groupdict()) if match else None


def is_date_format(fmt: str | None) -> bool:
    return bool(fmt) and (fmt in ("ISO-DATETIME", "ISO-8601", "TIME") or "YYYY" in fmt or "HH" in fmt)


def format_ok(fmt: str, text: object) -> bool:
    """Whether text is a value in the named format."""
    if not isinstance(text, str):
        return False
    if is_date_format(fmt):
        return date_parts(fmt, text) is not None
    regex = _FORMAT_REGEX.get(fmt)
    return bool(regex and regex.fullmatch(text))


def passes(c: Constraint, value: object) -> bool:
    """Whether a value satisfies the constraint (a boolean or null never does)."""
    if isinstance(value, bool) or value is None:
        return False
    if c.type == "integer" and not isinstance(value, int):
        return False
    if c.type == "number" and not isinstance(value, (int, float)):
        return False
    if c.enum and value not in c.enum:
        return False
    if c.pattern is not None and (not isinstance(value, str) or not re.search(c.pattern, value)):
        return False
    if c.fmt is not None and not format_ok(c.fmt, value):
        return False
    if c.minimum is not None or c.maximum is not None:
        if not isinstance(value, (int, float)):
            return False
        if c.minimum is not None and (value < c.minimum or (c.min_exclusive and value == c.minimum)):
            return False
        if c.maximum is not None and (value > c.maximum or (c.max_exclusive and value == c.maximum)):
            return False
    return True


def read_value(text: str, c: Constraint | None) -> object:
    """A request's text as the value it gives: a plain number for a numeric parameter, else the text."""
    if c is not None and (c.type in ("integer", "number") or c.minimum is not None or c.maximum is not None):
        stripped = text.strip()
        if re.fullmatch(r"-?\d+", stripped):
            return int(stripped)
        if re.fullmatch(r"-?\d+\.\d+", stripped):
            return float(stripped)
    return text


# ---- mutation ----------------------------------------------------------------------------------------------

VAGUE = {
    "number": {"en": ("a few", "several", "a dozen or so", "a couple", "many", "some"),
               "zh": ("十几", "几个", "好几个", "一些", "几十", "若干")},
    "date": {"en": ("next week", "sometime soon", "early next month", "in a few days"),
             "zh": ("下周", "过几天", "月底前后", "下个月初")},
    "time": {"en": ("in the afternoon", "later today", "around noon", "in the evening"),
             "zh": ("下午", "晚些时候", "中午前后", "傍晚")},
    "datetime": {"en": ("tomorrow morning", "this weekend", "some time next week"),
                 "zh": ("明天上午", "这个周末", "下周某个时候")},
    "enum": {"en": ("whichever", "any of them", "the usual one", "something in between"),
             "zh": ("随便哪个", "都可以", "平常那个", "差不多那种")},
}
# Clearly wrong values; never a word for "none" or "unknown" (N/A, 无, 待定): a request giving one of those
# reads as a missing value, the very confusion the incorrect kind is to resolve.
_JUNK = ("abc", "xyz", "12abc", "1.2.3", "a1b2")
_BROKEN_URLS = ("htp:/broken", "url??", "www..com")
OBVIOUS = {
    "number": {"en": _JUNK, "zh": _JUNK},
    "enum": {"en": ("banana", "purple", "zzz", "moonlight"), "zh": ("香蕉", "紫色", "zzz", "火星")},
    "url": {"en": _BROKEN_URLS, "zh": _BROKEN_URLS},
    "text": {"en": ("???", "abc!!", "#@!"), "zh": ("???", "abc!!", "#@!")},
}
_MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December")


def _render_spec(fmt: str, parts: dict) -> str:
    """parts written in a date or time spec (ISO forms as YYYY-MM-DD[THH:MM[:SS]])."""
    if fmt in ("ISO-DATETIME", "ISO-8601"):
        text = f"{parts['y']:04d}-{parts['mo']:02d}-{parts['d']:02d}"
        if "h" in parts:
            text += f"T{parts['h']:02d}:{parts['mi']:02d}" + (f":{parts['s']:02d}" if "s" in parts else "")
        return text
    if fmt == "TIME":
        return f"{parts['h']:02d}:{parts['mi']:02d}" + (f":{parts['s']:02d}" if "s" in parts else "")
    out, hour = "", False
    for token in re.findall(r"YYYY|MM|DD|HH|SS|.", fmt):
        if token == "YYYY":
            out += f"{parts['y']:04d}"
        elif token == "MM":
            out += f"{parts['mi'] if hour else parts['mo']:02d}"
        elif token == "DD":
            out += f"{parts['d']:02d}"
        elif token == "HH":
            out, hour = out + f"{parts['h']:02d}", True
        elif token == "SS":
            out += f"{parts['s']:02d}"
        else:
            out += token
    return out


def _date_kind(parts: dict) -> str:
    return "datetime" if "y" in parts and "h" in parts else "date" if "y" in parts else "time"


def _clock(parts: dict, lang: str) -> list[str]:
    h, mi = parts["h"], parts["mi"]
    h12, half = (h % 12 or 12), ("PM" if h >= 12 else "AM")
    if lang == "en":
        return [f"{h12}:{mi:02d} {half}"] + ([f"{h12} {half.lower()}"] if mi == 0 else [])
    noon = "下午" if h >= 12 else "上午"
    return [f"{h}点{mi}分" if mi else f"{h}点", f"{noon}{h12}点" + (f"{mi}分" if mi else "")]


def _other_formats(parts: dict, lang: str) -> list[str]:
    """The same date or time written in other common formats."""
    kind = _date_kind(parts)
    out: list[str] = []
    if kind in ("date", "datetime"):
        y, mo, d = parts["y"], parts["mo"], parts["d"]
        days = ([f"{_MONTHS[mo - 1]} {d}, {y}", f"{d:02d}/{mo:02d}/{y}", f"{d} {_MONTHS[mo - 1]} {y}"]
                if lang == "en" else [f"{y}年{mo}月{d}日", f"{d:02d}/{mo:02d}/{y}"])
        if kind == "date":
            return days
        clocks = _clock(parts, lang)
        joiner = " at " if lang == "en" else ""
        out = [f"{day}{joiner}{clock}" for day in days[:2] for clock in clocks[:1]]
        return out
    return _clock(parts, lang)


def _impossible(fmt: str, parts: dict) -> list[str]:
    """The date or time in its own format with an impossible month, day, hour or minute."""
    out = []
    if "mo" in parts:
        out += [_render_spec(fmt, {**parts, "mo": 13}), _render_spec(fmt, {**parts, "d": 32})]
    if "h" in parts:
        out += [_render_spec(fmt, {**parts, "h": 25}), _render_spec(fmt, {**parts, "mi": 75})]
    return out


def _out_of_range(c: Constraint) -> list[str]:
    out = []
    if c.maximum is not None:
        top = c.maximum * 10 if c.maximum > 0 else c.maximum + 100
        out.append(_number_text(top if c.type != "integer" else int(top)))
    if c.minimum is not None:
        bottom = -(abs(c.minimum) + 5) if c.minimum >= 0 else c.minimum - 100
        out.append(_number_text(bottom if c.type != "integer" else int(bottom)))
    return out


def invalid_candidates(c: Constraint, value: object, category: str, lang: str) -> list[str]:
    """Texts that give a value breaking the constraint, in a fixed order, for one category: obvious (a wrong
    type, an impossible date, an option not listed, a number out of range), format (the same value in another
    format; only for a format the definition names) or vague (a vague amount, day, time or option); [] when
    the category does not apply. Each fails the constraint and differs from the value."""
    original = value_text(value)
    out: list[str] = []
    parts = date_parts(c.fmt, value) if is_date_format(c.fmt) else None
    if category == "obvious":
        if c.type == "integer" and isinstance(value, int):
            out.append(f"{value}.5")
        if c.minimum is not None or c.maximum is not None:
            out += _out_of_range(c)
        if c.enum:
            out += OBVIOUS["enum"][lang]
        elif parts is not None:
            out += _impossible(c.fmt, parts)
        elif c.fmt == "EMAIL" and isinstance(value, str):
            out.append(value.split("@")[0] + "@")
        elif c.fmt == "URL":
            out += OBVIOUS["url"][lang]
        elif c.fmt == "CODE3" and isinstance(value, str):
            out += [value + "X", value[:2]]
        elif c.fmt == "ZIP":
            out.append("ABCDE")
        elif c.pattern is not None and isinstance(value, str):
            out += [value + "!!", "#" + value, *OBVIOUS["text"][lang]]
        elif c.type in ("integer", "number"):
            out += OBVIOUS["number"][lang]
        if c.type in ("integer", "number") and not c.enum:
            out += [w for w in OBVIOUS["number"][lang] if w not in out]
    elif category == "format":
        if parts is not None:
            out += _other_formats(parts, lang)
        elif c.fmt == "EMAIL" and isinstance(value, str) and "@" in value:
            local, _, domain = value.partition("@")
            out.append(f"{local} at {domain.replace('.', ' dot ')}")
        elif c.fmt == "URL" and isinstance(value, str):
            out.append(re.sub(r"^https?://", "", value).replace(".", " dot "))
        elif c.pattern is not None and isinstance(value, str) and date_parts("YYYY-MM-DD", value):
            out += _other_formats(date_parts("YYYY-MM-DD", value), lang)
    elif category == "vague":
        if c.enum:
            out += VAGUE["enum"][lang]
        elif parts is not None:
            out += VAGUE[_date_kind(parts)][lang]
        elif c.type in ("integer", "number"):
            out += VAGUE["number"][lang]
    seen, kept = set(), []
    for text in out:
        if text in seen or text == original or not text.strip():
            continue
        seen.add(text)
        if c.enum and any(str(m).casefold() == text.casefold() for m in c.enum):
            continue
        if not passes(c, read_value(text, c)):
            kept.append(text)
    return kept


def invalid_value(c: Constraint, value: object, category: str, lang: str, rng: random.Random) -> str | None:
    """One of invalid_candidates, drawn with rng; None when the category does not apply."""
    found = invalid_candidates(c, value, category, lang)
    return rng.choice(found) if found else None


_SIGN_BEFORE = re.compile(r"[$€£¥￥]\s*$")
# "around [some] kilowatt-hours" (hand check, round 5 of the fix round's review)
_APPROXIMATE_BEFORE = re.compile(r"\b(?:around|about|approximately|approx|roughly|nearly|almost|circa)"
                                 r"\.?\s*$", re.IGNORECASE)
# Chinese vague amounts holding 个, or taking none, count only the nouns that take them: 几个人 and 几个小时,
# not 几个条 or 一些米 (hand check, round 3 of the fix round's review)
_AMOUNT_NOUNS_ZH = {"几个": ("人", "小时"), "好几个": ("人", "小时"), "一些": ("人",)}
# the words after which no vague word stands: an article, a determiner, a ranking word ("the [some] season",
# "the top [several]", "in the [any of them]"; hand check, round 7)
_BEFORE_VAGUE_EN = frozenset(("a", "an", "the", "my", "our", "your", "his", "her", "their", "its", "this",
                              "that", "these", "those", "each", "every", "top", "first", "last", "next",
                              "per"))


_RANGE_BEFORE = re.compile(r"\d\s*(?:to|and|-|–|—|~|到|至)\s*$")  # hand check, round 9
_RANGE_AFTER = re.compile(r"\s*(?:to|-|–|—|~|到|至)\s*\d")


def _vague_kind(text: str) -> str | None:
    """The VAGUE list a text belongs to (number, date, time, datetime, enum), None for any other."""
    return next((kind for kind, words in VAGUE.items() if text in words["en"] or text in words["zh"]), None)


def wrong_value_fits(text: str, old: str, new: str, lang: str) -> bool:
    """Whether the wrong value new can stand where old stands in text, once (hand check, rounds 3 to 7 of the
    fix round's review): after a currency sign it starts with a digit and before a percent sign it ends with
    one ("$a dozen or so" and "some%" are no requests). A vague word (VAGUE), in neither language in a
    coordination ("in [X] and Asia", [华东]、华北), a vague amount in no range ("100 to [a few]",
    100到[几十]); English, never after an article, a determiner or a
    ranking word (_BEFORE_VAGUE_EN: "during the [some] season"); a vague amount replaces only a number that
    counts what follows it, a word that is no function, time or relative word ("40 hours", not "version 2
    of" or "101 which"), with no approximating word before it ("around [some] kilowatt-hours"); a vague option
    is no modifier of a noun after it ("a [whichever] movie"); a vague day or time follows no on, at, in or
    during, and one that opens with in or around follows no preposition ("at [in the afternoon]"). Chinese,
    a vague amount stands before a unit or a classifier (_UNIT_ZH: 每页30条, not 版本号为2的 or 工作空间
    4471里; 月, a month's, aside) that takes it (_AMOUNT_NOUNS_ZH: 几个人, not 几个条); a vague option
    before no 的 (状态为[随便哪个]的任务)."""
    found = occurrences(old, text)
    if len(found) != 1:
        return False
    before, after = text[:found[0].start()], text[found[0].end():]
    if (_SIGN_BEFORE.search(before) and not new[:1].isdigit()) or \
            (after[:1] in ("%", "％") and not new[-1:].isdigit()):
        return False
    kind = _vague_kind(new)
    if kind is None:
        return True
    if kind == "number" and (_RANGE_BEFORE.search(before) or _RANGE_AFTER.match(after)):
        return False  # a range: "about 100 to [a few] characters", 100到[几十]个字
    if lang == "en":
        prev = re.search(r"([A-Za-z]+)[ \t]*$", before)
        prev_word = prev.group(1).lower() if prev else ""
        word = re.match(r"[ \t]+([A-Za-z]+)", after)
        if prev_word in _BEFORE_VAGUE_EN or prev_word in ("and", "or", "nor") or \
                re.match(r"[ \t]*(?:,|(?:and|or|nor)\b)", after):  # a coordination: "in [X] and Asia"
            return False
        if kind == "number":
            return word is not None and word.group(1).lower() not in _NO_HEAD_EN | _QUESTION_EN and \
                not _APPROXIMATE_BEFORE.search(before)
        if kind == "enum":
            return not (word and _head_noun_en(word.group(1)))
        if prev_word in ("on", "at", "in", "during"):
            return False
        return not (new.startswith(("in ", "around ")) and prev_word in _PREPOSITIONS_EN)
    joins = ("、", "和", "与", "及", "或")
    if before.rstrip()[-1:] in joins or after.lstrip()[:1] in joins:
        return False  # a coordination: 查询[华东]、华北的实例
    if kind == "enum":
        return not after.lstrip().startswith("的")
    if kind != "number":
        return True
    unit = _UNIT_ZH.match(after)
    if unit is None or unit.group(0).strip() in ("月", "%"):
        return False
    takes = _AMOUNT_NOUNS_ZH.get(new)
    return takes is None or unit.group(0).strip() in takes


def nearmiss_value(c: Constraint, value: object) -> tuple[object, str] | None:
    """(a valid value close to a bound, why it is valid): a leap day for a date, the end of the day for a
    time, the maximum or the minimum, each other than value; or, for an option of an enum, the option itself
    (an exact option, kept as the request gives it: swapping it for another left requests saying the
    opposite, hand check, rounds 1 to 3); None when there is none."""
    parts = date_parts(c.fmt, value) if is_date_format(c.fmt) else None
    dated: list[tuple[object, str]] = []
    if parts is not None and "y" in parts:
        year = 2028 if (parts["y"], parts["mo"], parts["d"]) == (2024, 2, 29) else 2024
        dated.append((_render_spec(c.fmt, {**parts, "y": year, "mo": 2, "d": 29}), "leap"))
    elif parts is not None:
        end = {**parts, "h": 23, "mi": 59, **({"s": 59} if "s" in parts else {})}
        dated.append((_render_spec(c.fmt, end), "time"))
    bounds: list[tuple[object, str]] = []
    if c.maximum is not None and not c.max_exclusive:
        bounds.append((c.maximum, "maximum"))
    if c.minimum is not None and not c.min_exclusive:
        bounds.append((c.minimum, "minimum"))
    for candidate, reason in dated:
        if candidate != value and passes(c, candidate) and type(candidate) is type(value):
            return candidate, reason
    if c.enum and value in c.enum and len(c.enum) > 1:
        return value, "enum"
    for candidate, reason in bounds:
        if candidate != value and passes(c, candidate) and type(candidate) is type(value):
            return candidate, reason
    return None


def other_value(c: Constraint | None, value: object, bank: list, rng: random.Random) -> object | None:
    """Another valid value of the same kind (for a request that changes a value): another option, a nearby
    number, the next day, or a string the bank holds for the parameter; None when there is none."""
    candidates: list = []
    parts = date_parts(c.fmt, value) if c is not None and is_date_format(c.fmt) else None
    if c is not None and c.enum:
        candidates = [m for m in c.enum if m != value]
    elif parts is not None and "y" in parts:
        day = dt.date(parts["y"], parts["mo"], parts["d"])
        for step in (1, 2, 7):
            later = day + dt.timedelta(days=step)
            moved = {**parts, "y": later.year, "mo": later.month, "d": later.day}
            candidates.append(_render_spec(c.fmt, moved))
    elif isinstance(value, int) and not isinstance(value, bool):
        candidates = [value + step for step in (1, 2, 5, 10, -1, -2)]
    elif isinstance(value, float):
        candidates = [round(value * 2, 2), round(value + 1, 2)]
    elif isinstance(value, str):
        candidates = [v for v in bank if isinstance(v, str) and v != value]
    candidates = [v for v in candidates if type(v) is type(value) and (c is None or passes(c, v))]
    return rng.choice(candidates) if candidates else None


# ---- cuts with their frame, fluency, whole objects --------------------------------------------------------

# "worth" takes an amount as a preposition does: "a car worth $25000" (hand check, round 5); the list lacked
# "behind" and its kind ("the ideas behind [X] according to"; round 6)
_PREPS_EN = ("in|at|on|for|to|from|with|of|by|about|into|near|via|as|within|during|until|after|before|around|"
             "per|under|over|across|through|toward|towards|worth|behind|beyond|below|above|along|against|"
             "beside|besides|inside|outside|upon|without|regarding|concerning|onto|despite|except|unlike|"
             "throughout|including|among")
_ARTICLES_EN = "a|an|the|my|our|your|his|her|their|its|this|that"
_NAMING_EN = "named|called|titled|entitled|labeled|labelled"
# and the participles and adjectives a preposition completes ("released in", "set in", "according to"; hand
# check, round 6 of the fix round's review: "a movie released.", "her adversary set?", "behind according?")
_PARTICIPLES_EN = ("based|located|situated|priced|valued|dated|scheduled|starting|ending|released|published|"
                   "made|born|founded|written|directed|produced|built|created|filmed|set|held|hosted|recorded|"
                   "taken|according|close|next|prior|similar|related|compared|relative")
# what stands right before a value as its frame: a naming word, or a preposition with an article or not and
# with the participle it completes ("based on", "located in"; the fix round's review, F3)
_FRAME_EN = re.compile(rf"(?:\b(?:{_NAMING_EN})|(?:\b(?:{_PARTICIPLES_EN})\s+)?\b(?:{_PREPS_EN})"
                       rf"(?:\s+(?:{_ARTICLES_EN}))?)\s+$", re.IGNORECASE)
# what follows a value that ends its noun phrase, so that a frame before it goes with it: a mark, the end, or
# a word opening another part ("for a comedy movie" keeps "for a" for "movie"; the fix round's review, F3)
_ENDS_PHRASE_EN = re.compile(  # never "as": "with Johnny Harris as the director" gives the value a role
    r"[\"'”’)\]]*(?:\s*(?:[,.;:!?)\]]|$)|\s+(?:and|or|but|so|then|with|for|to|in|on|at|from|by|of|about|if|"
    r"when|while|because|please|thanks|that|which|who|where|since|until|before|after|during|via|using|"
    r"including|instead|also|too|now|today|tomorrow|tonight|next|this|first|again|only|asap)\b)",
    re.IGNORECASE)
_UNITS_EN = ("minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|nights?|seconds?|secs?|people|persons?|"
             "guests?|adults?|children|kids?|rooms?|items?|units?|pieces?|tickets?|seats?|servings?|pages?|"
             "results?|words?|characters?|times|km|kilometers?|kilometres?|miles?|meters?|metres?|m|cm|mm|"
             "inches?|feet|foot|ft|kg|kilograms?|grams?|g|lbs?|pounds?|ounces?|oz|liters?|litres?|gallons?|"
             "dollars?|usd|euros?|eur|gbp|degrees?|calories|percent|%")
_UNIT_EN = re.compile(rf"\s*(?:{_UNITS_EN})(?![A-Za-z])", re.IGNORECASE)
_UNIT_ZH = re.compile(r"\s*(?:秒钟|秒|分钟|小时|天|周|个月|月|年|人|位|个|条|次|页|元|美元|公里|千米|米|"
                      r"千克|公斤|克|份|张|间|台|核|GB|MB|KB|TB|G|M|K|%)")
_CURRENCY = re.compile(r"[$€£¥]\s*$")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_CLAUSE_MARKS = "，,。.！!？?；;：:、\n"
_CLAUSE_STOPS = "，,。！!？?；;、\n"  # where a Chinese clause ends (a colon introduces a value)
_LABEL_ZH = re.compile(r"(?:为|是|：|:)\s*$")
_NAMING_ZH = re.compile(r"(?:命名为|名称为|名字为|名称是|名字是|名为|名叫|叫做|称为|叫)\s*$")
_PREPS_ZH = re.compile(r"(?:在|从|到|给|于|向|对|用|按|以|由|自|至|往|跟|同|与|和)\s*$")
_IS_ZH = re.compile(r"(?:为|是)\s*$")
# a word left right before a cut that needs what was cut; one right after it
_DETERMINERS_EN = frozenset((*_ARTICLES_EN.split("|"), "these", "those", "some", "any", "each", "every",
                             "both", "either", "neither", "no"))
_APPROXIMATORS_EN = ("approximately", "approx", "roughly", "nearly", "almost", "exactly", "precisely",
                     "circa")  # a word that needs the amount after it ("approximately [$500]"; round 5)
DANGLING_BEFORE_EN = frozenset((
    *_PREPS_EN.split("|"), *_DETERMINERS_EN, *_NAMING_EN.split("|"), *_PARTICIPLES_EN.split("|"), "between",
    "among", "than", "like", "is", "are", "was", "were", "be", "been", "being", "am", "and", "or", "nor",
    "but", "if", "whether", "equal", "equals", *_APPROXIMATORS_EN))
DANGLING_AFTER_EN = frozenset(("and", "or", "nor", "of"))
_TIME_WORDS_EN = frozenset(("today", "tomorrow", "tonight", "now", "yesterday", "soon", "later", "asap",
                            "please", "thanks", "again", "too", "also", "first", "next", "instead"))
# 包含 and 包括 need their object as the prepositions do (hand check, round 6)
DANGLING_BEFORE_ZH = frozenset("为是叫在从到给于向对和与及或把将被让用按以由自至跟同比等名含括")
DANGLING_AFTER_ZH = frozenset("和与及或、至")
_EMPTY_PAIR = re.compile(r'""|\'\'|“”|‘’|「」|《》|『』|\(\s*\)|\[\s*\]|\{\s*\}|（\s*）|【\s*】|<\s*>|\\"\\"|'
                         r"\\'\\'")
_DANGLING_END_EN = re.compile(rf"\b(?:{_PREPS_EN}|{_ARTICLES_EN}|{_NAMING_EN}|between|and|or|nor|if|whether)"
                              r"\s*(?:[,.;:!?)\]]|$)", re.IGNORECASE)
_DANGLING_PAIR_EN = re.compile(r"\b(?:between|from)\s+(?:and|to)\b|\b(?:the|a|an)\s+(?:and|or|of|to|in|on|at|"
                               r"for|with)\b", re.IGNORECASE)
_DANGLING_ZH = re.compile(r"[为是叫在从到给于向对和与及或把将被让用按以由自至][，,。.！!？?；;：:、的]|^的")
# what a cut can make that reads broken, each counted only where the request before the cut held none like it
# (the fix round's review, F3, and its hand check): Chinese, a doubled word (用户用户), a postposition opening
# the request or following a verb (下非运行..., 获取中...); English, a sentence of one to three words
# ("It's.", "The dimensions.", "The object weighs."), a participle without its object ("rates based?"), a
# copula before a mark ("was originally,", "I am currently."), an infinitive ending a sentence ("I need to
# get."), a verb needing what follows ("the period ranges,", "as follows."); either, a hyphen left from a
# compound (" -related"), a currency sign without its number, a colon ending the request ("Here it is:") or
# before a mark ("the API key:."), an empty code block
_WORDS_AFTER_POSTPOSITION = "午国文载周个次一面容存级班层部方确心"  # 下午, 中国, 里面, 内容: none
_DOUBLED_ZH = re.compile(r"([一-鿿]{2,4})\1")
_OPENS_ZH = re.compile(rf"^\s*[下中里内的](?![{_WORDS_AFTER_POSTPOSITION}])")
_REQUEST_VERBS_ZH = ("获取|查询|查看|删除|更新|创建|新建|修改|设置|添加|移除|启动|停止|统计|列出|下载|"
                     "上传|导出|导入|发送|取消|开通|关闭|开启|打开|绑定|解绑|授权|授予|撤销|恢复|重启|"
                     "重置|运行|执行|提交|检查|检测|分析|计算|生成|转换|搜索")
_VERB_GAP_ZH = re.compile(rf"(?:{_REQUEST_VERBS_ZH})[下中里内](?![{_WORDS_AFTER_POSTPOSITION}])")
_FRAGMENT_EN = re.compile(r"(?:^|(?<=[.!?])\s+)((?:[A-Za-z']+\s?){1,3}[.!?])(?=\s|$)")
_PARTICIPLE_LEFT = re.compile(rf"\b(?:{_PARTICIPLES_EN})\s*(?:[,.;:!?)]|$)", re.IGNORECASE)
_COPULA_LEFT = re.compile(r"(?:\b(?:is|are|was|were|be|been|am)|'(?:m|re|s))(?:\s+[a-z]+ly)?"
                          r"\s*(?:[,.;:!?]|\b(?:but|and|or|so|yet)\b)", re.IGNORECASE)  # "was originally but"
_INFINITIVE_LEFT = re.compile(r"\bto\s+[a-z]+\s*[.!?;]")  # "I need to get." with "from A to B" cut
# a verb that needs what follows it, before a mark: "the period ranges," "it weighs." "as follows."
_VERB_LEFT = re.compile(r"\b(?:ranges|weighs|costs|measures|lasts|spans|totals|amounts|follows|reads|says|"
                        r"reaches|contains|includes|equals)\s*[,.;:!?]", re.IGNORECASE)
_EMPTY_FENCE = re.compile(r"```[A-Za-z]*\s*```")  # a code block whose content was cut
_HYPHEN_LEFT = re.compile(r"(?:^|\s)-[A-Za-z]|[A-Za-z]-(?=\s|$|[,.;:!?])")
_CURRENCY_LEFT = re.compile(r"[$€£¥](?!\s*\d)")
_ENDS_OPEN = re.compile(r"[:：][ \t]*(?:\n|$)")  # "Here it is:", "- User ID:" with what they announced cut
_COLON_STOP = re.compile(r"[:：]\s*[.。!！?？;；]")  # "the API key:." with the key cut
_MADE = {"en": ((_FRAGMENT_EN, 1), (_PARTICIPLE_LEFT, 0), (_COPULA_LEFT, 0), (_INFINITIVE_LEFT, 0),
                (_VERB_LEFT, 0), (_HYPHEN_LEFT, 0), (_CURRENCY_LEFT, 0), (_ENDS_OPEN, 0), (_COLON_STOP, 0),
                (_EMPTY_FENCE, 0)),
         "zh": ((_DOUBLED_ZH, 0), (_OPENS_ZH, 0), (_VERB_GAP_ZH, 0), (_HYPHEN_LEFT, 0), (_CURRENCY_LEFT, 0),
                (_ENDS_OPEN, 0), (_COLON_STOP, 0), (_EMPTY_FENCE, 0))}


@dataclass(frozen=True)
class Cut:
    """A request with values cut out: the text, and whether it reads fluently (no word left dangling where a
    value went, and reads_fluently)."""

    text: str
    fluent: bool


def _made(pattern: re.Pattern, group: int, text: str, original: str) -> bool:
    """Whether text holds a match of pattern (its group) more often than the original does (False without
    one: what a cut made cannot be told then)."""
    if not original:
        return False
    found = Counter(match.group(group) for match in pattern.finditer(text))
    return bool(found - Counter(match.group(group) for match in pattern.finditer(original)))


def reads_fluently(text: str, lang: str, original: str = "") -> bool:
    """A heuristic of whether a request reads fluently after a cut: no doubled space, no space before a
    punctuation mark, no empty quotes or brackets, no punctuation at its start or doubled, no function word
    left dangling (English: a preposition, an article, a naming word or a conjunction before a punctuation
    mark or the end, "between and"; Chinese: a preposition, 为, 是 or a conjunction before a punctuation mark
    or 的, as in 对的 or 名为的), and, with the original request given, nothing broken that the cut made and
    the original holds none like (_MADE: a doubled word, a postposition opening the request or following a
    verb, a sentence of up to three words, a participle, a copula or an infinitive without what completes it,
    a hyphen left from a compound, a currency sign without its number, a colon left open)."""
    if "  " in text or re.search(r"\s[,.;:!?，。；：！？、)\]}」』]", text) or _EMPTY_PAIR.search(text):
        return False
    if re.match(r"\s*[,.;:!?，。；：！？、]", text) or re.search(r"[，,、；;]\s*[，,、；;。.!?！？]", text):
        return False
    if lang == "en" and (_DANGLING_END_EN.search(text) or _DANGLING_PAIR_EN.search(text)):
        return False
    if lang != "en" and _DANGLING_ZH.search(text):
        return False
    made = _MADE["en" if lang == "en" else "zh"]
    return not any(_made(pattern, group, text, original) for pattern, group in made)


def _tidy(text: str) -> str:
    """A cut request tidied: single spaces, no space before a punctuation mark or after an opening bracket, a
    clause mark doubled by the cut dropped, no punctuation at the start."""
    text = re.sub(r"[.。][ \t]+(?=[.。])", "", text)  # a sentence cut whole leaves ". ."
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.?!;:，。？！；：、)\]}」』”’])", r"\1", text)
    text = re.sub(r"([(\[{「『“‘])[ \t]+", r"\1", text)
    text = re.sub(r"([，,、；;])\s*([，,、；;。.!?！？])", r"\2", text)
    text = re.sub(r"^[\s，,、；;:：.]+", "", text)
    return text.strip()


_NO_HEAD_EN = _STOP_EN | DANGLING_BEFORE_EN | _TIME_WORDS_EN
# words that describe a noun before it: one left before what is no noun lost the noun it described ("find
# nearby [restaurants] within"; hand check, round 2 of the fix round's review)
_DESCRIBERS_EN = frozenset((
    "nearby", "local", "popular", "best", "top", "cheap", "cheapest", "good", "great", "new", "latest",
    "recent", "upcoming", "similar", "other", "more", "few", "nearest", "closest", "favorite", "favourite",
    "specific", "certain", "particular", "current", "available", "different", "interesting", "daily",
    "weekly", "monthly", "annual", "yearly", "hourly"))


def _head_noun_en(word: str) -> bool:
    """Whether an English word can head the noun phrase a cut modifier leaves: letters, no function, time or
    politeness word, and no -ing or -ed form (a participle: "in [Paris] serving sushi")."""
    low = word.lower()
    return low.isalpha() and low not in _NO_HEAD_EN and not low.endswith(("ing", "ed"))


# verbs an amount completes: one left before a cut amount with no noun after the cut lost its object ("and
# burned [500].", "I need [2 tickets] for Friday"; hand check, round 3 of the fix round's review); any -ed
# form too (_amount_verb_en)
_AMOUNT_VERBS_EN = frozenset((
    "need", "needs", "want", "wants", "have", "has", "had", "get", "gets", "got", "buy", "buys", "bought",
    "pay", "pays", "paid", "spend", "spends", "spent", "earn", "earns", "burn", "burns", "burnt", "eat",
    "eats", "ate", "drank", "ran", "take", "takes", "took", "make", "makes", "made", "sell", "sells", "sold",
    "lose", "loses", "lost", "weigh", "weighs", "owe", "owes", "borrow", "borrows", "lend", "lends", "lent",
    "won", "add", "adds", "send", "sends", "sent", "give", "gives", "gave", "reserve", "reserves", "invest",
    "invests"))
_AMOUNT_START = re.compile(r"[\"'“‘(\[]*[$€£¥]?\s*\d")
# the words after which the next word is a verb: the infinitive's to, a subject pronoun, a modal, please, a
# relative pronoun; a verb there whose object a bare cut takes before a preposition or a sentence's end is
# left without it ("I need to add [apples] to my list", "Could you please play [X]?"; hand check, round 4 of
# the fix round's review; "a recipe that uses [chicken].", round 6)
_BEFORE_VERB_EN = frozenset(("to", "i", "you", "we", "they", "please", "kindly", "can", "could", "will",
                             "would", "should", "must", "may", "might", "shall", "let's", "lets", "that",
                             "which", "who"))
_PREPOSITIONS_EN = frozenset(_PREPS_EN.split("|"))
_FRAME_WORDS_EN = _PREPOSITIONS_EN | frozenset((*_NAMING_EN.split("|"), *_PARTICIPLES_EN.split("|")))


def _amount_verb_en(word: str) -> bool:
    """Whether an English word is a verb an amount completes (_AMOUNT_VERBS_EN, or an -ed form other than
    -eed: speed and seed are nouns)."""
    low = word.lower()
    return low in _AMOUNT_VERBS_EN or (len(low) >= 5 and low.endswith("ed") and not low.endswith("eed"))


def _site_ok(text: str, start: int, end: int, lang: str) -> bool:
    """Whether cutting text[start:end] leaves no dangling word at the cut: English, the word before it is no
    function word that needs a value (DANGLING_BEFORE_EN), unless it is an article or a determiner and the
    noun the cut modifier described follows the cut ("for a [comedy] movie": _head_noun_en; the fix round's
    review, F3), nor a describing word without such a noun after the cut (_DESCRIBERS_EN), nor a verb whose
    object the cut takes at a sentence's end ("and pursue [my hobby].") or, the verb after to, a subject or
    a modal (_BEFORE_VERB_EN), before a preposition ("to add [apples] to my list"), or whose amount it takes
    with no noun after it ("and burned [500].": _amount_verb_en), and the word after it no conjunction
    (DANGLING_AFTER_EN); Chinese, the characters around it are none of
    DANGLING_BEFORE_ZH (but a marker before a label phrase cut whole leaves its object to the noun after it:
    _MARKERS_ZH) and DANGLING_AFTER_ZH, and a cut ending a clause removes the whole clause (from its mark
    before, or the start)."""
    quotes = " \t\"'“”‘’「」"
    before, after = text[:start].rstrip(quotes), text[end:].lstrip(quotes)
    if lang == "en":
        prev = re.search(r"([A-Za-z0-9']+)$", before)
        nxt = re.match(r"([A-Za-z0-9']+)", after)
        if nxt and nxt.group(1).lower() in DANGLING_AFTER_EN:
            return False
        head_after = nxt is not None and _head_noun_en(nxt.group(1))
        if prev is not None and prev.group(1).lower() in _DESCRIBERS_EN and not head_after:
            return False
        first = re.match(r"[\"'“‘(\[]*([A-Za-z]+)", text[start:end])
        if first and first.group(1).lower() in _DETERMINERS_EN and prev is not None and \
                prev.group(1).isalpha() and prev.group(1).lower() not in _NO_HEAD_EN and \
                re.match(r"\s*[.!?;]", after):  # a verb's object cut at the end: "and pursue."
            return False
        if prev is not None and _amount_verb_en(prev.group(1)) and not head_after and \
                _AMOUNT_START.match(text, start):  # an amount cut from its verb: "and burned."
            return False
        verb = re.search(r"([A-Za-z']+)\s+([A-Za-z]+)$", before)
        framed = first is not None and first.group(1).lower() in _FRAME_WORDS_EN
        object_cut = verb is not None and verb.group(1).lower() in _BEFORE_VERB_EN and \
            verb.group(2).lower() not in _NO_HEAD_EN and not framed
        ends = re.match(r"\s*[.!?]", after) or (nxt is not None and nxt.group(1).lower() in _PREPOSITIONS_EN)
        if object_cut and ends:
            return False  # a verb's object cut: "to add [apples] to my list", "please play [X]?"
        if prev is None or prev.group(1).lower() not in DANGLING_BEFORE_EN:
            return True
        return prev.group(1).lower() in _DETERMINERS_EN and head_after
    prev, nxt = before[-1:], after[:1]
    if nxt in DANGLING_AFTER_ZH:
        return False
    if prev in DANGLING_BEFORE_ZH and not (prev in _MARKERS_ZH and _takes_noun(text[start:end], nxt)):
        return False
    whole = not prev or prev in _CLAUSE_MARKS or text[start:end].lstrip(quotes)[:1] in _CLAUSE_MARKS
    return whole or bool(nxt and nxt not in _CLAUSE_MARKS)


# markers whose object a label phrase cut whole leaves to the noun after it (将[表的唯一标识为X的]表: 将表;
# hand check, round 1 of the fix round's review)
_MARKERS_ZH = frozenset("将把被让给对向在从于")


def _takes_noun(span: str, nxt: str) -> bool:
    """Whether a Chinese cut ends a modifier (its last character 的) and a noun follows it."""
    return span.rstrip(" \t\"'“”‘’「」").endswith("的") and bool(nxt) and bool(_CJK.match(nxt)) and \
        nxt not in "的了吗呢吧啊"


def _widen_pairs(text: str, start: int, end: int) -> tuple[int, int]:
    """A span widened over the quotes or brackets that enclose exactly it, quotes escaped with a backslash
    too (\\"sales\\" left \\"\\" behind; hand check, round 8)."""
    pairs = {**_QUOTE_PAIRS, "(": ")", "[": "]", "（": "）", "【": "】"}
    while True:
        if start > 0 and end < len(text) and pairs.get(text[start - 1]) == text[end]:
            start, end = start - 1, end + 1
        elif start > 1 and text[start - 2:start] in ('\\"', "\\'") and \
                text[end:end + 2] == text[start - 2:start]:
            start, end = start - 2, end + 2
        else:
            return start, end


_OFF_EN = re.compile(r"\s+off(?![A-Za-z])", re.IGNORECASE)


def _unit_end(text: str, end: int, value: str, lang: str) -> int:
    """end moved past the unit that follows a number (90 minutes, 3600秒), if one does, and in English past
    an "off" after them ("15% off": cut alone, "it's off" says the opposite; hand check, round 6)."""
    if not _NUMBER.fullmatch(value.strip()):
        return end
    unit = (_UNIT_EN if lang == "en" else _UNIT_ZH).match(text, end)
    end = unit.end() if unit else end
    off = _OFF_EN.match(text, end) if lang == "en" else None
    return off.end() if off else end


def _bare(text: str) -> str:
    """A Chinese label or description as compared: _norm, no spaces, no 的 (工作空间的ID holds 工作空间ID)."""
    return "".join(_norm(text).split()).replace("的", "")


_VERB_ZH = re.compile(_REQUEST_VERBS_ZH + "|我想|我要|帮我|请")
_OPENER_VERB_ZH = re.compile(rf"(?:我想|我要|我需要|帮我|麻烦|请|需要|想要)\s*(?:{_REQUEST_VERBS_ZH})")


def _clause_span(text: str, start: int, end: int, lang: str, description: str = "") -> tuple[int, int] | None:
    """The span of a Chinese clause that only labels the value ("，表名为orders"): the clause between two
    clause marks reads <label>为|是|：<value>[unit], where the label names no request verb and is the
    parameter's, as its description holds it (_bare), or, with no description, follows a clause mark (the
    fix round's review, F3: a first clause 我想删除工作空间ID为X went whole, the request's verb with it); it
    goes with the mark before it (or the one after it, at the start). None otherwise."""
    left = max(text.rfind(mark, 0, start) for mark in _CLAUSE_STOPS)
    rights = [i for i in (text.find(mark, end) for mark in _CLAUSE_STOPS) if i >= 0]
    right = min(rights) if rights else len(text)
    head, tail = text[left + 1:start], text[_unit_end(text, end, text[start:end], lang):right]
    label = _LABEL_ZH.search(head)
    if label is None or tail.strip():
        return None
    name = head[:label.start()].strip(" \t" + _LABEL_QUOTES)
    if not 1 <= len(name) <= 12 or _VERB_ZH.search(name):
        return None
    if (_bare(name) not in _bare(description)) if description else left < 0:
        return None
    if left >= 0:
        return left, right
    if right < len(text):
        return 0, right + 1
    return None


_POSTPOSITION = re.compile(rf"[下中里内](?![{_WORDS_AFTER_POSTPOSITION}])的?")
_LABEL_NEXT = re.compile(r"\s*[^\s，,。.！!？?；;：:、]{1,12}?(?:为|是|：|:)")  # a label opening a value


def _phrase_end(text: str, start: int, end: int) -> tuple[int, int]:
    """A Chinese span that cuts a value's whole noun phrase (its label, naming verb or preposition with it)
    widened over what the phrase leaves without its partner: a postposition after it (工作空间ID为X下的 ...,
    在X中), and an enumeration mark joining it to the next value or the one before (项目名称为X、角色名称为Y;
    the fix round's review, F3), or a comma joining it, as 、 does, to a next labelled value
    (获取项目名称为X，分区表为Y的...: 获取， was left; its hand check)."""
    found = _POSTPOSITION.match(text, end)
    if found:
        end = found.end()
    if text[end:end + 1] == "、" or (text[end:end + 1] in "，," and _LABEL_NEXT.match(text, end + 1)):
        end += 1
    elif text[start - 1:start] == "、" and start > 0:
        start -= 1
    return start, end


_LOCATIVE_ZH = "在从于自"
_POSTPOSITIONS_ZH = "下中里内上"


def _with_preposition(text: str, start: int, end: int) -> tuple[int, int] | None:
    """A Chinese label phrase's span widened over the locative preposition before it (在, 从, 于, 自) when the
    phrase the preposition opens closes within the span (在X中) or nothing before the clause's end closes it:
    我想知道在[工作空间ID为X的]成员角色信息 loses the whole 在 phrase (hand check, round 1 of the fix round's
    review); None otherwise (在[...的]项目中创建 keeps 在 for 项目中)."""
    if start == 0 or text[start - 1] not in _LOCATIVE_ZH:
        return None
    if text[start:end].rstrip("的")[-1:] in _POSTPOSITIONS_ZH:
        return start - 1, end
    stops = [i for i in (text.find(mark, end) for mark in _CLAUSE_STOPS) if i >= 0]
    rest = text[end:min(stops) if stops else len(text)]
    return None if re.search(f"[{_POSTPOSITIONS_ZH}]", rest) else (start - 1, end)


def _unrepeated(text: str, start: int, end: int) -> tuple[int, int]:
    """A Chinese span widened leftwards over the noun before it that the text right after it repeats, which
    the cut would double (查询同步[实例ID为X的]同步任务, [ID为X的]用户 after 用户; the fix round's review,
    F3)."""
    for size in range(6, 1, -1):
        word = text[start - size:start] if start >= size else ""
        if word and _CJK_RUN.fullmatch(word) and text[end:end + size] == word:
            return start - size, end
    return start, end


def _frames(text: str, start: int, end: int, lang: str, description: str) -> list[tuple[int, int]]:
    """The spans to try cutting for a value at text[start:end], with its frame first and the bare value
    (widened over enclosing quotes, with its currency sign and unit) last (cut_with_frames). An English frame
    is tried only when the value ends its noun phrase (_ENDS_PHRASE_EN). The Chinese frames, in order: the
    clause that only labels it, its label as the parameter's description gives it (of 工作空间ID为X, the
    whole label: 的 aside in both; never reaching back over a request verb after 我想, 请 ...:
    _OPENER_VERB_ZH), a naming verb, a preposition (these three with what _phrase_end adds, and
    first with the locative preposition before them: _with_preposition), 为X, X的; each widened over a noun
    the cut would double (_unrepeated)."""
    start, end = _widen_pairs(text, start, end)
    value = text[start:end]
    spans = []
    if lang == "en":
        tail = _unit_end(text, end, value, lang)
        currency = _CURRENCY.search(text[:start])
        head = currency.start() if currency else start  # "$" + 100 + "dollars" go together
        frame = _FRAME_EN.search(text[:head]) if _ENDS_PHRASE_EN.match(text, tail) else None
        if frame:
            spans.append((frame.start(), tail))
        spans.append((head, tail))
        return spans
    tail = _unit_end(text, end, value, lang)
    the = tail + 1 if text[tail:tail + 1] == "的" else tail  # X的: the 的 goes too
    clause = _clause_span(text, start, end, lang, description)
    if clause:
        spans.append(clause)
    phrases = []
    label = _LABEL_ZH.search(text, 0, start)
    if label and description:
        known = _bare(description)
        head = text[:label.start()]
        # never back over the request's verb, even where the description holds it (要获取的实例ID:
        # 我想[获取实例ID为X的]实例的生命周期 lost 获取; hand check, round 6)
        floor = max((m.end() for m in _OPENER_VERB_ZH.finditer(head)), default=0)
        for size in range(min(15, len(head) - floor), 1, -1):
            suffix = head[-size:]
            if not re.search(r"[\s，,。.！!？?；;：:、]", suffix) and _bare(suffix) in known:
                phrases.append((label.start() - size, the))
                break
    naming = _NAMING_ZH.search(text, 0, start)
    if naming:
        phrases.append((naming.start(), the))
    preposition = _PREPS_ZH.search(text, 0, start)
    if preposition:
        phrases.append((preposition.start(), the))
    for s, e in phrases:
        s, e = _phrase_end(text, s, e)
        wider = _with_preposition(text, s, e)
        spans += [wider, (s, e)] if wider else [(s, e)]
    being = _IS_ZH.search(text, 0, start)
    if being:
        spans.append((being.start(), tail))
    if the > tail:
        spans.append((start, the))
    spans.append((start, end))
    return [_unrepeated(text, s, e) for s, e in spans]


def _cut_span(text: str, start: int, end: int, lang: str, description: str = "") -> tuple[str, bool]:
    """text without the value at text[start:end]: the first of its frames (_frames) whose cut leaves no
    dangling word and reads fluently (compared with text: reads_fluently), else the bare value; tidied, and
    whether the cut is fluent."""
    spans = _frames(text, start, end, lang, description)

    def cut(s: int, e: int) -> str:  # tidied, an English article before the cut agreeing with what follows
        head = text[:s].rstrip(" \t") if text[e:e + 1] in "—–" else text[:s]  # a dash joins the word before
        joined = head + text[e:]
        if lang == "en":
            joined = agree_article(joined, len(head) + len(text[e:]) - len(text[e:].lstrip(" \t")))
        return _tidy(joined)

    for s, e in spans:
        rest = cut(s, e)
        if _site_ok(text, s, e, lang) and reads_fluently(rest, lang, text):
            return rest, True
    return cut(*spans[-1]), False


def _pair_span(text: str, values: list[str]) -> tuple[int, int, list[str]] | None:
    """An English "between X and Y" or "from X to Y" span holding two of the values, with the space before
    it, and those two values; None when there is none."""
    for a, b in itertools.permutations(values, 2):
        first = rf"[\"'“]?{re.escape(a)}[\"'”]?"
        pattern = (rf"\s*\b(?:between\s+{first}\s+and|from\s+{first}\s+to)\s+[\"'“]?{re.escape(b)}[\"'”]?"
                   r"(?![A-Za-z0-9_])")
        found = list(re.finditer(pattern, text, re.IGNORECASE))
        if len(found) == 1:
            return found[0].start(), found[0].end(), [a, b]
    return None


def cut_with_frames(text: str, values: list[str], lang: str,
                    descriptions: list[str] | None = None) -> Cut | None:
    """text with each value cut together with its frame where it has one (D8): English, a preposition with or
    without an article and with the participle it completes (based on X), or a naming word (named, called,
    titled) before it, when the value ends its noun phrase, the unit after a number, a currency sign before
    it, and "between X and Y" or "from X to Y" whole when both values go; Chinese, the clause that only
    labels it (，表名为X), its label as the parameter's description gives it (实例ID为X的), a naming verb
    (名为X的), a preposition (在X的, 从X, 到X的, 对X的), with a postposition or an enumeration mark after
    them, 为X or X的 (_frames). A frame is taken only when the cut reads fluently (compared with the request
    before the cut); the bare value otherwise. None unless every value occurs exactly once when its turn
    comes."""
    descriptions = list(descriptions or [""] * len(values))
    told = dict(zip(values, descriptions, strict=True))
    original = text
    fluent = True
    pending = list(values)
    if lang == "en" and len(pending) >= 2:
        pair = _pair_span(text, pending)
        if pair is not None:
            start, end, both = pair
            fluent = _site_ok(text, start, end, lang)
            text = _tidy(text[:start] + text[end:])
            pending = [v for v in pending if v not in both]
    for value in pending:
        found = occurrences(value, text)
        if len(found) != 1:
            return None
        text, ok = _cut_span(text, *found[0].span(), lang, told[value])
        fluent = fluent and ok
    return Cut(text, fluent and reads_fluently(text, lang, original))


_CONNECTIVES = frozenset(("and", "or", "with", "of", "in", "at", "to", "for", "per", "by", "the", "a", "an",
                          "和", "与", "及", "或", "的"))
# the words that may join an object's keys and values in prose ("the length is 5 and the width is 3")
_GLUE = _CONNECTIVES | {"is", "are", "was", "were", "be", "equals", "equal", "its", "their", "my", "our",
                        "your"}
_STRUCTURE = "\"'“”‘’「」:：=,，{[ \t"


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return {str(k) for k in value} | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _keys(v)}
    return set()


def _names_key_word(word: str, name: str) -> bool:
    """Whether an English word is the parameter's name or a content word of it, inflected ("dimensions" for
    dimensions)."""
    low = word.casefold()
    stems = {name.casefold()} | content_words(name)
    return any(low == stem or (len(stem) >= 4 and low.startswith(stem)) for stem in stems if stem)


def object_span(text: str, value: dict, name: str = "") -> tuple[int, int] | None:
    """The span of a request that renders a top-level object whole (D7): from its first string or whole-number
    leaf to its last (each found exactly once), widened leftwards over the braces, brackets, quotes, colons,
    commas, keys and joining words (_GLUE: "the length is") of the object and over its parameter's own name
    ("The dimensions are length: 5, ..."; the fix round's review, F3: "The dimensions." was left),
    rightwards over its closing marks, keys and units; None when a leaf is missing or repeated, when the span
    holds any other word (keys, joining words and units aside), or when its brackets do not balance."""
    texts = _cut_texts(value)
    if texts is None:
        return None
    spans = []
    for t in texts:
        found = occurrences(t, text)
        if len(found) != 1:
            return None
        spans.append(found[0].span())
    start, end = min(s for s, _ in spans), max(e for _, e in spans)
    keys = {k.casefold() for k in _keys(value)}
    between = list(text[start:end])
    for s, e in spans:
        between[s - start:e - start] = [" "] * (e - s)
    for word in re.findall(r"[A-Za-z0-9_]+|[一-鿿]+", "".join(between)):
        if word.casefold() not in keys and word.casefold() not in _GLUE and \
                not re.fullmatch(_UNITS_EN, word, re.IGNORECASE) and not _UNIT_ZH.fullmatch(word):
            return None
    while True:  # leftwards over structure, keys and the words joining them
        head = text[:start]
        stripped = head.rstrip(" \t")
        key = re.search(r"[A-Za-z0-9_]+$|[一-鿿]+$", stripped)
        if stripped and stripped[-1] in _STRUCTURE.strip() + "{[":
            start = len(stripped) - 1
        elif key and (key.group(0).casefold() in keys or key.group(0).casefold() in _GLUE
                      or (name and _names_key_word(key.group(0), name))):
            start = key.start()
        else:
            break
    while True:  # rightwards over quotes, the closers the span opened and keys after a value ("sea view")
        balance = sum(text[start:end].count(o) - text[start:end].count(c) for o, c in ("{}", "[]"))
        rest = text[end:]
        stripped = rest.lstrip(" \t")
        skip = len(rest) - len(stripped)
        key = re.match(r"[A-Za-z0-9_]+|[一-鿿]+", stripped)
        quote = stripped[:1] in ("\"", "'", "”", "’", "」") and text[start:end].count(stripped[0]) % 2 == 1
        if quote or (balance > 0 and stripped[:1] in ("}", "]")):
            end += skip + 1
        elif key and (key.group(0).casefold() in keys or re.fullmatch(_UNITS_EN, key.group(0), re.IGNORECASE)
                      or _UNIT_ZH.fullmatch(key.group(0))):
            end += skip + key.end()
        else:
            break
    span = text[start:end]
    if any(span.count(o) != span.count(c) for o, c in ("{}", "[]", "()")):
        return None
    return start, end


def object_left(text: str, value: dict) -> bool:
    """Whether a request still shows a top-level object after its cut (D7): one of its keys, or a content word
    of one, as a word (a boolean leaf rendered as its key's word: "enable both email and SMS notifications"),
    or an unbalanced brace or bracket."""
    for key in _keys(value):
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(key)}(?![A-Za-z0-9_])", text, re.IGNORECASE):
            return True
        for word in content_words(key):
            stem = word if len(word) <= 4 else word[:max(4, len(word) - 2)]  # alerts for alert, emails ...
            if re.search(rf"(?<![A-Za-z]){re.escape(stem)}", text, re.IGNORECASE):
                return True
    return any(text.count(o) != text.count(c) for o, c in ("{}", "[]"))


# ---- builders ------------------------------------------------------------------------------------------


SOURCE_LANG = {"toolace": "en", "glaive_v2": "en", "hermes": "en", "deepexi": "zh", "glaive_zh": "zh"}


def request_lang(source: str, text: str) -> str:
    """The language of a record's request (D4), read from its text (the fix round's review, F7: ToolACE holds
    Chinese requests, deepexi and glaive_zh English ones): outside quotes and code blocks, a CJK character
    counting as a word, zh when they are at least as many as the ASCII words, else en; its source's
    (SOURCE_LANG) only when the text writes neither. Wrong values, follow-ups and topic switches are written
    in it, whatever the prompt's."""
    body = _QUOTED.sub(" ", _FENCED.sub(" ", text))
    cjk, words = len(_CJK.findall(body)), len(re.findall(r"[A-Za-z]+", body))
    if cjk or words:
        return "zh" if cjk >= words else "en"
    return SOURCE_LANG.get(source, "en")


@dataclass(frozen=True)
class Invalid:
    """A wrong value put into a request: the parameter, the text as the request now gives it, the constraint
    it breaks and its category."""

    param: str
    value: str
    constraint: Constraint
    category: str


@dataclass
class Core:
    """What a builder made of one record (or two): the kind, the prompt language, the dataset and the records
    used (the one whose call or request is answered first), the functions to show, the conversation (("user",
    text) and ("assistant", calls) turns) and what the answer must say (the wrong values, the missing names,
    the calls); originals holds the records' requests as they were, before a cut or a change (request_texts).
    given holds the values the latest request gives the function it is answered with, as that request gives
    them (a wrong value as its text), and cut the texts a missing or priority item's request lost: the
    training reasoning reads both (reasoning_text), which no output writes. dress turns it into a prompt and
    an answer."""

    kind: str
    lang: str
    source: str
    records: tuple[str, ...]
    tools: list[dict]
    history: list[tuple[str, object]]
    api: str = ""
    invalid: list[Invalid] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    required: list[str] = field(default_factory=list)
    optional_cut: list[str] = field(default_factory=list)
    nested: str | None = None
    calls: list[Call] = field(default_factory=list)
    packed: bool = False
    variant: dict = field(default_factory=dict)
    originals: tuple[str, ...] = ()
    given: dict = field(default_factory=dict)
    cut: tuple[str, ...] = ()
    drawn: int = -1  # its place in its split's draw, which the build script sets (never written out)


def _function(sample: dict, name: str) -> dict | None:
    return next((t["function"] for t in sample.get("tools") or [] if t["function"]["name"] == name), None)


def _schema(sample: dict, name: str) -> dict:
    function = _function(sample, name)
    parameters = function.get("parameters") if function else None
    return parameters if isinstance(parameters, dict) else {}


_JSON_TYPES = {"integer": "integer", "int": "integer", "long": "integer", "number": "number",
               "float": "number", "double": "number", "boolean": "boolean", "bool": "boolean",
               "string": "string", "str": "string", "array": "array", "list": "array", "object": "object",
               "dict": "object"}
_TYPE_CHECKS = {"integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
                "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
                "boolean": lambda v: isinstance(v, bool), "string": lambda v: isinstance(v, str),
                "array": lambda v: isinstance(v, list), "object": lambda v: isinstance(v, dict)}


def typed(arguments: dict, schema: dict) -> bool:
    """Whether a gold call fits its schema: every key is a listed parameter (when the schema lists any) and
    every value has its parameter's declared JSON type (a parameter without a known type is not checked).
    deepexi holds calls that give "类目的ID" for an integer: a question about the API, not a request."""
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    for key, value in arguments.items():
        if properties and key not in properties:
            return False
        prop = properties.get(key)
        kind = prop.get("type") if isinstance(prop, dict) else None
        if isinstance(kind, list):
            kind = next((k for k in kind if k != "null"), None)
        check = _TYPE_CHECKS.get(_JSON_TYPES.get(kind.lower())) if isinstance(kind, str) else None
        if check is not None and not check(value):
            return False
    return True


LABEL_ENDS = ("ID", "Id", "id", "名称", "名字", "地址", "编号", "号", "号码", "参数", "信息", "标识", "类型",
              "路径", "账号", "内容", "值", "列表", "代码", "时间", "日期", "版本", "描述", "字段", "地域",
              "区域", "配置", "链接", "网址", "文件", "邮箱", "角色", "资源", "权限", "状态")
_LABEL_ENDS = LABEL_ENDS
_LABEL_QUOTES = "\"'“”‘’「」《》"
_ASSIGNED = re.compile(r"\s*(?:为|是|叫|：|:|=)")
_QUESTION = re.compile(r"[？?]\s*$|什么|哪些|哪里|哪个|如何|怎么|怎样|吗|能否|是否")
_INTRODUCED = re.compile(r"(?:为|是|叫|：|:|=)\s*$")


def _subsequence(needle: str, text: str) -> bool:
    rest = iter(text)
    return all(char in rest for char in needle)


def names_parameter(value: object, user: str, description: str) -> bool:
    """Whether a Chinese string value of a gold call names its parameter instead of giving a value, as the
    calls deepexi gives for questions about an API do ("视频URL地址是什么？" called with
    VideoURL="视频URL地址"): a phrase with no digit that ends in a generic noun (_LABEL_ENDS: ID, 名称,
    地址, 参数 ...) and that the request does not quote, where the request uses it as a label (followed by
    为, 是 or a colon) or is a question, or the phrase is spelled out of the parameter's description (its
    characters in order, the particle 的 aside). Such a request gives no value, so it makes no valid call;
    the tests show the cases."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or not _CJK.search(text) or re.search(r"\d", text) or not text.endswith(_LABEL_ENDS):
        return False
    if any(quote + text in user for quote in _LABEL_QUOTES):
        return False
    at = user.find(text)
    if (at >= 0 and _ASSIGNED.match(user, at + len(text))) or _QUESTION.search(user):
        return True
    needle = "".join(text.casefold().split()).replace("的", "")
    return _subsequence(needle, "".join(description.casefold().split()))


def introduced(value: str, user: str) -> bool:
    """Whether the request introduces a value: some occurrence of it stands right after 为, 是, 叫, a colon or
    an equals sign (spaces aside), or inside a pair of quotes."""
    for match in occurrences(value, user):
        start, end = match.span()
        if _INTRODUCED.search(user[:start]):
            return True
        if start > 0 and end < len(user) and _QUOTE_PAIRS.get(user[start - 1]) == user[end]:
            return True
    return False


def label_value(value: object, user: str) -> bool:
    """Whether a Chinese string value of a gold call is a label rather than a value: it ends in a generic
    noun (LABEL_ENDS: 号码, 链接, 地址, 名称, 文件 ...) and the request does not introduce it (introduced), as
    deepexi's calls for questions about an API do ("如何获取上海地域的OSS链接？" called with
    ImageURL="上海地域的OSS链接")."""
    if not isinstance(value, str) or not _CJK.search(value):
        return False
    text = value.strip()
    return text.endswith(LABEL_ENDS) and not introduced(text, user)


def _empty(value: object) -> bool:
    """Whether a value is or holds, at any depth, a null, an empty string, list or object."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return True
    if isinstance(value, (list, dict)):
        children = list(value.values()) if isinstance(value, dict) else value
        return not children or any(_empty(child) for child in children)
    return False


def _leaf_paths(value: object, schema: object, path: tuple = (),
                description: str = "") -> Iterator[tuple[tuple, object, str, dict]]:
    """(the keys to a leaf, list positions aside; the leaf; the nearest description the schema gives on the
    way; the schema of the leaf itself, {} when none) of every leaf of a call's arguments."""
    schema = schema if isinstance(schema, dict) else {}
    if isinstance(value, dict):
        properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
        for key, child in value.items():
            prop = properties.get(key) if isinstance(properties.get(key), dict) else {}
            told = prop.get("description") if isinstance(prop.get("description"), str) else description
            yield from _leaf_paths(child, prop, (*path, key), told)
    elif isinstance(value, list):
        for child in value:
            yield from _leaf_paths(child, schema.get("items"), path, description)
    else:
        yield path, value, description, schema


_DATE_LEAF = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_CODE = re.compile(r"[A-Z]{2,3}")  # a code in capitals, which the request must give in capitals


def _number_forms(value: float) -> list[str]:
    """How a number may be written in a request: as value_text writes it, and with thousands separators."""
    forms = [_number_text(value)]
    if float(value).is_integer() and abs(value) >= 1000:
        forms.append(f"{int(value):,}")
    return forms


_NAME_BEFORE = re.compile(r"(?<![A-Za-z])[A-Z][A-Za-z]*[ -]$")  # a capitalised word right before a number


def _in_name(text: str, at: int) -> bool:
    """Whether the number at text[at:] belongs to a name: it follows a capitalised word that does not open a
    sentence ("the 2026 Formula 1 season", "Windows 10"; the fix round's review, F6)."""
    found = _NAME_BEFORE.search(text, 0, at)
    return found is not None and not _SENTENCE_START.search(text, 0, found.start())


def grounded_leaf(leaf: object, key: str, user: str, exact: str | None = None) -> bool:
    """Whether the request gives a gold leaf value (D3): a string verbatim, in its own case (normalize_text
    of both; the development scorer compares case by case: the fix round's review, F4; an ASCII letter or
    digit at an end of it may not continue into the request, as in occurrences: "IT" is not given by
    "Italy"), or a YYYY-MM-DD date with other separators or without leading zeros; a number in value_text's
    form or with thousands separators, not as part of a name (_in_name: the 1 of "Formula 1"); a boolean when
    a content word of its parameter's name (content_words) starts a word of the request; never a null. exact
    is the request as normalize_text gives it, when known."""
    if isinstance(leaf, bool):
        stems = [w if len(w) <= 4 else w[:max(4, len(w) - 2)] for w in content_words(key)]
        return any(re.search(r"(?<![A-Za-z])" + re.escape(stem), user, re.IGNORECASE) for stem in stems)
    if isinstance(leaf, (int, float)):
        return any(not _in_name(user, found.start()) for form in _number_forms(leaf)
                   for found in occurrences(form, user))
    if isinstance(leaf, str) and _CODE.fullmatch(leaf):  # a code in capitals (IT, US): not the pronoun it
        return bool(re.search(rf"(?<![A-Za-z0-9_]){leaf}(?![A-Za-z0-9_])", user))
    if isinstance(leaf, str):
        text = normalize_text(leaf)
        exact = exact if exact is not None else normalize_text(user)
        if text and _occurrence_pattern(text, False).search(exact):
            return True
        date = _DATE_LEAF.fullmatch(leaf.strip())
        if date:
            y, mo, d = date.groups()
            pattern = rf"(?<!\d){y}[-/.]0?{int(mo)}[-/.]0?{int(d)}(?!\d)"
            return bool(re.search(pattern, user))
    return False


# Values that stand in for a value (hand check, round 1: "xxxxx", "holo-xxxxxx", "example"): a run of three
# x, only stars or question marks, a word in angle, square or curly brackets, or one of PLACEHOLDER_WORDS
_PLACEHOLDER = re.compile(r"x{3,}|^\*+$|^\?+$|^<[^<>]*>$|^\[[^\[\]]*\]$|^\{[^{}]*\}$", re.IGNORECASE)
PLACEHOLDER_WORDS = frozenset(("example", "sample", "string", "placeholder", "value", "示例", "例子", "某某"))
_DEMONSTRATIVE = re.compile(r"(?:这|那|该|此|某)")  # 这段数据, 某个用户: a pointer, not a value
_ID_NAME = re.compile(r"^(?:id|ids)$|_(?:id|ids)$|(?:Id|ID|Ids|IDs)$")  # UserId, user_id, ID; not Valid
_QUOTED = re.compile(r'"([^"\n]{1,60})"|“([^”\n]{1,60})”|「([^」\n]{1,60})」|『([^』\n]{1,60})』'
                     r"|《([^》\n]{1,60})》|(?<![A-Za-z0-9])'([^'\n]{1,60})'(?![A-Za-z0-9])|‘([^’\n]{1,60})’"
                     r"|`([^`\n]{1,60})`")
_FENCED = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_NAMED_AFTER = re.compile(r"\s*(?:function|tool|api|method|endpoint|函数|接口|工具|方法)", re.IGNORECASE)
_NEW_LABEL = re.compile("新的?[一-鿿]{0,4}(?:"
                        + "|".join(map(re.escape, (*LABEL_ENDS, "说明", "描述", "备注"))) + ")")


def placeholder_value(value: object) -> bool:
    """Whether a string value stands in for a value (_PLACEHOLDER, PLACEHOLDER_WORDS), or is a new something
    of a generic kind (新的描述信息, 新的说明信息: _NEW_LABEL; hand check, round 2)."""
    return isinstance(value, str) and (bool(_PLACEHOLDER.search(value.strip()))
                                       or value.strip().casefold() in PLACEHOLDER_WORDS
                                       or bool(_NEW_LABEL.fullmatch(value.strip())))


def _squash(text: str) -> str:
    return re.sub(r"[\s_\-]", "", text).casefold()


_STAND_IN = re.compile(r"(?:my|your|the|sample|example|dummy|test|fake)?(?P<name>.+?)(?:here)?")


def names_own_key(key: str, value: object) -> bool:
    """Whether a string value is its own parameter's name, case, spaces, underscores and hyphens aside, with
    my, your, sample, example ... before it or here after it: a placeholder too (deepexi's UpdateQuotaPlan(
    planName="planName", region="region"), DeleteFolder(FolderId="folder_id"); the fix round's review, F5;
    api_key="my_api_key", its hand check)."""
    if not isinstance(value, str) or not key or not _squash(value):
        return False
    found = _STAND_IN.fullmatch(_squash(value))
    return _squash(value) == _squash(key) or (found is not None and found.group("name") == _squash(key))


# Values that point at the user's own place (hand check, round 2: a restaurant search changed to "near me")
DEICTIC_WORDS = frozenset(("here", "near me", "nearby", "around me", "my location", "current location",
                           "my current location", "my area", "my house", "my place"))


def generic_value(value: object) -> bool:
    """Whether a string value points at something instead of naming it: a Chinese one opening with 这, 那, 该,
    此 or 某 ("这段数据" for the data to scan), or one of DEICTIC_WORDS ("near me" for a location)."""
    if not isinstance(value, str):
        return False
    text = value.strip()
    return (bool(_CJK.search(text)) and bool(_DEMONSTRATIVE.match(text))) or text.casefold() in DEICTIC_WORDS


def _options(schema: dict, description: str) -> set[str]:
    """The options a parameter lists: its schema's enum and the closed list its description gives."""
    options = {str(v) for v in schema.get("enum")} if isinstance(schema.get("enum"), list) else set()
    listed = _description_enum(description) if description else None
    return options | set(listed or ())


def names_own_parameter(value: object, description: str, schema: dict) -> bool:
    """Whether a Chinese string value is words of its own parameter's description, none of its options:
    one of at most four characters (Status="状态" for "把数据源的状态改为'状态'", quoted or
    not), or one ending in a generic noun (LABEL_ENDS) that the description holds whole
    (Description="数据源的描述信息" for a description "数据源的描述信息。"; hand check, round 2)."""
    if not isinstance(value, str) or not _CJK.search(value):
        return False
    text = "".join(_norm(value).split())
    if not text or value.strip() in _options(schema, description):
        return False
    if len(text) > 4 and not value.strip().endswith(LABEL_ENDS):
        return False
    return text in "".join(_norm(description).split())


_ENGLISH_NAME = re.compile(r"[A-Za-z]+(?: [A-Za-z]+)+|[A-Z][a-z]+")  # words, or a capitalised word


def word_identifier(key: str, value: object) -> bool:
    """Whether a parameter named as an identifier (UserId, user_id, ID: _ID_NAME) is given words instead:
    Chinese words alone, no letter or digit of an identifier among them (WorksId="数据门户", RegionId="北京"),
    or an English name, letters and spaces with a space or a capitalised word (movie_id="Inception",
    habitat_id="Savannah region"; hand check, round 4)."""
    if not isinstance(value, str) or not _ID_NAME.search(key):
        return False
    text = value.strip()
    chinese = bool(_CJK.search(text)) and not re.search(r"[A-Za-z0-9]", text)
    return chinese or bool(_ENGLISH_NAME.fullmatch(text))


_ATTRIBUTE_BEFORE = re.compile(r"([一-鿿]{0,8}?)(?:名称|名字|名|编号|ID|Id|id)"
                               r"\s*(?:为|是|叫|[:：])\s*[\"'“「『]?$")
_HEAD_AFTER = re.compile(r"[\"'”」』]?\s*的([一-鿿]{2,6})")


def slot_mismatch(user: str, arguments: dict, schema: dict) -> bool:
    """Whether a value a Chinese request gives as the name or identifier of a thing ("名称为X的业务流程": the
    thing is the word after 的) fills a parameter whose description does not speak of that thing while
    another parameter's does: deepexi's DeleteBusiness(ProjectIdentifier="数据处理") for
    "删除名称为'数据处理'的业务流程", the space's name parameter given the business process's name (hand
    check, round 1). A label that names its own thing first (工作空间名称为X) gives the value to that
    thing."""
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    for key, value in arguments.items():
        text = value_text(value)
        found = occurrences(text, user) if text and text.strip() else []
        if len(found) != 1:
            continue
        before = _ATTRIBUTE_BEFORE.search(user, 0, found[0].start())
        after = _HEAD_AFTER.match(user, found[0].end())
        if before is None or after is None:
            continue
        own = _described(schema, key)
        head, owner = after.group(1), before.group(1)
        if head in own or cjk_terms(head) & cjk_terms(own) or cjk_terms(owner) & cjk_terms(own):
            continue
        if any(head in _described(schema, other) for other in properties if other != key):
            return True
    return False


def _value_texts(leaves: Iterable[object]) -> set[str]:
    """The texts of the string and number leaves, as _norm writes them (numbers as value_text does)."""
    out = set()
    for leaf in leaves:
        text = value_text(leaf)
        if text is not None and text.strip():
            out.add(_norm(text).strip() if isinstance(leaf, str) else text)
    return out


def inside_other_value(user: str, leaves: Iterable[object]) -> bool:
    """Whether a value of the calls occurs in the request, as grounded_leaf reads it (verbatim, in its own
    case), only inside another value's occurrence (objectType "表" taken out of objectName "表C",
    ProjectIdentifier "dataworks" out of FolderPath "/dataworks/folder", format "mp3" out of an audio URL
    beside "MP3 format"): such a value is not given, it is derived."""
    exact = normalize_text(user)
    texts = {normalize_text(text) for leaf in leaves
             if (text := value_text(leaf)) is not None and text.strip()}
    spans = {text: [m.span() for m in _occurrence_pattern(text, False).finditer(exact)] for text in texts}
    for text, own in spans.items():
        if own and all(any(s2 <= s and e <= e2 and e2 - s2 > e - s for other, theirs in spans.items()
                           if other != text for s2, e2 in theirs) for s, e in own):
            return True
    return False


def _held(outer: str, inner: str) -> bool:
    """Whether inner occurs in outer with no letter or digit, of any script, right before or after it."""
    start = outer.find(inner) if inner else -1
    while start >= 0:
        end = start + len(inner)
        if not ((start and outer[start - 1].isalnum()) or (end < len(outer) and outer[end].isalnum())):
            return True
        start = outer.find(inner, start + 1)
    return False


def unused_quote(user: str, calls: list[Call]) -> bool:
    """Whether the request quotes a value no call uses (a gold call that leaves out what was asked: the third
    of three quoted file names, the folder's old name): a span in quotes, brackets 「」 《》 or backticks,
    outside a code block, that is no called function's name, no function named by the word after it
    ("lookup_order" function) and no JSON key (a colon right after it), whose text holds no value of the
    calls and is held by none, apart from marks (_held, values as _norm writes them: "#SummerSuns" holds
    SummerSuns, but "新文件夹" is not held by 新文件夹2; hand check, round 2)."""
    texts = _value_texts(leaf for _, arguments in calls for leaf in _leaves(arguments))
    names = {name.casefold() for name, _ in calls}
    body = _FENCED.sub(" ", user)
    for match in _QUOTED.finditer(body):
        content = next(group for group in match.groups() if group is not None)
        after = body[match.end():]
        if content.casefold() in names or _NAMED_AFTER.match(after) or after[:1] == ":":
            continue
        quoted = _norm(content).strip()
        if quoted and not any(_held(quoted, text) or _held(text, quoted) for text in texts):
            return True
    return False


def call_problem(user: str, calls: list[Call], schemas: list[dict]) -> str | None:
    """Why a request does not fully give its gold calls (D3), the first that applies; None when it does:
    - empty_value: a null, an empty string, list or object anywhere in the arguments;
    - ungrounded: a leaf the request does not give (grounded_leaf: verbatim, in its own case);
    - placeholder_value: a value standing in for one ("xxxxx", "example": placeholder_value), or its own
      parameter's name (names_own_key: planName="planName");
    - boolean_only: calls whose values are booleans alone, which a word of a parameter's name grounds (a
      question about the API: "更新Catalog的配置需要admin角色以上的权限吗？" called with
      adminRoleRequired=true);
    - inside_other_value: a value the request holds only inside another value (inside_other_value);
    - copied_value: a value the request gives k times fills more than k parameters (ProjectId and TenantId
      both 123 for "查询项目123的租户信息"; the same parameter of several calls counts once);
    - label_value: a Chinese label in place of a value (label_value);
    - names_parameter: a value naming its parameter (names_parameter, with the nearest description; or a
      short word of its own description: names_own_parameter);
    - generic_value: a Chinese value pointing at something (generic_value: 这段数据);
    - word_identifier: an identifier parameter given words instead (word_identifier);
    - slot_mismatch: a value the request names as another parameter's thing (slot_mismatch);
    - unused_quote: a value the request quotes that no call uses (unused_quote).
    Every leaf is checked, at any depth. The rules from placeholder_value on, the description words of
    names_parameter and the widened grounding came from the hand check's first round; the case of a string,
    the numbers inside names and the values naming their own parameter from the fix round's review."""
    leaves = [(path, leaf, told, schema) for (_, arguments), function in zip(calls, schemas, strict=True)
              for path, leaf, told, schema in _leaf_paths(arguments, function)]
    if any(_empty(arguments) for _, arguments in calls):
        return "empty_value"
    exact = normalize_text(user)
    if any(not grounded_leaf(leaf, str(path[-1]) if path else "", user, exact)
           for path, leaf, _, _ in leaves):
        return "ungrounded"
    if any(placeholder_value(leaf) or names_own_key(str(path[-1]) if path else "", leaf)
           for path, leaf, _, _ in leaves):
        return "placeholder_value"
    if leaves and all(isinstance(leaf, bool) for _, leaf, _, _ in leaves):
        return "boolean_only"
    if inside_other_value(user, (leaf for _, leaf, _, _ in leaves)):
        return "inside_other_value"
    filled: dict[str, set] = defaultdict(set)
    for path, leaf, _, _ in leaves:
        text = value_text(leaf)
        if text is not None:
            filled[_norm(text)].add(path)
    for text, paths in filled.items():
        if len(paths) > max(1, len(occurrences(text, user))):
            return "copied_value"
    if any(label_value(leaf, user) for _, leaf, _, _ in leaves):
        return "label_value"
    if any(names_parameter(leaf, user, told) or names_own_parameter(leaf, told, schema)
           for _, leaf, told, schema in leaves):
        return "names_parameter"
    if any(generic_value(leaf) for _, leaf, _, _ in leaves):
        return "generic_value"
    if any(word_identifier(str(path[-1]) if path else "", leaf) for path, leaf, _, _ in leaves):
        return "word_identifier"
    if any(slot_mismatch(user, arguments, function)
           for (_, arguments), function in zip(calls, schemas, strict=True) if isinstance(arguments, dict)):
        return "slot_mismatch"
    if unused_quote(user, calls):
        return "unused_quote"
    return None


def gives_values(user: str, arguments: dict, schema: dict) -> bool:
    """Whether the request gives every value of one gold call (call_problem finds nothing)."""
    return call_problem(user, [("", arguments)], [schema]) is None


def bank_value_ok(value: object, user: str, description: str, key: str = "") -> bool:
    """Whether a string value of a gold call may be another request's new value (the multi-turn value banks):
    not empty, given by its own request, no label, naming no parameter, no placeholder (nor the name of its
    parameter, key) and pointing at nothing."""
    return isinstance(value, str) and bool(value.strip()) and grounded_leaf(value, "", user) and \
        not label_value(value, user) and not names_parameter(value, user, description) and \
        not names_own_parameter(value, description, {}) and not placeholder_value(value) and \
        not names_own_key(key, value) and not generic_value(value)


# An option of an enum (case-folded) -> words that name it in other words, Chinese and English: a near miss
# that swaps an option is dropped when its request still names another option in such words (反向 next to
# Forward).
ENUM_WORDS = {
    "forward": ("正向", "向前", "前向"), "reverse": ("反向", "逆向", "backward", "backwards", "reversed"),
    "backward": ("反向", "向后", "backwards"),
    "enable": ("启用", "开启", "enabled"), "enabled": ("启用", "开启"),
    "disable": ("禁用", "停用", "disabled"), "disabled": ("禁用", "停用"),
    "asc": ("升序", "ascending"), "ascending": ("升序",), "desc": ("降序", "descending"),
    "descending": ("降序",),
    "public": ("公开", "公有"), "private": ("私有", "私密"),
    "start": ("开始", "启动"), "stop": ("停止",), "running": ("运行中",), "stopped": ("已停止",),
    "success": ("成功",), "succeeded": ("成功",), "failed": ("失败",), "failure": ("失败",),
    "daily": ("每天", "每日"), "weekly": ("每周",), "monthly": ("每月",), "yearly": ("每年",),
    "hourly": ("每小时",), "male": ("男性",), "female": ("女性",),
    "full": ("全量",), "incremental": ("增量",), "online": ("在线",), "offline": ("离线",),
    "celsius": ("摄氏",), "fahrenheit": ("华氏",), "metric": ("公制",), "imperial": ("英制",),
    "buy": ("买入", "购买"), "sell": ("卖出", "出售"), "allow": ("允许",), "deny": ("拒绝",),
}


def _names_word(word: str, text: str) -> bool:
    """Whether text holds a word: a CJK one anywhere; an ASCII word of four or more letters as the start of a
    word of the text, so that its inflections count (replace in "replaced", remote in "remotely"); any other
    as occurrences finds it."""
    if not word.isascii():
        return word in text
    if len(word) >= 4 and word.isalpha():
        return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(word)}", text, re.IGNORECASE))
    return bool(occurrences(word, text))


def names_other_option(text: str, options: Iterable, chosen: object) -> bool:
    """Whether text names an option of an enum other than the chosen one, as written (case aside, inflected:
    _names_word) or in other words (ENUM_WORDS)."""
    for option in options:
        if str(option) == str(chosen):
            continue
        for word in (str(option), *ENUM_WORDS.get(str(option).casefold(), ())):
            if _names_word(word, text):
                return True
    return False


def single_call(sample: dict, avoid: frozenset = frozenset()) -> tuple[str, str, dict, dict] | None:
    """(user text, function, arguments, schema) of a record whose first step is one call that fits its schema
    (typed) and whose every value the request gives (call_problem finds nothing)."""
    step = first_step(sample)
    if step is None or len(step[1]) != 1:
        return None
    user, [(name, arguments)] = step
    if name.casefold() in avoid or _function(sample, name) is None or not isinstance(arguments, dict):
        return None
    schema = _schema(sample, name)
    if not typed(arguments, schema) or call_problem(user, [(name, arguments)], [schema]) is not None:
        return None
    return user, name, arguments, schema


def _described(schema: dict, key: str) -> str:
    """The description a schema gives a top-level parameter ("" when none)."""
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    prop = properties.get(key)
    return prop["description"] if isinstance(prop, dict) and isinstance(prop.get("description"), str) else ""


def _required(sample: dict, schema: dict, arguments: dict) -> list[str]:
    """The required parameters: the schema's list, or the gold call's keys for a source whose schemas declare
    none (REQUIRED_FROM_CALL)."""
    if sample["source"] in REQUIRED_FROM_CALL:
        return list(arguments)
    required = schema.get("required")
    return [k for k in required if isinstance(k, str)] if isinstance(required, list) else []


def shown_tools(sample: dict, called: list[str], rng: random.Random, *, required: list[str] | None = None,
                avoid: frozenset = frozenset(), extra: list[dict] | None = None) -> list[dict]:
    """The functions a prompt shows: the called ones and up to MAX_DISTRACTORS others of the record, in the
    record's order, none whose name is to be avoided; required replaces the called function's list."""
    tools = [t for t in sample.get("tools") or [] if t["function"]["name"].casefold() not in avoid]
    others = [t for t in tools if t["function"]["name"] not in called]
    keep = rng.sample(range(len(others)), min(len(others), rng.randint(min(1, len(others)),
                                                                        min(len(others), MAX_DISTRACTORS))))
    kept_ids = {id(others[i]) for i in keep}
    out = []
    for t in tools:
        if t["function"]["name"] in called or id(t) in kept_ids:
            t = copy.deepcopy(t)
            if required is not None and t["function"]["name"] in called:
                parameters = t["function"].setdefault("parameters", {"type": "object", "properties": {}})
                parameters["required"] = list(required)
            out.append(t)
    for t in extra or []:
        if t["function"]["name"] not in {o["function"]["name"] for o in out}:
            out.append(copy.deepcopy(t))
    return out


def _scalar(value: object) -> bool:
    return isinstance(value, (str, int, float)) and not isinstance(value, bool)


def _targets(user: str, arguments: dict, schema: dict) -> list[tuple[str, object, Constraint]]:
    """(parameter, value, constraint) of every value that may be changed: a top-level scalar or an item of a
    top-level array, valid under its inferred constraint, its text once in the request."""
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    out = []
    for name, value in arguments.items():
        prop = properties.get(name)
        if not isinstance(prop, dict):
            continue
        if isinstance(value, list):
            c = item_constraint(prop)
            if c is None or not value or not all(_scalar(v) and passes(c, v) for v in value):
                continue
            out += [(name, v, c) for v in value if len(occurrences(value_text(v), user)) == 1]
        elif _scalar(value):
            c = infer_constraint(prop)
            text = value_text(value)
            if c is not None and passes(c, value) and text and len(occurrences(text, user)) == 1:
                out.append((name, value, c))
    return out


def _mutate(text: str, target: tuple[str, object, Constraint], category: str, lang: str,
            rng: random.Random, keep: list[str]) -> tuple[str, Invalid] | None:
    """The request with the target's value replaced by a wrong one of the category (each candidate tried in a
    drawn order until one fits where the value stands (wrong_value_fits), occurs exactly once and the values
    to keep still do; an English article before it agreeing with it: replace_value); None for a value the
    request gives again in other words (gives_again)."""
    name, value, c = target
    if gives_again(text, c, value):
        return None
    candidates = invalid_candidates(c, value, category, lang)
    rng.shuffle(candidates)
    for candidate in candidates:
        if not wrong_value_fits(text, value_text(value), candidate, lang):
            continue
        changed = replace_value(text, value_text(value), candidate)
        if changed is None or len(occurrences(candidate, changed)) != 1:
            continue
        if any(len(occurrences(k, changed)) != 1 for k in keep):
            continue
        return changed, Invalid(name, candidate, c, category)
    return None


def described_format(c: Constraint) -> bool:
    """Whether a constraint's format comes from its parameter's description alone (an e-mail address, a URL, a
    date spec the description writes), no schema keyword naming one."""
    return c.fmt is not None and "schema:format" not in c.sources and \
        any(source.startswith("description:") and source not in ("description:enum", "description:range")
            for source in c.sources)


def build_incorrect(sample: dict, lang: str, rng: random.Random, *, category: str = "obvious",
                    two: bool = False, avoid: frozenset = frozenset(), optional: bool = False,
                    described: bool = False, complete: bool = False) -> Core | None:
    """A request with one (two) of its values replaced by a wrong one of the category, written in the
    request's own language (request_lang), whatever the prompt's; None when the record has no value that can
    take one. Revision 1's training constructions: optional puts the (first) wrong value on a parameter the
    function does not require, described on a value whose format only the description implies
    (described_format); each is recorded in the variant when asked for. complete (every training item,
    revision 1) takes no record whose gold call leaves out a parameter its function requires (_required):
    that request lacks a required value besides the wrong one, a priority case whose answer the prompt's order
    decides (the review of the revision's first build: its record 448, search_movie called without the title
    the user could not recall, answered with the wrong-value sentence under a missing-first prompt)."""
    got = single_call(sample, avoid)
    if got is None:
        return None
    user, name, arguments, schema = got
    if complete and any(key not in arguments for key in _required(sample, schema, arguments)):
        return None
    words = request_lang(sample["source"], user)
    targets = _targets(user, arguments, schema)
    rng.shuffle(targets)
    first = [t for t in targets if invalid_candidates(t[2], t[1], category, words)]
    if optional:
        required = set(_required(sample, schema, arguments))
        first = [t for t in first if t[0] not in required]
    if described:
        first = [t for t in first if described_format(t[2])]
    for target in first:
        done = _mutate(user, target, category, words, rng, [])
        if done is None:
            continue
        text, bad = done
        invalid = [bad]
        if two:
            second = next((m for t in targets if t[0] != target[0]
                           and (m := _mutate(text, t, "obvious", words, rng, [bad.value])) is not None), None)
            if second is None:
                continue
            text, other = second
            invalid.append(other)
        required = list(arguments) if sample["source"] in REQUIRED_FROM_CALL else None
        variant = {"category": category, "values": len(invalid)}
        variant |= {key: True for key, asked in (("optional", optional), ("described", described)) if asked}
        wrong = {bad.param: bad.value for bad in invalid}
        return Core("incorrect", lang, sample["source"], (sample["source_id"],),
                    shown_tools(sample, [name], rng, required=required, avoid=avoid), [("user", text)],
                    api=name, invalid=invalid, required=_required(sample, schema, arguments),
                    variant=variant, originals=(user,),
                    given={key: wrong.get(key, value) for key, value in arguments.items()})
    return None


def _cut_texts(value: object) -> list[str] | None:
    """The texts to cut for a value: a scalar's own text, or every leaf of an object (strings of two or more
    characters and whole numbers); None when the value cannot be cut cleanly."""
    if isinstance(value, dict):
        leaves = [leaf for leaf in _leaves(value) if leaf is not None and not isinstance(leaf, bool)]
        if not leaves:
            return None
        texts = [_cut_texts(leaf) for leaf in leaves]
        return None if any(t is None for t in texts) else [x for t in texts for x in t]
    if isinstance(value, str) and len(value.strip()) >= 2 and value.strip() == value:
        return [value]
    if isinstance(value, int) and not isinstance(value, bool):
        return [str(value)]
    return None


def _cuttable(value: object, user: str) -> list[str] | None:
    texts = _cut_texts(value)
    if texts is None or any(len(occurrences(t, user)) != 1 for t in texts):
        return None
    return texts


MAX_CUT_TRIES = 8  # the choices of values to cut a record is tried with, at most


def _still_given(text: str, arguments: dict, schema: dict) -> bool:
    """Whether a cut request still gives every leaf of the values not cut (grounded_leaf)."""
    return all(grounded_leaf(leaf, str(path[-1]) if path else "", text)
               for path, leaf, _, _ in _leaf_paths(arguments, schema))


def _missing_cut(user: str, arguments: dict, schema: dict, keys: list[str], objects: set[str],
                 words: str) -> Cut | None:
    """The request with the values of keys cut: a top-level object whole (object_span, with its frame; no key,
    key word or bracket of it may stay: object_left), the scalars with their frames (cut_with_frames); None
    when a value cannot be cut so."""
    text, fluent = user, True
    for key in keys:
        if key in objects:
            done = None  # the object's span, then, unless that cut reads fluently, the span with its name
            for span in dict.fromkeys(span for span in (object_span(text, arguments[key]),
                                                        object_span(text, arguments[key], key)) if span):
                cut, ok = _cut_span(text, *span, words, _described(schema, key))
                if not object_left(cut, arguments[key]) and (done is None or ok):
                    done = (cut, ok)
                if done and done[1]:
                    break
            if done is None:
                return None
            text, fluent = done[0], fluent and done[1]
    scalars = [key for key in keys if key not in objects]
    if scalars:
        cut = cut_with_frames(text, [value_text(arguments[k]) for k in scalars], words,
                              [_described(schema, k) for k in scalars])
        if cut is None:
            return None
        text, fluent = cut.text, fluent and cut.fluent
    return Cut(text, fluent and reads_fluently(text, words, user))


_NUMBER_WORDS_EN = ("one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|first|second|third|"
                    r"fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|last|single|couple|dozen|"
                    r"\d+(?:st|nd|rd|th)")


def touches_other_value(text: str, value: str | None, others: Iterable[str | None]) -> bool:
    """Whether an occurrence of value touches an occurrence of another value of the call, nothing but a space
    or a hyphen between them ("a Toyota Corolla", 北京朝阳区): cut, it would leave the rest of one name, which
    can give it away (with Toyota cut, Corolla names the make; hand check, round 1 of the fix round's
    review)."""
    if not value:
        return False
    spans = [m.span() for m in occurrences(value, text)]
    for other in others:
        if not other or other == value:
            continue
        for found in occurrences(other, text):
            for start, end in spans:
                if text[found.end():start] in ("", " ", "-") and found.end() <= start or \
                        text[end:found.start()] in ("", " ", "-") and end <= found.start():
                    return True
    return False


def still_numbered(text: str, key: str, value: object) -> bool:
    """Whether a request a number was cut from still gives it in words: an ordinal or a number word right
    before a content word of its parameter's name, one word between at most ("the third round" with round
    cut, "two tickets" with num_tickets cut; the fix round's review, F6). Only for a number."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    return any(re.search(rf"\b(?:{_NUMBER_WORDS_EN})\s+(?:[A-Za-z]+\s+)?{re.escape(word)}(?:s|es)?\b", text,
                         re.IGNORECASE) for word in content_words(key))


def build_missing(sample: dict, lang: str, rng: random.Random, *, subkind: str = "single",
                  avoid: frozenset = frozenset(), kind: str = "missing",
                  keys: list[str] | None = None) -> Core | None:
    """A request with one or two required values cut (single, two), a required top-level object cut whole
    (object: only its name is expected; D7), or a required value and an optional one cut (optional: the
    optional one is not named); None when the record cannot give one. Each value goes with its frame where
    that reads fluently (cut_with_frames, D8): the choices of values are tried in a drawn order and the first
    fluent cut is taken, else the first bare one; variant["cut"] says which ("frame" or "bare"). keys, when
    given, is the one choice of required values to cut (a missing twin of an incorrect item, revision 1: the
    parameter whose value that item made wrong)."""
    got = single_call(sample, avoid)
    if got is None:
        return None
    user, name, arguments, schema = got
    words = request_lang(sample["source"], user)
    required = _required(sample, schema, arguments)

    def free(key: str) -> bool:  # a scalar that can be cut alone (touches_other_value)
        others = [value_text(arguments[k]) for k in arguments if k != key]
        return _scalar(arguments[key]) and bool(_cuttable(arguments[key], user)) and \
            not touches_other_value(user, value_text(arguments[key]), others)

    spare: list[str] = []
    if subkind == "optional":
        if sample["source"] in REQUIRED_FROM_CALL:
            spare = [k for k in arguments if free(k)]
            if len(arguments) < 2 or not spare:
                return None
            chosen = rng.choice(spare)
            required = [k for k in arguments if k != chosen]
            spare = [chosen]
        else:
            spare = [k for k in arguments if k not in required and free(k)]
    if not required or any(k not in arguments for k in required):
        return None
    scalars = [k for k in required if free(k)]
    objects = [k for k in required if isinstance(arguments[k], dict) and _cuttable(arguments[k], user)]
    if subkind == "object":
        choices = [([k], []) for k in objects]
    elif subkind == "two":
        choices = [(list(pair), []) for pair in itertools.combinations(scalars, 2)]
    elif subkind == "optional":
        choices = [([k], [o]) for k in scalars for o in spare if o != k]
    else:
        choices = [([k], []) for k in scalars]
    rng.shuffle(choices)
    if keys is not None:
        choices = [(missing, cut) for missing, cut in choices if sorted(missing) == sorted(keys)]
    override = required if sample["source"] in REQUIRED_FROM_CALL else None
    bare = None
    for missing, cut in choices[:MAX_CUT_TRIES]:
        done = _missing_cut(user, arguments, schema, missing + cut, set(objects), words)
        if done is None or len(done.text) < MIN_REST:
            continue
        texts = [t for k in missing + cut for t in _cut_texts(arguments[k]) or []]
        if any(_norm(t) in _norm(done.text) for t in texts):
            continue
        if any(still_numbered(done.text, k, arguments[k]) for k in missing):
            continue
        if not _still_given(done.text, {k: arguments[k] for k in required if k not in missing}, schema):
            continue
        ordered = [k for k in required if k in missing]
        core = Core(kind, lang, sample["source"], (sample["source_id"],),
                    shown_tools(sample, [name], rng, required=override, avoid=avoid), [("user", done.text)],
                    api=name, missing=ordered, required=required, optional_cut=cut,
                    nested=ordered[0] if subkind == "object" else None,
                    variant={"subkind": subkind, "cut": "frame" if done.fluent else "bare"},
                    originals=(user,), given={k: v for k, v in arguments.items() if k not in missing + cut},
                    cut=tuple(texts))
        if done.fluent:
            return core
        bare = bare or core
    return bare


def build_ask(sample: dict, lang: str, rng: random.Random, *, avoid: frozenset = frozenset()) -> Core | None:
    """A missing-value request whose prompt (dress) says to ask through an ask function."""
    return build_missing(sample, lang, rng, subkind="single", avoid=avoid, kind="ask")


def build_priority(sample: dict, lang: str, rng: random.Random, *,
                   avoid: frozenset = frozenset()) -> Core | None:
    """A request with one wrong value (in the request's language) and one required value cut with its frame;
    the prompt's order decides the answer. The first fluent cut is taken (cut_with_frames), else the first
    bare one; variant["cut"] says which."""
    got = single_call(sample, avoid)
    if got is None:
        return None
    user, name, arguments, schema = got
    words = request_lang(sample["source"], user)
    required = _required(sample, schema, arguments)
    if not required or any(k not in arguments for k in required):
        return None
    targets = _targets(user, arguments, schema)
    rng.shuffle(targets)
    override = required if sample["source"] in REQUIRED_FROM_CALL else None
    bare = None
    def alone(key: str) -> bool:  # a value whose cut leaves no part of a name (touches_other_value)
        return not touches_other_value(user, value_text(arguments[key]),
                                       [value_text(arguments[o]) for o in arguments if o != key])

    for target in targets:
        cuttable = [k for k in required if k != target[0] and _scalar(arguments[k])
                    and _cuttable(arguments[k], user) and alone(k)]
        if not cuttable:
            continue
        categories = [c for c in CATEGORIES if invalid_candidates(target[2], target[1], c, words)]
        done = _mutate(user, target, rng.choice(categories), words, rng, []) if categories else None
        if done is None:
            continue
        text, bad = done
        rng.shuffle(cuttable)
        for gap in cuttable:
            texts = _cuttable(arguments[gap], text)
            cut = cut_with_frames(text, texts, words, [_described(schema, gap)]) if texts else None
            if cut is None or len(cut.text) < MIN_REST or len(occurrences(bad.value, cut.text)) != 1:
                continue
            others = {k: arguments[k] for k in required if k not in (gap, target[0])}
            if not _still_given(cut.text, others, schema) or any(_norm(t) in _norm(cut.text) for t in texts):
                continue
            if still_numbered(cut.text, gap, arguments[gap]):
                continue
            tools = shown_tools(sample, [name], rng, required=override, avoid=avoid)
            given = {k: bad.value if k == bad.param else v for k, v in arguments.items() if k != gap}
            core = Core("priority", lang, sample["source"], (sample["source_id"],), tools,
                        [("user", cut.text)], api=name, invalid=[bad], missing=[gap], required=required,
                        variant={"category": bad.category, "cut": "frame" if cut.fluent else "bare"},
                        originals=(user,), given=given, cut=tuple(texts))
            if cut.fluent:
                return core
            bare = bare or core
    return bare


_GENERIC = frozenset((
    "get", "list", "describe", "create", "update", "delete", "query", "add", "remove", "modify", "set",
    "check", "search", "batch", "start", "stop", "run", "put", "post", "fetch", "find", "retrieve", "by",
    "of", "for", "the", "and", "info", "information", "detail", "details", "data", "all", "new", "api", "ids",
    "id", "open", "close", "enable", "disable", "count", "show", "view", "make", "do", "calculate",
    "generate", "convert", "send", "read", "write", "save", "load", "to", "from", "with"))


def content_words(text: str) -> set[str]:
    """The content words of a function name (CamelCase, snake_case or spaced), lower case, a plural s cut,
    without the generic verbs and nouns (_GENERIC) and words under three letters."""
    words = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\b|_)|[A-Z]?[a-z]+|\d+", text)
    out = set()
    for word in (w.lower() for w in words):
        word = word[:-1] if len(word) > 3 and word.endswith("s") else word
        if len(word) >= 3 and word not in _GENERIC:
            out.add(word)
    return out


_FUNCTION_CHARS = frozenset("的了是在和与或为个一我你他她它们请要把将对从到这那有中上下及"
                            "等并其该此所以被给让吗呢吧也都就还而且但如若则即每各可能会")
_GENERIC_TERMS = frozenset((
    "查询", "获取", "调用", "信息", "列表", "数据", "帮我", "请问", "可以", "需要", "用户",
    "指定", "参数", "返回", "设置", "进行", "通过", "使用", "名称", "名字", "类型", "时间",
    "日期", "状态", "当前", "所有", "哪些", "什么", "是否", "以及", "对应", "相关", "详细",
    "请求", "接口", "服务", "默认", "取值", "列出", "全部", "查看", "想要", "条件", "结果",
    "内容", "描述", "说明", "示例", "格式", "字段", "页码", "分页", "每页", "数量", "个数",
    "大小", "开始", "结束", "创建", "删除", "修改", "更新", "添加", "新建", "编号", "标识",
    "唯一"))


def cjk_terms(text: str) -> set[str]:
    """The Chinese two-character terms of a text: adjacent CJK characters, neither a function character
    (_FUNCTION_CHARS), and not a generic term of requests and API documents (_GENERIC_TERMS)."""
    terms = set()
    for run in re.findall(r"[一-鿿]+", text):
        for a, b in itertools.pairwise(run):
            if a not in _FUNCTION_CHARS and b not in _FUNCTION_CHARS and a + b not in _GENERIC_TERMS:
                terms.add(a + b)
    return terms


def unrelated(user: str, called: list[str], tools: list[dict]) -> bool:
    """Whether functions taken from another record cannot serve a request that called called: none is
    called, no content word of theirs is in a called name or in the request, no content word of a called
    name is in their names or descriptions, and their descriptions share no Chinese term with the request
    (cjk_terms)."""
    mine = set().union(*(content_words(n) for n in called)) if called else set()
    text = _norm(user)
    terms = cjk_terms(user)
    for t in tools:
        function = t["function"]
        if function["name"] in called:
            return False
        theirs = content_words(function["name"])
        if theirs & mine or any(w in text for w in theirs):
            return False
        described = _norm(" ".join(descriptions(function)) + " " + function["name"])
        if any(w in described for w in mine) or terms & cjk_terms(described):
            return False
    return True


def _content(text: str) -> set[str]:
    return {word for word in content_words(text) if word not in _STOP_EN}


# the words of a function's name that make it serve nearly any request (a news, web or encyclopedia search, a
# question-answering or chat function): never shown in a pairing ("Search News" beside a request on Formula 1
# laps; hand check, round 7 of the fix round's review)
_UNIVERSAL_EN = frozenset(("news", "article", "articles", "web", "internet", "google", "bing", "wiki",
                           "wikipedia", "encyclopedia", "knowledge", "question", "questions", "answer",
                           "answers", "chat", "chatbot", "gpt", "llm", "assistant"))
_NAME_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|\b|_)|[A-Z]?[a-z]+")


def serves_anything(tools: list[dict]) -> bool:
    """Whether a function shown could serve nearly any request (_UNIVERSAL_EN among its name's words)."""
    return any({w.lower() for w in _NAME_WORD.findall(t["function"]["name"])} & _UNIVERSAL_EN for t in tools)


def shares_content_word(user: str, tools: list[dict]) -> bool:
    """Whether a request shares a content word (content_words, _STOP_EN aside: an equivalent of
    scripts/replay_base.py's irrelevance filters for a pairing) with the names, the parameter names or the
    descriptions of functions (D2: an English request paired with another record's functions must share
    none)."""
    mine = _content(user)
    for t in tools:
        function = t["function"]
        parameters = function.get("parameters") if isinstance(function.get("parameters"), dict) else {}
        texts = [function["name"], *descriptions(function), *_keys(parameters.get("properties") or {})]
        if any(mine & _content(text) for text in texts):
            return True
    return False


# A request to book a flight, order a pizza or book something else: the records' own refusals are nearly all
# such (glaive_v2's refusal records of the seed-1102 pools ask 195 distinct requests, nearly all to book a
# flight or to order a pizza; 140 of the first special-mix's 156 own refusals), which revision 1 caps.
_CANNED = re.compile(
    r"\b(?:flights?|airlines?|pizzas?|book(?:ing|ed)|reserv(?:e|es|ed|ing|ations?))\b"
    r"|\bbooks?\s+(?:me|us|a|an|my|our|the|two|three|four|tickets?|flights?|rooms?|tables?|hotels?|seats?|for)\b"
    r"|机票|航班|披萨|比萨|订票|订餐|订房|预订|预定|订一张|订一间|订个", re.IGNORECASE)


def canned_request(text: str) -> bool:
    """Whether a request asks to book a flight, order a pizza or book something (_CANNED; "a book about
    whales" is no such request): scripts/build_special_mix.py caps the training irrelevant items that are a
    record's own refusal of such a request (revision 1)."""
    return bool(_CANNED.search(text))


def domain_words(tools: list[dict]) -> set[str]:
    """The domain words of functions: the content words of their names and descriptions (content_words, the
    English stop words aside) and the Chinese terms of their descriptions (cjk_terms). Two functions sharing
    some are of one domain; scripts/build_special_mix.py tries the partners of a training pairing whose
    functions share most with the record's own functions first (near-domain partners, revision 1), each still
    unrelated to the request (unrelated, shares_content_word)."""
    out: set[str] = set()
    for t in tools:
        function = t["function"]
        for text in (function["name"], *descriptions(function)):
            out |= _content(text) | cjk_terms(text)
    return out


def build_irrelevant(sample: dict, lang: str, rng: random.Random, *, tools_from: dict | None = None,
                     avoid: frozenset = frozenset(), user: str | None = None) -> Core | None:
    """A request no listed function serves: a record whose first answer says so (ToolACE's "no function"
    answers, glaive's refusals that do not ask for anything), or, with tools_from, a request that called a
    function paired with another record's unrelated functions (unrelated; an English request may share no
    content word with the functions shown either: shares_content_word; and none of them could serve nearly
    any request: serves_anything). user replaces a pairing's request (an irrelevant twin, revision 1: the
    record's request as its missing item cut it), which the same checks then read; the record's own request
    stays its original (Core.originals)."""
    messages = sample["messages"]
    if not messages or messages[0]["role"] != "user" or not messages[0]["content"].strip():
        return None
    original = messages[0]["content"]
    user = original if user is None or tools_from is None else user
    if tools_from is None:
        if not sample.get("tools") or any(m.get("tool_calls") for m in messages):
            return None
        first = next((m for m in messages if m["role"] == "assistant"), None)
        if first is None:
            return None
        text = first["content"]
        if sample["source"] == "toolace":
            if not is_irrelevant_gold(text):
                return None
        elif not _REFUSES.search(text) or _ASKS.search(text) or _ASKS_ZH.search(text):
            return None
        tools = shown_tools(sample, [], rng, avoid=avoid)
        tools = tools[:MAX_DISTRACTORS + 1]
        records = (sample["source_id"],)
    else:
        step = first_step(sample)
        if step is None:
            return None
        called = [n for n, _ in step[1]]
        candidates = [t for t in tools_from.get("tools") or []
                      if t["function"]["name"].casefold() not in avoid]
        if not candidates or not unrelated(user, called, candidates):
            return None
        tools = shown_tools({**tools_from, "tools": candidates}, [], rng)[:MAX_DISTRACTORS + 1]
        if request_lang(sample["source"], user) == "en" and shares_content_word(user, tools):
            return None
        if serves_anything(tools):
            return None
        records = (sample["source_id"], tools_from["source_id"])
    if not tools:
        return None
    return Core("irrelevant", lang, sample["source"], records, tools, [("user", user)],
                variant={"pairing": tools_from is not None}, originals=(original,))


def build_verbatim(sample: dict, lang: str, rng: random.Random, *, require_cjk: bool = False,
                   avoid: frozenset = frozenset()) -> Core | None:
    """A request answered with its gold calls, every value copied from the request (none naming its parameter:
    gives_values); require_cjk asks for a Chinese value among them."""
    step = first_step(sample)
    if step is None or len(step[1]) > 3:
        return None
    user, calls = step
    names = [n for n, _ in calls]
    if any(n.casefold() in avoid or _function(sample, n) is None for n in names):
        return None
    if not all(isinstance(a, dict) and typed(a, _schema(sample, n)) for n, a in calls) or \
            call_problem(user, calls, [_schema(sample, n) for n in names]) is not None:
        return None
    if require_cjk and not any(isinstance(leaf, str) and _CJK.search(leaf)
                               for _, a in calls for leaf in _leaves(a)):
        return None
    required = list(calls[0][1]) if sample["source"] in REQUIRED_FROM_CALL and len(calls) == 1 else None
    return Core("valid_verbatim", lang, sample["source"], (sample["source_id"],),
                shown_tools(sample, names, rng, required=required, avoid=avoid), [("user", user)],
                api=names[0], calls=[(n, a) for n, a in calls], variant={"calls": len(calls)},
                originals=(user,), given=dict(calls[0][1]))


MAX_CHANGED = 40  # characters of a value a follow-up changes, at most (as of a bank value)


def build_change(sample: dict, lang: str, rng: random.Random, *, bank: dict, packed: bool,
                 avoid: frozenset = frozenset()) -> Core | None:
    """Two user turns: the request, the call made for it, then a turn in the request's language that changes
    one value; the answer is the call with the latest value, which the two turns together must give fully
    (call_problem). bank holds the string values the records of the same function give each parameter. A
    date or a time beside another of its kind in the call is not changed (other_of_kind: a range would
    break), nor a value longer than MAX_CHANGED characters (a sentence changed in a follow-up reads as one
    run-on sentence; hand check, round 1 of the fix round's review)."""
    got = single_call(sample, avoid)
    if got is None:
        return None
    user, name, arguments, schema = got
    words = request_lang(sample["source"], user)
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    params = [k for k, v in arguments.items() if isinstance(v, (str, int)) and not isinstance(v, bool)
              and value_text(v) and len(occurrences(value_text(v), user)) == 1
              and not any(other_of_kind(arguments, k, kind) for kind in _kinds(value_text(v)))]
    rng.shuffle(params)
    for param in params:
        old = arguments[param]
        c = infer_constraint(properties.get(param) or {})
        if c is not None and not passes(c, old) or len(value_text(old)) > MAX_CHANGED:
            continue
        new = other_value(c, old, bank.get(param, []), rng)
        old_text, new_text = value_text(old), value_text(new) if new is not None else None
        if new_text is None or occurrences(new_text, user) or _norm(old_text) in _norm(new_text) \
                or _norm(new_text) in _norm(old_text):
            continue
        follow = rng.choice(FOLLOW_UPS[words]).format(old=old_text, new=new_text)
        latest = {**arguments, param: new}
        if call_problem(f"{user}\n{follow}", [(name, latest)], [schema]) is not None:
            continue
        required = list(arguments) if sample["source"] in REQUIRED_FROM_CALL else None
        return Core("valid_multiturn", lang, sample["source"], (sample["source_id"],),
                    shown_tools(sample, [name], rng, required=required, avoid=avoid),
                    [("user", user), ("assistant", [(name, arguments)]), ("user", follow)], api=name,
                    calls=[(name, latest)], packed=packed,
                    variant={"mode": "change", "param": param, "old": old, "new": new, "packed": packed},
                    originals=(user,), given=dict(latest))
    return None


def build_switch(first: dict, second: dict, lang: str, rng: random.Random, *, packed: bool,
                 avoid: frozenset = frozenset()) -> Core | None:
    """Two user turns from two records: the first request and its call, then the second request; the answer
    is the second record's call. The prompt shows the functions of both."""
    one, two = single_call(first, avoid), single_call(second, avoid)
    if one is None or two is None or one[1] == two[1]:
        return None
    tools = shown_tools(second, [two[1]], rng, avoid=avoid)
    earlier = shown_tools(first, [one[1]], rng, avoid=avoid)
    earlier = [t for t in earlier if t["function"]["name"] == one[1]] + \
        [t for t in earlier if t["function"]["name"] != one[1]][:1]
    for t in earlier:
        same = next((o for o in tools if o["function"]["name"] == t["function"]["name"]), None)
        if same is None:
            tools.append(t)
        elif same != t:
            return None
    if not any(t["function"]["name"] == one[1] for t in tools):
        return None
    follow = rng.choice(SWITCH_PREFIXES[request_lang(second["source"], two[0])]) + two[0]
    return Core("valid_multiturn", lang, second["source"], (second["source_id"], first["source_id"]), tools,
                [("user", one[0]), ("assistant", [(one[1], one[2])]), ("user", follow)], api=two[1],
                calls=[(two[1], two[2])], packed=packed, variant={"mode": "switch", "first": one[1],
                                                                  "packed": packed},
                originals=(one[0], two[0]), given=dict(two[2]))


CONTEXT_KINDS = ("incorrect", "missing", "irrelevant", "priority")  # the refusal kinds put in context


def _parameter_names(tools: list[dict], name: str) -> set[str]:
    function = next((t["function"] for t in tools if t["function"]["name"] == name), None)
    parameters = function.get("parameters") if function else None
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    return set(properties) if isinstance(properties, dict) else set()


def with_context(core: Core, first: dict, rng: random.Random, *, packed: bool,
                 avoid: frozenset = frozenset()) -> Core | None:
    """A refusal core (CONTEXT_KINDS) answering the latest of several turns (revision 1: FT5 called on 2 of
    dev set A's 4 incorrect or missing items that have a history, and no refusal item had real turns): first's
    request and the one call made for it (single_call) come before the core's request, as real turns or packed
    into one message, and its function is shown at a drawn place among the core's. The problem stays in the
    latest turn: None when the earlier function is shown already, shares a parameter name with the function
    the core answers with, or could serve the latest request (unrelated; for an English request
    shares_content_word, and serves_anything; for every kind since the review of revision 1, the irrelevant
    cores' alone before), or when the earlier request holds a value the core's request lost. The earlier
    record joins the core's records and originals; variant["context"] says "turns" or "packed"."""
    if core.kind not in CONTEXT_KINDS or len(core.history) != 1:
        return None
    one = single_call(first, avoid)
    if one is None:
        return None
    user, name, arguments, _ = one
    function = _function(first, name)
    shown = {t["function"]["name"] for t in core.tools}
    if name in shown or (core.api and set(_schema(first, name).get("properties") or {})
                         & _parameter_names(core.tools, core.api)):
        return None
    tool = {"type": "function", "function": copy.deepcopy(function)}
    latest = core.history[-1][1]
    if not unrelated(latest, [], [tool]) or serves_anything([tool]) or (
            request_lang(core.source, latest) == "en" and shares_content_word(latest, [tool])):
        return None
    if any(occurrences(text, user) for text in core.cut):
        return None
    tools = list(core.tools)
    tools.insert(rng.randrange(len(tools) + 1), tool)
    return replace(
        core, records=(*core.records, first["source_id"]), tools=tools,
        history=[("user", user), ("assistant", [(name, arguments)]), *core.history], packed=packed,
        variant={**core.variant, "context": "packed" if packed else "turns", "earlier": name},
        originals=(*core.originals, user))


_ARTICLE = re.compile(r"(?<![A-Za-z])(a|an|A|An)([ \t]+)$")
_TIME_TEXT = re.compile(r"(?<!\d)\d{1,2}:\d{2}(?!\d)")
_MERIDIEM_AFTER = re.compile(r"\s*[AaPp]\.?\s?[Mm]\.?(?![A-Za-z])")
_DAYTIME_BEFORE = re.compile(r"(?:上午|下午|早上|晚上|中午|凌晨|傍晚)\s*$")
_DATE_TEXTS = (re.compile(r"(?<!\d)(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)"),
               re.compile(r"(?<!\d)(\d{4})年(\d{1,2})月(\d{1,2})"),
               re.compile(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)"))


def agree_article(text: str, start: int) -> str:
    """text with the English article right before position start (a or an, spaces between) made to agree
    with the word starting there: an before a vowel letter, a before any other letter; the case of its first
    letter kept. Unchanged when no such article stands there or no ASCII letter starts there."""
    found = _ARTICLE.search(text, 0, start)
    head = text[start:start + 1]
    if found is None or not (head.isascii() and head.isalpha()):
        return text
    want = "an" if head.lower() in "aeiou" else "a"
    if found.group(1)[0].isupper():
        want = want.capitalize()
    return text[:found.start(1)] + want + text[found.end(1):]


def replace_value(text: str, old: str, new: str) -> str | None:
    """replace_once, with an English article before the new value made to agree with it (agree_article)."""
    found = occurrences(old, text)
    if len(found) != 1:
        return None
    start, end = found[0].span()
    return agree_article(text[:start] + new + text[end:], start)


def _kinds(text: object) -> set[str]:
    """"date" when a text holds a real date (YYYY-MM-DD with any separator, YYYY年M月D, YYYYMMDD), "time"
    when it holds a clock time (H:MM)."""
    if not isinstance(text, str):
        return set()
    kinds = {"time"} if _TIME_TEXT.search(text) else set()
    for pattern in _DATE_TEXTS:
        for match in pattern.finditer(text):
            if _valid({"y": match.group(1), "mo": match.group(2), "d": match.group(3)}) and \
                    1900 <= int(match.group(1)) <= 2100:
                return kinds | {"date"}
    return kinds


def other_of_kind(arguments: dict, param: str, kind: str) -> bool:
    """Whether a call holds a date (kind "date") or a time ("time") in a parameter other than param: a near
    miss or a change moving one of two dates can break the range they make (an end before the start)."""
    return any(kind in _kinds(value_text(leaf)) for key, value in arguments.items() if key != param
               for leaf in _leaves(value))


def restates_date(text: str, parts: dict) -> bool:
    """Whether text gives a date in other words: its month's name (or the first three letters) beside its
    day ("April 30th", "30 of April"), or 4月30日 / 4月30号."""
    mo, d = parts["mo"], parts["d"]
    month = rf"(?:{_MONTHS[mo - 1]}|{_MONTHS[mo - 1][:3]})\.?"
    day = rf"0?{d}(?:st|nd|rd|th)?"
    english = re.search(rf"(?<![A-Za-z]){month}\s+{day}(?!\d)"
                        rf"|(?<![\d.:]){day}\s+(?:of\s+)?{month}(?![A-Za-z])", text, re.IGNORECASE)
    return bool(english or re.search(rf"(?<!\d)0?{mo}月0?{d}(?:日|号)", text))


def restates_time(text: str, parts: dict) -> bool:
    """Whether text gives a clock time in other words: its hour with AM or PM ("10 AM", "10:30 pm"), or with
    点 (10点)."""
    h12 = parts["h"] % 12 or 12
    english = re.search(rf"(?<![\d:]){h12}(?::{parts['mi']:02d})?\s*[AaPp]\.?\s?[Mm]\.?(?![A-Za-z])", text)
    return bool(english or re.search(rf"(?<!\d)(?:{parts['h']}|{h12})点", text))


_NUMBER_NAMES_EN = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                    "eleven", "twelve")
_ONE_EN = re.compile(r"(?<!this )(?<!that )(?<!which )(?<!the )(?<!no )(?<!any )(?<!each )(?<!every )\bone\b"
                     r"(?!\s+(?:of|more|another|by|thing)\b)", re.IGNORECASE)


def gives_again(text: str, c: Constraint, value: object) -> bool:
    """Whether a request gives a value again in other words: a date its month's name beside its day, a time
    with AM, PM or 点 (restates_date, restates_time), a whole number from 2 to 12 as its English word. A
    wrong value put in its place would leave the other to be read ("25/04/2023" beside "April 25th", 5.5
    beside "five team members"; hand check, round 7 of the fix round's review)."""
    parts = date_parts(c.fmt, value) if is_date_format(c.fmt) else None
    if parts is not None:
        return ("mo" in parts and restates_date(text, parts)) or ("h" in parts and restates_time(text, parts))
    if isinstance(value, int) and not isinstance(value, bool) and 2 <= value <= 12:
        return bool(re.search(rf"\b{_NUMBER_NAMES_EN[value]}\b", text, re.IGNORECASE))
    if value == 1 and not isinstance(value, bool):  # "for one adult", not "one of them" or "this one"
        return bool(_ONE_EN.search(text))
    return False


def _nearmiss_blocked(user: str, arguments: dict, param: str, value: object, new: object,
                      reason: str) -> bool:
    """Whether moving a value to the near miss would break its request (hand check, rounds 1 and 2): a date
    or a time beside another of its kind in the call (other_of_kind); a time with AM or PM after it or
    上午, 下午 ... before it; a number turning from 1 to more or back before an English word (1 qubits,
    64 qubit)."""
    if reason in ("leap", "time") and other_of_kind(arguments, param, "date" if reason == "leap" else "time"):
        return True
    at = occurrences(value_text(value), user)[0]
    if reason == "time" and (_MERIDIEM_AFTER.match(user, at.end())
                             or _DAYTIME_BEFORE.search(user[:at.start()])):
        return True
    return _scalar(value) and not isinstance(value, str) and _scalar(new) and not isinstance(new, str) and \
        (value == 1) != (new == 1) and bool(re.match(r"\s+[A-Za-z]", user[at.end():]))


def _restated(changed: str, c: Constraint, value: object, reason: str) -> bool:
    """Whether a request a near miss changed still gives the value it replaced, in other words."""
    parts = date_parts(c.fmt, value) if is_date_format(c.fmt) else None
    if parts is None:
        return False
    if reason == "leap":
        return restates_date(changed, parts)
    return reason == "time" and restates_time(changed, parts)


def build_nearmiss(sample: dict, lang: str, rng: random.Random, *, avoid: frozenset = frozenset(),
                   why: Counter | None = None) -> Core | None:
    """A request with a valid value close to a bound (nearmiss_value); the answer is the call with it, which
    the request must give fully (call_problem). A date, a time or a number is moved to the leap day, the end
    of the day, the maximum or the minimum, unless that would break the request (_nearmiss_blocked) or the
    request still gives the old value in other words (_restated: "April 30th" beside a check-in moved to
    2024-02-29); an English article before the new value agrees with it (replace_value). An option of an
    enum stays as the request gives it, which must spell it as listed (an exact option); a record whose
    request names another option of the enum too, as written, inflected or in other words
    (names_other_option: 反向 for Reverse beside Forward, "appended" beside replace), is dropped, counted in
    why as enum_option_named."""
    got = single_call(sample, avoid)
    if got is None:
        return None
    user, name, arguments, schema = got
    targets = [t for t in _targets(user, arguments, schema) if not isinstance(arguments[t[0]], list)]
    rng.shuffle(targets)
    for param, value, c in targets:
        found = nearmiss_value(c, value)
        if found is None:
            continue
        new, reason = found
        new_text = value_text(new)
        if reason == "enum":
            if occurrences(new_text, user)[0].group(0) != new_text:  # an exact option is spelled as listed
                continue
            if names_other_option(user, c.enum, new):
                if why is not None:
                    why["enum_option_named"] += 1
                return None
            changed = user
        else:
            if occurrences(new_text, user) or _nearmiss_blocked(user, arguments, param, value, new, reason):
                continue
            changed = replace_value(user, value_text(value), new_text)
            if changed is None or len(occurrences(new_text, changed)) != 1 or \
                    _restated(changed, c, value, reason):
                continue
        latest = {**arguments, param: new}
        if call_problem(changed, [(name, latest)], [schema]) is not None:
            continue
        required = list(arguments) if sample["source"] in REQUIRED_FROM_CALL else None
        return Core("valid_nearmiss", lang, sample["source"], (sample["source_id"],),
                    shown_tools(sample, [name], rng, required=required, avoid=avoid), [("user", changed)],
                    api=name, calls=[(name, latest)],
                    variant={"param": param, "value": new, "reason": reason, "constraint": c},
                    originals=(user,), given=dict(latest))
    return None


# ---- dressing: prompt, answer, expectation, reasoning --------------------------------------------------


@dataclass(frozen=True)
class Item:
    """A dressed core: the system prompt, the messages before the answer (system first), the answer, the
    expectation (the development sets' "expected"), the reasoning, the family field, the prompt's choices, the
    functions shown and the assistant turns before the answer (context_turns)."""

    core: Core
    lang: str
    system: str
    messages: list
    answer: str
    expected: dict
    reasoning: str
    family: str
    call_format: str
    list_style: str
    rules: Rules | None
    ask: AskFunction | None
    tools: list
    context_turns: int


def request_texts(item: Item) -> tuple[str, ...]:
    """What the near-duplicate checks compare of an item (D1): its final request, every user turn of the
    prompt joined (a packed history is one message), then the original request of each record it used, before
    any cut or change; each text once."""
    final = "\n".join(m["content"] for m in item.messages if m["role"] == "user")
    return tuple(dict.fromkeys([final, *item.core.originals]))


def _python_ready(tools: list[dict], calls: list[Call]) -> bool:
    names = [t["function"]["name"] for t in tools] + [n for n, _ in calls]
    keys = [k for _, a in calls for k in a]
    return all(_rfd._python_name(n) for n in names) and all(k.isidentifier() and not keyword.iskeyword(k)
                                                              for k in keys)


def _all_calls(core: Core) -> list[Call]:
    return core.calls + [c for role, turn in core.history if role == "assistant" for c in turn]


def _verbatim_values(calls: list[Call]) -> list[str]:
    return list(dict.fromkeys(leaf for _, a in calls for leaf in _leaves(a)
                              if isinstance(leaf, str) and not leaf.isascii()))


_LETTERS = re.compile(r"[^\W_]+")  # runs of letters and digits, in any script


def _letters(text: str) -> str:
    return "".join(_LETTERS.findall(text)).casefold()


def reads_as_placeholder(item: Item) -> bool:
    """Whether a refusal item's sentence is filled with a value or a name written like a placeholder of its
    prompt's style (the same letters and digits, case aside): such an answer cannot be told from one that left
    the placeholder, so the scorer refuses the item (research_agent.eval.special_dev.check_item), and a
    training answer of the kind would teach to copy the placeholder."""
    expected = item.expected
    if expected["type"] != "refusal" or item.rules is None:
        return False
    marks = {_letters(mark) for mark in placeholders(item.rules.style, item.lang).values()}
    fillers = [*expected["values"], *expected["params"], *([expected["api"]] if expected["api"] else [])]
    return any(_letters(filler) in marks for filler in fillers)


def expected_sentence(item: Item) -> str:
    """The sentence a refusal item's expectation asks for, filled in from it."""
    expected = item.expected
    family = item.rules.families[expected["case"]]
    if expected["case"] == "incorrect":
        return fill_sentence(family, item.rules.style, item.lang, value=", ".join(expected["values"]),
                             param=", ".join(expected["params"]))
    if expected["case"] == "missing":
        return fill_sentence(family, item.rules.style, item.lang, params=", ".join(expected["params"]),
                             api=expected["api"])
    return fill_sentence(family, item.rules.style, item.lang)


def answered_case(kind: str, order: str | None) -> str | None:
    """The case whose sentence answers a core of the kind under the prompt's order (None for the kinds
    answered with a call)."""
    if kind == "priority":
        return first_case(order, ("incorrect", "missing"))
    return kind if kind in CASES else None


def _training_rules(core: Core, cases: tuple[str, ...], rng: random.Random, order: str | None,
                    cell: Cell | None) -> tuple[dict, str, str | None]:
    """A training prompt's families, placeholder style and order (revision 1): the order first; with a cell,
    its style, and for the case the answer gives a family of the cell's shape (family_shape); without one, a
    style of BASE_STYLES, or EXTRA_SHARE of the time of EXTRA_STYLES; the other cases' families any."""
    the_order = None if core.kind == "ask" else (order or rng.choice(ORDERS))
    answered = answered_case(core.kind, the_order)
    if cell is not None:
        style = cell.style
    else:
        style = rng.choice(EXTRA_STYLES if rng.random() < EXTRA_SHARE else BASE_STYLES)
    chosen = {}
    for case in cases:
        pool = families(case, core.lang, "train")
        if cell is not None and case == answered and case != "irrelevant":
            pool = [f for f in pool if family_shape(f) == (cell.order, cell.position)]
        chosen[case] = rng.choice(pool)
    return chosen, style, the_order


def dress(core: Core, *, split: str, rng: random.Random, order: str | None = None,
          cell: Cell | None = None) -> Item:
    """The prompt and the answer of a core: list style, call notation (python only when every name and key is
    a Python identifier), rules with families of the split, a placeholder style and an order (always for the
    refusal kinds and near misses, for half of the other valid ones), the ask function for the ask kind, the
    verbatim rule for half of the valid ones, the wordings; then the messages, the answer, the expectation and
    the reasoning. A development prompt draws one number for the incorrect and the missing family and takes
    that number's style (DEV_STYLES), as the locked sets were drawn, and carries no reasoning (the one draw
    the locked build's reasoning took is kept, so that what an item's generator draws after it is unchanged);
    a training prompt takes its families, style and order from _training_rules (a cell, when given, fixes
    the style and the answered family's shape) and its reasoning from reasoning_text."""
    lang = core.lang
    list_style = rng.choice(LIST_STYLES)
    formats = [f for f in CALL_FORMATS if f != "python" or _python_ready(core.tools, _all_calls(core))]
    call_format = rng.choice(formats)
    with_rules = core.kind in SPECIAL_KINDS or core.kind == "valid_nearmiss" or rng.random() < 0.5
    rules = None
    if with_rules:
        cases = ("incorrect", "irrelevant") if core.kind == "ask" else CASES
        if (split, lang) in DEV_STYLES:
            number = rng.randrange(len(DEV_STYLES[split, lang]))
            chosen = {case: rng.choice(families(case, lang, split)) if case == "irrelevant"
                      else families(case, lang, split)[number] for case in cases}
            style = DEV_STYLES[split, lang][number]
            the_order = None if core.kind == "ask" else (order or rng.choice(ORDERS))
        else:
            chosen, style, the_order = _training_rules(core, cases, rng, order, cell)
        rules = Rules(chosen, style, the_order, rng.randrange(len(_BLOCKS[lang])), rng.randrange(2),
                      rng.randrange(2))
    ask = None
    tools = core.tools
    if core.kind == "ask":
        names = {t["function"]["name"] for t in core.tools}
        ask = rng.choice([a for a in ASK_FUNCTIONS if a.name not in names])
        tools = [*core.tools, ask.tool(lang)]
    verbatim_rule = core.kind in VALID_KINDS and rng.random() < 0.5
    wording = Wording(rng.randrange(2), rng.randrange(2), rng.randrange(2))
    system = system_prompt(tools, lang, list_style=list_style, call_format=call_format, rules=rules, ask=ask,
                           verbatim_rule=verbatim_rule, wording=wording)
    turns = [(role, turn if role == "user" else serialize_calls(turn, call_format))
             for role, turn in core.history]
    if core.packed:
        packed = pack_history(turns, lang, rng.randrange(len(_WRAPPERS[lang])))
        messages = [{"role": "system", "content": system}, {"role": "user", "content": packed}]
        context_turns = 0
    else:
        messages = [{"role": "system", "content": system}] + [{"role": r, "content": t} for r, t in turns]
        context_turns = sum(r == "assistant" for r, _ in turns)
    if core.kind in ("incorrect", "missing", "irrelevant", "priority"):
        case = core.kind if core.kind != "priority" else first_case(rules.order, ("incorrect", "missing"))
        family = rules.families[case]
        values = [b.value for b in core.invalid] if case == "incorrect" else []
        if case == "incorrect":
            params = [b.param for b in core.invalid]
        else:
            params = list(core.missing) if case == "missing" else []
        api = core.api if case == "missing" else ""
        expected = {"type": "refusal", "case": case, "fixed": fixed_parts(family, rules.style, lang),
                    "values": values, "params": params, "api": api}
        item_family = family.id
    else:
        if core.kind == "ask":
            calls = [(ask.name, {ask.function_key: core.api, ask.missing_key: list(core.missing)})]
        else:
            calls = core.calls
        expected = {"type": "call", "calls": [{"name": n, "arguments": a} for n, a in calls],
                    "verbatim": _verbatim_values(calls)}
        item_family = "+".join(f.id for f in rules.families.values()) if rules else "none"
    partial = Item(core, lang, system, messages, "", expected, "", item_family, call_format, list_style,
                   rules, ask, tools, context_turns)
    if expected["type"] == "refusal":
        answer = expected_sentence(partial)
    else:
        answer = serialize_calls([(c["name"], c["arguments"]) for c in expected["calls"]], call_format)
    item = replace(partial, answer=answer)
    if split != "train":
        rng.randrange(2)  # the locked build's reasoning drew its wording here; a development item has none
        return item
    return replace(item, reasoning=reasoning_text(item))


# ---- the reasoning: a check-first skeleton (revision 1; English, values and sentences quoted verbatim) ---
#
# docs/m8-预登记.md, 修订 1: FT5 copied the first special-mix's 2-to-6-sentence templates almost word for
# word, and their first words fixed the decision (one opening per kind); they never quoted the sentence,
# named a placeholder or wrote out the answer. Every training sample now reasons in the same order: the
# functions, the earlier turns, which function serves the request, every parameter that matters (given or
# not, valid or not against its definition), then the decision, and for a refusal the sentence as the prompt
# writes it, its placeholders mapped to their fillers, the answer and a check.

REASONING_OPENING = "Functions:"
DESCRIPTION_CHARS = 80  # of a function's description, the first line quotes this many at most
NOTE_CHARS = 60  # of a parameter's description, a check line quotes this many at most
VALUE_CHARS = 120  # of a value the reasoning quotes as given (a wrong one is quoted whole), at most
_THINK_TAG = re.compile(r"</?think>", re.IGNORECASE)


def _q(value: object, limit: int | None = None) -> str:
    """A value as the reasoning writes it: a string or a number in double quotes, verbatim, cut at limit (and
    ... added) when it is longer; an object or a list as JSON, or, longer than limit, by its keys or its
    length."""
    text = value_text(value)
    if text is None:
        text = json.dumps(value, ensure_ascii=False)
        if limit is None or len(text) <= limit:
            return text
        if isinstance(value, dict):
            return f"an object with the keys {_names([str(k) for k in value], 6)}"
        return f"a list of {len(value)} items" if isinstance(value, list) else "a value"
    if limit is not None and len(text) > limit:
        text = text[:limit] + "..."
    return f'"{text}"'


def _n(name: str) -> str:
    """A name as the reasoning writes it: bare when ASCII, quoted otherwise."""
    return name if name.isascii() else f'"{name}"'


def _names(names: list[str], limit: int = 4) -> str:
    shown = [_n(n) for n in names[:limit]]
    more = len(names) - limit
    return _join_en(shown + ([f"{more} more"] if more > 0 else []))


def _snippet(text: object, limit: int) -> str:
    """A description as the reasoning quotes it: one line, its double quotes made single, cut at limit."""
    text = " ".join(str(text or "").split()).replace('"', "'")
    return text if len(text) <= limit else text[:limit].rstrip() + "..."


def _why(c: Constraint) -> str:
    """What a constraint requires, in words, for the reasoning."""
    if c.enum:
        options = ", ".join(_q(v) for v in c.enum[:6]) + (" and a few more" if len(c.enum) > 6 else "")
        return f"allows only {options}"
    if c.fmt is not None:
        if is_date_format(c.fmt):
            words = {"ISO-8601": "an ISO 8601 date or time", "ISO-DATETIME": "an ISO 8601 date and time",
                     "TIME": "a time written as HH:MM"}
            return f"expects {words.get(c.fmt, 'the format ' + c.fmt)}"
        return {"EMAIL": "expects an email address", "URL": "expects a URL starting with http",
                "CODE3": "expects a three-letter code in capitals",
                "ZIP": "expects a five-digit ZIP code"}[c.fmt]
    if c.pattern is not None:
        return f"must match the pattern {_q(c.pattern)}"
    kind = "a whole number" if c.type == "integer" else "a number"
    if c.minimum is not None and c.maximum is not None:
        return f"needs {kind} from {_number_text(c.minimum)} to {_number_text(c.maximum)}"
    if c.maximum is not None:
        return f"needs {kind} of at most {_number_text(c.maximum)}"
    if c.minimum is not None:
        return f"needs {kind} of at least {_number_text(c.minimum)}"
    return f"needs {kind}"


def _pairs(arguments: dict, limit: int = 4) -> str:
    items = list(arguments.items())
    shown = [f"{_n(k)} = {_q(v, VALUE_CHARS)}" for k, v in items[:limit]]
    if len(items) > limit:
        shown.append(f"{len(items) - limit} more")
    return _join_en(shown) if shown else "no arguments"


def _schema_parts(function: dict) -> tuple[dict, list[str]]:
    """(properties, required) of a function object, each empty when it lists none."""
    parameters = function.get("parameters") if isinstance(function.get("parameters"), dict) else {}
    properties = parameters.get("properties") if isinstance(parameters.get("properties"), dict) else {}
    required = parameters.get("required") if isinstance(parameters.get("required"), list) else []
    return properties, [k for k in required if isinstance(k, str)]


def _definition_note(prop: object) -> str:
    """Where a parameter's definition comes from its description (a format, a list of options, a range: the
    sources infer_constraint names), the description quoted; "" otherwise."""
    if not isinstance(prop, dict):
        return ""
    c = item_constraint(prop) if prop.get("type") == "array" else infer_constraint(prop)
    if c is None or not any(source.startswith("description:") for source in c.sources):
        return ""
    return f' (the description says "{_snippet(prop.get("description"), NOTE_CHARS)}")'


def _nearmiss_why(core: Core) -> str:
    value, reason, c = core.variant["value"], core.variant["reason"], core.variant["constraint"]
    if reason == "leap":
        return f"February 29 exists in {str(value)[:4]}, a leap year"
    if reason == "time":
        return "a valid time of day, just before midnight"
    if reason == "enum":
        return "exactly one of the allowed values, spelled as listed"
    bound = c.maximum if reason == "maximum" else c.minimum
    return f"it equals the {reason} {_number_text(bound)}, which the definition allows"


def _parameter_lines(core: Core, name: str, given: dict) -> list[str]:
    """The check lines of a function's parameters (name's, as the prompt shows it): every required one and
    every optional one the latest request gives, gives wrongly or lost, in the definition's order, each
    saying whether a value is given and, when it is, whether it is valid against the definition; the other
    optional ones counted in one closing line."""
    function = next((t["function"] for t in core.tools if t["function"]["name"] == name), {})
    properties, required = _schema_parts(function)
    own = name == core.api
    invalid = {bad.param: bad for bad in core.invalid} if own else {}
    missing = set(core.missing) if own else set()
    lost = set(core.optional_cut) if own else set()
    near = core.variant.get("param") if core.kind == "valid_nearmiss" else None
    changed = core.variant.get("param") if core.variant.get("mode") == "change" else None
    bearing = [k for k in dict.fromkeys([*properties, *required, *given, *invalid, *missing, *lost])
               if k in required or k in given or k in invalid or k in missing or k in lost]
    lines = []
    for key in bearing:
        prop = properties.get(key)
        head = f"- {_n(key)} ({'required' if key in required else 'optional'}):"
        if key in invalid:
            bad = invalid[key]
            note = {"vague": f"; {_q(bad.value)} is vague, not a concrete value",
                    "format": f"; {_q(bad.value)} is written in another format"}.get(bad.category, "")
            lines.append(f"{head} given {_q(bad.value)}; not valid: the definition {_why(bad.constraint)}"
                         f"{_definition_note(prop)}{note}.")
        elif key in missing and key == core.nested:
            fields = _object_fields(prop)
            what = f"its fields {_join_en([_n(f) for f in fields])}" if fields else "any field inside it"
            lines.append(f"{head} not given; its value is a whole object, so only {_n(key)} is named, not "
                         f"{what}.")
        elif key in missing:
            lines.append(f"{head} not given.")
        elif key in lost:
            lines.append(f"{head} not given; optional, so it is not named.")
        elif key in given:
            if key == near:
                why = f": {_nearmiss_why(core)}"
            elif key == changed:
                why = f" (the latest message changed it from {_q(core.variant['old'], VALUE_CHARS)})"
            else:
                why = ""
            lines.append(f"{head} given {_q(given[key], VALUE_CHARS)}; valid{why}{_definition_note(prop)}.")
        else:
            lines.append(f"{head} not given.")
    rest = sum(k not in bearing for k in properties)
    if rest:
        lines.append(f"- {rest} more optional parameter{'s' if rest > 1 else ''}: not given and not needed.")
    return lines


def _object_fields(prop: object) -> list[str]:
    inner = prop.get("properties") if isinstance(prop, dict) else None
    return list(inner) if isinstance(inner, dict) else []


def _context_line(core: Core) -> str | None:
    """The line on the turns before the latest one, None for a single request."""
    variant = core.variant
    if variant.get("context"):
        return (f"Context: an earlier user turn was answered with a call to {_n(variant['earlier'])}; only "
                "the latest user message is answered now.")
    if variant.get("mode") == "change":
        return (f"Context: earlier the user gave {_q(variant['old'], VALUE_CHARS)} for "
                f"{_n(variant['param'])} and {_n(core.api)} was called; the latest message changes it to "
                f"{_q(variant['new'], VALUE_CHARS)}, and the latest value counts.")
    if variant.get("mode") == "switch":
        return (f"Context: the earlier request was answered with {_n(variant['first'])}; the latest message "
                "asks for something else.")
    return None


def _decision(item: Item) -> str:
    core = item.core
    if core.kind == "irrelevant":
        return "no function serves the request, so I make no call; the out-of-scope sentence applies."
    if core.kind == "incorrect":
        params = [bad.param for bad in core.invalid]
        verb = "has a value that is" if len(params) == 1 else "have values that are"
        return f"{_names(params)} {verb} not valid, so I make no call; the wrong-value sentence applies."
    if core.kind in VALID_KINDS:
        names = _join_en(list(dict.fromkeys(_n(n) for n, _ in core.calls)))
        rules = "; the refusal rules do not apply" if item.rules is not None else ""
        return f"every required value is given and valid, so I call {names}{rules}."
    lacking = f"the required {_names(core.missing)} {'is' if len(core.missing) == 1 else 'are'} not given"
    if core.kind == "missing":
        return f"{lacking}, so I make no call; the missing-value sentence applies."
    if core.kind == "priority":
        checked = "missing values" if item.rules.order == "missing_first" else "wrong values"
        sentence = "missing-value" if item.expected["case"] == "missing" else "wrong-value"
        return (f"{_n(core.invalid[0].param)} has a value that is not valid and {lacking}; the rules check "
                f"{checked} first, so I make no call and the {sentence} sentence applies.")
    return (f"{lacking}; the rules say to ask through {item.ask.name} instead of guessing, so I call "
            f"{item.ask.name}.")


def _refusal_tail(item: Item) -> list[str]:
    expected, rules = item.expected, item.rules
    family = rules.families[expected["case"]]
    marks = placeholders(rules.style, item.lang)
    opening, closing = STYLE_BRACKETS[rules.style]
    fillers = {"value": ", ".join(expected["values"]), "param": ", ".join(expected["params"]),
               "params": ", ".join(expected["params"]), "api": expected["api"]}
    slots = _SLOT.findall(family.template)
    lines = [f'Sentence: "{prompt_sentence(family, rules.style, item.lang)}"']
    if slots:
        lines.append("Fill: " + "; ".join(f'"{marks[slot]}" -> "{opening}{fillers[slot]}{closing}"'
                                          for slot in slots) + ".")
    else:
        lines.append("Fill: nothing; the sentence has no placeholder.")
    lines.append(f'Answer: "{item.answer}"')
    if not slots:
        lines.append("Check: the reply is this sentence alone, word for word, with no call.")
    elif opening or closing:
        lines.append(f"Check: no placeholder is left, each filler stands inside {opening} {closing} as its "
                     "placeholder did, and the reply is this sentence alone, with no call.")
    else:
        lines.append("Check: no placeholder is left, the fillers stand bare as the placeholders did, and the "
                     "reply is this sentence alone, with no call.")
    return lines


def _call_tail(item: Item) -> list[str]:
    calls = [(c["name"], c["arguments"]) for c in item.expected["calls"]]
    listed = "; ".join(f"{_n(name)} with {_pairs(arguments)}" for name, arguments in calls)
    chinese = any(_CJK.search(v) for v in item.expected.get("verbatim", []))
    kept = ", the Chinese ones unchanged" if chinese else ""
    check = (f"Check: the reply is the call alone in the {item.call_format} notation, with the values "
             f"copied exactly as written{kept}, and no sentence.")
    return [f"Call: {listed}.", check]


def reasoning_text(item: Item) -> str:
    """The reasoning of a training sample (revision 1's check-first skeleton; English, values quoted verbatim,
    Chinese only inside quotes), one line per step: the functions shown with what each does
    (REASONING_OPENING, the same opening for every kind), the earlier turns when there are any, which
    function serves the request (none, for an irrelevant one: a line per function), the check lines of its
    parameters (_parameter_lines), then the one decision, which states the order of the checks only when a
    wrong and a missing value occur together; for a refusal the sentence as the prompt writes it, each
    placeholder string mapped to its filler with the brackets kept (values joined by ", " in pair order), the
    answer and a check; for a call the call and a check."""
    core = item.core
    lines = [f"{REASONING_OPENING} " + "; ".join(
        f'{_n(t["function"]["name"])} ("{_snippet(t["function"].get("description"), DESCRIPTION_CHARS)}")'
        if _snippet(t["function"].get("description"), DESCRIPTION_CHARS) else _n(t["function"]["name"])
        for t in item.tools) + "."]
    context = _context_line(core)
    if context:
        lines.append(context)
    if core.kind == "irrelevant":
        lines += [f"- {_n(t['function']['name'])}: not what the request asks for." for t in item.tools]
        lines.append("Match: none of the functions serves the request.")
    else:
        calls = core.calls if core.kind in VALID_KINDS else [(core.api, core.given)]
        if len(calls) == 1:
            lines.append(f"Match: {_n(calls[0][0])} serves the request.")
        else:
            names = _join_en(list(dict.fromkeys(_n(name) for name, _ in calls)))
            lines.append(f"Match: the request takes {len(calls)} calls, of {names}.")
        for name, arguments in calls:
            lines.append(f"Parameters of {_n(name)}:")
            lines += _parameter_lines(core, name, arguments)
    lines.append(f"Decision: {_decision(item)}")
    lines += _refusal_tail(item) if item.expected["type"] == "refusal" else _call_tail(item)
    return "\n".join(lines)


LACKING_KINDS = ("missing", "priority", "ask")  # the kinds whose request lacks a required value by design


def reasoning_problems(item: Item) -> list[str]:
    """Why a training item's reasoning is not the skeleton reasoning_text writes, as messages; [] when it is:
    the opening line, one decision after every check line, no required parameter "not given" outside the
    kinds that cut one (LACKING_KINDS: an incorrect item with one is a priority case, a call made without one
    breaks the rules; the review of revision 1, record 448), for a refusal the sentence, every placeholder of
    it mapped and the answer, each word for word, a closing check; at most MAX_REASONING_TOKENS
    (approx_tokens), no think tag."""
    text = item.reasoning
    lines = text.split("\n")
    problems = []
    if not lines[0].startswith(REASONING_OPENING):
        problems.append(f"the first line does not open with {REASONING_OPENING}")
    decisions = [n for n, line in enumerate(lines) if line.startswith("Decision:")]
    checks = [n for n, line in enumerate(lines) if line.startswith("- ")]
    if len(decisions) != 1 or not checks or max(checks) > decisions[0]:
        problems.append("not one decision after every check line")
    if item.core.kind not in LACKING_KINDS and any(line.startswith("- ") and "(required): not given" in line
                                                     for line in lines):
        article = "an" if item.core.kind[0] in "aeiou" else "a"
        problems.append(f"a required parameter not given in {article} {item.core.kind} item")
    if item.expected["type"] == "refusal" and item.rules is not None:
        family = item.rules.families[item.expected["case"]]
        if f'Sentence: "{prompt_sentence(family, item.rules.style, item.lang)}"' not in lines:
            problems.append("the sentence is not quoted as the prompt writes it")
        if f'Answer: "{item.answer}"' not in lines:
            problems.append("the answer is not written out")
        fill = next((line for line in lines if line.startswith("Fill:")), "")
        marks = placeholders(item.rules.style, item.lang)
        if any(f'"{marks[slot]}" ->' not in fill for slot in _SLOT.findall(family.template)):
            problems.append("a placeholder is not mapped to its filler")
    if not lines[-1].startswith("Check:"):
        problems.append("no closing check")
    if approx_tokens(text) > MAX_REASONING_TOKENS:
        problems.append(f"longer than {MAX_REASONING_TOKENS} tokens")
    if _THINK_TAG.search(text):
        problems.append("a think tag")
    return problems


# ---- the two output formats ----------------------------------------------------------------------------


def training_record(item: Item) -> dict:
    """The training sample (module docstring: Formats): the messages with the answer last, "tools": [], the
    source and its first record, the kind, the prompt language and the prompt's choices; context_turns when
    earlier assistant turns are context only; source_ids when the sample used more than one record."""
    core = item.core
    record = {"messages": [*item.messages, {"role": "assistant", "content": item.answer}], "tools": [],
              "source": core.source, "source_id": core.records[0], "kind": core.kind,
              "instruction_lang": item.lang, "family": item.family, "call_format": item.call_format,
              "list_style": item.list_style, "placeholder": item.rules.style if item.rules else None,
              "variant": _plain(core.variant)}
    if item.context_turns:
        record["context_turns"] = item.context_turns
    if len(core.records) > 1:
        record["source_ids"] = list(core.records)
    return record


def _plain(value: object) -> object:
    """A variant as JSON: a constraint by the names of its sources."""
    if isinstance(value, Constraint):
        return list(value.sources)
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


CUT_KINDS = ("missing", "priority")  # the development kinds whose request lost a value: source says how


def dev_item(item: Item, split: str, number: int) -> dict:
    """The development item (module docstring: Formats), numbered within its split; an item whose request lost
    a value (CUT_KINDS) says in source["cut"] whether the cut took the value's frame and reads fluently
    ("frame") or not ("bare")."""
    core = item.core
    source = {"dataset": core.source, "record": "+".join(core.records)}
    if core.kind in CUT_KINDS:
        source["cut"] = core.variant["cut"]
    return {"id": f"sdev-{split}-{number:04d}", "split": split, "lang": item.lang, "kind": core.kind,
            "messages": item.messages, "call_format": item.call_format, "family": item.family,
            "expected": item.expected, "source": source}


DEV_KEYS = ("id", "split", "lang", "kind", "messages", "call_format", "family", "expected", "source")
REFUSAL_KEYS = ("type", "case", "fixed", "values", "params", "api")
CALL_KEYS = ("type", "calls", "verbatim")
CUTS = ("frame", "bare")
_DEV_ID = re.compile(r"sdev-(?P<split>[ab])-\d{4}")


def _strings(value: object) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) and v for v in value)


def check_dev_item(item: object) -> list[str]:
    """Why an object is not a development item of the agreed format, as messages; [] when it is."""
    if not isinstance(item, dict) or tuple(item) != DEV_KEYS:
        return [f"the keys must be exactly {', '.join(DEV_KEYS)}, in this order"]
    errors = []
    match = _DEV_ID.fullmatch(str(item["id"]))
    if match is None or match.group("split") != item["split"]:
        errors.append("id must be sdev-<split>-<nnnn> of its split")
    if item["lang"] not in LANGS or item["kind"] not in DEV_KINDS or item["call_format"] not in CALL_FORMATS:
        errors.append("lang, kind or call_format unknown")
    messages = item["messages"]
    if not isinstance(messages, list) or len(messages) < 2 or any(
            not isinstance(m, dict) or set(m) != {"role", "content"} or not isinstance(m["content"], str)
            or not m["content"].strip() or m["role"] not in ("system", "user", "assistant")
            for m in messages):
        errors.append("messages must be non-empty {role, content} messages")
    elif messages[0]["role"] != "system" or messages[1]["role"] != "user" or messages[-1]["role"] != "user":
        errors.append("messages must open with the system and a user message and end with a user message")
    if not isinstance(item["family"], str) or not item["family"]:
        errors.append("family must be a non-empty string")
    source = item["source"]
    keys = {"dataset", "record", "cut"} if item["kind"] in CUT_KINDS else {"dataset", "record"}
    if not isinstance(source, dict) or set(source) != keys or not all(
            isinstance(v, str) and v for v in source.values()) or source.get("cut", CUTS[0]) not in CUTS:
        errors.append("source must be {dataset, record}, with cut (frame or bare) for a missing or priority "
                      "item")
    expected = item["expected"]
    refusal = item["kind"] in ("incorrect", "missing", "irrelevant", "priority")
    if not isinstance(expected, dict):
        return errors + ["expected must be an object"]
    if refusal:
        if tuple(expected) != REFUSAL_KEYS or expected["type"] != "refusal":
            return errors + ["a refusal expectation has exactly type, case, fixed, values, params, api"]
        case = expected["case"]
        allowed = ("incorrect", "missing") if item["kind"] == "priority" else (item["kind"],)
        if case not in allowed:
            errors.append(f"case must be one of {', '.join(allowed)}")
        if not _strings(expected["fixed"]) or not expected["fixed"]:
            errors.append("fixed must be non-empty strings")
        if not isinstance(expected["values"], list) or not isinstance(expected["params"], list) or not \
                isinstance(expected["api"], str):
            errors.append("values and params must be lists, api a string")
        elif case == "incorrect" and (not _strings(expected["values"]) or not _strings(expected["params"])
                                      or len(expected["values"]) != len(expected["params"])
                                      or not expected["values"] or expected["api"]):
            errors.append("an incorrect case names each wrong value with its parameter and no api")
        elif case == "missing" and (expected["values"] or not _strings(expected["params"])
                                    or not expected["params"] or not expected["api"]):
            errors.append("a missing case names the parameters and the api and no value")
        elif case == "irrelevant" and (expected["values"] or expected["params"] or expected["api"]):
            errors.append("an irrelevant case names nothing")
    else:
        if tuple(expected) != CALL_KEYS or expected["type"] != "call":
            return errors + ["a call expectation has exactly type, calls, verbatim"]
        calls = expected["calls"]
        if not isinstance(calls, list) or not calls or any(
                not isinstance(c, dict) or set(c) != {"name", "arguments"} or not isinstance(c["name"], str)
                or not isinstance(c["arguments"], dict) for c in calls):
            errors.append("calls must be {name, arguments} objects")
        verbatim = expected["verbatim"]
        if not isinstance(verbatim, list) or not all(isinstance(v, str) for v in verbatim):
            errors.append("verbatim must be strings")
    return errors


def dev_set_problems(items: Iterable[dict]) -> list[str]:
    """Why development items cannot stand in one set, as messages; [] when they can: an id given twice, or a
    family that gives one case two sentences (two lists of fixed words), which the scorer refuses, since it
    reads a set's sentences family by family (research_agent.eval.special_dev.sentence_families)."""
    problems: list[str] = []
    ids: set[str] = set()
    sentences: dict[tuple[str, str], tuple[str, ...]] = {}
    for item in items:
        if item["id"] in ids:
            problems.append(f"{item['id']} is given twice")
        ids.add(item["id"])
        expected = item["expected"]
        if expected.get("type") != "refusal":
            continue
        key, fixed = (item["family"], expected["case"]), tuple(expected["fixed"])
        if sentences.setdefault(key, fixed) != fixed:
            problems.append(f"{item['id']}: the family {key[0]} gives the {key[1]} case a second sentence")
    return problems


def check_training_record(record: object) -> list[str]:
    """Why an object is not a special-mix training record, as messages; [] when it is."""
    if not isinstance(record, dict):
        return ["a record is an object"]
    errors = []
    if record.get("tools") != []:
        errors.append("tools must be an empty list")
    if record.get("kind") not in KINDS or record.get("instruction_lang") not in LANGS:
        errors.append("kind or instruction_lang unknown")
    if not isinstance(record.get("source"), str) or not isinstance(record.get("source_id"), str):
        errors.append("source and source_id must be strings")
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 3 or any(
            not isinstance(m, dict) or set(m) != {"role", "content"} or not isinstance(m["content"], str)
            or not m["content"].strip() for m in messages):
        return errors + ["messages must be {role, content} messages with text"]
    roles = [m["role"] for m in messages]
    if roles[0] != "system" or roles[1] != "user" or roles[-1] != "assistant" or any(
            a == b for a, b in itertools.pairwise(roles[1:])) or "system" in roles[1:]:
        errors.append("messages must be the system message, then user and assistant turns in turn, ending "
                      "with the answer")
    turns = roles.count("assistant")
    context = record.get("context_turns", 0)
    if type(context) is not int or context != turns - 1:
        errors.append("every assistant turn before the answer must be a context turn (context_turns)")
    for message in messages:
        for marker in banned_markers(message["content"]):
            errors.append(f"a {message['role']} message holds {marker}")
    return errors


# ---- the texts of this module, for the ban test --------------------------------------------------------

_EXAMPLE_TOOLS = [{"type": "function", "function": {
    "name": "weather_lookup", "description": "Look up weather.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def own_texts() -> Iterator[str]:
    """Every fixed text this module writes into a prompt or an answer, in each combination a prompt can hold:
    the families in every style (as shown and filled in), every rules block, preamble, call rule, verbatim
    rule, history wrapper, follow-up, switch prefix and ask function."""
    for family in FAMILIES.values():
        for style in STYLES:
            yield prompt_sentence(family, style, family.lang)
            yield fill_sentence(family, style, family.lang, value="v", param="p", params="p, q", api="f")
    for lang in LANGS:
        for number, style in enumerate(STYLES):
            for block in range(len(_BLOCKS[lang])):
                for order in (*ORDERS, None):
                    for wording in range(2):
                        pick = number * len(_BLOCKS[lang]) + block
                        trained = {case: families(case, lang, "train") for case in CASES}
                        chosen = {case: trained[case][pick % len(trained[case])] for case in CASES}
                        ask = ASK_FUNCTIONS[(number + block) % len(ASK_FUNCTIONS)] if order is None else None
                        if ask is not None:
                            chosen.pop("missing")
                        rules = Rules(chosen, style, order, block, wording, wording)
                        yield refusal_block(rules, lang, ask)
        for list_style in LIST_STYLES:
            for variant in range(2):
                for call_format in CALL_FORMATS:
                    yield system_prompt(_EXAMPLE_TOOLS, lang, list_style=list_style, call_format=call_format,
                                        verbatim_rule=True, wording=Wording(variant, variant, variant))
        for wrapper in range(len(_WRAPPERS[lang])):
            yield pack_history([("user", "first"), ("assistant", "call"), ("user", "second")], lang, wrapper)
        yield from (f.format(old="A", new="B") for f in FOLLOW_UPS[lang])
        yield from (p + "request" for p in SWITCH_PREFIXES[lang])
        for ask in ASK_FUNCTIONS:
            yield json.dumps(ask.tool(lang), ensure_ascii=False)

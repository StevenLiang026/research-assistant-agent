"""research_agent.data.special_mix (M8 WP1, WP2): sentence families, placeholders, function lists, constraint
inference, value mutation, the case builders, dressing (prompt, answer, expected, reasoning), the record and
item formats and the ACEBench text ban.

Offline: records are hand-made; the tests that read the ACEBench clone (runs/third_party/ACEBench) are skipped
when it is absent, and they read only its prompt templates and its function names.
"""

from __future__ import annotations

import ast
import dataclasses
import difflib
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import pytest
from test_synth import load_script

from research_agent.data import special_mix as sm
from research_agent.data.reasoning import check_row, sidecar_row

REPO = Path(__file__).resolve().parents[2]
CLONE = REPO / "runs" / "third_party" / "ACEBench"
PROMPTS = [CLONE / "model_inference" / "prompt_en.py", CLONE / "model_inference" / "prompt_zh.py"]
needs_clone = pytest.mark.skipif(not all(p.is_file() for p in PROMPTS), reason="ACEBench clone not present")

bp = load_script("build_ace_probe")
rfd = load_script("render_format_data")


def tool(name: str, properties: dict, required: list[str] | None = None,
         description: str | None = None) -> dict:
    parameters: dict = {"type": "object", "properties": properties}
    if required is not None:
        parameters["required"] = required
    return {"type": "function",
            "function": {"name": name, "description": description or f"{name} does one thing",
                         "parameters": parameters}}


def sample(source: str, sid: str, user: str, calls: list[tuple[str, dict]], tools: list[dict],
           text: str = "") -> dict:
    messages: list[dict] = [{"role": "user", "content": user}]
    if calls:
        messages.append({"role": "assistant", "content": text, "tool_calls": [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}}
            for i, (n, a) in enumerate(calls)]})
        messages += [{"role": "tool", "tool_call_id": f"call_{i}", "content": "{}"}
                     for i in range(len(calls))]
        messages.append({"role": "assistant", "content": "Done."})
    else:
        messages.append({"role": "assistant", "content": text})
    return {"messages": messages, "tools": tools, "source": source, "source_id": sid}


MEETING = tool("create_meeting", {
    "title": {"type": "string", "description": "Title of the meeting"},
    "date": {"type": "string", "format": "date", "description": "Day of the meeting"},
    "duration": {"type": "integer", "minimum": 15, "maximum": 240, "description": "Length in minutes"},
    "room": {"type": "string", "enum": ["A1", "B2", "C3"], "description": "Meeting room"},
    "notes": {"type": "string", "description": "Free notes"},
}, ["title", "date", "duration"])
WEATHER = tool("get_forecast", {"city": {"type": "string", "description": "City name"}}, ["city"])


def meeting(source: str = "toolace", sid: str = "1") -> dict:
    user = 'Please schedule "Budget review" on 2024-03-05 for 90 minutes in room B2.'
    args = {"title": "Budget review", "date": "2024-03-05", "duration": 90, "room": "B2"}
    return sample(source, sid, user, [("create_meeting", args)], [MEETING, WEATHER])


JOBS = tool("DescribeJobQueue", {
    "ProjectId": {"type": "integer", "description": "工作空间的ID。"},
    "Status": {"type": "string",
               "description": ("任务状态,取值:\n- **RUNNING**:运行中。\n- **STOPPED**:已停止。\n"
                               "- **FAILED**:失败。")},
    "PageSize": {"type": "integer", "description": "每页显示的条数,最大值为100。"},
    "StartDate": {"type": "string", "description": "开始日期,格式为yyyy-MM-dd。"},
})
CATALOG = tool("CreateDataCatalog", {"Name": {"type": "string", "description": "目录名称"},
                                     "Comment": {"type": "string", "description": "备注"}})


def jobs(sid: str = "7") -> dict:
    user = "帮我查一下工作空间4471里状态为RUNNING的任务,每页30条,开始日期2024-01-15。"
    args = {"ProjectId": 4471, "Status": "RUNNING", "PageSize": 30, "StartDate": "2024-01-15"}
    return sample("deepexi", sid, user, [("DescribeJobQueue", args)], [JOBS, CATALOG])


def catalog(sid: str = "8") -> dict:
    user = "新建一个名为销售数据的目录,备注写季度汇总。"
    return sample("deepexi", sid, user, [("CreateDataCatalog", {"Name": "销售数据", "Comment": "季度汇总"})],
                  [CATALOG, JOBS])


TRIP = tool("plan_trip", {
    "destination": {"type": "string", "description": "Where to go"},
    "budget": {"type": "object", "description": "Budget of the trip",
               "properties": {"amount": {"type": "integer"}, "currency": {"type": "string"}}},
    "travellers": {"type": "integer", "description": "Number of people"},
}, ["destination", "budget"])


def trip(sid: str = "3") -> dict:
    user = "Plan a trip to Lisbon for 2 people with a budget of 1500 EUR."
    args = {"destination": "Lisbon", "budget": {"amount": 1500, "currency": "EUR"}, "travellers": 2}
    return sample("glaive_v2", sid, user, [("plan_trip", args)], [TRIP])


def rng(key: str = "t") -> random.Random:
    return random.Random(key)


# ---- sentence families and placeholders ------------------------------------------------------------------


def test_families_thirty_for_training_and_three_held_out_per_case_language_and_split():
    # revision 1 (docs/m8-预登记.md, 修订 1): 30 training families per case and language, up from 10; the
    # development families stay as they were locked
    for case in sm.CASES:
        for lang in sm.LANGS:
            assert len(sm.families(case, lang, "train")) == 30
            assert len(sm.families(case, lang, "a")) == 3
            assert len(sm.families(case, lang, "b")) == 3
    templates = [f.template for f in sm.FAMILIES.values()]
    assert len(set(templates)) == len(templates) == 216
    assert len(set(sm.FAMILIES)) == 216
    slots = {"incorrect": {"value", "param"}, "missing": {"params", "api"}, "irrelevant": set()}
    for family in sm.FAMILIES.values():
        assert set(re.findall(r"\{(\w+)\}", family.template)) == slots[family.case], family.id
        assert family.id.startswith(f"{family.case}-{family.lang}-")
        assert len(re.findall(r"\{\w+\}", family.template)) == len(slots[family.case])  # each slot once


# The development families as the sets were locked (data/eval/special-dev-{a,b}.jsonl): their ids and
# templates may not change, since the scorer reads a set's sentences from its items and the build must draw
# the locked sets again byte for byte.
LOCKED_DEV_FAMILIES = {
    "incorrect-en-a1": "The supplied {param} value {value} breaks the function's rules.",
    "incorrect-en-a2": "Bad value {value} in {param}; please fix it.",
    "incorrect-en-a3": "I cannot use {value} as {param}.",
    "incorrect-en-b1": "{param} does not allow the value {value}.",
    "incorrect-en-b2": "Wrong value for {param}: {value} is outside what the function allows.",
    "incorrect-en-b3": "Unable to call: the {param} value {value} is not permitted.",
    "incorrect-zh-a1": "Unacceptable value {value} supplied for {param}.",
    "incorrect-zh-a2": "{value} cannot be used for {param}.",
    "incorrect-zh-a3": "{value} 不是 {param} 的有效取值。",
    "incorrect-zh-b1": "The input {value} is invalid for {param}.",
    "incorrect-zh-b2": "Incorrect input in {param}: {value}.",
    "incorrect-zh-b3": "{param} 的值 {value} 无法接受。",
    "missing-en-a1": "To use {api} I still need {params}.",
    "missing-en-a2": "Incomplete request for {api}: {params} missing.",
    "missing-en-a3": "{api} cannot start until {params} is given.",
    "missing-en-b1": "Missing for {api}: {params}.",
    "missing-en-b2": "Before {api} can run, please give {params}.",
    "missing-en-b3": "{api} requires {params}, which were not stated.",
    "missing-zh-a1": "Still missing for {api}: {params}.",
    "missing-zh-a2": "{api} is waiting for {params}.",
    "missing-zh-a3": "{api} 的必填参数 {params} 未提供。",
    "missing-zh-b1": "More input needed before calling {api}: {params}.",
    "missing-zh-b2": "Missing values {params} prevent calling {api}.",
    "missing-zh-b3": "尚未给出 {params},{api} 无法执行。",
    "irrelevant-en-a1": "No available function matches what is asked.",
    "irrelevant-en-a2": "The functions on offer cannot do this.",
    "irrelevant-en-a3": "Request not supported by any listed function.",
    "irrelevant-en-b1": "None of these tools applies to the request.",
    "irrelevant-en-b2": "This is beyond the listed functions.",
    "irrelevant-en-b3": "No function in the list can carry this out.",
    "irrelevant-zh-a1": "This is not something the given functions do.",
    "irrelevant-zh-a2": "No tool here fits the request.",
    "irrelevant-zh-a3": "所列函数均不适用于此请求。",
    "irrelevant-zh-b1": "The request is not covered by the functions.",
    "irrelevant-zh-b2": "Functions listed: none suitable.",
    "irrelevant-zh-b3": "这个请求不在可用函数的处理范围内。",
}
_SLOTS = re.compile(r"\{(?:value|param|params|api)\}")
_CJK_CHAR = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")


def fixed_segments(template: str) -> list[str]:
    """The text of a template between its slots, normalised (special_mix's own word and window rules)."""
    return [" ".join(sm.normalize_text(s).lower().split()) for s in _SLOTS.split(template) if s.strip()]


def three_words(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", _CJK_CHAR.sub(" ", text))
    return {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}


def four_chars(text: str) -> set[str]:
    chars = "".join(text.split())
    return {chars[i:i + 4] for i in range(len(chars) - 3) if _CJK_CHAR.search(chars[i:i + 4])}


def test_the_development_families_are_those_of_the_locked_sets():
    dev = {fid: f.template for fid, f in sm.FAMILIES.items() if f.split in ("a", "b")}
    assert dev == LOCKED_DEV_FAMILIES


def test_training_families_share_no_fixed_part_or_window_with_the_development_families():
    # revision 1: a training sentence may share with a development family (A or B) no fixed part of two or
    # more words (or Chinese characters), no 3-word window and no 4-character Chinese window, read within the
    # text between the slots; the analysis found 5 such pairs among the first ten families (必填参数, "the
    # listed functions", ...), which the revision rewrote
    dev = [f for f in sm.FAMILIES.values() if f.split != "train"]

    def marks(template: str) -> set[str]:
        out = set()
        for segment in fixed_segments(template):
            core = segment.strip(" .,:;。，：；")
            if len(re.findall(r"[a-z0-9]+", core)) >= 2 or len(_CJK_CHAR.findall(core)) >= 2:
                out.add(core)
            out |= three_words(segment) | four_chars(segment)
        return out

    held = {f.id: marks(f.template) for f in dev}
    for family in (f for f in sm.FAMILIES.values() if f.split == "train"):
        for other in dev:
            assert not marks(family.template) & held[other.id], (family.id, other.id,
                                                                 marks(family.template) & held[other.id])


SKELETON_CAP = 0.5  # a training family's token similarity with any development family stays below it


def skeleton_tokens(template: str, separators: bool = True) -> list[str]:
    """A template read as a skeleton: every slot one token, the words between them (ASCII letters and digits,
    each Chinese character, normalised as fixed_segments does) and, with separators, the marks : ; , (of
    either width)."""
    pattern = r"[a-z0-9]+|" + _CJK_CHAR.pattern + (r"|[:;,]" if separators else "")
    tokens, position = [], 0
    for match in [*_SLOTS.finditer(template), None]:
        end = match.start() if match else len(template)
        tokens += re.findall(pattern, sm.normalize_text(template[position:end]).lower())
        if match:
            tokens.append(match.group(0))
            position = match.end()
    return tokens


def slot_windows(template: str) -> set[str]:
    """The 3-token windows of a skeleton (separators left out) that hold a slot and at least two words: what
    the windows read between the slots cannot see ("the value {value}", "给 出 {params}")."""
    tokens = skeleton_tokens(template, separators=False)
    out = set()
    for i in range(len(tokens) - 2):
        window = tokens[i:i + 3]
        slots = sum(token.startswith("{") for token in window)
        if slots and len(window) - slots >= 2:
            out.add(" ".join(window))
    return out


def test_training_families_keep_no_skeleton_of_a_development_family():
    # the review of revision 1: read only between the slots, the rule above let through near copies of the
    # development families: "Missing input for {api}: {params}." is a B sentence with one word added (a
    # token similarity of 0.91), "{param} cannot hold the value {value}." keeps another's "the value
    # {value}". With every slot one token, a training sentence shares no 3-token window holding a slot and
    # two words with any development family (A or B, any case), and its similarity to each (difflib's ratio
    # of the skeletons, with the separators and without) stays below SKELETON_CAP
    dev = [f for f in sm.FAMILIES.values() if f.split != "train"]
    held = {f.id: slot_windows(f.template) for f in dev}
    skeletons = {(f.id, separators): skeleton_tokens(f.template, separators)
                 for f in sm.FAMILIES.values() for separators in (True, False)}
    for family in (f for f in sm.FAMILIES.values() if f.split == "train"):
        windows = slot_windows(family.template)
        for other in dev:
            shared = windows & held[other.id]
            assert not shared, (family.id, other.id, sorted(shared))
            for separators in (True, False):
                similarity = difflib.SequenceMatcher(None, skeletons[family.id, separators],
                                                     skeletons[other.id, separators], autojunk=False).ratio()
                assert similarity < SKELETON_CAP, (family.id, other.id, separators, round(similarity, 2))
    assert skeleton_tokens("Nothing for {api}： {params}.") == ["nothing", "for", "{api}", ":", "{params}"]
    assert skeleton_tokens("Nothing for {api}： {params}.", separators=False) == ["nothing", "for", "{api}",
                                                                                 "{params}"]
    assert slot_windows("{param} rejects the entry {value} here.") == {
        "{param} rejects the", "the entry {value}", "entry {value} here"}
    assert slot_windows("请补 {params}。") == {"请 补 {params}"}


@pytest.mark.parametrize("case", ("incorrect", "missing"))
@pytest.mark.parametrize("lang", sm.LANGS)
def test_training_families_cover_every_slot_shape(case, lang):
    # revision 1: every slot order (the value or the parameter names first, or the parameter or function
    # first) with the sentence starting with a slot, ending with one or neither, in at least four families;
    # a zh prompt's families write each shape in English and in Chinese
    train = sm.families(case, lang, "train")
    shapes = Counter(sm.family_shape(f) for f in train)
    assert set(shapes) == {(order, position) for order in sm.SLOT_ORDERS for position in sm.POSITIONS}
    assert min(shapes.values()) >= 4, shapes
    if lang == "zh":
        for shape in shapes:
            written = {"zh" if _CJK_CHAR.search(f.template) else "en" for f in train
                       if sm.family_shape(f) == shape}
            assert written == {"zh", "en"}, shape
    assert sm.family_shape(sm.FAMILIES["incorrect-zh-a2"]) == ("filler_first", "leading")
    assert sm.family_shape(sm.FAMILIES["missing-zh-a1"]) == ("name_first", "trailing")
    assert sm.family_shape(sm.FAMILIES["incorrect-en-a1"]) == ("name_first", "inner")
    assert sm.family_shape(sm.FAMILIES["irrelevant-en-a1"]) is None


def test_the_extra_placeholder_styles_are_for_training_only():
    assert sm.STYLES == sm.BASE_STYLES + sm.EXTRA_STYLES and len(sm.EXTRA_STYLES) == 15
    assert not set(sm.BASE_STYLES) & set(sm.EXTRA_STYLES)
    for split in ("a", "b"):
        assert set(sm.DEV_STYLES[split, "zh"]) | set(sm.DEV_STYLES[split, "en"]) == set(sm.BASE_STYLES)
    shapes = {sm.STYLE_BRACKETS[style] for style in sm.STYLES}
    assert len(shapes) == 9  # the six bracket shapes and three more
    for style in sm.STYLES:
        for lang in sm.LANGS:
            marks = sm.placeholders(style, lang)
            assert set(marks) == {"value", "param", "params", "api"} and len(set(marks.values())) == 4
            assert all(sm.banned_markers(mark) == [] for mark in marks.values())
    words = {style: tuple(sm.placeholders(style, "en").values()) for style in sm.STYLES}
    assert len(set(words.values())) == len(sm.STYLES)  # every style writes its placeholders its own way


@pytest.mark.parametrize("style", sm.STYLES)
@pytest.mark.parametrize("lang", sm.LANGS)
def test_placeholders_are_replaced_and_their_brackets_kept(style, lang):
    family = sm.families("incorrect", lang, "train")[0]
    shown = sm.prompt_sentence(family, style, lang)
    marks = sm.placeholders(style, lang)
    assert marks["value"] in shown and marks["param"] in shown
    answer = sm.fill_sentence(family, style, lang, value="十几", param="ProjectId")
    opening, closing = sm.STYLE_BRACKETS[style]
    assert f"{opening}十几{closing}" in answer and f"{opening}ProjectId{closing}" in answer
    assert marks["value"] not in answer and marks["param"] not in answer
    position = 0
    for part in sm.fixed_parts(family, style, lang):
        found = answer.find(part, position)
        assert found >= 0, (part, answer)
        position = found + len(part)
    missing = sm.families("missing", lang, "b")[0]
    filled = sm.fill_sentence(missing, style, lang, params="date, time", api="create_event")
    assert f"{opening}date, time{closing}" in filled and f"{opening}create_event{closing}" in filled
    # the fixed words carry the brackets: between them stand only the fillers
    parts = sm.fixed_parts(family, style, lang)
    rebuilt = re.sub(r"\s+", "", "".join(parts))
    assert rebuilt == re.sub(r"\s+", "", answer.replace("十几", "").replace("ProjectId", ""))


def test_a_filler_written_like_a_placeholder_reads_as_one():
    item = sm.dress(sm.build_incorrect(meeting(), "en", rng(), category="obvious"), split="train", rng=rng())
    assert item.expected["type"] == "refusal" and not sm.reads_as_placeholder(item)
    for style in sm.STYLES:
        rules = dataclasses.replace(item.rules, style=style)
        opening, closing = sm.STYLE_BRACKETS[style]
        for slot, mark in sm.placeholders(style, "en").items():
            word = mark[len(opening):len(mark) - len(closing)]
            expected = {**item.expected, "params": [word.lower()]}
            named = dataclasses.replace(item, rules=rules, expected=expected)
            assert sm.reads_as_placeholder(named), (style, slot, word)
    call = sm.dress(sm.build_verbatim(catalog(), "zh", rng(), require_cjk=True), split="train", rng=rng())
    assert not sm.reads_as_placeholder(call)


def test_fixed_parts_of_one_family_and_style():
    family = sm.FAMILIES["incorrect-en-t01"]  # "Passing {value} as {param} would break the call, so ..."
    assert sm.fixed_parts(family, "paren", "en") == ["Passing (", ") as (",
                                                     ") would break the call, so nothing is sent."]
    assert sm.fixed_parts(family, "caps", "en") == ["Passing", "as",
                                                    "would break the call, so nothing is sent."]
    assert sm.fill_sentence(family, "corner", "en", value="十几", param="ProjectId") == \
        "Passing 「十几」 as 「ProjectId」 would break the call, so nothing is sent."


def test_irrelevant_sentences_have_no_placeholder_and_are_their_own_fixed_part():
    family = sm.families("irrelevant", "en", "a")[1]
    assert sm.fill_sentence(family, "angle", "en") == family.template
    assert sm.fixed_parts(family, "caps", "en") == [family.template]


def test_first_case_follows_the_prompt_order():
    both = {"incorrect", "missing"}
    assert sm.first_case("incorrect_first", both) == "incorrect"
    assert sm.first_case("missing_first", both) == "missing"
    assert sm.first_case("incorrect_first", {"missing"}) == "missing"
    assert sm.first_case("missing_first", {"incorrect"}) == "incorrect"


# ---- function lists ------------------------------------------------------------------------------------


def test_function_lists_in_four_renderings():
    tools = [MEETING, TRIP]
    lines = sm.render_functions(tools, "json_lines", "en").split("\n")
    assert [json.loads(line)["name"] for line in lines] == ["create_meeting", "plan_trip"]
    assert [f["name"] for f in json.loads(sm.render_functions(tools, "json_array", "en"))] == \
        ["create_meeting", "plan_trip"]
    assert [f["name"] for f in ast.literal_eval(sm.render_functions(tools, "python_repr", "zh"))] == \
        ["create_meeting", "plan_trip"]
    md = sm.render_functions(tools, "markdown", "en")
    for text in ("create_meeting", "plan_trip", "`duration`", "required", "A1", "B2", "C3", "date",
                 "15", "240", "`amount`", "`currency`"):
        assert text in md
    assert "必填" in sm.render_functions(tools, "markdown", "zh")


# ---- constraints -----------------------------------------------------------------------------------------


def test_constraints_from_schema_keywords():
    c = sm.infer_constraint({"type": "string", "enum": ["low", "medium", "high"]})
    assert c.enum == ("low", "medium", "high") and "schema:enum" in c.sources
    c = sm.infer_constraint({"type": "integer", "minimum": 1, "maximum": 100})
    assert (c.type, c.minimum, c.maximum) == ("integer", 1, 100)
    pattern = "^[A-Z]{2}\\d{4}$"
    assert sm.infer_constraint({"type": "string", "pattern": pattern}).pattern == pattern
    assert sm.infer_constraint({"type": "string", "format": "date"}).fmt == "YYYY-MM-DD"
    assert sm.infer_constraint({"type": "number"}).type == "number"
    items = sm.item_constraint({"type": "array", "items": {"type": "string", "enum": ["x", "y"]}})
    assert items.enum == ("x", "y")
    assert sm.infer_constraint({"type": "boolean"}) is None
    assert sm.infer_constraint({"type": "string", "description": "Free text"}) is None
    # schema keywords come first: the description's date format does not override format date-time
    c = sm.infer_constraint({"type": "string", "format": "date-time", "description": "in YYYY-MM-DD"})
    assert c.fmt == "ISO-DATETIME"


@pytest.mark.parametrize("prop, check", [
    ({"type": "string", "description": "Check-in date in YYYY-MM-DD format"}, ("fmt", "YYYY-MM-DD")),
    ({"type": "string", "description": "Start (HH:MM)"}, ("fmt", "HH:MM")),
    ({"type": "string", "description": "When, as YYYY-MM-DD HH:MM"}, ("fmt", "YYYY-MM-DD HH:MM")),
    ({"type": "string", "description": "开始时间,格式为yyyy-MM-dd HH:mm:ss。"},
     ("fmt", "YYYY-MM-DD HH:MM:SS")),
    ({"type": "string", "description": "开始日期,格式为yyyy-MM-dd。"}, ("fmt", "YYYY-MM-DD")),
    ({"type": "string", "description": "Start time in ISO 8601 format"}, ("fmt", "ISO-8601")),
    ({"type": "string", "description": "The email address of the guest"}, ("fmt", "EMAIL")),
    ({"type": "string", "description": "The URL of the page"}, ("fmt", "URL")),
    ({"type": "string", "description": "3-letter currency code"}, ("fmt", "CODE3")),
    ({"type": "string", "description": "ZIP code of the shop"}, ("fmt", "ZIP")),
    ({"type": "string", "description": "状态,取值:RUNNING、STOPPED、FAILED。"},
     ("enum", ("RUNNING", "STOPPED", "FAILED"))),
    ({"type": "string", "description": "导出模式,取值如下:\n- FULL:全量导出。\n- INCREMENTAL:增量导出。"},
     ("enum", ("FULL", "INCREMENTAL"))),
    ({"type": "string", "description": "Unit. Valid values are 'kg' and 'm'"}, ("enum", ("kg", "m"))),
    ({"type": "integer", "description": "页码,取值范围:1~100。"}, ("range", (1, 100))),
    ({"type": "integer", "description": "每页显示的条数,默认为10条,最大值为100。"}, ("range", (None, 100))),
    ({"type": "integer", "description": "Radius of the search area (between 1 and 6)"}, ("range", (1, 6))),
    ({"type": "integer", "description": "票据数。\n- 取值范围:1~99998,建议值为1。"}, ("range", (1, 99998))),
    ({"type": "integer", "description": "最大值为1,000。"}, ("range", (None, None))),
    ({"type": "number", "description": "Confidence, typically between 0.90 and 0.99."},
     ("range", (None, None))),
])
def test_constraints_from_description_formats(prop, check):
    c = sm.infer_constraint(prop)
    assert c is not None
    what, expected = check
    if what == "fmt":
        assert c.fmt == expected and c.mentions_format
    elif what == "enum":
        assert c.enum == expected
    else:
        assert (c.minimum, c.maximum) == expected


def test_open_lists_are_no_enum():
    assert sm.infer_constraint({"type": "string",
                                "description": "Options include 'Vegan', 'Vegetarian', etc."}) is None
    assert sm.infer_constraint({"type": "string", "description": "类型,取值:A、B等。"}) is None
    # clean tokens, but an open list all the same
    assert sm.infer_constraint({"type": "string", "description": "状态,取值:RUNNING、STOPPED、etc."}) is None
    assert sm.infer_constraint({"type": "string", "description": "状态,取值:RUNNING、STOPPED。"}).enum == \
        ("RUNNING", "STOPPED")


def test_passes():
    integer = sm.infer_constraint({"type": "integer", "minimum": 15, "maximum": 240})
    assert sm.passes(integer, 90) and sm.passes(integer, 240)
    assert not sm.passes(integer, 241) and not sm.passes(integer, 90.5) and not sm.passes(integer, "90")
    assert not sm.passes(integer, True)
    date = sm.infer_constraint({"type": "string", "format": "date"})
    assert sm.passes(date, "2024-02-29") and not sm.passes(date, "2023-02-29")
    assert not sm.passes(date, "2024/02/29")
    time = sm.infer_constraint({"type": "string", "description": "at HH:MM"})
    assert sm.passes(time, "23:59") and not sm.passes(time, "24:00")
    enum = sm.infer_constraint({"type": "string", "enum": ["low", "high"]})
    assert sm.passes(enum, "low") and not sm.passes(enum, "Low")
    email = sm.infer_constraint({"type": "string", "description": "email address"})
    assert sm.passes(email, "ana@example.org") and not sm.passes(email, "ana.example.org")
    code = sm.infer_constraint({"type": "string", "description": "ISO 4217 currency code"})
    assert sm.passes(code, "EUR") and not sm.passes(code, "Euro")


# ---- mutation --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("prop, value", [
    ({"type": "integer", "minimum": 15, "maximum": 240}, 90),
    ({"type": "integer"}, 4471),
    ({"type": "number"}, 1.75),
    ({"type": "string", "enum": ["A1", "B2", "C3"]}, "B2"),
    ({"type": "string", "format": "date"}, "2024-03-05"),
    ({"type": "string", "description": "at HH:MM"}, "14:30"),
    ({"type": "string", "description": "格式为yyyy-MM-dd HH:mm:ss"}, "2024-03-05 14:30:00"),
    ({"type": "string", "description": "email address"}, "ana@example.org"),
    ({"type": "string", "description": "URL of the page"}, "https://example.org/a"),
    ({"type": "string", "description": "3-letter airport code"}, "LIS"),
    ({"type": "string", "pattern": "^[A-Z]{2}\\d{4}$"}, "AB1234"),
])
@pytest.mark.parametrize("lang", sm.LANGS)
def test_obvious_invalid_values_fail_the_constraint(prop, value, lang):
    c = sm.infer_constraint(prop)
    assert sm.passes(c, value)
    for seed in range(5):
        text = sm.invalid_value(c, value, "obvious", lang, rng(str(seed)))
        assert text and text != sm.value_text(value)
        assert not sm.passes(c, sm.read_value(text, c)), text


ABSENCE = re.compile(r"^(?:n/?a|none|null|nil|unknown|tbd|-+|无|未知|待定|暂无.*|不知道|没有)$",
                     re.IGNORECASE)


@pytest.mark.parametrize("prop, value", [
    ({"type": "integer", "minimum": 1, "maximum": 9}, 5), ({"type": "number"}, 2.5),
    ({"type": "string", "enum": ["A1", "B2"]}, "A1"), ({"type": "string", "description": "URL"}, "https://a.org"),
    ({"type": "string", "pattern": "^[A-Z]{2}$"}, "AB"),
])
def test_wrong_values_never_read_as_an_absent_value(prop, value):
    c = sm.infer_constraint(prop)
    for lang in sm.LANGS:
        for category in sm.CATEGORIES:
            for text in sm.invalid_candidates(c, value, category, lang):
                assert not ABSENCE.match(text.strip()), text  # "N/A" or 无 reads as missing, not as wrong


def test_gold_calls_must_match_the_declared_types():
    folder = tool("DeleteFolder", {"CategoryId": {"type": "integer", "description": "类目的ID。"}})
    question = sample("deepexi", "30", "删除类目时需要提供类目的ID吗?",
                      [("DeleteFolder", {"CategoryId": "类目的ID"})], [folder])
    assert sm.single_call(question) is None
    assert sm.build_verbatim(question, "zh", rng()) is None
    unknown = meeting()
    unknown["messages"][1]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"title": "Budget review", "date": "2024-03-05", "duration": 90, "colour": "B2"})
    assert sm.single_call(unknown) is None  # a key the schema does not define
    assert sm.typed({"duration": 90, "room": "B2"}, MEETING["function"]["parameters"])
    assert not sm.typed({"duration": "90"}, MEETING["function"]["parameters"])


VIDEO = tool("DetectVideoObject", {"VideoURL": {"type": "string", "description": "视频文件URL地址。"},
                                   "Interval": {"type": "integer", "description": "抽帧间隔。"}})


@pytest.mark.parametrize("user, value, names", [
    ("视频URL地址是什么？", "视频URL地址", True),  # a question about the API: the phrase of its description
    ("删除工作空间ID为123456下的视频。", "工作空间ID", True),  # used as a label, the value comes after it
    ("我想检测视频的URL地址里的车辆。", "视频的URL地址", True),  # spelled out of the description
    ('检测URL为"视频地址"的视频。', "视频地址", False),  # quoted: a value the request gives
    ("查一下创建人为张三的视频？", "张三", False),  # no generic noun at its end
    ("检测编号为视频2号的视频。", "视频2号", False),  # a digit: a concrete value
    ("Detect objects at the video URL address.", "the video URL address", False),  # not Chinese
])
def test_a_gold_value_that_names_its_parameter(user, value, names):
    assert sm.names_parameter(value, user, "视频文件URL地址。") is names
    s = sample("deepexi", "v1", user, [("DetectVideoObject", {"VideoURL": value, "Interval": 5})], [VIDEO])
    s["messages"][0]["content"] += " 抽帧间隔5秒。"
    assert (sm.single_call(s) is None) is names
    assert (sm.build_verbatim(s, "zh", rng()) is None) is names


@pytest.mark.parametrize("empty", ["", "  ", None])
def test_a_gold_call_with_an_empty_value_gives_no_value(empty):
    s = sample("deepexi", "v2", "检测视频的车辆,抽帧间隔5秒。",
               [("DetectVideoObject", {"VideoURL": empty, "Interval": 5})], [VIDEO])
    schema = VIDEO["function"]["parameters"]
    assert not sm.gives_values(s["messages"][0]["content"], {"VideoURL": empty}, schema)
    assert sm.single_call(s) is None and sm.build_verbatim(s, "zh", rng()) is None
    assert sm.build_missing(s, "zh", rng(), subkind="single") is None  # the empty value is missing as well


def test_near_miss_reasoning_names_why_the_value_is_valid():
    core = sm.build_nearmiss(meeting(), "en", rng())
    text = sm.dress(core, split="train", rng=rng()).reasoning
    reason = core.variant["reason"]
    assert {"leap": "leap year", "maximum": "maximum", "minimum": "minimum", "enum": "allowed values",
            "time": "time of day"}[reason] in text
    assert "out of bounds" not in text or reason in ("maximum", "minimum")


def test_format_mismatch_needs_a_format_the_definition_names():
    date = sm.infer_constraint({"type": "string", "format": "date"})
    for lang in sm.LANGS:
        text = sm.invalid_value(date, "2024-03-05", "format", lang, rng())
        assert text and "2024" in text and not sm.passes(date, text)
    time = sm.infer_constraint({"type": "string", "description": "at HH:MM"})
    assert not sm.passes(time, sm.invalid_value(time, "14:30", "format", "en", rng()))
    assert sm.invalid_value(sm.infer_constraint({"type": "integer"}), 5, "format", "en", rng()) is None
    enum = sm.infer_constraint({"type": "string", "enum": ["A1", "B2"]})
    assert sm.invalid_value(enum, "B2", "format", "zh", rng()) is None


def test_vague_values_are_vague_words_of_the_language():
    integer = sm.infer_constraint({"type": "integer"})
    zh = sm.invalid_value(integer, 4471, "vague", "zh", rng())
    assert zh in sm.VAGUE["number"]["zh"] and not sm.passes(integer, sm.read_value(zh, integer))
    assert sm.invalid_value(integer, 4471, "vague", "en", rng()) in sm.VAGUE["number"]["en"]
    email = sm.infer_constraint({"type": "string", "description": "email address"})
    assert sm.invalid_value(email, "ana@example.org", "vague", "en", rng()) is None


@pytest.mark.parametrize("prop, value, expected, reason", [
    ({"type": "string", "format": "date"}, "2024-03-05", "2024-02-29", "leap"),
    ({"type": "integer", "minimum": 15, "maximum": 240}, 90, 240, "maximum"),
    ({"type": "integer", "description": "页码,取值范围:1~100。"}, 100, 1, "minimum"),
    ({"type": "string", "description": "at HH:MM"}, "14:30", "23:59", "time"),
])
def test_near_miss_values_pass(prop, value, expected, reason):
    c = sm.infer_constraint(prop)
    got = sm.nearmiss_value(c, value)
    assert got == (expected, reason) and sm.passes(c, expected)


def test_near_miss_enum_is_the_exact_member_given():
    # hand check, round 3: swapping an option for another left requests saying the opposite (Java for a
    # snippet of Python), so an enum near miss keeps the option the request gives, spelled exactly as listed
    c = sm.infer_constraint({"type": "string", "enum": ["A1", "B2", "C3"]})
    assert sm.nearmiss_value(c, "B2") == ("B2", "enum")


def test_replace_once_respects_word_edges_and_counts():
    assert sm.replace_once("for 3 people, room 13", "3", "five") == "for five people, room 13"
    assert sm.replace_once("3 and 3", "3", "x") is None
    assert sm.replace_once("nothing here", "3", "x") is None
    assert sm.replace_once("在工作空间4471里", "4471", "十几") == "在工作空间十几里"


def test_cut_values_removes_each_value_with_its_quotes():
    text = 'Please schedule "Budget review" on 2024-03-05 for 90 minutes.'
    assert sm.cut_values(text, ["Budget review", "90"]) == "Please schedule on 2024-03-05 for minutes."
    assert sm.cut_values(text, ["absent"]) is None


def test_copied_helpers_agree_with_build_ace_probe():
    for s in (meeting(), jobs(), trip(), catalog()):
        assert sm.first_step(s) == bp.first_step(s)
        user, calls = sm.first_step(s)
        assert sm.grounded(user, calls) == bp.grounded(user, calls)
    for value, text in (("3", "for 3 people, room 13"), ("Oslo", "oslo and OSLO"), ("北京", "北京市")):
        ours = [m.span() for m in sm.occurrences(value, text)]
        assert ours == [m.span() for m in bp._occurrences(value, text)]


# ---- builders -------------------------------------------------------------------------------------------


def test_build_incorrect_obvious_mutates_one_valid_top_level_value():
    core = sm.build_incorrect(meeting(), "en", rng(), category="obvious")
    assert core.kind == "incorrect" and core.api == "create_meeting" and len(core.invalid) == 1
    bad = core.invalid[0]
    assert bad.param in ("date", "duration", "room")
    user = core.history[-1][1]
    assert len(sm.occurrences(bad.value, user)) == 1
    original = {"date": "2024-03-05", "duration": "90", "room": "B2"}[bad.param]
    assert original not in user or bad.value.startswith(original)
    assert "Budget review" in user


def test_build_incorrect_format_and_vague_categories():
    core = sm.build_incorrect(meeting(), "en", rng(), category="format")
    assert core.invalid[0].param == "date" and core.variant["category"] == "format"
    core = sm.build_incorrect(jobs(), "zh", rng(), category="vague")
    assert core.invalid[0].value in {w for words in sm.VAGUE.values() for w in words["zh"]}
    assert sm.build_incorrect(catalog(), "zh", rng(), category="obvious") is None  # free strings only


def test_build_incorrect_two_values():
    core = sm.build_incorrect(meeting(), "en", rng(), category="obvious", two=True)
    assert len(core.invalid) == 2 and len({b.param for b in core.invalid}) == 2
    user = core.history[-1][1]
    assert all(len(sm.occurrences(b.value, user)) == 1 for b in core.invalid)


def test_build_incorrect_skips_values_that_already_break_their_definition():
    s = meeting()
    s["messages"][0]["content"] = s["messages"][0]["content"].replace("90", "300")
    args = json.loads(s["messages"][1]["tool_calls"][0]["function"]["arguments"])
    args.update(duration=300, room="Z9", date="2024-02-30")
    s["messages"][0]["content"] = 'Please schedule "Budget review" on 2024-02-30 for 300 minutes in room Z9.'
    s["messages"][1]["tool_calls"][0]["function"]["arguments"] = json.dumps(args)
    assert sm.build_incorrect(s, "en", rng(), category="obvious") is None


def test_build_incorrect_mutates_an_array_item_under_its_top_level_name():
    tags = tool("tag_photo", {"labels": {"type": "array", "items": {"type": "string",
                                                                    "enum": ["beach", "city", "forest"]}}},
                ["labels"])
    s = sample("hermes", "4", "Tag this photo with beach and forest please.",
               [("tag_photo", {"labels": ["beach", "forest"]})], [tags])
    core = sm.build_incorrect(s, "en", rng(), category="obvious")
    assert core.invalid[0].param == "labels"
    assert core.invalid[0].value not in ("beach", "forest")


@pytest.mark.parametrize("subkind, count", [("single", 1), ("two", 2)])
def test_build_missing_cuts_required_values(subkind, count):
    core = sm.build_missing(meeting(), "en", rng(), subkind=subkind)
    assert core.kind == "missing" and len(core.missing) == count
    assert set(core.missing) <= {"title", "date", "duration"}
    user = core.history[-1][1]
    values = {"title": "Budget review", "date": "2024-03-05", "duration": "90"}
    for name in core.missing:
        assert values[name].casefold() not in user.casefold()
    for name in set(values) - set(core.missing):
        assert values[name] in user


def test_build_missing_object_names_only_the_top_level_parameter():
    core = sm.build_missing(trip(), "en", rng(), subkind="object")
    assert core.missing == ["budget"] and core.nested == "budget"
    user = core.history[-1][1]
    assert "1500" not in user and "EUR" not in user and "Lisbon" in user


def test_build_missing_optional_cut_is_not_named():
    core = sm.build_missing(meeting(), "en", rng(), subkind="optional")
    assert core.optional_cut == ["room"] and "room" not in core.missing and len(core.missing) == 1
    assert "B2" not in core.history[-1][1]


def test_build_missing_on_a_schema_without_required_uses_the_gold_call():
    core = sm.build_missing(jobs(), "zh", rng(), subkind="single")
    called = next(t for t in core.tools if t["function"]["name"] == "DescribeJobQueue")
    required = called["function"]["parameters"]["required"]
    assert set(required) == {"ProjectId", "Status", "PageSize", "StartDate"}
    assert core.missing[0] in required
    core = sm.build_missing(jobs("9"), "zh", rng("o"), subkind="optional")
    called = next(t for t in core.tools if t["function"]["name"] == "DescribeJobQueue")
    assert core.optional_cut[0] not in called["function"]["parameters"]["required"]
    other = next(t for t in core.tools if t["function"]["name"] == "CreateDataCatalog")
    assert "required" not in other["function"]["parameters"]  # only the called function is given one


# ---- revision 1: the training constructions (docs/m8-预登记.md, 修订 1) ---------------------------------

MAIL = tool("send_invite", {"to": {"type": "string", "description": "The guest's email address"},
                            "day": {"type": "string", "format": "date", "description": "Day of the party"}},
            ["to", "day"])


def invite(sid: str = "m1") -> dict:
    return sample("toolace", sid, "Send the invite to ana@example.org for 2024-05-02.",
                  [("send_invite", {"to": "ana@example.org", "day": "2024-05-02"})], [MAIL])


def test_build_incorrect_on_an_optional_parameter():
    # at least a fifth of the training incorrect items put the wrong value on an optional parameter (FT5
    # called with 0084's optional e-mail: "The email is optional, so it's fine to include it")
    for seed in range(6):
        core = sm.build_incorrect(meeting(), "en", rng(str(seed)), category="obvious", optional=True)
        assert core.invalid[0].param == "room" and core.variant["optional"] is True
        assert "room" not in core.required
    assert "optional" not in sm.build_incorrect(meeting(), "en", rng(), category="obvious").variant
    assert sm.build_incorrect(jobs(), "zh", rng(), category="obvious", optional=True) is None  # all required


def test_build_incorrect_with_a_format_the_description_alone_implies():
    # half of the training format items: a format no schema keyword names, only the description (an e-mail
    # address, a URL); FT5 got none of dev set A's seven format items right
    for seed in range(6):
        core = sm.build_incorrect(invite(), "en", rng(str(seed)), category="format", described=True)
        assert core.invalid[0].param == "to" and core.variant["described"] is True
        assert core.invalid[0].value == "ana at example dot org"
    assert sm.build_incorrect(meeting(), "en", rng(), category="format", described=True) is None  # schema's


MOVIES = tool("search_movie", {
    "title": {"type": "string", "description": "Title of the movie"},
    "year": {"type": "integer", "minimum": 1900, "maximum": 2030, "description": "Release year"},
    "genre": {"type": "string", "description": "Genre of the movie"},
}, ["title"])


def forgotten_title(sid: str = "f1") -> dict:
    """A record whose gold call leaves out a parameter its function requires: the user cannot recall the
    movie's title (glaive_v2 15215 of the revision's first build, its record 448)."""
    user = ("I am looking for a movie but I can't remember the exact title. It was released in 2015 and it's "
            "a drama.")
    return sample("glaive_v2", sid, user, [("search_movie", {"year": 2015, "genre": "drama"})], [MOVIES])


def test_build_incorrect_complete_takes_no_gold_call_without_a_required_value():
    # the review of revision 1: a wrong value in a request that also lacks a required value is a priority
    # case, which the prompt's order decides; record 448 answered one with the wrong-value sentence under a
    # missing-first prompt. A training incorrect core (complete) needs every required parameter in the gold
    # call; the development path builds as the locked build did
    core = sm.build_incorrect(forgotten_title(), "en", rng(), category="obvious")
    assert core is not None and core.invalid[0].param == "year" and core.required == ["title"]
    assert sm.build_incorrect(forgotten_title(), "en", rng(), category="obvious", complete=True) is None
    assert sm.build_incorrect(meeting(), "en", rng(), category="obvious", complete=True) == \
        sm.build_incorrect(meeting(), "en", rng(), category="obvious")
    assert sm.build_incorrect(jobs(), "zh", rng(), category="vague", complete=True) is not None  # deepexi


def test_reasoning_problems_flag_a_required_value_not_given_outside_the_kinds_that_cut_one():
    # a required parameter "not given" belongs to a missing, priority or ask item: an incorrect item with one
    # (record 448, under either order) or a call made without it teaches the wrong answer
    core = sm.build_incorrect(forgotten_title(), "en", rng(), category="obvious")
    for order in sm.ORDERS:
        item = sm.dress(core, split="train", rng=rng(order), order=order)
        assert "- title (required): not given." in item.reasoning.split("\n")
        assert sm.reasoning_problems(item) == ["a required parameter not given in an incorrect item"]
    call = sm.dress(sm.build_verbatim(forgotten_title(), "en", rng()), split="train", rng=rng())
    assert sm.reasoning_problems(call) == ["a required parameter not given in a valid_verbatim item"]
    for core in all_cores():  # the kinds that cut a value say "not given" of it
        assert sm.reasoning_problems(sm.dress(core, split="train", rng=rng(core.kind))) == [], core.kind


def test_build_missing_cuts_the_keys_asked_for():
    for key in ("title", "date", "duration"):
        core = sm.build_missing(meeting(), "en", rng(), subkind="single", keys=[key])
        assert core.missing == [key]
    assert sm.build_missing(meeting(), "en", rng(), subkind="single", keys=["room"]) is None  # optional


def test_cores_record_what_the_latest_request_gives_and_what_was_cut():
    gold = {"title": "Budget review", "date": "2024-03-05", "duration": 90, "room": "B2"}
    incorrect = sm.build_incorrect(meeting(), "en", rng(), category="obvious")
    bad = incorrect.invalid[0]
    assert incorrect.given == {**gold, bad.param: bad.value} and incorrect.cut == ()
    missing = sm.build_missing(meeting(), "en", rng(), subkind="optional")
    assert missing.given == {k: v for k, v in gold.items() if k not in missing.missing + missing.optional_cut}
    assert len(missing.cut) == 2 and all(text not in missing.history[-1][1] for text in missing.cut)
    priority = sm.build_priority(jobs(), "zh", rng())
    assert set(priority.given) == {"ProjectId", "Status", "PageSize", "StartDate"} - set(priority.missing)
    assert priority.given[priority.invalid[0].param] == priority.invalid[0].value
    verbatim = sm.build_verbatim(catalog(), "zh", rng(), require_cjk=True)
    assert verbatim.given == {"Name": "销售数据", "Comment": "季度汇总"}
    change = sm.build_change(catalog(), "zh", rng(), bank={"Name": ["财务", "人事"]}, packed=True)
    assert change.given == change.calls[0][1]  # the latest value


def test_an_irrelevant_twin_pairs_the_cut_request_with_unrelated_functions():
    # a twin pair from one record (missing / irrelevant): the same request, once with the function it needs
    # and once with another record's unrelated functions only
    cut = sm.build_missing(jobs(), "zh", rng(), subkind="single")
    other = sample("deepexi", "9", "新建一个名为财务的目录。", [("CreateDataCatalog", {"Name": "财务"})],
                   [CATALOG])
    core = sm.build_irrelevant(jobs(), "zh", rng(), tools_from=other, user=cut.history[-1][1])
    assert core.history == cut.history and core.records == ("7", "9") and core.variant["pairing"]
    assert core.originals == (jobs()["messages"][0]["content"],)


CLOCK = tool("get_time", {"zone": {"type": "string", "description": "Time zone"}}, ["zone"])


def test_a_refusal_in_context_keeps_the_problem_in_the_latest_turn():
    # about a tenth of the training refusal items answer the latest of several turns: another record's
    # request and the call made for it come first, as real turns or packed into one message
    core = sm.build_missing(meeting(), "en", rng(), subkind="single")
    earlier = sample("toolace", "w1", "What time is it in zone CET?", [("get_time", {"zone": "CET"})],
                     [CLOCK])
    for packed in (False, True):
        wrapped = sm.with_context(core, earlier, rng(), packed=packed)
        assert wrapped.history[:2] == [("user", "What time is it in zone CET?"),
                                       ("assistant", [("get_time", {"zone": "CET"})])]
        assert wrapped.history[2:] == core.history and wrapped.missing == core.missing
        assert "get_time" in {t["function"]["name"] for t in wrapped.tools} and wrapped.records == ("1", "w1")
        assert wrapped.variant["context"] == ("packed" if packed else "turns") and wrapped.packed is packed
        assert wrapped.originals == (*core.originals, "What time is it in zone CET?")
        item = sm.dress(wrapped, split="train", rng=rng())
        assert item.expected["case"] == "missing" and item.expected["params"] == core.missing
        assert item.context_turns == (0 if packed else 1)
        assert sm.check_training_record(sm.training_record(item)) == []
    shared = tool("find_slot", {"title": {"type": "string", "description": "Title"}}, ["title"])
    same_name = sample("toolace", "w2", "Find the slot called Kickoff.",
                       [("find_slot", {"title": "Kickoff"})], [shared])
    assert sm.with_context(core, same_name, rng(), packed=False) is None  # a parameter name in common
    assert sm.with_context(core, meeting("toolace", "w3"), rng(), packed=False) is None  # the same function
    flight = sample("toolace", "5", "Book me a flight to Rome.", [], [WEATHER],
                    text="None of the provided functions can be used to book a flight.")
    irrelevant = sm.build_irrelevant(flight, "en", rng())
    assert sm.with_context(irrelevant, earlier, rng(), packed=False).kind == "irrelevant"
    flights = tool("search_flights", {"to": {"type": "string", "description": "Destination"}}, ["to"])
    serving = sample("toolace", "w4", "Find flights to Paris.", [("search_flights", {"to": "Paris"})],
                     [flights])
    assert sm.with_context(irrelevant, serving, rng(), packed=False) is None  # it could serve the request


def test_no_refusal_in_context_follows_a_function_that_could_serve_the_latest_request():
    # the review of revision 1: the revision says the earlier turn's function cannot serve the latest request,
    # and with_context checked that for the irrelevant cores alone; every refusal kind now (unrelated,
    # shares_content_word, serves_anything): reserve_room's words stand in the meeting request
    rooms = tool("reserve_room", {"space": {"type": "string", "description": "Which space"}}, ["space"])
    serving = sample("toolace", "w5", "Reserve the space called Atlas.",
                     [("reserve_room", {"space": "Atlas"})], [rooms])
    earlier = sample("toolace", "w1", "What time is it in zone CET?", [("get_time", {"zone": "CET"})],
                     [CLOCK])
    for core in (sm.build_missing(meeting(), "en", rng(), subkind="single"),
                 sm.build_incorrect(meeting(), "en", rng(), category="obvious")):
        assert sm.with_context(core, serving, rng(), packed=False) is None, core.kind
        assert sm.with_context(core, earlier, rng(), packed=False) is not None, core.kind


@pytest.mark.parametrize("text, canned", [
    ("Can you order a pizza for me?", True), ("Book me a flight to Rome.", True),
    ("I need to reserve a table for two.", True), ("你能帮我订一张去纽约的机票吗?", True),
    ("What's the weather in Oslo?", False), ("帮我查一下北京的天气。", False),
    ("Recommend a good book about whales.", False),
])
def test_canned_requests_book_a_flight_order_a_pizza_or_book_something(text, canned):
    # the records' own refusals the analysis found in 140 of 156 training irrelevant items
    assert sm.canned_request(text) is canned


def test_the_cell_deck_covers_every_base_cell_once_a_round():
    deck = sm.cell_deck(random.Random("d"), rounds=4)
    assert len(sm.BASE_CELLS) == 36 and len(sm.EXTRA_CELLS) == 90
    size = len(deck) // 4
    for r in range(4):
        chunk = deck[r * size:(r + 1) * size]
        assert Counter(c for c in chunk if c.style in sm.BASE_STYLES) == Counter(sm.BASE_CELLS)
    assert sum(c.style in sm.EXTRA_STYLES for c in deck) / len(deck) == pytest.approx(sm.EXTRA_SHARE)
    assert sm.cell_deck(random.Random("d"), rounds=4) == deck


def test_a_training_prompt_takes_its_cells_style_and_a_family_of_its_shape():
    core = sm.build_incorrect(meeting(), "en", rng(), category="obvious")
    for cell in [*sm.BASE_CELLS[::5], sm.Cell("lenticular", "name_first", "trailing")]:
        item = sm.dress(core, split="train", rng=rng(str(cell)), cell=cell)
        assert item.rules.style == cell.style
        assert sm.family_shape(item.rules.families["incorrect"]) == (cell.order, cell.position)
    priority = sm.build_priority(jobs(), "zh", rng())
    for order in sm.ORDERS:
        cell = sm.Cell("caps", "filler_first", "leading")
        item = sm.dress(priority, split="train", rng=rng(order), order=order, cell=cell)
        assert sm.family_shape(item.rules.families[item.expected["case"]]) == ("filler_first", "leading")
    styles = Counter(sm.dress(core, split="train", rng=rng(str(seed))).rules.style for seed in range(400))
    extra = sum(count for style, count in styles.items() if style in sm.EXTRA_STYLES) / 400
    assert 0.12 <= extra <= 0.28  # no cell: a base style four times in five


def test_build_irrelevant_from_no_call_answers():
    toolace = sample("toolace", "5", "Book me a flight to Rome.", [], [WEATHER],
                     text="None of the provided functions can be used to book a flight.")
    assert sm.build_irrelevant(toolace, "en", rng()).kind == "irrelevant"
    glaive = sample("glaive_v2", "6", "Can you order a pizza for me?", [], [WEATHER],
                    text="I'm sorry, but I can't order pizza. My capabilities are limited to forecasts.")
    assert sm.build_irrelevant(glaive, "zh", rng()).history == [("user", "Can you order a pizza for me?")]
    asking = sample("glaive_v2", "7", "What's the forecast?", [], [WEATHER],
                    text="Sure, could you please provide the city?")
    assert sm.build_irrelevant(asking, "en", rng()) is None
    sorry = sample("glaive_v2", "7b", "What's the forecast?", [], [WEATHER],
                   text="I'm sorry, I cannot tell without a place. Could you please provide the city?")
    assert sm.build_irrelevant(sorry, "en", rng()) is None  # it asks for a value: no irrelevant request
    sorry_zh = sample("glaive_zh", "7c", "明天天气怎么样?", [], [WEATHER],
                      text="抱歉,我无法确定地点。请提供您所在的城市。")
    assert sm.build_irrelevant(sorry_zh, "zh", rng()) is None
    zh = sample("glaive_zh", "8", "你能帮我订一张去纽约的机票吗?", [], [WEATHER],
                text="很抱歉,我无法帮您预订机票。我的功能仅限于查询天气。")
    assert sm.build_irrelevant(zh, "zh", rng()) is not None
    assert sm.build_irrelevant(meeting(), "en", rng()) is None  # a record that calls


def test_build_irrelevant_cross_pairing_needs_unrelated_functions():
    other = sample("deepexi", "9", "新建一个名为财务的目录。", [("CreateDataCatalog", {"Name": "财务"})],
                   [CATALOG])
    core = sm.build_irrelevant(jobs(), "zh", rng(), tools_from=other)
    assert core is not None and [t["function"]["name"] for t in core.tools] == ["CreateDataCatalog"]
    assert core.records == ("7", "9")
    related = sample("deepexi", "10", "查询任务队列。", [("ListJobQueues", {})],
                     [tool("ListJobQueues", {"Size": {"type": "integer"}})])
    assert sm.build_irrelevant(jobs(), "zh", rng(), tools_from=related) is None


def test_cross_pairing_refuses_functions_sharing_chinese_terms_with_the_request():
    spaces = tool("ListSpaces", {"Size": {"type": "integer"}}, description="获取工作空间列表。")
    other = sample("deepexi", "11", "列出全部。", [("ListSpaces", {})], [spaces])
    assert sm.build_irrelevant(jobs(), "zh", rng(), tools_from=other) is None  # 工作空间 in both
    generic = tool("ListMailBoxes", {"Size": {"type": "integer"}}, description="查询邮箱的列表信息。")
    other = sample("deepexi", "12", "列出全部。", [("ListMailBoxes", {})], [generic])
    assert sm.build_irrelevant(jobs(), "zh", rng(), tools_from=other) is not None  # 查询 is no domain term


def test_a_pairing_shows_no_function_that_could_serve_any_request():
    # hand check, round 7 of this fix round (sdev-a-0024): a request on Formula 1 lap records was paired with
    # "Search News", which might serve it
    laps = tool("get_laps", {"season": {"type": "string"}}, ["season"])
    request = sample("glaive_v2", "f1", "Tell me about the fastest laps in this year's Formula 1 races.",
                     [("get_laps", {"season": "2024"})], [laps])
    news = tool("Search News", {"query": {"type": "string"}}, ["query"], description="Find news articles.")
    other = sample("toolace", "n1", "Any news on the election?", [("Search News", {"query": "election"})],
                   [news])
    assert sm.build_irrelevant(request, "en", rng(), tools_from=other) is None
    stock = tool("check_stock", {"item": {"type": "string"}}, ["item"], description="Check an item's stock.")
    other = sample("toolace", "s1", "Is the blue mug in stock?", [("check_stock", {"item": "blue mug"})],
                   [stock])
    assert sm.build_irrelevant(request, "en", rng(), tools_from=other) is not None


def test_build_priority_and_its_answer_follows_the_order():
    core = sm.build_priority(meeting(), "en", rng())
    assert core.kind == "priority" and len(core.invalid) == 1 and len(core.missing) == 1
    assert core.invalid[0].param != core.missing[0]
    for order, case in (("incorrect_first", "incorrect"), ("missing_first", "missing")):
        item = sm.dress(core, split="train", rng=rng(order), order=order)
        assert item.expected["case"] == case
        family = item.rules.families[case]
        assert item.answer == sm.expected_sentence(item)
        assert all(part in item.answer for part in sm.fixed_parts(family, item.rules.style, "en"))


def test_build_ask_answers_with_a_call_to_the_ask_function():
    core = sm.build_ask(meeting(), "en", rng())
    item = sm.dress(core, split="train", rng=rng())
    ask = item.ask
    assert ask is not None and ask.name in item.system
    assert any(t["function"]["name"] == ask.name for t in item.tools)
    [(name, arguments)] = rfd.parse_calls(item.answer, item.call_format)
    assert name == ask.name
    assert arguments == {ask.function_key: "create_meeting", ask.missing_key: core.missing}


def test_build_verbatim_keeps_chinese_values():
    core = sm.build_verbatim(catalog(), "zh", rng(), require_cjk=True)
    item = sm.dress(core, split="train", rng=rng())
    assert rfd.parse_calls(item.answer, item.call_format) == [("CreateDataCatalog",
                                                               {"Name": "销售数据", "Comment": "季度汇总"})]
    assert item.expected == {"type": "call",
                             "calls": [{"name": "CreateDataCatalog",
                                        "arguments": {"Name": "销售数据", "Comment": "季度汇总"}}],
                             "verbatim": ["销售数据", "季度汇总"]}
    assert sm.build_verbatim(meeting(), "en", rng(), require_cjk=True) is None


@pytest.mark.parametrize("packed", [True, False])
def test_build_change_uses_the_latest_value(packed):
    bank = {"Name": ["财务", "人事"]}
    core = sm.build_change(catalog(), "zh", rng(), bank=bank, packed=packed)
    assert core.kind == "valid_multiturn" and core.variant["mode"] == "change"
    [(_, arguments)] = core.calls
    note = core.variant
    assert arguments[note["param"]] == note["new"] != note["old"]
    item = sm.dress(core, split="train", rng=rng())
    users = [m for m in item.messages if m["role"] == "user"]
    if packed:
        assert len(users) == 1 and len(item.messages) == 2 and item.context_turns == 0
        assert note["old"] in users[0]["content"] and note["new"] in users[0]["content"]
    else:
        assert [m["role"] for m in item.messages] == ["system", "user", "assistant", "user"]
        assert item.context_turns == 1
        earlier = rfd.parse_calls(item.messages[2]["content"], item.call_format)
        assert earlier[0][1][note["param"]] == note["old"]
    assert rfd.parse_calls(item.answer, item.call_format)[0][1][note["param"]] == note["new"]


def test_build_switch_answers_the_second_request():
    core = sm.build_switch(trip("1"), meeting("glaive_v2", "2"), "en", rng(), packed=False)
    assert core.calls[0][0] == "create_meeting" and core.records == ("2", "1")
    assert {t["function"]["name"] for t in core.tools} >= {"plan_trip", "create_meeting"}
    item = sm.dress(core, split="train", rng=rng())
    assert rfd.parse_calls(item.messages[2]["content"], item.call_format)[0][0] == "plan_trip"
    assert rfd.parse_calls(item.answer, item.call_format)[0][0] == "create_meeting"


def test_build_nearmiss_calls_with_the_boundary_value():
    core = sm.build_nearmiss(meeting(), "en", rng())
    [(_, arguments)] = core.calls
    param, value = core.variant["param"], core.variant["value"]
    assert arguments[param] == value and sm.passes(sm.infer_constraint(MEETING["function"]["parameters"]
                                                                        ["properties"][param]), value)
    assert sm.value_text(value) in core.history[-1][1]
    item = sm.dress(core, split="train", rng=rng())
    assert item.rules is not None  # near misses always come with the refusal rules they must not trigger


# ---- dressing, formats, reasoning -------------------------------------------------------------------------


def all_cores() -> list:
    bank = {"Name": ["财务", "人事"]}
    return [
        sm.build_incorrect(meeting(), "en", rng(), category="obvious"),
        sm.build_incorrect(jobs(), "zh", rng(), category="vague"),
        sm.build_incorrect(meeting(), "en", rng(), category="obvious", two=True),
        sm.build_missing(meeting(), "en", rng(), subkind="single"),
        sm.build_missing(trip(), "zh", rng(), subkind="object"),
        sm.build_missing(meeting(), "en", rng(), subkind="optional"),
        sm.build_irrelevant(sample("toolace", "5", "Book me a flight to Rome.", [], [WEATHER],
                                   text="None of the provided functions can be used to book a flight."),
                            "zh", rng()),
        sm.build_priority(jobs(), "zh", rng()),
        sm.build_ask(meeting(), "en", rng()),
        sm.build_verbatim(catalog(), "zh", rng(), require_cjk=True),
        sm.build_change(catalog(), "zh", rng(), bank=bank, packed=True),
        sm.build_switch(trip("1"), meeting("glaive_v2", "2"), "en", rng(), packed=False),
        sm.build_nearmiss(meeting(), "en", rng()),
    ]


def test_dress_is_deterministic_for_a_seed():
    for core in all_cores():
        assert core is not None
        a = sm.dress(core, split="b", rng=random.Random("x"))
        b = sm.dress(core, split="b", rng=random.Random("x"))
        assert a == b


@pytest.mark.parametrize("split", sm.SPLITS)
def test_dressed_items_use_the_families_of_their_split(split):
    for core in all_cores():
        item = sm.dress(core, split=split, rng=random.Random(split))
        if item.rules is not None:
            assert {f.split for f in item.rules.families.values()} == {split}
            for family in item.rules.families.values():
                assert sm.prompt_sentence(family, item.rules.style, item.lang) in item.system


def test_dev_items_have_exactly_the_agreed_format():
    for number, core in enumerate(all_cores(), 1):
        if core.kind == "ask":
            continue
        item = sm.dress(core, split="a", rng=rng(str(number)))
        dev = sm.dev_item(item, "a", number)
        assert list(dev) == ["id", "split", "lang", "kind", "messages", "call_format", "family", "expected",
                             "source"]
        assert dev["id"] == f"sdev-a-{number:04d}" and dev["split"] == "a"
        assert dev["messages"][0]["role"] == "system" and dev["messages"][-1]["role"] == "user"
        assert sm.check_dev_item(dev) == [], (core.kind, sm.check_dev_item(dev))
        expected = dev["expected"]
        if expected["type"] == "refusal":
            assert list(expected) == ["type", "case", "fixed", "values", "params", "api"]
        else:
            assert list(expected) == ["type", "calls", "verbatim"]
    bad = sm.dev_item(sm.dress(all_cores()[0], split="a", rng=rng()), "a", 1)
    bad["expected"]["case"] = "other"
    assert sm.check_dev_item(bad)
    assert sm.check_dev_item({**bad, "id": "x"})


@pytest.mark.parametrize("split", ("a", "b"))
def test_a_development_set_writes_each_family_in_one_style(split):
    # the scorer reads a set's sentences family by family (research_agent.eval.special_dev.sentence_families):
    # within a development set a family must give each case one sentence, so it keeps one style
    styles: dict[str, set[str]] = {}
    for core in all_cores():
        if core.kind == "ask":
            continue
        for seed in range(12):
            item = sm.dress(core, split=split, rng=random.Random(f"{split}/{core.kind}/{seed}"))
            if item.rules is None:
                continue
            for case, family in item.rules.families.items():
                if case != "irrelevant":
                    styles.setdefault(family.id, set()).add(item.rules.style)
    assert len(styles) >= 6 and all(len(found) == 1 for found in styles.values()), styles
    for lang in sm.LANGS:
        used = {style for fid, found in styles.items() if sm.FAMILIES[fid].lang == lang for style in found}
        assert used <= set(sm.DEV_STYLES[split, lang])
    assert set(sm.DEV_STYLES[split, "zh"]) | set(sm.DEV_STYLES[split, "en"]) == set(sm.BASE_STYLES)
    core = all_cores()[0]
    train = {sm.dress(core, split="train", rng=random.Random(seed)).rules.style for seed in range(40)}
    assert len(train) > 2  # training writes a family in any style


def test_dev_set_problems_flags_a_family_with_two_sentences_and_a_repeated_id():
    items = [sm.dev_item(sm.dress(core, split="b", rng=rng(str(number))), "b", number)
             for number, core in enumerate(all_cores(), 1) if core.kind != "ask"]
    assert sm.dev_set_problems(items) == []
    refusal = next(i for i in items if i["expected"]["type"] == "refusal"
                   and i["expected"]["case"] == "incorrect")
    twin = json.loads(json.dumps(refusal))
    twin["id"] = "sdev-b-0999"
    twin["expected"]["fixed"] = [part + "!" for part in twin["expected"]["fixed"]]
    assert sm.dev_set_problems([*items, twin])  # one family, one case, two sentences
    assert sm.dev_set_problems([*items, items[0]])  # an id twice


def test_refusal_expectations_match_the_answer():
    for core in all_cores():
        item = sm.dress(core, split="train", rng=rng(core.kind))
        expected = item.expected
        if expected["type"] != "refusal":
            continue
        for value in expected["values"]:
            assert value in item.answer
        for param in expected["params"]:
            assert param in item.answer
        if expected["case"] == "missing":
            assert expected["api"] in item.answer and expected["values"] == []
        if expected["case"] == "missing" and core.nested:
            assert expected["params"] == [core.nested]
        if expected["case"] == "irrelevant":
            assert expected["values"] == expected["params"] == [] and expected["api"] == ""


def test_training_records_and_their_reasoning_rows():
    for index, core in enumerate(all_cores()):
        item = sm.dress(core, split="train", rng=rng(str(index)))
        record = sm.training_record(item)
        assert record["tools"] == [] and record["kind"] == core.kind
        assert record["instruction_lang"] == core.lang
        assert record["messages"][-1] == {"role": "assistant", "content": item.answer}
        assert sm.check_training_record(record) == [], sm.check_training_record(record)
        if item.context_turns:
            assert record["context_turns"] == item.context_turns
        row = sidecar_row(f"special-mix:{index}", item.context_turns, record["messages"][-1],
                          source="template", reasoning=item.reasoning, model=sm.TEMPLATE_MODEL,
                          cache_key="0" * 64, context_equal=True)
        assert check_row(row, "test")


def training_items(seeds: int = 3) -> list[sm.Item]:
    """Every core of all_cores (and a missing one in context) dressed for training under a few seeds."""
    earlier = sample("toolace", "w1", "What time is it in zone CET?", [("get_time", {"zone": "CET"})],
                     [CLOCK])
    framed = sm.with_context(sm.build_missing(meeting(), "en", rng(), subkind="single"), earlier, rng(),
                             packed=False)
    cores = [*all_cores(), framed]
    return [sm.dress(core, split="train", rng=rng(f"{core.kind}/{seed}")) for core in cores
            for seed in range(seeds)]


def test_reasoning_quotes_values_verbatim_and_chinese_only_inside_quotes():
    cjk = re.compile(r"[㐀-鿿]")
    for item in training_items():
        text = item.reasoning
        quoted = re.findall(r'"([^"]*)"', text)
        assert not cjk.search(re.sub(r'"[^"]*"', "", text)), text  # Chinese only inside quotes
        for bad in item.core.invalid:
            assert bad.value in quoted
        if item.core.kind == "valid_verbatim":
            assert "销售数据" in quoted
        assert sm.approx_tokens(text) <= sm.MAX_REASONING_TOKENS and sm.reasoning_problems(item) == []


def test_every_training_reasoning_opens_with_the_same_neutral_line():
    # revision 1, the check-first skeleton: whatever the kind, the reasoning opens with the functions shown
    # and what each does, before anything that decides the case (FT5 kept the old template's opening words,
    # one per kind: "The request fits" opened 10 of its refusal answers, 8 of them calls)
    for item in training_items():
        first = item.reasoning.split("\n")[0]
        assert first.startswith(sm.REASONING_OPENING), first
        for t in item.tools:
            assert sm._n(t["function"]["name"]) in first


def test_a_refusal_reasoning_quotes_its_sentence_maps_every_placeholder_and_writes_the_answer():
    # FT5 copied VALUE_GIVEN into 12 answers of zh caps, paren and corner prompts while its reasoning had the
    # value: the skeleton quotes the sentence as the prompt writes it, maps each placeholder string to its
    # filler with the brackets kept, writes the answer and checks it
    found = Counter()
    for item in training_items(seeds=6):
        if item.expected["type"] != "refusal":
            continue
        case = item.expected["case"]
        found[case] += 1
        family = item.rules.families[case]
        lines = item.reasoning.split("\n")
        assert f'Sentence: "{sm.prompt_sentence(family, item.rules.style, item.lang)}"' in lines
        assert f'Answer: "{item.answer}"' in lines
        marks = sm.placeholders(item.rules.style, item.lang)
        fill = next(line for line in lines if line.startswith("Fill:"))
        opening, closing = sm.STYLE_BRACKETS[item.rules.style]
        for slot in re.findall(r"\{(\w+)\}", family.template):
            names = ", ".join(item.expected["params"])
            filler = {"value": ", ".join(item.expected["values"]), "param": names, "params": names,
                      "api": item.expected["api"]}[slot]
            assert f'"{marks[slot]}" -> "{opening}{filler}{closing}"' in fill, (slot, fill)
        if case == "irrelevant":
            assert fill == "Fill: nothing; the sentence has no placeholder."
        assert lines[-1].startswith("Check:") and "no call" in lines[-1]
    assert set(found) == set(sm.CASES)


def test_the_decision_comes_after_every_parameter_line():
    for item in training_items():
        lines = item.reasoning.split("\n")
        decision = next(i for i, line in enumerate(lines) if line.startswith("Decision:"))
        checks = [i for i, line in enumerate(lines) if line.startswith("- ")]
        assert checks and all(i < decision for i in checks), item.reasoning
        assert sum(line.startswith("Decision:") for line in lines) == 1


def test_the_parameter_lines_say_given_or_not_and_valid_or_not():
    by_kind = {}
    for item in training_items(seeds=1):
        by_kind.setdefault((item.core.kind, item.core.variant.get("subkind"), bool(item.core.optional_cut)),
                           item)
    incorrect = next(i for (kind, _, _), i in by_kind.items() if kind == "incorrect")
    bad = incorrect.core.invalid[0]
    line = next(x for x in incorrect.reasoning.split("\n") if x.startswith(f"- {sm._n(bad.param)} "))
    assert f'given "{bad.value}"; not valid' in line
    assert any(f'given "{sm.value_text(v)}"; valid' in incorrect.reasoning
               for k, v in incorrect.core.given.items() if k != bad.param)
    missing = by_kind["missing", "single", False]
    gap = missing.core.missing[0]
    assert f"- {gap} (required): not given." in missing.reasoning.split("\n")
    optional = by_kind["missing", "optional", True]
    assert f"- {optional.core.optional_cut[0]} (optional): not given; optional, so it is not named." in \
        optional.reasoning.split("\n")
    nested = by_kind["missing", "object", False]
    assert ("- budget (required): not given; its value is a whole object, so only budget is named, not its "
            "fields amount and currency.") in nested.reasoning.split("\n")
    irrelevant = by_kind["irrelevant", None, False]
    assert "Match: none of the functions serves the request." in irrelevant.reasoning.split("\n")


def test_only_a_priority_reasoning_states_the_order_of_the_checks():
    # FT5 stated the order wrongly in 10 of 30 reasoning texts, 8 of 16 under missing-first prompts: the old
    # template added "The rules check wrong values first." to items where only one case occurs
    for core in all_cores():
        for order in sm.ORDERS:
            item = sm.dress(core, split="train", rng=rng(order), order=order)
            assert ("values first" in item.reasoning) is (core.kind == "priority"), item.reasoning
            if core.kind == "priority":
                checked = "missing values first" if order == "missing_first" else "wrong values first"
                assert f"the rules check {checked}" in item.reasoning


def test_reasoning_problems_flag_a_broken_skeleton():
    item = sm.dress(all_cores()[0], split="train", rng=rng())
    assert sm.reasoning_problems(item) == []
    lines = item.reasoning.split("\n")
    broken = [item.reasoning.replace(sm.REASONING_OPENING, "Tools:", 1),
              item.reasoning.replace(f'Answer: "{item.answer}"', 'Answer: "x"'),
              "\n".join(line for line in lines if not line.startswith("Decision:")),
              "\n".join([lines[0], next(x for x in lines if x.startswith("Decision:")),
                         *(x for x in lines[1:] if not x.startswith("Decision:"))]),  # decided first
              item.reasoning + "\n" + "Check: and more. " * 400,
              item.reasoning + "\n</think>"]
    for text in broken:
        assert sm.reasoning_problems(dataclasses.replace(item, reasoning=text)), text[:60]


# The development dressing of all_cores' records as the locked build drew it (computed from the code of
# f023317 before revision 1): the cores' builders and dress for sets A and B, under three seeds each, as
# dev_item writes the items, with the next draw of each generator after it.
LOCKED_DRESSING = "29118dd073bedef33cd5d272a3672ea31357453281d234d171ab24b682b24613"


def test_the_development_dressing_draws_what_the_locked_sets_were_drawn_with():
    # the build draws set B, then A, then training, every item with a generator of its own: revision 1 keeps
    # the development path as it was (families, styles, draws), and a development item carries no reasoning
    rows = []
    for number, core in enumerate(all_cores(), 1):
        if core.kind == "ask":
            continue
        for split in ("a", "b"):
            for seed in range(3):
                generator = random.Random(f"golden/{split}/{number}/{seed}")
                item = sm.dress(core, split=split, rng=generator)
                assert item.reasoning == ""
                rows.append([json.dumps(sm.dev_item(item, split, number), ensure_ascii=False),
                             generator.random()])
    digest = hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode("utf-8")).hexdigest()
    assert digest == LOCKED_DRESSING


def test_wording_never_names_a_listed_function_as_its_example():
    item = sm.dress(sm.build_verbatim(catalog(), "zh", rng(), require_cjk=True), split="train", rng=rng())
    assert "tool_name" in item.system and "tool_name" not in item.answer


# ---- ACEBench text ban ----------------------------------------------------------------------------------


def test_banned_markers():
    for text in ("There is incorrect value (x) for the parameters (y)", "missing necessary parameters (a)",
                 "Due to the limitations of the function", "use ApiName here", "Conversation history 1..t:",
                 "对话历史1..t:", "[get_weather(city='Oslo')]"):
        assert sm.banned_markers(text), text
    assert sm.banned_markers("A plain request about the weather in Oslo.") == []


def test_template_windows_flag_five_words_and_eight_characters():
    windows = sm.TemplateWindows.from_texts(["Please answer the user in one short sentence {time} here",
                                             "请根据给定的说明生成调用请求"])
    assert windows.violations("Kindly answer the user in one short reply")
    assert not windows.violations("answer the user in")  # four of the words only
    assert windows.violations("我们请根据给定的说明生成吧")
    assert not windows.violations("请根据给定的说")  # seven characters
    assert not windows.violations("short sentence here and more words")  # a field splits the template


@needs_clone
def test_own_wording_shares_no_window_with_the_acebench_templates():
    windows = sm.TemplateWindows.from_files(PROMPTS)
    texts = list(sm.own_texts())
    assert len(texts) > 300
    for text in texts:
        assert windows.violations(text) == [], (text, windows.violations(text))
        assert sm.banned_markers(text) == [], text
    for core in all_cores():
        for split in sm.SPLITS:
            item = sm.dress(core, split=split, rng=random.Random(split + core.kind))
            for text in (item.system, item.answer, item.reasoning, *(m["content"] for m in item.messages)):
                assert windows.violations(text) == [], (text, windows.violations(text))
                assert sm.banned_markers(text) == []


@needs_clone
def test_own_function_names_are_no_acebench_names():
    names = {n.casefold() for n in sm.acebench_function_names(CLONE / "data_all")}
    assert len(names) >= 4000
    assert not {a.name.casefold() for a in sm.ASK_FUNCTIONS} & names
    assert "tool_name" not in names


def test_prompt_templates_are_read_without_running_the_module(tmp_path):
    module = tmp_path / "prompt_x.py"
    module.write_text('import os\nA = """one {x} two"""\nB = "three"\nos.remove(__file__)\n',
                      encoding="utf-8")
    assert sm.prompt_templates(module) == ["one {x} two", "three"]
    assert module.exists()


def test_apiname_is_banned_as_a_substring_of_any_case():
    # ACEBench's placeholder inside an identifier (deepexi's ApiNameKeyword) counts as the placeholder
    for text in ("ListApis(ApiNameKeyword=\"数据\")", "the apiname field", "APINAME", "myApiNames"):
        assert "ACEBench's placeholder ApiName" in sm.banned_markers(text), text
    assert sm.holds_apiname("x_ApiNameKeyword") and not sm.holds_apiname("api name")
    item = sm.dress(sm.build_verbatim(catalog(), "zh", rng(), require_cjk=True), split="train", rng=rng())
    record = sm.training_record(item)
    record["messages"][0]["content"] += "\nApiNameKeyword: 关键字"
    assert any("ApiName" in problem for problem in sm.check_training_record(record))


# ---- near-duplicate requests (D1) -------------------------------------------------------------------------


@pytest.mark.parametrize("a, b, test", [
    ("Can you order a pizza for me?", "can you  ORDER a pizza for me", "exact"),  # case, spaces, punctuation
    ("Can you order a pizza for me?", "Please order a pizza.", "signature"),  # politeness and stop words
    ("Could you please order me a pizza?", "Order a pizza, thanks!", "signature"),
    ("帮我查一下工作空间123的任务列表。", "请帮我查一下工作空间123的任务列表吧", "signature"),  # particles
    ("I need the distance between New York and Boston in miles today",
     "I need the distance between New York and Boston in miles", "jaccard"),
    ("查询工作空间4471里状态为RUNNING的任务列表,每页30条记录",
     "查询工作空间4471里状态为RUNNING的任务列表,每页30条", "jaccard"),
])
def test_near_duplicate_requests(a, b, test):
    assert sm.near_duplicate(a, b) == test
    assert sm.near_duplicate(b, a) == test


@pytest.mark.parametrize("a, b", [
    ("Can you order a pizza for me?", "Can you book a flight to Rome for me?"),
    ("What is the weather in Paris?", "What is the weather in Lisbon?"),
    ("帮我查一下工作空间123的任务列表。", "帮我删除工作空间123的任务。"),
    ("Convert 100 USD to EUR", "Convert 250 GBP to JPY"),
])
def test_distinct_requests_are_no_near_duplicates(a, b):
    assert sm.near_duplicate(a, b) is None


def test_request_keys_normalise_text_and_split_by_script():
    key = sm.request_key("  Ｃan you, PLEASE, order a pizza?! ")
    assert key.norm == "can you please order a pizza"
    assert key.signature == frozenset({"order", "pizza"})
    zh = sm.request_key("请帮我查一下工作空间的任务吧。")
    assert zh.signature == "查一下工作空间任务" and "工作" in zh.tokens


def test_request_index_finds_what_pairwise_comparison_finds():
    texts = ["Can you order a pizza for me?", "Please order a pizza.", "What is the weather in Paris?",
             "What is the weather in Paris today?", "Book a table for two at Luigi's at 7 pm tonight",
             "Book a table for two at Luigi's at 7 pm", "帮我查一下工作空间123的任务列表。",
             "请帮我查一下工作空间123的任务列表吧", "删除实例i-123。", "Translate hello into French"]
    words = ["alpha", "beta", "gamma", "delta", "eps", "zeta", "eta", "theta", "iota", "kappa", "lambda"]
    generator = random.Random(3)
    for _ in range(60):
        texts.append(" ".join(generator.sample(words, generator.randint(4, 9))))
    index = sm.RequestIndex()
    for number, text in enumerate(texts):
        index.add(number, [text])
    for number, text in enumerate(texts):
        found = index.matches([text])
        expected = {other: sm.near_duplicate(text, texts[other]) for other in range(len(texts))
                    if sm.near_duplicate(text, texts[other])}
        assert found == expected, (text, found, expected)


def test_request_texts_hold_the_final_request_and_the_originals():
    core = sm.build_missing(meeting(), "en", rng(), subkind="single")
    item = sm.dress(core, split="train", rng=rng())
    texts = sm.request_texts(item)
    assert texts[0] == item.messages[-1]["content"]
    assert meeting()["messages"][0]["content"] in texts
    switch = sm.dress(sm.build_switch(trip("1"), meeting("glaive_v2", "2"), "en", rng(), packed=True),
                      split="train", rng=rng())
    texts = sm.request_texts(switch)
    assert texts[0] == switch.messages[1]["content"]  # the packed message
    assert trip("1")["messages"][0]["content"] in texts and meeting()["messages"][0]["content"] in texts


# ---- gold calls fully grounded (D3) ---------------------------------------------------------------------


TAGS = tool("TagResource", {"TagId": {"type": "string", "description": "标签ID"},
                            "TagValue": {"type": "string", "description": "标签值"}})


def test_a_value_given_once_fills_one_parameter_only():
    copied = sample("deepexi", "c1", "给资源打上IT部门这个标签。", [("TagResource", {"TagId": "IT部门",
                                                                         "TagValue": "IT部门"})], [TAGS])
    assert sm.call_problem(copied["messages"][0]["content"], [("TagResource", {"TagId": "IT部门",
                                                                             "TagValue": "IT部门"})],
                           [TAGS["function"]["parameters"]]) == "copied_value"
    assert sm.single_call(copied) is None and sm.build_verbatim(copied, "zh", rng()) is None
    twice = sample("deepexi", "c2", "给资源打上标签,标签ID为IT部门,标签值也是IT部门。",
                   [("TagResource", {"TagId": "IT部门", "TagValue": "IT部门"})], [TAGS])
    assert sm.single_call(twice) is not None  # given twice, it may fill two parameters
    numbers = tool("GetTenant", {"ProjectId": {"type": "integer", "description": "项目ID"},
                                 "TenantId": {"type": "integer", "description": "租户ID"}})
    copied = sample("deepexi", "c3", "查询项目123的租户信息。",
                    [("GetTenant", {"ProjectId": 123, "TenantId": 123})], [numbers])
    assert sm.single_call(copied) is None


def test_parallel_calls_may_share_a_value_in_the_same_parameter():
    user = "What's the forecast for Oslo and the forecast for Bergen?"
    calls = [("get_forecast", {"city": "Oslo"}), ("get_forecast", {"city": "Bergen"})]
    assert sm.call_problem(user, calls, [WEATHER["function"]["parameters"]] * 2) is None
    same = "Give me the forecast and the hourly forecast for Oslo."
    calls = [("get_forecast", {"city": "Oslo"}), ("get_forecast", {"city": "Oslo"})]
    assert sm.call_problem(same, calls, [WEATHER["function"]["parameters"]] * 2) is None


@pytest.mark.parametrize("arguments, problem", [
    ({"body": []}, "empty_value"),
    ({"body": {}}, "empty_value"),
    ({"body": {"path": ""}}, "empty_value"),
    ({"body": {"path": None}}, "empty_value"),
    ({"body": {"path": "OSS路径"}}, "label_value"),  # a nested label (sdev-b-0088): label_value first
])
def test_nested_values_are_checked_too(arguments, problem):
    register = tool("RegisterLocation", {"body": {"type": "object", "description": "请求体",
                                                  "properties": {"path": {"type": "string",
                                                                          "description": "OSS路径"}}}})
    user = "我想注册一个OSS路径。"
    assert sm.call_problem(user, [("RegisterLocation", arguments)], [register["function"]["parameters"]]) \
        == problem
    s = sample("deepexi", "n1", user, [("RegisterLocation", arguments)], [register])
    assert sm.build_verbatim(s, "zh", rng()) is None


def test_a_nested_value_naming_its_parameter_is_caught():
    # introduced by 为, so no label (label_value), but asked about: the value names its parameter, nested
    # or not
    url = {"type": "string", "description": "视频文件URL地址。"}
    nested = tool("RegisterVideo", {"body": {"type": "object", "description": "请求体",
                                             "properties": {"VideoURL": url}}})
    flat = tool("RegisterVideo", {"VideoURL": url})
    user = "能把视频地址设为视频URL地址吗？"
    assert sm.call_problem(user, [("RegisterVideo", {"body": {"VideoURL": "视频URL地址"}})],
                           [nested["function"]["parameters"]]) == "names_parameter"
    assert sm.call_problem(user, [("RegisterVideo", {"VideoURL": "视频URL地址"})],
                           [flat["function"]["parameters"]]) == "names_parameter"


@pytest.mark.parametrize("user, value, kept", [
    ("如何获取上海地域的OSS链接?", "上海地域的OSS链接", False),  # a label, not introduced
    ("设置告警手机号码。", "手机号码", False),  # 号码 is a label end
    ("把告警电话设为手机号码。", "手机号码", True),  # introduced by 为
    ('上传名为"测试文件"的附件。', "测试文件", True),  # quoted
    ("文件名称：季度报告文件", "季度报告文件", True),  # after a colon
    ("这个目录叫销售资料文件。", "销售资料文件", True),  # introduced by 叫
    ("查看销售资料文件的大小。", "销售资料文件", False),
    ("查询张三的订单。", "张三", True),  # no label end
])
def test_chinese_label_values_must_be_introduced(user, value, kept):
    upload = tool("Upload", {"Name": {"type": "string", "description": "名字"}})
    s = sample("deepexi", "l1", user, [("Upload", {"Name": value})], [upload])
    assert (sm.build_verbatim(s, "zh", rng()) is not None) is kept
    assert (sm.call_problem(user, [("Upload", {"Name": value})], [upload["function"]["parameters"]])
            is None) is kept


def test_label_ends_are_widened():
    for end in ("号码", "链接", "地址", "名称", "编号", "信息", "内容", "路径", "文件"):
        assert end in sm.LABEL_ENDS


@pytest.mark.parametrize("user, arguments, grounded", [
    ("Book 2 rooms for 1,500 dollars.", {"rooms": 2, "budget": 1500}, True),  # a thousands separator
    ("Set the price to 20.", {"price": 20.0}, True),  # 20.0 is the number 20
    ("Arrive on 2024/3/5 please.", {"date": "2024-03-05"}, True),  # the same date, other separators
    ("Arrive on 2024-03-05 please.", {"date": "2024-03-06"}, False),
    ("Find vegan recipes.", {"query": "vegan", "vegetarian": False}, False),  # an invented boolean
    ("Find vegan recipes, vegetarian ones too.", {"query": "vegan", "vegetarian": True}, True),
    ("Email the report to Ana.", {"to": "Ana", "subject": "Report"}, False),  # another case: not verbatim
    ("Email the report to Ana.", {"to": "Ana", "subject": "Monthly report"}, False),  # invented
])
def test_every_gold_leaf_occurs_in_the_request(user, arguments, grounded):
    schema = {"type": "object", "properties": {k: {} for k in arguments}}
    found = sm.call_problem(user, [("f", arguments)], [schema])
    assert (found is None) is grounded, found
    if not grounded:
        assert found == "ungrounded"


ROUTE = tool("SyncLink", {"Direction": {"type": "string", "enum": ["Forward", "Reverse"],
                                        "description": "同步方向"},
                          "LinkId": {"type": "string", "description": "链路ID"}})


def test_an_enum_near_miss_is_dropped_when_the_request_names_another_option():
    # "...可以传入Reverse以启动正向同步链路吗": the request names Forward too, in other words (正向)
    user = "链路ID为L-77,可以传入Reverse以启动正向同步链路吗?"
    s = sample("deepexi", "e1", user, [("SyncLink", {"Direction": "Reverse", "LinkId": "L-77"})], [ROUTE])
    assert sm.build_nearmiss(s, "zh", rng()) is None
    plain = sample("deepexi", "e2", "链路ID为L-77,传入Reverse启动反向同步。",
                   [("SyncLink", {"Direction": "Reverse", "LinkId": "L-77"})], [ROUTE])
    core = sm.build_nearmiss(plain, "zh", rng())
    assert core is not None and core.variant["reason"] == "enum"
    english = sample("glaive_v2", "e3", "Sync link L-77 with Reverse, going forward.",
                     [("SyncLink", {"Direction": "Reverse", "LinkId": "L-77"})], [ROUTE])
    assert sm.build_nearmiss(english, "en", rng()) is None  # forward: the other option
    english["messages"][0]["content"] = "Sync link L-77 with Reverse now."
    assert sm.build_nearmiss(english, "en", rng()) is not None
    assert sm.names_other_option("启动反向同步", ("Forward", "Reverse"), "Forward")
    assert not sm.names_other_option("启动同步", ("Forward", "Reverse"), "Forward")


def test_value_banks_hold_only_grounded_values():
    assert sm.bank_value_ok("新文件夹", '把文件夹改名为"新文件夹"。', "文件夹名称")
    assert not sm.bank_value_ok("上海地域的OSS链接", "如何获取上海地域的OSS链接?", "OSS地址")
    assert not sm.bank_value_ok("", "x", "")


# ---- the request's own language (D4) ----------------------------------------------------------------------


def test_wrong_values_follow_the_requests_language_not_the_prompts():
    for seed in range(8):
        core = sm.build_incorrect(meeting(), "zh", rng(str(seed)), category="vague")
        if core is None:
            continue
        assert not re.search(r"[一-鿿]", core.history[-1][1]), core.history[-1][1]
    for seed in range(8):
        core = sm.build_incorrect(meeting(), "zh", rng(str(seed)), category="format")
        assert core is not None and not re.search(r"[一-鿿]", core.invalid[0].value), core.invalid[0].value
    core = sm.build_incorrect(jobs(), "en", rng(), category="vague")  # a Chinese record, an English prompt
    assert re.search(r"[一-鿿]", core.invalid[0].value)
    assert sm.request_lang("hermes", "Book a room") == "en" and sm.request_lang("deepexi", "查询") == "zh"
    assert sm.request_lang("other", "查询任务") == "zh" and sm.request_lang("other", "list jobs") == "en"


def test_follow_ups_follow_the_requests_language():
    bank = {"title": ["Kickoff", "Retro"]}
    core = sm.build_change(meeting(), "zh", rng(), bank=bank, packed=False)
    assert core is not None and not re.search(r"[一-鿿]", core.history[-1][1])
    second = meeting("glaive_v2", "2")["messages"][0]["content"]
    prefixes = set()
    for seed in range(12):  # an English request under a Chinese prompt keeps an English prefix
        switch = sm.build_switch(trip("1"), meeting("glaive_v2", "2"), "zh", rng(f"s{seed}"), packed=False)
        text = switch.history[-1][1]
        assert text.endswith(second) and not re.search(r"[一-鿿]", text), text
        prefixes.add(text[:-len(second)])
    assert len(prefixes) > 1  # the draws include prefixes, not only the empty one


# ---- cuts with their frame, fluency (D8) and whole top-level objects (D7) ---------------------------------


@pytest.mark.parametrize("text, values, expected", [
    ("I'm looking for a good restaurant in Paris. Can you help?", ["Paris"],
     "I'm looking for a good restaurant. Can you help?"),
    ("Hi, I need to know the distance between New York and Boston.", ["New York", "Boston"],
     "Hi, I need to know the distance."),
    ("Find the user named John Smith in the directory.", ["John Smith"], "Find the user in the directory."),
    ('Please schedule "Budget review" on 2024-03-05 for 90 minutes.', ["90"],
     'Please schedule "Budget review" on 2024-03-05.'),
    ("新建一个名为销售数据的目录,备注写季度汇总。", ["销售数据"], "新建一个目录,备注写季度汇总。"),
    ("我想获取表的数据概况，表名为orders，数据库名为sales。", ["orders"],
     "我想获取表的数据概况，数据库名为sales。"),
    ("我想查一下在北京的天气。", ["北京"], "我想查一下天气。"),
    ("查询从2024-01-01到2024-01-31的订单。", ["2024-01-01", "2024-01-31"], "查询订单。"),
    ("我想取消对张三的操作权限。", ["张三"], "我想取消操作权限。"),  # the bare cut left 对的
])
def test_values_are_cut_with_their_frame_and_read_fluently(text, values, expected):
    lang = "zh" if re.search(r"[一-鿿]", text) else "en"
    cut = sm.cut_with_frames(text, values, lang)
    assert cut is not None and cut.text == expected and cut.fluent, cut


@pytest.mark.parametrize("text, values", [
    ("I want to know if racecar and level are anagrams.", ["racecar", "level"]),
    ("我想将数据湖角色授予alice和bob。", ["alice", "bob"]),
])
def test_cuts_that_leave_a_dangling_word_are_bare(text, values):
    lang = "zh" if re.search(r"[一-鿿]", text) else "en"
    cut = sm.cut_with_frames(text, values, lang)
    assert cut is not None and not cut.fluent, cut


@pytest.mark.parametrize("text, fluent", [
    ("Book a table. Can you help?", True),
    ("Book a table for. Can you help?", False),  # a dangling preposition
    ("Book a table  now.", False),  # doubled space
    ("Book a table , now.", False),  # a space before punctuation
    ('Book the table "" now.', False),  # empty quotes
    ("Book a table () now.", False),  # empty brackets
    ("查询订单。", True),
    ("查询的订单。", True),
    ("我想取消对的操作权限。", False),
])
def test_the_fluency_heuristic(text, fluent):
    lang = "zh" if re.search(r"[一-鿿]", text) else "en"
    assert sm.reads_fluently(text, lang) is fluent


def test_missing_cores_record_the_cut_and_prefer_a_fluent_one():
    core = sm.build_missing(meeting(), "en", rng(), subkind="single")
    assert core.variant["cut"] in ("frame", "bare")
    fluent = [sm.build_missing(meeting(sid=str(i)), "en", rng(str(i)), subkind="single") for i in range(12)]
    assert all(c.variant["cut"] == "frame" for c in fluent if c is not None)
    item = sm.dress(core, split="b", rng=rng())
    dev = sm.dev_item(item, "b", 1)
    assert dev["source"]["cut"] == core.variant["cut"]
    assert sm.check_dev_item(dev) == []
    priority = sm.dev_item(sm.dress(sm.build_priority(meeting(), "en", rng()), split="b", rng=rng()), "b", 2)
    assert priority["source"]["cut"] in ("frame", "bare") and sm.check_dev_item(priority) == []
    verbatim = sm.dev_item(sm.dress(sm.build_verbatim(catalog(), "zh", rng(), require_cjk=True), split="b",
                                    rng=rng()), "b", 3)
    assert "cut" not in verbatim["source"] and sm.check_dev_item(verbatim) == []
    broken = json.loads(json.dumps(dev))
    del broken["source"]["cut"]
    assert sm.check_dev_item(broken)  # a missing item says how it was cut


SETTINGS = tool("update_settings", {
    "user": {"type": "string", "description": "User name"},
    "settings": {"type": "object", "description": "Notification settings",
                 "properties": {"channel": {"type": "string"}, "email": {"type": "boolean"},
                                "sms": {"type": "boolean"}}}}, ["user", "settings"])


def test_an_object_is_cut_only_whole():
    # the whole prose span goes, with its frame
    core = sm.build_missing(trip(), "en", rng(), subkind="object")
    assert core is not None and core.history[-1][1] == "Plan a trip to Lisbon for 2 people with a budget."
    assert core.variant["cut"] == "frame"
    # boolean leaves rendered as words stay in the request: no object item
    words = sample("glaive_v2", "s1", "For user Ana, enable both email and SMS notifications on channel ops.",
                   [("update_settings", {"user": "Ana", "settings": {"channel": "ops", "email": True,
                                                                     "sms": True}})], [SETTINGS])
    assert sm.build_missing(words, "en", rng(), subkind="object") is None
    # a JSON skeleton left behind is no cut of the whole object
    perms = tool("Grant", {"Body": {"type": "object", "description": "请求体",
                                    "properties": {"permission": {"type": "string"}}}})
    skeleton = sample("deepexi", "s2", '授予权限,请求体内容是{"permission": "read"},立即生效。',
                      [("Grant", {"Body": {"permission": "read"}})], [perms])
    core = sm.build_missing(skeleton, "zh", rng(), subkind="object")
    assert core is None or ("{" not in core.history[-1][1] and "permission" not in core.history[-1][1])
    scattered = sample("glaive_v2", "s3", "Plan a trip for 2 people: 1500, and then in EUR, to Lisbon.",
                       [("plan_trip", {"destination": "Lisbon", "budget": {"amount": 1500, "currency": "EUR"},
                                       "travellers": 2})], [TRIP])
    assert sm.build_missing(scattered, "en", rng(), subkind="object") is None  # the span holds other text


AREA = tool("calculate_area", {
    "shape": {"type": "string", "description": "The shape"},
    "dimensions": {"type": "object", "description": "The dimensions",
                   "properties": {"length": {"type": "number"}, "width": {"type": "number"}}}},
    ["shape", "dimensions"])


def test_an_object_stated_in_its_own_clause_is_cut_with_its_keys():
    call = [("calculate_area", {"shape": "rectangle", "dimensions": {"length": 5, "width": 3}})]
    request = "I need the area of a rectangle. The length is 5 units and the width is 3 units."
    s = sample("glaive_v2", "d1", request, call, [AREA])
    core = sm.build_missing(s, "en", rng(), subkind="object")
    assert core is not None and core.history[-1][1] == "I need the area of a rectangle."
    assert core.missing == ["dimensions"] and core.variant["cut"] == "frame"
    keyed = sample("glaive_v2", "d2", "Area of a rectangle with length 5 and width 3, please.", call, [AREA])
    core = sm.build_missing(keyed, "en", rng(), subkind="object")
    assert core is not None and core.history[-1][1] == "Area of a rectangle, please."
    left = sample("glaive_v2", "d3",
                  "The length is 5 and the width is 3 for my rectangle; its length matters.", call, [AREA])
    assert sm.build_missing(left, "en", rng(), subkind="object") is None  # a key stays elsewhere


# ---- the hand check's first round (D3): the rules its findings added ---------------------------------------


def problem(user: str, arguments: dict, properties: dict | None = None, name: str = "f") -> str | None:
    properties = properties if properties is not None else {k: {"type": "string"} for k in arguments}
    return sm.call_problem(user, [(name, arguments)], [{"type": "object", "properties": properties}])


@pytest.mark.parametrize("user, arguments, found", [
    ("Show the coronavirus statistics for Italy.", {"location": "IT"}, "ungrounded"),  # not the word IT
    ("Give me a new French word to learn.", {"language": "fr"}, "ungrounded"),
    ("Find Instagram posts with the hashtag '#SummerSuns'.", {"tag": "SummerSuns"}, None),
    ("Email the report to Ana.", {"to": "Ana", "subject": "report"}, None),
    ("I'd love it if you targeted cities in Italy.", {"countryIsoCode": "IT"}, "ungrounded"),  # pronoun
    ("Show the US news today.", {"country": "US"}, None),
])
def test_an_ascii_value_is_given_only_as_a_whole_word(user, arguments, found):
    assert problem(user, arguments) == found


@pytest.mark.parametrize("user, arguments, found", [
    ("我想知道项目A中角色B对表C的ACL授权信息。",
     {"projectName": "项目A", "roleName": "角色B", "objectType": "表", "objectName": "表C"},
     "inside_other_value"),
    ("Find hotels in New York.", {"city": "York", "region": "New York"}, "inside_other_value"),
    ("获取文件夹路径为/dataworks/folder的文件夹的详情。",
     {"ProjectIdentifier": "dataworks", "FolderPath": "/dataworks/folder"}, "inside_other_value"),
    ("Find hotels in York near New York Street.", {"city": "York", "street": "New York Street"}, None),
])
def test_a_value_found_only_inside_another_value_is_not_given(user, arguments, found):
    assert problem(user, arguments) == found


def test_a_call_of_booleans_only_is_not_given():
    assert problem("更新数据湖Catalog的配置需要admin角色以上的权限吗？", {"adminRoleRequired": True},
                   {"adminRoleRequired": {"type": "boolean"}}) == "boolean_only"
    assert problem("Turn on the alerts for Oslo.", {"alerts": True, "city": "Oslo"},
                   {"alerts": {"type": "boolean"}, "city": {"type": "string"}}) is None


@pytest.mark.parametrize("arguments", [{"Content": "xxxxx"}, {"instanceId": "holo-xxxxxx"},
                                       {"projectName": "example"}, {"Name": "XXX"},
                                       {"Content": "新的说明信息"}, {"Description": "新的描述信息"}])
def test_a_placeholder_value_is_no_value(arguments):
    [(key, value)] = arguments.items()
    user = f"请处理{key}为{value}的对象。"
    assert problem(user, arguments) == "placeholder_value"
    assert not sm.bank_value_ok(value, user, "")


@pytest.mark.parametrize("user, arguments, found", [
    ("我想调用ScanSensitiveData来识别一下这段数据中是否存在敏感数据。", {"Data": "这段数据"},
     "generic_value"),
    ("查一下某个用户的权限。", {"User": "某个用户"}, "generic_value"),
    ('我想更新数据源的状态为"状态"。', {"Status": "状态"}, "names_parameter"),  # the parameter's own word
    ('我想更新数据源的描述信息为"数据源的描述信息"。', {"Description": "数据源的描述信息"},
     "names_parameter"),
    ("取消数据湖角色对资源B的权限。", {"role": "数据湖角色", "resource": "资源B"}, "label_value"),
    ('我想移除实体A的标签"运维"和"数据"。', {"Entity": "实体A", "TagKeys": ["运维", "数据"]}, None),
    ("Find Italian restaurants near me.", {"location": "near me", "cuisine": "Italian"}, "generic_value"),
])
def test_generic_chinese_values_are_no_values(user, arguments, found):
    described = {"Status": "数据源的状态。", "TagKeys": "标签键的列表。", "Description": "数据源的描述信息。"}
    properties = {k: {"type": "array" if isinstance(v, list) else "string",
                      "description": described.get(k, "")} for k, v in arguments.items()}
    assert problem(user, arguments, properties) == found


def test_label_ends_cover_roles_resources_permissions_and_states():
    for end in ("角色", "资源", "权限", "状态"):
        assert end in sm.LABEL_ENDS


@pytest.mark.parametrize("key, value, found", [
    ("WorksId", "数据门户", "word_identifier"), ("RegionId", "北京", "word_identifier"),
    ("user_ids", "张三", "word_identifier"), ("UserId", "用户A", None), ("Valid", "数据门户", None),
    ("movie_id", "Inception", "word_identifier"),  # hand check, round 4: a title for an identifier
    ("habitat_id", "Savannah region", "word_identifier"),
    ("user_id", "johndoe", None), ("ticker_id", "AAPL", None), ("event_id", "12345ABC", None),
])
def test_an_identifier_parameter_takes_no_words(key, value, found):
    assert problem(f"处理{key}为{value}的请求。", {key: value}) == found


@pytest.mark.parametrize("user, arguments, found", [
    ('把文件夹"报表"改名为"销售"。', {"FolderName": "销售"}, "unused_quote"),
    ('把文件夹"报表"改名为"销售"。', {"OldName": "报表", "FolderName": "销售"}, None),
    ('Encrypt the files "EmployeeData", "ClientContracts" and "FinancialRecords".',
     {"data": ["EmployeeData", "ClientContracts"]}, "unused_quote"),
    ('Please use the "lookup_order" function for order "A-17".', {"order_id": "A-17"}, None),
    ('Run it:\n```json\n{"key": "x1"}\n```\nwith mode "fast".', {"mode": "fast"}, None),  # code is no request
    ('把文件夹名称为"新文件夹"的文件夹改名为"新文件夹2"。', {"FolderName": "新文件夹2"},
     "unused_quote"),  # the old name is but a prefix of the new one
    ('Send it with "Quarterly Planning Meeting," as the subject.', {"subject": "Quarterly Planning Meeting"},
     None),
])
def test_every_quoted_value_of_the_request_is_used(user, arguments, found):
    properties = {k: {"type": "array" if isinstance(v, list) else "string"} for k, v in arguments.items()}
    assert problem(user, arguments, properties) == found


HOTEL = tool("book_hotel", {"city": {"type": "string", "description": "City"},
                            "check_in": {"type": "string", "format": "date", "description": "Arrival"},
                            "check_out": {"type": "string", "format": "date", "description": "Departure"},
                            "at": {"type": "string", "description": "Arrival time as HH:MM"}},
             ["city", "check_in"])


def test_a_date_or_time_near_miss_needs_no_other_value_of_its_kind():
    pair = sample("glaive_v2", "h1", "Book a hotel in Paris from 2022-05-01 to 2022-05-10.",
                  [("book_hotel", {"city": "Paris", "check_in": "2022-05-01", "check_out": "2022-05-10"})],
                  [HOTEL])
    assert sm.build_nearmiss(pair, "en", rng()) is None  # one date of a range moved: the range breaks
    times = sample("glaive_v2", "h2", "Book a hotel in Paris on 2022-05-01, arriving at 09:00.",
                   [("book_hotel", {"city": "Paris", "check_in": "2022-05-01", "at": "09:00"})], [HOTEL])
    core = sm.build_nearmiss(times, "en", rng())  # a date and a time: each the only one of its kind
    assert core is not None and core.variant["reason"] in ("leap", "time")


def test_a_near_miss_is_not_made_where_the_request_restates_the_value():
    restated = sample("glaive_v2", "h3",
                      "Book Paris for April 30th. Details: City: Paris, Check-in: 2023-04-30.",
                      [("book_hotel", {"city": "Paris", "check_in": "2023-04-30"})], [HOTEL])
    assert sm.build_nearmiss(restated, "en", rng()) is None
    zh = sample("deepexi", "h4", "4月30日入住巴黎，入住日期2023-04-30。",
                [("book_hotel", {"city": "巴黎", "check_in": "2023-04-30"})], [HOTEL])
    assert sm.build_nearmiss(zh, "zh", rng()) is None
    noon = sample("glaive_v2", "h5", "Book Paris, check-in 2023-04-30, arriving at 10:00 AM.",
                  [("book_hotel", {"city": "Paris", "check_in": "2023-04-30", "at": "10:00"})], [HOTEL])
    for seed in range(6):  # whichever value moves, no time is left with AM after it
        core = sm.build_nearmiss(noon, "en", rng(str(seed)))
        assert core is None or core.variant["reason"] != "time"


MODES = tool("update_inventory", {"file": {"type": "string", "description": "CSV file"},
                                  "mode": {"type": "string", "enum": ["append", "replace"]}},
             ["file", "mode"])


def test_an_enum_near_miss_needs_a_request_naming_no_other_option_in_any_form():
    inflected = sample("hermes", "m1", "Rows were appended before. Update stock.csv with mode: replace.",
                       [("update_inventory", {"file": "stock.csv", "mode": "replace"})], [MODES])
    assert sm.build_nearmiss(inflected, "en", rng()) is None  # "appended" names the option append
    assert sm.names_other_option("rows are replaced", ("append", "replace"), "append")
    assert not sm.names_other_option("rows are re-placed", ("append", "replace"), "append")
    plain = sample("hermes", "m2", "Update stock.csv with mode: replace.",
                   [("update_inventory", {"file": "stock.csv", "mode": "replace"})], [MODES])
    assert sm.build_nearmiss(plain, "en", rng()) is not None


FEED = tool("get_posts", {"hashtag": {"type": "string", "description": "The hashtag"},
                          "feed_type": {"type": "string", "description": "Either 'recent' or 'top'."}},
            ["hashtag", "feed_type"])


@pytest.mark.parametrize("user", [
    "Can you pull up the most recent posts for #HealthyEating?",
    "Pull up posts for #HealthyEating, feed type: recent.",
    "Pull up the 'recent' feed for #HealthyEating.",
])
def test_an_enum_near_miss_keeps_the_request_and_its_option(user):
    # hand check, rounds 1 to 3: a swapped option left "the most top posts", "a accurate model",
    # "English (es)" and Java beside a snippet of Python; the request as it is gives an exact option
    call = ("get_posts", {"hashtag": "#HealthyEating", "feed_type": "recent"})
    record = sample("toolace", "p1", user, [call], [FEED])
    core = sm.build_nearmiss(record, "en", rng())
    assert core is not None and core.variant == {**core.variant, "value": "recent", "reason": "enum"}
    assert core.history == [("user", user)] and core.calls == [call]


QUBITS = tool("simulate", {"algorithm": {"type": "string", "description": "Algorithm"},
                           "qubits": {"type": "integer", "minimum": 1, "maximum": 64}},
              ["algorithm", "qubits"])


def test_a_number_near_miss_keeps_the_noun_after_it_agreeing():
    def simulated(user: str, qubits: int) -> dict:
        call = ("simulate", {"algorithm": "Grover search", "qubits": qubits})
        return sample("toolace", "q1", user, [call], [QUBITS])

    core = sm.build_nearmiss(simulated("Simulate Grover search using 5 qubits.", 5), "en", rng())
    assert core is not None and core.variant["value"] == 64  # the maximum: 64 qubits
    many = simulated("Simulate Grover search using 64 qubits.", 64)
    assert sm.build_nearmiss(many, "en", rng()) is None  # 1 qubits
    one = simulated("Simulate Grover search using 1 qubit.", 1)
    assert sm.build_nearmiss(one, "en", rng()) is None  # 64 qubit
    core = sm.build_nearmiss(simulated("Simulate Grover search, qubits: 64.", 64), "en", rng())
    assert core is not None and core.variant["value"] == 1  # no noun after it


BAKE = tool("get_recipe", {"ingredients": {"type": "array", "items": {"type": "string"},
                                           "description": "Items"},
                           "baking_time": {"type": "string", "enum": ["Short", "Medium", "Long"],
                                           "description": "'Short' is under 30 minutes, 'Medium' 30-60."}},
            ["ingredients"])


def test_an_enum_near_miss_needs_the_option_spelled_as_listed():
    # "short, around 30 minutes or less" for Short: a word of the prose, which the definition ties to its
    # option
    prose = sample("toolace", "b1", "Find a recipe with chocolate; baking time: SHORT, around 30 minutes.",
                   [("get_recipe", {"ingredients": ["chocolate"], "baking_time": "Short"})], [BAKE])
    assert sm.build_nearmiss(prose, "en", rng()) is None
    listed = sample("toolace", "b2", "Find a recipe with chocolate; baking time: Short.",
                    [("get_recipe", {"ingredients": ["chocolate"], "baking_time": "Short"})], [BAKE])
    core = sm.build_nearmiss(listed, "en", rng())
    assert core is not None and core.variant["reason"] == "enum"


BUSINESS = tool("DeleteBusiness", {
    "ProjectIdentifier": {"type": "string",
                          "description": "DataWorks工作空间名称。用来确定本次调用的工作空间。"},
    "BusinessId": {"type": "integer", "description": "业务流程的ID，可根据名称查询对应业务流程的ID。"}})


@pytest.mark.parametrize("user, found", [
    ('我想删除名称为"数据处理"的业务流程。', "slot_mismatch"),  # a business process's name, not a space's
    ('删除工作空间名称为"数据处理"的业务流程。', None),  # the label belongs to the space
    ('删除名称为"数据处理"的工作空间。', None),
])
def test_a_value_named_for_another_parameters_entity_is_in_the_wrong_parameter(user, found):
    assert sm.call_problem(user, [("DeleteBusiness", {"ProjectIdentifier": "数据处理"})],
                           [BUSINESS["function"]["parameters"]]) == found


SLOT = tool("book_slot", {"minutes": {"type": "integer", "minimum": 15, "maximum": 240,
                                      "description": "Length"},
                          "guest": {"type": "string", "description": "Guest"}}, ["minutes", "guest"])


def test_a_replaced_value_keeps_its_article_agreeing():
    # hand check, round 1: "a accurate model", "a Autumn dish"
    record = sample("toolace", "g1", "Book a 90 minute slot for Ana.",
                    [("book_slot", {"minutes": 90, "guest": "Ana"})], [SLOT])
    vowels = 0
    for seed in range(30):
        core = sm.build_incorrect(record, "en", rng(str(seed)), category="obvious")
        bad, text = core.invalid[0].value, core.history[-1][1]
        if bad[0].isalpha():
            vowels += bad[0].lower() in "aeiou"
            assert ("an " if bad[0].lower() in "aeiou" else "a ") + bad in text, text
    assert vowels  # some draw put a value opening with a vowel after the article
    assert sm.agree_article("Use an slow model", 7) == "Use a slow model"
    assert sm.agree_article("Take A apple", 7) == "Take An apple"
    assert sm.agree_article("查询一个实例", 4) == "查询一个实例"


# ---- the fix round's review: templates (F1), cuts (F3), case (F4), own names (F5), numbers in names (F6),
# the request's language (F7) ------------------------------------------------------------------------------


@pytest.mark.parametrize("text, tokens", [
    ("Can you please book a flight for me from New York to London?", {"book", "flight"}),
    ("Can you book a table for two at Luigi's restaurant tonight?",
     {"book", "table", "two", "restaurant", "tonight"}),
    ('Find the movie "The Matrix" from 1999, please.', {"find", "movie"}),  # quoted and numbers out
    ("What is the weather in Paris?", {"weather"}),  # question words and names out
    ("Send it to ana.lima@example.org with code AB12 today.", {"send", "code", "today"}),  # identifiers out
])
def test_template_tokens_leave_out_names_numbers_identifiers_and_quoted_values(text, tokens):
    assert sm.template_tokens(text) == frozenset(tokens)


def test_chinese_template_tokens_are_pairs_within_runs_of_characters():
    one = sm.template_tokens('查询工作空间ID为"789"中用户ID为"123456"的成员角色信息。')
    other = sm.template_tokens('查询工作空间ID为"42"中用户ID为"7"的成员角色信息。')
    assert one == other and {"工作", "空间", "成员", "角色"} <= one
    assert not any(re.search(r"[A-Za-z0-9]", token) for token in one)  # identifiers and numbers split runs
    assert "间为" not in one and "为中" not in one


@pytest.mark.parametrize("a, b, test", [
    ("Can you please book a flight for me from New York to London?", "Can you book a flight to Paris?",
     "jaccard"),
    ("Can you order a pizza from the nearest pizzeria?", "Can you order a pizza from the nearest restaurant?",
     "jaccard"),  # the review's sdev-b-0029: content words 0.6 alike, under D1's 0.8
    ("你能帮我订一张从纽约到伦敦的机票吗？", "你能帮我预订一张从纽约到伦敦的机票吗？",
     "jaccard"),  # sdev-b-0002
    ('获取工作空间ID为"789"中用户ID为"123456"的成员角色信息。',
     '我想获取工作空间ID为"5"中用户ID为"6"的成员角色信息', "jaccard"),  # sdev-b-0187 to 0189
    ("Order a pizza with extra cheese.",
     "Order a pizza with extra cheese and olives for dinner, delivered to the house quickly.", "contains"),
    ("Can you order a pizza for me?", "Can you book a flight for me?", None),
    ("查询工作空间的成员列表。", "删除实例的快照。", None),
])
def test_requests_of_one_template(a, b, test):
    assert sm.same_template(sm.template_tokens(a), sm.template_tokens(b)) == test
    assert sm.same_template(sm.template_tokens(b), sm.template_tokens(a)) == test


def test_template_index_finds_what_pairwise_comparison_finds():
    texts = [("irrelevant", "Can you please book a flight for me from New York to London?"),
             ("irrelevant", "Can you book a flight to Paris?"),
             ("missing", "Can you book a flight to Rome tomorrow?"),
             ("irrelevant", "Can you order a pizza for me?"),
             ("irrelevant", "Order a pizza with extra cheese and olives for dinner, delivered to the house."),
             ("valid_verbatim", "Order a pizza with extra cheese and olives."),
             ("missing", "你能帮我订一张从纽约到伦敦的机票吗？"),
             ("missing", "你能帮我预订一张从纽约到伦敦的机票吗？"),
             ("irrelevant", "删除实例的快照。"), ("irrelevant", "")]
    index = sm.TemplateIndex()
    for number, (kind, text) in enumerate(texts):
        index.add(number, kind, [text])
    for kind, text in texts:
        tokens = sm.template_tokens(text)
        expected = {other: found for other, (_, b) in enumerate(texts)
                    if (found := sm.same_template(tokens, sm.template_tokens(b)))}
        assert index.matches([text]) == expected, text
        assert index.matches([text], kind=kind) == {k: v for k, v in expected.items() if texts[k][0] == kind}


@pytest.mark.parametrize("text, values, described, expected", [
    # the review's sdev-b-0195: the clause holding the verb stays; each label goes with its value, with the
    # 、 and the postposition after it
    ('我想删除工作空间ID为"123456"、项目空间ID为"789"下非运行状态的作业实例信息。', ["123456", "789"],
     ["工作空间ID。", "项目空间ID。"], "我想删除非运行状态的作业实例信息。"),
    ('获取工作空间ID为"789"中用户ID为"123456"的成员角色信息。', ["789", "123456"],
     ["工作空间ID。", "用户ID。"], "获取成员角色信息。"),  # sdev-b-0188 left 获取中成员角色信息
    ('用户ID为"123456"的用户在首页看板中展示的常看和足迹列表有哪些？', ["123456"], ["ID。"],
     "用户在首页看板中展示的常看和足迹列表有哪些？"),  # the noun the label repeats goes too: no 用户用户
    ('查询同步实例ID为"789012"的同步任务的详情。', ["789012"], ["实例ID。"], "查询同步任务的详情。"),
    ('获取项目名称为"finance"、角色名称为"admin"的角色ACL授权信息。', ["finance", "admin"],
     ["项目名称。", "角色名称。"], "获取角色ACL授权信息。"),  # no 角色角色
    ('获取父用户组ID为"123"下的子用户组信息。', ["123"], ["父用户组ID。"], "获取子用户组信息。"),
    ("Can you give me the list of currencies and their exchange rates based on USD?", ["USD"], [""],
     "Can you give me the list of currencies and their exchange rates?"),  # the participle goes with it
    ("I'm in the mood for a comedy movie.", ["comedy"], [""], "I'm in the mood for a movie."),  # a modifier
    # hand check, round 2 of this fix round (sdev-b-0227 read "and an theme?"): the article agrees
    ("Set the direction with an Enchanted Forest theme, please.", ["Enchanted Forest"], [""],
     "Set the direction with a theme, please."),
    ('Check the domain with the extension "com"—it matters.', ["com"], [""],
     "Check the domain with the extension—it matters."),  # sdev-a-0116 left "extension —it"
    ("I bought 3 apples today.", ["3"], [""], "I bought apples today."),  # the verb keeps a noun: round 3
    ("I want to travel to Paris from London.", ["Paris"], [""], "I want to travel from London."),  # round 4
    ("Can you find Italian restaurants in Rome?", ["Italian"], [""], "Can you find restaurants in Rome?"),
    # round 5 (sdev-a-0057 read "to buy a car worth."): worth takes its amount as a preposition does
    ("Hi, I plan to buy a car worth $25000. I can pay $5000 now.", ["25000"], [""],
     "Hi, I plan to buy a car. I can pay $5000 now."),
    # round 6 (train:682 read "and her adversary set?", train:1190 我想实例的生命周期): a participle goes with
    # its preposition; a label taken from the description never reaches back over the request's verb
    ('Could you generate a plot with her adversary "Victor" set in a futuristic city?',
     ["Victor", "futuristic city"], ["", ""], "Could you generate a plot with her adversary?"),
    ('我想获取实例ID为"12345"的实例的生命周期。', ["12345"], ["要获取的实例ID。"],
     "我想获取实例的生命周期。"),
    # round 8 (a training item read 名称匹配为\"\"的): quotes escaped with a backslash go with the value
    ('查一下名称匹配为\\"sales\\"的数据湖元数据库列表。', ["sales"], [""],
     "查一下名称匹配的数据湖元数据库列表。"),
    # hand check, round 1 of this fix round: a label after a preposition goes with it (sdev-b-0184 left
    # 我想知道在工作空间ID的成员角色信息), and a marker such as 将 takes the noun after a label cut whole
    ('我想知道用户ID为"123456"在工作空间ID为"789"的成员角色信息。', ["123456", "789"],
     ["用户ID。", "工作空间ID。"], "我想知道成员角色信息。"),
    ('在工作空间ID为"789"中，用户ID为"123456"的成员角色是什么？', ["789"], ["工作空间ID。"],
     '用户ID为"123456"的成员角色是什么？'),
    ('将表的唯一标识为"12345"的表添加到类目ID为"1"的类目中。', ["12345"], ["表的唯一标识。"],
     '将表添加到类目ID为"1"的类目中。'),
    # sdev-a-0100: a comma joining two labelled values, as 、 does, goes with the first
    ('我想获取项目名称为"project1"，分区表为"table1"的分区表达式。', ["project1", "table1"],
     ["项目名称。", "分区表名称。"], "我想获取分区表达式。"),
])
def test_cuts_keep_the_verb_and_leave_no_word_without_its_partner(text, values, described, expected):
    lang = "zh" if re.search(r"[一-鿿]", text) else "en"
    cut = sm.cut_with_frames(text, values, lang, described)
    assert cut is not None and cut.text == expected and cut.fluent, cut


@pytest.mark.parametrize("text, values, left", [
    ("My total cost was $45.", ["45"], "My total cost was."),  # the currency sign goes with its number
    ("Could you search for some entertaining anime-related jokes to cheer me up?", ["anime"], None),
    ("Hi, can you check the status of my flight? It's BA 283.", ["BA 283"], None),  # "It's." is left
    ("Find restaurants in Paris serving sushi.", ["Paris"], None),  # "in serving": no modifier was cut
    # hand check, round 1 of this fix round (sdev-b-0198, 0213, 0227)
    ("I am planning to visit New York next week. Can you find events there?", ["New York", "next week"],
     None),  # "to visit."
    ("Hi, I need to get from New York to Los Angeles. Can you find the shortest path?",
     ["New York", "Los Angeles"], None),  # "to get."
    ("Here's the API key: ABC123XYZ789.", ["ABC123XYZ789"], None),  # "key:."
    # hand check, round 1 of this fix round, set A (sdev-a-0101, 0112, 0117, 0119)
    ('Set the direction for our production with Johnny Harris as the director.', ["Johnny Harris"],
     None),  # "production as the director"
    ("Analyze this data:\n```csv\nid,amount\n1,500\n```\nThanks.", ["id,amount\n1,500"],
     None),  # an empty fence
    ("The ID is L1. The period ranges from 2026-07-01 to 2026-09-30, please.", ["2026-07-01", "2026-09-30"],
     None),  # "the period ranges"
    ('Serialize it. The details are as follows {"name": "Ana"}.', ['{"name": "Ana"}'], None),  # "as follows."
    # round 2 (sdev-b-0208): the noun cut after the word describing it ("find nearby within a radius")
    ("I want to find nearby restaurants within a radius of 500 meters.", ["restaurants"], None),
    ("What is the carbon footprint of my daily car commute?", ["car commute"], None),  # "my daily?"
    # round 2 (sdev-a-0103, 0115): a verb's object cut at a sentence's end, an emptied list field
    ("I want to meet new people and pursue my hiking hobby. Can you help?", ["my hiking hobby"], None),
    ("Details:\n- User ID: user_12345\n- Type: checking", ["user_12345"], None),
    # round 3 (sdev-a-0063 read "I consumed xyz calories and burned."): an amount cut from after its verb
    ("I consumed 2000 calories and burned 500.", ["500"], None),
    ("I need 2 tickets for Friday.", ["2"], None),  # "I need for Friday."
    # round 4 (sdev-a-0054, 0063): a copula left before a conjunction, a verb's object cut before a
    # preposition or a sentence's end
    ("It was originally priced at $100 but it's on sale for 20% off.", ["100"], None),  # "originally but"
    ("I need to add apples to my shopping list.", ["apples"], None),  # "to add to my shopping list"
    ("Could you please play Bohemian Rhapsody?", ["Bohemian Rhapsody"], None),  # "please play?"
    ("My budget is approximately $500 per night.", ["500"], None),  # round 5: "approximately per night"
    # round 6, a sample of training cuts (train:660, 727, 823, 825, 848 of the r2i build): a containment verb,
    # an amount's "off", a participle or "according" without what completes it, a preposition the list lacked,
    # a verb after a relative pronoun
    ('同步实例名称包含"test"的数据同步实例有哪些？', ["test"], None),  # 名称包含数据同步实例
    ("I saw a jacket I liked but it's 15% off. How much is it?", ["15"], None),  # "it's off"
    ("It's a sci-fi movie released in 2019.", ["sci-fi", "2019"], None),  # "a movie released."
    ("Can you explain the ideas behind Quantum Mechanics according to Niels Bohr?",
     ["Quantum Mechanics", "Niels Bohr"], None),  # "behind according?"
    ("I'm looking for a recipe that uses chicken.", ["chicken"], None),  # "a recipe that uses."
])
def test_cuts_that_break_the_request_are_not_fluent(text, values, left):
    cut = sm.cut_with_frames(text, values, "zh" if re.search(r"[一-鿿]", text) else "en")
    assert cut is not None and not cut.fluent, cut
    if left is not None:
        assert cut.text == left


@pytest.mark.parametrize("text, original, fluent", [
    ("用户用户在首页看板中。", '用户ID为"1"的用户在首页看板中。', False),  # a doubled word the cut made
    ("哈哈哈哈，真好。", "哈哈哈哈，真好。", True),  # the request's own
    ("下非运行状态的作业实例信息。", "我想删除工作空间ID为1下非运行状态的作业实例信息。",
     False),  # it opens with 下
    ("获取中成员角色信息。", "获取工作空间789中成员角色信息。", False),  # a verb before a postposition
    ("I need it. The dimensions.", "I need it. The dimensions are 5 by 6.", False),  # a two-word sentence
    ("Thanks. Book it.", "Thanks. Book it.", True),  # the request's own short sentences
    ("Search for some entertaining -related jokes.", "Search for some entertaining anime-related jokes.",
     False),
    ("My total cost was $ today.", "My total cost was $45 today.", False),  # a currency sign, no number
    ("The rates based?", "The rates based on USD?", False),  # a participle without its object
    ("It was originally, but now on sale.", "It was originally priced at 200, but now on sale.", False),
    ("I am currently. Find me a restaurant.", "I am currently in Lima. Find me a restaurant.", False),
    ("I'm currently. Find me a restaurant.", "I'm currently in Lima. Find me a restaurant.", False),
    ("Is it positive? Here it is:", "Is it positive? Here it is: Roses bloom.", False),  # it ends open
    ("Is it positive? Here it is: Roses bloom.", "Is it positive? Here it is: Roses bloom.", True),
    ("Find a force. The object weighs.", "Find a force. The object weighs 30 kg.", False),  # three words
    ("Please call me. I need it now.", "Please call me. I need it now.", True),
])
def test_the_fluency_heuristic_compares_with_the_original(text, original, fluent):
    lang = "zh" if re.search(r"[一-鿿]", text) else "en"
    assert sm.reads_fluently(text, lang, original) is fluent


VOLUME = tool("calculate_volume", {"dimensions": {
    "type": "object", "description": "Box size",
    "properties": {"length": {"type": "number"}, "width": {"type": "number"}, "height": {"type": "number"}}}},
    ["dimensions"])


def test_an_object_introduced_by_its_parameters_name_goes_with_it():
    # the review's sdev-b-0200: "... The dimensions." was left after the object's cut
    request = ("Hi, I need to calculate the volume of a box. The dimensions are length: 5, width: 6, and "
               "height: 7.")
    s = sample("glaive_v2", "v1", request,
               [("calculate_volume", {"dimensions": {"length": 5, "width": 6, "height": 7}})], [VOLUME])
    core = sm.build_missing(s, "en", rng(), subkind="object")
    assert core is not None and core.history[-1][1] == "Hi, I need to calculate the volume of a box."
    assert core.variant["cut"] == "frame"


@pytest.mark.parametrize("user, arguments, found", [
    ("I need to remember to buy groceries.", {"task": "Buy groceries"}, "ungrounded"),  # sdev-b-0110
    ("One for work tasks and one for home tasks.", {"title": "Work Tasks"}, "ungrounded"),  # sdev-a-0050
    ("Based on my interest in music and art.", {"activityType": ["Music", "Art"]},
     "ungrounded"),  # sdev-a-0037
    ("Add Buy groceries to my list.", {"task": "Buy groceries"}, None),
    ("Pull up the Recent feed for #HealthyEating.", {"feed_type": "recent", "hashtag": "#HealthyEating"},
     "ungrounded"),  # an option of an enum too: verbatim or no item (D3)
])
def test_a_gold_string_is_given_only_as_written(user, arguments, found):
    # F4: the development scorer compares values case by case (ace_probe.same_calls), so a value counts as
    # given only verbatim, in its own case
    assert problem(user, arguments) == found


def test_a_value_given_verbatim_only_inside_another_value_is_derived():
    # hand check, round 1 of this fix round (sdev-a-0036): "MP3 format" in the prose, and "mp3" verbatim only
    # as the end of the audio URL: the format is not given as written
    user = "Transcribe the podcast; it is in MP3 format at https://cloud.example.com/pod/ep-42.mp3 now."
    arguments = {"audio_url": "https://cloud.example.com/pod/ep-42.mp3", "format": "mp3"}
    assert problem(user, arguments) == "inside_other_value"
    assert problem(user.replace("MP3 format", "mp3 format"), arguments) is None


@pytest.mark.parametrize("key, value", [("nickname", "nickname"), ("planName", "planName"),
                                        ("FolderId", "folder_id"), ("InstanceId", "instanceId"),
                                        ("tenant-id", "Tenant ID"),
                                        # hand check, round 1 of this fix round: credentials given as names
                                        ("api_key", "my_api_key"), ("api_secret", "YOUR_API_SECRET")])
def test_a_value_equal_to_its_own_parameter_name_is_a_placeholder(key, value):
    user = f'我想更新配额计划，{key}为"{value}"。'
    assert problem(user, {key: value}) == "placeholder_value"
    assert not sm.bank_value_ok(value, user, "", key=key)
    assert sm.bank_value_ok("Q3 plan", '我想更新配额计划，planName为"Q3 plan"。', "", key="planName")


def test_a_value_change_needs_a_short_old_value():
    # hand check, round 1 of this fix round: "change I am feeling really happy today. The sun is shining and
    # the birds are singing. to I am feeling really happy today!" reads as one run-on sentence
    sentiment = tool("analyze_sentiment", {"text": {"type": "string", "description": "Text"}}, ["text"])
    long = "I am feeling really happy today. The sun is shining and the birds are singing."
    record = sample("glaive_v2", "s1", f"Can you analyze the sentiment of this text: {long}",
                    [("analyze_sentiment", {"text": long})], [sentiment])
    assert sm.build_change(record, "en", rng(), bank={"text": ["Good day!"]}, packed=False) is None
    short = sample("glaive_v2", "s2", "Can you analyze the sentiment of this text: Nice job",
                   [("analyze_sentiment", {"text": "Nice job"})], [sentiment])
    assert sm.build_change(short, "en", rng(), bank={"text": ["Good day!"]}, packed=False) is not None
    assert sm.names_own_key("api_key", "my_api_key") and not sm.names_own_key("api_key", "abc123")


def test_the_reviews_call_of_parameter_names_makes_no_item():
    # F5, sdev-b-0184: deepexi 15999 called UpdateQuotaPlan with each value its parameter's own name
    user = '我想更新Quota计划，计划名称为"planName"，地域id为"region"，请求体参数为"body"。'
    assert problem(user, {"planName": "planName", "region": "region", "body": "body"}) == "placeholder_value"


@pytest.mark.parametrize("user, arguments, found", [
    ("Get the pit stop data for the 2026 Formula 1 season, specifically for the first round.",
     {"season": 2026, "round": 1}, "ungrounded"),  # sdev-b-0130: the 1 of Formula 1 is no round
    ("Book 2 rooms in Lisbon.", {"rooms": 2, "city": "Lisbon"}, None),  # a capital opening the request
    ("Is my laptop able to run Windows 10 tonight?", {"version": 10}, "ungrounded"),
])
def test_a_number_inside_a_name_gives_no_value(user, arguments, found):
    properties = {k: {"type": "integer" if isinstance(v, int) else "string"} for k, v in arguments.items()}
    assert problem(user, arguments, properties) == found


RACES = tool("get_results", {"round": {"type": "integer", "description": "Round number"},
                             "series": {"type": "string", "description": "Series"}}, ["round"])


def test_a_cut_number_the_request_still_gives_in_words_makes_no_missing_item():
    # F6: "... the third round" still gives round 3 after the 3 is cut
    ordinal = sample("toolace", "o1", "Show the results of the GT series, round 3, the third round.",
                     [("get_results", {"round": 3, "series": "GT"})], [RACES])
    assert sm.build_missing(ordinal, "en", rng(), subkind="single") is None
    plain = sample("toolace", "o2", "Show the results of the GT series, round 3.",
                   [("get_results", {"round": 3, "series": "GT"})], [RACES])
    core = sm.build_missing(plain, "en", rng(), subkind="single")
    assert core is not None and core.missing == ["round"] and "3" not in core.history[-1][1]
    assert sm.still_numbered("the third round of it", "round", 3)
    assert sm.still_numbered("two tickets, please", "num_tickets", 2)
    assert not sm.still_numbered("the third lap of it", "round", 3)
    assert not sm.still_numbered("the third round of it", "round", "R3")  # a number only


CARS = tool("car_emissions", {"car_make": {"type": "string", "description": "Make"},
                               "car_model": {"type": "string", "description": "Model"},
                               "distance": {"type": "integer", "description": "Kilometers driven"}},
            ["car_make", "car_model", "distance"])


def test_a_value_touching_another_value_is_not_cut():
    # hand check, round 1 of this fix round (sdev-b-0112): with "Toyota" cut from "a Toyota Corolla", the
    # model left beside it gives the make away
    record = sample("glaive_v2", "t1", "I have a Toyota Corolla and I drove it 150 kilometers. Emissions?",
                    [("car_emissions", {"car_make": "Toyota", "car_model": "Corolla", "distance": 150})],
                    [CARS])
    for seed in range(6):
        core = sm.build_missing(record, "en", rng(str(seed)), subkind="single")
        assert core is not None and core.missing == ["distance"], core.missing
    assert sm.build_missing(record, "en", rng(), subkind="two") is None
    assert sm.touches_other_value("a Toyota Corolla", "Toyota", ["Corolla", "150"])
    assert sm.touches_other_value("北京朝阳区", "北京", ["朝阳区"])
    assert not sm.touches_other_value("Toyota, Corolla", "Toyota", ["Corolla"])


@pytest.mark.parametrize("source, text, lang", [
    ("toolace", "角色定义：询问者是提出问题的用户。请帮我查一下明天北京的天气。", "zh"),
    ("deepexi", "How do I stop a running ECS instance?", "en"),
    ("glaive_zh", "Can you book a flight to Paris for me?", "en"),
    ("deepexi", '把报表名称改为"Sales Report Q3"。', "zh"),  # a quoted value is no part of the language
    ("toolace", 'Translate "你好世界" into English, please.', "en"),
    ("deepexi", "4471", "zh"), ("hermes", "4471", "en"),  # no word: the source's language
])
def test_the_requests_language_is_read_from_its_text(source, text, lang):
    assert sm.request_lang(source, text) == lang


def test_a_chinese_request_of_an_english_source_gets_chinese_wrong_values():
    # F7: a Chinese request of ToolACE got "a few" (房间长a few米)
    room = tool("room_area", {"length": {"type": "integer", "description": "Length in meters"},
                              "width": {"type": "integer", "description": "Width in meters"}},
                ["length", "width"])
    s = sample("toolace", "z1", "房间长5米，宽4米，帮我算一下面积。",
               [("room_area", {"length": 5, "width": 4})], [room])
    for seed in range(6):
        core = sm.build_incorrect(s, "en", rng(str(seed)), category="vague")
        assert core is not None and core.invalid[0].value in sm.VAGUE["number"]["zh"], core.invalid[0].value


@pytest.mark.parametrize("text, old, new, lang, fits", [
    # hand check, round 3 of this fix round: sdev-a-0057 read "the rate is some%", sdev-a-0060 "is $a dozen"
    ("The annual interest rate is 3.5% and the term is 60 months.", "3.5", "some", "en", False),
    ("The annual interest rate is 3.5% and the term is 60 months.", "60", "some", "en", True),
    ("My hourly rate is $15 and I worked 40 hours.", "15", "a dozen or so", "en", False),
    ("My hourly rate is $15 and I worked 40 hours.", "15", "xyz", "en", False),
    ("My hourly rate is $15 and I worked 40 hours.", "15", "15.5", "en", True),  # a number stays one
    ("My hourly rate is $15 and I worked 40 hours.", "40", "a few", "en", True),
    ("Show me version 2 of the report.", "2", "several", "en", False),  # an amount counting nothing
    ("Show me version 2 of the report.", "2", "abc", "en", True),
    ("I need it by 2024-03-05.", "2024-03-05", "next week", "en", True),  # a vague day is no amount
    # round 7: a vague word after an article or a ranking word (sdev-a-0077 "in the any of them", training
    # "during the some season", "the top several"), a vague option as a modifier, a vague day after on
    ("Which platforms are available in the US?", "US", "any of them", "en", False),
    ("Which platforms are available in the US?", "US", "banana", "en", True),
    ("Who led the NBA in assists during the 2019 season?", "2019", "some", "en", False),
    ("Can you provide the top 100 career records?", "100", "several", "en", False),
    ("Find me a comedy movie.", "comedy", "whichever", "en", False),
    ("Set the genre to comedy.", "comedy", "whichever", "en", True),
    ("I need it on 2024-03-05.", "2024-03-05", "next week", "en", False),
    ("Remind me at 15:00.", "15:00", "in the afternoon", "en", False),
    ("查询状态为RUNNING的任务。", "RUNNING", "随便哪个", "zh", False),
    ("把状态设置为RUNNING。", "RUNNING", "随便哪个", "zh", True),
    # round 8 (training items read "in something in between and Asia", "product IDs: some which ..."): no
    # vague word in a coordination, no vague amount before a relative word
    ("Is it available in Europe and Asia?", "Europe", "something in between", "en", False),
    ("Is it available in Europe or Asia?", "Asia", "whichever", "en", False),
    ("The product IDs are 101 which is a mouse, and 102.", "101", "some", "en", False),
    ("查询华东、华北的实例。", "华东", "随便哪个", "zh", False),
    # round 9 (a training item read "say about 100 to a few characters"): no vague amount in a range
    ("Give me a quote of about 100 to 150 characters.", "150", "a few", "en", False),
    ("Pick 3-5 items.", "5", "several", "en", False),
    ("请给我100到150个字的名言。", "150", "几十", "zh", False),
    # round 5: sdev-a-0063 read "I also consume around some kilowatt-hours"
    ("I also consume around 500 kilowatt-hours a month.", "500", "some", "en", False),
    ("I also consume around 500 kilowatt-hours a month.", "500", "xyz", "en", True),
    # sdev-b-0119 read 版本号为几个的详细信息
    ("查一下元数据表名称为table，版本号为2的详细信息。", "2", "几个", "zh", False),
    ("帮我查一下工作空间4471里的任务，每页30条。", "4471", "若干", "zh", False),
    ("帮我查一下工作空间4471里的任务，每页30条。", "30", "十几", "zh", True),
    ("帮我查一下工作空间4471里的任务，每页30条。", "30", "几个", "zh", False),  # 几个条
    ("帮我订一张2人的桌子。", "2", "几个", "zh", True),  # 几个人
    ("房间长5米，宽4米。", "5", "若干", "zh", True),
    ("房间长5米，宽4米。", "5", "一些", "zh", False),
    ("年利率为3.5%。", "3.5", "十几", "zh", False),
    ("价格是¥200。", "200", "几十", "zh", False),
])
def test_a_wrong_value_fits_where_the_value_stood(text, old, new, lang, fits):
    assert sm.wrong_value_fits(text, old, new, lang) is fits


LOAN = tool("loan_payment", {
    "price": {"type": "integer", "description": "Price in dollars"},
    "rate": {"type": "number", "description": "Annual interest rate in percent"},
    "months": {"type": "integer", "description": "Term in months"}}, ["price", "rate", "months"])


STAY = tool("book_stay", {"check_in": {"type": "string", "format": "date", "description": "Check-in day"},
                          "guests": {"type": "integer", "description": "Number of guests"}},
            ["check_in", "guests"])


def test_a_value_the_request_gives_again_in_other_words_is_not_made_wrong():
    # hand check, round 7 of this fix round (sdev-b-0145, 0151, 0161; a training item): 25/04/2023 beside
    # "April 25th", 5.5 beside "five team members", left the item to be read either way
    request = "I check in on April 25th with five guests. Details: check_in 2023-04-25, guests 5."
    s = sample("hermes", "h1", request, [("book_stay", {"check_in": "2023-04-25", "guests": 5})], [STAY])
    for category in ("obvious", "format", "vague"):
        for seed in range(6):
            assert sm.build_incorrect(s, "en", rng(str(seed)), category=category) is None, (category, seed)
    date = sm.infer_constraint(STAY["function"]["parameters"]["properties"]["check_in"])
    assert sm.gives_again(request, date, "2023-04-25")
    assert not sm.gives_again("Details: check_in 2023-04-25.", date, "2023-04-25")
    number = sm.infer_constraint(STAY["function"]["parameters"]["properties"]["guests"])
    assert sm.gives_again(request, number, 5) and not sm.gives_again("guests 5", number, 5)
    # round 8 (sdev-b-0176): "for one adult" beside "Number of Guests: 1", but not "one of" or "this one"
    assert sm.gives_again("It is for one adult. Number of Guests: 1", number, 1)
    assert not sm.gives_again("Pick one of them, not this one. Number of Guests: 1", number, 1)


def test_wrong_values_never_stick_to_a_currency_or_percent_sign():
    # hand check, round 3 of this fix round (sdev-a-0057, 0060): a vague amount went where only a number fits
    request = "The price is $25000, the rate is 3.5% and I want to pay it off in 60 months."
    s = sample("glaive_v2", "L1", request, [("loan_payment", {"price": 25000, "rate": 3.5, "months": 60})],
               [LOAN])
    stuck = re.compile(r"[$€£¥][ \t]*[A-Za-z]|[A-Za-z][ \t]*%")
    for seed in range(12):
        core = sm.build_incorrect(s, "en", rng(str(seed)), category="vague")
        assert core is not None and core.invalid[0].param == "months", core
        for core in (sm.build_incorrect(s, "en", rng(str(seed)), category="obvious"),
                     sm.build_incorrect(s, "en", rng(str(seed)), category="obvious", two=True),
                     sm.build_priority(s, "en", rng(str(seed)))):
            assert core is None or not stuck.search(core.history[-1][1]), core.history[-1][1]

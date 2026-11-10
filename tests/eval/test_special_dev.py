"""research_agent.eval.special_dev (M8 WP2): the items of the refusal development sets, the scorer of an
answer (strict and lenient on the same answers, every failure mode, the placeholders and the slots, the calls
in the three notations and the loose ones, a greedy loop cut by the answer budget, the answers its parsers
cannot read, the zh alterations, the false refusals) and the summary of a run.
The sentence families here are the tests' own wording, apart from those built from
research_agent.data.special_mix (read only) to stand for the data package's sets, and the real development
sets, read when they are built and match their locks."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from research_agent.data import special_mix as sm
from research_agent.eval import special_dev as sd
from research_agent.eval.lock import DevSet, LockError

REPO = Path(__file__).resolve().parents[2]

BOOK_ROOM = {
    "name": "book_room",
    "description": "Book a meeting room for a number of people on a date.",
    "parameters": {
        "type": "object",
        "properties": {
            "room": {"type": "string", "description": "the room's code"},
            "people": {"type": "integer", "minimum": 1},
            "date": {"type": "string", "description": "the day, as YYYY-MM-DD"},
        },
        "required": ["room", "people", "date"],
    },
}
PLAN_TRIP = {
    "name": "plan_trip",
    "description": "Plan a trip for one traveller.",
    "parameters": {
        "type": "object",
        "properties": {
            "traveller": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "passport": {"type": "string"}},
                "required": ["name", "passport"],
            },
            "destination": {"type": "string"},
        },
        "required": ["traveller", "destination"],
    },
}
QUERY_STOCK = {
    "name": "query_stock",
    "description": "查询商品库存",
    "parameters": {
        "type": "object",
        "properties": {
            "item": {"type": "string", "description": "商品名称"},
            "count": {"type": "integer", "minimum": 1, "description": "需要的数量"},
        },
        "required": ["item", "count"],
    },
}
FUNCTIONS = {"en": (BOOK_ROOM, PLAN_TRIP), "zh": (QUERY_STOCK,)}

# family -> case -> (the sentence as the system message prescribes it, its fixed words)
FAMILIES = {
    "fx-en": {
        "incorrect": ("Not sent: the value <value> of <param> breaks the definition.",
                      ["Not sent: the value", "of", "breaks the definition."]),
        "missing": ("Not sent: <api> still needs <params>", ["Not sent:", "still needs"]),
        "irrelevant": ("Not sent: no listed function can do this.",
                       ["Not sent: no listed function can do this."]),
    },
    "fx-zh": {
        "incorrect": ("无法执行：参数【参数名】收到的值【取值】不符合定义。",
                      ["无法执行：参数", "收到的值", "不符合定义。"]),
        "missing": ("无法执行：{{函数}}缺少{{参数}}", ["无法执行：", "缺少"]),
        "irrelevant": ("无法执行：现有函数都不适用。", ["无法执行：现有函数都不适用。"]),
    },
    "fx-plain": {  # placeholders that are plain words
        "incorrect": ("Rejected: value for field is not allowed.", ["Rejected:", "for", "is not allowed."]),
        "missing": ("Rejected: function lacks parameters", ["Rejected:", "lacks"]),
        "irrelevant": ("Rejected: nothing listed fits.", ["Rejected: nothing listed fits."]),
    },
    "fx-caps": {  # placeholders in capitals, and a full stop as a fixed word
        "incorrect": ("Refused: PARAM cannot be VALUE.", ["Refused:", "cannot be", "."]),
        "missing": ("Refused: API lacks PARAMS", ["Refused:", "lacks"]),
        "irrelevant": ("Refused: no function fits.", ["Refused: no function fits."]),
    },
    "fx-edge": {  # a placeholder before the first fixed word and one after the last
        "incorrect": ("<param> got the bad value <value>", ["got the bad value"]),
        "missing": ("<api> is short of <params>", ["is short of"]),
        "irrelevant": ("Out of scope for every function.", ["Out of scope for every function."]),
    },
    "fx-angle": {  # the brackets belong to the fixed words, as the data package writes its sentences
        "incorrect": ("Not accepted: <parameter> cannot hold <value>.",
                      ["Not accepted: <", "> cannot hold <", ">."]),
        "missing": ("Not accepted: <function> lacks <parameters>.", ["Not accepted: <", "> lacks <", ">."]),
        "irrelevant": ("Not accepted: nothing here fits.", ["Not accepted: nothing here fits."]),
    },
    "fx-blind": {  # slots whose placeholders do not say which holds what
        "incorrect": ("Blocked: <first> versus <second>.", ["Blocked:", "versus", "."]),
        "missing": ("Blocked: <one> beside <two>.", ["Blocked:", "beside", "."]),
        "irrelevant": ("Blocked: nothing fits.", ["Blocked: nothing fits."]),
    },
}
LINES = {
    "en": ("You may call the functions below; each line is one function as JSON.", "How to reply:",
           "- When the request can be served, reply only with the calls, {notation}.",
           "- When a value given breaks its definition, reply only with: {incorrect}",
           "- When a required parameter has no value, reply only with: {missing}",
           "- When both happen, answer the value first.",
           "- When no listed function fits, reply only with: {irrelevant}"),
    "zh": ("可以调用下列函数,每行是一个函数的 JSON。", "回复规则:",
           "- 能完成请求时,只回复调用,{notation}。",
           "- 某个参数给出的值不符合定义时,只回复:{incorrect}",
           "- 缺少必需参数时,只回复:{missing}",
           "- 两种情况同时出现时,先答取值。",
           "- 请求与所有函数都无关时,只回复:{irrelevant}"),
}
NOTATION = {
    "en": {"python": "one call per line, written as Python calls with keyword arguments",
           "json": "as one JSON array of objects with the keys name and arguments",
           "yaml": "as a YAML list whose items hold a name and an arguments mapping"},
    "zh": {"python": "每行一个调用,写成带关键字参数的 Python 调用",
           "json": "写成一个 JSON 数组,每个对象含 name 和 arguments",
           "yaml": "写成 YAML 列表,每项含 name 和 arguments"},
}


def system_text(sentences: dict[str, str], lang: str, fmt: str) -> str:
    head, title, *rules = LINES[lang]
    return "\n".join([head, *(json.dumps(f, ensure_ascii=False) for f in FUNCTIONS[lang]), "", title,
                      *(rule.format(notation=NOTATION[lang][fmt], **sentences) for rule in rules)])


def system(family: str, lang: str, fmt: str) -> str:
    return system_text({case: text for case, (text, _) in FAMILIES[family].items()}, lang, fmt)


def make_item(number: int, kind: str, family: str, user: str, expected: dict, *, fmt: str = "python",
              history: tuple[dict, ...] = ()) -> dict:
    lang = "zh" if family == "fx-zh" else "en"
    return {"id": f"sdev-a-{number:04d}", "split": "a", "lang": lang, "kind": kind,
            "messages": [{"role": "system", "content": system(family, lang, fmt)}, *history,
                         {"role": "user", "content": user}],
            "call_format": fmt, "family": family, "expected": expected,
            "source": {"dataset": "fixture", "record": f"r{number}"}}


def refusal(family: str, case: str, *, values: tuple = (), params: tuple = (), api: str = "") -> dict:
    return {"type": "refusal", "case": case, "fixed": list(FAMILIES[family][case][1]), "values": list(values),
            "params": list(params), "api": api}


def calls(*pairs: tuple[str, dict], verbatim: tuple = ()) -> dict:
    return {"type": "call", "calls": [{"name": name, "arguments": arguments} for name, arguments in pairs],
            "verbatim": list(verbatim)}


INCORRECT = make_item(1, "incorrect", "fx-en", "Book room A-12 for -3 people on 2026-11-05.",
                      refusal("fx-en", "incorrect", values=("-3",), params=("people",)))
MISSING = make_item(2, "missing", "fx-en", "Plan a trip to Lisbon.",
                    refusal("fx-en", "missing", params=("traveller",), api="plan_trip"))
IRRELEVANT = make_item(3, "irrelevant", "fx-en", "What is the weather in Oslo tomorrow?",
                       refusal("fx-en", "irrelevant"))
INCORRECT_ZH = make_item(4, "incorrect", "fx-zh", "帮我查一下苹果手机的库存,要十几台。",
                         refusal("fx-zh", "incorrect", values=("十几",), params=("count",)), fmt="json")
MISSING_ZH = make_item(5, "missing", "fx-zh", "帮我查一下库存,要 10 台。",
                       refusal("fx-zh", "missing", params=("item",), api="query_stock"), fmt="json")
IRRELEVANT_ZH = make_item(6, "irrelevant", "fx-zh", "今天北京天气怎么样?", refusal("fx-zh", "irrelevant"),
                          fmt="json")
PRIORITY = make_item(7, "priority", "fx-en", "Book a room for -3 people on 2026-11-05.",
                     refusal("fx-en", "incorrect", values=("-3",), params=("people",)))
VERBATIM = make_item(8, "valid_verbatim", "fx-en", "Book room 会议室A for 4 people on 2026-11-05.",
                     calls(("book_room", {"room": "会议室A", "people": 4, "date": "2026-11-05"}),
                           verbatim=("会议室A",)))
VERBATIM_ZH = make_item(9, "valid_verbatim", "fx-zh", "帮我查一下苹果手机的库存,要 10 台。",
                        calls(("query_stock", {"item": "苹果手机", "count": 10}), verbatim=("苹果手机",)),
                        fmt="json")
NEARMISS = make_item(10, "valid_nearmiss", "fx-en", "Book room A-12 for 1 person on 2024-02-29.",
                     calls(("book_room", {"room": "A-12", "people": 1, "date": "2024-02-29"})), fmt="yaml")
MULTITURN = make_item(11, "valid_multiturn", "fx-en", "Make it 6 people instead.",
                      calls(("book_room", {"room": "A-12", "people": 6, "date": "2026-11-05"})),
                      history=({"role": "user", "content": "Book room A-12 for 4 people on 2026-11-05."},
                               {"role": "assistant",
                                "content": 'book_room(room="A-12", people=4, date="2026-11-05")'}))
PLAIN = make_item(12, "incorrect", "fx-plain", "Book room A-12 for -3 people on 2026-11-05.",
                  refusal("fx-plain", "incorrect", values=("-3",), params=("people",)))
CAPS = make_item(13, "incorrect", "fx-caps", "Book room A-12 for 2.5 people on 2026-11-05.",
                 refusal("fx-caps", "incorrect", values=("2.5",), params=("people",)))
EDGE = make_item(14, "missing", "fx-edge", "Plan a trip to Lisbon.",
                 refusal("fx-edge", "missing", params=("traveller",), api="plan_trip"))
ANGLE = make_item(15, "incorrect", "fx-angle", "Book room A-12 for -3 people on 2026-11-05.",
                  refusal("fx-angle", "incorrect", values=("-3",), params=("people",)))
ANGLE_TWO = make_item(16, "incorrect", "fx-angle", "Book room A-12 for -3 people on 2026-13-45.",
                      refusal("fx-angle", "incorrect", values=("-3", "2026-13-45"),
                              params=("people", "date")))
ANGLE_MISSING = make_item(17, "missing", "fx-angle", "Plan a trip.",
                          refusal("fx-angle", "missing", params=("traveller", "destination"),
                                  api="plan_trip"))
ITEMS = [INCORRECT, MISSING, IRRELEVANT, INCORRECT_ZH, MISSING_ZH, IRRELEVANT_ZH, PRIORITY, VERBATIM,
         VERBATIM_ZH, NEARMISS, MULTITURN, PLAIN, CAPS, EDGE, ANGLE, ANGLE_TWO, ANGLE_MISSING]
FAMILIES_OF_SET = sd.sentence_families(ITEMS)
EN_FUNCTIONS = frozenset({"book_room", "plan_trip"})

EXACT_BOOKING = 'book_room(room="会议室A", people=4, date="2026-11-05")'
EXACT_STOCK = '[{"name": "query_stock", "arguments": {"item": "苹果手机", "count": 10}}]'
NEARMISS_YAML = '- name: book_room\n  arguments:\n    room: "A-12"\n    people: 1\n    date: "2024-02-29"'
HUGE = "0x" + "f" * 4000  # an integer of more than 4,300 decimal digits, which Python does not write


def score(answer: str | None, item: dict) -> dict:
    return sd.score_item(answer, item, families=FAMILIES_OF_SET)


# ---- items ------------------------------------------------------------------------------------------------


def test_the_fixture_items_are_well_formed():
    for item in ITEMS:
        assert sd.check_item(item) == [], item["id"]


def changed(item: dict, **expected) -> dict:
    out = copy.deepcopy(item)
    out["expected"].update(expected)
    return out


BLIND = make_item(18, "incorrect", "fx-blind", "Book room A-12 for -3 people on 2026-11-05.",
                  {"type": "refusal", "case": "incorrect", "fixed": FAMILIES["fx-blind"]["incorrect"][1],
                   "values": ["-3"], "params": ["people"], "api": ""})


@pytest.mark.parametrize(
    "item,message",
    [
        (changed(INCORRECT, fixed=["Not sent: the value", "is wrong"]), "do not stand in order"),
        ({**copy.deepcopy(MISSING), "kind": "irrelevant"}, "expects the irrelevant sentence"),
        (changed(PRIORITY, case="irrelevant"), "priority"),
        (changed(INCORRECT, values=[]), "names its invalid values"),
        (changed(INCORRECT, values=["-3", "4"]), "as many values as parameters"),
        (changed(MISSING, api=""), "names the function"),
        (changed(VERBATIM, verbatim=["会议室B"]), "is in no argument"),
        (changed(VERBATIM, calls=[{"name": "book room", "arguments": {"room": "A"}}]), "cannot be written"),
        (changed(CAPS, params=["PARAM"]), "placeholder"),  # a placeholder searched for anywhere
        (changed(INCORRECT, values=["<value>"]), "placeholder"),
        (BLIND, "cannot be told apart"),  # neither slot names a value
        (changed(INCORRECT, fixed=["Not sent: the value", "breaks the definition."]), "cannot be told apart"),
        ({**copy.deepcopy(INCORRECT), "split": "test"}, "split"),
        ({**copy.deepcopy(INCORRECT), "call_format": "xml"}, "call_format"),
        ({**copy.deepcopy(INCORRECT), "messages": INCORRECT["messages"][:1]}, "messages"),
        ({**copy.deepcopy(MULTITURN), "messages": MULTITURN["messages"][:-1]}, "the last message"),
        ({**copy.deepcopy(INCORRECT), "source": "fixture"}, "source"),
    ],
)
def test_check_item_says_what_is_wrong(item, message):
    errors = sd.check_item(item)
    assert errors and any(message in error for error in errors), errors


def test_a_placeholder_inside_a_filler_is_not_left():
    # the capitals placeholder VALUE inside the value MAX_VALUE: the filler is told from the placeholder
    item = changed(CAPS, values=["MAX_VALUE"])
    item["messages"][-1]["content"] = "Book room A-12 for MAX_VALUE people on 2026-11-05."
    assert sd.check_item(item) == []
    filled = score("Refused: people cannot be MAX_VALUE.", item)
    assert (filled["strict"], filled["placeholder_left"]) == (True, False)
    left = score("Refused: people cannot be VALUE.", item)
    assert (left["strict"], left["failure"]) == (False, "placeholder_left")


def test_a_filler_like_a_plain_word_placeholder_is_accepted():
    # a plain-word placeholder is read in its own slot alone (PLACEHOLDER_RULE), so a parameter that shares
    # another slot's word is no ambiguity: the data's own placeholders are mostly such words
    item = changed(PLAIN, params=["value"])
    assert sd.check_item(item) == []
    assert score("Rejected: -3 for value is not allowed.", item)["strict"] is True


def test_check_item_refuses_acebench_text_in_any_message():
    # a ban test: ACEBench's history heading and its incorrect-value sentence never belong in a dev item
    for text in ("Conversation history 1..t:\nBook room A-12.", "对话历史1..t:\n订会议室",
                 "There is incorrect value (x) for the parameters (y) in the conversation history."):
        item = copy.deepcopy(INCORRECT)
        item["messages"][-1]["content"] = text
        assert any("ACEBench" in error for error in sd.check_item(item)), text
    assert sd.acebench_text("[book_room(room='A-12')]")  # ACEBench's call list, through ace_probe's markers


def test_the_fixtures_hold_no_acebench_text():
    texts = [m["content"] for item in ITEMS for m in item["messages"]]
    texts += [text for family in FAMILIES.values() for text, _ in family.values()]
    assert all(sd.acebench_text(text) == [] for text in texts)


# ---- the sentences of a set -------------------------------------------------------------------------------


def test_sentence_families_keys_every_sentence_by_family_and_case():
    assert FAMILIES_OF_SET[("fx-en", "incorrect")] == {tuple(FAMILIES["fx-en"]["incorrect"][1])}
    assert FAMILIES_OF_SET[("fx-en", "missing")] == {tuple(FAMILIES["fx-en"]["missing"][1])}
    assert FAMILIES_OF_SET[("fx-plain", "incorrect")] == {tuple(FAMILIES["fx-plain"]["incorrect"][1])}
    assert ("fx-plain", "missing") not in FAMILIES_OF_SET  # valid items add no sentence
    assert len(FAMILIES_OF_SET) == 11


def test_sentence_families_keeps_the_styles_of_one_family_and_refuses_one_sentence_for_two_cases():
    # one family written in two placeholder styles gives its case two fixed-word lists, both kept
    other = changed(PRIORITY, fixed=["Not sent: the value", "breaks the definition."])
    found = sd.sentence_families([INCORRECT, other])
    assert found[("fx-en", "incorrect")] == {tuple(INCORRECT["expected"]["fixed"]),
                                             ("Not sent: the value", "breaks the definition.")}
    # the same fixed words for two cases, by one family or by two, could not tell an answer's case
    for family in ("fx-en", "fx-twin"):
        twin = {**changed(MISSING, fixed=list(INCORRECT["expected"]["fixed"])), "id": "sdev-a-0099",
                "family": family}
        with pytest.raises(ValueError, match="the incorrect case .* and the missing case"):
            sd.sentence_families([INCORRECT, twin])


# ---- the template, its placeholders and its slots ---------------------------------------------------------


def test_the_template_is_found_in_the_system_message_with_its_placeholders():
    assert sd.sentence_template(INCORRECT) == sd.Template(("<value>", "<param>"), None, None)
    assert sd.sentence_template(MISSING) == sd.Template(("<api>",), None, "<params>")
    assert sd.sentence_template(MISSING_ZH) == sd.Template(("{{函数}}",), None, "{{参数}}")
    assert sd.sentence_template(EDGE) == sd.Template((), "<api>", "<params>")
    plain, caps = sd.sentence_template(PLAIN), sd.sentence_template(CAPS)
    assert plain == sd.Template(("value", "field"), None, None) and plain.placeholders == ()
    assert caps == sd.Template(("PARAM", "VALUE"), None, None) and caps.placeholders == ("PARAM", "VALUE")
    assert sd.sentence_template(ANGLE) == sd.Template(("parameter", "value"), None, None)
    assert sd.sentence_template(changed(INCORRECT, fixed=["Not sent: the value", "is wrong"])) is None


@pytest.mark.parametrize(
    "item,slots,roles",
    [
        (INCORRECT, ("<value>", "<param>"), ("value", "param")),
        (MISSING, ("<api>", "<params>"), ("api", "params")),
        (EDGE, ("<api>", "<params>"), ("api", "params")),  # the leading and the trailing edge
        (CAPS, ("PARAM", "VALUE"), ("param", "value")),
        (PLAIN, ("value", "field"), ("value", "param")),
        (INCORRECT_ZH, ("【参数名】", "【取值】"), ("param", "value")),
        (MISSING_ZH, ("{{函数}}", "{{参数}}"), ("api", "params")),
        (ANGLE, ("parameter", "value"), ("param", "value")),
        (ANGLE_MISSING, ("function", "parameters"), ("api", "params")),
        (IRRELEVANT, (), ()),
        (BLIND, ("<first>", "<second>"), None),
    ],
)
def test_every_slot_of_a_template_has_a_role(item, slots, roles):
    template = sd.sentence_template(item)
    assert template.every_slot == slots
    assert sd.slot_roles(template, item["expected"]["case"]) == roles


def test_slots_that_cannot_be_told_apart_have_no_roles():
    assert sd.slot_roles(sd.Template(("<value>",), None, None), "incorrect") is None  # one slot for two
    assert sd.slot_roles(sd.Template(("<value>", "<value too>"), None, None), "incorrect") is None
    assert sd.slot_roles(sd.Template(("<function>", "<api>"), None, None), "missing") is None
    assert sd.slot_roles(sd.Template(("<value>",), None, None), "irrelevant") is None  # nothing to fill


# ---- the calls --------------------------------------------------------------------------------------------

BOOKING_JSON = '[{"name": "book_room", "arguments": {"room": "会议室A", "people": 4, "date": "2026-11-05"}}]'
BOOKING_YAML = '- name: book_room\n  arguments:\n    room: "会议室A"\n    people: 4\n    date: "2026-11-05"'


@pytest.mark.parametrize(
    "fmt,text",
    [
        ("python", EXACT_BOOKING),
        ("python", f"```python\n{EXACT_BOOKING}\n```"),
        ("python", "book_room(room='会议室A', people=4, date='2026-11-05')"),
        ("json", BOOKING_JSON),
        ("yaml", BOOKING_YAML),
        ("yaml", f"```yaml\n{BOOKING_YAML}\n```"),
    ],
)
def test_parse_answer_reads_each_notation_and_one_fence(fmt, text):
    booking = ("book_room", {"room": "会议室A", "people": 4, "date": "2026-11-05"})
    assert sd.parse_answer(text, fmt) == [booking]


@pytest.mark.parametrize(
    "fmt,text",
    [
        ("python", ""),
        ("python", "Not sent: the value -3 of people breaks the definition."),
        ("python", '[{"name": "book_room", "arguments": {"room": "A"}}]'),  # another notation
        ("json", EXACT_BOOKING),
        ("python", f"Here it is:\n{EXACT_BOOKING}"),  # text around the call
        ("yaml", "- name: book_room is missing"),
        ("python", 'book_room("会议室A", 4, "2026-11-05")'),  # positional: a loose call, never exact
        ("json", '{"name": "book_room", "arguments": {"room": "A"}}'),  # one object, not the array
        # deeper than the parsers go: no call, and no RecursionError either
        pytest.param("json", "[" * 50000 + "]" * 50000, id="deep-json"),
        pytest.param("python", "f(x=" + "(" * 300 + "1" + ")" * 301, id="deep-python"),
    ],
)
def test_parse_answer_refuses_everything_else(fmt, text):
    assert sd.parse_answer(text, fmt) is None


@pytest.mark.parametrize(
    "text,notation",
    [
        (EXACT_BOOKING, "python"),
        ('[{"name": "book_room", "arguments": {"room": "A-12", "people": -3}}]', "json"),
        ("- name: book_room\n  arguments:\n    people: -3", "yaml"),
        ("[book_room(room='A-12', people=-3)]", "python_list"),
        ('<tool_call>\n{"name": "book_room", "arguments": {"people": -3}}\n</tool_call>', "tool_call"),
        (f"Not sent: the value -3 of people breaks the definition.\n{EXACT_BOOKING}", "python"),
        ('The call: [{"name": "book_room", "arguments": {"people": -3}}] as asked.', "json"),
        ("Here:\n- name: book_room\n  arguments:\n    people: -3\nDone.", "yaml"),
        ("Sure.\n```json\n[{\"name\": \"book_room\", \"arguments\": {}}]\n```\nThat is all.", "json"),
    ],
)
def test_find_calls_finds_a_call_in_any_notation_inside_any_text(text, notation):
    found = sd.find_calls(text, "python")
    assert found is not None and found[0] == notation and found[1][0][0] == "book_room"


def loose_shapes(room: str, people: int) -> list[str]:
    """A call of book_room in the shapes only the loose pass reads (CALL_RULE)."""
    arguments = {"room": room, "people": people, "date": "2026-11-05"}
    call = json.dumps({"name": "book_room", "arguments": arguments}, ensure_ascii=False)
    as_parameters = {"name": "book_room", "parameters": arguments}
    openai = {"type": "function",
              "function": {"name": "book_room", "arguments": json.dumps(arguments, ensure_ascii=False)}}
    return [
        call,
        f"```json\n{call}\n```",
        f"Calling it now: {call}",
        json.dumps(as_parameters, ensure_ascii=False),
        json.dumps([as_parameters], ensure_ascii=False),
        json.dumps(openai, ensure_ascii=False),
        f'book_room("{room}", {people}, "2026-11-05")',
        f'book_room("{room}", people={people}, date="2026-11-05")',
        f'I would send book_room(room="{room}", people={people}) and nothing else.',
        f"- name: book_room\n  arguments:\n    room: {room}\n    people: {people}\n    date: 2026-11-05",
        f"name: book_room\narguments:\n  room: {room}\n  people: {people}",
        f"Here:\n- name: book_room\n  parameters:\n    room: {room}\n    people: {people}\nThat is the call.",
    ]


SHAPES = range(len(loose_shapes("A-12", -3)))
YAML_SHAPES = (9, 10, 11)


@pytest.mark.parametrize("shape", SHAPES)
def test_find_calls_reads_the_loose_shapes(shape):
    text = loose_shapes("A-12", -3)[shape]
    found = sd.find_calls(text, "python", EN_FUNCTIONS)
    assert found is not None and found[0] == "loose", text
    (name, arguments), = found[1]
    assert name == "book_room" and -3 in arguments.values() and "A-12" in arguments.values()


@pytest.mark.parametrize("shape", SHAPES)
def test_a_loose_call_on_an_incorrect_item_is_called_with_the_invalid_value(shape):
    s = score(loose_shapes("A-12", -3)[shape], INCORRECT)
    assert (s["call_notation"], s["failure"], s["called_with_invalid"]) == ("loose", "called", True)
    assert s["strict"] is False
    corrected = score(loose_shapes("A-12", 3)[shape], INCORRECT)
    assert (corrected["failure"], corrected["called_with_invalid"]) == ("called", False)


@pytest.mark.parametrize("shape", SHAPES)
def test_a_loose_call_on_a_valid_item_tells_a_zh_alteration_and_is_no_refusal(shape):
    altered = score(loose_shapes("Meeting room A", 4)[shape], VERBATIM)
    got = (altered["call_notation"], altered["zh_altered"], altered["false_refusal"], altered["failure"])
    assert got == ("loose", True, False, "other_notation")
    kept = score(loose_shapes("会议室A", 4)[shape], VERBATIM)
    assert (kept["zh_altered"], kept["false_refusal"], kept["exact_call"]) == (False, False, False)


@pytest.mark.parametrize("shape", YAML_SHAPES)
def test_without_pyyaml_the_yaml_shapes_are_read_by_the_subset_after_quoting(shape, monkeypatch):
    monkeypatch.setattr(sd, "_yaml_load", lambda: None)
    s = score(loose_shapes("A-12", -3)[shape], INCORRECT)
    assert (s["call_notation"], s["called_with_invalid"]) == ("loose", True)
    assert score(loose_shapes("Meeting room A", 4)[shape], VERBATIM)["zh_altered"] is True


def test_a_loose_python_call_closes_past_the_brackets_and_quotes_of_its_strings():
    text = 'Then: book_room(room="A-12 (east)", note="a \\"quoted\\" word)", people=-3) as asked.'
    assert sd.find_calls(text, "json", EN_FUNCTIONS) == (
        "loose", [("book_room", {"room": "A-12 (east)", "note": 'a "quoted" word)', "people": -3})])
    assert sd.find_calls("book_room(" * 3000, "json", EN_FUNCTIONS) is None  # never closed
    assert sd.find_calls("book_room(" + "[" * 100 + "1" + "]" * 100 + ")", "json", EN_FUNCTIONS) is None


def test_a_loose_yaml_date_stays_text():
    found = sd.find_calls("- name: book_room\n  arguments:\n    date: 2026-11-05\n    room: A-12", "json")
    assert found == ("loose", [("book_room", {"date": "2026-11-05", "room": "A-12"})])


def test_a_python_list_of_calls_is_read_from_the_text_of_each_call():
    # each call of a list [f(k=v), ...] is read from its own text, never written back: a call the list's
    # brackets continue over two lines, text in another script before a call (the parser's positions count the
    # bytes of UTF-8 lines), an integer too long to write in decimal
    assert sd.find_calls("[book_room\n (room='A-12', people=-3)]", "json") == (
        "python_list", [("book_room", {"room": "A-12", "people": -3})])
    two = "[book_room(room='会议室A', people=4),\n plan_trip(destination='里斯本')]"
    assert sd.find_calls(two, "json") == ("python_list", [("book_room", {"room": "会议室A", "people": 4}),
                                                          ("plan_trip", {"destination": "里斯本"})])
    huge = sd.find_calls(f"[book_room(people={HUGE})]", "json")
    assert huge == ("python_list", [("book_room", {"people": int(HUGE, 16)})])


@pytest.mark.parametrize(
    "text",
    [
        "Not sent: the value <-3> of <people> breaks the definition.",
        "Not sent: book_room(people) still needs date",  # a name, not a value, in the brackets
        "I cannot call book_room() without a date.",  # no argument
        '["Not sent: no listed function can do this."]',
        "无法执行：参数【count】收到的值【十几】不符合定义。",
        "The value [2023-13-45] (a date) is not valid.",
        "Unacceptable value (-3) supplied for (people).",
        "Still missing for (plan_trip): (traveller).",
        "",
        '{"name": "Alice", "age": 3}',  # no arguments
        json.dumps(BOOK_ROOM),  # a function definition: its parameters are a schema
        "name: Alice\nage: 3",
        "- name: book_room\n- name: plan_trip",
        'The key "name": is missing here.',
        "other_function(-3)",  # a function the item does not show
    ],
)
def test_find_calls_finds_no_call_in_a_sentence_with_brackets(text):
    assert sd.find_calls(text, "python", EN_FUNCTIONS) is None


def test_the_item_functions_come_from_the_system_message_and_the_expectation():
    assert sd.item_functions(INCORRECT) == {"book_room", "plan_trip"}
    assert sd.item_functions(MISSING_ZH) == {"query_stock"}
    markdown = copy.deepcopy(VERBATIM)
    markdown["messages"][0]["content"] = ("Functions:\n\n### Get Teams\nList teams.\n\n"
                                          "### find_hotel\nFind one.")
    assert sd.item_functions(markdown) == {"Get Teams", "find_hotel", "book_room"}  # book_room: expected
    literal = copy.deepcopy(INCORRECT)
    literal["messages"][0]["content"] = "Functions:\n[{'name': 'lookup_city', 'description': 'x'}]"
    assert sd.item_functions(literal) == {"lookup_city"}


# ---- strict and lenient on the same answers ---------------------------------------------------------------

LONG = ("The request asks for -3 people.\nNot sent: the value -3 of people breaks the definition.\n"
        "A booking needs at least one person.")
TRIP_LONG = ("The trip needs a traveller.\n\nNot sent: plan_trip still needs traveller\n\n"
             "Please share the traveller's name and passport.")

# (item, answer, strict, lenient, failure)
REFUSALS = [
    (INCORRECT, "Not sent: the value <-3> of <people> breaks the definition.", True, True, None),
    (INCORRECT, "Not sent: the value -3 of people breaks the definition.", True, True, None),
    (INCORRECT, "  Not sent: the value -3 of people breaks the definition  \n", True, True, None),
    (INCORRECT, "Not sent: the value -3 of people breaks the definition!", True, True, None),
    (INCORRECT, "Not sent: the value -3 of people breaks the definition..", False, True, "extra_text"),
    (INCORRECT, LONG, False, True, "extra_text"),
    (INCORRECT, "Not sent: the value <value> of <param> breaks the definition.",
     False, False, "placeholder_left"),
    (INCORRECT, "Not sent: the value <-3> of <param> breaks the definition.",
     False, True, "placeholder_left"),
    (INCORRECT, 'book_room(room="A-12", people=-3, date="2026-11-05")', False, False, "called"),
    (INCORRECT, ("Not sent: the value -3 of people breaks the definition.\n"
                 'book_room(room="A-12", people=-3, date="2026-11-05")'), False, True, "called"),
    (INCORRECT, ('<tool_call>\n{"name": "book_room", "arguments": {"room": "A-12", "people": -3}}\n'
                 "</tool_call>"), False, False, "called"),
    (INCORRECT, "[book_room(room='A-12', people=-3, date='2026-11-05')]", False, False, "called"),
    (INCORRECT, "Not sent: book_room still needs people", False, False, "wrong_case"),
    (INCORRECT, "Sorry, a room cannot be booked for -3 people.", False, False, "no_sentence"),
    # ACEBench's rule as transposed reads the fixed words in order
    (INCORRECT, "-3 breaks the definition. Not sent: the value is not one of the allowed ones.",
     False, False, "no_sentence"),
    (INCORRECT, "Not sent: the value minus three of people breaks the definition.",
     False, False, "wrong_values"),
    (INCORRECT, "Not sent: the value -3 of headcount breaks the definition.", False, True, "wrong_params"),
    (INCORRECT, "Not sent: the value -3 of people_count breaks the definition.", False, True, "wrong_params"),
    (INCORRECT, "Not sent: the value -3 of room.people breaks the definition.", False, True, "wrong_params"),
    (INCORRECT, "Not sent: the value people of -3 breaks the definition.", False, True, "wrong_values"),
    (INCORRECT, "not sent: the value -3 of people breaks the definition.", False, False, "no_sentence"),
    (INCORRECT, "NOT SENT: THE VALUE -3 OF PEOPLE BREAKS THE DEFINITION.", False, False, "no_sentence"),
    (INCORRECT, "", False, False, "empty"),
    (INCORRECT, None, False, False, "no_answer"),
    (MISSING, "Not sent: <plan_trip> still needs <traveller>", True, True, None),
    (MISSING, "Not sent: plan_trip still needs traveller.", True, True, None),
    (MISSING, TRIP_LONG, False, True, "extra_text"),
    (MISSING, "Not sent: plan_trip still needs traveller.passport", False, True, "wrong_params"),
    (MISSING, "Not sent: plan_trip still needs traveller, destination", False, True, "wrong_params"),
    (MISSING, "Not sent: someone still needs traveller", False, False, "wrong_params"),
    (MISSING, "Not sent: plan_trip still needs <params>", False, False, "placeholder_left"),
    (IRRELEVANT, "Not sent: no listed function can do this.", True, True, None),
    (IRRELEVANT, "Not sent: no listed function can do this", True, True, None),
    (IRRELEVANT, "Not sent: no listed function can do this. A weather service can answer it.",
     False, True, "extra_text"),
    (IRRELEVANT, "I cannot check the weather with these functions.", False, False, "no_sentence"),
    (INCORRECT_ZH, "无法执行：参数【count】收到的值【十几】不符合定义。", True, True, None),
    (INCORRECT_ZH, "无法执行：参数count收到的值十几不符合定义。", True, True, None),
    (INCORRECT_ZH, "无法执行：参数【参数名】收到的值【十几】不符合定义。", False, True, "placeholder_left"),
    (INCORRECT_ZH, '[{"name": "query_stock", "arguments": {"item": "苹果手机", "count": 15}}]',
     False, False, "called"),
    (INCORRECT_ZH, "无法执行：query_stock缺少count", False, False, "wrong_case"),
    (INCORRECT_ZH, "无法执行：参数【count】收到的值【10到19】不符合定义。", False, False, "wrong_values"),
    (MISSING_ZH, "无法执行：{{query_stock}}缺少{{item}}", True, True, None),
    (MISSING_ZH, "无法执行：{{query_stock}}缺少{{item}}。", True, True, None),
    (MISSING_ZH, "无法执行：query_stock缺少item和count", False, True, "wrong_params"),
    (MISSING_ZH, "无法执行：{{函数}}缺少{{参数}}", False, False, "placeholder_left"),
    (IRRELEVANT_ZH, "无法执行：现有函数都不适用。", True, True, None),
    (PRIORITY, "Not sent: the value -3 of people breaks the definition.", True, True, None),
    (PRIORITY, "Not sent: book_room still needs room", False, False, "wrong_case"),
    (PLAIN, "Rejected: -3 for people is not allowed.", True, True, None),
    (PLAIN, "Rejected: value for field is not allowed.", False, False, "placeholder_left"),
    (PLAIN, "Rejected: -3 for field is not allowed.", False, True, "placeholder_left"),
    (CAPS, "Refused: people cannot be 2.5.", True, True, None),
    (CAPS, "Refused: people cannot be VALUE.", False, False, "placeholder_left"),
    (CAPS, "Refused: PARAM cannot be 2.5.", False, True, "placeholder_left"),
    (EDGE, "plan_trip is short of traveller", True, True, None),
    (EDGE, "<plan_trip> is short of <traveller>.", True, True, None),
    (EDGE, "Sorry. plan_trip is short of traveller", False, True, "extra_text"),
    (EDGE, "plan_trip is short of traveller and passport", False, True, "wrong_params"),
    (EDGE, "<api> is short of traveller", False, False, "placeholder_left"),
    (ANGLE, "Not accepted: <people> cannot hold <-3>.", True, True, None),
    (ANGLE, "Not accepted: <people> cannot hold <-3>", True, True, None),
    # the sentence's words with other brackets or quotes, and more prose: ACEBench's phrase, not ours
    (ANGLE, 'Not accepted: "people" cannot hold "-3". The minimum is 1.', False, True, "no_sentence"),
    (ANGLE, "Not accepted: (people) cannot hold (-3).", False, True, "no_sentence"),
    (ANGLE, "Not  accepted:\n<people>   cannot hold <-3>.", False, True, "no_sentence"),
    (ANGLE, "cannot hold <-3>. Not accepted: <people>", False, False, "no_sentence"),
    (ANGLE, "Not accepted: <people> cannot hold <minus three>.", False, False, "wrong_values"),
    (ANGLE_TWO, "Not accepted: <people, date> cannot hold <-3, 2026-13-45>.", True, True, None),
    (ANGLE_TWO, "Not accepted: <date, people> cannot hold <2026-13-45, -3>.", True, True, None),
    (ANGLE_TWO, "Not accepted: <people, date> cannot hold <2026-13-45, -3>.", False, True, "wrong_params"),
    (ANGLE_TWO, "Not accepted: <people; date> cannot hold <-3; 2026-13-45>.", False, True, "wrong_values"),
    (ANGLE_TWO, "Not accepted: <people> cannot hold <-3>.", False, False, "wrong_values"),
    (ANGLE_MISSING, "Not accepted: <plan_trip> lacks <traveller, destination>.", True, True, None),
    (ANGLE_MISSING, "Not accepted: <plan_trip> lacks <destination, traveller>.", True, True, None),
    (ANGLE_MISSING, "Not accepted: <plan_trip> lacks <traveller>.", False, False, "wrong_params"),
    (ANGLE_MISSING, "Not accepted: <plan_trip> lacks <traveller, destination, passport>.",
     False, True, "wrong_params"),
    (ANGLE_MISSING, "Not accepted: <plan_trip> lacks <traveller and destination>.",
     False, True, "wrong_params"),
    (ANGLE_MISSING, "Not accepted: <traveller, destination> lacks <plan_trip>.", False, True, "wrong_params"),
    # another case's sentence written without its final mark stands all the same: the end is read as its own
    (INCORRECT, "Not sent: no listed function can do this", False, False, "wrong_case"),
    (ANGLE, "Not accepted: <plan_trip> lacks <traveller>", False, False, "wrong_case"),
    # one sentence mark may end the answer, no other character
    (MISSING, "Not sent: plan_trip still needs traveller;", False, True, "wrong_params"),
    (MISSING, "Not sent: plan_trip still needs travellers", False, True, "wrong_params"),
    # a dot between two digits is no full stop of the sentence: the value slot holds 2.5.1
    (CAPS, "Refused: people cannot be 2.5.1.", False, True, "wrong_values"),
    # marks and symbols outside the sentence are text outside it
    (INCORRECT, "Not sent: the value -3 of people breaks the definition. :)", False, True, "extra_text"),
    (EDGE, "!! plan_trip is short of traveller", False, True, "extra_text"),
]


@pytest.mark.parametrize("item,answer,strict,lenient,failure", REFUSALS)
def test_strict_and_lenient_and_the_failure_mode(item, answer, strict, lenient, failure):
    s = score(answer, item)
    assert (s["strict"], s["lenient"], s["failure"]) == (strict, lenient, failure)
    assert s["exact_call"] is None and s["zh_altered"] is None and s["case"] == item["expected"]["case"]
    assert s["false_refusal"] is None and s["status"] == "ok"


def test_strict_implies_lenient_on_every_answer():
    for item, answer, *_ in REFUSALS:
        s = score(answer, item)
        assert not s["strict"] or s["lenient"], (item["id"], answer)


def test_the_extra_text_flag_agrees_with_the_failure_and_with_strict():
    # the failure extra_text says the sentence is filled right but not alone, and the flag that something
    # stands outside it: either both or neither, whatever stands outside (words, marks, symbols)
    rows = [(item, answer) for item, answer, *_ in REFUSALS] + [(item, answer) for item, answer, _ in PROBES]
    for item, answer in rows:
        s = score(answer, item)
        if s["failure"] == "extra_text":
            assert s["extra_text"] is True, (item["id"], answer)
        if s["strict"]:
            assert s["extra_text"] is False, (item["id"], answer)


def test_another_cases_sentence_without_its_final_mark_counts_for_k2():
    # fx-angle's missing sentence ends with ">."; written without its full stop on an incorrect item, it
    # stands in place of the item's own, which is what scripts/calibrate_special_dev.py's K2 counts
    s = score("Not accepted: <plan_trip> lacks <traveller>", ANGLE)
    assert (s["failure"], s["sentences"], s["sentence"]) == ("wrong_case", ["missing"], False)
    s = score("Not sent: no listed function can do this", INCORRECT)
    assert (s["failure"], s["sentences"], s["sentence"]) == ("wrong_case", ["irrelevant"], False)


@pytest.mark.parametrize(
    "answer,strict,extra",
    [
        ("Not sent: no listed function can do this", True, False),
        ("Not sent: no listed function can do this.", True, False),
        ("Not sent: no listed function can do this!", True, False),
        ("Not sent: no listed function can do this..", False, True),
        ("Not sent: no listed function can do this. Sorry", False, True),
    ],
)
def test_one_mark_after_a_sentence_that_ends_without_one_is_no_extra_text(answer, strict, extra):
    # a sentence whose last fixed word has no final mark and no slot after it: one sentence mark may end the
    # answer (STRICT_RULE), and it is no text outside the sentence; a second one, or a word, is
    item = changed(IRRELEVANT, fixed=["Not sent: no listed function can do this"])
    assert sd.check_item(item) == []
    s = sd.score_item(answer, item)
    assert (s["strict"], s["extra_text"], s["failure"]) == (strict, extra, None if strict else "extra_text")


# The answers the round-1 review found strict although they are not the sentence alone (2026-11-02).
PROBES = [
    (INCORRECT, "Not sent: the value <3> of <people> breaks the definition. (-3 was given)", "wrong_values"),
    (INCORRECT, "Not sent: the value <-3> of <room> breaks the definition. people", "wrong_params"),
    (INCORRECT, ("Not sent: the value -3 of people breaks the definition.\n"
                 "Not sent: book_room still needs date"), "extra_text"),
    (MISSING, "Not sent: plan_trip still needs traveller. Also traveller.passport", "extra_text"),
    (MISSING, "Not sent: plan_trip still needs traveller\ndestination too", "extra_text"),
]


@pytest.mark.parametrize("item,answer,failure", PROBES)
def test_the_review_probes_are_not_strict(item, answer, failure):
    s = score(answer, item)
    assert (s["strict"], s["failure"]) == (False, failure)


def test_text_outside_the_sentence_is_counted_and_is_not_strict():
    for answer, item in ((LONG, INCORRECT), (TRIP_LONG, MISSING),
                         ("Sorry. plan_trip is short of traveller", EDGE)):
        s = score(answer, item)
        assert (s["extra_text"], s["strict"]) == (True, False), answer
    assert score("Not sent: plan_trip still needs traveller.", MISSING)["extra_text"] is False
    assert score("Not sent: the value -3 of people breaks the definition.", INCORRECT)["extra_text"] is False
    # the full stop of the sentence is the last one, not the one inside the value 2.5
    assert score("Refused: people cannot be 2.5.", CAPS)["extra_text"] is False


def test_the_sentence_placement_prefers_whole_words_free_of_the_fillers_on_one_line_and_the_shortest():
    words = FAMILIES["fx-plain"]["incorrect"][1]
    text = "Rejected: value format for people is not allowed."
    assert sd.sentence_spans(text, words)[1] == (text.index(" for ") + 1, text.index(" for ") + 4)
    words = FAMILIES["fx-caps"]["incorrect"][1]
    text = "Refused: people cannot be 2.5."
    filler = (text.index("2.5"), text.index("2.5") + 3)
    assert sd.sentence_spans(text, words, [filler])[-1] == (len(text) - 1, len(text))
    # no filler given: the dot of 2.5 is part of a word all the same (NAME_RULE), so the last one is taken
    assert sd.sentence_spans(text, words)[-1] == (len(text) - 1, len(text))
    words = FAMILIES["fx-en"]["incorrect"][1]
    text = "Not sent: the value is odd.\nNot sent: the value -3 of people breaks the definition."
    assert sd.sentence_spans(text, words)[0][0] == text.index("\n") + 1
    assert sd.sentence_spans("A x B, then A B", ["A", "B"]) == [(12, 13), (14, 15)]
    assert sd.sentence_spans("B before A", ["A", "B"]) is None


def test_a_dot_between_two_word_characters_is_no_full_stop_of_the_sentence():
    # the full stop that ends a sentence is not placed inside a dotted name or a number (NAME_RULE: address in
    # address.city is not named), nor a fixed word right after one
    words = sm.fixed_parts(mix_family("missing"), "caps", "en")  # ["Missing for", ":", "."]
    for text in ("Missing for book_room: date.name.", "Missing for book_room: date.name.first.",
                 "Missing for book_room: 2.5."):
        assert sd.sentence_spans(text, words)[-1] == (len(text) - 1, len(text)), text
    assert sd.sentence_spans("x.name stands", ["name"]) == [(2, 6)]  # no other place: taken all the same


def test_the_lenient_words_drop_the_brackets_of_the_placeholders_around_them_and_the_final_mark():
    # ACEBench's phrases hold neither the brackets of a placeholder nor the sentence's final mark
    assert sd.lenient_words(FAMILIES["fx-angle"]["incorrect"][1]) == ["Not accepted:", "cannot hold"]
    assert sd.lenient_words(["「", "」 不是 「", "」 的有效取值。"]) == ["不是", "的有效取值"]
    assert sd.lenient_words(["Missing values {", "} prevent  calling {", "}:"]) == [
        "Missing values", "prevent calling", ":"]
    assert sd.lenient_words(["(No function)", "fits."]) == ["(No function)", "fits"]  # not at a slot's edge
    assert sd.lenient_words(["Refused:", "cannot be", "."]) == ["Refused:", "cannot be"]


def test_a_call_on_an_incorrect_item_says_whether_it_carries_the_invalid_value():
    carried = score('book_room(room="A-12", people=-3, date="2026-11-05")', INCORRECT)
    assert carried["called_with_invalid"] is True
    s = score('book_room(room="A-12", people=1, date="2026-11-05")', INCORRECT)
    assert s["called_with_invalid"] is False and s["failure"] == "called" and s["call_notation"] == "python"
    assert score('[{"name": "query_stock", "arguments": {"item": "苹果手机", "count": 15}}]',
                 INCORRECT_ZH)["called_with_invalid"] is False
    refused = score("Not sent: the value -3 of people breaks the definition.", INCORRECT)
    assert refused["called_with_invalid"] is None
    # an argument value that holds the invalid value keeps it (FAILURE_RULE: inside an argument value)
    assert score('book_room(room="A-12", people=-30, date="2026-11-05")', INCORRECT)["called_with_invalid"]


def test_wrong_case_needs_the_other_sentence_of_the_set():
    answer = "Not sent: book_room still needs people"
    assert score(answer, INCORRECT)["sentences"] == ["missing"]
    alone = sd.score_item(answer, INCORRECT)  # the item's own sentence is the only one known
    assert alone["failure"] == "no_sentence" and alone["sentences"] == []


# ---- one family in several placeholder styles (the data package's sets) ----------------------------------

STYLE_WORDS = {"angle": {"value": "value", "param": "parameter", "params": "parameters", "api": "function"},
               "caps": {"value": "VALUE_GIVEN", "param": "PARAM_NAME", "params": "MISSING_PARAMS",
                        "api": "FUNCTION_NAME"}}


def mix_family(case: str) -> sm.Family:
    """The first English development family of a case (special_mix, read only)."""
    return next(family for fid, family in sorted(sm.FAMILIES.items()) if fid.startswith(f"{case}-en-b"))


def mixed_item(number: int, kind: str, style: str, expected: dict, user: str) -> dict:
    """An item whose prompt writes special_mix's families in the style, as dress writes a prompt."""
    sentences = {case: sm.fill_sentence(mix_family(case), style, "en", **STYLE_WORDS[style])
                 for case in sd.CASES}
    family = mix_family(expected["case"]).id if expected["type"] == "refusal" else "mixed"
    return {"id": f"sdev-b-{number:04d}", "split": "b", "lang": "en", "kind": kind,
            "messages": [{"role": "system", "content": system_text(sentences, "en", "python")},
                         {"role": "user", "content": user}],
            "call_format": "python", "family": family, "expected": expected,
            "source": {"dataset": "fixture", "record": f"m{number}"}}


def mixed_refusal(case: str, style: str, **names) -> dict:
    return {"type": "refusal", "case": case, "fixed": sm.fixed_parts(mix_family(case), style, "en"),
            "values": names.get("values", []), "params": names.get("params", []), "api": names.get("api", "")}


MIXED = [
    mixed_item(1, "incorrect", "angle",
               mixed_refusal("incorrect", "angle", values=["-3"], params=["people"]),
               "Book room A-12 for -3 people on 2026-11-05."),
    mixed_item(2, "incorrect", "caps", mixed_refusal("incorrect", "caps", values=["0"], params=["people"]),
               "Book room A-12 for 0 people on 2026-11-05."),
    mixed_item(3, "missing", "angle",
               mixed_refusal("missing", "angle", params=["traveller"], api="plan_trip"),
               "Plan a trip to Lisbon."),
    mixed_item(4, "missing", "caps", mixed_refusal("missing", "caps", params=["date"], api="book_room"),
               "Book room A-12 for 2 people."),
    mixed_item(5, "irrelevant", "angle", mixed_refusal("irrelevant", "angle"), "Is it raining in Oslo?"),
    mixed_item(6, "valid_verbatim", "caps",
               calls(("book_room", {"room": "会议室A", "people": 4, "date": "2026-11-05"}),
                     verbatim=("会议室A",)),
               "Book room 会议室A for 4 people on 2026-11-05."),
]


def fill(case: str, style: str, **values: str) -> str:
    return sm.fill_sentence(mix_family(case), style, "en", **values)


def test_a_set_may_write_one_family_in_several_styles():
    assert all(sd.check_item(item) == [] for item in MIXED)
    found = sd.sentence_families(MIXED)
    incorrect, missing = mix_family("incorrect").id, mix_family("missing").id
    assert found[(incorrect, "incorrect")] == {tuple(sm.fixed_parts(mix_family("incorrect"), style, "en"))
                                               for style in ("angle", "caps")}
    assert len(found[(missing, "missing")]) == 2
    answers = [fill("incorrect", "angle", value="-3", param="people"),
               fill("incorrect", "caps", value="0", param="people"),
               fill("missing", "angle", api="plan_trip", params="traveller"),
               fill("missing", "caps", api="book_room", params="date"),
               fill("irrelevant", "angle"),
               EXACT_BOOKING]
    scores = sd.score_items(answers, MIXED)
    assert [(s["strict"], s["lenient"], s["failure"]) for s in scores[:5]] == [(True, True, None)] * 5
    assert (scores[5]["exact_call"], scores[5]["false_refusal"]) == (True, False)


def test_wrong_case_and_false_refusals_read_every_style_of_a_family():
    families = sd.sentence_families(MIXED)
    other_style = fill("missing", "caps", api="book_room", params="people")
    s = sd.score_item(other_style, MIXED[0], families=families)  # an angle item, the caps missing sentence
    assert (s["sentences"], s["failure"]) == (["missing"], "wrong_case")
    angle_only = sd.sentence_families([item for item in MIXED if item["id"] != MIXED[3]["id"]])
    assert sd.score_item(other_style, MIXED[0], families=angle_only)["failure"] == "no_sentence"
    caps_sentence = fill("incorrect", "caps", value="-3", param="people")
    same_case = sd.score_item(caps_sentence, MIXED[0], families=families)
    assert (same_case["strict"], same_case["lenient"], same_case["failure"]) == (False, True, "no_sentence")
    angle_sentence = fill("incorrect", "angle", value="4", param="people")
    refused = sd.score_item(angle_sentence, MIXED[5], families=families)
    got = (refused["false_refusal"], refused["refusal_form"], refused["failure"])
    assert got == (True, "sentence", "refused")


def test_a_nested_name_or_a_longer_value_before_the_full_stop_misfills_its_slot():
    # FT3 names nested fields (docs/m8-计划.md, 诊断): in the caps style the full stop right after the slot is
    # not read inside date.name, so the slot holds date.name, a wrong parameter, not date and extra text
    families = sd.sentence_families(MIXED)
    nested = sd.score_item(fill("missing", "caps", api="book_room", params="date.name"), MIXED[3],
                           families=families)
    assert (nested["strict"], nested["lenient"], nested["failure"]) == (False, True, "wrong_params")
    longer = sd.score_item(fill("incorrect", "caps", value="0.5", param="people"), MIXED[1],
                           families=families)
    assert (longer["strict"], longer["lenient"], longer["failure"]) == (False, True, "wrong_values")


# ---- the valid items --------------------------------------------------------------------------------------

ALTERED_BOOKING = 'book_room(room="Meeting room A", people=4, date="2026-11-05")'
ALTERED_TOOL_CALL = ('<tool_call>\n{"name": "book_room", "arguments": {"room": "Meeting room A", '
                     '"people": 4, "date": "2026-11-05"}}\n</tool_call>')
ESCAPED_STOCK = ('[{"name": "query_stock", "arguments": {"item": "\\u82f9\\u679c\\u624b\\u673a", '
                 '"count": 10}}]')

# (item, answer, exact, zh_altered, false_refusal, failure)
CALLS = [
    (VERBATIM, EXACT_BOOKING, True, False, False, None),
    (VERBATIM, f"```python\n{EXACT_BOOKING}\n```", True, False, False, None),
    (VERBATIM, "book_room(room='会议室A', people=4, date='2026-11-05')", True, False, False, None),
    (VERBATIM, ALTERED_BOOKING, False, True, False, "zh_altered"),
    (VERBATIM, 'book_room(room="会议室A", people=5, date="2026-11-05")', False, False, False, "wrong_call"),
    (VERBATIM, BOOKING_JSON, False, False, False, "other_notation"),
    (VERBATIM, f"Here is the call:\n{EXACT_BOOKING}", False, False, False, "with_text"),
    (VERBATIM, f"Not sent: the value <4> of <people> breaks the definition.\n{EXACT_BOOKING}",
     False, False, False, "with_text"),
    (VERBATIM, "Not sent: the value <4> of <people> breaks the definition.", False, None, True, "refused"),
    (VERBATIM, "Which floor should the room be on?", False, None, True, "no_call"),
    (VERBATIM, ALTERED_TOOL_CALL, False, True, False, "other_notation"),
    (VERBATIM, "", False, None, True, "empty"),
    (VERBATIM, None, False, None, False, "no_answer"),
    (VERBATIM_ZH, EXACT_STOCK, True, False, False, None),
    (VERBATIM_ZH, ESCAPED_STOCK, True, False, False, None),
    (VERBATIM_ZH, '[{"name": "query_stock", "arguments": {"item": "苹果手机", "count": 10.0}}]',
     True, False, False, None),
    (VERBATIM_ZH, '[{"name": "query_stock", "arguments": {"item": "iPhone", "count": 10}}]',
     False, True, False, "zh_altered"),
    (VERBATIM_ZH, '{"name": "query_stock", "arguments": {"item": "iPhone", "count": 10}}',
     False, True, False, "other_notation"),
    (VERBATIM_ZH, "无法执行：参数【count】收到的值【10】不符合定义。", False, None, True, "refused"),
    # a verbatim value inside a longer argument value is kept (FAILURE_RULE): no alteration, another call
    (VERBATIM_ZH, '[{"name": "query_stock", "arguments": {"item": "苹果手机Pro", "count": 10}}]',
     False, False, False, "wrong_call"),
    # calls that cannot be compared (ace_probe.same_calls raises) are not the expected ones
    pytest.param(VERBATIM, f'book_room(room="会议室A", people={HUGE}, date="2026-11-05")',
                 False, False, False, "wrong_call", id="an-integer-too-long-to-write"),
    pytest.param(VERBATIM_ZH, ('[{"name": "query_stock", "arguments": {"item": "苹果手机", "count": '
                               + "[" * 1200 + "]" * 1200 + "}}]"), False, False, False, "wrong_call",
                 id="a-value-nested-too-deep-to-compare"),
    (NEARMISS, NEARMISS_YAML, True, None, False, None),
    (NEARMISS, f"```yaml\n{NEARMISS_YAML}\n```", True, None, False, None),
    (NEARMISS, 'book_room(room="A-12", people=1, date="2024-02-29")', False, None, False, "other_notation"),
    (MULTITURN, 'book_room(room="A-12", people=6, date="2026-11-05")', True, None, False, None),
    (MULTITURN, 'book_room(room="A-12", people=4, date="2026-11-05")', False, None, False, "wrong_call"),
]


@pytest.mark.parametrize("item,answer,exact,altered,refused,failure", CALLS)
def test_the_calls_of_a_valid_item(item, answer, exact, altered, refused, failure):
    s = score(answer, item)
    got = (s["exact_call"], s["zh_altered"], s["false_refusal"], s["failure"])
    assert got == (exact, altered, refused, failure)
    assert s["strict"] is None and s["lenient"] is None and s["case"] is None


TRAINING_SENTENCE = sm.fill_sentence(next(f for fid, f in sorted(sm.FAMILIES.items())
                                          if fid.startswith("incorrect-en-t")), "angle", "en",
                                     value="4", param="people")


@pytest.mark.parametrize(
    "answer,refused,form",
    [
        ("I cannot book this room because the value is invalid.", True, "other_text"),
        ("Not sent: the value <4> of <people> breaks the definition.", True, "sentence"),  # the set's
        (TRAINING_SENTENCE, True, "sentence"),  # a training family's sentence, which no dev set holds
        ("", True, "empty"),
        ("  \n", True, "empty"),
        (None, False, None),  # the run recorded no answer: no refusal of the model's
        (EXACT_BOOKING, False, None),
        ('book_room("会议室A", 4, "2026-11-05")', False, None),
    ],
)
def test_a_false_refusal_is_a_valid_answer_without_a_call_whatever_its_wording(answer, refused, form):
    s = score(answer, VERBATIM)
    assert (s["false_refusal"], s["refusal_form"]) == (refused, form)


def test_a_false_refusal_is_a_boolean_even_without_the_sets_sentences():
    alone = sd.score_item("Not sent: the value <4> of <people> breaks the definition.", VERBATIM)
    assert (alone["false_refusal"], alone["refusal_form"]) == (True, "other_text")  # no family known: text
    assert sd.score_item(EXACT_BOOKING, VERBATIM)["false_refusal"] is False
    assert sd.score_item(None, VERBATIM)["false_refusal"] is False
    assert sd.score_item("Not sent: plan_trip still needs traveller", MISSING)["false_refusal"] is None


def test_the_training_sentences_cover_every_family_in_every_style():
    sentences = sd.training_sentences()
    assert len({words for _, words in sentences}) == len(sentences)
    for family in sm.FAMILIES.values():
        for style in sm.STYLES:
            assert (family.case, tuple(sm.fixed_parts(family, style, family.lang))) in sentences


def test_the_score_names_the_item_and_counts_the_verbatim_values():
    s = score(EXACT_BOOKING, VERBATIM)
    assert (s["id"], s["kind"], s["lang"], s["family"]) == ("sdev-a-0008", "valid_verbatim", "en", "fx-en")
    assert s["verbatim"] == 1 and s["chars"] == len(EXACT_BOOKING) and s["call_notation"] == "python"
    assert score(None, VERBATIM)["chars"] == 0


# ---- a greedy loop cut by the answer budget ---------------------------------------------------------------

INCORRECT_YAML = '- name: book_room\n  arguments:\n    room: "A-12"\n    people: -3\n    date: "2026-11-05"'
INCORRECT_MAPPING = "name: book_room\narguments:\n  room: A-12\n  people: -3\n  date: 2026-11-05"
ALTERED_YAML = ('- name: book_room\n  arguments:\n    room: "Meeting room A"\n    people: 4\n'
                '    date: "2026-11-05"')


def loop_cuts(unit: str, times: int = 3) -> list[str]:
    """A greedy loop of `times` repetitions of unit, cut by the answer budget at every character of its last
    repetition (which is never whole)."""
    loop = (unit + "\n") * times
    start = len(loop) - len(unit) - 1
    return [loop[:cut] for cut in range(start + 1, len(loop) - 1)]


def test_a_yaml_loop_cut_by_the_budget_still_holds_its_calls_on_a_valid_item():
    # the repetitions before the cut are calls (the review of 2026-11-02 found such answers read as no call,
    # so as false refusals): the YAML list is read item by item when it does not read whole
    for answer in loop_cuts(NEARMISS_YAML):
        s = score(answer, NEARMISS)
        assert (s["false_refusal"], s["call_notation"]) == (False, "yaml"), answer[-40:]
    for answer in loop_cuts(BOOKING_YAML):
        s = score(answer, VERBATIM)
        got = (s["false_refusal"], s["zh_altered"], s["call_notation"])
        assert got == (False, False, "yaml"), answer[-40:]
    for answer in loop_cuts(ALTERED_YAML):
        assert score(answer, VERBATIM)["zh_altered"] is True, answer[-40:]


@pytest.mark.parametrize("unit,notation", [(INCORRECT_YAML, "yaml"), (INCORRECT_MAPPING, "loose")])
def test_a_yaml_loop_cut_by_the_budget_is_called_with_the_invalid_value(unit, notation):
    for answer in loop_cuts(unit):
        s = score(answer, INCORRECT)
        got = (s["failure"], s["call_notation"], s["called_with_invalid"])
        assert got == ("called", notation, True), answer[-40:]


def test_without_pyyaml_a_mapping_loop_cut_by_the_budget_is_read_item_by_item(monkeypatch):
    monkeypatch.setattr(sd, "_yaml_load", lambda: None)
    for answer in loop_cuts(INCORRECT_MAPPING):
        s = score(answer, INCORRECT)
        assert (s["call_notation"], s["called_with_invalid"]) == ("loose", True), answer[-40:]


def test_a_yaml_list_whose_items_all_read_is_read_whole():
    # a whole block is read as before: every item's call, in order
    two = BOOKING_YAML + "\n" + NEARMISS_YAML
    assert sd.find_calls(f"Here:\n{two}", "python", EN_FUNCTIONS) == ("yaml", [
        ("book_room", {"room": "会议室A", "people": 4, "date": "2026-11-05"}),
        ("book_room", {"room": "A-12", "people": 1, "date": "2024-02-29"})])


# ---- answers the parsers cannot read ----------------------------------------------------------------------

# (answer, a call of book_room is found in it): the parsers raise on each (the Python parser's MemoryError,
# a RecursionError, the ValueError of an integer too long to write), which must not reach the scorer's caller
UNREADABLE = [
    pytest.param("[1, " * 200, False, id="python-parser-stack-list"),
    pytest.param('["a", ' * 200, False, id="python-parser-stack-strings"),
    pytest.param("(" * 200 + "1" + ")" * 200 + " done.", False, id="python-parser-stack-parentheses"),
    pytest.param("[" * 200 + "1" + "]" * 200 + " done.", False, id="python-parser-stack-brackets"),
    pytest.param("{" * 200 + "1" + "}" * 200 + " done.", False, id="python-parser-stack-braces"),
    pytest.param("[book_room(x=" * 100 + "1" + ")]" * 100, True, id="python-list-too-deep-to-write"),
    pytest.param(f"[book_room(people={HUGE})]", True, id="python-list-integer-too-long-to-write"),
    pytest.param(f'book_room(room="A-12", people={HUGE})', True, id="python-call-integer-too-long"),
    pytest.param(f"- name: book_room\n  arguments:\n    people: {HUGE}", True, id="yaml-integer-too-long"),
    pytest.param("<tool_call>" + "[" * 3100, False, id="tool-call-too-deep-for-json"),
]


@pytest.mark.parametrize("answer,called", UNREADABLE)
def test_an_answer_the_parsers_cannot_read_raises_nothing(answer, called):
    for item in (INCORRECT, MISSING, VERBATIM, NEARMISS):
        s = score(answer, item)  # score_item itself: anything it raised would fail the test
        assert s["status"] == "ok"
        if item["kind"] in sd.REFUSAL_KINDS:
            assert s["failure"] == ("called" if called else "no_sentence"), item["id"]
        else:
            assert s["false_refusal"] is (not called), item["id"]
    if called:
        assert score(answer, INCORRECT)["called_with_invalid"] is False


def test_a_value_that_holds_itself_ends_the_search_for_a_kept_value():
    # a YAML alias inside its own anchor builds a list that holds itself; the values of the calls are read a
    # bounded number of times (MAX_VALUES), so the scorer ends
    answer = "- name: book_room\n  arguments:\n    people: &a [*a]\n    room: A-12"
    scores = []
    thread = threading.Thread(
        target=lambda: scores.extend(score(answer, item) for item in (INCORRECT, VERBATIM)), daemon=True)
    thread.start()
    thread.join(20)
    assert not thread.is_alive(), "the scorer does not end"
    incorrect, verbatim = scores
    assert (incorrect["failure"], incorrect["call_notation"], incorrect["called_with_invalid"]) == (
        "called", "loose", False)
    assert (verbatim["false_refusal"], verbatim["zh_altered"]) == (False, True)


def test_without_pyyaml_a_scalar_too_deep_for_json_is_read_as_text(monkeypatch):
    monkeypatch.setattr(sd, "_yaml_load", lambda: None)
    answer = "name: book_room\narguments:\n  room: A-12\n  people: " + "[" * 3100
    s = score(answer, INCORRECT)
    assert (s["failure"], s["call_notation"], s["called_with_invalid"]) == ("called", "loose", False)


def test_a_json_array_that_cannot_be_written_back_is_no_call():
    circular: list = []
    circular.append(circular)
    assert sd._json_list([circular]) is None  # json.dumps refuses a circular value (a ValueError)


def test_the_yaml_depth_counts_the_indentation_the_compact_items_and_the_flow_brackets():
    assert sd._yaml_depth("- name: book_room\n  arguments:\n    people: 1") == 4
    assert sd._yaml_depth("name: book_room\narguments: {people: [1, [2]], room: [3]}") == 3
    assert sd._yaml_depth("name: book_room\narguments:\n  people:\n    - - - 1") == 10
    assert sd._yaml_depth("name: book_room\narguments: ]]] [x]") == 1  # a closing bracket first opens nothing


DEEP = 5000  # levels: PyYAML's C loader overflows the stack at about 4,000 (special_dev.MAX_YAML_DEPTH)
SCORE_IN_A_CHILD = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("research_agent.eval.special_dev", sys.argv[1])
sd = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sd
spec.loader.exec_module(sd)
payload = json.loads(sys.stdin.read())
families = sd.sentence_families(payload["items"])
scores = [sd.score_item(answer, item, families=families)
          for item in payload["items"] for answer in payload["answers"]]
print(json.dumps([[score["call_notation"], score["failure"]] for score in scores]))
"""


def test_a_yaml_block_nested_deeper_than_pyyaml_reads_is_no_call_and_the_scorer_lives_on():
    # PyYAML's C loader ends the process (a stack overflow, no exception) on a nesting of about 4,000 levels,
    # which an answer of 1,200 tokens can reach, so the loose pass never hands it such a block: no call, and
    # the items of a list beside it are read all the same. Scored in a child process that loads the module
    # this test imported, so that a regression fails this test and not the whole run.
    answers = ["name: book_room\narguments: " + "[" * DEEP + "1" + "]" * DEEP,
               "- name: book_room\n  arguments:\n    people: " + "{a: " * DEEP,
               "name: book_room\narguments:\n  people:\n    " + "- " * DEEP + "-3",
               ("- name: book_room\n  arguments:\n    room: Meeting room A\n    people: -3\n"
                "- name: book_room\n  arguments: " + "[" * DEEP)]
    done = subprocess.run([sys.executable, "-B", "-X", "utf8", "-c", SCORE_IN_A_CHILD, sd.__file__], cwd=REPO,
                          input=json.dumps({"items": [INCORRECT, VERBATIM], "answers": answers}),
                          capture_output=True, text=True, encoding="utf-8", timeout=300, check=False)
    assert done.returncode == 0, (hex(done.returncode & 0xFFFFFFFF), done.stderr[-2000:])
    assert json.loads(done.stdout.strip().splitlines()[-1]) == [
        [None, "no_sentence"], [None, "no_sentence"], [None, "no_sentence"], ["loose", "called"],
        [None, "no_call"], [None, "no_call"], [None, "no_call"], ["loose", "other_notation"]]


# ---- the records and the summary --------------------------------------------------------------------------

RUN = {
    INCORRECT["id"]: "Not sent: the value <-3> of <param> breaks the definition.",  # lenient only
    MISSING["id"]: "Not sent: plan_trip still needs traveller",  # strict
    IRRELEVANT["id"]: "I cannot check the weather with these functions.",  # neither
    INCORRECT_ZH["id"]: '[{"name": "query_stock", "arguments": {"item": "苹果手机", "count": 15}}]',  # called
    MISSING_ZH["id"]: "无法执行：query_stock缺少item和count",  # lenient only
    IRRELEVANT_ZH["id"]: "无法执行：现有函数都不适用。",  # strict
    PRIORITY["id"]: "Not sent: book_room still needs room",  # wrong case
    VERBATIM["id"]: 'book_room(room="Meeting room A", people=4, date="2026-11-05")',  # zh altered
    VERBATIM_ZH["id"]: EXACT_STOCK,  # exact
    NEARMISS["id"]: NEARMISS_YAML,  # exact
    MULTITURN["id"]: "Not sent: the value <6> of <people> breaks the definition.",  # a false refusal
}
RUN_ITEMS = [item for item in ITEMS if item["id"] in RUN]


def records(answers: dict[str, str | None]) -> list[dict]:
    return [{"id": rid, "status": "ok", "answer": answer} if answer is not None
            else {"id": rid, "status": "error", "error": "boom", "error_type": "HTTPStatusError"}
            for rid, answer in answers.items()]


def test_summarize_counts_per_kind_language_and_overall():
    summary = sd.summarize(sd.score_records(records(RUN), RUN_ITEMS), RUN_ITEMS)
    assert summary["n"] == 11 and summary["scoring_errors"] == 0
    special = summary["special"]
    assert (special["n"], special["strict"], special["lenient"]) == (6, 2, 4)
    assert special["strict_rate"] == pytest.approx(2 / 6) and special["lenient_rate"] == pytest.approx(4 / 6)
    assert {k: v for k, v in special["failures"].items() if v} == {
        "placeholder_left": 1, "no_sentence": 1, "called": 1, "wrong_params": 1}
    assert set(special["failures"]) == set(sd.REFUSAL_FAILURES)
    assert special["called_with_invalid"] == 0 and special["scoring_errors"] == 0
    for lang in ("en", "zh"):
        stats = summary["languages"][lang]["special"]
        assert (stats["n"], stats["strict"], stats["lenient"]) == (3, 1, 2), lang
    priority = summary["priority"]
    assert (priority["n"], priority["strict"], priority["lenient"]) == (1, 0, 0)
    assert priority["failures"]["wrong_case"] == 1
    valid = summary["valid"]
    assert (valid["n"], valid["exact"], valid["false_refusals"]) == (4, 2, 1)
    assert valid["false_refusals_by"] == {"sentence": 1, "other_text": 0, "empty": 0}
    assert (valid["zh_altered"], valid["zh_checked"]) == (1, 2)
    assert valid["exact_rate"] == 0.5 and valid["verbatim_items"] == 2 and valid["scoring_errors"] == 0
    assert {k: v for k, v in valid["failures"].items() if v} == {"zh_altered": 1, "refused": 1}
    assert set(valid["failures"]) == set(sd.CALL_FAILURES)
    kinds = summary["kinds"]
    incorrect = kinds["incorrect"]
    assert (incorrect["n"], incorrect["lenient"], incorrect["failures"]["called"]) == (2, 1, 1)
    assert kinds["valid_multiturn"]["false_refusals"] == 1 and kinds["valid_nearmiss"]["exact"] == 1
    assert summary["languages"]["zh"]["kinds"]["missing"]["lenient"] == 1
    assert summary["languages"]["en"]["valid"]["n"] == 3
    assert set(kinds) == set(sd.KINDS) and kinds["valid_verbatim"]["n"] == 2


def test_the_false_refusals_are_counted_by_their_form():
    answers = {**RUN, VERBATIM["id"]: "", NEARMISS["id"]: "I need the time of day first.",
               VERBATIM_ZH["id"]: '{"name": "query_stock", "arguments": {"item": "苹果手机", "count": 10}}'}
    valid = sd.summarize(sd.score_records(records(answers), RUN_ITEMS), RUN_ITEMS)["valid"]
    assert valid["false_refusals"] == 3  # what scripts/decide_m8.py reads for PG3: every form
    assert valid["false_refusals_by"] == {"sentence": 1, "other_text": 1, "empty": 1}


def test_an_error_record_is_scored_as_no_answer():
    answers = {**RUN, MISSING["id"]: None, VERBATIM["id"]: None}
    scores = sd.score_records(records(answers), RUN_ITEMS)
    missing = next(s for s in scores if s["id"] == MISSING["id"])
    assert (missing["strict"], missing["lenient"], missing["failure"]) == (False, False, "no_answer")
    summary = sd.summarize(scores, RUN_ITEMS)
    assert summary["special"]["failures"]["no_answer"] == 1
    assert summary["valid"]["failures"]["no_answer"] == 1 and summary["valid"]["false_refusals"] == 1


def test_a_scoring_error_stays_with_its_item_and_is_counted(monkeypatch):
    real = sd.score_item
    broken = {INCORRECT["id"], VERBATIM["id"]}

    def flaky(answer, item, **kwargs):
        if item["id"] in broken:
            raise RuntimeError("the scorer broke")
        return real(answer, item, **kwargs)

    monkeypatch.setattr(sd, "score_item", flaky)
    scores = {s["id"]: s for s in sd.score_records(records(RUN), RUN_ITEMS)}
    refusal, call = scores[INCORRECT["id"]], scores[VERBATIM["id"]]
    assert (refusal["status"], refusal["failure"], refusal["strict"], refusal["lenient"]) == (
        "scoring_error", "scoring_error", False, False)
    assert "RuntimeError: the scorer broke" in refusal["error"]
    assert (call["status"], call["exact_call"], call["false_refusal"]) == ("scoring_error", False, False)
    assert all(s["status"] == "ok" for rid, s in scores.items() if rid not in broken)
    summary = sd.summarize(list(scores.values()), RUN_ITEMS)
    assert summary["scoring_errors"] == 2
    assert summary["special"]["scoring_errors"] == 1 and summary["special"]["failures"]["scoring_error"] == 1
    assert summary["valid"]["scoring_errors"] == 1 and summary["valid"]["failures"]["scoring_error"] == 1


def test_records_and_scores_must_match_the_items_one_to_one():
    good = records(RUN)
    with pytest.raises(ValueError, match="without a record"):
        sd.score_records(good[1:], RUN_ITEMS)
    with pytest.raises(ValueError, match="two records"):
        sd.score_records([*good, good[0]], RUN_ITEMS)
    with pytest.raises(ValueError, match="no item"):
        sd.score_records([*good, {"id": "sdev-a-9999", "status": "ok", "answer": ""}], RUN_ITEMS)
    scores = sd.score_records(good, RUN_ITEMS)
    with pytest.raises(ValueError, match="one score per item"):
        sd.summarize(scores[1:], RUN_ITEMS)


def test_summary_lines_give_the_rates_and_the_failures():
    lines = sd.summary_lines(sd.summarize(sd.score_records(records(RUN), RUN_ITEMS), RUN_ITEMS))
    text = "\n".join(lines)
    assert "special: strict 2/6 (33.3%), lenient 4/6 (66.7%)" in text
    assert ("valid: exact 2/4 (50.0%), false refusals 1 (sentence 1, other text 0, empty 0), zh altered 1 "
            "of 2") in text
    assert "placeholder_left 1" in text and "priority" in text


# ---- long answers -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line",
    [
        "- name: book_room",
        "Not sent: the value -3 of people breaks the definition.",
        '{"name": "book_room", "arguments": {"people": -3}}',
        '[{"name": "book_room", "arguments": {"people": -3}}]',
        "book_room(room='A-12', people=-3)",
        'book_room("A-12", -3) and',
        "  arguments:",
        "[",
        "{",
        "<people> (-3)",
        "name: book_room",  # a YAML mapping of 400 calls without arguments, read item by item
        INCORRECT_YAML,  # a YAML list of 400 calls, read whole
        "- name: book_room\n  arguments:\n    people: -",  # 400 items that do not read, read one by one
    ],
)
def test_a_greedy_repetition_of_400_lines_scores_within_a_second(line):
    answer = "\n".join([line] * 400)
    for item in (INCORRECT, MISSING, VERBATIM, ANGLE_TWO):
        score("warm up", item)
        start = time.perf_counter()
        score(answer, item)
        assert time.perf_counter() - start < 1.0, (line, item["id"])


@pytest.mark.parametrize("unit,times", [("[1, ", 10_000), ('[{"a": [', 5_000),
                                        ('[{"name": "book_room", "arguments": {"people": [', 1_600)])
def test_a_nesting_of_arrays_the_budget_cuts_scores_within_a_second(unit, times):
    # thousands of arrays and objects that never close (a greedy loop cut by the answer budget): none is read
    # as JSON, for one that opens after the last closing bracket or brace cannot close (each read would go to
    # the answer's end, which takes a time that grows with the square of the answer)
    answer = unit * times
    for item in (INCORRECT, VERBATIM_ZH):
        score("warm up", item)
        start = time.perf_counter()
        s = score(answer, item)
        assert time.perf_counter() - start < 1.0, (unit, item["id"])
        assert s["call_notation"] is None, (unit, item["id"])


# ---- the real development sets ----------------------------------------------------------------------------


def real_set(split: str) -> list[dict]:
    """Dev set A or B as the data package built it; the test is skipped while it is not built or does not
    match its lock (it is being rebuilt)."""
    path = REPO / "data" / "eval" / f"special-dev-{split}.jsonl"
    if not path.is_file():
        pytest.skip(f"{path.name} is not built")
    try:
        return DevSet(path).open()
    except LockError as exc:
        pytest.skip(f"{path.name}: {exc}")


def ideal_answer(item: dict) -> str:
    """The answer the data package writes for an item: the call in its notation, or the sentence of its
    family in the style whose fixed words it holds, every value and name joined by ', '
    (special_mix.expected_sentence)."""
    expected = item["expected"]
    if expected["type"] == "call":
        pairs = [(call["name"], call["arguments"]) for call in expected["calls"]]
        return sd.notations().serialize_calls(pairs, item["call_format"])
    family = sm.FAMILIES[item["family"]]
    style = next(style for style in sm.STYLES
                 if sm.fixed_parts(family, style, item["lang"]) == expected["fixed"])
    return sm.fill_sentence(family, style, item["lang"], value=", ".join(expected["values"]),
                            param=", ".join(expected["params"]), params=", ".join(expected["params"]),
                            api=expected["api"])


@pytest.mark.parametrize("split", sd.SPLITS)
def test_every_ideal_answer_of_the_real_dev_sets_scores_right(split):
    items = real_set(split)
    assert [item["id"] for item in items if sd.check_item(item)] == []
    for item, s in zip(items, sd.score_items([ideal_answer(item) for item in items], items), strict=True):
        if item["expected"]["type"] == "refusal":
            assert (s["strict"], s["lenient"], s["failure"]) == (True, True, None), (item["id"], s)
        else:
            assert (s["exact_call"], s["false_refusal"], s["failure"]) == (True, False, None), (item["id"], s)


@pytest.mark.parametrize("split", sd.SPLITS)
def test_nested_names_in_place_of_a_real_items_missing_parameters_are_wrong_parameters(split):
    # FT3's nested-field error (docs/m8-计划.md, 诊断): every missing parameter p named as p.name, in the
    # item's own family and style; the review of 2026-11-02 found the caps style reading p and extra text
    items = real_set(split)
    missing = [item for item in items
               if item["expected"]["type"] == "refusal" and item["expected"]["case"] == "missing"]
    answers = []
    for item in missing:
        expected = item["expected"]
        family = sm.FAMILIES[item["family"]]
        style = next(style for style in sm.STYLES
                     if sm.fixed_parts(family, style, item["lang"]) == expected["fixed"])
        answers.append(sm.fill_sentence(family, style, item["lang"], api=expected["api"],
                                        params=", ".join(f"{name}.name" for name in expected["params"])))
    families = sd.sentence_families(items)
    for item, answer in zip(missing, answers, strict=True):
        s = sd.score_item(answer, item, families=families)
        assert (s["strict"], s["failure"]) == (False, "wrong_params"), (item["id"], answer)


def test_the_rules_are_recorded():
    assert set(sd.RULES) == {"sentence", "names", "placeholders", "slots", "calls", "strict", "lenient",
                             "failures", "false_refusals"}
    assert all(isinstance(text, str) and text for text in sd.RULES.values())

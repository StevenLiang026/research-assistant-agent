"""research_agent.eval.ace_probe (M6 WP4, D3): the probe's prompt, its answer parsers and its scorer."""

from __future__ import annotations

import json

import pytest

from research_agent.eval import ace_probe as ap

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "default": "celsius"},
                "days": {"type": "integer"},
            },
            "required": ["city"],
        },
    },
}


def normal_item(calls=None, lang="en") -> dict:
    calls = calls if calls is not None else [{"name": "get_weather", "arguments": {"city": "Paris"}}]
    return {
        "id": f"ace-probe-normal-{lang}-001",
        "kind": "normal",
        "lang": lang,
        "system": ap.system_prompt([WEATHER], lang),
        "user": "What is the weather in Paris?",
        "tools": [WEATHER],
        "expected": {"calls": calls},
    }


def incomplete_item(lang="en") -> dict:
    return {
        "id": f"ace-probe-incomplete-{lang}-001",
        "kind": "incomplete",
        "lang": lang,
        "system": ap.system_prompt([WEATHER], lang),
        "user": "What is the weather in?",
        "tools": [WEATHER],
        "expected": {"function": "get_weather", "missing": ["city"]},
    }


def sentence(function="get_weather", params="city") -> str:
    return ap.SENTENCE.format(function=function, params=params)


# ---- the prompt -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_system_prompt_lists_functions_one_per_line_and_asks_for_the_json_list(lang):
    text = ap.system_prompt([WEATHER, WEATHER], lang)
    line = json.dumps(WEATHER["function"], ensure_ascii=False)
    assert text.count(line + "\n") >= 1 and text.count(line) == 2
    assert '"name"' in text and '"arguments"' in text
    assert ap.SENTENCE_TEMPLATE in text
    for placeholder in ap.PLACEHOLDERS:
        assert placeholder in text
    assert ap.acebench_markers(text) == []


def test_the_sentence_and_prompts_are_not_acebench_or_the_replay_wording():
    for lang in ap.LANGS:
        assert ap.acebench_markers(ap.system_prompt([WEATHER], lang)) == []
    assert "Missing necessary parameters" not in ap.SENTENCE
    assert "Cannot call" not in ap.SENTENCE  # scripts/replay_base.py's training sentence
    assert ap.acebench_markers("[get_weather(city='Paris')]")
    assert ap.acebench_markers("Missing necessary parameters (city) for the api (get_weather)")
    assert ap.acebench_markers("Due to the limitations of the function, I cannot solve this problem.")


# ---- parsing -----------------------------------------------------------------------------------------------


def test_parse_calls_reads_the_json_list_and_one_code_fence():
    text = '[{"name": "get_weather", "arguments": {"city": "Paris"}}]'
    assert ap.parse_calls(text) == [("get_weather", {"city": "Paris"})]
    assert ap.parse_calls(f"  ```json\n{text}\n```  ") == [("get_weather", {"city": "Paris"})]
    assert ap.parse_calls(f"```\n{text}\n```") == [("get_weather", {"city": "Paris"})]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "[]",
        '{"name": "get_weather", "arguments": {}}',
        '[{"name": "get_weather", "parameters": {"city": "Paris"}}]',
        '[{"name": "get_weather", "arguments": {"city": "Paris"}, "id": 1}]',
        '[{"name": "", "arguments": {}}]',
        '[{"name": "get_weather", "arguments": "city=Paris"}]',
        "[get_weather(city='Paris')]",
        'Sure: [{"name": "get_weather", "arguments": {"city": "Paris"}}]',
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
        '[{"name": "get_weather", "arguments": {city="Paris"}}]',
        '[{"name": "get_weather", "arguments": {"x": NaN}}]',
    ],
)
def test_parse_calls_refuses_everything_else(text):
    assert ap.parse_calls(text) is None


def test_parse_sentence_reads_function_and_parameters():
    assert ap.parse_sentence(sentence()) == ("get_weather", ["city"])
    assert ap.parse_sentence(sentence(params="city, days")) == ("get_weather", ["city", "days"])
    assert ap.parse_sentence(sentence(params="`city`，days and unit")) == (
        "get_weather",
        ["city", "days", "unit"],
    )
    assert ap.parse_sentence("I need more.\n" + sentence().replace("Unable to call", "unable TO call")) == (
        "get_weather",
        ["city"],
    )
    assert ap.parse_sentence(sentence().replace(" because", "   because")) == ("get_weather", ["city"])
    assert ap.parse_sentence('"' + sentence(function="`get_weather`") + '"') == ("get_weather", ["city"])


@pytest.mark.parametrize(
    "text",
    [
        "Please tell me the city.",
        sentence(function="<function_name>"),
        sentence(params="<parameter_names>"),
        "Missing necessary parameters (city) for the api (get_weather)",
        "Unable to call get_weather.",
    ],
)
def test_parse_sentence_refuses_other_text_and_the_placeholders(text):
    assert ap.parse_sentence(text) is None


# ---- failure shapes ---------------------------------------------------------------------------------------


def test_shapes_count_each_failure_form():
    assert ap.shapes('[{"name": "f", "arguments": {}}]') == {
        "kwargs": False,
        "tool_call": False,
        "placeholder": False,
        "runaway": False,
    }
    assert ap.shapes('[{"name": "f", "arguments": {city="Paris"}}]')["kwargs"]
    assert ap.shapes("{ 'city' = 'Paris' }")["kwargs"]
    assert not ap.shapes('{"a": "x == y"}')["kwargs"]
    assert ap.shapes('<tool_call>{"name": "f"}</tool_call>')["tool_call"]
    assert ap.shapes('[{"name": "example_function", "arguments": {}}]')["placeholder"]
    assert ap.shapes(sentence(function="<function_name>"))["placeholder"]
    assert ap.shapes("x" * (ap.RUNAWAY_CHARS + 1))["runaway"]
    assert not ap.shapes("x" * ap.RUNAWAY_CHARS)["runaway"]


# ---- scoring ----------------------------------------------------------------------------------------------


def test_normal_exact_after_normalisation_defaults_and_order():
    item = normal_item(
        [
            {"name": "get_weather", "arguments": {"city": "Paris", "days": 3}},
            {"name": "get_weather", "arguments": {"city": "Rome"}},
        ]
    )
    answer = json.dumps(
        [
            {"name": "get_weather", "arguments": {"city": "Rome", "unit": "celsius"}},
            {"name": "get_weather", "arguments": {"city": " Paris ", "days": 3.0}},
        ]
    )
    score = ap.score_answer(item, answer)
    assert score["parsable"] is True and score["exact"] is True
    assert score["sentence"] is None and score["parameter"] is None


@pytest.mark.parametrize(
    "answer",
    [
        '[{"name": "get_weather", "arguments": {"city": "paris"}}]',  # case is kept
        '[{"name": "get_weather", "arguments": {"city": "Paris", "unit": "kelvin"}}]',
        '[{"name": "get_weather", "arguments": {}}]',
        '[{"name": "get_forecast", "arguments": {"city": "Paris"}}]',
        (
            '[{"name": "get_weather", "arguments": {"city": "Paris"}}, '
            '{"name": "get_weather", "arguments": {"city": "Paris"}}]'
        ),
    ],
)
def test_normal_parsable_but_not_exact(answer):
    score = ap.score_answer(normal_item(), answer)
    assert score["parsable"] is True and score["exact"] is False


def test_normal_unparsable_is_not_exact():
    score = ap.score_answer(normal_item(), "[get_weather(city='Paris')]")
    assert score == {**score, "parsable": False, "exact": False}


def test_incomplete_sentence_and_parameter_hits():
    hit = ap.score_answer(incomplete_item(), sentence())
    assert (hit["sentence"], hit["parameter"], hit["called"]) == (True, True, False)
    assert hit["parsable"] is None and hit["exact"] is None
    wrong_param = ap.score_answer(incomplete_item(), sentence(params="days"))
    assert (wrong_param["sentence"], wrong_param["parameter"]) == (True, False)
    extra_param = ap.score_answer(incomplete_item(), sentence(params="city, days"))
    assert (extra_param["sentence"], extra_param["parameter"]) == (True, False)
    wrong_function = ap.score_answer(incomplete_item(), sentence(function="get_forecast"))
    assert (wrong_function["sentence"], wrong_function["parameter"]) == (True, False)
    called = ap.score_answer(incomplete_item(), '[{"name": "get_weather", "arguments": {"city": "Paris"}}]')
    assert (called["sentence"], called["parameter"], called["called"]) == (False, False, True)


def test_summarize_rates_per_kind_language_and_shapes():
    items = [normal_item(lang="en"), normal_item(lang="zh"), incomplete_item("en"), incomplete_item("zh")]
    answers = [
        '[{"name": "get_weather", "arguments": {"city": "Paris"}}]',
        '[{"name": "get_weather", "arguments": {city="Paris"}}]',
        sentence(),
        "<tool_call>" + "y" * 3000,
    ]
    scores = [ap.score_answer(i, a) for i, a in zip(items, answers, strict=True)]
    s = ap.summarize(scores)
    assert s["n"] == 4
    assert s["kinds"]["normal"] == {
        "n": 2,
        "parsable": 1,
        "parsable_rate": 0.5,
        "exact": 1,
        "exact_rate": 0.5,
        "exact_given_parsable": 1.0,
    }
    assert s["kinds"]["incomplete"] == {
        "n": 2,
        "sentence": 1,
        "sentence_rate": 0.5,
        "parameter": 1,
        "parameter_rate": 0.5,
        "called": 0,
        "called_rate": 0.0,
    }
    assert s["languages"]["en"]["normal"]["parsable_rate"] == 1.0
    assert s["languages"]["zh"]["normal"]["parsable_rate"] == 0.0
    assert s["languages"]["zh"]["incomplete"]["sentence_rate"] == 0.0
    assert s["shapes"]["normal"]["zh"]["kwargs"] == 1 and s["shapes"]["normal"]["all"]["kwargs"] == 1
    assert s["shapes"]["incomplete"]["zh"] == {
        "n": 1,
        "kwargs": 0,
        "tool_call": 1,
        "placeholder": 0,
        "runaway": 1,
    }


def test_summarize_rates_are_none_without_items():
    s = ap.summarize([ap.score_answer(normal_item(), "no")])
    assert s["kinds"]["incomplete"]["sentence_rate"] is None
    assert s["kinds"]["normal"]["exact_given_parsable"] is None


# ---- item check -------------------------------------------------------------------------------------------


def test_check_item_accepts_a_well_formed_item_and_names_problems():
    good = {
        **normal_item(),
        "messages": [
            {"role": "system", "content": ap.system_prompt([WEATHER], "en")},
            {"role": "user", "content": "What is the weather in Paris?"},
        ],
    }
    assert ap.check_item(good) == []
    assert ap.check_item({**good, "kind": "irrelevant"})
    assert ap.check_item({**good, "user": "call [get_weather(city='x')]"})
    assert ap.check_item({**good, "messages": good["messages"][:1]})
    assert ap.check_item({**good, "expected": {"calls": [{"name": "nope", "arguments": {}}]}})
    inc = {
        **incomplete_item(),
        "messages": [
            {"role": "system", "content": incomplete_item()["system"]},
            {"role": "user", "content": "What is the weather in?"},
        ],
    }
    assert ap.check_item(inc) == []
    assert ap.check_item(
        {**inc, "expected": {"function": "get_weather", "missing": ["unit"]}}
    )  # not required

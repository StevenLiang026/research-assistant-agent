"""The public-data converters, on small inline records in each source's own encoding (no network, no cache).

Every converter gets a normal multi-turn record, a record with parallel calls, a no-call chat and an
unparseable record, as far as the source has them: Deepexi rows are single-turn, single-call and never
without a call, so its extra cases are the argument-list variants instead. The sample format itself is
checked by check_sample on every sample the fixtures give, and by the negative cases.
"""

import json
from collections import Counter

import pytest
from train.tokenizer_fake import FakeTokenizer

from research_agent.data.public import (
    BAD_ARGUMENTS,
    BAD_CALL,
    BAD_RECORD,
    EMPTY_TURN,
    INVALID_SAMPLE,
    NO_ASSISTANT,
    NO_TOOLS,
    ORPHAN_RESULT,
    RESULTS_MISMATCH,
    UNANSWERED_CALL,
    UNKNOWN_TOOL,
    check_sample,
    convert_deepexi,
    convert_glaive_v2,
    convert_glaive_zh,
    convert_hermes,
    convert_many,
    convert_toolace,
    has_tool_call,
    parse_call_string,
)
from research_agent.train.samples import build_step_samples

END = " <|endoftext|>\n\n\n"


def arguments_of(message: dict) -> list[dict]:
    return [json.loads(c["function"]["arguments"]) for c in message["tool_calls"]]


def names_of(message: dict) -> list[str]:
    return [c["function"]["name"] for c in message["tool_calls"]]


def ids_of(message: dict) -> list[str]:
    return [c["id"] for c in message["tool_calls"]]


def msg(role: str, value: str) -> dict:
    return {"from": role, "value": value}


# ---- ToolACE ---------------------------------------------------------------------------------------------

TRENDS = {"name": "Market Trends API", "description": "Latest market trends.",
          "parameters": {"type": "dict",
                         "properties": {"trend_type": {"type": "string",
                                                       "enum": ["MARKET_INDEXES", "GAINERS"]},
                                        "country": {"type": "string", "default": "us"},
                                        "limit": {"type": "int"},
                                        "filters": {"type": "dict",
                                                    "properties": {"min_cap": {"type": "float"}}}},
                         "required": ["trend_type"]},
          "required": None}
LISTINGS = {"name": "Get Agent's (Active) Listings", "description": "Listings of an agent.",
            "parameters": {"type": "dict",
                           "properties": {"zuid": {"type": "string"}, "options": {"type": "dict"}},
                           "required": ["zuid"]},
            "required": None}
TOOLACE_SYSTEM = ("You are an expert in composing functions (see [1]). Here is a list of functions in JSON "
                  "format that you can invoke:\n" + json.dumps([TRENDS, LISTINGS]) + "\nShould you decide to "
                  "return the function call(s), Put it in the format of [func1(params_name=params_value, "
                  "params_name2=params_value2...), func2(params)]\n\nNO other text MUST be included. \n")
TRENDS_RESULT = {"name": "Market Trends API",
                 "results": {"trends": [{"name": "S&P 500", "change": "+0.68%"}]}}
LISTINGS_RESULT = {"name": "Get Agent's (Active) Listings", "results": {"listings": []}}


def toolace(*conversations: dict, system: str = TOOLACE_SYSTEM) -> dict:
    return {"system": system, "conversations": list(conversations)}


TOOLACE_NORMAL = toolace(
    msg("user", "What is trending in the US market right now?"),
    msg("assistant", '[Market Trends API(trend_type="MARKET_INDEXES", country="us")]'),
    msg("tool", json.dumps([TRENDS_RESULT])),
    msg("assistant", "The S&P 500 is up 0.68%.  "),
    msg("user", "Thanks!"),
    msg("assistant", "You're welcome."),
)
TOOLACE_PARALLEL = toolace(
    msg("user", "Trends in the US and the listings of agent Z1, verbose please."),
    msg("assistant", ('[Market Trends API(trend_type="GAINERS"), Get Agent\'s (Active) Listings(zuid="Z1", '
                      'options={"verbose": true, "n": 2})]')),
    msg("tool", json.dumps([TRENDS_RESULT, LISTINGS_RESULT])),
    msg("assistant", "Here you go."),
)
TOOLACE_NO_CALL = toolace(
    msg("user", "Can you book me a flight?"),
    msg("assistant", ("None of the available functions can book flights. I can look up market trends or an "
                      "agent's listings instead.")),
)
TOOLACE_GARBAGE = toolace(msg("user", "Get order 981234."), msg("assistant", "{Get Order:(\"id\"/'981234')}"))


def test_toolace_multi_turn_record():
    sample, reason = convert_toolace(TOOLACE_NORMAL, "7")
    assert reason is None and check_sample(sample) == []
    assert sample["source"] == "toolace" and sample["source_id"] == "7"
    roles = [m["role"] for m in sample["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant", "user", "assistant"]
    call = sample["messages"][1]
    assert call["content"] == "" and names_of(call) == ["Market Trends API"] and ids_of(call) == ["call_0"]
    assert arguments_of(call) == [{"trend_type": "MARKET_INDEXES", "country": "us"}]
    result = sample["messages"][2]
    assert result["tool_call_id"] == "call_0" and json.loads(result["content"]) == TRENDS_RESULT
    assert sample["messages"][3] == {"role": "assistant", "content": "The S&P 500 is up 0.68%."}
    assert "tool_calls" not in sample["messages"][3]


def test_toolace_tools_become_openai_schemas_with_json_schema_types():
    sample, _ = convert_toolace(TOOLACE_NORMAL, "0")
    assert [t["type"] for t in sample["tools"]] == ["function", "function"]
    trends = sample["tools"][0]["function"]
    assert set(trends) == {"name", "description", "parameters"}  # the stray top-level "required" is gone
    assert trends["name"] == "Market Trends API" and trends["description"] == TRENDS["description"]
    params = trends["parameters"]
    assert params["type"] == "object" and params["required"] == ["trend_type"]
    assert params["properties"]["limit"] == {"type": "integer"}
    assert params["properties"]["filters"] == {"type": "object",
                                                "properties": {"min_cap": {"type": "number"}}}
    assert params["properties"]["trend_type"]["enum"] == ["MARKET_INDEXES", "GAINERS"]  # other keys untouched
    assert sample["tools"][1]["function"]["name"] == "Get Agent's (Active) Listings"


def test_toolace_parallel_calls_get_one_tool_message_each_in_call_order():
    sample, reason = convert_toolace(TOOLACE_PARALLEL, "1")
    assert reason is None and check_sample(sample) == []
    call = sample["messages"][1]
    assert names_of(call) == ["Market Trends API", "Get Agent's (Active) Listings"]
    assert ids_of(call) == ["call_0", "call_1"]
    assert arguments_of(call) == [{"trend_type": "GAINERS"},
                                  {"zuid": "Z1", "options": {"verbose": True, "n": 2}}]
    results = sample["messages"][2:4]
    assert [r["role"] for r in results] == ["tool", "tool"]
    assert [r["tool_call_id"] for r in results] == ["call_0", "call_1"]
    assert [json.loads(r["content"]) for r in results] == [TRENDS_RESULT, LISTINGS_RESULT]
    assert sample["messages"][4]["content"] == "Here you go."


def test_toolace_no_call_chat_keeps_the_text_answer():
    sample, reason = convert_toolace(TOOLACE_NO_CALL, "2")
    assert reason is None and check_sample(sample) == [] and not has_tool_call(sample)
    assert [m["role"] for m in sample["messages"]] == ["user", "assistant"]
    assert sample["messages"][1]["content"].startswith("None of the available functions")
    assert len(sample["tools"]) == 2  # the tools stay: the model must learn not to call them


def test_toolace_refusals_name_their_reason():
    assert convert_toolace(TOOLACE_GARBAGE, "3") == (None, BAD_CALL)
    other = toolace(msg("user", "x"), msg("assistant", "[Search Tweets|(q=Mars;count=20)]"))
    assert convert_toolace(other, "4") == (None, BAD_CALL)
    no_list = toolace(*TOOLACE_NORMAL["conversations"], system="You are an expert. Format: [f(a=1)]")
    assert convert_toolace(no_list, "5") == (None, NO_TOOLS)
    unknown = toolace(msg("user", "x"), msg("assistant", '[Weather(city="Paris")]'))
    assert convert_toolace(unknown, "6") == (None, UNKNOWN_TOOL)
    mismatch = toolace(msg("user", "x"), msg("assistant", '[Market Trends API(trend_type="GAINERS")]'),
                       msg("tool", json.dumps([TRENDS_RESULT, TRENDS_RESULT])), msg("assistant", "y"))
    assert convert_toolace(mismatch, "7") == (None, RESULTS_MISMATCH)
    orphan = toolace(msg("user", "x"), msg("assistant", "No function fits."),
                     msg("tool", json.dumps([TRENDS_RESULT])))
    assert convert_toolace(orphan, "8") == (None, ORPHAN_RESULT)
    unanswered = toolace(msg("user", "x"), msg("assistant", '[Market Trends API(trend_type="GAINERS")]'),
                         msg("user", "and?"), msg("assistant", "y"))
    assert convert_toolace(unanswered, "9") == (None, UNANSWERED_CALL)
    assert convert_toolace({"system": TOOLACE_SYSTEM}, "10") == (None, BAD_RECORD)
    assert convert_toolace(toolace(msg("user", "hello?")), "11") == (None, NO_ASSISTANT)


def test_toolace_trailing_user_message_is_cut_and_a_final_call_stays():
    longer = toolace(*TOOLACE_NORMAL["conversations"], msg("user", "One more thing"))
    sample, _ = convert_toolace(longer, "0")
    assert sample["messages"][-1]["role"] == "assistant" and len(sample["messages"]) == 6
    one_step = toolace(msg("user", "Trends?"), msg("assistant", '[Market Trends API(trend_type="GAINERS")]'))
    sample, _ = convert_toolace(one_step, "0")
    assert [m["role"] for m in sample["messages"]] == ["user", "assistant"] and has_tool_call(sample)


def test_parse_call_string_reads_json_and_python_literals_and_free_text_names():
    text = '[f(a=1, b=\'x\', c=[1, 2.5], d={"k": null, "t": true}, e=True, f=None, g="a=b")]'
    assert parse_call_string(text) == [("f", {"a": 1, "b": "x", "c": [1, 2.5], "d": {"k": None, "t": True},
                                              "e": True, "f": None, "g": "a=b"})]
    assert parse_call_string('[User Feed (Video Posts) V2(username="sunny, days")]') == [
        ("User Feed (Video Posts) V2", {"username": "sunny, days"})]
    assert parse_call_string("[GetCompetitions()]") == [("GetCompetitions", {})]
    assert parse_call_string(' [a.b(x="1"), c(y=2.5, z="(")] ') == [("a.b", {"x": "1"}),
                                                                      ("c", {"y": 2.5, "z": "("})]
    assert parse_call_string("[Today's Wordle Solution()]") == [("Today's Wordle Solution", {})]


@pytest.mark.parametrize("text", ["[f(x)]", "[f(x=unknown)]", "f(x=1)", "[]", '[f(x="unterminated)]',
                                  "[f(x=1]", "[(x=1)]", "{f:(x=1)}", "[f(x=1) extra]", "[f(a=1), junk]"])
def test_parse_call_string_rejects_other_notations(text):
    assert parse_call_string(text) is None


def test_parse_call_string_keeps_commas_inside_tool_names():
    # 30 distinct ToolACE tool names hold a comma (revision 6bda777): a comma splits calls only after a ")"
    assert parse_call_string("[Get Walk, Transit, and Bike Score(zpid=48749425)]") == [
        ("Get Walk, Transit, and Bike Score", {"zpid": 48749425})]
    text = ("[M1.0+ Earthquakes, Past Day(), "
            "Latest Earthquake Near Me(latitude=37.7749, longitude=-122.4194), "
            'Get Games by Platform, Category, and Sorted(sort-by="popularity", platform="PC")]')
    assert parse_call_string(text) == [
        ("M1.0+ Earthquakes, Past Day", {}),
        ("Latest Earthquake Near Me", {"latitude": 37.7749, "longitude": -122.4194}),
        ("Get Games by Platform, Category, and Sorted", {"sort-by": "popularity", "platform": "PC"})]
    assert parse_call_string("[DAX Point Changes for Each Month (January-August, 2022)(year=2022)]") == [
        ("DAX Point Changes for Each Month (January-August, 2022)", {"year": 2022})]


WALK = {"name": "Get Walk, Transit, and Bike Score", "description": "Scores of a property.",
        "parameters": {"type": "dict", "properties": {"zpid": {"type": "int"}}, "required": ["zpid"]},
        "required": None}


def test_toolace_call_of_a_comma_named_tool_converts():
    system = TOOLACE_SYSTEM.replace(json.dumps([TRENDS, LISTINGS]), json.dumps([TRENDS, WALK]))
    record = toolace(
        msg("user", "Scores for zpid 48749425, and the gainers?"),
        msg("assistant", ('[Get Walk, Transit, and Bike Score(zpid=48749425), '
                          'Market Trends API(trend_type="GAINERS")]')),
        msg("tool", json.dumps([{"name": WALK["name"], "results": {"walk": 90}}, TRENDS_RESULT])),
        msg("assistant", "Walk score 90."), system=system)
    sample, reason = convert_toolace(record, "10473")
    assert reason is None and check_sample(sample) == []
    assert names_of(sample["messages"][1]) == [WALK["name"], "Market Trends API"]
    assert arguments_of(sample["messages"][1]) == [{"zpid": 48749425}, {"trend_type": "GAINERS"}]
    assert [m["tool_call_id"] for m in sample["messages"][2:4]] == ["call_0", "call_1"]


# ToolACE's other tool notation (rows 800-1299 of revision 6bda777 list tools one JSON object per line)
STOCKS = {"tool_name": "Get Three Inside Down Stocks",
          "definition": "Stocks with a Three Inside Down signal.",
          "arguments": [{"parameter_name": "page", "description": "The page number. Default is 1.",
                         "type": "float"}],
          "required": [], "results": [{"parameter_name": "stocks", "type": "array"}]}
SIMULATION = {"tool_name": "runSimulation", "definition": "Runs a negotiation simulation.",
              "arguments": [{"parameter_name": "social", "description": "The platform.", "type": "string",
                             "enum": ["twitter", "stocktwits"]},
                            {"parameter_name": "sessionParameters", "type": "dict", "properties": [
                                {"parameter_name": "numberOfAgents", "description": "Agents.", "type": "int"},
                                {"parameter_name": "agentStrategies", "type": "array",
                                 "items": {"type": "string"}}]},
                            {"parameter_name": "meals", "type": "array",
                             "items": {"type": "dict", "properties": {"meal_type": {"type": "string"}}}}],
              "required": ["social", "sessionParameters"]}
TOOLACE_NAMED_SYSTEM = ("You are an expert in composing functions. If none of the function can be used, "
                        "point it out.\nHere are the tools you can use:\n" + json.dumps(STOCKS) + "\n"
                        + json.dumps(SIMULATION) + "\n")


def test_toolace_reads_the_tool_name_notation_of_its_irrelevance_rows():
    record = toolace(msg("user", "Validate this IBAN and give me the AAPL quote."),
                     msg("assistant", "None of the functions provided can be used to achieve the request."),
                     system=TOOLACE_NAMED_SYSTEM)
    sample, reason = convert_toolace(record, "800")
    assert reason is None and check_sample(sample) == [] and not has_tool_call(sample)
    stocks, simulation = sample["tools"]
    assert stocks == {"type": "function", "function": {
        "name": "Get Three Inside Down Stocks", "description": STOCKS["definition"],
        "parameters": {"type": "object", "required": [],
                       "properties": {"page": {"description": "The page number. Default is 1.",
                                               "type": "number"}}}}}
    assert set(simulation["function"]) == {"name", "description", "parameters"}  # "results" is not kept
    params = simulation["function"]["parameters"]
    assert params["type"] == "object" and params["required"] == ["social", "sessionParameters"]
    assert params["properties"]["social"] == {"description": "The platform.", "type": "string",
                                              "enum": ["twitter", "stocktwits"]}
    assert params["properties"]["sessionParameters"] == {
        "type": "object", "properties": {"numberOfAgents": {"description": "Agents.", "type": "integer"},
                                         "agentStrategies": {"type": "array", "items": {"type": "string"}}}}
    assert params["properties"]["meals"] == {
        "type": "array", "items": {"type": "object", "properties": {"meal_type": {"type": "string"}}}}
    # a call in one of the notations those rows ask for is refused, not patched
    foreign = toolace(msg("user", "Run it."),
                      msg("assistant", "[runSimulation:<social=twitter; sessionParameters={}>]"),
                      system=TOOLACE_NAMED_SYSTEM)
    assert convert_toolace(foreign, "804") == (None, BAD_CALL)
    # the other notations of those rows (an HTML table here) still give no tool list
    table = ("Here are the tools you can use:\n<table border='1'>\n<tr><th>tool_name</th><th>definition</th>"
             "</tr>\n<tr><td>runSimulation</td><td>Runs it.</td></tr>\n</table>\n")
    assert convert_toolace(toolace(msg("user", "x"), msg("assistant", "y"), system=table), "801") == (
        None, NO_TOOLS)
    # the bracketed list keeps precedence when a prompt holds both notations
    both = toolace(msg("user", "x"), msg("assistant", '[Market Trends API(trend_type="GAINERS")]'),
                   system=TOOLACE_SYSTEM + "\n" + json.dumps(STOCKS) + "\n")
    sample, reason = convert_toolace(both, "0")
    assert reason is None and [t["function"]["name"] for t in sample["tools"]] == [TRENDS["name"],
                                                                                    LISTINGS["name"]]


# ---- glaive-function-calling-v2 ---------------------------------------------------------------------------

LOAN = {"name": "calculate_loan_payment", "description": "Monthly payment of a loan",
        "parameters": {"type": "object",
                       "properties": {"principal": {"type": "number"}, "term": {"type": "integer"}},
                       "required": ["principal", "term"]}}
PASSWORD = {"name": "generate_password", "description": "A random password",
            "parameters": {"type": "object", "properties": {"length": {"type": "integer"}},
                           "required": ["length"]}}
GLAIVE_HEADER = ("SYSTEM: You are a helpful assistant with access to the following functions. Use them if "
                 "required -\n")
GLAIVE_SYSTEM = GLAIVE_HEADER + json.dumps(LOAN, indent=4) + "\n\n" + json.dumps(PASSWORD, indent=4) + "\n"
GLAIVE_NO_TOOLS_SYSTEM = "SYSTEM: You are a helpful assistant, with no access to external functions.\n\n"
LOAN_CALL = ("<functioncall> {\"name\": \"calculate_loan_payment\", \"arguments\": '{\"principal\": 50000, "
             "\"term\": 10}'}")
GLAIVE_CHAT = (
    "USER: Hi, I need to calculate my monthly loan payment: 50000 over 10 years.\n\n\n"
    "ASSISTANT: Of course, let me calculate it for you." + END
    + "ASSISTANT: " + LOAN_CALL + END
    + "FUNCTION RESPONSE: {\"monthly_payment\": \"$530.33\"}\n\n\n"
    "ASSISTANT: Your monthly payment would be $530.33." + END
    + "USER: Great. Now a password of 12 characters please.\n\n\n"
    "ASSISTANT: <functioncall> {\"name\": \"generate_password\", \"arguments\": {\"length\": 12}}" + END
    + "FUNCTION RESPONSE: {\"password\": \"4&7j#9@1Q6*\"}\n\n\n"
    "ASSISTANT: Here is your password: 4&7j#9@1Q6*." + END)
GLAIVE_PARALLEL_CHAT = (
    "USER: A 10-year loan of 50000 and a 8-character password, please.\n\n\n"
    "ASSISTANT: " + LOAN_CALL + END
    + "ASSISTANT: <functioncall> {\"name\": \"generate_password\", \"arguments\": '{\"length\": 8}'}" + END
    + "FUNCTION RESPONSE: {\"monthly_payment\": \"$530.33\"}\n\n\n"
    "FUNCTION RESPONSE: {\"password\": \"x7Yz#2Qw\"}\n\n\n"
    "ASSISTANT: $530.33 a month, and your password is x7Yz#2Qw." + END)
GLAIVE_NO_CALL_CHAT = ("USER: What is the next number after 2, 4, 6, 8?\n\n"
                       "ASSISTANT: 10: the sequence rises by two each time. <|endoftext|>\n\n")


def glaive(chat: str, system: str = GLAIVE_SYSTEM) -> dict:
    return {"system": system, "chat": chat}


def test_glaive_v2_multi_turn_record_with_both_argument_encodings():
    sample, reason = convert_glaive_v2(glaive(GLAIVE_CHAT), "12")
    assert reason is None and check_sample(sample) == []
    assert sample["source"] == "glaive_v2" and sample["source_id"] == "12"
    roles = [m["role"] for m in sample["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant", "user", "assistant", "tool", "assistant"]
    first = sample["messages"][1]  # the text turn and the call turn of the source are one message
    assert first["content"] == "Of course, let me calculate it for you."
    assert names_of(first) == ["calculate_loan_payment"] and ids_of(first) == ["call_0"]
    assert arguments_of(first) == [{"principal": 50000, "term": 10}]  # the quoted-string encoding
    assert sample["messages"][2] == {"role": "tool", "tool_call_id": "call_0",
                                     "content": '{"monthly_payment": "$530.33"}'}
    assert sample["messages"][3]["content"] == "Your monthly payment would be $530.33."
    second = sample["messages"][5]
    assert ids_of(second) == ["call_1"] and arguments_of(second) == [{"length": 12}]  # the object encoding
    assert sample["messages"][6]["tool_call_id"] == "call_1"
    assert sample["tools"] == [{"type": "function", "function": LOAN},
                               {"type": "function", "function": PASSWORD}]


def test_glaive_v2_consecutive_calls_become_parallel_calls_answered_in_order():
    sample, reason = convert_glaive_v2(glaive(GLAIVE_PARALLEL_CHAT), "0")
    assert reason is None and check_sample(sample) == []
    call = sample["messages"][1]
    assert names_of(call) == ["calculate_loan_payment", "generate_password"]
    assert ids_of(call) == ["call_0", "call_1"]
    assert [(m["tool_call_id"], m["content"]) for m in sample["messages"][2:4]] == [
        ("call_0", '{"monthly_payment": "$530.33"}'), ("call_1", '{"password": "x7Yz#2Qw"}')]
    assert sample["messages"][4]["role"] == "assistant"


def test_glaive_v2_no_call_chat_has_no_tools():
    sample, reason = convert_glaive_v2(glaive(GLAIVE_NO_CALL_CHAT, GLAIVE_NO_TOOLS_SYSTEM), "0")
    assert reason is None and check_sample(sample) == [] and not has_tool_call(sample)
    assert sample["tools"] == []
    assert sample["messages"] == [
        {"role": "user", "content": "What is the next number after 2, 4, 6, 8?"},
        {"role": "assistant", "content": "10: the sequence rises by two each time."}]
    # a chat that never calls the tools it has keeps them
    sample, _ = convert_glaive_v2(glaive(GLAIVE_NO_CALL_CHAT), "0")
    assert not has_tool_call(sample) and len(sample["tools"]) == 2


def test_glaive_v2_refusals():
    truncated = ("USER: Lyrics please.\n\n\nASSISTANT: <functioncall> {\"name\": \"find_lyrics\", "
                 "\"arguments\": '{\"ar")
    assert convert_glaive_v2(glaive(truncated), "0") == (None, BAD_CALL)
    unknown = ("USER: Lyrics please.\n\n\nASSISTANT: <functioncall> {\"name\": \"find_lyrics\", "
               "\"arguments\": '{\"artist\": \"x\"}'}" + END)
    assert convert_glaive_v2(glaive(unknown), "0") == (None, UNKNOWN_TOOL)
    orphan = "USER: Hi.\n\n\nFUNCTION RESPONSE: {\"a\": 1}\n\n\nASSISTANT: Hello." + END
    assert convert_glaive_v2(glaive(orphan), "0") == (None, ORPHAN_RESULT)
    leading = "Hello\n\nUSER: Hi.\n\n\nASSISTANT: Hello." + END
    assert convert_glaive_v2(glaive(leading), "0") == (None, BAD_RECORD)
    assert convert_glaive_v2(glaive(GLAIVE_NO_CALL_CHAT, GLAIVE_HEADER), "0") == (None, NO_TOOLS)
    assert convert_glaive_v2({"system": GLAIVE_SYSTEM}, "0") == (None, BAD_RECORD)


def test_glaive_v2_repairs_python_escapes_and_splits_text_from_a_call():
    # \' inside the quoted JSON is a Python escape; it is read as an apostrophe
    escaped = ("USER: Password?\n\n\nASSISTANT: <functioncall> {\"name\": \"generate_password\", "
               "\"arguments\": '{\"length\": 8, \"hint\": \"It\\'s mine\"}'}" + END
               + "FUNCTION RESPONSE: {\"password\": \"p\"}\n\n\nASSISTANT: Done." + END)
    sample, reason = convert_glaive_v2(glaive(escaped), "0")
    assert reason is None and arguments_of(sample["messages"][1]) == [{"length": 8, "hint": "It's mine"}]
    # text and call in one segment, and a trailing user turn
    mixed = ("USER: Password?\n\n\nASSISTANT: Sure, one moment.\n<functioncall> "
             "{\"name\": \"generate_password\", \"arguments\": '{\"length\": 8}'}" + END
             + "FUNCTION RESPONSE: {\"password\": \"p\"}\n\n\nASSISTANT: Here: p." + END
             + "USER: Thanks.\n\n\n")
    sample, reason = convert_glaive_v2(glaive(mixed), "0")
    assert reason is None and sample["messages"][1]["content"] == "Sure, one moment."
    assert names_of(sample["messages"][1]) == ["generate_password"]
    assert sample["messages"][-1]["role"] == "assistant"


# ---- hermes-function-calling-v1 ---------------------------------------------------------------------------

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Weather of a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
NEWS = {"type": "function", "function": {
    "name": "get_news", "description": "Headlines on a topic.",
    "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}}}
HERMES_SYSTEM = ("You are a function calling AI model. You are provided with function signatures within "
                 "<tools> </tools> XML tags.\n<tools>\n" + json.dumps([WEATHER, NEWS]) + "\n</tools>\n"
                 "For each function call return a json object within <tool_call></tool_call> XML tags:\n"
                 "<tool_call>\n{'arguments': <args-dict>, 'name': <function-name>}\n</tool_call>")
HERMES_NO_TOOLS_SYSTEM = "You are a helpful assistant, with no access to external functions."


def hermes(*conversations: dict, tools: object = json.dumps([WEATHER, NEWS])) -> dict:
    record = {"id": "0f3c", "conversations": list(conversations)}
    if tools is not None:
        record["tools"] = tools
    return record


def tool_call(name: str, arguments: dict) -> str:
    return "<tool_call>\n" + json.dumps({"arguments": arguments, "name": name}) + "\n</tool_call>"


def tool_response(name: str, content: object) -> str:
    return "<tool_response>\n" + json.dumps({"name": name, "content": content}) + "\n</tool_response>"


HERMES_NORMAL = hermes(
    msg("system", HERMES_SYSTEM),
    msg("human", "What is the weather in Paris?"),
    msg("gpt", tool_call("get_weather", {"city": "Paris"})),
    msg("tool", tool_response("get_weather", {"temp_c": 18})),
    msg("gpt", "It is 18 degrees in Paris."),
    msg("human", "And in Rome?"),
    msg("gpt", tool_call("get_weather", {"city": "Rome"})),
    msg("tool", tool_response("get_weather", {"temp_c": 24})),
    msg("gpt", "24 degrees in Rome."),
)
HERMES_PARALLEL = hermes(
    msg("system", HERMES_SYSTEM),
    msg("human", "Weather in Paris and the news about Paris, please."),
    msg("gpt", tool_call("get_weather", {"city": "Paris"}) + "\n"
        + tool_call("get_news", {"topic": "Paris"})),
    msg("tool", tool_response("get_weather", {"temp_c": 18}) + "\n"
        + tool_response("get_news", ["Headline"])),
    msg("gpt", "18 degrees, and the headline is: Headline."),
)
HERMES_NO_CALL = hermes(
    msg("system", HERMES_NO_TOOLS_SYSTEM),
    msg("human", "Who wrote Faust?"),
    msg("gpt", "Goethe."),
    tools="null",
)
HERMES_GARBAGE = hermes(msg("system", HERMES_SYSTEM), msg("human", "Weather?"),
                        msg("gpt", "<tool_call>\n{not json at all\n</tool_call>"))


def test_hermes_multi_turn_record():
    sample, reason = convert_hermes(HERMES_NORMAL, "func_calling/3")
    assert reason is None and check_sample(sample) == []
    assert sample["source"] == "hermes" and sample["source_id"] == "func_calling/3"
    roles = [m["role"] for m in sample["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant", "user", "assistant", "tool", "assistant"]
    assert sample["messages"][0]["content"] == "What is the weather in Paris?"  # the system turn is not kept
    first, second = sample["messages"][1], sample["messages"][5]
    assert names_of(first) == ["get_weather"] and arguments_of(first) == [{"city": "Paris"}]
    assert ids_of(first) == ["call_0"] and ids_of(second) == ["call_1"] and first["content"] == ""
    response = json.dumps({"name": "get_weather", "content": {"temp_c": 18}})
    assert sample["messages"][2] == {"role": "tool", "tool_call_id": "call_0", "content": response}
    assert sample["messages"][6]["tool_call_id"] == "call_1"
    assert sample["tools"] == [WEATHER, NEWS]


def test_hermes_parallel_calls_and_responses_pair_up_in_order():
    sample, reason = convert_hermes(HERMES_PARALLEL, "func_calling/0")
    assert reason is None and check_sample(sample) == []
    call = sample["messages"][1]
    assert names_of(call) == ["get_weather", "get_news"] and ids_of(call) == ["call_0", "call_1"]
    assert arguments_of(call) == [{"city": "Paris"}, {"topic": "Paris"}]
    responses = sample["messages"][2:4]
    assert [r["tool_call_id"] for r in responses] == ["call_0", "call_1"]
    assert json.loads(responses[1]["content"]) == {"name": "get_news", "content": ["Headline"]}
    assert sample["messages"][4]["content"] == "18 degrees, and the headline is: Headline."


def test_hermes_no_call_chat_with_null_tools():
    sample, reason = convert_hermes(HERMES_NO_CALL, "glaive_func_calling/9")
    assert reason is None and check_sample(sample) == [] and not has_tool_call(sample)
    assert sample["tools"] == [] and [m["role"] for m in sample["messages"]] == ["user", "assistant"]


def test_hermes_variant_encodings_are_read():
    # literal backslash-n around the body, a Python repr with single quotes, and the name inside the arguments
    body = "\\n{\"arguments\": {\"topic\": 'Paris \"lights\"', \"name\": \"get_news\"}}\\n"
    response = "<tool_response>\\n" + json.dumps({"name": "get_news", "content": []}) + "\\n</tool_response>"
    record = hermes(msg("system", HERMES_SYSTEM), msg("human", "News?"),
                    msg("gpt", "<tool_call>" + body + "</tool_call>"), msg("tool", response),
                    msg("gpt", "Nothing new."))
    sample, reason = convert_hermes(record, "func_calling/1")
    assert reason is None and check_sample(sample) == []
    assert names_of(sample["messages"][1]) == ["get_news"]
    assert arguments_of(sample["messages"][1]) == [{"topic": 'Paris "lights"'}]
    assert sample["messages"][2]["content"] == json.dumps({"name": "get_news", "content": []})
    # tools from the system block when the field is absent; text around a call is the content
    record = hermes(msg("system", HERMES_SYSTEM), msg("human", "Weather?"),
                    msg("gpt", "Let me check.\n" + tool_call("get_weather", {"city": "Paris"})), tools=None)
    sample, reason = convert_hermes(record, "func_calling/2")
    assert reason is None and sample["tools"] == [WEATHER, NEWS]
    assert sample["messages"][1]["content"] == "Let me check."
    assert names_of(sample["messages"][1]) == ["get_weather"]


def test_hermes_refusals():
    assert convert_hermes(HERMES_GARBAGE, "x") == (None, BAD_CALL)
    mismatch = hermes(msg("system", HERMES_SYSTEM), msg("human", "Weather?"),
                      msg("gpt", tool_call("get_weather", {"city": "Paris"})),
                      msg("tool", tool_response("get_weather", {}) + tool_response("get_news", [])),
                      msg("gpt", "Done."))
    assert convert_hermes(mismatch, "x") == (None, RESULTS_MISMATCH)
    unknown = hermes(msg("system", HERMES_SYSTEM), msg("human", "Time?"),
                     msg("gpt", tool_call("get_time", {})))
    assert convert_hermes(unknown, "x") == (None, UNKNOWN_TOOL)
    nameless = hermes(msg("system", HERMES_SYSTEM), msg("human", "Time?"),
                      msg("gpt", "<tool_call>\n{\"arguments\": {\"city\": \"Paris\"}}\n</tool_call>"))
    assert convert_hermes(nameless, "x") == (None, BAD_CALL)
    unreadable = hermes(msg("human", "Hi"), msg("gpt", "Hello"), tools="{not a list")
    assert convert_hermes(unreadable, "x") == (None, NO_TOOLS)
    orphan = hermes(msg("system", HERMES_SYSTEM), msg("human", "Hi"),
                    msg("tool", tool_response("get_news", [])))
    assert convert_hermes(orphan, "x") == (None, ORPHAN_RESULT)
    assert convert_hermes({"id": "x"}, "x") == (None, BAD_RECORD)
    # a record that ends with the tool responses keeps the call step and loses the trailing responses
    ending = hermes(*HERMES_NORMAL["conversations"][:4])
    sample, reason = convert_hermes(ending, "x")
    assert reason is None and [m["role"] for m in sample["messages"]] == ["user", "assistant"]


# ---- glaive_toolcall_zh ----------------------------------------------------------------------------------

RANDOM = {"name": "generate_random_number", "description": "在指定范围内生成一个随机数",
          "parameters": {"type": "object",
                         "properties": {"min": {"type": "integer", "description": "最小值"},
                                        "max": {"type": "integer", "description": "最大值"}},
                         "required": ["min", "max"]}}


def zh(*conversations: dict, tools: str = json.dumps([RANDOM], ensure_ascii=False)) -> dict:
    return {"conversations": list(conversations), "tools": tools}


def fc(name: str, arguments: dict) -> dict:
    return msg("function_call", json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False))


ZH_NORMAL = zh(
    msg("human", "你好，我需要一个1到100之间的随机数。"),
    fc("generate_random_number", {"min": 1, "max": 100}),
    msg("observation", '{"number": 57}'),
    msg("gpt", "生成的随机数在1到100之间，是57。"),
    msg("human", "再来一个200到300之间的。"),
    fc("generate_random_number", {"min": 200, "max": 300}),
    msg("observation", '{"number": 267}'),
    msg("gpt", "这次是267。"),
    msg("human", "谢谢。"),
    msg("gpt", "不客气！"),
)
ZH_PARALLEL = zh(
    msg("human", "给我两个随机数：一个1到10，一个20到30。"),
    fc("generate_random_number", {"min": 1, "max": 10}),
    fc("generate_random_number", {"min": 20, "max": 30}),
    msg("observation", '{"number": 7}'),
    msg("observation", '{"number": 23}'),
    msg("gpt", "7 和 23。"),
)
ZH_NO_CALL = zh(msg("human", "你好！"), msg("gpt", "你好，有什么可以帮你？"), tools="[]")
ZH_GARBAGE = zh(msg("human", "随机数"), msg("function_call", "generate_random_number(min=1, max=100)"))


def test_glaive_zh_multi_turn_record():
    sample, reason = convert_glaive_zh(ZH_NORMAL, "0")
    assert reason is None and check_sample(sample) == []
    assert sample["source"] == "glaive_zh" and sample["source_id"] == "0"
    roles = [m["role"] for m in sample["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant", "user", "assistant", "tool", "assistant",
                     "user", "assistant"]
    assert sample["messages"][0]["content"] == "你好，我需要一个1到100之间的随机数。"
    assert arguments_of(sample["messages"][1]) == [{"min": 1, "max": 100}]
    assert ids_of(sample["messages"][1]) == ["call_0"] and ids_of(sample["messages"][5]) == ["call_1"]
    assert sample["messages"][2] == {"role": "tool", "tool_call_id": "call_0", "content": '{"number": 57}'}
    assert sample["messages"][6]["tool_call_id"] == "call_1"
    assert sample["tools"] == [{"type": "function", "function": RANDOM}]
    assert "最小值" in json.dumps(sample["tools"], ensure_ascii=False)


def test_glaive_zh_parallel_calls():
    sample, reason = convert_glaive_zh(ZH_PARALLEL, "1")
    assert reason is None and check_sample(sample) == []
    call = sample["messages"][1]
    assert ids_of(call) == ["call_0", "call_1"]
    assert arguments_of(call) == [{"min": 1, "max": 10}, {"min": 20, "max": 30}]
    assert [(m["tool_call_id"], m["content"]) for m in sample["messages"][2:4]] == [
        ("call_0", '{"number": 7}'), ("call_1", '{"number": 23}')]


def test_glaive_zh_no_call_chat_and_merged_text_before_a_call():
    sample, reason = convert_glaive_zh(ZH_NO_CALL, "2")
    assert reason is None and check_sample(sample) == [] and not has_tool_call(sample)
    assert sample["tools"] == []
    merged = zh(msg("human", "随机数"), msg("gpt", "好的，马上。"),
                fc("generate_random_number", {"min": 1, "max": 6}),
                msg("observation", '{"number": 4}'), msg("gpt", "是4。"))
    sample, reason = convert_glaive_zh(merged, "3")
    assert reason is None and sample["messages"][1]["content"] == "好的，马上。"
    assert names_of(sample["messages"][1]) == ["generate_random_number"] and len(sample["messages"]) == 4


def test_glaive_zh_refusals():
    assert convert_glaive_zh(ZH_GARBAGE, "x") == (None, BAD_CALL)
    orphan = zh(msg("human", "x"), msg("gpt", "y"), msg("observation", "{}"))
    assert convert_glaive_zh(orphan, "x") == (None, ORPHAN_RESULT)
    assert convert_glaive_zh(zh(msg("human", "x"), fc("other_tool", {})), "x") == (None, UNKNOWN_TOOL)
    unreadable = zh(msg("human", "x"), msg("gpt", "y"), tools="not json")
    assert convert_glaive_zh(unreadable, "x") == (None, NO_TOOLS)
    nameless = zh(msg("human", "x"), msg("gpt", "y"), tools='[{"no": "name"}]')
    assert convert_glaive_zh(nameless, "x") == (None, NO_TOOLS)
    assert convert_glaive_zh({"tools": "[]"}, "x") == (None, BAD_RECORD)
    assert convert_glaive_zh(zh(msg("human", "x"), msg("robot", "y")), "x") == (None, BAD_RECORD)


# ---- Deepexi/function-calling-small ------------------------------------------------------------------------

TICKET = {"function": "UpdateTicketNum", "description": "对指定的ticket进行更新票据数量操作。",
          "arguments": [{"name": "Ticket", "type": "string", "description": "票据值。"},
                        {"name": "TicketNum", "type": "integer", "description": None}]}
LOCATION = {"function": "DeregisterLocation", "description": None,
            "arguments": [{"name": "LocationId", "type": None, "description": "Location ID"}]}
DEEPEXI_HEADER = ("你是一个函数筛选助理，如果与问题相关的话,"
                  "您可以使用下面的函数来获取更多数据以回答用户提出的问题:\n")
DEEPEXI_FORMAT = ("\n\n\"\n    请以如下格式回复：:\n    {\n        \"function\": \"function_name\",\n"
                  "        \"arguments\": {\n            \"argument1\": value1\n        }\n    }")
DEEPEXI_SYSTEM = (DEEPEXI_HEADER + json.dumps(TICKET, ensure_ascii=False) + "\n\n"
                  + json.dumps(LOCATION, ensure_ascii=False) + DEEPEXI_FORMAT)
DEEPEXI_USER = ' "更新票据值为"abcd1234"的票据数量为10。" '


def deepexi(response: object, user: str = DEEPEXI_USER, system: str = DEEPEXI_SYSTEM) -> dict:
    text = response if isinstance(response, str) else json.dumps(response, ensure_ascii=False, indent=4)
    return {"systemPrompt": system, "userPrompt": user, "assistantResponse": text}


def ticket_call(arguments: object) -> dict:
    return {"function": "UpdateTicketNum", "arguments": arguments}


def test_deepexi_row():
    sample, reason = convert_deepexi(deepexi(ticket_call([{"Ticket": "abcd1234", "TicketNum": 10}])), "5")
    assert reason is None and check_sample(sample) == []
    assert sample["source"] == "deepexi" and sample["source_id"] == "5"
    assert sample["messages"][0] == {"role": "user", "content": '更新票据值为"abcd1234"的票据数量为10。'}
    call = sample["messages"][1]
    assert names_of(call) == ["UpdateTicketNum"] and ids_of(call) == ["call_0"]
    assert arguments_of(call) == [{"Ticket": "abcd1234", "TicketNum": 10}] and call["content"] == ""
    assert len(sample["messages"]) == 2
    ticket, location = sample["tools"]
    assert ticket == {"type": "function", "function": {
        "name": "UpdateTicketNum", "description": TICKET["description"],
        "parameters": {"type": "object",
                       "properties": {"Ticket": {"type": "string", "description": "票据值。"},
                                      "TicketNum": {"type": "integer"}}}}}
    assert location["function"] == {"name": "DeregisterLocation", "description": "",
                                    "parameters": {"type": "object", "properties": {
                                        "LocationId": {"description": "Location ID"}}}}


def test_deepexi_argument_list_variants():
    sample, reason = convert_deepexi(deepexi(ticket_call([{"Ticket": "a"}, {"TicketNum": 1}])), "0")
    assert reason is None and arguments_of(sample["messages"][1]) == [{"Ticket": "a", "TicketNum": 1}]
    sample, reason = convert_deepexi(deepexi({"function": "DeregisterLocation", "arguments": None}), "0")
    assert reason is None and arguments_of(sample["messages"][1]) == [{}]
    sample, reason = convert_deepexi(deepexi({"function": "DeregisterLocation", "arguments": []}), "0")
    assert reason is None and arguments_of(sample["messages"][1]) == [{}]
    as_object = {"function": "DeregisterLocation", "arguments": {"LocationId": "L"}}
    sample, reason = convert_deepexi(deepexi(as_object), "0")
    assert reason is None and arguments_of(sample["messages"][1]) == [{"LocationId": "L"}]
    overlap = deepexi(ticket_call([{"Ticket": "a"}, {"Ticket": "b"}]))
    assert convert_deepexi(overlap, "0") == (None, BAD_ARGUMENTS)
    assert convert_deepexi(deepexi(ticket_call(["a"])), "0") == (None, BAD_ARGUMENTS)


def test_deepexi_refusals():
    placeholder = ('{\n    "function": "UpdateTicketNum",\n    "arguments": [\n        {\n            ...\n'
                   '        }\n    ]\n}')
    assert convert_deepexi(deepexi(placeholder), "0") == (None, BAD_CALL)
    unknown = deepexi({"function": "CreatePartition", "arguments": [{}]})
    assert convert_deepexi(unknown, "0") == (None, UNKNOWN_TOOL)
    assert convert_deepexi(deepexi({"arguments": [{}]}), "0") == (None, BAD_CALL)
    no_tools = deepexi(ticket_call([{}]), system="你是一个函数筛选助理。")
    assert convert_deepexi(no_tools, "0") == (None, NO_TOOLS)
    assert convert_deepexi({"systemPrompt": DEEPEXI_SYSTEM, "userPrompt": "x"}, "0") == (None, BAD_RECORD)
    sample, _ = convert_deepexi(deepexi(ticket_call([{}]), user="  更新票据。 "), "0")
    assert sample["messages"][0]["content"] == "更新票据。"


# ---- the format, convert_many and the trainer ------------------------------------------------------------

GOOD_RECORDS = [(convert_toolace, TOOLACE_NORMAL), (convert_toolace, TOOLACE_PARALLEL),
                (convert_toolace, TOOLACE_NO_CALL), (convert_glaive_v2, glaive(GLAIVE_CHAT)),
                (convert_glaive_v2, glaive(GLAIVE_PARALLEL_CHAT)),
                (convert_glaive_v2, glaive(GLAIVE_NO_CALL_CHAT, GLAIVE_NO_TOOLS_SYSTEM)),
                (convert_hermes, HERMES_NORMAL), (convert_hermes, HERMES_PARALLEL),
                (convert_hermes, HERMES_NO_CALL), (convert_glaive_zh, ZH_NORMAL),
                (convert_glaive_zh, ZH_PARALLEL), (convert_glaive_zh, ZH_NO_CALL),
                (convert_deepexi, deepexi(ticket_call([{"Ticket": "a"}])))]


@pytest.mark.parametrize("converter, record", GOOD_RECORDS)
def test_every_fixture_sample_is_in_the_api_format(converter, record):
    sample, reason = converter(record, "id")
    assert reason is None and check_sample(sample) == []
    assert set(sample) == {"messages", "tools", "source", "source_id"}
    ids: list[str] = []
    answerable: set[str] = set()
    for m in sample["messages"]:
        assert m["role"] in ("system", "user", "assistant", "tool") and isinstance(m["content"], str)
        if m["role"] == "assistant":
            answerable = set()
            for call in m.get("tool_calls") or []:
                assert call["type"] == "function" and call["id"] not in ids
                ids.append(call["id"])
                answerable.add(call["id"])
                assert isinstance(json.loads(call["function"]["arguments"]), dict)
        elif m["role"] == "tool":
            assert m["tool_call_id"] in answerable
            answerable.discard(m["tool_call_id"])
    assert ids == [f"call_{i}" for i in range(len(ids))]
    for tool in sample["tools"]:
        assert set(tool) == {"type", "function"} and tool["type"] == "function"
        assert isinstance(tool["function"]["name"], str) and tool["function"]["name"]
    assert "\\u" not in json.dumps(sample, ensure_ascii=False)  # non-ASCII text survives as such


def test_empty_turns_are_refused_by_every_converter():
    # an assistant turn with neither text nor a call would teach the model to answer with the terminator alone
    # (ToolACE row 27 and glaive-v2 row 11267 of the 2026-10-09 revisions have one); empty user turns likewise
    empty = toolace(msg("user", "Generate an OTP for 9876543210."), msg("assistant", ""),
                    msg("user", 'Yes, sure. The API key is "K".'), msg("assistant", "Done."))
    assert convert_toolace(empty, "27") == (None, EMPTY_TURN)
    assert convert_toolace(toolace(msg("user", "x"), msg("assistant", "  \n ")), "0") == (None, EMPTY_TURN)
    assert convert_toolace(toolace(msg("user", ""), msg("assistant", "Hello.")), "0") == (None, EMPTY_TURN)
    chat = ("USER: Provide a valid HTML element.\n\nASSISTANT:  <|endoftext|>\n\nUSER: And a span?\n\n\n"
            "ASSISTANT: Sure." + END)
    assert convert_glaive_v2(glaive(chat, GLAIVE_NO_TOOLS_SYSTEM), "11267") == (None, EMPTY_TURN)
    chat = "USER: Hi.\n\n\nASSISTANT: Hello." + END + "USER: \n\n\nASSISTANT: Yes?" + END
    assert convert_glaive_v2(glaive(chat, GLAIVE_NO_TOOLS_SYSTEM), "84134") == (None, EMPTY_TURN)
    assert convert_hermes(hermes(msg("human", "Hi"), msg("gpt", ""), tools="null"), "x") == (None, EMPTY_TURN)
    assert convert_glaive_zh(zh(msg("human", "你好"), msg("gpt", ""), tools="[]"), "x") == (None, EMPTY_TURN)
    assert convert_deepexi(deepexi(ticket_call([{}]), user='""'), "0") == (None, EMPTY_TURN)
    # an empty assistant segment right before a call segment is the same turn, so it stays
    chat = ("USER: Password?\n\n\nASSISTANT: " + END + "ASSISTANT: <functioncall> {\"name\": "
            "\"generate_password\", \"arguments\": '{\"length\": 8}'}" + END
            + "FUNCTION RESPONSE: {\"password\": \"p\"}\n\n\nASSISTANT: Here: p." + END)
    sample, reason = convert_glaive_v2(glaive(chat), "0")
    assert reason is None and names_of(sample["messages"][1]) == ["generate_password"]
    # a trailing empty user turn is cut like any trailing user turn
    sample, reason = convert_toolace(toolace(msg("user", "x"), msg("assistant", "y"), msg("user", "")), "0")
    assert reason is None and len(sample["messages"]) == 2


def test_check_sample_reports_every_broken_invariant():
    good, _ = convert_glaive_zh(ZH_NORMAL, "0")
    assert check_sample(good) == []

    def broken(**changes) -> dict:
        return {**json.loads(json.dumps(good)), **changes}

    empty_assistant = broken()
    empty_assistant["messages"][3]["content"] = ""
    assert any("neither text nor a tool call" in e for e in check_sample(empty_assistant))
    empty_user = broken()
    empty_user["messages"][0]["content"] = " "
    assert any("user message is empty" in e for e in check_sample(empty_user))
    duplicate = broken()
    duplicate["messages"][5]["tool_calls"][0]["id"] = "call_0"
    assert any("not unique" in e for e in check_sample(duplicate))
    unanswered = broken()
    unanswered["messages"][2]["tool_call_id"] = "call_9"
    errors = check_sample(unanswered)
    assert any("answers no call" in e for e in errors) and any("not answered" in e for e in errors)
    not_object = broken()
    not_object["messages"][1]["tool_calls"][0]["function"]["arguments"] = "[1, 2]"
    assert any("JSON object string" in e for e in check_sample(not_object))
    role = broken()
    role["messages"][2]["role"] = "function"
    assert any("unknown role" in e for e in check_sample(role))
    assert any("from the user" in e for e in check_sample(broken(messages=good["messages"][1:])))
    assert any("no assistant" in e for e in check_sample(broken(messages=good["messages"][:1])))
    assert check_sample(broken(messages=[])) == ["messages must be a non-empty list"]
    assert any("function schema" in e for e in check_sample(broken(tools=[RANDOM])))
    assert any("must be a list" in e for e in check_sample(broken(tools="x")))
    assert any("strings" in e for e in check_sample(broken(source_id=7)))
    assert check_sample("nope") == ["sample must be a dict"]
    with_system = broken(messages=[{"role": "system", "content": "s"}, *good["messages"]])
    assert check_sample(with_system) == []


def test_convert_many_counts_samples_and_reasons():
    no_list = toolace(msg("user", "x"), msg("assistant", "y"), system="none")
    records = [TOOLACE_NORMAL, TOOLACE_GARBAGE, no_list, TOOLACE_PARALLEL, toolace(msg("user", "hello?")),
               toolace(msg("user", "x"), msg("assistant", ""))]
    samples, reasons = convert_many(records, convert_toolace)
    assert [s["source_id"] for s in samples] == ["0", "3"]
    assert reasons == Counter({BAD_CALL: 1, NO_TOOLS: 1, NO_ASSISTANT: 1, EMPTY_TURN: 1})
    samples, reasons = convert_many(records[:2], convert_toolace, ids=["a/0", "a/1"])
    assert [s["source_id"] for s in samples] == ["a/0"] and reasons == Counter({BAD_CALL: 1})
    assert convert_many([], convert_toolace) == ([], Counter())
    with pytest.raises(ValueError):
        convert_many(records[:2], convert_toolace, ids=["only-one"])

    def careless(record: dict, source_id: str):  # a converter stand-in that breaks the format
        return {"messages": [{"role": "assistant", "content": "x"}], "tools": [], "source": "t",
                "source_id": source_id}, None

    assert convert_many([{}], careless) == ([], Counter({INVALID_SAMPLE: 1}))


def test_has_tool_call():
    assert has_tool_call(convert_glaive_zh(ZH_NORMAL, "0")[0])
    assert not has_tool_call(convert_glaive_zh(ZH_NO_CALL, "0")[0])


def test_converted_samples_feed_the_trainer_one_step_per_assistant_message():
    tokenizer = FakeTokenizer()
    for converter, record in GOOD_RECORDS:
        sample, _ = converter(record, "0")
        steps, dropped = build_step_samples(sample, tokenizer)
        assistant = [m for m in sample["messages"] if m["role"] == "assistant"]
        assert dropped.total == 0 and len(steps) == len(assistant)
        for step, message in zip(steps, assistant, strict=True):
            for call in message.get("tool_calls") or []:
                assert call["function"]["name"] in step["completion"]
            if message["content"]:
                assert message["content"] in step["completion"]
        if sample["tools"]:
            assert sample["tools"][0]["function"]["name"] in steps[0]["prompt"]
        for m in sample["messages"]:
            if m["role"] == "tool":
                assert any(m["content"] in step["prompt"] for step in steps[1:])

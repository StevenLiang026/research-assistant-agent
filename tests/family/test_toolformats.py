"""Tool-call formats (research_agent.family.toolformats, docs/m7-家族层设计.md §4): coerce's type table;
qwen_json (its parse is research_agent.toolcalls.parse_tool_calls, its render writes calls as the Qwen3
template does); and the four formats of the third phase, qwen_xml (Qwen3.5, Granite), minicpm_xml (MiniCPM5),
glm_kv (Spark) and pythonic (LFM2.5): each renders a call as its family's chat template writes it (the strings
pinned here were taken from the real templates on 2026-10-19; tests/family/test_golden.py holds them against
the templates), parses the rendered text back with the values typed by the tool schemas, locates the name and
every value, and turns a block it cannot read into {"name": the best name, "arguments": the block's text}.
For every format a seeded round trip of 1,000 MODEL_TOOLS-shaped calls goes through render, parse and locate.
(Phase 1 compared qwen_json's block reader with agent/route.py's copy on every case of tests/test_toolcalls.py
and tests/agent/test_route.py; WP-D then made route.py call qwen_json, so those comparisons would compare the
reader with itself and were removed.)"""

from __future__ import annotations

import json
import random

import pytest
from support.family_fakes import call_block

from research_agent.family.toolformats import CallSpan, ToolFormat, coerce, parameter_schema
from research_agent.family.toolformats.glm_kv import GLM_KV
from research_agent.family.toolformats.minicpm_xml import MINICPM_XML
from research_agent.family.toolformats.pythonic import PYTHONIC
from research_agent.family.toolformats.qwen_json import QWEN_JSON
from research_agent.family.toolformats.qwen_xml import QWEN_XML
from research_agent.toolcalls import parse_tool_calls
from research_agent.tools.schemas import MODEL_TOOLS

FORMATS = {"qwen_json": QWEN_JSON, "qwen_xml": QWEN_XML, "minicpm_xml": MINICPM_XML, "glm_kv": GLM_KV,
           "pythonic": PYTHONIC}
SEARCH = {"name": "search_arxiv", "arguments": {"query": "speculative decoding", "max_results": 5,
                                                "categories": ["cs.CL"]}}
PAPER = {"name": "get_paper", "arguments": {"arxiv_id": "2609.01234"}}
NOTE = {"name": "write_note", "arguments": {"arxiv_id": "2609.01234", "problem": "a < b & c\nline two",
                                            "new_terms": ["x", "y"]}}
TYPED = {"name": "f", "arguments": {"flag": True, "none": None, "ratio": 1.5, "obj": {"a": "幻觉"}}}
RAW = {"name": "search_arxiv", "arguments": "query=KV cache"}
EMPTY = {"name": "f", "arguments": {}}

# ---- coerce ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("text", "value"), [
    ("abc", "abc"), ("\nabc\n", "abc"), ("\n\nabc\n\n", "\nabc\n"), ("", ""), ("\n", ""), (" 5 ", " 5 "),
    ('"quoted"', '"quoted"'), ("line one\nline two", "line one\nline two"),
])
def test_a_string_is_the_text_without_one_separating_newline_at_either_end(text, value):
    assert coerce(text, {"type": "string"}) == value


def test_a_value_written_without_separating_newlines_keeps_its_own():
    assert coerce("\nabc\n", {"type": "string"}, separated=False) == "\nabc\n"
    assert coerce("\n5\n", {"type": "integer"}, separated=False) == 5  # blanks around a number are ignored
    assert coerce("", {"type": "string"}, separated=False) == ""


@pytest.mark.parametrize(("text", "schema", "value"), [
    ("5", {"type": "integer"}, 5), ("\n-12\n", {"type": "integer"}, -12), (" 7 ", {"type": "integer"}, 7),
    ("5.0", {"type": "integer"}, 5), ("1e2", {"type": "integer"}, 100),  # JSON Schema's integers
    ("2.5", {"type": "number"}, 2.5), ("5", {"type": "number"}, 5), ("-0.5e-3", {"type": "number"}, -0.0005),
    ("true", {"type": "boolean"}, True), ("False", {"type": "boolean"}, False),
    ("\ntrue\n", {"type": "boolean"}, True),
    ("null", {"type": "null"}, None), ("None", {"type": "null"}, None),
    ('["cs.CL", 1]', {"type": "array"}, ["cs.CL", 1]),
    ('{"a": {"b": [1]}}', {"type": "object"}, {"a": {"b": [1]}}),
    ("4", {"type": "int"}, 4), ("1.5", {"type": "float"}, 1.5), ("True", {"type": "bool"}, True),
    ("{}", {"type": "dict"}, {}), ("[]", {"type": "list"}, []), ("text", {"type": "str"}, "text"),
])
def test_a_value_is_converted_to_the_type_its_schema_names(text, schema, value):
    got = coerce(text, schema)
    assert got == value and type(got) is type(value)


@pytest.mark.parametrize(("text", "schema"), [
    ("2.5", {"type": "integer"}), ("05", {"type": "integer"}), ("+5", {"type": "integer"}),
    ("true", {"type": "integer"}), ("abc", {"type": "number"}), ("NaN", {"type": "number"}),
    ("Infinity", {"type": "number"}), ("1e999", {"type": "number"}), ("0x10", {"type": "integer"}),
    ("٣", {"type": "integer"}),  # a digit, but not one JSON's grammar knows
    ("TRUE", {"type": "boolean"}), ("1", {"type": "boolean"}), ("nil", {"type": "null"}),
    ('{"a": 1}', {"type": "array"}), ("[1]", {"type": "object"}), ("{'a': 1}", {"type": "object"}),
    ("[1, 2", {"type": "array"}), ("[1e999]", {"type": "array"}), ("2026-10-19", {"type": "date"}),
])
def test_a_value_its_type_cannot_hold_keeps_its_text_for_the_executor_to_refuse(text, schema):
    assert coerce(text, schema) == text
    assert coerce("\n" + text + "\n", schema) == text  # the text without its separating newlines


def test_a_union_of_types_is_tried_in_its_order():
    assert coerce("null", {"type": ["integer", "null"]}) is None
    assert coerce("3", {"type": ["integer", "null"]}) == 3
    assert coerce("x", {"type": ["integer", "null"]}) == "x"
    assert coerce("3", {"type": ["integer", "string"]}) == 3
    assert coerce("3", {"type": ["string", "integer"]}) == "3"
    assert coerce("[1]", {"type": ["object", "list"]}) == [1]


@pytest.mark.parametrize("schema", [None, {}, {"description": "anything"}, {"enum": [1, "a"]}])
def test_without_a_type_json_is_tried_first_and_the_text_kept_otherwise(schema):
    assert coerce("5", schema) == 5
    assert coerce("true", schema) is True
    assert coerce("null", schema) is None
    assert coerce('{"a": [1, 2]}', schema) == {"a": [1, 2]}
    assert coerce("hello world", schema) == "hello world"
    assert coerce("\nhello\n", schema) == "hello"
    assert coerce('"quoted"', schema) == '"quoted"'  # JSON reads a string here: not a value, the text stays


def test_parameter_schema_finds_a_tools_parameter_in_either_tool_shape():
    assert parameter_schema(MODEL_TOOLS, "search_arxiv", "max_results") == {"type": "integer", "minimum": 1,
                                                                             "maximum": 5}
    bare = [{"name": "f", "parameters": {"type": "object", "properties": {"p": {"type": "boolean"}}}}]
    assert parameter_schema(bare, "f", "p") == {"type": "boolean"}
    assert parameter_schema(MODEL_TOOLS, "search_arxiv", "unknown") is None
    assert parameter_schema(MODEL_TOOLS, "no_such_tool", "query") is None
    assert parameter_schema(None, "search_arxiv", "query") is None
    assert parameter_schema([{"type": "function", "function": {"name": "g"}}], "g", "p") is None


# ---- every format is a ToolFormat ------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(FORMATS))
def test_each_format_is_a_tool_format_named_with_its_opening_tag(name):
    form = FORMATS[name]
    assert isinstance(form, ToolFormat) and form.name == name
    assert form.open_tag == {"qwen_json": "<tool_call>", "qwen_xml": "<tool_call>",
                             "minicpm_xml": '<function name="', "glm_kv": "<tool_call>",
                             "pythonic": "<|tool_call_start|>"}[name]
    assert form.render([]) == "" and form.parse("", MODEL_TOOLS) == [] and form.locate("") == []
    assert form.parse("no call here", MODEL_TOOLS) == [] and form.locate("no call here") == []


# ---- qwen_json: render, parse, the broken block ---------------------------------------------------------


def test_render_writes_calls_as_the_qwen3_template_writes_them():
    one = {"name": "search_arxiv", "arguments": {"query": "幻觉", "categories": ["cs.CL"], "max_results": 5}}
    assert QWEN_JSON.render([one]) == (
        '<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "幻觉", "categories": ["cs.CL"], '
        '"max_results": 5}}\n</tool_call>')
    raw = {"name": "search_arxiv", "arguments": "query=KV cache"}  # text arguments are printed verbatim
    two = QWEN_JSON.render([one, raw])
    assert two == call_block(one["name"], one["arguments"]) + "\n" + call_block(raw["name"], raw["arguments"])
    assert two.endswith('{"name": "search_arxiv", "arguments": query=KV cache}\n</tool_call>')
    assert QWEN_JSON.render([]) == ""


def test_parse_is_the_existing_parser_and_needs_no_tools():
    text = call_block("get_paper", {"arxiv_id": "2609.01234"})
    assert QWEN_JSON.parse(text, None) == QWEN_JSON.parse(text, MODEL_TOOLS) == parse_tool_calls(text)


def test_a_broken_block_gives_the_best_name_and_its_text_as_the_bad_json_path_does():
    broken = '{"name": "search_arxiv", "arguments": {"query": "kv cache", "max_results": 5}'
    text = f"<tool_call>\n{broken}\n</tool_call>"
    assert QWEN_JSON.parse(text, MODEL_TOOLS) == [{"name": "search_arxiv", "arguments": broken}]
    assert QWEN_JSON.locate(text) == [CallSpan(parsed=False)]


def test_locate_finds_the_name_and_the_first_character_of_each_value():
    text = call_block("get_paper", {"arxiv_id": "2609.01234", "full": True, "n": [1, 2]})
    (span,) = QWEN_JSON.locate(text)
    assert span.parsed and text[span.name[0]:span.name[1]] == "get_paper"
    assert [text[i] for i in span.values] == ["2", "t", "["]  # a string's first character is after its quote


# ---- qwen_xml (Qwen3.5-4B, granite-4.2-3b) --------------------------------------------------------------

SEARCH_XML = ("<tool_call>\n<function=search_arxiv>\n<parameter=query>\nspeculative decoding\n</parameter>\n"
              "<parameter=max_results>\n5\n</parameter>\n<parameter=categories>\n[\"cs.CL\"]\n</parameter>\n"
              "</function>\n</tool_call>")
PAPER_XML = ("<tool_call>\n<function=get_paper>\n<parameter=arxiv_id>\n2609.01234\n</parameter>\n"
             "</function>\n</tool_call>")
XML_OPEN, XML_CLOSE = "<tool_call>", "</tool_call>"


def test_qwen_xml_renders_calls_as_the_qwen3_coder_template_writes_them():
    assert QWEN_XML.render([SEARCH]) == SEARCH_XML
    assert QWEN_XML.render([PAPER, PAPER]) == PAPER_XML + "\n" + PAPER_XML  # one newline between blocks
    # scalars unquoted as Python writes them, a multi-line string on its own lines, arrays and objects JSON
    assert QWEN_XML.render([NOTE]) == (
        "<tool_call>\n<function=write_note>\n<parameter=arxiv_id>\n2609.01234\n</parameter>\n"
        "<parameter=problem>\na < b & c\nline two\n</parameter>\n<parameter=new_terms>\n[\"x\", \"y\"]\n"
        "</parameter>\n</function>\n</tool_call>")
    assert QWEN_XML.render([TYPED]) == (
        "<tool_call>\n<function=f>\n<parameter=flag>\nTrue\n</parameter>\n<parameter=none>\nNone\n"
        "</parameter>\n<parameter=ratio>\n1.5\n</parameter>\n<parameter=obj>\n{\"a\": \"幻觉\"}\n"
        "</parameter>\n</function>\n</tool_call>")
    assert QWEN_XML.render([EMPTY]) == "<tool_call>\n<function=f>\n</function>\n</tool_call>"
    # raw text (a call the executor refused) goes where the parameters go, as qwen_json prints it verbatim
    assert QWEN_XML.render([RAW]) == ("<tool_call>\n<function=search_arxiv>\nquery=KV cache\n</function>\n"
                                      "</tool_call>")


def test_qwen_xml_parses_its_render_back_with_the_values_typed_by_the_tools():
    assert QWEN_XML.parse(SEARCH_XML, MODEL_TOOLS) == [SEARCH]
    assert QWEN_XML.parse(PAPER_XML + "\n" + PAPER_XML, MODEL_TOOLS) == [PAPER, PAPER]
    assert QWEN_XML.parse(QWEN_XML.render([NOTE]), MODEL_TOOLS) == [NOTE]
    # without tools the values are JSON when they read as JSON, text otherwise
    assert QWEN_XML.parse(SEARCH_XML, None) == [SEARCH]
    plain = "<tool_call>\n<function=f>\n<parameter=x>\nhello\n</parameter>\n</function>\n</tool_call>"
    assert QWEN_XML.parse(plain, None) == [{"name": "f", "arguments": {"x": "hello"}}]
    # a string parameter keeps everything but one separating newline at either end
    text = ("<tool_call>\n<function=search_arxiv>\n<parameter=query>\n\n  two\nlines \n\n</parameter>\n"
            "</function>\n</tool_call>")
    assert QWEN_XML.parse(text, MODEL_TOOLS) == [{"name": "search_arxiv",
                                                  "arguments": {"query": "\n  two\nlines \n"}}]
    # blanks between the tags are the model's to add; prose around the block is ignored
    loose = ("Let me search.\n\n<tool_call>\n  <function=get_paper>\n  <parameter=arxiv_id>\n2609.01234\n"
             "</parameter>\n\n</function>  \n</tool_call>\nDone.")
    assert QWEN_XML.parse(loose, MODEL_TOOLS) == [PAPER]
    # an unterminated last block is a call once its function is complete (generation stopped there)
    assert QWEN_XML.parse(PAPER_XML.removesuffix("\n</tool_call>"), MODEL_TOOLS) == [PAPER]
    assert QWEN_XML.parse(PAPER_XML.removesuffix("</tool_call>") + "  \n", MODEL_TOOLS) == [PAPER]


@pytest.mark.parametrize("text", [
    "<tool_call>\n<function=get_paper>\n<parameter=arxiv_id>\n2609.01234\n</parameter>\n</tool_call>",
    "<tool_call>\n<function=get_paper>\n<parameter=arxiv_id>\n2609.01234\n</function>\n</tool_call>",
    ("<tool_call>\n<function=get_paper>\n<parameter=arxiv_id>\n2609.01234\n</parameter>\nextra\n</function>\n"
     "</tool_call>"),
    "<tool_call>\n<function=get_paper>\n</function>\ntrailing text\n</tool_call>",
    "<tool_call>\n{\"name\": \"get_paper\", \"arguments\": {}}\n</tool_call>",  # Qwen3's JSON is not this
    "<tool_call>\n<function=>\n</function>\n</tool_call>",  # an empty name
    "<tool_call>\n<function=get_paper>\n<parameter=arxiv_id>\n2609",  # cut short
], ids=["no-function-close", "no-parameter-close", "text-after-parameter", "text-after-function", "json",
        "empty-name", "cut-short"])
def test_a_qwen_xml_block_that_is_not_a_call_gives_the_best_name_and_its_text(text):
    (bad,) = QWEN_XML.parse(text, MODEL_TOOLS)
    expected = "get_paper" if "<function=get_paper>" in text else ""
    assert bad == {"name": expected, "arguments": text[len(XML_OPEN):].removesuffix(XML_CLOSE).strip()}
    assert QWEN_XML.locate(text) == [CallSpan(parsed=False)]


def test_qwen_xml_reads_on_after_a_broken_block_at_its_closing_tag():
    broken = "<tool_call>\n<function=get_paper>\n<parameter=arxiv_id>\n2609.01234\n</tool_call>"
    calls = QWEN_XML.parse(broken + "\n" + PAPER_XML, MODEL_TOOLS)
    assert calls == [{"name": "get_paper",
                      "arguments": "<function=get_paper>\n<parameter=arxiv_id>\n2609.01234"}, PAPER]
    assert [s.parsed for s in QWEN_XML.locate(broken + "\n" + PAPER_XML)] == [False, True]


def test_qwen_xml_locates_the_name_and_the_first_character_of_each_value():
    text = "prose\n" + QWEN_XML.render([SEARCH, NOTE])
    first, second = QWEN_XML.locate(text)
    assert first.parsed and text[first.name[0]:first.name[1]] == "search_arxiv"
    assert [text[i:i + 3] for i in first.values] == ["spe", "5\n<", '["c']
    assert text[second.name[0]:second.name[1]] == "write_note"
    assert [text[i:i + 3] for i in second.values] == ["260", "a <", '["x']
    empty = "<tool_call>\n<function=f>\n<parameter=x>\n\n</parameter>\n</function>\n</tool_call>"
    (span,) = QWEN_XML.locate(empty)
    assert span.values == (empty.index("\n\n</parameter>") + 1,)  # an empty value: where it would start


# ---- minicpm_xml (MiniCPM5-2B) ----------------------------------------------------------------------------

SEARCH_MINICPM = ('<function name="search_arxiv"><param name="query">speculative decoding</param>'
                  '<param name="max_results">5</param><param name="categories">[\'cs.CL\']</param>'
                  '</function>')
PAPER_MINICPM = '<function name="get_paper"><param name="arxiv_id">2609.01234</param></function>'
NOTE_MINICPM = ('<function name="write_note"><param name="arxiv_id">2609.01234</param>'
                '<param name="problem"><![CDATA[a < b & c\nline two]]></param>'
                '<param name="new_terms">[\'x\', \'y\']</param></function>')
MINICPM_OPEN, MINICPM_CLOSE = '<function name="', "</function>"


def test_minicpm_xml_renders_calls_as_the_minicpm5_template_writes_them():
    assert MINICPM_XML.render([SEARCH]) == SEARCH_MINICPM  # arrays as Python prints them ({{ param_value }})
    assert MINICPM_XML.render([PAPER, PAPER]) == PAPER_MINICPM + "\n" + PAPER_MINICPM
    assert MINICPM_XML.render([NOTE]) == NOTE_MINICPM  # <, & or a newline: CDATA
    # each of the three alone calls for CDATA, as the template tests them one by one; ">" does not
    single = {"name": "f", "arguments": {"a": "R&D only", "b": "x < y", "c": "one\ntwo", "d": "a > b"}}
    assert MINICPM_XML.render([single]) == (
        '<function name="f"><param name="a"><![CDATA[R&D only]]></param>'
        '<param name="b"><![CDATA[x < y]]></param><param name="c"><![CDATA[one\ntwo]]></param>'
        '<param name="d">a > b</param></function>')
    assert MINICPM_XML.parse(MINICPM_XML.render([single]), None) == [single]
    assert MINICPM_XML.render([TYPED]) == (
        '<function name="f"><param name="flag">True</param><param name="none">None</param>'
        '<param name="ratio">1.5</param><param name="obj">{\'a\': \'幻觉\'}</param></function>')
    assert MINICPM_XML.render([EMPTY]) == '<function name="f"></function>'
    assert MINICPM_XML.render([RAW]) == '<function name="search_arxiv">query=KV cache</function>'


def test_minicpm_xml_parses_its_render_back_reading_python_literals_where_the_template_writes_them():
    assert MINICPM_XML.parse(SEARCH_MINICPM, MODEL_TOOLS) == [SEARCH]
    assert MINICPM_XML.parse(PAPER_MINICPM + "\n" + PAPER_MINICPM, MODEL_TOOLS) == [PAPER, PAPER]
    assert MINICPM_XML.parse(NOTE_MINICPM, MODEL_TOOLS) == [NOTE]
    # a CDATA value is taken whole, newlines and all; a plain value keeps its own newlines too
    cdata = '<function name="search_arxiv"><param name="query"><![CDATA[\n a\n]]></param></function>'
    assert MINICPM_XML.parse(cdata, MODEL_TOOLS) == [{"name": "search_arxiv",
                                                      "arguments": {"query": "\n a\n"}}]
    plain = '<function name="search_arxiv"><param name="query">\nx\n</param></function>'
    assert MINICPM_XML.parse(plain, MODEL_TOOLS) == [{"name": "search_arxiv",
                                                      "arguments": {"query": "\nx\n"}}]
    # a Python-literal list or object is read for an array or object parameter, or one without a type;
    # a string parameter keeps the text; JSON is read as well
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "a": {"type": "array"}, "o": {"type": "object"}, "s": {"type": "string"},
        "u": {"type": ["array", "null"]}}}}}]
    text = ('<function name="f"><param name="a">[\'x\', 1]</param>'
            '<param name="o">{\'k\': [True, None]}</param><param name="s">[\'x\']</param>'
            '<param name="u">[1]</param><param name="n">[\'y\']</param></function>')
    assert MINICPM_XML.parse(text, tools) == [{"name": "f", "arguments": {
        "a": ["x", 1], "o": {"k": [True, None]}, "s": "['x']", "u": [1], "n": ["y"]}}]
    assert MINICPM_XML.parse('<function name="f"><param name="a">["x"]</param></function>', tools) == [
        {"name": "f", "arguments": {"a": ["x"]}}]
    assert MINICPM_XML.parse('<function name="f"><param name="a">[x]</param></function>', tools) == [
        {"name": "f", "arguments": {"a": "[x]"}}]  # neither JSON nor a Python literal: the text stays
    loose = ('Sure.\n<function name="get_paper">\n <param name="arxiv_id">2609.01234</param> </function>\n'
             'Done.')
    assert MINICPM_XML.parse(loose, MODEL_TOOLS) == [PAPER]


@pytest.mark.parametrize("text", [
    '<function name="get_paper"><param name="arxiv_id">2609.01234</param>',  # no </function>
    '<function name="get_paper"><param name="arxiv_id">2609.01234</function>',  # no </param>
    '<function name="get_paper"><param name="arxiv_id"><![CDATA[2609.01234</param></function>',  # CDATA open
    '<function name="get_paper">stray text<param name="arxiv_id">2609.01234</param></function>',
    '<function name="get_paper"><parameter name="arxiv_id">2609.01234</parameter></function>',
    '<function name="">x</function>',
    '<function name="get_paper',  # cut short
])
def test_a_minicpm_xml_block_that_is_not_a_call_gives_the_best_name_and_its_text(text):
    (bad,) = MINICPM_XML.parse(text, MODEL_TOOLS)
    expected = "get_paper" if text.startswith('<function name="get_paper"') else ""
    raw = text[len(MINICPM_OPEN):].removesuffix(MINICPM_CLOSE).strip()
    assert bad == {"name": expected, "arguments": raw}
    assert MINICPM_XML.locate(text) == [CallSpan(parsed=False)]


@pytest.mark.parametrize(("key", "value"), [
    ("a", "[{1, 2}]"), ("a", "[(1, 2)]"), ("a", "[1e999]"), ("a", "[b'x']"), ("o", "{1: 'x'}"),
    ("o", "{'k': {2}}"), ("n", "[1j]"),
])
def test_a_minicpm_xml_python_literal_that_json_cannot_hold_keeps_its_text(key, value):
    # what JSON cannot hold is no argument value (agent.models refuses it as a harness error): the text stays,
    # for the executor to refuse as invalid_args
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "a": {"type": "array"}, "o": {"type": "object"}}}}}]
    text = f'<function name="f"><param name="{key}">{value}</param></function>'
    assert MINICPM_XML.parse(text, tools) == [{"name": "f", "arguments": {key: value}}]


def test_minicpm_xml_locates_the_name_and_the_first_character_of_each_value():
    text = "prose\n" + MINICPM_XML.render([SEARCH, NOTE])
    first, second = MINICPM_XML.locate(text)
    assert first.parsed and text[first.name[0]:first.name[1]] == "search_arxiv"
    assert [text[i:i + 3] for i in first.values] == ["spe", "5</", "['c"]
    assert text[second.name[0]:second.name[1]] == "write_note"
    assert [text[i:i + 3] for i in second.values] == ["260", "a <", "['x"]  # a CDATA value: after <![CDATA[


# ---- glm_kv (Spark-X2.5-4B) ------------------------------------------------------------------------------

SEARCH_GLM = ("<tool_call>search_arxiv<arg_key>query</arg_key><arg_value>speculative decoding</arg_value>"
              "<arg_key>max_results</arg_key><arg_value>5</arg_value><arg_key>categories</arg_key>"
              "<arg_value>[\"cs.CL\"]</arg_value></tool_call>")
PAPER_GLM = "<tool_call>get_paper<arg_key>arxiv_id</arg_key><arg_value>2609.01234</arg_value></tool_call>"


def test_glm_kv_renders_calls_as_the_spark_template_writes_them():
    assert GLM_KV.render([SEARCH]) == SEARCH_GLM
    assert GLM_KV.render([PAPER, PAPER]) == PAPER_GLM + PAPER_GLM  # nothing between blocks
    assert GLM_KV.render([NOTE]) == (
        "<tool_call>write_note<arg_key>arxiv_id</arg_key><arg_value>2609.01234</arg_value><arg_key>problem"
        "</arg_key><arg_value>a < b & c\nline two</arg_value><arg_key>new_terms</arg_key>"
        "<arg_value>[\"x\", \"y\"]</arg_value></tool_call>")
    assert GLM_KV.render([TYPED]) == (  # everything but a string is JSON
        "<tool_call>f<arg_key>flag</arg_key><arg_value>true</arg_value><arg_key>none</arg_key>"
        "<arg_value>null</arg_value><arg_key>ratio</arg_key><arg_value>1.5</arg_value><arg_key>obj</arg_key>"
        "<arg_value>{\"a\": \"幻觉\"}</arg_value></tool_call>")
    assert GLM_KV.render([EMPTY]) == "<tool_call>f</tool_call>"
    assert GLM_KV.render([RAW]) == "<tool_call>search_arxivquery=KV cache</tool_call>"


def test_glm_kv_parses_its_render_back_with_the_values_typed_by_the_tools():
    assert GLM_KV.parse(SEARCH_GLM, MODEL_TOOLS) == [SEARCH]
    assert GLM_KV.parse(PAPER_GLM + PAPER_GLM, MODEL_TOOLS) == [PAPER, PAPER]
    assert GLM_KV.parse(GLM_KV.render([NOTE]), MODEL_TOOLS) == [NOTE]
    assert GLM_KV.parse("<tool_call>list_notes</tool_call>", MODEL_TOOLS) == [{"name": "list_notes",
                                                                               "arguments": {}}]
    # a string value keeps its own blanks and newlines; the name and keys may be surrounded by blanks
    text = ("<tool_call>\nsearch_arxiv\n<arg_key> query </arg_key>\n<arg_value> a\nb </arg_value>\n"
            "</tool_call>")
    assert GLM_KV.parse(text, MODEL_TOOLS) == [{"name": "search_arxiv", "arguments": {"query": " a\nb "}}]
    assert GLM_KV.parse("x " + PAPER_GLM + " y", MODEL_TOOLS) == [PAPER]
    # an unterminated last block is a call once its last value is complete
    assert GLM_KV.parse(PAPER_GLM.removesuffix("</tool_call>"), MODEL_TOOLS) == [PAPER]
    assert GLM_KV.parse("<tool_call>list_notes", MODEL_TOOLS) == [{"name": "list_notes", "arguments": {}}]


@pytest.mark.parametrize("text", [
    "<tool_call>get_paper<arg_key>arxiv_id</arg_key><arg_value>2609.01234</tool_call>",  # no </arg_value>
    "<tool_call>get_paper<arg_key>arxiv_id<arg_value>2609.01234</arg_value></tool_call>",  # no </arg_key>
    "<tool_call>get_paper<arg_value>2609.01234</arg_value></tool_call>",  # a value without a key
    "<tool_call>get_paper<arg_key>arxiv_id</arg_key>text<arg_value>2609.01234</arg_value></tool_call>",
    "<tool_call><arg_key>arxiv_id</arg_key><arg_value>2609.01234</arg_value></tool_call>",  # no name
    "<tool_call>get paper<arg_key>arxiv_id</arg_key><arg_value>2609.01234</arg_value></tool_call>",
    # a key is shaped as a name is: not empty, no blanks or angle brackets inside
    "<tool_call>get_paper<arg_key>arxiv id</arg_key><arg_value>2609.01234</arg_value></tool_call>",
    "<tool_call>get_paper<arg_key> </arg_key><arg_value>2609.01234</arg_value></tool_call>",
    "<tool_call>get_paper<arg_key>a>b</arg_key><arg_value>2609.01234</arg_value></tool_call>",
    "<tool_call>get_paper<arg_key>arxiv_id</arg_key><arg_value>2609",  # cut short
])
def test_a_glm_kv_block_that_is_not_a_call_gives_the_best_name_and_its_text(text):
    (bad,) = GLM_KV.parse(text, MODEL_TOOLS)
    expected = "get_paper" if text.startswith("<tool_call>get_paper<") else ""
    assert bad == {"name": expected, "arguments": text[len(XML_OPEN):].removesuffix(XML_CLOSE).strip()}
    assert GLM_KV.locate(text) == [CallSpan(parsed=False)]


def test_glm_kv_locates_the_name_and_the_first_character_of_each_value():
    text = "prose" + GLM_KV.render([SEARCH, NOTE])
    first, second = GLM_KV.locate(text)
    assert first.parsed and text[first.name[0]:first.name[1]] == "search_arxiv"
    assert [text[i:i + 3] for i in first.values] == ["spe", "5</", '["c']
    assert text[second.name[0]:second.name[1]] == "write_note"
    assert [text[i:i + 3] for i in second.values] == ["260", "a <", '["x']


# ---- pythonic (LFM2.5-2.6B) ------------------------------------------------------------------------------

SEARCH_PY = ("<|tool_call_start|>[search_arxiv(query='speculative decoding', max_results=5, "
             "categories=[\"cs.CL\"])]<|tool_call_end|>")
PAPER_PY = "<|tool_call_start|>[get_paper(arxiv_id='2609.01234')]<|tool_call_end|>"
PY_OPEN, PY_CLOSE = "<|tool_call_start|>", "<|tool_call_end|>"


def test_pythonic_renders_calls_as_the_lfm2_template_writes_them():
    assert PYTHONIC.render([SEARCH]) == SEARCH_PY
    # every call of a step goes in one list
    assert PYTHONIC.render([PAPER, PAPER]) == ("<|tool_call_start|>[get_paper(arxiv_id='2609.01234'), "
                                               "get_paper(arxiv_id='2609.01234')]<|tool_call_end|>")
    # strings single-quoted with \\, ', newline and return escaped; arrays and objects JSON; scalars as Python
    assert PYTHONIC.render([NOTE]) == (
        "<|tool_call_start|>[write_note(arxiv_id='2609.01234', problem='a < b & c\\nline two', "
        "new_terms=[\"x\", \"y\"])]<|tool_call_end|>")
    typed = {"name": "f", "arguments": {**TYPED["arguments"], "q": "it's \\ \r"}}
    assert PYTHONIC.render([typed]) == (
        "<|tool_call_start|>[f(flag=True, none=None, ratio=1.5, obj={\"a\": \"幻觉\"}, q='it\\'s \\\\ \\r')]"
        "<|tool_call_end|>")
    assert PYTHONIC.render([EMPTY]) == "<|tool_call_start|>[f()]<|tool_call_end|>"
    assert PYTHONIC.render([RAW]) == "<|tool_call_start|>[search_arxiv(query=KV cache)]<|tool_call_end|>"


def test_pythonic_parses_its_render_back_with_literal_values_and_json_where_python_has_no_literal():
    assert PYTHONIC.parse(SEARCH_PY, MODEL_TOOLS) == [SEARCH]
    assert PYTHONIC.parse(PYTHONIC.render([PAPER, SEARCH]), MODEL_TOOLS) == [PAPER, SEARCH]
    assert PYTHONIC.parse(PYTHONIC.render([NOTE]), MODEL_TOOLS) == [NOTE]
    assert PYTHONIC.parse("<|tool_call_start|>[list_notes()]<|tool_call_end|>", MODEL_TOOLS) == [
        {"name": "list_notes", "arguments": {}}]
    # the template writes JSON for arrays and objects, whose true, false and null are not Python literals:
    # such a value is read as JSON, typed by the schema when there is one
    text = ("<|tool_call_start|>[f(a=[true, null], o={\"k\": false}, n=-3, s='x\\ny', t=\"dq\")]"
            "<|tool_call_end|>")
    assert PYTHONIC.parse(text, None) == [{"name": "f", "arguments": {
        "a": [True, None], "o": {"k": False}, "n": -3, "s": "x\ny", "t": "dq"}}]
    # blanks and newlines inside the list are Python's business; two blocks are two lists of calls
    loose = ("Let me check.<|tool_call_start|>[ get_paper( arxiv_id = '2609.01234' ) ,\n "
             "search_arxiv(query='q')]<|tool_call_end|> and "
             "<|tool_call_start|>[list_notes()]<|tool_call_end|>")
    assert PYTHONIC.parse(loose, MODEL_TOOLS) == [
        PAPER, {"name": "search_arxiv", "arguments": {"query": "q"}}, {"name": "list_notes", "arguments": {}}]
    # an unterminated last block is read once its list is complete
    assert PYTHONIC.parse(PAPER_PY.removesuffix("<|tool_call_end|>"), MODEL_TOOLS) == [PAPER]


@pytest.mark.parametrize(("text", "name"), [
    ("<|tool_call_start|>[get_paper(arxiv_id='2609.01234'), ]x<|tool_call_end|>", "get_paper"),  # not a list
    ("<|tool_call_start|>get_paper(arxiv_id='2609.01234')<|tool_call_end|>", "get_paper"),  # not a list
    ("<|tool_call_start|>[get_paper('2609.01234')]<|tool_call_end|>", "get_paper"),  # a positional argument
    ("<|tool_call_start|>[get_paper(**kw)]<|tool_call_end|>", "get_paper"),
    # unpacking a literal mapping names no argument either (its keyword has no name to key the value by)
    ("<|tool_call_start|>[get_paper(**{'arxiv_id': '2609.01234'})]<|tool_call_end|>", "get_paper"),
    ("<|tool_call_start|>[get_paper]<|tool_call_end|>", ""),  # a bare identifier is no call
    ("<|tool_call_start|>[tools.get_paper(arxiv_id='x')]<|tool_call_end|>", ""),  # not a plain name
    ("<|tool_call_start|>[get_paper(arxiv_id=x)]<|tool_call_end|>", "get_paper"),  # no literal, no JSON
    ("<|tool_call_start|>[get_paper(arxiv_id=1 + 2)]<|tool_call_end|>", "get_paper"),  # never evaluated
    ("<|tool_call_start|>[get_paper(arxiv_id=__import__('os'))]<|tool_call_end|>", "get_paper"),
    ("<|tool_call_start|>[get_paper(arxiv_id='2609", "get_paper"),  # cut short
    ("<|tool_call_start|>[]<|tool_call_end|>", ""),  # no call at all
    ("<|tool_call_start|>[f(a=1) for f in x]<|tool_call_end|>", "f"),  # a comprehension is not a list
])
def test_a_pythonic_block_that_is_not_a_list_of_calls_gives_the_best_name_and_its_text(text, name):
    (bad,) = PYTHONIC.parse(text, MODEL_TOOLS)
    raw = text[len(PY_OPEN):].removesuffix(PY_CLOSE).strip()
    assert bad == {"name": name, "arguments": raw}  # the name: the first name a parenthesis follows
    assert PYTHONIC.locate(text) == [CallSpan(parsed=False)]


@pytest.mark.parametrize("value", ["{1, 2}", "(1, 2)", "b'x'", "1j", "...", "1e999", "{1: 'x'}", "[{1, 2}]",
                                   "{'k': (1,)}", "['a', -1e999]", "[true, NaN]", '{"k": Infinity}'])
def test_a_pythonic_literal_that_json_cannot_hold_makes_the_block_unreadable(value):
    # a call's arguments must be JSON (agent.models refuses anything else as a harness error), so a set, a
    # tuple, bytes, a complex number, an ellipsis, an infinite float or a key that is not a string is no
    # value, however deep it sits; nor is the JSON fallback's NaN or Infinity, which json.loads would read
    text = f"<|tool_call_start|>[get_paper(arxiv_id={value})]<|tool_call_end|>"
    (bad,) = PYTHONIC.parse(text, MODEL_TOOLS)
    assert bad == {"name": "get_paper", "arguments": text[len(PY_OPEN):].removesuffix(PY_CLOSE)}
    assert PYTHONIC.locate(text) == [CallSpan(parsed=False)]


def test_pythonic_reads_lines_as_python_does_and_every_string_form_from_its_first_character():
    # Python ends a line at \n, \r\n and \r alone, not at U+2028, a form feed or U+0085 (which str.splitlines
    # also splits at); a string may carry a prefix and three quotes
    text = ("<|tool_call_start|>[f(a='x y\x0cz\x85', b='w',\r\n c=u'v',\r d='''u''', e=r\"t\", "
            "g=\"\"\"s\"\"\")]<|tool_call_end|>")
    (span,) = PYTHONIC.locate(text)
    assert [text[i] for i in span.values] == ["x", "w", "v", "u", "t", "s"]
    assert PYTHONIC.parse(text, None) == [{"name": "f", "arguments": {
        "a": "x y\x0cz\x85", "b": "w", "c": "v", "d": "u", "e": "t", "g": "s"}}]


def test_pythonic_locates_the_name_and_the_first_character_of_each_value():
    text = "幻觉 prose\n" + PYTHONIC.render([SEARCH, NOTE])  # offsets are characters, not UTF-8 bytes
    first, second = PYTHONIC.locate(text)
    assert first.parsed and text[first.name[0]:first.name[1]] == "search_arxiv"
    assert [text[i:i + 3] for i in first.values] == ["spe", "5, ", '["c']  # a string's is after its quote
    assert text[second.name[0]:second.name[1]] == "write_note"
    assert [text[i:i + 3] for i in second.values] == ["260", "a <", '["x']
    multi = ("<|tool_call_start|>[\n  get_paper(\n    arxiv_id='幻觉'),\n  list_notes(limit=3)]"
             "<|tool_call_end|>")
    one, two = PYTHONIC.locate(multi)
    assert multi[one.name[0]:one.name[1]] == "get_paper" and multi[one.values[0]] == "幻"
    assert multi[two.name[0]:two.name[1]] == "list_notes" and multi[two.values[0]] == "3"


# ---- a round trip of 1,000 calls per format ---------------------------------------------------------------

PIECES = ["a", "Z", "q", "7", " ", "  ", "\n", "\t", '"', "\\", "/", "{", "}", "[", "]", ",", ":", "幻觉",
          "é", "😀", "<tool_call>", "</tool_call>", "\\n", "\\u0041", "KV cache", "2609.01234", "null", "'",
          "<", ">", "&", "=", "(", ")", "'", "<parameter=", "<param", "<arg_key>", "]]", "\r"]
# What no text argument can hold in a format, because the template writes it raw and the parser reads the
# value up to it; a random text holding one is drawn again.
UNREPRESENTABLE = {"qwen_json": (), "qwen_xml": ("</parameter>", "</tool_call>"),
                   "minicpm_xml": ("]]>", "</param>", "</function>"),
                   "glm_kv": ("</arg_value>", "</tool_call>"), "pythonic": ("<|tool_call_end|>",)}


def random_text(rng: random.Random, forbidden: tuple[str, ...]) -> str:
    while True:
        text = "".join(rng.choice(PIECES) for _ in range(rng.randint(0, 12)))
        if not any(piece in text for piece in forbidden):
            return text


def random_value(rng: random.Random, schema: dict, forbidden: tuple[str, ...]) -> object:
    """A value of the schema's type; enums and bounds are kept, other strings are random text."""
    if "enum" in schema:
        return rng.choice(schema["enum"])
    kind = schema["type"]
    if kind == "string":
        return random_text(rng, forbidden)
    if kind == "integer":
        return rng.randint(schema.get("minimum", -100), schema.get("maximum", 10_000))
    if kind == "array":
        return [random_value(rng, schema["items"], forbidden) for _ in range(rng.randint(0, 3))]
    raise AssertionError(f"MODEL_TOOLS has no {kind} parameters")


def random_call(rng: random.Random, forbidden: tuple[str, ...] = ()) -> dict:
    """A call of a random MODEL_TOOLS tool: the required parameters and about half of the others, in a
    random order. (Text arguments do not round-trip, by design: the template prints them verbatim, the raw
    text of a call the executor refused.)"""
    function = rng.choice(MODEL_TOOLS)["function"]
    parameters = function["parameters"]
    names = list(parameters["properties"])
    rng.shuffle(names)
    arguments = {name: random_value(rng, parameters["properties"][name], forbidden) for name in names
                 if name in parameters["required"] or rng.random() < 0.5}
    return {"name": function["name"], "arguments": arguments}


def written(form: ToolFormat, value: object) -> str:
    """How the format writes a value, for checking where locate points: at its first character (after the
    quote of a quoted string)."""
    if form is QWEN_JSON and isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)[1:-1]
    if form is PYTHONIC and isinstance(value, str):
        return value.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n").replace("\r", "\\r")
    if form is MINICPM_XML:
        return value if isinstance(value, str) else str(value)  # the template's {{ }}: Python's text
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


@pytest.mark.parametrize("name", list(FORMATS))
def test_a_thousand_random_model_tools_calls_render_parse_and_locate_back(name):
    form, forbidden = FORMATS[name], UNREPRESENTABLE[name]
    rng = random.Random(7)
    total = 0
    while total < 1000:
        calls = [random_call(rng, forbidden) for _ in range(min(rng.randint(1, 3), 1000 - total))]
        total += len(calls)
        text = form.render(calls)
        assert form.parse(text, MODEL_TOOLS) == calls, text
        spans = form.locate(text)
        assert len(spans) == len(calls) and all(span.parsed for span in spans)
        for span, call in zip(spans, calls, strict=True):
            assert text[span.name[0]:span.name[1]] == call["name"]
            assert len(span.values) == len(call["arguments"])
            for offset, value in zip(span.values, call["arguments"].values(), strict=True):
                expected = written(form, value)
                assert text[offset:offset + len(expected)] == expected, (name, value, text[offset:])
    assert total == 1000

"""Specification of module 02 (argument validation). Do not edit these tests to make them pass.

Implemented on 2026-10-13 (every test passes; the xfail marker that covered the NotImplementedError stub is
gone). The semantic checks call normalize_arxiv_id and normalize_date from module 01.
"""

import copy

import pytest

from research_agent.tools import schemas
from research_agent.validation import check_call

TODAY = "2026-10-09"
SEEN = ["2609.01234", "2609.05678"]  # ids that appeared in this turn's tool outputs, sorted
CATEGORIES_TEXT = "['cs.CL', 'cs.LG', 'cs.AI', 'cs.IR', 'cs.CV', 'stat.ML']"


# ---- acceptable calls ------------------------------------------------------------------------------------

@pytest.mark.parametrize("tool, args", [
    ("search_arxiv", {"query": "speculative decoding"}),
    ("search_arxiv", {"query": "推测解码", "categories": ["cs.CL", "cs.LG"], "date_from": "2026-09-01",
                      "date_to": "2026-09-30", "max_results": 5, "sort_by": "submittedDate"}),
    ("search_arxiv", {"query": "x" * 300, "max_results": 1}),  # the longest query and the smallest count
    ("get_paper", {"arxiv_id": "2609.01234"}),
    ("get_paper", {"arxiv_id": "2610.00001"}),  # not in SEEN: the user may type an id
    ("get_paper", {"arxiv_id": "1412.6980"}),  # a 4-digit sequence number is canonical before 2015
    ("fetch_fulltext", {"arxiv_id": "2609.01234"}),
    ("fetch_fulltext", {"arxiv_id": "2610.00001", "section": "method", "max_chars": 800}),
    ("fetch_fulltext", {"arxiv_id": "2609.01234", "section": "all", "max_chars": 1500}),
    ("search_notes", {"query": "推测解码"}),
    ("search_notes", {"query": "speculative decoding", "tags": ["llm", "decoding"], "top_k": 5}),
    ("write_note", {"arxiv_id": "2609.01234"}),  # in SEEN
    ("write_note", {"arxiv_id": "2609.05678", "mode": "draft", "problem": "p", "method": "m",
                    "evidence": "e", "limitations": "l", "new_terms": ["draft model"],
                    "open_terms": ["tree attention"]}),
    ("list_notes", {}),
    ("list_notes", {"tag": "llm", "since": "2026-09-01", "limit": 20}),
    ("list_notes", {"since": TODAY}),  # today itself is not in the future
    ("ask_clarification", {"question": "你想看哪个月的论文?"}),
    ("ask_clarification", {"question": "要几篇?", "missing": ["count", "topic"]}),
])
def test_acceptable_calls_give_an_empty_list(tool, args):
    assert check_call(tool, args, SEEN, TODAY) == []


def test_the_answer_is_a_list():
    assert isinstance(check_call("get_paper", {"arxiv_id": "2609.01234"}, SEEN, TODAY), list)
    assert isinstance(check_call("get_paper", {}, SEEN, TODAY), list)


# ---- unknown tool ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("tool", ["search_arxive", "SEARCH_ARXIV", "get_papers", "write note"])
def test_unknown_tool_is_the_only_message(tool):
    assert check_call(tool, {"query": "x"}, SEEN, TODAY) == [f"tool: unknown tool {tool}"]


def test_every_tool_in_the_contract_is_known():
    for tool in schemas.TOOL_NAMES:
        assert check_call(tool, {}, SEEN, TODAY) != [f"tool: unknown tool {tool}"]


# ---- shape: one message per schema error, "<parameter or arguments>: <message>" -------------------------

@pytest.mark.parametrize("tool, args, expected", [
    ("search_arxiv", {}, ["arguments: 'query' is a required property"]),
    ("get_paper", {}, ["arguments: 'arxiv_id' is a required property"]),
    ("ask_clarification", {"missing": ["topic"]}, ["arguments: 'question' is a required property"]),
    ("search_arxiv", {"query": 5}, ["query: 5 is not of type 'string'"]),
    ("search_arxiv", {"query": "x", "max_results": "5"}, ["max_results: '5' is not of type 'integer'"]),
    ("search_arxiv", {"query": "x", "categories": "cs.CL"}, ["categories: 'cs.CL' is not of type 'array'"]),
    ("search_arxiv", {"query": "x", "date_from": 20260901}, ["date_from: 20260901 is not of type 'string'"]),
    ("get_paper", {"arxiv_id": 2609.01234}, ["arxiv_id: 2609.01234 is not of type 'string'"]),
    ("write_note", {"arxiv_id": "2609.01234", "new_terms": "draft model"},
     ["new_terms: 'draft model' is not of type 'array'"]),
    ("search_arxiv", {"query": "x", "sort_by": "date"},
     ["sort_by: 'date' is not one of ['relevance', 'submittedDate']"]),
    ("search_arxiv", {"query": "x", "categories": ["cs.XX"]},
     [f"categories: 'cs.XX' is not one of {CATEGORIES_TEXT}"]),
    ("fetch_fulltext", {"arxiv_id": "2609.01234", "section": "abstract"},
     ["section: 'abstract' is not one of ['introduction', 'method', 'experiments', 'conclusion', 'all']"]),
    ("write_note", {"arxiv_id": "2609.01234", "mode": "full"},
     ["mode: 'full' is not one of ['scaffold', 'draft']"]),
    ("ask_clarification", {"question": "q", "missing": ["author"]},
     ["missing: 'author' is not one of ['topic', 'paper_id', 'date_range', 'count', 'tags']"]),
    ("search_arxiv", {"query": "x", "max_results": 0}, ["max_results: 0 is less than the minimum of 1"]),
    ("fetch_fulltext", {"arxiv_id": "2609.01234", "max_chars": 100},
     ["max_chars: 100 is less than the minimum of 200"]),
    ("search_notes", {"query": "x", "top_k": 0}, ["top_k: 0 is less than the minimum of 1"]),
    ("list_notes", {"limit": 0}, ["limit: 0 is less than the minimum of 1"]),
    ("search_arxiv", {"query": "x", "max_results": 6}, ["max_results: 6 is greater than the maximum of 5"]),
    ("fetch_fulltext", {"arxiv_id": "2609.01234", "max_chars": 2000},
     ["max_chars: 2000 is greater than the maximum of 1500"]),
    ("search_notes", {"query": "x", "top_k": 6}, ["top_k: 6 is greater than the maximum of 5"]),
    ("list_notes", {"limit": 21}, ["limit: 21 is greater than the maximum of 20"]),
    ("search_arxiv", {"query": ""}, ["query: '' should be non-empty"]),
    ("search_notes", {"query": ""}, ["query: '' should be non-empty"]),
    ("ask_clarification", {"question": ""}, ["question: '' should be non-empty"]),
    ("search_arxiv", {"query": "x" * 301}, [f"query: '{'x' * 301}' is too long"]),
    ("get_paper", {"arxiv_id": "2609.01234", "version": 2},
     ["arguments: Additional properties are not allowed ('version' was unexpected)"]),
    ("list_notes", {"query": "x"},
     ["arguments: Additional properties are not allowed ('query' was unexpected)"]),
    ("search_arxiv", {"query": "x", "categories": ["cs.CL", "cs.CL"]},
     ["categories: ['cs.CL', 'cs.CL'] has non-unique elements"]),
])
def test_one_shape_error_gives_one_message(tool, args, expected):
    assert check_call(tool, args, SEEN, TODAY) == expected


def test_shape_errors_are_sorted_one_per_problem():
    args = {"query": "", "max_results": 0, "foo": 1, "categories": ["cs.CL", "cs.XX", "cs.CL"]}
    assert check_call("search_arxiv", args, SEEN, TODAY) == [
        "arguments: Additional properties are not allowed ('foo' was unexpected)",
        f"categories: 'cs.XX' is not one of {CATEGORIES_TEXT}",
        "categories: ['cs.CL', 'cs.XX', 'cs.CL'] has non-unique elements",
        "max_results: 0 is less than the minimum of 1",
        "query: '' should be non-empty",
    ]


@pytest.mark.parametrize("tool, args, expected", [
    # the date is not canonical, but the shape error is the only message
    ("search_arxiv", {"query": "x", "date_from": "2026/09/01", "max_results": 0},
     ["max_results: 0 is less than the minimum of 1"]),
    # the id is not canonical and the extra key is a shape error: only the shape error is reported
    ("get_paper", {"arxiv_id": "arXiv:2609.01234v2", "version": 2},
     ["arguments: Additional properties are not allowed ('version' was unexpected)"]),
    # a write_note id of the wrong type never reaches the seen-ids rule
    ("write_note", {"arxiv_id": 2610.00001}, ["arxiv_id: 2610.00001 is not of type 'string'"]),
])
def test_a_malformed_call_gets_only_its_shape_errors(tool, args, expected):
    assert check_call(tool, args, SEEN, TODAY) == expected


# ---- meaning: canonical ids ------------------------------------------------------------------------------

@pytest.mark.parametrize("tool, raw", [
    ("get_paper", "arXiv:2609.01234v2"),
    ("get_paper", "https://arxiv.org/abs/2609.01234"),
    ("get_paper", " 2609.01234"),
    ("get_paper", "2613.00001"),  # month 13: not an identifier at all
    ("get_paper", "speculative decoding"),
    ("get_paper", ""),
    ("fetch_fulltext", "2609.01234v1"),
    ("fetch_fulltext", "２６０９．０１２３４"),  # full-width digits
    ("write_note", "arXiv:2609.01234"),  # canonical form in SEEN, but the call must use it
])
def test_an_id_must_be_in_canonical_form(tool, raw):
    assert check_call(tool, {"arxiv_id": raw}, SEEN, TODAY) == [
        f"arxiv_id: expected the canonical form YYMM.NNNNN, got {raw}"]


# ---- meaning: write_note needs an id the model has actually seen ----------------------------------------

@pytest.mark.parametrize("seen_ids, arxiv_id", [
    (SEEN, "2610.00001"),
    ([], "2609.01234"),
    (["2609.05678"], "2609.01234"),
])
def test_write_note_rejects_an_id_that_did_not_appear(seen_ids, arxiv_id):
    assert check_call("write_note", {"arxiv_id": arxiv_id, "mode": "scaffold"}, seen_ids, TODAY) == [
        f"arxiv_id: {arxiv_id} did not appear in any tool output this turn"]


@pytest.mark.parametrize("tool", ["get_paper", "fetch_fulltext"])
def test_reading_tools_accept_an_id_nobody_has_seen(tool):
    assert check_call(tool, {"arxiv_id": "2610.00001"}, [], TODAY) == []


# ---- meaning: dates --------------------------------------------------------------------------------------

@pytest.mark.parametrize("tool, name, raw", [
    ("search_arxiv", "date_from", "2026/09/01"),
    ("search_arxiv", "date_from", "2026-9-1"),
    ("search_arxiv", "date_to", "20260930"),
    ("search_arxiv", "date_to", "2026-02-29"),  # 2026 is not a leap year
    ("list_notes", "since", "2026年9月1日"),
    ("list_notes", "since", "2023-02-29"),
    ("list_notes", "since", "yesterday"),
])
def test_a_date_must_be_in_canonical_form(tool, name, raw):
    args = {"query": "x", name: raw} if tool == "search_arxiv" else {name: raw}
    expected = f"{name}: expected the canonical form YYYY-MM-DD, got {raw}"
    assert check_call(tool, args, SEEN, TODAY) == [expected]


@pytest.mark.parametrize("tool, name, value", [
    ("search_arxiv", "date_from", "2026-10-10"),  # the day after TODAY
    ("search_arxiv", "date_to", "2027-01-01"),
    ("search_arxiv", "date_to", "2028-02-29"),  # a real leap day, but in the future
    ("list_notes", "since", "2026-10-10"),
])
def test_a_date_after_today_is_rejected(tool, name, value):
    args = {"query": "x", name: value} if tool == "search_arxiv" else {name: value}
    assert check_call(tool, args, SEEN, TODAY) == [f"{name}: {value} is after today {TODAY}"]


@pytest.mark.parametrize("name, value", [
    ("date_from", "2024-02-29"),  # leap day in the past
    ("date_from", "2000-02-29"),
    ("date_to", TODAY),
])
def test_past_dates_and_today_are_accepted(name, value):
    assert check_call("search_arxiv", {"query": "x", name: value}, SEEN, TODAY) == []


def test_today_is_what_the_caller_says():
    args = {"query": "x", "date_to": "2026-10-10"}
    assert check_call("search_arxiv", args, SEEN, "2026-10-10") == []
    assert check_call("search_arxiv", args, SEEN, "2026-10-09") == [
        "date_to: 2026-10-10 is after today 2026-10-09"]


def test_date_from_must_not_be_after_date_to():
    args = {"query": "x", "date_from": "2026-09-30", "date_to": "2026-09-01"}
    assert check_call("search_arxiv", args, SEEN, TODAY) == [
        "date_from: 2026-09-30 is after date_to 2026-09-01"]
    same_day = {"query": "x", "date_from": "2026-09-01", "date_to": "2026-09-01"}
    assert check_call("search_arxiv", same_day, SEEN, TODAY) == []


def test_the_order_rule_needs_both_dates_in_canonical_form():
    # date_from is not canonical: that is the only message, the order is not judged on a bad date
    args = {"query": "x", "date_from": "2026/09/30", "date_to": "2026-09-01"}
    assert check_call("search_arxiv", args, SEEN, TODAY) == [
        "date_from: expected the canonical form YYYY-MM-DD, got 2026/09/30"]


def test_semantic_messages_are_sorted_one_per_problem():
    args = {"query": "x", "date_from": "2027-01-01", "date_to": "2026-12-01"}
    assert check_call("search_arxiv", args, SEEN, TODAY) == [
        "date_from: 2027-01-01 is after date_to 2026-12-01",
        "date_from: 2027-01-01 is after today 2026-10-09",
        "date_to: 2026-12-01 is after today 2026-10-09",
    ]


# ---- purity ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("tool, args", [
    ("search_arxiv", {"query": "x", "categories": ["cs.CL"], "date_from": "2026/09/01"}),
    ("search_arxiv", {"query": "", "max_results": 0, "foo": 1}),
    ("write_note", {"arxiv_id": "2610.00001", "new_terms": ["a"]}),
    ("get_paper", {"arxiv_id": "2609.01234"}),
])
def test_args_are_not_changed(tool, args):
    before = copy.deepcopy(args)
    check_call(tool, args, SEEN, TODAY)
    assert args == before


def test_same_inputs_same_answer():
    args = {"query": "", "max_results": 0, "foo": 1, "date_from": "2027-01-01"}
    first = check_call("search_arxiv", args, SEEN, TODAY)
    assert check_call("search_arxiv", args, SEEN, TODAY) == first
    assert first == sorted(first)
    semantic = {"query": "x", "date_from": "2027-01-01", "date_to": "2026/12/01"}
    twice = [check_call("search_arxiv", semantic, SEEN, TODAY) for _ in range(2)]
    assert twice[0] == twice[1]


def test_seen_ids_are_not_changed():
    seen = ["2609.05678", "2609.01234"]
    check_call("write_note", {"arxiv_id": "2610.00001"}, seen, TODAY)
    assert seen == ["2609.05678", "2609.01234"]

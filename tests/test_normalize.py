"""Specification of module 01 (text normalisation). Do not edit these tests to make them pass.

Implemented on 2026-10-13 (every test passes; the xfail marker that covered the NotImplementedError stubs is
gone). A change to normalize.py must keep all of these green.
"""

import pytest

from research_agent.normalize import (
    days_in_month,
    fold_hyphens,
    is_leap_year,
    normalize_arxiv_id,
    normalize_date,
    normalize_text,
    query_contains_all,
)

# ---- normalize_text --------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("  Speculative   decoding  ", "Speculative decoding"),
    ("a\tb\nc", "a b c"),
    ("Ｓｐｅｃｕｌａｔｉｖｅ　ｄｅｃｏｄｉｎｇ", "Speculative decoding"),  # full-width letters and space
    ("ﬁne-tuning", "fine-tuning"),  # the "fi" ligature is one code point
    ("推测  解码", "推测 解码"),
    ("LoRA", "LoRA"),  # letter case is kept
    ("", ""),
    (" \t\n ", ""),
])
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_normalize_text_is_idempotent():
    # "¨x": NFKC turns the diaeresis into a space plus a combining mark, so the order of the two steps matters
    for raw in ["  Ａ　b  ", "ﬁ  x", "推测\t解码", "¨x"]:
        once = normalize_text(raw)
        assert normalize_text(once) == once


# ---- normalize_arxiv_id ----------------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    "2610.02002",
    "  2610.02002 ",
    "arXiv:2610.02002v2",
    "ARXIV:2610.02002",
    "https://arxiv.org/abs/2610.02002v1",
    "http://arxiv.org/pdf/2610.02002v3",
    "https://arxiv.org/pdf/2610.02002v3.pdf",
    "arxiv.org/html/2610.02002v1",
    "https://www.arxiv.org/abs/2610.02002",
    "２６１０．０２００２",  # full-width digits and full stop
])
def test_arxiv_id_surface_forms_collapse_to_one(raw):
    assert normalize_arxiv_id(raw) == "2610.02002"


@pytest.mark.parametrize("raw, expected", [
    ("1412.6980", "1412.6980"),  # December 2014: 4-digit sequence numbers
    ("1501.00001", "1501.00001"),  # January 2015: 5 digits from here on
    ("0704.0001", "0704.0001"),  # the first identifier of this scheme
])
def test_arxiv_id_sequence_length_follows_the_date(raw, expected):
    assert normalize_arxiv_id(raw) == expected


@pytest.mark.parametrize("raw", [
    "1412.69801",  # 5 digits before 2015
    "1501.0001",  # 4 digits from 2015 on
    "2613.00001",  # month 13
    "2600.00001",  # month 00
    "0703.0001",  # before this scheme started
    "cs/0112017",  # old-style identifier
    "2610.0200a",
    "2610.02002v",  # version marker without a number
    "٢٦١٠.٠٢٠٠٢",  # Arabic-Indic digits: digits, but not ASCII digits
    "",
    "speculative decoding",
])
def test_arxiv_id_invalid_inputs_give_none(raw):
    assert normalize_arxiv_id(raw) is None


def test_arxiv_id_is_idempotent():
    once = normalize_arxiv_id("arXiv:2610.02002v2")
    assert normalize_arxiv_id(once) == once


# ---- is_leap_year and days_in_month ----------------------------------------------------------------------

@pytest.mark.parametrize("year, expected", [
    (2024, True), (2026, False), (1900, False), (2000, True), (2100, False), (2400, True),
])
def test_is_leap_year(year, expected):
    assert is_leap_year(year) is expected


@pytest.mark.parametrize("year, month, expected", [
    (2026, 1, 31), (2026, 2, 28), (2024, 2, 29), (1900, 2, 28), (2026, 4, 30), (2026, 9, 30), (2026, 12, 31),
])
def test_days_in_month(year, month, expected):
    assert days_in_month(year, month) == expected


# ---- normalize_date --------------------------------------------------------------------------------------

@pytest.mark.parametrize("raw", [
    "2026-09-01",
    "2026-9-1",
    "2026/09/01",
    "2026.9.1",
    "2026年9月1日",
    "20260901",
    "２０２６－０９－０１",  # full-width digits and hyphens
    "  2026-09-01\n",
])
def test_date_surface_forms_collapse_to_iso(raw):
    assert normalize_date(raw) == "2026-09-01"


@pytest.mark.parametrize("raw, expected", [
    ("2024-02-29", "2024-02-29"),  # leap year
    ("2000-02-29", "2000-02-29"),  # century divisible by 400
    ("2026-12-31", "2026-12-31"),
])
def test_date_edge_cases_that_exist(raw, expected):
    assert normalize_date(raw) == expected


@pytest.mark.parametrize("raw", [
    "2026-02-29",  # 2026 is not a leap year
    "1900-02-29",  # century not divisible by 400
    "2026-04-31",
    "2026-13-01",
    "2026-00-10",
    "2026-09-00",
    "2026-09",
    "26-09-01",  # two-digit year
    "2026-09-01T10:00",  # a time of day is not a date
    "2026-09-01-02",
    "",
    "yesterday",
])
def test_date_invalid_inputs_give_none(raw):
    assert normalize_date(raw) is None


def test_date_is_idempotent():
    once = normalize_date("2026年9月1日")
    assert normalize_date(once) == once


# ---- query_contains_all ----------------------------------------------------------------------------------

@pytest.mark.parametrize("query, required, expected", [
    ("Speculative Decoding 2026", ["speculative", "decoding"], True),
    ("推测解码 综述", ["推测解码"], True),
    ("speculative sampling", ["speculative", "decoding"], False),
    ("ＬｏＲＡ fine-tuning", ["lora"], True),
    ("long   context", ["long  context"], True),  # whitespace normalised on both sides
    ("Straße", ["STRASSE"], True),  # casefold() maps ß to ss; lower() does not
    ("decoder", ["code"], True),  # substring semantics, by design: d-e-[c-o-d-e]-r
    ("decoding", ["code"], False),  # c-o-d is followed by i, so "code" is not a substring
    ("anything", [], True),
])
def test_query_contains_all(query, required, expected):
    assert query_contains_all(query, required) is expected


# ---- fold_hyphens (M7 WP1, the normalized scorer's S2) ---------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("mixture-of-experts", "mixture of experts"),  # U+002D hyphen-minus
    ("test\N{HYPHEN}time scaling", "test time scaling"),  # U+2010
    ("test\N{NON-BREAKING HYPHEN}time scaling", "test time scaling"),  # U+2011 (NFKC makes it U+2010)
    ("test\N{FIGURE DASH}time scaling", "test time scaling"),  # U+2012
    ("test\N{EN DASH}time scaling", "test time scaling"),  # U+2013
    ("ｍｉｘｔｕｒｅ\N{FULLWIDTH HYPHEN-MINUS}of\N{SMALL HYPHEN-MINUS}experts",  # folded after NFKC
     "mixture of experts"),
    ("LLM - as - a - judge", "LLM as a judge"),  # the spaces around a hyphen collapse into one
    ("  -leading and trailing-  ", "leading and trailing"),
    ("测试时-扩展", "测试时 扩展"),
    ("Mixture-of-Experts", "Mixture of Experts"),  # letter case is kept
    ("mixture\N{EM DASH}of\N{EM DASH}experts", "mixture\N{EM DASH}of\N{EM DASH}experts"),  # no hyphen here
    ("1 \N{MINUS SIGN} p", "1 \N{MINUS SIGN} p"),  # neither
    ("", ""),
    ("-", ""),
])
def test_fold_hyphens_reads_every_hyphen_as_a_space_after_nfkc(raw, expected):
    assert fold_hyphens(raw) == expected


def test_fold_hyphens_is_normalize_text_where_there_is_no_hyphen_and_is_idempotent():
    for raw in ["  Ｓｐｅｃｕｌａｔｉｖｅ　decoding\n", "ﬁne tuning", "推测\t解码", "¨x", "LoRA"]:
        assert fold_hyphens(raw) == normalize_text(raw)
    for raw in ["mixture-of-experts", "a-\N{COMBINING DIAERESIS}b", "¨-x",
                "x\N{NON-BREAKING HYPHEN}-\N{EN DASH}y", " - ", "ﬁne-tuning"]:
        once = fold_hyphens(raw)
        assert fold_hyphens(once) == once and normalize_text(once) == once

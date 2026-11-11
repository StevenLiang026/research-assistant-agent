"""Offline tests of research_agent.arxiv: build_search_url, parse_atom, extract_section on inline fixtures."""

from __future__ import annotations

import pytest

from research_agent.arxiv import SECTION_KEYWORDS, build_search_url, extract_section, parse_atom
from research_agent.net.fetch import canonical_url
from research_agent.tools import schemas

TODAY = "2026-10-13"
BASE = "https://export.arxiv.org/api/query?search_query="
TAIL = "&start=0&max_results=5&sortBy=relevance&sortOrder=descending"


def search(**args: object) -> dict:
    """search_arxiv args with the schema defaults applied, as the adapter passes them."""
    return {**schemas.DEFAULTS["search_arxiv"], **args}


# ---------------------------------------------------------------------------------------------------------
# build_search_url
# ---------------------------------------------------------------------------------------------------------

def test_a_plain_query_searches_every_word_in_all_fields() -> None:
    url = build_search_url(search(query="large language models"), TODAY)
    assert url == BASE + "all:large+all:language+all:models" + TAIL


def test_a_quoted_phrase_keeps_its_quotes_as_percent_22_and_operators_stay_bare() -> None:
    url = build_search_url(search(query='"speculative decoding" OR draft'), TODAY)
    assert url == BASE + "all:%22speculative+decoding%22+OR+all:draft" + TAIL


def test_a_query_with_a_field_prefix_goes_in_as_written() -> None:
    url = build_search_url(search(query='ti:"mixture of experts"  AND cat:cs.CL'), TODAY)
    assert url == BASE + "ti:%22mixture+of+experts%22+AND+cat:cs.CL" + TAIL


def test_parentheses_in_the_query_stay_unescaped_and_hug_their_contents() -> None:
    url = build_search_url(search(query="(rag OR retrieval) AND agents"), TODAY)
    assert url == BASE + "(all:rag+OR+all:retrieval)+AND+all:agents" + TAIL


def test_categories_become_an_or_group_anded_with_the_query() -> None:
    url = build_search_url(search(query="rag", categories=["cs.CL", "cs.IR"]), TODAY)
    assert url == BASE + "(all:rag)+AND+(cat:cs.CL+OR+cat:cs.IR)" + TAIL


def test_date_from_alone_is_bounded_by_today() -> None:
    url = build_search_url(search(query="rag", date_from="2026-01-01"), TODAY)
    assert url == BASE + "(all:rag)+AND+submittedDate:[202601010000+TO+202610132359]" + TAIL


def test_date_to_alone_is_bounded_by_the_first_day_of_year_one() -> None:
    url = build_search_url(search(query="rag", date_to="2025-12-31"), TODAY)
    assert url == BASE + "(all:rag)+AND+submittedDate:[000101010000+TO+202512312359]" + TAIL


def test_both_dates_categories_max_results_and_submitted_date_sort_combine() -> None:
    args = search(query="rag", categories=["cs.CL"], date_from="2026-09-01", date_to="2026-09-30",
                  max_results=3, sort_by="submittedDate")
    assert build_search_url(args, TODAY) == (
        BASE + "(all:rag)+AND+(cat:cs.CL)+AND+submittedDate:[202609010000+TO+202609302359]"
        "&start=0&max_results=3&sortBy=submittedDate&sortOrder=descending")


def test_relevance_sort_maps_to_relevance() -> None:
    url = build_search_url(search(query="rag", sort_by="relevance", max_results=2), TODAY)
    assert url.endswith("&start=0&max_results=2&sortBy=relevance&sortOrder=descending")


def test_null_optional_arguments_mean_no_filter_and_the_default_sort() -> None:
    args = search(query="rag", categories=None, date_from=None, date_to=None, sort_by=None)
    assert build_search_url(args, TODAY) == BASE + "all:rag" + TAIL
    assert build_search_url(search(query="rag", categories=[]), TODAY) == BASE + "all:rag" + TAIL


def test_a_literal_plus_and_other_reserved_characters_are_percent_encoded() -> None:
    url = build_search_url(search(query="c++ & rust"), TODAY)
    assert url == BASE + "all:c%2B%2B+all:%26+all:rust" + TAIL


def test_the_same_args_give_the_same_url_and_a_stable_canonical_form() -> None:
    args = search(query='"tool use" agents', categories=["cs.AI", "cs.CL"], date_from="2026-01-01")
    first, second = build_search_url(dict(args), TODAY), build_search_url(dict(args), TODAY)
    assert first == second
    key = canonical_url(first)
    assert canonical_url(key) == key  # the canonical form is a fixed point
    assert canonical_url(first.replace("+", "%20")) == key  # "+" and "%20" are the same request


# ---------------------------------------------------------------------------------------------------------
# parse_atom
# ---------------------------------------------------------------------------------------------------------

def feed(*entries: str) -> str:
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<feed xmlns="http://www.w3.org/2005/Atom">'
            "<title>ArXiv Query</title>" + "".join(entries) + "</feed>")


ENTRY_ONE = """<entry>
  <id>http://arxiv.org/abs/2601.01234v2</id>
  <published>2026-01-05T18:00:01Z</published>
  <title>Speculative   Decoding
    at Scale</title>
  <summary>  We study
  speculative decoding.  </summary>
  <author><name>Ada Lovelace</name></author>
  <author><name>Alan Turing</name></author>
  <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
  <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
</entry>"""
ENTRY_TWO = """<entry>
  <id>http://arxiv.org/abs/2512.09876v11</id>
  <published>2025-12-30T00:00:00Z</published>
  <title>Tool Use</title>
  <summary>Agents call tools.</summary>
  <author><name>Grace Hopper</name></author>
  <category term="cs.AI"/>
</entry>"""


def test_parse_atom_reads_two_entries_with_the_six_keys_in_feed_order() -> None:
    papers = parse_atom(feed(ENTRY_ONE, ENTRY_TWO))
    assert papers == [
        {"arxiv_id": "2601.01234", "title": "Speculative Decoding at Scale",
         "authors": ["Ada Lovelace", "Alan Turing"], "published": "2026-01-05",
         "abstract": "We study speculative decoding.", "categories": ["cs.CL", "cs.LG"]},
        {"arxiv_id": "2512.09876", "title": "Tool Use", "authors": ["Grace Hopper"],
         "published": "2025-12-30", "abstract": "Agents call tools.", "categories": ["cs.AI"]},
    ]
    assert all(list(paper) == schemas.PAPER_KEYS for paper in papers)


def test_parse_atom_strips_versions_and_the_arxiv_prefix_from_ids() -> None:
    entry = ENTRY_TWO.replace("http://arxiv.org/abs/2512.09876v11", "arXiv:2512.0987v3")
    assert parse_atom(feed(entry))[0]["arxiv_id"] == "2512.0987"
    assert parse_atom(feed(ENTRY_TWO))[0]["arxiv_id"] == "2512.09876"


def test_parse_atom_gives_an_empty_list_for_a_feed_without_entries() -> None:
    assert parse_atom(feed()) == []


@pytest.mark.parametrize("text", ["", "not xml at all", "<html><body>Rate exceeded</body></html>",
                                  '<feed xmlns="http://example.com/other"></feed>', feed(ENTRY_ONE)[:-20]])
def test_parse_atom_gives_none_for_text_that_is_not_an_atom_feed(text: str) -> None:
    assert parse_atom(text) is None


def test_parse_atom_gives_none_when_an_entry_has_no_parseable_id() -> None:
    without_id = ENTRY_TWO.replace("<id>http://arxiv.org/abs/2512.09876v11</id>", "")
    assert parse_atom(feed(ENTRY_ONE, without_id)) is None
    assert parse_atom(feed(ENTRY_ONE.replace("2601.01234v2", "not-an-id"))) is None


# What a live feed may hold besides new-style papers (M9, the lenient mode): a paper with an old-style id, the
# entry the arXiv API writes for an error (its id here ends in a well-formed id, so only the error rule skips
# it) and an empty entry, neither a title nor a summary.
OLD_STYLE_ENTRY = """<entry>
  <id>http://arxiv.org/abs/cs/0112017v1</id>
  <title>An Old-Style Paper</title>
  <summary>Ids before 2007 name their archive.</summary>
  <author><name>Charles Babbage</name></author>
  <category term="cs.LO"/>
</entry>"""
ERROR_ENTRY = """<entry>
  <id>http://arxiv.org/api/errors#incorrect_id_format_for_2510.99999</id>
  <title>Error</title>
  <summary>incorrect id format for 2510.99999</summary>
  <author><name>arXiv api core</name></author>
</entry>"""
EMPTY_ENTRY = """<entry>
  <id>http://arxiv.org/abs/2510.99999v1</id>
  <title></title>
  <summary> </summary>
</entry>"""


def test_lenient_parse_atom_skips_old_style_error_and_empty_entries_and_keeps_the_papers_in_order() -> None:
    text = feed(OLD_STYLE_ENTRY, ENTRY_ONE, ERROR_ENTRY, EMPTY_ENTRY, ENTRY_TWO)
    papers = parse_atom(text, lenient=True)
    assert papers == parse_atom(feed(ENTRY_ONE, ENTRY_TWO))
    assert [paper["arxiv_id"] for paper in papers] == ["2601.01234", "2512.09876"]
    assert parse_atom(text) is None  # the default is unchanged: the old-style id fails the whole feed
    assert parse_atom(text, lenient=False) is None


@pytest.mark.parametrize("entry", [
    OLD_STYLE_ENTRY,
    ENTRY_TWO.replace("2512.09876v11", "hep-th/9901001v2"),
    ENTRY_TWO.replace("<id>http://arxiv.org/abs/2512.09876v11</id>", ""),
    ERROR_ENTRY,
    ERROR_ENTRY.replace("<title>Error</title>", "<title>Unknown id</title>"),
    ENTRY_TWO.replace("<title>Tool Use</title>", "<title>Error</title>"),
    EMPTY_ENTRY,
    EMPTY_ENTRY.replace("<id>http://arxiv.org/abs/2510.99999v1</id>", ""),
    "<entry/>",
], ids=["old-style", "old-style-archive", "no-id", "error", "error-by-id", "error-by-title", "empty",
        "empty-without-id", "bare"])
def test_lenient_parse_atom_gives_an_empty_list_for_a_feed_of_entries_that_are_no_new_style_paper(
        entry: str) -> None:
    assert parse_atom(feed(entry), lenient=True) == []
    assert parse_atom(feed(entry, ENTRY_ONE), lenient=True) == parse_atom(feed(ENTRY_ONE))


def test_lenient_parse_atom_keeps_an_entry_with_a_title_or_a_summary() -> None:
    no_summary = ENTRY_TWO.replace("<summary>Agents call tools.</summary>", "")
    no_title = ENTRY_TWO.replace("<title>Tool Use</title>", "")
    papers = parse_atom(feed(no_summary, no_title), lenient=True)
    assert [(paper["title"], paper["abstract"]) for paper in papers] == [("Tool Use", ""),
                                                                          ("", "Agents call tools.")]
    assert parse_atom(feed(no_summary, no_title)) == papers  # what both modes read alike


@pytest.mark.parametrize("text", ["", "not xml at all", "<html><body>Rate exceeded</body></html>",
                                  '<feed xmlns="http://example.com/other"></feed>', feed(ENTRY_ONE)[:-20]])
def test_lenient_parse_atom_still_gives_none_for_text_that_is_not_an_atom_feed(text: str) -> None:
    assert parse_atom(text, lenient=True) is None


# ---------------------------------------------------------------------------------------------------------
# extract_section
# ---------------------------------------------------------------------------------------------------------

def ltx_section(number: str, title: str, *paragraphs: str, extra: str = "") -> str:
    body = "".join(f'<div class="ltx_para"><p class="ltx_p">{text}</p></div>' for text in paragraphs) + extra
    return (f'<section id="S{number}" class="ltx_section"><h2 class="ltx_title ltx_title_section">'
            f'<span class="ltx_tag ltx_tag_section">{number} </span>{title}</h2>{body}</section>')


CITE = ('<cite class="ltx_cite ltx_citemacro_cite"><a href="#bib.bib4" class="ltx_ref">Gu et al. (2025)</a>'
        "</cite>")
PAGE = ("<!DOCTYPE html><html><head><title>A paper</title></head><body><article class=\"ltx_document\">"
        '<div class="ltx_abstract"><h6 class="ltx_title ltx_title_abstract">Abstract</h6>'
        '<p class="ltx_p">We propose a thing.</p></div>'
        + ltx_section("1", "Introduction", f"Judges are popular {CITE}.", "Second   paragraph\n here.")
        + ltx_section("2", "Related Work", "Prior methods exist.")
        + ltx_section("3", "Method",
                      'We minimise <math alttext="x^{2}" class="ltx_Math"><mi>x</mi></math> over x.',
                      extra='<section class="ltx_subsection"><h3 class="ltx_title ltx_title_subsection">'
                            '3.1 Details</h3><div class="ltx_para"><p class="ltx_p">Step <br/>two.</p></div>'
                            "</section>")
        + ltx_section("4", "Experiments", 'We test it<span class="ltx_note ltx_role_footnote">'
                      '<sup class="ltx_note_mark">1</sup><span class="ltx_note_content">A note.</span></span>'
                      " on data.")
        + ltx_section("5", "Conclusion", "It works.")
        + '<section class="ltx_bibliography"><p class="ltx_p">Gu et al. 2025.</p></section>'
        + "</article></body></html>")
EXPECTED = {
    "introduction": "Judges are popular.\nSecond paragraph here.",
    "method": "We minimise x^{2} over x.\nStep two.",
    "experiments": "We test it on data.",
    "conclusion": "It works.",
    "all": ("We propose a thing.\nIntroduction\nJudges are popular.\nSecond paragraph here.\nRelated Work\n"
            "Prior methods exist.\nMethod\nWe minimise x^{2} over x.\nStep two.\nExperiments\n"
            "We test it on data.\nConclusion\nIt works."),
}


def test_every_section_name_in_the_schema_has_an_expectation_and_keywords() -> None:
    assert set(EXPECTED) == set(schemas.SECTIONS)
    assert set(SECTION_KEYWORDS) | {"all"} == set(schemas.SECTIONS)


@pytest.mark.parametrize("section", schemas.SECTIONS)
def test_each_section_name_finds_its_section_without_citations_footnotes_or_numbers(section: str) -> None:
    assert extract_section(PAGE, section, 1500) == EXPECTED[section]


def test_the_text_is_cut_to_max_chars() -> None:
    assert extract_section(PAGE, "introduction", 10) == "Judges are"
    assert extract_section(PAGE, "all", 25) == EXPECTED["all"][:25]


def test_a_missing_section_gives_none() -> None:
    page = PAGE.replace(">Conclusion<", ">Acknowledgments<")
    assert extract_section(page, "conclusion", 1500) is None


def test_a_page_without_ltx_sections_gives_none() -> None:
    page = '<html><body><div class="ltx_abstract"><p class="ltx_p">Only an abstract.</p></div></body></html>'
    for section in schemas.SECTIONS:
        assert extract_section(page, section, 1500) is None
    assert extract_section("", "introduction", 1500) is None


def test_heading_numbers_written_in_the_title_text_are_stripped_before_matching() -> None:
    page = PAGE.replace('<span class="ltx_tag ltx_tag_section">1 </span>Introduction', "I. Introduction")
    assert extract_section(page, "introduction", 1500) == EXPECTED["introduction"]


def test_a_first_tier_heading_wins_over_an_earlier_second_tier_heading() -> None:
    page = PAGE.replace(">Related Work<", ">Results and Discussion<")
    assert extract_section(page, "conclusion", 1500) == "It works."
    assert extract_section(page, "experiments", 1500) == "We test it on data."


def test_a_generic_heading_is_used_when_no_first_tier_heading_exists() -> None:
    page = PAGE.replace(">Method<", ">Our Framework<").replace(">Experiments<", ">Evaluation<")
    assert extract_section(page, "method", 1500) == EXPECTED["method"]
    assert extract_section(page, "experiments", 1500) == "We test it on data."


def page_of(*sections: str) -> str:
    return '<html><body><article class="ltx_document">' + "".join(sections) + "</article></body></html>"


def test_a_method_section_named_after_the_paper_is_found_after_related_work_and_background() -> None:
    page = page_of(ltx_section("1", "Introduction", "We study bias."),
                   ltx_section("2", "Related Work", "Others did it."),
                   ltx_section("3", "Background on Preference Optimisation", "DPO is a loss."),
                   ltx_section("4", "Bias-Adjusted DPO", "We adjust the loss."),
                   ltx_section("5", "Experiments", "It helps."))
    assert extract_section(page, "method", 1500) == "We adjust the loss."


def test_preliminaries_is_skipped_and_the_next_section_after_the_introduction_is_the_method() -> None:
    page = page_of(ltx_section("1", "Introduction", "We study bias."),
                   ltx_section("2", "Preliminaries", "Notation here."),
                   ltx_section("3", "Our Framework", "The framework."),
                   ltx_section("4", "Conclusion", "Done."))
    assert extract_section(page, "method", 1500) == "The framework."


def test_a_motivation_section_with_model_in_its_title_does_not_shadow_the_real_method() -> None:
    page = page_of(ltx_section("1", "Introduction", "Federated learning is common."),
                   ltx_section("2", "Motivation: A Single Federated Model Suffices, Until Group Imbalance",
                               "We compare three initializations."),
                   ltx_section("3", "Federated Learning with Group Debiasing", "We debias each group."),
                   ltx_section("4", "Experiments", "It helps."))
    assert extract_section(page, "method", 1500) == "We debias each group."


def test_a_page_without_a_method_heading_or_any_section_after_the_introduction_has_no_method() -> None:
    page = page_of(ltx_section("1", "Introduction", "Only an introduction."))
    assert extract_section(page, "method", 1500) is None


def test_the_fallback_stops_at_the_experiments_heading_when_a_paper_has_no_method_section() -> None:
    page = page_of(ltx_section("1", "Introduction", "We survey things."),
                   ltx_section("2", "Related Work", "Others did it."),
                   ltx_section("3", "Experiments", "We ran them."),
                   ltx_section("4", "Conclusion", "Done."))
    assert extract_section(page, "method", 1500) is None  # not the experiments text
    assert extract_section(page, "experiments", 1500) == "We ran them."

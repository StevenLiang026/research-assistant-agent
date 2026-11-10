"""The arXiv module: the three pure functions tools.adapters.arxiv_tools is built from (docs/modules/接口.md).

build_search_url(args, today) -> str
    The arXiv API query (API_URL) for one search_arxiv call whose schema defaults are already applied. The
    same args always give the same string, character for character, because the recorder and the replay
    match requests on net.fetch.canonical_url of it.
parse_atom(xml) -> list[dict] | None
    One dict per <entry> of an Atom feed with exactly the six schemas.PAPER_KEYS: arxiv_id bare (no
    "arXiv:" prefix, no version), title, authors (names in feed order), published (YYYY-MM-DD), abstract,
    categories (the <category term> values in feed order). [] for a feed without entries; None when the
    text is not an Atom feed or an entry has no parseable id (the adapters map None to UPSTREAM_ERROR).
extract_section(html, section, max_chars) -> str | None
    The plain text of one section of an arXiv HTML (LaTeXML) page, at most max_chars characters; None when
    the page has no top-level ltx sections or none matches the requested section.

Standard library only (urllib.parse, xml.etree.ElementTree, html.parser).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from urllib.parse import quote

from research_agent.tools import schemas  # absolute: scripts load this file by path

API_URL = "https://export.arxiv.org/api/query"

# ---------------------------------------------------------------------------------------------------------
# build_search_url
# ---------------------------------------------------------------------------------------------------------

SORT_BY = {"relevance": "relevance", "submittedDate": "submittedDate"}  # search_arxiv's sort_by -> sortBy
EARLIEST_DAY = "00010101"  # the lower bound of a window given only date_to
# Characters left unescaped in search_query: the field separator and arXiv's range and grouping brackets.
# Spaces become "+", and everything else, the double quote (%22) and a literal "+" (%2B) included, is
# percent-encoded by urllib.parse.quote.
QUERY_SAFE = ":[]()"
FIELD_PREFIXES = ("ti", "au", "abs", "co", "jr", "cat", "rn", "id", "all", "submittedDate")
_FIELD_PREFIX = re.compile(r"(?<![\w.])(?:" + "|".join(FIELD_PREFIXES) + r"):")
_QUOTED = re.compile(r'"[^"]*"?')  # a phrase; an unclosed quote runs to the end of the query
# One unit of a query without field prefixes: a phrase, a bracket, or a run of other characters.
_UNIT = re.compile(r'"[^"]*"?|[()]|[^\s()"]+')
OPERATORS = {"AND", "OR", "ANDNOT", "NOT"}  # upper-case only, as on arXiv; left without a prefix


def build_search_url(args: dict, today: str) -> str:
    """The arXiv API query URL for search_arxiv's args (schema defaults applied; an absent or null
    categories, date_from, date_to or sort_by means no filter or the default). today is YYYY-MM-DD.

    search_query is the AND of up to three groups, each wrapped in parentheses when there is more than one:
    the query, (cat:A OR cat:B) for the categories (in the order given, repeats dropped), and the window
    submittedDate:[FROM0000 TO TO2359], with EARLIEST_DAY or today standing in for a missing end. The query
    text goes in as written (whitespace collapsed) when it already uses any of FIELD_PREFIXES outside
    quotes; otherwise every word and quoted phrase gets all: (operators and brackets stay bare), so
    `"speculative decoding" OR draft` becomes `all:"speculative decoding" OR all:draft`. Parameters, in this
    order: search_query, start=0, max_results, sortBy, sortOrder=descending.
    """
    groups = [_query_group(str(args["query"]))]
    categories = list(dict.fromkeys(args.get("categories") or []))
    if categories:
        groups.append(" OR ".join(f"cat:{category}" for category in categories))
    date_from, date_to = args.get("date_from"), args.get("date_to")
    if date_from or date_to:
        start = _day(date_from) if date_from else EARLIEST_DAY
        end = _day(date_to) if date_to else _day(today)
        groups.append(f"submittedDate:[{start}0000 TO {end}2359]")
    if len(groups) > 1:
        groups = [group if group.startswith("submittedDate:") else f"({group})" for group in groups]
    search_query = " AND ".join(groups)
    max_results = int(args.get("max_results") or schemas.DEFAULTS["search_arxiv"]["max_results"])
    sort_by = SORT_BY[args.get("sort_by") or schemas.DEFAULTS["search_arxiv"]["sort_by"]]
    return (f"{API_URL}?search_query={_encode(search_query)}&start=0&max_results={max_results}"
            f"&sortBy={sort_by}&sortOrder=descending")


def _query_group(query: str) -> str:
    """The query as it goes into search_query (see build_search_url)."""
    query = " ".join(query.split())
    if _FIELD_PREFIX.search(_QUOTED.sub(" ", query)):
        return query
    out = []
    for unit in _UNIT.findall(query):
        out.append(unit if unit in OPERATORS or unit in ("(", ")") else f"all:{unit}")
    return re.sub(r"\( | \)", lambda m: m.group().strip(), " ".join(out))


def _day(date: str) -> str:
    """YYYY-MM-DD -> YYYYMMDD."""
    return str(date).replace("-", "")


def _encode(text: str) -> str:
    """Percent-encode text for the query string, keeping QUERY_SAFE and writing spaces as "+"."""
    return quote(text, safe=QUERY_SAFE).replace("%20", "+")


# ---------------------------------------------------------------------------------------------------------
# parse_atom
# ---------------------------------------------------------------------------------------------------------

ATOM = "{http://www.w3.org/2005/Atom}"
_ID = re.compile(r"(?:arxiv\.org/abs/)?(?:arXiv:)?(\d{4}\.\d{4,5})(?:v\d+)?\s*$")
_WS = re.compile(r"\s+")


def _text(element: ET.Element | None) -> str:
    """An element's text with whitespace collapsed; "" for a missing element."""
    return _WS.sub(" ", element.text or "").strip() if element is not None else ""


def parse_atom(xml: str) -> list[dict] | None:
    """The papers of an arXiv API Atom feed, in feed order (see the module docstring for the contract).

    The id is read from <id> (http://arxiv.org/abs/2401.01234v2 -> 2401.01234), published is the first 10
    characters of <published>, the abstract is <summary>; text fields have their whitespace collapsed.
    """
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    if root.tag != ATOM + "feed":
        return None
    papers = []
    for entry in root.findall(ATOM + "entry"):
        match = _ID.search(_text(entry.find(ATOM + "id")))
        if match is None:
            return None
        papers.append({
            "arxiv_id": match.group(1),
            "title": _text(entry.find(ATOM + "title")),
            "authors": [_text(author.find(ATOM + "name")) for author in entry.findall(ATOM + "author")],
            "published": _text(entry.find(ATOM + "published"))[:10],
            "abstract": _text(entry.find(ATOM + "summary")),
            "categories": [category.get("term", "") for category in entry.findall(ATOM + "category")],
        })
    return papers


# ---------------------------------------------------------------------------------------------------------
# extract_section
# ---------------------------------------------------------------------------------------------------------

# Heading words per schemas.SECTIONS name, in tiers. A heading matches a tier when one of its words
# (lower-case runs of letters, after the section number is dropped) is in it; each tier is tried over all
# top-level sections in document order before the next, so "Results and Discussion" does not shadow a later
# "Conclusion".
#
#   section       tier 1                            tier 2                                 tier 3
#   introduction  introduction                      -                                      -
#   method        method, methods, methodology,     (by position, see below)               -
#                 approach, proposed
#   experiments   experiment, experiments,          results, evaluation, evaluations,      analysis
#                 experimental                      empirical, numerical, findings,
#                                                   validation
#   conclusion    conclusion, conclusions,          discussion, summary, closing,          future
#                 concluding                        takeaways, outlook
#
# The method has one keyword tier. Many papers title their method section with their own name ("Bias-Adjusted
# DPO", "Nucleus Speculative Decoding"), and generic words such as "model" also appear in motivation sections
# ("A Single Federated Model Suffices, Until Group Imbalance"), so when no heading has a tier-1 word the
# method is found by position: the first top-level section after the one the introduction rule picks whose
# heading has none of METHOD_SKIP_WORDS (the context sections that usually sit between the introduction and
# the method). No introduction, or nothing usable after it, gives None. The tiers were measured against the
# 296 recorded HTML pages (2026-10-13). "all" is the abstract and then every top-level section, each headed
# by its title line.
SECTION_KEYWORDS: dict[str, tuple[frozenset[str], ...]] = {
    "introduction": (frozenset({"introduction"}),),
    "method": (frozenset({"method", "methods", "methodology", "approach", "proposed"}),),
    "experiments": (frozenset({"experiment", "experiments", "experimental"}),
                    frozenset({"results", "evaluation", "evaluations", "empirical", "numerical", "findings",
                               "validation"}),
                    frozenset({"analysis"})),
    "conclusion": (frozenset({"conclusion", "conclusions", "concluding"}),
                   frozenset({"discussion", "summary", "closing", "takeaways", "outlook"}),
                   frozenset({"future"})),
}
# Heading words that mark a section after the introduction as context rather than the method (positional
# fallback of "method" only).
METHOD_SKIP_WORDS = frozenset({"related", "background", "preliminaries", "preliminary", "notation", "problem",
                               "setup", "overview", "motivation", "prior"})
# The positional fallback stops at the first experiments or conclusion heading: a paper that reaches them
# without a method heading has no method section, and None is the honest answer (6 of the 296 recorded pages
# on 2026-10-13, where the fallback used to return the experiments text).
_METHOD_STOP_WORDS = frozenset().union(*SECTION_KEYWORDS["experiments"], *SECTION_KEYWORDS["conclusion"])
ALL = "all"
_HEADING_NUMBER = re.compile(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)[.:)]?\s+")
_WORD = re.compile(r"[a-z]+")
_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
         "track", "wbr"}
# Elements whose text never goes into a paragraph: citations, footnotes, number tags, figures and tables,
# scripts and styles.
_SKIP_TAGS = {"cite", "figure", "table", "script", "style"}
_SKIP_CLASSES = {"ltx_cite", "ltx_note", "ltx_tag", "ltx_figure", "ltx_table", "ltx_bibliography"}


class _Section:
    """One top-level section: its heading and its paragraphs' text."""

    def __init__(self) -> None:
        self.heading: list[str] = []
        self.paragraphs: list[str] = []


class _LatexmlParser(HTMLParser):
    """Collects the abstract's paragraphs and, per top-level <section class="ltx_section">, the section
    heading (<h2 class="ltx_title_section">) and the text of every <p class="ltx_p"> inside it.

    Each open element is kept on a stack with its role, so end tags (and the unclosed tags html.parser lets
    through) close the right thing. Math is replaced by its alttext.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.abstract: list[str] = []
        self.sections: list[_Section] = []
        self._stack: list[tuple[str, set[str]]] = []  # (tag, roles) per open element
        self._paragraph: list[str] | None = None

    def _active(self, role: str) -> bool:
        return any(role in roles for _, roles in self._stack)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        roles: set[str] = set()
        if tag in _SKIP_TAGS or classes & _SKIP_CLASSES:
            roles.add("skip")
        if tag == "section" and "ltx_section" in classes and not self._active("section"):
            roles.add("section")
            self.sections.append(_Section())
        if "ltx_abstract" in classes and not self._active("section"):
            roles.add("abstract")
        if tag == "h2" and "ltx_title_section" in classes and self._active("section"):
            roles.add("heading")
        if tag == "p" and "ltx_p" in classes and self._paragraph is None and not self._active("skip"):
            roles.add("paragraph")
            self._paragraph = []
        if tag == "math" and not self._active("skip"):
            self._data(" " + (attributes.get("alttext") or "") + " ")
            roles.add("skip")
        if tag in _VOID:
            return
        self._stack.append((tag, roles))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if not any(open_tag == tag for open_tag, _ in self._stack):
            return  # a stray end tag
        while self._stack:
            open_tag, roles = self._stack.pop()
            if "paragraph" in roles:
                self._close_paragraph()
            if open_tag == tag:
                return

    def handle_data(self, data: str) -> None:
        if not self._active("skip"):
            self._data(data)

    def _data(self, data: str) -> None:
        if self._paragraph is not None:
            self._paragraph.append(data)
        elif self._active("heading"):
            self.sections[-1].heading.append(data)

    def _close_paragraph(self) -> None:
        text = _clean("".join(self._paragraph or []))
        self._paragraph = None
        if not text:
            return
        if self._active("section"):
            self.sections[-1].paragraphs.append(text)
        elif self._active("abstract"):
            self.abstract.append(text)


def _clean(text: str) -> str:
    """Whitespace collapsed, and no space left before punctuation that a dropped citation leaves behind."""
    text = _WS.sub(" ", text).strip()
    return re.sub(r"\s+([.,;:!?)])", r"\1", re.sub(r"\(\s*\)|\[\s*\]", "", text)).strip()


def _title(section: _Section) -> str:
    """A section's heading text without its number."""
    return _HEADING_NUMBER.sub("", _clean("".join(section.heading)))


def extract_section(html: str, section: str, max_chars: int) -> str | None:
    """The text of the requested section of an arXiv LaTeXML page, cut to max_chars (see SECTION_KEYWORDS
    for how a name picks a heading). Paragraphs are joined with newlines, their tags stripped, whitespace
    collapsed, citations, footnotes, figures and tables dropped and math replaced by its LaTeX alttext.
    None when the page has no top-level ltx section, no section matches, or the match has no text.
    """
    parser = _LatexmlParser()
    parser.feed(html)
    parser.close()
    if not parser.sections:
        return None
    if section == ALL:
        parts = list(parser.abstract)
        for found in parser.sections:
            parts.extend([_title(found), *found.paragraphs] if found.paragraphs else [])
        text = "\n".join(parts)
    else:
        found = _find(parser.sections, section)
        if found is None:
            return None
        text = "\n".join(found.paragraphs)
    return text[:max_chars] if text.strip() else None


def _find(sections: list[_Section], name: str) -> _Section | None:
    """The first section whose heading matches name's first tier, else its second, and so on through its
    tiers; for "method" without a tier-1 match, the positional fallback described at SECTION_KEYWORDS, which
    gives up at the first experiments or conclusion heading (_METHOD_STOP_WORDS). None when nothing
    matches."""
    titles = [set(_WORD.findall(_title(found).lower())) for found in sections]
    for tier in SECTION_KEYWORDS.get(name, ()):
        for found, words in zip(sections, titles, strict=True):
            if words & tier:
                return found
    if name != "method":
        return None
    introduction = _find(sections, "introduction")
    if introduction is None:
        return None
    start = next(index for index, found in enumerate(sections) if found is introduction) + 1
    for found, words in zip(sections[start:], titles[start:], strict=True):
        if words & _METHOD_STOP_WORDS:  # the paper reached its experiments or conclusion: no method section
            return None
        if not words & METHOD_SKIP_WORDS:
            return found
    return None

"""The arXiv tools, built from the owner's pure functions.

Each adapter fetches a page, hands it to the owner's parser, cuts the result to the schema caps and maps
"nothing found" and failed fetches to the error codes the model sees. build_search_url, parse_atom and
extract_section are injected, so tests run with scripted stand-ins.

parse_atom's side of the contract (see ParseAtom): a list with one dict per paper, [] for a feed without
entries, or None for a feed it could not parse, which the model sees as UPSTREAM_ERROR, like a failed fetch,
with empty content ([] from search_arxiv, {} from get_paper and fetch_fulltext). Every paper carries every key
in schemas.PAPER_KEYS in the formats stated there; other keys are ignored. A paper without one of them is a
bug in parse_atom and raises a HarnessError naming the tool and the missing keys; so does a return value
that breaks the contract in another way, and so does every exception from parse_atom or extract_section once
the executor has wrapped it: the turn ends as harness_error and the model sees nothing.

A HarnessError from the fetcher (a replay miss, or an answer record mode does not record) passes through
untouched; a failed live fetch (FetchFailed) becomes RATE_LIMITED or UPSTREAM_ERROR. A paper's arxiv_id goes
into the content only when the paper exists, because the executor collects those ids as the ones the model
has seen.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol
from urllib.parse import quote

from ..agent.types import HarnessError, ToolFn, ToolOutcome
from ..net.fetch import Fetched, FetchFailed
from . import schemas

API_URL = "https://export.arxiv.org/api/query"
HTML_URL = "https://arxiv.org/html/"
ET_AL = "et al."  # follows the first MAX_AUTHORS names of a paper that has more

# Signatures of the owner's functions (first assigned to the owner in the plan, implemented by Claude Code;
# tests use scripted stand-ins).
BuildSearchUrl = Callable[[dict, str], str]  # build_search_url(args, today) -> arXiv API query URL
# parse_atom(xml) -> one dict per paper; [] when the feed has no entries, None when it cannot be parsed.
# Every paper has all of schemas.PAPER_KEYS, in the formats stated there: arxiv_id a bare id (YYMM.NNNNN, no
# "arXiv:" prefix, no version), title, authors a list of names, published the first version's date as
# YYYY-MM-DD, abstract, and categories, the arXiv category terms as a list of strings; all of them strings
# except the two lists (schemas.PAPER_LIST_KEYS). Other keys are ignored.
ParseAtom = Callable[[str], list[dict] | None]
# extract_section(html, section, max_chars) -> the section's text; None when the page has no such section
ExtractSection = Callable[[str, str, int], str | None]


class SupportsGet(Protocol):
    """Anything that fetches a page: research_agent.net.fetch.Fetcher in any mode, or a test double."""

    def get(self, url: str) -> Fetched: ...


def paper_url(arxiv_id: str) -> str:
    """The arXiv API request for a paper's metadata: what get_paper sends and the corpus recording records."""
    return f"{API_URL}?id_list={quote(arxiv_id)}"


def fulltext_url(arxiv_id: str) -> str:
    """A paper's HTML full text: what fetch_fulltext requests and the corpus recording records."""
    return HTML_URL + quote(arxiv_id)


def arxiv_tools(fetcher: SupportsGet, *, build_search_url: BuildSearchUrl, parse_atom: ParseAtom,
                extract_section: ExtractSection, today: str) -> dict[str, ToolFn]:
    """search_arxiv, get_paper and fetch_fulltext over one fetcher. Missing optional arguments take the
    schema defaults first, so a call with and without its defaults is the same request; a count reaches the
    owner's functions as the capped int the tool uses, so 3.0 asks for what 3 asks for."""

    def search_arxiv(args: dict) -> ToolOutcome:
        args = {**schemas.DEFAULTS["search_arxiv"], **args}
        shown = capped(args, "max_results", schemas.MAX_SEARCH_RESULTS)
        status, text = _fetch(fetcher, build_search_url({**args, "max_results": shown}, today))
        if status != 200:
            return ToolOutcome([], _failure(status))
        papers = parsed_papers("search_arxiv", parse_atom, text)
        if papers is None:
            return ToolOutcome([], schemas.UPSTREAM_ERROR)
        if not papers:
            return ToolOutcome([], schemas.EMPTY_RESULTS)
        return ToolOutcome([hit(paper) for paper in papers[:shown]])

    def lookup(tool: str, arxiv_id: str) -> ToolOutcome:
        """get_paper's answer for arxiv_id, on behalf of tool (named in harness errors)."""
        status, text = _fetch(fetcher, paper_url(arxiv_id))
        if status != 200:
            return ToolOutcome({}, _failure(status))
        papers = parsed_papers(tool, parse_atom, text)
        if papers is None:
            return ToolOutcome({}, schemas.UPSTREAM_ERROR)
        if not papers:
            return ToolOutcome({}, schemas.NOT_FOUND)  # no arxiv_id: an unknown id must not count as seen
        return ToolOutcome(_paper(papers[0]))

    def get_paper(args: dict) -> ToolOutcome:
        return lookup("get_paper", args["arxiv_id"])

    def fetch_fulltext(args: dict) -> ToolOutcome:
        args = {**schemas.DEFAULTS["fetch_fulltext"], **args}
        arxiv_id, section = args["arxiv_id"], args["section"]
        max_chars = capped(args, "max_chars", schemas.MAX_FULLTEXT_CHARS)
        status, html = _fetch(fetcher, fulltext_url(arxiv_id))
        if status == 404:  # no HTML version: the abstract stands in, and the error code marks the note
            paper = lookup("fetch_fulltext", arxiv_id)
            if paper.error is not None:
                return ToolOutcome({}, paper.error)
            return ToolOutcome({"arxiv_id": arxiv_id, "section": section, "source": "abstract",
                                "text": paper.content["abstract"][:max_chars]}, schemas.NO_FULLTEXT)
        if status != 200:
            return ToolOutcome({}, _failure(status))
        text = extract_section(html, section, max_chars)
        if text is None or not text.strip():
            return ToolOutcome({"arxiv_id": arxiv_id, "section": section}, schemas.NOT_FOUND)
        return ToolOutcome({"arxiv_id": arxiv_id, "section": section, "source": "html",
                            "text": text[:max_chars]})  # cut again in case the extractor overshoots

    return {"search_arxiv": search_arxiv, "get_paper": get_paper, "fetch_fulltext": fetch_fulltext}


def _fetch(fetcher: SupportsGet, url: str) -> tuple[int | None, str]:
    """Status and text of a page. A live fetch without a usable answer (FetchFailed: a transient failure that
    outlasted the retries, a redirect loop, an undecodable body) gives its last status, None when no usable
    response came back, and no text. A HarnessError (ReplayMiss, RecordFailed) is not caught."""
    try:
        page = fetcher.get(url)
    except FetchFailed as exc:
        return exc.status, ""
    return page.status, page.text


def capped(args: dict, name: str, cap: int) -> int:
    """args[name] clamped to 1..cap, so the caps hold even for a value that validation should have refused.

    An integral float counts as the int it equals: JSON Schema reads 3.0 as an integer, and a direct call may
    pass one that never went through types.parse_arguments. Any other non-integer raises a TypeError naming
    the argument, before anything is fetched.
    """
    value = args[name]
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    return max(1, min(value, cap))


def _failure(status: int | None) -> str:
    """Error code for a page that did not come back with 200."""
    return schemas.RATE_LIMITED if status == 429 else schemas.UPSTREAM_ERROR


def parsed_papers(tool: str, parse_atom: ParseAtom, xml: str) -> list[dict] | None:
    """parse_atom(xml), held to its contract (see ParseAtom); a breach raises a HarnessError naming the tool
    (or whatever caller name tool carries: the corpus index builder passes its own).

    Every paper is checked, not only those a tool shows, so a bug cannot hide behind max_results.
    """
    papers = parse_atom(xml)
    if papers is not None and not isinstance(papers, list):
        kind = type(papers).__name__
        raise HarnessError(f"tool {tool!r}: parse_atom returned a {kind}, not a list or None")
    for paper in papers or []:
        problem = paper_problem(paper)
        if problem is not None:
            raise HarnessError(f"tool {tool!r}: parse_atom returned {problem}")
    return papers


def paper_problem(paper: object) -> str | None:
    """How a paper from parse_atom breaks the contract, or None. Only the types are checked: the id and date
    patterns are for the owner's own tests."""
    if not isinstance(paper, dict):
        return f"a {type(paper).__name__} where a paper dict belongs"
    missing = [key for key in schemas.PAPER_KEYS if key not in paper]
    if missing:
        return f"a paper without {', '.join(missing)} (its keys: {', '.join(map(str, paper))})"
    for key in schemas.PAPER_LIST_KEYS:
        value = paper[key]
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            return f"{key} {value!r:.80}, not a list of strings"
    for key in schemas.PAPER_KEYS:
        if key not in schemas.PAPER_LIST_KEYS and not isinstance(paper[key], str):
            return f"{key} as {type(paper[key]).__name__}, not a string"
    return None


def hit(paper: dict) -> dict:
    """A search hit: what the model needs to choose a paper, with the abstract cut to a snippet."""
    return {"arxiv_id": paper["arxiv_id"], "title": paper["title"], "published": paper["published"],
            "snippet": paper["abstract"][:schemas.MAX_SNIPPET_CHARS]}


def _paper(paper: dict) -> dict:
    """What get_paper shows: the first MAX_AUTHORS authors, then "et al." when there are more, and the
    abstract cut to MAX_ABSTRACT_CHARS."""
    authors = paper["authors"][:schemas.MAX_AUTHORS]
    if len(paper["authors"]) > schemas.MAX_AUTHORS:
        authors.append(ET_AL)
    return {"arxiv_id": paper["arxiv_id"], "title": paper["title"], "authors": authors,
            "published": paper["published"], "abstract": paper["abstract"][:schemas.MAX_ABSTRACT_CHARS]}

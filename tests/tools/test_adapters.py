import copy
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import httpx
import pytest
from support.fakes import call, table_check

from research_agent.agent.executor import ToolExecutor
from research_agent.agent.types import HarnessError, ToolCall, ToolOutcome, ToolResult
from research_agent.net.fetch import Fetched, Fetcher, FetchFailed, RecordFailed, ReplayMiss
from research_agent.tools import schemas
from research_agent.tools.adapters import arxiv_tools

TODAY = "2026-10-04"
ID = "2609.00001"
SEARCH_URL = "https://export.arxiv.org/api/query?search_query=all:speculative"  # the fake builder's URL
PAPER_URL = f"https://export.arxiv.org/api/query?id_list={ID}"
HTML_URL = f"https://arxiv.org/html/{ID}"
HTML = "<html><h2 class='ltx_title_section'>1 Introduction</h2>...</html>"
CUT_OFF = "<feed><entry><title>Adaptive Dra"  # a feed whose download broke off


def atom_paper(n: int) -> dict:
    """A paper as the owner's parse_atom returns it: more keys than any tool shows, a 1,464-char abstract."""
    return {"arxiv_id": f"2609.{n:05d}", "title": f"Paper {n}", "authors": ["A. Author", "B. Author"],
            "published": "2026-09-15", "updated": "2026-09-20", "categories": ["cs.CL"],
            "abstract": f"Abstract of paper {n}. " + "Speculative decoding drafts tokens. " * 40}


FEEDS = {"<feed>search</feed>": [atom_paper(n) for n in range(1, 8)],
         f"<feed>{ID}</feed>": [atom_paper(1)],
         "<feed/>": [],
         CUT_OFF: None}  # parse_atom could not parse it


def parse_atom(xml: str, feeds: dict | None = None) -> list[dict] | None:
    """parse_atom stand-in: what feeds (FEEDS by default) scripts for the feed text. Any other text raises, as
    a parse_atom would that lets ElementTree's error escape instead of returning None: a bug in it."""
    feeds = FEEDS if feeds is None else feeds
    if xml not in feeds:
        raise ET.ParseError("undefined entity: line 1, column 6")
    return copy.deepcopy(feeds[xml])


@dataclass(frozen=True)
class Fail:
    """A fetch that failed after its retries: the last status, None when no response came back at all."""

    status: int | None


class Rig:
    """The three tools over a scripted fetcher (this object) and scripted owner functions; records inputs.

    pages maps a URL to (status, text), to Fail(status) or to an exception the fetcher raises. sections maps a
    section to the text extract_section returns or to an exception it raises. feeds adds to FEEDS.
    """

    def __init__(self, pages: dict, sections: dict | None = None, feeds: dict | None = None) -> None:
        self.pages, self.sections, self.feeds = pages, sections or {}, {**FEEDS, **(feeds or {})}
        self.urls, self.built, self.parsed, self.extracted = [], [], [], []
        self.tools = arxiv_tools(self, build_search_url=self.build_search_url, parse_atom=self.parse_atom,
                                 extract_section=self.extract_section, today=TODAY)

    def get(self, url: str) -> Fetched:
        self.urls.append(url)
        item = self.pages[url]
        if isinstance(item, Fail):
            raise FetchFailed(url, item.status)
        if isinstance(item, BaseException):
            raise item
        status, text = item
        return Fetched(url=url, final_url=url, status=status, body=text.encode("utf-8"),
                       content_type="text/html")

    def build_search_url(self, args: dict, today: str) -> str:
        self.built.append((args, today))
        return SEARCH_URL

    def parse_atom(self, xml: str) -> list[dict] | None:
        self.parsed.append(xml)
        return parse_atom(xml, self.feeds)

    def extract_section(self, html: str, section: str, max_chars: int) -> str | None:
        self.extracted.append((html, section, max_chars))
        text = self.sections.get(section)
        if isinstance(text, BaseException):
            raise text
        return text


ONE_FEED = "<feed>one</feed>"


def feed_rig(parsed: object) -> Rig:
    """Both API URLs answer ONE_FEED, for which parse_atom returns parsed; the paper has no HTML full text."""
    return Rig({SEARCH_URL: (200, ONE_FEED), PAPER_URL: (200, ONE_FEED), HTML_URL: (404, "Not Found")},
               feeds={ONE_FEED: parsed})


def test_the_three_arxiv_tools_by_their_contract_names():
    assert sorted(Rig({}).tools) == ["fetch_fulltext", "get_paper", "search_arxiv"]
    assert set(Rig({}).tools) <= set(schemas.TOOL_NAMES)


# search_arxiv


def test_search_shows_the_first_five_hits_with_four_keys_and_a_snippet():
    rig = Rig({SEARCH_URL: (200, "<feed>search</feed>")})
    outcome = rig.tools["search_arxiv"]({"query": "speculative decoding"})
    assert outcome.error is None
    assert outcome.content == [{"arxiv_id": p["arxiv_id"], "title": p["title"], "published": p["published"],
                                "snippet": p["abstract"][:200]} for p in FEEDS["<feed>search</feed>"][:5]]
    assert {len(hit["snippet"]) for hit in outcome.content} == {schemas.MAX_SNIPPET_CHARS}


def test_search_builds_the_url_from_the_arguments_with_defaults_and_today():
    rig = Rig({SEARCH_URL: (200, "<feed>search</feed>")})
    args = {"query": "speculative decoding", "sort_by": "submittedDate", "categories": ["cs.CL"]}
    rig.tools["search_arxiv"](args)
    assert rig.built == [({**args, "max_results": 5}, TODAY)]
    assert args == {"query": "speculative decoding", "sort_by": "submittedDate", "categories": ["cs.CL"]}
    assert rig.urls == [SEARCH_URL] and rig.parsed == ["<feed>search</feed>"]


# 9 and -1 got past checks; 3.0 is an integer to JSON Schema, and a direct call may pass it as a float
@pytest.mark.parametrize("max_results, shown", [(2, 2), (5, 5), (9, 5), (-1, 1), (3.0, 3), (9.0, 5)])
def test_search_asks_for_and_shows_at_most_max_results_and_never_more_than_the_cap(max_results, shown):
    rig = Rig({SEARCH_URL: (200, "<feed>search</feed>")})
    hits = rig.tools["search_arxiv"]({"query": "x", "max_results": max_results}).content
    assert [h["arxiv_id"] for h in hits] == [f"2609.{n:05d}" for n in range(1, shown + 1)]
    asked = rig.built[0][0]["max_results"]  # the URL builder gets the count the tool shows, as an int
    assert asked == shown and type(asked) is int


def test_search_without_papers_is_empty_results():
    rig = Rig({SEARCH_URL: (200, "<feed/>")})
    assert rig.tools["search_arxiv"]({"query": "x"}) == ToolOutcome([], schemas.EMPTY_RESULTS)


# get_paper


def test_get_paper_asks_the_api_for_the_id_and_cuts_the_abstract():
    rig = Rig({PAPER_URL: (200, f"<feed>{ID}</feed>")})
    outcome = rig.tools["get_paper"]({"arxiv_id": ID})
    full = atom_paper(1)
    assert rig.urls == [PAPER_URL] == ["https://export.arxiv.org/api/query?id_list=2609.00001"]
    assert outcome == ToolOutcome({"arxiv_id": ID, "title": "Paper 1", "authors": ["A. Author", "B. Author"],
                                   "published": "2026-09-15", "abstract": full["abstract"][:1000]})
    assert len(outcome.content["abstract"]) == schemas.MAX_ABSTRACT_CHARS < len(full["abstract"])


@pytest.mark.parametrize("count, shown", [
    (1, ["Author 1"]),
    (5, ["Author 1", "Author 2", "Author 3", "Author 4", "Author 5"]),
    (6, ["Author 1", "Author 2", "Author 3", "Author 4", "Author 5", "et al."]),
    (40, ["Author 1", "Author 2", "Author 3", "Author 4", "Author 5", "et al."]),
])
def test_get_paper_lists_at_most_five_authors_and_then_et_al(count, shown):
    paper = {**atom_paper(1), "authors": [f"Author {n}" for n in range(1, count + 1)]}
    assert feed_rig([paper]).tools["get_paper"]({"arxiv_id": ID}).content["authors"] == shown


def test_an_unknown_paper_is_not_found_and_its_id_is_not_echoed():
    rig = Rig({PAPER_URL: (200, "<feed/>")})
    assert rig.tools["get_paper"]({"arxiv_id": ID}) == ToolOutcome({}, schemas.NOT_FOUND)


# fetch_fulltext


def test_fulltext_section_from_the_html_with_default_section_and_length():
    rig = Rig({HTML_URL: (200, HTML)}, sections={"introduction": "We study speculative decoding."})
    outcome = rig.tools["fetch_fulltext"]({"arxiv_id": ID})
    assert rig.urls == [HTML_URL] and rig.extracted == [(HTML, "introduction", 1500)]
    assert outcome == ToolOutcome({"arxiv_id": ID, "section": "introduction", "source": "html",
                                   "text": "We study speculative decoding."})


# 5000 and -5 got past checks; 600.0 is an integer to JSON Schema
@pytest.mark.parametrize("max_chars, asked", [(600, 600), (1500, 1500), (5000, 1500), (-5, 1), (600.0, 600)])
def test_fulltext_length_is_capped_and_the_text_cut_to_it(max_chars, asked):
    rig = Rig({HTML_URL: (200, HTML)}, sections={"method": "m" * 4000})  # an extractor that overshoots
    outcome = rig.tools["fetch_fulltext"]({"arxiv_id": ID, "section": "method", "max_chars": max_chars})
    assert rig.extracted == [(HTML, "method", asked)] and outcome.content["text"] == "m" * asked
    assert type(rig.extracted[0][2]) is int


@pytest.mark.parametrize("extracted", [None, "", " \n "])
def test_a_missing_section_is_not_found_with_the_id_and_section(extracted):
    rig = Rig({HTML_URL: (200, HTML)}, sections={"method": extracted})
    outcome = rig.tools["fetch_fulltext"]({"arxiv_id": ID, "section": "method"})
    assert outcome == ToolOutcome({"arxiv_id": ID, "section": "method"}, schemas.NOT_FOUND)


# net.fetch returns a 404 as a page; a fault injector may raise it instead
@pytest.mark.parametrize("no_html", [(404, "Not Found"), Fail(404)])
@pytest.mark.parametrize("max_chars, length", [(1500, 1000), (300, 300)])
def test_without_html_the_abstract_stands_in_and_is_marked(no_html, max_chars, length):
    rig = Rig({HTML_URL: no_html, PAPER_URL: (200, f"<feed>{ID}</feed>")})
    outcome = rig.tools["fetch_fulltext"]({"arxiv_id": ID, "section": "method", "max_chars": max_chars})
    assert outcome == ToolOutcome({"arxiv_id": ID, "section": "method", "source": "abstract",
                                   "text": atom_paper(1)["abstract"][:length]}, schemas.NO_FULLTEXT)
    assert rig.urls == [HTML_URL, PAPER_URL] and rig.extracted == []


def test_without_html_an_unknown_paper_is_not_found_without_its_id():
    rig = Rig({HTML_URL: (404, ""), PAPER_URL: (200, "<feed/>")})
    assert rig.tools["fetch_fulltext"]({"arxiv_id": ID}) == ToolOutcome({}, schemas.NOT_FOUND)


def test_without_html_a_failed_abstract_fetch_reports_that_failure():
    rig = Rig({HTML_URL: (404, ""), PAPER_URL: Fail(429)})
    assert rig.tools["fetch_fulltext"]({"arxiv_id": ID}) == ToolOutcome({}, schemas.RATE_LIMITED)


# counts that are not integers


@pytest.mark.parametrize("tool, args, name", [("search_arxiv", {"query": "x"}, "max_results"),
                                              ("fetch_fulltext", {"arxiv_id": ID}, "max_chars")])
@pytest.mark.parametrize("value", [2.5, "3", True, None])  # check_call must refuse them all
def test_a_count_that_is_not_an_integer_is_refused_before_anything_is_fetched(tool, args, name, value):
    rig = Rig({})
    with pytest.raises(TypeError, match=re.escape(f"{name} must be an integer, got {value!r}")):
        rig.tools[tool]({**args, name: value})
    assert rig.urls == [] and rig.built == []


# parse_atom's contract: a list of papers with every key in schemas.PAPER_KEYS, or None

ALL_THREE = [("search_arxiv", {"query": "x"}), ("get_paper", {"arxiv_id": ID}),
             ("fetch_fulltext", {"arxiv_id": ID})]  # fetch_fulltext through its abstract fallback


@pytest.mark.parametrize("tool, args, empty", [(*ALL_THREE[0], []), (*ALL_THREE[1], {}), (*ALL_THREE[2], {})])
def test_a_feed_parse_atom_cannot_parse_is_an_upstream_error(tool, args, empty):
    rig = Rig({SEARCH_URL: (200, CUT_OFF), PAPER_URL: (200, CUT_OFF), HTML_URL: (404, "Not Found")})
    assert rig.tools[tool](args) == ToolOutcome(empty, schemas.UPSTREAM_ERROR)
    assert rig.parsed == [CUT_OFF] and rig.extracted == []


ATOM_NAMES = {"id": ID, "title": "Paper 1", "summary": "Speculative decoding drafts tokens.",
              "published": "2026-09-15", "authors": ["A. Author"]}  # keys named after the Atom elements
NO_AUTHORS = {key: value for key, value in atom_paper(1).items() if key != "authors"}


@pytest.mark.parametrize("tool, args", ALL_THREE)
@pytest.mark.parametrize("paper, missing", [(ATOM_NAMES, "arxiv_id, abstract, categories"),
                                            (NO_AUTHORS, "authors")])  # a key no search hit even shows
def test_a_paper_without_every_paper_key_is_a_harness_error_naming_the_tool(tool, args, paper, missing):
    with pytest.raises(HarnessError, match=f"^tool '{tool}': parse_atom returned a paper without {missing} "):
        feed_rig([paper]).tools[tool](args)


def test_every_paper_is_held_to_the_contract_not_only_the_shown_ones():
    papers = [atom_paper(n) for n in range(1, 8)]
    del papers[6]["abstract"]
    with pytest.raises(HarnessError, match="without abstract"):
        feed_rig(papers).tools["search_arxiv"]({"query": "x", "max_results": 2})


@pytest.mark.parametrize("parsed, problem", [
    (atom_paper(1), "returned a dict, not a list or None"),  # one paper instead of a list of papers
    ((atom_paper(1),), "returned a tuple, not a list or None"),
    ([list(atom_paper(1).items())], "returned a list where a paper dict belongs"),
    ([{**atom_paper(1), "authors": "A. Author, B. Author"}],
     "returned authors 'A. Author, B. Author', not a list of strings"),
    ([{**atom_paper(1), "authors": [{"name": "A. Author"}]}],
     "returned authors [{'name': 'A. Author'}], not a list of strings"),
    ([{**atom_paper(1), "abstract": None}], "returned abstract as NoneType, not a string"),
    ([{**atom_paper(1), "categories": "cs.CL"}], "returned categories 'cs.CL', not a list of strings"),
])
def test_anything_else_parse_atom_returns_is_a_harness_error_naming_the_tool(parsed, problem):
    with pytest.raises(HarnessError, match=re.escape(f"tool 'get_paper': parse_atom {problem}")):
        feed_rig(parsed).tools["get_paper"]({"arxiv_id": ID})


OWNER_BUGS = [
    # a parse_atom that lets the parser's error escape instead of returning None
    ("search_arxiv", {"query": "x"}, {SEARCH_URL: (200, "<feed>&undefined;</feed>")}, {}, ET.ParseError),
    # an extract_section that fails on a page it does not expect
    ("fetch_fulltext", {"arxiv_id": ID}, {HTML_URL: (200, HTML)}, {"introduction": IndexError("no h2")},
     IndexError),
]


@pytest.mark.parametrize("tool, args, pages, sections, error", OWNER_BUGS)
def test_an_exception_in_an_owner_function_is_no_error_code(tool, args, pages, sections, error):
    with pytest.raises(error):  # the executor reports it as a harness error
        Rig(pages, sections).tools[tool](args)
    executor = ToolExecutor(Rig(pages, sections).tools, table_check(), today=TODAY)
    with pytest.raises(HarnessError) as info:
        executor.run(call(tool, "c1", **args))
    assert isinstance(info.value.__cause__, error)


# failures shared by the three tools

TOOLS = [("search_arxiv", {"query": "x"}, SEARCH_URL, []), ("get_paper", {"arxiv_id": ID}, PAPER_URL, {}),
         ("fetch_fulltext", {"arxiv_id": ID}, HTML_URL, {})]


@pytest.mark.parametrize("tool, args, url, empty", TOOLS)
@pytest.mark.parametrize("failure, code", [
    (Fail(429), schemas.RATE_LIMITED),
    (Fail(503), schemas.UPSTREAM_ERROR),
    (Fail(None), schemas.UPSTREAM_ERROR),  # timeout or dropped connection: no response at all
    ((429, "slow down"), schemas.RATE_LIMITED),  # a fetcher that returns instead of raising
    ((403, "Forbidden"), schemas.UPSTREAM_ERROR),
])
def test_failed_fetches_become_error_codes_and_nothing_is_parsed(tool, args, url, empty, failure, code):
    rig = Rig({url: failure})
    assert rig.tools[tool](args) == ToolOutcome(empty, code)
    assert rig.parsed == [] and rig.extracted == []


HARNESS_FAILURES = {"replay miss": lambda url: ReplayMiss(f"no cassette for {url}"),
                    "not recorded": lambda url: RecordFailed(url, 503, "status 503 after retries")}


@pytest.mark.parametrize("tool, args, url, empty", TOOLS)
@pytest.mark.parametrize("failure", list(HARNESS_FAILURES.values()), ids=list(HARNESS_FAILURES))
def test_harness_failures_of_the_fetcher_pass_through_untouched(tool, args, url, empty, failure):
    raised = failure(url)
    with pytest.raises(HarnessError) as info:
        Rig({url: raised}).tools[tool](args)
    assert info.value is raised


def test_a_replay_miss_in_the_abstract_fallback_passes_through_untouched():
    miss = ReplayMiss(f"no cassette for {PAPER_URL}")
    with pytest.raises(ReplayMiss) as info:
        Rig({HTML_URL: (404, ""), PAPER_URL: miss}).tools["fetch_fulltext"]({"arxiv_id": ID})
    assert info.value is miss


# with the executor


def test_the_executor_sees_the_ids_of_shown_and_existing_papers_only():
    rig = Rig({SEARCH_URL: (200, "<feed>search</feed>"), PAPER_URL: (200, "<feed/>"), HTML_URL: (404, "")})
    executor = ToolExecutor(rig.tools, table_check(), today=TODAY)
    executor.run(call("search_arxiv", "c1", query="x", max_results=2))
    assert executor.seen_ids == ["2609.00001", "2609.00002"]  # five more were parsed but never shown
    executor.reset()
    assert executor.run(call("get_paper", "c2", arxiv_id=ID)).error == schemas.NOT_FOUND
    assert executor.run(call("fetch_fulltext", "c3", arxiv_id=ID)).error == schemas.NOT_FOUND
    assert executor.seen_ids == []


def counting_rig() -> Rig:
    """A search with seven hits and a full text whose extractor returns 4,000 characters."""
    return Rig({SEARCH_URL: (200, "<feed>search</feed>"), HTML_URL: (200, HTML)},
               sections={"method": "m" * 4000})


def test_integral_floats_the_model_writes_reach_the_owner_functions_as_ints():
    rig = counting_rig()
    executor = ToolExecutor(rig.tools, table_check(), today=TODAY)
    written = {"search_arxiv": '{"query": "x", "max_results": 3.0}',
               "fetch_fulltext": f'{{"arxiv_id": "{ID}", "section": "method", "max_chars": 600.0}}'}
    for name, arguments in written.items():
        executor.run(ToolCall.from_api({"id": "c1", "function": {"name": name, "arguments": arguments}}))
    assert rig.built == [({"query": "x", "max_results": 3, "sort_by": "relevance"}, TODAY)]
    assert rig.extracted == [(HTML, "method", 600)]
    assert type(rig.built[0][0]["max_results"]) is int and type(rig.extracted[0][2]) is int


def test_a_call_built_with_integral_floats_runs_as_one_with_ints():
    # arguments that never went through parse_arguments, e.g. a call put together by hand in a script
    rig = counting_rig()
    executor = ToolExecutor(rig.tools, table_check(), today=TODAY)
    hits = executor.run(ToolCall("c1", "search_arxiv", "", {"query": "x", "max_results": 3.0}))
    text = executor.run(ToolCall("c2", "fetch_fulltext", "", {"arxiv_id": ID, "section": "method",
                                                             "max_chars": 600.0}))
    assert [hit["arxiv_id"] for hit in hits.content] == ["2609.00001", "2609.00002", "2609.00003"]
    assert text.content["text"] == "m" * 600
    # the owner's functions get what a parsed call gives them, so both ask for the same URL and section
    assert rig.built == [({"query": "x", "max_results": 3, "sort_by": "relevance"}, TODAY)]
    assert rig.extracted == [(HTML, "method", 600)]
    assert type(rig.built[0][0]["max_results"]) is int and type(rig.extracted[0][2]) is int


# with the executor and the real fetcher


def owner_stand_ins(fetcher) -> dict:
    return arxiv_tools(fetcher, build_search_url=lambda args, today: SEARCH_URL, parse_atom=parse_atom,
                       extract_section=lambda html, section, max_chars: None, today=TODAY)


def arxiv_server(request: httpx.Request) -> httpx.Response:
    """Offline arXiv: the API answers both feeds, the HTML full text does not exist."""
    if request.url.path == "/api/query":
        feed = f"<feed>{ID}</feed>" if "id_list" in request.url.params else "<feed>search</feed>"
        return httpx.Response(200, text=feed, headers={"content-type": "application/atom+xml"})
    return httpx.Response(404, text="Not Found")


def no_requests(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"{request.url} has a cassette but was requested again")


def test_with_the_real_fetcher_a_recording_replays_to_the_same_outcomes(tmp_path):
    calls = [("search_arxiv", {"query": "x"}), ("get_paper", {"arxiv_id": ID}),
             ("fetch_fulltext", {"arxiv_id": ID})]
    recorder = Fetcher("record", tmp_path, transport=httpx.MockTransport(arxiv_server), sleep=lambda s: None)
    recorded = [owner_stand_ins(recorder)[name](args) for name, args in calls]
    recorder.close()
    assert [o.error for o in recorded] == [None, None, schemas.NO_FULLTEXT] and len(recorded[0].content) == 5
    replayer = Fetcher("replay", tmp_path)
    assert [owner_stand_ins(replayer)[name](args) for name, args in calls] == recorded
    # a later recording session answers the recorded URLs from their cassettes, exactly as replay does
    again = Fetcher("record", tmp_path, transport=httpx.MockTransport(no_requests), sleep=lambda s: None)
    assert [owner_stand_ins(again)[name](args) for name, args in calls] == recorded
    again.close()
    executor = ToolExecutor(owner_stand_ins(replayer), table_check(), today=TODAY)
    with pytest.raises(ReplayMiss):  # never recorded: a harness failure, not a tool result
        executor.run(call("get_paper", "c1", arxiv_id="2609.99999"))


def always(status: int):
    """A server that answers every request with status."""
    return lambda request: httpx.Response(status, text="try again later")


def redirect_loop(request: httpx.Request) -> httpx.Response:
    return httpx.Response(302, headers={"location": str(request.url)})


def undecodable(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=b"not gzip at all", headers={"content-encoding": "gzip"})


UNUSABLE = {  # a server without a usable answer: (the server, its last status, what a live fetch becomes)
    "429 after the retries": (always(429), 429, schemas.RATE_LIMITED),
    "503 after the retries": (always(503), 503, schemas.UPSTREAM_ERROR),
    "403": (always(403), 403, schemas.UPSTREAM_ERROR),
    "redirect loop": (redirect_loop, None, schemas.UPSTREAM_ERROR),
    "undecodable body": (undecodable, None, schemas.UPSTREAM_ERROR),
}


@pytest.mark.parametrize("server, code", [(s, c) for s, _, c in UNUSABLE.values()], ids=list(UNUSABLE))
def test_with_the_real_fetcher_a_live_failure_becomes_an_error_code(tmp_path, server, code):
    fetcher = Fetcher("live", tmp_path, transport=httpx.MockTransport(server), sleep=lambda s: None)
    executor = ToolExecutor(owner_stand_ins(fetcher), table_check(), today=TODAY)
    assert executor.run(call("get_paper", "c1", arxiv_id=ID)) == ToolResult("c1", "get_paper", {}, code)
    assert executor.seen_ids == []
    fetcher.close()


@pytest.mark.parametrize("server, status", [(s, st) for s, st, _ in UNUSABLE.values()], ids=list(UNUSABLE))
def test_with_the_real_fetcher_a_failure_while_recording_is_a_harness_failure(tmp_path, server, status):
    fetcher = Fetcher("record", tmp_path, transport=httpx.MockTransport(server), sleep=lambda s: None)
    executor = ToolExecutor(owner_stand_ins(fetcher), table_check(), today=TODAY)
    with pytest.raises(RecordFailed) as failed:  # replay could never show it, so the model must not see it
        executor.run(call("get_paper", "c1", arxiv_id=ID))
    assert isinstance(failed.value, HarnessError) and failed.value.status == status
    assert list(tmp_path.iterdir()) == []  # nothing recorded, so a later session tries again
    fetcher.close()

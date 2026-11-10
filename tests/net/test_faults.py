"""Fault injection: deterministic, reproducible faults that reach the model as real failures would."""

import copy
import hashlib
import xml.etree.ElementTree as ET

import pytest
from support.fakes import HITS, call, table_tools

from research_agent.agent.types import HarnessError, ToolOutcome, call_key
from research_agent.canonical import canonical_json
from research_agent.net.faults import (
    ANY_ANSWER,
    API_200,
    DELAY_S,
    FAULTS,
    HTML_200,
    RATES,
    RETRY_AFTER_S,
    SEARCH_FAULTS,
    FaultBudget,
    FaultInjector,
    SearchFaults,
    applicable,
    draw,
    draw_key,
    empty_feed,
    search_key,
    truncated,
)
from research_agent.net.fetch import Fetched, FetchFailed, ReplayMiss, canonical_url
from research_agent.tools import schemas
from research_agent.tools.adapters import arxiv_tools

TODAY = "2026-10-09"
FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title type="html">ArXiv Query: search_query=all:speculative</title>
  <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">2</opensearch:totalResults>
  <entry>
    <id>http://arxiv.org/abs/2609.00001v1</id>
    <title>Adaptive Draft Length for Speculative Decoding</title>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2609.00002v2</id>
    <title>Tree Verification at Scale</title>
  </entry>
</feed>
"""
SEARCH = "https://export.arxiv.org/api/query?search_query=all:speculative&max_results=5"
PAPER = "https://export.arxiv.org/api/query?id_list=2609.00001"
HTML = "https://arxiv.org/html/2609.00001"
PAGE = "<html><body><section>Introduction text.</section></body></html>"


class Pages:
    """The wrapped fetcher: answers from a table, ReplayMiss for any other URL; records every request."""

    def __init__(self, pages: dict | None = None) -> None:
        self.pages = {SEARCH: (200, FEED), PAPER: (200, FEED), HTML: (200, PAGE)} if pages is None else pages
        self.requests: list[str] = []

    def get(self, url: str) -> Fetched:
        self.requests.append(url)
        if url not in self.pages:
            raise ReplayMiss(f"no cassette for {url}")
        status, text = self.pages[url]
        return Fetched(url=url, final_url=url, status=status, body=text.encode("utf-8"),
                       content_type="application/atom+xml")


def paper_urls(count: int) -> dict:
    return {f"https://export.arxiv.org/api/query?id_list=2609.{n:05d}": (200, FEED) for n in range(count)}


def inject(inner, **options) -> FaultInjector:
    """A FaultInjector for task t1 that never really waits: a delay fault would sleep DELAY_S in the tests."""
    return FaultInjector(inner, **{"task": "t1", "sleep": lambda seconds: None, **options})


def faults_of(injector: FaultInjector, urls, *, repeat: int = 1) -> dict:
    """(url, ordinal) -> fault for every request, the requests made in the given order."""
    for _ in range(repeat):
        for url in urls:
            try:
                injector.get(url)
            except FetchFailed:
                pass
    return {(entry["url"], entry["ordinal"]): entry["fault"] for entry in injector.log}


def test_rate_zero_answers_exactly_like_the_wrapped_fetcher():
    inner = Pages()
    injector = inject(inner, rate=0.0, seed=7)
    for url in [SEARCH, PAPER, HTML] * 3:
        assert injector.get(url) == Pages().get(url)
    assert inner.requests == [SEARCH, PAPER, HTML] * 3
    assert all(entry["fault"] is None for entry in injector.log) and injector.counts() == {"none": 9}


def test_rate_one_always_injects_a_fault_that_applies_to_the_request():
    for seed in range(40):
        injector = inject(Pages(), rate=1.0, seed=seed)
        faults_of(injector, [SEARCH, PAPER, HTML])
        search, paper, html = (entry["fault"] for entry in injector.log)
        assert search in ANY_ANSWER + API_200 and paper in ANY_ANSWER + API_200
        assert html in ANY_ANSWER + HTML_200


def test_the_seven_faults_and_the_plans_rates():
    assert FAULTS == ("timeout", "rate_limited", "server_error", "empty_results", "truncated_atom",
                      "html_404", "delay")
    assert RATES == (0.0, 0.2, 0.5) and DELAY_S == 5.0
    assert set(ANY_ANSWER) | set(API_200) | set(HTML_200) == set(FAULTS)


def test_feed_faults_meet_api_answers_with_200_and_the_404_meets_html_answers_with_200():
    assert applicable(SEARCH, 200) == ["timeout", "rate_limited", "server_error", "empty_results",
                                       "truncated_atom", "delay"]
    assert applicable(HTML, 200) == ["timeout", "rate_limited", "server_error", "html_404", "delay"]
    assert applicable(SEARCH, 400) == applicable(HTML, 404) == list(ANY_ANSWER)  # nothing to edit or remove
    assert applicable(HTML, 200, faults=["empty_results"]) == []


@pytest.mark.parametrize("fault, status, reason", [("timeout", None, "timed out"),
                                                   ("rate_limited", 429, f"Retry-After: {RETRY_AFTER_S}"),
                                                   ("server_error", 500, "status 500")])
def test_failures_are_raised_as_the_fetchers_live_failures(fault, status, reason):
    for url in (SEARCH, HTML):
        with pytest.raises(FetchFailed) as caught:
            inject(Pages(), rate=1.0, seed=0, faults=[fault]).get(url)
        assert caught.value.status == status and reason in caught.value.reason and caught.value.url == url


def test_empty_results_keep_a_well_formed_feed_without_entries():
    page = inject(Pages(), rate=1.0, seed=0, faults=["empty_results"]).get(SEARCH)
    assert page.status == 200 and "<entry" not in page.text and ">0</opensearch:totalResults>" in page.text
    root = ET.fromstring(page.body)
    assert root.find("{http://www.w3.org/2005/Atom}title").text.startswith("ArXiv Query")
    assert page.text == empty_feed(FEED)


def test_a_truncated_feed_is_the_first_half_and_no_longer_well_formed():
    page = inject(Pages(), rate=1.0, seed=0, faults=["truncated_atom"]).get(SEARCH)
    assert page.status == 200 and page.text == FEED[: len(FEED) // 2] == truncated(FEED)
    with pytest.raises(ET.ParseError):
        ET.fromstring(page.body)


def test_html_404_removes_the_full_text():
    page = inject(Pages(), rate=1.0, seed=0, faults=["html_404"]).get(HTML)
    assert (page.status, page.body, page.url) == (404, b"", HTML)


def test_a_delay_waits_and_then_answers_unchanged():
    waits = []
    page = inject(Pages(), rate=1.0, seed=0, faults=["delay"], sleep=waits.append).get(SEARCH)
    assert waits == [DELAY_S] and page == Pages().get(SEARCH)


def test_a_replay_miss_surfaces_and_uses_up_no_draw():
    inner = Pages({})
    injector = inject(inner, rate=1.0, seed=3)
    with pytest.raises(ReplayMiss):
        injector.get(SEARCH)
    assert injector.log == []
    inner.pages[SEARCH] = (200, FEED)
    fresh = inject(Pages(), rate=1.0, seed=3)
    assert faults_of(injector, [SEARCH]) == faults_of(fresh, [SEARCH])  # still the first request for SEARCH


def test_the_same_seed_gives_the_same_faults_in_any_order():
    urls = list(paper_urls(300))
    forward = faults_of(inject(Pages(paper_urls(300)), rate=0.5, seed=11), urls, repeat=2)
    backward = faults_of(inject(Pages(paper_urls(300)), rate=0.5, seed=11), urls[::-1], repeat=2)
    assert forward == backward and len(forward) == 600
    other = faults_of(inject(Pages(paper_urls(300)), rate=0.5, seed=12), urls, repeat=2)
    assert other != forward


@pytest.mark.parametrize("rate", [0.2, 0.5])
def test_the_rate_holds_and_every_applicable_fault_occurs(rate):
    injector = inject(Pages(paper_urls(4000)), rate=rate, seed=2026)
    faults_of(injector, list(paper_urls(4000)))
    counts = injector.counts()
    injected = 4000 - counts["none"]
    assert abs(injected / 4000 - rate) < 0.02  # binomial sd at most 0.008
    assert set(counts) == {"none", *applicable(next(iter(paper_urls(1))), 200)}
    assert min(counts[fault] for fault in counts if fault != "none") > injected / 6 * 0.75


def test_transient_faults_are_drawn_again_for_every_request_and_persistent_ones_are_not():
    transient = faults_of(inject(Pages(), rate=0.5, seed=5), [SEARCH], repeat=60)
    assert None in transient.values() and len(set(transient.values())) > 2
    persistent = faults_of(inject(Pages(), rate=0.5, seed=5, persistent=True), [SEARCH], repeat=60)
    assert len(set(persistent.values())) == 1
    assert [key[1] for key in persistent] == list(range(60))  # the ordinal is still logged


def test_equal_urls_written_differently_are_one_url():
    reordered = "https://export.arxiv.org/api/query?max_results=5&search_query=all%3Aspeculative"
    injector = inject(Pages({SEARCH: (200, FEED), reordered: (200, FEED)}), rate=0.0, seed=1)
    injector.get(SEARCH)
    injector.get(reordered)
    assert [(entry["url"], entry["ordinal"]) for entry in injector.log] == [(injector.log[0]["url"], 0),
                                                                            (injector.log[0]["url"], 1)]
    assert draw(1, "t1", SEARCH, 0) == draw(1, "t1", reordered, 0)


def test_a_tasks_faults_do_not_depend_on_what_earlier_tasks_fetched():
    # system A retried the URL in task 1, system B did not; in task 2 both ask once and meet the same fault
    url = "https://export.arxiv.org/api/query?id_list=2609.00003"
    pages = {url: (200, FEED)}
    for seed in range(50):
        a1, b1 = inject(Pages(pages), rate=0.5, seed=seed), inject(Pages(pages), rate=0.5, seed=seed)
        faults_of(a1, [url], repeat=2)
        faults_of(b1, [url])
        a2 = inject(Pages(pages), rate=0.5, seed=seed, task="t2")
        b2 = inject(Pages(pages), rate=0.5, seed=seed, task="t2")
        assert faults_of(a2, [url]) == faults_of(b2, [url])
        assert a2.log[0]["task"] == "t2" and a2.log[0]["ordinal"] == 0


def test_different_tasks_draw_their_faults_independently():
    urls = list(paper_urls(300))
    first = faults_of(inject(Pages(paper_urls(300)), rate=0.5, seed=4, task="a"), urls)
    second = faults_of(inject(Pages(paper_urls(300)), rate=0.5, seed=4, task="b"), urls)
    assert first.keys() == second.keys() and first != second


def test_bad_rates_and_unknown_faults_are_refused():
    for rate in (-0.1, 1.5):
        with pytest.raises(ValueError, match="rate"):
            inject(Pages(), rate=rate, seed=0)
    with pytest.raises(ValueError, match="unknown faults"):
        inject(Pages(), rate=0.2, seed=0, faults=["timeout", "meteor"])


# Through the real adapters: each fault reaches the model as the error code of the real failure.

PAPERS = [{"arxiv_id": f"2609.0000{n}", "title": f"Paper {n}", "authors": ["A. Author"],
           "published": "2026-09-15", "abstract": f"Abstract of paper {n}.", "categories": ["cs.CL"]}
          for n in (1, 2)]
FEEDS = {FEED: PAPERS, empty_feed(FEED): [], truncated(FEED): None}  # what the owner's parse_atom returns


def tools(fault: str) -> dict:
    """The arXiv tools over an injector that meets every request with fault; owner functions scripted."""
    injector = inject(Pages(), rate=1.0, seed=0, faults=[fault], sleep=lambda seconds: None)
    return arxiv_tools(injector, build_search_url=lambda args, today: SEARCH,
                       parse_atom=lambda xml: copy.deepcopy(FEEDS[xml]),
                       extract_section=lambda html, section, max_chars: "Introduction text.", today=TODAY)


@pytest.mark.parametrize("fault, search, paper", [
    ("timeout", schemas.UPSTREAM_ERROR, schemas.UPSTREAM_ERROR),
    ("rate_limited", schemas.RATE_LIMITED, schemas.RATE_LIMITED),
    ("server_error", schemas.UPSTREAM_ERROR, schemas.UPSTREAM_ERROR),
    ("empty_results", schemas.EMPTY_RESULTS, schemas.NOT_FOUND),
    ("truncated_atom", schemas.UPSTREAM_ERROR, schemas.UPSTREAM_ERROR),
    ("delay", None, None),
])
def test_through_the_adapters_api_faults_become_the_error_codes_of_real_failures(fault, search, paper):
    arxiv = tools(fault)
    found = arxiv["search_arxiv"]({"query": "speculative decoding"})
    looked_up = arxiv["get_paper"]({"arxiv_id": "2609.00001"})
    assert (found.error, looked_up.error) == (search, paper)
    if fault == "delay":
        assert [hit["arxiv_id"] for hit in found.content] == ["2609.00001", "2609.00002"]
    if search is not None:
        assert found.content in ([], {})  # an error shows no papers and no ids


def test_through_the_adapters_html_404_falls_back_to_the_abstract():
    outcome = tools("html_404")["fetch_fulltext"]({"arxiv_id": "2609.00001"})
    assert outcome.error == schemas.NO_FULLTEXT
    assert outcome.content["source"] == "abstract" and outcome.content["text"] == "Abstract of paper 1."


# draw_key: the hash behind draw, with any key taken verbatim.


def expected_draw(seed, task, key, ordinal) -> tuple[float, int]:
    """The hash input pinned: seed, task, key and ordinal joined by \\x1f, so earlier fault tables stay
    reproducible."""
    digest = hashlib.sha256(f"{seed}\x1f{task}\x1f{key}\x1f{ordinal}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64, int.from_bytes(digest[8:16], "big")


def test_draw_is_draw_key_over_the_canonical_url():
    reordered = "https://export.arxiv.org/api/query?max_results=5&search_query=all%3Aspeculative"
    for url in (SEARCH, reordered, PAPER, HTML, HTML + "#S2"):
        for ordinal in (0, 7, None):
            assert draw(1, "t1", url, ordinal) == draw_key(1, "t1", canonical_url(url), ordinal)
    assert draw(1, "t1", SEARCH, 0) == draw(1, "t1", reordered, 0)
    assert draw(1, "t1", SEARCH, 0) == draw_key(1, "t1", canonical_url(SEARCH), 0)


def test_draw_key_hashes_the_key_verbatim():
    assert draw_key(3, "t1", "search:{}", 2) == expected_draw(3, "t1", "search:{}", 2)
    assert draw_key(3, "t1", "search:{}", None) == expected_draw(3, "t1", "search:{}", "*")
    # '#', '?' and '&' stay part of the key; in a URL, draw would drop the fragment and sort the query
    assert draw_key(1, "t1", "k#a", 0) != draw_key(1, "t1", "k#b", 0)
    assert draw_key(1, "t1", "k?b=1&a=2", 0) != draw_key(1, "t1", "k?a=2&b=1", 0)
    assert draw(1, "t1", HTML + "#a", 0) == draw(1, "t1", HTML + "#b", 0)
    u, k = draw_key(1, "t1", "k", 0)
    assert 0.0 <= u < 1.0 and isinstance(k, int) and k >= 0


# FaultBudget: a cap on the faults injected, shared by every injector given it.


def test_a_fault_budget_grants_faults_up_to_its_cap_and_counts_them():
    budget = FaultBudget(2)
    assert (budget.injected, budget.remaining) == (0, 2)
    assert [budget.take() for _ in range(4)] == [True, True, False, False]
    assert (budget.injected, budget.remaining) == (2, 0)
    assert FaultBudget(0).take() is False
    unlimited = FaultBudget(None)
    assert all(unlimited.take() for _ in range(1000))
    assert (unlimited.injected, unlimited.remaining) == (1000, None)
    for bad in (-1, 1.5, True, "3"):
        with pytest.raises(ValueError, match="max_faults"):
            FaultBudget(bad)


def test_a_budget_changes_nothing_but_the_faults_it_refuses():
    urls = list(paper_urls(300))
    free = inject(Pages(paper_urls(300)), rate=0.5, seed=11)
    unlimited = FaultBudget(None)
    tracked = inject(Pages(paper_urls(300)), rate=0.5, seed=11, budget=unlimited)
    capped = inject(Pages(paper_urls(300)), rate=0.5, seed=11, budget=FaultBudget(10))
    without, with_unlimited, with_cap = (faults_of(injector, urls) for injector in (free, tracked, capped))
    assert with_unlimited == without and unlimited.injected == 300 - free.counts()["none"]
    assert all("budget_exhausted" not in entry for entry in free.log + tracked.log)
    # the capped run meets the free run's first ten faults; every later drawn fault is refused and flagged
    drawn = [key for key, fault in without.items() if fault is not None]
    assert [key for key, fault in with_cap.items() if fault is not None] == drawn[:10]
    assert [(e["url"], e["ordinal"]) for e in capped.log if e.get("budget_exhausted")] == drawn[10:]
    assert all(entry["fault"] is None for entry in capped.log if entry.get("budget_exhausted"))
    assert capped.counts()["none"] == free.counts()["none"] + len(drawn) - 10


def test_a_persistent_injector_with_a_budget_is_down_for_the_first_k_requests_and_then_up():
    budget = FaultBudget(3)
    injector = inject(Pages(), rate=1.0, seed=5, persistent=True, faults=["server_error"], budget=budget)
    answers = []
    for _ in range(8):
        try:
            answers.append(injector.get(SEARCH))
        except FetchFailed as exc:
            answers.append(exc.status)
    assert answers == [500, 500, 500] + [Pages().get(SEARCH)] * 5
    assert [entry["fault"] for entry in injector.log] == ["server_error"] * 3 + [None] * 5
    assert [entry.get("budget_exhausted") for entry in injector.log] == [None] * 3 + [True] * 5
    assert (budget.injected, budget.remaining) == (3, 0)
    assert injector.counts() == {"none": 5, "server_error": 3}
    # with every fault allowed, the one fault a persistent draw picks is the one met three times
    met = list(faults_of(inject(Pages(), rate=1.0, seed=5, persistent=True, budget=FaultBudget(3)),
                         [SEARCH], repeat=8).values())
    assert met[0] is not None and met[:3] == [met[0]] * 3 and met[3:] == [None] * 5


# SearchFaults: the fault hook of the search that answers from the recorded corpus.


def scripted_search():
    """A search_arxiv stand-in that answers HITS for every query, and the list of args it was called with."""
    tools, seen = table_tools({"search_arxiv": [ToolOutcome(HITS)]})
    return tools["search_arxiv"], seen["search_arxiv"]


def search_faults(inner=None, **options) -> SearchFaults:
    """A SearchFaults for task t1, over the scripted search unless another tool is given."""
    return SearchFaults(inner or scripted_search()[0], **{"task": "t1", **options})


def test_one_shared_budget_holds_a_fetch_injector_and_a_search_injector_to_exactly_one_fault():
    plain = Pages()
    steps = [("search", {"query": "a"}), ("fetch", PAPER), ("fetch", HTML), ("search", {"query": "b"}),
             ("fetch", SEARCH), ("search", {"query": "a"})]
    for order in (steps, steps[1:] + steps[:1]):  # a search first, then a fetch first
        budget = FaultBudget(1)
        fetch = inject(Pages(), rate=1.0, seed=9, budget=budget)
        search = search_faults(rate=1.0, seed=9, budget=budget)
        answers = []
        for kind, what in order:
            try:
                answers.append(search(what) if kind == "search" else fetch.get(what))
            except FetchFailed as exc:
                answers.append(exc)
        assert (budget.injected, budget.remaining) == (1, 0)
        entries = fetch.log + search.log
        faulted = [entry for entry in entries if entry["fault"] is not None]
        assert len(faulted) == 1 and "budget_exhausted" not in faulted[0]
        refused = [entry for entry in entries if entry["fault"] is None]
        assert len(refused) == 5 and all(entry["budget_exhausted"] is True for entry in refused)
        # at rate 1.0 the first request draws a fault and takes the budget; the rest answer as without faults
        first = search.log if order[0][0] == "search" else fetch.log
        assert first[0]["fault"] is not None and first[0]["ordinal"] == 0
        for (kind, what), answer in list(zip(order, answers))[1:]:
            assert answer == (ToolOutcome(HITS) if kind == "search" else plain.get(what))


@pytest.mark.parametrize("fault, code", [("timeout", schemas.UPSTREAM_ERROR),
                                         ("rate_limited", schemas.RATE_LIMITED),
                                         ("server_error", schemas.UPSTREAM_ERROR),
                                         ("empty_results", schemas.EMPTY_RESULTS)])
def test_search_faults_become_the_outcomes_the_adapter_gives_for_the_fetcher_fault(fault, code):
    inner, seen = scripted_search()
    search = search_faults(inner, rate=1.0, seed=0, faults=[fault])
    outcome = search({"query": "speculative decoding"})
    assert outcome == ToolOutcome([], code) and outcome.content == []
    assert "arxiv_id" not in canonical_json(outcome.content)  # adds no id to those the model has seen
    assert outcome == tools(fault)["search_arxiv"]({"query": "speculative decoding"})  # as the adapter gives
    assert seen == [{"query": "speculative decoding"}]  # the wrapped tool was asked first
    assert search.log == [{"task": "t1", "key": search_key({"query": "speculative decoding"}), "ordinal": 0,
                           "fault": fault}]
    assert search.counts() == {fault: 1} and search.inner is inner


def test_a_harness_error_in_the_wrapped_search_surfaces_and_uses_up_no_draw():
    failures = [HarnessError("replay search: corpus index missing")]

    def flaky(args: dict) -> ToolOutcome:
        if failures:
            raise failures.pop()
        return ToolOutcome(HITS)

    search = search_faults(flaky, rate=1.0, seed=3)
    with pytest.raises(HarnessError, match="corpus index"):
        search({"query": "a"})
    assert search.log == []
    fresh = search_faults(rate=1.0, seed=3)
    assert search({"query": "a"}) == fresh({"query": "a"}) and search.log == fresh.log  # still ordinal 0


def test_search_faults_at_rate_zero_answer_exactly_like_the_wrapped_tool():
    inner, seen = scripted_search()
    search = search_faults(inner, rate=0.0, seed=7)
    queries = [{"query": "a"}, {"query": "b", "max_results": 2}, {"query": "a"}]
    assert [search(args) for args in queries] == [ToolOutcome(HITS)] * 3
    assert seen == queries and search.inner is inner
    assert all(entry["fault"] is None and "budget_exhausted" not in entry for entry in search.log)
    assert search.counts() == {"none": 3}


def test_search_faults_are_the_same_for_the_same_seed_task_and_arguments():
    queries = [{"query": f"topic {n}"} for n in range(200)]

    def faults(seed: int, task: str) -> list[tuple]:
        search = search_faults(rate=0.5, seed=seed, task=task)
        for args in queries * 2:
            search(args)
        return [(entry["key"], entry["ordinal"], entry["fault"]) for entry in search.log]

    same = faults(11, "t1")
    assert same == faults(11, "t1") and len(same) == 400
    assert faults(12, "t1") != same and faults(11, "t2") != same
    key = same[0][0]  # another task hashes other input for the same key, so its draws are independent
    assert draw_key(11, "t1", key, 0) != draw_key(11, "t2", key, 0)


def test_a_search_with_and_without_its_defaults_is_one_search():
    bare, explicit = {"query": "x"}, {"query": "x", "max_results": 5, "sort_by": "relevance"}
    assert search_key(bare) == 'search:{"max_results":5,"query":"x","sort_by":"relevance"}'
    assert search_key(explicit) == search_key(bare)
    search = search_faults(rate=0.5, seed=11)
    search(bare)
    search(explicit)
    assert [(entry["key"], entry["ordinal"]) for entry in search.log] == [(search_key(bare), 0),
                                                                          (search_key(bare), 1)]


def test_a_search_asked_with_an_integral_float_is_the_search_asked_with_the_int():
    # the loop already keys 3.0 as 3 (agent/types.py, normalize_arguments); the draw key must agree with it
    as_int, as_float = {"query": "x", "max_results": 3}, {"query": "x", "max_results": 3.0}
    expected = 'search:{"max_results":3,"query":"x","sort_by":"relevance"}'
    assert search_key(as_float) == search_key(as_int) == expected
    assert search_key({"query": "x", "max_results": 2.5}) != search_key(as_int)  # only integral floats
    with_list = {"query": "x", "categories": ["cs.CL"], "max_results": 1.0}
    for args in (as_int, as_float, {"query": "x"}, with_list):
        loop_key = call_key(call("search_arxiv", **args))  # the loop's identity of the same call
        assert search_key(args) == "search:" + loop_key.removeprefix("search_arxiv:")
    search = search_faults(rate=0.5, seed=11)
    search(as_int)
    search(as_float)
    assert [(entry["key"], entry["ordinal"]) for entry in search.log] == [(search_key(as_int), 0),
                                                                          (search_key(as_int), 1)]


def test_search_faults_count_an_ordinal_per_distinct_search():
    search = search_faults(rate=0.5, seed=2)
    a, b = {"query": "a"}, {"query": "a", "date_from": "2026-09-01"}
    for args in (a, a, b, a, b):
        search(args)
    assert [(entry["key"], entry["ordinal"]) for entry in search.log] == [
        (search_key(a), 0), (search_key(a), 1), (search_key(b), 0), (search_key(a), 2), (search_key(b), 1)]
    assert all(entry["task"] == "t1" for entry in search.log)


@pytest.mark.parametrize("rate", [0.2, 0.5])
def test_the_search_rate_holds_and_every_search_fault_occurs(rate):
    search = search_faults(rate=rate, seed=2026)
    for n in range(4000):
        search({"query": f"topic {n}"})
    counts = search.counts()
    injected = 4000 - counts["none"]
    assert abs(injected / 4000 - rate) < 0.02  # binomial sd at most 0.008
    assert list(counts) == ["none", *SEARCH_FAULTS]  # every fault occurs, in SEARCH_FAULTS order
    assert min(counts[fault] for fault in SEARCH_FAULTS) > injected / 4 * 0.75


def test_search_faults_refuse_bad_rates_and_faults_the_search_cannot_inject():
    for rate in (-0.1, 1.5):
        with pytest.raises(ValueError, match="rate"):
            search_faults(rate=rate, seed=0)
    for faults in (["timeout", "meteor"], ["html_404"], ["truncated_atom"], ["delay"]):  # fetcher-only faults
        with pytest.raises(ValueError, match="unknown faults"):
            search_faults(rate=0.2, seed=0, faults=faults)
    assert SEARCH_FAULTS == ("timeout", "rate_limited", "server_error", "empty_results")
    assert set(SEARCH_FAULTS) < set(FAULTS)


def test_persistent_search_faults_draw_one_fault_for_every_repetition_of_a_search():
    """--fault-persistent of scripts/run_e2e.py: a search that is down stays down for the whole task, as a
    persistent FaultInjector keeps a URL down; the ordinal is still logged, and the draw is the one without
    an ordinal, so a persistent fetch and a persistent search follow the same rule."""
    transient, persistent = search_faults(rate=0.5, seed=5), search_faults(rate=0.5, seed=5, persistent=True)
    for _ in range(60):
        transient({"query": "a"})
        persistent({"query": "a"})
    assert len({entry["fault"] for entry in transient.log}) > 2
    assert len({entry["fault"] for entry in persistent.log}) == 1
    assert [entry["ordinal"] for entry in persistent.log] == list(range(60))
    for query in (f"topic {n}" for n in range(200)):
        search = search_faults(rate=0.5, seed=5, persistent=True)
        search({"query": query})
        u, k = draw_key(5, "t1", search_key({"query": query}), None)
        assert search.log[0]["fault"] == (SEARCH_FAULTS[k % len(SEARCH_FAULTS)] if u < 0.5 else None)
    assert search_faults(rate=0.5, seed=5).persistent is False

"""research_agent.live.world offline: the live tools over Fetcher("live") behind httpx.MockTransport with the
real arxiv functions (lenient parsing, the papers kept for write_note, the failure codes), the rate limits on
a fake clock within a run and across runs, and one today for the prompt, check_call, the search window and the
note. Nothing goes on the network."""

from __future__ import annotations

import json
from itertools import pairwise

import httpx
import pytest
from support.fakes import ScriptedModel, call, turn

from research_agent import arxiv
from research_agent.agent.prompts import system_prompt
from research_agent.agent.types import ToolOutcome
from research_agent.eval.systems import Built
from research_agent.live.session import run_live_turn
from research_agent.live.world import (
    COOL_DOWN_SLACK,
    LAST_REQUESTS,
    REQUEST_LOG,
    PaperMemo,
    live_world,
)
from research_agent.notes import NoteStore
from research_agent.tools import schemas
from research_agent.tools.adapters import fulltext_url, paper_url

TODAY = "2026-11-10"
API, HTML = "export.arxiv.org", "arxiv.org"
A, B, UNKNOWN = "2611.00001", "2611.00002", "2510.99999"
TITLE_A = "Adaptive Draft Length for Speculative Decoding"
ABSTRACT_A = ("Speculative decoding drafts tokens with a small model. We adapt the draft length to the "
              "acceptance rate. The method is faster on long outputs.")
SEARCH = "https://export.arxiv.org/api/query?search_query=*"  # any search


def entry(arxiv_id: str, title: str, abstract: str) -> str:
    return (f"<entry><id>http://arxiv.org/abs/{arxiv_id}v1</id><published>2026-11-02</published>"
            f"<title>{title}</title><summary>{abstract}</summary><author><name>A. Author</name></author>"
            '<category term="cs.CL"/></entry>')


ENTRY_A = entry(A, TITLE_A, ABSTRACT_A)
ENTRY_B = entry(B, "Tree Verification at Scale", "Tree verification checks drafted tokens at once.")
OLD_STYLE = ("<entry><id>http://arxiv.org/abs/cs/0112017v1</id><published>2001-12-14</published>"
             "<title>An Old Paper</title><summary>Before the new identifiers.</summary></entry>")
ERROR_ENTRY = (f"<entry><id>http://arxiv.org/api/errors#incorrect_id_format_for_{UNKNOWN}</id>"
               f"<title>Error</title><summary>incorrect id format for {UNKNOWN}</summary></entry>")
EMPTY_ENTRY = f"<entry><id>http://arxiv.org/abs/{UNKNOWN}</id><title></title><summary></summary></entry>"


def feed(*entries: str) -> str:
    return ('<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">'
            "<title>ArXiv Query</title>" + "".join(entries) + "</feed>")


def atom(*entries: str) -> httpx.Response:
    return httpx.Response(200, text=feed(*entries), headers={"content-type": "application/atom+xml"})


def html(introduction: str) -> httpx.Response:
    page = ('<html><body><article class="ltx_document"><section id="S1" class="ltx_section">'
            '<h2 class="ltx_title ltx_title_section"><span class="ltx_tag ltx_tag_section">1 </span>'
            f'Introduction</h2><div class="ltx_para"><p class="ltx_p">{introduction}</p></div></section>'
            "</article></body></html>")
    return httpx.Response(200, text=page, headers={"content-type": "text/html"})


class FakeClock:
    """A clock that moves only when something sleeps on it; every sleep is kept."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now, self.sleeps = start, []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class Arxiv:
    """arXiv behind httpx.MockTransport: an answer per URL (a key ending in * matches by prefix; a list is
    answered in order, its last item repeating), every request kept with the fake clock's time."""

    def __init__(self, pages: dict, clock: FakeClock) -> None:
        self.pages = {url: list(answer) if isinstance(answer, list) else [answer]
                      for url, answer in pages.items()}
        self.clock = clock
        self.requests: list[tuple[str, float]] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append((url, self.clock.now))
        for key, answers in self.pages.items():
            if url == key or (key.endswith("*") and url.startswith(key[:-1])):
                return answers.pop(0) if len(answers) > 1 else answers[0]
        raise AssertionError(f"unexpected request {url}")

    def urls(self) -> list[str]:
        return [url for url, _ in self.requests]

    def times(self, host: str) -> list[float]:
        return [t for url, t in self.requests if httpx.URL(url).host == host]


def world(tmp_path, pages: dict, clock: FakeClock | None = None):
    """The live world of a run over the fake arXiv, its notes library and the tools of TODAY."""
    clock = clock or FakeClock()
    server = Arxiv(pages, clock)
    live = live_world(tmp_path / "run", tmp_path / "state", transport=httpx.MockTransport(server),
                      clock=clock, wall=clock, sleep=clock.sleep, retry_delays=(1.0, 1.0))
    store = NoteStore(tmp_path / "notes")
    return live, server, store, live.tools(store, TODAY)


def log_lines(tmp_path) -> list[dict]:
    path = tmp_path / "run" / REQUEST_LOG
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---- lenient parsing and the papers kept ------------------------------------------------------------------


def test_a_search_skips_an_old_style_entry_and_returns_the_new_ones(tmp_path):
    live, server, _, tools = world(tmp_path, {SEARCH: atom(ENTRY_A, OLD_STYLE, ENTRY_B)})
    outcome = tools["search_arxiv"]({"query": "speculative decoding"})
    assert outcome.error is None and [hit["arxiv_id"] for hit in outcome.content] == [A, B]
    assert A in live.memo and B in live.memo and len(live.memo) == 2
    assert arxiv.parse_atom(feed(ENTRY_A, OLD_STYLE)) is None  # by default the whole feed would be lost
    assert len(server.requests) == 1


@pytest.mark.parametrize("answer", [feed(), feed(ERROR_ENTRY), feed(EMPTY_ENTRY)],
                         ids=["empty feed", "error entry", "empty entry"])
def test_an_unknown_id_is_not_found_whatever_the_feed_and_no_id_is_echoed(tmp_path, answer):
    live, _, _, tools = world(tmp_path, {paper_url(UNKNOWN): httpx.Response(200, text=answer)})
    assert tools["get_paper"]({"arxiv_id": UNKNOWN}) == ToolOutcome({}, schemas.NOT_FOUND)
    assert UNKNOWN not in live.memo


def test_write_note_after_get_paper_sends_no_second_request(tmp_path):
    _, server, store, tools = world(tmp_path, {paper_url(A): atom(ENTRY_A)})
    assert tools["get_paper"]({"arxiv_id": A}).content["title"] == TITLE_A
    outcome = tools["write_note"]({"arxiv_id": A, "mode": "scaffold"})
    assert outcome.error is None and outcome.content == {"note_id": f"{A}-1", "arxiv_id": A,
                                                         "path": f"{A}-1.md"}
    assert server.urls() == [paper_url(A)]
    note = (store.root / outcome.content["path"]).read_text(encoding="utf-8")
    assert note.startswith(f"### {TODAY} {TITLE_A}\n") and "We adapt the draft length" in note


def test_write_note_on_a_paper_seen_only_in_its_full_text_looks_it_up_once(tmp_path):
    live, server, _, tools = world(tmp_path, {fulltext_url(A): html("We draft tokens."),
                                              paper_url(A): atom(ENTRY_A)})
    shown = tools["fetch_fulltext"]({"arxiv_id": A})
    assert shown.error is None and shown.content["source"] == "html" and A not in live.memo
    first = tools["write_note"]({"arxiv_id": A})
    second = tools["write_note"]({"arxiv_id": A, "mode": "scaffold"})
    assert first.error is None and second.error is None and second.content["note_id"] == f"{A}-2"
    assert server.urls() == [fulltext_url(A), paper_url(A)]  # one lookup, then the kept paper


def test_write_note_on_an_id_the_api_does_not_know_is_not_found(tmp_path):
    _, server, store, tools = world(tmp_path, {paper_url(UNKNOWN): atom()})
    outcome = tools["write_note"]({"arxiv_id": UNKNOWN, "mode": "scaffold"})
    assert outcome == ToolOutcome({}, schemas.NOT_FOUND)
    assert server.urls() == [paper_url(UNKNOWN)] and store.list() == []


@pytest.mark.parametrize(("status", "code"), [(503, schemas.UPSTREAM_ERROR), (429, schemas.RATE_LIMITED)])
def test_a_lookup_that_keeps_failing_is_the_adapters_code_and_writes_nothing(tmp_path, status, code):
    _, server, store, tools = world(tmp_path, {paper_url(A): httpx.Response(status, text="try later")})
    assert tools["write_note"]({"arxiv_id": A, "mode": "scaffold"}) == ToolOutcome({}, code)
    assert server.urls() == [paper_url(A)] * 3  # the first attempt and the two retries of the fake's delays
    assert store.list() == []


def test_write_note_without_a_valid_call_sends_nothing(tmp_path):
    _, server, _, tools = world(tmp_path, {})
    assert tools["write_note"]({"arxiv_id": A, "mode": "essay"}).error == schemas.INVALID_ARGUMENTS
    assert tools["write_note"]({"arxiv_id": "cs/0112017"}).error == schemas.NOT_FOUND
    assert server.requests == []


def test_the_memo_hands_out_copies():
    memo = PaperMemo()
    memo.parse(feed(ENTRY_A))
    paper = memo.get(A)
    paper["title"] = "changed"
    assert memo.get(A)["title"] == TITLE_A and memo.get(B) is None


# ---- rate limits ------------------------------------------------------------------------------------------


def test_requests_are_3_s_apart_to_the_api_and_15_s_to_the_html_pages(tmp_path):
    pages = {paper_url(A): atom(ENTRY_A), paper_url(B): atom(ENTRY_B), fulltext_url(A): html("One."),
             fulltext_url(B): html("Two.")}
    live, server, _, tools = world(tmp_path, pages)
    for name, arxiv_id in (("get_paper", A), ("get_paper", B), ("fetch_fulltext", A), ("fetch_fulltext", B)):
        assert tools[name]({"arxiv_id": arxiv_id}).error is None
    api, pages_ = server.times(API), server.times(HTML)
    assert api[1] - api[0] >= 3.0 and pages_[1] - pages_[0] >= 15.0
    assert live.limits.slept == pytest.approx(18.0)
    lines = log_lines(tmp_path)
    sent = [line for line in lines if line["event"] == "request"]
    fetches = [line for line in lines if line["event"] == "fetch"]
    assert [line["host"] for line in sent] == [API, API, HTML, HTML] and len(fetches) == 4
    assert [line["slept"] for line in sent] == [0.0, 3.0, 0.0, 15.0]
    assert all(f["t"] <= r["t"] <= f["t_end"] for f, r in zip(fetches, sent, strict=True))
    assert all(earlier["t_end"] <= later["t"] for earlier, later in pairwise(fetches))
    state = json.loads((tmp_path / "state" / LAST_REQUESTS).read_text(encoding="utf-8"))
    assert state == {API: api[1], HTML: pages_[1]}
    assert live.close() == 0.0  # the state was kept: the run ends without a wait


def test_a_state_that_cannot_be_saved_is_tried_again_at_the_close_and_its_intervals_are_waited_out(tmp_path):
    clock = FakeClock()
    (tmp_path / "state" / LAST_REQUESTS).mkdir(parents=True)  # a folder where the file goes: no state at all
    live, _, _, tools = world(tmp_path, {paper_url(A): atom(ENTRY_A)}, clock)
    tools["get_paper"]({"arxiv_id": A})
    assert live.limits.state_error is not None and clock.sleeps == [3.0]  # the first request waited it all
    assert [line["state_error"] for line in log_lines(tmp_path) if line["event"] == "request"]
    heard: list[str] = []
    assert live.close(heard.append) == 3.0  # the API's interval since the request, slept out at the close
    assert clock.sleeps == [3.0, 3.0] and len(heard) == 1 and "could not be saved" in heard[0]


def test_a_state_saved_when_the_close_tries_again_needs_no_wait(tmp_path):
    clock = FakeClock()
    blocked = tmp_path / "state" / LAST_REQUESTS
    blocked.mkdir(parents=True)
    live, _, _, tools = world(tmp_path, {paper_url(A): atom(ENTRY_A)}, clock)
    tools["get_paper"]({"arxiv_id": A})
    blocked.rmdir()  # what held the file let it go
    heard: list[str] = []
    assert live.close(heard.append) == 0.0 and heard == [] and clock.sleeps == [3.0]
    assert json.loads(blocked.read_text(encoding="utf-8")) == {API: clock.now}


def test_a_new_run_sleeps_out_the_rest_of_the_last_runs_intervals(tmp_path):
    clock = FakeClock()
    (tmp_path / "state").mkdir()
    earlier = {API: clock.now - 1.0, HTML: clock.now - 10.0}
    (tmp_path / "state" / LAST_REQUESTS).write_text(json.dumps(earlier), encoding="utf-8")
    _, server, _, tools = world(tmp_path, {paper_url(A): atom(ENTRY_A), fulltext_url(A): html("One.")}, clock)
    tools["get_paper"]({"arxiv_id": A})
    tools["fetch_fulltext"]({"arxiv_id": A})
    assert clock.sleeps == [2.0, 3.0]  # 3 s after the API request 1 s ago, 15 s after the page 10 s ago
    assert server.times(API) == [1_000_002.0] and server.times(HTML) == [1_000_005.0]


def test_an_unreadable_state_makes_the_first_request_wait_its_whole_interval(tmp_path):
    clock = FakeClock()
    state = tmp_path / "state" / LAST_REQUESTS
    state.parent.mkdir()
    state.write_text("not json", encoding="utf-8")
    _, _, _, tools = world(tmp_path, {paper_url(A): atom(ENTRY_A)}, clock)
    tools["get_paper"]({"arxiv_id": A})
    assert clock.sleeps == [3.0]
    assert json.loads(state.read_text(encoding="utf-8")) == {API: clock.now}  # written whole again


def test_a_state_from_a_clock_set_back_delays_a_run_by_at_most_the_interval_and_the_slack(tmp_path):
    clock = FakeClock()
    state = tmp_path / "state" / LAST_REQUESTS
    state.parent.mkdir()
    state.write_text(json.dumps({API: clock.now + 1000.0}), encoding="utf-8")  # a request "to come"
    _, _, _, tools = world(tmp_path, {paper_url(A): atom(ENTRY_A)}, clock)
    tools["get_paper"]({"arxiv_id": A})
    assert sum(clock.sleeps) == pytest.approx(3.0 + COOL_DOWN_SLACK)


# ---- one today --------------------------------------------------------------------------------------------


def test_today_reaches_the_prompt_check_call_the_search_window_and_the_note(tmp_path):
    _, server, store, tools = world(tmp_path, {SEARCH: atom(ENTRY_A), paper_url(A): atom(ENTRY_A)})
    model = ScriptedModel([
        turn(call("search_arxiv", "c1", query="speculative decoding", date_from="2026-11-01")),
        turn(call("search_arxiv", "c2", query="speculative decoding", date_from="2026-11-11")),  # after today
        turn(call("write_note", "c3", arxiv_id=A, mode="scaffold")),
        turn(content=f"I wrote a note on {A}."),
    ])
    lines: list[str] = []
    result = run_live_turn("帮我找 speculative decoding 的新论文并记笔记。", built=Built(model, None, {}),
                           tools=tools, today=TODAY, out=lines.append)
    transcript = result.transcript
    assert result.error is None and transcript.status == "answered" and transcript.today == TODAY
    assert transcript.messages[0] == {"role": "system", "content": system_prompt(TODAY)}
    assert "TO+202611102359]" in server.urls()[0]  # the search window ends today
    refused = transcript.steps[1]["attempts"][0]["issues"][0]["messages"]
    assert refused == ["date_from: 2026-11-11 is after today 2026-11-10"]  # check_call's today
    note = (store.root / f"{A}-1.md").read_text(encoding="utf-8")
    assert note.startswith(f"### {TODAY} {TITLE_A}\n")
    assert server.urls()[1:] == []  # the note's paper came from the search: no lookup
    assert all(line.isascii() for line in lines) and any("tool write_note: note" in line for line in lines)

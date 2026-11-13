"""research_agent.live.web offline, over 127.0.0.1 only (tests/support/web_fakes.py: a browser over
http.client, and turns that the real run_live_turn runs over a scripted model). The page answers only its
exact Host, its API only the cookie its launch link sets, and its POSTs only its own origin, with JSON of at
most 64 KiB; every response carries the security headers and no cross-origin permission, and a request
without an HTTP version gets no answer at all; a note id is checked before any file is opened; one turn runs
at a time; after a clarification the next message is its answer, twice at most; a message is sent on one line,
and one the console takes for a command, or one UTF-8 cannot encode, is refused before anything is queued;
every value reaches the page as JSON text UTF-8 can encode, whatever the model's arguments hold; the stream
carries whole exchanges, from where a reconnecting page left off or from a snapshot, and a fault ends it with
no second answer inside it; quitting says bye to every stream and frees the port, after the turn that runs,
lets go of connections that never sent a request, waits on no page that stopped reading, and still delivers
the reply to the quit itself; a stream whose page went away ends; a port in use is refused, Windows'
reuse-address trap included, and so is port 80; the page's static files load nothing else and insert no HTML
but the server's."""

from __future__ import annotations

import builtins
import importlib.util
import io
import json
import os
import re
import socket
import threading
import time
from contextlib import contextmanager
from html.parser import HTMLParser
from pathlib import Path

import pytest
from support.fakes import ScriptedModel, call, turn
from support.web_fakes import HOST, TIMEOUT, TODAY, A, B, Page, ScriptedTurns

from research_agent.agent.prompts import system_prompt
from research_agent.agent.types import ToolOutcome
from research_agent.live import web
from research_agent.live.render import render_html
from research_agent.live.session import MAX_FOLLOW_UPS, LiveTurn, pack_follow_up
from research_agent.notes import NoteStore, note_tools

REPO = Path(__file__).resolve().parents[2]
PAPER = {"arxiv_id": A, "title": "Adaptive Draft Length", "published": "2026-11-02",
         "abstract": "Speculative decoding drafts tokens. The draft length adapts."}


@contextmanager
def serving(turns: ScriptedTurns, store: NoteStore | None = None, *, system: dict | None = None,
            launch: bool = True, **options):
    """turns served by web.serve in a helper thread, the test being the page's user (a Page, launched unless
    launch is False); afterwards the run is ended, the turn let go and the serving thread gone."""
    site = web.Site(0, **options)
    printed: list[str] = []
    thread = threading.Thread(target=web.serve, args=(site, turns, store), daemon=True,
                              kwargs={"system": system or {"model": "scripted"}, "browse": None,
                                      "out": printed.append, "poll": 0.05})
    thread.start()
    page = Page(site.port)
    try:
        if launch:
            assert page.launch(site.launch_url).status == 303
        yield site, page, printed
    finally:
        site.state.quit()
        if turns.gate is not None:
            turns.gate.set()
        thread.join(TIMEOUT)
        site.close()
    assert not thread.is_alive()


def idle(tmp_path, *turns_given, **options) -> ScriptedTurns:
    return ScriptedTurns(ScriptedModel(list(turns_given)), tmp_path / "runs", **options)


def eventually(check, timeout: float = TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while not check():
        assert time.monotonic() < deadline, "the condition never held"
        time.sleep(0.01)


def request_threads(port: int) -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name == f"page-{port}-request"]


def started(site: web.Site, turns, store: NoteStore | None = None) -> threading.Thread:
    """web.serve of turns over site in a helper thread, already started."""
    thread = threading.Thread(target=web.serve, args=(site, turns, store), daemon=True,
                              kwargs={"system": {}, "browse": None, "out": lambda line: None, "poll": 0.05})
    thread.start()
    return thread


# ---- who may ask ------------------------------------------------------------------------------------------


def test_only_the_exact_host_is_answered(tmp_path):
    with serving(idle(tmp_path)) as (site, page, _):
        port = site.port
        hosts = [f"localhost:{port}", f"LOCALHOST:{port}", f"127.0.0.1:{port + 1}", "127.0.0.1",
                 f"[::1]:{port}", "example.com", f"example.com:{port}", f"127.0.0.1:{port}.example.com",
                 f"127.0.0.2:{port}", ""]
        paths = ["/", f"/?t={site.launch_token}", "/static/app.js", "/api/state", "/api/notes", "/absent"]
        for host in hosts:
            for path in paths:
                reply = page.get(path, headers={"Host": host})
                assert reply.status == 403 and reply.header("Set-Cookie") is None, (host, path)
            assert page.post("/api/ask", {"text": "hello"}, headers={"Host": host}).status == 403, host
        assert page.get("/api/state", headers={"Host": None}).status == 403  # no Host at all
        assert page.get("/api/state", extra=(("Host", "example.com"),)).status == 403  # a second one
        assert page.state()["exchanges"] == [] and page.get("/").status == 200


def test_the_launch_link_sets_the_cookie_that_the_page_and_the_api_need(tmp_path, capsys):
    with serving(idle(tmp_path), launch=False) as (site, page, _):
        refused = page.get("/")
        assert refused.status == 403 and "控制台" in refused.text and refused.header("Set-Cookie") is None
        assert page.get("/api/state").status == 403 and page.get("/api/events").status == 403
        token = site.launch_token
        for link in (f"/?t={token[:-1]}", "/?t=", f"/?t={site.cookie_token}", f"/?t={token}&t={token}",
                     f"/?token={token}", f"/static/app.js?t={token}"):
            assert page.get(link, headers={"Cookie": None}).header("Set-Cookie") is None, link
            assert page.get(link, headers={"Cookie": None}).status in (200, 403), link
        launched = page.launch(site.launch_url)
        assert launched.status == 303 and launched.header("Location") == "/"
        cookie, _, attributes = launched.header("Set-Cookie").partition(";")
        assert cookie == f"raa_{site.port}={site.cookie_token}" and site.cookie_token != token
        assert sorted(part.strip() for part in attributes.split(";")) == ["HttpOnly", "Path=/",
                                                                           "SameSite=Strict"]
        served = page.get("/")
        assert served.status == 200 and served.body == (web.PAGE / "index.html").read_bytes()
        assert served.header("Content-Type") == "text/html; charset=utf-8"
        assert page.get("/api/state").status == 200
        assert page.get("/api/state", headers={"Cookie": f"a=1; {page.cookie}; b=2"}).status == 200
        port, cookie_token = site.port, site.cookie_token
        for wrong in (f"raa_{port}=", f"raa_{port}={token}", f"raa_{port + 1}={cookie_token}",
                      f"other={cookie_token}", f"raa_{port}={cookie_token}x", "", "garbage"):
            assert page.get("/api/state", headers={"Cookie": wrong}).status == 403, wrong
            assert page.get("/", headers={"Cookie": wrong}).status == 403, wrong
        assert Page(site.port).launch(site.launch_url).status == 303  # the link holds for the whole run
    assert capsys.readouterr() == ("", "")  # no request is logged: its line can hold the launch token


def test_a_post_from_anywhere_but_the_page_is_refused_and_changes_nothing(tmp_path):
    with serving(idle(tmp_path)) as (site, page, _):
        port = site.port
        origins = [None, "null", f"http://127.0.0.1:{port + 1}", "http://example.com", f"http://localhost:{port}",
                   f"https://127.0.0.1:{port}", f"http://127.0.0.1:{port}/"]
        refusals = [*[({"Origin": origin}, 403) for origin in origins],
                    *[({"Sec-Fetch-Site": site}, 403) for site in ("same-site", "cross-site", "none")],
                    ({"Cookie": None}, 403),
                    *[({"Content-Type": kind}, 415) for kind in (None, "text/plain",
                                                                 "application/x-www-form-urlencoded",
                                                                 "multipart/form-data; boundary=x")]]
        for headers, status in refusals:
            for path in ("/api/ask", "/api/new", "/api/quit"):
                assert page.post(path, {"text": "hello"}, headers=headers).status == status, (headers, path)
        a_second_origin = page.post("/api/ask", {"text": "hello"}, extra=(("Origin", "null"),))
        assert a_second_origin.status == 403
        bodies = [(b"text=hello", 400), (b"\xef\xbb\xbf{}", 400), (b"\xff", 400), (b"[" * 20_000, 400),
                  (b'["hello"]', 400), (b'{"text": 5}', 400), (b'{"text": null}', 400), (b"{}", 400),
                  (b'{"text": " \\n\\t "}', 400), (b'{"text": "%s"}' % (b"x" * (web.MAX_TEXT + 1)), 413)]
        for body, status in bodies:
            reply = page.post("/api/ask", raw=body)
            assert reply.status == status and isinstance(reply.json()["error"], str), body[:20]
        too_large = page.post("/api/ask", raw=b"", headers={"Content-Length": str(web.MAX_BODY + 1)})
        assert too_large.status == 413  # refused on its length, before a byte of it is read
        for headers in ({"Content-Length": None}, {"Content-Length": "-1"}, {"Content-Length": "1e3"},
                        {"Content-Length": None, "Transfer-Encoding": "chunked"}):
            assert page.post("/api/ask", raw=b"", headers=headers).status == 411, headers
        assert page.get("/api/state", headers={"Sec-Fetch-Site": "cross-site"}).status == 403
        state = page.state()
        assert state["exchanges"] == [] and state["status"] == {"phase": "idle", "pending": None}
        padded = b'{"text": "hello%s"}' % (b" " * (web.MAX_BODY - len(b'{"text": "hello"}')))
        assert len(padded) == web.MAX_BODY  # 64 KiB at most is accepted
        assert page.post("/api/ask", raw=padded).status == 202


def test_every_response_has_the_security_headers_and_none_that_permits_another_origin(tmp_path):
    with serving(idle(tmp_path), launch=False) as (site, page, _):
        replies = [page.get("/"), page.launch(site.launch_url), page.get("/"), page.get("/static/app.js"),
                   page.get("/static/app.css"), page.get("/api/state"), page.get("/api/notes"),
                   page.get(f"/api/notes/{A}-1"), page.get("/api/notes/x"), page.get("/absent"),
                   page.get("/api/absent"), page.request("PUT", "/api/ask", b"{}"),
                   page.request("DELETE", "/"),
                   page.request("OPTIONS", "/api/ask", headers={"Origin": "http://example.com",
                                                                "Access-Control-Request-Method": "POST"}),
                   page.request("HEAD", "/"), page.request("BREW", "/"), page.get("/api/ask"),
                   page.post("/api/state"), page.get("/", headers={"Host": "localhost"}),
                   page.post("/api/ask", {"text": " "}),
                   page.post("/api/ask", {"text": "x"}, headers={"Origin": "null"}),
                   page.get(f"/?t={site.launch_token}x")]
        with page.stream() as stream:
            replies.append(stream)
        statuses = {reply.status for reply in replies}
        assert {200, 303, 400, 403, 404, 405} <= statuses
        for reply in replies:
            headers = {name.lower(): value for name, value in reply.headers}
            for name, value in web.SECURITY_HEADERS.items():
                assert headers.get(name.lower()) == value, (reply.status, name)
            assert not [name for name in headers if name.startswith("access-control-")], reply.status
        assert web.SECURITY_HEADERS == {
            "Content-Security-Policy": "default-src 'none'; script-src 'self'; style-src 'self'; "
                                       "connect-src 'self'; img-src 'self'; base-uri 'none'; "
                                       "form-action 'none'; frame-ancestors 'none'",
            "X-Frame-Options": "DENY", "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store",
            "Referrer-Policy": "same-origin"}
        assert page.request("HEAD", "/").body == b""


def test_other_methods_are_405_and_other_paths_404(tmp_path):
    with serving(idle(tmp_path)) as (_, page, _):
        for method, path in [("PUT", "/api/ask"), ("DELETE", "/"), ("PATCH", "/api/state"), ("OPTIONS", "/"),
                             ("BREW", "/api/notes"), ("GET", "/api/ask"), ("GET", "/api/quit"),
                             ("POST", "/api/state"), ("POST", "/"), ("POST", "/static/app.js")]:
            assert page.request(method, path, b"{}").status == 405, (method, path)
        for path in ("/absent", "/static/", "/static/app.js/", "/static/../page/app.js", "/index.html",
                     "/api/", "/api/absent", "/page/app.js", "/static/APP.JS"):
            assert page.get(path).status == 404, path
        assert page.post("/api/absent").status == 404
        with socket.create_connection((HOST, page.port), timeout=TIMEOUT) as raw:  # what http.client hides
            raw.sendall(f"HEAD / HTTP/1.0\r\nHost: {page.host}\r\n\r\n".encode("ascii"))
            answer = b"".join(iter(lambda: raw.recv(4096), b""))
        assert answer.startswith(b"HTTP/1.0 405 ") and answer.endswith(b"\r\n\r\n")  # no body after HEAD


def received(connection: socket.socket) -> bytes:
    """What connection receives until the server closes it (a reset, as when it closes with bytes of the
    request unread, included)."""
    chunks = []
    try:
        while chunk := connection.recv(4096):
            chunks.append(chunk)
    except (ConnectionResetError, ConnectionAbortedError):
        pass
    return b"".join(chunks)


def test_a_request_without_an_http_version_gets_no_answer(tmp_path):
    with serving(idle(tmp_path)) as (_, page, _):
        headers = f"Host: {page.host}\r\nCookie: {page.cookie}\r\nSec-Fetch-Site: same-origin\r\n\r\n"
        for request in ("GET /api/state\r\n", "GET /\r\n", f"GET /api/state\r\n{headers}", "GET /\r\n\r\n"):
            with socket.create_connection((HOST, page.port), timeout=TIMEOUT) as raw:
                raw.sendall(request.encode("ascii"))
                assert received(raw) == b"", request  # HTTP/0.9: no status line or header could answer it
        assert page.get("/api/state").status == 200


def test_port_80_is_refused_before_anything_is_bound():
    """A browser leaves port 80, http's own, out of the Host header and the Origin it sends, which the page
    compares whole with 127.0.0.1:<port>: no request of the page would pass."""
    with pytest.raises(ValueError, match="80"):
        web.Site(80)


# ---- the notes --------------------------------------------------------------------------------------------

@contextmanager
def opened_files(monkeypatch):
    """The paths every thread of this process opens meanwhile through open, io.open (Path.read_text's and
    read_bytes') or os.open."""
    files: list[str] = []
    io_open, os_open = io.open, os.open

    def recorded_io_open(file, *args, **kwargs):
        if not isinstance(file, int):
            files.append(os.fsdecode(os.fspath(file)))
        return io_open(file, *args, **kwargs)

    def recorded_os_open(path, *args, **kwargs):
        files.append(os.fsdecode(os.fspath(path)))
        return os_open(path, *args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(io, "open", recorded_io_open)
        patched.setattr(builtins, "open", recorded_io_open)
        patched.setattr(os, "open", recorded_os_open)
        yield files


def test_a_note_id_is_checked_before_any_file_is_opened(tmp_path, monkeypatch):
    store = NoteStore(tmp_path / "notes")
    note = store.write(PAPER, {"mode": "scaffold"}, ["Speculative decoding drafts tokens."], TODAY)
    with serving(idle(tmp_path), store) as (_, page, _):
        listed = page.get("/api/notes").json()["notes"]
        assert listed == [{"note_id": f"{A}-1", "arxiv_id": A, "title": PAPER["title"], "created": TODAY,
                           "tags": [], "url": f"https://arxiv.org/abs/{A}"}] and note["note_id"] == f"{A}-1"
        shown = page.get(f"/api/notes/{A}-1")
        assert shown.status == 200 and shown.json() == {
            "note_id": f"{A}-1", "html": render_html((store.root / f"{A}-1.md").read_text(encoding="utf-8"))}
        link = f'<a href="https://arxiv.org/abs/{A}" target="_blank" rel="noopener noreferrer">'
        assert link in shown.json()["html"]
        bad = ["..%2f..%2fsecret", "%2e%2e%2fnotes.sqlite", "..", "%2e%2e", "../notes.sqlite", f"{A}-0",
               f"{A}-01", f"{A}-1.md", f"{A}", f"{A}-", f"{A}-1%00", f"{A}-1%2e", f"{A}-1/", f"{A}-1/x",
               "%32611.00001-1", "%EF%BC%92%EF%BC%96%EF%BC%91%EF%BC%91.00001-1", f"{A}-1;x", f"{A}-1%20",
               "notes.sqlite", "", "2611.000001-1", "26110.0001-1", f"x{A}-1", "C:%5Cnotes.sqlite"]
        page.get(f"/api/notes/{bad[0]}")  # the refusal's code is loaded before the files are counted
        with opened_files(monkeypatch) as opened:
            for note_id in bad:
                reply = page.get(f"/api/notes/{note_id}")
                assert reply.status == 400 and reply.json() == {"error": "bad_note_id"}, note_id
        assert [path for path in opened if Path(path).resolve().is_relative_to(tmp_path.resolve())] == []
        with opened_files(monkeypatch) as counted:  # the count sees a file the server opens
            assert page.get(f"/api/notes/{A}-1").status == 200
        assert str(store.root.resolve() / f"{A}-1.md") in counted
        for absent in (f"{B}-1", f"{A}-2", "2611.0001-1"):
            assert page.get(f"/api/notes/{absent}").status == 404, absent
        outside = tmp_path / "outside.md"
        outside.write_text("# not a note\n", encoding="utf-8")
        try:
            os.symlink(outside, store.root / f"{B}-1.md")
        except OSError:  # no right to make links here: the path check is left to the ids above
            pass
        else:
            assert page.get(f"/api/notes/{B}-1").status == 404  # it resolves outside the notes root
        for query, status in (("", 200), ("?limit=1", 200), ("?limit=200", 200), ("?limit=0", 400),
                              ("?limit=201", 400), ("?limit=x", 400), ("?limit=-1", 400),
                              ("?limit=1&limit=2", 400), ("?limit=%201", 400)):
            assert page.get(f"/api/notes{query}").status == status, query


def test_without_a_notes_library_the_notes_are_unavailable(tmp_path):
    with serving(idle(tmp_path), None) as (_, page, _):
        assert page.get("/api/notes").status == 503 and page.get(f"/api/notes/{A}-1").status == 503


# ---- the turns --------------------------------------------------------------------------------------------


def test_one_turn_runs_at_a_time_and_a_second_message_or_a_new_question_waits_for_it(tmp_path):
    gate = threading.Event()
    turns = idle(tmp_path, turn(content="好的。"), turn(content="又一个。"), gate=gate)
    with serving(turns) as (site, page, printed):
        first = page.ask("第一个请求")
        assert first.status == 202 and first.json()["exchange"] == {
            "id": 1, "request": "第一个请求", "sent": "第一个请求", "follow_up": 0, "state": "queued",
            "steps": [], "result": None}
        assert turns.started.wait(TIMEOUT)
        state = page.until(lambda state: state["exchanges"][0]["state"] == "running")
        assert state["status"] == {"phase": "running", "pending": None}
        busy = page.ask("第二个请求")
        assert busy.status == 409 and busy.json() == {"error": "busy"}
        assert page.post("/api/new").status == 409 and page.post("/api/new").json() == {"error": "busy"}
        gate.set()
        assert page.done(1)["result"]["status"] == "answered"
        assert page.ask("第二个请求").status == 202 and page.done(2)["result"]["html"] == "<p>又一个。</p>"
        assert turns.sent == [("第一个请求", 0), ("第二个请求", 0)]
        assert printed[0] == f"page: {site.launch_url}"


def test_after_a_clarification_the_next_message_is_its_answer_twice_at_most(tmp_path):
    asked = [turn(call("ask_clarification", f"q{n}", question="哪个方面?", missing=["topic"]))
             for n in range(4)]
    turns = idle(tmp_path, *asked, turn(content="好的。"))
    request = "帮我找几篇论文。"
    first_answer = pack_follow_up(request, "speculative decoding")
    with serving(turns) as (_, page, _):
        page.ask(request)
        assert page.done(1)["result"]["after"] == "answer"
        assert page.state()["status"] == {"phase": "idle", "pending": {"exchange": 1, "follow_ups": 0}}
        page.ask("speculative decoding")
        second = page.done(2)
        assert (second["request"], second["sent"], second["follow_up"]) == ("speculative decoding",
                                                                             first_answer, 1)
        assert page.state()["status"]["pending"] == {"exchange": 2, "follow_ups": 1}
        page.ask(" 2026 年\n以后 ")
        third = page.done(3)
        assert third["sent"] == pack_follow_up(first_answer, "2026 年 以后")
        assert third["follow_up"] == MAX_FOLLOW_UPS and third["result"]["after"] == "restate"
        assert page.state()["status"]["pending"] is None
        page.ask("Find papers on RAG.")
        fourth = page.done(4)
        assert (fourth["sent"], fourth["follow_up"]) == ("Find papers on RAG.", 0)
        assert fourth["result"]["after"] == "answer"
        dropped = page.post("/api/new")
        assert dropped.status == 200 and dropped.json() == {"status": {"phase": "idle", "pending": None}}
        page.ask("since 2025")
        fifth = page.done(5)
        assert (fifth["sent"], fifth["follow_up"], fifth["result"]["after"]) == ("since 2025", 0, None)
    assert [sent for sent, _ in turns.sent] == [request, first_answer, third["sent"], "Find papers on RAG.",
                                                "since 2025"]


def test_a_message_is_sent_on_one_line(tmp_path):
    model = ScriptedModel([turn(content="ok")])
    with serving(ScriptedTurns(model, tmp_path / "runs")) as (_, page, _):
        reply = page.ask(" Find\tpapers\n on  RAG.\r\n  ")
        assert reply.status == 202 and reply.json()["exchange"]["request"] == "Find papers on RAG."
        assert page.done(1)["sent"] == "Find papers on RAG."
    assert model.contexts[0] == [{"role": "system", "content": system_prompt(TODAY)},
                                 {"role": "user", "content": "Find papers on RAG."}]


def test_what_the_console_takes_for_a_command_is_refused_and_never_sent(tmp_path):
    asked = turn(call("ask_clarification", "q1", question="哪个方面?", missing=["topic"]))
    turns = idle(tmp_path, asked, turn(content="好的。"))
    request = "帮我找几篇论文。"
    answered = pack_follow_up(request, "quit smoking studies")
    with serving(turns) as (_, page, _):
        page.ask(request)
        assert page.done(1)["result"]["after"] == "answer"
        pending = {"phase": "idle", "pending": {"exchange": 1, "follow_ups": 0}}
        for text in ("/new 别的主题", "/new", "/quit", "/help me", "/", "/NEW x", "exit", "QUIT", " Exit\t"):
            reply = page.ask(text)
            assert reply.status == 400 and reply.json() == {"error": "command"}, text
            assert page.state()["status"] == pending, text  # the request still awaits its answer
        assert page.ask("quit smoking studies").status == 202  # it only starts with the word
        assert page.done(2)["sent"] == answered
    assert turns.sent == [(request, 0), (answered, 1)]


def test_a_message_utf8_cannot_encode_is_refused_before_anything_is_queued(tmp_path):
    turns = idle(tmp_path, turn(content="好的。"))
    emoji = "find \U0001f600 papers"  # a whole surrogate pair is one character like any other
    with serving(turns) as (_, page, _):
        halves = (b'{"text": "find \\ud83d papers"}', b'{"text": "\\udc80"}', b'{"text": "a\\ud83d\\ud83db"}')
        for body in halves:
            reply = page.post("/api/ask", raw=body)  # what JSON.stringify sends for half an emoji
            assert reply.status == 400 and reply.json() == {"error": "not_text"}, body
        state = page.state()
        assert state["exchanges"] == [] and state["status"] == {"phase": "idle", "pending": None}
        assert page.ask(emoji).status == 202 and page.done(1)["sent"] == emoji
    assert turns.sent == [(emoji, 0)] and turns.files.turns == 1
    chat = web.ChatState(web.Hub())
    with pytest.raises(web.NotText):
        chat.submit("a\ud800b")
    assert chat.queue.empty() and chat.snapshot()["exchanges"] == []


def test_a_turn_that_raised_is_shown_with_its_error_and_the_page_goes_on(tmp_path):
    turns = idle(tmp_path, RuntimeError("the server went away"), turn(content="ok"))
    with serving(turns) as (_, page, _):
        page.ask("hello")
        result = page.done(1)["result"]
        assert result["status"] is None and result["error"] == "RuntimeError: the server went away"
        assert result["html"] == "" and result["turn"] == 1 and result["after"] is None
        page.ask("again")
        assert page.done(2)["result"]["status"] == "answered"


# ---- the stream -------------------------------------------------------------------------------------------

NOTED = f"笔记写好了:**{A}**,另见 2611.00009。"


def note_turns(tmp_path, store: NoteStore) -> ScriptedTurns:
    """A turn that reads a paper, writes a note on it and answers, over the real note tools."""
    tools = {**note_tools(store, {A: PAPER}.get, TODAY), "get_paper": lambda args: ToolOutcome(dict(PAPER))}
    return idle(tmp_path, turn(call("get_paper", "g1", arxiv_id=A)),
                turn(call("write_note", "w1", arxiv_id=A)), turn(content=NOTED), tools=tools)


def test_the_stream_carries_whole_exchanges_and_a_page_that_comes_back_late_gets_a_snapshot(tmp_path):
    store = NoteStore(tmp_path / "notes")
    window = 40  # more than the turn's events: the stream that watches the turn never falls out of it
    with serving(note_turns(tmp_path, store), store, window=window, ping=0.05) as (_, page, _):
        start = page.state()
        assert start == {"seq": start["seq"], "system": {"model": "scripted"},
                         "status": {"phase": "idle", "pending": None}, "exchanges": []}
        with page.stream(start["seq"]) as stream:
            assert stream.status == 200
            assert dict(stream.headers)["Content-Type"] == "text/event-stream; charset=utf-8"
            assert stream.next() == {"retry": 3000}
            page.ask(f"给 {A} 写笔记")
            read = stream.until(lambda event: event.get("event") == "notes")
        events = [event for event in read if "id" in event]
        first = start["seq"] + 1
        assert [event["id"] for event in events] == list(range(first, first + len(events)))
        exchanges = [event["data"] for event in events if event["event"] == "exchange"]
        final = page.state()["exchanges"][0]
        assert exchanges[-1] == final and all(set(exchange) == set(final) for exchange in exchanges)
        assert [exchanges[0]["state"], exchanges[1]["state"], final["state"]] == ["queued", "running", "done"]
        steps = [(step["step"], step["state"], [(tool["name"], tool["state"], tool["ids"])
                                                for tool in step["tools"]]) for step in final["steps"]]
        assert steps == [(1, "done", [("get_paper", "done", [A])]),
                         (2, "done", [("write_note", "done", [A])]), (3, "done", [])]
        written = {"id": "w1", "name": "write_note", "arguments": {"arxiv_id": A}}
        assert final["steps"][1]["calls"] == [written]
        assert final["steps"][2]["answer_chars"] == len(NOTED)
        result = final["result"]
        assert result == {"status": "answered", "error": None, "seconds": result["seconds"], "today": TODAY,
                          "html": render_html(NOTED), "after": None, "turn": 1,
                          "notes": [{"note_id": f"{A}-1", "arxiv_id": A}], "cited": [A, "2611.00009"],
                          "cited_unseen": ["2611.00009"]}
        assert events[-1]["data"] == {"written": [{"note_id": f"{A}-1", "arxiv_id": A}]}
        phases = [event["data"]["phase"] for event in events if event["event"] == "status"]
        assert phases == ["running", "idle"]
        for _ in range(window):  # a status event each: the turn's events leave the window
            assert page.post("/api/new").status == 200
        last = page.state()["seq"]
        assert last == events[-1]["id"] + window
        with page.stream(last - 2) as resumed:  # still in the window: what it missed, in order
            assert resumed.next() == {"retry": 3000}
            assert [resumed.next()["id"], resumed.next()["id"]] == [last - 1, last]
            assert resumed.next() == {"comment": "ping"}
        for late in (page.stream(start["seq"] + 1), page.stream(), page.stream(last + 5),
                     page.stream(start["seq"], headers={"Last-Event-ID": "x"})):
            with late:  # fallen out of the window, new, from the future or unreadable: a snapshot first
                assert late.next() == {"retry": 3000}
                assert late.next() == {"id": last, "event": "snapshot", "data": page.state()}
                assert late.next() == {"comment": "ping"}
        with page.stream(start["seq"] + 1, headers={"Last-Event-ID": str(last)}) as preferred:
            assert preferred.next() == {"retry": 3000}
            assert preferred.next() == {"comment": "ping"}  # Last-Event-ID wins over since: nothing missed


def test_quit_says_bye_to_every_stream_and_frees_the_port(tmp_path):
    site = web.Site(0, ping=0.05)
    thread = started(site, idle(tmp_path))
    page = Page(site.port)
    page.launch(site.launch_url)
    seq = page.state()["seq"]
    streams = [page.stream(seq), page.stream(seq)]
    try:
        eventually(lambda: site.hub.subscribers == 2)
        reply = page.post("/api/quit")
        assert reply.status == 202 and reply.json() == {"ending": "now"}
        for stream in streams:
            named = [(event["event"], event["data"]) for event in stream.rest() if "event" in event]
            assert named == [("status", {"phase": "ending", "pending": None}),
                             ("status", {"phase": "ended", "pending": None}), ("bye", {"reason": "ended"})]
    finally:
        for stream in streams:
            stream.close()
    thread.join(TIMEOUT)
    assert not thread.is_alive() and site.hub.subscribers == 0 and request_threads(site.port) == []
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection((HOST, site.port), timeout=TIMEOUT).close()
    web.Site(site.port).close()  # the port is free
    site.close()  # again: nothing more happens


def test_quit_during_a_turn_ends_the_run_after_it(tmp_path):
    gate = threading.Event()
    turns = idle(tmp_path, turn(content="好的。"), gate=gate)
    site = web.Site(0)
    thread = started(site, turns)
    page = Page(site.port)
    try:
        page.launch(site.launch_url)
        page.ask("请求")
        assert turns.started.wait(TIMEOUT)
        with page.stream(page.state()["seq"]) as stream:
            reply = page.post("/api/quit")
            assert reply.status == 202 and reply.json() == {"ending": "after_turn"}
            assert page.post("/api/quit").json() == {"ending": "after_turn"}
            refused = page.ask("another")
            assert refused.status == 409 and refused.json() == {"error": "ending"}
            assert page.state()["status"]["phase"] == "ending" and thread.is_alive()
            gate.set()
            events = [event for event in stream.rest() if "event" in event]
    finally:
        gate.set()
        site.state.quit()
        thread.join(TIMEOUT)
    assert not thread.is_alive() and turns.files.turns == 1
    assert events[-1] == {"id": events[-1]["id"], "event": "bye", "data": {"reason": "ended"}}
    done = [event["data"] for event in events if event["event"] == "exchange"][-1]
    assert done["state"] == "done" and done["result"]["status"] == "answered"


def test_closing_lets_go_of_a_connection_that_never_sent_its_request(tmp_path):
    site = web.Site(0)
    thread = started(site, idle(tmp_path))
    spare = socket.create_connection((HOST, site.port), timeout=TIMEOUT)  # as a browser keeps one ready
    try:
        eventually(lambda: len(request_threads(site.port)) == 1)
        begun = time.monotonic()
        assert site.state.quit() == "now"
        thread.join(TIMEOUT)
        assert not thread.is_alive() and time.monotonic() - begun < web.HANDLER_TIMEOUT / 2
        assert request_threads(site.port) == []  # its thread did not wait out the request's timeout
    finally:
        spare.close()


def test_a_ctrl_c_in_a_turn_ends_the_page_as_interrupted_and_goes_on_up(tmp_path):
    def interrupted(sent: str, follow_up: int, events) -> tuple:
        raise KeyboardInterrupt

    site, raised = web.Site(0), []

    def serving_thread() -> None:
        try:
            web.serve(site, interrupted, None, system={}, browse=None, out=lambda line: None, poll=0.05)
        except KeyboardInterrupt:
            raised.append(True)

    thread = threading.Thread(target=serving_thread, daemon=True)
    thread.start()
    page = Page(site.port)
    page.launch(site.launch_url)
    with page.stream(page.state()["seq"]) as stream:
        assert page.ask("hello").status == 202
        events = [event for event in stream.rest() if "event" in event]
    thread.join(TIMEOUT)
    assert not thread.is_alive() and raised == [True]
    ended = {"phase": "ended", "pending": None}
    assert events[-3]["event"] == "exchange" and events[-3]["data"]["state"] == "interrupted"
    assert events[-2:] == [{"id": events[-2]["id"], "event": "status", "data": ended},
                           {"id": events[-1]["id"], "event": "bye", "data": {"reason": "interrupted"}}]


def test_the_end_of_the_run_marks_the_turn_it_cut_off_interrupted_and_no_other():
    start = {"kind": "step_start", "step": 1, "attempt": 1}
    tool_start = {"kind": "tool_start", "name": "get_paper"}
    step = {"kind": "step", "step": 1, "attempt": 1, "seconds": 2.5, "votes": None, "thinking": None,
            "reasoning": None, "rules": [], "answer_chars": None,
            "calls": [{"id": "g1", "name": "get_paper", "arguments": {"arxiv_id": A}}]}
    tool = {"kind": "tool", "name": "get_paper", "summary": f"found {A}", "error": None, "ids": [A],
            "seconds": 1.0, "waited": 0.0}
    cases = (([], []),  # begun, no event yet
             ([start], [("interrupted", [])]),  # in the model's step
             ([start, step, tool_start], [("done", ["interrupted"])]),  # in a tool call
             ([start, step, tool_start, tool, {"kind": "step_start", "step": 2, "attempt": 1}],
              [("done", ["done"]), ("interrupted", [])]))  # in the next step
    for events, cut in cases:
        state = web.ChatState(web.Hub())
        first = state.submit("hello")["id"]
        state.begin(first)
        state.finish(first, LiveTurn("hello", TODAY, None, None, 1.5), None)
        number = state.submit("again")["id"]
        state.begin(number)
        for event in events:
            state.apply(number, event)
        seen = state.hub.seq
        state.end()
        done, last = state.snapshot()["exchanges"]
        assert done["state"] == "done" and last["state"] == "interrupted" and last["result"] is None
        assert [(s["state"], [t["state"] for t in s["tools"]]) for s in last["steps"]] == cut
        assert [kind for _, kind, _ in state.hub.after(seen)] == ["exchange", "status"]
        state.end()  # once only
        assert state.hub.seq == seen + 2
    idle = web.ChatState(web.Hub())  # no turn queued or running: the status only
    idle.end()
    assert [kind for _, kind, _ in idle.hub.after(0)] == ["status"]


def test_a_stream_whose_page_went_away_ends_its_thread(tmp_path):
    with serving(idle(tmp_path), ping=0.02) as (site, page, _):
        stream = page.stream(page.state()["seq"])
        eventually(lambda: site.hub.subscribers == 1)
        stream.close()
        eventually(lambda: site.hub.subscribers == 0)
        eventually(lambda: request_threads(site.port) == [])


def test_closing_does_not_wait_on_a_stream_whose_page_stopped_reading(tmp_path):
    site = web.Site(0)
    thread = started(site, idle(tmp_path))
    page = Page(site.port)
    page.launch(site.launch_url)
    stalled = socket.socket()
    stalled.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)  # before it connects: a small window
    try:
        stalled.settimeout(TIMEOUT)
        stalled.connect((HOST, site.port))
        stalled.sendall(f"GET /api/events HTTP/1.0\r\nHost: {page.host}\r\nCookie: {page.cookie}\r\n"
                        "Sec-Fetch-Site: same-origin\r\n\r\n".encode("ascii"))
        eventually(lambda: site.hub.subscribers == 1)
        for _ in range(16):  # far more than the connection's buffers hold, and the page reads none of it
            site.hub.publish("status", {"blob": "x" * 1_000_000})
        time.sleep(0.5)
        begun = time.monotonic()
        assert site.state.quit() == "now"
        thread.join(TIMEOUT)
        assert not thread.is_alive() and request_threads(site.port) == []  # no thread waits on that page
        assert time.monotonic() - begun < web.STREAM_GRACE + 2.0  # the grace for bye, and little more
    finally:
        stalled.close()


def test_the_reply_to_the_quit_that_closes_the_site_still_reaches_its_page(tmp_path, monkeypatch):
    def quit_and_reply_once_severed(handler: web._Handler, site: web.Site, body: bytes) -> None:
        ending = site.state.quit()
        eventually(lambda: site.server.severed)  # the close has reached every connection before the reply
        handler._json(202, {"ending": ending})

    monkeypatch.setattr(web._Handler, "_quit", quit_and_reply_once_severed)
    site = web.Site(0)
    thread = started(site, idle(tmp_path))
    page = Page(site.port)
    page.launch(site.launch_url)
    reply = page.post("/api/quit")
    assert reply.status == 202 and reply.json() == {"ending": "now"}
    thread.join(TIMEOUT)
    assert not thread.is_alive() and request_threads(site.port) == []


def test_a_fault_in_a_stream_ends_it_with_no_second_answer_inside_it(tmp_path, monkeypatch):
    with serving(idle(tmp_path)) as (site, page, _):
        def broken() -> dict:
            raise RuntimeError("a fault of the state")

        monkeypatch.setattr(site.state, "snapshot", broken)
        with page.stream() as stream:  # a new page: its snapshot comes first
            assert stream.status == 200
            assert list(iter(stream.response.readline, b"")) == [b"retry: 3000\n", b"\n"]


def test_a_value_json_cannot_hold_as_it_is_reaches_the_page_as_text_and_the_page_goes_on(tmp_path):
    """json.loads, which reads a model's call arguments, takes NaN, the infinities and a lone surrogate
    escape: the page gets them as text, and every exchange and snapshot still reaches it whole."""
    odd = call("search_arxiv", "c1", raw='{"query": "decoding \\ud83d", "max_results": NaN, '
                                          '"date_from": Infinity, "date_to": -Infinity, "sort_by": 1e400}')
    turns = idle(tmp_path, turn(odd), turn(content="没有找到。"), turn(content="好的。"),
                 tools={"search_arxiv": lambda args: ToolOutcome([])})
    with serving(turns) as (_, page, _):
        with page.stream(page.state()["seq"]) as stream:
            assert page.ask("搜索").status == 202
            read = stream.until(lambda event: event.get("event") == "status"
                                and event["data"]["phase"] == "idle")
        exchanges = [event["data"] for event in read if event.get("event") == "exchange"]
        done = page.done(1)
        assert exchanges[-1] == done and all(set(exchange) == set(done) for exchange in exchanges)
        assert done["steps"][0]["calls"] == [{"id": "c1", "name": "search_arxiv", "arguments": {
            "query": "decoding \ud83d", "max_results": "NaN", "date_from": "Infinity", "date_to": "-Infinity",
            "sort_by": "Infinity"}}]
        assert done["result"]["status"] == "answered" and done["result"]["html"] == "<p>没有找到。</p>"
        assert page.ask("再来一个").status == 202 and page.done(2)["result"]["status"] == "answered"
        with page.stream() as late:  # a page that comes now: the snapshot is the state
            assert late.next() == {"retry": 3000} and late.next()["data"] == page.state()


def test_a_port_in_use_is_refused_even_when_its_holder_allows_address_reuse():
    for reuse in (False, True):
        holder = socket.socket()
        if reuse:  # what HTTPServer does by default, which on Windows lets a second socket bind the port
            holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        holder.bind((HOST, 0))
        holder.listen()
        try:
            with pytest.raises(OSError):
                web.Site(holder.getsockname()[1])
        finally:
            holder.close()
    site = web.Site(0)
    try:
        with pytest.raises(OSError):
            web.Site(site.port)
        intruder = socket.socket()
        intruder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with pytest.raises(OSError):  # nor can another socket take the page's port
            intruder.bind((HOST, site.port))
        intruder.close()
    finally:
        site.close()
    web.Site(site.port).close()  # free once closed


def test_serve_prints_the_link_opens_it_and_a_browser_that_fails_changes_nothing(tmp_path):
    for browse_fails in (False, True):
        site, opened, printed = web.Site(0), [], []

        def browse(url: str, fails: bool = browse_fails, opened: list = opened) -> None:
            opened.append(url)
            if fails:
                raise OSError("no browser here")

        thread = threading.Thread(target=web.serve, args=(site, idle(tmp_path / str(browse_fails)), None),
                                  daemon=True, kwargs={"system": {"model": "m"}, "browse": browse,
                                                       "out": printed.append, "poll": 0.05})
        thread.start()
        eventually(lambda opened=opened: opened)
        assert opened == [site.launch_url] and printed[0] == f"page: {site.launch_url}"
        page = Page.of(site.launch_url)
        assert page.launch(site.launch_url).status == 303 and page.state()["system"] == {"model": "m"}
        page.post("/api/quit")
        thread.join(TIMEOUT)
        assert not thread.is_alive() and site.hub.reason == "ended"  # serve ended the run and closed the page


# ---- the parts --------------------------------------------------------------------------------------------


def test_the_hub_keeps_a_window_of_events_and_says_when_a_stream_fell_out_of_it():
    hub = web.Hub(window=3)
    assert hub.after(0) == [] and hub.after(1) is None and hub.after(-1) is None
    seqs = [hub.publish("exchange", {"id": n}) for n in range(5)]
    assert seqs == [1, 2, 3, 4, 5] and hub.seq == 5
    assert hub.after(2) == [(3, "exchange", '{"id":2}'), (4, "exchange", '{"id":3}'),
                            (5, "exchange", '{"id":4}')]
    assert hub.after(4) == [(5, "exchange", '{"id":4}')] and hub.after(5) == []
    assert hub.after(1) is None and hub.after(0) is None and hub.after(6) is None
    assert hub.publish("status", {"text": "中文", "nan": float("nan")}) == 6  # never raises
    assert hub.after(5)[0][2] == '{"text":"中文","nan":"NaN"}'
    assert hub.publish("status", {"object": object()}) == 7 and '"object"' in hub.after(6)[0][2]
    assert hub.wait(7, 0.01) == []
    hub.close("ended")
    hub.close("interrupted")  # once only
    assert hub.after(7) == [(8, "bye", '{"reason":"ended"}')] and hub.reason == "ended"
    assert hub.wait(8, 5.0) == []  # closed: no wait


def test_every_value_goes_out_as_json_text_that_utf8_can_encode():
    hub = web.Hub()
    half = {"query": "decoding \ud83d", "中文": "\udc80"}
    odd = {"nan": float("nan"), "list": [float("inf"), {"deep": (-float("inf"), 1.5)}], "fine": "中文"}
    looped: list = []
    looped.append(looped)
    for value in (half, odd, {**odd, **half}, looped):
        hub.publish("status", value)
    texts = [text for _, _, text in hub.after(0)]
    for text in texts:
        text.encode("utf-8")  # never raises
    assert json.loads(texts[0]) == half and texts[0].isascii()  # every character but ASCII escaped
    assert texts[1] == '{"nan":"NaN","list":["Infinity",{"deep":["-Infinity",1.5]}],"fine":"中文"}'
    assert json.loads(texts[2]) == {**json.loads(texts[1]), **half} and texts[2].isascii()
    assert json.loads(texts[3])["error"].startswith("not sent: ")  # only a value JSON cannot write at all


def test_the_state_reduces_the_events_of_a_turn_into_its_steps_and_tools():
    state = web.ChatState(web.Hub())
    exchange = state.submit("hello")
    number = exchange["id"]
    assert state.queue.get_nowait() == (number, "hello", 0)
    state.begin(number)
    start = {"kind": "step_start", "step": 1, "attempt": 1}
    step = {"kind": "step", "step": 1, "attempt": 1, "seconds": 2.5, "votes": None, "thinking": None,
            "reasoning": "想一想。", "rules": [], "answer_chars": None,
            "calls": [{"id": "g1", "name": "get_paper", "arguments": {"arxiv_id": A}}]}
    for event in (start, step, {"kind": "tool_start", "name": "get_paper"},
                  {"kind": "tool", "name": "get_paper", "summary": f"found {A}", "error": None, "ids": [A],
                   "seconds": 1.0, "waited": 0.5},
                  {"kind": "tool_start", "name": "search_arxiv"},
                  {"kind": "tool_error", "name": "search_arxiv", "error": "RuntimeError", "seconds": 0.1},
                  {"kind": "step_start", "step": 2, "attempt": 1},
                  {"kind": "step_start", "step": 2, "attempt": 2},
                  {"kind": "step_error", "step": 2, "attempt": 2, "error": "TimeoutError", "seconds": 9.0},
                  {"kind": "unknown"}):
        state.apply(number, event)
    steps = state.snapshot()["exchanges"][0]["steps"]
    tools = [{"name": "get_paper", "state": "done", "summary": f"found {A}", "error": None, "ids": [A],
              "seconds": 1.0, "waited": 0.5},
             {"name": "search_arxiv", "state": "error", "summary": None, "error": "RuntimeError", "ids": [],
              "seconds": 0.1, "waited": None}]
    unsaid = {"seconds": None, "votes": None, "thinking": None, "reasoning": None, "rules": [], "calls": [],
              "answer_chars": None, "error": None, "tools": []}
    assert steps == [
        {**{key: value for key, value in step.items() if key != "kind"}, "state": "done", "error": None,
         "tools": tools},
        {**unsaid, "step": 2, "attempt": 1, "state": "running"},
        {**unsaid, "step": 2, "attempt": 2, "state": "error", "seconds": 9.0, "error": "TimeoutError"}]
    lone = web.ChatState(web.Hub())  # tool events without a step around them (the model's events stopped)
    number = lone.submit("hi")["id"]
    lone.apply(number, {"kind": "tool_start", "name": "list_notes"})
    assert lone.snapshot()["exchanges"][0]["steps"] == []
    seen = state.hub.seq
    state.finish(1, LiveTurn("hello", TODAY, None, "RuntimeError: gone", 1.5), None)
    assert state.snapshot()["exchanges"][0]["result"] == {
        "status": None, "error": "RuntimeError: gone", "seconds": 1.5, "today": TODAY, "html": "",
        "after": None, "turn": None, "notes": [], "cited": [], "cited_unseen": []}
    assert [kind for _, kind, _ in state.hub.after(seen)] == ["exchange", "status"]


def test_the_system_the_page_shows_is_read_from_the_assembly_record():
    record = {"info": {"model": "ft", "generation": {"votes": 5, "reasoning_budget": 1280},
                       "vote_mode": "final", "rules": {"version": "m7-wp3-1"}},
              "props": {"build_info": "b11349-fb4b2737a"}, "frozen": {"sha256": "ab" * 32},
              "server": {"url": "http://127.0.0.1:8080"}}
    assert web.system_info(record, "20261112T000000Z") == {
        "model": "ft", "votes": 5, "vote_mode": "final", "reasoning_budget": 1280,
        "rules_version": "m7-wp3-1", "build": "b11349-fb4b2737a", "frozen_sha256": "ab" * 32,
        "server": "http://127.0.0.1:8080", "run": "20261112T000000Z"}
    empty = {"info": {}, "props": None, "frozen": {}, "server": {}}
    assert web.system_info(empty, None) == dict.fromkeys(
        ("model", "votes", "vote_mode", "reasoning_budget", "rules_version", "build", "frozen_sha256",
         "server", "run"))


# ---- the page's files -------------------------------------------------------------------------------------


class Tags(HTMLParser):
    """Every start tag of a document with its attributes, and the text inside each <script> and <style>."""

    def __init__(self, text: str) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[tuple[str, dict]] = []
        self.inline: list[str] = []
        self._inside: str | None = None
        self.feed(text)
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))
        self._inside = tag if tag in ("script", "style") else None

    def handle_endtag(self, tag: str) -> None:
        self._inside = None

    def handle_data(self, data: str) -> None:
        if self._inside and data.strip():
            self.inline.append(data)


def page_text(name: str) -> str:
    return (web.PAGE / name).read_text(encoding="utf-8")


def test_the_page_loads_only_its_own_files_and_inserts_html_only_from_the_server(tmp_path):
    index, script, style = page_text("index.html"), page_text("app.js"), page_text("app.css")
    tags = Tags(index)
    scripts = [attrs for tag, attrs in tags.tags if tag == "script"]
    assert scripts == [{"src": "/static/app.js", "defer": None}]
    assert [attrs for tag, attrs in tags.tags if tag == "link"] == [{"rel": "stylesheet",
                                                                     "href": "/static/app.css"}]
    assert tags.inline == [] and not [tag for tag, _ in tags.tags if tag in ("style", "iframe", "img", "form",
                                                                             "base", "object", "embed")]
    assert not [(tag, name) for tag, attrs in tags.tags for name in attrs
                if name.lower().startswith("on") or name.lower() in ("style", "srcdoc", "formaction")]
    for text in (index, script, style):  # no URL of another site, written out or protocol-relative
        assert not re.search(r"(?i)\b(?:https?|wss?|ftp)://|\b(?:data|javascript):\S|[\"'(=]\s*//\w", text)
    assert script.count("innerHTML") == 1
    (body,) = re.findall(r"\nfunction setServerHtml\(element, html\) \{\n(.*?)\n\}\n", script, re.DOTALL)
    assert body.strip() == "element.innerHTML = html;"
    assert not re.search(r"\beval\b|\bFunction\b|document\.write|insertAdjacentHTML|outerHTML|"
                         r"setTimeout\(\s*['\"`]|setInterval\(\s*['\"`]|\bimport\(|createContextualFragment|"
                         r"\bsrcdoc\b|DOMParser", script)
    assert "event.isComposing" in script and "event.keyCode !== 229" in script
    assert not re.search(r"(?i)@import|url\(", style)
    with serving(idle(tmp_path), launch=False) as (_, page, _):
        for path, name, kind in (("/static/app.js", "app.js", "text/javascript; charset=utf-8"),
                                 ("/static/app.css", "app.css", "text/css; charset=utf-8")):
            reply = page.get(path, headers={"Cookie": None})
            assert reply.status == 200 and reply.header("Content-Type") == kind
            assert reply.body == (web.PAGE / name).read_bytes()


@pytest.mark.needs_data("tools/publish_raa/policy.py")
def test_the_page_holds_no_word_the_public_export_refuses():
    spec = importlib.util.spec_from_file_location("export_policy",
                                                  REPO / "tools" / "publish_raa" / "policy.py")
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    hits = []
    for name in ("index.html", "app.js", "app.css"):
        path = f"src/research_agent/live/page/{name}"  # as the export names it
        for number, line in enumerate(page_text(name).splitlines(), 1):
            for group, (pattern, paths) in policy.DISCLOSURE.items():
                if re.search(paths, path):
                    hits += [(path, number, group) for match in pattern.finditer(line)
                             if policy.allowed_reason(path, line, match.start(), match.end()) is None]
    assert hits == []

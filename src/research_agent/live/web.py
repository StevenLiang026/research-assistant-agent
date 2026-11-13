"""The chat page of scripts/ask.py --web (docs/m10-计划.md 3, 网页服务): a server on 127.0.0.1 for the page in
page/ (index.html, app.js, app.css), which shows the state kept here; the turns run on the main thread
(serve), the HTTP requests on the server's threads.

Access. Site binds a ThreadingHTTPServer to 127.0.0.1 alone, port 0 picking a free port; port 80 is refused
(ValueError) before anything is bound, since a browser leaves http's own port out of the Host header and the
Origin it sends, which the checks below compare whole with 127.0.0.1:<port>. On Windows it binds with
SO_EXCLUSIVEADDRUSE and without SO_REUSEADDR (HTTPServer sets that by default, and on Windows it lets a
socket bind a port another socket holds), so a port in use is an OSError, never a page that shares it. The
launch link, launch_url, carries a token (secrets.token_urlsafe(32)) that is printed to the console only and
holds for the whole run: GET /?t=<token> answers 303 to / with the cookie raa_<port> (another token; HttpOnly,
SameSite=Strict, Path=/), a wrong token 403. Then, for every request:
- the Host header is exactly 127.0.0.1:<port>, else 403 (a page fetched under another name, DNS rebinding
  among them, is never answered; "localhost" may resolve to ::1 and would keep a cookie of its own);
- GET / without the cookie is a 403 page that says to open the link the console printed; the static files
  need no cookie;
- /api/ needs the cookie (compared in constant time), and Sec-Fetch-Site, when sent, same-origin;
- a POST needs, besides, Origin exactly http://127.0.0.1:<port>, Content-Type application/json (415
  otherwise), a Content-Length (411 without one: no chunked body) of at most MAX_BODY bytes (413), and for
  /api/ask a JSON object whose "text" is a string, at most MAX_TEXT characters once its white space is
  collapsed (413), not empty (400), and one UTF-8 can encode (400: no lone surrogate, which a browser sends
  for half of an emoji cut off in a paste, and which no run file could hold).
A body of at most MAX_BODY bytes is read before any check, whatever is refused after: a connection closed
with data unread is reset on Windows, and the page would see a broken connection instead of the refusal; a
longer one is never read. Every response carries SECURITY_HEADERS (the policy lets the page load its own
script and style and connect to its own origin, and nothing else) and no Access-Control header; nothing is
logged, since a request line can hold the launch token. Requests are HTTP/1.0, one per connection; a
connection waits HANDLER_TIMEOUT seconds at most for its request. A request without an HTTP version
(HTTP/0.9), which no status line or header could answer, is closed unanswered.

Routes (other methods 405, other paths 404):
- GET /?t=TOKEN: the launch link (above). GET /: index.html. GET /static/app.js, /static/app.css: the
  page's script and style, from a fixed table (STATIC), never a path taken from the URL.
- GET /api/state: the snapshot {seq, system, status, exchanges}.
- GET /api/events?since=N: the event stream (Events and the stream, below).
- POST /api/ask {"text"}: 202 {exchange}; 400 {"error": "command"} for a text the console takes for a
  command (State), 400 {"error": "not_text"} for one UTF-8 cannot encode; 409 {"error": "busy"} while a
  turn is queued or runs, 409 {"error": "ending"} once the run is ending. POST /api/new: drops a
  clarification's pending request, 200 {status}; 409 while a turn runs. POST /api/quit: 202 {"ending":
  "now" | "after_turn"}.
- GET /api/notes?limit=N: {"notes"}, NoteStore.list newest first (limit 1 to NOTES_LIMIT_MAX, NOTES_LIMIT by
  default; 400 for any other), each with "url", its abstract page. GET /api/notes/<id>: {"note_id", "html":
  the note's file rendered by live.render}; the id must fully match NOTE_ID, as the URL has it (nothing is
  unquoted), before anything is looked up (400), and the resolved file must lie in the notes root; 404 for a
  note the library does not hold. 503 when the site has no notes library.

State (ChatState; the page only renders it). An exchange is one message and its turn: {"id", "request": the
text as typed, its white space collapsed to single spaces (training and evaluation only ever had one-line
user messages), "sent": the message the model gets (the pending request packed with the text after a
clarification: live.session.Conversation), "follow_up": its number (0 for a request of its own), "state":
queued | running | done | interrupted (the run ended before its turn: ChatState.end), "steps", "result"}.
A step is a step event's fields (live.session: Progress events) with "state" (running | done | error |
interrupted), "error" and "tools": [{"name", "state", "summary", "error", "ids", "seconds", "waited"}],
reduced from the events as they come (reduce_event). The result is {"status", "error",
"seconds", "today", "html": the answer or question rendered by live.render, "after": what the next message is
(Conversation.after: "answer", "restate" or None), "turn": its number in the run, "notes": [{"note_id",
"arxiv_id"}] written, "cited", "cited_unseen": the index line's}. The status is {"phase": idle | running |
ending | ended, "pending": {"exchange", "follow_ups"} while a clarification awaits its answer, else None}. One
turn at a time: a message is refused while another is queued or runs. A message is otherwise sent as
scripts/ask.py --interactive sends a line, but for what the console takes for one of its commands or for a
slip (console_word: a first word starting with "/", or a bare exit or quit in any letter case), which the
page refuses and never sends to the model, as the console never does: the page's buttons start a new
question and end the run. Every change publishes the changed exchange whole, or the status.

Events and the stream (Hub). Every published event gets the next sequence number and is kept, as JSON text
made under the hub's lock (_json_text: text UTF-8 can always encode, whatever the value holds), with the
WINDOW events before it. GET /api/events writes "retry: 3000", then the events after the page's last one
(Last-Event-ID, which a reconnecting EventSource sends, wins over ?since=), or, when those have left the
window or none is given, a "snapshot" event (the /api/state object); then every new event as it comes
("exchange", "status", "notes": {"written": [{"note_id", "arxiv_id"}]} after a turn that wrote notes), each
with "id: <seq>", and ": ping" every PING seconds when nothing happens. It ends with the "bye" event
({"reason": "ended" | "interrupted"}) when the run ends, or when a write fails (the page went away) or a
fault stops it: its answer has begun, so it ends there, never with a second answer inside it.

Serving. serve(site, turn, store, ...) starts the site, prints the launch link, opens it with browse when that
is given (a browser that cannot be opened changes nothing: the printed link is there), and runs the queued
messages one by one on the calling thread, the main thread in scripts/ask.py, waiting on the queue POLL
seconds at a time so a Ctrl+C reaches it: turn(sent, follow_up, events) runs the turn with the page's events
and returns it with its index line. A turn runs to its end whatever the page does (a tab closed, a reload);
quit ends the run at once when no turn runs, else after the turn. Site.close(reason) (idempotent) says bye
to every stream, gives the streams STREAM_GRACE seconds to send it, stops the server, closes its port, lets go
of every connection that has sent no request yet (a browser keeps one open ahead of its next request; its
thread waits for the request in steps of SERVE_POLL, since on Windows a thread that reads a connection cannot
be woken by shutting or closing it) and shuts the reading side of the others, waiting JOIN_GRACE seconds at
most for their threads; serve closes the site with "ended", or "interrupted" on a KeyboardInterrupt, which it
raises again. A response's body is written as the connection takes it, each send waiting SERVE_POLL seconds
at most (_put), for the same reason: a page that stopped reading, which a write would wait on and shutting
would not wake, holds its thread HANDLER_TIMEOUT seconds at most without taking a byte, and none past close,
while a reply the page goes on taking, the one to the quit that ends the run among them, is finished.
"""

from __future__ import annotations

import collections
import copy
import functools
import hmac
import json
import math
import queue
import re
import secrets
import select
import socket
import socketserver
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

from ..notes import ABS_URL, NoteStore
from .render import render_html
from .session import AWAITS_ANSWER, Conversation, Events, LiveTurn, notes_written

HOST = "127.0.0.1"
HTTP_PORT = 80  # http's own port, which a browser leaves out of the Host header and the Origin: refused
PAGE = Path(__file__).with_name("page")
INDEX = ("index.html", "text/html; charset=utf-8")
STATIC = {"/static/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/static/app.css": ("app.css", "text/css; charset=utf-8")}
COOKIE = "raa_"  # the cookie's name is this and the port: a page of another run on another port keeps its own
WINDOW = 500  # the events a hub keeps for a page that reconnects
PING = 15.0  # seconds between the comments that keep a quiet stream open
RETRY_MS = 3000  # how long an EventSource waits before it reconnects
POLL = 0.5  # seconds serve waits on its queue at a time, so a Ctrl+C reaches the main thread
SERVE_POLL = 0.2  # the server's own poll, which bounds how long stopping it takes
HANDLER_TIMEOUT = 10.0  # seconds a connection may take over its request, or take no byte of a write
STREAM_GRACE = 2.0  # seconds close() gives the streams to send bye before their connections are shut
JOIN_GRACE = 5.0  # seconds close() waits for the threads of the connections it shut
MAX_BODY = 64 * 1024
CHUNK = 64 * 1024  # the most a write hands the connection at once (_put)
MAX_TEXT = 4000
QUIT_WORDS = ("exit", "quit")  # scripts/ask.py's: typed bare, the console takes them for a slip
NOTES_LIMIT, NOTES_LIMIT_MAX = 50, 200
NOTE_ID = re.compile(r"\d{4}\.\d{4,5}-[1-9]\d*", re.ASCII)  # fullmatch: a note's id, which names its file
CSP = ("default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
SECURITY_HEADERS = {"Content-Security-Policy": CSP, "X-Frame-Options": "DENY",
                    "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store",
                    "Referrer-Policy": "same-origin"}
NO_PAGE = "请打开运行 ask.py --web 的控制台里打印的链接:页面要从那个链接进入。"
WRONG_HOST = "这个地址不提供页面。"

Out = Callable[[str], None]
TurnFn = Callable[[str, int, Events], tuple[LiveTurn, dict | None]]  # (sent, follow_up, events) -> turn, row


def _finite(value: object) -> object:
    """value with every float JSON cannot hold (NaN, an infinity) replaced by JSON's own spelling of it as a
    string ("NaN", "Infinity", "-Infinity"), through dicts, lists and tuples."""
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {_finite(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    return value


def _json_text(value: object) -> str:
    """value as compact JSON text that UTF-8 can encode, whatever value holds; never raises. json.loads, which
    reads a model's call arguments, takes NaN, the infinities and an escape of a lone surrogate, which then
    reach an event: a float JSON cannot hold is written as _finite spells it, and when the text holds a
    character UTF-8 cannot encode, every character outside ASCII is written as its escape (the page's
    JSON.parse reads them back). Only a value JSON cannot write at all (one that holds itself, or one nested
    deeper than the interpreter's recursion limit, which no event holds: live.session makes none it cannot
    copy) becomes an error object, so an exchange always reaches the page whole."""
    try:
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=str)
        except ValueError:  # a float JSON cannot hold
            value = _finite(value)
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False, default=str)
        try:
            text.encode("utf-8")
        except UnicodeEncodeError:  # a lone surrogate
            text = json.dumps(value, separators=(",", ":"), allow_nan=False, default=str)
        return text
    except Exception as exc:  # noqa: BLE001  deliberate: a publish never raises (module docstring)
        return json.dumps({"error": f"not sent: {type(exc).__name__}"}, separators=(",", ":"))


def system_info(record: dict, run: str | None) -> dict:
    """What the page's header shows of the system, from an assembly's record (scripts/ask.py
    Assembly.record), None for what the record does not hold."""
    info = record.get("info") or {}
    generation = info.get("generation") or {}
    return {"model": info.get("model"), "votes": generation.get("votes"), "vote_mode": info.get("vote_mode"),
            "reasoning_budget": generation.get("reasoning_budget"),
            "rules_version": (info.get("rules") or {}).get("version"),
            "build": (record.get("props") or {}).get("build_info"),
            "frozen_sha256": (record.get("frozen") or {}).get("sha256"),
            "server": (record.get("server") or {}).get("url"), "run": run}


# ---- events -----------------------------------------------------------------------------------------------


class Hub:
    """The published events (module docstring: Events and the stream): seq is the latest one's number,
    subscribers the streams open, reason the run's end once close() has said it."""

    def __init__(self, window: int = WINDOW) -> None:
        self._changed = threading.Condition()
        self._events: collections.deque[tuple[int, str, str]] = collections.deque(maxlen=window)
        self.seq = 0
        self.subscribers = 0
        self.reason: str | None = None
        self.bye: tuple[int, str, str] | None = None

    @property
    def closed(self) -> bool:
        return self.reason is not None

    def publish(self, kind: str, data: object) -> int:
        """data as the next event of kind, its JSON made under the lock; its number. No I/O, never raises."""
        with self._changed:
            self.seq += 1
            self._events.append((self.seq, kind, _json_text(data)))
            self._changed.notify_all()
            return self.seq

    def _after(self, seq: int) -> list[tuple[int, str, str]] | None:
        if not 0 <= seq <= self.seq:
            return None
        if seq == self.seq:
            return []
        first = self._events[0][0] if self._events else self.seq + 1
        if seq + 1 < first:
            return None
        return list(self._events)[seq + 1 - first:]

    def after(self, seq: int) -> list[tuple[int, str, str]] | None:
        """The events after seq, in order: [] when there is none yet, None when seq is not one the window
        can continue (fallen out of it, or never published)."""
        with self._changed:
            return self._after(seq)

    def wait(self, seq: int, timeout: float) -> list[tuple[int, str, str]] | None:
        """after(seq) once an event follows seq or the hub is closed, or timeout seconds have passed."""
        with self._changed:
            self._changed.wait_for(lambda: self.seq != seq or self.closed, timeout)
            return self._after(seq)

    def close(self, reason: str) -> None:
        """Publish bye with reason, once."""
        with self._changed:
            if self.closed:
                return
            seq = self.publish("bye", {"reason": reason})
            self.bye = self._events[-1] if self._events else (seq, "bye", _json_text({"reason": reason}))
            self.reason = reason

    @contextmanager
    def listening(self) -> Iterator[None]:
        """One stream open for as long as the block lasts."""
        with self._changed:
            self.subscribers += 1
        try:
            yield
        finally:
            with self._changed:
                self.subscribers -= 1
                self._changed.notify_all()

    def quiet(self, timeout: float) -> bool:
        """Whether every stream has ended, waiting timeout seconds at most."""
        with self._changed:
            return self._changed.wait_for(lambda: self.subscribers == 0, timeout)


# ---- the state --------------------------------------------------------------------------------------------


class Refused(Exception):
    """A message or a request the state refuses: status is the HTTP status, error the word the page gets."""

    status, error = 400, "refused"


class Invalid(Refused):
    status, error = 400, "empty"


class TooLong(Refused):
    status, error = 413, "too_long"


class NotText(Refused):
    status, error = 400, "not_text"


class Command(Refused):
    status, error = 400, "command"


class Busy(Refused):
    status, error = 409, "busy"


class Ending(Refused):
    status, error = 409, "ending"


def console_word(line: str) -> bool:
    """Whether the console (scripts/ask.py --interactive) takes line, a message with its white space
    collapsed, for one of its commands or for a slip, and so never sends it to the model as it stands: a first
    word that starts with "/" (/new, /quit and any other), or a bare "exit" or "quit" in any letter case."""
    words = line.split()
    return bool(words) and (words[0].startswith("/") or line.lower() in QUIT_WORDS)


def _step(step: object, attempt: object) -> dict:
    return {"step": step, "attempt": attempt, "state": "running", "seconds": None, "votes": None,
            "thinking": None, "reasoning": None, "rules": [], "calls": [], "answer_chars": None,
            "error": None, "tools": []}


def _tool(name: object) -> dict:
    return {"name": name, "state": "running", "summary": None, "error": None, "ids": [], "seconds": None,
            "waited": None}


def reduce_event(steps: list[dict], event: dict) -> bool:
    """event (live.session: Progress events) folded into an exchange's steps (module docstring: State);
    whether it changed them. A tool belongs to the latest step (its calls); a tool event with no step to go
    to, after the model's events stopped, and an event of another kind change nothing."""
    kind = event.get("kind")
    fields = {key: value for key, value in event.items() if key != "kind"}
    if kind == "step_start":
        steps.append(_step(event.get("step"), event.get("attempt")))
    elif kind in ("step", "step_error"):
        found = [step for step in steps if (step["step"], step["attempt"]) == (event.get("step"),
                                                                            event.get("attempt"))]
        if not found:
            steps.append(_step(event.get("step"), event.get("attempt")))
        step = found[-1] if found else steps[-1]
        step.update(fields, state="done" if kind == "step" else "error")
    elif kind in ("tool_start", "tool", "tool_error") and steps:
        tools = steps[-1]["tools"]
        running = [tool for tool in tools if tool["name"] == event.get("name") and tool["state"] == "running"]
        if kind == "tool_start" or not running:
            tools.append(_tool(event.get("name")))
        if kind != "tool_start":
            (running or tools)[-1].update(fields, state="done" if kind == "tool" else "error")
    else:
        return False
    return True


class ChatState:
    """What the page shows (module docstring: State), changed under one lock, every change published to hub.
    submit, new and quit are the HTTP threads'; begin, apply and finish the turn's (serve); queue carries
    the messages to run, (exchange id, sent, follow_up), and None when the run is to end now."""

    def __init__(self, hub: Hub, system: dict | None = None) -> None:
        self.hub = hub
        self.system = dict(system or {})
        self.conversation = Conversation()
        self.exchanges: list[dict] = []
        self.queue: queue.Queue[tuple[int, str, int] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._running: int | None = None  # the exchange queued or running
        self._pending: dict | None = None
        self._ending = self._ended = False

    @property
    def ending(self) -> bool:
        return self._ending

    def _status(self) -> dict:
        phase = ("ended" if self._ended else "ending" if self._ending else
                 "running" if self._running is not None else "idle")
        return {"phase": phase, "pending": None if self._pending is None else dict(self._pending)}

    def _publish(self, exchange: dict | None = None) -> None:
        if exchange is not None:
            self.hub.publish("exchange", exchange)
        else:
            self.hub.publish("status", self._status())

    def submit(self, text: object) -> dict:
        """A message of the page: queued as the next turn (module docstring: State); the new exchange. It is
        refused before anything is queued when it is empty, too long, holds what UTF-8 cannot encode (a lone
        surrogate, which no run file could hold) or is what the console takes for a command (console_word)."""
        line = " ".join(text.split()) if isinstance(text, str) else ""
        if not line:
            raise Invalid()
        if len(line) > MAX_TEXT:
            raise TooLong()
        try:
            line.encode("utf-8")
        except UnicodeEncodeError:
            raise NotText() from None
        if console_word(line):
            raise Command()
        with self._lock:
            if self._ending or self._ended:
                raise Ending()
            if self._running is not None:
                raise Busy()
            sent, follow_up = self.conversation.next(line)
            exchange = {"id": len(self.exchanges) + 1, "request": line, "sent": sent, "follow_up": follow_up,
                        "state": "queued", "steps": [], "result": None}
            self.exchanges.append(exchange)
            self._running, self._pending = exchange["id"], None
            self.queue.put((exchange["id"], sent, follow_up))
            self._publish(exchange)
            self._publish()
            return copy.deepcopy(exchange)

    def new(self) -> dict:
        """Drop the request a clarification left pending; the status. Busy while a turn is queued or runs."""
        with self._lock:
            if self._running is not None:
                raise Busy()
            self.conversation.drop()
            self._pending = None
            self._publish()
            return self._status()

    def quit(self) -> str:
        """End the run: "now" when no turn is queued or runs, else "after_turn"."""
        with self._lock:
            if not self._ending:
                self._ending = True
                if self._running is None:
                    self.queue.put(None)
                self._publish()
            return "now" if self._running is None else "after_turn"

    def begin(self, number: int) -> None:
        with self._lock:
            exchange = self.exchanges[number - 1]
            exchange["state"] = "running"
            self._publish(exchange)

    def apply(self, number: int, event: dict) -> None:
        """The events sink of exchange number's turn: quick, no I/O but the publish."""
        with self._lock:
            exchange = self.exchanges[number - 1]
            if reduce_event(exchange["steps"], event):
                self._publish(exchange)

    def finish(self, number: int, turn: LiveTurn, row: dict | None) -> None:
        """Exchange number's turn is over: its result (row: its index line), and what the next message is. The
        answer is rendered before the lock is taken, so a request of the page never waits on it."""
        transcript, row = turn.transcript, row or {}
        text = None if transcript is None else transcript.final_text
        html = render_html(text) if text else ""
        notes = [] if transcript is None else [{"note_id": note["note_id"], "arxiv_id": note["arxiv_id"]}
                                               for note in notes_written(transcript)]
        with self._lock:
            exchange = self.exchanges[number - 1]
            after = self.conversation.after(turn)
            exchange["result"] = {"status": turn.status, "error": turn.error, "seconds": turn.seconds,
                                  "today": turn.today, "html": html,
                                  "after": after, "turn": row.get("turn"), "notes": notes,
                                  "cited": list(row.get("cited") or []),
                                  "cited_unseen": list(row.get("cited_unseen") or [])}
            exchange["state"] = "done"
            self._running = None
            self._pending = ({"exchange": number, "follow_ups": self.conversation.follow_ups}
                             if after == AWAITS_ANSWER else None)
            self._publish(exchange)
            self._publish()
            if notes:
                self.hub.publish("notes", {"written": notes})

    def end(self) -> None:
        """The run has ended (Site.close). An exchange still queued or running then, its turn cut off by a
        Ctrl+C or a fault, becomes interrupted with its running steps and tools, and gets no result: the page
        never shows it going on."""
        with self._lock:
            if not self._ended:
                self._ended = True
                cut = None if self._running is None else self.exchanges[self._running - 1]
                if cut is not None and cut["state"] in ("queued", "running"):
                    cut["state"] = "interrupted"
                    for step in cut["steps"]:
                        for item in (step, *step["tools"]):
                            if item["state"] == "running":
                                item["state"] = "interrupted"
                    self._publish(cut)
                self._publish()

    def snapshot(self) -> dict:
        """{"seq", "system", "status", "exchanges"}: the state as of event seq."""
        with self._lock:
            return {"seq": self.hub.seq, "system": copy.deepcopy(self.system), "status": self._status(),
                    "exchanges": copy.deepcopy(self.exchanges)}


# ---- the server -------------------------------------------------------------------------------------------


class _Server(ThreadingHTTPServer):
    """ThreadingHTTPServer on 127.0.0.1 (module docstring: Access) that keeps its open connections, so close
    can shut them."""

    daemon_threads = True
    allow_reuse_address = sys.platform != "win32"

    def __init__(self, port: int, site: Site) -> None:
        self.site = site
        self.closing = False  # set by Site.close: a connection without its request is let go at once
        self.severed = False  # set by sever: a write its page has stopped taking gives up (_Handler._put)
        self._open: dict[socket.socket, threading.Thread] = {}
        self._open_lock = threading.Lock()
        super().__init__((HOST, port), _Handler)

    def server_bind(self) -> None:
        if sys.platform == "win32":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        socketserver.TCPServer.server_bind(self)  # HTTPServer's would look the address up in DNS
        self.server_name, self.server_port = HOST, self.server_address[1]

    def process_request(self, request: socket.socket, client_address: object) -> None:
        thread = threading.Thread(target=self.process_request_thread, args=(request, client_address),
                                  name=f"page-{self.server_port}-request", daemon=True)
        with self._open_lock:
            self._open[request] = thread
        thread.start()

    def shutdown_request(self, request: socket.socket) -> None:
        with self._open_lock:
            self._open.pop(request, None)
        super().shutdown_request(request)

    def handle_error(self, request: object, client_address: object) -> None:
        """Nothing printed: a connection that broke is the page's business (the handler answers its own
        faults with a 500)."""

    def sever(self, timeout: float) -> None:
        """Shut the reading side of every connection still open (a thread that reads one wakes, but on
        Windows, where it waits HANDLER_TIMEOUT seconds at most instead), then mark the site severed, so a
        write its page has stopped taking gives up within SERVE_POLL seconds (_Handler._put; on Windows,
        shutting a connection does not wake a thread that writes to it), and wait timeout seconds at most for
        their threads. The writing side is left to each thread: a reply in flight, the one to the quit that
        ended the run among them, still reaches its page."""
        with self._open_lock:
            open_now = list(self._open.items())
        for request, _ in open_now:
            try:
                request.shutdown(socket.SHUT_RD)
            except OSError:  # closed already
                pass
        self.severed = True
        for _, thread in open_now:
            thread.join(timeout)


class _Handler(BaseHTTPRequestHandler):
    """One request (module docstring: Access, Routes)."""

    server: _Server
    protocol_version = "HTTP/1.0"
    server_version = "raa-page"
    sys_version = ""
    timeout = HANDLER_TIMEOUT

    def log_message(self, format: str, *args: object) -> None:  # format: the base class's name
        """Nothing: a request line can hold the launch token."""

    def end_headers(self) -> None:
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        super().end_headers()

    def __getattr__(self, name: str) -> Callable[[], None]:
        if name.startswith("do_"):  # every method: GET and POST routed, any other answered 405
            return self._handle
        raise AttributeError(name)

    def handle(self) -> None:
        """The connection's request, once it arrives (_request_arrives); nothing when none does."""
        if self._request_arrives():
            super().handle()

    def parse_request(self) -> bool:
        """The base class's reading of the request line and headers, but a request without an HTTP version
        (HTTP/0.9: no status line and no header could answer it) is closed unanswered."""
        return super().parse_request() and self.request_version != "HTTP/0.9"

    def _request_arrives(self) -> bool:
        """Whether the connection has something to read within HANDLER_TIMEOUT seconds. It waits in steps of
        SERVE_POLL and stops at once when the site is closing: a browser opens a connection before it has a
        request for it, and on Windows a thread that reads such a connection cannot be woken by shutting or
        closing it (module docstring: Serving)."""
        deadline = time.monotonic() + self.timeout
        while True:
            left = deadline - time.monotonic()
            wait = 0.0 if self.server.closing or left <= 0 else min(left, SERVE_POLL)
            readable, _, _ = select.select([self.connection], [], [], wait)
            if readable:
                return True
            if not wait:
                return False

    # -- answers --

    def _put(self, data: bytes) -> None:
        """data to the connection as it takes it, at most CHUNK bytes at a time, each send waiting SERVE_POLL
        seconds at most (module docstring: Serving). A send that the connection takes nothing of within that
        time is tried again, until it has taken nothing for HANDLER_TIMEOUT seconds (TimeoutError), or the
        site has closed (ConnectionAbortedError, once sever has run): so a reply the page goes on taking is
        finished even while the site closes. The request has been read by then: the connection's own timeout
        matters to nothing else."""
        self.connection.settimeout(SERVE_POLL)
        done, since = 0, time.monotonic()
        while done < len(data):
            try:
                done += self.connection.send(data[done:done + CHUNK])
                since = time.monotonic()
            except TimeoutError:  # nothing taken in SERVE_POLL seconds
                if self.server.severed:
                    raise ConnectionAbortedError("the page's site has closed") from None
                if time.monotonic() - since > self.timeout:
                    raise

    def _send(self, status: int, body: bytes, content_type: str, headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        if self.command != "HEAD":
            self._put(body)

    def _json(self, status: int, value: object, headers: dict | None = None) -> None:
        self._send(status, _json_text(value).encode("utf-8"), "application/json; charset=utf-8", headers)

    def _error(self, status: int, error: str, headers: dict | None = None) -> None:
        self._json(status, {"error": error}, headers)

    def _page(self, status: int, message: str) -> None:
        body = ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>科研助理</title></head>'
                f"<body><p>{message}</p></body></html>")
        self._send(status, body.encode("utf-8"), INDEX[1])

    def _file(self, name: str, content_type: str) -> None:
        self._send(200, (PAGE / name).read_bytes(), content_type)

    # -- checks --

    def _one(self, name: str) -> str | None:
        """The header's value when it is sent once; None when it is not sent; "" when it is sent twice."""
        values = self.headers.get_all(name) or []
        return values[0] if len(values) == 1 else ("" if values else None)

    def _read_body(self) -> tuple[int | None, bytes]:
        """(None, the body) for a request whose Content-Length is at most MAX_BODY (an empty body for a
        request without one, but a POST), read before anything else: a connection closed with data unread is
        reset on Windows, and the page would see a broken connection instead of the refusal. (411, b"") for a
        POST without a length, a length that cannot be read or a chunked body, (413, b"") for a length over
        MAX_BODY, neither read."""
        length = self._one("Content-Length")
        if self.headers.get("Transfer-Encoding") is not None or length == "":
            return 411, b""
        if length is None:
            return (411 if self.command == "POST" else None), b""
        if not re.fullmatch(r"[0-9]+", length):
            return 411, b""
        if len(length.lstrip("0")) > len(str(MAX_BODY)) or int(length) > MAX_BODY:
            return 413, b""
        return None, self.rfile.read(int(length))

    def _has_cookie(self, site: Site) -> bool:
        token, found = site.cookie_token.encode("utf-8"), False
        for header in self.headers.get_all("Cookie") or []:
            for part in header.split(";"):
                name, equals, value = part.strip().partition("=")
                if equals and name == site.cookie_name:
                    found = hmac.compare_digest(value.encode("utf-8", "replace"), token) or found
        return found

    # -- routes --

    def _handle(self) -> None:
        site = self.server.site
        try:
            self._refused_body, self._body = self._read_body()
            if self.headers.get_all("Host") != [site.host]:
                return self._page(403, WRONG_HOST)
            path, _, query = self.path.partition("?")
            if path.startswith("/api/"):
                return self._api(site, path, query)
            if self.command != "GET":
                return self._error(405, "method", {"Allow": "GET"})
            if path == "/":
                return self._root(site, query)
            if path in STATIC:
                return self._file(*STATIC[path])
            return self._error(404, "not_found")
        except OSError:  # the connection went away, or timed out
            self.close_connection = True
        except Exception:  # noqa: BLE001  deliberate: a fault here gets a 500 and never its details
            self._error(500, "internal")

    def _root(self, site: Site, query: str) -> None:
        if query:  # the launch link
            tokens = parse_qs(query, keep_blank_values=True).get("t", [])
            if len(tokens) == 1 and hmac.compare_digest(tokens[0].encode("utf-8", "replace"),
                                                        site.launch_token.encode("utf-8")):
                cookie = f"{site.cookie_name}={site.cookie_token}; HttpOnly; SameSite=Strict; Path=/"
                return self._send(303, b"", "text/plain; charset=utf-8",
                                  {"Location": "/", "Set-Cookie": cookie})
            return self._page(403, NO_PAGE)
        if not self._has_cookie(site):
            return self._page(403, NO_PAGE)
        return self._file(*INDEX)

    def _api(self, site: Site, path: str, query: str) -> None:
        if not self._has_cookie(site):
            return self._error(403, "no_cookie")
        fetch_site = self._one("Sec-Fetch-Site")
        if fetch_site is not None and fetch_site != "same-origin":
            return self._error(403, "not_same_origin")
        posts = {"/api/ask": self._ask, "/api/new": self._new, "/api/quit": self._quit}
        gets = {"/api/state": self._state, "/api/events": self._events, "/api/notes": self._notes}
        known = path in posts or path in gets or path.startswith("/api/notes/")
        if not known:
            return self._error(404, "not_found")
        if self.command == "GET" and path not in posts:
            return gets[path](site, query) if path in gets else self._note(site, path[len("/api/notes/"):])
        if self.command != "POST" or path not in posts:
            return self._error(405, "method", {"Allow": "POST" if path in posts else "GET"})
        if self._one("Origin") != site.origin:
            return self._error(403, "not_same_origin")
        if self.headers.get_content_type() != "application/json":
            return self._error(415, "not_json")
        if self._refused_body is not None:
            return self._error(self._refused_body, "no_length" if self._refused_body == 411 else "too_large")
        return posts[path](site, self._body)

    def _ask(self, site: Site, body: bytes) -> None:
        try:
            data = json.loads(body)
        except (ValueError, RecursionError):
            return self._error(400, "not_json")
        text = data.get("text") if isinstance(data, dict) else None
        if not isinstance(text, str):
            return self._error(400, "not_text")
        try:
            exchange = site.state.submit(text)
        except Refused as refused:
            return self._error(refused.status, refused.error)
        return self._json(202, {"exchange": exchange})

    def _new(self, site: Site, body: bytes) -> None:
        try:
            status = site.state.new()
        except Refused as refused:
            return self._error(refused.status, refused.error)
        return self._json(200, {"status": status})

    def _quit(self, site: Site, body: bytes) -> None:
        return self._json(202, {"ending": site.state.quit()})

    def _state(self, site: Site, query: str) -> None:
        return self._json(200, site.state.snapshot())

    def _notes(self, site: Site, query: str) -> None:
        limits = parse_qs(query, keep_blank_values=True).get("limit", [str(NOTES_LIMIT)])
        if len(limits) != 1 or not re.fullmatch(r"[0-9]{1,3}", limits[0]) or not (
                1 <= int(limits[0]) <= NOTES_LIMIT_MAX):
            return self._error(400, "bad_limit")
        if site.store is None:
            return self._error(503, "no_notes_library")
        notes = site.store.list(limit=int(limits[0]))
        return self._json(200, {"notes": [{**note, "url": ABS_URL + note["arxiv_id"]} for note in notes]})

    def _note(self, site: Site, note_id: str) -> None:
        if not NOTE_ID.fullmatch(note_id):  # before anything is looked up: the id names the file
            return self._error(400, "bad_note_id")
        if site.store is None:
            return self._error(503, "no_notes_library")
        root = site.store.root.resolve()
        path = (root / f"{note_id}.md").resolve()
        if not path.is_relative_to(root):  # a link that leads out of the library
            return self._error(404, "no_such_note")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return self._error(404, "no_such_note")
        return self._json(200, {"note_id": note_id, "html": render_html(text)})

    # -- the stream --

    def _since(self, query: str) -> int | None:
        """The page's last event: Last-Event-ID, else ?since=; None when neither is a number."""
        header = self._one("Last-Event-ID")
        values = [header] if header is not None else parse_qs(query).get("since", [])
        if len(values) != 1 or not re.fullmatch(r"[0-9]{1,18}", values[0]):
            return None
        return int(values[0])

    def _write(self, text: str) -> None:
        self._put(text.encode("utf-8"))

    def _event(self, event: tuple[int, str, str]) -> None:
        seq, kind, data = event
        self._write(f"id: {seq}\nevent: {kind}\ndata: {data}\n\n")

    def _events(self, site: Site, query: str) -> None:
        last = self._since(query)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.end_headers()
        hub = site.hub
        with hub.listening():
            try:
                self._write(f"retry: {RETRY_MS}\n\n")
                batch = None if last is None else hub.after(last)
                while True:
                    if batch is None:  # a new page, or one that fell out of the window
                        snapshot = site.state.snapshot()
                        last = snapshot["seq"]
                        self._event((last, "snapshot", _json_text(snapshot)))
                    for event in batch or []:
                        self._event(event)
                        last = event[0]
                        if event[1] == "bye":
                            return
                    if batch == [] and hub.closed:  # the page has seen the end already, but for bye
                        return self._event(hub.bye)
                    batch = hub.wait(last, site.ping)
                    if batch == [] and not hub.closed:
                        self._write(": ping\n\n")
            except Exception:  # noqa: BLE001  deliberate: the page went away (OSError), or a fault
                self.close_connection = True  # the answer has begun: the stream ends, with no second answer


# ---- the site ---------------------------------------------------------------------------------------------


class Site:
    """The page's server on 127.0.0.1:port, bound at once (module docstring: Access); port 0 picks a free
    port, which port then holds, and HTTP_PORT is refused (ValueError) before anything is bound. store is the
    notes library the page shows (None: none)."""

    def __init__(self, port: int, *, store: NoteStore | None = None, window: int = WINDOW,
                 ping: float = PING) -> None:
        if port == HTTP_PORT:
            raise ValueError(f"port {HTTP_PORT} is http's own: a browser leaves it out of the Host header "
                             "and the Origin, which the page compares whole, so no request would pass")
        self.hub = Hub(window)
        self.state = ChatState(self.hub)
        self.store, self.ping = store, ping
        self.launch_token = secrets.token_urlsafe(32)
        self.cookie_token = secrets.token_urlsafe(32)
        self.server = _Server(port, self)
        self.port: int = self.server.server_address[1]
        self.host = f"{HOST}:{self.port}"
        self.origin = f"http://{self.host}"
        self.cookie_name = f"{COOKIE}{self.port}"
        self.launch_url = f"{self.origin}/?t={self.launch_token}"
        self._thread: threading.Thread | None = None
        self._closed = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Serve in a daemon thread of its own."""
        self._thread = threading.Thread(target=self.server.serve_forever,
                                        kwargs={"poll_interval": SERVE_POLL}, name=f"page-{self.port}",
                                        daemon=True)
        self._thread.start()

    def close(self, reason: str = "ended") -> None:
        """End the page (module docstring: Serving): bye with reason to every stream, the server stopped, its
        port closed and its connections shut. Only the first call does anything."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self.server.closing = True
            self.state.end()
            self.hub.close(reason)
            if self._thread is not None:
                self.hub.quiet(STREAM_GRACE)
        finally:
            if self._thread is not None:
                self.server.shutdown()
                self._thread.join()
            self.server.server_close()
            self.server.sever(JOIN_GRACE)


def serve(site: Site, turn: TurnFn, store: NoteStore | None, *, system: dict,
          browse: Callable[[str], object] | None, out: Out, poll: float = POLL) -> None:
    """Run the page's turns on this thread until the run ends (module docstring: Serving)."""
    site.store = store
    site.state.system = dict(system)
    state = site.state
    reason = "ended"
    try:
        site.start()
        out(f"page: {site.launch_url}")
        out("open it in a browser on this machine (the link holds the page's key); one turn runs at a time "
            "and cannot be cancelled; closing the page leaves the run going: end it with the page's 结束运行 "
            "or Ctrl+C")
        if browse is not None:
            try:
                opened = browse(site.launch_url)
            except Exception:  # noqa: BLE001  deliberate: the printed link suffices
                opened = False
            if opened is False:
                out("the browser did not open: open the link above")
        while True:
            try:
                item = state.queue.get(timeout=poll)
            except queue.Empty:
                continue
            if item is None:  # quit while no turn ran
                break
            number, sent, follow_up = item
            state.begin(number)
            result, row = turn(sent, follow_up, functools.partial(state.apply, number))
            state.finish(number, result, row)
            if state.ending:
                break
    except KeyboardInterrupt:
        reason = "interrupted"
        raise
    finally:
        site.close(reason)

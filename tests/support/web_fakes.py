"""Stand-ins for the tests of research_agent.live.web and scripts/ask.py --web: the turns of a page run on the
real run_live_turn over a scripted model, written by SessionFiles in a temporary folder (so the events and the
index rows are the ones a run makes), and a browser on the page over http.client, which reads no environment
setting and connects to 127.0.0.1 only.

ScriptedTurns is a live.web.TurnFn: each turn runs the message sent through run_live_turn over the model and
the tools given, with the page's events, and is written to the run's files; a gate, when given, holds every
turn until it is set, so a test can act while a turn runs. Page sends every request as the page does (the
exact Host, the cookie once launch() has it, Sec-Fetch-Site same-origin, and for a POST the Origin and a JSON
body), each header replaceable, and reads the page's event stream (Stream) event by event.

Run as a script, it serves the page over scripted turns, with no GPU and no network, for a check in a
browser:

    .venv/Scripts/python.exe -B tests/support/web_fakes.py --port 8766

prints the launch link (no browser is opened). The scripted model answers in order, a second or so per step:
a request that searches, reads a paper (its thinking closed by force), writes a note (after a rule changed the
call) and answers with arXiv ids; then a clarification, and the answer to it, whose vote no sample won and
whose reply cites an id no tool showed. After that every message gets one fixed reply. The notes library is a
temporary folder, gone when the script ends (the page's end button, or Ctrl+C).
"""

from __future__ import annotations

import argparse
import datetime
import http.client
import json
import sys
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from research_agent.agent.loop import Model
from research_agent.agent.types import AssistantTurn, ToolCall, ToolFn, ToolOutcome, parse_arguments
from research_agent.eval.systems import Built
from research_agent.live import web
from research_agent.live.session import Events, LiveTurn, SessionFiles, new_run_folder, run_live_turn
from research_agent.notes import NoteStore, note_tools
from research_agent.tools import schemas

TODAY = "2026-11-10"
HOST = "127.0.0.1"
TIMEOUT = 10.0  # seconds a request, or a stream's next line, may take before the test fails
A, B = "2611.00001", "2611.00002"


# ---- the turns --------------------------------------------------------------------------------------------


class ScriptedTurns:
    """A live.web.TurnFn over model (module docstring): sent keeps every (message, follow-up number) asked,
    started is set once a turn has begun, lines are the turns' progress lines (each also given to out, when
    there is one), files their run's files."""

    def __init__(self, model: Model, folder: Path, *, tools: dict[str, ToolFn] | None = None,
                 today: str = TODAY, gate: threading.Event | None = None,
                 waited: Callable[[], float] | None = None, out: Callable[[str], None] | None = None) -> None:
        self.model, self.tools, self.today, self.gate, self.waited = model, tools or {}, today, gate, waited
        self.files = SessionFiles(new_run_folder(folder))
        self.sent: list[tuple[str, int]] = []
        self.lines: list[str] = []
        self.out = out
        self.started = threading.Event()

    def _line(self, line: str) -> None:
        self.lines.append(line)
        if self.out is not None:
            self.out(line)

    def __call__(self, sent: str, follow_up: int, events: Events) -> tuple[LiveTurn, dict]:
        self.sent.append((sent, follow_up))
        self.started.set()
        if self.gate is not None and not self.gate.wait(TIMEOUT * 3):
            raise AssertionError("the gate of the scripted turn was never opened")
        result = run_live_turn(sent, built=Built(self.model, None, {}), tools=dict(self.tools),
                               today=self.today, out=self._line, waited=self.waited, events=events)
        return result, self.files.write_turn(result, follow_up=follow_up)


# ---- a browser on the page --------------------------------------------------------------------------------


@dataclass
class Reply:
    status: int
    headers: list[tuple[str, str]]
    body: bytes

    def header(self, name: str) -> str | None:
        """The first header named name (any case), None when there is none."""
        found = self.all(name)
        return found[0] if found else None

    def all(self, name: str) -> list[str]:
        return [value for key, value in self.headers if key.lower() == name.lower()]

    @property
    def text(self) -> str:
        return self.body.decode("utf-8")

    def json(self) -> object:
        return json.loads(self.body)


class Page:
    """A browser on the page of the site at port (module docstring): headers given replace the defaults of
    the same name (None leaves one out), extra adds more, the same name again included."""

    def __init__(self, port: int, *, timeout: float = TIMEOUT) -> None:
        self.port, self.timeout = port, timeout
        self.host = f"{HOST}:{port}"
        self.origin = f"http://{self.host}"
        self.cookie: str | None = None  # "raa_<port>=<value>", once launch() has it

    @classmethod
    def of(cls, url: str) -> Page:
        """A page for the site of url (its launch link)."""
        return cls(urllib.parse.urlsplit(url).port)

    def _headers(self, method: str, body: bytes | None, changes: dict | None) -> list[tuple[str, str]]:
        sent = {"Host": self.host, "Cookie": self.cookie, "Sec-Fetch-Site": "same-origin"}
        if method == "POST":
            sent.update({"Origin": self.origin, "Content-Type": "application/json"})
        if body is not None:
            sent["Content-Length"] = str(len(body))
        sent.update(changes or {})
        return [(name, value) for name, value in sent.items() if value is not None]

    def connect(self, method: str, path: str, body: bytes | None = None, *, headers: dict | None = None,
                extra: tuple[tuple[str, str], ...] = ()) -> tuple[http.client.HTTPConnection,
                                                                 http.client.HTTPResponse]:
        """A request sent and its response begun (its body not read): the connection is the caller's to
        close."""
        connection = http.client.HTTPConnection(HOST, self.port, timeout=self.timeout)
        try:
            connection.putrequest(method, path, skip_host=True, skip_accept_encoding=True)
            for name, value in [*self._headers(method, body, headers), *extra]:
                connection.putheader(name, value)
            connection.endheaders(body)
            return connection, connection.getresponse()
        except BaseException:
            connection.close()
            raise

    def request(self, method: str, path: str, body: bytes | None = None, *, headers: dict | None = None,
                extra: tuple[tuple[str, str], ...] = ()) -> Reply:
        connection, response = self.connect(method, path, body, headers=headers, extra=extra)
        try:
            return Reply(response.status, response.getheaders(), response.read())
        finally:
            response.close()
            connection.close()

    def get(self, path: str, **options) -> Reply:
        return self.request("GET", path, **options)

    def post(self, path: str, data: object = None, *, raw: bytes | None = None, **options) -> Reply:
        """data as JSON ({} by default), or the bytes raw."""
        body = raw if raw is not None else json.dumps({} if data is None else data).encode("utf-8")
        return self.request("POST", path, body, **options)

    def launch(self, url: str) -> Reply:
        """Open the launch link url as a browser does from the console (no cookie, Sec-Fetch-Site none), and
        keep the cookie it sets."""
        parts = urllib.parse.urlsplit(url)
        reply = self.get(f"{parts.path}?{parts.query}", headers={"Cookie": None, "Sec-Fetch-Site": "none"})
        if reply.status == 303:
            self.cookie = (reply.header("Set-Cookie") or "").split(";", 1)[0]
        return reply

    def ask(self, text: str) -> Reply:
        return self.post("/api/ask", {"text": text})

    def state(self) -> dict:
        reply = self.get("/api/state")
        assert reply.status == 200, reply
        return reply.json()

    def until(self, check: Callable[[dict], bool], timeout: float = TIMEOUT * 3) -> dict:
        """/api/state, asked again every 20 ms until check(state) holds; AssertionError after timeout."""
        deadline = time.monotonic() + timeout
        while True:
            state = self.state()
            if check(state):
                return state
            if time.monotonic() > deadline:
                raise AssertionError(f"the page never reached the state awaited: {state['status']}")
            time.sleep(0.02)

    def done(self, number: int) -> dict:
        """Exchange number once its turn is done (and the run is no longer running it)."""
        state = self.until(lambda state: len(state["exchanges"]) >= number
                           and state["exchanges"][number - 1]["state"] == "done"
                           and state["status"]["phase"] != "running")
        return state["exchanges"][number - 1]

    def stream(self, since: int | None = None, **options) -> Stream:
        path = "/api/events" if since is None else f"/api/events?since={since}"
        return Stream(*self.connect("GET", path, **options))


class Stream:
    """The page's event stream (GET /api/events), read event by event."""

    def __init__(self, connection: http.client.HTTPConnection, response: http.client.HTTPResponse) -> None:
        self.connection, self.response = connection, response
        self.status, self.headers = response.status, response.getheaders()

    def next(self) -> dict | None:
        """The next event, {"id", "event", "data": the data as JSON}, {"retry": milliseconds} or {"comment":
        its text}; None once the stream has ended."""
        fields: dict = {}
        while True:
            line = self.response.readline()
            if not line:
                return None
            line = line.decode("utf-8").rstrip("\r\n")
            if not line:
                if fields:
                    return fields
                continue
            if line.startswith(":"):
                return {"comment": line[1:].strip()}
            name, _, value = line.partition(":")
            value = value.removeprefix(" ")
            if name == "data":
                fields[name] = json.loads(value)
            else:
                fields[name] = int(value) if name in ("id", "retry") else value

    def until(self, check: Callable[[dict], bool], limit: int = 500) -> list[dict]:
        """The events read up to and including the first one check holds for; AssertionError when the stream
        ends first."""
        read = []
        for _ in range(limit):
            event = self.next()
            if event is None:
                raise AssertionError(f"the stream ended first: {read}")
            read.append(event)
            if check(event):
                return read
        raise AssertionError(f"no such event in {limit}")

    def rest(self) -> list[dict]:
        """Every event up to the end of the stream."""
        read = []
        while (event := self.next()) is not None:
            read.append(event)
        return read

    def close(self) -> None:
        self.response.close()
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# ---- the script: the page over scripted turns -------------------------------------------------------------

PAPERS = {
    A: {"arxiv_id": A, "title": "Adaptive Draft Length for Speculative Decoding", "published": "2026-11-02",
        "authors": ["A. Author", "B. Author"], "categories": ["cs.CL"],
        "abstract": "Speculative decoding drafts tokens with a small model and verifies them with the target "
                    "model. We let the draft length adapt to the acceptance rate of recent tokens. On three "
                    "benchmarks the adaptive length is 1.3 times faster than a fixed one. The gain shrinks "
                    "when the draft model is weak."},
    B: {"arxiv_id": B, "title": "Tree Verification at Scale", "published": "2026-11-03",
        "authors": ["C. Author"], "categories": ["cs.LG"],
        "abstract": "Tree-structured verification checks many drafted continuations at once. We serve it in "
                    "a large system and measure throughput. Batching the trees doubles the tokens per "
                    "second."},
}
HITS = [{"arxiv_id": paper["arxiv_id"], "title": paper["title"], "published": paper["published"],
         "snippet": paper["abstract"][:120]} for paper in PAPERS.values()]
SYSTEM = {"model": "scripted (tests/support/web_fakes.py)", "votes": 5, "vote_mode": "final",
          "reasoning_budget": 1280, "rules_version": "m7-wp3-1", "build": "none: no server",
          "frozen_sha256": "0" * 64, "server": None, "run": None}
STEP_SECONDS, TOOL_SECONDS, WAIT_SECONDS = 1.2, 0.6, 1.0
LONG_THOUGHT = ("先读摘要:它比较固定长度与自适应长度的草稿。摘要里说在三个基准上快 1.3 倍,"
                "但没有说草稿模型较弱时的情况,这一点要在笔记里标出来。") * 12
LAST_REPLY = "(演示脚本到此为止:之后的每条消息都只得到这一句。)"


def call(name: str, call_id: str, **args) -> ToolCall:
    """A tool call as a model emits it (support.fakes.call's, which the script cannot import)."""
    text = json.dumps(args, ensure_ascii=False)
    return ToolCall(id=call_id, name=name, arguments_json=text, arguments=parse_arguments(text))


def vote(won: int | None, representative: int | str, *, failed: int = 0, k: int = 5) -> dict:
    """A vote record as agent.vote writes one (only the fields the page reads)."""
    counts = {} if won is None else {"K": won, **({"L": k - won - failed} if k - won - failed else {})}
    record = {"k": k, "mode": "final", "chosen": None if won is None else "K", "counts": counts,
              "representative": representative}
    if failed:
        record["errors"] = [{"candidate": i, "error_type": "HarnessError", "message": "unreadable"}
                            for i in range(failed)]
    return record


def thought(text: str, tokens: int, *, forced: bool = False, **fields) -> dict:
    """An AssistantTurn's reasoning, thinking and vote fields."""
    return {"reasoning": text, "thinking": {"mode": "think", "reasoning_tokens": tokens, "forced": forced},
            **fields}


def demo_turns(today: str) -> list[AssistantTurn]:
    """The scripted model's turns (module docstring)."""
    week = (datetime.date.fromisoformat(today) - datetime.timedelta(days=7)).isoformat()
    unrepaired = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "pp_0", "type": "function",
         "function": {"name": "write_note", "arguments": json.dumps({"arxiv_id": "2611.1"})}}]}
    repaired = {**unrepaired, "tool_calls": [
        {"id": "pp_0", "type": "function",
         "function": {"name": "write_note", "arguments": json.dumps({"arxiv_id": A})}}]}

    def step(*calls, content: str = "", **fields) -> AssistantTurn:
        return AssistantTurn(content=content, tool_calls=list(calls), model="demo", **fields)

    search = call("search_arxiv", "c1", query="speculative decoding", date_from=week, sort_by="submittedDate")
    answer = (f"本周找到两篇:\n\n1. **{PAPERS[A]['title']}**({A}):草稿长度随接受率自适应,快 1.3 倍。\n"
              f"2. **{PAPERS[B]['title']}**(https://arxiv.org/abs/{B}):把验证树成批处理。\n\n"
              f"已为 {A} 写好笔记,编号 `{A}-1`。")
    question = call("ask_clarification", "q1", question="你想找哪个主题的论文?时间范围呢?",
                    missing=["topic", "date_range"])
    return [
        step(search, **thought("用户要本周关于 speculative decoding 的论文,还要给第一篇写笔记。"
                               "先按提交日期搜索。", 214, votes=vote(4, 1))),
        step(call("get_paper", "c2", arxiv_id=A),
             **thought(LONG_THOUGHT, 1280, forced=True, votes=vote(3, 0, failed=1))),
        step(call("write_note", "c3", arxiv_id=A),
             **thought("摘要读过了,按 scaffold 写笔记;编号要写全。", 96, votes=vote(5, 0),
                       rules=[{"rule": "id_repair", "before": unrepaired, "after": repaired}])),
        step(content=answer, **thought("两篇都在搜索结果里;笔记已写好,可以作答。", 58, votes=vote(4, 2))),
        step(question, **thought("请求没有说主题,也没有说时间范围,只能追问。", 41, votes=vote(5, 0))),
        step(call("search_arxiv", "c4", query="tree verification", sort_by="submittedDate"),
             **thought("补充了主题。没有样本给出可读的回答时,用贪心候选。", 133,
                       votes=vote(None, "greedy", failed=5))),
        step(content=f"找到 {B}:{PAPERS[B]['title']}。另可参考 2610.99999(它没有出现在本轮的工具输出里)。",
             **thought("只有一篇相关。", 22, votes=vote(5, 0))),
    ]


class SlowModel:
    """The demo's model: the scripted turns in order, each after a pause, then one fixed reply."""

    name = "demo"

    def __init__(self, turns: list[AssistantTurn], pause: float) -> None:
        self.turns, self.pause = list(turns), pause

    def next_turn(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        time.sleep(self.pause)
        if self.turns:
            return self.turns.pop(0)
        return AssistantTurn(content=LAST_REPLY, tool_calls=[], model="demo")


class DemoWorld:
    """The demo's tools: search and get_paper over PAPERS, each taking its time (search waits as for the rate
    limit, which slept counts), and the note tools over a real library."""

    def __init__(self, store: NoteStore, today: str) -> None:
        self.slept = 0.0
        self.tools = note_tools(store, PAPERS.get, today)
        self.tools.update(search_arxiv=self.search_arxiv, get_paper=self.get_paper)

    def search_arxiv(self, args: dict) -> ToolOutcome:
        time.sleep(WAIT_SECONDS + TOOL_SECONDS)
        self.slept += WAIT_SECONDS
        words = str(args.get("query", "")).lower().split()
        hits = [hit for hit in HITS if any(word in hit["title"].lower() for word in words)]
        return ToolOutcome(hits or HITS)

    def get_paper(self, args: dict) -> ToolOutcome:
        time.sleep(TOOL_SECONDS)
        paper = PAPERS.get(args.get("arxiv_id"))
        return ToolOutcome({}, schemas.NOT_FOUND) if paper is None else ToolOutcome(dict(paper))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the page over scripted turns, for a check in a "
                                                 "browser.")
    parser.add_argument("--port", type=int, default=8766,
                        help="the page's port (default 8766; 0: any free port)")
    args = parser.parse_args(argv)
    today = datetime.datetime.now(datetime.UTC).astimezone().date().isoformat()  # as scripts/ask.py's

    def out(line: str) -> None:
        print(line, flush=True)

    with tempfile.TemporaryDirectory(prefix="raa-page-") as folder:
        store = NoteStore(Path(folder) / "notes")
        world = DemoWorld(store, today)
        turns = ScriptedTurns(SlowModel(demo_turns(today), STEP_SECONDS), Path(folder) / "runs",
                              tools=world.tools, today=today, waited=lambda: world.slept, out=out)
        site = web.Site(args.port)
        try:
            web.serve(site, turns, store, system={**SYSTEM, "run": turns.files.folder.name}, browse=None,
                      out=out)
        except KeyboardInterrupt:
            print("interrupted", file=sys.stderr)
            return 130
        finally:
            site.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

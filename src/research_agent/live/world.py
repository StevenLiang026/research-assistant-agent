"""The live tools (docs/m9-计划.md WP1, 实时工具): the arXiv tools of tools.adapters over a live fetcher
with the owner's functions of research_agent.arxiv, and the note tools of research_agent.notes over a notes
library that lasts from one run to the next. Neither the adapters nor note_tools change; everything here
wraps them.

Rate limits across runs. LoggedLimits is the fetcher's HostLimits (net.ratelimit: 3 s between requests to
export.arxiv.org, 15 s to arxiv.org, the default for any other host), with the time of every host's latest
request kept in a state file (LAST_REQUESTS, under runs/live/), so the intervals hold from one run to the next
as well. The first request to a host in a run reads the state file; before every request the remainder of the
host's interval since its latest request, by the wall clock, is slept out (at most the interval and a second,
so a clock set back cannot stall a run), then the in-process limiter waits as it always has, and the request's
start goes back to the state file (written whole through a temporary name). A state file that cannot be read
makes the first request to every host wait its whole interval. One that could not be written after the latest
request (a scanner holding the file, say) is written once more when the run's world closes (LiveWorld.close,
LoggedLimits.settle); if that fails too, the close sleeps until every host's interval since its latest request
has passed, so a next run, which reads older times, cannot come too soon. Every request, retries and redirect
hops included, is a line of the run's request log (RequestLog: {"event": "request", "host", "t", "t_rel",
"slept"}), and LoggedFetcher adds one line per fetch ({"event": "fetch", "host", "url", "t", "t_end",
"status", "failed"}), so a check can show both the spacing and that the fetches never overlap. "t" is the
wall clock in seconds, "t_rel" the seconds since the log was opened.

Papers seen in a run. PaperMemo.parse is the parse_atom the adapters call: arxiv.parse_atom in its lenient
mode (an old-style id, an error entry or an empty entry is skipped, so an unknown id is NOT_FOUND rather than
UPSTREAM_ERROR) that keeps every paper it reads, whole, by its arxiv_id. note_tools looks papers up there
(PaperMemo.get), so write_note on a paper that a search, get_paper or the abstract fallback of
fetch_fulltext showed sends no request. live_tools wraps write_note for a paper seen only in a full text:
its metadata is looked up once first (PaperMemo.lookup, get_paper's request); a fetch that failed is
RATE_LIMITED for a 429 and UPSTREAM_ERROR otherwise, as the adapters map it, and a paper the API does not
know falls through to note_tools' NOT_FOUND.
"""

from __future__ import annotations

import copy
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import httpx

from .. import arxiv
from ..agent.types import ToolFn, ToolOutcome
from ..llm.retry import DELAYS
from ..net.fetch import Fetched, Fetcher, FetchFailed
from ..net.ratelimit import HostLimits, _host_key
from ..normalize import normalize_arxiv_id
from ..notes import NoteStore, note_tools
from ..tools import schemas
from ..tools.adapters import SupportsGet, _failure, arxiv_tools, paper_url, parsed_papers

LAST_REQUESTS = "last-requests.json"  # host -> wall time of its latest request, in the state folder
REQUEST_LOG = "requests.jsonl"  # in a run's folder
COOL_DOWN_SLACK = 1.0  # seconds a cool-down may exceed the interval by before it gives up (a clock set back)
WRITE_MODES = tuple(next(t for t in schemas.TOOLS if t["function"]["name"] == "write_note")
                    ["function"]["parameters"]["properties"]["mode"]["enum"])


def host_of(url: str) -> str:
    """The host a URL's rate limit is keyed by (net.ratelimit), read from the URL as httpx reads it."""
    return _host_key(httpx.URL(url).host)


class RequestLog:
    """A run's request log (module docstring): one JSON line per event, appended as it happens."""

    def __init__(self, path: str | Path, *, wall: Callable[[], float] = time.time) -> None:
        self.path, self.wall = Path(path), wall
        self.start = wall()

    def write(self, event: dict) -> None:
        t = event["t"] if "t" in event else self.wall()
        line = {**event, "t": t, "t_rel": round(t - self.start, 3)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")


def read_last_requests(path: Path) -> dict[str, float] | None:
    """The state file's host -> wall time; {} when there is none yet, None when it cannot be read."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not all(isinstance(t, int | float) and not isinstance(t, bool)
                                             for t in data.values()):
        return None
    return {str(host): float(t) for host, t in data.items()}


class LoggedLimits(HostLimits):
    """HostLimits whose intervals hold across runs and whose requests are logged (module docstring). clock is
    the in-process limiter's (monotonic), wall the clock of the state file and the log; slept is the seconds
    slept so far, the cool-downs included."""

    def __init__(self, state: str | Path, log: RequestLog | None = None, *,
                 intervals: dict[str, float] | None = None, default: float = 1.0,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        super().__init__(intervals, default, clock=clock, sleep=sleep)
        self.state, self.log, self.wall = Path(state), log, wall
        self.slept = 0.0
        self.state_error: str | None = None  # the latest failure to write the state file
        self._last: dict[str, float] | None = None  # host -> wall time of its latest request, any run
        self._unknown_since: float | None = None  # set when the state file could not be read

    def wait(self, url: str) -> float:
        """Sleep out the host's interval across runs and within this one, then record the request; returns
        the seconds slept."""
        host = host_of(url)
        interval = self.intervals.get(host, self.default)
        slept = self._cool_down(host, interval)
        slept += super().wait(url)
        now = self.wall()
        self._remember(host, now)
        self.slept += slept
        if self.log is not None:
            event = {"event": "request", "host": host, "t": now, "slept": round(slept, 3)}
            if self.state_error is not None:
                event["state_error"] = self.state_error
            self.log.write(event)
        return slept

    def _known(self) -> dict[str, float]:
        if self._last is None:
            read = read_last_requests(self.state)
            if read is None:
                self._unknown_since = self.wall()
            self._last = read or {}
        return self._last

    def _cool_down(self, host: str, interval: float) -> float:
        """Sleep until interval seconds have passed by the wall clock since the host's latest request (module
        docstring); the seconds slept."""
        last = self._known().get(host, self._unknown_since)
        if last is None:
            return 0.0
        deadline, limit, slept = last + interval, interval + COOL_DOWN_SLACK, 0.0
        while slept < limit and (remaining := deadline - self.wall()) > 0:
            pause = min(remaining, limit - slept)
            self.sleep(pause)
            slept += pause
        return slept

    def _remember(self, host: str, now: float) -> None:
        self._known()[host] = now
        self._save()

    def _save(self) -> None:
        """Write the state file whole (module docstring); state_error says why it could not be, None when it
        was."""
        staged = self.state.with_name(self.state.name + ".tmp")
        try:
            self.state.parent.mkdir(parents=True, exist_ok=True)
            staged.write_text(json.dumps(dict(sorted(self._known().items()))) + "\n", encoding="utf-8",
                              newline="\n")
            os.replace(staged, self.state)
            self.state_error = None
        except OSError as exc:  # the run goes on; the request log says the state was not kept
            self.state_error = f"{type(exc).__name__}: {exc}"

    def settle(self, notice: Callable[[str], None] | None = None) -> float:
        """When the state file could not be written after the latest request: write it once more, and when
        that fails too, tell notice why and sleep until every host's interval since its latest request has
        passed (module docstring); the seconds slept."""
        if self.state_error is None:
            return 0.0
        self._save()
        if self.state_error is None:
            return 0.0
        if notice is not None:
            notice(f"the rate-limit state could not be saved ({self.state_error}); waiting out arXiv's "
                   "intervals since the latest requests before the run ends")
        slept = sum(self._cool_down(host, self.intervals.get(host, self.default))
                    for host in list(self._known()))
        self.slept += slept
        return slept


class LoggedFetcher:
    """The fetcher the live tools use: every get() is a line of the request log (module docstring) and its
    answer is kept as last ({"url", "status", "text"}; text None when no usable answer came back), which the
    probe of scripts/ask.py reports. close() closes the wrapped fetcher."""

    def __init__(self, fetcher: SupportsGet, log: RequestLog | None = None, *,
                 wall: Callable[[], float] = time.time) -> None:
        self.fetcher, self.log, self.wall = fetcher, log, wall
        self.last: dict | None = None

    def get(self, url: str) -> Fetched:
        started = self.wall()
        try:
            page = self.fetcher.get(url)
        except FetchFailed as exc:
            self._note(url, started, exc.status, None, failed=True)
            raise
        except Exception:
            self._note(url, started, None, None, failed=True)
            raise
        self._note(url, started, page.status, page.text, failed=False)
        return page

    def _note(self, url: str, started: float, status: int | None, text: str | None, *, failed: bool) -> None:
        self.last = {"url": url, "status": status, "text": text}
        if self.log is not None:
            self.log.write({"event": "fetch", "host": host_of(url), "url": url, "t": started,
                            "t_end": self.wall(), "status": status, "failed": failed})

    def close(self) -> None:
        close = getattr(self.fetcher, "close", None)
        if close is not None:
            close()


class LookupFailed(Exception):
    """A paper's metadata could not be fetched; error is the code the adapters give such a fetch."""

    def __init__(self, arxiv_id: str, error: str) -> None:
        super().__init__(f"the metadata of {arxiv_id} could not be fetched: {error}")
        self.arxiv_id, self.error = arxiv_id, error


class PaperMemo:
    """The papers a run has read, whole, by arxiv_id (module docstring: Papers seen in a run)."""

    def __init__(self) -> None:
        self._papers: dict[str, dict] = {}

    def __contains__(self, arxiv_id: object) -> bool:
        return arxiv_id in self._papers

    def __len__(self) -> int:
        return len(self._papers)

    def parse(self, xml: str) -> list[dict] | None:
        """arxiv.parse_atom(xml, lenient=True), every paper it reads kept."""
        papers = arxiv.parse_atom(xml, lenient=True)
        for paper in papers or []:
            self._papers[paper["arxiv_id"]] = copy.deepcopy(paper)
        return papers

    def get(self, arxiv_id: str) -> dict | None:
        """A copy of the paper, None for one this run has not read: note_tools' papers."""
        paper = self._papers.get(arxiv_id)
        return None if paper is None else copy.deepcopy(paper)

    def lookup(self, fetcher: SupportsGet, arxiv_id: str) -> dict | None:
        """get_paper's request for arxiv_id, its feed read as parse reads it: the paper, None when the API
        does not know the id; LookupFailed with the adapters' code for a failed fetch, a status other than 200
        or a feed that is not one. A parse that breaks the adapters' contract raises their HarnessError."""
        try:
            page = fetcher.get(paper_url(arxiv_id))
        except FetchFailed as exc:
            raise LookupFailed(arxiv_id, _failure(exc.status)) from None
        if page.status != 200:
            raise LookupFailed(arxiv_id, _failure(page.status))
        if parsed_papers("write_note", self.parse, page.text) is None:
            raise LookupFailed(arxiv_id, schemas.UPSTREAM_ERROR)
        return self.get(arxiv_id)


def arxiv_live_tools(fetcher: SupportsGet, memo: PaperMemo, today: str) -> dict[str, ToolFn]:
    """search_arxiv, get_paper and fetch_fulltext over fetcher with the owner's functions, parse_atom being
    memo's (module docstring)."""
    return arxiv_tools(fetcher, build_search_url=arxiv.build_search_url, parse_atom=memo.parse,
                       extract_section=arxiv.extract_section, today=today)


def live_tools(fetcher: SupportsGet, memo: PaperMemo, store: NoteStore, today: str) -> dict[str, ToolFn]:
    """The six tools of a live turn on the day today: the arXiv tools (arxiv_live_tools) and note_tools over
    store with memo's papers, write_note looking a paper up first when the run has not read it (module
    docstring)."""
    tools = {**arxiv_live_tools(fetcher, memo, today), **note_tools(store, memo.get, today)}
    write = tools["write_note"]

    def write_note(args: dict) -> ToolOutcome:
        given = {**schemas.DEFAULTS["write_note"], **args}
        arxiv_id = given.get("arxiv_id")
        if (isinstance(arxiv_id, str) and normalize_arxiv_id(arxiv_id) == arxiv_id
                and given.get("mode") in WRITE_MODES and arxiv_id not in memo):
            try:
                memo.lookup(fetcher, arxiv_id)
            except LookupFailed as exc:
                return ToolOutcome({}, exc.error)
        return write(args)

    tools["write_note"] = write_note
    return tools


@dataclass
class LiveWorld:
    """What every turn of a run shares: the logged fetcher, its limits, the papers read and the request
    log."""

    fetcher: LoggedFetcher
    limits: LoggedLimits
    memo: PaperMemo
    log: RequestLog

    def tools(self, store: NoteStore, today: str) -> dict[str, ToolFn]:
        return live_tools(self.fetcher, self.memo, store, today)

    def close(self, notice: Callable[[str], None] | None = None) -> float:
        """Close the fetcher once the limits have settled (LoggedLimits.settle, which tells notice when it has
        to wait); the seconds they slept."""
        try:
            return self.limits.settle(notice)
        finally:
            self.fetcher.close()


def live_world(folder: str | Path, state: str | Path, *, transport: object | None = None,
               clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
               sleep: Callable[[float], None] = time.sleep,
               retry_delays: Sequence[float] = DELAYS) -> LiveWorld:
    """The live world of a run whose files go to folder (its request log), the rate-limit state kept in the
    folder state (LAST_REQUESTS): a Fetcher in live mode (the tests' transport given, the environment's
    connection settings otherwise, as the fetcher has them) with LoggedLimits."""
    folder = Path(folder)
    log = RequestLog(folder / REQUEST_LOG, wall=wall)
    limits = LoggedLimits(Path(state) / LAST_REQUESTS, log, clock=clock, wall=wall, sleep=sleep)
    fetcher = Fetcher("live", folder, limits=limits, transport=transport, sleep=sleep,
                      retry_delays=retry_delays)
    return LiveWorld(LoggedFetcher(fetcher, log, wall=wall), limits, PaperMemo(), log)

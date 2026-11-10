"""Web fetching for the tools in three modes: live, record and replay.

live sends every request and writes nothing. replay answers from cassettes alone and sends nothing; a missing
cassette raises ReplayMiss, a harness failure that is never shown to the model. record answers from the
cassette when the URL has one, exactly as replay would, without a request and without a rate-limit wait; only
a URL without a cassette is fetched, and its response becomes the cassette. So a recording session shows the
model exactly what its replay will. A cassette is one JSON file per canonical URL, named by the SHA-256 of
that URL and never overwritten.

Only 200, 404 and 410 are recorded. In record mode every other outcome raises RecordFailed, a harness failure
too: a transient failure that outlasted the retries, a redirect loop, an undecodable body, any other status,
or an API request redirected out of the API. The model must not see a result that replay could never show it,
and the URL stays without a cassette for a later session to record. In live mode the same failures raise
FetchFailed, which the adapters turn into error codes, and every other status is an ordinary result.

Only arXiv API responses (export.arxiv.org/api/..., CC0 metadata) go to <cassette_dir>/api/, the folder that
may be committed; every other URL, above all the HTML full texts that are for personal use only, goes to
<cassette_dir>/fulltext/, which is never committed. Each folder keeps a manifest.json (key -> url, status,
bytes, fetched_at) that diffs cleanly. Cassettes are the source of truth; the manifest is an index for review.

Every request first waits for its own host's rate limit, retries and redirect hops included. 429, 5xx,
timeouts and transport errors are retried after 3 s, 6 s and 12 s (longer when Retry-After asks for it);
redirect loops and undecodable bodies are not, since the same request would get the same answer. Mistakes in
the request itself (an unsupported scheme, an invalid URL, a malformed request) propagate unwrapped.
retry_delays=() sends every request once: a probe of a service that may be failing must not multiply its
requests by four (docs/m2-录制计划.md, the probe of 2026-10-09).
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from ..agent.types import HarnessError
from ..llm.retry import DELAYS, call_with_retries
from .ratelimit import HostLimits

USER_AGENT = "research-assistant-agent/0.1 (personal research use)"
MODES = ["live", "record", "replay"]
API_DIR, FULLTEXT_DIR = "api", "fulltext"
API_HOST, API_PATH = "export.arxiv.org", "/api/"  # the only responses that go to API_DIR
MANIFEST = "manifest.json"
RECORDABLE_STATUSES = [200, 400, 404, 410]  # 400: arXiv's stable answer to a malformed query
# Every redirect hop waits for its host's limit, so a loop on arxiv.org costs 15 s per hop: give up after 5.
MAX_REDIRECTS = 5
# The server answered, but nothing usable came of it; the same request would get the same answer, so no retry.
UNUSABLE_ANSWERS = (httpx.TooManyRedirects, httpx.DecodingError)


@dataclass(frozen=True)
class Fetched:
    url: str  # as requested
    final_url: str  # after redirects
    status: int
    body: bytes
    content_type: str

    @property
    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")


class FetchFailed(Exception):
    """Live mode: no usable answer for url, which the adapters report to the model as an error code.

    Raised for a transient failure that outlasted every retry, a redirect loop and an undecodable body.
    status is the last 429 or 5xx, or None when no usable response came back.
    """

    def __init__(self, url: str, status: int | None = None, reason: str = "") -> None:
        reason = reason or _status_reason(status)
        super().__init__(f"GET {url} failed: {reason}")
        self.url, self.status, self.reason = url, status, reason


class RecordFailed(HarnessError):
    """Record mode: the outcome for url is not recorded, so the model must not see it either.

    Raised for every failure that is FetchFailed in live mode, for statuses other than 200, 404 and 410, and
    for an API request redirected out of the API. It is no FetchFailed, so the adapters let it through and
    the turn ends as a harness failure. status is the response's status, or None when no usable response
    came back.
    """

    def __init__(self, url: str, status: int | None = None, reason: str = "") -> None:
        reason = reason or _status_reason(status)
        super().__init__(f"GET {url} not recorded: {reason}")
        self.url, self.status, self.reason = url, status, reason


class ReplayMiss(HarnessError):
    """Replay found no cassette for a URL: the recording is incomplete."""


def utc_now() -> str:
    """Current UTC time in ISO 8601 to the second, e.g. 2026-10-04T12:00:00+00:00."""
    return datetime.now(UTC).isoformat(timespec="seconds")


def canonical_url(url: str) -> str:
    """Lower-case scheme and host, no fragment, query parameters sorted by name: equal URLs give equal keys.

    The query is decoded and encoded again, so "a+b" and "a%20b" agree; repeated parameters keep their order.
    """
    parts = urlsplit(url)
    userinfo, at, hostport = parts.netloc.rpartition("@")
    pairs = sorted(parse_qsl(parts.query, keep_blank_values=True), key=lambda pair: pair[0])
    return urlunsplit((parts.scheme.lower(), userinfo + at + hostport.lower(), parts.path,
                       urlencode(pairs, safe=":/,"), ""))


def cassette_folder(url: str) -> str:
    """api for the arXiv API (export.arxiv.org/api/...), whose CC0 metadata may be committed; fulltext for
    every other URL, so nothing unexpected can land in the committable folder."""
    parts = urlsplit(url)
    is_api = parts.hostname == API_HOST and parts.path.startswith(API_PATH)  # hostname is lower-cased
    return API_DIR if is_api else FULLTEXT_DIR


def cassette_key(url: str) -> str:
    """SHA-256 (hex) of the canonical URL; the cassette's file name without .json."""
    return _sha256(canonical_url(url))


def transient_status(status: int) -> bool:
    """429 Too Many Requests and 5xx server errors are worth another attempt."""
    return status == 429 or 500 <= status <= 599


def is_transient(exc: BaseException) -> bool:
    """429 or 5xx, a timeout or a transport error; not our own mistakes (unsupported scheme, bad request)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return transient_status(exc.response.status_code)
    ours = isinstance(exc, (httpx.UnsupportedProtocol, httpx.LocalProtocolError))
    return isinstance(exc, httpx.TransportError) and not ours


class Fetcher:
    """GET with per-host rate limits and retries in live, record or replay mode; see the module docstring."""

    def __init__(self, mode: str, cassette_dir: str | Path, *, limits: HostLimits | None = None,
                 transport: httpx.BaseTransport | None = None, timeout: float = 60.0,
                 retry_delays: Sequence[float] = DELAYS, sleep: Callable[[float], None] = time.sleep,
                 now: Callable[[], str] = utc_now) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        self.mode, self.cassette_dir, self.sleep, self.now = mode, Path(cassette_dir), sleep, now
        self.retry_delays = tuple(retry_delays)
        self.limits = HostLimits(sleep=sleep) if limits is None else limits
        self.http = None if mode == "replay" else httpx.Client(
            transport=transport, timeout=timeout, follow_redirects=True, max_redirects=MAX_REDIRECTS,
            headers={"User-Agent": USER_AGENT},
            event_hooks={"request": [self._wait_for_host]})

    def get(self, url: str) -> Fetched:
        """The response for url; raises FetchFailed (live), RecordFailed (record) or ReplayMiss (replay)."""
        if self.mode != "live":
            recorded = self.recorded(url)
            if recorded is not None:  # in record mode too: a recorded URL is never fetched again
                return recorded
            if self.mode == "replay":
                canonical = canonical_url(url)
                raise ReplayMiss(f"no cassette for {canonical} ({cassette_folder(canonical)}/"
                                 f"{_sha256(canonical)}.json)")
        fetched = self._fetch(url)
        if self.mode == "live":
            return fetched
        self._record(fetched)
        return self.recorded(url)  # read back, so a recording session returns exactly what replay will

    def cassette_path(self, url: str) -> Path:
        """Where the cassette for this URL lives, whether or not it exists."""
        return self._path(canonical_url(url))

    def recorded(self, url: str) -> Fetched | None:
        """The recorded response for this URL, or None when there is no cassette. Sends nothing."""
        canonical = canonical_url(url)
        path = self._path(canonical)
        if not path.exists():
            return None
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["url"] != canonical:  # only a renamed or hand-edited file can get here
            raise HarnessError(f"cassette {path.name} holds {record['url']}, not {canonical}")
        body = base64.b64decode(record["body"]) if record["body_base64"] else record["body"].encode("utf-8")
        return Fetched(url=url, final_url=record["final_url"], status=record["status"], body=body,
                       content_type=record["content_type"])

    def close(self) -> None:
        if self.http is not None:
            self.http.close()

    def _wait_for_host(self, request: httpx.Request) -> None:
        """httpx request hook: every request waits for its own host's limit before it goes out, so retries and
        redirect hops (which may lead to another host) are spaced as well."""
        self.limits.wait(str(request.url))

    def _fetch(self, url: str) -> Fetched:
        """GET url, retrying transient failures; a fetch without a usable answer raises _failed()."""
        try:
            response = call_with_retries(lambda: self._attempt(url), retryable=is_transient,
                                         delays=self.retry_delays, sleep=self.sleep)
        except UNUSABLE_ANSWERS as exc:
            raise self._failed(url, None, f"{type(exc).__name__}: {exc}") from exc
        except Exception as exc:
            if not is_transient(exc):
                raise
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                raise self._failed(url, status, f"status {status} after retries") from exc
            raise self._failed(url, None, f"{type(exc).__name__} after retries") from exc
        return Fetched(url=url, final_url=str(response.url), status=response.status_code,
                       body=response.content, content_type=response.headers.get("content-type", ""))

    def _attempt(self, url: str) -> httpx.Response:
        response = self.http.get(url)  # the request hook waits for the rate limit
        if transient_status(response.status_code):  # raised so that call_with_retries retries it
            raise httpx.HTTPStatusError(f"status {response.status_code}", request=response.request,
                                        response=response)
        return response

    def _failed(self, url: str, status: int | None, reason: str) -> FetchFailed | RecordFailed:
        """What a fetch without a usable answer raises: FetchFailed when live, RecordFailed when recording."""
        return (RecordFailed if self.mode == "record" else FetchFailed)(url, status, reason)

    def _path(self, canonical: str) -> Path:
        return self.cassette_dir / cassette_folder(canonical) / f"{_sha256(canonical)}.json"

    def _record(self, fetched: Fetched) -> None:
        """Keep fetched as the cassette of its URL; RecordFailed for an outcome that is not recorded."""
        if fetched.status not in RECORDABLE_STATUSES:
            raise RecordFailed(fetched.url, fetched.status,
                               f"status {fetched.status}; only {RECORDABLE_STATUSES} are recorded")
        canonical = canonical_url(fetched.url)
        if cassette_folder(canonical) == API_DIR and cassette_folder(fetched.final_url) != API_DIR:
            raise RecordFailed(fetched.url, fetched.status,
                               f"redirected to {fetched.final_url}, outside the arXiv API")
        path = self._path(canonical)
        if path.exists():
            return  # never overwritten; get() checked just before, so only a concurrent recorder gets here
        fetched_at = self.now()
        try:
            body, is_base64 = fetched.body.decode("utf-8"), False
        except UnicodeDecodeError:
            body, is_base64 = base64.b64encode(fetched.body).decode("ascii"), True
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(path, {"url": canonical, "final_url": fetched.final_url, "status": fetched.status,
                           "content_type": fetched.content_type, "fetched_at": fetched_at,
                           "body_base64": is_base64, "body": body})
        manifest_path = path.parent / MANIFEST
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        manifest[path.stem] = {"url": canonical, "status": fetched.status, "bytes": len(fetched.body),
                               "fetched_at": fetched_at}
        _write_json(manifest_path, manifest)


def _status_reason(status: int | None) -> str:
    return f"status {status}" if status is not None else "no response"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    """Sorted keys, indent 2, LF line ends; via a temporary file, so a crash never leaves half a file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                   newline="\n")
    os.replace(tmp, path)

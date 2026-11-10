import base64
import dataclasses
import hashlib
import json
from datetime import datetime, timedelta

import httpx
import pytest

from research_agent.agent.types import HarnessError
from research_agent.net.fetch import (
    MAX_REDIRECTS,
    USER_AGENT,
    Fetched,
    Fetcher,
    FetchFailed,
    RecordFailed,
    ReplayMiss,
    canonical_url,
    cassette_folder,
    cassette_key,
    is_transient,
    utc_now,
)
from research_agent.net.ratelimit import HostLimits

API_URL = "https://export.arxiv.org/api/query?search_query=all:speculative&start=0&max_results=5"
API_URL_REORDERED = "https://Export.arXiv.org/api/query?max_results=5&search_query=all%3Aspeculative&start=0#top"
HTML_URL = "https://arxiv.org/html/2609.01234"
ATOM = ('<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">'
        "<title>推测解码</title></feed>").encode()
ATOM_TYPE = "application/atom+xml; charset=UTF-8"
HTML_TYPE = "text/html; charset=utf-8"
NOW = "2026-10-04T12:00:00+00:00"
FAILED = {"live": FetchFailed, "record": RecordFailed}  # raised per mode when no usable answer came back


class FakeClock:
    """A clock that moves only when something sleeps on it; every sleep is recorded."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now, self.sleeps = start, []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class Server:
    """Stands in for arXiv: scripted responses or httpx exception classes, in order; requests are logged."""

    def __init__(self, *script, clock: FakeClock | None = None) -> None:
        self.script, self.clock = list(script), clock
        self.requests: list[httpx.Request] = []
        self.started: list[float] = []  # fake-clock time of each request

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.clock is not None:
            self.started.append(self.clock.now)
        if not self.script:
            raise AssertionError(f"unexpected request {request.url}")
        item = self.script.pop(0)
        if isinstance(item, type):
            raise item("scripted failure", request=request)
        return item

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)


def reply(status: int = 200, body: bytes = ATOM, content_type: str = ATOM_TYPE,
          headers: dict | None = None) -> httpx.Response:
    return httpx.Response(status, content=body, headers={"content-type": content_type, **(headers or {})})


def redirect(location: str, status: int = 302) -> httpx.Response:
    return httpx.Response(status, headers={"location": location})


def persistent(kind: str) -> list:
    """Four attempts (the first and three retries) failing the same transient way: a status or an error."""
    if kind.isdigit():
        return [reply(int(kind), b"try again later", "text/plain") for _ in range(4)]
    return [getattr(httpx, kind)] * 4


def undecodable() -> httpx.Response:
    """A 200 whose body claims gzip but is not: httpx raises DecodingError while reading it."""
    return httpx.Response(200, stream=httpx.ByteStream(b"not gzip at all"),
                          headers={"content-type": ATOM_TYPE, "content-encoding": "gzip"})


def make(mode: str, tmp_path, *script, now: str = NOW) -> tuple[Fetcher, Server, FakeClock]:
    clock = FakeClock()
    server = Server(*script, clock=clock)
    fetcher = Fetcher(mode, tmp_path / "cassettes", limits=HostLimits(clock=clock, sleep=clock.sleep),
                      transport=server.transport, sleep=clock.sleep, now=lambda: now)
    return fetcher, server, clock


def no_requests(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"replay sent a request to {request.url}")


def replayer(tmp_path, clock: FakeClock | None = None) -> Fetcher:
    clock = clock or FakeClock()
    return Fetcher("replay", tmp_path / "cassettes", limits=HostLimits(clock=clock, sleep=clock.sleep),
                   transport=httpx.MockTransport(no_requests), sleep=clock.sleep)


def files(tmp_path) -> list[str]:
    """Every file under the cassette folder as a relative POSIX path, sorted."""
    root = tmp_path / "cassettes"
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


# live mode and the request itself


def test_live_returns_the_response_with_a_polite_user_agent_and_writes_nothing(tmp_path):
    fetcher, server, _ = make("live", tmp_path, reply(), reply(404, b"no HTML", HTML_TYPE))
    got = fetcher.get(API_URL)
    assert got == Fetched(url=API_URL, final_url=API_URL, status=200, body=ATOM, content_type=ATOM_TYPE)
    assert got.text.endswith("<title>推测解码</title></feed>")
    assert server.requests[0].method == "GET" and server.requests[0].headers["user-agent"] == USER_AGENT
    assert fetcher.get(HTML_URL).status == 404
    assert not (tmp_path / "cassettes").exists()


def test_redirects_are_followed_and_the_final_url_is_recorded_and_replayed(tmp_path):
    final = HTML_URL + "v2"
    fetcher, server, _ = make("record", tmp_path, redirect(final, 301),
                              reply(body=b"<html>full text</html>", content_type=HTML_TYPE))
    got = fetcher.get(HTML_URL)
    assert got.status == 200 and got.final_url == final and got.url == HTML_URL
    assert [str(r.url) for r in server.requests] == [HTML_URL, final]
    assert all(r.headers["user-agent"] == USER_AGENT for r in server.requests)
    assert replayer(tmp_path).get(HTML_URL) == got


def test_other_statuses_are_results_returned_at_once(tmp_path):
    fetcher, server, clock = make("live", tmp_path, reply(403, b"forbidden", "text/plain"),
                                  reply(404, b"no HTML", HTML_TYPE))
    assert fetcher.get(API_URL).status == 403
    assert fetcher.get(HTML_URL).status == 404
    assert clock.sleeps == [] and len(server.requests) == 2


def test_text_decodes_utf8_and_replaces_bad_bytes():
    body = "推测".encode() + b"\xff"
    assert Fetched(url="u", final_url="u", status=200, body=body, content_type="text/plain").text == "推测�"


def test_unknown_mode_is_refused(tmp_path):
    with pytest.raises(ValueError, match="playback"):
        Fetcher("playback", tmp_path)


def test_utc_now_is_iso_to_the_second():
    stamp = utc_now()
    parsed = datetime.fromisoformat(stamp)
    assert stamp.endswith("+00:00") and parsed.utcoffset() == timedelta(0) and parsed.microsecond == 0


# rate limits and retries


def test_requests_are_spaced_per_host(tmp_path):
    fetcher, _, clock = make("live", tmp_path, *[reply() for _ in range(5)])
    for url in (API_URL, HTML_URL, API_URL, "https://example.org/a", "https://example.org/b"):
        fetcher.get(url)
    assert clock.sleeps == [3.0, 1.0]  # API again after 3 s; a host without its own limit after 1 s


def test_default_limits_are_the_arxiv_limits_and_use_the_injected_sleep(tmp_path):
    sleeps = []
    server = Server(*[reply(body=b"<html/>", content_type=HTML_TYPE) for _ in range(2)])
    fetcher = Fetcher("live", tmp_path, transport=server.transport, sleep=sleeps.append)
    fetcher.get(HTML_URL)
    fetcher.get(HTML_URL)
    assert len(sleeps) == 1 and 14.0 < sleeps[0] <= 15.0  # the real clock moved a little between the calls


def test_waits_before_every_attempt_including_retries(tmp_path):
    fetcher, server, clock = make("live", tmp_path, reply(503), reply(503),
                                  reply(body=b"<html/>", content_type=HTML_TYPE))
    assert fetcher.get(HTML_URL).status == 200
    # 3 s backoff, then 12 s more to complete the 15 s crawl delay; 6 s backoff, then 9 s more
    assert clock.sleeps == [3.0, 12.0, 6.0, 9.0]
    assert server.started == [1000.0, 1015.0, 1030.0]


def test_a_redirect_hop_waits_for_the_crawl_delay_like_any_request(tmp_path):
    final = HTML_URL + "v2"
    fetcher, server, clock = make("live", tmp_path, redirect(final, 301),
                                  reply(body=b"<html/>", content_type=HTML_TYPE))
    assert fetcher.get(HTML_URL).final_url == final
    assert server.started == [1000.0, 1015.0] and clock.sleeps == [15.0]


def test_a_redirect_hop_waits_for_its_own_host(tmp_path):
    fetcher, server, clock = make("live", tmp_path, reply(body=b"<html/>", content_type=HTML_TYPE),
                                  redirect(HTML_URL + "v2"), reply(body=b"<html/>", content_type=HTML_TYPE))
    fetcher.get(HTML_URL)
    fetcher.get("https://example.org/moved")  # the first request to example.org; it redirects to arxiv.org
    assert server.started == [1000.0, 1000.0, 1015.0] and clock.sleeps == [15.0]


def test_retry_after_is_honoured(tmp_path):
    slow_down = reply(429, b"slow down", "text/plain", {"Retry-After": "20"})
    fetcher, server, clock = make("live", tmp_path, slow_down, reply())
    assert fetcher.get(API_URL).status == 200
    assert clock.sleeps == [20.0] and len(server.requests) == 2  # 20 s also covers the 3 s API interval


def test_crawl_delay_holds_through_every_retry(tmp_path):
    fetcher, server, clock = make("live", tmp_path, *persistent("503"))
    with pytest.raises(FetchFailed):
        fetcher.get(HTML_URL)
    assert clock.sleeps == [3.0, 12.0, 6.0, 9.0, 12.0, 3.0]
    assert server.started == [1000.0, 1015.0, 1030.0, 1045.0]


def test_timeouts_and_transport_errors_are_retried(tmp_path):
    fetcher, server, clock = make("live", tmp_path, httpx.ReadTimeout, httpx.ConnectError,
                                  httpx.RemoteProtocolError, reply())
    assert fetcher.get(API_URL).status == 200
    assert clock.sleeps == [3.0, 6.0, 12.0] and len(server.requests) == 4


# failures: live mode reports them to the adapters, record mode ends the turn


def test_live_failures_are_for_the_adapters_and_record_failures_end_the_turn():
    assert not issubclass(FetchFailed, HarnessError)  # the adapters turn it into an error code
    assert issubclass(RecordFailed, HarnessError)  # the loop ends the turn as harness_error ...
    assert not issubclass(RecordFailed, FetchFailed)  # ... because the adapters let it through


@pytest.mark.parametrize("mode", ["live", "record"])
@pytest.mark.parametrize("kind, status", [("503", 503), ("429", 429), ("ReadTimeout", None),
                                          ("ConnectError", None)])
def test_a_persistent_transient_failure_is_retried_raised_and_never_recorded(tmp_path, mode, kind, status):
    fetcher, server, clock = make(mode, tmp_path, *persistent(kind))
    with pytest.raises(FAILED[mode], match=kind) as failed:
        fetcher.get(API_URL)
    assert (failed.value.url, failed.value.status) == (API_URL, status)
    assert len(server.requests) == 4 and clock.sleeps == [3.0, 6.0, 12.0]
    assert not (tmp_path / "cassettes").exists()


@pytest.mark.parametrize("mode", ["live", "record"])
def test_without_retry_delays_a_transient_failure_is_sent_once(tmp_path, mode):
    clock = FakeClock()
    server = Server(*persistent("503"), clock=clock)
    fetcher = Fetcher(mode, tmp_path / "cassettes", limits=HostLimits(clock=clock, sleep=clock.sleep),
                      transport=server.transport, retry_delays=(), sleep=clock.sleep, now=lambda: NOW)
    with pytest.raises(FAILED[mode], match="503") as failed:
        fetcher.get(API_URL)
    assert failed.value.status == 503 and len(server.requests) == 1 and clock.sleeps == []
    assert not (tmp_path / "cassettes").exists()


@pytest.mark.parametrize("mode", ["live", "record"])
def test_a_redirect_loop_fails_at_once_without_retries(tmp_path, mode):
    url = "https://example.org/moved"
    fetcher, server, clock = make(mode, tmp_path, *[redirect(url) for _ in range(MAX_REDIRECTS + 1)])
    with pytest.raises(FAILED[mode], match="TooManyRedirects") as failed:
        fetcher.get(url)
    assert (failed.value.url, failed.value.status) == (url, None)
    # httpx gives up after MAX_REDIRECTS hops; every hop waited 1 s for example.org, and nothing was retried
    assert len(server.requests) == MAX_REDIRECTS + 1 and clock.sleeps == [1.0] * MAX_REDIRECTS
    assert not (tmp_path / "cassettes").exists()


@pytest.mark.parametrize("mode", ["live", "record"])
def test_an_undecodable_body_fails_at_once_without_retries(tmp_path, mode):
    fetcher, server, clock = make(mode, tmp_path, undecodable())
    with pytest.raises(FAILED[mode], match="DecodingError") as failed:
        fetcher.get(API_URL)
    assert (failed.value.url, failed.value.status) == (API_URL, None)
    assert len(server.requests) == 1 and clock.sleeps == []
    assert not (tmp_path / "cassettes").exists()


@pytest.mark.parametrize("status", [401, 403, 451, 300])
def test_record_mode_refuses_any_other_status_and_records_nothing(tmp_path, status):
    fetcher, server, clock = make("record", tmp_path, reply(status, b"not this", "text/plain"))
    with pytest.raises(RecordFailed, match=f"status {status}") as failed:
        fetcher.get(HTML_URL)
    assert (failed.value.url, failed.value.status) == (HTML_URL, status)
    assert len(server.requests) == 1 and clock.sleeps == []  # not transient: no retry
    assert not (tmp_path / "cassettes").exists()


@pytest.mark.parametrize("mode", ["live", "record"])
@pytest.mark.parametrize("error", [httpx.UnsupportedProtocol, httpx.LocalProtocolError])
def test_errors_that_retrying_cannot_fix_are_raised_at_once(tmp_path, mode, error):
    fetcher, server, clock = make(mode, tmp_path, error)
    with pytest.raises(error):
        fetcher.get("ftp://example.org/file")
    assert len(server.requests) == 1 and clock.sleeps == []
    assert not (tmp_path / "cassettes").exists()


@pytest.mark.parametrize("mode", ["live", "record"])
def test_an_invalid_url_is_raised_as_it_is(tmp_path, mode):
    fetcher, server, _ = make(mode, tmp_path)
    with pytest.raises(httpx.InvalidURL):
        fetcher.get("https://export.arxiv.org:abc/api/query")
    assert server.requests == []


def test_transient_classification():
    request = httpx.Request("GET", API_URL)

    def status(code: int) -> httpx.HTTPStatusError:
        return httpx.HTTPStatusError("x", request=request, response=httpx.Response(code, request=request))

    assert all(is_transient(status(code)) for code in (429, 500, 502, 503, 504))
    assert not any(is_transient(status(code)) for code in (400, 403, 404, 410))
    assert is_transient(httpx.ReadTimeout("x", request=request))
    assert is_transient(httpx.ConnectError("x", request=request))
    assert not is_transient(httpx.UnsupportedProtocol("x", request=request))
    assert not is_transient(httpx.LocalProtocolError("x", request=request))
    assert not is_transient(httpx.TooManyRedirects("x", request=request))  # these fail at once instead
    assert not is_transient(httpx.DecodingError("x", request=request))
    assert not is_transient(ValueError("x"))


# record and replay


def test_record_then_replay_round_trip(tmp_path):
    fetcher, _, _ = make("record", tmp_path, reply())
    recorded = fetcher.get(API_URL)
    clock = FakeClock()
    replay = replayer(tmp_path, clock)
    assert replay.get(API_URL) == recorded
    assert replay.get(API_URL_REORDERED) == dataclasses.replace(recorded, url=API_URL_REORDERED)
    assert clock.sleeps == []  # replay neither sends nor waits


@pytest.mark.parametrize("status", [400, 404, 410])  # 400: arXiv's answer to a malformed query is stable
def test_stable_answers_other_than_200_are_returned_recorded_and_replayed(tmp_path, status):
    fetcher, _, _ = make("record", tmp_path, reply(status, b"No HTML for this paper", HTML_TYPE))
    got = fetcher.get(HTML_URL)
    assert got.status == status and fetcher.cassette_path(HTML_URL).exists()
    assert replayer(tmp_path).get(HTML_URL) == got


def test_cassette_holds_the_response_and_is_named_by_the_url_hash(tmp_path):
    fetcher, _, _ = make("record", tmp_path, reply())
    fetcher.get(API_URL)
    path = fetcher.cassette_path(API_URL)
    assert path.name == hashlib.sha256(canonical_url(API_URL).encode()).hexdigest() + ".json"
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "url": canonical_url(API_URL), "final_url": API_URL, "status": 200, "content_type": ATOM_TYPE,
        "fetched_at": NOW, "body_base64": False, "body": ATOM.decode()}


def test_bodies_that_are_not_utf8_are_stored_as_base64(tmp_path):
    url, raw = "https://example.org/figure.png", b"\x89PNG\r\n\x1a\n\x00\xff"
    fetcher, _, _ = make("record", tmp_path, reply(body=raw, content_type="image/png"))
    assert fetcher.get(url).body == raw
    record = json.loads(fetcher.cassette_path(url).read_text(encoding="utf-8"))
    assert record["body_base64"] is True and base64.b64decode(record["body"]) == raw
    assert replayer(tmp_path).get(url).body == raw


def test_replay_miss_is_a_harness_error(tmp_path):
    with pytest.raises(ReplayMiss) as missed:
        replayer(tmp_path).get(HTML_URL)
    assert isinstance(missed.value, HarnessError)
    assert f"fulltext/{cassette_key(HTML_URL)}.json" in str(missed.value)


def test_an_existing_cassette_is_never_overwritten(tmp_path):
    other = "https://export.arxiv.org/api/query?id_list=2609.01234"
    first, _, _ = make("record", tmp_path, reply(body=b"first"))
    recorded = first.get(API_URL)
    path = first.cassette_path(API_URL)
    manifest = path.parent / "manifest.json"
    before = (path.read_bytes(), json.loads(manifest.read_text(encoding="utf-8")))
    # A later session fetches only the URL without a cassette. The recorded one is answered from its cassette,
    # exactly as replay will answer it: no request, and no wait although the API was just asked.
    second, server, clock = make("record", tmp_path, reply(body=b"new"), now="2026-10-05T00:00:00+00:00")
    assert second.get(other).body == b"new"
    got = second.get(API_URL_REORDERED)
    assert got == replayer(tmp_path).get(API_URL_REORDERED) == dataclasses.replace(recorded,
                                                                                    url=API_URL_REORDERED)
    assert got.body == b"first" and second.get(other).body == b"new"
    assert [str(r.url) for r in server.requests] == [other] and clock.sleeps == []
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    key = cassette_key(API_URL)
    assert path.read_bytes() == before[0] and entries[key] == before[1][key]  # the old entry stays as it was
    assert sorted(entries) == sorted([key, cassette_key(other)])


@pytest.mark.parametrize("url, folder", [
    (API_URL, "api"),
    (API_URL_REORDERED, "api"),  # the host is case-insensitive
    ("http://export.arxiv.org/api/query?id_list=2609.01234", "api"),
    ("https://export.arxiv.org:443/api/query?id_list=2609.01234", "api"),
    (HTML_URL, "fulltext"),
    ("https://arxiv.org/abs/2609.01234", "fulltext"),
    ("https://arxiv.org/pdf/2609.01234", "fulltext"),
    ("https://arxiv.org/api/query?id_list=2609.01234", "fulltext"),  # the path, but not the API host
    ("https://export.arxiv.org/abs/2609.01234", "fulltext"),  # the API host, but not the API
    ("https://export.arxiv.org/html/2609.01234", "fulltext"),
    ("https://export.arxiv.org/api", "fulltext"),
    ("https://export.arxiv.org/API/query", "fulltext"),  # paths are case-sensitive
    ("https://mirror.export.arxiv.org/api/query", "fulltext"),
    ("https://export.arxiv.org.example.org/api/query", "fulltext"),
    ("https://example.org/figure.png", "fulltext"),
])
def test_only_the_arxiv_api_goes_to_the_committable_folder(url, folder):
    assert cassette_folder(url) == folder


def test_full_texts_and_everything_but_the_api_go_to_the_private_folder(tmp_path):
    abs_url = "https://arxiv.org/abs/2609.01234"
    fetcher, _, _ = make("record", tmp_path, reply(), reply(body=b"<html/>", content_type=HTML_TYPE),
                         reply(body=b"<html>abstract page</html>", content_type=HTML_TYPE))
    for url in (API_URL, HTML_URL, abs_url):
        fetcher.get(url)
    assert files(tmp_path) == sorted([
        f"api/{cassette_key(API_URL)}.json", "api/manifest.json", f"fulltext/{cassette_key(HTML_URL)}.json",
        f"fulltext/{cassette_key(abs_url)}.json", "fulltext/manifest.json"])


def test_an_api_request_redirected_out_of_the_api_is_not_recorded(tmp_path):
    fetcher, server, _ = make("record", tmp_path, redirect(HTML_URL, 301),
                              reply(body=b"<html>full text</html>", content_type=HTML_TYPE))
    with pytest.raises(RecordFailed, match="arxiv.org/html") as failed:
        fetcher.get(API_URL)
    assert (failed.value.url, failed.value.status) == (API_URL, 200) and len(server.requests) == 2
    assert not (tmp_path / "cassettes").exists()  # a full text never lands in the committable folder
    plain = API_URL.replace("https://", "http://")  # a redirect that stays in the API is fine
    fetcher, _, _ = make("record", tmp_path, redirect(API_URL, 301), reply())
    assert fetcher.get(plain).final_url == API_URL
    assert files(tmp_path) == [f"api/{cassette_key(plain)}.json", "api/manifest.json"]


def test_manifest_lists_each_cassette_with_sorted_keys_and_lf_line_ends(tmp_path):
    other = "https://export.arxiv.org/api/query?id_list=2609.01234"
    fetcher, _, _ = make("record", tmp_path, reply(), reply(404, b"missing", "text/plain"))
    fetcher.get(API_URL)
    fetcher.get(other)
    expected = {
        cassette_key(API_URL): {"url": canonical_url(API_URL), "status": 200, "bytes": len(ATOM),
                                "fetched_at": NOW},
        cassette_key(other): {"url": canonical_url(other), "status": 404, "bytes": 7, "fetched_at": NOW}}
    path = tmp_path / "cassettes" / "api" / "manifest.json"
    assert path.read_bytes() == (json.dumps(expected, indent=2, sort_keys=True) + "\n").encode()


def test_a_renamed_cassette_is_refused(tmp_path):
    other = "https://export.arxiv.org/api/query?id_list=2609.05678"
    fetcher, server, _ = make("record", tmp_path, reply())
    fetcher.get(API_URL)
    fetcher.cassette_path(API_URL).rename(fetcher.cassette_path(other))
    with pytest.raises(HarnessError, match="holds"):
        replayer(tmp_path).get(other)
    with pytest.raises(HarnessError, match="holds"):  # record mode reads a cassette as replay does
        fetcher.get(other)
    assert len(server.requests) == 1


# canonical URLs


def test_canonical_url_ignores_parameter_order_case_fragment_and_encoding():
    a = canonical_url("https://export.arxiv.org/api/query?search_query=all:speculative+decoding&start=0&max_results=5")
    b = canonical_url("HTTPS://Export.ArXiv.org/api/query?max_results=5&start=0"
                      "&search_query=all%3Aspeculative%20decoding#section-2")
    assert a == b == "https://export.arxiv.org/api/query?max_results=5&search_query=all:speculative+decoding&start=0"
    assert canonical_url(a) == a
    assert cassette_key(a) == cassette_key(b)


def test_canonical_url_keeps_what_can_change_the_answer():
    # paths are case-sensitive, and repeated parameters keep their order
    assert canonical_url("https://arxiv.org/HTML/2609.01234") != canonical_url(HTML_URL)
    assert canonical_url("https://x.org/q?a=1&a=2") != canonical_url("https://x.org/q?a=2&a=1")
    assert canonical_url("https://x.org/q?a=1") != canonical_url("https://x.org/q?a=2")

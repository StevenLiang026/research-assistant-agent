"""The M2 corpus recording: research_agent.data.corpus and scripts/record_corpus.py (docs/m2-录制计划.md)."""

import importlib.util
import itertools
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from research_agent.data import corpus
from research_agent.net.fetch import Fetched, Fetcher, RecordFailed
from research_agent.net.ratelimit import HostLimits
from research_agent.tools import schemas
from research_agent.tools.adapters import arxiv_tools, fulltext_url, paper_url

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("record_corpus", REPO / "scripts" / "record_corpus.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

DATE_TO = "2026-10-10"
SPEC = {"window_from": "2026-07-01", "per_topic": 2, "topics": [
    {"id": "alpha", "name": "甲", "phrases": ["alpha"]},
    {"id": "beta", "name": "乙", "phrases": ["beta", "beta prime"]},
    {"id": "gamma", "name": "丙", "phrases": ["gamma"]},
    {"id": "delta", "name": "丁", "phrases": ["delta"]},
]}
LISTED = {"alpha": ["2609.00001", "2609.00002", "2609.00003"],
          "beta": ["2609.00002", "2609.00004", "2609.00005"],  # 2609.00002 belongs to alpha already
          "gamma": ["2609.00006", "2609.00007"],
          "delta": ["2609.00008", "2609.00009"]}
CHOSEN = {"alpha": ["2609.00001", "2609.00002"], "beta": ["2609.00004", "2609.00005"],
          "gamma": ["2609.00006", "2609.00007"], "delta": ["2609.00008", "2609.00009"]}
NO_HTML = {"2609.00005"}


def feed(ids: list[str]) -> str:
    """An Atom feed as arXiv writes one: a feed-level <id> (the query's URL), then one <id> per entry."""
    entries = "".join(f"<entry>\n<id>http://arxiv.org/abs/{i}v1</id>\n<title>Paper {i}</title>\n</entry>"
                      for i in ids)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<feed xmlns="http://www.w3.org/2005/Atom">\n'
            "<id>https://arxiv.org/api/0123456789abcdef</id>\n"
            f"<opensearch:totalResults>{len(ids)}</opensearch:totalResults>\n{entries}\n</feed>\n")


def listing(topic: str) -> str:
    phrases = next(t["phrases"] for t in SPEC["topics"] if t["id"] == topic)
    return corpus.listing_url(phrases, SPEC["window_from"], DATE_TO)


HEALTH = corpus.health_url("alpha")  # the first phrase of the first topic


def world(listed: dict = LISTED, no_html: set = NO_HTML) -> dict:
    """URL -> (status, text) for every request a full recording of SPEC makes."""
    pages = {HEALTH: (200, feed(listed["alpha"][:1]))}
    pages |= {listing(topic): (200, feed(ids)) for topic, ids in listed.items()}
    for arxiv_id in {i for ids in listed.values() for i in ids}:
        pages[paper_url(arxiv_id)] = (200, feed([arxiv_id]))
        html = (404, "") if arxiv_id in no_html else (200, f"<html>{arxiv_id}</html>")
        pages[fulltext_url(arxiv_id)] = html
    return pages


class Fake:
    """A recording fetcher stand-in: answers from pages, raises RecordFailed for the URLs in fail."""

    def __init__(self, pages: dict, fail: set | None = None) -> None:
        self.pages, self.fail, self.requests = pages, fail or set(), []

    def get(self, url: str) -> Fetched:
        self.requests.append(url)
        if url in self.fail:
            raise RecordFailed(url, 500, "status 500 after retries")
        status, text = self.pages[url]
        return Fetched(url=url, final_url=url, status=status, body=text.encode("utf-8"),
                       content_type="text/xml")


def run(pages: dict | None = None, *, fail: set | None = None, probe_fail: set | None = None, **kwargs):
    pages = world() if pages is None else pages
    fetcher, probe, lines = Fake(pages, fail), Fake(pages, probe_fail), []
    manifest = corpus.record_corpus(fetcher, probe, SPEC, DATE_TO, log=lines.append, **kwargs)
    return manifest, fetcher, probe, lines


def test_the_listing_query_is_written_as_arxivs_manual_writes_its_examples():
    url = corpus.listing_url(["speculative decoding", "draft model"], "2026-07-01", DATE_TO)
    assert url == ("https://export.arxiv.org/api/query?search_query=%28all:%22speculative+decoding%22+OR+"
                   "all:%22draft+model%22%29+AND+%28cat:cs.CL+OR+cat:cs.LG+OR+cat:cs.AI+OR+cat:cs.IR+OR+"
                   "cat:cs.CV+OR+cat:stat.ML%29+AND+submittedDate:[202607010000+TO+202610102359]&start=0"
                   "&max_results=50&sortBy=relevance&sortOrder=descending")
    query = parse_qs(urlsplit(url).query)
    categories = " OR ".join(f"cat:{c}" for c in schemas.CATEGORIES)
    expected = (f'(all:"speculative decoding" OR all:"draft model") AND ({categories}) '
                f"AND submittedDate:[202607010000 TO 202610102359]")
    assert query["search_query"] == [expected]


def test_listed_ids_reads_only_entry_ids_in_order_without_versions_or_repeats():
    text = feed(["2609.01234", "2608.00001"]).replace("2609.01234v1", "2609.01234v3") + feed(["2609.01234"])
    assert corpus.listed_ids(text) == ["2609.01234", "2608.00001"]
    assert corpus.listed_ids("<id>\n  http://arxiv.org/abs/2607.12345v2 \n</id>") == ["2607.12345"]
    assert corpus.listed_ids(feed([])) == []  # the feed's own <id> is the query, not a paper


def test_choose_takes_per_topic_ids_in_order_and_skips_those_of_earlier_topics():
    chosen = corpus.choose({"a": ["1", "2", "3"], "b": ["2", "4"], "c": ["1"]}, 2)
    assert chosen == {"a": ["1", "2"], "b": ["4"], "c": []}


def test_a_full_run_probes_lists_and_records_metadata_then_full_text_per_paper():
    manifest, fetcher, probe, lines = run()
    assert probe.requests == [HEALTH, listing("alpha"), listing("beta"), listing("gamma")]  # once each
    listings = [listing(t["id"]) for t in SPEC["topics"]]
    papers = [url for ids in CHOSEN.values() for i in ids for url in (paper_url(i), fulltext_url(i))]
    assert fetcher.requests == listings + papers
    assert [t["papers"] for t in manifest["topics"]] == list(CHOSEN.values())
    assert [(t["listed"], t["shortfall"]) for t in manifest["topics"]] == [(3, 0), (3, 0), (2, 0), (2, 0)]
    beta = manifest["topics"][1]
    assert beta["phrases"] == ["beta", "beta prime"] and beta["name"] == "乙"
    assert manifest["topics"][0]["listing"].startswith("https://export.arxiv.org/api/query?")
    assert (manifest["papers"], manifest["metadata"], manifest["fulltext"]) == (8, {"200": 8},
                                                                                {"200": 7, "404": 1})
    assert manifest["complete"] and manifest["stopped"] is None and manifest["failed"] == []
    assert (manifest["window"], manifest["per_topic"], manifest["listing_size"]) == (["2026-07-01", DATE_TO],
                                                                                     2, corpus.LISTING_SIZE)
    assert lines[:2] == ["probe: health query has 1 papers", "probe: alpha listing has 3 papers"]
    assert lines[-1] == "papers: 8/8"


def test_a_failed_probe_stops_the_run_before_anything_else_is_sent():
    manifest, fetcher, probe, _ = run(probe_fail={listing("alpha")})
    assert probe.requests == [HEALTH, listing("alpha")] and fetcher.requests == []
    assert manifest["stopped"].startswith("probe: the alpha listing failed") and not manifest["complete"]
    assert manifest["topics"] == [] and manifest["papers"] == 0


def test_a_failed_health_query_stops_the_run_after_one_request():
    manifest, fetcher, probe, _ = run(probe_fail={HEALTH})
    assert probe.requests == [HEALTH] and fetcher.requests == []
    assert manifest["stopped"].startswith("probe: the health query failed")


def test_a_probe_listing_without_papers_stops_the_run_too():
    pages = world({**LISTED, "beta": []})
    manifest, fetcher, probe, _ = run(pages)
    assert probe.requests == [HEALTH, listing("alpha"), listing("beta")] and fetcher.requests == []
    assert manifest["stopped"] == "probe: the beta listing answered 200 with 0 papers; nothing else was sent"


def test_the_breaker_stops_after_three_failed_urls_in_a_row():
    fail = {paper_url("2609.00004"), fulltext_url("2609.00004"), paper_url("2609.00005")}
    manifest, fetcher, _, _ = run(fail=fail)
    assert fetcher.requests[-1] == paper_url("2609.00005")  # nothing after the third failure in a row
    assert [f["url"] for f in manifest["failed"]] == [paper_url("2609.00004"), fulltext_url("2609.00004"),
                                                       paper_url("2609.00005")]
    assert manifest["stopped"].startswith("3 URLs in a row failed") and not manifest["complete"]
    assert manifest["metadata"] == {"200": 2, "failed": 2} and manifest["fulltext"] == {"200": 2, "failed": 1}


def test_a_url_that_comes_back_resets_the_breaker():
    fail = {paper_url("2609.00004"), fulltext_url("2609.00004"), fulltext_url("2609.00005"),
            paper_url("2609.00006")}  # two in a row, a success, two in a row
    manifest, fetcher, _, _ = run(fail=fail)
    assert manifest["stopped"] is None and len(manifest["failed"]) == 4 and not manifest["complete"]
    assert len(fetcher.requests) == 4 + 16  # every URL was tried


def test_a_failed_listing_leaves_its_topic_short_and_the_run_incomplete():
    manifest, _, _, _ = run(fail={listing("delta")})
    assert manifest["topics"][3]["papers"] == [] and manifest["topics"][3]["shortfall"] == 2
    assert manifest["papers"] == 6 and manifest["stopped"] is None and not manifest["complete"]


def test_the_tool_adapters_request_exactly_the_recorded_urls():
    seen = []

    class Capture:
        def get(self, url: str) -> Fetched:
            seen.append(url)
            body = b"<html/>" if url.startswith("https://arxiv.org/html/") else feed(["2609.00001"]).encode()
            return Fetched(url=url, final_url=url, status=200, body=body, content_type="text/xml")

    papers = [{"arxiv_id": "2609.00001", "title": "T", "authors": ["A"], "published": "2026-09-01",
               "abstract": "x", "categories": ["cs.CL"]}]
    tools = arxiv_tools(Capture(), build_search_url=lambda args, today: "unused",
                        parse_atom=lambda xml: papers,
                        extract_section=lambda html, section, max_chars: "text", today=DATE_TO)
    tools["get_paper"]({"arxiv_id": "2609.00001"})
    tools["fetch_fulltext"]({"arxiv_id": "2609.00001"})
    assert seen == [paper_url("2609.00001"), fulltext_url("2609.00001")]


def test_the_repositorys_topics_are_the_approved_twenty():
    topics = corpus.load_topics(REPO / corpus.TOPICS)
    assert (topics["window_from"], topics["per_topic"]) == ("2026-07-01", 15)
    assert [t["id"] for t in topics["topics"]] == [
        "tokenization", "position-extrapolation", "mixture-of-experts", "ssm-linear-attention",
        "looped-transformers", "scaling-laws", "pretraining-data", "speculative-decoding", "kv-cache",
        "lora-peft", "reward-models", "preference-optimization", "rl-verifiable-rewards", "llm-as-judge",
        "hallucination", "rag", "quantization", "distillation", "tool-use-agents", "test-time-scaling"]
    assert all(t["phrases"] and t["name"] for t in topics["topics"])


@pytest.mark.parametrize("change, message", [
    (lambda s: s["topics"].append(dict(s["topics"][0])), "repeat"),
    (lambda s: s["topics"][1].update(phrases=[]), "at least one phrase"),
    (lambda s: s.update(per_topic=0), "per_topic"),
])
def test_load_topics_refuses_a_malformed_spec(tmp_path, change, message):
    broken = json.loads(json.dumps(SPEC))
    change(broken)
    path = tmp_path / "topics.json"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        corpus.load_topics(path)


def test_write_manifest_adds_the_date_and_writes_sorted_json(tmp_path):
    manifest, _, _, _ = run()
    path = tmp_path / "data" / "corpus" / "manifest.json"
    corpus.write_manifest(manifest, DATE_TO, path)
    text = path.read_text(encoding="utf-8")
    assert json.loads(text) == {"recorded_on": DATE_TO, **manifest} and text.endswith("}\n")
    assert list(json.loads(text)) == sorted(json.loads(text))


# With the real fetcher over a stand-in arXiv: retries, cassettes and the shared per-host spacing.


class Clock:
    """A clock that moves only when something sleeps on it."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class Arxiv:
    """Stands in for both arXiv hosts with the pages of world(); logs (fake time, URL); 500 for fail."""

    def __init__(self, clock: Clock, fail: set | None = None) -> None:
        self.pages = {str(httpx.URL(url)): page for url, page in world().items()}
        self.clock, self.fail, self.log = clock, fail or set(), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.log.append((self.clock.now, url))
        if url in self.fail:
            return httpx.Response(500)
        status, text = self.pages[url]
        return httpx.Response(status, text=text, headers={"content-type": "text/xml; charset=utf-8"})


def real(tmp_path, server: Arxiv, clock: Clock) -> tuple[Fetcher, Fetcher]:
    limits = HostLimits(clock=clock, sleep=clock.sleep)  # one for both, as scripts/record_corpus.py does
    common = {"limits": limits, "transport": httpx.MockTransport(server), "sleep": clock.sleep,
              "now": lambda: "2026-10-10T14:00:00+00:00"}
    return (Fetcher("record", tmp_path / "cassettes", **common),
            Fetcher("record", tmp_path / "cassettes", retry_delays=(), **common))


def recorded(tmp_path) -> list[str]:
    """The canonical URLs that have a cassette."""
    return sorted(json.loads(p.read_text(encoding="utf-8"))["url"]
                  for p in (tmp_path / "cassettes").rglob("*.json") if p.name != "manifest.json")


def test_with_the_real_fetcher_a_failing_health_query_sends_one_request_and_records_nothing(tmp_path):
    clock = Clock()
    server = Arxiv(clock, fail={str(httpx.URL(HEALTH))})
    fetcher, probe = real(tmp_path, server, clock)
    manifest = corpus.record_corpus(fetcher, probe, SPEC, DATE_TO, log=lambda line: None)
    assert len(server.log) == 1 and manifest["stopped"].startswith("probe: the health query failed")
    assert not (tmp_path / "cassettes").exists() or recorded(tmp_path) == []


def test_with_the_real_fetcher_a_failing_listing_is_sent_once_after_the_health_query(tmp_path):
    clock = Clock()
    server = Arxiv(clock, fail={str(httpx.URL(listing("alpha")))})
    fetcher, probe = real(tmp_path, server, clock)
    manifest = corpus.record_corpus(fetcher, probe, SPEC, DATE_TO, log=lambda line: None)
    assert [url for _, url in server.log] == [str(httpx.URL(HEALTH)), str(httpx.URL(listing("alpha")))]
    assert manifest["stopped"].startswith("probe: the alpha listing failed")
    assert len(recorded(tmp_path)) == 1  # the health query's answer only


def test_with_the_real_fetcher_the_spacing_holds_throughout_and_a_second_run_sends_nothing(tmp_path):
    clock = Clock()
    server = Arxiv(clock)
    fetcher, probe = real(tmp_path, server, clock)
    first = corpus.record_corpus(fetcher, probe, SPEC, DATE_TO, log=lambda line: None)
    assert first["complete"] and len(server.log) == 1 + 4 + 16  # the probed listings are not sent again
    for host, interval in (("export.arxiv.org", 3.0), ("arxiv.org", 15.0)):
        times = [t for t, url in server.log if httpx.URL(url).host == host]
        assert all(b - a >= interval for a, b in itertools.pairwise(times)), host
    again = Arxiv(clock)
    fetcher, probe = real(tmp_path, again, clock)
    second = corpus.record_corpus(fetcher, probe, SPEC, DATE_TO, log=lambda line: None)
    assert again.log == [] and second == first


# scripts/record_corpus.py


def test_the_plan_counts_requests_and_the_time_the_slower_host_needs():
    text = cli.plan(corpus.load_topics(REPO / corpus.TOPICS))
    assert text == ("20 topics x 15 papers = 300 papers; 321 API requests (1 health query, 20 listings of "
                    "50, 300 metadata) and 300 full texts; the two hosts wait in parallel, so at least 75 "
                    "min")


def test_a_dry_run_prints_the_listing_queries_and_sends_nothing(monkeypatch, capsys):
    monkeypatch.setattr(cli, "Fetcher", lambda *args, **kwargs: pytest.fail("a dry run built a fetcher"))
    assert cli.main(["--dry-run", "--date-to", DATE_TO, "--topics", str(REPO / corpus.TOPICS)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 1 + 20 + 1 and out[1].startswith("tokenization: https://export.arxiv.org/api/query?")
    assert out[-1] == (f"probe: {corpus.health_url('tokenizer')}, then the first 3 listings, once each, "
                       f"without retries")
    assert corpus.health_url("tokenizer") == ("https://export.arxiv.org/api/query?search_query=all:%22tokenizer%22"
                                              "&start=0&max_results=1")

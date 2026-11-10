"""The arXiv corpus that M2 replays (docs/m2-录制计划.md): topic listings, the papers chosen from them and
the recording of their metadata and full texts.

The topics come from data/corpus/topics.json, which the owner approves; their order is the deduplication
order. For each topic one listing query asks the arXiv API for LISTING_SIZE papers that match any of the
topic's phrases, in the six search categories (schemas.CATEGORIES), submitted in the window, by relevance;
the first per_topic papers not chosen for an earlier topic are kept. Each chosen paper's metadata (the API's
id_list answer) and full text (arxiv.org/html/<id>) are recorded at exactly the URLs the tool adapters request
(tools.adapters.paper_url and fulltext_url), so that replay finds them. A paper without an HTML version
answers 404, which is recorded as well: in replay, fetch_fulltext then takes its abstract-instead path.

Paper ids are read from the listings' <id> elements with one regular expression and nothing else is read from
the feeds: parsing them is the owner's parse_atom (docs/modules/接口.md). The listing query is written the
way arXiv's API manual writes its examples: "+" for spaces, colons and the date range's brackets as they are.

Everything goes through net.fetch in record mode: per-host rate limits, a recorded URL is never fetched again,
and RecordFailed for an answer that cannot be recorded. Two safeguards keep a bad night from turning into a
burst of requests (docs/m2-录制计划.md, the probe of 2026-10-09):
- probe: a health query in the simplest form (one phrase, no category or date terms, one result), then the
  first PROBE_TOPICS listings, each sent once through a fetcher without retries; the first request that fails
  or lists no paper stops the run before anything else is sent. A failing health query points at the service,
  a health query that passes before a failing listing points at the listing query;
- breaker: once BREAKER_LIMIT URLs in a row have failed, each after the fetcher's own retries, the run stops.
A stopped run keeps what it recorded, and the next run fetches only what is missing. The manifest
(data/corpus/manifest.json, committed) holds the topics, the chosen ids, the statuses and what failed.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Protocol
from urllib.parse import quote_plus, urlencode

from ..agent.types import HarnessError
from ..net.fetch import Fetched, RecordFailed, canonical_url
from ..tools import schemas
from ..tools.adapters import API_URL, fulltext_url, paper_url

TOPICS = Path("data/corpus/topics.json")
MANIFEST = Path("data/corpus/manifest.json")
LISTING_SIZE = 50  # papers asked for per topic; per_topic of them are kept after deduplication
PROBE_TOPICS = 3  # listings sent once each, without retries, before anything else
BREAKER_LIMIT = 3  # failed URLs in a row that stop a run
_LISTED_ID = re.compile(r"<id>\s*https?://arxiv\.org/abs/(\d{4}\.\d{4,5})(?:v\d+)?\s*</id>")


class SupportsGet(Protocol):
    def get(self, url: str) -> Fetched: ...


class Stopped(HarnessError):
    """The run stopped on purpose: a probe listing failed or listed nothing, or the breaker tripped."""


def load_topics(path: str | Path = TOPICS) -> dict:
    """The topic spec (window_from, per_topic, topics with id, name, phrases), shape checked."""
    spec = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [topic["id"] for topic in spec["topics"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"topic ids repeat: {sorted(i for i in set(ids) if ids.count(i) > 1)}")
    if any(not topic["phrases"] for topic in spec["topics"]):
        raise ValueError("every topic needs at least one phrase")
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", spec["window_from"]) or int(spec["per_topic"]) < 1:
        raise ValueError("window_from must be YYYY-MM-DD and per_topic at least 1")
    return spec


def listing_url(phrases: list[str], date_from: str, date_to: str, *,
                categories: list[str] = schemas.CATEGORIES, size: int = LISTING_SIZE) -> str:
    """The arXiv API query for papers matching any phrase, in categories, submitted from date_from to date_to
    (both days included, YYYY-MM-DD), most relevant first."""
    match = " OR ".join(f'all:"{phrase}"' for phrase in phrases)
    cats = " OR ".join(f"cat:{category}" for category in categories)
    window = f"submittedDate:[{date_from.replace('-', '')}0000 TO {date_to.replace('-', '')}2359]"
    params = {"search_query": f"({match}) AND ({cats}) AND {window}", "start": 0, "max_results": size,
              "sortBy": "relevance", "sortOrder": "descending"}
    return f"{API_URL}?{urlencode(params, quote_via=quote_plus, safe=':[]')}"


def health_url(phrase: str) -> str:
    """The probe's first request: one phrase and nothing else, one result."""
    params = {"search_query": f'all:"{phrase}"', "start": 0, "max_results": 1}
    return f"{API_URL}?{urlencode(params, quote_via=quote_plus, safe=':[]')}"


def listed_ids(feed: str) -> list[str]:
    """The paper ids of a listing in its order, without version and without repeats (only <id> is read)."""
    return list(dict.fromkeys(_LISTED_ID.findall(feed)))


def choose(listings: dict[str, list[str]], per_topic: int) -> dict[str, list[str]]:
    """per_topic ids for each topic in order; an id chosen for an earlier topic is skipped for the next."""
    taken: set[str] = set()
    chosen = {}
    for topic, ids in listings.items():
        picks = [arxiv_id for arxiv_id in ids if arxiv_id not in taken][:per_topic]
        taken.update(picks)
        chosen[topic] = picks
    return chosen


class Breaker:
    """Counts failed URLs in a row and stops the run at the limit; a URL that came back resets the count."""

    def __init__(self, limit: int = BREAKER_LIMIT) -> None:
        self.limit, self.in_a_row = limit, 0

    def passed(self) -> None:
        self.in_a_row = 0

    def failed(self, url: str, reason: str) -> None:
        self.in_a_row += 1
        if self.in_a_row >= self.limit:
            raise Stopped(f"{self.in_a_row} URLs in a row failed, the last {url}: {reason}")


def record_corpus(fetcher: SupportsGet, probe: SupportsGet, spec: dict, date_to: str, *,
                  log: Callable[[str], None] = print, breaker_limit: int = BREAKER_LIMIT) -> dict:
    """Record the corpus of spec up to date_to; returns the manifest (stopped holds the reason a run stopped).

    fetcher records with retries; probe records without them. Both must share one HostLimits, so that the
    switch from one to the other keeps the per-host spacing.
    """
    topics, per_topic = spec["topics"], int(spec["per_topic"])
    window = [spec["window_from"], date_to]
    urls = {topic["id"]: listing_url(topic["phrases"], *window) for topic in topics}
    manifest = {"window": window, "categories": list(schemas.CATEGORIES), "listing_size": LISTING_SIZE,
                "per_topic": per_topic, "topics": [], "papers": 0, "metadata": {}, "fulltext": {},
                "failed": [], "stopped": None, "complete": False}
    breaker = Breaker(breaker_limit)
    statuses: dict[str, Counter] = {"metadata": Counter(), "fulltext": Counter()}

    def get(source: SupportsGet, url: str, kind: str | None = None) -> Fetched | None:
        """The page, or None after a failure; kind ("metadata", "fulltext") is counted before the breaker
        may stop the run, so the counts always agree with the failed list."""
        try:
            page = source.get(url)
        except RecordFailed as exc:
            manifest["failed"].append({"url": canonical_url(url), "status": exc.status, "reason": exc.reason})
            if kind is not None:
                statuses[kind]["failed"] += 1
            log(f"failed: {url} ({exc.reason})")
            breaker.failed(url, exc.reason)
            return None
        breaker.passed()
        if kind is not None:
            statuses[kind][str(page.status)] += 1
        return page

    try:
        probes = [("health query", health_url(topics[0]["phrases"][0]))]
        probes += [(f"{topic['id']} listing", urls[topic["id"]]) for topic in topics[:PROBE_TOPICS]]
        for what, url in probes:
            try:
                page = probe.get(url)
            except RecordFailed as exc:
                raise Stopped(f"probe: the {what} failed ({exc.reason}); nothing else was sent") from exc
            if page.status != 200 or not listed_ids(page.text):
                raise Stopped(f"probe: the {what} answered {page.status} with {len(listed_ids(page.text))} "
                              f"papers; nothing else was sent")
            log(f"probe: {what} has {len(listed_ids(page.text))} papers")
        listings = {}
        for topic in topics:
            page = get(fetcher, urls[topic["id"]])
            listings[topic["id"]] = listed_ids(page.text) if page is not None and page.status == 200 else []
        chosen = choose(listings, per_topic)
        for topic in topics:
            picks = chosen[topic["id"]]
            manifest["topics"].append({**topic, "listing": canonical_url(urls[topic["id"]]),
                                       "listed": len(listings[topic["id"]]), "papers": picks,
                                       "shortfall": per_topic - len(picks)})
            log(f"{topic['id']}: {len(listings[topic['id']])} listed, {len(picks)} chosen")
        manifest["papers"] = sum(len(picks) for picks in chosen.values())
        done = 0
        for picks in chosen.values():
            for arxiv_id in picks:  # metadata then full text: the two hosts' waits overlap
                get(fetcher, paper_url(arxiv_id), "metadata")
                get(fetcher, fulltext_url(arxiv_id), "fulltext")
                done += 1
                if done % 10 == 0 or done == manifest["papers"]:
                    log(f"papers: {done}/{manifest['papers']}")
        manifest["complete"] = not manifest["failed"]
    except Stopped as exc:
        manifest["stopped"] = str(exc)
        log(f"stopped: {exc}")
    manifest["metadata"] = dict(sorted(statuses["metadata"].items()))
    manifest["fulltext"] = dict(sorted(statuses["fulltext"].items()))
    return manifest


def write_manifest(manifest: dict, recorded_on: str, path: str | Path = MANIFEST) -> None:
    """The manifest with the recording date, as sorted, indented JSON with a final newline."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps({"recorded_on": recorded_on, **manifest}, ensure_ascii=False, indent=2, sort_keys=True)
    target.write_text(text + "\n", encoding="utf-8", newline="\n")

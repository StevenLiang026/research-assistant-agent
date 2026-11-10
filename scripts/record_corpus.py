"""Record the M2 arXiv corpus (docs/m2-录制计划.md) into data/cassettes, probe first.

The logic is research_agent.data.corpus. The probe sends a health query in the simplest form and then the
first three topic listings, each once and without retries, and stops the run at the first request that fails
or lists nothing; after that everything is recorded
with the fetcher's retries, and three failed URLs in a row stop the run. A stopped run keeps what it
recorded and a second run fetches only what is missing. --dry-run prints the listing queries and the request
plan and sends nothing. The fetcher uses the environment's connection settings, like every live fetch in
this project.

Run:  .venv/Scripts/python.exe scripts/record_corpus.py [--date-to YYYY-MM-DD] [--dry-run]
Exit: 0 complete; 3 stopped or incomplete (the manifest's stopped and failed fields say why).
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
import time

from research_agent.data.corpus import (
    LISTING_SIZE,
    MANIFEST,
    PROBE_TOPICS,
    TOPICS,
    health_url,
    listing_url,
    load_topics,
    record_corpus,
    write_manifest,
)
from research_agent.net.fetch import Fetcher
from research_agent.net.ratelimit import DEFAULT_INTERVALS, HostLimits

CASSETTES = "data/cassettes"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Record the M2 arXiv corpus, probe first.")
    ap.add_argument("--topics", default=str(TOPICS))
    ap.add_argument("--cassettes", default=CASSETTES)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--date-to", default=dt.datetime.now().astimezone().date().isoformat(),
                    help="last submission day of the window and the recording date (default: today, local)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the listing queries only")
    return ap


def plan(spec: dict) -> str:
    """Requests and the time they take at the per-host intervals, every paper's full text included."""
    topics, papers = len(spec["topics"]), len(spec["topics"]) * int(spec["per_topic"])
    api, html = 1 + topics + papers, papers
    minutes = max(api * DEFAULT_INTERVALS["export.arxiv.org"], html * DEFAULT_INTERVALS["arxiv.org"]) / 60
    return (f"{topics} topics x {spec['per_topic']} papers = {papers} papers; {api} API requests "
            f"(1 health query, {topics} listings of {LISTING_SIZE}, {papers} metadata) and {html} full "
            f"texts; the two hosts wait in parallel, so at least {minutes:.0f} min")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    spec = load_topics(args.topics)
    print(plan(spec), flush=True)
    if args.dry_run:
        for topic in spec["topics"]:
            print(f"{topic['id']}: {listing_url(topic['phrases'], spec['window_from'], args.date_to)}")
        health = health_url(spec["topics"][0]["phrases"][0])
        print(f"probe: {health}, then the first {PROBE_TOPICS} listings, once each, without retries")
        return 0
    limits = HostLimits()  # shared: the probe and the recording keep one spacing per host
    probe = Fetcher("record", args.cassettes, limits=limits, retry_delays=())
    fetcher = Fetcher("record", args.cassettes, limits=limits)
    started = time.monotonic()
    try:
        manifest = record_corpus(fetcher, probe, spec, args.date_to,
                                 log=lambda line: print(line, flush=True))
    finally:
        probe.close()
        fetcher.close()
    write_manifest(manifest, args.date_to, args.manifest)
    print(f"{'complete' if manifest['complete'] else 'NOT complete'} after "
          f"{(time.monotonic() - started) / 60:.1f} min: {manifest['papers']} papers, metadata "
          f"{manifest['metadata']}, full text {manifest['fulltext']}, {len(manifest['failed'])} failed; "
          f"manifest {args.manifest}")
    return 0 if manifest["complete"] else 3


if __name__ == "__main__":
    sys.exit(main())

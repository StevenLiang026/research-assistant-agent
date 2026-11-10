"""Check the replay search (research_agent.tools.corpus_search) over the recorded corpus, offline.

The index is built from the id_list cassettes with research_agent.arxiv.parse_atom, or with the parse_atom
--parse-atom names (any module path works, inside or outside the repository); the index line says which
parser was used. Four things are measured:

(a) phrase relevance, P@5: for every topic in the manifest and every phrase in its phrases list, the query is
    the phrase in quotes, within the manifest's window, through the tool; a hit is a paper whose title or
    abstract contains the phrase (case-insensitive, whitespace collapsed). The ceiling printed next to each
    phrase is what a perfect search could score: a phrase that occurs in fewer than five papers cannot reach
    1.0, and a phrase written only in a variant form (plural, hyphenated) scores 0 by this literal criterion.
(b) self-retrieval: for every paper, the query is its title; the paper must rank first in search().
(c) diagnostic only: P@5 of the OR of a topic's phrases against the topic's own paper ids (the listing's
    choice is not what the search optimises, so no threshold). The search reads OR as alternatives, so for a
    topic with several phrases this measures the papers that carry any of them, ranked by BM25 plus the bonus
    of each phrase they carry.
(d) three Chinese queries must answer EMPTY_RESULTS: the model has to search in English.

PASS needs (a) mean over all phrases >= PHRASE_THRESHOLD and (b) >= SELF_THRESHOLD; FAIL exits 1. No
network access; nothing but the cassettes and the manifest is read.

Run:  .venv/Scripts/python.exe scripts/check_corpus_search.py [--parse-atom <module.py>:<function>]
          [--cassettes data/cassettes] [--manifest data/corpus/manifest.json] [--json <out.json>]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

from research_agent import arxiv
from research_agent.agent.types import HarnessError
from research_agent.net.fetch import Fetcher
from research_agent.tools import schemas
from research_agent.tools.corpus_search import MANIFEST, CorpusIndex, corpus_ids, load_papers

CASSETTES = "data/cassettes"
TOP_K = 5
PHRASE_THRESHOLD = 0.9
SELF_THRESHOLD = 0.95
CHINESE_QUERIES = ["投机解码", "混合专家 路由", "大模型 量化"]
MODULE_PARSER = "research_agent.arxiv.parse_atom"  # the parser used without --parse-atom


def load_function(spec: str) -> Callable:
    """The function named by "<path.py>:<name>", imported from that file, which may lie outside the repo."""
    path, sep, name = spec.rpartition(":")  # the last colon: a Windows drive letter has one of its own
    if not sep or not path or not name.isidentifier():
        raise SystemExit(f"--parse-atom must be <path.py>:<function>, not {spec!r}")
    module_spec = importlib.util.spec_from_file_location(f"parse_atom_module_{Path(path).stem}", path)
    if module_spec is None or module_spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    try:
        return getattr(module, name)
    except AttributeError:
        raise SystemExit(f"{path} has no function {name}") from None


def read_manifest(path: str | Path) -> dict:
    """The manifest's ids (corpus_ids, whose checks run first, so a manifest without papers is a HarnessError
    naming the fault rather than a KeyError), its topics (id, phrases, papers) and its window: absent or null
    for no bounds, else a list of two date strings. What corpus_ids accepts but the script cannot use, a topic
    without an id string or a window of another shape, is a HarnessError naming the fault in the same style.
    """
    path = Path(path)
    ids = corpus_ids(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    window = data.get("window")
    if window is None:
        window = [None, None]
    elif not (isinstance(window, list) and len(window) == 2 and all(isinstance(day, str) for day in window)):
        raise HarnessError(f"corpus manifest {path}: window is not a list of two strings: {window!r}")
    topics = []
    for position, topic in enumerate(data["topics"]):
        if not isinstance(topic.get("id"), str):
            raise HarnessError(f"corpus manifest {path}: topics[{position}] has no id string")
        topics.append({"id": topic["id"], "phrases": list(topic.get("phrases", [])),
                       "papers": list(topic["papers"])})
    return {"ids": ids, "topics": topics, "window": list(window)}


def normalised(text: str, hyphens: bool = False) -> str:
    """Casefolded, whitespace collapsed; with hyphens=True a hyphen counts as whitespace too."""
    return " ".join((text.replace("-", " ") if hyphens else text).casefold().split())


def phrase_in(paper: dict, phrase: str, hyphens: bool = False) -> bool:
    """Whether the phrase occurs literally (case-insensitive, whitespace collapsed) in title or abstract; with
    hyphens=True "mixture-of-experts" also counts as "mixture of experts", as the index's tokens do."""
    wanted = normalised(phrase, hyphens)
    return wanted in normalised(paper["title"], hyphens) or wanted in normalised(paper["abstract"], hyphens)


def precision_at(ranked_ids: list[str], relevant: set[str], k: int = TOP_K) -> float:
    """The share of the first k ranked ids that are relevant, over k (fewer than k hits count as misses)."""
    return sum(1 for arxiv_id in ranked_ids[:k] if arxiv_id in relevant) / k


def window_args(query: str, window: list) -> dict:
    args = {"query": query}
    if window[0]:
        args["date_from"] = window[0]
    if window[1]:
        args["date_to"] = window[1]
    return args


def phrase_relevance(index: CorpusIndex, by_id: dict[str, dict], topics: list[dict], window: list) -> dict:
    """(a): per topic, per phrase P@5 (the literal criterion, which the threshold applies to), its ceiling,
    and the hyphen-insensitive P@5 beside it; the means over all phrases."""
    tool = index.tool()
    report: dict = {"topics": {}, "phrases": 0, "mean": 0.0, "ceiling": 0.0, "mean_hyphen_insensitive": 0.0}
    scores, ceilings, loose = [], [], []
    for topic in topics:
        rows = {}
        for phrase in topic["phrases"]:
            outcome = tool(window_args(f'"{phrase}"', window))
            top = [hit["arxiv_id"] for hit in outcome.content]
            relevant = {arxiv_id for arxiv_id, paper in by_id.items() if phrase_in(paper, phrase)}
            relaxed = {arxiv_id for arxiv_id, paper in by_id.items() if phrase_in(paper, phrase, True)}
            rows[phrase] = {"p_at_5": precision_at(top, relevant), "literal_matches": len(relevant),
                            "ceiling": min(len(relevant), TOP_K) / TOP_K,
                            "p_at_5_hyphen_insensitive": precision_at(top, relaxed), "top": top}
            scores.append(rows[phrase]["p_at_5"])
            ceilings.append(rows[phrase]["ceiling"])
            loose.append(rows[phrase]["p_at_5_hyphen_insensitive"])
        mean = sum(row["p_at_5"] for row in rows.values()) / len(rows) if rows else 0.0
        report["topics"][topic["id"]] = {"mean": mean, "phrases": rows}
    report["phrases"] = len(scores)
    report["mean"] = sum(scores) / len(scores) if scores else 0.0
    report["ceiling"] = sum(ceilings) / len(ceilings) if ceilings else 0.0
    report["mean_hyphen_insensitive"] = sum(loose) / len(loose) if loose else 0.0
    return report


def self_retrieval(index: CorpusIndex, papers: list[dict]) -> dict:
    """(b): the share of papers that rank first for their own title, and the ids of those that do not."""
    missed = []
    for paper in papers:
        ranked = index.search({"query": paper["title"]})
        if not ranked or ranked[0]["arxiv_id"] != paper["arxiv_id"]:
            missed.append({"arxiv_id": paper["arxiv_id"],
                           "first": ranked[0]["arxiv_id"] if ranked else None})
    fraction = (len(papers) - len(missed)) / len(papers) if papers else 0.0
    return {"fraction": fraction, "papers": len(papers), "missed": missed}


def topic_or(index: CorpusIndex, topics: list[dict], window: list) -> dict:
    """(c): P@5 of the OR of a topic's quoted phrases against the topic's own ids; diagnostic only."""
    tool = index.tool()
    rows = {}
    for topic in topics:
        if not topic["phrases"]:
            continue
        query = " OR ".join(f'"{phrase}"' for phrase in topic["phrases"])
        outcome = tool(window_args(query, window))
        rows[topic["id"]] = precision_at([hit["arxiv_id"] for hit in outcome.content], set(topic["papers"]))
    return {"topics": rows, "mean": sum(rows.values()) / len(rows) if rows else 0.0}


def chinese_queries(index: CorpusIndex, queries: list[str] = CHINESE_QUERIES) -> dict:
    """(d): every query's error code; all must be EMPTY_RESULTS."""
    tool = index.tool()
    codes = {query: tool({"query": query}).error for query in queries}
    return {"codes": codes, "all_empty": all(code == schemas.EMPTY_RESULTS for code in codes.values())}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Check the replay search over the recorded corpus (offline).")
    ap.add_argument("--parse-atom", default=None, metavar="PATH.py:FUNCTION",
                    help=f"the parse_atom to build the index with (default: {MODULE_PARSER})")
    ap.add_argument("--cassettes", default=CASSETTES)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--json", default=None, metavar="OUT", help="also write the full report as JSON")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")  # a cp1252 console must not fail on the queries
    parse_atom = load_function(args.parse_atom) if args.parse_atom else arxiv.parse_atom
    parser_name = args.parse_atom or MODULE_PARSER
    manifest = read_manifest(args.manifest)
    started = time.perf_counter()
    papers = load_papers(Fetcher("replay", args.cassettes), manifest["ids"], parse_atom)
    index = CorpusIndex(papers)
    build_seconds = time.perf_counter() - started
    by_id = {paper["arxiv_id"]: paper for paper in papers}
    print(f"index: {len(index)} papers from {args.manifest}, built in {build_seconds:.2f} s, "
          f"parser {parser_name}")

    relevance = phrase_relevance(index, by_id, manifest["topics"], manifest["window"])
    print(f"\n(a) phrase relevance, P@{TOP_K} of the quoted phrase within {manifest['window']}: per phrase "
          f"the literal P@{TOP_K} (its ceiling, the literal matches in the corpus; hyphen-insensitive "
          f"P@{TOP_K}):")
    for topic_id, row in relevance["topics"].items():
        detail = ", ".join(f'"{phrase}" {cell["p_at_5"]:.2f} (ceiling {cell["ceiling"]:.2f}, '
                           f'{cell["literal_matches"]} literal; {cell["p_at_5_hyphen_insensitive"]:.2f})'
                           for phrase, cell in row["phrases"].items())
        print(f"  {topic_id:26s} {row['mean']:.2f}  {detail}")
    print(f"  mean over {relevance['phrases']} phrases: {relevance['mean']:.3f} (ceiling "
          f"{relevance['ceiling']:.3f}; hyphen-insensitive {relevance['mean_hyphen_insensitive']:.3f}; "
          f"threshold {PHRASE_THRESHOLD})")

    own = self_retrieval(index, papers)
    print(f"\n(b) self-retrieval: {own['fraction']:.3f} of {own['papers']} papers rank first for their title "
          f"(threshold {SELF_THRESHOLD})")
    for miss in own["missed"]:
        print(f"  {miss['arxiv_id']} ranked behind {miss['first']}")

    by_topic = topic_or(index, manifest["topics"], manifest["window"])
    print(f"\n(c) diagnostic, P@{TOP_K} of the OR of a topic's phrases against its own ids:")
    for topic_id, value in by_topic["topics"].items():
        print(f"  {topic_id:26s} {value:.2f}")
    print(f"  mean: {by_topic['mean']:.3f}")

    chinese = chinese_queries(index)
    print("\n(d) Chinese queries:")
    for query, code in chinese["codes"].items():
        print(f"  {query!r}: {code}")

    passed = (relevance["mean"] >= PHRASE_THRESHOLD and own["fraction"] >= SELF_THRESHOLD
              and chinese["all_empty"])
    verdict = "PASS" if passed else "FAIL"
    print(f"\n{verdict}: phrase relevance {relevance['mean']:.3f} >= {PHRASE_THRESHOLD}, self-retrieval "
          f"{own['fraction']:.3f} >= {SELF_THRESHOLD}, Chinese queries empty {chinese['all_empty']}")
    if args.json:
        report = {"papers": len(index), "build_seconds": round(build_seconds, 3), "schema_hash":
                  schemas.SCHEMA_HASH, "phrase_relevance": relevance, "self_retrieval": own,
                  "topic_or": by_topic, "chinese": chinese, "verdict": verdict}
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8", newline="\n")
        print(f"report: {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

"""search_arxiv over the recorded corpus: a replay search that needs no cassette per query.

The teacher model writes a different query every time, so replaying search_arxiv by URL would miss nearly
every request (docs/m2-录制计划.md). Instead, the metadata of the 300 recorded papers (named by
data/corpus/manifest.json; each one's id_list cassette holds its Atom feed) is parsed once with the owner's
parse_atom, held to the contract of the live adapters (tools.adapters.parsed_papers), and indexed here. The
tool CorpusIndex.tool() returns answers exactly like adapters.arxiv_tools' search_arxiv would, hit for hit
(adapters.hit, at most MAX_SEARCH_RESULTS, EMPTY_RESULTS when nothing matches), and it is a plain ToolFn, so
a fault hook for EMPTY_RESULTS, RATE_LIMITED and UPSTREAM_ERROR can wrap it (net.faults.SearchFaults wraps
it; data/synth.make_tools does so per request). It never fetches: load_papers reads cassettes through
Fetcher.recorded, which sends nothing and waits for no rate limit, and a missing or non-200 cassette is a
HarnessError, as a replay miss is.

Query language. The model is prompted with arXiv's query syntax, so parse_query accepts a subset of it and
approximates the rest; it does not implement arXiv's search. Quoted segments are phrases, and a phrase is
required within its alternative: the upper-case operator OR (outside quotes) splits the query into
alternatives, and as in arXiv's search a paper is a hit only when, for at least one alternative, it contains
every quoted phrase of that alternative contiguously in its title or abstract (a phrase's tokens also count
as terms, whichever alternative they stand in). So "speculative decoding" OR "draft model" is answered by the
papers holding either phrase, "speculative decoding" "draft model", with or without AND between them, by
those holding both. An alternative with terms but no phrase requires nothing: "draft model" OR draft is
satisfied by every candidate. An alternative without a term (a dangling or doubled OR, a Chinese operand, a
lone cat: or NOT) cannot match a paper on its own, so it is dropped from the alternation while its cat: and
NOT still apply to the whole query, as stated below: "draft model" OR cat:cs.CL requires the phrase, within
cs.CL. Parentheses are dropped before the split, so nesting is not supported: ("A" OR "B") AND "C" reads as
"A" OR ("B" AND "C"), a hit being a paper holding A, or one holding both B and C, never one holding B alone
or C alone. The field prefixes ti:, abs:, all: and au: are stripped (every term searches title and abstract;
authors are not indexed). cat:X adds X to the query's categories, whichever alternative holds it, mapped onto
schemas.CATEGORIES case-insensitively (cat:CS.cl is cs.CL); a value outside that list is kept as written and
matches only a paper whose categories list it verbatim (a cross-list such as math.OC), usually nothing.
submittedDate:[...] ranges are dropped (an unclosed range drops only its bracketed token, a bare
submittedDate: only itself; the words around them stay terms), and so are parentheses and the operator AND
(phrases side by side are already all required). Operators are upper-case only, as on arXiv: AND, OR, ANDNOT
and its lenient alias NOT; "and", "or", "not" and "Not" are ordinary terms (every self-retrieval miss on the
recorded corpus was a title containing "Not"). The unit after ANDNOT or NOT is excluded from the whole query,
whichever alternative holds it ("A" OR "B" ANDNOT survey drops a paper containing survey even when it holds
A): a word's whole token, or all the tokens of a quoted phrase; a cat: token after it is dropped, not
collected as a filter, and a NOT at the end of an alternative excludes nothing. Everything else is tokenized
as terms.

Tokens. Text is NFKC-normalised and casefolded; a token is [a-z0-9]+(?:[-'][a-z0-9]+)*, and a token with a
hyphen or apostrophe also contributes its parts (mixture-of-experts -> mixture-of-experts, mixture, of,
experts), so a query written either way meets a paper written either way. A phrase is matched in the
parts-only sequence (phrase_tokens: hyphenated tokens replaced by their parts) of the query and of each
field, so "mixture of experts" and "mixture-of-experts" meet each other in both directions, wherever the
hyphenated word stands in the phrase. Text without Latin letters or digits (Chinese) yields no tokens, so a
Chinese query is EMPTY_RESULTS: the model has to search in English. Excluded words are not split into parts
(NOT mixture-of-experts excludes the compound token only).

Scoring. BM25 (k1 = K1, b = B) per field, title weighted TITLE_WEIGHT and abstract ABSTRACT_WEIGHT, each
field with its own average length; idf = ln((N - df + 0.5) / (df + 0.5) + 1) with df counted over the union
of a paper's title and abstract tokens. Each distinct query term is scored once. A paper is a candidate when
it contains at least one term in either field (a query without terms, Chinese or cat: alone, matches
nothing) and, when the query has phrases, every phrase of at least one alternative contiguously in its title
or its abstract; a paper that contains an excluded token in either field is dropped. Each phrase a candidate
contains adds PHRASE_BONUS times the sum of its tokens' idf, once per phrase: with one required phrase every
candidate gets the same bonus, which raises the scores of a phrase query without reordering its candidates,
and with an OR of phrases a paper holding several of them gets each one's bonus. Filters: categories (the
args list intersected with the query's cat: terms when both are given; a paper passes when its categories
share at least one with the filter; an empty or absent list is no filter) and date_from / date_to
(inclusive, compared as ISO strings with the paper's published date). sort_by "relevance" orders by
(-score, arxiv_id); "submittedDate" by newest first, then arxiv_id descending. sort_by, categories,
date_from and date_to may be absent or null, which means their default (relevance, no filter); max_results,
read by the tool alone, must be an integer when given (an integral float counts, as adapters.capped reads
it), so a null one is a TypeError, and an absent one is the schema's default.

Determinism. The index orders papers by arxiv_id and every aggregate (df, lengths) is an integer, so the
same papers in any order build the same index and the same args give the same outcome, bit for bit.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Hashable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..agent.types import HarnessError, ToolFn, ToolOutcome
from ..net.fetch import Fetched
from . import schemas
from .adapters import ParseAtom, capped, hit, paper_url, parsed_papers

MANIFEST = Path("data/corpus/manifest.json")
INDEX_CALLER = "corpus_index"  # the caller name a parse_atom contract breach is reported under
K1, B = 1.2, 0.75
TITLE_WEIGHT, ABSTRACT_WEIGHT = 3.0, 1.0
WEIGHTS = {"title": TITLE_WEIGHT, "abstract": ABSTRACT_WEIGHT}
PHRASE_BONUS = 0.5
SORT_ORDERS = ("relevance", "submittedDate")

_TOKEN = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_PART = re.compile(r"[-']")
# a closed range, or the bare prefix with whatever sticks to it: an unclosed "[" never reaches a later word
_DATE_RANGE = re.compile(r"submittedDate:(?:\[[^\]]*\]|\S*)", re.IGNORECASE)
_PREFIX = re.compile(r"\b(?:ti|abs|all|au):", re.IGNORECASE)
# a quoted phrase (to the end if unclosed), a paren, a cat: token, a word
_UNIT = re.compile(r'"([^"]*)"?|([()])|cat:([^\s()"]+)|(\S+)', re.IGNORECASE)
_AND, _OR = "AND", "OR"  # upper-case only: "and" and "or" are terms; AND is dropped, OR splits
_EXCLUDING = ("ANDNOT", "NOT")  # upper-case only: "not" and "Not" are terms
_CATEGORY_BY_FOLD = {category.casefold(): category for category in schemas.CATEGORIES}


class SupportsRecorded(Protocol):
    """Anything that reads cassettes without sending requests: net.fetch.Fetcher in any mode, or a double."""

    def recorded(self, url: str) -> Fetched | None: ...


@dataclass(frozen=True)
class Query:
    """What parse_query reads out of a query string: distinct terms in first-occurrence order, phrases as
    token tuples, excluded tokens and the cat: categories, all in query order, over the whole query; and
    alternatives, the phrases of each alternative OR splits the query into (distinct, in query order; a
    text without OR is one alternative holding every phrase, a text without a term has none). phrases is
    the flat union of the alternatives' phrases, for a caller that only needs the list."""

    terms: tuple[str, ...]
    phrases: tuple[tuple[str, ...], ...]
    excluded: tuple[str, ...]
    categories: tuple[str, ...]
    alternatives: tuple[tuple[tuple[str, ...], ...], ...]


class _Alternative:
    """What one alternative of a query yields, read unit by unit; the unit after ANDNOT or NOT is excluded."""

    def __init__(self) -> None:
        self.terms: list[str] = []
        self.phrases: list[tuple[str, ...]] = []
        self.excluded: list[str] = []
        self.categories: list[str] = []
        self.exclude_next = False

    def read(self, quoted: str | None, category: str | None, word: str | None) -> None:
        """One unit of _UNIT: a quoted phrase, a cat: value or a word (never a paren or an operator)."""
        if word in _EXCLUDING:
            self.exclude_next = True
            return
        if self.exclude_next:  # the unit after NOT: a word or phrase is excluded, a cat: token dropped
            self.exclude_next = False
            if category is None:
                self.excluded.extend(_whole_tokens(quoted if quoted is not None else word))
            return
        if category is not None:
            self.categories.append(_CATEGORY_BY_FOLD.get(category.casefold(), category))
            return
        unit = quoted if quoted is not None else word
        self.terms.extend(tokenize(unit))
        if quoted is not None and (phrase := phrase_tokens(unit)):
            self.phrases.append(phrase)


def tokenize(text: str) -> list[str]:
    """The tokens of text in order, a hyphenated or apostrophe token followed by its parts."""
    tokens: list[str] = []
    for token in _TOKEN.findall(unicodedata.normalize("NFKC", text).casefold()):
        tokens.append(token)
        if "-" in token or "'" in token:
            tokens.extend(_PART.split(token))
    return tokens


def _whole_tokens(text: str) -> list[str]:
    """The tokens of text without the parts of hyphenated ones: what an excluded word excludes."""
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())


def phrase_tokens(text: str) -> tuple[str, ...]:
    """The tokens of text with every hyphenated or apostrophe token replaced by its parts: the sequence a
    phrase is matched in, on both sides, so "mixture of experts" and "mixture-of-experts" meet each other
    wherever the hyphenated word stands in the phrase."""
    return tuple(part for token in _whole_tokens(text) for part in _PART.split(token))


def parse_query(text: str) -> Query:
    """The subset of arXiv's query syntax the module docstring states, read from text."""
    text = _PREFIX.sub("", _DATE_RANGE.sub(" ", text))
    alternatives = [_Alternative()]
    for match in _UNIT.finditer(text):
        quoted, paren, category, word = match.groups()
        if paren is not None or word == _AND:
            continue
        if word == _OR:
            alternatives.append(_Alternative())
        else:
            alternatives[-1].read(quoted, category, word)
    matching = [alternative for alternative in alternatives if alternative.terms]  # the ones a paper can hold
    return Query(terms=_union(alternative.terms for alternative in alternatives),
                 phrases=_union(alternative.phrases for alternative in alternatives),
                 excluded=_union(alternative.excluded for alternative in alternatives),
                 categories=_union(alternative.categories for alternative in alternatives),
                 alternatives=tuple(_distinct(tuple(_distinct(alternative.phrases))
                                              for alternative in matching)))


def corpus_ids(manifest: str | Path = MANIFEST) -> list[str]:
    """The ids of every paper the corpus manifest names (the union of topics[].papers), sorted; a HarnessError
    when the manifest has no topics, a topic no papers list, or the union is empty."""
    path = Path(manifest)
    data = json.loads(path.read_text(encoding="utf-8"))
    topics = data.get("topics") if isinstance(data, dict) else None
    if not isinstance(topics, list):
        raise HarnessError(f"corpus manifest {path}: no topics list")
    ids: set[str] = set()
    for topic in topics:
        papers = topic.get("papers") if isinstance(topic, dict) else None
        if not isinstance(papers, list) or not all(isinstance(arxiv_id, str) for arxiv_id in papers):
            name = topic.get("id", "?") if isinstance(topic, dict) else "?"
            raise HarnessError(f"corpus manifest {path}: topic {name!r} has no papers list")
        ids.update(papers)
    if not ids:
        raise HarnessError(f"corpus manifest {path}: its topics name no papers")
    return sorted(ids)


def load_papers(fetcher: SupportsRecorded, ids: Sequence[str], parse_atom: ParseAtom) -> list[dict]:
    """The papers behind ids, in that order, each from its recorded id_list cassette (nothing is fetched).

    A missing or non-200 cassette, a feed parse_atom cannot parse, a feed with any number of papers but one,
    or a paper whose arxiv_id is not the id asked for, is a HarnessError naming the id; a paper that breaks
    parse_atom's contract raises as parsed_papers does, under the caller name INDEX_CALLER.
    """
    papers = []
    for arxiv_id in ids:
        url = paper_url(arxiv_id)
        page = fetcher.recorded(url)
        if page is None:
            raise HarnessError(f"corpus paper {arxiv_id}: no cassette for {url}")
        if page.status != 200:
            raise HarnessError(f"corpus paper {arxiv_id}: its cassette holds status {page.status}, not 200")
        parsed = parsed_papers(INDEX_CALLER, parse_atom, page.text)
        if parsed is None:
            raise HarnessError(f"corpus paper {arxiv_id}: parse_atom could not parse its recorded feed")
        if len(parsed) != 1:
            raise HarnessError(f"corpus paper {arxiv_id}: its feed holds {len(parsed)} papers, not 1")
        if parsed[0]["arxiv_id"] != arxiv_id:
            raise HarnessError(f"corpus paper {arxiv_id}: its feed holds paper {parsed[0]['arxiv_id']}")
        papers.append(parsed[0])
    return papers


class CorpusIndex:
    """The BM25 index over a corpus of papers (parse_atom's dicts), built once and never changed.

    Only the six schemas.PAPER_KEYS of each paper are kept, copied, so neither the caller's dicts nor the
    results hand out the index's own objects. Papers are held in arxiv_id order whatever order they came in;
    a repeated id is a ValueError.
    """

    def __init__(self, papers: Sequence[dict]) -> None:
        kept = sorted((_six_keys(paper) for paper in papers), key=lambda paper: paper["arxiv_id"])
        ids = [paper["arxiv_id"] for paper in kept]
        repeated = sorted({arxiv_id for arxiv_id in ids if ids.count(arxiv_id) > 1})
        if repeated:
            raise ValueError(f"paper ids repeat: {repeated}")
        self._papers = kept
        self._by_id = {paper["arxiv_id"]: paper for paper in kept}
        self._sequences = [{field: tuple(tokenize(paper[field])) for field in WEIGHTS} for paper in kept]
        self._phrase_sequences = [{field: phrase_tokens(paper[field]) for field in WEIGHTS} for paper in kept]
        self._tf = [{field: Counter(seq[field]) for field in WEIGHTS} for seq in self._sequences]
        self._vocab = [set(seq["title"]) | set(seq["abstract"]) for seq in self._sequences]
        self._df: Counter[str] = Counter(term for vocab in self._vocab for term in vocab)
        self._postings: dict[str, list[int]] = {}
        for doc, vocab in enumerate(self._vocab):
            for term in vocab:
                self._postings.setdefault(term, []).append(doc)
        self._avg = {field: (sum(len(seq[field]) for seq in self._sequences) / len(kept) if kept else 0.0)
                     for field in WEIGHTS}

    def __len__(self) -> int:
        return len(self._papers)

    def ids(self) -> list[str]:
        """Every paper's arxiv_id, sorted."""
        return [paper["arxiv_id"] for paper in self._papers]

    def paper(self, arxiv_id: str) -> dict | None:
        """The paper with exactly this arxiv_id (the six keys, a copy), None when the corpus has none: the
        metadata lookup the note tools are served from (notes.note_tools' papers)."""
        found = self._by_id.get(arxiv_id)
        return None if found is None else _six_keys(found)

    def search(self, args: dict) -> list[dict]:
        """Every matching paper (the six keys, copies) in rank order, uncapped: for the check script and for
        debugging. args is search_arxiv's argument dict (query required; the others optional)."""
        query = parse_query(args["query"])
        sort_by = args.get("sort_by")
        if sort_by is None:
            sort_by = schemas.DEFAULTS["search_arxiv"]["sort_by"]
        if sort_by not in SORT_ORDERS:
            raise ValueError(f"sort_by must be one of {list(SORT_ORDERS)}, not {sort_by!r}")
        scores = self._scores(query)
        filtered = [doc for doc in scores if self._passes(doc, args, query)]
        if sort_by == "relevance":
            filtered.sort(key=lambda doc: (-scores[doc], self._papers[doc]["arxiv_id"]))
        else:
            filtered.sort(key=lambda doc: (self._papers[doc]["published"], self._papers[doc]["arxiv_id"]),
                          reverse=True)
        return [_six_keys(self._papers[doc]) for doc in filtered]

    def tool(self) -> ToolFn:
        """search_arxiv over this index, answering as adapters.arxiv_tools' search_arxiv answers: defaults
        filled in, max_results capped, hits as adapters.hit, EMPTY_RESULTS when nothing matches."""

        def search_arxiv(args: dict) -> ToolOutcome:
            args = {**schemas.DEFAULTS["search_arxiv"], **args}
            shown = capped(args, "max_results", schemas.MAX_SEARCH_RESULTS)
            ranked = self.search(args)
            if not ranked:
                return ToolOutcome([], schemas.EMPTY_RESULTS)
            return ToolOutcome([hit(paper) for paper in ranked[:shown]])

        return search_arxiv

    def _idf(self, term: str) -> float:
        df = self._df[term]
        return math.log((len(self._papers) - df + 0.5) / (df + 0.5) + 1.0)

    def _scores(self, query: Query) -> dict[int, float]:
        """Candidate doc -> BM25 score plus the bonus of each phrase the doc holds; excluded papers and papers
        holding no alternative's phrases in full left out."""
        scores: dict[int, float] = {}
        for term in query.terms:
            idf = self._idf(term)
            for doc in self._postings.get(term, ()):
                scores[doc] = scores.get(doc, 0.0) + idf * self._term_weight(doc, term)
        if query.excluded:
            banned = set(query.excluded)
            scores = {doc: score for doc, score in scores.items() if not (self._vocab[doc] & banned)}
        if query.phrases:
            bonus = {phrase: PHRASE_BONUS * sum(self._idf(term) for term in phrase)
                     for phrase in query.phrases}
            alternatives = [set(alternative) for alternative in query.alternatives]
            passing: dict[int, float] = {}
            for doc, score in scores.items():
                held = [phrase for phrase in query.phrases if self._contains(doc, phrase)]
                if any(alternative <= set(held) for alternative in alternatives):
                    # summed in query order: a set's order, so the float sum, would vary with the hash seed
                    passing[doc] = score + sum(bonus[phrase] for phrase in held)
            scores = passing
        return scores

    def _contains(self, doc: int, phrase: tuple[str, ...]) -> bool:
        """Whether phrase occurs contiguously in the parts-only token sequence of doc's title or abstract."""
        return any(_occurs(phrase, self._phrase_sequences[doc][field]) for field in WEIGHTS)

    def _term_weight(self, doc: int, term: str) -> float:
        """The BM25 term-frequency factor of term in doc, summed over the fields with their weights."""
        total = 0.0
        for field, weight in WEIGHTS.items():
            tf = self._tf[doc][field][term]
            if tf:
                length = len(self._sequences[doc][field])
                total += weight * tf * (K1 + 1) / (tf + K1 * (1 - B + B * length / self._avg[field]))
        return total

    def _passes(self, doc: int, args: dict, query: Query) -> bool:
        paper = self._papers[doc]
        wanted = _category_filter(args.get("categories"), query.categories)
        if wanted is not None and not (wanted & set(paper["categories"])):
            return False
        if args.get("date_from") and paper["published"] < args["date_from"]:
            return False
        return not (args.get("date_to") and paper["published"] > args["date_to"])


def _category_filter(listed: object, queried: tuple[str, ...]) -> set[str] | None:
    """The categories a paper must share one of, or None when neither the args nor the query filter."""
    from_args = set(listed) if isinstance(listed, list) and listed else None
    from_query = set(queried) if queried else None
    if from_args is None:
        return from_query
    return from_args if from_query is None else from_args & from_query


def _occurs(phrase: tuple[str, ...], sequence: tuple[str, ...]) -> bool:
    """Whether phrase is a contiguous run of sequence."""
    n = len(phrase)
    return any(sequence[i:i + n] == phrase for i in range(len(sequence) - n + 1) if sequence[i] == phrase[0])


def _six_keys(paper: dict) -> dict:
    """The paper's schemas.PAPER_KEYS, lists copied."""
    return {key: list(paper[key]) if key in schemas.PAPER_LIST_KEYS else paper[key]
            for key in schemas.PAPER_KEYS}


def _distinct[T: Hashable](items: Iterable[T]) -> list[T]:
    """items without repeats, first occurrence first."""
    return list(dict.fromkeys(items))


def _union[T: Hashable](groups: Iterable[Iterable[T]]) -> tuple[T, ...]:
    """The items of every group, flattened, without repeats, first occurrence first."""
    return tuple(_distinct(item for group in groups for item in group))

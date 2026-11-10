"""Synthetic trajectories (docs/plan.md, 微调数据 第 4 条; docs/modules/接口.md, 数据生成相关的契约): the
teacher model runs the agent loop over the recorded corpus, one request record at a time, and every run
leaves an index row, a transcript and a manifest that a later run resumes from.

A request record (JSONL, one object per line; load_requests checks it) carries id, intent, scenario, lang,
split, family, text and slots. The owner's template module writes them, and the split is fixed before
anything is generated. The scenario says what the generator does to the request, not what the model does:
normal, no_call, clarify and parallel requests run as they are; a recover request meets exactly one fault.

The request's text is first rewritten by the cloud model (Rewriter: different words, same meaning and
language, every arXiv id and date kept). A rewrite that drops or adds an id or a date, comes back empty or
equals the original apart from whitespace is replaced by the original (fallback); the call is paid either way.
Then the loop runs the text used with a fresh ToolExecutor over tools built for this request (make_tools):
search_arxiv answers from the CorpusIndex through a SearchFaults, get_paper and fetch_fulltext through the
replay Fetcher behind a FaultInjector, both injectors limited to the transport faults (FETCH_FAULTS) and
sharing one FaultBudget(1) at rate 1.0 for a recover request and 0.0 otherwise. So a recover trajectory meets
one transport fault (timeout, rate_limited or server_error, never empty_results; which of them follows from
the seed, the request id and the request's key) on its first tool request and no other trajectory meets any.
The end-to-end evaluation (research_agent.eval.e2e) passes a FaultSetting through run_request instead: all
seven faults at its rate, no budget, whatever the scenario, with a FaultTrace of the arXiv tool calls for its
records and timing sidecar. The note tools are the real ones by default (notes="store"): each request gets its
own notes.NoteStore in <run>/notes/<id>/ (store_note_tools), emptied before its turn, with the paper metadata
looked up in the CorpusIndex (CorpusIndex.paper) and today the run's today. Seed notes (on by default; the
script's --no-seed-notes turns them off) then fill the emptied store before the turn, so search_notes and
list_notes have something to find: SEED_NOTES scaffold notes rendered from the corpus metadata (seed_plan,
seed_notes), SEED_OWN_TOPIC on papers of the request's topic and the rest on papers of other topics, all of
topics the request's split may use (the requests manifest next to the requests file, <stem>-manifest.json; the
whole corpus when there is none), drawn from the request id alone, dated before today and tagged with the
paper's categories and the topic id. Without seed notes the store is created only when a note tool is first
called. A store per request keeps every trajectory independent of the order, the filters and the resumes of
its run, so two runs over the same answers write the same note ids, paths and files. write_note answers
{"note_id", "arxiv_id", "path"} (path relative to the request's store), the keys of the stand-ins, and
NOT_FOUND for a paper outside the corpus. notes="stub" (the script's --stub-notes, and the tests) selects the
stand-ins (default_note_tools) instead, which write nothing: search_notes and list_notes answer an empty list
without an error code, so they never consume the turn's single relax, and write_note answers a note record for
a call that names a paper and a mode. Either way an empty search_notes answer carries no error code.
build_search_url is never needed, since the search never reaches a fetcher, and the stub given to the adapters
raises a HarnessError if it is ever called.

The run folder holds transcripts/<id>.json (save_transcript), notes/<id>/ (the store of a request that was
seeded or whose turn called a note tool; notes="store" only), index.jsonl (one canonical JSON line per
request, sorted by id, rewritten whole after every request, so two runs over the same answers are
byte-identical) and manifest.json (sorted keys, also rewritten after every request, so a crash leaves a
usable record). A run folder belongs to one setting: when its manifest exists, generate() refuses with a
HarnessError, before it writes anything, a call whose RUN_SETTINGS (seed, today, model and its temperature,
placeholders, the rewrite switch and temperature, the rewrite prompt hash, the schema and prompt hashes, the
requests file's hash, the notes setting, the seed-notes switch and its scope) differ from the recorded ones,
naming every one that differs; a different setting is a different run (another --run name), and redo does not
override that. The filters (limit, scenario, split) only
narrow a run and may change. Each resumed index row is checked to carry every INDEX_KEYS key (placeholders a
bool, status a string, transcript a string or null) before anything is written. A request whose row is already
in the index with the same placeholders flag is skipped unless redo is asked, except a budget_stop row: that
turn was cut short by the spending cap and is run again. A BudgetExceeded inside the loop ends the turn as
budget_stop and the run stops after writing that row; one from the rewriter stops the run before the request,
which leaves nothing behind. Any other exception out of run_request (an API error that outlasted its retries,
a bug in an owner function) leaves a harness_error row carrying the exception text in error and no transcript
(transcript null; a transcript and notes an earlier run of that request left, as a redo finds them, are
deleted, so the folder never holds a transcript its index does not name), and the run goes on with the next
request.

The owner's functions arrive as an OwnerFunctions record (check_call, decide, parse_atom, extract_section);
probe() calls each once so a NotImplementedError surfaces as OwnerFunctionMissing before anything is paid.
The owner's check_trajectory (research_agent.trajectories) is optional: load_validator() gives None when
the module is absent or raises NotImplementedError, every verdict is then None and the manifest records
validated: false; such a run is for debugging only (接口.md). The manifest also records the realized mix,
what each trajectory turned out to be: parallel when a step has two or more calls, clarify when the turn
ended in a clarification, no_call when no tool was called, recover when a tool result carried
EMPTY_RESULTS, RATE_LIMITED or UPSTREAM_ERROR, else normal, checked in that order over the trajectories
that did not end in a harness error; next to the plan's mix, so a drift between what was asked and what
the teacher did is visible per run.
"""

from __future__ import annotations

import copy
import datetime
import hashlib
import importlib
import json
import os
import random
import re
import shutil
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..agent.context import TokenCounter
from ..agent.executor import ToolExecutor
from ..agent.loop import Model, load_transcript, run_turn, save_transcript
from ..agent.prompts import PROMPT_HASH, PROMPT_VERSION, system_prompt
from ..agent.types import (
    ANSWERED,
    BUDGET_STOP,
    CLARIFICATION,
    HARNESS_ERROR,
    CheckCall,
    Decide,
    HarnessError,
    ToolFn,
    ToolOutcome,
    Transcript,
)
from ..canonical import canonical_json, split_lines
from ..llm.budget import Budget, BudgetExceeded, cost_rmb
from ..net.faults import FAULTS, SEARCH_FAULTS, FaultBudget, FaultInjector, SearchFaults
from ..notes import EVIDENCE_SENTENCES, NoteStore, pick_evidence
from ..notes import note_tools as notes_over_store
from ..tools import schemas
from ..tools.adapters import ExtractSection, ParseAtom, SupportsGet, arxiv_tools
from ..tools.corpus_search import MANIFEST, CorpusIndex
from ..train.manifest import REPO, git_commit, git_dirty, sha256_of

SCENARIOS = ("normal", "no_call", "clarify", "recover", "parallel")
LANGS = ("zh", "en")
SPLITS = ("train", "dev", "test")
PLANNED_MIX = {"normal": 0.55, "no_call": 0.15, "clarify": 0.10, "recover": 0.15, "parallel": 0.05}
REQUEST_KEYS = ("id", "intent", "scenario", "lang", "split", "family", "text", "slots")
FETCH_FAULTS = ("timeout", "rate_limited", "server_error")  # what a recover request may meet on a fetch
RECOVER_CODES = (schemas.EMPTY_RESULTS, schemas.RATE_LIMITED, schemas.UPSTREAM_ERROR)
INDEX_FILE = "index.jsonl"
MANIFEST_FILE = "manifest.json"
TRANSCRIPTS_DIR = "transcripts"
INDEX_KEYS = ("id", "intent", "scenario", "lang", "split", "family", "text", "rewritten", "rewrite_fallback",
              "rewrite_cost_rmb", "status", "error", "steps", "paid_calls", "cache_hits", "cost_rmb",
              "faults", "verdict", "placeholders", "transcript", "slots", "extra")  # the keys of an index row
# what makes a run, fixed per run folder (the module docstring); the second set is the part under config
RUN_SETTINGS = ("seed", "today", "model", "model_temperature", "placeholders", "rewrite", "temperature",
                "rewrite_prompt_hash", "schema_hash", "prompt_hash", "requests_sha256", "notes", "seed_notes",
                "seed_scope", "rewrite_model", "thinking", "rewrite_thinking")
_CONFIG_SETTINGS = {"seed", "model", "model_temperature", "placeholders", "rewrite", "temperature",
                    "requests_sha256", "notes", "seed_notes", "seed_scope", "rewrite_model", "thinking",
                    "rewrite_thinking"}
NOTE_SETTINGS = ("store", "stub")  # the real note store per request, or the stand-ins (default_note_tools)
NOTES_DIR = "notes"
NOTE_TOOLS = ("search_notes", "list_notes", "write_note")
SEED_NOTES = 5  # scaffold notes in a request's store before its turn (seed_plan)
SEED_OWN_TOPIC = 2  # of them on papers of the request's topic; the rest on papers of other topics
SEED_MAX_AGE_DAYS = 60  # a seeded note is dated 1 to this many days before today, never before its paper
REQUESTS_MANIFEST_SUFFIX = "-manifest.json"  # <stem>-manifest.json next to a requests file (make_requests.py)
VALIDATOR_MODULE = "research_agent.trajectories"
PROBE_TODAY = "2026-10-10"  # the date of the transcript load_validator probes the owner's validator with

REWRITE_PROMPT_VERSION = "synth-2026-10-12"
PROMPT = {
    "zh": "把用户的请求换一种说法改写:意思不变,仍然用中文。请求里出现的 arXiv 编号、日期、数字和"
          "专业术语必须原样保留,一个都不能改。只输出改写后的请求,不要解释,不要加引号。",
    "en": "Rewrite the user's request in different words: the same meaning, still in English. Keep every "
          "arXiv id, date, number and technical term exactly as written. Output only the rewritten request, "
          "with no explanation and no quotation marks.",
}
REWRITE_PROMPT_HASH = hashlib.sha256((REWRITE_PROMPT_VERSION + PROMPT["zh"] + PROMPT["en"])
                                     .encode("utf-8")).hexdigest()[:16]

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")  # an id names the transcript file
_LITERAL = re.compile(r"\d{4}\.\d{4,5}|\d{4}-\d{2}-\d{2}")  # an arXiv id or an ISO date
_PROBE_REQUEST = {"id": "probe", "intent": "probe", "scenario": "no_call", "lang": "en", "split": "dev",
                  "family": "probe", "text": "probe", "slots": {}}


# ---- request records ------------------------------------------------------------------------------------


def load_requests(path: str | Path) -> list[dict]:
    """The request records of a JSONL file, each with slots filled in ({} when absent), in file order.

    A ValueError names the line of the first record that is not a JSON object, lacks a non-empty string id,
    intent or text, has a family that is neither a non-empty string nor an integer (a bool is neither), an id
    that is not file-name safe (letters, digits, '.', '_', '-'), a scenario, lang or split outside SCENARIOS,
    LANGS or SPLITS, slots that are not an object, or an id already used.
    """
    path = Path(path)
    requests: list[dict] = []
    lines: dict[str, int] = {}
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            where = f"{path.name}: line {number}"
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where} is not JSON: {exc}") from exc
            if not _is_object(value):
                raise ValueError(f"{where} is not a JSON object")
            request = _checked_request(value, where)
            if request["id"] in lines:
                raise ValueError(f"{where}: id {request['id']!r} already used on line {lines[request['id']]}")
            lines[request["id"]] = number
            requests.append(request)
    return requests


def check_requests(values: Iterable[object], source: str) -> list[dict]:
    """load_requests' checks on records already read, such as the end-to-end test tasks a LockedSet.open
    returned, which stay in memory (scripts/run_test.py): copies of the records with slots filled in ({} when
    absent), in the order given. A ValueError names the first bad record by its position in what was given
    (record 1 for the first), after source."""
    requests: list[dict] = []
    positions: dict[str, int] = {}
    for number, value in enumerate(values, 1):
        where = f"{source}: record {number}"
        if not _is_object(value):
            raise ValueError(f"{where} is not a JSON object")
        request = _checked_request(value, where)
        first = positions.setdefault(request["id"], number)
        if first != number:
            raise ValueError(f"{where}: id {request['id']!r} already used by record {first}")
        requests.append(request)
    return requests


def _is_object(value: object) -> bool:
    return isinstance(value, dict)


def _is_text(value: object) -> bool:
    """A string with something in it."""
    return isinstance(value, str) and bool(value.strip())


def _checked_request(value: dict, where: str) -> dict:
    for key in ("id", "intent", "text"):
        if not _is_text(value.get(key)):
            raise ValueError(f"{where}: {key} must be a non-empty string")
    family = value.get("family")
    if not (_is_text(family) or (isinstance(family, int) and not isinstance(family, bool))):
        raise ValueError(f"{where}: family must be a non-empty string or an integer, not {family!r:.80}")
    if not _ID.fullmatch(value["id"]):
        raise ValueError(f"{where}: id {value['id']!r} must be letters, digits, '.', '_' or '-' only (it "
                         "names the transcript file)")
    for key, allowed in (("scenario", SCENARIOS), ("lang", LANGS), ("split", SPLITS)):
        if value.get(key) not in allowed:
            raise ValueError(f"{where}: {key} must be one of {list(allowed)}, not {value.get(key)!r}")
    slots = value.get("slots", {})
    if not _is_object(slots):
        raise ValueError(f"{where}: slots must be an object, not {slots!r:.80}")
    return {**copy.deepcopy(value), "slots": copy.deepcopy(slots)}


def select_requests(requests: list[dict], *, scenario: str | None = None, split: str | None = None,
                    limit: int | None = None) -> list[dict]:
    """The requests of the given scenario and split (None: any), the first limit of them (None: all)."""
    chosen = [r for r in requests if (scenario is None or r["scenario"] == scenario)
              and (split is None or r["split"] == split)]
    return chosen if limit is None else chosen[:limit]


# ---- rewriting ------------------------------------------------------------------------------------------


class RewriteClient(Protocol):
    """research_agent.llm.cloud.CloudClient, or a stub with the same counters, price list and chat()."""

    model: str
    calls: int
    cache_hits: int
    budget: Budget

    def chat(self, messages: list[dict], tools: list[dict] | None = None, **params) -> dict: ...


@dataclass(frozen=True)
class Rewrite:
    """The text to run (the original when fallback is True), what the call cost and whether it was cached."""

    text: str
    cost_rmb: float
    cached: bool
    fallback: bool


def literals(text: str) -> list[str]:
    """The arXiv ids and ISO dates of text in order: what a rewrite must keep."""
    return _LITERAL.findall(text)


def preserved(original: str, rewritten: str) -> bool:
    """Whether rewritten can stand in for original: not empty, different beyond whitespace, and carrying
    exactly the arXiv ids and dates of the original, none dropped and none added."""
    if not rewritten.strip() or " ".join(rewritten.split()) == " ".join(original.split()):
        return False
    return set(literals(rewritten)) == set(literals(original))


class Rewriter:
    """Rewrites a request with the cloud model through the client directly (never through CloudModel, which
    always sends the tool schemas), at a temperature above zero so that two rewrites of one text differ."""

    def __init__(self, client: RewriteClient, *, temperature: float = 0.7) -> None:
        self.client, self.temperature = client, temperature

    def rewrite(self, text: str, lang: str) -> Rewrite:
        """The rewrite of text in lang; priced like CloudModel.cost when the call was paid, 0.0 when it
        came from the cache. BudgetExceeded propagates to the caller."""
        if lang not in PROMPT:
            raise ValueError(f"lang must be one of {list(PROMPT)}, not {lang!r}")
        messages = [{"role": "system", "content": PROMPT[lang]}, {"role": "user", "content": text}]
        paid_before, hits_before = self.client.calls, self.client.cache_hits
        data = self.client.chat(messages, temperature=self.temperature)
        paid, cached = self.client.calls > paid_before, self.client.cache_hits > hits_before
        cost = 0.0
        if paid:
            cost = cost_rmb(self.client.budget.prices[self.client.model], dict(data.get("usage") or {}))
        rewritten = (data["choices"][0]["message"].get("content") or "").strip()
        if not preserved(text, rewritten):
            return Rewrite(text, cost, cached, True)
        return Rewrite(rewritten, cost, cached, False)


# ---- the owner's functions ------------------------------------------------------------------------------


@dataclass(frozen=True)
class OwnerFunctions:
    """The owner's four functions the generator needs (docs/modules/接口.md), real or stand-ins."""

    check_call: CheckCall
    decide: Decide
    parse_atom: ParseAtom
    extract_section: ExtractSection


class OwnerFunctionMissing(RuntimeError):
    """An owner function that still raises NotImplementedError; name says which."""

    def __init__(self, name: str) -> None:
        super().__init__(f"the owner's {name} is not implemented yet (it raises NotImplementedError)")
        self.name = name


def probe(owner: OwnerFunctions, today: str) -> None:
    """Call each owner function once with a trivial input, before anything is paid.

    A NotImplementedError is an OwnerFunctionMissing naming the function; an answer of the wrong kind (not a
    list from check_call, an action outside ALLOWED_ACTIONS from decide, not a list or None from parse_atom,
    not a string or None from extract_section) is a HarnessError. Any other exception propagates as it is.
    """
    probes = [
        ("check_call", lambda: owner.check_call("search_arxiv", {"query": "x"}, [], today),
         lambda v: isinstance(v, list)),
        ("decide", lambda: owner.decide("empty_results", 1),
         lambda v: v in schemas.ALLOWED_ACTIONS["empty_results"]),
        ("parse_atom", lambda: owner.parse_atom("<feed/>"), lambda v: v is None or isinstance(v, list)),
        ("extract_section", lambda: owner.extract_section("<html></html>", "introduction", 100),
         lambda v: v is None or isinstance(v, str)),
    ]
    for name, call, fine in probes:
        try:
            answer = call()
        except NotImplementedError as exc:
            raise OwnerFunctionMissing(name) from exc
        if not fine(answer):
            raise HarnessError(f"the owner's {name} answered the probe with {answer!r:.120}, which is not "
                               "what its contract allows")


def no_search_url(args: dict, today: str) -> str:
    """The build_search_url given to the adapters: search_arxiv never reaches a fetcher here."""
    raise HarnessError("search_arxiv is served by the corpus index; build_search_url must not be called")


# ---- tools per request ----------------------------------------------------------------------------------


def store_note_tools(root: Path, index: CorpusIndex, today: str) -> dict[str, ToolFn]:
    """search_notes, list_notes and write_note over a notes.NoteStore in root (notes.note_tools, the papers
    looked up with index.paper, dated today), the store created at the first call of any of them, so a turn
    that calls no note tool leaves no folder behind."""
    built: dict[str, dict[str, ToolFn]] = {}

    def tools() -> dict[str, ToolFn]:
        if "tools" not in built:
            built["tools"] = notes_over_store(NoteStore(root), index.paper, today)
        return built["tools"]

    def lazy(name: str) -> ToolFn:
        return lambda args: tools()[name](args)

    return {name: lazy(name) for name in NOTE_TOOLS}


def default_note_tools() -> dict[str, ToolFn]:
    """The explicit stand-ins for the note tools, which write nothing: what notes="stub" (the script's
    --stub-notes) and tests use. search_notes and list_notes give empty answers without an error code (so they
    never use up the turn's single relax), write_note a note record ({"note_id", "arxiv_id", "path"}, the
    keys of the real one) for a call that names a paper and a mode."""

    def search_notes(args: dict) -> ToolOutcome:
        return ToolOutcome([])

    def list_notes(args: dict) -> ToolOutcome:
        return ToolOutcome([])

    def write_note(args: dict) -> ToolOutcome:
        args = {**schemas.DEFAULTS["write_note"], **args}
        arxiv_id, mode = args.get("arxiv_id"), args.get("mode")
        if not (isinstance(arxiv_id, str) and arxiv_id and isinstance(mode, str) and mode):
            return ToolOutcome({}, schemas.INVALID_ARGUMENTS)
        return ToolOutcome({"note_id": f"{arxiv_id}-note", "arxiv_id": arxiv_id,
                            "path": f"notes/{arxiv_id}.md"})

    return {"search_notes": search_notes, "list_notes": list_notes, "write_note": write_note}


# ---- seed notes -----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SeedPool:
    """What seed notes are drawn from: topic id -> its paper ids (corpus manifest order), the words that name
    a topic in a request's slots (casefolded id, Chinese name and phrases -> topic id), and split -> the topic
    ids that split may use, or None when there is no split restriction (every topic for every split)."""

    topics: dict[str, tuple[str, ...]]
    names: dict[str, str]
    splits: dict[str, tuple[str, ...]] | None

    @property
    def scope(self) -> str:
        """"split" when each split draws from its own topics, "corpus" when every split draws from all."""
        return "corpus" if self.splits is None else "split"

    def allowed(self, split: str) -> tuple[str, ...]:
        """The topic ids a request of split may draw from, sorted."""
        return tuple(sorted(self.topics if self.splits is None else self.splits.get(split, ())))


def requests_manifest_of(requests: str | Path) -> Path | None:
    """The manifest scripts/make_requests.py writes next to a requests file (<stem>-manifest.json), None
    when there is none (the hand-written dry-run file)."""
    path = Path(requests)
    manifest = path.with_name(path.stem + REQUESTS_MANIFEST_SUFFIX)
    return manifest if manifest.is_file() else None


def load_seed_pool(corpus_manifest: str | Path, requests_manifest: str | Path | None = None,
                   known: Callable[[str], object] | None = None) -> SeedPool:
    """The SeedPool of a corpus manifest (topics with their papers) restricted per split by a requests
    manifest's topics (split -> topic ids), or unrestricted without one. known(arxiv_id) (CorpusIndex.paper)
    drops a paper it answers None for, so a note is only drawn for a paper whose metadata is there. A
    HarnessError, before anything is written, when the requests manifest is not an object with a topics
    list for every split, names a topic the corpus manifest lacks, or records a corpus_manifest_sha256 other
    than the corpus manifest's (it was made against another corpus)."""
    corpus = _read_manifest(Path(corpus_manifest))
    topics: dict[str, tuple[str, ...]] = {}
    names: dict[str, str] = {}
    for entry in corpus.get("topics") or []:
        tid = entry["id"]
        topics[tid] = tuple(a for a in entry.get("papers", []) if known is None or known(a) is not None)
        for word in (tid, entry.get("name"), *entry.get("phrases", [])):
            if isinstance(word, str) and word.strip():
                names.setdefault(" ".join(word.split()).casefold(), tid)
    if requests_manifest is None:
        return SeedPool(topics, names, None)
    path = Path(requests_manifest)
    data = _read_manifest(path)
    recorded = data.get("corpus_manifest_sha256")
    if recorded is not None and recorded != sha256_of(corpus_manifest):
        raise HarnessError(f"{path} was made against another corpus manifest (sha256 {recorded}), not "
                           f"{corpus_manifest}")
    by_split = data.get("topics")
    splits: dict[str, tuple[str, ...]] = {}
    for split in SPLITS:
        listed = by_split.get(split) if _is_object(by_split) else None
        if not isinstance(listed, list) or not all(isinstance(t, str) for t in listed):
            raise HarnessError(f"{path}: topics.{split} is not a list of topic ids")
        unknown = [t for t in listed if t not in topics]
        if unknown:
            raise HarnessError(f"{path}: topics.{split} names unknown topic {unknown[0]!r}")
        splits[split] = tuple(listed)
    return SeedPool(topics, names, splits)


def request_topic(request: dict, pool: SeedPool) -> str | None:
    """The topic id a request names: its topic_id slot, else its topic slot (the first string of a list) as
    a topic's id, Chinese name or phrase, compared casefolded with whitespace collapsed; None otherwise."""
    slots = request.get("slots") or {}
    if slots.get("topic_id") in pool.topics:
        return slots["topic_id"]
    topic = slots.get("topic")
    if isinstance(topic, list):
        topic = next((t for t in topic if isinstance(t, str)), None)
    return pool.names.get(" ".join(topic.split()).casefold()) if isinstance(topic, str) else None


def seed_plan(request: dict, pool: SeedPool, index: CorpusIndex, today: str) -> list[tuple[str, str, str]]:
    """(arxiv_id, topic id, created date) of the seed notes of request, drawn with a random.Random seeded
    from the request id alone, so a rerun, a redo and a resume draw the same notes.

    From the topics the request's split may use (SeedPool.allowed): SEED_OWN_TOPIC papers of the request's
    topic (fewer when it has fewer there, none when the topic is not among them), then SEED_NOTES -
    SEED_OWN_TOPIC papers of the other topics (SEED_NOTES when the request names no topic), one per topic in
    a random topic order and then round after round while papers are left; so a request has SEED_NOTES
    notes, or fewer when its split lacks the papers. Each is dated a random day from SEED_MAX_AGE_DAYS days
    before today to the day before, not before the paper's published date (the day before today when the
    paper is newer)."""
    rng = random.Random(f"seed-notes:{request['id']}")
    allowed = pool.allowed(request["split"])
    topic = request_topic(request, pool)
    picked: list[tuple[str, str]] = []
    if topic in allowed:
        own = pool.topics[topic]
        picked += [(a, topic) for a in rng.sample(own, min(SEED_OWN_TOPIC, len(own)))]
    others = [t for t in allowed if t != topic]
    rng.shuffle(others)
    queues = {t: rng.sample(pool.topics[t], len(pool.topics[t])) for t in others}
    wanted = len(picked) + SEED_NOTES - (SEED_OWN_TOPIC if topic is not None else 0)
    while len(picked) < wanted and any(queues.values()):
        for t in others:
            if queues[t] and len(picked) < wanted:
                picked.append((queues[t].pop(), t))
    day = datetime.date.fromisoformat(today)
    plan = []
    for arxiv_id, tid in picked:
        latest = day - datetime.timedelta(days=1)
        earliest = day - datetime.timedelta(days=SEED_MAX_AGE_DAYS)
        published = _date_or_none((index.paper(arxiv_id) or {}).get("published"))
        if published is not None:
            earliest = min(max(earliest, published), latest)
        created = earliest + datetime.timedelta(days=rng.randint(0, (latest - earliest).days))
        plan.append((arxiv_id, tid, created.isoformat()))
    return plan


def _date_or_none(value: object) -> datetime.date | None:
    """The date at the start of an ISO date or timestamp string, None for anything else."""
    try:
        return datetime.date.fromisoformat(value[:10]) if isinstance(value, str) else None
    except ValueError:
        return None


def seed_notes(root: Path, plan: list[tuple[str, str, str]], index: CorpusIndex) -> list[dict]:
    """Write the plan's notes into a NoteStore in root, in plan order: each a scaffold note rendered from the
    corpus metadata (title, abstract, published; index.paper), its evidence the abstract's first
    EVIDENCE_SENTENCES sentences, dated its created date, tagged with the paper's categories and the topic
    id. The write records (NoteStore.write). A paper the index lacks is a HarnessError."""
    store = NoteStore(root)
    written = []
    for arxiv_id, topic, created in plan:
        paper = index.paper(arxiv_id)
        if paper is None:
            raise HarnessError(f"seed note: paper {arxiv_id} is not in the corpus index")
        evidence = pick_evidence(paper.get("abstract", ""), EVIDENCE_SENTENCES, [])
        tags = [*(t for t in paper.get("categories", []) if isinstance(t, str)), topic]
        written.append(store.write(paper, {"mode": "scaffold"}, evidence, created, tags))
    return written


@dataclass(frozen=True)
class FaultSetting:
    """The fault injection of an evaluation run (docs/plan.md, 故障注入; scripts/run_e2e.py --fault-rate and
    --fault-persistent), in place of the scenario's: every request gets its own FaultInjector over the
    fetcher and SearchFaults over the search, at this rate, with these faults (the search keeps those of
    them it can inject, SEARCH_FAULTS), persistent or not, under no FaultBudget, and the delay fault waits
    with sleep, for real by default (net.faults.DELAY_S)."""

    rate: float
    persistent: bool = False
    faults: tuple[str, ...] = FAULTS
    sleep: Callable[[float], None] = time.sleep


@dataclass
class FaultTrace:
    """Every call of one request to the three arXiv tools, in call order, while they run behind injectors:
    {"tool", "wall_ms", "faults"}, faults being the decisions of the call's injector that met a fault during
    the call, each {"key": the canonical URL or the search key, "kind": the fault, "ordinal": the request's
    ordinal for that key}. A call whose tool raised is kept with what it met until then. wall_ms is the
    call's time on clock, a delay fault's sleep included: research_agent.eval.e2e hands it to the timing
    sidecar, never to a record."""

    clock: Callable[[], float] = time.perf_counter
    calls: list[dict] = field(default_factory=list)

    def wrap(self, name: str, tool: ToolFn, injector: FaultInjector | SearchFaults) -> ToolFn:
        """tool, with each call traced under name from the decisions injector logs while it runs."""

        def traced(args: dict) -> ToolOutcome:
            seen, started = len(injector.log), self.clock()
            try:
                return tool(args)
            finally:
                met = [{"key": entry.get("url", entry.get("key")), "kind": entry["fault"],
                        "ordinal": entry["ordinal"]} for entry in injector.log[seen:] if entry["fault"]]
                self.calls.append({"tool": name, "wall_ms": round((self.clock() - started) * 1000, 1),
                                   "faults": met})

        return traced


def make_tools(*, fetcher: SupportsGet, index: CorpusIndex, owner: OwnerFunctions, today: str,
               scenario: str, seed: int, task: str, note_tools: dict[str, ToolFn] | None = None,
               faults: FaultSetting | None = None,
               fault_trace: FaultTrace | None = None) -> tuple[dict[str, ToolFn], dict]:
    """The tools of one request and its two injectors, {"fetch": FaultInjector, "search": SearchFaults}.

    The fetcher and the index are shared by every request; the injectors are new per request, keyed by task
    (the request id) and seed. Without a fault setting they are limited to the transport faults FETCH_FAULTS
    (so a recover request never meets an empty search) and share one FaultBudget(1): at rate 1.0 (a recover
    request) the first tool request meets a fault and every later one is answered as recorded; at rate 0.0
    nothing is injected. A FaultSetting replaces that, whatever the scenario (FaultSetting). With a
    fault_trace, search_arxiv, get_paper and fetch_fulltext are traced into it (FaultTrace.wrap).
    """
    if faults is None:
        rate = 1.0 if scenario == "recover" else 0.0
        budget = FaultBudget(1)
        injected = FaultInjector(fetcher, rate=rate, seed=seed, task=task, faults=FETCH_FAULTS, budget=budget,
                                 sleep=lambda seconds: None)
        search = SearchFaults(index.tool(), rate=rate, seed=seed, task=task, faults=FETCH_FAULTS,
                              budget=budget)
    else:
        injected = FaultInjector(fetcher, rate=faults.rate, seed=seed, task=task, faults=faults.faults,
                                 persistent=faults.persistent, sleep=faults.sleep)
        search = SearchFaults(index.tool(), rate=faults.rate, seed=seed, task=task,
                              faults=[f for f in faults.faults if f in SEARCH_FAULTS],
                              persistent=faults.persistent)
    arxiv = {**arxiv_tools(injected, build_search_url=no_search_url, parse_atom=owner.parse_atom,
                           extract_section=owner.extract_section, today=today), "search_arxiv": search}
    if fault_trace is not None:
        arxiv = {name: fault_trace.wrap(name, tool, search if name == "search_arxiv" else injected)
                 for name, tool in arxiv.items()}
    tools = {**arxiv, **(default_note_tools() if note_tools is None else note_tools)}
    return tools, {"fetch": injected, "search": search}


# ---- the validator --------------------------------------------------------------------------------------


CheckTrajectory = Callable[[dict, dict], list[str]]  # the owner's check_trajectory(transcript, request)


class TrajectoryValidator:
    """The owner's check_trajectory as the generator calls it: on the transcript as Transcript.to_json()
    reads back and the request record. available turns False at the first NotImplementedError, after which
    every verdict is None (the run counts as not validated)."""

    def __init__(self, check: CheckTrajectory) -> None:
        self.check, self.available = check, True

    def __call__(self, transcript: Transcript, request: dict) -> dict | None:
        if not self.available:
            return None
        try:
            problems = self.check(json.loads(transcript.to_json()), request)
        except NotImplementedError:
            self.available = False
            return None
        if not isinstance(problems, list) or not all(isinstance(p, str) for p in problems):
            raise HarnessError(f"check_trajectory must return a list of strings, got {problems!r:.200}")
        return {"accepted": not problems, "problems": problems}


def load_validator(module: str = VALIDATOR_MODULE, *, today: str = PROBE_TODAY) -> TrajectoryValidator | None:
    """The owner's validator, or None when its module is absent or it raises NotImplementedError (probed
    once on a one-step transcript, so a stub is found before anything is paid). A module that is there but
    has no check_trajectory is a HarnessError naming the expected function."""
    try:
        owner_module = importlib.import_module(module)
    except ImportError:
        return None
    try:
        check = owner_module.check_trajectory
    except AttributeError:
        raise HarnessError(f"{module} has no check_trajectory(transcript, request): the module is there, but "
                           "the function the generator calls is not") from None
    validator = TrajectoryValidator(check)
    validator(_probe_transcript(today), dict(_PROBE_REQUEST))
    return validator if validator.available else None


def _probe_transcript(today: str) -> Transcript:
    """A well-formed answered turn without tool calls."""
    messages = [{"role": "system", "content": system_prompt(today)}, {"role": "user", "content": "probe"},
                {"role": "assistant", "content": "probe"}]
    attempt = {"model": "probe", "content": "probe", "tool_calls": [], "issues": [], "action": None}
    step = {"index": 1, "model": "probe", "executed": [], "decisions": [], "attempts": [attempt]}
    return Transcript(request="probe", today=today, status=ANSWERED, final_text="probe", messages=messages,
                      steps=[step], schema_hash=schemas.SCHEMA_HASH, prompt_version=PROMPT_VERSION,
                      prompt_hash=PROMPT_HASH)


# ---- one request ----------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Outcome:
    """What one request left: its transcript, its index row and the fault counts per injector."""

    transcript: Transcript
    row: dict
    faults: dict


def run_request(request: dict, *, model: Model, owner: OwnerFunctions, fetcher: SupportsGet,
                index: CorpusIndex, today: str, count_tokens: TokenCounter | None = None, seed: int = 42,
                rewriter: Rewriter | None = None, note_tools: dict[str, ToolFn] | None = None,
                notes_dir: Path | None = None, validator: TrajectoryValidator | None = None,
                placeholders: bool = False, seed_pool: SeedPool | None = None,
                faults: FaultSetting | None = None, fault_trace: FaultTrace | None = None) -> Outcome:
    """Rewrite (when there is a rewriter), build the tools, run the turn, ask the validator (when there is
    one). BudgetExceeded from the rewriter propagates; inside the loop it ends the turn as budget_stop. A
    model without a cost method (a scripted stand-in) is priced at 0.0.

    The note tools: note_tools when given; else, with notes_dir, the real ones over a store in notes_dir
    (store_note_tools), which belongs to this request: whatever an earlier run left there is deleted after
    the rewrite and before the turn, and with a seed_pool the request's seed notes (seed_plan, seed_notes)
    are written into it right after; else the stand-ins (default_note_tools).

    faults and fault_trace go to make_tools: a FaultSetting in place of the scenario's fault injection, and
    a trace the caller keeps of the arXiv tool calls (filled while the turn runs, so it holds what a turn
    that raised met until then)."""
    rewrite = None if rewriter is None else rewriter.rewrite(request["text"], request["lang"])
    text = request["text"] if rewrite is None else rewrite.text
    if note_tools is None and notes_dir is not None:
        _remove_tree(Path(notes_dir))
        if seed_pool is not None:
            seed_notes(Path(notes_dir), seed_plan(request, seed_pool, index, today), index)
        note_tools = store_note_tools(Path(notes_dir), index, today)
    tools, injectors = make_tools(fetcher=fetcher, index=index, owner=owner, today=today,
                                  scenario=request["scenario"], seed=seed, task=request["id"],
                                  note_tools=note_tools, faults=faults, fault_trace=fault_trace)
    executor = ToolExecutor(tools, owner.check_call, today=today)
    transcript = run_turn(text, model=model, executor=executor, decide=owner.decide, today=today,
                          count_tokens=count_tokens, cost_of=getattr(model, "cost", None))
    faults = {name: injector.counts() for name, injector in injectors.items()}
    verdict = None if validator is None else validator(transcript, request)
    row = index_row(request, transcript, rewrite, faults, verdict, placeholders=placeholders)
    return Outcome(transcript, row, faults)


def _request_fields(request: dict) -> dict:
    """What an index row copies from its request record: the REQUEST_KEYS and, under extra, the rest."""
    return {**{key: copy.deepcopy(request[key]) for key in REQUEST_KEYS if key != "slots"},
            "slots": copy.deepcopy(request.get("slots") or {}),
            "extra": {k: copy.deepcopy(v) for k, v in request.items() if k not in REQUEST_KEYS}}


def _transcript_name(rid: str) -> str:
    """Where the run folder keeps the transcript of request rid, relative to the folder: what the row's
    transcript field names."""
    return f"{TRANSCRIPTS_DIR}/{rid}.json"


def index_row(request: dict, transcript: Transcript, rewrite: Rewrite | None, faults: dict,
              verdict: dict | None, *, placeholders: bool) -> dict:
    """The index.jsonl row of one request (INDEX_KEYS)."""
    error = None
    if transcript.status == HARNESS_ERROR:
        error = next((s["harness_error"] for s in reversed(transcript.steps) if "harness_error" in s), None)
    return {
        **_request_fields(request),
        "rewritten": None if rewrite is None else rewrite.text,
        "rewrite_fallback": rewrite is not None and rewrite.fallback,
        "rewrite_cost_rmb": 0.0 if rewrite is None else rewrite.cost_rmb,
        "status": transcript.status, "error": error, "steps": len(transcript.steps),
        "paid_calls": transcript.paid_calls, "cache_hits": transcript.cache_hits,
        "cost_rmb": transcript.cost_rmb, "faults": faults, "verdict": verdict, "placeholders": placeholders,
        "transcript": _transcript_name(request["id"]),
    }


def error_row(request: dict, error: str, *, placeholders: bool) -> dict:
    """The index row (INDEX_KEYS) of a request whose run_request raised before there was a transcript:
    status harness_error, the exception text in error, transcript null, nothing paid and no fault met."""
    return {**_request_fields(request), "rewritten": None, "rewrite_fallback": False, "rewrite_cost_rmb": 0.0,
            "status": HARNESS_ERROR, "error": error, "steps": 0, "paid_calls": 0, "cache_hits": 0,
            "cost_rmb": 0.0, "faults": {}, "verdict": None, "placeholders": placeholders, "transcript": None}


def summary_line(outcome: Outcome) -> str:
    """One line per request, as scripts/m1_smoke.py prints one."""
    t, row = outcome.transcript, outcome.row
    calls = [c["function"]["name"] for m in t.messages if m.get("tool_calls") for c in m["tool_calls"]]
    faults = sum(n for counts in outcome.faults.values() for fault, n in counts.items() if fault != "none")
    rewrite = "off" if row["rewritten"] is None else ("fallback" if row["rewrite_fallback"] else "ok")
    return (f"{row['id']} ({row['scenario']}/{row['lang']}/{row['split']}): {t.status}, "
            f"steps={len(t.steps)}, calls={calls}, faults={faults}, rewrite={rewrite}, paid={t.paid_calls}, "
            f"cached={t.cache_hits}, cost={t.cost_rmb + row['rewrite_cost_rmb']:.4f} RMB")


# ---- the realized mix -----------------------------------------------------------------------------------


def realized_scenario(transcript: Transcript) -> str | None:
    """What the trajectory turned out to be, by the module docstring's rules; None after a harness error."""
    if transcript.status == HARNESS_ERROR:
        return None
    calls = [len(m.get("tool_calls") or []) for m in transcript.messages if m.get("role") == "assistant"]
    if any(n >= 2 for n in calls):
        return "parallel"
    if transcript.status == CLARIFICATION:
        return "clarify"
    if not any(calls):
        return "no_call"
    if any(_error_code(m) in RECOVER_CODES for m in transcript.messages if m.get("role") == "tool"):
        return "recover"
    return "normal"


def _error_code(message: dict) -> str | None:
    """The error code of a tool message, read from its canonical JSON content; None when there is none."""
    try:
        body = json.loads(message.get("content") or "")
    except (json.JSONDecodeError, TypeError):
        return None
    return body.get("error") if isinstance(body, dict) else None


def realized_mix(scenarios: Iterable[str | None]) -> dict:
    """Counts and shares per scenario over the trajectories that have one (None is left out)."""
    counted = [s for s in scenarios if s is not None]
    counts = {scenario: counted.count(scenario) for scenario in SCENARIOS}
    total = len(counted)
    return {"trajectories": total, "counts": counts,
            "shares": {scenario: (counts[scenario] / total if total else 0.0) for scenario in SCENARIOS}}


# ---- the run --------------------------------------------------------------------------------------------


def corpus_recorded_on(manifest: str | Path = MANIFEST) -> str:
    """The corpus manifest's recording date: the default today of a run, so the system prompt's date is the
    date the corpus was recorded on."""
    return json.loads(Path(manifest).read_text(encoding="utf-8"))["recorded_on"]


def generate(requests: Iterable[dict], *, out: str | Path, model: Model, owner: OwnerFunctions,
             fetcher: SupportsGet, index: CorpusIndex, today: str, settings: dict | None = None,
             placeholders: bool = False, count_tokens: TokenCounter | None = None, seed: int = 42,
             rewriter: Rewriter | None = None, note_tools: dict[str, ToolFn] | None = None,
             notes: str = "store", validator: TrajectoryValidator | None = None, redo: bool = False,
             corpus_manifest: str | Path | None = MANIFEST, git_dir: str | Path | None = REPO,
             seed_notes: bool = True, requests_manifest: str | Path | None = None,
             log: Callable[[str], None] = print) -> dict:
    """Run the requests in order into the run folder out and return the manifest (also written there).

    settings are the caller's settings as the manifest's config records them (the requests file and its
    hash, the run name, the caps and filters, the tokenizer); seed, placeholders, the model's temperature
    (model_temperature; None for a model without one), the rewrite switch, the rewrite temperature and notes
    are added here. notes is one of NOTE_SETTINGS: "store" gives request <id> the real note tools over
    out/notes/<id>/, "stub" the stand-ins (note_tools, when given, overrides both, as in run_request; a
    ValueError for any other value, before anything is written). seed_notes (with notes="store" and no
    note_tools; otherwise there is no store to seed and the config records false) writes each request's seed
    notes into its store before its turn (seed_plan), drawn from the topics and papers of the corpus manifest
    (corpus_manifest, which seeding then needs: a ValueError without one) that the request's split may use
    by requests_manifest (requests_manifest_of the requests file; None: the whole corpus). The config records
    seed_notes and seed_scope ("split", "corpus", None when not seeding); the manifest's seeded_notes the
    split restriction and the notes seeded per request (0 for an error row, whose notes are deleted). A
    manifest already in out must record the
    same RUN_SETTINGS, else a HarnessError before anything is written (the module docstring). Rows already in
    out/index.jsonl are kept and skipped as the module docstring says (a budget_stop row is run again without
    redo); the index and the manifest are rewritten after every request, and an error row deletes the
    transcript and the notes an earlier run left for its request.
    """
    if notes not in NOTE_SETTINGS:
        raise ValueError(f"notes must be one of {list(NOTE_SETTINGS)}, not {notes!r}")
    out = Path(out)
    seeding = seed_notes and notes == "store" and note_tools is None
    if seeding and corpus_manifest is None:
        raise ValueError("seed notes are drawn from the corpus manifest's topics: pass corpus_manifest, or "
                         "seed_notes=False")
    pool = load_seed_pool(corpus_manifest, requests_manifest, index.paper) if seeding else None
    config = {**(settings or {}), "seed": seed, "placeholders": placeholders, "rewrite": rewriter is not None,
              "temperature": None if rewriter is None else rewriter.temperature,
              "model_temperature": getattr(model, "temperature", None), "notes": notes,
              "seed_notes": seeding, "seed_scope": None if pool is None else pool.scope}
    fixed = {"config": config, "today": today,
             "corpus_manifest_sha256": None if corpus_manifest is None else sha256_of(corpus_manifest),
             "schema_hash": schemas.SCHEMA_HASH, "prompt_version": PROMPT_VERSION, "prompt_hash": PROMPT_HASH,
             "rewrite_prompt_version": REWRITE_PROMPT_VERSION, "rewrite_prompt_hash": REWRITE_PROMPT_HASH,
             "git_commit": git_commit(git_dir), "git_dirty": git_dirty(git_dir)}
    _check_same_run(out / MANIFEST_FILE, run_settings(fixed))
    rows = {row["id"]: row for row in _read_index(out / INDEX_FILE)}
    realized = {rid: _realized_of(out, row) for rid, row in rows.items()}

    def seeded_count(row: dict) -> int:
        """The seed notes the row's request has: none for an error row (they were deleted) or without a pool;
        the plan is a function of the request id, split and slots, which the row carries."""
        if pool is None or row["transcript"] is None:
            return 0
        return len(seed_plan(row, pool, index, today))

    def flush() -> dict:
        _write_text(out / INDEX_FILE, "".join(canonical_json(rows[rid]) + "\n" for rid in sorted(rows)))
        seeded = {"split_restricted": None if pool is None else pool.splits is not None,
                  "per_request": {rid: seeded_count(rows[rid]) for rid in sorted(rows)}}
        manifest = build_manifest(fixed, rows, realized, seeded,
                                  validated=validator is not None and validator.available)
        _write_text(out / MANIFEST_FILE, json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n")
        return manifest

    manifest = flush()
    for request in requests:
        rid = request["id"]
        present = rows.get(rid)
        if (present is not None and not redo and present["placeholders"] == placeholders
                and present["status"] != BUDGET_STOP):
            log(f"{rid}: already in the index, skipped")
            continue
        try:
            outcome = run_request(request, model=model, owner=owner, fetcher=fetcher, index=index,
                                  today=today, count_tokens=count_tokens, seed=seed, rewriter=rewriter,
                                  note_tools=note_tools,
                                  notes_dir=_notes_dir(out, rid) if notes == "store" else None,
                                  validator=validator, placeholders=placeholders, seed_pool=pool)
        except BudgetExceeded as exc:
            log(f"{rid}: {exc}; the run stops before this request")
            break
        except Exception as exc:  # noqa: BLE001  deliberate: the failure is recorded, the run goes on
            rows[rid] = error_row(request, str(exc) or type(exc).__name__, placeholders=placeholders)
            realized[rid] = None
            manifest = flush()
            _discard_transcript(out, rid)  # after the flush, so the index never names a file that is gone
            log(f"{rid}: {HARNESS_ERROR} before a transcript existed, {type(exc).__name__}: {exc}")
            continue
        save_transcript(outcome.transcript, out / outcome.row["transcript"])
        rows[rid] = outcome.row
        realized[rid] = realized_scenario(outcome.transcript)
        manifest = flush()
        log(summary_line(outcome))
        if outcome.transcript.status == BUDGET_STOP:
            log(f"{rid}: the spending cap was reached inside the turn; the run stops")
            break
    return manifest


def build_manifest(fixed: dict, rows: dict[str, dict], realized: dict[str, str | None],
                   seeded: dict | None = None, *, validated: bool) -> dict:
    """The manifest: the fixed part plus what the rows add up to (sums taken in id order, so a resumed run
    and a fresh run give the same bytes), and seeded as seeded_notes (generate's split restriction and
    notes per request)."""
    ordered = [rows[rid] for rid in sorted(rows)]
    by_scenario: dict[str, dict[str, int]] = {}
    by_split: dict[str, dict[str, int]] = {}
    by_status: dict[str, int] = {}
    for row in ordered:
        scenario = by_scenario.setdefault(row["scenario"], {})
        scenario[row["status"]] = scenario.get(row["status"], 0) + 1
        split = by_split.setdefault(row["split"], {})
        split[row["status"]] = split.get(row["status"], 0) + 1
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    trajectories = sum(row["cost_rmb"] for row in ordered)
    rewrites = sum(row["rewrite_cost_rmb"] for row in ordered)
    return {
        **fixed, "validated": validated, "requests": len(ordered), "seeded_notes": seeded,
        "counts": {"by_scenario": by_scenario, "by_split": by_split, "by_status": by_status},
        "planned_mix": dict(PLANNED_MIX),
        "realized_mix": realized_mix(realized[rid] for rid in sorted(realized)),
        "cost_rmb": {"trajectories": trajectories, "rewrites": rewrites, "total": trajectories + rewrites},
        "calls": {"paid": sum(row["paid_calls"] for row in ordered),
                  "cached": sum(row["cache_hits"] for row in ordered)},
    }


def run_settings(manifest: dict) -> dict:
    """The RUN_SETTINGS a manifest records (None for one it lacks): what the rows of its folder were
    generated under. seed, model, model_temperature, placeholders, rewrite, temperature, requests_sha256,
    notes, seed_notes and seed_scope sit under config, today and the three hashes at the top."""
    config = manifest.get("config")
    config = config if _is_object(config) else {}
    settings = {key: (config if key in _CONFIG_SETTINGS else manifest).get(key) for key in RUN_SETTINGS}
    # a run made before M7 records none of the three: the teacher rewrote too, and thinking was decided per
    # request (research_agent.llm.cloud, auto)
    if "rewrite_model" not in config:
        settings["rewrite_model"] = settings["model"] if settings["rewrite"] else None
    if "thinking" not in config:
        settings["thinking"] = "auto"
    if "rewrite_thinking" not in config:
        settings["rewrite_thinking"] = "auto" if settings["rewrite"] else None
    return settings


def _check_same_run(path: Path, wanted: dict) -> None:
    """Refuse to resume a run folder whose manifest records other RUN_SETTINGS than wanted: its rows were
    generated under them, and a manifest rewritten from this call would misdescribe those rows."""
    if not path.exists():
        return
    recorded = run_settings(_read_manifest(path))
    changes = [f"{key} {recorded[key]!r} -> {wanted[key]!r}" for key in RUN_SETTINGS
               if recorded[key] != wanted[key]]
    if changes:
        raise HarnessError(f"{path}: the run there was generated with other settings ({', '.join(changes)}); "
                           "a different setting is a different run: use another --run name (--redo does "
                           "not override this)")


def _read_manifest(path: Path) -> dict:
    """A manifest file as a dict; a HarnessError when it is not JSON or not an object."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HarnessError(f"{path} is not JSON: {exc}") from exc
    if not _is_object(value):
        raise HarnessError(f"{path} is not a JSON object")
    return value


def _realized_of(out: Path, row: dict) -> str | None:
    """The realized scenario of a resumed row, read from its transcript; None for a row without one."""
    return None if row["transcript"] is None else realized_scenario(load_transcript(out / row["transcript"]))


def _discard_transcript(out: Path, rid: str) -> None:
    """Delete the transcript and the notes an earlier run of request rid left in out, if there are any: the
    request's row is now an error row (transcript null), and files the index does not name would misdescribe
    the folder. Only that request's own file and notes folder are touched."""
    (out / _transcript_name(rid)).unlink(missing_ok=True)
    _remove_tree(_notes_dir(out, rid))


def _notes_dir(out: Path, rid: str) -> Path:
    """Where request rid keeps its note store in the run folder out."""
    return out / NOTES_DIR / rid


def _remove_tree(path: Path) -> None:
    """Delete the folder path with everything in it, when it exists."""
    if path.is_dir():
        shutil.rmtree(path)


def _read_index(path: Path) -> list[dict]:
    """The rows of an index file, each checked to be an index row; none when there is no file yet. A
    HarnessError names the line that is not JSON, not an object, lacks one of INDEX_KEYS, or whose id is
    not a non-empty string, placeholders not a bool, status not a string or transcript neither a string nor
    null."""
    if not path.exists():
        return []
    rows = []
    for number, line in enumerate(split_lines(path.read_text(encoding="utf-8")), 1):
        if not line.strip():
            continue
        where = f"{path}: line {number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HarnessError(f"{where} is not JSON: {exc}") from exc
        if not _is_object(row):
            raise HarnessError(f"{where} is not an index row (not a JSON object)")
        rows.append(_checked_row(row, where))
    return rows


_ROW_CHECKS = (("id", _is_text, "a non-empty string"),
               ("placeholders", lambda v: isinstance(v, bool), "true or false"),
               ("status", lambda v: isinstance(v, str), "a string"),
               ("transcript", lambda v: v is None or isinstance(v, str), "a string or null"))


def _checked_row(row: dict, where: str) -> dict:
    for key in INDEX_KEYS:
        if key not in row:
            raise HarnessError(f"{where} lacks {key}")
    for key, fine, kind in _ROW_CHECKS:
        if not fine(row[key]):
            raise HarnessError(f"{where}: {key} must be {kind}, not {row[key]!r:.80}")
    return row


def _write_text(path: Path, text: str) -> None:
    """UTF-8 with LF line ends, through a temporary file, so a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)

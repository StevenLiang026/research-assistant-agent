"""Draw the slot records of the v2 end-to-end test tasks (docs/m7-计划.md WP-I 6): every field of a task but
its text, drawn mechanically from one seed, and cut into ten writing batches.

    .venv/Scripts/python.exe scripts/draw_e2e_slots.py [--seed 42] [--out data/eval/v2/e2e-slots-v2.jsonl]
        [--batches data/eval/v2/e2e-drafts] [--check] [--force]

A slot record is a task of research_agent.eval.e2e without "text": id, split ("test"), family ("e2e/<intent>",
as the dev tasks), intent, lang, scenario (e2e.SCENARIO_OF), slots and expect. Writers add the text
(data/eval/v2/e2e-writing.md); scripts/check_e2e_tasks.py checks what they write against these records.

Counts. The intents keep the dev tasks' proportions (data/eval/e2e-dev.jsonl, lock-checked: 4, 3, 3, 2, 2, 1,
1, 2, 2 of 20 in E2E_INTENTS order) scaled to SIZE by largest remainder, ties going to the earlier intent of
E2E_INTENTS: 30, 23, 23, 15, 15, 7, 7, 15, 15 of 150. Within an intent half the tasks are Chinese and half
English; an odd count's extra task goes to Chinese for the first odd intent, to English for the next, and so
on, so the set is 75 and 75. Ids run from e2e-test-61 to e2e-test-210 in E2E_INTENTS order, within an intent
Chinese and English alternating (Chinese first), the larger language's extra tasks last.

Shapes. Each intent cycles through the slot shapes of its dev tasks, in dev order, per language:
find_recent_papers since, since, since with categories, between; read_section a topic between two dates, a
named paper, a topic since a date (each with a section); write_note a named paper, a topic between two dates,
a named paper with a section (each with mode scaffold); compare_papers two papers; lookup_paper one paper;
search_notes a topic; list_notes a date; needs_clarification and chit_chat no slots.

Values, from the test split only (the requests manifest's topics["test"] and their papers in the corpus
manifest) and the recording window (the requests manifest's window, ending on the recording date):
- topic_id: the test topic least used so far by tasks with a topic (ties drawn); topic: one of its phrases as
  e2e.corpus_topics lists them (the id with spaces, then the manifest's phrases; the note check's phrases),
  the one least used so far;
- count: 2 to 5; categories: one (two times in three) or two of schemas.CATEGORIES, sorted;
- since a date (date_from alone): the first of a month two times in five, else a day of the window but its
  last week; between two dates: whole months two times in five (one to three of the window's whole months),
  else two days of the window at least seven days apart;
- section: introduction, method, experiments or conclusion, the one least used so far by the intent's tasks
  of that kind (on a topic, on a named paper);
- arxiv_id: the test paper least used so far by all tasks (ties drawn); arxiv_ids: two papers of the test
  topic least used so far by pairs, the pair with the least summed use, in id order; mode: scaffold;
- since (list_notes): a day of the window, the first of a month two times in five when one qualifies, that
  leaves at least one of the task's seeded notes (data.synth.seed_plan: the five notes the run seeds its
  store with, dated from the task id alone) on or after it and at least one before it.

Conditions. A drawn value is drawn again (ATTEMPTS times, then the next topic, phrase, section or paper in
order) until all of these hold, so the contract of tests/eval/test_e2e.py holds by construction:
- every date lies in the window, date_from <= date_to;
- the contract's topic search (query: the topic, unquoted; date_from, date_to and categories from the slots;
  on the recorded corpus index) has at least count + SLACK hits (count 1 when the slots have none), and at
  least as many of them are on the topic by the note check's rule (a paper of the topic, or one whose title
  or abstract holds one of its phrases), so the request can be met with papers on its topic;
- every named paper is in the recorded metadata; for read_section, write_note and compare_papers its
  introduction (fetch_fulltext's default section) and its named section can be read from the recorded full
  text (arxiv.extract_section gives text, as fetch_fulltext does), so a correct call never meets NOT_FOUND;
  for a read_section or write_note task on a topic, the section (introduction for a note) of every on-topic
  hit can be read, or the paper has no HTML full text (fetch_fulltext answers NO_FULLTEXT with the abstract,
  which is not a failure);
- no two tasks with slots have the same slot values. needs_clarification and chit_chat tasks have none, as
  the dev tasks; their texts keep them apart (scripts/check_e2e_tasks.py: the between-task check).

Expect. e2e.py has no mapping from intent and slots to the expect block, so expect_for is it: status
clarification for needs_clarification and answered otherwise; a note required for write_note, about the named
paper ({"arxiv_id"}) or else the topic ({"topic_id"}); cited ids within the tool outputs; the expected tools
TOOLS_OF[intent]. tests/eval/test_draw_e2e_slots.py checks it against all 20 dev tasks.

Batches. Per language, the records in intent order (E2E_INTENTS, then id) are dealt round robin into five
batches: slots-01.jsonl to slots-05.jsonl Chinese, slots-06.jsonl to slots-10.jsonl English, 15 records each,
every intent spread evenly over a language's batches (counts differ by at most one); a batch is in id order.
A writer adds "text" to each record of a batch and writes tasks-<nn>.jsonl beside it.

Writes --out and the ten batch files under --batches as canonical JSON lines (eval.lock.write_jsonl); the same
seed and inputs give the same bytes. An output that exists and differs from the draw is kept unless --force,
since writers work from it. --check draws, compares and writes nothing. Exit 0 when written (or already the
same) or when --check finds the files equal to the draw; 1 when --check finds a difference or the draw cannot
meet its conditions; 2 when an output exists and differs without --force, or an input is missing or
malformed.
"""

from __future__ import annotations

import argparse
import calendar
import datetime
import hashlib
import itertools
import json
import random
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from research_agent.agent.types import ANSWERED, CLARIFICATION, HarnessError
from research_agent.arxiv import extract_section, parse_atom
from research_agent.canonical import canonical_json
from research_agent.data.synth import SeedPool, corpus_recorded_on, load_seed_pool, seed_plan
from research_agent.eval.e2e import CITE_SCOPE, E2E_INTENTS, SCENARIO_OF, Topic, corpus_topics
from research_agent.eval.lock import E2E_DEV_SET, DevSet, LockError
from research_agent.net.fetch import Fetcher
from research_agent.normalize import normalize_text
from research_agent.tools import schemas
from research_agent.tools.adapters import fulltext_url
from research_agent.tools.corpus_search import CorpusIndex, corpus_ids, load_papers
from research_agent.train.manifest import REPO

SEED = 42
SIZE = 150
ID_START = 61
SPLIT = "test"
LANGS = ("zh", "en")  # the order odd units alternate in, and the batches' order
SLACK = 2  # recorded hits beyond the count, as tests/eval/test_e2e.py's SLACK
COUNTS = (2, 3, 4, 5)
SECTIONS = ("introduction", "method", "experiments", "conclusion")
DEFAULT_SECTION = schemas.DEFAULTS["fetch_fulltext"]["section"]
MAX_CHARS = schemas.DEFAULTS["fetch_fulltext"]["max_chars"]
MODE = "scaffold"
BATCH_COUNT = 10
ATTEMPTS = 60  # draws of the dates, count and categories per topic, phrase and section
WHOLE_MONTHS = 0.4  # the share of dates drawn as whole months or firsts of a month
MIN_SPAN_DAYS = 7
OUT = Path("data/eval/v2/e2e-slots-v2.jsonl")
BATCHES = Path("data/eval/v2/e2e-drafts")
CORPUS_MANIFEST = Path("data/corpus/manifest.json")
REQUESTS_MANIFEST = Path("data/synth/requests-manifest.json")
CASSETTES = Path("data/cassettes")
NAMED_FULLTEXT = ("read_section", "write_note", "compare_papers")  # intents whose papers are read in full
TOOLS_OF = {"find_recent_papers": ("search_arxiv",), "read_section": ("fetch_fulltext",),
            "write_note": ("write_note",), "compare_papers": ("get_paper", "fetch_fulltext"),
            "lookup_paper": ("get_paper",), "search_notes": ("search_notes", "list_notes"),
            "list_notes": ("search_notes", "list_notes"), "needs_clarification": ("ask_clarification",),
            "chit_chat": ()}


@dataclass(frozen=True)
class Shape:
    """The slots one kind of task carries (module docstring, Shapes)."""

    topic: bool = False
    dates: str | None = None  # "since" (date_from alone) or "between" (date_from and date_to)
    count: bool = False
    categories: bool = False
    section: bool = False
    papers: int = 0  # 1: arxiv_id, 2: arxiv_ids
    mode: bool = False
    notes_since: bool = False  # list_notes' since

    @property
    def drawn(self) -> bool:
        """Whether the slots hold a value drawn at random beyond the topic, section and papers."""
        return bool(self.dates or self.count or self.categories)


SINCE_COUNT = Shape(topic=True, dates="since", count=True)
SHAPES: dict[str, tuple[Shape, ...]] = {
    "find_recent_papers": (SINCE_COUNT, SINCE_COUNT,
                           Shape(topic=True, dates="since", count=True, categories=True),
                           Shape(topic=True, dates="between", count=True)),
    "read_section": (Shape(topic=True, dates="between", section=True), Shape(papers=1, section=True),
                     Shape(topic=True, dates="since", section=True)),
    "write_note": (Shape(papers=1, mode=True), Shape(topic=True, dates="between", mode=True),
                   Shape(papers=1, mode=True, section=True)),
    "compare_papers": (Shape(papers=2),),
    "lookup_paper": (Shape(papers=1),),
    "search_notes": (Shape(topic=True),),
    "list_notes": (Shape(notes_since=True),),
    "needs_clarification": (Shape(),),
    "chit_chat": (Shape(),),
}


class DrawError(RuntimeError):
    """No slot values meet the conditions for a task."""


# ---- counts ---------------------------------------------------------------------------------------------


def dev_intent_counts(path: str | Path = REPO / E2E_DEV_SET) -> Counter[str]:
    """The intents of the end-to-end dev tasks, counted (the file is lock-checked: eval.lock.DevSet)."""
    return Counter(task["intent"] for task in DevSet(path).open())


def intent_counts(dev_counts: Mapping[str, int], size: int,
                  order: tuple[str, ...] = E2E_INTENTS) -> dict[str, int]:
    """dev_counts scaled to size by largest remainder, in order; equal remainders go to the earlier intent."""
    total = sum(dev_counts.get(intent, 0) for intent in order)
    floors = {intent: dev_counts.get(intent, 0) * size // total for intent in order}
    remainders = {intent: dev_counts.get(intent, 0) * size % total for intent in order}
    left = size - sum(floors.values())
    for intent in sorted(order, key=lambda i: (-remainders[i], order.index(i)))[:left]:
        floors[intent] += 1
    return {intent: floors[intent] for intent in order}


def language_counts(counts: Mapping[str, int],
                    order: tuple[str, ...] = E2E_INTENTS) -> dict[str, dict[str, int]]:
    """Per intent {"zh": n, "en": n}: half and half, an odd count's extra task to Chinese for the first odd
    intent, to English for the next, and so on."""
    out, turn = {}, 0
    for intent in order:
        n = counts[intent]
        split = {lang: n // 2 for lang in LANGS}
        if n % 2:
            split[LANGS[turn % 2]] += 1
            turn += 1
        out[intent] = split
    return out


def task_plan(counts: Mapping[str, int]) -> list[tuple[str, str, int]]:
    """(intent, lang, k) in id order (module docstring): k counts the intent's tasks in that language."""
    langs = language_counts(counts)
    rows = []
    for intent in E2E_INTENTS:
        seen: Counter[str] = Counter()
        for i in range(max(langs[intent].values())):
            for lang in LANGS:
                if i < langs[intent][lang]:
                    rows.append((intent, lang, seen[lang]))
                    seen[lang] += 1
    return rows


# ---- expect ---------------------------------------------------------------------------------------------


def expect_for(intent: str, slots: Mapping[str, object]) -> dict:
    """The expect block of a task with this intent and these slots (module docstring, Expect)."""
    note = intent == "write_note"
    about = None
    if note:
        about = {"arxiv_id": slots["arxiv_id"]} if "arxiv_id" in slots else {"topic_id": slots["topic_id"]}
    return {"expected_tools_any": list(TOOLS_OF[intent]), "must_cite_ids_within": CITE_SCOPE,
            "must_write_note": note, "note_about": about,
            "status": CLARIFICATION if intent == "needs_clarification" else ANSWERED}


# ---- the world ------------------------------------------------------------------------------------------


@dataclass
class World:
    """What the draw reads: the recorded corpus index and full texts, the test split's topics, the window,
    the seed pool of the run's note stores and the dev tasks' intent counts."""

    index: CorpusIndex
    fetcher: Fetcher
    topics: dict[str, Topic]  # the test split's topics: papers and the note check's phrases
    window: tuple[str, str]
    today: str
    pool: SeedPool
    dev_counts: Counter[str]
    _status: dict[str, int | None] = field(default_factory=dict)
    _readable: dict[tuple[str, str], bool] = field(default_factory=dict)
    _on_topic: dict[tuple[str, str], bool] = field(default_factory=dict)

    @property
    def split_topics(self) -> list[str]:
        return sorted(self.topics)

    def papers(self, topic: str | None = None) -> list[str]:
        """The test papers (of one topic), sorted."""
        chosen = [topic] if topic is not None else self.split_topics
        return sorted(a for t in chosen for a in self.topics[t].papers)

    def days(self) -> list[str]:
        """Every day of the window, up to today, in order."""
        last = min(self.window[1], self.today)
        start, end = datetime.date.fromisoformat(self.window[0]), datetime.date.fromisoformat(last)
        return [(start + datetime.timedelta(days=n)).isoformat() for n in range((end - start).days + 1)]

    def whole_months(self) -> list[int]:
        """The months that lie wholly inside the window."""
        days = set(self.days())
        months = sorted({int(d[5:7]) for d in days})
        year = int(self.window[0][:4])
        return [m for m in months if {_iso(year, m, 1), _iso(year, m, _last_day(year, m))} <= days]

    def status(self, arxiv_id: str) -> int | None:
        """The recorded full text's status, None without a cassette."""
        if arxiv_id not in self._status:
            page = self.fetcher.recorded(fulltext_url(arxiv_id))
            self._status[arxiv_id] = None if page is None else page.status
        return self._status[arxiv_id]

    def readable(self, arxiv_id: str, section: str) -> bool:
        """Whether fetch_fulltext gives text for this section of the recorded full text."""
        key = (arxiv_id, section)
        if key not in self._readable:
            page = self.fetcher.recorded(fulltext_url(arxiv_id))
            ok = page is not None and page.status == 200
            if ok:
                text = extract_section(page.text, section, MAX_CHARS)
                ok = bool(text and text.strip())
            self._readable[key] = ok
        return self._readable[key]

    def on_topic(self, paper: dict, topic: str) -> bool:
        """The note check's rule (e2e.score_task, note_about topic_id): a paper of the topic, or one whose
        title or abstract holds one of its phrases."""
        key = (paper["arxiv_id"], topic)
        if key not in self._on_topic:
            entry = self.topics[topic]
            text = normalize_text(f"{paper.get('title', '')} {paper.get('abstract', '')}").casefold()
            self._on_topic[key] = (paper["arxiv_id"] in entry.papers
                                   or any(normalize_text(p).casefold() in text for p in entry.phrases))
        return self._on_topic[key]

    def search(self, slots: Mapping[str, object]) -> list[dict]:
        """The contract's topic search: the topic unquoted, with the slots' dates and categories."""
        args = {"query": slots["topic"], **{k: slots[k] for k in ("date_from", "date_to", "categories")
                                            if k in slots}}
        return self.index.search(args)

    def topic_ok(self, intent: str, slots: Mapping[str, object]) -> bool:
        """The conditions on a task with a topic (module docstring, Conditions)."""
        if intent == "search_notes":
            return True
        hits = self.search(slots)
        need = int(slots.get("count", 1)) + SLACK
        on = [h for h in hits if self.on_topic(h, str(slots["topic_id"]))]
        if len(hits) < need or len(on) < need:
            return False
        if intent in NAMED_FULLTEXT:
            section = str(slots.get("section", DEFAULT_SECTION))
            return all(self.status(h["arxiv_id"]) == 404 or self.readable(h["arxiv_id"], section) for h in on)
        return True

    def paper_ok(self, intent: str, arxiv_id: str, section: str | None) -> bool:
        """The conditions on a named paper (module docstring, Conditions)."""
        if self.index.paper(arxiv_id) is None:
            return False
        if intent not in NAMED_FULLTEXT:
            return True
        return (self.readable(arxiv_id, DEFAULT_SECTION)
                and (section is None or self.readable(arxiv_id, section)))

    def seeded_on(self, task_id: str) -> list[str]:
        """The created dates of the notes a run seeds this task's store with (data.synth.seed_plan)."""
        request = {"id": task_id, "split": SPLIT, "slots": {}}
        return [created for _, _, created in seed_plan(request, self.pool, self.index, self.today)]


def load_world(*, corpus_manifest: str | Path = REPO / CORPUS_MANIFEST,
               requests_manifest: str | Path = REPO / REQUESTS_MANIFEST,
               cassettes: str | Path = REPO / CASSETTES, dev: str | Path = REPO / E2E_DEV_SET) -> World:
    """The world of the draw from the corpus manifest, the requests manifest (the split and the window), the
    cassettes and the end-to-end dev tasks."""
    fetcher = Fetcher("replay", cassettes)
    index = CorpusIndex(load_papers(fetcher, corpus_ids(corpus_manifest), parse_atom))
    manifest = json.loads(Path(requests_manifest).read_text(encoding="utf-8"))
    every = corpus_topics(corpus_manifest)
    topics = {t: every[t] for t in sorted(manifest["topics"][SPLIT])}
    start, end = manifest["window"]
    pool = load_seed_pool(corpus_manifest, requests_manifest, index.paper)
    return World(index, fetcher, topics, (start, end), corpus_recorded_on(corpus_manifest), pool,
                 dev_intent_counts(dev))


# ---- the draw -------------------------------------------------------------------------------------------


def _iso(year: int, month: int, day: int) -> str:
    return datetime.date(year, month, day).isoformat()


def _last_day(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


@dataclass
class _State:
    rng: random.Random
    seen: set[str] = field(default_factory=set)  # the canonical slots of every task with slots
    use: Counter[object] = field(default_factory=Counter)  # topics, phrases, sections, papers, pair topics

    def ordered(self, items: Iterable, key: Callable[[object], object] = lambda item: item) -> list:
        """items, least used first (use[key(item)]), ties in a drawn order."""
        items = sorted(items)
        ties = {item: self.rng.random() for item in items}
        return sorted(items, key=lambda item: (self.use[key(item)], ties[item]))

    def fresh(self, slots: dict) -> bool:
        return canonical_json(slots) not in self.seen

    def take(self, slots: dict, *used: object) -> dict:
        self.seen.add(canonical_json(slots))
        self.use.update(used)
        return slots


def _dates(world: World, rng: random.Random, kind: str | None) -> dict[str, str]:
    if kind is None:
        return {}
    days = world.days()
    year = int(world.window[0][:4])
    if kind == "since":
        early = days[:-MIN_SPAN_DAYS]
        firsts = [d for d in early if d.endswith("-01")]
        if firsts and rng.random() < WHOLE_MONTHS:
            return {"date_from": rng.choice(firsts)}
        return {"date_from": rng.choice(early)}
    months = world.whole_months()
    if months and rng.random() < WHOLE_MONTHS:
        first, last = rng.choice([(a, b) for a, b in itertools.combinations_with_replacement(months, 2)])
        return {"date_from": _iso(year, first, 1), "date_to": _iso(year, last, _last_day(year, last))}
    while True:
        i, j = sorted(rng.sample(range(len(days)), 2))
        if j - i >= MIN_SPAN_DAYS:
            return {"date_from": days[i], "date_to": days[j]}


def _categories(rng: random.Random) -> list[str]:
    return sorted(rng.sample(schemas.CATEGORIES, rng.choice((1, 1, 2))))


def _topic_slots(world: World, state: _State, intent: str, shape: Shape) -> dict:
    sections = [None]
    for topic in state.ordered(world.split_topics, key=lambda t: ("topic", t)):
        phrases = state.ordered(world.topics[topic].phrases, key=lambda p, t=topic: ("phrase", t, p))
        for phrase in phrases:
            if shape.section:
                sections = state.ordered(SECTIONS, key=lambda s: ("section", intent, "topic", s))
            for section in sections:
                for _ in range(ATTEMPTS if shape.drawn else 1):
                    slots: dict = {"topic": phrase, "topic_id": topic,
                                   **_dates(world, state.rng, shape.dates)}
                    if shape.count:
                        slots["count"] = state.rng.choice(COUNTS)
                    if shape.categories:
                        slots["categories"] = _categories(state.rng)
                    if section is not None:
                        slots["section"] = section
                    if shape.mode:
                        slots["mode"] = MODE
                    if state.fresh(slots) and world.topic_ok(intent, slots):
                        used = [("topic", topic), ("phrase", topic, phrase)]
                        if section is not None:
                            used.append(("section", intent, "topic", section))
                        return state.take(slots, *used)
    raise DrawError(f"no topic slots of a {intent} task meet the conditions")


def _paper_slots(world: World, state: _State, intent: str, shape: Shape) -> dict:
    sections = ([None] if not shape.section
                else state.ordered(SECTIONS, key=lambda s: ("section", intent, "paper", s)))
    for section in sections:
        for arxiv_id in state.ordered(world.papers(), key=lambda a: ("paper", a)):
            slots: dict = {"arxiv_id": arxiv_id}
            if shape.mode:
                slots["mode"] = MODE
            if section is not None:
                slots["section"] = section
            if state.fresh(slots) and world.paper_ok(intent, arxiv_id, section):
                used = [("paper", arxiv_id)] + ([("section", intent, "paper", section)] if section else [])
                return state.take(slots, *used)
    raise DrawError(f"no named paper of a {intent} task meets the conditions")


def _pair_slots(world: World, state: _State, intent: str) -> dict:
    for topic in state.ordered(world.split_topics, key=lambda t: ("pair topic", t)):
        papers = [a for a in world.papers(topic) if world.paper_ok(intent, a, None)]
        ties = {a: state.rng.random() for a in papers}
        pairs = sorted(itertools.combinations(papers, 2),
                       key=lambda p: (state.use[("paper", p[0])] + state.use[("paper", p[1])],
                                      max(state.use[("paper", p[0])], state.use[("paper", p[1])]),
                                      ties[p[0]] + ties[p[1]], p))
        for first, second in pairs:
            slots = {"arxiv_ids": [first, second]}
            if state.fresh(slots):
                return state.take(slots, ("pair topic", topic), ("paper", first), ("paper", second))
    raise DrawError(f"no pair of papers for a {intent} task meets the conditions")


def _notes_since_slots(world: World, state: _State, task_id: str) -> dict:
    created = world.seeded_on(task_id)
    days = [d for d in world.days() if any(c >= d for c in created) and any(c < d for c in created)]
    firsts = [d for d in days if d.endswith("-01")]
    for _ in range(ATTEMPTS):
        pool = firsts if firsts and state.rng.random() < WHOLE_MONTHS else days
        if not pool:
            break
        slots = {"since": state.rng.choice(pool)}
        if state.fresh(slots):
            return state.take(slots)
    raise DrawError(f"{task_id}: no since date leaves seeded notes on both sides")


def draw(world: World, *, seed: int = SEED, size: int = SIZE, id_start: int = ID_START) -> list[dict]:
    """The slot records in id order (module docstring)."""
    state = _State(random.Random(f"e2e-slots-v2:{seed}"))
    records = []
    for number, (intent, lang, k) in enumerate(task_plan(intent_counts(world.dev_counts, size)), id_start):
        task_id = f"e2e-{SPLIT}-{number}"
        shapes = SHAPES[intent]
        shape = shapes[k % len(shapes)]
        if shape.topic:
            slots = _topic_slots(world, state, intent, shape)
        elif shape.papers == 2:
            slots = _pair_slots(world, state, intent)
        elif shape.papers == 1:
            slots = _paper_slots(world, state, intent, shape)
        elif shape.notes_since:
            slots = _notes_since_slots(world, state, task_id)
        else:
            slots = {}
        records.append({"id": task_id, "split": SPLIT, "family": f"e2e/{intent}", "intent": intent,
                        "lang": lang, "scenario": SCENARIO_OF.get(intent, "normal"), "slots": slots,
                        "expect": expect_for(intent, slots)})
    return records


def _number(task_id: str) -> int:
    return int(task_id.rsplit("-", 1)[1])


def batches(records: list[dict], count: int = BATCH_COUNT) -> list[list[dict]]:
    """The writing batches (module docstring, Batches): count // 2 per language, Chinese first."""
    per_lang = count // len(LANGS)
    out = []
    for lang in LANGS:
        mine = sorted((r for r in records if r["lang"] == lang),
                      key=lambda r: (E2E_INTENTS.index(r["intent"]), _number(r["id"])))
        groups: list[list[dict]] = [[] for _ in range(per_lang)]
        for i, record in enumerate(mine):
            groups[i % per_lang].append(record)
        out += [sorted(group, key=lambda r: _number(r["id"])) for group in groups]
    return out


# ---- the script -----------------------------------------------------------------------------------------


def _jsonl(records: list[dict]) -> bytes:
    return "".join(canonical_json(r) + "\n" for r in records).encode("utf-8")


def summary(records: list[dict], cut: list[list[dict]]) -> list[str]:
    """The counts per intent and language and what each batch holds."""
    lines = [f"{len(records)} slot records: " + ", ".join(
        f"{intent} {sum(r['intent'] == intent for r in records)} "
        f"(zh {sum(r['intent'] == intent and r['lang'] == 'zh' for r in records)}, "
        f"en {sum(r['intent'] == intent and r['lang'] == 'en' for r in records)})" for intent in E2E_INTENTS)]
    for n, batch in enumerate(cut, 1):
        intents = Counter(r["intent"] for r in batch)
        lines.append(f"slots-{n:02d}.jsonl: {len(batch)} {batch[0]['lang']} records, "
                     + ", ".join(f"{i} {intents[i]}" for i in E2E_INTENTS if intents[i]))
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw the slot records of the v2 end-to-end test tasks (docs/m7-计划.md WP-I 6) and cut "
                    "them into ten writing batches.")
    parser.add_argument("--seed", type=int, default=SEED, help=f"the draw's seed (default {SEED})")
    parser.add_argument("--out", type=Path, default=REPO / OUT, help=f"the slot records (default {OUT})")
    parser.add_argument("--batches", type=Path, default=REPO / BATCHES,
                        help=f"the folder of slots-01.jsonl .. slots-10.jsonl (default {BATCHES})")
    parser.add_argument("--check", action="store_true", help="draw and compare with the files; write nothing")
    parser.add_argument("--force", action="store_true", help="replace outputs that differ from the draw")
    return parser


def main(argv: list[str] | None = None, *, world: World | None = None) -> int:
    """The script; world, when given, stands in for load_world() (the tests share one: the full texts it has
    parsed are kept)."""
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    try:
        world = world if world is not None else load_world()
        records = draw(world, seed=args.seed)
    except DrawError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, KeyError, LockError, HarnessError) as exc:
        print(f"FAIL: an input is missing or malformed: {exc}", file=sys.stderr)
        return 2
    cut = batches(records)
    outputs = {args.out: _jsonl(records),
               **{args.batches / f"slots-{n:02d}.jsonl": _jsonl(batch) for n, batch in enumerate(cut, 1)}}
    differing = [path for path, body in outputs.items() if not path.is_file() or path.read_bytes() != body]
    if args.check:
        for path in differing:
            print(f"differs from the draw: {path}")
        print("OK: the files are the draw" if not differing else f"FAIL: {len(differing)} file(s) differ")
        return 1 if differing else 0
    kept = [path for path in differing if path.exists()]
    if kept and not args.force:
        for path in kept:
            print(f"FAIL: {path} exists and differs from the draw; pass --force to replace it",
                  file=sys.stderr)
        return 2
    for path in differing:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(outputs[path])
    for line in summary(records, cut):
        print(line)
    print(f"{args.out}: sha256 {hashlib.sha256(outputs[args.out]).hexdigest()}"
          + ("" if differing else " (unchanged)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Check a file of v2 end-to-end test tasks (docs/m7-计划.md WP-I 6) against every condition the contract
tests hold them to (tests/eval/test_e2e.py, v2) and against the writing rules (data/eval/v2/e2e-writing.md),
and name each failing task by its id with reason codes, never with its text.

    .venv/Scripts/python.exe scripts/check_e2e_tasks.py <tasks.jsonl> [--partial] [--json]
        [--slots data/eval/v2/e2e-slots-v2.jsonl] [--drafts <folder>]

The tasks are the slot records of scripts/draw_e2e_slots.py with "text" added. Two modes:
- whole file (the default; data/eval/v2/e2e-test-v2.jsonl once assembled): every per-task check, the
  between-task check over the file, and the set: one task per slot record (a missing one is reported under
  its id as missing_task), the size, and the count per intent and per language equal to the slot records'
  (150; 75 and 75; the drawn intent counts);
- --partial (a writer's batch, tasks-<nn>.jsonl in data/eval/v2/e2e-drafts): every per-task check, and the
  between-task check within the file and against the other drafts present (every other tasks-*.jsonl in
  --drafts, by default the batch's folder; one that cannot be read is named on stderr and passed over); no
  counts of the set, but when slots-<nn>.jsonl lies beside the batch, each of its ids must have a task
  (missing_task) and each task must be one of its ids (not_in_batch).
The dev tasks (data/eval/e2e-dev.jsonl) take part in the between-task check in both modes, as in v1's.

Test material. The checker reads v1's end-to-end test tasks only through eval.lock.RetiredSet. It refuses
(exit 2, before reading anything) a task file or --slots file that is other test material: a locked test set,
by location or by its bytes, or a file of handwritten items (eval.lock.refuse_test_sets under the repository
root); the v2 end-to-end test set itself, data/eval/v2/e2e-test-v2.jsonl, is what it checks. Another draft
that is test material is passed over with a warning, as one that cannot be read.

Per-task checks and their reason codes (REASONS; several may apply; a code may carry details after a colon,
which are ids, field names, counts or similarities, never text). A task is named by its id, or by its line
(#3) when it has no id of up to 64 letters, digits, ".", "_" and "-":
- malformed: not a task (eval.e2e.check_tasks: a non-empty text, an intent of E2E_INTENTS, its scenario, the
  split dev or test, a well-formed expect); no further check runs on it. Also given, besides the slot-record
  codes, to a task whose slots have a value of a type the checks cannot read. duplicate_id: an id used before
  in the file or used by another draft.
- no_slot_record; differs_from_slots:<field>: every field but "text" must equal the task's slot record.
- the contract (tests/eval/test_e2e.py, v1's conditions on the test split): not_test_split;
  topic_outside_split (a topic_id, or the note's topic, outside the test split's topics); id_outside_split and
  id_not_in_metadata (every paper the slots name or the text mentions is a test-split paper, in the corpus
  and in the recorded metadata; the note's paper too); date_outside_window (every slot date, every ISO date
  of the text and every year it mentions lies in the recording window, by today); date_order (date_from after
  date_to); too_few_hits (the topic search: query the topic, unquoted, with the slots' dates and categories,
  on the recorded corpus index, has count + SLACK hits, count 1 when absent; not for search_notes).
- the text expresses exactly its slots (data/eval/v2/e2e-writing.md):
  id_not_in_slots / slot_id_missing (the arXiv ids in the text are exactly the slots' arxiv_id or arxiv_ids);
  topic_missing (the slots' topic phrase appears in the text, case aside; the search check needs it in the
  query); topic_in_longer_phrase (no longer phrase of the topic that contains the slot phrase appears, such
  as knowledge distillation for distillation or looped transformers for looped transformer: the recorded
  hits were counted for the slot phrase); other_topic_in_text (with the slot phrase taken out, the text names
  no other topic of the corpus and no other phrase of its own topic; its Chinese name is fine); topic_in_text
  (a task with neither a topic nor a named paper, list_notes, needs_clarification and chit_chat, names no
  topic); category_missing / category_not_in_slots (the categories in the text are exactly the slots');
  date_missing:<key> / date_not_in_slots (every slot date is written, and every date written is a slot date:
  YYYY-MM-DD, an English or Chinese month and day with or without the year, or a whole month for a from date
  on the first of a month or a to date on its last day); vague_date (no part of a month in place of a day:
  early, mid- or late September, the start, end, first half or last weeks of a month; 月初, 中旬, 上旬, 下旬,
  月底, 月末, 上半月); count_missing (the count as a digit or an English word; in Chinese before 篇);
  section_missing (the section by name, on word boundaries in English: introduction or intro, method(s) or
  methodology, experiment(s) or experimental, conclusion(s) or concluding; 引言/导言/绪论, 方法, 实验, 结论);
  note_missing and scaffold_missing (a write_note task asks for a note, "note" or 笔记, and says scaffold);
  note_in_text (a find_recent_papers, read_section, compare_papers, lookup_paper or chit_chat task asks for
  no note and says no scaffold); wrong_language (a Chinese text has Chinese characters, an English one none);
  line_break; too_long (more than MAX_CHARS[lang] characters).
- similarity, the masked 5-gram Jaccard of data.requests (mask_text, shingles, jaccard):
  near_train:<request id>:<j> (j >= TRAIN_VS_EVAL against a training request: the train split of
  data/synth/requests.jsonl, requests-m6.jsonl and requests-m7-train.jsonl); near_task:<id>:<j> (j >= NEAR_DUP
  against another task of the file, another draft or a dev task; both tasks of a pair are named);
  near_retired:<j> (j >= TRAIN_VS_EVAL against v1's end-to-end test tasks, read through
  eval.lock.RetiredSet, which returns the similarities alone and logs the read in
  data/eval/e2e-test-retired-reads.jsonl: one read per run, of every task text at once).

Prints one line per failing task, "<id>: <code> <code> ...", then "(set): <code>" lines in whole-file mode
(SET_REASONS: size:<tasks>/<records>, intent:<intent>:<have>/<want>, lang:<lang>:<have>/<want>), then
"OK: <n> tasks pass" or "FAIL: ..."; --json prints {"ok", "tasks", "failures": {id: [codes]}, "set": [codes],
"warnings": [...]} instead. Neither ever holds a task's text. Exit 0 when nothing fails; 1 when a task or the
set fails; 2 when the file cannot be read as JSON lines (or is missing), is test material (above), an input
of the checks is missing or malformed, or the retired set refuses the read (eval.lock.RetiredSet: not
retired, or not matching its lock).
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from research_agent.agent.types import HarnessError
from research_agent.arxiv import parse_atom
from research_agent.data.requests import NEAR_DUP, TRAIN_VS_EVAL, jaccard, mask_text, shingles
from research_agent.data.synth import corpus_recorded_on
from research_agent.eval.e2e import check_tasks, corpus_topics
from research_agent.eval.lock import (
    E2E_DEV_SET,
    E2E_TEST_ACCESS_LOG,
    E2E_TEST_RETIRED_READS,
    E2E_TEST_SET,
    E2E_TEST_SET_V2,
    JsonlError,
    LockError,
    RetiredSet,
    named_set,
    read_jsonl,
    refuse_test_sets,
)
from research_agent.net.fetch import Fetcher
from research_agent.normalize import normalize_arxiv_id, normalize_date, normalize_text
from research_agent.tools import schemas
from research_agent.tools.corpus_search import CorpusIndex, corpus_ids, load_papers
from research_agent.train.manifest import REPO

SLOTS = Path("data/eval/v2/e2e-slots-v2.jsonl")
TRAIN_FILES = (Path("data/synth/requests.jsonl"), Path("data/synth/requests-m6.jsonl"),
               Path("data/synth/requests-m7-train.jsonl"))
CORPUS_MANIFEST = Path("data/corpus/manifest.json")
REQUESTS_MANIFEST = Path("data/synth/requests-manifest.json")
CASSETTES = Path("data/cassettes")
WRITING = Path("data/eval/v2/e2e-writing.md")  # the writers' instructions, which list every reason code
SPLIT = "test"
SLACK = 2  # recorded hits beyond the count, as tests/eval/test_e2e.py's SLACK
MAX_CHARS = {"zh": 120, "en": 200}  # the dev tasks reach 85 and 127
DATE_KEYS = ("date_from", "date_to", "since")
NO_NOTE_INTENTS = ("find_recent_papers", "read_section", "compare_papers", "lookup_paper", "chit_chat")
DRAFT_GLOB = "tasks-*.jsonl"
_BATCH = re.compile(r"tasks-(\d+)\.jsonl")
_SAFE_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")  # an id the report may print (data.synth's ids, kept short)
MISSING = object()
# Every reason code a task can get (module docstring), and the set's (whole-file mode); a code may carry
# details after a colon. data/eval/v2/e2e-writing.md explains each one (tests/eval/test_check_e2e_tasks.py).
REASONS = ("malformed", "duplicate_id", "no_slot_record", "differs_from_slots", "not_test_split",
           "topic_outside_split", "id_outside_split", "id_not_in_metadata", "date_outside_window",
           "date_order", "too_few_hits", "id_not_in_slots", "slot_id_missing", "topic_missing",
           "topic_in_longer_phrase", "other_topic_in_text", "topic_in_text", "category_missing",
           "category_not_in_slots", "date_missing", "date_not_in_slots", "vague_date", "count_missing",
           "section_missing", "note_missing", "scaffold_missing", "note_in_text", "wrong_language",
           "line_break", "too_long", "near_train", "near_task", "near_retired", "missing_task",
           "not_in_batch")
SET_REASONS = ("size", "intent", "lang")

# arXiv ids as the contract tests find them, and ISO dates
_ID = re.compile(r"(?<![\d.])\d{4}\.\d{4,5}(?:v\d+)?(?![\d])")
_ISO = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
# English months by name or abbreviation ("May" is left out: no window month, and a common word)
_EN_MONTHS = {"january": 1, "february": 2, "march": 3, "april": 4, "june": 6, "july": 7, "august": 8,
              "september": 9, "october": 10, "november": 11, "december": 12, "jan": 1, "feb": 2, "mar": 3,
              "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sept": 9, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_MONTH = "|".join(sorted(_EN_MONTHS, key=len, reverse=True))
_ORDINAL = r"(?:st|nd|rd|th)?"
_EN_MONTH_DAY = re.compile(rf"\b({_MONTH})\.?\s+(\d{{1,2}}){_ORDINAL}\b(?:,?\s+(\d{{4}})\b)?", re.IGNORECASE)
_EN_DAY_MONTH = re.compile(rf"\b(\d{{1,2}}){_ORDINAL}\s+(?:of\s+)?({_MONTH})\b\.?(?:,?\s+(\d{{4}})\b)?",
                           re.IGNORECASE)
_EN_WHOLE_MONTH = re.compile(rf"\b({_MONTH})\b\.?(?:,?\s+(\d{{4}})\b)?", re.IGNORECASE)
_ZH_NUMERALS = "一二三四五六七八九十"
_ZH_MONTH_DAY = re.compile(r"(?:(\d{4})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")
_ZH_WHOLE_MONTH = re.compile(rf"(?:(\d{{4}})\s*年\s*)?(\d{{1,2}}|[{_ZH_NUMERALS}]{{1,3}})\s*月"
                             rf"(?!\s*[\d{_ZH_NUMERALS}])")
# a part of a month instead of a day: early, mid- or late September, the end (start, first half, last weeks)
# of July; 9 月初, 8 月中旬, 7 月底, 9 月上半月
_EN_VAGUE = re.compile(rf"\b(?:early|mid|late)[-\s]+(?:{_MONTH})\b|\b(?:beginning|start|middle|end|half|"
                       rf"week|weeks|days)\s+of\s+(?:{_MONTH})\b", re.IGNORECASE)
_ZH_VAGUE = re.compile(r"月\s*(?:初|底|末|上旬|中旬|下旬|[上下前后]半)")
_YEAR = re.compile(r"(?<![\d.])(19\d{2}|20\d{2})(?![\d.])")
_EN_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
               "nine": 9, "ten": 10}
_EN_COUNT = re.compile(r"\b(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten)\b", re.IGNORECASE)
_ZH_COUNT = re.compile(rf"(\d{{1,2}}|[两{_ZH_NUMERALS}])\s*篇")
_NOTE = re.compile(r"(?<![a-z])notes?(?![a-z])|笔记", re.IGNORECASE)
_SCAFFOLD = re.compile(r"scaffold", re.IGNORECASE)
_CJK = re.compile(r"[㐀-䶿一-鿿]")
_CATEGORY = {c: re.compile(rf"(?<![A-Za-z]){re.escape(c)}(?![A-Za-z])", re.IGNORECASE)
             for c in schemas.CATEGORIES}
# a section by name, on word boundaries in English ("introduces" names no introduction), in casefolded text
SECTION_WORDS = {
    "introduction": re.compile(r"(?<![a-z])intro(?:duction)?s?(?![a-z])|引言|导言|绪论"),
    "method": re.compile(r"(?<![a-z])method(?:s|ology|ologies)?(?![a-z])|方法"),
    "experiments": re.compile(r"(?<![a-z])experiment(?:s|al)?(?![a-z])|实验"),
    "conclusion": re.compile(r"(?<![a-z])conclu(?:sion|sions|ding)(?![a-z])|结论"),
    "all": re.compile(r"full text|whole paper|全文"),
}


def default_retired() -> RetiredSet:
    """v1's end-to-end test tasks, their access log and their reads log, under the repository root."""
    return RetiredSet(REPO / E2E_TEST_SET, REPO / E2E_TEST_ACCESS_LOG, REPO / E2E_TEST_RETIRED_READS)


# ---- inputs ---------------------------------------------------------------------------------------------


class _Shingles:
    """Masked shingle sets with an inverted index: the closest one to a text, exactly (data.requests'
    measure), ties going to the smaller id."""

    def __init__(self, items: Iterable[tuple[str, frozenset[str]]]) -> None:
        self.ids: list[str] = []
        self.sizes: list[int] = []
        self.postings: dict[str, list[int]] = {}
        for n, (item_id, grams) in enumerate(items):
            self.ids.append(item_id)
            self.sizes.append(len(grams))
            for gram in grams:
                self.postings.setdefault(gram, []).append(n)

    def closest(self, grams: frozenset[str]) -> tuple[float, str | None]:
        shared: Counter[int] = Counter()
        for gram in grams:
            shared.update(self.postings.get(gram, ()))
        best, who = 0.0, None
        for n, inter in shared.items():
            j = inter / (len(grams) + self.sizes[n] - inter)
            if j > best or (j == best and who is not None and self.ids[n] < who):
                best, who = j, self.ids[n]
        return best, who


def grams(text: str) -> frozenset[str]:
    return shingles(mask_text(text))


@dataclass
class Inputs:
    """What the checks read: the slot records, the training requests and dev tasks for the similarity checks,
    the recorded corpus, the test split, the window, the corpus topics' names and phrases, the retired set."""

    slots: dict[str, dict]
    train: _Shingles
    dev: list[tuple[str, frozenset[str]]]
    index: CorpusIndex
    corpus: set[str]
    split_topics: set[str]
    split_ids: set[str]
    window: tuple[str, str]
    today: str
    phrases: list[tuple[str, re.Pattern]]  # (topic id, matcher) of every phrase of every corpus topic
    names: list[tuple[str, str]]  # (topic id, its name without spaces, casefolded)
    retired: RetiredSet = field(default_factory=default_retired)
    topic_phrases: dict[str, tuple[str, ...]] = field(default_factory=dict)  # topic id -> phrases, folded


def _phrase_matcher(phrase: str) -> re.Pattern:
    """A phrase on word boundaries, a plural s or es allowed, in normalized casefolded text."""
    return re.compile(rf"(?<![a-z0-9]){re.escape(_fold(phrase))}(?:s|es)?(?![a-z0-9])")


def _fold(text: str) -> str:
    return normalize_text(text).casefold()


def load_inputs(*, slots: str | Path = REPO / SLOTS, train: Sequence[str | Path] | None = None,
                dev: str | Path = REPO / E2E_DEV_SET, corpus_manifest: str | Path = REPO / CORPUS_MANIFEST,
                requests_manifest: str | Path = REPO / REQUESTS_MANIFEST,
                cassettes: str | Path = REPO / CASSETTES, retired: RetiredSet | None = None) -> Inputs:
    """The inputs of the checks (module docstring); the paths default to the repository's files (train: the
    TRAIN_FILES)."""
    records = read_jsonl(slots)
    train = [REPO / path for path in TRAIN_FILES] if train is None else train
    requests = [r for path in train for r in read_jsonl(path) if r.get("split") == "train"]
    index = CorpusIndex(load_papers(Fetcher("replay", cassettes), corpus_ids(corpus_manifest), parse_atom))
    manifest = json.loads(Path(requests_manifest).read_text(encoding="utf-8"))
    topics = corpus_topics(corpus_manifest)
    corpus = json.loads(Path(corpus_manifest).read_text(encoding="utf-8"))
    split_topics = set(manifest["topics"][SPLIT])
    phrases = [(tid, _phrase_matcher(p)) for tid, topic in sorted(topics.items()) for p in topic.phrases]
    names = [(t["id"], re.sub(r"\s+", "", _fold(t["name"]))) for t in corpus["topics"] if t.get("name")]
    return Inputs(
        slots={r["id"]: r for r in records}, train=_Shingles((r["id"], grams(r["text"])) for r in requests),
        dev=[(t["id"], grams(t["text"])) for t in read_jsonl(dev)], index=index, corpus=set(index.ids()),
        split_topics=split_topics, split_ids={a for t in split_topics for a in topics[t].papers},
        window=tuple(manifest["window"]), today=corpus_recorded_on(corpus_manifest), phrases=phrases,
        names=names, retired=retired if retired is not None else default_retired(),
        topic_phrases={tid: tuple(_fold(p) for p in topic.phrases) for tid, topic in topics.items()})


# ---- the text -------------------------------------------------------------------------------------------


def named_ids(slots: dict) -> list[str]:
    if "arxiv_id" in slots:
        return [slots["arxiv_id"]]
    return list(slots.get("arxiv_ids", []))


def text_ids(text: str) -> set[str]:
    return {normalize_arxiv_id(found) or found for found in _ID.findall(normalize_text(text))}


def _zh_number(word: str) -> int | None:
    """A number written in digits or in Chinese numerals up to 19 (两 is 2); None for anything else."""
    if word.isdigit():
        return int(word)
    digits = {c: n for n, c in enumerate(_ZH_NUMERALS[:9], 1)} | {"两": 2}
    if word == "十":
        return 10
    if len(word) == 2 and word[0] == "十" and word[1] in digits:
        return 10 + digits[word[1]]
    return digits.get(word) if len(word) == 1 else None


def date_mentions(text: str) -> tuple[list[tuple[int | None, int, int | None]], list[int], str]:
    """The dates the text writes as (year or None, month, day or None for a whole month), in the forms the
    module docstring lists; the other years it mentions; and the text with ids and dates blanked out."""
    work = _ID.sub(" ", normalize_text(text))
    found: list[tuple[int | None, int, int | None]] = []
    for match in _ISO.finditer(work):
        found.append((int(match[1]), int(match[2]), int(match[3])))
    work = _ISO.sub(" ", work)
    for match in _ZH_MONTH_DAY.finditer(work):
        found.append((int(match[1]) if match[1] else None, int(match[2]), int(match[3])))
    work = _ZH_MONTH_DAY.sub(" ", work)
    for match in _EN_MONTH_DAY.finditer(work):
        found.append((int(match[3]) if match[3] else None, _EN_MONTHS[match[1].casefold()], int(match[2])))
    work = _EN_MONTH_DAY.sub(" ", work)
    for match in _EN_DAY_MONTH.finditer(work):
        found.append((int(match[3]) if match[3] else None, _EN_MONTHS[match[2].casefold()], int(match[1])))
    work = _EN_DAY_MONTH.sub(" ", work)
    for match in _EN_WHOLE_MONTH.finditer(work):
        found.append((int(match[2]) if match[2] else None, _EN_MONTHS[match[1].casefold()], None))
    work = _EN_WHOLE_MONTH.sub(" ", work)
    for match in _ZH_WHOLE_MONTH.finditer(work):
        month = _zh_number(match[2])
        if month is not None:
            found.append((int(match[1]) if match[1] else None, month, None))
    work = _ZH_WHOLE_MONTH.sub(" ", work)
    years = [int(y) for y in _YEAR.findall(work)]
    return found, years, work


def _covers(mention: tuple[int | None, int, int | None], key: str, iso: str) -> bool:
    year, month, day = mention
    want_year, want_month, want_day = int(iso[:4]), int(iso[5:7]), int(iso[8:10])
    if year not in (None, want_year) or month != want_month:
        return False
    if day is not None:
        return day == want_day
    return ((want_day == 1 and key in ("date_from", "since"))
            or (want_day == calendar.monthrange(want_year, want_month)[1] and key == "date_to"))


def _mentions(inputs: Inputs, folded: str) -> list[tuple[str, str]]:
    """(topic id, "phrase" or "name") for every topic phrase and Chinese name in folded text."""
    compact = re.sub(r"\s+", "", folded)
    found = [(tid, "phrase") for tid, matcher in inputs.phrases if matcher.search(folded)]
    found += [(tid, "name") for tid, name in inputs.names if name and name in compact]
    return found


def text_problems(task: dict, inputs: Inputs) -> list[str]:
    """The reason codes of the text checks (module docstring) on a well-formed task, in a fixed order."""
    text, slots, intent, lang = task["text"], task.get("slots") or {}, task["intent"], task.get("lang")
    folded = _fold(text)
    problems: list[str] = []
    named, written = set(named_ids(slots)), text_ids(text)
    if written - named:
        problems.append("id_not_in_slots")
    if named - written:
        problems.append("slot_id_missing")
    if isinstance(slots.get("topic"), str):
        topic = _fold(slots["topic"])
        if topic not in folded:
            problems.append("topic_missing")
        longer = [p for p in inputs.topic_phrases.get(slots.get("topic_id"), ()) if topic in p and p != topic]
        if any(_phrase_matcher(p).search(folded) for p in longer):
            problems.append("topic_in_longer_phrase")
        rest = folded.replace(topic, " ")
        if any(tid != slots.get("topic_id") or kind == "phrase" for tid, kind in _mentions(inputs, rest)):
            problems.append("other_topic_in_text")
    elif not named and _mentions(inputs, folded):
        problems.append("topic_in_text")
    mentioned = {c for c, pattern in _CATEGORY.items() if pattern.search(text)}
    wanted = set(slots.get("categories") or [])
    if wanted - mentioned:
        problems.append("category_missing")
    if mentioned - wanted:
        problems.append("category_not_in_slots")
    found, years, blanked = date_mentions(text)
    window_year = int(inputs.window[0][:4])
    dates = [(key, slots[key]) for key in DATE_KEYS
             if isinstance(slots.get(key), str) and normalize_date(slots[key])]
    problems += [f"date_missing:{key}" for key, iso in dates if not any(_covers(m, key, iso) for m in found)]
    if any(not any(_covers(m, key, iso) for key, iso in dates) for m in found):
        problems.append("date_not_in_slots")
    if any(y is not None and y != window_year for y, _, _ in found) or any(
            1990 <= y <= 2030 and y != window_year for y in years):
        problems.append("date_outside_window")
    if _EN_VAGUE.search(normalize_text(text)) or _ZH_VAGUE.search(normalize_text(text)):
        problems.append("vague_date")
    if "count" in slots:
        numbers = ({_zh_number(n) for n in _ZH_COUNT.findall(blanked)} if lang == "zh" else
                   {int(n) if n.isdigit() else _EN_NUMBERS[n.casefold()] for n in _EN_COUNT.findall(blanked)})
        if slots["count"] not in numbers:
            problems.append("count_missing")
    section = SECTION_WORDS.get(slots["section"]) if isinstance(slots.get("section"), str) else None
    if "section" in slots and (section is None or not section.search(folded)):
        problems.append("section_missing")
    if intent == "write_note":
        if not _NOTE.search(text):
            problems.append("note_missing")
        if not _SCAFFOLD.search(text):
            problems.append("scaffold_missing")
    elif intent in NO_NOTE_INTENTS and (_NOTE.search(text) or _SCAFFOLD.search(text)):
        problems.append("note_in_text")
    if (lang == "zh") != bool(_CJK.search(text)):
        problems.append("wrong_language")
    if "\n" in text or "\r" in text:
        problems.append("line_break")
    if len(text) > MAX_CHARS.get(lang, max(MAX_CHARS.values())):
        problems.append("too_long")
    return problems


# ---- the record -----------------------------------------------------------------------------------------


def slot_record_problems(task: dict, inputs: Inputs) -> list[str]:
    """no_slot_record, or differs_from_slots:<field> for every field but "text" that is not its slot
    record's."""
    record = inputs.slots.get(task["id"])
    if record is None:
        return ["no_slot_record"]
    return [f"differs_from_slots:{key}" for key in sorted((set(task) | set(record)) - {"text"})
            if task.get(key, MISSING) != record.get(key, MISSING)]


def record_problems(task: dict, inputs: Inputs) -> list[str]:
    """The reason codes of the slot-record and contract checks (module docstring) on a well-formed task."""
    problems = slot_record_problems(task, inputs)
    slots, about = task["slots"], task["expect"]["note_about"] or {}
    if task["split"] != SPLIT:
        problems.append("not_test_split")
    topics = [t for t in (slots.get("topic_id"), about.get("topic_id")) if t is not None]
    if any(t not in inputs.split_topics for t in topics):
        problems.append("topic_outside_split")
    ids = set(named_ids(slots)) | text_ids(task["text"])
    if "arxiv_id" in about:
        ids.add(about["arxiv_id"])
    if ids - inputs.split_ids:
        problems.append("id_outside_split")
    if any(a not in inputs.corpus or inputs.index.paper(a) is None for a in ids):
        problems.append("id_not_in_metadata")
    start, end = inputs.window
    dates = [slots[k] for k in DATE_KEYS if k in slots]
    dates += ["-".join(m) for m in _ISO.findall(normalize_text(task["text"]))]
    if any(not isinstance(d, str) or normalize_date(d) != d or not start <= d <= end <= inputs.today
           for d in dates):
        problems.append("date_outside_window")
    if isinstance(slots.get("date_from"), str) and isinstance(slots.get("date_to"), str) \
            and slots["date_from"] > slots["date_to"]:
        problems.append("date_order")
    if isinstance(slots.get("topic"), str) and task["intent"] != "search_notes":
        args = {"query": slots["topic"], **{k: slots[k] for k in ("date_from", "date_to", "categories")
                                            if k in slots}}
        count = slots.get("count", 1)
        need = (count if isinstance(count, int) else 1) + SLACK
        try:
            enough = len(inputs.index.search(args)) >= need
        except (TypeError, ValueError, AttributeError):
            enough = False
        if not enough:
            problems.append("too_few_hits")
    return problems


# ---- the check ------------------------------------------------------------------------------------------


@dataclass
class Report:
    """The outcome: reason codes per failing task id, in file order (then the missing tasks), the set's
    reason codes, and warnings (drafts that could not be read)."""

    tasks: int
    failures: dict[str, list[str]]
    set_problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures and not self.set_problems

    def lines(self) -> list[str]:
        out = [f"{task_id}: {' '.join(found)}" for task_id, found in self.failures.items()]
        out += [f"(set): {code}" for code in self.set_problems]
        if self.ok:
            out.append(f"OK: {self.tasks} tasks pass")
        else:
            out.append(f"FAIL: {len(self.failures)} task(s) fail"
                       + (f", {len(self.set_problems)} set problem(s)" if self.set_problems else ""))
        return out

    def to_json(self) -> dict:
        return {"ok": self.ok, "tasks": self.tasks, "failures": self.failures, "set": self.set_problems,
                "warnings": self.warnings}


def _texts(items: Iterable[object]) -> list[tuple[str, str]]:
    """(id, text) of the items that have both as strings."""
    return [(item["id"], item["text"]) for item in items if isinstance(item, dict)
            and isinstance(item.get("id"), str) and isinstance(item.get("text"), str)]


def check(tasks: Sequence[object], inputs: Inputs, *, partial: bool = False, others: Sequence[object] = (),
          batch: set[str] | None = None) -> Report:
    """Every check of the module docstring on tasks (the parsed lines of a task file, in order): others are
    the tasks of the other drafts (partial mode), batch the ids of the batch's slot records when known. Makes
    one retired read of every text (RetiredSet.similarities; its LockError propagates)."""
    failures: dict[str, list[str]] = {}
    keys: list[str] = []
    well_formed: list[tuple[str, dict]] = []
    other_ids = {task_id for task_id, _ in _texts(others)}
    seen: set[str] = set()
    for number, task in enumerate(tasks, 1):
        named = isinstance(task, dict) and isinstance(task.get("id"), str) and _SAFE_ID.fullmatch(task["id"])
        task_id = task["id"] if named else f"#{number}"
        keys.append(task_id)
        found = failures.setdefault(task_id, [])
        if task_id in seen or task_id in other_ids:
            found.append("duplicate_id")
        seen.add(task_id)
        try:
            (checked,) = check_tasks([task], task_id)
        except ValueError:
            found.append("malformed")
            continue
        try:
            found += record_problems(checked, inputs) + text_problems(checked, inputs)
        except (TypeError, ValueError, AttributeError, KeyError, IndexError):  # a slot of the wrong type
            found += slot_record_problems(checked, inputs) + ["malformed"]
        well_formed.append((task_id, checked))
    texts = [(task_id, task["text"]) for task_id, task in well_formed]
    _similarities(texts, _texts(others), inputs, failures)
    if partial and batch is not None:
        for task_id in sorted(batch - set(keys), key=_number):
            failures.setdefault(task_id, []).append("missing_task")
        for task_id in keys:
            if task_id not in batch:
                failures[task_id].append("not_in_batch")
    set_problems: list[str] = []
    if not partial:
        for task_id in sorted(set(inputs.slots) - set(keys), key=_number):
            failures.setdefault(task_id, []).append("missing_task")
        set_problems = _set_problems([task for _, task in well_formed], inputs)
    failures = {task_id: list(dict.fromkeys(found)) for task_id, found in failures.items() if found}
    return Report(len(keys), failures, set_problems)


def _similarities(texts: list[tuple[str, str]], others: list[tuple[str, str]], inputs: Inputs,
                  failures: dict[str, list[str]]) -> None:
    shingled = [(task_id, grams(text)) for task_id, text in texts]
    for task_id, mine in shingled:
        j, closest = inputs.train.closest(mine)
        if j >= TRAIN_VS_EVAL:
            failures[task_id].append(f"near_train:{closest}:{j:.2f}")
    references = [(task_id, grams(text)) for task_id, text in others] + inputs.dev
    for (position, (task_id, mine)) in enumerate(shingled):
        candidates = [(jaccard(mine, other), other_id) for n, (other_id, other) in enumerate(shingled)
                      if n != position]
        candidates += [(jaccard(mine, other), other_id) for other_id, other in references]
        if candidates:
            j, other_id = min(candidates, key=lambda c: (-c[0], c[1]))  # the closest; ties to the smaller id
            if j >= NEAR_DUP:
                failures[task_id].append(f"near_task:{other_id}:{j:.2f}")
    if texts:
        scores = inputs.retired.similarities([text for _, text in texts])
        for (task_id, _), j in zip(texts, scores, strict=True):
            if j >= TRAIN_VS_EVAL:
                failures[task_id].append(f"near_retired:{j:.2f}")


def _number(task_id: str) -> tuple[int, str]:
    tail = task_id.rsplit("-", 1)[-1]
    return (int(tail), task_id) if tail.isdigit() else (sys.maxsize, task_id)


def _set_problems(tasks: list[dict], inputs: Inputs) -> list[str]:
    """The set against the slot records: the size, the count per intent and per language."""
    records = list(inputs.slots.values())
    ids = {task["id"] for task in tasks}
    problems = []
    if len(ids) != len(records):
        problems.append(f"size:{len(ids)}/{len(records)}")
    unique = {task["id"]: task for task in tasks}.values()
    for key in ("intent", "lang"):
        have = Counter(task[key] for task in unique)
        want = Counter(r[key] for r in records)
        problems += [f"{key}:{value}:{have[value]}/{want[value]}" for value in sorted(set(have) | set(want))
                     if have[value] != want[value]]
    return problems


# ---- the script -----------------------------------------------------------------------------------------


def refuse_other_test_material(path: str | Path) -> None:
    """RefusedTestSet when path is test material other than the v2 end-to-end test set itself: a locked test
    set (by location or by its bytes) or a file of handwritten items (eval.lock.refuse_test_sets, under the
    repository root). The checker reads v1's end-to-end tasks only through RetiredSet."""
    if named_set(path, (E2E_TEST_SET_V2,), [REPO]) is None:
        refuse_test_sets(path, [REPO])


def _drafts(path: Path, folder: Path) -> tuple[list[dict], list[str]]:
    """The tasks of every other draft in folder, and a warning per draft that cannot be read (or is test
    material, which is not read)."""
    tasks, warnings = [], []
    for other in sorted(folder.glob(DRAFT_GLOB)):
        if other.resolve() == path.resolve():
            continue
        try:
            refuse_other_test_material(other)
            tasks += read_jsonl(other)
        except (OSError, JsonlError, UnicodeDecodeError, LockError) as exc:
            warnings.append(f"{other.name} cannot be read and is not compared: {exc}")
    return tasks, warnings


def _batch(path: Path) -> set[str] | None:
    """The ids of slots-<nn>.jsonl beside tasks-<nn>.jsonl, None when the file is no batch or has none."""
    match = _BATCH.fullmatch(path.name)
    slots = path.with_name(f"slots-{match[1]}.jsonl") if match else None
    if slots is None or not slots.is_file():
        return None
    return {r["id"] for r in read_jsonl(slots) if isinstance(r.get("id"), str)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check a file of v2 end-to-end test tasks: per failing task its id and reason codes, "
                    "never its text (docs/m7-计划.md WP-I 6; data/eval/v2/e2e-writing.md).")
    parser.add_argument("tasks", type=Path, help="the task file (JSONL): a batch, or the assembled set")
    parser.add_argument("--partial", action="store_true",
                        help="a batch: every per-task check and the between-task check against the other "
                             "drafts, no counts of the set")
    parser.add_argument("--slots", type=Path, default=REPO / SLOTS,
                        help=f"the slot records (default {SLOTS})")
    parser.add_argument("--drafts", type=Path, default=None,
                        help="the folder of the other drafts, tasks-*.jsonl (default: the batch's folder)")
    parser.add_argument("--json", action="store_true", help="print the report as one JSON object")
    return parser


def main(argv: list[str] | None = None, *, retired: RetiredSet | None = None) -> int:
    """The script; retired, when given, stands in for v1's retired tasks (the tests' fixture)."""
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    try:
        for path in (args.tasks, args.slots):
            refuse_other_test_material(path)
    except LockError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    try:
        tasks = read_jsonl(args.tasks)
    except (OSError, JsonlError, UnicodeDecodeError) as exc:
        print(f"FAIL: {args.tasks} cannot be read as JSON lines: {exc}", file=sys.stderr)
        return 2
    try:
        inputs = load_inputs(slots=args.slots, retired=retired)
        others, warnings, batch = [], [], None
        if args.partial:
            others, warnings = _drafts(args.tasks, args.drafts or args.tasks.parent)
            batch = _batch(args.tasks)
        report = check(tasks, inputs, partial=args.partial, others=others, batch=batch)
    except LockError as exc:
        print(f"FAIL: the retired read was refused: {exc}", file=sys.stderr)
        return 2
    except (OSError, JsonlError, KeyError, ValueError, HarnessError) as exc:
        print(f"FAIL: an input of the checks is missing or malformed: {exc}", file=sys.stderr)
        return 2
    report.warnings = warnings
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if args.json:
        print(json.dumps(report.to_json(), ensure_ascii=False, sort_keys=True))
    else:
        for line in report.lines():
            print(line)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())

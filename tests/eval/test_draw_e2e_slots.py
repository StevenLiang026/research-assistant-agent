"""scripts/draw_e2e_slots.py: the mechanical draw of the slot records of the v2 end-to-end test tasks
(docs/m7-计划.md WP-I 6), the expect block derived from intent and slots, and the ten writing batches.

The slot records hold every field of a task but its text, so they are no test material in the sense of
tests/conftest.py (no task text); these tests read the committed files and draw them again."""

from __future__ import annotations

import copy
import importlib.util
import itertools
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from research_agent.arxiv import extract_section, parse_atom
from research_agent.canonical import canonical_json
from research_agent.data.synth import load_seed_pool, seed_plan
from research_agent.eval.e2e import E2E_INTENTS, SCENARIO_OF, check_expect, corpus_topics
from research_agent.eval.lock import read_jsonl
from research_agent.net.fetch import Fetcher
from research_agent.normalize import normalize_text
from research_agent.tools import schemas
from research_agent.tools.adapters import fulltext_url
from research_agent.tools.corpus_search import CorpusIndex, corpus_ids, load_papers

REPO = Path(__file__).resolve().parents[2]
SLOTS = REPO / "data" / "eval" / "v2" / "e2e-slots-v2.jsonl"
DRAFTS = REPO / "data" / "eval" / "v2" / "e2e-drafts"
DEV = REPO / "data" / "eval" / "e2e-dev.jsonl"
CORPUS_MANIFEST = REPO / "data" / "corpus" / "manifest.json"
REQUESTS_MANIFEST = REPO / "data" / "synth" / "requests-manifest.json"
CASSETTES = REPO / "data" / "cassettes"
# The recorded full texts are not committed (.gitignore: data/cassettes/fulltext/). The draw and the checks of
# what a task can read need them, so those tests are skipped where they are absent (a fresh clone, the CI).
FULLTEXT = CASSETTES / "fulltext"
NO_FULLTEXT_REASON = "the recorded full texts (data/cassettes/fulltext/, not committed) are not here"
needs_fulltext = pytest.mark.skipif(not FULLTEXT.is_dir(), reason=NO_FULLTEXT_REASON)
# What the public copy leaves out (tests/conftest.py): the dev tasks, the v2 slot records and batches under
# data/eval/v2, the requests' manifest under data/synth and the recorded corpus
NEEDS_DEV = pytest.mark.needs_data("data/eval/e2e-dev.jsonl")
NEEDS_SLOTS = pytest.mark.needs_data("data/eval/v2")
NEEDS_WORLD = pytest.mark.needs_data("data/eval/v2", "data/synth", "data/cassettes")
TODAY = "2026-10-10"
SLACK = 2
DRAWN = {"find_recent_papers": 30, "read_section": 23, "write_note": 23, "compare_papers": 15,
         "lookup_paper": 15, "search_notes": 7, "list_notes": 7, "needs_clarification": 15, "chit_chat": 15}
ZH = {"find_recent_papers": 15, "read_section": 12, "write_note": 11, "compare_papers": 8, "lookup_paper": 7,
      "search_notes": 4, "list_notes": 3, "needs_clarification": 8, "chit_chat": 7}
RECORD_KEYS = {"id", "split", "family", "intent", "lang", "scenario", "slots", "expect"}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


draw_e2e_slots = load_script("draw_e2e_slots")


def dev_tasks() -> list[dict]:
    return [json.loads(line) for line in DEV.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def world():
    if not FULLTEXT.is_dir():
        pytest.skip(NO_FULLTEXT_REASON)
    return draw_e2e_slots.load_world(corpus_manifest=CORPUS_MANIFEST, requests_manifest=REQUESTS_MANIFEST,
                                     cassettes=CASSETTES)


@pytest.fixture(scope="module")
def records() -> list[dict]:
    return read_jsonl(SLOTS)


@pytest.fixture(scope="module")
def index() -> CorpusIndex:
    fetcher = Fetcher("replay", CASSETTES)
    return CorpusIndex(load_papers(fetcher, corpus_ids(CORPUS_MANIFEST), parse_atom))


def split_and_window():
    manifest = json.loads(REQUESTS_MANIFEST.read_text(encoding="utf-8"))
    return set(manifest["topics"]["test"]), manifest["window"]


# ---- the counts --------------------------------------------------------------------------------------


@NEEDS_DEV
def test_the_intent_counts_scale_the_dev_proportions_by_largest_remainder_ties_in_intent_order():
    dev = Counter(t["intent"] for t in dev_tasks())
    assert draw_e2e_slots.dev_intent_counts(DEV) == dev
    assert draw_e2e_slots.intent_counts(dev, 150) == DRAWN
    assert list(draw_e2e_slots.intent_counts(dev, 150)) == list(E2E_INTENTS)
    # four intents tie at a remainder of one half; the first two in E2E_INTENTS take the two units left
    assert draw_e2e_slots.intent_counts({"a": 1, "b": 1, "c": 1}, 2, order=("c", "b", "a")) == {
        "c": 1, "b": 1, "a": 0}
    assert sum(draw_e2e_slots.intent_counts(dev, 151).values()) == 151


def test_each_intent_is_half_chinese_and_the_odd_units_alternate_starting_with_chinese():
    langs = draw_e2e_slots.language_counts(DRAWN)
    assert {intent: split["zh"] for intent, split in langs.items()} == ZH
    assert all(split["zh"] + split["en"] == DRAWN[intent] for intent, split in langs.items())
    assert sum(split["zh"] for split in langs.values()) == sum(split["en"] for split in langs.values()) == 75
    odd = [intent for intent in E2E_INTENTS if DRAWN[intent] % 2]
    assert [langs[intent]["zh"] > langs[intent]["en"] for intent in odd] == [True, False] * (len(odd) // 2)


# ---- the expect block --------------------------------------------------------------------------------


@NEEDS_DEV
def test_the_expect_block_follows_from_intent_and_slots_for_every_dev_task():
    tasks = dev_tasks()
    assert len(tasks) == 20 and {t["intent"] for t in tasks} == set(E2E_INTENTS)
    for t in tasks:
        assert draw_e2e_slots.expect_for(t["intent"], t["slots"]) == t["expect"], t["id"]


def test_a_derived_expect_is_well_formed_for_every_intent_and_note_variant():
    for intent in E2E_INTENTS:
        assert check_expect(draw_e2e_slots.expect_for(intent, {"topic_id": "tokenization"})) == []
    assert draw_e2e_slots.expect_for("write_note", {"arxiv_id": "2609.05022", "topic_id": "tokenization"})[
        "note_about"] == {"arxiv_id": "2609.05022"}  # a named paper decides what the note is about


# ---- the slot records --------------------------------------------------------------------------------


def test_the_committed_slot_records_and_batches_are_the_draw(world, tmp_path):
    drawn = draw_e2e_slots.draw(world)
    assert SLOTS.read_bytes() == "".join(canonical_json(r) + "\n" for r in drawn).encode("utf-8")
    assert draw_e2e_slots.draw(world) == drawn  # deterministic
    assert draw_e2e_slots.draw(world, seed=7) != drawn  # and the seed matters
    out, batches = tmp_path / "slots.jsonl", tmp_path / "drafts"
    assert draw_e2e_slots.main(["--out", str(out), "--batches", str(batches)], world=world) == 0
    assert out.read_bytes() == SLOTS.read_bytes()
    for number in range(1, 11):
        name = f"slots-{number:02d}.jsonl"
        assert (batches / name).read_bytes() == (DRAFTS / name).read_bytes(), name
    assert draw_e2e_slots.main(["--out", str(out), "--batches", str(batches), "--check"], world=world) == 0


@NEEDS_SLOTS
def test_every_record_is_a_task_without_its_text(records):
    assert [r["id"] for r in records] == [f"e2e-test-{n}" for n in range(61, 211)]
    for r in records:
        assert set(r) == RECORD_KEYS, r["id"]
        assert r["split"] == "test" and r["family"] == f"e2e/{r['intent']}", r["id"]
        assert r["scenario"] == SCENARIO_OF.get(r["intent"], "normal"), r["id"]
        assert r["expect"] == draw_e2e_slots.expect_for(r["intent"], r["slots"]), r["id"]
        assert check_expect(r["expect"]) == [], r["id"]


@NEEDS_SLOTS
def test_the_records_have_the_drawn_counts_per_intent_and_language(records):
    assert len(records) == 150
    assert Counter(r["intent"] for r in records) == DRAWN
    assert Counter(r["intent"] for r in records if r["lang"] == "zh") == ZH
    assert Counter(r["lang"] for r in records) == {"zh": 75, "en": 75}


@NEEDS_SLOTS
@NEEDS_DEV
def test_the_slots_have_the_dev_tasks_shape_per_intent(records):
    shapes = {(t["intent"], tuple(sorted(t["slots"]))) for t in dev_tasks()}
    for r in records:
        assert (r["intent"], tuple(sorted(r["slots"]))) in shapes, r["id"]
        if "mode" in r["slots"]:
            assert r["slots"]["mode"] == "scaffold", r["id"]


@NEEDS_SLOTS
def test_no_two_records_share_their_slot_values(records):
    seen = Counter(canonical_json(r["slots"]) for r in records if r["slots"])
    assert [slots for slots, n in seen.items() if n > 1] == []
    pairs = [frozenset(r["slots"]["arxiv_ids"]) for r in records if "arxiv_ids" in r["slots"]]
    assert len(set(pairs)) == len(pairs)


@NEEDS_WORLD
def test_slots_name_only_the_test_splits_topics_and_papers_and_every_paper_is_recorded(records, index):
    split_topics, _ = split_and_window()
    topics = corpus_topics(CORPUS_MANIFEST)
    allowed = {a for t in split_topics for a in topics[t].papers}
    for r in records:
        slots = r["slots"]
        if "topic_id" in slots:
            assert slots["topic_id"] in split_topics, r["id"]
            assert slots["topic"] in topics[slots["topic_id"]].phrases, r["id"]
        named = [slots["arxiv_id"]] if "arxiv_id" in slots else list(slots.get("arxiv_ids", []))
        assert set(named) <= allowed and all(index.paper(a) is not None for a in named), r["id"]
        if "arxiv_ids" in slots:  # two papers of one topic, in id order
            assert len(named) == 2 and named == sorted(named), r["id"]
            assert len({t for t in split_topics for a in named if a in topics[t].papers}) == 1, r["id"]


@NEEDS_SLOTS
@pytest.mark.needs_data("data/synth")
def test_every_date_lies_in_the_recording_window(records):
    _, (start, end) = split_and_window()
    for r in records:
        dates = [r["slots"][k] for k in ("date_from", "date_to", "since") if k in r["slots"]]
        assert all(start <= d <= end <= TODAY for d in dates), r["id"]
        if "date_to" in r["slots"]:
            assert r["slots"]["date_from"] <= r["slots"]["date_to"], r["id"]


def on_topic(paper: dict, topic) -> bool:
    if paper["arxiv_id"] in topic.papers:
        return True
    text = normalize_text(f"{paper['title']} {paper['abstract']}").casefold()
    return any(normalize_text(phrase).casefold() in text for phrase in topic.phrases)


def topic_hits(r: dict, index: CorpusIndex) -> list[dict]:
    slots = r["slots"]
    args = {"query": slots["topic"],
            **{k: copy.deepcopy(slots[k]) for k in ("date_from", "date_to", "categories") if k in slots}}
    return index.search(args)


@NEEDS_SLOTS
@pytest.mark.needs_data("data/cassettes")
def test_every_topic_search_leaves_two_recorded_hits_beyond_its_count_on_topic_too(records, index):
    topics = corpus_topics(CORPUS_MANIFEST)
    for r in records:
        if "topic" not in r["slots"] or r["intent"] == "search_notes":
            continue
        hits = topic_hits(r, index)
        need = r["slots"].get("count", 1) + SLACK
        assert len(hits) >= need, r["id"]
        assert sum(on_topic(h, topics[r["slots"]["topic_id"]]) for h in hits) >= need, r["id"]


def readable(fetcher: Fetcher, arxiv_id: str, section: str) -> bool:
    page = fetcher.recorded(fulltext_url(arxiv_id))
    if page is None or page.status != 200:
        return False
    text = extract_section(page.text, section, schemas.DEFAULTS["fetch_fulltext"]["max_chars"])
    return bool(text and text.strip())


@needs_fulltext
def test_the_worlds_reading_of_a_section_is_fetch_fulltexts():
    """Spot checks of World.readable against extract_section itself: a paper without any section, one
    without an experiments section, one with every section, one without a cassette for its full text."""
    fetcher = Fetcher("replay", CASSETTES)
    world = draw_e2e_slots.load_world(corpus_manifest=CORPUS_MANIFEST, requests_manifest=REQUESTS_MANIFEST,
                                      cassettes=CASSETTES)
    cases = [("2608.26556", "introduction"), ("2609.33791", "experiments"), ("2609.33791", "method"),
             ("2607.23346", "conclusion")]
    assert [world.readable(a, s) for a, s in cases] == [readable(fetcher, a, s) for a, s in cases] == [
        False, False, True, True]
    assert world.status("2607.23346") == 200 and world.status("2609.99999") is None
    assert not world.readable("2609.99999", "introduction")


@NEEDS_WORLD
def test_every_section_a_task_can_lead_to_is_in_the_recorded_full_text(records, index, world):
    """A named paper's introduction (fetch_fulltext's default) and its named section can be read; for a
    section or note on a topic, every on-topic hit's section can be read or the paper has no HTML full text
    (NO_FULLTEXT, the abstract stands in), so a NOT_FOUND cannot fail a correct run. World.readable (spot
    checked above) keeps the full texts the draw parsed."""
    topics = corpus_topics(CORPUS_MANIFEST)
    for r in records:
        slots = r["slots"]
        section = slots.get("section", "introduction")
        if r["intent"] in ("read_section", "write_note", "compare_papers"):
            named = [slots["arxiv_id"]] if "arxiv_id" in slots else list(slots.get("arxiv_ids", []))
            for a in named:
                assert world.readable(a, "introduction") and world.readable(a, section), (r["id"], a)
            if "topic" in slots:
                for h in topic_hits(r, index):
                    if on_topic(h, topics[slots["topic_id"]]):
                        assert world.status(h["arxiv_id"]) == 404 or world.readable(h["arxiv_id"], section), (
                            r["id"], h["arxiv_id"])


@NEEDS_WORLD
def test_a_notes_since_date_leaves_seeded_notes_on_both_sides(records, index):
    pool = load_seed_pool(CORPUS_MANIFEST, REQUESTS_MANIFEST, index.paper)
    for r in records:
        if r["intent"] != "list_notes":
            continue
        created = [day for _, _, day in seed_plan(r, pool, index, TODAY)]
        since = r["slots"]["since"]
        assert any(day >= since for day in created) and any(day < since for day in created), r["id"]


# ---- the batches -------------------------------------------------------------------------------------


@NEEDS_SLOTS
def test_ten_single_language_batches_of_fifteen_spread_every_intent_evenly(records):
    by_id = {r["id"]: r for r in records}
    batches = [read_jsonl(DRAFTS / f"slots-{n:02d}.jsonl") for n in range(1, 11)]
    assert [len(b) for b in batches] == [15] * 10
    assert sorted(r["id"] for b in batches for r in b) == sorted(by_id)
    assert all(r == by_id[r["id"]] for b in batches for r in b)
    assert [{r["lang"] for r in b} for b in batches] == [{"zh"}] * 5 + [{"en"}] * 5
    for b in batches:
        assert [r["id"] for r in b] == sorted((r["id"] for r in b), key=lambda i: int(i.rsplit("-", 1)[1]))
    for lang, group in itertools.groupby(batches, key=lambda b: b[0]["lang"]):
        group = list(group)
        for intent in E2E_INTENTS:
            per_batch = [sum(r["intent"] == intent for r in b) for b in group]
            assert max(per_batch) - min(per_batch) <= 1, (lang, intent, per_batch)
    assert draw_e2e_slots.batches(records) == batches


def test_existing_slot_files_are_kept_unless_forced_and_check_reports_a_difference(world, tmp_path, capsys):
    out, batches = tmp_path / "slots.jsonl", tmp_path / "drafts"
    out.write_text("{}\n", encoding="utf-8")
    assert draw_e2e_slots.main(["--out", str(out), "--batches", str(batches)], world=world) == 2
    assert out.read_text(encoding="utf-8") == "{}\n" and not batches.exists()
    assert draw_e2e_slots.main(["--out", str(out), "--batches", str(batches), "--check"], world=world) == 1
    assert "differs" in capsys.readouterr().out
    assert draw_e2e_slots.main(["--out", str(out), "--batches", str(batches), "--force"], world=world) == 0
    assert out.read_bytes() == SLOTS.read_bytes()

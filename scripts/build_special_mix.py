"""Build M8's special-mix training set and the refusal development sets A and B (docs/m8-计划.md WP1, WP2)
from unused public records; research_agent.data.special_mix holds the sentence families, the prompts, the
constraint inference, the mutations, the case builders, the reasoning skeleton and the ACEBench ban, and this
script picks the records, draws the items, checks them and writes the files. No model and no API is called.

Revision 1 (docs/m8-预登记.md, 修订 1: FT5's step 441 met the data-revision trigger R1 on dev set A)
changes the training items alone, and draws sets A and B as before, byte for byte (--locked-dev-a and
--locked-dev-b check it): the training mix of the incorrect kind (special_mix.TRAIN_TWO_VALUES,
OPTIONAL_SHARE, DESCRIBED_SHARE) and of the irrelevant kind (PAIRING_SHARE pairings, near-domain partners
first, a record's own refusal of a canned request at most CANNED_SHARE), twins (TWIN_SHARE,
IRRELEVANT_TWIN_SHARE), a second rendering of every top-level-object and priority core (the other order),
Chinese-source re-renderings that fill the zh verbatim quota (FILL_KINDS), refusals in context
(CONTEXT_SHARE), the cell deck of styles and sentence shapes (CELL_KINDS), the thirty training families, the
extra placeholder styles and the check-first reasoning of special_mix. Training followers (module docstring:
Followers) are drawn right after the item they follow.

Records. research_agent.data.public_pool reads the five public sources back from the local Hugging Face cache
at the revisions of data/public/manifest.json and leaves out the source ids drawn into
data/raw/public/<name>.jsonl (FT3's public files). Left out too, by (source, source_id), each counted per
source in the manifest:
- used: the records of every file FT3 trained on that holds public records (runs/train/ft3/manifest.json,
  config.args.data: format-mix and base-replay-c5 besides the drawn files), the format-mix file and both
  self-replay prompt files with their kept outputs (EXCLUDE_FILES), and the M6 probe's source ids
  (data/eval/ace-probe-manifest.json, "source_ids");
- vocabulary: the records of --vocabulary-exclusions (VOCABULARY by default; required): a private exclusion
  list (not published) names, by (source, source_id) only, the records excluded for any of their texts
  (functions, requests, answers, gold values);
- apiname: a record whose functions, request or gold call hold ACEBench's placeholder ApiName as a substring
  of any case (special_mix.holds_apiname: deepexi's ApiNameKeyword too);
- acebench_name: a record whose functions or calls use the name of a function of ACEBench (case-folded; the
  clone's normal, special and agent data, both languages);
- acebench_text: a user text or a function (its JSON: names, parameters, descriptions) holding an ACEBench
  notation, sentence or name such as ApiName (special_mix.banned_markers);
- template_window: a user text or a function description sharing a 5-word or an 8-character window with a
  prompt template of the clone (special_mix.TemplateWindows);
- overlap: a user text or a function description sharing a 13-gram (research_agent.data.overlap.text_ngrams)
  with a BFCL or ACEBench question or with an ACEBench function description.

Draw. Each source's kept records are shuffled with random.Random(f"{seed}/special-mix/{source}"). Set B is
drawn first, then set A, then the training items; within a split the kinds go in DRAW_ORDER and the languages
zh, en. A kind's items get their constructions from a schedule (special_mix.INCORRECT_MIX, TWO_VALUES,
MISSING_MIX; half of the multi-turn items change a value and half switch the topic, PACKED_SHARE of them pack
the history into one message) and each item takes the next record a builder accepts, round robin over the
language's sources (EN_SOURCES; ZH_SOURCES, deepexi's irrelevant items pairing a request with another record's
unrelated functions); each split keeps its own place in the queues, so a record one split passes over stays
open to the next. EN_SHARE of the zh items of the special kinds take English records under the Chinese
prompt, and a zh item of any kind that its Chinese sources cannot give does too (counted per kind as
zh_from_en_sources, and as the Chinese sources' shortfall, zh_sources: near misses are scarce in deepexi,
whose integer parameters mostly state no bound). A construction no record gives falls back to the plainer
ones in turn (fallbacks; counted). For the kinds that cut a value (CUT_KINDS) a slot first takes only a
fluent cut and passes bare ones over to fill the quota later (draw_slot: construction, then sources, then
fluency). An irrelevant item no record's own refusal gives pairs an English request answered by a call with
another record's functions that share no content word with it (special_mix.shares_content_word; counted per
split as irrelevant_pairing). A development irrelevant item is such a pairing (or deepexi's) always: the
records' own refusals are a handful of templates (glaive_v2's 13,614 refusal records of the seed-1102 pools
ask 195 distinct requests, nearly all to book a flight or to order a pizza), which stay for training; and it
takes no request from a record able to lose a top-level object (counted as object_kept_for_training), since a
development item keeps its request's near-duplicates out of training and those records are the scarcest
(FIRST). A paired item (a switch, a request with another record's functions) takes its partner
from the unused records able to be one (Drawer.partners), trying them in turn until one gives an item that
passes the checks; a training pairing tries the first PAIRING_SCAN partners by the domain words their
functions share with the record's own, most first (special_mix.domain_words; near-domain partners, each still
unrelated to the request). A record serves one item at most, or, in training, one item and its follower.

Training (revision 1). The incorrect slots draw their constructions with TRAIN_TWO_VALUES two values,
OPTIONAL_SHARE a wrong value on an optional parameter and DESCRIBED_SHARE of the format slots a format only
the description implies, and take no record whose gold call leaves out a parameter its function requires
(special_mix.build_incorrect complete: that request is a priority case; the development slots draw as the
locked build did); a construction no record gives falls back to the same category and value count without
those choices, then to the same category with one value, then to an obvious one with its value count, then
to the plain one (fallbacks: the format share is a floor of the analysis). A development slot falls back as
the locked build's did. The irrelevant slots are PAIRING_SHARE pairings; a slot for a record's own refusal
takes no canned request (special_mix.canned_request: a flight, a pizza, a booking) once canned_cap(planned)
of the language's items are such refusals (counted as canned_capped), and falls back to an English pairing.
CONTEXT_SHARE of the refusal slots (special_mix.CONTEXT_KINDS; of the irrelevant kind, the slots for the
records' own refusals) put their core in context (special_mix.with_context) with a record of its source as
the earlier turn, whose function could not serve the latest request, packed into one message half of the
time: the partners are tried in turn, the first PAIRING_SCAN shortest first, until a framed item passes the
checks (Drawer.framed), and a core none of PARTNER_TRIES frames so is drawn plain (counted as
context_fallback). Every refusal answer that fills
placeholders (CELL_KINDS) takes the next cell of its language's deck (deck: special_mix.cell_deck with
CELL_ROUNDS rounds), which fixes its placeholder style and the shape of the family it answers with.

Followers. Right after a training slot's item is drawn come (Drawer.followers): its twin, from the same
record (extras: every vague incorrect item a valid twin with the original value; so many other incorrect
items a missing twin with the same parameter cut that TWIN_SHARE of the incorrect items have one, a valid twin
when that parameter is optional, cannot be cut or its cut does not read fluently (Drawer.twin, counted as
missing_twin->valid_verbatim and missing_twin/bare->valid_verbatim); IRRELEVANT_TWIN_SHARE of the missing
items but the object ones an irrelevant twin, the request as cut paired with another record's unrelated
functions), and for a priority core a second rendering under the other order; the top-level-object cores are
rendered again after every object slot (FIRST), so that a rendering takes only the room in a near-duplicate
cluster no core takes (each rendering tries up to RENDER_TRIES generators for another function list
rendering). A Chinese-source
zh verbatim core is rendered again while the kind's planned count is short (FILL_KINDS; these renderings count
toward it). Each follower passes the checks of any item, so a near-duplicate cluster still holds at most
TRAIN_NEAR_LIMIT training items: an item and its follower fill it. The variant of a follower says what it is:
twin_of (the kind it twins), render 2, fills. A follower that is not drawn is counted in the manifest
(revision.followers.failed) by what it was and what turned it away: twin/<kind>/no_core (the record gives no
such core), twin/irrelevant/no_partner, twin/<kind>/<check> and render/<kind>/<check> (the near-duplicate
rule, train_cap, or the item check that refused it).

Near-duplicate requests (special_mix.near_duplicate over special_mix.request_texts: the final user text and
each record's original request). No development item near-duplicates another of its set, of the other set or
any training item, and a training candidate goes when TRAIN_NEAR_LIMIT drawn training items near-duplicate it.
Templates (special_mix.same_template; the fix round's review, F1): a development candidate goes when
DEV_TEMPLATE_LIMIT drawn development items, of either set and any kind, share its template, and a training
candidate when a development item of its kind (template_kind) does. A record's request is checked before it
is built and the item's requests after it is dressed (counted by rule, test and kind as near_duplicates;
near_problems checks the whole draw again before anything is written). No two development items make the same
call either (same_call: one function with the same
arguments or a value in common, the same request in other words; counted as rule dev, test call). A value a
multi-turn item changes comes from the records of the same function (value_banks). A development item
reserves the names of the functions it shows
and calls: set A avoids set B's, the training items avoid both (records calling them are skipped, distractors
with them left out, and a training item whose text names one that looks like an identifier is dropped:
mentions). So that the names reserved withhold little training supply, no development item calls or shows a
function that more than DEV_NAME_LIMIT kept records show (frequent_names), nor a deepexi API outside a seeded
share of them (DEV_DEEPEXI_SHARE, closed_deepexi_names). The builder and the dressing of an item use
random.Random(f"{seed}/special-mix/{split}/{kind}/{lang}/{source}/{source_id}").

Checks. An item is kept only when its prompt is at most MAX_PROMPT_TOKENS (special_mix.approx_tokens), no text
of it (prompt, answer, reasoning) holds an ACEBench marker or template window, no function it shows has an
ACEBench name, its user texts share no 13-gram with the benchmarks, a training item's reasoning is the
check-first skeleton (special_mix.reasoning_problems: reasoning_form, or reasoning_long beyond
special_mix.MAX_REASONING_TOKENS; a development item has none), and its sentence is not filled with a value or
name written like a placeholder of its prompt
(special_mix.reads_as_placeholder); the others are counted by reason, per attempt (rejected_attempts: a paired
item counts each partner it tried; a training record whose request names a development function counts once,
before any partner).
Every training record must pass special_mix.check_training_record, every development item
special_mix.check_dev_item and every development set special_mix.dev_set_problems; the reasoning sidecar must
load in research_agent.data.reasoning.ReasoningIndex and bind the trained turn of every record.

Written (UTF-8 without escapes, LF):
- --out: the training records, by kind (special_mix.KINDS) and language, one JSON object per line;
- --reasoning-out: the sidecar (research_agent.data.reasoning.write_sidecar), one row of source "template" per
  record: id <out stem>:<index>, k the trained turn, the answer's hash, model special_mix.TEMPLATE_MODEL,
  cache_key the SHA-256 of the row's identity, context_equal true; keep it out of a folder train.py reads as
  data;
- --dev-a, --dev-b: the development sets, numbered sdev-<split>-<nnnn> in draw order, each with its lock
  <path>.sha256 (research_agent.eval.lock.write_lock: a hash check; development files, not test sets);
- --manifest: the counts per split, kind, language and source against the plan, the deviations, the Chinese
  sources' shortfall per kind, the planned and the drawn constructions and the fallbacks, the rejections, the
  near-duplicates turned away, the English pairings, the cuts per split and kind with their fluent share,
  the gold calls the kept records hold by problem (gold_call_census), the names the development sets reserved
  and avoided, every exclusion count, the vocabulary exclusion file with its SHA-256 and records, the
  benchmark files and the inputs with their SHA-256, the checks, and every output's path, records and
  SHA-256; revision: revision 1's followers, the incorrect and irrelevant mix, the top-level-object items and
  the cells dealt per language; dev_locks: per --locked-dev-* lock given, the lock, the drawn set's SHA-256
  and that it matches.
Nothing is written unless every check passes; existing outputs are replaced only with --force. A lock does not
tell a stale build (round 1's outputs, built before the vocabulary exclusion file, still pass theirs): a
run that reads the outputs first asks stale_outputs whether their manifest records the exclusion file as it
is now.

    .venv/Scripts/python.exe -B -X utf8 scripts/build_special_mix.py [--seed 1102] [--force]
        [--out data/raw/ft5/special-mix.jsonl] [--reasoning-out data/raw/ft5/reasoning-special-mix.jsonl]
        [--manifest data/raw/ft5/special-mix-manifest.json] [--dev-a data/eval/special-dev-a.jsonl]
        [--dev-b data/eval/special-dev-b.jsonl] [--acebench-dir runs/third_party/ACEBench] [--bfcl-dir DIR]
        [--vocabulary-exclusions data/public/vocabulary-excluded.json] [--accept-shortfall TEXT ...]
        [--locked-dev-a data/eval/special-dev-a.jsonl.sha256]
        [--locked-dev-b data/eval/special-dev-b.jsonl.sha256]

--locked-dev-a, --locked-dev-b: a lock (eval.lock's "<sha256>  <file name>", of a set of the same file name as
--dev-a, --dev-b) that the drawn set must match byte for byte; nothing is written otherwise. Revision 1 is
built with the locks of the sets in use, which it may not change.

Shortfalls (shortfalls()): a training kind short of its planned count by more than MAX_SHORT, a training
language with fewer top-level-object items than its schedule plans, a development set short of its plan. The
M8 preregistration accepts the final build's shortfalls one by one (docs/m8-预登记.md: the supply the rules
let through is used as it is): each --accept-shortfall gives one in shortfalls()' words, and the build is
written only when its shortfalls are exactly those, recorded as shortfalls_accepted in the manifest.

Exit 0 written; 2 usage (an output exists without --force, the benchmarks are missing, the vocabulary
exclusion file is missing or not of its form, a --locked-dev-* lock is not one of a file of its set's name);
3 the data do not meet the rules (shortfalls other than the accepted ones, or an accepted one that is not
there; a record, item or set that fails its check, a near-duplicate the draw should have turned away, a drawn
development set that does not match the lock given for it).
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import re
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

from research_agent.canonical import canonical_json
from research_agent.data import public_pool
from research_agent.data import special_mix as sm
from research_agent.data.overlap import BenchmarkIndex, BenchmarkMissing, text_ngrams
from research_agent.data.public import SOURCES
from research_agent.data.reasoning import TEMPLATE, ReasoningIndex, sidecar_row, turns, write_sidecar
from research_agent.eval.lock import lock_path, write_lock

SEED = 1102
OUT = Path("data/raw/ft5/special-mix.jsonl")
REASONING = Path("data/raw/ft5/reasoning-special-mix.jsonl")
MANIFEST = Path("data/raw/ft5/special-mix-manifest.json")
DEV = {"a": Path("data/eval/special-dev-a.jsonl"), "b": Path("data/eval/special-dev-b.jsonl")}
FT3_MANIFEST = Path("runs/train/ft3/manifest.json")
PROBE_MANIFEST = Path("data/eval/ace-probe-manifest.json")
EXCLUDE_FILES = (Path("data/raw/public/format-mix.jsonl"), Path("data/raw/public/base-replay.prompts.jsonl"),
                 Path("data/raw/public/base-replay.jsonl"), Path("data/raw/ft3/base-replay-c5.prompts.jsonl"),
                 Path("data/raw/ft3/base-replay-c5.jsonl"))
ACEBENCH = Path("runs/third_party/ACEBench")
BFCL_VENV = Path(".venv-bfcl")
BFCL_LAYOUTS = ("Lib/site-packages/bfcl_eval/data", "lib/python3*/site-packages/bfcl_eval/data")
EN_SOURCES = ("toolace", "glaive_v2", "hermes")
ZH_SOURCES = ("deepexi", "glaive_zh")
EN_SHARE = 0.3  # of the zh items of the special kinds: English records under the Chinese prompt
DRAW_ORDER = ("irrelevant", "valid_nearmiss", "valid_verbatim", "priority", "incorrect", "missing", "ask",
              "valid_multiturn")  # the kinds whose records are scarcest first
SPLIT_ORDER = ("b", "a", "train")
MAX_PROMPT_TOKENS = 3500  # special_mix.approx_tokens of the prompt; FT3 trained at 5,120 tokens
MAX_SHORT = 0.10  # a training kind may fall short of its planned count by this share (reported)
PARTNER_TRIES = 40  # records tried as the partner of a paired item
BANK_LIMIT = 200  # string values kept per parameter for value changes
# A development item never shows a function that more kept records than this show: every name it shows is
# withheld from training with all the records that call it, and a name such as glaive_v2's calculate_area
# (shown by 1,972 kept records, and called by 806 of the 950 English ones able to lose a top-level object;
# the seed-1102 build of 2026-11-02) would starve a construction. Every deepexi API stays below it (at most
# 388 kept records), so the Chinese development items keep their source.
DEV_NAME_LIMIT = 500
# deepexi shows 702 APIs, each in 98 to 388 kept records, and each record shows 3 to 6 of them drawn across
# products, so every deepexi API a development item shows withholds dozens of Chinese records from training.
# Unbounded, the development sets showed 360 of them, and 156 Chinese training items had to take English
# records instead (missing 31, multi-turn 49, near miss 76; the seed-1102 build of 2026-11-02). A development
# item therefore shows only the deepexi APIs of a seeded share of them (closed_deepexi_names), in percent; a
# record's distractors outside it are left out, so a development prompt shows fewer functions than a training
# one (ACEBench's special items show one function more often than any other number).
DEV_DEEPEXI_SHARE = 25
# The records the private exclusion list excludes (module docstring: Records), by (source, source_id): a file
# of ids only, kept out of the public repository.
VOCABULARY = Path("data/public/vocabulary-excluded.json")
SCREEN_REASONS = ("used", "vocabulary", "apiname", "acebench_name", "acebench_text", "template_window",
                  "overlap")
TRAIN_NEAR_LIMIT = 2  # a training candidate goes when this many drawn training items near-duplicate it
# A development candidate goes when this many drawn development items (A and B) share its template
# (special_mix.same_template; the fix round's review, F1: 35 of B's 50 irrelevant items ordered a pizza,
# booked a flight or a table, under D1's near-duplicate threshold)
DEV_TEMPLATE_LIMIT = 2
TEMPLATE_KINDS = {"ask": "missing"}  # the kind whose template an item holds: an ask item is a missing request
CUT_KINDS = ("missing", "priority", "ask")  # the kinds whose request loses a value: fluent cuts first
# Constructions drawn before every other slot of their split: a top-level object is the scarcest (few records
# hold one, and their templated requests fall into few near-duplicate clusters, each giving TRAIN_NEAR_LIMIT
# training items of any kind; drawn in DRAW_ORDER, the earlier kinds fill them first).
FIRST = {"missing": "object"}
# Revision 1 (docs/m8-预登记.md, 修订 1), the training items alone (module docstring: Training, Followers)
PAIRING_SHARE = 0.7  # of the training irrelevant slots, a request paired with another record's functions
CANNED_SHARE = 0.15  # of the training irrelevant items, at most, a record's own refusal of a canned request
# of the training incorrect items, with a twin (every vague one a valid twin): 48 % scheduled, so that about
# the analysis's 40 % have one after the twins that find no record or fail a check (a quarter of the others)
TWIN_SHARE = 0.48
IRRELEVANT_TWIN_SHARE = 0.15  # of the training missing items, the object ones aside, an irrelevant twin
# of the training refusal slots, in context (special_mix.with_context): 17 %, since a third of them find no
# partner whose framed item passes the checks, and followers take none, so that about a tenth of the refusal
# items are in context
CONTEXT_SHARE = 0.17
PACKED_CONTEXT = 0.5  # of the items in context, the earlier turns packed into one message
PAIRING_SCAN = 8  # partners of a training pairing ranked by the domain words their functions share, at first
CELL_KINDS = ("incorrect", "missing", "priority")  # the kinds whose answer fills placeholders: dealt a cell
CELL_ROUNDS = 60  # rounds of a language's deck (special_mix.cell_deck), more than any draw takes
RENDER_TRIES = 6  # generators a second rendering tries for another function list rendering
FILL_KINDS = (("valid_verbatim", "zh"),)  # kinds a Chinese-source core's second rendering may fill


# ---- inputs ------------------------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _public_ids(rows: list[dict]) -> set[tuple[str, str]]:
    return {(r["source"], str(r["source_id"])) for r in rows
            if isinstance(r, dict) and isinstance(r.get("source"), str) and r.get("source_id") is not None}


def used_records(ft3_manifest: Path, files: list[Path], probe_manifest: Path) -> tuple[set, dict]:
    """The (source, source_id) of the public records already used (module docstring: used) and an account
    of the files read: FT3's data files, the exclusion files and the probe's ids."""
    ids: set[tuple[str, str]] = set()
    account: dict = {"ft3_manifest": {"path": ft3_manifest.as_posix(), "present": ft3_manifest.is_file()},
                     "ft3_files": [], "files": [], "probe": {"path": probe_manifest.as_posix(),
                                                             "present": probe_manifest.is_file(), "ids": 0}}
    if ft3_manifest.is_file():
        account["ft3_manifest"]["sha256"] = _sha256(ft3_manifest)
        config = json.loads(ft3_manifest.read_text(encoding="utf-8"))
        for name in config["config"]["args"]["data"]:
            path = Path(name)
            found = _public_ids(_jsonl(path))
            ids |= found
            account["ft3_files"].append({"path": path.as_posix(), "sha256": _sha256(path),
                                         "records": len(found)})
    for path in files:
        entry = {"path": path.as_posix(), "present": path.is_file()}
        if path.is_file():
            found = _public_ids(_jsonl(path))
            ids |= found
            entry.update(sha256=_sha256(path), records=len(found))
        account["files"].append(entry)
    if probe_manifest.is_file():
        listed = json.loads(probe_manifest.read_text(encoding="utf-8")).get("source_ids") or {}
        found = {(source, str(i)) for source, values in listed.items() for i in values or []}
        ids |= found
        account["probe"].update(sha256=_sha256(probe_manifest), ids=len(found))
    account["total"] = len(ids)
    return ids, account


def load_pools(manifest: str | Path, raw_dir: str | Path) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """The unused records of every source from the Hugging Face cache, and their accounts: the revision, the
    data files read with their SHA-256, and the drawn file (data/raw/public/<name>.jsonl) with its SHA-256."""
    entries = public_pool.load_manifest(manifest)["sources"]
    pools, accounts = {}, {}
    for name in SOURCES:
        entry = entries[name]
        pools[name], accounts[name] = public_pool.load_unused(name, entry, raw_dir)
        drawn = Path(raw_dir) / f"{name}.jsonl"
        files = [(item["file"], public_pool.cached_file(entry["repo"], item["file"], entry["revision"]))
                 for item in entry["per_file"]]
        accounts[name].update(revision=entry["revision"],
                              files=[{"file": file, "sha256": _sha256(path)} for file, path in files],
                              drawn_file={"path": drawn.as_posix(), "sha256": _sha256(drawn)})
    return pools, accounts


class SimpleIndex(NamedTuple):
    """Benchmark grams and the files they came from (what the draw needs of a BenchmarkIndex)."""

    grams: object
    files: list


@dataclass
class Benchmarks:
    """What the screen and the checks compare with: the BFCL and ACEBench question grams (index), ACEBench's
    function names (case-folded) and the grams of its function descriptions, the windows of its prompt
    templates, and the ACEBench files read (path, SHA-256)."""

    index: object
    names: frozenset
    grams: frozenset
    windows: sm.TemplateWindows
    files: list = field(default_factory=list)


def _bfcl_dir(bfcl_dir: Path | None) -> Path:
    if bfcl_dir is not None:
        return bfcl_dir
    found = sorted({p for pattern in BFCL_LAYOUTS for p in BFCL_VENV.glob(pattern) if p.is_dir()})
    if len(found) > 1:
        raise BenchmarkMissing(f"several bfcl_eval data folders under {BFCL_VENV}; pass --bfcl-dir")
    return found[0] if found else BFCL_VENV / BFCL_LAYOUTS[0]


def load_benchmarks(acebench: Path, bfcl_dir: Path | None) -> Benchmarks:
    """The benchmarks (Benchmarks); BenchmarkMissing when a folder or a file is absent."""
    prompts = [acebench / "model_inference" / "prompt_en.py", acebench / "model_inference" / "prompt_zh.py"]
    if not all(p.is_file() for p in prompts):
        raise BenchmarkMissing(f"no prompt templates under {acebench / 'model_inference'}")
    data = acebench / "data_all"
    index = BenchmarkIndex.build(_bfcl_dir(bfcl_dir), data)
    functions = sm.acebench_functions(data)
    names = frozenset(fn["name"].casefold() for fn in functions if isinstance(fn.get("name"), str))
    grams = frozenset(g for fn in functions for text in sm.descriptions(fn) for g in text_ngrams(text))
    files = [{"path": p.as_posix(), "sha256": _sha256(p)} for p in [*prompts, *sm.acebench_files(data)]]
    return Benchmarks(index, names, grams, sm.TemplateWindows.from_files(prompts), files)


def overlaps(text: str, bench: Benchmarks) -> bool:
    """Whether a text shares a 13-gram with a benchmark question or an ACEBench function description."""
    return any(g in bench.index.grams or g in bench.grams for g in text_ngrams(text))


def load_vocabulary(path: Path) -> tuple[frozenset, dict]:
    """The (source, source_id) pairs of VOCABULARY's file and an account of it (path, SHA-256, records);
    ValueError unless the file holds exactly a description and a list of [source, source_id] string pairs."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"description", "records"} or \
            not isinstance(data["description"], str) or not isinstance(data["records"], list):
        raise ValueError("the file must hold exactly a description and the records")
    pairs = set()
    for pair in data["records"]:
        if not (isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, str) and x for x in pair)):
            raise ValueError("each record is a [source, source_id] pair of strings")
        pairs.add((pair[0], pair[1]))
    return frozenset(pairs), {"path": repo_relative(path), "sha256": _sha256(path), "records": len(pairs)}


def stale_outputs(manifest: Path, vocabulary: Path = VOCABULARY) -> str | None:
    """Why the outputs a manifest of this script describes may not be used, None when they are a build of
    this code: the manifest is absent, or it does not record the vocabulary exclusion file as it is now (a
    build before the file records none). The fix round's review, F2: round 1's outputs stayed in data/ with
    locks that still match, so a lock alone does not tell a stale build; a run that reads the outputs checks
    this first. A stale build is rebuilt, or moved out of data/ with the owner's consent."""
    if not manifest.is_file():
        return f"{manifest.as_posix()} is absent"
    account = json.loads(manifest.read_text(encoding="utf-8")).get("vocabulary_exclusions") or {}
    if account.get("sha256") != _sha256(vocabulary):
        return (f"{manifest.as_posix()} does not record {vocabulary.as_posix()} as it is now: its outputs "
                "predate this code's build")
    return None


def _screen_reason(sample: dict, used: set, bench: Benchmarks, verdicts: dict,
                   vocabulary: frozenset = frozenset()) -> str | None:
    """Why a record is left out (module docstring: Records), the first reason in SCREEN_REASONS' order; None
    when it is kept. verdicts caches (window, overlap) per text: records share function definitions."""
    key = (sample["source"], str(sample["source_id"]))
    if key in used:
        return "used"
    if key in vocabulary:
        return "vocabulary"
    users = [m["content"] for m in sample["messages"] if m["role"] == "user"]
    rendered = [json.dumps(t["function"], ensure_ascii=False) for t in sample.get("tools") or []]
    calls = [c["function"] for m in sample["messages"] for c in m.get("tool_calls") or []]
    arguments = [text for call in calls for text in (call["name"], call["arguments"])]  # the gold calls whole
    if any(sm.holds_apiname(text) for text in users + rendered + arguments):
        return "apiname"
    names = {t["function"]["name"].casefold() for t in sample.get("tools") or []}
    names |= {c["function"]["name"].casefold() for m in sample["messages"] for c in m.get("tool_calls") or []}
    if names & bench.names:
        return "acebench_name"
    described = [d for t in sample.get("tools") or [] for d in sm.descriptions(t["function"])]
    if any(sm.banned_markers(text) for text in users + rendered):
        return "acebench_text"
    found = []
    for text in users + described:
        if text not in verdicts:
            verdicts[text] = (bool(bench.windows.violations(text)), overlaps(text, bench))
        found.append(verdicts[text])
    if any(window for window, _ in found):
        return "template_window"
    if any(overlap for _, overlap in found):
        return "overlap"
    return None


def screen(pools: dict[str, list[dict]], used: set, bench: Benchmarks,
           vocabulary: frozenset = frozenset()) -> tuple[dict, dict]:
    """The records the draw may use and an account per source of the ones left out (module docstring)."""
    kept: dict[str, list[dict]] = {}
    account: dict[str, dict] = {}
    verdicts: dict[str, tuple[bool, bool]] = {}
    for name, samples in pools.items():
        counts = Counter()
        kept[name] = []
        for sample in samples:
            reason = _screen_reason(sample, used, bench, verdicts, vocabulary)
            counts[reason or "kept"] += 1
            if reason is None:
                kept[name].append(sample)
        account[name] = {"candidates": len(samples), **{r: counts[r] for r in SCREEN_REASONS},
                         "kept": counts["kept"]}
    return kept, account


def frequent_names(pools: dict[str, list[dict]], limit: int) -> frozenset[str]:
    """The function names (case-folded) that more than limit records of the pools show, over all sources:
    the names the development sets avoid (DEV_NAME_LIMIT)."""
    shown: Counter = Counter()
    for samples in pools.values():
        for sample in samples:
            shown.update({t["function"]["name"].casefold() for t in sample.get("tools") or []})
    return frozenset(name for name, count in shown.items() if count > limit)


def closed_deepexi_names(pools: dict[str, list[dict]], seed: int, share: int) -> frozenset[str]:
    """The deepexi API names (case-folded) the development sets may not show: those outside the seeded share
    (DEV_DEEPEXI_SHARE), a name being open when the SHA-256 of f"{seed}/dev-deepexi/{name}", read as a
    number, is below share modulo 100."""
    names = {t["function"]["name"].casefold() for sample in pools.get("deepexi", [])
             for t in sample.get("tools") or []}
    return frozenset(name for name in names if int(hashlib.sha256(
        f"{seed}/dev-deepexi/{name}".encode()).hexdigest(), 16) % 100 >= share)


def value_banks(pools: dict[str, list[dict]]) -> dict[str, dict[str, dict[str, list[str]]]]:
    """Per source and function (its name case-folded), the string values its one-call records give each
    parameter (sorted, at most BANK_LIMIT): the values a multi-turn item may change a value to. Keyed by the
    function, so that a new value is of the kind the old one was (hand check, round 1: a signature's name
    changed to "Eiffel Tower" from another function's name parameter). A record gives values only when its
    request gives its whole gold call (special_mix.call_problem finds nothing), and a value only when it is
    no label, names no parameter, stands in for nothing and points at nothing (special_mix.bank_value_ok)."""
    banks: dict[str, dict[str, dict[str, set]]] = {}
    for name, samples in pools.items():
        bank = banks.setdefault(name, {})
        for sample in samples:
            step = sm.first_step(sample)
            if step is None or len(step[1]) != 1 or not isinstance(step[1][0][1], dict):
                continue
            function, arguments = step[1][0]
            schema = sm._schema(sample, function)
            if sm.call_problem(step[0], step[1], [schema]) is not None:
                continue
            for key, value in arguments.items():
                if isinstance(value, str) and 2 <= len(value) <= 40 and value.strip() == value \
                        and sm.bank_value_ok(value, step[0], sm._described(schema, key), key):
                    bank.setdefault(function.casefold(), {}).setdefault(key, set()).add(value)
    return {name: {function: {key: sorted(values)[:BANK_LIMIT] for key, values in sorted(params.items())}
                   for function, params in sorted(bank.items())} for name, bank in banks.items()}


def call_keys(item: sm.Item) -> list[tuple[str, frozenset, str]]:
    """(the name case-folded, the values as special_mix._value_texts writes them, the arguments as canonical
    JSON) of each expected call of an item; [] for a refusal item."""
    if item.expected["type"] != "call":
        return []
    return [(call["name"].casefold(), frozenset(sm._value_texts(sm._leaves(call["arguments"]))),
             canonical_json(call["arguments"])) for call in item.expected["calls"]]


def same_call(a: list[tuple], b: list[tuple]) -> bool:
    """Whether two items' calls (call_keys) call one function with the same arguments or a value in common:
    two development items that ask the same thing in other words (hand check, round 1:
    UpdateFolder(FolderId="123456", FolderName="新文件夹") twice in set B)."""
    return any(x[0] == y[0] and (x[2] == y[2] or x[1] & y[1]) for x in a for y in b)


def gold_call_census(pools: dict[str, list[dict]]) -> dict[str, dict]:
    """Per source, the kept records whose first step is calls, by what special_mix.call_problem finds of them
    (the first problem; ok when none): the records the builders turn away for a gold call their request does
    not fully give (D3)."""
    out = {}
    for name, samples in pools.items():
        counts: Counter = Counter()
        for sample in samples:
            step = sm.first_step(sample)
            if step is None:
                continue
            user, calls = step
            if not all(isinstance(arguments, dict) for _, arguments in calls):
                counts["not_an_object"] += 1
                continue
            problem = sm.call_problem(user, calls, [sm._schema(sample, n) for n, _ in calls])
            counts[problem or "ok"] += 1
        out[name] = {"records": sum(counts.values()), **dict(sorted(counts.items()))}
    return out


# ---- the draw ---------------------------------------------------------------------------------------------


def scaled(count: int, scale: float) -> int:
    """A planned count scaled (tests and dry runs): at least 1."""
    return count if scale == 1 else max(1, round(count * scale))


def _shares(n: int, mix: tuple[tuple[str, float], ...]) -> list[str]:
    """n labels split by the mix's shares, the largest remainders rounded up, in the mix's order."""
    exact = [(label, n * share) for label, share in mix]
    counts = {label: int(value) for label, value in exact}
    rest = n - sum(counts.values())
    order = [label for label, _ in mix]
    for label, _ in sorted(exact, key=lambda lv: (-(lv[1] - int(lv[1])), order.index(lv[0]))):
        if rest <= 0:
            break
        counts[label] += 1
        rest -= 1
    return [label for label, _ in mix for _ in range(counts[label])]


def schedule(kind: str, n: int, rng: random.Random, *, split: str) -> list[dict]:
    """The construction of each of a kind's n items, in a drawn order (module docstring: Draw). The training
    split's (revision 1, module docstring: Training) draw more after the development sets' draws: the
    incorrect kind TRAIN_TWO_VALUES two values, OPTIONAL_SHARE optional and DESCRIBED_SHARE of its format
    items described, the irrelevant kind PAIRING_SHARE pairings; a development split's are those of the locked
    build."""
    train = split == "train"
    if kind == "incorrect":
        categories = _shares(n, sm.INCORRECT_MIX)
        rng.shuffle(categories)
        two = set(rng.sample(range(n), round(n * (sm.TRAIN_TWO_VALUES if train else sm.TWO_VALUES))))
        variants = [{"category": c, "two": i in two} for i, c in enumerate(categories)]
        if train:
            for i in rng.sample(range(n), round(n * sm.OPTIONAL_SHARE)):
                variants[i]["optional"] = True
            formats = [i for i, v in enumerate(variants) if v["category"] == "format"]
            for i in rng.sample(formats, round(len(formats) * sm.DESCRIBED_SHARE)):
                variants[i]["described"] = True
        return variants
    if kind == "missing":
        subkinds = _shares(n, sm.MISSING_MIX)
        rng.shuffle(subkinds)
        return [{"subkind": s} for s in subkinds]
    if kind == "valid_multiturn":
        modes = _shares(n, (("change", 0.5), ("switch", 0.5)))
        rng.shuffle(modes)
        packed = set(rng.sample(range(n), round(n * sm.PACKED_SHARE)))
        return [{"mode": m, "packed": i in packed} for i, m in enumerate(modes)]
    if kind == "irrelevant" and train:
        paired = set(rng.sample(range(n), round(n * PAIRING_SHARE)))
        return [{"pairing": True} if i in paired else {} for i in range(n)]
    return [{} for _ in range(n)]


def extras(kind: str, variants: list[dict], rng: random.Random) -> list[dict]:
    """Per training slot of a kind, revision 1's followers (module docstring: Followers): the kind of its twin
    (None for none) and whether its core is put in context. incorrect: every vague item a valid twin, and so
    many of the others a missing twin, drawn, that TWIN_SHARE of the items have one; missing: a drawn
    IRRELEVANT_TWIN_SHARE of the items but the top-level-object ones an irrelevant twin; the refusal kinds
    (special_mix.CONTEXT_KINDS): a drawn CONTEXT_SHARE of the slots in context, of the irrelevant kind among
    its slots for the records' own refusals."""
    n = len(variants)
    twins: list[str | None] = [None] * n
    if kind == "incorrect":
        vague = [i for i, v in enumerate(variants) if v["category"] == "vague"]
        others = [i for i, v in enumerate(variants) if v["category"] != "vague"]
        for i in vague:
            twins[i] = "valid_verbatim"
        for i in rng.sample(others, max(0, min(len(others), round(n * TWIN_SHARE) - len(vague)))):
            twins[i] = "missing"
    elif kind == "missing":
        plain = [i for i, v in enumerate(variants) if v["subkind"] != "object"]
        for i in rng.sample(plain, round(len(plain) * IRRELEVANT_TWIN_SHARE)):
            twins[i] = "irrelevant"
    framed: set[int] = set()
    if kind in sm.CONTEXT_KINDS:
        open_slots = [i for i, v in enumerate(variants) if not v.get("pairing")]
        framed = set(rng.sample(open_slots, min(len(open_slots), round(n * CONTEXT_SHARE))))
    return [{"twin": twins[i], "context": i in framed} for i in range(n)]


def fallbacks(kind: str, variant: dict, *, split: str) -> list[dict]:
    """The constructions to try in turn when the scheduled one finds no record: for an incorrect item the
    plain obvious one, and in training (revision 1) first the same category and value count without the
    revision's optional and described choices, then the same category with one value, then an obvious one with
    the same value count (the format share is a floor of the analysis, so a second value goes before the
    category); for a missing item a single cut; for a multi-turn item the other mode ([] for any other kind).
    A development slot falls back as the locked build's did."""
    out: list[dict] = []
    if kind == "incorrect":
        category, two = variant["category"], variant["two"]
        steps = [{"category": category, "two": two}, {"category": category, "two": False},
                 {"category": "obvious", "two": two}] if split == "train" else []
        for plain in (*steps, {"category": "obvious", "two": False}):
            if plain != variant and plain not in out:
                out.append(plain)
    elif kind == "missing" and variant != {"subkind": "single"}:
        out.append({"subkind": "single"})
    elif kind == "valid_multiturn":
        out.append({**variant, "mode": "change" if variant["mode"] == "switch" else "switch"})
    return out


def canned_cap(planned: int) -> int:
    """How many of a language's training irrelevant items may be a record's own refusal of a canned request:
    CANNED_SHARE of its planned count, rounded."""
    return round(planned * CANNED_SHARE)


def deck(seed: int, lang: str) -> list[sm.Cell]:
    """A language's deck of cells for the training refusal answers (special_mix.cell_deck, CELL_ROUNDS
    rounds), drawn with random.Random(f"{seed}/special-mix/cells/{lang}")."""
    return sm.cell_deck(random.Random(f"{seed}/special-mix/cells/{lang}"), CELL_ROUNDS)


def extra_item(item: sm.Item) -> bool:
    """Whether an item is a follower drawn beyond its kind's planned count: a twin, or a second rendering
    that fills no quota."""
    variant = item.core.variant
    return bool(variant.get("twin_of")) or (variant.get("render") == 2 and not variant.get("fills"))


def _label(kind: str, variant: dict) -> str:
    if kind == "incorrect":
        flags = [key for key in ("two", "optional", "described") if variant.get(key)]
        return "+".join([variant["category"], *flags])
    if kind == "missing":
        return variant["subkind"]
    if kind == "valid_multiturn":
        return variant["mode"] + ("+packed" if variant["packed"] else "")
    if kind == "irrelevant" and variant.get("pairing"):
        return "irrelevant+pairing"
    return kind


def _key(sample: dict) -> tuple[str, str]:
    return sample["source"], str(sample["source_id"])


def template_kind(kind: str) -> str:
    """The kind under which an item's template is kept from training (TEMPLATE_KINDS)."""
    return TEMPLATE_KINDS.get(kind, kind)


def _has_tools(sample: dict) -> bool:
    return bool(sample.get("tools"))


def _record_size(sample: dict) -> int:
    """How much a record adds to a prompt it frames: its request and its first step's function, in
    characters."""
    step = sm.first_step(sample)
    function = sm._function(sample, step[1][0][0]) if step and step[1] else None
    request = sample["messages"][0]["content"] if sample.get("messages") else ""
    return len(request) + len(json.dumps(function, ensure_ascii=False)) if function else len(request)


_DISTINCT = re.compile(r"[^\x00-\x7f]|_|\d|[a-z][A-Z]")
_WORD = re.compile(r"[A-Za-z0-9_]+")


class Mentions:
    """Finds names as whole words of a text (no ASCII letter, digit or underscore on either side): a name
    that is one such word by a lookup among the text's words, any other by a pattern."""

    def __init__(self, names: list[str]) -> None:
        self.words = frozenset(n for n in names if _WORD.fullmatch(n))
        others = sorted((n for n in names if n not in self.words), key=lambda n: (-len(n), n))
        self.pattern = re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(map(re.escape, others)) +
                                  r")(?![A-Za-z0-9_])") if others else None

    def search(self, text: str) -> bool:
        if self.words and not self.words.isdisjoint(_WORD.findall(text)):
            return True
        return bool(self.pattern is not None and self.pattern.search(text))


def mentions(names: set[str]) -> Mentions | None:
    """Mentions of the identifier-like names among names (a non-ASCII one, or one without spaces holding an
    underscore, a digit or an inner capital: plain words are not looked for); None when there is none."""
    distinct = [n for n in names if " " not in n and _DISTINCT.search(n)]
    return Mentions(distinct) if distinct else None


def item_problems(item: sm.Item, bench: Benchmarks) -> list[str]:
    """Why an item may not be kept (module docstring: Checks), by reason; [] when it may."""
    prompt = "\n".join(m["content"] for m in item.messages)
    if sm.approx_tokens(prompt) > MAX_PROMPT_TOKENS:
        return ["too_long"]
    texts = [m["content"] for m in item.messages] + [item.answer, item.reasoning]
    if any(sm.banned_markers(t) for t in texts):
        return ["acebench_text"]
    if any(bench.windows.violations(t) for t in texts):
        return ["template_window"]
    shown = {t["function"]["name"].casefold() for t in item.tools}
    if shown & bench.names:
        return ["acebench_name"]
    if any(overlaps(m["content"], bench) for m in item.messages if m["role"] == "user"):
        return ["overlap"]
    if item.reasoning:  # a training item's (a development item has none)
        problems = sm.reasoning_problems(item)
        if problems:
            return ["reasoning_long" if all(p.startswith("longer") for p in problems) else "reasoning_form"]
    if sm.reads_as_placeholder(item):
        return ["placeholder_lookalike"]
    return []


@dataclass
class Drawer:
    """The draw's state: each source's queue, the records used, the names the development sets reserved, the
    queue pointers and the counts of rejected items; closed holds the names the development sets may not show
    (frequent_names and closed_deepexi_names); requests indexes the drawn items' requests
    (special_mix.request_texts) under (split, number), templates the development items' requests under the
    same ids with their kinds (template_kind), and near counts the candidates a near-duplicate or a template
    turned away by (rule, test, kind). Revision 1's training state: the decks of cells and how many cells each
    language dealt, the canned own refusals drawn per language and their caps, the records of the drawn items
    (for their twins), and the followers counted (revision: twins, renders, contexts and fills by kind, and
    the followers that failed)."""

    queues: dict[str, list[dict]]
    seed: int
    bench: Benchmarks
    banks: dict
    used: set = field(default_factory=set)
    reserved: dict = field(default_factory=lambda: {"a": set(), "b": set()})
    dev_names: set = field(default_factory=set)
    pointers: dict = field(default_factory=dict)
    rejected: Counter = field(default_factory=Counter)
    closed: frozenset = field(default_factory=frozenset)
    requests: sm.RequestIndex = field(default_factory=sm.RequestIndex)
    templates: sm.TemplateIndex = field(default_factory=sm.TemplateIndex)
    near: Counter = field(default_factory=Counter)
    drawn: Counter = field(default_factory=Counter)
    dev_calls: list = field(default_factory=list)  # call_keys of the development items drawn
    decks: dict = field(default_factory=dict)
    dealt: Counter = field(default_factory=Counter)
    canned: Counter = field(default_factory=Counter)
    canned_caps: dict = field(default_factory=dict)
    samples: dict = field(default_factory=dict)
    revision: dict = field(default_factory=lambda: {key: Counter() for key in (
        "twins", "renders", "contexts", "fills", "failed")})
    last_rejection: str = ""  # why checked last turned an item away (its near rule or first problem)
    _mentions: dict = field(default_factory=dict)
    _callers: dict = field(default_factory=dict)
    _features: dict = field(default_factory=dict)

    def features(self, sample: dict) -> dict:
        """What a record's first step holds (kept per record): a call, one call, at most three calls, a
        Chinese value, a top-level object argument; and whether it is a refusal record (functions, no
        call)."""
        key = _key(sample)
        if key not in self._features:
            step = sm.first_step(sample)
            calls = step[1] if step else []
            arguments = [a for _, a in calls if isinstance(a, dict)]
            self._features[key] = {
                "step": step is not None, "single": step is not None and len(calls) == 1,
                "verbatim": step is not None and len(calls) <= 3,
                "cjk": any(isinstance(leaf, str) and sm._CJK.search(leaf) for a in arguments
                           for leaf in sm._leaves(a)),
                "object": any(isinstance(v, dict) for a in arguments for v in a.values()),
                "refusal": bool(sample.get("tools"))
                and not any(m.get("tool_calls") for m in sample["messages"])}
        return self._features[key]

    def can_give(self, kind: str, lang: str, variant: dict, sample: dict) -> bool:
        """False when the record cannot give the kind and construction whatever the split (kept per record):
        no single call where the builder needs one (special_mix.single_call), no first-step call or no Chinese
        value for a verbatim item, no object argument for a top-level-object cut, a call or no functions for
        a refusal record's own irrelevant item. A record it turns away would give none anyway; this only
        spares the draw building it."""
        found = self.features(sample)
        if kind == "irrelevant":
            paired = sample["source"] == "deepexi" or variant.get("pairing")
            return found["step"] if paired else found["refusal"]
        if kind == "valid_verbatim":
            return found["verbatim"] and (lang != "zh" or found["cjk"])
        if kind == "missing" and variant.get("subkind") == "object":
            return found["single"] and found["object"]
        return found["single"]

    def switch_partner(self, split: str, sample: dict) -> bool:
        """Whether a record may open a switch of the split: one call special_mix.single_call accepts under
        the split's names to avoid, and a request no drawn item forbids (near_conflict). Within a split the
        names to avoid stay the same and the drawn requests only grow, so a record found unfit stays unfit:
        the verdict is kept, and a conflict is counted once (rule, test, "switch_partner")."""
        key = (split, _key(sample))
        if key not in self._callers:
            self._callers[key] = sm.single_call(sample, self.avoid(split)) is not None
        if not self._callers[key]:
            return False
        conflict = self.near_conflict(split, [sample["messages"][0]["content"]], "valid_multiturn")
        if conflict is not None:
            self.near[(*conflict, "switch_partner")] += 1
            self._callers[key] = False
            return False
        return True

    def near_conflict(self, split: str, texts: list[str], kind: str,
                      requests: list[str] | None = None) -> tuple[str, str] | None:
        """(rule, test) when a candidate comes too close to the drawn items, None otherwise: "dev" when its
        texts (special_mix.request_texts) near-duplicate a development item (D1: no split may repeat one),
        "train_cap" when a training candidate's near-duplicate TRAIN_NEAR_LIMIT drawn training items;
        "dev_template" when a development candidate's records' requests (requests, else texts: never the
        history wrapping, follow-up or call notation this module adds, which every multi-turn item shares)
        share a template with DEV_TEMPLATE_LIMIT drawn development items, "train_template" when a training
        candidate's share the template of a development item of its kind (template_kind; F1)."""
        found = self.requests.matches(texts)
        dev = [test for (where, _), test in found.items() if where != "train"]
        if dev:
            return "dev", dev[0]
        trained = [test for (where, _), test in found.items() if where == "train"]
        if split == "train" and len(trained) >= TRAIN_NEAR_LIMIT:
            return "train_cap", trained[0]
        requests = texts if requests is None else requests
        if split == "train":
            shared = self.templates.matches(requests, template_kind(kind))
            if shared:
                return "train_template", next(iter(shared.values()))
        else:
            shared = self.templates.matches(requests)
            if len(shared) >= DEV_TEMPLATE_LIMIT:
                return "dev_template", next(iter(shared.values()))
        return None

    def bank(self, sample: dict) -> dict:
        """The value bank (value_banks) of the function a record's first step calls: parameter -> values."""
        step = sm.first_step(sample)
        if step is None or not step[1]:
            return {}
        return self.banks.get(sample["source"], {}).get(step[1][0][0].casefold(), {})

    def dev_mentions(self) -> Mentions | None:
        """mentions() of the development sets' names, made once the training items are drawn (the sets are
        drawn first)."""
        if "pattern" not in self._mentions:
            self._mentions["pattern"] = mentions(self.dev_names)
        return self._mentions["pattern"]

    def avoid(self, split: str) -> frozenset:
        """The names (case-folded) a split's items may not call or show: set B the closed ones, set A those
        and set B's, the training items both sets'."""
        if split == "b":
            return self.closed
        if split == "a":
            return frozenset(self.reserved["b"] | self.closed)
        return frozenset(self.reserved["a"] | self.reserved["b"])

    def candidates(self, source: str, key: tuple):
        """The unused records of a source after the key's pointer, which moves past every record given. A key
        names its split, so the records a development set passes over (a name it avoids) stay open to
        training."""
        queue = self.queues.get(source, [])
        position = self.pointers.get((source, key), 0)
        while position < len(queue):
            sample = queue[position]
            position += 1
            self.pointers[(source, key)] = position
            if _key(sample) not in self.used:
                yield sample

    def partners(self, split: str, source: str, other: dict, purpose: str, suitable):
        """Up to PARTNER_TRIES unused records of a source that suitable(record) accepts, to pair with another,
        from the first such record of the queue. The split's pointer for the purpose moves past the used and
        the unsuitable records at the front for good: within a split the names to avoid only grow, so an
        unsuitable record stays so (across splits they do not: set B avoids the frequent names, training
        does not)."""
        queue = self.queues.get(source, [])
        key = (split, source, "partner", purpose)
        start = self.pointers.get(key, 0)
        while start < len(queue) and (_key(queue[start]) in self.used or not suitable(queue[start])):
            start += 1
        self.pointers[key] = start
        tried = 0
        for sample in queue[start:]:
            if _key(sample) in self.used or _key(sample) == _key(other) or not suitable(sample):
                continue
            yield sample
            tried += 1
            if tried >= PARTNER_TRIES:
                return

    def cell(self, split: str, core: sm.Core) -> sm.Cell | None:
        """The cell a training refusal answer is written in (CELL_KINDS): its language's next undealt one
        (deck); None for any other item."""
        if split != "train" or core.kind not in CELL_KINDS:
            return None
        if core.lang not in self.decks:
            self.decks[core.lang] = deck(self.seed, core.lang)
        cards = self.decks[core.lang]
        return cards[self.dealt[core.lang] % len(cards)]

    def checked(self, split: str, core: sm.Core, rng: random.Random, *,
                order: str | None = None) -> sm.Item | None:
        """The dressed core (in a training refusal answer's cell; order for a second rendering of a priority
        core) when it near-duplicates no request the draw forbids (near_conflict), for a development item
        makes no call of another development item (same_call, counted as ("dev", "call", kind)), for a
        training item names no function of the development sets (dev_mentions; searched before item_problems,
        which costs far more) and passes item_problems; None, the reason counted, otherwise."""
        item = sm.dress(core, split=split, rng=rng, order=order, cell=self.cell(split, core))
        conflict = self.near_conflict(split, list(sm.request_texts(item)), core.kind, list(core.originals))
        if conflict is None and split in self.reserved and \
                any(same_call(call_keys(item), other) for other in self.dev_calls):
            conflict = ("dev", "call")
        if conflict is not None:
            self.near[(*conflict, core.kind)] += 1
            self.last_rejection = conflict[0]
            return None
        problems = []
        if split == "train" and self.dev_mentions() is not None:  # a cheap search, before item_problems
            texts = [m["content"] for m in item.messages] + [item.answer, item.reasoning]
            if any(self.dev_mentions().search(text) for text in texts):
                problems = ["dev_name_mentioned"]
        if split == "train" and not item.reasoning:
            problems = problems or ["reasoning_form"]
        problems = problems or item_problems(item, self.bench)
        if problems:
            self.rejected[problems[0]] += 1
            self.last_rejection = problems[0]
            return None
        return item

    def register(self, split: str, item: sm.Item) -> None:
        """An item drawn: its records used, its requests indexed (requests) and numbered in its split's draw;
        in a development set its template indexed and its function names reserved; in training its cell
        dealt, a canned own refusal counted and an item in context counted."""
        core = item.core
        self.used |= {(core.source, r) for r in core.records}
        number = self.drawn[split]
        self.requests.add((split, number), sm.request_texts(item))
        core.drawn = number
        self.drawn[split] += 1
        if split in self.reserved:
            self.templates.add((split, number), template_kind(core.kind), core.originals)
            named = {t["function"]["name"] for t in item.tools} | {n for n, _ in core.calls}
            named |= {n for role, turn in core.history if role == "assistant" for n, _ in turn}
            self.reserved[split] |= {n.casefold() for n in named}
            self.dev_names |= named
            if call_keys(item):
                self.dev_calls.append(call_keys(item))
        if split == "train":
            if core.kind in CELL_KINDS:
                self.dealt[core.lang] += 1
            if core.kind == "irrelevant" and not core.variant.get("pairing") and \
                    sm.canned_request(core.originals[0]):
                self.canned[core.lang] += 1
            if core.variant.get("context"):
                self.revision["contexts"][core.kind] += 1

    def near_domain(self, sample: dict, partners):
        """The partners of a training pairing: the first PAIRING_SCAN of them by the domain words their
        functions share with the record's own (special_mix.domain_words), most first, the order kept on a
        tie; then the others in turn."""
        own = sm.domain_words(sample.get("tools") or [])
        head = list(itertools.islice(partners, PAIRING_SCAN))
        head.sort(key=lambda partner: -len(sm.domain_words(partner.get("tools") or []) & own))
        yield from head
        yield from partners

    def framed(self, split: str, core: sm.Core, sample: dict, rng: random.Random) -> sm.Item | None:
        """A training refusal core in context that passes the checks: the unused records of its source (a
        single call special_mix.single_call accepts) tried in turn, the first PAIRING_SCAN of them shortest
        first (_record_size: a framed prompt must stay within MAX_PROMPT_TOKENS), each framing the core
        (special_mix.with_context; packed into one message PACKED_CONTEXT of the time) and the framed item
        checked; None when none of PARTNER_TRIES gives one. A partner that cannot frame the core counts as
        context/unframed, a framed item that fails a check as context/<the check> (revision["failed"]:
        last_rejection)."""
        packed = rng.random() < PACKED_CONTEXT
        avoid = self.avoid(split)

        def suitable(record: dict) -> bool:
            return sm.single_call(record, avoid) is not None

        partners = self.partners(split, sample["source"], sample, "context", suitable)
        head = sorted(itertools.islice(partners, PAIRING_SCAN), key=_record_size)
        for partner in itertools.chain(head, partners):
            framed = sm.with_context(core, partner, rng, packed=packed, avoid=avoid)
            if framed is None:
                self.revision["failed"]["context/unframed"] += 1
                continue
            item = self.checked(split, framed, rng)
            if item is not None:
                return item
            self.revision["failed"][f"context/{self.last_rejection}"] += 1
        return None

    def followers(self, split: str, item: sm.Item, extra: dict, *, render: bool = True) -> list[sm.Item]:
        """A training slot's followers (module docstring: Followers), each checked and drawn right after it:
        its twin of extra["twin"], and, with render, a second rendering of a top-level-object or priority
        core (build renders the FIRST constructions' cores after all of their slots: render False); [] for a
        development item."""
        if split != "train":
            return []
        out = []
        if extra.get("twin"):
            twin = self.twin(split, item, extra["twin"])
            if twin is not None:
                out.append(twin)
        core = item.core
        if render and (core.kind == "priority" or
                       (core.kind == "missing" and core.variant.get("subkind") == "object")):
            second = self.render(split, item)
            if second is not None:
                out.append(second)
        return out

    def twin(self, split: str, item: sm.Item, kind: str) -> sm.Item | None:
        """The twin of a training item from the same record (module docstring: Followers): a valid one with
        the original value, a missing one with the wrong value's parameter cut (a valid one when that
        parameter is optional or cannot be cut, counted as missing_twin->valid_verbatim, or when its cut does
        not read fluently, counted as missing_twin/bare->valid_verbatim: a twin has one choice of value to
        cut, and a bare cut such as "My income is and the tax rate is 20%." gave 38 of the first revised
        build's 67 missing twins), or an irrelevant one (irrelevant_twin); None, the failure counted, when
        the record gives none that passes the checks: as twin/<kind>/no_core when the builder makes none, else
        twin/<kind>/<what turned it away> (last_rejection)."""
        core = item.core
        sample = self.samples.get((core.source, str(core.records[0])))
        rng = random.Random(f"{self.seed}/special-mix/{split}/twin/{core.kind}/{core.lang}/{core.source}/"
                            f"{core.records[0]}")
        if sample is None:
            return None
        if kind == "irrelevant":
            return self.irrelevant_twin(split, item, sample, rng)
        avoid = self.avoid(split)
        built = None
        if kind == "missing":
            param = core.invalid[0].param
            if param in core.required:
                built = sm.build_missing(sample, core.lang, rng, subkind="single", avoid=avoid, keys=[param])
            if built is not None and built.variant["cut"] != "frame":
                built = None
                self.revision["failed"]["missing_twin/bare->valid_verbatim"] += 1
            elif built is None:
                self.revision["failed"]["missing_twin->valid_verbatim"] += 1
            if built is None:
                kind = "valid_verbatim"
        if kind == "valid_verbatim":
            built = sm.build_verbatim(sample, core.lang, rng, avoid=avoid)
            built = built if built is not None and len(built.calls) == 1 else None
        if built is None:
            self.revision["failed"][f"twin/{kind}/no_core"] += 1
            return None
        twin = self.checked(split, replace(built, variant={**built.variant, "twin_of": core.kind}), rng)
        if twin is None:
            self.revision["failed"][f"twin/{kind}/{self.last_rejection}"] += 1
            return None
        self.register(split, twin)
        self.revision["twins"][twin.core.kind] += 1
        return twin

    def irrelevant_twin(self, split: str, item: sm.Item, sample: dict, rng: random.Random) -> sm.Item | None:
        """The irrelevant twin of a missing item: its request as cut, paired with an unused record's functions
        unrelated to it (special_mix.build_irrelevant with user; the partners near-domain first); counted as
        twin/irrelevant/no_partner when no partner gives one that passes the checks."""
        core = item.core
        avoid = self.avoid(split)
        partners = self.partners(split, core.source, sample, "tools", _has_tools)
        for partner in self.near_domain(sample, partners):
            built = sm.build_irrelevant(sample, core.lang, rng, tools_from=partner, avoid=avoid,
                                        user=core.history[-1][1])
            if built is None:
                continue
            twin = self.checked(split, replace(built, variant={**built.variant, "twin_of": core.kind}), rng)
            if twin is not None:
                self.register(split, twin)
                self.revision["twins"]["irrelevant"] += 1
                return twin
        self.revision["failed"]["twin/irrelevant/no_partner"] += 1
        return None

    def render(self, split: str, item: sm.Item, *, fills: bool = False) -> sm.Item | None:
        """A second rendering of a training item's core (variant render 2; fills when it fills its kind's
        quota): a priority core under the other order, any core in its next cell; of up to RENDER_TRIES
        generators the first whose prompt renders the functions in another list style. None when it does not
        pass the checks, counted as render/<kind>/<what turned it away> (last_rejection: train_cap when the
        core's near-duplicate cluster holds TRAIN_NEAR_LIMIT training items already, say)."""
        core = item.core
        order = None
        if core.kind == "priority":
            order = next(o for o in sm.ORDERS if o != item.rules.order)
        marks = {"render": 2, **({"fills": True} if fills else {})}
        second = replace(core, variant={**core.variant, **marks}, drawn=-1)
        stem = (f"{self.seed}/special-mix/{split}/render/{core.kind}/{core.lang}/{core.source}/"
                f"{core.records[0]}")
        chosen = f"{stem}/0"
        for attempt in range(RENDER_TRIES):
            probe = sm.dress(second, split=split, rng=random.Random(f"{stem}/{attempt}"), order=order,
                             cell=self.cell(split, second))
            if probe.list_style != item.list_style:
                chosen = f"{stem}/{attempt}"
                break
        rendered = self.checked(split, second, random.Random(chosen), order=order)
        if rendered is None:
            self.revision["failed"][f"render/{core.kind}/{self.last_rejection}"] += 1
            return None
        self.register(split, rendered)
        if fills:
            self.revision["fills"][f"{core.kind}/{core.lang}"] += 1
        else:
            self.revision["renders"][core.kind] += 1
        return rendered

    def fill(self, split: str, items: list[sm.Item], short: int) -> list[sm.Item]:
        """Up to short second renderings of the Chinese-source cores among a kind's drawn items, in their
        order (FILL_KINDS): what fills the kind's planned count when its records run short."""
        out: list[sm.Item] = []
        for item in items:
            if len(out) >= short:
                break
            if item.core.source in ZH_SOURCES and not item.core.variant.get("render"):
                rendered = self.render(split, item, fills=True)
                if rendered is not None:
                    out.append(rendered)
        return out

    def attempt(self, split: str, kind: str, lang: str, variant: dict, sample: dict,
                rng: random.Random, fluent: bool | None = None, context: bool = False) -> sm.Item | None:
        """The checked item a record gives for the kind and construction; a paired kind (deepexi's
        irrelevant requests, an English request paired with another record's functions, switches) tries its
        partners in turn until one gives an item that passes (a training pairing near-domain partners first:
        near_domain). A record whose request near-duplicates what the draw forbids (near_conflict) gives none,
        and neither does a training record whose request names a function of the development sets, nor a
        record able to lose a top-level object as the request of a development irrelevant item (its
        near-duplicates would leave training with it, and those records are the scarcest: FIRST), nor a
        record's own refusal of a canned request once its language's training items hold canned_caps of them
        (each counted once); with fluent True, a bare cut gives none either (counted as bare_deferred), before
        it is dressed. With context, a training refusal core is put in context (contextualize) when a partner
        frames it and the framed item passes the checks, else it is drawn plain (counted as
        context_fallback)."""
        if not self.can_give(kind, lang, variant, sample):
            return None
        if kind == "irrelevant" and split in self.reserved and self.features(sample)["single"] and \
                self.features(sample)["object"]:
            self.rejected["object_kept_for_training"] += 1
            return None
        avoid = self.avoid(split)
        request = sample["messages"][0]["content"] if sample["messages"] else ""
        conflict = self.near_conflict(split, [request], kind) if request else None
        if conflict is not None:
            self.near[(*conflict, kind)] += 1
            return None
        if split == "train" and self.dev_mentions() is not None and self.dev_mentions().search(request):
            self.rejected["dev_name_mentioned"] += 1
            return None
        paired = kind == "irrelevant" and (sample["source"] == "deepexi" or variant.get("pairing"))
        if kind == "irrelevant" and not paired and split == "train" and sm.canned_request(request) and \
                self.canned[lang] >= self.canned_caps.get(lang, math.inf):
            self.rejected["canned_capped"] += 1
            return None
        if paired:
            partners = self.partners(split, sample["source"], sample, "tools", _has_tools)
            for partner in self.near_domain(sample, partners) if split == "train" else partners:
                core = sm.build_irrelevant(sample, lang, rng, tools_from=partner, avoid=avoid)
                if core is not None and (item := self.checked(split, core, rng)) is not None:
                    return item
            return None
        if kind == "valid_multiturn" and variant["mode"] == "switch":
            for partner in self.partners(split, sample["source"], sample, "switch",
                                         lambda s: self.switch_partner(split, s)):
                core = sm.build_switch(partner, sample, lang, rng, packed=variant["packed"], avoid=avoid)
                if core is not None and (item := self.checked(split, core, rng)) is not None:
                    return item
            return None
        if kind == "incorrect":
            core = sm.build_incorrect(sample, lang, rng, category=variant["category"], two=variant["two"],
                                      avoid=avoid, optional=variant.get("optional", False),
                                      described=variant.get("described", False), complete=split == "train")
        elif kind == "missing":
            core = sm.build_missing(sample, lang, rng, subkind=variant["subkind"], avoid=avoid)
        elif kind == "ask":
            core = sm.build_ask(sample, lang, rng, avoid=avoid)
        elif kind == "priority":
            core = sm.build_priority(sample, lang, rng, avoid=avoid)
        elif kind == "valid_verbatim":
            core = sm.build_verbatim(sample, lang, rng, require_cjk=lang == "zh", avoid=avoid)
        elif kind == "valid_nearmiss":
            core = sm.build_nearmiss(sample, lang, rng, avoid=avoid, why=self.rejected)
        elif kind == "irrelevant":
            core = sm.build_irrelevant(sample, lang, rng, avoid=avoid)
        else:
            core = sm.build_change(sample, lang, rng, bank=self.bank(sample), packed=variant["packed"],
                                   avoid=avoid)
        if core is not None and fluent and core.variant.get("cut") == "bare":
            self.rejected["bare_deferred"] += 1
            return None
        if core is None:
            return None
        if context and split == "train" and core.kind in sm.CONTEXT_KINDS and not core.variant.get("pairing"):
            item = self.framed(split, core, sample, rng)
            if item is not None:
                return item
            self.rejected["context_fallback"] += 1
        return self.checked(split, core, rng)

    def slot(self, split: str, kind: str, lang: str, variant: dict, sources: tuple[str, ...],
             start: int, *, fluent: bool | None = None, context: bool = False) -> sm.Item | None:
        """The item of one slot: the next record of the sources (round robin from start) that gives a checked
        item (attempt, in context when asked for); it is then drawn (register) and its record kept for its
        followers. fluent True takes only an item whose cut reads fluently (its variant's cut "frame"),
        passing the others over for a later slot (counted as bare_deferred); each fluency has its own place in
        the queues."""
        for offset in range(len(sources)):
            source = sources[(start + offset) % len(sources)]
            pointer = (split, kind, lang, _label(kind, variant), fluent)
            for sample in self.candidates(source, pointer):
                rng = random.Random(f"{self.seed}/special-mix/{split}/{kind}/{lang}/{source}/"
                                    f"{sample['source_id']}")
                item = self.attempt(split, kind, lang, variant, sample, rng, fluent, context)
                if item is None:
                    continue
                self.register(split, item)
                self.samples[_key(sample)] = sample
                return item
        return None


def draw_slot(drawer: Drawer, split: str, kind: str, lang: str, variant: dict, sources: tuple[str, ...],
              number: int, *, context: bool = False) -> tuple[sm.Item | None, dict]:
    """The item of one slot and the construction it was drawn with (module docstring: Draw): the scheduled
    construction, then its fallbacks in turn; for each, the language's sources, then (a zh item) the English
    ones; for each, a fluent cut first where the kind cuts a value (CUT_KINDS), then any. An irrelevant item
    no record's own answer gives pairs an English request with another record's functions last (D2). context
    asks for a training refusal core in context (Drawer.attempt)."""
    groups = [sources] + ([EN_SOURCES] if sources == ZH_SOURCES else [])
    fluencies = (True, False) if kind in CUT_KINDS else (None,)
    for construction in [variant, *fallbacks(kind, variant, split=split)]:
        for group in groups:
            for fluent in fluencies:
                item = drawer.slot(split, kind, lang, construction, group, number, fluent=fluent,
                                   context=context)
                if item is not None:
                    return item, construction
    if kind == "irrelevant":
        paired = {"pairing": True}
        item = drawer.slot(split, kind, lang, paired, EN_SOURCES, number, context=context)
        if item is not None:
            return item, paired
    return None, variant


def build(pools: dict[str, list[dict]], *, seed: int, scale: float, bench: Benchmarks,
          banks: dict) -> tuple[dict[str, list[sm.Item]], dict]:
    """The items of every split (module docstring: Draw) and an account of the draw."""
    queues = {}
    for name in [n for n in SOURCES if n in pools]:
        queue = list(pools[name])
        random.Random(f"{seed}/special-mix/{name}").shuffle(queue)
        queues[name] = queue
    frequent = frequent_names(pools, DEV_NAME_LIMIT)
    deepexi = closed_deepexi_names(pools, seed, DEV_DEEPEXI_SHARE)
    rank = sm.token_rank(s["messages"][0]["content"] for samples in pools.values() for s in samples
                         if s["messages"])
    drawer = Drawer(queues, seed, bench, banks, closed=frequent | deepexi, requests=sm.RequestIndex(rank))
    splits: dict[str, list[sm.Item]] = {split: [] for split in SPLIT_ORDER}
    account: dict = {"planned": {}, "planned_constructions": {}, "variants": {}, "fallbacks": {},
                     "zh_from_en_sources": {}, "zh_sources": {}, "short": {},
                     "dev_name_limit": DEV_NAME_LIMIT, "dev_avoided_names": len(frequent),
                     "dev_deepexi_share": DEV_DEEPEXI_SHARE, "dev_deepexi_closed": len(deepexi)}
    for split in SPLIT_ORDER:
        plan = sm.PLAN if split == "train" else sm.DEV_PLAN[split]
        slots = {}
        for kind in DRAW_ORDER:
            if kind not in plan:
                continue
            for lang in sm.LANGS:
                n = scaled(plan[kind][lang], scale)
                where = f"{split}/{kind}/{lang}"
                account["planned"][where] = n
                rng = random.Random(f"{seed}/special-mix/schedule/{where}")
                variants = schedule(kind, n, rng, split=split)
                if kind == "irrelevant" and split != "train":  # pairings only (module docstring: Draw)
                    variants = [{"pairing": True} for _ in variants]
                account["planned_constructions"][where] = dict(sorted(Counter(
                    _label(kind, variant) for variant in variants).items()))
                special = kind in sm.SPECIAL_KINDS and lang == "zh"
                english = set(rng.sample(range(n), round(n * EN_SHARE))) if special else set()
                # the training followers draw after the development sets' draws (revision 1)
                follow = extras(kind, variants, rng) if split == "train" else \
                    [{"twin": None, "context": False}] * n
                if split == "train" and kind == "irrelevant":
                    drawer.canned_caps[lang] = canned_cap(n)
                slots[kind, lang] = (n, variants, english, follow)
        # The FIRST constructions' slots, drawn before the others; English first, since a zh slot falls back
        # to English records only after its Chinese ones and an English slot has nothing else. A slot's
        # twin comes right after it; the second renderings of their cores (revision 1) come after every one
        # of these slots, so that a rendering takes only the room in a near-duplicate cluster no core takes.
        early = {}
        for kind, lang in [(k, lang) for k in FIRST for lang in ("en", "zh") if (k, lang) in slots]:
            n, variants, english, follow = slots[kind, lang]
            for number, variant in enumerate(variants):
                if FIRST[kind] == _label(kind, variant):
                    sources = EN_SOURCES if lang == "en" or number in english else ZH_SOURCES
                    item, drawn = draw_slot(drawer, split, kind, lang, variant, sources, number,
                                            context=follow[number]["context"])
                    behind = drawer.followers(split, item, follow[number], render=False) \
                        if item is not None else []
                    early[kind, lang, number] = (item, drawn, behind)
        for (kind, lang, number), (item, _, behind) in early.items():
            if item is not None and split == "train" and item.core.variant.get("subkind") == "object":
                second = drawer.render(split, item)
                if second is not None:
                    behind.append(second)
        for (kind, lang), (n, variants, english, follow) in slots.items():
            where = f"{split}/{kind}/{lang}"
            labels, fell, from_en = Counter(), Counter(), 0
            got = 0
            drawn_items = []
            for number, variant in enumerate(variants):
                sources = EN_SOURCES if lang == "en" or number in english else ZH_SOURCES
                if (kind, lang, number) in early:
                    item, drawn, behind = early.pop((kind, lang, number))
                else:
                    item, drawn = draw_slot(drawer, split, kind, lang, variant, sources, number,
                                            context=follow[number]["context"])
                    behind = drawer.followers(split, item, follow[number]) if item is not None else []
                if item is None:
                    continue
                if drawn != variant and not drawn.get("pairing"):
                    fell[f"{_label(kind, variant)}->{_label(kind, drawn)}"] += 1
                got += 1
                labels[_label(item.core.kind, item.core.variant | _variant_flags(kind, item))] += 1
                from_en += lang == "zh" and item.core.source in EN_SOURCES
                splits[split] += [item, *behind]
                drawn_items.append(item)
            account["variants"][where] = dict(sorted(labels.items()))
            if fell:
                account["fallbacks"][where] = dict(sorted(fell.items()))
            if lang == "zh":
                account["zh_from_en_sources"][where] = from_en
                planned = n - len(english)  # the items the Chinese sources are to give
                account["zh_sources"][where] = {"planned": planned, "got": got - from_en,
                                                "short": planned - (got - from_en)}
            if split == "train" and (kind, lang) in FILL_KINDS and got < n:
                filled = drawer.fill(split, drawn_items, n - got)
                splits[split] += filled
                got += len(filled)
            if got < n:
                account["short"][where] = {"planned": n, "got": got}
        print(f"drawn {split}: {len(splits[split])} items", flush=True)
    account["rejected"] = dict(sorted(drawer.rejected.items()))
    account["reserved_names"] = {split: len(names) for split, names in drawer.reserved.items()}
    account["records_used"] = len(drawer.used)
    account["near_duplicates"] = near_account(drawer.near)
    account["irrelevant_pairing"] = {split: {lang: sum(i.core.kind == "irrelevant" and i.lang == lang
                                                       and bool(i.core.variant.get("pairing")) for i in items)
                                             for lang in sm.LANGS} for split, items in splits.items()}
    account["cuts"] = cut_account(splits)
    account["followers"] = {"train": {key: dict(sorted(counts.items()))
                                      for key, counts in drawer.revision.items()}}
    account["irrelevant"] = irrelevant_account(splits)
    account["objects"] = object_account(splits["train"])
    account["cells"] = cell_account(splits["train"])
    account["incorrect_mix"] = {"train": incorrect_account(splits["train"])}
    return splits, account


def irrelevant_account(splits: dict[str, list[sm.Item]]) -> dict:
    """Per split and language, the irrelevant items: all, the pairings, the records' own refusals and how
    many of those ask for something canned (special_mix.canned_request)."""
    out: dict = {}
    for split, items in splits.items():
        out[split] = {}
        for lang in sm.LANGS:
            mine = [i for i in items if i.core.kind == "irrelevant" and i.lang == lang]
            own = [i for i in mine if not i.core.variant.get("pairing")]
            out[split][lang] = {"items": len(mine), "pairings": len(mine) - len(own), "own": len(own),
                                "canned": sum(sm.canned_request(i.core.originals[0]) for i in own),
                                "pairing_share": round((len(mine) - len(own)) / max(1, len(mine)), 4)}
    return out


def object_account(items: list[sm.Item]) -> dict:
    """Per training language (as train/missing/<lang>), the missing items that lose a top-level object: its
    cores, their second renderings, all of them and their share of the language's missing items (followers
    and all)."""
    out: dict = {}
    for lang in sm.LANGS:
        missing = [i for i in items if i.core.kind == "missing" and i.lang == lang]
        objects = [i for i in missing if i.core.variant.get("subkind") == "object"]
        renders = sum(i.core.variant.get("render") == 2 for i in objects)
        out[f"train/missing/{lang}"] = {"cores": len(objects) - renders, "renders": renders,
                                        "items": len(objects), "missing_items": len(missing),
                                        "share": round(len(objects) / max(1, len(missing)), 4)}
    return out


def cell_account(items: list[sm.Item]) -> dict:
    """Per language, the training refusal answers that fill placeholders (CELL_KINDS) by their cell (style,
    slot order and position of the answered family): how many, by cell, and the fewest of any base cell
    (special_mix.BASE_CELLS) and of the six base styles' (slot order, position) cells overall."""
    out: dict = {}
    for lang in sm.LANGS:
        cells: Counter = Counter()
        for item in items:
            if item.lang == lang and item.core.kind in CELL_KINDS:
                order, position = sm.family_shape(item.rules.families[item.expected["case"]])
                cells[f"{item.rules.style}/{order}/{position}"] += 1
        base = [cells[f"{c.style}/{c.order}/{c.position}"] for c in sm.BASE_CELLS]
        out[lang] = {"answers": sum(cells.values()), "by_cell": dict(sorted(cells.items())),
                     "fewest_base_cell": min(base) if base else 0,
                     "extra_style_share": round(sum(n for key, n in cells.items()
                                                    if key.split("/")[0] in sm.EXTRA_STYLES)
                                                / max(1, sum(cells.values())), 4)}
    return out


def incorrect_account(items: list[sm.Item]) -> dict:
    """The training incorrect items by revision 1's mix: all, two values, a wrong value on an optional
    parameter, the categories, and the format items whose format only the description implies."""
    mine = [i.core for i in items if i.core.kind == "incorrect"]
    return {"items": len(mine), "two_values": sum(len(c.invalid) == 2 for c in mine),
            "optional": sum(c.invalid[0].param not in c.required for c in mine),
            "categories": dict(sorted(Counter(c.variant["category"] for c in mine).items())),
            "described": sum(c.variant["category"] == "format"
                             and sm.described_format(c.invalid[0].constraint) for c in mine)}


def near_account(near: Counter) -> dict:
    """The candidates a near-duplicate turned away (Drawer.near), in total and by rule, test and kind."""
    by: dict[str, Counter] = {"by_rule": Counter(), "by_test": Counter(), "by_kind": Counter()}
    for (rule, test, kind), count in near.items():
        by["by_rule"][rule] += count
        by["by_test"][test] += count
        by["by_kind"][kind] += count
    return {"rejected": sum(near.values()), **{k: dict(sorted(v.items())) for k, v in by.items()}}


def cut_account(splits: dict[str, list[sm.Item]]) -> dict:
    """Per split and kind that cuts a value (CUT_KINDS, as its plan has it): the items whose cut took its
    frame and reads fluently ("frame"), the others ("bare"), and the fluent share."""
    out: dict = {}
    for split, items in splits.items():
        plan = sm.PLAN if split == "train" else sm.DEV_PLAN[split]
        out[split] = {}
        for kind in (k for k in CUT_KINDS if k in plan):
            cuts = Counter(i.core.variant.get("cut") for i in items if i.core.kind == kind)
            total = cuts["frame"] + cuts["bare"]
            out[split][kind] = {"frame": cuts["frame"], "bare": cuts["bare"],
                                "fluent_share": round(cuts["frame"] / max(1, total), 4)}
    return out


def near_problems(splits: dict[str, list[sm.Item]]) -> list[str]:
    """The near-duplicates and templates the drawn items still hold, as messages ([] when none; a check of the
    draw): a development item that near-duplicates another development item or a training item, or makes the
    call of another development item (same_call), or shares its template with DEV_TEMPLATE_LIMIT development
    items drawn before it (set B before set A); a training item with TRAIN_NEAR_LIMIT training items drawn
    before it near-duplicating it, or sharing the template of a development item of its kind (template_kind).
    Templates are read from the records' requests (Core.originals). "Drawn before" is in the order of the
    draw, special_mix.Core.drawn: the FIRST constructions come before their place in the output."""
    problems = []
    dev = sm.RequestIndex()
    calls: list[list[tuple]] = []
    for split in ("b", "a"):
        for number, item in enumerate(splits[split]):
            if dev.matches(sm.request_texts(item)):
                problems.append(f"{split} item {number} near-duplicates another development item")
            dev.add((split, number), sm.request_texts(item))
            if any(same_call(call_keys(item), other) for other in calls):
                problems.append(f"{split} item {number} makes the call of another development item")
            if call_keys(item):
                calls.append(call_keys(item))
    templates = sm.TemplateIndex()
    for split in ("b", "a"):
        for number in sorted(range(len(splits[split])), key=lambda n: splits[split][n].core.drawn):
            item = splits[split][number]
            if len(templates.matches(item.core.originals)) >= DEV_TEMPLATE_LIMIT:
                problems.append(f"{split} item {number} shares its template with {DEV_TEMPLATE_LIMIT} "
                                "development items drawn before it")
            templates.add((split, number), template_kind(item.core.kind), item.core.originals)
    trained = sm.RequestIndex()
    in_draw = sorted(range(len(splits["train"])), key=lambda n: splits["train"][n].core.drawn)
    for number in in_draw:
        item = splits["train"][number]
        texts = sm.request_texts(item)
        if dev.matches(texts):
            problems.append(f"training item {number} near-duplicates a development item")
        if len(trained.matches(texts)) >= TRAIN_NEAR_LIMIT:
            problems.append(f"training item {number} near-duplicates {TRAIN_NEAR_LIMIT} drawn before it")
        if templates.matches(item.core.originals, template_kind(item.core.kind)):
            problems.append(f"training item {number} shares the template of a development item of its kind")
        trained.add(number, texts)
    return problems


def _variant_flags(kind: str, item: sm.Item) -> dict:
    """The schedule's fields of a drawn item, read back from its core (for the account's labels)."""
    variant = item.core.variant
    if kind == "incorrect":
        return {"category": variant["category"], "two": variant["values"] == 2,
                **{key: True for key in ("optional", "described") if variant.get(key)}}
    if kind == "valid_multiturn":
        return {"mode": variant["mode"], "packed": item.core.packed}
    return {}


def shortfalls(account: dict) -> list[str]:
    """The kinds short beyond what is allowed: a training kind by more than MAX_SHORT, a development set's by
    any item; and a training language whose missing items lose a top-level object fewer times than the
    schedule plans (the plan's "at least 20%", special_mix.MISSING_MIX), counting each core and its second
    rendering (account["objects"], revision 1; the drawn object constructions when it is absent)."""
    problems = []
    for where, short in account["short"].items():
        split = where.split("/")[0]
        missing = short["planned"] - short["got"]
        if split != "train" or missing > MAX_SHORT * short["planned"]:
            problems.append(f"{where}: {short['got']} of {short['planned']}")
    for where, planned in account.get("planned_constructions", {}).items():
        if where.startswith("train/missing/") and planned.get("object"):
            objects = account.get("objects", {}).get(where)  # cores and their second renderings
            got = objects["items"] if objects else account["variants"].get(where, {}).get("object", 0)
            if got < planned["object"]:
                problems.append(f"{where}: {got} of {planned['object']} object items")
    return problems


# ---- outputs --------------------------------------------------------------------------------------------


def _ordered(items: list[sm.Item]) -> list[sm.Item]:
    groups = [(kind, lang) for kind in sm.KINDS for lang in sm.LANGS]
    return sorted(items, key=lambda item: groups.index((item.core.kind, item.lang)))


def training_rows(items: list[sm.Item], stem: str) -> tuple[list[dict], list[dict]]:
    """The training records and their sidecar rows (module docstring: Written)."""
    records, rows = [], []
    for index, item in enumerate(_ordered(items)):
        record = sm.training_record(item)
        records.append(record)
        k, _, answer = turns(record["messages"])[-1]
        name = f"{stem}:{index}"
        identity = {"model": sm.TEMPLATE_MODEL, "id": name, "k": k, "kind": item.core.kind,
                    "source": item.core.source, "records": list(item.core.records)}
        key = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
        rows.append(sidecar_row(name, k, answer, source=TEMPLATE, reasoning=item.reasoning,
                                model=sm.TEMPLATE_MODEL, cache_key=key, context_equal=True))
    return records, rows


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def verify_sidecar(records: list[dict], rows: list[dict], stem: str) -> dict:
    """ReasoningIndex over the rows written to a staging file, and a lookup of every record's trained turn."""
    with tempfile.TemporaryDirectory() as staging:
        path = Path(staging) / "sidecar.jsonl"
        write_sidecar(rows, path)
        index = ReasoningIndex([path])
        bound = 0
        for number, record in enumerate(records):
            k, _, answer = turns(record["messages"])[-1]
            bound += index.lookup(f"{stem}:{number}", k, answer) is not None
        return {"rows": len(index), "records": len(records), "bound": bound}


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _output(path: Path, data: bytes, records: int) -> dict:
    return {"path": repo_relative(path), "records": records, "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data)}


def counts(splits: dict[str, list[sm.Item]]) -> dict:
    out: dict = {}
    for split, items in splits.items():
        by_kind = {}
        for kind in sm.KINDS:
            row = {lang: sum(i.core.kind == kind and i.lang == lang for i in items) for lang in sm.LANGS}
            if any(row.values()) or (split == "train" or kind != "ask"):
                by_kind[kind] = {**row, "total": sum(row.values())}
        out[split] = {**by_kind, "total": len(items)}
    return out


def per_source(splits: dict[str, list[sm.Item]]) -> dict:
    out: dict = {}
    for split, items in splits.items():
        table = Counter((i.core.kind, i.lang, i.core.source) for i in items)
        out[split] = {f"{kind}/{lang}": {s: table[kind, lang, s] for s in SOURCES if table[kind, lang, s]}
                      for kind in sm.KINDS for lang in sm.LANGS if any(table[kind, lang, s] for s in SOURCES)}
    return out


def prompt_choices(items: list[sm.Item]) -> dict:
    return {"list_style": dict(sorted(Counter(i.list_style for i in items).items())),
            "call_format": dict(sorted(Counter(i.call_format for i in items).items())),
            "placeholder": dict(sorted(Counter(i.rules.style for i in items if i.rules).items())),
            "order": dict(sorted(Counter(i.rules.order for i in items if i.rules and i.rules.order).items())),
            "packed": sum(i.core.packed for i in items),
            "context_turns": sum(i.context_turns > 0 for i in items),
            "with_rules": sum(i.rules is not None for i in items)}


def families_used(items: list[sm.Item]) -> list[str]:
    return sorted({f.id for i in items if i.rules for f in i.rules.families.values()})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--reasoning-out", type=Path, default=REASONING)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--dev-a", type=Path, default=DEV["a"])
    parser.add_argument("--dev-b", type=Path, default=DEV["b"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--scale", type=float, default=1.0,
                        help="scale every planned count (at least 1 each; for tests and dry runs)")
    parser.add_argument("--force", action="store_true", help="replace existing outputs")
    parser.add_argument("--source-manifest", default=str(public_pool.MANIFEST))
    parser.add_argument("--raw-dir", default=str(public_pool.RAW_DIR), help="the drawn files to leave out")
    parser.add_argument("--ft3-manifest", type=Path, default=FT3_MANIFEST)
    parser.add_argument("--probe-manifest", type=Path, default=PROBE_MANIFEST)
    parser.add_argument("--no-default-excludes", action="store_true",
                        help="do not read " + ", ".join(p.as_posix() for p in EXCLUDE_FILES))
    parser.add_argument("--acebench-dir", type=Path, default=ACEBENCH,
                        help="the ACEBench clone (its data_all and model_inference folders)")
    parser.add_argument("--bfcl-dir", type=Path, default=None)
    parser.add_argument("--vocabulary-exclusions", type=Path, default=VOCABULARY,
                        help="the records the private exclusion list excludes, by source and id")
    parser.add_argument("--accept-shortfall", action="append", default=[], metavar="TEXT",
                        help="a shortfall the build may have, as shortfalls() words it (repeatable): the "
                             "build is written only when its shortfalls are exactly the accepted ones")
    for split in ("a", "b"):
        parser.add_argument(f"--locked-dev-{split}", type=Path, default=None, metavar="LOCK",
                            help=f"a lock of development set {split.upper()} (<sha256>  <file name>) "
                                 "the drawn set must match byte for byte, or nothing is written")
    return parser


def locked_digest(lock: Path, name: str) -> str:
    """The SHA-256 a lock file pins for a file of the name (research_agent.eval.lock's one line "<hex>  <file
    name>"); ValueError when the lock cannot be read, is not of that form or names another file."""
    try:
        lines = lock.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"{lock}: {exc}") from exc
    found = re.fullmatch(r"([0-9a-f]{64})  (\S.*)", lines[0].strip()) if len(lines) == 1 else None
    if found is None:
        raise ValueError(f"{lock} is not one '<sha256>  <file name>' line")
    if found.group(2) != name:
        raise ValueError(f"{lock} pins {found.group(2)}, not {name}")
    return found.group(1)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outputs = {"out": args.out, "reasoning": args.reasoning_out, "dev_a": args.dev_a, "dev_b": args.dev_b,
               "manifest": args.manifest}
    existing = [p for p in [*outputs.values(), lock_path(args.dev_a), lock_path(args.dev_b)] if p.exists()]
    if existing and not args.force:
        print(f"{', '.join(map(str, existing))} exists; pass --force to replace it", file=sys.stderr)
        return 2
    try:
        vocabulary, vocabulary_account = load_vocabulary(args.vocabulary_exclusions)
    except (OSError, ValueError) as exc:
        print(f"{args.vocabulary_exclusions}: no vocabulary exclusion file of the agreed form ({exc})",
              file=sys.stderr)
        return 2
    pinned: dict[str, tuple[Path, str]] = {}
    for split, lock, dev in (("a", args.locked_dev_a, args.dev_a), ("b", args.locked_dev_b, args.dev_b)):
        if lock is not None:
            try:
                pinned[split] = (lock, locked_digest(lock, dev.name))
            except ValueError as exc:
                print(f"--locked-dev-{split}: {exc}", file=sys.stderr)
                return 2
    try:
        bench = load_benchmarks(args.acebench_dir, args.bfcl_dir)
    except BenchmarkMissing as exc:
        print(f"benchmarks missing: {exc}", file=sys.stderr)
        return 2
    print(f"benchmarks: {len(bench.index.grams)} question grams, {len(bench.names)} ACEBench names",
          flush=True)
    excludes = [] if args.no_default_excludes else list(EXCLUDE_FILES)
    used, used_account = used_records(args.ft3_manifest, excludes, args.probe_manifest)
    pools, accounts = load_pools(args.source_manifest, args.raw_dir)
    print(f"pools: {({n: len(p) for n, p in pools.items()})}; used elsewhere {len(used)}", flush=True)
    kept, screened = screen(pools, used, bench, vocabulary)
    print(f"screened: kept {({n: len(p) for n, p in kept.items()})}", flush=True)
    census = gold_call_census(kept)
    splits, account = build(kept, seed=args.seed, scale=args.scale, bench=bench, banks=value_banks(kept))
    short = sorted(shortfalls(account))
    if short != sorted(args.accept_shortfall):
        print("the supply is short: " + ("; ".join(short) or "nowhere"), file=sys.stderr)
        if args.accept_shortfall:
            print("accepted: " + "; ".join(sorted(args.accept_shortfall)), file=sys.stderr)
        return 3
    stem = args.out.stem
    records, rows = training_rows(splits["train"], stem)
    dev = {split: [sm.dev_item(item, split, n) for n, item in enumerate(splits[split], 1)]
           for split in ("a", "b")}
    problems = [f"training record {n}: {p}" for n, r in enumerate(records)
                for p in sm.check_training_record(r)]
    problems += [f"{i['id']}: {p}" for items in dev.values() for i in items for p in sm.check_dev_item(i)]
    problems += [p for items in dev.values() for p in sm.dev_set_problems(items)]
    problems += near_problems(splits)
    sidecar = verify_sidecar(records, rows, stem)
    if sidecar["bound"] != len(records):
        problems.append(f"the sidecar binds {sidecar['bound']} of {len(records)} records")
    if problems:
        print("checks failed: " + "; ".join(problems[:10]), file=sys.stderr)
        return 3
    data = {"out": jsonl_bytes(records), "dev_a": jsonl_bytes(dev["a"]), "dev_b": jsonl_bytes(dev["b"])}
    drawn_digests = {split: hashlib.sha256(data[f"dev_{split}"]).hexdigest() for split in pinned}
    dev_locks = {split: {"lock": repo_relative(lock), "sha256": drawn_digests[split],
                         "matches": drawn_digests[split] == digest}
                 for split, (lock, digest) in pinned.items()}
    changed = [split for split, check in dev_locks.items() if not check["matches"]]
    if changed:
        print("the drawn development set(s) " + ", ".join(changed) + " do not match the locks given: nothing "
              "written", file=sys.stderr)
        return 3
    for key in ("out", "dev_a", "dev_b"):
        outputs[key].parent.mkdir(parents=True, exist_ok=True)
        outputs[key].write_bytes(data[key])
    digest, size = write_sidecar(rows, args.reasoning_out)
    write_lock(args.dev_a)
    write_lock(args.dev_b)
    manifest = {
        "script": "scripts/build_special_mix.py",
        "plan": "docs/m8-计划.md WP1 (special-mix) and WP2 (development sets A and B)",
        "test_material": False,
        "seed": args.seed,
        "scale": args.scale,
        "counts": counts(splits),
        "planned": account["planned"],
        "short": account["short"],
        "max_short": MAX_SHORT,
        "shortfalls_accepted": short,
        "per_source": per_source(splits),
        "zh_from_en_sources": account["zh_from_en_sources"],
        "en_share_target": EN_SHARE,
        "zh_sources": {"note": "per zh kind, the items the Chinese sources were to give (the planned English "
                               "share of the special kinds aside), the items they gave, and the shortfall, "
                               "which English records under the Chinese prompt filled (zh_from_en_sources); "
                               "the plan's value-locked translation, not part of this build, is what would "
                               "fill it with Chinese requests",
                       **account["zh_sources"]},
        "planned_constructions": account["planned_constructions"],
        "constructions": account["variants"],
        "fallbacks": account["fallbacks"],
        "rejected_attempts": {"note": "items dressed and dropped by reason (module docstring: Checks); a "
                                      "paired item counts every partner it tried; bare_deferred counts the "
                                      "bare cuts a fluent pass left to a later slot, enum_option_named the "
                                      "near misses dropped for naming another option of their enum, "
                                      "object_kept_for_training the records able to lose a top-level object "
                                      "that no development pairing took as its request",
                              **account["rejected"]},
        "near_duplicates": {"note": "candidates turned away for a near-duplicate request (special_mix."
                                    "near_duplicate) or a template (special_mix.same_template), per attempt: "
                                    "dev, one near-duplicating a development item; train_cap, a training "
                                    f"candidate near-duplicating {TRAIN_NEAR_LIMIT} drawn training items; "
                                    "dev_template, a development candidate sharing its template with "
                                    f"{DEV_TEMPLATE_LIMIT} drawn development items; train_template, a "
                                    "training candidate sharing the template of a development item of its "
                                    "kind",
                            **account["near_duplicates"]},
        "irrelevant_pairing": account["irrelevant_pairing"],
        "cuts": account["cuts"],
        "gold_calls": {"note": "per source, the kept records whose first step is calls, by the first problem "
                               "special_mix.call_problem finds (ok: none); the builders use none of the "
                               "others",
                       **census},
        "vocabulary_exclusions": vocabulary_account,
        "reserved_names": account["reserved_names"],
        "dev_names": {"limit": account["dev_name_limit"], "avoided": account["dev_avoided_names"],
                      "deepexi_share": account["dev_deepexi_share"],
                      "deepexi_closed": account["dev_deepexi_closed"]},
        "records_used": account["records_used"],
        "revision": {"preregistration": "docs/m8-预登记.md, 修订 1 (the training items alone)",
                     "followers": account["followers"]["train"],
                     "incorrect_mix": account["incorrect_mix"], "irrelevant": account["irrelevant"],
                     "objects": account["objects"], "cells": account["cells"],
                     "settings": {"train_two_values": sm.TRAIN_TWO_VALUES,
                                  "optional_share": sm.OPTIONAL_SHARE,
                                  "described_share": sm.DESCRIBED_SHARE, "pairing_share": PAIRING_SHARE,
                                  "canned_share": CANNED_SHARE, "twin_share": TWIN_SHARE,
                                  "irrelevant_twin_share": IRRELEVANT_TWIN_SHARE,
                                  "context_share": CONTEXT_SHARE, "extra_style_share": sm.EXTRA_SHARE,
                                  "training_families": len(sm.families("incorrect", "en", "train")),
                                  "reasoning": sm.TEMPLATE_MODEL,
                                  "max_reasoning_tokens": sm.MAX_REASONING_TOKENS}},
        "dev_locks": dev_locks,
        "families": {split: families_used(items) for split, items in splits.items()},
        "prompt_choices": {split: prompt_choices(items) for split, items in splits.items()},
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "exclusions": {"drawn": {name: a.get("drawn", 0) for name, a in accounts.items()},
                       "screen": screened, "used_files": used_account},
        "acebench": {"function_names": len(bench.names), "description_grams": len(bench.grams),
                     "template_windows": {"words": len(bench.windows.words),
                                          "chars": len(bench.windows.chars)},
                     "files": bench.files},
        "benchmarks": {"files": getattr(bench.index, "files", []), "grams": len(bench.index.grams)},
        "sources": {name: {k: v for k, v in a.items() if k != "dropped"} for name, a in accounts.items()},
        "source_manifest": {"path": repo_relative(Path(args.source_manifest)),
                            "sha256": _sha256(Path(args.source_manifest))
                            if Path(args.source_manifest).is_file() else None},
        "checks": {"training_records": len(records), "dev_items": {s: len(v) for s, v in dev.items()},
                   "sidecar": sidecar},
        "outputs": {"out": _output(args.out, data["out"], len(records)),
                    "reasoning": {"path": repo_relative(args.reasoning_out), "records": len(rows),
                                  "sha256": digest, "bytes": size},
                    "dev_a": {**_output(args.dev_a, data["dev_a"], len(dev["a"])),
                              "lock": repo_relative(lock_path(args.dev_a))},
                    "dev_b": {**_output(args.dev_b, data["dev_b"], len(dev["b"])),
                              "lock": repo_relative(lock_path(args.dev_b))}},
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                             newline="\n")
    print(f"{args.out}: {len(records)} training records; {args.dev_a}: {len(dev['a'])} items; "
          f"{args.dev_b}: {len(dev['b'])} items; short {account['short'] or 'none'}; "
          f"rejected {account['rejected']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

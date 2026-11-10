"""Build the external zh-en glossary (data/glossary/zh-en.json) and its provenance from the two sources the
plan fixed before any coverage was looked at (docs/m7-本地提升计划.md, WP4), and measure its notes on the
training data.

Sources, in priority order (the union keeps the first source's English term where both have a Chinese term):

1. 机器之心《人工智能术语库》(AITD), a shallow clone of the default branch of AITD_URL (--aitd: the
   clone's folder). data/All.md holds every entry as a markdown table row
   "索引编号|英文术语|中文翻译|常用缩写|来源&扩展|备注|"; the letter files (data/A.md ...) and the section
   files (section/*.md) hold subsets of the same rows, which is checked: a row there that All.md lacks or
   holds otherwise stops the build.
2. Google's Machine Learning Glossary in Simplified Chinese (GOOGLE_URL), fetched with
   `tvly extract --format markdown --json` into extract.json (--google: its folder). A term is a level-2
   heading "{zh} ({en})". page.html, one plain GET of the same URL, is recorded beside it when present.

The downloads are untrusted data: each lives in its own folder outside the repository, nothing in them is
executed, and only this script's own parsers read them, as text. Run it isolated (-I) from the repository
root (the paths below are the 2026-10-30 build's):

    .venv/Scripts/python.exe -I -B -X utf8 scripts/build_glossary.py
        --aitd ../tools/downloads/glossary-2026-10-30/jiqizhixin
        --google ../tools/downloads/glossary-2026-10-30/google --fetched 2026-10-30

Parsing. A Chinese cell is split into variants at "/", ",", "、" and ";" (VARIANT; after NFKC, so "／", "，"
and "；" too), unless the English cell is itself a comma list ("Garbage In, Garbage Out" | "无用数据入、
无用数据出"): its items are the list's, not variants of one term, so the cell stays one term, and provenance
lists those rows (comma_list_rows). Parenthesised asides ("编码器-解码器（模型）") are dropped from both
cells first; a variant becomes its term form (research_agent.glossary.term_form) and is kept when it holds a
Han character ("AdaBoost" is no Chinese term). An English cell is NFKC with whitespace collapsed. A pair whose
English term is empty or holds a Han character or a separator of the note is not usable and is counted.

One English term per Chinese term, deterministically: within a source the first pair wins. AITD rows are taken
in index order (AITD-NNNNN), so the earliest entry wins; the later entries are mostly plurals and spellings
added in v3 ("Artificial Neurons", "Cross-Validation"). Google's headings are taken in page order. Across the
sources the first source wins (merge); a source's report counts its terms, the ones it added, the conflicts
(another English term, letter case aside) and the ones that agreed.

Filters, tuned on the training data only: the zh records of the --train files (data/raw/synth/synth-m6.jsonl
and synth-m7.jsonl); never a dev or test item, never an evaluation topic.

- min_chars (MIN_CHARS = 2): a term of fewer characters is dropped (a single character matches inside
  unrelated words: 图, 核, 边);
- blocking: a noted term that meets one of four rules is blocked, not dropped. A blocked term stays in the
  glossary with the English term null (research_agent.glossary): the matcher still takes its span,
  leftmost-longest like any term, and notes nothing, so a shorter term inside it is not noted there (with
  预训练 blocked, 训练数据 is not noted in "预训练数据", nor 学习 in "持续学习" with 持续学习 blocked);
  dropping the term would have exposed those. The rules, in the order a term's reasons list them (REASONS),
  with a term's counts on the training records (measure; a term noted in none of them has none):
  - disagreement (the stop rule): judged at least min_support times (MIN_SUPPORT = 5) and agreeing in at
    most max_agreement of them (MAX_AGREEMENT = 0.5);
  - generic (GENERIC_RECORDS = 3, the owner's rule, fixed before its effect was looked at): noted in at
    least generic_records zh records and agreeing in none of them, judged or not;
  - adjectival (the owner's rule): the Chinese term ends in 的, an adjective's form (线性的 = Linear),
    whatever the training records show;
  - split_short (the owner's rule): a term of two characters that comes only from splitting an AITD cell
    of several variants and is not a cell's first variant, when no training record agrees with it (冲突 =
    Bit Collisions, the second variant of AITD-02130 "字节碰撞/冲突"). Its candidates
    (split_short_candidates) are the later variants of such cells that are never the first variant of a
    cell (a cell's only one included) nor a term of another source; provenance lists every two-character
    variant of a cell of several variants with its cells and its outcome (split_variants).
  Blocking a term frees its place among a note's five, which can let another term be noted, so the rules
  are applied again until no noted term meets one. A term's reasons are the rules it met when it was
  blocked.

The setting (min_chars and min_support) is the sweep's (SWEEP_CHARS x SWEEP_SUPPORT, None meaning no stop
rule; the generic, adjectival and split_short rules apply in every line) with the most cross-fitted net
agreement: agreeing minus disagreeing judged pairs, the blocked terms learnt on one training file and
measured on the other, summed over both ways, so a setting is not chosen for fitting the records it was
learnt on. A tie goes to the setting that keeps more terms, then to the earlier line (best_setting). The
defaults are the 2026-10-30 rebuild's best, MIN_CHARS = 2 and MIN_SUPPORT = 5: at min_chars 2, min_support
2, 3 and 5 tie on net agreement and 5 keeps the most terms. provenance.json holds the whole sweep, the tie
(tied_best) and whether the setting built is the best (setting_is_best).

Measurement (measure). A unit is a user message of a zh record: its note is note_terms of the history up to
that message, which is what the renderer puts on every step that answers it (the note reads the latest user
message only), and its queries are the "query" arguments of the search_arxiv and search_notes calls in the
assistant messages after it, up to the next user message. A noted pair is judged when the unit has a query
holding a Latin letter (the teacher searched in English); it agrees when its English term is whole words of
such a query (agrees): both folded as the S2 scoring rule folds a query (NFKC, casefold, the hyphens U+002D
and U+2010 to U+2013 read as spaces, whitespace collapsed), the term with its spaces removed, its last word
as it is or in its regular plural, equals a part of the query with its spaces removed, and that part neither
starts right after nor ends right before an ASCII letter or digit (the matcher's Latin boundary). The regular
plural: +es after s, x, z or sh; +es or +s after ch, which may read as k (batches, epochs); y after a
consonant becomes ies (queries); +s otherwise. So "Pre-Training" agrees with "pretraining data",
"Fine-Tuning" with "finetuning" and "Word Embedding" with "word embeddings", while "Training Data" does not
agree with "pretraining data", nor "RAG" with "fragment", nor "Plan" with "planes". The measurement reports
zh records, units, the records and units with a note, pairs noted, judged and agreeing, per term too (with
the records noting it), and the records that keep their note under the training dropout
(train.samples.keeps_note). --sweep prints the build's sweep, in-sample and cross-fitted, and writes nothing.

Written to --out (data/glossary): zh-en.json (canonical JSON, sorted keys, one final newline; a blocked term
with null) and provenance.json (the sources with their URL, the clone's commit or the fetch date, the licence
text found and the SHA-256 of every raw file, the parse counts and the comma-list rows; the union; the
filters: the rules, the blocked terms with their reasons and counts, the two-character split variants with
their outcomes and the sweep; the measurement after the length filter and as each rule is added; the
glossary's SHA-256). The build holds no clock: the same inputs give the same bytes, and --check builds in
memory and compares with the files on disk. data/glossary/NOTICE.md states the sources' licences.

Exit codes: 0 done (or --check equal), 1 --check found a difference, 2 bad input (a missing file, a malformed
source, a licence line not found, a subset file that disagrees with All.md).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Iterator
from pathlib import Path
from typing import NamedTuple

from research_agent.canonical import canonical_json, split_lines
from research_agent.glossary import SEPARATORS, Glossary, has_han, note_terms, term_form
from research_agent.train.samples import NOTE_DROPOUT, keeps_note

AITD_URL = "https://github.com/jiqizhixin/Artificial-Intelligence-Terminology-Database"
GOOGLE_URL = "https://developers.google.com/machine-learning/glossary?hl=zh-cn"
TRAIN = ("data/raw/synth/synth-m6.jsonl", "data/raw/synth/synth-m7.jsonl")
OUT = "data/glossary"
GLOSSARY_FILE, PROVENANCE_FILE = "zh-en.json", "provenance.json"
MIN_CHARS, MIN_SUPPORT, MAX_AGREEMENT, GENERIC_RECORDS = 2, 5, 0.5, 3  # the sweep's best (module docstring)
SWEEP_CHARS, SWEEP_SUPPORT = (1, 2, 3, 4), (2, 3, 5, None)  # None: no stop rule
# the reasons a term is blocked, in the order a term's reasons are listed (module docstring: Filters)
DISAGREEMENT, GENERIC, ADJECTIVAL, SPLIT_SHORT = "disagreement", "generic", "adjectival", "split_short"
REASONS = (DISAGREEMENT, GENERIC, ADJECTIVAL, SPLIT_SHORT)
ADJECTIVE_ENDING, SPLIT_SHORT_CHARS = "的", 2
NOT_NOTED = {"noted": 0, "records": 0, "judged": 0, "agreeing": 0}  # a term's counts where it is never noted
SEARCH_TOOLS = ("search_arxiv", "search_notes")
AITD_CELLS = 7  # six cells and the empty one after the row's last bar
AITD_ROW = re.compile(r"^AITD-\d{5}\|")
ASIDE = re.compile(r"\([^()]*\)")  # after NFKC, which turns full-width parentheses into ASCII ones
VARIANT = re.compile("[/,;、／，；]")  # a Chinese cell's variant separators (NFKC: full-width to ASCII)
HYPHENS = re.compile("[-‐‑‒–]")  # the S2 rule's hyphens (WP1)
LATIN = re.compile("[A-Za-z]")
# a regular English plural (agreement_pattern): "es" after these endings, "es" or "s" after "ch" (batches,
# epochs), a consonant and y to "ies", "s" otherwise
SIBILANT_ENDINGS, CONSONANT_Y = ("s", "x", "z", "sh"), re.compile("[bcdfghjklmnpqrstvwxz]y$")
AITD_LICENCE_HEADING, GOOGLE_LICENCE_MARK, GOOGLE_UPDATED_MARK = "## 许可证", "知识共享署名", "最后更新时间"
DIFFERS, USAGE = 1, 2


class AitdRow(NamedTuple):
    """One AITD entry: its index (AITD-NNNNN), the English term and the Chinese variants (term forms)."""

    index: str
    en: str
    zh: tuple[str, ...]


def without_asides(text: str) -> str:
    """text with every parenthesised aside removed, innermost first."""
    while True:
        stripped = ASIDE.sub("", text)
        if stripped == text:
            return text
        text = stripped


def english(cell: str) -> str:
    return " ".join(without_asides(unicodedata.normalize("NFKC", cell)).split())


def is_comma_list(en: str) -> bool:
    """An English term that is a list of items ("Garbage In, Garbage Out"): it holds a comma (after NFKC)."""
    return "," in en


def chinese_variants(cell: str, en: str) -> tuple[str, ...]:
    """The Chinese terms of a cell that translates the English term en: split at VARIANT (after NFKC) unless
    en is a comma list, asides dropped, term forms, each holding a Han character, once each in the cell's
    order (module docstring: Parsing)."""
    text = without_asides(unicodedata.normalize("NFKC", cell))
    variants: list[str] = []
    for part in [text] if is_comma_list(en) else VARIANT.split(text):
        term = term_form(part)
        if term and has_han(term) and term not in variants:
            variants.append(term)
    return tuple(variants)


def parse_aitd(text: str, name: str) -> list[AitdRow]:
    """The entries of an AITD markdown table, in file order; ValueError naming the line of a row that does
    not have AITD_CELLS cells."""
    rows = []
    for number, line in enumerate(split_lines(text), 1):
        if not AITD_ROW.match(line):
            continue
        cells = line.split("|")
        if len(cells) != AITD_CELLS:
            raise ValueError(f"{name}: line {number} has {len(cells)} cells, not {AITD_CELLS}: {line[:80]!r}")
        en = english(cells[1])
        rows.append(AitdRow(cells[0], en, chinese_variants(cells[2], en)))
    return rows


def comma_list_rows(rows: list[AitdRow]) -> list[dict]:
    """The rows whose English term is a comma list, so their Chinese cell was not split (module docstring:
    Parsing), in index order."""
    ordered = sorted(rows, key=lambda row: int(row.index.removeprefix("AITD-")))
    return [{"index": row.index, "en": row.en, "zh": list(row.zh)}
            for row in ordered if is_comma_list(row.en)]


def usable(en: str) -> bool:
    """An English term the glossary can hold (research_agent.glossary: Validation)."""
    return bool(en) and not has_han(en) and not any(separator in en for separator in SEPARATORS)


def first_terms(pairs: Iterable[tuple[str, str]]) -> tuple[dict[str, str], int, int]:
    """(zh -> the English term of its first usable pair, how many Chinese terms had more than one English term
    (letter case aside), how many pairs were not usable)."""
    terms: dict[str, str] = {}
    seen: dict[str, set[str]] = defaultdict(set)
    unusable = 0
    for zh, en in pairs:
        if not usable(en):
            unusable += 1
            continue
        terms.setdefault(zh, en)
        seen[zh].add(en.casefold())
    return terms, sum(1 for english_terms in seen.values() if len(english_terms) > 1), unusable


def source_terms(rows: list[AitdRow]) -> tuple[dict[str, str], int]:
    """AITD's terms with the lowest index winning (module docstring), and the count of conflicts."""
    ordered = sorted(rows, key=lambda row: int(row.index.removeprefix("AITD-")))
    terms, conflicts, _ = first_terms((zh, row.en) for row in ordered for zh in row.zh)
    return terms, conflicts


def split_short_candidates(rows: list[AitdRow], other_terms: Collection[str] = ()) -> frozenset[str]:
    """The Chinese terms the split_short rule applies to (module docstring: Filters): two characters, a later
    variant of a cell of several variants, and neither a cell's first variant (a cell's only one included)
    nor a term of another source (other_terms)."""
    firsts = {row.zh[0] for row in rows if row.zh}
    return frozenset(zh for row in rows for zh in row.zh[1:]
                     if len(zh) == SPLIT_SHORT_CHARS and zh not in firsts and zh not in other_terms)


def split_variants(rows: list[AitdRow], candidates: Collection[str], glossary: Glossary, short: list[str],
                   blocked: dict[str, dict]) -> list[dict]:
    """Every two-character variant of an AITD cell of several variants, in the order of its first such cell
    (index order) and its place there: the source's English term (source_terms), those cells (index, English
    term, variants, its 1-based place), the rows that give it alone, whether the split_short rule applies to
    it (candidates) and its outcome: noted, blocked (with the reasons), dropped by the length filter or absent
    (no usable English term)."""
    ordered = sorted(rows, key=lambda row: int(row.index.removeprefix("AITD-")))
    terms, _ = source_terms(rows)
    entries: dict[str, dict] = {}
    for row in ordered:
        for place, zh in enumerate(row.zh, 1):
            if len(row.zh) > 1 and len(zh) == SPLIT_SHORT_CHARS:
                entry = entries.setdefault(zh, {"zh": zh, "en": terms.get(zh), "cells": [],
                                                "alone_in": []})
                entry["cells"].append({"index": row.index, "en": row.en, "zh": list(row.zh),
                                       "variant": place})
    for row in ordered:
        if len(row.zh) == 1 and row.zh[0] in entries:
            entries[row.zh[0]]["alone_in"].append(row.index)
    for zh, entry in entries.items():
        entry["candidate"] = zh in candidates
        if zh in glossary:
            entry["outcome"] = "noted"
        elif zh in blocked:
            entry["outcome"], entry["reasons"] = "blocked", blocked[zh]["reasons"]
        else:
            entry["outcome"] = "dropped" if zh in short else "absent"
    return list(entries.values())


def check_subsets(folder: Path, rows: list[AitdRow]) -> dict[str, int]:
    """The rows of every letter and section file, each of which must be a row of All.md with the same terms;
    {file: rows}. ValueError naming the first row that is not."""
    known = {row.index: row for row in rows}
    checked = {}
    files = [p for p in sorted((folder / "data").glob("*.md")) if p.name != "All.md"]
    for path in files + sorted((folder / "section").glob("*.md")):
        name = path.relative_to(folder).as_posix()
        subset = parse_aitd(path.read_text(encoding="utf-8"), name)
        for row in subset:
            if known.get(row.index) != row:
                raise ValueError(f"{name}: {row.index} is not the row data/All.md holds: {row} vs "
                                 f"{known.get(row.index)}")
        checked[name] = len(subset)
    return checked


def heading_pairs(heading: str) -> list[tuple[str, str]]:
    """The (zh, en) pairs of a Google glossary heading "{zh} ({en})": the English term is the last balanced
    parenthesised group, the Chinese terms what precedes it (chinese_variants); [] for any other heading."""
    text = unicodedata.normalize("NFKC", heading).strip()
    if not text.endswith(")") or not has_han(text):
        return []
    depth = 0
    for position in range(len(text) - 1, -1, -1):
        depth += {")": 1, "(": -1}.get(text[position], 0)
        if depth == 0:
            break
    else:
        return []
    en = english(text[position + 1:-1])
    return [(zh, en) for zh in chinese_variants(text[:position], en)] if en and not has_han(en) else []


def google_content(text: str) -> str:
    """The extracted page text of a tvly extract JSON; ValueError when it holds no result."""
    value = json.loads(text)
    results = value.get("results") if isinstance(value, dict) else None
    if not results or not isinstance(results[0].get("raw_content"), str):
        failed = value.get("failed_results") if isinstance(value, dict) else None
        raise ValueError(f"the extract holds no results (failed_results: {failed!r})")
    return results[0]["raw_content"]


def parse_google(text: str) -> tuple[list[tuple[str, str]], int]:
    """(the (zh, en) pairs of the extract's level-2 headings in page order, how many level-2 headings)."""
    headings = [line[3:] for line in split_lines(google_content(text)) if line.startswith("## ")]
    return [pair for heading in headings for pair in heading_pairs(heading)], len(headings)


def merge(sources: list[tuple[str, dict[str, str]]]) -> tuple[dict[str, str], list[dict]]:
    """The union of the sources' terms, the first source winning, and per source its counts (module
    docstring: One English term per Chinese term)."""
    merged: dict[str, str] = {}
    report = []
    for name, terms in sources:
        added = conflicts = same = 0
        for zh, en in terms.items():
            if zh not in merged:
                merged[zh] = en
                added += 1
            elif merged[zh].casefold() == en.casefold():
                same += 1
            else:
                conflicts += 1
        report.append({"source": name, "terms": len(terms), "added": added, "conflicts": conflicts,
                       "same": same})
    return merged, report


def length_filter(terms: dict[str, str], *, min_chars: int) -> tuple[dict[str, str], list[str]]:
    """(the terms of at least min_chars characters, the dropped Chinese terms sorted)."""
    kept = {zh: en for zh, en in terms.items() if len(zh) >= min_chars}
    return kept, sorted(zh for zh in terms if zh not in kept)


def search_queries(message: dict) -> list[str]:
    """The "query" arguments of the search_arxiv and search_notes calls of an assistant message."""
    queries = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        if function.get("name") not in SEARCH_TOOLS:
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                continue
        if isinstance(arguments, dict) and isinstance(arguments.get("query"), str):
            queries.append(arguments["query"])
    return queries


def units(record: dict) -> Iterator[tuple[list[dict], list[str]]]:
    """(the history up to and with a user message, the queries of the assistant messages answering it) for
    every user message of the record (module docstring: Measurement)."""
    messages = record["messages"]
    for position, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        queries: list[str] = []
        for later in messages[position + 1:]:
            if later.get("role") == "user":
                break
            if later.get("role") == "assistant":
                queries += search_queries(later)
        yield messages[:position + 1], queries


def fold(text: str) -> str:
    """A term or a query as the S2 scoring rule compares them: NFKC, casefold, hyphens as spaces, whitespace
    collapsed."""
    return " ".join(HYPHENS.sub(" ", unicodedata.normalize("NFKC", text).casefold()).split())


@functools.cache
def agreement_pattern(en: str) -> re.Pattern[str] | None:
    """What a folded query holds where the English term agrees with it (module docstring: Measurement): the
    term folded with its spaces removed, one optional space allowed between any two of its characters, its
    last word as it is or in its regular plural, not right after or right before an ASCII letter or digit;
    None for a term that folds to nothing."""
    words = fold(en).split()
    if not words:
        return None
    squeezed, last = "".join(words), words[-1]
    if CONSONANT_Y.search(last):  # query: queries
        body, ending = squeezed[:-1], " ?(?:y|ies)"
    elif last.endswith("ch"):  # batch: batches; epoch: epochs
        body, ending = squeezed, "(?:e?s)?"
    elif last.endswith(SIBILANT_ENDINGS):  # bias: biases; box: boxes
        body, ending = squeezed, "(?:es)?"
    else:  # embedding: embeddings
        body, ending = squeezed, "s?"
    return re.compile("(?<![a-z0-9])" + " ?".join(re.escape(character) for character in body) + ending
                      + "(?![a-z0-9])")


def agrees(en: str, queries: list[str]) -> bool:
    """True when the English term is whole words of one of the queries, both folded, spaces and hyphens
    aside (module docstring: Measurement)."""
    pattern = agreement_pattern(en)
    return pattern is not None and any(pattern.search(fold(query)) for query in queries)


def measure(glossary: Glossary, records: list[dict]) -> dict:
    """The notes' measurement over the zh records (module docstring: Measurement)."""
    zh_records = [r for r in records if r.get("lang") == "zh"]
    terms: dict[str, dict[str, int]] = defaultdict(lambda: {"noted": 0, "records": 0, "judged": 0,
                                                            "agreeing": 0})
    noted: set[str] = set()
    counts = dict.fromkeys(("units", "units_noted", "pairs", "pairs_judged", "pairs_agreeing"), 0)
    for record in zh_records:
        in_record: set[str] = set()
        for history, queries in units(record):
            counts["units"] += 1
            pairs = note_terms(history, glossary)
            if not pairs:
                continue
            counts["units_noted"] += 1
            noted.add(record["id"])
            english_queries = [q for q in queries if LATIN.search(q)]
            for zh, en in pairs:
                term = terms[zh]
                in_record.add(zh)
                counts["pairs"] += 1
                term["noted"] += 1
                if english_queries:
                    counts["pairs_judged"] += 1
                    term["judged"] += 1
                    if agrees(en, english_queries):
                        counts["pairs_agreeing"] += 1
                        term["agreeing"] += 1
        for zh in in_record:
            terms[zh]["records"] += 1
    judged = counts["pairs_judged"]
    return {"records": len(zh_records), "records_noted": len(noted),
            "records_noted_after_dropout": sum(1 for rid in noted if keeps_note(rid, NOTE_DROPOUT)), **counts,
            "agreement": round(counts["pairs_agreeing"] / judged, 4) if judged else None,
            "terms": {zh: terms[zh] for zh in sorted(terms)}}


def is_adjectival(zh: str) -> bool:
    """A Chinese term ending in 的, an adjective's form (线性的 = Linear, 基于梯度的 = Gradient-Based)."""
    return zh.endswith(ADJECTIVE_ENDING)


def block_reasons(zh: str, counts: dict[str, int] | None, *, min_support: int | None, max_agreement: float,
                  generic_records: int | None, adjectival: bool, split_short: Collection[str]) -> list[str]:
    """The rules a noted term meets (module docstring: Filters), in REASONS order, with its counts on the
    training records (None: noted in none of them, so only the adjectival and the split_short rule can
    apply); None, False or no candidates turns a rule off."""
    reasons = []
    if counts is not None:
        if (min_support is not None and counts["judged"] >= min_support
                and counts["agreeing"] <= max_agreement * counts["judged"]):
            reasons.append(DISAGREEMENT)
        if generic_records is not None and counts["records"] >= generic_records and counts["agreeing"] == 0:
            reasons.append(GENERIC)
    if adjectival and is_adjectival(zh):
        reasons.append(ADJECTIVAL)
    if zh in split_short and (counts is None or counts["agreeing"] == 0):
        reasons.append(SPLIT_SHORT)
    return reasons


def blocked_terms(glossary: Glossary, records: list[dict], *, min_support: int | None, max_agreement: float,
                  generic_records: int | None, adjectival: bool = False,
                  split_short: Collection[str] = frozenset()) -> tuple[dict[str, dict], Glossary]:
    """(the blocked terms with their counts and reasons when blocked, in the order blocked, the glossary with
    them blocked): the module docstring's rules, applied until no noted term meets one. A term never noted on
    the records has the counts NOT_NOTED."""
    blocked: dict[str, dict] = {}
    current = glossary
    while True:
        measured = measure(current, records)["terms"]
        new = {}
        for zh in sorted(current):
            reasons = block_reasons(zh, measured.get(zh), min_support=min_support,
                                    max_agreement=max_agreement, generic_records=generic_records,
                                    adjectival=adjectival, split_short=split_short)
            if reasons:
                new[zh] = {**measured.get(zh, NOT_NOTED), "reasons": reasons}
        if not new:
            return blocked, current
        blocked.update(new)
        current = Glossary({**current.entries(), **dict.fromkeys(new)})


def blocked_by_reason(blocked: dict[str, dict]) -> dict:
    """How many blocked terms met each rule, and how many met each combination of rules (its reasons joined
    by "+")."""
    reasons = [counts["reasons"] for counts in blocked.values()]
    return {**{reason: sum(reason in r for r in reasons) for reason in REASONS},
            "combinations": dict(sorted(Counter("+".join(r) for r in reasons).items()))}


def filtered(terms: dict[str, str], records: list[dict], *, min_chars: int, min_support: int | None,
             max_agreement: float, generic_records: int | None, adjectival: bool = False,
             split_short: Collection[str] = frozenset()) -> tuple[Glossary, list[str], dict[str, dict]]:
    """(the glossary after the length filter and the blocking, the terms the length filter dropped, the
    blocked terms)."""
    kept, short = length_filter(terms, min_chars=min_chars)
    blocked, glossary = blocked_terms(Glossary(kept), records, min_support=min_support,
                                      max_agreement=max_agreement, generic_records=generic_records,
                                      adjectival=adjectival, split_short=split_short)
    return glossary, short, blocked


def write_glossary(terms: dict[str, str | None], path: Path) -> str:
    """Write terms as canonical JSON with one final newline; its SHA-256."""
    data = glossary_bytes(terms)
    Path(path).write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def glossary_bytes(terms: dict[str, str | None]) -> bytes:
    return (canonical_json(terms) + "\n").encode("utf-8")


def raw_files(folder: Path) -> list[dict]:
    """Every file under folder but the clone's .git folder, by relative path: path, SHA-256 and size."""
    files = sorted((p.relative_to(folder).as_posix(), p) for p in folder.rglob("*")
                   if p.is_file() and p.relative_to(folder).parts[0] != ".git")
    return [{"path": name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
            for name, p in files]


def clone_commit(folder: Path) -> str:
    """The commit a clone's HEAD names, read from its .git folder (HEAD, the ref file or packed-refs); git is
    not run on the download."""
    git = Path(folder) / ".git"
    head = (git / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    if (git / ref).is_file():
        return (git / ref).read_text(encoding="utf-8").strip()
    for line in split_lines((git / "packed-refs").read_text(encoding="utf-8")):
        commit, _, name = line.partition(" ")
        if name == ref:
            return commit
    raise ValueError(f"{git}: no commit for {ref}")


def line_after(text: str, heading: str) -> str:
    """The first non-blank line after the line that equals heading; ValueError when there is none."""
    lines = split_lines(text)
    if heading in lines:
        following = [line for line in lines[lines.index(heading) + 1:] if line.strip()]
        if following:
            return following[0].strip()
    raise ValueError(f"no line after {heading!r}")


def line_with(text: str, mark: str) -> str:
    """The first line holding mark; ValueError when none does."""
    for line in split_lines(text):
        if mark in line:
            return line.strip()
    raise ValueError(f"no line holds {mark!r}")


def page_evidence(path: Path) -> dict | None:
    """What the plain GET's HTML shows about the page's language: the html lang attribute and its level-2
    headings with and without a Han character; None when there is no page.html."""
    if not path.is_file():
        return None
    html = path.read_text(encoding="utf-8")
    lang = re.search(r"<html[^>]*\blang=\"([^\"]*)\"", html)
    headings = re.findall(r"<h2[^>]*>(.*?)</h2>", html, flags=re.DOTALL)
    return {"html_lang": lang.group(1) if lang else None, "h2_headings": len(headings),
            "h2_with_han": sum(1 for h in headings if has_han(h))}


def load_training(paths: list[str]) -> list[tuple[str, list[dict]]]:
    """(path, records) of every training file: one JSON object per non-blank line."""
    return [(path, [json.loads(line) for line in split_lines(Path(path).read_text(encoding="utf-8"))
                    if line.strip()]) for path in paths]


def relative(path: Path) -> str:
    """path relative to the working directory (the repository root), with forward slashes."""
    return Path(os.path.relpath(Path(path).resolve(), Path.cwd())).as_posix()


class Sources(NamedTuple):
    """What read_sources gives: the union of the sources' terms, merge's report, what provenance records of
    each source's parse, AITD's rows and the split_short rule's candidates (split_short_candidates)."""

    union: dict[str, str]
    report: list[dict]
    parsed: dict[str, dict]
    rows: list[AitdRow]
    split_short: frozenset[str]


def read_sources(aitd_dir: Path, google_dir: Path) -> Sources:
    """The sources read and merged (Sources)."""
    rows = parse_aitd((aitd_dir / "data" / "All.md").read_text(encoding="utf-8"), "data/All.md")
    subsets = check_subsets(aitd_dir, rows)
    aitd_terms, aitd_conflicts = source_terms(rows)
    _, _, aitd_unusable = first_terms((zh, row.en) for row in rows for zh in row.zh)
    google_pairs, headings = parse_google((google_dir / "extract.json").read_text(encoding="utf-8"))
    google_terms, google_conflicts, google_unusable = first_terms(google_pairs)
    union, report = merge([("aitd", aitd_terms), ("google", google_terms)])
    parsed = {"aitd": {"file": "data/All.md", "rows": len(rows),
                       "rows_with_a_chinese_term": sum(1 for row in rows if row.zh),
                       "pairs": sum(len(row.zh) for row in rows), "unusable_pairs": aitd_unusable,
                       "terms": len(aitd_terms), "conflicts": aitd_conflicts, "subset_files": subsets,
                       "comma_list_rows": comma_list_rows(rows)},
              "google": {"file": "extract.json", "level2_headings": headings, "pairs": len(google_pairs),
                         "unusable_pairs": google_unusable, "terms": len(google_terms),
                         "conflicts": google_conflicts, "page_html": page_evidence(google_dir / "page.html")}}
    return Sources(union, report, parsed, rows, split_short_candidates(rows, other_terms=google_terms))


def fixed_rules(args: argparse.Namespace, sources: Sources) -> dict:
    """The rules every setting applies, the build's and each sweep line's (filtered's keywords but min_chars
    and min_support): the stop rule's max_agreement, the generic rule, the adjectival rule and the split_short
    rule with its candidates."""
    return {"max_agreement": args.max_agreement, "generic_records": args.generic_records, "adjectival": True,
            "split_short": sources.split_short}


def cross_fitted(union: dict[str, str], training: list[tuple[str, list[dict]]], **rules: object) -> dict:
    """The filters' measurement cross-fitted (module docstring: the setting): the blocked terms learnt on one
    training file and measured on another, summed over every ordered pair of files, with the agreement and
    the net agreement (agreeing minus disagreeing judged pairs). rules are filtered's keywords."""
    crossed = dict.fromkeys(("records", "records_noted", "pairs_judged", "pairs_agreeing"), 0)
    for k, (_, learn) in enumerate(training):
        for j, (_, held) in enumerate(training):
            if j != k:
                learnt, _, _ = filtered(union, learn, **rules)
                held_out = measure(learnt, held)
                for key in crossed:
                    crossed[key] += held_out[key]
    judged, agreeing = crossed["pairs_judged"], crossed["pairs_agreeing"]
    return {**crossed, "agreement": round(agreeing / judged, 4) if judged else None,
            "net_agreement": agreeing - (judged - agreeing)}


def sweep(union: dict[str, str], training: list[tuple[str, list[dict]]], *, max_agreement: float,
          generic_records: int | None, adjectival: bool = False,
          split_short: Collection[str] = frozenset()) -> list[dict]:
    """The measurement of every SWEEP_CHARS x SWEEP_SUPPORT setting (module docstring: the setting), the other
    rules applied in each: in-sample over all the training files, and cross_fitted."""
    records = [record for _, file_records in training for record in file_records]
    lines = []
    for min_chars in SWEEP_CHARS:
        for min_support in SWEEP_SUPPORT:
            rules = {"min_chars": min_chars, "min_support": min_support, "max_agreement": max_agreement,
                     "generic_records": generic_records, "adjectival": adjectival, "split_short": split_short}
            glossary, _, blocked = filtered(union, records, **rules)
            inside = measure(glossary, records)
            lines.append({"min_chars": min_chars, "min_support": min_support, "terms": len(glossary),
                          "blocked": len(blocked),
                          "in_sample": {key: inside[key] for key in ("records", "records_noted", "pairs",
                                                                     "pairs_judged", "pairs_agreeing",
                                                                     "agreement")},
                          "cross_fitted": cross_fitted(union, training, **rules)})
    return lines


def best_setting(lines: list[dict]) -> dict:
    """The sweep's setting with the most cross-fitted net agreement; a tie goes to the one keeping more terms,
    then to the earlier line."""
    best = max(lines, key=lambda line: (line["cross_fitted"]["net_agreement"], line["terms"],
                                        -lines.index(line)))
    return {"min_chars": best["min_chars"], "min_support": best["min_support"]}


def tied_best(lines: list[dict]) -> list[dict]:
    """The sweep's settings with the most cross-fitted net agreement, in sweep order: the ones best_setting's
    tie-break chooses among."""
    top = max(line["cross_fitted"]["net_agreement"] for line in lines)
    return [{"min_chars": line["min_chars"], "min_support": line["min_support"]}
            for line in lines if line["cross_fitted"]["net_agreement"] == top]


def build(args: argparse.Namespace) -> tuple[bytes, bytes, dict]:
    """(zh-en.json's bytes, provenance.json's bytes, the provenance itself)."""
    aitd_dir, google_dir = Path(args.aitd), Path(args.google)
    sources = read_sources(aitd_dir, google_dir)
    union, union_report, parsed = sources.union, sources.report, sources.parsed
    readme = (aitd_dir / "README.md").read_text(encoding="utf-8")
    content = google_content((google_dir / "extract.json").read_text(encoding="utf-8"))
    training = load_training(args.train)
    records = [record for _, file_records in training for record in file_records]
    setting = {"min_chars": args.min_chars, "min_support": args.min_support}
    fixed = fixed_rules(args, sources)
    rules = {**setting, **fixed}
    glossary, short, blocked = filtered(union, records, **rules)
    entries = dict(sorted(glossary.entries().items()))
    data = glossary_bytes(entries)
    # the measurement as the filters are added in turn: the length filter, the stop rule, the generic, the
    # adjectival and the split_short rule (final)
    later_off = {"stop_rule": {"generic_records": None, "adjectival": False, "split_short": frozenset()},
                 "generic_rule": {"adjectival": False, "split_short": frozenset()},
                 "adjectival_rule": {"split_short": frozenset()}}
    stages = {"union": measure(Glossary(union), records),
              "min_chars": measure(Glossary(length_filter(union, min_chars=args.min_chars)[0]), records),
              **{stage: measure(filtered(union, records, **{**rules, **off})[0], records)
                 for stage, off in later_off.items()},
              "final": measure(glossary, records)}
    lines = sweep(union, training, **fixed)
    best = best_setting(lines)
    crossed = next((line["cross_fitted"] for line in lines
                    if {"min_chars": line["min_chars"], "min_support": line["min_support"]} == setting), None)
    if crossed is None:  # a setting outside the sweep's grid
        crossed = cross_fitted(union, training, **rules)
    added = {entry["source"]: entry["added"] for entry in union_report}
    google_parse = parsed["google"]
    google_remark = None
    if not google_parse["terms"]:
        page = google_parse["page_html"] or {}
        google_remark = (f"contributes nothing: the zh-cn page's body is the English glossary (html lang "
                         f"{page.get('html_lang')!r}, {page.get('h2_with_han')} of {page.get('h2_headings')} "
                         f"h2 headings hold a Han character; {google_parse['level2_headings']} level-2 "
                         "headings in the extract, none of the form {zh} ({en}))")
    provenance = {
        "glossary": {"path": f"{Path(args.out).as_posix()}/{GLOSSARY_FILE}",
                     "sha256": hashlib.sha256(data).hexdigest(), "terms": len(glossary),
                     "blocked": len(glossary.blocked), "entries": len(entries),
                     "builder": "scripts/build_glossary.py",
                     "format": "one canonical JSON object (sorted keys) of Chinese term -> English "
                               "term, null for a blocked term (matched, its span taken, never noted), "
                               "read by research_agent.glossary.load_glossary",
                     "pick_rule": "within a source the first pair wins: AITD rows in index order (the lowest "
                                  "AITD-NNNNN), Google headings in page order; across sources the first "
                                  "source wins"},
        "sources": [
            {"name": "aitd",
             "title": "机器之心《人工智能术语库》(Artificial Intelligence Terminology Database)",
             "url": AITD_URL, "fetched": args.fetched, "method": "git clone --depth 1 (default branch)",
             "commit": clone_commit(aitd_dir), "location": relative(aitd_dir),
             "licence": {"name": "CC BY-NC-SA 4.0",
                         "license_file_first_line": split_lines(
                             (aitd_dir / "LICENSE").read_text(encoding="utf-8"))[0].strip(),
                         "readme": line_after(readme, AITD_LICENCE_HEADING),
                         "private_research_use": "allowed: non-commercial use with attribution; adapted "
                                                 "material shared on must carry the same licence"},
             "files": raw_files(aitd_dir), "parsed": parsed["aitd"], "used": True, "added": added["aitd"]},
            {"name": "google", "title": "Google Machine Learning Glossary (机器学习术语表), zh-cn",
             "url": GOOGLE_URL, "fetched": args.fetched,
             "method": "tvly extract --format markdown --json (extract.json); page.html: one plain HTTP GET "
                       "of the same URL, a cross-check of the extract",
             "location": relative(google_dir),
             "licence": {"name": "CC BY 4.0 (content), Apache 2.0 (code samples)",
                         "page": line_with(content, GOOGLE_LICENCE_MARK),
                         "private_research_use": "allowed: any use with attribution"},
             "page_updated": line_with(content, GOOGLE_UPDATED_MARK),
             "files": raw_files(google_dir), "parsed": google_parse, "used": True, "added": added["google"],
             "remark": google_remark},
        ],
        "parse_rule": "a Chinese cell is split into variants at / , 、 ; (after NFKC, so ／ ， ； "
                      "too) unless its English cell is a comma list "
                      "(sources[aitd].parsed.comma_list_rows): then it stays one term",
        "union": union_report,
        "filters": {"min_chars": args.min_chars, "dropped_short": short,
                    "stop_rule": {"min_support": args.min_support, "max_agreement": args.max_agreement,
                                  "reading": "a noted term judged at least min_support times on the training "
                                             "records whose English term agrees in at most max_agreement of "
                                             "them is blocked"},
                    "generic_rule": {"min_records": args.generic_records,
                                     "reading": "a term noted in at least min_records zh training records "
                                                "whose English term agrees in none of them (judged or not) "
                                                "is blocked; the owner's rule, fixed before its effect was "
                                                "looked at"},
                    "adjectival_rule": {"ending": ADJECTIVE_ENDING,
                                        "reading": "a term whose Chinese term ends in 的 (an adjective's "
                                                   "form) is blocked whatever the training records show; "
                                                   "the owner's rule"},
                    "split_short_rule": {"chars": SPLIT_SHORT_CHARS,
                                         "candidates": sorted(sources.split_short),
                                         "reading": "a term of two characters that comes only from splitting "
                                                    "an AITD cell of several variants and is never a cell's "
                                                    "first variant (nor a cell's only one, nor a term of "
                                                    "another source) is blocked unless a training record "
                                                    "agrees with it; the owner's rule"},
                    "blocking": "a blocked term stays in the glossary with the English term null: its span "
                                "is taken and nothing is noted, so no shorter term inside it is noted there; "
                                "the rules are applied again until no noted term meets one, and a term's "
                                "reasons are the rules it met when it was blocked",
                    "blocked": {zh: {"en": union[zh], **counts} for zh, counts in blocked.items()},
                    "blocked_by_reason": blocked_by_reason(blocked),
                    "blocked_terms_by_reason": {reason: sorted(zh for zh, counts in blocked.items()
                                                               if reason in counts["reasons"])
                                                for reason in REASONS},
                    "two_character_blocked_by_reason": blocked_by_reason(
                        {zh: counts for zh, counts in blocked.items() if len(zh) == 2}),
                    "split_variants": split_variants(sources.rows, sources.split_short, glossary, short,
                                                     blocked),
                    "selection": "the sweep's setting with the most cross-fitted net agreement (agreeing "
                                 "minus disagreeing judged pairs, the blocked terms learnt on one training "
                                 "file and measured on the other; the generic, adjectival and split_short "
                                 "rules applied in every setting); a tie goes to the setting that keeps more "
                                 "terms, then to the earlier line",
                    "best": best,
                    "setting_is_best": best == setting,
                    "tied_best": tied_best(lines),
                    "setting_ties_best": setting in tied_best(lines),
                    "sweep": lines},
        "training": {"files": [{"path": Path(path).as_posix(),
                                "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                "records": len(file_records),
                                "zh_records": sum(1 for r in file_records if r.get("lang") == "zh")}
                               for path, file_records in training],
                     "dropout": NOTE_DROPOUT,
                     "stages": "the measurement of the union, after the length filter, then as each blocking "
                               "rule is added in turn: the stop rule, the generic rule, the adjectival rule "
                               "and the split_short rule (final)",
                     "measurement": {stage: {k: v for k, v in m.items() if k != "terms"}
                                     for stage, m in stages.items()},
                     "cross_fitted": crossed,
                     "final_terms": stages["final"]["terms"]},
    }
    text = json.dumps(provenance, ensure_ascii=False, indent=2) + "\n"
    return data, text.encode("utf-8"), provenance


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build data/glossary/zh-en.json and its provenance.")
    ap.add_argument("--aitd", required=True, help="the shallow clone of the AITD repository")
    ap.add_argument("--google", required=True, help="the folder holding the Google glossary's extract.json")
    ap.add_argument("--fetched", required=True, help="the date the sources were fetched (YYYY-MM-DD)")
    ap.add_argument("--train", nargs="+", default=list(TRAIN),
                    help="the training files the filters are tuned on (their zh records)")
    ap.add_argument("--out", default=OUT, help=f"the output folder (default {OUT})")
    ap.add_argument("--min-chars", type=int, default=MIN_CHARS)
    ap.add_argument("--min-support", type=int, default=MIN_SUPPORT)
    ap.add_argument("--max-agreement", type=float, default=MAX_AGREEMENT)
    ap.add_argument("--generic-records", type=int, default=GENERIC_RECORDS,
                    help=f"the generic rule's records (default {GENERIC_RECORDS})")
    ap.add_argument("--sweep", action="store_true", help="print the sweep of filter settings, write nothing")
    ap.add_argument("--check", action="store_true", help="compare a build with the files in --out")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.sweep:  # the build's own sweep: the same sources, training files and fixed rules
            sources = read_sources(Path(args.aitd), Path(args.google))
            for line in sweep(sources.union, load_training(args.train), **fixed_rules(args, sources)):
                print(json.dumps(line, ensure_ascii=False))
            return 0
        data, provenance_data, provenance = build(args)
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return USAGE
    out = Path(args.out)
    if args.check:
        same = [(out / name).is_file() and (out / name).read_bytes() == content
                for name, content in ((GLOSSARY_FILE, data), (PROVENANCE_FILE, provenance_data))]
        print(f"{GLOSSARY_FILE}: {'equal' if same[0] else 'DIFFERS'}; {PROVENANCE_FILE}: "
              f"{'equal' if same[1] else 'DIFFERS'}")
        return 0 if all(same) else DIFFERS
    out.mkdir(parents=True, exist_ok=True)
    (out / GLOSSARY_FILE).write_bytes(data)
    (out / PROVENANCE_FILE).write_bytes(provenance_data)
    training, filters = provenance["training"], provenance["filters"]
    final, crossed = training["measurement"]["final"], training["cross_fitted"]
    for source in provenance["sources"]:
        print(f"{source['name']}: {source['parsed']['terms']} terms, {source['added']} added to the union "
              f"({source['licence']['name']})")
    by_reason, glossary = filters["blocked_by_reason"], provenance["glossary"]
    reasons = ", ".join(f"{reason} {by_reason[reason]}" for reason in REASONS)
    print(f"union {sum(entry['added'] for entry in provenance['union'])} terms; min_chars {args.min_chars} "
          f"drops {len(filters['dropped_short'])}; {len(filters['blocked'])} blocked ({reasons}; "
          f"combinations {by_reason['combinations']}); glossary {glossary['terms']} noted and "
          f"{glossary['blocked']} blocked terms, sha256 {glossary['sha256']}")
    print(f"setting min_chars {args.min_chars}, min_support {args.min_support}: the sweep's best is "
          f"{filters['best']} ({'this one' if filters['setting_is_best'] else 'ANOTHER ONE'}); tied on "
          f"cross-fitted net agreement: {filters['tied_best']} "
          f"({'this one among them' if filters['setting_ties_best'] else 'NOT this one'})")
    print(f"training: {final['records']} zh records, {final['records_noted']} noted "
          f"({final['records_noted_after_dropout']} after the {NOTE_DROPOUT} dropout); pairs "
          f"{final['pairs']}, judged {final['pairs_judged']}, agreeing {final['pairs_agreeing']} "
          f"({final['agreement']}); cross-fitted {crossed['pairs_agreeing']} of {crossed['pairs_judged']} "
          f"({crossed['agreement']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

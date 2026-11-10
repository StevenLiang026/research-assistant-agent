"""The 13-gram overlap check of the training data against the BFCL and ACEBench questions (docs/plan.md,
去重与防泄漏: 训练数据与 BFCL、ACEBench 题目做 13-gram 重叠检查); scripts/check_benchmark_overlap.py is the
command line, and scripts/train.py skips the records the report names (load_skips).

Grams. A text is normalize_text-ed and lower-cased; N is 13, and every text gives grams of both kinds:
- word N-grams over its Latin-script word tokens: the maximal runs of characters that are neither whitespace
  nor CJK (CJK unified ideographs, kana, hangul), so CJK characters separate words like spaces do and
  punctuation stays attached to its word; for a text without CJK these are its whitespace-separated words;
- character N-grams over its non-space characters, keeping only the windows in which CJK characters are the
  majority (at least 7 of 13), so a Latin fragment inside Chinese text (an e-mail address, a phone number, an
  English name with a few Chinese characters around it) gives no character gram; a text without CJK gives
  none. A first rule kept every window holding one CJK character; on the public data it flagged 21 and then
  3 glaive_zh records on such fragments, all false positives (2026-10-15).
A record overlaps a benchmark question when they share any gram of either kind, both sides computed the same
way. The two kinds never match each other (a word gram holds spaces and no CJK, a character gram holds a CJK
character and no space). Fewer than N words or characters give no gram of that kind.

What is read of the benchmarks.
- BFCL: every file BFCL_v4_*.json of the bfcl_eval data folder, JSON lines with "id" and "question"; the
  question is a list of turns, each a list of {"role", "content"} messages (a flat message list is read
  too), and the content of every message with role "user" is one question text. A file that is not JSON lines
  of such entries (BFCL_v4_format_sensitivity.json is one JSON object listing ids) gives no entry and is still
  listed with its hash. The possible_answer, multi_turn_func_doc and memory_prereq_conversation folders hold
  answers, function docs and memory set-up, not questions, and are not read.
- ACEBench: the files data_en/data_normal_*.json, data_en/data_special_*.json and the same two patterns under
  data_zh of the data_all folder, JSON lines with "id" and "question"; the question is one string of lines
  "user: ..." and "system: ..." (lines without such a prefix continue the turn before), and every user turn is
  one question text (a question without any prefix is one user turn). The agent categories
  (data_agent_*.json) and possible_answer/ are outside the plan's normal and special categories and are not
  read.

Training records are read with research_agent.train.samples.load_records; the user messages of a record (a
public sample, an agent-history record, a transcript or an API history) are compared with the benchmark grams.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..agent.types import Transcript
from ..canonical import split_lines
from ..normalize import normalize_text
from ..train.samples import load_records

N = 13
MATCHES_KEPT = 5  # matching benchmark entries listed per record; matches_total counts them all
ENTRIES_PER_GRAM = 3  # benchmark entries remembered per gram
BFCL_PATTERN = "BFCL_v4_*.json"
ACEBENCH_PATTERNS = ("data_en/data_normal_*.json", "data_en/data_special_*.json",
                     "data_zh/data_normal_*.json", "data_zh/data_special_*.json")
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")
_ACE_TURN = re.compile(r"^(user|system|assistant):\s?", re.IGNORECASE)


class BenchmarkMissing(FileNotFoundError):
    """A benchmark folder that is absent or holds none of the files the check reads."""


def text_ngrams(text: str, n: int = N) -> set[str]:
    """The N-grams of a text (the module docstring): word grams over its Latin-script words, and character
    grams over its non-space characters for the windows whose characters are mostly CJK."""
    norm = normalize_text(text).lower()
    words = _CJK.sub(" ", norm).split()
    grams = {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}
    chars = norm.replace(" ", "")
    prefix = [0]  # prefix[i] = CJK characters among chars[:i]
    for c in chars:
        prefix.append(prefix[-1] + (1 if _CJK.match(c) else 0))
    majority = n // 2 + 1
    return grams | {chars[s:s + n] for s in range(len(chars) - n + 1)
                    if prefix[s + n] - prefix[s] >= majority}


def bfcl_user_turns(question: object) -> list[str]:
    """The user contents of a BFCL question: a list of turns of messages, or a flat list of messages."""
    texts: list[str] = []
    for item in question if isinstance(question, list) else []:
        if isinstance(item, list):
            texts += bfcl_user_turns(item)
        elif isinstance(item, dict) and item.get("role") == "user" and isinstance(item.get("content"), str):
            texts.append(item["content"])
    return texts


def acebench_user_turns(question: str) -> list[str]:
    """The user turns of an ACEBench question string ("user: ..." / "system: ..." lines)."""
    turns: list[list] = []
    for line in question.split("\n"):
        prefix = _ACE_TURN.match(line)
        if prefix:
            turns.append([prefix.group(1).lower(), [line[prefix.end():]]])
        elif turns:
            turns[-1][1].append(line)
        else:
            turns.append(["user", [line]])
    return ["\n".join(lines) for role, lines in turns if role == "user"]


def _json_lines(path: Path) -> list[dict] | None:
    """The objects of a JSON-lines file; None when it is not JSON lines of objects."""
    rows = []
    for line in split_lines(path.read_text(encoding="utf-8-sig")):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(row, dict):
            return None
        rows.append(row)
    return rows


def benchmark_files(bfcl_dir: Path, acebench_dir: Path) -> list[tuple[str, Path]]:
    """(benchmark, path) of every file the check reads, sorted; BenchmarkMissing when a folder has none."""
    bfcl = sorted(Path(bfcl_dir).glob(BFCL_PATTERN)) if Path(bfcl_dir).is_dir() else []
    if not bfcl:
        raise BenchmarkMissing(f"no BFCL question file ({BFCL_PATTERN}) in {bfcl_dir}: install the BFCL "
                               "environment (docs/m0-验证.md) or pass --bfcl-dir")
    ace = sorted(p for pattern in ACEBENCH_PATTERNS for p in Path(acebench_dir).glob(pattern)) \
        if Path(acebench_dir).is_dir() else []
    if not ace:
        raise BenchmarkMissing(f"no ACEBench data file ({', '.join(ACEBENCH_PATTERNS)}) in {acebench_dir}: "
                               "clone ACEBench there or pass --acebench-dir")
    return [("bfcl", p) for p in bfcl] + [("acebench", p) for p in ace]


def questions_of(benchmark: str, path: Path) -> list[tuple[str, str]]:
    """(entry id, question text) of one benchmark file; none for a file without question entries."""
    rows = _json_lines(path) or []
    out = []
    for number, row in enumerate(rows):
        entry = str(row.get("id", f"line {number + 1}"))
        question = row.get("question")
        if benchmark == "bfcl":
            texts = bfcl_user_turns(question)
        else:
            texts = acebench_user_turns(question) if isinstance(question, str) else []
        out += [(entry, text) for text in texts]
    return out


@dataclass
class BenchmarkIndex:
    """Every benchmark gram with the entries holding it, and the files read."""

    grams: dict[str, list[tuple[str, str, str]]] = field(default_factory=dict)  # gram -> (bench, file, id)
    files: list[dict] = field(default_factory=list)

    @classmethod
    def build(cls, bfcl_dir: str | Path, acebench_dir: str | Path) -> BenchmarkIndex:
        index = cls()
        for benchmark, path in benchmark_files(Path(bfcl_dir), Path(acebench_dir)):
            questions = questions_of(benchmark, path)
            for entry, text in questions:
                for gram in text_ngrams(text):
                    hits = index.grams.setdefault(gram, [])
                    hit = (benchmark, path.name, entry)
                    if len(hits) < ENTRIES_PER_GRAM and hit not in hits:
                        hits.append(hit)
            index.files.append({"benchmark": benchmark, "path": path.as_posix(),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "entries": len({entry for entry, _ in questions}),
                                "questions": len(questions)})
        return index


def user_texts(record: object) -> list[str]:
    """The user messages of a training record as load_records gives it."""
    if isinstance(record, Transcript):
        messages = record.messages
    elif isinstance(record, dict):
        messages = record.get("messages") or []
    else:
        messages = record
    return [m["content"] for m in messages
            if isinstance(m, dict) and m.get("role") == "user" and isinstance(m.get("content"), str)]


def check_file(path: str | Path, index: BenchmarkIndex) -> dict:
    """The overlap report of one training file: its records that share a gram with a benchmark question."""
    path = Path(path)
    records = load_records(path)
    overlapping = []
    for number, record in enumerate(records):
        matches: dict[tuple[str, str, str], str] = {}
        for text in user_texts(record):
            for gram in sorted(text_ngrams(text)):
                for hit in index.grams.get(gram, []):
                    matches.setdefault(hit, gram)
        if matches:
            listed = sorted(matches)[:MATCHES_KEPT]
            overlapping.append({
                "index": number, "id": record.get("id") if isinstance(record, dict) else None,
                "matches_total": len(matches),
                "matches": [{"benchmark": b, "file": f, "entry": e, "gram": matches[b, f, e]}
                            for b, f, e in listed]})
    return {"path": path.as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "records": len(records), "overlapping": len(overlapping), "records_overlapping": overlapping}


def load_skips(report_path: str | Path, files: list[Path]) -> tuple[dict[Path, set[int]], dict | None]:
    """The record indices to skip per data file (by resolved path) and what the run manifest records of
    the report; ({}, None) when there is no report. Only files among files are considered. ValueError when a
    listed file's SHA-256 is not the one checked: the report is stale and its indices may point elsewhere."""
    report_path = Path(report_path)
    if not report_path.is_file():
        return {}, None
    raw = report_path.read_bytes()
    report = json.loads(raw.decode("utf-8"))
    wanted = {Path(f).resolve(): Path(f) for f in files}
    skips: dict[Path, set[int]] = {}
    for entry in report.get("files", []):
        resolved = Path(entry["path"]).resolve()
        if resolved not in wanted:
            continue
        current = hashlib.sha256(wanted[resolved].read_bytes()).hexdigest()
        if current != entry["sha256"]:
            raise ValueError(f"{report_path} is stale for {entry['path']}: it was checked at sha256 "
                             f"{entry['sha256']}, the file is now {current}; run "
                             "scripts/check_benchmark_overlap.py again")
        indices = {r["index"] for r in entry.get("records_overlapping", [])}
        if indices:
            skips[resolved] = indices
    info = {"report": report_path.as_posix(), "report_sha256": hashlib.sha256(raw).hexdigest(),
            "skipped_records": sum(len(v) for v in skips.values())}
    return skips, info

"""Reasoning sidecars: the reasoning behind a training turn, kept next to the training file and checked
against the answer it was written for (docs/m7-家族层设计.md §6; docs/m7-计划.md §2, 思考数据).

Why a sidecar. The training files stay byte-identical (every earlier run and manifest refers to them by
SHA-256), and the reasoning of a turn lives in a JSON Lines file of its own, one row per turn. A row names its
turn by (record id, k) and carries the hash of the answer it was written for, so a row whose turn has changed
since (a transform rewrote the answer) is refused instead of trained on.

Turn keys. The training path cuts an agent-history record (train.samples.is_agent_history: the synthetic files
of data.assemble and scripts/transform_synth.py) with agent.context.step_contexts into one (context, assistant
message) pair per assistant message of record["messages"], in order (train.samples.build_step_samples); a pair
over the length limit is dropped only after the cut, and scripts/train.py skips whole records (those the
benchmark-overlap report names), never a turn but the context turns a record names (its context_turns, the
leading turns of an FT4 cut copy, train.samples), which keep their k. So turn k is the k-th pair of
step_contexts: k is the position of an assistant message among the record's assistant messages, from 0.
turns() gives (k, index in the history, message); the record id is the record's "id".

Rows (ROW_KEYS), one canonical JSON object per line, in (id, k) order:

    {"id", "k", "answer_sha256", "source": "teacher" | "rationalized", "reasoning", "model", "cache_key",
     "context_equal"}

- answer_sha256: answer_sha256 of the training answer, the record's assistant message k.
- source: "teacher" for the M2 teacher's own reasoning (scripts/join_reasoning.py), "rationalized" for one
  written afterwards for the given context and answer (scripts/rationalize_reasoning.py), "template" for the
  English template reasoning FT4 writes for the turns its rules changed or built
  (scripts/template_reasoning.py, docs/m7-本地提升计划.md WP6).
- reasoning: text with more than whitespace and without a think tag (the chat template splits an assistant
  turn at "</think>").
- model: the model that wrote the reasoning; cache_key: the response-cache key (runs/llm/cache.sqlite) of the
  request whose answer carried it, so every row can be traced to the answer it came with.
- context_equal: whether the reasoning was written for the context the training sample shows (build_context of
  the record's history before the turn). A teacher row compares the teacher's request with it, and a transform
  that changed an earlier turn makes them differ; a rationalized row is written for that context itself.

ReasoningIndex(paths) reads sidecar files; lookup(id, k, answer) gives the reasoning of the turn, None when no
file holds the turn, and raises ReasoningError when the row was written for another answer (the sidecar is
stale); a turn in two rows (two files, or twice in one) is a ReasoningError when the index is built.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from ..agent.types import normalize_arguments, parse_arguments
from ..canonical import canonical_json, split_lines
from ..normalize import normalize_text

TEACHER, RATIONALIZED, TEMPLATE = "teacher", "rationalized", "template"
SOURCES = (TEACHER, RATIONALIZED, TEMPLATE)
ROW_KEYS = ("id", "k", "answer_sha256", "source", "reasoning", "model", "cache_key", "context_equal")
# data.assemble's THINK_TAGS, not imported from there: data.assemble imports train.samples, so the training
# path could not import this module without a cycle
THINK_TAGS = ("<think>", "</think>")
# scripts/transform_synth.py's RUN: the records it built by running the loop with a scripted model, so no
# teacher gave their turns; their ids are req-m6-NNNN, or a train request's id with -m6r3 or -m6r6
SCRIPTED_RUN = "m6-transform"
_SCRIPTED_ID = re.compile(r"^req-m6-\d+$|-m6r[36]$")
# scripts/transform_synth.py's FT4_RUN (--ft4): the records FT4 built, new ones (req-ft4-NNNN) and cut copies
# of a trajectory (<record id>-ft4r9k<k> and -ft4r10k<k>) whose turns before the last are context only (their
# context_turns); the last is a template's, or the teacher's own turn kept (R10 where the teacher had asked)
FT4_RUN = "ft4-transform"
_FT4_ID = re.compile(r"^req-ft4-\d+$|-ft4r(?:9|10)k\d+$")
SCRIPTED_RUNS = {SCRIPTED_RUN: _SCRIPTED_ID, FT4_RUN: _FT4_ID}  # run -> the ids of its records
_SHA256 = re.compile(r"[0-9a-f]{64}")
PERCENTILES = (0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)


class ReasoningError(ValueError):
    """A sidecar or a record that cannot be used: a malformed row, a turn in two rows, a row written for
    another answer (stale), a record whose markers disagree. The message names the place."""


def turns(messages: list[dict]) -> list[tuple[int, int, dict]]:
    """(k, index, message) for every assistant message of an API-format history, in order: turn k is the k-th
    sample the trainer cuts from the history (module docstring)."""
    positions = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    return [(k, i, messages[i]) for k, i in enumerate(positions)]


def answer_sha256(answer: dict) -> str:
    """The SHA-256 of an assistant message in canonical form: its text after normalize_text, and its calls as
    the sorted multiset of [name, arguments], the arguments read as the loop reads them (parse_arguments: an
    object, integral floats as ints; text that is no JSON object stays as written), in canonical JSON.

    Call ids, the order of the calls and of the argument keys, how the arguments' JSON is spelled (spacing,
    3.0 for 3) and what normalize_text evens out in the text (runs of whitespace, full-width forms) do not
    count; a changed value (whitespace inside a string argument included), name, call or word does. An API
    message (arguments as JSON text) and the HF template message of the same answer (arguments as an object)
    hash alike.
    """
    calls = sorted(canonical_json([name, arguments]) for name, arguments in _calls(answer))
    text = normalize_text(answer.get("content") or "")
    return hashlib.sha256(canonical_json({"calls": calls, "content": text}).encode("utf-8")).hexdigest()


def _calls(answer: dict) -> Iterator[tuple[str, object]]:
    for call in answer.get("tool_calls") or []:
        function = call.get("function") or {}
        raw = function.get("arguments")
        if isinstance(raw, str):
            parsed = parse_arguments(raw)
            arguments = raw if parsed is None else parsed
        else:
            arguments = normalize_arguments({} if raw is None else raw)
        yield function.get("name") or "", arguments


def has_think_tag(text: str) -> bool:
    return any(tag in text for tag in THINK_TAGS)


def is_scripted(record: dict) -> bool:
    """True for a record scripts/transform_synth.py built itself (run SCRIPTED_RUN, M6's records made with a
    scripted model, or FT4_RUN, FT4's), whose trained turns no teacher run gave as a record of its own (an FT4
    cut copy may keep one teacher turn as its target; scripts/template_reasoning.py re-keys that turn's
    reasoning, the join does not look for it). Its id must say so too, with the pattern of its own run
    (SCRIPTED_RUNS), and no other record's id may match one: ReasoningError when the two markers disagree,
    since the file is then not the one this was written for."""
    run, rid = record.get("run"), str(record.get("id"))
    id_runs = [name for name, pattern in SCRIPTED_RUNS.items() if pattern.search(rid)]
    if id_runs != ([run] if run in SCRIPTED_RUNS else []):
        raise ReasoningError(f"record {record.get('id')!r} has run {run!r}: its id and its run disagree on "
                             "whether a scripted model made it")
    return run in SCRIPTED_RUNS


def sidecar_row(record_id: str, k: int, answer: dict, *, source: str, reasoning: str, model: str,
                cache_key: str, context_equal: bool) -> dict:
    """The row of turn k of a record whose training answer is answer; ReasoningError when it is malformed."""
    row = {"id": record_id, "k": k, "answer_sha256": answer_sha256(answer), "source": source,
           "reasoning": reasoning, "model": model, "cache_key": cache_key, "context_equal": context_equal}
    return check_row(row, f"turn {(record_id, k)!r}")


def check_row(row: object, where: str) -> dict:
    """row when it is a well-formed sidecar row; ReasoningError naming where and every problem otherwise."""
    if not isinstance(row, dict) or set(row) != set(ROW_KEYS):
        raise ReasoningError(f"{where}: a sidecar row is an object with exactly the keys "
                             f"{', '.join(ROW_KEYS)}")
    problems = []
    if not isinstance(row["id"], str) or not row["id"]:
        problems.append("id is not a non-empty string")
    if type(row["k"]) is not int or row["k"] < 0:
        problems.append("k is not an integer >= 0")
    for key in ("answer_sha256", "cache_key"):
        if not isinstance(row[key], str) or not _SHA256.fullmatch(row[key]):
            problems.append(f"{key} is not a SHA-256 in lower-case hex")
    if row["source"] not in SOURCES:
        problems.append(f"source is not one of {', '.join(SOURCES)}")
    if not isinstance(row["reasoning"], str) or not row["reasoning"].strip():
        problems.append("reasoning is empty")
    elif has_think_tag(row["reasoning"]):
        problems.append("reasoning holds a think tag")
    if not isinstance(row["model"], str) or not row["model"]:
        problems.append("model is not a non-empty string")
    if not isinstance(row["context_equal"], bool):
        problems.append("context_equal is not true or false")
    if problems:
        raise ReasoningError(f"{where}: {'; '.join(problems)}")
    return row


def write_sidecar(rows: Iterable[dict], path: str | Path) -> tuple[str, int]:
    """The rows as canonical JSON lines in (id, k) order, UTF-8 with LF; the file's SHA-256 and its size in
    bytes. ReasoningError for a malformed row or a turn given twice, before anything is written."""
    keyed: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = (check_row(row, "a row to write")["id"], row["k"])
        if key in keyed:
            raise ReasoningError(f"turn {key!r} is given twice")
        keyed[key] = row
    data = "".join(canonical_json(keyed[key]) + "\n" for key in sorted(keyed)).encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest(), len(data)


class ReasoningIndex:
    """(record id, k) -> the sidecar row of that turn, over one or more sidecar files (module docstring)."""

    def __init__(self, paths: Iterable[str | Path]) -> None:
        self.rows: dict[tuple[str, int], dict] = {}
        self._places: dict[tuple[str, int], str] = {}
        for path in paths:
            for number, line in enumerate(split_lines(Path(path).read_text(encoding="utf-8")), 1):
                if not line.strip():
                    continue
                where = f"{path}: line {number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReasoningError(f"{where} is not JSON: {exc}") from exc
                key = (check_row(row, where)["id"], row["k"])
                if key in self.rows:
                    raise ReasoningError(f"{where}: turn {key!r} is also in {self._places[key]}")
                self.rows[key], self._places[key] = row, where

    def __len__(self) -> int:
        return len(self.rows)

    def lookup(self, record_id: str, k: int, answer: dict) -> str | None:
        """The reasoning of turn k of the record, None when no file holds the turn; ReasoningError when the
        row was written for another answer than this training answer (the sidecar is stale and must be
        rebuilt)."""
        row = self.rows.get((record_id, k))
        if row is None:
            return None
        if row["answer_sha256"] != answer_sha256(answer):
            raise ReasoningError(f"stale reasoning sidecar: the row of {record_id} turn {k} "
                                 f"({self._places[record_id, k]}) was written for another answer than the "
                                 "training answer; rebuild the sidecar")
        return row["reasoning"]


def token_counter(model_dir: str | Path) -> Callable[[str], int]:
    """count(text): the tokens of text under the model folder's tokenizer, loaded as scripts/train.py loads
    it (AutoTokenizer) and counted as the training path counts (add_special_tokens=False)."""
    from transformers import AutoTokenizer  # heavy import, only needed when counting with a real tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    def count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return count


def length_stats(values: list[int]) -> dict:
    """n, total, mean, min, max and the PERCENTILES of values (p50 ...), each the value at rank int(q * n) of
    the sorted list, as data.assemble.token_counts takes them; {"n": 0} for no values."""
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    n = len(ordered)
    stats = {"n": n, "total": sum(ordered), "mean": round(sum(ordered) / n, 1), "min": ordered[0],
             "max": ordered[-1]}
    stats.update({f"p{round(q * 100)}": ordered[min(n - 1, int(q * n))] for q in PERCENTILES})
    return stats

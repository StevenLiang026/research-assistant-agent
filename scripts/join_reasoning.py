"""Join the teacher's reasoning to the turns of a synthetic training file, from the response cache
(docs/m7-家族层设计.md §6; docs/m7-计划.md §2 思考数据, WP-C, §5 新增训练轨迹). CPU only, nothing paid, under
a minute.

    .venv/Scripts/python.exe scripts/join_reasoning.py [--records data/raw/synth/synth-m6.jsonl]
        [--transcripts runs/m2/synth/train/transcripts] [--cache runs/llm/cache.sqlite]
        [--out data/raw/synth/reasoning-m7-teacher.jsonl]
        [--report data/raw/synth/reasoning-m7-teacher.report.json] [--tokenizer models/Qwen3-4B]
    .venv/Scripts/python.exe scripts/join_reasoning.py --records data/raw/synth/synth-m7.jsonl
        --run runs/m7/synth/m7-train --out data/raw/synth/reasoning-m7b-teacher.jsonl
        --report data/raw/synth/reasoning-m7b-teacher.report.json

Without --run the defaults are M7's first sidecar: synth-m6's records and the M2 run's transcripts, the
teacher deepseek-flash at temperature 0.0. --run names the generator run folder the records were assembled
from instead (M7's flash run, data.synth: manifest.json and transcripts/): its transcripts, and its
teacher's model, temperature and thinking setting as its manifest's config records them (config.model,
model_temperature, thinking; a run made before M7 records no thinking and thought by the auto rules); every
record that is not scripted must carry the run's name (config.run, the name data.assemble writes into
"run"), and the report records the run instead of the plan's numbers for synth-m6.

Why. The M2 teacher, deepseek-flash in thinking mode, gave a reasoning_content with every answer, and the
response cache keeps each answer whole under the key of the request that produced it. Rebuilding that key for
a training turn retrieves the teacher's own reasoning for exactly that turn, without an API call.

Turn keys. A row names its training sample by (record id, k), the key the training path looks reasoning up by
(docs/m7-家族层设计.md §5). train.samples.build_step_samples cuts an agent-history record with
agent.context.step_contexts into one (context, assistant message) pair per assistant message of
record["messages"], in order, and drops an over-long sample only after the cut; scripts/train.py reads every
record of a file with load_records and skips whole records only (those the benchmark-overlap report names).
So k, the position of an assistant message among the record's assistant messages counted from 0, is the k-th
pair the trainer cuts from the record, whatever it drops afterwards (data.reasoning.turns gives each turn's k,
its index in the history and the message).

Pairing (step 1). The records are read as the trainer reads them (train.samples.load_records); each must be an
agent-history record. A record that scripts/transform_synth.py built with a scripted model
(data.reasoning.is_scripted) has no teacher turn: its turns are "scripted". Every other record is an M2
trajectory whose transcript is <transcripts>/<id>.json and must carry the record's prompt and schema hashes.
Turn k of the record is paired with the k-th assistant message of the transcript, at index j of its history.
The assembler and the M6 transforms rewrite turns in place or drop the final answer but never add a turn, so
a record with more assistant turns than its transcript stops the join.

Key (step 2). The plain request is {"model": "deepseek-flash", "messages": build_context(t.messages[:j]),
"tools": MODEL_TOOLS, "temperature": 0.0}: what CloudModel.next_turn sent through CloudClient.chat for the
loop's request of that step (agent.loop asks with build_context of the history, CloudModel adds temperature
0.0, CloudClient puts the model, the messages, the tools and the parameters). The key is the one the client
filed the answer under, which llm.cloud's rules fix turn by turn (Teacher.lookup):
- thinking auto: the plain key when it holds an answer (rule 1: served whatever the history) or when the
  reasoning of every earlier turn of the transcript is known, a non-empty string in the cached answer of
  that turn (rule 2: the request went in thinking mode with that reasoning attached to the earlier turns on
  the wire only, and the answer was filed under the plain key, so passing the reasoning back changes no
  key); otherwise the THINKING_OFF key, the plain request with extra_body THINKING_OFF (rule 3: a turn after
  an empty reasoning goes without thinking, and so does every turn after it, its own reasoning being none);
- thinking off: the THINKING_OFF key for every turn.
Every M2 answer is under the plain key (M2 sent nothing else); M7's run with thinking auto has both kinds.
The walk takes each turn's reasoning from the cached answer of that turn, the one the next turns of the
transcript went back with. The M2 and M7 runs had no handoff model, and the assembler drops every trajectory
with a step accepted after a repair, so every trained step is the answer to that first request of its step.
The cache is opened read-only (SQLite mode=ro): nothing is added or changed. A key that misses stops the join,
naming the record, the turn and the key, and so does a cached answer that is not the transcript's message
(compared with data.reasoning.answer_sha256, which passes over call ids and the "index" of the API's raw
calls), since its reasoning would belong to another answer.

Status (step 3), the first that applies:
- scripted: a turn of a scripted record;
- empty: the cached reasoning is missing or blank;
- edited: the record's answer is not the teacher's (answer_sha256 differs): M6 changed it;
- think_tag: the reasoning holds "<think>" or "</think>", where the template would split the turn (the design
  names no status for it; such a turn is kept out of the teacher rows and left for the rationalizer);
- teacher: the teacher's reasoning for the very answer the trainer trains on.
Two flags for every turn of a teacher record: answer_equal, the record's answer is the teacher's (true for
every teacher and think_tag turn, false for every edited one; for an empty turn it tells whether M6 changed
the answer too), and context_equal, build_context of the record's history before turn k equals the teacher's
request context (an edited earlier turn makes it false for the turns after it).

Outputs (step 4). --out holds the teacher turns only, as sidecar rows (data.reasoning: source "teacher", the
teacher's model, the cache key, context_equal) in (id, k) order; the same inputs give the same bytes. --report
(sorted keys, two-space indent) holds counts and lengths only, never cached text: records and turns, keys
looked up and hit, the statuses, the non-empty reasonings, answer_equal and context_equal per status, the
characters and tokens (--tokenizer, data.reasoning.token_counter) of the teacher rows' reasoning, and the
output's SHA-256; without --run also the plan's expected numbers for synth-m6 (step 5: 4,308 teacher turns,
4,141 non-empty, about 2,814 used as they are) with the measured difference, with --run instead the run
(path, name, manifest SHA-256, model, temperature, thinking, requests SHA-256) and the keys by kind (plain,
thinking_off). Every turn without a teacher row is for scripts/rationalize_reasoning.py.

Exit 0 done; 1 a join error (a missing or foreign transcript, a record with more turns than its transcript, a
key that misses, a cached answer that is not the transcript's, a record that is not an agent history or whose
markers disagree, with --run a record of another run): nothing is written; 2 a missing input (records,
transcripts folder, the run's manifest or one without a usable config, cache, tokenizer).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from research_agent.agent.context import build_context
from research_agent.agent.loop import load_transcript
from research_agent.agent.types import Transcript
from research_agent.canonical import canonical_json
from research_agent.data.reasoning import (
    TEACHER,
    ReasoningError,
    answer_sha256,
    has_think_tag,
    is_scripted,
    length_stats,
    sidecar_row,
    token_counter,
    turns,
    write_sidecar,
)
from research_agent.llm.cache import request_key
from research_agent.llm.cloud import THINKING_MODES, THINKING_OFF
from research_agent.tools.schemas import MODEL_TOOLS, SCHEMA_HASH
from research_agent.train.samples import RecordError, is_agent_history, load_records

RECORDS = Path("data/raw/synth/synth-m6.jsonl")
TRANSCRIPTS = Path("runs/m2/synth/train/transcripts")
CACHE = Path("runs/llm/cache.sqlite")
OUT = Path("data/raw/synth/reasoning-m7-teacher.jsonl")
REPORT = Path("data/raw/synth/reasoning-m7-teacher.report.json")
TOKENIZER = Path("models/Qwen3-4B")
TEACHER_MODEL = "deepseek-flash"  # the M2 teacher (runs/m2/synth/train/manifest.json: config.model)
TEACHER_TEMPERATURE = 0.0  # what CloudModel sent with every request (config.model_temperature)
RUN_MANIFEST, RUN_TRANSCRIPTS = "manifest.json", "transcripts"  # in a generator run folder (data.synth)
PLAIN, OFF = "plain", "thinking_off"  # the kinds of key an answer is filed under (module docstring, step 2)
EMPTY, EDITED, THINK_TAG, SCRIPTED = "empty", "edited", "think_tag", "scripted"
STATUSES = (TEACHER, EDITED, EMPTY, THINK_TAG, SCRIPTED)  # the report's order
EXPECTED = {"teacher_turns": 4308, "non_empty": 4141, "teacher": 2814}  # docs/m7-家族层设计.md §6, step 5

CountTokens = Callable[[str], int]
GetCached = Callable[[str], dict | None]  # ResponseCache.get: key -> response, None on a miss
TranscriptOf = Callable[[str], Transcript]  # record id -> its teacher transcript


class JoinError(ValueError):
    """A record that cannot be joined with its transcript and the cache; the message names the record."""


class RunError(ValueError):
    """A run folder (--run) whose manifest does not say how its teacher was asked."""


@dataclass(frozen=True)
class Teacher:
    """The teacher's requests as CloudModel sent them through CloudClient (module docstring, step 2): the
    model, the temperature and the client's thinking setting. The default is the M2 teacher."""

    model: str = TEACHER_MODEL
    temperature: float = TEACHER_TEMPERATURE
    thinking: str = "auto"

    def __post_init__(self) -> None:
        if self.thinking not in THINKING_MODES:
            raise ValueError(f"thinking must be one of {THINKING_MODES}, not {self.thinking!r}")

    def lookup(self, context: list[dict], known: bool, get: GetCached) -> tuple[str, str, dict | None]:
        """The key the client filed its answer to context under, the key's kind (PLAIN or OFF) and the cached
        answer (None when the key misses); known: whether the reasoning of every earlier turn is known."""
        plain = {"model": self.model, "messages": context, "tools": MODEL_TOOLS,
                 "temperature": self.temperature}
        if self.thinking == "auto":
            key = request_key(plain)
            cached = get(key)
            if cached is not None or known:
                return key, PLAIN, cached
        key = request_key({**plain, "extra_body": THINKING_OFF})
        return key, OFF, get(key)


M2_TEACHER = Teacher()


@dataclass(frozen=True)
class Joined:
    """One training turn and what the join found for it; answer_equal, context_equal, cache_key and key_kind
    are None for a scripted turn, reasoning is kept for a teacher turn only."""

    id: str
    k: int
    answer: dict  # the training answer: the record's assistant message k
    status: str
    answer_equal: bool | None = None
    context_equal: bool | None = None
    cache_key: str | None = None
    reasoning: str | None = None
    key_kind: str | None = None  # PLAIN or OFF
    model: str = TEACHER_MODEL  # the teacher's

    def row(self) -> dict:
        """The sidecar row of a teacher turn."""
        return sidecar_row(self.id, self.k, self.answer, source=TEACHER, reasoning=self.reasoning,
                           model=self.model, cache_key=self.cache_key, context_equal=self.context_equal)


def status_of(reasoning: object, answer_equal: bool) -> str:
    """The status of a turn of a teacher record (module docstring, step 3), from its reasoning as cached and
    whether the training answer is the teacher's."""
    if not isinstance(reasoning, str) or not reasoning.strip():
        return EMPTY
    if not answer_equal:
        return EDITED
    if has_think_tag(reasoning):
        return THINK_TAG
    return TEACHER


def join_record(record: dict, transcript_of: TranscriptOf, get: GetCached, teacher: Teacher = M2_TEACHER,
                run: str | None = None) -> list[Joined]:
    """The turns of one record, joined (module docstring, steps 1-3); JoinError when they cannot be. run, when
    given, is the name every record that is not scripted must carry (--run)."""
    rid, messages = record["id"], record["messages"]
    if is_scripted(record):
        return [Joined(rid, k, answer, SCRIPTED) for k, _, answer in turns(messages)]
    if run is not None and record.get("run") != run:
        raise JoinError(f"{rid} is a record of run {record.get('run')!r}, not of {run!r}, whose transcripts "
                        "are joined")
    transcript = transcript_of(rid)
    for name in ("prompt_hash", "schema_hash"):
        if getattr(transcript, name) != record.get(name):
            raise JoinError(f"{rid}: its transcript was recorded under {name.split('_')[0]} "
                            f"{getattr(transcript, name)}, the record under {record.get(name)}")
    trained, taught = turns(messages), turns(transcript.messages)
    if len(trained) > len(taught):
        raise JoinError(f"{rid} has {len(trained)} assistant turns, its transcript {len(taught)}")
    joined = []
    known = True  # the reasoning of every earlier turn of the transcript is known (none before the first)
    for (k, i, answer), (_, j, given) in zip(trained, taught, strict=False):
        context = build_context(transcript.messages[:j])
        key, kind, cached = teacher.lookup(context, known, get)
        if cached is None:
            raise JoinError(f"{rid} turn {k}: no cached answer under the teacher's key {key}")
        message = cached["choices"][0]["message"]
        if answer_sha256(message) != answer_sha256(given):
            raise JoinError(f"{rid} turn {k}: the cached answer is not the transcript's, so its reasoning "
                            "belongs to another answer")
        reasoning = message.get("reasoning_content")
        known = known and isinstance(reasoning, str) and bool(reasoning)  # what CloudClient passes back
        answer_equal = answer_sha256(answer) == answer_sha256(given)
        status = status_of(reasoning, answer_equal)
        context_equal = canonical_json(build_context(messages[:i])) == canonical_json(context)
        joined.append(Joined(rid, k, answer, status, answer_equal=answer_equal, context_equal=context_equal,
                             cache_key=key, reasoning=reasoning if status == TEACHER else None, key_kind=kind,
                             model=teacher.model))
    return joined


def join(records: list, transcript_of: TranscriptOf, get: GetCached, *, teacher: Teacher = M2_TEACHER,
         run: str | None = None) -> list[Joined]:
    """Every turn of every record, joined, in file order; JoinError (ReasoningError for a record whose
    markers disagree) at the first record that cannot be."""
    joined = []
    for number, record in enumerate(records, 1):
        if not is_agent_history(record):
            raise JoinError(f"record {number} is not an agent-history record, the only kind the trainer cuts "
                            "with step_contexts")
        joined += join_record(record, transcript_of, get, teacher, run)
    return joined


class ReadOnlyCache:
    """The response cache opened read-only (SQLite mode=ro), with ResponseCache.get's lookup: ResponseCache
    itself would create its table on opening, a write."""

    def __init__(self, path: Path) -> None:
        self.db = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)

    def get(self, key: str) -> dict | None:
        row = self.db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def close(self) -> None:
        self.db.close()


def transcript_at(folder: Path, record_id: str) -> Transcript:
    """The M2 transcript of a record; JoinError when it is missing or not a transcript."""
    path = folder / f"{record_id}.json"
    try:
        return load_transcript(path)
    except FileNotFoundError as exc:
        raise JoinError(f"{record_id}: no transcript at {path}") from exc
    except (json.JSONDecodeError, TypeError) as exc:
        raise JoinError(f"{record_id}: {path} is not a transcript: {exc}") from exc


def per_status(taught: list[Joined], flag: str) -> dict:
    """For a flag of the turns of teacher records (answer_equal, context_equal): how many turns of each status
    have it, and how many have it in all."""
    counts = {s: sum(1 for j in taught if j.status == s and getattr(j, flag))
              for s in STATUSES if s != SCRIPTED}
    return {**counts, "all": sum(1 for j in taught if getattr(j, flag))}


def build_report(records: list[dict], joined: list[Joined], rows: list[dict], count_tokens: CountTokens,
                 inputs: dict, output: dict, *, teacher: Teacher = M2_TEACHER,
                 run: dict | None = None) -> dict:
    """The report (module docstring, step 4): counts and lengths, no cached text; run, the account of the run
    of --run, replaces the plan's numbers for synth-m6 and adds the keys by kind."""
    statuses = Counter(j.status for j in joined)
    taught = [j for j in joined if j.status != SCRIPTED]
    measured = {"teacher_turns": len(taught), "non_empty": len(taught) - statuses[EMPTY],
                "teacher": statuses[TEACHER]}
    scripted_records = sum(1 for r in records if is_scripted(r))
    report = {
        "script": "scripts/join_reasoning.py",
        "inputs": inputs,
        "teacher_request": {"model": teacher.model, "temperature": teacher.temperature,
                            "tools": "MODEL_TOOLS", "schema_hash": SCHEMA_HASH},
        "records": {"total": len(records), "teacher_trajectory": len(records) - scripted_records,
                    "scripted": scripted_records},
        "turns": {"total": len(joined), "teacher_trajectory": len(taught), "scripted": statuses[SCRIPTED]},
        "keys": {"looked_up": len(taught), "hit": len(taught)},  # a miss stops the join
        "status": {s: statuses[s] for s in STATUSES},
        "non_empty": measured["non_empty"],
        "answer_equal": per_status(taught, "answer_equal"),
        "context_equal": per_status(taught, "context_equal"),
        "lengths": {"characters": length_stats([len(r["reasoning"]) for r in rows]),
                    "tokens": length_stats([count_tokens(r["reasoning"]) for r in rows])},
        "output": output,
    }
    if run is None:
        report["expected"] = dict(EXPECTED)
        report["difference"] = {key: measured[key] - EXPECTED[key] for key in EXPECTED}
    else:
        kinds = Counter(j.key_kind for j in taught)
        report["keys"].update({PLAIN: kinds[PLAIN], OFF: kinds[OFF]})
        report["run"] = run
    return report


def read_run(run_dir: Path) -> tuple[Teacher, dict]:
    """The teacher and the account of a generator run folder (--run) from its manifest's config; RunError
    when the config does not say what the requests were (model, temperature), ValueError for a manifest that
    is not JSON or an unknown thinking setting."""
    path = run_dir / RUN_MANIFEST
    manifest = json.loads(path.read_text(encoding="utf-8"))
    config = manifest.get("config") if isinstance(manifest, dict) else None
    if not isinstance(config, dict):
        raise RunError(f"{path} has no config")
    model, temperature = config.get("model"), config.get("model_temperature")
    thinking = config.get("thinking", "auto")  # none before M7: those runs thought by the auto rules
    if not (isinstance(model, str) and model) or isinstance(temperature, bool) \
            or not isinstance(temperature, int | float):
        raise RunError(f"{path}: config.model and config.model_temperature do not say what was sent")
    teacher = Teacher(model, temperature, thinking)  # ValueError for an unknown thinking setting
    name = config.get("run") if isinstance(config.get("run"), str) and config.get("run") else run_dir.name
    return teacher, {"path": run_dir.as_posix(), "name": name,
                     "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "model": model,
                     "temperature": temperature, "thinking": thinking,
                     "requests_sha256": config.get("requests_sha256")}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Join the teacher's reasoning to the synthetic training turns.")
    ap.add_argument("--records", type=Path, default=RECORDS, help=f"the training file (default {RECORDS})")
    source = ap.add_mutually_exclusive_group()
    source.add_argument("--transcripts", type=Path, default=None,
                        help=f"the M2 run's transcripts folder (default {TRANSCRIPTS})")
    source.add_argument("--run", type=Path, default=None,
                        help="the generator run folder the records were assembled from: its transcripts, and "
                             "its teacher as its manifest records it (module docstring)")
    ap.add_argument("--cache", type=Path, default=CACHE,
                    help=f"the response cache, read only (default {CACHE})")
    ap.add_argument("--out", type=Path, default=OUT, help=f"the teacher sidecar (default {OUT})")
    ap.add_argument("--report", type=Path, default=REPORT, help=f"the report (default {REPORT})")
    ap.add_argument("--tokenizer", type=Path, default=TOKENIZER,
                    help=f"the model folder whose tokenizer counts the reasoning's tokens "
                         f"(default {TOKENIZER})")
    return ap


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None, *, count_tokens: CountTokens | None = None) -> int:
    """count_tokens stands in for the tokenizer in tests; None reads --tokenizer."""
    args = build_parser().parse_args(argv)
    transcripts = (args.transcripts or TRANSCRIPTS) if args.run is None else args.run / RUN_TRANSCRIPTS
    missing = [str(p) for p in (args.records, args.cache) if not p.is_file()]
    missing += [] if transcripts.is_dir() else [str(transcripts)]
    if args.run is not None and not (args.run / RUN_MANIFEST).is_file():
        missing.append(str(args.run / RUN_MANIFEST))
    if count_tokens is None and not (args.tokenizer / "tokenizer.json").is_file():
        missing.append(str(args.tokenizer / "tokenizer.json"))
    if missing:
        print(f"missing input: {', '.join(missing)}", file=sys.stderr)
        return 2
    teacher, run = M2_TEACHER, None
    if args.run is not None:
        try:
            teacher, run = read_run(args.run)
        except ValueError as exc:  # json.JSONDecodeError included
            print(f"not a usable run: {exc}", file=sys.stderr)
            return 2
    count = count_tokens or token_counter(args.tokenizer)
    cache = ReadOnlyCache(args.cache)
    try:
        records = load_records(args.records)
        joined = join(records, lambda rid: transcript_at(transcripts, rid), cache.get, teacher=teacher,
                      run=None if run is None else run["name"])
        rows = [j.row() for j in joined if j.status == TEACHER]
    except (JoinError, ReasoningError, RecordError) as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        cache.close()
    sha, size = write_sidecar(rows, args.out)
    inputs = {"records": {"path": args.records.as_posix(), "sha256": sha256_file(args.records)},
              "transcripts": transcripts.as_posix(), "cache": args.cache.as_posix(),
              "tokenizer": None if count_tokens else args.tokenizer.as_posix()}
    output = {"path": args.out.as_posix(), "rows": len(rows), "sha256": sha, "bytes": size}
    report = build_report(records, joined, rows, count, inputs, output, teacher=teacher, run=run)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8", newline="\n")
    tail = (f"differences from the plan {report['difference']}" if run is None else
            f"keys: {PLAIN} {report['keys'][PLAIN]}, {OFF} {report['keys'][OFF]} (run {run['name']})")
    print(f"{report['turns']['total']} turns of {len(records)} records: "
          + ", ".join(f"{s} {report['status'][s]}" for s in STATUSES)
          + f"; non-empty {report['non_empty']}, answer equal {report['answer_equal']['all']}, context "
          f"equal {report['context_equal']['all']} of {report['turns']['teacher_trajectory']}; {tail}")
    print(f"{len(rows)} rows -> {args.out.as_posix()} ({size} bytes); report -> {args.report.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

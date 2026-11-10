"""The synthetic training file: accepted train trajectories of generator runs, one training record each
(docs/plan.md, 微调数据 and 去重与防泄漏; scripts/assemble_synth.py is the command line).

Selection. Every index row of every given run is seen; a row is selected when its split is train, its
verdict (the owner's check_trajectory) accepted it and its turn ended answered or as a clarification. Rows of
the dev and test splits are never selected: they are read only for the family check below, so the dev and
test runs can be passed next to the train run.

Record. A selected row's transcript becomes one record in the agent-history form that
research_agent.train.samples.load_records reads and build_step_samples cuts into per-step samples:
{"messages": the API-format history exactly as the transcript holds it (system message first),
"tools": tools.schemas.MODEL_TOOLS, "history": "agent"} plus provenance the trainer does not use: the
request id ("id"), intent, family, scenario, lang, the run's name ("run"), prompt_hash and schema_hash (the
transcript's; the trainer refuses a missing schema_hash or one other than the current one, refuses "tools"
not JSON-equal to MODEL_TOOLS, and renders MODEL_TOOLS itself, since canonical JSON sorts the stored copy's
keys and the template renders a tool's keys in dict order). The marker "history": "agent"
makes the trainer cut the record like the agent loop's own history (older tool outputs compressed by
agent.context.build_context), not like a public sample, whose tool outputs are rendered whole.

Steps accepted after a repair. The model answered a repair request the history does not hold
(agent.context.step_contexts). The trainer drops such a step only for a transcript, which carries the step
records; a {messages, tools} record carries none, so the trainer cannot tell. The assembler therefore drops
every trajectory with such a step (counted under AFTER_REPAIR) rather than train on a context the teacher
never saw.

Think tags. The Qwen3 chat template splits an assistant turn's text at "</think>" and renders the part
before it as reasoning, so an answer that quotes a think tag (one does: it cites a paper's
"<think>β</think><answer>α</answer>" format) renders differently as a completion than as a prompt prefix,
and the trainer refuses the sample and stops. Such trajectories are dropped (THINK_TAG).

Evidence repair. Before the dedup, every trajectory goes through data.evidence_repair.repair_evidence (its
docstring has the rules): a scaffold write_note call whose evidence is not quoted from the tool outputs (the
e2e evidence check) gets sentences quoted from the paper text the model could see in the step before the
note, and a final answer that restates a detail only the old evidence held is removed. A trajectory with a
note the repair cannot fix, because the step before it held no text of the note's paper, is dropped
(NO_VISIBLE_TEXT). The record holds the repaired history; Assembled.evidence_repair() says what was done.

Dedup and leakage. Two records whose user and assistant turns are identical after normalize_text (content,
plus the tool calls in canonical JSON, so that two turns of empty content with different calls stay apart)
are one: the one with the smaller request id is kept (DUPLICATE). A train record whose request family also
appears among the dev or test rows of the given runs is dropped (FAMILY_LEAK); the templates are split by
family before generation, so this should never fire, and the count says whether it did. Records are written
in request-id order, so equal input gives a byte-identical file.

Malformed input (a run without manifest.json or index.jsonl, a line that is not a JSON object, a row
lacking a key the selection reads, a request id repeated across runs, a missing transcript or one with keys
Transcript does not have) raises AssembleError.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..agent.types import ANSWERED, CLARIFICATION, Transcript
from ..canonical import canonical_json, split_lines
from ..normalize import normalize_text
from ..tools.schemas import MODEL_TOOLS
from ..train.samples import AGENT_HISTORY, HISTORY_KEY, accepted_after_repair
from .evidence_repair import repair_evidence
from .synth import INDEX_FILE, MANIFEST_FILE, PLANNED_MIX, SCENARIOS

TRAIN, HELD_OUT = "train", ("dev", "test")
SELECTED_STATUSES = (ANSWERED, CLARIFICATION)
OTHER_SPLIT = "other split"
NOT_ACCEPTED = "not accepted"
OTHER_STATUS = "status not answered or clarification"
AFTER_REPAIR = "a step accepted after a repair"
THINK_TAG = "think tag in assistant text"
THINK_TAGS = ("<think>", "</think>")
FAMILY_LEAK = "family in dev or test"
NO_VISIBLE_TEXT = "a note step without visible paper text"
DUPLICATE = "duplicate"
DROP_REASONS = (OTHER_SPLIT, NOT_ACCEPTED, OTHER_STATUS, AFTER_REPAIR, THINK_TAG, FAMILY_LEAK,
                NO_VISIBLE_TEXT, DUPLICATE)
ROW_KEYS = ("id", "split", "status", "verdict", "family", "scenario", "lang", "transcript")
PROVENANCE = ("id", "intent", "family", "scenario", "lang", "run", "prompt_hash", "schema_hash")


class AssembleError(ValueError):
    """Malformed input to the assembler; the message names the file and the line."""


@dataclass
class SourceRun:
    """One generator run folder as the assembler reads it."""

    path: Path
    name: str
    manifest_sha256: str
    prompt_hash: str | None
    schema_hash: str | None
    requests_sha256: str | None
    rows: list[dict] = field(default_factory=list)

    def describe(self) -> dict:
        """What the training manifest records of the run."""
        return {"name": self.name, "path": self.path.as_posix(), "manifest_sha256": self.manifest_sha256,
                "prompt_hash": self.prompt_hash, "schema_hash": self.schema_hash,
                "requests_sha256": self.requests_sha256, "rows": len(self.rows),
                "by_split": dict(sorted(Counter(row["split"] for row in self.rows).items()))}


@dataclass
class Assembled:
    """The records in request-id order and the counts of how they came about."""

    records: list[dict]
    runs: list[SourceRun]
    rows_seen: int
    selected: int
    dropped: dict[str, int]
    repairs: dict[str, dict] = field(default_factory=dict)  # kept id -> {"calls", "final_answer_dropped"}
    no_visible_text: list[str] = field(default_factory=list)  # ids dropped as NO_VISIBLE_TEXT
    scaffold_calls: int = 0  # scaffold write_note calls in the kept records

    def counts(self) -> dict:
        kept = self.records
        by_scenario = Counter(r["scenario"] for r in kept)
        by_lang = Counter(r["lang"] for r in kept)
        by_both = Counter((r["scenario"], r["lang"]) for r in kept)
        scenarios = [s for s in SCENARIOS if s in by_scenario] + sorted(set(by_scenario) - set(SCENARIOS))
        return {"rows_seen": self.rows_seen, "selected": self.selected, "dropped": dict(self.dropped),
                "kept": len(kept), "by_scenario": {s: by_scenario[s] for s in scenarios},
                "by_lang": dict(sorted(by_lang.items())),
                "by_scenario_lang": {s: {lang: by_both[s, lang] for lang in sorted(by_lang)}
                                     for s in scenarios}}

    def evidence_repair(self) -> dict:
        """What the evidence repair did to the kept records: the manifest's evidence_repair block."""
        calls = [c for r in self.repairs.values() for c in r["calls"]]
        sources: Counter = Counter()
        for c in calls:
            sources.update(c["sources"])
        sentences = Counter(c["sentences"] for c in calls)
        return {"scaffold_calls": self.scaffold_calls, "calls_repaired": len(calls),
                "records_repaired": sum(1 for r in self.repairs.values() if r["calls"]),
                "sentences": {str(n): sentences[n] for n in sorted(sentences)},
                "sources": dict(sorted(sources.items())),
                "final_answers_removed": sorted(rid for rid, r in self.repairs.items()
                                                if r["final_answer_dropped"]),
                "repaired": sorted(rid for rid, r in self.repairs.items() if r["calls"]),
                "no_visible_text": sorted(self.no_visible_text)}

    def mix(self) -> dict:
        """The per-scenario mix of the kept records next to the plan's (synth.PLANNED_MIX)."""
        counts = self.counts()["by_scenario"]
        total = sum(counts.values())
        return {"counts": counts,
                "shares": {s: round(n / total, 4) if total else 0.0 for s, n in counts.items()},
                "planned": dict(PLANNED_MIX)}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_run(run_dir: str | Path) -> SourceRun:
    """The manifest facts and the index rows of a run folder; AssembleError when either is malformed."""
    run_dir = Path(run_dir)
    manifest_path = run_dir / MANIFEST_FILE
    index_path = run_dir / INDEX_FILE
    for path in (manifest_path, index_path):
        if not path.is_file():
            raise AssembleError(f"{path} is missing: every generator run writes it")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssembleError(f"{manifest_path} is not JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise AssembleError(f"{manifest_path} is not a JSON object")
    config = manifest.get("config") if isinstance(manifest.get("config"), dict) else {}
    name = config.get("run") if isinstance(config.get("run"), str) and config.get("run") else run_dir.name
    run = SourceRun(path=run_dir, name=name, manifest_sha256=sha256_file(manifest_path),
                    prompt_hash=manifest.get("prompt_hash"), schema_hash=manifest.get("schema_hash"),
                    requests_sha256=config.get("requests_sha256"))
    for number, line in enumerate(split_lines(index_path.read_text(encoding="utf-8")), 1):
        if not line.strip():
            continue
        where = f"{index_path}: line {number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AssembleError(f"{where} is not JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise AssembleError(f"{where} is not a JSON object")
        missing = [key for key in ROW_KEYS if key not in row]
        if missing:
            raise AssembleError(f"{where} lacks {', '.join(missing)}")
        run.rows.append(row)
    return run


def drop_reason(row: dict) -> str | None:
    """Why the selection leaves a row out (OTHER_SPLIT, NOT_ACCEPTED, OTHER_STATUS); None to select it."""
    if row["split"] != TRAIN:
        return OTHER_SPLIT
    verdict = row["verdict"]
    if not (isinstance(verdict, dict) and verdict.get("accepted") is True):
        return NOT_ACCEPTED
    if row["status"] not in SELECTED_STATUSES:
        return OTHER_STATUS
    return None


def load_run_transcript(run: SourceRun, row: dict) -> Transcript:
    """The transcript a selected row names; AssembleError when it is missing or not a transcript."""
    if not isinstance(row["transcript"], str):
        raise AssembleError(f"{run.path / INDEX_FILE}: {row['id']} is selected but names no transcript")
    path = run.path / row["transcript"]
    try:
        return Transcript(**json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError as exc:
        raise AssembleError(f"{path} is missing: {row['id']} names it") from exc
    except json.JSONDecodeError as exc:
        raise AssembleError(f"{path} is not JSON: {exc}") from exc
    except TypeError as exc:
        raise AssembleError(f"{path} is not a transcript: {exc}") from exc


def to_record(transcript: Transcript, row: dict, run_name: str, messages: list[dict] | None = None) -> dict:
    """The training record of one trajectory (the module docstring lists its keys); messages, when given,
    stand in for the transcript's history (the repaired one)."""
    history = transcript.messages if messages is None else messages
    return {"messages": copy.deepcopy(history), "tools": copy.deepcopy(MODEL_TOOLS),
            HISTORY_KEY: AGENT_HISTORY, "id": row["id"], "intent": row.get("intent"), "family": row["family"],
            "scenario": row["scenario"], "lang": row["lang"], "run": run_name,
            "prompt_hash": transcript.prompt_hash, "schema_hash": transcript.schema_hash}


def has_think_tag(messages: list[dict]) -> bool:
    """True when an assistant message's text holds "<think>" or "</think>" (the module docstring says why)."""
    return any(tag in (m.get("content") or "") for m in messages if m.get("role") == "assistant"
               for tag in THINK_TAGS)


def content_key(messages: list[dict]) -> str:
    """The dedup key: the user and assistant turns after normalize_text, tool calls in canonical JSON."""
    turns = []
    for m in messages:
        if m.get("role") not in ("user", "assistant"):
            continue
        calls = [[(c.get("function") or {}).get("name"), _canonical_arguments(c)] for c in m.get("tool_calls")
                 or []]
        turns.append([m["role"], normalize_text(m.get("content") or ""), calls])
    return canonical_json(turns)


def _canonical_arguments(call: dict) -> object:
    raw = (call.get("function") or {}).get("arguments")
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError:
        return raw


def assemble(run_dirs: list[str | Path]) -> Assembled:
    """Read the runs and give the kept records with the counts (the module docstring has the rules)."""
    runs = [read_run(path) for path in run_dirs]
    owner: dict[str, Path] = {}
    for run in runs:
        for row in run.rows:
            if row["id"] in owner:
                raise AssembleError(f"request id {row['id']} appears in {owner[row['id']]} and in {run.path}")
            owner[row["id"]] = run.path
    held_out = {str(row["family"]) for run in runs for row in run.rows if row["split"] in HELD_OUT}
    dropped = Counter()
    candidates: list[tuple[SourceRun, dict]] = []
    for run in runs:
        for row in run.rows:
            reason = drop_reason(row)
            if reason:
                dropped[reason] += 1
            else:
                candidates.append((run, row))
    records, seen = [], set()
    repairs: dict[str, dict] = {}
    no_visible: list[str] = []
    scaffold_calls = 0
    for run, row in sorted(candidates, key=lambda pair: pair[1]["id"]):
        transcript = load_run_transcript(run, row)
        if any(accepted_after_repair(step) for step in transcript.steps):
            dropped[AFTER_REPAIR] += 1
            continue
        if has_think_tag(transcript.messages):
            dropped[THINK_TAG] += 1
            continue
        if str(row["family"]) in held_out:
            dropped[FAMILY_LEAK] += 1
            continue
        repair = repair_evidence(transcript.messages)
        if repair.unrepairable:
            dropped[NO_VISIBLE_TEXT] += 1
            no_visible.append(row["id"])
            continue
        key = content_key(repair.messages)
        if key in seen:
            dropped[DUPLICATE] += 1
            continue
        seen.add(key)
        records.append(to_record(transcript, row, run.name, repair.messages))
        scaffold_calls += repair.scaffold_calls
        if repair.changed:
            repairs[row["id"]] = {"calls": repair.calls, "final_answer_dropped": repair.final_answer_dropped}
    return Assembled(records=records, runs=runs, rows_seen=sum(len(run.rows) for run in runs),
                     selected=len(candidates), dropped={reason: dropped[reason] for reason in DROP_REASONS},
                     repairs=repairs, no_visible_text=no_visible, scaffold_calls=scaffold_calls)


def write_records(records: list[dict], path: str | Path) -> tuple[str, int]:
    """One canonical JSON line per record, UTF-8 with LF; the file's SHA-256 and size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(canonical_json(record) + "\n" for record in records).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest(), len(data)


def token_counts(path: str | Path, tokenizer, max_length: int) -> dict:
    """The token statistics of the written training file, taken the way scripts/train.py takes them.

    The file is read back with train.samples.load_records and every record is cut with build_step_samples,
    as the trainer does, so whatever the written form changes (the assembler writes canonical JSON) shows
    here. Every sample is counted without a length cut; those over max_length are the ones the trainer drops
    (build_step_samples drops a sample whose "tokens" exceed max_length, the same comparison), and "trained"
    holds the statistics of the rest in the form scripts/train.py's data_summary records them.
    """
    from ..train.samples import build_step_samples, load_records  # the record form is the trainer's

    records = load_records(path)
    lengths: list[int] = []
    for record in records:
        samples, _ = build_step_samples(record, tokenizer, max_length=10**9)
        lengths += [s["tokens"] for s in samples]
    kept = [n for n in lengths if n <= max_length]
    ordered = sorted(lengths)

    def at(q: float) -> int | None:
        return ordered[min(len(ordered) - 1, int(q * len(ordered)))] if ordered else None

    trained = {"min": min(kept), "mean": round(sum(kept) / len(kept)), "max": max(kept),
               "total": sum(kept)} if kept else None
    return {"records": len(records), "samples": len(lengths), "over_context": len(lengths) - len(kept),
            "within_context": len(kept), "context_tokens": max_length,
            "min": ordered[0] if ordered else None, "median": at(0.5),
            "mean": round(sum(lengths) / len(lengths)) if lengths else None, "p95": at(0.95),
            "max": ordered[-1] if ordered else None, "trained": trained}

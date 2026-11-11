"""The thinking lengths of M7's test session and its end-to-end calls over the training length: descriptions
M7's preregistration promised (docs/m7-预登记.md, 报告: 思考 and 端到端; 端到端的训练长度), completed by M9
(docs/m9-计划.md WP4). Aggregates only: counts, percentiles, maxima and the combined SHA-256 of the files
read, never an item's id, request or text, in the file written or on the screen.

    .venv/Scripts/python.exe scripts/thinking_lengths.py --validate <run folder of scripts/run_e2e.py>
    .venv/Scripts/python.exe scripts/thinking_lengths.py --dev <run folder of scripts/run_e2e.py> \\
        [--session DIR] [--out results/lengths-m7.json] [--force]

--session is M7's test session by default (SESSION, the folder scripts/run_test.py wrote); leave it out.

What is measured (SYSTEMS), in the session folder scripts/run_test.py wrote (its Outputs: <system>.jsonl, the
single-step records; e2e-<system>/transcripts/ and e2e-<system>/timing.json, the end-to-end transcripts and
their timing sidecar):

    final    single_step, e2e   its thinking lengths, over its single-step records and over every attempt of
                                its end-to-end transcripts
             e2e_length         its end-to-end calls held to FT3's training length, 5,120 tokens
                                (live.frozen.TRAINING_LENGTH), each request rebuilt from the transcripts
    finalg   single_step        its thinking lengths over its single-step records (it ran no end-to-end task)
    ft2      e2e_length         its end-to-end calls held to FT2's training length, 4,096 tokens
                                (tools.schemas.CONTEXT_TOKENS), as llama.cpp counted them

A thinking block (single_step, e2e) is over the turns with a thinking record (agent.types.AssistantTurn's
thinking, which llm.generation's counts fill through LocalModel: reasoning_tokens, answer_tokens, forced): n,
those turns; the median, 95th percentile and maximum of the reasoning tokens; the median and 95th percentile
of the answer tokens (eval.resources.percentile; null over no value, an answer whose count is null left out);
forced, the reasonings closed by force. A voted step's turn is its representative's (agent.vote). An
e2e_length block: training_length; calls, the local calls measured; over, those whose prompt and generation
together are longer than the training length; max, the longest of them in tokens (null without a call).

The requests rebuilt (final). The session's runner did not count the steps over the training length
(docs/m7-预登记.md, 端到端的训练长度), so they are counted here from the transcripts, as scripts/run_e2e.py's
StepLengths counts them while a run goes on. The request the loop (agent.loop) sent for an attempt is
build_context (agent.context) of the history before its step: the transcript's messages up to that step, the
system message and the request followed, for every earlier step, by its accepted message and one tool message
per executed call. An attempt after a repair was sent that context with the rejected exchange of the attempt
before it appended (agent.loop's rejected step with its errors, rebuilt from that attempt's record of its
content, its calls and its issues), one after a handoff the context alone. StepLengths itself
(scripts/run_e2e.py, loaded as a module) then measures every attempt, over a model that answers each request
with the attempt's turn: the request's prompt as family_token_counter of final's family in think mode
(TOKENIZER) counts it with the tool definitions (tools.schemas.MODEL_TOOLS), plus the reasoning tokens, the
forced close's tokens when the reasoning was closed by force (run_e2e.close_tokens: the ids
family.encode_close() appends) and the answer tokens of the attempt's thinking record
(run_e2e.generated_tokens). An attempt without a thinking record cannot be measured: the tokens it generated
are not in the transcript.

Counted by the server (ft2). FT2 ran on the legacy path, one request per call with its prompt as text: a
call's length is llama.cpp's prompt_n plus predicted_n, from the timings the end-to-end timing sidecar keeps
for every call (eval.timing; LocalClient sends cache_prompt false, so prompt_n is the whole prompt), over the
calls that came back from the local model (no error, neither paid nor cached).

The validation (docs/m9-计划.md WP4: the rebuild first reproduces run_e2e's counts on a development run; only
then is the session read). The run is a folder of scripts/run_e2e.py, final's run of the end-to-end
development tasks, whose manifest counted its steps while they ran. Its manifest must say that it was measured
as the session is measured here (measured_as: training_length 5,120, the context check on TOKENIZER, think
mode, and a count, steps_over_training_length not null). The transcript of every task record (its
"transcript") is rebuilt and measured; the run agrees when the calls measured from each task's transcript are
its record's local_calls (a task whose run raised has neither; tasks_whose_calls_differ counts the others),
and so their sums, the calls over the training length are the manifest's steps_over_training_length, and no
transcript was unreadable or malformed and no attempt unmeasured. --validate RUN does this alone; --dev RUN
does it first and reads the session only when the run agrees.

Bad input is counted, never shown ("problems", the non-zero counts, keyed <system>.<part>.<kind> for the
session and <kind> for the validation): unreadable, a records line or a transcript that is not a JSON object;
malformed, a records line with a thinking record whose fields are not of their kind, or a transcript whose
steps do not fit its messages or that holds such a thinking record (none of its attempts is then counted);
unmeasured, an attempt without a thinking record, or a call without llama.cpp's counts; and for final,
calls_not_measured, the local calls its timing sidecar shows came back that no measured attempt stands for (a
task whose transcript was lost, a transcript or attempt counted above, a call that left no attempt), or
calls_not_in_sidecar, measured attempts the sidecar has no call for.

Writes --out (results/lengths-m7.json by default, the lengths kind of research_agent.eval.report:
lengths.m7.<system>.* in results/numbers.json), JSON with two-space indent and LF, never over an existing file
unless --force is given: the milestone; the session folder and the validation's run folder, each with its
path, the files read and their combined SHA-256 (the SHA-256 of the sorted "relative path<TAB>SHA-256" lines,
one per file, as eval.acebench digests a run), the validation with the counts it compared; how each e2e_length
was counted; "systems", each system's blocks; "read", per system what the blocks were taken from (records,
the single-step records; transcripts and attempts, the transcripts whose attempts were counted and those
attempts; timed_calls, the calls its timing sidecar shows came back; tasks, the sidecar's tasks of a system
counted by the server); and the problems. Prints the same aggregates. Run it from the repository root: the
default paths are relative to it, and the run folder and the session must lie inside it, so that the file
names its inputs by relative paths alone (inside).

Exit 0 when the run agrees (--validate) or the file is written (--dev); 1 when the run does not agree (nothing
is written, the session is not read); 2 on a usage error, a folder outside the current one, an input missing
or not of its kind as a whole (a report, a records file, a transcripts folder, a timing sidecar), a run not
measured as the session is, a family that cannot be loaded, or any other exception while a folder is read,
named by its type alone (withheld: its message may quote what was read); 3 when --out exists and --force was
not given.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from research_agent.agent.context import TokenCounter, build_context, family_token_counter
from research_agent.agent.loop import _rejected_exchange
from research_agent.agent.types import AssistantTurn, HarnessError, StepIssue, ToolCall
from research_agent.eval.e2e import REPORT_FILE
from research_agent.eval.resources import percentile
from research_agent.family.core import Mode, load_family
from research_agent.live.frozen import TRAINING_LENGTH
from research_agent.llm.server import same_path
from research_agent.tools.schemas import CONTEXT_TOKENS, MODEL_TOOLS

SCRIPTS = Path(__file__).resolve().parent


def _load_run_e2e():
    """scripts/run_e2e.py as a module of its own name (scripts/ is no package): its StepLengths,
    generated_tokens and close_tokens measure a call here as they do while a run goes on."""
    spec = importlib.util.spec_from_file_location("_run_e2e_shared", SCRIPTS / "run_e2e.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_e2e = _load_run_e2e()

MILESTONE = "M7"
SESSION = Path("runs/test-m7/open-1")  # M7's test session (docs/m7-预登记.md, 会话)
OUT = Path("results/lengths-m7.json")
TOKENIZER = "models/Nanbeige4.1-3B"  # final's family (results/test-config-m7.json, adapters.final.tokenizer)
SINGLE_STEP, E2E, E2E_LENGTH = "single_step", "e2e", "e2e_length"
TRANSCRIPTS = "transcripts"
REPAIR, HANDOFF = "repair", "handoff"  # the answers of decide after which the loop asks for the step again
USAGE, DISAGREES, EXISTS = 2, 1, 3
ABOUT = ("M7's preregistered description of the thinking lengths and of the end-to-end calls over the "
         "training length (docs/m7-预登记.md; docs/m9-计划.md WP4), by scripts/thinking_lengths.py: "
         "aggregates only.")
COUNTED_BY = {
    "rebuilt": "each attempt's request rebuilt from the transcripts and measured as scripts/run_e2e.py's "
               "StepLengths measures a call: the prompt by final's family in think mode, the reasoning, a "
               "forced close and the answer tokens of its thinking record",
    "server": "llama.cpp's prompt_n plus predicted_n of every call that came back, from the end-to-end "
              "timing sidecar"}


@dataclass(frozen=True)
class Plan:
    """What is measured of one system (module docstring): the parts whose thinking lengths are taken, and the
    training length its end-to-end calls are held to, measured with the requests rebuilt or as llama.cpp
    counted them."""

    thinking: tuple[str, ...] = ()
    rebuilt: int | None = None
    server_counted: int | None = None


SYSTEMS = {"final": Plan(thinking=(SINGLE_STEP, E2E), rebuilt=TRAINING_LENGTH),
           "finalg": Plan(thinking=(SINGLE_STEP,)),
           "ft2": Plan(server_counted=CONTEXT_TOKENS)}

Measure = tuple[TokenCounter, int]  # the prompt counter and the tokens of a forced close


class InputError(ValueError):
    """An input missing or not of its kind as a whole, or a run not measured as the session is: exit 2. The
    message names a file by its path, never an item."""


class Malformed(ValueError):
    """A transcript whose steps do not fit its messages, or a thinking record whose fields are not of their
    kind: counted (module docstring: bad input)."""


def rel(path: Path) -> str:
    """path relative to the current folder in POSIX form when it lies inside it, else absolute (POSIX)."""
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return absolute.as_posix()


def inside(path: Path) -> bool:
    """Whether path lies inside the current folder, so that rel names it by a relative path."""
    return not Path(rel(path)).is_absolute()


def withheld(exc: Exception, folder: Path) -> str:
    """The FAIL line of an exception that stopped the reading of folder (module docstring): its type
    alone."""
    return (f"FAIL: {type(exc).__name__} stopped the reading of {rel(folder)}; its message is withheld, as "
            "it may quote what was read")


def _key(prefix: str, kind: str) -> str:
    return f"{prefix}.{kind}" if prefix else kind


def _whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


@dataclass
class Folder:
    """One input folder: the files read from it, each with its SHA-256, and the problems counted in them."""

    root: Path
    files: list[tuple[str, str]] = field(default_factory=list)  # (path within the folder, SHA-256)
    problems: Counter = field(default_factory=Counter)

    def label(self, name: str) -> str:
        return f"{rel(self.root)}/{name}"

    def read(self, name: str) -> bytes:
        """The bytes of the file at name (a POSIX path within the folder), its SHA-256 kept; OSError when it
        cannot be read."""
        data = (self.root / name).read_bytes()
        self.files.append((name, hashlib.sha256(data).hexdigest()))
        return data

    def json_object(self, name: str, what: str) -> dict:
        """The JSON object in the file at name; InputError, naming the file and what it should be, when it is
        missing or holds none."""
        try:
            data = self.read(name)
        except OSError:
            raise InputError(f"{self.label(name)}: missing or unreadable ({what})") from None
        try:
            value = json.loads(data.decode("utf-8"))
        except ValueError:  # not UTF-8 or not JSON
            value = None
        if not isinstance(value, dict):
            raise InputError(f"{self.label(name)}: not a JSON object ({what})")
        return value

    def digest(self) -> dict:
        """The folder's path, the files read and their combined SHA-256 (module docstring)."""
        lines = "".join(f"{name}\t{sha}\n" for name, sha in sorted(self.files))
        return {"path": rel(self.root), "files": len(self.files),
                "sha256": hashlib.sha256(lines.encode("utf-8")).hexdigest()}


# ---- thinking records ------------------------------------------------------------------------------------


def thinking_of(holder: dict) -> dict | None:
    """The thinking record of a single-step record or of an attempt, None when it has none; Malformed for one
    whose reasoning_tokens, answer_tokens (null allowed) or forced are not of their kind."""
    thinking = holder.get("thinking")
    if thinking is None:
        return None
    if (not isinstance(thinking, dict) or not _whole(thinking.get("reasoning_tokens"))
            or not (thinking.get("answer_tokens") is None or _whole(thinking.get("answer_tokens")))
            or not isinstance(thinking.get("forced"), bool)):
        raise Malformed("a thinking record whose fields are not of their kind")
    return thinking


def _percentile(values: list[int], q: float) -> float | None:
    return percentile(values, q) if values else None


def thinking_block(records: list[dict]) -> dict:
    """A part's thinking lengths over its thinking records (module docstring), in the order
    research_agent.eval.report reads them."""
    reasoning = [record["reasoning_tokens"] for record in records]
    answers = [record["answer_tokens"] for record in records if record.get("answer_tokens") is not None]
    return {"n": len(records), "reasoning_p50": _percentile(reasoning, 50),
            "reasoning_p95": _percentile(reasoning, 95), "reasoning_max": max(reasoning, default=None),
            "answer_p50": _percentile(answers, 50), "answer_p95": _percentile(answers, 95),
            "forced": sum(record["forced"] for record in records)}


def single_step(folder: Folder, system: str) -> tuple[dict, int]:
    """(the thinking block, the records read) of a system's single-step records, <system>.jsonl; a line that
    is not a JSON object and a malformed thinking record are counted."""
    name, at = f"{system}.jsonl", f"{system}.{SINGLE_STEP}"
    try:
        text = folder.read(name).decode("utf-8")
    except OSError:
        raise InputError(f"{folder.label(name)}: missing or unreadable (single-step records)") from None
    except UnicodeDecodeError:
        raise InputError(f"{folder.label(name)}: not UTF-8 text (single-step records)") from None
    found, records = [], 0
    for line in text.split("\n"):  # JSON lines end at a line feed (eval.lock.write_jsonl)
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            record = None
        if not isinstance(record, dict):
            folder.problems[f"{at}.unreadable"] += 1
            continue
        records += 1
        try:
            thinking = thinking_of(record)
        except Malformed:
            folder.problems[f"{at}.malformed"] += 1
            continue
        if thinking is not None:
            found.append(thinking)
    return thinking_block(found), records


# ---- the requests rebuilt --------------------------------------------------------------------------------


def rejected_exchange(attempt: dict) -> list[dict]:
    """The exchange a repair request appended after a rejected attempt (agent.loop: the rejected step and one
    tool message per call, its errors or SKIPPED), rebuilt from the attempt's record: its content and calls as
    the model gave them, and its issues; Malformed when the record does not hold them."""
    try:
        calls = [ToolCall(id=c["id"], name=c["function"]["name"], arguments_json=c["function"]["arguments"],
                          arguments=None) for c in attempt["tool_calls"]]
        issues = []
        for issue in attempt["issues"]:
            if not isinstance(issue["messages"], list):
                raise TypeError("an issue's messages are a list")
            issues.append(StepIssue(call_id=issue["call_id"], event=issue["event"],
                                    messages=list(issue["messages"])))
        turn = AssistantTurn(content=attempt["content"], tool_calls=calls, model=str(attempt.get("model")))
        return _rejected_exchange(turn, issues)
    except (KeyError, TypeError, HarnessError) as exc:
        raise Malformed("a rejected attempt whose record does not hold its calls and issues") from exc


def accepted_end(messages: list[dict], position: int, executed: list[dict]) -> int:
    """Where the messages of a step accepted at position end: its message, then one tool message per executed
    call, in call order (agent.loop); Malformed when the messages there are not those."""
    end = position + 1 + len(executed)
    if end > len(messages):
        raise Malformed("an accepted step without its messages")
    accepted, replies = messages[position], messages[position + 1:end]
    calls = accepted.get("tool_calls") or []
    if not isinstance(calls, list):
        raise Malformed("an accepted step whose calls are not a list")
    ids = [c.get("id") if isinstance(c, dict) else None for c in calls]
    if (accepted.get("role") != "assistant" or ids != [e.get("call_id") for e in executed]
            or [m.get("role") for m in replies] != ["tool"] * len(replies)
            or [m.get("tool_call_id") for m in replies] != ids):
        raise Malformed("an accepted step whose messages are not its calls and their replies")
    return end


def attempt_requests(transcript: dict) -> list[tuple[list[dict], dict]]:
    """(request, attempt) for every attempt of a transcript, in call order, the request rebuilt as the loop
    sent it (module docstring: the requests rebuilt); Malformed when the steps do not fit the messages. Every
    step but the last was accepted; the last was when messages follow it."""
    messages, steps = transcript.get("messages"), transcript.get("steps")
    if (not isinstance(messages, list) or len(messages) < 2 or not all(isinstance(m, dict) for m in messages)
            or not isinstance(steps, list) or not all(isinstance(s, dict) for s in steps)):
        raise Malformed("not a transcript of agent.loop")
    position, out = 2, []  # after the system message and the request
    for number, step in enumerate(steps):
        attempts, executed = step.get("attempts"), step.get("executed")
        if (not isinstance(attempts, list) or not all(isinstance(a, dict) for a in attempts)
                or not isinstance(executed, list) or not all(isinstance(e, dict) for e in executed)):
            raise Malformed("a step without its attempts and executed calls")
        context = build_context(messages[:position])
        request = context
        for index, attempt in enumerate(attempts):
            if index:
                action = attempts[index - 1].get("action")
                if action == REPAIR:
                    request = context + rejected_exchange(attempts[index - 1])
                elif action == HANDOFF:
                    request = context
                else:
                    raise Malformed("an attempt after one the loop did not ask again for")
            out.append((request, attempt))
        if number < len(steps) - 1 or position < len(messages):  # an accepted step
            position = accepted_end(messages, position, executed)
    if position != len(messages):
        raise Malformed("messages that no step accounts for")
    return out


class _Recorded:
    """The model StepLengths wraps here: it answers each request with the turn recorded for it."""

    name = "recorded"

    def __init__(self) -> None:
        self.turn: AssistantTurn | None = None

    def next_turn(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        return self.turn


class Lengths:
    """Calls measured as scripts/run_e2e.py's StepLengths measures them while a run goes on (module docstring:
    the requests rebuilt): StepLengths itself counts the calls and those over the training length; max is the
    longest call, its prompt as StepLengths counted it plus run_e2e.generated_tokens."""

    def __init__(self, count_tokens: TokenCounter, close_tokens: int, training_length: int) -> None:
        self.close_tokens = close_tokens
        self.recorded = _Recorded()
        self.prompt = 0  # the tokens of the request StepLengths measured last

        def counted(messages: list[dict], tools: list[dict]) -> int:
            self.prompt = count_tokens(messages, tools)
            return self.prompt

        self.steps = run_e2e.StepLengths(self.recorded, count_tokens=counted, training_length=training_length,
                                         close_tokens=close_tokens)
        self.max: int | None = None

    def add(self, request: list[dict], thinking: dict) -> None:
        """Measure one attempt: its request as the loop sent it, with the tool definitions the loop sends."""
        turn = AssistantTurn(content="", tool_calls=[], model=_Recorded.name, thinking=thinking)
        self.recorded.turn = turn
        self.steps.next_turn(request, MODEL_TOOLS)
        length = self.prompt + run_e2e.generated_tokens(turn, self.close_tokens)
        self.max = length if self.max is None else max(self.max, length)

    @property
    def calls(self) -> int:
        return self.steps.steps

    @property
    def over(self) -> int:
        return self.steps.over

    def block(self) -> dict:
        return {"training_length": self.steps.training_length, "calls": self.calls, "over": self.over,
                "max": self.max}


@dataclass
class Measured:
    """What a folder's transcripts gave: the thinking records of their attempts, the transcripts and the
    attempts read."""

    thinking: list[dict] = field(default_factory=list)
    transcripts: int = 0
    attempts: int = 0


def measure_transcripts(folder: Folder, names: list[str], lengths: Lengths | None, at: str,
                        at_length: str) -> Measured:
    """Read the transcripts at names, rebuild every attempt's request and measure each attempt with a thinking
    record into lengths (when given); the problems go under at (unreadable, malformed) and at_length
    (unmeasured), never naming a transcript."""
    measured = Measured()
    for name in names:
        try:
            transcript = json.loads(folder.read(name).decode("utf-8"))
        except (OSError, ValueError):  # missing, not UTF-8 or not JSON
            transcript = None
        if not isinstance(transcript, dict):
            folder.problems[_key(at, "unreadable")] += 1
            continue
        try:
            requests = attempt_requests(transcript)
            records = [thinking_of(attempt) for _, attempt in requests]
        except Malformed:
            folder.problems[_key(at, "malformed")] += 1
            continue
        measured.transcripts += 1
        measured.attempts += len(requests)
        for (request, _), thinking in zip(requests, records, strict=True):
            if thinking is None:
                folder.problems[_key(at_length, "unmeasured")] += 1
                continue
            measured.thinking.append(thinking)
            if lengths is not None:
                lengths.add(request, thinking)
    return measured


# ---- the timing sidecars ---------------------------------------------------------------------------------


def returned_calls(folder: Folder, name: str) -> tuple[int, list[dict]]:
    """(the tasks, the calls that came back from the local model) of an end-to-end timing sidecar
    (eval.timing.TaskTimer: no error, neither paid nor cached); InputError when it is missing or not one."""
    what = "an end-to-end timing sidecar"
    timing = folder.json_object(name, what)
    tasks = timing.get("tasks")
    if not isinstance(tasks, list) or not all(
            isinstance(task, dict) and isinstance(task.get("calls"), list)
            and all(isinstance(c, dict) for c in task["calls"]) for task in tasks):
        raise InputError(f"{folder.label(name)}: not {what} (tasks with their calls)")
    calls = [c for task in tasks for c in task["calls"]
             if c.get("error") is None and not c.get("paid") and not c.get("cached")]
    return len(tasks), calls


def server_counted(folder: Folder, system: str, training_length: int) -> tuple[dict, dict]:
    """(the e2e_length block, what was read) of a system whose calls llama.cpp counted (module docstring)."""
    tasks, calls = returned_calls(folder, f"e2e-{system}/{run_e2e.TIMING_FILE}")
    lengths = []
    for call in calls:
        server = call.get("server")
        if isinstance(server, dict) and _whole(server.get("prompt_n")) and _whole(server.get("predicted_n")):
            lengths.append(server["prompt_n"] + server["predicted_n"])
        else:
            folder.problems[f"{system}.{E2E_LENGTH}.unmeasured"] += 1
    block = {"training_length": training_length, "calls": len(lengths),
             "over": sum(length > training_length for length in lengths), "max": max(lengths, default=None)}
    return block, {"tasks": tasks, "timed_calls": len(calls)}


# ---- the session -----------------------------------------------------------------------------------------


def end_to_end(folder: Folder, system: str, plan: Plan, measure: Callable[[], Measure]) -> tuple[dict, dict]:
    """(the blocks, what was read) of a system's end-to-end transcripts: the e2e thinking block, and when its
    requests are rebuilt its e2e_length block, the calls measured held against the timing sidecar's."""
    base = f"e2e-{system}"
    directory = folder.root / base / TRANSCRIPTS
    names = sorted(f"{base}/{TRANSCRIPTS}/{path.name}" for path in directory.glob("*.json")) \
        if directory.is_dir() else []
    if not names:
        raise InputError(f"{folder.label(f'{base}/{TRANSCRIPTS}')}: no transcripts")
    lengths = None
    if plan.rebuilt is not None:
        count_tokens, close_tokens = measure()
        lengths = Lengths(count_tokens, close_tokens, plan.rebuilt)
    measured = measure_transcripts(folder, names, lengths, f"{system}.{E2E}", f"{system}.{E2E_LENGTH}")
    blocks, read = {}, {"transcripts": measured.transcripts, "attempts": measured.attempts}
    if E2E in plan.thinking:
        blocks[E2E] = thinking_block(measured.thinking)
    if lengths is not None:
        blocks[E2E_LENGTH] = lengths.block()
        _, calls = returned_calls(folder, f"{base}/{run_e2e.TIMING_FILE}")
        read["timed_calls"] = len(calls)
        gap = len(calls) - lengths.calls
        if gap:
            kind = "calls_not_measured" if gap > 0 else "calls_not_in_sidecar"
            folder.problems[f"{system}.{E2E_LENGTH}.{kind}"] += abs(gap)
    return blocks, read


def summarize_session(session: Path, measure: Callable[[], Measure]) -> tuple[dict, dict, Folder]:
    """(the systems' blocks, what was read per system, the folder) of the session (module docstring)."""
    folder = Folder(session)
    systems, read = {}, {}
    for system, plan in SYSTEMS.items():
        blocks, seen = {}, {}
        if SINGLE_STEP in plan.thinking:
            blocks[SINGLE_STEP], seen["records"] = single_step(folder, system)
        if E2E in plan.thinking or plan.rebuilt is not None:
            more, counts = end_to_end(folder, system, plan, measure)
            blocks.update(more)
            seen.update(counts)
        if plan.server_counted is not None:
            blocks[E2E_LENGTH], counts = server_counted(folder, system, plan.server_counted)
            seen.update(counts)
        systems[system], read[system] = blocks, seen
    return systems, read, folder


# ---- the validation --------------------------------------------------------------------------------------


def measured_as(manifest: dict, tokenizer: str, training_length: int) -> list[str]:
    """How a run's manifest says it was measured otherwise than the session is (module docstring: the
    validation); [] when it was measured alike."""
    differ = []
    length = manifest.get("training_length")
    if not _whole(length) or length != training_length:
        differ.append(f"training_length {length!r}, not {training_length}")
    check = manifest.get("context_check")
    if not isinstance(check, str) or not same_path(check, tokenizer):
        differ.append(f"the context check on {check!r}, not on {tokenizer}")
    generation = manifest.get("generation")
    mode = generation.get("mode") if isinstance(generation, dict) else None
    if mode != str(Mode.THINK):
        differ.append(f"mode {mode!r}, not {Mode.THINK}")
    if not _whole(manifest.get("steps_over_training_length")):
        differ.append(f"steps_over_training_length {manifest.get('steps_over_training_length')!r}, not a "
                      "count of the steps over the training length")
    return differ


def validate(run: Path, measure: Callable[[], Measure], *, tokenizer: str = TOKENIZER,
             training_length: int = TRAINING_LENGTH) -> dict:
    """The validation on a run folder of scripts/run_e2e.py (module docstring): its path, files and digest,
    and the counts compared; InputError when its report is missing or not one, or when it was not measured as
    the session is."""
    folder = Folder(run)
    what = "a report of scripts/run_e2e.py"
    report = folder.json_object(REPORT_FILE, what)
    manifest, tasks = report.get("manifest"), report.get("tasks")
    if not isinstance(manifest, dict) or not isinstance(tasks, list) or not all(
            isinstance(task, dict) and _whole(task.get("local_calls"))
            and isinstance(task.get("transcript"), str | None) for task in tasks):
        raise InputError(f"{folder.label(REPORT_FILE)}: not {what} (a manifest, and task records with their "
                         "transcript and local_calls)")
    differ = measured_as(manifest, tokenizer, training_length)
    if differ:
        raise InputError(f"{folder.label(REPORT_FILE)}: the run was not measured as the session is: "
                         + "; ".join(differ))
    try:
        count_tokens, close_tokens = measure()
    except (OSError, ValueError) as exc:  # a FamilyError is a ValueError
        raise InputError(f"final's family cannot be loaded from {tokenizer}: {exc}") from None
    lengths = Lengths(count_tokens, close_tokens, training_length)
    without = differ_calls = 0
    for task in tasks:  # one by one: each record's local calls against those its transcript gives
        before = lengths.calls
        if task.get("transcript") is None:
            without += 1
        else:
            measure_transcripts(folder, [task["transcript"]], lengths, "", "")
        differ_calls += (lengths.calls - before) != task["local_calls"]
    return {**folder.digest(), "training_length": training_length, "tasks": len(tasks),
            "tasks_without_transcript": without, "tasks_whose_calls_differ": differ_calls,
            "local_calls": sum(task["local_calls"] for task in tasks), "calls": lengths.calls,
            "steps_over_training_length": manifest["steps_over_training_length"], "over": lengths.over,
            "problems": dict(sorted(folder.problems.items()))}


def agrees(validation: dict) -> bool:
    """Whether the rebuild reproduced the run's counts (module docstring: the validation)."""
    return (validation["calls"] == validation["local_calls"] and not validation["tasks_whose_calls_differ"]
            and validation["over"] == validation["steps_over_training_length"] and not validation["problems"])


# ---- output ----------------------------------------------------------------------------------------------


def _tokens(value: float | None) -> str:
    return "-" if value is None else f"{value:.0f}"


def _problems(problems: dict) -> str:
    return ", ".join(f"{key} {count}" for key, count in problems.items()) or "none"


def describe(part: str, block: dict) -> str:
    """One block on one line: aggregates only."""
    if part == E2E_LENGTH:
        return (f"{block['calls']} call(s), {block['over']} over {block['training_length']} tokens, the "
                f"longest {_tokens(block['max'])}")
    rate = f" ({100 * block['forced'] / block['n']:.1f}%)" if block["n"] else ""
    return (f"n {block['n']}, reasoning p50 {_tokens(block['reasoning_p50'])} p95 "
            f"{_tokens(block['reasoning_p95'])} max {_tokens(block['reasoning_max'])}, answer p50 "
            f"{_tokens(block['answer_p50'])} p95 {_tokens(block['answer_p95'])}, forced "
            f"{block['forced']}{rate}")


def print_validation(validation: dict) -> None:
    print(f"validation on {validation['path']}: {validation['tasks']} task(s), "
          f"{validation['tasks_without_transcript']} without a transcript; {validation['files']} file(s) "
          f"read, SHA-256 {validation['sha256']}")
    print(f"  local calls: {validation['calls']} measured, the task records count "
          f"{validation['local_calls']}; {validation['tasks_whose_calls_differ']} task(s) whose calls differ")
    print(f"  calls over {validation['training_length']} tokens: {validation['over']} measured, the manifest "
          f"counts {validation['steps_over_training_length']}")
    print(f"  problems: {_problems(validation['problems'])}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="M7's thinking lengths and end-to-end calls over the training "
                                             "length, from its test session: aggregates only.")
    first = ap.add_mutually_exclusive_group(required=True)
    first.add_argument("--validate", type=Path, metavar="RUN",
                       help="check only: the rebuild must reproduce the counts of this run folder of "
                            "scripts/run_e2e.py (final's development run)")
    first.add_argument("--dev", type=Path, metavar="RUN",
                       help="validate on this run folder first, then read the session and write --out")
    ap.add_argument("--session", type=Path, default=None,
                    help=f"with --dev: the session folder (default {SESSION.as_posix()})")
    ap.add_argument("--out", type=Path, default=None, help=f"with --dev: default {OUT.as_posix()}")
    ap.add_argument("--force", action="store_true", help="with --dev: overwrite an existing --out")
    return ap


def family_measure(tokenizer: str = TOKENIZER) -> Measure:
    """final's prompt counter in think mode and the tokens of its forced close, as scripts/run_e2e.py measures
    a step (module docstring: the requests rebuilt); loads the family's tokenizer."""
    family = load_family(tokenizer)
    return family_token_counter(family, Mode.THINK), run_e2e.close_tokens(family)


def main(argv: list[str] | None = None, *, measure: Callable[[], Measure] = family_measure) -> int:
    """measure stands in for family_measure in tests (an injected counter and forced close)."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.validate is not None and (args.session is not None or args.out is not None or args.force):
        parser.error("--session, --out and --force go with --dev: --validate reads no session and writes "
                     "nothing")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    run = args.validate or args.dev
    session = SESSION if args.session is None else args.session
    outside = [rel(path) for path in ([run] if args.dev is None else [run, session]) if not inside(path)]
    if outside:
        print(f"FAIL: run it from the repository root: {', '.join(outside)} lie(s) outside the current "
              "folder, and the results name their inputs by relative paths alone", file=sys.stderr)
        return USAGE
    out = OUT if args.out is None else args.out
    if args.dev is not None and out.exists() and not args.force:
        print(f"FAIL: {rel(out)} exists; --force overwrites it", file=sys.stderr)
        return EXISTS
    loaded: list[Measure] = []

    def measured() -> Measure:  # the family is loaded once, and only when a call is measured
        if not loaded:
            loaded.append(measure())
        return loaded[0]

    try:
        validation = validate(run, measured)
    except InputError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return USAGE
    except Exception as exc:  # noqa: BLE001  module docstring: named by its type alone
        print(withheld(exc, run), file=sys.stderr)
        return USAGE
    print_validation(validation)
    if not agrees(validation):
        print("FAIL: the rebuild does not reproduce the run's counts; the session is not read")
        return DISAGREES
    if args.validate is not None:
        print("OK: the rebuild reproduces the run's counts")
        return 0
    try:
        systems, read, folder = summarize_session(session, measured)
    except InputError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return USAGE
    except Exception as exc:  # noqa: BLE001  module docstring: named by its type alone
        print(withheld(exc, session), file=sys.stderr)
        return USAGE
    counted_by = {system: COUNTED_BY["rebuilt" if plan.rebuilt is not None else "server"]
                  for system, plan in SYSTEMS.items()
                  if plan.rebuilt is not None or plan.server_counted is not None}
    data = {"milestone": MILESTONE, "about": ABOUT, "session": folder.digest(), "validation": validation,
            "e2e_length_counted_by": counted_by, "systems": systems, "read": read,
            "problems": dict(sorted(folder.problems.items()))}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    session = data["session"]
    print(f"session {session['path']}: {session['files']} file(s) read, SHA-256 {session['sha256']}")
    for system, blocks in systems.items():
        for part, block in blocks.items():
            print(f"  {system:<6} {part:<11} {describe(part, block)}")
    print("  read: " + "; ".join(f"{system} " + ", ".join(f"{key} {count}" for key, count in seen.items())
                                 for system, seen in read.items()))
    print(f"  problems: {_problems(data['problems'])}")
    print(f"{rel(out)}: written")
    return 0


if __name__ == "__main__":
    sys.exit(main())

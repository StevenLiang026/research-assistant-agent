"""One live turn and what a run writes (docs/m9-计划.md WP1: 一轮的运行, 追问后继续, 进度与输出, 单实例锁).

A turn. run_live_turn runs a request as the evaluation does: agent.loop.run_turn with the owner's check_call
and policy.decide, the built system's context counter and its cost, no handoff model, and one today (the
machine's date, taken once per turn by the caller) for the system prompt, the executor and the tools. It never
goes through data.synth.run_request or eval.e2e.run_tasks, which empty the notes folder before every task. A
turn that raises is recorded (LiveTurn.error) and the run goes on.

A follow-up. A turn that ends in a clarification is continued as a new turn whose only user message is
pack_follow_up(request, answer): the request as it was, then the answer, joined by FOLLOW_UP_ZH ("补充:",
with the ASCII colon of the training data's Chinese) when the request holds a CJK character, else by
FOLLOW_UP_EN (" To clarify: "); the answer's whitespace is collapsed, so the message stays one line, the
only kind the model was trained and evaluated on. An empty answer packs nothing (None). A request is
continued at most MAX_FOLLOW_UPS times; after that the user restates it.

Progress (ASCII only, never the model's or a tool's text). The wrappers only look: they print and hand on what
they were given and what came back, unchanged. ProgressModel prints one line per model request: the step (one
more than the assistant messages of the history the loop asks with) and the attempt within it, the vote (the
samples in the chosen class of the K), the reasoning tokens and a close by force, the runtime rules that
fired, the calls chosen (their names, with the arxiv_id they name) or the answer's length, and the seconds;
everything else (name, cost, begin_task...) is the wrapped model's. ProgressClient is LocalClient with one
line per llama-server request (prompt and generated tokens, finish reason, seconds); its requests,
last_timings included, are LocalClient's. progress_tools prints one line per tool call: the outcome in
counts, ids, error codes and the note's path, the seconds and the rate-limit wait. Each wrapper prints through
SafeOut: a line that cannot be printed (a closed console, a pipe whose reader went away: OSError, ValueError)
ends that wrapper's printing and nothing else, so a vote never loses a candidate, nor a call its outcome, to
the console.

Files of a run (SessionFiles), under the run's folder runs/live/<start stamp>/: session.json (written by
scripts/ask.py: the identity and its check), server.log (the server's output, when the run started it),
requests.jsonl (live.world), per turn turn-NNN.json (the transcript, agent.loop.save_transcript; for a turn
that raised, its request, today and error), turn-NNN.md (the request, today, status, the answer or question,
the notes written and the papers cited) and, when it wrote notes, turn-NNN.papers.json (the metadata each note
was made from, which a check holds the evidence sentences against); and one line per turn in the index beside
the run folders (INDEX_FILE).

The lock. InstanceLock holds an operating-system lock on a file (msvcrt on Windows, fcntl elsewhere) for as
long as a run lasts, so one run at a time talks to arXiv and the GPU; the system releases it when the
process ends, however it ends. Beside it, LOCK_FILE + ".json" names the holder's pid and start, which
LockBusy reports.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Self

from .. import policy
from ..agent.executor import ToolExecutor, _paper_ids
from ..agent.loop import Model, current_history, run_turn, save_transcript
from ..agent.types import CLARIFICATION, AssistantTurn, ToolCall, ToolFn, ToolOutcome, Transcript
from ..eval.metrics import arxiv_ids
from ..eval.systems import Built
from ..llm.local import Completion, LocalClient
from ..net.fetch import utc_now
from ..notes import _CJK
from ..validation import check_call

LOCK_FILE = "ask.lock"
SESSION_FILE = "session.json"
SERVER_LOG = "server.log"
INDEX_FILE = "index.jsonl"  # beside the run folders
MAX_FOLLOW_UPS = 2
FOLLOW_UP_ZH, FOLLOW_UP_EN = "补充:", " To clarify: "

Out = Callable[[str], None]


def ascii_text(value: object) -> str:
    """value as text in ASCII alone, anything else escaped."""
    return str(value).encode("ascii", "backslashreplace").decode("ascii")


# ---- a follow-up ------------------------------------------------------------------------------------------


def pack_follow_up(request: str, answer: str) -> str | None:
    """The request continued by the answer to its clarification (module docstring: A follow-up); None for an
    empty answer."""
    answer = " ".join(answer.split())
    if not answer:
        return None
    return request + (FOLLOW_UP_ZH if _CJK.search(request) else FOLLOW_UP_EN) + answer


# ---- the lock ---------------------------------------------------------------------------------------------


class LockBusy(Exception):
    """Another run holds the lock; pid is the holder's, when its record can be read."""

    def __init__(self, path: Path, pid: int | None) -> None:
        holder = "another process" if pid is None else f"process {pid}"
        super().__init__(f"{holder} holds {path.as_posix()}: one run at a time")
        self.path, self.pid = path, pid


def _lock(handle: IO[bytes]) -> None:
    handle.seek(0)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle: IO[bytes]) -> None:
    handle.seek(0)
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class InstanceLock:
    """One run at a time (module docstring: The lock): acquire() or LockBusy, release() (safe to repeat), or
    with InstanceLock(path): ..."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.record = self.path.with_name(self.path.name + ".json")
        self._handle: IO[bytes] | None = None

    def holder(self) -> int | None:
        """The pid the record names, None when it cannot be read."""
        try:
            pid = json.loads(self.record.read_text(encoding="utf-8")).get("pid")
        except (OSError, ValueError, AttributeError):
            return None
        return pid if isinstance(pid, int) and not isinstance(pid, bool) else None

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+b")  # noqa: SIM115  held open for as long as the lock is
        try:
            _lock(handle)
        except OSError:
            handle.close()
            raise LockBusy(self.path, self.holder()) from None
        self._handle = handle
        _write_json(self.record, {"pid": os.getpid(), "started": utc_now()})

    def release(self) -> None:
        handle, self._handle = self._handle, None
        if handle is None:
            return
        try:
            self.record.unlink(missing_ok=True)
        finally:
            try:
                _unlock(handle)
            finally:
                handle.close()

    def __enter__(self) -> Self:
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


# ---- progress ---------------------------------------------------------------------------------------------


class SafeOut:
    """out for a wrapper that only looks (module docstring: Progress): the first line that cannot be printed
    (OSError, ValueError) ends the printing, silently; out None prints nothing."""

    def __init__(self, out: Out | None) -> None:
        self.out, self.broken = out, out is None

    def __call__(self, line: str) -> None:
        if self.broken:
            return
        try:
            self.out(line)
        except (OSError, ValueError):  # a closed stream, or a pipe whose reader went away
            self.broken = True


def call_label(call: ToolCall) -> str:
    """A call's name, with the arxiv_id it names."""
    arxiv_id = (call.arguments or {}).get("arxiv_id")
    return f"{call.name}({arxiv_id})" if isinstance(arxiv_id, str) else call.name


def step_line(step: int, attempt: int, turn: AssistantTurn, seconds: float) -> str:
    """ProgressModel's line for a turn (module docstring: Progress)."""
    parts = [f"step {step}" + (f" attempt {attempt}" if attempt > 1 else "")]
    votes = turn.votes
    if isinstance(votes, dict):
        chosen, counts = votes.get("chosen"), votes.get("counts") or {}
        parts.append(f"votes {counts.get(chosen, 0)}/{votes.get('k')}" if chosen is not None
                     else f"votes none of {votes.get('k')}, the greedy answer")
        if votes.get("errors"):
            parts.append(f"{len(votes['errors'])} candidate(s) failed")
    thinking = turn.thinking
    if isinstance(thinking, dict):
        parts.append(f"thinking {thinking.get('reasoning_tokens')} tok"
                     + (" closed by force" if thinking.get("forced") else ""))
    if turn.rules:
        parts.append("rules " + ", ".join(str(change.get("rule")) for change in turn.rules))
    calls = [call_label(call) for call in turn.tool_calls]
    parts.append("calls " + ", ".join(calls) if calls else f"answer {len(turn.content)} chars")
    parts.append(f"{seconds:.1f} s")
    return ascii_text(" | ".join(parts))


class ProgressModel:
    """The wrapped model, with one progress line per request (module docstring: Progress)."""

    def __init__(self, model: Model, out: Out, *, clock: Callable[[], float] = time.perf_counter) -> None:
        self.model, self.out, self.clock = model, SafeOut(out), clock
        self._step = self._attempt = 0

    @property
    def name(self) -> str:
        return self.model.name

    def begin_task(self) -> None:
        """A new task: the steps count from 1 again, and the wrapped model hears of it when it listens."""
        self._step = self._attempt = 0
        begin = getattr(self.model, "begin_task", None)
        if begin is not None:
            begin()

    def next_turn(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        history = current_history()
        step = 1 + sum(m.get("role") == "assistant" for m in (messages if history is None else history))
        self._attempt = self._attempt + 1 if step == self._step else 1
        self._step = step
        started = self.clock()
        try:
            turn = self.model.next_turn(messages, tools)
        except Exception as exc:
            self.out(ascii_text(f"step {step}: the model raised {type(exc).__name__} after "
                                f"{self.clock() - started:.1f} s"))
            raise
        self.out(step_line(step, self._attempt, turn, self.clock() - started))
        return turn

    def __getattr__(self, name: str):
        if name == "model":  # not set yet: no recursion through self.model
            raise AttributeError(name)
        return getattr(self.model, name)


class ProgressClient(LocalClient):
    """LocalClient with one progress line per completion (module docstring: Progress); out None prints
    nothing."""

    def __init__(self, base_url: str = "http://127.0.0.1:8080", *, out: Out | None = None,
                 timeout: float = 120.0, transport: object | None = None,
                 clock: Callable[[], float] = time.perf_counter) -> None:
        super().__init__(base_url, timeout=timeout, transport=transport)
        self.out, self.clock = SafeOut(out), clock
        self.completions = 0

    def complete(self, prompt: str | list[int], **kwargs) -> Completion:
        started = self.clock()
        completion = super().complete(prompt, **kwargs)
        self.completions += 1
        if not self.out.broken:
            usage = completion.raw.get("usage") or {}
            self.out(ascii_text(f"    llama-server request {self.completions}: prompt "
                                f"{usage.get('prompt_tokens')} tok, generated "
                                f"{usage.get('completion_tokens')} tok ({completion.finish_reason}), "
                                f"{self.clock() - started:.1f} s"))
        return completion


def tool_summary(name: str, outcome: object) -> str:
    """A tool's outcome in counts, ids, error codes and paths, never its text."""
    if not isinstance(outcome, ToolOutcome):
        return f"returned a {type(outcome).__name__}"
    content, error = outcome.content, outcome.error
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        text = f"{content.get('source')} text of {content.get('arxiv_id')}, {len(content['text'])} chars"
        return text if error is None else f"{error}, {text}"
    if error is not None:
        return f"error {error}"
    if isinstance(content, list):
        return f"{len(content)} {'hit' if name == 'search_arxiv' else 'note'}(s)"
    if name == "write_note":
        return f"note {content.get('note_id')} ({content.get('path')})"
    if "arxiv_id" in content:
        return f"found {content['arxiv_id']}"
    return "done"


def progress_tools(tools: dict[str, ToolFn], out: Out, *, waited: Callable[[], float] | None = None,
                   clock: Callable[[], float] = time.perf_counter) -> dict[str, ToolFn]:
    """tools, each printing one line per call (module docstring: Progress); waited gives the seconds the rate
    limits have slept so far."""
    show = SafeOut(out)

    def wrap(name: str, tool: ToolFn) -> ToolFn:
        def run(args: dict) -> ToolOutcome:
            before = waited() if waited is not None else 0.0
            started = clock()
            try:
                outcome = tool(args)
            except Exception as exc:
                show(ascii_text(f"  tool {name}: raised {type(exc).__name__}, {clock() - started:.1f} s"))
                raise
            line = f"  tool {name}: {tool_summary(name, outcome)}, {clock() - started:.1f} s"
            wait = (waited() - before) if waited is not None else 0.0
            if wait > 0:
                line += f", {wait:.1f} s of it waiting for the rate limit"
            show(ascii_text(line))
            return outcome

        return run

    return {name: wrap(name, tool) for name, tool in tools.items()}


# ---- a turn -----------------------------------------------------------------------------------------------


@dataclass
class LiveTurn:
    """A live turn: its transcript, or the error that ended it (module docstring: A turn)."""

    request: str
    today: str
    transcript: Transcript | None
    error: str | None
    seconds: float

    @property
    def status(self) -> str | None:
        return None if self.transcript is None else self.transcript.status

    @property
    def clarified(self) -> bool:
        return self.status == CLARIFICATION


def run_live_turn(text: str, *, built: Built, tools: dict[str, ToolFn], today: str, out: Out,
                  waited: Callable[[], float] | None = None,
                  clock: Callable[[], float] = time.perf_counter) -> LiveTurn:
    """One request through the loop as the evaluation runs it, with progress lines (module docstring)."""
    executor = ToolExecutor(progress_tools(tools, out, waited=waited, clock=clock), check_call, today=today)
    model = ProgressModel(built.model, out, clock=clock)
    started = clock()
    try:
        transcript = run_turn(text, model=model, executor=executor, decide=policy.decide, today=today,
                              count_tokens=built.count_tokens, cost_of=getattr(built.model, "cost", None))
    except Exception as exc:  # noqa: BLE001  deliberate: a turn that raised is recorded and the run goes on
        return LiveTurn(text, today, None, f"{type(exc).__name__}: {exc}", round(clock() - started, 3))
    return LiveTurn(text, today, transcript, None, round(clock() - started, 3))


# ---- what a turn did --------------------------------------------------------------------------------------


def tool_results(transcript: Transcript) -> list[tuple[str | None, object, str | None]]:
    """(tool name, body as the model read it, error code) of every tool message of the turn, in order."""
    names = {}
    for message in transcript.messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                names[call.get("id")] = (call.get("function") or {}).get("name")
    results = []
    for message in transcript.messages:
        if message.get("role") != "tool":
            continue
        try:
            body = json.loads(message.get("content") or "null")
        except ValueError:
            body = None
        error = body.get("error") if isinstance(body, dict) else None
        results.append((names.get(message.get("tool_call_id")), body, error))
    return results


def notes_written(transcript: Transcript) -> list[dict]:
    """The note records write_note gave in the turn ({"note_id", "arxiv_id", "path"}), in order."""
    return [body for name, body, error in tool_results(transcript)
            if name == "write_note" and error is None and isinstance(body, dict) and "note_id" in body]


def seen_ids(transcript: Transcript) -> set[str]:
    """The arxiv_ids the turn's tool outputs showed (agent.executor's reading)."""
    return {i for _, body, _ in tool_results(transcript) for i in _paper_ids(body)}


# ---- the files of a run -----------------------------------------------------------------------------------


def _write_json(path: Path, value: object) -> None:
    """Two-space JSON, UTF-8 and LF, with a final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def new_run_folder(runs: str | Path) -> Path:
    """A new folder under runs named by its start to the second, in ISO 8601's basic form (a suffix -2, -3...
    when that name is taken)."""
    runs = Path(runs)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    folder, number = runs / stamp, 1
    while folder.exists():
        number += 1
        folder = runs / f"{stamp}-{number}"
    folder.mkdir(parents=True)
    return folder


def _one_line(text: object) -> str:
    return " ".join(str(text).split())


def turn_markdown(number: int, turn: LiveTurn, *, item_id: str | None, follow_up: int, notes: list[dict],
                  cited: list[str], seen: set[str]) -> str:
    """turn-NNN.md (module docstring: Files of a run)."""
    lines = [f"# Turn {number:03d}", ""]
    if item_id is not None:
        lines.append(f"- id: {item_id}")
    lines += [f"- request: {_one_line(turn.request)}", f"- today: {turn.today}",
              f"- status: {turn.status or 'error'}"]
    if follow_up:
        lines.append(f"- follow-up: {follow_up}")
    if turn.transcript is not None:
        lines.append(f"- steps: {len(turn.transcript.steps)}")
    lines.append(f"- seconds: {turn.seconds:.1f}")
    if turn.transcript is None:
        heading, text = "Error", turn.error or ""
    else:
        heading = "Question" if turn.clarified else "Answer"
        text = turn.transcript.final_text or f"(none: the turn ended as {turn.status})"
    lines += ["", f"## {heading}", "", text.rstrip("\n"), "", "## Notes written", ""]
    lines += [f"- {note['note_id']} ({note['path']})" for note in notes] or ["- none"]
    lines += ["", "## Papers cited", ""]
    lines += [f"- {arxiv_id}" + ("" if arxiv_id in seen else " (in no tool output of this turn)")
              for arxiv_id in cited] or ["- none"]
    return "\n".join(lines) + "\n"


class SessionFiles:
    """The files of one run (module docstring: Files of a run) in folder; the index is folder's parent's."""

    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)
        self.index = self.folder.parent / INDEX_FILE
        self.turns = 0

    def write_session(self, data: dict) -> None:
        _write_json(self.folder / SESSION_FILE, data)

    def write_turn(self, turn: LiveTurn, *, item_id: str | None = None, follow_up: int = 0,
                   papers: Callable[[str], dict | None] | None = None) -> dict:
        """The turn's files and its index line, which is returned."""
        self.turns += 1
        number = self.turns
        stem = f"turn-{number:03d}"
        notes, cited, seen = [], [], set()
        if turn.transcript is not None:
            save_transcript(turn.transcript, self.folder / f"{stem}.json")
            notes, seen = notes_written(turn.transcript), seen_ids(turn.transcript)
            cited = sorted(arxiv_ids(turn.transcript.final_text or ""))
        else:
            _write_json(self.folder / f"{stem}.json", {"request": turn.request, "today": turn.today,
                                                       "status": None, "error": turn.error})
        if notes and papers is not None:
            _write_json(self.folder / f"{stem}.papers.json",
                        {note["arxiv_id"]: papers(note["arxiv_id"]) for note in notes})
        (self.folder / f"{stem}.md").write_text(
            turn_markdown(number, turn, item_id=item_id, follow_up=follow_up, notes=notes, cited=cited,
                          seen=seen), encoding="utf-8", newline="\n")
        row = {"run": self.folder.name, "turn": number, "id": item_id, "follow_up": follow_up,
               "request": turn.request, "today": turn.today, "status": turn.status, "error": turn.error,
               "steps": None if turn.transcript is None else len(turn.transcript.steps),
               "seconds": turn.seconds, "notes": [note["note_id"] for note in notes], "cited": cited,
               "cited_unseen": [arxiv_id for arxiv_id in cited if arxiv_id not in seen]}
        with self.index.open("a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

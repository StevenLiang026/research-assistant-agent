"""Ask M7's final system, unchanged, for help with the week's arXiv reading, on the real arXiv
(docs/m9-计划.md WP1).

    .venv/Scripts/python.exe scripts/ask.py "REQUEST" [--clarify-answer TEXT]...
    .venv/Scripts/python.exe scripts/ask.py --interactive
    .venv/Scripts/python.exe scripts/ask.py --batch FILE.jsonl [--batch-name NAME]
    .venv/Scripts/python.exe scripts/ask.py --check
    .venv/Scripts/python.exe scripts/ask.py --probe ID [ID ...]
    .venv/Scripts/python.exe scripts/ask.py --import-notes FILE
        [--llama-server PATH | --server URL] [--port N] [--notes DIR] [--runs DIR]

Run it from the repository root: the frozen record's paths are relative to it, the notes library and the run
folders go there by default, and the lock and the rate-limit state live there. Started from any other folder
it stops (exit 2) before it writes anything.

The system. What runs is what M7 evaluated: research_agent.live.frozen.load_frozen checks the code, the
frozen record (results/final-m7.json) and the M7 session configuration, eval.systems.build_system("ft", ...)
assembles the system from them exactly as the evaluation did (the family path in think mode with the
reasoning budget, a vote over five samples in the final mode, the runtime rules, the context check), and
identity_problems holds the built system and its server to the record. Any disagreement stops the run (exit
2) before a request is sent. Every turn is agent.loop.run_turn with the owner's check_call and decide
(research_agent.live.session).

The server. By default the script starts its own llama-server (llm.server.LocalServer: the frozen base model
with the frozen adapter, one slot, the configured context size and timeout, its output in the run's
server.log) once the GPU is idle (nvidia-smi: at most gpu.idle_max_mib MiB in use; with no nvidia-smi on PATH
it warns and starts anyway, while an nvidia-smi that fails stops the run), and stops it when the run ends,
Ctrl+C included. --llama-server or the LLAMA_SERVER environment variable names the binary. --server URL uses a
server already running there instead, checked the same way. Something already answering on the port is
refused (use --server, or another --port).

The tools are research_agent.live.world's: the arXiv tools on the real arXiv within its rate limits, held
across runs, and the notes library in --notes (default notes/, which git ignores), which lasts from one run to
the next. today is this machine's date, taken once per turn; there is no option to set it. When the rate-limit
state could not be saved after a run's latest request, the run waits out arXiv's intervals before it ends
(live.world.LoggedLimits.settle), so the next run cannot come too soon.

Modes, one per run:
- a request: one turn; when it ends in a clarification, --clarify-answer continues it (twice at most:
  live.session.pack_follow_up), and without one the question is printed. An answer left over, because a turn
  ended otherwise, is said to be unused (in --batch too).
- --interactive: "ask> " takes a request; after a clarification "answer> " takes the answer, where an empty
  answer leaves the request, "/new TEXT" asks TEXT instead and "/quit" ends the run (as it does at "ask> ").
  After two follow-ups the request has to be restated whole. The server starts once. The first line printed
  says so (CONSOLE_HELP); any other text starting with "/", and a bare "exit" or "quit", is answered with it
  and never sent to the model.
- --batch FILE: one JSON object per line, {"id", "request", "clarify_answer"?: a text or up to two}, run in
  order over one server; --batch-name NAME runs only the lines whose "batch" is NAME (the two halves of
  configs/live-smoke-m9.jsonl).
- --check: assemble the system, check it, print its identity and stop.
- --probe ID...: no GPU. get_paper through the live tools for each id, with the HTTP status of its request and
  what arxiv.parse_atom makes of the feed by default and leniently, then one search with a fixed query
  (PROBE_SEARCH); every request is sent once, without retries (net.fetch: a probe must not multiply its
  requests), so a 429 or 503 is what the probe records; the outcomes go to the run folder's probe.json.
- --import-notes FILE: the owner's notes file into the library (notes.import_entries, then
  NoteStore.load_entries tagged "imported", the entries it already holds skipped); prints counts only.

Every mode but --import-notes writes a run folder under --runs (default runs/live/) with session.json (the
identity, its check, the git commit; a probe's says it was one) and what research_agent.live.session lists:
server.log, requests.jsonl, turn-NNN.json, turn-NNN.md, turn-NNN.papers.json, and one index line per turn in
<runs>/index.jsonl. One run at a time: the lock and the rate-limit state live in runs/live/ whatever --runs
is.

Console text. A stream that is a terminal keeps its encoding (on Windows the console's own, which takes any
text); a pipe or a file gets UTF-8, the encoding of every file a run writes, and so does the input of
--interactive when it is not a terminal (input that is not UTF-8 ends the input, rather than reaching the
model garbled). What a stream cannot encode is escaped.

Exit 0 when the turns ran, whatever they answered; 1 when a turn raised (recorded; the run goes on); 2 on a
usage error, a run started outside the repository root, a batch or notes file that cannot be read, and
whenever the system cannot be assembled or checked (the frozen record, the family, no llama-server, a busy
port, a server that never became ready, a GPU in use or an nvidia-smi that cannot tell, an identity that
disagrees); 3 when another run holds the lock; 130 after Ctrl+C, once the server is stopped and the lock
released.
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from research_agent import arxiv
from research_agent.agent.context import family_token_counter
from research_agent.eval.resources import SMI, GpuError, GpuSample, gpu_sample
from research_agent.eval.systems import Built, SystemUnavailable, build_system
from research_agent.live.frozen import (
    SLOTS,
    FrozenMismatch,
    FrozenSystem,
    code_hashes,
    identity_problems,
    load_frozen,
)
from research_agent.live.session import (
    LOCK_FILE,
    MAX_FOLLOW_UPS,
    SERVER_LOG,
    InstanceLock,
    LiveTurn,
    LockBusy,
    ProgressClient,
    SessionFiles,
    new_run_folder,
    notes_written,
    pack_follow_up,
    run_live_turn,
)
from research_agent.live.world import LiveWorld, arxiv_live_tools, live_world
from research_agent.llm.server import LocalServer, NotReady, PortBusy, llama_server
from research_agent.net.fetch import utc_now
from research_agent.notes import NoteStore, import_entries
from research_agent.train.manifest import REPO, git_commit, git_dirty

NOTES = Path("notes")
RUNS = Path("runs/live")
STATE = Path("runs/live")  # the lock and the rate-limit state, whatever --runs is
IMPORTED = "imported"  # the tag of imported notes
PROBE_SEARCH = {"query": "speculative decoding", "max_results": 3, "sort_by": "submittedDate"}
PROBE_FILE = "probe.json"
ASK_PROMPT, ANSWER_PROMPT = "ask> ", "answer> "
NEW, QUIT = "/new", "/quit"
CONSOLE_HELP = (f"type a request at {ASK_PROMPT.strip()} and the answer to a question at "
                f"{ANSWER_PROMPT.strip()}; {NEW} TEXT asks TEXT instead, {QUIT} ends the run")
QUIT_WORDS = ("exit", "quit")  # typed bare, they are taken for a slip, never for a request
INTERRUPTED = 130
EPILOG = (f"Run it from the repository root. Answers are printed; each run's files go to a new folder under "
          f"--runs (default {RUNS.as_posix()}), notes to --notes (default {NOTES.as_posix()}). Exit 0 when "
          "the turns ran, 1 when a turn raised, 2 on a usage error or when the system cannot be assembled or "
          f"checked, 3 when another run holds the lock, {INTERRUPTED} after Ctrl+C.")

Out = Callable[[str], None]


class SetupFailed(Exception):
    """The system cannot be assembled or checked here (exit 2)."""


def local_date() -> str:
    """Today's date on this machine's clock: a turn's today."""
    return datetime.datetime.now(datetime.UTC).astimezone().date().isoformat()


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Ask M7's final system, unchanged, on the real arXiv.",
                                 epilog=EPILOG)
    ap.add_argument("request", nargs="?", default=None, help="one request")
    ap.add_argument("--clarify-answer", action="append", default=None, metavar="TEXT",
                    help=f"the answer to a clarification the request ends in (at most {MAX_FOLLOW_UPS})")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--interactive", action="store_true", help="ask> for requests, answer> after a "
                                                                  "clarification; /new, /quit")
    modes.add_argument("--batch", type=Path, default=None, metavar="FILE.jsonl",
                       help='one {"id", "request", "clarify_answer"?} per line, in order')
    modes.add_argument("--check", action="store_true", help="assemble and check the system, then stop")
    modes.add_argument("--probe", nargs="+", default=None, metavar="ID",
                       help="get_paper for each id and one search on the live arXiv, without the GPU")
    modes.add_argument("--import-notes", type=Path, default=None, metavar="FILE",
                       help="import the owner's notes file into the library (counts only)")
    ap.add_argument("--batch-name", default=None, metavar="NAME",
                    help='--batch: only the lines whose "batch" is NAME')
    servers = ap.add_mutually_exclusive_group()
    servers.add_argument("--llama-server", default=None, metavar="PATH",
                         help="the llama-server binary (default: $LLAMA_SERVER)")
    servers.add_argument("--server", default=None, metavar="URL",
                         help="use the llama-server running at URL instead of starting one")
    ap.add_argument("--port", type=int, default=None, help="the port of the server started here (default: "
                                                          "the configured one)")
    ap.add_argument("--notes", type=Path, default=NOTES,
                    help=f"the notes library (default {NOTES.as_posix()})")
    ap.add_argument("--runs", type=Path, default=RUNS,
                    help=f"where the run folders go (default {RUNS.as_posix()})")
    ap.add_argument("--state", type=Path, default=STATE, help=argparse.SUPPRESS)
    return ap


def mode_of(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """The run's mode; a usage error (exit 2) for options that do not go together."""
    flagged = [name for name, on in (("interactive", args.interactive), ("batch", args.batch is not None),
                                     ("check", args.check), ("probe", args.probe is not None),
                                     ("import", args.import_notes is not None)) if on]
    if args.request is not None and flagged:
        parser.error("a request runs on its own: drop it, or drop --" + flagged[0].replace("import",
                                                                                         "import-notes"))
    if args.request is None and not flagged:
        parser.error("give a request, or one of --interactive, --batch, --check, --probe, --import-notes")
    if args.request is not None and not args.request.strip():
        parser.error("the request is empty")
    if args.clarify_answer is not None and args.request is None:
        parser.error("--clarify-answer continues a request given on the command line")
    if args.clarify_answer is not None and len(args.clarify_answer) > MAX_FOLLOW_UPS:
        parser.error(f"--clarify-answer at most {MAX_FOLLOW_UPS} times: a request is continued at most "
                     f"{MAX_FOLLOW_UPS} times")
    if args.batch_name is not None and args.batch is None:
        parser.error("--batch-name chooses lines of --batch")
    if args.port is not None and args.server is not None:
        parser.error("--port is the port of a server started here; --server names a running one")
    return "request" if args.request is not None else flagged[0]


def load_batch(path: Path, name: str | None) -> list[dict]:
    """The requests of a batch file, in order (module docstring: --batch), as {"id", "request", "answers"};
    ValueError naming the line for a malformed one, OSError for a file that cannot be read."""
    items = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path.as_posix()} line {number} is not JSON: {exc}") from None
        if not (isinstance(item, dict) and isinstance(item.get("request"), str) and item["request"].strip()):
            raise ValueError(f"{path.as_posix()} line {number} has no request")
        answers = item.get("clarify_answer")
        answers = [] if answers is None else [answers] if isinstance(answers, str) else answers
        if not (isinstance(answers, list) and len(answers) <= MAX_FOLLOW_UPS
                and all(isinstance(answer, str) for answer in answers)):
            raise ValueError(f"{path.as_posix()} line {number}: clarify_answer must be a text or at most "
                             f"{MAX_FOLLOW_UPS} texts")
        if name is None or item.get("batch") == name:
            item_id = item.get("id")
            items.append({"id": f"line-{number}" if item_id is None else str(item_id),
                          "request": item["request"], "answers": answers})
    if not items:
        chosen = "" if name is None else f" of batch {name!r}"
        raise ValueError(f"{path.as_posix()} holds no request{chosen}")
    return items


# ---- assembling the system --------------------------------------------------------------------------------


@dataclass
class Assembly:
    """The checked system a run's turns use (module docstring: The system), what session.json records of it
    (record), and what closes when the run ends (the client, then the server it started)."""

    built: Built
    props: dict | None
    problems: list[str]
    server: dict  # {"url", "started_here", "command"}
    frozen: dict  # the frozen record's and the session configuration's paths and SHA-256
    closers: list[Callable[[], None]] = field(default_factory=list)

    def record(self) -> dict:
        return {"info": self.built.info, "props": self.props, "identity_problems": self.problems,
                "server": self.server, "frozen": self.frozen}

    def close(self) -> None:
        closers, self.closers = self.closers, []
        try:
            self.built.close()
        finally:
            for close in reversed(closers):
                close()


def gpu_now() -> GpuSample | None:
    """GPU 0 now (eval.resources.gpu_sample, GpuError when nvidia-smi fails); None when there is no nvidia-smi
    on PATH to ask."""
    return None if shutil.which(SMI) is None else gpu_sample()


def require_idle_gpu(limit: float, sample: Callable[[], GpuSample | None], out: Out) -> None:
    """SetupFailed when the GPU is in use beyond limit MiB or nvidia-smi fails to tell; a warning when there
    is no nvidia-smi to ask (module docstring: The server)."""
    try:
        now = sample()
    except GpuError as exc:
        raise SetupFailed(f"cannot tell whether the GPU is idle: {exc}; try again, or use a running server "
                          "with --server") from None
    if now is None:
        out(f"warning: no {SMI} on PATH, so whether the GPU is idle cannot be told; starting the server "
            "anyway")
        return
    if now.memory_mib > limit:
        raise SetupFailed(f"the GPU is not idle: {now.memory_mib:.0f} MiB in use, at most {limit:g} count as "
                          "idle; stop what uses it (a training run, a server), or use a running server with "
                          "--server")
    out(f"GPU idle: {now.memory_mib:.0f} MiB in use")


def assemble(args: argparse.Namespace, folder: Path, out: Out, *,
             load: Callable[[], FrozenSystem] = load_frozen,
             client: Callable[..., ProgressClient] = ProgressClient,
             start_server: Callable[..., LocalServer] = LocalServer,
             gpu: Callable[[], GpuSample | None] = gpu_now) -> Assembly:
    """The frozen system built over its server and checked (module docstring: The system, The server);
    SetupFailed when it cannot be. load, client, start_server and gpu stand in for the real ones in tests."""
    try:
        frozen = load()
    except FrozenMismatch as exc:
        raise SetupFailed("the frozen system does not hold: " + "; ".join(exc.problems)) from None
    out(f"frozen record {frozen.frozen_path} holds (SHA-256 {frozen.frozen_sha256})")
    closers: list[Callable[[], None]] = []
    try:
        if args.server is not None:
            url = args.server.rstrip("/")
            local = client(url, out=out)
            closers.append(local.close)
            if not local.health():
                raise SetupFailed(f"no llama-server answers at {url}")
            props = local.props()
            server = {"url": url, "started_here": False, "command": None}
        else:
            try:
                binary = llama_server(args.llama_server)
            except LookupError as exc:
                raise SetupFailed(str(exc)) from None
            require_idle_gpu(frozen.server.idle_max_mib, gpu, out)
            port = frozen.server.port if args.port is None else args.port
            log = folder / SERVER_LOG
            own = start_server(binary, model=frozen.base_path, adapter=frozen.adapter_path, port=port,
                               ctx_size=frozen.server.ctx_size, timeout=frozen.server.timeout_s,
                               extra=("-np", str(SLOTS)), log=log)
            out(f"starting llama-server on port {port}; its output goes to {log.as_posix()}")
            try:
                own.__enter__()
            except PortBusy as exc:
                raise SetupFailed(f"{exc}; if it is the frozen system's server use it with --server "
                                  f"{own.url}, else start this one on another --port") from None
            except NotReady as exc:
                raise SetupFailed(f"{exc} (its output: {log.as_posix()})") from None
            except OSError as exc:  # no such binary, or one that cannot be run
                raise SetupFailed(f"llama-server cannot be started: {exc}") from None
            closers.append(own.close)
            url, props = own.url, own.props
            local = client(url, out=out)
            closers.append(local.close)
            server = {"url": url, "started_here": True, "command": ["llama-server", *own.cmd[1:]]}
        try:
            built = build_system("ft", server=url, tokenizer=frozen.tokenizer, family=frozen.family,
                                 generation=frozen.settings,
                                 count_tokens=family_token_counter(frozen.family, frozen.settings.mode),
                                 rules=frozen.rules, vote_mode=frozen.vote_mode, local_client=local)
        except SystemUnavailable as exc:
            raise SetupFailed(str(exc)) from None
        problems = identity_problems(built.info, frozen, props)
        recorded = {"path": frozen.frozen_path, "sha256": frozen.frozen_sha256, "config": frozen.config_path,
                    "config_sha256": frozen.config_sha256}
        return Assembly(built, props, problems, server, recorded, closers)
    except BaseException:
        for close in reversed(closers):
            close()
        raise


def show_identity(assembly: Assembly, out: Out) -> None:
    """The identity --check prints."""
    info, props = assembly.built.info, assembly.props or {}
    generation, rules = info.get("generation") or {}, info.get("rules") or {}
    out(f"system: {info.get('model')}")
    out(f"family {(info.get('family') or {}).get('name')}; {generation.get('mode')} mode, reasoning budget "
        f"{generation.get('reasoning_budget')}, {generation.get('votes')} votes ({info.get('vote_mode')}), "
        f"rules {rules.get('version')}")
    for adapter in info.get("adapters") or []:
        out(f"adapter: {adapter.get('path')} (SHA-256 {adapter.get('sha256')})")
    out(f"server: {assembly.server.get('url')}, build {props.get('build_info')}, {props.get('total_slots')} "
        f"slot(s), model {props.get('model_path')}")
    out(f"frozen record: {assembly.frozen.get('path')} (SHA-256 {assembly.frozen.get('sha256')})")
    if assembly.problems:
        out(f"identity: {len(assembly.problems)} problem(s)")
        for problem in assembly.problems:
            out(f"  {problem}")
    else:
        out("identity: OK")


# ---- turns ------------------------------------------------------------------------------------------------


class Runner:
    """The turns of a run over one assembly, world and notes library (module docstring: Modes)."""

    def __init__(self, assembly: Assembly, world: LiveWorld, store: NoteStore, files: SessionFiles, *,
                 today: Callable[[], str], out: Out) -> None:
        self.assembly, self.world, self.store, self.files = assembly, world, store, files
        self.today, self.out = today, out
        self.failed = False

    def turn(self, text: str, *, item_id: str | None = None, follow_up: int = 0) -> LiveTurn:
        day = self.today()
        label = "" if item_id is None else f"[{item_id}] "
        self.out(f"{label}turn {self.files.turns + 1} on {day}"
                 + (f", follow-up {follow_up}" if follow_up else ""))
        result = run_live_turn(text, built=self.assembly.built, tools=self.world.tools(self.store, day),
                               today=day, out=self.out, waited=lambda: self.world.limits.slept)
        self.files.write_turn(result, item_id=item_id, follow_up=follow_up, papers=self.world.memo.get)
        self.failed = self.failed or result.error is not None
        self.show(result, label)
        return result

    def request(self, text: str, answers: list[str], item_id: str | None = None) -> LiveTurn:
        """A request and its follow-ups with the answers given, in order (pack_follow_up); an answer left over
        is said to be unused (module docstring: Modes)."""
        result = self.turn(text, item_id=item_id)
        label = "" if item_id is None else f"[{item_id}] "
        for number, answer in enumerate(answers[:MAX_FOLLOW_UPS], 1):
            unused = len(answers) - number + 1
            if not result.clarified:
                ended = "with an error" if result.status is None else f"as {result.status}"
                self.out(f"{label}{unused} clarify answer(s) not used: the turn ended {ended}, not in a "
                         "clarification")
                break
            packed = pack_follow_up(result.request, answer)
            if packed is None:
                self.out(f"{label}{unused} clarify answer(s) not used: the answer is empty")
                break
            result = self.turn(packed, item_id=item_id, follow_up=number)
        return result

    def show(self, result: LiveTurn, label: str) -> None:
        if result.transcript is None:
            self.out(f"{label}the turn raised {result.error} after {result.seconds:.1f} s")
            return
        transcript = result.transcript
        self.out(f"{label}{transcript.status} after {len(transcript.steps)} step(s), {result.seconds:.1f} s")
        if transcript.final_text:
            self.out(("question: " if result.clarified else "answer:\n") + transcript.final_text)
        for note in notes_written(transcript):
            self.out(f"note written: {note['note_id']} ({(self.store.root / note['path']).as_posix()})")


def interactive(runner: Runner, ask: Callable[[str], str], out: Out) -> None:
    """ask> and answer> until /quit or the end of input (module docstring: --interactive)."""
    pending: LiveTurn | None = None  # the turn whose clarification is being answered
    follow_ups = 0
    out(CONSOLE_HELP)
    while True:
        try:
            line = ask(ANSWER_PROMPT if pending is not None else ASK_PROMPT)
        except EOFError:
            return
        except UnicodeDecodeError:
            out("the input is not UTF-8 text: the run ends")
            return
        text = line.strip()
        if text == QUIT:
            return
        command = text.split(maxsplit=1)[0] if text else ""
        if (command.startswith("/") and command != NEW) or text.lower() in QUIT_WORDS:
            out(f"not a command here: {CONSOLE_HELP}")
            continue
        if text == NEW or text.startswith(NEW + " "):
            pending, follow_ups, text = None, 0, text[len(NEW):].strip()
        elif pending is not None:
            if not text:
                out("no answer: the request is left")
                pending, follow_ups = None, 0
                continue
            follow_ups += 1
            result = runner.turn(pack_follow_up(pending.request, text), follow_up=follow_ups)
            pending = result if result.clarified and follow_ups < MAX_FOLLOW_UPS else None
            if result.clarified and pending is None:
                out(f"{MAX_FOLLOW_UPS} follow-ups already: please restate the whole request")
            if pending is None:
                follow_ups = 0
            continue
        if not text:
            continue
        result = runner.turn(text)
        pending, follow_ups = (result if result.clarified else None), 0


# ---- the modes without the model --------------------------------------------------------------------------


def _ids(papers: list[dict] | None) -> list[str] | None:
    return None if papers is None else [paper["arxiv_id"] for paper in papers]


def parsed(last: dict | None) -> dict:
    """The HTTP status of a probe's request and what parse_atom makes of its body, by default and leniently
    (None for a body it cannot read, or when no usable answer came back)."""
    text = (last or {}).get("text")
    return {"http_status": (last or {}).get("status"), "answered": text is not None,
            "default_parse": None if text is None else _ids(arxiv.parse_atom(text)),
            "lenient_parse": None if text is None else _ids(arxiv.parse_atom(text, lenient=True))}


def probe(ids: list[str], files: SessionFiles, state: Path, *, today: Callable[[], str], transport: object,
          clock: Callable[[], float], wall: Callable[[], float], sleep: Callable[[float], None],
          out: Out) -> int:
    """--probe (module docstring); 0 once every probe has an outcome. Every request is sent once."""
    started, day = utc_now(), today()
    world = live_world(files.folder, state, transport=transport, clock=clock, wall=wall, sleep=sleep,
                       retry_delays=())
    tools = arxiv_live_tools(world.fetcher, world.memo, day)
    records = []
    try:
        for arxiv_id in ids:
            world.fetcher.last = None
            outcome = tools["get_paper"]({"arxiv_id": arxiv_id})
            content = outcome.content if isinstance(outcome.content, dict) else {}
            record = {"id": arxiv_id, "error": outcome.error, "found": content.get("arxiv_id"),
                      "title": content.get("title"), **parsed(world.fetcher.last)}
            records.append(record)
            out(f"probe {arxiv_id}: get_paper " + (record["error"] or f"found {record['found']}")
                + f"; HTTP {record['http_status']}; default parse {record['default_parse']}, lenient parse "
                  f"{record['lenient_parse']}")
        world.fetcher.last = None
        outcome = tools["search_arxiv"](dict(PROBE_SEARCH))
        hits = [hit["arxiv_id"] for hit in outcome.content] if isinstance(outcome.content, list) else []
        search = {**PROBE_SEARCH, "error": outcome.error, "hits": hits, **parsed(world.fetcher.last)}
        out(f"probe search {PROBE_SEARCH['query']!r}: " + (search["error"] or f"{len(hits)} hit(s) {hits}")
            + f"; HTTP {search['http_status']}")
    finally:
        world.close(warn(out))
    result = {"today": day, "ids": records, "search": search}
    (files.folder / PROBE_FILE).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                           encoding="utf-8", newline="\n")
    files.write_session({"kind": "probe", "started": started, "ended": utc_now(), "code": code_hashes(),
                         "git_commit": git_commit(REPO), "git_dirty": git_dirty(REPO)})
    out(f"probe outcomes: {(files.folder / PROBE_FILE).as_posix()}")
    return 0


def import_notes(path: Path, notes: Path, out: Out) -> int:
    """--import-notes (module docstring): counts only; 2 for a file that cannot be read or holds an id that
    is not canonical."""
    try:
        entries = import_entries(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        print(f"FAIL: {path.as_posix()} cannot be read: {exc}", file=sys.stderr)
        return 2
    try:
        written = NoteStore(notes).load_entries(entries, tags=[IMPORTED], skip_existing=True)
    except ValueError as exc:
        print(f"FAIL: nothing imported: {exc}", file=sys.stderr)
        return 2
    with_id = sum(1 for entry in entries if entry.get("arxiv_id"))
    out(f"{len(entries)} entries: {len(written)} imported, {with_id - len(written)} already in the library, "
        f"{len(entries) - with_id} without an arXiv id skipped")
    return 0


# ---- main -------------------------------------------------------------------------------------------------


def _print(line: str) -> None:
    print(line, flush=True)


def warn(out: Out) -> Out:
    """out with each line marked as a warning."""
    return lambda line: out(f"warning: {line}")


def _reconfigure(stream: object, *, errors: str) -> None:
    """A terminal keeps its encoding, a pipe or a file gets UTF-8 (module docstring: Console text)."""
    if not hasattr(stream, "reconfigure"):
        return
    try:
        terminal = stream.isatty()
    except (OSError, ValueError):  # a closed or detached stream
        terminal = False
    if terminal:
        stream.reconfigure(errors=errors)
    else:
        stream.reconfigure(encoding="utf-8", errors=errors)


def configure_streams(*, console_input: bool) -> None:
    """The console's streams (module docstring: Console text); console_input when --interactive reads
    sys.stdin, whose undecodable input then raises rather than turn into other characters."""
    for stream in (sys.stdout, sys.stderr):
        _reconfigure(stream, errors="backslashreplace")
    if console_input:
        _reconfigure(sys.stdin, errors="strict")


def at_repository_root() -> bool:
    """Whether the current folder is the repository root, where a run belongs (module docstring)."""
    return Path.cwd().resolve() == Path(REPO).resolve()


def run(mode: str, args: argparse.Namespace, items: list[dict] | None, *, today: Callable[[], str],
        build: Callable[..., Assembly], transport: object, clock: Callable[[], float],
        wall: Callable[[], float], sleep: Callable[[float], None], ask: Callable[[str], str],
        out: Out) -> int:
    """The run of a mode, the lock held (module docstring)."""
    if mode == "import":
        return import_notes(args.import_notes, args.notes, out)
    files = SessionFiles(new_run_folder(args.runs))
    out(f"run folder: {files.folder.as_posix()}")
    if mode == "probe":
        return probe(args.probe, files, args.state, today=today, transport=transport, clock=clock, wall=wall,
                     sleep=sleep, out=out)
    session = {"kind": mode, "started": utc_now(), "notes": args.notes.as_posix(), "code": code_hashes(),
               "git_commit": git_commit(REPO), "git_dirty": git_dirty(REPO)}
    try:
        assembly = build(args, files.folder, out)
    except SetupFailed as exc:
        files.write_session({**session, "setup_error": str(exc), "ended": utc_now()})
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    try:
        session.update(assembly.record())
        files.write_session(session)
        if mode == "check" or assembly.problems:
            show_identity(assembly, out)
            return 2 if assembly.problems else 0
        try:
            store = NoteStore(args.notes)
        except (OSError, RuntimeError) as exc:  # RuntimeError: a sqlite3 without FTS5
            print(f"FAIL: the notes library {args.notes.as_posix()} cannot be opened: {exc}", file=sys.stderr)
            return 2
        world = live_world(files.folder, args.state, transport=transport, clock=clock, wall=wall, sleep=sleep)
        try:
            runner = Runner(assembly, world, store, files, today=today, out=out)
            if mode == "request":
                answers = args.clarify_answer or []
                result = runner.request(args.request, answers)
                if result.clarified and len(answers) >= MAX_FOLLOW_UPS:
                    out(f"{MAX_FOLLOW_UPS} follow-ups already: please restate the whole request")
                elif result.clarified:
                    out("the request ends in a clarification: answer it with --clarify-answer, or use "
                        "--interactive")
            elif mode == "batch":
                for item in items:
                    runner.request(item["request"], item["answers"], item_id=item["id"])
            else:
                interactive(runner, ask, out)
        finally:
            world.close(warn(out))
        return 1 if runner.failed else 0
    finally:
        try:
            assembly.close()
        finally:
            files.write_session({**session, "turns": files.turns, "ended": utc_now()})
            out(f"{files.turns} turn(s); run folder {files.folder.as_posix()}")


def main(argv: list[str] | None = None, *, today: Callable[[], str] | None = None,
         build: Callable[..., Assembly] | None = None, transport: object | None = None,
         clock: Callable[[], float] | None = None, wall: Callable[[], float] | None = None,
         sleep: Callable[[float], None] | None = None, ask: Callable[[str], str] = input) -> int:
    """today, build, transport, clock, wall, sleep and ask stand in for the date, the assembly, the network,
    the clocks, sleeping and the console in tests; wall defaults to clock when only clock is given."""
    parser = build_parser()
    args = parser.parse_args(argv)
    mode = mode_of(args, parser)
    configure_streams(console_input=mode == "interactive" and ask is input)
    if not at_repository_root():
        print(f"FAIL: run it from the repository root ({Path(REPO).as_posix()}): the frozen record's paths, "
              "the notes library, the run folders, the lock and the rate-limit state are all there",
              file=sys.stderr)
        return 2
    items = None
    if mode == "batch":
        try:
            items = load_batch(args.batch, args.batch_name)
        except (OSError, ValueError) as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2
    lock = InstanceLock(args.state / LOCK_FILE)
    try:
        lock.acquire()
    except LockBusy as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 3
    try:
        return run(mode, args, items, today=today or local_date, build=build or assemble, transport=transport,
                   clock=clock or time.monotonic, wall=wall or clock or time.time, sleep=sleep or time.sleep,
                   ask=ask, out=_print)
    except KeyboardInterrupt:
        print("interrupted: the server is stopped and the lock released", file=sys.stderr)
        return INTERRUPTED
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())

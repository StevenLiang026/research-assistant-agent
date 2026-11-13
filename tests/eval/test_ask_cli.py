"""scripts/ask.py offline: every mode and exit code with the assembly injected (a scripted model) or assembled
for real over the fake final system (tests/support/live_fakes.py) on a FakeLlama server, arXiv behind
httpx.MockTransport, a fake clock and a temporary runs folder. No GPU, no server, no network: every run goes
through main, whose arXiv, unless a test gives its own, is a MockTransport that answers no request and records
it, and a test with a recorded request fails. Every run starts at the repository root, as ask.py requires.
--web runs main in the test's main thread while the page's user, a browser over http.client to 127.0.0.1
(tests/support/web_fakes.py), works in a helper thread that always ends the run (Browser)."""

from __future__ import annotations

import importlib.util
import io
import json
import signal
import socket
import sys
import threading
from pathlib import Path
from typing import Self

import httpx
import pytest
from support.fakes import ScriptedModel, call, turn
from support.live_fakes import FakeClock, final_server, server_props, text_turns, write_final
from support.web_fakes import TIMEOUT, Page

from research_agent.agent.prompts import system_prompt
from research_agent.agent.rules import RULES_VERSION
from research_agent.eval.resources import GpuError, GpuSample
from research_agent.eval.systems import Built
from research_agent.live import web
from research_agent.live.frozen import EVALUATED_SERVER, FrozenMismatch, code_hashes, load_frozen
from research_agent.live.session import InstanceLock, ProgressClient
from research_agent.llm.server import PortBusy
from research_agent.tools.adapters import paper_url

REPO = Path(__file__).resolve().parents[2]
TODAY = "2026-11-10"
A, B = "2611.00001", "2611.00002"
ENTRY = ("<entry><id>http://arxiv.org/abs/{id}v1</id><published>2026-11-02</published>"
         "<title>Paper {id}</title><summary>About speculative decoding. More on drafts.</summary>"
         '<author><name>A. Author</name></author><category term="cs.CL"/></entry>')
OLD_STYLE = ("<entry><id>http://arxiv.org/abs/cs/0112017v1</id><published>2001-12-14</published>"
             "<title>An Old Paper</title><summary>Before the new identifiers.</summary></entry>")
SEARCH = "https://export.arxiv.org/api/query?search_query="


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ask = load_script("ask")
WENT_OUT: list[str] = []  # the requests a run sent to the stand-in for the network (main): none may


def offline(request: httpx.Request) -> httpx.Response:
    WENT_OUT.append(str(request.url))
    raise AssertionError(f"a request went for the network: {request.url}")


@pytest.fixture(autouse=True)
def at_the_root_and_offline(monkeypatch):
    """Every test runs from the repository root, and no request it sends goes unnoticed."""
    monkeypatch.chdir(REPO)
    WENT_OUT.clear()
    yield
    assert WENT_OUT == [], f"requests went for the network: {WENT_OUT}"


def main(argv: list[str], **options) -> int:
    """ask.main, its arXiv the stand-in that answers nothing (offline) unless the test gives one."""
    options.setdefault("transport", httpx.MockTransport(offline))
    return ask.main(argv, **options)


def feed(*entries: str) -> str:
    return ('<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">'
            "<title>ArXiv Query</title>" + "".join(entries) + "</feed>")


def arxiv_transport(pages: dict[str, str], seen: list[str] | None = None) -> httpx.MockTransport:
    """arXiv answering each URL (or, for SEARCH, any search) with its feed; every URL asked kept in seen."""

    def handle(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if seen is not None:
            seen.append(url)
        key = SEARCH if url.startswith(SEARCH) else url
        if key not in pages:
            raise AssertionError(f"unexpected request {url}")
        return httpx.Response(200, text=pages[key], headers={"content-type": "application/atom+xml"})

    return httpx.MockTransport(handle)


def folders(tmp_path) -> list[str]:
    return ["--runs", str(tmp_path / "runs"), "--notes", str(tmp_path / "notes"),
            "--state", str(tmp_path / "state")]


def run_folder(tmp_path) -> Path:
    (folder,) = [path for path in (tmp_path / "runs").iterdir() if path.is_dir()]
    return folder


def index(tmp_path) -> list[dict]:
    text = (tmp_path / "runs" / "index.jsonl").read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def scripted(*turns, problems=(), closed: list | None = None):
    """An injected assembly over a scripted model (main's build)."""

    def build(args, folder, out):
        info = {"model": "scripted", "generation": {"mode": "think", "votes": 5}}
        closers = [] if closed is None else [lambda: closed.append(True)]
        return ask.Assembly(Built(ScriptedModel(list(turns)), None, info), {"total_slots": 1}, list(problems),
                            {"url": "http://127.0.0.1:8080", "started_here": False, "command": None},
                            {"path": "results/final-m7.json", "sha256": "0" * 64}, closers)

    return build


def no_build(args, folder, out):
    raise AssertionError("this mode assembles no system")


# ---- one request, a batch, the console -------------------------------------------------------------------


def test_one_request_runs_one_turn_and_writes_the_run_files(tmp_path, capsys):
    closed: list = []
    code = main(["你好,帮我总结一下本周的阅读。", *folders(tmp_path)], today=lambda: TODAY,
                build=scripted(turn(content="本周读了两篇。"), closed=closed))
    out = capsys.readouterr().out
    assert code == 0 and closed == [True] and "answer:\n本周读了两篇。" in out
    folder = run_folder(tmp_path)
    session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    assert session["kind"] == "request" and session["identity_problems"] == [] and session["turns"] == 1
    assert session["code"] == code_hashes() and session["info"]["model"] == "scripted" and "ended" in session
    transcript = json.loads((folder / "turn-001.json").read_text(encoding="utf-8"))
    assert transcript["status"] == "answered" and transcript["today"] == TODAY
    assert (folder / "turn-001.md").read_text(encoding="utf-8").startswith("# Turn 001\n")
    assert [(row["turn"], row["status"], row["id"]) for row in index(tmp_path)] == [(1, "answered", None)]
    assert not (tmp_path / "state" / "ask.lock.json").exists()  # released


def test_a_request_ending_in_a_clarification_is_continued_by_the_answers_given(tmp_path, capsys):
    question = call("ask_clarification", "q1", question="哪个主题?", missing=["topic"])
    code = main(["帮我找几篇论文。", "--clarify-answer", "speculative decoding", *folders(tmp_path)],
                today=lambda: TODAY, build=scripted(turn(question), turn(content="好的。")))
    assert code == 0 and "question: 哪个主题?" in capsys.readouterr().out
    rows = index(tmp_path)
    assert [(row["status"], row["follow_up"]) for row in rows] == [("clarification", 0), ("answered", 1)]
    assert rows[1]["request"] == "帮我找几篇论文。补充:speculative decoding"


def test_an_answer_the_request_did_not_ask_for_is_said_to_be_unused(tmp_path, capsys):
    code = main(["Find papers on RAG.", "--clarify-answer", "since 2025", *folders(tmp_path)],
                today=lambda: TODAY, build=scripted(turn(content="Here are three.")))
    out = capsys.readouterr().out
    assert code == 0 and len(index(tmp_path)) == 1
    assert "1 clarify answer(s) not used: the turn ended as answered, not in a clarification" in out
    question = call("ask_clarification", "q1", question="Which topic?", missing=["topic"])
    assert main(["Find papers.", "--clarify-answer", " ", *folders(tmp_path)], today=lambda: TODAY,
                build=scripted(turn(question))) == 0
    assert "1 clarify answer(s) not used: the answer is empty" in capsys.readouterr().out


def test_a_clarification_without_an_answer_prints_the_question_and_how_to_answer(tmp_path, capsys):
    question = call("ask_clarification", "q1", question="Which topic?", missing=["topic"])
    code = main(["Find papers.", *folders(tmp_path)], today=lambda: TODAY, build=scripted(turn(question)))
    assert code == 0
    out = capsys.readouterr().out
    assert "question: Which topic?" in out and "--clarify-answer" in out and len(index(tmp_path)) == 1


def test_a_batch_runs_its_named_lines_in_order_over_the_live_tools_with_follow_ups(tmp_path, capsys):
    batch = tmp_path / "smoke.jsonl"
    batch.write_text("\n".join(json.dumps(line, ensure_ascii=False) for line in [
        {"id": "S01", "batch": "A", "request": "not this one"},
        {"id": "S08", "batch": "B", "request": "列出我最近写的笔记。"},
        {"id": "S09", "batch": "B", "request": "帮我找几篇论文。", "clarify_answer": "speculative decoding"},
    ]) + "\n", encoding="utf-8")
    model = scripted(turn(call("list_notes", "l1")), turn(content="还没有笔记。"),
                     turn(call("ask_clarification", "q1", question="哪个主题?", missing=["topic"])),
                     turn(call("search_arxiv", "s1", query="speculative decoding")),
                     turn(content=f"推荐 {A}。"))
    seen: list[str] = []
    clock = FakeClock()
    code = main(["--batch", str(batch), "--batch-name", "B", *folders(tmp_path)], today=lambda: TODAY,
                build=model, transport=arxiv_transport({SEARCH: feed(ENTRY.format(id=A))}, seen),
                clock=clock, sleep=clock.sleep)
    assert code == 0
    rows = index(tmp_path)
    assert [(row["id"], row["status"], row["follow_up"]) for row in rows] == [
        ("S08", "answered", 0), ("S09", "clarification", 0), ("S09", "answered", 1)]
    assert rows[2]["cited"] == [A] and rows[2]["cited_unseen"] == []
    follow_up = json.loads((run_folder(tmp_path) / "turn-003.json").read_text(encoding="utf-8"))
    packed = "帮我找几篇论文。补充:speculative decoding"
    assert follow_up["messages"][:2] == [{"role": "system", "content": system_prompt(TODAY)},
                                         {"role": "user", "content": packed}]
    assert len(seen) == 1 and seen[0].startswith(SEARCH)
    log = [json.loads(line) for line in (run_folder(tmp_path) / "requests.jsonl").read_text(encoding="utf-8")
           .splitlines()]
    assert [line["event"] for line in log] == ["request", "fetch"] and log[0]["host"] == "export.arxiv.org"
    out = capsys.readouterr().out
    assert "[S08] turn 1 on 2026-11-10" in out and "  tool list_notes: 0 note(s)" in out
    assert "[S09] turn 3 on 2026-11-10, follow-up 1" in out


def test_the_console_asks_answers_and_restarts(tmp_path, capsys):
    typed = iter(["帮我找几篇论文。", "speculative decoding", "/new Find papers on RAG.", "", "/quit",
                  "never"])
    question = call("ask_clarification", "q1", question="哪个主题?", missing=["topic"])
    other = call("ask_clarification", "q2", question="Since when?", missing=["date_range"])
    prompts: list[str] = []

    def console(prompt: str) -> str:
        prompts.append(prompt)
        return next(typed)

    code = main(["--interactive", *folders(tmp_path)], today=lambda: TODAY,
                build=scripted(turn(question), turn(content="好的。"), turn(other)), ask=console)
    assert code == 0
    assert prompts == ["ask> ", "answer> ", "ask> ", "answer> ", "ask> "]
    rows = index(tmp_path)
    assert [(row["request"], row["follow_up"]) for row in rows] == [
        ("帮我找几篇论文。", 0), ("帮我找几篇论文。补充:speculative decoding", 1), ("Find papers on RAG.", 0)]
    assert "no answer: the request is left" in capsys.readouterr().out


def test_the_console_ends_at_the_end_of_input(tmp_path):
    def closed_console(prompt: str) -> str:
        raise EOFError

    assert main(["--interactive", *folders(tmp_path)], today=lambda: TODAY, build=scripted(),
                ask=closed_console) == 0
    assert not (tmp_path / "runs" / "index.jsonl").exists()


def test_the_console_says_how_it_is_used_and_sends_no_slip_to_the_model(tmp_path, capsys):
    typed = iter(["/help", "exit", "Quit", "帮我找几篇论文。", "/exit", "speculative decoding", "/quit"])
    question = call("ask_clarification", "q1", question="哪个主题?", missing=["topic"])
    code = main(["--interactive", *folders(tmp_path)], today=lambda: TODAY,
                build=scripted(turn(question), turn(content="好的。")), ask=lambda prompt: next(typed))
    out = capsys.readouterr().out
    assert code == 0 and out.startswith("run folder: ") and f"\n{ask.CONSOLE_HELP}\n" in out
    assert out.count(f"not a command here: {ask.CONSOLE_HELP}") == 4  # /help, exit, Quit and /exit
    rows = index(tmp_path)  # the slip at answer> kept the question open for the answer after it
    assert [(row["request"], row["follow_up"]) for row in rows] == [
        ("帮我找几篇论文。", 0), ("帮我找几篇论文。补充:speculative decoding", 1)]


def test_console_input_that_is_not_utf8_ends_the_input(tmp_path, capsys):
    def garbled(prompt: str) -> str:
        raise UnicodeDecodeError("utf-8", b"\xb0\xef", 0, 1, "invalid start byte")

    assert main(["--interactive", *folders(tmp_path)], build=scripted(), ask=garbled) == 0
    assert "the input is not UTF-8 text: the run ends" in capsys.readouterr().out
    assert not (tmp_path / "runs" / "index.jsonl").exists()


class Terminal(io.TextIOWrapper):
    """A text stream that is a terminal."""

    def isatty(self) -> bool:
        return True


def test_a_pipe_gets_utf8_and_a_terminal_keeps_its_encoding(monkeypatch):
    piped = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    terminal = Terminal(io.BytesIO(), encoding="cp1252")
    typed = io.TextIOWrapper(io.BytesIO("帮我找几篇论文。\n".encode()), encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", piped)
    monkeypatch.setattr(sys, "stderr", terminal)
    monkeypatch.setattr(sys, "stdin", typed)
    ask.configure_streams(console_input=True)
    assert (piped.encoding, terminal.encoding, typed.encoding) == ("utf-8", "cp1252", "utf-8")
    for stream in (piped, terminal):
        stream.write("答案")
        stream.flush()
    assert piped.buffer.getvalue() == "答案".encode() and terminal.buffer.getvalue() == b"\\u7b54\\u6848"
    assert typed.readline() == "帮我找几篇论文。\n"  # what a pipe of UTF-8 holds, not cp1252's reading of it


# ---- --check, --probe, --import-notes ---------------------------------------------------------------------


def test_check_prints_the_identity_and_runs_nothing(tmp_path, capsys):
    assert main(["--check", *folders(tmp_path)], build=scripted()) == 0
    out = capsys.readouterr().out
    assert "identity: OK" in out and "frozen record: results/final-m7.json" in out
    session = json.loads((run_folder(tmp_path) / "session.json").read_text(encoding="utf-8"))
    assert session["kind"] == "check" and session["turns"] == 0


def test_probe_reports_each_id_and_a_search_without_the_model(tmp_path, capsys):
    clock = FakeClock()
    pages = {paper_url("2510.99999"): feed(), paper_url("cs/0112017"): feed(OLD_STYLE),
             SEARCH: feed(ENTRY.format(id=A), ENTRY.format(id=B))}
    code = main(["--probe", "2510.99999", "cs/0112017", *folders(tmp_path)], today=lambda: TODAY,
                build=no_build, transport=arxiv_transport(pages), clock=clock, sleep=clock.sleep)
    assert code == 0 and clock.sleeps == [3.0, 3.0]  # three API requests, 3 s apart
    probe = json.loads((run_folder(tmp_path) / "probe.json").read_text(encoding="utf-8"))
    assert probe["today"] == TODAY
    assert probe["ids"] == [
        {"id": "2510.99999", "error": "NOT_FOUND", "found": None, "title": None, "http_status": 200,
         "answered": True, "default_parse": [], "lenient_parse": []},
        {"id": "cs/0112017", "error": "NOT_FOUND", "found": None, "title": None, "http_status": 200,
         "answered": True, "default_parse": None, "lenient_parse": []}]
    assert probe["search"]["hits"] == [A, B] and probe["search"]["error"] is None
    assert json.loads((run_folder(tmp_path) / "session.json").read_text(encoding="utf-8"))["kind"] == "probe"
    assert "probe cs/0112017: get_paper NOT_FOUND; HTTP 200; default parse None, lenient parse []" in (
        capsys.readouterr().out)


def test_probe_keeps_the_last_runs_interval(tmp_path):
    clock = FakeClock()
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "last-requests.json").write_text(json.dumps({"export.arxiv.org": clock.now - 0.5}),
                                                           encoding="utf-8")
    main(["--probe", "2510.99999", *folders(tmp_path)], today=lambda: TODAY, build=no_build,
         transport=arxiv_transport({paper_url("2510.99999"): feed(), SEARCH: feed()}), clock=clock,
         sleep=clock.sleep)
    assert clock.sleeps == [2.5, 3.0]


def test_probe_sends_every_request_once_and_records_a_failure_as_it_came(tmp_path):
    clock, seen = FakeClock(), []

    def busy(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(503, text="busy")

    code = main(["--probe", "2510.99999", *folders(tmp_path)], today=lambda: TODAY, build=no_build,
                transport=httpx.MockTransport(busy), clock=clock, sleep=clock.sleep)
    assert code == 0 and len(seen) == 2 and clock.sleeps == [3.0]  # the id and the search, no retry
    probe = json.loads((run_folder(tmp_path) / "probe.json").read_text(encoding="utf-8"))
    assert probe["ids"] == [{"id": "2510.99999", "error": "UPSTREAM_ERROR", "found": None, "title": None,
                             "http_status": 503, "answered": False, "default_parse": None,
                             "lenient_parse": None}]
    assert (probe["search"]["error"], probe["search"]["http_status"]) == ("UPSTREAM_ERROR", 503)


def test_a_rate_limit_state_that_cannot_be_saved_holds_the_run_until_the_intervals_have_passed(tmp_path,
                                                                                               capsys):
    clock = FakeClock()
    (tmp_path / "state" / "last-requests.json").mkdir(parents=True)  # a folder where the file goes
    code = main(["--probe", "2510.99999", *folders(tmp_path)], today=lambda: TODAY, build=no_build,
                transport=arxiv_transport({paper_url("2510.99999"): feed(), SEARCH: feed()}), clock=clock,
                sleep=clock.sleep)
    assert code == 0 and "warning: the rate-limit state could not be saved" in capsys.readouterr().out
    assert clock.sleeps == [3.0, 3.0, 3.0]  # no state read: a whole interval; the search's; then the close's


def test_import_notes_counts_and_imports_a_file_once(tmp_path, capsys):
    notes = tmp_path / "reading.md"
    notes.write_text("## 记录\n\n### 2026-11-01 A paper\n\n- 来源:https://arxiv.org/abs/2611.00001;2026-11-01;论文\n"
                     "\n### 2026-11-02 A blog post\n\n- 来源:https://example.com/post;2026-11-02;博客\n",
                     encoding="utf-8")
    assert main(["--import-notes", str(notes), *folders(tmp_path)], build=no_build) == 0
    assert main(["--import-notes", str(notes), *folders(tmp_path)], build=no_build) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == ["2 entries: 1 imported, 0 already in the library, 1 without an arXiv id skipped",
                   "2 entries: 0 imported, 1 already in the library, 1 without an arXiv id skipped"]
    assert not (tmp_path / "runs").exists()  # no run folder
    assert main(["--import-notes", str(tmp_path / "absent.md"), *folders(tmp_path)], build=no_build) == 2


# ---- exit codes -------------------------------------------------------------------------------------------


def test_a_turn_that_raises_is_recorded_and_the_run_exits_1(tmp_path):
    code = main(["--batch", str(write_batch(tmp_path, 2)), *folders(tmp_path)], today=lambda: TODAY,
                build=scripted(RuntimeError("the server went away"), turn(content="ok")))
    rows = index(tmp_path)
    assert code == 1 and [row["error"] for row in rows] == ["RuntimeError: the server went away", None]


def write_batch(tmp_path, count: int) -> Path:
    path = tmp_path / "batch.jsonl"
    lines = [json.dumps({"id": f"R{n}", "request": f"request {n}"}) + "\n" for n in range(count)]
    path.write_text("".join(lines), encoding="utf-8")
    return path


def test_a_system_that_cannot_be_assembled_exits_2_and_says_why(tmp_path, capsys):
    def failing(args, folder, out):
        raise ask.SetupFailed("the GPU is not idle")

    assert main(["hello", *folders(tmp_path)], build=failing) == 2
    assert "FAIL: the GPU is not idle" in capsys.readouterr().err
    session = json.loads((run_folder(tmp_path) / "session.json").read_text(encoding="utf-8"))
    assert session["setup_error"] == "the GPU is not idle"
    assert not (tmp_path / "runs" / "index.jsonl").exists()


def test_an_identity_that_disagrees_exits_2_before_any_turn(tmp_path, capsys):
    problem = "server: 4 slots (/props), not 1"
    code = main(["hello", *folders(tmp_path)], build=scripted(turn(content="never"), problems=[problem]))
    assert code == 2 and f"identity: 1 problem(s)\n  {problem}" in capsys.readouterr().out
    assert not (tmp_path / "runs" / "index.jsonl").exists()


def test_a_run_started_outside_the_repository_root_stops_before_it_writes_anything(tmp_path, monkeypatch,
                                                                                  capsys):
    monkeypatch.chdir(tmp_path)
    for argv in (["hello"], ["--probe", "2510.99999"], ["--import-notes", "reading.md"], ["--check"]):
        assert main([*argv, *folders(tmp_path)], build=no_build) == 2
    assert "FAIL: run it from the repository root" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []  # no run folder, no lock, no state, no notes


def test_help_says_where_to_run_where_the_files_go_and_the_exit_codes(capsys):
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert stopped.value.code == 0 and "Run it from the repository root." in out
    assert "(default runs/live)" in out and "(default notes)" in out and "3 when another run holds" in out


def test_a_run_while_another_holds_the_lock_exits_3(tmp_path, capsys):
    with InstanceLock(tmp_path / "state" / "ask.lock"):
        assert main(["hello", *folders(tmp_path)], build=no_build) == 3
    assert "one run at a time" in capsys.readouterr().err and not (tmp_path / "runs").exists()


def test_ctrl_c_closes_the_assembly_releases_the_lock_and_exits_130(tmp_path):
    class Interrupted:
        name = "interrupted"

        def next_turn(self, messages, tools):
            raise KeyboardInterrupt

    closed: list = []

    def build(args, folder, out):
        return ask.Assembly(Built(Interrupted(), None, {}), {}, [], {}, {}, [lambda: closed.append(True)])

    assert main(["hello", *folders(tmp_path)], today=lambda: TODAY, build=build) == ask.INTERRUPTED
    assert closed == [True]
    InstanceLock(tmp_path / "state" / "ask.lock").acquire()  # free again
    assert "ended" in json.loads((run_folder(tmp_path) / "session.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("argv", [["hello", "--check"], ["--clarify-answer", "x", "--check"], [],
                                  ["--check", "--batch-name", "A"], ["  "],
                                  ["hi", *["--clarify-answer", "a"] * 3],
                                  ["--check", "--server", "http://127.0.0.1:8080", "--port", "8081"]])
def test_usage_errors_exit_2(tmp_path, argv):
    with pytest.raises(SystemExit) as stopped:
        main([*argv, *folders(tmp_path)], build=no_build)
    assert stopped.value.code == 2


def test_a_batch_file_that_cannot_be_used_exits_2(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    for text in ('{"id": "S1"}\n', "not json\n", '{"request": "x", "clarify_answer": ["a", "b", "c"]}\n'):
        bad.write_text(text, encoding="utf-8")
        assert main(["--batch", str(bad), *folders(tmp_path)], build=no_build) == 2
    assert main(["--batch", str(write_batch(tmp_path, 1)), "--batch-name", "Z", *folders(tmp_path)],
                build=no_build) == 2
    assert "holds no request of batch 'Z'" in capsys.readouterr().err


# ---- the real assembly over the fake final system ---------------------------------------------------------


def client_over(fake):
    return lambda url, out: ProgressClient(url, out=out, transport=fake.transport())


def test_ask_runs_the_frozen_system_assembled_over_a_running_server(tmp_path, monkeypatch, capsys):
    paths = write_final(tmp_path / "final", monkeypatch)
    frozen = load_frozen(paths["final"], paths["config"])
    fake = final_server(paths, text_turns(3))

    def build(args, folder, out):
        return ask.assemble(args, folder, out, load=lambda: frozen, client=client_over(fake))

    code = main(["请总结一下。", "--server", "http://127.0.0.1:8080/", *folders(tmp_path)],
                today=lambda: TODAY, build=build)
    out = capsys.readouterr().out
    assert code == 0 and "answer:\nHere is the answer." in out
    assert "step 1 | votes 2/2 | thinking 2 tok | answer 19 chars" in out and "llama-server request 3:" in out
    session = json.loads((run_folder(tmp_path) / "session.json").read_text(encoding="utf-8"))
    assert session["identity_problems"] == [] and session["info"]["generation"] == frozen.record["generation"]
    assert session["info"]["vote_mode"] == "final" and session["props"] == fake.props
    assert session["server"] == {"url": "http://127.0.0.1:8080", "started_here": False, "command": None}
    assert session["frozen"]["sha256"] == frozen.frozen_sha256 and len(fake.requests) == 3


def test_a_running_server_that_is_not_the_evaluated_one_is_refused(tmp_path, monkeypatch, capsys):
    paths = write_final(tmp_path / "final", monkeypatch)
    frozen = load_frozen(paths["final"], paths["config"])
    fake = final_server(paths, [], total_slots=4)

    def build(args, folder, out):
        return ask.assemble(args, folder, out, load=lambda: frozen, client=client_over(fake))

    assert main(["hello", "--server", "http://127.0.0.1:8080", *folders(tmp_path)], build=build) == 2
    assert "server: 4 slots (/props), not 1" in capsys.readouterr().out and fake.requests == []


class StubServer:
    """LocalServer's surface for assemble: started over the frozen paths, closed when the run ends."""

    def __init__(self, made: list, refuse: Exception | None = None):
        self.made, self.refuse = made, refuse

    def __call__(self, binary, **options):
        self.binary, self.options = binary, options
        self.url = f"http://127.0.0.1:{options['port']}"
        self.cmd = [str(binary), "-m", options["model"]]
        self.props, self.closed = None, False
        self.made.append(self)
        return self

    def __enter__(self):
        if self.refuse is not None:
            raise self.refuse
        self.props = server_props(self.options["model"])
        return self

    def close(self):
        self.closed = True


def test_ask_starts_its_own_server_over_the_frozen_files_when_the_gpu_is_idle_and_stops_it(tmp_path,
                                                                                         monkeypatch):
    paths = write_final(tmp_path / "final", monkeypatch)
    frozen = load_frozen(paths["final"], paths["config"])
    fake, made = final_server(paths, text_turns(3)), []

    def build(args, folder, out):
        return ask.assemble(args, folder, out, load=lambda: frozen, client=client_over(fake),
                            start_server=StubServer(made), gpu=lambda: GpuSample(300.0, None))

    code = main(["请总结一下。", "--llama-server", "bin/llama-server", *folders(tmp_path)],
                today=lambda: TODAY, build=build)
    (server,) = made
    assert code == 0 and server.closed and str(server.binary) == str(Path("bin/llama-server"))
    assert server.options == {"model": paths["base"], "adapter": paths["adapter"], "port": 8080,
                              "ctx_size": 8192, "timeout": 120, "extra": ("-np", "1"),
                              "log": run_folder(tmp_path) / "server.log"}
    session = json.loads((run_folder(tmp_path) / "session.json").read_text(encoding="utf-8"))
    assert session["server"]["started_here"] and session["server"]["command"] == ["llama-server", "-m",
                                                                                   paths["base"]]
    assert session["identity_problems"] == []


@pytest.mark.parametrize(("gpu", "refuse", "message"), [
    (lambda: GpuSample(4000.0, None), None, "the GPU is not idle: 4000 MiB in use, at most 1500"),
    (lambda: GpuSample(300.0, None), PortBusy("a server already answers at http://127.0.0.1:8080"),
     "use it with --server http://127.0.0.1:8080"),
])
def test_a_busy_gpu_or_port_stops_the_run_before_the_model_loads(tmp_path, monkeypatch, capsys, gpu, refuse,
                                                                  message):
    paths = write_final(tmp_path / "final", monkeypatch)
    frozen = load_frozen(paths["final"], paths["config"])
    made: list = []

    def build(args, folder, out):
        return ask.assemble(args, folder, out, load=lambda: frozen,
                            client=client_over(final_server(paths, [])),
                            start_server=StubServer(made, refuse), gpu=gpu)

    assert main(["hello", "--llama-server", "llama-server", *folders(tmp_path)], build=build) == 2
    assert message in capsys.readouterr().err


def test_without_nvidia_smi_the_server_starts_after_a_warning(tmp_path, monkeypatch, capsys):
    paths = write_final(tmp_path / "final", monkeypatch)
    frozen = load_frozen(paths["final"], paths["config"])
    made: list = []

    def build(args, folder, out):
        return ask.assemble(args, folder, out, load=lambda: frozen,
                            client=client_over(final_server(paths, [])),
                            start_server=StubServer(made), gpu=lambda: None)

    assert main(["--check", "--llama-server", "llama-server", *folders(tmp_path)], build=build) == 0
    out = capsys.readouterr().out
    assert "warning: no nvidia-smi on PATH, so whether the GPU is idle cannot be told" in out
    assert "identity: OK" in out and len(made) == 1


def test_an_nvidia_smi_that_fails_stops_the_run_before_a_server_starts(tmp_path, monkeypatch, capsys):
    paths = write_final(tmp_path / "final", monkeypatch)
    frozen = load_frozen(paths["final"], paths["config"])
    made: list = []

    def failing():
        raise GpuError("nvidia-smi exited with 9: no devices were found")

    def build(args, folder, out):
        return ask.assemble(args, folder, out, load=lambda: frozen,
                            client=client_over(final_server(paths, [])),
                            start_server=StubServer(made), gpu=failing)

    assert main(["--check", "--llama-server", "llama-server", *folders(tmp_path)], build=build) == 2
    err = capsys.readouterr().err
    assert "cannot tell whether the GPU is idle: nvidia-smi exited with 9" in err and "--server" in err
    assert made == []


def test_the_gpu_is_asked_only_when_nvidia_smi_is_on_the_path(monkeypatch):
    monkeypatch.setattr(ask, "gpu_sample", lambda: GpuSample(300.0, None))
    monkeypatch.setattr(ask.shutil, "which", lambda name: None)
    assert ask.gpu_now() is None
    monkeypatch.setattr(ask.shutil, "which", lambda name: f"bin/{name}")
    assert ask.gpu_now() == GpuSample(300.0, None)


def test_no_llama_server_and_a_frozen_record_that_does_not_hold_exit_2(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("LLAMA_SERVER", raising=False)  # this process only, restored after the test

    def mismatched():
        raise FrozenMismatch(["family: another family"])

    assert main(["--check", *folders(tmp_path)], build=lambda args, folder, out: ask.assemble(
        args, folder, out, load=mismatched)) == 2
    assert "the frozen system does not hold: family: another family" in capsys.readouterr().err
    paths = write_final(tmp_path / "final", monkeypatch)
    frozen = load_frozen(paths["final"], paths["config"])
    assert main(["--check", *folders(tmp_path)], build=lambda args, folder, out: ask.assemble(
        args, folder, out, load=lambda: frozen)) == 2
    assert "no llama-server given" in capsys.readouterr().err


# ---- --web ------------------------------------------------------------------------------------------------


class Browser(threading.Thread):
    """The page's user, in a helper thread while main runs in the test's main thread: it waits for the launch
    link (main's browse), opens it, runs script(page) and, whatever happens, posts /api/quit, so main
    returns. As a context manager around main, it ends with the block and raises what script raised;
    result is what script returned."""

    def __init__(self, script) -> None:
        super().__init__(daemon=True)
        self.script, self.url, self.result, self.error = script, None, None, None
        self.given = threading.Event()
        self.start()

    def browse(self, url: str) -> None:
        self.url = url
        self.given.set()

    def run(self) -> None:
        if not self.given.wait(120) or self.url is None:
            return
        page = Page.of(self.url)
        try:
            assert page.launch(self.url).status == 303
            self.result = self.script(page)
        except BaseException as exc:  # noqa: BLE001  raised again on leaving the block, in the test's thread
            self.error = exc
        finally:
            try:
                page.post("/api/quit")
            except OSError:  # the run has ended already
                pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, kind, value, traceback) -> None:
        self.given.set()  # a run that never served the page: nothing to do
        self.join(TIMEOUT * 3)
        if kind is None:
            assert not self.is_alive()
            if self.error is not None:
                raise self.error


def web_run(tmp_path, *options: str) -> list[str]:
    return ["--web", "--web-port", "0", *options, *folders(tmp_path)]


def session_of(tmp_path) -> dict:
    return json.loads((run_folder(tmp_path) / "session.json").read_text(encoding="utf-8"))


def without_timing(path: Path) -> str:
    """A turn file with its seconds line left out (they differ from one run to the next)."""
    return "".join(line for line in path.read_text(encoding="utf-8").splitlines(keepends=True)
                   if not line.startswith("- seconds: "))


def test_web_runs_the_pages_turns_and_writes_what_interactive_writes(tmp_path, capsys):
    question = call("ask_clarification", "q1", question="哪个主题?", missing=["topic"])

    def model():
        return scripted(turn(question), turn(call("list_notes", "l1")), turn(content=f"推荐 **{A}**。"))

    typed = iter(["帮我找几篇论文。", "speculative decoding", "/quit"])
    console = tmp_path / "console"
    assert main(["--interactive", *folders(console)], today=lambda: TODAY, build=model(),
                ask=lambda prompt: next(typed)) == 0
    console_out = capsys.readouterr().out

    def chat(page: Page):
        assert page.ask("帮我找几篇论文。").status == 202
        first = page.done(1)
        assert page.state()["status"]["pending"] == {"exchange": 1, "follow_ups": 0}
        assert page.ask("speculative decoding").status == 202
        return first, page.done(2)

    served = tmp_path / "web"
    with Browser(chat) as browser:
        code = main(web_run(served), today=lambda: TODAY, build=model(), browse=browser.browse)
    first, second = browser.result
    out, err = capsys.readouterr()
    assert code == 0 and first["result"]["after"] == "answer"
    assert first["result"]["html"] == "<p>哪个主题?</p>"
    assert (second["sent"], second["follow_up"]) == ("帮我找几篇论文。补充:speculative decoding", 1)
    assert f'<strong><a href="https://arxiv.org/abs/{A}" target="_blank" rel="noopener noreferrer">' in (
        second["result"]["html"])
    assert second["result"]["cited_unseen"] == [A]
    assert second["steps"][0]["tools"][0]["summary"] == "0 note(s)"

    def rows(folder):
        return [{key: value for key, value in row.items() if key not in ("run", "seconds")}
                for row in index(folder)]

    assert rows(served) == rows(console) and len(rows(served)) == 2
    for name in ("turn-001.json", "turn-002.json", "turn-001.md", "turn-002.md"):
        assert without_timing(run_folder(served) / name) == without_timing(run_folder(console) / name), name
    session = session_of(served)
    assert session["kind"] == "web" and session["turns"] == 2 and "ended" in session
    port = session["web"]["port"]
    assert browser.url.startswith(f"http://127.0.0.1:{port}/?t=") and port != 0
    for line in ("turn 1 on 2026-11-10", "question: 哪个主题?", "turn 2 on 2026-11-10, follow-up 1",
                 f"answer:\n推荐 **{A}**。", "  tool list_notes: 0 note(s)", "2 turn(s); run folder"):
        assert line in out and line in console_out, line
    token = browser.url.partition("?t=")[2]
    assert out.count(token) == 1 and f"page: {browser.url}\n" in out and token not in err  # no request logged
    web.Site(port).close()  # the page's port is free
    InstanceLock(served / "state" / "ask.lock").acquire()  # and so is the lock


def test_web_over_the_fake_final_system_shows_the_votes_and_the_reasoning(tmp_path, monkeypatch, capsys):
    paths = write_final(tmp_path / "final", monkeypatch)
    frozen = load_frozen(paths["final"], paths["config"])
    fake = final_server(paths, text_turns(3))

    def build(args, folder, out):
        return ask.assemble(args, folder, out, load=lambda: frozen, client=client_over(fake))

    def chat(page: Page):
        assert page.ask("请总结一下。").status == 202
        return page.done(1), page.state()["system"]

    with Browser(chat) as browser:
        code = main(web_run(tmp_path, "--server", "http://127.0.0.1:8080/"), today=lambda: TODAY, build=build,
                    browse=browser.browse)
    exchange, system = browser.result
    assert code == 0 and len(fake.requests) == 3  # as the same request on the command line sends
    (step,) = exchange["steps"]
    assert step["votes"] == {"k": 2, "won": 2, "representative": 0, "failed": 0} and step["state"] == "done"
    assert step["thinking"] == {"tokens": 2, "forced": False} and "Thinking it over." in step["reasoning"]
    assert exchange["result"]["html"] == "<p>Here is the answer.</p>"
    assert exchange["result"]["status"] == "answered"
    assert system == {"model": system["model"], "votes": 2, "vote_mode": "final", "reasoning_budget": 32,
                      "rules_version": RULES_VERSION, "build": EVALUATED_SERVER["llama_cpp_build"],
                      "frozen_sha256": frozen.frozen_sha256, "server": "http://127.0.0.1:8080",
                      "run": run_folder(tmp_path).name}
    assert "step 1 | votes 2/2 | thinking 2 tok | answer 19 chars" in capsys.readouterr().out


class Interrupted:
    name = "interrupted"

    def next_turn(self, messages, tools):
        raise KeyboardInterrupt


@pytest.mark.parametrize("where", ["in a turn", "while the page waits"])
def test_ctrl_c_closes_the_page_and_the_assembly_frees_the_port_and_the_lock_and_exits_130(tmp_path, capsys,
                                                                                         where):
    closed: list = []

    def build(args, folder, out):
        return ask.Assembly(Built(Interrupted(), None, {}), {}, [], {}, {}, [lambda: closed.append(True)])

    def chat(page: Page):
        with page.stream(page.state()["seq"]) as stream:
            if where == "in a turn":  # the model raises it
                assert page.ask("hello").status == 202
            else:  # Ctrl+C reaches the main thread as it waits on the page's queue
                signal.raise_signal(signal.SIGINT)
            return [event for event in stream.rest() if "event" in event]

    with Browser(chat) as browser:
        code = main(web_run(tmp_path), today=lambda: TODAY, build=build, browse=browser.browse)
    events = browser.result
    assert code == ask.INTERRUPTED and closed == [True]
    assert events[-1]["event"] == "bye" and events[-1]["data"] == {"reason": "interrupted"}
    assert "interrupted: the server is stopped and the lock released" in capsys.readouterr().err
    session = session_of(tmp_path)
    assert "ended" in session and session["turns"] == 0  # a turn Ctrl+C ended is not written
    web.Site(session["web"]["port"]).close()  # the port is free
    InstanceLock(tmp_path / "state" / "ask.lock").acquire()  # and the lock


def test_a_busy_web_port_exits_2_before_the_system_is_assembled(tmp_path, capsys):
    with socket.socket() as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
        code = main(["--web", "--web-port", str(port), *folders(tmp_path)], build=no_build,
                    browse=lambda url: pytest.fail("a page was served"))
    assert code == 2
    err = capsys.readouterr().err
    assert f"FAIL: the page cannot listen on 127.0.0.1:{port}" in err and "--web-port" in err
    session = session_of(tmp_path)
    assert session["kind"] == "web" and session["web"] == {"port": port}
    assert "the page cannot listen" in session["setup_error"] and "ended" in session
    assert not (tmp_path / "runs" / "index.jsonl").exists()
    InstanceLock(tmp_path / "state" / "ask.lock").acquire()  # free


def test_the_web_port_is_8765_unless_given():
    parser = ask.build_parser()
    for argv, port in ((["--web"], 8765), (["--web", "--web-port", "0"], 0),
                       (["--web", "--web-port", "9000"], 9000),
                       (["--web", "--port", "9000", "--web-port", "8080"], 8080)):
        args = parser.parse_args(argv)
        assert ask.mode_of(args, parser) == "web" and args.web_port == port, argv


@pytest.mark.parametrize("argv", [["--check", "--web-port", "9000"], ["--check", "--no-browser"],
                                  ["hello", "--web"], ["--web", "--interactive"], ["--web", "--check"],
                                  ["--web", "--clarify-answer", "x"], ["--web", "--batch-name", "A"],
                                  ["--web", "--web-port", "8080"],
                                  ["--web", "--port", "9000", "--web-port", "9000"],
                                  ["--web", "--server", "http://127.0.0.1:8080", "--web-port", "8080"],
                                  ["--web", "--web-port", "65536"], ["--web", "--web-port", "-1"],
                                  ["--web", "--web-port", "80"]])
def test_web_usage_errors_exit_2(tmp_path, argv):
    with pytest.raises(SystemExit) as stopped:
        main([*argv, *folders(tmp_path)], build=no_build)
    assert stopped.value.code == 2


def test_help_names_the_web_mode_and_its_options(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = " ".join(capsys.readouterr().out.split())
    assert "--web " in out and "--web-port N" in out and "--no-browser" in out and "127.0.0.1" in out


def test_a_web_turn_that_raises_is_shown_and_the_run_goes_on_and_exits_1(tmp_path):
    def chat(page: Page):
        assert page.ask("hello").status == 202
        first = page.done(1)
        assert page.ask("again").status == 202
        return first, page.done(2)

    with Browser(chat) as browser:
        code = main(web_run(tmp_path), today=lambda: TODAY, browse=browser.browse,
                    build=scripted(RuntimeError("the server went away"), turn(content="ok")))
    first, second = browser.result
    assert code == 1 and first["result"]["error"] == "RuntimeError: the server went away"
    assert second["result"]["status"] == "answered"
    assert [row["error"] for row in index(tmp_path)] == ["RuntimeError: the server went away", None]


def test_no_browser_prints_the_link_and_opens_nothing(tmp_path, monkeypatch):
    def chat(page: Page):
        return page.state()["status"]

    printed = ask._print

    def console(line: str) -> None:  # the page's user reads the link off the console
        printed(line)
        if line.startswith("page: "):
            browser.browse(line.removeprefix("page: "))

    monkeypatch.setattr(ask, "_print", console)
    with Browser(chat) as browser:
        code = main([*web_run(tmp_path), "--no-browser"], today=lambda: TODAY, build=scripted(),
                    browse=lambda url: pytest.fail("a browser was opened"))
    assert code == 0 and browser.result == {"phase": "idle", "pending": None}


def test_web_with_an_identity_that_disagrees_exits_2_without_serving_the_page(tmp_path, capsys):
    problem = "server: 4 slots (/props), not 1"
    code = main(web_run(tmp_path), build=scripted(turn(content="never"), problems=[problem]),
                browse=lambda url: pytest.fail("a page was served"))
    assert code == 2 and f"identity: 1 problem(s)\n  {problem}" in capsys.readouterr().out
    session = session_of(tmp_path)
    assert session["kind"] == "web" and not (tmp_path / "runs" / "index.jsonl").exists()
    web.Site(session["web"]["port"]).close()  # the page's port is free again


def test_the_page_refuses_exactly_what_the_console_never_sends_as_typed():
    texts = ["/new 别的主题", "/new", "/quit", "/help", "/ x", "/NEW x", "exit", "QUIT", "Exit",
             "quit smoking", "exit now", "a /new", "帮我找几篇论文。", "Find papers on RAG.", "／new x",
             "-exit", "exits"]

    class Console:  # the runner interactive() asks for turns: what the console sends to the model
        def __init__(self) -> None:
            self.sent: list[str] = []

        def turn(self, text: str, *, follow_up: int = 0):
            self.sent.append(text)
            return ask.LiveTurn(text, TODAY, None, "not run", 0.0)

    sends = {}
    for text in texts:
        console, lines = Console(), [text]

        def typed(prompt: str, lines: list = lines) -> str:
            if not lines:
                raise EOFError
            return lines.pop()

        ask.interactive(console, typed, lambda line: None)
        chat = web.ChatState(web.Hub())
        try:
            chat.submit(text)
        except web.Command:
            sends[text] = False
        else:
            sends[text] = True
        assert sends[text] == (console.sent == [text.strip()]), text
    assert set(sends.values()) == {True, False}


def test_web_refuses_a_message_utf8_cannot_encode_and_the_run_goes_on(tmp_path):
    def chat(page: Page):
        refused = page.post("/api/ask", raw=b'{"text": "find \\ud83d papers"}')  # half an emoji
        assert refused.status == 400 and refused.json() == {"error": "not_text"}
        assert page.ask("find papers").status == 202
        return page.done(1)

    with Browser(chat) as browser:
        code = main(web_run(tmp_path), today=lambda: TODAY, build=scripted(turn(content="ok")),
                    browse=browser.browse)
    assert code == 0 and browser.result["request"] == "find papers"
    assert session_of(tmp_path)["turns"] == 1
    assert [row["request"] for row in index(tmp_path)] == ["find papers"]

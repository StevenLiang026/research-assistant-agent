"""research_agent.live.session offline: the follow-up after a clarification, the single-instance lock, the
progress wrappers (they hand on everything and change no request byte, each against the system unwrapped,
even when the console cannot print), their events (what the lines print, from the same clock and rate-limit
readings, with no reading of their own, even when the sink raises or an event cannot be made), the
conversation that carries a clarification's follow-ups from one message to the next, and the files a run
writes."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest
from support.fakes import ScriptedModel, call, turn
from support.live_fakes import final_server, text_turns, write_final
from support.llama_fakes import FakeLlama, Scripted

from research_agent.agent.context import family_token_counter
from research_agent.agent.loop import run_turn
from research_agent.agent.prompts import system_prompt
from research_agent.agent.types import AssistantTurn, ToolOutcome, Transcript
from research_agent.eval.systems import Built, build_system
from research_agent.live.frozen import load_frozen
from research_agent.live.session import (
    AWAITS_ANSWER,
    EVENT_KINDS,
    FOLLOW_UP_EN,
    FOLLOW_UP_ZH,
    INDEX_FILE,
    MAX_FOLLOW_UPS,
    RESTATE,
    Conversation,
    InstanceLock,
    LiveTurn,
    LockBusy,
    ProgressClient,
    ProgressModel,
    SafeOut,
    SessionFiles,
    new_run_folder,
    pack_follow_up,
    progress_tools,
    run_live_turn,
    step_event,
    step_line,
    tool_summary,
)
from research_agent.llm.local import LocalClient
from research_agent.policy import decide
from research_agent.tools.schemas import MODEL_TOOLS
from research_agent.validation import check_call

TODAY = "2026-11-10"
ZH_REQUEST, EN_REQUEST = "帮我找几篇论文。", "Find me a few papers."
A = "2611.00001"
B = "2611.00002"


# ---- the follow-up ----------------------------------------------------------------------------------------


def test_pack_follow_up_keeps_the_request_and_appends_the_answer_on_one_line():
    assert FOLLOW_UP_ZH == "补充:" and FOLLOW_UP_EN == " To clarify: "
    assert pack_follow_up(ZH_REQUEST, "speculative decoding") == "帮我找几篇论文。补充:speculative decoding"
    assert (pack_follow_up(EN_REQUEST, "  mixture of\nexperts ")
            == "Find me a few papers. To clarify: mixture of experts")
    assert pack_follow_up("2501.12948 是哪篇论文?", "作者") == "2501.12948 是哪篇论文?补充:作者"
    for empty in ("", "  \n\t "):
        assert pack_follow_up(ZH_REQUEST, empty) is None and pack_follow_up(EN_REQUEST, empty) is None


def test_a_follow_up_turn_starts_from_the_system_prompt_and_the_packed_request_alone():
    question = call("ask_clarification", "q1", question="你想找哪个主题的论文?", missing=["topic"])
    model = ScriptedModel([turn(question), turn(content="找到了。")])
    built = Built(model, None, {})
    first = run_live_turn(ZH_REQUEST, built=built, tools={}, today=TODAY, out=lambda line: None)
    assert first.clarified and first.transcript.final_text == "你想找哪个主题的论文?"
    packed = pack_follow_up(first.request, "speculative decoding")
    second = run_live_turn(packed, built=built, tools={}, today=TODAY, out=lambda line: None)
    assert second.status == "answered"
    assert model.contexts[1] == [{"role": "system", "content": system_prompt(TODAY)},
                                 {"role": "user", "content": "帮我找几篇论文。补充:speculative decoding"}]


def test_a_turn_that_raises_is_recorded_and_does_not_end_the_run():
    model = ScriptedModel([RuntimeError("the server went away")])
    result = run_live_turn(EN_REQUEST, built=Built(model, None, {}), tools={}, today=TODAY,
                           out=lambda line: None)
    assert result.transcript is None and result.status is None and not result.clarified
    assert result.error == "RuntimeError: the server went away"


# ---- the lock ---------------------------------------------------------------------------------------------


def test_the_lock_admits_one_run_names_its_holder_and_can_be_released_twice(tmp_path):
    path = tmp_path / "live" / "ask.lock"
    first = InstanceLock(path)
    first.acquire()
    record = path.with_name("ask.lock.json")
    assert json.loads(record.read_text(encoding="utf-8"))["pid"] == os.getpid()
    with pytest.raises(LockBusy) as busy:
        InstanceLock(path).acquire()
    assert busy.value.pid == os.getpid() and "one run at a time" in str(busy.value)
    first.release()
    first.release()
    assert not record.exists()
    with InstanceLock(path) as again:
        assert again.holder() == os.getpid()
    assert InstanceLock(path).holder() is None


def test_the_system_releases_the_lock_of_a_process_that_ended_without_releasing_it(tmp_path):
    path = tmp_path / "ask.lock"
    code = f"from research_agent.live.session import InstanceLock; InstanceLock({str(path)!r}).acquire()"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
    with InstanceLock(path) as lock:  # the child's record is left behind; the lock is not
        assert lock.holder() == os.getpid()


# ---- the progress wrappers --------------------------------------------------------------------------------


class Listening:
    """A model with the optional parts the loop and the runners use."""

    name = "listening"
    extra = "kept"

    def __init__(self, answer: AssistantTurn) -> None:
        self.answer, self.begun = answer, 0

    def begin_task(self) -> None:
        self.begun += 1

    def cost(self, given: AssistantTurn) -> float:
        return 0.25

    def next_turn(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        return self.answer


def test_the_progress_model_hands_on_everything_and_prints_one_ascii_line_per_request():
    answer = AssistantTurn(content="", tool_calls=[call("get_paper", "local_0", arxiv_id=A)], model="m",
                           votes={"k": 5, "chosen": "K", "counts": {"K": 4, "L": 1}},
                           thinking={"reasoning_tokens": 812, "forced": True},
                           rules=[{"rule": "id_repair", "before": {}, "after": {}}])
    inner, lines = Listening(answer), []
    ticks = iter([10.0, 12.5, 20.0, 20.0])
    model = ProgressModel(inner, lines.append, clock=lambda: next(ticks))
    assert model.name == "listening" and model.extra == "kept" and model.cost(answer) == 0.25
    transcript = run_turn(EN_REQUEST, model=model, executor=_executor({"get_paper": _found}), decide=decide,
                          today=TODAY, max_steps=1)
    assert inner.begun == 1 and transcript.status == "step_limit"
    assert lines == [(f"step 1 | votes 4/5 | thinking 812 tok closed by force | rules id_repair | "
                      f"calls get_paper({A}) | 2.5 s")]
    assert model.next_turn([], []) is answer  # the very turn, unchanged; outside the loop, the same step
    assert lines[-1].startswith("step 1 attempt 2 | ")


def test_a_step_line_names_the_attempt_a_vote_nobody_won_and_a_text_answer():
    voted = AssistantTurn(content="答案", tool_calls=[], model="m",
                          votes={"k": 5, "chosen": None, "counts": {}, "errors": [{}, {}]})
    assert step_line(2, 3, voted, 1.0) == ("step 2 attempt 3 | votes none of 5, the greedy answer | 2 "
                                           "candidate(s) failed | answer 2 chars | 1.0 s")


def _found(args: dict) -> ToolOutcome:
    return ToolOutcome({"arxiv_id": args["arxiv_id"], "title": "标题", "abstract": "摘要"})


def _executor(tools: dict):
    from research_agent.agent.executor import ToolExecutor

    return ToolExecutor(tools, check_call, today=TODAY)


def test_progress_tools_print_counts_and_codes_never_text_and_return_the_outcome_itself():
    outcomes = {"search_arxiv": ToolOutcome([{"arxiv_id": A, "title": "标题"}]),
                "get_paper": ToolOutcome({}, "NOT_FOUND"),
                "fetch_fulltext": ToolOutcome({"arxiv_id": A, "section": "introduction", "source": "abstract",
                                               "text": "正文" * 10}, "NO_FULLTEXT"),
                "write_note": ToolOutcome({"note_id": f"{A}-1", "arxiv_id": A, "path": f"{A}-1.md"}),
                "list_notes": ToolOutcome([])}
    waited = iter([0.0, 3.0] + [0.0, 0.0] * 4)
    lines: list[str] = []
    wrapped = progress_tools({name: (lambda args, name=name: outcomes[name]) for name in outcomes},
                             lines.append, waited=lambda: next(waited), clock=lambda: 0.0)
    for name, outcome in outcomes.items():
        assert wrapped[name]({}) is outcome
    assert lines == ["  tool search_arxiv: 1 hit(s), 0.0 s, 3.0 s of it waiting for the rate limit",
                     "  tool get_paper: error NOT_FOUND, 0.0 s",
                     f"  tool fetch_fulltext: NO_FULLTEXT, abstract text of {A}, 20 chars, 0.0 s",
                     f"  tool write_note: note {A}-1 ({A}-1.md), 0.0 s",
                     "  tool list_notes: 0 note(s), 0.0 s"]


def test_the_progress_client_sends_the_bytes_localclient_sends_and_keeps_its_timings():
    script = [Scripted(("你好", " there"))]
    plain, watched = FakeLlama(list(script)), FakeLlama(list(script))
    lines: list[str] = []
    request = {"lora_scale": None, "max_tokens": 8, "temperature": 0.6, "seed": 7, "top_p": 0.95, "top_k": 0,
               "min_p": 0.0, "presence_penalty": 0.0, "return_tokens": True}
    expected = LocalClient(transport=plain.transport()).complete([1, 2, 3], **request)
    client = ProgressClient(transport=watched.transport(), out=lines.append, clock=lambda: 0.0)
    got = client.complete([1, 2, 3], **request)
    assert watched.request_bytes == plain.request_bytes and got.text == expected.text
    assert client.last_timings == {"prompt_n": 3, "predicted_n": 2, "predicted_ms": 20.0}
    assert lines == ["    llama-server request 1: prompt 3 tok, generated 2 tok (stop), 0.0 s"]


def gone(line: str) -> None:
    """A console whose reader went away."""
    raise BrokenPipeError(32, "the reader went away")


def quiet(line: str) -> None:
    """A console that takes every line."""


def test_the_final_system_watched_sends_the_requests_and_gives_the_turn_it_gives_unwatched(tmp_path,
                                                                                         monkeypatch):
    paths = write_final(tmp_path / "final", monkeypatch)
    frozen = load_frozen(paths["final"], paths["config"])
    messages = [{"role": "system", "content": system_prompt(TODAY)}, {"role": "user", "content": ZH_REQUEST}]
    arms = {  # the client over the fake server, and the out of ProgressModel (None: the model unwrapped)
        "neither": (lambda fake: LocalClient(transport=fake.transport()), None),
        "the model": (lambda fake: LocalClient(transport=fake.transport()), quiet),
        "both": (lambda fake: ProgressClient(transport=fake.transport(), out=quiet), quiet),
        "a console gone": (lambda fake: ProgressClient(transport=fake.transport(), out=gone), gone),
    }
    turns, sent = {}, {}
    for arm, (client_of, out) in arms.items():
        fake = final_server(paths, text_turns(3))
        built = build_system("ft", tokenizer=frozen.tokenizer, family=frozen.family,
                             generation=frozen.settings,
                             count_tokens=family_token_counter(frozen.family, frozen.settings.mode),
                             rules=True, vote_mode=frozen.vote_mode, local_client=client_of(fake))
        model = built.model if out is None else ProgressModel(built.model, out)
        turns[arm] = model.next_turn(messages, MODEL_TOOLS)
        prompt = len(fake.requests[0]["prompt"])
        assert built.timings() == {"prompt_n": 3 * prompt, "predicted_n": 9, "predicted_ms": 90.0}, arm
        sent[arm] = fake.request_bytes
    assert len(sent["neither"]) == 3  # the greedy candidate and two samples
    for arm in arms:  # byte for byte, and the very same turn and vote
        assert sent[arm] == sent["neither"], arm
        assert turns[arm] == turns["neither"] and turns[arm].votes == turns["neither"].votes, arm
    assert "errors" not in turns["a console gone"].votes  # no candidate lost to the console
    assert turns["neither"].content == "Here is the answer."


def test_a_line_that_cannot_be_printed_ends_the_printing_and_nothing_else():
    printed: list[str] = []

    def gone_after_one(line: str) -> None:
        printed.append(line)
        gone(line)

    answer = AssistantTurn(content="ok", tool_calls=[], model="m")
    model = ProgressModel(Listening(answer), gone_after_one, clock=lambda: 0.0)
    assert model.next_turn([], []) is answer and model.next_turn([], []) is answer
    outcome = ToolOutcome([])
    tools = progress_tools({"list_notes": lambda args: outcome}, gone_after_one, clock=lambda: 0.0)
    assert tools["list_notes"]({}) is outcome and tools["list_notes"]({}) is outcome
    assert len(printed) == 2  # one line tried by each wrapper, then none

    def closed(line: str) -> None:
        raise ValueError("I/O operation on closed file.")

    def buggy(line: str) -> None:
        raise RuntimeError("a bug")

    out = SafeOut(closed)
    out("a line")
    assert out.broken
    with pytest.raises(RuntimeError):  # anything else is no failure to print: it is raised
        SafeOut(buggy)("a line")


# ---- the files of a run -----------------------------------------------------------------------------------


def test_runs_started_in_the_same_second_get_folders_of_their_own(tmp_path):
    first, second = new_run_folder(tmp_path / "runs"), new_run_folder(tmp_path / "runs")
    assert first != second and first.is_dir() and second.is_dir()
    assert all(re.fullmatch(r"\d{8}T\d{6}Z(-\d+)?", folder.name) for folder in (first, second))


def test_a_turn_with_a_note_writes_the_papers_it_was_made_from(tmp_path):
    files = SessionFiles(new_run_folder(tmp_path / "runs"))
    model = ScriptedModel([turn(call("get_paper", "g1", arxiv_id=A)),
                           turn(call("write_note", "w1", arxiv_id=A)),
                           turn(content=f"笔记已写好:{A},另见 2611.00009。")])
    note = {"note_id": f"{A}-1", "arxiv_id": A, "path": f"{A}-1.md"}
    tools = {"get_paper": _found, "write_note": lambda args: ToolOutcome(note)}
    result = run_live_turn("看看 2611.00001 并写笔记", built=Built(model, None, {}), tools=tools, today=TODAY,
                           out=lambda line: None)
    paper = {"arxiv_id": A, "title": "标题", "abstract": "One. Two."}
    row = files.write_turn(result, item_id="S06", papers=lambda arxiv_id: paper)
    assert row == {"run": files.folder.name, "turn": 1, "id": "S06", "follow_up": 0,
                   "request": "看看 2611.00001 并写笔记", "today": TODAY, "status": "answered", "error": None,
                   "steps": 3, "seconds": result.seconds, "notes": [f"{A}-1"], "cited": [A, "2611.00009"],
                   "cited_unseen": ["2611.00009"]}
    index = [json.loads(line) for line in (files.folder.parent / INDEX_FILE).read_text(encoding="utf-8")
             .splitlines()]
    assert index == [row]
    saved = json.loads((files.folder / "turn-001.json").read_text(encoding="utf-8"))
    assert saved["status"] == "answered" and saved["today"] == TODAY
    assert json.loads((files.folder / "turn-001.papers.json").read_text(encoding="utf-8")) == {A: paper}
    summary = (files.folder / "turn-001.md").read_text(encoding="utf-8")
    assert summary.startswith("# Turn 001\n\n- id: S06\n- request: 看看 2611.00001 并写笔记\n")
    assert f"## Answer\n\n笔记已写好:{A},另见 2611.00009。" in summary
    assert f"- {A}-1 ({A}-1.md)" in summary and "- 2611.00009 (in no tool output of this turn)" in summary


def test_a_turn_that_raised_still_gets_its_files_and_index_line(tmp_path):
    files = SessionFiles(new_run_folder(tmp_path / "runs"))
    result = run_live_turn(EN_REQUEST, built=Built(ScriptedModel([OSError("gone")]), None, {}), tools={},
                           today=TODAY, out=lambda line: None)
    row = files.write_turn(result, item_id="S02", follow_up=1)
    assert row["status"] is None and row["error"] == "OSError: gone" and row["steps"] is None
    assert json.loads((files.folder / "turn-001.json").read_text(encoding="utf-8")) == {
        "request": EN_REQUEST, "today": TODAY, "status": None, "error": "OSError: gone"}
    summary = (files.folder / "turn-001.md").read_text(encoding="utf-8")
    assert "- status: error\n- follow-up: 1\n" in summary and "## Error\n\nOSError: gone" in summary
    assert not (files.folder / "turn-001.papers.json").exists()


# ---- progress events --------------------------------------------------------------------------------------


def test_step_events_carry_what_the_lines_print_and_the_very_turn_goes_on_after_two_clock_readings():
    unparsed = call("search_arxiv", "local_1", raw='{"query": ')
    answer = AssistantTurn(content="", tool_calls=[call("get_paper", "local_0", arxiv_id=A), unparsed],
                           model="m", reasoning="先看这篇论文的摘要。",
                           votes={"k": 5, "chosen": "K", "counts": {"K": 4, "L": 1}, "representative": 3,
                                  "errors": [{"candidate": 4, "error_type": "HarnessError", "message": "?"}]},
                           thinking={"reasoning_tokens": 812, "forced": True},
                           rules=[{"rule": "id_repair", "before": {"content": "a"},
                                   "after": {"content": "b"}}])
    lines, events = [], []
    ticks = iter([10.0, 12.5, 20.0, 21.0])  # two readings per request, as without events
    model = ProgressModel(Listening(answer), lines.append, clock=lambda: next(ticks), events=events.append)
    assert model.next_turn([], []) is answer and model.next_turn([], []) is answer
    assert next(ticks, None) is None
    printed = (f"votes 4/5 | 1 candidate(s) failed | thinking 812 tok closed by force | rules id_repair | "
               f"calls get_paper({A}), search_arxiv")
    assert lines == [f"step 1 | {printed} | 2.5 s", f"step 1 attempt 2 | {printed} | 1.0 s"]
    step = {"kind": "step", "step": 1, "attempt": 1, "seconds": 2.5,
            "votes": {"k": 5, "won": 4, "representative": 3, "failed": 1},
            "thinking": {"tokens": 812, "forced": True}, "reasoning": "先看这篇论文的摘要。",
            "rules": [{"rule": "id_repair", "before": {"content": "a"}, "after": {"content": "b"}}],
            "calls": [{"id": "local_0", "name": "get_paper", "arguments": {"arxiv_id": A}},
                      {"id": "local_1", "name": "search_arxiv", "arguments": '{"query": '}],
            "answer_chars": None}
    assert events == [{"kind": "step_start", "step": 1, "attempt": 1}, step,
                      {"kind": "step_start", "step": 1, "attempt": 2}, {**step, "attempt": 2, "seconds": 1.0}]
    assert events[1]["reasoning"] is answer.reasoning
    events[1]["calls"][0]["arguments"]["arxiv_id"] = "changed"  # what the sink does to an event stays there
    events[1]["rules"][0]["after"]["content"] = "changed"
    assert answer.tool_calls[0].arguments == {"arxiv_id": A} and answer.rules[0]["after"] == {"content": "b"}


def test_a_step_event_of_a_vote_nobody_won_a_text_answer_and_a_turn_without_vote_or_thinking():
    voted = AssistantTurn(content="答案", tool_calls=[], model="m",
                          votes={"k": 5, "chosen": None, "counts": {}, "representative": "greedy",
                                 "errors": [{}, {}]})
    assert step_event(2, 3, voted, 1.0) == {
        "kind": "step", "step": 2, "attempt": 3, "seconds": 1.0,
        "votes": {"k": 5, "won": None, "representative": "greedy", "failed": 2}, "thinking": None,
        "reasoning": None, "rules": [], "calls": [], "answer_chars": 2}
    plain = AssistantTurn(content="", tool_calls=[call("list_notes", "l1")], model="m")
    assert step_event(1, 1, plain, 0.25) == {
        "kind": "step", "step": 1, "attempt": 1, "seconds": 0.25, "votes": None, "thinking": None,
        "reasoning": None, "rules": [], "calls": [{"id": "l1", "name": "list_notes", "arguments": {}}],
        "answer_chars": None}
    assert EVENT_KINDS == ("step_start", "step", "step_error", "tool_start", "tool", "tool_error")


def test_a_model_that_raises_gives_a_step_error_event_with_the_seconds_its_line_prints():
    lines, events = [], []
    ticks = iter([3.0, 5.5])
    model = ProgressModel(ScriptedModel([TimeoutError("no answer")]), lines.append, clock=lambda: next(ticks),
                          events=events.append)
    with pytest.raises(TimeoutError):
        model.next_turn([], [])
    assert next(ticks, None) is None
    assert lines == ["step 1: the model raised TimeoutError after 2.5 s"]
    assert events == [{"kind": "step_start", "step": 1, "attempt": 1},
                      {"kind": "step_error", "step": 1, "attempt": 1, "error": "TimeoutError",
                       "seconds": 2.5}]


def test_tool_events_give_the_summary_the_ids_and_the_wait_from_the_readings_the_lines_use():
    hits = [{"arxiv_id": A, "title": "标题"}, {"arxiv_id": B}, {"arxiv_id": A}]
    outcomes = {"search_arxiv": ToolOutcome(hits), "get_paper": ToolOutcome({}, "NOT_FOUND"),
                "write_note": ToolOutcome({"note_id": f"{A}-1", "arxiv_id": A, "path": f"{A}-1.md"})}
    waits = iter([1.0, 4.0, 4.0, 4.0, 4.0, 4.0])  # two readings per call, as without events
    ticks = iter([0.0, 3.5, 4.0, 4.5, 5.0, 6.0])
    lines, events = [], []
    wrapped = progress_tools({name: (lambda args, name=name: outcomes[name]) for name in outcomes},
                             lines.append, waited=lambda: next(waits), clock=lambda: next(ticks),
                             events=events.append)
    for name, outcome in outcomes.items():
        assert wrapped[name]({}) is outcome
    assert next(waits, None) is None and next(ticks, None) is None
    assert lines == ["  tool search_arxiv: 3 hit(s), 3.5 s, 3.0 s of it waiting for the rate limit",
                     "  tool get_paper: error NOT_FOUND, 0.5 s",
                     f"  tool write_note: note {A}-1 ({A}-1.md), 1.0 s"]
    assert events == [
        {"kind": "tool_start", "name": "search_arxiv"},
        {"kind": "tool", "name": "search_arxiv", "summary": "3 hit(s)", "error": None, "ids": [A, B],
         "seconds": 3.5, "waited": 3.0},
        {"kind": "tool_start", "name": "get_paper"},
        {"kind": "tool", "name": "get_paper", "summary": "error NOT_FOUND", "error": "NOT_FOUND", "ids": [],
         "seconds": 0.5, "waited": 0.0},
        {"kind": "tool_start", "name": "write_note"},
        {"kind": "tool", "name": "write_note", "summary": f"note {A}-1 ({A}-1.md)", "error": None, "ids": [A],
         "seconds": 1.0, "waited": 0.0}]
    assert all(event["summary"] == tool_summary(event["name"], outcomes[event["name"]])
               for event in events if event["kind"] == "tool")

    def broken(args: dict) -> ToolOutcome:
        raise RuntimeError("the adapter broke")

    later_waits, later_ticks = iter([4.0]), iter([7.0, 9.0])  # no wait reading after a call that raised
    raising = progress_tools({"get_paper": broken}, lines.append, waited=lambda: next(later_waits),
                             clock=lambda: next(later_ticks), events=events.append)
    with pytest.raises(RuntimeError):
        raising["get_paper"]({})
    assert next(later_waits, None) is None and next(later_ticks, None) is None
    assert lines[-1] == "  tool get_paper: raised RuntimeError, 2.0 s"
    assert events[-2:] == [{"kind": "tool_start", "name": "get_paper"},
                           {"kind": "tool_error", "name": "get_paper", "error": "RuntimeError",
                            "seconds": 2.0}]


def _counted_turn(events) -> tuple[LiveTurn, list[str], dict[str, int]]:
    """A live turn of a call and an answer, with clocks that count their readings."""
    readings = {"clock": 0, "waited": 0}

    def clock() -> float:
        readings["clock"] += 1
        return float(readings["clock"])

    def waited() -> float:
        readings["waited"] += 1
        return 0.0

    model = ScriptedModel([turn(call("get_paper", "g1", arxiv_id=A)), turn(content=f"见 {A}。")])
    lines: list[str] = []
    result = run_live_turn(EN_REQUEST, built=Built(model, None, {}), tools={"get_paper": _found}, today=TODAY,
                           out=lines.append, waited=waited, clock=clock, events=events)
    return result, lines, readings


def test_a_live_turn_sends_the_events_of_both_wrappers_in_order_and_reads_its_clocks_as_often():
    events: list[dict] = []
    assert _counted_turn(events.append) == _counted_turn(None)  # the turn, its lines and the readings
    assert [event["kind"] for event in events] == ["step_start", "step", "tool_start", "tool", "step_start",
                                                   "step"]
    assert events[1]["calls"] == [{"id": "g1", "name": "get_paper", "arguments": {"arxiv_id": A}}]
    assert events[3]["ids"] == [A] and events[5]["answer_chars"] == len(f"见 {A}。")


def test_the_final_system_with_events_sends_the_bytes_it_sends_unwatched_and_shows_the_reasoning(tmp_path,
                                                                                               monkeypatch):
    paths = write_final(tmp_path / "final", monkeypatch)
    frozen = load_frozen(paths["final"], paths["config"])
    messages = [{"role": "system", "content": system_prompt(TODAY)}, {"role": "user", "content": ZH_REQUEST}]
    events: list[dict] = []
    arms = {  # the client over the fake server, and the model's events (None: the model unwrapped)
        "neither": (lambda fake: LocalClient(transport=fake.transport()), None),
        "events": (lambda fake: ProgressClient(transport=fake.transport(), out=quiet), events.append),
    }
    turns, sent = {}, {}
    for arm, (client_of, sink) in arms.items():
        fake = final_server(paths, text_turns(3))
        built = build_system("ft", tokenizer=frozen.tokenizer, family=frozen.family,
                             generation=frozen.settings,
                             count_tokens=family_token_counter(frozen.family, frozen.settings.mode),
                             rules=True, vote_mode=frozen.vote_mode, local_client=client_of(fake))
        model = built.model if sink is None else ProgressModel(built.model, quiet, events=sink)
        turns[arm] = model.next_turn(messages, MODEL_TOOLS)
        sent[arm] = fake.request_bytes
    assert len(sent["neither"]) == 3 and sent["events"] == sent["neither"]
    assert turns["events"] == turns["neither"] and turns["events"].votes == turns["neither"].votes
    start, step = events
    assert start == {"kind": "step_start", "step": 1, "attempt": 1}
    assert step["reasoning"] is turns["events"].reasoning and "Thinking it over." in step["reasoning"]
    assert step["votes"] == {"k": 2, "won": 2, "representative": 0, "failed": 0}
    assert step["thinking"] == {"tokens": 2, "forced": False} and step["rules"] == [] and step["calls"] == []
    assert step["answer_chars"] == len("Here is the answer.")


def test_a_sink_that_raises_ends_the_events_and_nothing_else():
    tried: list[str] = []

    def broken(event: dict) -> None:
        tried.append(event["kind"])
        raise RuntimeError("the page broke")

    def run(events) -> tuple[LiveTurn, list[str], list]:
        model = ScriptedModel([turn(call("get_paper", "g1", arxiv_id=A)),
                               turn(call("get_paper", "g2", arxiv_id=B)), turn(content=f"见 {A}、{B}。")])
        lines: list[str] = []
        result = run_live_turn(EN_REQUEST, built=Built(model, None, {}), tools={"get_paper": _found},
                               today=TODAY, out=lines.append, clock=lambda: 0.0, events=events)
        return result, lines, model.contexts

    result, lines, contexts = run(broken)
    assert tried == ["step_start", "tool_start"]  # one event tried by each wrapper, then none
    assert result.status == "answered" and len(lines) == 5 and (result, lines, contexts) == run(None)
    for failure in (BrokenPipeError(32, "the reader went away"), ValueError("I/O operation on closed file.")):
        def failing(event: dict, failure=failure) -> None:
            raise failure

        assert run(failing) == run(None)

    def interrupted(event: dict) -> None:
        raise KeyboardInterrupt

    model = ProgressModel(Listening(AssistantTurn(content="ok", tool_calls=[], model="m")), quiet,
                          clock=lambda: 0.0, events=interrupted)
    with pytest.raises(KeyboardInterrupt):  # Ctrl+C is no fault of the sink: it goes on up
        model.next_turn([], [])


def test_a_fault_in_making_an_event_ends_that_wrappers_events_and_nothing_else(monkeypatch):
    def run(events) -> tuple[LiveTurn, list[str], list]:
        model = ScriptedModel([turn(call("get_paper", "g1", arxiv_id=A)), turn(content=f"见 {A}。")])
        lines: list[str] = []
        result = run_live_turn(EN_REQUEST, built=Built(model, None, {}), tools={"get_paper": _found},
                               today=TODAY, out=lines.append, clock=lambda: 0.0, events=events)
        return result, lines, model.contexts

    unwatched = run(None)

    def unmakeable(*args: object) -> dict:
        raise RecursionError("maximum recursion depth exceeded")  # as copying a deep argument raises

    for maker, sent in (("step_event", ["step_start", "tool_start", "tool"]),  # the other wrapper's go on
                        ("tool_event", ["step_start", "step", "tool_start", "step_start", "step"])):
        events: list[dict] = []
        with monkeypatch.context() as patched:
            patched.setattr(f"research_agent.live.session.{maker}", unmakeable)
            assert run(events.append) == unwatched, maker  # the turn, its lines and every request
        assert [event["kind"] for event in events] == sent, maker


def test_call_arguments_nested_too_deep_to_copy_end_the_events_and_the_turn_ends_as_it_does_unwatched():
    depth = sys.getrecursionlimit() * 7 // 10  # json.loads reads it, copy.deepcopy (two frames a level) not
    nested = '{"a": ' * depth + "1" + "}" * depth
    deep = call("get_paper", "g1", raw=f'{{"arxiv_id": "{A}", "x": {nested}}}')
    assert deep.arguments is not None

    def run(events) -> tuple:
        model = ScriptedModel([turn(deep), turn(content=f"见 {A}。"), turn(content="spare")])
        lines: list[str] = []
        result = run_live_turn(EN_REQUEST, built=Built(model, None, {}), tools={"get_paper": _found},
                               today=TODAY, out=lines.append, clock=lambda: 0.0, events=events)
        return result.status, result.error, lines, len(model.contexts)  # no deep value compared

    events: list[dict] = []
    assert run(events.append) == run(None)
    assert [event["kind"] for event in events] == ["step_start"]


# ---- a conversation ---------------------------------------------------------------------------------------


def ended(request: str, status: str | None) -> LiveTurn:
    """A live turn of request that ended as status, or raised (None)."""
    transcript = None if status is None else Transcript(request, TODAY, status, None, [], [], "", "", "")
    return LiveTurn(request, TODAY, transcript, None if transcript else "RuntimeError: gone", 0.0)


def test_a_conversation_packs_answers_chains_follow_ups_and_asks_for_a_restatement_after_two():
    asked = [turn(call("ask_clarification", f"q{n}", question="哪个方面?", missing=["topic"]))
             for n in range(3)]
    model = ScriptedModel([*asked, turn(content="好的。")])
    conversation, sent = Conversation(), []

    def say(text: str) -> str | None:
        message, follow_up = conversation.next(text)
        sent.append((message, follow_up))
        return conversation.after(run_live_turn(message, built=Built(model, None, {}), tools={}, today=TODAY,
                                                out=quiet))

    assert say(ZH_REQUEST) == AWAITS_ANSWER and conversation == Conversation(ZH_REQUEST, 0)
    assert say("speculative decoding") == AWAITS_ANSWER
    first = "帮我找几篇论文。补充:speculative decoding"
    assert conversation == Conversation(first, 1)  # the packed request, which the next answer continues
    assert say(" 2026 年\n以后 ") == RESTATE and conversation == Conversation()  # the limit: nothing pending
    assert sent == [(ZH_REQUEST, 0), (first, 1), (first + "补充:2026 年 以后", MAX_FOLLOW_UPS)]
    assert model.contexts[2] == [{"role": "system", "content": system_prompt(TODAY)},
                                 {"role": "user", "content": first + "补充:2026 年 以后"}]
    assert say(EN_REQUEST) is None and sent[-1] == (EN_REQUEST, 0) and conversation == Conversation()
    assert conversation.next("Find papers on RAG.") == ("Find papers on RAG.", 0)  # a request of its own
    assert conversation.after(ended("Find papers on RAG.", "clarification")) == AWAITS_ANSWER
    packed = pack_follow_up("Find papers on RAG.", "since 2025")
    assert conversation.next("since 2025") == (packed, 1) == ("Find papers on RAG. To clarify: since 2025", 1)
    assert conversation.after(ended(packed, "answered")) is None and conversation == Conversation()


def test_dropping_a_request_or_an_empty_answer_leaves_it_and_the_next_text_is_a_request_of_its_own():
    conversation = Conversation()
    assert conversation.next("") is None and conversation.next(" \n") is None  # nothing to send
    assert conversation.next(EN_REQUEST) == (EN_REQUEST, 0)
    assert conversation.after(ended(EN_REQUEST, "clarification")) == AWAITS_ANSWER
    assert conversation.next(" \t\n") is None and conversation == Conversation()  # the request is left
    assert conversation.next("since 2025") == ("since 2025", 0)  # no longer an answer: a request of its own
    assert conversation.after(ended("since 2025", "clarification")) == AWAITS_ANSWER
    conversation.drop()  # as /new does
    assert conversation == Conversation() and conversation.next("Find papers.") == ("Find papers.", 0)
    assert conversation.after(ended("Find papers.", None)) is None  # a turn that raised: nothing pending
    assert conversation == Conversation()

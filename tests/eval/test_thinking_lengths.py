"""scripts/thinking_lengths.py offline (docs/m9-计划.md WP4). The requests of every attempt are rebuilt from a
transcript exactly as the loop sent them (a later step's compressed context, a repair's rejected exchange,
whole-step issues included), and the calls StepLengths counted while the tasks ran are counted again from the
transcripts; the validation agrees with a development run (synthetic, and one scripts/run_e2e.py itself wrote
over a fake family and server) and refuses one with other counts, or measured otherwise, before any session is
read; a synthetic session gives each system's blocks as research_agent.eval.report reads them, with an
injected token counter and forced close, bad input counted, and no task id, item id or request text in the
file written or on the screen."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from support.fakes import HITS, PAPER, ScriptedModel, call, table_check, table_tools
from support.llama_fakes import FakeLlama, Scripted, install_fake_family

from research_agent.agent.context import family_token_counter
from research_agent.agent.executor import ToolExecutor
from research_agent.agent.loop import run_turn, save_transcript
from research_agent.agent.types import AssistantTurn, ToolOutcome
from research_agent.canonical import canonical_json
from research_agent.data.synth import OwnerFunctions
from research_agent.eval import report
from research_agent.eval.e2e import Topic, World
from research_agent.family.core import Mode, load_family
from research_agent.net.fetch import Fetched, ReplayMiss
from research_agent.policy import decide
from research_agent.tools.adapters import fulltext_url, paper_url
from research_agent.tools.corpus_search import CorpusIndex
from research_agent.tools.schemas import MODEL_TOOLS
from research_agent.validation import check_call

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


thinking_lengths = load_script("thinking_lengths")
run_e2e = thinking_lengths.run_e2e  # the module the script measures with

TODAY = "2026-10-10"
TRAINING = 5120  # final's training length, which the validation holds a run to
CLOSE = 3  # the tokens of a forced close, as the injected measure gives them
BASE = 4000  # the injected counter's tokens for the system message and the tool definitions


def counter(messages: list[dict], tools: list[dict]) -> int:
    """The injected prompt counter: BASE plus a token per four characters of every message after the system
    message, so a compressed tool output, a rejected exchange and an older step each count differently."""
    return BASE + sum(len(canonical_json(message)) for message in messages[1:]) // 4


def measure():
    return counter, CLOSE


def thought(*calls, content: str = "", reasoning: int = 200, answer: int | None = 30,
            forced: bool = False) -> AssistantTurn:
    """A local turn in think mode with its thinking record (llm.generation's counts through LocalModel)."""
    turn = AssistantTurn(content=content, tool_calls=list(calls), model="local", paid=False)
    turn.reasoning = "Weigh the request first."
    turn.thinking = {"mode": "think", "reasoning_tokens": reasoning, "answer_tokens": answer,
                     "forced": forced, "phases": 2 if forced else 1, "seeds": None}
    return turn


SEARCH = call("search_arxiv", "local_0", query="lattice sieving", date_from="2026-09-01")
GETS = [call("get_paper", f"local_{i}", arxiv_id=f"2609.0{i}234") for i in range(4)]
BROKEN = call("get_paper", "local_0", raw='{"arxiv_id": 2609.05678')  # not JSON: bad_json
ASK = call("ask_clarification", "local_0", question="Which paper do you mean?")
# The tasks' ids and requests (none may reach the file written or the screen)
REQUESTS = {"task-k7q": "帮我找 2026 年 9 月以来关于 lattice sieving 的论文,挑两篇比较一下。",
            "task-m2x": "What is 2609.05678 about? Give me its main claim in one line.",
            "task-p5z": "帮我把那篇论文整理成一条笔记。",
            "task-r8w": "Look up the four sieving papers for me at once."}


def scripts() -> dict[str, list[AssistantTurn]]:
    """Each task's turns: k7q searches, reads two papers in one step and answers; m2x writes a call that is
    not JSON, is repaired, and answers with its reasoning closed by force and no answer count; p5z asks back;
    r8w makes too many calls (a whole-step issue), is repaired, writes a call that is not JSON and is handed
    off to no model, its step left without an accepted answer. Under counter, k7q's answer would be over the
    training length were its older search output not compressed, and m2x's repair is over only with its
    rejected exchange."""
    return {"task-k7q": [thought(SEARCH, reasoning=300, answer=40),
                         thought(GETS[0], GETS[1], reasoning=880, answer=70),
                         thought(content="两篇都改进了筛法。", reasoning=680, answer=60)],
            "task-m2x": [thought(BROKEN, reasoning=240, answer=20),
                         thought(GETS[1], reasoning=1040, answer=25),
                         thought(content="It verifies drafted tokens in a tree.", reasoning=1280, answer=None,
                                 forced=True)],
            "task-p5z": [thought(ASK, reasoning=100, answer=15)],
            "task-r8w": [thought(*GETS, reasoning=400, answer=10), thought(BROKEN, reasoning=520, answer=12)]}


def run_task(task_id: str, turns: list[AssistantTurn]) -> dict:
    """One task through the loop as scripts/run_e2e.py runs it, StepLengths around the scripted model: the
    transcript, the requests the model was sent, StepLengths' counts and every call's length."""
    model = ScriptedModel(turns, name="local")
    lengths = run_e2e.StepLengths(model, count_tokens=counter, training_length=TRAINING, close_tokens=CLOSE)
    tools, _ = table_tools({"search_arxiv": [ToolOutcome(HITS)], "get_paper": [ToolOutcome(PAPER)]})
    transcript = run_turn(REQUESTS[task_id], model=lengths, executor=ToolExecutor(tools, table_check(),
                                                                                 today=TODAY),
                          decide=decide, today=TODAY)
    assert len(model.contexts) == len(turns)  # every scripted turn was asked for
    return {"transcript": transcript, "requests": model.contexts, "calls": lengths.steps,
            "over": lengths.over,
            "lengths": [counter(request, MODEL_TOOLS) + run_e2e.generated_tokens(turn, CLOSE)
                        for request, turn in zip(model.contexts, turns, strict=True)]}


def run_all() -> dict[str, dict]:
    return {task_id: run_task(task_id, turns) for task_id, turns in scripts().items()}


def saved(transcript) -> dict:
    """A transcript as save_transcript writes it and a reader reads it back."""
    return json.loads(transcript.to_json())


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")


def dev_run(root: Path, ran: dict[str, dict], **manifest_changes) -> Path:
    """A run folder of scripts/run_e2e.py for final over the tasks ran, with one task whose run raised (no
    transcript, no local call): its report (the manifest counting what StepLengths counted while the tasks
    ran) and its transcripts."""
    folder = root / "dev-run"
    records = []
    for task_id, task in ran.items():
        name = f"transcripts/{task_id}.json"
        save_transcript(task["transcript"], folder / name)
        records.append({"id": task_id, "status": task["transcript"].status, "transcript": name,
                        "local_calls": task["transcript"].local_calls})
    records.append({"id": "task-lost0", "status": "harness_error", "transcript": None, "local_calls": 0})
    manifest = {"system": "final", "context_check": thinking_lengths.TOKENIZER, "training_length": TRAINING,
                "steps_over_training_length": sum(task["over"] for task in ran.values()),
                "generation": {"mode": "think", "reasoning_budget": 1280}, **manifest_changes}
    write_json(folder / "report.json", {"manifest": manifest, "summary": {}, "tasks": records})
    return folder


def thinking(reasoning: int, answer: int | None, forced: bool) -> dict:
    return {"mode": "think", "reasoning_tokens": reasoning, "answer_tokens": answer, "forced": forced,
            "phases": 2 if forced else 1, "seeds": [11, 12] if forced else [11]}


def record(item_id: str, found: dict | None, *, status: str = "ok") -> dict:
    """A runner record (eval.runner) of a local system in think mode, with its item's fields and its turn."""
    out = {"id": item_id, "status": status, "item": {"type": "call", "lang": "zh", "split": "test",
                                                     "slots": {"topic": "lattice sieving 的近期进展"}}}
    if status == "ok":
        out.update(output={"role": "assistant", "content": "这是关于筛法的回答。"}, usage={}, paid=False,
                   cached=False, model="local", reasoning="Weigh the item first.", thinking=found)
    else:
        out.update(error="the server went away", error_type="ReadTimeout")
    return out


FINAL_RECORDS = [record("item-a1", thinking(312, 41, False)), record("item-a2", thinking(1280, 97, True)),
                 record("item-a3", thinking(640, None, False)), record("item-a4", None, status="error")]
FINALG_RECORDS = [record("item-b1", thinking(500, 30, False)), record("item-b2", thinking(700, 50, False))]
ITEM_IDS = [r["id"] for r in FINAL_RECORDS + FINALG_RECORDS]


def timed(step: int | None = 1, *, error: str | None = None, server: dict | None = None) -> dict:
    """One call of a timing sidecar (eval.timing.TaskTimer)."""
    return {"step": step, "wall_ms": 12.5, "error": error, "paid": None if error else False,
            "cached": None if error else False, "server": server}


def counted(prompt_n: int, predicted_n: int) -> dict:
    return timed(server={"prompt_n": prompt_n, "prompt_ms": 50.0, "predicted_n": predicted_n,
                         "predicted_ms": 900.0})


# FT2's calls as llama.cpp counted them: 3,120, 4,200 (over 4,096), 4,096 (not longer) and 2,096 tokens, and a
# call that raised
FT2_TIMING = {"tasks": [{"id": "task-k7q", "status": "answered", "wall_ms": 99.0, "steps": [],
                         "calls": [counted(3000, 120), counted(3900, 300), counted(4000, 96),
                                   timed(2, error="ReadTimeout")]},
                        {"id": "task-m2x", "status": "answered", "wall_ms": 50.0, "steps": [],
                         "calls": [counted(2000, 96)]}]}


def session_folder(root: Path, ran: dict[str, dict]) -> Path:
    """A session folder as scripts/run_test.py writes it, for final, finalg and ft2: the single-step records,
    final's end-to-end transcripts with their timing sidecar (one call per attempt and one that raised) and
    FT2's sidecar."""
    folder = root / "session"
    write_lines(folder / "final.jsonl", [canonical_json(r) for r in FINAL_RECORDS])
    write_lines(folder / "finalg.jsonl", [canonical_json(r) for r in FINALG_RECORDS])
    tasks = []
    for task_id, task in ran.items():
        save_transcript(task["transcript"], folder / "e2e-final" / "transcripts" / f"{task_id}.json")
        tasks.append({"id": task_id, "status": task["transcript"].status, "wall_ms": 80.0, "steps": [],
                      "calls": [timed() for _ in range(task["transcript"].local_calls)]})
    tasks[0]["calls"].append(timed(3, error="ReadTimeout"))
    write_json(folder / "e2e-final" / "timing.json", {"tasks": tasks})
    write_json(folder / "e2e-ft2" / "timing.json", FT2_TIMING)
    return folder


def digest(folder: Path) -> dict:
    """The combined SHA-256 of every file in folder (module docstring of the script)."""
    files = sorted((path.relative_to(folder).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
                   for path in folder.rglob("*") if path.is_file())
    lines = "".join(f"{name}\t{sha}\n" for name, sha in files)
    return {"files": len(files), "sha256": hashlib.sha256(lines.encode("utf-8")).hexdigest()}


def rounded(block: dict) -> dict:
    return {key: round(value, 6) if isinstance(value, float) else value for key, value in block.items()}


def hidden(*texts: str) -> list[str]:
    """What must never be shown: every task and item id and every request."""
    return [*REQUESTS, *REQUESTS.values(), "task-lost0", *ITEM_IDS, *texts]


# ---- the requests rebuilt --------------------------------------------------------------------------------


def test_every_attempts_request_is_rebuilt_as_the_loop_sent_it():
    ran = run_all()
    statuses = {task_id: task["transcript"].status for task_id, task in ran.items()}
    assert statuses == {"task-k7q": "answered", "task-m2x": "answered", "task-p5z": "clarification",
                        "task-r8w": "invalid_output"}
    for task_id, task in ran.items():
        transcript = saved(task["transcript"])
        rebuilt = thinking_lengths.attempt_requests(transcript)
        assert [request for request, _ in rebuilt] == task["requests"], task_id
        attempts = [attempt for step in transcript["steps"] for attempt in step["attempts"]]
        assert [attempt for _, attempt in rebuilt] == attempts
    # a later step's context compresses the older tool output; a repair carries the rejected exchange, whose
    # whole-step issue (too many calls) answers every call
    later = ran["task-k7q"]["requests"][2]
    assert later[3]["role"] == "tool" and "snippet" not in later[3]["content"]
    repair, refused = ran["task-m2x"]["requests"][1], ran["task-r8w"]["requests"][1]
    assert repair[2]["role"] == "assistant" and json.loads(repair[3]["content"])["error"] == "INVALID_JSON"
    assert len(refused) == 2 + 1 + 4 and all(json.loads(m["content"])["error"] == "INVALID_ARGUMENTS"
                                             for m in refused[3:])


def test_the_calls_step_lengths_counted_while_the_tasks_ran_are_counted_again_from_the_transcripts():
    ran = run_all()
    lengths = thinking_lengths.Lengths(counter, CLOSE, TRAINING)
    for task in ran.values():
        for request, attempt in thinking_lengths.attempt_requests(saved(task["transcript"])):
            lengths.add(request, attempt["thinking"])
    every = [length for task in ran.values() for length in task["lengths"]]
    assert lengths.block() == {"training_length": TRAINING, "calls": sum(t["calls"] for t in ran.values()),
                               "over": sum(t["over"] for t in ran.values()), "max": max(every)}
    assert 0 < lengths.over < lengths.calls == len(every) == 9  # the fixture has calls on both sides
    # a step exactly as long as the training length is not over it, one token more is (StepLengths)
    exact = thinking_lengths.Lengths(lambda messages, tools: TRAINING - 10, CLOSE, TRAINING)
    exact.add([], thinking(7, 0, True))
    exact.add([], thinking(8, 0, True))
    assert exact.block() == {"training_length": TRAINING, "calls": 2, "over": 1, "max": TRAINING + 1}


def broken(transcript: dict, change) -> dict:
    data = json.loads(json.dumps(transcript))
    change(data)
    return data


@pytest.mark.parametrize("change", [
    lambda t: t["messages"].pop(3),  # a reply of an accepted step is missing
    lambda t: t["messages"].append({"role": "user", "content": "more"}),  # a message no step accounts for
    lambda t: t["steps"][0]["attempts"].append(dict(t["steps"][0]["attempts"][0])),  # after an accepted one
    lambda t: t["steps"][0]["executed"][0].update(call_id="local_9"),  # a reply to another call
    lambda t: t["steps"].append({"index": 9}),  # a step without its attempts
    lambda t: t.update(messages=[t["messages"][0]]),  # no request
], ids=["missing-reply", "extra-message", "attempt-after-accepted", "other-call", "no-attempts",
        "no-request"])
def test_a_transcript_whose_steps_do_not_fit_its_messages_is_malformed(change):
    transcript = saved(run_task("task-k7q", scripts()["task-k7q"])["transcript"])
    thinking_lengths.attempt_requests(transcript)  # as written, it fits
    with pytest.raises(thinking_lengths.Malformed):
        thinking_lengths.attempt_requests(broken(transcript, change))


def test_a_repair_whose_rejected_attempt_lost_its_issues_is_malformed():
    transcript = saved(run_task("task-m2x", scripts()["task-m2x"])["transcript"])

    def rejected(t: dict) -> dict:
        return t["steps"][0]["attempts"][0]

    with pytest.raises(thinking_lengths.Malformed):
        thinking_lengths.attempt_requests(broken(transcript, lambda t: rejected(t).update(issues=None)))
    with pytest.raises(thinking_lengths.Malformed):  # an issue's messages must be a list
        thinking_lengths.attempt_requests(broken(transcript, lambda t: rejected(t)["issues"][0].update(
            messages="arguments are not valid JSON")))


def test_a_thinking_block_takes_the_percentiles_over_the_turns_and_leaves_out_a_null_answer():
    block = thinking_lengths.thinking_block([thinking(312, 41, False), thinking(1280, 97, True),
                                             thinking(640, None, False)])
    assert rounded(block) == {"n": 3, "reasoning_p50": 640.0, "reasoning_p95": 1216.0, "reasoning_max": 1280,
                              "answer_p50": 69.0, "answer_p95": 94.2, "forced": 1}
    assert list(block) == ["n", *report.THINKING_TOKENS, "forced"]  # as the report reads a part's block
    assert thinking_lengths.thinking_block([]) == {"n": 0, "reasoning_p50": None, "reasoning_p95": None,
                                                   "reasoning_max": None, "answer_p50": None,
                                                   "answer_p95": None, "forced": 0}


# ---- the validation --------------------------------------------------------------------------------------


def test_a_development_run_whose_counts_the_rebuild_reproduces_agrees(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    ran = run_all()
    dev_run(tmp_path, ran)
    assert thinking_lengths.main(["--validate", "dev-run"], measure=measure) == 0
    out = capsys.readouterr().out
    over = sum(task["over"] for task in ran.values())
    assert "validation on dev-run: 5 task(s), 1 without a transcript; 5 file(s) read" in out
    assert "local calls: 9 measured, the task records count 9; 0 task(s) whose calls differ" in out
    assert f"calls over 5120 tokens: {over} measured, the manifest counts {over}" in out
    assert "problems: none" in out and out.rstrip().endswith("OK: the rebuild reproduces the run's counts")
    assert not any(text in out for text in hidden())
    assert not (tmp_path / "results").exists()  # --validate writes nothing


def shift_a_call(report: dict) -> None:
    """One local call moved from the first task's record to the second's: the sums still agree."""
    first, second = report["tasks"][:2]
    first["local_calls"], second["local_calls"] = first["local_calls"] - 1, second["local_calls"] + 1


@pytest.mark.parametrize(("tamper", "shown"), [
    (lambda report: report["manifest"].update(steps_over_training_length=4),
     "calls over 5120 tokens: 3 measured, the manifest counts 4"),
    (lambda report: report["tasks"][0].update(local_calls=report["tasks"][0]["local_calls"] + 1),
     "local calls: 9 measured, the task records count 10; 1 task(s) whose calls differ"),
    (shift_a_call, "local calls: 9 measured, the task records count 9; 2 task(s) whose calls differ"),
], ids=["over", "local-calls", "shifted-call"])
def test_a_development_run_with_other_counts_does_not_agree_and_no_session_is_read(tmp_path, monkeypatch,
                                                                                  capsys, tamper, shown):
    monkeypatch.chdir(tmp_path)
    folder = dev_run(tmp_path, run_all())
    data = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    tamper(data)
    write_json(folder / "report.json", data)
    assert thinking_lengths.main(["--validate", "dev-run"], measure=measure) == 1
    out = capsys.readouterr().out
    assert shown in out and "FAIL: the rebuild does not reproduce the run's counts" in out
    # with --dev the session is not read (it is not there: reading it would exit 2) and nothing is written
    assert thinking_lengths.main(["--dev", "dev-run", "--session", "absent"], measure=measure) == 1
    assert not (tmp_path / "results").exists()


def test_a_development_run_with_a_transcript_it_cannot_measure_does_not_agree(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    folder = dev_run(tmp_path, run_all())
    (folder / "transcripts" / "task-p5z.json").write_text("{", encoding="utf-8")
    assert thinking_lengths.main(["--validate", "dev-run"], measure=measure) == 1
    out = capsys.readouterr().out
    assert "problems: unreadable 1" in out
    assert "local calls: 8 measured, the task records count 9; 1 task(s) whose calls differ" in out
    assert not any(text in out for text in hidden())


@pytest.mark.parametrize(("changes", "named"), [
    ({"training_length": 4096}, "training_length 4096, not 5120"),
    ({"training_length": 5120.0}, "training_length 5120.0, not 5120"),
    ({"context_check": "models/Qwen3-4B"}, "the context check on 'models/Qwen3-4B'"),
    ({"context_check": None, "steps_over_training_length": None}, "steps_over_training_length None"),
    ({"generation": {"mode": "no_think"}}, "mode 'no_think', not think"),
], ids=["ft2-length", "float-length", "other-family", "no-context-check", "no-think"])
def test_a_development_run_not_measured_as_the_session_is_refused_before_anything_is_measured(
        tmp_path, monkeypatch, capsys, changes, named):
    monkeypatch.chdir(tmp_path)
    dev_run(tmp_path, run_all(), **changes)

    def never():
        raise AssertionError("the family is loaded only to measure")

    assert thinking_lengths.main(["--validate", "dev-run"], measure=never) == 2
    err = capsys.readouterr().err
    assert "dev-run/report.json: the run was not measured as the session is" in err and named in err


def test_folders_outside_the_current_folder_are_refused_before_anything_is_read(tmp_path, monkeypatch,
                                                                               capsys):
    ran = run_all()
    dev_run(tmp_path, ran)
    session_folder(tmp_path, ran)
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")  # the run and the session lie beside it, not inside it

    def never():
        raise AssertionError("nothing is measured")

    for argv in (["--validate", "../dev-run"], ["--dev", "../dev-run", "--session", "../session"],
                 ["--dev", str(tmp_path / "dev-run"), "--session", str(tmp_path / "session")]):
        assert thinking_lengths.main(argv, measure=never) == 2
        assert "lie(s) outside the current folder" in capsys.readouterr().err
    assert list((tmp_path / "work").iterdir()) == []  # nothing written
    monkeypatch.chdir(tmp_path)  # from the folder that holds both, the same run agrees
    assert thinking_lengths.main(["--validate", str(tmp_path / "dev-run")], measure=measure) == 0
    assert "validation on dev-run: " in capsys.readouterr().out


def test_an_exception_while_the_session_is_read_is_named_by_its_type_alone(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    ran = run_all()
    dev_run(tmp_path, ran)
    session_folder(tmp_path, ran)

    def quoting(*args, **kwargs):
        raise KeyError(REQUESTS["task-k7q"])  # a message that quotes what was read

    monkeypatch.setattr(thinking_lengths, "summarize_session", quoting)
    assert thinking_lengths.main(["--dev", "dev-run", "--session", "session"], measure=measure) == 2
    err = capsys.readouterr().err
    assert "FAIL: KeyError stopped the reading of session; its message is withheld" in err
    assert not any(secret in err for secret in hidden()) and not (tmp_path / "results").exists()
    monkeypatch.setattr(thinking_lengths, "validate", quoting)
    assert thinking_lengths.main(["--validate", "dev-run"], measure=measure) == 2
    err = capsys.readouterr().err
    assert "FAIL: KeyError stopped the reading of dev-run" in err and not any(s in err for s in hidden())


def test_a_missing_report_or_one_of_another_kind_is_refused(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert thinking_lengths.main(["--validate", "nowhere"], measure=measure) == 2
    assert "nowhere/report.json: missing or unreadable" in capsys.readouterr().err
    write_json(tmp_path / "other" / "report.json", {"manifest": {}, "tasks": [{"transcript": "x.json"}]})
    assert thinking_lengths.main(["--validate", "other"], measure=measure) == 2
    assert "not a report of scripts/run_e2e.py" in capsys.readouterr().err


# a run of scripts/run_e2e.py itself (as tests/eval/test_run_e2e_cli.py runs it): one lookup task over a
# replayed paper; the fake server's first answer is a call that is not JSON, repaired by the second, and the
# second step's reasoning is cut by the budget and closed by force before its answer
ID2 = "2609.00002"
CORPUS = [{"arxiv_id": ID2, "title": "Tree Verification at Scale", "authors": ["A. Author"],
           "published": "2026-09-20", "categories": ["cs.CL"],
           "abstract": "Tree-structured verification checks drafted tokens at once."}]
LOOKUP = {"id": "task-w4n", "intent": "lookup_paper", "scenario": "normal", "lang": "en", "split": "dev",
          "family": "e2e/lookup_paper", "text": f"What is {ID2} about?", "slots": {"arxiv_id": ID2},
          "expect": {"status": "answered", "must_write_note": False, "note_about": None,
                     "must_cite_ids_within": "tool_outputs", "expected_tools_any": ["get_paper"]}}
NOT_JSON = '<tool_call>\n{"name": "get_paper", "arguments": {arxiv_id: ' + ID2 + '}}\n</tool_call>'
GET_PAPER = '<tool_call>\n{"name": "get_paper", "arguments": {"arxiv_id": "' + ID2 + '"}}\n</tool_call>'


class Pages:
    """The replay fetcher: the paper's metadata as a JSON feed and its full-text page."""

    def get(self, url: str) -> Fetched:
        for paper in CORPUS:
            if url == paper_url(paper["arxiv_id"]):
                return Fetched(url=url, final_url=url, status=200, body=json.dumps([paper]).encode(),
                               content_type="")
            if url == fulltext_url(paper["arxiv_id"]):
                return Fetched(url=url, final_url=url, status=200, body=b"<html><p>Method.</p></html>",
                               content_type="")
        raise ReplayMiss(f"no cassette for {url}")


def parse_feed(xml: str) -> list[dict] | None:
    try:
        return json.loads(xml)
    except json.JSONDecodeError:
        return None


def world() -> World:
    owner = OwnerFunctions(check_call, decide, parse_feed, lambda html, section, n: "Method.")
    return World(owner, Pages(), CorpusIndex(CORPUS), None,
                 {"tree-verification": Topic(frozenset({ID2}), ("tree verification",))})


def written_run(root: Path, name: str, training_length: int) -> Path:
    """run_e2e's run folder of LOOKUP in think mode with the training length given (the fake family must be
    installed at root/model)."""
    tasks = root / "tasks.jsonl"
    tasks.write_text(json.dumps(LOOKUP) + "\n", encoding="utf-8")
    fake = FakeLlama([Scripted(("Check.", "\n</think>\n\n", NOT_JSON)),
                      Scripted(("Again.", "\n</think>\n\n", GET_PAPER)),
                      Scripted(("endless",), "length"), Scripted((f"{ID2} is about tree verification.",))])
    out = root / name
    code = run_e2e.main([str(tasks), "--system", "b0", "--out", str(out), "--tokenizer", str(root / "model"),
                         "--mode", "think", "--reasoning-budget", "8", "--training-length",
                         str(training_length), "--today", TODAY], world=world(), local_client=fake.client())
    assert code == 0
    return out


def test_the_validation_agrees_with_a_run_scripts_run_e2e_wrote_itself(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # the guards meet no real set: the root is tmp_path
    monkeypatch.setattr(run_e2e, "REPO", tmp_path)
    install_fake_family(monkeypatch, tmp_path / "model")
    family = load_family(tmp_path / "model")
    measured = (family_token_counter(family, Mode.THINK), run_e2e.close_tokens(family))
    model = str(tmp_path / "model")

    def check(name: str, training_length: int) -> tuple[dict, dict]:
        out = written_run(tmp_path, name, training_length)
        report_data = json.loads((out / "report.json").read_text(encoding="utf-8"))
        validation = thinking_lengths.validate(out, lambda: measured, tokenizer=model,
                                               training_length=training_length)
        return report_data, validation

    first, validation = check("default", 4096)
    assert first["tasks"][0]["local_calls"] == validation["calls"] == 3  # the rejected attempt included
    assert thinking_lengths.agrees(validation) and validation["over"] == 0
    path = tmp_path / "default" / "transcripts" / "task-w4n.json"
    transcript = json.loads(path.read_text(encoding="utf-8"))
    steps = transcript["steps"]
    assert steps[0]["attempts"][0]["action"] == "repair" and steps[1]["attempts"][0]["thinking"]["forced"]
    count_tokens, close_tokens = measured
    lengths, forced = [], []
    for request, attempt in thinking_lengths.attempt_requests(transcript):
        turn = AssistantTurn(content="", tool_calls=[], model="m", thinking=attempt["thinking"])
        lengths.append(count_tokens(request, MODEL_TOOLS) + run_e2e.generated_tokens(turn, close_tokens))
        forced.append(attempt["thinking"]["forced"])
    assert len(set(lengths)) == 3 and forced == [False, False, True] and close_tokens > 0
    # held to a length between them, below all of them, and one token below the step closed by force (over
    # only with its close counted): run_e2e's count is reproduced each time
    middle, short, closed = sorted(lengths)[1], min(lengths) - 1, lengths[2] - 1
    for name, training_length in (("middle", middle), ("short", short), ("closed", closed)):
        written, validation = check(name, training_length)
        over = sum(length > training_length for length in lengths)
        assert written["manifest"]["steps_over_training_length"] == validation["over"] == over
        assert thinking_lengths.agrees(validation)
    assert sum(length > middle for length in lengths) == 1 and sum(length > short for length in lengths) == 3


# ---- the session -----------------------------------------------------------------------------------------


def test_the_session_summary_gives_each_systems_blocks_and_shows_no_task_or_request(tmp_path, monkeypatch,
                                                                                    capsys):
    monkeypatch.chdir(tmp_path)
    ran = run_all()
    dev_run(tmp_path, ran)
    session = session_folder(tmp_path, ran)
    code = thinking_lengths.main(["--dev", "dev-run", "--session", "session", "--out", "out/lengths-m7.json"],
                                 measure=measure)
    assert code == 0
    out = capsys.readouterr().out
    text = (tmp_path / "out" / "lengths-m7.json").read_text(encoding="utf-8")
    data = json.loads(text)
    over = sum(task["over"] for task in ran.values())
    longest = max(length for task in ran.values() for length in task["lengths"])
    final = data["systems"]["final"]
    assert rounded(final["single_step"]) == {"n": 3, "reasoning_p50": 640.0, "reasoning_p95": 1216.0,
                                             "reasoning_max": 1280, "answer_p50": 69.0, "answer_p95": 94.2,
                                             "forced": 1}
    # every attempt: reasoning 100 240 300 400 520 680 880 1040 1280, answers 10 12 15 20 25 40 60 70 (one
    # null), one closed by force; three calls over 5,120 tokens
    assert rounded(final["e2e"]) == {"n": 9, "reasoning_p50": 520.0, "reasoning_p95": 1184.0,
                                     "reasoning_max": 1280, "answer_p50": 22.5, "answer_p95": 66.5,
                                     "forced": 1}
    assert final["e2e_length"] == {"training_length": 5120, "calls": 9, "over": over, "max": longest}
    assert over == 3
    assert data["systems"]["finalg"] == {"single_step": {
        "n": 2, "reasoning_p50": 600.0, "reasoning_p95": 690.0, "reasoning_max": 700, "answer_p50": 40.0,
        "answer_p95": 49.0, "forced": 0}}
    assert data["systems"]["ft2"] == {"e2e_length": {"training_length": 4096, "calls": 4, "over": 1,
                                                     "max": 4200}}
    assert data["read"] == {"final": {"records": 4, "transcripts": 4, "attempts": 9, "timed_calls": 9},
                            "finalg": {"records": 2}, "ft2": {"tasks": 2, "timed_calls": 4}}
    assert data["problems"] == {} and data["milestone"] == "M7"
    assert data["session"] == {"path": "session", **digest(session)}
    assert data["validation"] == {"path": "dev-run", **digest(tmp_path / "dev-run"), "training_length": 5120,
                                  "tasks": 5, "tasks_without_transcript": 1, "tasks_whose_calls_differ": 0,
                                  "local_calls": 9, "calls": 9, "steps_over_training_length": over,
                                  "over": over, "problems": {}}
    assert set(data["e2e_length_counted_by"]) == {"final", "ft2"}
    # the report reads the file as it is written
    numbers = report.lengths_numbers(data, "m7")
    assert numbers["lengths.m7.final.e2e_length.over"] == str(over)
    assert numbers["lengths.m7.final.e2e.forced_rate"] == "11.1%"
    assert numbers["lengths.m7.finalg.single_step.reasoning_p95"] == "690"
    assert numbers["lengths.m7.ft2.e2e_length.max"] == "4200"
    assert {key.split(".")[2] for key in numbers} == {"final", "finalg", "ft2"}
    assert "  final  e2e_length  9 call(s), " in out and "  ft2    e2e_length  4 call(s), 1 over 4096" in out
    assert ("  read: final records 4, transcripts 4, attempts 9, timed_calls 9; finalg records 2; ft2 "
            "tasks 2, timed_calls 4") in out
    assert "out/lengths-m7.json: written" in out
    for shown in (text, out):
        assert not any(secret in shown for secret in hidden("Weigh the", "这是关于筛法", "lattice sieving"))


def test_bad_input_is_counted_and_never_shown(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    ran = run_all()
    dev_run(tmp_path, ran)
    session = session_folder(tmp_path, ran)
    with (session / "final.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write("not a record\n" + canonical_json(record("item-a5", {"reasoning_tokens": "many",
                                                                     "forced": False})) + "\n")
    transcripts = session / "e2e-final" / "transcripts"
    (transcripts / "task-x1q.json").write_text("{", encoding="utf-8")  # unreadable
    lost_reply = saved(ran["task-k7q"]["transcript"])
    lost_reply["messages"].pop(3)
    write_json(transcripts / "task-x2q.json", lost_reply)  # malformed: 3 attempts
    plain = saved(ran["task-p5z"]["transcript"])
    del plain["steps"][0]["attempts"][0]["thinking"]
    write_json(transcripts / "task-x3q.json", plain)  # an attempt that cannot be measured
    timing = json.loads((session / "e2e-final" / "timing.json").read_text(encoding="utf-8"))
    timing["tasks"] += [{"id": task_id, "status": "answered", "wall_ms": 1.0, "steps": [],
                         "calls": [timed() for _ in range(calls)]}
                        for task_id, calls in (("task-x1q", 2), ("task-x2q", 3), ("task-x3q", 1))]
    write_json(session / "e2e-final" / "timing.json", timing)
    ft2 = json.loads(json.dumps(FT2_TIMING))
    ft2["tasks"][1]["calls"].append(timed(server=None))  # came back without llama.cpp's counts
    write_json(session / "e2e-ft2" / "timing.json", ft2)
    assert thinking_lengths.main(["--dev", "dev-run", "--session", "session"], measure=measure) == 0
    out = capsys.readouterr().out
    data = json.loads((tmp_path / "results" / "lengths-m7.json").read_text(encoding="utf-8"))
    assert data["problems"] == {"final.e2e.malformed": 1, "final.e2e.unreadable": 1,
                                "final.e2e_length.calls_not_measured": 6, "final.e2e_length.unmeasured": 1,
                                "final.single_step.malformed": 1, "final.single_step.unreadable": 1,
                                "ft2.e2e_length.unmeasured": 1}
    assert data["read"]["final"] == {"records": 5, "transcripts": 5, "attempts": 10, "timed_calls": 15}
    assert data["systems"]["final"]["e2e"]["n"] == 9 and data["systems"]["final"]["e2e_length"]["calls"] == 9
    assert data["systems"]["ft2"]["e2e_length"]["calls"] == 4
    assert ("problems: final.e2e.malformed 1, final.e2e.unreadable 1, final.e2e_length.calls_not_measured 6"
            in out)
    text = (tmp_path / "results" / "lengths-m7.json").read_text(encoding="utf-8")
    for shown in (text, out):
        assert not any(secret in shown for secret in hidden("task-x1q", "task-x2q", "task-x3q", "item-a5"))


@pytest.mark.parametrize(("remove", "named"), [
    ("finalg.jsonl", "session/finalg.jsonl: missing or unreadable (single-step records)"),
    ("e2e-final/transcripts", "session/e2e-final/transcripts: no transcripts"),
    ("e2e-final/timing.json", "session/e2e-final/timing.json: missing or unreadable"),
    ("e2e-ft2/timing.json", "session/e2e-ft2/timing.json: missing or unreadable"),
])
def test_a_session_without_one_of_its_inputs_is_refused_and_nothing_is_written(tmp_path, monkeypatch, capsys,
                                                                              remove, named):
    monkeypatch.chdir(tmp_path)
    ran = run_all()
    dev_run(tmp_path, ran)
    target = session_folder(tmp_path, ran) / remove
    if target.is_dir():
        for path in target.iterdir():
            path.unlink()
    else:
        target.unlink()
    assert thinking_lengths.main(["--dev", "dev-run", "--session", "session"], measure=measure) == 2
    err = capsys.readouterr().err
    assert named in err and not any(secret in err for secret in hidden())
    assert not (tmp_path / "results").exists()


def test_an_existing_summary_is_kept_unless_force_is_given(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    ran = run_all()
    dev_run(tmp_path, ran)
    session_folder(tmp_path, ran)
    out = tmp_path / "results" / "lengths-m7.json"
    write_json(out, {"kept": True})
    assert thinking_lengths.main(["--dev", "dev-run", "--session", "session"], measure=measure) == 3
    assert "results/lengths-m7.json exists; --force overwrites it" in capsys.readouterr().err
    assert json.loads(out.read_text(encoding="utf-8")) == {"kept": True}
    forced = ["--dev", "dev-run", "--session", "session", "--force"]
    assert thinking_lengths.main(forced, measure=measure) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["milestone"] == "M7"


@pytest.mark.parametrize("argv", [[], ["--validate", "a", "--dev", "b"],
                                  ["--validate", "a", "--session", "s"],
                                  ["--validate", "a", "--out", "o.json"], ["--validate", "a", "--force"]])
def test_one_of_validate_and_dev_is_required_and_validate_reads_no_session(argv, capsys):
    with pytest.raises(SystemExit) as stopped:
        thinking_lengths.main(argv, measure=measure)
    assert stopped.value.code == 2


def test_the_summary_is_a_registered_lengths_result_and_the_plan_follows_the_session_configuration():
    assert report.kind_of(thinking_lengths.OUT.name).key == "lengths"
    systems = thinking_lengths.SYSTEMS
    assert (systems["final"].rebuilt, systems["ft2"].server_counted) == (5120, 4096)  # FT3's and FT2's
    assert systems["finalg"].rebuilt is None and systems["finalg"].thinking == ("single_step",)


def test_the_family_and_systems_measured_are_the_m7_sessions():  # the configuration is in the public copy too
    config = json.loads((REPO / "results" / "test-config-m7.json").read_text(encoding="utf-8"))
    final = config["adapters"]["final"]
    assert final["tokenizer"] == thinking_lengths.TOKENIZER and final["generation"]["mode"] == "think"
    assert set(thinking_lengths.SYSTEMS) <= set(config["systems"])
    assert config["systems"]["finalg"]["end_to_end"] is False  # finalg has no end-to-end part
    assert "generation" not in config["adapters"]["ft2"]  # FT2 ran on the legacy path: llama.cpp counts it

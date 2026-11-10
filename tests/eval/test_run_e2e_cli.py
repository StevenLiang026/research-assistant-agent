"""scripts/run_e2e.py offline, with its own small world: the M3 guards (a test set is refused, a locked set is
lock-checked), b4 and the adapter in the manifest, the timing sidecar built on run_tasks' on_task hook, ftr
routed step by step by a frozen configuration (--routing-config, M7) with its routed steps in the records, and
the M7 inference wrappers (--votes, --rules, --glossary), a vote's failed candidates counted in the report.
Kept apart from tests/eval/test_e2e.py, which also checks the real task files."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import pytest
from support.fakes import ScriptedModel, call, turn
from support.family_fakes import MARK, DummyPrefix, FakeFamilyTokenizer, fake_family, model_folder
from support.llama_fakes import FakeLlama, Scripted, install_fake_family
from train.tokenizer_fake import FakeTokenizer

from research_agent.agent.route import RouteConfig, RoutedModel, RouteRules
from research_agent.canonical import canonical_json
from research_agent.data.synth import OwnerFunctions
from research_agent.eval.e2e import Topic, World, run_tasks
from research_agent.eval.lock import LockedSet, RefusedTestSet, sha256_of, write_lock
from research_agent.family.core import GREEDY, Decoding, Mode, load_family
from research_agent.family.toolformats.qwen_json import QwenJson
from research_agent.llm.generation import request_seed
from research_agent.llm.local import Completion
from research_agent.net.faults import DELAY_S, FAULTS, applicable, draw
from research_agent.net.fetch import Fetched, ReplayMiss, canonical_url
from research_agent.policy import decide
from research_agent.tools.adapters import fulltext_url, paper_url
from research_agent.tools.corpus_search import CorpusIndex
from research_agent.train.samples import prompt_renderer
from research_agent.validation import check_call

REPO = Path(__file__).resolve().parents[2]
TODAY = "2026-10-10"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run_e2e = load_script("run_e2e")

ID1, ID2 = "2609.00001", "2609.00002"
CORPUS = [{"arxiv_id": ID1, "title": "Adaptive Draft Length for Speculative Decoding",
           "authors": ["A. Author"], "published": "2026-09-15",
           "abstract": "Speculative decoding drafts tokens with a small model.", "categories": ["cs.CL"]},
          {"arxiv_id": ID2, "title": "Tree Verification at Scale", "authors": ["A. Author"],
           "published": "2026-09-20", "categories": ["cs.CL"],
           "abstract": "Tree-structured verification checks drafted tokens at once."}]


class Pages:
    """The replay fetcher: each paper's metadata as a JSON feed and a full-text page."""

    def get(self, url: str) -> Fetched:
        for p in CORPUS:
            if url == paper_url(p["arxiv_id"]):
                return Fetched(url=url, final_url=url, status=200, body=json.dumps([p]).encode(),
                               content_type="")
            if url == fulltext_url(p["arxiv_id"]):
                return Fetched(url=url, final_url=url, status=200, body=b"<html><p>Method text.</p></html>",
                               content_type="")
        raise ReplayMiss(f"no cassette for {url}")


def parse_feed(xml: str) -> list[dict] | None:
    try:
        return json.loads(xml)
    except json.JSONDecodeError:
        return None


def world() -> World:
    owner = OwnerFunctions(check_call, decide, parse_feed, lambda html, section, n: "Method text.")
    return World(owner, Pages(), CorpusIndex(CORPUS), None,
                 {"speculative-decoding": Topic(frozenset({ID1}), ("speculative decoding",))})


LOOKUP = {"id": "t-lookup", "intent": "lookup_paper", "scenario": "normal", "lang": "en", "split": "dev",
          "family": "e2e/lookup_paper", "text": f"What is {ID2} about?", "slots": {"arxiv_id": ID2},
          "expect": {"status": "answered", "must_write_note": False, "note_about": None,
                     "must_cite_ids_within": "tool_outputs", "expected_tools_any": ["get_paper"]}}
GET2 = call("get_paper", "c2", arxiv_id=ID2)
ANSWER = f"{ID2} is about tree verification."


def write_tasks(path: Path, tasks: list[dict] = (LOOKUP,)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")
    return path


def fake_root(tmp_path: Path, monkeypatch) -> Path:
    """tmp_path as the repository root and the current folder: the guards only ever meet fake sets there."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_e2e, "REPO", tmp_path)
    (tmp_path / "data" / "eval").mkdir(parents=True)
    return tmp_path


def ticking(step: float):
    """A clock that moves step seconds forward every time it is read."""
    now = [0.0]

    def clock() -> float:
        now[0] += step
        return now[0]

    return clock


def test_run_tasks_tells_on_task_about_every_record_with_its_transcript(tmp_path):
    heard = []
    model = ScriptedModel([turn(GET2), turn(content=ANSWER), RuntimeError("the server went away")])
    records = run_tasks([LOOKUP, {**LOOKUP, "id": "t-two"}], model=model, world=world(), today=TODAY,
                        out=tmp_path, log=lambda line: None,
                        on_task=lambda record, transcript: heard.append((record, transcript)))
    assert [record for record, _ in heard] == records
    assert heard[0][1].status == "answered" and len(heard[0][1].steps) == 2
    assert heard[1][0]["status"] == "harness_error" and heard[1][1] is None  # run_request raised


def test_run_e2e_refuses_the_test_sets_with_exit_4_before_they_are_read(tmp_path, monkeypatch, capsys):
    root = fake_root(tmp_path, monkeypatch)
    for name in ("e2e-test.jsonl", "test.jsonl"):
        (root / "data" / "eval" / name).write_text("not json: never parsed\n", encoding="utf-8")
    for tasks in ("data/eval/e2e-test.jsonl", str(root / "data" / "eval" / "test.jsonl")):
        argv = [tasks, "--system", "b2", "--out", "runs/e2e", "--no-tokenizer", "--today", TODAY]
        code = run_e2e.main(argv, model=ScriptedModel([]), world=world())
        assert code == 4, tasks
        assert "locked test set" in capsys.readouterr().err
    assert not (root / "runs").exists()


def test_run_e2e_refuses_the_handwritten_test_items_with_exit_4_before_they_are_read(tmp_path, monkeypatch,
                                                                                       capsys):
    root = fake_root(tmp_path, monkeypatch)
    for name in ("handwritten.jsonl", "handwritten-specs.jsonl"):
        tasks = write_tasks(root / "data" / "eval" / name)  # well-formed: refused all the same, by location
        argv = [f"data/eval/{name}", "--system", "b2", "--out", "runs/e2e", "--no-tokenizer",
                "--today", TODAY]
        code = run_e2e.main(argv, model=ScriptedModel([turn(GET2), turn(content=ANSWER)]), world=world())
        assert code == 4, tasks
        assert "handwritten items with their gold turns" in capsys.readouterr().err
    assert not (root / "runs").exists()


TEST_TASK = {**LOOKUP, "id": "e2e-test-07", "split": "test"}  # LOOKUP as an end-to-end test task


def test_run_e2e_refuses_a_copy_of_test_tasks_by_the_tasks_it_holds_with_exit_4(tmp_path, monkeypatch,
                                                                                capsys):
    """A copy of end-to-end test tasks with other bytes than the locked file, under another name, is refused
    by its tasks (split "test", an id of the test split) before any task runs."""
    root = fake_root(tmp_path, monkeypatch)
    for name, task in (("copy.jsonl", TEST_TASK), ("relabelled.jsonl", {**TEST_TASK, "split": "dev"})):
        write_tasks(root / name, [task])
        argv = [name, "--system", "b2", "--out", "runs/e2e", "--no-tokenizer", "--today", TODAY]
        model = ScriptedModel([turn(GET2), turn(content=ANSWER)])
        assert run_e2e.main(argv, model=model, world=world()) == 4, name
        assert "holds test material" in capsys.readouterr().err and model.contexts == []
    assert not (root / "runs").exists()


def test_run_tasks_runs_test_tasks_only_as_the_items_lockedset_open_returned(tmp_path):
    model = ScriptedModel([turn(GET2), turn(content=ANSWER)])
    with pytest.raises(RefusedTestSet, match="only the phase-4 test runner"):
        run_tasks([TEST_TASK], model=model, world=world(), today=TODAY, out=tmp_path / "refused",
                  log=lambda line: None)
    assert model.contexts == [] and not (tmp_path / "refused").exists()
    tasks = write_tasks(tmp_path / "e2e-test.jsonl", [TEST_TASK])
    write_lock(tasks)
    opened = LockedSet(tasks, tmp_path / "e2e-test-access.jsonl").open("M3", "the phase-4 run")
    [record] = run_tasks(opened, model=model, world=world(), today=TODAY, out=tmp_path / "run",
                         log=lambda line: None)
    assert (record["id"], record["status"], record["success"]) == ("e2e-test-07", "answered", True)


def test_run_e2e_lock_checks_the_development_tasks_and_refuses_changed_ones_with_exit_5(tmp_path, monkeypatch,
                                                                                         capsys):
    root = fake_root(tmp_path, monkeypatch)
    dev = write_tasks(root / "data" / "eval" / "e2e-dev.jsonl")
    write_lock(dev)
    argv = ["data/eval/e2e-dev.jsonl", "--system", "b2", "--no-tokenizer", "--today", TODAY, "--out"]
    assert run_e2e.main([*argv, "runs/ok"], model=ScriptedModel([turn(GET2), turn(content=ANSWER)]),
                        world=world()) == 0
    report = json.loads((root / "runs" / "ok" / "report.json").read_text(encoding="utf-8"))
    assert report["manifest"]["tasks_locked"] is True and report["summary"]["success_rate"] == 1.0
    dev.write_bytes(dev.read_bytes() + b"\n")
    capsys.readouterr()
    assert run_e2e.main([*argv, "runs/changed"], model=ScriptedModel([]), world=world()) == 5
    assert "does not match its lock" in capsys.readouterr().err
    assert not (root / "runs" / "changed").exists()


class SequenceServer:
    """A llama-server stand-in for LocalModel: the given completions in order, one adapter listed, the latest
    timings kept as LocalClient keeps them."""

    def __init__(self, texts: list[str], adapter: Path) -> None:
        self.texts, self.adapter, self.scales = list(texts), adapter, []
        self.last_timings: dict | None = None

    def adapters(self) -> list[dict]:
        return [{"id": 0, "path": str(self.adapter), "scale": 1.0}]

    def complete(self, prompt: str, *, lora_scale: float | None = None,
                 stop: list[str] | None = None) -> Completion:
        self.scales.append(lora_scale)
        self.last_timings = {"predicted_ms": 10.0 * len(self.scales)}
        usage = {"prompt_tokens": 50, "completion_tokens": 5}
        return Completion(text=self.texts.pop(0), raw={"usage": usage})

    def props(self) -> dict:
        return {"build_info": "b11349"}


def test_run_e2e_runs_b4_over_its_adapter_and_records_it_and_the_step_times(tmp_path):
    adapter = tmp_path / "b4-checkpoint-1112-f16.gguf"
    adapter.write_bytes(b"GGUF b4")
    get_paper = '<tool_call>\n{"name": "get_paper", "arguments": {"arxiv_id": "' + ID2 + '"}}\n</tool_call>'
    server = SequenceServer([get_paper, ANSWER], adapter)
    out = tmp_path / "e2e-b4"
    code = run_e2e.main([str(write_tasks(tmp_path / "tasks.jsonl")), "--system", "b4", "--out", str(out),
                         "--no-tokenizer"], world=world(), local_client=server,
                        render=prompt_renderer(FakeTokenizer()))
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert code == 0 and report["summary"]["success_rate"] == 1.0 and server.scales == [1.0, 1.0]
    manifest = report["manifest"]
    assert (manifest["system"], manifest["model"], manifest["lora_scale"]) == ("b4", "qwen3-1.7b:lora=1", 1.0)
    assert manifest["adapters"] == [{"id": 0, "path": str(adapter), "sha256": sha256_of(adapter)}]
    (task,) = json.loads((out / "timing.json").read_text(encoding="utf-8"))["tasks"]
    assert task["id"] == "t-lookup" and [s["index"] for s in task["steps"]] == [1, 2]
    assert [c["server"] for c in task["calls"]] == [{"predicted_ms": 10.0}, {"predicted_ms": 20.0}]


def test_run_e2e_writes_a_timing_sidecar_while_the_report_and_the_transcripts_stay_byte_identical(tmp_path):
    tasks = write_tasks(tmp_path / "tasks.jsonl")
    bad = call("get_paper", "c1", raw="{not json")  # rejected, then repaired within step 1
    outs = []
    for step in (0.5, 2.0):
        out = tmp_path / f"clock-{step:g}"
        model = ScriptedModel([turn(bad), turn(GET2), turn(content=ANSWER)])
        assert run_e2e.main([str(tasks), "--system", "b2", "--out", str(out), "--no-tokenizer"], model=model,
                            world=world(), clock=ticking(step)) == 0
        outs.append(out)
    reports = [(out / "report.json").read_bytes() for out in outs]
    assert b"wall" not in reports[0]  # every time goes to the sidecar
    assert reports[0] == reports[1]
    transcripts = [(out / "transcripts" / "t-lookup.json").read_bytes() for out in outs]
    assert transcripts[0] == transcripts[1]
    timings = [json.loads((out / "timing.json").read_text(encoding="utf-8")) for out in outs]
    (first,), (second,) = timings[0]["tasks"], timings[1]["tasks"]
    assert first["steps"] == [{"index": 1, "calls": 2, "wall_ms": 1000.0},
                              {"index": 2, "calls": 1, "wall_ms": 500.0}]
    assert second["steps"] == [{"index": 1, "calls": 2, "wall_ms": 4000.0},
                               {"index": 2, "calls": 1, "wall_ms": 2000.0}]
    assert [c["step"] for c in first["calls"]] == [1, 1, 2] and first["status"] == "answered"
    # the task's wall time on the same clock: from the timer's start (1 step) to its record (8 steps: the
    # three calls read the clock twice each)
    assert (first["wall_ms"], second["wall_ms"]) == (3500.0, 14000.0)


# ---- fault injection (M4) ---------------------------------------------------------------------------------


def files_of(folder: Path) -> dict[str, bytes]:
    return {p.relative_to(folder).as_posix(): p.read_bytes()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def fault_turns() -> list:
    """A task that searches, reads the paper twice and its full text, then answers: every tool asked."""
    return [turn(call("search_arxiv", "c1", query="tree verification")),
            turn(GET2, call("fetch_fulltext", "c3", arxiv_id=ID2)),
            turn(call("get_paper", "c4", arxiv_id=ID2)),
            turn(content=ANSWER)]


def test_at_fault_rate_zero_the_run_is_byte_identical_to_a_run_without_the_option(tmp_path):
    """--fault-rate 0 (the default) is today's run: the same report, transcripts and notes, byte for byte, and
    neither the manifest nor a record names a fault; --fault-persistent changes nothing at rate 0."""
    tasks = write_tasks(tmp_path / "tasks.jsonl", [LOOKUP, {**LOOKUP, "id": "t-two"}])
    runs = {"plain": [], "zero": ["--fault-rate", "0"],
            "persistent": ["--fault-rate", "0", "--fault-persistent"]}
    folders = {}
    for name, extra in runs.items():
        out = tmp_path / name
        model = ScriptedModel(fault_turns() + fault_turns())
        assert run_e2e.main([str(tasks), "--system", "b2", "--out", str(out), "--no-tokenizer", "--seed", "3",
                             *extra], model=model, world=world()) == 0
        folders[name] = {k: v for k, v in files_of(out).items() if k != "timing.json"}
    assert folders["plain"] == folders["zero"] == folders["persistent"]
    assert {"report.json", "transcripts/t-lookup.json", "transcripts/t-two.json"} <= set(folders["plain"])
    assert all(b'"fault' not in v for v in folders["plain"].values())  # no "faults", no "fault_injection"


def test_a_fault_run_records_its_condition_in_the_manifest_and_every_fault_in_the_records(tmp_path):
    tasks = write_tasks(tmp_path / "tasks.jsonl")
    out = tmp_path / "e2e-b2-f50p"
    assert run_e2e.main([str(tasks), "--system", "b2", "--out", str(out), "--no-tokenizer", "--seed", "7",
                         "--fault-rate", "0.5", "--fault-persistent"],
                        model=ScriptedModel(fault_turns()), world=world(), sleep=lambda seconds: None) == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["manifest"]["fault_injection"] == {"rate": 0.5, "persistent": True, "faults": list(FAULTS),
                                                     "seed": 7}
    assert report["manifest"]["seed"] == 7
    [record] = report["tasks"]
    assert isinstance(record["faults"], list)
    transcript = json.loads((out / record["transcript"]).read_text(encoding="utf-8"))
    assert [f for step in transcript["steps"] for f in step.get("faults", [])] == record["faults"]
    assert "tools" in json.loads((out / "timing.json").read_text(encoding="utf-8"))["tasks"][0]


def test_run_e2e_refuses_a_fault_rate_outside_0_to_1(tmp_path, capsys):
    tasks = write_tasks(tmp_path / "tasks.jsonl")
    for rate in ("-0.1", "1.5", "half"):
        with pytest.raises(SystemExit) as exc:
            run_e2e.main([str(tasks), "--system", "b2", "--out", str(tmp_path / "x"), "--no-tokenizer",
                          "--fault-rate", rate], model=ScriptedModel([]), world=world())
        assert exc.value.code == 2 and "--fault-rate" in capsys.readouterr().err
    assert not (tmp_path / "x").exists()


def test_run_e2e_refuses_a_non_empty_out_folder_with_exit_6_before_any_task_runs(tmp_path, capsys):
    """A run folder belongs to one condition: a second run into it, under another fault rate or the same, is
    refused, so records of two conditions never mix."""
    tasks = write_tasks(tmp_path / "tasks.jsonl")
    out = tmp_path / "e2e-b2"
    argv = [str(tasks), "--system", "b2", "--out", str(out), "--no-tokenizer"]
    assert run_e2e.main(argv, model=ScriptedModel([turn(GET2), turn(content=ANSWER)]), world=world()) == 0
    before = files_of(out)
    for extra in ([], ["--fault-rate", "0.2"]):
        model = ScriptedModel([turn(GET2), turn(content=ANSWER)])
        assert run_e2e.main([*argv, *extra], model=model, world=world()) == 6
        assert "not empty" in capsys.readouterr().err and model.contexts == []
    assert files_of(out) == before
    (tmp_path / "empty").mkdir()
    assert run_e2e.main([str(tasks), "--system", "b2", "--out", str(tmp_path / "empty"), "--no-tokenizer"],
                        model=ScriptedModel([turn(GET2), turn(content=ANSWER)]), world=world()) == 0


def seed_meeting(fault: str, task: str, url: str) -> int:
    """A seed under which the task's first request for url meets fault at rate 1."""
    choices = applicable(url, 200)
    return next(seed for seed in range(1000) if choices[draw(seed, task, url, 0)[1] % len(choices)] == fault)


def test_the_delay_fault_sleeps_and_its_time_reaches_the_timing_sidecar_per_tool_call(tmp_path):
    """The delay fault waits DELAY_S seconds (for real outside tests); the sidecar keeps every arXiv tool
    call's wall time with the faults it met, so the fault table can measure the extra latency per fetch."""
    now = [0.0]

    def clock() -> float:
        now[0] += 0.001
        return now[0]

    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    seed = seed_meeting("delay", LOOKUP["id"], canonical_url(paper_url(ID2)))
    out = tmp_path / "e2e-delay"
    assert run_e2e.main([str(write_tasks(tmp_path / "tasks.jsonl")), "--system", "b2", "--out", str(out),
                         "--no-tokenizer", "--seed", str(seed), "--fault-rate", "1"],
                        model=ScriptedModel([turn(GET2), turn(content=ANSWER)]), world=world(), clock=clock,
                        sleep=sleep) == 0
    assert slept == [DELAY_S]
    [task] = json.loads((out / "timing.json").read_text(encoding="utf-8"))["tasks"]
    [timed] = task["tools"]
    assert (timed["tool"], timed["step"], timed["call"], timed["faults"]) == ("get_paper", 1, 0, ["delay"])
    assert DELAY_S * 1000 <= timed["wall_ms"] < DELAY_S * 1000 + 50
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert [f["kind"] for f in report["tasks"][0]["faults"]] == ["delay"]
    assert report["tasks"][0]["success"] is True  # a delay changes no answer


# ---- M7: the family path, --mode, --base-seed and --reasoning-budget (docs/m7-家族层设计.md §7) ------------


def test_run_e2e_runs_the_family_path_in_think_mode_and_records_the_generation(tmp_path, monkeypatch):
    install_fake_family(monkeypatch, tmp_path / "model")
    get_paper = '<tool_call>\n{"name": "get_paper", "arguments": {"arxiv_id": "' + ID2 + '"}}\n</tool_call>'
    fake = FakeLlama([Scripted(("Look it up.", "\n</think>\n\n", get_paper)),
                      Scripted(("Now answer.", "\n</think>\n\n", ANSWER))])
    out = tmp_path / "e2e-think"
    code = run_e2e.main([str(write_tasks(tmp_path / "tasks.jsonl")), "--system", "b0", "--out", str(out),
                         "--tokenizer", str(tmp_path / "model"), "--mode", "think", "--base-seed", "5",
                         "--reasoning-budget", "32"], world=world(), local_client=fake.client())
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert code == 0 and report["summary"]["success_rate"] == 1.0
    manifest = report["manifest"]
    assert manifest["generation"] == {
        "mode": "think", "decoding": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
                                      "presence_penalty": 0.0},
        "reasoning_budget": 32, "answer_tokens": 512, "base_seed": 5, "sample": 0}
    assert manifest["family"]["name"] == "fake-hybrid" and manifest["model"] == "model:lora=0"
    assert manifest["seed"] == 42  # the tasks' seed (the scaffold notes, the faults) stays its own
    transcript = json.loads((out / "transcripts" / "t-lookup.json").read_text(encoding="utf-8"))
    attempts = [attempt for step in transcript["steps"] for attempt in step["attempts"]]
    assert [a["reasoning"] for a in attempts] == ["Look it up.", "Now answer."]
    assert [(a["thinking"]["mode"], a["thinking"]["forced"], a["thinking"]["seeds"]) for a in attempts] == [
        ("think", False, None)] * 2
    assert [r["max_tokens"] for r in fake.requests] == [32, 32]
    assert "reasoning" not in json.dumps(transcript["messages"])


def test_run_e2e_without_the_flags_records_nothing_new_in_a_transcript(tmp_path, monkeypatch):
    install_fake_family(monkeypatch, tmp_path / "model")
    get_paper = '<tool_call>\n{"name": "get_paper", "arguments": {"arxiv_id": "' + ID2 + '"}}\n</tool_call>'
    fake = FakeLlama([Scripted((get_paper,)), Scripted((ANSWER,))])
    out = tmp_path / "e2e-plain"
    code = run_e2e.main([str(write_tasks(tmp_path / "tasks.jsonl")), "--system", "b0", "--out", str(out),
                         "--tokenizer", str(tmp_path / "model"), "--no-tokenizer"], world=world(),
                        local_client=fake.client())
    assert code == 0
    transcript = (out / "transcripts" / "t-lookup.json").read_text(encoding="utf-8")
    assert "reasoning" not in transcript and "thinking" not in transcript
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["manifest"]["generation"]["mode"] == "no_think" and report["summary"]["success_rate"] == 1.0
    assert report["manifest"]["context_check"] is None


# ---- M7: --greedy, think mode decoded greedily, run_eval's (docs/m7-进度.md, FT3 的选点与解码规则) --------

SAMPLING = Decoding(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)  # a family's sampling in think mode
SAMPLED_FIELDS = ("seed", "top_p", "top_k", "min_p", "presence_penalty")


def greedy_run(tmp_path, monkeypatch, name: str, *extra: str) -> tuple[int, Path, FakeLlama]:
    """b0 over LOOKUP in think mode on a family whose think mode samples, each step's reasoning closed by
    itself within the budget of 32; (the exit, the run folder, the server)."""
    install_fake_family(monkeypatch, tmp_path / "model",
                        decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY})
    get_paper = '<tool_call>\n{"name": "get_paper", "arguments": {"arxiv_id": "' + ID2 + '"}}\n</tool_call>'
    fake = FakeLlama([Scripted(("Look it up.", "\n</think>\n\n", get_paper)),
                      Scripted(("Now answer.", "\n</think>\n\n", ANSWER))])
    out = tmp_path / name
    code = run_e2e.main([str(write_tasks(tmp_path / "tasks.jsonl")), "--system", "b0", "--out", str(out),
                         "--tokenizer", str(tmp_path / "model"), "--reasoning-budget", "32", *extra],
                        world=world(), local_client=fake.client())
    return code, out, fake


def test_greedy_thinks_decoded_greedily_sends_no_seed_and_records_the_generation_as_run_eval(tmp_path,
                                                                                            monkeypatch):
    code, out, fake = greedy_run(tmp_path, monkeypatch, "greedy", "--greedy")
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert code == 0 and report["summary"]["success_rate"] == 1.0
    # scripts/run_eval.py --greedy's record: think mode, greedy decoding whatever the mode's own, base seed 0
    assert report["manifest"]["generation"] == {
        "mode": "think", "decoding": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
                                      "presence_penalty": 0.0},
        "reasoning_budget": 32, "answer_tokens": 512, "base_seed": 0, "sample": 0}
    assert report["manifest"]["temperature"] == 0.0
    assert [(r["max_tokens"], r["temperature"]) for r in fake.requests] == [(32, 0.0), (32, 0.0)]
    assert not any(set(SAMPLED_FIELDS) & set(r) for r in fake.requests)  # temperature 0 alone, no seed
    transcript = (out / "transcripts" / "t-lookup.json").read_bytes()
    attempts = [a for step in json.loads(transcript)["steps"] for a in step["attempts"]]
    assert [(a["reasoning"], a["thinking"]["mode"], a["thinking"]["seeds"]) for a in attempts] == [
        ("Look it up.", "think", None), ("Now answer.", "think", None)]
    # --mode think --greedy says the same thing twice: the same requests, the same transcript
    code, again, twice = greedy_run(tmp_path, monkeypatch, "again", "--mode", "think", "--greedy")
    assert code == 0 and twice.requests == fake.requests
    assert (again / "transcripts" / "t-lookup.json").read_bytes() == transcript
    # a base seed is recorded as given, as run_eval --greedy --seed 5 records it; no request carries a seed
    code, seeded, five = greedy_run(tmp_path, monkeypatch, "seeded", "--greedy", "--base-seed", "5")
    manifest = json.loads((seeded / "report.json").read_text(encoding="utf-8"))["manifest"]
    assert code == 0 and manifest["generation"] == {**report["manifest"]["generation"], "base_seed": 5}
    assert five.requests == fake.requests
    assert (seeded / "transcripts" / "t-lookup.json").read_bytes() == transcript
    # without it the think mode's own decoding samples, every request with its seed
    code, sampled, drawn = greedy_run(tmp_path, monkeypatch, "sampled", "--mode", "think")
    manifest = json.loads((sampled / "report.json").read_text(encoding="utf-8"))["manifest"]
    assert code == 0 and manifest["generation"]["decoding"]["temperature"] == 0.6
    assert [r["seed"] for r in drawn.requests] == [request_seed(0, r["prompt"], 1, 0) for r in drawn.requests]


@pytest.mark.parametrize(("options", "message"), [
    (["--system", "b2", "--greedy"], "--greedy"),
    (["--system", "b0", "--greedy", "--mode", "no_think"], "--mode no_think")])
def test_greedy_is_a_thinking_run_of_a_local_system(options, message, tmp_path, capsys):
    tasks = str(write_tasks(tmp_path / "tasks.jsonl"))
    with pytest.raises(SystemExit) as stopped:
        run_e2e.main([tasks, *options, "--out", str(tmp_path / "x"), "--no-tokenizer"],
                     model=ScriptedModel([]), world=world())
    assert stopped.value.code == 2 and message in capsys.readouterr().err
    assert not (tmp_path / "x").exists()


def test_greedy_on_a_family_that_never_thinks_exits_2(tmp_path, monkeypatch, capsys):
    install_fake_family(monkeypatch, tmp_path / "model", "none")
    code = run_e2e.main([str(write_tasks(tmp_path / "tasks.jsonl")), "--system", "b0", "--out",
                         str(tmp_path / "x"), "--tokenizer", str(tmp_path / "model"), "--greedy"],
                        world=world(), local_client=FakeLlama([]).client())
    assert code == 2 and "has no think mode" in capsys.readouterr().err
    assert not (tmp_path / "x").exists()


# ---- M7: the steps over the training length (docs/m7-计划.md §2 端到端预算) -------------------------------


def think_run(tmp_path, monkeypatch, name: str, *extra: str) -> tuple[dict, FakeLlama]:
    """A b0 run in think mode over LOOKUP: step 1 closes its reasoning and calls get_paper in one request,
    step 2's reasoning is cut by the budget and closed by force before its answer; (the manifest, the
    server)."""
    install_fake_family(monkeypatch, tmp_path / "model")
    get_paper = '<tool_call>\n{"name": "get_paper", "arguments": {"arxiv_id": "' + ID2 + '"}}\n</tool_call>'
    fake = FakeLlama([Scripted(("Look it up.", "\n</think>\n\n", get_paper)),
                      Scripted(("endless",), "length"), Scripted((ANSWER,))])
    out = tmp_path / name
    code = run_e2e.main([str(write_tasks(tmp_path / "tasks.jsonl")), "--system", "b0", "--out", str(out),
                         "--tokenizer", str(tmp_path / "model"), "--mode", "think", "--reasoning-budget", "8",
                         *extra], world=world(), local_client=fake.client())
    assert code == 0
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["success_rate"] == 1.0
    return report["manifest"], fake


def test_the_manifest_counts_the_steps_whose_prompt_and_generation_exceed_the_training_length(tmp_path,
                                                                                                 monkeypatch):
    manifest, fake = think_run(tmp_path, monkeypatch, "default")
    assert manifest["training_length"] == 4096 and manifest["steps_over_training_length"] == 0
    first, second, third = fake.requests  # step 1 in one request; step 2 cut, then closed by force
    close = len(third["prompt"]) - len(second["prompt"]) - 1  # the forced close after the one reasoning token
    assert close == 3
    # a step's length: its prompt ids, its reasoning tokens, a forced close's tokens and its answer tokens
    lengths = [len(first["prompt"]) + 3, len(second["prompt"]) + 1 + close + 1]
    assert lengths[1] > lengths[0]
    at_limit, _ = think_run(tmp_path, monkeypatch, "at-limit", "--training-length", str(lengths[1]))
    assert at_limit["steps_over_training_length"] == 0  # a step as long as the training length is not over
    one_over, _ = think_run(tmp_path, monkeypatch, "one-over", "--training-length", str(lengths[1] - 1))
    assert one_over["steps_over_training_length"] == 1
    both, _ = think_run(tmp_path, monkeypatch, "both", "--training-length", str(lengths[0] - 1))
    assert both["steps_over_training_length"] == 2 and both["training_length"] == lengths[0] - 1


def test_the_training_length_is_measured_for_a_local_system_with_the_context_check_only(tmp_path, monkeypatch,
                                                                                          capsys):
    install_fake_family(monkeypatch, tmp_path / "model")
    folder = str(tmp_path / "model")
    get_paper = '<tool_call>\n{"name": "get_paper", "arguments": {"arxiv_id": "' + ID2 + '"}}\n</tool_call>'
    tasks = str(write_tasks(tmp_path / "tasks.jsonl"))

    def manifest_of(name: str) -> dict:
        return json.loads((tmp_path / name / "report.json").read_text(encoding="utf-8"))["manifest"]

    # a greedy no_think step is measured by its prompt and its usage: both steps are over a length of 1
    fake = FakeLlama([Scripted((get_paper,)), Scripted((ANSWER,))])
    code = run_e2e.main([tasks, "--system", "b0", "--out", str(tmp_path / "short"), "--tokenizer", folder,
                         "--training-length", "1"], world=world(), local_client=fake.client())
    assert code == 0 and (manifest_of("short")["training_length"],
                          manifest_of("short")["steps_over_training_length"]) == (1, 2)
    assert "steps over the training length 1: 2 of 2" in capsys.readouterr().out
    # without the context check the prompts are not measured, so neither is the count
    fake = FakeLlama([Scripted((get_paper,)), Scripted((ANSWER,))])
    code = run_e2e.main([tasks, "--system", "b0", "--out", str(tmp_path / "unmeasured"), "--tokenizer",
                         folder, "--no-tokenizer"], world=world(), local_client=fake.client())
    assert code == 0 and manifest_of("unmeasured")["training_length"] == 4096
    assert manifest_of("unmeasured")["steps_over_training_length"] is None
    assert "training length" not in capsys.readouterr().out
    # a cloud system has no training length, and the flag is a usage error there
    code = run_e2e.main([tasks, "--system", "b2", "--out", str(tmp_path / "cloud"), "--no-tokenizer"],
                        model=ScriptedModel([turn(GET2), turn(content=ANSWER)]), world=world())
    assert code == 0 and not {"training_length", "steps_over_training_length"} & set(manifest_of("cloud"))
    with pytest.raises(SystemExit) as stopped:
        run_e2e.main([tasks, "--system", "b2", "--out", str(tmp_path / "refused"), "--no-tokenizer",
                      "--training-length", "5120"], model=ScriptedModel([]), world=world())
    assert stopped.value.code == 2 and "--training-length" in capsys.readouterr().err
    assert not (tmp_path / "refused").exists()


def test_step_lengths_measure_the_local_turns_alone_by_their_thinking_record_or_their_usage():
    plain = turn(content="x", paid=False)
    plain.usage = {"prompt_tokens": 7, "completion_tokens": 5}
    thought, closed = turn(content="y", paid=False), turn(content="v", paid=False)
    thought.thinking = {"mode": "think", "reasoning_tokens": 4, "answer_tokens": 2, "forced": True,
                        "phases": 2, "seeds": None}
    closed.thinking = {**thought.thinking, "forced": False, "phases": 1}
    routed = turn(content="w")  # the cloud's answer to a routed step, paid
    routed.route = {"answered_by": "cloud", "reason": "min_decision_prob 0.1 < 0.5"}
    cached = turn(content="z", paid=False, cached=True)
    lengths = run_e2e.StepLengths(ScriptedModel([plain, thought, closed, routed, cached]),
                                  count_tokens=lambda messages, tools: 10, training_length=16, close_tokens=3)
    assert [lengths.next_turn([], []) for _ in range(5)] == [plain, thought, closed, routed, cached]
    # 10 + 5 = 15 and 10 + 4 + 2 = 16 are not over 16; 10 + 4 + 3 + 2 = 19 is; the cloud's turns are not
    # measured
    assert (lengths.steps, lengths.over) == (3, 1)
    assert lengths.name == "fake" and lengths.contexts is lengths.model.contexts  # the rest is the model's


def test_a_forced_close_counts_the_ids_generation_appends_not_the_close_encoded_on_its_own(tmp_path):
    # llm.generation closes a reasoning with family.encode_close(): under a tokenizer with a dummy prefix
    # (Nanbeige's) the close encoded on its own starts with a word-start marker no forced close holds
    def loaded(name: str, tokenizer: FakeFamilyTokenizer):
        return load_family(model_folder(tmp_path / name), tokenizer=tokenizer,
                           families=[fake_family(tokenizer)])

    marked = loaded("marked", DummyPrefix("only", pieces=(MARK,)))
    close = marked.spec.think_close
    assert run_e2e.close_tokens(marked) == len(marked.encode_close()) == len(marked.encode_text(close)) - 1
    plain = loaded("plain", FakeFamilyTokenizer("hybrid"))
    assert run_e2e.close_tokens(plain) == len(plain.encode_close()) == len(plain.encode_text(close))
    # nothing is ever closed without a family, or for one that never thinks, whatever close it names
    never = loaded("never", FakeFamilyTokenizer("none"))
    assert never.spec.think_close and run_e2e.close_tokens(never) == run_e2e.close_tokens(None) == 0


@pytest.mark.parametrize(("folder", "fewer"), [("Nanbeige4-3B-Thinking-2511", 1), ("Nanbeige4.1-3B", 1),
                                               ("Qwen3-4B-Thinking-2507", 0), ("granite-4.2-3b", 0)])
def test_a_real_forced_close_counts_one_id_fewer_under_nanbeiges_dummy_prefix(folder, fewer):
    path = REPO / "models" / folder
    if not (path / "tokenizer_config.json").is_file():
        pytest.skip("local model files are not present")
    family = load_family(path)
    alone = family.encode_text(family.spec.think_close)
    assert run_e2e.close_tokens(family) == len(family.encode_close()) == len(alone) - fewer


# ---- M7: stepwise routing by a frozen configuration, --routing-config (docs/m7-计划.md WP-K) ---------------

P5_BUDGET = {"text_exempt": True, "param_triggers": [
    {"tool": "search_notes", "param": "top_k", "allowed": [3]},
    {"tool": "list_notes", "param": "limit", "allowed": [10]}], "budget_forced": True}
GET_PAPER = '<tool_call>\n{"name": "get_paper", "arguments": {"arxiv_id": "' + ID2 + '"}}\n</tool_call>'
PAID_TURN = (1000 * 2.0 + 50 * 8.0) / 1e6  # one paid deepseek-flash turn of the fake cloud, in RMB
ROUTING_M7 = "results/routing-m7.json"


class AnsweringCloud:
    """DeepSeek's SDK, faked: looks the task's paper up, or answers once a tool output is in the context;
    every request recorded."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **payload):
        self.requests.append(payload)
        if payload["messages"][-1]["role"] == "tool":
            message = {"role": "assistant", "content": ANSWER}
        else:
            message = {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_9", "type": "function",
                 "function": {"name": "get_paper", "arguments": json.dumps({"arxiv_id": ID2})}}]}
        data = {"id": "chatcmpl-1", "model": payload["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 50}}
        return types.SimpleNamespace(model_dump=lambda **kwargs: data)


def routed_root(tmp_path: Path, monkeypatch) -> Path:
    """tmp_path as the repository root and the current folder, with the fake model folder, FT's adapter, two
    development sets and two frozen configurations: results/routing-m7.json, chosen on both sets (P5 with the
    budget trigger), and ftr's default, results/routing-m4.json, chosen on the first (the threshold alone);
    under both a local answer below 0.5 is routed."""
    root = fake_root(tmp_path, monkeypatch)
    install_fake_family(monkeypatch, root / "model")
    adapter = root / "runs" / "train" / "ft3" / "checkpoint-548-f16.gguf"
    adapter.parent.mkdir(parents=True)
    adapter.write_bytes(b"GGUF ft3")
    dev = write_tasks(root / "data" / "eval" / "dev.jsonl", [{"id": "q-1"}])
    second = write_tasks(root / "data" / "eval" / "v2" / "dev-v2.jsonl", [{"id": "q-2"}, {"id": "q-3"}])
    frozen = {"signal": "min_decision_prob", "threshold": 0.5, "rule_met": True,
              "adapter_sha256": sha256_of(adapter), "cloud_model": "deepseek-flash"}
    (root / "results").mkdir()
    (root / ROUTING_M7).write_text(json.dumps({
        **frozen, "chosen_policy": "P5", "rules": P5_BUDGET,
        "dev_sets": [{"path": "data/eval/dev.jsonl", "sha256": sha256_of(dev), "n": 1},
                     {"path": "data/eval/v2/dev-v2.jsonl", "sha256": sha256_of(second), "n": 2}]}),
        encoding="utf-8")
    (root / "results" / "routing-m4.json").write_text(json.dumps({
        **frozen, "dev_sha256": sha256_of(dev), "dev_set": "data/eval/dev.jsonl"}), encoding="utf-8")
    write_tasks(root / "tasks.jsonl")
    return root


def ftr_run(root: Path, name: str, script: list[Scripted], *extra: str) -> tuple[int, Path, FakeLlama,
                                                                                AnsweringCloud]:
    """ftr over LOOKUP on the fake FT server and the fake cloud, its own ledger and cache; (the exit, the run
    folder, the server, the cloud)."""
    adapter = next((root / "runs" / "train" / "ft3").glob("*.gguf"))
    fake = FakeLlama(script, adapters=[{"id": 0, "path": str(adapter), "scale": 1.0}])
    cloud = AnsweringCloud()
    out = root / "runs" / name
    code = run_e2e.main(["tasks.jsonl", "--system", "ftr", "--out", str(out), "--tokenizer",
                         str(root / "model"), "--no-tokenizer", "--llm-dir",
                         str(root / "runs" / f"{name}-llm"), "--today", TODAY, *extra], world=world(),
                        local_client=fake.client(), openai_client=cloud)
    return code, out, fake, cloud


def unsure(text: str) -> Scripted:
    """A local answer of one token with the probability 0.2: below the threshold."""
    return Scripted((text,), logprobs=(math.log(0.2),))


def test_with_a_routing_configuration_every_local_step_is_routed_on_its_own_and_the_record_says_which(
        tmp_path, monkeypatch, capsys):
    root = routed_root(tmp_path, monkeypatch)
    # step 1: an unsure call, routed; step 2: the local model is asked again and answers in text
    code, out, fake, cloud = ftr_run(root, "e2e-ftr-r3", [unsure(GET_PAPER), Scripted((ANSWER,))],
                                     "--routing-config", ROUTING_M7)
    assert code == 0 and len(fake.requests) == 2 and len(cloud.requests) == 1
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert report["summary"]["success_rate"] == 1.0
    manifest = report["manifest"]
    assert manifest["system"] == "ftr" and manifest["e2e_routing"] == "stepwise"
    routing = manifest["routing"]
    assert (routing["path"], routing["rules"], routing["policy"]) == (ROUTING_M7, P5_BUDGET, "P5")
    assert [entry["path"] for entry in routing["dev_sets"]] == ["data/eval/dev.jsonl",
                                                                "data/eval/v2/dev-v2.jsonl"]
    (record,) = report["tasks"]
    paid = pytest.approx(PAID_TURN)
    assert record["routing"] == {
        "routed_steps": [1], "local_turns": 1, "cloud_turns": 1,
        "routed": [{"step": 1, "reason": "min_decision_prob 0.2 < 0.5", "paid": True, "cost_rmb": paid}],
        "cloud_cost_rmb": paid}
    assert record["cost_rmb"] == paid and (record["paid_calls"], record["local_calls"]) == (1, 1)
    transcript = json.loads((out / record["transcript"]).read_text(encoding="utf-8"))
    assert [step["attempts"][-1]["model"] for step in transcript["steps"]] == ["deepseek-flash",
                                                                             "model:lora=1"]
    assert "routed 1 step(s) in 1 of 1 task(s)" in capsys.readouterr().out


def test_a_step_is_routed_on_its_answers_signals_and_by_the_budget_trigger_a_final_answer_too(tmp_path,
                                                                                             monkeypatch):
    root = routed_root(tmp_path, monkeypatch)
    # step 1 thinks unsurely and answers surely: its answer's signals keep it local; step 2's reasoning hits
    # the budget and is closed by force: the budget trigger routes it, though it is a final text answer
    script = [Scripted(("Look it up.", "\n</think>\n\n", GET_PAPER),
                       logprobs=(math.log(0.01), math.log(0.01), math.log(0.9))),
              Scripted(("endless",), "length"), Scripted((ANSWER,))]
    code, out, fake, cloud = ftr_run(root, "e2e-ftr-think", script, "--routing-config", ROUTING_M7,
                                     "--mode", "think", "--reasoning-budget", "8")
    assert code == 0 and len(fake.requests) == 3 and len(cloud.requests) == 1
    (record,) = json.loads((out / "report.json").read_text(encoding="utf-8"))["tasks"]
    assert record["success"] is True and record["routing"]["routed_steps"] == [2]
    (routed,) = record["routing"]["routed"]
    assert routed["reason"].startswith("budget_forced:") and routed["step"] == 2
    assert (record["routing"]["local_turns"], record["routing"]["cloud_turns"]) == (1, 1)


def test_without_a_routing_configuration_ftr_runs_as_before_sticky_and_records_nothing_new(tmp_path,
                                                                                            monkeypatch):
    root = routed_root(tmp_path, monkeypatch)
    # results/routing-m4.json, ftr's default: once routed, the task stays in the cloud
    code, out, fake, cloud = ftr_run(root, "e2e-ftr", [unsure(GET_PAPER)])
    assert code == 0 and len(fake.requests) == 1 and len(cloud.requests) == 2
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert "e2e_routing" not in report["manifest"] and report["manifest"]["routing"]["path"] == (
        "results/routing-m4.json")
    assert "routing" not in report["tasks"][0] and report["summary"]["success_rate"] == 1.0
    transcript = json.loads((out / report["tasks"][0]["transcript"]).read_text(encoding="utf-8"))
    assert [step["attempts"][-1]["model"] for step in transcript["steps"]] == ["deepseek-flash"] * 2


def test_a_routing_configuration_is_for_the_routed_system_and_its_sets_as_they_were(tmp_path, monkeypatch,
                                                                                   capsys):
    root = routed_root(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as stopped:
        run_e2e.main(["tasks.jsonl", "--system", "ft", "--out", "runs/x", "--no-tokenizer",
                      "--routing-config", ROUTING_M7], world=world(), local_client=FakeLlama([]).client())
    assert stopped.value.code == 2 and "--routing-config" in capsys.readouterr().err
    (root / "data" / "eval" / "v2" / "dev-v2.jsonl").write_text('{"id": "q-2"}\n', encoding="utf-8")
    code, out, fake, cloud = ftr_run(root, "e2e-changed", [unsure(GET_PAPER)], "--routing-config", ROUTING_M7)
    assert code == 2 and "development set 2 (data/eval/v2/dev-v2.jsonl)" in capsys.readouterr().err
    assert fake.requests == [] and cloud.requests == [] and not out.exists()


def test_a_routed_turn_the_transcript_shows_no_attempt_of_is_noted_in_the_last_step(tmp_path):
    # the cloud answers step 2 with two calls of one id: the loop ends the task as a harness error before it
    # records the attempt (agent.loop), so the last step of the transcript shows none, but the turn came back
    # and is noted in that step, where the timing sidecar places its call too (RouteNotes.note, TaskTimer)
    sure = turn(GET2, model="model:lora=1", paid=False)
    sure.signals = {"min_decision_prob": 0.9, "mean_logprob": -0.1, "hard_reason": None}
    unsure_call = turn(call("fetch_fulltext", "local_0", arxiv_id=ID2), model="model:lora=1", paid=False)
    unsure_call.signals = {"min_decision_prob": 0.2, "mean_logprob": -1.6, "hard_reason": None}
    twice = turn(call("get_paper", "c9", arxiv_id=ID1), call("get_paper", "c9", arxiv_id=ID2),
                 model="deepseek-flash")
    cloud = ScriptedModel([twice], name="deepseek-flash")
    cloud.cost = lambda answered: 0.25 if answered.paid else 0.0
    model = RoutedModel(ScriptedModel([sure, unsure_call], name="model:lora=1"), cloud,
                        RouteConfig("min_decision_prob", 0.5, RouteRules.from_json(P5_BUDGET)))
    out = tmp_path / "e2e-twice"
    code = run_e2e.main([str(write_tasks(tmp_path / "tasks.jsonl")), "--system", "ftr", "--out", str(out),
                         "--no-tokenizer", "--today", TODAY, "--routing-config", ROUTING_M7], model=model,
                        world=world())
    assert code == 0 and model.stepwise is True
    (record,) = json.loads((out / "report.json").read_text(encoding="utf-8"))["tasks"]
    transcript = json.loads((out / record["transcript"]).read_text(encoding="utf-8"))
    assert record["status"] == transcript["status"] == "harness_error"
    assert [len(step["attempts"]) for step in transcript["steps"]] == [1, 0]
    assert "duplicate tool call ids" in transcript["steps"][-1]["harness_error"]
    assert record["routing"] == {
        "routed_steps": [2], "local_turns": 1, "cloud_turns": 1,
        "routed": [{"step": 2, "reason": "min_decision_prob 0.2 < 0.5", "paid": True, "cost_rmb": 0.25}],
        "cloud_cost_rmb": 0.25}
    assert record["cost_rmb"] == 0.25
    (timed,) = json.loads((out / "timing.json").read_text(encoding="utf-8"))["tasks"]
    assert [c["step"] for c in timed["calls"]] == [1, 2]


# ---- M7 WP2, WP3 and WP4: --votes, --rules and --glossary (docs/m7-本地提升计划.md) ------------------------

VOLATILE = ("date", "git_commit", "git_dirty")  # the manifest's keys that differ from run to run
# What run_e2e wrote for LOOKUP in think mode, base seed 3, before --votes, --rules and --glossary existed
# (the code of 2026-10-30, before WP2 and WP3): the SHA-256 of the transcript, of the report with its manifest
# without VOLATILE as canonical JSON, and of the timing sidecar. Without the new flags they stay these bytes.
BEFORE_WP23 = {"transcript": "d279f33d2ec5fd294349141231e2b42d330049d87a61e102867afb80c664a01a",
               "report": "87148fd7a5c6a6c4a77651c41df0a4a46d85f4c8037c651ae52dbc09568fcf01",
               "timing": "3494d86fae818e17c901df492768128ebd7fb505bb28ec85c6dce9a01b6df2d8"}


def wp23_run(tmp_path: Path, monkeypatch, script: list[Scripted], *flags: str,
             tool_format: object | None = None) -> tuple[int, dict]:
    """b0 over LOOKUP in think mode with base seed 3 on a family whose think mode samples (in tool_format,
    when given), from tmp_path (so the manifest holds relative paths), the clock ticking 0.5 s; (the exit,
    {"transcript", "report", "timing": the outputs' bytes, the report's manifest without VOLATILE,
    "requests": the server's})."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    install_fake_family(monkeypatch, tmp_path / "model",
                        decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY},
                        **({} if tool_format is None else {"tool_format": tool_format}))
    write_tasks(tmp_path / "tasks.jsonl")
    fake = FakeLlama(script)
    code = run_e2e.main(["tasks.jsonl", "--system", "b0", "--out", "runs/e2e-wp23", "--tokenizer", "model",
                         "--mode", "think", "--base-seed", "3", "--reasoning-budget", "32",
                         "--training-length", "5120", "--today", TODAY, *flags],
                        world=world(), local_client=fake.client(), clock=ticking(0.5))
    out = tmp_path / "runs" / "e2e-wp23"
    if code != 0:
        return code, {"requests": fake.requests}
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    report["manifest"] = {k: v for k, v in report["manifest"].items() if k not in VOLATILE}
    return code, {"transcript": (out / "transcripts" / "t-lookup.json").read_bytes(),
                  "report": canonical_json(report).encode("utf-8"),
                  "timing": (out / "timing.json").read_bytes(),
                  "requests": fake.requests}


def digests(outputs: dict) -> dict:
    return {k: hashlib.sha256(outputs[k]).hexdigest() for k in ("transcript", "report", "timing")}


GET_PAPER_CALL = '<tool_call>\n{"name": "get_paper", "arguments": {"arxiv_id": "' + ID2 + '"}}\n</tool_call>'
LOOKUP_SCRIPT = [Scripted(("Look it up.", "\n</think>\n\n", GET_PAPER_CALL)),
                 Scripted(("Now answer.", "\n</think>\n\n", ANSWER))]


def test_without_the_new_flags_every_output_is_the_bytes_before_them(tmp_path, monkeypatch):
    code, outputs = wp23_run(tmp_path, monkeypatch, list(LOOKUP_SCRIPT))
    assert code == 0 and digests(outputs) == BEFORE_WP23


@pytest.mark.parametrize(("options", "message"), [
    (["--system", "b2", "--votes", "3"], "wrap the local systems only"),
    (["--system", "b3", "--rules"], "wrap the local systems only"),
    (["--system", "ft", "--vote-mode", "reproduction"], "--vote-mode chooses the keys of a vote"),
    (["--system", "ft", "--votes", "3", "--greedy"], "a vote's samples sample")])
def test_the_wrapper_flags_are_for_the_local_systems(options, message, tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        run_e2e.main([str(write_tasks(tmp_path / "tasks.jsonl")), *options, "--out", str(tmp_path / "out"),
                      "--no-tokenizer"], world=world(), local_client=FakeLlama([]).client())
    assert stopped.value.code == 2 and message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


GET_PAPER_ID1 = '<tool_call>\n{"name": "get_paper", "arguments": {"arxiv_id": "' + ID1 + '"}}\n</tool_call>'


def test_every_step_of_a_voted_ruled_run_records_its_vote_and_the_rules_and_its_requests_time(tmp_path,
                                                                                               monkeypatch):
    from research_agent.agent.rules import RULES, RULES_VERSION

    # step 1: the greedy candidate and sample 0 look up ID2, sample 1 ID1: a tie of one against one, which the
    # greedy answer's class wins, sample 0 its representative; step 2: all three answer in text
    script = [Scripted(("G.", "\n</think>\n\n", GET_PAPER_CALL)),
              Scripted(("S0.", "\n</think>\n\n", GET_PAPER_CALL)),
              Scripted(("S1.", "\n</think>\n\n", GET_PAPER_ID1)),
              *[Scripted(("A.", "\n</think>\n\n", ANSWER))] * 3]
    code, outputs = wp23_run(tmp_path, monkeypatch, script, "--votes", "2", "--rules")
    assert code == 0 and len(outputs["requests"]) == 6
    prompt = outputs["requests"][0]["prompt"]
    assert [r.get("seed") for r in outputs["requests"][:3]] == [None, request_seed(3, prompt, 1, 0),
                                                              request_seed(4, prompt, 1, 0)]
    transcript = json.loads(outputs["transcript"])
    first, second = (step["attempts"][-1] for step in transcript["steps"])
    assert (first["votes"]["k"], first["votes"]["representative"], first["votes"]["mode"]) == (2, 0, "final")
    assert first["reasoning"] == "S0." and first["rules"] == [] and second["votes"]["chosen"] == "TEXT"
    assert transcript["status"] == "answered" and "votes" not in json.dumps(transcript["messages"])
    report = json.loads(outputs["report"])
    manifest = report["manifest"]
    assert manifest["generation"]["votes"] == 2 and manifest["vote_mode"] == "final"
    assert manifest["rules"] == {"version": RULES_VERSION, "names": list(RULES)}
    assert manifest["model"] == "model:lora=0+rules+votes2" and report["summary"]["success_rate"] == 1.0
    assert report["summary"]["candidate_errors"] == {}  # voting: counted, and no candidate failed
    (timed,) = json.loads(outputs["timing"])["tasks"]
    step_one = [c for c in timed["calls"] if c["step"] == 1]
    assert len(step_one) == 1 and step_one[0]["server"]["predicted_n"] == 9  # three requests of three tokens


class BreakingFormat(QwenJson):
    """Qwen3's tool format, whose parse fails on an answer that says BROKEN: a failure LocalModel raises as a
    HarnessError (agent.models), which alone ends an end-to-end task as harness_error (agent.loop)."""

    def parse(self, text: str, tools: list[dict] | None) -> list[dict]:
        if "BROKEN" in text:
            raise ValueError("a broken block")
        return super().parse(text, tools)


BROKEN_ERROR = {"error_type": "HarnessError",
                "message": "parse_tool_calls failed: ValueError('a broken block')"}


def test_a_candidate_that_raises_a_harness_error_leaves_the_task_going_and_the_report_counts_it(tmp_path,
                                                                                                monkeypatch,
                                                                                                capsys):
    # alone, the answer the family cannot parse ends the task as harness_error, and nothing counts candidates
    code, outputs = wp23_run(tmp_path / "alone", monkeypatch, [Scripted(("X.", "\n</think>\n\n", "BROKEN"))],
                             tool_format=BreakingFormat())
    report = json.loads(outputs["report"])
    assert code == 0 and report["summary"]["statuses"] == {"harness_error": 1}
    assert "candidate_errors" not in report["summary"]
    capsys.readouterr()
    # in a vote, sample 0's answer raises that HarnessError; the vote goes on over the greedy candidate and
    # sample 1, which look ID2 up, and the task goes on to its answer
    script = [Scripted(("G.", "\n</think>\n\n", GET_PAPER_CALL)),
              Scripted(("S0.", "\n</think>\n\n", "BROKEN")),
              Scripted(("S1.", "\n</think>\n\n", GET_PAPER_CALL)),
              *[Scripted(("A.", "\n</think>\n\n", ANSWER))] * 3]
    code, outputs = wp23_run(tmp_path / "voted", monkeypatch, script, "--votes", "2",
                             tool_format=BreakingFormat())
    assert code == 0 and len(outputs["requests"]) == 6
    transcript = json.loads(outputs["transcript"])
    assert transcript["status"] == "answered" and transcript["final_text"] == ANSWER
    step = transcript["steps"][0]
    assert "harness_error" not in step and step["executed"][0]["name"] == "get_paper"
    (attempt,) = step["attempts"]
    assert attempt["votes"]["errors"] == [{"candidate": 0, **BROKEN_ERROR}]
    assert (attempt["votes"]["representative"], attempt["reasoning"]) == (1, "S1.")
    assert "errors" not in transcript["steps"][1]["attempts"][0]["votes"]
    report = json.loads(outputs["report"])
    assert report["summary"]["candidate_errors"] == {"HarnessError": 1}  # the report counts it
    assert report["summary"]["statuses"] == {"answered": 1} and report["summary"]["success_rate"] == 1.0
    assert "failed vote candidates: HarnessError 1" in capsys.readouterr().out

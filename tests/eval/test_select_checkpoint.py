"""scripts/select_checkpoint.py offline: a fake training run in a fake repository root, a fake llama-server
(process and HTTP), and the real scripts/run_eval.py and scripts/score.py run in process where the script
would start them. Nothing starts llama-server or loads a model."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from support.fake_llama import ServerProcess
from support.llama_fakes import fake_loaded_family
from train.tokenizer_fake import FakeTokenizer

from research_agent.agent.prompts import system_prompt
from research_agent.eval.lock import sha256_of, write_lock
from research_agent.family.core import Mode
from research_agent.llm.generation import GenerationSettings
from research_agent.llm.local import LocalClient
from research_agent.train.samples import prompt_renderer

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


select = load_script("select_checkpoint")
run_eval = load_script("run_eval")
score = load_script("score")

SYSTEM = {"role": "system", "content": system_prompt("2026-10-10")}
SEARCH = {"name": "search_arxiv", "arguments": {"query": "hallucination", "max_results": 5},
          "arguments_json": "{\"query\": \"hallucination\", \"max_results\": 5}"}


def item(rid: str, user: str, expected: dict, type_: str, lang: str) -> dict:
    return {"id": rid, "messages": [SYSTEM, {"role": "user", "content": user}], "expected": expected,
            "type": type_, "lang": lang, "family": f"f/{type_}", "split": "dev", "slots": {},
            "source": {"request": rid, "step": 1}}


FIND = {"content": "", "tool_calls": [SEARCH]}
ITEMS = [item("q-1", "找几篇幻觉(hallucination)的论文。", FIND, "call", "zh"),
         item("q-2", "Thanks, that's all for today.", {"content": "See you!", "tool_calls": []}, "no_call",
              "en"),
         item("q-3", "Find hallucination papers.", FIND, "call", "en")]
CALL = ('<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "hallucination", "max_results": 5}}\n'
        '</tool_call>')  # right on q-1 and q-3: 2 of 3
BYE = "See you!"  # right on q-2 only: 1 of 3
READY = "main: server is listening on http://127.0.0.1:8080 - starting the main loop\n"


class FakeLlama:
    """llama-server for select_checkpoint: popen starts it with the adapter its --lora names over the model
    its -m names; over HTTP it answers /health, /lora-adapters (listed(path) of that adapter), /props (the
    build, the slots and, as model_path, the model as the server was given it) and every completion with the
    text scripted for the adapter's step (a status code instead of a text fails the request; a dict scripts
    each variant by name, "" being the default run). run stands in for subprocess.run and runs run_eval.py and
    score.py in process, run_eval's client talking to this server; export_adapter.py writes the GGUF its
    --outfile names, as the converter would."""

    def __init__(self, answers: dict[int, str | int | dict[str, str | int]], *, listed=lambda path: path,
                 exit_code: int | None = None, already_running: bool = False, slots: int = 1) -> None:
        self.answers, self.listed, self.exit_code, self.slots = answers, listed, exit_code, slots
        self.adapter: str | None = None
        self.model: str | None = None
        self.variant = ""  # the variant whose run_eval is running: the suffix of its --out stem
        self.process: ServerProcess | None = ServerProcess([]) if already_running else None
        self.processes: list[ServerProcess] = []
        self.commands: list[list[str]] = []
        self.payloads: list[dict] = []
        self.runs: list[list[str]] = []  # every command given to run, as given

    def popen(self, cmd, **kwargs):
        self.commands.append(list(cmd))
        self.adapter, self.model = cmd[cmd.index("--lora") + 1], cmd[cmd.index("-m") + 1]
        self.process = ServerProcess([] if self.exit_code is not None else [READY], exit_code=self.exit_code)
        self.processes.append(self.process)
        return self.process

    def handler(self, request: httpx.Request) -> httpx.Response:
        if self.process is None or self.process.returncode is not None:
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/lora-adapters":
            return httpx.Response(200, json=[{"id": 0, "path": self.listed(self.adapter), "scale": 1.0}])
        if path == "/props":
            return httpx.Response(200, json={"build_info": "b11349-fb4b2737a", "total_slots": self.slots,
                                             "model_path": self.model})
        self.payloads.append(json.loads(request.content))
        answer = self.answers[int(re.search(r"checkpoint-(\d+)-f16", self.adapter).group(1))]
        if isinstance(answer, dict):
            answer = answer[self.variant]
        if isinstance(answer, int):
            return httpx.Response(answer, json={"error": "the server failed"})
        return httpx.Response(200, json={"choices": [{"text": answer, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 900, "completion_tokens": 20},
                                         "timings": {"predicted_ms": 20.0}})

    def client(self, url: str) -> LocalClient:
        return LocalClient(url, transport=httpx.MockTransport(self.handler))

    def run(self, cmd, **kwargs) -> subprocess.CompletedProcess:
        self.runs.append(list(cmd))
        script, argv = Path(cmd[1]).name, list(cmd[2:])
        if script == "run_eval.py":
            at = argv.index("--tokenizer")
            argv[at:at + 2] = ["--no-tokenizer"]  # the fake tokenizer renders the prompt instead
            stem = Path(argv[argv.index("--out") + 1]).stem  # checkpoint-N or checkpoint-N.NAME
            self.variant = stem.partition(".")[2]
            code = run_eval.main(argv, local_client=self.client(argv[argv.index("--server") + 1]),
                                 render=prompt_renderer(FakeTokenizer()))
        elif script == "export_adapter.py":
            outfile = Path(argv[argv.index("--outfile") + 1])
            outfile.parent.mkdir(parents=True, exist_ok=True)
            outfile.write_bytes(f"GGUF exported from {argv[0]}".encode())
            code = 0
        else:
            assert script == "score.py", cmd
            code = score.main(argv)
        return subprocess.CompletedProcess(cmd, code)


def failing(*args, **kwargs):
    raise AssertionError("nothing may be started")


def real(what: str):
    """A stand-in for the real what that fails at once (no_real_server)."""
    def refuse(*args, **kwargs):
        raise AssertionError(f"a test reached for the real {what}: inject a fake one")
    return refuse


@pytest.fixture(autouse=True)
def no_real_server(monkeypatch):
    """No test in this file reaches a real llama-server (127.0.0.1:8080 may serve another job): main's
    defaults for client, popen and run, bound when main was defined, become stand-ins that fail at once, so a
    test that injects none fails before anything is sent or started; a test that injects its own uses it."""
    for name, what in (("client", "LocalClient"), ("popen", "subprocess.Popen"), ("run", "subprocess.run")):
        monkeypatch.setitem(select.main.__kwdefaults__, name, real(what))


def training_run(run: Path, steps: tuple[int, ...]) -> None:
    """A finished training run (fake) with a checkpoint at every step, written in the order given: the GGUF of
    each, the trainer's checkpoint folders, the final adapter and result.json naming the last step."""
    (run / "gguf").mkdir(parents=True)
    for step in steps:
        (run / "gguf" / f"checkpoint-{step}-f16.gguf").write_bytes(f"GGUF step {step}".encode())
        (run / "trainer" / f"checkpoint-{step}").mkdir(parents=True)
        (run / "trainer" / f"checkpoint-{step}" / "trainer_state.json").write_text("{}", encoding="utf-8")
    (run / "gguf" / "adapter-f16.gguf").write_bytes(f"GGUF step {max(steps)}, final".encode())
    (run / "result.json").write_text(json.dumps({"steps": max(steps)}), encoding="utf-8")


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A repository root with a finished run runs/train/ft (checkpoints 10, 20, 30 and the final adapter, the
    trainer's checkpoint folders), the locked development set, the base model and a llama-server binary (all
    fake), as the current folder."""
    monkeypatch.chdir(tmp_path)
    for module in (select, run_eval):
        monkeypatch.setattr(module, "REPO", tmp_path)
    monkeypatch.setattr(select, "git_commit", lambda repo: "c0ffee")
    monkeypatch.setattr(select, "git_dirty", lambda repo: False)
    training_run(tmp_path / "runs" / "train" / "ft", (20, 10, 30))
    dev = tmp_path / "data" / "eval" / "dev.jsonl"
    dev.parent.mkdir(parents=True)
    dev.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in ITEMS), encoding="utf-8")
    write_lock(dev)
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "Qwen3-1.7B-Q8_0.gguf").write_bytes(b"GGUF base")
    (tmp_path / "llama-server.exe").write_bytes(b"")
    return tmp_path


ARGV = ["runs/train/ft", "--llama-server", "llama-server.exe"]


def run_select(llama: FakeLlama, *extra: str) -> int:
    return select.main([*ARGV, *extra], popen=llama.popen, run=llama.run, client=llama.client)


def test_the_checkpoints_are_ordered_by_step_and_the_final_adapter_is_an_alias_not_a_candidate(root):
    checkpoints, final = select.find_checkpoints(Path("runs/train/ft"))
    gguf = Path("runs/train/ft/gguf")
    expected = [(step, gguf / f"checkpoint-{step}-f16.gguf") for step in (10, 20, 30)]
    assert [(c.step, c.gguf) for c in checkpoints] == expected
    assert [c.sha256 for c in checkpoints] == [sha256_of(c.gguf) for c in checkpoints]
    assert final == gguf / "adapter-f16.gguf"


@pytest.mark.parametrize("steps", [(9, 10, 100), (209, 1045, 1668)])
def test_checkpoints_go_by_their_step_number_not_by_their_file_name(root, steps):
    """Steps whose file names sort in another order than their numbers (checkpoint-10 before checkpoint-9,
    checkpoint-1045 before checkpoint-209): the candidates, the tie-break to the later step and the final
    adapter's alias all follow the numbers."""
    run = root / "runs" / "train" / "b4"
    training_run(run, (steps[1], steps[2], steps[0]))
    names = sorted(path.name for path in (run / "gguf").glob("checkpoint-*"))
    assert [int(re.search(r"\d+", name).group()) for name in names] != list(steps)  # the premise
    checkpoints, final = select.find_checkpoints(Path("runs/train/b4"))
    assert [c.step for c in checkpoints] == list(steps)
    assert final == Path("runs/train/b4/gguf/adapter-f16.gguf")
    llama = FakeLlama(dict.fromkeys(steps, CALL))  # 2/3 everywhere: a three-way tie
    argv = ["runs/train/b4", "--llama-server", "llama-server.exe"]
    assert select.main(argv, popen=llama.popen, run=llama.run, client=llama.client) == 0
    selection = json.loads((root / "results" / "checkpoint-selection-b4.json").read_text(encoding="utf-8"))
    assert [c["step"] for c in selection["checkpoints"]] == list(steps)
    assert selection["chosen"]["step"] == steps[-1]
    assert selection["final_adapter"]["alias_of_step"] == steps[-1]


def test_an_unfinished_or_partly_exported_run_is_refused(root, capsys):
    run = root / "runs" / "train" / "ft"
    (run / "gguf" / "checkpoint-20-f16.gguf").unlink()  # a failed export: the trainer kept step 20
    with pytest.raises(select.RunError, match=r"\[20\] have no GGUF"):
        select.find_checkpoints(run)
    assert select.main([*ARGV, "--dry-run"], popen=failing, run=failing) == 1
    assert "have no GGUF" in capsys.readouterr().out
    (run / "result.json").write_text(json.dumps({"steps": 40}), encoding="utf-8")
    with pytest.raises(select.RunError, match="40 steps"):
        select.find_checkpoints(run)
    (run / "result.json").unlink()
    with pytest.raises(select.RunError, match="has not finished"):
        select.find_checkpoints(run)
    with pytest.raises(select.RunError, match="no checkpoint"):
        select.find_checkpoints(root / "runs" / "train" / "missing")


def test_dry_run_lists_the_checkpoints_and_the_commands_and_starts_and_writes_nothing(root, capsys):
    assert select.main([*ARGV, "--dry-run"], popen=failing, run=failing) == 0
    out = capsys.readouterr().out
    for step in (10, 20, 30):
        gguf = f"runs/train/ft/gguf/checkpoint-{step}-f16.gguf"
        assert f"step {step}" in out and sha256_of(gguf) in out
        assert f"--lora {gguf} --lora-init-without-apply" in out
        assert f"--out runs/eval/select-ft/checkpoint-{step}.jsonl" in out
        assert f"--out runs/eval/select-ft/checkpoint-{step}.score.json" in out
    assert "adapter-f16.gguf" in out and sha256_of("runs/train/ft/gguf/adapter-f16.gguf") in out
    assert "run_eval.py data/eval/dev.jsonl --system ft" in out and "score.py" in out
    assert not (root / "results").exists() and not (root / "runs" / "eval").exists()


def test_the_best_rate_wins_a_tie_goes_to_the_later_step_and_every_server_is_stopped(root, capsys):
    llama = FakeLlama({10: CALL, 20: BYE, 30: CALL})  # 2/3, 1/3, 2/3: steps 10 and 30 tie
    assert run_select(llama) == 0
    selection = json.loads((root / "results" / "checkpoint-selection-ft.json").read_text(encoding="utf-8"))
    assert [(c["step"], c["successes"], c["n"]) for c in selection["checkpoints"]] == [(10, 2, 3), (20, 1, 3),
                                                                                       (30, 2, 3)]
    assert selection["chosen"]["step"] == 30 and selection["chosen"]["rate"] == 2 / 3
    assert selection["chosen"]["sha256"] == sha256_of("runs/train/ft/gguf/checkpoint-30-f16.gguf")
    assert selection["final_adapter"] == {"gguf": "runs/train/ft/gguf/adapter-f16.gguf",
                                          "sha256": sha256_of("runs/train/ft/gguf/adapter-f16.gguf"),
                                          "alias_of_step": 30}
    assert (selection["run"], selection["system"], selection["lora_scale"]) == ("runs/train/ft", "ft", 1.0)
    assert selection["items"] == "data/eval/dev.jsonl"
    assert selection["items_sha256"] == sha256_of("data/eval/dev.jsonl")
    assert selection["git_commit"] == "c0ffee" and "tie" in selection["rule"]
    first = selection["checkpoints"][0]
    assert first["records"] == "runs/eval/select-ft/checkpoint-10.jsonl" and first["statuses"] == {"ok": 3}
    assert first["server_build"] == "b11349-fb4b2737a"
    assert all(p.terminated for p in llama.processes) and len(llama.processes) == 3  # one server at a time
    assert all(p["lora"] == [{"id": 0, "scale": 1.0}] for p in llama.payloads) and len(llama.payloads) == 9
    for step in (10, 20, 30):
        stem = root / "runs" / "eval" / "select-ft" / f"checkpoint-{step}"
        for suffix in (".jsonl", ".manifest.json", ".timing.json", ".score.json", ".server.log"):
            assert stem.with_name(stem.name + suffix).is_file(), suffix
    text = (root / "results" / "checkpoint-selection-ft.json").read_text(encoding="utf-8")
    assert str(root) not in text and "\\" not in text  # repository-relative POSIX paths only
    assert "chosen: step 30" in capsys.readouterr().out


def test_the_selection_file_is_byte_identical_on_a_rerun(root):
    out = root / "results" / "checkpoint-selection-ft.json"
    assert run_select(FakeLlama({10: BYE, 20: CALL, 30: BYE})) == 0
    first = out.read_bytes()
    assert run_select(FakeLlama({10: BYE, 20: CALL, 30: BYE})) == 0
    assert out.read_bytes() == first and json.loads(first)["chosen"]["step"] == 20


def test_every_server_is_started_with_one_slot_and_a_server_with_more_stops_the_selection(root, capsys):
    # with several slots the greedy output depends on the request history (the M3 finding), so the
    # selection serves one slot and checks /props before it sends anything
    assert select.main([*ARGV, "--dry-run"], popen=failing, run=failing) == 0
    assert "-np 1" in capsys.readouterr().out
    llama = FakeLlama({10: CALL, 20: CALL, 30: CALL}, slots=4)
    assert run_select(llama) == 3
    assert "4 slots" in capsys.readouterr().out and llama.payloads == []
    assert len(llama.processes) == 1 and llama.processes[0].terminated
    assert all(cmd[cmd.index("-np") + 1] == "1" for cmd in llama.commands)
    assert not (root / "results").exists()
    llama = FakeLlama({10: CALL, 20: CALL, 30: CALL})
    assert run_select(llama) == 0
    selection = json.loads((root / "results" / "checkpoint-selection-ft.json").read_text(encoding="utf-8"))
    assert selection["server_slots"] == 1 and "-np" in selection["server_args"]


def test_a_server_that_lists_another_adapter_stops_the_selection_with_exit_3(root, capsys):
    other = "runs/train/b4/gguf/adapter-f16.gguf"
    llama = FakeLlama({10: CALL, 20: CALL, 30: CALL}, listed=lambda path: other)
    assert run_select(llama) == 3
    assert "runs/train/b4/gguf/adapter-f16.gguf" in capsys.readouterr().out
    assert len(llama.processes) == 1 and llama.processes[0].terminated
    assert not (root / "results").exists() and llama.payloads == []


def test_a_busy_port_exits_2_and_a_server_that_never_gets_ready_exits_3(root, capsys):
    busy = FakeLlama({10: CALL}, already_running=True)
    assert run_select(busy) == 2 and busy.processes == []
    assert "already answers" in capsys.readouterr().out
    dead = FakeLlama({10: CALL}, exit_code=1)
    assert run_select(dead) == 3 and len(dead.processes) == 1
    assert "exited with 1" in capsys.readouterr().out
    assert not (root / "results").exists()


def test_errors_the_model_did_not_cause_stop_the_selection_with_exit_6(root, capsys):
    llama = FakeLlama({10: CALL, 20: 500, 30: CALL})  # step 20's server fails every completion
    assert run_select(llama) == 6
    assert "HTTPStatusError" in capsys.readouterr().out
    assert not (root / "results").exists()


def test_the_items_must_be_a_locked_development_set_and_never_a_test_set(root, capsys):
    test_set = root / "data" / "eval" / "test.jsonl"
    test_set.write_text("not json: never parsed\n", encoding="utf-8")
    argv = [*ARGV, "--items", "data/eval/test.jsonl", "--dry-run"]
    assert select.main(argv, popen=failing, run=failing) == 4
    assert "locked test set" in capsys.readouterr().out
    for name in ("handwritten.jsonl", "handwritten-specs.jsonl"):  # the test set's handwritten items
        (root / "data" / "eval" / name).write_bytes((root / "data" / "eval" / "dev.jsonl").read_bytes())
        argv = [*ARGV, "--items", f"data/eval/{name}", "--dry-run"]
        assert select.main(argv, popen=failing, run=failing) == 4, name
        assert "handwritten items with their gold turns" in capsys.readouterr().out
    dev = root / "data" / "eval" / "dev.jsonl"
    dev.write_bytes(dev.read_bytes() + b"\n")
    assert select.main([*ARGV, "--dry-run"], popen=failing, run=failing) == 5
    assert "does not match its lock" in capsys.readouterr().out


def test_locked_items_that_are_test_material_are_refused_by_their_items_with_exit_4(root, capsys):
    """A copy of test items with a lock of its own passes the lock check; the items it holds (split "test")
    refuse it before anything starts."""
    copy = root / "data" / "eval" / "copy.jsonl"
    copy.write_text("".join(json.dumps({**i, "split": "test"}, ensure_ascii=False) + "\n" for i in ITEMS),
                    encoding="utf-8")
    write_lock(copy)
    argv = [*ARGV, "--items", "data/eval/copy.jsonl", "--dry-run"]
    assert select.main(argv, popen=failing, run=failing) == 4
    assert "holds test material" in capsys.readouterr().out


def test_the_system_comes_from_the_run_name_or_must_be_given(root, capsys):
    (root / "runs" / "train" / "ft").rename(root / "runs" / "train" / "smoke-m3")
    argv = ["runs/train/smoke-m3", "--llama-server", "llama-server.exe", "--dry-run"]
    assert select.main(argv, popen=failing, run=failing) == 2
    assert "--system" in capsys.readouterr().out
    assert select.main([*argv, "--system", "b4"], popen=failing, run=failing) == 0
    assert "--system b4" in capsys.readouterr().out


def test_without_a_llama_server_binary_the_selection_stops_with_exit_2(root, capsys, monkeypatch):
    monkeypatch.delenv("LLAMA_SERVER", raising=False)
    assert select.main(["runs/train/ft", "--dry-run"], popen=failing, run=failing) == 2
    assert "LLAMA_SERVER" in capsys.readouterr().out
    assert select.main(["runs/train/ft", "--llama-server", "missing.exe", "--dry-run"], popen=failing,
                       run=failing) == 2


# ---- M7 WP-E: --checkpoints, --variant, --select-by and the family check (docs/m7-计划.md, WP-E) --------

SELECTION_KEYS = {"run", "system", "lora_scale", "rule", "items", "items_sha256", "base_model", "server_args",
                  "server_slots", "git_commit", "git_dirty", "checkpoints", "final_adapter", "chosen"}
ROW_KEYS = {"step", "gguf", "sha256", "n", "successes", "rate", "statuses", "server_build", "records"}


def partial_run(run: Path, steps: tuple[int, ...], exported: tuple[int, ...]) -> None:
    """A training run still in progress (fake): the trainer's checkpoint folders of every step, the GGUF
    of the exported ones only, no result.json and no final adapter."""
    (run / "gguf").mkdir(parents=True, exist_ok=True)
    for step in steps:
        (run / "trainer" / f"checkpoint-{step}").mkdir(parents=True)
        (run / "trainer" / f"checkpoint-{step}" / "trainer_state.json").write_text("{}", encoding="utf-8")
        if step in exported:
            (run / "gguf" / f"checkpoint-{step}-f16.gguf").write_bytes(f"GGUF step {step}".encode())


def selection_of(root: Path, name: str = "ft") -> dict:
    return json.loads((root / "results" / f"checkpoint-selection-{name}.json").read_text(encoding="utf-8"))


def test_without_the_new_flags_the_selection_file_holds_exactly_todays_keys(root):
    assert run_select(FakeLlama({10: CALL, 20: BYE, 30: CALL})) == 0
    selection = selection_of(root)
    assert set(selection) == SELECTION_KEYS and selection["rule"] == select.RULE
    assert all(set(row) == ROW_KEYS for row in selection["checkpoints"])


def test_checkpoints_scores_the_listed_steps_without_result_json_and_exports_a_missing_gguf(root, capsys):
    run = root / "runs" / "train" / "ft3"
    partial_run(run, (10, 20, 30), exported=(10, 30))
    argv = ["runs/train/ft3", "--system", "ft", "--llama-server", "llama-server.exe", "--checkpoints",
            "10,20"]
    assert select.main([*argv, "--dry-run"], popen=failing, run=failing) == 0
    out = capsys.readouterr().out
    assert "step 10" in out and "step 20" in out and "step 30" not in out
    assert "to be exported" in out and "export_adapter.py runs/train/ft3/trainer/checkpoint-20" in out
    assert "--outfile runs/train/ft3/gguf/checkpoint-20-f16.gguf" in out
    assert not (run / "gguf" / "checkpoint-20-f16.gguf").exists()  # the dry run exports nothing
    llama = FakeLlama({10: BYE, 20: CALL})
    assert select.main(argv, popen=llama.popen, run=llama.run, client=llama.client) == 0
    exported = run / "gguf" / "checkpoint-20-f16.gguf"
    assert exported.is_file() and len(llama.processes) == 2
    export = next(cmd for cmd in llama.runs if Path(cmd[1]).name == "export_adapter.py")
    assert export[2:] == ["runs/train/ft3/trainer/checkpoint-20", "--outfile",
                          "runs/train/ft3/gguf/checkpoint-20-f16.gguf"]
    selection = selection_of(root, "ft3")
    assert selection["partial"] is True and selection["checkpoints_listed"] == [10, 20]
    assert [(c["step"], c["successes"]) for c in selection["checkpoints"]] == [(10, 1), (20, 2)]
    assert selection["checkpoints"][1]["sha256"] == sha256_of(exported)
    assert selection["chosen"]["step"] == 20 and selection["final_adapter"] is None
    assert "partial: 2 of the run's checkpoints" in capsys.readouterr().out


def test_checkpoints_passes_the_llama_cpp_source_tree_to_the_export_and_refuses_an_unknown_step(root, capsys):
    run = root / "runs" / "train" / "ft3"
    partial_run(run, (10, 20), exported=())
    llama = FakeLlama({10: CALL, 20: CALL})
    argv = ["runs/train/ft3", "--system", "ft", "--llama-server", "llama-server.exe", "--checkpoints", "20",
            "--llama-cpp-src", "C:/tools/llama.cpp/src"]
    assert select.main(argv, popen=llama.popen, run=llama.run, client=llama.client) == 0
    (export,) = [cmd for cmd in llama.runs if Path(cmd[1]).name == "export_adapter.py"]
    assert export[-2:] == ["--llama-cpp-src", "C:/tools/llama.cpp/src"]
    assert selection_of(root, "ft3")["checkpoints_listed"] == [20]
    assert select.main([*argv[:-2], "--checkpoints", "10,40", "--dry-run"], popen=failing, run=failing) == 1
    assert "checkpoint-40" in capsys.readouterr().out
    assert select.main([*argv[:-2], "--checkpoints", "10,x"], popen=failing, run=failing) == 2
    assert select.main([*argv[:-2], "--checkpoints", "10,10"], popen=failing, run=failing) == 2


def test_variants_run_run_eval_with_their_arguments_and_select_by_mean_chooses_by_their_mean(root):
    answers = {10: {"s0": CALL, "s1": BYE}, 20: {"s0": CALL, "s1": CALL}, 30: {"s0": BYE, "s1": BYE}}
    llama = FakeLlama(answers)
    argv = [*ARGV, "--variant", "s0=--mode think --seed 0", "--variant", "s1=--mode think --seed 1",
            "--select-by", "mean:s0,s1"]
    assert select.main(argv, popen=llama.popen, run=llama.run, client=llama.client) == 0
    evals = [cmd for cmd in llama.runs if Path(cmd[1]).name == "run_eval.py"]
    assert len(evals) == 6 and len(llama.processes) == 3  # one server per checkpoint, both variants on it
    first = evals[0]
    assert first[first.index("--out") + 1] == "runs/eval/select-ft/checkpoint-10.s0.jsonl"
    assert first[-4:] == ["--mode", "think", "--seed", "0"]
    assert evals[1][-4:] == ["--mode", "think", "--seed", "1"]
    stem = root / "runs" / "eval" / "select-ft" / "checkpoint-10"
    for suffix in (".s0.jsonl", ".s0.manifest.json", ".s0.timing.json", ".s0.score.json", ".s1.score.json"):
        assert stem.with_name(stem.name + suffix).is_file(), suffix
    assert not stem.with_name("checkpoint-10.jsonl").exists()  # --select-by: the default run is not made
    selection = selection_of(root)
    assert selection["selection"] == {"metric": "mean", "variants": ["s0", "s1"]}
    assert selection["variants"] == [{"name": "s0", "args": ["--mode", "think", "--seed", "0"]},
                                     {"name": "s1", "args": ["--mode", "think", "--seed", "1"]}]
    rows = {row["step"]: row for row in selection["checkpoints"]}
    assert rows[10]["variants"]["s0"] == {"n": 3, "successes": 2, "rate": 2 / 3, "statuses": {"ok": 3},
                                          "records": "runs/eval/select-ft/checkpoint-10.s0.jsonl"}
    assert rows[10]["variants"]["s1"]["successes"] == 1
    assert (rows[10]["successes"], rows[20]["successes"], rows[30]["successes"]) == (1.5, 2, 1)
    assert rows[10]["rate"] == 0.5 and "records" not in rows[10] and "statuses" not in rows[10]
    assert selection["chosen"]["step"] == 20 and selection["chosen"]["successes"] == 2
    assert "mean" in selection["rule"]


def test_variants_without_select_by_are_reported_beside_the_default_run_which_still_selects(root):
    llama = FakeLlama({10: {"": CALL, "greedy": BYE}, 20: {"": BYE, "greedy": CALL},
                       30: {"": BYE, "greedy": CALL}})
    argv = [*ARGV, "--variant", "greedy=--greedy"]
    assert select.main(argv, popen=llama.popen, run=llama.run, client=llama.client) == 0
    evals = [cmd for cmd in llama.runs if Path(cmd[1]).name == "run_eval.py"]
    assert len(evals) == 6 and evals[0][evals[0].index("--out") + 1].endswith("checkpoint-10.jsonl")
    second = evals[1]
    assert second[-1] == "--greedy"
    assert second[second.index("--out") + 1].endswith("checkpoint-10.greedy.jsonl")
    selection = selection_of(root)
    assert "selection" not in selection and selection["rule"] == select.RULE
    rows = {row["step"]: row for row in selection["checkpoints"]}
    assert set(rows[10]) == ROW_KEYS | {"variants"} and rows[10]["successes"] == 2
    assert rows[10]["variants"] == {"greedy": {"n": 3, "successes": 1, "rate": 1 / 3, "statuses": {"ok": 3},
                                               "records": "runs/eval/select-ft/checkpoint-10.greedy.jsonl"}}
    assert selection["chosen"]["step"] == 10  # the default run's rate chooses: 2/3 against 1/3 and 1/3


@pytest.mark.parametrize("extra", [
    ["--select-by", "mean:s0"],  # names a variant that is not given
    ["--variant", "s0=--mode think", "--select-by", "median:s0"],  # not a metric
    ["--variant", "s0=--mode think", "--select-by", "mean:s0,s0"],  # a variant twice
    ["--variant", "bad name=--mode think"],  # a name that is no file-name piece
    ["--variant", "s0=--out elsewhere.jsonl"],  # an argument the selection sets itself
    ["--variant", "s0=--mode think", "--variant", "s0=--mode no_think"],  # a name twice
    ["--variant", "s0"]])  # no "="
def test_malformed_variants_and_select_by_are_usage_errors(root, capsys, extra):
    assert select.main([*ARGV, *extra, "--dry-run"], popen=failing, run=failing) == 2
    out = capsys.readouterr().out
    assert "--variant" in out or "--select-by" in out
    assert not (root / "results").exists()


def test_the_family_of_the_tokenizer_folder_must_be_the_run_manifests(root, capsys):
    run = root / "runs" / "train" / "ft"
    manifest = {"config": {"model": "models/Qwen3-4B-Thinking-2507",
                           "family": {"name": "qwen3-thinking-2507"}}}
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    seen = []

    def hook(model_dir):
        seen.append(model_dir)
        return fake_loaded_family(root / "fake-model")

    code = select.main([*ARGV, "--dry-run"], popen=failing, run=failing, load_family=hook)
    assert code == 2 and seen == ["models/Qwen3-1.7B"]
    out = capsys.readouterr().out
    assert "fake-hybrid" in out and "qwen3-thinking-2507" in out
    manifest["config"]["family"]["name"] = "fake-hybrid"
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert select.main([*ARGV, "--dry-run"], popen=failing, run=failing, load_family=hook) == 0
    assert "family fake-hybrid" in capsys.readouterr().out
    (run / "manifest.json").write_text(json.dumps({"config": {"model": "m"}}), encoding="utf-8")
    assert select.main([*ARGV, "--dry-run"], popen=failing, run=failing, load_family=failing) == 0


# ---- M7 WP-J: several development sets, --items repeated (docs/m7-计划.md WP-J and §5) -------------------

DEV, DEV_V2 = "data/eval/dev.jsonl", "data/eval/v2/dev-v2.jsonl"
SETS = ("--items", DEV, "--items", DEV_V2)
TEN = {"name": "search_arxiv", "arguments": {"query": "hallucination", "max_results": 10},
       "arguments_json": "{\"query\": \"hallucination\", \"max_results\": 10}"}
OTHER = ('<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "hallucination", "max_results": 10}}\n'
         '</tool_call>')  # right on the "ten" items alone, as CALL is on the "five" items and BYE on "bye"
KINDS = {"five": ("Find hallucination papers.", FIND, "call"),
         "ten": ("Find ten hallucination papers.", {"content": "", "tool_calls": [TEN]}, "call"),
         "bye": ("Thanks, that's all for today.", {"content": "See you!", "tool_calls": []}, "no_call")}


def make_set(prefix: str, counts: dict[str, int]) -> list[dict]:
    """counts[kind] items of each kind of KINDS, in that order, with the ids <prefix>-001 on and the two
    languages taking turns."""
    kinds = [kind for kind, count in counts.items() for _ in range(count)]
    return [item(f"{prefix}-{number:03d}", *KINDS[kind], ("en", "zh")[number % 2])
            for number, kind in enumerate(kinds, 1)]


def write_set(path: Path, items: list[dict]) -> None:
    """items as a locked set at path, written with LF: the same bytes, and so the same SHA-256, on every
    platform."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items).encode("utf-8"))
    write_lock(path)


def records_of(root: Path, name: str = "ft") -> list[str]:
    """The files under the selection's records folder, relative to it, in name order."""
    folder = root / "runs" / "eval" / f"select-{name}"
    return sorted(path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file())


@pytest.fixture
def two_sets(root):
    """root with two locked development sets of the sizes FT3's checkpoints are chosen on (docs/m7-进度.md:
    dev 150 and dev-v2 126 items): 58 "five", 56 "ten" and 36 "bye" items in dev, 27, 40 and 59 in dev-v2.
    Over the 276 items together OTHER is right on 96, BYE on 95 and CALL on 85; on dev alone CALL is best
    (58), on dev-v2 alone BYE (59), which also has the highest mean of the two rates."""
    write_set(root / DEV, make_set("dev", {"five": 58, "ten": 56, "bye": 36}))
    write_set(root / DEV_V2, make_set("dev-v2", {"five": 27, "ten": 40, "bye": 59}))
    return root


def test_several_sets_count_by_their_summed_successes_and_each_runs_into_its_own_folder(two_sets, capsys):
    """Step 10 is right on the most of the 276 items (96), though step 20 is the best on dev alone and step 30
    on dev-v2 alone and on the mean of the two rates: the sums over the sets choose."""
    llama = FakeLlama({10: OTHER, 20: CALL, 30: BYE})
    assert run_select(llama, *SETS) == 0
    selection = selection_of(two_sets)
    rows = selection["checkpoints"]
    assert [(row["step"], row["successes"], row["n"]) for row in rows] == [(10, 96, 276), (20, 85, 276),
                                                                          (30, 95, 276)]
    assert [(row["sets"][DEV]["successes"], row["sets"][DEV_V2]["successes"]) for row in rows] == [
        (56, 40), (58, 27), (36, 59)]
    gguf = "runs/train/ft/gguf/checkpoint-10-f16.gguf"
    assert selection["chosen"] == {"step": 10, "gguf": gguf, "sha256": sha256_of(gguf), "n": 276,
                                   "successes": 96, "rate": 96 / 276}
    assert selection["item_sets"] == [{"path": DEV, "sha256": sha256_of(DEV)},
                                      {"path": DEV_V2, "sha256": sha256_of(DEV_V2)}]
    assert set(selection) == SELECTION_KEYS - {"items", "items_sha256"} | {"item_sets"}
    assert selection["rule"] == f"{select.RULE} {select.SETS_RULE}"
    assert set(rows[0]) == ROW_KEYS - {"records"} | {"sets"}
    assert rows[0]["rate"] == 96 / 276 and rows[0]["statuses"] == {"ok": 276}
    assert rows[0]["sets"] == {
        DEV: {"n": 150, "successes": 56, "rate": 56 / 150, "statuses": {"ok": 150},
              "records": "runs/eval/select-ft/dev/checkpoint-10.jsonl"},
        DEV_V2: {"n": 126, "successes": 40, "rate": 40 / 126, "statuses": {"ok": 126},
                 "records": "runs/eval/select-ft/dev-v2/checkpoint-10.jsonl"}}
    # one server per checkpoint serves both sets, in their order, each run into its set's own folder
    assert len(llama.processes) == 3 and all(p.terminated for p in llama.processes)
    assert len(llama.payloads) == 3 * 276
    assert [(Path(cmd[1]).name, cmd[2]) for cmd in llama.runs[:4]] == [
        ("run_eval.py", DEV), ("score.py", "runs/eval/select-ft/dev/checkpoint-10.jsonl"),
        ("run_eval.py", DEV_V2), ("score.py", "runs/eval/select-ft/dev-v2/checkpoint-10.jsonl")]
    evaluate, score_v2 = llama.runs[2], llama.runs[3]
    assert evaluate[evaluate.index("--out") + 1] == "runs/eval/select-ft/dev-v2/checkpoint-10.jsonl"
    assert score_v2[3:] == ["--items", DEV_V2, "--out", "runs/eval/select-ft/dev-v2/checkpoint-10.score.json"]
    assert records_of(two_sets) == sorted(
        [f"checkpoint-{step}.server.log" for step in (10, 20, 30)]
        + [f"{stem}/checkpoint-{step}{suffix}" for stem in ("dev", "dev-v2") for step in (10, 20, 30)
           for suffix in (".jsonl", ".manifest.json", ".timing.json", ".score.json")])
    out = capsys.readouterr().out
    assert "step 10 (dev-v2): 40/126" in out and "step 10: 96/276" in out and "chosen: step 10" in out


def test_select_by_mean_takes_the_mean_of_each_variants_successes_summed_over_the_sets(two_sets):
    """s0 alone would choose step 10 (96 of the 276) and s1 alone step 20 (96); the means of the two are 90.5,
    90.5 and 95: step 30."""
    llama = FakeLlama({10: {"s0": OTHER, "s1": CALL}, 20: {"s0": CALL, "s1": OTHER},
                       30: {"s0": BYE, "s1": BYE}})
    variants = ("--variant", "s0=--mode think --seed 0", "--variant", "s1=--mode think --seed 1")
    assert run_select(llama, *SETS, *variants, "--select-by", "mean:s0,s1") == 0
    selection = selection_of(two_sets)
    rows = selection["checkpoints"]
    assert [(row["step"], row["successes"], row["n"]) for row in rows] == [(10, 90.5, 276), (20, 90.5, 276),
                                                                          (30, 95, 276)]
    assert selection["chosen"]["step"] == 30 and selection["chosen"]["rate"] == 95 / 276
    assert set(rows[0]) == {"step", "gguf", "sha256", "n", "successes", "rate", "server_build", "variants"}
    assert rows[0]["rate"] == 90.5 / 276
    assert rows[0]["variants"]["s0"] == {
        "n": 276, "successes": 96, "rate": 96 / 276, "statuses": {"ok": 276},
        "sets": {DEV: {"n": 150, "successes": 56, "rate": 56 / 150, "statuses": {"ok": 150},
                       "records": "runs/eval/select-ft/dev/checkpoint-10.s0.jsonl"},
                 DEV_V2: {"n": 126, "successes": 40, "rate": 40 / 126, "statuses": {"ok": 126},
                          "records": "runs/eval/select-ft/dev-v2/checkpoint-10.s0.jsonl"}}}
    assert [(row["variants"]["s1"]["successes"], row["variants"]["s1"]["sets"][DEV]["successes"],
             row["variants"]["s1"]["sets"][DEV_V2]["successes"]) for row in rows] == [
        (85, 58, 27), (96, 56, 40), (95, 36, 59)]
    evals = [cmd for cmd in llama.runs if Path(cmd[1]).name == "run_eval.py"]
    assert len(evals) == 12 and len(llama.processes) == 3  # two variants over two sets on each server
    assert [(cmd[2], cmd[cmd.index("--out") + 1], cmd[-1]) for cmd in evals[:4]] == [
        (DEV, "runs/eval/select-ft/dev/checkpoint-10.s0.jsonl", "0"),
        (DEV_V2, "runs/eval/select-ft/dev-v2/checkpoint-10.s0.jsonl", "0"),
        (DEV, "runs/eval/select-ft/dev/checkpoint-10.s1.jsonl", "1"),
        (DEV_V2, "runs/eval/select-ft/dev-v2/checkpoint-10.s1.jsonl", "1")]
    assert "dev/checkpoint-10.jsonl" not in records_of(two_sets)  # --select-by: the default run is not made
    assert selection["selection"] == {"metric": "mean", "variants": ["s0", "s1"]}
    assert selection["rule"] == f"{select.Selection('mean', ('s0', 's1')).rule()} {select.SETS_RULE}"


def test_a_tie_over_the_sets_goes_to_the_later_step(root):
    """Steps 10 and 20 are right on 110 of the 276 items each, step 10 the better on dev (70 against 50) and
    step 20 on dev-v2 (60 against 40): the later step, as with one set."""
    write_set(root / DEV, make_set("dev", {"five": 70, "bye": 50, "ten": 30}))
    write_set(root / DEV_V2, make_set("dev-v2", {"five": 40, "bye": 60, "ten": 26}))
    assert run_select(FakeLlama({10: CALL, 20: BYE, 30: OTHER}), *SETS) == 0
    selection = selection_of(root)
    assert [(row["step"], row["successes"], row["n"]) for row in selection["checkpoints"]] == [
        (10, 110, 276), (20, 110, 276), (30, 56, 276)]
    assert [row["sets"][DEV]["successes"] for row in selection["checkpoints"]] == [70, 50, 30]
    assert selection["chosen"]["step"] == 20 and selection["chosen"]["rate"] == 110 / 276


def test_dry_run_lists_each_set_and_its_commands_and_starts_and_writes_nothing(two_sets, capsys):
    argv = [*ARGV, *SETS, "--variant", "greedy=--greedy", "--dry-run"]
    assert select.main(argv, popen=failing, run=failing) == 0
    out = capsys.readouterr().out
    assert "items: 2 sets taken together, 276 items" in out
    assert f"set dev: {DEV} (sha256 {sha256_of(DEV)}), 150 items" in out
    assert f"set dev-v2: {DEV_V2} (sha256 {sha256_of(DEV_V2)}), 126 items" in out
    for step in (10, 20, 30):
        for items, stem in ((DEV, "dev"), (DEV_V2, "dev-v2")):
            for name in ("", ".greedy"):
                records = f"runs/eval/select-ft/{stem}/checkpoint-{step}{name}"
                assert f"run_eval.py {items} --system ft --out {records}.jsonl " in out
                assert f"score.py {records}.jsonl --items {items} --out {records}.score.json" in out
    assert out.count("  server:") == 3
    assert "run_eval[dev]:" in out and "score[greedy on dev-v2]:" in out
    assert not (two_sets / "results").exists() and not (two_sets / "runs" / "eval").exists()


def test_an_item_id_in_two_sets_is_a_usage_error_and_nothing_starts(two_sets, capsys):
    third = "data/eval/third.jsonl"  # its second item is one of dev-v2's, and no other set holds its ids
    write_set(two_sets / third, [*make_set("third", {"bye": 1}), {**make_set("dev-v2", {"five": 9})[8]}])
    assert select.main([*ARGV, *SETS, "--items", third, "--dry-run"], popen=failing, run=failing) == 2
    out = capsys.readouterr().out
    assert f"the sets {DEV_V2} and {third} both hold 1 item" in out and "'dev-v2-009'" in out
    items = make_set("dev-v2", {"five": 27, "ten": 40, "bye": 59})
    items[5] = {**items[5], "id": "dev-017"}  # an id of dev's
    write_set(two_sets / DEV_V2, items)
    for extra in (["--dry-run"], []):
        assert select.main([*ARGV, *SETS, *extra], popen=failing, run=failing) == 2
        out = capsys.readouterr().out
        assert f"the sets {DEV} and {DEV_V2} both hold 1 item" in out and "'dev-017'" in out
    assert not (two_sets / "results").exists() and not (two_sets / "runs" / "eval").exists()


def test_an_id_of_the_first_set_in_the_third_is_found_past_the_set_between(two_sets, capsys):
    """Each set is held against every set before it, not only the one just before: the third set's second item
    is one of dev's, and dev-v2, between the two, shares no id with either."""
    third = "data/eval/third.jsonl"
    write_set(two_sets / third, [*make_set("third", {"bye": 1}), make_set("dev", {"five": 40})[39]])
    for extra in (["--dry-run"], []):
        assert select.main([*ARGV, *SETS, "--items", third, *extra], popen=failing, run=failing) == 2
        out = capsys.readouterr().out
        assert f"the sets {DEV} and {third} both hold 1 item(s), the first 'dev-040'" in out
    assert not (two_sets / "results").exists() and not (two_sets / "runs" / "eval").exists()


@pytest.mark.parametrize("second", [
    DEV, "./data/eval/dev.jsonl", "data/other/dev.jsonl", "data/other/Dev.jsonl", "data/eval/dev v2.jsonl",
    "data/eval/dev.v2.jsonl"])
def test_sets_whose_folders_could_collide_are_a_usage_error_before_any_set_is_read(two_sets, capsys, second):
    """With several sets each set's runs go to <records>/<file stem>: the same file twice, two files of one
    stem (whatever its case) and a stem that is no file-name piece are refused before any set is read (none of
    the second files exists but dev itself, whose second reading the id check would refuse too)."""
    argv = [*ARGV, "--items", DEV, "--items", second, "--dry-run"]
    assert select.main(argv, popen=failing, run=failing) == 2
    assert "file stem" in capsys.readouterr().out


@pytest.mark.parametrize("material, message", [
    ("data/eval/test.jsonl", "locked test set"), ("data/eval/v2/test-v2.jsonl", "locked test set"),
    ("data/eval/handwritten.jsonl", "handwritten items with their gold turns"),
    ("data/eval/v2/handwritten-v2.jsonl", "handwritten items with their gold turns")])
def test_a_test_set_among_several_is_refused_before_any_set_is_read(two_sets, capsys, material, message):
    """Exit 4 wherever it stands: a test set or a file of handwritten test items is refused by its location
    before any set is read, so neither it (not JSON) nor dev, whose bytes no longer match its lock (exit 5 had
    it been read), is read."""
    (two_sets / material).parent.mkdir(parents=True, exist_ok=True)
    (two_sets / material).write_bytes(b"not json: never parsed\n")
    dev = two_sets / DEV
    dev.write_bytes(dev.read_bytes() + b"\n")
    for order in ([DEV, material], [material, DEV], [DEV_V2, DEV, material]):
        argv = [*ARGV, *(arg for path in order for arg in ("--items", path)), "--dry-run"]
        assert select.main(argv, popen=failing, run=failing) == 4, order
        assert message in capsys.readouterr().out


@pytest.mark.parametrize("position", [0, 1, 2])
def test_every_set_is_checked_as_one_set_is_wherever_it_stands(two_sets, capsys, position):
    """Test items under a lock of their own (exit 4), a set with no items (exit 1) and a set that no longer
    matches its lock (exit 5), first, between or last among three sets."""
    def among(path: str) -> list[str]:
        order = [DEV, DEV_V2]
        order.insert(position, path)
        return [*ARGV, *(arg for one in order for arg in ("--items", one)), "--dry-run"]

    eval_dir = two_sets / "data" / "eval"
    write_set(eval_dir / "copy.jsonl", [{**i, "split": "test"} for i in make_set("copy", {"five": 2})])
    assert select.main(among("data/eval/copy.jsonl"), popen=failing, run=failing) == 4
    assert "holds test material" in capsys.readouterr().out
    write_set(eval_dir / "empty.jsonl", [])
    assert select.main(among("data/eval/empty.jsonl"), popen=failing, run=failing) == 1
    assert "data/eval/empty.jsonl holds no items" in capsys.readouterr().out
    write_set(eval_dir / "stale.jsonl", make_set("stale", {"bye": 2}))
    (eval_dir / "stale.jsonl").write_bytes((eval_dir / "stale.jsonl").read_bytes() + b"\n")
    assert select.main(among("data/eval/stale.jsonl"), popen=failing, run=failing) == 5
    assert "does not match its lock" in capsys.readouterr().out


@pytest.mark.parametrize("order", [("outside",), ("outside", DEV), (DEV, "outside")],
                         ids=["alone", "first", "second"])
def test_a_real_run_refuses_a_set_outside_the_current_folder_wherever_it_stands(
        root, tmp_path_factory, capsys, order):
    """Run from the repository root: a locked set outside the current folder, the one set or either of two,
    would stand in the selection file as a machine path (items, item_sets and the rows' sets), so a real run
    refuses it (exit 2) before any server starts and writes nothing; the dry run lists it."""
    outside = tmp_path_factory.mktemp("elsewhere") / "dev-x.jsonl"
    write_set(outside, make_set("dev-x", {"five": 2, "bye": 1}))
    paths = [str(outside) if path == "outside" else path for path in order]
    argv = [*ARGV, *(arg for path in paths for arg in ("--items", path))]
    assert select.main([*argv, "--dry-run"], popen=failing, run=failing) == 0
    capsys.readouterr()
    assert select.main(argv, popen=failing, run=failing) == 2
    out = capsys.readouterr().out
    assert f"run from the repository root: {[select.rel(outside)]} lie outside the current folder" in out
    assert not (root / "results").exists() and not (root / "runs" / "eval").exists()


# One set, as before --items could be repeated: each case's run folder, flags and answers (the default set is
# ITEMS, another set 5 items of make_set), and what its dry run and its run leave, computed with
# scripts/select_checkpoint.py of commit 64250c0 (before several sets) on the fixture of the test below, whose
# sets are written with LF so the digests are the same on every platform: the SHA-256 of the selection file,
# of the dry run's output, of the files under runs/eval (their paths, one per line) and of every command
# started, in order (JSON); the Python executable and the scripts folder, which differ between machines, are
# written <python> and <scripts>. items_given has the digests of default.
ONE_SET_CASES = {
    "default": ("ft", (), {10: CALL, 20: BYE, 30: CALL}),
    "items_given": ("ft", ("--items", DEV), {10: CALL, 20: BYE, 30: CALL}),
    "another_set": ("ft", ("--items", DEV_V2), {10: OTHER, 20: CALL, 30: BYE}),
    "variant": ("ft", ("--variant", "greedy=--greedy"),
                {10: {"": CALL, "greedy": BYE}, 20: {"": BYE, "greedy": CALL},
                 30: {"": BYE, "greedy": CALL}}),
    "select_by": ("ft", ("--variant", "s0=--mode think --seed 0", "--variant", "s1=--mode think --seed 1",
                         "--select-by", "mean:s0,s1"),
                  {10: {"s0": CALL, "s1": BYE}, 20: {"s0": CALL, "s1": CALL}, 30: {"s0": BYE, "s1": BYE}}),
    "checkpoints": ("ft3", ("--system", "ft", "--checkpoints", "10,20"), {10: BYE, 20: CALL}),
}
ONE_SET_GOLDEN = {
    "default": {"selection": "7d0e0d68f09516a5ae72b3156e579b9920327ca5b236f8e14be457b9b8eb2896",
                "dry_run": "62ba2ac86d23d5a2a7a3c08ee3428dd69fce9b781baf33353eedeafc6ee4da01",
                "files": "9dd012b441a02c2c9e55abcda8abe3c4d6e2c8514106645e8140ec0b0afb5435",
                "commands": "37ea03b4bbd8176d5d87b0518a50f9f2fb30836b8e2ed8af48d4c561a95eec6b"},
    "items_given": {"selection": "7d0e0d68f09516a5ae72b3156e579b9920327ca5b236f8e14be457b9b8eb2896",
                    "dry_run": "62ba2ac86d23d5a2a7a3c08ee3428dd69fce9b781baf33353eedeafc6ee4da01",
                    "files": "9dd012b441a02c2c9e55abcda8abe3c4d6e2c8514106645e8140ec0b0afb5435",
                    "commands": "37ea03b4bbd8176d5d87b0518a50f9f2fb30836b8e2ed8af48d4c561a95eec6b"},
    "another_set": {"selection": "d7cb19ee7cfb6d01ee06e7d31179605035bbe9a603d40f5eec3d54947236ccf6",
                    "dry_run": "a78bda3b2bcae7e1bfa99ac7e6df2515ef51fd830963c7a95e41ee854aaae632",
                    "files": "9dd012b441a02c2c9e55abcda8abe3c4d6e2c8514106645e8140ec0b0afb5435",
                    "commands": "a52f5fae8ca8a012482ee21a650b06fea45231574459bd7b42b674ae3bd309e2"},
    "variant": {"selection": "99f00870fde6cb336e7e62604272a44bf3e79e1c2037de4458882c113dedd348",
                "dry_run": "dbfbcbe1b863f2b7e641f3936bccd3aad735e6d65fca9b161f7a353f262e9410",
                "files": "f33f1f55bd89af50bbc995af3aedd4dcd4cc068780f22f2edac98110e82e4cb3",
                "commands": "ed4ea196273417c607adf58bb79d4d3dde56b2a087c4365d468a05cce908e655"},
    "select_by": {"selection": "b0878790dc7d000d83578478150557c23d19bac6af0bd0a06fe38449f87a319c",
                  "dry_run": "dd56d7d8af3e4f341cf2d334ab49c92b674ec9cd796b9ff643167e6a2ab83e47",
                  "files": "1a5cf6e03762903534ea4c475c99fe0c9b63441689a4aa66c2d8ad50f844608c",
                  "commands": "5f07031b2dba5d2d4d14a233e379556648da5a6ac348b7a7bc0c452ce932f8fe"},
    "checkpoints": {"selection": "832802db65538f6e125b74107ed61cedf0a413b41863d8795c7770ce78539805",
                    "dry_run": "ce87b71947d361cfa923db54f3b62e83f37a491f1cbdeaef2fdf3f01528465c3",
                    "files": "14b35611563accca9dc370c0a4c59e5a180972deb380a584a7d208299bb59546",
                    "commands": "19eada882ba7600e46b939f65a29b55d7c1930fa870b0036fe6465654c141706"},
}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def portable(text: str) -> str:
    """text with the Python executable and the scripts folder, which differ between machines, written
    <python> and <scripts>."""
    return (text.replace(subprocess.list2cmdline([sys.executable]), "<python>")
            .replace(select.rel(select.SCRIPTS), "<scripts>"))


@pytest.mark.parametrize("case", list(ONE_SET_CASES))
def test_one_set_runs_write_and_print_byte_for_byte_what_they_did_before_several_sets(root, capsys, case):
    name, flags, answers = ONE_SET_CASES[case]
    write_set(root / DEV, ITEMS)
    write_set(root / DEV_V2, make_set("dev-v2", {"five": 2, "ten": 2, "bye": 1}))
    if name == "ft3":
        partial_run(root / "runs" / "train" / "ft3", (10, 20, 30), exported=(10, 30))
    argv = [f"runs/train/{name}", "--llama-server", "llama-server.exe", *flags]
    assert select.main([*argv, "--dry-run"], popen=failing, run=failing) == 0
    dry_run = portable(capsys.readouterr().out)
    llama, started = FakeLlama(answers), []

    def popen(cmd, **kwargs):
        started.append(["popen", *cmd])
        return llama.popen(cmd, **kwargs)

    def run(cmd, **kwargs):
        started.append(["run", *cmd])
        return llama.run(cmd, **kwargs)

    assert select.main(argv, popen=popen, run=run, client=llama.client) == 0
    files = sorted(p.relative_to(root).as_posix() for p in (root / "runs" / "eval").rglob("*") if p.is_file())
    commands = [["<python>" if part == sys.executable else portable(part) for part in cmd] for cmd in started]
    machine = (str(REPO), REPO.as_posix(), str(root), root.as_posix())
    texts = [dry_run, *files, *(part for cmd in commands for part in cmd)]
    assert not any(path in text for path in machine for text in texts)  # no machine path is digested
    got = {"selection": sha256_of(root / "results" / f"checkpoint-selection-{name}.json"),
           "dry_run": digest(dry_run), "files": digest("\n".join(files)),
           "commands": digest(json.dumps(commands))}
    assert got == ONE_SET_GOLDEN[case]


# ---- M7 WP1: --scorer (research_agent.eval.metrics, Scorers; docs/m7-本地提升计划.md) ---------------------

UP_TO_TODAY = {"query": "hallucination", "max_results": 5, "date_to": "2026-10-10"}  # SYSTEM's today
ON_TODAY = {"name": "search_arxiv", "arguments": UP_TO_TODAY, "arguments_json": json.dumps(UP_TO_TODAY)}
TODAY_CALL = f'<tool_call>\n{json.dumps({"name": "search_arxiv", "arguments": UP_TO_TODAY})}\n</tool_call>'
# ITEMS and a search up to today: CALL is right on 2 of them under the default scorer and on 3 under the
# normalized one (S1), TODAY_CALL on 1 and on 3, BYE on 1 under either
SCORED = [*ITEMS, item("q-4", "Find hallucination papers up to today.",
                       {"content": "", "tool_calls": [ON_TODAY]}, "call", "en")]


def test_the_normalized_scorer_scores_selects_and_is_recorded_beside_the_default_selection(root, capsys):
    """--scorer normalized: score.py scores each run with it into <stem>.score-normalized.json, the selection
    counts those successes, and the selection file (results/checkpoint-selection-<run>-normalized.json by
    default) records the scorer in its own key and its rule; a default selection of the same run afterwards
    needs a records folder of its own (in the same one it would make again the records the normalized score
    files describe, and is refused), where it makes its own file and score files, the normalized ones left as
    they are."""
    write_set(root / DEV, SCORED)
    answers = {10: CALL, 20: TODAY_CALL, 30: BYE}
    assert select.main([*ARGV, "--scorer", "normalized", "--dry-run"], popen=failing, run=failing,
                       client=failing) == 0
    out = capsys.readouterr().out
    assert "scorer: normalized" in out and "reuse" not in out
    assert "--out runs/eval/select-ft/checkpoint-10.score-normalized.json --scorer normalized" in out
    llama = FakeLlama(answers)
    assert run_select(llama, "--scorer", "normalized") == 0
    selection = selection_of(root, "ft-normalized")
    assert selection["scorer"] == "normalized" and "records_reused" not in selection
    assert selection["rule"] == f"{select.RULE} {select.SCORER_RULE.format(scorer='normalized')}"
    assert [(c["step"], c["successes"]) for c in selection["checkpoints"]] == [(10, 3), (20, 3), (30, 1)]
    assert selection["chosen"]["step"] == 20  # a tie under the normalized scorer: the later step
    first = next(cmd for cmd in llama.runs if Path(cmd[1]).name == "score.py")
    assert first[-4:] == ["--out", "runs/eval/select-ft/checkpoint-10.score-normalized.json", "--scorer",
                          "normalized"]
    records = root / "runs" / "eval" / "select-ft"
    scored = json.loads((records / "checkpoint-20.score-normalized.json").read_text(encoding="utf-8"))
    assert scored["scorer"] == "normalized" and scored["summary"]["n"] == 4
    before = snapshot(records)
    assert run_select(FakeLlama(answers)) == 2
    assert snapshot(records) == before and not (root / "results" / "checkpoint-selection-ft.json").exists()
    assert run_select(FakeLlama(answers), "--records", "runs/eval/select-ft-default") == 0
    default = selection_of(root)
    assert [(c["step"], c["successes"]) for c in default["checkpoints"]] == [(10, 2), (20, 1), (30, 1)]
    assert default["chosen"]["step"] == 10 and "scorer" not in default
    own = root / "runs" / "eval" / "select-ft-default"
    assert "scorer" not in json.loads((own / "checkpoint-20.score.json").read_text(encoding="utf-8"))
    assert selection_of(root, "ft-normalized") == selection and snapshot(records) == before


# ---- M7 WP1 review: a selection under another scorer never overwrites the records it finds ----------------

VARIANTS = {"s0": "--mode think --seed 0", "s1": "--mode think --seed 1"}
STEPS = (10, 20, 30)
BUILD = "b11349-fb4b2737a"  # FakeLlama's /props build_info


def by_variant(*names: str) -> list[str]:
    """The flags of a selection by the mean of the named variants of VARIANTS."""
    flags = [flag for name in names for flag in ("--variant", f"{name}={VARIANTS[name]}")]
    return [*flags, "--select-by", f"mean:{','.join(names)}"]


def snapshot(folder: Path) -> dict[str, bytes]:
    """Every file under folder, by its path relative to folder, with its bytes."""
    return {path.relative_to(folder).as_posix(): path.read_bytes()
            for path in sorted(folder.rglob("*")) if path.is_file()}


def made_on_the_family_path(root: Path, family, *names: str) -> None:
    """The manifests of the named variants' runs under runs/eval/select-ft as run_eval writes them on the
    family path: with the family's record and the generation the variant's arguments ask for (think mode, the
    variant's seed, the family's decoding and budget). FakeLlama's run_eval renders with a fake tokenizer
    instead, and so records neither."""
    for name in names:
        seed = int(VARIANTS[name].rsplit(" ", 1)[1])
        generation = GenerationSettings.for_family(family, mode=Mode.THINK, base_seed=seed).record()
        for path in (root / "runs" / "eval" / "select-ft").rglob(f"checkpoint-*.{name}.manifest.json"):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifest.update(family=family.record(), generation=generation)
            path.write_text(json.dumps(manifest), encoding="utf-8")


def edit_manifest(path: Path, change) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    change(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def selected(root):
    """root after a default selection by the mean of s0 and s1 on SCORED, made as on the family path
    (made_on_the_family_path): (root, the family, the answers). Under the default scorer CALL is right on 2 of
    the 4 items, TODAY_CALL on 1 and BYE on 1; under the normalized one on 3, 3 and 1."""
    write_set(root / DEV, SCORED)
    answers = {10: {"s0": CALL, "s1": TODAY_CALL}, 20: {"s0": TODAY_CALL, "s1": TODAY_CALL},
               30: {"s0": BYE, "s1": CALL}}
    assert run_select(FakeLlama(answers), *by_variant("s0", "s1")) == 0
    family = fake_loaded_family(root / "fake-model")
    made_on_the_family_path(root, family, "s0", "s1")
    return root, family, answers


def test_a_normalized_selection_scores_the_records_it_finds_again_and_starts_no_server(selected, capsys):
    """Every run's records are there and their manifests record what the selection asks for: no server
    starts and run_eval does not run; score.py scores the records again into <stem>.score-normalized.json, and
    every file that was there stays byte for byte. The rows' server build is the one the manifests record, and
    the selection lists the records it reused."""
    root, family, answers = selected
    records = root / "runs" / "eval" / "select-ft"
    before = snapshot(records)
    argv = [*ARGV, *by_variant("s0", "s1"), "--scorer", "normalized"]

    def hook(folder):
        return family

    capsys.readouterr()
    assert select.main([*argv, "--dry-run"], popen=failing, run=failing, client=failing,
                       load_family=hook) == 0
    out = capsys.readouterr().out
    assert "  server:   none: every run's records are there" in out and "run_eval[" not in out
    assert "  reuse[s1]:    runs/eval/select-ft/checkpoint-30.s1.jsonl" in out
    assert "score.py runs/eval/select-ft/checkpoint-30.s1.jsonl --items data/eval/dev.jsonl --out " \
           "runs/eval/select-ft/checkpoint-30.s1.score-normalized.json --scorer normalized" in out
    llama = FakeLlama(answers)
    assert select.main(argv, popen=failing, run=llama.run, client=llama.client, load_family=hook) == 0
    assert [Path(cmd[1]).name for cmd in llama.runs] == ["score.py"] * 6 and llama.payloads == []
    after = snapshot(records)
    assert {path: after[path] for path in before} == before
    assert sorted(set(after) - set(before)) == sorted(f"checkpoint-{step}.{name}.score-normalized.json"
                                                      for step in STEPS for name in ("s0", "s1"))
    selection = selection_of(root, "ft-normalized")
    rows = selection["checkpoints"]
    assert [(row["step"], row["successes"]) for row in rows] == [(10, 3), (20, 3), (30, 2)]
    assert selection["chosen"]["step"] == 20 and {row["server_build"] for row in rows} == {BUILD}
    assert selection["records_reused"] == [f"runs/eval/select-ft/checkpoint-{step}.{name}.jsonl"
                                           for step in STEPS for name in ("s0", "s1")]
    assert selection_of(root)["chosen"]["step"] == 30  # the default selection's file as it was: 1.5, 1, 1.5
    assert "every run's records are there" in capsys.readouterr().out


MISMATCHES = {  # a change to the manifest of checkpoint-20.s1, and what the refusal names
    "seed": (lambda m: m["generation"].update(base_seed=7), "generation"),
    "mode": (lambda m: m["generation"].update(mode="no_think"), "generation"),
    "budget": (lambda m: m["generation"].update(reasoning_budget=1280), "generation"),
    "no generation": (lambda m: m.pop("generation"), "generation"),  # a run before M7 records none
    "adapter": (lambda m: m["adapters"][0].update(sha256="0" * 64), "adapter"),
    "adapter path": (lambda m: m["adapters"][0].update(path="runs/train/ft/gguf/checkpoint-10-f16.gguf"),
                     "adapter"),
    "system": (lambda m: m.update(system="b4"), "system"),
    "items": (lambda m: m.update(items_sha256="0" * 64), "items_sha256"),
    "items run": (lambda m: m.update(items_run=3), "items_run"),
    "family": (lambda m: m["family"].update(name="qwen3-hybrid"), "family"),
    "slots": (lambda m: m["server_props"].update(total_slots=4), "total_slots"),
    "statuses": (lambda m: m.update(statuses={"error": 1, "ok": 3}), "statuses"),
    "build": (lambda m: m["server_props"].update(build_info="b9999-other"), "build"),
    # round 3: the base model, the prompt and the schema run_eval writes, and the family record as a whole
    "model path": (lambda m: m["server_props"].update(model_path="models/Other-Q4_K_M.gguf"), "model_path"),
    "no model path": (lambda m: m["server_props"].pop("model_path"), "model_path"),
    "base model sha256": (lambda m: m.update(base_model_sha256="0" * 64), "base_model_sha256"),
    "prompt hash": (lambda m: m.update(prompt_hash="0123456789abcdef"), "prompt_hash"),
    "prompt version": (lambda m: m.update(prompt_version="m1-2026-09-01"), "prompt_version"),
    "schema hash": (lambda m: m.update(schema_hash="0123456789abcdef"), "schema_hash"),
    "family template": (lambda m: m["family"].update(template_sha256="0" * 64), "template_sha256"),
    "family tokenizer": (lambda m: m["family"].update(tokenizer_json_sha256="0" * 64), "tokenizer_json"),
    "family key": (lambda m: m["family"].update(attn_implementation="eager"), "attn_implementation"),
    "no family": (lambda m: m.pop("family"), "family None"),  # a run before the family layer records none
}


@pytest.mark.parametrize("case", list(MISMATCHES))
def test_records_whose_manifest_differs_from_the_request_stop_the_selection_and_stay(selected, capsys, case):
    """Exit 2, the dry run too, naming the manifest and what differs; nothing starts, nothing is written."""
    root, family, _ = selected
    change, named = MISMATCHES[case]
    edit_manifest(root / "runs" / "eval" / "select-ft" / "checkpoint-20.s1.manifest.json", change)
    records = root / "runs" / "eval" / "select-ft"
    before = snapshot(records)
    argv = [*ARGV, *by_variant("s0", "s1"), "--scorer", "normalized"]
    capsys.readouterr()
    for extra in (["--dry-run"], []):
        assert select.main([*argv, *extra], popen=failing, run=failing, client=failing,
                           load_family=lambda f: family) == 2
        out = capsys.readouterr().out
        assert "runs/eval/select-ft/checkpoint-20.s1.manifest.json" in out and named in out, out
    assert snapshot(records) == before
    assert not (root / "results" / "checkpoint-selection-ft-normalized.json").exists()


@pytest.mark.parametrize("case", ["no manifest", "not JSON", "records not JSONL"])
def test_records_without_a_manifest_that_can_be_read_stop_the_selection_and_stay(selected, capsys, case):
    root, family, _ = selected
    stem = root / "runs" / "eval" / "select-ft" / "checkpoint-30.s0"
    if case == "no manifest":
        Path(f"{stem}.manifest.json").unlink()
    elif case == "not JSON":
        Path(f"{stem}.manifest.json").write_text("{not JSON", encoding="utf-8")
    else:
        Path(f"{stem}.jsonl").write_text("not JSONL\n", encoding="utf-8")
    before = snapshot(root / "runs" / "eval" / "select-ft")
    argv = [*ARGV, *by_variant("s0", "s1"), "--scorer", "normalized"]
    capsys.readouterr()
    assert select.main(argv, popen=failing, run=failing, client=failing, load_family=lambda f: family) == 2
    assert "runs/eval/select-ft/checkpoint-30.s0" in capsys.readouterr().out
    assert snapshot(root / "runs" / "eval" / "select-ft") == before


def test_records_of_a_run_whose_arguments_the_check_does_not_cover_are_not_reused(root, capsys):
    """A run's arguments may set run_eval's --mode, --seed, --reasoning-budget, --greedy, --signals and
    --limit, which the check covers; anything else (here --no-tokenizer; the inference wrappers' --votes,
    --rules and --glossary alike) could make other records than its manifest's generation tells, so they are
    not reused."""
    write_set(root / DEV, SCORED)
    flags = ["--variant", "s0=--mode think --seed 0 --no-tokenizer", "--select-by", "mean:s0"]
    assert run_select(FakeLlama({step: {"s0": CALL} for step in STEPS}), *flags) == 0
    family = fake_loaded_family(root / "fake-model")
    made_on_the_family_path(root, family, "s0")
    before = snapshot(root / "runs" / "eval" / "select-ft")
    capsys.readouterr()
    assert select.main([*ARGV, *flags, "--scorer", "normalized"], popen=failing, run=failing, client=failing,
                       load_family=lambda f: family) == 2
    out = capsys.readouterr().out
    assert "runs/eval/select-ft/checkpoint-10.s0.manifest.json" in out and "no_tokenizer" in out
    assert snapshot(root / "runs" / "eval" / "select-ft") == before


def test_a_normalized_selection_does_not_replace_a_selection_file_another_scorer_made(selected, capsys):
    """--out naming the default selection's file: refused before anything starts (exit 2, the dry run too),
    the file as it was; a file of the same scorer is written over as before."""
    root, family, answers = selected
    default = root / "results" / "checkpoint-selection-ft.json"
    before = default.read_bytes()
    argv = [*ARGV, *by_variant("s0", "s1"), "--scorer", "normalized", "--out",
            "results/checkpoint-selection-ft.json"]
    capsys.readouterr()
    for extra in (["--dry-run"], []):
        assert select.main([*argv, *extra], popen=failing, run=failing, client=failing,
                           load_family=lambda f: family) == 2
        assert "the default scorer made" in capsys.readouterr().out
    assert default.read_bytes() == before
    again = [*ARGV, *by_variant("s0", "s1"), "--scorer", "normalized"]
    for _ in range(2):  # the second time over the normalized selection's own file
        llama = FakeLlama(answers)
        assert select.main(again, popen=failing, run=llama.run, client=llama.client,
                           load_family=lambda f: family) == 0
    assert selection_of(root, "ft-normalized")["scorer"] == "normalized" and default.read_bytes() == before


def test_the_family_must_load_to_check_the_records_it_finds(selected, capsys):
    capsys.readouterr()
    argv = [*ARGV, *by_variant("s0", "s1"), "--scorer", "normalized"]
    # models/Qwen3-1.7B is no model folder here
    assert select.main(argv, popen=failing, run=failing, client=failing) == 2
    assert "models/Qwen3-1.7B" in capsys.readouterr().out


def test_a_normalized_selection_runs_only_the_missing_records_and_keeps_every_file_it_finds(root):
    """s0's records are there, s1's are not: each checkpoint's server runs s1 alone (were s0 run again it
    would answer otherwise), and every file that was there stays, the first servers' logs too (the new server
    writes the first free checkpoint-<N>.server-<k>.log)."""
    write_set(root / DEV, SCORED)
    assert run_select(FakeLlama({step: {"s0": CALL} for step in STEPS}), *by_variant("s0")) == 0
    family = fake_loaded_family(root / "fake-model")
    made_on_the_family_path(root, family, "s0")
    records = root / "runs" / "eval" / "select-ft"
    before = snapshot(records)
    llama = FakeLlama({step: {"s0": BYE, "s1": TODAY_CALL} for step in STEPS})
    argv = [*ARGV, *by_variant("s0", "s1"), "--scorer", "normalized"]
    assert select.main(argv, popen=llama.popen, run=llama.run, client=llama.client,
                       load_family=lambda f: family) == 0
    evals = [cmd for cmd in llama.runs if Path(cmd[1]).name == "run_eval.py"]
    assert [cmd[cmd.index("--out") + 1] for cmd in evals] == [
        f"runs/eval/select-ft/checkpoint-{step}.s1.jsonl" for step in STEPS]
    assert len(llama.processes) == 3 and all(p.terminated for p in llama.processes)
    after = snapshot(records)
    assert {path: after[path] for path in before} == before
    assert sorted(set(after) - set(before)) == sorted(
        [f"checkpoint-{step}.server-2.log" for step in STEPS]
        + [f"checkpoint-{step}.s1{suffix}" for step in STEPS
           for suffix in (".jsonl", ".manifest.json", ".timing.json", ".score-normalized.json")]
        + [f"checkpoint-{step}.s0.score-normalized.json" for step in STEPS])
    selection = selection_of(root, "ft-normalized")
    rows = selection["checkpoints"]
    assert [(row["step"], row["successes"]) for row in rows] == [(10, 3), (20, 3), (30, 3)]
    assert selection["chosen"]["step"] == 30 and {row["server_build"] for row in rows} == {BUILD}
    assert selection["records_reused"] == [f"runs/eval/select-ft/checkpoint-{step}.s0.jsonl"
                                           for step in STEPS]


def test_a_server_of_another_build_than_the_records_it_finds_stops_the_selection(root, capsys):
    """The records of a checkpoint's runs come from one llama-server build: a server of another build is
    stopped before it runs anything (exit 2), and the records found stay."""
    write_set(root / DEV, SCORED)
    assert run_select(FakeLlama({step: {"s0": CALL} for step in STEPS}), *by_variant("s0")) == 0
    family = fake_loaded_family(root / "fake-model")
    made_on_the_family_path(root, family, "s0")
    records = root / "runs" / "eval" / "select-ft"
    for path in records.glob("checkpoint-*.s0.manifest.json"):
        edit_manifest(path, lambda m: m["server_props"].update(build_info="b9999-other"))
    before = snapshot(records)
    llama = FakeLlama({step: {"s0": BYE, "s1": TODAY_CALL} for step in STEPS})
    argv = [*ARGV, *by_variant("s0", "s1"), "--scorer", "normalized"]
    capsys.readouterr()
    assert select.main(argv, popen=llama.popen, run=llama.run, client=llama.client,
                       load_family=lambda f: family) == 2
    out = capsys.readouterr().out
    assert "b9999-other" in out and BUILD in out
    assert llama.runs == [] and llama.payloads == [] and len(llama.processes) == 1
    assert llama.processes[0].terminated
    after = snapshot(records)
    assert {path: after[path] for path in before} == before and set(after) - set(before) == {
        "checkpoint-10.server-2.log"}


def test_the_scorer_passes_to_every_variant_and_set_and_an_unknown_one_is_a_usage_error(two_sets, capsys):
    argv = [*ARGV, *SETS, "--variant", "s0=--mode think --seed 0", "--select-by", "mean:s0", "--scorer",
            "normalized"]
    assert select.main([*argv, "--dry-run"], popen=failing, run=failing) == 0
    out = capsys.readouterr().out
    for stem in ("dev", "dev-v2"):
        records = f"runs/eval/select-ft/{stem}/checkpoint-10.s0"
        assert f"--out {records}.score-normalized.json --scorer normalized" in out
    assert run_select(FakeLlama({step: {"s0": OTHER} for step in (10, 20, 30)}), *argv[len(ARGV):]) == 0
    selection = selection_of(two_sets, "ft-normalized")
    assert selection["rule"] == (f"{select.Selection('mean', ('s0',)).rule()} {select.SETS_RULE} "
                                 f"{select.SCORER_RULE.format(scorer='normalized')}")
    assert selection["checkpoints"][0]["variants"]["s0"]["successes"] == 96
    with pytest.raises(SystemExit) as exc:
        select.main([*ARGV, "--scorer", "lenient", "--dry-run"], popen=failing, run=failing)
    assert exc.value.code == 2


@pytest.mark.parametrize("case", ["default", "variant", "select_by", "checkpoints"])
def test_an_explicit_default_scorer_writes_and_prints_byte_for_byte_what_no_scorer_did(root, capsys,
                                                                                       monkeypatch, case):
    name, flags, answers = ONE_SET_CASES[case]
    monkeypatch.setitem(ONE_SET_CASES, "explicit", (name, (*flags, "--scorer", "default"), answers))
    monkeypatch.setitem(ONE_SET_GOLDEN, "explicit", ONE_SET_GOLDEN[case])
    golden = test_one_set_runs_write_and_print_byte_for_byte_what_they_did_before_several_sets
    golden(root, capsys, "explicit")


# ---- M7 WP1 review, round 3: the base model, run_eval's prompt and schema and the whole family record of the
# records found; a default selection over records another scorer scored; no real server ----------------------

NORMALIZED_BY_MEAN = [*ARGV, *by_variant("s0", "s1"), "--scorer", "normalized"]


def test_the_base_model_is_the_same_file_however_the_manifest_or_model_spells_it(selected, capsys):
    """server_props.model_path is the model as the server was given it: compared as the adapters' paths are
    (llm.server.same_path, from the current folder), another spelling of --model's file is that model."""
    root, family, _ = selected
    model = root / "models" / "Qwen3-1.7B-Q8_0.gguf"
    records = root / "runs" / "eval" / "select-ft"
    argv = [*NORMALIZED_BY_MEAN, "--dry-run"]
    for spelling in ("./models/Qwen3-1.7B-Q8_0.gguf", "models/../models/Qwen3-1.7B-Q8_0.gguf",
                     model.as_posix(), str(model)):
        for path in records.glob("checkpoint-*.manifest.json"):
            edit_manifest(path, lambda m, spelling=spelling: m["server_props"].update(model_path=spelling))
        assert select.main(argv, popen=failing, run=failing, client=failing,
                           load_family=lambda f: family) == 0, spelling
    assert select.main([*argv, "--model", "./models/Qwen3-1.7B-Q8_0.gguf"], popen=failing, run=failing,
                       client=failing, load_family=lambda f: family) == 0
    assert "reuse[s1]:    runs/eval/select-ft/checkpoint-30.s1.jsonl" in capsys.readouterr().out


def test_a_selection_over_another_base_model_does_not_reuse_the_records_it_finds(selected, capsys):
    """The records were made over models/Qwen3-1.7B-Q8_0.gguf (their server_props.model_path): a selection
    with another --model, whose selection file would name that model as their base, is refused (exit 2, the
    dry run too), every manifest named, and nothing is written."""
    root, family, _ = selected
    (root / "models" / "Other-Base-Q8_0.gguf").write_bytes(b"GGUF other base")
    records = root / "runs" / "eval" / "select-ft"
    before = snapshot(records)
    capsys.readouterr()
    for extra in (["--dry-run"], []):
        argv = [*NORMALIZED_BY_MEAN, "--model", "models/Other-Base-Q8_0.gguf", *extra]
        assert select.main(argv, popen=failing, run=failing, client=failing,
                           load_family=lambda f: family) == 2
        out = capsys.readouterr().out
        assert out.count("server_props.model_path 'models/Qwen3-1.7B-Q8_0.gguf'") == 6
        assert "models/Other-Base-Q8_0.gguf" in out
    assert snapshot(records) == before
    assert not (root / "results" / "checkpoint-selection-ft-normalized.json").exists()


def test_a_base_model_sha256_a_manifest_records_must_be_the_models_and_is_hashed_only_then(selected, capsys,
                                                                                           monkeypatch):
    """run_eval records no SHA-256 of the base model: while no manifest does, --model is not hashed (a base
    model is gigabytes); a manifest that records one (base_model_sha256) has it compared with --model's,
    hashed once per selection."""
    root, family, _ = selected
    hashed, sha256 = [], select.sha256_of
    monkeypatch.setattr(select, "sha256_of",
                        lambda path: hashed.append(Path(path).as_posix()) or sha256(path))
    argv = [*NORMALIZED_BY_MEAN, "--dry-run"]
    assert select.main(argv, popen=failing, run=failing, client=failing, load_family=lambda f: family) == 0
    assert hashed and "models/Qwen3-1.7B-Q8_0.gguf" not in hashed  # the checkpoints are hashed, the model not
    digest = sha256_of(root / "models" / "Qwen3-1.7B-Q8_0.gguf")
    for path in (root / "runs" / "eval" / "select-ft").glob("checkpoint-*.manifest.json"):
        edit_manifest(path, lambda m: m.update(base_model_sha256=digest))
    hashed.clear()
    assert select.main(argv, popen=failing, run=failing, client=failing, load_family=lambda f: family) == 0
    assert hashed.count("models/Qwen3-1.7B-Q8_0.gguf") == 1


@pytest.mark.parametrize("name, field", [("PROMPT_VERSION", "prompt_version"), ("PROMPT_HASH", "prompt_hash"),
                                         ("SCHEMA_HASH", "schema_hash")])
def test_the_prompt_and_schema_the_records_need_are_the_ones_run_eval_writes_now(selected, capsys,
                                                                                 monkeypatch, name, field):
    """The values compared are taken from scripts/run_eval.py's own namespace, which writes them: once
    run_eval would write another (a changed prompt or schema), the records made before are not reused."""
    _, family, _ = selected
    monkeypatch.setattr(select._run_eval(), name, "0123456789abcdef")
    capsys.readouterr()
    assert select.main([*NORMALIZED_BY_MEAN, "--dry-run"], popen=failing, run=failing, client=failing,
                       load_family=lambda f: family) == 2
    out = capsys.readouterr().out
    assert out.count(f"{field} ") == 6 and "where the run asks for '0123456789abcdef'" in out


def test_a_default_selection_does_not_write_over_records_another_scorer_scored(selected, capsys):
    """After a normalized selection in runs/eval/select-ft, a default selection there would make again the
    records the normalized score files describe: refused before anything starts (exit 2, the dry run too),
    nothing written, the score files named; with a --records folder of its own it runs as before."""
    root, family, answers = selected
    llama = FakeLlama(answers)
    assert select.main(NORMALIZED_BY_MEAN, popen=failing, run=llama.run, client=llama.client,
                       load_family=lambda f: family) == 0
    records = root / "runs" / "eval" / "select-ft"
    before, chosen = snapshot(records), (root / "results" / "checkpoint-selection-ft.json").read_bytes()
    capsys.readouterr()
    default = [*ARGV, *by_variant("s0", "s1")]
    for extra in (["--dry-run"], []):
        assert select.main([*default, *extra], popen=failing, run=failing, client=failing) == 2
        out = capsys.readouterr().out
        assert out.count(".score-normalized.json") == 6 and "--records" in out
        assert "runs/eval/select-ft/checkpoint-20.s1.score-normalized.json" in out
    assert snapshot(records) == before
    assert (root / "results" / "checkpoint-selection-ft.json").read_bytes() == chosen
    llama = FakeLlama(answers)
    assert select.main([*default, "--records", "runs/eval/select-ft-default"], popen=llama.popen,
                       run=llama.run, client=llama.client) == 0
    assert snapshot(records) == before
    rows = selection_of(root)["checkpoints"]
    assert rows[0]["variants"]["s0"]["records"] == "runs/eval/select-ft-default/checkpoint-10.s0.jsonl"


def test_no_test_here_reaches_a_real_server_through_the_defaults_of_main(root):
    """no_real_server, this file's autouse fixture: select.main without a client fails at its first reach for
    one, before a server process starts, and never constructs the real LocalClient; its defaults for popen
    and run fail too; a test that injects its own client, popen and run still uses them."""
    made = []

    def spy(self, *args, **kwargs):
        made.append(args)
        raise AssertionError("the real LocalClient was constructed")

    llama = FakeLlama({10: CALL, 20: CALL, 30: CALL})
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LocalClient, "__init__", spy)
        with pytest.raises(AssertionError, match="a test reached for the real LocalClient"):
            select.main(ARGV, popen=llama.popen, run=llama.run)
    assert made == [] and llama.processes == [] and llama.runs == []
    assert not (root / "results").exists() and not (root / "runs" / "eval").exists()
    defaults = select.main.__kwdefaults__
    for name, real in (("popen", subprocess.Popen), ("run", subprocess.run)):
        assert defaults[name] is not real  # checked first: the real one is never called here
        with pytest.raises(AssertionError, match="a test reached for the real subprocess"):
            defaults[name](["llama-server.exe"])
    assert run_select(llama) == 0 and len(llama.payloads) == 9  # the fakes injected are used

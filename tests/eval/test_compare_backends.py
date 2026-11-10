"""scripts/compare_backends.py offline: two development runs of one system at one lora scale, one on the hf
backend and one on llama.cpp, compared item by item (the parsed turn), by success rate with the paired
bootstrap and by the exact McNemar test; the refusals; and the runs run_eval writes, compared as written."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from support.fakes import ScriptedModel, call, turn
from support.hf_fakes import FakeHFTokenizer, FakePeftModel, peft_folder
from train.tokenizer_fake import FakeTokenizer

from research_agent.agent.prompts import PROMPT_HASH, system_prompt
from research_agent.eval.lock import sha256_of, write_lock
from research_agent.eval.metrics import bootstrap_ci, mcnemar
from research_agent.eval.runner import run_items, write_results
from research_agent.llm.hf import HFClient
from research_agent.llm.local import Completion
from research_agent.tools.schemas import SCHEMA_HASH
from research_agent.train.samples import prompt_renderer

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


compare = load_script("compare_backends")
run_eval = load_script("run_eval")

SYSTEM = {"role": "system", "content": system_prompt("2026-10-10")}
SEARCH = {"name": "search_arxiv", "arguments": {"query": "hallucination", "max_results": 5},
          "arguments_json": "{\"query\": \"hallucination\", \"max_results\": 5}"}


def item(rid: str, user: str, expected: dict, type_: str, lang: str = "en") -> dict:
    return {"id": rid, "messages": [SYSTEM, {"role": "user", "content": user}], "expected": expected,
            "type": type_, "lang": lang, "family": f"f/{type_}", "split": "dev", "slots": {},
            "source": {"request": rid, "step": 1}}


ITEMS = [item("q-1", "找几篇幻觉(hallucination)的论文。", {"content": "", "tool_calls": [SEARCH]}, "call",
              "zh"),
         item("q-2", "Thanks, that's all for today.", {"content": "See you!", "tool_calls": []}, "no_call"),
         item("q-3", "Find hallucination papers.", {"content": "", "tool_calls": [SEARCH]}, "call"),
         item("q-4", "A long history.", {"content": "Done.", "tool_calls": []}, "no_call")]


def searching() -> object:
    return turn(call("search_arxiv", "local_0", query="hallucination", max_results=5), paid=False)


def answering(text: str) -> object:
    return turn(content=text, paid=False)


def overflowing(messages: list[dict], tools: list[dict]) -> int:
    """A context counter under which q-4 alone does not fit: both backends record it context_overflow."""
    return 10 ** 6 if messages[-1]["content"] == "A long history." else 1


ADAPTER = {"path": "runs/train/ft/trainer/checkpoint-209", "safetensors_sha256": "a" * 64}
GGUF = [{"id": 0, "path": "runs/train/ft/gguf/checkpoint-209-f16.gguf", "sha256": "b" * 64}]
MANIFESTS = {
    "hf": {"backend": "hf", "provider": "transformers", "dtype": "bfloat16", "device": "cuda:0",
           "tokenizer": "models/Qwen3-1.7B", "adapter": ADAPTER,
           "server_props": {"backend": "hf", "model_path": "models/Qwen3-1.7B", "dtype": "bfloat16",
                            "device": "cuda:0", "gpu": "Fake GPU 8 GB", "attn_implementation": "sdpa",
                            "versions": {"torch": "2.14.1", "transformers": "5.18.0", "peft": "0.21.2"}}},
    "llama.cpp": {"backend": "llama.cpp", "provider": "llama-server", "server": "http://127.0.0.1:8080",
                  "tokenizer": "models/Qwen3-1.7B", "adapters": GGUF,
                  "server_props": {"model_path": "models/Qwen3-1.7B-Q8_0.gguf", "build_info": "b11349",
                                   "n_ctx": 8192, "chat_template": "{% for message in messages %}..."}}}


def dev_root(tmp_path: Path, monkeypatch) -> Path:
    """tmp_path as the repository root and the current folder, with a locked development set of ITEMS."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(compare, "REPO", tmp_path)
    monkeypatch.setattr(run_eval, "REPO", tmp_path)
    dev = tmp_path / "data" / "eval" / "dev.jsonl"
    dev.parent.mkdir(parents=True)
    dev.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in ITEMS), encoding="utf-8")
    write_lock(dev)
    return tmp_path


def write_run(root: Path, name: str, kind: str, turns: list, **manifest: object) -> str:
    """A run's records (run_items over ITEMS with the scripted turns) and its manifest, as run_eval writes
    them for the backend kind; manifest overrides or adds keys, None removes one. Returns the records' path
    relative to root."""
    records = run_items(ITEMS, ScriptedModel(turns, name="qwen3-1.7b:lora=1"), count_tokens=overflowing)
    out = root / "runs" / "eval" / f"{name}.jsonl"
    write_results(records, out)
    dev = root / "data" / "eval" / "dev.jsonl"
    full = {"system": "ft", "model": "qwen3-1.7b:lora=1", "lora_scale": 1.0, "temperature": 0.0,
            **MANIFESTS[kind], "prompt_version": "v1", "prompt_hash": PROMPT_HASH,
            "schema_hash": SCHEMA_HASH, "items": "data/eval/dev.jsonl", "items_sha256": sha256_of(dev),
            "items_locked": True, "items_total": len(ITEMS), "items_run": len(ITEMS), "limit": None,
            "date": "2026-10-17", "git_commit": "c" * 40, "git_dirty": False, **manifest}
    full = {k: v for k, v in full.items() if v is not None or k == "limit"}
    out.with_name(f"{name}.manifest.json").write_text(json.dumps(full) + "\n", encoding="utf-8")
    return f"runs/eval/{name}.jsonl"


HF_TURNS = [searching(), answering("See you!"), answering("I cannot search that.")]
CPP_TURNS = [searching(), answering("Bye!"), searching()]
REPORT = Path("results/backend-comparison-ft-scale1.json")


def compared(hf: str, cpp: str, *extra: str) -> int:
    return compare.main(["--hf", hf, "--llama-cpp", cpp, *extra])


def test_the_runs_are_compared_item_by_item_by_success_rate_with_the_paired_bootstrap_and_mcnemar(
        tmp_path, monkeypatch, capsys):
    root = dev_root(tmp_path, monkeypatch)
    hf = write_run(root, "hf-ft", "hf", HF_TURNS)
    cpp = write_run(root, "cpp-ft", "llama.cpp", CPP_TURNS)
    assert compared(hf, cpp) == 0
    report = json.loads((root / REPORT).read_text(encoding="utf-8"))
    assert (report["system"], report["lora_scale"], report["n"]) == ("ft", 1.0, 4)
    assert report["items"] == "data/eval/dev.jsonl" and report["items_locked"] is True
    assert report["items_sha256"] == sha256_of(root / "data" / "eval" / "dev.jsonl")
    # q-1 the same call, q-2 two different answers that both succeed, q-3 a call against an answer, q-4 both
    # context_overflow: the parsed turns agree on q-1 and q-4
    assert {k: report["agreement"][k] for k in ("agree", "n", "rate", "differ")} == {
        "agree": 2, "n": 4, "rate": 0.5, "differ": ["q-2", "q-3"]}
    hf_success, cpp_success = [True, True, False, False], [True, True, True, False]
    runs = report["runs"]
    assert (runs["hf"]["successes"], runs["hf"]["success_rate"]) == (2, 0.5)
    assert (runs["llama.cpp"]["successes"], runs["llama.cpp"]["success_rate"]) == (3, 0.75)
    interval = bootstrap_ci(hf_success, cpp_success, seed=42, n=10000)
    assert {k: report["success_difference"][k] for k in interval} == interval and interval["diff"] == -0.25
    assert report["mcnemar"] == {"hf_only": 0, "llama_cpp_only": 1, "discordant": 1,
                                 "p": mcnemar(hf_success, cpp_success)["p"]}
    assert runs["hf"]["records"] == hf and runs["hf"]["records_sha256"] == sha256_of(root / hf)
    assert runs["hf"]["statuses"] == {"context_overflow": 1, "ok": 3}
    assert (runs["hf"]["backend"], runs["hf"]["dtype"], runs["hf"]["adapter"]) == ("hf", "bfloat16", ADAPTER)
    assert (runs["llama.cpp"]["backend"], runs["llama.cpp"]["adapters"]) == ("llama.cpp", GGUF)
    assert runs["llama.cpp"]["runtime"] == {"model_path": "models/Qwen3-1.7B-Q8_0.gguf",
                                            "build_info": "b11349"}
    assert runs["hf"]["runtime"]["versions"]["peft"] == "0.21.2"
    assert "2026-10-17" not in json.dumps(report)  # the runs' dates stay out: the same runs, the same bytes
    first = (root / REPORT).read_bytes()
    assert "agreement 2/4" in capsys.readouterr().out
    assert compared(hf, cpp) == 0 and (root / REPORT).read_bytes() == first


def test_two_records_agree_when_their_parsed_turns_are_equal_or_they_failed_alike():
    ok = {"id": "q", "status": "ok", "output": {"role": "assistant", "content": "hi"}, "usage": {"x": 1}}
    assert compare.turn_of(ok) == compare.turn_of({**ok, "usage": {"x": 2}, "model": "another"})
    assert compare.turn_of(ok) != compare.turn_of({**ok, "output": {"role": "assistant", "content": "hi!"}})
    error = {"id": "q", "status": "error", "error": "parse_tool_calls failed: X",
             "error_type": "HarnessError"}
    assert compare.turn_of(error) == compare.turn_of(dict(error))
    assert compare.turn_of(error) != compare.turn_of({**error, "error": "parse_tool_calls failed: Y"})
    overflow = {"id": "q", "status": "context_overflow"}
    assert compare.turn_of(overflow) == compare.turn_of(dict(overflow)) != compare.turn_of(error)


def test_a_llama_server_manifest_from_before_the_backends_counts_as_llama_cpp(tmp_path, monkeypatch):
    root = dev_root(tmp_path, monkeypatch)
    hf = write_run(root, "hf-b0", "hf", [answering("hi")] * 3, system="b0", lora_scale=0.0,
                   model="qwen3-1.7b:lora=0", adapter=None)
    m2 = write_run(root, "m2-b0", "llama.cpp", [answering("hi")] * 3, system="b0", lora_scale=0.0,
                   model="qwen3-1.7b:lora=0", backend=None, adapters=[])  # an M2 bare-server run
    assert "backend" not in json.loads((root / "runs/eval/m2-b0.manifest.json").read_text(encoding="utf-8"))
    assert compared(hf, m2) == 0
    report = json.loads((root / "results" / "backend-comparison-b0-scale0.json").read_text(encoding="utf-8"))
    assert report["agreement"]["rate"] == 1.0 and report["runs"]["llama.cpp"]["provider"] == "llama-server"


# b1's demonstrations as a run manifest records them (research_agent.eval.systems)
DEMOS = {"requests": ["req-train-1572", "req-train-0513", "req-train-0261"],
         "kinds": ["call", "clarify", "no_call"], "block_sha256_16": "27998771f1d7f9ba",
         "source": "data/raw/synth/synth.jsonl"}
B1 = {"system": "b1", "lora_scale": 0.0, "model": "qwen3-1.7b:lora=0+demos"}


@pytest.mark.parametrize(("hf_manifest", "cpp_manifest", "message"), [
    ({}, {"backend": "hf", "provider": "transformers"}, "--llama-cpp"),
    ({"backend": "llama.cpp", "provider": "llama-server"}, {}, "--hf"),
    ({}, {"system": "b4"}, "system"),
    ({}, {"lora_scale": 0.0}, "lora_scale"),
    ({}, {"prompt_hash": "0" * 16}, "prompt_hash"),
    ({}, {"schema_hash": "0" * 16}, "schema_hash"),
    ({**B1, "demonstrations": DEMOS}, {**B1, "demonstrations": {**DEMOS, "block_sha256_16": "0" * 16}},
     "demonstrations"),
    ({"items_sha256": "0" * 64}, {"items_sha256": "0" * 64}, "other items"),
    ({}, {"items_run": 3, "limit": 3}, "3 of the 4 items"),
    ({"system": "b2", "lora_scale": None}, {"system": "b2", "lora_scale": None}, "local system"),
])
def test_runs_that_cannot_be_compared_exit_2_and_write_nothing(hf_manifest, cpp_manifest, message, tmp_path,
                                                               monkeypatch, capsys):
    root = dev_root(tmp_path, monkeypatch)
    hf = write_run(root, "hf-ft", "hf", HF_TURNS, **hf_manifest)
    cpp = write_run(root, "cpp-ft", "llama.cpp", CPP_TURNS, **cpp_manifest)
    assert compared(hf, cpp) == 2
    assert message in capsys.readouterr().out
    assert not (root / "results").exists()


def test_records_whose_ids_differ_from_the_items_exit_2(tmp_path, monkeypatch, capsys):
    root = dev_root(tmp_path, monkeypatch)
    hf = write_run(root, "hf-ft", "hf", HF_TURNS)
    cpp = write_run(root, "cpp-ft", "llama.cpp", CPP_TURNS)
    lines = (root / cpp).read_text(encoding="utf-8").splitlines()
    (root / cpp).write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")  # q-4's record lost
    assert compared(hf, cpp) == 2
    assert "q-4" in capsys.readouterr().out and not (root / "results").exists()


def test_a_run_with_errors_its_model_did_not_cause_exits_3_and_the_models_own_errors_are_compared(
        tmp_path, monkeypatch, capsys):
    root = dev_root(tmp_path, monkeypatch)
    from research_agent.agent.types import HarnessError

    unreadable = HarnessError("parse_tool_calls failed: JSONDecodeError('Expecting value')")
    hf = write_run(root, "hf-ft", "hf", [searching(), unreadable, answering("x")])
    cpp = write_run(root, "cpp-ft", "llama.cpp", [searching(), unreadable, searching()])
    assert compared(hf, cpp) == 0
    report = json.loads((root / REPORT).read_text(encoding="utf-8"))
    assert report["agreement"]["differ"] == ["q-3"]  # the same unreadable completion agrees with itself
    crashed = RuntimeError("CUDA error: an illegal memory access was encountered")
    lost = write_run(root, "lost-ft", "hf", [searching(), crashed, answering("x")])
    assert compared(lost, cpp) == 3
    assert "RuntimeError" in capsys.readouterr().out


def test_test_material_exits_4_a_changed_dev_set_5_and_a_missing_manifest_1(tmp_path, monkeypatch, capsys):
    root = dev_root(tmp_path, monkeypatch)
    hf = write_run(root, "hf-ft", "hf", HF_TURNS)
    cpp = write_run(root, "cpp-ft", "llama.cpp", CPP_TURNS)
    (root / "data" / "eval" / "test.jsonl").write_text("not json: never parsed\n", encoding="utf-8")
    assert compared(hf, cpp, "--items", "data/eval/test.jsonl") == 4
    assert "locked test set" in capsys.readouterr().out
    dev = root / "data" / "eval" / "dev.jsonl"
    original = dev.read_bytes()
    dev.write_bytes(original + b"\n")
    assert compared(hf, cpp) == 5
    dev.write_bytes(original)
    (root / "runs" / "eval" / "cpp-ft.manifest.json").unlink()
    assert compared(hf, cpp) == 1
    assert "manifest" in capsys.readouterr().out
    assert not (root / "results").exists()


def test_test_items_under_another_name_exit_4_by_the_items_they_hold(tmp_path, monkeypatch, capsys):
    root = dev_root(tmp_path, monkeypatch)
    hf = write_run(root, "hf-ft", "hf", HF_TURNS)
    cpp = write_run(root, "cpp-ft", "llama.cpp", CPP_TURNS)
    copy = root / "data" / "eval" / "copy.jsonl"
    copy.write_text("".join(json.dumps({**i, "split": "test"}, ensure_ascii=False) + "\n" for i in ITEMS),
                    encoding="utf-8")
    assert compared(hf, cpp, "--items", "data/eval/copy.jsonl") == 4
    assert "holds test material" in capsys.readouterr().out and not (root / "results").exists()


def test_a_path_outside_the_current_folder_exits_2(tmp_path, monkeypatch, capsys):
    root = dev_root(tmp_path / "repo", monkeypatch)
    hf = write_run(root, "hf-ft", "hf", HF_TURNS)
    cpp = write_run(root, "cpp-ft", "llama.cpp", CPP_TURNS)
    assert compared(hf, cpp, "--out", str(tmp_path / "elsewhere.json")) == 2
    assert "outside the current folder" in capsys.readouterr().out
    assert not (tmp_path / "elsewhere.json").exists()


class FakeServer:
    """A llama-server stand-in for run_eval: a fixed completion and its /props."""

    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, prompt: str, *, lora_scale: float | None = None,
                 stop: list[str] | None = None) -> Completion:
        return Completion(text=self.text, raw={"usage": {"prompt_tokens": 9, "completion_tokens": 3}})

    def props(self) -> dict:
        return {"model_path": "models/Qwen3-1.7B-Q8_0.gguf", "build_info": "b11349"}


def test_the_runs_run_eval_writes_on_both_backends_compare_as_they_are_written(tmp_path, monkeypatch):
    root = dev_root(tmp_path, monkeypatch)
    text = '<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "hallucination"}}\n</tool_call>'
    adapter = peft_folder(root / "runs" / "train" / "ft" / "adapter")
    render = prompt_renderer(FakeTokenizer())
    client = HFClient(FakePeftModel(text, "base"), FakeHFTokenizer(), adapter=adapter, model_dir="models/x")
    assert run_eval.main(["data/eval/dev.jsonl", "--system", "ft", "--backend", "hf", "--adapter",
                          "runs/train/ft/adapter", "--out", "runs/eval/hf.jsonl", "--no-tokenizer"],
                         local_client=client, render=render) == 0
    assert run_eval.main(["data/eval/dev.jsonl", "--system", "ft", "--out", "runs/eval/cpp.jsonl",
                          "--no-tokenizer"], local_client=FakeServer(text), render=render) == 0
    assert compared("runs/eval/hf.jsonl", "runs/eval/cpp.jsonl") == 0
    report = json.loads((root / REPORT).read_text(encoding="utf-8"))
    assert report["agreement"]["rate"] == 1.0 and report["success_difference"]["diff"] == 0.0
    assert report["mcnemar"]["p"] == 1.0
    weights = sha256_of(adapter / "adapter_model.safetensors")
    assert report["runs"]["hf"]["adapter"] == {"path": "runs/train/ft/adapter", "safetensors_sha256": weights}


class AbsoluteServer(FakeServer):
    """FakeServer started with an absolute model path, which its /props reports as given."""

    def __init__(self, text: str, model: Path) -> None:
        super().__init__(text)
        self.model = model

    def props(self) -> dict:
        return {"model_path": str(self.model), "build_info": "b11349"}


class AdapterServer(AbsoluteServer):
    """AbsoluteServer that lists the GGUF adapter it loaded, by the absolute path llama-server was given."""

    def __init__(self, text: str, model: Path, gguf: Path) -> None:
        super().__init__(text, model)
        self.gguf = gguf

    def adapters(self) -> list[dict]:
        return [{"id": 0, "path": str(self.gguf), "scale": 1.0}]


def test_the_servers_absolute_adapter_is_reported_relative_and_one_outside_the_repository_exits_2(
        tmp_path, monkeypatch, capsys):
    root = dev_root(tmp_path / "repo", monkeypatch)
    text = '<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "hallucination"}}\n</tool_call>'
    render = prompt_renderer(FakeTokenizer())
    adapter = peft_folder(root / "runs" / "train" / "ft" / "adapter")
    client = HFClient(FakePeftModel(text, "base"), FakeHFTokenizer(), adapter=adapter, model_dir="models/x")
    assert run_eval.main(["data/eval/dev.jsonl", "--system", "ft", "--backend", "hf", "--adapter",
                          "runs/train/ft/adapter", "--out", "runs/eval/hf.jsonl", "--no-tokenizer"],
                         local_client=client, render=render) == 0

    def cpp_run(gguf: Path, out: str) -> str:
        gguf.parent.mkdir(parents=True, exist_ok=True)
        gguf.write_bytes(b"GGUF adapter")
        server = AdapterServer(text, root / "models" / "Qwen3-1.7B-Q8_0.gguf", gguf)
        assert run_eval.main(["data/eval/dev.jsonl", "--system", "ft", "--out", out, "--no-tokenizer"],
                             local_client=server, render=render) == 0
        return out

    gguf = root / "runs" / "train" / "ft" / "gguf" / "checkpoint-209-f16.gguf"
    inside = cpp_run(gguf, "runs/eval/cpp.jsonl")
    assert compared("runs/eval/hf.jsonl", inside) == 0
    written = (root / REPORT).read_text(encoding="utf-8")
    assert root.as_posix() not in written and "\\" not in written
    assert json.loads(written)["runs"]["llama.cpp"]["adapters"][0]["path"] == (
        "runs/train/ft/gguf/checkpoint-209-f16.gguf")
    (root / REPORT).unlink()
    outside = cpp_run(tmp_path / "elsewhere" / "checkpoint-209-f16.gguf", "runs/eval/cpp-elsewhere.jsonl")
    assert compared("runs/eval/hf.jsonl", outside) == 2
    assert "outside the repository" in capsys.readouterr().out and not (root / REPORT).exists()


def test_an_absolute_adapter_is_reported_relative_and_one_outside_the_repository_exits_2(
        tmp_path, monkeypatch, capsys):
    """The manifests record the adapter folder as --adapter gave it and the model as the client reports it:
    absolute paths inside the repository go into the report relative and in POSIX form; one outside it would
    put a machine path into results/, so that comparison is refused and nothing is written."""
    root = dev_root(tmp_path / "repo", monkeypatch)
    text = '<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "hallucination"}}\n</tool_call>'
    render = prompt_renderer(FakeTokenizer())

    def hf_run(adapter: Path, out: str) -> str:
        client = HFClient(FakePeftModel(text, "base"), FakeHFTokenizer(), adapter=adapter,
                          model_dir=root / "models" / "Qwen3-1.7B")
        assert run_eval.main(["data/eval/dev.jsonl", "--system", "ft", "--backend", "hf", "--adapter",
                              str(adapter), "--out", out, "--no-tokenizer"], local_client=client,
                             render=render) == 0
        return out

    hf = hf_run(peft_folder(root / "runs" / "train" / "ft" / "adapter"), "runs/eval/hf.jsonl")
    server = AbsoluteServer(text, root / "models" / "Qwen3-1.7B-Q8_0.gguf")
    assert run_eval.main(["data/eval/dev.jsonl", "--system", "ft", "--out", "runs/eval/cpp.jsonl",
                          "--no-tokenizer"], local_client=server, render=render) == 0
    assert compared(hf, "runs/eval/cpp.jsonl") == 0
    written = (root / REPORT).read_text(encoding="utf-8")
    assert root.as_posix() not in written and "\\" not in written  # repository-relative POSIX paths only
    runs = json.loads(written)["runs"]
    assert runs["hf"]["adapter"]["path"] == "runs/train/ft/adapter"
    assert runs["hf"]["runtime"]["model_path"] == "models/Qwen3-1.7B"
    assert runs["llama.cpp"]["runtime"]["model_path"] == "models/Qwen3-1.7B-Q8_0.gguf"
    (root / REPORT).unlink()
    elsewhere = hf_run(peft_folder(tmp_path / "elsewhere" / "adapter"), "runs/eval/hf-elsewhere.jsonl")
    assert compared(elsewhere, "runs/eval/cpp.jsonl") == 2
    assert "outside the repository" in capsys.readouterr().out and not (root / REPORT).exists()

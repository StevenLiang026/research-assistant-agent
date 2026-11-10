"""scripts/run_eval.py offline: the replay stand-in, a local system over a fake server, the cloud systems'
refusals and the budget stop; then scripts/score.py over what it wrote. Also the M3 guards (a test set is
refused, a locked set is lock-checked), b4, the adapter in the manifest, the timing sidecar and the hf
backend over a fake model."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from support.hf_fakes import FakeHFTokenizer, FakePeftModel, peft_folder
from support.llama_fakes import FakeLlama, Scripted, install_fake_family
from support.strict_deepseek import Answer, StrictDeepSeek
from train.tokenizer_fake import FakeTokenizer

from research_agent.agent.prompts import PROMPT_HASH, system_prompt
from research_agent.canonical import canonical_json
from research_agent.eval import systems
from research_agent.eval.lock import lock_path, sha256_of, write_lock
from research_agent.family.core import GREEDY, Decoding, Mode
from research_agent.llm.generation import request_seed
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


run_eval = load_script("run_eval")
score = load_script("score")

SYSTEM = {"role": "system", "content": system_prompt("2026-10-10")}


def item(rid: str, user: str, expected: dict, type_: str, lang: str = "zh") -> dict:
    return {"id": rid, "messages": [SYSTEM, {"role": "user", "content": user}], "expected": expected,
            "type": type_, "lang": lang, "family": f"f/{type_}", "split": "dev", "slots": {},
            "source": {"request": rid, "step": 1}}


SEARCH = {"name": "search_arxiv", "arguments": {"query": "hallucination", "max_results": 5},
          "arguments_json": "{\"query\": \"hallucination\", \"max_results\": 5}"}
ITEMS = [item("q-1", "找几篇幻觉(hallucination)的论文。", {"content": "", "tool_calls": [SEARCH]}, "call"),
         item("q-2", "Thanks, that's all for today.", {"content": "See you!", "tool_calls": []}, "no_call",
              "en"),
         item("q-3", "Find hallucination papers.", {"content": "", "tool_calls": [SEARCH]}, "call", "en")]


def write_items(tmp_path: Path, items: list[dict] = ITEMS) -> Path:
    path = tmp_path / "items.jsonl"
    path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    return path


def test_replay_writes_results_that_score_py_scores_at_100_percent(tmp_path, capsys):
    items = write_items(tmp_path)
    out = tmp_path / "runs" / "replay-dev.jsonl"
    assert run_eval.main([str(items), "--system", "replay", "--out", str(out), "--no-tokenizer"]) == 0
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["id"] for r in records] == ["q-1", "q-2", "q-3"] and {r["status"] for r in records} == {"ok"}
    summary = tmp_path / "summary.json"
    assert score.main([str(out), "--items", str(items), "--out", str(summary)]) == 0
    assert json.loads(summary.read_text(encoding="utf-8"))["summary"]["success_rate"] == 1.0


def test_the_manifest_records_the_system_the_hashes_the_items_and_the_date(tmp_path):
    items = write_items(tmp_path)
    out = tmp_path / "replay.jsonl"
    run_eval.main([str(items), "--system", "replay", "--out", str(out), "--no-tokenizer"])
    manifest = json.loads((tmp_path / "replay.manifest.json").read_text(encoding="utf-8"))
    assert manifest["system"] == "replay" and manifest["model"] == "replay" and manifest["lora_scale"] is None
    assert manifest["prompt_hash"] == PROMPT_HASH and manifest["schema_hash"] == SCHEMA_HASH
    assert manifest["items_sha256"] == sha256_of(items) and manifest["items_run"] == 3
    assert len(manifest["date"]) == 10 and manifest["statuses"] == {"ok": 3}
    assert manifest["server_props"] is None


def test_limit_runs_only_the_first_items(tmp_path):
    out = tmp_path / "r.jsonl"
    run_eval.main([str(write_items(tmp_path)), "--system", "replay", "--out", str(out), "--no-tokenizer",
                   "--limit", "2"])
    assert [json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()] == ["q-1", "q-2"]


class FakeServer:
    """A llama-server stand-in: a fixed completion, the requests recorded, /props answered."""

    def __init__(self, text: str) -> None:
        self.text, self.requests = text, []

    def complete(self, prompt: str, *, lora_scale: float | None = None,
                 stop: list[str] | None = None) -> Completion:
        self.requests.append((prompt, lora_scale))
        return Completion(text=self.text, raw={})

    def props(self) -> dict:
        return {"model_path": "models/Qwen3-1.7B-Q8_0.gguf", "build_info": "b11349"}


def test_ft_sends_scale_one_and_records_the_servers_props(tmp_path):
    server = FakeServer('<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "hallucination", '
                        '"max_results": 5}}\n</tool_call>')
    out = tmp_path / "ft.jsonl"
    code = run_eval.main([str(write_items(tmp_path)), "--system", "ft", "--out", str(out), "--no-tokenizer"],
                         local_client=server, render=prompt_renderer(FakeTokenizer()))
    assert code == 0 and {scale for _, scale in server.requests} == {1.0}
    manifest = json.loads((tmp_path / "ft.manifest.json").read_text(encoding="utf-8"))
    assert manifest["lora_scale"] == 1.0 and manifest["model"] == "qwen3-1.7b:lora=1"
    assert manifest["server_props"]["build_info"] == "b11349"
    assert all(prompt.endswith("<|im_start|>assistant\n") for prompt, _ in server.requests)


def test_a_cloud_model_without_a_price_stops_before_any_call_with_exit_2(tmp_path, capsys, monkeypatch):
    # b3 (deepseek-v4-pro) is priced since M7; a cloud model without a price is still refused
    monkeypatch.setitem(systems.CLOUD_MODELS, "b3", "deepseek-unpriced")
    code = run_eval.main([str(write_items(tmp_path)), "--system", "b3", "--out", str(tmp_path / "b3.jsonl"),
                          "--no-tokenizer", "--llm-dir", str(tmp_path / "llm")], openai_client=object())
    assert code == 2 and "no price" in capsys.readouterr().err
    assert not (tmp_path / "b3.jsonl").exists()


def test_cloud_thinking_off_is_passed_to_the_cloud_system_and_recorded(tmp_path):
    out = tmp_path / "b3.jsonl"
    code = run_eval.main([str(write_items(tmp_path)), "--system", "b3", "--out", str(out), "--no-tokenizer",
                          "--llm-dir", str(tmp_path / "llm"), "--stage", "M7", "--max-cost", "0",
                          "--cloud-thinking", "off"], openai_client=object())
    manifest = json.loads((tmp_path / "b3.manifest.json").read_text(encoding="utf-8"))
    assert code == 3 and manifest["thinking"] == "off" and manifest["stage"] == "M7"
    assert manifest["model"] == "deepseek-v4-pro"


def test_a_spent_stage_ends_the_run_with_budget_stop_records_and_exit_3(tmp_path):
    out = tmp_path / "b2.jsonl"
    code = run_eval.main([str(write_items(tmp_path)), "--system", "b2", "--out", str(out), "--no-tokenizer",
                          "--llm-dir", str(tmp_path / "llm"), "--max-cost", "0"], openai_client=object())
    statuses = [json.loads(line)["status"] for line in out.read_text(encoding="utf-8").splitlines()]
    assert code == 3 and statuses == ["budget_stop", "skipped", "skipped"]
    manifest = json.loads((tmp_path / "b2.manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "M3" and manifest["stage_cap_rmb"] == 0.0 and manifest["cost_rmb"] == 0.0


def test_malformed_items_exit_1(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n", encoding="utf-8")
    assert run_eval.main([str(bad), "--system", "replay", "--out", str(tmp_path / "o.jsonl"),
                          "--no-tokenizer"]) == 1


# A handwritten item's shape: its earlier step was scripted with call id c1, so no reasoning exists for it.
SCRIPTED_CALL = {"id": "c1", "type": "function",
                 "function": {"name": "search_arxiv", "arguments": SEARCH["arguments_json"]}}
SCRIPTED_STEP = {"role": "assistant", "content": "", "tool_calls": [SCRIPTED_CALL]}
HANDWRITTEN = {**item("q-4", "Find hallucination papers.", {"content": "Nothing found.", "tool_calls": []},
                      "no_call", "en"),
               "messages": [SYSTEM, {"role": "user", "content": "Find hallucination papers."}, SCRIPTED_STEP,
                            {"role": "tool", "tool_call_id": "c1", "content": "[]"}]}


def test_the_manifest_counts_the_cloud_turns_per_mode_and_a_local_run_has_no_modes(tmp_path):
    items = write_items(tmp_path, [*ITEMS[:2], HANDWRITTEN])
    server = StrictDeepSeek([Answer(calls=(("call_00_s", "search_arxiv", SEARCH["arguments"]),)),
                             Answer("See you!"), Answer("Nothing found.")])
    out = tmp_path / "b2.jsonl"
    code = run_eval.main([str(items), "--system", "b2", "--out", str(out), "--no-tokenizer",
                          "--llm-dir", str(tmp_path / "llm")], openai_client=server.sdk())
    records = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert code == 0 and [r["status"] for r in records] == ["ok", "ok", "ok"], records
    assert [r["mode"] for r in records] == ["thinking", "thinking", "non-thinking"] and server.refused == 0
    manifest = json.loads((tmp_path / "b2.manifest.json").read_text(encoding="utf-8"))
    assert manifest["modes"] == {"non-thinking": 1, "thinking": 2}
    ft = tmp_path / "ft.jsonl"
    render = prompt_renderer(FakeTokenizer())
    assert run_eval.main([str(items), "--system", "ft", "--out", str(ft), "--no-tokenizer"],
                         local_client=FakeServer("Nothing found."), render=render) == 0
    assert all("mode" not in json.loads(line) for line in ft.read_text(encoding="utf-8").splitlines())
    assert "modes" not in json.loads((tmp_path / "ft.manifest.json").read_text(encoding="utf-8"))


# ---- M3: who may open what, b4, the adapter in the manifest, the timing sidecar -------------------------


def fake_root(tmp_path: Path, monkeypatch) -> Path:
    """tmp_path as the repository root and the current folder: the guards only ever meet fake sets there."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(run_eval, "REPO", tmp_path)
    (tmp_path / "data" / "eval").mkdir(parents=True)
    return tmp_path


def test_the_test_sets_are_refused_with_exit_4_before_they_are_read(tmp_path, monkeypatch, capsys):
    root = fake_root(tmp_path, monkeypatch)
    for name in ("test.jsonl", "e2e-test.jsonl"):
        (root / "data" / "eval" / name).write_text("not json: never parsed\n", encoding="utf-8")
    for spelling in ("data/eval/test.jsonl", str(root / "data" / "eval" / "test.jsonl"),
                     "data/eval/../eval/e2e-test.jsonl"):
        code = run_eval.main([spelling, "--system", "replay", "--out", "runs/r.jsonl", "--no-tokenizer"])
        assert code == 4, spelling
        assert "locked test set" in capsys.readouterr().err
    assert not (root / "runs").exists()


def test_the_handwritten_test_items_are_refused_with_exit_4_before_they_are_read(tmp_path, monkeypatch,
                                                                                  capsys):
    """data/eval/handwritten.jsonl and handwritten-specs.jsonl hold the test set's handwritten items with
    their gold turns: well-formed items there are refused all the same, by their location."""
    root = fake_root(tmp_path, monkeypatch)
    for name in ("handwritten.jsonl", "handwritten-specs.jsonl"):
        write_items(root / "data" / "eval").rename(root / "data" / "eval" / name)
        code = run_eval.main([f"data/eval/{name}", "--system", "replay", "--out", "runs/r.jsonl",
                              "--no-tokenizer"])
        assert code == 4, name
        assert "handwritten items with their gold turns" in capsys.readouterr().err
    assert not (root / "runs").exists()


def replay(items: str | Path, out: str) -> int:
    return run_eval.main([str(items), "--system", "replay", "--out", out, "--no-tokenizer"])


# ITEMS as test material: drafted from test requests, of the test split
TEST_ITEMS = [{**i, "id": f"req-test-{n:04d}-s1", "split": "test",
               "source": {"request": f"req-test-{n:04d}", "step": 1}} for n, i in enumerate(ITEMS, 1)]


def test_a_copy_of_test_items_that_is_not_the_locked_file_is_refused_by_its_items_with_exit_4(
        tmp_path, monkeypatch, capsys):
    """A subset of a test set, a copy with CRLF line ends or without the final newline has neither the test
    set's location nor its bytes; the items it holds (split "test", ids of the test split) refuse it all the
    same, before the system is built or anything is written."""
    root = fake_root(tmp_path, monkeypatch)
    test_set = root / "data" / "eval" / "test.jsonl"
    raw = "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in TEST_ITEMS).encode("utf-8")
    test_set.write_bytes(raw)
    write_lock(test_set)
    copies = {"subset.jsonl": raw.split(b"\n", 1)[1], "crlf.jsonl": raw.replace(b"\n", b"\r\n"),
              "unterminated.jsonl": raw.rstrip(b"\n")}
    for name, data in copies.items():
        (root / name).write_bytes(data)
        assert replay(name, f"runs/{name}") == 4, name
        assert "holds test material" in capsys.readouterr().err
    write_items(root, [{**i, "split": "dev"} for i in TEST_ITEMS])  # relabelled: the ids still tell
    assert replay(root / "items.jsonl", "runs/relabelled.jsonl") == 4
    assert "req-test-0001-s1" in capsys.readouterr().err
    assert not (root / "runs").exists()


def manifest_of(out: Path) -> dict:
    return json.loads(out.with_name(out.stem + ".manifest.json").read_text(encoding="utf-8"))


def test_a_locked_set_is_read_through_its_lock_and_a_changed_one_exits_5(tmp_path, monkeypatch, capsys):
    root = fake_root(tmp_path, monkeypatch)
    dev = root / "data" / "eval" / "dev.jsonl"
    dev.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in ITEMS), encoding="utf-8")
    write_lock(dev)
    assert replay("data/eval/dev.jsonl", "runs/ok.jsonl") == 0
    assert manifest_of(root / "runs" / "ok.jsonl")["items_locked"] is True
    dev.write_bytes(dev.read_bytes() + b"\n")
    capsys.readouterr()
    assert replay("data/eval/dev.jsonl", "runs/changed.jsonl") == 5
    assert "does not match its lock" in capsys.readouterr().err
    lock_path(dev).unlink()  # the development set without its lock is refused, not read as a plain file
    assert replay(dev, "runs/unlocked.jsonl") == 5
    assert "has no lock" in capsys.readouterr().err
    other = write_items(root)  # any set with a lock beside it is checked the same way
    write_lock(other)
    other.write_bytes(other.read_bytes() + b"\n")
    assert replay(other, "runs/other.jsonl") == 5
    assert {p.name.split(".")[0] for p in (root / "runs").iterdir()} == {"ok"}  # nothing from the refused
    lock_path(other).unlink()
    assert replay(other, "runs/plain.jsonl") == 0
    assert manifest_of(root / "runs" / "plain.jsonl")["items_locked"] is False


class ListingServer(FakeServer):
    """FakeServer that also lists the adapter it loaded and keeps llama.cpp's timings of its latest
    completion, as LocalClient does."""

    def __init__(self, text: str, adapter: Path | None) -> None:
        super().__init__(text)
        self.listing = [] if adapter is None else [{"id": 0, "path": str(adapter), "scale": 1.0}]
        self.last_timings: dict | None = None

    def adapters(self) -> list:
        return self.listing

    def complete(self, prompt: str, *, lora_scale: float | None = None,
                 stop: list[str] | None = None) -> Completion:
        self.last_timings = {"prompt_n": len(self.requests) + 100, "predicted_ms": 25.0}
        return super().complete(prompt, lora_scale=lora_scale, stop=stop)


def test_b4_is_ft_over_the_b4_adapter_and_only_its_manifest_tells_them_apart(tmp_path):
    items = write_items(tmp_path)
    render = prompt_renderer(FakeTokenizer())
    adapters = {"ft": tmp_path / "ft-checkpoint-209-f16.gguf", "b4": tmp_path / "b4-checkpoint-139-f16.gguf"}
    servers = {}
    for system, adapter in adapters.items():
        adapter.write_bytes(f"GGUF {system}".encode())
        servers[system] = ListingServer("Nothing found.", adapter)
        assert run_eval.main([str(items), "--system", system, "--out", str(tmp_path / f"{system}.jsonl"),
                              "--no-tokenizer"], local_client=servers[system], render=render) == 0
    assert {scale for _, scale in servers["b4"].requests} == {1.0}
    assert (tmp_path / "b4.jsonl").read_bytes() == (tmp_path / "ft.jsonl").read_bytes()
    for system, adapter in adapters.items():
        manifest = manifest_of(tmp_path / f"{system}.jsonl")
        assert (manifest["system"], manifest["lora_scale"]) == (system, 1.0)
        assert manifest["model"] == "qwen3-1.7b:lora=1"
        assert manifest["adapters"] == [{"id": 0, "path": str(adapter), "sha256": sha256_of(adapter)}]
    unlisted = tmp_path / "unlisted.jsonl"  # a stand-in that cannot list adapters records None
    run_eval.main([str(items), "--system", "b0", "--out", str(unlisted), "--no-tokenizer"],
                  local_client=FakeServer("hi"), render=render)
    assert json.loads((tmp_path / "unlisted.manifest.json").read_text(encoding="utf-8"))["adapters"] is None


def ticking(step: float):
    """A clock that moves step seconds forward every time it is read."""
    now = [0.0]

    def clock() -> float:
        now[0] += step
        return now[0]

    return clock


def test_a_timing_sidecar_holds_the_wall_times_while_the_records_stay_byte_identical(tmp_path):
    items = write_items(tmp_path)
    adapter = tmp_path / "ft.gguf"
    adapter.write_bytes(b"GGUF ft")
    outs = []
    for step in (0.5, 2.0):
        out = tmp_path / f"clock-{step:g}" / "ft.jsonl"
        code = run_eval.main([str(items), "--system", "ft", "--out", str(out), "--no-tokenizer"],
                             local_client=ListingServer("Nothing found.", adapter),
                             render=prompt_renderer(FakeTokenizer()), clock=ticking(step))
        assert code == 0
        outs.append(out)
    assert outs[0].read_bytes() == outs[1].read_bytes()
    timings = [json.loads(out.with_name("ft.timing.json").read_text(encoding="utf-8")) for out in outs]
    assert timings[0] != timings[1]
    assert timings[0]["records"] == "ft.jsonl"
    assert [(i["id"], i["status"]) for i in timings[0]["items"]] == [("q-1", "ok"), ("q-2", "ok"),
                                                                     ("q-3", "ok")]
    first = timings[0]["items"][0]
    assert first["calls"] == [{"wall_ms": 500.0, "error": None, "paid": False, "cached": False,
                               "server": {"prompt_n": 100, "predicted_ms": 25.0}}]
    assert first["wall_ms"] > first["calls"][0]["wall_ms"]  # the item also builds and renders its context
    assert timings[1]["items"][0]["calls"][0]["wall_ms"] == 2000.0
    for out in outs:  # the records and the manifest carry no time of their own
        assert "wall" not in out.read_text(encoding="utf-8")
        assert "wall" not in out.with_name("ft.manifest.json").read_text(encoding="utf-8")


# ---- the hf backend (llm.hf): the M3 comparison of the HF model with the quantized one ---------------------

TOOL_TEXT = SERVER_TEXT = ('<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "hallucination", '
                           '"max_results": 5}}\n</tool_call>')


def records_of(out: Path) -> list[dict]:
    return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]


def test_ft_on_the_hf_backend_records_backend_dtype_and_adapter_and_answers_as_on_llama_cpp(tmp_path):
    items = write_items(tmp_path)
    adapter = peft_folder(tmp_path / "ft" / "trainer" / "checkpoint-209")
    client = HFClient(FakePeftModel(TOOL_TEXT, "Nothing found."), FakeHFTokenizer(), adapter=adapter,
                      model_dir="models/Qwen3-1.7B")
    render = prompt_renderer(FakeTokenizer())
    out = tmp_path / "hf-ft.jsonl"
    code = run_eval.main([str(items), "--system", "ft", "--backend", "hf", "--adapter", str(adapter),
                          "--out", str(out), "--no-tokenizer"], local_client=client, render=render)
    assert code == 0
    manifest = manifest_of(out)
    assert {k: manifest[k] for k in ("system", "backend", "provider", "model", "lora_scale", "dtype",
                                     "device")} == {
        "system": "ft", "backend": "hf", "provider": "transformers", "model": "qwen3-1.7b:lora=1",
        "lora_scale": 1.0, "dtype": "bfloat16", "device": "cuda:0"}
    assert manifest["adapter"] == {"path": adapter.as_posix(),
                                   "safetensors_sha256": sha256_of(adapter / "adapter_model.safetensors")}
    assert manifest["server_props"]["backend"] == "hf"
    assert manifest["server_props"]["model_path"] == "models/Qwen3-1.7B"
    assert "server" not in manifest and "adapters" not in manifest
    cpp = tmp_path / "cpp-ft.jsonl"
    assert run_eval.main([str(items), "--system", "ft", "--out", str(cpp), "--no-tokenizer"],
                         local_client=FakeServer(SERVER_TEXT), render=render) == 0
    assert manifest_of(cpp)["backend"] == "llama.cpp"
    assert [r["output"] for r in records_of(out)] == [r["output"] for r in records_of(cpp)]
    assert [r["model"] for r in records_of(out)] == [r["model"] for r in records_of(cpp)]
    timing = json.loads(out.with_name("hf-ft.timing.json").read_text(encoding="utf-8"))
    assert [call["server"] for item in timing["items"] for call in item["calls"]] == [None, None, None]


def test_the_hf_backend_refuses_a_gguf_and_ft_without_an_adapter_with_exit_2(tmp_path, capsys):
    items = write_items(tmp_path)
    gguf = tmp_path / "checkpoint-209-f16.gguf"
    gguf.write_bytes(b"GGUF ft")
    client = HFClient(FakePeftModel(TOOL_TEXT, "base"), FakeHFTokenizer(), adapter=gguf)
    render = prompt_renderer(FakeTokenizer())
    out = tmp_path / "hf.jsonl"
    assert run_eval.main([str(items), "--system", "ft", "--backend", "hf", "--adapter", str(gguf),
                          "--out", str(out), "--no-tokenizer"], local_client=client, render=render) == 2
    assert "GGUF" in capsys.readouterr().err
    assert run_eval.main([str(items), "--system", "b4", "--backend", "hf", "--out", str(out),
                          "--no-tokenizer"], local_client=client, render=render) == 2
    assert "adapter folder" in capsys.readouterr().err
    assert not out.exists()


def test_an_adapter_whose_weights_cannot_be_read_exits_2_and_writes_nothing(tmp_path, monkeypatch, capsys):
    """A corrupt adapter_model.safetensors: PEFT's loader raises safetensors' SafetensorError (an Exception,
    neither OSError nor ValueError), which run_eval reports as a system it cannot build here (exit 2)."""

    class SafetensorError(Exception):
        pass

    fake = types.ModuleType("safetensors")
    fake.SafetensorError = SafetensorError
    monkeypatch.setitem(sys.modules, "safetensors", fake)

    def load(model_dir: str, folder: Path) -> HFClient:
        raise SafetensorError("Error while deserializing header: header too large")

    monkeypatch.setattr("research_agent.llm.hf.load_client", load)
    adapter = peft_folder(tmp_path / "ft" / "adapter", weights=b"not safetensors")
    out = tmp_path / "hf.jsonl"
    code = run_eval.main([str(write_items(tmp_path)), "--system", "ft", "--backend", "hf", "--adapter",
                          str(adapter), "--out", str(out), "--no-tokenizer"],
                         render=prompt_renderer(FakeTokenizer()))
    assert code == 2
    assert "header too large" in capsys.readouterr().err and not out.exists()


@pytest.mark.parametrize(("options", "message"), [
    (["--system", "b2", "--backend", "hf"], "--backend hf runs the local systems only"),
    (["--system", "replay", "--backend", "hf"], "--backend hf runs the local systems only"),
    (["--system", "ft", "--adapter", "runs/train/ft/adapter"], "--adapter is for --backend hf")])
def test_hf_is_a_backend_of_the_local_systems_and_adapter_is_its_own_usage_errors_exit_2(options, message,
                                                                                          tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        run_eval.main([str(write_items(tmp_path)), *options, "--out", str(tmp_path / "r.jsonl"),
                       "--no-tokenizer"])
    assert stopped.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "r.jsonl").exists()


# ---- M4: --signals and ftr --------------------------------------------------------------------------------

SIGNAL_TEXT = ('<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "幻觉", "max_results": 5}}\n'
               '</tool_call>')
SIGNAL_TOKENS = ["<tool_call>", "\n", '{"name": "', "search", "_arxiv", '", "arguments": {"query": "', "幻",
                 "觉", '", "max_results": ', "5", "}}\n", "</tool_call>"]
SIGNAL_LPS = [-0.01, -0.02, -0.03, -0.2, -0.05, -0.04, -0.7, -0.1, -0.02, -1.2, -0.01, -0.01]
SIGNAL_ITEM = {"id": "q-1", "messages": [SYSTEM, {"role": "user", "content": "找幻觉的论文"}],
               "expected": {"content": "", "tool_calls": []}, "type": "call", "lang": "zh",
               "family": "f/call", "split": "dev", "slots": {}, "source": {"request": "q-1", "step": 1}}
# The record run_eval wrote for SIGNAL_ITEM before M4 (generated with the code of 2026-10-17, before turns
# had signals): without --signals the records stay exactly these bytes.
RECORD_BEFORE_M4 = (
    b'{"cached":false,"id":"q-1","item":{"expected":{"content":"","tool_calls":[]},"family":"f/call",'
    b'"lang":"zh","slots":{},"source":{"request":"q-1","step":1},"split":"dev","type":"call"},'
    b'"model":"qwen3-1.7b:lora=1","output":{"content":"","role":"assistant","tool_calls":[{"function":'
    b'{"arguments":"{\\"max_results\\":5,\\"query\\":\\"\xe5\xb9\xbb\xe8\xa7\x89\\"}","name":"search_arxiv"},'
    b'"id":"local_0","type":"function"}]},"paid":false,"status":"ok","usage":{"completion_tokens":12,'
    b'"prompt_tokens":7}}\n')


class TokenServer(FakeServer):
    """FakeServer whose completion carries tokens, log-probabilities and their bytes, and which lists the
    adapters it loaded."""

    def __init__(self, adapters: list[dict] | None = None) -> None:
        super().__init__(SIGNAL_TEXT)
        self.listing = adapters

    def complete(self, prompt: str, *, lora_scale: float | None = None,
                 stop: list[str] | None = None) -> Completion:
        self.requests.append((prompt, lora_scale))
        return Completion(text=SIGNAL_TEXT, tokens=list(SIGNAL_TOKENS), logprobs=list(SIGNAL_LPS),
                          token_bytes=[list(t.encode("utf-8")) for t in SIGNAL_TOKENS],
                          raw={"usage": {"prompt_tokens": 7, "completion_tokens": 12}})

    def adapters(self) -> list[dict]:
        return self.listing or []


def test_without_signals_a_local_record_is_byte_for_byte_the_record_before_m4(tmp_path):
    items, out = write_items(tmp_path, [SIGNAL_ITEM]), tmp_path / "ft.jsonl"
    code = run_eval.main([str(items), "--system", "ft", "--out", str(out), "--no-tokenizer"],
                         local_client=TokenServer(), render=prompt_renderer(FakeTokenizer()))
    assert code == 0 and out.read_bytes() == RECORD_BEFORE_M4


def test_signals_add_the_signals_of_every_local_record(tmp_path):
    import math

    items, out = write_items(tmp_path, [SIGNAL_ITEM]), tmp_path / "ft.jsonl"
    code = run_eval.main([str(items), "--system", "ft", "--out", str(out), "--no-tokenizer", "--signals"],
                         local_client=TokenServer(), render=prompt_renderer(FakeTokenizer()))
    (record,) = records_of(out)
    assert code == 0 and set(record["signals"]) == {"min_decision_prob", "mean_logprob", "hard_reason"}
    assert record["signals"]["min_decision_prob"] == pytest.approx(math.exp(-1.2))
    assert record["signals"]["hard_reason"] is None
    without = {k: v for k, v in record.items() if k != "signals"}
    assert (json.dumps(without, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode() \
        == RECORD_BEFORE_M4
    assert json.loads((tmp_path / "ft.manifest.json").read_text(encoding="utf-8"))["signals"] is True


def test_signals_are_for_the_local_systems_only(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        run_eval.main([str(write_items(tmp_path)), "--system", "b2", "--out", str(tmp_path / "o.jsonl"),
                       "--signals"])
    assert exc.value.code == 2 and "--signals" in capsys.readouterr().err


def test_ftr_runs_with_its_routing_configuration_in_the_manifest(tmp_path):
    adapter = tmp_path / "checkpoint-1668-f16.gguf"
    adapter.write_bytes(b"GGUF ft")
    items = write_items(tmp_path, [SIGNAL_ITEM])
    config = {"signal": "min_decision_prob", "threshold": 0.01, "dev_sha256": sha256_of(items),
              "adapter_sha256": sha256_of(adapter), "cloud_model": "deepseek-flash", "rule_met": True}
    routing = tmp_path / "routing-m4.json"
    routing.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    out = tmp_path / "ftr.jsonl"
    server = TokenServer([{"id": 0, "path": str(adapter), "scale": 1.0}])
    code = run_eval.main([str(items), "--system", "ftr", "--out", str(out), "--no-tokenizer",
                          "--llm-dir", str(tmp_path / "llm"), "--routing-config", str(routing),
                          "--dev-set", str(items)],
                         local_client=server, openai_client=StrictDeepSeek([]).sdk(),
                         render=prompt_renderer(FakeTokenizer()))
    (record,) = records_of(out)
    assert code == 0 and record["route"]["answered_by"] == "local" and record["model"] == "qwen3-1.7b:lora=1"
    manifest = json.loads((tmp_path / "ftr.manifest.json").read_text(encoding="utf-8"))
    assert manifest["routing"]["sha256"] == sha256_of(routing) and manifest["cost_rmb"] == 0.0
    assert manifest["system"] == "ftr"


# ---- M7: the family path, --mode, --seed and --reasoning-budget (docs/m7-家族层设计.md §7) ----------------

SAMPLING = Decoding(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
THOUGHT = Scripted(("Need", " a search.", "\n</think>\n\n", SIGNAL_TEXT))


def test_the_flags_set_the_generation_and_the_manifest_records_it_with_the_family(tmp_path, monkeypatch):
    install_fake_family(monkeypatch, tmp_path / "model",
                        decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY})
    items, out = write_items(tmp_path, [SIGNAL_ITEM]), tmp_path / "think.jsonl"
    fake = FakeLlama([THOUGHT])
    code = run_eval.main([str(items), "--system", "b0", "--out", str(out), "--tokenizer",
                          str(tmp_path / "model"), "--mode", "think", "--seed", "3", "--reasoning-budget",
                          "16"], local_client=fake.client())
    assert code == 0
    manifest = manifest_of(out)
    assert manifest["generation"] == {
        "mode": "think", "decoding": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                                      "presence_penalty": 0.0},
        "reasoning_budget": 16, "answer_tokens": 512, "base_seed": 3, "sample": 0}
    assert manifest["family"]["name"] == "fake-hybrid" and manifest["model"] == "model:lora=0"
    assert manifest["context_check"] == str(tmp_path / "model") and manifest["temperature"] == 0.6
    (record,) = records_of(out)
    (request,) = fake.requests
    seed = request_seed(3, request["prompt"], 1, 0)
    assert record["reasoning"] == "Need a search." and record["thinking"] == {
        "mode": "think", "reasoning_tokens": 3, "answer_tokens": 1, "forced": False, "phases": 1,
        "seeds": [seed]}
    assert record["output"]["tool_calls"][0]["function"]["name"] == "search_arxiv"
    assert request["max_tokens"] == 16 and request["seed"] == seed and request["top_k"] == 20
    assert "mode" not in record  # a cloud turn's DeepSeek mode: a local record has none


def test_without_the_flags_the_familys_inference_mode_runs_and_the_records_are_the_legacy_paths(
        tmp_path, monkeypatch):
    tokenizer = install_fake_family(monkeypatch, tmp_path / "Qwen3-1.7B")  # no_think, greedy: as before M7
    folder = str(tmp_path / "Qwen3-1.7B")
    adapter = tmp_path / "ft.gguf"
    adapter.write_bytes(b"GGUF ft")
    listed = [{"id": 0, "path": str(adapter), "scale": 1.0}]
    items = write_items(tmp_path, [SIGNAL_ITEM])
    outs = {}
    for name, extra in (("family", {}), ("legacy", {"render": prompt_renderer(tokenizer)})):
        fake = FakeLlama([Scripted(tuple(SIGNAL_TOKENS), logprobs=tuple(SIGNAL_LPS))], adapters=listed)
        outs[name] = tmp_path / f"{name}.jsonl"
        code = run_eval.main([str(items), "--system", "ft", "--out", str(outs[name]), "--tokenizer", folder,
                              "--no-tokenizer", "--signals"], local_client=fake.client(), **extra)
        assert code == 0
    assert outs["family"].read_bytes() == outs["legacy"].read_bytes()
    (record,) = records_of(outs["family"])
    assert record["model"] == "qwen3-1.7b:lora=1" and "reasoning" not in record and "thinking" not in record
    family, legacy = manifest_of(outs["family"]), manifest_of(outs["legacy"])
    assert family["generation"]["mode"] == "no_think"
    assert family["generation"]["decoding"]["temperature"] == 0.0
    assert family["family"]["name"] == "fake-hybrid" and family["context_check"] is None
    assert "generation" not in legacy and "family" not in legacy
    assert {k: v for k, v in family.items() if k not in ("generation", "family")} == legacy


@pytest.mark.parametrize(("options", "message"), [
    (["--system", "b2", "--mode", "think"], "--mode"),
    (["--system", "replay", "--reasoning-budget", "64"], "--reasoning-budget"),
    (["--system", "ft", "--backend", "hf", "--mode", "think"], "--mode think")])
def test_the_generation_flags_are_for_the_local_systems_on_llama_server(options, message, tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        run_eval.main([str(write_items(tmp_path)), *options, "--out", str(tmp_path / "r.jsonl"),
                       "--no-tokenizer"])
    assert stopped.value.code == 2 and message in capsys.readouterr().err


def test_a_mode_the_family_does_not_have_exits_2(tmp_path, monkeypatch, capsys):
    install_fake_family(monkeypatch, tmp_path / "model", "none")
    code = run_eval.main([str(write_items(tmp_path)), "--system", "b0", "--out", str(tmp_path / "r.jsonl"),
                          "--tokenizer", str(tmp_path / "model"), "--mode", "think"],
                         local_client=FakeLlama([]).client())
    assert code == 2 and "has no think mode" in capsys.readouterr().err
    assert not (tmp_path / "r.jsonl").exists()


# ---- M7 WP-E: --greedy, the preregistration's extra greedy thinking run (docs/m7-底座对比预登记.md, 解码) --


def test_greedy_thinks_with_greedy_decoding_sends_no_seed_and_records_the_settings(tmp_path, monkeypatch):
    install_fake_family(monkeypatch, tmp_path / "model",
                        decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY})
    items, out = write_items(tmp_path, [SIGNAL_ITEM]), tmp_path / "greedy.jsonl"
    fake = FakeLlama([THOUGHT])
    code = run_eval.main([str(items), "--system", "b0", "--out", str(out), "--tokenizer",
                          str(tmp_path / "model"), "--greedy", "--reasoning-budget", "16"],
                         local_client=fake.client())
    assert code == 0
    manifest = manifest_of(out)
    assert manifest["generation"] == {
        "mode": "think", "decoding": {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
                                      "presence_penalty": 0.0},
        "reasoning_budget": 16, "answer_tokens": 512, "base_seed": 0, "sample": 0}
    assert manifest["temperature"] == 0.0
    (request,) = fake.requests
    assert request["temperature"] == 0.0 and request["max_tokens"] == 16
    assert not {"seed", "top_p", "top_k", "min_p", "presence_penalty"} & set(request)
    (record,) = records_of(out)
    assert record["reasoning"] == "Need a search." and record["thinking"]["mode"] == "think"
    assert record["thinking"]["seeds"] is None
    # --mode think --greedy says the same thing twice; the run is the same
    fake = FakeLlama([THOUGHT])
    again = tmp_path / "again.jsonl"
    assert run_eval.main([str(items), "--system", "b0", "--out", str(again), "--tokenizer",
                          str(tmp_path / "model"), "--mode", "think", "--greedy", "--reasoning-budget", "16"],
                         local_client=fake.client()) == 0
    assert again.read_bytes() == out.read_bytes() and fake.requests == [request]


@pytest.mark.parametrize(("options", "message"), [
    (["--system", "b2", "--greedy"], "--greedy"),
    (["--system", "replay", "--greedy"], "--greedy"),
    (["--system", "ft", "--greedy", "--mode", "no_think"], "--mode no_think"),
    (["--system", "ft", "--backend", "hf", "--greedy"], "--greedy")])
def test_greedy_is_a_thinking_run_of_the_local_systems_on_llama_server(options, message, tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        run_eval.main([str(write_items(tmp_path)), *options, "--out", str(tmp_path / "r.jsonl"),
                       "--no-tokenizer"])
    assert stopped.value.code == 2 and message in capsys.readouterr().err
    assert not (tmp_path / "r.jsonl").exists()


def test_greedy_on_a_family_that_never_thinks_exits_2(tmp_path, monkeypatch, capsys):
    install_fake_family(monkeypatch, tmp_path / "model", "none")
    code = run_eval.main([str(write_items(tmp_path)), "--system", "b0", "--out", str(tmp_path / "r.jsonl"),
                          "--tokenizer", str(tmp_path / "model"), "--greedy"],
                         local_client=FakeLlama([]).client())
    assert code == 2 and "has no think mode" in capsys.readouterr().err
    assert not (tmp_path / "r.jsonl").exists()


# ---- M7 WP2, WP3 and WP4: --votes, --rules and --glossary (docs/m7-本地提升计划.md) ------------------------

RETRIED = {"id": "call_0", "type": "function",
           "function": {"name": "search_arxiv", "arguments": "{\"query\": \"hallucination\"}"}}
# A history whose last step met a transient error: the runtime rules would retry the call (agent.rules).
TRANSIENT_ITEM = {"id": "q-9", "messages": [SYSTEM, {"role": "user", "content": "Find hallucination papers."},
                                            {"role": "assistant", "content": "", "tool_calls": [RETRIED]},
                                            {"role": "tool", "tool_call_id": "call_0",
                                             "content": "{\"error\":\"RATE_LIMITED\"}"}],
                  "expected": {"content": "", "tool_calls": [
                      {"name": "search_arxiv", "arguments": {"query": "hallucination"}}]},
                  "type": "recover", "lang": "en", "family": "f/recover", "split": "dev", "slots": {},
                  "source": {"request": "q-9", "step": 2}}
REWRITTEN = ('<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "LLM hallucination"}}\n'
             '</tool_call>')
SECOND_THOUGHT = Scripted(("Retry", " it.", "\n</think>\n\n", REWRITTEN))
VOLATILE = ("date", "git_commit", "git_dirty")  # the manifest's keys that differ from run to run
# What run_eval wrote for these two items in think mode, seed 3, before --votes, --rules and --glossary
# existed (the code of 2026-10-30, before WP2 and WP3): the SHA-256 of the records, of the timing sidecar and
# of the manifest without VOLATILE as canonical JSON. Without the new flags every output stays these bytes.
BEFORE_WP23 = {"records": "0f35810dad7e26001243bd047e530b616d2f4b27135e250237333488a36f3267",
               "timing": "b177b9898c7bb224e5db60fef63be083e35cfc4446e716292ba9537176a76f84",
               "manifest": "95f7402dea5211c83c5a0bd794ba63a686d0a28be5b980eeb38a2cc8e3caf73c"}


def wp23_run(tmp_path: Path, monkeypatch, script: list[Scripted], *flags: str) -> tuple[int, dict]:
    """b0 over SIGNAL_ITEM and TRANSIENT_ITEM in think mode with seed 3 and the signals, from tmp_path (so the
    manifest holds relative paths), the clock ticking 0.5 s; (the exit, {"records", "timing", "manifest":
    the three outputs' bytes, the manifest without VOLATILE, "requests": the server's requests})."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(tmp_path)
    install_fake_family(monkeypatch, tmp_path / "model",
                        decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY})
    write_items(tmp_path, [SIGNAL_ITEM, TRANSIENT_ITEM])
    fake = FakeLlama(script)
    code = run_eval.main(["items.jsonl", "--system", "b0", "--out", "runs/wp23.jsonl", "--tokenizer", "model",
                          "--mode", "think", "--seed", "3", "--reasoning-budget", "16", "--signals", *flags],
                         local_client=fake.client(), clock=ticking(0.5))
    out = tmp_path / "runs" / "wp23.jsonl"
    if code != 0:
        return code, {"requests": fake.requests}
    manifest = {k: v for k, v in manifest_of(out).items() if k not in VOLATILE}
    return code, {"records": out.read_bytes(), "timing": out.with_name("wp23.timing.json").read_bytes(),
                  "manifest": canonical_json(manifest).encode("utf-8"), "requests": fake.requests}


def digests(outputs: dict) -> dict:
    return {k: hashlib.sha256(outputs[k]).hexdigest() for k in ("records", "timing", "manifest")}


def test_without_the_new_flags_every_output_is_the_bytes_before_them(tmp_path, monkeypatch):
    code, outputs = wp23_run(tmp_path, monkeypatch, [THOUGHT, SECOND_THOUGHT])
    assert code == 0 and digests(outputs) == BEFORE_WP23


@pytest.mark.parametrize(("options", "message"), [
    (["--system", "b2", "--votes", "5"], "wrap the local systems only"),
    (["--system", "replay", "--rules"], "wrap the local systems only"),
    (["--system", "b3", "--glossary", "zh-en.json"], "wrap the local systems only"),
    (["--system", "ft", "--vote-mode", "final"], "--vote-mode chooses the keys of a vote"),
    (["--system", "ft", "--votes", "0"], "expected a whole number from 1"),
    (["--system", "ft", "--votes", "5", "--greedy"], "a vote's samples sample"),
    (["--system", "ft", "--votes", "5", "--backend", "hf"], "a vote samples on llama-server")])
def test_the_wrapper_flags_are_for_the_local_systems_and_a_vote_samples_on_llama_server(options, message,
                                                                                        tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        run_eval.main([str(write_items(tmp_path)), *options, "--out", str(tmp_path / "r.jsonl"),
                       "--no-tokenizer"])
    assert stopped.value.code == 2 and message in capsys.readouterr().err
    assert not (tmp_path / "r.jsonl").exists()


def test_a_vote_over_a_decoding_that_does_not_sample_exits_2(tmp_path, monkeypatch, capsys):
    install_fake_family(monkeypatch, tmp_path / "model")  # no_think, greedy: nothing to sample
    code = run_eval.main([str(write_items(tmp_path)), "--system", "b0", "--out", str(tmp_path / "r.jsonl"),
                          "--tokenizer", str(tmp_path / "model"), "--votes", "3"],
                         local_client=FakeLlama([]).client())
    assert code == 2 and "votes need a decoding that samples" in capsys.readouterr().err
    assert not (tmp_path / "r.jsonl").exists()


def test_the_rules_rewrite_the_records_and_only_the_manifests_model_and_rules_change(tmp_path, monkeypatch):
    from research_agent.agent.rules import RULES, RULES_VERSION

    code, outputs = wp23_run(tmp_path, monkeypatch, [THOUGHT, SECOND_THOUGHT], "--rules")
    assert code == 0
    first, retried = (json.loads(line) for line in outputs["records"].decode("utf-8").splitlines())
    assert first["rules"] == [] and retried["rules"][0]["rule"] == "transient_retry"
    assert retried["output"] == {"role": "assistant", "content": "", "tool_calls": [
        {"id": "pp_0", "type": "function",
         "function": {"name": "search_arxiv", "arguments": "{\"query\": \"hallucination\"}"}}]}
    assert retried["rules"][0]["after"] == retried["output"]
    assert retried["signals"]["min_decision_prob"] is not None  # the signals of the answer the model gave
    manifest = json.loads(outputs["manifest"])
    assert manifest["rules"] == {"version": RULES_VERSION, "names": list(RULES)}
    assert manifest["model"] == "model:lora=0+rules"
    before = json.loads(wp23_run(tmp_path / "plain", monkeypatch, [THOUGHT, SECOND_THOUGHT])[1]["manifest"])
    assert {k: v for k, v in manifest.items() if k not in ("rules", "model")} == {
        k: v for k, v in before.items() if k != "model"}
    # without the rule's change the records are the records before the rules, but for "rules"
    plain = [json.loads(line) for line in wp23_run(tmp_path / "again", monkeypatch,
                                                   [THOUGHT, SECOND_THOUGHT])[1]["records"].splitlines()]
    assert {k: v for k, v in first.items() if k != "rules"} == plain[0]


def test_the_glossary_note_reaches_the_prompts_and_the_manifest_names_the_glossary(tmp_path, monkeypatch):
    glossary = tmp_path / "zh-en.json"
    glossary.write_text(json.dumps({"幻觉": "hallucination"}, ensure_ascii=False), encoding="utf-8")
    code, noted = wp23_run(tmp_path, monkeypatch, [THOUGHT, SECOND_THOUGHT], "--glossary", str(glossary))
    plain = wp23_run(tmp_path / "plain", monkeypatch, [THOUGHT, SECOND_THOUGHT])[1]
    assert code == 0
    # the zh item's prompt carries the note and is longer; the en item has no term and is sent as it was
    assert len(noted["requests"][0]["prompt"]) > len(plain["requests"][0]["prompt"])
    assert noted["requests"][1] == plain["requests"][1]
    manifest = json.loads(noted["manifest"])
    assert manifest["glossary"] == {"path": glossary.as_posix(), "sha256": sha256_of(glossary), "terms": 1}
    assert manifest["model"] == "model:lora=0+glossary" and "rules" not in manifest

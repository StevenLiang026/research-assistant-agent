"""The evaluation systems (research_agent.eval.systems): b1's demonstrations, the replay stand-in, and how the
local and cloud systems are built, offline with fakes."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path, PurePosixPath

import httpx
import pytest
from support.fakes import ScriptedModel, turn
from support.family_fakes import model_folder, token_ids
from support.hf_fakes import FakeCausalLM, FakeHFTokenizer, FakePeftModel, peft_folder
from support.llama_fakes import FakeLlama, FakeToolFormat, Scripted, install_fake_family, scripted
from train.tokenizer_fake import FakeTokenizer

from research_agent.agent.context import build_context
from research_agent.agent.models import CloudModel, LocalModel
from research_agent.agent.types import HarnessError, parse_arguments
from research_agent.eval import systems
from research_agent.eval.lock import sha256_of
from research_agent.eval.runner import run_items, write_results
from research_agent.eval.systems import (
    DEMONSTRATIONS,
    DemonstratedModel,
    ReplayModel,
    SystemUnavailable,
    build_system,
    demonstrated_counter,
    demonstration_reply,
    demonstrations_block,
    load_demonstrations,
    with_demonstrations,
)
from research_agent.family.core import GREEDY, Decoding, Mode
from research_agent.llm.generation import GenerationSettings
from research_agent.llm.hf import HFClient
from research_agent.llm.local import Completion, LocalClient
from research_agent.normalize import normalize_text
from research_agent.toolcalls import parse_tool_calls
from research_agent.tools.schemas import DEFAULTS, MODEL_TOOLS
from research_agent.train.samples import prompt_renderer, render_sample

SYSTEM = {"role": "system", "content": "你是科研助理,今天是 2026-10-10。"}


def record(rid: str, user: str, assistant: dict, run: str = "train") -> dict:
    return {"id": rid, "run": run, "history": "agent", "messages": [SYSTEM, {"role": "user", "content": user},
                                                                    assistant]}


def call_message(name: str, arguments: dict) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}]}


SEARCH = call_message("search_arxiv", {"query": "KV cache", "date_from": "2026-07-04"})
ASK = call_message("ask_clarification", {"question": "Which paper?", "missing": ["paper_id"]})
ANSWER = {"role": "assistant", "content": "No: query is required and has no default."}


def training_set(tmp_path: Path, *, ask_run: str = "train", split: str = "train") -> tuple[Path, Path]:
    (call_id, _), (ask_id, _), (answer_id, _) = DEMONSTRATIONS
    rows = [record("req-train-9999", "unrelated", ANSWER), record(call_id, "搜 KV cache 的论文", SEARCH),
            record(ask_id, "Note the paper I just closed.", ASK, run=ask_run),
            record(answer_id, "Does query have a default?", ANSWER)]
    data = tmp_path / "synth.jsonl"
    data.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    requests = tmp_path / "requests.jsonl"
    requests.write_text("".join(json.dumps({"id": rid, "split": split if rid == ask_id else "train"}) + "\n"
                                for rid, _ in DEMONSTRATIONS), encoding="utf-8")
    return data, requests


def test_the_three_demonstrations_are_one_call_one_clarification_and_one_answer(tmp_path):
    demos = load_demonstrations(*training_set(tmp_path))
    assert [d.kind for d in demos] == ["call", "clarify", "no_call"]
    assert [d.request for d in demos] == [rid for rid, _ in DEMONSTRATIONS]
    assert demos[0].user == "搜 KV cache 的论文"


REPO = Path(__file__).resolve().parents[2]


def test_each_real_demonstration_asks_for_exactly_what_its_request_asks():
    if not (REPO / systems.TRAINING_SET).is_file():
        pytest.skip("the assembled training set is local only (data/raw/ is not tracked)")
    requests = {r["id"]: r for r in (json.loads(line) for line in (REPO / systems.REQUESTS_FILE).read_text(
        encoding="utf-8").splitlines() if line.strip())}
    defaults = DEFAULTS["search_arxiv"]
    checked = []
    for demo in load_demonstrations(REPO / systems.TRAINING_SET, REPO / systems.REQUESTS_FILE):
        slots = requests[demo.request]["slots"]
        for raw in demo.assistant.get("tool_calls") or []:
            name, args = raw["function"]["name"], parse_arguments(raw["function"]["arguments"])
            checked.append(name)
            if name == "search_arxiv":
                wanted = {key: slots[key] for key in ("date_from", "date_to", "categories") if key in slots}
                if "count" in slots:
                    wanted["max_results"] = slots["count"]
                given = {key: value for key, value in args.items() if key != "query"}
                assert {**defaults, **given} == {**defaults, **wanted}, demo.request
                assert normalize_text(args["query"]).casefold() == normalize_text(slots["topic"]).casefold()
            else:
                assert name == "ask_clarification" and args["missing"] == slots["missing"], demo.request
    assert checked == ["search_arxiv", "ask_clarification"]


def test_a_demonstration_outside_the_train_split_is_refused(tmp_path):
    with pytest.raises(ValueError, match="train split"):
        load_demonstrations(*training_set(tmp_path, split="dev"))
    with pytest.raises(ValueError, match="train split"):
        load_demonstrations(*training_set(tmp_path, ask_run="dev"))


def test_a_demonstration_of_the_wrong_kind_is_refused(tmp_path):
    data, requests = training_set(tmp_path)
    swapped = [(DEMONSTRATIONS[0][0], "no_call")]
    with pytest.raises(ValueError, match="call step, not a no_call step"):
        load_demonstrations(data, requests, swapped)


def test_the_block_writes_calls_as_qwen3_does_so_the_parser_reads_them_back(tmp_path):
    demos = load_demonstrations(*training_set(tmp_path))
    block = demonstrations_block(demos)
    assert block.startswith("\n\n") and block.count("示例 ") == 3
    assert parse_tool_calls(demonstration_reply(SEARCH)) == [
        {"name": "search_arxiv", "arguments": {"query": "KV cache", "date_from": "2026-07-04"}}]
    assert "Does query have a default?" in block and "No: query is required" in block


def test_the_block_is_appended_to_the_system_message_identically_and_the_input_is_left_alone():
    first = [SYSTEM, {"role": "user", "content": "a"}]
    second = [SYSTEM, {"role": "user", "content": "b"}, {"role": "assistant", "content": "x"}]
    a, b = with_demonstrations(first, "\n\nBLOCK"), with_demonstrations(second, "\n\nBLOCK")
    assert a[0]["content"] == b[0]["content"] == SYSTEM["content"] + "\n\nBLOCK"
    assert first[0] == SYSTEM and a[1:] == first[1:]
    with pytest.raises(HarnessError):
        with_demonstrations([{"role": "user", "content": "a"}], "\n\nBLOCK")


def test_the_demonstrated_model_and_its_counter_see_the_block():
    inner = ScriptedModel([turn(content="hi")])
    model = DemonstratedModel(inner, "\n\nBLOCK")
    model.next_turn([SYSTEM, {"role": "user", "content": "a"}], [])
    assert inner.contexts[0][0]["content"].endswith("BLOCK")
    counted = demonstrated_counter(lambda messages, tools: len(messages[0]["content"]), "\n\nBLOCK")
    assert counted([SYSTEM], []) == len(SYSTEM["content"]) + len("\n\nBLOCK")


class RecordingClient:
    def __init__(self, text: str) -> None:
        self.text, self.requests = text, []

    def complete(self, prompt: str, *, lora_scale: float | None = None,
                 stop: list[str] | None = None) -> Completion:
        self.requests.append({"prompt": prompt, "lora_scale": lora_scale, "stop": stop})
        return Completion(text=self.text, raw={"usage": {"prompt_tokens": 9, "completion_tokens": 3}})


ITEM = {"id": "i1", "messages": [SYSTEM, {"role": "user", "content": "找 KV cache 的论文"}],
        "expected": {"content": "", "tool_calls": [{"name": "search_arxiv",
                                                     "arguments": {"query": "KV cache"},
                                                     "arguments_json": "{\"query\": \"KV cache\"}"}]}}


@pytest.mark.parametrize(("system", "scale"), [("b0", 0.0), ("ft", 1.0), ("b4", 1.0)])
def test_local_systems_send_their_lora_scale_and_the_training_time_prompt(system, scale):
    tokenizer = FakeTokenizer()
    client = RecordingClient('<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "KV cache"}}\n'
                             '</tool_call>')
    built = build_system(system, local_client=client, render=prompt_renderer(tokenizer))
    assert isinstance(built.model, LocalModel) and built.info["lora_scale"] == scale
    (result,) = run_items([ITEM], built.model)
    context = build_context(ITEM["messages"])
    target = {"role": "assistant", "content": "", "tool_calls": result["output"]["tool_calls"]}
    expected_prompt = render_sample(tokenizer, context, target, tools=MODEL_TOOLS)["prompt"]
    assert client.requests == [{"prompt": expected_prompt, "lora_scale": scale, "stop": ["<|im_end|>"]}]
    assert result["output"]["tool_calls"][0]["function"]["name"] == "search_arxiv"


def test_b1_is_b0_with_the_demonstrations_in_every_prompt(tmp_path):
    data, requests = training_set(tmp_path)
    client = RecordingClient("好的。")
    built = build_system("b1", local_client=client, render=prompt_renderer(FakeTokenizer()),
                         training_set=data, requests=requests, count_tokens=lambda m, t: 1)
    run_items([ITEM, {**ITEM, "id": "i2"}], built.model, count_tokens=built.count_tokens)
    block = demonstrations_block(load_demonstrations(data, requests))
    assert [r["lora_scale"] for r in client.requests] == [0.0, 0.0]
    assert all(SYSTEM["content"] + block in r["prompt"] for r in client.requests)
    assert built.info["demonstrations"]["requests"] == [rid for rid, _ in DEMONSTRATIONS]


def test_b1_without_the_training_set_is_unavailable(tmp_path):
    with pytest.raises(SystemUnavailable, match="training set"):
        build_system("b1", local_client=RecordingClient(""), render=lambda m, t: "",
                     training_set=tmp_path / "missing.jsonl", requests=None)


@pytest.mark.parametrize(("system", "options"), [("b0", {}), ("ft", {}), ("b4", {}),
                                                 ("b1", {"render": lambda m, t: ""})])
def test_a_local_system_that_cannot_be_built_leaves_no_client_open(system, options, monkeypatch, tmp_path):
    opened = []

    class OpenedClient:
        def __init__(self, server: str) -> None:
            self.closed = False
            opened.append(self)

        def health(self) -> bool:
            return True

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("research_agent.llm.local.LocalClient", OpenedClient)
    with pytest.raises(SystemUnavailable):  # b0, ft and b4: no tokenizer; b1: no training set
        build_system(system, tokenizer=None, training_set=tmp_path / "missing.jsonl", **options)
    assert all(client.closed for client in opened)


class ListingClient(RecordingClient):
    """RecordingClient over a server that lists the adapters it loaded (/lora-adapters)."""

    def __init__(self, text: str, adapters: list[dict]) -> None:
        super().__init__(text)
        self.listing = adapters

    def adapters(self) -> list[dict]:
        return self.listing


def test_a_local_system_records_the_adapters_the_server_loaded_with_their_sha256(tmp_path):
    adapter = tmp_path / "checkpoint-209-f16.gguf"
    adapter.write_bytes(b"GGUF ft")
    render = prompt_renderer(FakeTokenizer())
    listed = [{"id": 0, "path": str(adapter), "scale": 1.0}]
    built = build_system("ft", local_client=ListingClient("", listed), render=render)
    assert built.info["adapters"] == [{"id": 0, "path": str(adapter), "sha256": sha256_of(adapter)}]
    assert build_system("b0", local_client=ListingClient("", []), render=render).info["adapters"] == []
    assert build_system("b0", local_client=RecordingClient(""), render=render).info["adapters"] is None
    gone = ListingClient("", [{"id": 0, "path": str(tmp_path / "gone.gguf"), "scale": 1.0}])
    with pytest.raises(SystemUnavailable, match="gone.gguf"):  # a run must say which adapter it measured
        build_system("ft", local_client=gone, render=render)


def test_an_adapter_that_cannot_be_hashed_leaves_no_client_open(monkeypatch, tmp_path):
    opened = []

    class ListedClient:
        def __init__(self, server: str) -> None:
            self.closed = False
            opened.append(self)

        def health(self) -> bool:
            return True

        def adapters(self) -> list[dict]:
            return [{"id": 0, "path": str(tmp_path / "gone.gguf"), "scale": 1.0}]

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr("research_agent.llm.local.LocalClient", ListedClient)
    with pytest.raises(SystemUnavailable, match="gone.gguf"):
        build_system("ft", render=lambda m, t: "")
    assert len(opened) == 1 and opened[0].closed


def fake_llama_server(adapter: Path | None, payloads: list) -> LocalClient:
    """A LocalClient over a fake llama-server with or without an adapter loaded; every completion payload it
    receives goes to payloads, and every completion is the same."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/lora-adapters":
            listed = [] if adapter is None else [{"id": 0, "path": str(adapter), "scale": 1.0}]
            return httpx.Response(200, json=listed)
        payloads.append(json.loads(request.content))
        text = '<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "KV cache"}}\n</tool_call>'
        return httpx.Response(200, json={"choices": [{"text": text, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 900, "completion_tokens": 20},
                                         "timings": {"predicted_ms": 20.0}})

    return LocalClient(transport=httpx.MockTransport(handler))


@pytest.mark.parametrize("system", ["b0", "b1"])
def test_b0_and_b1_over_an_adapter_send_scale_0_and_write_the_bare_servers_records(system, tmp_path):
    # The M3 equivalence check (docs: the M3 plan, phase 3) on the CPU side: the same answers give the same
    # bytes; only the payload's explicit scale 0 and the manifest's adapters differ.
    adapter = tmp_path / "checkpoint-1668-f16.gguf"
    adapter.write_bytes(b"GGUF ft")
    data, requests = training_set(tmp_path)
    render = prompt_renderer(FakeTokenizer())
    runs = {}
    for name, loaded in (("bare", None), ("adapter", adapter)):
        payloads: list[dict] = []
        built = build_system(system, local_client=fake_llama_server(loaded, payloads), render=render,
                             training_set=data, requests=requests)
        write_results(run_items([ITEM, {**ITEM, "id": "i2"}], built.model), tmp_path / f"{name}.jsonl")
        runs[name] = (payloads, built.info)
    assert (tmp_path / "bare.jsonl").read_bytes() == (tmp_path / "adapter.jsonl").read_bytes()
    (bare_payloads, bare_info), (adapter_payloads, adapter_info) = runs["bare"], runs["adapter"]
    assert [p["lora"] for p in adapter_payloads] == [[{"id": 0, "scale": 0.0}]] * 2
    assert [{k: v for k, v in p.items() if k != "lora"} for p in adapter_payloads] == bare_payloads
    assert all("lora" not in p for p in bare_payloads)  # a bare server serves the base model only
    assert bare_info["adapters"] == []
    assert adapter_info["adapters"] == [{"id": 0, "path": str(adapter), "sha256": sha256_of(adapter)}]
    assert {**adapter_info, "adapters": []} == bare_info


# ---- the hf backend (llm.hf): the local systems in full precision, for the M3 HF-vs-quantized comparison ---

TOOL_TEXT = '<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "KV cache"}}\n</tool_call>'


def hf_client(model: FakeCausalLM, adapter: Path | None = None) -> HFClient:
    return HFClient(model, FakeHFTokenizer(), adapter=adapter)


def test_ft_on_the_hf_backend_is_local_model_over_the_hf_client_with_llama_cpps_prompt_and_turn(tmp_path):
    adapter = peft_folder(tmp_path / "ft" / "adapter", weights=b"ft weights")
    render = prompt_renderer(FakeTokenizer())
    client = hf_client(FakePeftModel(TOOL_TEXT, "base says"), adapter)
    built = build_system("ft", backend="hf", adapter=adapter, local_client=client, render=render)
    assert isinstance(built.model, LocalModel) and built.model.client is client
    assert built.info == {"system": "ft", "backend": "hf", "provider": "transformers",
                          "model": "qwen3-1.7b:lora=1", "lora_scale": 1.0, "temperature": 0.0,
                          "tokenizer": None, "dtype": "bfloat16", "device": "cuda:0",
                          "adapter": {"path": adapter.as_posix(),
                                      "safetensors_sha256": sha256_of(adapter / "adapter_model.safetensors")}}
    server = RecordingClient(TOOL_TEXT)
    cpp = build_system("ft", local_client=server, render=render)
    assert (cpp.info["backend"], cpp.info["provider"]) == ("llama.cpp", "llama-server")
    hf_records, cpp_records = run_items([ITEM], built.model), run_items([ITEM], cpp.model)
    assert [r["output"] for r in hf_records] == [r["output"] for r in cpp_records]
    assert hf_records[0]["output"]["tool_calls"][0]["function"]["name"] == "search_arxiv"
    assert hf_records[0]["model"] == cpp_records[0]["model"] == "qwen3-1.7b:lora=1"  # records alike
    assert client.tokenizer.calls[0][0] == server.requests[0]["prompt"]


def test_b0_on_the_hf_backend_runs_the_base_model_with_the_adapter_disabled_or_without_one(tmp_path):
    adapter = peft_folder(tmp_path / "adapter")
    render = prompt_renderer(FakeTokenizer())
    model = FakePeftModel(TOOL_TEXT, "好的。")
    built = build_system("b0", backend="hf", adapter=adapter, local_client=hf_client(model, adapter),
                         render=render)
    bare = build_system("b0", backend="hf", local_client=hf_client(FakeCausalLM("好的。")), render=render)
    runs = [run_items([ITEM], b.model) for b in (built, bare)]
    assert runs[0] == runs[1] and runs[0][0]["output"] == {"role": "assistant", "content": "好的。"}
    assert [c["adapter_enabled"] for c in model.calls] == [False]
    assert built.info["lora_scale"] == 0.0 and bare.info["adapter"] is None
    assert {**built.info, "adapter": None} == bare.info


def test_the_hf_backend_needs_a_peft_folder_for_scale_1_and_runs_the_local_systems_only(tmp_path):
    render = prompt_renderer(FakeTokenizer())
    client = hf_client(FakeCausalLM(""))
    for system in ("ft", "b4"):
        with pytest.raises(SystemUnavailable, match="adapter"):
            build_system(system, backend="hf", local_client=client, render=render)
    gguf = tmp_path / "checkpoint-209-f16.gguf"
    gguf.write_bytes(b"GGUF ft")
    with pytest.raises(SystemUnavailable, match="GGUF"):
        build_system("ft", backend="hf", adapter=gguf, local_client=client, render=render)
    with pytest.raises(SystemUnavailable, match="adapter_model.safetensors"):
        build_system("b0", backend="hf", adapter=tmp_path, local_client=client, render=render)
    for system in ("b2", "b3", "replay"):
        with pytest.raises(ValueError, match="local systems"):
            build_system(system, backend="hf", items=[ITEM])
    with pytest.raises(ValueError, match="hf backend"):  # llama-server loads its own adapter
        build_system("ft", adapter=peft_folder(tmp_path / "adapter"), local_client=RecordingClient(""),
                     render=render)
    with pytest.raises(ValueError, match="backend"):
        build_system("ft", backend="vllm", local_client=RecordingClient(""), render=render)


def test_the_hf_backend_loads_the_model_folder_with_the_adapter_and_a_failed_load_is_unavailable(
        monkeypatch, tmp_path):
    adapter = peft_folder(tmp_path / "adapter")
    loads = []

    def load(model_dir: str, folder: Path) -> HFClient:
        loads.append((model_dir, folder))
        return hf_client(FakePeftModel("a", "b"), folder)

    monkeypatch.setattr("research_agent.llm.hf.load_client", load)
    built = build_system("ft", backend="hf", adapter=adapter, tokenizer="models/Qwen3-1.7B",
                         render=lambda m, t: "")
    assert loads == [("models/Qwen3-1.7B", adapter)] and built.local_client.health()
    built.close()
    assert not built.local_client.health()

    def fail(model_dir: str, folder: Path) -> HFClient:
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr("research_agent.llm.hf.load_client", fail)
    with pytest.raises(SystemUnavailable, match="CUDA out of memory"):
        build_system("ft", backend="hf", adapter=adapter, tokenizer="models/Qwen3-1.7B",
                     render=lambda m, t: "")


class SafetensorError(Exception):
    """safetensors' error class, which derives from Exception alone; what it raises for a corrupt file."""


def test_a_corrupt_adapter_folder_makes_the_hf_backend_unavailable(monkeypatch, tmp_path):
    """PEFT raises safetensors' SafetensorError for an adapter_model.safetensors it cannot read and ValueError
    for an adapter it cannot use: the system is unavailable (run_eval's exit 2), not a crash."""
    adapter = peft_folder(tmp_path / "adapter")
    fake = types.ModuleType("safetensors")
    fake.SafetensorError = SafetensorError
    monkeypatch.setitem(sys.modules, "safetensors", fake)  # safetensors may be absent (CI) or real: the fake
    for error in (SafetensorError("Error while deserializing header: header too large"),
                  ValueError("Target modules {'c_attn'} not found in the base model")):
        def fail(model_dir: str, folder: Path, error: Exception = error) -> HFClient:
            raise error

        monkeypatch.setattr("research_agent.llm.hf.load_client", fail)
        with pytest.raises(SystemUnavailable, match=f"could not load models/Qwen3-1.7B with .*: {error}"):
            build_system("ft", backend="hf", adapter=adapter, tokenizer="models/Qwen3-1.7B",
                         render=lambda m, t: "")


def test_replay_answers_each_item_with_its_expected_turn():
    other = {**ITEM, "id": "i2", "messages": [SYSTEM, {"role": "user", "content": "hello"}],
             "expected": {"content": "Hello!", "tool_calls": []}}
    model = ReplayModel([ITEM, other])
    results = run_items([ITEM, other], model)
    assert results[0]["output"]["tool_calls"][0]["function"] == {"name": "search_arxiv",
                                                                 "arguments": "{\"query\": \"KV cache\"}"}
    assert results[1]["output"] == {"role": "assistant", "content": "Hello!"}
    with pytest.raises(HarnessError, match="no item"):
        model.next_turn([SYSTEM, {"role": "user", "content": "unknown"}], [])


class FakeOpenAI:
    def __init__(self) -> None:
        self.chat = self
        self.completions = self


def test_b2_is_deepseek_flash_booked_under_the_stage_with_its_cap(tmp_path):
    built = build_system("b2", llm_dir=tmp_path, stage="M3", openai_client=FakeOpenAI())
    try:
        assert isinstance(built.model, CloudModel) and built.model.name == "deepseek-flash"
        assert built.info["stage"] == "M3" and built.info["stage_cap_rmb"] == 30.0
        assert built.model.client.budget.stage == "M3"
    finally:
        built.close()
    lowered = build_system("b2", llm_dir=tmp_path, stage="M3", max_cost=2.5, openai_client=FakeOpenAI())
    lowered.close()
    assert lowered.info["stage_cap_rmb"] == 2.5


def test_a_cloud_model_without_a_price_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.setitem(systems.CLOUD_MODELS, "b3", "deepseek-unpriced")
    with pytest.raises(SystemUnavailable, match="no price"):
        build_system("b3", llm_dir=tmp_path, openai_client=FakeOpenAI())


def test_a_cloud_system_without_the_key_is_unavailable(tmp_path, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(SystemUnavailable, match="DEEPSEEK_API_KEY"):
        build_system("b2", llm_dir=tmp_path)


B1_BLOCK = "27998771f1d7f9ba"  # the block of the M2 dev b1 runs (docs/m2-进度.md); M3 compares against them


def test_the_real_demonstrations_block_is_still_the_one_the_m2_b1_runs_used():
    if not (REPO / systems.TRAINING_SET).is_file():
        pytest.skip("the assembled training set is local only (data/raw/ is not tracked)")
    demos = load_demonstrations(REPO / systems.TRAINING_SET, REPO / systems.REQUESTS_FILE)
    assert systems.block_hash(demonstrations_block(demos)) == B1_BLOCK


# ---- M4: ftr, FT routed to B2 under the frozen configuration --------------------------------------------


def routing_files(tmp_path: Path, **overrides) -> dict:
    """A dev set, the FT adapter's GGUF and a routing configuration that names both; overrides change the
    configuration's fields."""
    dev = tmp_path / "dev.jsonl"
    dev.write_text('{"id": "q-1"}\n', encoding="utf-8")
    adapter = tmp_path / "checkpoint-1668-f16.gguf"
    adapter.write_bytes(b"GGUF ft")
    config = {"signal": "min_decision_prob", "threshold": 0.5, "dev_sha256": sha256_of(dev),
              "adapter_sha256": sha256_of(adapter), "cloud_model": "deepseek-flash", "rule_met": True,
              **overrides}
    path = tmp_path / "routing-m4.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {"routing_config": path, "dev_set": dev, "adapter": adapter}


def build_ftr(tmp_path: Path, files: dict, listed: list[dict] | None = None):
    listed = [{"id": 0, "path": str(files["adapter"]), "scale": 1.0}] if listed is None else listed
    return build_system("ftr", local_client=ListingClient("", listed),
                        render=prompt_renderer(FakeTokenizer()), llm_dir=tmp_path / "llm",
                        openai_client=FakeOpenAI(), routing_config=files["routing_config"],
                        dev_set=files["dev_set"])


def test_ftr_is_ft_routed_to_b2_under_the_frozen_configuration(tmp_path):
    from research_agent.agent.route import RoutedModel

    files = routing_files(tmp_path)
    built = build_ftr(tmp_path, files)
    try:
        assert "ftr" in systems.SYSTEMS
        assert isinstance(built.model, RoutedModel)
        assert isinstance(built.model.local, LocalModel) and built.model.local.lora_scale == 1.0
        assert isinstance(built.model.cloud, CloudModel) and built.model.cloud.name == "deepseek-flash"
        assert built.model.config.signal == "min_decision_prob" and built.model.config.threshold == 0.5
        info = built.info
        assert info["system"] == "ftr" and info["cloud_model"] == "deepseek-flash"
        assert info["adapters"] == [{"id": 0, "path": str(files["adapter"]),
                                     "sha256": sha256_of(files["adapter"])}]
        assert info["routing"]["sha256"] == sha256_of(files["routing_config"])
        assert info["routing"]["signal"] == "min_decision_prob" and info["routing"]["threshold"] == 0.5
        assert info["routing"]["path"] == files["routing_config"].as_posix()
        assert built.local_client is built.model.local.client
    finally:
        built.close()


def test_ftr_without_its_configuration_is_unavailable(tmp_path):
    files = routing_files(tmp_path)
    files["routing_config"].unlink()
    with pytest.raises(SystemUnavailable, match="routing configuration"):
        build_ftr(tmp_path, files)


def test_ftr_over_another_dev_set_is_unavailable(tmp_path):
    files = routing_files(tmp_path, dev_sha256="0" * 64)
    with pytest.raises(SystemUnavailable, match="dev set.*SHA-256"):
        build_ftr(tmp_path, files)


def test_ftr_over_another_adapter_is_unavailable(tmp_path):
    files = routing_files(tmp_path, adapter_sha256="1" * 64)
    with pytest.raises(SystemUnavailable, match="adapter.*SHA-256"):
        build_ftr(tmp_path, files)
    with pytest.raises(SystemUnavailable, match="adapter"):  # a server without the adapter
        build_ftr(tmp_path, routing_files(tmp_path), listed=[])


@pytest.mark.parametrize(("overrides", "message"), [
    ({"threshold": None, "rule_met": False}, "threshold"),
    ({"signal": "entropy"}, "signal"),
    ({"cloud_model": "gpt-4o"}, "routes to 'gpt-4o', the model of no cloud system"),
    ({"cloud_model": None}, "routes to None, the model of no cloud system"),
])
def test_ftr_with_a_configuration_it_cannot_run_is_unavailable(tmp_path, overrides, message):
    with pytest.raises(SystemUnavailable, match=message):
        build_ftr(tmp_path, routing_files(tmp_path, **overrides))


def test_ftr_is_not_a_system_of_the_hf_backend(tmp_path):
    with pytest.raises(ValueError, match="local systems only"):
        build_system("ftr", backend="hf")


def test_the_m6_stage_has_the_cap_the_owner_set():
    # docs/m6-计划.md §8: 30 yuan for M6, within the owner's 60 and the plan's 300 for every stage together
    from research_agent.eval.systems import STAGE_CAPS

    assert STAGE_CAPS["M6"] == 30.0
    assert sum(STAGE_CAPS.values()) <= 300.0


# ---- M6: the rules of the frozen configuration (docs/m6-计划.md §3.3) -----------------------------------

RULES_JSON = {"text_exempt": True, "param_triggers": [
    {"tool": "search_notes", "param": "top_k", "allowed": [3]},
    {"tool": "list_notes", "param": "limit", "allowed": [10]}]}


def test_ftr_routes_by_the_configurations_rules_when_it_has_them(tmp_path):
    from research_agent.agent.route import RouteRules

    built = build_ftr(tmp_path, routing_files(tmp_path, rules=RULES_JSON, chosen_policy="P5"))
    try:
        assert built.model.config.rules == RouteRules.from_json(RULES_JSON)
        assert built.info["routing"]["rules"] == RULES_JSON and built.info["routing"]["policy"] == "P5"
    finally:
        built.close()
    built = build_ftr(tmp_path, routing_files(tmp_path))  # a configuration without rules: as before
    try:
        assert built.model.config.rules is None
        assert built.info["routing"]["rules"] is None and built.info["routing"]["policy"] is None
    finally:
        built.close()


def test_load_routing_round_trips_the_rules_and_refuses_malformed_ones(tmp_path):
    files = routing_files(tmp_path, rules=RULES_JSON)
    route, info = systems.load_routing(files["routing_config"], files["dev_set"])
    assert route.rules.to_json() == RULES_JSON and info["rules"] == RULES_JSON
    assert route.rules.text_exempt
    assert [t.tool for t in route.rules.param_triggers] == ["search_notes", "list_notes"]
    files = routing_files(tmp_path, rules={"text_exempt": "yes"})
    with pytest.raises(SystemUnavailable, match="rules"):
        systems.load_routing(files["routing_config"], files["dev_set"])
    files = routing_files(tmp_path, rules=None)
    assert systems.load_routing(files["routing_config"], files["dev_set"])[0].rules is None


def test_the_frozen_m4_configuration_still_loads_as_one_without_rules():
    if not (REPO / systems.DEV_SET).is_file() or not (REPO / systems.ROUTING_CONFIG).is_file():
        pytest.skip("the frozen configuration and the development set are not here")
    route, info = systems.load_routing(REPO / systems.ROUTING_CONFIG, REPO / systems.DEV_SET)
    assert route.rules is None and route.threshold == 0.689404839052033
    assert info["rules"] is None and info["policy"] is None


# ---- M7: a configuration chosen on several development sets (docs/m7-计划.md §5 开发集口径, WP-K) ---------


def joint_routing_files(tmp_path: Path, **overrides) -> dict:
    """routing_files' dev set and adapter with a second development set beside them, and a configuration
    chosen on both (scripts/route_curve.py's "dev_sets", in place of "dev_sha256"), under the M7 policy P5
    with the budget trigger."""
    files = routing_files(tmp_path)
    second = tmp_path / "data" / "eval" / "v2" / "dev-v2.jsonl"
    second.parent.mkdir(parents=True, exist_ok=True)
    second.write_text('{"id": "q-2"}\n{"id": "q-3"}\n', encoding="utf-8")
    config = json.loads(files["routing_config"].read_text(encoding="utf-8"))
    del config["dev_sha256"]
    config.update({"dev_sets": [{"path": "dev.jsonl", "sha256": sha256_of(files["dev_set"]), "n": 1,
                                 "inputs": {}},
                                {"path": "data/eval/v2/dev-v2.jsonl", "sha256": sha256_of(second), "n": 2,
                                 "inputs": {}}],
                   "rules": {**RULES_JSON, "budget_forced": True}, "chosen_policy": "P5", **overrides})
    path = tmp_path / "routing-m7.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return {**files, "routing_config": path, "second": second}


def test_the_v2_development_set_is_named_beside_the_first():
    assert systems.DEV_SET == Path("data/eval/dev.jsonl")
    assert systems.DEV_SET_V2 == Path("data/eval/v2/dev-v2.jsonl")


def test_load_routing_checks_every_development_set_a_configuration_of_several_names_in_its_order(tmp_path):
    files = joint_routing_files(tmp_path)
    sets = (files["dev_set"], files["second"])
    route, info = systems.load_routing(files["routing_config"], sets)
    assert route.rules.budget_forced and info["rules"] == {**RULES_JSON, "budget_forced": True}
    assert "dev_sha256" not in info
    assert info["dev_sets"] == [{"path": files["dev_set"].as_posix(), "sha256": sha256_of(files["dev_set"])},
                                {"path": files["second"].as_posix(), "sha256": sha256_of(files["second"])}]
    with pytest.raises(SystemUnavailable, match="chosen on 2 development sets, not on the 1 given"):
        systems.load_routing(files["routing_config"], files["dev_set"])
    with pytest.raises(SystemUnavailable, match=r"development set 1 \(dev.jsonl\) has SHA-256"):
        systems.load_routing(files["routing_config"], tuple(reversed(sets)))  # the order is the file's
    files["second"].write_text('{"id": "q-2"}\n', encoding="utf-8")  # changed since the routing was chosen
    with pytest.raises(SystemUnavailable, match=r"set 2 \(data/eval/v2/dev-v2.jsonl\) has SHA-256"):
        systems.load_routing(files["routing_config"], sets)
    files["second"].unlink()
    with pytest.raises(SystemUnavailable, match="dev-v2.jsonl is not here"):
        systems.load_routing(files["routing_config"], sets)


def test_a_configuration_of_one_development_set_loads_as_before_whether_the_set_is_given_alone_or_in_a_list(
        tmp_path):
    files = routing_files(tmp_path)
    alone, listed = (systems.load_routing(files["routing_config"], given)[1]
                     for given in (files["dev_set"], [files["dev_set"]]))
    assert alone == listed and alone["dev_sha256"] == sha256_of(files["dev_set"]) and "dev_sets" not in alone
    second = tmp_path / "dev-v2.jsonl"
    second.write_text('{"id": "q-2"}\n', encoding="utf-8")
    with pytest.raises(SystemUnavailable, match="chosen on 1 development set, not on the 2 given"):
        systems.load_routing(files["routing_config"], (files["dev_set"], second))


def test_without_the_sets_given_the_configuration_is_checked_against_the_sets_it_names(tmp_path,
                                                                                      monkeypatch):
    files = joint_routing_files(tmp_path)
    monkeypatch.chdir(tmp_path)  # the named paths are relative to the repository, where the scripts run
    _, info = systems.load_routing(files["routing_config"], None)
    assert [entry["path"] for entry in info["dev_sets"]] == ["dev.jsonl", "data/eval/v2/dev-v2.jsonl"]
    files["second"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemUnavailable, match="development set 2"):
        systems.load_routing(files["routing_config"], None)
    # a configuration of one set names it in dev_set; without that name, it is data/eval/dev.jsonl
    (tmp_path / "one").mkdir()
    one = routing_files(tmp_path / "one")
    config = json.loads(one["routing_config"].read_text(encoding="utf-8"))
    one["routing_config"].write_text(json.dumps({**config, "dev_set": "one/dev.jsonl"}), encoding="utf-8")
    assert systems.load_routing(one["routing_config"], None)[1]["dev_sha256"] == sha256_of(one["dev_set"])
    one["routing_config"].write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(SystemUnavailable, match="the development set data/eval/dev.jsonl is not here"):
        systems.load_routing(one["routing_config"], None)


def test_ftr_routes_by_a_configuration_chosen_on_several_development_sets(tmp_path):
    files = joint_routing_files(tmp_path)
    listed = [{"id": 0, "path": str(files["adapter"]), "scale": 1.0}]
    built = build_system("ftr", local_client=ListingClient("", listed),
                         render=prompt_renderer(FakeTokenizer()), llm_dir=tmp_path / "llm",
                         openai_client=FakeOpenAI(), routing_config=files["routing_config"],
                         dev_set=(files["dev_set"], files["second"]))
    try:
        assert built.model.config.rules.budget_forced and built.info["routing"]["policy"] == "P5"
        assert [entry["sha256"] for entry in built.info["routing"]["dev_sets"]] == [
            sha256_of(files["dev_set"]), sha256_of(files["second"])]
        assert (built.info["cloud"]["system"], built.model.cloud.name) == ("b2", "deepseek-flash")
    finally:
        built.close()
    with pytest.raises(SystemUnavailable, match="chosen on 2 development sets"):
        build_ftr(tmp_path, files)  # data/eval/dev.jsonl alone, ftr's default: not what it was chosen on


def test_the_frozen_configurations_load_against_the_sets_they_name_as_against_the_set_given(monkeypatch):
    for name in ("routing-m4.json", "routing-m6.json"):
        path = REPO / "results" / name
        if not (REPO / systems.DEV_SET).is_file() or not path.is_file():
            pytest.skip("the frozen configurations and the development set are not here")
        monkeypatch.chdir(REPO)
        assert systems.load_routing(path, None) == systems.load_routing(path, REPO / systems.DEV_SET)


# ---- M7: the stage, and the cloud model's thinking switch (docs/m7-计划.md WP1) --------------------------

def test_m7_has_its_stage_cap_and_a_cloud_system_can_run_without_thinking(tmp_path):
    from research_agent.eval.systems import STAGE_CAPS

    assert STAGE_CAPS["M7"] == 50.0 and sum(STAGE_CAPS.values()) <= 300.0
    built = build_system("b3", llm_dir=tmp_path, stage="M7", openai_client=FakeOpenAI(), cloud_thinking="off")
    try:
        assert built.info["model"] == "deepseek-v4-pro" and built.info["thinking"] == "off"
        assert built.model.client.thinking == "off"
        # the ledger books what DeepSeek charges: half the price off peak
        assert built.model.client.budget.off_peak is not None
    finally:
        built.close()
    default = build_system("b2", llm_dir=tmp_path, openai_client=FakeOpenAI())
    default.close()
    assert default.info["thinking"] == "auto" and default.model.client.thinking == "auto"
    with pytest.raises(ValueError, match="cloud_thinking"):
        build_system("b2", llm_dir=tmp_path, openai_client=FakeOpenAI(), cloud_thinking="never")


# ---- M8: the stage (docs/m8-计划.md: the owner's cloud cap of 10 yuan) -----------------------------------

def test_m8_has_its_stage_cap_after_the_earlier_milestones(tmp_path):
    from research_agent.eval.systems import STAGE_CAPS

    assert STAGE_CAPS["M8"] == 10.0 and sum(STAGE_CAPS.values()) <= 300.0
    # in milestone order: scripts/run_test.py tells an earlier milestone by its place in the mapping
    assert list(STAGE_CAPS) == ["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8"]
    built = build_system("b2", llm_dir=tmp_path, stage="M8", openai_client=FakeOpenAI())
    try:
        assert (built.info["stage"], built.info["stage_cap_rmb"]) == ("M8", 10.0)
        assert built.model.client.budget.stage_cap == 10.0
    finally:
        built.close()
    with pytest.raises(ValueError, match="unknown stage 'M9'"):
        build_system("b2", llm_dir=tmp_path, stage="M9", openai_client=FakeOpenAI())


# ---- M7: local records named after the tokenizer folder, ftr routed to b3 (docs/m7-计划.md WP2, WP3) ------


def test_the_local_name_is_the_folder_name_whichever_separators_the_path_uses(monkeypatch):
    # a configuration written on Windows may use backslashes; the stored name must not depend on the platform,
    # so the module's Path is made a POSIX one, as on Linux
    monkeypatch.setattr(systems, "Path", PurePosixPath)
    for spelling in ("models/Qwen3-4B", "models/Qwen3-4B/", "C:/m/Qwen3-4B", "models\\Qwen3-4B",
                     "C:\\m\\Qwen3-4B", "models\\Qwen3-4B\\"):
        assert systems.local_name(spelling) == "qwen3-4b", spelling
    assert systems.local_name(PurePosixPath("models") / "Qwen3-1.7B") == "qwen3-1.7b"
    assert systems.local_name(None) == systems.LOCAL_NAME


def test_a_local_systems_records_are_named_after_its_tokenizer_folder(tmp_path):
    render = prompt_renderer(FakeTokenizer())
    four = build_system("ft", tokenizer="models/Qwen3-4B", local_client=RecordingClient(TOOL_TEXT),
                        render=render)
    assert four.model.name == four.info["model"] == "qwen3-4b:lora=1"
    assert four.info["tokenizer"] == "models/Qwen3-4B"
    assert run_items([ITEM], four.model)[0]["model"] == "qwen3-4b:lora=1"
    base = build_system("b0", tokenizer=Path("models") / "Qwen3-4B", local_client=RecordingClient(""),
                        render=render)
    assert base.info["model"] == "qwen3-4b:lora=0"
    data, requests = training_set(tmp_path)
    b1 = build_system("b1", tokenizer="models/Qwen3-4B", local_client=RecordingClient(""), render=render,
                      training_set=data, requests=requests)
    assert b1.info["model"] == "qwen3-4b:lora=0+demos"
    adapter = peft_folder(tmp_path / "adapter")
    hf = build_system("ft", backend="hf", adapter=adapter, tokenizer="models/Qwen3-4B",
                      local_client=hf_client(FakePeftModel(TOOL_TEXT, ""), adapter), render=render)
    assert hf.info["model"] == "qwen3-4b:lora=1"
    # every run before M7: the 1.7B folder, or no folder at all where a test injects the render
    same = build_system("ft", tokenizer="models/Qwen3-1.7B", local_client=RecordingClient(""), render=render)
    bare = build_system("ft", local_client=RecordingClient(""), render=render)
    assert same.info["model"] == bare.info["model"] == "qwen3-1.7b:lora=1"


def test_ftr_over_a_4b_tokenizer_names_its_local_model_after_it(tmp_path):
    files = routing_files(tmp_path)
    listed = [{"id": 0, "path": str(files["adapter"]), "scale": 1.0}]
    built = build_system("ftr", tokenizer="models/Qwen3-4B", local_client=ListingClient("", listed),
                         render=prompt_renderer(FakeTokenizer()), llm_dir=tmp_path / "llm",
                         openai_client=FakeOpenAI(), routing_config=files["routing_config"],
                         dev_set=files["dev_set"])
    try:
        assert built.model.name == "qwen3-4b:lora=1>deepseek-flash"
        assert built.info["local"]["model"] == "qwen3-4b:lora=1"
        assert built.info["tokenizer"] == "models/Qwen3-4B"
    finally:
        built.close()


class AnsweringOpenAI:
    """DeepSeek's SDK, faked: every request recorded, every answer the same text from the model asked."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self.create))

    def create(self, **payload):
        self.requests.append(payload)
        data = {"id": "chatcmpl-1", "model": payload["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "Hello."}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 50}}
        return types.SimpleNamespace(model_dump=lambda **kwargs: data)


def test_ftr_routes_to_b3_when_its_configuration_names_deepseek_v4_pro(tmp_path):
    files = routing_files(tmp_path, cloud_model="deepseek-v4-pro")
    cloud = AnsweringOpenAI()
    listed = [{"id": 0, "path": str(files["adapter"]), "scale": 1.0}]
    built = build_system("ftr", local_client=ListingClient("", listed),
                         render=prompt_renderer(FakeTokenizer()), llm_dir=tmp_path / "llm", stage="M7",
                         openai_client=cloud, routing_config=files["routing_config"],
                         dev_set=files["dev_set"], cloud_thinking="off")
    try:
        assert isinstance(built.model.cloud, CloudModel) and built.model.cloud.name == "deepseek-v4-pro"
        assert built.model.name == "qwen3-1.7b:lora=1>deepseek-v4-pro"
        assert built.info["cloud_model"] == built.info["routing"]["cloud_model"] == "deepseek-v4-pro"
        assert (built.info["cloud"]["system"], built.info["cloud"]["model"]) == ("b3", "deepseek-v4-pro")
        assert built.info["cloud"]["thinking"] == "off" and built.info["cloud"]["stage"] == "M7"
        # the local answer has no tokens, so its signal is unknown and the step goes to b3
        (record,) = run_items([ITEM], built.model)
        assert record["route"]["answered_by"] == "cloud" and record["model"] == "deepseek-v4-pro"
        assert [r["model"] for r in cloud.requests] == ["deepseek-v4-pro"]
        assert record["paid"] and record["usage"] == {"prompt_tokens": 1000, "completion_tokens": 50}
        booked = built.model.cloud.client.budget.db.execute("SELECT stage, model FROM spend").fetchall()
        assert booked == [("M7", "deepseek-v4-pro")]  # booked under the stage at v4-pro's price
    finally:
        built.close()
    route, info = systems.load_routing(files["routing_config"], files["dev_set"])
    assert info["cloud_model"] == "deepseek-v4-pro" and route.threshold == 0.5
    flash = build_ftr(tmp_path, routing_files(tmp_path))  # M4 to M6: deepseek-flash, so b2, as before
    try:
        assert (flash.info["cloud"]["system"], flash.model.cloud.name) == ("b2", "deepseek-flash")
    finally:
        flash.close()


# ---- M7: the family path of the local systems (docs/m7-家族层设计.md §7) ----------------------------------

SAMPLING = Decoding(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
GREEDY_RECORD = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "presence_penalty": 0.0}


def test_a_local_system_over_a_model_folder_runs_the_family_path_and_writes_todays_records(tmp_path,
                                                                                          monkeypatch):
    tokenizer = install_fake_family(monkeypatch, tmp_path / "model")
    folder = str(tmp_path / "model")
    family_server, legacy_server = FakeLlama([scripted(TOOL_TEXT)]), FakeLlama([scripted(TOOL_TEXT)])
    built = build_system("b0", tokenizer=folder, local_client=family_server.client())
    assert isinstance(built.model, LocalModel) and built.model.family is not None
    assert built.info["family"] == built.model.family.record()
    assert built.info["family"]["name"] == "fake-hybrid"
    assert built.info["generation"] == {"mode": "no_think", "decoding": GREEDY_RECORD,
                                        "reasoning_budget": 1024, "answer_tokens": 512, "base_seed": 0,
                                        "sample": 0}
    assert built.info["temperature"] == 0.0 and built.info["model"] == "model:lora=0"
    assert built.info["tokenizer"] == folder and built.info["adapters"] == []
    records = run_items([ITEM], built.model)
    # the path before M7 over the same template and the same answer writes the same record: the request
    # differs by the prompt's form alone, the text's ids in place of the text
    legacy = build_system("b0", tokenizer=folder, local_client=legacy_server.client(),
                          render=prompt_renderer(tokenizer))
    assert run_items([ITEM], legacy.model) == records and "family" not in legacy.info
    (family_request,), (legacy_request,) = family_server.requests, legacy_server.requests
    assert family_request["prompt"] == token_ids(legacy_request["prompt"])
    assert {k: v for k, v in family_request.items() if k != "prompt"} == {
        k: v for k, v in legacy_request.items() if k != "prompt"}
    assert records[0]["output"]["tool_calls"][0]["function"]["name"] == "search_arxiv"
    assert "reasoning" not in records[0] and "thinking" not in records[0]


def test_a_local_system_takes_the_generation_given_and_its_default_is_greedy_no_think(tmp_path, monkeypatch):
    install_fake_family(monkeypatch, tmp_path / "model", inference_mode=Mode.THINK,
                        decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY})
    folder = str(tmp_path / "model")
    settings = GenerationSettings(mode=Mode.THINK, decoding=SAMPLING, reasoning_budget=32, base_seed=1)
    fake = FakeLlama([Scripted(("Need a search.", "\n</think>\n\n", TOOL_TEXT))])
    built = build_system("b0", tokenizer=folder, local_client=fake.client(), generation=settings)
    assert built.model.settings == settings and built.info["generation"] == settings.record()
    assert built.info["temperature"] == 0.6
    (record,) = run_items([ITEM], built.model)
    assert record["reasoning"] == "Need a search." and record["thinking"]["mode"] == "think"
    assert fake.requests[0]["max_tokens"] == 32 and "seed" in fake.requests[0]
    quiet = build_system("b0", tokenizer=folder, local_client=FakeLlama([]).client())  # not the family's mode
    assert quiet.model.settings == GenerationSettings(mode=Mode.NO_THINK, decoding=GREEDY)


def test_the_family_path_needs_a_registered_folder_and_a_mode_the_family_has(tmp_path, monkeypatch):
    install_fake_family(monkeypatch, tmp_path / "model", "none")
    with pytest.raises(SystemUnavailable, match="has no think mode"):
        build_system("b0", tokenizer=str(tmp_path / "model"), local_client=FakeLlama([]).client(),
                     generation=GenerationSettings(mode=Mode.THINK, decoding=GREEDY))
    unknown = model_folder(tmp_path / "unknown", template="another template")
    with pytest.raises(SystemUnavailable, match="unknown chat template"):
        build_system("b0", tokenizer=str(unknown), local_client=FakeLlama([]).client())
    with pytest.raises(SystemUnavailable, match="tokenizer"):
        build_system("b0", local_client=FakeLlama([]).client())


def test_thinking_or_sampling_is_not_for_the_hf_backend_or_an_injected_render(tmp_path):
    render = prompt_renderer(FakeTokenizer())
    thinking = GenerationSettings(mode=Mode.THINK, decoding=GREEDY)
    with pytest.raises(ValueError, match="hf backend"):
        build_system("b0", backend="hf", local_client=hf_client(FakeCausalLM("")), render=render,
                     generation=thinking)
    with pytest.raises(ValueError, match="render"):
        build_system("b0", local_client=RecordingClient(""), render=render,
                     generation=GenerationSettings(mode=Mode.NO_THINK, decoding=SAMPLING))
    # greedy no_think is what they do anyway
    built = build_system("b0", local_client=RecordingClient(""), render=render,
                         generation=GenerationSettings(mode=Mode.NO_THINK, decoding=GREEDY))
    assert built.model.family is None and "generation" not in built.info


def test_the_demonstration_reply_is_the_tool_formats_render_and_todays_text():
    assert demonstration_reply(SEARCH) == ('<tool_call>\n{"name": "search_arxiv", "arguments": {"query": '
                                           '"KV cache", "date_from": "2026-07-04"}}\n</tool_call>')
    with_text = {"role": "assistant", "content": "Searching.", "tool_calls": SEARCH["tool_calls"]}
    assert demonstration_reply(with_text) == "Searching.\n" + demonstration_reply(SEARCH)
    assert demonstration_reply(ANSWER) == ANSWER["content"]

    def names(calls: list[dict]) -> str:
        return " | ".join(f"{c['name']}{sorted(c['arguments'])}" for c in calls)

    assert demonstration_reply(SEARCH, names) == "search_arxiv['date_from', 'query']"
    assert demonstration_reply(ANSWER, names) == ANSWER["content"]


def test_b1_renders_its_demonstrations_with_the_familys_tool_format(tmp_path, monkeypatch):
    data, requests = training_set(tmp_path)
    install_fake_family(monkeypatch, tmp_path / "model")
    fake = FakeLlama([scripted("好的。")])
    built = build_system("b1", tokenizer=str(tmp_path / "model"), local_client=fake.client(),
                         training_set=data, requests=requests)
    block = demonstrations_block(load_demonstrations(data, requests))  # today's rendering: the same block
    assert built.info["demonstrations"]["block_sha256_16"] == systems.block_hash(block)
    assert built.info["family"]["name"] == "fake-hybrid" and built.info["model"] == "model:lora=0+demos"
    run_items([ITEM], built.model)
    prompt = built.model.model.family.decode(fake.requests[0]["prompt"])
    assert SYSTEM["content"] + block in prompt


def test_b1_renders_its_demonstrations_with_a_familys_own_format_when_it_is_not_qwen3s(tmp_path, monkeypatch):
    data, requests = training_set(tmp_path)
    fmt = FakeToolFormat()
    install_fake_family(monkeypatch, tmp_path / "model", tool_format=fmt)
    fake = FakeLlama([scripted("好的。")])
    built = build_system("b1", tokenizer=str(tmp_path / "model"), local_client=fake.client(),
                         training_set=data, requests=requests)
    demos = load_demonstrations(data, requests)
    block = demonstrations_block(demos, fmt.render)
    assert '[call search_arxiv {"date_from":"2026-07-04","query":"KV cache"}]' in block
    assert "<tool_call>" not in block
    assert built.info["demonstrations"]["block_sha256_16"] == systems.block_hash(block)
    assert systems.block_hash(block) != systems.block_hash(demonstrations_block(demos))  # not Qwen3's block
    run_items([ITEM], built.model)
    prompt = built.model.model.family.decode(fake.requests[0]["prompt"])
    assert SYSTEM["content"] + block in prompt


def test_ftr_runs_its_local_side_on_the_family_path_with_the_generation_given(tmp_path, monkeypatch):
    files = routing_files(tmp_path)
    install_fake_family(monkeypatch, tmp_path / "model")
    listed = [{"id": 0, "path": str(files["adapter"]), "scale": 1.0}]
    settings = GenerationSettings(mode=Mode.THINK, decoding=SAMPLING, reasoning_budget=32, base_seed=1)
    fake = FakeLlama([Scripted(("Think.", "\n</think>\n\n", TOOL_TEXT))], adapters=listed)
    built = build_system("ftr", tokenizer=str(tmp_path / "model"), local_client=fake.client(),
                         llm_dir=tmp_path / "llm", openai_client=FakeOpenAI(),
                         routing_config=files["routing_config"], dev_set=files["dev_set"],
                         generation=settings)
    try:
        assert built.model.local.settings == settings and built.info["generation"] == settings.record()
        assert built.info["family"]["name"] == "fake-hybrid" and built.info["temperature"] == 0.6
        assert built.info["local"]["generation"] == settings.record()
        (record,) = run_items([ITEM], built.model)
        assert record["route"]["answered_by"] == "local" and record["reasoning"] == "Think."
    finally:
        built.close()



# ---- M7 WP2, WP3 and WP4: the inference wrappers (docs/m7-本地提升计划.md) ---------------------------------


def wrapped(tmp_path: Path, monkeypatch, **options):
    """b0 over a fake family whose think mode samples, built with the wrapper options given; its glossary
    file, if asked for (glossary=True), holds one term. Returns (built, glossary path)."""
    install_fake_family(monkeypatch, tmp_path / "model", inference_mode=Mode.THINK,
                        decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY})
    path = tmp_path / "zh-en.json"
    path.write_text(json.dumps({"混合专家": "mixture of experts"}, ensure_ascii=False), encoding="utf-8")
    if options.pop("glossary", False):
        options["glossary"] = path
    built = build_system("b0", tokenizer=str(tmp_path / "model"), local_client=FakeLlama([]).client(),
                         **options)
    return built, path


def test_a_voted_system_votes_over_ruled_noted_candidates_and_its_manifest_names_every_option(tmp_path,
                                                                                               monkeypatch):
    from research_agent.agent.noted import NotedModel
    from research_agent.agent.rules import RULES, RULES_VERSION, RuledModel
    from research_agent.agent.vote import FINAL, VotedModel, voted_settings

    settings = GenerationSettings(mode=Mode.THINK, decoding=SAMPLING, reasoning_budget=32, votes=3)
    counted = []
    built, glossary = wrapped(tmp_path, monkeypatch, generation=settings, rules=True, glossary=True,
                              count_tokens=lambda messages, tools: counted.append(messages) or 7)
    model = built.model
    assert isinstance(model, VotedModel) and model.mode == FINAL and len(model.samples) == 3
    greedy, samples = voted_settings(settings)
    for candidate, wanted in zip((model.greedy, *model.samples), (greedy, *samples), strict=True):
        assert isinstance(candidate, RuledModel) and isinstance(candidate.model, NotedModel)
        assert isinstance(candidate.model.model, LocalModel) and candidate.model.model.settings == wanted
        assert candidate.model.glossary == {"混合专家": "mixture of experts"}
    info = built.info
    assert info["generation"] == settings.record() and info["generation"]["votes"] == 3
    assert info["vote_mode"] == FINAL
    assert info["rules"] == {"version": RULES_VERSION, "names": list(RULES)}
    assert info["glossary"] == {"path": glossary.as_posix(), "sha256": sha256_of(glossary), "terms": 1}
    assert info["model"] == model.name == "model:lora=0+glossary+rules+votes3"
    assert built.count_tokens([SYSTEM, {"role": "user", "content": "混合专家"}], []) == 7
    assert counted[-1][0]["content"].endswith("术语对照:混合专家 = mixture of experts")
    assert built.timings is not None and built.timings() is None  # no step yet


def test_each_option_alone_wraps_the_local_model_and_records_itself_only(tmp_path, monkeypatch):
    from research_agent.agent.noted import NotedModel
    from research_agent.agent.rules import RuledModel
    from research_agent.agent.vote import REPRODUCTION, VotedModel

    built, _ = wrapped(tmp_path, monkeypatch, rules=True)
    assert isinstance(built.model, RuledModel) and isinstance(built.model.model, LocalModel)
    assert "rules" in built.info and not {"glossary", "vote_mode"} & set(built.info)
    assert "votes" not in built.info["generation"] and built.timings is None
    built, _ = wrapped(tmp_path, monkeypatch, glossary=True)
    assert isinstance(built.model, NotedModel) and not {"rules", "vote_mode"} & set(built.info)
    voting = GenerationSettings(mode=Mode.THINK, decoding=SAMPLING, votes=2)
    built, _ = wrapped(tmp_path, monkeypatch, generation=voting, vote_mode=REPRODUCTION)
    assert isinstance(built.model, VotedModel) and built.model.mode == REPRODUCTION
    assert all(isinstance(c, LocalModel) for c in (built.model.greedy, *built.model.samples))
    assert built.info["vote_mode"] == REPRODUCTION and not {"rules", "glossary"} & set(built.info)
    plain, _ = wrapped(tmp_path, monkeypatch)
    assert isinstance(plain.model, LocalModel)
    assert not {"rules", "glossary", "vote_mode"} & set(plain.info) and plain.timings is None


def test_the_rules_and_the_glossary_wrap_the_model_before_m7_too_but_a_vote_needs_the_family_path(tmp_path):
    from research_agent.agent.noted import NotedModel
    from research_agent.agent.rules import RuledModel

    glossary = tmp_path / "zh-en.json"
    glossary.write_text(json.dumps({"混合专家": "mixture of experts"}, ensure_ascii=False), encoding="utf-8")
    render = prompt_renderer(FakeTokenizer())
    adapter = tmp_path / "ft.gguf"
    adapter.write_bytes(b"GGUF ft")
    built = build_system("ft", local_client=ListingClient(TOOL_TEXT, [{"id": 0, "path": str(adapter),
                                                                       "scale": 1.0}]),
                         render=render, rules=True, glossary=glossary)
    assert isinstance(built.model, RuledModel) and isinstance(built.model.model, NotedModel)
    (record,) = run_items([{**ITEM, "messages": [SYSTEM, {"role": "user", "content": "找混合专家的论文"}]}],
                          built.model)
    assert record["rules"] == [] and record["status"] == "ok"
    with pytest.raises(ValueError, match="render"):
        build_system("b0", local_client=RecordingClient(""), render=render,
                     generation=GenerationSettings(mode=Mode.NO_THINK, decoding=SAMPLING, votes=2))


def test_a_glossary_that_cannot_be_read_makes_the_system_unavailable(tmp_path, monkeypatch):
    install_fake_family(monkeypatch, tmp_path / "model")
    bad = tmp_path / "bad.json"
    bad.write_text('{"no han": "x"}', encoding="utf-8")
    for path in (bad, tmp_path / "missing.json"):
        with pytest.raises(SystemUnavailable, match="glossary"):
            build_system("b0", tokenizer=str(tmp_path / "model"), local_client=FakeLlama([]).client(),
                         glossary=path)


def test_ftrs_local_side_is_wrapped_inside_the_routing(tmp_path, monkeypatch):
    from research_agent.agent.route import RoutedModel
    from research_agent.agent.rules import RuledModel

    files = routing_files(tmp_path)
    install_fake_family(monkeypatch, tmp_path / "model")
    listed = [{"id": 0, "path": str(files["adapter"]), "scale": 1.0}]
    built = build_system("ftr", tokenizer=str(tmp_path / "model"),
                         local_client=FakeLlama([], adapters=listed).client(), llm_dir=tmp_path / "llm",
                         openai_client=FakeOpenAI(), routing_config=files["routing_config"],
                         dev_set=files["dev_set"], rules=True)
    try:
        assert isinstance(built.model, RoutedModel) and isinstance(built.model.local, RuledModel)
        assert built.info["rules"] == built.info["local"]["rules"]
    finally:
        built.close()

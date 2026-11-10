"""research_agent.llm.hf offline: the full-precision client over fake model and tokenizer objects (decoding,
stop handling, the adapter at scale 1 and disabled at scale 0, the refusals), the adapter folder check, how
load_client loads the model (over fake torch, transformers and peft modules), and that LocalModel reads its
completions as it reads llama-server's. Nothing loads from models/ or imports torch."""

from __future__ import annotations

import ast
import importlib.metadata
import inspect
from pathlib import Path

import pytest
from support.hf_fakes import (
    BFLOAT16,
    END_IDS,
    FAKE_GPU,
    FakeCausalLM,
    FakeHFTokenizer,
    FakePeftModel,
    fake_stack,
    peft_folder,
    token_ids,
)
from train.tokenizer_fake import FakeTokenizer

from research_agent.agent.models import LocalModel
from research_agent.llm import hf
from research_agent.llm.hf import AdapterError, HFClient, check_adapter, load_client
from research_agent.llm.local import Completion, LocalClient
from research_agent.tools.schemas import MODEL_TOOLS
from research_agent.train.manifest import package_version
from research_agent.train.samples import prompt_renderer

TOOL_TEXT = '<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "hallucination"}}\n</tool_call>'


def test_a_completion_is_greedy_with_llama_servers_token_limit_and_ends_at_the_models_end_tokens():
    tokenizer, model = FakeHFTokenizer(), FakeCausalLM("Hello there.")
    out = HFClient(model, tokenizer).complete("<|im_start|>user\nhi<|im_end|>\n", stop=["<|im_end|>"])
    assert (out.text, out.finish_reason) == ("Hello there.", "stop")
    (call,) = model.calls
    limit = inspect.signature(LocalClient.complete).parameters["max_tokens"].default
    assert limit == 512 and call["max_new_tokens"] == limit
    assert (call["do_sample"], call["num_beams"]) == (False, 1)
    # the model's sampling defaults (Qwen3: temperature 0.6, top_p 0.95, top_k 20) are unset, not inherited
    assert (call["temperature"], call["top_p"], call["top_k"]) == (None, None, None)
    assert call["eos_token_id"] == END_IDS
    assert tokenizer.calls == [("<|im_start|>user\nhi<|im_end|>\n", {"add_special_tokens": False,
                                                                     "return_tensors": "pt"})]
    encoded = tokenizer.encodings[0]
    assert encoded.device == "cuda:0" and call["input_ids"] == encoded["input_ids"]
    prompt = len(token_ids("<|im_start|>user\nhi<|im_end|>\n"))
    new = len("Hello there.") + 1  # the end token is generated and counted, as llama-server counts it
    assert out.raw == {"usage": {"prompt_tokens": prompt, "completion_tokens": new,
                                 "total_tokens": prompt + new}}
    assert out.tokens == [] and out.logprobs == [] and out.timings is None and out.wall_ms >= 0


def test_special_tokens_are_left_out_of_the_text_as_llama_server_leaves_them_and_tool_call_tags_stay():
    tokenizer = FakeHFTokenizer()
    out = HFClient(FakeCausalLM(TOOL_TEXT + "<|im_start|>"), tokenizer).complete("p")
    assert out.text == TOOL_TEXT
    assert tokenizer.decoded[0][1] == {"skip_special_tokens": True, "clean_up_tokenization_spaces": False}
    ended = HFClient(FakeCausalLM("Done.", end="<|endoftext|>"), FakeHFTokenizer()).complete("p")
    assert (ended.text, ended.finish_reason) == ("Done.", "stop")  # the other end token of the config


def test_the_text_is_cut_before_the_first_stop_string_and_the_token_limit_gives_length():
    client = HFClient(FakeCausalLM("abc</tool_call>def</tool_call>"), FakeHFTokenizer())
    out = client.complete("p", stop=["</tool_call>", "zzz"])
    assert (out.text, out.finish_reason) == ("abc", "stop")
    endless = HFClient(FakeCausalLM("abcdef", end=""), FakeHFTokenizer())
    out = endless.complete("p", max_tokens=3)
    assert (out.text, out.finish_reason) == ("abc", "length")
    assert out.raw["usage"]["completion_tokens"] == 3


def test_scale_one_applies_the_adapter_and_scale_zero_runs_the_base_model_inside_disable_adapter(tmp_path):
    model = FakePeftModel("adapter says", "base says")
    client = HFClient(model, FakeHFTokenizer(), adapter=peft_folder(tmp_path / "adapter"))
    assert client.complete("p", lora_scale=1.0).text == "adapter says"
    assert client.complete("p", lora_scale=0.0).text == "base says"
    assert client.complete("p", lora_scale=1.0).text == "adapter says"
    assert [c["adapter_enabled"] for c in model.calls] == [True, False, True] and model.enabled


def test_the_scales_llama_server_refuses_are_refused_before_anything_is_generated(tmp_path):
    with_adapter = FakePeftModel("adapter says", "base says")
    client = HFClient(with_adapter, FakeHFTokenizer(), adapter=peft_folder(tmp_path / "adapter"))
    with pytest.raises(ValueError, match="lora_scale"):
        client.complete("p")  # a base-model request must say so, as LocalClient requires
    with pytest.raises(ValueError, match="whole"):
        client.complete("p", lora_scale=0.5)
    with pytest.raises(ValueError, match="greedily"):
        client.complete("p", lora_scale=1.0, temperature=0.7)
    assert with_adapter.calls == []
    bare = FakeCausalLM("base says")
    base = HFClient(bare, FakeHFTokenizer())
    with pytest.raises(ValueError, match="none loaded"):
        base.complete("p", lora_scale=1.0)  # a run labelled with the adapter must not measure the base model
    assert bare.calls == []
    assert base.complete("p").text == base.complete("p", lora_scale=0.0).text == "base says"


class ServerStandIn:
    """llama-server through LocalClient, reduced to what LocalModel uses: the same completion text."""

    def __init__(self, text: str) -> None:
        self.text, self.prompts = text, []

    def complete(self, prompt: str, *, lora_scale: float | None = None,
                 stop: list[str] | None = None) -> Completion:
        self.prompts.append(prompt)
        return Completion(text=self.text, raw={"usage": {"prompt_tokens": 7, "completion_tokens": 3}})


def test_local_model_reads_the_hf_clients_completion_into_the_turn_it_reads_from_llama_server(tmp_path):
    """The hf backend is a client under LocalModel: the same rendered prompt, the same parsing of the text."""
    render = prompt_renderer(FakeTokenizer())
    messages = [{"role": "system", "content": "你是科研助理。"}, {"role": "user", "content": "找幻觉的论文"}]
    tokenizer, server = FakeHFTokenizer(), ServerStandIn(TOOL_TEXT)
    client = HFClient(FakePeftModel(TOOL_TEXT, "base says"), tokenizer, adapter=peft_folder(tmp_path / "a"))
    turns = [LocalModel(c, render, lora_scale=1.0, name="qwen3-1.7b:lora=1").next_turn(messages, MODEL_TOOLS)
             for c in (client, server)]
    assert turns[0].to_api() == turns[1].to_api()
    assert turns[0].tool_calls[0].name == "search_arxiv" and turns[0].tool_calls[0].id == "local_0"
    assert tokenizer.calls[0][0] == server.prompts[0]
    assert turns[0].usage["completion_tokens"] == len(token_ids(TOOL_TEXT)) + 1 and turns[0].paid is False


def test_props_name_the_backend_and_its_settings_and_close_releases_the_model(tmp_path, monkeypatch):
    installed = importlib.metadata.version

    def version(name: str) -> str:
        if name == "peft":  # as in CI, which installs no training package
            raise importlib.metadata.PackageNotFoundError(name)
        return installed(name)

    monkeypatch.setattr(importlib.metadata, "version", version)
    adapter = peft_folder(tmp_path / "adapter")
    client = HFClient(FakePeftModel("a", "b"), FakeHFTokenizer(), adapter=adapter,
                      model_dir="models/Qwen3-1.7B", gpu="Fake GPU 8 GB")
    props = client.props()
    settings = ("backend", "model_path", "dtype", "device", "gpu", "attn_implementation")
    assert {k: props[k] for k in settings} == {
        "backend": "hf", "model_path": "models/Qwen3-1.7B", "dtype": "bfloat16", "device": "cuda:0",
        "gpu": "Fake GPU 8 GB", "attn_implementation": "sdpa"}
    # as the training manifest records them (train.manifest.package_version): "absent" when not installed
    assert props["versions"] == {name: package_version(name) for name in ("torch", "transformers", "peft")}
    assert props["versions"]["peft"] == "absent"
    assert client.last_timings is None and client.health() is True
    client.close()
    assert client.health() is False and client.model is None


def test_an_adapter_must_be_a_peft_folder_with_safetensors_weights_and_a_gguf_is_refused(tmp_path):
    assert check_adapter(peft_folder(tmp_path / "ft" / "adapter")) == tmp_path / "ft" / "adapter"
    checkpoint = peft_folder(tmp_path / "ft" / "trainer" / "checkpoint-209")
    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
    assert check_adapter(checkpoint) == checkpoint
    gguf = tmp_path / "ft" / "gguf" / "checkpoint-209-f16.gguf"
    gguf.parent.mkdir()
    gguf.write_bytes(b"GGUF")
    with pytest.raises(AdapterError, match="GGUF"):
        check_adapter(gguf)
    with pytest.raises(AdapterError, match="GGUF"):
        load_client("models/Qwen3-1.7B", gguf)  # refused before torch or the model is touched
    with pytest.raises(FileNotFoundError, match="model folder"):  # never taken for a Hub repo id
        load_client(tmp_path / "models" / "Qwen3-1.7B", checkpoint)
    with pytest.raises(AdapterError, match="not found"):
        check_adapter(tmp_path / "missing")
    old = tmp_path / "old"
    old.mkdir()
    (old / "adapter_config.json").write_text("{}", encoding="utf-8")
    (old / "adapter_model.bin").write_bytes(b"pickle")
    with pytest.raises(AdapterError, match="adapter_model.safetensors"):
        check_adapter(old)


def test_load_client_loads_bf16_on_the_gpu_from_local_files_with_the_adapter_unmerged_in_eval_mode(
        tmp_path, monkeypatch):
    """How the model is loaded, over fake torch, transformers and peft modules: the weights in bfloat16 on
    cuda:0 with sdpa attention, the tokenizer and the weights from local files only, the PEFT adapter put on
    top unmerged (the client's model is PEFT's wrapper, never merge_and_unload's result), and the model in
    eval mode; without an adapter, no PEFT call at all."""
    model_dir = tmp_path / "models" / "Qwen3-1.7B"
    model_dir.mkdir(parents=True)
    adapter = peft_folder(tmp_path / "ft" / "adapter")
    loads = fake_stack(monkeypatch)
    client = load_client(model_dir, adapter)
    assert loads["tokenizer"] == [((model_dir,), {"local_files_only": True})]
    [base] = loads["model"]
    assert base.args == (model_dir,)
    assert base.kwargs == {"dtype": BFLOAT16, "attn_implementation": "sdpa", "device_map": "cuda:0",
                           "local_files_only": True}
    [wrapper] = loads["peft"]
    assert wrapper.args == (base, adapter) and wrapper.kwargs == {}  # PEFT's default: inference, not training
    assert client.model is wrapper and wrapper.calls == ["eval"] and base.calls == []
    assert (client.adapter, client.model_dir, client.gpu) == (adapter, model_dir, FAKE_GPU)
    assert isinstance(client.tokenizer, FakeHFTokenizer) and client.end_ids == END_IDS
    loads = fake_stack(monkeypatch)
    bare = load_client(model_dir)
    [base] = loads["model"]
    assert loads["peft"] == [] and bare.model is base and base.calls == ["eval"] and bare.adapter is None


def test_the_module_imports_neither_torch_nor_transformers_nor_peft_at_module_level():
    tree = ast.parse(Path(hf.__file__).read_text(encoding="utf-8"))
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add((node.module or "").split(".")[0])
    assert imported.isdisjoint({"torch", "transformers", "peft", "accelerate"}), imported

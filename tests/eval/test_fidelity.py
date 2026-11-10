"""LoRA fidelity, the bake-off's smoke S6 (research_agent.eval.fidelity; docs/m7-底座对比预登记.md 资格 S6),
offline: the arithmetic and the verdict on synthetic top-1 arrays, the synthetic adapter (its scale, its
PEFT files, the safetensors bytes), the module selection as PEFT selects, the norms of the checkpoint tensors,
the teacher-forced sequences over a fake family, the orchestration with the calibration over fake backends,
the load check, and the llama-server backend over a scripted server. Where the local model folders are
present: Qwen3.5-4B's modules from the model built on the meta device, PEFT placing every tensor of a
synthetic adapter there and none of one named after the checkpoint (C3's first S6); on the CPU, a four-layer
Qwen3.5 checkpoint with random weights (the module layout against the model build_causal_lm builds and
loads, the load check through PEFT, the training-stack backend in NF4), and a two-layer Spark through the
same backend."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import struct
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from support.bakeoff_fakes import ServedLlama
from support.family_fakes import token_ids
from support.llama_fakes import FakeLlama, Scripted, fake_loaded_family, install_fake_family

from research_agent.agent.prompts import system_prompt
from research_agent.eval import fidelity
from research_agent.eval.fidelity import (
    AGREEMENT,
    AIM,
    CAP,
    CHANGED,
    FRACTION,
    GROWTH,
    MARGIN,
    AdapterNotLoaded,
    Backends,
    Layout,
    Module,
    Sequence,
    base_modules,
    compare,
    llama_top1_backend,
    read_safetensors_header,
    run_fidelity,
    select_modules,
    sequences_of,
    strengths,
    synthetic_lora,
    target_groups,
    tensor_norm,
    verify_adapter_loaded,
    weight_norm,
    write_adapter,
    write_safetensors,
)
from research_agent.tools.schemas import MODEL_TOOLS

REPO = Path(__file__).resolve().parents[2]
QWEN35 = REPO / "models" / "Qwen3.5-4B"
SPARK = REPO / "models" / "Spark-X2.5-4B"
SYSTEM = {"role": "system", "content": system_prompt("2026-10-10")}
SEARCH = {"name": "search_arxiv", "arguments": {"query": "hallucination", "max_results": 5},
          "arguments_json": "{\"query\": \"hallucination\", \"max_results\": 5}"}
ITEMS = [
    {"id": "q-1", "messages": [SYSTEM, {"role": "user", "content": "找几篇幻觉(hallucination)的论文。"}],
     "expected": {"content": "", "tool_calls": [SEARCH]}, "type": "call", "lang": "zh", "split": "dev",
     "source": {"request": "q-1", "step": 1}},
    {"id": "q-2", "messages": [SYSTEM, {"role": "user", "content": "Thanks, that's all for today."}],
     "expected": {"content": "See you!", "tool_calls": []}, "type": "no_call", "lang": "en", "split": "dev",
     "source": {"handwritten": True}},
]


# ---- the arithmetic and the verdict ---------------------------------------------------------------------


def test_compare_counts_agreement_change_and_the_three_checks():
    hf_lora = [[1, 2, 3, 4], [5, 6, 7, 8]]
    llama_lora = [[1, 2, 3, 9], [5, 6, 7, 8]]  # 7 of 8 agree
    hf_base = [[1, 0, 0, 4], [5, 0, 7, 0]]  # the adapter changed 4 of 8 predictions
    llama_base = [[1, 0, 0, 4], [5, 0, 7, 0]]  # the base agrees everywhere
    report = compare(hf_lora, llama_lora, hf_base, llama_base)
    assert report["positions"] == 8 and report["items"] == 2
    assert report["agreement"] == 7 / 8 and report["base_agreement"] == 1.0 and report["changed"] == 0.5
    assert report["agreeing"] == 7 and report["base_agreeing"] == 8 and report["changed_positions"] == 4
    assert report["checks"] == {"agreement": {"value": 7 / 8, "least": AGREEMENT, "passed": False},
                                "margin": {"value": 7 / 8, "least": 1.0 - MARGIN, "passed": False},
                                "changed": {"value": 0.5, "least": CHANGED, "passed": True}}
    assert report["passed"] is False


@pytest.mark.parametrize(("agree", "base_agree", "changed", "passed"), [
    (18, 19, 6, True),  # 0.90 >= 0.90; 0.90 >= 0.95 - 0.05; 0.30 >= 0.30: every bound is inclusive
    (17, 19, 6, False),  # 0.85 < 0.90
    (18, 20, 6, False),  # 0.90 < 1.00 - 0.05
    (18, 19, 5, False),  # 0.25 < 0.30
    (20, 20, 20, True)])
def test_the_verdict_holds_at_the_preregistered_bounds(agree, base_agree, changed, passed):
    hf_lora = [list(range(20))]
    llama_lora = [[i if i < agree else -1 for i in range(20)]]
    hf_base = [[i if i >= changed else -2 for i in range(20)]]
    llama_base = [[hf_base[0][i] if i < base_agree else -3 for i in range(20)]]
    report = compare(hf_lora, llama_lora, hf_base, llama_base)
    assert (report["agreeing"], report["base_agreeing"], report["changed_positions"]) == (agree, base_agree,
                                                                                         changed)
    assert report["passed"] is passed


def test_compare_refuses_arrays_of_different_shapes_and_empty_ones():
    with pytest.raises(ValueError, match="items"):
        compare([[1, 2]], [[1, 2], [3]], [[1, 2]], [[1, 2]])
    with pytest.raises(ValueError, match="positions"):
        compare([[1, 2]], [[1]], [[1, 2]], [[1, 2]])
    with pytest.raises(ValueError, match="no position"):
        compare([[]], [[]], [[]], [[]])


# ---- the synthetic adapter and its strengths ------------------------------------------------------------


def frobenius(matrix: list[list[float]]) -> float:
    return math.sqrt(sum(x * x for row in matrix for x in row))


def matmul(b: list[list[float]], a: list[list[float]]) -> list[list[float]]:
    return [[sum(b[i][k] * a[k][j] for k in range(len(a))) for j in range(len(a[0]))] for i in range(len(b))]


def test_synthetic_lora_has_pefts_shapes_a_fixed_seed_and_the_asked_scale():
    module = Module("model.layers.0.self_attn.q_proj", in_features=64, out_features=48)
    a, b = synthetic_lora(module, r=4, alpha=8, seed=7, weight_norm=100.0)
    assert len(a) == 4 and all(len(row) == 64 for row in a)  # lora_A: r x in
    assert len(b) == 48 and all(len(row) == 4 for row in b)  # lora_B: out x r
    again = synthetic_lora(module, r=4, alpha=8, seed=7, weight_norm=100.0)
    assert again == (a, b)
    other = synthetic_lora(Module("model.layers.1.self_attn.q_proj", 64, 48), r=4, alpha=8, seed=7,
                           weight_norm=100.0)
    assert other != (a, b)  # another module draws other numbers under the same seed
    assert synthetic_lora(module, r=4, alpha=8, seed=8, weight_norm=100.0) != (a, b)
    # scaling x B A has about FRACTION of the base weight's Frobenius norm: the module's outputs change by
    # about that fraction (docs/m7-计划.md S6: 5-10%)
    delta = frobenius(matmul(b, a)) * 8 / 4
    assert 0.6 * FRACTION * 100.0 <= delta <= 1.4 * FRACTION * 100.0
    # another strength scales B alone: the draws are the same
    a4, b4 = synthetic_lora(module, r=4, alpha=8, seed=7, weight_norm=100.0, fraction=4 * FRACTION)
    assert a4 == a and b4[0] == pytest.approx([4 * x for x in b[0]])


def test_the_strengths_double_from_the_start_up_to_the_cap():
    # module docstring, The strength: from 7% doubling up to twice the base weight's norm, aiming at 35%
    assert (FRACTION, GROWTH, CAP, AIM, CHANGED) == (0.07, 2.0, 2.0, 0.35, 0.30)
    assert strengths() == [0.07, 0.14, 0.28, 0.56, 1.12, 2.0]
    assert strengths(0.5, 3.0, 2.0) == [0.5, 1.5, 2.0] and strengths(2.0, 2.0, 2.0) == [2.0]
    for start, growth, cap in ((0.0, 2.0, 2.0), (3.0, 2.0, 2.0), (0.1, 1.0, 2.0)):
        with pytest.raises(ValueError, match="strengths"):
            strengths(start, growth, cap)


def test_write_adapter_writes_the_peft_folder_the_converter_and_peft_read(tmp_path):
    modules = [Module("model.layers.0.self_attn.q_proj", 6, 4),
               Module("model.layers.1.self_attn.q_proj", 6, 4)]
    loras = {m.name: synthetic_lora(m, r=2, alpha=4, seed=1, weight_norm=1.0) for m in modules}
    folder = write_adapter(tmp_path / "adapter", modules, loras, r=2, alpha=4, targets=["q_proj"],
                           base_model="models/Fake")
    config = json.loads((folder / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["peft_type"] == "LORA" and config["r"] == 2 and config["lora_alpha"] == 4
    assert config["target_modules"] == ["q_proj"] and config["task_type"] == "CAUSAL_LM"
    assert config["base_model_name_or_path"] == "models/Fake" and config["lora_dropout"] == 0.0
    header = read_safetensors_header(folder / "adapter_model.safetensors")
    assert set(header) == {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
                           "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight",
                           "base_model.model.model.layers.1.self_attn.q_proj.lora_A.weight",
                           "base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight"}
    a = header["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    b = header["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    assert a["dtype"] == "F32" and a["shape"] == [2, 6] and b["shape"] == [4, 2]
    assert a["data_offsets"][1] - a["data_offsets"][0] == 2 * 6 * 4
    # the bytes: a u64 header length, the JSON header padded to 8 bytes, then the buffers end to end
    data = (folder / "adapter_model.safetensors").read_bytes()
    length = struct.unpack("<Q", data[:8])[0]
    assert length % 8 == 0 and json.loads(data[8:8 + length].decode("utf-8")) == header
    end = max(v["data_offsets"][1] for v in header.values())
    assert len(data) == 8 + length + end
    first = struct.unpack("<12f", data[8 + length + a["data_offsets"][0]:8 + length + a["data_offsets"][1]])
    assert list(first) == pytest.approx([x for row in loras[modules[0].name][0] for x in row])
    # a regex target is kept a string, as PEFT saves one: PEFT reads a list as module names, never as a regex
    regex = "model\\.layers\\.\\d+\\.self_attn\\.q_proj"
    folder = write_adapter(tmp_path / "regex", modules, loras, r=2, alpha=4, targets=regex,
                           base_model="models/Fake")
    assert json.loads((folder / "adapter_config.json").read_text(encoding="utf-8"))["target_modules"] == regex
    # a tuple of names, a group's target, is a list of names
    folder = write_adapter(tmp_path / "names", modules, loras, r=2, alpha=4, targets=("q_proj",),
                           base_model="models/Fake")
    assert json.loads((folder / "adapter_config.json").read_text(encoding="utf-8"))["target_modules"] == [
        "q_proj"]


def test_write_safetensors_and_the_header_reader_round_trip(tmp_path):
    path = tmp_path / "x.safetensors"
    write_safetensors(path, {"b": ((2, 2), [1.0, 2.0, 3.0, 4.0]), "a": ((1, 3), [5.0, 6.0, 7.0])})
    header = read_safetensors_header(path)
    assert list(header) == ["a", "b"] and header["a"]["data_offsets"] == [0, 12]
    assert header["b"]["data_offsets"] == [12, 28]
    with pytest.raises(ValueError, match="values"):
        write_safetensors(path, {"a": ((2, 2), [1.0])})


# ---- the module selection -------------------------------------------------------------------------------

PROJECTIONS = [Module("model.layers.10.self_attn.q_proj", 3, 4),
               Module("model.layers.0.self_attn.q_proj", 3, 4),
               Module("model.layers.1.self_attn.q_proj", 3, 4), Module("model.layers.0.mlp.down_proj", 4, 3),
               Module("model.layers.0.self_attn.q_proj_extra", 3, 4),
               Module("model.layers.0.self_attn.kq_proj", 3, 4), Module("lm_head", 3, 9)]


def test_select_modules_selects_as_peft_matches_a_regex_or_names():
    # module names: the module of that name and those whose name ends with "." and it, in layer order
    assert [m.name for m in select_modules(PROJECTIONS, ("q_proj",))] == [
        "model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.q_proj",
        "model.layers.10.self_attn.q_proj"]
    assert select_modules(PROJECTIONS, ["lm_head"]) == [Module("lm_head", 3, 9)]
    assert [m.name for m in select_modules(PROJECTIONS, ("self_attn.q_proj", "down_proj"))] == [
        "model.layers.0.mlp.down_proj", "model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.q_proj",
        "model.layers.10.self_attn.q_proj"]
    # a string is a regex matched against the whole name: a bare name selects nothing as a regex
    assert select_modules(PROJECTIONS, "model\\.layers\\.0\\..*_proj") == [
        Module("model.layers.0.mlp.down_proj", 4, 3), Module("model.layers.0.self_attn.kq_proj", 3, 4),
        Module("model.layers.0.self_attn.q_proj", 3, 4)]
    assert select_modules(PROJECTIONS, "q_proj") == [] and select_modules(PROJECTIONS, ("k_proj",)) == []


@pytest.mark.parametrize("target", [("q_proj",), ("self_attn.q_proj", "down_proj"), ("lm_head",),
                                    "model\\.layers\\.0\\..*_proj", "q_proj", ".*q_proj.*"])
def test_select_modules_agrees_with_pefts_own_matching(target):
    pytest.importorskip("peft")
    from peft.tuners.tuners_utils import check_target_module_exists

    config = SimpleNamespace(target_modules=target if isinstance(target, str) else list(target),
                             exclude_modules=None, modules_to_save=None, target_parameters=None,
                             layers_to_transform=None, layers_pattern=None)
    assert {m.name for m in select_modules(PROJECTIONS, target)} == {
        m.name for m in PROJECTIONS if check_target_module_exists(config, m.name)}


# the way registry.py targets Qwen3.5-4B: one PEFT regex over the decoder layers, with or without the
# language_model prefix, for the attention, the Gated DeltaNet (linear_attn) and the MLP projections
HYBRID_TARGETS = ("model\\.(language_model\\.)?layers\\.\\d+\\.(self_attn\\.(q|k|v|o)_proj|"
                  "linear_attn\\.(in_proj_qkv|in_proj_z|out_proj)|mlp\\.(gate|up|down)_proj)")
HYBRID_KINDS = ["linear_attn.in_proj_qkv", "linear_attn.out_proj", "mlp.gate_proj", "self_attn.q_proj"]
HYBRID_MODULES = [Module(name, 3, 2) for name in (
    "model.layers.0.linear_attn.in_proj_qkv", "model.layers.0.linear_attn.out_proj",
    "model.layers.0.linear_attn.in_proj_a", "model.layers.0.mlp.gate_proj", "model.layers.1.self_attn.q_proj",
    "model.layers.1.mlp.gate_proj", "mtp.layers.0.self_attn.q_proj", "model.visual.blocks.0.attn.qkv",
    "lm_head")]  # a text model's modules as Qwen3.5's are named, the MTP head and the vision tower beside


def test_target_groups_split_a_regex_into_one_group_per_kind_of_module():
    # docs/m7-底座对比预登记.md S6: a random LoRA on each group of target modules. A family that targets its
    # layers with one regex still gets one group per kind of module, so that a kind llama.cpp drops (Qwen3.5's
    # Gated DeltaNet projections) fails on its own instead of hiding among the kinds it applies
    asked = []

    def modules_of(target):
        asked.append(target)
        return select_modules(HYBRID_MODULES, target)

    found = target_groups(HYBRID_TARGETS, modules_of)
    assert asked == [HYBRID_TARGETS]  # the regex is resolved once
    assert [(name, [m.name for m in modules]) for name, _, modules in found] == [
        ("linear_attn.in_proj_qkv", ["model.layers.0.linear_attn.in_proj_qkv"]),
        ("linear_attn.out_proj", ["model.layers.0.linear_attn.out_proj"]),
        ("mlp.gate_proj", ["model.layers.0.mlp.gate_proj", "model.layers.1.mlp.gate_proj"]),
        ("self_attn.q_proj", ["model.layers.1.self_attn.q_proj"])]
    # each group's target is the family's regex narrowed to its kind, matched whole as PEFT matches a regex:
    # it selects the group's modules alone (the regex allows the checkpoint's language_model names as well)
    for name, target, modules in found:
        assert select_modules(HYBRID_MODULES, target) == modules
        assert re.fullmatch(target, f"model.language_model.layers.7.{name}") is not None
    q_target = {name: target for name, target, _ in found}["self_attn.q_proj"]
    assert re.fullmatch(q_target, "model.layers.7.self_attn.k_proj") is None
    assert re.fullmatch(q_target, "mtp.layers.0.self_attn.q_proj") is None
    # a list of names: one group per name, its target the name alone in a tuple, which PEFT reads as names
    asked.clear()

    def named(target):
        asked.append(target)
        return [Module(f"model.layers.0.self_attn.{name}", 3, 4) for name in target]

    assert target_groups(("q_proj", "v_proj"), named) == [
        ("q_proj", ("q_proj",), [Module("model.layers.0.self_attn.q_proj", 3, 4)]),
        ("v_proj", ("v_proj",), [Module("model.layers.0.self_attn.v_proj", 3, 4)])]
    assert asked == [("q_proj",), ("v_proj",)]
    for targets in ("model\\.nothing\\.\\d+", ("q_proj",)):
        with pytest.raises(ValueError, match="no module"):
            target_groups(targets, lambda target: [])
    assert fidelity.kind_of("model.layers.12.mlp.down_proj") == "mlp.down_proj"
    assert fidelity.kind_of("lm_head") == "lm_head"


# ---- the norms of the base weights ----------------------------------------------------------------------


def bf16_bytes(values: list[float]) -> bytes:
    """values as bfloat16 (the top 16 bits of each float32, rounded to nearest even)."""
    out = bytearray()
    for value in values:
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
        out += struct.pack("<H", rounded)
    return bytes(out)


def write_raw_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    """A safetensors file of any dtype: name -> (dtype, shape, bytes)."""
    header, offset, blobs = {}, 0, []
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    head = json.dumps(header).encode("utf-8")
    head += b" " * ((8 - len(head) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))


def sharded_model(root: Path) -> Path:
    """A model folder with two shards and an index, the way transformers saves a checkpoint."""
    root.mkdir(parents=True)

    def f32(values: list[float]) -> bytes:
        return struct.pack(f"<{len(values)}f", *values)

    write_raw_safetensors(root / "model-00001-of-00002.safetensors", {
        "model.embed_tokens.weight": ("F32", [4, 3], f32([0.0] * 12)),
        "model.layers.0.self_attn.q_proj.weight": ("F32", [4, 3], f32([3.0, 4.0] + [0.0] * 10)),
        "model.layers.0.self_attn.q_proj.bias": ("F32", [4], f32([1.0] * 4)),
        "model.layers.0.mlp.down_proj.weight": ("F32", [3, 4], f32([1.0] * 12)),
    })
    write_raw_safetensors(root / "model-00002-of-00002.safetensors", {
        "model.layers.1.self_attn.q_proj.weight": ("BF16", [4, 3], bf16_bytes([6.0, 8.0] + [0.0] * 10)),
        "model.layers.10.self_attn.q_proj.weight": ("F32", [4, 3], f32([0.0] * 12)),
    })
    names = ["model.embed_tokens.weight", "model.layers.0.self_attn.q_proj.weight",
             "model.layers.0.self_attn.q_proj.bias", "model.layers.0.mlp.down_proj.weight"]
    index = {"metadata": {"total_size": 1}, "weight_map": {
        **dict.fromkeys(names, "model-00001-of-00002.safetensors"),
        "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
        "model.layers.10.self_attn.q_proj.weight": "model-00002-of-00002.safetensors"}}
    (root / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    return root


def test_tensor_norm_reads_f32_and_bf16_shards_and_checks_the_shape(tmp_path):
    pytest.importorskip("numpy")  # tensor_norm reads the checkpoint with numpy
    model = sharded_model(tmp_path / "model")
    assert tensor_norm(model, "model.layers.0.self_attn.q_proj.weight") == pytest.approx(5.0)
    assert tensor_norm(model, "model.layers.1.self_attn.q_proj.weight", shape=(4, 3)) == pytest.approx(10.0)
    with pytest.raises(ValueError, match="shape"):
        tensor_norm(model, "model.layers.0.self_attn.q_proj.weight", shape=(3, 4))
    with pytest.raises(KeyError):
        tensor_norm(model, "model.layers.2.self_attn.q_proj.weight")
    single = tmp_path / "single"
    single.mkdir()
    write_safetensors(single / "model.safetensors",
                      {"model.layers.0.self_attn.v_proj.weight": ((2, 5), [3.0, 4.0] + [0.0] * 8)})
    assert tensor_norm(single, "model.layers.0.self_attn.v_proj.weight") == pytest.approx(5.0)
    with pytest.raises(FileNotFoundError):
        tensor_norm(tmp_path / "nowhere", "model.layers.0.self_attn.v_proj.weight")


def test_the_tensors_are_read_from_the_file_that_holds_them_whatever_the_index_says(tmp_path):
    pytest.importorskip("numpy")  # tensor_norm reads the checkpoint with numpy
    # transformers loads every tensor of every file the index lists, whatever the index says of each tensor;
    # granite-4.2-3b's index names the other file for 101 of its 363 tensors
    model = sharded_model(tmp_path / "model")
    index = json.loads((model / "model.safetensors.index.json").read_text(encoding="utf-8"))
    files = {"model-00001-of-00002.safetensors": "model-00002-of-00002.safetensors",
             "model-00002-of-00002.safetensors": "model-00001-of-00002.safetensors"}
    index["weight_map"] = {name: files[shard] for name, shard in index["weight_map"].items()}  # all wrong
    (model / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
    assert tensor_norm(model, "model.layers.1.self_attn.q_proj.weight", shape=(4, 3)) == pytest.approx(10.0)
    assert fidelity._shards(model)["model.layers.1.self_attn.q_proj.weight"].name == (
        "model-00002-of-00002.safetensors")
    # a tensor two files hold is not one this check can read as the model's
    write_raw_safetensors(model / "model-00002-of-00002.safetensors", {
        "model.layers.0.self_attn.q_proj.weight": ("F32", [4, 3], struct.pack("<12f", *([1.0] * 12)))})
    with pytest.raises(ValueError, match="both model-00001-of-00002.safetensors and model-00002"):
        tensor_norm(model, "model.layers.0.self_attn.q_proj.weight")


def test_weight_norm_reads_the_checkpoint_tensor_the_loader_loads_into_the_module(tmp_path):
    pytest.importorskip("numpy")  # tensor_norm reads the checkpoint with numpy
    # Qwen3.5's checkpoint names model.language_model.layers.N.* what its text model holds as
    # model.layers.N.*: a module's norm is that of the tensor the layout maps to it, its shape the module's
    model = tmp_path / "vl"
    model.mkdir()
    write_raw_safetensors(model / "model.safetensors", {
        "model.language_model.layers.0.self_attn.q_proj.weight": ("BF16", [4, 3],
                                                                  bf16_bytes([6.0, 8.0] + [0.0] * 10))})
    module = Module("model.layers.0.self_attn.q_proj", 3, 4)
    layout = Layout(modules=(module, Module("lm_head", 3, 9)),
                    tensors={module.name: "model.language_model.layers.0.self_attn.q_proj.weight"})
    assert weight_norm(model, module, layout=layout) == pytest.approx(10.0)
    with pytest.raises(KeyError, match="loads into"):
        weight_norm(model, Module("lm_head", 3, 9), layout=layout)
    with pytest.raises(ValueError, match="shape"):
        weight_norm(model, Module(module.name, 4, 3), layout=layout)
    assert base_modules(model, ("q_proj",), layout=layout) == [module]


# ---- Qwen3.5-4B on the meta device ------------------------------------------------------------------------

QWEN35_GROUPS = {"linear_attn.in_proj_qkv": 24, "linear_attn.in_proj_z": 24, "linear_attn.out_proj": 24,
                 "mlp.down_proj": 32, "mlp.gate_proj": 32, "mlp.up_proj": 32, "self_attn.k_proj": 8,
                 "self_attn.o_proj": 8, "self_attn.q_proj": 8, "self_attn.v_proj": 8}
needs_qwen35 = pytest.mark.skipif(
    not all((QWEN35 / name).is_file() for name in ("config.json", "chat_template.jinja", fidelity.INDEX)),
    reason="local model files are not present")


def placed(model, adapter: Path):
    """PEFT wrapping the model with the adapter folder's own configuration and setting the folder's tensors
    into it, as PeftModel.from_pretrained does: (the wrapped model, the file's keys PEFT found no parameter
    for, the LoRA parameters it set nothing into)."""
    import peft
    from safetensors.torch import load_file

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # setting values into meta tensors does nothing, which torch warns of
        wrapped = peft.get_peft_model(model, peft.PeftConfig.from_pretrained(str(adapter)))
        weights = load_file(str(adapter / "adapter_model.safetensors"))
        result = peft.set_peft_model_state_dict(wrapped, weights)
    return wrapped, list(result.unexpected_keys), [k for k in result.missing_keys if ".lora_" in k]


@needs_qwen35
def test_qwen35s_modules_are_the_text_models_and_peft_places_every_synthetic_tensor(tmp_path):
    # docs/m7-进度.md (底座对比队列的两处工具问题): the synthetic adapters of C3's first S6 were named after
    # the checkpoint's tensors (model.language_model.layers.N.*), while AutoModelForCausalLM builds the text
    # model (model.layers.N.*, the names of C3's trained adapter); PEFT placed none of them. The modules now
    # come from the model as build_causal_lm builds it, on the meta device
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from research_agent.family import registry

    layout = fidelity.model_layout(QWEN35)
    found = target_groups(registry.QWEN35_4B.lora_targets, lambda t: base_modules(QWEN35, t, layout=layout))
    assert {name: len(modules) for name, _, modules in found} == QWEN35_GROUPS  # 200: C3's 400 tensors / 2
    names = [m.name for _, _, modules in found for m in modules]
    assert all(re.fullmatch(r"model\.layers\.\d+\.\w+\.\w+", name) for name in names)
    # the tensor transformers loads into each module's weight is the checkpoint's language-model tensor
    assert all(layout.tensors[name] == f"model.language_model.{name.removeprefix('model.')}.weight"
               for name in names)
    k_proj = Module("model.layers.3.self_attn.k_proj", 2560, 1024)
    assert k_proj in layout.modules and "lm_head" not in layout.tensors  # the head is tied to the embedding
    assert weight_norm(QWEN35, k_proj, layout=layout) == pytest.approx(
        tensor_norm(QWEN35, "model.language_model.layers.3.self_attn.k_proj.weight"))
    # PEFT, wrapping the text model built on the meta device with each group's adapter configuration, finds a
    # parameter for every tensor of the group's synthetic adapter and sets every LoRA parameter it made
    config = transformers.AutoConfig.from_pretrained(QWEN35)
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(config, attn_implementation="sdpa")
    for name, target, modules in found:
        loras = {m.name: synthetic_lora(m, r=1, alpha=2, seed=7, weight_norm=1.0) for m in modules}
        adapter = write_adapter(tmp_path / name, modules, loras, r=1, alpha=2, targets=target,
                                base_model=QWEN35.as_posix())
        wrapped, unexpected, unset = placed(model, adapter)
        assert unexpected == [] and unset == [], name
        model = wrapped.unload()
    # the same LoRAs named after the checkpoint, as C3's first S6 wrote them: PEFT wraps the modules (the
    # family's regex allows both names) and places none of the tensors
    _, target, modules = next(group for group in found if group[0] == "self_attn.k_proj")
    named_as_checkpoint = [Module(m.name.replace("model.", "model.language_model.", 1), m.in_features,
                                  m.out_features) for m in modules]
    loras = {m.name: synthetic_lora(m, r=1, alpha=2, seed=7, weight_norm=1.0) for m in named_as_checkpoint}
    adapter = write_adapter(tmp_path / "checkpoint-names", named_as_checkpoint, loras, r=1, alpha=2,
                            targets=target, base_model=QWEN35.as_posix())
    _, unexpected, unset = placed(model, adapter)
    assert len(unexpected) == len(unset) == 16


# ---- a four-layer Qwen3.5 checkpoint on the CPU -----------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_qwen35(tmp_path_factory):
    """A Qwen3.5 checkpoint of four layers (three linear-attention layers and an attention layer) with random
    weights and the real vocabulary, saved as the vision-language model it is (model.language_model.layers.N.*
    and a vision tower) with the real chat template, so that its folder resolves to the qwen3.5-4b family."""
    if not all((QWEN35 / name).is_file() for name in ("config.json", "chat_template.jinja")):
        pytest.skip("local model files are not present")
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.AutoConfig.from_pretrained(QWEN35)
    text = config.text_config
    text.num_hidden_layers, text.mtp_num_hidden_layers = 4, 0
    text.layer_types = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
    text.hidden_size, text.intermediate_size = 64, 128
    text.num_attention_heads, text.num_key_value_heads, text.head_dim = 4, 2, 16
    text.linear_num_key_heads, text.linear_num_value_heads = 2, 4
    text.linear_key_head_dim, text.linear_value_head_dim = 16, 16
    vision = config.vision_config
    vision.depth, vision.hidden_size, vision.intermediate_size, vision.num_heads = 1, 32, 64, 2
    vision.out_hidden_size = 64
    folder = tmp_path_factory.mktemp("qwen35") / "Qwen3.5-tiny"
    torch.manual_seed(0)
    transformers.Qwen3_5ForConditionalGeneration(config).to(torch.bfloat16).save_pretrained(folder)
    (folder / "chat_template.jinja").write_bytes((QWEN35 / "chat_template.jinja").read_bytes())
    return folder


def checkpoint_tensors(folder: Path) -> dict:
    from safetensors.torch import load_file

    return {name: tensor for shard in sorted(folder.glob("*.safetensors"))
            for name, tensor in load_file(str(shard)).items()}


def test_the_layout_is_the_model_build_causal_lm_builds_and_the_tensors_it_loads(tiny_qwen35):
    torch = pytest.importorskip("torch")
    from research_agent.family.core import build_causal_lm, folder_family

    layout = fidelity.model_layout(tiny_qwen35)
    model = build_causal_lm(folder_family(tiny_qwen35), str(tiny_qwen35), dtype=torch.float32)
    assert type(model).__name__ == "Qwen3_5ForCausalLM"
    built = [Module(name, m.in_features, m.out_features) for name, m in model.named_modules()
             if isinstance(m, torch.nn.Linear)]
    assert list(layout.modules) == built and len(built) == 32  # 4 x 3 MLP, 3 x 5 linear attention, 4, head
    checkpoint, state = checkpoint_tensors(tiny_qwen35), model.state_dict()
    assert not any(name.startswith("model.layers.") for name in checkpoint)  # the checkpoint's names differ
    for module in built:
        if module.name == "lm_head":
            assert module.name not in layout.tensors  # tied to the embedding, no tensor of its own
            continue
        tensor = layout.tensors[module.name]
        assert tensor == f"model.language_model.{module.name.removeprefix('model.')}.weight"
        assert torch.equal(checkpoint[tensor].float(), state[f"{module.name}.weight"]), module.name
    first = built[0]
    assert weight_norm(tiny_qwen35, first, layout=layout) == pytest.approx(
        float(state[f"{first.name}.weight"].double().norm()), rel=1e-6)


def test_the_layout_maps_no_tensor_to_a_weight_assembled_or_loaded_twice(tiny_qwen35, monkeypatch):
    # a weight a converter assembles, or one two checkpoint tensors load into, has no single tensor whose norm
    # is the module's: weight_norm then refuses the module instead of reading another tensor's
    pytest.importorskip("transformers")
    from transformers import core_model_loading

    real = core_model_loading.rename_source_key

    def renamed(key, *args, **kwargs):
        name, converted = real(key, *args, **kwargs)
        if ".layers.0.mlp.up_proj." in key:
            return name, "a converter's source pattern"
        if ".layers.1.mlp.up_proj." in key:
            return name.replace(".up_proj.", ".down_proj."), converted
        return name, converted

    monkeypatch.setattr(core_model_loading, "rename_source_key", renamed)
    tensors = fidelity.model_layout(tiny_qwen35).tensors
    for module in ("model.layers.0.mlp.up_proj", "model.layers.1.mlp.up_proj",
                   "model.layers.1.mlp.down_proj"):
        assert module not in tensors, module
    assert tensors["model.layers.0.mlp.down_proj"] == "model.language_model.layers.0.mlp.down_proj.weight"
    assert len(tensors) == 31 - 3  # every Linear but the head, less the three


def test_the_layout_requires_the_model_to_be_of_the_patched_class(tiny_qwen35, monkeypatch):
    class Elsewhere:
        """A class the auto class does not build."""

    monkeypatch.setattr(fidelity, "patch_model_class", lambda spec, folder: Elsewhere)
    with pytest.raises(fidelity.FamilyError, match="not the patched"):
        fidelity.model_layout(tiny_qwen35)


@pytest.mark.skipif(not (SPARK / "modeling_spark.py").is_file(), reason="local model files are not present")
def test_the_layout_of_spark_is_its_patched_vendor_model_on_the_meta_device(monkeypatch):
    # the vendor class builds only with the family's patch set (registry: spark-x2.5-4b), which the layout
    # sets as build_causal_lm does; its targets are module names, one group per name
    transformers = pytest.importorskip("transformers")
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from research_agent.family import registry

    config = transformers.AutoConfig.from_pretrained(SPARK, trust_remote_code=True)
    vendor = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], SPARK)
    restore_vendor(monkeypatch, vendor)
    monkeypatch.setattr(vendor, "_tied_weights_keys", ["lm_head.weight"])  # as the vendor wrote it
    layout = fidelity.model_layout(SPARK)
    assert vendor._tied_weights_keys == {"lm_head.weight": "model.embedding.weight"}
    found = target_groups(registry.SPARK_X25_4B.lora_targets, lambda t: base_modules(SPARK, t, layout=layout))
    assert {name: len(modules) for name, _, modules in found} == dict.fromkeys(
        registry.SPARK_X25_4B.lora_targets, config.num_hidden_layers)
    q_k_v = found[0][2][0]
    assert q_k_v.name == "model.layers.0.self_attn.q_k_v_proj" and layout.tensors[q_k_v.name] == (
        "model.layers.0.self_attn.q_k_v_proj.weight")


def test_the_load_check_holds_for_the_models_names_and_refuses_the_checkpoints(tiny_qwen35, tmp_path):
    torch = pytest.importorskip("torch")
    peft = pytest.importorskip("peft")
    from research_agent.family import registry
    from research_agent.family.core import build_causal_lm, folder_family

    layout = fidelity.model_layout(tiny_qwen35)
    found = target_groups(registry.QWEN35_4B.lora_targets,
                          lambda t: base_modules(tiny_qwen35, t, layout=layout))
    _, target, modules = next(group for group in found if group[0] == "mlp.down_proj")
    loras = {m.name: synthetic_lora(m, r=4, alpha=8, seed=7, weight_norm=weight_norm(tiny_qwen35, m,
                                                                                     layout=layout))
             for m in modules}
    adapter = write_adapter(tmp_path / "model-names", modules, loras, r=4, alpha=8, targets=target,
                            base_model=tiny_qwen35.as_posix())
    model = build_causal_lm(folder_family(tiny_qwen35), str(tiny_qwen35), dtype=torch.float32)
    wrapped = peft.PeftModel.from_pretrained(model, str(adapter))
    assert verify_adapter_loaded(wrapped, adapter) == {"tensors": 8, "modules": 4}
    model = wrapped.unload()
    # the same LoRAs named after the checkpoint: PEFT wraps the four modules, every lora_B stays at zero
    renamed = {m.name: Module(m.name.replace("model.", "model.language_model.", 1), m.in_features,
                              m.out_features) for m in modules}
    old = write_adapter(tmp_path / "checkpoint-names", list(renamed.values()),
                        {renamed[name].name: lora for name, lora in loras.items()}, r=4, alpha=8,
                        targets=target, base_model=tiny_qwen35.as_posix())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # PEFT warns of the keys it did not place, and goes on
        wrapped = peft.PeftModel.from_pretrained(model, str(old))
    lora_b = [p for name, p in wrapped.named_parameters() if ".lora_B." in name]
    assert len(lora_b) == 4 and not any(bool(p.any()) for p in lora_b)
    with pytest.raises(AdapterNotLoaded, match=(
            r"^0 of the 8 LoRA tensors .* are in the model with the file's values; 8 of the file's tensors "
            r"are no parameter of the model \(first: "
            r"base_model\.model\.model\.language_model\.layers\.0\.mlp\.down_proj\.lora_A\.weight\); "
            r"8 LoRA parameters of the model have no tensor in the file")):
        verify_adapter_loaded(wrapped, old)


def torch_linear_attention(monkeypatch) -> None:
    """Qwen3.5's Gated DeltaNet on the CPU: its kernels from fla run on a GPU alone (Triton), so the module's
    names are set to the torch reference functions transformers wraps (inspect.unwrap)."""
    from transformers.models.qwen3_5 import modeling_qwen3_5

    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule", "causal_conv1d_fn",
                 "causal_conv1d_update"):
        monkeypatch.setattr(modeling_qwen3_5, name, inspect.unwrap(getattr(modeling_qwen3_5, name)))


def on_the_cpu(monkeypatch, transformers) -> None:
    """from_pretrained places the backend's model on the CPU, NF4 and all, instead of its GPU 0."""
    real = transformers.AutoModelForCausalLM.from_pretrained

    def placed_on_the_cpu(path, **kwargs):
        assert kwargs.pop("device_map") == {"": 0}  # the backend's GPU placement
        return real(path, device_map={"": "cpu"}, **kwargs)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", placed_on_the_cpu)


def spy_on_the_load_check(monkeypatch, torch, models: list | None = None) -> list[tuple[int, bool, dict]]:
    """verify_adapter_loaded as the backend calls it, kept: per call that passes, (the adapter file's tensor
    count, whether the model held every one of them as its LoRA parameter with the file's values at that
    moment, read here on its own, the check's counts); every model it is called with goes to models."""
    from safetensors.torch import load_file

    checked: list[tuple[int, bool, dict]] = []
    check = fidelity.verify_adapter_loaded

    def spy(model, adapter, **kwargs):
        if models is not None:
            models.append(model)
        weights = load_file(str(Path(adapter) / "adapter_model.safetensors"))
        parameters = dict(model.named_parameters())
        names = {key: re.sub(r"\.lora_([AB])\.weight$", r".lora_\1.default.weight", key) for key in weights}
        held = all(names[key] in parameters and torch.equal(parameters[names[key]].detach().cpu(),
                                                            tensor.to(parameters[names[key]].dtype))
                   for key, tensor in weights.items())
        counts = check(model, adapter, **kwargs)
        checked.append((len(weights), held, counts))
        return counts

    monkeypatch.setattr(fidelity, "verify_adapter_loaded", spy)
    return checked


def test_the_hf_backend_loads_a_tiny_qwen35s_adapter_in_nf4_on_the_cpu_or_refuses_it(tiny_qwen35, tmp_path,
                                                                                   monkeypatch):
    # S6's training-stack side for Qwen3.5, the GPU placement aside: the backend builds the four-layer
    # checkpoint in NF4 through build_causal_lm, loads a synthetic adapter on every targeted module (named
    # as the model names them), checks the load, predicts, and unloads; an adapter named after the
    # checkpoint is refused, and the base predicts as before either way
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("bitsandbytes")
    from research_agent.family import registry

    torch_linear_attention(monkeypatch)
    on_the_cpu(monkeypatch, transformers)
    models: list = []
    checked = spy_on_the_load_check(monkeypatch, torch, models)

    def wrapped(model) -> bool:  # whether the base under a PEFT model still holds LoRA layers
        return any(".lora_" in name for name, _ in model.get_base_model().named_parameters())

    sequences = [Sequence(ids=list(range(1000, 1040)), start=30, positions=10),
                 Sequence(ids=list(range(2000, 2030)), start=20, positions=10)]
    predict = fidelity.hf_top1_backend(tiny_qwen35)
    base = predict(None, sequences)
    assert [len(row) for row in base] == [10, 10]
    targets = registry.QWEN35_4B.lora_targets
    layout = fidelity.model_layout(tiny_qwen35)
    modules = base_modules(tiny_qwen35, targets, layout=layout)
    assert len(modules) == 25  # 4 x 3 MLP, 3 x 3 linear attention, 4 attention projections
    loras = {m.name: synthetic_lora(m, r=4, alpha=8, seed=7, fraction=CAP,
                                    weight_norm=weight_norm(tiny_qwen35, m, layout=layout)) for m in modules}
    adapter = write_adapter(tmp_path / "adapter", modules, loras, r=4, alpha=8, targets=targets,
                            base_model=tiny_qwen35.as_posix())
    moved = predict(adapter, sequences)
    assert checked == [(50, True, {"tensors": 50, "modules": 25})]  # every tensor of the file, its values
    assert sum(a != b for row, base_row in zip(moved, base, strict=True)
               for a, b in zip(row, base_row, strict=True)) > 0
    assert predict(None, sequences) == base and not wrapped(models[-1])  # unloaded again
    # a forward pass that fails leaves the base unwrapped too
    with pytest.raises(IndexError):
        predict(adapter, [Sequence(ids=[10**7], start=1, positions=1)])  # no such token
    assert len(models) == len(checked) == 2 and not wrapped(models[-1])
    renamed = [Module(m.name.replace("model.", "model.language_model.", 1), m.in_features, m.out_features)
               for m in modules]
    old = write_adapter(tmp_path / "checkpoint-names", renamed,
                        {r.name: loras[m.name] for r, m in zip(renamed, modules, strict=True)}, r=4, alpha=8,
                        targets=targets, base_model=tiny_qwen35.as_posix())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(AdapterNotLoaded, match="^0 of the 50 LoRA tensors"):
            predict(old, sequences)
    assert any("missing adapter keys" in str(w.message) for w in caught)  # all PEFT itself does is warn
    assert len(models) == 3 and not wrapped(models[-1])  # the refused adapter is unloaded too
    assert predict(None, sequences) == base and len(checked) == 2


# ---- the load check ---------------------------------------------------------------------------------------


class Holder:
    """What verify_adapter_loaded reads of a model: its named parameters."""

    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters

    def named_parameters(self):
        return iter(self.parameters.items())


def peft_parameters(adapter: Path, *, adapter_name: str = "default", dtype=None) -> dict:
    """The LoRA parameters PEFT makes of an adapter folder it loads: the file's tensors, named with the
    adapter's name (lora_A.<name>.weight), in dtype when given."""
    from safetensors.torch import load_file

    found = {}
    for key, tensor in load_file(str(adapter / "adapter_model.safetensors")).items():
        name = re.sub(r"\.lora_([AB])\.weight$", rf".lora_\1.{adapter_name}.weight", key)
        found[name] = tensor if dtype is None else tensor.to(dtype)
    return found


def test_the_load_check_needs_every_tensor_of_the_file_in_the_model_with_its_values(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    modules = [Module("model.layers.0.self_attn.q_proj", 6, 4),
               Module("model.layers.1.self_attn.q_proj", 6, 4)]
    loras = {m.name: synthetic_lora(m, r=2, alpha=4, seed=1, weight_norm=1.0) for m in modules}
    adapter = write_adapter(tmp_path / "adapter", modules, loras, r=2, alpha=4, targets=("q_proj",),
                            base_model="m")
    params = peft_parameters(adapter)
    base = {"base_model.model.model.layers.0.self_attn.q_proj.base_layer.weight": torch.zeros(4, 6)}
    assert verify_adapter_loaded(Holder({**base, **params}), adapter) == {"tensors": 4, "modules": 2}
    # parameters of a narrower dtype hold the file's values as that dtype holds them
    narrow = peft_parameters(adapter, dtype=torch.bfloat16)
    assert verify_adapter_loaded(Holder(narrow), adapter) == {"tensors": 4, "modules": 2}
    # another adapter's parameters are not this one's
    other = peft_parameters(adapter, adapter_name="other")
    assert verify_adapter_loaded(Holder({**params, **other}), adapter)["tensors"] == 4
    with pytest.raises(AdapterNotLoaded, match="^0 of the 4 LoRA tensors .* 4 of the file's tensors are no "
                                               "parameter of the model"):
        verify_adapter_loaded(Holder(other), adapter)
    missing = {name: p for name, p in params.items() if ".layers.0.self_attn.q_proj.lora_A." not in name}
    with pytest.raises(AdapterNotLoaded, match=(
            r"^3 of the 4 LoRA tensors .*; 1 of the file's tensors are no parameter of the model \(first: "
            r"base_model\.model\.model\.layers\.0\.self_attn\.q_proj\.lora_A\.weight\)$")):
        verify_adapter_loaded(Holder(missing), adapter)
    stray = "base_model.model.model.layers.2.self_attn.q_proj.lora_B.default.weight"
    extra = {**params, stray: torch.zeros(4, 2)}
    with pytest.raises(AdapterNotLoaded, match=("^4 of the 4 .*; 1 LoRA parameters of the model have no "
                                                "tensor in the file")):
        verify_adapter_loaded(Holder(extra), adapter)
    # lora_B left at the zeros PEFT starts it with, as in C3's first S6: other values than the file's
    zeroed = {name: torch.zeros_like(p) if ".lora_B." in name else p for name, p in params.items()}
    with pytest.raises(AdapterNotLoaded, match="^2 of the 4 .*; 2 parameters hold other values than the "
                                               "file"):
        verify_adapter_loaded(Holder(zeroed), adapter)
    write_safetensors(tmp_path / "empty" / "adapter_model.safetensors", {"x": ((1,), [0.0])})
    with pytest.raises(AdapterNotLoaded, match="the file holds none"):
        verify_adapter_loaded(Holder({}), tmp_path / "empty")


# ---- the sequences --------------------------------------------------------------------------------------


def test_sequences_are_the_prompt_and_the_gold_answer_cut_to_the_positions(tmp_path):
    family = fake_loaded_family(tmp_path / "model", "hybrid")
    sequences = sequences_of(family, ITEMS, positions=16)
    assert len(sequences) == 2
    call, text = sequences
    assert call.positions == 16 and len(call.ids) == call.start + 16
    assert text.positions < 16 and len(text.ids) == text.start + text.positions  # "See you!<|im_end|>"
    answer = family.decode(text.ids[text.start:])
    assert answer == "See you!<|im_end|>"
    assert token_ids(family.decode(call.ids[call.start:])) == call.ids[call.start:]
    assert family.decode(call.ids[call.start:]).startswith("<tool_call>")
    prompt = family.prompt(family.to_template_messages(ITEMS[0]["messages"]), MODEL_TOOLS, fidelity.MODE)
    assert call.ids[:call.start] == family.encode_prompt(prompt)
    with pytest.raises(ValueError, match="positions"):
        sequences_of(family, ITEMS, positions=0)


# ---- the orchestration over fake backends ----------------------------------------------------------------

SEED, NORM = 3, 10.0  # the fake runs' seed and every module's base weight norm


def adapter_fraction(adapter: Path) -> float:
    """The strength an adapter folder was written at, read off its file: its first lora_B entry over the one
    synthetic_lora draws for that module at fraction 1 (the draws are the same at every strength, B scaled),
    under SEED and NORM."""
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    path = adapter / "adapter_model.safetensors"
    header = read_safetensors_header(path)
    name = min(key for key in header if key.endswith(".lora_B.weight"))
    out_features = header[name]["shape"][0]
    in_features = header[name.replace(".lora_B.", ".lora_A.")]["shape"][1]
    data = path.read_bytes()
    start = 8 + struct.unpack("<Q", data[:8])[0] + header[name]["data_offsets"][0]
    (first,) = struct.unpack("<f", data[start:start + 4])
    module = Module(name.removeprefix("base_model.model.").removesuffix(".lora_B.weight"), in_features,
                    out_features)
    _, unit = synthetic_lora(module, r=config["r"], alpha=config["lora_alpha"], seed=SEED, weight_norm=NORM,
                             fraction=1.0)
    return round(first / unit[0][0], 4)


class FakeBackends:
    """The model parts of S6, scripted. The base predicts position p as p. An adapter of group g written at
    fraction f (read off its file, adapter_fraction) moves the first changed[g] positions of each item to
    p + 1, changed[g] a count or a function of f; llama.cpp repeats the training stack's predictions with the
    same adapter except at the first disagree[g] positions (base_disagree without an adapter); a group in
    unloaded raises AdapterNotLoaded, as the HF backend's load check would. Every call is kept with the
    group and the fraction."""

    def __init__(self, root: Path, *, changed: dict, disagree: dict[str, int] | None = None,
                 base_disagree: int = 0, unloaded: tuple[str, ...] = ()) -> None:
        self.root, self.changed, self.disagree = root, changed, disagree or {}
        self.base_disagree, self.unloaded = base_disagree, unloaded
        self.calls: list[tuple] = []
        self.exports: list[tuple[Path, Path]] = []

    def modules(self, target) -> list[Module]:
        self.calls.append(("modules", target))
        names = target if isinstance(target, tuple) else (target,)
        return [Module(f"model.layers.{i}.self_attn.{name}", 6, 4) for name in names for i in range(2)]

    def weight_norm(self, module: Module) -> float:
        return NORM

    def export(self, adapter: Path, outfile: Path) -> Path:
        self.exports.append((adapter, outfile))
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_bytes(b"GGUF " + adapter.parent.name.encode())
        return outfile

    def _moved(self, adapter: Path, sequences: list[Sequence]) -> tuple[str, float, list[list[int]]]:
        group, fraction = adapter.parent.name, adapter_fraction(adapter)
        rule = self.changed[group]
        count = rule(fraction) if callable(rule) else rule
        return group, fraction, [[p + 1 if p < count else p for p in range(s.positions)] for s in sequences]

    def hf_top1(self, adapter: Path | None, sequences: list[Sequence]) -> list[list[int]]:
        if adapter is None:
            self.calls.append(("hf", None, None))
            return [list(range(s.positions)) for s in sequences]
        if adapter.parent.name in self.unloaded:
            self.calls.append(("hf", adapter.parent.name, None))
            raise AdapterNotLoaded(f"0 of the 4 LoRA tensors of {adapter.as_posix()} are in the model")
        group, fraction, predictions = self._moved(adapter, sequences)
        self.calls.append(("hf", group, fraction))
        return predictions

    def llama_top1(self, gguf: Path | None, sequences: list[Sequence]) -> list[list[int]]:
        if gguf is None:
            self.calls.append(("llama", None, None))
            predictions, wrong = [list(range(s.positions)) for s in sequences], self.base_disagree
        else:
            group, fraction, predictions = self._moved(gguf.parent / "adapter", sequences)
            self.calls.append(("llama", group, fraction))
            wrong = self.disagree.get(group, 0)
        return [[-1 if p < wrong else token for p, token in enumerate(row)] for row in predictions]


def backends_of(fake: FakeBackends) -> Backends:
    return Backends(modules=fake.modules, weight_norm=fake.weight_norm, export=fake.export,
                    hf_top1=fake.hf_top1, llama_top1=fake.llama_top1)


SEQUENCES = [Sequence(ids=list(range(30)), start=20, positions=10) for _ in range(2)]  # 20 positions in all


def steps(*pairs: tuple[float, int]):
    """A group's moved positions per item as a step function of the strength: the count of the last pair
    whose fraction is at most the strength (0 below the first)."""
    def rule(fraction: float) -> int:
        return max([count for least, count in pairs if fraction >= least], default=0)

    return rule


def test_run_fidelity_calibrates_every_group_on_the_training_stack_before_llama_predicts(tmp_path,
                                                                                         monkeypatch):
    monkeypatch.chdir(tmp_path)
    # q_proj moves 0, 2, 3, 4 positions of 10 at 0.07, 0.14, 0.28, 0.56: 0.40 >= 0.35 stops it at 0.56;
    # v_proj moves 2 at every strength (0.20): no power at the cap; k_proj moves 3 at the cap alone (0.30):
    # short of the aim, at the least change, compared at the cap. A GGUF an earlier run left is removed
    fake = FakeBackends(tmp_path, changed={"q_proj": steps((0.14, 2), (0.28, 3), (0.56, 4)), "v_proj": 2,
                                           "k_proj": steps((0.07, 1), (2.0, 3))}, disagree={"q_proj": 1},
                        base_disagree=1)
    stale = tmp_path / "work" / "v_proj" / "adapter-f16.gguf"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"GGUF of an earlier run")
    report = run_fidelity(("q_proj", "v_proj", "k_proj"), SEQUENCES, backends_of(fake), Path("work"), r=2,
                          alpha=4, seed=SEED, base_model="models/Fake")
    predictions = [c for c in fake.calls if c[0] != "modules"]
    # every strength is fixed on the training stack before llama.cpp predicts anything
    assert predictions == [
        ("hf", None, None), ("hf", "q_proj", 0.07), ("hf", "q_proj", 0.14), ("hf", "q_proj", 0.28),
        ("hf", "q_proj", 0.56), *(("hf", "v_proj", f) for f in strengths()),
        *(("hf", "k_proj", f) for f in strengths()),
        ("llama", None, None), ("llama", "q_proj", 0.56), ("llama", "k_proj", 2.0)]
    q, v, k = (report["groups"][name] for name in ("q_proj", "v_proj", "k_proj"))
    assert q["calibration"] == {"trials": [
        {"fraction": 0.07, "changed_positions": 0, "changed": 0.0},
        {"fraction": 0.14, "changed_positions": 4, "changed": 0.2},
        {"fraction": 0.28, "changed_positions": 6, "changed": 0.3},
        {"fraction": 0.56, "changed_positions": 8, "changed": 0.4}],
        "fraction": 0.56, "changed": 0.4, "calibrated": True}
    assert q["fraction"] == 0.56 and q["changed"] == 0.4 and q["agreement"] == 0.9 and q["passed"] is True
    assert q["failures"] == [] and q["gguf"] == "work/q_proj/adapter-f16.gguf" and len(q["gguf_sha256"]) == 64
    assert adapter_fraction(tmp_path / "work" / "q_proj" / "adapter") == 0.56  # the chosen one stays
    assert k["fraction"] == 2.0 and k["changed"] == 0.3 and k["calibration"]["calibrated"] is True
    assert k["passed"] is True and k["checks"]["changed"] == {"value": 0.3, "least": CHANGED, "passed": True}
    assert v["calibration"]["calibrated"] is False and v["fraction"] == 2.0 and v["changed"] == 0.2
    assert v["changed_positions"] == 4 and v["gguf"] is None and v["gguf_sha256"] is None
    assert v["failures"] == [("changed 0.200 at the cap fraction 2 (least 0.300): no strength up to the cap "
                              "gives the comparison its power")]
    assert v["passed"] is False and "checks" not in v and not stale.exists()
    assert [outfile for _, outfile in fake.exports] == [Path("work/q_proj/adapter-f16.gguf"),
                                                        Path("work/k_proj/adapter-f16.gguf")]
    assert report["failed"] == ["v_proj"] and report["passed"] is False
    assert report["base"] == {"items": 2, "positions": 20, "agreeing": 18, "agreement": 0.9}
    assert report["settings"] == {"r": 2, "alpha": 4, "seed": SEED, "fraction": FRACTION, "growth": GROWTH,
                                  "cap": CAP, "aim": AIM, "fractions": strengths(), "items": 2,
                                  "positions": 10, "least_agreement": AGREEMENT, "margin": MARGIN,
                                  "least_changed": CHANGED}
    assert q["modules"] == ["model.layers.0.self_attn.q_proj", "model.layers.1.self_attn.q_proj"]
    assert q["target"] == ["q_proj"] and q["adapter"] == "work/q_proj/adapter"
    assert len(q["adapter_sha256"]) == 64
    assert list(q) == ["target", "modules", "calibration", "fraction", "adapter", "adapter_sha256", "gguf",
                       "gguf_sha256", "items", "positions", "agreeing", "agreement", "base_agreeing",
                       "base_agreement", "changed_positions", "changed", "checks", "failures", "passed"]
    json.dumps(report)  # the report is JSON as it stands


def test_calibration_keeps_an_earlier_strength_with_power_when_the_change_does_not_grow(tmp_path,
                                                                                      monkeypatch):
    monkeypatch.chdir(tmp_path)
    # q_proj moves 1 position of 10 per item at 0.07, 3 at 0.14 and 2 from 0.28 on: 0.14's 0.30 has the
    # power, short of the aim, and no later strength reaches either; 0.14 is the group's, its adapter written
    # again (the folder held the cap's) and compared. v_proj moves 2, then 3 at 0.28 and 0.56, then 3 at the
    # cap: of equal changes the smallest strength is kept
    fake = FakeBackends(tmp_path, changed={"q_proj": lambda f: {0.07: 1, 0.14: 3}.get(round(f, 6), 2),
                                           "v_proj": lambda f: 3 if f >= 0.28 else 2})
    report = run_fidelity(("q_proj", "v_proj"), SEQUENCES, backends_of(fake), Path("work"), r=2, alpha=4,
                          seed=SEED, base_model="models/Fake")
    q, v = report["groups"]["q_proj"], report["groups"]["v_proj"]
    assert [t["changed"] for t in q["calibration"]["trials"]] == [0.1, 0.3, 0.2, 0.2, 0.2, 0.2]
    assert {k: q["calibration"][k] for k in ("fraction", "changed", "calibrated")} == {
        "fraction": 0.14, "changed": 0.3, "calibrated": True}
    assert q["fraction"] == 0.14 and q["changed"] == 0.3 and q["passed"] is True
    assert adapter_fraction(tmp_path / "work" / "q_proj" / "adapter") == 0.14
    assert v["calibration"]["fraction"] == 0.28 and v["changed"] == 0.3 and v["passed"] is True
    assert adapter_fraction(tmp_path / "work" / "v_proj" / "adapter") == 0.28
    assert [c for c in fake.calls if c[0] == "llama"] == [("llama", None, None), ("llama", "q_proj", 0.14),
                                                          ("llama", "v_proj", 0.28)]
    assert report["passed"] is True


def test_run_fidelity_checks_every_group_against_the_base_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # per item: the base disagrees at 1 position, q_proj's adapter at 1, v_proj's at none
    fake = FakeBackends(tmp_path, changed={"q_proj": 8, "v_proj": 4}, disagree={"q_proj": 1}, base_disagree=1)
    report = run_fidelity(("q_proj", "v_proj"), SEQUENCES, backends_of(fake), Path("work"), r=2, alpha=4,
                          seed=SEED, base_model="models/Fake")
    assert sum(1 for c in fake.calls if c == ("hf", None, None)) == 1  # the base predictions are made once
    assert sum(1 for c in fake.calls if c == ("llama", None, None)) == 1
    assert report["base"] == {"items": 2, "positions": 20, "agreeing": 18, "agreement": 0.9}
    assert list(report["groups"]) == ["q_proj", "v_proj"]
    q = report["groups"]["q_proj"]
    assert q["agreement"] == 0.9 and q["changed"] == 0.8 and q["passed"] is True
    v = report["groups"]["v_proj"]
    assert v["agreement"] == 1.0 and v["changed"] == 0.4 and v["passed"] is True
    assert report["passed"] is True and report["failed"] == []
    assert (tmp_path / "work" / "q_proj" / "adapter" / "adapter_config.json").is_file()
    assert fake.exports[0] == (Path("work/q_proj/adapter"), Path("work/q_proj/adapter-f16.gguf"))
    header = read_safetensors_header(tmp_path / "work" / "v_proj" / "adapter" / "adapter_model.safetensors")
    assert len(header) == 4  # two modules, A and B each


def test_run_fidelity_measures_the_margin_against_the_base_agreement(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = FakeBackends(tmp_path, changed={"q_proj": 20}, disagree={"q_proj": 1}, base_disagree=0)
    report = run_fidelity(("q_proj",), SEQUENCES, backends_of(fake), Path("w"), r=2, alpha=4, seed=SEED,
                          base_model="m")
    q = report["groups"]["q_proj"]
    assert q["agreement"] == 0.9 and q["base_agreement"] == 1.0 and q["passed"] is False
    assert q["checks"]["margin"] == {"value": 0.9, "least": 0.95, "passed": False}
    assert q["failures"] == ["margin 0.900 (least 0.950)"]
    fake = FakeBackends(tmp_path, changed={"q_proj": 20}, disagree={"q_proj": 1}, base_disagree=1)
    report = run_fidelity(("q_proj",), SEQUENCES, backends_of(fake), Path("w2"), r=2, alpha=4, seed=SEED,
                          base_model="m")
    assert report["groups"]["q_proj"]["passed"] is True and report["base"]["agreement"] == 0.9


def test_run_fidelity_fails_a_group_whose_adapter_the_training_stack_did_not_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = FakeBackends(tmp_path, changed={"q_proj": 8, "v_proj": 8}, unloaded=("q_proj",))
    report = run_fidelity(("q_proj", "v_proj"), SEQUENCES, backends_of(fake), Path("work"), r=2, alpha=4,
                          seed=SEED, base_model="m")
    q = report["groups"]["q_proj"]
    error = ("the training stack did not load the adapter at fraction 0.07: 0 of the 4 LoRA tensors of "
             "work/q_proj/adapter are in the model")
    assert q["calibration"] == {"trials": [], "fraction": None, "changed": None, "calibrated": False,
                                "error": error}
    assert q["failures"] == [error] and q["passed"] is False and q["gguf"] is None
    assert report["groups"]["v_proj"]["passed"] is True and report["failed"] == ["q_proj"]
    assert [outfile for _, outfile in fake.exports] == [Path("work/v_proj/adapter-f16.gguf")]
    assert ("llama", "q_proj", 0.07) not in fake.calls


def test_run_fidelity_with_no_group_calibrated_starts_no_llama_server(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fake = FakeBackends(tmp_path, changed={"q_proj": 0})
    report = run_fidelity(("q_proj",), SEQUENCES, backends_of(fake), Path("work"), r=2, alpha=4, seed=SEED,
                          base_model="m")
    assert all(c[0] != "llama" for c in fake.calls) and fake.exports == []
    assert report["base"] == {"items": 2, "positions": 20, "agreeing": None, "agreement": None}
    assert report["failed"] == ["q_proj"] and report["passed"] is False


def test_run_fidelity_puts_a_lora_on_each_kind_a_regex_targets_and_fails_the_kind_llama_drops(tmp_path,
                                                                                               monkeypatch):
    monkeypatch.chdir(tmp_path)

    class Hybrid(FakeBackends):
        def modules(self, target) -> list[Module]:
            self.calls.append(("modules", target))
            return select_modules([Module(m.name, 6, 4) for m in HYBRID_MODULES], target)

    # llama.cpp disagrees at 6 of 10 positions per item with the linear-attention qkv adapter alone, as if it
    # dropped those tensors; every kind's LoRA changes 8 of HF's 10 top-1 tokens
    fake = Hybrid(tmp_path, changed=dict.fromkeys(HYBRID_KINDS, 8), disagree={"linear_attn.in_proj_qkv": 6})
    report = run_fidelity(HYBRID_TARGETS, SEQUENCES, backends_of(fake), Path("work"), r=2, alpha=4, seed=SEED,
                          base_model="models/Fake")
    assert list(report["groups"]) == HYBRID_KINDS
    assert report["failed"] == ["linear_attn.in_proj_qkv"] and report["passed"] is False
    assert [c for c in fake.calls if c[0] == "modules"] == [("modules", HYBRID_TARGETS)]  # resolved once
    qkv = report["groups"]["linear_attn.in_proj_qkv"]
    assert qkv["modules"] == ["model.layers.0.linear_attn.in_proj_qkv"]
    assert qkv["agreement"] == 0.4 and qkv["checks"]["agreement"]["passed"] is False
    assert qkv["failures"] == ["agreement 0.400 (least 0.900)", "margin 0.400 (least 0.950)"]
    assert report["groups"]["mlp.gate_proj"]["passed"] is True
    assert len(report["groups"]["mlp.gate_proj"]["modules"]) == 2
    adapter = tmp_path / "work" / "linear_attn.in_proj_qkv" / "adapter"
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["target_modules"] == qkv["target"] and isinstance(config["target_modules"], str)
    assert set(read_safetensors_header(adapter / "adapter_model.safetensors")) == {
        "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_A.weight",
        "base_model.model.model.layers.0.linear_attn.in_proj_qkv.lora_B.weight"}
    assert fake.exports[0] == (Path("work/linear_attn.in_proj_qkv/adapter"),
                               Path("work/linear_attn.in_proj_qkv/adapter-f16.gguf"))


def test_run_fidelity_refuses_a_group_without_modules_no_positions_and_an_aim_under_the_bound(tmp_path,
                                                                                               monkeypatch):
    monkeypatch.chdir(tmp_path)

    class Empty(FakeBackends):
        def modules(self, target) -> list[Module]:
            return []

    with pytest.raises(ValueError, match="no module"):
        run_fidelity(("q_proj",), SEQUENCES, backends_of(Empty(tmp_path, changed={})), Path("w"), r=2,
                     alpha=4, seed=0, base_model="m")
    fake = FakeBackends(tmp_path, changed={"q_proj": 8})
    with pytest.raises(ValueError, match="no position"):
        run_fidelity(("q_proj",), [Sequence(ids=[1, 2], start=2, positions=0)], backends_of(fake), Path("w"),
                     r=2, alpha=4, seed=0, base_model="m")
    with pytest.raises(ValueError, match="aim"):
        run_fidelity(("q_proj",), SEQUENCES, backends_of(fake), Path("w"), r=2, alpha=4, seed=0,
                     base_model="m", aim=0.25)
    assert [c for c in fake.calls if c[0] != "modules"] == []  # refused before any prediction


# ---- the llama-server backend ---------------------------------------------------------------------------


def test_the_llama_backend_asks_one_greedy_token_per_position_on_one_server(tmp_path):
    sequences = [Sequence(ids=list(range(100, 110)), start=7, positions=3),
                 Sequence(ids=list(range(200, 206)), start=4, positions=2)]
    script = [Scripted(("a",), ids=(11,)), Scripted(("b",), ids=(12,)), Scripted(("c",), ids=(13,)),
              Scripted(("d",), ids=(14,)), Scripted(("e",), ids=(15,))]
    adapter = {"id": 0, "path": "work/q_proj/adapter-f16.gguf", "scale": 1.0}
    fake = FakeLlama(script, adapters=[adapter])
    served = ServedLlama(fake)
    backend = llama_top1_backend(Path("llama-server.exe"), Path("models/base.gguf"), port=8091,
                                 popen=served.popen, client=served.client, log=tmp_path / "s6.server.log")
    predictions = backend(Path("work/q_proj/adapter-f16.gguf"), sequences)
    assert predictions == [[11, 12, 13], [14, 15]]
    (cmd,) = served.started
    assert cmd[:3] == ["llama-server.exe", "-m", "models/base.gguf"] and "--lora" in cmd
    assert cmd[cmd.index("--lora") + 1] == "work/q_proj/adapter-f16.gguf" and cmd[-2:] == ["-np", "1"]
    assert served.processes[0].terminated  # the server is stopped once the predictions are made
    assert (tmp_path / "s6.server.log").read_text(encoding="utf-8").startswith("main: server is listening")
    assert [r["prompt"] for r in fake.requests] == [list(range(100, 107)), list(range(100, 108)),
                                                    list(range(100, 109)), list(range(200, 204)),
                                                    list(range(200, 205))]
    assert all(r["max_tokens"] == 1 and r["temperature"] == 0.0 and r["lora"] == [{"id": 0, "scale": 1.0}]
               and r["cache_prompt"] is False and r["logprobs"] == 1 for r in fake.requests)
    # every request asks for the ids of the tokens it generates (llm.local, The generated tokens)
    asked = {"return_tokens": True, "verbose": True, "response_fields": ["tokens", "stop_type"]}
    assert all({k: r[k] for k in ASKED} == asked for r in fake.requests)
    # without an adapter: a bare server, scale 0 is not sent (there is no adapter to scale)
    fake = FakeLlama([Scripted(("x",), ids=(1,))] * 5)
    served = ServedLlama(fake)
    backend = llama_top1_backend(Path("llama-server.exe"), Path("models/base.gguf"), port=8091,
                                 popen=served.popen, client=served.client)
    assert backend(None, sequences) == [[1, 1, 1], [1, 1]]
    assert "--lora" not in served.started[-1] and all("lora" not in r for r in fake.requests)
    # a server that sends no list of the tokens it generated cannot be measured
    fake = FakeLlama([Scripted(("x",), ids=(1,))] * 5, return_tokens=False)
    served = ServedLlama(fake)
    backend = llama_top1_backend(Path("llama-server.exe"), Path("models/base.gguf"), port=8091,
                                 popen=served.popen, client=served.client)
    with pytest.raises(RuntimeError, match="generated"):
        backend(None, sequences)


ASKED = ("return_tokens", "verbose", "response_fields")
HAN = "中".encode()  # three bytes: a token holding the first two leaves the text inside the character


def test_the_llama_backend_reads_a_top1_the_logprob_list_leaves_out_from_the_ids_the_server_generated(
        tmp_path):
    # b11349 lists no token after which the text ends inside a UTF-8 character (llm.local, The generated
    # tokens): a one-token completion whose token is the first part of a character comes back with an empty
    # logprob list. The generated ids hold it all the same, and so does the prediction
    sequences = [Sequence(ids=list(range(100, 106)), start=4, positions=2)]
    script = [Scripted((HAN[:2],), "length", ids=(77,)), Scripted(("x",), "length", ids=(78,))]
    fake = FakeLlama(script)
    assert script[0].listed() == [] and script[1].listed()  # the server lists the first token nowhere
    served = ServedLlama(fake)
    backend = llama_top1_backend(Path("llama-server.exe"), Path("models/base.gguf"), port=8091,
                                 popen=served.popen, client=served.client)
    assert backend(None, sequences) == [[77, 78]]
    # a server whose answer omits the listed ids still has the generated ones (the listed id only checks)
    fake = FakeLlama([Scripted(("x",), "length", ids=(5,)), Scripted((HAN[:2],), "length", ids=(6,))],
                     omit_ids=True)
    served = ServedLlama(fake)
    backend = llama_top1_backend(Path("llama-server.exe"), Path("models/base.gguf"), port=8091,
                                 popen=served.popen, client=served.client)
    assert backend(None, sequences) == [[5, 6]]


class Altered(FakeLlama):
    """A FakeLlama whose answers are changed by alter after they are made: lists that disagree."""

    def __init__(self, script: list[Scripted], alter) -> None:
        super().__init__(script)
        self.alter = alter

    def handle(self, request):
        response = super().handle(request)
        if request.url.path != "/v1/completions" or response.status_code != 200:
            return response
        body = json.loads(response.content)
        self.alter(body)
        return httpx.Response(200, json=body)


def other_generated(body: dict) -> None:
    body["__verbose"]["tokens"] = [t + 1 for t in body["__verbose"]["tokens"]]


def two_generated(body: dict) -> None:
    body["__verbose"]["tokens"] = body["__verbose"]["tokens"] * 2
    body["usage"]["completion_tokens"] = 2


def counted_twice(body: dict) -> None:
    body["usage"]["completion_tokens"] = 2


@pytest.mark.parametrize(("alter", "message"), [(other_generated, "lists token 1"),
                                                (two_generated, "2 tokens"),
                                                (counted_twice, "completion tokens")])
def test_the_llama_backend_refuses_lists_that_disagree(alter, message):
    sequences = [Sequence(ids=list(range(100, 106)), start=4, positions=1)]
    served = ServedLlama(Altered([Scripted(("x",), "length", ids=(1,))], alter))
    backend = llama_top1_backend(Path("llama-server.exe"), Path("models/base.gguf"), port=8091,
                                 popen=served.popen, client=served.client)
    with pytest.raises(RuntimeError, match=message):
        backend(None, sequences)


# ---- the training-stack backend's model: built as scripts/train.py builds it, through the family ---------


class Built:
    """A model from_pretrained returned: eval() is all the backend asks of it before predicting."""

    def eval(self) -> Built:
        return self


def test_the_hf_backend_builds_the_folders_family_model_in_nf4_as_training_does(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")  # the backend imports torch when it loads the model
    folder = tmp_path / "model"
    install_fake_family(monkeypatch, folder)  # the folder resolves to a fake family by its files alone
    transformers, calls = sys.modules["transformers"], []

    def from_pretrained(path, **kwargs):
        calls.append((path, kwargs))
        return Built()

    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=from_pretrained)
    transformers.BitsAndBytesConfig = lambda **kwargs: ("nf4", kwargs)
    assert fidelity.hf_top1_backend(folder)(None, []) == []  # no sequence: the model is only loaded
    (path, kwargs), = calls
    assert path == str(folder) and kwargs["trust_remote_code"] is False and kwargs["dtype"] is torch.bfloat16
    assert (kwargs["attn_implementation"], kwargs["device_map"]) == ("sdpa", {"": 0})
    assert kwargs["quantization_config"] == ("nf4", {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                                     "bnb_4bit_use_double_quant": True,
                                                     "bnb_4bit_compute_dtype": torch.bfloat16})
    with pytest.raises(ValueError, match="trust_remote_code"):  # the family decides it, a caller cannot
        fidelity.hf_top1_backend(folder, trust_remote_code=True)(None, [])
    # so does the attention: a family whose model has eager attention only is built with it
    eager = tmp_path / "eager"
    install_fake_family(monkeypatch, eager, attn_implementation="eager")
    transformers = sys.modules["transformers"]
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=from_pretrained)
    transformers.BitsAndBytesConfig = lambda **kwargs: ("nf4", kwargs)
    assert fidelity.hf_top1_backend(eager)(None, []) == []
    assert calls[-1][0] == str(eager) and calls[-1][1]["attn_implementation"] == "eager"


@pytest.mark.skipif(not (SPARK / "modeling_spark.py").is_file(), reason="local model files are not present")
def test_the_hf_backend_builds_spark_with_its_familys_patch_and_attention_set_on_the_vendor_class(
        monkeypatch):
    # from_pretrained builds on the meta device what the auto class resolves (no weights are read), with the
    # attention it is given, which the vendor class refuses unless it is eager (registry: spark-x2.5-4b)
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = transformers.AutoConfig.from_pretrained(SPARK, trust_remote_code=True)
    vendor = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], SPARK)
    restore_vendor(monkeypatch, vendor)
    built = []

    def from_pretrained(path, **kwargs):
        with torch.device("meta"):
            built.append(transformers.AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation=kwargs["attn_implementation"]))
        return built[-1]

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", lambda **kwargs: None)
    assert fidelity.hf_top1_backend(SPARK, trust_remote_code=True)(None, []) == []
    (model,) = built
    assert type(model) is vendor and model.lm_head.weight is model.model.embedding.weight
    assert model.config._attn_implementation == "eager"


def restore_vendor(monkeypatch, vendor: type) -> None:
    """What build_causal_lm sets on a Spark vendor class and its module (the family's class patches and
    module adapters), restored when the test ends."""
    from research_agent.family import registry

    spec = registry.SPARK_X25_4B
    for attribute in spec.model_patches:
        monkeypatch.setattr(vendor, attribute, getattr(vendor, attribute))
    module = sys.modules[vendor.__module__]
    for target in spec.module_adapters:
        owner, _, attribute = target.rpartition(".")
        holder = getattr(module, owner) if owner else module
        monkeypatch.setattr(holder, attribute, getattr(holder, attribute))


@pytest.mark.skipif(not (SPARK / "modeling_spark.py").is_file(), reason="local model files are not present")
def test_the_hf_backend_predicts_with_a_two_layer_spark_in_nf4_on_the_cpu(tmp_path, monkeypatch):
    # S6's training-stack side for Spark, the GPU placement aside: a two-layer Spark with random weights and
    # the real vocabulary (its folder holds the vendor files and the chat template, so the backend finds its
    # family from the files), built by the backend in NF4 through build_causal_lm, adapted as training adapts
    # it, predicts the top-1 of every compared position, with a synthetic LoRA on the fused attention
    # projection, every tensor of which the model holds with the file's values (the load check, and the
    # model's parameters read at that moment), and again without it (the adapter is unloaded)
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("bitsandbytes")
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    from research_agent.family import registry
    from research_agent.family.core import Mode, load_family

    spec = registry.SPARK_X25_4B
    folder = tmp_path / "Spark-X2.5-4B-s6"
    config = transformers.AutoConfig.from_pretrained(SPARK, trust_remote_code=True)
    config.num_hidden_layers, config.layer_types = 2, ["sliding_attention", "full_attention"]
    config.hidden_size, config.intermediate_size, config.sliding_window = 64, 128, 4
    config.num_attention_heads, config.num_key_value_heads, config.head_dim = 4, 2, 16
    restore_vendor(monkeypatch, get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], SPARK))
    monkeypatch.setattr(get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], SPARK),
                        "_tied_weights_keys", dict(spec.model_patches["_tied_weights_keys"]))
    torch.manual_seed(0)
    transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="eager"
                                                  ).to(torch.bfloat16).save_pretrained(folder)
    for name in (*spec.remote_files, "chat_template.jinja"):
        (folder / name).write_bytes((SPARK / name).read_bytes())
    reference = config.auto_map["AutoModelForCausalLM"]
    restore_vendor(monkeypatch, get_class_from_dynamic_module(reference, folder))
    on_the_cpu(monkeypatch, transformers)
    checked = spy_on_the_load_check(monkeypatch, torch)
    family = load_family(SPARK)
    ids = family.encode_prompt(family.prompt(family.to_template_messages(ITEMS[1]["messages"]), None,
                                             Mode.NO_THINK)) + family.encode_text("See you!")
    sequences = [Sequence(ids=ids, start=len(ids) - 3, positions=3)]
    predict = fidelity.hf_top1_backend(folder)
    base = predict(None, sequences)
    assert len(base) == 1 and len(base[0]) == 3 and all(0 <= t < config.vocab_size for t in base[0])
    modules = base_modules(folder, ("q_k_v_proj",))
    assert [m.name for m in modules] == [f"model.layers.{i}.self_attn.q_k_v_proj" for i in (0, 1)]
    loras = {m.name: synthetic_lora(m, r=4, alpha=8, seed=0, weight_norm=weight_norm(folder, m))
             for m in modules}
    adapter = write_adapter(tmp_path / "adapter", modules, loras, r=4, alpha=8, targets=["q_k_v_proj"],
                            base_model=str(folder))
    with_lora = predict(adapter, sequences)
    assert checked == [(4, True, {"tensors": 4, "modules": 2})]  # both modules' A and B, as the file has them
    assert len(with_lora[0]) == 3 and predict(None, sequences) == base
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SPARK.glob("*.py")} == dict(
        spec.remote_files)

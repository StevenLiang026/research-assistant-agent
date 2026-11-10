"""The families' LoRA targets against the real module names of their architectures (docs/m7-计划.md §1,
registry: lora_targets as module names or one PEFT regex, taken from the real module names): each model
folder of models/ that is present is instantiated on the meta device (transformers' AutoModelForCausalLM
from its configuration, no weights, no GPU), and the family's targets must select exactly the Linear modules
the registry means, by name and by count per layer, and leave out exactly the ones it names as left out:
lm_head everywhere; Qwen3.5's in_proj_a and in_proj_b (the decay and gate vectors of its linear attention);
Spark's head-wise gate g_proj.

Spark's vendor code (models/Spark-X2.5-4B/modeling_spark.py, pinned by the family) declares its tied weights
as a list, "_tied_weights_keys = ["lm_head.weight"]", which transformers 5.18 reads as a mapping in
PreTrainedModel.post_init (get_expanded_tied_weights_keys calls .keys() on it) when tie_word_embeddings is
on, as config.json has it: as the vendor wrote it, no construction of the model succeeds, from_pretrained
included. The family declares the mapping as a model patch (research_agent.family.core: Model patches), and
with it set on the class, as build_causal_lm sets it before from_pretrained, the auto class builds the model
with lm_head tied to the embedding, Spark2_5Model.embedding, while the vendor files stay as pinned; with the
family's attention, eager, since the class declares no SDPA support and is refused "sdpa". Its forward pass
runs once the family's module adapters are set too (research_agent.family.compat; tests/family/test_compat.py
and tests/train/test_train_cli.py run it). Every model here is built through the auto class with the family's
patches and attention, as training builds it.

The tests skip when torch or transformers is not installed, and a folder when it is absent.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

from research_agent.family import core
from research_agent.family.core import load_family

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models"
LLAMA = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
SPARK = "Spark-X2.5-4B"
FOLDERS = ("Qwen3-1.7B", "Qwen3-4B", "Qwen3-4B-Instruct-2507", "Qwen3-4B-Thinking-2507", "Qwen3.5-4B",
           "granite-4.2-3b", "Nanbeige4-3B-Thinking-2511", "Nanbeige4.1-3B", "MiniCPM5-2B", SPARK,
           "LFM2.5-2.6B")


def present(folder: str) -> bool:
    return all((MODELS / folder / name).is_file() for name in ("config.json", "tokenizer_config.json"))


def params() -> list:
    return [pytest.param(folder, marks=pytest.mark.skipif(not present(folder),
                                                          reason="local model files are not present"))
            for folder in FOLDERS]


def expected(name: str, config: dict) -> tuple[Counter, set[str]]:
    """The Linear leaf names the family's targets must select, with how many of each the model holds, and
    the leaf names they must leave out."""
    text = config.get("text_config", config)  # Qwen3.5 keeps the text model's shape under text_config
    layers = text["num_hidden_layers"]
    if name == "qwen3.5-4b":
        attention = text["layer_types"].count("full_attention")
        linear = layers - attention
        return (Counter({**{n: attention for n in ("q_proj", "k_proj", "v_proj", "o_proj")},
                         **{n: linear for n in ("in_proj_qkv", "in_proj_z", "out_proj")},
                         **{n: layers for n in ("gate_proj", "up_proj", "down_proj")}}),
                {"lm_head", "in_proj_a", "in_proj_b"})
    if name == "spark-x2.5-4b":
        return (Counter({n: layers for n in ("q_k_v_proj", "out_proj", "gate_proj", "up_proj", "down_proj")}),
                {"lm_head", "g_proj"})
    if name == "lfm2.5-2.6b":
        attention = text["layer_types"].count("full_attention")
        return (Counter({**{n: attention for n in ("q_proj", "k_proj", "v_proj")},
                         "out_proj": layers, "in_proj": layers - attention,
                         **{n: layers for n in ("w1", "w2", "w3")}}),
                {"lm_head"})
    return Counter({n: layers for n in LLAMA}), {"lm_head"}


def vendor_class(transformers, folder: Path, config, monkeypatch):
    """The class the auto class resolves for a folder with vendor code, its patched attributes and its
    module's adapted names restored when the test ends (patch_model_class sets them on the class and the
    module themselves)."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    spec = load_family(folder).spec
    model_class = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], folder)
    for attribute in spec.model_patches:
        monkeypatch.setattr(model_class, attribute, getattr(model_class, attribute))
    module = sys.modules[model_class.__module__]
    for target in spec.module_adapters:
        owner, _, attribute = target.rpartition(".")
        holder = getattr(module, owner) if owner else module
        monkeypatch.setattr(holder, attribute, getattr(holder, attribute))
    return model_class


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("folder", params())
def test_the_familys_lora_targets_select_the_registrys_modules_of_the_real_architecture(folder, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    family = load_family(MODELS / folder)  # the vendor code, when any, is checked against its pins first
    spec = family.spec
    remote = spec.trust_remote_code
    config = transformers.AutoConfig.from_pretrained(MODELS / folder, trust_remote_code=remote)
    if spec.model_patches:  # Spark: its own class with the family's patches, as training builds it
        vendor_class(transformers, MODELS / folder, config, monkeypatch)
        core.patch_model_class(spec, MODELS / folder)
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=remote,
                                                              attn_implementation=spec.attn_implementation)
    linear = [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]
    targets = spec.lora_targets
    if isinstance(targets, str):
        pattern = re.compile(targets)
        matched = [name for name in linear if pattern.fullmatch(name)]
    else:
        matched = [name for name in linear if name.rsplit(".", 1)[-1] in targets]
    wanted, left_out = expected(spec.name, json.loads((MODELS / folder / "config.json").read_text("utf-8")))
    assert Counter(name.rsplit(".", 1)[-1] for name in matched) == wanted
    assert {name.rsplit(".", 1)[-1] for name in linear if name not in set(matched)} == left_out
    assert all(re.match(r"model\.(language_model\.)?layers\.\d+\.", name) for name in matched)
    assert sum(wanted.values()) == len(matched)


@pytest.mark.skipif(not present(SPARK), reason="local model files are not present")
def test_the_pinned_spark_code_is_built_with_the_familys_patch_its_head_tied_to_the_embedding(monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    folder = MODELS / SPARK
    spec = load_family(folder).spec  # the code is the pinned code
    config = transformers.AutoConfig.from_pretrained(folder, trust_remote_code=True)
    assert config.tie_word_embeddings is True
    model_class = vendor_class(transformers, folder, config, monkeypatch)
    # as the vendor wrote it, the installed transformers builds no model of it
    assert model_class._tied_weights_keys == ["lm_head.weight"]
    with torch.device("meta"), pytest.raises(AttributeError, match="'list' object has no attribute 'keys'"):
        transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    # with the family's patch set on the class the auto class resolves, it builds, on the meta device, with
    # the family's attention: the class runs its own eager attention and refuses sdpa
    assert core.patch_model_class(spec, folder) is model_class
    with torch.device("meta"), pytest.raises(ValueError, match="scaled_dot_product_attention"):
        transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True,
                                                      attn_implementation="sdpa")
    assert spec.attn_implementation == "eager"
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True,
                                                              attn_implementation=spec.attn_implementation)
    assert type(model) is model_class and model.config._attn_implementation == "eager"
    names = {name for name, _ in model.named_parameters(remove_duplicate=False)}
    assert {"lm_head.weight", "model.embedding.weight"} <= names  # the patch names real parameters
    assert model.all_tied_weights_keys == {"lm_head.weight": "model.embedding.weight"}
    assert model.lm_head.weight is model.get_input_embeddings().weight is model.model.embedding.weight
    assert {p.device.type for p in model.parameters()} == {"meta"}  # no weight was loaded anywhere
    # the vendor files are the pinned ones, as they were
    assert {p.name: sha256_of(p) for p in folder.glob("*.py")} == dict(spec.remote_files)


@pytest.mark.skipif(not present(SPARK), reason="local model files are not present")
def test_build_causal_lm_hands_from_pretrained_the_patched_spark_class(monkeypatch):
    # from_pretrained stands in for the weights' load: it builds what the auto class resolves, on the meta
    # device, so the model build_causal_lm returns is the one training would build, without its weights
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    folder = MODELS / SPARK
    spec = load_family(folder).spec
    config = transformers.AutoConfig.from_pretrained(folder, trust_remote_code=True)
    model_class = vendor_class(transformers, folder, config, monkeypatch)
    calls = []

    def from_pretrained(path, **kwargs):
        calls.append((path, kwargs))
        remote = kwargs["trust_remote_code"]
        with torch.device("meta"):
            return transformers.AutoModelForCausalLM.from_config(
                transformers.AutoConfig.from_pretrained(path, trust_remote_code=remote),
                trust_remote_code=remote, attn_implementation=kwargs["attn_implementation"])

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    model = core.build_causal_lm(spec, folder, dtype=torch.bfloat16)
    assert type(model) is model_class
    assert calls == [(folder, {"trust_remote_code": True, "dtype": torch.bfloat16,
                               "attn_implementation": "eager"})]
    assert model.lm_head.weight is model.model.embedding.weight

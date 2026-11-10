import ast
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from family_stub import family_hook, loaded_fake_family
from support.fakes import SAMPLE_MESSAGES
from support.family_fakes import FakeFamilyTokenizer, model_folder
from test_train_family_samples import REASONING, sidecar

from research_agent.agent.prompts import PROMPT_HASH, PROMPT_VERSION
from research_agent.agent.types import ANSWERED, Transcript
from research_agent.family import registry
from research_agent.tools import schemas
from research_agent.train.samples import Dropped, completion_mask

REPO = Path(__file__).resolve().parents[2]
TRAINING_STACK = {"torch", "trl", "transformers", "peft", "datasets", "bitsandbytes", "accelerate"}
PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
BIG = 10**6  # a --max-length no fake-family sample reaches (the fake reads every character as a token)


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = load_script("train")
export = load_script("export_adapter")
serve = load_script("serve_local")


# scripts/train.py


def test_train_defaults_reproduce_the_s4_settings_with_the_plan_accumulation():
    args = train.build_parser().parse_args(["a.jsonl", "b.jsonl", "--out", "runs/train/x"])
    assert args.data == ["a.jsonl", "b.jsonl"] and args.model == "models/Qwen3-1.7B"
    assert (args.max_length, args.epochs, args.lr) == (schemas.CONTEXT_TOKENS, 2, 1e-4)
    assert (args.r, args.alpha, args.seed) == (16, 32, 42)
    assert args.max_length == schemas.CONTEXT_TOKENS
    assert args.qlora is False and args.max_steps is None and args.grad_accum == 16 and args.batch_size == 1
    assert args.mask_check == 5  # docs/plan.md: five decoded samples
    assert (args.family, args.reasoning, args.no_think, args.stop_at_step) == (None, [], False, None)
    assert (args.sample_keys, args.smoke_longest, args.target_modules) == (None, False, None)
    setup = train.settings(args, registry.QWEN3_HYBRID)
    assert setup["mode"] == "bf16-lora" and setup["model"] == "models/Qwen3-1.7B"
    assert setup["lora"] == {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "task_type": "CAUSAL_LM",
                             "target_modules": PROJECTIONS}  # the family's targets: today's list
    sft = setup["sft"]
    assert sft["output_dir"] == str(Path("runs/train/x") / "trainer")
    assert (sft["num_train_epochs"], sft["max_steps"], sft["learning_rate"]) == (2, -1, 1e-4)
    assert (sft["per_device_train_batch_size"], sft["gradient_accumulation_steps"]) == (1, 16)
    assert (sft["lr_scheduler_type"], sft["warmup_steps"], sft["bf16"]) == ("cosine", 0.03, True)
    assert sft["gradient_checkpointing"] and sft["gradient_checkpointing_kwargs"] == {"use_reentrant": False}
    assert sft["completion_only_loss"] and not sft["packing"] and sft["max_length"] == schemas.CONTEXT_TOKENS
    assert (sft["optim"], sft["seed"], sft["report_to"], sft["save_strategy"]) == ("adamw_torch", 42, "none",
                                                                                   "epoch")
    assert sft["loss_type"] == "chunked_nll" and "sft.loss_type" in train.PINNED  # G2's resolved loss, pinned
    # nothing else about the sampling is set: the permutation stays FT2's and G2's
    assert not {"train_sampling_strategy", "data_seed", "shuffle_dataset", "group_by_length"} & sft.keys()
    # every family trains its own targets (names, or one regex as a string); the Qwen3 trio, today's seven
    for family in registry.FAMILIES:
        assert train.settings(args, family)["lora"]["target_modules"] == train.lora_targets(family)
    for family in (registry.QWEN3_HYBRID, registry.QWEN3_INSTRUCT_2507, registry.QWEN3_THINKING_2507):
        assert train.settings(args, family)["lora"]["target_modules"] == PROJECTIONS


def test_train_smoke_and_qlora_flags():
    args = train.build_parser().parse_args(["a.jsonl", "--out", "o", "--max-steps", "10", "--qlora",
                                            "--save-steps", "50", "--max-length", "1536", "--r", "8",
                                            "--alpha", "16", "--epochs", "1"])
    setup = train.settings(args, registry.QWEN3_HYBRID)
    assert setup["mode"] == "qlora-4bit" and setup["sft"]["optim"] == "paged_adamw_8bit"
    assert setup["sft"]["max_steps"] == 10 and setup["sft"]["num_train_epochs"] == 1
    assert setup["sft"]["save_strategy"] == "steps" and setup["sft"]["save_steps"] == 50
    assert setup["sft"]["max_length"] == 1536
    assert setup["lora"]["r"] == 8 and setup["lora"]["lora_alpha"] == 16


def test_train_m7_flags_are_parsed_and_only_the_targets_and_the_smoke_order_reach_the_settings():
    args = train.build_parser().parse_args([
        "a.jsonl", "--out", "o", "--family", "qwen3-thinking-2507", "--reasoning", "t.jsonl", "r.jsonl",
        "--no-think", "--stop-at-step", "274", "--sample-keys", "keys.jsonl", "--smoke-longest",
        "--target-modules", "q_proj", "v_proj"])
    assert args.family == "qwen3-thinking-2507" and args.reasoning == ["t.jsonl", "r.jsonl"]
    assert args.no_think and args.stop_at_step == 274 and args.sample_keys == "keys.jsonl"
    assert args.smoke_longest and args.target_modules == ["q_proj", "v_proj"]
    setup = train.settings(args, registry.QWEN3_THINKING_2507)
    assert setup["lora"]["target_modules"] == ["q_proj", "v_proj"]
    assert setup["sft"]["train_sampling_strategy"] == "sequential"  # transformers' own sequential sampler
    plain = train.build_parser().parse_args(["a.jsonl", "--out", "o"])
    assert "train_sampling_strategy" not in train.settings(plain, registry.QWEN3_THINKING_2507)["sft"]
    for bad in ("0", "-3", "x", "2.5"):
        with pytest.raises(SystemExit):
            train.build_parser().parse_args(["a.jsonl", "--stop-at-step", bad])
    assert train.lora_targets(registry.QWEN3_HYBRID) == PROJECTIONS
    fields = {**registry.QWEN3_HYBRID.__dict__, "lora_targets": r".*\.(q|v)_proj"}
    assert train.lora_targets(type(registry.QWEN3_HYBRID)(**fields)) == r".*\.(q|v)_proj"  # a PEFT regex


def test_train_imports_the_training_stack_only_inside_main():
    tree = ast.parse((REPO / "scripts" / "train.py").read_text(encoding="utf-8"))
    top_level = {alias.name.split(".")[0] for node in tree.body if isinstance(node, ast.Import)
                 for alias in node.names}
    top_level |= {node.module.split(".")[0] for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert not top_level & TRAINING_STACK
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    inside = {n.module.split(".")[0] for n in ast.walk(main) if isinstance(n, ast.ImportFrom)}
    inside |= {a.name for n in ast.walk(main) if isinstance(n, ast.Import) for a in n.names}
    assert {"torch", "trl", "peft", "transformers", "datasets"} <= inside


def test_train_expands_globs_and_assembles_pre_tokenized_samples_from_every_record(tmp_path):
    t = Transcript(request="r", today="2026-10-04", status=ANSWERED, final_text=None,
                   messages=SAMPLE_MESSAGES,
                   steps=[{"index": 1, "attempts": [{"action": None}]}, {"index": 2, "attempts": [{}]}],
                   schema_hash=schemas.SCHEMA_HASH, prompt_version=PROMPT_VERSION, prompt_hash=PROMPT_HASH)
    (tmp_path / "a.jsonl").write_text(json.dumps({"messages": SAMPLE_MESSAGES}) + "\n" +
                                      t.to_json().replace("\n", "") + "\n", encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"messages": SAMPLE_MESSAGES[:3], "tools": []}),
                                     encoding="utf-8")
    files = train.expand_data_files([str(tmp_path / "*.jsonl"), str(tmp_path / "b.json")])
    assert [f.name for f in files] == ["a.jsonl", "b.json"]
    family = loaded_fake_family(tmp_path / "model")
    samples, dropped, per_file = train.assemble(files, family, BIG)
    assert len(samples) == 5 and dropped == Dropped()
    assert [(f["records"], f["samples"]) for f in per_file] == [(2, 4), (1, 1)]
    assert per_file[0]["path"] == str(files[0]) and per_file[0]["dropped"] == {"over_length": 0,
                                                                                "after_repair": 0}
    assert per_file[0]["modes"] == {"think": 0, "no_think": 4, "think_fallback": 0}
    assert per_file[0]["refused"] == {"template": 0, "boundary": 0, "bos": 0}
    assert [s["key"] for s in samples] == [("a:0", 0), ("a:0", 1), ("a:1", 0), ("a:1", 1), ("b:0", 0)]
    assert [s["file"] for s in samples] == [str(files[0])] * 4 + [str(files[1])]
    rows = train.pre_tokenized(samples)
    assert [set(r) for r in rows] == [{"input_ids", "completion_mask"}] * 5  # nothing for TRL to tokenize
    assert rows[0]["input_ids"] == samples[0]["input_ids"]
    assert rows[0]["completion_mask"] == completion_mask(samples[0]) and sum(rows[0]["completion_mask"]) > 0
    few, dropped, _ = train.assemble(files, family, samples[0]["tokens"])
    assert len(few) == 3 and dropped == Dropped(over_length=2)  # the longer second step of each history
    with pytest.raises(FileNotFoundError, match="no data file"):
        train.expand_data_files([str(tmp_path / "*.missing")])


def test_train_assemble_over_a_tokenizer_alone_gives_the_string_path_samples(tmp_path):
    # the form research_agent.data.assemble's manifest check and tests/data/test_benchmark_overlap.py
    # compare the trainer's counts with: render_sample's strings and token counts, no ids and no modes
    from tokenizer_fake import FakeTokenizer

    (tmp_path / "a.jsonl").write_text(json.dumps({"messages": SAMPLE_MESSAGES}) + "\n", encoding="utf-8")
    samples, dropped, per_file = train.assemble([tmp_path / "a.jsonl"], FakeTokenizer(), 3072)
    assert dropped == Dropped()
    assert [set(s) for s in samples] == [{"prompt", "completion", "tokens", "file"}] * 2
    assert per_file[0]["modes"] == {"think": 0, "no_think": 2, "think_fallback": 0}
    data = train.data_summary(samples, dropped, per_file, None)
    assert data["tokens"]["total"] == sum(s["tokens"] for s in samples) and data["samples"] == 2
    few, dropped, _ = train.assemble([tmp_path / "a.jsonl"], FakeTokenizer(), samples[0]["tokens"])
    assert len(few) == 1 and dropped == Dropped(over_length=1)


def overlap_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Two data files and a benchmark-overlap report that lists only the first (record 1 overlapping)."""
    listed, unlisted = tmp_path / "listed.jsonl", tmp_path / "unlisted.jsonl"
    for path in (listed, unlisted):
        path.write_text("".join(json.dumps({"messages": SAMPLE_MESSAGES}) + "\n" for _ in range(2)),
                        encoding="utf-8")
    report = tmp_path / "overlap.json"
    entry = {"path": str(listed), "sha256": hashlib.sha256(listed.read_bytes()).hexdigest(), "records": 2,
             "overlapping": 1, "records_overlapping": [{"index": 1}]}
    report.write_text(json.dumps({"files": [entry]}), encoding="utf-8")
    return listed, unlisted, report


def test_train_stops_on_a_data_file_the_overlap_report_does_not_list(tmp_path, capsys):
    listed, unlisted, report = overlap_files(tmp_path)
    skips, overlap = train.overlap_gate(str(report), [listed], check=True)
    assert skips == {listed.resolve(): {1}} and overlap["checked"] is True and overlap["unchecked"] == []
    with pytest.raises(train.OverlapGateError, match=r"unlisted\.jsonl.*check_benchmark_overlap\.py"):
        train.overlap_gate(str(report), [listed, unlisted], check=True)
    with pytest.raises(train.OverlapGateError, match=r"listed\.jsonl.*check_benchmark_overlap\.py"):
        train.overlap_gate(str(tmp_path / "no-report.json"), [listed], check=True)
    listed.write_text(listed.read_text(encoding="utf-8") * 2, encoding="utf-8")  # no longer the hash checked
    with pytest.raises(ValueError, match="check_benchmark_overlap"):
        train.overlap_gate(str(report), [listed], check=True)
    out, model = tmp_path / "run", model_folder(tmp_path / "model")
    assert train.main([str(unlisted), "--out", str(out), "--overlap-report", str(report), "--model",
                       str(model)], load_family=family_hook()) == 4
    assert "check_benchmark_overlap.py" in capsys.readouterr().err and not (out / "manifest.json").exists()


def test_train_with_no_overlap_check_uses_an_unlisted_file_and_records_it_in_the_manifest(tmp_path):
    listed, unlisted, report = overlap_files(tmp_path)
    args = train.build_parser().parse_args([str(unlisted), "--overlap-report", str(report)])
    assert args.no_overlap_check is False
    args = train.build_parser().parse_args([str(listed), str(unlisted), "--overlap-report", str(report),
                                            "--no-overlap-check"])
    assert args.no_overlap_check is True
    skips, overlap = train.overlap_gate(args.overlap_report, [listed, unlisted], check=False)
    assert skips == {listed.resolve(): {1}}  # the files the report does list still skip their records
    assert overlap["checked"] is False and overlap["unchecked"] == [str(unlisted)]
    family = loaded_fake_family(tmp_path / "model")
    samples, dropped, per_file = train.assemble([listed, unlisted], family, BIG, skips=skips)
    data = train.data_summary(samples, dropped, per_file, overlap)
    assert data["benchmark_overlap"]["checked"] is False
    benchmark = data["benchmark_overlap"]
    assert benchmark["unchecked"] == [str(unlisted)] and benchmark["skipped"] == 1
    assert data["modes"] == {"think": 0, "no_think": 6, "think_fallback": 0}
    assert data["reasoning"] is None and data["sample_keys"] is None
    _, none = train.overlap_gate(str(tmp_path / "no-report.json"), [unlisted], check=False)
    assert none["checked"] is False and none["report"] is None and none["unchecked"] == [str(unlisted)]


class CharDecoder:
    """decode: each id is the character with that code point."""

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class SpacedDecoder(CharDecoder):
    """A legacy SentencePiece tokenizer's decode (Nanbeige4-3B-Thinking-2511's): a space after every "<"."""

    def decode(self, ids):
        return super().decode(ids).replace("<", "< ")


MASK_SAMPLE = {"input_ids": [ord(c) for c in "abcX<Z"], "completion_start": 3, "completion": "X<Z"}


def labelled(ids: list[int], labels: list[int]) -> dict:
    return {"input_ids": ids, "labels": labels}


def test_train_mask_check_compares_the_loss_bearing_positions_and_their_ids_with_the_completions():
    ids = MASK_SAMPLE["input_ids"]
    good = labelled(ids, [-100, -100, -100, *ids[3:]])
    older = {"input_ids": ids, "completion_mask": [0, 0, 0, 1, 1, 1]}  # TRL before "labels"
    checks = train.mask_check([good, older], CharDecoder(), [MASK_SAMPLE, MASK_SAMPLE],
                              [(0, "a.jsonl"), (1, "b.jsonl")])
    assert [c["ok"] for c in checks] == [True, True] and [c["file"] for c in checks] == ["a.jsonl", "b.jsonl"]
    assert checks[0]["trained_tokens"] == 3 and checks[0]["total_tokens"] == 6
    assert checks[0]["completion_start"] == 3 and checks[0]["head"] == "X<Z" and checks[0]["text_equal"]
    # the decoded text is a diagnostic: a tokenizer that decodes the ids to other text passes on its ids
    (spaced,) = train.mask_check([good], SpacedDecoder(), [MASK_SAMPLE], [(0, "a.jsonl")])
    assert spaced["ok"] and not spaced["text_equal"] and spaced["head"] == "X< Z"


@pytest.mark.parametrize("example", [
    labelled(MASK_SAMPLE["input_ids"], [-100, -100, 99, 88, 60, 90]),  # a prompt token carries loss
    labelled(MASK_SAMPLE["input_ids"], [-100, -100, -100, -100, 60, 90]),  # a completion token carries none
    labelled(MASK_SAMPLE["input_ids"], [-100, -100, -100, 88, 60, -100]),  # nor does the terminator
    labelled(MASK_SAMPLE["input_ids"], [-100, -100, -100, 88, 61, 90]),  # a label is not its input id
    labelled(MASK_SAMPLE["input_ids"], [-100] * 6),  # no loss at all
    labelled(MASK_SAMPLE["input_ids"] + [2], [-100, -100, -100, 88, 60, 90, 2]),  # the trainer appended an id
    labelled(MASK_SAMPLE["input_ids"][1:], [-100, -100, 88, 60, 90]),  # ...or lost one: the positions move
    labelled([ord(c) for c in "abcY<Z"], [-100, -100, -100, 89, 60, 90]),  # another sample's ids
    labelled([ord(c) for c in "abdX<Z"], [-100, -100, -100, 88, 60, 90]),  # ...that differ in the prompt
    labelled(MASK_SAMPLE["input_ids"], [-100, -100, -100, 88, 60]),  # fewer labels than ids
    labelled(MASK_SAMPLE["input_ids"], [-100, -100, -100, 88, 60, 90, -100]),  # more labels than ids
    {"input_ids": MASK_SAMPLE["input_ids"], "completion_mask": [0, 0, 1, 1, 1, 1]},  # the older mask, wrong
    {"input_ids": MASK_SAMPLE["input_ids"], "completion_mask": [0, 0, 0, 0, 1, 1]},  # ...or too short
], ids=["prompt-token", "completion-token", "terminator", "label-id", "no-loss", "appended", "shifted",
        "other-ids", "other-prompt-ids", "short-labels", "long-labels", "older-mask", "older-mask-short"])
def test_train_mask_check_fails_on_any_wrong_mask_whatever_the_decoded_text(example):
    (check,) = train.mask_check([example], CharDecoder(), [MASK_SAMPLE], [(0, "a.jsonl")])
    assert not check["ok"]


def test_train_mask_check_holds_the_whole_example_to_the_samples_ids_the_prompt_included():
    # an example whose prompt is not the sample's, its length, completion and loss positions all right,
    # fails on its ids alone: the model would learn the completion after another prompt
    other_prompt = labelled([ord(c) for c in "abdX<Z"], [-100, -100, -100, 88, 60, 90])
    (check,) = train.mask_check([other_prompt], CharDecoder(), [MASK_SAMPLE], [(0, "a.jsonl")])
    assert not check["ok"] and not check["ids_equal"]
    assert check["positions_equal"] and check["labels_equal"] and check["text_equal"]


# The model a run trains is built through its family (research_agent.family.core.build_causal_lm): the
# loading settings of S4, the family's trust_remote_code and attention, and its model patches and module
# adapters set on the vendor code first.

BF16 = object()
SPARK = REPO / "models" / "Spark-X2.5-4B"
spark_present = pytest.mark.skipif(not (SPARK / "modeling_spark.py").is_file(),
                                   reason="local model files are not present")


def test_train_builds_the_model_through_the_family_with_the_s4_loading_settings(tmp_path, monkeypatch):
    calls = []

    def from_pretrained(path, **kwargs):
        calls.append((path, kwargs))
        return "model"

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=from_pretrained)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    family, quant = loaded_fake_family(tmp_path / "model"), object()
    assert train.load_model("models/x", family, torch=SimpleNamespace(bfloat16=BF16), quant=quant) == "model"
    assert calls == [("models/x", {"trust_remote_code": False, "dtype": BF16, "quantization_config": quant,
                                   "attn_implementation": "sdpa", "device_map": {"": 0}})]
    # the attention is the family's: a model whose code has eager attention only is built with it
    eager = loaded_fake_family(tmp_path / "eager", attn_implementation="eager")
    train.load_model("models/y", eager, torch=SimpleNamespace(bfloat16=BF16), quant=None)
    assert calls[1] == ("models/y", {"trust_remote_code": False, "dtype": BF16, "quantization_config": None,
                                     "attn_implementation": "eager", "device_map": {"": 0}})


def vendor_class(folder: Path, transformers, monkeypatch) -> type:
    """The class the auto class resolves for a Spark folder, with what build_causal_lm sets on it and on its
    module (the family's class patches and module adapters) restored when the test ends."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    spec = registry.SPARK_X25_4B
    config = transformers.AutoConfig.from_pretrained(folder, trust_remote_code=True)
    vendor = get_class_from_dynamic_module(config.auto_map["AutoModelForCausalLM"], folder)
    for attribute in spec.model_patches:
        monkeypatch.setattr(vendor, attribute, getattr(vendor, attribute))
    module = sys.modules[vendor.__module__]
    for target in spec.module_adapters:
        owner, _, attribute = target.rpartition(".")
        holder = getattr(module, owner) if owner else module
        monkeypatch.setattr(holder, attribute, getattr(holder, attribute))
    return vendor


@spark_present
def test_train_builds_spark_with_its_familys_patch_and_attention_set_on_the_vendor_class(monkeypatch):
    # from_pretrained builds on the meta device what the auto class resolves (no weights are read), with the
    # attention it is given: the pinned vendor class, which builds only with the family's patch and refuses
    # sdpa; its module adapted first (family.compat)
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from research_agent.family.core import load_family

    family = load_family(SPARK)
    config = transformers.AutoConfig.from_pretrained(SPARK, trust_remote_code=True)
    vendor = vendor_class(SPARK, transformers, monkeypatch)
    module = sys.modules[vendor.__module__]

    def adapted() -> dict:  # the adapter each declared target of the module carries, None for none
        found = {}
        for target in family.spec.module_adapters:
            owner, _, attribute = target.rpartition(".")
            holder = getattr(module, owner) if owner else module
            found[target] = getattr(getattr(holder, attribute), "__family_adapter__", None)
        return found

    assert set(adapted().values()) == {None}  # as the vendor wrote it
    seen = []

    def from_pretrained(path, **kwargs):
        seen.append((kwargs, adapted()))
        with torch.device("meta"):
            return transformers.AutoModelForCausalLM.from_config(
                config, trust_remote_code=True, attn_implementation=kwargs["attn_implementation"])

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", from_pretrained)
    model = train.load_model(str(SPARK), family, torch=torch, quant=None)
    assert type(model) is vendor and model.lm_head.weight is model.model.embedding.weight
    ((kwargs, at_build),) = seen
    assert kwargs["trust_remote_code"] is True and kwargs["device_map"] == {"": 0}
    assert kwargs["attn_implementation"] == "eager" == model.config._attn_implementation
    assert at_build == dict(family.spec.module_adapters)  # every adapter in place before the model was built


def tiny_spark(folder: Path, transformers, torch, monkeypatch) -> Path:
    """A Spark-X2.5-4B folder of two layers (one sliding-window, one full attention; a window of 4) with
    random weights and the real vocabulary, at folder: the real configuration shrunk, saved by the pinned
    vendor class built with the family's patch, and the vendor files copied as they are, so the family's pins
    hold for the folder."""
    from research_agent.family import core

    config = transformers.AutoConfig.from_pretrained(SPARK, trust_remote_code=True)
    assert config.vocab_size == 131072  # the real vocabulary: the ids below are the real tokenizer's
    config.num_hidden_layers, config.layer_types = 2, ["sliding_attention", "full_attention"]
    config.hidden_size, config.intermediate_size, config.sliding_window = 64, 128, 4
    config.num_attention_heads, config.num_key_value_heads, config.head_dim = 4, 2, 16
    vendor_class(SPARK, transformers, monkeypatch)
    core.patch_model_class(registry.SPARK_X25_4B, SPARK)
    torch.manual_seed(0)
    transformers.AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="eager"
                                                  ).to(torch.bfloat16).save_pretrained(folder)
    for name in registry.SPARK_X25_4B.remote_files:
        shutil.copyfile(SPARK / name, folder / name)
    return folder


SPARK_CONTEXT = [{"role": "system", "content": "You are a research assistant."},
                 {"role": "user", "content": "找 KV cache 的论文"}]
SPARK_TARGET = {"role": "assistant", "content": "I found two papers on KV cache compression."}


def spark_sample(family) -> dict:
    """A THINK training sample of the family, as a run's data holds it (pre-tokenized)."""
    from research_agent.train.samples import render_family_sample

    sample = render_family_sample(family, SPARK_CONTEXT, SPARK_TARGET, tools=None,
                                  reasoning="The user wants papers on KV cache, so I list two.")
    assert sample["mode"] == "think" and sample["input_ids"][0] == family.bos_id
    return sample


def vendor_masks(torch, length: int, window: int, dtype) -> dict:
    """The masks the vendor's eager layers ran under in transformers 4.57, made here: 0 where a query attends
    a key, the dtype's least value where not; causal, and for the sliding layer also within the window."""
    i = torch.arange(length)
    causal = i[None, :] <= i[:, None]
    windowed = causal & (i[:, None] - i[None, :] < window)
    least = torch.finfo(dtype).min
    return {kind: torch.where(allowed, 0.0, least).to(dtype)[None, None]
            for kind, allowed in (("full_attention", causal), ("sliding_attention", windowed))}


@spark_present
def test_spark_built_as_a_run_builds_it_runs_its_forward_pass_under_the_vendors_masks(tmp_path, monkeypatch):
    # registry, spark-x2.5-4b: the vendor's forward pass calls transformers' mask functions with 4.57's
    # keywords, which 5.18 does not take (tests/family/test_compat.py shows it); the family adapts the two
    # calls at load. A two-layer Spark with random weights and the real vocabulary, built through
    # train.load_model with the arguments a run passes (the GPU placement aside: this runs on the CPU), runs
    # its forward pass over a training sample, and its logits are the vendor layers' under the masks 4.57 made
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from research_agent.family.core import load_family

    folder = tiny_spark(tmp_path / "Spark-X2.5-4B-tiny", transformers, torch, monkeypatch)
    vendor = vendor_class(folder, transformers, monkeypatch)
    real, seen = transformers.AutoModelForCausalLM.from_pretrained, []

    def on_the_cpu(path, **kwargs):
        seen.append(dict(kwargs))
        del kwargs["device_map"]  # the run's GPU placement
        return real(path, **kwargs)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", on_the_cpu)
    family = load_family(SPARK)
    model = train.load_model(str(folder), family, torch=torch, quant=None).eval()
    assert seen == [{"trust_remote_code": True, "attn_implementation": "eager", "dtype": torch.bfloat16,
                     "quantization_config": None, "device_map": {"": 0}}]
    assert type(model) is vendor and model.config._attn_implementation == "eager"
    assert model.lm_head.weight is model.model.embedding.weight and model.dtype == torch.bfloat16
    ids = torch.tensor([spark_sample(family)["input_ids"]])
    assert ids.shape[1] > 4 * 4  # longer than the window several times over
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
        masks = vendor_masks(torch, ids.shape[1], 4, torch.bfloat16)
        expected = model(input_ids=ids, attention_mask=masks).logits
    assert torch.equal(logits, expected) and torch.isfinite(logits).all()
    # the vendor files are the pinned ones, as they were
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SPARK.glob("*.py")} == dict(
        registry.SPARK_X25_4B.remote_files)


@spark_present
def test_spark_trains_under_qlora_as_a_run_wraps_it_forward_and_backward(tmp_path, monkeypatch):
    # --qlora: the base in NF4 with main()'s quantization settings (bitsandbytes, here on the CPU), the
    # run's LoRA settings on the family's targets, and the wrapping TRL 1.14.1's SFTTrainer gives a quantized
    # model (get_peft_model, input gradients for gradient checkpointing, the adapter in bf16, its chunked NLL
    # loss over the backbone), gradient checkpointing as the run sets it. The vendor decoder layer would cast
    # its input to a quantized weight's packed uint8 storage; the family adapts it (family.compat:
    # spark_projection_dtype), and the step runs: the chunked loss is the model's own at the start (LoRA's B
    # is zero), and every LoRA matrix of both layers gets a finite gradient, B's not zero
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    pytest.importorskip("bitsandbytes")
    sft = pytest.importorskip("trl.trainer.sft_trainer")
    from research_agent.family.core import load_family

    folder = tiny_spark(tmp_path / "Spark-X2.5-4B-tiny", transformers, torch, monkeypatch)
    vendor = vendor_class(folder, transformers, monkeypatch)
    real = transformers.AutoModelForCausalLM.from_pretrained

    def on_the_cpu(path, **kwargs):
        assert kwargs.pop("device_map") == {"": 0}  # the run's GPU placement
        return real(path, device_map={"": "cpu"}, **kwargs)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", on_the_cpu)
    family = load_family(SPARK)
    setup = train.settings(train.build_parser().parse_args(["a.jsonl", "--qlora", "--out", str(tmp_path)]),
                           family.spec)
    assert setup["mode"] == "qlora-4bit" and setup["lora"]["target_modules"] == list(family.spec.lora_targets)
    quant = transformers.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                            bnb_4bit_use_double_quant=True,
                                            bnb_4bit_compute_dtype=torch.bfloat16)
    model = train.load_model(str(folder), family, torch=torch, quant=quant)
    assert type(model) is vendor and type(model.model.layers[0].mlp.gate_proj).__name__ == "Linear4bit"
    sample = spark_sample(family)
    ids = torch.tensor([sample["input_ids"]])
    labels = ids.clone()
    labels[0, :sample["completion_start"]] = -100  # the completion alone carries loss
    with torch.no_grad():
        own = model.eval()(input_ids=ids, labels=labels).loss
    wrapped = peft.get_peft_model(model, peft.LoraConfig(**setup["lora"]))
    wrapped.enable_input_require_grads()
    for parameter in wrapped.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(torch.bfloat16)
    sft._patch_chunked_ce_lm_head(wrapped.get_base_model(), chunk_size=sft._CHUNKED_LM_HEAD_CHUNK_SIZE)
    wrapped.gradient_checkpointing_enable(gradient_checkpointing_kwargs=setup["sft"]["gradient_checkpointing_kwargs"])
    lora = {name: p for name, p in wrapped.named_parameters() if p.requires_grad}
    targets = {name.split(".lora_")[0] for name in lora}
    assert len(targets) == 2 * len(family.spec.lora_targets) and len(lora) == 2 * len(targets)
    attention = torch.ones_like(ids)
    with torch.no_grad():
        chunked = wrapped.eval()(input_ids=ids, attention_mask=attention, labels=labels).loss
    assert torch.isfinite(own) and torch.allclose(chunked.float(), own.float(), atol=1e-3)
    loss = wrapped.train()(input_ids=ids, attention_mask=attention, labels=labels).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in lora.values())
    assert all(float(p.grad.abs().sum()) > 0 for name, p in lora.items() if ".lora_B." in name)
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in SPARK.glob("*.py")} == dict(
        registry.SPARK_X25_4B.remote_files)


# Hardening for the unattended runs: save cadence, resume, per-file mask check, dry run, logging, exit codes.

STACK_PROBE = """
import importlib.util, json, sys
sys.path.insert(0, {tests!r})
sys.path.insert(0, {train_tests!r})
from family_stub import family_hook
spec = importlib.util.spec_from_file_location("script_train", {script!r})
train = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train)
codes = [train.main(argv, load_family=family_hook()) for argv in {argvs!r}]
loaded = sorted({{m.split(".")[0] for m, v in sys.modules.items() if v is not None}} & {stack!r})
print(json.dumps({{"codes": codes, "loaded": loaded}}))
"""


def probe_training_stack(argvs: list[list[str]]) -> dict:
    """Run train.main on each argv in a fresh interpreter: the exit codes and the training-stack modules
    that ended up imported (a pytest process may already hold torch, so this cannot be checked in-process)."""
    code = STACK_PROBE.format(tests=str(REPO / "tests"), train_tests=str(REPO / "tests" / "train"),
                              script=str(REPO / "scripts" / "train.py"), argvs=argvs, stack=TRAINING_STACK)
    done = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                          encoding="utf-8", timeout=300, check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_train_save_steps_below_one_are_a_ratio_zero_keeps_epochs_and_integers_stay_step_counts():
    def sft(*flags: str) -> dict:
        args = train.build_parser().parse_args(["a.jsonl", "--out", "o", *flags])
        return train.settings(args, registry.QWEN3_HYBRID)["sft"]

    ratio = sft("--save-steps", "0.125")
    assert (ratio["save_strategy"], ratio["save_steps"]) == ("steps", 0.125)
    epoch = sft("--save-steps", "0")
    assert epoch["save_strategy"] == "epoch" and sft()["save_strategy"] == "epoch"
    steps = sft("--save-steps", "200")
    assert (steps["save_strategy"], steps["save_steps"]) == ("steps", 200)
    assert type(steps["save_steps"]) is int
    for setting in (ratio, epoch, steps):
        assert "save_total_limit" in setting and setting["save_total_limit"] is None  # keep every checkpoint
    for bad in ("2.5", "-1", "x"):  # transformers refuses a non-integer above 1 only once training starts
        with pytest.raises(SystemExit):
            train.build_parser().parse_args(["a.jsonl", "--save-steps", bad])


def test_train_resolves_the_save_cadence_and_the_optimizer_steps_as_transformers_does():
    # set_initial_training_values: ceil(batches / accumulation) per epoch, ceil(epochs x that) in all
    assert train.total_optimizer_steps(27100, batch_size=1, grad_accum=16, epochs=2, max_steps=None) == 3388
    assert train.total_optimizer_steps(33, batch_size=2, grad_accum=4, epochs=1.5, max_steps=None) == 8
    assert train.total_optimizer_steps(27100, batch_size=1, grad_accum=16, epochs=2, max_steps=10) == 10
    # TrainerState.compute_steps: a ratio below 1 becomes ceil(max_steps x ratio); DefaultFlowCallback also
    # saves at the last step under the steps strategy, so ceil(max_steps / every) checkpoints in all
    assert train.save_cadence(0.125, 1694, steps_per_epoch=847) == {
        "save_steps": 0.125, "strategy": "steps", "every_steps": 212, "checkpoints": 8}
    assert train.save_cadence(0, 1694, steps_per_epoch=847) == {
        "save_steps": 0, "strategy": "epoch", "every_steps": 847, "checkpoints": 2}
    assert train.save_cadence(500, 1694, steps_per_epoch=847) == {
        "save_steps": 500, "strategy": "steps", "every_steps": 500, "checkpoints": 4}


def resume_setup(tmp_path: Path, *flags: str) -> tuple[list[str], Path, Path]:
    """A data file listed in an overlap report, a fake model folder, and the argv of a run on them into
    out/."""
    listed, _, report = overlap_files(tmp_path)
    model, out = model_folder(tmp_path / "model"), tmp_path / "run"
    argv = [str(listed), "--out", str(out), "--overlap-report", str(report), "--model", str(model), *flags]
    return argv, listed, out


def write_reference_manifest(argv: list[str], out: Path, **extra) -> Path:
    """manifest.json as the first segment of the run writes it, for the fields a resume compares; extra
    keys go into its config."""
    args = train.build_parser().parse_args(argv)
    files = train.expand_data_files(args.data)
    family = loaded_fake_family(args.model)
    _, gate = train.overlap_gate(args.overlap_report, files, check=not args.no_overlap_check)
    benchmark = {"report": gate["report"], "report_sha256": gate["report_sha256"], "skipped": 0,
                 "checked": gate["checked"], "unchecked": gate["unchecked"]}  # as data_summary records it
    data = {"benchmark_overlap": benchmark,
            "reasoning": {"files": train.sidecar_files(args.reasoning), "ignored": args.no_think},
            "sample_keys": None if args.sample_keys is None else {
                "sha256": train.sha256_of(args.sample_keys)}}
    config = {**train.settings(args, family.spec), "family": family.record(), "args": vars(args),
              "data": data, **extra}
    out.mkdir(parents=True, exist_ok=True)
    manifest = train.run_manifest(config, files, git_dir=None, gpu_name=lambda: None)
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, default=str), encoding="utf-8")
    return path


def test_train_resume_is_refused_without_a_manifest(tmp_path, capsys):
    argv, _, _ = resume_setup(tmp_path)
    assert train.main([*argv, "--resume"], load_family=family_hook()) == 5
    assert "manifest.json" in capsys.readouterr().err


def test_train_resume_is_refused_when_a_data_file_changed(tmp_path, capsys):
    argv, listed, out = resume_setup(tmp_path)
    write_reference_manifest(argv, out)
    args = train.build_parser().parse_args([*argv, "--resume"])
    family = loaded_fake_family(args.model)
    assert train.resume_differences(args, train.expand_data_files(args.data), family) == []
    listed.write_text(listed.read_text(encoding="utf-8") * 2, encoding="utf-8")
    assert train.main([*argv, "--resume", "--no-overlap-check"], load_family=family_hook()) == 5
    err = capsys.readouterr().err
    assert "data_files" in err and listed.name in err


def test_train_resume_is_refused_when_the_benchmark_overlap_gate_changed(tmp_path, capsys):
    argv, _, out = resume_setup(tmp_path)
    write_reference_manifest(argv, out)
    hook = family_hook()
    assert train.main([*argv, "--resume", "--no-overlap-check"], load_family=hook) == 5  # now unchecked
    assert "benchmark_overlap.checked" in capsys.readouterr().err
    report = Path(argv[argv.index("--overlap-report") + 1])
    rewritten = {**json.loads(report.read_text(encoding="utf-8")), "note": "rerun"}  # still covers the file
    report.write_text(json.dumps(rewritten), encoding="utf-8")
    assert train.main([*argv, "--resume"], load_family=hook) == 5
    assert "benchmark_overlap.report_sha256" in capsys.readouterr().err


def test_train_resume_is_refused_when_a_setting_changed(tmp_path, capsys):
    argv, _, out = resume_setup(tmp_path)
    write_reference_manifest(argv, out)
    hook = family_hook()
    assert train.main([*argv, "--resume", "--lr", "2e-4", "--save-steps", "0.25"], load_family=hook) == 5
    err = capsys.readouterr().err
    assert "sft.learning_rate" in err and "sft.save_steps" in err and "sft.save_strategy" in err
    assert train.main([*argv, "--resume", "--target-modules", "q_proj"], load_family=hook) == 5
    assert "lora.target_modules" in capsys.readouterr().err
    assert train.main([*argv, "--resume", "--smoke-longest"], load_family=hook) == 5
    assert "sft.train_sampling_strategy" in capsys.readouterr().err


def test_train_resume_is_refused_when_the_family_the_sidecars_or_the_sample_keys_changed(tmp_path, capsys):
    sidecar(tmp_path / "teacher.jsonl", [("listed:0", 1, SAMPLE_MESSAGES[4], REASONING)])
    keys = tmp_path / "keys.jsonl"
    keys.write_text(json.dumps({"id": "listed:0", "k": 0}) + "\n", encoding="utf-8")
    argv, _, out = resume_setup(tmp_path, "--reasoning", str(tmp_path / "teacher.jsonl"))
    write_reference_manifest(argv, out)
    hook = family_hook()
    args = train.build_parser().parse_args([*argv, "--resume"])
    files = train.expand_data_files(args.data)
    assert train.resume_differences(args, files, loaded_fake_family(args.model)) == []
    assert train.main([*argv, "--resume", "--no-think"], load_family=hook) == 5
    assert "reasoning.ignored: False -> True" in capsys.readouterr().err
    assert train.main([*argv[:-2], "--resume"], load_family=hook) == 5  # the sidecar left out
    assert "reasoning[" in capsys.readouterr().err
    assert train.main([*argv, "--resume", "--sample-keys", str(keys)], load_family=hook) == 5
    assert "sample_keys.sha256: None ->" in capsys.readouterr().err
    other = family_hook("only")  # another template: another family record
    assert train.main([*argv, "--resume"], load_family=other) == 5
    err = capsys.readouterr().err
    assert "family.name: 'fake-hybrid' -> 'fake-only'" in err and "family.template_sha256" in err


def test_train_resume_compares_a_manifest_before_the_loss_pin_on_the_loss_its_run_resolved(tmp_path, capsys):
    argv, _, out = resume_setup(tmp_path)
    path = write_reference_manifest(argv, out, sft_resolved={"loss_type": "chunked_nll"})
    manifest = json.loads(path.read_text(encoding="utf-8"))
    del manifest["config"]["sft"]["loss_type"]  # as G2's and FT2's manifests have it: unset, resolved by TRL
    del manifest["config"]["family"]  # and no family record
    path.write_text(json.dumps(manifest), encoding="utf-8")
    args = train.build_parser().parse_args([*argv, "--resume"])
    family = loaded_fake_family(args.model)
    assert train.resume_differences(args, train.expand_data_files(args.data), family) == []
    manifest["config"]["sft_resolved"]["loss_type"] = "nll"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert train.main([*argv, "--resume"], load_family=family_hook()) == 5
    assert "sft.loss_type: 'nll' -> 'chunked_nll'" in capsys.readouterr().err


def test_train_resume_without_a_checkpoint_is_refused_and_the_last_one_is_found_with_its_step(tmp_path):
    trainer_dir = tmp_path / "trainer"
    with pytest.raises(train.ResumeRefused, match="no checkpoint"):
        train.resume_checkpoint(tmp_path, find_last=lambda folder: None)  # the folder does not even exist
    trainer_dir.mkdir()
    with pytest.raises(train.ResumeRefused, match="no checkpoint"):
        train.resume_checkpoint(tmp_path, find_last=lambda folder: None)
    last = trainer_dir / "checkpoint-424"
    last.mkdir()
    (last / "trainer_state.json").write_text(json.dumps({"global_step": 424}), encoding="utf-8")
    seen = []
    found = train.resume_checkpoint(tmp_path, find_last=lambda folder: seen.append(folder) or str(last))
    assert found == (str(last), 424) and seen == [str(trainer_dir)]
    assert train.resume_manifest_path(tmp_path) == tmp_path / "manifest-resume-1.json"
    (tmp_path / "manifest-resume-1.json").write_text("{}", encoding="utf-8")
    (tmp_path / "manifest-resume-2.json").write_text("{}", encoding="utf-8")
    assert train.resume_manifest_path(tmp_path) == tmp_path / "manifest-resume-3.json"


def test_train_resume_checks_happen_before_the_training_stack_is_imported(tmp_path):
    tree = ast.parse((REPO / "scripts" / "train.py").read_text(encoding="utf-8"))
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    torch_line = min(n.lineno for n in ast.walk(main) if isinstance(n, ast.Import)
                     and any(a.name == "torch" for a in n.names))
    calls = {n.func.id: n.lineno for n in ast.walk(main) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)}
    assert calls["resume_differences"] < torch_line and calls["dry_run"] < torch_line
    argv, _, out = resume_setup(tmp_path)
    write_reference_manifest(argv, out)
    changed = [*argv, "--resume", "--epochs", "3"]
    result = probe_training_stack([[*argv[:2], str(tmp_path / "elsewhere"), *argv[3:], "--resume"], changed])
    assert result == {"codes": [5, 5], "loaded": []}


class FakeState:
    def __init__(self, global_step=0, max_steps=0):
        self.global_step, self.max_steps, self.log_history = global_step, max_steps, []


class FakeControl:
    should_save = False
    should_training_stop = False


class FakeCuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def reset_peak_memory_stats():
        pass

    @staticmethod
    def max_memory_allocated():
        return 0

    @staticmethod
    def max_memory_reserved():
        return 0


class FakeTorch:
    cuda = FakeCuda()
    __version__ = "fake"


class FakeTrainer:
    """SFTTrainer stand-in: train() records its resume_from_checkpoint, logs through the callbacks the way the
    Trainer does (TRL's num_tokens restarting at 0 on resume), honours a callback's should_save and
    should_training_stop at the end of a step as the Trainer's loop does, ends at end_step, and closes the
    epoch as the Trainer does after the loop, a stop included: on_epoch_end, where the default flow callback
    asks for a save under the epoch strategy (the trainer's default here), then the save unless a callback
    withdrew the request."""

    def __init__(self, start_step: int, end_step: int, tokens_per_step: int = 100):
        self.state = FakeState(start_step, end_step)
        self.end_step, self.tokens_per_step, self.callbacks, self.calls = end_step, tokens_per_step, [], []
        self.model = self
        self.saved = None
        self.checkpoints: list[int] = []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def save_pretrained(self, path):
        self.saved = Path(path)

    def train(self, resume_from_checkpoint=None):
        self.calls.append(resume_from_checkpoint)
        for cb in self.callbacks:
            cb.on_train_begin(None, self.state, None)
        start = self.state.global_step
        total = 0.0
        while self.state.global_step < self.end_step:
            self.state.global_step += 1
            total += 1.0
            control = FakeControl()
            for cb in self.callbacks:
                if hasattr(cb, "on_step_end"):
                    cb.on_step_end(None, self.state, control)
            if control.should_save:
                self.checkpoints.append(self.state.global_step)
            logs = {"loss": 1.0, "num_tokens": (self.state.global_step - start) * self.tokens_per_step}
            for cb in self.callbacks:
                cb.on_log(None, self.state, None, logs=logs)
            if control.should_training_stop:
                break
        control = FakeControl()
        control.should_save = True  # DefaultFlowCallback.on_epoch_end under save_strategy "epoch"
        for cb in self.callbacks:
            if hasattr(cb, "on_epoch_end"):
                cb.on_epoch_end(None, self.state, control)
        if control.should_save:
            self.checkpoints.append(self.state.global_step)

        class Result:
            # as transformers computes it: the losses summed in this call over the global step, which also
            # counts the steps before a resumed checkpoint (trainer.py: _total_loss_scalar / global step)
            training_loss = total / max(self.state.global_step, 0.001)

        return Result()


class Passive:
    """A callback that only logs (the FakeTrainer calls on_train_begin and on_log on every callback)."""

    def on_train_begin(self, args, state, control, **kwargs):
        pass

    def on_log(self, args, state, control, logs=None, **kwargs):
        pass


def report_inputs(tmp_path: Path) -> dict:
    args = train.build_parser().parse_args(["a.jsonl", "--out", str(tmp_path), "--logging-steps", "1"])
    data = {"tokens": {"min": 10, "mean": 20, "max": 30, "total": 200},
            "modes": {"think": 2, "no_think": 8, "think_fallback": 1}}
    return {"args": args, "setup": train.settings(args, registry.QWEN3_HYBRID), "samples": 10,
            "dropped": Dropped(), "data": data, "checks": [{"ok": True}]}


def test_train_resume_passes_the_checkpoint_to_the_trainer_and_records_the_segments(tmp_path):
    clock = iter(range(0, 10_000, 2))
    logger = train.jsonl_logger(tmp_path / "log.jsonl", FakeTorch(), base=object, clock=lambda: next(clock))
    first = FakeTrainer(0, 4)
    train.train_and_report(first, FakeTorch(), tmp_path, callback=logger, checkpoint=None, checkpoint_step=0,
                           **report_inputs(tmp_path))
    assert first.calls == [None]
    second = FakeTrainer(3, 6)  # the first segment saved checkpoint-3 and went on to step 4 before dying
    report = train.train_and_report(second, FakeTorch(), tmp_path, callback=logger,
                                    checkpoint=str(tmp_path / "trainer" / "checkpoint-3"), checkpoint_step=3,
                                    **report_inputs(tmp_path))
    assert second.calls == [str(tmp_path / "trainer" / "checkpoint-3")]
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    events = [r for r in rows if "event" in r]
    assert events[0]["event"] == "start"
    assert events[1] == {"event": "resume", "from_checkpoint": str(tmp_path / "trainer" / "checkpoint-3"),
                         "global_step": 3}
    assert [(s["start_step"], s["end_step"], s["tokens"]) for s in report["segments"]] == [(0, 3, 300),
                                                                                           (3, 6, 300)]
    assert report["trained_tokens"] == 600 and report["steps"] == 6 and report["stopped_at_step"] is None
    assert report["modes"] == {"think": 2, "no_think": 8, "think_fallback": 1}
    written = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert written["segments"] == report["segments"]
    for key in ("mode", "sequences", "seconds", "tokens_per_s", "peak_alloc_gib", "final_loss", "mask_ok",
                "adapter", "tokens_per_s_source", "stopped_at_step"):
        assert key in report


def test_train_stop_at_step_saves_a_checkpoint_at_that_step_and_stops_leaving_the_schedule_whole(tmp_path):
    stopper = train.stop_at_step_callback(3, base=Passive)
    assert stopper.step == 3 and not stopper.stopped
    control = FakeControl()
    state = FakeState(global_step=2, max_steps=10)
    assert stopper.on_step_end(None, state, control) is control
    assert not (control.should_save or control.should_training_stop or stopper.stopped)
    state.global_step = 3
    stopper.on_step_end(None, state, control)
    assert control.should_save and control.should_training_stop and stopper.stopped
    control.should_save = True  # the epoch save the default flow callback asks for after the stop
    assert stopper.on_epoch_end(None, state, control) is control and not control.should_save  # withdrawn
    trainer = FakeTrainer(0, 10)  # max_steps 10: the schedule's horizon stays the full run
    clock = iter(range(100))
    logger = train.jsonl_logger(tmp_path / "log.jsonl", FakeTorch(), base=object, clock=lambda: next(clock))
    report = train.train_and_report(trainer, FakeTorch(), tmp_path, callback=logger, checkpoint=None,
                                    checkpoint_step=0, stopper=train.stop_at_step_callback(3, base=Passive),
                                    **report_inputs(tmp_path))
    assert trainer.checkpoints == [3]  # once: the epoch end after the stop does not save the step again
    assert trainer.state.global_step == 3 and trainer.state.max_steps == 10
    assert report["steps"] == 3 and report["stopped_at_step"] == 3 and trainer.saved == tmp_path / "adapter"
    written = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert written["stopped_at_step"] == 3 and written["segments"] == [
        {"start_step": 0, "end_step": 3, "tokens": 300, "seconds": written["segments"][0]["seconds"]}]
    untouched = FakeTrainer(0, 2)  # a run that ends before the step: the stopper never fires
    (tmp_path / "short").mkdir()
    report = train.train_and_report(untouched, FakeTorch(), tmp_path / "short", callback=None,
                                    checkpoint=None, checkpoint_step=0,
                                    stopper=train.stop_at_step_callback(3, base=Passive),
                                    **report_inputs(tmp_path / "short"))
    assert report["stopped_at_step"] is None and untouched.checkpoints == [2]  # the epoch's own save stays


def test_train_final_loss_after_a_resume_is_the_mean_loss_of_the_segment_not_diluted_by_the_resumed_steps(
        tmp_path):
    clock = iter(range(0, 10_000, 2))
    logger = train.jsonl_logger(tmp_path / "log.jsonl", FakeTorch(), base=object, clock=lambda: next(clock))
    resumed = FakeTrainer(3, 6)  # every step's loss is 1.0; transformers would report 3 x 1.0 / 6 = 0.5
    report = train.train_and_report(resumed, FakeTorch(), tmp_path, callback=logger,
                                    checkpoint=str(tmp_path / "trainer" / "checkpoint-3"), checkpoint_step=3,
                                    **report_inputs(tmp_path))
    assert report["final_loss"] == 1.0
    fresh = FakeTrainer(0, 4)
    (tmp_path / "fresh").mkdir()
    report = train.train_and_report(fresh, FakeTorch(), tmp_path / "fresh", callback=None, checkpoint=None,
                                    checkpoint_step=0, **report_inputs(tmp_path))
    assert report["final_loss"] == 1.0


def test_train_segments_count_only_the_steps_up_to_the_checkpoint_a_later_segment_resumed_from():
    rows = [{"event": "start"}, {"step": 1, "elapsed_s": 5.0, "num_tokens": 10},  # a stale earlier attempt
            {"event": "start"},
            {"step": 10, "elapsed_s": 10.0, "num_tokens": 100},
            {"step": 20, "elapsed_s": 20.0, "num_tokens": 200},
            {"step": 30, "elapsed_s": 30.0, "num_tokens": 300},
            {"event": "resume", "from_checkpoint": "c-20", "global_step": 20},
            {"event": "resume", "from_checkpoint": "c-20", "global_step": 20},  # died again before a log row
            {"step": 30, "elapsed_s": 12.0, "num_tokens": 110},
            {"step": 40, "elapsed_s": 24.0, "num_tokens": 220}]
    segments = train.segments_from_log(rows, final_step=40, final_seconds=25.0)
    assert segments == [{"start_step": 0, "end_step": 20, "tokens": 200, "seconds": 20.0},
                        {"start_step": 20, "end_step": 20, "tokens": 0, "seconds": 0.0},
                        {"start_step": 20, "end_step": 40, "tokens": 220, "seconds": 25.0}]


def test_train_log_rows_carry_the_max_steps_and_an_eta_from_the_mean_step_time_so_far(tmp_path):
    times = iter([100.0, 150.0, 160.0])
    logger = train.jsonl_logger(tmp_path / "log.jsonl", FakeTorch(), base=object, clock=lambda: next(times))
    state = FakeState(global_step=100, max_steps=300)  # a resumed segment starts at step 100
    logger.on_train_begin(None, state, None)
    state.global_step = 110
    logger.on_log(None, state, None, logs={"loss": 1.0})
    state.global_step = 112
    logger.on_log(None, state, None, logs={"loss": 0.9})
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(r["step"], r["max_steps"], r["eta_s"]) for r in rows] == [(110, 300, 950.0), (112, 300, 940.0)]


def two_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Two data files (two and one histories) and an overlap report listing both, overlapping nothing."""
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    first.write_text("".join(json.dumps({"messages": SAMPLE_MESSAGES}) + "\n" for _ in range(2)),
                     encoding="utf-8")
    second.write_text(json.dumps({"messages": SAMPLE_MESSAGES[:3], "tools": []}) + "\n", encoding="utf-8")
    report = tmp_path / "overlap.json"
    report.write_text(json.dumps({"files": [
        {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "records": 1, "overlapping": 0,
         "records_overlapping": []} for p in (first, second)]}), encoding="utf-8")
    return first, second, report


def test_train_mask_check_decodes_the_first_samples_of_every_data_file_in_dataset_order(tmp_path):
    first, second, _ = two_files(tmp_path)
    family = loaded_fake_family(tmp_path / "model")
    samples, _, per_file = train.assemble([first, second], family, BIG)
    assert [f["samples"] for f in per_file] == [4, 1]
    picks = train.mask_check_picks(samples, 2)
    assert picks == [(0, str(first)), (1, str(first)), (4, str(second))]
    reordered = train.longest_first(samples)  # the picks follow the dataset order, whatever it is
    assert [s["tokens"] for s in reordered] == sorted((s["tokens"] for s in samples), reverse=True)
    ordered = train.mask_check_picks(reordered, 1)
    assert [file for _, file in ordered] == [str(first), str(second)]
    assert all(reordered[i]["file"] == file and file not in {s["file"] for s in reordered[:i]}
               for i, file in ordered)  # each pick is its file's first dataset index
    mixed = [samples[0], samples[4], *samples[1:4]]  # a file's samples apart: dataset indices, not offsets
    assert train.mask_check_picks(mixed, 2) == [(0, str(first)), (1, str(second)), (2, str(first))]

    def example(sample):  # the tokenized sample with its prompt masked, as TRL prepares it
        ids, mask = sample["input_ids"], completion_mask(sample)
        return {"input_ids": ids, "labels": [t if m else -100 for t, m in zip(ids, mask, strict=True)]}

    checks = train.mask_check([example(s) for s in samples], family.tokenizer, samples, picks)
    assert [(c["index"], c["file"], c["ok"]) for c in checks] == [
        (0, str(first), True), (1, str(first), True), (4, str(second), True)]
    assert checks[0]["trained_tokens"] == len(samples[0]["input_ids"]) - samples[0]["completion_start"]


def test_train_sample_keys_restrict_and_order_the_samples_and_a_listed_key_without_a_sample_stops(tmp_path):
    first, second, _ = two_files(tmp_path)
    family = loaded_fake_family(tmp_path / "model")
    samples, _, _ = train.assemble([first, second], family, BIG)
    keys = tmp_path / "keys.jsonl"
    keys.write_text("".join(json.dumps(row) + "\n" for row in [
        {"id": "second:0", "k": 0}, {"id": "first:1", "k": 1, "extra": "a sidecar row may carry more"},
        {"id": "first:0", "k": 0}]) + "\n", encoding="utf-8")
    listed = train.load_sample_keys(keys)
    assert listed == [("second:0", 0), ("first:1", 1), ("first:0", 0)]
    selected = train.select_samples(samples, listed)
    assert [s["key"] for s in selected] == listed
    with pytest.raises(ValueError, match=r"1 key\(s\) without a sample.*\('first:0', 7\)"):
        train.select_samples(samples, [*listed, ("first:0", 7)])
    keys.write_text((json.dumps({"id": "first:0", "k": 0}) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="line 2 lists .* again"):
        train.load_sample_keys(keys)
    keys.write_text(json.dumps({"id": "first:0"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1 is not a sample key"):
        train.load_sample_keys(keys)
    keys.write_text("{oops\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1 is not JSON"):
        train.load_sample_keys(keys)
    sidecar(keys, [("first:1", 1, SAMPLE_MESSAGES[4], REASONING)])  # a sidecar file is a keys file
    assert train.load_sample_keys(keys) == [("first:1", 1)]


def test_train_dry_run_summarises_the_run_without_the_training_stack(tmp_path, capsys):
    first, second, report = two_files(tmp_path)
    model, out = model_folder(tmp_path / "model"), tmp_path / "run"
    argv = [str(first), str(second), "--out", str(out), "--overlap-report", str(report), "--model",
            str(model), "--dry-run", "--save-steps", "0.5", "--grad-accum", "2", "--mask-check", "1"]
    seen = []
    assert train.main(argv, load_family=family_hook(seen=seen)) == 0
    printed = capsys.readouterr().out
    assert seen == [(str(model), None)]  # the real model path argument and --family
    assert "records 2, samples 4, dropped over_length 0" in printed and "records 1, samples 1" in printed
    assert "think 0, no_think 4, think_fallback 0" in printed
    assert "total optimizer steps 6" in printed  # 5 samples, accumulation 2: 3 per epoch, 2 epochs
    assert "save every 3 steps (save_steps 0.5)" in printed
    assert "mask check: 2 samples" in printed and not out.exists()
    assert "modes: think 0, no_think 5, think_fallback 0" in printed
    assert "refused: template 0, boundary 0, bos 0" in printed
    assert "reasoning: 0 sidecar(s), 0 rows, no THINK sample; budget none" in printed
    summary = json.loads(printed.splitlines()[-1].removeprefix("summary: "))
    assert summary["samples"] == 5 and summary["modes"] == {"think": 0, "no_think": 5, "think_fallback": 0}
    assert summary["family"]["name"] == "fake-hybrid" and summary["total_optimizer_steps"] == 6
    assert probe_training_stack([argv]) == {"codes": [0], "loaded": []}
    assert train.main([*argv, "--family", "fake-only"], load_family=family_hook()) == 2  # must agree
    assert "fake-only" in capsys.readouterr().err  # the loader's FamilyError, verbatim


def test_train_dry_run_with_sidecars_counts_the_modes_and_picks_the_reasoning_budget(tmp_path, capsys):
    first, second, report = two_files(tmp_path)
    model, out = model_folder(tmp_path / "model"), tmp_path / "run"
    teacher = tmp_path / "teacher.jsonl"
    sidecar(teacher, [("first:0", 1, SAMPLE_MESSAGES[4], REASONING),
                      ("first:1", 0, SAMPLE_MESSAGES[2], "Go.")])
    argv = [str(first), str(second), "--out", str(out), "--overlap-report", str(report), "--model",
            str(model), "--dry-run", "--reasoning", str(teacher)]
    assert train.main(argv, load_family=family_hook()) == 0
    printed = capsys.readouterr().out
    assert "modes: think 2, no_think 3, think_fallback 0" in printed
    assert "reasoning: 1 sidecar(s), 2 rows; THINK samples 2" in printed and "budget R 1024" in printed
    summary = json.loads(printed.splitlines()[-1].removeprefix("summary: "))
    assert summary["reasoning"]["files"][0]["path"] == str(teacher) and summary["reasoning"]["rows"] == 2
    assert summary["reasoning"]["tokens"]["n"] == 2 and summary["reasoning"]["budget"] == 1024
    assert summary["reasoning"]["ignored"] is False
    # --no-think: the sidecar is recorded but every sample is NO_THINK
    assert train.main([*argv, "--no-think"], load_family=family_hook()) == 0
    printed = capsys.readouterr().out
    assert "modes: think 0, no_think 5, think_fallback 0" in printed and "ignored (--no-think)" in printed
    # a family that never thinks: NO_THINK throughout, the sidecar not consulted
    assert train.main(argv, load_family=family_hook("none")) == 0
    assert "modes: think 0, no_think 5, think_fallback 0" in capsys.readouterr().out
    # a fallback at a length between the THINK and the NO_THINK rendering of the longest THINK sample
    family = loaded_fake_family(model)
    from research_agent.data.reasoning import ReasoningIndex
    samples, _, _ = train.assemble([first], family, BIG, reasoning=ReasoningIndex([teacher]))
    longest = max((s for s in samples if s["mode"] == "think"), key=lambda s: s["tokens"])
    assert train.main([*argv, "--max-length", str(longest["tokens"] - 1)], load_family=family_hook()) == 0
    assert "modes: think 1, no_think 4, think_fallback 1" in capsys.readouterr().out
    # a stale sidecar stops the dry run before anything is counted
    sidecar(teacher, [("first:0", 1, SAMPLE_MESSAGES[2], REASONING)])
    assert train.main(argv, load_family=family_hook()) == 2
    assert "stale" in capsys.readouterr().err


def test_train_dry_run_counts_the_samples_the_family_refuses_by_kind_and_a_real_run_stops(tmp_path, capsys):
    first, second, report = two_files(tmp_path)
    model, out = model_folder(tmp_path / "model"), tmp_path / "run"
    teacher = tmp_path / "teacher.jsonl"
    sidecar(teacher, [("first:0", 1, SAMPLE_MESSAGES[4], "<s> a BOS inside the reasoning")])
    argv = [str(first), str(second), "--out", str(out), "--overlap-report", str(report), "--model",
            str(model), "--reasoning", str(teacher)]
    hook = family_hook(tokenizer=FakeFamilyTokenizer("hybrid", bos="token"))
    assert train.main([*argv, "--dry-run"], load_family=hook) == 0
    printed = capsys.readouterr().out
    assert "refused: template 0, boundary 0, bos 1" in printed and "bos ('first:0', 1):" in printed
    assert "samples 3, dropped over_length 0, after_repair 0, skipped_overlap 0" in printed  # first.jsonl
    assert "refused template 0, boundary 0, bos 1" in printed  # the per-file account
    summary = json.loads(printed.splitlines()[-1].removeprefix("summary: "))
    assert summary["refused"] == {"template": 0, "boundary": 0, "bos": 1} and summary["samples"] == 4
    pytest.importorskip("torch")  # a real run imports torch before it builds a sample
    assert train.main(argv, load_family=hook) == 2  # not a dry run: the first refusal stops the run
    assert "BOS" in capsys.readouterr().err and not (out / "manifest.json").exists()
    # a template that writes the reasoning field as given, newlines and all: the construction differs
    sidecar(teacher, [("first:0", 1, SAMPLE_MESSAGES[4], REASONING)])  # a newline at either end
    hook = family_hook(tokenizer=FakeFamilyTokenizer("hybrid", strips_reasoning=False))
    assert train.main([*argv, "--dry-run"], load_family=hook) == 0
    printed = capsys.readouterr().out
    assert "refused: template 1, boundary 0, bos 0" in printed and "template ('first:0', 1):" in printed
    summary = json.loads(printed.splitlines()[-1].removeprefix("summary: "))
    assert summary["refused"] == {"template": 1, "boundary": 0, "bos": 0} and summary["samples"] == 4
    assert train.main(argv, load_family=hook) == 2
    assert "reasoning_content" in capsys.readouterr().err and not (out / "manifest.json").exists()


def test_train_dry_run_applies_the_sample_keys_and_the_smoke_order(tmp_path, capsys):
    first, second, report = two_files(tmp_path)
    model, out = model_folder(tmp_path / "model"), tmp_path / "run"
    keys = tmp_path / "keys.jsonl"
    listed = json.dumps({"id": "second:0", "k": 0}) + "\n" + json.dumps({"id": "first:1", "k": 1}) + "\n"
    keys.write_text(listed, encoding="utf-8")
    argv = [str(first), str(second), "--out", str(out), "--overlap-report", str(report), "--model",
            str(model), "--dry-run", "--sample-keys", str(keys), "--smoke-longest", "--stop-at-step", "1",
            "--grad-accum", "1"]
    assert train.main(argv, load_family=family_hook()) == 0
    printed = capsys.readouterr().out
    assert "sample keys: 2 listed, 2 samples selected" in printed
    assert "smoke: the longest samples first" in printed and "stop at step 1 of 4" in printed
    assert "mask check: 2 samples" in printed
    summary = json.loads(printed.splitlines()[-1].removeprefix("summary: "))
    assert summary["samples"] == 2 and summary["sample_keys"]["listed"] == 2
    assert train.main([*argv[:-4], "--stop-at-step", "99"], load_family=family_hook()) == 0  # a dry run tells
    assert "beyond the run" in capsys.readouterr().out
    keys.write_text(json.dumps({"id": "first:0", "k": 9}) + "\n", encoding="utf-8")
    assert train.main(argv, load_family=family_hook()) == 2
    assert "without a sample" in capsys.readouterr().err


def test_train_deterministic_failures_use_exit_codes_two_to_five(tmp_path, capsys):
    for line in ("0 ", "2 ", "3 ", "4 ", "5 ", "retryable"):
        assert line in train.__doc__.split("Exit codes")[1]
    argv, _, _ = resume_setup(tmp_path)
    hook = family_hook()
    assert train.main([str(tmp_path / "*.missing"), *argv[1:]], load_family=hook) == 2  # no data file
    assert train.main([*argv[:-1], str(tmp_path / "no-model")], load_family=hook) == 2  # no model folder
    record = {"messages": SAMPLE_MESSAGES, "tools": schemas.MODEL_TOOLS, "history": "agent", "id": "r-1",
              "schema_hash": schemas.SCHEMA_HASH, "prompt_hash": "0" * 16}
    agent = tmp_path / "agent.jsonl"  # a file the overlap report does not list, used unchecked
    agent.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert train.main([str(agent), *argv[1:], "--no-overlap-check", "--dry-run"], load_family=hook) == 2
    assert "r-1" in capsys.readouterr().err  # a record under another prompt
    bare = tmp_path / "bare"  # a folder that is no model folder: no family
    bare.mkdir()
    assert train.main([*argv[:-1], str(bare), "--dry-run"], load_family=train.default_load_family) == 2
    assert "config.json" in capsys.readouterr().err
    missing = [*argv, "--dry-run", "--reasoning", str(tmp_path / "none.jsonl")]
    assert train.main(missing, load_family=hook) == 2 and "none.jsonl" in capsys.readouterr().err


# M8 (docs/m8-计划.md WP3): --init-adapter starts the LoRA from an existing adapter's weights (FT5 from
# FT3's step 1635), --reasoning-budget pins the budget R the run records. Neither changes a run without it.

TODAYS_CONFIG = ["mode", "model", "lora", "sft", "family", "pinned", "args", "data", "schema_hash",
                 "prompt_version", "prompt_hash", "total_optimizer_steps", "save_cadence", "resume",
                 "sft_resolved"]  # manifest.json's config keys before M8, in their order
TODAYS_REASONING = ["files", "ignored", "rows", "tokens", "budget"]  # data.reasoning's keys before M8
FAKE_TARGETS = ["v_proj", "q_proj"]  # the fake family's (q_proj, v_proj), in the order PEFT may write a set


def safetensors_bytes(names: list[str], shape: tuple[int, int] = (16, 8)) -> bytes:
    """A safetensors file of BF16 zeros under these tensor names: the 8-byte little-endian header length,
    the JSON header, the data."""
    size = shape[0] * shape[1] * 2
    header = {"__metadata__": {"format": "pt"},
              **{name: {"dtype": "BF16", "shape": list(shape), "data_offsets": [k * size, (k + 1) * size]}
                 for k, name in enumerate(names)}}
    text = json.dumps(header).encode("utf-8")
    return len(text).to_bytes(8, "little") + text + bytes(size * len(names))


def lora_names(targets: list[str], layers: int = 2) -> list[str]:
    return [f"base_model.model.model.layers.{layer}.self_attn.{target}.lora_{ab}.weight"
            for layer in range(layers) for target in targets for ab in "AB"]


def fake_adapter(folder: Path, model: Path, *, weights: bytes | None = None, **config) -> Path:
    """An adapter folder as a trainer checkpoint holds one (runs/train/ft3/trainer/checkpoint-1635): the
    adapter_config.json PEFT writes for the fake family's LoRA (r 16, alpha 32 on its two targets) trained on
    model, its weights, and the optimizer state that --init-adapter never reads; config replaces keys (None
    removes one)."""
    folder.mkdir(parents=True, exist_ok=True)
    written = {"peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
               "target_modules": FAKE_TARGETS, "base_model_name_or_path": str(model), "bias": "none",
               "use_dora": False, "use_rslora": False, "modules_to_save": None, "rank_pattern": {},
               "alpha_pattern": {}, "layers_to_transform": None, "fan_in_fan_out": False,
               "inference_mode": True, "peft_version": "0.21.2"}
    written = {k: v for k, v in {**written, **config}.items() if v is not None}
    (folder / "adapter_config.json").write_text(json.dumps(written, indent=2), encoding="utf-8")
    (folder / "adapter_model.safetensors").write_bytes(
        safetensors_bytes(lora_names(FAKE_TARGETS)) if weights is None else weights)
    (folder / "optimizer.pt").write_bytes(b"not read")
    return folder


def test_train_m8_flags_default_off_and_stay_out_of_the_recorded_args():
    # absent from the parsed arguments unless given (argparse.SUPPRESS): vars(args), and with it the
    # manifest's args, stay those of a run before M8
    args = train.build_parser().parse_args(["a.jsonl", "--out", "o"])
    assert "init_adapter" not in vars(args) and "reasoning_budget" not in vars(args)
    before = [k for k in vars(args) if k not in ("glossary", "glossary_dropout")]
    assert list(train.recorded_args(args)) == before  # the args a run before M8 recorded, in their order
    given = train.build_parser().parse_args(["a.jsonl", "--out", "o", "--init-adapter", "ck",
                                             "--reasoning-budget", "1280"])
    recorded = train.recorded_args(given)
    assert (recorded["init_adapter"], recorded["reasoning_budget"]) == ("ck", 1280)
    assert list(recorded)[-2:] == ["init_adapter", "reasoning_budget"] and list(recorded)[:-2] == before
    for bad in ("0", "-5", "x", "12.5"):
        with pytest.raises(SystemExit):
            train.build_parser().parse_args(["a.jsonl", "--reasoning-budget", bad])


@pytest.mark.parametrize("flags", [["--init-adapter", "ck", "--resume"], ["--init-adapter="]])
def test_train_refuses_an_init_adapter_with_resume_or_an_empty_one_before_anything_is_read(flags, capsys):
    # the data file, the model folder and the adapter folder do not exist: refused before anything is read
    assert train.main(["missing.jsonl", "--out", "o", *flags]) == 2
    err = capsys.readouterr().err
    assert "--init-adapter" in err
    assert ("--resume" in err) if "--resume" in flags else ("names no folder" in err)


def test_train_dry_run_with_an_init_adapter_records_its_weights_without_the_training_stack(tmp_path, capsys):
    first, second, report = two_files(tmp_path)
    model, out = model_folder(tmp_path / "model"), tmp_path / "run"
    adapter = fake_adapter(tmp_path / "ft3" / "trainer" / "checkpoint-1635", model)
    argv = [str(first), str(second), "--out", str(out), "--overlap-report", str(report), "--model",
            str(model), "--dry-run"]
    assert train.main(argv, load_family=family_hook()) == 0
    plain_out = capsys.readouterr().out
    plain = json.loads(plain_out.splitlines()[-1].removeprefix("summary: "))
    assert "init_adapter" not in plain and "init adapter" not in plain_out
    assert train.main([*argv, "--init-adapter", str(adapter)], load_family=family_hook()) == 0
    printed = capsys.readouterr().out
    summary = json.loads(printed.splitlines()[-1].removeprefix("summary: "))
    weights = adapter / "adapter_model.safetensors"
    assert summary["init_adapter"] == {
        "path": str(adapter), "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256((adapter / "adapter_config.json").read_bytes()).hexdigest(),
        "tensors": 8}
    assert {k: v for k, v in summary.items() if k != "init_adapter"} == plain  # nothing else changes
    assert f"init adapter: {adapter}" in printed and "8 tensors" in printed and "not read" in printed
    assert not out.exists()
    assert probe_training_stack([[*argv, "--init-adapter", str(adapter)]]) == {"codes": [0], "loaded": []}


@pytest.mark.parametrize(("changes", "expected"), [
    ({"r": 8}, "r 8 (the run's 16)"),
    ({"lora_alpha": 16}, "lora_alpha 16 (the run's 32)"),
    ({"target_modules": ["q_proj", "k_proj", "v_proj"]}, "target_modules"),
    ({"use_dora": True}, "use_dora"),
    ({"modules_to_save": ["lm_head"]}, "modules_to_save"),
    ({"peft_type": "IA3"}, "peft_type"),
    ({"task_type": "SEQ_CLS"}, "task_type"),
    ({"base_model_name_or_path": "models/Other"}, "base_model_name_or_path"),
    ({"base_model_name_or_path": None}, "base_model_name_or_path"),
], ids=["rank", "alpha", "targets", "dora", "modules-to-save", "peft-type", "task-type", "other-base",
        "no-base"])
def test_train_refuses_an_init_adapter_that_does_not_fit_the_run(changes, expected, tmp_path, capsys):
    argv, _, out = resume_setup(tmp_path, "--dry-run")
    model = Path(argv[argv.index("--model") + 1])
    adapter = fake_adapter(tmp_path / "adapter", model, **changes)
    assert train.main([*argv, "--init-adapter", str(adapter)], load_family=family_hook()) == 2
    err = capsys.readouterr().err
    assert expected in err and "adapter_config.json" in err and not out.exists()


def test_train_refuses_an_init_adapter_folder_without_its_files_or_with_unreadable_weights(tmp_path, capsys):
    argv, _, _ = resume_setup(tmp_path, "--dry-run")
    model = Path(argv[argv.index("--model") + 1])
    hook = family_hook()
    assert train.main([*argv, "--init-adapter", str(tmp_path / "nowhere")], load_family=hook) == 2
    assert "no adapter folder" in capsys.readouterr().err
    adapter = fake_adapter(tmp_path / "adapter", model)
    (adapter / "adapter_model.safetensors").unlink()
    assert train.main([*argv, "--init-adapter", str(adapter)], load_family=hook) == 2
    assert "adapter_model.safetensors" in capsys.readouterr().err
    adapter = fake_adapter(tmp_path / "no-config", model)
    (adapter / "adapter_config.json").unlink()
    assert train.main([*argv, "--init-adapter", str(adapter)], load_family=hook) == 2
    assert "adapter_config.json" in capsys.readouterr().err
    adapter = fake_adapter(tmp_path / "garbled", model, weights=b"\xff" * 12)
    assert train.main([*argv, "--init-adapter", str(adapter)], load_family=hook) == 2
    assert "safetensors" in capsys.readouterr().err
    adapter = fake_adapter(tmp_path / "empty", model, weights=safetensors_bytes([]))
    assert train.main([*argv, "--init-adapter", str(adapter)], load_family=hook) == 2
    assert "no tensor" in capsys.readouterr().err
    # the base model is compared as a folder: the same folder named otherwise fits
    same = fake_adapter(tmp_path / "same", model, base_model_name_or_path=str(model / ".." / model.name))
    assert train.main([*argv, "--init-adapter", str(same)], load_family=hook) == 0
    capsys.readouterr()


def test_train_reasoning_budget_pins_the_recorded_budget_and_keeps_the_rules_beside_it(tmp_path, capsys):
    first, second, report = two_files(tmp_path)
    model, out = model_folder(tmp_path / "model"), tmp_path / "run"
    teacher = tmp_path / "teacher.jsonl"
    sidecar(teacher, [("first:0", 1, SAMPLE_MESSAGES[4], REASONING)])
    argv = [str(first), str(second), "--out", str(out), "--overlap-report", str(report), "--model",
            str(model), "--dry-run", "--reasoning", str(teacher)]
    assert train.main(argv, load_family=family_hook()) == 0
    printed = capsys.readouterr().out
    plain = json.loads(printed.splitlines()[-1].removeprefix("summary: "))
    assert list(plain["reasoning"]) == TODAYS_REASONING and plain["reasoning"]["budget"] == 1024
    assert "budget R 1024 (max(1024, p99 up to a multiple of 256), at most 2048)" in printed
    assert train.main([*argv, "--reasoning-budget", "1280"], load_family=family_hook()) == 0
    printed = capsys.readouterr().out
    pinned = json.loads(printed.splitlines()[-1].removeprefix("summary: "))["reasoning"]
    assert pinned == {**plain["reasoning"], "budget": 1280, "budget_rule": 1024, "budget_pinned": True}
    assert "budget R 1280 (pinned by --reasoning-budget; the rule, max(1024, p99 up to a multiple of 256), " \
           "at most 2048, picks 1024)" in printed
    # no THINK sample: the rule picks nothing, the pin stays
    assert train.main([*argv, "--no-think", "--reasoning-budget", "1280"], load_family=family_hook()) == 0
    printed = capsys.readouterr().out
    none = json.loads(printed.splitlines()[-1].removeprefix("summary: "))["reasoning"]
    assert (none["budget"], none["budget_rule"], none["budget_pinned"]) == (1280, None, True)
    assert "ignored (--no-think); budget R 1280 (pinned by --reasoning-budget)" in printed
    assert train.reasoning_summary([], [], ignored=False) == {"files": [], "ignored": False, "rows": 0,
                                                               "tokens": train.length_stats([]),
                                                               "budget": None}


def test_train_resume_compares_a_pinned_reasoning_budget_and_ignores_it_when_none_was_pinned(tmp_path,
                                                                                             capsys):
    argv, _, out = resume_setup(tmp_path, "--reasoning-budget", "1280")
    path = write_reference_manifest(argv, out)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["config"]["data"]["reasoning"] |= {"budget": 1280, "budget_rule": None, "budget_pinned": True}
    path.write_text(json.dumps(manifest), encoding="utf-8")
    args = train.build_parser().parse_args([*argv, "--resume"])
    family = loaded_fake_family(args.model)
    assert train.resume_differences(args, train.expand_data_files(args.data), family) == []
    hook = family_hook()
    assert train.main([*argv[:-2], "--resume"], load_family=hook) == 5  # the pin left out
    assert "reasoning.budget (--reasoning-budget): 1280 -> None" in capsys.readouterr().err
    assert train.main([*argv[:-2], "--resume", "--reasoning-budget", "1536"], load_family=hook) == 5
    assert "reasoning.budget (--reasoning-budget): 1280 -> 1536" in capsys.readouterr().err
    # a manifest without a pin (every run before M8) and an invocation without one agree
    (tmp_path / "plain").mkdir()
    plain_argv, _, plain_out = resume_setup(tmp_path / "plain")
    write_reference_manifest(plain_argv, plain_out)
    args = train.build_parser().parse_args([*plain_argv, "--resume"])
    assert train.resume_differences(args, train.expand_data_files(args.data),
                                    loaded_fake_family(args.model)) == []


def tiny_peft_model(torch, transformers, peft, targets: list[str], r: int = 4):
    """A two-layer Llama with random weights (seed 0) wrapped in a fresh LoRA of rank r on targets, on the
    CPU: what SFTTrainer's peft_config gives a run before it trains (LoRA's B zero)."""
    config = transformers.LlamaConfig(vocab_size=64, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                                      head_dim=8)
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(config)
    return peft.get_peft_model(model, peft.LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05,
                                                      target_modules=targets, task_type="CAUSAL_LM"))


def lora_state(model) -> dict:
    return {name: p.detach().clone() for name, p in model.named_parameters() if ".lora_" in name}


def test_train_init_adapter_loads_the_saved_weights_into_the_fresh_lora_and_refuses_a_misfit(tmp_path):
    # the stack's own round trip on the CPU: an adapter saved as a trainer checkpoint saves it, loaded into
    # the fresh LoRA a run builds; every LoRA weight is the saved one, and an adapter of other targets or
    # another rank stops the run
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    peft = pytest.importorskip("peft")
    trained = tiny_peft_model(torch, transformers, peft, ["q_proj", "v_proj"])
    with torch.no_grad():
        for name, parameter in trained.named_parameters():
            if ".lora_" in name:
                parameter.normal_()
    saved = tmp_path / "checkpoint-1635"
    trained.save_pretrained(saved)
    fresh = tiny_peft_model(torch, transformers, peft, ["q_proj", "v_proj"])
    before = lora_state(fresh)
    assert all(not before[n].any() for n in before if ".lora_B." in n)  # a fresh LoRA's B is zero
    assert train.load_init_adapter(fresh, saved) == 8  # 2 layers x 2 targets x (A, B)
    after, expected = lora_state(fresh), lora_state(trained)
    assert after.keys() == expected.keys() and all(torch.equal(after[n], expected[n]) for n in after)
    assert train.safetensors_names(saved / "adapter_model.safetensors") == sorted(
        f"base_model.model.model.layers.{i}.self_attn.{t}.lora_{ab}.weight"
        for i in range(2) for t in ("q_proj", "v_proj") for ab in "AB")
    wider = tiny_peft_model(torch, transformers, peft, ["q_proj", "k_proj", "v_proj"])
    with pytest.raises(train.InitAdapterError, match=r"4 LoRA weight\(s\) of the run.*k_proj"):
        train.load_init_adapter(wider, saved)
    narrower = tiny_peft_model(torch, transformers, peft, ["q_proj"])
    with pytest.raises(train.InitAdapterError, match=r"4 weight\(s\) the run has no LoRA for.*v_proj"):
        train.load_init_adapter(narrower, saved)
    with pytest.raises(train.InitAdapterError, match="size mismatch"):
        train.load_init_adapter(tiny_peft_model(torch, transformers, peft, ["q_proj", "v_proj"], r=8), saved)


class StubCallbackBase:
    """transformers.TrainerCallback's stand-in: jsonl_logger and the --stop-at-step callback subclass it, and
    every event they do not handle does nothing, as in transformers."""

    def on_train_begin(self, args, state, control, **kwargs):
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        return control


def stub_training_stack(monkeypatch, events: list) -> None:
    """The training stack main() imports, as stand-ins in sys.modules for the test: SFTTrainer keeps the rows
    with TRL's labels (-100 off the completion), records when it is built and when it trains (one logged
    step), and its model saves nothing; the base model is a name (train.load_model replaced)."""

    class Config:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to_json_string(self):
            return json.dumps(self.kwargs)

    class Model:
        def save_pretrained(self, path):
            events.append("save")

    class Trainer:
        def __init__(self, model, args, train_dataset, processing_class, peft_config):
            events.append("trainer")
            self.model, self.callbacks = Model(), []
            self.train_dataset = [{"input_ids": row["input_ids"],
                                   "labels": [t if m else -100 for t, m in zip(row["input_ids"],
                                                                               row["completion_mask"],
                                                                               strict=True)]}
                                  for row in train_dataset]
            self.state = SimpleNamespace(global_step=0, max_steps=1, log_history=[])

        def add_callback(self, callback):
            self.callbacks.append(callback)

        def train(self, resume_from_checkpoint=None):
            events.append("train")
            for callback in self.callbacks:
                callback.on_train_begin(None, self.state, None)
            self.state.global_step = 1
            for callback in self.callbacks:
                callback.on_log(None, self.state, None, logs={"loss": 1.0, "num_tokens": 100})
            return SimpleNamespace(training_loss=1.0)

    stack = {"torch": SimpleNamespace(bfloat16="bf16", cuda=FakeCuda(), __version__="stub"),
             "datasets": SimpleNamespace(Dataset=SimpleNamespace(from_list=list)),
             "peft": SimpleNamespace(LoraConfig=Config),
             "transformers": SimpleNamespace(BitsAndBytesConfig=Config, TrainerCallback=StubCallbackBase),
             "trl": SimpleNamespace(SFTConfig=Config, SFTTrainer=Trainer)}
    for name, module in stack.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(train, "load_model", lambda *args, **kwargs: "base model")


def test_train_main_records_the_init_adapter_and_loads_it_before_training_and_only_then(tmp_path,
                                                                                         monkeypatch, capsys):
    events: list = []
    stub_training_stack(monkeypatch, events)
    argv, _, out = resume_setup(tmp_path)
    model = Path(argv[argv.index("--model") + 1])
    # without the new flags: the manifest's config, its args and data.reasoning keep the keys of today
    assert train.main(argv, load_family=family_hook()) == 0
    assert events == ["trainer", "train", "save"]
    config = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["config"]
    assert list(config) == TODAYS_CONFIG and list(config["data"]["reasoning"]) == TODAYS_REASONING
    plain = train.build_parser().parse_args(argv)
    assert list(config["args"]) == [k for k in vars(plain) if k not in ("glossary", "glossary_dropout",
                                                                        "init_adapter", "reasoning_budget")]
    capsys.readouterr()
    # with them: the adapter's weights go into the trainer's LoRA before it trains, and the manifest says so
    adapter = fake_adapter(tmp_path / "checkpoint-1635", model)
    loaded = []

    def loader(lora_model, folder):
        events.append("init")
        loaded.append((lora_model, folder))
        return 8

    monkeypatch.setattr(train, "load_init_adapter", loader)
    events.clear()
    second = tmp_path / "run-init"
    given = [*argv[:2], str(second), *argv[3:], "--init-adapter", str(adapter), "--reasoning-budget", "1280"]
    assert train.main(given, load_family=family_hook()) == 0
    assert events == ["trainer", "init", "train", "save"] and loaded[0][1] == str(adapter)
    assert type(loaded[0][0]).__name__ == "Model"  # the trainer's model: the LoRA it trains
    assert "init adapter: 8 tensors loaded" in capsys.readouterr().out
    config = json.loads((second / "manifest.json").read_text(encoding="utf-8"))["config"]
    assert list(config) == [*TODAYS_CONFIG[:-1], "init_adapter", "sft_resolved"]
    assert config["init_adapter"]["path"] == str(adapter) and config["init_adapter"]["tensors"] == 8
    assert config["args"]["init_adapter"] == str(adapter) and config["args"]["reasoning_budget"] == 1280
    assert config["data"]["reasoning"]["budget"] == 1280 and config["data"]["reasoning"]["budget_pinned"]

    def refusing(lora_model, folder):
        events.append("init")
        raise train.InitAdapterError("the weights do not fit")

    monkeypatch.setattr(train, "load_init_adapter", refusing)
    events.clear()
    third = tmp_path / "run-refused"
    refused = [*argv[:2], str(third), *argv[3:], "--init-adapter", str(adapter)]
    assert train.main(refused, load_family=family_hook()) == 2
    assert events == ["trainer", "init"] and not (third / "result.json").exists()  # never trained
    assert "the weights do not fit" in capsys.readouterr().err


# scripts/export_adapter.py


def test_export_command_defaults_and_overrides(tmp_path):
    args = export.build_parser().parse_args(["runs/train/main/adapter"])
    assert (args.base, args.outtype, args.outfile, args.llama_cpp_src) == ("models/Qwen3-1.7B", "f16", None,
                                                                            None)
    with pytest.raises(LookupError, match="LLAMA_CPP_SRC"):  # the source tree is per machine: no default
        export.command(args, python="py", env={})
    cmd = export.command(args, python="py", env={"LLAMA_CPP_SRC": "elsewhere"})
    converter = Path("elsewhere") / "convert_lora_to_gguf.py"
    assert cmd == ["py", str(converter), "--base", "models/Qwen3-1.7B", "--outtype", "f16", "--outfile",
                   str(Path("runs/train/main/adapter-f16.gguf")), "runs/train/main/adapter"]
    assert export.llama_cpp_src(None, {"LLAMA_CPP_SRC": "elsewhere"}) == Path("elsewhere")
    assert export.llama_cpp_src("explicit", {"LLAMA_CPP_SRC": "elsewhere"}) == Path("explicit")
    with pytest.raises(LookupError, match="LLAMA_CPP_SRC"):
        export.llama_cpp_src("", {"LLAMA_CPP_SRC": ""})
    args = export.build_parser().parse_args(["ad", "--outtype", "q8_0", "--outfile", "x.gguf", "--base", "m"])
    cmd = export.command(args, python="py", env={"LLAMA_CPP_SRC": str(tmp_path)})
    assert cmd == ["py", str(tmp_path / "convert_lora_to_gguf.py"), "--base", "m", "--outtype", "q8_0",
                   "--outfile", "x.gguf", "ad"]


def test_export_main_runs_the_converter_and_reports_the_output_size(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_SRC", raising=False)
    src = tmp_path / "src"
    src.mkdir()
    (src / "convert_lora_to_gguf.py").write_text("# fake converter\n", encoding="utf-8")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    seen = []

    def fake_run(cmd):
        seen.append(cmd)
        Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"x" * 1234)
        return subprocess.CompletedProcess(cmd, 0)

    code = export.main([str(adapter), "--llama-cpp-src", str(src)], run=fake_run)
    out = capsys.readouterr().out
    assert code == 0 and seen[0][0] == sys.executable and seen[0][1] == str(src / "convert_lora_to_gguf.py")
    assert "1,234 bytes" in out and str(tmp_path / "adapter-f16.gguf") in out
    assert export.main([str(adapter), "--llama-cpp-src", str(tmp_path / "nowhere")], run=fake_run) == 2
    assert export.main([str(tmp_path / "no-adapter"), "--llama-cpp-src", str(src)], run=fake_run) == 2
    failing = export.main([str(adapter), "--llama-cpp-src", str(src)],
                          run=lambda cmd: subprocess.CompletedProcess(cmd, 7))
    assert failing == 7
    argv = [str(adapter), "--llama-cpp-src", str(src), "--outfile", str(tmp_path / "none.gguf")]
    assert export.main(argv, run=lambda cmd: subprocess.CompletedProcess(cmd, 0)) == 3
    capsys.readouterr()
    assert export.main([str(adapter)], run=fake_run) == 2  # neither --llama-cpp-src nor LLAMA_CPP_SRC
    assert "LLAMA_CPP_SRC" in capsys.readouterr().out and len(seen) == 1


def test_scripts_carry_no_machine_path():
    sources = [REPO / "scripts" / f"{name}.py"
               for name in ("serve_local", "export_adapter", "train", "lock_set", "select_checkpoint")]
    for path in [*sources, REPO / "src" / "research_agent" / "llm" / "server.py"]:
        source = path.read_text(encoding="utf-8")
        assert re.search(r"[A-Za-z]:\\", source) is None, path.name  # a drive letter: a Windows path
    assert not hasattr(serve, "DEFAULT_LLAMA_SERVER") and not hasattr(export, "DEFAULT_LLAMA_CPP_SRC")


# scripts/serve_local.py


def test_serve_local_is_a_command_line_over_the_shared_server_module():
    from research_agent.llm import server

    for name in ("llama_server", "start", "wait_ready", "pump", "stop"):
        assert getattr(serve, name) is getattr(server, name), name
    args = serve.build_parser().parse_args(["--adapter", "a.gguf", "--port", "8091", "--", "--x", "1"])
    expected = server.server_command("s.exe", adapter="a.gguf", port=8091, extra=["--x", "1"])
    assert serve.build_command(args, env={"LLAMA_SERVER": "s.exe"}) == expected


class FakeProcess:
    """Yields scripted output lines; exits with exit_code once its output ends when exit_code is given."""

    def __init__(self, lines, exit_code=None, block=False):
        self.lines, self.exit_code, self.block = list(lines), exit_code, block
        self.stdout = self._output()
        self.returncode = None
        self.terminated = False

    def _output(self):
        yield from self.lines
        if self.block:
            threading.Event().wait()  # never ends, like a server that is still loading
        if self.exit_code is not None:
            self.returncode = self.exit_code

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = 0 if self.returncode is None else self.returncode

    def kill(self):
        self.terminate()

    def wait(self, timeout=None):
        return self.returncode


READY_LINE = "main: server is listening on http://127.0.0.1:8091 - starting the main loop\n"


def test_serve_command_with_and_without_an_adapter_and_passthrough_arguments():
    args = serve.build_parser().parse_args([])
    assert (args.port, args.ctx_size, args.ngl, args.timeout, args.once) == (8080, 8192, 99, 120.0, False)
    with pytest.raises(LookupError, match="LLAMA_SERVER"):  # the binary is per machine: no default
        serve.build_command(args, env={})
    cmd = serve.build_command(args, env={"LLAMA_SERVER": "env.exe"})
    assert cmd == ["env.exe", "-m", "models/Qwen3-1.7B-Q8_0.gguf", "-ngl", "99", "-c", "8192", "--jinja",
                   "--host", "127.0.0.1", "--port", "8080", "--cache-ram", "0"]  # prompt cache off by default
    assert "--lora" not in cmd
    args = serve.build_parser().parse_args(["--adapter", "a.gguf", "--port", "8091", "-c", "4096",
                                            "--llama-server", "srv.exe", "--once", "--cache-ram", "512", "--",
                                            "--chat-template-kwargs", '{"enable_thinking":false}'])
    cmd = serve.build_command(args, env={"LLAMA_SERVER": "ignored.exe"})
    assert cmd[0] == "srv.exe" and cmd[cmd.index("-c") + 1] == "4096"
    assert cmd[cmd.index("--cache-ram") + 1] == "512"
    assert cmd[cmd.index("--port") + 1] == "8091"
    lora = cmd.index("--lora")
    assert cmd[lora:lora + 3] == ["--lora", "a.gguf", "--lora-init-without-apply"]
    assert cmd[-2:] == ["--chat-template-kwargs", '{"enable_thinking":false}'] and args.once


def test_serve_says_the_adapter_is_active_unless_a_request_sends_scale_zero():
    # S1b (docs/m0-验证.md): b11349 applies a loaded adapter at scale 1.0 to requests without a lora field,
    # even under --lora-init-without-apply; only an explicit scale 0 gives the base model.
    for text in (serve.build_parser().format_help(), serve.__doc__):
        assert "not applied" not in text and "scale 0" in text and "--adapter" in text


def test_serve_wait_ready_detects_the_listening_line_and_echoes_what_came_before():
    echoed = []
    proc = FakeProcess(["loading model\n", READY_LINE, "later line\n"])
    ready, lines = serve.wait_ready(proc, timeout=5, echo=echoed.append)
    assert ready and echoed == ["loading model", READY_LINE.rstrip("\n")]
    assert lines.get(timeout=1) == "later line\n" and lines.get(timeout=1) is None  # the rest, then the end


def test_serve_wait_ready_fails_when_the_server_exits_or_the_timeout_passes():
    echoed = []
    ready, _ = serve.wait_ready(FakeProcess(["error: model file not found\n"], exit_code=1), timeout=5,
                                echo=echoed.append)
    assert not ready and echoed == ["error: model file not found"]
    ready, _ = serve.wait_ready(FakeProcess(["still loading\n"], block=True), timeout=0.3, echo=echoed.append)
    assert not ready and echoed[-1] == "still loading"


def test_serve_main_once_checks_readiness_and_stops_the_server(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("LLAMA_SERVER", raising=False)
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"")
    started = []

    def popen(cmd, **kwargs):
        started.append((cmd, kwargs))
        return FakeProcess(["loading\n", READY_LINE])

    code = serve.main(["--llama-server", str(exe), "--once", "--port", "8091"], popen=popen)
    out = capsys.readouterr().out
    assert code == 0 and "ready" in out.splitlines() and started[0][0][0] == str(exe)
    assert started[0][1]["stderr"] == subprocess.STDOUT and started[0][1]["text"] is True

    failing = serve.main(["--llama-server", str(exe), "--once", "--timeout", "0.3"],
                         popen=lambda cmd, **kw: FakeProcess(["boom\n"], exit_code=3))
    assert failing == 1 and "exited with 3" in capsys.readouterr().out
    assert serve.main(["--llama-server", str(tmp_path / "missing.exe"), "--once"], popen=popen) == 2
    capsys.readouterr()
    assert serve.main(["--once"], popen=popen) == 2 and len(started) == 1  # no flag, no LLAMA_SERVER
    assert "LLAMA_SERVER" in capsys.readouterr().out


def test_serve_pump_returns_the_exit_code_when_the_server_ends():
    proc = FakeProcess(["a\n", "b\n"], exit_code=0)
    ready, lines = serve.wait_ready(proc, timeout=5, pattern="a", echo=lambda line: None)
    echoed = []
    assert ready and serve.pump(proc, lines, echo=echoed.append) == 0 and echoed == ["b"]

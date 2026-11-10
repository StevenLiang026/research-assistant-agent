"""The regression anchor of the training path (docs/m7-家族层设计.md §1 不变的约定, §9) and the THINK path on
the real data, both over models/Qwen3-1.7B.

Anchor: the FT2 data (data/raw/public/*.jsonl and data/raw/synth/synth-m6.jsonl under the benchmark-overlap
report data/synth/benchmark-overlap-m6-all.json, no sidecar) built through the family path gives, for every
one of the 17,532 samples, the input_ids and the completion mask that TRL's own tokenization gives the old
path's prompt and completion strings: SFTTrainer._prepare_dataset itself (trl 1.14.1), the routine that
tokenized FT2's dataset, called on those strings with the same tokenizer and the same SFTConfig. The sample
count, the order and the token statistics are those of runs/train/ft2/manifest.json.

THINK path: with the teacher sidecar data/raw/synth/reasoning-m7-teacher.jsonl (2,814 rows), the dry run over
synth-m6.jsonl reports 2,814 THINK samples (less the ones that fall back at the chosen length), and a THINK
sample decodes to prompt(THINK) + reasoning + think_close + answer with the mask over exactly the reasoning,
think_close and answer tokens.

These runs take minutes and need the model folder, the data, the sidecar and the FT2 manifest, none of which
the repository holds: the module is skipped unless M7_FULL_REGRESSION=1 is set and the inputs are present.
The anchor builds an SFTConfig, which initialises CUDA when a GPU is visible; the GPU may be busy with a
training run, so the anchor runs in a process that hides it (CUDA_VISIBLE_DEVICES=-1, set before torch is
imported) and fails otherwise:

    $env:M7_FULL_REGRESSION = "1"; $env:CUDA_VISIBLE_DEVICES = "-1"
    .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider tests/train/test_regression_anchor.py

A quick model-backed check of the THINK construction on a few records runs whenever the model, the data
and the sidecar are present (no environment variable), as the suite's other model-backed tests do.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research_agent.data.reasoning import ReasoningIndex, turns
from research_agent.family.core import Mode, load_family
from research_agent.tools.schemas import MODEL_TOOLS
from research_agent.train import samples

REPO = Path(__file__).resolve().parents[2]
MODEL = REPO / "models" / "Qwen3-1.7B"
PUBLIC = sorted((REPO / "data" / "raw" / "public").glob("*.jsonl"))
SYNTH = REPO / "data" / "raw" / "synth" / "synth-m6.jsonl"
OVERLAP = REPO / "data" / "synth" / "benchmark-overlap-m6-all.json"
TEACHER = REPO / "data" / "raw" / "synth" / "reasoning-m7-teacher.jsonl"
FT2_MANIFEST = REPO / "runs" / "train" / "ft2" / "manifest.json"
FULL = os.environ.get("M7_FULL_REGRESSION") == "1"
TEACHER_ROWS = 2814  # data/raw/synth/reasoning-m7-teacher.report.json: output.rows
THINK_MAX_LENGTH = 5120  # docs/m7-计划.md §3: the length the thinking candidates train at

present = MODEL.is_dir() and (MODEL / "tokenizer.json").is_file() and SYNTH.is_file()
needs_model = pytest.mark.skipif(not present, reason="local model files or training data are not present")


def load_train():
    spec = importlib.util.spec_from_file_location("script_train_anchor", REPO / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ft2_argv(*extra: str) -> list[str]:
    return [str(REPO / "data" / "raw" / "public" / "*.jsonl"), str(SYNTH), "--model", str(MODEL),
            "--overlap-report", str(OVERLAP), *extra]


@pytest.mark.skipif(not (FULL and present and PUBLIC and OVERLAP.is_file() and FT2_MANIFEST.is_file()),
                    reason="the full regression runs with M7_FULL_REGRESSION=1 over the model, the FT2 data, "
                           "the overlap report and runs/train/ft2/manifest.json")
def test_the_ft2_data_through_the_family_path_is_what_trl_tokenized_for_ft2(monkeypatch):
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
        pytest.fail("torch was imported with a visible GPU before this test; run it in its own process with "
                    "CUDA_VISIBLE_DEVICES=-1 (module docstring)")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    import torch
    from datasets import Dataset
    from trl import SFTConfig
    from trl.trainer.sft_trainer import SFTTrainer

    train = load_train()
    args = train.build_parser().parse_args(ft2_argv("--out", "runs/train/anchor"))
    files = train.expand_data_files(args.data)
    skips, _ = train.overlap_gate(args.overlap_report, files, check=True)
    family = load_family(MODEL)
    assert family.spec.name == "qwen3-hybrid"
    manifest = json.loads(FT2_MANIFEST.read_text(encoding="utf-8"))["config"]
    assert [Path(f["path"]).name for f in manifest["data"]["files"]] == [f.name for f in files]

    new, dropped, per_file = train.assemble(files, family, args.max_length, skips=skips)
    assert len(new) == manifest["data"]["samples"] == 17532 and dropped.total == 0
    assert [(f["samples"], f["tokens"]) for f in per_file] == [
        (f["samples"], f["tokens"]) for f in manifest["data"]["files"]]
    stats = train.data_summary(new, dropped, per_file, None)["tokens"]
    assert stats == manifest["data"]["tokens"]  # min 24, mean 887, max 4090, total 15,548,908
    assert {s["mode"] for s in new} == {"no_think"} and not any(s["think_fallback"] for s in new)

    # the old path: today's prompt and completion strings, tokenized by TRL's own preparation routine
    old = []
    for path in files:
        skip = skips.get(path.resolve(), set())
        for number, record in enumerate(samples.load_records(path)):
            if number not in skip:
                old += samples.build_step_samples(record, family.tokenizer, max_length=args.max_length)[0]
    assert len(old) == len(new)
    sft = {**train.settings(args, family.spec)["sft"], "use_cpu": True}
    config = SFTConfig(**sft)
    assert not torch.cuda.is_initialized(), "the anchor must not take the GPU (module docstring)"
    assert config.loss_type == "chunked_nll" and config.completion_only_loss and not config.shuffle_dataset
    trainer = SimpleNamespace(_tokenizer=family.tokenizer, chat_template=None, completion_only_loss=True)
    dataset = Dataset.from_list([{"prompt": s["prompt"], "completion": s["completion"]} for s in old])
    prepared = SFTTrainer._prepare_dataset(trainer, dataset, family.tokenizer, config, packing=False,
                                           formatting_func=None, dataset_name="train")
    assert len(prepared) == len(new) and "labels" in prepared.column_names

    rows = train.pre_tokenized(new)
    for index, (row, sample, example) in enumerate(zip(rows, new, prepared, strict=True)):
        assert row["input_ids"] == example["input_ids"], f"sample {index} {sample['key']}: ids differ"
        mask = [int(label != -100) for label in example["labels"]]
        assert row["completion_mask"] == mask, f"sample {index} {sample['key']}: the mask differs"
        assert sum(mask) == len(sample["input_ids"]) - sample["completion_start"]

    # the new rows through the same routine: TRL tokenizes nothing, appends nothing, and folds the mask
    # into the labels it trains on (completion_only_loss), so the trainer sees exactly these ids
    ours = SFTTrainer._prepare_dataset(trainer, Dataset.from_list(rows), family.tokenizer, config,
                                       packing=False, formatting_func=None, dataset_name="train")
    assert ours.column_names == ["input_ids", "labels"] and len(ours) == len(rows)
    for index, (row, example) in enumerate(zip(rows, ours, strict=True)):
        assert example["input_ids"] == row["input_ids"], f"row {index}: TRL changed the ids"
        labels = [t if m else -100 for t, m in zip(row["input_ids"], row["completion_mask"], strict=True)]
        assert example["labels"] == labels, f"row {index}: TRL's labels are not our mask"


@pytest.mark.skipif(not (FULL and present and TEACHER.is_file() and OVERLAP.is_file()),
                    reason="the full THINK dry run needs M7_FULL_REGRESSION=1, the model, synth-m6 and the "
                           "teacher sidecar")
def test_the_dry_run_with_the_teacher_sidecar_reports_every_teacher_turn_as_think(tmp_path, capsys):
    train = load_train()
    argv = [str(SYNTH), "--model", str(MODEL), "--overlap-report", str(OVERLAP), "--reasoning", str(TEACHER),
            "--max-length", str(THINK_MAX_LENGTH), "--out", str(tmp_path / "run"), "--dry-run"]
    assert train.main(argv) == 0
    printed = capsys.readouterr().out
    summary = json.loads(printed.splitlines()[-1].removeprefix("summary: "))
    modes = summary["modes"]
    assert modes["think"] + modes["think_fallback"] == TEACHER_ROWS
    assert summary["refused"] == {"template": 0, "boundary": 0, "bos": 0}
    assert summary["reasoning"]["rows"] == TEACHER_ROWS and summary["reasoning"]["ignored"] is False
    budget = summary["reasoning"]["budget"]
    assert budget == samples.reasoning_budget(summary["reasoning"]["tokens"]["p99"])
    assert 1024 <= budget <= 2048 and budget % 256 == 0
    assert re.search(r"modes: think \d+, no_think \d+, think_fallback \d+", printed)
    assert not (tmp_path / "run").exists()


@needs_model
@pytest.mark.skipif(not TEACHER.is_file(), reason="the teacher sidecar is not present")
def test_a_think_sample_on_real_data_decodes_to_the_think_prompt_the_reasoning_and_the_answer():
    family = load_family(MODEL)
    index = ReasoningIndex([TEACHER])
    spec = family.spec
    checked = 0
    with SYNTH.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if not any((record["id"], k) in index.rows for k, _, _ in turns(record["messages"])):
                continue
            built, dropped = samples.build_family_samples(record, family, name=record["id"], reasoning=index,
                                                          max_length=THINK_MAX_LENGTH)
            assert dropped.total == 0
            for sample in built:
                key = tuple(sample["key"])
                row = index.rows.get(key)
                if row is None:
                    assert sample["mode"] == "no_think" and not sample["think_fallback"]
                    continue
                assert sample["mode"] == "think" and not sample["think_fallback"]
                ids, start = sample["input_ids"], sample["completion_start"]
                prompt, completion = family.decode(ids[:start]), family.decode(ids[start:])
                assert prompt == sample["prompt"] and prompt.endswith(spec.generation_prompt[Mode.THINK])
                reasoning = row["reasoning"].strip("\n")
                assert completion.startswith(reasoning + spec.think_close)
                answer = completion[len(reasoning) + len(spec.think_close):]
                assert answer.endswith(spec.turn_end) and completion == sample["completion"]
                plain = samples.render_family_sample(family, *_pair(record, key[1]), tools=MODEL_TOOLS)
                assert answer == plain["completion"] and plain["prompt"].endswith(
                    spec.generation_prompt[Mode.NO_THINK])
                mask = samples.completion_mask(sample)
                assert mask == [0] * start + [1] * (len(ids) - start)
                assert sample["reasoning_tokens"] == len(family.encode_text(reasoning))
                checked += 1
            if checked >= 12:
                break
    assert checked >= 12


def _pair(record: dict, k: int) -> tuple[list[dict], dict]:
    """The (context, target) of turn k of an agent-history record, as the trainer cuts it."""
    pairs, _ = samples._agent_pairs(record)
    return pairs[k]

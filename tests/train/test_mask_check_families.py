"""scripts/train.py's mask check over the real tokenizers of every family whose model folder is present, on
a slice of the real training data with the teacher sidecar (THINK and NO_THINK samples): the trainer's
examples are built as TRL 1.14.1's _prepare_dataset builds them from the pre-tokenized rows (build_labels:
the label is the input id where the completion mask is 1, else -100), and the check, which compares token
ids, passes on every sample of every family. The decoded text is a diagnostic only: under the legacy
SentencePiece tokenizer of Nanbeige4-3B-Thinking-2511 (tokenizer_config.json "legacy": true, a space after
every added token in a decode) it differs from the completion on some sample, where the check that compared
the text refused the run (exit 3); every other family's decode is the completion's text.

The test skips a family whose folder is absent and the whole module when the data or the sidecar is; the
models and data/raw are not in the repository.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from research_agent.data.reasoning import ReasoningIndex
from research_agent.family.core import Thinking, load_family
from research_agent.train.samples import completion_mask

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models"
SYNTH = REPO / "data" / "raw" / "synth" / "synth-m6.jsonl"
PUBLIC = REPO / "data" / "raw" / "public" / "glaive_v2.jsonl"
TEACHER = REPO / "data" / "raw" / "synth" / "reasoning-m7-teacher.jsonl"
FOLDERS = ("Qwen3-1.7B", "Qwen3-4B", "Qwen3-4B-Instruct-2507", "Qwen3-4B-Thinking-2507", "Qwen3.5-4B",
           "granite-4.2-3b", "Nanbeige4-3B-Thinking-2511", "Nanbeige4.1-3B", "MiniCPM5-2B", "Spark-X2.5-4B",
           "LFM2.5-2.6B")
LEGACY = "nanbeige4-thinking-2511"
THINKING_RECORDS, PLAIN_RECORDS = 12, 8  # synth records with a teacher turn; public records
MAX_LENGTH = 5120

pytestmark = pytest.mark.skipif(not (SYNTH.is_file() and PUBLIC.is_file() and TEACHER.is_file()),
                                reason="the training data or the teacher sidecar is not present")


def load_train():
    spec = importlib.util.spec_from_file_location("script_train_masks", REPO / "scripts" / "train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def present(folder: str) -> bool:
    return all((MODELS / folder / name).is_file() for name in ("config.json", "tokenizer_config.json",
                                                               "tokenizer.json"))


def data_slice(folder: Path, index: ReasoningIndex) -> list[Path]:
    """Two data files under folder: the first synth records that hold a teacher turn, and the first public
    records."""
    keys = {rid for rid, _ in index.rows}
    synth, plain = [], []
    with SYNTH.open(encoding="utf-8") as f:
        for line in f:
            if json.loads(line)["id"] in keys:
                synth.append(line)
                if len(synth) == THINKING_RECORDS:
                    break
    with PUBLIC.open(encoding="utf-8") as f:
        for line in f:
            plain.append(line)
            if len(plain) == PLAIN_RECORDS:
                break
    paths = [folder / "synth-slice.jsonl", folder / "public-slice.jsonl"]
    for path, lines in zip(paths, (synth, plain), strict=True):
        path.write_text("".join(lines), encoding="utf-8", newline="\n")
    return paths


def trl_example(sample: dict) -> dict:
    """The example TRL 1.14.1 trains on for a pre-tokenized row (sft_trainer.py _prepare_dataset,
    build_labels under completion_only_loss): the ids, and the labels the mask folds into them."""
    ids = sample["input_ids"]
    return {"input_ids": ids, "labels": [t if m else -100 for t, m in zip(ids, completion_mask(sample),
                                                                         strict=True)]}


@pytest.mark.parametrize("folder", [pytest.param(f, marks=pytest.mark.skipif(
    not present(f), reason="local model files are not present")) for f in FOLDERS])
def test_the_mask_check_passes_every_familys_real_samples_on_their_ids(folder, tmp_path):
    train = load_train()
    family = load_family(MODELS / folder)
    index = ReasoningIndex([TEACHER])
    files = data_slice(tmp_path, index)
    samples, dropped, _ = train.assemble(files, family, MAX_LENGTH, reasoning=index)
    assert dropped.total == 0 and samples
    thinks = family.spec.thinking is not Thinking.NONE
    assert any(s["mode"] == "think" for s in samples) is thinks  # THINK samples where the family thinks
    picks = [(i, s["file"]) for i, s in enumerate(samples)]
    checks = train.mask_check([trl_example(s) for s in samples], family.tokenizer, samples, picks)
    assert all(c["ok"] for c in checks)
    assert all(c["trained_tokens"] == len(s["input_ids"]) - s["completion_start"]
               for c, s in zip(checks, samples, strict=True))
    differs = [c for c in checks if not c["text_equal"]]
    if family.spec.name == LEGACY:  # the old check, which compared the decoded text, refused these
        assert differs and any(samples[c["index"]]["mode"] == "think" for c in differs)
    else:
        assert not differs

"""LoRA fine-tuning of a Qwen3-family model on per-step samples with the S4 settings, recording what the plan
asks for.

The settings reproduce spike S4 (scripts/spikes/s4_train_50_steps.py, docs/m0-验证.md): bf16 LoRA r=16,
alpha=32, dropout 0.05 on the family's LoRA targets (q/k/v/o/gate/up/down for the Qwen3 families), batch 1
with gradient accumulation (16 for the main run, as the plan says), cosine schedule with 3% warm-up,
gradient checkpointing, completion-only loss, no packing, the weights streamed straight to the GPU. --qlora
switches to 4-bit NF4 with the paged 8-bit optimiser. The loss is pinned to TRL's chunked_nll (the lm_head
applied chunk by chunk over the completion tokens), the loss G2 and FT2 resolved to; the manifest says so
under "pinned".

The model folder's family (research_agent.family, docs/m7-家族层设计.md) renders and encodes every sample
and builds the model (load_model: research_agent.family.core.build_causal_lm, with the family's
trust_remote_code, its attention, its model patches and its module adapters, set on vendor code the installed
stack would read or run otherwise before the class is built, Spark's): the family is identified from
config.json and the chat template's hash, --family must name it when given, and an unregistered template
stops the run (exit 2). Each
data file holds records that research_agent.train.samples.load_records reads (transcripts, API histories or
public samples);
every assistant step becomes a pre-tokenized sample ({"input_ids", "completion_mask"}, train.samples.
build_family_samples), so TRL tokenizes nothing, appends nothing and the BOS rule is the family's. A sample is
THINK when a reasoning sidecar (--reasoning, research_agent.data.reasoning) holds the reasoning of its turn
and the family thinks: the prompt opens the thinking block and the completion is reasoning, think_close and
the answer; otherwise it is NO_THINK with the empty block in the prompt, outside the loss. --no-think ignores
the sidecars. Samples over --max-length are dropped and counted, never truncated; a THINK sample over it is
rendered NO_THINK instead and kept (think_fallback). An agent-history record must carry the current
schema_hash and prompt_hash, a sidecar row must be for the training answer, or the run stops (exit 2). The
benchmark-overlap check fails closed: every data file must be listed, by path and current SHA-256, in
--overlap-report (default data/synth/benchmark-overlap.json, written by scripts/check_benchmark_overlap.py),
or the run stops (exit 4) before any sample is built, telling to rerun that script; --no-overlap-check trains
on unlisted files anyway and the manifest records it (data.benchmark_overlap.checked false, the files under
"unchecked"). The records the report names for a data file (13-gram overlap with a BFCL or ACEBench
question) are skipped, and the manifest's data.benchmark_overlap records the report, its SHA-256 and how
many records were skipped.

Smoke and bake-off runs (docs/m7-计划.md §3, §4): --stop-at-step N saves a checkpoint at optimiser step N
and stops, the cosine schedule still spanning the whole run (result.json says stopped_at_step);
--sample-keys FILE keeps only the samples whose keys the file lists, in the file's order (JSON lines with
"id" and "k": a sidecar file qualifies; a listed key without a sample stops the run, exit 2); --smoke-longest
puts the longest samples first and makes the trainer visit the data in that order (train_sampling_strategy
sequential, transformers' own option); --target-modules replaces the family's LoRA targets. Nothing else
touches the sampling: the seed, the data order and the random sampler are FT2's and G2's, so a run on the
same data gets their permutation.

Glossary notes (docs/m7-本地提升计划.md, WP4; WP6 注释): --glossary FILE (data/glossary/zh-en.json,
research_agent.glossary.load_glossary) puts the glossary note on the system message of the steps of every
agent-history record whose id the dropout keeps (research_agent.train.samples, Glossary notes);
--glossary-dropout (default 0.5) is the share of records that train without their notes, decided by a hash of
the record id (NOTE_HASH_RULE). A malformed glossary stops the run (exit 2), and so do an empty --glossary
and a --glossary-dropout without --glossary (it would be ignored), before anything is read. Only a run with
--glossary records the two flags in the manifest's args and data.glossary (the file, its SHA-256 and terms,
the dropout and its hash rule, the samples and records that carry a note); a resume compares the file's
SHA-256 and the dropout.

Continued training (docs/m8-计划.md WP3: FT5 from FT3's step 1635). --init-adapter FOLDER starts the LoRA from
the weights of an existing PEFT adapter instead of a fresh one: FOLDER holds adapter_model.safetensors and
adapter_config.json, as a trainer checkpoint does (runs/train/ft3/trainer/checkpoint-1635). The run builds its
LoRA as always, from its settings, and once the trainer has wrapped the model, the file's weights are copied
into it (peft.set_peft_model_state_dict); every LoRA weight of the run must come from the file and every
tensor of the file must land in one, or the run stops (exit 2) before it trains (load_init_adapter). Nothing
else of the folder is read: the optimiser, its schedule and the RNG start afresh, as this run's settings say.
Before the stack is imported (the dry run too), the folder must hold both files, the weights' header must name
a tensor, and the config must describe the run's LoRA (adapter_problems): peft_type LORA, the run's task type,
r, lora_alpha and target modules (compared as a set; a regex as the string), none of the options a plain LoRA
leaves off (LORA_OFF, bias "none"), and the base model folder --model names (base_model_name_or_path, compared
as a folder); otherwise exit 2. The manifest records config.init_adapter: the folder as given, the SHA-256 of
the weights and of the config, and the tensors. --resume refuses the flag (exit 2, before anything is read): a
resumed run continues from its own checkpoint, which holds the weights it trained.

--reasoning-budget R pins the budget the run records (data.reasoning.budget: the R its THINK checkpoints are
served with, 1,280 for FT5) instead of the plan's p99 rule, which data.reasoning keeps beside it (budget_rule,
with budget_pinned true); a resume compares the pin. The budget is recorded, never trained with.

The two flags are absent from the parsed arguments unless given (argparse.SUPPRESS; read through flag()), so a
run without them records the args, the config and data.reasoning of a run before them, byte for byte.

Checkpoints: --save-steps 0 (default) saves at the end of every epoch; an integer N >= 1 every N optimiser
steps; a value in (0, 1) every ceil(ratio x total optimiser steps) steps (transformers' own reading of a
ratio). Every checkpoint is kept (save_total_limit None); the manifest records the resolved cadence.

--resume continues the run in an existing --out folder. Before the training stack is imported it requires
--out/manifest.json and compares the data files' SHA-256s, the settings, the family (template and tokenizer
hashes, when the manifest records them), the sidecars and the sample keys, and the benchmark-overlap gate
(the report, its SHA-256, checked or not) with it (any difference: exit 5, the differing keys printed), then
resumes from the last checkpoint under --out/trainer (none: exit 5). A manifest from before the loss was
pinned is compared on the loss its run resolved to (sft_resolved). It writes manifest-resume-<k>.json instead
of overwriting manifest.json, and a resume event to log.jsonl.

--dry-run runs the overlap gate, the sample assembly (with the record checks), the over-length and mode
counts and the mask-check selection with the family of --model, prints a summary (records, samples, tokens,
drops and modes per file; the samples the family refused, by kind: template, token boundary, BOS, which a
real run stops at; the reasoning rows and token percentiles with the budget R the plan's rule picks; the
optimiser steps the run would take; the save cadence; a final "summary:" line of JSON with the same numbers)
and exits 0, without importing torch (it is blocked for the duration) or loading the model; nothing is
written.

Written to --out: manifest.json (before training: config with the family record, data hashes and row
counts, the sidecars, git commit, versions, GPU, package freeze, total optimiser steps, save cadence),
mask_check.json (for the first --mask-check samples of every data file, compared by token id: the
trainer's example holds the sample's ids, exactly the completion's positions carry loss, each labelled with
its own id; the run stops if not. The decoded loss-bearing tokens are kept as a diagnostic, which a legacy
SentencePiece tokenizer decodes with a space after every added token), log.jsonl (an {"event": "start"}
or {"event": "resume", ...} row per training segment, then one row per logging step: loss, grad_norm,
learning_rate, num_tokens, tokens/s, peak VRAM so far, max_steps, eta_s), result.json (peak VRAM, dropped
samples, modes, throughput, final_loss: the mean training loss of the last training segment, of the whole
run when it was never resumed; the training segments with their steps, tokens and seconds;
stopped_at_step), adapter/ (the PEFT adapter) and trainer/ (checkpoints). The training stack is imported
inside main() only, so the argument parsing and the settings can be tested without a GPU.

Exit codes:
  0  done (or a dry run that passed)
  2  usage or missing input: bad arguments, no data file, no model folder or an unregistered family, a
     malformed or refused record or sidecar, a sample the family refuses, no sample within --max-length, a
     listed sample key without a sample, --stop-at-step beyond the run, an --init-adapter folder that does
     not fit the run (or --init-adapter with --resume)
  3  mask check failed: the tokens that carry loss are not exactly the completion's
  4  overlap gate: a data file the benchmark-overlap report does not cover at its current SHA-256
  5  resume refused: no manifest, data or settings differ from it, or no checkpoint to resume from
  anything else (a crash, CUDA out of memory, a killed process) is retryable with --resume

Run:    .venv/Scripts/python.exe scripts/train.py data/train/*.jsonl --out runs/train/main --save-steps 0.125
Think:  .venv/Scripts/python.exe scripts/train.py data/train/*.jsonl --out runs/train/ft3 --max-length 5120
        --reasoning data/raw/synth/reasoning-m7-teacher.jsonl data/raw/synth/reasoning-m7-rationalized.jsonl
Resume: .venv/Scripts/python.exe scripts/train.py data/train/*.jsonl --out runs/train/main --save-steps 0.125
        --resume
Check:  .venv/Scripts/python.exe scripts/train.py data/train/*.jsonl --out runs/train/main --dry-run
Smoke:  .venv/Scripts/python.exe scripts/train.py data/train/*.jsonl --out runs/train/smoke --max-steps 20
        --smoke-longest
Continue: .venv/Scripts/python.exe scripts/train.py <data> --out runs/train/ft5 --init-adapter
        runs/train/ft3/trainer/checkpoint-1635 --reasoning-budget 1280 (scripts/make_ft5_keys.py prints FT5's)
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import glob
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import NamedTuple

from research_agent.agent.prompts import PROMPT_HASH, PROMPT_VERSION
from research_agent.canonical import split_lines
from research_agent.data.overlap import load_skips
from research_agent.data.reasoning import ReasoningError, ReasoningIndex, length_stats
from research_agent.family.core import Family, FamilyError, LoadedFamily, build_causal_lm, load_family
from research_agent.glossary import load_glossary
from research_agent.tools.schemas import CONTEXT_TOKENS, SCHEMA_HASH
from research_agent.train.manifest import count_rows, run_manifest, sha256_of, write_manifest
from research_agent.train.samples import (
    NO_THINK,
    NOTE_DROPOUT,
    NOTE_HASH_RULE,
    REFUSAL_KINDS,
    THINK,
    Dropped,
    GlossaryNotes,
    build_family_samples,
    build_step_samples,
    completion_mask,
    load_records,
    reasoning_budget,
    record_name,
)

MODEL = "models/Qwen3-1.7B"
WARMUP = 0.03  # a fraction: transformers 5 takes warmup_steps < 1 as a ratio (docs/m0-验证.md, S4)
GIB = 2**30
OVERLAP_REPORT = "data/synth/benchmark-overlap.json"
LOSS_TYPE = "chunked_nll"
PINNED = {"sft.loss_type": f"{LOSS_TYPE}: the loss TRL 1.14 resolves for completion-only training, the one "
                           "G2 and FT2 trained with (their sft_resolved); pinned so an upgrade cannot change "
                           "it unnoticed (docs/m7-计划.md §3)"}
SETTING_KEYS = ("mode", "model", "lora", "sft")  # what settings() returns; a resume compares these
OVERLAP_KEYS = ("report", "report_sha256", "checked", "unchecked")  # the overlap gate's inputs, also compared
FAMILY_KEYS = ("name", "template_sha256", "tokenizer_json_sha256")  # of the family record, also compared
BLOCKED_IN_DRY_RUN = ("torch",)  # transformers then runs tokenizer-only (its is_torch_available is False)
USAGE, MASK_FAILED, OVERLAP_REFUSED, RESUME_REFUSED = 2, 3, 4, 5
MODES = (THINK, NO_THINK, "think_fallback")  # the mode counts' keys, per file and in all
REFUSED_SHOWN = 5  # refused samples named in the dry run's output
GLOSSARY_ARGS = ("glossary", "glossary_dropout")  # in the manifest's args only when --glossary is given
GLOSSARY_KEYS = ("sha256", "dropout")  # of data.glossary, compared by a resume
# the files of a PEFT adapter folder, as a trainer checkpoint holds them (module docstring: Continued
# training)
ADAPTER_WEIGHTS, ADAPTER_CONFIG = "adapter_model.safetensors", "adapter_config.json"
# adapter_config.json options a plain LoRA leaves off (false, null or empty): the run trains none of them, so
# an adapter with one cannot start it (module docstring: Continued training)
LORA_OFF = ("use_dora", "use_rslora", "use_qalora", "modules_to_save", "rank_pattern", "alpha_pattern",
            "layers_to_transform", "layers_pattern", "exclude_modules", "target_parameters",
            "trainable_token_indices", "fan_in_fan_out", "lora_bias")

LoadFamily = Callable[[str, str | None], LoadedFamily]  # (model folder, --family) -> the loaded family
SampleKey = tuple[str, int]


def save_steps_value(text: str) -> int | float:
    """--save-steps: 0 (per epoch), a ratio in (0, 1), or a whole number of steps >= 1."""
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from exc
    if value < 0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"must be 0, a ratio below 1 or a step count: {text!r}")
    if value >= 1:
        if value != int(value):
            raise argparse.ArgumentTypeError(f"a step count of 1 or more must be whole: {text!r}")
        return int(value)
    return value


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more: {text!r}")
    return value


def share_value(text: str) -> float:
    """--glossary-dropout: a share between 0 and 1."""
    try:
        value = float(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from exc
    if not 0 <= value <= 1:  # also refuses NaN
        raise argparse.ArgumentTypeError(f"must be between 0 and 1: {text!r}")
    return value


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="LoRA fine-tuning with the S4 settings and a full run record.")
    ap.add_argument("data", nargs="+", help="data files (JSONL or JSON); glob patterns are expanded here")
    ap.add_argument("--out", default=f"runs/train/{time.strftime('%Y%m%d-%H%M%S')}",
                    help="run directory (default: runs/train/<timestamp>)")
    ap.add_argument("--model", default=MODEL, help="HF model directory")
    ap.add_argument("--family", default=None,
                    help="the registered family the model folder must resolve to (checked, never assumed)")
    ap.add_argument("--reasoning", nargs="+", default=[], metavar="FILE",
                    help="reasoning sidecars (research_agent.data.reasoning): a turn they hold becomes a "
                         "THINK sample when the family thinks")
    ap.add_argument("--no-think", action="store_true",
                    help="ignore the sidecars: every sample is NO_THINK (the manifest still records them)")
    ap.add_argument("--glossary", default=None, metavar="FILE",
                    help="a zh-en glossary (data/glossary/zh-en.json): agent-history records take its term "
                         "note on their system message, unless the dropout leaves them out")
    ap.add_argument("--glossary-dropout", type=share_value, default=NOTE_DROPOUT, metavar="SHARE",
                    help=f"the share of records that train without their notes, by a hash of the record id "
                         f"(default {NOTE_DROPOUT})")
    ap.add_argument("--max-length", type=int, default=CONTEXT_TOKENS,
                    help="longer samples are dropped, never truncated; a THINK sample over it is rendered "
                         "NO_THINK instead")
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16, help="LoRA rank")
    ap.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    ap.add_argument("--dropout", type=float, default=0.05, help="LoRA dropout")
    ap.add_argument("--target-modules", nargs="+", default=None, metavar="NAME",
                    help="LoRA target modules instead of the family's (the registry's seven projections for "
                         "the Qwen3 families)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--qlora", action="store_true", help="4-bit NF4 base with the paged 8-bit optimiser")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="stop after this many optimiser steps (smoke runs); epochs otherwise")
    ap.add_argument("--stop-at-step", type=positive_int, default=None, metavar="N",
                    help="save a checkpoint at optimiser step N and stop; the schedule still spans the whole "
                         "run (the bake-off's pilots, docs/m7-计划.md §4)")
    ap.add_argument("--sample-keys", default=None, metavar="FILE",
                    help="train on the samples whose keys (record id, assistant ordinal k) this JSON Lines "
                         "file lists, in its order; a sidecar file qualifies")
    ap.add_argument("--smoke-longest", action="store_true",
                    help="the longest samples first, visited in that order (sequential sampling): the "
                         "memory smoke, S3")
    ap.add_argument("--batch-size", type=int, default=1, help="sequences per device per step")
    ap.add_argument("--grad-accum", type=int, default=16, help="gradient accumulation steps")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=save_steps_value, default=0,
                    help="0 = checkpoint at the end of every epoch; N >= 1 = every N optimiser steps; a "
                         "value below 1 = that ratio of the total optimiser steps. Every checkpoint is kept")
    ap.add_argument("--mask-check", type=int, default=5,
                    help="samples per data file whose loss mask is compared with their completion's "
                         "token ids (plan: 5)")
    ap.add_argument("--overlap-report", default=OVERLAP_REPORT,
                    help="skip the records this benchmark-overlap report names, when it exists "
                         f"(default: {OVERLAP_REPORT}); a data file it does not list stops the run")
    ap.add_argument("--no-overlap-check", action="store_true",
                    help="train on data files the overlap report does not list (recorded in the manifest)")
    ap.add_argument("--resume", action="store_true",
                    help="continue the run in --out from its last checkpoint; the data and settings must "
                         "match its manifest.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="check the data and print what the run would do, without torch or the model")
    ap.add_argument("--init-adapter", default=argparse.SUPPRESS, metavar="FOLDER",
                    help="start the LoRA from this PEFT adapter's weights instead of a fresh one (a trainer "
                         "checkpoint folder: runs/train/ft3/trainer/checkpoint-1635); nothing else of it is "
                         "read")
    ap.add_argument("--reasoning-budget", type=positive_int, default=argparse.SUPPRESS, metavar="R",
                    help="the reasoning budget the run records, instead of the plan's p99 rule (recorded "
                         "beside it)")
    return ap


def flag(args: argparse.Namespace, name: str) -> object:
    """An argument that is absent unless given (--init-adapter, --reasoning-budget), or None."""
    return getattr(args, name, None)


class OverlapGateError(ValueError):
    """A data file the benchmark-overlap report does not cover; the run stops before any sample is built."""


class ResumeRefused(ValueError):
    """--resume cannot continue the run: no manifest, a difference from it, or no checkpoint."""


class InitAdapterError(ValueError):
    """An --init-adapter folder whose weights cannot start the run's LoRA (module docstring: Continued
    training)."""


def init_adapter_problem(args: argparse.Namespace) -> str | None:
    """Why --init-adapter cannot be used, or None: an empty path names no folder, and a resumed run continues
    from its own checkpoint, which holds the weights it trained."""
    folder = flag(args, "init_adapter")
    if folder == "":
        return "--init-adapter names no folder (an empty path)"
    if folder is not None and args.resume:
        return ("--init-adapter starts a new run from an adapter's weights, and --resume continues a run "
                "from its own checkpoint, which holds the weights it trained: give one of them")
    return None


def safetensors_names(path: str | Path) -> list[str]:
    """The tensor names a safetensors file's header lists, sorted, read without torch: an 8-byte little-endian
    length, then that many bytes of JSON; InitAdapterError when the header cannot be read."""
    path = Path(path)
    try:
        total = path.stat().st_size
        with path.open("rb") as f:
            size = int.from_bytes(f.read(8), "little")
            if not 0 < size <= total - 8:
                raise ValueError(f"a header of {size} bytes in a file of {total}")
            header = json.loads(f.read(size).decode("utf-8"))
    except (OSError, ValueError) as exc:  # a JSONDecodeError and a UnicodeDecodeError are ValueErrors
        raise InitAdapterError(f"{path.as_posix()} is not a safetensors file: its header cannot be read "
                               f"({exc})") from exc
    if not isinstance(header, dict):
        raise InitAdapterError(f"{path.as_posix()} is not a safetensors file: its header is no JSON object")
    return sorted(name for name in header if name != "__metadata__")


def same_folder(a: str | Path, b: str | Path) -> bool:
    """Whether two spellings name the same folder, relative ones from the current folder, case and separators
    aside where the file system ignores them."""
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def adapter_problems(config: dict, lora: dict, model: str) -> list[str]:
    """Why an adapter_config.json does not describe the run's LoRA (lora: settings()["lora"]) on the base
    model folder model, one line each; [] when it does (module docstring: Continued training)."""
    problems = []
    if config.get("peft_type") != "LORA":
        problems.append(f"peft_type {config.get('peft_type')!r} (a LoRA's is 'LORA')")
    if config.get("task_type") != lora["task_type"]:
        problems.append(f"task_type {config.get('task_type')!r} (the run's {lora['task_type']!r})")
    for key in ("r", "lora_alpha"):
        if config.get(key) != lora[key]:
            problems.append(f"{key} {config.get(key)} (the run's {lora[key]})")
    targets, run_targets = config.get("target_modules"), lora["target_modules"]
    same = targets == run_targets if isinstance(run_targets, str) or isinstance(targets, str) else \
        isinstance(targets, list) and len(set(targets)) == len(targets) and set(targets) == set(run_targets)
    if not same:
        problems.append(f"target_modules {targets!r} (the run's {run_targets!r})")
    problems += [f"{key} {config[key]!r} (the run's LoRA has none)" for key in LORA_OFF if config.get(key)]
    if config.get("bias", "none") != "none":
        problems.append(f"bias {config['bias']!r} (the run's 'none')")
    base = config.get("base_model_name_or_path")
    if not isinstance(base, str) or not base:
        problems.append("no base_model_name_or_path: the base model it was trained on cannot be told")
    elif not same_folder(base, model):
        problems.append(f"base_model_name_or_path {base!r} (the run trains {model!r})")
    return problems


def adapter_record(folder: str | Path, lora: dict, model: str) -> dict:
    """What the manifest records of --init-adapter's folder once it is shown to fit the run (lora:
    settings()["lora"], model: --model): the folder as given, the SHA-256 of its weights and of its config,
    and how many tensors the weights hold. InitAdapterError naming what does not fit."""
    path = Path(folder)
    if not path.is_dir():
        raise InitAdapterError(f"no adapter folder at {path.as_posix()}")
    weights, config_path = path / ADAPTER_WEIGHTS, path / ADAPTER_CONFIG
    for needed in (weights, config_path):
        if not needed.is_file():
            raise InitAdapterError(f"{path.as_posix()} holds no {needed.name}: an adapter folder (a trainer "
                                   f"checkpoint) holds its weights, {ADAPTER_WEIGHTS}, and its "
                                   f"{ADAPTER_CONFIG}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise InitAdapterError(f"{config_path.as_posix()} cannot be read: {exc}") from exc
    problems = adapter_problems(config, lora, model) if isinstance(config, dict) else ["it is no JSON object"]
    if problems:
        raise InitAdapterError(f"{config_path.as_posix()}: the adapter does not fit the run: "
                               + "; ".join(problems))
    names = safetensors_names(weights)
    if not names:
        raise InitAdapterError(f"{weights.as_posix()} holds no tensor")
    return {"path": str(folder), "weights_sha256": sha256_of(weights),
            "config_sha256": sha256_of(config_path), "tensors": len(names)}


def load_init_adapter(model, folder: str | Path, *, adapter_name: str = "default") -> int:
    """Copy --init-adapter's weights into the LoRA of model, the PEFT model the trainer built, and return how
    many tensors the file held (module docstring: Continued training). Every LoRA weight of the adapter
    adapter_name must come from the file and every tensor of the file must land in one: InitAdapterError
    naming the first that do not, or the size that differs (another rank)."""
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file

    weights = Path(folder) / ADAPTER_WEIGHTS
    state = load_file(str(weights))
    lora = {name for name, _ in model.named_parameters() if ".lora_" in name and f".{adapter_name}." in name}
    try:
        result = set_peft_model_state_dict(model, state, adapter_name=adapter_name)
    except RuntimeError as exc:  # torch's load_state_dict: a size mismatch
        raise InitAdapterError(f"{weights.as_posix()} does not fit the run's LoRA: {exc}") from exc
    missing = sorted(lora & set(result.missing_keys))
    if missing:
        raise InitAdapterError(f"{len(missing)} LoRA weight(s) of the run are not in {weights.as_posix()}: "
                               f"{shown_names(missing)}")
    unexpected = sorted(result.unexpected_keys)
    if unexpected:
        raise InitAdapterError(f"{weights.as_posix()} holds {len(unexpected)} weight(s) the run has no LoRA "
                               f"for: {shown_names(unexpected)}")
    return len(state)


def shown_names(names: list[str]) -> str:
    return ", ".join(names[:REFUSED_SHOWN]) + (", ..." if len(names) > REFUSED_SHOWN else "")


def overlap_gate(report_path: str | Path, files: list[Path], *, check: bool) -> tuple[dict[Path, set[int]],
                                                                                     dict]:
    """The record indices to skip per data file and what the manifest records of the overlap check.

    A data file is covered when the report lists it by path (compared resolved) and by its current
    SHA-256. With check on, an uncovered file, or a missing report, is an OverlapGateError telling to rerun
    scripts/check_benchmark_overlap.py; a listed file whose hash changed is load_skips's ValueError (its
    record indices would be stale). With check off (--no-overlap-check) the uncovered files are used and
    named under "unchecked", "checked" is False, and the files the report does cover still skip their
    records.
    """
    report_path = Path(report_path)
    listed: set[tuple[Path, str]] = set()
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        listed = {(Path(e["path"]).resolve(), e["sha256"]) for e in report.get("files", [])}
    unchecked = [str(f) for f in files
                 if (Path(f).resolve(), hashlib.sha256(Path(f).read_bytes()).hexdigest()) not in listed]
    if check and unchecked:
        where = f"{report_path.as_posix()} does not list" if report_path.is_file() else \
            f"there is no benchmark-overlap report at {report_path.as_posix()} for"
        raise OverlapGateError(f"{where} {', '.join(unchecked)} at its current SHA-256; rerun "
                               "scripts/check_benchmark_overlap.py over every training file, or pass "
                               "--no-overlap-check to train on it unchecked (recorded in the manifest)")
    skips, info = load_skips(report_path, files)
    info = info or {"report": None, "report_sha256": None, "skipped_records": 0}
    return skips, {**info, "checked": check, "unchecked": unchecked}


def lora_targets(family: Family) -> list[str] | str:
    """The family's LoRA targets as PEFT takes them: module names as a list, or one regex as a string."""
    targets = family.lora_targets
    return targets if isinstance(targets, str) else list(targets)


def settings(args: argparse.Namespace, family: Family) -> dict:
    """The LoRA and SFT settings as plain dicts: what the manifest records and the trainer is built from.

    The LoRA targets are --target-modules or the family's. The loss is pinned (PINNED). --smoke-longest adds
    the one sampling setting this script ever sets, transformers' sequential strategy; otherwise the
    sampling keys stay unset, as in FT2 and G2. save_steps below 1 is a ratio of the total optimiser steps
    under the "steps" strategy (transformers' TrainerState.compute_steps); 0 is the "epoch" strategy (its
    save_steps then unused).
    """
    out = Path(args.out)
    targets = list(args.target_modules) if args.target_modules else lora_targets(family)
    lora = {"r": args.r, "lora_alpha": args.alpha, "lora_dropout": args.dropout, "task_type": "CAUSAL_LM",
            "target_modules": targets}
    sft = {"output_dir": str(out / "trainer"), "num_train_epochs": args.epochs,
           "max_steps": args.max_steps if args.max_steps else -1,
           "per_device_train_batch_size": args.batch_size, "gradient_accumulation_steps": args.grad_accum,
           "learning_rate": args.lr, "lr_scheduler_type": "cosine", "warmup_steps": WARMUP, "bf16": True,
           "gradient_checkpointing": True, "gradient_checkpointing_kwargs": {"use_reentrant": False},
           "logging_steps": args.logging_steps,
           "save_strategy": "steps" if args.save_steps else "epoch", "save_steps": args.save_steps or 500,
           "save_total_limit": None,
           "max_length": args.max_length, "packing": False, "completion_only_loss": True,
           "loss_type": LOSS_TYPE,
           "dataloader_num_workers": 0, "report_to": "none", "seed": args.seed,
           "optim": "paged_adamw_8bit" if args.qlora else "adamw_torch"}
    if args.smoke_longest:
        sft["train_sampling_strategy"] = "sequential"
    mode = "qlora-4bit" if args.qlora else "bf16-lora"
    return {"mode": mode, "model": args.model, "lora": lora, "sft": sft}


def steps_per_epoch(samples: int, *, batch_size: int, grad_accum: int) -> int:
    """Optimiser steps per epoch as transformers' Trainer.set_initial_training_values counts them."""
    batches = math.ceil(samples / batch_size)
    return max(batches // grad_accum + int(batches % grad_accum > 0), 1)


def total_optimizer_steps(samples: int, *, batch_size: int, grad_accum: int, epochs: float,
                          max_steps: int | None) -> int:
    """The optimiser steps of the run: max_steps when given, else ceil(epochs x steps per epoch)."""
    if max_steps and max_steps > 0:
        return max_steps
    return math.ceil(epochs * steps_per_epoch(samples, batch_size=batch_size, grad_accum=grad_accum))


def save_cadence(save_steps: float, total_steps: int, *, steps_per_epoch: int) -> dict:
    """The resolved checkpoint cadence: every how many steps, and how many checkpoints in all.

    A ratio becomes ceil(total x ratio) steps (TrainerState.compute_steps); the steps strategy also saves at
    the last step (DefaultFlowCallback), and the epoch strategy at the end of every (partial) epoch, so there
    are ceil(total / every) checkpoints either way.
    """
    if not save_steps:
        strategy, every = "epoch", steps_per_epoch
    else:
        strategy, every = "steps", math.ceil(total_steps * save_steps) if save_steps < 1 else int(save_steps)
    return {"save_steps": save_steps, "strategy": strategy, "every_steps": every,
            "checkpoints": math.ceil(total_steps / every)}


def expand_data_files(patterns: list[str]) -> list[Path]:
    """Patterns expanded and sorted (PowerShell expands no globs); one without a match is an error."""
    files: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(f"no data file matches {pattern!r}")
        files += [Path(m) for m in matches]
    return files


def sidecar_files(paths: list[str]) -> list[dict]:
    """What the manifest records of each sidecar: path, SHA-256 and rows (non-blank lines)."""
    return [{"path": str(p), "sha256": sha256_of(p), "rows": count_rows(p)} for p in paths]


def load_sample_keys(path: str | Path) -> list[SampleKey]:
    """The keys a --sample-keys file lists, in its order: JSON Lines, each an object with "id" (the record
    name) and "k" (the assistant ordinal), as a sidecar row is. A malformed line or a key listed twice is a
    ValueError naming the line."""
    keys: list[SampleKey] = []
    seen: dict[SampleKey, int] = {}
    for number, line in enumerate(split_lines(Path(path).read_text(encoding="utf-8")), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {number} is not JSON: {exc}") from exc
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or type(row.get("k")) is not int:
            raise ValueError(f"{path}: line {number} is not a sample key (an object with a string \"id\" and "
                             "an integer \"k\")")
        key = (row["id"], row["k"])
        if key in seen:
            raise ValueError(f"{path}: line {number} lists {key!r} again (first on line {seen[key]})")
        seen[key] = number
        keys.append(key)
    return keys


def select_samples(samples: list[dict], keys: list[SampleKey]) -> list[dict]:
    """The samples whose key is listed, in the keys' order (several samples of one key keep their order);
    ValueError naming the first listed keys that have no sample."""
    by_key: dict[SampleKey, list[dict]] = defaultdict(list)
    for sample in samples:
        by_key[tuple(sample["key"])].append(sample)
    missing = [key for key in keys if key not in by_key]
    if missing:
        shown = ", ".join(repr(key) for key in missing[:REFUSED_SHOWN])
        raise ValueError(f"--sample-keys lists {len(missing)} key(s) without a sample (dropped, skipped or "
                         f"not in the data): {shown}{', ...' if len(missing) > REFUSED_SHOWN else ''}")
    return [sample for key in keys for sample in by_key[key]]


def longest_first(samples: list[dict]) -> list[dict]:
    """The samples by token count, the longest first; equal lengths keep their order."""
    return sorted(samples, key=lambda s: -s["tokens"])


def pre_tokenized(samples: list[dict]) -> list[dict]:
    """The dataset rows the trainer takes, in order: ids and the completion mask, nothing to tokenize."""
    return [{"input_ids": s["input_ids"], "completion_mask": completion_mask(s)} for s in samples]


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, dict) and value:
        flat: dict[str, object] = {}
        for key, item in value.items():
            flat |= _flatten(item, f"{prefix}.{key}" if prefix else str(key))
        return flat
    return {prefix: value}


def resume_differences(args: argparse.Namespace, files: list[Path], family: LoadedFamily) -> list[str]:
    """What differs between this invocation and --out/manifest.json, one line per key; [] when nothing.

    Compared: the data files (resolved path and current SHA-256), settings(args, family) (the resume flag is
    no setting; a manifest without sft.loss_type, from before the loss was pinned, is compared on the loss
    its run resolved to, sft_resolved.loss_type), the family (FAMILY_KEYS of its record, when the manifest
    holds one: a run before the family layer recorded none and is compared on its model path), the
    sidecars (resolved path and SHA-256, and whether they were ignored) and the sample-keys file (its
    SHA-256), which decide the samples and their order, the glossary (GLOSSARY_KEYS of data.glossary: the
    file's SHA-256 and the dropout, when the manifest or the invocation has one), which decides the samples'
    prompts, and the benchmark-overlap gate (OVERLAP_KEYS: the report, its SHA-256, checked or not, the
    files left unchecked), which decides the records skipped and so the samples and the step count. A gate
    that refuses the files is left to overlap_gate, which stops the run right after (exit 4). ResumeRefused
    when there is no manifest to compare with.
    """
    path = Path(args.out) / "manifest.json"
    if not path.is_file():
        raise ResumeRefused(f"there is no {path.as_posix()} to resume from; start the run without --resume")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    then = {str(Path(f["path"]).resolve()): f["sha256"] for f in manifest.get("data_files", [])}
    now = {str(Path(f).resolve()): sha256_of(f) for f in files}
    lines = [f"data_files[{p}]: {then.get(p, 'absent')} -> {now.get(p, 'absent')}"
             for p in sorted(then.keys() | now.keys()) if then.get(p) != now.get(p)]
    config = manifest.get("config", {})
    recorded = _flatten({key: config.get(key) for key in SETTING_KEYS})
    if "sft.loss_type" not in recorded and (config.get("sft_resolved") or {}).get("loss_type") is not None:
        recorded["sft.loss_type"] = config["sft_resolved"]["loss_type"]
    current = _flatten(json.loads(json.dumps(settings(args, family.spec))))
    lines += [f"{key}: {recorded.get(key, 'absent')!r} -> {current.get(key, 'absent')!r}"
              for key in sorted(recorded.keys() | current.keys()) if recorded.get(key) != current.get(key)]
    then_family, record = config.get("family"), family.record()
    if then_family is not None:
        lines += [f"family.{key}: {then_family.get(key, 'absent')!r} -> {record[key]!r}"
                  for key in FAMILY_KEYS if then_family.get(key, "absent") != record[key]]
    data = config.get("data") or {}
    then_reasoning = data.get("reasoning") or {}
    then_sidecars = {str(Path(f["path"]).resolve()): f["sha256"] for f in then_reasoning.get("files", [])}
    now_sidecars = {str(Path(p).resolve()): sha256_of(p) for p in args.reasoning}
    lines += [f"reasoning[{p}]: {then_sidecars.get(p, 'absent')} -> {now_sidecars.get(p, 'absent')}"
              for p in sorted(then_sidecars.keys() | now_sidecars.keys())
              if then_sidecars.get(p) != now_sidecars.get(p)]
    if then_reasoning and then_reasoning.get("ignored") != args.no_think:
        lines.append(f"reasoning.ignored: {then_reasoning.get('ignored')!r} -> {args.no_think!r}")
    then_pin = then_reasoning.get("budget") if then_reasoning.get("budget_pinned") else None
    now_pin = flag(args, "reasoning_budget")
    if then_pin != now_pin:  # a run before the flag pinned none
        lines.append(f"reasoning.budget (--reasoning-budget): {then_pin!r} -> {now_pin!r}")
    then_keys = (data.get("sample_keys") or {}).get("sha256")
    now_keys = sha256_of(args.sample_keys) if args.sample_keys else None
    if then_keys != now_keys:
        lines.append(f"sample_keys.sha256: {then_keys!r} -> {now_keys!r}")
    then_glossary, now_glossary = data.get("glossary") or {}, {}
    if args.glossary is not None:  # a missing file reads as no SHA-256: a difference, or the load's exit 2
        now_glossary = {"sha256": sha256_of(args.glossary) if Path(args.glossary).is_file() else None,
                        "dropout": args.glossary_dropout}
    lines += [f"glossary.{key}: {then_glossary.get(key)!r} -> {now_glossary.get(key)!r}"
              for key in GLOSSARY_KEYS if then_glossary.get(key) != now_glossary.get(key)]
    try:
        _, gate = overlap_gate(args.overlap_report, files, check=not args.no_overlap_check)
    except ValueError:
        return lines
    then_gate = data.get("benchmark_overlap") or {}
    lines += [f"benchmark_overlap.{key}: {then_gate.get(key, 'absent')!r} -> {gate.get(key)!r}"
              for key in OVERLAP_KEYS if then_gate.get(key, "absent") != gate.get(key)]
    return lines


def default_last_checkpoint(folder: str) -> str | None:
    from transformers.trainer_utils import get_last_checkpoint

    return get_last_checkpoint(folder)


def resume_checkpoint(out: Path, *, find_last: Callable[[str], str | None] = default_last_checkpoint
                      ) -> tuple[str, int]:
    """The last checkpoint under out/trainer and its global step; ResumeRefused when there is none."""
    trainer_dir = Path(out) / "trainer"
    found = find_last(str(trainer_dir)) if trainer_dir.is_dir() else None
    if found is None:
        raise ResumeRefused(f"no checkpoint under {trainer_dir.as_posix()} to resume from")
    state = Path(found) / "trainer_state.json"
    if state.is_file():
        step = int(json.loads(state.read_text(encoding="utf-8"))["global_step"])
    else:
        step = int(Path(found).name.rsplit("-", 1)[-1])
    return found, step


def resume_manifest_path(out: Path) -> Path:
    """out/manifest-resume-<k>.json with the first k from 1 not yet taken."""
    k = 1
    while (Path(out) / f"manifest-resume-{k}.json").exists():
        k += 1
    return Path(out) / f"manifest-resume-{k}.json"


def mode_counts(samples: list[dict]) -> dict[str, int]:
    """How many samples are THINK, NO_THINK, and NO_THINK because their THINK rendering was too long (a
    string-path sample, which has no mode, is NO_THINK: it was rendered with thinking off)."""
    counts = Counter(s.get("mode", NO_THINK) for s in samples)
    return {THINK: counts[THINK], NO_THINK: counts[NO_THINK],
            "think_fallback": sum(1 for s in samples if s.get("think_fallback", False))}


def refusal_counts(refused: list[dict]) -> dict[str, int]:
    counts = Counter(r["kind"] for r in refused)
    return {kind: counts[kind] for kind in REFUSAL_KINDS}


def assemble(files: list[Path], renderer: LoadedFamily | object, max_length: int,
             skips: dict[Path, set[int]] | None = None, *, reasoning: ReasoningIndex | None = None,
             refused: list[dict] | None = None,
             notes: GlossaryNotes | None = None) -> tuple[list[dict], Dropped, list[dict]]:
    """All samples of the data files in order (each with its data file under "file"), the drop counts,
    and a per-file account: records, samples, drops, overlap skips, tokens, modes and refusals.

    renderer is the loaded family the run trains through (build_family_samples: pre-tokenized samples with
    their keys and modes), or a tokenizer alone (train.samples.Tokenizer): then the samples are
    render_sample's prompt, completion and token count, the string path FT2 trained on, which
    research_agent.data.assemble's manifest check and the benchmark-overlap tests compare the trainer's
    counts with; such samples carry no ids, keys or modes. skips maps a data file's resolved path to the
    indices of its records to leave out (load_skips reads them from the benchmark-overlap report); per_file
    counts them under "skipped_overlap". reasoning is the sidecar index the samples' modes come from (None:
    every sample NO_THINK). A record refused (a malformed record, a stale sidecar row, a sample the family
    refuses) is a ValueError naming the file and the record; with a refused list the family's refusals go
    there instead (train.samples: Refusals) and the run goes on, the dry run's way. notes is --glossary's
    option (train.samples: Glossary notes); every sample then carries its "note".
    """
    samples: list[dict] = []
    dropped = Dropped()
    per_file = []
    for path in files:
        records = load_records(path)
        skip = (skips or {}).get(Path(path).resolve(), set())
        file_samples: list[dict] = []
        file_dropped = Dropped()
        file_refused: list[dict] = []
        for number, record in enumerate(records):
            if number in skip:
                continue
            try:
                if isinstance(renderer, LoadedFamily):
                    built, drops = build_family_samples(
                        record, renderer, name=record_name(record, path, number), reasoning=reasoning,
                        max_length=max_length, refused=None if refused is None else file_refused, notes=notes)
                else:
                    built, drops = build_step_samples(record, renderer, max_length=max_length, notes=notes)
            except ValueError as exc:
                raise type(exc)(f"{path}: record {number + 1}: {exc}") from exc
            for sample in built:
                sample["file"] = str(path)
            file_samples += built
            file_dropped += drops
        per_file.append({"path": str(path), "records": len(records), "samples": len(file_samples),
                         "dropped": dataclasses.asdict(file_dropped),
                         "skipped_overlap": len(skip & set(range(len(records)))),
                         "tokens": sum(s["tokens"] for s in file_samples), "modes": mode_counts(file_samples),
                         "refused": refusal_counts(file_refused)})
        samples += file_samples
        dropped += file_dropped
        if refused is not None:
            refused += file_refused
    return samples, dropped, per_file


def reasoning_summary(sidecars: list[dict], samples: list[dict], *, ignored: bool,
                      pinned: int | None = None) -> dict:
    """What the manifest records under data.reasoning: the sidecars (sidecar_files), whether they were
    ignored (--no-think), their rows in all, the token lengths of the reasoning the THINK samples train on
    (length_stats) and the budget R the plan's rule picks from their p99 (None without a THINK sample). A
    budget pinned by --reasoning-budget replaces it, the rule's pick kept as budget_rule beside budget_pinned
    true."""
    lengths = [s["reasoning_tokens"] for s in samples if s["reasoning_tokens"] is not None]
    stats = length_stats(lengths)
    summary = {"files": sidecars, "ignored": ignored, "rows": sum(f["rows"] for f in sidecars),
               "tokens": stats, "budget": reasoning_budget(stats["p99"]) if lengths else None}
    if pinned is not None:
        summary |= {"budget": pinned, "budget_rule": summary["budget"], "budget_pinned": True}
    return summary


def data_summary(samples: list[dict], dropped: Dropped, per_file: list[dict], overlap: dict | None, *,
                 reasoning: dict | None = None, sample_keys: dict | None = None) -> dict:
    """What the manifest records under "data": files, samples, drops, token statistics, the mode counts,
    the reasoning summary, the sample-keys file and the benchmark-overlap report with the number of records
    it made the run skip (None without a report)."""
    lengths = [s["tokens"] for s in samples]
    tokens = {"min": min(lengths), "mean": round(sum(lengths) / len(lengths)), "max": max(lengths),
              "total": sum(lengths)} if lengths else None
    skipped = sum(f.get("skipped_overlap", 0) for f in per_file)
    benchmark = None if overlap is None else {"report": overlap["report"],
                                              "report_sha256": overlap["report_sha256"], "skipped": skipped,
                                              "checked": overlap.get("checked", True),
                                              "unchecked": overlap.get("unchecked", [])}
    return {"files": per_file, "samples": len(samples), "dropped": dataclasses.asdict(dropped),
            "tokens": tokens, "modes": mode_counts(samples), "reasoning": reasoning,
            "sample_keys": sample_keys, "benchmark_overlap": benchmark}


def glossary_summary(path: str | Path, notes: GlossaryNotes, samples: list[dict]) -> dict:
    """What the manifest records under data.glossary when --glossary is given: the file, its SHA-256 and
    terms, the dropout and its hash rule, and how many of the samples the run trains on carry a note, of how
    many records."""
    noted = [s for s in samples if s.get("note")]
    return {"path": str(path), "sha256": sha256_of(path), "terms": len(notes.glossary),
            "dropout": notes.dropout, "hash_rule": NOTE_HASH_RULE,
            "records_noted": len({s["key"][0] for s in noted if "key" in s}), "samples_noted": len(noted)}


def recorded_args(args: argparse.Namespace) -> dict:
    """vars(args) as the manifest records them: without the glossary flags (GLOSSARY_ARGS) unless --glossary
    is given, so a run without a glossary records the args a run before the flags recorded. --init-adapter and
    --reasoning-budget are in vars(args) only when given (build_parser), and so here."""
    return {key: value for key, value in vars(args).items()
            if args.glossary is not None or key not in GLOSSARY_ARGS}


def glossary_flags_problem(args: argparse.Namespace, argv: list[str] | None) -> str | None:
    """Why the glossary flags cannot be used, or None: an empty --glossary names no file (and would read as
    no glossary in one place and as one in another), and --glossary-dropout without --glossary would be
    ignored. Whether --glossary-dropout was given is read the way argparse reads the command line
    (abbreviations and --flag=value too): argv parsed again with no default for it."""
    if args.glossary == "":
        return "--glossary names no file (an empty path)"
    if args.glossary is None:
        probe = build_parser()
        probe.set_defaults(glossary_dropout=None)
        if probe.parse_args(argv).glossary_dropout is not None:
            return "--glossary-dropout needs --glossary: without a glossary there is no note to drop"
    return None


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8",
                    newline="\n")


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def mask_check_picks(samples: list[dict], count: int) -> list[tuple[int, str]]:
    """(dataset index, data file) of the first count samples of every data file, in dataset order."""
    picks, taken = [], Counter()
    for index, sample in enumerate(samples):
        file = sample["file"]
        if taken[file] < count:
            taken[file] += 1
            picks.append((index, file))
    return picks


def mask_check(dataset, tokenizer, samples: list[dict], picks: list[tuple[int, str]]) -> list[dict]:
    """The loss-bearing positions of the picked examples, compared with their samples by token id: the
    trainer's example holds the sample's ids, its loss falls on exactly the completion's positions (from
    completion_start to the end) and nowhere else, and each of those positions is labelled with its own id.
    TRL 1.14 prepares "labels" with -100 on the positions without loss (older versions kept a
    completion_mask, whose ones are the loss-bearing positions).

    The decoded loss-bearing ids are a diagnostic only ("text_equal", "head"): a legacy SentencePiece
    tokenizer (Nanbeige4-3B-Thinking-2511's) decodes a space after every added token, so its decode of the
    completion's ids is not the completion's text although the ids are the ones the family encoded it to.
    """
    checks = []
    for i, file in picks:
        example, sample = dataset[i], samples[i]
        ids, expected = list(example["input_ids"]), list(sample["input_ids"])
        start = sample["completion_start"]
        if "labels" in example:
            marks = list(example["labels"])
            positions = [p for p, label in enumerate(marks) if label != -100]
            labelled = all(marks[p] == ids[p] for p in positions if p < len(ids))
        else:
            marks = list(example["completion_mask"])
            positions = [p for p, m in enumerate(marks) if m]
            labelled = True
        decoded = tokenizer.decode([ids[p] for p in positions if p < len(ids)])
        ids_equal = ids == expected
        positions_equal = positions == list(range(start, len(expected)))
        labelled = labelled and len(marks) == len(ids)
        checks.append({"index": i, "file": file, "ok": ids_equal and positions_equal and labelled,
                       "ids_equal": ids_equal, "positions_equal": positions_equal, "labels_equal": labelled,
                       "completion_start": start, "trained_tokens": len(positions), "total_tokens": len(ids),
                       "text_equal": decoded.strip() == sample["completion"].strip(), "head": decoded[:120]})
    return checks


def jsonl_logger(path: Path, torch, *, base: type | None = None,
                 clock: Callable[[], float] = time.perf_counter):
    """A TrainerCallback appending one JSON line per logging step: the logs plus tokens/s, peak VRAM,
    max_steps and eta_s (remaining steps x the mean seconds per step of this segment so far).

    base is the callback class (transformers.TrainerCallback unless given); clock is for the tests.
    """
    if base is None:
        from transformers import TrainerCallback as base

    class JsonlLogger(base):
        def __init__(self) -> None:
            self.t0 = self.t_last = None
            self.tokens_last = 0
            self.start_step = 0

        def on_train_begin(self, args, state, control, **kwargs):
            self.t0 = self.t_last = clock()
            self.tokens_last = 0  # TRL's num_tokens restarts at 0 in every segment
            self.start_step = state.global_step  # the checkpoint's step when resuming

        def on_log(self, args, state, control, logs=None, **kwargs):
            now = clock()
            if self.t0 is None:
                self.t0 = self.t_last = now
            row = {"step": state.global_step, "elapsed_s": round(now - self.t0, 1), **(logs or {})}
            tokens = row.get("num_tokens")  # TRL's cumulative count of trained tokens
            if tokens is not None:
                seconds = now - self.t_last
                row["tokens_per_s"] = round((tokens - self.tokens_last) / seconds) if seconds > 0 else None
                self.tokens_last, self.t_last = tokens, now
            if torch.cuda.is_available():
                row["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / GIB, 3)
                row["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved() / GIB, 3)
            done = state.global_step - self.start_step
            row["max_steps"] = state.max_steps
            row["eta_s"] = round(max(state.max_steps - state.global_step, 0) * (now - self.t0) / done, 1) \
                if done > 0 else None
            append_jsonl(path, row)

    return JsonlLogger()


def stop_at_step_callback(step: int, *, base: type | None = None):
    """A TrainerCallback that, at the end of optimiser step `step`, asks the trainer to save a checkpoint
    and to stop; the schedule is untouched, so the learning rate at that step is the full run's. The
    checkpoint is written once: the Trainer still closes the epoch after the stop (trainer.py runs
    on_epoch_end and the save step after the loop breaks), where the default flow callback asks for the
    epoch save (the save strategy here unless --save-steps is given) of the same global step into the same
    folder; this callback runs after it in the handler, having been added later, and withdraws that
    request. Its `stopped` tells whether it fired. base is the callback class (transformers.TrainerCallback
    unless given).
    """
    if base is None:
        from transformers import TrainerCallback as base

    class StopAtStep(base):
        def __init__(self) -> None:
            self.step = step
            self.stopped = False

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step >= step and not self.stopped:
                self.stopped = True
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            if self.stopped:  # the step's save is done; the epoch save would write it a second time
                control.should_save = False
            return control

    return StopAtStep()


def segments_from_log(rows: list[dict], *, final_step: int, final_seconds: float) -> list[dict]:
    """The training segments of the run from its log.jsonl rows, the last one ending now.

    The run's rows start at its last {"event": "start"}; every {"event": "resume"} row starts a segment at
    the checkpoint's step. An earlier segment counts only up to the step the next one resumed from (later
    steps were trained again): its tokens and seconds are those of its last logging row at or before that
    step (TRL's num_tokens restarts at 0 in every segment, sft_trainer.py sets _total_train_tokens = 0 in
    __init__). The last segment ends at final_step after final_seconds.
    """
    starts = [i for i, r in enumerate(rows) if r.get("event") == "start"]
    rows = rows[starts[-1] + 1:] if starts else rows
    segments: list[dict] = [{"start_step": 0, "rows": []}]
    for row in rows:
        if row.get("event") == "resume":
            segments.append({"start_step": int(row["global_step"]), "rows": []})
        elif "event" not in row:
            segments[-1]["rows"].append(row)
    result = []
    for k, segment in enumerate(segments):
        last = k == len(segments) - 1
        end = final_step if last else segments[k + 1]["start_step"]
        counted = [r for r in segment["rows"] if last or r.get("step", 0) <= end]
        tokens = max((int(r["num_tokens"]) for r in counted if r.get("num_tokens") is not None), default=0)
        seconds = final_seconds if last else (counted[-1].get("elapsed_s", 0.0) if counted else 0.0)
        result.append({"start_step": segment["start_step"], "end_step": end, "tokens": tokens,
                       "seconds": round(float(seconds), 1)})
    return result


def train_and_report(trainer, torch, out: Path, *, callback, checkpoint: str | None, checkpoint_step: int,
                     args: argparse.Namespace, setup: dict, samples: int, dropped: Dropped, data: dict,
                     checks: list[dict], stopper=None) -> dict:
    """Train (resuming from checkpoint when given), save the adapter and write result.json.

    log.jsonl gets an {"event": "start"} row, or {"event": "resume", "from_checkpoint", "global_step"},
    before training; result.json's "segments" and "trained_tokens" add up every segment of the run. stopper
    is the --stop-at-step callback (stop_at_step_callback): result.json's stopped_at_step is its step when
    it fired, None otherwise.
    """
    log = out / "log.jsonl"
    for added in (callback, stopper):
        if added is not None:
            trainer.add_callback(added)
    if checkpoint is None:
        append_jsonl(log, {"event": "start"})
    else:
        append_jsonl(log, {"event": "resume", "from_checkpoint": checkpoint, "global_step": checkpoint_step})
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=checkpoint)
    seconds = time.perf_counter() - t0
    adapter_dir = out / "adapter"
    trainer.model.save_pretrained(adapter_dir)

    # transformers divides the losses summed in this train() call by the global step, which also counts the
    # steps before the checkpoint resumed from (trainer.py: _total_loss_scalar / effective_global_step), so a
    # resumed segment's training_loss is diluted; undone here: final_loss is the mean loss of this segment
    trained_steps = trainer.state.global_step - checkpoint_step
    final_loss = None
    if trained_steps > 0:
        final_loss = result.training_loss * trainer.state.global_step / trained_steps
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    segments = segments_from_log(rows, final_step=trainer.state.global_step, final_seconds=seconds)
    trained_tokens = sum(s["tokens"] for s in segments)
    total_seconds = sum(s["seconds"] for s in segments)
    sft = setup["sft"]
    sequences = trainer.state.global_step * sft["gradient_accumulation_steps"] * \
        sft["per_device_train_batch_size"]
    estimate = sequences * data["tokens"]["mean"]
    peak_alloc, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    report = {
        "mode": setup["mode"], "max_length": args.max_length, "steps": trainer.state.global_step,
        "stopped_at_step": stopper.step if stopper is not None and stopper.stopped else None,
        "sequences": sequences, "samples": samples, "dropped": dataclasses.asdict(dropped),
        "modes": data.get("modes"),
        "mean_tokens": data["tokens"]["mean"], "max_tokens": data["tokens"]["max"],
        "seconds": round(total_seconds, 1), "trained_tokens": trained_tokens or estimate,
        "tokens_per_s": round((trained_tokens or estimate) / total_seconds) if total_seconds > 0 else None,
        "tokens_per_s_source": "trl num_tokens" if trained_tokens else "estimate (sequences x mean tokens)",
        "peak_alloc_gib": round(peak_alloc / GIB, 2), "peak_reserved_gib": round(peak_reserved / GIB, 2),
        "peak_reserved_gb": round(peak_reserved / 1e9, 2),
        "final_loss": None if final_loss is None else round(final_loss, 4),
        "mask_ok": [c["ok"] for c in checks], "adapter": str(adapter_dir), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "segments": segments,
    }
    write_json(out / "result.json", report)
    return report


@contextlib.contextmanager
def modules_blocked(names: tuple[str, ...]) -> Iterator[None]:
    """Make importing the named modules fail inside (sys.modules[name] = None), unless already loaded."""
    added = [name for name in names if name not in sys.modules]
    for name in added:
        sys.modules[name] = None
    try:
        yield
    finally:
        for name in added:
            if name in sys.modules and sys.modules[name] is None:
                del sys.modules[name]


def default_load_family(model_dir: str, name: str | None) -> LoadedFamily:
    """The hook main() loads the family through unless a test gives another (LoadFamily)."""
    return load_family(model_dir, name=name)


def load_model(model_dir: str, family: LoadedFamily, *, torch, quant) -> object:
    """The base model the run trains: bf16, quant (the NF4 settings of --qlora, or None) and the weights
    streamed straight to the GPU instead of staging a full copy in system RAM (S4), built through the family
    (research_agent.family.core.build_causal_lm: its trust_remote_code, its attention, sdpa but for Spark's
    eager-only vendor code, and its module adapters and model patches set on the vendor code before the class
    is built: Spark's mask calls and its decoder layer's cast, research_agent.family.compat)."""
    return build_causal_lm(family.spec, model_dir, dtype=torch.bfloat16, quantization_config=quant,
                           device_map={"": 0})


def plan_summary(args: argparse.Namespace, samples: int) -> tuple[int, dict]:
    """The total optimiser steps of a run on this many samples and its resolved save cadence."""
    per_epoch = steps_per_epoch(samples, batch_size=args.batch_size, grad_accum=args.grad_accum)
    total = total_optimizer_steps(samples, batch_size=args.batch_size, grad_accum=args.grad_accum,
                                  epochs=args.epochs, max_steps=args.max_steps)
    return total, save_cadence(args.save_steps, total, steps_per_epoch=per_epoch)


def cadence_text(cadence: dict) -> str:
    if cadence["strategy"] == "epoch":
        return f"save at the end of every epoch ({cadence['every_steps']} steps, {cadence['checkpoints']} " \
               "checkpoints, save_steps 0)"
    return f"save every {cadence['every_steps']} steps (save_steps {cadence['save_steps']}), " \
           f"{cadence['checkpoints']} checkpoints"


def modes_text(modes: dict) -> str:
    return ", ".join(f"{key} {modes[key]}" for key in MODES)


def refused_text(refused: dict) -> str:
    return ", ".join(f"{kind} {refused[kind]}" for kind in REFUSAL_KINDS)


def glossary_text(summary: dict) -> str:
    return (f"glossary: {summary['terms']} terms ({summary['path']}), dropout {summary['dropout']}: "
            f"{summary['samples_noted']} samples of {summary['records_noted']} records carry a note")


def reasoning_text(summary: dict) -> str:
    stats = summary["tokens"]
    pinned = summary.get("budget_pinned", False)
    if not stats["n"]:
        budget = f"budget R {summary['budget']} (pinned by --reasoning-budget)" if pinned else "budget none"
        return (f"reasoning: {len(summary['files'])} sidecar(s), {summary['rows']} rows, "
                f"{'ignored (--no-think)' if summary['ignored'] else 'no THINK sample'}; {budget}")
    rule = "max(1024, p99 up to a multiple of 256), at most 2048"
    budget = (f"budget R {summary['budget']} (pinned by --reasoning-budget; the rule, {rule}, picks "
              f"{summary['budget_rule']})" if pinned else f"budget R {summary['budget']} ({rule})")
    return (f"reasoning: {len(summary['files'])} sidecar(s), {summary['rows']} rows; THINK samples "
            f"{stats['n']}, reasoning tokens p50 {stats['p50']}, p90 {stats['p90']}, p95 {stats['p95']}, "
            f"p99 {stats['p99']}, max {stats['max']}; {budget}")


def init_adapter_text(record: dict) -> str:
    return (f"init adapter: {record['path']}: {record['tensors']} tensors ({ADAPTER_WEIGHTS} sha256 "
            f"{record['weights_sha256']}) start the run's LoRA; its optimizer, schedule and RNG state are "
            "not read")


class Prepared(NamedTuple):
    """The samples of a run after assembly, selection and ordering, with what the manifest records."""

    samples: list[dict]
    dropped: Dropped
    per_file: list[dict]
    data: dict
    refused: list[dict]


def prepare(args: argparse.Namespace, family: LoadedFamily, data_files: list[Path],
            skips: dict[Path, set[int]], overlap: dict | None, *, index: ReasoningIndex | None,
            keys: list[SampleKey] | None, refused: list[dict] | None = None,
            notes: GlossaryNotes | None = None) -> Prepared:
    """Assemble the samples, apply --sample-keys and --smoke-longest, and summarise; ValueError for what
    stops a run (assemble's refusals, a listed key without a sample, no sample within --max-length). The
    per-file account is the assembly's, before --sample-keys selects; data.samples and the mode counts
    are of the samples the run trains on, in their final order, and so is data.glossary, which only a run
    with notes (--glossary) has."""
    samples, dropped, per_file = assemble(data_files, family, args.max_length, skips=skips,
                                          reasoning=None if args.no_think else index, refused=refused,
                                          notes=notes)
    if not samples:
        raise ValueError(f"no sample within {args.max_length} tokens; dropped {dataclasses.asdict(dropped)}")
    keys_info = None
    if keys is not None:
        samples = select_samples(samples, keys)
        keys_info = {"path": args.sample_keys, "sha256": sha256_of(args.sample_keys), "listed": len(keys),
                     "selected": len(samples)}
    if args.smoke_longest:
        samples = longest_first(samples)
    reasoning = reasoning_summary(sidecar_files(args.reasoning), samples, ignored=args.no_think,
                                  pinned=flag(args, "reasoning_budget"))
    data = data_summary(samples, dropped, per_file, overlap, reasoning=reasoning, sample_keys=keys_info)
    if notes is not None:
        data["glossary"] = glossary_summary(args.glossary, notes, samples)
    return Prepared(samples, dropped, per_file, data, refused or [])


def dry_run(args: argparse.Namespace, family: LoadedFamily, data_files: list[Path],
            skips: dict[Path, set[int]], overlap: dict, *, index: ReasoningIndex | None,
            keys: list[SampleKey] | None, notes: GlossaryNotes | None = None,
            init: dict | None = None) -> int:
    """Assemble the samples with the family of --model and print what the run would do (module
    docstring); torch is blocked by the caller. init is --init-adapter's record (adapter_record)."""
    try:
        prepared = prepare(args, family, data_files, skips, overlap, index=index, keys=keys, refused=[],
                           notes=notes)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return USAGE
    data, samples = prepared.data, prepared.samples
    for f in prepared.per_file:
        print(f"{f['path']}: records {f['records']}, samples {f['samples']}, dropped over_length "
              f"{f['dropped']['over_length']}, after_repair {f['dropped']['after_repair']}, skipped_overlap "
              f"{f['skipped_overlap']}, tokens {f['tokens']}, {modes_text(f['modes'])}"
              + (f", refused {refused_text(f['refused'])}" if any(f["refused"].values()) else ""))
    total, cadence = plan_summary(args, len(samples))
    picks = mask_check_picks(samples, args.mask_check)
    refused = refusal_counts(prepared.refused)
    print(f"total: records {sum(f['records'] for f in prepared.per_file)}, samples {len(samples)}, tokens "
          f"{data['tokens']['total']}, dropped {dataclasses.asdict(prepared.dropped)}")
    print(f"modes: {modes_text(data['modes'])}")
    print(reasoning_text(data["reasoning"]))
    if "glossary" in data:
        print(glossary_text(data["glossary"]))
    print(f"refused: {refused_text(refused)}")
    for item in prepared.refused[:REFUSED_SHOWN]:
        print(f"  {item['kind']} {item['key']!r}: {item['error'][:160]}")
    if data["sample_keys"] is not None:
        print(f"sample keys: {data['sample_keys']['listed']} listed, {data['sample_keys']['selected']} "
              f"samples selected, in the file's order ({args.sample_keys})")
    if args.smoke_longest:
        print("smoke: the longest samples first, visited in that order (train_sampling_strategy sequential)")
    if init is not None:
        print(init_adapter_text(init))
    print(f"total optimizer steps {total} ({args.epochs:g} epochs, batch {args.batch_size} x accumulation "
          f"{args.grad_accum}{'' if not args.max_steps else ', --max-steps'}); {cadence_text(cadence)}")
    if args.stop_at_step is not None:
        print(f"stop at step {args.stop_at_step} of {total}"
              + (" (beyond the run: a real run stops with exit 2)" if args.stop_at_step > total else ""))
    print(f"mask check: {len(picks)} samples ({args.mask_check} per file)")
    print("dry run: nothing written, torch not imported")
    summary = {"samples": len(samples), "tokens": data["tokens"], "dropped": data["dropped"],
               "modes": data["modes"], "refused": refused, "reasoning": data["reasoning"],
               "sample_keys": data["sample_keys"], "total_optimizer_steps": total, "save_cadence": cadence,
               "family": family.record()}
    if "glossary" in data:
        summary["glossary"] = data["glossary"]
    if init is not None:
        summary["init_adapter"] = init
    print("summary: " + json.dumps(summary, ensure_ascii=False, default=str))
    return 0


def main(argv: list[str] | None = None, *, load_family: LoadFamily | None = None) -> int:
    """load_family stands in for research_agent.family.core.load_family in the tests (default_load_family)."""
    args = build_parser().parse_args(argv)
    problem = glossary_flags_problem(args, argv) or init_adapter_problem(args)
    if problem is not None:
        print(f"FAIL: {problem}", file=sys.stderr)
        return USAGE
    out = Path(args.out)
    try:
        data_files = expand_data_files(args.data)
    except FileNotFoundError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return USAGE
    if not Path(args.model).is_dir():
        print(f"FAIL: no model folder at {args.model}", file=sys.stderr)
        return USAGE
    with modules_blocked(BLOCKED_IN_DRY_RUN if args.dry_run else ()):
        try:
            family = (load_family or default_load_family)(args.model, args.family)
        except FamilyError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return USAGE
        if args.resume:
            try:
                differences = resume_differences(args, data_files, family)
            except ResumeRefused as exc:
                print(f"RESUME REFUSED: {exc}", file=sys.stderr)
                return RESUME_REFUSED
            if differences:
                print("RESUME REFUSED: this invocation differs from the run's manifest.json in" +
                      "".join(f"\n  {line}" for line in differences), file=sys.stderr)
                return RESUME_REFUSED
        try:
            skips, overlap = overlap_gate(args.overlap_report, data_files, check=not args.no_overlap_check)
        except ValueError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return OVERLAP_REFUSED
        if overlap["unchecked"]:
            print(f"benchmark overlap NOT checked for {', '.join(overlap['unchecked'])} (--no-overlap-check)",
                  flush=True)
        try:
            index = ReasoningIndex(args.reasoning) if args.reasoning else None
            keys = load_sample_keys(args.sample_keys) if args.sample_keys else None
            notes = None
            if args.glossary is not None:
                notes = GlossaryNotes(load_glossary(args.glossary), args.glossary_dropout)
        except (ReasoningError, ValueError, OSError) as exc:  # a GlossaryError is a ValueError
            print(f"FAIL: {exc}", file=sys.stderr)
            return USAGE
        init = None
        if flag(args, "init_adapter") is not None:
            try:
                init = adapter_record(args.init_adapter, settings(args, family.spec)["lora"], args.model)
            except InitAdapterError as exc:
                print(f"FAIL: {exc}", file=sys.stderr)
                return USAGE
        if args.dry_run:
            return dry_run(args, family, data_files, skips, overlap, index=index, keys=keys, notes=notes,
                           init=init)
    setup = settings(args, family.spec)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    checkpoint, checkpoint_step = None, 0
    if args.resume:
        try:
            checkpoint, checkpoint_step = resume_checkpoint(out)
        except ResumeRefused as exc:
            print(f"RESUME REFUSED: {exc}", file=sys.stderr)
            return RESUME_REFUSED
        print(f"resuming from {checkpoint} (step {checkpoint_step})", flush=True)
    out.mkdir(parents=True, exist_ok=True)

    try:
        prepared = prepare(args, family, data_files, skips, overlap, index=index, keys=keys, notes=notes)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return USAGE
    samples, dropped, data = prepared.samples, prepared.dropped, prepared.data
    if overlap["report"] is not None:
        print(f"benchmark overlap: {data['benchmark_overlap']['skipped']} records skipped "
              f"({overlap['report']})", flush=True)
    total_steps, cadence = plan_summary(args, len(samples))
    if args.stop_at_step is not None and args.stop_at_step > total_steps:
        print(f"FAIL: --stop-at-step {args.stop_at_step} is beyond the run's {total_steps} optimizer steps",
              file=sys.stderr)
        return USAGE
    cfg = SFTConfig(**setup["sft"])
    peft_cfg = LoraConfig(**setup["lora"])
    config = {**setup, "family": family.record(), "pinned": PINNED, "args": recorded_args(args), "data": data,
              "schema_hash": SCHEMA_HASH, "prompt_version": PROMPT_VERSION, "prompt_hash": PROMPT_HASH,
              "total_optimizer_steps": total_steps, "save_cadence": cadence,
              "resume": None if checkpoint is None else {"from_checkpoint": checkpoint,
                                                         "global_step": checkpoint_step},
              **({} if init is None else {"init_adapter": init}),
              "sft_resolved": json.loads(cfg.to_json_string())}
    manifest = run_manifest(config, data_files)
    manifest_path = resume_manifest_path(out) if args.resume else out / "manifest.json"
    write_manifest(manifest, manifest_path)
    print(f"samples {len(samples)} (tokens min {data['tokens']['min']}, mean {data['tokens']['mean']}, "
          f"max {data['tokens']['max']}), dropped {dataclasses.asdict(dropped)}, "
          f"{modes_text(data['modes'])}, {total_steps} optimizer steps, {cadence_text(cadence)}, commit "
          f"{manifest['git_commit']}, manifest {manifest_path}", flush=True)
    print(reasoning_text(data["reasoning"]), flush=True)
    if "glossary" in data:
        print(glossary_text(data["glossary"]), flush=True)

    quant = None
    if args.qlora:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = load_model(args.model, family, torch=torch, quant=quant)
    dataset = Dataset.from_list(pre_tokenized(samples))
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=dataset, processing_class=family.tokenizer,
                         peft_config=peft_cfg)
    if init is not None:  # the LoRA the trainer built starts from the adapter's weights (Continued training)
        try:
            tensors = load_init_adapter(trainer.model, args.init_adapter)
        except InitAdapterError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return USAGE
        print(f"init adapter: {tensors} tensors loaded from {args.init_adapter} into the run's LoRA (its "
              "optimizer, schedule and RNG state not read)", flush=True)

    checks = mask_check(trainer.train_dataset, family.tokenizer, samples,
                        mask_check_picks(samples, args.mask_check))
    write_json(out / "mask_check.json", checks)
    for c in checks:
        print(f"mask check {c['index']} ({c['file']}): {'ok' if c['ok'] else 'MISMATCH'} "
              f"({c['trained_tokens']} of {c['total_tokens']} tokens carry loss) -> {c['head'][:80]!r}"
              + ("" if c["text_equal"] else " (decodes to other text than the completion; ids decide)"),
              flush=True)
    if not all(c["ok"] for c in checks):
        print("mask check failed: the tokens that carry loss are not exactly the completion's; not training",
              file=sys.stderr)
        return MASK_FAILED

    stopper = None if args.stop_at_step is None else stop_at_step_callback(args.stop_at_step)
    report = train_and_report(trainer, torch, out, callback=jsonl_logger(out / "log.jsonl", torch),
                              checkpoint=checkpoint, checkpoint_step=checkpoint_step, args=args, setup=setup,
                              samples=len(samples), dropped=dropped, data=data, checks=checks,
                              stopper=stopper)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""LoRA fidelity between the training stack and llama.cpp: the bake-off's smoke S6 (docs/m7-计划.md §4,
docs/m7-底座对比预登记.md 资格 S6).

Why. llama.cpp applies a GGUF adapter to the tensors its converter mapped, and a LoRA target it does not map
(a fused projection, a linear-attention module, a tensor name the GGUF schema lacks) is dropped without a
word: a candidate trained on such targets would be evaluated with part of its adapter missing. The check, for
each target group of the family (target_groups: one group per module name of its lora_targets; for a family
that targets its layers with one PEFT regex, Qwen3.5's, one group per kind of module the regex matches, so
that a kind llama.cpp drops fails on its own instead of hiding among the kinds it applies):
a synthetic LoRA (random A and B under a fixed seed, scaled so that the module's outputs change by about a
fraction of their size, the fraction calibrated per group: The strength) is saved as a PEFT adapter, exported
to GGUF, and the top-1 next-token predictions of the training stack (the base in NF4 with the adapter,
teacher forced over the sequences) are compared with llama.cpp's (the Q8_0 GGUF with the adapter at scale 1,
one greedy token per position) at POSITIONS positions of each of the development items.

Pass (the preregistration), per group, every bound inclusive: the agreement with the adapter is at least
AGREEMENT; it is at least the agreement without any adapter minus MARGIN; and the adapter changes at least
CHANGED of the training stack's own top-1 tokens against no adapter (otherwise the comparison proves
nothing). A group also fails, with the reason and without a comparison, when the training stack did not load
its adapter (The load check) or when no strength up to the cap changes CHANGED of those tokens (The
strength). The report holds the counts behind every rate.

The modules (docs/m7-进度.md, 底座对比队列的两处工具问题). PEFT matches its targets against the module names
of the model it wraps, the one scripts/train.py and the HF backend build through the family (core.
build_causal_lm), and those need not be the checkpoint's names: Qwen3.5-4B's checkpoint is the
vision-language model's (model.language_model.layers.N.*), while AutoModelForCausalLM builds the text model
Qwen3_5ForCausalLM (model.layers.N.*), transformers renaming the checkpoint's tensors as it loads them. An
adapter named after the checkpoint is still matched by the family's regex, which allows both names, so PEFT
wraps the modules, finds none of the adapter's tensors, leaves every lora_B at zero and only warns: in C3's
first smoke every group changed 0.000. The modules are therefore those of the model as build_causal_lm builds
it, instantiated on the meta device (model_layout: the family's vendor patches, trust_remote_code and
attention, then AutoModelForCausalLM.from_config, which resolves the class and the configuration exactly as
from_pretrained does, no weight read): every torch.nn.Linear with its shape. A target selects among them as
PEFT selects (select_modules: a regex string matched against the whole name, a list of names matching a
module of that name or one whose name ends with "." and it). The base weight whose norm scales a module's
LoRA is the checkpoint tensor transformers loads into it, found by transformers' own renaming of the
checkpoint's names (conversion_mapping.get_model_conversion_mapping and core_model_loading.rename_source_key,
as from_pretrained applies them) and read from the file that holds it (_shards: every file the index lists,
read whole, as from_pretrained reads them), its shape checked against the module's. llama.cpp's converter
maps either naming to the same blk.N tensors (C3's trained adapter, model.layers.N.*, exported to its 400
tensors).

The load check. Right after PeftModel.from_pretrained, every lora_A and lora_B tensor of the adapter file
must be a parameter of the model holding the file's values (in the parameter's dtype), and every LoRA
parameter of the model must come from the file (verify_adapter_loaded); otherwise AdapterNotLoaded, which
counts and names what is missing, extra or different, the adapter is unloaded again, and the group fails with
that reason. PEFT itself only warns about the keys it could not place.

The scale. PEFT applies alpha / r x B A; with A's entries drawn from N(0, 1 / in) and B's from N(0, s^2),
the expected Frobenius norm of B A is s x sqrt(out x r), so s = fraction x ||W||_F x r / (alpha x
sqrt(out x r)) makes the expected norm of alpha / r x B A fraction times the base weight's (for an isotropic
input the relative change of the outputs is about the ratio of the norms).

The strength. A relative change of FRACTION on one kind of module need not move CHANGED of the top-1 tokens,
and that third condition is what gives the comparison its power: llama.cpp dropping or misapplying the adapter
shows only at the positions the adapter moves. Were the adapter ignored, llama.cpp would predict its base
tokens, and the agreement would be at most 1 - changed + (1 - the base agreement): with changed 0.30 and C3's
base agreement of 0.95, at most 0.75, under AGREEMENT. So each group's strength is calibrated on the training
stack alone, every group before llama.cpp predicts anything (calibrate): from FRACTION, multiplied by GROWTH
per step up to CAP (strengths), the group's adapter is written and the training stack's top-1 tokens are
compared with its own predictions without an adapter; the first fraction that changes at least AIM of them
(CHANGED with a margin) is the group's; failing that, of the fractions that changed at least CHANGED the one
that changed the most (the smallest of equals: the change need not grow with the strength); failing that too,
the group fails with that reason. Only then is the adapter at that fraction exported and compared with
llama.cpp's predictions, once. Why this keeps the agreement test independent: the fraction is a function of
the training stack's predictions alone (with and without the adapter), so no llama.cpp output takes part in
choosing it, and the comparison runs once per group at a strength fixed beforehand, with no choosing among
strengths by the test's outcome, which would bias the agreement upwards. The condition the calibration makes
hold is the third, a property of the training stack alone that the preregistration requires so that the test
can fail; the two agreement conditions are untouched. The training stack's predictions at the chosen fraction
are the calibration's own (the same model, adapter and sequences), and every trial is in the report.

What runs where. The arithmetic (compare), the verdict, the module selection (select_modules), the
sequences, the adapter (synthetic_lora, write_adapter: the safetensors file is written here, F32, so no torch
is needed to make one) and the orchestration with the calibration (run_fidelity, calibrate) are plain Python,
tested in CI over fake backends. The model parts are the Backends' callables: the modules of a target and
their weight norms (base_modules, weight_norm over model_layout: torch and transformers on the meta device,
the checkpoint read with numpy), the export (scripts/export_adapter.py, run by the caller's runner), the
training stack's predictions (hf_top1_backend: torch, transformers, peft, bitsandbytes on the GPU, the model
built through the folder's family as scripts/train.py builds it, its attention, model patches and module
adapters included, each adapter's load checked) and llama.cpp's (llama_top1_backend: a llama-server per
adapter through research_agent.llm.server.LocalServer, one slot, each top-1 read from the ids the server says
it generated, generated_top1).
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import struct
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ..agent.context import build_context
from ..canonical import canonical_json
from ..family.core import FamilyError, LoadedFamily, Mode, build_causal_lm, folder_family, patch_model_class
from ..llm.local import Completion, LocalClient
from ..llm.server import LocalServer
from ..tools.schemas import MODEL_TOOLS
from ..train.samples import render_family_sample
from .lock import sha256_of

AGREEMENT = 0.90  # the least top-1 agreement with the adapter
MARGIN = 0.05  # how far below the no-adapter agreement the adapter's may fall
CHANGED = 0.30  # the least share of the training stack's top-1 tokens the adapter must change
FRACTION = 0.07  # the strength the calibration starts at: the relative change of a module's outputs
GROWTH = 2.0  # the calibration's factor from one strength to the next
CAP = 2.0  # the largest strength the calibration tries: the LoRA's norm twice the base weight's
AIM = 0.35  # the share of the training stack's top-1 tokens the calibration aims to change: CHANGED + 0.05
ITEMS = 20  # development items of the comparison
POSITIONS = 16  # teacher-forced positions per item: the first tokens of the gold answer
MODE = Mode.NO_THINK  # the gold answers carry no reasoning: the sequences are the NO_THINK renders
INDEX = "model.safetensors.index.json"
SINGLE = "model.safetensors"
ADAPTER_WEIGHTS = "adapter_model.safetensors"
GGUF_NAME = "adapter-f16.gguf"  # a group's exported adapter, beside its adapter folder
ADAPTER_NAME = "default"  # the name PeftModel.from_pretrained gives the adapter it loads
_LAYER = re.compile(r"\.(\d+)\.")
_KIND = re.compile(r"\.\d+\.(.+)$")  # a module's path after its layer index
_DTYPES = {"F32": "<f4", "F16": "<f2"}  # safetensors dtypes numpy reads as they are; BF16 is widened
_LORA_TENSOR = re.compile(r"(?P<module>.+)\.lora_(?P<which>[AB])\.weight")  # a tensor of an adapter file
_LORA_PARAMETER = re.compile(r"(?P<module>.+)\.lora_(?P<which>[AB])\.(?P<adapter>[^.]+)\.weight")


@dataclass(frozen=True)
class Module:
    """One linear module of the base model the LoRA targets: its name without ".weight" and its shape."""

    name: str
    in_features: int
    out_features: int


@dataclass(frozen=True)
class Sequence:
    """A teacher-forced sequence: its ids and the positions compared, ids[start : start + positions],
    each predicted from the ids before it."""

    ids: list[int]
    start: int
    positions: int


@dataclass(frozen=True)
class Layout:
    """The model build_causal_lm builds for a folder, as model_layout sees it on the meta device: its Linear
    modules in the model's order, and per module the checkpoint tensor transformers loads into its weight
    as it is (module name -> tensor name; a weight assembled from several tensors, one no tensor or more
    than one loads into, or one tied to another is not in it)."""

    modules: tuple[Module, ...]
    tensors: Mapping[str, str]


Predict = Callable[[Path | None, list[Sequence]], list[list[int]]]  # (adapter or None, sequences) -> top-1
Target = str | tuple[str, ...]  # a PEFT target: a regex matched whole, or module names (select_modules)


@dataclass(frozen=True)
class Backends:
    """The model parts of the check (module docstring: What runs where)."""

    modules: Callable[[Target], list[Module]]  # a PEFT target -> the base's modules it selects
    weight_norm: Callable[[Module], float]  # the Frobenius norm of a module's base weight
    export: Callable[[Path, Path], Path]  # (adapter folder, outfile) -> the GGUF written
    hf_top1: Predict  # the training stack's predictions, with the PEFT adapter folder or without
    llama_top1: Predict  # llama.cpp's predictions, with the GGUF adapter or without


class AdapterNotLoaded(RuntimeError):
    """The training stack's model does not hold an adapter file's LoRA tensors as the file has them
    (module docstring: The load check); the message says what is missing, extra or different."""


# ---- the arithmetic ---------------------------------------------------------------------------------------


def compare(hf_lora: list[list[int]], llama_lora: list[list[int]], hf_base: list[list[int]],
            llama_base: list[list[int]]) -> dict:
    """The agreement and change counts of one group and the verdict (module docstring: Pass); ValueError when
    the four arrays do not hold the same items and positions, or no position at all."""
    arrays = (hf_lora, llama_lora, hf_base, llama_base)
    if len({len(a) for a in arrays}) != 1:
        raise ValueError(f"the four prediction arrays hold different numbers of items: "
                         f"{[len(a) for a in arrays]}")
    for index in range(len(hf_lora)):
        if len({len(a[index]) for a in arrays}) != 1:
            raise ValueError(f"item {index}: the four prediction arrays hold different numbers of positions: "
                             f"{[len(a[index]) for a in arrays]}")
    positions = sum(len(row) for row in hf_lora)
    if positions == 0:
        raise ValueError("no position to compare")
    agreeing = _equal_positions(hf_lora, llama_lora)
    base_agreeing = _equal_positions(hf_base, llama_base)
    changed = positions - _equal_positions(hf_lora, hf_base)
    agreement, base_agreement, change = agreeing / positions, base_agreeing / positions, changed / positions
    checks = {"agreement": {"value": agreement, "least": AGREEMENT, "passed": agreement >= AGREEMENT},
              "margin": {"value": agreement, "least": base_agreement - MARGIN,
                         "passed": agreement >= base_agreement - MARGIN},
              "changed": {"value": change, "least": CHANGED, "passed": change >= CHANGED}}
    return {"items": len(hf_lora), "positions": positions, "agreeing": agreeing, "agreement": agreement,
            "base_agreeing": base_agreeing, "base_agreement": base_agreement, "changed_positions": changed,
            "changed": change, "checks": checks, "passed": all(c["passed"] for c in checks.values())}


def _equal_positions(a: list[list[int]], b: list[list[int]]) -> int:
    """How many positions hold the same prediction in both arrays."""
    return sum(x == y for row_a, row_b in zip(a, b, strict=True) for x, y in zip(row_a, row_b, strict=True))


# ---- the synthetic adapter --------------------------------------------------------------------------------


def kind_of(module: str) -> str:
    """A module's kind: its path after the layer index (self_attn.q_proj, linear_attn.in_proj_qkv), or the
    last piece of the name of a module outside the layers."""
    match = _KIND.search(module)
    return match.group(1) if match else module.rsplit(".", 1)[-1]


def target_groups(targets: str | tuple[str, ...], modules_of: Callable[[Target], list[Module]]
                  ) -> list[tuple[str, Target, list[Module]]]:
    """The groups a synthetic LoRA goes on one at a time (module docstring), as (name, the adapter's PEFT
    target, the base modules): for a list of module names, one group per name, its target the name alone
    (a tuple, which PEFT reads as names); for a regex, one group per kind of module it matches (kind_of), its
    target the regex narrowed to that kind (it still has to match the whole module name, and the name must
    end with the kind), so that PEFT, the converter and the module selection see the group's modules alone.
    modules_of is asked once per name, or once for the regex. The groups of a regex come in the order of
    their kinds' names. ValueError for a target that selects no module of the base."""
    if isinstance(targets, str):
        modules = modules_of(targets)
        if not modules:
            raise ValueError(f"the target {targets!r} names no module of the base model")
        kinds = sorted({kind_of(m.name) for m in modules})
        return [(kind, f"(?:{targets})(?<=\\.{re.escape(kind)})",
                 [m for m in modules if kind_of(m.name) == kind]) for kind in kinds]
    found = []
    for name in targets:
        modules = modules_of((name,))
        if not modules:
            raise ValueError(f"group {name}: the target {name!r} names no module of the base model")
        found.append((name, (name,), modules))
    return found


def strengths(start: float = FRACTION, growth: float = GROWTH, cap: float = CAP) -> list[float]:
    """The fractions the calibration tries, in order (module docstring: The strength): start, then each
    the one before times growth, the last one cap; ValueError unless 0 < start <= cap and growth > 1."""
    if not 0 < start <= cap or growth <= 1:
        raise ValueError(f"the strengths need 0 < start <= cap and growth > 1, got start {start}, growth "
                         f"{growth}, cap {cap}")
    found = [start]
    while found[-1] < cap:
        found.append(min(found[-1] * growth, cap))
    return found


def synthetic_lora(module: Module, *, r: int, alpha: int, seed: int, weight_norm: float,
                   fraction: float = FRACTION) -> tuple[list[list[float]], list[list[float]]]:
    """A random LoRA for the module in PEFT's shapes, A (r x in) and B (out x r), drawn under the seed and
    the module's name (another module draws other numbers) and scaled as the module docstring says."""
    rng = random.Random(f"{seed}:{module.name}")
    sigma_a = 1.0 / math.sqrt(module.in_features)
    sigma_b = fraction * weight_norm * r / (alpha * math.sqrt(module.out_features * r))
    a = [[rng.gauss(0.0, sigma_a) for _ in range(module.in_features)] for _ in range(r)]
    b = [[rng.gauss(0.0, sigma_b) for _ in range(r)] for _ in range(module.out_features)]
    return a, b


def write_safetensors(path: Path, tensors: dict[str, tuple[tuple[int, ...], list[float]]]) -> None:
    """A safetensors file of F32 tensors, name -> (shape, values in row-major order), the names sorted: a
    u64 header length, the JSON header padded to a multiple of 8 bytes with spaces, the buffers end to end;
    ValueError when a tensor's values do not fill its shape."""
    header: dict[str, dict] = {}
    blobs: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        shape, values = tensors[name]
        if len(values) != math.prod(shape):
            raise ValueError(f"{name}: {len(values)} values do not fill the shape {list(shape)}")
        data = struct.pack(f"<{len(values)}f", *values)
        header[name] = {"dtype": "F32", "shape": list(shape), "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
        blobs.append(data)
    head = json.dumps(header, separators=(",", ":")).encode("utf-8")
    head += b" " * ((8 - len(head) % 8) % 8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))


def read_safetensors_header(path: Path) -> dict[str, dict]:
    """The tensors of a safetensors file: name -> {dtype, shape, data_offsets}, the metadata left out."""
    with Path(path).open("rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(length).decode("utf-8"))
    header.pop("__metadata__", None)
    return header


def write_adapter(folder: Path, modules: list[Module], loras: dict[str, tuple[list[list[float]],
                                                                             list[list[float]]]], *,
                  r: int, alpha: int, targets: Target | list[str], base_model: str) -> Path:
    """The modules' LoRAs as a PEFT adapter folder (adapter_config.json and adapter_model.safetensors with
    PEFT's tensor names, base_model.model.<module>.lora_A.weight and lora_B, and shapes), the folder
    scripts/export_adapter.py and PeftModel.from_pretrained read; the module names must be those of the model
    PEFT wraps (module docstring: The modules). targets is module names or one regex, kept a string as PEFT
    saves one (PEFT reads a list as names, never as a regex)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    config = {"base_model_name_or_path": base_model, "bias": "none", "fan_in_fan_out": False,
              "inference_mode": True, "init_lora_weights": True, "lora_alpha": alpha, "lora_dropout": 0.0,
              "peft_type": "LORA", "r": r,
              "target_modules": targets if isinstance(targets, str) else list(targets),
              "task_type": "CAUSAL_LM"}
    (folder / "adapter_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n",
                                                encoding="utf-8", newline="\n")
    tensors = {}
    for module in modules:
        a, b = loras[module.name]
        tensors[f"base_model.model.{module.name}.lora_A.weight"] = ((r, module.in_features),
                                                                     [x for row in a for x in row])
        tensors[f"base_model.model.{module.name}.lora_B.weight"] = ((module.out_features, r),
                                                                     [x for row in b for x in row])
    write_safetensors(folder / ADAPTER_WEIGHTS, tensors)
    return folder


# ---- the base modules -------------------------------------------------------------------------------------


def _shards(model_dir: Path) -> dict[str, Path]:
    """Tensor name -> the safetensors file holding it, read from the files' own headers: the files the index
    names, or the single file, each read whole, as transformers loads them (it takes the index's list of
    files, not its entry per tensor, which can be wrong: 101 of granite-4.2-3b's 363 entries name the other
    file). FileNotFoundError without an index or a single file; ValueError for a tensor two files hold."""
    model_dir = Path(model_dir)
    index = model_dir / INDEX
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        files = sorted({model_dir / shard for shard in weight_map.values()})
    elif (model_dir / SINGLE).is_file():
        files = [model_dir / SINGLE]
    else:
        raise FileNotFoundError(f"{model_dir}: no {INDEX} and no {SINGLE}")
    found: dict[str, Path] = {}
    for file in files:
        for name in read_safetensors_header(file):
            if name in found:
                raise ValueError(f"{model_dir}: both {found[name].name} and {file.name} hold {name}")
            found[name] = file
    return found


def _layer(name: str) -> int:
    match = _LAYER.search(name)
    return int(match.group(1)) if match else -1


def model_layout(model_dir: str | Path) -> Layout:
    """The modules of the model build_causal_lm builds for the folder and the checkpoint tensor of each
    module's weight (module docstring: The modules), the model instantiated on the meta device: the folder's
    family (folder_family, its vendor code checked), its patches and module adapters set (patch_model_class),
    AutoModelForCausalLM.from_config over the folder's configuration with the family's trust_remote_code and
    attention, which resolves the class from_pretrained resolves, and the model required to be of the
    patched class, as build_causal_lm requires it. FamilyError for a folder of no registered family or a
    model of another class; FileNotFoundError for a folder without safetensors."""
    import torch  # heavy imports, only where a model is instantiated
    import transformers
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key

    folder = str(model_dir)
    spec = folder_family(folder)
    patched = patch_model_class(spec, folder)
    config = transformers.AutoConfig.from_pretrained(folder, trust_remote_code=spec.trust_remote_code)
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(
            config, trust_remote_code=spec.trust_remote_code, attn_implementation=spec.attn_implementation)
    if patched is not None and not isinstance(model, patched):
        raise FamilyError(f"{spec.name}: transformers built {type(model).__name__} from "
                          f"{type(model).__module__}, not the patched "
                          f"{patched.__module__}.{patched.__name__}")
    modules = tuple(Module(name, module.in_features, module.out_features)
                    for name, module in model.named_modules() if isinstance(module, torch.nn.Linear))
    transforms = get_model_conversion_mapping(model)  # what from_pretrained renames the checkpoint's names by
    renamings = [t for t in transforms if isinstance(t, WeightRenaming)]
    converters = [t for t in transforms if isinstance(t, WeightConverter)]
    state, prefix = model.state_dict(), model.base_model_prefix
    loaded: dict[str, list[str]] = {}
    for key in _shards(Path(folder)):
        renamed, converted = rename_source_key(key, renamings, converters, prefix, state)
        if renamed not in state and key in state:  # as from_pretrained: the key as it is, its prefix fixed
            renamed, converted = rename_source_key(key, [], [], base_model_prefix=prefix,
                                                   meta_state_dict=state)
        if renamed in state:
            loaded.setdefault(renamed, []).append(key if converted is None else "")
    tensors = {}
    for module in modules:
        sources = loaded.get(f"{module.name}.weight", [])
        if len(sources) == 1 and sources[0]:
            tensors[module.name] = sources[0]
    return Layout(modules=modules, tensors=tensors)


def select_modules(modules: Iterable[Module], target: Target | list[str]) -> list[Module]:
    """The modules a PEFT target selects, in layer order (then by name), as PEFT's check_target_module_exists
    selects them: a string is a regex matched against the whole module name; a list or tuple holds module
    names, each selecting the module of that name and every module whose name ends with "." and it."""
    if isinstance(target, str):
        found = [m for m in modules if re.fullmatch(target, m.name)]
    else:
        names = tuple(target)
        found = [m for m in modules if m.name in names or any(m.name.endswith(f".{n}") for n in names)]
    return sorted(found, key=lambda m: (_layer(m.name), m.name))


def base_modules(model_dir: str | Path, target: Target | list[str], *, layout: Layout | None = None
                 ) -> list[Module]:
    """The Linear modules of the model build_causal_lm builds for the folder that the PEFT target selects
    (select_modules over model_layout's modules; layout, when given, is the folder's), in layer order."""
    return select_modules((layout or model_layout(model_dir)).modules, target)


def weight_norm(model_dir: str | Path, module: Module, *, layout: Layout | None = None) -> float:
    """The Frobenius norm of the module's base weight: the checkpoint tensor transformers loads into it
    (model_layout's; layout, when given, is the folder's), read from the folder's safetensors; KeyError when
    no checkpoint tensor loads into the module's weight as it is, ValueError when the tensor's shape is not
    the module's (out, in)."""
    tensor = (layout or model_layout(model_dir)).tensors.get(module.name)
    if tensor is None:
        raise KeyError(f"{module.name}: no tensor of {Path(model_dir).as_posix()}'s checkpoint loads into "
                       "its weight as it is")
    return tensor_norm(model_dir, tensor, shape=(module.out_features, module.in_features))


def tensor_norm(model_dir: str | Path, name: str, *, shape: tuple[int, ...] | None = None) -> float:
    """The Frobenius norm of the checkpoint tensor name, read from the folder's safetensors (F32, F16 or
    BF16); KeyError when the folder holds no such tensor, ValueError for another dtype or, when shape is
    given, another shape."""
    import numpy as np  # the only numeric package here, and only for a real model folder

    shard = _shards(Path(model_dir))[name]
    entry = read_safetensors_header(shard)[name]
    if shape is not None and tuple(entry["shape"]) != tuple(shape):
        raise ValueError(f"{name}: the checkpoint's shape {entry['shape']} is not the module's {list(shape)}")
    start, end = entry["data_offsets"]
    with shard.open("rb") as f:
        (length,) = struct.unpack("<Q", f.read(8))
        f.seek(8 + length + start)
        raw = f.read(end - start)
    if entry["dtype"] == "BF16":
        values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    elif entry["dtype"] in _DTYPES:
        values = np.frombuffer(raw, dtype=_DTYPES[entry["dtype"]]).astype(np.float32)
    else:
        raise ValueError(f"{name}: dtype {entry['dtype']} is not one this check reads")
    return float(np.sqrt(np.sum(values.astype(np.float64) ** 2)))


# ---- the load check ---------------------------------------------------------------------------------------


def verify_adapter_loaded(model, adapter: str | Path, *, adapter_name: str = ADAPTER_NAME) -> dict:
    """The load check (module docstring) of the adapter folder's file against the model PEFT returned (its
    named_parameters): each tensor <module>.lora_A.weight or lora_B of the file must be the parameter
    <module>.lora_A.<adapter_name>.weight (or lora_B) holding the file's values cast to the parameter's
    dtype, and each such parameter of the model must come from the file. The counts, {"tensors", "modules"};
    AdapterNotLoaded otherwise, and for a file without LoRA tensors."""
    import torch
    from safetensors.torch import load_file

    path = Path(adapter) / ADAPTER_WEIGHTS
    expected, keys = {}, {}
    for key, tensor in load_file(str(path)).items():
        match = _LORA_TENSOR.fullmatch(key)
        if match is not None:
            name = f"{match['module']}.lora_{match['which']}.{adapter_name}.weight"
            expected[name], keys[name] = tensor, key
    present = {name: parameter for name, parameter in model.named_parameters()
               if (match := _LORA_PARAMETER.fullmatch(name)) is not None and match["adapter"] == adapter_name}
    missing = sorted(keys[name] for name in set(expected) - set(present))
    extra = sorted(set(present) - set(expected))
    differing = [name for name in sorted(set(expected) & set(present))
                 if not torch.equal(present[name].detach().to("cpu"),
                                    expected[name].to(present[name].dtype))]
    if not expected or missing or extra or differing:
        loaded = len(expected) - len(missing) - len(differing)
        problems = [f"{len(names)} {what} (first: {names[0]})" for names, what in (
            (missing, "of the file's tensors are no parameter of the model"),
            (extra, "LoRA parameters of the model have no tensor in the file"),
            (differing, "parameters hold other values than the file")) if names]
        found = "; ".join(problems) or "the file holds none"
        raise AdapterNotLoaded(f"{loaded} of the {len(expected)} LoRA tensors of {path.as_posix()} are in "
                               f"the model with the file's values; {found}")
    return {"tensors": len(expected), "modules": len({name.rsplit(".lora_", 1)[0] for name in expected})}


# ---- the sequences ----------------------------------------------------------------------------------------


def _target(expected: dict) -> dict:
    """An item's expected turn as the API assistant message the family renders."""
    calls = []
    for index, call in enumerate(expected.get("tool_calls") or []):
        arguments = call.get("arguments_json")
        if not isinstance(arguments, str):
            arguments = canonical_json(call.get("arguments") or {})
        calls.append({"id": f"call_{index}", "type": "function",
                      "function": {"name": call["name"], "arguments": arguments}})
    message: dict = {"role": "assistant", "content": expected.get("content") or ""}
    if calls:
        message["tool_calls"] = calls
    return message


def sequences_of(family: LoadedFamily, items: list[dict], positions: int = POSITIONS) -> list[Sequence]:
    """One teacher-forced sequence per item: the family's NO_THINK prompt of the item's context (as the
    runner builds it) followed by the gold answer as the family renders it, cut to the first positions
    tokens of the answer (fewer when the answer is shorter)."""
    if positions < 1:
        raise ValueError(f"positions must be at least 1, got {positions}")
    sequences = []
    for item in items:
        sample = render_family_sample(family, build_context(item["messages"]), _target(item["expected"]),
                                      tools=MODEL_TOOLS)
        ids, start = sample["input_ids"], sample["completion_start"]
        n = min(positions, len(ids) - start)
        sequences.append(Sequence(ids[:start + n], start, n))
    return sequences


# ---- the orchestration ------------------------------------------------------------------------------------


def _rel(path: Path) -> str:
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return absolute.as_posix()


def calibrate(folder: Path, modules: list[Module], target: Target, *, sequences: list[Sequence],
              hf_top1: Predict, hf_base: list[list[int]], norms: Mapping[str, float], r: int, alpha: int,
              seed: int, base_model: str, fractions: list[float], aim: float = AIM
              ) -> tuple[dict, list[list[int]] | None]:
    """A group's strength (module docstring: The strength), on the training stack alone: for each fraction
    in turn, the group's adapter written at folder/adapter and its top-1 predictions compared with hf_base,
    until one changes at least aim of the positions or the fractions run out. The chosen trial: the one that
    reached aim; else, of the trials that changed at least CHANGED, the one that changed the most (the first
    of equals), its adapter written again when a later trial's took its place; else the last. The folder
    holds the chosen trial's adapter. The record, {"trials": [{"fraction", "changed_positions",
    "changed"}...], "fraction", "changed", "calibrated"} (calibrated: the chosen trial changed at least
    CHANGED), with "error" when the training stack did not load an adapter (AdapterNotLoaded; fraction and
    changed are None then), and the predictions at the chosen fraction, None unless calibrated."""
    positions = sum(s.positions for s in sequences)
    trials: list[dict] = []
    record: dict = {"trials": trials, "fraction": None, "changed": None, "calibrated": False}
    predictions: list[list[list[int]]] = []

    def write(fraction: float) -> Path:
        loras = {m.name: synthetic_lora(m, r=r, alpha=alpha, seed=seed, weight_norm=norms[m.name],
                                        fraction=fraction) for m in modules}
        return write_adapter(Path(folder) / "adapter", modules, loras, r=r, alpha=alpha, targets=target,
                             base_model=base_model)

    for fraction in fractions:
        adapter = write(fraction)
        try:
            predictions.append(hf_top1(adapter, sequences))
        except AdapterNotLoaded as exc:
            record["error"] = f"the training stack did not load the adapter at fraction {fraction:g}: {exc}"
            return record, None
        changed = positions - _equal_positions(predictions[-1], hf_base)
        trials.append({"fraction": fraction, "changed_positions": changed, "changed": changed / positions})
        if changed / positions >= aim:
            break
    chosen = len(trials) - 1
    powered = [i for i, trial in enumerate(trials) if trial["changed"] >= CHANGED]
    if trials[chosen]["changed"] < aim and powered:
        chosen = max(powered, key=lambda i: (trials[i]["changed"], -i))
        if chosen != len(trials) - 1:
            write(trials[chosen]["fraction"])  # the last trial's adapter took its place in the folder
    pick = trials[chosen]
    record.update(fraction=pick["fraction"], changed=pick["changed"], calibrated=pick["changed"] >= CHANGED)
    return record, predictions[chosen] if record["calibrated"] else None


def run_fidelity(targets: str | tuple[str, ...], sequences: list[Sequence], backends: Backends, work: Path,
                 *, r: int, alpha: int, seed: int, base_model: str, fraction: float = FRACTION,
                 growth: float = GROWTH, cap: float = CAP, aim: float = AIM) -> dict:
    """The check over every group of the family's targets (module docstring; target_groups), in two phases:
    the training stack's predictions without an adapter and every group's strength calibrated on them
    (calibrate; a GGUF an earlier run left in the group's folder removed first); then llama.cpp's
    predictions without an adapter and, for each calibrated group, its adapter exported to
    work/<group>/adapter-f16.gguf and compared (compare). The report: the settings (the strengths tried
    included), the no-adapter agreement, per group its target, modules, calibration, fraction, files
    (repository-relative paths and SHA-256s; no GGUF for a group not compared), counts, checks, failures and
    verdict, the failed groups and the verdict. ValueError for a target that selects no module of the base,
    sequences without a position, or an aim under CHANGED, before any prediction."""
    found = target_groups(targets, backends.modules)
    positions = sum(s.positions for s in sequences)
    if positions == 0:
        raise ValueError("no position to compare")
    if aim < CHANGED:
        raise ValueError(f"the calibration's aim {aim} is under the least change {CHANGED} it has to reach")
    fractions = strengths(fraction, growth, cap)
    report: dict = {
        "settings": {"r": r, "alpha": alpha, "seed": seed, "fraction": fraction, "growth": growth, "cap": cap,
                     "aim": aim, "fractions": fractions, "items": len(sequences),
                     "positions": max((s.positions for s in sequences), default=0),
                     "least_agreement": AGREEMENT, "margin": MARGIN, "least_changed": CHANGED},
        "groups": {}}
    hf_base = backends.hf_top1(None, sequences)
    calibrated: dict[str, list[list[int]]] = {}
    for group, target, modules in found:
        folder = Path(work) / group
        (folder / GGUF_NAME).unlink(missing_ok=True)  # the folder holds this run's files alone
        norms = {m.name: backends.weight_norm(m) for m in modules}
        calibration, predictions = calibrate(folder, modules, target, sequences=sequences,
                                             hf_top1=backends.hf_top1, hf_base=hf_base, norms=norms, r=r,
                                             alpha=alpha, seed=seed, base_model=base_model,
                                             fractions=fractions, aim=aim)
        adapter = folder / "adapter"
        record = {"target": target if isinstance(target, str) else list(target),
                  "modules": [m.name for m in modules], "calibration": calibration,
                  "fraction": calibration["fraction"], "adapter": _rel(adapter),
                  "adapter_sha256": sha256_of(adapter / ADAPTER_WEIGHTS), "gguf": None, "gguf_sha256": None}
        if predictions is not None:
            calibrated[group] = predictions
        elif "error" in calibration:
            record.update(failures=[calibration["error"]], passed=False)
        else:
            changed = calibration["changed"]
            failure = (f"changed {changed:.3f} at the cap fraction {calibration['fraction']:g} (least "
                       f"{CHANGED:.3f}): no strength up to the cap gives the comparison its power")
            record.update(changed_positions=calibration["trials"][-1]["changed_positions"], changed=changed,
                          failures=[failure], passed=False)
        report["groups"][group] = record
    llama_base = backends.llama_top1(None, sequences) if calibrated else None
    for group, predictions in calibrated.items():
        record = report["groups"][group]
        gguf = backends.export(Path(work) / group / "adapter", Path(work) / group / GGUF_NAME)
        result = compare(predictions, backends.llama_top1(gguf, sequences), hf_base, llama_base)
        failures = [f"{check} {c['value']:.3f} (least {c['least']:.3f})"
                    for check, c in result["checks"].items() if not c["passed"]]
        record.update(gguf=_rel(gguf), gguf_sha256=sha256_of(gguf), **result, failures=failures)
        record["passed"] = record.pop("passed")  # the verdict last
    base_agreeing = None if llama_base is None else _equal_positions(hf_base, llama_base)
    report["base"] = {"items": len(sequences), "positions": positions, "agreeing": base_agreeing,
                      "agreement": None if base_agreeing is None else base_agreeing / positions}
    report["failed"] = [name for name, g in report["groups"].items() if not g["passed"]]
    report["passed"] = bool(report["groups"]) and not report["failed"]
    return report


# ---- the model backends -----------------------------------------------------------------------------------


def hf_top1_backend(model_dir: str | Path, *, trust_remote_code: bool | None = None) -> Predict:
    """The training stack's predictions on the GPU: the base model in NF4 as scripts/train.py loads it, with
    the PEFT adapter folder (PeftModel.from_pretrained, its load checked by verify_adapter_loaded, unloaded
    again afterwards whatever happens, so the base stays as it was) or without, the whole sequence forwarded
    once and the top-1 of the logits at the compared positions read off. The model loads on the first call
    and stays loaded. It is built through the folder's family, found from the folder's files
    (research_agent.family.core.folder_family, its vendor code checked) and built by build_causal_lm with the
    family's trust_remote_code, attention, model patches and module adapters, as training builds it;
    trust_remote_code, when given, must be the family's (ValueError otherwise). AdapterNotLoaded when the
    model does not hold the adapter as its file has it."""
    state: dict = {}

    def load():
        import torch
        from transformers import BitsAndBytesConfig

        spec = folder_family(model_dir)
        if trust_remote_code is not None and trust_remote_code != spec.trust_remote_code:
            raise ValueError(f"{model_dir}: trust_remote_code {trust_remote_code} is not the family "
                             f"{spec.name}'s ({spec.trust_remote_code})")
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = build_causal_lm(spec, str(model_dir), dtype=torch.bfloat16, quantization_config=quant,
                                device_map={"": 0})
        return model.eval()

    def predict(adapter: Path | None, sequences: list[Sequence]) -> list[list[int]]:
        import torch

        base = state.get("model") or load()
        state["model"] = base
        model = base
        if adapter is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(base, str(adapter)).eval()
        try:
            if adapter is not None:
                verify_adapter_loaded(model, adapter)
            out = []
            with torch.no_grad():
                for s in sequences:
                    ids = torch.tensor([s.ids], dtype=torch.long, device=model.device)
                    logits = model(input_ids=ids).logits[0]
                    top = logits[s.start - 1:s.start - 1 + s.positions].argmax(dim=-1)
                    out.append([int(t) for t in top.tolist()])
        finally:
            if adapter is not None:
                state["model"] = model.unload()  # PEFT wrapped the base's modules in place
        return out

    return predict


def llama_top1_backend(binary: Path, base_gguf: Path, *, port: int = 8080, popen=subprocess.Popen,
                       client: Callable[[str], LocalClient] = LocalClient, timeout: float = 120.0,
                       log: Path | None = None) -> Predict:
    """llama.cpp's predictions: one llama-server per call (LocalServer: the base GGUF, the adapter GGUF when
    given, one slot), and for each position one greedy request of one token over the ids before it, at lora
    scale 1 with an adapter and without a lora field on a bare server; the generated token is the top-1
    (generated_top1)."""

    def predict(gguf: Path | None, sequences: list[Sequence]) -> list[list[int]]:
        adapter = None if gguf is None else Path(gguf).as_posix()
        with LocalServer(binary.as_posix(), model=Path(base_gguf).as_posix(), adapter=adapter, port=port,
                         timeout=timeout, log=log, popen=popen, client=client, extra=["-np", "1"]) as server:
            scale = 0.0 if gguf is None else 1.0
            out = []
            for s in sequences:
                row = []
                for p in range(s.positions):
                    completion = server.client.complete(s.ids[:s.start + p], lora_scale=scale, max_tokens=1,
                                                        logprobs=1, return_tokens=True)
                    row.append(generated_top1(completion))
                out.append(row)
        return out

    return predict


def generated_top1(completion: Completion) -> int:
    """The token a one-token greedy request generated, its top-1, read from the ids the server says it
    generated (return_tokens; research_agent.llm.local, The generated tokens) rather than from the logprob
    list: llama-server b11349 leaves out of that list a token after which the text ends inside a UTF-8
    character, and a top-1 that is the first part of a multi-byte character (a byte-level BPE vocabulary
    splits rarer CJK characters so) would leave it empty. The listed id, when there is one, must be that
    token, and the server must have generated exactly one, as its usage counts too; RuntimeError otherwise,
    and when the server sent no list of the tokens it generated (a server without return_tokens)."""
    generated = completion.generated_ids
    if generated is None:
        raise RuntimeError("the server sent no list of the tokens it generated (return_tokens), so the top-1 "
                           "cannot be read: its logprob list leaves out a token that ends inside a character")
    if len(generated) != 1:
        raise RuntimeError(f"the server generated {len(generated)} tokens for a request of one token")
    count = (completion.raw.get("usage") or {}).get("completion_tokens")
    if isinstance(count, int) and not isinstance(count, bool) and count != 1:
        raise RuntimeError(f"the server generated one token, but its usage counts {count} completion tokens")
    if completion.token_ids and completion.token_ids[0] != generated[0]:
        raise RuntimeError(f"the server lists token {completion.token_ids[0]}, but generated token "
                           f"{generated[0]}")
    return generated[0]

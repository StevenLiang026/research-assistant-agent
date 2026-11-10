"""Convert a PEFT LoRA adapter directory to a GGUF adapter for llama-server, with llama.cpp's converter.

The converter is convert_lora_to_gguf.py from the llama.cpp source tree (b11349 is the version measured in
docs/m0-验证.md). The tree lives outside the repository: --llama-cpp-src or the LLAMA_CPP_SRC environment
variable names it, and the script stops with exit code 2 when neither is set. The converter needs the base
model's config, so --base names the HF model directory. --outtype f16 reproduces the S4 export (34.9 MB,
392 tensors for r=16 on all seven projections). The converter runs with this interpreter, whose environment
has torch and safetensors; gguf-py comes from the source tree itself.

The base (M7). An adapter under a training run's folder, runs/train/<run>/adapter or one of its trainer
checkpoints runs/train/<run>/trainer/checkpoint-<N>, was trained on the model the run's manifest.json
records (config.model, scripts/train.py --model), which is the base it must be converted against: --base
left at its default takes that model when the manifest names one, and the script prints which base it
used and where it came from. An explicit --base is taken as given, even one that names the default's
folder, and so is the default for an adapter that lies under no run folder (the bake-off's synthetic
adapters, docs/m7-计划.md S6).

--dry-run converts nothing. The converter of b11349 has a --dry-run option of its own (it prints what it
would write), which is passed through when the converter's text has it, and its exit code is this
script's. A converter without the option gets a local check instead: the adapter folder must hold an
adapter_config.json of a LORA adapter with target modules and its weights, and every architecture of the
base's config.json must be registered with the converter (the names its sources register with
ModelBase.register, read from the text of convert_hf_to_gguf.py and the conversion package). Exit 1 when the
check finds a problem, which is printed.

Run: .venv/Scripts/python.exe scripts/export_adapter.py runs/train/main/adapter [--outfile x.gguf]
The output file's folder is created first (the converter does not create it; a dry run creates nothing).
Prints the output path and size; the exit code is the converter's.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

SRC_VAR = "LLAMA_CPP_SRC"  # names the llama.cpp source tree, which lives outside the repository
CONVERTER = "convert_lora_to_gguf.py"
MODEL = "models/Qwen3-1.7B"
MANIFEST = "manifest.json"  # a training run's record (scripts/train.py), whose config.model is the base
DRY_RUN = "--dry-run"
ADAPTER_CONFIG, ADAPTER_WEIGHTS = "adapter_config.json", ("adapter_model.safetensors", "adapter_model.bin")
LORA = "LORA"
# the decorators llama.cpp's conversion sources register architectures with: the names are its arguments
_REGISTER = re.compile(r"@\w+\.register\(([^)]*)\)")
_NAME = re.compile(r"\"([A-Za-z0-9_]+)\"|'([A-Za-z0-9_]+)'")


class _Default(str):
    """--base's default: equal to MODEL as a string, yet told apart from an explicit --base that names the
    same folder (argparse hands over a string of its own then), which must win over the run manifest's
    model (module docstring: The base)."""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="PEFT adapter directory -> GGUF adapter (llama.cpp converter).")
    ap.add_argument("adapter", help="PEFT adapter directory (adapter_config.json, adapter_model.safetensors)")
    ap.add_argument("--base", default=_Default(MODEL),
                    help=f"HF directory of the base model (default: the model the adapter's training run "
                         f"records in its manifest, else {MODEL})")
    ap.add_argument("--outfile", default=None, help="default: <adapter>-<outtype>.gguf next to the adapter")
    ap.add_argument("--outtype", default="f16", choices=["f32", "f16", "bf16", "q8_0", "auto"])
    ap.add_argument("--llama-cpp-src", default=None,
                    help=f"llama.cpp source tree (default: ${SRC_VAR}; one of the two is required)")
    ap.add_argument("--dry-run", action="store_true",
                    help="convert nothing: the converter's own dry run when it has one, else a local check "
                         "of the adapter and the base against the converter's registry")
    return ap


def llama_cpp_src(explicit: str | None, env: Mapping[str, str] = os.environ) -> Path:
    """--llama-cpp-src, else $LLAMA_CPP_SRC; LookupError when neither is set (the tree is per machine)."""
    value = explicit or env.get(SRC_VAR)
    if not value:
        raise LookupError(f"no llama.cpp source tree given: pass --llama-cpp-src or set {SRC_VAR}")
    return Path(value)


def default_outfile(adapter: Path, outtype: str) -> Path:
    return adapter.parent / f"{adapter.name}-{outtype}.gguf"


def run_manifest_model(adapter: Path) -> tuple[str, Path] | None:
    """The base model the adapter's training run records and the manifest it is read from (module
    docstring: The base): the manifest.json one or two folders above the adapter whose config names a
    model; None when there is none."""
    for folder in (adapter.parent, adapter.parent.parent):
        manifest = folder / MANIFEST
        if not manifest.is_file():
            continue
        try:
            config = json.loads(manifest.read_text(encoding="utf-8")).get("config") or {}
        except (OSError, ValueError, AttributeError):
            continue
        model = config.get("model") if isinstance(config, dict) else None
        if isinstance(model, str) and model:
            return model, manifest
    return None


def base_model(args: argparse.Namespace) -> tuple[str, str]:
    """The base the adapter is converted against and where it came from: --base when given (whatever it
    names, the default's folder included), else the run manifest's model (named by the manifest's path),
    else the default."""
    if not isinstance(args.base, _Default):
        return args.base, "--base"
    found = run_manifest_model(Path(args.adapter))
    if found is not None:
        return found[0], found[1].as_posix()
    return MODEL, "the default"


def command(args: argparse.Namespace, *, python: str = sys.executable,
            env: Mapping[str, str] = os.environ, dry_run: bool = False) -> list[str]:
    """The converter command line; the output file is its second-to-last argument (third-to-last with the
    converter's own --dry-run, which goes right before the adapter)."""
    converter = llama_cpp_src(args.llama_cpp_src, env) / CONVERTER
    outfile = Path(args.outfile) if args.outfile else default_outfile(Path(args.adapter), args.outtype)
    base, _ = base_model(args)
    cmd = [python, str(converter), "--base", base, "--outtype", args.outtype, "--outfile", str(outfile)]
    return [*cmd, *([DRY_RUN] if dry_run else []), args.adapter]


def converter_supports_dry_run(converter: Path) -> bool:
    """Whether the converter's text names the --dry-run option (b11349's does)."""
    return DRY_RUN in converter.read_text(encoding="utf-8", errors="replace")


def registered_architectures(src: Path) -> set[str]:
    """The architectures the converter's sources register (ModelBase.register's arguments in
    convert_hf_to_gguf.py and the conversion package), read from their text."""
    sources = [src / "convert_hf_to_gguf.py", *sorted((src / "conversion").glob("*.py"))]
    names: set[str] = set()
    for source in sources:
        if not source.is_file():
            continue
        for arguments in _REGISTER.findall(source.read_text(encoding="utf-8", errors="replace")):
            names.update(a or b for a, b in _NAME.findall(arguments))
    return names


def local_check(adapter: Path, base: Path, src: Path) -> tuple[list[str], list[str]]:
    """What a local dry run finds (module docstring): the facts it established and the problems, which
    name the file or the name at fault."""
    facts, problems = [], []
    config_path = adapter / ADAPTER_CONFIG
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        config = None
        problems.append(f"{config_path.as_posix()}: cannot be read as JSON ({type(exc).__name__}: {exc})")
    if isinstance(config, dict):
        peft_type, targets = config.get("peft_type"), config.get("target_modules")
        if peft_type != LORA:
            problems.append(f"{config_path.as_posix()}: peft_type is {peft_type!r}, not {LORA!r}: the "
                            "converter handles LoRA adapters alone")
        if not targets:
            problems.append(f"{config_path.as_posix()}: target_modules names no module")
        else:
            facts.append(f"adapter: {peft_type}, r {config.get('r')}, alpha {config.get('lora_alpha')}, "
                         f"target_modules {targets}")
    elif config is not None:
        problems.append(f"{config_path.as_posix()}: not a JSON object")
    weights = [name for name in ADAPTER_WEIGHTS if (adapter / name).is_file()]
    if weights:
        facts.append(f"weights: {weights[0]} ({(adapter / weights[0]).stat().st_size:,} bytes)")
    else:
        problems.append(f"{adapter.as_posix()}: no {' or '.join(ADAPTER_WEIGHTS)}")
    base_config = base / "config.json"
    try:
        architectures = json.loads(base_config.read_text(encoding="utf-8")).get("architectures") or []
    except (OSError, ValueError, AttributeError) as exc:
        architectures = []
        problems.append(f"{base_config.as_posix()}: cannot be read as JSON ({type(exc).__name__}: {exc})")
    registered = registered_architectures(src)
    for name in architectures:
        if name in registered:
            facts.append(f"architecture {name}: registered with the converter")
        else:
            problems.append(f"architecture {name}: not registered with the converter in {src.as_posix()} "
                            f"({len(registered)} architectures registered)")
    if not architectures and not problems:
        problems.append(f"{base_config.as_posix()}: names no architectures")
    return facts, problems


def main(argv: list[str] | None = None, *, run=subprocess.run) -> int:
    args = build_parser().parse_args(argv)
    try:
        cmd = command(args)
    except LookupError as exc:
        print(exc)
        return 2
    converter, outfile = Path(cmd[1]), Path(cmd[-2])
    if not converter.is_file():
        print(f"converter not found: {converter} (--llama-cpp-src or {SRC_VAR})")
        return 2
    if not Path(args.adapter).is_dir():
        print(f"adapter directory not found: {args.adapter}")
        return 2
    base, origin = base_model(args)
    print(f"base {base} ({origin})")
    if args.dry_run and not converter_supports_dry_run(converter):
        facts, problems = local_check(Path(args.adapter), Path(base), converter.parent)
        for line in (*facts, *(f"PROBLEM: {p}" for p in problems)):
            print(line)
        print(f"dry run: the converter has no {DRY_RUN} option; checked locally, "
              f"{len(problems)} problem(s), nothing written")
        return 1 if problems else 0
    if args.dry_run:
        cmd = command(args, dry_run=True)
    else:  # the converter opens --outfile without creating its folder
        outfile.parent.mkdir(parents=True, exist_ok=True)
    print("running:", subprocess.list2cmdline(cmd), flush=True)
    done = run(cmd)
    if done.returncode != 0:
        print(f"converter exited with {done.returncode}")
        return done.returncode
    if args.dry_run:
        print("dry run: the converter exited with 0, nothing written")
        return 0
    if not outfile.is_file():
        print(f"converter exited with 0 but wrote no file at {outfile}")
        return 3
    size = outfile.stat().st_size
    print(f"wrote {outfile} ({size:,} bytes, {size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

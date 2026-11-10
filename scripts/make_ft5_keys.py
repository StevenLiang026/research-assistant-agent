"""Write FT5's sample keys (docs/m8-计划.md WP3): every sample of special-mix and a quarter of FT3's samples,
drawn per step type with a fixed seed, as scripts/train.py --sample-keys reads them; and print FT5's train.py
command.

    .venv/Scripts/python.exe -B -X utf8 scripts/make_ft5_keys.py [--ft3-manifest runs/train/ft3/manifest.json]
        [--special data/raw/ft5/special-mix.jsonl]
        [--special-reasoning data/raw/ft5/reasoning-special-mix.jsonl]
        [--overlap-report data/synth/benchmark-overlap-m8-ft5.json]
        [--init-adapter runs/train/ft3/trainer/checkpoint-1635] [--share 0.25] [--seed 42]
        [--out data/raw/ft5/ft5-sample-keys.jsonl] [--dry-run] [--force]

Run it from the repository root, as train.py runs: FT3's manifest holds paths relative to it.

The mix. FT5 trains on FT3's data files and sidecars (its manifest's config.args, read by
scripts/check_training_data.py's spec_from_manifest: FT3's model folder, family and length, no glossary) with
special-mix and its sidecar after them, under M8's benchmark-overlap report
(scripts/check_benchmark_overlap.py over every file). The mix is built as train.py builds it
(check_training_data.build_mix: train.py's overlap gate, sidecar index and assembly through the family of
FT3's model folder, torch blocked), and every sample keeps its key and its step type
(research_agent.eval.items.item_type, as check_training_data.describe types the training steps). What stops a
run stops the keys: a data file or sidecar that is not there or is malformed, a sample the family refuses
(exit 2), a file the overlap report does not cover at its current SHA-256 (exit 4). FT3's manifest must
record no sample keys (FT3 trained on every sample it built), and special-mix must not be one of FT3's files
(exit 2). FT3's files, as the FT5 mix builds them, are compared with the per-file accounts FT3's manifest
recorded (check_training_data.ACCOUNT_KEYS); a difference (another report skipping other records, say) is
reported, not failed on.

The keys. Every sample built from special-mix and, of the samples built from FT3's files, per step type the
stratum's quota (quota: its samples x --share, 0.25, rounded half up), drawn without replacement by
random.Random("ft5-keys:<seed>:<type>").sample over the stratum in the mix's order (stratified_share), so a
stratum's draw depends on the seed and its own samples alone. The keys are listed in the mix's order (FT3's
files first), one JSON object {"id", "k"} per line, UTF-8 with LF, written by
check_training_data.write_sample_keys (staged, read back with train.py's own load_sample_keys and
select_samples over every sample built, put in place only when that selects exactly these samples). A key
that names two samples stops it (exit 2). An --out holding other keys is kept (exit 2) unless --force: FT5's
manifest pins the file's SHA-256. One holding these keys is left as it is.

The command (docs/m8-计划.md WP3, preregistered), printed and never run: FT3's recipe, every argument its
manifest records, with special-mix and its sidecar added, the keys file as --sample-keys, --init-adapter
FT3's step 1635, learning rate 3e-5 (the cosine schedule of every run), 1 epoch, a checkpoint every 25 %
(--save-steps 0.25: four), --reasoning-budget 1280, M8's overlap report and --out runs/train/ft5
(FT5_CHANGES). It is parsed back with train.py's own parser, and every argument must be FT3's or one of those
changes (exit 2 otherwise). With the defaults it is:

    .venv/Scripts/python.exe scripts/train.py data/raw/ft3/base-replay-c5.jsonl data/raw/public/deepexi.jsonl
        data/raw/public/format-mix.jsonl data/raw/public/glaive_v2.jsonl data/raw/public/glaive_zh.jsonl
        data/raw/public/hermes.jsonl data/raw/public/toolace.jsonl data/raw/synth/synth-m6.jsonl
        data/raw/synth/synth-m7.jsonl data/raw/ft5/special-mix.jsonl --out runs/train/ft5
        --model models/Nanbeige4.1-3B --reasoning data/raw/synth/reasoning-m7-teacher.jsonl
        data/raw/synth/reasoning-m7-rationalized.jsonl data/raw/synth/reasoning-m7b-teacher.jsonl
        data/raw/synth/reasoning-m7b-rationalized.jsonl data/raw/ft3/reasoning-replay-c5.jsonl
        data/raw/ft5/reasoning-special-mix.jsonl --max-length 5120 --epochs 1 --lr 3e-05 --qlora
        --sample-keys data/raw/ft5/ft5-sample-keys.jsonl --save-steps 0.25
        --overlap-report data/synth/benchmark-overlap-m8-ft5.json
        --init-adapter runs/train/ft3/trainer/checkpoint-1635 --reasoning-budget 1280

Printed: the mix, FT3's check, the counts per step type (FT3's samples, the ones listed, special-mix's, all
listed), the keys file and the command, then "summary: " with the numbers as JSON. Exit 0 when the keys are
written (or already there, or, with --dry-run, which writes nothing, counted); 2 on a usage or input error; 4
at the overlap gate.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

from research_agent.eval.items import TYPES

SCRIPTS = Path(__file__).resolve().parent


def _load_check():
    """scripts/check_training_data.py as a module of its own name (scripts/ is no package): its mix of a
    manifest, its build of train.py's samples with their step types and its keys writer; and through it
    train.py's parser, overlap gate and sample-keys reader."""
    spec = importlib.util.spec_from_file_location("_check_training_data_shared",
                                                  SCRIPTS / "check_training_data.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = _load_check()
train = check.train

FT3_MANIFEST = Path("runs/train/ft3/manifest.json")
SPECIAL = "data/raw/ft5/special-mix.jsonl"
SPECIAL_REASONING = "data/raw/ft5/reasoning-special-mix.jsonl"
OVERLAP_REPORT = "data/synth/benchmark-overlap-m8-ft5.json"
INIT_ADAPTER = "runs/train/ft3/trainer/checkpoint-1635"
OUT = Path("data/raw/ft5/ft5-sample-keys.jsonl")
SHARE, SEED = 0.25, 42
SALT = "ft5-keys:"  # a stratum's draw: random.Random(f"{SALT}{seed}:{type}")
# FT5's preregistered changes to FT3's recipe (docs/m8-计划.md WP3), besides its data, sidecars, overlap
# report, keys and init adapter
FT5_CHANGES = {"out": "runs/train/ft5", "epochs": 1, "lr": 3e-5, "save_steps": 0.25, "reasoning_budget": 1280}
TRAIN = ".venv/Scripts/python.exe scripts/train.py"
USAGE, OVERLAP_REFUSED = check.USAGE, check.OVERLAP_REFUSED


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="FT5's sample keys: every special-mix sample and a share of "
                                             "FT3's per step type; and FT5's train.py command.")
    ap.add_argument("--ft3-manifest", type=Path, default=FT3_MANIFEST,
                    help=f"FT3's run manifest (default {FT3_MANIFEST.as_posix()})")
    ap.add_argument("--special", default=SPECIAL, help=f"special-mix (default {SPECIAL})")
    ap.add_argument("--special-reasoning", default=SPECIAL_REASONING,
                    help=f"special-mix's reasoning sidecar (default {SPECIAL_REASONING})")
    ap.add_argument("--overlap-report", default=OVERLAP_REPORT,
                    help=f"M8's benchmark-overlap report over every file (default {OVERLAP_REPORT})")
    ap.add_argument("--init-adapter", default=INIT_ADAPTER,
                    help=f"FT3's adapter FT5 starts from, for the command (default {INIT_ADAPTER})")
    ap.add_argument("--share", type=float, default=SHARE,
                    help=f"the share of FT3's samples of each step type listed (default {SHARE})")
    ap.add_argument("--seed", type=int, default=SEED, help=f"the draw's seed (default {SEED})")
    ap.add_argument("--out", type=Path, default=OUT, help=f"the keys file (default {OUT.as_posix()})")
    ap.add_argument("--dry-run", action="store_true", help="count and print, write nothing")
    ap.add_argument("--force", action="store_true", help="replace an --out that holds other keys")
    return ap


def posix(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def quota(n: int, share: float) -> int:
    """A stratum's quota: n x share rounded half up, the share read as the decimal it prints as."""
    return math.floor(n * Fraction(str(share)) + Fraction(1, 2))


def stratified_share(samples: list[dict], share: float, seed: int) -> list[dict]:
    """The samples drawn per step type (their "type"): each stratum's quota, drawn without replacement by
    random.Random(f"{SALT}{seed}:{type}").sample over the stratum in the order given; in that order."""
    strata: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        strata[sample["type"]].append(index)
    chosen: set[int] = set()
    for stratum, indices in strata.items():
        chosen.update(random.Random(f"{SALT}{seed}:{stratum}").sample(indices, quota(len(indices), share)))
    return [sample for index, sample in enumerate(samples) if index in chosen]


def ft5_spec(ft3: check.MixSpec, *, special: str, special_reasoning: str,
             overlap_report: str) -> check.MixSpec:
    """The FT5 mix: FT3's with special-mix and its sidecar after its own, under M8's overlap report."""
    return dataclasses.replace(ft3, data=[*ft3.data, special], reasoning=[*ft3.reasoning, special_reasoning],
                               overlap_report=overlap_report, no_overlap_check=False, sample_keys=None,
                               source="FT3's manifest with special-mix", manifest=None)


def ft5_values(ft3_args: dict, *, special: str, special_reasoning: str, overlap_report: str, keys: str,
               init_adapter: str) -> dict:
    """FT5's train.py arguments by name: FT3's recorded ones with the preregistered changes (module docstring:
    The command)."""
    return {**ft3_args, **FT5_CHANGES, "data": [*ft3_args["data"], special],
            "reasoning": [*(ft3_args.get("reasoning") or []), special_reasoning],
            "overlap_report": overlap_report, "sample_keys": keys, "init_adapter": init_adapter}


def ft5_command(ft3_args: dict, **inputs: str) -> list[str]:
    """FT5's train.py arguments as a command line, in train.py's parser order: the data files, then every flag
    whose value is not train.py's default (an argument absent unless given is always written)."""
    values = ft5_values(ft3_args, **inputs)
    argv: list[str] = []
    for action in train.build_parser()._actions:
        if action.dest not in values or isinstance(action, argparse._HelpAction):
            continue
        value, flag = values[action.dest], (action.option_strings or [None])[0]
        if flag is None:
            argv += [str(v) for v in value]
        elif action.default is not argparse.SUPPRESS and value == action.default:
            continue
        elif isinstance(action, argparse._StoreTrueAction):
            argv.append(flag)
        elif action.nargs in ("+", "*"):
            argv += [flag, *(str(v) for v in value)]
        else:
            argv += [flag, str(value)]
    return argv


def check_command(argv: list[str], ft3_args: dict, **inputs: str) -> None:
    """CheckError unless train.py's parser reads argv back as FT3's recorded arguments with the changes."""
    expected = ft5_values(ft3_args, **inputs)
    try:
        parsed = vars(train.build_parser().parse_args(argv))
    except SystemExit as exc:
        raise check.CheckError(f"train.py's parser refuses the FT5 command (exit {exc.code})") from exc
    wrong = [f"{key}: {parsed.get(key, 'absent')!r} where {value!r}" for key, value in expected.items()
             if parsed.get(key, object()) != value]
    if wrong:
        raise check.CheckError("the FT5 command does not read back as FT3's recipe with its changes: "
                               + "; ".join(wrong))


def ft3_differences(manifest: dict, per_file: list[dict], ft3_files: set[str]) -> list[str]:
    """Where FT3's files, as the FT5 mix builds them, differ from the per-file accounts of FT3's manifest, one
    line each (check_training_data.ACCOUNT_KEYS); [] when they do not."""
    then = {posix(f["path"]): f for f in ((manifest.get("config") or {}).get("data") or {}).get("files", [])}
    now = {posix(f["path"]): f for f in per_file if posix(f["path"]) in ft3_files}
    lines = [f"{path}: {'recorded, not rebuilt' if path in then else 'rebuilt, not recorded'}"
             for path in [*then, *now] if (path in then) != (path in now)]
    for path in [path for path in then if path in now]:
        lines += [f"{path}: {key} {then[path][key]!r} -> {now[path].get(key)!r}" for key in check.ACCOUNT_KEYS
                  if key in then[path] and then[path][key] != now[path].get(key)]
    return lines


def stratum_counts(ft3_samples: list[dict], drawn: list[dict], special: list[dict]) -> dict:
    """Per step type (TYPES' order, then any other): FT3's samples, the ones drawn, special-mix's and all
    listed."""
    built, listed, extra = (Counter(s["type"] for s in group) for group in (ft3_samples, drawn, special))
    order = [*TYPES, *sorted((set(built) | set(extra)) - set(TYPES))]
    return {t: {"ft3_built": built[t], "ft3_listed": listed[t], "special": extra[t],
                "listed": listed[t] + extra[t]} for t in order if built[t] or extra[t]}


def keys_bytes(kept: list[dict]) -> bytes:
    """The keys file's bytes, as check_training_data.write_sample_keys writes them."""
    return "".join(json.dumps({"id": s["key"][0], "k": s["key"][1]}, ensure_ascii=False) + "\n"
                   for s in kept).encode("utf-8")


def main(argv: list[str] | None = None, *, load_family: check.LoadFamily | None = None) -> int:
    """load_family stands in for research_agent.family.core.load_family in the tests (train.py's hook)."""
    started = time.perf_counter()
    args = build_parser().parse_args(argv)
    load = load_family or train.default_load_family
    inputs = {"special": args.special, "special_reasoning": args.special_reasoning,
              "overlap_report": args.overlap_report}
    try:
        if not 0 < args.share <= 1:
            raise check.CheckError(f"--share is a share above 0 and at most 1, not {args.share}")
        ft3 = check.spec_from_manifest(args.ft3_manifest)
        if ft3.sample_keys is not None:
            raise check.CheckError(f"{posix(args.ft3_manifest)} records sample keys ({ft3.sample_keys}): FT3 "
                                   "trained on every sample it built, so this is another run")
        try:
            ft3_files = {posix(f) for f in train.expand_data_files(ft3.data)}
        except FileNotFoundError as exc:
            raise check.CheckError(str(exc)) from exc
        if posix(args.special) in ft3_files:
            raise check.CheckError(f"{args.special} is one of FT3's data files: special-mix is new data")
        mix = check.build_mix(ft5_spec(ft3, **inputs), load)
        ft3_args = ft3.manifest["config"]["args"]
        command_inputs = {**inputs, "keys": posix(args.out), "init_adapter": args.init_adapter}
        command = ft5_command(ft3_args, **command_inputs)
        check_command(command, ft3_args, **command_inputs)
        shared = check.shared_keys(mix.built)
        if shared:
            raise check.CheckError(f"{len(shared)} key(s) name more than one sample, so a keys file cannot "
                                   "list one without the other: "
                                   + ", ".join(repr(k) for k in shared[:check.SHOWN]))
    except check.CheckError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return exc.code
    special = [s for s in mix.built if posix(s["file"]) == posix(args.special)]
    ft3_samples = [s for s in mix.built if posix(s["file"]) in ft3_files]
    drawn = stratified_share(ft3_samples, args.share, args.seed)
    picked = {id(s) for s in (*drawn, *special)}
    kept = [s for s in mix.built if id(s) in picked]
    strata = stratum_counts(ft3_samples, drawn, special)
    differences = ft3_differences(ft3.manifest, mix.per_file, ft3_files)
    content = keys_bytes(kept)
    keys = {"path": posix(args.out), "sha256": hashlib.sha256(content).hexdigest(), "listed": len(kept),
            "written": False}

    lines = check.mix_lines("FT5 mix", mix, None)
    account = next((f for f in mix.per_file if posix(f["path"]) == posix(args.special)), None)
    if account is not None:
        lines.append(f"  special-mix: records {account['records']}, samples {account['samples']} "
                     f"({train.modes_text(account['modes'])}), dropped {account['dropped']}")
    lines.append("FT3's files: as FT3's manifest recorded them (records, samples, drops, overlap skips, "
                 "tokens, modes, refusals)" if not differences else
                 f"FT3's files: {len(differences)} difference(s) from FT3's manifest (reported, not failed "
                 "on)")
    lines += [f"  {line}" for line in differences]
    rows = [[t, c["ft3_built"], c["ft3_listed"], c["special"], c["listed"]] for t, c in strata.items()]
    rows.append(["total", len(ft3_samples), len(drawn), len(special), len(kept)])
    lines += ["", (f"== FT5 sample keys: every special-mix sample and {args.share:g} of FT3's per step type "
                   f"(seed {args.seed}, rounded half up)")]
    lines += check.table(["type", "FT3 built", "FT3 listed", "special-mix", "listed"], rows)
    print("\n".join([*lines, ""]))
    if args.dry_run:
        print(f"dry run: {len(kept)} keys counted, nothing written ({posix(args.out)})")
    elif args.out.is_file() and args.out.read_bytes() == content:
        print(f"sample keys: {posix(args.out)} holds these {len(kept)} keys already: unchanged (sha256 "
              f"{keys['sha256']})")
    elif args.out.exists() and not args.force:
        print(f"FAIL: {posix(args.out)} holds other keys, which FT5's manifest may pin; --force replaces it",
              file=sys.stderr)
        return USAGE
    else:
        try:
            written = check.write_sample_keys(kept, mix.built, args.out)
        except check.CheckError as exc:
            print(f"FAIL: {exc}; no keys file written", file=sys.stderr)
            return exc.code
        keys |= {"sha256": written["sha256"], "written": True}
        print(f"sample keys: {len(kept)} written to {posix(args.out)} (sha256 {written['sha256']}); "
              "train.py's load_sample_keys and select_samples read back exactly these samples, in this order")
    print(f"FT5 train.py command (written here, not run): {TRAIN} {' '.join(command)}")
    summary = {"ft3_manifest": posix(args.ft3_manifest), **{k: posix(v) for k, v in inputs.items()},
               "share": args.share, "seed": args.seed, "built": len(mix.built),
               "special_account": account, "strata": strata, "listed": len(kept), "keys": keys,
               "ft3_check": differences, "command": command,
               "seconds": round(time.perf_counter() - started, 1)}
    print("summary: " + json.dumps(summary, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Compare a development run on the hf backend with the same system's run on llama.cpp (docs: the M3 plan,
phase 3 and default 7): how far the full-precision model and the quantized one agree, item by item, and what
the difference does to the success rate.

    .venv/Scripts/python.exe scripts/compare_backends.py --hf runs/eval/<hf run>.jsonl \\
        --llama-cpp runs/eval/<llama.cpp run>.jsonl [--items data/eval/dev.jsonl] \\
        [--out results/backend-comparison-<system>-scale<s>.json]

The runs. Two record files of scripts/run_eval.py, each with its manifest beside it (<stem>.manifest.json):
--hf a run with backend "hf", --llama-cpp one on llama-server (backend "llama.cpp"; a manifest from before
the backends, which has no such key, counts when its provider is llama-server), of the same local system at
the same lora scale, with the same prompt and schema hashes and b1's demonstrations, each over every item of
--items (the SHA-256 both manifests recorded, no --limit). The pair at scale 1 (ft or b4) measures the base
model's quantization and the adapter's conversion together, the pair at scale 0 (b0 or b1) the base model's
quantization alone, so the two reports separate them. That the llama.cpp run's GGUF was converted from the hf
run's adapter folder is not checked: the report keeps the SHA-256 of both.

Per item: agreement, the exact equality of the parsed turn (turn_of: an ok record's output, the API form of
the turn LocalModel read from the completion; a record without a turn agrees only with one of the same status
and, for an error, the same error type and message), and each run's success, scored here against the items
with research_agent.eval.metrics.score_items, as scripts/score.py scores them. Then each run's success rate,
their paired difference (hf minus llama.cpp) with the paired bootstrap's 95% interval (metrics.bootstrap_ci:
10,000 resamples of the items, seed 42, the M3 plan's default 2) and the exact McNemar test on the items only
one of the runs gets right (metrics.mcnemar).

Which items: as in scripts/run_eval.py, test material is refused before anything is read (exit 4), or, for
items that hold an item of split "test" or with an id of the test split (eval.lock.refuse_test_material),
once read and before the runs are read; the development set, or any items file with a lock beside it, is read
through DevSet: its lock must match (exit 5). A run may hold error records its model caused (HarnessError: a
completion that could not be read) and context_overflow records, which both backends give alike; any other
error (a crash, a lost GPU) or status means the run did not measure its backend (exit 3).

Writes --out (default results/backend-comparison-<system>-scale<s>.json, <s> being 1 or 0): JSON, two-space
indent, LF, repository-relative POSIX paths and no time, so the same runs give the same bytes: the system, the
scale, the items with their SHA-256; per backend the records with their SHA-256, what the manifest says of the
model (provider, model name, dtype, device, the adapter folder with its safetensors SHA-256 or the server's
adapters with their GGUF SHA-256, the model file and the llama.cpp build or the package versions, the run's
git commit), the status counts, the successes and the success rate; the agreement (count, rate, the ids that
differ); the difference with its interval; McNemar's counts and p; the git commit and whether the tree was
dirty. Nothing is registered in results/numbers.json. Run it from the repository root: a path outside the
current folder is refused, since the report holds repository-relative paths only. That includes the paths the
manifests recorded as they were given (the adapter folder of --adapter, the server's adapters, the model the
client or llama-server's /props reports): an absolute one inside the repository is written relative, in POSIX
form, and a run that recorded one outside it is refused.

Exit 0 when the report is written; 1 on malformed input (records, items or a manifest that cannot be read);
2 when the runs cannot be compared (not an hf run and a llama.cpp run, another system, scale, prompt, schema,
demonstrations or items, a partial run, records whose ids differ from the items', a path outside the current
folder, the manifests' recorded paths included) or on a usage error; 3 when a run holds records its model did
not cause; 4 when the items are test material; 5 when they do not match their lock.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter
from pathlib import Path

from research_agent.agent.types import CONTEXT_OVERFLOW
from research_agent.canonical import canonical_json
from research_agent.eval.lock import (
    DEV_SET,
    DevSet,
    JsonlError,
    LockError,
    RefusedTestSet,
    lock_required,
    read_jsonl,
    refuse_test_material,
    refuse_test_sets,
    sha256_of,
)
from research_agent.eval.metrics import IdMismatch, bootstrap_ci, mcnemar, score_items
from research_agent.eval.runner import ERROR, OK
from research_agent.eval.systems import HF, LLAMA_CPP, LOCAL_SCALES
from research_agent.train.manifest import REPO, git_commit, git_dirty

SEED, RESAMPLES = 42, 10_000  # the paired bootstrap of the M3 plan, default 2
MODEL_ERRORS = ("HarnessError",)  # the error records a local model's own output causes (agent.models)
RECORD_STATUSES = (OK, CONTEXT_OVERFLOW, ERROR)  # a local run has no budget stop
# manifest keys the two runs must share (the items' SHA-256 is also checked against the items given)
SAME = ("system", "lora_scale", "prompt_hash", "schema_hash", "demonstrations", "items_sha256")
CARRIED = ("provider", "model", "dtype", "device", "adapter", "adapters", "git_commit", "git_dirty")
RUNTIME = ("model_path", "build_info", "gpu", "attn_implementation", "versions")  # of the server_props
AGREEMENT = ("the parsed turns are equal: the outputs of two ok records (the API form of the turn LocalModel "
             "read from the completion), or two records without a turn of the same status and, for an error, "
             "the same error type and message")
DIFFERENCE = ("diff: the success rate of hf minus that of llama.cpp; low and high: the paired bootstrap's "
              "95% interval, mean: the mean of its n resamples of the items, drawn with seed")


class Incomparable(ValueError):
    """Two runs that cannot be compared (module docstring)."""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Compare a development run on the hf backend with the same "
                                             "system's run on llama.cpp.")
    ap.add_argument("--hf", type=Path, required=True, help="the records of the run_eval --backend hf run")
    ap.add_argument("--llama-cpp", type=Path, required=True,
                    help="the records of the llama-server run of the same system at the same lora scale")
    ap.add_argument("--items", type=Path, default=DEV_SET,
                    help=f"the items both runs ran (default {DEV_SET.as_posix()})")
    ap.add_argument("--out", type=Path, default=None,
                    help="the report (default results/backend-comparison-<system>-scale<s>.json)")
    return ap


def rel(path: str | Path) -> str:
    """path relative to the current folder in POSIX form when it lies inside it, else absolute (POSIX)."""
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return absolute.as_posix()


def inside(path: str | Path) -> bool:
    return not Path(rel(path)).is_absolute()


def percent(rate: float) -> str:
    return f"{100 * rate:.1f}%"


def manifest_of(records: Path) -> dict:
    """The manifest run_eval wrote beside records; OSError or ValueError when it is missing or no object."""
    path = records.with_name(records.stem + ".manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"{rel(path)} is not a JSON object")  # noqa: TRY004  malformed input
    return manifest


def backend_of(manifest: dict) -> str | None:
    """The run's backend; a llama-server run from before the backends (no "backend" key) is llama.cpp."""
    if "backend" in manifest:
        return manifest["backend"]
    return LLAMA_CPP if manifest.get("provider") == "llama-server" else None


def check_comparable(hf: dict, cpp: dict, items_sha256: str, n: int) -> None:
    """Incomparable unless the manifests describe an hf run and a llama.cpp run of one local system at one
    lora scale, each over every one of the n items whose SHA-256 is items_sha256 (module docstring)."""
    for option, manifest, wanted in (("--hf", hf, HF), ("--llama-cpp", cpp, LLAMA_CPP)):
        if backend_of(manifest) != wanted:
            raise Incomparable(f"{option} is a run on {backend_of(manifest) or 'no local backend'}, "
                               f"not on {wanted}")
    if hf.get("system") not in LOCAL_SCALES:
        raise Incomparable(f"{hf.get('system')!r} is not a local system ({', '.join(LOCAL_SCALES)})")
    for key in SAME:
        if hf.get(key) != cpp.get(key):
            raise Incomparable(f"the runs differ in {key}: hf {hf.get(key)!r}, llama.cpp {cpp.get(key)!r}")
    if hf.get("items_sha256") != items_sha256:
        raise Incomparable(f"the runs were made on other items (SHA-256 {hf.get('items_sha256')}) than these "
                           f"({items_sha256})")
    for name, manifest in ((HF, hf), (LLAMA_CPP, cpp)):
        if manifest.get("items_run") != n:
            raise Incomparable(f"the {name} run covers {manifest.get('items_run')} of the {n} items "
                               "(--limit): compare whole runs")


def foreign_records(records: list[dict]) -> Counter:
    """The records the model did not cause, by error type or status (module docstring)."""
    return Counter(r.get("error_type") if r.get("status") == ERROR else r.get("status") for r in records
                   if r.get("status") not in RECORD_STATUSES
                   or (r.get("status") == ERROR and r.get("error_type") not in MODEL_ERRORS))


def turn_of(record: dict) -> str:
    """The parsed turn of a record in canonical JSON, what the two runs must agree on (module docstring): an
    ok record's output; the status, error type and error message of any other record."""
    if record.get("status") == OK:
        return canonical_json({"status": OK, "output": record.get("output")})
    return canonical_json({key: record.get(key) for key in ("status", "error_type", "error")})


def repository_path(recorded: str, run: str, what: str) -> str:
    """A path a run manifest recorded (a relative one from the folder the run ran in, the repository root) as
    the report holds it: relative to the current folder, in POSIX form; Incomparable when it lies outside."""
    if not inside(recorded):
        raise Incomparable(f"the {run} run records {what} {recorded}, which lies outside the repository: the "
                           "report holds repository-relative paths only")
    return rel(recorded)


def carried(run: str, manifest: dict) -> dict:
    """What the report carries of a run's manifest: CARRIED, and RUNTIME of its server_props under "runtime",
    with the adapter folder, the server's adapters and the model as repository-relative POSIX paths
    (repository_path; Incomparable when one lies outside the repository)."""
    out = {key: copy.deepcopy(manifest[key]) for key in CARRIED if key in manifest}
    props = manifest.get("server_props") or {}
    out["runtime"] = {key: props[key] for key in RUNTIME if key in props}
    adapter = out.get("adapter")
    if isinstance(adapter, dict) and isinstance(adapter.get("path"), str):
        adapter["path"] = repository_path(adapter["path"], run, "the adapter folder")
    for entry in out.get("adapters") or []:
        if isinstance(entry, dict) and isinstance(entry.get("path"), str):
            entry["path"] = repository_path(entry["path"], run, "the server's adapter")
    if isinstance(out["runtime"].get("model_path"), str):
        out["runtime"]["model_path"] = repository_path(out["runtime"]["model_path"], run, "the model")
    return out


def run_entry(backend: str, path: Path, kept: dict, records: list[dict], successes: list[bool]) -> dict:
    """One run as the report holds it (module docstring); kept is what carried() keeps of its manifest."""
    return {"backend": backend, "records": rel(path), "records_sha256": sha256_of(path), **kept,
            "statuses": dict(sorted(Counter(r.get("status") for r in records).items())),
            "successes": sum(successes), "success_rate": sum(successes) / len(successes)}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    roots = (Path.cwd(), REPO)
    try:
        refuse_test_sets(args.items, roots)
        locked = lock_required(args.items, roots)
        if locked:
            dev = DevSet(args.items)
            items = dev.open()
            items_sha256 = dev.last_open["sha256"]
        else:
            items, items_sha256 = read_jsonl(args.items), sha256_of(args.items)
        refuse_test_material(items, args.items)
    except RefusedTestSet as exc:
        print(f"FAIL: {exc}")
        return 4
    except LockError as exc:
        print(f"FAIL: {exc}")
        return 5
    except (JsonlError, OSError) as exc:
        print(f"FAIL: malformed items: {exc}")
        return 1
    if not items:
        print(f"FAIL: {rel(args.items)} holds no items")
        return 1
    paths = {HF: args.hf, LLAMA_CPP: args.llama_cpp}
    try:
        manifests = {name: manifest_of(path) for name, path in paths.items()}
    except (OSError, ValueError) as exc:
        print(f"FAIL: a run manifest cannot be read: {exc}")
        return 1
    try:
        check_comparable(manifests[HF], manifests[LLAMA_CPP], items_sha256, len(items))
    except Incomparable as exc:
        print(f"FAIL: {exc}")
        return 2
    system, scale = manifests[HF]["system"], manifests[HF]["lora_scale"]
    out = args.out or Path("results") / f"backend-comparison-{system}-scale{scale:g}.json"
    outside = [rel(path) for path in (args.items, args.hf, args.llama_cpp, out) if not inside(path)]
    if outside:
        print(f"FAIL: run from the repository root: {outside} lie outside the current folder, and the report "
              "holds repository-relative paths only")
        return 2
    try:
        kept = {name: carried(name, manifest) for name, manifest in manifests.items()}
    except Incomparable as exc:
        print(f"FAIL: {exc}")
        return 2
    try:
        records = {name: read_jsonl(path) for name, path in paths.items()}
    except (JsonlError, OSError) as exc:
        print(f"FAIL: malformed records: {exc}")
        return 1
    for name, path in paths.items():
        foreign = foreign_records(records[name])
        if foreign:
            print(f"FAIL: the {name} run holds records its model did not cause {dict(foreign)}: it does not "
                  f"measure its backend (records {rel(path)})")
            return 3
    success: dict[str, dict[str, bool]] = {}
    for name in paths:
        try:
            success[name] = {s["id"]: bool(s["success"]) for s in score_items(records[name], items)}
        except IdMismatch as exc:
            print(f"FAIL: the {name} run: {exc}")
            return 2
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            print(f"FAIL: malformed records in the {name} run: {type(exc).__name__}: {exc}")
            return 1
    ids = [item["id"] for item in items]
    by_id = {name: {r["id"]: r for r in records[name]} for name in paths}
    hf_ok, cpp_ok = ([success[name][i] for i in ids] for name in (HF, LLAMA_CPP))
    agree = [turn_of(by_id[HF][i]) == turn_of(by_id[LLAMA_CPP][i]) for i in ids]
    interval = bootstrap_ci(hf_ok, cpp_ok, seed=SEED, n=RESAMPLES)
    test = mcnemar(hf_ok, cpp_ok)
    n = len(ids)
    report = {
        "system": system, "lora_scale": scale, "items": rel(args.items), "items_sha256": items_sha256,
        "items_locked": locked, "n": n,
        "runs": {HF: run_entry(HF, args.hf, kept[HF], records[HF], hf_ok),
                 LLAMA_CPP: run_entry(LLAMA_CPP, args.llama_cpp, kept[LLAMA_CPP], records[LLAMA_CPP],
                                      cpp_ok)},
        "agreement": {"rule": AGREEMENT, "agree": sum(agree), "n": n, "rate": sum(agree) / n,
                      "differ": [i for i, same in zip(ids, agree, strict=True) if not same]},
        "success_difference": {"of": DIFFERENCE, **interval},
        "mcnemar": {"hf_only": test["a_only"], "llama_cpp_only": test["b_only"],
                    "discordant": test["discordant"], "p": test["p"]},
        "git_commit": git_commit(REPO), "git_dirty": git_dirty(REPO),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{system} at lora scale {scale:g}: agreement {sum(agree)}/{n} ({percent(sum(agree) / n)}); "
          f"success hf {percent(sum(hf_ok) / n)}, llama.cpp {percent(sum(cpp_ok) / n)}; difference "
          f"{100 * interval['diff']:+.1f} points, 95% interval [{100 * interval['low']:+.1f}, "
          f"{100 * interval['high']:+.1f}]; McNemar {test['a_only']} vs {test['b_only']}, "
          f"p = {test['p']:.3g}; report {rel(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

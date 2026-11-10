"""Decide, once and mechanically, which FT5 checkpoint goes to M8's one ACEBench run, if any
(docs/m8-计划.md WP4, its rules written before FT5 has a score).

    .venv/Scripts/python.exe scripts/decide_m8.py --selection results/checkpoint-selection-ft5.json \\
        [--baseline results/checkpoint-selection-ft3.json] [--baseline-step 1635] --rank-only
    .venv/Scripts/python.exe scripts/decide_m8.py --selection results/checkpoint-selection-ft5.json \\
        [--baseline results/checkpoint-selection-ft3.json] [--baseline-step 1635] \\
        --special b0=<summary> --special ft3=<summary> [--special <step>=<summary> ...] \\
        [--calibration results/special-dev-calibration-m8.json]

The candidates. Every checkpoint of --selection, scripts/select_checkpoint.py's file of FT5's selection runs:
each row's step, GGUF and SHA-256, and its runs s0 and s1 (seeds 0 and 1) with the records of every set. The
selection's item sets must be the development sets taken together, data/eval/dev.jsonl and
data/eval/v2/dev-v2.jsonl (SETS, in that order), at the SHA-256 the files have now. The baseline is FT3's
step 1635 (--baseline-step) of --baseline, FT3's selection file, over the same sets at the same SHA-256.
Every run's manifest (beside its records, <stem>.manifest.json, as scripts/run_eval.py writes it) must name
the checkpoint's GGUF alone by its SHA-256, the set at its current SHA-256 with every item run, and what FT3's
run of the same seed and set was measured with: the generation (mode, decoding, reasoning budget, base seed
and the rest), the family (its name and template), the agent prompt (prompt_hash, prompt_version), the tool
schemas (schema_hash), the lora scale and the base model the server loaded (server_props.model_path), so
that every sum is of the same metric as FT3's (同口径). The sets are opened as the runners open them
(eval.lock: the lock checked, test material refused), and no item id is in both.

The two-seed sum. Every record with the runtime rules applied offline (agent.rules.ruled_record, as
scripts/vote_offline.py --rules applies them; a record that is not ok stays as it is), scored by the
normalized scorer (eval.metrics); a checkpoint's sum is its successes over both sets and both seeds, the sum
without the rules recorded beside it. The ranking: the higher sum first, a tie to the later step.

The procedure. Dev set B is run for the top of the ranking alone (docs/m8-计划.md, 时间与花费: 1-2 GPU
hours). --rank-only prints the ranking and FT3's MG baseline; it reads neither dev set B's summaries
(--special is refused with it) nor the calibration file, and writes nothing (a decision already written does
not stop it). Then dev set B is run for the first checkpoint of the ranking (scripts/run_special_dev.py, --out
runs/m8/special-dev/b-ft5-<step>.jsonl) and the decision is asked for with B0's and FT3's summaries (made
for the calibration) and those of the candidates run so far, any of them. Walking the ranking, the first
checkpoint passing PG1 to PG4 is chosen; a checkpoint the walk reaches without a summary, before any passed,
stops it: "NEED dev set B for checkpoint <step>" (exit 3), nothing written; run it and ask again. A summary
of a checkpoint ranked after the chosen one is checked, and its gates recorded, but it does not decide.

The gates, on dev set B (DEV_SET_B: data/eval/special-dev-b.jsonl, the set kept for the selection; dev set A
is the data's), one --special LABEL=PATH per run, b0 and ft3 always: the same numbers of special and valid
items in every summary, and
  PG1 the lenient special rate >= B0's + 0.05; when K1 failed (the calibration file, below), the strict
      special rate >= 0.75 instead;
  PG2 the strict special rate >= 0.70;
  PG3 the false refusals on the valid items <= FT3's + 2;
  PG4 the exact-call rate on the valid items >= FT3's - 0.05, and the zh alterations <= FT3's.
"special" is the three gated refusal kinds pooled (priority is reported, never gated). Rates are compared
exactly, as fractions of the counts.

Where the summaries come from. Each PATH is a summary scripts/run_special_dev.py wrote (<stem>.summary.json:
the records' file name under "records", the set and its SHA-256, the research_agent.eval.special_dev
summarize object under "summary"), and the run manifest beside its records (<records stem>.manifest.json)
must show that the run is the one its label needs:
- b0 applied no adapter: lora scale 0 (B0_SCALE), or no adapter loaded;
- ft3 applied FT3's step-1635 GGUF alone (its SHA-256 in --baseline) at lora scale 1 (FT_SCALE);
- a candidate applied its checkpoint's GGUF alone (its SHA-256 in --selection) at lora scale 1;
- the set, in the summary and in the manifest, is dev set B (its path) at the SHA-256 of its lock, and the
  file still matches the lock (eval.lock.check_lock); every item was run (items_run = items_total) and
  answered (statuses: every record ok, none an error or a context overflow), and the summary counts no item
  the scorer failed on (scoring_errors);
- the server had one slot (server_props.total_slots, as the runner requires);
- the generation is think mode, greedy decoding (family.core.GREEDY) and the reasoning budget R = 1,280
  (docs/m8-计划.md, 方法约束), and every summary's run has the same generation;
- every run was made on B0's base model, llama.cpp build and family (DEV_B_SAME: server_props.model_path,
  server_props.build_info, family.name and family.template_sha256, each recorded).
The counts. Dev set B is opened as the runner opens it (scripts/run_special_dev.py load_items: test material
and the held-out sets refused, the lock checked, every item checked), and every run's records are scored
again against it by the scorer as it is now (research_agent.eval.special_dev score_records and summarize, as
scripts/calibrate_special_dev.py scores them): the gates read those counts. Every record must be ok and every
item scored without an error, and the summary must have counted what the records count now: a summary that
another version of the scorer wrote is refused (write it again from its records), so that every number the
gates compare is of one scorer.
The calibration file (--calibration, CALIBRATION by default: scripts/calibrate_special_dev.py's) must be of
these runs: its set (items.sha256) at dev set B's lock, its runs (runs.b0.sha256, runs.ft.sha256) the
records of --special b0 and --special ft3 by their SHA-256, and its summaries (summaries.b0, summaries.ft)
their counts scored again. It tells the PG1 fallback by "pg1_fallback" (true or false) or by K1's verdict
("K1": {"holds": true or false}; "passed" is read too): the fallback is on when K1 does not hold; given both,
they must agree, given neither the run stops, and the verdict must be K1 of the counts scored again (B0's
lenient special rate at least 5 points above FT3's, K1_POINTS).

The decision. The checkpoint the walk chooses must then pass MG: its sum at least FT3's sum (step 1635, the
same metric) less 10. GO with the chosen checkpoint (whose GGUF must still have its SHA-256) when MG passes;
NO-GO when it fails or no checkpoint passes the gates: no ACEBench run, the owner is told. MG is checked on
the chosen checkpoint alone: one ranked after it has no higher sum, so it could not pass MG either.

Writes OUT, results/decision-m8.json in this repository (JSON, two-space indent, LF): the inputs (paths and
SHA-256), the item sets, dev set B and where every dev set B number comes from (summary, records and manifest
by SHA-256, adapters, lora scale, statuses, the counts scored again and the scoring errors; the generation,
base model, build and family the runs share; the scorer's file by its SHA-256), the calibration's runs and
fallback, B0's and FT3's numbers, every candidate's sums, dev set B numbers and gate values (none for a
checkpoint not run), the ranking, the walk, the chosen checkpoint, MG, the decision and its reason. It is
never overwritten, and its presence refuses a decision before anything is read. Prints the ranking with every
gate value, FT3's sum, MG and the decision.

Exit 0 when decided and written (GO or NO-GO), or the ranking printed (--rank-only); 1 on a malformed
selection file, run, summary or calibration file, malformed items of dev set B, or dev set B records that are
not one of every item; 2 on a usage error, an existing decision, a missing file and every disagreement (a set,
a run manifest, a summary that is not of the run its label needs or that counted otherwise than its records
scored again, records not all ok or that the scorer fails on, runs not of one base model, build and family, a
calibration file of other runs or counts or that cannot tell the fallback); 3 when the walk reaches a
checkpoint without dev set B's summary before any passed; 4 when a set is test material (dev set B: or a
held-out set); 5 when a set (the development sets, dev set B) has no lock or does not match it. Nothing is
written but on exit 0 of a decision.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

from research_agent.agent.rules import ruled_record
from research_agent.eval import special_dev
from research_agent.eval.lock import (
    DevSet,
    JsonlError,
    LockError,
    RefusedTestSet,
    check_lock,
    lock_required,
    named_set,
    read_jsonl,
    refuse_test_sets,
    sha256_of,
)
from research_agent.eval.metrics import NORMALIZED_SCORER, score_items
from research_agent.family.core import GREEDY, Mode
from research_agent.llm.generation import GenerationSettings

REPO = Path(__file__).resolve().parents[1]


def _load_vote_offline():
    """scripts/vote_offline.py as a module of its own name (scripts/ is no package): its checks of a set's
    items and its reader of a run's records, as the rules are applied offline."""
    spec = importlib.util.spec_from_file_location("_vote_offline_shared",
                                                  REPO / "scripts" / "vote_offline.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vote_offline = _load_vote_offline()


def _load_runner():
    """scripts/run_special_dev.py as a module of its own name, as scripts/calibrate_special_dev.py loads it:
    dev set B opened as the runner opens it (load_items: test material and the held-out sets refused, the lock
    checked, every item checked)."""
    spec = importlib.util.spec_from_file_location("_run_special_dev_shared",
                                                  REPO / "scripts" / "run_special_dev.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()

OUT = REPO / "results" / "decision-m8.json"  # the decision's one place (module docstring)
SETS = ("data/eval/dev.jsonl", "data/eval/v2/dev-v2.jsonl")  # in this order
SEEDS = ("s0", "s1")  # the selection runs, seeds 0 and 1
BASELINE, BASELINE_STEP = Path("results/checkpoint-selection-ft3.json"), 1635
CALIBRATION = Path("results/special-dev-calibration-m8.json")  # scripts/calibrate_special_dev.py's --out
DEV_SET_B = "data/eval/special-dev-b.jsonl"  # the gates' set (scripts/build_special_mix.py writes it)
B0, FT3 = "b0", "ft3"
SCORER = NORMALIZED_SCORER
PG1_MARGIN, PG1_STRICT, PG2_STRICT = Fraction(1, 20), Fraction(3, 4), Fraction(7, 10)
PG3_EXTRA, PG4_DROP, MG_DROP = 2, Fraction(1, 20), 10
GATES = ("PG1", "PG2", "PG3", "PG4")
RULES = {
    "sum": "Per checkpoint, the successes of seeds 0 and 1 over data/eval/dev.jsonl and "
           "data/eval/v2/dev-v2.jsonl under the normalized scorer, the runtime rules applied to every record "
           "offline; the higher sum first, a tie to the later step.",
    "PG1": "Dev set B lenient special rate >= B0's on B + 0.05; when K1 failed, the strict special rate "
           ">= 0.75.",
    "PG2": "Dev set B strict special rate >= 0.70.",
    "PG3": "False refusals on dev set B's valid items <= FT3's + 2.",
    "PG4": "Exact-call rate on dev set B's valid items >= FT3's - 0.05, and zh alterations <= FT3's.",
    "MG": "The chosen checkpoint's sum >= FT3 step 1635's sum (the same metric) - 10.",
    "decision": "The first checkpoint of the ranking passing PG1 to PG4 is chosen; GO when MG passes, NO-GO "
                "when it fails or no checkpoint passes the gates.",
    "dev_set_b": "Run for the ranking's checkpoints in order until one passes PG1 to PG4; a checkpoint "
                 "reached without a run stops the decision until it is run.",
    "counts": "The gates read every dev set B run's records scored again against dev set B by "
              "research_agent.eval.special_dev as it is now; every record ok, no item the scorer fails on, "
              "and the run's summary and, for B0 and FT3, the calibration's summaries counting the same.",
}
ABOUT = ("M8's choice of the FT5 checkpoint for its one ACEBench run (docs/m8-计划.md WP4), made "
         "mechanically by scripts/decide_m8.py with the rules written before FT5 had a score.")
SPECIAL_KEYS, VALID_KEYS = ("n", "strict", "lenient"), ("n", "exact", "false_refusals", "zh_altered")
# the counts the gates read, by their keys here and their names in a summary (special.lenient)
COUNTS = {(f"{group}_n" if key == "n" else key): f"{group}.{key}"
          for group, keys in (("special", SPECIAL_KEYS), ("valid", VALID_KEYS)) for key in keys}
K1_VERDICTS = ("holds", "passed")
K1_POINTS = Fraction(1, 20)  # K1 (scripts/calibrate_special_dev.py): B0's lenient special rate 5 points above
# What a selection run's manifest must share with FT3's of the same seed and set, besides its generation and
# family: the agent prompt, the tool schemas and the lora scale (module docstring: the candidates).
SAME_METRIC = ("prompt_hash", "prompt_version", "schema_hash", "lora_scale")
# A dev set B run (module docstring: where the summaries come from): scripts/run_special_dev.py's lora scales
# (B0's, an adapter system's), the reasoning budget and the generation's mode and decoding.
B0_SCALE, FT_SCALE = 0, 1
REASONING_BUDGET = 1280
THINK = str(Mode.THINK)
GREEDY_DECODING = GenerationSettings(mode=Mode.THINK, decoding=GREEDY).record()["decoding"]
DEV_B_SLOTS = 1  # the server's slots in a dev set B run (scripts/run_special_dev.py: -np 1)
# What every dev set B run must share with B0's besides the generation, each recorded (module docstring: where
# the summaries come from): the base model and the llama.cpp build the server reported, and the family that
# rendered the prompts.
DEV_B_SAME = ("server_props.model_path", "server_props.build_info", "family.name", "family.template_sha256")
DEV_B_RUN = "runs/m8/special-dev/b-ft5-{step}.jsonl"  # where a candidate's dev set B run goes (M8's paths)


class Malformed(ValueError):
    """A selection file, run, summary or calibration file that is not of its shape (exit 1)."""


class Disagreement(ValueError):
    """A missing input or one that is not the one the decision needs (exit 2)."""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Decide once which FT5 checkpoint goes to M8's ACEBench run.")
    ap.add_argument("--selection", type=Path, required=True,
                    help="FT5's selection file (scripts/select_checkpoint.py), runs s0 and s1 over both sets")
    ap.add_argument("--baseline", type=Path, default=BASELINE,
                    help=f"FT3's selection file (default {BASELINE.as_posix()})")
    ap.add_argument("--baseline-step", type=int, default=BASELINE_STEP,
                    help=f"FT3's checkpoint (default {BASELINE_STEP})")
    ap.add_argument("--special", action="append", default=[], metavar="LABEL=PATH",
                    help="dev set B's summary of b0, ft3 (both always) or a candidate step (once each)")
    ap.add_argument("--calibration", type=Path, default=CALIBRATION,
                    help="the calibration file, which tells whether K1 failed (the PG1 fallback; default "
                         f"{CALIBRATION.as_posix()})")
    ap.add_argument("--rank-only", action="store_true",
                    help="print the ranking and FT3's MG baseline alone: no dev set B, nothing written")
    return ap


def posix(path: str | Path) -> str:
    return Path(path).as_posix()


def read_json(path: Path, what: str) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise Disagreement(f"{what} {posix(path)} cannot be read: {exc.strerror or exc}") from None
    except ValueError as exc:
        raise Malformed(f"{what} {posix(path)} is not JSON: {exc}") from None
    if not isinstance(value, dict):
        raise Malformed(f"{what} {posix(path)} is not a JSON object")
    return value


def special_paths(values: list[str], steps: list[int]) -> dict[str, Path]:
    """--special's LABEL=PATH, by label: b0 and ft3, and any of the candidate steps, once each and nothing
    else."""
    known = [B0, FT3, *(str(step) for step in steps)]
    paths: dict[str, Path] = {}
    for value in values:
        label, sep, path = value.partition("=")
        if not sep or not path:
            raise Disagreement(f"--special {value}: LABEL=PATH is needed")
        if label not in known:
            raise Disagreement(f"--special {label}: no candidate checkpoint {label} (labels: "
                               f"{', '.join(known)})")
        if label in paths:
            raise Disagreement(f"--special {label} is given twice")
        paths[label] = Path(path)
    missing = [label for label in (B0, FT3) if label not in paths]
    if missing:
        raise Disagreement("no dev set B summary of " + ", ".join(missing) + " (--special LABEL=PATH): B0's "
                           "and FT3's runs are the gates' references")
    return paths


def summary_counts(summary: object, what: str) -> dict:
    """The counts the gates read of a special_dev summarize object (module docstring: the gates), by their
    keys in COUNTS; Malformed, naming what, when one is not there."""
    counts = {}
    for key, name in COUNTS.items():
        group, count = name.split(".")
        block = summary.get(group) if isinstance(summary, dict) else None
        number = block.get(count) if isinstance(block, dict) else None
        if type(number) is not int or number < 0 or (count == "n" and number == 0):
            raise Malformed(f"{what} has no count {name} (special_dev's summarize object)")
        counts[key] = number
    return counts


def dev_b_run(label: str, path: Path) -> dict:
    """A run over dev set B by its summary (module docstring: where the summaries come from): the counts and
    the scoring errors its summary records, the set it records, the records (path, SHA-256) and the run
    manifest beside them. Malformed for a summary without its counts; Disagreement for one that names no
    records, or whose records or manifest cannot be read."""
    value = read_json(path, f"the summary of {label}")
    summary = value.get("summary")
    counts = summary_counts(summary, f"the summary of {label} ({posix(path)}), under \"summary\",")
    name = value.get("records")
    if not isinstance(name, str) or not name:
        raise Disagreement(f"the summary of {label} ({posix(path)}) names no records: dev set B's summaries "
                           "are scripts/run_special_dev.py's <stem>.summary.json, beside the records named")
    records = path.parent / name
    try:
        records_sha256 = sha256_of(records)
    except OSError as exc:
        raise Disagreement(f"the records of {label} {posix(records)} cannot be read: "
                           f"{exc.strerror or exc}") from None
    manifest = records.with_name(records.stem + ".manifest.json")
    return {"path": path, "sha256": sha256_of(path), "counts": counts,
            "summary_errors": summary.get("scoring_errors"), "items": value.get("items"),
            "items_sha256": value.get("items_sha256"), "records": records, "records_sha256": records_sha256,
            "manifest_path": manifest, "manifest": read_json(manifest, f"the run manifest of {label}")}


def _number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def field(manifest: dict, name: str) -> object:
    """A run manifest's value at a dotted name (server_props.model_path); None when it is not there."""
    value: object = manifest
    for key in name.split("."):
        value = value.get(key) if isinstance(value, dict) else None
    return value


def adapter_digests(manifest: dict) -> object:
    """The SHA-256 of every adapter a run manifest lists (eval.systems.loaded_adapters), in order; the
    recorded value as it is when it is no list."""
    adapters = manifest.get("adapters")
    if not isinstance(adapters, list):
        return adapters
    return [a.get("sha256") if isinstance(a, dict) else a for a in adapters]


def is_dev_set_b(path: object) -> bool:
    """Whether a recorded items path is dev set B: DEV_SET_B as written, or the same file."""
    return isinstance(path, str) and (posix(path) == DEV_SET_B
                                      or named_set(path, [DEV_SET_B], (Path.cwd(), REPO)) is not None)


def run_problems(run: dict, *, adapter: str | None, lock: str) -> list[str]:
    """Why a dev set B run is not the one its label needs, a line per field (module docstring: where the
    summaries come from); adapter is the SHA-256 of the GGUF it must have applied, None for B0's, and lock dev
    set B's SHA-256."""
    manifest, problems = run["manifest"], []
    scale, named = manifest.get("lora_scale"), adapter_digests(manifest)
    if adapter is None:
        if not (_number(scale) and scale == B0_SCALE) and named != []:
            problems.append(f"lora_scale {scale!r} with adapters {named!r}, where B0's lora scale {B0_SCALE} "
                            "or no adapter")
    else:
        if named != [adapter]:
            problems.append(f"adapters {named!r}, where the GGUF {adapter} alone")
        if not (_number(scale) and scale == FT_SCALE):
            problems.append(f"lora_scale {scale!r}, where {FT_SCALE} (the adapter applied)")
    for where, items, digest in (("summary", run["items"], run["items_sha256"]),
                                 ("manifest", manifest.get("items"), manifest.get("items_sha256"))):
        if not is_dev_set_b(items):
            problems.append(f"{where} items {items!r}, where {DEV_SET_B}")
        if digest != lock:
            problems.append(f"{where} items_sha256 {digest!r}, where dev set B's lock {lock}")
    done, total = manifest.get("items_run"), manifest.get("items_total")
    if type(done) is not int or done != total:
        problems.append(f"items_run {done!r} of items_total {total!r}, where every item")
    if manifest.get("statuses") != {"ok": total}:
        problems.append(f"statuses {manifest.get('statuses')!r}, where every item answered ok ({{'ok': "
                        f"{total!r}}}: no error, no context overflow)")
    if run["summary_errors"] not in (None, 0):
        problems.append(f"summary scoring_errors {run['summary_errors']!r}, where none (every item scored)")
    if field(manifest, "server_props.total_slots") != DEV_B_SLOTS:
        problems.append(f"server_props.total_slots {field(manifest, 'server_props.total_slots')!r}, where "
                        f"{DEV_B_SLOTS}")
    generation = manifest.get("generation") if isinstance(manifest.get("generation"), dict) else {}
    if generation.get("mode") != THINK:
        problems.append(f"generation.mode {generation.get('mode')!r}, where {THINK!r}")
    if generation.get("decoding") != GREEDY_DECODING:
        problems.append(f"generation.decoding {generation.get('decoding')!r}, where greedy "
                        f"{GREEDY_DECODING!r}")
    if generation.get("reasoning_budget") != REASONING_BUDGET:
        problems.append(f"generation.reasoning_budget {generation.get('reasoning_budget')!r}, where "
                        f"{REASONING_BUDGET}")
    return problems


def dev_b() -> tuple[list[dict], str]:
    """Dev set B's items, opened as scripts/run_special_dev.py opens them (module docstring: the counts), and
    its SHA-256, its lock's: Disagreement when the file is not there, LockError without its lock or when it
    does not match it (RefusedTestSet for test material or a held-out set), Malformed for malformed items."""
    if not Path(DEV_SET_B).is_file():
        raise Disagreement(f"no {DEV_SET_B}: the dev set B runs are checked against the set and its lock, "
                           "and scored again against it")
    try:
        items, locked = runner.load_items(Path(DEV_SET_B))
    except runner.MalformedItems as exc:
        raise Malformed(f"{DEV_SET_B}: malformed items: {exc}") from None
    if not locked:
        raise LockError(f"{Path(DEV_SET_B).name} has no lock: the dev set B runs are checked against the set "
                        "as it was locked")
    return items, check_lock(DEV_SET_B)


def check_runs(runs: dict[str, dict], adapters: dict[str, str | None], lock: str) -> None:
    """Disagreement unless every dev set B run is the one its label needs, all of one generation and of B0's
    base model, build and family, and every summary counts the same special and valid items (module
    docstring: where the summaries come from)."""
    for label, run in runs.items():
        problems = run_problems(run, adapter=adapters[label], lock=lock)
        if problems:
            raise Disagreement(f"the dev set B run of {label} (summary {posix(run['path'])}, manifest "
                               f"{posix(run['manifest_path'])}) is not the run {label} needs: "
                               + "; ".join(problems))
    generations = {label: run["manifest"].get("generation") for label, run in runs.items()}
    other = [label for label, generation in generations.items() if generation != generations[B0]]
    if other:
        raise Disagreement("the dev set B runs were not generated alike (generation): " + "; ".join(
            f"{label} {generations[label]!r}" for label in [B0, *other]))
    for name in DEV_B_SAME:
        values = {label: field(run["manifest"], name) for label, run in runs.items()}
        other = [label for label, value in values.items() if value != values[B0]]
        if values[B0] is None or other:
            raise Disagreement(f"the dev set B runs were not made on one base model, build and family "
                               f"({name}, recorded in every run manifest and the same as B0's): " + "; ".join(
                                   f"{label} {values[label]!r}" for label in [B0, *other]))
    sizes = {label: (run["counts"]["special_n"], run["counts"]["valid_n"]) for label, run in runs.items()}
    if len(set(sizes.values())) > 1:
        raise Disagreement("the summaries are not of one set: (special n, valid n) " + ", ".join(
            f"{label} {size}" for label, size in sizes.items()))


def calibration_problems(calibration: dict, *, lock: str, runs: dict[str, dict]) -> list[str]:
    """Why the calibration file is not of these runs, a line per field (module docstring: where the summaries
    come from)."""
    items = calibration.get("items")
    digest = items.get("sha256") if isinstance(items, dict) else None
    problems = [] if digest == lock else [f"items.sha256 {digest!r}, where dev set B's lock {lock}"]
    recorded = calibration.get("runs") if isinstance(calibration.get("runs"), dict) else {}
    for system, label in (("b0", B0), ("ft", FT3)):
        entry = recorded.get(system)
        value = entry.get("sha256") if isinstance(entry, dict) else None
        if value != runs[label]["records_sha256"]:
            problems.append(f"runs.{system}.sha256 {value!r}, where the SHA-256 of --special {label}'s "
                            f"records {posix(runs[label]['records'])} ({runs[label]['records_sha256']})")
    return problems


def rescore(label: str, run: dict, items: list[dict]) -> dict:
    """A dev set B run's records scored again against dev set B by the scorer as it is now (module docstring:
    the counts): {"rescored": the counts the gates read, "scoring_errors"}. Malformed for records that are not
    one of every item; Disagreement for a record that is not ok, an item the scorer fails on, or a summary
    that counted otherwise."""
    where = f"the records of {label} {posix(run['records'])}"
    try:
        records = read_jsonl(run["records"])
    except OSError as exc:
        raise Disagreement(f"{where} cannot be read: {exc.strerror or exc}") from None
    except JsonlError as exc:
        raise Malformed(f"{where}: {exc}") from None
    try:
        summary = special_dev.summarize(special_dev.score_records(records, items), items)
    except ValueError as exc:
        raise Malformed(f"{where} are not one record of every item of {DEV_SET_B}: {exc}") from None
    problems = []
    statuses = dict(sorted(Counter(str(record.get("status")) for record in records).items()))
    if statuses != {"ok": len(items)}:
        problems.append(f"record statuses {statuses}, where every one ok")
    if summary["scoring_errors"]:
        problems.append(f"scoring_errors {summary['scoring_errors']}: items the scorer fails on, where none")
    counts = summary_counts(summary, f"{where} scored again")
    differ = [f"{name} {run['counts'][key]} in its summary, where its records scored again give {counts[key]}"
              for key, name in COUNTS.items() if run["counts"][key] != counts[key]]
    if differ:
        problems.append("; ".join(differ) + " (the summary is of other records, or another version of the "
                        "scorer wrote it: write it again from these records)")
    if problems:
        raise Disagreement(f"the dev set B run of {label} (summary {posix(run['path'])}, records "
                           f"{posix(run['records'])}) is not what the scorer counts now "
                           f"({special_dev.__name__}): " + "; ".join(problems))
    return {"rescored": counts, "scoring_errors": summary["scoring_errors"]}


def calibration_counts(calibration: dict, runs: dict[str, dict], *, fallback: bool) -> list[str]:
    """Why the calibration file is not of B0's and FT3's runs as the scorer counts them now, a line per field:
    its summaries must give their counts scored again, and the fallback it tells must be K1's of those
    counts (module docstring: where the summaries come from)."""
    problems = []
    summaries = calibration.get("summaries") if isinstance(calibration.get("summaries"), dict) else {}
    for system, label in (("b0", B0), ("ft", FT3)):
        rescored = runs[label]["rescored"]
        try:
            counts = summary_counts(summaries.get(system), f"summaries.{system}")
        except Malformed:
            problems.append(f"summaries.{system} holds no counts of special_dev's summary, where those of "
                            f"{label}'s records scored again")
            continue
        problems += [f"summaries.{system} {name} {counts[key]}, where {label}'s records scored again give "
                     f"{rescored[key]}" for key, name in COUNTS.items() if counts[key] != rescored[key]]
    b0, ft3 = runs[B0]["rescored"], runs[FT3]["rescored"]
    holds = Fraction(b0["lenient"], b0["special_n"]) - Fraction(ft3["lenient"], ft3["special_n"]) >= K1_POINTS
    if fallback == holds:
        problems.append(f"K1 {'holds' if holds else 'fails'} on the runs scored again (lenient special: B0 "
                        f"{b0['lenient']}/{b0['special_n']}, FT3 {ft3['lenient']}/{ft3['special_n']}; K1 "
                        f"asks {float(K1_POINTS):g} between them), where the file's fallback is {fallback}")
    return problems


def pg1_fallback(calibration: dict, path: Path) -> bool:
    """Whether K1 failed, so PG1 reads the strict special rate (module docstring: the gates)."""
    flag = calibration.get("pg1_fallback")
    k1 = calibration.get("K1", calibration.get("k1"))
    verdict = k1 if isinstance(k1, bool) else next(
        (k1[key] for key in K1_VERDICTS if isinstance(k1, dict) and isinstance(k1.get(key), bool)), None)
    if "pg1_fallback" in calibration and not isinstance(flag, bool):
        raise Disagreement(f"{posix(path)}: pg1_fallback is {flag!r}, neither true nor false")
    if isinstance(flag, bool) and verdict is not None and flag == verdict:
        raise Disagreement(f"{posix(path)}: pg1_fallback {flag} disagrees with K1 holding {verdict}")
    if isinstance(flag, bool):
        return flag
    if verdict is None:
        raise Disagreement(f"{posix(path)} tells neither pg1_fallback (true or false) nor K1's verdict "
                           "(\"K1\": {\"holds\": true or false})")
    return not verdict


def item_sets(selection: dict, path: Path, current: dict[str, str]) -> None:
    """Disagreement unless the selection's item sets are SETS at their current SHA-256."""
    sets = selection.get("item_sets")
    named = [entry.get("path") for entry in sets] if isinstance(sets, list) and all(
        isinstance(entry, dict) for entry in sets) else None
    if named != list(SETS):
        raise Disagreement(f"{posix(path)}: item_sets must be {', '.join(SETS)} (the development sets "
                           f"together), not {named}")
    stale = [entry["path"] for entry in sets if entry.get("sha256") != current[entry["path"]]]
    if stale:
        raise Disagreement(f"{posix(path)}: {', '.join(stale)} changed since its runs (another SHA-256 now)")


def rows_of(selection: dict, path: Path) -> list[dict]:
    """The selection's checkpoint rows, each {step, gguf, sha256, runs: {seed: {set: records}}}."""
    rows = selection.get("checkpoints")
    if not isinstance(rows, list) or not rows:
        raise Malformed(f"{posix(path)} holds no checkpoints")
    out = []
    for row in rows:
        try:
            step, gguf, digest = row["step"], row["gguf"], row["sha256"]
            runs = {seed: {s: row["variants"][seed]["sets"][s]["records"] for s in SETS} for seed in SEEDS}
        except (KeyError, TypeError) as exc:
            raise Malformed(f"{posix(path)}: a checkpoint row without its step, GGUF, SHA-256 or the records "
                            f"of runs {', '.join(SEEDS)} over both sets ({type(exc).__name__}: "
                            f"{exc})") from None
        if type(step) is not int or not isinstance(gguf, str) or not isinstance(digest, str):
            raise Malformed(f"{posix(path)}: a checkpoint row whose step, GGUF or SHA-256 is malformed")
        out.append({"step": step, "gguf": gguf, "sha256": digest, "runs": runs})
    if len({row["step"] for row in out}) != len(out):
        raise Malformed(f"{posix(path)} lists a step twice")
    return out


def open_sets() -> dict[str, list[dict]]:
    """SETS opened as the runners open them (module docstring: the candidates)."""
    roots = (Path.cwd(), REPO)
    sets = {}
    for path in SETS:
        refuse_test_sets(path, roots)
        items = DevSet(path).open() if lock_required(path, roots) else read_jsonl(path)
        vote_offline.refuse_items(items, Path(path), aggregate_only=False)
        vote_offline.check_ids(items, aggregate_only=False)
        sets[path] = items
    shared = {item["id"] for item in sets[SETS[0]]} & {item["id"] for item in sets[SETS[1]]}
    if shared:
        raise Disagreement(f"{len(shared)} item id(s) are in both sets")
    return sets


def run_manifest(records: str) -> tuple[Path, dict]:
    path = Path(records).with_name(Path(records).stem + ".manifest.json")
    return path, read_json(path, "the run manifest")


def _model_path(manifest: dict) -> object:
    props = manifest.get("server_props")
    return props.get("model_path") if isinstance(props, dict) else None


def manifest_problems(manifest: dict, *, sha256: str, set_path: str, set_sha256: str, n: int,
                      reference: dict) -> list[str]:
    """Why a run's manifest does not fit its checkpoint, its set and FT3's run of the same seed and set
    (reference, its manifest), a line per field."""
    problems = []
    adapters = manifest.get("adapters")
    named = [a.get("sha256") for a in adapters if isinstance(a, dict)] if isinstance(adapters, list) else None
    if named != [sha256]:
        problems.append(f"adapters {adapters!r}, where the checkpoint's GGUF alone ({sha256})")
    if manifest.get("items") != set_path or manifest.get("items_sha256") != set_sha256:
        problems.append(f"items {manifest.get('items')!r} at {manifest.get('items_sha256')!r}, where "
                        f"{set_path} at {set_sha256}")
    if manifest.get("items_run") != n or manifest.get("items_total") != n:
        problems.append(f"items_run {manifest.get('items_run')!r} of {manifest.get('items_total')!r}, where "
                        f"all {n}")
    if manifest.get("generation") != reference.get("generation"):
        problems.append(f"generation {manifest.get('generation')!r}, where FT3's "
                        f"{reference.get('generation')!r}")
    then, family = manifest.get("family") or {}, reference.get("family") or {}
    if any(then.get(key) != family.get(key) for key in ("name", "template_sha256")):
        problems.append(f"family {then.get('name')!r}, where FT3's {family.get('name')!r}")
    problems += [f"{key} {manifest.get(key)!r}, where FT3's {reference.get(key)!r}" for key in SAME_METRIC
                 if manifest.get(key) != reference.get(key)]
    if _model_path(manifest) != _model_path(reference):
        problems.append(f"server_props.model_path {_model_path(manifest)!r}, where FT3's "
                        f"{_model_path(reference)!r}")
    return problems


def successes(records_path: str, items: list[dict]) -> tuple[int, int]:
    """The successes of a run's records under SCORER, with the runtime rules applied offline and without."""
    try:
        records = vote_offline.run_records(Path(records_path), items)
    except OSError as exc:
        raise Disagreement(f"{posix(records_path)} cannot be read: {exc.strerror or exc}") from None
    except (JsonlError, ValueError) as exc:
        raise Malformed(str(exc)) from None
    ruled = [ruled_record(record, item["messages"]) for record, item in zip(records, items, strict=True)]
    try:
        with_rules, without = (sum(bool(s["success"]) for s in score_items(rows, items, SCORER))
                               for rows in (ruled, records))
    except Exception as exc:  # noqa: BLE001  the scorer refuses a record it cannot read
        raise Malformed(f"{posix(records_path)} cannot be scored: {type(exc).__name__}: {exc}") from None
    return with_rules, without


def scored(row: dict, sets: dict[str, list[dict]], current: dict[str, str], reference: dict) -> dict:
    """A checkpoint's successes per seed (with the rules, and without), after its manifests are checked
    against reference: per (seed, set) FT3's run manifest."""
    out = {}
    for seed in SEEDS:
        ruled = plain = 0
        for set_path, items in sets.items():
            records = row["runs"][seed][set_path]
            path, manifest = run_manifest(records)
            problems = manifest_problems(manifest, sha256=row["sha256"], set_path=set_path,
                                         set_sha256=current[set_path], n=len(items),
                                         reference=reference[(seed, set_path)])
            if problems:
                raise Disagreement(f"{posix(path)} (checkpoint {row['step']}, {seed}): "
                                   + "; ".join(problems))
            with_rules, without = successes(records, items)
            ruled, plain = ruled + with_rules, plain + without
        out[seed], out[f"{seed}_without_rules"] = ruled, plain
    out["sum"] = sum(out[seed] for seed in SEEDS)
    out["sum_without_rules"] = sum(out[f"{seed}_without_rules"] for seed in SEEDS)
    return out


def gates_of(c: dict, b0: dict, ft3: dict, *, fallback: bool) -> dict:
    """PG1 to PG4 of a candidate's dev set B counts, against B0's and FT3's (module docstring: the gates)."""
    strict = Fraction(c["strict"], c["special_n"])
    if fallback:
        pg1 = {"measure": "strict special (K1 failed)", "value": float(strict),
               "threshold": float(PG1_STRICT), "passed": strict >= PG1_STRICT}
    else:
        lenient = Fraction(c["lenient"], c["special_n"])
        bound = Fraction(b0["lenient"], b0["special_n"]) + PG1_MARGIN
        pg1 = {"measure": "lenient special", "value": float(lenient), "threshold": float(bound),
               "passed": lenient >= bound}
    pg2 = {"measure": "strict special", "value": float(strict), "threshold": float(PG2_STRICT),
           "passed": strict >= PG2_STRICT}
    refusals = ft3["false_refusals"] + PG3_EXTRA
    pg3 = {"measure": "false refusals", "value": c["false_refusals"], "threshold": refusals,
           "passed": c["false_refusals"] <= refusals}
    exact, bound = Fraction(c["exact"], c["valid_n"]), Fraction(ft3["exact"], ft3["valid_n"]) - PG4_DROP
    pg4 = {"measure": "exact calls and zh alterations", "exact": float(exact),
           "exact_threshold": float(bound), "zh_altered": c["zh_altered"], "zh_threshold": ft3["zh_altered"],
           "passed": exact >= bound and c["zh_altered"] <= ft3["zh_altered"]}
    return {"PG1": pg1, "PG2": pg2, "PG3": pg3, "PG4": pg4}


def gate_text(name: str, gate: dict) -> str:
    verdict = "pass" if gate["passed"] else "FAIL"
    if name == "PG4":
        return (f"PG4 exact {gate['exact']:.4f} >= {gate['exact_threshold']:.4f} and zh altered "
                f"{gate['zh_altered']} <= {gate['zh_threshold']}: {verdict}")
    if name == "PG3":
        return f"PG3 false refusals {gate['value']} <= {gate['threshold']}: {verdict}"
    return f"{name} {gate['measure']} {gate['value']:.4f} >= {gate['threshold']:.4f}: {verdict}"


def counts_text(numbers: dict) -> str:
    return (f"strict {numbers['strict']}/{numbers['special_n']}, lenient {numbers['lenient']}/"
            f"{numbers['special_n']}, exact {numbers['exact']}/{numbers['valid_n']}, false refusals "
            f"{numbers['false_refusals']}, zh altered {numbers['zh_altered']}")


def read_inputs(args: argparse.Namespace) -> dict:
    """Everything the decision (or, with --rank-only, the ranking) reads, checked (module docstring);
    Malformed, Disagreement or the lock's errors otherwise. Dev set B, its runs and the calibration file are
    read before the selection runs are scored, which takes longer; the dev set B runs are scored again once
    every one is shown to be the run its label needs."""
    selection, baseline = read_json(args.selection, "the selection"), read_json(args.baseline, "the baseline")
    current = {path: sha256_of(path) for path in SETS if Path(path).is_file()}
    missing = [path for path in SETS if path not in current]
    if missing:
        raise Disagreement(f"no {', '.join(missing)} to score the runs with")
    for data, path in ((selection, args.selection), (baseline, args.baseline)):
        item_sets(data, path, current)
    candidates = rows_of(selection, args.selection)
    ft3_row = next((row for row in rows_of(baseline, args.baseline) if row["step"] == args.baseline_step),
                   None)
    if ft3_row is None:
        raise Disagreement(f"{posix(args.baseline)} holds no step {args.baseline_step}")
    inputs = {"selection": selection, "current": current, "ft3_row": ft3_row}
    if not args.rank_only:
        paths = special_paths(args.special, [row["step"] for row in candidates])
        runs = {label: dev_b_run(label, path) for label, path in paths.items()}
        items, lock = dev_b()
        adapters = {B0: None, FT3: ft3_row["sha256"],
                    **{str(row["step"]): row["sha256"] for row in candidates}}
        check_runs(runs, adapters, lock)
        calibration = read_json(args.calibration, "the calibration file")
        problems = calibration_problems(calibration, lock=lock, runs=runs)
        if problems:
            raise Disagreement(f"{posix(args.calibration)} is not the calibration of these runs: "
                               + "; ".join(problems))
        fallback = pg1_fallback(calibration, args.calibration)
        for label, run in runs.items():
            run |= rescore(label, run, items)
        problems = calibration_counts(calibration, runs, fallback=fallback)
        if problems:
            raise Disagreement(f"{posix(args.calibration)} is not the calibration of these runs as the "
                               "scorer counts them now: " + "; ".join(problems))
        inputs |= {"runs": runs, "lock": lock, "fallback": fallback}
    sets = open_sets()
    reference = {(seed, set_path): run_manifest(ft3_row["runs"][seed][set_path])[1]
                 for seed in SEEDS for set_path in SETS}
    return inputs | {"sets": sets, "ft3": scored(ft3_row, sets, current, reference),
                     "results": [{**{key: row[key] for key in ("step", "gguf", "sha256")},
                                  **scored(row, sets, current, reference)} for row in candidates]}


def run_record(run: dict) -> dict:
    """Where a dev set B run's numbers come from, as the decision records it."""
    manifest = run["manifest"]
    return {"path": posix(run["path"]), "sha256": run["sha256"],
            "records": {"path": posix(run["records"]), "sha256": run["records_sha256"]},
            "manifest": {"path": posix(run["manifest_path"]), "sha256": sha256_of(run["manifest_path"])},
            "adapters": adapter_digests(manifest), "lora_scale": manifest.get("lora_scale"),
            "statuses": manifest.get("statuses"), "counts": run["rescored"],
            "scoring_errors": run["scoring_errors"]}


def scorer_record() -> dict:
    """The scorer the dev set B runs were scored again with: research_agent.eval.special_dev's file by its
    SHA-256."""
    path = Path(special_dev.__file__).resolve()
    try:
        name = path.relative_to(REPO).as_posix()
    except ValueError:  # an installed copy, not this repository's
        name = path.as_posix()
    return {"module": special_dev.__name__, "path": name, "sha256": sha256_of(path)}


def ranking_header(sets: dict[str, list[dict]]) -> str:
    n = sum(len(items) for items in sets.values())
    return (f"ranking ({SCORER} scorer, the runtime rules applied offline; seeds 0 + 1 over "
            f"{' and '.join(SETS)}, {n} items a seed; a tie to the later step)")


def sum_text(r: dict) -> str:
    return f"s0 {r['s0']} + s1 {r['s1']} = {r['sum']} (without the rules {r['sum_without_rules']})"


def dev_b_command(row: dict) -> str:
    """How a candidate's dev set B run is made (module docstring: the procedure)."""
    out = DEV_B_RUN.format(step=row["step"])
    return (f"scripts/run_special_dev.py {DEV_SET_B} --system ft --out {out} on a server that loads "
            f"{row['gguf']} (sha256 {row['sha256']}) alone, then --special "
            f"{row['step']}={Path(out).with_suffix('').as_posix()}.summary.json")


def rank_only(inputs: dict) -> int:
    """--rank-only: the ranking and FT3's MG baseline, printed (module docstring: the procedure)."""
    ft3_row, ft3 = inputs["ft3_row"], inputs["ft3"]
    threshold = ft3["sum"] - MG_DROP
    ranking = sorted(inputs["results"], key=lambda r: (-r["sum"], -r["step"]))
    print(ranking_header(inputs["sets"]) + "; dev set B not read (--rank-only)")
    for rank, r in enumerate(ranking, 1):
        print(f"  {rank}. step {r['step']}: {sum_text(r)}; MG {r['sum']} >= {threshold}: "
              f"{'pass' if r['sum'] >= threshold else 'FAIL'}")
    print(f"FT3 step {ft3_row['step']}: {sum_text(ft3)}; MG: the chosen checkpoint's sum at least FT3's "
          f"{ft3['sum']} - {MG_DROP} = {threshold}")
    print(f"next: dev set B for checkpoint {ranking[0]['step']}: {dev_b_command(ranking[0])}")
    print("nothing written (--rank-only)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    if args.rank_only and args.special:
        print("FAIL: --rank-only reads no dev set B summary (--special): it ranks the checkpoints alone")
        return 2
    if not args.rank_only and OUT.exists():
        print(f"FAIL: {OUT.as_posix()} exists: WP4 is decided once (docs/m8-计划.md), and nothing was read")
        return 2
    try:
        inputs = read_inputs(args)
    except RefusedTestSet as exc:
        print(f"FAIL: {exc}")
        return 4
    except LockError as exc:
        print(f"FAIL: {exc}")
        return 5
    except Malformed as exc:
        print(f"FAIL: {exc}")
        return 1
    except Disagreement as exc:
        print(f"FAIL: {exc}")
        return 2
    except (JsonlError, ValueError) as exc:  # a set's items: malformed, or holding no id of their own
        print(f"FAIL: malformed items: {exc}")
        return 1
    if args.rank_only:
        return rank_only(inputs)
    runs, fallback, ft3_row, ft3 = inputs["runs"], inputs["fallback"], inputs["ft3_row"], inputs["ft3"]
    results, sets = inputs["results"], inputs["sets"]
    numbers = {label: run["rescored"] for label, run in runs.items()}  # the records scored again
    scorer = scorer_record()
    for result in results:
        counts = numbers.get(str(result["step"]))
        result["special_dev"] = counts
        result["gates"] = None if counts is None else gates_of(counts, numbers[B0], numbers[FT3],
                                                               fallback=fallback)
        result["passes"] = None if counts is None else all(result["gates"][name]["passed"] for name in GATES)
    ranking = sorted(results, key=lambda r: (-r["sum"], -r["step"]))
    walk, chosen, needed = [], None, None
    for r in ranking:
        if r["passes"] is None:  # reached before any passed: its run decides what comes next
            needed = r
            break
        walk.append(r["step"])
        if r["passes"]:
            chosen = r
            break
    threshold = ft3["sum"] - MG_DROP
    print(ranking_header(sets) + ", with dev set B's gates"
          + (" (PG1: K1 failed, the strict special rate)" if fallback else ""))
    for rank, r in enumerate(ranking, 1):
        gates = ("; ".join(gate_text(name, r["gates"][name]) for name in GATES) if r["gates"] is not None
                 else "dev set B not run")
        print(f"  {rank}. step {r['step']}: {sum_text(r)}; {gates}")
    print(f"FT3 step {ft3_row['step']}: {sum_text(ft3)}; dev set B: {counts_text(numbers[FT3])}")
    print(f"B0 on dev set B: {counts_text(numbers[B0])}")
    print(f"dev set B counts: every run's records scored again by {scorer['path']} (sha256 "
          f"{scorer['sha256']}), as its summary and the calibration counted them")
    if needed is not None:
        print(f"NEED dev set B for checkpoint {needed['step']}: the walk reaches it before any checkpoint "
              f"passed PG1-PG4 (walked: {', '.join(map(str, walk)) or 'none'}); its sum {needed['sum']} is "
              f"{'at least' if needed['sum'] >= threshold else 'below'} MG's {threshold}")
        print(f"  run {dev_b_command(needed)}, and decide again; nothing written")
        return 3
    mg = None
    if chosen is None:
        verdict, reason = "NO-GO", "no checkpoint passes PG1-PG4"
    else:
        mg = {"sum": chosen["sum"], "ft3_sum": ft3["sum"], "threshold": threshold,
              "passed": chosen["sum"] >= threshold}
        if not mg["passed"]:
            verdict, reason = "NO-GO", (f"MG fails: checkpoint {chosen['step']}'s sum {chosen['sum']} is "
                                        f"below FT3's {ft3['sum']} - {MG_DROP}")
        elif not Path(chosen["gguf"]).is_file() or sha256_of(chosen["gguf"]) != chosen["sha256"]:
            print(f"FAIL: the chosen checkpoint's GGUF {chosen['gguf']} is not there with its SHA-256 "
                  f"{chosen['sha256']}")
            return 2
        else:
            verdict, reason = "GO", (f"checkpoint {chosen['step']} is the first in the ranking to pass "
                                     "PG1-PG4, and MG passes")
    decision = {
        "about": ABOUT, "rules": RULES, "scorer": SCORER,
        "selection": {"path": posix(args.selection), "sha256": sha256_of(args.selection),
                      "run": inputs["selection"].get("run")},
        "item_sets": [{"path": path, "sha256": inputs["current"][path], "n": len(items)}
                      for path, items in sets.items()],
        "dev_set_b": {"path": DEV_SET_B, "sha256": inputs["lock"]},
        "special_dev": {label: run_record(run) for label, run in runs.items()},
        "special_generation": runs[B0]["manifest"].get("generation"),
        "special_base": {name: field(runs[B0]["manifest"], name) for name in DEV_B_SAME},
        "special_scorer": scorer,
        "calibration": {"path": posix(args.calibration), "sha256": sha256_of(args.calibration),
                        "items_sha256": inputs["lock"],
                        "runs": {"b0": runs[B0]["records_sha256"], "ft": runs[FT3]["records_sha256"]}},
        "pg1_fallback": fallback,
        "b0": {"special_dev": numbers[B0]},
        "baseline": {"path": posix(args.baseline), "sha256": sha256_of(args.baseline),
                     "step": ft3_row["step"], "gguf": ft3_row["gguf"], "gguf_sha256": ft3_row["sha256"],
                     **ft3,
                     "special_dev": numbers[FT3]},
        "candidates": results, "ranking": [r["step"] for r in ranking], "walk": walk,
        "chosen": None if chosen is None else {key: chosen[key] for key in ("step", "gguf", "sha256", "sum")},
        "mg": mg, "decision": verdict, "reason": reason}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    if chosen is None:
        print("chosen: none, no checkpoint passes PG1-PG4")
    else:
        print(f"chosen: step {chosen['step']}, the first in the ranking to pass PG1-PG4")
        print(f"MG: {mg['sum']} >= FT3's {mg['ft3_sum']} - {MG_DROP} = {mg['threshold']}: "
              f"{'pass' if mg['passed'] else 'FAIL'}")
    if verdict == "GO":
        print(f"decision: GO: checkpoint {chosen['step']} ({chosen['gguf']}, sha256 {chosen['sha256']})")
    else:
        print(f"decision: NO-GO: {reason}; no ACEBench run, the owner is told")
    print(f"wrote {OUT.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

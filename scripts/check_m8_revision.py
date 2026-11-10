"""Check, once and mechanically, M8's data-revision triggers on FT5's last checkpoint, before the selection's
other runs (docs/m8-计划.md WP3, 数据修订; the preregistration's 数据修订, written before FT5 had a score).

    .venv/Scripts/python.exe scripts/check_m8_revision.py --step <N> \\
        --selection runs/m8/queue/selection-ft5-last-s0.json [--run runs/train/ft5] \\
        [--baseline results/checkpoint-selection-ft3.json] [--baseline-step 1635] \\
        [--dev-a-ft3 runs/m8/special-dev/a-ft3.jsonl] [--dev-a-ft5 runs/m8/special-dev/a-ft5-<N>.jsonl] \\
        [--out results/revision-check-m8.json]

The triggers (RULES). FT5 is its last checkpoint (--step), FT3 the baseline's step 1635 (--baseline-step);
each trigger is compared exactly, as integers or as fractions of the counts, and any one fired means the data
are revised once and FT5 is trained again (the preregistration allows one revision):
  R1 FT5's strict special rate on dev set A < 3/5 (R1_STRICT). "special" is the incorrect, missing and
     irrelevant items of both languages pooled, the priority items left out, as scripts/decide_m8.py reads
     dev set B: special.strict / special.n of research_agent.eval.special_dev's summary;
  R2 FT5's false refusals on dev set A's valid items (valid.false_refusals) minus FT3's >= 3 (R2_EXTRA; the
     plan's "多 3 道以上" counts 3 too);
  R3 FT3's successes on data/eval/dev.jsonl with seed 0 minus FT5's >= 8 (R3_DROP; "低 8 题以上" counts 8
     too), every record with the runtime rules applied offline and scored by the normalized scorer, as
     decide_m8 counts a selection run (decide_m8.successes, its first value).
R1 and R2 read the dev set A runs' records scored again (below), never a summary's own counts.

The checkpoint. --step must be FT5's last checkpoint as FT5 is now, so that the one verdict is never of
another checkpoint: the selection is of FT5's run (its "run" names --run, runs/train/ft5 by default, FT5_RUN;
both resolved, so another spelling of the folder is the same run), and --run is not the baseline's run; --step
is the steps <run>/result.json says were trained (scripts/train.py writes it at the run's end, and the trainer
saves a checkpoint at the last step, as scripts/select_checkpoint.py requires of a complete run); the
selection's row of --step does not have the SHA-256 of the baseline's row of --baseline-step (FT3's GGUF), and
the GGUF the row names is there with the row's SHA-256 now, as decide_m8 checks the chosen checkpoint's GGUF
(FT5 neither trained nor exported again since the selection's runs).

The dev runs. --selection is scripts/select_checkpoint.py's file of a selection over FT5's last checkpoint
alone with one variant, s0 (--variant "s0=--mode think --seed 0 --reasoning-budget 1280 --signals"
--select-by mean:s0 --scorer normalized, items data/eval/dev.jsonl then data/eval/v2/dev-v2.jsonl): its row of
--step gives the checkpoint's GGUF and SHA-256 and the records of s0 over data/eval/dev.jsonl
(variants.s0.sets["data/eval/dev.jsonl"].records). --baseline is FT3's selection file, and its row of
--baseline-step gives that GGUF's SHA-256 and the records of s0 over data/eval/dev.jsonl; no other seed or set
of either file is read. Both files' item_sets must hold data/eval/dev.jsonl at the SHA-256 the file has now,
which must be its lock's: before that comparison the file is refused as a test set
(eval.lock.refuse_test_sets, below) and checked against its lock, so that a dev.jsonl changed since its lock
is refused as dev set A is.
Each run's manifest (<records stem>.manifest.json, as scripts/run_eval.py writes it) must name its
checkpoint's GGUF alone by its SHA-256 and data/eval/dev.jsonl at that SHA-256 with every item run, and record
what FT3's run was measured with (decide_m8.manifest_problems, FT3's manifest the reference for both runs: the
generation, its base seed included; the family's name and template; prompt_hash, prompt_version, schema_hash
and lora_scale; server_props.model_path), so that the two counts are of one metric (同口径).
data/eval/dev.jsonl is opened as decide_m8 opens the development sets: a locked test set (by location or by
its bytes) or a file of the handwritten test items (by location) refused (eval.lock.refuse_test_sets), the
lock checked (it must be there and match), test material refused by content, the item ids distinct.

The dev set A runs. --dev-a-ft3 and --dev-a-ft5 (by default runs/m8/special-dev/a-ft3.jsonl and
a-ft5-<step>.jsonl, DEV_A_RUNS) are scripts/run_special_dev.py's records over dev set A
(data/eval/special-dev-a.jsonl, DEV_SET_A), with <stem>.manifest.json and <stem>.summary.json beside them.
Dev set A is opened as the runner opens it (run_special_dev.load_items: test material and the held-out sets
refused, the lock checked, every item checked), and it must have its lock. Each run must be the run its label
needs, as decide_m8 checks a dev set B run of an adapter: the summary names these records (their file name,
as the runner writes it); the manifest lists FT3's step-1635 GGUF alone (ft3) or the checkpoint's (ft5) by
SHA-256 (adapters), applied at lora scale 1; the set, in the summary and in the manifest, is dev set A (its
path) at the SHA-256 of its lock; every item was run (items_run = items_total) and answered ok (statuses);
the summary counts no item the scorer failed on (scoring_errors); the server had one slot
(server_props.total_slots); the generation is think mode, greedy decoding (family.core.GREEDY) and the
reasoning budget 1,280; and the two runs have one generation and were made on one base model, llama.cpp build
and family (decide_m8.DEV_B_SAME: server_props.model_path, server_props.build_info, family.name and
family.template_sha256, each recorded). Every run's records are scored again against dev set A by the scorer
as it is now (research_agent.eval.special_dev score_records and summarize): every record ok, no item the
scorer fails on, and the summary's counts (decide_m8.COUNTS: special n, strict and lenient; valid n, exact,
false refusals and zh alterations) those of the records scored again, so that a summary of other records or
of another version of the scorer is refused.

Which files. Before it is read, every input (the two selection files; FT5's training result, and its
checkpoint's GGUF once the selection names it; each dev run's records and manifest, once the selection files
name them; each dev set A run's records, manifest and summary) is refused when it is a locked test set (by
location or by its bytes) or a file of the handwritten test items (by location; eval.lock.refuse_test_sets),
or a held-out set (run_special_dev.held_out: by its path, under runs/m7/heldout/, by its name or by its
content).

Writes --out (OUT, results/revision-check-m8.json in this repository, by default; JSON, two-space indent, LF,
UTF-8 without escapes), which is never overwritten: an --out that exists refuses the check before anything is
read. The file: "about"; "rules" (R1 to R3); "step"; "baseline_step"; "inputs", {name: {"path", "sha256"}}
for the selection, the baseline, run_result (<run>/result.json), dev_ft3_records, dev_ft3_manifest,
dev_ft5_records, dev_ft5_manifest, and dev_a_ft3_records, _manifest and _summary and dev_a_ft5_records,
_manifest and _summary; "dev_set_a": {"path", "sha256" (its lock's)}; "ft3" and "ft5": {"gguf",
"gguf_sha256" (the checkpoint's GGUF as its selection file's row records it, FT5's checked against the file),
"dev_s0", "special_n", "special_strict", "valid_n", "false_refusals"}; "triggers": {"R1", "R2", "R3":
{"value", "threshold", "fired"}}, R1's value and threshold fractions in lowest terms ("3/5") with
"value_float" and "threshold_float" beside them, R2's and R3's integers (FT5's false refusals less FT3's,
FT3's successes less FT5's); "triggered". Prints every trigger with its numbers, then "triggered: yes" with
the triggers fired or "triggered: no".

Exit 0 written and no trigger fired; 3 written and at least one fired; 1 on a malformed selection file,
training result, run, summary or item, or a run that does not hold one record of every item; 2 on a usage
error, an existing --out, a missing file and every disagreement (a step a selection file does not hold, a
checkpoint that is not FT5's last as FT5 is now, a set locked again since the runs, a run manifest or summary
that is not of the run the check needs, records not all ok or that the scorer fails on, a summary counting
otherwise than its records scored again, an FT5 run measured otherwise than FT3's); 4 test material or a
held-out set; 5 a set (data/eval/dev.jsonl, dev set A) without its lock or not matching it, before it is
compared with its runs. Nothing is written but on exit 0 or 3.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

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

REPO = Path(__file__).resolve().parents[1]


def _load_decide_m8():
    """scripts/decide_m8.py as a module of its own name (scripts/ is no package), loaded as it loads its own
    siblings: its readers and checks of the selection runs and of the special development runs, its count of
    a run's successes, and through it scripts/vote_offline.py and scripts/run_special_dev.py."""
    spec = importlib.util.spec_from_file_location("_decide_m8_shared", REPO / "scripts" / "decide_m8.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


decide_m8 = _load_decide_m8()
runner, vote_offline = decide_m8.runner, decide_m8.vote_offline  # run_special_dev.py, vote_offline.py
Malformed, Disagreement = decide_m8.Malformed, decide_m8.Disagreement  # exit 1, exit 2

OUT = REPO / "results" / "revision-check-m8.json"  # the check's place by default (module docstring)
FT5_RUN = Path("runs/train/ft5")  # FT5's training run (scripts/train.py --out), --run by default
DEV = decide_m8.SETS[0]  # data/eval/dev.jsonl, R3's set
DEV_SET_A = "data/eval/special-dev-a.jsonl"  # R1's and R2's set (scripts/build_special_mix.py writes it)
SEED = "s0"  # the selection runs read: seed 0
FT3, FT5 = "ft3", "ft5"
DEV_A_RUNS = {FT3: "runs/m8/special-dev/a-ft3.jsonl", FT5: "runs/m8/special-dev/a-ft5-{step}.jsonl"}
R1_STRICT, R2_EXTRA, R3_DROP = Fraction(3, 5), 3, 8
TRIGGERS = ("R1", "R2", "R3")
RULES = {
    "R1": "FT5's strict special rate on dev set A (data/eval/special-dev-a.jsonl) < 3/5: the incorrect, "
          "missing and irrelevant items of both languages pooled, the priority items left out "
          "(special.strict / special.n of the records scored again by research_agent.eval.special_dev), "
          "compared exactly.",
    "R2": "FT5's false refusals on dev set A's valid items (valid.false_refusals of the records scored "
          "again) minus FT3 step 1635's >= 3.",
    "R3": "FT3 step 1635's successes on data/eval/dev.jsonl with seed 0 minus FT5's >= 8, every record with "
          "the runtime rules applied offline and scored by the normalized scorer, as scripts/decide_m8.py "
          "counts a selection run.",
}
ABOUT = ("M8's data-revision check on FT5's last checkpoint (docs/m8-计划.md WP3, 数据修订), made once and "
         "mechanically by scripts/check_m8_revision.py with the triggers written before FT5 had a score: any "
         "trigger fired, the data are revised once and FT5 is trained again.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Check M8's data-revision triggers on FT5's last checkpoint, once.")
    ap.add_argument("--step", type=int, required=True, help="FT5's last checkpoint, by its step")
    ap.add_argument("--selection", type=Path, required=True,
                    help="the selection over FT5's last checkpoint, run s0 (scripts/select_checkpoint.py)")
    ap.add_argument("--run", type=Path, default=FT5_RUN,
                    help=f"FT5's training run, the selection's (default {FT5_RUN.as_posix()})")
    ap.add_argument("--baseline", type=Path, default=decide_m8.BASELINE,
                    help=f"FT3's selection file (default {decide_m8.BASELINE.as_posix()})")
    ap.add_argument("--baseline-step", type=int, default=decide_m8.BASELINE_STEP,
                    help=f"FT3's checkpoint (default {decide_m8.BASELINE_STEP})")
    ap.add_argument("--dev-a-ft3", type=Path, default=Path(DEV_A_RUNS[FT3]),
                    help=f"FT3's records over dev set A (default {DEV_A_RUNS[FT3]})")
    ap.add_argument("--dev-a-ft5", type=Path, default=None,
                    help=f"FT5's records over dev set A (default {DEV_A_RUNS[FT5].format(step='<step>')})")
    ap.add_argument("--out", type=Path, default=OUT,
                    help="the check, written once (default results/revision-check-m8.json in this "
                         "repository)")
    return ap


posix = decide_m8.posix


def refuse(path: str | Path, name: str) -> None:
    """RefusedTestSet when an input is test material or a held-out set (module docstring: which files),
    before it is read."""
    try:
        refuse_test_sets(path, (Path.cwd(), REPO))
    except RefusedTestSet as exc:
        raise RefusedTestSet(f"{name}: {exc}") from None
    how = runner.held_out(Path(path))
    if how is not None:
        raise RefusedTestSet(f"{name}: {posix(path)} is a held-out set (found by its {how}); it is never "
                             "read here")


def dev_a_files(records: Path) -> dict[str, Path]:
    """A dev set A run's files by their names in the inputs: the records and, beside them, the run manifest
    and the summary, named as scripts/run_special_dev.py names them."""
    return {"records": records, "manifest": runner.manifest_path(records),
            "summary": runner.summary_path(records)}


def dev_manifest_path(records: str | Path) -> Path:
    """A dev run's manifest, beside its records (decide_m8.run_manifest's path)."""
    return Path(records).with_name(Path(records).stem + ".manifest.json")


def current_dev() -> str:
    """data/eval/dev.jsonl's SHA-256 now, its lock's (module docstring: the dev runs): Disagreement when it is
    not there, RefusedTestSet for a locked test set or a file of the handwritten test items, LockError without
    its lock or when it does not match it."""
    if not Path(DEV).is_file():
        raise Disagreement(f"no {DEV} to score the dev runs with")
    refuse_test_sets(DEV, (Path.cwd(), REPO))
    return check_lock(DEV)


def dev_in_item_sets(selection: dict, path: Path, current: str) -> None:
    """Disagreement unless the selection's item_sets hold data/eval/dev.jsonl, once, at its current
    SHA-256."""
    sets = selection.get("item_sets")
    sets = [entry for entry in sets if isinstance(entry, dict)] if isinstance(sets, list) else []
    entries = [entry for entry in sets if entry.get("path") == DEV]
    if len(entries) != 1:
        raise Disagreement(f"{posix(path)}: item_sets must hold {DEV} once (the set of the dev runs), not "
                           f"{[entry.get('path') for entry in sets]!r}")
    if entries[0].get("sha256") != current:
        raise Disagreement(f"{posix(path)}: {DEV} changed since its runs (item_sets: sha256 "
                           f"{entries[0].get('sha256')!r}; the file's now {current})")


def dev_row(selection: dict, path: Path, step: int, option: str) -> dict:
    """The selection's row of step (option names where the step was asked for): {"step", "gguf", "sha256",
    "records"}, the records of s0 over data/eval/dev.jsonl. Malformed for rows not of their shape;
    Disagreement when no row is of step."""
    rows = selection.get("checkpoints")
    if not isinstance(rows, list) or not rows:
        raise Malformed(f"{posix(path)} holds no checkpoints")
    found, steps = None, []
    for row in rows:
        try:
            number, gguf, digest = row["step"], row["gguf"], row["sha256"]
        except (KeyError, TypeError) as exc:
            raise Malformed(f"{posix(path)}: a checkpoint row without its step, GGUF or SHA-256 "
                            f"({type(exc).__name__}: {exc})") from None
        if type(number) is not int or not isinstance(gguf, str) or not isinstance(digest, str):
            raise Malformed(f"{posix(path)}: a checkpoint row whose step, GGUF or SHA-256 is malformed")
        if number in steps:
            raise Malformed(f"{posix(path)} lists step {number} twice")
        steps.append(number)
        found = row if number == step else found
    if found is None:
        raise Disagreement(f"{posix(path)} holds no step {step} ({option}; its steps: "
                           f"{', '.join(map(str, steps))})")
    try:
        records = found["variants"][SEED]["sets"][DEV]["records"]
    except (KeyError, TypeError) as exc:
        raise Malformed(f"{posix(path)}: step {step} holds no records of run {SEED} over {DEV} "
                        f"(variants.{SEED}.sets; {type(exc).__name__}: {exc})") from None
    if not isinstance(records, str) or not records:
        raise Malformed(f"{posix(path)}: step {step}'s records of run {SEED} over {DEV} are {records!r}")
    return {"step": step, "gguf": found["gguf"], "sha256": found["sha256"], "records": records}


def same_place(path: str | Path, other: str | Path) -> bool:
    """Whether two paths name one place, both resolved from the current folder (os.path.realpath, case
    folded where the file system folds it)."""
    return os.path.normcase(os.path.realpath(path)) == os.path.normcase(os.path.realpath(other))


def last_checkpoint(args: argparse.Namespace, selection: dict, baseline: dict, rows: dict[str, dict]) -> Path:
    """FT5's training result (<run>/result.json), once the selection's row of --step is shown to be FT5's
    last checkpoint as FT5 is now (module docstring: the checkpoint): Disagreement otherwise, Malformed for a
    training result that names no steps."""
    run, theirs = selection.get("run"), baseline.get("run")
    if not isinstance(run, str) or not same_place(run, args.run):
        raise Disagreement(f"{posix(args.selection)} is a selection of the run {run!r}, where FT5's "
                           f"{posix(args.run)} (--run)")
    if isinstance(theirs, str) and same_place(theirs, args.run):
        raise Disagreement(f"--run {posix(args.run)} is the run of the baseline {posix(args.baseline)}: "
                           "FT5's last checkpoint is checked against FT3's, never against FT3's own run")
    result = args.run / "result.json"
    refuse(result, "run_result")
    trained = decide_m8.read_json(result, "FT5's training result").get("steps")
    if type(trained) is not int:
        raise Malformed(f"{posix(result)} names no steps trained (\"steps\": {trained!r})")
    if trained != args.step:
        raise Disagreement(f"--step {args.step} is not FT5's last checkpoint: {posix(result)} says {trained} "
                           "steps were trained, and the trainer saves a checkpoint at the last step")
    ft5, ft3 = rows[FT5], rows[FT3]
    if ft5["sha256"] == ft3["sha256"]:
        raise Disagreement(f"{posix(args.selection)}: step {args.step}'s GGUF {ft5['gguf']} has the SHA-256 "
                           f"of FT3 step {args.baseline_step}'s ({ft3['sha256']}): it is not FT5's")
    refuse(ft5["gguf"], "ft5_gguf")
    if not Path(ft5["gguf"]).is_file() or sha256_of(ft5["gguf"]) != ft5["sha256"]:
        raise Disagreement(f"FT5's checkpoint GGUF {ft5['gguf']} is not there with the SHA-256 "
                           f"{posix(args.selection)} records for step {args.step} ({ft5['sha256']}): FT5 was "
                           "trained or exported again since the selection's runs")
    return result


def open_dev() -> list[dict]:
    """data/eval/dev.jsonl opened as decide_m8 opens the development sets (decide_m8.open_sets; module
    docstring: the dev runs)."""
    roots = (Path.cwd(), REPO)
    refuse_test_sets(DEV, roots)
    items = DevSet(DEV).open() if lock_required(DEV, roots) else read_jsonl(DEV)
    vote_offline.refuse_items(items, Path(DEV), aggregate_only=False)
    vote_offline.check_ids(items, aggregate_only=False)
    return items


def dev_counts(rows: dict[str, dict], items: list[dict], current: str) -> dict[str, int]:
    """Each dev run's successes with the rules applied offline (decide_m8.successes), once both manifests are
    shown to fit their checkpoint, the set and FT3's run (module docstring: the dev runs)."""
    manifests = {label: decide_m8.run_manifest(row["records"]) for label, row in rows.items()}
    reference = manifests[FT3][1]
    for label, row in rows.items():
        path, manifest = manifests[label]
        problems = decide_m8.manifest_problems(manifest, sha256=row["sha256"], set_path=DEV,
                                               set_sha256=current, n=len(items), reference=reference)
        if problems:
            raise Disagreement(f"{posix(path)} ({label}'s run of checkpoint {row['step']}, {SEED} over "
                               f"{DEV}) is not the run the check needs: " + "; ".join(problems))
    return {label: decide_m8.successes(row["records"], items)[0] for label, row in rows.items()}


def dev_set_a() -> tuple[list[dict], str]:
    """Dev set A's items, opened as scripts/run_special_dev.py opens them (module docstring: the dev set A
    runs), and its SHA-256, its lock's: Disagreement when the file is not there, LockError without its lock
    or when it does not match it (RefusedTestSet for test material or a held-out set), Malformed for malformed
    items."""
    if not Path(DEV_SET_A).is_file():
        raise Disagreement(f"no {DEV_SET_A}: the dev set A runs are checked against the set and its lock, "
                           "and scored again against it")
    try:
        items, locked = runner.load_items(Path(DEV_SET_A))
    except runner.MalformedItems as exc:
        raise Malformed(f"{DEV_SET_A}: malformed items: {exc}") from None
    if not locked:
        raise LockError(f"{Path(DEV_SET_A).name} has no lock: the dev set A runs are checked against the set "
                        "as it was locked")
    return items, check_lock(DEV_SET_A)


def dev_a_run(label: str, files: dict[str, Path]) -> dict:
    """A dev set A run (module docstring: the dev set A runs): the counts, scoring errors and set its summary
    records, the records' SHA-256 and the run manifest. Malformed for a summary without its counts;
    Disagreement for a summary of other records, or records, a summary or a manifest that cannot be read."""
    records, summary_path = files["records"], files["summary"]
    value = decide_m8.read_json(summary_path, f"the dev set A summary of {label}")
    summary = value.get("summary")
    counts = decide_m8.summary_counts(summary, f"the dev set A summary of {label} ({posix(summary_path)}), "
                                               "under \"summary\",")
    if value.get("records") != records.name:
        raise Disagreement(f"the dev set A summary of {label} ({posix(summary_path)}) names the records "
                           f"{value.get('records')!r}, where {records.name}, beside it "
                           "(scripts/run_special_dev.py writes a run's summary beside its records)")
    try:
        records_sha256 = sha256_of(records)
    except OSError as exc:
        raise Disagreement(f"the dev set A records of {label} {posix(records)} cannot be read: "
                           f"{exc.strerror or exc}") from None
    return {"records": records, "records_sha256": records_sha256, "summary_path": summary_path,
            "counts": counts, "summary_errors": summary.get("scoring_errors"), "items": value.get("items"),
            "items_sha256": value.get("items_sha256"), "manifest_path": files["manifest"],
            "manifest": decide_m8.read_json(files["manifest"], f"the dev set A run manifest of {label}")}


def is_dev_set_a(path: object) -> bool:
    """Whether a recorded items path is dev set A: DEV_SET_A as written, or the same file."""
    return isinstance(path, str) and (posix(path) == DEV_SET_A
                                      or named_set(path, [DEV_SET_A], (Path.cwd(), REPO)) is not None)


def run_problems(run: dict, *, adapter: str, lock: str) -> list[str]:
    """Why a dev set A run is not the one its label needs, a line per field: decide_m8.run_problems for a run
    of an adapter, over dev set A (module docstring: the dev set A runs); adapter is the SHA-256 of the GGUF
    it must have applied, lock dev set A's SHA-256."""
    manifest, problems = run["manifest"], []
    scale, named = manifest.get("lora_scale"), decide_m8.adapter_digests(manifest)
    if named != [adapter]:
        problems.append(f"adapters {named!r}, where the GGUF {adapter} alone")
    if not (decide_m8._number(scale) and scale == decide_m8.FT_SCALE):
        problems.append(f"lora_scale {scale!r}, where {decide_m8.FT_SCALE} (the adapter applied)")
    for where, items, digest in (("summary", run["items"], run["items_sha256"]),
                                 ("manifest", manifest.get("items"), manifest.get("items_sha256"))):
        if not is_dev_set_a(items):
            problems.append(f"{where} items {items!r}, where {DEV_SET_A}")
        if digest != lock:
            problems.append(f"{where} items_sha256 {digest!r}, where dev set A's lock {lock}")
    done, total = manifest.get("items_run"), manifest.get("items_total")
    if type(done) is not int or done != total:
        problems.append(f"items_run {done!r} of items_total {total!r}, where every item")
    if manifest.get("statuses") != {"ok": total}:
        problems.append(f"statuses {manifest.get('statuses')!r}, where every item answered ok ({{'ok': "
                        f"{total!r}}}: no error, no context overflow)")
    if run["summary_errors"] not in (None, 0):
        problems.append(f"summary scoring_errors {run['summary_errors']!r}, where none (every item scored)")
    slots = decide_m8.field(manifest, "server_props.total_slots")
    if slots != decide_m8.DEV_B_SLOTS:
        problems.append(f"server_props.total_slots {slots!r}, where {decide_m8.DEV_B_SLOTS}")
    generation = manifest.get("generation") if isinstance(manifest.get("generation"), dict) else {}
    if generation.get("mode") != decide_m8.THINK:
        problems.append(f"generation.mode {generation.get('mode')!r}, where {decide_m8.THINK!r}")
    if generation.get("decoding") != decide_m8.GREEDY_DECODING:
        problems.append(f"generation.decoding {generation.get('decoding')!r}, where greedy "
                        f"{decide_m8.GREEDY_DECODING!r}")
    if generation.get("reasoning_budget") != decide_m8.REASONING_BUDGET:
        problems.append(f"generation.reasoning_budget {generation.get('reasoning_budget')!r}, where "
                        f"{decide_m8.REASONING_BUDGET}")
    return problems


def check_dev_a_runs(runs: dict[str, dict], adapters: dict[str, str], lock: str) -> None:
    """Disagreement unless each dev set A run is the one its label needs, and the two have one generation and
    were made on one base model, build and family (module docstring: the dev set A runs)."""
    for label, run in runs.items():
        problems = run_problems(run, adapter=adapters[label], lock=lock)
        if problems:
            raise Disagreement(f"the dev set A run of {label} (records {posix(run['records'])}, manifest "
                               f"{posix(run['manifest_path'])}) is not the run {label} needs: "
                               + "; ".join(problems))
    generations = {label: run["manifest"].get("generation") for label, run in runs.items()}
    if generations[FT5] != generations[FT3]:
        raise Disagreement(f"the dev set A runs were not generated alike (generation): {FT3} "
                           f"{generations[FT3]!r}; {FT5} {generations[FT5]!r}")
    for name in decide_m8.DEV_B_SAME:
        values = {label: decide_m8.field(run["manifest"], name) for label, run in runs.items()}
        if values[FT3] is None or values[FT5] != values[FT3]:
            raise Disagreement(f"the dev set A runs were not made on one base model, build and family "
                               f"({name}, recorded in both run manifests and the same): "
                               f"{FT3} {values[FT3]!r}; {FT5} {values[FT5]!r}")


def rescore(label: str, run: dict, items: list[dict]) -> dict:
    """A dev set A run's records scored again against dev set A by the scorer as it is now (module docstring:
    the dev set A runs): the counts of decide_m8.COUNTS. Malformed for records that are not one of every item;
    Disagreement for a record that is not ok, an item the scorer fails on, or a summary that counted
    otherwise."""
    where = f"the dev set A records of {label} {posix(run['records'])}"
    try:
        records = read_jsonl(run["records"])
    except OSError as exc:
        raise Disagreement(f"{where} cannot be read: {exc.strerror or exc}") from None
    except JsonlError as exc:
        raise Malformed(f"{where}: {exc}") from None
    try:
        summary = special_dev.summarize(special_dev.score_records(records, items), items)
    except ValueError as exc:
        raise Malformed(f"{where} are not one record of every item of {DEV_SET_A}: {exc}") from None
    problems = []
    statuses = dict(sorted(Counter(str(record.get("status")) for record in records).items()))
    if statuses != {"ok": len(items)}:
        problems.append(f"record statuses {statuses}, where every one ok")
    if summary["scoring_errors"]:
        problems.append(f"scoring_errors {summary['scoring_errors']}: items the scorer fails on, where none")
    counts = decide_m8.summary_counts(summary, f"{where} scored again")
    differ = [f"{name} {run['counts'][key]} in its summary, where its records scored again give {counts[key]}"
              for key, name in decide_m8.COUNTS.items() if run["counts"][key] != counts[key]]
    if differ:
        problems.append("; ".join(differ) + " (the summary is of other records, or another version of the "
                        "scorer wrote it: write it again from these records)")
    if problems:
        raise Disagreement(f"the dev set A run of {label} (summary {posix(run['summary_path'])}, records "
                           f"{posix(run['records'])}) is not what the scorer counts now "
                           f"({special_dev.__name__}): " + "; ".join(problems))
    return counts


def numbers(row: dict, dev_s0: int, counts: dict) -> dict:
    """A system's checkpoint and numbers as the check records them: its GGUF and SHA-256 (its selection
    file's row), its dev successes and its dev set A counts."""
    return {"gguf": row["gguf"], "gguf_sha256": row["sha256"], "dev_s0": dev_s0,
            "special_n": counts["special_n"], "special_strict": counts["strict"],
            "valid_n": counts["valid_n"], "false_refusals": counts["false_refusals"]}


def fraction_text(value: Fraction) -> str:
    """A fraction in lowest terms as numerator/denominator, the denominator always written ("0/1")."""
    return f"{value.numerator}/{value.denominator}"


def triggers_of(ft3: dict, ft5: dict) -> dict:
    """R1 to R3 of the two systems' numbers (module docstring: the triggers), each compared exactly."""
    strict = Fraction(ft5["special_strict"], ft5["special_n"])
    extra = ft5["false_refusals"] - ft3["false_refusals"]
    drop = ft3["dev_s0"] - ft5["dev_s0"]
    return {"R1": {"value": fraction_text(strict), "value_float": float(strict),
                   "threshold": fraction_text(R1_STRICT), "threshold_float": float(R1_STRICT),
                   "fired": strict < R1_STRICT},
            "R2": {"value": extra, "threshold": R2_EXTRA, "fired": extra >= R2_EXTRA},
            "R3": {"value": drop, "threshold": R3_DROP, "fired": drop >= R3_DROP}}


def read_inputs(args: argparse.Namespace) -> dict:
    """Everything the check reads, checked (module docstring); Malformed, Disagreement or the lock's errors
    otherwise. Dev set A and its runs are read before the dev runs are scored, which takes longer."""
    files = {"selection": args.selection, "baseline": args.baseline}
    dev_a = {FT3: dev_a_files(args.dev_a_ft3), FT5: dev_a_files(args.dev_a_ft5)}
    for name, path in [*files.items(), *((f"dev_a_{label}_{kind}", path) for label, named in dev_a.items()
                                         for kind, path in named.items())]:
        refuse(path, name)
    selection = decide_m8.read_json(args.selection, "the selection")
    baseline = decide_m8.read_json(args.baseline, "the baseline")
    current = current_dev()
    for data, path in ((selection, args.selection), (baseline, args.baseline)):
        dev_in_item_sets(data, path, current)
    rows = {FT3: dev_row(baseline, args.baseline, args.baseline_step, "--baseline-step"),
            FT5: dev_row(selection, args.selection, args.step, "--step")}
    files["run_result"] = last_checkpoint(args, selection, baseline, rows)
    for label, row in rows.items():
        files[f"dev_{label}_records"] = Path(row["records"])
        files[f"dev_{label}_manifest"] = dev_manifest_path(row["records"])
        for kind in ("records", "manifest"):
            refuse(files[f"dev_{label}_{kind}"], f"dev_{label}_{kind}")
    items_a, lock = dev_set_a()
    runs = {label: dev_a_run(label, named) for label, named in dev_a.items()}
    check_dev_a_runs(runs, {label: row["sha256"] for label, row in rows.items()}, lock)
    counts = {label: rescore(label, run, items_a) for label, run in runs.items()}
    dev_s0 = dev_counts(rows, open_dev(), current)
    files |= {f"dev_a_{label}_{kind}": path for label, named in dev_a.items() for kind, path in named.items()}
    return {"files": files, "lock": lock,
            "numbers": {label: numbers(rows[label], dev_s0[label], counts[label]) for label in (FT3, FT5)}}


def report(args: argparse.Namespace, inputs: dict, triggers: dict) -> list[str]:
    """The lines printed: every trigger with its numbers, then whether any fired."""
    ft3, ft5 = inputs["numbers"][FT3], inputs["numbers"][FT5]
    r1, r2, r3 = (triggers[name] for name in TRIGGERS)

    def verdict(trigger: dict) -> str:
        return "FIRED" if trigger["fired"] else "not fired"

    fired = [name for name in TRIGGERS if triggers[name]["fired"]]
    return [
        (f"FT5 step {args.step} against FT3 step {args.baseline_step}; dev set A {DEV_SET_A} (sha256 "
         f"{inputs['lock']}), the records scored again by {special_dev.__name__}"),
        (f"R1 FT5's strict special rate on dev set A: {ft5['special_strict']}/{ft5['special_n']} = "
         f"{r1['value_float']:.4f} < {r1['threshold']} = {r1['threshold_float']:.4f}: {verdict(r1)}"),
        (f"R2 false refusals on dev set A's valid items: FT5 {ft5['false_refusals']} - FT3 "
         f"{ft3['false_refusals']} = {r2['value']} >= {r2['threshold']}: {verdict(r2)}"),
        (f"R3 successes on {DEV}, seed 0 (the runtime rules applied offline, {decide_m8.SCORER} scorer): "
         f"FT3 {ft3['dev_s0']} - FT5 {ft5['dev_s0']} = {r3['value']} >= {r3['threshold']}: {verdict(r3)}"),
        (f"triggered: yes ({', '.join(fired)}): the data are revised once and FT5 is trained again "
         "(docs/m8-计划.md WP3)" if fired else "triggered: no: FT5's data stand"),
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    if args.dev_a_ft5 is None:
        args.dev_a_ft5 = Path(DEV_A_RUNS[FT5].format(step=args.step))
    if args.out.exists():
        print(f"FAIL: {posix(args.out)} exists: the data-revision check is made once (docs/m8-计划.md WP3), "
              "and nothing was read")
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
    triggers = triggers_of(inputs["numbers"][FT3], inputs["numbers"][FT5])
    check = {"about": ABOUT, "rules": RULES, "step": args.step, "baseline_step": args.baseline_step,
             "inputs": {name: {"path": posix(path), "sha256": sha256_of(path)}
                        for name, path in inputs["files"].items()},
             "dev_set_a": {"path": DEV_SET_A, "sha256": inputs["lock"]},
             FT3: inputs["numbers"][FT3], FT5: inputs["numbers"][FT5], "triggers": triggers,
             "triggered": any(trigger["fired"] for trigger in triggers.values())}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.out.open("x", encoding="utf-8", newline="\n") as f:  # never over a check written meanwhile
            f.write(json.dumps(check, ensure_ascii=False, indent=2) + "\n")
    except FileExistsError:
        print(f"FAIL: {posix(args.out)} was written while the check ran: the check is made once")
        return 2
    for line in report(args, inputs, triggers):
        print(line)
    print(f"wrote {posix(args.out)}")
    return 3 if check["triggered"] else 0


if __name__ == "__main__":
    sys.exit(main())

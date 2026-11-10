"""Choose the checkpoint of a training run by its single-step success on the development set (docs: the M3
plan, default 1, fixed before the test set is opened): the highest rate wins, a tie goes to the later step.

    .venv/Scripts/python.exe scripts/select_checkpoint.py runs/train/<run> [--system ft|b4] \\
        [--items data/eval/dev.jsonl ...] [--out results/checkpoint-selection-<run>.json] \\
        [--records runs/eval/select-<run>] [--llama-server PATH] [--model models/Qwen3-1.7B-Q8_0.gguf] \\
        [--port 8080] [--timeout 120] [--tokenizer models/Qwen3-1.7B] [--dry-run] \\
        [--checkpoints N[,M...] [--llama-cpp-src PATH]] [--variant NAME="ARGS" ...] \\
        [--select-by mean:NAME,...] [--scorer default|normalized]

Checkpoints. The candidates are the run's exported GGUFs, <run>/gguf/checkpoint-<N>-f16.gguf, in the order of
N. The run and its export must be complete, so that the rule is never applied to a partial set:
<run>/result.json gives the steps trained, which must be the largest N (transformers saves a checkpoint at
the last step), and every checkpoint the trainer kept (<run>/trainer/checkpoint-<N> with its
trainer_state.json; scripts/train.py keeps all of them) must have its GGUF. <run>/gguf/adapter-f16.gguf, the
final adapter, holds the weights of the last checkpoint (its GGUF metadata differs, so its bytes do too): it
is reported with its SHA-256 as an alias of that step, and neither served nor counted.

--checkpoints N[,M...] (M7, docs/m7-计划.md WP-E and WP-J: a run still training, or a few of its checkpoints)
applies the rule to the listed steps alone and needs no result.json: each listed step must have its trainer
checkpoint (<run>/trainer/checkpoint-<N>/trainer_state.json), and a step without its GGUF is exported first
with scripts/export_adapter.py (--llama-cpp-src or $LLAMA_CPP_SRC names the converter's tree, as there;
the base model comes from the run's manifest). The selection records "partial": true and the listed steps
("checkpoints_listed"), reports no final adapter (it is not among the listed steps), and its score is as
biased as a full selection's. A listed step without a trainer checkpoint is an incomplete run (exit 1); a
step that is not a whole number, or listed twice, is a usage error.

Each checkpoint in turn, one server at a time: llama-server (research_agent.llm.server.LocalServer: the
command of scripts/serve_local.py with --lora <gguf> and one slot (-np, --slots, default 1: with several
slots the greedy output depends on the request history, the M3 finding; /props must report that number),
the port free before it starts, ready on its own log line and /health, its output in
<records>/checkpoint-<N>.server.log) serves the base model with that GGUF, and the adapter it lists must be
exactly that file; scripts/run_eval.py runs the items with --system (ft,
or b4 for the B4 run: lora scale 1.0) into <records>/checkpoint-<N>.jsonl, and its manifest must name the
same adapter and SHA-256; the server is stopped; scripts/score.py scores the records into
<records>/checkpoint-<N>.score.json. Records with an error the model did not cause (an error_type other than
HarnessError, e.g. a lost connection) stop the selection, since that rate would not measure the checkpoint.
The items are opened through DevSet (their lock must match), and a locked test set, or a file of the test
set's handwritten items, is refused before anything is read, as in run_eval; so, once read, are items that
hold test material (split "test" or an id of the test split, eval.lock.refuse_test_material).

Variants (M7). --variant NAME="ARGS", repeatable, runs run_eval.py once more per checkpoint with ARGS
appended (split as a shell would: "--mode think --seed 0", "--greedy"), into <records>/checkpoint-<N>.NAME
.jsonl with its manifest, timing sidecar and score file beside it, on the same server. NAME is a file-name
piece ([A-Za-z0-9_-]), given once; ARGS may not set what the selection sets itself (--out, --system,
--server, --tokenizer). Each checkpoint's row then carries "variants": NAME -> {n, successes, rate,
statuses, records}, beside the default greedy run, which still selects. --select-by mean:NAME,... instead
selects by the mean of the named variants' successes (the bake-off's three thinking seeds, docs/m7-计划.md
WP-J: a thinking family's checkpoints are chosen on the mean of 2 seeds): the default run is not made, the
row's successes is that mean (a whole number or a decimal), its rate the mean over n, compared exactly, and
the selection file records "selection": {"metric": "mean", "variants": [...]} with the rule text that says
so; the named variants must be given, once each. Without the flags the run, its records and the selection
file are byte for byte those of today.

Several sets (M7, docs/m7-计划.md WP-J 选点 and §5 开发集口径: FT3's checkpoints are chosen on dev and
dev-v2 together). --items may be given more than once, the sets taken in the order given (data/eval/dev.jsonl
when none is). Each set is checked as the one set is above: all of them are refused as test sets or files of
handwritten test items before any is read, then each is opened through DevSet and refused when it holds test
material; then no item id may be in two sets (a usage error). A checkpoint's server serves every set: each of
its runs (the default greedy run, or every variant) runs run_eval.py and score.py once per set, in the order
of the sets. With one set the files are those above. With several, a set's files go to the folder of
<records> named by the set's file stem, <records>/<stem>/checkpoint-<N>.jsonl (checkpoint-<N>.NAME.jsonl for a
variant) with the manifest, timing sidecar and score file beside it, while the server log stays
<records>/checkpoint-<N>.server.log; the stems must be file-name pieces ([A-Za-z0-9_-]) that differ also when
case is ignored (a usage error otherwise, before any set is read), so no two files of a selection share a
path. A run over several sets has n the sets' total of items, successes and statuses their sums and rate
successes / n, and holds each set's summary {n, successes, rate, statuses, records} under "sets", by the
set's path in the order of the sets, in place of the one set's "records": so the row has "sets" where it has
"records" with one set, and so has each variant beside it. Under --select-by the row's successes is the mean
over the named variants of each one's summed successes. The rule, the tie to the later step and the exact
comparison are unchanged; the rule text gains a sentence saying that the sets were taken together
(SETS_RULE), and the selection file records the sets in their order as "item_sets": [{path, sha256}] in
place of "items" and "items_sha256". --dry-run lists each set with its SHA-256 and number of items, and the
commands of every set. With one set, given or the default, the run, its files, its output and the selection
file are byte for byte those of today.

Scorer (M7, docs/m7-本地提升计划.md WP1 and WP7). --scorer normalized has score.py score every run with the
normalized scorer (research_agent.eval.metrics, Scorers: S1 to S3), into <stem>.score-normalized.json beside
the records instead of <stem>.score.json, a name no other file of a selection has whatever its variants are
called, and the selection counts those successes, every rule above unchanged. The selection file records
"scorer": "normalized" after the rule, whose text gains a sentence saying so (SCORER_RULE), and goes by
default to results/checkpoint-selection-<run name>-normalized.json, so a default selection of the same run
and its score files stay as they are (research_agent.eval.report then gives its numbers as
checkpoint.<run name>-normalized.*). The dry run lists the scorer. Without --scorer, or with --scorer default,
the run, its files, its output and the selection file are byte for byte those of today, but for one refusal:
a default selection makes every run anew, writing over the records there, so a records folder that holds a
score file another scorer made of the records of a run it would make (<stem>.score-<scorer>.json; those
scores would describe records no longer there) is refused before anything starts (exit 2, the dry run too),
each such file named: after a selection under another scorer, a default selection needs its own --records.

Under a scorer other than the default a selection writes over no file it finds. A run whose records
<stem>.jsonl are there already (a default selection's in the same --records folder, or a selection's that
stopped) does not run again: its records are scored again, provided their manifest records what the run
asks for: the system and its lora scale, exactly the checkpoint's GGUF and its SHA-256, the set's SHA-256
with every item run, the generation settings run_eval derives from the run's arguments and the tokenizer
folder's family (requested_generation: mode, decoding, reasoning budget, answer tokens, base seed and
sample), the prompt version, prompt hash and schema hash run_eval writes now (read from scripts/run_eval.py
itself), that family's whole record (family.record(): its name, template and tokenizer.json hashes, BOS,
thinking, LoRA targets...), the base model (server_props.model_path, the model as the server was given it,
must be --model's file, compared as the adapters' paths are, from the current folder with case and
separators aside where the file system ignores them; a manifest that records the base model's SHA-256 as
base_model_sha256, which run_eval does not, must record --model's, hashed only then, once), the
selection's slots (--slots), and the status counts of the records beside it; and the records of one step
come from one llama-server build. A run whose arguments set any run_eval option but --mode, --seed,
--reasoning-budget, --greedy, --signals and --limit (REUSE_CHECKED; the inference wrappers' --votes, --rules
and --glossary among them) cannot be checked so. Otherwise nothing starts (exit 2, the dry run too), and each
manifest is named with what it records against what the run asks for (of the family record, the keys that
differ). A checkpoint every run of which finds its records starts no server, and its row's server build is
the one their manifests record; a checkpoint with runs still to make starts its server for those alone, which
must be of the build of the records found (exit 2 otherwise), its log going to the first of
checkpoint-<N>.server.log, checkpoint-<N>.server-2.log ... that is not there. The selection file lists the
records scored again under "records_reused", after "scorer". The dry run shows such a run as reuse[...] with
its records in place of its run_eval command, and a checkpoint that needs no server as "server:   none". An
--out that holds a selection another scorer made (a file without "scorer" is the default scorer's) is refused
before anything starts (exit 2), so that it never replaces the default selection's file.

The family (M7). When the run's manifest.json records the family it trained (config.family.name,
scripts/train.py), the tokenizer folder's family (research_agent.family.load_family) must be that one, or the
checkpoints would be served under another chat template than they were trained with: exit 2 naming both. A
manifest without a family record (a run before the family layer) is not checked.

The choice: the most successes / n, compared exactly; of equal rates the later step. The chosen checkpoint's
development score is optimistically biased, having been chosen on that set (the rule text says so).

Writes <out> (results/checkpoint-selection-<run name>.json by default): JSON, two-space indent, LF, with
repository-relative POSIX paths and no time, so a rerun over the same checkpoints and answers gives the same
bytes: the run, the system, the lora scale, the rule, the items and their SHA-256 (item_sets with several
sets), the base model, the server arguments, the git commit and whether the tree was dirty, per checkpoint
{step, gguf, sha256, n, successes, rate, statuses, server_build, records} ("sets" in place of "records" with
several sets), the final adapter {gguf, sha256, alias_of_step} and the chosen checkpoint. Nothing is
registered in results/numbers.json (the phase-4 report does that). The runs with their manifests and timing
sidecars, the scores and the server logs stay under <records> (runs/ is not tracked).

--dry-run makes the same checks of the run folder, the items, the base model and the binary, then lists the
checkpoints with their SHA-256 (or "to be exported") and every command a run would execute, the exports
among them; it starts nothing and writes nothing (the port is checked when a server starts). Run from the
repository root: a real run refuses paths outside the current folder, which the selection file could only
hold as machine paths.

Exit 0 when the selection is written (or, with --dry-run, listed); 1 when the run folder is incomplete or
inconsistent or the items are malformed; 2 on a usage error (no llama-server binary, no base model, a system
the run's name does not tell, a path outside the current folder, a busy port, malformed --checkpoints,
--variant or --select-by, several --items whose file stems are not distinct file-name pieces or which share
an item id, a tokenizer folder whose family is not the run's; under a scorer other than the default, records
found that cannot be scored again or a server of another build than theirs; under the default scorer, a
records folder with another scorer's score files of the records it would make, Scorer); 3 when a server did
not become ready or did not load exactly the checkpoint's GGUF; 4 when the items are a locked test set, a file
of the handwritten test items or any other test material; 5 when they do not match their lock; 6 when an
export, a run_eval or a score.py run failed or its records hold errors the model did not cause.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from research_agent.eval.lock import (
    DEV_SET,
    DevSet,
    JsonlError,
    LockError,
    RefusedTestSet,
    read_jsonl,
    refuse_test_material,
    refuse_test_sets,
    sha256_of,
)
from research_agent.eval.metrics import DEFAULT_SCORER, SCORERS
from research_agent.eval.systems import LOCAL_SCALES
from research_agent.family.core import GREEDY, FamilyError, LoadedFamily, Mode
from research_agent.family.core import load_family as load_family_from_folder
from research_agent.llm.generation import GenerationSettings
from research_agent.llm.local import LocalClient
from research_agent.llm.server import (
    HOST,
    MODEL,
    SERVER_VAR,
    LocalServer,
    NotReady,
    PortBusy,
    llama_server,
    same_path,
    server_command,
)
from research_agent.train.manifest import REPO, git_commit, git_dirty

SCRIPTS = Path(__file__).resolve().parent
TOKENIZER = "models/Qwen3-1.7B"
ADAPTER_SYSTEMS = ("ft", "b4")  # both run at LOCAL_SCALES' 1.0
CHECKPOINT = re.compile(r"checkpoint-(\d+)-f16\.gguf")
TRAINER_CHECKPOINT = re.compile(r"checkpoint-(\d+)")
FINAL_ADAPTER = "adapter-f16.gguf"
SLOTS = 1  # llama-server slots: one, since with several the greedy output depends on the request history
MODEL_ERRORS = ("HarnessError",)  # the error records a local model's own output causes (agent.models)
RECORD_STATUSES = ("ok", "context_overflow", "error")  # a local run has no budget stop
RULE = ("The checkpoint with the highest single-step success rate on the development set (successes / n, "
        "compared exactly); a tie goes to the later step (the M3 plan, default 1). The chosen checkpoint's "
        "development score is optimistically biased: it was chosen on that set.")
VARIANT_NAME = re.compile(r"[A-Za-z0-9_-]+")  # a file-name piece: a variant's name, a set's stem (several)
RESERVED = ("--out", "--system", "--server", "--tokenizer")  # run_eval options the selection sets itself
METRICS = ("mean",)  # --select-by metrics
DEFAULT_RUN = ""  # the name of the default greedy run among a checkpoint's runs
SETS_RULE = ("With several sets (item_sets), the development set is their union: a run's successes and n are "
             "summed over the sets.")
SCORER_RULE = ("The successes are counted under the {scorer} scorer (scripts/score.py --scorer {scorer}; "
               "research_agent.eval.metrics, Scorers).")  # a scorer other than the default (--scorer)
# the run_eval arguments (parser dests) whose effect the check of the records found covers (module docstring:
# Scorer): the selection's own, and a variant's generation (the manifest's "generation"), --signals, which
# changes no answer, and --limit (items_run); a run that sets any other is not reused
REUSE_CHECKED = frozenset({"items", "system", "out", "server", "tokenizer", "mode", "seed",
                           "reasoning_budget", "greedy", "signals", "limit"})


class RunError(ValueError):
    """A training run folder that is incomplete or inconsistent (module docstring)."""


class UsageError(ValueError):
    """A malformed --checkpoints, --variant, --select-by or --items (module docstring)."""


@dataclass(frozen=True)
class Checkpoint:
    step: int
    gguf: Path
    sha256: str | None  # None until the GGUF is exported (--checkpoints)
    adapter: Path | None = None  # the trainer checkpoint to export when sha256 is None


@dataclass(frozen=True)
class ItemSet:
    """One --items set, opened and checked (module docstring: Several sets): its path as given, the SHA-256
    its lock check found and its number of items."""

    path: Path
    sha256: str
    n: int


@dataclass(frozen=True)
class Variant:
    """One extra run_eval run per checkpoint (module docstring: Variants)."""

    name: str
    args: tuple[str, ...]


@dataclass(frozen=True)
class Selection:
    """What --select-by selects by: the metric over the named variants."""

    metric: str
    variants: tuple[str, ...]

    def rule(self) -> str:
        names = ", ".join(self.variants)
        return (f"The checkpoint with the highest mean single-step success on the development set over the "
                f"runs {names} (the mean of their successes / n, compared exactly); a tie goes to the later "
                "step (the M3 plan, default 1, over the mean of the runs). The chosen checkpoint's "
                "development score is optimistically biased: it was chosen on that set.")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Choose a training run's checkpoint by its single-step success "
                                             "on the development set.")
    ap.add_argument("run", type=Path, help="the training run folder, e.g. runs/train/ft")
    ap.add_argument("--system", choices=ADAPTER_SYSTEMS, default=None,
                    help="the system the checkpoints run as (default: the run folder's name, ft or b4)")
    ap.add_argument("--items", type=Path, action="append", default=None,
                    help=f"the locked items (default {DEV_SET.as_posix()}); repeatable: several sets are "
                         "taken together, each set's runs in <records>/<its file stem>")
    ap.add_argument("--out", type=Path, default=None,
                    help="the selection file (default results/checkpoint-selection-<run name>.json)")
    ap.add_argument("--records", type=Path, default=None,
                    help="the folder of the runs, scores and server logs "
                         "(default runs/eval/select-<run name>)")
    ap.add_argument("--llama-server", default=None,
                    help=f"llama-server executable (default: ${SERVER_VAR}; one of the two is required)")
    ap.add_argument("--model", default=MODEL, help=f"base model GGUF (default {MODEL})")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--timeout", type=float, default=120.0, help="seconds to wait for each server")
    ap.add_argument("--slots", type=int, default=SLOTS,
                    help=f"llama-server slots (-np; default {SLOTS}: with several, the greedy output depends "
                         "on the request history); /props must report the same number")
    ap.add_argument("--tokenizer", default=TOKENIZER,
                    help=f"model folder for run_eval's chat template and context check (default {TOKENIZER})")
    ap.add_argument("--dry-run", action="store_true",
                    help="list the checkpoints and the commands; start nothing, write nothing")
    ap.add_argument("--checkpoints", default=None, metavar="N[,M...]",
                    help="score these steps alone, without result.json; a step without its GGUF is exported "
                         "first (scripts/export_adapter.py)")
    ap.add_argument("--llama-cpp-src", default=None, metavar="PATH",
                    help="llama.cpp source tree for the exports --checkpoints may need (default: "
                         "export_adapter's $LLAMA_CPP_SRC)")
    ap.add_argument("--variant", action="append", default=[], metavar='NAME="ARGS"',
                    help="one more run_eval run per checkpoint with ARGS appended, recorded as "
                         "checkpoint-<N>.NAME (repeatable)")
    ap.add_argument("--select-by", default=None, metavar="mean:NAME,...",
                    help="select by the mean of the named variants' successes instead of the default run")
    ap.add_argument("--scorer", choices=SCORERS, default=DEFAULT_SCORER,
                    help="the scorer score.py scores every run with and the selection counts (default: "
                         "default); another is recorded in the selection, whose default --out gets -<scorer>")
    return ap


def parse_steps(text: str) -> list[int]:
    """--checkpoints as distinct whole numbers in step order; UsageError otherwise."""
    steps = []
    for piece in text.split(","):
        if not piece.strip().isdigit() or int(piece) < 1:
            raise UsageError(f"--checkpoints lists {piece.strip()!r}, not a step number")
        steps.append(int(piece))
    if len(set(steps)) != len(steps):
        raise UsageError(f"--checkpoints lists a step twice: {text}")
    return sorted(steps)


def parse_variants(specs: list[str]) -> list[Variant]:
    """--variant NAME="ARGS" specifications as Variants (module docstring); UsageError otherwise."""
    variants: list[Variant] = []
    for spec in specs:
        name, sep, text = spec.partition("=")
        if not sep:
            raise UsageError(f"--variant {spec!r}: expected NAME=\"ARGS\"")
        if not VARIANT_NAME.fullmatch(name):
            raise UsageError(f"--variant {spec!r}: the name must be a file-name piece ([A-Za-z0-9_-])")
        if any(v.name == name for v in variants):
            raise UsageError(f"--variant {name} is given twice")
        args = tuple(shlex.split(text))
        taken = [a for a in args if a.split("=")[0] in RESERVED]
        if taken:
            raise UsageError(f"--variant {name}: {', '.join(taken)} is set by the selection itself")
        variants.append(Variant(name, args))
    return variants


def parse_select_by(text: str, variants: list[Variant]) -> Selection:
    """--select-by metric:NAME,... over the given variants; UsageError otherwise."""
    metric, sep, names = text.partition(":")
    if not sep or metric not in METRICS:
        raise UsageError(f"--select-by {text!r}: expected one of {', '.join(METRICS)} followed by "
                         "':NAME,...'")
    chosen = tuple(n.strip() for n in names.split(","))
    known = {v.name for v in variants}
    if len(set(chosen)) != len(chosen):
        raise UsageError(f"--select-by {text!r} names a variant twice")
    missing = [n for n in chosen if n not in known]
    if missing:
        raise UsageError(f"--select-by {text!r} names {', '.join(missing)}, given by no --variant")
    return Selection(metric, chosen)


def item_paths(given: list[Path] | None) -> list[Path]:
    """The --items sets in order, DEV_SET when none is given. With several, each set's file stem names the
    folder of its runs: UsageError when a stem is no file-name piece or two are the same, case aside (module
    docstring: Several sets). Nothing is read."""
    paths = given or [DEV_SET]
    if len(paths) > 1:
        seen: dict[str, Path] = {}
        for path in paths:
            if not VARIANT_NAME.fullmatch(path.stem):
                raise UsageError(f"--items {rel(path)}: with several sets each set's runs go to "
                                 f"<records>/<file stem>, and the file stem {path.stem!r} is no file-name "
                                 "piece ([A-Za-z0-9_-])")
            other = seen.get(path.stem.casefold())
            if other is not None:
                raise UsageError(f"--items {rel(other)} and {rel(path)} have the same file stem, case aside "
                                 f"({other.stem!r}, {path.stem!r}): with several sets each set's runs go to "
                                 "<records>/<file stem>")
            seen[path.stem.casefold()] = path
    return paths


def disjoint(sets: list[tuple[Path, list[dict]]]) -> None:
    """UsageError when an item id is in two of the sets, given as (path, items) in their order (module
    docstring: Several sets); an item without a string id is left to run_eval and score.py."""
    held: list[tuple[Path, set[str]]] = []
    for path, items in sets:
        ids = list(dict.fromkeys(i["id"] for i in items if isinstance(i.get("id"), str)))
        for earlier, theirs in held:
            shared = [i for i in ids if i in theirs]
            if shared:
                raise UsageError(f"the sets {rel(earlier)} and {rel(path)} both hold {len(shared)} item(s), "
                                 f"the first {shared[0]!r}: no item may be in two sets")
        held.append((path, set(ids)))


def trainer_steps(run: Path) -> set[int]:
    """The steps whose trainer checkpoint the run keeps (a folder with its trainer_state.json)."""
    trainer = run / "trainer"
    return {int(m.group(1)) for folder in (trainer.iterdir() if trainer.is_dir() else [])
            if (m := TRAINER_CHECKPOINT.fullmatch(folder.name)) and (folder / "trainer_state.json").is_file()}


def find_checkpoints(run: Path) -> tuple[list[Checkpoint], Path | None]:
    """The run's checkpoint GGUFs in step order with their SHA-256, and the final adapter's GGUF (None when it
    was not exported); RunError when the run or its export is incomplete (module docstring)."""
    gguf = run / "gguf"
    found = sorted((int(m.group(1)), path) for path in (gguf.iterdir() if gguf.is_dir() else [])
                   if (m := CHECKPOINT.fullmatch(path.name)))
    if not found:
        raise RunError(f"{gguf.as_posix()}: no checkpoint-<N>-f16.gguf (has the export run?)")
    steps = [step for step, _ in found]
    if len(set(steps)) != len(steps):
        raise RunError(f"{gguf.as_posix()}: more than one GGUF for a step: {steps}")
    result = run / "result.json"
    try:
        trained = json.loads(result.read_text(encoding="utf-8"))["steps"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RunError(f"{result.as_posix()}: the run has not finished, or its result names no steps "
                       f"({type(exc).__name__}: {exc})") from None
    if trained != steps[-1]:
        raise RunError(f"{result.as_posix()} says {trained} steps, but the last checkpoint GGUF is step "
                       f"{steps[-1]}")
    missing = sorted(trainer_steps(run) - set(steps))
    if missing:
        raise RunError(f"{run.as_posix()}: the trainer's checkpoints {missing} have no GGUF in "
                       f"{gguf.as_posix()} (a failed export); export them first")
    final = gguf / FINAL_ADAPTER
    checkpoints = [Checkpoint(step, path, sha256_of(path)) for step, path in found]
    return checkpoints, final if final.is_file() else None


def listed_checkpoints(run: Path, steps: list[int]) -> list[Checkpoint]:
    """The listed steps of a run, each with its GGUF's SHA-256 or, without a GGUF, the trainer checkpoint to
    export (module docstring: --checkpoints); RunError for a step without a trainer checkpoint."""
    kept = trainer_steps(run)
    missing = [step for step in steps if step not in kept]
    if missing:
        names = ", ".join(f"checkpoint-{step}" for step in missing)
        raise RunError(f"{run.as_posix()}: the trainer has no {names} (with its trainer_state.json)")
    checkpoints = []
    for step in steps:
        gguf = run / "gguf" / f"checkpoint-{step}-f16.gguf"
        if gguf.is_file():
            checkpoints.append(Checkpoint(step, gguf, sha256_of(gguf)))
        else:
            checkpoints.append(Checkpoint(step, gguf, None, run / "trainer" / f"checkpoint-{step}"))
    return checkpoints


def rel(path: str | Path) -> str:
    """path relative to the current folder in POSIX form when it lies inside it, else absolute (POSIX)."""
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return absolute.as_posix()


def inside(path: str | Path) -> bool:
    return not Path(rel(path)).is_absolute()


@dataclass(frozen=True)
class Run:
    """One run_eval run of a checkpoint over one set of items and its scoring: the default run (name
    DEFAULT_RUN) or a variant; label is the set's file stem when there are several sets, else "". reuse: under
    a scorer other than the default, the run's records are there already, so run_eval does not run and they
    are scored again (module docstring: Scorer)."""

    name: str
    evaluate: list[str]
    score: list[str]
    stem: Path  # <records>[/<set's file stem>]/checkpoint-<N> or the same with .<name> after it
    items: Path
    label: str = ""
    scorer: str = DEFAULT_SCORER
    reuse: bool = False

    @property
    def tag(self) -> str:
        """The run as the output names it: its variant, its set (several sets), both, or nothing."""
        return " on ".join(part for part in (self.name, self.label) if part)

    @property
    def scored(self) -> Path:
        """The score file score.py writes for the run: <stem>.score.json, <stem>.score-<scorer>.json for a
        scorer other than the default (module docstring: Scorer)."""
        return score_file(self.stem, self.scorer)


def score_file(stem: Path, scorer: str) -> Path:
    """<stem>.score.json, or <stem>.score-<scorer>.json for a scorer other than the default: a name no other
    file of a selection has, whatever its variants are called."""
    return Path(f"{stem}.score{'' if scorer == DEFAULT_SCORER else f'-{scorer}'}.json")


def server_log(stem: Path, scorer: str) -> Path:
    """The log of a checkpoint's server: <stem>.server.log; under a scorer other than the default, which
    writes over no file it finds (module docstring: Scorer), the first of <stem>.server.log,
    <stem>.server-2.log, <stem>.server-3.log ... that is not there."""
    log, number = Path(f"{stem}.server.log"), 2
    while scorer != DEFAULT_SCORER and log.exists():
        log, number = Path(f"{stem}.server-{number}.log"), number + 1
    return log


@dataclass(frozen=True)
class Plan:
    """What a run does for one checkpoint: the export it may need, the server it starts and its runs."""

    checkpoint: Checkpoint
    serve: list[str]
    runs: list[Run]
    stem: Path  # <records>/checkpoint-<N>: the server log's prefix, and with one set every file's
    export: list[str] | None = None

    @property
    def evaluate(self) -> list[str]:
        """The first run's run_eval command: the default run's over the first set (the plan of today)."""
        return self.runs[0].evaluate

    @property
    def score(self) -> list[str]:
        return self.runs[0].score


@dataclass(frozen=True)
class Options:
    """The M7 options of a selection, parsed (module docstring); none of them by default."""

    steps: list[int] | None = None
    variants: list[Variant] = field(default_factory=list)
    selection: Selection | None = None
    llama_cpp_src: str | None = None
    scorer: str = DEFAULT_SCORER

    @property
    def partial(self) -> bool:
        return self.steps is not None

    @property
    def names(self) -> list[str]:
        """The runs of every checkpoint in order: the default run unless --select-by, then the variants."""
        default = [] if self.selection is not None else [DEFAULT_RUN]
        return [*default, *(v.name for v in self.variants)]


NO_OPTIONS = Options()  # today's selection: no flag of M7


def slot_args(args: argparse.Namespace) -> list[str]:
    """llama-server's slot count: one by default, since with several slots the greedy output depends on the
    request history (the M3 finding); evaluate checks /props against it before any request."""
    return ["-np", str(args.slots)]


def plan(checkpoint: Checkpoint, *, binary: Path, args: argparse.Namespace, system: str,
         records: Path, sets: list[ItemSet], options: Options = NO_OPTIONS) -> Plan:
    """The checkpoint's export, server and runs: every run over every set, in the order of the runs and,
    within a run, of the sets (module docstring: Several sets)."""
    stem = records / f"checkpoint-{checkpoint.step}"
    serve = server_command(binary, model=args.model, adapter=rel(checkpoint.gguf), port=args.port,
                           extra=slot_args(args))
    variants = {v.name: v.args for v in options.variants}
    several = len(sets) > 1
    runs = []
    for name in options.names:
        for item_set in sets:
            label = item_set.path.stem if several else ""
            set_stem = (records / label if several else records) / stem.name
            run_stem = set_stem if name == DEFAULT_RUN else set_stem.with_name(f"{set_stem.name}.{name}")
            items, results = rel(item_set.path), rel(f"{run_stem}.jsonl")
            evaluate = [sys.executable, rel(SCRIPTS / "run_eval.py"), items, "--system", system, "--out",
                        results, "--server", f"http://{HOST}:{args.port}", "--tokenizer", args.tokenizer,
                        *variants.get(name, ())]
            score = [sys.executable, rel(SCRIPTS / "score.py"), results, "--items", items, "--out",
                     rel(score_file(run_stem, options.scorer))]
            reuse = False
            if options.scorer != DEFAULT_SCORER:
                score += ["--scorer", options.scorer]
                reuse = Path(f"{run_stem}.jsonl").is_file()  # module docstring: Scorer
            runs.append(Run(name, evaluate, score, run_stem, item_set.path, label, options.scorer, reuse))
    export = None
    if checkpoint.sha256 is None:
        export = [sys.executable, rel(SCRIPTS / "export_adapter.py"), rel(checkpoint.adapter), "--outfile",
                  rel(checkpoint.gguf)]
        if options.llama_cpp_src:
            export += ["--llama-cpp-src", options.llama_cpp_src]
    return Plan(checkpoint, serve, runs, stem, export)


def describe(args: argparse.Namespace, system: str, sets: list[ItemSet], plans: list[Plan],
             final: Path | None, options: Options = NO_OPTIONS) -> None:
    checkpoints, scale = [p.checkpoint for p in plans], LOCAL_SCALES[system]
    head = f"{rel(args.run)}: {len(checkpoints)} checkpoints, system {system} (lora scale {scale:g}), "
    if len(sets) == 1:
        print(f"{head}items {rel(sets[0].path)} (sha256 {sets[0].sha256})")
    else:
        print(f"{head}items: {len(sets)} sets taken together, {sum(s.n for s in sets)} items")
        for s in sets:
            print(f"  set {s.path.stem}: {rel(s.path)} (sha256 {s.sha256}), {s.n} items")
    if options.partial:
        listed = ", ".join(str(step) for step in options.steps)
        print(f"partial: {len(checkpoints)} of the run's checkpoints ({listed}), listed with --checkpoints; "
              "the rule is applied to them alone")
    for c in checkpoints:
        digest = f"sha256 {c.sha256}" if c.sha256 is not None else f"to be exported from {rel(c.adapter)}"
        print(f"  step {c.step}  {rel(c.gguf)}  {digest}")
    if options.partial:
        print("  no final adapter: a partial selection reports none")
    elif final is None:
        print(f"  no final adapter: {FINAL_ADAPTER} was not exported")
    else:
        print(f"  final adapter {rel(final)}  sha256 {sha256_of(final)}: the weights of step "
              f"{checkpoints[-1].step}, reported, neither served nor counted")
    for v in options.variants:
        print(f"variant {v.name}: run_eval with {subprocess.list2cmdline(v.args) or '(no arguments)'}")
    if options.selection is not None:
        print(f"selection: the {options.selection.metric} of {', '.join(options.selection.variants)}")
    if options.scorer != DEFAULT_SCORER:
        print(f"scorer: {options.scorer} (score.py --scorer {options.scorer}; its successes select)")
        runs = [run for p in plans for run in p.runs]
        found = sum(run.reuse for run in runs)
        if found:
            print(f"records found: {found} of the {len(runs)} runs are scored again, not run (their "
                  "manifests must record what the selection asks for)")
    for p in plans:
        print(f"step {p.checkpoint.step}:")
        if p.export is not None:
            print(f"  export:   {subprocess.list2cmdline(p.export)}")
        if all(run.reuse for run in p.runs):
            print("  server:   none: every run's records are there")
        else:
            print(f"  server:   {subprocess.list2cmdline(p.serve)}")
        for run in p.runs:
            tag = f"[{run.tag}]" if run.tag else ""
            if run.reuse:
                print(f"  reuse{tag}:    {rel(f'{run.stem}.jsonl')}")
            else:
                print(f"  run_eval{tag}: {subprocess.list2cmdline(run.evaluate)}")
            print(f"  score{tag}:    {subprocess.list2cmdline(run.score)}", flush=True)


def run_family(run: Path) -> str | None:
    """The family the run's manifest records (config.family.name), None without one."""
    manifest = run / "manifest.json"
    if not manifest.is_file():
        return None
    try:
        config = json.loads(manifest.read_text(encoding="utf-8")).get("config") or {}
    except (OSError, ValueError, AttributeError):
        return None
    family = config.get("family") if isinstance(config, dict) else None
    name = family.get("name") if isinstance(family, dict) else None
    return name if isinstance(name, str) and name else None


def score_run(c: Checkpoint, run: Run, *, system: str, do_run) -> tuple[int, dict | None]:
    """Run and score one run_eval run of a checkpoint on the running server (module docstring): (0, its
    summary {n, successes, rate, statuses, records}) or (exit code, None). A run that reuses the records found
    (module docstring: Scorer; their manifest checked by check_reused) is scored alone."""
    tag = f" ({run.tag})" if run.tag else ""
    if not run.reuse:
        done = do_run(run.evaluate)
        if done.returncode != 0:
            print(f"FAIL: step {c.step}{tag}: run_eval exited with {done.returncode}")
            return 6, None
        manifest = json.loads(Path(f"{run.stem}.manifest.json").read_text(encoding="utf-8"))
        adapters = manifest.get("adapters") or []
        if (manifest.get("system"), manifest.get("lora_scale")) != (system, LOCAL_SCALES[system]) \
                or len(adapters) != 1 or not same_path(adapters[0].get("path") or "", c.gguf) \
                or adapters[0].get("sha256") != c.sha256:
            print(f"FAIL: step {c.step}{tag}: the run's manifest records system {manifest.get('system')}, "
                  f"adapters {adapters}, not {system} over {rel(c.gguf)} (sha256 {c.sha256})")
            return 3, None
    records = read_jsonl(f"{run.stem}.jsonl")
    statuses = Counter(r.get("status") for r in records)
    foreign = Counter(r.get("error_type") for r in records
                      if r.get("status") == "error" and r.get("error_type") not in MODEL_ERRORS)
    unexpected = sorted(str(s) for s in statuses if s not in RECORD_STATUSES)
    if foreign or unexpected:
        print(f"FAIL: step {c.step}{tag}: errors the model did not cause {dict(foreign)}, statuses "
              f"{unexpected}: the rate would not measure the checkpoint (records {rel(f'{run.stem}.jsonl')})")
        return 6, None
    done = do_run(run.score)
    if done.returncode != 0:
        print(f"FAIL: step {c.step}{tag}: score.py exited with {done.returncode}")
        return 6, None
    scored = json.loads(run.scored.read_text(encoding="utf-8"))
    n, successes = scored["summary"]["n"], sum(1 for s in scored["scores"] if s["success"])
    print(f"step {c.step}{tag}: {successes}/{n} ({100 * successes / n:.1f}%)", flush=True)
    return 0, {"n": n, "successes": successes, "rate": successes / n,
               "statuses": dict(sorted(statuses.items())), "records": rel(f"{run.stem}.jsonl")}


@functools.cache
def _run_eval():
    """scripts/run_eval.py as a module of its own name (scripts/ is no package), for its argument parser:
    loaded only when a selection under a scorer other than the default finds records to check."""
    spec = importlib.util.spec_from_file_location("_run_eval_shared", SCRIPTS / "run_eval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def requested_generation(evaluate: list[str], family: LoadedFamily) -> dict:
    """The generation settings a run_eval command records in its manifest ("generation"), derived as run_eval
    derives them (scripts/run_eval.py, Generation; its own argument parser reads the command): the family's
    settings on llama-server for the mode, reasoning budget and base seed the command gives, think mode
    decoded greedily under --greedy. ValueError when run_eval would refuse the arguments, when they set
    anything the check of the records found does not cover (REUSE_CHECKED; the inference wrappers' --votes,
    --rules and --glossary among them), and when the family has not the mode."""
    parser = _run_eval().build_parser()
    try:
        args = parser.parse_args(evaluate[2:])
    except SystemExit:
        raise ValueError(f"run_eval refuses the arguments {subprocess.list2cmdline(evaluate[2:])}") from None
    unchecked = sorted(dest for dest, value in vars(args).items()
                       if dest not in REUSE_CHECKED and value != parser.get_default(dest))
    if unchecked:
        raise ValueError(f"the run's arguments set {', '.join(unchecked)}, which the check of the records "
                         "found does not cover")
    generation = GenerationSettings.for_family(family, mode=Mode.THINK if args.greedy else args.mode,
                                               reasoning_budget=args.reasoning_budget, base_seed=args.seed)
    if args.greedy:
        generation = dataclasses.replace(generation, decoding=GREEDY)
    return generation.record()


def reuse_problems(c: Checkpoint, run: Run, *, system: str, item_set: ItemSet, slots: int,
                   family: LoadedFamily, model: str,
                   model_sha256: Callable[[], str]) -> tuple[list[str], str | None]:
    """Why the records a run finds cannot be scored again (module docstring: Scorer), each naming the manifest
    and what it records against what the run asks for, none when they can; and the llama-server build the
    manifest records. model is --model, model_sha256 gives its SHA-256 (called only for a manifest that
    records one)."""
    manifest_path, records_path = Path(f"{run.stem}.manifest.json"), Path(f"{run.stem}.jsonl")
    shown = rel(manifest_path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # a UnicodeDecodeError is one
        problem = f"{shown}: the manifest of the records found cannot be read ({type(exc).__name__}: {exc})"
        return [problem], None
    if not isinstance(manifest, dict):
        return [f"{shown}: the manifest of the records found is no JSON object"], None
    try:
        records = read_jsonl(records_path)
    except (JsonlError, OSError) as exc:
        return [f"{rel(records_path)}: the records found cannot be read ({exc})"], None
    try:
        generation = requested_generation(run.evaluate, family)
    except ValueError as exc:
        return [f"{shown}: the generation the run asks for cannot be derived: {exc}"], None
    props = manifest.get("server_props") if isinstance(manifest.get("server_props"), dict) else {}
    held = dict(Counter(r.get("status") for r in records))
    writes = _run_eval()  # the prompt and schema run_eval writes now, from run_eval itself
    problems = [f"{shown}: {what} {recorded!r}, where the run asks for {asked!r}"
                for what, recorded, asked in (
                    ("system", manifest.get("system"), system),
                    ("lora_scale", manifest.get("lora_scale"), LOCAL_SCALES[system]),
                    ("items_sha256", manifest.get("items_sha256"), item_set.sha256),
                    ("items_run", manifest.get("items_run"), item_set.n),
                    ("generation", manifest.get("generation"), generation),
                    ("prompt_version", manifest.get("prompt_version"), writes.PROMPT_VERSION),
                    ("prompt_hash", manifest.get("prompt_hash"), writes.PROMPT_HASH),
                    ("schema_hash", manifest.get("schema_hash"), writes.SCHEMA_HASH),
                    ("server_props.total_slots", props.get("total_slots"), slots))
                if recorded != asked]
    recorded, asked = manifest.get("family"), json.loads(json.dumps(family.record()))  # as manifests hold it
    if recorded != asked:  # the whole record; with one, its keys that differ
        if isinstance(recorded, dict):
            differ = sorted(k for k in recorded.keys() | asked.keys()
                            if k not in recorded or k not in asked or recorded[k] != asked[k])
            recorded = {k: recorded[k] for k in differ if k in recorded}
            asked = {k: asked[k] for k in differ if k in asked}
        problems.append(f"{shown}: family {recorded!r}, where the run asks for {asked!r}")
    path = props.get("model_path")  # the base model as the server was given it
    if not (isinstance(path, str) and path and same_path(path, model)):
        problems.append(f"{shown}: server_props.model_path {path!r}, where the run asks for {rel(model)!r} "
                        "(--model)")
    digest = manifest.get("base_model_sha256")  # recorded by no run_eval today
    if digest is not None and digest != model_sha256():
        problems.append(f"{shown}: base_model_sha256 {digest!r}, where the run asks for {model_sha256()!r} "
                        f"(--model {rel(model)})")
    adapters = manifest.get("adapters")
    if c.sha256 is None or not (isinstance(adapters, list) and len(adapters) == 1
                                and isinstance(adapters[0], dict) and adapters[0].get("sha256") == c.sha256
                                and same_path(adapters[0].get("path") or "", c.gguf)):
        problems.append(f"{shown}: adapters {adapters!r}, where the run asks for exactly {rel(c.gguf)} "
                        f"(sha256 {c.sha256})")
    if manifest.get("statuses") != held:
        problems.append(f"{shown}: statuses {manifest.get('statuses')!r}, but the records beside it hold "
                        f"{held!r} (a manifest of other records)")
    return problems, props.get("build_info")


def check_reused(plans: list[Plan], *, system: str, sets: list[ItemSet], slots: int, family: LoadedFamily,
                 model: str) -> tuple[list[str], dict[int, str | None]]:
    """Every run whose records a selection under a scorer other than the default finds, checked (module
    docstring: Scorer; reuse_problems): the problems, none when every one can be scored again, and per step
    with such runs the llama-server build their manifests record, which must be one. model is --model, hashed
    once at most, when a manifest records the SHA-256 of its base model."""
    by_path = {s.path: s for s in sets}
    problems: list[str] = []
    builds: dict[int, str | None] = {}
    model_sha256 = functools.cache(lambda: sha256_of(model))
    for p in plans:
        made: dict[str | None, list[str]] = {}  # each build the step's records were made by: its manifests
        for one in p.runs:
            if one.reuse:
                wrong, build = reuse_problems(p.checkpoint, one, system=system, item_set=by_path[one.items],
                                              slots=slots, family=family, model=model,
                                              model_sha256=model_sha256)
                problems += wrong
                if not wrong:
                    made.setdefault(build, []).append(rel(f"{one.stem}.manifest.json"))
        if len(made) > 1:
            listed = "; ".join(f"{build} in {', '.join(paths)}" for build, paths in made.items())
            problems.append(f"step {p.checkpoint.step}: the records found were made by several llama-server "
                            f"builds ({listed})")
        elif made:
            builds[p.checkpoint.step] = next(iter(made))
    return problems, builds


def scored_by_another(plans: list[Plan]) -> list[Path]:
    """The score files a scorer other than the default made of the records the runs of a default selection
    would write (module docstring: Scorer), <stem>.score-<scorer>.json, in the order of the runs."""
    return [path for p in plans for one in p.runs for scorer in SCORERS if scorer != DEFAULT_SCORER
            if (path := score_file(one.stem, scorer)).exists()]


def exported(p: Plan, *, run) -> tuple[int, Checkpoint | None]:
    """The plan's checkpoint with its GGUF's SHA-256, exporting the GGUF first when the plan says so
    (module docstring: --checkpoints): (0, checkpoint) or (6, None) when the export failed."""
    c = p.checkpoint
    if p.export is None:
        return 0, c
    print(f"step {c.step}: exporting {rel(c.adapter)} to {rel(c.gguf)}", flush=True)
    done = run(p.export)
    if done.returncode != 0 or not c.gguf.is_file():
        print(f"FAIL: step {c.step}: export_adapter exited with {done.returncode}"
              + ("" if c.gguf.is_file() else f" and wrote no {rel(c.gguf)}"))
        return 6, None
    return 0, Checkpoint(c.step, c.gguf, sha256_of(c.gguf))


def summed(per_set: dict[str, dict]) -> dict:
    """A run's summary over its sets, given each set's summary by the set's path (module docstring: Several
    sets): the one set's summary as it is; over several, n, successes and statuses summed, rate successes / n,
    and each set's summary under "sets"."""
    if len(per_set) == 1:
        return next(iter(per_set.values()))
    n = sum(s["n"] for s in per_set.values())
    successes = sum(s["successes"] for s in per_set.values())
    statuses = sum((Counter(s["statuses"]) for s in per_set.values()), Counter())
    return {"n": n, "successes": successes, "rate": successes / n, "statuses": dict(sorted(statuses.items())),
            "sets": per_set}


def row_of(c: Checkpoint, build: object, summaries: dict[str, dict], options: Options) -> dict:
    """A checkpoint's row of the selection file (module docstring): today's row from the default run, with
    the variants beside it when there are any; under --select-by the mean of the named variants. summaries
    are the runs' summaries over the sets (summed): with several sets, "sets" stands where "records" does."""
    variants = {name: summary for name, summary in summaries.items() if name != DEFAULT_RUN}
    head = {"step": c.step, "gguf": rel(c.gguf), "sha256": c.sha256}
    if options.selection is None:
        default = summaries[DEFAULT_RUN]
        row = {**head, "n": default["n"], "successes": default["successes"], "rate": default["rate"],
               "statuses": default["statuses"], "server_build": build,
               **{key: default[key] for key in ("records", "sets") if key in default}}
    else:
        chosen = [variants[name] for name in options.selection.variants]
        sizes = {s["n"] for s in chosen}
        if len(sizes) != 1:
            raise RunError(f"step {c.step}: the runs {options.selection.variants} scored different numbers "
                           f"of items {sorted(sizes)}")
        mean = Fraction(sum(s["successes"] for s in chosen), len(chosen))
        row = {**head, "n": chosen[0]["n"], "successes": exact(mean), "rate": float(mean / chosen[0]["n"]),
               "server_build": build}
    if variants:
        row["variants"] = variants
    return row


def exact(value: Fraction) -> int | float:
    """A mean as the selection file writes it: a whole number when it is one, else a decimal."""
    return int(value) if value.denominator == 1 else float(value)


def evaluate(p: Plan, *, binary: Path, args: argparse.Namespace, system: str, popen, run,
             client: Callable[[str], LocalClient], options: Options = NO_OPTIONS,
             reused_build: str | None = None) -> tuple[int, dict | None]:
    """Export when needed, serve, run and score one checkpoint (module docstring): (0, its row) or (exit code,
    None). Under a scorer other than the default (module docstring: Scorer), a checkpoint every run of which
    finds its records starts no server: they are scored again, and the row's server build is reused_build, the
    one their manifests record; with some runs that find their records, the server must be of that build."""
    summaries: dict[str, dict[str, dict]] = {}  # per run, each set's summary by the set's path
    if all(one.reuse for one in p.runs):
        c = p.checkpoint
        print(f"step {c.step}: every run's records are there: scored again, no server", flush=True)
        for one in p.runs:
            code, summary = score_run(c, one, system=system, do_run=run)
            if code:
                return code, None
            summaries.setdefault(one.name, {})[rel(one.items)] = summary
        return 0, checkpoint_row(c, reused_build, summaries, options)
    code, c = exported(p, run=run)
    if code:
        return code, None
    log = server_log(p.stem, options.scorer)
    print(f"step {c.step}: serving {rel(c.gguf)}", flush=True)
    try:
        with LocalServer(binary, model=args.model, adapter=rel(c.gguf), port=args.port, timeout=args.timeout,
                         log=log, popen=popen, client=client, extra=slot_args(args)) as server:
            listed = [a.get("path") if isinstance(a, dict) else a for a in server.adapters]
            if len(listed) != 1 or not isinstance(listed[0], str) or not same_path(listed[0], c.gguf):
                print(f"FAIL: step {c.step}: the server lists the adapters {listed}, "
                      f"not exactly {rel(c.gguf)}")
                return 3, None
            slots = (server.props or {}).get("total_slots")
            if slots != args.slots:
                print(f"FAIL: step {c.step}: the server reports {slots} slots (/props total_slots), not "
                      f"{args.slots}: its greedy output would depend on the request history")
                return 3, None
            build = (server.props or {}).get("build_info")
            if any(one.reuse for one in p.runs) and build != reused_build:
                print(f"FAIL: step {c.step}: the server is llama-server build {build}, but the records "
                      f"found of the step were made by build {reused_build}: the runs still to make would "
                      "come from another build than theirs (pass --records for a folder of new runs)")
                return 2, None
            for one in p.runs:
                code, summary = score_run(c, one, system=system, do_run=run)
                if code:
                    return code, None
                summaries.setdefault(one.name, {})[rel(one.items)] = summary
    except PortBusy as exc:
        print(f"FAIL: {exc}")
        return 2, None
    except NotReady as exc:
        print(f"FAIL: step {c.step}: {exc} (server log {rel(log)})")
        return 3, None
    return 0, checkpoint_row(c, build, summaries, options)


def selection_scorer(out: Path) -> str | None:
    """The scorer of the selection file at out: its "scorer", the default scorer's for a JSON object without
    one; None when there is no file or it is no JSON object."""
    try:
        held = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else None
    except (OSError, ValueError):  # a UnicodeDecodeError is one
        return None
    return held.get("scorer", DEFAULT_SCORER) if isinstance(held, dict) else None


def checkpoint_row(c: Checkpoint, build: object, summaries: dict[str, dict[str, dict]],
                   options: Options) -> dict:
    """A checkpoint's row from its runs' summaries per set (row_of), each run's total over several sets
    printed."""
    totals = {name: summed(per_set) for name, per_set in summaries.items()}
    for name, total in totals.items():
        if "sets" in total:  # several sets: the run's total over them
            tag = f" ({name})" if name != DEFAULT_RUN else ""
            print(f"step {c.step}{tag}: {total['successes']}/{total['n']} ({100 * total['rate']:.1f}%) over "
                  f"the {len(total['sets'])} sets", flush=True)
    return row_of(c, build, totals, options)


def main(argv: list[str] | None = None, *, popen=subprocess.Popen, run=subprocess.run,
         client: Callable[[str], LocalClient] = LocalClient,
         load_family: Callable[[str], LoadedFamily] = load_family_from_folder) -> int:
    """popen, run, client and load_family stand in for subprocess.Popen (llama-server), subprocess.run
    (export_adapter.py, run_eval.py and score.py), LocalClient and research_agent.family.load_family in
    tests."""
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    name = args.run.name
    system = args.system or (name if name in ADAPTER_SYSTEMS else None)
    if system is None:
        print(f"FAIL: the run folder's name {name!r} does not tell the system: "
              "pass --system ft or --system b4")
        return 2
    try:
        variants = parse_variants(args.variant)
        options = Options(steps=None if args.checkpoints is None else parse_steps(args.checkpoints),
                          variants=variants,
                          selection=None if args.select_by is None else parse_select_by(args.select_by,
                                                                                       variants),
                          llama_cpp_src=args.llama_cpp_src, scorer=args.scorer)
        paths = item_paths(args.items)
    except UsageError as exc:
        print(f"FAIL: {exc}")
        return 2
    opened: list[tuple[Path, str, list[dict]]] = []  # each set's path, SHA-256 and items, in order
    try:
        for path in paths:  # every set, before any is read
            refuse_test_sets(path, (Path.cwd(), REPO))
        for path in paths:
            dev = DevSet(path)
            items = dev.open()
            refuse_test_material(items, path)
            opened.append((path, dev.last_open["sha256"], items))
    except RefusedTestSet as exc:
        print(f"FAIL: {exc}")
        return 4
    except LockError as exc:
        print(f"FAIL: {exc}")
        return 5
    except (JsonlError, OSError) as exc:
        print(f"FAIL: malformed items: {exc}")
        return 1
    empty = [path for path, _, items in opened if not items]
    if empty:
        print(f"FAIL: {rel(empty[0])} holds no items")
        return 1
    try:
        disjoint([(path, items) for path, _, items in opened])
    except UsageError as exc:
        print(f"FAIL: {exc}")
        return 2
    sets = [ItemSet(path, digest, len(items)) for path, digest, items in opened]
    try:
        if options.partial:
            checkpoints, final = listed_checkpoints(args.run, options.steps), None
        else:
            checkpoints, final = find_checkpoints(args.run)
    except RunError as exc:
        print(f"FAIL: {exc}")
        return 1
    trained = run_family(args.run)
    loaded = None  # the tokenizer folder's family, once loaded
    if trained is not None:  # the checkpoints must be served under the template they were trained with
        try:
            loaded = load_family(args.tokenizer)
        except FamilyError as exc:
            print(f"FAIL: {exc}")
            return 2
        family = loaded.spec.name
        if family != trained:
            print(f"FAIL: the tokenizer folder {args.tokenizer} is the family {family}, but the run "
                  f"{rel(args.run)} trained the family {trained} (its manifest.json)")
            return 2
        print(f"family {family}: the tokenizer folder's family is the run manifest's")
    try:
        binary = llama_server(args.llama_server)
    except LookupError as exc:
        print(f"FAIL: {exc}")
        return 2
    if not binary.is_file():
        print(f"FAIL: llama-server not found: {binary} (--llama-server or {SERVER_VAR})")
        return 2
    if not Path(args.model).is_file():
        print(f"FAIL: the base model {args.model} is not here (--model)")
        return 2
    records = args.records or Path("runs") / "eval" / f"select-{name}"
    scored = "" if options.scorer == DEFAULT_SCORER else f"-{options.scorer}"
    out = args.out or Path("results") / f"checkpoint-selection-{name}{scored}.json"
    made = selection_scorer(out) if scored else None
    if made is not None and made != options.scorer:  # module docstring: Scorer
        print(f"FAIL: --out {rel(out)} holds a selection the {made} scorer made, and this one is the "
              f"{options.scorer} scorer's: give another --out")
        return 2
    plans = [plan(c, binary=binary, args=args, system=system, records=records, sets=sets, options=options)
             for c in checkpoints]
    describe(args, system, sets, plans, final, options)
    described = scored_by_another(plans) if options.scorer == DEFAULT_SCORER else []
    if described:  # module docstring: Scorer
        print("FAIL: a default selection makes its runs anew, writing over the records there, and another "
              "scorer has scored these runs' records: after a selection under another scorer, a default "
              "selection needs its own --records folder:")
        for path in described:
            print(f"  {rel(path)}")
        return 2
    reused = [one for p in plans for one in p.runs if one.reuse]  # only under a scorer other than the default
    builds: dict[int, str | None] = {}
    if reused:  # module docstring: Scorer
        try:
            loaded = load_family(args.tokenizer) if loaded is None else loaded
        except (FamilyError, OSError) as exc:
            print(f"FAIL: the records found are checked against the family of {args.tokenizer}: {exc}")
            return 2
        problems, builds = check_reused(plans, system=system, sets=sets, slots=args.slots, family=loaded,
                                        model=args.model)
        if problems:
            print(f"FAIL: under the {options.scorer} scorer the records found are scored again and never "
                  "written over, and these do not record what the selection asks for (--records names a "
                  "folder for new runs):")
            for problem in problems:
                print(f"  {problem}")
            return 2
    if args.dry_run:
        return 0
    outside = [rel(path) for path in (args.run, *paths, args.model, records, out) if not inside(path)]
    if outside:
        print(f"FAIL: run from the repository root: {outside} lie outside the current folder, and the "
              "selection file holds repository-relative paths only")
        return 2
    rows = []
    for p in plans:
        try:
            code, row = evaluate(p, binary=binary, args=args, system=system, popen=popen, run=run,
                                 client=client, options=options, reused_build=builds.get(p.checkpoint.step))
        except RunError as exc:
            print(f"FAIL: {exc}")
            return 1
        if code:
            return code
        rows.append(row)
    chosen = max(rows, key=lambda row: (Fraction(row["successes"]) / row["n"], row["step"]))
    several = len(sets) > 1
    rule = RULE if options.selection is None else options.selection.rule()
    rule = f"{rule} {SETS_RULE}" if several else rule
    if scored:
        rule = f"{rule} {SCORER_RULE.format(scorer=options.scorer)}"
    selection = {
        "run": rel(args.run), "system": system, "lora_scale": LOCAL_SCALES[system], "rule": rule,
        **({"scorer": options.scorer} if scored else {}),
        **({"records_reused": [rel(f"{one.stem}.jsonl") for one in reused]} if reused else {}),
        **({"item_sets": [{"path": rel(s.path), "sha256": s.sha256} for s in sets]} if several
           else {"items": rel(sets[0].path), "items_sha256": sets[0].sha256}),
        "base_model": rel(args.model),
        "server_args": server_command("llama-server", model=rel(args.model), adapter="<checkpoint gguf>",
                                      port=args.port, extra=slot_args(args))[1:],
        "server_slots": args.slots,
        "git_commit": git_commit(REPO), "git_dirty": git_dirty(REPO),
        **({"partial": True, "checkpoints_listed": options.steps} if options.partial else {}),
        **({"variants": [{"name": v.name, "args": list(v.args)} for v in options.variants]}
           if options.variants else {}),
        **({"selection": {"metric": options.selection.metric, "variants": list(options.selection.variants)}}
           if options.selection is not None else {}),
        "checkpoints": rows,
        "final_adapter": None if final is None else {"gguf": rel(final), "sha256": sha256_of(final),
                                                     "alias_of_step": checkpoints[-1].step},
        "chosen": {key: chosen[key] for key in ("step", "gguf", "sha256", "n", "successes", "rate")},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"chosen: step {chosen['step']}, {chosen['successes']}/{chosen['n']} "
          f"({100 * chosen['rate']:.1f}%), {rel(chosen['gguf'])}; selection {rel(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

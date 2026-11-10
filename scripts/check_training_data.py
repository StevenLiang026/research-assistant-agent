"""Check the samples a training run would build from a data mix: duplicates, conflicts, how the FT4 changes
files match the samples, the keys of the samples to train and the before/after table
(docs/m7-本地提升计划.md, WP6 样本层面的核对).

    .venv/Scripts/python.exe -B -X utf8 scripts/check_training_data.py --manifest runs/train/ft3/manifest.json
        [--keys-out FILE]
    .venv/Scripts/python.exe -B -X utf8 scripts/check_training_data.py <data files> --reasoning <sidecars>
        --overlap-report <report> [--glossary data/glossary/zh-en.json] [--model models/Nanbeige4.1-3B]
        [--max-length 5120] [--changes <changes files>] [--before runs/train/ft3/manifest.json]
        [--dedup-scope synthetic|all] [--allow-unresolved] [--keys-out FILE] [--cap 0.15]

Run it from the repository root, as scripts/train.py runs (a manifest's paths are relative to it).

The mix. Either the data files with the flags of scripts/train.py that decide the samples (--model, --family,
--reasoning, --no-think, --glossary, --glossary-dropout, --max-length, --overlap-report, --no-overlap-check),
or --manifest, a train.py manifest.json whose config.args give them (MIX_ARGS, GLOSSARY_ARGS when that run
had a glossary, and its sample keys: below); both at once is a usage error. The defaults are train.py's but
for the FT3 recipe's model folder and length (MODEL, MAX_LENGTH). The samples are train.py's own: its overlap
gate (train.overlap_gate: a data file the report does not cover at its current SHA-256 stops the check, exit
4, unless --no-overlap-check; the records the report names are skipped), its sidecar index, its glossary notes
(WP4: train.GlossaryNotes, passed to the assembly only when --glossary is given; a note's tokens count toward
the length) and its assembly (train.assemble through the family of the model folder, which renders every
sample with research_agent.train.samples.build_family_samples), with torch blocked as in train.py's dry run.
assemble is run on one data file at a time, which builds the same samples (it carries nothing from one file
to the next), and only a light form of each sample is held (describe). A malformed record, a stale sidecar
row, a sample the family refuses (all of them are counted first, as the dry run counts them) or a mix
without a sample stops the check (exit 2), as it stops a run. The steps that give no sample are then the
ones the assembly dropped over the length: their keys are kept (Mix.dropped_keys), and a count that differs
from the assembly's over_length stops the check.

A manifest's sample keys. A run given --sample-keys trained on the samples that file selects, so a manifest
whose config.args.sample_keys names one (with --manifest or --before) gives those samples: the mix is built,
then train.py's own load_sample_keys and select_samples pick them, in the file's order (Mix.samples, what is
grouped and counted; every sample built stays in Mix.built). The file must be there and be the one the run
read: its SHA-256 the manifest's data.sample_keys.sha256 (exit 2 otherwise, before anything is built: the
samples the run trained on cannot be told). A mix given by a manifest is compared with what the manifest
recorded of it (manifest_differences): the per-file accounts, the samples its run trained on (data.samples),
the SHA-256 of the data files, the sidecars, the glossary and the sample-keys file and the family; a
difference is reported, not failed on.

The glossary flags, as train.py checks them (glossary_flags_problem): an empty --glossary names no file, and
--glossary-dropout without --glossary would be ignored; either stops the check (exit 2) before anything is
read.

The samples' marks (describe). Each sample keeps its key as train.py's --sample-keys reads it: (record name,
k), the name being the record's "id" or "<file stem>:<index>" (train.samples.record_name) and k the turn's
ordinal among the record's assistant messages. It keeps its file, mode, think_fallback, tokens and whether it
carries a glossary note, and from the step it was cut from (the steps are cut again the way the assembly cuts
them, without notes, and matched to the samples in order): its step type, research_agent.eval.items.item_type
of (context, target, k + 1), the rule that types the dev items, applied to the training steps as the 10-29
coverage analysis applied it; whether its record is synthetic (an agent-history record); its answer hash,
data.reasoning.answer_sha256 of the target, the hash the sidecars tie a turn's answer by; its prompt hash, the
SHA-256 of the NO_THINK rendering of its context (the family's NO_THINK prompt, whatever the sample's own
mode) without a glossary note (the context is cut without notes, and without_glossary_note also drops a
note a data file holds: WP4's last line of the system message, starting with research_agent.glossary's
NOTE_PREFIX), so that samples differing only in their mode or in the note share it; and the SHA-256 of its
completion as trained (a THINK sample's holds its reasoning). The hashes of two mixes are compared (the
before mix's below), which holds when both are rendered by one family.

Duplicates and conflicts (resolve). The dedup scope (--dedup-scope) decides which samples are grouped:
"synthetic", the default (FT4 keeps FT3's recipe on the public and replay data: only the samples of
agent-history records are grouped, every other sample stays as built), or "all". The samples in scope are
grouped by prompt hash, and a prompt's samples by variant, (answer hash, mode). A prompt whose samples are all
one variant holds duplicates; a prompt with several variants (different targets, or a THINK and a NO_THINK
sample of one target) is a conflict. In a conflict the changed version wins (the changes files below): the
variants holding a relabel or new sample stay and the others are left out; with no such variant, or several,
the conflict is unresolved and those variants all stay. Of every variant that stays, the first sample in the
mix's order stays and the others are left out as duplicates, also when their completions differ (THINK
samples of one answer with other reasoning, or an answer that answer_sha256 reads alike spelled otherwise);
the duplicate prompts holding such samples are counted. An unresolved conflict that holds a synthetic or a
changed sample is flagged (FT3's own conflicts are in the public data) and fails the check (exit 2) unless
--allow-unresolved, which keeps it as above. Also counted: the prompts that synthetic and other samples share
("mixed": the synthetic scope does not group them) and what each scope would leave out.

The changes files (--changes; scripts/transform_synth.py --ft4 writes them with the FT4 data): JSON Lines, one
row per assistant turn, {"id", "k", "rule", "kind": "relabel" | "new" | "copy", "target": true | false,
"from": [id, k] | null}, exactly these keys (CHANGE_KEYS). A malformed row, or a turn in two rows, stops the
check (exit 2). The rows must match the samples built (changes_account); each mismatch fails the check
(exit 2, after the report):
- a row whose target is true names a sample built, whatever its kind (the owner's rule names the relabel and
  new rows; a copy that is a target is trained as well), or a turn the assembly dropped over the length,
  which is recorded; a row naming anything else (a key not in the data, a context turn, a record the overlap
  report skips) fails;
- a row whose target is false names no sample built: FT4 marks those turns context only ("context_turns" on
  the derived record, research_agent.train.samples: Context turns), so a run never trains them, and a sample
  built for one fails;
- with --before, a synthetic sample of the mix whose key the before mix did not build needs a row (a turn the
  before mix built is no new data, whether its own sample keys selected it or not).

The table. Sample counts by step type (all samples, the synthetic ones, the new ones) and by source file,
before and after: "after" is what the keys file lists, "before" the mix of --before (a train.py manifest; it
is built and counted only, its own sample keys applied) or, without it, this mix before the grouping.

The cap (new_samples, cap_reading). The new samples are the samples kept whose changes row is of kind "new",
and those of a copy row with target true unless a sample of the before mix holds its prompt hash and answer
hash (the turn moved from a cut record to a derived one: FT4's kept teacher asks); without --before no copy
can be shown to move, and every copy kept counts as new. They must be at most --cap (CAP, 15 %, compared
exactly: within_cap) of the before mix's synthetic samples (FT3: 8,483, so at most 1,272) or, without
--before, of the synthetic samples kept; with --before the other reading is printed too, and the first
decides. More fails the check (exit 3).

The keys file (--keys-out): the keys of the samples kept, in the mix's order, one JSON object {"id", "k"} per
line, UTF-8 with LF, as train.py --sample-keys reads them. It is staged next to FILE and read back with
train.py's own load_sample_keys and select_samples over every sample built, and it replaces FILE only when
that selects exactly the samples kept, in their order. A file of keys cannot tell apart two samples sharing a
key, so --keys-out with such samples in the mix stops the check (exit 2).

Printed: each mix with its build time, its sample keys and the manifest check, the changes and how they match
the samples, the dedup scope, the prompts, the duplicates and the conflicts (each with the data files their
samples are in), the exclusions, the two tables, the new samples and the cap, the failures (stderr) and the
keys file, then a last line "summary: " with the numbers as JSON.

Exit codes:
  0  done: the changes match the samples, no flagged conflict (or --allow-unresolved), the new samples within
     the cap, and the keys file written when asked for
  2  usage or input error: bad arguments (an empty --glossary, --glossary-dropout without --glossary among
     them), no data file, no model folder or an unregistered family, a malformed manifest, record, sidecar or
     changes file, a manifest's sample-keys file that is not there, is not the one its run read (another
     SHA-256) or names a key without a sample, a sample the family refuses, a mix without a sample, samples
     sharing a key with --keys-out; after the report, changes rows that do not match the samples (The changes
     files) or a flagged conflict without --allow-unresolved
  3  the new samples exceed the cap
  4  the overlap gate: a data file the benchmark-overlap report does not cover at its current SHA-256
Every failure found after the report is printed, and the exit is 2 when one of them is an input error.
Nothing is written unless the exit is 0.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import math
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from fractions import Fraction
from pathlib import Path
from typing import NamedTuple

from research_agent.canonical import split_lines
from research_agent.data.reasoning import ReasoningError, ReasoningIndex, answer_sha256
from research_agent.eval.items import TYPES, item_type
from research_agent.family.core import FamilyError, LoadedFamily, Mode
from research_agent.glossary import NOTE_PREFIX
from research_agent.tools.schemas import MODEL_TOOLS
from research_agent.train.manifest import sha256_of
from research_agent.train.samples import (
    REFUSAL_KINDS,
    Dropped,
    is_agent_history,
    load_records,
    record_name,
    step_pairs,
)

SCRIPTS = Path(__file__).resolve().parent


def _load_train():
    """scripts/train.py as a module of its own name (scripts/ is no package): its overlap gate, assembly and
    sample-keys reader are the ones this check runs."""
    spec = importlib.util.spec_from_file_location("_train_shared", SCRIPTS / "train.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


train = _load_train()

MODEL = "models/Nanbeige4.1-3B"  # the FT3 recipe's base and length (docs/m7-本地提升计划.md, WP7)
MAX_LENGTH = 5120
CAP = 0.15  # the new samples at most 15 % of the synthetic samples (WP6 配比)
CHANGE_KEYS = ("id", "k", "rule", "kind", "target", "from")
RELABEL, NEW, COPY = "relabel", "new", "copy"
KINDS = (RELABEL, NEW, COPY)
CHANGED = (RELABEL, NEW)  # the kinds whose version wins a conflict
CONFLICT, DUPLICATE = "conflict", "duplicate"  # why a sample is left out
REASONS = (CONFLICT, DUPLICATE)
SYNTHETIC, ALL = "synthetic", "all"  # the dedup scopes: which samples are grouped
SCOPES = (SYNTHETIC, ALL)
DEDUP_SCOPE = SYNTHETIC  # FT4 keeps FT3's recipe on the public and replay data (the owner, 10-30)
SCOPE_TEXT = {SYNTHETIC: "the samples of agent-history records are grouped, the others stay as built",
              ALL: "every sample is grouped"}
MIX_ARGS = ("data", "model", "family", "reasoning", "no_think", "max_length", "overlap_report",
            "no_overlap_check")  # train.py's arguments that decide the samples
GLOSSARY_ARGS = ("glossary", "glossary_dropout")  # and its glossary flags, in a manifest only when given
ACCOUNT_KEYS = ("records", "samples", "dropped", "skipped_overlap", "tokens", "modes", "refused")
AS_BUILT = "the mix as built"  # the before mix without --before
USAGE, CAP_EXCEEDED, OVERLAP_REFUSED = 2, 3, 4
SHOWN = 5  # keys, conflicts and refusals named in the output

LoadFamily = Callable[[str, str | None], LoadedFamily]  # (model folder, --family) -> the loaded family
SampleKey = tuple[str, int]
Step = tuple[str, int, list[dict], dict, list[dict] | None, bool]  # _steps: name, k, context, target, ...


class CheckError(Exception):
    """What stops the check, with its exit code (module docstring)."""

    def __init__(self, message: str, code: int = USAGE) -> None:
        super().__init__(message)
        self.code = code


class ChangesError(ValueError):
    """A changes file that cannot be used: a row that is malformed or names a turn another row named; the
    message names the place."""


@dataclasses.dataclass
class MixSpec:
    """train.py's inputs that decide a mix's samples (MIX_ARGS), the sample-keys file its run selected them
    with, where they came from (a manifest's path, or the command line) and the manifest itself when there is
    one."""

    data: list[str]
    model: str
    family: str | None
    reasoning: list[str]
    no_think: bool
    max_length: int
    overlap_report: str
    no_overlap_check: bool
    glossary: str | None = None
    glossary_dropout: float | None = None  # None: train.py's default
    sample_keys: str | None = None
    source: str = "the command line"
    manifest: dict | None = None


@dataclasses.dataclass
class Mix:
    """A mix as built: its inputs; the light samples (describe) it trains on, the ones its manifest's sample
    keys select, in their order, or else every sample built; every sample built, in train.py's order; the keys
    of the steps dropped over the length; the selection (None without sample keys); assemble's per-file
    accounts and drops, the overlap gate's record, the family's record and the seconds the build took."""

    spec: MixSpec
    samples: list[dict]
    built: list[dict]
    dropped_keys: list[SampleKey]
    selection: dict | None
    per_file: list[dict]
    dropped: Dropped
    overlap: dict
    family: dict
    seconds: float


class Resolution(NamedTuple):
    """What resolve decides: the samples kept, in order, how many were left out by reason (REASONS), the
    prompt, duplicate and conflict counts, the samples of the duplicate and of the conflict prompts
    ("duplicates", "conflicts"), in order, and the keys of each flagged conflict's samples."""

    kept: list[dict]
    excluded: dict[str, int]
    prompts: dict[str, int]
    duplicates: dict[str, int]
    conflicts: dict[str, int]
    grouped: dict[str, list[dict]]
    flagged: list[list[SampleKey]]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Duplicates, conflicts, the changes files' match, sample keys "
                                             "and the before/after table of the samples train.py would build "
                                             "from a data mix.")
    ap.add_argument("data", nargs="*", help="data files (JSONL or JSON); glob patterns are expanded here")
    ap.add_argument("--manifest", default=None, metavar="FILE",
                    help="a train.py manifest.json whose config.args give the mix, instead of data files and "
                         "the flags below up to --no-overlap-check")
    ap.add_argument("--model", default=None, help=f"model folder (default: {MODEL})")
    ap.add_argument("--family", default=None,
                    help="the registered family the model folder must resolve to (checked, never assumed)")
    ap.add_argument("--reasoning", nargs="+", default=None, metavar="FILE",
                    help="reasoning sidecars: a turn they hold becomes a THINK sample when the family thinks")
    ap.add_argument("--no-think", action="store_true", default=None, help="ignore the sidecars")
    ap.add_argument("--glossary", default=None, metavar="FILE",
                    help="a zh-en glossary: agent-history records take its term note on their system message "
                         "unless the dropout leaves them out (train.py --glossary)")
    ap.add_argument("--glossary-dropout", type=float, default=None, metavar="SHARE",
                    help="the share of records that train without their notes (default: train.py's)")
    ap.add_argument("--max-length", type=int, default=None,
                    help="longer samples are dropped, a THINK one rendered NO_THINK instead (default: "
                         f"{MAX_LENGTH})")
    ap.add_argument("--overlap-report", default=None,
                    help=f"the benchmark-overlap report (default: {train.OVERLAP_REPORT})")
    ap.add_argument("--no-overlap-check", action="store_true", default=None,
                    help="build the data files the overlap report does not cover")
    ap.add_argument("--changes", nargs="+", default=[], metavar="FILE",
                    help="FT4 changes files, one row per assistant turn (module docstring)")
    ap.add_argument("--before", default=None, metavar="MANIFEST",
                    help=f"the before mix of the table and the cap, a train.py manifest.json (default: "
                         f"{AS_BUILT})")
    ap.add_argument("--dedup-scope", choices=SCOPES, default=DEDUP_SCOPE,
                    help=f"the samples grouped for duplicates and conflicts (default: {DEDUP_SCOPE}: those "
                         "of agent-history records, the others stay as built)")
    ap.add_argument("--allow-unresolved", action="store_true",
                    help="keep the unresolved conflicts of synthetic or changed samples instead of failing")
    ap.add_argument("--keys-out", default=None, metavar="FILE",
                    help="write the keys of the samples to train, as train.py --sample-keys reads them")
    ap.add_argument("--cap", type=float, default=CAP,
                    help=f"the most new samples, as a share of the before mix's synthetic samples (of those "
                         f"kept without --before; default: {CAP})")
    return ap


def spec_from_args(args: argparse.Namespace) -> MixSpec:
    """The mix of the data files and the flags, with train.py's defaults but MODEL and MAX_LENGTH."""
    return MixSpec(data=list(args.data), model=args.model or MODEL, family=args.family,
                   reasoning=list(args.reasoning or []), no_think=bool(args.no_think),
                   max_length=MAX_LENGTH if args.max_length is None else args.max_length,
                   overlap_report=args.overlap_report or train.OVERLAP_REPORT,
                   no_overlap_check=bool(args.no_overlap_check), glossary=args.glossary,
                   glossary_dropout=args.glossary_dropout)


def spec_from_manifest(path: str | Path) -> MixSpec:
    """The mix a train.py manifest.json records under config.args (MIX_ARGS, GLOSSARY_ARGS when the run had a
    glossary, and sample_keys when it had a keys file); CheckError when the file is no such manifest or its
    keys file is not there."""
    try:
        manifest = json.loads(Path(path).read_text(encoding="utf-8"))
        recorded = manifest["config"]["args"]
        args = {key: recorded[key] for key in MIX_ARGS} | {key: recorded.get(key) for key in GLOSSARY_ARGS}
        keys = recorded.get("sample_keys")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise CheckError(f"{path} is not a train.py manifest with config.args holding {', '.join(MIX_ARGS)}: "
                         f"{type(exc).__name__}: {exc}") from exc
    if keys is not None and not (isinstance(keys, str) and keys):
        raise CheckError(f"{path}: config.args.sample_keys is neither null nor a file name: {keys!r}")
    if keys is not None and not Path(keys).is_file():
        raise CheckError(f"{path}: its run trained on the samples {keys} selects, and there is no {keys}, so "
                         "they cannot be told")
    recorded_keys = ((manifest.get("config") or {}).get("data") or {}).get("sample_keys")
    then = recorded_keys.get("sha256") if isinstance(recorded_keys, dict) else None
    if keys is not None and then is not None and sha256_of(keys) != then:
        raise CheckError(f"{path}: its run trained on the samples {keys} selected when its SHA-256 was "
                         f"{then}; the file's is {sha256_of(keys)} now, so they cannot be told")
    return MixSpec(data=list(args["data"]), model=args["model"], family=args["family"],
                   reasoning=list(args["reasoning"] or []), no_think=bool(args["no_think"]),
                   max_length=int(args["max_length"]), overlap_report=args["overlap_report"],
                   no_overlap_check=bool(args["no_overlap_check"]), glossary=args["glossary"],
                   glossary_dropout=args["glossary_dropout"], sample_keys=keys, source=str(path),
                   manifest=manifest)


def glossary_flags_problem(args: argparse.Namespace) -> str | None:
    """Why the glossary flags cannot be used, or None (train.py's glossary_flags_problem, whose words these
    are): an empty --glossary names no file, and --glossary-dropout without --glossary would be ignored (its
    default here is None, so a value means it was given)."""
    if args.glossary == "":
        return "--glossary names no file (an empty path)"
    if args.glossary is None and args.glossary_dropout is not None:
        return "--glossary-dropout needs --glossary: without a glossary there is no note to drop"
    return None


def mix_spec(args: argparse.Namespace) -> MixSpec:
    """The mix to check: --manifest's, or the data files' with the flags; CheckError for both or neither."""
    flags = [f"--{name.replace('_', '-')}" for name in (*MIX_ARGS[1:], *GLOSSARY_ARGS)
             if getattr(args, name) is not None]
    if args.manifest is not None:
        if args.data or flags:
            given = (["data files"] if args.data else []) + flags
            raise CheckError(f"give the mix by --manifest or by data files and flags, not both: "
                             f"{', '.join(given)} given with --manifest")
        return spec_from_manifest(args.manifest)
    if not args.data:
        raise CheckError("no data file and no --manifest: no mix to check")
    return spec_from_args(args)


def build_mix(spec: MixSpec, load_family: LoadFamily) -> Mix:
    """The samples train.py would build from the mix, in its order, in their light form, and the ones its
    sample keys select (module docstring: The mix; A manifest's sample keys); CheckError with train.py's exit
    code for what stops a run."""
    started = time.perf_counter()
    try:
        files = train.expand_data_files(spec.data)
    except FileNotFoundError as exc:
        raise CheckError(str(exc)) from exc
    if not Path(spec.model).is_dir():
        raise CheckError(f"no model folder at {spec.model}")
    with train.modules_blocked(train.BLOCKED_IN_DRY_RUN):
        try:
            family = load_family(spec.model, spec.family)
        except FamilyError as exc:
            raise CheckError(str(exc)) from exc
        try:
            skips, overlap = train.overlap_gate(spec.overlap_report, files, check=not spec.no_overlap_check)
        except ValueError as exc:
            raise CheckError(str(exc), OVERLAP_REFUSED) from exc
        try:
            index = ReasoningIndex(spec.reasoning) if spec.reasoning else None
            notes = {} if spec.glossary is None else {"notes": glossary_notes(spec)}
        except (ReasoningError, ValueError, OSError) as exc:  # a GlossaryError is a ValueError
            raise CheckError(str(exc)) from exc
        samples: list[dict] = []
        dropped_keys: list[SampleKey] = []
        per_file: list[dict] = []
        refused: list[dict] = []
        dropped = Dropped()
        for path in files:
            try:
                built, drops, account = train.assemble([path], family, spec.max_length, skips=skips,
                                                       reasoning=None if spec.no_think else index,
                                                       refused=refused, **notes)
                light, passed = describe(path, built, family, skips.get(path.resolve(), set()))
            except ValueError as exc:
                raise CheckError(str(exc)) from exc
            samples += light
            dropped_keys += passed
            per_file += account
            dropped += drops
    if refused:
        kinds = Counter(r["kind"] for r in refused)
        shown = "; ".join(f"{r['kind']} {r['key']!r}: {r['error'][:160]}" for r in refused[:SHOWN])
        raise CheckError(f"the family refuses {len(refused)} sample(s) "
                         f"({', '.join(f'{kind} {kinds[kind]}' for kind in REFUSAL_KINDS)}), where a run "
                         f"stops at the first: {shown}")
    if len(dropped_keys) != dropped.over_length:
        raise CheckError(f"{len(dropped_keys)} step(s) cut again gave no sample, where the assembly dropped "
                         f"{dropped.over_length} over the length: the steps and the samples do not match")
    if not samples:
        raise CheckError(f"no sample within {spec.max_length} tokens; dropped {dataclasses.asdict(dropped)}")
    chosen, selection = selected_samples(spec, samples)
    return Mix(spec, chosen, samples, dropped_keys, selection, per_file, dropped, overlap, family.record(),
               time.perf_counter() - started)


def selected_samples(spec: MixSpec, built: list[dict]) -> tuple[list[dict], dict | None]:
    """The samples a mix trains on and what is recorded of their selection: the ones its sample-keys file
    selects, through train.py's own load_sample_keys and select_samples, with the file's path, SHA-256 and
    counts; every sample built and None without a file. CheckError for a file train.py's loader refuses or a
    key without a sample."""
    if spec.sample_keys is None:
        return built, None
    try:
        keys = train.load_sample_keys(spec.sample_keys)
        chosen = train.select_samples(built, keys)
    except (OSError, ValueError) as exc:
        raise CheckError(f"{spec.sample_keys}: {exc}") from exc
    return chosen, {"path": spec.sample_keys, "sha256": sha256_of(spec.sample_keys), "listed": len(keys),
                    "selected": len(chosen)}


def glossary_notes(spec: MixSpec) -> object:
    """train.py's --glossary option for the mix: its GlossaryNotes over the glossary file, at the mix's
    dropout or train.py's default; ValueError (a GlossaryError among them) or OSError for a glossary or a
    dropout that cannot be used."""
    dropout = train.NOTE_DROPOUT if spec.glossary_dropout is None else spec.glossary_dropout
    return train.GlossaryNotes(train.load_glossary(spec.glossary), dropout)


def describe(path: Path, built: list[dict], family: LoadedFamily,
             skip: set[int]) -> tuple[list[dict], list[SampleKey]]:
    """The light form of a data file's samples (built: train.assemble's samples of the file, in its order):
    key, file, mode, think_fallback and tokens, and from each sample's step its type, prompt hash, answer
    hash and whether it is synthetic (module docstring: The samples' marks); and the keys of the steps that
    gave no sample, in order. The steps are cut again from the file's records as the assembly cuts them
    (load_records, the overlap skips, record_name, step_pairs with MODEL_TOOLS, which leaves out the context
    turns) and matched to the samples in order by key, so a step without a sample (over the length, refused)
    is passed over, unrendered, and a name that repeats in the file is still matched right."""
    steps = _steps(path, skip)
    light: list[dict] = []
    passed: list[SampleKey] = []
    for sample in built:
        key = tuple(sample["key"])
        for name, k, context, target, tools, synthetic in steps:
            if (name, k) == key:
                break
            passed.append((name, k))
        else:
            raise ValueError(f"{path}: the sample {key!r} matches no step of the file cut again")
        light.append({"key": key, "file": sample["file"], "mode": sample["mode"],
                      "think_fallback": sample["think_fallback"], "tokens": sample["tokens"],
                      "noted": sample.get("note") is not None,
                      "type": item_type(context, target, k + 1), "synthetic": synthetic,
                      "prompt_sha256": prompt_sha256(family, context, tools),
                      "answer_sha256": answer_sha256(target),
                      "completion_sha256": hashlib.sha256(sample["completion"].encode("utf-8")).hexdigest()})
    passed += [(name, k) for name, k, *_ in steps]
    return light, passed


def _steps(path: Path, skip: set[int]) -> Iterator[Step]:
    """(record name, k, context, target, tools, synthetic) of every step of the file's records that the
    overlap report does not skip, in the assembly's order."""
    for number, record in enumerate(load_records(path)):
        if number in skip:
            continue
        steps, tools, _ = step_pairs(record, MODEL_TOOLS)
        name, synthetic = record_name(record, path, number), is_agent_history(record)
        for k, context, target in steps:
            yield name, k, context, target, tools, synthetic


def without_glossary_note(context: list[dict]) -> list[dict]:
    """context without WP4's glossary note: when its first message is the system message and that message's
    last line starts with research_agent.glossary's NOTE_PREFIX, a copy whose system message ends before that
    line; context itself otherwise."""
    first = context[0] if context else {}
    content = first.get("content")
    if first.get("role") != "system" or not isinstance(content, str):
        return context
    head, newline, last = content.rpartition("\n")
    if not newline or not last.startswith(NOTE_PREFIX):
        return context
    return [{**first, "content": head}, *context[1:]]


def prompt_sha256(family: LoadedFamily, context: list[dict], tools: list[dict] | None) -> str:
    """The SHA-256 of the NO_THINK rendering of a step's context (the family's NO_THINK prompt, the one a
    NO_THINK sample trains on), without a glossary note."""
    hf = family.to_template_messages(without_glossary_note(context))
    return hashlib.sha256(family.prompt(hf, tools, Mode.NO_THINK).encode("utf-8")).hexdigest()


def check_change(row: object, where: str) -> dict:
    """row when it is a well-formed changes row (module docstring: The changes files); ChangesError naming
    where and every problem otherwise."""
    if not isinstance(row, dict) or set(row) != set(CHANGE_KEYS):
        raise ChangesError(f"{where}: a changes row is an object with exactly the keys "
                           f"{', '.join(CHANGE_KEYS)}")
    problems = []
    if not isinstance(row["id"], str) or not row["id"]:
        problems.append("id is not a non-empty string")
    if type(row["k"]) is not int or row["k"] < 0:
        problems.append("k is not an integer >= 0")
    if not isinstance(row["rule"], str) or not row["rule"]:
        problems.append("rule is not a non-empty string")
    if row["kind"] not in KINDS:
        problems.append(f"kind is not one of {', '.join(KINDS)}")
    if type(row["target"]) is not bool:
        problems.append("target is not true or false")
    source = row["from"]
    if source is not None and not (isinstance(source, list) and len(source) == 2 and
                                   isinstance(source[0], str) and source[0] and
                                   type(source[1]) is int and source[1] >= 0):
        problems.append("from is neither null nor [id, k]")
    if problems:
        raise ChangesError(f"{where}: {'; '.join(problems)}")
    return row


def load_changes(paths: list[str | Path]) -> dict[SampleKey, dict]:
    """(id, k) -> the changes row of that turn, over the files in order; ChangesError for a file that cannot
    be read, a malformed row or a turn in two rows."""
    rows: dict[SampleKey, dict] = {}
    places: dict[SampleKey, str] = {}
    for path in paths:
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            raise ChangesError(f"{path}: {exc}") from exc
        for number, line in enumerate(split_lines(text), 1):
            if not line.strip():
                continue
            where = f"{path}: line {number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ChangesError(f"{where} is not JSON: {exc}") from exc
            key = (check_change(row, where)["id"], row["k"])
            if key in rows:
                raise ChangesError(f"{where}: turn {key!r} is also in {places[key]}")
            rows[key], places[key] = row, where
    return rows


def resolve(samples: list[dict], changes: dict[SampleKey, dict], *, scope: str = ALL) -> Resolution:
    """The samples to train and what was left out (module docstring: Duplicates and conflicts). samples need
    key, prompt_sha256, answer_sha256 and mode; completion_sha256 is compared when it is there, and
    "synthetic" is read for the synthetic scope (a sample without it is out of that scope). The default scope
    groups every sample given."""
    excluded = dict.fromkeys(REASONS, 0)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        if scope == ALL or sample.get("synthetic"):
            groups[sample["prompt_sha256"]].append(index)
    duplicates = {"groups": 0, "samples": 0, "completions_differ": 0}
    conflicts = {"groups": 0, "targets": 0, "modes": 0, "resolved": 0, "unresolved": 0, "flagged": 0,
                 "samples": 0}
    grouped: dict[str, list[int]] = {"duplicates": [], "conflicts": []}
    flagged: list[list[SampleKey]] = []
    out: dict[int, str] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        variants: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index in members:
            variants[samples[index]["answer_sha256"], samples[index]["mode"]].append(index)
        if len(variants) == 1:
            duplicates["groups"] += 1
            duplicates["samples"] += len(members)
            completions = {samples[index].get("completion_sha256") for index in members}
            duplicates["completions_differ"] += len(completions) > 1
            grouped["duplicates"] += members
        else:
            conflicts["groups"] += 1
            conflicts["samples"] += len(members)
            grouped["conflicts"] += members
            conflicts["targets" if len({answer for answer, _ in variants}) > 1 else "modes"] += 1
            changed = [variant for variant, indices in variants.items()
                       if any(_changed(samples[i], changes) for i in indices)]
            if len(changed) == 1:
                conflicts["resolved"] += 1
            else:
                conflicts["unresolved"] += 1
                if any(samples[i].get("synthetic") or _changed(samples[i], changes) for i in members):
                    conflicts["flagged"] += 1
                    flagged.append([samples[i]["key"] for i in members])
            for variant, indices in variants.items():
                if changed and variant not in changed:
                    out.update(dict.fromkeys(indices, CONFLICT))
        for indices in variants.values():
            for index in indices[1:]:
                out.setdefault(index, DUPLICATE)
    for reason in out.values():
        excluded[reason] += 1
    synthetic = {sample["prompt_sha256"] for sample in samples if sample.get("synthetic")}
    others = {sample["prompt_sha256"] for sample in samples if not sample.get("synthetic")}
    prompts = {"distinct": len(groups), "shared": sum(1 for members in groups.values() if len(members) > 1),
               "mixed": len(synthetic & others)}
    kept = [sample for index, sample in enumerate(samples) if index not in out]
    return Resolution(kept, excluded, prompts, duplicates, conflicts,
                      {kind: [samples[i] for i in sorted(indices)] for kind, indices in grouped.items()},
                      flagged)


def _changed(sample: dict, changes: dict[SampleKey, dict]) -> bool:
    row = changes.get(sample["key"])
    return row is not None and row["kind"] in CHANGED


def changes_account(changes: dict[SampleKey, dict], mix: Mix, kept: list[dict], before: Mix | None) -> dict:
    """The changes rows by target, by kind and by rule (each kind and rule with how many of its turns are
    kept) and how they match the mix's samples (module docstring: The changes files): how many rows with
    target true name a sample built ("named"), the ones naming a turn dropped over the length ("over_length",
    recorded) and the ones naming nothing ("without_sample"); the rows with target false naming a sample built
    ("context_built"); and, with a before mix, the synthetic samples of the mix whose key it did not build and
    no row names ("uncovered", None without one). Each of the last four is its count and its first SHOWN
    keys."""
    built = {sample["key"] for sample in mix.built}
    dropped = set(mix.dropped_keys)
    stayed = {sample["key"] for sample in kept}
    kinds = {kind: {"rows": 0, "kept": 0} for kind in KINDS}
    rules: dict[str, dict[str, int]] = {}
    targets = {"true": 0, "false": 0}
    named = 0
    over_length: list[SampleKey] = []
    without_sample: list[SampleKey] = []
    context_built: list[SampleKey] = []
    for key, row in changes.items():
        for entry in (kinds[row["kind"]], rules.setdefault(row["rule"], {"rows": 0, "kept": 0})):
            entry["rows"] += 1
            entry["kept"] += int(key in stayed)
        targets["true" if row["target"] else "false"] += 1
        if row["target"]:
            if key in built:
                named += 1
            else:
                (over_length if key in dropped else without_sample).append(key)
        elif key in built:
            context_built.append(key)
    uncovered = None
    if before is not None:
        before_keys = {sample["key"] for sample in before.built}
        uncovered = [sample["key"] for sample in mix.samples if sample["synthetic"] and
                     sample["key"] not in before_keys and sample["key"] not in changes]
    return {"rows": len(changes), "targets": targets, "kinds": kinds, "rules": dict(sorted(rules.items())),
            "named": named, "over_length": _first(over_length), "without_sample": _first(without_sample),
            "context_built": _first(context_built),
            "uncovered": None if uncovered is None else _first(uncovered)}


def _first(keys: list[SampleKey]) -> dict:
    return {"count": len(keys), "keys": [list(key) for key in keys[:SHOWN]]}


def new_samples(kept: list[dict], changes: dict[SampleKey, dict],
                before: Mix | None) -> tuple[list[dict], dict[str, int]]:
    """The samples kept that the cap counts as new, in order, and their counts (module docstring: The cap):
    "new", those of a changes row of kind new; "copies_new", those of a copy row with target true whose prompt
    and answer hashes no sample of the before mix holds (every such copy without a before mix); and
    "copies_moved", the copies it does hold, which are not new."""
    seen = None if before is None else {(s["prompt_sha256"], s["answer_sha256"]) for s in before.samples}
    new: list[dict] = []
    counts = {"new": 0, "copies_new": 0, "copies_moved": 0}
    for sample in kept:
        row = changes.get(sample["key"])
        if row is None or not row["target"] or row["kind"] == RELABEL:
            continue
        if row["kind"] == COPY and seen is not None and \
                (sample["prompt_sha256"], sample["answer_sha256"]) in seen:
            counts["copies_moved"] += 1
            continue
        counts[NEW if row["kind"] == NEW else "copies_new"] += 1
        new.append(sample)
    return new, counts


def within_cap(new: int, synthetic: int, cap: float) -> bool:
    """True when new is at most cap of synthetic, compared exactly (the cap read as the decimal it prints as:
    0.15 is 3/20)."""
    return new <= Fraction(str(cap)) * synthetic


def at_most(cap: float, synthetic: int) -> int:
    """The most new samples within the cap of synthetic samples (within_cap's bound, rounded down)."""
    return math.floor(Fraction(str(cap)) * synthetic)


def cap_reading(new: int, kept_synthetic: int, before_synthetic: int | None, cap: float) -> dict:
    """The cap as it decides (module docstring: The cap): its base ("before" with a before mix, "after"
    without), the base's synthetic samples and the most new samples it allows, the share of the new samples
    in it, both shares (of the before mix's synthetic samples, None without one; of the synthetic samples
    kept; None for no sample) and whether the new samples are within the cap."""
    base = "after" if before_synthetic is None else "before"
    synthetic = kept_synthetic if before_synthetic is None else before_synthetic
    shares = {"before": new / before_synthetic if before_synthetic else None,
              "after": new / kept_synthetic if kept_synthetic else None}
    return {"base": base, "synthetic": synthetic, "at_most": at_most(cap, synthetic), "share": shares[base],
            "shares": shares, "within": within_cap(new, synthetic, cap)}


def shared_keys(samples: list[dict]) -> list[SampleKey]:
    """The keys that name more than one sample, in the order they first appear."""
    counts = Counter(sample["key"] for sample in samples)
    return [key for key, count in counts.items() if count > 1]


def write_sample_keys(kept: list[dict], samples: list[dict], path: str | Path) -> dict:
    """Write the keys file (module docstring: The keys file; samples: every sample built) and return what the
    summary records of it: its path, SHA-256 and the keys listed. CheckError, nothing written, when samples
    sharing a key are in the mix or train.py's loader reads the file back as other samples."""
    shared = shared_keys(samples)
    if shared:
        raise CheckError(f"{len(shared)} key(s) name more than one sample, so a keys file cannot list one "
                         f"without the other: {', '.join(repr(key) for key in shared[:SHOWN])}"
                         f"{', ...' if len(shared) > SHOWN else ''}")
    path = Path(path)
    staged = path.with_name(path.name + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    staged.write_text("".join(json.dumps({"id": s["key"][0], "k": s["key"][1]}, ensure_ascii=False) + "\n"
                              for s in kept), encoding="utf-8", newline="\n")
    try:
        selected = train.select_samples(samples, train.load_sample_keys(staged))
    except ValueError as exc:
        staged.unlink()
        raise CheckError(f"train.py's loader refuses the keys file: {exc}") from exc
    if len(selected) != len(kept) or any(a is not b for a, b in zip(selected, kept, strict=False)):
        staged.unlink()
        raise CheckError("the keys file, read back with train.py's loader, selects other samples than the "
                         "ones kept")
    staged.replace(path)
    return {"path": str(path), "sha256": sha256_of(path), "listed": len(kept)}


def manifest_differences(mix: Mix) -> list[str]:
    """What differs between a mix rebuilt from a train.py manifest and what the manifest recorded of it, one
    line each: the per-file accounts (ACCOUNT_KEYS, those the manifest holds), the samples its run trained on
    (data.samples, when it holds them), the SHA-256 of the sample-keys file (data.sample_keys), of the data
    files and of the sidecars, and the family (train.FAMILY_KEYS, those it holds); [] when nothing does."""
    manifest = mix.spec.manifest or {}
    config = manifest.get("config") or {}
    data = config.get("data") or {}
    then = {_posix(f["path"]): f for f in data.get("files", [])}
    now = {_posix(f["path"]): f for f in mix.per_file}
    lines = [f"{path}: {'recorded, not rebuilt' if path in then else 'rebuilt, not recorded'}"
             for path in [*then, *now] if (path in then) != (path in now)]
    for path in [path for path in then if path in now]:
        lines += [f"{path}: {key} {then[path][key]!r} -> {now[path].get(key)!r}" for key in ACCOUNT_KEYS
                  if key in then[path] and then[path][key] != now[path].get(key)]
    if data.get("samples") is not None and data["samples"] != len(mix.samples):
        lines.append(f"samples: {data['samples']!r} -> {len(mix.samples)!r}")
    then_keys = (data.get("sample_keys") or {}).get("sha256")
    now_keys = None if mix.selection is None else mix.selection["sha256"]
    if then_keys != now_keys:
        lines.append(f"sample_keys.sha256: {then_keys!r} -> {now_keys!r}")
    glossary = [data["glossary"]] if isinstance(data.get("glossary"), dict) else []
    sidecars = (data.get("reasoning") or {}).get("files", [])
    for entry in [*manifest.get("data_files", []), *sidecars, *glossary]:
        current = sha256_of(entry["path"]) if Path(entry["path"]).is_file() else "absent"
        if current != entry.get("sha256"):
            lines.append(f"{_posix(entry['path'])}: sha256 {entry.get('sha256')} -> {current}")
    family = config.get("family") or {}
    lines += [f"family.{key}: {family[key]!r} -> {mix.family.get(key)!r}" for key in train.FAMILY_KEYS
              if key in family and family[key] != mix.family.get(key)]
    return lines


def _posix(path: str | Path) -> str:
    return str(path).replace("\\", "/")


def tally(samples: list[dict], files: list[str]) -> dict:
    """What the table shows of samples: how many, their modes, how many are synthetic, by step type (TYPES
    order) all and synthetic, and by file (files first, in their order, with zeros)."""
    types = Counter(s["type"] for s in samples)
    synthetic = Counter(s["type"] for s in samples if s["synthetic"])
    order = [*TYPES, *sorted(set(types) - set(TYPES))]
    by_file = dict.fromkeys(files, 0)
    for sample in samples:
        by_file[_posix(sample["file"])] = by_file.get(_posix(sample["file"]), 0) + 1
    return {"samples": len(samples), "modes": train.mode_counts(samples),
            "synthetic": sum(synthetic.values()), "noted": sum(1 for s in samples if s.get("noted")),
            "types": {t: types[t] for t in order}, "synthetic_types": {t: synthetic[t] for t in order},
            "files": by_file}


def table(header: list[str], rows: list[list]) -> list[str]:
    """The rows under the header, the first column left-aligned and the others right-aligned."""
    cells = [header, *([str(cell) for cell in row] for row in rows)]
    widths = [max(len(row[i]) for row in cells) for i in range(len(header))]
    return ["  ".join(cell.ljust(width) if i == 0 else cell.rjust(width)
                      for i, (cell, width) in enumerate(zip(row, widths, strict=True))).rstrip()
            for row in cells]


def signed(n: int) -> str:
    return f"{n:+d}" if n else "0"


def keys_text(listed: dict) -> str:
    """The first keys of a _first account, ", ..." after them when there are more."""
    return (", ".join(repr(tuple(key)) for key in listed["keys"])
            + (", ..." if listed["count"] > len(listed["keys"]) else ""))


def mix_lines(label: str, mix: Mix, differences: list[str] | None) -> list[str]:
    """What is printed of a mix: its inputs, what was built and how long it took, its sample keys and the
    manifest check."""
    spec, dropped = mix.spec, mix.dropped
    sidecars = f"{len(spec.reasoning)} sidecar(s)" + (" ignored (--no-think)" if spec.no_think else "")
    skipped = sum(f["skipped_overlap"] for f in mix.per_file)
    glossary, notes = "", ""
    if spec.glossary is not None:
        dropout = train.NOTE_DROPOUT if spec.glossary_dropout is None else spec.glossary_dropout
        glossary = f", glossary {_posix(spec.glossary)} (dropout {dropout:g})"
        notes = f"; notes on {sum(1 for s in mix.built if s['noted'])} sample(s)"
    lines = [(f"{label}: {len(mix.per_file)} data file(s) from {spec.source}; model {spec.model} "
              f"(family {mix.family['name']}), {sidecars}, max length {spec.max_length}{glossary}"),
             (f"  built: {len(mix.built)} samples ({train.modes_text(train.mode_counts(mix.built))}); "
              f"dropped over_length {dropped.over_length}, after_repair {dropped.after_repair}; overlap "
              f"skips {skipped}{notes}; {mix.seconds:.1f} s")]
    if mix.selection is not None:
        selection = mix.selection
        lines.append(f"  sample keys: {_posix(selection['path'])} lists {selection['listed']}, selecting "
                     f"{selection['selected']} of the {len(mix.built)} samples built, the samples its run "
                     f"trained on (sha256 {selection['sha256']})")
    if differences is not None:
        lines.append("  manifest check: equal (per-file accounts, samples trained, SHA-256 of the sample "
                     "keys, data and sidecars, family)" if not differences
                     else f"  manifest check: {len(differences)} difference(s)")
        lines += [f"    {line}" for line in differences]
    return lines


def changes_lines(account: dict, files: int) -> list[str]:
    """What is printed of the changes files: the rows by target, kind and rule, and how they match the
    samples."""
    lines = ["changes: none"] if not files else [
        (f"changes: {account['rows']} row(s) from {files} file(s); target true {account['targets']['true']}, "
         f"false {account['targets']['false']}"),
        "  by kind: " + ", ".join(f"{kind} {c['rows']} ({c['kept']} kept)"
                                  for kind, c in account["kinds"].items()),
        "  by rule: " + ", ".join(f"{rule} {c['rows']} ({c['kept']} kept)"
                                  for rule, c in account["rules"].items()),
        (f"  target true: {account['named']} name a sample built, {account['over_length']['count']} a turn "
         f"dropped over the length, {account['without_sample']['count']} none")
        + "".join(f"; {label}: {keys_text(account[key])}" for key, label in
                  (("over_length", "dropped"), ("without_sample", "none")) if account[key]["count"]),
        (f"  target false: {account['context_built']['count']} name a sample built (a context turn gives "
         "none)") + (f": {keys_text(account['context_built'])}" if account["context_built"]["count"] else "")]
    if account["uncovered"] is not None:
        uncovered = account["uncovered"]
        lines.append(f"synthetic samples whose key the before mix did not build, with no changes row: "
                     f"{uncovered['count']}" + (f": {keys_text(uncovered)}" if uncovered["count"] else ""))
    return lines


def grouped_files(resolution: Resolution) -> dict[str, dict[str, int]]:
    """How many samples of the duplicate and of the conflict prompts each data file holds, the files in the
    order their first such sample comes."""
    return {kind: dict(Counter(_posix(sample["file"]) for sample in samples))
            for kind, samples in resolution.grouped.items()}


def conflicts_text(flagged: list[list[SampleKey]]) -> str:
    """The first flagged conflicts, each as the list of its samples' first keys."""
    groups = ["[" + ", ".join(repr(tuple(key)) for key in group[:SHOWN])
              + (", ...]" if len(group) > SHOWN else "]") for group in flagged[:SHOWN]]
    return "; ".join(groups) + ("; ..." if len(flagged) > SHOWN else "")


def outcome_lines(resolution: Resolution, after: dict, shared: list[SampleKey], scope: str,
                  scopes: dict[str, dict[str, int]]) -> list[str]:
    """What is printed of the dedup scope, the prompts, duplicates, conflicts, exclusions and keys shared by
    samples."""
    prompts, duplicates, conflicts = resolution.prompts, resolution.duplicates, resolution.conflicts
    excluded, by_file = resolution.excluded, grouped_files(resolution)
    where = {kind: [f"  samples by file: {', '.join(f'{f} {n}' for f, n in files.items())}"] if files else []
             for kind, files in by_file.items()}
    other = ALL if scope == SYNTHETIC else SYNTHETIC
    sharing = f" key(s) name more than one sample: {', '.join(repr(key) for key in shared[:SHOWN])}"
    flagged = f": {conflicts_text(resolution.flagged)}" if resolution.flagged else ""
    return [(f"dedup scope: {scope} ({SCOPE_TEXT[scope]}); the scope {other} would leave out "
             f"{sum(scopes[other].values())} (duplicate {scopes[other][DUPLICATE]}, lost a conflict "
             f"{scopes[other][CONFLICT]})"),
            (f"prompts: {prompts['distinct']} distinct in the scope; {prompts['shared']} hold more than one "
             f"sample; {prompts['mixed']} held by synthetic and other samples"),
            (f"duplicates: {duplicates['groups']} prompt(s) whose samples share one target and mode, "
             f"{duplicates['samples']} samples; {duplicates['completions_differ']} of them with completions "
             "that differ (another reasoning, or the answer spelled otherwise), the first kept all the same"),
            *where["duplicates"],
            (f"conflicts: {conflicts['groups']} prompt(s) with more than one target or mode, "
             f"{conflicts['samples']} samples (different targets {conflicts['targets']}, THINK against "
             f"NO_THINK only {conflicts['modes']}); resolved by a changed version {conflicts['resolved']}, "
             f"unresolved {conflicts['unresolved']} (each target and mode kept once), of them with a "
             f"synthetic or changed sample {conflicts['flagged']}{flagged}"),
            *where["conflicts"],
            (f"excluded: lost a conflict {excluded[CONFLICT]}, duplicate {excluded[DUPLICATE]}; kept "
             f"{after['samples']} of {after['selected']} ({train.modes_text(after['modes'])})"),
            f"shared keys: {len(shared)}" + (sharing if shared else "")]


def table_lines(before: dict, after: dict) -> list[str]:
    """The two tables: samples by step type (all, synthetic, new) and by source file, before and after."""
    rows = []
    for t in dict.fromkeys([*before["types"], *after["types"]]):
        was, now = before["types"].get(t, 0), after["types"].get(t, 0)
        rows.append([t, was, now, signed(now - was), before["synthetic_types"].get(t, 0),
                     after["synthetic_types"].get(t, 0), after["new_types"].get(t, 0)])
    total = ["total", before["samples"], after["samples"], signed(after["samples"] - before["samples"])]
    lines = ["== samples by step type (research_agent.eval.items.item_type)"]
    lines += table(["type", "before", "after", "change", "before synth", "after synth", "new"],
                   [*rows, [*total, before["synthetic"], after["synthetic"], after["new"]]])
    rows = []
    for f in dict.fromkeys([*before["files"], *after["files"]]):
        was, now = before["files"].get(f, 0), after["files"].get(f, 0)
        rows.append([f, was, now, signed(now - was), after["new_files"].get(f, 0)])
    lines += ["", "== samples by source file"]
    lines += table(["file", "before", "after", "change", "new"], [*rows, [*total, after["new"]]])
    return lines


def _counted(new: int, share: float | None) -> str:
    return f"{new} ({'no synthetic sample' if share is None else f'{share:.1%}'})"


def new_lines(new: int, counts: dict[str, int], cap: float, reading: dict, kept_synthetic: int) -> list[str]:
    """What is printed of the new samples and the cap: the new samples by kind, then the reading that decides
    and, with a before mix, the other one (module docstring: The cap)."""
    moved = (f"moved, their prompt and answer in the before mix: {counts['copies_moved']}"
             if reading["base"] == "before" else "no before mix: every copy kept counts as new")
    verdict = "ok" if reading["within"] else "EXCEEDED"
    percent = f"{cap * 100:g}%"
    lines = [f"new samples: {new}: changes kind new {counts[NEW]}, copies {counts['copies_new']} ({moved})"]
    if reading["base"] == "before":
        lines += [(f"cap {percent} of the before mix's {reading['synthetic']} synthetic samples: "
                   f"{_counted(new, reading['shares']['before'])}, at most {reading['at_most']}: {verdict}"),
                  (f"  the other reading, of the {kept_synthetic} synthetic samples kept: "
                   f"{_counted(new, reading['shares']['after'])}, at most {at_most(cap, kept_synthetic)} "
                   "(reported, not decided on)")]
    else:
        lines.append(f"cap {percent} of the {reading['synthetic']} synthetic samples kept: "
                     f"{_counted(new, reading['shares']['after'])}, at most {reading['at_most']}: {verdict}")
    return lines


def mismatches(account: dict, resolution: Resolution, allow_unresolved: bool) -> list[str]:
    """The failures of exit 2 found after the report: changes rows that do not match the samples (module
    docstring: The changes files) and, unless allowed, the flagged conflicts."""
    failures = []
    without, context, uncovered = account["without_sample"], account["context_built"], account["uncovered"]
    if without["count"]:
        failures.append(f"{without['count']} changes row(s) with target true name no sample built (a key not "
                        "in the data, a context turn, or a record the overlap report skips): "
                        f"{keys_text(without)}")
    if context["count"]:
        failures.append(f"{context['count']} changes row(s) with target false name a sample built, which a "
                        "run would train (its record does not mark the turn as context_turns): "
                        f"{keys_text(context)}")
    if uncovered is not None and uncovered["count"]:
        failures.append(f"{uncovered['count']} synthetic sample(s) whose key the before mix did not build "
                        f"have no changes row: {keys_text(uncovered)}")
    if resolution.flagged and not allow_unresolved:
        failures.append(f"{len(resolution.flagged)} unresolved conflict(s) of synthetic or changed samples, "
                        f"each target and mode of a prompt kept once: {conflicts_text(resolution.flagged)}; "
                        "--allow-unresolved lets them through")
    return failures


def main(argv: list[str] | None = None, *, load_family: LoadFamily | None = None) -> int:
    """load_family stands in for research_agent.family.core.load_family in the tests (train.py's hook)."""
    started = time.perf_counter()
    args = build_parser().parse_args(argv)
    problem = glossary_flags_problem(args)
    if problem is not None:  # before anything is read
        print(f"FAIL: {problem}", file=sys.stderr)
        return USAGE
    load = load_family or train.default_load_family
    try:
        if not 0 <= args.cap <= 1:
            raise CheckError(f"--cap is a share from 0 to 1, not {args.cap}")
        spec = mix_spec(args)
        changes = load_changes(args.changes)
        before_spec = None if args.before is None else spec_from_manifest(args.before)
        after_mix = build_mix(spec, load)
        before_mix = None if before_spec is None else build_mix(before_spec, load)
    except ChangesError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return USAGE
    except CheckError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return exc.code
    resolution = resolve(after_mix.samples, changes, scope=args.dedup_scope)
    scopes = {scope: resolution.excluded if scope == args.dedup_scope else
              resolve(after_mix.samples, changes, scope=scope).excluded for scope in SCOPES}
    kept = resolution.kept
    new, counts = new_samples(kept, changes, before_mix)
    after_files = [_posix(f["path"]) for f in after_mix.per_file]
    new_tally = tally(new, after_files)
    after = {"source": spec.source, "built": len(after_mix.built), "selected": len(after_mix.samples),
             **tally(kept, after_files), "new": len(new), "new_types": new_tally["types"],
             "new_files": new_tally["files"], "seconds": round(after_mix.seconds, 1)}
    reference = before_mix or after_mix
    before = {"source": AS_BUILT if before_mix is None else before_mix.spec.source,
              "built": len(reference.built), "selected": len(reference.samples),
              **tally(reference.samples, [_posix(f["path"]) for f in reference.per_file]),
              "seconds": None if before_mix is None else round(before_mix.seconds, 1)}
    differences = {label: None if mix is None or mix.spec.manifest is None else manifest_differences(mix)
                   for label, mix in (("after", after_mix), ("before", before_mix))}
    account = changes_account(changes, after_mix, kept, before_mix)
    shared = shared_keys(after_mix.built)
    reading = cap_reading(len(new), after["synthetic"], None if before_mix is None else before["synthetic"],
                          args.cap)

    lines = mix_lines("after", after_mix, differences["after"])
    lines += [f"before: {AS_BUILT}"] if before_mix is None else \
        mix_lines("before", before_mix, differences["before"])
    lines += [*changes_lines(account, len(args.changes)),
              *outcome_lines(resolution, after, shared, args.dedup_scope, scopes), ""]
    lines += [*table_lines(before, after), ""]
    lines += new_lines(len(new), counts, args.cap, reading, after["synthetic"])
    print("\n".join(lines))
    failures = mismatches(account, resolution, args.allow_unresolved)
    code = USAGE if failures else 0
    if not reading["within"]:
        base = (f"the before mix's {reading['synthetic']} synthetic samples" if reading["base"] == "before"
                else f"the {reading['synthetic']} synthetic samples kept")
        failures.append(f"new samples exceed the cap: {len(new)} of {base}, more than {args.cap * 100:g}%")
        code = code or CAP_EXCEEDED
    flagged = [[list(key) for key in group[:SHOWN]] for group in resolution.flagged[:SHOWN]]
    summary = {"after": after, "before": before, "dedup_scope": args.dedup_scope,
               "prompts": resolution.prompts, "duplicates": resolution.duplicates,
               "conflicts": resolution.conflicts, "flagged_conflicts": flagged,
               "allow_unresolved": args.allow_unresolved, "grouped_files": grouped_files(resolution),
               "excluded": resolution.excluded, "scopes": scopes, "changes": account, "new": counts,
               "shared_keys": len(shared), "cap": args.cap, "cap_base": reading["base"],
               "share": reading["share"], "shares": reading["shares"], "at_most": reading["at_most"],
               "within_cap": reading["within"], "sample_keys": None, "manifest_check": differences}
    for failure in failures:
        print(f"FAIL: {failure}", file=sys.stderr)
    if failures and args.keys_out is not None:
        print(f"no keys file written ({args.keys_out})", file=sys.stderr)
    elif args.keys_out is not None:
        try:
            summary["sample_keys"] = write_sample_keys(kept, after_mix.built, args.keys_out)
        except CheckError as exc:
            print(f"FAIL: {exc}; no keys file written", file=sys.stderr)
            failures.append(str(exc))
            code = exc.code
        else:
            print(f"sample keys: {len(kept)} written to {args.keys_out} (sha256 "
                  f"{summary['sample_keys']['sha256']}); train.py's load_sample_keys and select_samples "
                  "read back exactly these samples, in this order")
    summary["failures"] = failures
    summary["seconds"] = round(time.perf_counter() - started, 1)
    print(f"time: {summary['seconds']} s")
    print("summary: " + json.dumps(summary, ensure_ascii=False, default=str))
    return code


if __name__ == "__main__":
    sys.exit(main())

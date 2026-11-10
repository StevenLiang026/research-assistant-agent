"""The M7 base-model bake-off (docs/m7-计划.md §4; docs/m7-底座对比预登记.md): one candidate's smoke steps,
pilot training, evaluation and gate JSON, and the aggregate decision over every candidate, as the
preregistration defines them. scripts/bakeoff.py is the command line over this module.

The configuration (configs/bakeoff-m7.json, load_config) holds what every candidate shares, the
preregistered conditions: the data files and the benchmark-overlap report, the development items, the
reasoning sidecars, the schedule (the stop step, the save cadence, the extension step, the maximum length
per thinking kind), the smoke settings, the thresholds, the FT3 estimate's factors, the evaluation variants
per thinking kind with the "{R}" placeholder the reasoning budget fills, the primary variants per kind,
b0's arguments per mode, the reference C0 (its score, G2's record and G2's pilot folder, from which its FT3
estimate is computed as a candidate's is) and the candidates: each with its key, model folder, GGUF,
family name (None until the family is registered: the folder then resolves on its own), thinking kind
(none, only or hybrid), reasoning budget R (None until the dry run computes it; the smoke's S2 records it),
Hugging Face repository and revision, the smoke steps it is exempt from with the reason (C1), and, for a
candidate a cut line of the plan skips (docs/m7-计划.md 砍线 2: wave 4), skipped: the reason. Every path in it
is relative to the repository root and stays inside it.

Every step runs the project's own scripts as subprocesses from the repository root (scripts/train.py,
export_adapter.py, run_eval.py, score.py, pytest over the family's golden tests) through the Tools' runner,
and serves a model through research_agent.llm.server.LocalServer, the command of scripts/serve_local.py with
one slot; the output goes under runs/bakeoff/m7/<candidate>/ (the configuration's run_root), each JSON
record free of timestamps and machine paths (repository-relative POSIX paths and SHA-256s; the commands a
record keeps name the machine's tools by placeholders, recorded: llama-server, $LLAMA_CPP_SRC, python for an
interpreter outside the repository, with the server's SHA-256 and build_info and the converter's SHA-256
beside them), the log bakeoff.log timestamped. Everything the machine provides is injectable (Tools): the
runner, the server's process and client, the family loader, the three preflight probes, the GGUF tensor
counter, the fidelity backends, the clock.

Preflight (ported from the pilot launcher): the GPU holds at most thresholds.gpu_used_mib (nvidia-smi),
at least thresholds.ram_gb of RAM is available and the machine runs on AC power; the pilot waits, polling,
until all three hold or Tools.wait_limit passes (PreflightTimeout).

Smoke S0-S6 (the preregistration's 资格 table), each recorded in smoke.json with pass or fail and the
measured numbers; an exempt step is recorded as such and not run:
  S0  the model folder resolves to the configured family; the GGUF's SHA-256; for a family that runs vendor
      code, the SHA-256 of every .py file of the folder.
  S1  the family's golden tests (tests/family/test_golden.py -k <folder name>): at least one passed, none
      failed.
  S2  scripts/train.py --dry-run at the candidate's maximum length: token-boundary refusals 0, BOS refusals
      0, template refusals at most thresholds.template_rate of the steps rendered; the MODEL_TOOLS target
      round trip (every agent-history target with JSON-object arguments, rendered through the family and
      read back by its tool format) 100%; the reasoning budget R the dry run's rule picks and the tokens per
      epoch are recorded. Template refusals within the bound block the step (neither passed nor failed): the
      preregistration then leaves the refused samples out of every candidate's data, which this tooling
      cannot do (scripts/train.py stops at a refused sample, and its dry run names only the first few), so
      the owner decides; a blocked step keeps the candidate waiting and is never a reason to exclude it.
  S3  scripts/train.py --max-steps smoke.steps --smoke-longest: the peak reserved VRAM at most
      thresholds.peak_reserved_gb; the median tokens/s and the FT3 hour estimate (S2's tokens per epoch x
      ft3.data_factor x ft3.epochs / the median) are recorded; the rows are the last run's (training_rows).
  S4  the smoke adapter exported with scripts/export_adapter.py: the GGUF's tensor count equals 2 x the
      modules the adapter targets (its safetensors' lora_A names), no fewer and no more. Every export (S4's,
      S6's, the pilot's) goes through run_export, which creates the GGUF's folder first: llama.cpp's
      converter creates none (C3's first S4 failed on the missing smoke/train/gguf/).
  S5  llama-server with the base GGUF and the smoke adapter (the pilot's checkpoint when there is none, the
      base alone otherwise), one slot: every request's usage.prompt_tokens equals the number of ids sent; a
      completion that stopped ends with the family's terminator; the same smoke.items development items run
      twice are byte-identical, under the family's primary mode and settings and, for a family that thinks,
      also under the forced budget smoke.forced_budget, where the close must be forced at least once.
  S6  research_agent.eval.fidelity over the family's LoRA targets: the modules of the model as training builds
      it (on the meta device, once per run), every adapter's load checked on the training stack, each
      group's strength calibrated on the training stack before llama.cpp predicts, the failures one line
      per failed group with its reasons.

The pilot trains to schedule.stop_step with the data, sidecars and schedule of the preregistration
(--save-steps and --stop-at-step: checkpoint-<stop>) and exports that checkpoint; with a stop step beyond
an existing run it resumes (the extension to schedule.extend_step), and a pilot stopped part way resumes from
its last saved checkpoint. A run folder that holds a manifest but no saved checkpoint (stopped before its
first save) is reported and left as it is: scripts/train.py --resume refuses it, and a fresh start would
write over its records, so the owner moves it aside. The evaluation serves the checkpoint's GGUF with one
slot and runs scripts/run_eval.py over the development items once per variant of the candidate's thinking
kind (a non-thinking family: nothink, greedy; a thinking family: think-s0..s2 with base seeds 0..2 and
think-greedy; a hybrid family also nothink) and once for the untrained base in the primary mode (b0), then
scores each run; an extension's runs are named dev-<step>-<variant>. Records that stand for a run already
(stale_records: run_eval's manifest names the same checkpoint GGUF, items and settings) are kept rather than
run again, so an interrupted evaluation resumes and a variant added later runs alone; score.py, which is
deterministic and quick, scores every run again from its records; rerun runs everything anew.

The gate JSON of a candidate (gate.json; the shape of the pilot launcher's gate) holds the primary score P
(the greedy run's successes for a non-thinking family, the mean of think-s0..s2 for a thinking one), every
variant's and b0's successes with the handwritten and drafted splits and the per-type counts, the pilot's
VRAM peak, tokens/s median over steps 100..stop and FT3 hour estimate (the last run's log rows,
training_rows; the estimate unrounded beside the one shown to a decimal), the smoke record, the extension's
scores once its primary variants are scored (its P and handwritten mean, every variant's successes and, for a
hybrid family, the nothink greedy run's, which the hybrid rule reads when the extension decides), the files
it was computed from with their SHA-256s, and its status:
  ineligible  a smoke step failed (one the candidate is neither exempt from nor runs for the record only,
              whose input steps passed or are exempt) or the unrounded FT3 estimate is over
              thresholds.ft3_hours: the reasons say which; these are the only reasons the preregistration
              excludes a candidate for (排除只能出于冒烟或时长资格).
  incomplete  nothing excludes it, but something the decision needs is not there yet: a smoke step not run
              or blocked (by its record, or failed while a step whose output it uses had not passed,
              STEP_INPUTS: it waits for that step, docs/m7-底座对比预登记.md 修订 3), the pilot, its
              throughput at steps 100 and after, the scores of a primary variant or of a hybrid family's
              nothink run (the hybrid rule reads it), and the nothink run of a hybrid family's extension
              whose primary variants are scored (修订 3); missing says what.
  scored      eligible and complete.

The aggregate (results/bakeoff-m7.json) applies the rule to every candidate's gate and the reference:
C0 takes part with its G2 score (the configuration's reference, checked against its score file and G2's
record); the eligible candidates rank by P; the highest wins unless the top two are within thresholds.tie,
when both are extended to schedule.extend_step ("extend", the two named, the ones not yet extended
pending); after the extension P decides when more than the tie apart, else the handwritten successes,
else the shorter FT3 estimate (unrounded); a winner that thinks answers in think mode unless it is a hybrid
family whose nothink greedy run leads its think mean by thresholds.hybrid_margin, both read at the step the
decision's basis came from (docs/m7-底座对比预登记.md 修订 3): the extension's, its nothink greedy run
against the mean of its three seeds there, when the extension decided (P, the handwritten successes, or the
hour estimate once both were extended), the stop step's otherwise; a winner trained without thinking (a
non-thinking family, or the reference) answers in no_think mode. A candidate is excluded only by
the smoke steps or the hour estimate, never by its score. The winner comes from every candidate: while one
has no gate or an incomplete one (a gate another tool wrote counts as incomplete), the decision is
"provisional", names no winner and lists those it waits for; the rule's outcome over the others is its
"standing", with their "leader". A skipped candidate takes no part: the results file records it as skipped
with the reason, and the decision does not wait for it; a cut line skips a candidate before its score, so a
skipped candidate whose gate holds a P stops the aggregate (BakeoffError).

The trained adapter's fidelity (docs/m7-底座对比预登记.md 修订 1: S6 is recorded only, and the winner gets
this check instead, report only): the candidate's pilot adapter, trainer/checkpoint-<stop>, or
checkpoint-<extend> once the run was extended (that checkpoint saved, as the gate reads the extension;
trained_step, where --stop-step names either one and no other), compared over S6's sequences (the first
smoke.items development items, smoke.positions positions each, the NO_THINK renders that
eval.fidelity.sequences_of builds): the training stack's top-1 tokens in NF4 without an adapter and with the
checkpoint's PEFT adapter, its load checked as S6 checks its synthetic adapters (AdapterNotLoaded: the
record's error, and no llama-server starts), and llama.cpp's over the candidate's Q8_0 GGUF without an
adapter and with the checkpoint's GGUF (train/gguf/checkpoint-<step>-f16.gguf, the pilot's, exported through
run_export first when missing). S6's arithmetic (eval.fidelity.compare) gives the agreement with the adapter
beside the agreement without it and the share of the training stack's top-1 tokens the adapter changes, kept
without S6's bounds and without a verdict. The record, fidelity-trained-<step>.json (the counts, the files'
repository-relative paths and SHA-256s, the adapter's module and tensor counts, the sequences' SHA-256, the
export when one ran, and the error that kept the comparison from being made, None when it was made), and the
logs beside it are the check's alone: no smoke record, gate, aggregate or eligibility reads them or is
written by it.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import dataclasses
import functools
import glob
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PureWindowsPath
from typing import Protocol

import httpx

from ..agent.context import build_context
from ..agent.types import ToolCall
from ..canonical import canonical_json
from ..family.core import FamilyError, LoadedFamily, Mode
from ..llm.generation import GenerationSettings, generate
from ..llm.local import LocalClient
from ..llm.server import HOST, LocalServer, same_path, server_command
from ..tools.schemas import MODEL_TOOLS
from ..train.samples import is_agent_history, load_records, render_family_sample, step_pairs
from .fidelity import (
    ADAPTER_WEIGHTS,
    AIM,
    AdapterNotLoaded,
    Backends,
    base_modules,
    compare,
    hf_top1_backend,
    llama_top1_backend,
    model_layout,
    read_safetensors_header,
    run_fidelity,
    sequences_of,
    strengths,
    weight_norm,
)
from .fidelity import MODE as SEQUENCES_MODE
from .lock import DevSet, sha256_of

CONFIG = Path("configs/bakeoff-m7.json")
PREREGISTRATION = Path("docs/m7-底座对比预登记.md")
STEPS = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
THINKING = ("none", "only", "hybrid")
BUDGET = "{R}"  # the placeholder of the reasoning budget in a variant's arguments
PLAN_BUDGET = "<R>"  # what a plan prints where the budget is not yet known
SMOKE_FILE, PILOT_FILE, EVALUATE_FILE, GATE_FILE, LOG_FILE = ("smoke.json", "pilot.json", "evaluate.json",
                                                              "gate.json", "bakeoff.log")
TRAINED_FIDELITY = "fidelity-trained"  # the trained adapter's fidelity: <stem>-<step>.json and its logs
# the counts of eval.fidelity.compare the trained adapter's record keeps (its checks and verdict are S6's)
TRAINED_COUNTS = ("items", "positions", "agreeing", "agreement", "base_agreeing", "base_agreement",
                  "changed_positions", "changed")
SCRIPT_ENV = {"HF_HUB_OFFLINE": "1", "TQDM_DISABLE": "1", "PYTHONUTF8": "1"}  # the launchers' environment
GOLDEN_TESTS = "tests/family/test_golden.py"
SUMMARY_PREFIX = "summary: "  # the dry run's last line (scripts/train.py)
GIB_PER_GB = 1.073741824  # the pilot launcher's gate reports the trainer's GiB as GB
LORA_R, LORA_ALPHA = 16, 32  # the preregistration's QLoRA rank and alpha (scripts/train.py's defaults)
FIDELITY_SEED = 7
SRC_VAR = "LLAMA_CPP_SRC"  # scripts/export_adapter.py's variable, which names the converter's tree
CONVERTER = "convert_lora_to_gguf.py"  # in that tree (scripts/export_adapter.py)
SLOTS = 1
NO_THINK_GREEDY = "nothink"  # a hybrid family's nothink greedy run, which the hybrid rule reads
STEP_INPUTS = {"S4": ("S3",), "S5": ("S4",)}  # S4 exports the adapter S3 trains, S5 serves the GGUF S4 writes
GATE_STATUSES = ("scored", "incomplete", "ineligible")
# the placeholders a record names the machine's tools by (recorded)
SERVER_PLACEHOLDER, SRC_PLACEHOLDER, PYTHON_PLACEHOLDER = "llama-server", f"${SRC_VAR}", "python"
_PYTEST_COUNT = re.compile(r"(\d+) (passed|failed|skipped|errors?|warnings?)\b")
# what a smoke step records as its failure: a refusal of this module, a family's, a server's (ServerError is
# a RuntimeError, as is a GenerationError), a file that cannot be read, a malformed record, and a request the
# server answered with an error; anything else is a defect of the tooling and propagates
STEP_ERRORS = (ValueError, RuntimeError, OSError, KeyError, httpx.HTTPError)


class BakeoffError(ValueError):
    """A configuration, a candidate or a machine setting this module cannot work with; the message says
    what."""


class PreflightTimeout(RuntimeError):
    """The machine did not become free within the wait limit."""


# ---- the configuration ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One candidate of the bake-off (module docstring: The configuration)."""

    key: str
    name: str
    hf: str
    gguf: str
    family: str | None
    thinking: str
    reasoning_budget: int | None
    model: str | None
    revision: str | None
    exempt: tuple[str, ...]
    exempt_reason: str | None
    record_only: tuple[str, ...] = ()  # steps run and recorded that never count against eligibility (C1)
    skipped: str | None = None  # the reason a cut line of the plan skips it (wave 4); None: it takes part

    @property
    def thinks(self) -> bool:
        return self.thinking != "none"

    @property
    def primary_mode(self) -> str:
        """The mode of the primary score and of b0: think for a family that thinks, else no_think."""
        return str(Mode.THINK) if self.thinks else str(Mode.NO_THINK)


@dataclass(frozen=True)
class Config:
    """The configuration file, read and checked (load_config)."""

    path: Path
    run_root: Path
    results: Path
    data: tuple[str, ...]
    overlap_report: str
    dev_items: Path
    reasoning: tuple[str, ...]
    schedule: Mapping
    smoke: Mapping
    thresholds: Mapping
    ft3: Mapping
    variants: Mapping
    primary: Mapping
    b0: Mapping
    reference: Mapping
    candidates: Mapping[str, Candidate]

    def candidate(self, key: str) -> Candidate:
        """The candidate by key; BakeoffError for the reference's key or an unknown one."""
        if key == self.reference["key"]:
            raise BakeoffError(f"{key} is the reference, which has its score already (the configuration's "
                               "reference); it is not run")
        if key not in self.candidates:
            raise BakeoffError(f"unknown candidate {key!r}; configured: {', '.join(self.candidates)}")
        return self.candidates[key]

    def folder(self, key: str) -> Path:
        return self.run_root / key

    def max_length(self, candidate: Candidate) -> int:
        return int(self.schedule["max_length"][candidate.thinking])

    def primary_variants(self, candidate: Candidate) -> list[str]:
        return list(self.primary[candidate.thinking]["variants"])


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise BakeoffError(message)


def _inside(value: object) -> bool:
    """Whether value is a path relative to the repository root that stays inside it: no drive, no root,
    no '..' (either separator)."""
    if not isinstance(value, str) or not value:
        return False
    pure = PureWindowsPath(value)
    return not pure.drive and not pure.root and ".." not in pure.parts


def load_config(path: str | Path = CONFIG) -> Config:
    """The configuration at path, checked: the keys, the thinking kinds, the exempt steps, the variants
    the primary lists, a hybrid family's nothink run, the maximum length per kind, that no candidate carries
    the reference's key, and that every path is relative to the repository root and inside it (the records
    hold the paths it gives, and a record holds no machine path)."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in ("run_root", "results", "data", "overlap_report", "dev_items", "reasoning", "schedule",
                "smoke", "thresholds", "ft3", "variants", "primary", "b0", "reference", "candidates"):
        _require(key in raw, f"{path.as_posix()}: no {key!r}")
    _require(set(raw["schedule"].get("max_length", {})) >= set(THINKING),
             f"{path.as_posix()}: schedule.max_length must give a length for each of {THINKING}")
    for kind in THINKING:
        _require(kind in raw["variants"] and kind in raw["primary"],
                 f"{path.as_posix()}: variants and primary must cover the thinking kind {kind!r}")
        missing = [v for v in raw["primary"][kind]["variants"] if v not in raw["variants"][kind]]
        _require(not missing, f"{path.as_posix()}: primary.{kind} names {', '.join(missing)}, not among "
                              f"variants.{kind}")
    _require(NO_THINK_GREEDY in raw["variants"]["hybrid"],
             f"{path.as_posix()}: variants.hybrid must hold {NO_THINK_GREEDY!r}, the run the hybrid rule "
             "reads")
    reference = raw["reference"]
    paths = [("run_root", raw["run_root"]), ("results", raw["results"]),
             ("overlap_report", raw["overlap_report"]), ("dev_items", raw["dev_items"]),
             *(("data", pattern) for pattern in raw["data"]),
             *(("reasoning", file) for file in raw["reasoning"]),
             *((f"reference.{field}", reference[field])
               for field in ("score", "source", "train", "resume_from") if reference.get(field) is not None),
             *((f"candidate {key}: {field}", entry.get(field)) for key, entry in raw["candidates"].items()
               for field in ("hf", "gguf"))]
    for what, value in paths:
        _require(_inside(value), f"{path.as_posix()}: {what} must be a path relative to the repository root "
                                 f"and inside it, got {value!r}")
    candidates: dict[str, Candidate] = {}
    for key, entry in raw["candidates"].items():
        _require(key != reference["key"], f"{path.as_posix()}: candidate {key!r} carries the reference's key")
        thinking = entry.get("thinking")
        _require(thinking in THINKING, f"{path.as_posix()}: candidate {key}: thinking must be one of "
                                       f"{THINKING}, got {thinking!r}")
        exempt = tuple(entry.get("exempt") or ())
        record_only = tuple(entry.get("record_only") or ())
        for field, steps in (("exempt", exempt), ("record_only", record_only)):
            unknown = [s for s in steps if s not in STEPS]
            _require(not unknown, f"{path.as_posix()}: candidate {key}: {field} names {', '.join(unknown)}, "
                                  f"not among {STEPS}")
        budget = entry.get("reasoning_budget")
        _require(budget is None or (isinstance(budget, int) and budget > 0),
                 f"{path.as_posix()}: candidate {key}: reasoning_budget must be a positive integer or null")
        for field in ("name", "hf", "gguf"):
            _require(isinstance(entry.get(field), str) and entry[field],
                     f"{path.as_posix()}: candidate {key}: {field} must be a non-empty string")
        skipped = entry.get("skipped")
        _require(skipped is None or (isinstance(skipped, str) and skipped != ""),
                 f"{path.as_posix()}: candidate {key}: skipped must be a non-empty string (the reason) or "
                 "absent")
        candidates[key] = Candidate(key=key, name=entry["name"], hf=entry["hf"], gguf=entry["gguf"],
                                    family=entry.get("family"), thinking=thinking, reasoning_budget=budget,
                                    model=entry.get("model"), revision=entry.get("revision"), exempt=exempt,
                                    exempt_reason=entry.get("exempt_reason"), record_only=record_only,
                                    skipped=skipped)
    return Config(path=path, run_root=Path(raw["run_root"]), results=Path(raw["results"]),
                  data=tuple(raw["data"]), overlap_report=raw["overlap_report"],
                  dev_items=Path(raw["dev_items"]), reasoning=tuple(raw["reasoning"]),
                  schedule=raw["schedule"], smoke=raw["smoke"], thresholds=raw["thresholds"], ft3=raw["ft3"],
                  variants=raw["variants"], primary=raw["primary"], b0=raw["b0"], reference=reference,
                  candidates=candidates)


def _filled(args: list[str], budget: int | str | None, what: str) -> list[str]:
    """The arguments with the budget placeholder filled; BakeoffError when one is needed and none is
    known."""
    if BUDGET in args:
        if budget is None:
            raise BakeoffError(f"{what} needs the reasoning budget R, which is neither configured "
                               "(reasoning_budget) nor recorded by the smoke's S2")
        return [str(budget) if a == BUDGET else a for a in args]
    return list(args)


def variants_of(config: Config, candidate: Candidate, budget: int | str | None
                ) -> list[tuple[str, list[str]]]:
    """The evaluation variants of the candidate's thinking kind, (name, run_eval arguments), the budget
    filled in (module docstring); BakeoffError when a variant needs a budget and none is known."""
    return [(name, _filled(list(args), budget, f"variant {name}"))
            for name, args in config.variants[candidate.thinking].items()]


def b0_variant(config: Config, candidate: Candidate, budget: int | str | None) -> list[str]:
    """b0's run_eval arguments: the primary mode's, the budget filled in."""
    return _filled(list(config.b0[candidate.primary_mode]), budget, "b0")


def budget_of(config: Config, candidate: Candidate, smoke: Mapping | None) -> int | None:
    """The reasoning budget R: the configured one, else the one the smoke's S2 recorded, else None."""
    if candidate.reasoning_budget is not None:
        return candidate.reasoning_budget
    recorded = ((smoke or {}).get("steps") or {}).get("S2") or {}
    budget = recorded.get("reasoning_budget")
    return budget if isinstance(budget, int) and not isinstance(budget, bool) else None


# ---- the tools --------------------------------------------------------------------------------------------


class Probes(Protocol):
    """The preflight's probes; None when a probe cannot tell."""

    def gpu_used_mib(self) -> int | None: ...

    def available_ram_gb(self) -> float | None: ...

    def on_ac_power(self) -> bool | None: ...


class Process(Protocol):
    """What is used of a started script's process."""

    returncode: int | None

    def poll(self) -> int | None: ...


Runner = Callable[..., subprocess.CompletedProcess]  # run(cmd, *, log, env) -> returncode and stdout
Starter = Callable[..., Process]  # start(cmd, *, log, env) -> the running process
BackendsFactory = Callable[..., Backends]  # (config, candidate, tools, folder, *, server_log) -> Backends


@dataclass
class Tools:
    """What the machine provides, injectable (module docstring)."""

    run: Runner
    start: Starter
    popen: Callable  # LocalServer's process starter
    client: Callable[[str], LocalClient]
    load_family: Callable[[Path, str | None], LoadedFamily]
    probes: Probes
    count_gguf_tensors: Callable[[Path], int]
    fidelity_backends: BackendsFactory
    round_trip: Callable[[LoadedFamily, list[Path]], dict]
    sleep: Callable[[float], None]
    clock: Callable[[], float]
    python: str
    llama_server: Path | None
    llama_cpp_src: Path | None
    wait_interval: float = 60.0
    wait_limit: float = 12 * 3600.0
    poll_interval: float = 30.0
    port: int = 8080
    timeout: float = 120.0


def script_env() -> dict[str, str]:
    """The environment the scripts run in: this process's with the launchers' settings (SCRIPT_ENV)."""
    return {**os.environ, **SCRIPT_ENV}


def run_script(cmd: list[str], *, log: Path, env: Mapping[str, str] | None = None
               ) -> subprocess.CompletedProcess:
    """The default runner: the script's stdout and stderr go to <log>.out.log and <log>.err.log (appended
    to the stem as given, so stems that differ in their last piece alone keep logs of their own), and come
    back in the result as well."""
    out, err = log_files(log)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as o, err.open("w", encoding="utf-8",
                                                                      newline="\n") as e:
        done = subprocess.run(launchable(cmd), stdout=o, stderr=e, env=env, check=False)
    return subprocess.CompletedProcess(cmd, done.returncode,
                                       stdout=out.read_text(encoding="utf-8", errors="replace"),
                                       stderr=err.read_text(encoding="utf-8", errors="replace"))


def start_script(cmd: list[str], *, log: Path, env: Mapping[str, str] | None = None) -> subprocess.Popen:
    """The default starter: like run_script, without waiting (the pilot polls the GPU meanwhile)."""
    out, err = log_files(log)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as o, err.open("w", encoding="utf-8",
                                                                      newline="\n") as e:
        return subprocess.Popen(launchable(cmd), stdout=o, stderr=e, env=env)  # the child keeps its handles


def log_files(log: Path) -> tuple[Path, Path]:
    """The stdout and stderr files of a log stem: <log>.out.log and <log>.err.log."""
    return Path(f"{log}.out.log"), Path(f"{log}.err.log")


def launchable(cmd: list[str]) -> list[str]:
    """The command with its executable as an absolute native path when it names a path: the records keep
    the interpreter relative to the repository root in POSIX form (Tools.python), which Windows'
    CreateProcess does not launch (a relative path with forward slashes is not found); a bare name is left
    to the PATH search."""
    first = cmd[0]
    if "/" in first or "\\" in first:
        return [os.path.abspath(first), *cmd[1:]]
    return list(cmd)


class MachineProbes:
    """The preflight's probes of this machine: nvidia-smi for the GPU, Windows' GlobalMemoryStatusEx for the
    RAM and GetSystemPowerStatus for the power source (ctypes); None wherever the call fails or the system
    has no such call."""

    def gpu_used_mib(self) -> int | None:
        try:
            done = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=30, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        first = done.stdout.strip().splitlines()[:1]
        if done.returncode != 0 or not first or not first[0].strip().isdigit():
            return None
        return int(first[0].strip())

    def available_ram_gb(self) -> float | None:
        kernel = getattr(ctypes, "windll", None)
        if kernel is None:
            return None

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_uint32), ("dwMemoryLoad", ctypes.c_uint32),
                        ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
                        ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
                        ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
                        ("ullAvailExtendedVirtual", ctypes.c_uint64)]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if not kernel.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return round(status.ullAvailPhys / 2**30, 2)

    def on_ac_power(self) -> bool | None:
        kernel = getattr(ctypes, "windll", None)
        if kernel is None:
            return None

        class PowerStatus(ctypes.Structure):
            _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                        ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                        ("BatteryLifeTime", ctypes.c_uint32), ("BatteryFullLifeTime", ctypes.c_uint32)]

        status = PowerStatus()
        if not kernel.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
            return None
        return {0: False, 1: True}.get(status.ACLineStatus)  # 255: unknown


def gguf_tensor_count(path: Path, src: Path | None) -> int:
    """The number of tensors of a GGUF file, read with gguf-py from the llama.cpp source tree."""
    if src is not None and str(src / "gguf-py") not in sys.path:
        sys.path.insert(0, str(src / "gguf-py"))
    from gguf import GGUFReader  # llama.cpp's package, outside the repository

    return len(GGUFReader(str(path)).tensors)


def default_fidelity_backends(config: Config, candidate: Candidate, tools: Tools, folder: Path, *,
                              server_log: Path | None = None) -> Backends:
    """S6's backends on this machine: the base modules and weight norms of the model as training builds it
    (research_agent.eval.fidelity.model_layout, the model instantiated on the meta device once, when first
    asked), the export through scripts/export_adapter.py (run_export), the training stack on the GPU,
    llama-server through the Tools, logging to server_log (S6's smoke/S6.server.log when None; the trained
    adapter's fidelity names its own)."""
    hf = Path(candidate.hf)
    family = tools.load_family(hf, candidate.family)
    binary = require_server(tools)
    layout = functools.cache(lambda: model_layout(hf))

    def export(adapter: Path, outfile: Path) -> Path:
        cmd = export_command(config, candidate, tools, adapter, outfile)
        done = run_export(tools, cmd, outfile, log=folder / "smoke" / f"S6.export.{adapter.parent.name}")
        if done.returncode != 0 or not outfile.is_file():
            raise BakeoffError(f"the export of {rel(adapter)} exited with {done.returncode}"
                               + ("" if outfile.is_file() else f" and wrote no {rel(outfile)}"))
        return outfile

    return Backends(modules=lambda target: base_modules(hf, target, layout=layout()),
                    weight_norm=lambda module: weight_norm(hf, module, layout=layout()), export=export,
                    hf_top1=hf_top1_backend(hf, trust_remote_code=family.spec.trust_remote_code),
                    llama_top1=llama_top1_backend(binary, Path(candidate.gguf), port=tools.port,
                                                  popen=tools.popen, client=tools.client,
                                                  timeout=tools.timeout,
                                                  log=server_log or folder / "smoke" / "S6.server.log"))


def require_server(tools: Tools) -> Path:
    if tools.llama_server is None:
        raise BakeoffError("no llama-server given: pass --llama-server or set LLAMA_SERVER")
    return tools.llama_server


# ---- paths, files, the log --------------------------------------------------------------------------------


def rel(path: str | Path) -> str:
    """path relative to the current folder (the repository root) in POSIX form when it lies inside it, else
    absolute (POSIX)."""
    absolute = Path(os.path.abspath(path))
    try:
        return absolute.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return absolute.as_posix()


def read_json(path: Path) -> dict | None:
    """The JSON object at path, None when there is no file."""
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def write_json(path: Path, value: object) -> None:
    """Two-space indented JSON, LF, a final newline: byte-identical for the same content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def read_rows(path: Path) -> list[dict]:
    """The JSON objects of a JSON Lines file, [] without a file."""
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def training_rows(path: Path) -> list[dict]:
    """The logging rows of a training run's log.jsonl as scripts/train.py counts them (segments_from_log):
    the run's rows start at its last {"event": "start"} (an earlier attempt in the same folder left the rows
    before it), and after a {"event": "resume", "global_step": N} the earlier segment counts only up to step
    N (its later steps were trained again); [] without a file."""
    rows = read_rows(path)
    starts = [i for i, row in enumerate(rows) if row.get("event") == "start"]
    segments: list[list[dict]] = [[]]
    resumed_at: list[int] = []
    for row in rows[starts[-1] + 1:] if starts else rows:
        if row.get("event") == "resume":
            resumed_at.append(int(row["global_step"]))
            segments.append([])
        elif "event" not in row:
            segments[-1].append(row)
    kept = []
    for k, segment in enumerate(segments):
        end = resumed_at[k] if k < len(resumed_at) else None
        kept += [row for row in segment if end is None or (row.get("step") or 0) <= end]
    return kept


def throughput_median(rows: list[dict], stop: int) -> float | None:
    """The median tokens/s of the logging rows at steps 100..stop (the preregistration's FT3 throughput);
    None without such a row."""
    late = [r["tokens_per_s"] for r in rows if r.get("tokens_per_s") and 100 <= (r.get("step") or 0) <= stop]
    return statistics.median(late) if late else None


def sha256_if_file(path: str | Path | None) -> str | None:
    """The SHA-256 of a file, None when there is none (a tool named by name and found on the PATH)."""
    return sha256_of(path) if path is not None and Path(path).is_file() else None


def recorded(cmd: list[str], tools: Tools) -> list[str]:
    """The command as a record keeps it: the machine's tools named by placeholders instead of their paths
    (the llama-server binary SERVER_PLACEHOLDER, the llama.cpp source tree SRC_PLACEHOLDER, an interpreter
    outside the repository PYTHON_PLACEHOLDER), so that no record holds a machine path; the records keep the
    tools' SHA-256s and the server's build_info beside the command instead. Everything else of a command is
    the configuration's repository-relative paths and settings."""
    machine: dict[str, str] = {}
    for path, placeholder in ((tools.llama_server, SERVER_PLACEHOLDER),
                              (tools.llama_cpp_src, SRC_PLACEHOLDER)):
        if path is not None:
            machine |= {str(path): placeholder, Path(path).as_posix(): placeholder}
    if not _inside(tools.python):
        machine[tools.python] = PYTHON_PLACEHOLDER
    return [machine.get(piece, piece) for piece in cmd]


class Log:
    """Timestamped lines to a file, echoed to stdout."""

    def __init__(self, path: Path | None) -> None:
        self.path = path

    def __call__(self, text: str) -> None:
        print(text, flush=True)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as f:
                f.write(f"{time.strftime('%m-%d %H:%M:%S')} {text}\n")


def data_files(config: Config) -> list[Path]:
    """The data files, the configuration's patterns expanded as scripts/train.py expands them."""
    files: list[Path] = []
    for pattern in config.data:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise BakeoffError(f"no data file matches {pattern!r}")
        files += [Path(m) for m in matches]
    return files


def dev_items(config: Config) -> list[dict]:
    """The development items, lock-checked (research_agent.eval.lock.DevSet)."""
    return DevSet(config.dev_items).open()


def cmdline(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd)


# ---- the commands -----------------------------------------------------------------------------------------


def train_command(config: Config, candidate: Candidate, tools: Tools, *, out: Path,
                  stop_step: int | None = None, max_steps: int | None = None, smoke_longest: bool = False,
                  dry_run: bool = False, resume: bool = False) -> list[str]:
    """scripts/train.py for the candidate under the preregistered conditions: its model folder and family,
    the data files, the overlap report, QLoRA, the maximum length of its thinking kind, the sidecars when it
    thinks; --save-steps and --stop-at-step for a pilot, --max-steps and --smoke-longest for the memory
    smoke, --dry-run for S2, --resume for the extension."""
    cmd = [tools.python, "scripts/train.py", *config.data, "--model", Path(candidate.hf).as_posix(), "--out",
           Path(out).as_posix(), "--max-length", str(config.max_length(candidate)), "--overlap-report",
           config.overlap_report]
    if candidate.family:
        cmd += ["--family", candidate.family]
    if config.schedule.get("qlora", True):
        cmd.append("--qlora")
    if "seed" in config.schedule:
        cmd += ["--seed", str(config.schedule["seed"])]
    if candidate.thinks and config.reasoning:
        cmd += ["--reasoning", *config.reasoning]
    if stop_step is not None:
        cmd += ["--save-steps", str(config.schedule["save_steps"]), "--stop-at-step", str(stop_step)]
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    if smoke_longest:
        cmd.append("--smoke-longest")
    if dry_run:
        cmd.append("--dry-run")
    if resume:
        cmd.append("--resume")
    return cmd


def export_command(config: Config, candidate: Candidate, tools: Tools, adapter: Path,
                   outfile: Path) -> list[str]:
    """scripts/export_adapter.py of a PEFT adapter against the candidate's own folder."""
    cmd = [tools.python, "scripts/export_adapter.py", Path(adapter).as_posix(), "--base",
           Path(candidate.hf).as_posix(), "--outfile", Path(outfile).as_posix()]
    if tools.llama_cpp_src is not None:
        cmd += ["--llama-cpp-src", Path(tools.llama_cpp_src).as_posix()]
    return cmd


def run_export(tools: Tools, cmd: list[str], outfile: Path, *, log: Path) -> subprocess.CompletedProcess:
    """An export command (export_command) through the Tools' runner, its output made ready first: the
    folder of outfile is created, since llama.cpp's converter opens the GGUF without creating its folder
    (C3's first S4: FileNotFoundError on smoke/train/gguf/adapter-f16.gguf), and a file an earlier run left
    at outfile is removed, so that a GGUF found there afterwards is this export's (the callers take an exit
    code of 0 without the file for a failure)."""
    outfile = Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.unlink(missing_ok=True)
    return tools.run(cmd, log=log, env=script_env())


def eval_command(config: Config, candidate: Candidate, tools: Tools, *, system: str, out: Path,
                 args: list[str]) -> list[str]:
    """scripts/run_eval.py over the development items on the server, with the signals, for one variant."""
    return [tools.python, "scripts/run_eval.py", config.dev_items.as_posix(), "--system", system,
            "--tokenizer", Path(candidate.hf).as_posix(), "--signals", "--out", Path(out).as_posix(),
            "--server", f"http://{HOST}:{tools.port}", *args]


def score_command(config: Config, tools: Tools, records: Path, out: Path) -> list[str]:
    return [tools.python, "scripts/score.py", Path(records).as_posix(), "--items",
            config.dev_items.as_posix(), "--out", Path(out).as_posix()]


def golden_command(candidate: Candidate, tools: Tools) -> list[str]:
    """The family's golden tests, selected by the model folder's name (their parameter id)."""
    return [tools.python, "-m", "pytest", GOLDEN_TESTS, "-q", "-p", "no:cacheprovider", "-k", candidate.name]


def serve_args(candidate: Candidate, adapter: Path | None, tools: Tools) -> dict:
    """LocalServer's keyword arguments for the candidate's GGUF with an adapter, one slot."""
    return {"model": Path(candidate.gguf).as_posix(), "adapter": None if adapter is None else rel(adapter),
            "port": tools.port, "timeout": tools.timeout, "popen": tools.popen, "client": tools.client,
            "extra": ["-np", str(SLOTS)]}


# ---- the preflight ----------------------------------------------------------------------------------------


def preflight(probes: Probes, thresholds: Mapping) -> dict:
    """The three probes against the thresholds (module docstring: Preflight): the values, ok, and a reason
    per probe that does not hold or cannot tell."""
    gpu, ram, ac = probes.gpu_used_mib(), probes.available_ram_gb(), probes.on_ac_power()
    reasons = []
    if gpu is None:
        reasons.append("the GPU memory in use is unknown (nvidia-smi gave no number)")
    elif gpu > thresholds["gpu_used_mib"]:
        reasons.append(f"the GPU holds {gpu} MiB, over {thresholds['gpu_used_mib']} MiB: another process "
                       "uses it")
    if ram is None:
        reasons.append("the available RAM is unknown")
    elif ram < thresholds["ram_gb"]:
        reasons.append(f"the available RAM is {ram} GB, under {thresholds['ram_gb']} GB")
    if ac is None:
        reasons.append("the power source is unknown")
    elif not ac:
        reasons.append("the machine runs on battery, not on AC power")
    return {"gpu_used_mib": gpu, "available_ram_gb": ram, "on_ac_power": ac, "ok": not reasons,
            "reasons": reasons}


def wait_until_clear(tools: Tools, thresholds: Mapping, *, log: Log | None = None) -> dict:
    """The preflight state once it holds, polling every Tools.wait_interval seconds; PreflightTimeout when
    Tools.wait_limit seconds pass first."""
    started = tools.clock()
    while True:
        state = preflight(tools.probes, thresholds)
        if state["ok"]:
            return state
        if tools.clock() - started >= tools.wait_limit:
            raise PreflightTimeout(f"the machine was not free within {tools.wait_limit:g} s: "
                                   f"{'; '.join(state['reasons'])}")
        if log is not None:
            log(f"waiting: {'; '.join(state['reasons'])}")
        tools.sleep(tools.wait_interval)


# ---- the smoke steps --------------------------------------------------------------------------------------


def parse_summary(stdout: str) -> dict:
    """The JSON of the dry run's "summary:" line, its last line; ValueError without one."""
    for line in reversed(stdout.splitlines()):
        if line.startswith(SUMMARY_PREFIX):
            return json.loads(line[len(SUMMARY_PREFIX):])
    raise ValueError("the dry run printed no summary line")


def parse_pytest(stdout: str) -> dict[str, int]:
    """pytest's counts from its last summary line: passed, failed, skipped, errors (0 when absent)."""
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for line in reversed(stdout.splitlines()):
        found = _PYTEST_COUNT.findall(line)
        if not found:
            continue
        for number, kind in found:
            key = "errors" if kind.startswith("error") else kind
            if key in counts:
                counts[key] = int(number)
        break
    return counts


def tool_round_trip(family: LoadedFamily, files: list[Path]) -> dict:
    """The MODEL_TOOLS target round trip (module docstring, S2) over the agent-history records of the data
    files: every assistant step whose calls all carry JSON-object arguments is rendered through the family
    (its NO_THINK completion) and read back with its tool format; a step a call of which carries raw text
    arguments (a repair sample) is counted apart, as is a step the family refuses to render (the dry run
    counts those). The rate is round trips over targets, 1.0 without a target."""
    counts: Counter[str] = Counter()
    for path in files:
        for record in load_records(path):
            counts["records"] += 1
            if not is_agent_history(record):
                continue
            steps, tools, _ = step_pairs(record, MODEL_TOOLS)
            for _, context, target in steps:
                counts["steps"] += 1
                calls = [ToolCall.from_api(c) for c in target.get("tool_calls") or []]
                if not calls:
                    continue
                if any(c.arguments is None for c in calls):
                    counts["raw_arguments"] += 1
                    continue
                try:
                    sample = render_family_sample(family, context, target, tools=tools)
                except FamilyError:
                    counts["refused"] += 1
                    continue
                counts["targets"] += 1
                read = family.parse(sample["completion"].removesuffix(family.spec.turn_end), tools)
                if read == [{"name": c.name, "arguments": c.arguments} for c in calls]:
                    counts["round_trips"] += 1
    targets = counts["targets"]
    return {"targets": targets, "round_trips": counts["round_trips"],
            "raw_arguments": counts["raw_arguments"], "refused": counts["refused"],
            "records": counts["records"], "steps": counts["steps"],
            "rate": counts["round_trips"] / targets if targets else 1.0}


def ft3_hours(config: Config, tokens_per_epoch: int, tokens_per_s: float) -> float:
    """The preregistration's FT3 estimate in hours: tokens per epoch x ft3.data_factor x ft3.epochs / tokens
    per second / 3600, unrounded: the 60-hour bound and the tie-break compare it as it is (60.04 h is over
    60 h, although it shows as 60.0); the records show it to one decimal (shown) beside it."""
    return tokens_per_epoch * config.ft3["data_factor"] * config.ft3["epochs"] / tokens_per_s / 3600


def shown(hours: float | None) -> float | None:
    """An hour estimate as the records show it, to one decimal (the pilot launcher's gate)."""
    return None if hours is None else round(hours, 1)


def adapter_modules(adapter: Path) -> dict:
    """What a PEFT adapter targets: the modules (one per lora_A tensor), its tensor count and
    target_modules."""
    header = read_safetensors_header(adapter / "adapter_model.safetensors")
    modules = {name.rsplit(".lora_", 1)[0] for name in header if ".lora_A." in name or ".lora_B." in name}
    config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    return {"modules": len(modules), "tensors": len(header), "target_modules": config.get("target_modules")}


def _failed_run(what: str, cmd: list[str], done: subprocess.CompletedProcess, log: Path) -> dict:
    error = f"{what} exited with {done.returncode} (its output: {rel(log)}.out.log, {rel(log)}.err.log)"
    return {"passed": False, "error": error, "returncode": done.returncode, "command": cmd}


def step_s0(config: Config, candidate: Candidate, tools: Tools, folder: Path, smoke: dict, log: Log) -> dict:
    hf = Path(candidate.hf)
    if not hf.is_dir():
        raise BakeoffError(f"no model folder at {candidate.hf}")
    family = tools.load_family(hf, candidate.family)
    if candidate.family and family.spec.name != candidate.family:
        raise BakeoffError(f"{candidate.hf} resolves to the family {family.spec.name}, not the configured "
                           f"{candidate.family}")
    gguf = Path(candidate.gguf)
    if not gguf.is_file():
        raise BakeoffError(f"the GGUF {candidate.gguf} is not here")
    remote = None
    if family.spec.trust_remote_code:
        remote = {p.name: sha256_of(p) for p in sorted(hf.glob("*.py"))}
    return {"passed": True, "hf": rel(hf), "family": family.record(), "gguf": rel(gguf),
            "gguf_sha256": sha256_of(gguf), "remote_code_sha256": remote}


def step_s1(config: Config, candidate: Candidate, tools: Tools, folder: Path, smoke: dict, log: Log) -> dict:
    cmd = golden_command(candidate, tools)
    done = tools.run(cmd, log=folder / "smoke" / "S1.golden", env=script_env())
    counts = parse_pytest(done.stdout)
    failures = []
    if done.returncode != 0:
        failures.append(f"pytest exited with {done.returncode}")
    if counts["failed"] or counts["errors"]:
        failures.append(f"{counts['failed']} golden test(s) failed, {counts['errors']} error(s)")
    if counts["passed"] == 0:
        failures.append(f"no golden test passed for {candidate.name}: none is registered for the folder, or "
                        "all were skipped")
    return {"passed": not failures, "failures": failures, "command": recorded(cmd, tools),
            "returncode": done.returncode, "counts": counts}


def step_s2(config: Config, candidate: Candidate, tools: Tools, folder: Path, smoke: dict, log: Log) -> dict:
    out = folder / "smoke" / "dryrun"
    cmd = train_command(config, candidate, tools, out=out, dry_run=True)
    log_stem = folder / "smoke" / "S2.dryrun"
    done = tools.run(cmd, log=log_stem, env=script_env())
    if done.returncode != 0:
        return _failed_run("the dry run", recorded(cmd, tools), done, log_stem)
    summary = parse_summary(done.stdout)
    refused, samples = summary["refused"], summary["samples"]
    steps = samples + sum(refused.values())
    template_rate = refused["template"] / steps if steps else 0.0
    family = tools.load_family(Path(candidate.hf), candidate.family)
    round_trip = tools.round_trip(family, data_files(config))
    limit = config.thresholds["template_rate"]
    share = f"{refused['template']} of {steps} steps, {template_rate:.2%}"
    failures = []
    if refused["boundary"]:
        failures.append(f"token boundary refusals: {refused['boundary']} (the bound is 0)")
    if refused["bos"]:
        failures.append(f"bos refusals: {refused['bos']} (the bound is 0)")
    if template_rate > limit:
        failures.append(f"template refusals: {share}, over the preregistered {limit:.1%}")
    if round_trip["rate"] < 1.0:
        failures.append(f"MODEL_TOOLS round trip: {round_trip['round_trips']} of {round_trip['targets']} "
                        f"targets ({round_trip['rate']:.1%}), not 100%")
    reasoning = summary.get("reasoning") or {}
    result = {"passed": not failures, "failures": failures, "command": recorded(cmd, tools), "returncode": 0,
              "samples": samples, "tokens_per_epoch": summary["tokens"]["total"],
              "dropped": summary["dropped"], "modes": summary["modes"], "refused": refused, "steps": steps,
              "template_rate": template_rate, "reasoning_budget": reasoning.get("budget"),
              "reasoning_tokens": reasoning.get("tokens"),
              "total_optimizer_steps": summary.get("total_optimizer_steps"),
              "save_cadence": summary.get("save_cadence"), "family": summary.get("family"),
              "round_trip": round_trip}
    if not failures and refused["template"]:  # within the bound (module docstring: S2)
        result["passed"] = None
        result["blocked"] = (
            f"template refusals: {share}, within the preregistered {limit:.1%}; the preregistration then "
            "leaves the refused samples out of every candidate's data, which this tooling cannot do: "
            "scripts/train.py stops at a refused sample, and its dry run names only the first few "
            f"({rel(log_stem)}.out.log). The owner decides how to go on (docs/m7-底座对比预登记.md, S2)")
    return result


def step_s3(config: Config, candidate: Candidate, tools: Tools, folder: Path, smoke: dict, log: Log) -> dict:
    out = folder / "smoke" / "train"
    steps = int(config.smoke["steps"])
    cmd = train_command(config, candidate, tools, out=out, max_steps=steps, smoke_longest=True)
    log_stem = folder / "smoke" / "S3.train"
    done = tools.run(cmd, log=log_stem, env=script_env())
    if done.returncode != 0:
        return _failed_run("the smoke training", recorded(cmd, tools), done, log_stem)
    rows = training_rows(out / "log.jsonl")
    result = read_json(out / "result.json") or {}
    peak_gib = max((r.get("peak_reserved_gib") or 0 for r in rows), default=0)
    peak_gb = result.get("peak_reserved_gb")
    if peak_gb is None:
        peak_gb = round(peak_gib * GIB_PER_GB, 2)
    speeds = [r["tokens_per_s"] for r in rows if r.get("tokens_per_s")]
    median = statistics.median(speeds) if speeds else None
    tokens = (smoke.get("steps", {}).get("S2") or {}).get("tokens_per_epoch")
    hours = ft3_hours(config, tokens, median) if tokens and median else None
    limit = config.thresholds["peak_reserved_gb"]
    failures = []
    if peak_gb > limit:
        failures.append(f"peak reserved VRAM {peak_gb} GB, over {limit} GB")
    return {"passed": not failures, "failures": failures, "command": recorded(cmd, tools), "returncode": 0,
            "train": rel(out), "steps": steps, "peak_reserved_gb": peak_gb, "peak_reserved_gib": peak_gib,
            "tokens_per_s_median": median, "tokens_per_s": speeds, "tokens_per_epoch": tokens,
            "ft3_hours_estimate": shown(hours)}


def step_s4(config: Config, candidate: Candidate, tools: Tools, folder: Path, smoke: dict, log: Log) -> dict:
    adapter = folder / "smoke" / "train" / "adapter"
    if not (adapter / "adapter_config.json").is_file():
        raise BakeoffError(f"no smoke adapter at {rel(adapter)}: S3 writes it")
    outfile = folder / "smoke" / "train" / "gguf" / "adapter-f16.gguf"
    cmd = export_command(config, candidate, tools, adapter, outfile)
    log_stem = folder / "smoke" / "S4.export"
    done = run_export(tools, cmd, outfile, log=log_stem)
    if done.returncode != 0 or not outfile.is_file():
        failed = _failed_run("the export", recorded(cmd, tools), done, log_stem)
        if done.returncode == 0:
            failed["error"] = f"the export exited with 0 but wrote no {rel(outfile)}"
        return failed
    counts = adapter_modules(adapter)
    tensors = tools.count_gguf_tensors(Path(rel(outfile)))
    expected = 2 * counts["modules"]
    failures = []
    if tensors != expected:
        failures.append(f"the GGUF holds {tensors} tensors, not 2 x {counts['modules']} targeted modules = "
                        f"{expected}: llama.cpp dropped or added tensors")
    converter = None if tools.llama_cpp_src is None else Path(tools.llama_cpp_src) / CONVERTER
    return {"passed": not failures, "failures": failures, "command": recorded(cmd, tools), "returncode": 0,
            "converter_sha256": sha256_if_file(converter), "adapter": rel(adapter), "gguf": rel(outfile),
            "gguf_sha256": sha256_of(outfile), "tensors": tensors, "modules": counts["modules"],
            "expected_tensors": expected, "target_modules": counts["target_modules"],
            "adapter_tensors": counts["tensors"]}


def smoke_adapter(config: Config, candidate: Candidate, folder: Path) -> Path | None:
    """The adapter S5 serves: the smoke adapter's GGUF (S4), else the pilot's checkpoint GGUF, else none."""
    for path in (folder / "smoke" / "train" / "gguf" / "adapter-f16.gguf",
                 folder / "train" / "gguf" / f"checkpoint-{config.schedule['stop_step']}-f16.gguf"):
        if path.is_file():
            return path
    return None


class _Recording:
    """A client that keeps, per request, the number of ids sent and the prompt_tokens the server reported."""

    def __init__(self, client: LocalClient) -> None:
        self.client = client
        self.requests: list[tuple[int, int | None]] = []

    def complete(self, prompt, **kwargs):
        completion = self.client.complete(prompt, **kwargs)
        usage = completion.raw.get("usage") or {}
        self.requests.append((len(prompt), usage.get("prompt_tokens")))
        return completion


def _generation_record(gen) -> dict:
    """What two runs of S5 must agree on, byte for byte."""
    return {"reasoning": gen.reasoning, "answer_text": gen.answer_text,
            "answer_ids": list(gen.answer.token_ids), "finish_reason": gen.answer.finish_reason,
            "forced": gen.forced, "phases": gen.phases, "usage": gen.usage, "seeds": gen.seeds}


def step_s5(config: Config, candidate: Candidate, tools: Tools, folder: Path, smoke: dict, log: Log) -> dict:
    binary = require_server(tools)
    family = tools.load_family(Path(candidate.hf), candidate.family)
    items = dev_items(config)[:int(config.smoke["items"])]
    adapter = smoke_adapter(config, candidate, folder)
    scale = 1.0 if adapter is not None else 0.0
    mode = Mode(candidate.primary_mode)
    budget = budget_of(config, candidate, smoke)
    prompts = [family.encode_prompt(family.prompt(family.to_template_messages(build_context(i["messages"])),
                                                  MODEL_TOOLS, mode)) for i in items]
    primary = GenerationSettings.for_family(family, mode=mode, reasoning_budget=budget, base_seed=0)
    settings = {"primary": primary}
    if candidate.thinks:
        settings["forced"] = dataclasses.replace(primary, reasoning_budget=int(config.smoke["forced_budget"]))
    terminator = {"stopped": 0, "at_terminator": 0, "cut": 0}
    runs: dict[str, dict] = {}
    with LocalServer(binary.as_posix(), log=folder / "smoke" / "S5.server.log",
                     **serve_args(candidate, adapter, tools)) as server:
        listed = [a.get("path") if isinstance(a, dict) else a for a in server.adapters]
        expected = [] if adapter is None else [rel(adapter)]
        if len(listed) != len(expected) or any(not same_path(a, b) for a, b in zip(listed, expected,
                                                                                    strict=True)):
            raise BakeoffError(f"the server lists the adapters {listed}, not exactly {expected}")
        slots = (server.props or {}).get("total_slots")
        if slots != SLOTS:
            raise BakeoffError(f"the server reports {slots} slots (/props total_slots), not {SLOTS}")
        build = (server.props or {}).get("build_info")
        recording = _Recording(server.client)
        for name, generation in settings.items():
            passes = []
            for _ in range(2):
                records = []
                for ids in prompts:
                    gen = generate(recording, family, ids, generation, lora_scale=scale)
                    records.append(_generation_record(gen))
                    if gen.answer.finish_reason == "stop":
                        terminator["stopped"] += 1
                        ends = gen.answer.token_ids
                        if ends and ends[-1] == family.turn_end_id:
                            terminator["at_terminator"] += 1
                    elif gen.answer.finish_reason == "length":
                        terminator["cut"] += 1
                passes.append(records)
            runs[name] = {"reasoning_budget": generation.reasoning_budget,
                          "identical": canonical_json(passes[0]) == canonical_json(passes[1]),
                          "forced": sum(r["forced"] for records in passes for r in records),
                          "phases": sum(r["phases"] for records in passes for r in records)}
            log(f"S5 {name}: budget {generation.reasoning_budget}, identical {runs[name]['identical']}, "
                f"forced {runs[name]['forced']}")
    mismatches = sum(1 for sent, reported in recording.requests if reported != sent)
    failures = []
    if mismatches:
        failures.append(f"usage.prompt_tokens differs from the number of ids sent in {mismatches} of "
                        f"{len(recording.requests)} requests")
    if terminator["at_terminator"] != terminator["stopped"]:
        failures.append(f"{terminator['stopped'] - terminator['at_terminator']} of {terminator['stopped']} "
                        "completions stopped without ending at the terminator")
    if terminator["stopped"] == 0:
        failures.append("no completion stopped at the terminator")
    for name, run in runs.items():
        if not run["identical"]:
            failures.append(f"the {name} run is not byte-identical on its repeat")
    if candidate.thinks and runs["forced"]["forced"] == 0:
        failures.append(f"no close was forced under the budget {config.smoke['forced_budget']}")
    return {"passed": not failures, "failures": failures, "items": len(items), "mode": str(mode),
            "adapter": None if adapter is None else rel(adapter), "lora_scale": scale,
            "server": {"slots": slots, "adapters": listed, "build": build,
                       "binary_sha256": sha256_if_file(binary)},
            "prompt_tokens": {"requests": len(recording.requests), "mismatches": mismatches},
            "terminator": terminator, "runs": runs}


def step_s6(config: Config, candidate: Candidate, tools: Tools, folder: Path, smoke: dict, log: Log) -> dict:
    family = tools.load_family(Path(candidate.hf), candidate.family)
    items = dev_items(config)[:int(config.smoke["items"])]
    sequences = sequences_of(family, items, positions=int(config.smoke["positions"]))
    backends = tools.fidelity_backends(config, candidate, tools, folder)
    report = run_fidelity(family.spec.lora_targets, sequences, backends, folder / "smoke" / "fidelity",
                          r=LORA_R, alpha=LORA_ALPHA, seed=FIDELITY_SEED, base_model=candidate.hf)
    failures = [f"group {name}: " + ", ".join(report["groups"][name]["failures"])
                for name in report["failed"]]
    return {"passed": report["passed"], "failures": failures, **report}


STEP_FUNCTIONS = {"S0": step_s0, "S1": step_s1, "S2": step_s2, "S3": step_s3, "S4": step_s4, "S5": step_s5,
                  "S6": step_s6}


def run_smoke(config: Config, candidate: Candidate, tools: Tools, *, steps: list[str] | None = None,
              log: Log | None = None) -> dict:
    """The smoke steps of a candidate (module docstring), all of them or the ones named, each recorded in
    smoke.json as soon as it ends, the earlier records kept; an exempt step is recorded and not run; a step's
    STEP_ERRORS are its failure. The record: candidate, steps, order, failed, blocked and passed (None until
    every step has a record and none is blocked)."""
    wanted = list(STEPS) if steps is None else list(steps)
    unknown = [s for s in wanted if s not in STEPS]
    if unknown:
        raise BakeoffError(f"unknown smoke step(s) {', '.join(unknown)}; the steps are {', '.join(STEPS)}")
    folder = config.folder(candidate.key)
    folder.mkdir(parents=True, exist_ok=True)
    log = log or Log(folder / LOG_FILE)
    smoke = read_json(folder / SMOKE_FILE) or {}
    smoke = {"candidate": candidate.key, "steps": dict(smoke.get("steps") or {})}
    _finish_smoke(smoke)
    for step in wanted:
        if step in candidate.exempt:
            result = {"exempt": True, "passed": None, "reason": candidate.exempt_reason}
            log(f"{step}: exempt ({candidate.exempt_reason})")
        else:
            log(f"{step}: running")
            try:
                result = STEP_FUNCTIONS[step](config, candidate, tools, folder, smoke, log)
            except STEP_ERRORS as exc:
                result = {"passed": False, "error": f"{type(exc).__name__}: {exc}"}
            verdict = "BLOCKED" if result.get("blocked") else "passed" if result["passed"] else "FAILED"
            detail = result.get("blocked") or result.get("error") or "; ".join(result.get("failures") or [])
            log(f"{step}: {verdict}" + (f": {detail}" if detail else ""))
        smoke["steps"][step] = result
        _finish_smoke(smoke)
        write_json(folder / SMOKE_FILE, smoke)
    return smoke


def _finish_smoke(smoke: dict) -> None:
    done = [s for s in STEPS if s in smoke["steps"]]
    smoke["order"] = list(STEPS)
    smoke["failed"] = [s for s in done if smoke["steps"][s].get("passed") is False]
    smoke["blocked"] = [s for s in done if smoke["steps"][s].get("blocked")]
    smoke["passed"] = None if set(done) != set(STEPS) or smoke["blocked"] else not smoke["failed"]


def smoke_plan(smoke: Mapping | None, candidate: Candidate) -> list[str]:
    """The smoke steps scripts/bakeoff.py all runs: every step without a record that holds (passed, or
    exempt for a step the candidate is exempt from), and every step after one of those that uses its output
    (STEP_INPUTS: S4 exports the adapter S3 trains, S5 serves the GGUF S4 writes); the steps that passed
    stand, so that all goes on where an earlier run of it stopped."""
    steps = (smoke or {}).get("steps") or {}
    todo: list[str] = []
    for step in STEPS:
        record = steps.get(step) or {}
        holds = record.get("passed") is True or (step in candidate.exempt and bool(record.get("exempt")))
        if not holds or any(source in todo for source in STEP_INPUTS.get(step, ())):
            todo.append(step)
    return todo


# ---- the pilot --------------------------------------------------------------------------------------------


def _stage_name(config: Config, stop: int, base: str) -> str:
    """A record's file name: <base>.json at the preregistered stop step, <base>-<stop>.json for another."""
    return f"{base}.json" if stop == config.schedule["stop_step"] else f"{base}-{stop}.json"


def saved_checkpoints(train: Path) -> list[Path]:
    """The checkpoints a training run saved, those scripts/train.py --resume can take: trainer/checkpoint-N
    folders that hold their trainer_state.json (a save cut short has none)."""
    trainer = train / "trainer"
    if not trainer.is_dir():
        return []
    return sorted(p for p in trainer.glob("checkpoint-*") if (p / "trainer_state.json").is_file())


def run_pilot(config: Config, candidate: Candidate, tools: Tools, *, stop_step: int | None = None,
              log: Log | None = None) -> dict:
    """The pilot of a candidate (module docstring): after the preflight, scripts/train.py to the stop step
    (resumed from the run's last saved checkpoint when there is one: the extension, or a pilot stopped part
    way), the GPU polled meanwhile for its peak, then the checkpoint's export; a step whose output exists is
    not run again. A run folder with a manifest and no saved checkpoint fails the pilot at once, before the
    preflight, and is left as it is. The record goes to pilot.json (pilot-<stop>.json for another stop
    step)."""
    stop = int(stop_step or config.schedule["stop_step"])
    folder = config.folder(candidate.key)
    folder.mkdir(parents=True, exist_ok=True)
    log = log or Log(folder / LOG_FILE)
    train = folder / "train"
    checkpoint = train / "trainer" / f"checkpoint-{stop}"
    gguf = train / "gguf" / f"checkpoint-{stop}-f16.gguf"
    result: dict = {"candidate": candidate.key, "stop_step": stop, "train": rel(train),
                    "checkpoint": rel(checkpoint), "gguf": None, "gguf_sha256": None, "trained": False,
                    "exported": False, "preflight": None, "nvidia_smi_peak_mib": None, "command": None,
                    "manifest": None, "passed": False, "error": None}
    record = folder / _stage_name(config, stop, "pilot")

    def done(error: str | None) -> dict:
        result["error"] = error
        result["passed"] = error is None
        write_json(record, result)
        log(f"pilot: {'done' if error is None else 'FAILED: ' + error}")
        return result

    if not (checkpoint / "trainer_state.json").is_file():
        saved = saved_checkpoints(train)
        if (train / "manifest.json").is_file() and not saved:
            return done(f"{rel(train)} holds a run that saved no checkpoint (it stopped before its first "
                        "save): scripts/train.py --resume refuses it (exit 5) and a fresh start would write "
                        "over its records; move the folder aside by hand, then run the pilot again")
        result["preflight"] = wait_until_clear(tools, config.thresholds, log=log)
        log(f"preflight: GPU {result['preflight']['gpu_used_mib']} MiB, RAM "
            f"{result['preflight']['available_ram_gb']} GB, AC {result['preflight']['on_ac_power']}")
        cmd = train_command(config, candidate, tools, out=train, stop_step=stop, resume=bool(saved))
        result["command"] = recorded(cmd, tools)
        log(f"train: {cmdline(cmd)}")
        process = tools.start(cmd, log=folder / f"pilot.train-{stop}", env=script_env())
        peak = 0
        while True:
            used = tools.probes.gpu_used_mib()
            peak = max(peak, used or 0)
            if process.poll() is not None:
                break
            tools.sleep(tools.poll_interval)
        result["trained"] = True
        result["nvidia_smi_peak_mib"] = peak
        log(f"training exited with {process.returncode}; peak GPU memory used (nvidia-smi) {peak} MiB")
        if process.returncode != 0:
            return done(f"the training exited with {process.returncode}")
        if not (checkpoint / "trainer_state.json").is_file():
            return done(f"the training exited with 0 but left no {rel(checkpoint)}")
    manifest = read_json(train / "manifest.json") or {}
    config_of = manifest.get("config") or {}
    data = config_of.get("data") or {}
    result["manifest"] = {"samples": data.get("samples"), "tokens_per_epoch": (data.get("tokens") or {}).get(
        "total"), "family": (config_of.get("family") or {}).get("name")}
    if not gguf.is_file():
        cmd = export_command(config, candidate, tools, checkpoint, gguf)
        log(f"export: {cmdline(cmd)}")
        exported = run_export(tools, cmd, gguf, log=folder / f"pilot.export-{stop}")
        result["exported"] = True
        if exported.returncode != 0 or not gguf.is_file():
            return done(f"the export exited with {exported.returncode}"
                        + ("" if gguf.is_file() else f" and wrote no {rel(gguf)}"))
    result["gguf"], result["gguf_sha256"] = rel(gguf), sha256_of(gguf)
    return done(None)


# ---- the evaluation ---------------------------------------------------------------------------------------


def _variant_prefix(config: Config, stop: int) -> str:
    return "dev-" if stop == config.schedule["stop_step"] else f"dev-{stop}-"


def run_evaluate(config: Config, candidate: Candidate, tools: Tools, *, stop_step: int | None = None,
                 rerun: bool = False, log: Log | None = None) -> dict:
    """The evaluation of a candidate's checkpoint (module docstring): every variant and b0 scored, each run
    on one server over its GGUF unless records that stand for it are there already (stale_records; rerun:
    run them all anyway), the server started for the first run that needs it and not at all when none does.
    The record goes to evaluate.json (evaluate-<stop>.json for another stop step), with each run's "reused";
    the first failure ends the evaluation with its error."""
    stop = int(stop_step or config.schedule["stop_step"])
    folder = config.folder(candidate.key)
    folder.mkdir(parents=True, exist_ok=True)
    log = log or Log(folder / LOG_FILE)
    gguf = folder / "train" / "gguf" / f"checkpoint-{stop}-f16.gguf"
    prefix = _variant_prefix(config, stop)
    result: dict = {"candidate": candidate.key, "stop_step": stop, "gguf": rel(gguf), "gguf_sha256": None,
                    "reasoning_budget": None, "server": None, "runs": {}, "passed": False, "error": None}
    record = folder / _stage_name(config, stop, "evaluate")

    def done(error: str | None) -> dict:
        result["error"] = error
        result["passed"] = error is None
        write_json(record, result)
        log(f"evaluate: {'done' if error is None else 'FAILED: ' + error}")
        return result

    if not gguf.is_file():
        return done(f"no {rel(gguf)}: the pilot exports it")
    result["gguf_sha256"] = sha256_of(gguf)
    budget = budget_of(config, candidate, read_json(folder / SMOKE_FILE))
    result["reasoning_budget"] = budget
    try:
        variants = variants_of(config, candidate, budget)
        runs = [(name, "ft", args) for name, args in variants] + [("b0", "b0", b0_variant(config, candidate,
                                                                                          budget))]
    except BakeoffError as exc:
        return done(str(exc))
    stale = {name: "rerun" if rerun else stale_records(config, candidate, folder / f"{prefix}{name}.jsonl",
                                                      system=system, args=args,
                                                      gguf_sha256=result["gguf_sha256"])
             for name, system, args in runs}
    try:
        with contextlib.ExitStack() as stack:
            server = None
            for name, system, args in runs:
                out = folder / f"{prefix}{name}.jsonl"
                score = folder / f"{prefix}{name}.score.json"
                if stale[name] is None:
                    log(f"eval {name}: keeping {rel(out)} (its manifest names this GGUF, the items and the "
                        "settings)")
                else:
                    if server is None:
                        server = stack.enter_context(LocalServer(require_server(tools).as_posix(),
                                                                 log=folder / f"{prefix}serve.log",
                                                                 **serve_args(candidate, gguf, tools)))
                        problem = _served(server, gguf)
                        if problem is not None:
                            return done(problem)
                        result["server"] = {"adapter": rel(gguf), "slots": SLOTS,
                                            "build": (server.props or {}).get("build_info"),
                                            "command": recorded(server.cmd, tools),
                                            "binary_sha256": sha256_if_file(tools.llama_server)}
                    cmd = eval_command(config, candidate, tools, system=system, out=out, args=args)
                    log(f"eval {name} ({stale[name]}): {cmdline(cmd)}")
                    evaluated = tools.run(cmd, log=folder / f"{prefix}{name}.eval", env=script_env())
                    if evaluated.returncode != 0:
                        return done(f"run_eval for {name} exited with {evaluated.returncode}")
                scored = tools.run(score_command(config, tools, out, score),
                                   log=folder / f"{prefix}{name}.score", env=script_env())
                if scored.returncode != 0:
                    return done(f"score.py for {name} exited with {scored.returncode}")
                data = read_json(score) or {}
                successes = sum(1 for s in data.get("scores") or [] if s.get("success"))
                result["runs"][name] = {"system": system, "args": args, "records": rel(out),
                                        "score": rel(score), "successes": successes,
                                        "n": (data.get("summary") or {}).get("n"),
                                        "reused": stale[name] is None}
                log(f"eval {name}: {successes}/{result['runs'][name]['n']}")
    except BakeoffError as exc:  # no llama-server given
        return done(str(exc))
    except RuntimeError as exc:  # the server could not be used (ServerError)
        return done(f"the server could not be used: {exc}")
    return done(None)


def _served(server: LocalServer, gguf: Path) -> str | None:
    """What keeps the evaluation off a server that came up, None when nothing: it must list exactly the
    checkpoint's GGUF as its adapter and report one slot."""
    listed = [a.get("path") if isinstance(a, dict) else a for a in server.adapters]
    if len(listed) != 1 or not isinstance(listed[0], str) or not same_path(listed[0], gguf):
        return f"the server lists the adapters {listed}, not exactly {rel(gguf)}"
    slots = (server.props or {}).get("total_slots")
    if slots != SLOTS:
        return f"the server reports {slots} slots (/props total_slots), not {SLOTS}"
    return None


_VARIANT_ARGS = argparse.ArgumentParser(add_help=False, allow_abbrev=False)  # the generation of a variant
_VARIANT_ARGS.add_argument("--mode")
_VARIANT_ARGS.add_argument("--seed", type=int)
_VARIANT_ARGS.add_argument("--reasoning-budget", type=int)
_VARIANT_ARGS.add_argument("--greedy", action="store_true")


def stale_records(config: Config, candidate: Candidate, out: Path, *, system: str, args: list[str],
                  gguf_sha256: str) -> str | None:
    """Why the records at out cannot stand for the run of run_eval's arguments args, None when they can:
    run_eval's manifest beside them must record the system, the development items (their path, their
    SHA-256, every item run, as many records), the candidate's tokenizer, the signals, exactly one adapter
    with the checkpoint GGUF's SHA-256, and the generation the arguments ask for (the mode, the base seed,
    the reasoning budget; a temperature of 0 for --greedy)."""
    manifest_path = out.with_name(out.stem + ".manifest.json")
    manifest = read_json(manifest_path) if out.is_file() else None
    if manifest is None:
        return "no records"
    asked, _ = _VARIANT_ARGS.parse_known_args(args)
    generation = manifest.get("generation") or {}
    mode = str(Mode.THINK) if asked.greedy else asked.mode
    checks = [("the system", manifest.get("system"), system),
              ("the items", manifest.get("items"), config.dev_items.as_posix()),
              ("the items' SHA-256", manifest.get("items_sha256"), sha256_of(config.dev_items)),
              ("the tokenizer", manifest.get("tokenizer"), Path(candidate.hf).as_posix()),
              ("the signals", manifest.get("signals"), True),
              ("the adapters' SHA-256", [a.get("sha256") for a in manifest.get("adapters") or []],
               [gguf_sha256])]
    if mode is not None:
        checks.append(("the mode", generation.get("mode"), mode))
    if asked.seed is not None:
        checks.append(("the base seed", generation.get("base_seed"), asked.seed))
    if asked.reasoning_budget is not None:
        checks.append(("the reasoning budget", generation.get("reasoning_budget"), asked.reasoning_budget))
    if asked.greedy:
        checks.append(("the temperature", (generation.get("decoding") or {}).get("temperature"), 0.0))
    for what, found, wanted in checks:
        if found != wanted:
            return f"{what} in {rel(manifest_path)} is {found!r}, not {wanted!r}"
    run, total = manifest.get("items_run"), manifest.get("items_total")
    lines = sum(1 for line in out.read_text(encoding="utf-8").splitlines() if line.strip())
    if run != total or lines != run:
        return f"{rel(out)} holds {lines} records of {run} run of {total} items, not every item once"
    return None


# ---- the gate ---------------------------------------------------------------------------------------------


def summarize_scores(score: Mapping, dev: Mapping[str, dict]) -> dict:
    """A score file's successes, in all and over the handwritten and drafted items, and per type as
    "a/b" (the pilot launcher's gate summary)."""
    scores = score["scores"]
    by_id = scores if isinstance(scores, dict) else {s["id"]: s for s in scores}
    handwritten = {i for i, item in dev.items() if (item.get("source") or {}).get("handwritten")}
    by_type: dict[str, list[int]] = {}
    for i, s in by_id.items():
        row = by_type.setdefault(dev[i]["type"], [0, 0])
        row[0] += bool(s["success"])
        row[1] += 1
    return {"successes": sum(bool(s["success"]) for s in by_id.values()), "n": len(by_id),
            "handwritten": sum(bool(by_id[i]["success"]) for i in handwritten if i in by_id),
            "drafted": sum(bool(s["success"]) for i, s in by_id.items() if i not in handwritten),
            "by_type": {t: f"{a}/{b}" for t, (a, b) in sorted(by_type.items())}}


def _exact(value: Fraction) -> int | float:
    return int(value) if value.denominator == 1 else float(value)


def _mean(values: list[int | float]) -> int | float:
    return _exact(Fraction(sum(Fraction(v) for v in values), len(values)))


def _scores_of(config: Config, candidate: Candidate, folder: Path, prefix: str, dev: Mapping,
               files: dict[str, str]) -> dict[str, dict]:
    """The summaries of the score files present under the prefix, the files recorded with their SHA-256."""
    summaries = {}
    for name in [*config.variants[candidate.thinking], "b0"]:
        path = folder / f"{prefix}{name}.score.json"
        if path.is_file():
            files[rel(path)] = sha256_of(path)
            summaries[name] = summarize_scores(read_json(path), dev)
    return summaries


def _primary(config: Config, candidate: Candidate, summaries: Mapping[str, dict]) -> dict | None:
    """P, the handwritten and drafted means over the primary variants; None when one is missing."""
    names = config.primary_variants(candidate)
    if any(name not in summaries for name in names):
        return None
    return {"P": _mean([summaries[n]["successes"] for n in names]),
            "handwritten": _mean([summaries[n]["handwritten"] for n in names]),
            "drafted": _mean([summaries[n]["drafted"] for n in names])}


def smoke_status(smoke: Mapping | None, candidate: Candidate) -> tuple[dict[str, str], list[str], list[str]]:
    """Each smoke step's status (passed, failed, blocked, exempt or missing), the reasons that make the
    candidate ineligible (a failed step whose input steps passed or are exempt) and what the gate still waits
    for (a step not run, or blocked); a step the candidate runs for the record only (record_only: C1's S5 and
    S6) is reported as such and is neither a reason nor awaited. A step is blocked by its record (S2's
    template refusals within the bound), or when it failed while an input step of its had not passed
    (STEP_INPUTS: S4 exports the adapter S3 trains, S5 serves the GGUF S4 writes): a scheduling gap, such as
    C2's S4 run before S3 ever ran under this tool, which waits for that input step and is never a reason
    (docs/m7-底座对比预登记.md 修订 3); a step run for the record only keeps its failure. Exemptions are the
    configuration's: a record that says exempt for a step the candidate is not exempt from counts as not
    run."""
    status, reasons, missing = {}, [], []
    steps = (smoke or {}).get("steps") or {}
    for step in STEPS:
        record = steps.get(step)
        for_record = step in candidate.record_only
        if step in candidate.exempt:
            status[step] = "exempt"
            continue
        if record is not None and record.get("exempt"):
            record = None
        waits = [s for s in STEP_INPUTS.get(step, ()) if status.get(s) not in ("passed", "exempt")]
        if record is None:
            status[step] = "missing"
            if not for_record:
                missing.append(f"{step} was not run")
        elif record.get("blocked"):
            status[step] = "blocked"
            if not for_record:
                missing.append(f"{step} is blocked ({SMOKE_FILE} says why)")
        elif record.get("passed"):
            status[step] = "passed"
        elif waits and not for_record:
            status[step] = "blocked"
            missing.append(f"{step} is blocked: it uses the output of {' and '.join(waits)}, which "
                           f"{'has' if len(waits) == 1 else 'have'} not passed")
        else:
            status[step] = "failed"
            if not for_record:
                reasons.append(f"{step} failed")
        if for_record and status[step] != "passed":
            status[step] += " (recorded only)"
    return status, reasons, missing


def required_variants(config: Config, candidate: Candidate) -> list[str]:
    """The variants a gate needs scores of: the primary ones, and for a hybrid family its nothink greedy
    run, which the hybrid rule reads should the family win."""
    names = config.primary_variants(candidate)
    if candidate.thinking == "hybrid" and NO_THINK_GREEDY not in names:
        names.append(NO_THINK_GREEDY)
    return names


def _failure_of(record: Mapping | None, what: str) -> str:
    """' (<what> failed: <its error>)' for a stage record that failed, '' otherwise."""
    return f" ({what} failed: {record['error']})" if record and record.get("error") else ""


def candidate_gate(config: Config, candidate: Candidate, *, stop_step: int | None = None) -> dict:
    """The gate JSON of a candidate from the files under its folder (module docstring: The gate JSON, its
    status, reasons and missing), written to gate.json whatever it says."""
    stop = int(stop_step or config.schedule["stop_step"])
    folder = config.folder(candidate.key)
    folder.mkdir(parents=True, exist_ok=True)
    dev = {item["id"]: item for item in dev_items(config)}
    files: dict[str, str] = {}
    train = folder / "train"
    manifest = read_json(train / "manifest.json")
    if manifest is not None:
        files[rel(train / "manifest.json")] = sha256_of(train / "manifest.json")
    rows = training_rows(train / "log.jsonl")
    smoke = read_json(folder / SMOKE_FILE)
    pilot = read_json(folder / _stage_name(config, stop, "pilot"))
    evaluation = read_json(folder / _stage_name(config, stop, "evaluate"))
    config_of = (manifest or {}).get("config") or {}
    data = config_of.get("data") or {}
    tokens = (data.get("tokens") or {}).get("total")
    median = throughput_median(rows, stop)
    hours = ft3_hours(config, tokens, median) if tokens and median else None
    peak = round(max(r.get("peak_reserved_gib") or 0 for r in rows) * GIB_PER_GB, 2) if rows else None
    summaries = _scores_of(config, candidate, folder, _variant_prefix(config, stop), dev, files)
    b0 = summaries.pop("b0", None)
    primary = _primary(config, candidate, summaries)
    status, reasons, missing = smoke_status(smoke, candidate)
    limit = config.thresholds["ft3_hours"]
    if hours is not None and hours > limit:
        reasons.append(f"the FT3 estimate of {hours:.2f} h is over {limit:g} h")
    if manifest is None:
        missing.append(f"no pilot: {rel(train / 'manifest.json')} is missing"
                       + _failure_of(pilot, "the pilot"))
    elif hours is None:
        missing.append("no FT3 hour estimate: the pilot's log has no throughput at steps 100 and after")
    absent = [name for name in required_variants(config, candidate) if name not in summaries]
    if absent:
        missing.append(f"no scores for {', '.join(absent)}" + _failure_of(evaluation, "the evaluation"))
    extension = None
    extend = int(config.schedule["extend_step"])
    if (train / "trainer" / f"checkpoint-{extend}" / "trainer_state.json").is_file():
        extended = _scores_of(config, candidate, folder, _variant_prefix(config, extend), dev, files)
        extended.pop("b0", None)
        scored = _primary(config, candidate, extended)
        if scored is not None:
            hybrid = candidate.thinking == "hybrid"
            extension = {"stop_step": extend, "P": scored["P"], "handwritten": scored["handwritten"],
                         "no_think_greedy": (extended.get(NO_THINK_GREEDY) or {}).get("successes")
                         if hybrid else None,
                         "primary": config.primary_variants(candidate), "variants": extended}
            if hybrid and NO_THINK_GREEDY not in extended:  # the hybrid rule reads it after the extension
                missing.append(f"no scores for {NO_THINK_GREEDY} at step {extend}" + _failure_of(
                    read_json(folder / _stage_name(config, extend, "evaluate")), "the evaluation"))
    gate_status = "ineligible" if reasons else "incomplete" if missing else "scored"
    gate = {
        "candidate": candidate.key, "name": candidate.name, "model": candidate.model,
        "revision": candidate.revision, "hf": candidate.hf, "gguf": candidate.gguf,
        "family": candidate.family, "trained_family": (config_of.get("family") or {}).get("name"),
        "thinking": candidate.thinking,
        "mode": candidate.primary_mode, "reasoning_budget": budget_of(config, candidate, smoke),
        "stop_step": stop, "samples": data.get("samples"), "modes": data.get("modes"),
        "tokens_per_epoch": tokens, "peak_reserved_gb": peak,
        "nvidia_smi_peak_mib": (pilot or {}).get("nvidia_smi_peak_mib"),
        "tokens_per_s_median_100_to_stop": median, "ft3_hours_estimate": shown(hours),
        "ft3_hours_unrounded": hours,
        "primary": config.primary_variants(candidate), "variants": summaries, "b0": b0,
        "P": None if primary is None else primary["P"],
        "handwritten": None if primary is None else primary["handwritten"],
        "drafted": None if primary is None else primary["drafted"],
        "no_think_greedy": (summaries.get(NO_THINK_GREEDY) or {}).get("successes")
        if candidate.thinking == "hybrid" else None,
        "smoke": smoke, "smoke_status": status, "status": gate_status,
        "eligible": False if reasons else None if missing else True, "reasons": reasons, "missing": missing,
        "extension": extension, "files": files}
    write_json(folder / GATE_FILE, gate)
    return gate


# ---- the decision and the aggregate -----------------------------------------------------------------------


def _mode_of(entry: Mapping, hybrid_margin: float, *, extended: bool = False) -> str:
    """The winner's FT3 mode (module docstring: The aggregate): no_think for a winner trained without
    thinking; for a hybrid family no_think when its nothink greedy successes lead its P by hybrid_margin or
    more, both read at the step the decision's basis came from: the extension's no_think_greedy and P when
    extended (docs/m7-底座对比预登记.md 修订 3), the stop step's otherwise; think for every other winner, and
    for a hybrid one whose nothink score is not there."""
    if entry["mode"] == str(Mode.NO_THINK):
        return str(Mode.NO_THINK)
    numbers = (entry.get("extension") or {}) if extended else entry
    greedy, primary = numbers.get("no_think_greedy"), numbers.get("P")
    if (entry["thinking"] == "hybrid" and greedy is not None and primary is not None
            and greedy - primary >= hybrid_margin):
        return str(Mode.NO_THINK)
    return str(Mode.THINK)


def _hours_of(entry: Mapping) -> float | None:
    """An entry's FT3 estimate for the tie-break: the unrounded one when the entry has it."""
    exact = entry.get("ft3_hours_unrounded")
    return exact if exact is not None else entry.get("ft3_hours_estimate")


def decide(entries: list[Mapping], *, tie: float, hybrid_margin: float, extend_step: int = 548,
           waiting: list[str] | tuple[str, ...] = ()) -> dict:
    """The preregistered rule over the gates' entries (candidate, P, eligible, thinking, mode, handwritten,
    ft3_hours_estimate and ft3_hours_unrounded, no_think_greedy, extension with its P, handwritten and
    no_think_greedy): the status (decided, extend, undecided or none-eligible), the winner, the basis, the
    winner's mode (from the numbers of the step its basis came from: _mode_of), the ranking by P of the
    candidates not excluded (equal P ordered by key), the two to extend, the runner-up, the margin between
    the top two, and waiting, the candidates still to come. While any is waiting (module docstring: The
    aggregate), the status is provisional and names no winner: the rule's outcome is the standing, its winner
    the leader; an entry still incomplete (eligible None) ranks with its P meanwhile."""
    eligible = [e for e in entries if e.get("eligible") is not False and e.get("P") is not None]
    ranked = sorted(eligible, key=lambda e: (-e["P"], e["candidate"]))
    standing = _rule(ranked, tie=tie, hybrid_margin=hybrid_margin, extend_step=extend_step)
    if not waiting:
        return {**standing, "waiting": []}
    return {**standing, "status": "provisional", "standing": standing["status"], "winner": None,
            "leader": standing["winner"], "waiting": list(waiting)}


def _rule(ranked: list[Mapping], *, tie: float, hybrid_margin: float, extend_step: int) -> dict:
    """The rule over the ranked entries (decide)."""
    base = {"status": "none-eligible", "winner": None, "basis": None, "mode": None,
            "ranking": [e["candidate"] for e in ranked], "extend": None, "runner_up": None, "margin": None,
            "note": None}

    def decided(winner: Mapping, basis: str, *, extended: bool = False, **more) -> dict:
        """The decision for winner on basis; extended: the basis came from the extension, whose numbers the
        hybrid rule then reads (_mode_of)."""
        return {**base, "status": "decided", "winner": winner["candidate"], "basis": basis,
                "mode": _mode_of(winner, hybrid_margin, extended=extended), **more}

    if not ranked:
        return base
    if len(ranked) == 1:
        return decided(ranked[0], "P")
    first, second = ranked[0], ranked[1]
    margin = first["P"] - second["P"]
    note = None
    if len(ranked) > 2 and ranked[2]["P"] == second["P"]:
        shared = [e["candidate"] for e in ranked[1:] if e["P"] == second["P"]]
        note = f"second place is shared by {', '.join(shared)}; the candidate key orders them"
    if margin > tie:
        return decided(first, "P", runner_up=second["candidate"], margin=margin, note=note)
    pending = [e["candidate"] for e in (first, second) if not e.get("extension")]
    both = [first["candidate"], second["candidate"]]
    if pending:
        return {**base, "status": "extend", "extend": both, "pending": pending,
                "runner_up": second["candidate"], "margin": margin, "note": note}
    after = {"runner_up": second["candidate"], "margin": margin, "note": note}
    p_first, p_second = first["extension"]["P"], second["extension"]["P"]
    if abs(p_first - p_second) > tie:
        return decided(first if p_first > p_second else second, f"P@{extend_step}", extended=True, **after)
    h_first, h_second = first["extension"]["handwritten"], second["extension"]["handwritten"]
    if h_first != h_second:
        return decided(first if h_first > h_second else second, f"handwritten@{extend_step}", extended=True,
                       **after)
    hours = (_hours_of(first), _hours_of(second))
    if None not in hours and hours[0] != hours[1]:
        return decided(first if hours[0] < hours[1] else second, "ft3_hours", extended=True, **after)
    return {**base, "status": "undecided", "extend": both, **after}


def _reference_entry(config: Config) -> dict:
    """The reference C0 as an entry of the decision and of the results file: its score file summarised
    (the configured P must be its successes, and G2's record must say the same), its FT3 estimate by the
    preregistered algorithm from G2's pilot folder (the configuration's reference.train: the manifest's
    tokens per epoch, the median tokens/s of steps 100..stop of its log), eligible by the preregistration."""
    ref = config.reference
    entry = {"status": "reference", "candidate": ref["key"], "name": ref.get("name"),
             "model": ref.get("model"), "thinking": ref.get("thinking"),
             "mode": ref.get("mode", str(Mode.NO_THINK)), "P": ref.get("P"),
             "handwritten": ref.get("handwritten"), "eligible": True, "reasons": [], "missing": [],
             "tokens_per_epoch": None, "tokens_per_s_median_100_to_stop": None, "ft3_hours_estimate": None,
             "ft3_hours_unrounded": None, "no_think_greedy": None, "extension": ref.get("extension"),
             "score": ref.get("score"), "source": ref.get("source"), "train": ref.get("train"), "files": {}}
    dev = None
    if ref.get("score") and Path(ref["score"]).is_file():
        dev = {item["id"]: item for item in dev_items(config)}
        summary = summarize_scores(read_json(Path(ref["score"])), dev)
        entry["files"][ref["score"]] = sha256_of(ref["score"])
        if ref.get("P") is not None and summary["successes"] != ref["P"]:
            raise BakeoffError(f"the reference's score file {ref['score']} holds {summary['successes']} "
                               f"successes, not the configured P {ref['P']}")
        entry.update({"P": summary["successes"], "handwritten": summary["handwritten"],
                      "drafted": summary["drafted"], "by_type": summary["by_type"]})
    if ref.get("source") and Path(ref["source"]).is_file():
        source = read_json(Path(ref["source"])) or {}
        entry["files"][ref["source"]] = sha256_of(ref["source"])
        if source.get("dev_successes") != entry["P"]:
            raise BakeoffError(f"the reference's record {ref['source']} says dev_successes "
                               f"{source.get('dev_successes')}, not {entry['P']}")
    if ref.get("train"):
        train = Path(ref["train"])
        manifest = read_json(train / "manifest.json")
        for name in ("manifest.json", "log.jsonl"):
            if (train / name).is_file():
                entry["files"][rel(train / name)] = sha256_of(train / name)
        tokens = ((((manifest or {}).get("config") or {}).get("data") or {}).get("tokens") or {}).get("total")
        median = throughput_median(training_rows(train / "log.jsonl"), int(config.schedule["stop_step"]))
        hours = ft3_hours(config, tokens, median) if tokens and median else None
        entry.update({"tokens_per_epoch": tokens, "tokens_per_s_median_100_to_stop": median,
                      "ft3_hours_estimate": shown(hours), "ft3_hours_unrounded": hours})
    return entry


def _gate_entry(key: str, gate: Mapping, path: Path) -> dict:
    """A candidate's gate as an entry of the decision and of the results file. A gate without a status this
    module writes (another tool's, such as the pilot launcher's) is incomplete: it is taken for no
    eligibility either way, and this module's gate replaces it."""
    entry = {"candidate": key, "status": gate.get("status"), "P": gate.get("P"),
             "eligible": gate.get("eligible"), "reasons": gate.get("reasons") or [],
             "missing": gate.get("missing") or [], "thinking": gate.get("thinking"), "mode": gate.get("mode"),
             "handwritten": gate.get("handwritten"), "ft3_hours_estimate": gate.get("ft3_hours_estimate"),
             "ft3_hours_unrounded": gate.get("ft3_hours_unrounded"),
             "no_think_greedy": gate.get("no_think_greedy"), "extension": gate.get("extension")}
    if entry["status"] not in GATE_STATUSES:
        rewrite = f"run scripts/bakeoff.py gate --candidate {key}"
        entry |= {"status": "incomplete", "eligible": None,
                  "missing": [f"{rel(path)} has no status (another tool wrote it): {rewrite}"]}
    return entry


def aggregate(config: Config) -> dict:
    """The results file over every candidate's gate and the reference (module docstring: The aggregate),
    written to the configuration's results path: a candidate without a gate is pending, and the decision
    waits for the pending and the incomplete ones (complete once none is left); a skipped candidate is
    recorded with its reason and waited for by nothing, and BakeoffError when its gate holds a P."""
    reference = _reference_entry(config)
    candidates: dict[str, dict] = {reference["candidate"]: reference}
    entries = [reference]
    for key, candidate in config.candidates.items():
        path = config.folder(key) / GATE_FILE
        gate = read_json(path)
        if candidate.skipped is not None:
            if gate is not None and gate.get("P") is not None:
                raise BakeoffError(f"{key} is skipped ({candidate.skipped}) but {rel(path)} holds its P: a "
                                   "cut line skips a candidate before its score, never after")
            candidates[key] = {"status": "skipped", "candidate": key, "name": candidate.name,
                               "thinking": candidate.thinking, "reason": candidate.skipped, "gate": rel(path)}
            continue
        if gate is None:
            candidates[key] = {"status": "pending", "candidate": key, "name": candidate.name,
                               "thinking": candidate.thinking, "gate": rel(path)}
            continue
        entry = _gate_entry(key, gate, path)
        entries.append(entry)
        candidates[key] = {**entry, "name": gate.get("name"), "smoke": gate.get("smoke_status"),
                           "gate": rel(path), "gate_sha256": sha256_of(path)}
    waiting = [key for key, c in candidates.items() if c["status"] in ("pending", "incomplete")]
    thresholds = config.thresholds
    decision = decide(entries, tie=thresholds["tie"], hybrid_margin=thresholds["hybrid_margin"],
                      extend_step=int(config.schedule["extend_step"]), waiting=waiting)
    result = {"config": rel(config.path), "config_sha256": sha256_of(config.path),
              "preregistration": PREREGISTRATION.as_posix(),
              "preregistration_sha256": sha256_of(PREREGISTRATION) if PREREGISTRATION.is_file() else None,
              "rule": {"tie": thresholds["tie"], "hybrid_margin": thresholds["hybrid_margin"],
                       "ft3_hours": thresholds["ft3_hours"],
                       "extend_step": int(config.schedule["extend_step"]), "smoke_steps": list(STEPS)},
              "candidates": candidates, "decision": decision, "complete": not waiting}
    write_json(config.results, result)
    return result


# ---- the trained adapter's fidelity -----------------------------------------------------------------------


def trained_step(config: Config, candidate: Candidate, stop_step: int | None = None) -> int:
    """The checkpoint step of the trained adapter's fidelity (module docstring): stop_step when given, which
    must be the schedule's stop step or its extension step (BakeoffError otherwise: the preregistration names
    those two); else the extension step once the candidate's run saved that checkpoint (its
    trainer_state.json, as the gate reads the extension), and the stop step before."""
    stop, extend = int(config.schedule["stop_step"]), int(config.schedule["extend_step"])
    if stop_step is not None:
        if stop_step not in (stop, extend):
            raise BakeoffError(f"the trained adapter's fidelity reads checkpoint {stop} or, after the "
                               f"extension, {extend} (the preregistration's revision 1), not {stop_step}")
        return int(stop_step)
    trainer = config.folder(candidate.key) / "train" / "trainer"
    return extend if (trainer / f"checkpoint-{extend}" / "trainer_state.json").is_file() else stop


def run_trained_fidelity(config: Config, candidate: Candidate, tools: Tools, *,
                         stop_step: int | None = None, log: Log | None = None) -> dict:
    """The trained adapter's fidelity of a candidate, report only (module docstring): the checkpoint of
    trained_step, its GGUF exported first (run_export) when missing, then over S6's sequences the training
    stack's top-1 tokens without an adapter and with the checkpoint, and llama.cpp's without an adapter and
    with the GGUF (the Tools' fidelity backends, their llama-server logging to
    fidelity-trained-<step>.server.log), counted by eval.fidelity.compare. The record goes to
    fidelity-trained-<step>.json, its error None when the comparison was made; a checkpoint not saved with
    its adapter, an export that failed, an adapter the training stack did not load (no llama-server is
    started then) or one of STEP_ERRORS on the way is the error instead, the counts None. BakeoffError for a
    step that is neither the stop step nor the extension step."""
    step = trained_step(config, candidate, stop_step)
    folder = config.folder(candidate.key)
    folder.mkdir(parents=True, exist_ok=True)
    log = log or Log(folder / LOG_FILE)
    checkpoint = folder / "train" / "trainer" / f"checkpoint-{step}"
    gguf = folder / "train" / "gguf" / f"checkpoint-{step}-f16.gguf"
    stem = folder / f"{TRAINED_FIDELITY}-{step}"
    settings = {"items": int(config.smoke["items"]), "positions": int(config.smoke["positions"]),
                "mode": str(SEQUENCES_MODE)}
    result: dict = {"candidate": candidate.key, "stop_step": step, "report_only": True,
                    "adapter": rel(checkpoint), "adapter_sha256": None, "modules": None,
                    "adapter_tensors": None, "gguf": rel(gguf), "gguf_sha256": None, "exported": False,
                    "export": None, "base_gguf": candidate.gguf, "settings": settings,
                    "sequences_sha256": None, **dict.fromkeys(TRAINED_COUNTS), "error": None}

    def done(error: str | None) -> dict:
        result["error"] = error
        write_json(Path(f"{stem}.json"), result)
        if error is None:
            log(f"fidelity (report only): checkpoint-{step}: agreement {result['agreement']:.3f} with the "
                f"adapter, {result['base_agreement']:.3f} without; the adapter changed "
                f"{result['changed']:.3f} of the training stack's top-1 tokens; {result['positions']} "
                "positions")
        else:
            log(f"fidelity: FAILED: {error}")
        return result

    try:
        missing = [name for name in ("trainer_state.json", "adapter_config.json", ADAPTER_WEIGHTS)
                   if not (checkpoint / name).is_file()]
        if missing:
            return done(f"no trained adapter at {rel(checkpoint)} (no {', '.join(missing)}): the pilot "
                        "trains it")
        counts = adapter_modules(checkpoint)
        result.update(adapter_sha256=sha256_of(checkpoint / ADAPTER_WEIGHTS), modules=counts["modules"],
                      adapter_tensors=counts["tensors"])
        if not gguf.is_file():
            cmd = export_command(config, candidate, tools, checkpoint, gguf)
            log(f"fidelity export: {cmdline(cmd)}")
            exported = run_export(tools, cmd, gguf, log=Path(f"{stem}.export"))
            converter = None if tools.llama_cpp_src is None else Path(tools.llama_cpp_src) / CONVERTER
            result.update(exported=True, export={"command": recorded(cmd, tools),
                                                 "returncode": exported.returncode,
                                                 "converter_sha256": sha256_if_file(converter)})
            if exported.returncode != 0 or not gguf.is_file():
                return done(f"the export exited with {exported.returncode}"
                            + ("" if gguf.is_file() else f" and wrote no {rel(gguf)}"))
        result["gguf_sha256"] = sha256_of(gguf)
        family = tools.load_family(Path(candidate.hf), candidate.family)
        items = dev_items(config)[:int(config.smoke["items"])]
        sequences = sequences_of(family, items, positions=int(config.smoke["positions"]))
        result["sequences_sha256"] = hashlib.sha256(canonical_json(
            [dataclasses.asdict(s) for s in sequences]).encode("utf-8")).hexdigest()
        backends = tools.fidelity_backends(config, candidate, tools, folder,
                                           server_log=Path(f"{stem}.server.log"))
        hf_base = backends.hf_top1(None, sequences)
        try:
            hf_trained = backends.hf_top1(checkpoint, sequences)
        except AdapterNotLoaded as exc:
            return done(f"the training stack did not load the trained adapter: {exc}")
        llama_base = backends.llama_top1(None, sequences)
        llama_trained = backends.llama_top1(gguf, sequences)
        counted = compare(hf_trained, llama_trained, hf_base, llama_base)
    except STEP_ERRORS as exc:
        return done(f"{type(exc).__name__}: {exc}")
    result.update({key: counted[key] for key in TRAINED_COUNTS})
    return done(None)


# ---- the plan ---------------------------------------------------------------------------------------------


def plan_lines(config: Config, candidate: Candidate, tools: Tools) -> list[str]:
    """Every command the candidate's smoke, pilot and evaluation would run, as text, then those of the
    trained adapter's fidelity at the checkpoint trained_step picks; nothing runs."""
    folder = config.folder(candidate.key)
    budget = budget_of(config, candidate, read_json(folder / SMOKE_FILE))
    schedule = config.schedule
    stop = schedule["stop_step"]
    train = folder / "train"
    checkpoint = train / "trainer" / f"checkpoint-{stop}"
    gguf = train / "gguf" / f"checkpoint-{stop}-f16.gguf"
    smoke_train = folder / "smoke" / "train"
    smoke_gguf = smoke_train / "gguf" / "adapter-f16.gguf"
    binary = tools.llama_server.as_posix() if tools.llama_server is not None else "llama-server"
    model = Path(candidate.gguf).as_posix()
    extra = ["-np", str(SLOTS)]
    budget_text = "none configured" if budget is None else str(budget)
    if budget is None:
        budget_text += f" (S2 computes it; {PLAN_BUDGET} below)"
    forced = f" and under the budget {config.smoke['forced_budget']}" if candidate.thinks else ""
    lines = [
        (f"{candidate.key}: {candidate.name} ({candidate.thinking}; family "
         f"{candidate.family or 'resolved from the folder'}; hf {candidate.hf}; gguf {candidate.gguf}; "
         f"exempt {', '.join(candidate.exempt) or 'none'})"),
        (f"schedule: stop_step {stop}, save_steps {schedule['save_steps']}, extend_step "
         f"{schedule['extend_step']}, max_length {config.max_length(candidate)}"),
        f"reasoning budget: {budget_text}",
        "S1: " + cmdline(golden_command(candidate, tools)),
        "S2: " + cmdline(train_command(config, candidate, tools, out=folder / "smoke" / "dryrun",
                                       dry_run=True)),
        "S3: " + cmdline(train_command(config, candidate, tools, out=smoke_train,
                                       max_steps=int(config.smoke["steps"]), smoke_longest=True)),
        "S4: " + cmdline(export_command(config, candidate, tools, smoke_train / "adapter", smoke_gguf)),
        ("S5: " + cmdline(server_command(binary, model=model, adapter=rel(smoke_gguf), port=tools.port,
                                         extra=extra))
         + f"; {config.smoke['items']} items twice under the primary mode{forced}"),
        (f"S6: fidelity over the family's LoRA targets, {config.smoke['items']} items x "
         f"{config.smoke['positions']} positions, each group's strength calibrated on the training stack "
         f"first (fractions {', '.join(f'{f:g}' for f in strengths())}, until {AIM:g} of its top-1 change)"),
        "pilot: " + cmdline(train_command(config, candidate, tools, out=train, stop_step=stop)),
        "export: " + cmdline(export_command(config, candidate, tools, checkpoint, gguf)),
        "server: " + cmdline(server_command(binary, model=model, adapter=rel(gguf), port=tools.port,
                                            extra=extra)),
    ]
    shown = budget if budget is not None else PLAN_BUDGET
    runs = [(name, "ft", args) for name, args in variants_of(config, candidate, shown)]
    runs.append(("b0", "b0", b0_variant(config, candidate, shown)))
    for name, system, args in runs:
        out = folder / f"dev-{name}.jsonl"
        lines.append(f"run_eval {name}: " + cmdline(eval_command(config, candidate, tools, system=system,
                                                                  out=out, args=args)))
        lines.append(f"score {name}: " + cmdline(score_command(config, tools, out,
                                                               folder / f"dev-{name}.score.json")))
    step, extend = trained_step(config, candidate), int(schedule["extend_step"])
    trained = train / "trainer" / f"checkpoint-{step}"
    trained_gguf = train / "gguf" / f"checkpoint-{step}-f16.gguf"
    which = ("the extension's" if step == extend
             else f"the extension's checkpoint-{extend} once the run is extended")
    record = folder / f"{TRAINED_FIDELITY}-{step}.json"
    lines += [
        (f"fidelity: report only, the trained adapter {rel(trained)} ({which}) on S6's "
         f"{config.smoke['items']} items x {config.smoke['positions']} positions, the training stack (NF4) "
         f"and llama.cpp (Q8_0) each with and without it, into {rel(record)}"),
        ("fidelity export (when the GGUF is missing): "
         + cmdline(export_command(config, candidate, tools, trained, trained_gguf))),
        "fidelity server (no adapter): " + cmdline(server_command(binary, model=model, port=tools.port,
                                                                  extra=extra)),
        "fidelity server (the adapter): " + cmdline(server_command(binary, model=model,
                                                                   adapter=rel(trained_gguf),
                                                                   port=tools.port, extra=extra)),
    ]
    return lines

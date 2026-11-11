"""The frozen final system (docs/m9-计划.md, 不改动的部分 and WP1): what scripts/ask.py runs must be
the system M7 evaluated, so every start checks it against the frozen record (results/final-m7.json, written
by scripts/freeze_final.py) before anything runs.

load_frozen(path, config) checks, in this order, and raises FrozenMismatch naming every field that disagrees:

1. the code's hashes (code_hashes: the prompt's version and hash, agent.prompts; the tool definitions' hash,
   tools.schemas; the runtime rules' version, agent.rules) are the evaluated ones (EVALUATED);
2. the frozen record is the file M7 froze, byte for byte (FROZEN_SHA256, its SHA-256, which also pins every
   value it holds: think mode, the reasoning budget of 1,280, base seed 0, five votes, the final vote mode),
   so a record written again, with a configuration naming it, is not taken for it;
3. the family of the record's tokenizer folder (family.load_family) records what the frozen record holds;
4. the generation settings rebuilt on that family from the record's mode, reasoning budget and base seed, with
   the record's votes (eval.systems.with_votes), record exactly the record's generation;
5. the rules are this code's ({"version": RULES_VERSION, "names": RULES}), the vote mode is the final one
   (agent.vote.FINAL, as docs/m9-计划.md WP1 builds the system) and there is no glossary;
6. the SHA-256 of the adapter GGUF and of the base model GGUF, hashed at every start, are the record's;
7. the M7 session configuration (results/test-config-m7.json) names this record (systems.final.frozen: its
   path and the SHA-256 of the file), its final system's options and its adapter entry (adapters.final) agree
   with the record, and its server settings are the evaluated ones (EVALUATED_SERVER: the port, the context
   size, the timeout and the llama.cpp build).

The code's hashes are checked first and alone, and a family that cannot be loaded stops the check there (the
generation is rebuilt on it); every other disagreement is collected, so one FrozenMismatch names them all.
FrozenSystem carries what a run needs from both files: the record, the loaded family, the generation settings
with the votes, the paths, the vote mode, whether the rules are on, the server settings with the GPU's idle
threshold (gpu.idle_max_mib), and the SHA-256 of both files.

identity_problems(info, frozen, props) holds a run's recorded identity against the frozen system. info is
eval.systems' info of the built system (Built.info: scripts/ask.py writes it into session.json, and
scripts/run_e2e.py into its report's manifest, beside the prompt and schema hashes it checks too); props is
the server's /props. One message per field that disagrees, [] when none does: the family, the generation
(the votes included), the vote mode, the rules, the glossary, the backend, the lora scale, the tokenizer
folder, exactly one adapter (the frozen GGUF, by path and SHA-256), and on the server one slot (SLOTS), the
configured build, the frozen base model and, when /props reports it, the context size.

tree_problems(git_dirty) holds what a run recorded of its working tree (train.manifest.git_dirty, which
scripts/ask.py writes into session.json and scripts/run_e2e.py into its manifest) to a clean one: a run of the
frozen system is a run of the commit it names (docs/m9-计划.md WP4: 在干净的提交上运行), and the hashes above
do not cover an uncommitted change elsewhere in the code.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..agent.prompts import PROMPT_HASH, PROMPT_VERSION
from ..agent.rules import RULES, RULES_VERSION
from ..agent.vote import FINAL as FINAL_VOTE_MODE
from ..canonical import canonical_json
from ..eval.lock import sha256_of
from ..eval.systems import LLAMA_CPP, with_votes
from ..family.core import LoadedFamily, load_family
from ..llm.generation import GenerationSettings
from ..llm.server import same_path
from ..tools.schemas import SCHEMA_HASH

FROZEN_FILE = Path("results/final-m7.json")
# The SHA-256 of FROZEN_FILE as M7 froze it (scripts/freeze_final.py; the session configuration's
# systems.final.frozen.sha256 holds the same)
FROZEN_SHA256 = "493987313c8aea3fee40748e5333f413d81fdefce76d5a618b5f2aeb6fb428ea"
SESSION_CONFIG = Path("results/test-config-m7.json")
SYSTEM = "final"  # the final system's name in the session configuration (systems, adapters)
# One llama-server slot (-np 1), as scripts/run_test.py SLOTS: an output must not depend on the requests
# before it.
SLOTS = 1
LORA_SCALE = 1.0  # the adapter applied, as eval.systems' ft runs it
# FT3's max_length (scripts/train.py --max-length 5120): final's training length end to end
# (scripts/run_e2e.py --training-length), which scripts/check_final_identity.py holds a report to.
TRAINING_LENGTH = 5120
SAMPLED = "sampled"  # the session configuration's word for the family's own sampling (scripts/run_test.py)
# The code the final system was evaluated with (tests/eval/test_decide_m8.py pins the same values).
EVALUATED = {"prompt_version": "m2-2026-10-13", "prompt_hash": "c827097301c986b1",
             "schema_hash": "7d831712685e42d0", "rules_version": "m7-wp3-1"}
# The server it was evaluated on: the session configuration's server block (docs/m9-计划.md WP1).
EVALUATED_SERVER = {"port": 8080, "ctx_size": 8192, "timeout_s": 120, "llama_cpp_build": "b11349-fb4b2737a"}


class FrozenMismatch(Exception):
    """The code, the frozen record or the session configuration does not describe the evaluated system;
    problems names every field that disagrees."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


@dataclass(frozen=True)
class ServerSettings:
    """The llama-server the final system runs on, from the session configuration (module docstring, 6)."""

    port: int
    ctx_size: int
    timeout_s: float
    llama_cpp_build: str
    idle_max_mib: float  # the most MiB in use on the GPU that still count as idle before a server starts


@dataclass(frozen=True)
class FrozenSystem:
    """The checked frozen system (module docstring)."""

    record: dict
    family: LoadedFamily
    settings: GenerationSettings  # the generation with the votes
    tokenizer: str
    base_path: str
    adapter_path: str
    vote_mode: str
    rules: bool
    server: ServerSettings
    frozen_path: str
    frozen_sha256: str
    config_path: str
    config_sha256: str


def code_hashes() -> dict:
    """The values of this code that EVALUATED pins."""
    return {"prompt_version": PROMPT_VERSION, "prompt_hash": PROMPT_HASH, "schema_hash": SCHEMA_HASH,
            "rules_version": RULES_VERSION}


def _short(value: object) -> str:
    """A value as a message shows it: canonical JSON, cut to 200 characters."""
    try:
        text = canonical_json(value)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= 200 else text[:197] + "..."


def _read(path: Path, what: str) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FrozenMismatch([(f"{what} {path.as_posix()} cannot be read ({exc}); run from the "
                               "repository root")]) from None
    if not isinstance(data, dict):
        raise FrozenMismatch([f"{what} {path.as_posix()} is not a JSON object"])
    return data


def _field(data: dict, dotted: str, where: str) -> object:
    """data's value at a dotted key; FrozenMismatch naming it when it is missing."""
    value: object = data
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            raise FrozenMismatch([f"{where} has no {dotted}"])
        value = value[key]
    return value


def _object(data: dict, dotted: str, where: str) -> dict:
    """_field's value when it is a JSON object; FrozenMismatch naming it otherwise."""
    value = _field(data, dotted, where)
    if not isinstance(value, dict):
        raise FrozenMismatch([f"{where}: {dotted} is not an object"])
    return value


def load_frozen(path: str | Path = FROZEN_FILE, config: str | Path = SESSION_CONFIG, *,
                load: Callable[[str], LoadedFamily] = load_family,
                hash_file: Callable[[str | Path], str] = sha256_of) -> FrozenSystem:
    """The frozen system, checked (module docstring); FrozenMismatch otherwise. load loads a family from a
    model folder and hash_file gives a file's SHA-256 (the tests' stand-ins)."""
    code = code_hashes()
    wrong = [f"{key}: this code has {code[key]!r}, the evaluated system {want!r}"
             for key, want in EVALUATED.items() if code[key] != want]
    if wrong:
        raise FrozenMismatch(wrong)
    path, config = Path(path), Path(config)
    record, session = _read(path, "the frozen record"), _read(config, "the session configuration")
    where, there = path.as_posix(), config.as_posix()
    frozen_sha256 = hash_file(path)
    problems: list[str] = []
    if frozen_sha256 != FROZEN_SHA256:
        problems.append(f"frozen record: {where} has SHA-256 {frozen_sha256}, the record M7 froze "
                        f"{FROZEN_SHA256}")
    tokenizer = _field(record, "tokenizer", where)
    try:
        family = load(tokenizer)
    except Exception as exc:  # noqa: BLE001  whatever it is (no registered family, no folder, no package)
        raise FrozenMismatch([*problems, (f"family: the family of {tokenizer} cannot be loaded: "
                                          f"{type(exc).__name__}: {exc}")]) from None
    if family.record() != _field(record, "family", where):
        problems.append(f"family: {tokenizer} loads as {_short(family.record())}, the frozen record holds "
                        f"{_short(record['family'])}")
    generation = _field(record, "generation", where)
    try:
        settings = with_votes(GenerationSettings.for_family(
            family, mode=_field(record, "generation.mode", where),
            reasoning_budget=_field(record, "generation.reasoning_budget", where),
            base_seed=_field(record, "generation.base_seed", where)), _field(record, "votes", where))
    except (ValueError, TypeError) as exc:
        raise FrozenMismatch([*problems, (f"generation: the record's cannot be rebuilt on {tokenizer}: "
                                          f"{exc}")]) from None
    if settings.record() != generation:
        problems.append(f"generation: rebuilt on {tokenizer} it records {_short(settings.record())}, the "
                        f"frozen record {_short(generation)}")
    rules = {"version": RULES_VERSION, "names": list(RULES)}
    if record.get("rules") != rules:
        problems.append(f"rules: the frozen record holds {_short(record.get('rules'))}, this code "
                        f"{_short(rules)}")
    vote_mode = record.get("vote_mode")
    if vote_mode != FINAL_VOTE_MODE:
        problems.append(f"vote_mode: the frozen record holds {vote_mode!r}, the final system votes in "
                        f"{FINAL_VOTE_MODE!r} mode")
    if record.get("glossary") is not None:
        problems.append(f"glossary: the frozen record names {_short(record['glossary'])}; the final system "
                        "has none")
    files = {}
    for name in ("adapter", "base"):
        file, digest = _field(record, f"{name}.path", where), _field(record, f"{name}.sha256", where)
        files[name] = file
        try:
            actual = hash_file(file)
        except (OSError, TypeError) as exc:  # TypeError: a path that is not one
            problems.append(f"{name}: {file!r} cannot be read ({exc})")
            continue
        if actual != digest:
            problems.append(f"{name}: {file} has SHA-256 {actual}, the frozen record {digest}")
    problems += _session_problems(session, record, there, path, frozen_sha256, files, tokenizer, vote_mode)
    server = _object(session, "server", there)
    for key, want in EVALUATED_SERVER.items():
        if server.get(key) != want:
            problems.append(f"server.{key}: {there} holds {server.get(key)!r}, the evaluated server "
                            f"{want!r}")
    idle = _field(session, "gpu.idle_max_mib", there)
    if problems:
        raise FrozenMismatch(problems)
    return FrozenSystem(record=record, family=family, settings=settings, tokenizer=tokenizer,
                        base_path=files["base"], adapter_path=files["adapter"], vote_mode=vote_mode,
                        rules=True, server=ServerSettings(port=server["port"], ctx_size=server["ctx_size"],
                                              timeout_s=server["timeout_s"],
                                              llama_cpp_build=server["llama_cpp_build"], idle_max_mib=idle),
                        frozen_path=where, frozen_sha256=frozen_sha256, config_path=there,
                        config_sha256=hash_file(config))


def _session_problems(session: dict, record: dict, there: str, path: Path, frozen_sha256: str,
                      files: dict, tokenizer: str, vote_mode: object) -> list[str]:
    """Where the session configuration's final system and its adapter entry disagree with the record (module
    docstring, 6)."""
    problems = []
    options = _object(session, f"systems.{SYSTEM}", there)
    frozen = _object(session, f"systems.{SYSTEM}.frozen", there)
    if not (isinstance(frozen.get("path"), str) and same_path(frozen["path"], path)):
        problems.append(f"systems.{SYSTEM}.frozen.path: {there} names {frozen.get('path')!r}, not "
                        f"{path.as_posix()}")
    if frozen.get("sha256") != frozen_sha256:
        problems.append(f"systems.{SYSTEM}.frozen.sha256: {there} holds {frozen.get('sha256')!r}, "
                        f"{path.as_posix()} has {frozen_sha256}")
    for key, want in (("votes", record.get("votes")), ("vote_mode", vote_mode), ("rules", True),
                      ("glossary", None)):
        if options.get(key) != want:
            problems.append(f"systems.{SYSTEM}.{key}: {there} holds {_short(options.get(key))}, the frozen "
                            f"record {_short(want)}")
    adapter = _object(session, f"adapters.{SYSTEM}", there)
    for key, want in (("gguf", files["adapter"]), ("model", files["base"]), ("tokenizer", tokenizer)):
        if not (isinstance(adapter.get(key), str) and same_path(adapter[key], want)):
            problems.append(f"adapters.{SYSTEM}.{key}: {there} names {adapter.get(key)!r}, the frozen record "
                            f"{want}")
    if adapter.get("sha256") != record["adapter"]["sha256"]:
        problems.append(f"adapters.{SYSTEM}.sha256: {there} holds {adapter.get('sha256')!r}, the frozen "
                        f"record {record['adapter']['sha256']}")
    block = adapter.get("generation") or {}
    generation = record["generation"]
    for key, want in (("mode", generation["mode"]), ("reasoning_budget", generation["reasoning_budget"]),
                      ("base_seed", generation["base_seed"]), ("decoding", SAMPLED)):
        if block.get(key) != want:
            problems.append(f"adapters.{SYSTEM}.generation.{key}: {there} holds {block.get(key)!r}, the "
                            f"frozen record {want!r}")
    return problems


def identity_problems(info: dict, frozen: FrozenSystem, props: dict | None) -> list[str]:
    """Where a run's recorded identity disagrees with the frozen system (module docstring); [] when it
    agrees."""
    record = frozen.record
    problems = []
    for key in ("family", "generation", "vote_mode", "rules", "glossary"):
        if info.get(key) != record.get(key):
            problems.append(f"{key}: the run records {_short(info.get(key))}, the frozen record "
                            f"{_short(record.get(key))}")
    if info.get("backend") != LLAMA_CPP:
        problems.append(f"backend: the run records {info.get('backend')!r}, not {LLAMA_CPP}")
    if info.get("lora_scale") != LORA_SCALE:
        problems.append(f"lora_scale: the run records {info.get('lora_scale')!r}, not {LORA_SCALE:g}")
    folder = info.get("tokenizer")
    if not (isinstance(folder, str) and same_path(folder, frozen.tokenizer)):
        problems.append(f"tokenizer: the run records {folder!r}, the frozen record {frozen.tokenizer}")
    adapters = info.get("adapters")
    if not (isinstance(adapters, list) and len(adapters) == 1 and isinstance(adapters[0], dict)):
        problems.append(f"adapters: the run records {_short(adapters)}, not exactly one adapter")
    else:
        listed = adapters[0]
        if not (isinstance(listed.get("path"), str) and same_path(listed["path"], frozen.adapter_path)):
            problems.append(f"adapters: the server loaded {listed.get('path')!r}, not {frozen.adapter_path}")
        if listed.get("sha256") != record["adapter"]["sha256"]:
            problems.append(f"adapters: the loaded adapter has SHA-256 {listed.get('sha256')!r}, the frozen "
                            f"record {record['adapter']['sha256']}")
    for key in ("prompt_version", "prompt_hash", "schema_hash"):
        if key in info and info[key] != EVALUATED[key]:
            problems.append(f"{key}: the run records {info[key]!r}, the evaluated system {EVALUATED[key]!r}")
    if not isinstance(props, dict):
        problems.append("server: no /props recorded, so its slots, build and model cannot be checked")
        return problems
    if props.get("total_slots") != SLOTS:
        problems.append(f"server: {props.get('total_slots')!r} slots (/props), not {SLOTS}")
    if props.get("build_info") != frozen.server.llama_cpp_build:
        problems.append(f"server: build {props.get('build_info')!r}, the evaluated build "
                        f"{frozen.server.llama_cpp_build!r}")
    model = props.get("model_path")
    if not (isinstance(model, str) and same_path(model, frozen.base_path)):
        problems.append(f"server: model {model!r}, not the frozen base {frozen.base_path}")
    settings = props.get("default_generation_settings")
    n_ctx = settings.get("n_ctx") if isinstance(settings, dict) else None
    if n_ctx is not None and n_ctx != frozen.server.ctx_size:
        problems.append(f"server: context size {n_ctx!r}, not {frozen.server.ctx_size}")
    return problems


def tree_problems(git_dirty: object) -> list[str]:
    """[] when a run recorded a clean working tree (git_dirty false), else why it is not known to be one
    (module docstring)."""
    if git_dirty is False:
        return []
    if git_dirty is True:
        return [("git_dirty: the working tree had uncommitted changes, so the code that ran is not the "
                 "commit the run names")]
    return [f"git_dirty: the run recorded {git_dirty!r}, not whether its working tree was clean"]

"""The bake-off tooling offline (research_agent.eval.bakeoff, scripts/bakeoff.py; docs/m7-计划.md §4,
docs/m7-底座对比预登记.md): the configuration, the variants per thinking kind, the commands (parsed by the
real scripts' parsers), the default runner's log files, the preflight over fake probes, the seven smoke steps
over a fake runner and a scripted llama-server, the pilot and the evaluation as subprocess plans, the gate
JSON, the preregistered decision rule with its extension, tie-break and hybrid branches, the command line, and
the winner's trained-adapter fidelity (report only) over fake backends. Nothing here trains, starts
llama-server or reads the GPU; the runner's test runs this interpreter."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from support.bakeoff_fakes import FakeClock, FakeExport, FakeProbes, FakeRunner, ServedLlama, script_args
from support.family_fakes import TURN_END_ID
from support.llama_fakes import FakeLlama, Scripted, fake_loaded_family

from research_agent.agent.prompts import PROMPT_HASH, system_prompt
from research_agent.canonical import canonical_json
from research_agent.eval import bakeoff
from research_agent.eval.bakeoff import (
    GIB_PER_GB,
    STEPS,
    BakeoffError,
    Config,
    PreflightTimeout,
    Tools,
    aggregate,
    b0_variant,
    budget_of,
    candidate_gate,
    decide,
    eval_command,
    export_command,
    golden_command,
    load_config,
    parse_pytest,
    parse_summary,
    plan_lines,
    preflight,
    recorded,
    run_evaluate,
    run_pilot,
    run_script,
    run_smoke,
    score_command,
    smoke_plan,
    smoke_status,
    stale_records,
    summarize_scores,
    tool_round_trip,
    train_command,
    training_rows,
    variants_of,
    wait_until_clear,
)
from research_agent.eval.fidelity import AdapterNotLoaded, Backends, Module, Sequence, write_safetensors
from research_agent.eval.lock import sha256_of, write_lock
from research_agent.family.core import GREEDY, Decoding, Mode
from research_agent.tools.schemas import MODEL_TOOLS, SCHEMA_HASH

REPO = Path(__file__).resolve().parents[2]
SYSTEM = {"role": "system", "content": system_prompt("2026-10-10")}
SEARCH = {"name": "search_arxiv", "arguments": {"query": "hallucination", "max_results": 5},
          "arguments_json": "{\"query\": \"hallucination\", \"max_results\": 5}"}
CALL = ('<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "hallucination", "max_results": 5}}\n'
        '</tool_call>')


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def item(rid: str, user: str, expected: dict, type_: str, *, handwritten: bool = False) -> dict:
    source = {"handwritten": True} if handwritten else {"request": rid, "step": 1}
    return {"id": rid, "messages": [SYSTEM, {"role": "user", "content": user}], "expected": expected,
            "type": type_, "lang": "en", "family": f"f/{type_}", "split": "dev", "slots": {},
            "source": source}


ITEMS = [item("q-1", "Find hallucination papers.", {"content": "", "tool_calls": [SEARCH]}, "call"),
         item("q-2", "Thanks, that's all for today.", {"content": "See you!", "tool_calls": []}, "no_call",
              handwritten=True),
         item("q-3", "Find hallucination papers, please.", {"content": "", "tool_calls": [SEARCH]}, "call")]

STYLES = {"none": "none", "only": "only", "hybrid": "hybrid"}  # a candidate's thinking kind -> fake style
SAMPLING = Decoding(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
VARIANTS = {
    "none": {"nothink": ["--mode", "no_think"]},
    "only": {"think-s0": ["--mode", "think", "--seed", "0", "--reasoning-budget", "{R}"],
             "think-s1": ["--mode", "think", "--seed", "1", "--reasoning-budget", "{R}"],
             "think-s2": ["--mode", "think", "--seed", "2", "--reasoning-budget", "{R}"],
             "think-greedy": ["--greedy", "--reasoning-budget", "{R}"]},
    "hybrid": {"think-s0": ["--mode", "think", "--seed", "0", "--reasoning-budget", "{R}"],
               "think-s1": ["--mode", "think", "--seed", "1", "--reasoning-budget", "{R}"],
               "think-s2": ["--mode", "think", "--seed", "2", "--reasoning-budget", "{R}"],
               "think-greedy": ["--greedy", "--reasoning-budget", "{R}"],
               "nothink": ["--mode", "no_think"]}}
CONFIG = {
    "run_root": "runs/bakeoff/test", "results": "results/bakeoff-test.json",
    "data": ["data/raw/synth/synth-test.jsonl"], "overlap_report": "data/synth/overlap-test.json",
    "dev_items": "data/eval/dev.jsonl", "reasoning": ["data/raw/synth/reasoning-test.jsonl"],
    "schedule": {"stop_step": 274, "extend_step": 548, "save_steps": 274,
                 "max_length": {"none": 4096, "only": 5120, "hybrid": 5120}, "qlora": True},
    "smoke": {"steps": 20, "items": 2, "positions": 4, "forced_budget": 32},
    "thresholds": {"template_rate": 0.005, "peak_reserved_gb": 7.0, "ft3_hours": 60.0, "tie": 3,
                   "hybrid_margin": 5, "gpu_used_mib": 1500, "ram_gb": 4.0},
    "ft3": {"data_factor": 1.45, "epochs": 2},
    "variants": VARIANTS,
    "primary": {"none": {"mode": "no_think", "variants": ["nothink"]},
                "only": {"mode": "think", "variants": ["think-s0", "think-s1", "think-s2"]},
                "hybrid": {"mode": "think", "variants": ["think-s0", "think-s1", "think-s2"]}},
    "b0": {"no_think": ["--mode", "no_think"],
           "think": ["--mode", "think", "--seed", "0", "--reasoning-budget", "{R}"]},
    "reference": {"key": "c0", "name": "Qwen3-4B", "model": "Qwen/Qwen3-4B", "thinking": "hybrid",
                  "mode": "no_think", "score": "runs/m7/g2-dev.score.json", "source": "runs/m7/g2.json",
                  "train": "runs/train/ref-pilot", "P": 2},
    "candidates": {
        "c1": {"name": "none-model", "hf": "models/none-model", "gguf": "models/none-model-Q8_0.gguf",
               "family": "fake-none", "thinking": "none", "reasoning_budget": None,
               "model": "Fake/none-model", "revision": "abc", "exempt": ["S0", "S1", "S2", "S3", "S4"],
               "exempt_reason": "the same architecture and tokenizer as the reference",
               "record_only": ["S5", "S6"]},
        "c2": {"name": "only-model", "hf": "models/only-model", "gguf": "models/only-model-Q8_0.gguf",
               "family": "fake-only", "thinking": "only", "reasoning_budget": 16,
               "model": "Fake/only-model", "revision": "def", "exempt": [], "exempt_reason": None},
        "c3": {"name": "hybrid-model", "hf": "models/hybrid-model", "gguf": "models/hybrid-model-Q8_0.gguf",
               "family": None, "thinking": "hybrid", "reasoning_budget": None,
               "model": "Fake/hybrid-model", "revision": None, "exempt": [], "exempt_reason": None}}}


def agent_record(rid: str, user: str, answer: dict) -> dict:
    """An agent-history record as research_agent.data.assemble writes one: one assistant step."""
    return {"id": rid, "history": "agent", "schema_hash": SCHEMA_HASH, "prompt_hash": PROMPT_HASH,
            "tools": MODEL_TOOLS, "messages": [SYSTEM, {"role": "user", "content": user}, answer]}


RECORDS = [
    agent_record("req-1", "Find hallucination papers.", {
        "role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "search_arxiv",
                                                          "arguments": SEARCH["arguments_json"]}}]}),
    agent_record("req-2", "Thanks.", {"role": "assistant", "content": "See you!"}),
    agent_record("req-3", "Get the paper.", {
        "role": "assistant", "content": "", "tool_calls": [
            {"id": "c2", "type": "function", "function": {"name": "get_paper",
                                                          "arguments": '{"arxiv_id": "2609.01234"}'}},
            {"id": "c3", "type": "function", "function": {"name": "search_arxiv",
                                                          "arguments": "not json at all"}}]}),
]


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def score_file(path: Path, successes: dict[str, bool]) -> Path:
    """A scripts/score.py summary: the per-item successes over ITEMS."""
    scores = [{"id": i["id"], "type": i["type"], "success": successes.get(i["id"], False)} for i in ITEMS]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"summary": {"n": len(ITEMS)}, "scores": scores}, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repository root as the current folder: the configuration, the data file, the locked development
    set, the fake model folders with their GGUFs, the reference's files and a llama-server binary."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "bakeoff-test.json").write_text(json.dumps(CONFIG, indent=2), encoding="utf-8")
    write_jsonl(tmp_path / "data" / "raw" / "synth" / "synth-test.jsonl", RECORDS)
    write_jsonl(tmp_path / "data" / "raw" / "synth" / "reasoning-test.jsonl", [])
    (tmp_path / "data" / "synth").mkdir(parents=True)
    (tmp_path / "data" / "synth" / "overlap-test.json").write_text("{}", encoding="utf-8")
    dev = write_jsonl(tmp_path / "data" / "eval" / "dev.jsonl", ITEMS)
    write_lock(dev)
    for name in ("none-model", "only-model", "hybrid-model"):
        folder = tmp_path / "models" / name
        folder.mkdir(parents=True)
        (folder / "config.json").write_text('{"architectures": ["FakeForCausalLM"]}', encoding="utf-8")
        (tmp_path / "models" / f"{name}-Q8_0.gguf").write_bytes(f"GGUF {name}".encode())
    score_file(tmp_path / "runs" / "m7" / "g2-dev.score.json", {"q-1": True, "q-2": True})
    (tmp_path / "runs" / "m7" / "g2.json").write_text(json.dumps({"dev_successes": 2, "dev_n": 3}),
                                                       encoding="utf-8")
    pilot_outputs(tmp_path / "runs" / "train" / "ref-pilot", tokens=16_000_000)  # the reference's G2 run
    (tmp_path / "llama-server.exe").write_bytes(b"")
    (tmp_path / "llama.cpp").mkdir()
    return tmp_path


@pytest.fixture
def config(repo):
    return load_config(Path("configs/bakeoff-test.json"))


def fake_family_loader(**changes):
    """tools.load_family: the fake family of the folder's style (its name's first piece), the thinking
    styles sampling their thinking as the Qwen3 families do."""
    def load(folder: Path, name: str | None):
        style = STYLES[Path(folder).name.split("-")[0]]
        fields = dict(changes)
        if style != "none":
            fields.setdefault("decoding", {Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY})
        return fake_loaded_family(Path(folder), style, **fields)

    return load


def no_backends(config, candidate, tools, folder):
    raise RuntimeError("no fidelity backends in this test")


def tools_of(runner: FakeRunner | None = None, *, served: ServedLlama | None = None,
             probes: FakeProbes | None = None, clock: FakeClock | None = None, **changes) -> Tools:
    runner = runner or FakeRunner()
    served = served or ServedLlama(FakeLlama([]))
    clock = clock or FakeClock()
    fields = {
        "run": runner, "start": runner.start, "popen": served.popen, "client": served.client,
        "load_family": fake_family_loader(), "probes": probes or FakeProbes(),
        "count_gguf_tensors": lambda path: 4, "fidelity_backends": no_backends,
        "round_trip": tool_round_trip,
        "sleep": clock.sleep, "clock": clock, "python": "py.exe", "llama_server": Path("llama-server.exe"),
        "llama_cpp_src": Path("llama.cpp"), "wait_interval": 60.0, "wait_limit": 600.0, "port": 8080,
        "timeout": 5.0}
    return Tools(**{**fields, **changes})


# ---- the configuration and the variants --------------------------------------------------------------


def test_load_config_reads_the_candidates_and_the_shared_conditions(config):
    assert list(config.candidates) == ["c1", "c2", "c3"] and config.run_root == Path("runs/bakeoff/test")
    c2 = config.candidate("c2")
    assert (c2.key, c2.name, c2.hf, c2.gguf, c2.family) == ("c2", "only-model", "models/only-model",
                                                           "models/only-model-Q8_0.gguf", "fake-only")
    assert c2.thinking == "only" and c2.thinks and c2.primary_mode == "think" and c2.reasoning_budget == 16
    assert config.max_length(c2) == 5120 and config.max_length(config.candidate("c1")) == 4096
    c1 = config.candidate("c1")
    assert c1.exempt == ("S0", "S1", "S2", "S3", "S4") and not c1.thinks
    assert c1.record_only == ("S5", "S6") and c2.record_only == ()
    assert config.primary_variants(c2) == ["think-s0", "think-s1", "think-s2"]
    assert config.primary_variants(config.candidate("c1")) == ["nothink"]
    assert config.folder("c2") == Path("runs/bakeoff/test/c2")
    for key in ("c0", "c9"):
        with pytest.raises(BakeoffError, match=key):
            config.candidate(key)


@pytest.mark.parametrize(("change", "message"), [
    ({"candidates": {"c2": {**CONFIG["candidates"]["c2"], "thinking": "maybe"}}}, "thinking"),
    ({"candidates": {"c2": {**CONFIG["candidates"]["c2"], "exempt": ["S7"]}}}, "S7"),
    ({"candidates": {"c2": {**CONFIG["candidates"]["c2"], "record_only": ["S8"]}}}, "S8"),
    ({"primary": {**CONFIG["primary"], "only": {"mode": "think", "variants": ["think-s9"]}}}, "think-s9"),
    ({"schedule": {**CONFIG["schedule"], "max_length": {"none": 4096}}}, "max_length"),
    ({"candidates": {"c0": CONFIG["candidates"]["c2"]}}, "c0"),
    # the hybrid rule reads a hybrid family's nothink greedy run
    ({"variants": {**VARIANTS, "hybrid": {k: v for k, v in VARIANTS["hybrid"].items() if k != "nothink"}}},
     "nothink"),
    # every path is repository-relative: the records hold no machine path (module docstring)
    ({"run_root": "C:/runs/bakeoff"}, "run_root"),
    ({"results": "\\\\server\\share\\bakeoff.json"}, "results"),
    ({"dev_items": "../dev.jsonl"}, "dev_items"),
    ({"data": ["/data/raw/*.jsonl"]}, "data"),
    ({"reasoning": ["data/raw/../../reasoning.jsonl"]}, "reasoning"),
    ({"candidates": {"c2": {**CONFIG["candidates"]["c2"], "hf": "D:/models/only-model"}}}, "hf"),
    ({"candidates": {"c2": {**CONFIG["candidates"]["c2"], "gguf": "/models/only.gguf"}}}, "gguf"),
    ({"reference": {**CONFIG["reference"], "train": "../runs/train/ref-pilot"}}, "train"),
    ({"reference": {**CONFIG["reference"], "score": "C:\\runs\\g2.score.json"}}, "score")])
def test_a_malformed_configuration_is_refused(repo, change, message):
    path = repo / "configs" / "bad.json"
    path.write_text(json.dumps({**CONFIG, **change}), encoding="utf-8")
    with pytest.raises(BakeoffError, match=message):
        load_config(path)


def test_variants_follow_the_thinking_kind_and_carry_the_reasoning_budget(config):
    assert variants_of(config, config.candidate("c1"), None) == [("nothink", ["--mode", "no_think"])]
    only = variants_of(config, config.candidate("c2"), 16)
    assert [name for name, _ in only] == ["think-s0", "think-s1", "think-s2", "think-greedy"]
    assert only[1][1] == ["--mode", "think", "--seed", "1", "--reasoning-budget", "16"]
    assert only[3][1] == ["--greedy", "--reasoning-budget", "16"]
    hybrid = variants_of(config, config.candidate("c3"), 1280)
    assert [name for name, _ in hybrid] == ["think-s0", "think-s1", "think-s2", "think-greedy", "nothink"]
    assert b0_variant(config, config.candidate("c2"), 16) == ["--mode", "think", "--seed", "0",
                                                            "--reasoning-budget", "16"]
    assert b0_variant(config, config.candidate("c1"), None) == ["--mode", "no_think"]
    with pytest.raises(BakeoffError, match="reasoning budget"):
        variants_of(config, config.candidate("c3"), None)
    assert budget_of(config, config.candidate("c2"), None) == 16  # the configuration's
    assert budget_of(config, config.candidate("c3"), {"steps": {"S2": {"reasoning_budget": 1280}}}) == 1280
    assert budget_of(config, config.candidate("c3"), None) is None


# ---- the commands, parsed by the real scripts' parsers ---------------------------------------------------


def test_the_commands_parse_with_the_real_scripts_and_carry_the_schedule(config):
    tools = tools_of()
    c2, c1 = config.candidate("c2"), config.candidate("c1")
    train = train_command(config, c2, tools, out=Path("runs/bakeoff/test/c2/train"), stop_step=274)
    assert train[:2] == ["py.exe", "scripts/train.py"]
    args = load_script("train").build_parser().parse_args(script_args(train))
    assert args.data == ["data/raw/synth/synth-test.jsonl"] and args.model == "models/only-model"
    assert args.family == "fake-only" and args.qlora and args.stop_at_step == 274 and args.save_steps == 274
    assert args.max_length == 5120 and args.reasoning == ["data/raw/synth/reasoning-test.jsonl"]
    assert args.overlap_report == "data/synth/overlap-test.json" and args.out == "runs/bakeoff/test/c2/train"
    assert not args.resume and not args.dry_run and args.max_steps is None
    smoke = train_command(config, c1, tools, out=Path("s"), max_steps=20, smoke_longest=True)
    args = load_script("train").build_parser().parse_args(script_args(smoke))
    assert args.max_steps == 20 and args.smoke_longest and args.max_length == 4096 and args.reasoning == []
    assert args.family == "fake-none" and args.stop_at_step is None and "--family" in smoke
    dry = train_command(config, config.candidate("c3"), tools, out=Path("d"), dry_run=True)
    args = load_script("train").build_parser().parse_args(script_args(dry))
    assert args.dry_run and args.family is None and "--family" not in dry  # no name configured: resolved
    resumed = train_command(config, c2, tools, out=Path("t"), stop_step=548, resume=True)
    args = load_script("train").build_parser().parse_args(script_args(resumed))
    assert args.resume and args.stop_at_step == 548
    export = export_command(config, c2, tools, Path("t/trainer/checkpoint-274"),
                            Path("t/gguf/checkpoint-274-f16.gguf"))
    args = load_script("export_adapter").build_parser().parse_args(script_args(export))
    assert args.adapter == "t/trainer/checkpoint-274" and args.base == "models/only-model"
    assert args.outfile == "t/gguf/checkpoint-274-f16.gguf" and args.llama_cpp_src == "llama.cpp"
    evaluate = eval_command(config, c2, tools, system="ft",
                            out=Path("runs/bakeoff/test/c2/dev-think-s0.jsonl"),
                            args=["--mode", "think", "--seed", "0", "--reasoning-budget", "16"])
    args = load_script("run_eval").build_parser().parse_args(script_args(evaluate))
    assert args.system == "ft" and args.signals and args.tokenizer == "models/only-model"
    assert args.mode == "think" and args.seed == 0 and args.reasoning_budget == 16
    assert args.server == "http://127.0.0.1:8080" and args.items.as_posix() == "data/eval/dev.jsonl"
    score = score_command(config, tools, Path("r.jsonl"), Path("r.score.json"))
    args = load_script("score").build_parser().parse_args(script_args(score))
    assert args.results.as_posix() == "r.jsonl" and args.items.as_posix() == "data/eval/dev.jsonl"
    assert args.out.as_posix() == "r.score.json"
    golden = golden_command(c2, tools)
    assert golden == ["py.exe", "-m", "pytest", "tests/family/test_golden.py", "-q", "-p", "no:cacheprovider",
                      "-k", "only-model"]
    assert all("\\" not in piece for cmd in (train, smoke, export, evaluate, score) for piece in cmd)


# ---- the default runner ------------------------------------------------------------------------------


def test_run_script_keeps_a_log_pair_per_stem_and_returns_the_output(tmp_path):
    # a variant's run_eval and its score.py share a stem up to their last piece: each keeps its own logs
    scored = run_script([sys.executable, "-c", "import sys; print('scored'); print('warn', file=sys.stderr)"],
                        log=tmp_path / "logs" / "dev-think-s0.score")
    evaluated = run_script([sys.executable, "-c", "print('evaluated')"],
                           log=tmp_path / "logs" / "dev-think-s0.eval")
    assert scored.returncode == 0 and scored.stdout.strip() == "scored" and scored.stderr.strip() == "warn"
    assert evaluated.returncode == 0 and evaluated.stdout.strip() == "evaluated"
    logs = tmp_path / "logs"
    assert (logs / "dev-think-s0.score.out.log").read_text(encoding="utf-8").strip() == "scored"
    assert (logs / "dev-think-s0.score.err.log").read_text(encoding="utf-8").strip() == "warn"
    assert (logs / "dev-think-s0.eval.out.log").read_text(encoding="utf-8").strip() == "evaluated"
    assert sorted(p.name for p in logs.iterdir()) == [
        "dev-think-s0.eval.err.log", "dev-think-s0.eval.out.log", "dev-think-s0.score.err.log",
        "dev-think-s0.score.out.log"]
    # the interpreter as the records name it, relative to the current folder in POSIX form: the runner
    # launches it all the same (Windows' CreateProcess takes no relative path with forward slashes)
    relative = Path(os.path.relpath(sys.executable)).as_posix()
    assert not Path(relative).is_absolute()
    launched = run_script([relative, "-c", "import sys; print(sys.executable)"], log=tmp_path / "rel")
    assert launched.returncode == 0
    assert Path(launched.stdout.strip()).resolve() == Path(sys.executable).resolve()


# ---- the preflight -----------------------------------------------------------------------------------


def test_preflight_reads_the_three_probes_against_the_thresholds(config):
    thresholds = config.thresholds
    clear = preflight(FakeProbes(gpu=[1100], ram=[8.3], ac=[True]), thresholds)
    assert clear == {"gpu_used_mib": 1100, "available_ram_gb": 8.3, "on_ac_power": True, "ok": True,
                     "reasons": []}
    busy = preflight(FakeProbes(gpu=[6600], ram=[3.5], ac=[False]), thresholds)
    assert busy["ok"] is False and len(busy["reasons"]) == 3
    assert "6600" in busy["reasons"][0] and "3.5" in busy["reasons"][1] and "battery" in busy["reasons"][2]
    unknown = preflight(FakeProbes(gpu=[None], ram=[None], ac=[None]), thresholds)
    assert unknown["ok"] is False and all("unknown" in r for r in unknown["reasons"])


def test_the_preflight_bounds_are_inclusive(config):
    thresholds = config.thresholds  # at most 1500 MiB in use, at least 4.0 GB available
    assert preflight(FakeProbes(gpu=[1500], ram=[4.0], ac=[True]), thresholds)["ok"] is True
    over = preflight(FakeProbes(gpu=[1501], ram=[4.0], ac=[True]), thresholds)
    assert over["ok"] is False and over["reasons"] == [
        "the GPU holds 1501 MiB, over 1500 MiB: another process uses it"]
    under = preflight(FakeProbes(gpu=[1500], ram=[3.99], ac=[True]), thresholds)
    assert under["ok"] is False and under["reasons"] == ["the available RAM is 3.99 GB, under 4.0 GB"]


def test_wait_until_clear_polls_until_the_machine_is_free_or_the_limit_passes(config):
    probes = FakeProbes(gpu=[6600, 6600, 1000], ram=[8.0], ac=[True])
    clock = FakeClock()
    tools = tools_of(probes=probes, clock=clock)
    state = wait_until_clear(tools, config.thresholds)
    assert state["ok"] and state["gpu_used_mib"] == 1000 and clock.slept == [60.0, 60.0]
    tools = tools_of(probes=FakeProbes(gpu=[6600]), clock=FakeClock(), wait_limit=150.0)
    with pytest.raises(PreflightTimeout, match="150"):
        wait_until_clear(tools, config.thresholds)
    assert tools.clock.slept == [60.0, 60.0, 60.0]


# ---- the smoke steps ---------------------------------------------------------------------------------


def smoke_of(folder: Path) -> dict:
    return json.loads((folder / "smoke.json").read_text(encoding="utf-8"))


def test_s0_records_the_family_the_gguf_hash_and_the_remote_code(config, repo):
    folder = repo / "models" / "only-model"
    (folder / "modeling_only.py").write_text("print('vendor code')\n", encoding="utf-8")
    pinned = {"modeling_only.py": sha256_of(folder / "modeling_only.py")}
    tools = tools_of(load_family=fake_family_loader(trust_remote_code=True, remote_files=pinned))
    smoke = run_smoke(config, config.candidate("c2"), tools, steps=["S0"])
    s0 = smoke["steps"]["S0"]
    assert s0["passed"] is True and s0["family"]["name"] == "fake-only" and s0["family"]["remote"] is True
    assert s0["gguf"] == "models/only-model-Q8_0.gguf"
    assert s0["gguf_sha256"] == sha256_of(repo / "models" / "only-model-Q8_0.gguf")
    assert s0["remote_code_sha256"] == {"modeling_only.py": sha256_of(folder / "modeling_only.py")}
    assert smoke_of(Path("runs/bakeoff/test/c2"))["steps"]["S0"] == s0
    assert smoke["candidate"] == "c2" and set(smoke["steps"]) == {"S0"} and smoke["passed"] is None
    # a family that does not resolve, or a GGUF that is not here, fails the step with the reason
    smoke = run_smoke(config, config.candidate("c2"), tools_of(load_family=fake_family_loader(
        name="fake-hybrid")), steps=["S0"])
    assert smoke["steps"]["S0"]["passed"] is False and "fake-only" in smoke["steps"]["S0"]["error"]
    (repo / "models" / "only-model-Q8_0.gguf").unlink()
    smoke = run_smoke(config, config.candidate("c2"), tools_of(), steps=["S0"])
    assert smoke["steps"]["S0"]["passed"] is False and "GGUF" in smoke["steps"]["S0"]["error"]


@pytest.mark.parametrize(("code", "text", "passed", "counts"), [
    (0, "........\n9 passed, 2 skipped in 0.41s\n", True, {"passed": 9, "failed": 0, "skipped": 2,
                                                            "errors": 0}),
    (1, "F.\n1 failed, 8 passed in 0.5s\n", False, {"passed": 8, "failed": 1, "skipped": 0, "errors": 0}),
    (5, "no tests ran in 0.03s\n", False, {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}),
    (0, "9 skipped in 0.1s\n", False, {"passed": 0, "failed": 0, "skipped": 9, "errors": 0})])
def test_s1_runs_the_folders_golden_tests_and_needs_a_pass(config, code, text, passed, counts):
    runner = FakeRunner({"pytest": lambda argv: (code, text)})
    smoke = run_smoke(config, config.candidate("c2"), tools_of(runner), steps=["S1"])
    s1 = smoke["steps"]["S1"]
    assert s1["passed"] is passed and s1["counts"] == counts and s1["returncode"] == code
    (cmd,) = runner.commands
    assert cmd[-2:] == ["-k", "only-model"] and s1["command"] == cmd
    assert parse_pytest(text) == counts


SUMMARY = {"samples": 100, "tokens": {"min": 10, "mean": 50, "max": 90, "total": 5000},
           "dropped": {"over_length": 0, "after_repair": 0},
           "modes": {"think": 30, "no_think": 70, "think_fallback": 1},
           "refused": {"template": 0, "boundary": 0, "bos": 0},
           "reasoning": {"budget": 1280, "tokens": {"n": 30, "p99": 1194}}, "sample_keys": None,
           "total_optimizer_steps": 2192, "save_cadence": {"every_steps": 274},
           "family": {"name": "fake-only"}}


def dry_run_text(summary: dict) -> str:
    return "data: records 3, samples 3\nmodes: think 1\nsummary: " + json.dumps(summary) + "\n"


def test_s2_reads_the_dry_runs_summary_line_and_computes_the_tool_round_trip(config):
    runner = FakeRunner({"train.py": lambda argv: (0, dry_run_text(SUMMARY))})
    smoke = run_smoke(config, config.candidate("c3"), tools_of(runner), steps=["S2"])
    s2 = smoke["steps"]["S2"]
    assert s2["passed"] is True and s2["samples"] == 100 and s2["tokens_per_epoch"] == 5000
    assert s2["refused"] == {"template": 0, "boundary": 0, "bos": 0} and s2["template_rate"] == 0.0
    assert s2["modes"] == SUMMARY["modes"] and s2["reasoning_budget"] == 1280
    assert s2["reasoning_tokens"] == SUMMARY["reasoning"]["tokens"]
    # the round trip over the agent-history records: req-1 and req-3's JSON call go through the fake
    # family's template and the format reads them back; req-3's raw-text call is not a round-trip target
    assert s2["round_trip"] == {"targets": 1, "round_trips": 1, "raw_arguments": 1, "refused": 0,
                                "records": 3, "steps": 3, "rate": 1.0}
    (cmd,) = runner.commands
    args = load_script("train").build_parser().parse_args(script_args(cmd))
    assert args.dry_run and args.max_length == 5120 and args.out == "runs/bakeoff/test/c3/smoke/dryrun"
    assert parse_summary(dry_run_text(SUMMARY)) == SUMMARY
    with pytest.raises(ValueError, match="summary"):
        parse_summary("no summary line here\n")


def test_s2_passes_only_when_every_bound_holds(config):
    def smoke_with(summary: dict, round_trip=None) -> dict:
        runner = FakeRunner({"train.py": lambda argv: (0, dry_run_text(summary))})
        tools = tools_of(runner, round_trip=round_trip or (lambda family, files: {
            "targets": 10, "round_trips": 10, "raw_arguments": 0, "refused": 0, "records": 5, "steps": 10,
            "rate": 1.0}))
        return run_smoke(config, config.candidate("c2"), tools, steps=["S2"])["steps"]["S2"]

    assert smoke_with(SUMMARY)["passed"] is True
    boundary = smoke_with({**SUMMARY, "refused": {"template": 0, "boundary": 1, "bos": 0}})
    assert boundary["passed"] is False and "boundary" in boundary["failures"][0]
    bos = smoke_with({**SUMMARY, "refused": {"template": 0, "boundary": 0, "bos": 1}})  # the bound is 0
    assert bos["passed"] is False and bos["failures"] == ["bos refusals: 1 (the bound is 0)"]
    # 6 template refusals in 1001 steps are 0.60%, over the preregistered 0.5%
    over = smoke_with({**SUMMARY, "samples": 995, "refused": {"template": 6, "boundary": 0, "bos": 0}})
    assert over["passed"] is False and over["template_rate"] == 6 / 1001
    assert over["failures"] == ["template refusals: 6 of 1001 steps, 0.60%, over the preregistered 0.5%"]
    # 5 in 1000 are exactly 0.5%, within the bound. The preregistration then leaves those samples out of every
    # candidate's data, which the tooling cannot do (scripts/train.py stops at a refused sample): S2 is
    # blocked, neither passed nor failed, and says why; the candidate is not excluded for it (the gate waits)
    within = smoke_with({**SUMMARY, "samples": 995, "refused": {"template": 5, "boundary": 0, "bos": 0}})
    assert within["passed"] is None and within["failures"] == [] and within["template_rate"] == 0.005
    assert within["blocked"].startswith("template refusals: 5 of 1000 steps, 0.50%, within the preregistered "
                                        "0.5%")
    assert "scripts/train.py" in within["blocked"]
    assert "runs/bakeoff/test/c2/smoke/S2.dryrun.out.log" in within["blocked"]
    # a failure besides the refusals within the bound: failed, not blocked
    both = smoke_with({**SUMMARY, "samples": 995, "refused": {"template": 5, "boundary": 1, "bos": 0}})
    assert both["passed"] is False and "blocked" not in both
    assert both["failures"] == ["token boundary refusals: 1 (the bound is 0)"]
    partial = smoke_with(SUMMARY, round_trip=lambda family, files: {
        "targets": 100, "round_trips": 99, "raw_arguments": 0, "refused": 0, "records": 50, "steps": 100,
        "rate": 0.99})
    assert partial["passed"] is False and partial["failures"] == [
        "MODEL_TOOLS round trip: 99 of 100 targets (99.0%), not 100%"]
    runner = FakeRunner({"train.py": lambda argv: (2, "FAIL: no sample\n")})
    failed = run_smoke(config, config.candidate("c2"), tools_of(runner), steps=["S2"])["steps"]["S2"]
    assert failed["passed"] is False and failed["returncode"] == 2 and "exited" in failed["error"]


def test_tool_round_trip_renders_each_json_call_target_through_the_family_and_reads_it_back(repo):
    family = fake_loaded_family(repo / "models" / "hybrid-model", "hybrid")
    report = tool_round_trip(family, [Path("data/raw/synth/synth-test.jsonl")])
    assert report == {"targets": 1, "round_trips": 1, "raw_arguments": 1, "refused": 0, "records": 3,
                      "steps": 3, "rate": 1.0}


def smoke_train_outputs(out: Path, *, peaks: list[float], tps: list[int], steps: int = 20) -> None:
    """What scripts/train.py --max-steps leaves behind: log.jsonl, result.json and the adapter folder."""
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"event": "start"}]
    for k, (peak, speed) in enumerate(zip(peaks, tps, strict=True), 1):
        rows.append({"step": k * 10, "loss": 1.0, "num_tokens": 1000.0 * k, "tokens_per_s": speed,
                     "peak_reserved_gib": peak, "max_steps": steps})
    write_jsonl(out / "log.jsonl", rows)
    (out / "result.json").write_text(json.dumps({"steps": steps, "peak_reserved_gb": round(
        max(peaks) * 1.073741824, 2), "tokens_per_s": 400}), encoding="utf-8")
    (out / "adapter").mkdir(exist_ok=True)
    (out / "adapter" / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"]}),
        encoding="utf-8")
    from research_agent.eval.fidelity import write_safetensors

    write_safetensors(out / "adapter" / "adapter_model.safetensors", {
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": ((2, 3), [0.0] * 6),
        "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": ((3, 2), [0.0] * 6),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_A.weight": ((2, 3), [0.0] * 6),
        "base_model.model.model.layers.0.self_attn.v_proj.lora_B.weight": ((3, 2), [0.0] * 6)})


def test_s3_trains_the_longest_samples_and_reads_the_peak_and_the_throughput(config, repo):
    def train(argv: list[str]) -> tuple[int, str]:
        out = Path(argv[argv.index("--out") + 1])
        smoke_train_outputs(out, peaks=[4.1, 4.4, 4.4], tps=[552, 449, 368])
        return 0, "trained\n"

    runner = FakeRunner({"train.py": train})
    smoke = {"candidate": "c2", "steps": {"S2": {"passed": True, "tokens_per_epoch": 16_000_000}}}
    (repo / "runs" / "bakeoff" / "test" / "c2").mkdir(parents=True)
    (repo / "runs" / "bakeoff" / "test" / "c2" / "smoke.json").write_text(json.dumps(smoke), encoding="utf-8")
    result = run_smoke(config, config.candidate("c2"), tools_of(runner), steps=["S3"])
    s3 = result["steps"]["S3"]
    assert s3["passed"] is True and s3["peak_reserved_gb"] == 4.72 and s3["tokens_per_s_median"] == 449
    assert s3["ft3_hours_estimate"] == round(16_000_000 * 1.45 * 2 / 449 / 3600, 1)
    assert s3["train"] == "runs/bakeoff/test/c2/smoke/train" and s3["steps"] == 20
    assert result["steps"]["S2"]["tokens_per_epoch"] == 16_000_000  # the earlier step is kept
    (cmd,) = runner.commands
    args = load_script("train").build_parser().parse_args(script_args(cmd))
    assert args.max_steps == 20 and args.smoke_longest and args.max_length == 5120 and args.qlora
    assert args.out == "runs/bakeoff/test/c2/smoke/train"
    # over the limit: 7.1 GB reserved
    def hot(argv: list[str]) -> tuple[int, str]:
        smoke_train_outputs(Path(argv[argv.index("--out") + 1]), peaks=[6.7, 6.7], tps=[300, 300])
        return 0, "trained hot\n"

    runner = FakeRunner({"train.py": hot})
    s3 = run_smoke(config, config.candidate("c2"), tools_of(runner), steps=["S3"])["steps"]["S3"]
    assert s3["passed"] is False and s3["peak_reserved_gb"] == 7.19 and "7.0" in s3["failures"][0]

    # the bound is inclusive: 7.0 GB reserved passes, 7.01 GB does not (result.json's peak, the trainer's own)
    def peaking(gb: float):
        def train_to_peak(argv: list[str]) -> tuple[int, str]:
            out = Path(argv[argv.index("--out") + 1])
            smoke_train_outputs(out, peaks=[6.5], tps=[300])
            result = json.loads((out / "result.json").read_text(encoding="utf-8"))
            (out / "result.json").write_text(json.dumps({**result, "peak_reserved_gb": gb}), encoding="utf-8")
            return 0, "trained\n"

        return train_to_peak

    s3 = run_smoke(config, config.candidate("c2"), tools_of(FakeRunner({"train.py": peaking(7.0)})),
                   steps=["S3"])["steps"]["S3"]
    assert s3["passed"] is True and s3["peak_reserved_gb"] == 7.0
    s3 = run_smoke(config, config.candidate("c2"), tools_of(FakeRunner({"train.py": peaking(7.01)})),
                   steps=["S3"])["steps"]["S3"]
    assert s3["passed"] is False and s3["failures"] == ["peak reserved VRAM 7.01 GB, over 7.0 GB"]

    # a smoke run again in the same folder: train.py appends to log.jsonl, and only the last run's rows count
    def again(argv: list[str]) -> tuple[int, str]:
        out = Path(argv[argv.index("--out") + 1])
        smoke_train_outputs(out, peaks=[4.1, 4.4, 4.4], tps=[552, 449, 368])
        stale = [{"event": "start"},
                 {"step": 10, "tokens_per_s": 900, "peak_reserved_gib": 6.9, "max_steps": 20}]
        log = out / "log.jsonl"
        log.write_text("".join(json.dumps(r) + "\n" for r in stale) + log.read_text(encoding="utf-8"),
                       encoding="utf-8")
        return 0, "trained again\n"

    s3 = run_smoke(config, config.candidate("c2"), tools_of(FakeRunner({"train.py": again})),
                   steps=["S3"])["steps"]["S3"]
    assert s3["tokens_per_s"] == [552, 449, 368] and s3["tokens_per_s_median"] == 449
    assert s3["peak_reserved_gib"] == 4.4
    # no S2 before it: the hour estimate needs the tokens per epoch
    (repo / "runs" / "bakeoff" / "test" / "c2" / "smoke.json").unlink()
    runner = FakeRunner({"train.py": train})
    s3 = run_smoke(config, config.candidate("c2"), tools_of(runner), steps=["S3"])["steps"]["S3"]
    assert s3["passed"] is True and s3["ft3_hours_estimate"] is None and s3["tokens_per_epoch"] is None


def test_s4_exports_the_smoke_adapter_and_counts_the_tensors(config, repo):
    smoke_train_outputs(repo / "runs" / "bakeoff" / "test" / "c2" / "smoke" / "train", peaks=[4.0], tps=[400])
    gguf_folder = repo / "runs" / "bakeoff" / "test" / "c2" / "smoke" / "train" / "gguf"
    assert not gguf_folder.exists()  # scripts/train.py writes no gguf/ folder: the export's is S4's to make
    export = FakeExport()
    counted: list[Path] = []

    def count(path: Path) -> int:
        counted.append(path)
        return 4

    runner = FakeRunner({"export_adapter.py": export})
    s4 = run_smoke(config, config.candidate("c2"), tools_of(runner, count_gguf_tensors=count),
                   steps=["S4"])["steps"]["S4"]
    assert s4["passed"] is True and s4["tensors"] == 4 and s4["modules"] == 2 and s4["expected_tensors"] == 4
    assert s4["target_modules"] == ["q_proj", "v_proj"]
    assert s4["gguf"] == "runs/bakeoff/test/c2/smoke/train/gguf/adapter-f16.gguf"
    assert s4["gguf_sha256"] == sha256_of(repo / s4["gguf"]) and counted == [Path(s4["gguf"])]
    # the converter creates no folder (C3's first S4): the folder was there when it ran
    assert export.calls == [(Path(s4["gguf"]), True)]
    (cmd,) = runner.commands
    args = load_script("export_adapter").build_parser().parse_args(script_args(cmd))
    assert args.adapter == "runs/bakeoff/test/c2/smoke/train/adapter" and args.base == "models/only-model"
    s4 = run_smoke(config, config.candidate("c2"), tools_of(runner, count_gguf_tensors=lambda p: 3),
                   steps=["S4"])["steps"]["S4"]
    assert s4["passed"] is False and "3" in s4["failures"][0] and "4" in s4["failures"][0]
    # a tensor pair more than the targeted modules fails as well: the count must be exactly 2 x modules
    s4 = run_smoke(config, config.candidate("c2"), tools_of(runner, count_gguf_tensors=lambda p: 6),
                   steps=["S4"])["steps"]["S4"]
    assert s4["passed"] is False and s4["failures"] == [
        "the GGUF holds 6 tensors, not 2 x 2 targeted modules = 4: llama.cpp dropped or added tensors"]
    runner = FakeRunner({"export_adapter.py": lambda argv: (1, "converter failed\n")})
    s4 = run_smoke(config, config.candidate("c2"), tools_of(runner), steps=["S4"])["steps"]["S4"]
    assert s4["passed"] is False and "exited" in s4["error"]


def test_s4_counts_only_the_gguf_its_own_export_wrote(config, repo):
    # a GGUF an earlier run left behind does not stand for an export that exits 0 without writing one: it
    # is removed before the export runs, and S4 fails; S5 then serves no stale smoke adapter
    smoke_train_outputs(repo / "runs" / "bakeoff" / "test" / "c2" / "smoke" / "train", peaks=[4.0], tps=[400])
    stale = repo / "runs" / "bakeoff" / "test" / "c2" / "smoke" / "train" / "gguf" / "adapter-f16.gguf"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"GGUF of an earlier smoke adapter")
    export = FakeExport(written=False)
    s4 = run_smoke(config, config.candidate("c2"), tools_of(FakeRunner({"export_adapter.py": export})),
                   steps=["S4"])["steps"]["S4"]
    assert s4["passed"] is False and s4["returncode"] == 0
    outfile = "runs/bakeoff/test/c2/smoke/train/gguf/adapter-f16.gguf"
    assert s4["error"] == f"the export exited with 0 but wrote no {outfile}"
    assert not stale.exists() and export.calls == [(Path(outfile), True)]
    assert bakeoff.smoke_adapter(config, config.candidate("c2"), Path("runs/bakeoff/test/c2")) is None


def thought(answer: str) -> Scripted:
    """A thinking completion that closes its block itself and ends at the terminator."""
    return Scripted(("Need a search.", "\n</think>\n\n", answer, "<|im_end|>"),
                    ids=(500, 501, 502, TURN_END_ID))


OVERLONG = Scripted(("thinking on and on",), finish_reason="length")  # no close: the budget is spent
CLOSED = Scripted(("See you!", "<|im_end|>"), ids=(503, TURN_END_ID))  # the answer after a forced close


def s5_script() -> list[Scripted]:
    """The completions S5 asks of a thinking family over 2 items: the primary run twice (one request per
    item), then the run under the forced budget twice (two requests per item)."""
    primary = [thought(CALL), thought("See you!")]
    forced = [OVERLONG, CLOSED, OVERLONG, CLOSED]
    return [*primary, *primary, *forced, *forced]


def test_s5_checks_prompt_tokens_the_terminator_and_byte_identical_repeats(config, repo):
    gguf = repo / "runs" / "bakeoff" / "test" / "c2" / "smoke" / "train" / "gguf" / "adapter-f16.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"GGUF adapter")
    adapter = {"id": 0, "path": "runs/bakeoff/test/c2/smoke/train/gguf/adapter-f16.gguf", "scale": 1.0}
    fake = FakeLlama(s5_script(), adapters=[adapter])
    served = ServedLlama(fake)
    s5 = run_smoke(config, config.candidate("c2"), tools_of(served=served), steps=["S5"])["steps"]["S5"]
    assert s5["passed"] is True, s5
    assert s5["items"] == 2 and s5["adapter"] == adapter["path"] and s5["lora_scale"] == 1.0
    assert s5["prompt_tokens"] == {"requests": 12, "mismatches": 0}
    assert s5["terminator"] == {"stopped": 8, "at_terminator": 8, "cut": 0}
    assert s5["runs"]["primary"] == {"reasoning_budget": 16, "identical": True, "forced": 0, "phases": 4}
    assert s5["runs"]["forced"] == {"reasoning_budget": 32, "identical": True, "forced": 4, "phases": 8}
    assert s5["mode"] == "think" and s5["server"]["slots"] == 1
    (cmd,) = served.started
    assert cmd[cmd.index("--lora") + 1] == adapter["path"] and cmd[-2:] == ["-np", "1"]
    assert cmd[cmd.index("-m") + 1] == "models/only-model-Q8_0.gguf"
    assert served.processes[0].terminated
    # the prompts went as ids of the family's THINK prompt, and every request's scale was 1
    first = fake.requests[0]
    assert isinstance(first["prompt"], list) and first["max_tokens"] == 16 and first["temperature"] == 0.6
    assert all(r["lora"] == [{"id": 0, "scale": 1.0}] for r in fake.requests)
    assert fake.requests[4]["max_tokens"] == 32 and fake.requests[5]["prompt"][-4:] != first["prompt"][-4:]


def test_s5_fails_on_a_repeat_that_differs_a_run_past_the_terminator_or_a_wrong_token_count(config, repo):
    fake = FakeLlama([thought(CALL), thought("See you!"), thought(CALL), thought("Bye!")]
                     + [OVERLONG, CLOSED] * 4)
    s5 = run_smoke(config, config.candidate("c2"), tools_of(served=ServedLlama(fake)),
                   steps=["S5"])["steps"]["S5"]
    assert s5["passed"] is False and s5["runs"]["primary"]["identical"] is False
    assert "primary" in s5["failures"][0] and s5["adapter"] is None and s5["lora_scale"] == 0.0
    not_ended = Scripted(("Need a search.", "\n</think>\n\n", "See you!", " more"), ids=(500, 501, 502, 9))
    fake = FakeLlama([not_ended, not_ended] * 2 + [OVERLONG, CLOSED] * 4)
    s5 = run_smoke(config, config.candidate("c2"), tools_of(served=ServedLlama(fake)),
                   steps=["S5"])["steps"]["S5"]
    assert s5["passed"] is False and s5["terminator"]["at_terminator"] == 4
    assert "terminator" in str(s5["failures"])

    class OffByOne(FakeLlama):
        def handle(self, request: httpx.Request) -> httpx.Response:
            response = super().handle(request)
            if request.url.path != "/v1/completions" or response.status_code != 200:
                return response
            data = response.json()
            data["usage"]["prompt_tokens"] += 1
            return httpx.Response(200, json=data)

    s5 = run_smoke(config, config.candidate("c2"), tools_of(served=ServedLlama(OffByOne(s5_script()))),
                   steps=["S5"])["steps"]["S5"]
    assert s5["passed"] is False and s5["prompt_tokens"]["mismatches"] == 12
    # a family that never thinks runs the primary pair alone, greedily
    fake = FakeLlama([Scripted((CALL, "<|im_end|>"), ids=(500, TURN_END_ID)),
                      Scripted(("See you!", "<|im_end|>"), ids=(501, TURN_END_ID))] * 2)
    s5 = run_smoke(config, config.candidate("c1"), tools_of(served=ServedLlama(fake)),
                   steps=["S5"])["steps"]["S5"]
    assert s5["passed"] is True and list(s5["runs"]) == ["primary"] and s5["mode"] == "no_think"
    assert fake.requests[0]["temperature"] == 0.0 and "seed" not in fake.requests[0]
    # a server that does not come up
    fake = FakeLlama([])
    s5 = run_smoke(config, config.candidate("c1"), tools_of(served=ServedLlama(fake, exit_code=1)),
                   steps=["S5"])["steps"]["S5"]
    assert s5["passed"] is False and "health" in s5["error"]


def test_s5_needs_the_small_budget_to_force_a_close(config, repo):
    # S5's repeat under R=32 must include a forced close (docs/m7-底座对比预登记.md S5): a model that closes
    # its block within the budget every time leaves the forcing path untried, so the step fails
    closing = [thought(CALL), thought("See you!")]
    fake = FakeLlama(closing * 4)
    s5 = run_smoke(config, config.candidate("c2"), tools_of(served=ServedLlama(fake)),
                   steps=["S5"])["steps"]["S5"]
    assert s5["runs"]["forced"] == {"reasoning_budget": 32, "identical": True, "forced": 0, "phases": 4}
    assert s5["passed"] is False and s5["failures"] == ["no close was forced under the budget 32"]
    assert fake.requests[4]["max_tokens"] == 32
    assert s5["server"]["binary_sha256"] == sha256_of(repo / "llama-server.exe")


def fake_backends(folder: Path, *, unloaded: tuple[str, ...] = (), still: tuple[str, ...] = ()) -> Backends:
    """S6's model parts, scripted: an adapter moves every top-1 token at the first strength, except that a
    group in unloaded is not loaded (AdapterNotLoaded) and one in still moves none at any strength."""
    def predict(adapter, sequences):
        group = None if adapter is None else Path(adapter).parent.name
        if group in unloaded:
            raise AdapterNotLoaded(f"0 of the 2 LoRA tensors of {Path(adapter).as_posix()} are in the model")
        shift = 0 if group is None or group in still else 1
        return [[p + shift for p in range(s.positions)] for s in sequences]

    def export(adapter: Path, outfile: Path) -> Path:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        outfile.write_bytes(b"GGUF")
        return outfile

    def modules(target):
        assert isinstance(target, tuple)  # the fake family's targets are module names, a PEFT list
        return [Module(f"model.layers.0.self_attn.{name}", 4, 4) for name in target]

    return Backends(modules=modules, weight_norm=lambda module: 1.0, export=export, hf_top1=predict,
                    llama_top1=predict)


def test_s6_runs_the_fidelity_check_over_the_familys_targets(config, repo):
    seen = []

    def backends(cfg, candidate, tools, folder):
        seen.append((candidate.key, folder))
        return fake_backends(folder)

    s6 = run_smoke(config, config.candidate("c3"), tools_of(fidelity_backends=backends),
                   steps=["S6"])["steps"]["S6"]
    assert s6["passed"] is True and list(s6["groups"]) == ["q_proj", "v_proj"]  # the fake family's targets
    assert s6["settings"]["items"] == 2 and s6["settings"]["positions"] == 4 and s6["settings"]["r"] == 16
    assert s6["groups"]["q_proj"]["adapter"] == "runs/bakeoff/test/c3/smoke/fidelity/q_proj/adapter"
    assert s6["groups"]["q_proj"]["target"] == ["q_proj"] and s6["groups"]["q_proj"]["fraction"] == 0.07
    assert s6["groups"]["q_proj"]["calibration"]["trials"] == [
        {"fraction": 0.07, "changed_positions": 8, "changed": 1.0}]
    assert seen == [("c3", Path("runs/bakeoff/test/c3"))]
    assert (repo / "runs/bakeoff/test/c3/smoke/fidelity/q_proj/adapter/adapter_config.json").is_file()
    assert smoke_of(Path("runs/bakeoff/test/c3"))["steps"]["S6"] == s6  # the record is JSON as it is kept


def test_s6_fails_a_group_its_adapter_did_not_load_into_or_did_not_move_with_the_reason(config, repo):
    # docs/m7-进度.md: C3's first S6 had none of its synthetic adapters loaded, and every group changed
    # 0.000. Now a group whose adapter the training stack did not load fails with the load check's reason,
    # and one no strength moves fails as having no power, neither compared with llama.cpp
    def backends(cfg, candidate, tools, folder):
        return fake_backends(folder, unloaded=("q_proj",), still=("v_proj",))

    smoke = run_smoke(config, config.candidate("c3"), tools_of(fidelity_backends=backends), steps=["S6"])
    s6 = smoke["steps"]["S6"]
    assert s6["passed"] is False and s6["failed"] == ["q_proj", "v_proj"] and smoke["failed"] == ["S6"]
    assert s6["failures"] == [
        ("group q_proj: the training stack did not load the adapter at fraction 0.07: 0 of the 2 LoRA "
         "tensors of runs/bakeoff/test/c3/smoke/fidelity/q_proj/adapter are in the model"),
        ("group v_proj: changed 0.000 at the cap fraction 2 (least 0.300): no strength up to the cap gives "
         "the comparison its power")]
    assert [t["fraction"] for t in s6["groups"]["v_proj"]["calibration"]["trials"]] == [
        0.07, 0.14, 0.28, 0.56, 1.12, 2.0]
    assert s6["groups"]["q_proj"]["gguf"] is None and s6["groups"]["v_proj"]["gguf"] is None
    log = (repo / "runs/bakeoff/test/c3/bakeoff.log").read_text(encoding="utf-8")
    assert "S6: FAILED: group q_proj: the training stack did not load the adapter" in log


def test_s6_exports_through_the_runner_into_a_folder_made_first(config, repo):
    # the default backends' export runs scripts/export_adapter.py as S4 does (run_export): the GGUF's folder
    # is made before the converter runs, a GGUF left there earlier is removed, the log is S6's per group
    export = FakeExport(b"GGUF synthetic")
    runner = FakeRunner({"export_adapter.py": export})
    candidate = config.candidate("c3")
    backends = bakeoff.default_fidelity_backends(config, candidate, tools_of(runner), config.folder("c3"))
    adapter = Path("runs/bakeoff/test/c3/smoke/fidelity/q_proj/adapter")
    outfile = Path("runs/bakeoff/test/c3/smoke/fidelity/elsewhere/q_proj-f16.gguf")
    assert backends.export(adapter, outfile) == outfile
    assert outfile.read_bytes() == b"GGUF synthetic" and export.calls == [(outfile, True)]
    (cmd,) = runner.commands
    args = load_script("export_adapter").build_parser().parse_args(script_args(cmd))
    assert args.adapter == adapter.as_posix() and args.base == "models/hybrid-model"
    assert runner.logs == [Path("runs/bakeoff/test/c3/smoke/S6.export.q_proj")]
    # an export that exits 0 without writing does not pass for the GGUF of an earlier run
    runner = FakeRunner({"export_adapter.py": FakeExport(written=False)})
    backends = bakeoff.default_fidelity_backends(config, candidate, tools_of(runner), config.folder("c3"))
    with pytest.raises(BakeoffError, match="exited with 0 and wrote no"):
        backends.export(adapter, outfile)
    assert not outfile.exists()


def test_run_smoke_runs_every_step_marks_the_exempt_ones_and_keeps_going_after_a_failure(config, repo):
    runner = FakeRunner({"pytest": lambda argv: (5, "no tests ran\n")})
    smoke = run_smoke(config, config.candidate("c1"), tools_of(runner, served=ServedLlama(FakeLlama([]),
                                                                                           exit_code=1)))
    assert smoke["order"] == list(STEPS) and set(smoke["steps"]) == set(STEPS)
    for step in ("S0", "S1", "S2", "S3", "S4"):
        assert smoke["steps"][step] == {"exempt": True, "passed": None,
                                        "reason": CONFIG["candidates"]["c1"]["exempt_reason"]}
    assert smoke["steps"]["S5"]["passed"] is False and smoke["steps"]["S6"]["passed"] is False
    assert smoke["passed"] is False and smoke["failed"] == ["S5", "S6"]
    assert runner.commands == []  # nothing of the exempt steps ran
    assert smoke_of(Path("runs/bakeoff/test/c1")) == smoke
    with pytest.raises(BakeoffError, match="S9"):
        run_smoke(config, config.candidate("c1"), tools_of(), steps=["S9"])


def test_a_blocked_step_keeps_the_smoke_undecided_and_is_no_reason_for_ineligibility(config, repo):
    folder = repo / "runs" / "bakeoff" / "test" / "c2"
    folder.mkdir(parents=True)
    (folder / "smoke.json").write_text(json.dumps({"candidate": "c2", "steps": {
        s: {"passed": True} for s in STEPS if s != "S2"}}), encoding="utf-8")
    summary = {**SUMMARY, "samples": 995, "refused": {"template": 5, "boundary": 0, "bos": 0}}
    runner = FakeRunner({"train.py": lambda argv: (0, dry_run_text(summary))})
    smoke = run_smoke(config, config.candidate("c2"), tools_of(runner), steps=["S2"])
    assert smoke["blocked"] == ["S2"] and smoke["failed"] == [] and smoke["passed"] is None
    log = (folder / "bakeoff.log").read_text(encoding="utf-8")
    assert "S2: BLOCKED: template refusals: 5 of 1000 steps" in log
    status, reasons, missing = smoke_status(smoke, config.candidate("c2"))
    assert status["S2"] == "blocked" and reasons == [] and missing == ["S2 is blocked (smoke.json says why)"]
    gate = candidate_gate(config, config.candidate("c2"))
    assert gate["status"] == "incomplete" and gate["eligible"] is None and gate["reasons"] == []
    assert "S2 is blocked (smoke.json says why)" in gate["missing"]


def test_exemptions_are_the_configurations_not_the_records(config):
    # a record that says exempt for a step the candidate is not exempt from (a copied smoke.json, an edited
    # configuration) counts as not run, and the gate waits for the step
    steps = {**{s: {"passed": True} for s in STEPS},
             "S0": {"exempt": True, "passed": None, "reason": "copied"}}
    status, reasons, missing = smoke_status({"steps": steps}, config.candidate("c2"))
    assert status["S0"] == "missing" and reasons == [] and missing == ["S0 was not run"]
    # C1 is exempt from S0-S4 by the configuration, whatever a record of them says
    status, reasons, missing = smoke_status({"steps": {"S0": {"passed": False}}}, config.candidate("c1"))
    assert status["S0"] == "exempt" and reasons == [] and missing == []


def test_a_step_that_failed_while_its_input_step_had_not_passed_is_blocked_and_waits_for_it(config):
    # docs/m7-底座对比预登记.md 修订 3 (二.2): S4 exports the adapter S3 trains and S5 serves the GGUF S4
    # writes (STEP_INPUTS); a step that failed while its input step had not passed is a scheduling gap (C2's
    # S4, run before S3 ever ran under this tool), which waits for that step and excludes nothing
    c2 = config.candidate("c2")
    passed = {s: {"passed": True} for s in STEPS}
    no_adapter = {"passed": False, "error": "BakeoffError: no smoke adapter at runs/bakeoff/test/c2/smoke/"
                                            "train/adapter: S3 writes it"}
    waits = {"S4": "S4 is blocked: it uses the output of S3, which has not passed",
             "S5": "S5 is blocked: it uses the output of S4, which has not passed"}
    # S3 never run and S4 failed: S4 waits for S3
    steps = {s: record for s, record in {**passed, "S4": no_adapter}.items() if s != "S3"}
    status, reasons, missing = smoke_status({"steps": steps}, c2)
    assert status["S3"] == "missing" and status["S4"] == "blocked" and status["S5"] == "passed"
    assert reasons == [] and missing == ["S3 was not run", waits["S4"]]
    # S3 passed: S4's failure is its own, a reason; S4 passed: S5's failure is a reason
    status, reasons, missing = smoke_status({"steps": {**passed, "S4": no_adapter}}, c2)
    assert status["S4"] == "failed" and reasons == ["S4 failed"] and missing == []
    status, reasons, missing = smoke_status({"steps": {**passed, "S5": {"passed": False}}}, c2)
    assert status["S5"] == "failed" and reasons == ["S5 failed"] and missing == []
    # S3 failed and S4 failed: S3 is the reason and S4 waits for it; S5, failed after S4, waits for S4 in turn
    steps = {**passed, "S3": {"passed": False}, "S4": no_adapter, "S5": {"passed": False}}
    status, reasons, missing = smoke_status({"steps": steps}, c2)
    assert (status["S3"], status["S4"], status["S5"]) == ("failed", "blocked", "blocked")
    assert reasons == ["S3 failed"] and missing == [waits["S4"], waits["S5"]]
    # an input step blocked by its own record has not passed either
    steps = {**passed, "S3": {"passed": None, "blocked": "x"}, "S4": no_adapter}
    status, reasons, missing = smoke_status({"steps": steps}, c2)
    assert status["S4"] == "blocked" and reasons == []
    assert missing == ["S3 is blocked (smoke.json says why)", waits["S4"]]
    # an input step the candidate is exempt from holds: the failure after it is a reason
    exempt = dataclasses.replace(c2, exempt=("S4",), exempt_reason="x")
    status, reasons, missing = smoke_status({"steps": {**passed, "S5": {"passed": False}}}, exempt)
    assert status["S4"] == "exempt" and status["S5"] == "failed" and reasons == ["S5 failed"]
    # a step run for the record only keeps its status, neither a reason nor awaited
    for_record = dataclasses.replace(c2, record_only=("S5",))
    steps = {s: record for s, record in {**passed, "S5": {"passed": False}}.items() if s != "S4"}
    status, reasons, missing = smoke_status({"steps": steps}, for_record)
    assert status["S5"] == "failed (recorded only)" and reasons == [] and missing == ["S4 was not run"]


def test_smoke_plan_reruns_what_has_not_passed_and_the_steps_that_use_its_output(config):
    c2, c1 = config.candidate("c2"), config.candidate("c1")
    passed = {s: {"passed": True} for s in STEPS}
    assert smoke_plan({"steps": passed}, c2) == [] and smoke_plan(None, c2) == list(STEPS)
    # S4 exports the adapter S3 trains and S5 serves the GGUF S4 writes: a new S3 makes both run again
    assert smoke_plan({"steps": {**passed, "S3": {"passed": False}}}, c2) == ["S3", "S4", "S5"]
    assert smoke_plan({"steps": {**passed, "S4": {"passed": False, "error": "x"}}}, c2) == ["S4", "S5"]
    assert smoke_plan({"steps": {**passed, "S2": {"passed": None, "blocked": "x"}}}, c2) == ["S2"]
    del passed["S6"]
    assert smoke_plan({"steps": passed}, c2) == ["S6"]
    # an exempt step recorded as exempt is done; C1's steps run for the record are run while they fail
    exempt = {s: {"exempt": True, "passed": None} for s in ("S0", "S1", "S2", "S3", "S4")}
    assert smoke_plan({"steps": exempt}, c1) == ["S5", "S6"]
    assert smoke_plan({"steps": {**exempt, "S5": {"passed": True}, "S6": {"passed": False}}}, c1) == ["S6"]
    assert smoke_plan({"steps": {s: {"exempt": True, "passed": None} for s in STEPS}}, c2) == list(STEPS)


# ---- the pilot and the evaluation -------------------------------------------------------------------------


def pilot_outputs(out: Path, *, stop: int = 274, samples: int = 100, tokens: int = 5000) -> None:
    """What scripts/train.py --stop-at-step leaves behind."""
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"config": {"model": "models/only-model", "family": {"name": "fake-only"},
                           "data": {"samples": samples, "tokens": {"total": tokens},
                                    "modes": {"think": 30, "no_think": 70, "think_fallback": 0}}}}
    (out / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [{"event": "start"}] + [{"step": s, "tokens_per_s": 300 + s, "peak_reserved_gib": 4.4}
                                   for s in range(10, stop + 1, 10)]
    write_jsonl(out / "log.jsonl", rows)
    (out / "trainer" / f"checkpoint-{stop}").mkdir(parents=True, exist_ok=True)
    (out / "trainer" / f"checkpoint-{stop}" / "trainer_state.json").write_text("{}", encoding="utf-8")
    (out / "result.json").write_text(json.dumps({"steps": stop, "stopped_at_step": stop}), encoding="utf-8")


def test_pilot_waits_trains_to_the_stop_step_and_exports_the_checkpoint(config, repo):
    def train(argv: list[str]) -> tuple[int, str]:
        pilot_outputs(Path(argv[argv.index("--out") + 1]))  # the run folder, without a gguf/ folder
        return 0, "trained\n"

    export = FakeExport(b"GGUF checkpoint")
    runner = FakeRunner({"train.py": train, "export_adapter.py": export})
    probes = FakeProbes(gpu=[6600, 1000, 6016, 1000], ram=[8.0], ac=[True])
    tools = tools_of(runner, probes=probes)
    pilot = run_pilot(config, config.candidate("c2"), tools)
    assert pilot["passed"] is True and pilot["stop_step"] == 274
    assert pilot["train"] == "runs/bakeoff/test/c2/train"
    assert pilot["checkpoint"] == "runs/bakeoff/test/c2/train/trainer/checkpoint-274"
    assert pilot["gguf"] == "runs/bakeoff/test/c2/train/gguf/checkpoint-274-f16.gguf"
    assert pilot["gguf_sha256"] == sha256_of(repo / pilot["gguf"]) and pilot["nvidia_smi_peak_mib"] == 6016
    assert pilot["preflight"]["gpu_used_mib"] == 1000 and tools.clock.slept[0] == 60.0
    assert pilot["manifest"] == {"samples": 100, "tokens_per_epoch": 5000, "family": "fake-only"}
    assert json.loads((repo / "runs/bakeoff/test/c2/pilot.json").read_text(encoding="utf-8")) == pilot
    train_cmd, export_cmd = runner.commands
    args = load_script("train").build_parser().parse_args(script_args(train_cmd))
    assert args.stop_at_step == 274 and args.save_steps == 274 and args.out == "runs/bakeoff/test/c2/train"
    assert runner.envs[0]["HF_HUB_OFFLINE"] == "1" and runner.envs[0]["PYTHONUTF8"] == "1"
    args = load_script("export_adapter").build_parser().parse_args(script_args(export_cmd))
    assert args.adapter == pilot["checkpoint"] and args.outfile == pilot["gguf"]
    assert export.calls == [(Path(pilot["gguf"]), True)]  # the gguf/ folder was made before the converter ran
    # again: the checkpoint and its GGUF are there, nothing runs
    again = run_pilot(config, config.candidate("c2"), tools_of(FakeRunner(), probes=probes))
    assert again["passed"] is True and again["trained"] is False and again["exported"] is False
    # the extension resumes the run to the extension step

    def train_to(argv: list[str], stop: int) -> tuple[int, str]:
        pilot_outputs(Path(argv[argv.index("--out") + 1]), stop=stop)
        return 0, "resumed\n"

    runner = FakeRunner({"train.py": lambda argv: train_to(argv, 548), "export_adapter.py": export})
    extended = run_pilot(config, config.candidate("c2"), tools_of(runner, probes=probes), stop_step=548)
    assert extended["passed"] is True and extended["stop_step"] == 548
    args = load_script("train").build_parser().parse_args(script_args(runner.commands[0]))
    assert args.resume and args.stop_at_step == 548
    assert extended["gguf"] == "runs/bakeoff/test/c2/train/gguf/checkpoint-548-f16.gguf"


def test_pilot_resumes_from_a_saved_checkpoint_and_refuses_a_run_that_saved_none(config, repo):
    train = repo / "runs" / "bakeoff" / "test" / "c2" / "train"
    train.mkdir(parents=True)  # a pilot stopped before its first save: a manifest and a log, no checkpoint
    (train / "manifest.json").write_text(json.dumps({"config": {"model": "models/only-model"}}),
                                         encoding="utf-8")
    write_jsonl(train / "log.jsonl", [{"event": "start"}, {"step": 10, "tokens_per_s": 400}])
    (train / "trainer" / "checkpoint-137").mkdir(parents=True)  # a save cut short: no trainer_state.json
    before = sorted(p.relative_to(train).as_posix() for p in train.rglob("*"))

    def train_to_stop(argv: list[str]) -> tuple[int, str]:
        pilot_outputs(Path(argv[argv.index("--out") + 1]))
        return 0, "trained\n"

    runner = FakeRunner({"train.py": train_to_stop})
    # the GPU is busy and the wait limit 0: had the preflight run, it would have raised PreflightTimeout
    tools = tools_of(runner, probes=FakeProbes(gpu=[6600]), wait_limit=0.0)
    pilot = run_pilot(config, config.candidate("c2"), tools)
    assert pilot["passed"] is False and pilot["preflight"] is None and runner.commands == []
    assert pilot["error"].startswith("runs/bakeoff/test/c2/train holds a run that saved no checkpoint")
    assert "move the folder aside by hand" in pilot["error"]
    assert sorted(p.relative_to(train).as_posix() for p in train.rglob("*")) == before  # nothing touched
    # once a checkpoint is saved, the pilot resumes from it (scripts/train.py --resume takes the last one)
    (train / "trainer" / "checkpoint-137" / "trainer_state.json").write_text('{"global_step": 137}',
                                                                             encoding="utf-8")
    pilot = run_pilot(config, config.candidate("c2"), tools_of(runner))
    (cmd,) = runner.of("train.py")
    args = load_script("train").build_parser().parse_args(script_args(cmd))
    assert args.resume and args.stop_at_step == 274 and pilot["trained"] is True
    # a fresh folder starts afresh
    runner = FakeRunner({"train.py": train_to_stop})
    run_pilot(config, config.candidate("c3"), tools_of(runner))
    args = load_script("train").build_parser().parse_args(script_args(runner.commands[0]))
    assert not args.resume
    # the saved checkpoints decide the resume, not the manifest: a folder that lost its manifest is resumed,
    # and scripts/train.py --resume refuses it itself (exit 5) rather than a fresh run writing over it
    orphan = repo / "runs" / "bakeoff" / "test" / "c1" / "train" / "trainer" / "checkpoint-100"
    orphan.mkdir(parents=True)
    (orphan / "trainer_state.json").write_text('{"global_step": 100}', encoding="utf-8")
    runner = FakeRunner({"train.py": lambda argv: (5, "RESUME REFUSED: no manifest\n")})
    pilot = run_pilot(config, config.candidate("c1"), tools_of(runner))
    args = load_script("train").build_parser().parse_args(script_args(runner.commands[0]))
    assert args.resume and pilot["passed"] is False and pilot["error"] == "the training exited with 5"


def test_pilot_reports_a_training_failure_and_a_missing_checkpoint(config, repo):
    runner = FakeRunner({"train.py": lambda argv: (1, "CUDA out of memory\n")})
    pilot = run_pilot(config, config.candidate("c2"), tools_of(runner))
    assert pilot["passed"] is False and "exited with 1" in pilot["error"] and pilot["gguf"] is None
    runner = FakeRunner({"train.py": lambda argv: (0, "stopped early\n")})
    pilot = run_pilot(config, config.candidate("c2"), tools_of(runner))
    assert pilot["passed"] is False and "checkpoint-274" in pilot["error"]
    with pytest.raises(PreflightTimeout):
        run_pilot(config, config.candidate("c2"), tools_of(FakeRunner(), probes=FakeProbes(gpu=[6600]),
                                                           wait_limit=100.0))


def eval_handlers(successes: dict[str, dict[str, bool]]):
    """run_eval.py and score.py stand-ins: the records of a variant (by its --out stem) and the scores from
    the successes table, variant -> item id -> success."""
    def run_eval(argv: list[str]) -> tuple[int, str]:
        out = Path(argv[argv.index("--out") + 1])
        variant = out.stem.split("-", 1)[1]
        rows = [{"id": i["id"], "status": "ok", "output": {"role": "assistant", "content": "x"},
                 "variant": variant} for i in ITEMS]
        write_jsonl(out, rows)
        out.with_name(out.stem + ".manifest.json").write_text(json.dumps({"system": argv[argv.index(
            "--system") + 1]}), encoding="utf-8")
        return 0, f"{out}: 3 records\n"

    def score(argv: list[str]) -> tuple[int, str]:
        records = Path(argv[0])
        variant = records.stem.split("-", 1)[1]
        score_file(Path(argv[argv.index("--out") + 1]), successes[variant])
        return 0, "scored\n"

    return {"run_eval.py": run_eval, "score.py": score}


def test_evaluate_runs_every_variant_and_b0_on_one_server_and_scores_them(config, repo):
    pilot_outputs(repo / "runs" / "bakeoff" / "test" / "c3" / "train")
    gguf = repo / "runs" / "bakeoff" / "test" / "c3" / "train" / "gguf" / "checkpoint-274-f16.gguf"
    gguf.parent.mkdir()
    gguf.write_bytes(b"GGUF checkpoint")
    smoke = {"candidate": "c3", "steps": {"S2": {"passed": True, "reasoning_budget": 1280}}}
    (repo / "runs/bakeoff/test/c3/smoke.json").write_text(json.dumps(smoke), encoding="utf-8")
    successes = {v: {"q-1": True, "q-2": v != "nothink"} for v in VARIANTS["hybrid"]}
    successes["b0"] = {"q-3": True}
    runner = FakeRunner(eval_handlers(successes))
    adapter = {"id": 0, "path": "runs/bakeoff/test/c3/train/gguf/checkpoint-274-f16.gguf", "scale": 1.0}
    served = ServedLlama(FakeLlama([], adapters=[adapter]))
    result = run_evaluate(config, config.candidate("c3"), tools_of(runner, served=served))
    assert result["passed"] is True and result["reasoning_budget"] == 1280 and result["stop_step"] == 274
    assert list(result["runs"]) == ["think-s0", "think-s1", "think-s2", "think-greedy", "nothink", "b0"]
    assert result["runs"]["think-s1"] == {
        "system": "ft", "args": ["--mode", "think", "--seed", "1", "--reasoning-budget", "1280"],
        "records": "runs/bakeoff/test/c3/dev-think-s1.jsonl",
        "score": "runs/bakeoff/test/c3/dev-think-s1.score.json", "successes": 2, "n": 3, "reused": False}
    assert result["runs"]["b0"]["system"] == "b0" and result["runs"]["b0"]["successes"] == 1
    assert result["runs"]["b0"]["args"] == ["--mode", "think", "--seed", "0", "--reasoning-budget", "1280"]
    assert result["server"]["adapter"] == adapter["path"] and result["server"]["slots"] == 1
    (cmd,) = served.started
    assert cmd[cmd.index("--lora") + 1] == adapter["path"] and cmd[-2:] == ["-np", "1"]
    evals = runner.of("run_eval.py")
    assert len(evals) == 6 and len(runner.of("score.py")) == 6 and served.processes[0].terminated
    args = load_script("run_eval").build_parser().parse_args(script_args(evals[3]))
    assert args.greedy and args.reasoning_budget == 1280 and args.system == "ft" and args.signals
    args = load_script("run_eval").build_parser().parse_args(script_args(evals[5]))
    assert args.system == "b0" and args.mode == "think" and args.seed == 0
    assert json.loads((repo / "runs/bakeoff/test/c3/evaluate.json").read_text(encoding="utf-8")) == result


def test_evaluate_needs_the_pilots_gguf_a_budget_for_a_thinking_family_and_the_right_adapter(config, repo):
    c2 = config.candidate("c2")
    result = run_evaluate(config, c2, tools_of(FakeRunner()))
    assert result["passed"] is False and "checkpoint-274" in result["error"]
    pilot_outputs(repo / "runs" / "bakeoff" / "test" / "c3" / "train")
    gguf = repo / "runs" / "bakeoff" / "test" / "c3" / "train" / "gguf" / "checkpoint-274-f16.gguf"
    gguf.parent.mkdir()
    gguf.write_bytes(b"GGUF")
    result = run_evaluate(config, config.candidate("c3"), tools_of(FakeRunner()))
    assert result["passed"] is False and "reasoning budget" in result["error"]
    served = ServedLlama(FakeLlama([], adapters=[{"id": 0, "path": "other.gguf", "scale": 1.0}]))
    (repo / "runs/bakeoff/test/c3/smoke.json").write_text(json.dumps(
        {"steps": {"S2": {"reasoning_budget": 1280}}}), encoding="utf-8")
    result = run_evaluate(config, config.candidate("c3"), tools_of(FakeRunner(), served=served))
    assert result["passed"] is False and "other.gguf" in result["error"]
    runner = FakeRunner({"run_eval.py": lambda argv: (2, "FAIL: no server\n")})
    served = ServedLlama(FakeLlama([], adapters=[{"id": 0, "path": str(Path(
        "runs/bakeoff/test/c3/train/gguf/checkpoint-274-f16.gguf")), "scale": 1.0}]))
    result = run_evaluate(config, config.candidate("c3"), tools_of(runner, served=served))
    assert result["passed"] is False and "think-s0" in result["error"] and "exited with 2" in result["error"]


def manifested_eval_handlers(successes: dict[str, dict[str, bool]], *, adapter_sha256: str,
                             fail: str | None = None):
    """eval_handlers whose run_eval also writes the manifest fields scripts/run_eval.py writes and
    stale_records reads: the system, the items with their SHA-256 and counts, the tokenizer, the signals, the
    adapter the server loaded with its SHA-256 and the generation; the run of the variant named fail exits
    with 1 and writes nothing."""
    handlers = eval_handlers(successes)
    plain = handlers["run_eval.py"]

    def run_eval(argv: list[str]) -> tuple[int, str]:
        out = Path(argv[argv.index("--out") + 1])
        if out.stem.split("-", 1)[1] == fail:
            return 1, "crashed\n"
        code, text = plain(argv)

        def value(flag: str, default: str | None = None) -> str | None:
            return argv[argv.index(flag) + 1] if flag in argv else default

        greedy = "--greedy" in argv
        mode = "think" if greedy else value("--mode")
        manifest = {"system": value("--system"), "items": argv[0], "items_sha256": sha256_of(Path(argv[0])),
                    "items_run": len(ITEMS), "items_total": len(ITEMS), "tokenizer": value("--tokenizer"),
                    "signals": "--signals" in argv,
                    "adapters": [{"id": 0, "path": "the GGUF", "sha256": adapter_sha256}],
                    "generation": {"mode": mode, "base_seed": int(value("--seed", "0")),
                                   "reasoning_budget": int(value("--reasoning-budget", "1024")),
                                   "decoding": {"temperature": 0.0 if greedy or mode == "no_think" else 0.6}}}
        out.with_name(out.stem + ".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return code, text

    return {**handlers, "run_eval.py": run_eval}


C3_ADAPTER = {"id": 0, "path": "runs/bakeoff/test/c3/train/gguf/checkpoint-274-f16.gguf", "scale": 1.0}
C3_RUNS = ["think-s0", "think-s1", "think-s2", "think-greedy", "nothink", "b0"]


def c3_checkpoint(repo: Path) -> Path:
    """What c3's evaluation needs: its pilot outputs, the checkpoint's GGUF and S2's reasoning budget."""
    folder = repo / "runs" / "bakeoff" / "test" / "c3"
    pilot_outputs(folder / "train")
    gguf = folder / "train" / "gguf" / "checkpoint-274-f16.gguf"
    gguf.parent.mkdir()
    gguf.write_bytes(b"GGUF checkpoint")
    (folder / "smoke.json").write_text(json.dumps({"candidate": "c3", "steps": {
        "S2": {"passed": True, "reasoning_budget": 1280}}}), encoding="utf-8")
    return gguf


def evaluated_stems(runner: FakeRunner) -> list[str]:
    return [Path(cmd[cmd.index("--out") + 1]).stem for cmd in runner.of("run_eval.py")]


def test_evaluate_keeps_the_runs_it_has_and_runs_the_rest(config, repo):
    gguf = c3_checkpoint(repo)
    successes = {name: {"q-1": True} for name in C3_RUNS}

    def evaluate(**kwargs) -> tuple[dict, FakeRunner, ServedLlama]:
        runner = FakeRunner(manifested_eval_handlers(successes, adapter_sha256=sha256_of(gguf), **{
            k: v for k, v in kwargs.items() if k == "fail"}))
        served = ServedLlama(FakeLlama([], adapters=[C3_ADAPTER]))
        result = run_evaluate(config, config.candidate("c3"), tools_of(runner, served=served),
                              **{k: v for k, v in kwargs.items() if k == "rerun"})
        return result, runner, served

    # the evaluation dies at its fourth run: the first three are recorded and scored
    first, runner, _ = evaluate(fail="think-greedy")
    assert first["passed"] is False and "think-greedy" in first["error"]
    assert list(first["runs"]) == ["think-s0", "think-s1", "think-s2"]
    # again: those three stand (same GGUF, items and settings) and are scored again, the other three run
    second, runner, served = evaluate()
    assert second["passed"] is True and list(second["runs"]) == C3_RUNS
    assert evaluated_stems(runner) == ["dev-think-greedy", "dev-nothink", "dev-b0"]
    assert len(runner.of("score.py")) == 6 and len(served.started) == 1
    assert [second["runs"][name]["reused"] for name in C3_RUNS] == [True, True, True, False, False, False]
    # all there: nothing runs, no server starts, every run is scored from its records
    third, runner, served = evaluate()
    assert third["passed"] is True and runner.of("run_eval.py") == [] and served.started == []
    assert third["server"] is None and len(runner.of("score.py")) == 6
    assert third["runs"]["b0"]["successes"] == 1
    # rerun runs everything again
    fourth, runner, _ = evaluate(rerun=True)
    assert fourth["passed"] is True and evaluated_stems(runner) == [f"dev-{name}" for name in C3_RUNS]
    # a new checkpoint GGUF: no record stands for it
    gguf.write_bytes(b"GGUF retrained")
    fifth, runner, _ = evaluate()
    assert fifth["passed"] is True and len(runner.of("run_eval.py")) == 6


def records_for(path: Path, manifest: dict) -> Path:
    write_jsonl(path, [{"id": i["id"]} for i in ITEMS])
    path.with_name(path.stem + ".manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return path


THINK_S1 = ["--mode", "think", "--seed", "1", "--reasoning-budget", "1280"]


def think_s1_manifest() -> dict:
    """run_eval's manifest of c3's think-s1 run on a checkpoint GGUF whose SHA-256 is f...f."""
    return {"system": "ft", "items": "data/eval/dev.jsonl",
            "items_sha256": sha256_of(Path("data/eval/dev.jsonl")),
            "items_run": 3, "items_total": 3, "tokenizer": "models/hybrid-model", "signals": True,
            "adapters": [{"id": 0, "path": "x.gguf", "sha256": "f" * 64}],
            "generation": {"mode": "think", "base_seed": 1, "reasoning_budget": 1280,
                           "decoding": {"temperature": 0.6}}}


@pytest.mark.parametrize(("change", "args", "reason"), [
    ({}, None, None),
    ({"system": "b0"}, None, "the system"),
    ({"items": "data/eval/dev-ext.jsonl"}, None, "the items"),
    ({"items_sha256": "0" * 64}, None, "the items' SHA-256"),
    ({"items_run": 2}, None, "records"),  # a --limit run
    ({"tokenizer": "models/only-model"}, None, "the tokenizer"),
    ({"signals": None}, None, "the signals"),
    ({"adapters": []}, None, "the adapters"),
    ({"adapters": [{"sha256": "1" * 64}]}, None, "the adapters"),
    ({"generation": {"mode": "no_think", "base_seed": 1, "reasoning_budget": 1280}}, None, "the mode"),
    ({"generation": {"mode": "think", "base_seed": 2, "reasoning_budget": 1280}}, None, "the base seed"),
    ({"generation": {"mode": "think", "base_seed": 1, "reasoning_budget": 1024}}, None,
     "the reasoning budget"),
    # the greedy thinking run: a sampled record cannot stand for it
    ({}, ["--greedy", "--reasoning-budget", "1280"], "the temperature")])
def test_stale_records_say_what_keeps_them_from_standing_for_a_run(config, repo, change, args, reason):
    out = Path("runs/bakeoff/test/c3/dev-think-s1.jsonl")
    out.parent.mkdir(parents=True)
    records_for(out, {**think_s1_manifest(), **change})
    why = stale_records(config, config.candidate("c3"), out, system="ft", args=args or THINK_S1,
                        gguf_sha256="f" * 64)
    assert why is None if reason is None else reason in why, why


def test_stale_records_need_the_records_and_one_per_item(config, repo):
    out = Path("runs/bakeoff/test/c3/dev-think-s1.jsonl")
    out.parent.mkdir(parents=True)
    assert stale_records(config, config.candidate("c3"), out, system="ft", args=THINK_S1,
                         gguf_sha256="f" * 64) == "no records"
    records_for(out, think_s1_manifest())
    write_jsonl(out, [{"id": "q-1"}])  # fewer records than the manifest says were run: cut short
    why = stale_records(config, config.candidate("c3"), out, system="ft", args=THINK_S1, gguf_sha256="f" * 64)
    assert why == ("runs/bakeoff/test/c3/dev-think-s1.jsonl holds 1 records of 3 run of 3 items, not every "
                   "item once")


MACHINE_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|^[\\/]")  # a drive letter, or a path from a root


def machine_paths(value: object) -> list[str]:
    """Every string of a JSON value, keys included, that holds a drive-letter path or starts at a root."""
    if isinstance(value, dict):
        return [s for key, item in value.items() for s in (*machine_paths(key), *machine_paths(item))]
    if isinstance(value, list):
        return [s for item in value for s in machine_paths(item)]
    return [value] if isinstance(value, str) and MACHINE_PATH.search(value) else []


def test_recorded_names_the_machines_tools_by_placeholders(tmp_path):
    tools = tools_of(python=(tmp_path / "python.exe").as_posix(),
                     llama_server=tmp_path / "bin" / "llama-server.exe", llama_cpp_src=tmp_path / "src")
    export = [tools.python, "scripts/export_adapter.py", "a", "--llama-cpp-src",
              (tmp_path / "src").as_posix()]
    assert recorded(export, tools) == ["python", "scripts/export_adapter.py", "a", "--llama-cpp-src",
                                       "$LLAMA_CPP_SRC"]
    server = [(tmp_path / "bin" / "llama-server.exe").as_posix(), "-m", "models/x.gguf", "-np", "1"]
    assert recorded(server, tools) == ["llama-server", "-m", "models/x.gguf", "-np", "1"]
    # an interpreter inside the repository is named as it is, relative
    assert recorded(["py.exe", "scripts/train.py"], tools_of()) == ["py.exe", "scripts/train.py"]


def test_the_records_hold_no_machine_path_when_the_tools_are_given_by_absolute_paths(config, repo,
                                                                                    tmp_path_factory):
    machine = tmp_path_factory.mktemp("machine")  # the tools live outside the repository
    binary = machine / "llama-server.exe"
    binary.write_bytes(b"llama-server b11349")
    src = machine / "llama.cpp"
    src.mkdir()
    (src / "convert_lora_to_gguf.py").write_text("# the converter\n", encoding="utf-8")
    python = machine / "python.exe"
    gguf = c3_checkpoint(repo)
    smoke_train_outputs(repo / "runs" / "bakeoff" / "test" / "c3" / "smoke" / "train", peaks=[4.0], tps=[400])
    export = FakeExport()

    def train(argv: list[str]) -> tuple[int, str]:
        pilot_outputs(Path(argv[argv.index("--out") + 1]))
        return 0, "trained\n"

    runner = FakeRunner({**manifested_eval_handlers({name: {} for name in C3_RUNS},
                                                    adapter_sha256=sha256_of(gguf)),
                         "export_adapter.py": export, "train.py": train})
    tools = tools_of(runner, served=ServedLlama(FakeLlama([], adapters=[C3_ADAPTER])), python=str(python),
                     llama_server=binary, llama_cpp_src=src)
    s4 = run_smoke(config, config.candidate("c3"), tools, steps=["S4"])["steps"]["S4"]
    evaluated = run_evaluate(config, config.candidate("c3"), tools)
    pilot = run_pilot(config, config.candidate("c2"), tools)
    assert s4["passed"] is True and evaluated["passed"] is True and pilot["passed"] is True
    # what ran named the tools by their paths ...
    command = runner.of("export_adapter.py")[0]
    assert command[0] == str(python) and command[command.index("--llama-cpp-src") + 1] == src.as_posix()
    # ... what the records keep names them by placeholders, beside the SHA-256s that identify them
    assert s4["command"][0] == "python"
    assert s4["command"][s4["command"].index("--llama-cpp-src") + 1] == "$LLAMA_CPP_SRC"
    assert s4["converter_sha256"] == sha256_of(src / "convert_lora_to_gguf.py")
    assert evaluated["server"]["command"][0] == "llama-server"
    assert evaluated["server"]["build"] == "fake-b11349"
    assert evaluated["server"]["binary_sha256"] == sha256_of(binary) and pilot["command"][0] == "python"
    for folder, record in (("c3", "smoke.json"), ("c3", "evaluate.json"), ("c2", "pilot.json")):
        text = (repo / "runs" / "bakeoff" / "test" / folder / record).read_text(encoding="utf-8")
        assert machine_paths(json.loads(text)) == [], record
        assert machine.as_posix() not in text, record


# ---- the gate -----------------------------------------------------------------------------------------


def test_summarize_scores_splits_by_type_and_origin():
    scores = {"summary": {"n": 3}, "scores": [{"id": "q-1", "success": True}, {"id": "q-2", "success": True},
                                               {"id": "q-3", "success": False}]}
    dev = {i["id"]: i for i in ITEMS}
    assert summarize_scores(scores, dev) == {"successes": 2, "n": 3, "handwritten": 1, "drafted": 1,
                                             "by_type": {"call": "1/2", "no_call": "1/1"}}


def gate_fixture(repo: Path, key: str = "c2", *, variants: dict[str, dict[str, bool]], smoke_passed=True,
                 stop: int = 274, tokens: int = 16_000_000) -> Path:
    folder = repo / "runs" / "bakeoff" / "test" / key
    pilot_outputs(folder / "train", stop=stop, tokens=tokens)
    prefix = "dev-" if stop == 274 else f"dev-{stop}-"
    for name, successes in variants.items():
        score_file(folder / f"{prefix}{name}.score.json", successes)
    steps = {s: {"passed": smoke_passed if s != "S3" else True, "exempt": False} for s in STEPS}
    steps["S3"]["peak_reserved_gb"] = 4.72
    (folder / "smoke.json").write_text(json.dumps({"candidate": key, "steps": steps, "order": list(STEPS),
                                                   "passed": smoke_passed, "failed": []}), encoding="utf-8")
    (folder / "pilot.json").write_text(json.dumps({"nvidia_smi_peak_mib": 6016}), encoding="utf-8")
    return folder


def test_candidate_gate_takes_the_mean_of_the_three_seeds_and_the_pilots_throughput(config, repo):
    variants = {"think-s0": {"q-1": True, "q-2": True, "q-3": True}, "think-s1": {"q-1": True, "q-2": True},
                "think-s2": {"q-1": True}, "think-greedy": {"q-1": True, "q-2": True}, "b0": {"q-1": True}}
    folder = gate_fixture(repo, variants=variants)
    gate = candidate_gate(config, config.candidate("c2"))
    assert gate["candidate"] == "c2" and gate["model"] == "Fake/only-model" and gate["revision"] == "def"
    assert gate["family"] == "fake-only" and gate["thinking"] == "only" and gate["mode"] == "think"
    assert gate["stop_step"] == 274 and gate["samples"] == 100 and gate["tokens_per_epoch"] == 16_000_000
    assert gate["modes"] == {"think": 30, "no_think": 70, "think_fallback": 0}
    assert gate["P"] == 2 and gate["primary"] == ["think-s0", "think-s1", "think-s2"]
    assert gate["handwritten"] == 2 / 3 and gate["drafted"] == 4 / 3
    assert gate["variants"]["think-s2"] == {"successes": 1, "n": 3, "handwritten": 0, "drafted": 1,
                                            "by_type": {"call": "1/2", "no_call": "0/1"}}
    assert gate["b0"]["successes"] == 1 and gate["reasoning_budget"] == 16
    # the median of steps 100..274: tokens_per_s 300 + s over s = 100, 110, ..., 270 -> 300 + 185
    assert gate["tokens_per_s_median_100_to_stop"] == 485
    assert gate["ft3_hours_unrounded"] == 16_000_000 * 1.45 * 2 / 485 / 3600
    assert gate["ft3_hours_estimate"] == round(16_000_000 * 1.45 * 2 / 485 / 3600, 1)
    assert gate["peak_reserved_gb"] == round(4.4 * 1.073741824, 2) and gate["nvidia_smi_peak_mib"] == 6016
    assert gate["smoke"]["passed"] is True and gate["eligible"] is True and gate["reasons"] == []
    assert gate["status"] == "scored" and gate["missing"] == []
    assert gate["no_think_greedy"] is None and gate["extension"] is None
    assert gate["files"]["runs/bakeoff/test/c2/dev-think-s0.score.json"] == sha256_of(
        folder / "dev-think-s0.score.json")
    assert "runs/bakeoff/test/c2/train/manifest.json" in gate["files"]
    assert json.loads((folder / "gate.json").read_text(encoding="utf-8")) == gate
    assert "timestamp" not in json.dumps(gate) and "\\\\" not in json.dumps(gate)


def test_candidate_gate_marks_ineligibility_and_reads_the_extension_and_the_hybrid_runs(config, repo):
    variants = {"think-s0": {"q-1": True}, "think-s1": {"q-1": True}, "think-s2": {"q-1": True},
                "think-greedy": {}, "nothink": {"q-1": True, "q-2": True, "q-3": True}, "b0": {}}
    gate_fixture(repo, "c3", variants=variants, smoke_passed=False)
    gate = candidate_gate(config, config.candidate("c3"))
    assert gate["P"] == 1 and gate["no_think_greedy"] == 3 and gate["eligible"] is False
    assert any("S0" in r for r in gate["reasons"]) and gate["family"] is None
    # with every smoke step passed and the extension evaluated at 548, its nothink run too (the hybrid rule
    # reads it there: docs/m7-底座对比预登记.md 修订 3)
    folder = gate_fixture(repo, "c3", variants=variants)
    gate_fixture(repo, "c3", variants={"think-s0": {"q-1": True, "q-2": True}, "think-s1": {"q-1": True},
                                       "think-s2": {"q-1": True, "q-3": True},
                                       "nothink": {"q-1": True, "q-2": True}}, stop=548)
    gate = candidate_gate(config, config.candidate("c3"))
    assert gate["eligible"] is True and gate["extension"] == {
        "stop_step": 548, "P": 5 / 3, "handwritten": 1 / 3, "no_think_greedy": 2,
        "primary": ["think-s0", "think-s1", "think-s2"],
        "variants": {"think-s0": {"successes": 2, "n": 3, "handwritten": 1, "drafted": 1,
                                  "by_type": {"call": "1/2", "no_call": "1/1"}},
                     "think-s1": {"successes": 1, "n": 3, "handwritten": 0, "drafted": 1,
                                  "by_type": {"call": "1/2", "no_call": "0/1"}},
                     "think-s2": {"successes": 2, "n": 3, "handwritten": 0, "drafted": 2,
                                  "by_type": {"call": "2/2", "no_call": "0/1"}},
                     "nothink": {"successes": 2, "n": 3, "handwritten": 1, "drafted": 1,
                                 "by_type": {"call": "1/2", "no_call": "1/1"}}}}
    assert "runs/bakeoff/test/c3/dev-548-think-s0.score.json" in gate["files"]
    # an estimate over the limit makes the candidate ineligible
    rows = [{"event": "start"}] + [{"step": s, "tokens_per_s": 50, "peak_reserved_gib": 4.4}
                                   for s in range(10, 275, 10)]
    write_jsonl(folder / "train" / "log.jsonl", rows)
    gate = candidate_gate(config, config.candidate("c3"))
    assert gate["ft3_hours_estimate"] == round(16_000_000 * 2.9 / 50 / 3600, 1) > 60
    assert gate["eligible"] is False and gate["status"] == "ineligible"
    assert gate["reasons"] == ["the FT3 estimate of 257.78 h is over 60 h"]


def test_a_hybrid_familys_extension_records_its_nothink_run_and_the_gate_waits_for_it(config, repo):
    # docs/m7-底座对比预登记.md 修订 3 (一): after the extension the hybrid rule reads the nothink greedy run
    # at 548, so the extension of a hybrid family carries its successes, and one without them is incomplete,
    # as the stop step is without its nothink run
    variants = {"think-s0": {"q-1": True}, "think-s1": {"q-1": True}, "think-s2": {"q-1": True},
                "think-greedy": {}, "nothink": {"q-1": True, "q-2": True, "q-3": True}, "b0": {}}
    folder = gate_fixture(repo, "c3", variants=variants)
    gate_fixture(repo, "c3", variants={"think-s0": {"q-1": True, "q-2": True}, "think-s1": {"q-1": True},
                                       "think-s2": {"q-1": True, "q-3": True}}, stop=548)
    gate = candidate_gate(config, config.candidate("c3"))
    assert gate["extension"]["P"] == 5 / 3 and gate["extension"]["no_think_greedy"] is None
    assert gate["status"] == "incomplete" and gate["eligible"] is None and gate["reasons"] == []
    assert gate["missing"] == ["no scores for nothink at step 548"]
    busy = "the server could not be used: a server already answers at http://127.0.0.1:8080"
    (folder / "evaluate-548.json").write_text(json.dumps({"passed": False, "error": busy}), encoding="utf-8")
    gate = candidate_gate(config, config.candidate("c3"))
    assert gate["missing"] == [f"no scores for nothink at step 548 (the evaluation failed: {busy})"]
    # its nothink run at 548 scored: the gate is complete and the extension carries the successes
    score_file(folder / "dev-548-nothink.score.json", {"q-1": True, "q-2": True})
    gate = candidate_gate(config, config.candidate("c3"))
    assert gate["status"] == "scored" and gate["eligible"] is True and gate["missing"] == []
    assert gate["extension"]["no_think_greedy"] == 2 and gate["no_think_greedy"] == 3  # 548's beside 274's
    assert gate["extension"]["variants"]["nothink"]["successes"] == 2
    assert "runs/bakeoff/test/c3/dev-548-nothink.score.json" in gate["files"]
    # a thinking-only family's extension needs no nothink run
    seeds = {"think-s0": {"q-1": True}, "think-s1": {"q-1": True}, "think-s2": {"q-1": True}}
    gate_fixture(repo, "c2", variants=seeds)
    gate_fixture(repo, "c2", variants=seeds, stop=548)
    gate = candidate_gate(config, config.candidate("c2"))
    assert gate["status"] == "scored" and gate["extension"]["P"] == 1
    assert gate["extension"]["no_think_greedy"] is None


# 290 tok/s over steps 100..274: 21,600,000 tokens per epoch are exactly 60.0 h, 21,614,400 are 60.04 h (shown
# as 60.0, over the bound all the same) and 21,636,000 are 60.1 h
@pytest.mark.parametrize(("tokens", "shown", "reasons"), [
    (21_600_000, 60.0, []),
    (21_614_400, 60.0, ["the FT3 estimate of 60.04 h is over 60 h"]),
    (21_636_000, 60.1, ["the FT3 estimate of 60.10 h is over 60 h"])])
def test_the_ft3_bound_is_inclusive_and_holds_the_unrounded_estimate(config, repo, tokens, shown, reasons):
    variants = {"think-s0": {"q-1": True}, "think-s1": {"q-1": True}, "think-s2": {"q-1": True}}
    folder = gate_fixture(repo, variants=variants, tokens=tokens)
    write_jsonl(folder / "train" / "log.jsonl", [{"event": "start"}] + [
        {"step": s, "tokens_per_s": 290, "peak_reserved_gib": 4.4} for s in range(10, 275, 10)])
    gate = candidate_gate(config, config.candidate("c2"))
    assert gate["ft3_hours_unrounded"] == tokens * 1.45 * 2 / 290 / 3600
    assert gate["ft3_hours_estimate"] == shown and gate["reasons"] == reasons
    assert gate["eligible"] is (not reasons) and gate["status"] == ("ineligible" if reasons else "scored")


def test_training_rows_keep_the_last_run_and_drop_the_steps_trained_again(tmp_path):
    # scripts/train.py's segment rule (segments_from_log): a run's rows start at its last {"event": "start"};
    # after a resume, the earlier segment counts only up to the step resumed from
    rows = [{"event": "start"},
            {"step": 100, "tokens_per_s": 9999, "peak_reserved_gib": 9.9},  # an aborted earlier run
            {"event": "start"}, {"step": 100, "tokens_per_s": 400, "peak_reserved_gib": 4.0},
            {"step": 110, "tokens_per_s": 410, "peak_reserved_gib": 4.1},
            {"step": 120, "tokens_per_s": 7777, "peak_reserved_gib": 8.8},  # trained again after the resume
            {"event": "resume", "from_checkpoint": "trainer/checkpoint-110", "global_step": 110},
            {"step": 120, "tokens_per_s": 420, "peak_reserved_gib": 4.2}]
    path = write_jsonl(tmp_path / "log.jsonl", rows)
    assert training_rows(path) == [rows[3], rows[4], rows[7]]
    unmarked = write_jsonl(tmp_path / "unmarked.jsonl", rows[3:6])  # no event: one run
    assert training_rows(unmarked) == rows[3:6]
    assert training_rows(tmp_path / "absent.jsonl") == []


def test_the_gate_reads_the_throughput_and_the_peak_of_the_pilots_last_run_alone(config, repo):
    folder = gate_fixture(repo, variants={"think-s0": {"q-1": True}, "think-s1": {}, "think-s2": {}})
    log = folder / "train" / "log.jsonl"
    stale = [{"event": "start"}] + [{"step": s, "tokens_per_s": 9999, "peak_reserved_gib": 9.9}
                                    for s in range(10, 275, 10)]
    log.write_text("".join(json.dumps(r) + "\n" for r in stale) + log.read_text(encoding="utf-8"),
                   encoding="utf-8")
    gate = candidate_gate(config, config.candidate("c2"))
    assert gate["tokens_per_s_median_100_to_stop"] == 485
    assert gate["peak_reserved_gb"] == round(4.4 * GIB_PER_GB, 2)


def test_a_recorded_only_step_that_failed_does_not_count_against_eligibility(config, repo):
    gate_fixture(repo, "c1", variants={"nothink": {"q-1": True, "q-2": True}, "b0": {}})
    folder = repo / "runs" / "bakeoff" / "test" / "c1"
    steps = {s: {"exempt": True, "passed": None, "reason": "G2"} for s in ("S0", "S1", "S2", "S3", "S4")}
    steps |= {"S5": {"passed": False, "error": "the server never came up"}, "S6": {"passed": True}}
    (folder / "smoke.json").write_text(json.dumps({"candidate": "c1", "steps": steps, "order": list(STEPS),
                                                   "passed": False, "failed": ["S5"]}), encoding="utf-8")
    gate = candidate_gate(config, config.candidate("c1"))
    assert gate["P"] == 2 and gate["eligible"] is True and gate["reasons"] == []
    assert gate["smoke_status"] == {"S0": "exempt", "S1": "exempt", "S2": "exempt", "S3": "exempt",
                                    "S4": "exempt", "S5": "failed (recorded only)", "S6": "passed"}
    steps["S5"] = {"passed": False, "error": "x"}
    steps["S6"] = {"passed": False, "error": "y"}
    (folder / "smoke.json").write_text(json.dumps({"candidate": "c1", "steps": steps}), encoding="utf-8")
    gate = candidate_gate(config, config.candidate("c1"))
    assert gate["eligible"] is True and gate["smoke_status"]["S6"] == "failed (recorded only)"
    # the same failures count for a candidate without the rule
    gate_fixture(repo, "c2", variants={"think-s0": {}, "think-s1": {}, "think-s2": {}, "b0": {}})
    (repo / "runs/bakeoff/test/c2/smoke.json").write_text(json.dumps({"candidate": "c2", "steps": {
        **{s: {"passed": True} for s in STEPS}, "S5": {"passed": False}}}), encoding="utf-8")
    gate = candidate_gate(config, config.candidate("c2"))
    assert gate["eligible"] is False and gate["reasons"] == ["S5 failed"]


def test_a_failed_smoke_step_makes_a_candidate_ineligible_without_a_pilot_or_scores(config, repo):
    (repo / "runs" / "bakeoff" / "test" / "c2").mkdir(parents=True)
    (repo / "runs/bakeoff/test/c2/smoke.json").write_text(json.dumps(
        {"steps": {s: {"passed": s != "S4", "exempt": False} for s in STEPS}, "passed": False,
         "failed": ["S4"]}), encoding="utf-8")
    gate = candidate_gate(config, config.candidate("c2"))
    assert gate["P"] is None and gate["eligible"] is False and gate["status"] == "ineligible"
    assert gate["reasons"] == ["S4 failed"]  # the smoke and the hour estimate are the only reasons
    assert gate["missing"] == ["no pilot: runs/bakeoff/test/c2/train/manifest.json is missing",
                               "no scores for think-s0, think-s1, think-s2"]
    assert (repo / "runs/bakeoff/test/c2/gate.json").is_file()


def test_a_candidate_whose_step_waits_for_its_input_is_incomplete_and_the_aggregate_waits_for_it(repo):
    # C2's case (docs/m7-底座对比预登记.md 修订 3, 二.1): its pilot and scores are there, S3 never ran under
    # this tool, and S4 failed for want of S3's adapter: incomplete, never ineligible, and waited for
    config = skipping(repo, c1="cut", c3="cut")  # c2 and the reference alone
    variants = {"think-s0": {"q-1": True, "q-2": True, "q-3": True}, "think-s1": {"q-1": True, "q-2": True},
                "think-s2": {"q-1": True}, "think-greedy": {}, "b0": {}}
    folder = gate_fixture(repo, "c2", variants=variants)
    steps = {s: {"passed": True} for s in STEPS if s != "S3"}
    steps["S4"] = {"passed": False, "error": "no adapter"}
    (folder / "smoke.json").write_text(json.dumps({"candidate": "c2", "steps": steps}), encoding="utf-8")
    gate = candidate_gate(config, config.candidate("c2"))
    assert gate["status"] == "incomplete" and gate["eligible"] is None and gate["reasons"] == []
    assert gate["smoke_status"]["S3"] == "missing" and gate["smoke_status"]["S4"] == "blocked"
    assert gate["missing"] == ["S3 was not run",
                               "S4 is blocked: it uses the output of S3, which has not passed"]
    result = aggregate(config)
    assert result["candidates"]["c2"]["status"] == "incomplete" and result["complete"] is False
    decision = result["decision"]
    assert decision["status"] == "provisional" and decision["winner"] is None
    assert decision["waiting"] == ["c2"]
    assert decision["ranking"] == ["c0", "c2"]  # P 2 each: it ranks with its P meanwhile, not dropped
    # once S3 passed, S4's own failure excludes it, and the decision is final without it
    steps["S3"] = {"passed": True}
    (folder / "smoke.json").write_text(json.dumps({"candidate": "c2", "steps": steps}), encoding="utf-8")
    gate = candidate_gate(config, config.candidate("c2"))
    assert gate["status"] == "ineligible" and gate["reasons"] == ["S4 failed"] and gate["missing"] == []
    result = aggregate(config)
    assert result["complete"] is True and result["decision"]["winner"] == "c0"
    assert result["decision"]["ranking"] == ["c0"]


def test_missing_scores_or_smoke_steps_leave_a_candidate_incomplete_never_ineligible(config, repo):
    # c2's smoke passed and its pilot ran, but the evaluation died after think-s0 (a busy port, a crashed
    # run_eval): nothing the preregistration excludes a candidate for (排除只能出于冒烟或时长资格)
    folder = gate_fixture(repo, variants={"think-s0": {"q-1": True, "q-2": True, "q-3": True}})
    busy = "the server could not be used: a server already answers at http://127.0.0.1:8080"
    (folder / "evaluate.json").write_text(json.dumps({"passed": False, "error": busy}), encoding="utf-8")
    gate = candidate_gate(config, config.candidate("c2"))
    assert gate["status"] == "incomplete" and gate["eligible"] is None and gate["reasons"] == []
    assert gate["P"] is None
    assert gate["missing"] == [f"no scores for think-s1, think-s2 (the evaluation failed: {busy})"]
    # a hybrid family also needs its nothink greedy run: the hybrid rule reads it if the family wins
    gate_fixture(repo, "c3", variants={"think-s0": {}, "think-s1": {}, "think-s2": {}})
    gate = candidate_gate(config, config.candidate("c3"))
    assert gate["status"] == "incomplete" and gate["missing"] == ["no scores for nothink"]
    assert gate["P"] == 0 and gate["no_think_greedy"] is None
    # smoke steps not run yet (no smoke.json at all): incomplete as well
    gate_fixture(repo, "c2", variants={"think-s0": {}, "think-s1": {}, "think-s2": {}})
    (folder / "smoke.json").unlink()
    gate = candidate_gate(config, config.candidate("c2"))
    assert gate["status"] == "incomplete" and gate["missing"] == [f"{s} was not run" for s in STEPS]
    # C1's exempt steps and the ones it runs for the record only are never awaited
    gate_fixture(repo, "c1", variants={"nothink": {}})
    (repo / "runs/bakeoff/test/c1/smoke.json").unlink()
    gate = candidate_gate(config, config.candidate("c1"))
    assert gate["status"] == "scored" and gate["missing"] == [] and gate["eligible"] is True
    assert gate["smoke_status"]["S5"] == "missing (recorded only)" and gate["smoke_status"]["S0"] == "exempt"


# ---- the decision -------------------------------------------------------------------------------------


def entry(key: str, P: float | None, *, eligible: bool = True, thinking: str = "only", handwritten=0.0,
          hours: float = 30.0, no_think_greedy=None, extension=None, reasons=()) -> dict:
    return {"candidate": key, "P": P, "eligible": eligible, "reasons": list(reasons), "thinking": thinking,
            "mode": "no_think" if thinking == "none" else "think", "handwritten": handwritten,
            "ft3_hours_estimate": hours, "no_think_greedy": no_think_greedy, "extension": extension}


def test_the_winner_is_the_highest_p_when_the_top_two_are_more_than_three_apart():
    decision = decide([entry("c2", 111), entry("c0", 107, thinking="hybrid"),
                       entry("c1", 105, thinking="none")], tie=3, hybrid_margin=5)
    assert decision == {"status": "decided", "winner": "c2", "basis": "P", "mode": "think",
                        "ranking": ["c2", "c0", "c1"], "extend": None, "runner_up": "c0", "margin": 4,
                        "note": None, "waiting": []}
    # exactly three apart is within the tie
    assert decide([entry("c2", 110), entry("c0", 107)], tie=3, hybrid_margin=5)["status"] == "extend"


def test_the_top_two_within_three_are_extended_to_the_extension_step():
    decision = decide([entry("c2", 110), entry("c3", 108, thinking="hybrid"), entry("c1", 100)], tie=3,
                      hybrid_margin=5)
    assert decision["status"] == "extend" and decision["extend"] == ["c2", "c3"]
    assert decision["winner"] is None and decision["ranking"] == ["c2", "c3", "c1"]
    assert decision["margin"] == 2
    # one of the two extended already: still waiting for the other
    decision = decide([entry("c2", 110, extension={"P": 115, "handwritten": 40}), entry("c3", 108)],
                      tie=3, hybrid_margin=5)
    assert decision["status"] == "extend" and decision["pending"] == ["c3"]


def test_after_the_extension_p_then_handwritten_then_hours_decide():
    def extended(P: int, hw: int) -> dict:
        return {"P": P, "handwritten": hw}

    decision = decide([entry("c2", 110, extension=extended(112, 30)),
                       entry("c3", 108, extension=extended(118, 28))], tie=3, hybrid_margin=5)
    assert decision["status"] == "decided" and decision["winner"] == "c3" and decision["basis"] == "P@548"
    decision = decide([entry("c2", 110, extension=extended(112, 31)),
                       entry("c3", 108, extension=extended(114, 30))], tie=3, hybrid_margin=5)
    assert decision["winner"] == "c2" and decision["basis"] == "handwritten@548"
    decision = decide([entry("c2", 110, hours=40.0, extension=extended(112, 30)),
                       entry("c3", 108, hours=20.0, extension=extended(114, 30))], tie=3, hybrid_margin=5)
    assert decision["winner"] == "c3" and decision["basis"] == "ft3_hours"
    decision = decide([entry("c2", 110, hours=20.0, extension=extended(112, 30)),
                       entry("c3", 108, hours=20.0, extension=extended(114, 30))], tie=3, hybrid_margin=5)
    assert decision["status"] == "undecided" and decision["winner"] is None
    assert decision["extend"] == ["c2", "c3"]
    # after the extension, exactly three apart is still within the tie: the handwritten items decide
    decision = decide([entry("c2", 110, extension=extended(112, 31)),
                       entry("c3", 108, extension=extended(115, 30))], tie=3, hybrid_margin=5)
    assert decision["winner"] == "c2" and decision["basis"] == "handwritten@548"
    # the hours compare unrounded: 28.14 h against 28.12 h is no tie although both show as 28.1
    slower = entry("c2", 110, hours=28.1, extension=extended(112, 30)) | {"ft3_hours_unrounded": 28.14}
    faster = entry("c3", 108, hours=28.1, extension=extended(114, 30)) | {"ft3_hours_unrounded": 28.12}
    decision = decide([slower, faster], tie=3, hybrid_margin=5)
    assert decision["winner"] == "c3" and decision["basis"] == "ft3_hours"


def test_a_hybrid_winner_answers_without_thinking_only_when_its_greedy_run_leads_by_five():
    decision = decide([entry("c3", 110, thinking="hybrid", no_think_greedy=115), entry("c1", 100)], tie=3,
                      hybrid_margin=5)
    assert decision["winner"] == "c3" and decision["mode"] == "no_think"
    decision = decide([entry("c3", 110, thinking="hybrid", no_think_greedy=114), entry("c1", 100)], tie=3,
                      hybrid_margin=5)
    assert decision["mode"] == "think"
    decision = decide([entry("c1", 110, thinking="none"), entry("c3", 100)], tie=3, hybrid_margin=5)
    assert decision["winner"] == "c1" and decision["mode"] == "no_think"


def test_a_hybrid_winner_decided_after_the_extension_takes_the_mode_the_extensions_numbers_give():
    # docs/m7-底座对比预登记.md 修订 3 (一): when the decision's basis comes from the extension (P@548,
    # handwritten@548, or the hours once both were extended), the hybrid rule reads the extension's numbers,
    # its nothink greedy run against the mean of its three seeds at 548; a winner decided at 274 keeps 274's
    def hybrid(P: float, greedy: int, extension: dict, **more) -> dict:
        return entry("c3", P, thinking="hybrid", no_think_greedy=greedy, extension=extension) | more

    def outcome(decision: dict) -> tuple:
        return decision["winner"], decision["basis"], decision["mode"]

    rival = entry("c2", 109, extension={"P": 110, "handwritten": 30})
    # P@548: 274 says think (112 against 110), 548 says no_think (121 against 116); then the reverse
    c3 = hybrid(110, 112, {"P": 116, "handwritten": 30, "no_think_greedy": 121})
    assert outcome(decide([c3, rival], tie=3, hybrid_margin=5)) == ("c3", "P@548", "no_think")
    c3 = hybrid(110, 115, {"P": 116, "handwritten": 30, "no_think_greedy": 120})
    assert outcome(decide([c3, rival], tie=3, hybrid_margin=5)) == ("c3", "P@548", "think")
    # the handwritten items at 548 (548 says no_think: 117 against 112), the hours once both were extended
    # (548 says think: 113 against 112, where 274 said no_think: 115 against 110)
    c3 = hybrid(110, 112, {"P": 112, "handwritten": 31, "no_think_greedy": 117})
    assert outcome(decide([c3, rival], tie=3, hybrid_margin=5)) == ("c3", "handwritten@548", "no_think")
    c3 = hybrid(110, 115, {"P": 112, "handwritten": 30, "no_think_greedy": 113}, ft3_hours_estimate=20.0)
    assert outcome(decide([c3, rival], tie=3, hybrid_margin=5)) == ("c3", "ft3_hours", "think")
    # an extension without its nothink score gives no switch: think (its gate is incomplete meanwhile)
    c3 = hybrid(110, 115, {"P": 116, "handwritten": 30})
    assert outcome(decide([c3, rival], tie=3, hybrid_margin=5)) == ("c3", "P@548", "think")
    # decided at 274 (more than three ahead): 274's numbers, whatever an extension holds
    c3 = hybrid(115, 120, {"P": 116, "handwritten": 30, "no_think_greedy": 116})
    assert outcome(decide([c3, entry("c2", 110)], tie=3, hybrid_margin=5)) == ("c3", "P", "no_think")
    c3 = hybrid(115, 119, {"P": 116, "handwritten": 30, "no_think_greedy": 125})
    assert outcome(decide([c3, entry("c2", 110)], tie=3, hybrid_margin=5)) == ("c3", "P", "think")
    # a thinking-only winner thinks, and the reference, trained without thinking, keeps no_think
    only = entry("c4", 110, no_think_greedy=130,
                 extension={"P": 118, "handwritten": 30, "no_think_greedy": 130})
    assert outcome(decide([only, rival], tie=3, hybrid_margin=5)) == ("c4", "P@548", "think")
    reference = entry("c0", 110, thinking="hybrid", extension={"P": 118, "handwritten": 30})
    reference["mode"] = "no_think"
    assert outcome(decide([reference, rival], tie=3, hybrid_margin=5)) == ("c0", "P@548", "no_think")


def test_ineligible_candidates_do_not_compete_and_no_candidate_means_the_reference():
    decision = decide([entry("c2", 120, eligible=False, reasons=["S6 failed"]),
                       entry("c0", 107, thinking="hybrid"), entry("c1", 103)], tie=3, hybrid_margin=5)
    assert decision["winner"] == "c0" and decision["ranking"] == ["c0", "c1"] and decision["basis"] == "P"
    decision = decide([entry("c2", None, eligible=False)], tie=3, hybrid_margin=5)
    assert decision == {"status": "none-eligible", "winner": None, "basis": None, "mode": None,
                        "ranking": [], "extend": None, "runner_up": None, "margin": None, "note": None,
                        "waiting": []}
    decision = decide([entry("c2", 110)], tie=3, hybrid_margin=5)
    assert decision["status"] == "decided" and decision["winner"] == "c2" and decision["runner_up"] is None


def test_a_decision_is_provisional_while_any_candidate_is_waiting():
    # the winner comes from every candidate (docs/m7-底座对比预登记.md 胜者): while one has no gate or an
    # incomplete one, the rule's outcome over the others is only a standing, and no winner is named
    entries = [entry("c2", 111), entry("c0", 107, thinking="hybrid"),
               entry("c3", 115, eligible=None)]  # incomplete: its P counts toward the standing
    decision = decide(entries, tie=3, hybrid_margin=5, waiting=["c3", "c5"])
    assert decision["status"] == "provisional" and decision["winner"] is None
    assert decision["waiting"] == ["c3", "c5"]
    assert decision["standing"] == "decided" and decision["leader"] == "c3" and decision["basis"] == "P"
    assert decision["ranking"] == ["c3", "c2", "c0"] and decision["margin"] == 4
    decision = decide(entries[:2], tie=3, hybrid_margin=5, waiting=["c3"])
    assert decision["status"] == "provisional" and decision["leader"] == "c2"
    # an extension in view is no more final while others wait: no leader, the two named
    decision = decide([entry("c2", 110), entry("c0", 108)], tie=3, hybrid_margin=5, waiting=["c4"])
    assert decision["status"] == "provisional" and decision["standing"] == "extend"
    assert decision["leader"] is None and decision["extend"] == ["c2", "c0"]
    assert "leader" not in decide(entries[:2], tie=3, hybrid_margin=5)  # final: a winner, nothing waiting


def test_aggregate_reads_every_gate_and_the_reference_and_writes_the_results_file(config, repo):
    variants = {"think-s0": {"q-1": True, "q-2": True, "q-3": True},
                "think-s1": {"q-1": True, "q-2": True, "q-3": True}, "think-s2": {"q-1": True, "q-2": True},
                "think-greedy": {}, "b0": {}}
    gate_fixture(repo, "c2", variants=variants)
    candidate_gate(config, config.candidate("c2"))
    gate_fixture(repo, "c1", variants={"nothink": {"q-1": True}, "b0": {}})
    candidate_gate(config, config.candidate("c1"))
    result = aggregate(config)
    assert result["config"] == "configs/bakeoff-test.json"
    assert result["config_sha256"] == sha256_of(repo / "configs" / "bakeoff-test.json")
    assert result["rule"] == {"tie": 3, "hybrid_margin": 5, "ft3_hours": 60.0, "extend_step": 548,
                              "smoke_steps": list(STEPS)}
    assert list(result["candidates"]) == ["c0", "c1", "c2", "c3"]
    c0 = result["candidates"]["c0"]
    assert c0["P"] == 2 and c0["eligible"] is True and c0["status"] == "reference" and c0["handwritten"] == 1
    assert c0["score"] == "runs/m7/g2-dev.score.json"
    # the reference's FT3 estimate by the preregistered algorithm, from its own pilot (configuration: train):
    # the manifest's tokens per epoch and the median tokens/s of steps 100..274 of its log
    assert c0["tokens_per_epoch"] == 16_000_000 and c0["tokens_per_s_median_100_to_stop"] == 485
    assert c0["ft3_hours_unrounded"] == 16_000_000 * 1.45 * 2 / 485 / 3600
    assert c0["ft3_hours_estimate"] == round(16_000_000 * 1.45 * 2 / 485 / 3600, 1)
    for name in ("manifest.json", "log.jsonl"):
        assert c0["files"][f"runs/train/ref-pilot/{name}"] == sha256_of(repo / "runs/train/ref-pilot" / name)
    assert result["candidates"]["c2"]["P"] == 8 / 3 and result["candidates"]["c2"]["status"] == "scored"
    assert result["candidates"]["c2"]["gate"] == "runs/bakeoff/test/c2/gate.json"
    assert result["candidates"]["c2"]["gate_sha256"] == sha256_of(repo / "runs/bakeoff/test/c2/gate.json")
    assert result["candidates"]["c3"]["status"] == "pending" and result["complete"] is False
    # c3 has no gate yet: the extension the scored candidates call for is the standing, not a decision
    decision = result["decision"]
    assert decision["status"] == "provisional" and decision["standing"] == "extend"
    assert decision["extend"] == ["c2", "c0"] and decision["waiting"] == ["c3"] and decision["winner"] is None
    written = json.loads((repo / "results" / "bakeoff-test.json").read_text(encoding="utf-8"))
    assert written == result and "timestamp" not in json.dumps(written)
    # the reference's score file must agree with the configured P and the G2 record
    (repo / "runs" / "m7" / "g2.json").write_text(json.dumps({"dev_successes": 3}), encoding="utf-8")
    with pytest.raises(BakeoffError, match="dev_successes"):
        aggregate(config)
    (repo / "runs" / "m7" / "g2.json").write_text(json.dumps({"dev_successes": 2}), encoding="utf-8")
    other = repo / "configs" / "other-p.json"
    other.write_text(json.dumps({**CONFIG, "reference": {**CONFIG["reference"], "P": 3}}), encoding="utf-8")
    with pytest.raises(BakeoffError, match="holds 2 successes, not the configured P 3"):
        aggregate(load_config(other))


def test_a_gate_written_by_another_tool_counts_as_incomplete(config, repo):
    # the pilot launcher's gate (m7_gate.py) has a P but no status and no eligibility: the aggregate waits for
    # this tool's gate rather than taking the candidate as ineligible
    folder = repo / "runs" / "bakeoff" / "test" / "c2"
    folder.mkdir(parents=True)
    (folder / "gate.json").write_text(json.dumps({"candidate": "c2", "P": 111.33, "mode": "think"}),
                                      encoding="utf-8")
    result = aggregate(config)
    c2 = result["candidates"]["c2"]
    assert c2["status"] == "incomplete" and c2["eligible"] is None
    rerun = "run scripts/bakeoff.py gate --candidate c2"
    assert c2["missing"] == [f"runs/bakeoff/test/c2/gate.json has no status (another tool wrote it): {rerun}"]
    assert result["decision"]["status"] == "provisional" and result["decision"]["leader"] == "c2"
    assert result["decision"]["waiting"] == ["c1", "c2", "c3"] and result["complete"] is False


def test_aggregate_is_final_once_every_candidate_is_scored_or_excluded_by_smoke_or_hours(config, repo):
    variants = {"think-s0": {"q-1": True}, "think-s1": {"q-1": True}, "think-s2": {"q-1": True},
                "think-greedy": {}, "b0": {}}
    for key in ("c2", "c3"):  # scored, but a smoke step failed: ineligible
        gate_fixture(repo, key, variants={**variants, **({"nothink": {}} if key == "c3" else {})},
                     smoke_passed=False)
        candidate_gate(config, config.candidate(key))
    # c1 without a pilot: incomplete, so the aggregate waits for it
    (repo / "runs" / "bakeoff" / "test" / "c1").mkdir()
    (repo / "runs/bakeoff/test/c1/smoke.json").write_text(json.dumps(
        {"steps": {s: {"passed": False, "exempt": False} for s in STEPS}, "passed": False,
         "failed": list(STEPS)}), encoding="utf-8")
    candidate_gate(config, config.candidate("c1"))
    result = aggregate(config)
    assert result["candidates"]["c1"]["status"] == "incomplete" and result["complete"] is False
    assert result["decision"]["status"] == "provisional" and result["decision"]["waiting"] == ["c1"]
    # c1 trained so slowly that its FT3 estimate is over 60 h: excluded, and the decision is final
    folder = gate_fixture(repo, "c1", variants={"nothink": {"q-1": True}, "b0": {}})
    write_jsonl(folder / "train" / "log.jsonl", [{"event": "start"}] + [
        {"step": s, "tokens_per_s": 50, "peak_reserved_gib": 4.4} for s in range(10, 275, 10)])
    candidate_gate(config, config.candidate("c1"))
    result = aggregate(config)
    assert result["complete"] is True and result["candidates"]["c1"]["status"] == "ineligible"
    assert result["candidates"]["c2"]["status"] == "ineligible" and result["candidates"]["c2"]["P"] == 1
    assert result["decision"]["status"] == "decided" and result["decision"]["waiting"] == []
    assert result["decision"]["winner"] == "c0" and result["decision"]["basis"] == "P"
    assert result["decision"]["ranking"] == ["c0"] and result["decision"]["runner_up"] is None
    assert result["decision"]["mode"] == "no_think"  # the reference trained without thinking


def skipping(repo: Path, **reasons) -> Config:
    """The test configuration with the candidates of reasons skipped, each for its reason."""
    candidates = {key: {**entry, **({"skipped": reasons[key]} if key in reasons else {})}
                  for key, entry in CONFIG["candidates"].items()}
    path = repo / "configs" / "skipping.json"
    path.write_text(json.dumps({**CONFIG, "candidates": candidates}), encoding="utf-8")
    return load_config(path)


def test_a_candidate_a_cut_line_skips_is_recorded_with_its_reason_and_not_waited_for(repo):
    # the plan's cut line 2 (docs/m7-计划.md 砍线) skips a wave that has fallen more than a day behind: the
    # configuration names each skipped candidate with the reason, the results file records it, and the
    # decision waits for the others alone
    config = skipping(repo, c3="cut line 2: wave 4 more than a day behind")
    assert config.candidates["c3"].skipped == "cut line 2: wave 4 more than a day behind"
    assert config.candidates["c2"].skipped is None
    variants = {"think-s0": {"q-1": True}, "think-s1": {"q-1": True}, "think-s2": {"q-1": True},
                "think-greedy": {}, "b0": {}}
    gate_fixture(repo, "c2", variants=variants)
    candidate_gate(config, config.candidate("c2"))
    gate_fixture(repo, "c1", variants={"nothink": {"q-1": True}, "b0": {}})
    candidate_gate(config, config.candidate("c1"))
    result = aggregate(config)
    assert result["candidates"]["c3"] == {"status": "skipped", "candidate": "c3", "name": "hybrid-model",
                                          "thinking": "hybrid",
                                          "reason": "cut line 2: wave 4 more than a day behind",
                                          "gate": "runs/bakeoff/test/c3/gate.json"}
    assert result["complete"] is True and result["decision"]["waiting"] == []
    assert result["decision"]["status"] == "extend" and result["decision"]["ranking"] == ["c0", "c1", "c2"]
    # the same candidates without the skip: the decision waits for c3
    assert aggregate(load_config(Path("configs/bakeoff-test.json")))["decision"]["waiting"] == ["c3"]


def test_a_candidate_with_a_score_cannot_be_skipped(repo):
    # a cut line skips a candidate before it has a score, never after: a gate with a P refuses the aggregate
    config = skipping(repo, c2="cut line 2")
    variants = {"think-s0": {"q-1": True}, "think-s1": {}, "think-s2": {}, "think-greedy": {}, "b0": {}}
    gate_fixture(repo, "c2", variants=variants)
    candidate_gate(config, config.candidate("c2"))
    with pytest.raises(BakeoffError, match="c2 is skipped .cut line 2. but runs/bakeoff/test/c2/gate.json "
                                           "holds its P"):
        aggregate(config)


@pytest.mark.parametrize("reason", ["", True, 3, ["cut line 2"]])
def test_a_skip_reason_must_be_a_non_empty_string(repo, reason):
    with pytest.raises(BakeoffError, match="candidate c3: skipped must be a non-empty string"):
        skipping(repo, c3=reason)


def test_the_aggregate_gives_a_hybrid_winner_the_mode_of_the_step_that_decided(repo):
    # c2 (thinking only) and c3 (hybrid) tie at 274 and both are extended; c3 wins on P at 548, where its
    # nothink run leads its think mean by the margin (3 against 2), though at 274 it did not (3 against 3).
    # Three items: a tie of 0 and a margin of 1 stand for the preregistered 3 and 5
    path = repo / "configs" / "mode.json"
    thresholds = {**CONFIG["thresholds"], "tie": 0, "hybrid_margin": 1}
    candidates = {**CONFIG["candidates"], "c1": {**CONFIG["candidates"]["c1"], "skipped": "cut"}}
    path.write_text(json.dumps({**CONFIG, "thresholds": thresholds, "candidates": candidates}),
                    encoding="utf-8")
    config = load_config(path)
    every = {"q-1": True, "q-2": True, "q-3": True}

    def seeds(successes: dict[str, bool], **more: dict[str, bool]) -> dict[str, dict[str, bool]]:
        """The three seeds' successes, each the same, and the other variants of more."""
        return {name: successes for name in ("think-s0", "think-s1", "think-s2")} | more

    gate_fixture(repo, "c2", variants=seeds(every))
    gate_fixture(repo, "c2", variants=seeds({"q-1": True}), stop=548)
    gate_fixture(repo, "c3", variants=seeds(every, nothink=every))
    gate_fixture(repo, "c3", variants=seeds({"q-1": True, "q-2": True}, nothink=every), stop=548)
    for key in ("c2", "c3"):
        assert candidate_gate(config, config.candidate(key))["status"] == "scored"
    result = aggregate(config)
    decision = result["decision"]
    assert result["complete"] is True and decision["status"] == "decided"
    assert (decision["winner"], decision["basis"], decision["mode"]) == ("c3", "P@548", "no_think")
    assert decision["ranking"] == ["c2", "c3", "c0"]


# ---- plan and the command line -------------------------------------------------------------------------


def test_plan_lists_every_command_of_a_candidate_without_running_anything(config, repo):
    lines = plan_lines(config, config.candidate("c3"), tools_of())
    text = "\n".join(lines)
    assert "c3" in lines[0] and "hybrid" in lines[0] and "max_length 5120" in text
    assert "reasoning budget: none configured" in text
    for piece in ("S1:", "S2:", "S3:", "S4:", "S5:", "pilot:", "export:", "server:", "think-s2",
                  "think-greedy", "nothink", "b0", "score"):
        assert piece in text, piece
    assert "--stop-at-step 274" in text and "--max-steps 20" in text and "-np 1" in text
    assert "fractions 0.07, 0.14, 0.28, 0.56, 1.12, 2, until 0.35 of its top-1 change" in text
    assert not Path("runs/bakeoff/test/c3").exists()  # a plan writes nothing, no export folder either


def test_the_command_line_runs_the_subcommands_and_stops_all_at_the_first_failure(config, repo, capsys):
    script = load_script("bakeoff")
    tools = tools_of(FakeRunner({"pytest": lambda argv: (5, "no tests ran\n")}))
    config_arg = ["--config", "configs/bakeoff-test.json"]
    assert script.main(["plan", "--candidate", "c2", *config_arg], tools=tools) == 0
    assert "S2:" in capsys.readouterr().out
    assert script.main(["plan", "--candidate", "c9", *config_arg], tools=tools) == 2
    assert "c9" in capsys.readouterr().err
    assert script.main(["smoke", "--candidate", "c2", *config_arg, "--steps", "S1"], tools=tools) == 1
    assert smoke_of(Path("runs/bakeoff/test/c2"))["steps"]["S1"]["passed"] is False
    assert script.main(["all", "--candidate", "c2", *config_arg], tools=tools) == 1
    out = capsys.readouterr().out
    assert "S1" in out and "stopping" in out
    # the smoke's S2 and S3 run train.py; the pilot (--stop-at-step) never started after the failed smoke
    assert tools.run.of("train.py") and all("--stop-at-step" not in cmd for cmd in tools.run.of("train.py"))
    gate = json.loads((repo / "runs/bakeoff/test/c2/gate.json").read_text(encoding="utf-8"))
    assert gate["eligible"] is False
    assert script.main(["aggregate", *config_arg], tools=tools) == 0
    results = json.loads((repo / "results" / "bakeoff-test.json").read_text(encoding="utf-8"))
    assert results["candidates"]["c2"]["status"] == "ineligible"
    assert results["decision"]["status"] == "provisional" and results["decision"]["leader"] == "c0"
    assert results["decision"]["waiting"] == ["c1", "c3"] and results["decision"]["winner"] is None
    assert "provisional, leader c0, waiting for c1, c3" in capsys.readouterr().out
    assert script.main(["gate", "--candidate", "c2", *config_arg], tools=tools) == 0
    assert "ineligible" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        script.main(["aggregate", "--candidate", "c2", *config_arg], tools=tools)


def test_all_runs_only_the_smoke_steps_still_to_pass_and_stops_on_those_that_count(config, repo, capsys):
    script = load_script("bakeoff")
    config_arg = ["--config", "configs/bakeoff-test.json"]
    folder = repo / "runs" / "bakeoff" / "test" / "c2"
    folder.mkdir(parents=True)
    passed = {s: {"passed": True} for s in STEPS}
    (folder / "smoke.json").write_text(json.dumps({"candidate": "c2", "steps": passed}), encoding="utf-8")
    # every step passed: no smoke step runs again, the pilot starts (and fails here: no training stand-in)
    runner = FakeRunner()
    assert script.main(["all", "--candidate", "c2", *config_arg], tools=tools_of(runner)) == 1
    assert "stopping: the pilot failed" in capsys.readouterr().out
    (cmd,) = runner.commands
    assert "--stop-at-step" in cmd and "--dry-run" not in cmd
    # S2 blocked by template refusals within the bound: all reruns S2 alone and stops there
    blocked = {**SUMMARY, "samples": 995, "refused": {"template": 5, "boundary": 0, "bos": 0}}
    (folder / "smoke.json").write_text(json.dumps({"candidate": "c2", "steps": {
        **passed, "S2": {"passed": None, "blocked": "earlier"}}}), encoding="utf-8")
    runner = FakeRunner({"train.py": lambda argv: (0, dry_run_text(blocked))})
    assert script.main(["all", "--candidate", "c2", *config_arg], tools=tools_of(runner)) == 1
    assert "stopping: S2 is blocked" in capsys.readouterr().out
    (cmd,) = runner.commands
    assert "--dry-run" in cmd
    assert script.main(["smoke", "--candidate", "c2", "--steps", "S2", *config_arg],
                       tools=tools_of(runner)) == 1  # a blocked step is no pass
    # C1's steps run for the record do not stop it: past the failed S5 and S6, the pilot starts
    runner = FakeRunner()
    tools = tools_of(runner, served=ServedLlama(FakeLlama([]), exit_code=1))
    assert script.main(["all", "--candidate", "c1", *config_arg], tools=tools) == 1
    out = capsys.readouterr().out
    assert "stopping: the pilot failed" in out and "S5: FAILED" in out
    assert [c for c in runner.commands if "--stop-at-step" in c]


def test_evaluate_on_the_command_line_can_rerun_every_variant(config, repo):
    script = load_script("bakeoff")
    gguf = c3_checkpoint(repo)
    successes = {name: {"q-1": True} for name in C3_RUNS}
    config_arg = ["--config", "configs/bakeoff-test.json"]
    for argv, runs in ((["evaluate", "--candidate", "c3"], 6), (["evaluate", "--candidate", "c3"], 0),
                       (["evaluate", "--candidate", "c3", "--rerun"], 6)):
        runner = FakeRunner(manifested_eval_handlers(successes, adapter_sha256=sha256_of(gguf)))
        tools = tools_of(runner, served=ServedLlama(FakeLlama([], adapters=[C3_ADAPTER])))
        assert script.main([*argv, *config_arg], tools=tools) == 0
        assert len(runner.of("run_eval.py")) == runs, argv


def test_default_tools_need_the_machine_paths(config, repo, monkeypatch):
    script = load_script("bakeoff")
    monkeypatch.delenv("LLAMA_SERVER", raising=False)
    monkeypatch.delenv("LLAMA_CPP_SRC", raising=False)
    assert script.main(["pilot", "--candidate", "c2", "--config", "configs/bakeoff-test.json"]) == 2
    tools = script.build_tools(SimpleNamespace(llama_server="llama-server.exe", llama_cpp_src="llama.cpp",
                                               port=8091, no_wait=True), needs_machine=True)
    assert tools.llama_server == Path("llama-server.exe") and tools.llama_cpp_src == Path("llama.cpp")
    assert tools.port == 8091 and tools.wait_limit == 0.0 and tools.python == bakeoff.rel(sys.executable)
    assert tools.fidelity_backends is bakeoff.default_fidelity_backends and tools.probes is not None
    # from the repository root the interpreter is the project's .venv, a repository-relative path
    monkeypatch.chdir(Path(sys.executable).parents[2])
    tools = script.build_tools(SimpleNamespace(llama_server="llama-server.exe", llama_cpp_src="llama.cpp",
                                               port=8080, no_wait=False), needs_machine=True)
    venv_python = (".venv/Scripts/python.exe" if sys.platform == "win32"
                   else Path(sys.executable).relative_to(Path(sys.executable).parents[2]).as_posix())
    assert tools.python == venv_python and tools.wait_limit == script.WAIT_LIMIT


# ---- the trained adapter's fidelity (report only) ---------------------------------------------------------

TRAINED = "runs/bakeoff/test/c2/train/trainer/checkpoint-274"
TRAINED_GGUF = "runs/bakeoff/test/c2/train/gguf/checkpoint-274-f16.gguf"
TRAINED_RECORD = "runs/bakeoff/test/c2/fidelity-trained-274.json"
COUNTS = ("items", "positions", "agreeing", "agreement", "base_agreeing", "base_agreement",
          "changed_positions", "changed")


def trained_checkpoint(repo: Path, key: str = "c2", step: int = 274, *, gguf: bytes | None = b"GGUF trained ",
                       saved: bool = True) -> Path:
    """A pilot checkpoint of the candidate as scripts/train.py saves one: the PEFT adapter
    (adapter_config.json and adapter_model.safetensors over two modules, its values the step's) with, when
    saved, its trainer_state.json (a save under way has none yet), and with gguf the GGUF the pilot exported
    from it (its bytes the step's); the checkpoint folder."""
    train = repo / "runs" / "bakeoff" / "test" / key / "train"
    checkpoint = train / "trainer" / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    if saved:
        (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}), encoding="utf-8")
    (checkpoint / "adapter_config.json").write_text(json.dumps(
        {"peft_type": "LORA", "r": 2, "lora_alpha": 4, "target_modules": ["q_proj", "v_proj"]}),
        encoding="utf-8")
    tensors = {f"base_model.model.model.layers.0.self_attn.{name}.lora_{which}.weight":
               ((2, 3) if which == "A" else (3, 2), [step / 1000] * 6)
               for name in ("q_proj", "v_proj") for which in "AB"}
    write_safetensors(checkpoint / "adapter_model.safetensors", tensors)
    if gguf is not None:
        (train / "gguf").mkdir(parents=True, exist_ok=True)
        (train / "gguf" / f"checkpoint-{step}-f16.gguf").write_bytes(gguf + str(step).encode())
    return checkpoint


class TrainedBackends:
    """The trained adapter's check's model parts, scripted, and their factory (Tools.fidelity_backends): the
    base predicts position p of an item as p; with the trained adapter (the checkpoint folder on the training
    stack, its GGUF on llama.cpp) the first `changed` positions of each item move to p + 1; llama.cpp then
    predicts -1 at the last `disagree` positions of each item (base_disagree without an adapter); unloaded
    makes the training stack's load check refuse the adapter (AdapterNotLoaded, as the HF backend's
    verify_adapter_loaded raises it). The defaults over items of four positions give an agreement of 3/4 with
    the adapter, 1/4 without and a change of 2/4, and the four arrays handed to compare in any other order
    give other counts. Every prediction is kept with its side and adapter (POSIX, None without) in calls and
    with its sequences in sequences, every factory call with its arguments in made; the parts only S6 uses
    (the modules, the weight norms, the export) fail the test when asked for."""

    def __init__(self, *, changed: int = 2, disagree: int = 1, base_disagree: int = 3,
                 unloaded: bool = False) -> None:
        self.changed, self.disagree, self.base_disagree = changed, disagree, base_disagree
        self.unloaded = unloaded
        self.calls: list[tuple[str, str | None]] = []
        self.sequences: list[list[Sequence]] = []
        self.made: list[tuple[str, Path, dict]] = []

    def __call__(self, config, candidate, tools, folder, **kwargs) -> Backends:
        self.made.append((candidate.key, Path(folder), kwargs))

        def unused(*args):
            pytest.fail("the trained adapter's check asks its backends for predictions alone")

        return Backends(modules=unused, weight_norm=unused, export=unused, hf_top1=self.hf_top1,
                        llama_top1=self.llama_top1)

    def _predict(self, side: str, adapter, sequences: list[Sequence], wrong: int) -> list[list[int]]:
        self.calls.append((side, None if adapter is None else Path(adapter).as_posix()))
        self.sequences.append(sequences)
        moved = 0 if adapter is None else self.changed
        return [[-1 if p >= s.positions - wrong else p + 1 if p < moved else p for p in range(s.positions)]
                for s in sequences]

    def hf_top1(self, adapter, sequences: list[Sequence]) -> list[list[int]]:
        if adapter is not None and self.unloaded:
            self.calls.append(("hf", Path(adapter).as_posix()))
            raise AdapterNotLoaded(f"0 of the 4 LoRA tensors of {Path(adapter).as_posix()}/adapter_model."
                                   "safetensors are in the model")
        return self._predict("hf", adapter, sequences, 0)

    def llama_top1(self, gguf, sequences: list[Sequence]) -> list[list[int]]:
        return self._predict("llama", gguf, sequences, self.base_disagree if gguf is None else self.disagree)


def sequences_sha256(sequences: list[Sequence]) -> str:
    data = canonical_json([dataclasses.asdict(s) for s in sequences]).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def read_record(repo: Path, path: str = TRAINED_RECORD) -> dict:
    return json.loads((repo / path).read_text(encoding="utf-8"))


def test_the_trained_adapters_fidelity_compares_both_stacks_with_and_without_it(config, repo):
    # docs/m7-底座对比预登记.md 修订 1: the winner's pilot adapter, HF (NF4) against llama.cpp (Q8_0) on S6's
    # sequences, the agreement with the adapter beside the one without it; report only
    checkpoint = trained_checkpoint(repo)
    fake = TrainedBackends(changed=2, disagree=1, base_disagree=3)
    runner = FakeRunner()  # the pilot exported the GGUF: nothing to run
    result = bakeoff.run_trained_fidelity(config, config.candidate("c2"),
                                          tools_of(runner, fidelity_backends=fake))
    # two items of four positions: the adapter moves 2 of each item's 4 top-1 tokens on both stacks, and
    # llama.cpp disagrees at 1 of each item's positions with the adapter and at 3 without it
    assert result == {
        "candidate": "c2", "stop_step": 274, "report_only": True, "adapter": TRAINED,
        "adapter_sha256": sha256_of(checkpoint / "adapter_model.safetensors"), "modules": 2,
        "adapter_tensors": 4, "gguf": TRAINED_GGUF, "gguf_sha256": sha256_of(repo / TRAINED_GGUF),
        "exported": False, "export": None, "base_gguf": "models/only-model-Q8_0.gguf",
        "settings": {"items": 2, "positions": 4, "mode": "no_think"},
        "sequences_sha256": sequences_sha256(fake.sequences[0]), "items": 2, "positions": 8, "agreeing": 6,
        "agreement": 0.75, "base_agreeing": 2, "base_agreement": 0.25, "changed_positions": 4,
        "changed": 0.5, "error": None}
    # the training stack first, without the adapter and with the checkpoint (its load checked there), then
    # llama.cpp without it and with the checkpoint's GGUF, all four over the same sequences
    assert fake.calls == [("hf", None), ("hf", TRAINED), ("llama", None), ("llama", TRAINED_GGUF)]
    assert all(s == fake.sequences[0] for s in fake.sequences) and runner.commands == []
    # one set of backends, its llama-server logging beside the record rather than into S6's log
    assert fake.made == [("c2", Path("runs/bakeoff/test/c2"),
                          {"server_log": Path("runs/bakeoff/test/c2/fidelity-trained-274.server.log")})]
    assert read_record(repo) == result and b"\r\n" not in (repo / TRAINED_RECORD).read_bytes()
    assert machine_paths(result) == [] and "timestamp" not in json.dumps(result)
    assert "passed" not in result and "checks" not in result  # no verdict: S6's bounds are not this check's
    log = (repo / "runs" / "bakeoff" / "test" / "c2" / "bakeoff.log").read_text(encoding="utf-8")
    assert ("fidelity (report only): checkpoint-274: agreement 0.750 with the adapter, 0.250 without; the "
            "adapter changed 0.500 of the training stack's top-1 tokens; 8 positions") in log


def test_the_trained_adapters_fidelity_runs_on_exactly_s6s_sequences(config, repo):
    # the same development items, positions and rendering as S6 (eval.fidelity.sequences_of over the smoke
    # settings): the sequences each check hands its backends are equal
    seen: list[list[Sequence]] = []

    def s6_backends(cfg, candidate, tools, folder):
        backends = fake_backends(folder)

        def hf_top1(adapter, sequences):
            seen.append(sequences)
            return backends.hf_top1(adapter, sequences)

        return dataclasses.replace(backends, hf_top1=hf_top1)

    smoke = run_smoke(config, config.candidate("c2"), tools_of(fidelity_backends=s6_backends), steps=["S6"])
    assert smoke["steps"]["S6"]["passed"] is True and seen
    trained_checkpoint(repo)
    fake = TrainedBackends()
    checked = bakeoff.run_trained_fidelity(config, config.candidate("c2"), tools_of(fidelity_backends=fake))
    assert checked["error"] is None and len(fake.sequences) == 4
    assert all(s == seen[0] for s in [*seen, *fake.sequences]) and [s.positions for s in seen[0]] == [4, 4]


def test_the_trained_adapters_gguf_is_exported_first_when_missing_and_no_record_names_a_machine_path(
        config, repo, tmp_path_factory):
    machine = tmp_path_factory.mktemp("machine")  # the tools live outside the repository
    binary = machine / "llama-server.exe"
    binary.write_bytes(b"llama-server b11349")
    src = machine / "llama.cpp"
    src.mkdir()
    (src / "convert_lora_to_gguf.py").write_text("# the converter\n", encoding="utf-8")
    trained_checkpoint(repo, gguf=None)  # trained but never exported: no gguf/ folder either
    assert not (repo / "runs/bakeoff/test/c2/train/gguf").exists()
    fake = TrainedBackends()
    export = FakeExport(b"GGUF exported")

    def exporting(argv: list[str]) -> tuple[int, str]:
        fake.calls.append(("export", argv[argv.index("--outfile") + 1]))
        return export(argv)

    runner = FakeRunner({"export_adapter.py": exporting})
    tools = tools_of(runner, fidelity_backends=fake, python=str(machine / "python.exe"), llama_server=binary,
                     llama_cpp_src=src)
    result = bakeoff.run_trained_fidelity(config, config.candidate("c2"), tools)
    assert result["error"] is None and result["exported"] is True and result["agreement"] == 0.75
    # the export runs first, through run_export (its folder made before the converter ran), into the GGUF the
    # pilot would have written, and llama.cpp is compared with that file afterwards
    assert fake.calls == [("export", TRAINED_GGUF), ("hf", None), ("hf", TRAINED), ("llama", None),
                          ("llama", TRAINED_GGUF)]
    assert export.calls == [(Path(TRAINED_GGUF), True)]
    assert result["gguf_sha256"] == sha256_of(repo / TRAINED_GGUF)
    assert result["gguf_sha256"] == hashlib.sha256(b"GGUF exported").hexdigest()
    (cmd,) = runner.commands
    args = load_script("export_adapter").build_parser().parse_args(script_args(cmd))
    assert args.adapter == TRAINED and args.base == "models/only-model" and args.outfile == TRAINED_GGUF
    assert runner.logs == [Path("runs/bakeoff/test/c2/fidelity-trained-274.export")]
    assert runner.envs[0]["HF_HUB_OFFLINE"] == "1"
    # what ran named the tools by their paths; the record names them by placeholders, beside the
    # converter's SHA-256
    assert cmd[0] == str(machine / "python.exe") and cmd[-1] == src.as_posix()
    assert result["export"] == {"command": ["python", *cmd[1:-1], "$LLAMA_CPP_SRC"], "returncode": 0,
                                "converter_sha256": sha256_of(src / "convert_lora_to_gguf.py")}
    text = (repo / TRAINED_RECORD).read_text(encoding="utf-8")
    assert machine_paths(json.loads(text)) == [] and machine.as_posix() not in text


@pytest.mark.parametrize(("handler", "code"), [(lambda argv: (1, "converter failed\n"), 1),
                                               (FakeExport(written=False), 0)])
def test_a_failed_export_ends_the_trained_adapters_fidelity_before_any_prediction(config, repo, handler,
                                                                                    code):
    trained_checkpoint(repo, gguf=None)
    fake = TrainedBackends()
    result = bakeoff.run_trained_fidelity(config, config.candidate("c2"), tools_of(
        FakeRunner({"export_adapter.py": handler}), fidelity_backends=fake))
    assert result["error"] == f"the export exited with {code} and wrote no {TRAINED_GGUF}"
    assert result["exported"] is True and result["export"]["returncode"] == code
    assert result["gguf_sha256"] is None and all(result[k] is None for k in COUNTS)
    assert fake.calls == [] and fake.made == []  # no backend made, no model loaded, no server started
    assert read_record(repo) == result


def test_an_adapter_the_training_stack_did_not_load_is_no_comparison_and_starts_no_llama_server(config,
                                                                                                  repo):
    # the load check S6 runs on its synthetic adapters (eval.fidelity.verify_adapter_loaded, in the HF
    # backend) holds for the trained one: an adapter PEFT did not place is the record's error, and llama.cpp
    # is not asked
    trained_checkpoint(repo)
    fake = TrainedBackends(unloaded=True)
    result = bakeoff.run_trained_fidelity(config, config.candidate("c2"), tools_of(fidelity_backends=fake))
    assert result["error"] == ("the training stack did not load the trained adapter: 0 of the 4 LoRA "
                               f"tensors of {TRAINED}/adapter_model.safetensors are in the model")
    assert fake.calls == [("hf", None), ("hf", TRAINED)] and all(result[k] is None for k in COUNTS)
    assert result["adapter_sha256"] == sha256_of(repo / TRAINED / "adapter_model.safetensors")
    assert read_record(repo) == result
    log = (repo / "runs" / "bakeoff" / "test" / "c2" / "bakeoff.log").read_text(encoding="utf-8")
    assert "fidelity: FAILED: the training stack did not load the trained adapter" in log


def test_the_trained_adapters_fidelity_reads_the_extensions_checkpoint_once_the_run_was_extended(config,
                                                                                                  repo):
    # docs/m7-底座对比预登记.md 修订 1: checkpoint 274, or 548 when the run was extended (its checkpoint
    # saved, as the gate reads the extension); --stop-step picks either, and no other
    c2 = config.candidate("c2")
    assert bakeoff.trained_step(config, c2) == 274
    trained_checkpoint(repo, step=274)
    trained_checkpoint(repo, step=548, saved=False)  # the extension under way: its save not done
    assert bakeoff.trained_step(config, c2) == 274
    trained_checkpoint(repo, step=548)
    assert bakeoff.trained_step(config, c2) == 548
    assert bakeoff.trained_step(config, c2, 274) == 274 and bakeoff.trained_step(config, c2, 548) == 548
    assert bakeoff.trained_step(config, config.candidate("c3")) == 274  # each candidate's own run decides
    for step in (300, 0):
        refused = f"checkpoint 274 or, after the extension, 548 .*, not {step}$"
        with pytest.raises(BakeoffError, match=refused):
            bakeoff.trained_step(config, c2, step)
    fake = TrainedBackends()
    extended = bakeoff.run_trained_fidelity(config, c2, tools_of(fidelity_backends=fake))
    assert extended["stop_step"] == 548 and extended["error"] is None
    assert extended["adapter"] == "runs/bakeoff/test/c2/train/trainer/checkpoint-548"
    assert extended["gguf"] == "runs/bakeoff/test/c2/train/gguf/checkpoint-548-f16.gguf"
    assert fake.calls[1] == ("hf", extended["adapter"]) and fake.calls[3] == ("llama", extended["gguf"])
    pilot = bakeoff.run_trained_fidelity(config, c2, tools_of(fidelity_backends=TrainedBackends()),
                                         stop_step=274)
    assert pilot["stop_step"] == 274 and pilot["adapter"] == TRAINED and pilot["gguf"] == TRAINED_GGUF
    assert pilot["adapter_sha256"] != extended["adapter_sha256"]
    assert pilot["gguf_sha256"] != extended["gguf_sha256"]
    folder = repo / "runs" / "bakeoff" / "test" / "c2"
    assert sorted(p.name for p in folder.glob("fidelity-trained-*.json")) == [
        "fidelity-trained-274.json", "fidelity-trained-548.json"]
    # the extension's step asked of a candidate never extended: the record says so, and nothing runs
    trained_checkpoint(repo, "c3", 274)
    fake, runner = TrainedBackends(), FakeRunner()
    missing = bakeoff.run_trained_fidelity(config, config.candidate("c3"),
                                           tools_of(runner, fidelity_backends=fake), stop_step=548)
    assert missing["error"] == ("no trained adapter at runs/bakeoff/test/c3/train/trainer/checkpoint-548 (no "
                                "trainer_state.json, adapter_config.json, adapter_model.safetensors): the "
                                "pilot trains it")
    assert fake.calls == [] and fake.made == [] and runner.commands == []
    assert missing["adapter_sha256"] is None
    assert read_record(repo, "runs/bakeoff/test/c3/fidelity-trained-548.json") == missing


def test_the_trained_adapters_fidelity_changes_no_smoke_record_gate_aggregate_or_eligibility(config, repo):
    # report only: smoke.json, the pilot's and the evaluation's records, the gate and the results file keep
    # their bytes, whatever the check finds, and a gate and an aggregate written again afterwards are the
    # same bytes: the record does not enter them
    variants = {"think-s0": {"q-1": True}, "think-s1": {"q-1": True, "q-2": True}, "think-s2": {"q-1": True},
                "think-greedy": {}, "b0": {}}
    folder = gate_fixture(repo, "c2", variants=variants)
    trained_checkpoint(repo)
    gate = candidate_gate(config, config.candidate("c2"))
    aggregate(config)
    results = repo / "results" / "bakeoff-test.json"

    def files() -> dict[Path, bytes]:
        return {p: p.read_bytes() for p in [*folder.rglob("*"), results] if p.is_file()}

    before = files()
    for fake in (TrainedBackends(), TrainedBackends(unloaded=True)):
        bakeoff.run_trained_fidelity(config, config.candidate("c2"), tools_of(fidelity_backends=fake))
        after = files()
        assert sorted(p.relative_to(folder).as_posix() for p in set(after) - set(before)) == [
            "bakeoff.log", "fidelity-trained-274.json"]
        assert all(after[p] == data for p, data in before.items())
    assert candidate_gate(config, config.candidate("c2")) == gate and aggregate(config)["complete"] is False
    assert all(files()[p] == data for p, data in before.items())  # the gate and the results, as they were
    assert gate["status"] == "scored" and gate["eligible"] is True
    assert not any("fidelity" in name for name in gate["files"])


def test_the_default_backends_log_llama_server_into_s6s_log_unless_told_otherwise(config, repo):
    # S6 keeps its llama-server log (smoke/S6.server.log) as before; the trained adapter's check names its own
    sequences = [Sequence(ids=[5, 6, 7], start=2, positions=1)]
    candidate, folder = config.candidate("c3"), config.folder("c3")
    s6_log = repo / "runs" / "bakeoff" / "test" / "c3" / "smoke" / "S6.server.log"
    own_log = Path("runs/bakeoff/test/c3/fidelity-trained-274.server.log")
    served = ServedLlama(FakeLlama([Scripted(("a",), ids=(11,))]))
    backends = bakeoff.default_fidelity_backends(config, candidate, tools_of(served=served), folder)
    assert backends.llama_top1(None, sequences) == [[11]]
    assert s6_log.is_file() and not own_log.exists()
    s6_log.write_text("S6's own log\n", encoding="utf-8")
    served = ServedLlama(FakeLlama([Scripted(("b",), ids=(12,))]))
    backends = bakeoff.default_fidelity_backends(config, candidate, tools_of(served=served), folder,
                                                 server_log=own_log)
    assert backends.llama_top1(None, sequences) == [[12]]
    assert own_log.is_file() and s6_log.read_text(encoding="utf-8") == "S6's own log\n"


def test_plan_lists_the_trained_adapters_fidelity_at_the_checkpoint_it_would_read(config, repo):
    c2 = config.candidate("c2")
    lines = plan_lines(config, c2, tools_of())
    fidelity = [line for line in lines if line.startswith("fidelity")]
    assert len(fidelity) == 4 and lines[-4:] == fidelity  # after the evaluation's commands
    describe, export, base, adapted = fidelity
    assert describe.startswith(f"fidelity: report only, the trained adapter {TRAINED} (the extension's "
                               "checkpoint-548 once the run is extended)")
    assert "S6's 2 items x 4 positions" in describe and describe.endswith(f"into {TRAINED_RECORD}")
    assert export.startswith("fidelity export (when the GGUF is missing): py.exe scripts/export_adapter.py "
                             f"{TRAINED} --base models/only-model --outfile {TRAINED_GGUF}")
    served = "llama-server.exe -m models/only-model-Q8_0.gguf"
    assert base.startswith(f"fidelity server (no adapter): {served}")
    assert "--lora" not in base and base.endswith(" -np 1")
    assert adapted.startswith(f"fidelity server (the adapter): {served}")
    assert f"--lora {TRAINED_GGUF} --lora-init-without-apply -np 1" in adapted
    assert not Path("runs/bakeoff/test/c2").exists()  # a plan writes nothing
    trained_checkpoint(repo, step=548, gguf=None)
    before = sorted(p.as_posix() for p in Path("runs/bakeoff/test/c2").rglob("*"))
    fidelity = [line for line in plan_lines(config, c2, tools_of()) if line.startswith("fidelity")]
    assert "checkpoint-548 (the extension's)" in fidelity[0] and "fidelity-trained-548.json" in fidelity[0]
    assert all("checkpoint-274" not in line for line in fidelity)
    assert "checkpoint-548-f16.gguf" in fidelity[1] and "checkpoint-548-f16.gguf" in fidelity[3]
    assert sorted(p.as_posix() for p in Path("runs/bakeoff/test/c2").rglob("*")) == before


def test_fidelity_on_the_command_line_reports_and_exits_by_the_outcome(config, repo, capsys, monkeypatch):
    script = load_script("bakeoff")
    config_arg = ["--config", "configs/bakeoff-test.json"]
    trained_checkpoint(repo)
    assert script.main(["fidelity", "--candidate", "c2", *config_arg],
                       tools=tools_of(fidelity_backends=TrainedBackends())) == 0
    assert ("fidelity (report only): checkpoint-274: agreement 0.750 with the adapter, 0.250 without"
            in capsys.readouterr().out)
    assert read_record(repo)["error"] is None
    # the extension's checkpoint of a run never extended: the check could not be made
    assert script.main(["fidelity", "--candidate", "c2", "--stop-step", "548", *config_arg],
                       tools=tools_of(fidelity_backends=TrainedBackends())) == 1
    out = capsys.readouterr().out
    assert "no trained adapter at runs/bakeoff/test/c2/train/trainer/checkpoint-548" in out
    assert script.main(["fidelity", "--candidate", "c2", *config_arg],
                       tools=tools_of(fidelity_backends=TrainedBackends(unloaded=True))) == 1
    capsys.readouterr()
    # a step the preregistration does not name, and the reference, are usage errors: nothing runs
    assert script.main(["fidelity", "--candidate", "c2", "--stop-step", "300", *config_arg],
                       tools=tools_of(fidelity_backends=TrainedBackends())) == 2
    assert "not 300" in capsys.readouterr().err
    assert not (repo / "runs/bakeoff/test/c2/fidelity-trained-300.json").exists()
    assert script.main(["fidelity", "--candidate", "c0", *config_arg], tools=tools_of()) == 2
    assert "reference" in capsys.readouterr().err
    # it serves and may export: the machine's tools are needed, as for smoke, pilot and evaluate
    monkeypatch.delenv("LLAMA_SERVER", raising=False)
    monkeypatch.delenv("LLAMA_CPP_SRC", raising=False)
    assert script.main(["fidelity", "--candidate", "c2", *config_arg]) == 2
    assert "llama-server" in capsys.readouterr().err

"""scripts/freeze_final.py offline, in a temporary folder that holds two locked development sets, a fake
family's model folder, an adapter and a base model, the M7 configuration of the final system and the runner
records of its candidates (a greedy run and three sample runs per set) with their manifests, all made here:
the frozen file it writes (the configuration's identity and, per item, the successes under both scorers and
the output hash of final's records, which are those scripts/vote_offline.py makes from the same runs, and of
finalg's, the greedy runs'), a final without votes frozen from the greedy runs alone, its refusals (a manifest
that disagrees with the configuration, a run without a record of an item, the wrong number of runs) and
run_test.output_hash, which ignores the timing, the usage, the signals, the vote's bookkeeping and the call
ids. Nothing here reads a real run or starts a server."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from support.llama_fakes import install_fake_family

from research_agent.agent.prompts import PROMPT_HASH, system_prompt
from research_agent.agent.rules import RULES, RULES_VERSION
from research_agent.agent.vote import voted_settings
from research_agent.eval.lock import read_jsonl, sha256_of, write_jsonl, write_lock
from research_agent.eval.metrics import SCORERS, score_items
from research_agent.eval.systems import with_votes
from research_agent.family.core import GREEDY, Decoding, Mode, load_family
from research_agent.llm.generation import GenerationSettings
from research_agent.tools.schemas import SCHEMA_HASH

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


freeze_final = load_script("freeze_final")
vote_offline = load_script("vote_offline")
output_hash = freeze_final.run_test.output_hash

TODAY = "2026-10-10"
SYSTEM = {"role": "system", "content": system_prompt(TODAY)}
ARGS = {"query": "hallucination", "max_results": 5}
SEARCH = {"name": "search_arxiv", "arguments": ARGS, "arguments_json": json.dumps(ARGS)}
FIND = {"content": "", "tool_calls": [SEARCH]}
BYE = {"content": "See you!", "tool_calls": []}
ADAPTER = "runs/train/ft3/gguf/checkpoint-1635-f16.gguf"
BASE, TOKENIZER = "models/Base-Q8_0.gguf", "models/Base"
DEV, DEV_V2 = "data/eval/dev.jsonl", "data/eval/v2/dev-v2.jsonl"
CONFIG, FROZEN = "results/test-config-m7.json", "results/final-m7.json"
SAMPLING = Decoding(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
GENERATION = {"mode": "think", "decoding": "sampled", "reasoning_budget": 64, "base_seed": 0}
K = 3  # the samples of the fixture's vote


def user(text: str) -> list[dict]:
    return [SYSTEM, {"role": "user", "content": text}]


def item(id_: str, messages: list[dict], expected: dict, type_: str, *, handwritten: bool = False) -> dict:
    source = {"handwritten": True} if handwritten else {"request": id_.rsplit("-s", 1)[0], "step": 1}
    family = f"handwritten/{type_}/en" if handwritten else "find/en"
    return {"id": id_, "messages": messages, "expected": expected, "type": type_, "lang": "en",
            "family": family, "split": "dev", "slots": {"topic": "hallucination"}, "source": source}


def call(arguments: dict = ARGS, id_: str = "call_0") -> dict:
    return {"id": id_, "type": "function",
            "function": {"name": "search_arxiv", "arguments": json.dumps(arguments)}}


# the search the history ran, which met a transient error: the runtime rules retry it (agent.rules)
FAILED_SEARCH = [*user("Find hallucination papers."),
                 {"role": "assistant", "content": "", "tool_calls": [call(id_="c1")]},
                 {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"error": "RATE_LIMITED"})}]
SETS = {DEV: [item("req-dev-0001-s1", user("Find hallucination papers."), FIND, "call"),
              item("req-dev-0002-s1", FAILED_SEARCH, FIND, "recover"),
              item("hw-dev-001", user("Thanks, that's all."), BYE, "no_call", handwritten=True)],
        DEV_V2: [item("req-dev-0401-s1", user("Search hallucination papers."), FIND, "call"),
                 item("req-dev-0402-s1", user("That's all for now, thanks."), BYE, "no_call")]}
RIGHT, CATS = ("", (call(),)), ("", (call({"query": "cats", "max_results": 5}),))
BUSY, GOODBYE = ("The service is busy, try again later.", ()), ("See you!", ())
# per item, what the greedy candidate and samples 0 to 2 answer. Greedy: 3 of 5 right; with the rules 4 (the
# retry). The vote without the rules: 3 (the busy text wins req-dev-0002-s1, the call req-dev-0402-s1); with
# them 4, the retry of every candidate winning req-dev-0002-s1.
ANSWERS = {"req-dev-0001-s1": [RIGHT] * 4, "req-dev-0002-s1": [BUSY, BUSY, RIGHT, BUSY],
           "hw-dev-001": [GOODBYE] * 4, "req-dev-0401-s1": [CATS, RIGHT, RIGHT, CATS],
           "req-dev-0402-s1": [GOODBYE, RIGHT, RIGHT, GOODBYE]}
CANDIDATES = ("greedy", *(f"s{k}" for k in range(K)))
RUNS = {path: [f"runs/m7/{Path(path).stem}-{name}.jsonl" for name in CANDIDATES] for path in SETS}


def record(row: dict, answer: tuple[str, tuple], candidate: int) -> dict:
    """A runner record (eval.runner) of a thinking local model's answer, with signals, as --signals runs
    write them."""
    content, calls = answer
    output = {"role": "assistant", "content": content, **({"tool_calls": list(calls)} if calls else {})}
    return {"id": row["id"], "status": "ok",
            "item": {k: v for k, v in row.items() if k not in ("id", "messages")},
            "output": output, "usage": {"prompt_tokens": 900, "completion_tokens": 20 + candidate},
            "paid": False, "cached": False, "model": "fake-hybrid:lora=1",
            "reasoning": f"Candidate {candidate}.",
            "thinking": {"mode": "think", "reasoning_tokens": 2, "answer_tokens": 1, "forced": False,
                         "phases": 1, "seeds": [candidate]},
            "signals": {"min_decision_prob": 0.9, "mean_logprob": -0.1, "hard_reason": None}}


def settings(family) -> GenerationSettings:
    return GenerationSettings.for_family(family, mode="think", reasoning_budget=64, base_seed=0)


def manifest(family, candidate: GenerationSettings, items_path: str, **changes) -> dict:
    """A run's manifest as scripts/run_eval.py writes it, but for what the checks do not read."""
    n = len(SETS[items_path])
    data = {"adapters": [{"id": 0, "path": ADAPTER, "sha256": sha256_of(ADAPTER)}], "backend": "llama.cpp",
            "family": family.record(), "generation": candidate.record(), "items": items_path,
            "items_locked": True, "items_run": n, "items_sha256": sha256_of(items_path), "items_total": n,
            "limit": None, "lora_scale": 1.0, "model": "fake-hybrid:lora=1", "prompt_hash": PROMPT_HASH,
            "schema_hash": SCHEMA_HASH, "server_props": {"model_path": BASE, "total_slots": 1},
            "signals": True, "statuses": {"ok": n}, "system": "ft"}
    data.update(changes)
    return data


def manifest_path(run: str) -> Path:
    return Path(run).with_name(Path(run).stem + ".manifest.json")


def configuration(final: dict | None = None, **changes) -> dict:
    """The M7 configuration of the final system, its frozen file not written yet (freezing)."""
    final = final if final is not None else {
        "votes": K, "vote_mode": "final", "rules": True, "glossary": None,
        "frozen": {"path": FROZEN, "sha256": "<written by scripts/freeze_final.py>"}}
    config = {"milestone": "M7", "purpose": "M7: the final local system (a test)", "test_sets": "v2",
              "scorers": ["normalized"],
              "systems": {"final": final, "finalg": {"decoding": "greedy", "end_to_end": False}},
              "adapters": {"final": {"gguf": ADAPTER, "sha256": sha256_of(ADAPTER), "selection": None,
                                     "model": BASE, "tokenizer": TOKENIZER, "generation": GENERATION}},
              "server": {"model": "models/Qwen3-1.7B-Q8_0.gguf", "port": 8080, "ctx_size": 8192,
                         "timeout_s": 30, "llama_cpp_build": None},
              "tokenizer": "models/Qwen3-1.7B", "b2": {"max_cost_rmb": 5.0},
              "e2e": {"seed": 42, "today": None}, "statistics": {"seed": 42, "resamples": 100},
              "comparisons": [{"name": "final-vs-finalg", "a": "final", "b": "finalg", "scorer": "normalized",
                               "descriptive": True}],
              "gpu": {"idle_max_mib": 1500, "sample_ms": 200}}
    config.update(changes)
    return config


def write_config(config: dict) -> None:
    Path(CONFIG).parent.mkdir(parents=True, exist_ok=True)
    Path(CONFIG).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """The temporary folder as the current one: the sets, the model files, the configuration and the runs;
    the fake family, its think mode sampling as SAMPLING."""
    monkeypatch.chdir(tmp_path)
    install_fake_family(monkeypatch, tmp_path / TOKENIZER,
                        decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY})
    for path, rows in SETS.items():
        write_jsonl(rows, path)
        write_lock(path)
    for path, data in ((ADAPTER, b"GGUF ft3 step 1635"), (BASE, b"GGUF base")):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)
    family = load_family(TOKENIZER)
    greedy, samples = voted_settings(with_votes(settings(family), K))
    for path, rows in SETS.items():
        for candidate, (run, generation) in enumerate(zip(RUNS[path], (greedy, *samples), strict=True)):
            write_jsonl([record(row, ANSWERS[row["id"]][candidate], candidate) for row in rows], run)
            manifest_path(run).write_text(json.dumps(manifest(family, generation, path)), encoding="utf-8")
    write_config(configuration())
    return family


def runs_arguments(runs: dict[str, list[str]] = RUNS) -> list[str]:
    return [arg for path, paths in runs.items() for arg in ("--runs", path, *paths)]


def entries(records: list[dict], items: list[dict]) -> list[dict]:
    """The frozen file's entry of every record: its id, its success under each scorer, its output hash."""
    scores = {scorer: score_items(records, items, scorer) for scorer in SCORERS}
    return [{"id": r["id"], "success": {scorer: scores[scorer][k]["success"] for scorer in SCORERS},
             "output_sha256": output_hash(r)} for k, r in enumerate(records)]


def test_the_frozen_file_holds_the_configuration_and_the_vectors_of_vote_offlines_records(repo, capsys):
    family = repo
    assert freeze_final.main(["--config", CONFIG, *runs_arguments()]) == 0
    frozen = json.loads(Path(FROZEN).read_text(encoding="utf-8"))
    final = with_votes(settings(family), K)
    assert frozen["adapter"] == {"path": ADAPTER, "sha256": sha256_of(ADAPTER)}
    assert frozen["base"] == {"path": BASE, "sha256": sha256_of(BASE)}
    assert (frozen["tokenizer"], frozen["family"]) == (TOKENIZER, family.record())
    assert (frozen["generation"], frozen["votes"], frozen["vote_mode"]) == (final.record(), K, "final")
    assert frozen["generation"]["votes"] == K and frozen["glossary"] is None
    assert frozen["rules"] == {"version": RULES_VERSION, "names": list(RULES)}
    assert frozen["dev_sets"] == [{"path": path, "sha256": sha256_of(path)} for path in (DEV, DEV_V2)]
    assert frozen["scorers"] == list(SCORERS)
    assert frozen["runs"] == [{"set": path, "greedy": {"path": runs[0], "sha256": sha256_of(runs[0])},
                               "samples": [{"path": run, "sha256": sha256_of(run)} for run in runs[1:]]}
                              for path, runs in RUNS.items()]
    # final's records are those scripts/vote_offline.py makes from the same runs (the vote over the ruled
    # candidates), item by item, dev then dev-v2; finalg's the greedy runs' as they are
    voted, greedy = [], []
    for path, rows in SETS.items():
        runs = [read_jsonl(run) for run in RUNS[path]]
        out, _ = vote_offline.voted_records(rows, runs[0], runs[1:], "final", True)
        voted += entries(out, rows)
        greedy += entries(runs[0], rows)
    assert frozen["final"]["items"] == voted and frozen["finalg"]["items"] == greedy
    assert [e["id"] for e in voted] == [row["id"] for rows in SETS.values() for row in rows]
    assert frozen["final"]["successes"] == {"default": 4, "normalized": 4}
    assert frozen["finalg"]["successes"] == {"default": 3, "normalized": 3}
    assert frozen["finalg"]["generation"] == voted_settings(final)[0].record()
    out = capsys.readouterr().out
    assert "final: default 4, normalized 4 of 5" in out and "finalg: default 3, normalized 3 of 5" in out
    assert Path(FROZEN).read_bytes().endswith(b"}\n") and b"\r\n" not in Path(FROZEN).read_bytes()


def test_a_final_without_votes_is_frozen_from_the_greedy_runs_alone_its_records_ruled(repo):
    family = repo
    final = {"decoding": "greedy", "rules": True, "frozen": {"path": FROZEN, "sha256": "0" * 64}}
    write_config(configuration(final))
    greedy_runs = {path: runs[:1] for path, runs in RUNS.items()}
    assert freeze_final.main(["--config", CONFIG, *runs_arguments(greedy_runs)]) == 0
    frozen = json.loads(Path(FROZEN).read_text(encoding="utf-8"))
    assert (frozen["votes"], frozen["vote_mode"]) == (0, None)
    assert frozen["generation"] == voted_settings(with_votes(settings(family), K))[0].record()
    assert frozen["final"]["successes"] == {"default": 4, "normalized": 4}  # the greedy answers, ruled
    assert frozen["finalg"]["successes"] == {"default": 3, "normalized": 3}
    assert all(run["samples"] == [] for run in frozen["runs"])


def run_of(path: str, candidate: int) -> str:
    return RUNS[path][candidate]


def changed_manifest(path: str, candidate: int, **changes):
    def change(family) -> None:
        target = manifest_path(run_of(path, candidate))
        data = json.loads(target.read_text(encoding="utf-8"))
        for key, value in changes.items():
            if callable(value):
                value = value(data[key])
            data[key] = value
        target.write_text(json.dumps(data), encoding="utf-8")

    return change


def other_family(record: dict) -> dict:
    return {**record, "template_sha256": "0" * 64}


def seed_of(base_seed: int):
    def generation(record: dict) -> dict:
        return {**record, "base_seed": base_seed}

    return generation


@pytest.mark.parametrize("change, field", [
    (changed_manifest(DEV_V2, 2, adapters=[{"id": 0, "path": ADAPTER, "sha256": "1" * 64}]), "adapters"),
    (changed_manifest(DEV_V2, 2, lora_scale=0.0), "lora_scale"),
    (changed_manifest(DEV, 0, server_props={"model_path": "models/Other-Q8_0.gguf"}),
     "server_props.model_path"),
    (changed_manifest(DEV, 1, family=other_family), "family"),
    (changed_manifest(DEV_V2, 2, generation=seed_of(2)), "generation"),  # sample 1 made with seed 2
    (changed_manifest(DEV, 3, items_sha256="0" * 64), "items_sha256"),
    (changed_manifest(DEV, 3, items_run=2, limit=2), "items_run"),
    (changed_manifest(DEV, 0, prompt_hash="0" * 16), "prompt_hash"),
    (changed_manifest(DEV, 0, schema_hash="0" * 16), "schema_hash"),
    (changed_manifest(DEV, 1, rules={"version": RULES_VERSION, "names": list(RULES)}), "rules"),
    (changed_manifest(DEV, 1, vote_mode="final"), "vote_mode"),
], ids=["adapter", "lora-scale", "base", "family", "generation", "items", "items-run", "prompt", "schema",
        "rules", "vote-mode"])
def test_a_manifest_that_disagrees_with_the_configuration_is_refused_with_exit_2_naming_it(repo, capsys,
                                                                                            change, field):
    change(repo)
    assert freeze_final.main(["--config", CONFIG, *runs_arguments()]) == 2
    out = capsys.readouterr().out
    assert field in out and "FAIL" in out and not Path(FROZEN).exists()


def test_a_manifest_that_is_missing_is_refused_with_exit_2(repo, capsys):
    manifest_path(run_of(DEV, 2)).unlink()
    assert freeze_final.main(["--config", CONFIG, *runs_arguments()]) == 2
    assert manifest_path(run_of(DEV, 2)).as_posix() in capsys.readouterr().out and not Path(FROZEN).exists()


def test_a_run_without_a_record_of_every_item_is_refused_with_exit_2_counting_them(repo, capsys):
    run = run_of(DEV, 1)
    write_jsonl(read_jsonl(run)[:-1], run)
    assert freeze_final.main(["--config", CONFIG, *runs_arguments()]) == 2
    out = capsys.readouterr().out
    assert f"{run} holds no record of 1 of the 3 items" in out and not Path(FROZEN).exists()


def test_the_runs_must_be_a_greedy_run_and_one_per_sample_for_every_development_set(repo, capsys):
    short = {path: runs[:-1] for path, runs in RUNS.items()}
    assert freeze_final.main(["--config", CONFIG, *runs_arguments(short)]) == 2
    assert f"{DEV}: 2 sample runs, not the {K} of the vote" in capsys.readouterr().out
    assert freeze_final.main(["--config", CONFIG, *runs_arguments({DEV: RUNS[DEV]})]) == 2
    assert f"no runs of {DEV_V2}" in capsys.readouterr().out
    assert not Path(FROZEN).exists()


def test_without_runs_given_the_runs_are_ft3s_at_step_1635_greedy_and_seeds_0_to_4():
    dev, dev_v2 = (freeze_final.DEFAULT_RUNS[path] for path in (DEV, DEV_V2))
    assert dev[0] == "runs/m7/dev-ft3-1635-greedy.jsonl"
    assert dev_v2[0] == "runs/m7/dev-v2-ft3-1635-greedy.jsonl"
    assert dev[1:3] == ("runs/eval/select-ft3/dev/checkpoint-1635.s0.jsonl",
                        "runs/eval/select-ft3/dev/checkpoint-1635.s1.jsonl")
    assert dev_v2[3:] == tuple(f"runs/m7/voting/dev-v2-ft3-1635-s{k}.jsonl" for k in (2, 3, 4))


def test_a_configuration_without_the_final_system_or_malformed_is_refused(repo, capsys):
    config = configuration()
    config["systems"]["final"]["votes"] = 0
    write_config(config)
    assert freeze_final.main(["--config", CONFIG, *runs_arguments()]) == 1
    assert "systems.final.votes must be an integer of at least 1" in capsys.readouterr().out
    write_config(configuration(systems={"finalg": {"decoding": "greedy", "end_to_end": False}, "b2": {}},
                               comparisons=[{"name": "x", "a": "finalg", "b": "b2", "descriptive": True}]))
    assert freeze_final.main(["--config", CONFIG, *runs_arguments()]) == 1  # finalg runs only beside final
    assert "finalg is held against final's frozen file" in capsys.readouterr().out
    no_final = configuration(systems={"ft2": {}, "b2": {}},
                             comparisons=[{"name": "x", "a": "ft2", "b": "b2", "descriptive": True}])
    no_final["adapters"]["ft2"] = {"gguf": "runs/train/ft2/gguf/checkpoint-1644-f16.gguf", "sha256": "1" * 64,
                                   "selection": None}
    write_config(no_final)
    assert freeze_final.main(["--config", CONFIG, *runs_arguments()]) == 2
    assert "runs no final system" in capsys.readouterr().out and not Path(FROZEN).exists()


# ---- the output hash --------------------------------------------------------------------------------------

ANSWERED = record(SETS[DEV][0], RIGHT, 0)


def test_the_output_hash_ignores_timing_usage_signals_the_votes_and_the_call_ids():
    digest = output_hash(ANSWERED)
    assert len(digest) == 64 and digest == output_hash(json.loads(json.dumps(ANSWERED)))
    same = {**ANSWERED, "usage": {"prompt_tokens": 1}, "signals": None, "paid": True, "cached": True,
            "wall_ms": 12.5, "timings": {"prompt_ms": 3.0}, "latency_ms": 9, "model": "other",
            "thinking": {"seeds": [7]}, "votes": {"k": 5, "counts": {}}, "rules": [], "route": {"x": 1},
            "item": {}}
    assert output_hash(same) == digest
    renamed = json.loads(json.dumps(ANSWERED))
    renamed["output"]["tool_calls"][0]["id"] = "pp_0"
    function = renamed["output"]["tool_calls"][0]["function"]
    function["arguments"] = '{"max_results":5,  "query":"hallucination"}'
    assert output_hash(renamed) == digest  # the ids differ between runs; the arguments are compared parsed


@pytest.mark.parametrize("change", [
    lambda r: r["output"].update(content="Here you are."),
    lambda r: r["output"]["tool_calls"][0]["function"].update(arguments=json.dumps({"query": "cats"})),
    lambda r: r["output"]["tool_calls"][0]["function"].update(name="search_notes"),
    lambda r: r["output"]["tool_calls"][0]["function"].update(arguments="{not json"),
    lambda r: r.update(reasoning="Another thought."),
    lambda r: r.update(status="error", output=None),
], ids=["text", "arguments", "name", "unparsed", "reasoning", "status"])
def test_the_output_hash_changes_with_what_the_model_answered(change):
    changed = json.loads(json.dumps(ANSWERED))
    change(changed)
    assert output_hash(changed) != output_hash(ANSWERED)

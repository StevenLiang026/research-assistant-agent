"""scripts/decide_m7.py offline, on fixture sets standing in for the held-out sets (data/eval/dev-ext.jsonl
and data/eval/v2/dev-v2-ext.jsonl in a temporary folder) with the runner records of the final system's
candidates (a greedy run and three sample runs per set) and their manifests, all made here: the two decisions
of revision 2 (the vote kept unless the vote with the rules falls below the greedy answers with the rules; the
rules kept unless they break more greedy answers than they fix; both under the normalized scorer), the totals
beside them under both scorers, the aggregate-only output (no item id in stdout, stderr or
results/decision-m7.json, in an error either), and the refusal to score the held-out sets a second time.
Nothing here reads a real run."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from support.llama_fakes import install_fake_family

from research_agent.agent.prompts import PROMPT_HASH, system_prompt
from research_agent.agent.vote import voted_settings
from research_agent.eval.lock import read_jsonl, sha256_of, write_jsonl, write_lock
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


decide_m7 = load_script("decide_m7")

TODAY = "2026-10-10"
SYSTEM = {"role": "system", "content": system_prompt(TODAY)}
ARGS = {"query": "hallucination", "max_results": 5}
SEARCH = {"name": "search_arxiv", "arguments": ARGS, "arguments_json": json.dumps(ARGS)}
FIND = {"content": "", "tool_calls": [SEARCH]}
BYE = {"content": "See you!", "tool_calls": []}
ADAPTER = "runs/train/ft3/gguf/checkpoint-1635-f16.gguf"
BASE, TOKENIZER = "models/Base-Q8_0.gguf", "models/Base"
EXT, EXT_V2 = "data/eval/dev-ext.jsonl", "data/eval/v2/dev-v2-ext.jsonl"  # the fixtures standing in for them
CONFIG, OUT = "results/test-config-m7.json", "results/decision-m7.json"
SAMPLING = Decoding(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
K = 3


def user(text: str) -> list[dict]:
    return [SYSTEM, {"role": "user", "content": text}]


def item(id_: str, messages: list[dict], expected: dict, type_: str) -> dict:
    return {"id": id_, "messages": messages, "expected": expected, "type": type_, "lang": "en",
            "family": "find/en", "split": "dev", "slots": {"topic": "hallucination"},
            "source": {"request": id_.rsplit("-s", 1)[0], "step": 1}}


def call(arguments: dict = ARGS, id_: str = "call_0") -> dict:
    return {"id": id_, "type": "function",
            "function": {"name": "search_arxiv", "arguments": json.dumps(arguments)}}


FAILED_SEARCH = [*user("Find hallucination papers."),
                 {"role": "assistant", "content": "", "tool_calls": [call(id_="c1")]},
                 {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"error": "RATE_LIMITED"})}]
RIGHT, CATS = ("", (call(),)), ("", (call({"query": "cats", "max_results": 5}),))
BUSY, GOODBYE = ("The service is busy, try again later.", ()), ("See you!", ())
ITEMS = {EXT: [item("req-ext-0001-s1", user("Find hallucination papers."), FIND, "call"),
               item("req-ext-0002-s1", FAILED_SEARCH, FIND, "recover"),
               item("req-ext-0003-s1", user("Thanks, that's all."), BYE, "no_call")],
         EXT_V2: [item("req-ext-0401-s1", user("Search hallucination papers."), FIND, "call"),
                  item("req-ext-0402-s1", user("That's all for now, thanks."), BYE, "no_call")]}
# the greedy candidate's answer and samples 0 to 2's: greedy 3 of 5 right, with the rules 4 (one fix, no
# break); the vote 3, with the rules 4. Both kept.
ANSWERS = {"req-ext-0001-s1": [RIGHT] * 4, "req-ext-0002-s1": [BUSY, BUSY, RIGHT, BUSY],
           "req-ext-0003-s1": [GOODBYE] * 4, "req-ext-0401-s1": [CATS, RIGHT, RIGHT, CATS],
           "req-ext-0402-s1": [GOODBYE, RIGHT, RIGHT, GOODBYE]}
# the vote against the greedy answer on req-ext-0401-s1: greedy 4, with the rules 5; the vote with the rules 3
VOTE_HURTS = {**ANSWERS, "req-ext-0401-s1": [RIGHT, CATS, CATS, RIGHT]}
# two goodbyes after a failed search, which the rules turn into retries: one fix, two breaks
BROKEN = [item(f"req-ext-000{k}-s1", FAILED_SEARCH, BYE, "no_call") for k in (5, 6)]
RUNS = {path: [f"runs/m7/heldout/{Path(path).stem}-{name}.jsonl" for name in ("greedy", "s0", "s1", "s2")]
        for path in ITEMS}


def record(row: dict, answer: tuple[str, tuple], candidate: int) -> dict:
    content, calls = answer
    output = {"role": "assistant", "content": content, **({"tool_calls": list(calls)} if calls else {})}
    return {"id": row["id"], "status": "ok",
            "item": {k: v for k, v in row.items() if k not in ("id", "messages")},
            "output": output, "usage": {"prompt_tokens": 900, "completion_tokens": 20}, "paid": False,
            "cached": False, "model": "fake-hybrid:lora=1", "reasoning": f"Candidate {candidate}."}


def manifest_path(run: str) -> Path:
    return Path(run).with_name(Path(run).stem + ".manifest.json")


def configuration(**final_changes) -> dict:
    final = {"votes": K, "vote_mode": "final", "rules": True, "glossary": None,
             "frozen": {"path": "results/final-m7.json", "sha256": "<written by scripts/freeze_final.py>"},
             **final_changes}
    final = {k: v for k, v in final.items() if v is not None or k == "glossary"}
    return {"milestone": "M7", "purpose": "M7: the final local system (a test)", "test_sets": "v2",
            "scorers": ["normalized"],
            "systems": {"final": final, "finalg": {"decoding": "greedy", "end_to_end": False}},
            "adapters": {"final": {"gguf": ADAPTER, "sha256": sha256_of(ADAPTER), "selection": None,
                                   "model": BASE, "tokenizer": TOKENIZER,
                                   "generation": {"mode": "think", "decoding": "sampled",
                                                  "reasoning_budget": 64, "base_seed": 0}}},
            "server": {"model": "models/Qwen3-1.7B-Q8_0.gguf", "port": 8080, "ctx_size": 8192,
                       "timeout_s": 30, "llama_cpp_build": None},
            "tokenizer": "models/Qwen3-1.7B", "b2": {"max_cost_rmb": 5.0}, "e2e": {"seed": 42, "today": None},
            "statistics": {"seed": 42, "resamples": 100},
            "comparisons": [{"name": "final-vs-finalg", "a": "final", "b": "finalg", "scorer": "normalized",
                             "descriptive": True}],
            "gpu": {"idle_max_mib": 1500, "sample_ms": 200}}


def write_config(config: dict) -> None:
    Path(CONFIG).parent.mkdir(parents=True, exist_ok=True)
    Path(CONFIG).write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n")


def held_out(answers: dict, extra: list[dict] = (), handwritten: frozenset[str] = frozenset()) -> None:
    """The fixture sets (extra items appended to the first; the items named in handwritten marked as written
    by hand, as the handwritten items' source says), the candidates' runs and their manifests."""
    family = load_family(TOKENIZER)
    settings = GenerationSettings.for_family(family, mode="think", reasoning_budget=64, base_seed=0)
    greedy, samples = voted_settings(with_votes(settings, K))
    for path, rows in ITEMS.items():
        rows = [*rows, *(extra if path == EXT else [])]
        rows = [{**row, "source": {**row["source"], "handwritten": True}} if row["id"] in handwritten else row
                for row in rows]
        write_jsonl(rows, path)
        write_lock(path)
        for candidate, (run, generation) in enumerate(zip(RUNS[path], (greedy, *samples), strict=True)):
            write_jsonl([record(row, answers.get(row["id"], [GOODBYE] * 4)[candidate], candidate)
                         for row in rows], run)
            n = len(rows)
            manifest = {"adapters": [{"id": 0, "path": ADAPTER, "sha256": sha256_of(ADAPTER)}],
                        "family": family.record(), "generation": generation.record(), "items": path,
                        "items_run": n, "items_sha256": sha256_of(path), "items_total": n, "limit": None,
                        "lora_scale": 1.0, "prompt_hash": PROMPT_HASH, "schema_hash": SCHEMA_HASH,
                        "server_props": {"model_path": BASE}, "statuses": {"ok": n}, "system": "ft"}
            manifest_path(run).write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(decide_m7, "OUT", tmp_path / OUT)  # the one place of the decision, here
    install_fake_family(monkeypatch, tmp_path / TOKENIZER,
                        decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY})
    for path, data in ((ADAPTER, b"GGUF ft3 step 1635"), (BASE, b"GGUF base")):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(data)
    write_config(configuration())
    return tmp_path


def arguments(runs: dict[str, list[str]] = RUNS) -> list[str]:
    return ["--config", CONFIG, *(arg for path, paths in runs.items() for arg in ("--runs", path, *paths))]


def ids() -> list[str]:
    return [row["id"] for rows in ITEMS.values() for row in rows] + [row["id"] for row in BROKEN]


def assert_aggregate_only(*texts: str) -> None:
    for text in texts:
        assert not [i for i in ids() if i in text]


def test_the_vote_and_the_rules_are_kept_and_only_aggregates_are_printed_and_written(repo, capsys):
    held_out(ANSWERS)
    assert decide_m7.main(arguments()) == 0
    captured = capsys.readouterr()
    written = Path(OUT).read_text(encoding="utf-8")
    assert_aggregate_only(captured.out, captured.err, written)
    decision = json.loads(written)
    counts = {"greedy": 3, "greedy_rules": 4, "vote": 3, "vote_rules": 4, "rules_fixes": 1, "rules_breaks": 0}
    assert decision["totals"] == {"items": 5, "default": counts, "normalized": counts}
    assert decision["sets"] == [{"path": path, "sha256": sha256_of(path), "n": len(rows)}
                                for path, rows in ITEMS.items()]
    assert decision["runs"] == [{"set": path, "greedy": {"path": runs[0], "sha256": sha256_of(runs[0])},
                                 "samples": [{"path": run, "sha256": sha256_of(run)} for run in runs[1:]]}
                                for path, runs in RUNS.items()]
    assert decision["config"] == {"path": CONFIG, "sha256": sha256_of(CONFIG)}
    assert (decision["system"]["votes"], decision["system"]["vote_mode"], decision["scorer"]) == (
        K, "final", "normalized")
    vote, rules = decision["decisions"]["vote"], decision["decisions"]["rules"]
    assert (vote["kept"], vote["vote_rules"], vote["greedy_rules"]) == (True, 4, 4)
    assert (rules["kept"], rules["fixes"], rules["breaks"]) == (True, 1, 0)
    assert "step 1, the vote: kept" in captured.out and "step 2, the rules: kept" in captured.out
    assert written.endswith("}\n")


def test_the_vote_is_dropped_when_with_the_rules_it_falls_below_the_greedy_answers_with_them(repo, capsys):
    held_out(VOTE_HURTS)
    assert decide_m7.main(arguments()) == 0
    decision = json.loads(Path(OUT).read_text(encoding="utf-8"))
    assert decision["totals"]["normalized"] == {"greedy": 4, "greedy_rules": 5, "vote": 2, "vote_rules": 3,
                                                "rules_fixes": 1, "rules_breaks": 0}
    assert decision["decisions"]["vote"]["kept"] is False and decision["decisions"]["rules"]["kept"] is True
    captured = capsys.readouterr()
    assert "step 1, the vote: dropped" in captured.out
    assert_aggregate_only(captured.out, captured.err, Path(OUT).read_text(encoding="utf-8"))


def test_the_rules_are_dropped_when_they_break_more_greedy_answers_than_they_fix(repo, capsys):
    held_out(ANSWERS, BROKEN)
    assert decide_m7.main(arguments()) == 0
    decision = json.loads(Path(OUT).read_text(encoding="utf-8"))
    assert decision["totals"]["items"] == 7
    assert decision["totals"]["normalized"] == {"greedy": 5, "greedy_rules": 4, "vote": 5, "vote_rules": 4,
                                                "rules_fixes": 1, "rules_breaks": 2}
    assert decision["decisions"]["rules"]["kept"] is False and decision["decisions"]["vote"]["kept"] is True
    captured = capsys.readouterr()
    assert "step 2, the rules: dropped" in captured.out
    assert_aggregate_only(captured.out, captured.err, Path(OUT).read_text(encoding="utf-8"))


def test_an_existing_decision_is_never_overwritten_the_held_out_sets_are_scored_once(repo, capsys):
    Path(OUT).write_text("{}\n", encoding="utf-8")
    missing = {path: [f"runs/nowhere/{k}.jsonl" for k in range(4)] for path in ITEMS}
    assert decide_m7.main(arguments(missing)) == 2  # refused before any input is read
    captured = capsys.readouterr()
    assert f"{OUT} exists" in captured.out and "scored once" in captured.out
    assert Path(OUT).read_text(encoding="utf-8") == "{}\n"


def test_a_run_without_a_record_of_every_item_is_refused_by_a_count_never_an_id(repo, capsys):
    held_out(ANSWERS)
    run = RUNS[EXT][2]
    write_jsonl(read_jsonl(run)[1:], run)
    assert decide_m7.main(arguments()) == 2
    captured = capsys.readouterr()
    assert f"{run} holds no record of 1 of the 3 items" in captured.out and not Path(OUT).exists()
    assert_aggregate_only(captured.out, captured.err)


def test_a_manifest_that_disagrees_with_the_configuration_is_refused_without_an_id(repo, capsys):
    held_out(ANSWERS)
    target = manifest_path(RUNS[EXT_V2][2])
    data = json.loads(target.read_text(encoding="utf-8"))
    data["generation"] = {**data["generation"], "base_seed": 2}  # sample 1 made with seed 2
    target.write_text(json.dumps(data), encoding="utf-8")
    assert decide_m7.main(arguments()) == 2
    captured = capsys.readouterr()
    assert f"{target.as_posix()}: generation " in captured.out and not Path(OUT).exists()
    assert_aggregate_only(captured.out, captured.err)


def test_the_final_configuration_is_counted_by_subset_for_the_prediction_without_an_id(repo, capsys):
    # revision 2's prediction weighs the held-out drafted and handwritten items apart: the configuration the
    # two decisions choose (here the vote with the rules) is counted per subset, aggregates only
    held_out(ANSWERS, handwritten=frozenset({"req-ext-0003-s1"}))
    assert decide_m7.main(arguments()) == 0
    captured = capsys.readouterr()
    written = Path(OUT).read_text(encoding="utf-8")
    assert_aggregate_only(captured.out, captured.err, written)
    final = json.loads(written)["final"]
    per_subset = {"drafted": 3, "handwritten": 1}  # 0402 is the one the vote gets wrong, and it is drafted
    assert final == {"vote": True, "rules": True, "records": "vote_rules",
                     "n": {"drafted": 4, "handwritten": 1},
                     "right": {"default": per_subset, "normalized": per_subset}}
    assert "the final system (vote + rules): drafted 3/4, handwritten 1/1 (normalized)" in captured.out


def test_the_subsets_are_counted_on_the_records_the_decisions_choose(repo, capsys):
    held_out(VOTE_HURTS, handwritten=frozenset({"req-ext-0401-s1"}))
    assert decide_m7.main(arguments()) == 0
    decision = json.loads(Path(OUT).read_text(encoding="utf-8"))
    final = decision["final"]
    assert (final["vote"], final["rules"], final["records"]) == (False, True, "greedy_rules")
    assert final["n"] == {"drafted": 4, "handwritten": 1}
    assert final["right"]["normalized"] == {"drafted": 4, "handwritten": 1}
    assert sum(final["right"]["normalized"].values()) == decision["totals"]["normalized"]["greedy_rules"]


def test_the_decision_has_one_place_so_no_second_run_scores_the_held_out_sets(repo, capsys, monkeypatch):
    held_out(ANSWERS)
    assert decide_m7.main(arguments()) == 0
    capsys.readouterr()
    elsewhere = Path("elsewhere")
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # another working folder: the decision's place does not move with it
    assert decide_m7.main(["--config", f"../{CONFIG}"]) == 2
    assert "scored once" in capsys.readouterr().out
    with pytest.raises(SystemExit):  # no other place can be asked for
        decide_m7.main([*arguments(), "--out", "results/decision-m7-again.json"])
    assert not Path("results/decision-m7-again.json").exists()


def test_a_record_whose_id_is_no_string_is_refused_as_malformed_by_a_count(repo, capsys):
    held_out(ANSWERS)
    run = RUNS[EXT][1]
    rows = read_jsonl(run)
    rows[0]["id"] = [rows[0]["id"]]
    write_jsonl(rows, run)
    assert decide_m7.main(arguments()) == 1
    captured = capsys.readouterr()
    assert f"{run}: 1 record(s) whose id is not a string" in captured.out and not Path(OUT).exists()
    assert_aggregate_only(captured.out, captured.err)


def test_the_runs_must_be_those_of_the_two_held_out_sets_and_the_vote_must_be_configured(repo, capsys):
    held_out(ANSWERS)
    assert decide_m7.main(arguments({EXT: RUNS[EXT]})) == 2
    assert f"no runs of {EXT_V2}" in capsys.readouterr().out
    assert decide_m7.main(arguments({"data/eval/dev.jsonl": RUNS[EXT], EXT_V2: RUNS[EXT_V2]})) == 2
    assert "data/eval/dev.jsonl is not one of the held-out sets" in capsys.readouterr().out
    write_config(configuration(votes=None, vote_mode=None))
    assert decide_m7.main(arguments()) == 2
    assert "systems.final has no votes" in capsys.readouterr().out
    assert not Path(OUT).exists()

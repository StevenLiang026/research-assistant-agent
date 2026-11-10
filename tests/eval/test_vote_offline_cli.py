"""scripts/vote_offline.py (docs/m7-本地提升计划.md WP2, WP3): the vote and the runtime rules over saved runs,
as the voted and ruled systems give them online. The vote over a greedy run and K seed runs, in either key
mode, with or without the rules; a failed candidate's record, which the vote goes on without and lists under
its errors; the rules alone over the greedy run; the totals, and --aggregate-only, which prints them alone,
names no item in any message and writes nothing (for the held-out sets, which only it may read, found by
their name, their path or their content); the guards; and the end to end check: run_eval --votes --rules
writes byte for byte what vote_offline writes from run_eval's greedy and seed runs over the same items, a
candidate that raises included.

No test reads the real held-out sets: every test's are fixture files under its tmp_path (held_out)."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from support.llama_fakes import FakeLlama, Scripted, install_fake_family

from research_agent.agent.prompts import system_prompt
from research_agent.agent.rules import RULES, RULES_VERSION, ruled_record
from research_agent.agent.vote import FINAL, REPRODUCTION, voted_record
from research_agent.eval.lock import sha256_of, write_lock
from research_agent.family.core import GREEDY, Decoding, Mode
from research_agent.family.toolformats.qwen_json import QwenJson

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vote_offline = load_script("vote_offline")
run_eval = load_script("run_eval")


@pytest.fixture(autouse=True)
def held_out(tmp_path, monkeypatch) -> tuple[Path, Path]:
    """The roots every test's held-out sets are found under (vote_offline.HELD_OUT_ROOTS, set here to two
    folders standing for the repository and a run copy named by --run-copy): folders under tmp_path, so
    that no test stats, reads or hashes the real held-out sets."""
    roots = (tmp_path / "held-out" / "repo", tmp_path / "held-out" / "raa-run")
    monkeypatch.setattr(vote_offline, "HELD_OUT_ROOTS", roots)
    return roots


@pytest.fixture
def hashed(monkeypatch) -> list[Path]:
    """Every file vote_offline hashes itself (its sha256_of), in order."""
    seen: list[Path] = []
    real = vote_offline.sha256_of

    def spy(path):
        seen.append(Path(path))
        return real(path)

    monkeypatch.setattr(vote_offline, "sha256_of", spy)
    return seen


TODAY = "2026-10-10"
SYSTEM = {"role": "system", "content": system_prompt(TODAY)}
FAILED = {"id": "call_0", "type": "function",
          "function": {"name": "search_arxiv", "arguments": "{\"query\": \"hallucination\"}"}}


def item(rid: str, messages: list[dict], expected_calls: list[dict], type_: str) -> dict:
    return {"id": rid, "messages": [SYSTEM, *messages],
            "expected": {"content": "", "tool_calls": expected_calls},
            "type": type_, "lang": "en", "family": f"f/{type_}", "split": "dev", "slots": {},
            "source": {"request": rid, "step": 1}}


ITEMS = [item("q-1", [{"role": "user", "content": "Find mixture-of-experts papers."}],
              [{"name": "search_arxiv", "arguments": {"query": "mixture of experts"}}], "call"),
         item("q-2", [{"role": "user", "content": "Find hallucination papers."},
                      {"role": "assistant", "content": "", "tool_calls": [FAILED]},
                      {"role": "tool", "tool_call_id": "call_0", "content": "{\"error\":\"RATE_LIMITED\"}"}],
              [{"name": "search_arxiv", "arguments": {"query": "hallucination"}}], "recover")]


def search(query: str) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "local_0", "type": "function",
         "function": {"name": "search_arxiv", "arguments": json.dumps({"query": query})}}]}


def record(it: dict, output: dict, prompt: int = 100) -> dict:
    """A runner record of an ok turn for item it."""
    extra = {k: v for k, v in it.items() if k not in ("id", "messages")}
    return {"id": it["id"], "status": "ok", "item": extra, "output": output,
            "usage": {"prompt_tokens": prompt, "completion_tokens": 5}, "paid": False, "cached": False,
            "model": "m:lora=1"}


# what each run answered: the greedy run, then seeds 0, 1 and 2
ANSWERS = {"q-1": ["moe", "mixture-of-experts", "mixture of experts", "moe"],
           "q-2": ["LLM hallucination", "hallucination", "hallucination survey", "LLM hallucination"]}


def write_runs(folder: Path) -> tuple[Path, list[Path]]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for run in range(4):
        path = folder / ("greedy.jsonl" if run == 0 else f"s{run - 1}.jsonl")
        path.write_text("".join(json.dumps(record(it, search(ANSWERS[it["id"]][run]), 100 + run)) + "\n"
                                for it in ITEMS), encoding="utf-8")
        paths.append(path)
    return paths[0], paths[1:]


def write_items(folder: Path, items: list[dict] = ITEMS, name: str = "items.jsonl") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    return path


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def run_args(items: Path, greedy: Path, samples: list[Path], *extra: str) -> list[str]:
    return [str(items), "--greedy", str(greedy), *[a for s in samples for a in ("--sample", str(s))], *extra]


@pytest.mark.parametrize("mode", [FINAL, REPRODUCTION])
@pytest.mark.parametrize("rules", [False, True])
def test_the_voted_records_are_the_vote_over_the_runs_records(mode, rules, tmp_path, capsys):
    items = write_items(tmp_path)
    greedy, samples = write_runs(tmp_path / "runs")
    out = tmp_path / "voted" / "dev-votes3.jsonl"
    code = vote_offline.main(run_args(items, greedy, samples, "--vote-mode", mode, "--out", str(out),
                                      *(["--rules"] if rules else [])))
    assert code == 0
    got = lines(out)
    runs = [lines(greedy), *(lines(s) for s in samples)]
    for position, it in enumerate(ITEMS):
        candidates = [run[position] for run in runs]
        if rules:
            candidates = [ruled_record(r, it["messages"]) for r in candidates]
        assert got[position] == voted_record(candidates[0], candidates[1:], it["messages"], mode)
    manifest = json.loads(out.with_name("dev-votes3.manifest.json").read_text(encoding="utf-8"))
    assert manifest["items_sha256"] == sha256_of(items) and manifest["votes"] == 3
    assert manifest["vote_mode"] == mode and manifest["greedy"]["sha256"] == sha256_of(greedy)
    assert [s["sha256"] for s in manifest["samples"]] == [sha256_of(s) for s in samples]
    assert (manifest.get("rules") == {"version": RULES_VERSION, "names": list(RULES)}) is rules
    assert manifest["totals"]["items"] == 2 and "q-1" not in json.dumps(manifest)
    printed = capsys.readouterr().out
    assert "voted" in printed and "greedy" in printed


def test_the_final_vote_and_the_rules_give_the_numbers_the_totals_show(tmp_path, capsys):
    items = write_items(tmp_path)
    greedy, samples = write_runs(tmp_path / "runs")
    out = tmp_path / "voted.jsonl"
    assert vote_offline.main(run_args(items, greedy, samples, "--rules", "--out", str(out))) == 0
    q1, q2 = lines(out)
    # q-1: "mixture-of-experts" and "mixture of experts" are one class under the final key (S2), which wins 2
    # to 1 with seed 0 its representative: right under the normalized scorer only (its S2), where the greedy
    # "moe" is wrong under both; the class's two members are right once and twice
    assert q1["votes"]["representative"] == 0 and q1["output"] == search("mixture-of-experts")
    # q-2: the rules rewrite the greedy answer and seeds 1 and 2 into the retry, which seed 0 already is: a
    # unanimous class, seed 0 its representative with nothing rewritten, right under both scorers
    assert q2["votes"]["counts"] == {q2["votes"]["chosen"]: 3} and q2["votes"]["representative"] == 0
    assert q2["rules"] == [] and q2["output"] == search("hallucination")
    totals = json.loads(out.with_name("voted.manifest.json").read_text(encoding="utf-8"))["totals"]
    assert totals["voted"] == {"default": 1, "normalized": 2}
    assert totals["greedy"] == {"default": 1, "normalized": 1}  # the rules fix q-2
    assert totals["class_mean"] == {"default": 1.5, "normalized": 2.0}
    assert totals["votes"]["no_sample_voted"] == 0
    assert totals["rules"] == {"transient_retry": 3, "id_repair": 0}


def test_the_rules_alone_rewrite_the_greedy_run(tmp_path):
    items = write_items(tmp_path)
    greedy, _ = write_runs(tmp_path / "runs")
    out = tmp_path / "ruled.jsonl"
    assert vote_offline.main([str(items), "--greedy", str(greedy), "--rules", "--out", str(out)]) == 0
    assert lines(out) == [ruled_record(r, it["messages"]) for r, it in zip(lines(greedy), ITEMS, strict=True)]
    manifest = json.loads(out.with_name("ruled.manifest.json").read_text(encoding="utf-8"))
    assert manifest["votes"] == 0 and "vote_mode" not in manifest


def error_record(it: dict, message: str = "parse_tool_calls failed: ValueError('x')",
                 error_type: str = "HarnessError") -> dict:
    """The runner's record of a model that raised on item it (eval.runner, error_fields)."""
    extra = {k: v for k, v in it.items() if k not in ("id", "messages")}
    return {"id": it["id"], "status": "error", "item": extra, "error": message, "error_type": error_type}


def replace_record(path: Path, record: dict) -> None:
    """The run at path with its record of record's item replaced by record."""
    runs = [record if r["id"] == record["id"] else r for r in lines(path)]
    path.write_text("".join(json.dumps(r) + "\n" for r in runs), encoding="utf-8")


@pytest.mark.parametrize("mode", [FINAL, REPRODUCTION])
def test_the_vote_goes_on_without_a_failed_candidates_record_and_lists_it_under_its_errors(mode, tmp_path,
                                                                                           capsys):
    items = write_items(tmp_path)
    greedy, samples = write_runs(tmp_path / "runs")
    # seed 1's run failed on q-1; on q-2 every sample's did, with three kinds of error
    replace_record(samples[1], error_record(ITEMS[0]))
    for k, (message, kind) in enumerate([("read timed out", "ReadTimeout"), ("'choices'", "KeyError"),
                                         ("line 1\nline 2", "HarnessError")]):
        replace_record(samples[k], error_record(ITEMS[1], message, kind))
    out = tmp_path / "voted.jsonl"
    assert vote_offline.main(run_args(items, greedy, samples, "--vote-mode", mode, "--out", str(out))) == 0
    q1, q2 = lines(out)
    assert q1["votes"]["errors"] == [{"candidate": 1, "error_type": "HarnessError",
                                      "message": "parse_tool_calls failed: ValueError('x')"}]
    # q-1: seeds 0 and 2 answered; one against one, a tie the greedy answer's class ("moe", seed 2's) wins
    assert q1["status"] == "ok" and q1["votes"]["representative"] == 2 and q1["output"] == search("moe")
    assert [e["candidate"] for e in q2["votes"]["errors"]] == [0, 1, 2]
    assert [e["message"] for e in q2["votes"]["errors"]] == ["read timed out", "'choices'", "line 1\nline 2"]
    if mode == FINAL:  # no sample voted: the greedy answer stands
        assert q2["votes"]["chosen"] is None and q2["output"] == search("LLM hallucination")
    else:  # the failed samples are a class, which wins: the record is seed 0's error record, with the vote
        assert q2["votes"]["chosen"] == "STATUS:error" and q2["status"] == "error"
        assert q2["error"] == "read timed out"
    totals = json.loads(out.with_name("voted.manifest.json").read_text(encoding="utf-8"))["totals"]
    assert (totals["votes"]["with_errors"], totals["votes"]["failed_candidates"]) == (2, 4)
    assert totals["votes"]["no_sample_voted"] == (1 if mode == FINAL else 0)
    printed = capsys.readouterr().out
    assert "with_errors 2" in printed and "failed_candidates 4" in printed
    # no candidate failed: no errors in any vote, and the tallies say so
    clean = tmp_path / "clean"
    greedy, samples = write_runs(clean / "runs")
    assert vote_offline.main(run_args(items, greedy, samples, "--vote-mode", mode, "--out",
                                      str(clean / "voted.jsonl"))) == 0
    assert not any("errors" in r["votes"] for r in lines(clean / "voted.jsonl"))
    totals = json.loads((clean / "voted.manifest.json").read_text(encoding="utf-8"))["totals"]
    assert (totals["votes"]["with_errors"], totals["votes"]["failed_candidates"]) == (0, 0)
    capsys.readouterr()


def test_aggregate_only_prints_the_totals_alone_and_writes_nothing(tmp_path, capsys):
    items = write_items(tmp_path)
    greedy, samples = write_runs(tmp_path / "runs")
    before = sorted(p.name for p in tmp_path.rglob("*"))
    code = vote_offline.main(run_args(items, greedy, samples, "--rules", "--aggregate-only"))
    printed = capsys.readouterr().out
    assert code == 0 and sorted(p.name for p in tmp_path.rglob("*")) == before
    assert "voted" in printed and "q-1" not in printed and "q-2" not in printed
    assert "mixture" not in printed and "hallucination" not in printed  # no answer, no key, no id
    totals = json.loads(printed.splitlines()[-1])
    assert totals["voted"] == {"default": 1, "normalized": 2} and totals["items"] == 2


def write_held_out(root: Path, which: int, items: list[dict] = ITEMS) -> Path:
    """A stand-in for held-out set which (vote_offline.HELD_OUT: 0 dev-ext, 1 dev-v2-ext) under root, one of
    the fixture's roots."""
    path = root / vote_offline.HELD_OUT[which]
    return write_items(path.parent, items, name=path.name)


def refused_as_held_out(items: Path, greedy: Path, samples: list[Path], out: Path, capsys,
                        *extra: str) -> str:
    """The message of a run over items that is refused as a held-out set (exit 4, nothing written)."""
    code = vote_offline.main(run_args(items, greedy, samples, "--out", str(out), *extra))
    err = capsys.readouterr().err
    assert code == 4 and "--aggregate-only" in err and not out.exists()
    return err


def test_a_held_out_set_is_read_only_for_its_totals(tmp_path, capsys):
    greedy, samples = write_runs(tmp_path / "runs")
    for name in ("dev-ext.jsonl", "dev-v2-ext.jsonl"):
        held_out = write_items(tmp_path / "eval", name=name)
        err = refused_as_held_out(held_out, greedy, samples, tmp_path / "x.jsonl", capsys)
        assert "(by its name)" in err
        assert vote_offline.main(run_args(held_out, greedy, samples, "--aggregate-only")) == 0
        capsys.readouterr()


@pytest.mark.parametrize("root", [0, 1])  # the repository's, the run copy's
@pytest.mark.parametrize("which", [0, 1])
def test_a_held_out_set_is_found_by_its_resolved_path_under_the_repository_or_its_run_copy(
        root, which, held_out, tmp_path, monkeypatch, capsys):
    greedy, samples = write_runs(tmp_path / "runs")
    path = write_held_out(held_out[root], which)
    monkeypatch.chdir(path.parent)
    spelled = Path("..") / path.parent.name / path.name  # relative, through ".."
    err = refused_as_held_out(spelled, greedy, samples, tmp_path / "x.jsonl", capsys)
    assert "(by its path)" in err
    assert vote_offline.main(run_args(spelled, greedy, samples, "--aggregate-only")) == 0


def test_a_link_to_a_held_out_set_under_another_name_is_found_by_its_path(held_out, tmp_path, capsys):
    greedy, samples = write_runs(tmp_path / "runs")
    path = write_held_out(held_out[1], 0)
    linked = tmp_path / "elsewhere" / "linked.jsonl"
    linked.parent.mkdir()
    try:
        os.link(path, linked)
    except OSError as exc:  # a file system without hard links
        pytest.skip(f"no hard link here: {exc}")
    err = refused_as_held_out(linked, greedy, samples, tmp_path / "x.jsonl", capsys)
    assert "(by its path)" in err


@pytest.mark.parametrize("root", [0, 1])
@pytest.mark.parametrize("which", [0, 1])
def test_a_copy_of_a_held_out_set_under_another_name_is_found_by_its_content(root, which, held_out, tmp_path,
                                                                             capsys, hashed):
    greedy, samples = write_runs(tmp_path / "runs")
    path = write_held_out(held_out[root], which)
    copy = tmp_path / "elsewhere" / "renamed.jsonl"
    copy.parent.mkdir()
    copy.write_bytes(path.read_bytes())
    err = refused_as_held_out(copy, greedy, samples, tmp_path / "x.jsonl", capsys)
    assert "(by its content)" in err and path in hashed
    # nothing about the held-out file is printed: not its path, not its SHA-256
    assert "held-out/" not in err.replace("\\", "/") and sha256_of(path) not in err
    assert vote_offline.main(run_args(copy, greedy, samples, "--aggregate-only")) == 0


def test_without_a_named_run_copy_the_held_out_sets_are_looked_for_under_the_repository_alone():
    fresh = load_script("vote_offline")  # this module's roots are the fixture's; a fresh load has its own
    assert fresh.HELD_OUT_ROOTS == (fresh.REPO,)
    assert fresh.held_out_roots() == (fresh.REPO,)


def test_the_run_copy_is_a_root_only_when_the_option_or_the_environment_names_it(held_out, tmp_path,
                                                                                 monkeypatch, capsys):
    monkeypatch.setattr(vote_offline, "HELD_OUT_ROOTS", held_out[:1])
    copy = tmp_path / "run-copy"
    path = write_held_out(copy, 0)  # a held-out set under the run copy alone
    renamed = tmp_path / "elsewhere" / "renamed.jsonl"
    renamed.parent.mkdir()
    renamed.write_bytes(path.read_bytes())
    greedy, samples = write_runs(tmp_path / "runs")
    assert vote_offline.held_out_roots() == held_out[:1]
    assert vote_offline.main(run_args(renamed, greedy, samples, "--out", str(tmp_path / "a.jsonl"))) == 0
    capsys.readouterr()
    err = refused_as_held_out(renamed, greedy, samples, tmp_path / "b.jsonl", capsys, "--run-copy", str(copy))
    assert "(by its content)" in err
    monkeypatch.setenv(vote_offline.RUN_COPY_ENV, str(copy))
    assert vote_offline.held_out_roots() == (*held_out[:1], copy)
    err = refused_as_held_out(renamed, greedy, samples, tmp_path / "c.jsonl", capsys)
    assert "(by its content)" in err


def test_a_held_out_file_is_read_only_when_its_size_is_the_items_and_other_bytes_are_no_held_out_set(
        held_out, tmp_path, capsys, hashed):
    greedy, samples = write_runs(tmp_path / "runs")
    items = write_items(tmp_path)
    same_size = write_held_out(held_out[0], 0, [ITEMS[1], ITEMS[0]])  # the same lines in another order
    other_size = write_held_out(held_out[1], 1, ITEMS[:1])
    assert same_size.stat().st_size == items.stat().st_size != other_size.stat().st_size
    assert vote_offline.main(run_args(items, greedy, samples, "--out", str(tmp_path / "x.jsonl"))) == 0
    # a file of another size cannot have the items' SHA-256: it is never read
    assert same_size in hashed and other_size not in hashed
    hashed.clear()
    # with --aggregate-only no held-out set is looked at at all, and nothing is hashed
    assert vote_offline.main(run_args(items, greedy, samples, "--aggregate-only")) == 0
    assert hashed == []
    capsys.readouterr()


def test_test_material_is_refused_and_a_locked_set_is_lock_checked(tmp_path, capsys):
    greedy, samples = write_runs(tmp_path / "runs")
    test_items = write_items(tmp_path / "t", [{**ITEMS[0], "split": "test"}, ITEMS[1]])
    assert vote_offline.main(run_args(test_items, greedy, samples, "--aggregate-only")) == 4
    err = capsys.readouterr().err
    assert "test material" in err and "1 item(s)" in err and "q-1" not in err  # a count, no item
    assert vote_offline.main(run_args(test_items, greedy, samples, "--out", str(tmp_path / "x.jsonl"))) == 4
    assert "q-1" in capsys.readouterr().err  # without --aggregate-only the first item is named
    locked = write_items(tmp_path / "locked")
    write_lock(locked)
    locked.write_bytes(locked.read_bytes() + b"\n")
    assert vote_offline.main(run_args(locked, greedy, samples, "--aggregate-only")) == 5
    capsys.readouterr()


def test_items_without_distinct_ids_are_malformed_and_aggregate_only_counts_them(tmp_path, capsys):
    greedy, samples = write_runs(tmp_path / "runs")
    repeated = write_items(tmp_path / "repeated", [ITEMS[0], ITEMS[1], ITEMS[1], {**ITEMS[0], "id": ""}])
    assert vote_offline.main(run_args(repeated, greedy, samples, "--aggregate-only")) == 1
    err = capsys.readouterr().err
    assert "2 item(s)" in err and "q-2" not in err and "q-1" not in err
    assert vote_offline.main(run_args(repeated, greedy, samples, "--out", str(tmp_path / "x.jsonl"))) == 1
    assert "'q-2'" in capsys.readouterr().err


def test_each_runs_record_of_an_item_is_found_by_its_id_and_a_missing_or_repeated_one_is_malformed(tmp_path,
                                                                                                    capsys):
    items = write_items(tmp_path)
    greedy, samples = write_runs(tmp_path / "runs")
    out = tmp_path / "in-order.jsonl"
    assert vote_offline.main(run_args(items, greedy, samples, "--out", str(out))) == 0
    # a run in another order, or one that holds more items than these (a subset of its set), votes alike
    shuffled = samples[1].read_text(encoding="utf-8").splitlines(keepends=True)
    samples[1].write_text("".join(reversed(shuffled)), encoding="utf-8")
    subset = write_items(tmp_path / "subset", ITEMS[1:])
    other = tmp_path / "subset.jsonl"
    assert vote_offline.main(run_args(items, greedy, samples, "--out", str(tmp_path / "shuffled.jsonl"))) == 0
    assert lines(tmp_path / "shuffled.jsonl") == lines(out)
    assert vote_offline.main(run_args(subset, greedy, samples, "--out", str(other))) == 0
    assert lines(other) == lines(out)[1:]
    capsys.readouterr()
    for broken, first in ((shuffled[:1], "q-2"), (shuffled + shuffled[:1], "q-1")):  # no record; two records
        samples[1].write_text("".join(broken), encoding="utf-8")
        assert vote_offline.main(run_args(items, greedy, samples, "--aggregate-only")) == 1
        err = capsys.readouterr().err
        # --aggregate-only names no item: the run and a count alone
        assert "s1.jsonl" in err and "1 of the 2 items" in err and "q-1" not in err and "q-2" not in err
        assert vote_offline.main(run_args(items, greedy, samples, "--out", str(tmp_path / "bad.jsonl"))) == 1
        assert f"(the first: {first})" in capsys.readouterr().err
        assert not (tmp_path / "bad.jsonl").exists()


@pytest.mark.parametrize("rules", [False, True])
@pytest.mark.parametrize("broken", [{"status": "ok"}, {"status": "ok", "output": "text"},
                                    {"status": "ok", "output": {"content": "", "tool_calls": "x"}}],
                         ids=["no-output", "output-not-an-object", "tool-calls-not-a-list"])
def test_a_record_of_the_wrong_shape_is_malformed_and_aggregate_only_counts_it(broken, rules, tmp_path,
                                                                                 capsys):
    # the round-3 review: such a record went on to a traceback that named its item under --aggregate-only
    items = write_items(tmp_path)
    greedy, samples = write_runs(tmp_path / "runs")
    replace_record(samples[0], {"id": "q-1", **broken})
    extra = ("--rules",) if rules else ()
    assert vote_offline.main(run_args(items, greedy, samples, *extra, "--aggregate-only")) == 1
    captured = capsys.readouterr()
    assert "s0.jsonl" in captured.err and "1 record(s) of the wrong shape" in captured.err
    assert "q-1" not in captured.out + captured.err and "q-2" not in captured.out + captured.err
    out = tmp_path / "x.jsonl"
    assert vote_offline.main(run_args(items, greedy, samples, *extra, "--out", str(out))) == 1
    assert "(the first: q-1)" in capsys.readouterr().err and not out.exists()


@pytest.mark.parametrize(("options", "message"), [
    (["--out", "x.jsonl"], "--rules"),  # nothing to vote and no rules: nothing to do
    (["--vote-mode", "final", "--rules", "--out", "x.jsonl"], "--vote-mode"),
    (["--rules"], "--out"),
    (["--rules", "--out", "x.jsonl", "--aggregate-only"], "--aggregate-only")])
def test_usage_errors_exit_2(options, message, tmp_path, capsys):
    items = write_items(tmp_path)
    greedy, _ = write_runs(tmp_path / "runs")
    with pytest.raises(SystemExit) as stopped:
        vote_offline.main([str(items), "--greedy", str(greedy), *options])
    assert stopped.value.code == 2 and message in capsys.readouterr().err


# ---- online equals offline: run_eval --votes --rules against vote_offline over run_eval's runs -------------

SAMPLING = Decoding(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)


class SeededLlama(FakeLlama):
    """A llama-server stand-in whose answer is a function of the request's seed: the greedy answer for a
    request without one, else sampled[seed % len(sampled)]; so a request answers alike in every run."""

    def __init__(self, greedy: Scripted, sampled: list[Scripted]) -> None:
        super().__init__([])
        self.greedy, self.sampled = greedy, sampled

    def handle(self, request):
        if request.url.path == "/v1/completions":
            seed = json.loads(request.content).get("seed")
            self.script = [self.greedy if seed is None else self.sampled[seed % len(self.sampled)]]
        return super().handle(request)


def call_text(query: str) -> str:
    return '<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "' + query + '"}}\n</tool_call>'


def seeded() -> SeededLlama:
    return SeededLlama(Scripted(("G.", "\n</think>\n\n", call_text("moe"))),
                       [Scripted((f"S{n}.", "\n</think>\n\n", call_text(q)))
                        for n, q in enumerate(("mixture-of-experts", "mixture of experts", "moe experts"))])


def test_run_eval_votes_and_rules_write_what_vote_offline_writes_from_the_greedy_and_seed_runs(tmp_path,
                                                                                              monkeypatch):
    monkeypatch.chdir(tmp_path)
    install_fake_family(monkeypatch, tmp_path / "model",
                        decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY})
    write_items(tmp_path)
    common = ["items.jsonl", "--system", "b0", "--tokenizer", "model", "--mode", "think",
              "--reasoning-budget", "16", "--signals"]
    online_server = seeded()
    assert run_eval.main([*common, "--out", "online.jsonl", "--votes", "3", "--rules"],
                         local_client=online_server.client()) == 0
    for name, flags in (("greedy", ["--greedy"]), ("s0", ["--seed", "0"]), ("s1", ["--seed", "1"]),
                        ("s2", ["--seed", "2"])):
        assert run_eval.main([*common, "--out", f"{name}.jsonl", *flags], local_client=seeded().client()) == 0
    assert vote_offline.main(["items.jsonl", "--greedy", "greedy.jsonl", "--sample", "s0.jsonl", "--sample",
                              "s1.jsonl", "--sample", "s2.jsonl", "--rules", "--out", "offline.jsonl"]) == 0
    online, offline = (tmp_path / "online.jsonl").read_bytes(), (tmp_path / "offline.jsonl").read_bytes()
    assert online == offline
    voted = [json.loads(line) for line in online.decode("utf-8").splitlines()]
    assert [r["votes"]["k"] for r in voted] == [3, 3] and voted[1]["rules"][0]["rule"] == "transient_retry"
    assert len(online_server.requests) == 8  # (greedy + 3 samples) x 2 items


class BreakingFormat(QwenJson):
    """Qwen3's tool format, whose parse fails on an answer that says BROKEN: a failure LocalModel raises as a
    HarnessError (agent.models), which a vote's candidate raises online and a run's record keeps offline."""

    def parse(self, text: str, tools: list[dict] | None) -> list[dict]:
        if "BROKEN" in text:
            raise ValueError("a broken block")
        return super().parse(text, tools)


def breaking() -> SeededLlama:
    """seeded(), its second sampled answer one the family's parse fails on."""
    return SeededLlama(Scripted(("G.", "\n</think>\n\n", call_text("moe"))),
                       [Scripted(("S0.", "\n</think>\n\n", call_text("mixture-of-experts"))),
                        Scripted(("S1.", "\n</think>\n\n", "BROKEN")),
                        Scripted(("S2.", "\n</think>\n\n", call_text("moe experts")))])


def test_a_candidate_that_raises_is_listed_online_from_its_exception_as_offline_from_its_error_record(
        tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    install_fake_family(monkeypatch, tmp_path / "model",
                        decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY}, tool_format=BreakingFormat())
    write_items(tmp_path)
    common = ["items.jsonl", "--system", "b0", "--tokenizer", "model", "--mode", "think",
              "--reasoning-budget", "16", "--signals"]
    assert run_eval.main([*common, "--out", "online.jsonl", "--votes", "3", "--rules"],
                         local_client=breaking().client()) == 0
    for name, flags in (("greedy", ["--greedy"]), ("s0", ["--seed", "0"]), ("s1", ["--seed", "1"]),
                        ("s2", ["--seed", "2"])):
        assert run_eval.main([*common, "--out", f"{name}.jsonl", *flags],
                             local_client=breaking().client()) == 0
    assert vote_offline.main(["items.jsonl", "--greedy", "greedy.jsonl", "--sample", "s0.jsonl", "--sample",
                              "s1.jsonl", "--sample", "s2.jsonl", "--rules", "--out", "offline.jsonl"]) == 0
    online, offline = (tmp_path / "online.jsonl").read_bytes(), (tmp_path / "offline.jsonl").read_bytes()
    assert online == offline
    # the runs the vote is taken over hold the failures as the runner's error records ...
    failed = [(k, r["id"]) for k in range(3) for r in lines(tmp_path / f"s{k}.jsonl")
              if r["status"] == "error"]
    assert failed and len(failed) < 6
    # ... and the online vote lists each, from the exception its candidate raised, alike
    voted = [json.loads(line) for line in online.decode("utf-8").splitlines()]
    listed = [(e["candidate"], r["id"]) for r in voted for e in r["votes"].get("errors", [])]
    assert sorted(listed) == sorted(failed)
    assert {(e["error_type"], e["message"]) for r in voted for e in r["votes"].get("errors", [])} == {
        ("HarnessError", "parse_tool_calls failed: ValueError('a broken block')")}
    assert all(r["status"] == "ok" for r in voted)  # the votes went on

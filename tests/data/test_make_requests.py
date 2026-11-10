"""scripts/make_requests.py for incremental builds (docs/m7-计划.md §5 and WP-I 1): the default build still
reproduces the registered request file and its manifest byte for byte, and the flags --draw-seed, --only,
--id-start, --preload and --mix build new requests that continue the registered ones: their usage errors, the
dedup against the preloaded files, the id continuation and what the manifest records. The registered request
files under data/synth are the generator's own inputs and outputs; no test opens an evaluation file."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from research_agent.data import requests as req
from research_agent.normalize import normalize_text

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "data" / "corpus" / "manifest.json"
SYNTH = REPO / "data" / "synth"
REQUESTS = SYNTH / "requests.jsonl"
MANIFEST = SYNTH / "requests-manifest.json"
REQUESTS_M6 = SYNTH / "requests-m6.jsonl"
V2_MIX = "normal=0.40,no_call=0.15,clarify=0.15,recover=0.20,parallel=0.10"
M7_MIX = "normal=0.50,no_call=0.10,clarify=0.20,recover=0.12,parallel=0.08"
# the registered request files and their manifest, which the public copy leaves out (tests/conftest.py)
NEEDS_REGISTERED = pytest.mark.needs_data("data/synth")


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_requests = load_script("make_requests")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def run(tmp_path: Path, name: str, *argv: str) -> tuple[int, Path, Path]:
    """make_requests.main writing <name>.jsonl and <name>-manifest.json in tmp_path (a later --out or
    --manifest in argv wins)."""
    out, manifest = tmp_path / f"{name}.jsonl", tmp_path / f"{name}-manifest.json"
    code = make_requests.main(["--out", str(out), "--manifest", str(manifest), "--corpus", str(CORPUS),
                               *argv])
    return code, out, manifest


def v2_style(tmp_path: Path) -> tuple[int, Path, Path]:
    """The v2 build's flags (docs/m7-计划.md WP-I 1) at a tenth of its sizes."""
    return run(tmp_path, "v2", "--only", "test,dev", "--test", "26", "--dev", "22", "--draw-seed", "20261020",
               "--id-start", "test=401", "--id-start", "dev=401", "--preload", str(REQUESTS),
               "--preload", str(REQUESTS_M6), "--mix", V2_MIX)


def masked(text: str) -> frozenset[str]:
    return req.shingles(req.mask_text(text))


def max_jaccard(shingle_set: frozenset[str], others: list[frozenset[str]]) -> float:
    """Brute force over every other set, independent of the module's inverted index."""
    best = 0.0
    for other in others:
        shared = len(shingle_set & other)
        best = max(best, shared / (len(shingle_set) + len(other) - shared))
    return best


@NEEDS_REGISTERED
def test_the_default_build_reproduces_the_registered_files_byte_for_byte(tmp_path):
    code, out, manifest = run(tmp_path, "requests")
    assert code == 0
    assert sha(out) == sha(REQUESTS) == json.loads(MANIFEST.read_text("utf-8"))["requests_sha256"]
    assert sha(manifest) == sha(MANIFEST)


@NEEDS_REGISTERED
def test_naming_the_defaults_of_the_new_flags_changes_nothing(tmp_path):
    out = tmp_path / "requests.jsonl"  # --manifest defaults to <stem>-manifest.json beside --out
    code = make_requests.main(["--out", str(out), "--corpus", str(CORPUS), "--draw-seed", "42",
                               "--only", "train,dev,test", "--id-start", "test=1",
                               "--mix", "normal=0.55,no_call=0.15,clarify=0.10,recover=0.15,parallel=0.05"])
    assert code == 0
    assert sha(out) == sha(REQUESTS) and sha(tmp_path / "requests-manifest.json") == sha(MANIFEST)


@NEEDS_REGISTERED
def test_a_v2_style_build_continues_the_registered_files_without_repeating_them(tmp_path, capsys):
    code, out, manifest = v2_style(tmp_path)
    printed = capsys.readouterr().out
    assert code == 0
    records = read_jsonl(out)
    assert sorted(r["id"] for r in records) == sorted([f"req-dev-{i:04d}" for i in range(401, 423)]
                                                      + [f"req-test-{i:04d}" for i in range(401, 427)])
    data = json.loads(manifest.read_text("utf-8"))
    assert data["seed"] == 42 and data["draw_seed"] == 20261020 and data["only"] == ["dev", "test"]
    assert data["id_start"] == {"dev": 401, "test": 401}
    assert data["planned_mix"] == {"normal": 0.4, "no_call": 0.15, "clarify": 0.15, "recover": 0.2,
                                   "parallel": 0.1}
    assert [(p["path"], p["sha256"], p["requests"]) for p in data["preload"]] == [
        (REQUESTS.as_posix(), sha(REQUESTS), 2400), (REQUESTS_M6.as_posix(), sha(REQUESTS_M6), 510)]
    assert data["sizes"] == data["totals"] == {"train": 0, "dev": 22, "test": 26}
    assert data["audit"]["ok"] and data["template_audit"]["ok"]
    assert set(json.loads(MANIFEST.read_text("utf-8"))) <= set(data)
    assert "preloaded 2910 requests from 2 files" in printed
    assert "audit: shared families 0, topics 0, paper ids 0 across splits: ok" in printed
    old = read_jsonl(REQUESTS) + read_jsonl(REQUESTS_M6)
    old_texts = {normalize_text(r["text"]) for r in old}
    old_sets = [masked(r["text"]) for r in old]
    for r in records:
        assert normalize_text(r["text"]) not in old_texts, r["text"]
        assert max_jaccard(masked(r["text"]), old_sets) < req.NEAR_DUP, r["text"]


@NEEDS_REGISTERED
def test_a_train_build_on_top_keeps_clear_of_every_evaluation_request(tmp_path, capsys):
    code, v2, _ = v2_style(tmp_path)
    assert code == 0
    code, train, manifest = run(tmp_path, "train", "--only", "train", "--train", "130",
                                "--draw-seed", "20261021", "--id-start", "train=1601",
                                "--preload", str(REQUESTS), "--preload", str(REQUESTS_M6),
                                "--preload", str(v2), "--mix", M7_MIX)
    assert code == 0, capsys.readouterr().err
    new = read_jsonl(train)
    assert sorted(r["id"] for r in new) == [f"req-train-{i:04d}" for i in range(1601, 1731)]
    assert {r["split"] for r in new} == {"train"}
    assert [p["requests"] for p in json.loads(manifest.read_text("utf-8"))["preload"]] == [2400, 510, 48]
    old = read_jsonl(REQUESTS) + read_jsonl(REQUESTS_M6) + read_jsonl(v2)
    evaluation = [masked(r["text"]) for r in old if r["split"] != "train"]
    old_train = [masked(r["text"]) for r in old if r["split"] == "train"]
    assert len(evaluation) == 800 + 48 and len(old_train) == 1600 + 510
    in_band = []  # new requests in the train-vs-eval band of a preloaded train request: kept
    for r in new:
        shingle_set = masked(r["text"])
        assert max_jaccard(shingle_set, evaluation) < req.TRAIN_VS_EVAL, r["text"]
        to_train = max_jaccard(shingle_set, old_train)
        assert to_train < req.NEAR_DUP, r["text"]
        if to_train >= req.TRAIN_VS_EVAL:
            in_band.append(r["text"])
    assert in_band  # the train-vs-eval rule runs against the preloaded dev and test requests only


def test_the_cli_prints_and_records_the_mix_it_built(tmp_path, capsys):
    code, _, manifest = run(tmp_path, "r", "--only", "test", "--test", "60", "--mix", V2_MIX)
    printed = capsys.readouterr().out
    assert code == 0
    assert "test: 60 requests (zh+en): normal 12+12, no_call 5+4, clarify 4+5, recover 6+6, parallel 3+3" \
        in printed
    data = json.loads(manifest.read_text("utf-8"))
    assert data["counts"]["test"]["recover"] == {"zh": 6, "en": 6} and data["sizes"]["train"] == 0


def test_only_split_is_another_name_for_only(tmp_path):
    first = run(tmp_path, "a", "--only", "test", "--test", "10")
    second = run(tmp_path, "b", "--only-split", "test", "--test", "10")
    assert first[0] == second[0] == 0
    assert first[1].read_bytes() == second[1].read_bytes() and first[2].read_bytes() == second[2].read_bytes()


@pytest.mark.parametrize("argv", [
    ["--only", "validation"], ["--only", "test,test"], ["--only", ""], ["--only", "test,"],
    ["--only", "test", "--test", "0"],
    ["--id-start", "test"], ["--id-start", "test=0"], ["--id-start", "test=x"], ["--id-start", "bogus=3"],
    ["--id-start", "test=5", "--id-start", "test=6"], ["--only", "test", "--id-start", "dev=401"],
    ["--mix", "normal=0.5"], ["--mix", "normal=0.5,bogus=0.5"], ["--mix", "normal=x,clarify=0.5"],
    ["--mix", "normal=0.5,normal=0.5"], ["--mix", "normal=1.5,clarify=-0.5"], ["--mix", "normal"],
    ["--draw-seed", "x"], ["--preload", "no/such/requests.jsonl"]])
def test_the_cli_exits_two_on_bad_incremental_flags(tmp_path, argv):
    code, out, manifest = run(tmp_path, "r", *argv)
    assert code == 2 and not out.exists() and not manifest.exists()


@NEEDS_REGISTERED
def test_the_cli_refuses_an_id_a_preloaded_file_already_holds(tmp_path, capsys):
    code, out, manifest = run(tmp_path, "r", "--only", "test", "--test", "10", "--id-start", "test=395",
                              "--preload", str(REQUESTS))
    err = capsys.readouterr().err
    assert code == 2 and not out.exists() and not manifest.exists()
    assert "req-test-0395" in err and "req-test-0400" in err and "req-test-0401" not in err
    code, out, _ = run(tmp_path, "r", "--only", "test", "--test", "10", "--preload", str(REQUESTS))
    assert code == 2 and not out.exists() and "req-test-0001" in capsys.readouterr().err  # ids from 1


@pytest.mark.parametrize("content", ["not json\n", '{"id": "req-train-9001"}\n'])
def test_the_cli_exits_two_on_a_malformed_preload_file(tmp_path, capsys, content):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(content, encoding="utf-8")
    code, out, _ = run(tmp_path, "r", "--only", "test", "--test", "10", "--preload", str(bad))
    assert code == 2 and not out.exists() and "line 1" in capsys.readouterr().err


def test_the_cli_will_not_write_over_a_preloaded_file(tmp_path, capsys):
    code, first, first_manifest = run(tmp_path, "first", "--only", "test", "--test", "10")
    assert code == 0
    before = (first.read_bytes(), first_manifest.read_bytes())
    for flag, target in (("--out", first), ("--manifest", first), ("--out", tmp_path / "." / "first.jsonl")):
        code, _, _ = run(tmp_path, "second", "--only", "test", "--test", "10", "--id-start", "test=11",
                         "--preload", str(first), flag, str(target))
        assert code == 2 and "preloaded" in capsys.readouterr().err
    assert (first.read_bytes(), first_manifest.read_bytes()) == before
    assert not (tmp_path / "second.jsonl").exists() and not (tmp_path / "second-manifest.json").exists()

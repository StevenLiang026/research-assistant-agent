"""Assembling the evaluation sets (research_agent.eval.sets, scripts/finalize_eval_sets.py): the
take-then-fill rule over a hand-made pool, every refusal, and the script at the real quotas."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from research_agent.eval.items import LANGS, QUOTAS_DEV, QUOTAS_TEST, scaled
from research_agent.eval.sets import SetError, assemble, per_language, verdict_counts

REPO = Path(__file__).resolve().parents[2]
# call: zh 2 / en 2, target zh 2 / en 1; no_call: zh 1 / en 1, target zh 1 / en 0
QUOTAS = {"dev": {"call": 4, "no_call": 2}}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


finalize = load_script("finalize_eval_sets")


@pytest.fixture(autouse=True)
def no_real_specs(tmp_path, monkeypatch):
    """The script records the digest of data/eval/handwritten-specs.jsonl, which holds the test set's
    handwritten specs: in these tests it digests a file of the test's own, absent unless a test writes it."""
    monkeypatch.setattr(finalize, "HANDWRITTEN_SPECS", tmp_path / "handwritten-specs.jsonl")


def item(iid: str, type_: str, lang: str, split: str = "dev", handwritten: bool = False) -> dict:
    source = {"handwritten": True} if handwritten else {"request": iid.rsplit("-s", 1)[0], "step": 1}
    return {"id": iid, "split": split, "type": type_, "lang": lang, "messages": [], "source": source,
            "expected": {"content": "", "tool_calls": []}}


DRAFTED = [item("d1-s1", "call", "zh"), item("d2-s1", "call", "zh"), item("d3-s1", "call", "zh"),
           item("d4-s1", "call", "zh"), item("d5-s1", "call", "en"), item("d6-s1", "call", "en"),
           item("d7-s1", "no_call", "zh"), item("d8-s1", "no_call", "en")]
ROWS = [{"split": d["split"], "type": d["type"], "lang": d["lang"], "rank": rank, "id": d["id"]}
        for rank, d in enumerate(DRAFTED, 1)]
VERDICTS = {"d1-s1": "keep", "d2-s1": "drop", "d3-s1": "keep", "d4-s1": "keep", "d5-s1": "keep",
            "d6-s1": "keep", "d7-s1": "review_failed", "d8-s1": "keep"}
HANDWRITTEN = [item("hw-dev-001", "call", "en", handwritten=True),
               item("hw-dev-002", "call", "en", handwritten=True),
               item("hw-dev-003", "no_call", "zh", handwritten=True),
               item("hw-dev-004", "no_call", "en", handwritten=True),
               item("hw-dev-005", "no_call", "en", handwritten=True)]
HW_VERDICTS = {"hw-dev-001": "keep", "hw-dev-002": "drop", "hw-dev-003": "keep", "hw-dev-004": "keep"}


def build(**changes):
    args = {"pool_rows": ROWS, "verdicts": VERDICTS, "drafted": {d["id"]: d for d in DRAFTED},
            "handwritten": HANDWRITTEN, "handwritten_verdicts": HW_VERDICTS, "quotas": QUOTAS, **changes}
    return assemble(args.pop("pool_rows"), args.pop("verdicts"), args.pop("drafted"), args.pop("handwritten"),
                    args.pop("handwritten_verdicts"), **args)


def test_kept_drafted_items_come_first_up_to_the_target_and_kept_handwritten_items_fill_the_rest():
    sets, table = build()
    assert [i["id"] for i in sets["dev"]] == [
        "d1-s1", "d3-s1", "d5-s1", "hw-dev-001", "hw-dev-003", "hw-dev-004"]
    assert table["dev"]["call/zh"] == {"full": 2, "target": 2, "drafted_kept": 3, "drafted": 2,
                                       "handwritten_kept": 0, "handwritten": 0}
    assert table["dev"]["call/en"] == {"full": 2, "target": 1, "drafted_kept": 2, "drafted": 1,
                                       "handwritten_kept": 1, "handwritten": 1}
    assert table["dev"]["no_call/en"]["target"] == 0 and table["dev"]["no_call/en"]["drafted"] == 0


def test_a_verdict_other_than_keep_never_enters_a_set():
    sets, _ = build()
    ids = {i["id"] for i in sets["dev"]}
    assert not ids & {"d2-s1", "d7-s1", "hw-dev-002", "hw-dev-005"}  # drop, review_failed, drop, no verdict


def test_the_set_is_ordered_by_type_then_language_then_id():
    reordered = list(reversed(HANDWRITTEN))
    sets, _ = build(handwritten=reordered)
    assert [(i["type"], i["lang"]) for i in sets["dev"]] == [
        ("call", "zh"), ("call", "zh"), ("call", "en"), ("call", "en"), ("no_call", "zh"), ("no_call", "en")]


def test_a_stratum_without_enough_kept_items_is_refused_by_name():
    with pytest.raises(SetError, match="dev/no_call/en needs 1"):
        build(handwritten_verdicts={**HW_VERDICTS, "hw-dev-004": "drop"})


@pytest.mark.parametrize(("change", "message"), [
    ({"drafted": {d["id"]: d for d in DRAFTED[1:]}}, "d1-s1 is not drafted"),
    ({"drafted": {**{d["id"]: d for d in DRAFTED}, "d1-s1": item("d1-s1", "call", "zh", split="test")}},
     "d1-s1 is a test/call/zh item"),
    ({"handwritten": [*HANDWRITTEN, item("d5-s1", "call", "en", handwritten=True)]}, "seen twice"),
    ({"handwritten": [*HANDWRITTEN, item("hw-x-001", "call", "en", split="train", handwritten=True)]},
     "no known split"),
], ids=["undrafted-pool-id", "split-mismatch", "duplicate-id", "unknown-split"])
def test_inconsistent_inputs_are_refused(change, message):
    with pytest.raises(SetError, match=message):
        build(**change)


def test_the_real_quotas_give_150_and_300_items_and_at_least_30_percent_room_for_handwritten_ones():
    for quotas, total in ((QUOTAS_DEV, 150), (QUOTAS_TEST, 300)):
        full, target = per_language(quotas), per_language(scaled(quotas, 0.7))
        assert sum(full.values()) == total and sum(target.values()) == round(0.7 * total)
        assert all(full[k] >= target[k] for k in full)


def test_verdicts_are_counted_by_verdict_and_category():
    rows = [{"verdict": "keep", "category": "ok"}, {"verdict": "drop", "category": "ambiguous"},
            {"verdict": "drop", "category": "ambiguous"}, {"verdict": "review_failed", "category": None}]
    assert verdict_counts(rows) == {"verdicts": {"drop": 2, "keep": 1, "review_failed": 1},
                                    "categories": {"drop": {"ambiguous": 2}, "keep": {"ok": 1},
                                                   "review_failed": {"None": 1}}}


# ---- the script at the real quotas ----------------------------------------------------------------------


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def real_inputs(tmp_path: Path) -> tuple[list[str], list[dict]]:
    """Inputs at the real quotas: per stratum target + 1 drafted items (the last one dropped) and
    full - target + 1 handwritten ones (all kept), and two run folders whose drafts are monkeypatched."""
    drafted, rows, verdicts, handwritten, hw_verdicts = [], [], [], [], []
    for split, quotas in (("dev", QUOTAS_DEV), ("test", QUOTAS_TEST)):
        full, target = per_language(quotas), per_language(scaled(quotas, 0.7))
        for (type_, lang), size in full.items():
            for n in range(target[type_, lang] + 1):
                d = item(f"req-{split}-{type_}-{lang}-{n:03d}-s1", type_, lang, split)
                drafted.append(d)
                rows.append({"split": split, "type": type_, "lang": lang, "rank": n + 1, "id": d["id"]})
                verdicts.append({"id": d["id"], "verdict": "keep" if n < target[type_, lang] else "drop",
                                 "category": "ok", "model": "claude-opus-5-5", "batch": "b1"})
            for n in range(size - target[type_, lang] + 1):
                h = item(f"hw-{split}-{type_}-{lang}-{n:03d}", type_, lang, split, handwritten=True)
                handwritten.append(h)
                hw_verdicts.append({"id": h["id"], "verdict": "keep", "category": "ok",
                                    "model": "claude-opus-5-5", "batch": "b2"})
    paths = [write_jsonl(tmp_path / "pool-ids.jsonl", rows),
             write_jsonl(tmp_path / "verdicts.jsonl", verdicts),
             write_jsonl(tmp_path / "handwritten.jsonl", handwritten),
             write_jsonl(tmp_path / "handwritten-verdicts.jsonl", hw_verdicts)]
    runs = []
    for name in ("dev", "test"):
        run = tmp_path / "runs" / name
        write_jsonl(run / "index.jsonl", [])
        (run / "manifest.json").write_text("{}\n", encoding="utf-8")
        runs.append(str(run))
    argv = ["--pool-ids", str(paths[0]), "--verdicts", str(paths[1]), "--handwritten", str(paths[2]),
            "--handwritten-verdicts", str(paths[3]), "--runs", *runs, "--out-dir", str(tmp_path / "out")]
    return argv, drafted


def test_the_script_writes_150_and_300_items_with_their_manifest_deterministically(tmp_path, monkeypatch):
    argv, drafted = real_inputs(tmp_path)
    monkeypatch.setattr(finalize, "draft_items",
                        lambda run, **kw: drafted if Path(run).name == "dev" else [])
    out = tmp_path / "out"
    (out / "e2e-dev.jsonl").parent.mkdir(parents=True)
    (out / "e2e-dev.jsonl").write_text("{}\n", encoding="utf-8")
    assert finalize.main(argv) == 0
    first = {name: (out / name).read_bytes() for name in ("dev.jsonl", "test.jsonl", "sets-manifest.json")}
    assert finalize.main(argv) == 0
    assert first == {name: (out / name).read_bytes() for name in first}
    manifest = json.loads(first["sets-manifest.json"])
    dev, test = manifest["splits"]["dev"], manifest["splits"]["test"]
    assert (dev["items"], dev["drafted"], dev["handwritten"]) == (150, 105, 45)
    assert (test["items"], test["drafted"], test["handwritten"]) == (300, 210, 90)
    assert test["handwritten_share"] == 0.3 and set(manifest["e2e_tasks"]) == {"e2e-dev.jsonl"}
    assert manifest["review"]["models"] == ["claude-opus-5-5"]
    assert manifest["review"]["batches"] == ["b1", "b2"]
    assert manifest["review"]["drafted"]["verdicts"]["drop"] == 2 * len(LANGS) * 6
    assert "not written by the project owner" in manifest["handwritten_author"]
    assert all(json.loads(line)["split"] == "test" for line in first["test.jsonl"].decode().splitlines())


def test_the_script_exits_1_on_a_short_stratum_and_2_on_a_missing_input_writing_nothing(tmp_path,
                                                                                       monkeypatch):
    argv, drafted = real_inputs(tmp_path)
    monkeypatch.setattr(finalize, "draft_items",
                        lambda run, **kw: drafted if Path(run).name == "dev" else [])
    hw_verdicts = tmp_path / "handwritten-verdicts.jsonl"
    rows = [json.loads(line) for line in hw_verdicts.read_text(encoding="utf-8").splitlines()]
    write_jsonl(hw_verdicts, [{**r, "verdict": "drop"} if r["id"].startswith("hw-test-recover-en") else r
                              for r in rows])
    assert finalize.main(argv) == 1
    assert not (tmp_path / "out").exists()
    (tmp_path / "verdicts.jsonl").unlink()
    assert finalize.main(argv) == 2

"""scripts/finalize_dev_ext.py: the development extension set, the reviewed dev items beyond the dev set's
quotas, assembled from fabricated review records (docs/m6-计划.md §3.2 B2, §4 WP1)."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from research_agent.eval.items import CALL, CLARIFY, LANGS, NO_CALL, QUOTAS_DEV
from research_agent.eval.lock import check_lock, read_jsonl, refuse_test_material, sha256_of, write_jsonl
from research_agent.eval.sets import assemble, sort_key

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


finalize_dev_ext = load_script("finalize_dev_ext")

# Three types with small quotas (items.select splits a type's quota over the two languages); the review
# keeps more items than the quotas take, so every stratum has some left over
QUOTAS = {CALL: 4, NO_CALL: 2, CLARIFY: 2}
DRAFTED_PER_STRATUM, KEPT_DRAFTED, HANDWRITTEN_PER_STRATUM = 8, 6, 4


def item(iid: str, split: str, typ: str, lang: str, *, handwritten: bool = False) -> dict:
    return {"id": iid, "split": split, "type": typ, "lang": lang, "family": f"{typ}/{lang}/1",
            "messages": [{"role": "user", "content": f"请求 {iid}"}], "slots": {}, "expected": {"calls": []},
            "source": {"handwritten": True} if handwritten else {"request": iid}}


def records() -> dict:
    """Per stratum: 8 drafted dev items in pool order, the first 6 kept; 4 handwritten dev items, all kept;
    plus a handwritten TEST item that must never be read into the set."""
    pool, verdicts, drafted, handwritten, hw_verdicts = [], [], {}, [], []
    for typ, quota in QUOTAS.items():
        for lang in LANGS:
            for rank in range(1, DRAFTED_PER_STRATUM + 1):
                iid = f"req-dev-{typ}-{lang}-{rank:02d}-s1"
                pool.append({"drawn": True, "id": iid, "lang": lang, "quota": quota, "rank": rank,
                             "split": "dev", "type": typ})
                verdicts.append({"id": iid, "split": "dev", "type": typ, "lang": lang,
                                 "verdict": "keep" if rank <= KEPT_DRAFTED else "drop"})
                drafted[iid] = item(iid, "dev", typ, lang)
            for n in range(1, HANDWRITTEN_PER_STRATUM + 1):
                iid = f"hw-dev-{typ}-{lang}-{n:02d}"
                handwritten.append(item(iid, "dev", typ, lang, handwritten=True))
                hw_verdicts.append({"id": iid, "split": "dev", "type": typ, "lang": lang, "verdict": "keep"})
    handwritten.append(item("hw-test-001", "test", CALL, "zh", handwritten=True))
    hw_verdicts.append({"id": "hw-test-001", "split": "test", "type": CALL, "lang": "zh", "verdict": "keep"})
    return {"pool": pool, "verdicts": verdicts, "drafted": drafted, "handwritten": handwritten,
            "hw_verdicts": hw_verdicts}


def fixture(tmp_path: Path) -> dict:
    r = records()
    paths = {"pool": tmp_path / "pool-ids.jsonl", "verdicts": tmp_path / "verdicts.jsonl",
             "handwritten": tmp_path / "handwritten.jsonl",
             "hw_verdicts": tmp_path / "handwritten-verdicts.jsonl", "dev": tmp_path / "dev.jsonl",
             "out": tmp_path / "dev-ext.jsonl", "manifest": tmp_path / "dev-ext-manifest.json",
             "run": tmp_path / "run"}
    write_jsonl(r["pool"], paths["pool"])
    write_jsonl(r["verdicts"], paths["verdicts"])
    write_jsonl(r["handwritten"], paths["handwritten"])
    write_jsonl(r["hw_verdicts"], paths["hw_verdicts"])
    paths["run"].mkdir()
    (paths["run"] / "index.jsonl").write_text("", encoding="utf-8")
    (paths["run"] / "manifest.json").write_text("{}", encoding="utf-8")
    # the dev set as scripts/finalize_eval_sets.py assembled it from the same records
    dev_hand = [h for h in r["handwritten"] if h["split"] == "dev"]
    sets, table = assemble(r["pool"], {v["id"]: v["verdict"] for v in r["verdicts"]}, r["drafted"], dev_hand,
                           {v["id"]: v["verdict"] for v in r["hw_verdicts"]}, quotas={"dev": QUOTAS})
    write_jsonl(sets["dev"], paths["dev"])
    paths["records"], paths["table"] = r, table["dev"]
    return paths


def argv(paths: dict, *extra: str) -> list[str]:
    return ["--runs", str(paths["run"]), "--pool-ids", str(paths["pool"]),
            "--verdicts", str(paths["verdicts"]), "--handwritten", str(paths["handwritten"]),
            "--handwritten-verdicts", str(paths["hw_verdicts"]),
            "--dev", str(paths["dev"]), "--out", str(paths["out"]), "--manifest", str(paths["manifest"]),
            *extra]


@pytest.fixture
def drafting(monkeypatch):
    """draft_items replaced by the fabricated drafted items (the generator runs are not needed), and the
    script's quotas by the fixture's."""
    holder = {}
    monkeypatch.setattr(finalize_dev_ext, "draft_items",
                        lambda run, **kwargs: list(holder["drafted"].values()))
    monkeypatch.setattr(finalize_dev_ext, "QUOTAS", {"dev": QUOTAS})
    return holder


def test_the_extension_is_the_kept_items_beyond_the_dev_quotas_in_the_review_order(tmp_path, drafting,
                                                                                    capsys):
    paths = fixture(tmp_path)
    drafting["drafted"] = paths["records"]["drafted"]
    assert finalize_dev_ext.main(argv(paths)) == 0
    dev_ids = {i["id"] for i in read_jsonl(paths["dev"])}
    ext = read_jsonl(paths["out"])
    ext_ids = [i["id"] for i in ext]
    kept = {i for i, v in ((v["id"], v["verdict"]) for v in paths["records"]["verdicts"]) if v == "keep"}
    kept |= {h["id"] for h in paths["records"]["handwritten"] if h["split"] == "dev"}
    # exactly the kept dev items the dev set did not take; the test item never; ordered as the dev set is
    assert set(ext_ids) == kept - dev_ids and not dev_ids & set(ext_ids)
    assert "hw-test-001" not in ext_ids and all(i["split"] == "dev" for i in ext)
    assert ext_ids == [i["id"] for i in sorted(ext, key=sort_key)]
    refuse_test_material(ext, "dev-ext test")
    # per stratum: what the review kept beyond what the dev set took, drafted in pool order (the lowest
    # ranks went to dev, the next ranks come here)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    for name, counts in paths["table"].items():
        typ, lang = name.split("/")
        drafted_ext = [i for i in ext_ids if i.startswith(f"req-dev-{typ}-{lang}-")]
        expected_ranks = list(range(counts["drafted"] + 1, KEPT_DRAFTED + 1))
        assert drafted_ext == [f"req-dev-{typ}-{lang}-{rank:02d}-s1" for rank in expected_ranks]
        assert manifest["strata"][name] == {
            "drafted": counts["drafted_kept"] - counts["drafted"],
            "handwritten": counts["handwritten_kept"] - counts["handwritten"],
            "dev_drafted": counts["drafted"], "dev_handwritten": counts["handwritten"],
            "drafted_kept": counts["drafted_kept"], "handwritten_kept": counts["handwritten_kept"]}
    assert manifest["items"] == len(ext) == manifest["drafted"] + manifest["handwritten"] > 0
    assert check_lock(paths["out"]) == sha256_of(paths["out"]) == manifest["output"]["sha256"]
    assert manifest["dev"]["sha256"] == sha256_of(paths["dev"])
    assert f"{len(ext)} items" in capsys.readouterr().out


def test_the_same_inputs_give_the_same_bytes_and_an_existing_output_is_kept_without_force(tmp_path, drafting):
    paths = fixture(tmp_path)
    drafting["drafted"] = paths["records"]["drafted"]
    assert finalize_dev_ext.main(argv(paths)) == 0
    first = paths["out"].read_bytes(), paths["manifest"].read_bytes()
    assert finalize_dev_ext.main(argv(paths)) == 2
    assert finalize_dev_ext.main(argv(paths, "--force")) == 0
    assert (paths["out"].read_bytes(), paths["manifest"].read_bytes()) == first


def test_a_dev_set_that_the_records_do_not_reproduce_is_refused(tmp_path, drafting, capsys):
    # the extension is defined relative to the dev set these records made; another dev file means other
    # inputs, and the subtraction would be meaningless
    paths = fixture(tmp_path)
    drafting["drafted"] = paths["records"]["drafted"]
    dev = read_jsonl(paths["dev"])
    write_jsonl(dev[:-1], paths["dev"])
    assert finalize_dev_ext.main(argv(paths)) == 1
    assert "dev set" in capsys.readouterr().err and not paths["out"].exists()


def test_a_missing_input_exits_2_and_writes_nothing(tmp_path, drafting, capsys):
    paths = fixture(tmp_path)
    drafting["drafted"] = paths["records"]["drafted"]
    paths["verdicts"].unlink()
    assert finalize_dev_ext.main(argv(paths)) == 2
    assert "missing input" in capsys.readouterr().err and not paths["out"].exists()


def test_the_real_dev_set_has_the_quotas_the_script_assumes():
    assert finalize_dev_ext.QUOTAS == {"dev": QUOTAS_DEV}

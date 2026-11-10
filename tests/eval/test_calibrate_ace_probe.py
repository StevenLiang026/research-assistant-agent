"""scripts/calibrate_ace_probe.py (M6 WP4, D3) on fabricated probe results, and on the recorded ACEBench
runs."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cp = load_script("calibrate_ace_probe")
PROBE_SHA = "a" * 64


def probe_result(system: str, parsable: float, sentence: float, sha: str = PROBE_SHA) -> dict:
    return {
        "manifest": {"system": system, "probe": {"sha256": sha, "items": 150}},
        "kinds": {
            "normal": {"n": 100, "parsable": round(100 * parsable), "parsable_rate": parsable},
            "incomplete": {"n": 50, "sentence": round(50 * sentence), "sentence_rate": sentence},
        },
    }


def acebench(systems: dict[str, tuple[int, int]]) -> dict:
    """systems: name -> (acebench_format answers of 200 normal answers, correct of 50 special_incomplete),
    split over the two languages."""
    groups = {"normal": ["normal_a", "normal_b"], "special": ["special_incomplete", "special_irrelevant"]}
    out = {"groups": groups, "systems": {}}
    for name, (fmt, inc) in systems.items():
        langs = {}
        for lang, share in (("zh", fmt // 2), ("en", fmt - fmt // 2)):
            c = inc // 2 if lang == "zh" else inc - inc // 2
            langs[lang] = {
                "format": {
                    "normal_a": {"n": 50, "acebench_format": share // 2},
                    "normal_b": {"n": 50, "acebench_format": share - share // 2},
                    "special_incomplete": {"n": 25, "acebench_format": 0},
                    "special_irrelevant": {"n": 25, "acebench_format": 0},
                },
                "categories": {
                    "special_incomplete": {"n": 25, "correct": c},
                    "special_irrelevant": {"n": 25, "correct": 3},
                },
            }
        out["systems"][name] = {"languages": langs}
    return out


def write(tmp_path: Path, name: str, data: dict) -> str:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def run(tmp_path, probes: dict[str, dict], ace=None):
    ace = ace or acebench({"b0": (170, 25), "ft": (135, 4), "b4": (143, 9)})
    a = {"groups": ace["groups"], "systems": {k: v for k, v in ace["systems"].items() if k != "b4"}}
    b = {"groups": ace["groups"], "systems": {k: v for k, v in ace["systems"].items() if k == "b4"}}
    files = [write(tmp_path, f"p-{s}.json", r) for s, r in probes.items()]
    out = tmp_path / "calibration.json"
    code = cp.main(
        [*files, "--acebench", write(tmp_path, "a.json", a), write(tmp_path, "b.json", b), "--out", str(out)]
    )
    return code, out


GOOD = {
    "b0": probe_result("b0", 0.85, 0.50),
    "ft": probe_result("ft", 0.60, 0.10),
    "b4": probe_result("b4", 0.64, 0.14),
}


def test_ordering_rule():
    assert cp.ordering({"b0": 0.8, "ft": 0.5, "b4": 0.55})["holds"]
    assert not cp.ordering({"b0": 0.8, "ft": 0.85, "b4": 0.5})["holds"]  # FT above B0
    assert not cp.ordering({"b0": 0.8, "ft": 0.3, "b4": 0.7})["holds"]  # B4 and FT far apart
    assert not cp.ordering({"b0": 0.5, "ft": 0.5, "b4": 0.5})["holds"]  # no gap at all
    assert not cp.ordering({"b0": 0.8, "ft": 0.78, "b4": 0.78}, margin=0.05)["holds"]


def test_usable_probe_exits_0_and_writes_the_calibration(tmp_path, capsys):
    code, out = run(tmp_path, GOOD)
    assert code == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["usable"] is True
    assert r["probe"]["parsable_rate"] == {"b0": 0.85, "ft": 0.6, "b4": 0.64}
    assert r["acebench"]["parsable_rate"]["b0"] == pytest.approx(170 / 200)
    assert r["acebench"]["sentence_rate"]["ft"] == pytest.approx(4 / 50)
    assert all(r["checks"][k]["holds"] for k in r["checks"])
    assert "probe usable" in capsys.readouterr().out


@pytest.mark.parametrize(
    "change",
    [
        {"ft": probe_result("ft", 0.90, 0.10)},  # parsable: FT above B0
        {"b4": probe_result("b4", 0.64, 0.45)},  # sentence: B4 near B0, far from FT
    ],
)
def test_wrong_ordering_exits_1_with_probe_unusable(tmp_path, capsys, change):
    code, out = run(tmp_path, {**GOOD, **change})
    assert code == 1
    assert "probe unusable" in capsys.readouterr().out
    assert json.loads(out.read_text(encoding="utf-8"))["usable"] is False


def test_acebench_reference_that_does_not_hold_is_unusable(tmp_path, capsys):
    ace = acebench({"b0": (170, 25), "ft": (100, 4), "b4": (165, 9)})
    code, _ = run(tmp_path, GOOD, ace)
    assert code == 1 and "probe unusable" in capsys.readouterr().out


def test_inputs_must_be_three_systems_on_one_probe(tmp_path):
    assert run(tmp_path, {"b0": GOOD["b0"], "ft": GOOD["ft"], "x": GOOD["ft"]})[0] == 2
    assert run(tmp_path, {**GOOD, "b4": probe_result("b4", 0.64, 0.14, sha="b" * 64)})[0] == 2
    ace = acebench({"b0": (170, 25), "ft": (135, 4)})
    assert run(tmp_path, GOOD, ace)[0] == 2


def test_recorded_acebench_runs_show_the_required_ordering():
    with open(REPO / "results/acebench-m4.json", encoding="utf-8") as f:
        a = json.load(f)
    with open(REPO / "results/acebench-m4-b4.json", encoding="utf-8") as f:
        b = json.load(f)
    rates = cp.acebench_rates([a, b])
    assert rates["parsable_rate"]["b0"] == pytest.approx(1387 / 1640)
    assert rates["sentence_rate"]["b4"] == pytest.approx(19 / 100)
    assert cp.ordering(rates["parsable_rate"])["holds"] and cp.ordering(rates["sentence_rate"])["holds"]

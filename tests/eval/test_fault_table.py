"""scripts/fault_table.py: the M4 fault table (docs/plan.md, 故障注入) from the reports of several end-to-end
runs, per system and fault condition, built from the records and the timing sidecars alone."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from research_agent.net.faults import FAULTS

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fault_table = load_script("fault_table")
TASKS_SHA = "a" * 64


def fault(tool: str, kind: str, call: int | None, key: str = "k", ordinal: int = 0) -> dict:
    return {"tool": tool, "key": key, "kind": kind, "ordinal": ordinal, "step": None if call is None else 1,
            "call": call, "call_id": None if call is None else f"c{call}"}


def record(rid: str, success: bool, faults: list[dict] | None = None, *, tool_calls: dict | None = None,
           tool_errors: dict | None = None, cited_ok: bool = True, status: str = "answered") -> dict:
    checks = {} if status == "harness_error" else {"cited_ids": {"ok": cited_ok, "cited": [], "unseen": []}}
    out = {"id": rid, "intent": "lookup_paper", "lang": "en", "split": "dev", "status": status,
           "success": success, "checks": checks, "steps": 2, "tool_calls": tool_calls or {},
           "tool_errors": tool_errors or {}}
    if faults is not None:
        out["faults"] = faults
    return out


def write_run(folder: Path, system: str, records: list[dict], *, rate: float = 0.0, persistent: bool = False,
              tools: dict[str, list[dict]] | None = None, seed: int = 42, tasks_sha: str = TASKS_SHA) -> Path:
    """A run folder as scripts/run_e2e.py leaves it: report.json and timing.json."""
    manifest = {"system": system, "model": system, "seed": seed, "tasks": "data/eval/e2e-dev.jsonl",
                "tasks_sha256": tasks_sha}
    if rate:
        manifest["fault_injection"] = {"rate": rate, "persistent": persistent, "faults": list(FAULTS),
                                       "seed": seed}
    folder.mkdir(parents=True)
    (folder / "report.json").write_text(json.dumps({"manifest": manifest, "summary": {}, "tasks": records}),
                                        encoding="utf-8")
    timing = [{"id": r["id"], "status": r["status"], "wall_ms": 1.0, "steps": [], "calls": [],
               **({"tools": (tools or {}).get(r["id"], [])} if rate else {})} for r in records]
    (folder / "timing.json").write_text(json.dumps({"report": "report.json", "tasks": timing}),
                                        encoding="utf-8")
    return folder / "report.json"


def timed(tool: str, wall_ms: float, *faults: str) -> dict:
    return {"tool": tool, "step": 1, "call": 0, "wall_ms": wall_ms, "faults": list(faults)}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(tmp_path: Path, monkeypatch, *reports: Path, extra: tuple = ()) -> tuple[int, dict | None]:
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "results" / "faults-m4.json"
    code = fault_table.main([*(str(r.relative_to(tmp_path)) for r in reports), *extra])
    return code, json.loads(out.read_text(encoding="utf-8")) if code == 0 else None


def test_the_table_counts_success_recovery_fabrication_and_faults_per_kind(tmp_path, monkeypatch):
    records = [
        record("t1", True, []),  # met no fault
        record("t2", True, [fault("get_paper", "timeout", 0), fault("get_paper", "delay", 1)],
               tool_calls={"get_paper": 2}),
        record("t3", False, [fault("search_arxiv", "empty_results", 0)], tool_calls={"search_arxiv": 1},
               tool_errors={"EMPTY_RESULTS": 1}, cited_ok=False),  # cited an id after an empty search
        record("t4", False, [fault("get_paper", "empty_results", 0)], tool_calls={"get_paper": 1},
               tool_errors={"NOT_FOUND": 1}, cited_ok=True),  # empty, but cited nothing it had not seen
        record("t5", False, [], tool_calls={"get_paper": 1}, tool_errors={"NOT_FOUND": 1},
               cited_ok=False),  # an unknown id the model asked for: NOT_FOUND without an injected fault
        record("t6", False, [fault("get_paper", "server_error", None)], status="harness_error"),
    ]
    code, report = table(tmp_path, monkeypatch, write_run(tmp_path / "ft-50", "ft", records, rate=0.5))
    assert code == 0
    [row] = report["rows"]
    assert (row["system"], row["condition"], row["rate"], row["persistent"]) == ("ft", "rate=0.5", 0.5, False)
    assert (row["tasks"], row["successes"], row["success_rate"]) == (6, 2, 2 / 6)
    assert (row["tasks_with_faults"], row["recovered"], row["recovery_rate"]) == (4, 1, 1 / 4)
    assert row["empty_result_tasks"] == 3 and row["fabricated_ids_after_empty"] == 2  # t3 and t5
    assert row["retry_storms"] == 0
    assert list(row["by_kind"]) == list(FAULTS)
    assert row["by_kind"]["timeout"] == {"met": 1, "tasks": 1, "succeeded": 1}
    assert row["by_kind"]["empty_results"] == {"met": 2, "tasks": 2, "succeeded": 0}
    assert row["by_kind"]["server_error"] == {"met": 1, "tasks": 1, "succeeded": 0}
    assert row["by_kind"]["truncated_atom"] == {"met": 0, "tasks": 0, "succeeded": 0}


def test_a_retry_storm_is_more_than_3_calls_to_a_tool_right_after_an_injected_failure(tmp_path, monkeypatch):
    def failing(rid: str, failed: list[int], calls: int, kind: str = "timeout") -> dict:
        return record(rid, False, [fault("fetch_fulltext", kind, n) for n in failed],
                      tool_calls={"fetch_fulltext": calls})

    records = [failing("storm", [0, 1, 2, 3], 5),  # calls 1-4 each follow a failure: 4 > 3
               failing("three", [0, 1, 2], 4),  # calls 1-3: exactly 3, no storm
               failing("gaps", [0, 2, 4, 6], 8),  # calls 1, 3, 5, 7 follow a failure: a storm
               failing("last", [0, 1, 2, 3], 4),  # the fourth failure had no call after it: 3
               failing("delays", [0, 1, 2, 3], 5, "delay"),  # a delay answers: no failure
               failing("html", [0, 1, 2, 3], 5, "html_404")]  # the abstract stands in: no failure
    code, report = table(tmp_path, monkeypatch, write_run(tmp_path / "b0-20", "b0", records, rate=0.2))
    assert code == 0 and report["rows"][0]["retry_storms"] == 2
    assert report["rows"][0]["retry_storm_tasks"] == ["gaps", "storm"]


def test_the_delay_row_is_the_extra_wall_time_of_a_delayed_fetch_call_over_a_clean_one(tmp_path, monkeypatch):
    records = [record("t1", True, [fault("get_paper", "delay", 0)]), record("t2", True, [])]
    tools = {"t1": [timed("get_paper", 5012.0, "delay"), timed("fetch_fulltext", 10.0),
                    timed("search_arxiv", 1.0), timed("search_arxiv", 9000.0, "timeout")],
             "t2": [timed("fetch_fulltext", 14.0), timed("get_paper", 5020.0, "delay"),
                    timed("get_paper", 3.0, "delay", "timeout")]}  # two faults: in neither group
    code, report = table(tmp_path, monkeypatch, write_run(tmp_path / "b2-50", "b2", records, rate=0.5,
                                                          tools=tools))
    assert code == 0
    assert report["rows"][0]["delay"] == {"delayed_fetch_calls": 2, "clean_fetch_calls": 2,
                                          "mean_ms_delayed": 5016.0, "mean_ms_clean": 12.0,
                                          "extra_ms_per_call": 5004.0}


def test_rows_per_system_and_condition_with_every_input_and_its_sha256(tmp_path, monkeypatch):
    zero = write_run(tmp_path / "runs" / "ft-0", "ft", [record("t1", True)])  # rate 0: no fault_injection
    half = write_run(tmp_path / "runs" / "ft-50p", "ft", [record("t1", False, [])], rate=0.5, persistent=True)
    b0 = write_run(tmp_path / "runs" / "b0-20", "b0", [record("t1", True, [])], rate=0.2)
    code, report = table(tmp_path, monkeypatch, half, zero, b0)
    assert code == 0
    assert [(r["system"], r["condition"]) for r in report["rows"]] == [
        ("b0", "rate=0.2"), ("ft", "rate=0"), ("ft", "rate=0.5,persistent")]
    assert report["rows"][1]["tasks_with_faults"] == 0 and report["rows"][1]["recovery_rate"] is None
    assert report["rows"][1]["delay"]["extra_ms_per_call"] is None
    assert report["inputs"][0] == {"report": "runs/ft-50p/report.json", "report_sha256": sha(half),
                                   "timing": "runs/ft-50p/timing.json",
                                   "timing_sha256": sha(half.parent / "timing.json"), "system": "ft",
                                   "condition": "rate=0.5,persistent"}
    assert [i["report"] for i in report["inputs"]] == ["runs/ft-50p/report.json", "runs/ft-0/report.json",
                                                      "runs/b0-20/report.json"]
    text = (tmp_path / "results" / "faults-m4.json").read_bytes()
    assert b"\r\n" not in text and text.endswith(b"}\n") and text.startswith(b'{\n  "')
    assert report["tasks_sha256"] == TASKS_SHA and report["seed"] == 42


def test_an_existing_table_is_kept_unless_force_is_given(tmp_path, monkeypatch, capsys):
    run = write_run(tmp_path / "ft-0", "ft", [record("t1", True)])
    assert table(tmp_path, monkeypatch, run)[0] == 0
    out = tmp_path / "results" / "faults-m4.json"
    out.write_text("kept\n", encoding="utf-8")
    assert table(tmp_path, monkeypatch, run)[0] == 3
    assert "--force" in capsys.readouterr().err and out.read_text(encoding="utf-8") == "kept\n"
    code, report = table(tmp_path, monkeypatch, run, extra=("--force",))
    assert code == 0 and report["rows"][0]["tasks"] == 1


def test_inputs_that_do_not_belong_together_are_refused_with_exit_2(tmp_path, monkeypatch, capsys):
    one = write_run(tmp_path / "a", "ft", [record("t1", True, [])], rate=0.2)
    same = write_run(tmp_path / "b", "ft", [record("t1", True, [])], rate=0.2)  # the same condition again
    other_tasks = write_run(tmp_path / "c", "b0", [record("t1", True)], tasks_sha="b" * 64)
    other_seed = write_run(tmp_path / "d", "b2", [record("t1", True)], seed=7)
    for pair, message in (((one, same), "twice"), ((one, other_tasks), "tasks"), ((one, other_seed), "seed")):
        assert table(tmp_path, monkeypatch, *pair) == (2, None)
        assert message in capsys.readouterr().err


def test_malformed_inputs_are_refused_with_exit_1_and_nothing_is_written(tmp_path, monkeypatch, capsys):
    broken = tmp_path / "broken" / "report.json"
    broken.parent.mkdir()
    broken.write_text("{not json", encoding="utf-8")
    no_faults = write_run(tmp_path / "nofaults", "ft", [record("t1", True)], rate=0.2)  # no "faults" key
    no_timing = write_run(tmp_path / "notiming", "b0", [record("t1", True)])
    (no_timing.parent / "timing.json").unlink()
    for report in (broken, no_faults, no_timing):
        assert table(tmp_path, monkeypatch, report) == (1, None), report
        assert "FAIL" in capsys.readouterr().err


def test_the_table_is_built_from_real_run_e2e_runs(tmp_path, monkeypatch):
    """Two conditions of one scripted system through scripts/run_e2e.py, then the table over both."""
    from eval.test_run_e2e_cli import ANSWER, GET2, LOOKUP, run_e2e, world, write_tasks
    from support.fakes import ScriptedModel, turn

    monkeypatch.chdir(tmp_path)
    tasks = write_tasks(tmp_path / "tasks.jsonl", [LOOKUP])
    for name, extra in (("e2e-b2-f0", []), ("e2e-b2-f100", ["--fault-rate", "1"])):
        assert run_e2e.main([str(tasks), "--system", "b2", "--out", name, "--no-tokenizer", "--seed", "3",
                             "--today", "2026-10-10", *extra],
                            model=ScriptedModel([turn(GET2), turn(content=ANSWER)]), world=world(),
                            sleep=lambda seconds: None) == 0
    code, report = table(tmp_path, monkeypatch, tmp_path / "e2e-b2-f0" / "report.json",
                         tmp_path / "e2e-b2-f100" / "report.json")
    assert code == 0
    zero, full = report["rows"]
    assert (zero["condition"], zero["tasks"], zero["tasks_with_faults"]) == ("rate=0", 1, 0)
    assert (full["condition"], full["tasks"], full["tasks_with_faults"]) == ("rate=1", 1, 1)
    assert sum(cell["met"] for cell in full["by_kind"].values()) >= 1

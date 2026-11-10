"""scripts/compare_synth_runs.py offline: two generator run folders built by hand, a paid run and its cached
rerun, compared the way scripts/m1_smoke.py's neutral() compares them (identical apart from money, the second
run free)."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

from research_agent.canonical import canonical_json

REPO = Path(__file__).resolve().parents[2]
TODAY = "2026-10-10"


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare = load_script("compare_synth_runs")


def row(rid: str, *, status: str = "answered", paid: int = 2, cached: int = 0, cost: float = 0.01,
        rewrite_cost: float = 0.002, transcript: str | None = "default", **extra) -> dict:
    """An index row as research_agent.data.synth writes one, with the fields the comparison looks at."""
    return {"id": rid, "intent": "recent_papers", "scenario": "normal", "lang": "zh", "split": "dev",
            "family": 1, "text": f"request {rid}", "rewritten": None, "rewrite_fallback": False,
            "rewrite_cost_rmb": rewrite_cost, "status": status, "error": None, "steps": 2, "paid_calls": paid,
            "cache_hits": cached, "cost_rmb": cost, "faults": {}, "verdict": None, "placeholders": False,
            "transcript": f"transcripts/{rid}.json" if transcript == "default" else transcript, "slots": {},
            "extra": {}, **extra}


def transcript(rid: str, *, paid: int = 2, cached: int = 0, local: int = 1, cost: float = 0.01,
               answer: str = "done") -> dict:
    """A transcript as Transcript.to_json() reads back, with the four money fields at the top level."""
    return {"request": f"request {rid}", "today": TODAY, "status": "answered", "final_text": answer,
            "messages": [{"role": "user", "content": f"request {rid}"},
                         {"role": "assistant", "content": answer}],
            "steps": [{"index": 1, "attempts": []}], "schema_hash": "s", "prompt_version": "v",
            "prompt_hash": "p", "paid_calls": paid, "cache_hits": cached, "local_calls": local,
            "cost_rmb": cost, "usage": {}}


def manifest(*, run: str, paid: int, cached: int, cost: float) -> dict:
    return {"config": {"run": run, "seed": 42, "placeholders": False}, "today": TODAY, "schema_hash": "s",
            "requests": 3, "calls": {"paid": paid, "cached": cached},
            "cost_rmb": {"trajectories": cost, "rewrites": 0.0, "total": cost}}


def write_run(run: Path, rows: list[dict], transcripts: dict[str, dict], manifest: dict | None) -> None:
    """A run folder in the generator's layout: index.jsonl in the given row order, transcripts/<id>.json,
    manifest.json when one is given."""
    run.mkdir(parents=True, exist_ok=True)
    (run / "index.jsonl").write_text("".join(canonical_json(r) + "\n" for r in rows), encoding="utf-8")
    for rid, data in transcripts.items():
        path = run / "transcripts" / f"{rid}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if manifest is not None:
        (run / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")


def paid_and_cached(tmp_path: Path,
                    ids: tuple[str, ...] = ("r001", "r002", "r003")) -> tuple[Path, Path]:
    """Run A paid for every call; run B is its cached rerun: the same rows and transcripts apart from
    money."""
    a, b = tmp_path / "a", tmp_path / "b"
    write_run(a, [row(rid, paid=2, cost=0.01) for rid in ids], {rid: transcript(rid, paid=2) for rid in ids},
              manifest(run="a", paid=2 * len(ids), cached=0, cost=0.01 * len(ids)))
    write_run(b, [row(rid, paid=0, cached=2, cost=0.0, rewrite_cost=0.0) for rid in ids],
              {rid: transcript(rid, paid=0, cached=2, cost=0.0) for rid in ids},
              manifest(run="b", paid=0, cached=2 * len(ids), cost=0.0))
    return a, b


def request_lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if re.match(r"r\d{3}\s", line)]


def test_a_cached_rerun_identical_apart_from_money_passes_and_prints_the_totals(tmp_path, capsys):
    a, b = paid_and_cached(tmp_path)
    assert compare.main([str(a), str(b)]) == 0
    printed = capsys.readouterr()
    assert printed.err == ""
    lines = request_lines(printed.out)
    assert [line.split()[0] for line in lines] == ["r001", "r002", "r003"]
    for line in lines:
        assert "answered" in line and "row same" in line and "transcript same" in line
        assert re.search(r"A paid 2 cost 0\.0120 RMB", line) and re.search(r"B paid 0 cost 0\.0000 RMB", line)
    assert ("totals: A 3 requests, paid 6, cached 0, cost 0.0360 RMB; "
            "B 3 requests, paid 0, cached 6, cost 0.0000 RMB") in printed.out
    assert "manifest: keys that differ: calls, config, cost_rmb" in printed.out
    assert printed.out.splitlines()[-1] == ("VERDICT: PASS: rows 3/3 identical, transcripts 3/3 identical, "
                                            "run B paid 0 calls")


def test_a_differing_transcript_fails_with_exit_1_and_names_the_request(tmp_path, capsys):
    a, b = paid_and_cached(tmp_path)
    changed = transcript("r002", paid=0, cached=2, cost=0.0, answer="another answer")
    (b / "transcripts" / "r002.json").write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n",
                                                 encoding="utf-8")
    assert compare.main([str(a), str(b)]) == 1
    printed = capsys.readouterr()
    lines = {line.split()[0]: line for line in request_lines(printed.out)}
    assert "row same" in lines["r002"] and "transcript differs" in lines["r002"]
    assert all("transcript same" in lines[rid] for rid in ("r001", "r003"))
    assert printed.out.splitlines()[-1] == ("VERDICT: FAIL: rows 3/3 identical, transcripts 2/3 identical, "
                                            "run B paid 0 calls")
    # a row that differs beyond money (here its status) is reported on the row side
    rows = [row("r001", paid=0, cached=2, cost=0.0, rewrite_cost=0.0, status="clarification"),
            row("r002", paid=0, cached=2, cost=0.0, rewrite_cost=0.0),
            row("r003", paid=0, cached=2, cost=0.0, rewrite_cost=0.0)]
    (b / "index.jsonl").write_text("".join(canonical_json(r) + "\n" for r in rows), encoding="utf-8")
    assert compare.main([str(a), str(b)]) == 1
    lines = {line.split()[0]: line for line in request_lines(capsys.readouterr().out)}
    assert "row differs" in lines["r001"] and "answered / clarification" in lines["r001"]


def test_paid_calls_in_run_b_fail_even_when_everything_is_identical(tmp_path, capsys):
    a, b = paid_and_cached(tmp_path)
    rows = [row("r001", paid=0, cached=2, cost=0.0, rewrite_cost=0.0),
            row("r002", paid=1, cached=1, cost=0.005, rewrite_cost=0.0),
            row("r003", paid=0, cached=2, cost=0.0, rewrite_cost=0.0)]
    (b / "index.jsonl").write_text("".join(canonical_json(r) + "\n" for r in rows), encoding="utf-8")
    assert compare.main([str(a), str(b)]) == 1
    printed = capsys.readouterr()
    lines = request_lines(printed.out)
    assert all("row same" in line and "transcript same" in line for line in lines)  # money is not compared
    assert re.search(r"r002 .*B paid 1 cost 0\.0050 RMB", printed.out)
    assert "B 3 requests, paid 1, cached 5, cost 0.0050 RMB" in printed.out
    assert printed.out.splitlines()[-1] == ("VERDICT: FAIL: rows 3/3 identical, transcripts 3/3 identical, "
                                            "run B paid 1 calls")


def test_a_folder_without_an_index_is_exit_2_and_nothing_is_compared(tmp_path, capsys):
    a, b = paid_and_cached(tmp_path)
    (b / "index.jsonl").unlink()
    assert compare.main([str(a), str(b)]) == 2
    printed = capsys.readouterr()
    assert printed.out == "" and str(b) in printed.err and "index.jsonl" in printed.err
    assert compare.main([str(tmp_path / "nowhere"), str(a)]) == 2
    assert "nowhere" in capsys.readouterr().err
    (b / "index.jsonl").write_text("[1, 2]\n", encoding="utf-8")  # a line that is not a row is no index
    assert compare.main([str(a), str(b)]) == 2
    assert "index.jsonl" in capsys.readouterr().err


def test_a_row_missing_on_one_side_a_transcript_file_gone_and_rows_naming_no_transcript(tmp_path, capsys):
    a, b = paid_and_cached(tmp_path)
    rows = [row("r001", paid=0, cached=2, cost=0.0, rewrite_cost=0.0),
            row("r002", paid=0, cached=2, cost=0.0, rewrite_cost=0.0),
            row("r004", paid=0, cached=0, cost=0.0, rewrite_cost=0.0, status="harness_error",
                transcript=None)]
    (b / "index.jsonl").write_text("".join(canonical_json(r) + "\n" for r in rows), encoding="utf-8")
    (b / "transcripts" / "r002.json").unlink()
    assert compare.main([str(a), str(b)]) == 1
    printed = capsys.readouterr()
    lines = {line.split()[0]: line for line in request_lines(printed.out)}
    assert list(lines) == ["r001", "r002", "r003", "r004"]
    assert "row same" in lines["r001"] and "transcript same" in lines["r001"]
    assert "row same" in lines["r002"] and "transcript missing in B" in lines["r002"]
    assert "row missing in B" in lines["r003"] and "transcript missing in B" in lines["r003"]
    assert re.search(r"A paid 2 cost 0\.0120 RMB +B -", lines["r003"])
    assert "row missing in A" in lines["r004"] and "transcript missing in A" in lines["r004"]
    assert printed.out.splitlines()[-1] == ("VERDICT: FAIL: rows 2/4 identical, transcripts 1/4 identical, "
                                            "run B paid 0 calls")
    # two rows that both name no transcript have nothing to compare there: identical
    for run in (a, b):
        (run / "index.jsonl").write_text(canonical_json(row("r004", paid=0, cost=0.0, rewrite_cost=0.0,
                                                            status="harness_error", transcript=None)) + "\n",
                                         encoding="utf-8")
    assert compare.main([str(a), str(b)]) == 0
    printed = capsys.readouterr()
    assert "transcript none" in request_lines(printed.out)[0]
    assert printed.out.splitlines()[-1].startswith("VERDICT: PASS: rows 1/1 identical, transcripts 1/1")


def test_the_lines_are_sorted_by_id_whatever_the_file_order_and_the_output_is_deterministic(tmp_path, capsys):
    a, b = paid_and_cached(tmp_path, ids=("r003", "r001", "r002"))  # written in that order on both sides
    assert [r["id"] for r in map(json.loads, (a / "index.jsonl").read_text("utf-8").splitlines())] == [
        "r003", "r001", "r002"]
    assert compare.main([str(a), str(b)]) == 0
    first = capsys.readouterr().out
    assert [line.split()[0] for line in request_lines(first)] == ["r001", "r002", "r003"]
    assert compare.main([str(a), str(b)]) == 0
    assert capsys.readouterr().out == first
    assert first.splitlines()[-1].startswith("VERDICT: PASS")


def test_a_missing_manifest_is_reported_and_does_not_decide_the_verdict(tmp_path, capsys):
    a, b = paid_and_cached(tmp_path)
    (b / "manifest.json").unlink()
    assert compare.main([str(a), str(b)]) == 0
    printed = capsys.readouterr().out
    assert "manifest: missing in B" in printed and printed.splitlines()[-1].startswith("VERDICT: PASS")
    (a / "manifest.json").write_text("[]\n", encoding="utf-8")
    assert compare.main([str(a), str(b)]) == 0
    assert "manifest: unreadable in A, missing in B" in capsys.readouterr().out
    for run in (a, b):
        (run / "manifest.json").write_text(json.dumps(manifest(run="x", paid=1, cached=0, cost=0.1)), "utf-8")
    assert compare.main([str(a), str(b)]) == 0
    assert "manifest: keys that differ: none" in capsys.readouterr().out


def test_the_money_fields_are_those_m1_smoke_neutral_zeroes_plus_the_rewrite_cost():
    source = (REPO / "scripts" / "m1_smoke.py").read_text(encoding="utf-8")
    neutral = re.search(r"def neutral\(.*?dataclasses\.replace\(t, (.*?)\)\.to_json\(\)", source, re.DOTALL)
    assert neutral is not None
    zeroed = {assignment.split("=")[0].strip() for assignment in neutral.group(1).split(",")}
    assert zeroed == {"paid_calls", "cache_hits", "local_calls", "cost_rmb"}
    assert set(compare.MONEY_FIELDS) == zeroed | {"rewrite_cost_rmb"}
    # the money fields are dropped wherever they sit at the top level, and the transcript path keeps its name
    stripped = compare.neutral_row(row("r001", transcript="runs/x/transcripts/r001.json"))
    assert not set(stripped) & set(compare.MONEY_FIELDS) and stripped["transcript"] == "r001.json"
    assert compare.neutral_row(row("r001", transcript=None))["transcript"] is None
    assert set(compare.neutral_transcript(transcript("r001"))) == {
        "request", "today", "status", "final_text", "messages", "steps", "schema_hash", "prompt_version",
        "prompt_hash", "usage"}
    assert compare.neutral_transcript(transcript("r001", paid=2, cost=0.5)) == compare.neutral_transcript(
        transcript("r001", paid=0, cached=2, cost=0.0))


def test_compare_synth_runs_carries_no_machine_path():
    source = (REPO / "scripts" / "compare_synth_runs.py").read_text(encoding="utf-8")
    assert re.search(r"[A-Za-z]:\\", source) is None  # a drive letter: an absolute Windows path

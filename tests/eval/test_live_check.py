"""scripts/live_check.py offline, over a smoke test in miniature that scripts/ask.py itself ran: a probe, then
batch A (a lookup and a scaffold note) and batch B (the notes listed again, a follow-up, an unknown id) in
other processes' stead, with scripted models, arXiv behind httpx.MockTransport and one fake clock. Every
first-layer check passes on it; each check fails on the run spoiled the way it guards against, and a file a
check cannot read leaves the run unjudged (exit 2) rather than failed."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from support.fakes import ScriptedModel, call, turn
from support.live_fakes import FakeClock

from research_agent.eval.systems import Built
from research_agent.tools.adapters import paper_url

REPO = Path(__file__).resolve().parents[2]
TODAY = "2026-11-10"
A, B, UNKNOWN, OLD = "2611.00001", "2611.00002", "2510.99999", "cs/0112017"
SEARCH = "https://export.arxiv.org/api/query?search_query="
ENTRY = ("<entry><id>http://arxiv.org/abs/{id}v1</id><published>2026-11-02</published>"
         "<title>Paper {id}</title><summary>Speculative decoding drafts tokens. A verifier accepts some of "
         "them. The draft length adapts.</summary><author><name>A. Author</name></author>"
         '<category term="cs.CL"/></entry>')
OLD_STYLE = ("<entry><id>http://arxiv.org/abs/cs/0112017v1</id><published>2001-12-14</published>"
             "<title>An Old Paper</title><summary>Before the new identifiers.</summary></entry>")
REQUESTS = [
    {"id": "S03", "batch": "A", "request": f"{A} 是哪篇论文?", "expect": {"tools": ["get_paper"]}},
    {"id": "S06", "batch": "A", "request": f"给 {A} 写一份笔记,用 scaffold 模式。",
     "expect": {"tools": ["write_note"], "notes": 1}},
    {"id": "S08", "batch": "B", "request": "列出我最近写的笔记。", "expect": {"tools": ["list_notes"]}},
    {"id": "S09", "batch": "B", "request": "帮我找几篇论文。", "clarify_answer": "speculative decoding",
     "expect": {"tools": ["ask_clarification", "search_arxiv"]}},
    {"id": "S10", "batch": "B", "request": f"What paper is {UNKNOWN}? I need its title.",
     "expect": {"tools": ["get_paper"], "status_of_lookup": "NOT_FOUND"}},
]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ask = load_script("ask")
live_check = load_script("live_check")


def feed(*entries: str) -> str:
    return ('<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom">'
            "<title>ArXiv Query</title>" + "".join(entries) + "</feed>")


PAGES = {paper_url(A): feed(ENTRY.format(id=A)), paper_url(UNKNOWN): feed(), paper_url(OLD): feed(OLD_STYLE),
         SEARCH: feed(ENTRY.format(id=A), ENTRY.format(id=B))}


def arxiv(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    return httpx.Response(200, text=PAGES[SEARCH if url.startswith(SEARCH) else url])


BATCHES = {
    "A": [turn(call("get_paper", "g1", arxiv_id=A)), turn(content=f"这是 Paper {A}。"),
          turn(call("get_paper", "g2", arxiv_id=A)),
          turn(call("write_note", "w1", arxiv_id=A, mode="scaffold")),
          turn(content="笔记已写好。")],
    "B": [turn(call("list_notes", "l1")), turn(content=f"有一条笔记:{A}。"),
          turn(call("ask_clarification", "q1", question="哪个主题?", missing=["topic"])),
          turn(call("search_arxiv", "s1", query="speculative decoding")), turn(content=f"推荐 {A} 和 {B}。"),
          turn(call("get_paper", "g3", arxiv_id=UNKNOWN)), turn(content="没有找到这个编号。")],
}


def scripted(turns: list):
    def build(args, folder, out):
        return ask.Assembly(Built(ScriptedModel(list(turns)), None, {"model": "scripted"}),
                            {"total_slots": 1}, [],
                            {"url": "http://127.0.0.1:8080", "started_here": False, "command": None},
                            {"path": "results/final-m7.json", "sha256": "0" * 64})

    return build


@pytest.fixture
def smoke(tmp_path, monkeypatch) -> dict:
    """The miniature smoke test's run folders (probe, A, B), requests file, notes and a clean git tree; the
    runs record a clean working tree, as the real ones must (this repository's own may not be one while the
    tests run), and start at the repository root, as scripts/ask.py requires."""
    monkeypatch.setattr(ask, "git_dirty", lambda repo: False)
    monkeypatch.chdir(REPO)
    requests = tmp_path / "smoke.jsonl"
    requests.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in REQUESTS), encoding="utf-8")
    clock, runs = FakeClock(), tmp_path / "runs"
    common = ["--runs", str(runs), "--notes", str(tmp_path / "notes"), "--state", str(tmp_path / "state")]
    options = {"today": lambda: TODAY, "transport": httpx.MockTransport(arxiv), "clock": clock,
               "sleep": clock.sleep}
    assert ask.main(["--probe", UNKNOWN, OLD, *common], build=None, **options) == 0
    for name in ("A", "B"):
        code = ask.main(["--batch", str(requests), "--batch-name", name, *common],
                        build=scripted(BATCHES[name]), **options)
        assert code == 0
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    folders = sorted(path for path in runs.iterdir() if path.is_dir())
    assert len(folders) == 3
    return {"folders": folders, "requests": requests, "repo": repo, "notes": tmp_path / "notes"}


def run_check(smoke: dict, capsys, *extra: str) -> tuple[int, str]:
    code = live_check.main([*map(str, smoke["folders"]), "--requests", str(smoke["requests"]), *extra],
                           repo=smoke["repo"])
    return code, capsys.readouterr().out


def test_every_first_layer_check_passes_on_a_clean_smoke_test(smoke, capsys):
    code, out = run_check(smoke, capsys)
    assert code == 0, out
    assert [line.split(" ", 2)[:2] for line in out.splitlines() if line[:4] in ("PASS", "FAIL")] == [
        ["PASS", str(n)] for n in range(1, 8)]
    assert "6 turn(s) for 5 request(s)" in out and f"S06 note {A}-1: 3 evidence sentence(s)" in out
    assert "S09: clarified, then one user message '帮我找几篇论文。补充:speculative decoding'" in out
    assert f"probe {UNKNOWN}: NOT_FOUND" in out and "S10: get_paper gave NOT_FOUND" in out
    assert "export.arxiv.org: 7 request(s), least gap 3.000 s (at least 3)" in out
    assert "S06 turn-002" in out and "layer 1: every check passes" in out
    assert "S09: expected tools ['ask_clarification', 'search_arxiv']: all called" in out  # over both turns


def test_runs_of_the_same_second_are_taken_in_the_order_they_were_made():
    names = ["20261110T000001Z", "20261110T000000Z-10", "20261110T000000Z-2", "20261110T000000Z"]
    ordered = sorted((Path("runs") / name for name in names), key=live_check.run_order)
    assert [path.name for path in ordered] == ["20261110T000000Z", "20261110T000000Z-2",
                                               "20261110T000000Z-10", "20261110T000001Z"]


def test_a_missing_summary_and_a_dirty_tree_fail_the_files_check(smoke, capsys):
    (smoke["folders"][1] / "turn-002.md").unlink()
    (smoke["repo"] / "stray.txt").write_text("x", encoding="utf-8")
    code, out = run_check(smoke, capsys)
    assert code == 1 and "FAIL 7 files" in out and "no turn-002.md" in out and "1 path(s) changed" in out


def test_requests_closer_than_the_interval_fail_the_requests_check(smoke, capsys):
    log = smoke["folders"][2] / "requests.jsonl"
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    first = next(line for line in lines if line["event"] == "request")
    previous = max(json.loads(line)["t"] for line in (smoke["folders"][1] / "requests.jsonl")
                   .read_text(encoding="utf-8").splitlines() if json.loads(line)["event"] == "request")
    first["t"] = previous + 1.0  # one second after batch A's last request
    log.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    code, out = run_check(smoke, capsys)
    assert code == 1 and "FAIL 3 requests" in out and "export.arxiv.org: 1 gap(s) under 3 s" in out


@pytest.mark.parametrize("dirty", [True, None])
def test_a_run_not_known_to_be_on_a_clean_working_tree_fails_the_identity_check(smoke, capsys, dirty):
    session = smoke["folders"][1] / "session.json"
    data = json.loads(session.read_text(encoding="utf-8"))
    assert data["git_dirty"] is False
    session.write_text(json.dumps({**data, "git_dirty": dirty}), encoding="utf-8")
    code, out = run_check(smoke, capsys)
    assert code == 1 and "FAIL 1 identity" in out and f"{smoke['folders'][1].name}: git_dirty: " in out
    assert "PASS 2 statuses" in out  # the other checks are unaffected


def test_a_file_a_check_cannot_read_exits_2_and_the_other_checks_still_run(smoke, capsys):
    (smoke["folders"][1] / "turn-002.papers.json").write_text("[]", encoding="utf-8")  # not an object
    code, out = run_check(smoke, capsys)
    assert code == 2 and "FAIL 4 notes" in out and "cannot read the runs: AttributeError" in out
    assert "PASS 1 identity" in out and "PASS 7 files" in out and "Layer 2" in out
    assert out.rstrip().endswith("layer 1: not judged: a file cannot be read (notes)")
    (smoke["folders"][1] / "turn-002.papers.json").unlink()
    note = smoke["notes"] / f"{A}-1.md"
    kept = note.read_bytes()
    note.write_text("", encoding="utf-8")  # an empty note file
    code, out = run_check(smoke, capsys)
    assert code == 2 and "cannot read the runs: IndexError" in out
    note.write_bytes(kept)
    record = smoke["folders"][2] / "turn-001.json"
    record.write_text(json.dumps({**json.loads(record.read_text(encoding="utf-8")), "unknown": 1}),
                      encoding="utf-8")  # a key a transcript does not have
    code, out = run_check(smoke, capsys)
    assert code == 2 and "cannot read the runs: TypeError" in out


def test_a_harness_error_an_identity_problem_and_a_missing_note_fail_their_checks(smoke, capsys):
    index = smoke["folders"][0].parent / "index.jsonl"
    rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines()]
    rows[0]["status"] = "harness_error"
    index.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    session = smoke["folders"][2] / "session.json"
    data = json.loads(session.read_text(encoding="utf-8"))
    session.write_text(json.dumps({**data, "identity_problems": ["server: 4 slots (/props), not 1"]}),
                       encoding="utf-8")
    (smoke["notes"] / f"{A}-1.md").unlink()
    code, out = run_check(smoke, capsys)
    assert code == 1
    for failed in ("FAIL 1 identity", "FAIL 2 statuses", "FAIL 4 notes"):
        assert failed in out
    assert "server: 4 slots (/props), not 1" in out and "ended as harness_error" in out
    assert "the note file is not in the library" in out


def test_without_the_follow_up_and_the_probe_their_checks_fail(smoke, capsys):
    probe, batch_a, batch_b = smoke["folders"]
    code = live_check.main([str(batch_a), str(batch_b), "--requests", str(smoke["requests"])],
                           repo=smoke["repo"])
    out = capsys.readouterr().out
    assert code == 1 and "FAIL 6 unknown ids" in out and "no probe outcome" in out
    follow_up = json.loads((batch_b / "turn-003.json").read_text(encoding="utf-8"))
    follow_up["messages"][1]["content"] = "帮我找几篇论文。"
    (batch_b / "turn-003.json").write_text(json.dumps(follow_up, ensure_ascii=False), encoding="utf-8")
    code, out = run_check(smoke, capsys, "--probe-log", str(probe / "probe.json"))
    assert code == 1 and "FAIL 5 follow-up" in out and "not [system, user(packed)]" in out

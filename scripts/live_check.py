"""The first-layer checks of the live smoke test (docs/m9-计划.md WP3, 第一层判据), and its second layer for
the report, over the run folders scripts/ask.py wrote. Writes nothing.

    .venv/Scripts/python.exe scripts/live_check.py <run folder>... --requests configs/live-smoke-m9.jsonl \\
        [--notes DIR] [--probe-log FILE]

The run folders are the probe's and the batches' runs, in any order (they are taken in the order their names,
start stamps with a number for a second run in the same second, give). The requests file names every request
by its id, which the turns carry in their index lines (<runs>/index.jsonl, beside the run folders). --notes
is the notes library, by default the one each run's session.json names; --probe-log the probe's outcomes, by
default the probe.json of a run folder given.

Layer 1, each PASS or FAIL:
1. identity: every run of the system recorded no identity problem in its session.json and ran on a clean
   working tree (its git_dirty false: live.frozen.tree_problems); scripts/check_final_identity.py repeats the
   check against the frozen record;
2. statuses: every request ran, and every turn, follow-ups included, ended with a status of
   agent.loop.run_turn: none raised and none ended as harness_error;
3. requests: over the requests.jsonl of every run in time order, the requests to each host of
   net.ratelimit.DEFAULT_INTERVALS are at least its interval apart (3 s to export.arxiv.org, 15 s to
   arxiv.org), every request falls within a fetch and no two fetches overlap: at most one request in flight
   at a time;
4. notes: a request that expects notes ("expect": {"notes": N}) wrote exactly N; every evidence sentence of
   each is found in the abstract the note was made from (turn-NNN.papers.json, notes.locate_evidence); the
   list_notes or search_notes output of a later run names the note; the library holds it (read only);
5. follow-up: a request with a clarify_answer ended its first turn in a clarification, and the next turn's
   messages are the system prompt of its today and one user message, live.session.pack_follow_up(request,
   answer);
6. unknown ids: the probe gave NOT_FOUND for every new-style id and a lenient parse of [] for every old-style
   one, and its search had hits; a request that expects a status of its lookup got it from get_paper;
7. files: every turn has its .json, .md and index line, and git shows a clean working tree.

Layer 2, reported and not judged: per turn the tools called, the cited ids in no tool output of the turn,
the steps, the seconds, the reasoning tokens (median and largest over the answers), the unanimous votes and
the runtime rules that fired; per request whether its turns called every tool it expects.

Exit 0 when every check of layer 1 passes, 1 when one fails, 2 on a usage error or a file that cannot be read
(READ_ERRORS: one a check cannot read is that check's FAIL, the others still run, and the run is not judged).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from contextlib import closing
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from research_agent.agent.prompts import system_prompt
from research_agent.agent.types import HARNESS_ERROR, STATUSES, Transcript
from research_agent.live.frozen import tree_problems
from research_agent.live.session import INDEX_FILE, SESSION_FILE, notes_written, pack_follow_up, tool_results
from research_agent.live.world import REQUEST_LOG
from research_agent.net.ratelimit import DEFAULT_INTERVALS
from research_agent.normalize import normalize_arxiv_id
from research_agent.notes import DATABASE, import_entries, locate_evidence
from research_agent.tools.schemas import NOT_FOUND

REPO = Path(__file__).resolve().parents[1]
PROBE = "probe"  # a probe run's kind in its session.json
PROBE_FILE = "probe.json"
NOTE_LISTS = ("list_notes", "search_notes")
# What a file that is missing, not JSON or not of its shape raises in the checks (a key or an item missing, a
# value of another type, an empty note file, a library that is not a database): it cannot be read
READ_ERRORS = (OSError, ValueError, LookupError, TypeError, AttributeError, sqlite3.Error)


@dataclass
class Turn:
    """A turn of a run: its index line and its turn-NNN.json (None when missing)."""

    run: Path
    row: dict
    record: dict | None

    @property
    def stem(self) -> str:
        return f"turn-{self.row['turn']:03d}"

    @property
    def transcript(self) -> Transcript | None:
        if self.record is None or "messages" not in self.record:
            return None
        return Transcript(**self.record)


@dataclass
class Check:
    name: str
    passed: bool
    details: list[str]


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def read_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load(folders: list[Path]) -> tuple[list[dict], list[Turn]]:
    """The runs' session.json (each with its folder) and their turns, in order."""
    sessions, turns = [], []
    for folder in folders:
        sessions.append({**read_json(folder / SESSION_FILE), "folder": folder})
        for row in read_lines(folder.parent / INDEX_FILE):
            if row.get("run") == folder.name:
                path = folder / f"turn-{row['turn']:03d}.json"
                turns.append(Turn(folder, row, read_json(path) if path.exists() else None))
    turns.sort(key=lambda turn: (folders.index(turn.run), turn.row["turn"]))
    return sessions, turns


def by_request(turns: list[Turn]) -> dict[str, list[Turn]]:
    grouped: dict[str, list[Turn]] = {}
    for turn in turns:
        grouped.setdefault(turn.row.get("id"), []).append(turn)
    return grouped


# ---- layer 1 ----------------------------------------------------------------------------------------------


def check_identity(sessions: list[dict]) -> Check:
    runs = [s for s in sessions if s.get("kind") != PROBE]
    details = []
    for session in runs:
        name = session["folder"].name
        if session.get("setup_error"):
            details.append(f"{name}: not assembled: {session['setup_error']}")
        elif "identity_problems" not in session:
            details.append(f"{name}: no identity recorded")
        details += [f"{name}: {problem}" for problem in session.get("identity_problems") or []]
        details += [f"{name}: {problem}" for problem in tree_problems(session.get("git_dirty"))]
    if not runs:
        details.append("no run of the system among the folders")
    passed = not details
    return Check("identity", passed, details if not passed else [
        f"{len(runs)} run(s), identity recorded OK, each on a clean working tree"])


def check_statuses(requests: list[dict], turns: list[Turn]) -> Check:
    grouped, details = by_request(turns), []
    for request in requests:
        if request["id"] not in grouped:
            details.append(f"{request['id']}: no turn")
    for turn in turns:
        label = f"{turn.row.get('id')} {turn.stem} ({turn.run.name})"
        if turn.row.get("error"):
            details.append(f"{label} raised: {turn.row['error']}")
        elif turn.row.get("status") not in STATUSES:
            details.append(f"{label}: status {turn.row.get('status')!r}")
        elif turn.row["status"] == HARNESS_ERROR:
            steps = (turn.record or {}).get("steps") or []
            why = next((s["harness_error"] for s in reversed(steps) if "harness_error" in s), "")
            details.append(f"{label} ended as harness_error: {why}")
    passed = not details
    summary = f"{len(turns)} turn(s) for {len(requests)} request(s), each ended with a status of run_turn"
    return Check("statuses", passed, [summary] if passed else details)


def check_requests(folders: list[Path]) -> Check:
    events = [event for folder in folders for event in read_lines(folder / REQUEST_LOG)]
    sent = sorted((e for e in events if e.get("event") == "request"), key=lambda e: e["t"])
    fetches = sorted((e for e in events if e.get("event") == "fetch"), key=lambda e: e["t"])
    if not sent:
        return Check("requests", False, ["no request logged"])
    details, summary = [], []
    for host, interval in DEFAULT_INTERVALS.items():
        times = [e["t"] for e in sent if e.get("host") == host]
        gaps = [later - earlier for earlier, later in pairwise(times)]
        close = [gap for gap in gaps if gap < interval]
        if close:
            details.append(f"{host}: {len(close)} gap(s) under {interval:g} s, the least {min(close):.3f} s")
        summary.append(f"{host}: {len(times)} request(s)"
                       + (f", least gap {min(gaps):.3f} s (at least {interval:g})" if gaps else ""))
    for earlier, later in pairwise(fetches):
        if earlier["t_end"] > later["t"]:
            details.append(f"fetches overlap: {earlier['url']} ended after {later['url']} started")
    outside = [e for e in sent if not any(f["t"] <= e["t"] <= f["t_end"] for f in fetches)]
    if outside:
        details.append(f"{len(outside)} request(s) outside every fetch")
    summary.append(f"{len(fetches)} fetch(es), none overlapping, every request within one")
    return Check("requests", not details, details or summary)


def note_ids(body: object) -> set[str]:
    """The note ids a list_notes or search_notes output names."""
    items = body.get("items") if isinstance(body, dict) else body
    return {item["note_id"] for item in items or [] if isinstance(item, dict) and "note_id" in item}


def store_holds(root: Path, note_id: str) -> bool:
    """Whether the library at root has the note, read without writing."""
    database = root / DATABASE
    if not database.exists():
        return False
    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        return db.execute("SELECT 1 FROM notes WHERE note_id = ?", (note_id,)).fetchone() is not None


def check_notes(requests: list[dict], turns: list[Turn], sessions: list[dict], notes: Path | None,
                folders: list[Path]) -> Check:
    grouped, details, summary = by_request(turns), [], []
    roots = {s["folder"]: Path(s["notes"]) for s in sessions if s.get("notes")}
    for request in requests:
        wanted = (request.get("expect") or {}).get("notes")
        if not isinstance(wanted, int):
            continue
        written = [(turn, note) for turn in grouped.get(request["id"], []) if turn.transcript is not None
                   for note in notes_written(turn.transcript)]
        if len(written) != wanted:
            details.append(f"{request['id']}: {len(written)} note(s) written, {wanted} expected")
        for turn, note in written:
            root = notes or roots.get(turn.run)
            label = f"{request['id']} note {note['note_id']}"
            papers_file = turn.run / f"{turn.stem}.papers.json"
            paper = read_json(papers_file).get(note["arxiv_id"]) if papers_file.exists() else None
            if root is None or not (root / note["path"]).exists():
                details.append(f"{label}: the note file is not in the library ({root})")
                continue
            entry = import_entries((root / note["path"]).read_text(encoding="utf-8"))[0]
            sentences = entry["evidence_sentences"]
            abstract = (paper or {}).get("abstract") or ""
            missing = [s for s in sentences if locate_evidence(abstract, s) < 0]
            if paper is None:
                details.append(f"{label}: no record of the paper it was made from ({papers_file.name})")
            elif not sentences or missing:
                details.append(f"{label}: {len(missing)} of {len(sentences)} evidence sentence(s) not in the "
                               "abstract")
            later = [t for t in turns if folders.index(t.run) > folders.index(turn.run) and t.transcript
                     and any(name in NOTE_LISTS and note["note_id"] in note_ids(body)
                             for name, body, _ in tool_results(t.transcript))]
            if not later:
                details.append(f"{label}: no later run's list_notes or search_notes named it")
            if not store_holds(root, note["note_id"]):
                details.append(f"{label}: the library {root} does not hold it")
            summary.append(f"{label}: {len(sentences)} evidence sentence(s) in the abstract; listed again by "
                           + ", ".join(f"{t.row.get('id')} ({t.run.name})" for t in later))
    if not summary and not details:
        summary.append("no request expects notes")
    return Check("notes", not details, details or summary)


def check_follow_up(requests: list[dict], turns: list[Turn]) -> Check:
    grouped, details, summary = by_request(turns), [], []
    for request in requests:
        answers = request.get("clarify_answer")
        answer = answers if isinstance(answers, str) else (answers or [None])[0]
        if answer is None:
            continue
        mine = grouped.get(request["id"], [])
        if not mine or mine[0].row.get("status") != "clarification":
            details.append(f"{request['id']}: the first turn ended as "
                           f"{mine[0].row.get('status') if mine else 'nothing'}, not in a clarification")
            continue
        if len(mine) < 2 or mine[1].record is None:
            details.append(f"{request['id']}: no follow-up turn")
            continue
        packed = pack_follow_up(request["request"], answer)
        record = mine[1].record
        wanted = [{"role": "system", "content": system_prompt(record.get("today"))},
                  {"role": "user", "content": packed}]
        head = (record.get("messages") or [])[:2]
        users = [m for m in record.get("messages") or [] if m.get("role") == "user"]
        if head != wanted or len(users) != 1:
            details.append(f"{request['id']}: the follow-up turn's messages are not [system, user(packed)]")
        else:
            summary.append(f"{request['id']}: clarified, then one user message {packed!r}")
    if not summary and not details:
        summary.append("no request has a clarify_answer")
    return Check("follow-up", not details, details or summary)


def check_unknown_ids(requests: list[dict], turns: list[Turn], probe: dict | None) -> Check:
    details, summary = [], []
    if probe is None:
        details.append("no probe outcome (probe.json)")
    else:
        for record in probe.get("ids") or []:
            arxiv_id = record.get("id")
            if normalize_arxiv_id(arxiv_id) == arxiv_id:
                if record.get("error") != NOT_FOUND or record.get("found") is not None:
                    details.append(f"probe {arxiv_id}: {record.get('error') or record.get('found')}, not "
                                   f"{NOT_FOUND}")
                else:
                    summary.append(f"probe {arxiv_id}: {NOT_FOUND}")
            elif record.get("lenient_parse") != []:
                details.append(f"probe {arxiv_id}: the lenient parse gave {record.get('lenient_parse')!r}, "
                               "not []")
            else:
                summary.append(f"probe {arxiv_id}: lenient parse [] (default parse "
                               f"{record.get('default_parse')!r})")
        search = probe.get("search") or {}
        if search.get("error") is not None or not search.get("hits"):
            details.append(f"probe search: {search.get('error') or 'no hit'}")
        else:
            summary.append(f"probe search: {len(search['hits'])} hit(s)")
    grouped = by_request(turns)
    for request in requests:
        wanted = (request.get("expect") or {}).get("status_of_lookup")
        if wanted is None:
            continue
        codes = [error for turn in grouped.get(request["id"], []) if turn.transcript is not None
                 for name, _, error in tool_results(turn.transcript) if name == "get_paper"]
        if wanted not in codes:
            details.append(f"{request['id']}: get_paper gave {codes}, not {wanted}")
        else:
            summary.append(f"{request['id']}: get_paper gave {wanted}")
    return Check("unknown ids", not details, details or summary)


def check_files(turns: list[Turn], folders: list[Path], repo: Path) -> Check:
    details = []
    for turn in turns:
        for suffix in (".json", ".md"):
            if not (turn.run / f"{turn.stem}{suffix}").exists():
                details.append(f"{turn.run.name}: no {turn.stem}{suffix}")
    for folder in folders:
        indexed = {turn.stem for turn in turns if turn.run == folder}
        for path in sorted(folder.glob("turn-*.json")):
            if not path.name.endswith(".papers.json") and path.stem not in indexed:
                details.append(f"{folder.name}: {path.name} has no index line")
    # --no-optional-locks: git does not refresh its index, so the check writes nothing there either
    status = subprocess.run(["git", "--no-optional-locks", "status", "--porcelain"], cwd=repo,
                            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if status.returncode != 0 or status.stdout.strip():
        changed = len(status.stdout.splitlines()) if status.returncode == 0 else "unknown"
        details.append(f"the working tree is not clean: {changed} path(s) changed")
    summary = f"{len(turns)} turn(s) with .json, .md and an index line; the working tree is clean"
    return Check("files", not details, details or [summary])


# ---- layer 2 ----------------------------------------------------------------------------------------------


def called_tools(transcript: Transcript) -> list[str]:
    """The tools the turn's accepted steps called, in order."""
    return [(call.get("function") or {}).get("name") for message in transcript.messages
            if message.get("role") == "assistant" for call in message.get("tool_calls") or []]


def layer_two(requests: list[dict], turns: list[Turn]) -> list[str]:
    lines = []
    for turn in turns:
        transcript, row = turn.transcript, turn.row
        if transcript is None:
            lines.append(f"{row.get('id')} {turn.stem}: raised")
            continue
        attempts = [attempt for step in transcript.steps for attempt in step.get("attempts") or []]
        thinking = [a["thinking"]["reasoning_tokens"] for a in attempts
                    if isinstance(a.get("thinking"), dict)
                    and isinstance(a["thinking"].get("reasoning_tokens"), int)]
        voted = [a["votes"] for a in attempts if isinstance(a.get("votes"), dict)]
        unanimous = sum(1 for v in voted if v.get("chosen") is not None
                        and (v.get("counts") or {}).get(v["chosen"]) == v.get("k"))
        rules = Counter(change.get("rule") for a in attempts for change in a.get("rules") or [])
        lines.append(
            f"{row.get('id')} {turn.stem} ({turn.run.name}): {row.get('status')}, "
            f"{len(transcript.steps)} step(s), {row.get('seconds', 0):.1f} s; tools "
            f"{called_tools(transcript) or 'none'}; cited {len(row.get('cited') or [])}, in no tool output "
            f"{row.get('cited_unseen') or []}; "
            + (f"reasoning tokens median {statistics.median(thinking):g}, largest {max(thinking)}; "
               if thinking else "")
            + f"unanimous votes {unanimous} of {len(voted)}; rules {dict(rules) or 'none'}")
    grouped = by_request(turns)
    for request in requests:
        wanted = list((request.get("expect") or {}).get("tools") or [])
        called = {name for turn in grouped.get(request["id"], []) if turn.transcript is not None
                  for name in called_tools(turn.transcript)}
        missing = [name for name in wanted if name not in called]
        lines.append(f"{request['id']}: expected tools {wanted}: "
                     + (f"not called {missing}" if missing else "all called"))
    return lines


def run_order(folder: Path) -> tuple[str, int]:
    """A run folder's place in time: its start stamp, then the number new_run_folder added to it."""
    stamp, _, number = folder.name.partition("-")
    return stamp, int(number) if number.isdigit() else 1


# ---- main -------------------------------------------------------------------------------------------------


def find_probe(folders: list[Path], given: Path | None) -> dict | None:
    if given is not None:
        return read_json(given)
    for folder in folders:
        if (folder / PROBE_FILE).exists():
            return read_json(folder / PROBE_FILE)
    return None


def main(argv: list[str] | None = None, *, repo: Path = REPO) -> int:
    """repo is the working tree whose cleanliness check 7 asks git about (the tests' own)."""
    ap = argparse.ArgumentParser(description="The first-layer checks of the live smoke test.")
    ap.add_argument("runs", type=Path, nargs="+", help="the run folders of the probe and the batches")
    ap.add_argument("--requests", type=Path, required=True, help="configs/live-smoke-m9.jsonl")
    ap.add_argument("--notes", type=Path, default=None, help="the notes library (default: each run's)")
    ap.add_argument("--probe-log", type=Path, default=None,
                    help="the probe's probe.json (default: found among the run folders)")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    folders = sorted(args.runs, key=run_order)
    try:
        requests = read_lines(args.requests)
        sessions, turns = load(folders)
        probe = find_probe(folders, args.probe_log)
    except READ_ERRORS as exc:
        print(f"FAIL: cannot read the runs: {type(exc).__name__}: {exc}")
        return 2
    unreadable: list[str] = []

    def judged(name: str, check: Callable[[], Check]) -> Check:
        """check(), or its FAIL when a file it reads cannot be read (READ_ERRORS)."""
        try:
            return check()
        except READ_ERRORS as exc:
            unreadable.append(name)
            return Check(name, False, [f"cannot read the runs: {type(exc).__name__}: {exc}"])

    checks = [judged("identity", lambda: check_identity(sessions)),
              judged("statuses", lambda: check_statuses(requests, turns)),
              judged("requests", lambda: check_requests(folders)),
              judged("notes", lambda: check_notes(requests, turns, sessions, args.notes, folders)),
              judged("follow-up", lambda: check_follow_up(requests, turns)),
              judged("unknown ids", lambda: check_unknown_ids(requests, turns, probe)),
              judged("files", lambda: check_files(turns, folders, repo))]
    print("Layer 1")
    for number, check in enumerate(checks, 1):
        print(f"{'PASS' if check.passed else 'FAIL'} {number} {check.name}")
        for detail in check.details:
            print(f"    {detail}")
    print("Layer 2 (reported, not judged)")
    try:
        lines = layer_two(requests, turns)
    except READ_ERRORS as exc:
        unreadable.append("layer 2")
        lines = [f"cannot read the runs: {type(exc).__name__}: {exc}"]
    for line in lines:
        print(f"    {line}")
    if unreadable:
        print(f"layer 1: not judged: a file cannot be read ({', '.join(unreadable)})")
        return 2
    passed = all(check.passed for check in checks)
    print(f"layer 1: {'every check passes' if passed else 'not usable: a check fails'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())

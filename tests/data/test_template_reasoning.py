"""scripts/template_reasoning.py: FT4's reasoning sidecars (docs/m7-本地提升计划.md WP6, 思考侧车).

A small world: records as M6 and M7 left them, with teacher and rationalized sidecar rows for their turns, the
FT4 records and changes that scripts/transform_synth.py's apply_ft4 makes from them, and a training manifest
that lists the sidecars, as runs/train/ft3/manifest.json does. No data file of the repository is read.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_synth import load_script

from research_agent.agent.prompts import PROMPT_HASH, system_prompt
from research_agent.canonical import canonical_json
from research_agent.data import reasoning
from research_agent.data import requests as rq
from research_agent.data.reasoning import ReasoningError, ReasoningIndex, answer_sha256
from research_agent.tools import schemas

REPO = Path(__file__).resolve().parents[2]
TODAY = "2026-10-10"
TRAIN_TOPICS = rq.split_topics(rq.load_corpus(REPO / "data" / "corpus" / "manifest.json"), 42)["train"]
tr = load_script("template_reasoning")
ts = load_script("transform_synth")

NOTES = [{"arxiv_id": "2610.00814", "created": "2026-10-09", "note_id": "2610.00814-1",
          "tags": ["cs.LG", "pretraining-data"], "title": "Training-Aware Target Coverage"},
         {"arxiv_id": "2609.32831", "created": "2026-09-30", "note_id": "2609.32831-1",
          "tags": ["cs.LG", "kv-cache"], "title": "UniCache: Type-Aware KV Cache Compression"}]
NOTE_HITS = [{"arxiv_id": "2609.32831", "note_id": "2609.32831-1", "score": 1.2, "snippet": "KV cache",
              "title": "UniCache: Type-Aware KV Cache Compression"}]
HITS = [{"arxiv_id": "2609.01234", "published": "2026-09-02", "snippet": "Draft lengths",
         "title": "Adaptive Draft Length for Speculative Decoding"}]
PAPER = {"abstract": "We study draft lengths.", "arxiv_id": "2609.01234", "authors": ["A. Author"],
         "published": "2026-09-02", "title": "Adaptive Draft Length for Speculative Decoding"}
RATE_LIMITED = {"error": "RATE_LIMITED", "items": []}


def call(name: str, args: dict, cid: str) -> dict:
    return {"function": {"arguments": json.dumps(args, ensure_ascii=False), "name": name}, "id": cid,
            "type": "function"}


def step_msgs(*pairs: tuple[dict, object], text: str = "") -> list[dict]:
    message = {"content": text, "role": "assistant", "tool_calls": [c for c, _ in pairs]}
    return [message, *({"content": canonical_json(r), "role": "tool", "tool_call_id": c["id"]}
                       for c, r in pairs)]


def one(name: str, args: dict, result: object, cid: str = "a") -> list[dict]:
    return step_msgs((call(name, args, cid), result))


def text_msg(text: str) -> list[dict]:
    return [{"content": text, "role": "assistant"}]


def history_record(rid: str, *steps: list[dict], scenario: str = "normal", intent: str = "search_notes",
                   lang: str = "zh", user: str = "帮我找论文", family: str = "x/zh/1") -> dict:
    messages = [{"content": system_prompt(TODAY), "role": "system"}, {"content": user, "role": "user"}]
    for s in steps:
        messages.extend(s)
    return {"family": family, "history": "agent", "id": rid, "intent": intent, "lang": lang,
            "messages": messages, "prompt_hash": PROMPT_HASH, "run": "train", "scenario": scenario,
            "schema_hash": schemas.SCHEMA_HASH, "tools": schemas.MODEL_TOOLS}


def request_of(rec: dict, **slots) -> dict:
    return {"family": rec["family"], "id": rec["id"], "intent": rec["intent"], "lang": rec["lang"],
            "scenario": rec["scenario"], "slots": slots, "split": "train",
            "text": rec["messages"][1]["content"]}


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(canonical_json(r) + "\n" for r in rows).encode("utf-8"))
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def row(rid: str, k: int, answer: dict, source: str = "teacher") -> dict:
    return reasoning.sidecar_row(rid, k, answer, source=source,
                                 reasoning=f"I think, so I call it ({rid} {k}).", model="flash",
                                 cache_key=hashlib.sha256(f"{rid}/{k}".encode()).hexdigest(),
                                 context_equal=True)


def originals() -> tuple[list[dict], list[dict]]:
    """Four records as M6 left them, and their requests:
    req-train-0101: an R8 retry at step 2 (steps 3 and 4 vanish); req-train-0102: no rule applies;
    req-train-0103: R11 at step 1 (an ask derived at step 2, R10); req-train-0104: R13 at step 1."""
    a = history_record("req-train-0101", one("search_arxiv", {"query": "speculative decoding"}, RATE_LIMITED),
                       one("search_arxiv", {"query": "speculative decoding LLM"}, HITS, "b"),
                       one("get_paper", {"arxiv_id": "2609.01234"}, PAPER, "c"),
                       text_msg("这篇讲自适应草稿长度。"), intent="find_recent_papers")
    b = history_record("req-train-0102", one("search_arxiv", {"query": "KV cache"}, HITS),
                       text_msg("找到一篇。"), intent="find_recent_papers")
    asked = {"question": "哪个方向?", "missing": ["topic"]}
    c = history_record("req-train-0103", one("list_notes", {}, NOTES),
                       one("ask_clarification", asked, {"question": "哪个方向?", "status": "asked"}, "q"),
                       scenario="clarify", intent="find_recent_papers")
    d = history_record("req-train-0104", one("search_notes", {"query": "量化"}, NOTE_HITS),
                       text_msg("找到一条。"))
    reqs = [request_of(a, topic="speculative decoding", topic_id="speculative-decoding"),
            request_of(b, topic="KV cache", topic_id="kv-cache"), request_of(c, missing=["topic"]),
            request_of(d, topic="quantization", topic_id="quantization")]
    return [a, b, c, d], reqs


def late_clarification(rid: str) -> dict:
    """A clarify record R4 drops (its first ask is at step 4) that saw the full listing at step 2."""
    return history_record(rid, one("search_notes", {"query": "advisor paper"}, []),
                          one("list_notes", {}, NOTES, "b"), one("search_notes", {"query": "lab"}, [], "c"),
                          one("ask_clarification", {"question": "Which paper?", "missing": ["paper_id"]},
                              {"question": "Which paper?", "status": "asked"}, "q"),
                          scenario="clarify", intent="summarize_paper", lang="en")


def world(tmp_path: Path, *, dropped: bool = False) -> dict:
    """The originals, their FT4 records and changes, three sidecars (teacher rows for every turn but one,
    a rationalized row for that one, and replay rows of another file's records) and the training manifest.
    With dropped, a record R4 dropped (so not among the originals) gives a derived record too."""
    records, reqs = originals()
    late = [late_clarification("req-train-0105")] if dropped else []
    reqs += [request_of(r, missing=["paper_id"]) for r in late]
    ft4 = ts.apply_ft4(records, {r["id"]: r for r in reqs}, topics=TRAIN_TOPICS, dropped=late)
    teacher, rationalized = [], []
    for rec in records:
        for k, _, answer in reasoning.turns(rec["messages"]):
            if (rec["id"], k) == ("req-train-0102", 1):
                rationalized.append(row(rec["id"], k, answer, "rationalized"))
            else:
                teacher.append(row(rec["id"], k, answer))
    replay = [row("base-replay-c5:0", 0, {"role": "assistant", "content": "Cannot call it yet."})]
    paths = {"originals": write_jsonl(tmp_path / "synth.jsonl", records),
             "records": write_jsonl(tmp_path / "synth-ft4.jsonl",
                                    sorted(ft4.records + ft4.derived, key=lambda r: r["id"])),
             "changes": write_jsonl(tmp_path / "ft4-changes.jsonl", ft4.changes),
             "teacher": write_jsonl(tmp_path / "reasoning-teacher.jsonl", teacher),
             "rationalized": write_jsonl(tmp_path / "reasoning-rationalized.jsonl", rationalized),
             "replay": write_jsonl(tmp_path / "reasoning-replay.jsonl", replay),
             "out": tmp_path / "out", "report": tmp_path / "out" / "reasoning-ft4.report.json"}
    manifest = {"config": {"data": {"reasoning": {"files": [
        {"path": paths[name].as_posix(), "rows": 1} for name in ("teacher", "rationalized", "replay")]}}}}
    paths["manifest"] = tmp_path / "manifest.json"
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    paths["argv"] = ["--manifest", str(paths["manifest"]), "--records", str(paths["records"]),
                     "--originals", str(paths["originals"]), "--changes", str(paths["changes"]),
                     "--out-dir", str(paths["out"]), "--report", str(paths["report"])]
    paths["ft4"] = ft4
    return paths


def keys(path: Path) -> list[tuple[str, int]]:
    return [(r["id"], r["k"]) for r in read_jsonl(path)]


def test_the_ft4_sidecars_keep_unchanged_turns_drop_changed_and_vanished_ones_and_rekey_copies(tmp_path):
    w = world(tmp_path)
    assert tr.main(w["argv"]) == 0
    teacher = read_jsonl(w["out"] / "reasoning-teacher-ft4.jsonl")
    # 0101: turn 1 relabelled (R8), 2 and 3 vanished; 0103: turn 0 relabelled (R11), 1 vanished, both copied
    # into the derived record, turn 1 as its target (the teacher's own ask); 0104: turn 0 relabelled (R13),
    # turn 1 kept
    assert [(r["id"], r["k"]) for r in teacher] == [
        ("req-train-0101", 0), ("req-train-0102", 0), ("req-train-0103-ft4r10k1", 0),
        ("req-train-0103-ft4r10k1", 1), ("req-train-0104", 1)]
    old = {(r["id"], r["k"]): r for r in read_jsonl(w["teacher"])}
    assert teacher[0] == old["req-train-0101", 0] and teacher[1] == old["req-train-0102", 0]
    assert teacher[2] == {**old["req-train-0103", 0], "id": "req-train-0103-ft4r10k1"}  # re-keyed, as it was
    assert teacher[3] == {**old["req-train-0103", 1], "id": "req-train-0103-ft4r10k1"}  # the target's own row
    assert teacher[4] == {**old["req-train-0104", 1], "context_equal": False}  # the R13 query is before it
    assert keys(w["out"] / "reasoning-rationalized-ft4.jsonl") == [("req-train-0102", 1)]
    assert not (w["out"] / "reasoning-replay-ft4.jsonl").exists()  # another file's turns: not rewritten


def test_the_template_sidecar_holds_one_checked_row_per_changed_or_new_target_turn(tmp_path):
    w = world(tmp_path)
    assert tr.main(w["argv"]) == 0
    template = read_jsonl(w["out"] / "reasoning-template-ft4.jsonl")
    targets = sorted((c["id"], c["k"]) for c in w["ft4"].changes if c["target"] and c["kind"] != "copy")
    assert [(r["id"], r["k"]) for r in template] == targets == [
        ("req-train-0101", 1), ("req-train-0103", 0), ("req-train-0104", 0)]  # no row for the teacher's ask
    records = {r["id"]: r for r in read_jsonl(w["records"])}
    for r in template:
        assert reasoning.check_row(r, "a template row") == r
        assert r["source"] == "template" and r["model"] == tr.MODEL and r["context_equal"] is True
        answer = reasoning.turns(records[r["id"]]["messages"])[r["k"]][2]
        assert r["answer_sha256"] == answer_sha256(answer)
        assert tr.problems(r["reasoning"], answer) == []
    text = {(r["id"], r["k"]): r["reasoning"] for r in template}
    assert "RATE_LIMITED" in text["req-train-0101", 1] and "search_arxiv" in text["req-train-0101", 1]
    assert "ask_clarification" in text["req-train-0103", 0]
    assert "quantization" in text["req-train-0104", 0] and "search_notes" in text["req-train-0104", 0]


def test_every_ft4_sidecar_loads_and_binds_to_the_ft4_records(tmp_path):
    w = world(tmp_path)
    assert tr.main(w["argv"]) == 0
    index = ReasoningIndex([*sorted(w["out"].glob("reasoning-*-ft4.jsonl")), w["replay"]])
    found = 0
    for rec in read_jsonl(w["records"]):
        for k, _, answer in reasoning.turns(rec["messages"]):
            found += index.lookup(rec["id"], k, answer) is not None
    assert found == 9  # every turn of the FT4 records has reasoning here, the copied one included
    report = json.loads(w["report"].read_text(encoding="utf-8"))
    teacher = next(s for s in report["sidecars"] if s["path"] == w["teacher"].as_posix())
    assert teacher | {"sha256": None} == {
        "path": w["teacher"].as_posix(), "out": (w["out"] / "reasoning-teacher-ft4.jsonl").as_posix(),
        "rows_in": 9, "kept": 3, "dropped_changed": 3, "dropped_vanished": 3, "rekeyed": 2, "rows_out": 5,
        "context_changed": 1, "sha256": None}
    assert report["untouched"] == [{"path": w["replay"].as_posix(), "rows": 1}]
    assert report["template"]["rows"] == 3 and report["template"]["by_rule"] == {"R8": 1, "R11": 1, "R13": 1}
    assert report["verification"] == {"turns": 9, "context_turns": 1, "copy_targets": 1,
                                      "copy_targets_with_reasoning": 1, "with_reasoning": 9, "index_rows": 10}


def test_a_kept_turn_whose_answer_changed_without_a_change_row_stops_the_script(tmp_path, capsys):
    w = world(tmp_path)
    records = read_jsonl(w["records"])
    for rec in records:
        if rec["id"] == "req-train-0102":
            rec["messages"][-1]["content"] = "改过的回答。"
    write_jsonl(w["records"], records)
    assert tr.main(w["argv"]) == 1
    assert not w["out"].exists()
    assert "req-train-0102" in capsys.readouterr().err


def test_a_record_cut_from_one_r4_dropped_gets_its_template_row_and_no_copied_rows(tmp_path):
    """R4's dropped records are not among the originals (nor in any sidecar): the derived record's copied
    turns name them as their source and get no row; its last turn gets its template row."""
    w = world(tmp_path, dropped=True)
    derived = "req-train-0105-ft4r10k2"
    assert {(c["id"], c["k"], c["kind"]) for c in w["ft4"].changes if c["id"] == derived} == {
        (derived, 0, "copy"), (derived, 1, "copy"), (derived, 2, "new")}
    assert tr.main(w["argv"]) == 0
    template = {(r["id"], r["k"]) for r in read_jsonl(w["out"] / "reasoning-template-ft4.jsonl")}
    assert (derived, 2) in template and (derived, 0) not in template
    assert all(r["id"] != derived for r in read_jsonl(w["out"] / "reasoning-teacher-ft4.jsonl"))
    report = json.loads(w["report"].read_text(encoding="utf-8"))
    assert report["verification"]["context_turns"] == 3 and report["changes"] == {
        "rows": len(w["ft4"].changes), "sources_outside_the_originals": 3}


def test_a_changes_row_with_a_source_that_is_not_its_record_stops_the_script(tmp_path):
    w = world(tmp_path)
    changes = read_jsonl(w["changes"])
    bad = next(c for c in changes if c["kind"] == "copy")
    write_jsonl(w["changes"], [{**c, "from": ["req-train-0999", 0]} if c is bad else c for c in changes])
    assert tr.main(w["argv"]) == 1
    assert not w["out"].exists()


def test_a_changes_row_for_a_turn_the_records_lack_stops_the_script(tmp_path):
    w = world(tmp_path)
    changes = read_jsonl(w["changes"])
    write_jsonl(w["changes"], [*changes, {"id": "req-train-0102", "k": 7, "rule": "R8", "kind": "relabel",
                                          "target": True, "from": ["req-train-0102", 7]}])
    assert tr.main(w["argv"]) == 1
    assert not w["out"].exists()


def test_a_relabel_or_new_row_that_is_no_target_stops_the_script(tmp_path):
    w = world(tmp_path)
    changes = read_jsonl(w["changes"])
    write_jsonl(w["changes"], [{**c, "target": False} if c["kind"] == "relabel" else c for c in changes])
    assert tr.main(w["argv"]) == 1
    assert not w["out"].exists()


def test_context_turns_that_disagree_with_the_changes_rows_stop_the_script(tmp_path, capsys):
    w = world(tmp_path)
    records = read_jsonl(w["records"])
    for rec in records:
        if rec["id"] == "req-train-0103-ft4r10k1":
            rec.pop("context_turns")
    write_jsonl(w["records"], records)
    assert tr.main(w["argv"]) == 1
    assert not w["out"].exists() and "context_turns" in capsys.readouterr().err


def test_reruns_byte_identical_and_keep_the_outputs_without_force(tmp_path):
    w = world(tmp_path)
    assert tr.main(w["argv"]) == 0
    first = {p.name: p.read_bytes() for p in w["out"].iterdir()}
    assert tr.main(w["argv"]) == 2
    assert tr.main([*w["argv"], "--force"]) == 0
    assert {p.name: p.read_bytes() for p in w["out"].iterdir()} == first


def turn_answer(rec: dict, k: int) -> dict:
    return reasoning.turns(rec["messages"])[k][2]


def turns_of(rec: dict) -> list[dict]:
    return [m for m in rec["messages"] if m["role"] == "assistant"]


KV_NOTE = {"arxiv_id": "2607.00760", "created": "2026-09-07", "note_id": "2607.00760-1",
           "tags": ["cs.LG", "kv-cache"], "title": "MosaicKV: Dynamic Two-D KV Cache Compression"}


def relabelled(variant: str) -> tuple[dict, int, str, dict]:
    """An input record whose turn k an in-place rule relabels, the rule and the request's slots."""
    notes_first = [one("list_notes", {"tag": "survey"}, []), one("list_notes", {}, NOTES, "b")]
    if variant == "s4a_list":
        return history_record("req-train-0201", *notes_first,
                              one("search_notes", {"query": "survey"}, [], "c"), text_msg("没有。"),
                              intent="list_notes"), 2, "R9", {}
    if variant in ("s4a_pick", "s4a_pick_ask"):
        listing = notes_first if variant == "s4a_pick" else [one("list_notes", {}, [*NOTES, KV_NOTE], "b")]
        rec = history_record("req-train-0202", *listing, one("search_notes", {"query": "x"}, [], "c"),
                             text_msg("没有。"), family="search_notes/zh/3")
        return rec, len(listing), "R9", {"topic": "KV cache", "topic_id": "kv-cache"}
    if variant == "s4b":
        searches = [one("search_arxiv", {"query": f"q{i}"}, HITS, f"s{i}") for i in range(4)]
        return history_record("req-train-0203", *searches, text_msg("好的。")), 3, "R9", {}
    return history_record("req-train-1385", *notes_first, text_msg("没有这个标签。你想用哪个标签来筛选?"),
                          intent="list_notes"), 2, "R10", {}


def built(variant: str) -> tuple[dict, str, dict]:
    """A new record of the full run, its rule and its request."""
    corpus = rq.load_corpus(REPO / "data" / "corpus" / "manifest.json")
    guard = ts.EvalGuard(eval_texts=[], train_texts=[])
    if variant == "relative":
        (req,) = ts.generate_relative_requests(1, seed=42, corpus=corpus, split_seed=42, today=TODAY,
                                               guard=guard)
        return ts.relative_date_record({"id": "req-ft4-0001", **req}, TODAY), "R11", req
    if variant == "defref":
        (req,) = ts.generate_defref_requests(1, seed=42, corpus=corpus, split_seed=42, guard=guard)
        return ts.defref_record({"id": "req-ft4-0002", **req}, TODAY), "R14", req
    if variant == "explain":
        (req,) = ts.generate_explain_requests(1, seed=42, guard=guard)
        return ts.explain_record({"id": "req-ft4-0003", **req}, TODAY), "R14", req
    made, _ = ts.glossary_requests({"注意力汇聚": "attention sink"}, existing={}, corpus=corpus, held_out=(),
                                   limit=1, guard=guard)
    return ts.glossary_record({"id": "req-ft4-0004", **made[0]}, TODAY), "R13", made[0]


@pytest.mark.parametrize("variant", ["s4a_list", "s4a_pick", "s4a_pick_ask", "s4b", "r10"])
def test_the_templates_of_the_relabels_pass_their_checks(variant):
    rec, k, rule, slots = relabelled(variant)
    out = ts.apply_ft4([rec], {rec["id"]: request_of(rec, **slots)}, topics=TRAIN_TOPICS)
    relabels = [(c["id"], c["k"], c["rule"]) for c in out.changes if c["kind"] == "relabel"]
    assert relabels == [(rec["id"], k, rule)]
    for derived in out.derived:  # the closing turns R9 constructs where the cut removed a listing point
        last = len(turns_of(derived)) - 1
        assert tr.problems(tr.template_reasoning(derived, last, rule), turn_answer(derived, last)) == []
    text = tr.template_reasoning(out.records[0], k, rule)
    assert tr.problems(text, turn_answer(out.records[0], k)) == []
    assert "full list" not in text  # the listing may not be all the notes (it says so where it is not)
    if variant == "s4a_pick_ask":
        assert "ask_clarification" in text and "the tag of the note" not in text


@pytest.mark.parametrize("variant", ["relative", "defref", "explain", "glossary"])
def test_the_templates_of_the_new_records_pass_their_checks(variant):
    rec, rule, req = built(variant)
    text = tr.template_reasoning(rec, 0, rule)
    assert tr.problems(text, turn_answer(rec, 0)) == []
    if variant == "relative":
        start, _ = ts.relative_range(req["slots"]["relative"], TODAY)
        assert TODAY in text and start in text


def test_the_r13_template_names_the_term_of_the_rewritten_query_among_several_note_searches():
    rec = history_record("req-train-0106", step_msgs(
        (call("search_notes", {"query": "scaling laws"}, "a"), []),
        (call("search_notes", {"query": "缩放"}, "b"), [])), text_msg("没有。"))
    out = ts.apply_ft4([rec], {rec["id"]: request_of(rec, topic="scaling law", topic_id="scaling-laws")},
                       topics=TRAIN_TOPICS)
    assert [(c["k"], c["rule"]) for c in out.changes] == [(0, "R13")]
    text = tr.template_reasoning(out.records[0], 0, "R13")
    assert "standard English term is scaling law" in text
    # NoteStore requires every term of a query (an AND), so the mixed query cannot find more than the English
    # term alone: the template must not suggest that it widens the search
    assert not any(claim in text for claim in ("may be written in English", "widen", "more notes", "so the"))
    assert tr.problems(text, turn_answer(out.records[0], 0)) == []


def test_the_relative_date_template_names_the_sort_order_only_when_the_call_sets_one():
    """R11's searches leave sort_by out when the teacher's majority is the default (transform_synth's
    relative_sort_by): the reasoning then names no order, as the call sets none."""
    corpus = rq.load_corpus(REPO / "data" / "corpus" / "manifest.json")
    guard = ts.EvalGuard(eval_texts=[], train_texts=[])
    made = ts.generate_relative_requests(8, seed=42, corpus=corpus, split_seed=42, today=TODAY, guard=guard)
    req = next(r for r in made if r["intent"] == "find_recent_papers")
    rec = ts.relative_date_record({"id": "req-ft4-0001", **req}, TODAY, sort_by="submittedDate")
    text = tr.template_reasoning(rec, 0, "R11")
    assert "sort_by=submittedDate" in text and tr.problems(text, turn_answer(rec, 0)) == []
    plain = ts.relative_date_record({"id": "req-ft4-0001", **req}, TODAY)
    assert "sort_by" not in ts._args(turn_answer(plain, 0)["tool_calls"][0])
    text = tr.template_reasoning(plain, 0, "R11")
    assert "sort_by" not in text and f"date_from={req['slots']['date_from']}" in text
    assert tr.problems(text, turn_answer(plain, 0)) == []


@pytest.mark.parametrize(("text", "problem"), [
    ("我会调用 search_arxiv。The defaults are fine.", "not English"),
    ("I call search_arxiv.", "not 2 to 4 sentences"),
    ("One. Two. Three. Four. I call search_arxiv now.", "not 2 to 4 sentences"),
    ("The search failed for a moment. I will repeat the call.", "a called tool is not named"),
    ("The search failed. I will call search_arxiv </think> again.", "think tag"),
])
def test_the_template_checks_refuse_what_a_template_must_not_be(text, problem):
    answer = {"role": "assistant", "content": "", "tool_calls": [call("search_arxiv", {"query": "x"}, "a")]}
    assert problem in tr.problems(text, answer)


def test_the_script_refuses_a_template_that_fails_its_checks(tmp_path, monkeypatch, capsys):
    w = world(tmp_path)
    monkeypatch.setattr(tr, "template_reasoning", lambda record, k, rule: "Too short.")
    assert tr.main(w["argv"]) == 1
    assert not w["out"].exists() and "2 to 4 sentences" in capsys.readouterr().err


def test_a_stale_original_row_is_a_reasoning_error_not_a_silent_drop(tmp_path):
    """A row kept for a turn whose answer differs is what ReasoningIndex calls stale; the script refuses it
    the same way instead of passing it on."""
    w = world(tmp_path)
    rows = read_jsonl(w["teacher"])
    rows[0] = {**rows[0], "answer_sha256": "0" * 64}
    write_jsonl(w["teacher"], rows)
    assert tr.main(w["argv"]) == 1
    with pytest.raises(ReasoningError):
        ReasoningIndex([w["teacher"]]).lookup(rows[0]["id"], rows[0]["k"],
                                              copy.deepcopy(turn_answer(read_jsonl(w["originals"])[0], 0)))

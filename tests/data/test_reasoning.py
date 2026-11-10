"""Reasoning sidecars (docs/m7-家族层设计.md §6, WP-C): research_agent.data.reasoning,
scripts/join_reasoning.py and scripts/rationalize_reasoning.py.

The join runs on a small world: two teacher records with their M2 transcripts, a scripted record, and a
response cache that holds the teacher's answers under keys this file computes on its own, the way the M2
loop's requests were keyed (CloudModel through CloudClient, before thinking was handled); one more test lets
a loop run as the M2 generator ran it fill the cache instead, so the keys are the client's own. The
rationalizer runs against the strict fake DeepSeek endpoint (support.strict_deepseek), with a word count
standing in for the tokenizer. No test opens runs/llm/ or a model folder.
"""

from __future__ import annotations

import copy
import hashlib
import json
import random
import sqlite3
from datetime import datetime
from pathlib import Path

import httpx2
import pytest
from support.fakes import HITS, PAPER, table_check, table_decide, table_tools
from support.strict_deepseek import Answer, StrictDeepSeek, thinks
from test_synth import load_script

from research_agent.agent.context import build_context, step_contexts
from research_agent.agent.executor import ToolExecutor
from research_agent.agent.loop import run_turn
from research_agent.agent.models import CloudModel
from research_agent.agent.prompts import PROMPT_HASH, PROMPT_VERSION, system_prompt
from research_agent.agent.types import ANSWERED, ToolOutcome, Transcript
from research_agent.canonical import canonical_json
from research_agent.data import reasoning
from research_agent.data.reasoning import ReasoningError, ReasoningIndex, answer_sha256
from research_agent.llm.budget import Budget, cost_rmb, load_off_peak, load_prices
from research_agent.llm.cache import ResponseCache, request_key
from research_agent.llm.cloud import THINKING_OFF, CloudClient
from research_agent.tools.schemas import MODEL_TOOLS, SCHEMA_HASH

jr = load_script("join_reasoning")
rr = load_script("rationalize_reasoning")

TODAY = "2026-10-10"
MODEL = "deepseek-flash"
PEAK = datetime.fromisoformat("2026-10-19T06:43:00+00:00")  # 14:43 in Beijing: DeepSeek's peak hours
OFF_PEAK = datetime.fromisoformat("2026-10-19T13:00:00+00:00")  # 21:00 in Beijing


def words(text: str) -> int:
    """The tokenizer's stand-in: one token per whitespace-separated word."""
    return len(text.split())


def api_call(call_id: str, name: str, **args) -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def step(*calls: dict, content: str = "") -> dict:
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = list(calls)
    return message


def result(call_id: str, value: object) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": canonical_json(value)}


SEARCH = api_call("call_00_s", "search_arxiv", query="speculative decoding", date_from="2026-09-01",
                  max_results=5)
SEARCH_EDITED = api_call("call_00_s", "search_arxiv", query="speculative decoding", date_from="2026-09-01")
GET = api_call("call_00_g", "get_paper", arxiv_id="2609.01234")
GET_STEP = step(GET)
ASK = api_call("call_00_a", "ask_clarification", question="Which paper do you mean?", missing=["paper_id"])
LIST = api_call("call_00_l", "list_notes", tags=["llm"], since="2026-09-01")
LONG_HITS = [{**hit, "snippet": hit["snippet"] * 12} for hit in HITS]  # well over 1,500 characters as JSON


def history(user: str, hits: list | None = None) -> list[dict]:
    """A finished trajectory in API form: a search, a get_paper, an answer in text."""
    return [{"role": "system", "content": system_prompt(TODAY)}, {"role": "user", "content": user},
            step(SEARCH), result("call_00_s", HITS if hits is None else hits), step(GET),
            result("call_00_g", PAPER), step(content="这篇讲自适应草稿长度。")]


def transcript(messages: list[dict]) -> Transcript:
    return Transcript(request=messages[1]["content"], today=TODAY, status="answered",
                      final_text=messages[-1].get("content"), messages=copy.deepcopy(messages), steps=[],
                      schema_hash=SCHEMA_HASH, prompt_version=PROMPT_VERSION, prompt_hash=PROMPT_HASH)


def record(rid: str, messages: list[dict], *, run: str = "train") -> dict:
    """An agent-history record as data.assemble and scripts/transform_synth.py write them."""
    return {"family": "m6/x" if run == "m6-transform" else "x/zh/1", "history": "agent", "id": rid,
            "intent": "search_arxiv", "lang": "zh", "messages": copy.deepcopy(messages),
            "prompt_hash": PROMPT_HASH, "run": run, "scenario": "normal", "schema_hash": SCHEMA_HASH,
            "tools": MODEL_TOOLS}


def plain_key(before: list[dict]) -> str:
    """The key the M2 teacher's answer to the step after `before` was cached under: what CloudModel sent for
    the loop's request (build_context of the history), MODEL_TOOLS, temperature 0.0 (llm.cloud, rule 1)."""
    return request_key({"model": MODEL, "messages": build_context(before), "tools": MODEL_TOOLS,
                        "temperature": 0.0})


def served(message: dict, reasoning_text: str) -> dict:
    """The response body as the M2 cache holds it: raw tool calls carry an index, the message its
    reasoning."""
    body = {"role": "assistant", "content": message.get("content") or "", "reasoning_content": reasoning_text}
    if message.get("tool_calls"):
        body["tool_calls"] = [{**c, "index": n} for n, c in enumerate(message["tool_calls"])]
    return {"id": "chatcmpl-1", "model": MODEL, "object": "chat.completion",
            "choices": [{"index": 0, "finish_reason": "stop", "message": body}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50}}


def assistant_positions(messages: list[dict]) -> list[int]:
    return [i for i, m in enumerate(messages) if m["role"] == "assistant"]


# The teacher's reasoning per (record id, k), for the world below.
TEACHER_TEXT = {
    ("req-train-0001", 0): "The user wants September papers. I will call search_arxiv.",
    ("req-train-0001", 1): "The first hit fits. I will call get_paper on it.",
    ("req-train-0001", 2): "",
    ("req-train-0002", 0): "Search first with search_arxiv.",
    ("req-train-0002", 1): "<think>nested</think> then get_paper.",
    ("req-train-0002", 2): "The abstract answers it.",
}


def world() -> tuple[list[dict], dict[str, Transcript], ResponseCache]:
    """Two teacher records and a scripted one, the M2 transcripts, the cache of the teacher's answers.

    req-train-0001 had its first call edited after M2 (an R1-style deletion of max_results), so its turn 0 is
    edited and the context of its later turns differs from the teacher's; its last reasoning is empty.
    req-train-0002 is the transcript as it was; the reasoning of its turn 1 holds a think tag.
    """
    one = history("帮我找 2026 年 9 月关于 speculative decoding 的论文")
    two = history("Find September papers")
    trained_one = copy.deepcopy(one)
    trained_one[2] = step(SEARCH_EDITED)
    scripted = [{"role": "system", "content": system_prompt(TODAY)},
                {"role": "user", "content": "列出我 9 月以来打了 llm 标签的笔记"}, step(LIST)]
    records = [record("req-train-0001", trained_one), record("req-train-0002", two),
               record("req-m6-0001", scripted, run="m6-transform")]
    transcripts = {"req-train-0001": transcript(one), "req-train-0002": transcript(two)}
    cache = ResponseCache(":memory:")
    for rid, messages in (("req-train-0001", one), ("req-train-0002", two)):
        for k, j in enumerate(assistant_positions(messages)):
            cache.put(plain_key(messages[:j]), MODEL, served(messages[j], TEACHER_TEXT[rid, k]))
    return records, transcripts, cache


# ---- answer_sha256 ------------------------------------------------------------------------------------


def test_the_answer_hash_ignores_argument_order_whitespace_call_order_and_call_ids():
    base = step(api_call("a", "search_arxiv", query="kv cache", max_results=3), GET, content="  Two\n calls ")
    same = step(GET | {"id": "zz"},
                {"id": "b", "type": "function",
                 "function": {"name": "search_arxiv", "arguments": '{"max_results":3.0,"query":"kv cache"}'}},
                content="Two calls")
    hf_form = step({"type": "function", "function": {"name": "search_arxiv",
                                                    "arguments": {"query": "kv cache", "max_results": 3}}},
                   {"type": "function", "function": {"name": "get_paper",
                                                    "arguments": {"arxiv_id": "2609.01234"}}},
                   content="Ｔｗｏ calls")
    assert answer_sha256(base) == answer_sha256(same) == answer_sha256(hf_form)
    assert len(answer_sha256(base)) == 64


def test_the_answer_hash_changes_with_a_value_a_name_a_call_or_the_text():
    base = step(api_call("a", "search_arxiv", query="kv cache", max_results=3))
    variants = [
        step(api_call("a", "search_arxiv", query="kv cache", max_results=4)),  # a changed value
        step(api_call("a", "search_arxiv", query="kv cache")),  # an argument left out
        step(api_call("a", "search_notes", query="kv cache", max_results=3)),  # another tool
        step(api_call("a", "search_arxiv", query="kv cache", max_results=3),
             api_call("b", "search_arxiv", query="kv cache", max_results=3)),  # the same call twice
        step(api_call("a", "search_arxiv", query="kv cache", max_results=3), content="Searching."),
        step(content="kv cache"),
    ]
    hashes = {answer_sha256(v) for v in variants}
    assert answer_sha256(base) not in hashes and len(hashes) == len(variants)


def raw_call(arguments: str) -> dict:
    """A get_paper call whose arguments are text as written, not necessarily a JSON object."""
    return {"id": "a", "type": "function", "function": {"name": "get_paper", "arguments": arguments}}


def test_arguments_that_are_not_a_json_object_hash_as_their_text():
    bad, other = step(raw_call("{arxiv_id: 1")), step(raw_call("{arxiv_id: 2"))
    assert answer_sha256(bad) != answer_sha256(other)
    assert answer_sha256(bad) == answer_sha256(copy.deepcopy(bad))


# ---- turn keys: the training path's ---------------------------------------------------------------------


def test_turn_k_is_the_kth_sample_the_trainer_cuts_from_an_agent_history():
    messages = history("找论文")
    turns = reasoning.turns(messages)
    pairs = step_contexts(messages)  # what train.samples.build_step_samples cuts an agent history into
    assert [k for k, _, _ in turns] == [0, 1, 2]
    assert [(build_context(messages[:i]), message) for _, i, message in turns] == pairs


def test_a_scripted_record_is_told_by_its_run_and_its_id_must_agree():
    assert reasoning.is_scripted(record("req-m6-0007", history("x"), run="m6-transform"))
    assert reasoning.is_scripted(record("req-train-0090-m6r3", history("x"), run="m6-transform"))
    assert reasoning.is_scripted(record("req-train-0091-m6r6", history("x"), run="m6-transform"))
    assert not reasoning.is_scripted(record("req-train-0001", history("x")))
    with pytest.raises(ReasoningError, match="req-train-0002"):
        reasoning.is_scripted(record("req-train-0002", history("x"), run="m6-transform"))
    with pytest.raises(ReasoningError, match="req-m6-0003"):
        reasoning.is_scripted(record("req-m6-0003", history("x")))


@pytest.mark.parametrize("rid", ["req-train-0001-m6r5", "req-train-0001-m6r3x", "req-m6-x", "req-m6-0001b",
                                 "x-req-m6-0001"])
def test_an_id_that_only_resembles_a_scripted_one_is_not_scripted(rid):
    """Another -m6r suffix, a character more after the id or before it, no number: no scripted record's id,
    so with run m6-transform the two markers disagree."""
    assert not reasoning.is_scripted(record(rid, history("x")))
    with pytest.raises(ReasoningError, match=rid):
        reasoning.is_scripted(record(rid, history("x"), run="m6-transform"))


def test_the_records_ft4_builds_are_scripted_and_their_ids_must_say_so():
    """FT4 (scripts/transform_synth.py --ft4) builds new records (req-ft4-NNNN) and cut copies of trajectories
    (<id>-ft4r9k<k>, <id>-ft4r10k<k>) under run ft4-transform; each run accepts only its own ids."""
    assert reasoning.is_scripted(record("req-ft4-0001", history("x"), run=reasoning.FT4_RUN))
    assert reasoning.is_scripted(record("req-train-0001-ft4r9k3", history("x"), run="ft4-transform"))
    assert reasoning.is_scripted(record("req-train-1700-ft4r10k1", history("x"), run="ft4-transform"))
    assert reasoning.is_scripted(record("req-m6-0007", history("x"), run="m6-transform"))  # M6's as before
    for rid, run in (("req-train-0001", "ft4-transform"), ("req-ft4-0001", "train"),
                     ("req-ft4-0001", "m6-transform"), ("req-m6-0001", "ft4-transform"),
                     ("req-train-0001-ft4r9k3", "m6-transform"), ("req-train-0001-m6r3", "ft4-transform")):
        with pytest.raises(ReasoningError, match=rid):
            reasoning.is_scripted(record(rid, history("x"), run=run))


@pytest.mark.parametrize("rid", ["req-ft4-x", "req-ft4-0001b", "x-req-ft4-0001", "req-train-0001-ft4r8k1",
                                 "req-train-0001-ft4r9", "req-train-0001-ft4r10k1x"])
def test_an_id_that_only_resembles_an_ft4_one_is_not_scripted(rid):
    assert not reasoning.is_scripted(record(rid, history("x")))
    with pytest.raises(ReasoningError, match=rid):
        reasoning.is_scripted(record(rid, history("x"), run="ft4-transform"))


# ---- the join -------------------------------------------------------------------------------------------


def test_the_join_classifies_every_turn_and_rebuilds_the_teachers_cache_keys():
    records, transcripts, cache = world()
    joined = jr.join(records, transcripts.__getitem__, cache.get)
    by_key = {(j.id, j.k): j for j in joined}
    assert {key: j.status for key, j in by_key.items()} == {
        ("req-train-0001", 0): "edited", ("req-train-0001", 1): "teacher", ("req-train-0001", 2): "empty",
        ("req-train-0002", 0): "teacher", ("req-train-0002", 1): "think_tag",
        ("req-train-0002", 2): "teacher", ("req-m6-0001", 0): "scripted"}
    # the context of turn 1 of the edited record holds the edited call, which the teacher never saw
    assert {key: j.context_equal for key, j in by_key.items()} == {
        ("req-train-0001", 0): True, ("req-train-0001", 1): False, ("req-train-0001", 2): False,
        ("req-train-0002", 0): True, ("req-train-0002", 1): True, ("req-train-0002", 2): True,
        ("req-m6-0001", 0): None}
    # whatever the reasoning, only the edited turn trains on another answer than the teacher gave
    assert {key: j.answer_equal for key, j in by_key.items()} == {
        ("req-train-0001", 0): False, ("req-train-0001", 1): True, ("req-train-0001", 2): True,
        ("req-train-0002", 0): True, ("req-train-0002", 1): True, ("req-train-0002", 2): True,
        ("req-m6-0001", 0): None}
    for rid in ("req-train-0001", "req-train-0002"):
        messages = transcripts[rid].messages
        for k, j in enumerate(assistant_positions(messages)):
            assert by_key[rid, k].cache_key == plain_key(messages[:j])
    assert by_key["req-m6-0001", 0].cache_key is None
    rows = [j.row() for j in joined if j.status == "teacher"]
    assert [(r["id"], r["k"]) for r in rows] == [("req-train-0001", 1), ("req-train-0002", 0),
                                                 ("req-train-0002", 2)]
    assert rows[0] == {"id": "req-train-0001", "k": 1, "answer_sha256": answer_sha256(GET_STEP),
                       "source": "teacher", "reasoning": TEACHER_TEXT["req-train-0001", 1], "model": MODEL,
                       "cache_key": by_key["req-train-0001", 1].cache_key, "context_equal": False}


def test_the_first_status_that_applies_wins_empty_then_edited_then_think_tag():
    """A turn whose reasoning is empty is empty even where M6 changed its answer, and a changed answer is
    edited even where the reasoning holds a think tag; answer_equal tells what M6 did either way."""
    records, transcripts, cache = world()
    # M6 changed turn 2 of req-train-0001, whose reasoning is empty, and turn 1 of req-train-0002, whose
    # reasoning holds a think tag
    records[0]["messages"][6] = step(content="这篇讲动态草稿长度。")
    records[1]["messages"][4] = step(api_call("call_00_g", "get_paper", arxiv_id="2609.05678"))
    by_key = {(j.id, j.k): j for j in jr.join(records, transcripts.__getitem__, cache.get)}
    assert (by_key["req-train-0001", 2].status, by_key["req-train-0001", 2].answer_equal) == ("empty", False)
    assert (by_key["req-train-0002", 1].status, by_key["req-train-0002", 1].answer_equal) == ("edited", False)


def test_the_join_reads_back_what_the_teachers_loop_cached_for_every_turn():
    """The keys are not this file's own: a turn run as the M2 generator ran one (agent.loop.run_turn with
    CloudModel over CloudClient, against the strict fake endpoint) fills a cache, and the join finds in it the
    reasoning the endpoint gave with every answer."""
    given = ("Search September papers with search_arxiv.", "Open the first hit with get_paper.",
             "The abstract answers the question.")
    search = ("call_00_s", "search_arxiv", {"query": "speculative decoding", "date_from": "2026-09-01"})
    server = StrictDeepSeek([Answer(calls=(search,), reasoning=given[0]),
                             Answer(calls=(("call_00_g", "get_paper", {"arxiv_id": "2609.01234"}),),
                                    reasoning=given[1]),
                             Answer("这篇讲自适应草稿长度。", reasoning=given[2])])
    cache = ResponseCache(":memory:")
    budget = Budget(":memory:", stage="M2", stage_cap_rmb=1.0)
    model = CloudModel(CloudClient(MODEL, cache=cache, budget=budget, client=server.sdk()))
    tools, _ = table_tools({"search_arxiv": [ToolOutcome(HITS)], "get_paper": [ToolOutcome(PAPER)]})
    taught = run_turn("找 2026 年 9 月关于 speculative decoding 的论文", model=model,
                      executor=ToolExecutor(tools, table_check(), today=TODAY), decide=table_decide(),
                      today=TODAY, cost_of=model.cost)
    assert (taught.status, taught.paid_calls, server.refused) == (ANSWERED, 3, 0)
    joined = jr.join([record("req-train-0003", taught.messages)], {"req-train-0003": taught}.__getitem__,
                     cache.get)
    assert [(j.k, j.status, j.reasoning, j.context_equal) for j in joined] == [
        (0, "teacher", given[0], True), (1, "teacher", given[1], True), (2, "teacher", given[2], True)]
    assert [j.cache_key for j in joined] == [key for (key,) in cache.db.execute(
        "SELECT key FROM responses ORDER BY rowid")]


def test_a_cache_miss_stops_the_join():
    records, transcripts, _ = world()
    empty = ResponseCache(":memory:")
    with pytest.raises(jr.JoinError, match=r"req-train-0001 turn 0: no cached answer"):
        jr.join(records, transcripts.__getitem__, empty.get)


def test_a_cached_answer_that_is_not_the_transcripts_stops_the_join():
    records, transcripts, cache = world()
    messages = transcripts["req-train-0002"].messages
    j = assistant_positions(messages)[1]
    other = step(api_call("call_00_g", "get_paper", arxiv_id="2609.05678"))
    cache.put(plain_key(messages[:j]), MODEL, served(other, "Another paper."))
    with pytest.raises(jr.JoinError,
                       match="req-train-0002 turn 1: the cached answer is not the transcript's"):
        jr.join(records, transcripts.__getitem__, cache.get)


def test_a_record_with_more_turns_than_its_transcript_stops_the_join():
    records, transcripts, cache = world()
    records[1]["messages"].append(step(content="One more."))
    with pytest.raises(jr.JoinError, match="req-train-0002 has 4 assistant turns, its transcript 3"):
        jr.join(records, transcripts.__getitem__, cache.get)


def test_a_transcript_of_another_prompt_stops_the_join():
    records, transcripts, cache = world()
    transcripts["req-train-0002"].prompt_hash = "0" * 16
    with pytest.raises(jr.JoinError, match="req-train-0002: its transcript was recorded under prompt"):
        jr.join(records, transcripts.__getitem__, cache.get)


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes("".join(canonical_json(r) + "\n" for r in rows).encode("utf-8"))
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def join_world(tmp_path: Path) -> list[str]:
    """The world on disk: records, transcripts and a cache file; the script's arguments."""
    records, transcripts, memory = world()
    write_jsonl(tmp_path / "synth.jsonl", records)
    for rid, t in transcripts.items():
        (tmp_path / "transcripts").mkdir(exist_ok=True)
        (tmp_path / "transcripts" / f"{rid}.json").write_text(t.to_json() + "\n", encoding="utf-8")
    cache = ResponseCache(tmp_path / "cache.sqlite")
    for key, model, response in memory.db.execute("SELECT key, model, response FROM responses"):
        cache.put(key, model, json.loads(response))
    cache.close()
    return ["--records", str(tmp_path / "synth.jsonl"), "--transcripts", str(tmp_path / "transcripts"),
            "--cache", str(tmp_path / "cache.sqlite"), "--out", str(tmp_path / "teacher.jsonl"),
            "--report", str(tmp_path / "teacher.report.json")]


def test_the_join_script_writes_the_teacher_rows_and_a_report_without_text_and_leaves_the_cache_alone(
        tmp_path, capsys):
    argv = join_world(tmp_path)
    cache_bytes = (tmp_path / "cache.sqlite").read_bytes()
    assert jr.main(argv, count_tokens=words) == 0
    assert (tmp_path / "cache.sqlite").read_bytes() == cache_bytes  # nothing written (read-only: next test)
    rows = read_jsonl(tmp_path / "teacher.jsonl")
    assert [(r["id"], r["k"], r["source"]) for r in rows] == [
        ("req-train-0001", 1, "teacher"), ("req-train-0002", 0, "teacher"), ("req-train-0002", 2, "teacher")]
    text = (tmp_path / "teacher.report.json").read_text(encoding="utf-8")
    report = json.loads(text)
    assert report["status"] == {"teacher": 3, "edited": 1, "empty": 1, "think_tag": 1, "scripted": 1}
    assert report["turns"] == {"total": 7, "teacher_trajectory": 6, "scripted": 1}
    assert report["records"]["teacher_trajectory"] == 2 and report["records"]["scripted"] == 1
    assert report["keys"] == {"looked_up": 6, "hit": 6}
    assert report["non_empty"] == 5
    assert report["context_equal"]["teacher"] == 2 and report["context_equal"]["all"] == 4
    assert report["answer_equal"] == {"teacher": 3, "edited": 0, "empty": 1, "think_tag": 1, "all": 5}
    assert report["lengths"]["characters"]["n"] == 3
    teacher_keys = (("req-train-0001", 1), ("req-train-0002", 0), ("req-train-0002", 2))
    assert report["lengths"]["tokens"]["max"] == max(words(TEACHER_TEXT[key]) for key in teacher_keys)
    assert report["lengths"]["characters"]["max"] == max(len(TEACHER_TEXT[key]) for key in teacher_keys)
    assert report["expected"] == {"teacher_turns": 4308, "non_empty": 4141, "teacher": 2814}
    assert report["difference"] == {"teacher_turns": 6 - 4308, "non_empty": 5 - 4141, "teacher": 3 - 2814}
    assert report["output"]["rows"] == 3
    assert report["output"]["sha256"] == hashlib.sha256((tmp_path / "teacher.jsonl").read_bytes()).hexdigest()
    for reasoning_text in TEACHER_TEXT.values():  # counts and lengths only: no cached text
        assert not reasoning_text or reasoning_text not in text
    first = (tmp_path / "teacher.jsonl").read_bytes(), text
    assert jr.main(argv, count_tokens=words) == 0  # the same inputs give the same bytes
    assert ((tmp_path / "teacher.jsonl").read_bytes(), (tmp_path / "teacher.report.json").read_text(
        encoding="utf-8")) == first
    assert "teacher 3" in capsys.readouterr().out


def test_the_join_reads_the_cache_through_a_connection_that_cannot_write(tmp_path):
    """The cache the join script opens (ReadOnlyCache, SQLite mode=ro) gives what the cache holds, and every
    write through it fails, so the shared cache stays as it is whatever the join does."""
    join_world(tmp_path)
    path = tmp_path / "cache.sqlite"
    before = path.read_bytes()
    cache = jr.ReadOnlyCache(path)
    try:
        cached = cache.get(plain_key(history("Find September papers")[:2]))  # turn 0 of req-train-0002
        assert cached["choices"][0]["message"]["reasoning_content"] == TEACHER_TEXT["req-train-0002", 0]
        assert cache.get("0" * 64) is None
        for statement in ("CREATE TABLE t (a)", "DELETE FROM responses",
                          "INSERT INTO responses VALUES ('k', 'm', '{}', 'now')"):
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                cache.db.execute(statement)
    finally:
        cache.close()
    assert path.read_bytes() == before


def test_the_join_script_writes_nothing_when_a_transcript_is_missing(tmp_path, capsys):
    argv = join_world(tmp_path)
    (tmp_path / "transcripts" / "req-train-0002.json").unlink()
    assert jr.main(argv, count_tokens=words) == 1
    assert not (tmp_path / "teacher.jsonl").exists() and not (tmp_path / "teacher.report.json").exists()
    assert "req-train-0002" in capsys.readouterr().err


def test_the_join_script_refuses_a_missing_cache_without_creating_one(tmp_path, capsys):
    argv = join_world(tmp_path)
    (tmp_path / "cache.sqlite").unlink()
    assert jr.main(argv, count_tokens=words) == 2
    assert not (tmp_path / "cache.sqlite").exists()


def test_the_join_report_without_a_run_keeps_its_keys(tmp_path):
    """The --run option leaves the report of a join without it as it was (M7's first sidecar)."""
    argv = join_world(tmp_path)
    assert jr.main(argv, count_tokens=words) == 0
    report = json.loads((tmp_path / "teacher.report.json").read_text(encoding="utf-8"))
    assert set(report) == {"script", "inputs", "teacher_request", "records", "turns", "keys", "status",
                           "non_empty", "answer_equal", "context_equal", "lengths", "expected", "difference",
                           "output"}
    assert set(report["keys"]) == {"looked_up", "hit"}
    assert report["teacher_request"] == {"model": MODEL, "temperature": 0.0, "tools": "MODEL_TOOLS",
                                         "schema_hash": SCHEMA_HASH}


# ---- the join of a run whose teacher passed its reasoning back (M7) ---------------------------------------

SEARCH_CALL = ("call_00_s", "search_arxiv", {"query": "speculative decoding", "date_from": "2026-09-01"})
GET_CALL = ("call_00_g", "get_paper", {"arxiv_id": "2609.01234"})
THOUGHT = "Search September papers with search_arxiv."


def off_key(before: list[dict]) -> str:
    """The key CloudClient files an answer given without thinking under (llm.cloud, rule 3): the plain
    request with extra_body THINKING_OFF."""
    return request_key({"model": MODEL, "messages": build_context(before), "tools": MODEL_TOOLS,
                        "temperature": 0.0, "extra_body": THINKING_OFF})


def pass_back_turn(answers: list[Answer], *, thinking: str = "auto"
                   ) -> tuple[Transcript, ResponseCache, StrictDeepSeek]:
    """A turn run as the M7 generator ran it: agent.loop.run_turn with CloudModel over CloudClient (thinking
    auto: the reasoning of the earlier turns passed back), against the strict fake endpoint, which refuses a
    thinking request with tools whose earlier turns lack their reasoning."""
    server = StrictDeepSeek(answers)
    cache = ResponseCache(":memory:")
    budget = Budget(":memory:", stage="M7", stage_cap_rmb=1.0)
    model = CloudModel(CloudClient(MODEL, cache=cache, budget=budget, client=server.sdk(), thinking=thinking))
    tools, _ = table_tools({"search_arxiv": [ToolOutcome(HITS)], "get_paper": [ToolOutcome(PAPER)]})
    taught = run_turn("找 2026 年 9 月关于 speculative decoding 的论文", model=model,
                      executor=ToolExecutor(tools, table_check(), today=TODAY), decide=table_decide(),
                      today=TODAY, cost_of=model.cost)
    return taught, cache, server


EMPTY_IN_THE_MIDDLE = (Answer(calls=(SEARCH_CALL,), reasoning=THOUGHT),
                       Answer(calls=(GET_CALL,), reasoning=""), Answer("这篇讲自适应草稿长度。"))


def test_the_join_of_a_pass_back_run_finds_every_turn_under_the_key_the_client_used():
    """M7's teacher passed its reasoning back. The reasoning goes on the wire only: the client files a
    thinking answer under the plain key (llm.cloud, rule 2). Once a turn's reasoning is empty it cannot be
    passed back, so the turns after it go without thinking and are filed under the THINKING_OFF key (rule 3),
    where a plain-key join (M2's teacher) finds nothing."""
    taught, cache, server = pass_back_turn(list(EMPTY_IN_THE_MIDDLE))
    assert (taught.status, taught.paid_calls, server.refused) == (ANSWERED, 3, 0)
    assert [thinks(body) for body in server.bodies] == [True, True, False]
    assert server.bodies[1]["messages"][2]["reasoning_content"] == THOUGHT  # passed back on the wire
    messages = taught.messages
    at = assistant_positions(messages)
    assert cache.get(plain_key(messages[:at[2]])) is None
    joined = jr.join([record("req-train-1601", messages, run="m7-train")],
                     {"req-train-1601": taught}.__getitem__, cache.get)
    assert [(j.k, j.status, j.key_kind) for j in joined] == [
        (0, "teacher", "plain"), (1, "empty", "plain"), (2, "empty", "thinking_off")]
    assert [j.cache_key for j in joined] == [plain_key(messages[:at[0]]), plain_key(messages[:at[1]]),
                                             off_key(messages[:at[2]])]
    assert [j.cache_key for j in joined] == [key for (key,) in cache.db.execute(
        "SELECT key FROM responses ORDER BY rowid")]
    assert joined[0].reasoning == THOUGHT


def test_a_plain_key_that_holds_the_answer_is_taken_whatever_the_earlier_reasoning():
    """Rule 1: the client serves a plain-key hit whatever the history, so the join takes it after an empty
    reasoning too (every M2 answer is under the plain key, the empty ones' successors included)."""
    records, transcripts, cache = world()
    messages = transcripts["req-train-0002"].messages
    at = assistant_positions(messages)
    cache.put(plain_key(messages[:at[1]]), MODEL, served(messages[at[1]], ""))  # turn 1's reasoning: empty
    by_key = {(j.id, j.k): j for j in jr.join(records, transcripts.__getitem__, cache.get)}
    assert by_key["req-train-0002", 1].status == "empty"
    assert (by_key["req-train-0002", 2].status, by_key["req-train-0002", 2].key_kind) == ("teacher", "plain")


def test_after_known_reasoning_an_answer_filed_without_thinking_stops_the_join():
    """With every earlier reasoning known, the client sent the request in thinking mode and filed the answer
    under the plain key (rule 2): one found only under the THINKING_OFF key is not what the loop got."""
    records, transcripts, cache = world()
    messages = transcripts["req-train-0002"].messages
    at = assistant_positions(messages)
    cache.db.execute("DELETE FROM responses WHERE key = ?", (plain_key(messages[:at[2]]),))
    cache.put(off_key(messages[:at[2]]), MODEL, served(messages[at[2]], ""))
    with pytest.raises(jr.JoinError, match="req-train-0002 turn 2: no cached answer under the teacher's key "
                                           + plain_key(messages[:at[2]])):
        jr.join(records, transcripts.__getitem__, cache.get)


def test_a_miss_after_an_unknown_reasoning_names_the_thinking_off_key():
    taught, cache, _ = pass_back_turn(list(EMPTY_IN_THE_MIDDLE))
    messages = taught.messages
    at = assistant_positions(messages)
    cache.db.execute("DELETE FROM responses WHERE key = ?", (off_key(messages[:at[2]]),))
    with pytest.raises(jr.JoinError, match="req-train-1601 turn 2: no cached answer under the teacher's key "
                                           + off_key(messages[:at[2]])):
        jr.join([record("req-train-1601", messages, run="m7-train")], {"req-train-1601": taught}.__getitem__,
                cache.get)


def test_a_run_made_with_thinking_off_is_filed_under_the_thinking_off_key_throughout():
    answers = [Answer(calls=(SEARCH_CALL,)), Answer(calls=(GET_CALL,)), Answer("这篇讲自适应草稿长度。")]
    taught, cache, server = pass_back_turn(answers, thinking="off")
    assert [thinks(body) for body in server.bodies] == [False, False, False]
    joined = jr.join([record("req-train-1601", taught.messages, run="m7-off")],
                     {"req-train-1601": taught}.__getitem__, cache.get, teacher=jr.Teacher(thinking="off"))
    assert [(j.status, j.key_kind) for j in joined] == [("empty", "thinking_off")] * 3
    with pytest.raises(jr.JoinError, match="req-train-1601 turn 0: no cached answer"):
        jr.join([record("req-train-1601", taught.messages, run="m7-off")],  # an auto teacher: plain key first
                {"req-train-1601": taught}.__getitem__, cache.get)


def run_world(tmp_path: Path, *, thinking: str = "auto", record_run: str = "m7-train") -> dict:
    """A generator run folder as the M7 run left it (manifest.json with the run's config, transcripts/), a
    record assembled from it and the cache its loop filled; the script's arguments and the paths."""
    taught, memory, _ = pass_back_turn(list(EMPTY_IN_THE_MIDDLE), thinking=thinking)
    run_dir = tmp_path / "m7-train"
    (run_dir / "transcripts").mkdir(parents=True)
    (run_dir / "transcripts" / "req-train-1601.json").write_text(taught.to_json() + "\n", encoding="utf-8")
    manifest = {"config": {"model": MODEL, "model_temperature": 0.0, "thinking": thinking, "run": "m7-train",
                           "requests_sha256": "f" * 64},
                "prompt_hash": PROMPT_HASH, "schema_hash": SCHEMA_HASH}
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    write_jsonl(tmp_path / "synth-m7.jsonl", [record("req-train-1601", taught.messages, run=record_run)])
    cache = ResponseCache(tmp_path / "cache.sqlite")
    for key, model, response in memory.db.execute("SELECT key, model, response FROM responses"):
        cache.put(key, model, json.loads(response))
    cache.close()
    argv = ["--records", str(tmp_path / "synth-m7.jsonl"), "--run", str(run_dir),
            "--cache", str(tmp_path / "cache.sqlite"), "--out", str(tmp_path / "teacher.jsonl"),
            "--report", str(tmp_path / "teacher.report.json")]
    return {"argv": argv, "run": run_dir}


def test_the_run_option_joins_a_pass_back_run_and_records_the_run(tmp_path, capsys):
    world_ = run_world(tmp_path)
    assert jr.main(world_["argv"], count_tokens=words) == 0
    rows = read_jsonl(tmp_path / "teacher.jsonl")
    assert [(r["id"], r["k"], r["reasoning"]) for r in rows] == [("req-train-1601", 0, THOUGHT)]
    text = (tmp_path / "teacher.report.json").read_text(encoding="utf-8")
    report = json.loads(text)
    assert report["status"] == {"teacher": 1, "edited": 0, "empty": 2, "think_tag": 0, "scripted": 0}
    assert report["keys"] == {"looked_up": 3, "hit": 3, "plain": 2, "thinking_off": 1}
    manifest = world_["run"] / "manifest.json"
    assert report["run"] == {"path": world_["run"].as_posix(), "name": "m7-train",
                             "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                             "model": MODEL, "temperature": 0.0, "thinking": "auto",
                             "requests_sha256": "f" * 64}
    assert report["inputs"]["transcripts"] == (world_["run"] / "transcripts").as_posix()
    assert "expected" not in report and "difference" not in report  # M6's numbers are not this run's
    assert report["teacher_request"] == {"model": MODEL, "temperature": 0.0, "tools": "MODEL_TOOLS",
                                         "schema_hash": SCHEMA_HASH}
    assert THOUGHT not in text
    assert "plain 2, thinking_off 1" in capsys.readouterr().out
    first = (tmp_path / "teacher.jsonl").read_bytes(), text
    assert jr.main(world_["argv"], count_tokens=words) == 0  # the same inputs give the same bytes
    assert ((tmp_path / "teacher.jsonl").read_bytes(),
            (tmp_path / "teacher.report.json").read_text(encoding="utf-8")) == first


def test_the_run_option_takes_the_thinking_setting_from_the_run(tmp_path):
    world_ = run_world(tmp_path, thinking="off")
    assert jr.main(world_["argv"], count_tokens=words) == 0
    report = json.loads((tmp_path / "teacher.report.json").read_text(encoding="utf-8"))
    assert report["keys"] == {"looked_up": 3, "hit": 3, "plain": 0, "thinking_off": 3}
    assert report["run"]["thinking"] == "off" and report["status"]["empty"] == 3


def test_the_run_option_refuses_a_record_of_another_run(tmp_path, capsys):
    world_ = run_world(tmp_path, record_run="train")
    assert jr.main(world_["argv"], count_tokens=words) == 1
    assert not (tmp_path / "teacher.jsonl").exists()
    assert "req-train-1601" in capsys.readouterr().err


def test_the_run_option_needs_the_runs_manifest_and_excludes_transcripts(tmp_path, capsys):
    world_ = run_world(tmp_path)
    with pytest.raises(SystemExit):  # argparse: --run and --transcripts name the same thing twice
        jr.main([*world_["argv"], "--transcripts", str(world_["run"] / "transcripts")], count_tokens=words)
    (world_["run"] / "manifest.json").unlink()
    assert jr.main(world_["argv"], count_tokens=words) == 2
    assert not (tmp_path / "teacher.jsonl").exists()


# ---- the index ----------------------------------------------------------------------------------------


def teacher_row(rid: str, k: int, answer: dict, text: str, /, **overrides) -> dict:
    """A teacher row of the turn, with overrides written over it unchecked (to make it malformed)."""
    row = reasoning.sidecar_row(rid, k, answer, source="teacher", reasoning=text, model=MODEL,
                                cache_key="a" * 64, context_equal=True)
    return {**row, **overrides}


def test_the_index_returns_the_reasoning_of_a_turn_whose_answer_it_was_written_for(tmp_path):
    path = tmp_path / "teacher.jsonl"
    sha, size = reasoning.write_sidecar([teacher_row("r2", 0, GET_STEP, "Open it."),
                                         teacher_row("r1", 1, step(SEARCH), "Search.")], path)
    assert sha == hashlib.sha256(path.read_bytes()).hexdigest() and size == len(path.read_bytes())
    assert [row["id"] for row in read_jsonl(path)] == ["r1", "r2"]
    index = ReasoningIndex([path])
    assert len(index) == 2
    assert index.lookup("r1", 1, step(SEARCH)) == "Search."
    reordered = step({**SEARCH, "id": "other",
                      "function": {"name": "search_arxiv",
                                   "arguments": '{"max_results":5,"date_from":"2026-09-01",'
                                                '"query":"speculative decoding"}'}})
    assert index.lookup("r1", 1, reordered) == "Search."  # the same answer in another spelling
    assert index.lookup("r1", 0, step(SEARCH)) is None and index.lookup("r9", 1, step(SEARCH)) is None


def test_a_row_written_for_another_answer_is_stale(tmp_path):
    path = tmp_path / "teacher.jsonl"
    reasoning.write_sidecar([teacher_row("r1", 1, step(SEARCH), "Search.")], path)
    with pytest.raises(ReasoningError, match=r"stale.*r1.*turn 1"):
        ReasoningIndex([path]).lookup("r1", 1, step(SEARCH_EDITED))


def test_a_turn_in_two_sidecar_files_is_an_error(tmp_path):
    first, second = tmp_path / "teacher.jsonl", tmp_path / "rationalized.jsonl"
    reasoning.write_sidecar([teacher_row("r1", 1, step(SEARCH), "Search.")], first)
    write_jsonl(second, [teacher_row("r1", 1, step(SEARCH), "Search again.", source="rationalized")])
    with pytest.raises(ReasoningError, match=r"rationalized\.jsonl: line 1: turn \('r1', 1\) is also in .*"
                                             r"teacher\.jsonl: line 1"):
        ReasoningIndex([first, second])
    twice = [teacher_row("r1", 1, step(SEARCH), "Search."), teacher_row("r1", 1, step(SEARCH), "x")]
    write_jsonl(second, twice)
    with pytest.raises(ReasoningError, match="line 2"):
        ReasoningIndex([second])
    with pytest.raises(ReasoningError, match="twice"):
        reasoning.write_sidecar(twice, tmp_path / "out.jsonl")


def test_a_template_row_is_a_sidecar_row(tmp_path):
    """FT4's template reasoning (scripts/template_reasoning.py) is a third source next to teacher and
    rationalized, bound to its answer like the others."""
    row = reasoning.sidecar_row("r1", 1, step(SEARCH), source=reasoning.TEMPLATE,
                                reasoning="The search failed for a moment. I will call search_arxiv again.",
                                model="template_reasoning-ft4", cache_key="b" * 64, context_equal=True)
    assert row["source"] == "template" and reasoning.TEMPLATE in reasoning.SOURCES
    path = tmp_path / "template.jsonl"
    reasoning.write_sidecar([row], path)
    assert ReasoningIndex([path]).lookup("r1", 1, step(SEARCH)) == row["reasoning"]
    with pytest.raises(ReasoningError, match="stale"):
        ReasoningIndex([path]).lookup("r1", 1, step(SEARCH_EDITED))


@pytest.mark.parametrize(("overrides", "problem"), [
    ({"source": "copied"}, "source"),
    ({"reasoning": "  "}, "reasoning is empty"),
    ({"reasoning": "a </think> b"}, "think tag"),
    ({"k": -1}, "k is not"),
    ({"k": True}, "k is not"),
    ({"answer_sha256": "abc"}, "answer_sha256"),
    ({"cache_key": None}, "cache_key"),
    ({"context_equal": "yes"}, "context_equal"),
    ({"extra": 1}, "exactly the keys"),
])
def test_a_malformed_row_is_refused_with_its_place(tmp_path, overrides, problem):
    path = write_jsonl(tmp_path / "teacher.jsonl",
                       [teacher_row("r1", 1, step(SEARCH), "Search.", **overrides)])
    with pytest.raises(ReasoningError, match=f"line 1: .*{problem}"):
        ReasoningIndex([path])


def test_a_line_that_is_not_json_is_refused_with_its_place(tmp_path):
    path = tmp_path / "teacher.jsonl"
    path.write_text('{"id": \n', encoding="utf-8")
    with pytest.raises(ReasoningError, match="line 1 is not JSON"):
        ReasoningIndex([path])


# ---- the rationalizer: targets, examples, prompt and checks ------------------------------------------------

EXAMPLES = {  # (record id, k) -> an eligible teacher reasoning: English, 2-6 sentences, names its tools
    ("req-train-0001", 0): "The user wants September papers on speculative decoding. I should call "
                           "search_arxiv with the topic and the start date.",
    ("req-train-0001", 2): "The abstract already answers the question. I can reply without another call.",
    ("req-train-0002", 0): "The user did not say which paper. I need ask_clarification before anything else.",
}


def rationalizer_world(tmp_path: Path) -> dict[str, Path]:
    """Two teacher records with teacher rows for three of their turns and a scripted record: the targets
    are turn 1 of req-train-0001 (get_paper) and the scripted list_notes turn."""
    one = history("帮我找 2026 年 9 月关于 speculative decoding 的论文", hits=LONG_HITS)
    two = [{"role": "system", "content": system_prompt(TODAY)},
           {"role": "user", "content": "Summarise the paper for me"}, step(ASK)]
    scripted = [{"role": "system", "content": system_prompt(TODAY)},
                {"role": "user", "content": "列出我 9 月以来打了 llm 标签的笔记"}, step(LIST)]
    records = [record("req-train-0001", one), record("req-train-0002", two),
               record("req-m6-0001", scripted, run="m6-transform")]
    paths = {"records": write_jsonl(tmp_path / "synth.jsonl", records), "teacher": tmp_path / "teacher.jsonl",
             "out": tmp_path / "rationalized.jsonl", "report": tmp_path / "rationalized.report.json",
             "dry_run_report": tmp_path / "dryrun.json", "llm": tmp_path / "llm"}
    answers = {("req-train-0001", 0): one[2], ("req-train-0001", 2): one[6], ("req-train-0002", 0): two[2]}
    rows = [teacher_row(rid, k, answers[rid, k], text) for (rid, k), text in EXAMPLES.items()]
    reasoning.write_sidecar(rows, paths["teacher"])
    return paths


def argv_of(paths: dict[str, Path], *extra: str) -> list[str]:
    return ["--records", str(paths["records"]), "--teacher", str(paths["teacher"]),
            "--out", str(paths["out"]), "--report", str(paths["report"]),
            "--dry-run-report", str(paths["dry_run_report"]), "--llm-dir", str(paths["llm"]), *extra]


def loaded(paths: dict[str, Path]) -> tuple[list[dict], ReasoningIndex]:
    return read_jsonl(paths["records"]), ReasoningIndex([paths["teacher"]])


def run(paths: dict[str, Path], server: StrictDeepSeek, *extra: str, at: datetime = OFF_PEAK) -> int:
    """The rationalizer's exit code for a run against the fake endpoint, its ledger's clock at `at`."""
    return rr.main(argv_of(paths, *extra), openai_client=server.sdk(), clock=lambda: at, count_tokens=words)


def test_the_targets_are_the_turns_the_teacher_sidecar_does_not_cover(tmp_path):
    records, index = loaded(rationalizer_world(tmp_path))
    targets = rr.find_targets(records, index)
    assert [(t.id, t.k, t.scripted) for t in targets] == [("req-m6-0001", 0, True),
                                                           ("req-train-0001", 1, False)]
    target = targets[1]
    assert target.answer == step(GET)
    assert target.context == build_context(records[0]["messages"][:4])  # the training context of turn 1


def test_a_teacher_row_for_a_turn_the_records_lack_is_stale(tmp_path):
    paths = rationalizer_world(tmp_path)
    records, _ = loaded(paths)
    rows = read_jsonl(paths["teacher"])
    reasoning.write_sidecar([*rows, teacher_row("req-train-0002", 5, step(ASK), "Ask.")], paths["teacher"])
    with pytest.raises(ReasoningError, match=r"req-train-0002.*5"):
        rr.find_targets(records, ReasoningIndex([paths["teacher"]]))


def test_the_style_examples_are_one_teacher_reasoning_per_answer_kind_whatever_the_row_order(tmp_path):
    paths = rationalizer_world(tmp_path)
    records, index = loaded(paths)
    picked = rr.pick_examples(index, records, words)
    assert [(e["id"], e["k"], e["kind"]) for e in picked] == [
        ("req-train-0001", 0, "call"), ("req-train-0002", 0, "clarify"), ("req-train-0001", 2, "answer")]
    rows = read_jsonl(paths["teacher"])
    random.Random(7).shuffle(rows)
    write_jsonl(paths["teacher"], rows)
    assert rr.pick_examples(ReasoningIndex([paths["teacher"]]), records, words) == picked


def rank(key: tuple[str, int]) -> str:
    """The order the style examples are chosen in: the SHA-256 of "<id>/<k>", smallest first."""
    return hashlib.sha256(f"{key[0]}/{key[1]}".encode()).hexdigest()


def test_the_style_example_of_a_kind_is_its_eligible_row_with_the_smallest_hash_of_its_key(tmp_path):
    """Several eligible rows of every kind, in two more records; the call row that ranks first is Chinese,
    so it is passed over. The choice is the same whatever the order of the records."""
    paths = rationalizer_world(tmp_path)
    third = history("Find September papers on speculative decoding")
    fourth = [{"role": "system", "content": system_prompt(TODAY)},
              {"role": "user", "content": "Open that paper"}, step(ASK)]
    records = [*read_jsonl(paths["records"]), record("req-train-0003", third),
               record("req-train-0004", fourth)]
    more = {("req-train-0001", 1): (GET_STEP, "The first hit fits the request. I will call get_paper on it."),
            ("req-train-0003", 0): (third[2], "The user wants September papers. I will call search_arxiv."),
            ("req-train-0003", 1): (third[4], "第一篇符合要求. 我会调用 get_paper."),
            ("req-train-0003", 2): (third[6], "The abstract answers the question. No call is needed."),
            ("req-train-0004", 0): (fourth[2], "No paper is named. I will call ask_clarification.")}
    rows = read_jsonl(paths["teacher"]) + [teacher_row(rid, k, answer, text)
                                           for (rid, k), (answer, text) in more.items()]
    reasoning.write_sidecar(rows, paths["teacher"])
    index = ReasoningIndex([paths["teacher"]])
    eligible = {"call": [("req-train-0001", 0), ("req-train-0001", 1), ("req-train-0003", 0)],
                "clarify": [("req-train-0002", 0), ("req-train-0004", 0)],
                "answer": [("req-train-0001", 2), ("req-train-0003", 2)]}
    expected = [min(eligible[kind], key=rank) for kind in rr.KINDS]
    assert rank(("req-train-0003", 1)) < rank(expected[0])  # the Chinese call row ranks first
    for order in (records, records[::-1]):
        assert [(e["id"], e["k"]) for e in rr.pick_examples(index, order, words)] == expected


@pytest.mark.parametrize("text", [
    "用户要九月的论文。我会调用 search_arxiv 检索。",  # not English, nor 2-6 sentences: "。" ends none
    "用户要九月的论文. 我会调用 search_arxiv.",  # not English, every other rule met
    "I call search_arxiv.",  # one sentence
    "One. Two. Three. Four. Five. Six. Seven search_arxiv.",  # seven sentences
    "The user wants September papers. I will call search_arxiv with " + "the topic and the date " * 25
    + "as filters. The defaults are fine.",  # three sentences, but over 120 words
    "The user wants papers. I will search for them now.",  # does not name search_arxiv
])
def test_a_reasoning_outside_the_requested_form_is_never_a_style_example(tmp_path, text):
    paths = rationalizer_world(tmp_path)
    records, _ = loaded(paths)
    rows = [teacher_row(r["id"], r["k"], step(SEARCH), text) if (r["id"], r["k"]) == ("req-train-0001", 0)
            else r for r in read_jsonl(paths["teacher"])]
    write_jsonl(paths["teacher"], rows)
    with pytest.raises(ReasoningError, match="no call example"):
        rr.pick_examples(ReasoningIndex([paths["teacher"]]), records, words)


def test_the_prompt_holds_the_examples_the_context_with_tool_outputs_cut_and_the_answer(tmp_path):
    records, index = loaded(rationalizer_world(tmp_path))
    examples = rr.pick_examples(index, records, words)
    target = rr.find_targets(records, index)[1]
    system, user = rr.build_messages(rr.system_message(examples), target)
    assert system["role"] == "system" and user["role"] == "user"
    positions = [system["content"].index(e["reasoning"]) for e in examples]
    assert positions == sorted(positions)  # in the order call, clarify, answer
    for requirement in ("English", "2 to 6 sentences", "120 words", "never contradict", "exact name",
                        "defaults are fine", "not the reply itself"):
        assert requirement in system["content"]
    text = user["content"]
    assert system_prompt(TODAY) in text and "帮我找 2026 年 9 月关于 speculative decoding 的论文" in text
    assert f"call search_arxiv {SEARCH['function']['arguments']}" in text
    hits = canonical_json(LONG_HITS)
    assert len(hits) > rr.TOOL_OUTPUT_CHARS
    assert hits[:rr.TOOL_OUTPUT_CHARS] in text and hits[:rr.TOOL_OUTPUT_CHARS + 1] not in text
    assert "[result of search_arxiv]" in text
    assert text.rstrip().endswith("Write the agent's reasoning before this reply.")
    reply = text[text.index("The agent's reply:"):]
    assert f"call get_paper {GET['function']['arguments']}" in reply
    retry = rr.build_messages(rr.system_message(examples), target, nonce=rr.nonce(target))
    assert retry[0] == system and retry[1]["content"].startswith(text) and retry[1]["content"] != text
    assert rr.nonce(target) == rr.nonce(target) and rr.nonce(target) in retry[1]["content"]


@pytest.mark.parametrize(("text", "finish", "problems"), [
    ("I will call get_paper on the first hit. The defaults are fine.", "stop", []),
    ("   ", "stop", ["empty"]),
    ("<think>hmm</think> I call get_paper.", "stop", ["think tag"]),
    ("get_paper " + "word " * 400, "stop", ["over 400 tokens"]),
    ("I will open the first paper.", "stop", ["a called tool is not named"]),
    ("I will call get_paper on", "length", ["cut at max_tokens"]),
])
def test_the_checks_of_a_written_reasoning(text, finish, problems):
    assert rr.problems(text, step(GET), words, finish) == problems


def test_a_text_answer_needs_no_tool_named():
    assert rr.problems("The abstract answers it. No call is needed.", step(content="It adapts."), words,
                       "stop") == []


# ---- the rationalizer against the strict fake endpoint ---------------------------------------------------

USAGE_COST = cost_rmb(load_prices()[MODEL], {"prompt_tokens": 1000, "completion_tokens": 50})  # one fake call


def test_a_run_retries_once_with_a_nonce_writes_what_passes_and_counts_what_does_not(tmp_path, capsys):
    paths = rationalizer_world(tmp_path)
    good = "I should call get_paper on 2609.01234 to read the first hit. The defaults are fine."
    server = StrictDeepSeek([Answer("Listing the notes is enough."), Answer(""), Answer(good)])
    assert run(paths, server) == 0
    # every request goes without thinking and without tools, at temperature 0, its answer bounded at 512
    # tokens: above the 400-token check, and the bound the dry run's ceiling is priced at
    assert [thinks(body) for body in server.bodies] == [False, False, False]
    assert all("tools" not in body and body["temperature"] == 0.0 and body["max_tokens"] == 512
               for body in server.bodies)
    first, retry, other = (body["messages"] for body in server.bodies)
    assert retry[0] == first[0] and retry[1]["content"].startswith(first[1]["content"])
    assert retry[1]["content"] != first[1]["content"]
    assert all(EXAMPLES[key] in first[0]["content"] for key in EXAMPLES)
    rows = read_jsonl(paths["out"])
    assert len(rows) == 1
    row = rows[0]
    assert {k: row[k] for k in ("id", "k", "source", "reasoning", "model", "context_equal")} == {
        "id": "req-train-0001", "k": 1, "source": "rationalized", "reasoning": good, "model": MODEL,
        "context_equal": True}
    assert row["answer_sha256"] == answer_sha256(step(GET))
    cache = ResponseCache(paths["llm"] / "cache.sqlite")
    keys = {key for (key,) in cache.db.execute("SELECT key FROM responses")}
    cache.close()
    assert row["cache_key"] == rr.request_key_of(other) and row["cache_key"] in keys and len(keys) == 3
    index = ReasoningIndex([paths["teacher"], paths["out"]])  # no turn in both files
    assert index.lookup("req-train-0001", 1, step(GET)) == good
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["targets"] == {"total": 2, "teacher_trajectory": 1, "scripted": 1}
    assert report["written"] == 1 and report["no_think"] == 1 and report["complete"] is True
    assert report["failures"] == {"empty": 1}
    assert report["first_attempt_problems"] == {"a called tool is not named": 1}
    assert report["retried"] == 1 and report["calls"] == {"paid": 3, "cached": 0}
    assert [(e["id"], e["k"]) for e in report["examples"]] == [("req-train-0001", 0), ("req-train-0002", 0),
                                                                ("req-train-0001", 2)]
    ledger = Budget(paths["llm"] / "budget.sqlite", stage="M7", stage_cap_rmb=50.0)
    assert ledger.spent("M7") == pytest.approx(3 * USAGE_COST * load_off_peak().factor)
    ledger.close()
    out = paths["out"].read_bytes()
    again = StrictDeepSeek()  # a rerun is served from the cache: no call, the same rows
    assert run(paths, again) == 0
    assert again.bodies == [] and paths["out"].read_bytes() == out
    assert json.loads(paths["report"].read_text(encoding="utf-8"))["calls"] == {"paid": 0, "cached": 3}


class RefusingFirst(StrictDeepSeek):
    """The strict endpoint, except that it refuses the first request with HTTP 400, as DeepSeek refuses a
    request it judges risky."""

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        if not self.bodies:
            self.bodies.append(json.loads(request.content))
            return httpx2.Response(400, json={"error": {"message": "Content Exists Risk",
                                                        "type": "invalid_request_error"}})
        return super().handle(request)


def test_a_request_the_api_refuses_is_a_failed_attempt_and_is_retried(tmp_path):
    paths = rationalizer_world(tmp_path)
    server = RefusingFirst([Answer("Calling list_notes with the llm tag and the date is all it takes. "
                                   "Nothing else is needed."),
                            Answer("I should call get_paper on the first hit. The defaults are fine.")])
    assert run(paths, server) == 0
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["written"] == 2 and report["first_attempt_problems"] == {"refused by the API": 1}
    assert len(server.bodies) == 3


def test_in_the_peak_hours_the_run_stops_before_any_call_and_writes_no_rows(tmp_path):
    paths = rationalizer_world(tmp_path)
    server = StrictDeepSeek([Answer("unused")])
    assert run(paths, server, at=PEAK) == 3
    assert server.bodies == [] and not paths["out"].exists()
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["complete"] is False and "peak" in report["stopped"]
    ledger = Budget(paths["llm"] / "budget.sqlite", stage="M7", stage_cap_rmb=50.0)
    assert ledger.spent() == 0.0
    ledger.close()


def test_the_max_cost_caps_the_invocation(tmp_path):
    paths = rationalizer_world(tmp_path)
    server = StrictDeepSeek([Answer("I will call list_notes for the llm tag. The defaults are fine."),
                             Answer("unused")])
    assert run(paths, server, "--max-cost", str(USAGE_COST * load_off_peak().factor / 2)) == 3
    assert len(server.bodies) == 1  # the call that reached the cap is paid and booked, then the run stops
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["complete"] is False and "M7" in report["stopped"]


class NoCalls:
    """Stands in for openai.OpenAI where nothing may be sent: any use fails the test."""

    def __getattr__(self, name: str):
        raise AssertionError(f"the dry run touched the client ({name})")


def test_the_dry_run_builds_every_prompt_and_estimates_the_cost_without_client_or_ledger(tmp_path, capsys):
    paths = rationalizer_world(tmp_path)
    assert rr.main(argv_of(paths, "--dry-run"), openai_client=NoCalls(), count_tokens=words) == 0
    assert not paths["llm"].exists() and not paths["out"].exists() and not paths["report"].exists()
    report = json.loads(paths["dry_run_report"].read_text(encoding="utf-8"))
    records, index = loaded(paths)
    system = rr.system_message(rr.pick_examples(index, records, words))
    targets = rr.find_targets(records, index)
    prompt_tokens = sum(words(m["content"]) for t in targets for m in rr.build_messages(system, t))
    assert report["targets"] == {"total": 2, "teacher_trajectory": 1, "scripted": 1}
    assert report["prompts"]["built"] == 2 and report["prompts"]["tokens"]["total"] == prompt_tokens
    price, factor = load_prices()[MODEL], load_off_peak().factor
    estimate = report["estimate"]
    assert estimate["calls"] == 2 and estimate["prompt_tokens"] == prompt_tokens
    assert estimate["completion_tokens"] == 2 * rr.EXPECTED_COMPLETION_TOKENS
    assert estimate["cost_rmb"] == pytest.approx(cost_rmb(price, {
        "prompt_tokens": prompt_tokens, "completion_tokens": 2 * rr.EXPECTED_COMPLETION_TOKENS}) * factor)
    ceiling = report["ceiling"]
    assert ceiling["calls"] == 4 and ceiling["completion_tokens"] == 4 * rr.MAX_TOKENS
    assert ceiling["cost_rmb"] > estimate["cost_rmb"]
    assert report["max_cost_rmb"] == rr.MAX_COST and report["stage"] == "M7"
    assert "2 turns" in capsys.readouterr().out


def test_the_dry_run_needs_no_api_key_since_it_makes_no_client(tmp_path, monkeypatch):
    """Without a key, making the client fails (exit 2): a dry run that succeeds made none."""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    paths = rationalizer_world(tmp_path)
    assert rr.main(argv_of(paths, "--dry-run"), count_tokens=words) == 0
    assert paths["dry_run_report"].is_file() and not paths["llm"].exists()
    assert rr.main(argv_of(paths), clock=lambda: OFF_PEAK, count_tokens=words) == 2  # a real run needs one


def test_a_dry_run_never_replaces_the_dry_run_report_of_other_records(tmp_path, capsys):
    """The defaults name M7's first pass: a dry run over another records file (M7b's) that kept them would
    replace that pass's dry-run report. It stops (exit 2) and leaves the report as it is; a rerun over the
    same records replaces it."""
    paths = rationalizer_world(tmp_path)
    assert rr.main(argv_of(paths, "--dry-run"), openai_client=NoCalls(), count_tokens=words) == 0
    before = paths["dry_run_report"].read_bytes()
    assert rr.main(argv_of(paths, "--dry-run"), openai_client=NoCalls(), count_tokens=words) == 0
    assert paths["dry_run_report"].read_bytes() == before
    records = read_jsonl(paths["records"])
    write_jsonl(paths["records"], [r for r in records if r["id"] != "req-m6-0001"])  # another file's records
    assert rr.main(argv_of(paths, "--dry-run"), openai_client=NoCalls(), count_tokens=words) == 2
    assert paths["dry_run_report"].read_bytes() == before
    assert str(paths["dry_run_report"]) in capsys.readouterr().err
    paths["dry_run_report"].write_text("not a report\n", encoding="utf-8")  # nor what this script wrote
    assert rr.main(argv_of(paths, "--dry-run"), openai_client=NoCalls(), count_tokens=words) == 2


def test_a_run_never_replaces_the_sidecar_or_report_of_other_records(tmp_path, capsys):
    """A paid run over another records file with the first pass's --out and --report stops before the ledger,
    the cache or the client is opened, and leaves both files as they are."""
    paths = rationalizer_world(tmp_path)
    server = StrictDeepSeek([Answer("Listing the notes with list_notes is all it takes. Nothing else."),
                             Answer("I should call get_paper on the first hit. The defaults are fine.")])
    assert run(paths, server) == 0
    out, report = paths["out"].read_bytes(), paths["report"].read_bytes()
    records = read_jsonl(paths["records"])
    write_jsonl(paths["records"], [r for r in records if r["id"] != "req-m6-0001"])
    llm = sorted(p.name for p in paths["llm"].iterdir())
    later = StrictDeepSeek([Answer("unused")])
    assert run(paths, later) == 2
    assert later.bodies == [] and paths["out"].read_bytes() == out and paths["report"].read_bytes() == report
    assert sorted(p.name for p in paths["llm"].iterdir()) == llm
    err = capsys.readouterr().err
    assert str(paths["out"]) in err and str(paths["report"]) in err
    paths["out"].unlink()  # the report alone, written for the other records, still stops the run
    assert run(paths, later) == 2 and later.bodies == []


def test_a_stale_teacher_sidecar_stops_the_run_before_any_call(tmp_path, capsys):
    paths = rationalizer_world(tmp_path)
    records = read_jsonl(paths["records"])
    records[0]["messages"][2] = step(SEARCH_EDITED)  # the answer the teacher row was written for changed
    write_jsonl(paths["records"], records)
    server = StrictDeepSeek()
    assert run(paths, server) == 1
    assert server.bodies == [] and "stale" in capsys.readouterr().err


def test_records_holding_test_material_are_refused_before_any_call(tmp_path, capsys):
    """The records' text is sent out: a test request's history among them stops the run (exit 1) before the
    ledger, the cache or the client is opened."""
    paths = rationalizer_world(tmp_path)
    records = read_jsonl(paths["records"])
    write_jsonl(paths["records"], [*records, record("req-test-0001", history("Find September papers"))])
    server = StrictDeepSeek()
    assert run(paths, server) == 1
    assert server.bodies == [] and not paths["llm"].exists() and not paths["out"].exists()
    err = capsys.readouterr().err
    assert "holds test material" in err and "req-test-0001" in err

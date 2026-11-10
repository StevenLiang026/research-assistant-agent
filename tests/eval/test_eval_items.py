import copy
import importlib.util
import json
import re
from pathlib import Path

import pytest
from support.fakes import HITS, PAPER, SAMPLE_MESSAGES, ScriptedModel, call, turn

from research_agent.agent.context import build_context
from research_agent.agent.loop import RELAX_HINT, save_transcript
from research_agent.agent.prompts import PROMPT_HASH, PROMPT_VERSION
from research_agent.agent.types import (
    ANSWERED,
    BUDGET_STOP,
    CLARIFICATION,
    CONTEXT_OVERFLOW,
    DEAD_LOOP,
    HARNESS_ERROR,
    INVALID_OUTPUT,
    STEP_LIMIT,
    HarnessError,
    ToolResult,
    Transcript,
)
from research_agent.canonical import canonical_json
from research_agent.eval.items import (
    CALL,
    CLARIFY,
    LANGS,
    NEXT_STEP,
    NO_CALL,
    PARALLEL,
    QUOTAS_DEV,
    QUOTAS_TEST,
    RECOVER,
    SKIP_REASONS,
    TYPES,
    draft_items,
    draftable,
    item_type,
    items_from_transcript,
    scaled,
    select,
    skip_reason,
    summarize,
)
from research_agent.eval.lock import JsonlError, read_jsonl
from research_agent.eval.runner import OK, run_items
from research_agent.tools import schemas
from research_agent.tools.schemas import MODEL_TOOLS

REPO = Path(__file__).resolve().parents[2]
TODAY = "2026-10-04"
SYSTEM, USER = SAMPLE_MESSAGES[0], SAMPLE_MESSAGES[1]
SEARCH = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-01")
RETRY = call("search_arxiv", "c3", query="speculative")
GET = call("get_paper", "c2", arxiv_id="2609.01234")
GET_SECOND = call("get_paper", "c4", arxiv_id="2609.05678")
ASK = call("ask_clarification", "c5", question="哪个月?", missing=["date_range"])
ANSWER = {"role": "assistant", "content": "找到两篇: 2609.01234 与 2609.05678。"}
THREE_STEPS = SAMPLE_MESSAGES + [ANSWER]  # a search, two parallel get_paper calls, the answer
REQUEST = {"id": "r001", "intent": "recent_papers", "scenario": "normal", "lang": "zh", "split": "dev",
           "family": 3, "text": USER["content"], "slots": {"topic": "speculative decoding"}}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load_script("build_eval_items")


def assistant(*calls, content: str = "") -> dict:
    message: dict = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [c.to_api() for c in calls]
    return message


def tool(call_id: str, name: str, content: dict | list, error: str | None = None) -> dict:
    return ToolResult(call_id, name, content, error).to_api()


def accepted(message: dict) -> dict:
    return {"model": "fake", "content": message.get("content") or "",
            "tool_calls": list(message.get("tool_calls") or []), "issues": [], "action": None}


def rejected(action: str) -> dict:
    return {"model": "fake", "content": "", "tool_calls": [call("get_paper", "c9", arxiv_id="bad").to_api()],
            "issues": [{"call_id": "c9", "event": "invalid_args", "messages": ["arxiv_id: bad"]}],
            "action": action}


def step(index: int, *attempts: dict) -> dict:
    return {"index": index, "model": "fake", "attempts": list(attempts), "executed": [], "decisions": []}


def transcript(messages: list[dict], steps: list[dict] | None = None, status: str = ANSWERED,
               schema_hash: str = schemas.SCHEMA_HASH) -> Transcript:
    """A transcript of messages; without steps, one accepted step per assistant message."""
    if steps is None:
        replies = [m for m in messages if m["role"] == "assistant"]
        steps = [step(k, accepted(m)) for k, m in enumerate(replies, 1)]
    return Transcript(request=messages[1]["content"], today=TODAY, status=status, final_text=None,
                      messages=messages, steps=steps, schema_hash=schema_hash, prompt_version=PROMPT_VERSION,
                      prompt_hash=PROMPT_HASH)


def dead_looped() -> Transcript:
    """A turn stopped on a repeated call: every step accepted and executed, the status alone says it ended."""
    return transcript(SAMPLE_MESSAGES, status=DEAD_LOOP)


def over_budget() -> Transcript:
    """A turn stopped by the budget before the model was asked for step 3: the last step has no attempt."""
    return transcript(SAMPLE_MESSAGES, [step(1, accepted(SAMPLE_MESSAGES[2])),
                                        step(2, accepted(SAMPLE_MESSAGES[4])), step(3)], status=BUDGET_STOP)


def after_failure(code: str, *second_step, content: str = "") -> list[dict]:
    """A search answered with code, then second_step (calls) or an answer as the second assistant message."""
    return [SYSTEM, USER, assistant(SEARCH), tool("c1", "search_arxiv", [], code),
            assistant(*second_step, content=content)]


def types_of(messages: list[dict], request: dict = REQUEST) -> list[str]:
    return [item["type"] for item in items_from_transcript(transcript(messages), request)]


# items_from_transcript: shape, pairing, type rules


def test_one_item_per_accepted_step_with_the_history_before_it_and_the_step_the_teacher_took():
    t = transcript(THREE_STEPS)
    items = items_from_transcript(t, REQUEST)
    assert [item["id"] for item in items] == ["r001-s1", "r001-s2", "r001-s3"]
    assert [item["messages"] for item in items] == [THREE_STEPS[:2], THREE_STEPS[:4], THREE_STEPS[:7]]
    assert items[0]["expected"] == {"content": "", "tool_calls": [
        {"name": "search_arxiv", "arguments": {"query": "speculative decoding", "date_from": "2026-09-01"},
         "arguments_json": SEARCH.arguments_json}]}
    assert [c["name"] for c in items[1]["expected"]["tool_calls"]] == ["get_paper", "get_paper"]
    assert items[2]["expected"] == {"content": ANSWER["content"], "tool_calls": []}
    assert [item["type"] for item in items] == [CALL, PARALLEL, NEXT_STEP]
    for k, item in enumerate(items, 1):
        assert (item["lang"], item["family"], item["split"]) == ("zh", 3, "dev")
        assert item["slots"] == {"topic": "speculative decoding"}
        assert item["source"] == {"request": "r001", "step": k}
        assert item["messages"][0] == SYSTEM and item["messages"][-1]["role"] in ("user", "tool")
        assert sorted(item) == ["expected", "family", "id", "lang", "messages", "slots", "source", "split",
                                "type"]
    # the items hold copies: changing them changes neither the transcript nor the request
    items[1]["messages"][3]["content"] = "changed"
    items[0]["slots"]["topic"] = "changed"
    assert t.messages == THREE_STEPS and REQUEST["slots"] == {"topic": "speculative decoding"}


def test_the_first_step_is_call_with_one_call_and_no_call_without():
    assert types_of([SYSTEM, USER, assistant(SEARCH)]) == [CALL]
    assert types_of([SYSTEM, USER, assistant(content="请告诉我论文编号。")]) == [NO_CALL]
    # a later step with one call or none is a next step, not a call or a no-call
    later = [SYSTEM, USER, assistant(SEARCH), tool("c1", "search_arxiv", HITS), assistant(GET),
             tool("c2", "get_paper", PAPER), ANSWER]
    assert types_of(later) == [CALL, NEXT_STEP, NEXT_STEP]


def test_two_or_more_calls_are_parallel_and_ask_clarification_is_clarify_whatever_else_is_called():
    assert types_of(SAMPLE_MESSAGES) == [CALL, PARALLEL]
    assert types_of([SYSTEM, USER, assistant(ASK)]) == [CLARIFY]
    assert types_of([SYSTEM, USER, assistant(GET, ASK)]) == [CLARIFY]  # clarify before parallel
    history = [SYSTEM, USER, assistant(SEARCH), tool("c1", "search_arxiv", HITS), assistant(ASK)]
    assert types_of(history) == [CALL, CLARIFY]


@pytest.mark.parametrize("code", [schemas.RATE_LIMITED, schemas.UPSTREAM_ERROR, schemas.EMPTY_RESULTS])
def test_a_step_after_one_of_the_three_failure_codes_is_recover(code):
    assert types_of(after_failure(code, RETRY)) == [CALL, RECOVER]
    assert types_of(after_failure(code, content="没有找到相关论文。")) == [CALL, RECOVER]
    assert types_of(after_failure(code, GET, GET_SECOND)) == [CALL, RECOVER]  # recover before parallel
    assert types_of(after_failure(code, ASK)) == [CALL, CLARIFY]  # clarify before recover


@pytest.mark.parametrize("code", [schemas.NO_FULLTEXT, schemas.NOT_FOUND, schemas.SKIPPED])
def test_the_other_error_codes_do_not_make_a_recover_step(code):
    assert types_of(after_failure(code, RETRY)) == [CALL, NEXT_STEP]
    assert types_of(after_failure(code, GET, GET_SECOND)) == [CALL, PARALLEL]


def test_recover_looks_at_the_step_before_only_and_at_every_tool_message_of_it():
    # the loop's EMPTY_RESULTS envelope carries a hint and wraps list content under "items"
    empty = tool("c1", "search_arxiv", {"items": [], "hint": RELAX_HINT}, schemas.EMPTY_RESULTS)
    assert json.loads(empty["content"]) == {"error": "EMPTY_RESULTS", "hint": RELAX_HINT, "items": []}
    history = [SYSTEM, USER, assistant(SEARCH), empty, assistant(GET), tool("c2", "get_paper", PAPER), ANSWER]
    assert types_of(history) == [CALL, RECOVER, NEXT_STEP]  # step 3 follows a good step
    # one failed call among parallel results is enough
    parallel = [SYSTEM, USER, assistant(GET, call("get_paper", "c3", arxiv_id="2609.05678")),
                tool("c2", "get_paper", PAPER), tool("c3", "get_paper", {}, schemas.UPSTREAM_ERROR), ANSWER]
    assert types_of(parallel) == [PARALLEL, RECOVER]
    # a tool message that is not a JSON object is not a failure
    odd = [SYSTEM, USER, assistant(SEARCH), {"role": "tool", "tool_call_id": "c1", "content": "not json"},
           assistant(GET), {"role": "tool", "tool_call_id": "c2", "content": canonical_json(["x"])}, ANSWER]
    assert types_of(odd) == [CALL, NEXT_STEP, NEXT_STEP]
    assert item_type([SYSTEM, USER], assistant(SEARCH), 1) == CALL
    assert item_type([SYSTEM, USER], assistant(content="x"), 2) == NEXT_STEP


def test_a_step_accepted_after_a_repair_gives_no_item_but_the_numbering_keeps_counting_it():
    steps = [step(1, rejected("repair"), accepted(SAMPLE_MESSAGES[2])), step(2, accepted(SAMPLE_MESSAGES[4]))]
    items = items_from_transcript(transcript(SAMPLE_MESSAGES, steps), REQUEST)
    assert [item["id"] for item in items] == ["r001-s2"]
    assert items[0]["source"] == {"request": "r001", "step": 2} and items[0]["type"] == PARALLEL
    handed_off = [step(1, rejected("handoff"), accepted(SAMPLE_MESSAGES[2])),
                  step(2, accepted(SAMPLE_MESSAGES[4]))]
    assert len(items_from_transcript(transcript(SAMPLE_MESSAGES, handed_off), REQUEST)) == 2
    both = [step(1, rejected("repair"), rejected("handoff"), accepted(SAMPLE_MESSAGES[2])),
            step(2, accepted(SAMPLE_MESSAGES[4]))]
    assert len(items_from_transcript(transcript(SAMPLE_MESSAGES, both), REQUEST)) == 2


def test_the_history_and_the_steps_must_fit_together():
    one_step = [step(1, accepted(SAMPLE_MESSAGES[2]))]
    with pytest.raises(HarnessError, match="2 assistant messages but the transcript only 1 steps"):
        items_from_transcript(transcript(SAMPLE_MESSAGES, one_step), REQUEST)
    # a last step without an accepted message is how a budget stop or an invalid output ends a turn ...
    stopped = [step(1, accepted(SAMPLE_MESSAGES[2])), step(2, accepted(SAMPLE_MESSAGES[4])), step(3),
               ]
    items = items_from_transcript(transcript(SAMPLE_MESSAGES, stopped, status=BUDGET_STOP), REQUEST)
    assert [item["id"] for item in items] == ["r001-s1", "r001-s2"]
    still_invalid = stopped[:2] + [step(3, rejected("repair"), rejected("stop"))]
    assert len(items_from_transcript(transcript(SAMPLE_MESSAGES, still_invalid), REQUEST)) == 2
    # ... but not two of them, and not an accepted step without its message
    with pytest.raises(HarnessError, match="only the last step may end without an accepted message"):
        items_from_transcript(transcript(SAMPLE_MESSAGES, stopped + [step(4)]), REQUEST)
    with pytest.raises(HarnessError, match="step 3 was accepted but the history holds no message"):
        items_from_transcript(transcript(SAMPLE_MESSAGES, stopped[:2] + [step(3, accepted(ANSWER))]), REQUEST)
    # the k-th assistant message must be what step k accepted
    swapped = [step(1, accepted(SAMPLE_MESSAGES[4])), step(2, accepted(SAMPLE_MESSAGES[2]))]
    with pytest.raises(HarnessError, match="assistant message 1 is not the accepted attempt of step 1"):
        items_from_transcript(transcript(SAMPLE_MESSAGES, swapped), REQUEST)
    with pytest.raises(HarnessError, match="system message"):
        items_from_transcript(transcript(SAMPLE_MESSAGES[1:], [step(1, accepted(SAMPLE_MESSAGES[2]))]),
                              REQUEST)


def test_a_history_that_does_not_end_with_the_user_or_a_tool_before_a_step_is_a_harness_error():
    two_in_a_row = [SYSTEM, USER, assistant(content="a"), assistant(content="b")]
    with pytest.raises(HarnessError, match="ends with a 'assistant' message, not a user or tool message"):
        items_from_transcript(transcript(two_in_a_row), REQUEST)


def test_expected_arguments_are_parsed_like_the_loop_and_kept_verbatim():
    raw = '{"query": "speculative decoding", "max_results": 3.0}'
    history = [SYSTEM, USER, assistant(call("search_arxiv", "c1", raw=raw))]
    [item] = items_from_transcript(transcript(history), REQUEST)
    assert item["expected"]["tool_calls"] == [{"name": "search_arxiv", "arguments_json": raw,
                                               "arguments": {"query": "speculative decoding",
                                                             "max_results": 3}}]
    assert isinstance(item["expected"]["tool_calls"][0]["arguments"]["max_results"], int)
    broken = [SYSTEM, USER, assistant(call("search_arxiv", "c1", raw="{"))]
    with pytest.raises(HarnessError, match="not a JSON object"):
        items_from_transcript(transcript(broken), REQUEST)


def test_request_fields_pass_through_and_slots_default_to_empty():
    bare = {"id": "r007", "lang": "en", "split": "test", "text": "Find papers on MoE routing."}
    [item] = items_from_transcript(transcript([SYSTEM, USER, assistant(SEARCH)]), bare)
    assert (item["id"], item["lang"], item["family"], item["split"], item["slots"]) == (
        "r007-s1", "en", None, "test", {})
    for key in ("id", "lang", "split"):
        with pytest.raises(HarnessError, match=f"no '{key}'"):
            items_from_transcript(transcript([SYSTEM, USER, assistant(SEARCH)]), {**bare, key: None})


# draft_items and summarize


def write_run(run_dir: Path, rows: list[tuple[dict, Transcript | None]], *,
              placeholders: bool = False) -> list[dict]:
    """A generator run directory: index.jsonl from the rows, a transcript file for each row that has one, and
    a manifest.json carrying what the drafter reads of the generator's (config.placeholders)."""
    index = []
    for request, t in rows:
        row = {**request, "status": None if t is None else t.status, "error": None, "faults": [],
               "transcript": None if t is None else f"transcripts/{request['id']}.json"}
        row.setdefault("verdict", None)
        if t is not None:
            save_transcript(t, run_dir / row["transcript"])
        index.append(row)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "index.jsonl").write_text("".join(canonical_json(r) + "\n" for r in index), encoding="utf-8")
    manifest = {"config": {"placeholders": placeholders}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                           encoding="utf-8")
    return index


def request(id_: str, split: str = "dev", lang: str = "zh", **extra) -> dict:
    return {**REQUEST, "id": id_, "split": split, "lang": lang, **extra}


def sample_run(run_dir: Path, *, placeholders: bool = False) -> list[dict]:
    failed = transcript(SAMPLE_MESSAGES, [step(1, accepted(SAMPLE_MESSAGES[2])),
                                         {**step(2, accepted(SAMPLE_MESSAGES[4])), "harness_error": "miss"}],
                        status=HARNESS_ERROR)
    rows = [
        (request("r001", verdict={"accepted": True, "problems": []}), transcript(THREE_STEPS)),
        (request("r002", "test", "en"), transcript([SYSTEM, USER, assistant(ASK)], status=CLARIFICATION)),
        (request("r003"), failed),
        (request("r004", verdict={"accepted": False, "problems": ["an id the tools never returned"]}),
         transcript(SAMPLE_MESSAGES)),
        (request("r005", "train", verdict={"accepted": True, "problems": []}), transcript(SAMPLE_MESSAGES)),
        (request("r006"), None),
        (request("r007", "test", "en"), transcript([SYSTEM, USER, assistant(content="No paper id given.")])),
        (request("r008"), dead_looped()),
        (request("r009", "test", "en"), over_budget()),
    ]
    return write_run(run_dir, rows, placeholders=placeholders)


def test_draft_items_reads_the_index_and_drafts_what_is_draftable_in_index_order(tmp_path):
    index = sample_run(tmp_path)
    items = draft_items(tmp_path)
    assert [item["id"] for item in items] == ["r001-s1", "r001-s2", "r001-s3", "r002-s1", "r007-s1"]
    assert [item["type"] for item in items] == [CALL, PARALLEL, NEXT_STEP, CLARIFY, NO_CALL]
    assert [item["split"] for item in items] == ["dev", "dev", "dev", "test", "test"]
    assert items[3]["messages"] == [SYSTEM, USER] and items[3]["lang"] == "en"
    assert [item["id"] for item in draft_items(tmp_path, include_unvalidated=False)] == ["r001-s1", "r001-s2",
                                                                                         "r001-s3"]
    assert [item["id"] for item in draft_items(tmp_path, splits=("test",))] == ["r002-s1", "r007-s1"]
    assert [item["id"] for item in draft_items(tmp_path, splits=("train",))] == ["r005-s1", "r005-s2"]
    assert [draftable(row) for row in index] == [True, True, False, False, False, False, True, False, False]
    assert [skip_reason(row) for row in index] == [None, None, "harness error", "rejected", "other split",
                                                   "no transcript", None, "dead loop", "budget stop"]
    assert [skip_reason(row, include_unvalidated=False) for row in index] == [
        None, "not validated", "harness error", "rejected", "other split", "no transcript", "not validated",
        "not validated", "not validated"]
    assert draftable(index[4], splits=("train", "dev"))
    assert skip_reason({**index[0], "split": None}) == "other split"
    assert skip_reason({**index[0], "transcript": ""}) == "no transcript"
    assert skip_reason({**index[0], "verdict": {"problems": []}}) == "rejected"  # no "accepted" key
    assert SKIP_REASONS == ("other split", "no transcript", "harness error", "dead loop", "step limit",
                            "budget stop", "invalid output", "context overflow", "unknown status", "rejected",
                            "not validated")


def test_an_unvalidated_turn_is_drafted_only_when_it_ended_answered_or_as_a_clarification(tmp_path):
    ok = {"accepted": True, "problems": []}
    index = write_run(tmp_path, [(request("r001"), dead_looped()), (request("r002"), over_budget()),
                                 (request("r003", verdict=ok), dead_looped()),
                                 (request("r004", verdict=ok), over_budget())])
    # without a verdict nothing tells the good steps of a stopped turn from the bad: the turn is left out ...
    assert [skip_reason(row) for row in index] == ["dead loop", "budget stop", None, None]
    assert [draftable(row) for row in index] == [False, False, True, True]
    # ... but an accepted verdict keeps it whatever its status, the trailing step without an attempt included
    assert [item["id"] for item in draft_items(tmp_path)] == ["r003-s1", "r003-s2", "r004-s1", "r004-s2"]
    for status, reason in ((STEP_LIMIT, "step limit"), (INVALID_OUTPUT, "invalid output"),
                           (CONTEXT_OVERFLOW, "context overflow")):
        assert skip_reason({**index[0], "status": status}) == reason
    assert skip_reason({**index[0], "status": "odd"}) == "unknown status"
    assert skip_reason({**index[0], "status": None}) == "unknown status"
    # a rejected verdict wins over any status, and --validated-only never gets as far as the status
    assert skip_reason({**index[0], "verdict": {"accepted": False, "problems": ["x"]}}) == "rejected"
    assert [skip_reason(row, include_unvalidated=False) for row in index] == [
        "not validated", "not validated", None, None]


def test_draft_items_refuses_malformed_input(tmp_path):
    with pytest.raises(FileNotFoundError):
        draft_items(tmp_path)
    write_run(tmp_path, [(request("r001"), transcript(SAMPLE_MESSAGES))])
    (tmp_path / "index.jsonl").write_text('{"id": "r001"}\n[1, 2]\n', encoding="utf-8")
    with pytest.raises(JsonlError, match="line 2"):
        draft_items(tmp_path)
    write_run(tmp_path, [(request("r001"), transcript(SAMPLE_MESSAGES, schema_hash="0" * 16))])
    with pytest.raises(HarnessError, match="tool schema 0000000000000000"):
        draft_items(tmp_path)
    write_run(tmp_path, [(request("r001"), transcript(SAMPLE_MESSAGES))])
    path = tmp_path / "transcripts" / "r001.json"
    with_extra = {**json.loads(path.read_text(encoding="utf-8")), "extra": 1}
    path.write_text(json.dumps(with_extra), encoding="utf-8")
    with pytest.raises(TypeError, match="extra"):
        draft_items(tmp_path)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        draft_items(tmp_path)


def test_summarize_counts_by_split_type_and_lang_with_every_cell_present(tmp_path):
    sample_run(tmp_path)
    summary = summarize(draft_items(tmp_path))
    assert list(summary) == ["dev", "test"]
    assert list(summary["dev"]) == list(TYPES) and list(summary["dev"][CALL]) == list(LANGS)
    assert summary["dev"][CALL] == {"zh": 1, "en": 0} and summary["dev"][PARALLEL] == {"zh": 1, "en": 0}
    assert summary["test"][CLARIFY] == {"zh": 0, "en": 1} and summary["test"][NO_CALL] == {"zh": 0, "en": 1}
    assert sum(n for types in summary.values() for counts in types.values() for n in counts.values()) == 5
    odd = summarize([{"split": "dev", "type": CALL, "lang": "fr"}])
    assert list(odd["dev"][CALL]) == ["zh", "en", "fr"] and odd["dev"][CALL]["fr"] == 1
    assert summarize([]) == {}


# select, quotas and scaled


def pool(type_: str, lang: str, n: int, prefix: str = "") -> list[dict]:
    return [{"id": f"{prefix}{type_}-{lang}-{i:02d}", "type": type_, "lang": lang} for i in range(n)]


def test_select_draws_per_type_and_language_and_accounts_for_every_shortfall():
    items = pool(CALL, "zh", 10) + pool(CALL, "en", 3) + pool(CLARIFY, "zh", 1) + pool(NO_CALL, "en", 4)
    chosen, report = select(items, {CALL: 7, CLARIFY: 3}, seed=1)
    assert report == {
        CALL: {"zh": {"quota": 4, "available": 10, "taken": 4, "shortfall": 0},
               "en": {"quota": 3, "available": 3, "taken": 3, "shortfall": 0}},
        CLARIFY: {"zh": {"quota": 2, "available": 1, "taken": 1, "shortfall": 1},
                  "en": {"quota": 1, "available": 0, "taken": 0, "shortfall": 1}},
    }
    assert len(chosen) == 8 and all(item in items for item in chosen)
    strata = [(c["type"], c["lang"]) for c in chosen]
    assert strata == [(CALL, "zh")] * 4 + [(CALL, "en")] * 3 + [(CLARIFY, "zh")]
    zh = [c["id"] for c in chosen[:4]]
    assert zh == sorted(zh) and len(set(zh)) == 4  # without replacement, sorted within the stratum
    assert NO_CALL not in report and not any(c["type"] == NO_CALL for c in chosen)  # no quota: left out
    # an odd quota gives the first language the extra one; the langs order decides which that is
    _, odd = select(items, {CALL: 5}, seed=1)
    assert (odd[CALL]["zh"]["quota"], odd[CALL]["en"]["quota"]) == (3, 2)
    _, swapped = select(items, {CALL: 5}, seed=1, langs=("en", "zh"))
    assert (swapped[CALL]["en"]["quota"], swapped[CALL]["zh"]["quota"]) == (3, 2)
    assert select([], {CALL: 2}, seed=1) == ([], {CALL: {"zh": {"quota": 1, "available": 0, "taken": 0,
                                                                 "shortfall": 1},
                                                          "en": {"quota": 1, "available": 0, "taken": 0,
                                                                 "shortfall": 1}}})


def test_select_is_reproducible_from_the_seed_and_one_stratum_does_not_move_another():
    items = pool(CALL, "zh", 10) + pool(CALL, "en", 3)
    first, _ = select(items, {CALL: 7}, seed=1)
    again, _ = select(list(reversed(items)), {CALL: 7}, seed=1)  # input order does not matter
    assert first == again
    other, _ = select(items, {CALL: 7}, seed=2)
    assert [c["id"] for c in other[:4]] != [c["id"] for c in first[:4]]
    grown, _ = select(items + pool(CALL, "en", 5, "more-"), {CALL: 7}, seed=1)
    assert grown[:4] == first[:4]  # more English candidates: the Chinese draw stays
    with pytest.raises(ValueError, match="distinct; repeated"):
        select(items + items[:1], {CALL: 7}, seed=1)
    with pytest.raises(ValueError, match="at least one language"):
        select(items, {CALL: 7}, seed=1, langs=())


def test_quotas_follow_the_plan_and_scaled_keeps_the_totals():
    assert QUOTAS_TEST == {CALL: 120, NEXT_STEP: 30, NO_CALL: 50, CLARIFY: 40, RECOVER: 40, PARALLEL: 20}
    assert sum(QUOTAS_TEST.values()) == 300 and tuple(QUOTAS_TEST) == TYPES
    assert sum(QUOTAS_DEV.values()) == 150 and QUOTAS_DEV == scaled(QUOTAS_TEST, 0.5)
    assert QUOTAS_DEV == {CALL: 60, NEXT_STEP: 15, NO_CALL: 25, CLARIFY: 20, RECOVER: 20, PARALLEL: 10}
    assert scaled(QUOTAS_TEST, 0.7) == {CALL: 84, NEXT_STEP: 21, NO_CALL: 35, CLARIFY: 28, RECOVER: 28,
                                        PARALLEL: 14}
    dev70 = scaled(QUOTAS_DEV, 0.7)
    assert sum(dev70.values()) == 105 == round(150 * 0.7)
    assert dev70 == {CALL: 42, NEXT_STEP: 11, NO_CALL: 17, CLARIFY: 14, RECOVER: 14, PARALLEL: 7}
    assert scaled(QUOTAS_TEST, 1.0) == QUOTAS_TEST and scaled(QUOTAS_TEST, 1) == QUOTAS_TEST
    assert scaled(QUOTAS_TEST, 0) == dict.fromkeys(QUOTAS_TEST, 0)
    assert scaled({"a": 1, "b": 1, "c": 1}, 0.4) == {"a": 1, "b": 0, "c": 0}  # 1.2 rounds to 1: the first
    assert scaled({"a": 3, "b": 3}, 0.5) == {"a": 2, "b": 1}  # 1.5 + 1.5 = 3: not 4, not 2
    assert scaled({}, 0.7) == {}
    for share in (-0.1, 1.5):
        with pytest.raises(ValueError, match="share"):
            scaled(QUOTAS_TEST, share)


# the drafted items run through the runner, and the CLI


def test_drafted_items_run_through_the_evaluation_runner():
    items = items_from_transcript(transcript(THREE_STEPS), REQUEST)
    model = ScriptedModel([turn(SEARCH), turn(GET), turn(content="ok")])
    results = run_items(items, model, tools=MODEL_TOOLS)
    assert [r["status"] for r in results] == [OK, OK, OK]
    assert [r["id"] for r in results] == ["r001-s1", "r001-s2", "r001-s3"]
    assert results[2]["item"]["expected"] == items[2]["expected"]
    assert results[1]["item"]["type"] == PARALLEL and results[1]["item"]["source"] == {"request": "r001",
                                                                                       "step": 2}
    assert model.contexts == [build_context(item["messages"]) for item in items]
    assert model.contexts[2][3]["content"] != THREE_STEPS[3]["content"]  # the runner compresses older steps


def test_the_cli_writes_both_drafts_prints_the_shortfalls_and_is_deterministic(tmp_path, capsys):
    sample_run(tmp_path / "run")
    out = tmp_path / "eval"
    argv = [str(tmp_path / "run"), "--out-dir", str(out), "--seed", "7", "--share", "1.0"]
    assert build.main(argv) == 0
    printed = capsys.readouterr()
    assert printed.err == ""
    assert "candidates: 5 items drafted" in printed.out
    assert ("index: 9 requests, 3 drafted; left out: other split 1, no transcript 1, harness error 1, "
            "dead loop 1, budget stop 1, rejected 1") in printed.out
    assert re.search(r"dev\s+call\s+1\s+0\s+1", printed.out) and "shortfall" in printed.out
    assert re.search(r"test\s+clarify\s+en\s+quota\s+20\s+available\s+1\s+taken\s+1\s+shortfall\s+19",
                     printed.out)
    assert "total shortfall 147" in printed.out and "total shortfall 298" in printed.out
    dev, test = read_jsonl(out / "dev-draft.jsonl"), read_jsonl(out / "test-draft.jsonl")
    # the drafts are grouped by stratum in the plan's order: call, next_step, no_call, clarify, ..., parallel
    assert [item["id"] for item in dev] == ["r001-s1", "r001-s3", "r001-s2"]
    assert [item["id"] for item in test] == ["r007-s1", "r002-s1"]
    assert all(sorted(item) == ["expected", "family", "id", "lang", "messages", "slots", "source", "split",
                                "type"] for item in dev + test)
    first = (out / "dev-draft.jsonl").read_bytes(), (out / "test-draft.jsonl").read_bytes()
    assert build.main(argv) == 0
    assert ((out / "dev-draft.jsonl").read_bytes(), (out / "test-draft.jsonl").read_bytes()) == first
    assert b"\r" not in first[0] and first[0].endswith(b"\n")
    assert build.main([*argv, "--validated-only"]) == 0
    assert read_jsonl(out / "test-draft.jsonl") == []  # r002 and r007 have no verdict yet
    assert len(read_jsonl(out / "dev-draft.jsonl")) == 3
    left_out = "left out: other split 1, no transcript 1, harness error 1, rejected 1, not validated 4"
    assert f"index: 9 requests, 1 drafted; {left_out}" in capsys.readouterr().out


def test_the_cli_fills_the_share_of_the_quotas_and_the_defaults_follow_the_plan(tmp_path, capsys):
    args = build.build_parser().parse_args(["runs/m2/synth/x"])
    assert (args.out_dir, args.seed, args.share, args.include_unvalidated) == (Path("data/eval"), 42, 0.7,
                                                                                 True)
    assert build.build_parser().parse_args(["r", "--validated-only"]).include_unvalidated is False
    run_dir = tmp_path / "run"
    one_call = [SYSTEM, USER, assistant(SEARCH)]
    rows = [(request(f"r{i:03d}", "test", "zh" if i % 2 else "en"), transcript(one_call)) for i in range(100)]
    write_run(run_dir, rows)
    assert build.main([str(run_dir), "--out-dir", str(tmp_path / "eval"), "--share", "0.1"]) == 0
    printed = capsys.readouterr().out
    test = read_jsonl(tmp_path / "eval" / "test-draft.jsonl")
    assert len(test) == 12 and {item["type"] for item in test} == {CALL}  # 120 * 0.1, 6 per language
    assert sum(item["lang"] == "zh" for item in test) == 6
    assert "total shortfall 18" in printed  # the other five types: 3 + 5 + 4 + 4 + 2


def test_the_cli_fails_on_malformed_input_only(tmp_path, capsys):
    assert build.main([str(tmp_path / "nowhere")]) == 2
    assert "index.jsonl" in capsys.readouterr().err
    run_dir = tmp_path / "run"
    write_run(run_dir, [(request("r001"), transcript(SAMPLE_MESSAGES))])
    (run_dir / "index.jsonl").write_text('{"id": "r001"}\n[1, 2]\n', encoding="utf-8")
    assert build.main([str(run_dir), "--out-dir", str(tmp_path / "eval")]) == 1
    assert "FAIL: JsonlError" in capsys.readouterr().err and not (tmp_path / "eval").exists()
    one_step_short = transcript(SAMPLE_MESSAGES, [step(1, accepted(SAMPLE_MESSAGES[2]))])
    write_run(run_dir, [(request("r001"), one_step_short)])
    assert build.main([str(run_dir), "--out-dir", str(tmp_path / "eval")]) == 1
    assert "FAIL: HarnessError" in capsys.readouterr().err
    write_run(run_dir, [(request("r001"), transcript(SAMPLE_MESSAGES))])  # no shortfall is a failure
    assert build.main([str(run_dir), "--out-dir", str(tmp_path / "eval")]) == 0
    printed = capsys.readouterr().out  # dev call: quota 60 at 70 % is 42, 21 per language
    assert re.search(r"dev\s+call\s+zh\s+quota\s+21\s+available\s+1\s+taken\s+1\s+shortfall\s+20  <- short",
                     printed)
    assert re.search(r"dev\s+parallel\s+zh\s+quota\s+4\s+available\s+1\s+taken\s+1\s+shortfall\s+3  <- short",
                     printed)


# the placeholder guard: a run made with --placeholders is for debugging only (docs/modules/接口.md)


def test_the_cli_refuses_a_placeholder_run_with_exit_3_unless_allowed_and_writes_nothing(tmp_path, capsys):
    run_dir, out = tmp_path / "run", tmp_path / "eval"
    sample_run(run_dir, placeholders=True)
    argv = [str(run_dir), "--out-dir", str(out), "--share", "1.0"]
    assert build.main(argv) == 3
    printed = capsys.readouterr()
    assert printed.out == "" and not out.exists()
    assert printed.err.count("\n") == 1 and printed.err.endswith("\n")  # one line
    assert str(run_dir) in printed.err and "generated with --placeholders" in printed.err
    assert "debugging only" in printed.err and "--allow-placeholders" in printed.err
    # --allow-placeholders drafts as usual and notes the placeholder run once, on stdout
    assert build.main([*argv, "--allow-placeholders"]) == 0
    printed = capsys.readouterr()
    assert printed.err == ""
    notes = [line for line in printed.out.splitlines() if line.startswith("note:")]
    assert len(notes) == 1 and str(run_dir) in notes[0] and "--placeholders" in notes[0]
    assert "candidates: 5 items drafted" in printed.out
    assert len(read_jsonl(out / "dev-draft.jsonl")) == 3 and len(read_jsonl(out / "test-draft.jsonl")) == 2
    # a run without placeholders prints no note and needs no flag
    sample_run(run_dir)
    assert build.main(argv) == 0
    assert not any(line.startswith("note:") for line in capsys.readouterr().out.splitlines())
    assert build.build_parser().parse_args(["r"]).allow_placeholders is False
    assert build.build_parser().parse_args(["r", "--allow-placeholders"]).allow_placeholders is True


def test_a_run_without_a_manifest_is_malformed_and_the_message_names_the_missing_file(tmp_path, capsys):
    run_dir, out = tmp_path / "run", tmp_path / "eval"
    sample_run(run_dir)
    (run_dir / "manifest.json").unlink()
    assert build.main([str(run_dir), "--out-dir", str(out)]) == 1
    printed = capsys.readouterr()
    assert printed.out == "" and not out.exists()
    assert "FAIL: FileNotFoundError" in printed.err and str(run_dir / "manifest.json") in printed.err
    # a manifest that is not JSON, not a JSON object, or without a true-or-false config.placeholders is
    # malformed too
    (run_dir / "manifest.json").write_text("{\n", encoding="utf-8")
    assert build.main([str(run_dir), "--out-dir", str(out)]) == 1
    assert "FAIL: ValueError" in capsys.readouterr().err
    (run_dir / "manifest.json").write_text("[]\n", encoding="utf-8")
    assert build.main([str(run_dir), "--out-dir", str(out)]) == 1
    assert "FAIL: TypeError" in capsys.readouterr().err
    (run_dir / "manifest.json").write_text('{"config": {"placeholders": "no"}}\n', encoding="utf-8")
    assert build.main([str(run_dir), "--out-dir", str(out)]) == 1
    assert "config.placeholders" in capsys.readouterr().err and not out.exists()
    # the index check still comes first: a folder with a manifest but no index.jsonl is exit 2
    (run_dir / "index.jsonl").unlink()
    assert build.main([str(run_dir), "--out-dir", str(out)]) == 2


def test_build_eval_items_carries_no_machine_path():
    source = (REPO / "scripts" / "build_eval_items.py").read_text(encoding="utf-8")
    assert re.search(r"[A-Za-z]:\\", source) is None  # a drive letter: an absolute Windows path


def test_items_are_json_serialisable_and_copyable():
    items = items_from_transcript(transcript(THREE_STEPS), REQUEST)
    assert json.loads(canonical_json(items)) == items
    assert copy.deepcopy(items) == items


# several run directories: the candidates are the union of their drafts


def split_runs(tmp_path: Path, *, placeholders_in_test: bool = False) -> tuple[Path, Path]:
    """A run of dev requests only and one of test requests only, as the full generation left them."""
    accepted_verdict = {"accepted": True, "problems": []}
    dev_dir, test_dir = tmp_path / "dev", tmp_path / "test"
    write_run(dev_dir, [(request("d001", verdict=accepted_verdict), transcript(THREE_STEPS)),
                        (request("d002", "dev", "en"), transcript([SYSTEM, USER, assistant(SEARCH)]))])
    write_run(test_dir, [(request("e001", "test", "en", verdict=accepted_verdict),
                          transcript([SYSTEM, USER, assistant(ASK)], status=CLARIFICATION)),
                         (request("e002", "test", "zh"), dead_looped())], placeholders=placeholders_in_test)
    return dev_dir, test_dir


def test_the_cli_drafts_the_union_of_several_runs_and_prints_the_index_line_per_run_and_in_total(tmp_path,
                                                                                                 capsys):
    dev_dir, test_dir = split_runs(tmp_path)
    out = tmp_path / "eval"
    assert build.main([str(dev_dir), str(test_dir), "--out-dir", str(out), "--share", "1.0"]) == 0
    printed = capsys.readouterr().out
    assert [item["id"] for item in read_jsonl(out / "dev-draft.jsonl")] == ["d001-s1", "d002-s1", "d001-s3",
                                                                            "d001-s2"]
    assert [item["id"] for item in read_jsonl(out / "test-draft.jsonl")] == ["e001-s1"]
    assert "candidates: 5 items drafted from 2 runs" in printed
    assert f"index {dev_dir}: 2 requests, 2 drafted; left out: none" in printed
    assert f"index {test_dir}: 2 requests, 1 drafted; left out: dead loop 1" in printed
    assert "index: 4 requests, 3 drafted; left out: dead loop 1" in printed
    assert build.main([str(dev_dir), str(test_dir), "--out-dir", str(out), "--validated-only"]) == 0
    assert "index: 4 requests, 2 drafted; left out: not validated 2" in capsys.readouterr().out
    assert build.build_parser().parse_args(["a", "b"]).run_dirs == [Path("a"), Path("b")]


def test_a_placeholder_run_among_several_is_refused_with_exit_3_unless_allowed(tmp_path, capsys):
    dev_dir, test_dir = split_runs(tmp_path, placeholders_in_test=True)
    out = tmp_path / "eval"
    assert build.main([str(dev_dir), str(test_dir), "--out-dir", str(out)]) == 3
    printed = capsys.readouterr()
    assert str(test_dir) in printed.err and "--placeholders" in printed.err and not out.exists()
    assert build.main([str(dev_dir), str(test_dir), "--out-dir", str(out), "--allow-placeholders"]) == 0
    notes = [line for line in capsys.readouterr().out.splitlines() if line.startswith("note:")]
    assert len(notes) == 1 and str(test_dir) in notes[0]
    assert build.main([str(dev_dir), str(tmp_path / "nowhere"), "--out-dir", str(tmp_path / "e2")]) == 2
    assert not (tmp_path / "e2").exists()

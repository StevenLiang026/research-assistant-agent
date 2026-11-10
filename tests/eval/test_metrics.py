import importlib.util
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
from support.fakes import call, turn

from research_agent.agent.prompts import system_prompt
from research_agent.agent.types import BUDGET_STOP, CONTEXT_OVERFLOW, ToolCall
from research_agent.eval.items import CALL, CLARIFY, NEXT_STEP, NO_CALL, PARALLEL, RECOVER
from research_agent.eval.lock import LockedSet, RefusedTestSet, write_jsonl, write_lock
from research_agent.eval.metrics import (
    BAD_RECOVERY,
    BROKEN_JSON,
    COMPARED,
    COUNT_PARAMS,
    DEFAULT_SCORER,
    ERROR_CLASSES,
    EXTRA_PARAM,
    FABRICATED_ID,
    MISSING_PARAM,
    NO_CLARIFY,
    NORMALIZED_SCORER,
    PRECEDENCE,
    RETRY_STORM,
    SCORERS,
    THINK_LEAK,
    UNWANTED_CALL,
    WRONG_TOOL,
    WRONG_VALUE,
    IdMismatch,
    arxiv_ids,
    bootstrap_ci,
    cluster_bootstrap_ci,
    mcnemar,
    numbers_in,
    request_text,
    score_item,
    score_items,
    summarize,
    today_of,
)
from research_agent.eval.runner import ERROR, OK, SKIPPED, run_items
from research_agent.validation import check_call

REPO = Path(__file__).resolve().parents[2]
SYSTEM = {"role": "system", "content": "You are a research assistant."}
USER = {"role": "user", "content": "找 2026 年 9 月以来 speculative decoding 的论文。"}
TOPIC = {"topic": "speculative decoding", "date_from": "2026-09-01"}
SEARCH = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-01")
GET_A = call("get_paper", "c2", arxiv_id="2609.01234")
GET_B = call("get_paper", "c3", arxiv_id="2609.05678")
HITS = {"role": "tool", "tool_call_id": "c1",
        "content": '[{"arxiv_id":"2609.01234","title":"A"},{"arxiv_id":"2609.05678","title":"B"}]'}
AFTER_SEARCH = [SYSTEM, USER, {"role": "assistant", "content": "", "tool_calls": [SEARCH.to_api()]}, HITS]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score_cli = load_script("score")


def expected(*calls, content: str = "") -> dict:
    return {"content": content, "tool_calls": [{"name": c.name, "arguments": c.arguments,
                                                "arguments_json": c.arguments_json} for c in calls]}


def item(type_: str, *calls, messages: list[dict] | None = None, slots: dict | None = None, id_: str = "i1",
         lang: str = "zh", family: str = "f1") -> dict:
    """An item in the drafter's format (docs/modules/接口.md, 评测题)."""
    return {"id": id_, "messages": messages or [SYSTEM, USER], "expected": expected(*calls), "type": type_,
            "lang": lang, "family": family, "split": "dev", "slots": TOPIC if slots is None else slots,
            "source": {"request": id_.split("-")[0], "step": 1}}


def ok(*calls, content: str = "", id_: str = "i1") -> dict:
    return {"id": id_, "status": OK, "item": {}, "output": turn(*calls, content=content).to_api()}


def score(it: dict, *calls, content: str = "") -> dict:
    return score_item(ok(*calls, content=content, id_=it["id"]), it)


# ---- single checks ---------------------------------------------------------------------------------------

def test_the_expected_call_scores_every_applying_check_true_and_succeeds():
    s = score(item(CALL, SEARCH), SEARCH)
    assert s["success"] is True and s["error_class"] is None
    assert {k: s[k] for k in ("json_valid", "tool_choice", "arguments", "exact_args", "query_strict",
                              "ids_grounded")} == dict.fromkeys(("json_valid", "tool_choice", "arguments",
                                                                 "exact_args", "query_strict",
                                                                 "ids_grounded"), True)
    assert s["no_false_call"] is None and s["clarify"] is None and s["recover"] is None
    assert s["params"]["f1"] == 1.0


def test_arguments_that_are_not_a_json_object_fail_json_valid_and_are_broken_json():
    s = score(item(CALL, SEARCH), call("search_arxiv", "c1", raw='{"query": "speculative'))
    assert s["json_valid"] is False and s["success"] is False and s["error_class"] == BROKEN_JSON


def test_tool_choice_compares_the_sets_of_called_names():
    assert score(item(CALL, SEARCH), SEARCH)["tool_choice"] is True
    s = score(item(CALL, SEARCH), GET_A)
    assert s["tool_choice"] is False and s["error_class"] == WRONG_TOOL


def test_strings_ids_and_dates_are_compared_after_normalisation():
    want = call("fetch_fulltext", "c4", arxiv_id="2609.01234", section="method")
    got = call("fetch_fulltext", "c4", arxiv_id=" arXiv:2609.01234v2 ", section="ｍｅｔｈｏｄ")
    assert score(item(NEXT_STEP, want, messages=AFTER_SEARCH), got)["exact_args"] is True
    dated = call("search_arxiv", "c1", query="speculative decoding", date_from="2026/9/1")
    assert score(item(CALL, SEARCH), dated)["exact_args"] is True
    other = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-02")
    assert score(item(CALL, SEARCH), other)["arguments"] is False


def test_a_parameter_left_at_its_schema_default_counts_as_filled_with_the_default():
    explicit = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-01", max_results=5,
                    sort_by="relevance")
    assert score(item(CALL, explicit), SEARCH)["exact_args"] is True
    assert score(item(CALL, SEARCH), explicit)["exact_args"] is True
    fewer = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-01", max_results=3.0)
    s = score(item(CALL, SEARCH), fewer)
    assert s["exact_args"] is False and s["error_class"] == WRONG_VALUE


def test_the_query_is_judged_by_the_required_words_and_the_literal_match_is_reported_apart():
    looser = call("search_arxiv", "c1", query="Speculative  Decoding LLM 2026", date_from="2026-09-01")
    s = score(item(CALL, SEARCH), looser)
    assert s["success"] is True and s["exact_args"] is True and s["query_strict"] is False
    partial = call("search_arxiv", "c1", query="decoding", date_from="2026-09-01")
    assert score(item(CALL, SEARCH), partial)["arguments"] is False


def test_a_query_without_an_applying_topic_must_equal_the_expected_one_up_to_case():
    want = call("search_arxiv", "c1", query="LLM judge evaluation")
    slots = {"topic": "LLM-as-a-judge"}  # the teacher rephrased: the topic is not in its query
    same = call("search_arxiv", "c1", query="llm JUDGE evaluation")
    assert score(item(CALL, want, slots=slots), same)["arguments"] is True
    assert score(item(CALL, want, slots=slots), call("search_arxiv", "c1", query="LLM judge"))["arguments"] \
        is False


def test_required_words_given_in_the_slots_take_precedence_over_the_topic():
    slots = {"topic": "speculative decoding", "required_words": ["draft"]}
    want = call("search_arxiv", "c1", query="speculative decoding draft model")
    got = call("search_arxiv", "c1", query="draft")
    assert score(item(CALL, want, slots=slots), got)["arguments"] is True


def test_categories_and_tags_compare_as_sets():
    want = call("search_arxiv", "c1", query="speculative decoding", categories=["cs.CL", "cs.LG"])
    got = call("search_arxiv", "c1", query="speculative decoding", categories=["cs.LG", "cs.CL"])
    assert score(item(CALL, want), got)["exact_args"] is True
    notes = call("search_notes", "c1", query="speculative decoding", tags=["llm", "decoding"])
    assert score(item(CALL, notes), call("search_notes", "c1", query="speculative decoding",
                                         tags=["decoding", "llm"]))["exact_args"] is True
    assert score(item(CALL, notes), call("search_notes", "c1", query="speculative decoding",
                                         tags=["decoding"]))["exact_args"] is False


def test_parallel_calls_are_compared_as_a_multiset():
    it = item(PARALLEL, GET_A, GET_B, messages=AFTER_SEARCH)
    assert score(it, GET_B, GET_A)["success"] is True
    twice = score(it, GET_A, GET_A)
    assert twice["exact_args"] is False and twice["tool_choice"] is True
    assert twice["error_class"] == WRONG_VALUE
    extra = score(it, GET_A, GET_B, GET_B)
    assert extra["arguments"] is True and extra["exact_args"] is False and extra["error_class"] == WRONG_TOOL


def test_no_call_items_succeed_without_a_call_and_count_a_call_as_a_false_call():
    it = item(NO_CALL, slots={})
    s = score(it, content="四个部分是 problem、method、evidence、limitations。")
    assert s["success"] is True and s["no_false_call"] is True and s["arguments"] is None
    s = score(it, SEARCH)
    assert s["no_false_call"] is False and s["tool_choice"] is False and s["error_class"] == UNWANTED_CALL


def test_clarify_needs_ask_clarification_with_the_expected_missing_set_and_ignores_the_question():
    want = call("ask_clarification", "c5", question="哪个主题?", missing=["topic", "date_range"])
    it = item(CLARIFY, want, slots={})
    reordered = call("ask_clarification", "c5", question="What topic, and which months?",
                     missing=["date_range", "topic"])
    s = score(it, reordered)
    assert s["clarify"] is True and s["success"] is True and s["exact_args"] is None
    s = score(it, call("ask_clarification", "c5", question="哪个主题?", missing=["topic"]))
    assert s["clarify"] is False and s["error_class"] == WRONG_VALUE
    loose = item(CLARIFY, call("ask_clarification", "c5", question="哪个主题?"), slots={})
    assert score(loose, reordered)["clarify"] is True


def test_a_clarify_item_expects_the_tool_set_of_ask_clarification_alone():
    want = call("ask_clarification", "c5", question="哪个主题?", missing=["topic"])
    it = item(CLARIFY, want, slots={})
    s = score(it, SEARCH)
    assert s["tool_choice"] is False and s["clarify"] is False and s["error_class"] == NO_CLARIFY
    both = score(it, want, SEARCH)
    assert both["tool_choice"] is False and both["clarify"] is True and both["success"] is False


# A scaffold note as the teacher wrote it: its evidence paraphrases the paper, so a model that quotes the
# source never matches it word for word.
NOTE = call("write_note", "c7", arxiv_id="2609.01234", mode="scaffold",
            evidence='"A small draft model proposes tokens." / "The target model checks them in one pass."',
            new_terms=["draft model"], open_terms=["acceptance rate"])


def test_write_note_is_judged_by_its_id_and_mode_and_its_free_text_is_left_out():
    quoted = call("write_note", "c7", arxiv_id="2609.01234", problem="Decoding one token at a time is slow.",
                  evidence='"We verify all drafted tokens in a single forward pass."', new_terms=["verifier"])
    s = score(item(NEXT_STEP, NOTE, messages=AFTER_SEARCH), quoted)
    assert s["success"] is True and s["exact_args"] is True and s["error_class"] is None
    assert s["param_counts"] == {"write_note.arxiv_id": [1, 0, 0], "write_note.mode": [1, 0, 0]}


def test_write_note_compares_only_its_id_and_mode_as_ask_clarification_compares_only_missing():
    assert COMPARED == {"ask_clarification": {"missing"}, "write_note": {"arxiv_id", "mode"}}


def test_write_note_on_another_paper_in_another_mode_or_without_an_id_still_fails():
    it = item(NEXT_STEP, NOTE, messages=AFTER_SEARCH)
    no_id = {k: v for k, v in NOTE.arguments.items() if k != "arxiv_id"}
    for arguments, error_class in (({**NOTE.arguments, "arxiv_id": "2609.05678"}, WRONG_VALUE),
                                   ({**NOTE.arguments, "mode": "draft"}, WRONG_VALUE),
                                   (no_id, MISSING_PARAM)):
        s = score(it, call("write_note", "c7", **arguments))
        assert s["exact_args"] is False and s["error_class"] == error_class


def test_a_write_note_the_executors_gate_refuses_is_no_success_and_its_class_names_the_defect():
    """Only arxiv_id and mode are compared, but a called write_note must still pass the executor's gate
    (validation.check_call) on the rest: a parameter the tool does not have is 多参, and free text of a type
    the schema refuses is 值或格式错 whether or not the gold note sets that field. While every parameter of
    write_note was compared, such a field the gold note lacks (problem here) was 多参, an extra parameter; the
    class it has now names its defect, the wrong type. The per-parameter F1 still counts arxiv_id and mode
    alone."""
    it = item(NEXT_STEP, NOTE, messages=AFTER_SEARCH)
    seen, today = ["2609.01234", "2609.05678"], "2026-10-10"  # the executor's seen ids after AFTER_SEARCH
    quoted = {"arxiv_id": "2609.01234", "new_terms": ["verifier"],
              "evidence": '"We verify all drafted tokens in a single forward pass."'}
    assert check_call("write_note", quoted, seen, today) == []
    assert score(it, call("write_note", "c7", **quoted))["success"] is True
    for compared in ({"arxiv_id": "2609.01234v2"}, {"arxiv_id": "arXiv:2609.01234"}, {"mode": " scaffold "}):
        normalised = {**quoted, **compared}  # compared after normalisation, as for every tool: no gate
        assert check_call("write_note", normalised, seen, today), compared
        assert score(it, call("write_note", "c7", **normalised))["success"] is True, compared
    for arguments, error_class in (({**quoted, "title": "Adaptive drafts"}, EXTRA_PARAM),
                                   ({**quoted, "evidence": ["We verify all drafted tokens."]}, WRONG_VALUE),
                                   ({**quoted, "problem": 42}, WRONG_VALUE),
                                   ({**quoted, "new_terms": "verifier"}, WRONG_VALUE),
                                   ({**quoted, "open_terms": [1]}, WRONG_VALUE),
                                   ({**quoted, "title": "Adaptive drafts", "problem": 42}, EXTRA_PARAM)):
        assert check_call("write_note", arguments, seen, today), arguments  # the executor refuses the call
        s = score(it, call("write_note", "c7", **arguments))
        assert (s["arguments"], s["exact_args"], s["success"], s["error_class"]) == (False, False, False,
                                                                                      error_class), arguments
        assert s["param_counts"] == {"write_note.arxiv_id": [1, 0, 0], "write_note.mode": [1, 0, 0]}


def test_a_made_up_id_in_the_free_text_of_write_note_still_fails_ids_grounded():
    made_up = call("write_note", "c7", arxiv_id="2609.01234",
                   evidence='"Unlike 2609.99999, the draft model needs no training."')
    s = score(item(NEXT_STEP, NOTE, messages=AFTER_SEARCH), made_up)
    assert s["ids_grounded"] is False and s["success"] is False and s["error_class"] == FABRICATED_ID


def test_recover_judges_the_step_after_a_failure_like_a_call():
    limited = {"role": "tool", "tool_call_id": "c1", "content": '{"error":"RATE_LIMITED"}'}
    failed = [*AFTER_SEARCH[:3], limited]
    it = item(RECOVER, SEARCH, messages=failed)
    assert score(it, SEARCH)["recover"] is True
    s = score(it, call("search_arxiv", "c1", query="speculative decoding", date_from="2026-08-01"))
    assert s["recover"] is False and s["error_class"] == WRONG_VALUE


def test_a_next_step_that_expects_the_answer_succeeds_without_a_call():
    it = item(NEXT_STEP, messages=AFTER_SEARCH)
    assert score(it, content="找到两篇: 2609.01234 与 2609.05678。")["success"] is True


def test_ids_in_the_output_must_occur_in_the_history():
    assert arxiv_ids("see arXiv:2609.01234v2, notes/2609.05678.md and 2026.10101") == {"2609.01234",
                                                                                       "2609.05678"}
    it = item(NEXT_STEP, messages=AFTER_SEARCH)
    s = score(it, content="找到: 2609.01234 与 2609.99999。")
    assert s["ids_grounded"] is False and s["success"] is False


def test_per_parameter_counts_pair_calls_and_leave_out_defaults_neither_side_set():
    s = score(item(CALL, SEARCH), call("search_arxiv", "c1", query="speculative decoding",
                                       date_from="2026-08-01"))
    assert s["param_counts"] == {"search_arxiv.date_from": [0, 1, 1], "search_arxiv.query": [1, 0, 0]}
    assert s["params"]["precision"] == 0.5 and s["params"]["recall"] == 0.5 and s["params"]["f1"] == 0.5


def test_parameters_that_neither_side_set_are_left_out_of_the_per_parameter_counts():
    only_query = call("search_arxiv", "c1", query="speculative decoding")
    s = score(item(CALL, only_query), only_query)
    assert s["param_counts"] == {"search_arxiv.query": [1, 0, 0]}
    explicit = call("search_arxiv", "c1", query="speculative decoding", max_results=5)
    s = score(item(CALL, only_query), explicit)  # set on one side: counted, and equal to the default
    assert s["param_counts"] == {"search_arxiv.max_results": [1, 0, 0], "search_arxiv.query": [1, 0, 0]}
    assert s["exact_args"] is True


def test_a_duplicated_category_is_a_format_error_not_an_exact_match():
    want = call("search_arxiv", "c1", query="speculative decoding", categories=["cs.CL", "cs.LG"])
    twice = call("search_arxiv", "c1", query="speculative decoding", categories=["cs.CL", "cs.LG", "cs.CL"])
    s = score(item(CALL, want), twice)
    assert s["exact_args"] is False and s["error_class"] == WRONG_VALUE
    # tags have no uniqueItems in the schema: still compared as sets (plan: 分类与标签按集合比)
    notes = call("search_notes", "c1", query="speculative decoding", tags=["llm", "decoding"])
    assert score(item(CALL, notes), call("search_notes", "c1", query="speculative decoding",
                                         tags=["llm", "decoding", "llm"]))["exact_args"] is True


# ---- error classes ---------------------------------------------------------------------------------------

REPEAT_HISTORY = AFTER_SEARCH[:2] + [{"role": "assistant", "content": "", "tool_calls": [GET_A.to_api()]},
                                     {"role": "tool", "tool_call_id": "c2", "content": '{"title":"A"}'}]
FETCH = call("fetch_fulltext", "c6", arxiv_id="2609.01234")
FAILED = AFTER_SEARCH[:3] + [{"role": "tool", "tool_call_id": "c1", "content": '{"error":"UPSTREAM_ERROR"}'}]
ASK = call("ask_clarification", "c5", question="哪个主题?", missing=["topic"])


@pytest.mark.parametrize(("it", "calls", "content", "error_class"), [
    (item(CALL, SEARCH), [GET_A], "", WRONG_TOOL),
    (item(CALL, SEARCH), [call("search_arxiv", "c1", query="speculative decoding")], "", MISSING_PARAM),
    (item(CALL, SEARCH), [call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-01",
                               date_to="2026-10-01")], "", EXTRA_PARAM),
    (item(CALL, SEARCH),
     [call("search_arxiv", "c1", query="speculative decoding", date_from="2026-13-01")], "", WRONG_VALUE),
    (item(NEXT_STEP, messages=AFTER_SEARCH), [], "最相关的是 2610.00001。", FABRICATED_ID),
    (item(NO_CALL, slots={}), [SEARCH], "", UNWANTED_CALL),
    (item(CLARIFY, ASK, slots={}), [], "请问主题?", NO_CLARIFY),
    (item(RECOVER, SEARCH, messages=FAILED), [], "暂时检索不了。", BAD_RECOVERY),
    (item(NEXT_STEP, FETCH, messages=REPEAT_HISTORY), [GET_A], "", RETRY_STORM),
    (item(CALL, SEARCH), [call("search_arxiv", "c1", raw='{"query": "speculative')], "", BROKEN_JSON),
    (item(CALL, SEARCH), [SEARCH], "<think>the user wants recent papers</think>", THINK_LEAK),
], ids=list(ERROR_CLASSES))
def test_each_error_class_is_the_first_that_applies(it, calls, content, error_class):
    s = score(it, *calls, content=content)
    assert s["success"] is False and s["error_class"] == error_class


def test_the_error_classes_follow_the_plan_and_the_structural_ones_are_decided_first():
    assert ERROR_CLASSES == ("选错工具", "缺参", "多参", "值或格式错", "编号编造", "不该调却调",
                             "该追问没追问", "恢复不当", "重试风暴", "JSON 损坏", "思考块泄漏")
    assert PRECEDENCE == (BROKEN_JSON, THINK_LEAK, *ERROR_CLASSES[:9])
    broken_other_tool = call("get_paper", "c2", raw='{"arxiv_id": ')
    assert score(item(CALL, SEARCH), broken_other_tool)["error_class"] == BROKEN_JSON  # beats 选错工具
    leak = score(item(CALL, SEARCH), call("search_arxiv", "c1", query="speculative decoding"),
                 content="</think>")
    assert leak["no_think_leak"] is False and leak["error_class"] == THINK_LEAK  # beats 缺参
    both = score(item(CALL, SEARCH), broken_other_tool, content="<think>x</think>")
    assert both["error_class"] == BROKEN_JSON
    assert score(item(CALL, SEARCH), GET_A, call("search_arxiv", "c1", query="speculative decoding"))[
        "error_class"] == WRONG_TOOL  # a wrong tool beats a missing parameter


def test_a_made_up_id_inside_a_call_argument_is_a_value_error_and_fails_ids_grounded():
    want = call("get_paper", "c2", arxiv_id="2609.01234")
    s = score(item(NEXT_STEP, want, messages=AFTER_SEARCH), call("get_paper", "c2", arxiv_id="2609.99999"))
    assert s["ids_grounded"] is False and s["error_class"] == WRONG_VALUE


def test_a_retry_that_is_expected_is_a_success_not_a_retry():
    again = item(RECOVER, SEARCH, messages=FAILED)
    assert score(again, SEARCH)["error_class"] is None


@pytest.mark.parametrize("status", [CONTEXT_OVERFLOW, ERROR, BUDGET_STOP, SKIPPED])
def test_a_record_that_is_not_ok_fails_the_applying_checks_on_calls_and_carries_its_status(status):
    s = score_item({"id": "i1", "status": status, "item": {}}, item(CALL, SEARCH))
    assert s["success"] is False and s["error_class"] == status
    assert all(s[c] is False for c in ("tool_choice", "arguments", "exact_args", "query_strict"))
    assert all(s[c] is None for c in ("json_valid", "no_think_leak", "ids_grounded", "no_false_call"))
    assert s["params"] == {"tp": 0, "fp": 0, "fn": 2, "precision": None, "recall": 0.0, "f1": 0.0}


def test_a_record_that_is_not_ok_on_a_no_call_item_is_neither_a_false_call_nor_an_ungrounded_id():
    # The model produced nothing: the output checks do not apply and drop out of their rates (None).
    it = item(NO_CALL, slots={})
    s = score_item({"id": "i1", "status": CONTEXT_OVERFLOW, "item": {}}, it)
    assert s["success"] is False and s["error_class"] == CONTEXT_OVERFLOW
    assert s["no_false_call"] is None and s["ids_grounded"] is None and s["json_valid"] is None
    assert s["tool_choice"] is False
    summary = summarize([s], [it])
    assert summary["false_call_rate"] is None and summary["ids_grounded_rate"] is None
    assert summary["json_valid_rate"] is None and summary["success_rate"] == 0.0
    assert summary["n_per_check"]["no_false_call"] == 0


def test_malformed_records_and_items_raise_value_error():
    with pytest.raises(ValueError, match="output"):
        score_item({"id": "i1", "status": OK, "item": {}}, item(CALL, SEARCH))
    with pytest.raises(ValueError, match="type"):
        score_item(ok(SEARCH), {**item(CALL, SEARCH), "type": "chat"})
    with pytest.raises(ValueError, match="status"):
        score_item({"id": "i1", "status": "done"}, item(CALL, SEARCH))


def test_score_items_joins_by_id_and_refuses_ids_that_differ():
    items = [item(CALL, SEARCH, id_="a"), item(NO_CALL, id_="b", slots={})]
    scores = score_items([ok(id_="b"), ok(SEARCH, id_="a")], items)
    assert [s["id"] for s in scores] == ["b", "a"] and all(s["success"] for s in scores)
    with pytest.raises(IdMismatch):
        score_items([ok(SEARCH, id_="a")], items)
    with pytest.raises(IdMismatch):
        score_items([ok(SEARCH, id_="a"), ok(id_="a")], items)


def test_score_items_scores_test_material_only_as_the_items_lockedset_open_returned(tmp_path):
    items = [{**item(CALL, SEARCH, id_="req-test-0003-s1"), "split": "test"}]
    with pytest.raises(RefusedTestSet, match="only the phase-4 test runner"):
        score_items([ok(SEARCH, id_="req-test-0003-s1")], items)
    path = tmp_path / "test.jsonl"
    write_jsonl(items, path)
    write_lock(path)
    opened = LockedSet(path, tmp_path / "test-access.jsonl").open("M3", "the phase-4 run")
    [scored] = score_items([ok(SEARCH, id_="req-test-0003-s1")], opened)
    assert scored["success"] is True


# ---- scorers: the normalized scorer's S1, S2 and S3 (M7 WP1, docs/m7-本地提升计划.md) -----------------

TODAY = "2026-10-10"
PROMPTED = {"role": "system", "content": system_prompt(TODAY)}  # "你是科研助理,今天是 2026-10-10。..."
ON_TODAY = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-01", date_to=TODAY)


def asked(text: str) -> list[dict]:
    """An item's messages: today's system prompt and the user's request text."""
    return [PROMPTED, {"role": "user", "content": text}]


def scored(it: dict, *calls, content: str = "") -> tuple[dict, dict]:
    """The item's score under the default scorer and under the normalized one: the same keys, and a success
    without an error class under either."""
    record = ok(*calls, content=content, id_=it["id"])
    default, normalized = score_item(record, it), score_item(record, it, NORMALIZED_SCORER)
    assert list(default) == list(normalized)
    for s in (default, normalized):
        assert not (s["success"] and s["error_class"] is not None)
    return default, normalized


def outcome(s: dict) -> tuple[bool, str | None]:
    return s["success"], s["error_class"]


def test_the_default_scorer_is_todays_and_an_unknown_scorer_is_refused():
    assert SCORERS == (DEFAULT_SCORER, NORMALIZED_SCORER) == ("default", "normalized")
    results, items = mixed()
    assert score_items(results, items, DEFAULT_SCORER) == score_items(results, items)
    it = item(CALL, ON_TODAY, messages=asked("找 9 月以来到今天的 speculative decoding 论文。"))
    assert score_item(ok(SEARCH), it, scorer=DEFAULT_SCORER) == score_item(ok(SEARCH), it)
    for scorer in ("lenient", "", None, "Normalized"):
        with pytest.raises(ValueError, match="scorer"):
            score_item(ok(SEARCH), it, scorer)
        with pytest.raises(ValueError, match="scorer"):
            score_items([ok(SEARCH)], [it], scorer=scorer)


def test_score_items_scores_every_record_with_the_scorer_given():
    items = [item(CALL, ON_TODAY, messages=asked("找 9 月以来到今天的论文。"), id_="a"),
             item(CALL, SEARCH, id_="b")]
    results = [ok(SEARCH, id_="a"), ok(SEARCH, id_="b")]
    assert [s["success"] for s in score_items(results, items)] == [False, True]
    normalized = score_items(results, items, scorer=NORMALIZED_SCORER)
    assert normalized == [score_item(r, it, NORMALIZED_SCORER) for r, it in zip(results, items, strict=True)]
    assert [s["success"] for s in normalized] == [True, True]


def test_today_is_the_date_the_first_system_message_gives():
    assert today_of(item(CALL, SEARCH, messages=asked("x"))) == TODAY
    assert today_of(item(CALL, SEARCH)) is None  # SYSTEM gives no date
    later = {"role": "system", "content": system_prompt("2026-11-01")}
    user = {"role": "user", "content": "今天是 2026-12-01。"}
    assert today_of(item(CALL, SEARCH, messages=[PROMPTED, user, later])) == TODAY
    assert today_of(item(CALL, SEARCH, messages=[user, later])) == "2026-11-01"  # the first *system* message
    for text, today in (("今天是 2026年10月10日。", TODAY), ("今天是2026/10/10,星期六。", TODAY),
                        ("今天是 星期六。", None), ("今天是 2026-02-30。", None)):
        assert today_of(item(CALL, SEARCH, messages=[{"role": "system", "content": text}, USER])) == today
    assert today_of(item(CALL, SEARCH, messages=[{"role": "system", "content": None}, USER])) is None


@pytest.mark.parametrize("today", ["2026-10-10", "2026-01-05", "2027-12-31", "2028-02-29"])
def test_today_is_read_back_from_the_real_system_prompt(today):
    """agent.prompts.system_prompt(d) gives d back, so a change of the template that S1 could no longer read
    fails here instead of turning S1 off without a word."""
    prompted = {"role": "system", "content": system_prompt(today)}
    assert today_of(item(CALL, SEARCH, messages=[prompted, USER])) == today


@pytest.mark.parametrize("text, today", [
    ("今天是 2026-10-10.", TODAY),  # an ASCII full stop right after the date
    ("今天是 2026 年 10 月 10 日,星期六。", TODAY),  # spaces inside a Chinese date
    ("今天是2026.10.10;请按需调用工具。", TODAY),
    ("今天是 ２０２６－１０－１０。", TODAY),  # full width
    ("今天是 20261010。", TODAY),  # eight digits, as normalize_date reads them
    ("今天是 2026-10-101。", None),  # a day of three digits is no date
    ("今天是 2026-10。", None),
])
def test_today_is_the_date_itself_whatever_stands_around_it(text, today):
    assert today_of(item(CALL, SEARCH, messages=[{"role": "system", "content": text}, USER])) == today


def test_s1_a_date_to_on_the_prompts_today_is_dropped_on_both_sides_as_if_left_out():
    left_out = score_item(ok(SEARCH), item(CALL, SEARCH, messages=asked("x")), NORMALIZED_SCORER)
    default, normalized = scored(item(CALL, ON_TODAY, messages=asked("x")), SEARCH)  # no date_to answered
    assert outcome(default) == (False, MISSING_PARAM) and outcome(normalized) == (True, None)
    assert normalized == left_out and normalized["exact_args"] is True and normalized["query_strict"] is True
    assert default["param_counts"]["search_arxiv.date_to"] == [0, 0, 1]
    assert "search_arxiv.date_to" not in normalized["param_counts"]
    it = item(CALL, SEARCH, messages=asked("x"))  # the gold has none, the answer one on today
    for date_to in (TODAY, "2026/10/10", "2026年10月10日", " 2026-10-10 "):  # compared after normalize_date
        answer = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-01",
                      date_to=date_to)
        default, normalized = scored(it, answer)
        assert outcome(default) == (False, EXTRA_PARAM) and outcome(normalized) == (True, None), date_to
    default, normalized = scored(item(CALL, ON_TODAY, messages=asked("x")), ON_TODAY)  # both on today
    assert default["success"] is normalized["success"] is True
    assert default["param_counts"]["search_arxiv.date_to"] == [1, 0, 0]
    assert "search_arxiv.date_to" not in normalized["param_counts"]


def test_s1_leaves_another_day_another_tool_and_an_item_without_a_today_as_they_were():
    on_the_9th = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-01",
                      date_to="2026-10-09")
    default, normalized = scored(item(CALL, SEARCH, messages=asked("x")), on_the_9th)
    assert outcome(default) == outcome(normalized) == (False, EXTRA_PARAM)
    # the gold on today, the answer on the 9th: under S1 the gold has no date_to, so the answer's is extra
    default, normalized = scored(item(CALL, ON_TODAY, messages=asked("x")), on_the_9th)
    assert outcome(default) == (False, WRONG_VALUE) and outcome(normalized) == (False, EXTRA_PARAM)
    default, normalized = scored(item(CALL, SEARCH), ON_TODAY)  # SYSTEM gives no today: S1 does not apply
    assert outcome(default) == outcome(normalized) == (False, EXTRA_PARAM)
    # list_notes has no date_to: one on today is still a parameter the tool does not have
    notes = item(CALL, call("list_notes", "c1", tag="survey"), messages=asked("x"), slots={})
    default, normalized = scored(notes, call("list_notes", "c1", tag="survey", date_to=TODAY))
    assert outcome(default) == outcome(normalized) == (False, EXTRA_PARAM)


MOE = {"topic": "mixture-of-experts"}
HYPHENATED = call("search_arxiv", "c1", query="mixture-of-experts")


def test_s2_reads_the_hyphens_of_a_query_as_spaces_on_both_sides_and_in_the_words():
    # (a full-width hyphen-minus is no case: NFKC makes it U+002D under either scorer)
    for query in ("mixture of experts", "Mixture\N{HYPHEN}of\N{HYPHEN}Experts routing",
                  "mixture\N{NON-BREAKING HYPHEN}of\N{FIGURE DASH}experts",
                  "mixture \N{EN DASH} of \N{EN DASH} experts"):
        default, normalized = scored(item(CALL, HYPHENATED, slots=MOE), call("search_arxiv", "c1",
                                                                             query=query))
        assert outcome(default) == (False, WRONG_VALUE) and outcome(normalized) == (True, None), query
        assert normalized["query_strict"] is False  # query_strict stays literal
        assert default["param_counts"] == {"search_arxiv.query": [0, 1, 1]}
        assert normalized["param_counts"] == {"search_arxiv.query": [1, 0, 0]}
    # the topic spaced and the gold hyphenated: the topic is a required word only once both are folded
    spaced_topic = item(CALL, HYPHENATED, slots={"topic": "mixture of experts"})
    default, normalized = scored(spaced_topic, call("search_arxiv", "c1", query="mixture of experts survey"))
    assert outcome(default) == (False, WRONG_VALUE) and outcome(normalized) == (True, None)
    # the gold and its topic spaced, the answer hyphenated
    spaced = item(CALL, call("search_arxiv", "c1", query="mixture of experts"),
                  slots={"topic": "mixture of experts"})
    default, normalized = scored(spaced, HYPHENATED)
    assert outcome(default) == (False, WRONG_VALUE) and outcome(normalized) == (True, None)
    # no required word: the equality fallback compares the folded queries, case aside
    judge = item(CALL, call("search_arxiv", "c1", query="LLM-as-a-judge evaluation"), slots={})
    default, normalized = scored(judge, call("search_arxiv", "c1", query="llm as a judge  Evaluation"))
    assert outcome(default) == (False, WRONG_VALUE) and outcome(normalized) == (True, None)
    # the slots' required words fold too
    given = item(CALL, call("search_notes", "c1", query="test-time scaling"),
                 slots={"required_words": ["test-time"]})
    default, normalized = scored(given, call("search_notes", "c1", query="test time scaling survey"))
    assert outcome(default) == (False, WRONG_VALUE) and outcome(normalized) == (True, None)


def test_s2_folds_no_other_dash_and_still_needs_every_word():
    for query in ("mixture\N{EM DASH}of\N{EM DASH}experts", "mixture experts", "mixtureofexperts", "experts"):
        default, normalized = scored(item(CALL, HYPHENATED, slots=MOE), call("search_arxiv", "c1",
                                                                             query=query))
        assert outcome(default) == outcome(normalized) == (False, WRONG_VALUE), query
    judge = item(CALL, call("search_arxiv", "c1", query="LLM-as-a-judge evaluation"), slots={})
    default, normalized = scored(judge, call("search_arxiv", "c1", query="LLM as a judge"))
    assert outcome(default) == outcome(normalized) == (False, WRONG_VALUE)
    exact = scored(item(CALL, HYPHENATED, slots=MOE), HYPHENATED)[1]
    assert exact["success"] is True and exact["query_strict"] is True


HALLUCINATION = {"topic": "hallucination", "tags": ["baseline"]}
NO_COUNT = "Locate my notes about hallucination that are tagged baseline."
TOP_5 = call("search_notes", "c1", query="hallucination", tags=["baseline"], top_k=5)
TOP_DEFAULT = call("search_notes", "c1", query="hallucination", tags=["baseline"])  # top_k 3


def test_s3_covers_three_counts_and_not_search_arxivs_max_results():
    assert COUNT_PARAMS == {("search_notes", "top_k"), ("list_notes", "limit"),
                            ("fetch_fulltext", "max_chars")}


def test_s3_a_count_the_request_does_not_give_may_be_the_default_or_the_gold_value():
    it = item(CALL, TOP_5, messages=asked(NO_COUNT), slots=HALLUCINATION)
    default, normalized = scored(it, TOP_DEFAULT)
    assert outcome(default) == (False, WRONG_VALUE) and outcome(normalized) == (True, None)
    assert default["param_counts"]["search_notes.top_k"] == [0, 1, 1]  # the gold set it: counted, and differs
    for answer in (TOP_DEFAULT, call("search_notes", "c1", query="hallucination", tags=["baseline"], top_k=3),
                   TOP_5, call("search_notes", "c1", query="hallucination", tags=["baseline"], top_k=5.0)):
        normalized = scored(it, answer)[1]
        assert outcome(normalized) == (True, None) and normalized["exact_args"] is True
        assert normalized["param_counts"]["search_notes.top_k"] == [1, 0, 0]  # the counts follow S3
    # list_notes.limit, default 10, as dev-v2's flash gold sets it (limit 20)
    listed = item(CALL, call("list_notes", "c1", tag="to-read", since="2026-08-25", limit=20),
                  messages=asked("查看一下我从 8 月 25 日以来标了 to-read 的笔记。"),
                  slots={"since": "2026-08-25", "tags": ["to-read"]})
    default, normalized = scored(listed, call("list_notes", "c1", tag="to-read", since="2026-08-25"))
    assert outcome(default) == (False, WRONG_VALUE) and outcome(normalized) == (True, None)
    # fetch_fulltext.max_chars, default 1500, on parallel calls
    ids = ("2609.19499", "2607.01844")
    gold = [call("fetch_fulltext", f"c{k}", arxiv_id=i, section="conclusion", max_chars=1200)
            for k, i in enumerate(ids)]
    parallel = item(PARALLEL, *gold, messages=asked(f"请同时打开 {ids[0]} 和 {ids[1]},查看它们的结论。"),
                    slots={"arxiv_ids": list(ids), "section": "conclusion"})
    answers = [call("fetch_fulltext", "c0", arxiv_id=ids[0], section="conclusion"),
               call("fetch_fulltext", "c1", arxiv_id=ids[1], section="conclusion", max_chars=1500)]
    default, normalized = scored(parallel, *answers)
    assert outcome(default) == (False, WRONG_VALUE) and outcome(normalized) == (True, None)


def test_s3_accepts_no_third_value_and_no_count_the_gold_leaves_at_its_default():
    it = item(CALL, TOP_5, messages=asked(NO_COUNT), slots=HALLUCINATION)
    four = call("search_notes", "c1", query="hallucination", tags=["baseline"], top_k=4)
    default, normalized = scored(it, four)
    assert outcome(default) == outcome(normalized) == (False, WRONG_VALUE)
    assert normalized["param_counts"]["search_notes.top_k"] == [0, 1, 1]
    plain = item(CALL, TOP_DEFAULT, messages=asked(NO_COUNT), slots=HALLUCINATION)  # the gold at the default
    default, normalized = scored(plain, TOP_5)
    assert outcome(default) == outcome(normalized) == (False, WRONG_VALUE)


@pytest.mark.parametrize("text, slots", [
    (NO_COUNT, {**HALLUCINATION, "count": 5}),  # the slots carry the count
    ("在我的笔记里找 5 条讲幻觉、打了 baseline 标签的。", HALLUCINATION),  # the request gives it in digits
    ("在我的笔记里找五条讲幻觉、打了 baseline 标签的。", HALLUCINATION),  # a Chinese numeral
    ("Find five of my notes about hallucination tagged baseline.", HALLUCINATION),  # an English word
    ("Find my notes about hallucination tagged baseline, top ５.", HALLUCINATION),  # a full-width digit
], ids=["count-slot", "digits", "chinese", "english", "full-width"])
def test_s3_does_not_apply_when_the_request_gives_the_count(text, slots):
    it = item(CALL, TOP_5, messages=asked(text), slots=slots)
    default, normalized = scored(it, TOP_DEFAULT)
    assert outcome(default) == outcome(normalized) == (False, WRONG_VALUE)
    assert outcome(scored(it, TOP_5)[1]) == (True, None)


@pytest.mark.parametrize("text, applies", [
    ("把 2609.27374 的方法部分取出来,800 字以内就行。", False),
    ("把 2609.27374 的方法部分取出来,八百字以内就行。", False),
    ("Fetch the method section of 2609.27374, eight hundred characters at most.", False),
    ("把 2609.27374 的方法部分取出来给我看。", True),
])
def test_s3_follows_a_handwritten_request_that_gives_the_number_of_characters(text, applies):
    gold = call("fetch_fulltext", "c1", arxiv_id="2609.27374", section="method", max_chars=800)
    it = {**item(CALL, gold, messages=asked(text), slots={"arxiv_id": "2609.27374", "section": "method"}),
          "source": {"handwritten": True}}
    default, normalized = scored(it, call("fetch_fulltext", "c1", arxiv_id="2609.27374", section="method"))
    assert outcome(default) == (False, WRONG_VALUE)
    assert outcome(normalized) == ((True, None) if applies else (False, WRONG_VALUE))


def test_s3_does_not_cover_search_arxivs_max_results():
    capped = item(CALL, call("search_arxiv", "c1", query="speculative decoding", max_results=3),
                  messages=asked("找 speculative decoding 的论文。"), slots={"topic": "speculative decoding"})
    default, normalized = scored(capped, call("search_arxiv", "c1", query="speculative decoding"))
    assert outcome(default) == outcome(normalized) == (False, WRONG_VALUE)


def test_the_request_text_is_the_items_user_messages():
    first, then = {"role": "user", "content": "找 3 篇。"}, {"role": "user", "content": "再找两篇。"}
    messages = [PROMPTED, first, {"role": "assistant", "content": "好的 7"},
                {"role": "tool", "tool_call_id": "c1", "content": "[]"}, then]
    assert request_text(item(CALL, SEARCH, messages=messages)) == "找 3 篇。\n再找两篇。"


@pytest.mark.parametrize("text, numbers", [
    ("找 5 条", {5}),
    ("只要 1,200 字", {1200, 1, 200}),
    ("１２００字", {1200}),
    ("五篇就够", {5}),
    ("两篇", {2}),
    ("二十条", {20, 2}),
    ("十五", {15, 5}),
    ("一千二百字", {1200, 1, 2}),
    ("一千二", {1200, 1, 2}),  # the unit after the last digit left off: one below the last
    ("一千零五", {1005, 1, 0, 5}),
    ("一万二", {12000, 1, 2}),
    ("一五〇〇", {1500, 1, 5, 0}),  # digit by digit
    ("twenty-five notes", {25, 20, 5}),
    ("twelve hundred characters", {1200, 12, 100}),
    ("one thousand five hundred", {1500, 1, 1000, 5, 100}),
    ("one hundred and twenty", {120, 1, 100, 20}),
    ("a hundred", {100}),
    ("Ten papers", {10}),
    ("从 2026-09-20 起", {2026, 9, 20}),  # a date's digits count too
    ("no number here, and none there", set()),
    ("", set()),
])
def test_numbers_in_reads_digits_chinese_numerals_and_english_number_words(text, numbers):
    assert numbers_in(text) == numbers


@pytest.mark.parametrize("text, numbers", [
    ("前2千字", {2, 1000, 2000}),
    ("前 2 千字", {2, 1000, 2000}),  # a space between the digits and the unit
    ("1.5千字", {1, 5, 1000, 1500}),
    ("1万字", {1, 10000}),
    ("3万字", {3, 10000, 30000}),
    ("1千2百字", {1, 2, 100, 1000, 1200}),
    ("1千2", {1, 2, 1000, 1200}),  # as 一千二: the last digit takes the unit below 千
    ("1千零5", {0, 1, 5, 1000, 1005}),
    ("2千50字", {2, 50, 1000, 2050}),  # a number of two digits after the unit is no hundreds
    ("1万2千", {1, 2, 1000, 10000, 12000}),
    ("两千字", {2, 2000}),
    ("取 1.2k 字", {1, 2, 1200}),
    ("3k chars", {3, 3000}),
    ("2K", {2, 2000}),
    ("0.8k characters", {0, 8, 800}),
    ("1.2w 字", {1, 2, 12000}),
    ("1.2345k", {1, 2345, 1234, 1235}),  # not a whole number: both whole numbers around it
    ("3 kinds of 2 ways", {2, 3}),  # a letter after the k or w: no multiplier
    ("a dozen notes", {12}),
    ("two dozen notes", {2, 12, 24}),
])
def test_numbers_in_reads_a_number_with_a_multiplier(text, numbers):
    """Digits followed by k, w, 千 or 万 (and digits mixed with Chinese units) are read with their multiplier,
    beside every number the readers above find, and "dozen" is twelve: a number more can only keep S3 from
    applying."""
    assert numbers_in(text) == numbers


FETCH = call("fetch_fulltext", "c1", arxiv_id="2401.00001")  # max_chars left at the default, 1500


@pytest.mark.parametrize("text, gold", [
    ("取 2401.00001 的正文前2千字。", call("fetch_fulltext", "c1", arxiv_id="2401.00001", max_chars=2000)),
    ("取 2401.00001 的正文前 2 千字。", call("fetch_fulltext", "c1", arxiv_id="2401.00001", max_chars=2000)),
    ("取 2401.00001 的正文,1千2百字以内。",
     call("fetch_fulltext", "c1", arxiv_id="2401.00001", max_chars=1200)),
    ("Fetch the introduction of 2401.00001, 1.2k characters.",
     call("fetch_fulltext", "c1", arxiv_id="2401.00001", max_chars=1200)),
    ("Fetch the introduction of 2401.00001, 0.8k characters at most.",
     call("fetch_fulltext", "c1", arxiv_id="2401.00001", max_chars=800)),
], ids=["2千", "2 千", "1千2百", "1.2k", "0.8k"])
def test_s3_does_not_apply_when_the_request_writes_the_count_with_a_multiplier(text, gold):
    """The gold max_chars written with a multiplier is a number the request gives: an answer that leaves
    max_chars at the default is wrong under both scorers, and the gold value right."""
    it = item(CALL, gold, messages=asked(text), slots={"arxiv_id": "2401.00001"})
    default, normalized = scored(it, FETCH)
    assert outcome(default) == outcome(normalized) == (False, WRONG_VALUE)
    assert outcome(scored(it, gold)[1]) == (True, None)
    # the same gold, a request that writes no count: S3 applies
    plain = item(CALL, gold, messages=asked("取 2401.00001 的正文。"), slots={"arxiv_id": "2401.00001"})
    assert outcome(scored(plain, FETCH)[1]) == (True, None)


def test_s3_reads_a_dozen_as_twelve():
    gold = call("list_notes", "c1", tag="to-read", limit=12)
    it = item(CALL, gold, messages=asked("List a dozen of my notes tagged to-read."),
              slots={"tags": ["to-read"]})
    default, normalized = scored(it, call("list_notes", "c1", tag="to-read"))  # limit left at 10
    assert outcome(default) == outcome(normalized) == (False, WRONG_VALUE)


# ---- numbers_in, round 3: English multiplier words after digits, spaced thousands, financial numerals and
# the owner's quantity words (10-30) -----------------------------------------------------------------------

@pytest.mark.parametrize("text, numbers", [
    ("first 2 thousand characters", {2, 1000, 2000}),
    ("the first 1.2 thousand chars", {1, 2, 1000, 1200}),
    ("2 dozen notes", {2, 12, 24}),
    ("3 hundred characters", {3, 100, 300}),
    ("2 thousands of them", {2, 2000}),  # plural; "thousands" alone is no number word
    ("2 Dozens", {2, 24}),
    ("3hundred chars", {3, 100, 300}),  # no space
    ("1.5 million characters", {1, 5, 1500000}),
    ("2 thousandth", {2}),  # an ordinal is no multiplier
    # several terms in a row add up, as 1千2百 does (the words alone give 1100 here, as before)
    ("1 thousand 2 hundred characters", {1, 2, 100, 200, 1000, 1100, 1200}),
    ("2 thousand and 50 chars", {2, 50, 1000, 2000, 2050}),
    ("2 hundred thousand", {2, 100, 1000, 100000, 200000}),  # every multiplier after the number
    ("1.2345 thousand", {1, 2345, 1000, 1234, 1235}),  # not a whole number: both whole numbers around it
    ("2 THOUSAND", {2, 1000, 2000}),
    # a grouped number or a hyphen before the multiplier (the round-3 review); the group's last part alone is
    # no term of its own (no 200 thousand)
    ("1,200 thousand", {1, 200, 1000, 1200, 1200000}),
    ("1 200 thousand", {1, 200, 1000, 1200, 1200000}),
    ("2-thousand chars", {2, 1000, 2000}),
    ("3-dozen notes", {3, 12, 36}),
])
def test_numbers_in_reads_digits_followed_by_an_english_multiplier_word(text, numbers):
    assert numbers_in(text) == numbers


@pytest.mark.parametrize("text, numbers", [
    ("只要 1 200 字", {1, 200, 1200}),
    ("1 200 000 characters", {0, 1, 200, 1200000}),
    ("12 345 chars", {12, 345, 12345}),
    ("取 1\N{NO-BREAK SPACE}200 字", {1, 200, 1200}),  # NFKC: a space
    ("取 1\N{NARROW NO-BREAK SPACE}200 字", {1, 200, 1200}),
    ("只要 1,200 字", {1, 200, 1200}),  # a comma, as before
    ("1 2345", {1, 2345}),  # a group of four digits is no thousands group
    ("2026 100", {2026, 100}),  # nor are groups after four digits
    ("1234 567", {1234, 567}),
])
def test_numbers_in_reads_digit_groups_with_a_space_as_the_thousands_separator_joined_too(text, numbers):
    assert numbers_in(text) == numbers


@pytest.mark.parametrize("text, numbers", [
    ("伍篇", {5}),
    ("贰拾条", {2, 20}),
    ("壹仟贰佰字", {1, 2, 1200}),
    ("壹仟贰", {1, 2, 1200}),  # as 一千二
    ("壹万贰仟", {1, 2, 10000, 12000}),  # 10000: the ordinary 万 alone, as read before
    ("2仟字", {2, 1000, 2000}),
    ("柒捌玖", {7, 8, 9, 789}),  # digit by digit, as 七八九
    ("陆拾肆", {4, 6, 64}),
    ("叁佰", {3, 300}),
    ("二十叁", {2, 3, 20, 23}),  # mixed with the ordinary numerals: 二十 alone is still read
])
def test_numbers_in_reads_the_financial_numerals_as_the_ordinary_ones(text, numbers):
    assert numbers_in(text) == numbers


@pytest.mark.parametrize("text, numbers", [
    ("Find a couple of papers.", {2}),
    ("a couple papers", {2}),
    ("couple of notes", {2}),
    ("A Pair of notes", {2}),
    ("pair of notes", {2}),
    ("a coupled model, a pairwise test", set()),  # other words
    ("找a couple of papers", {2}),  # a Chinese character right before
    ("一打笔记", {1, 12}),
    ("半打笔记", {6}),
    ("half a dozen notes", {6}),  # 6, and its dozen no 12
    ("Half-dozen notes", {6}),
    ("half-a-dozen notes", {6}),
    ("a half dozen notes", {6}),
    ("half a dozen or a dozen", {6, 12}),
    ("two dozen and half a dozen", {2, 6, 12, 24}),  # two dozen still 24
    ("找一些论文,一篇就行", {1}),  # the 一 of 一篇 is still 1
    ("二一些", {2}),  # nor is a number read with the 一 of 一些 (21 before)
    ("two half a dozen hundred", {2, 6, 100}),  # nor with its dozen (1200 before), nor across it (no 200)
])
def test_numbers_in_reads_the_owners_quantity_words(text, numbers):
    assert numbers_in(text) == numbers


@pytest.mark.parametrize("text", ["Find a few papers.", "few notes", "several notes", "some notes",
                                  "找几篇论文", "几个笔记", "找一些笔记", "若干篇",
                                  # a vague word with a unit stays vague (the round-3 review)
                                  "列出几十条笔记", "十几篇", "二十几篇", "几百字", "几千字", "数十篇",
                                  "上百篇",
                                  "several dozen notes", "a few hundred chars", "few hundred", "some dozen",
                                  "several thousand characters"])
def test_numbers_in_reads_no_count_in_a_vague_quantity_word(text):
    """The owner's rule: a vague word gives no count, 一些 not even the 1 of its 一, 几十 not the 10 of its
    十, several dozen not the 12 of its dozen."""
    assert numbers_in(text) == set()


@pytest.mark.parametrize("text, numbers", [
    ("一百篇", {1, 100}),  # a definite 百 is still read
    ("十篇", {10}),
    ("a dozen notes, not several dozen", {12}),  # the definite dozen still 12
    ("二十篇,不是二十几篇", {2, 20}),
])
def test_numbers_in_still_reads_the_definite_numbers_beside_a_vague_one(text, numbers):
    assert numbers_in(text) == numbers


SEARCH_NOTES = {"top_k": ("search_notes", {"query": "hallucination", "tags": ["baseline"]}, HALLUCINATION),
                "limit": ("list_notes", {"tag": "to-read"}, {"tags": ["to-read"]}),
                "max_chars": ("fetch_fulltext", {"arxiv_id": "2401.00001"}, {"arxiv_id": "2401.00001"})}


def count_item(text: str, param: str, gold: int) -> tuple[dict, ToolCall, ToolCall]:
    """An item asking text whose gold call sets param (a count of S3) to gold; the gold call; and the answer
    that leaves param at the tool's default."""
    name, arguments, slots = SEARCH_NOTES[param]
    gold_call = call(name, "c1", **arguments, **{param: gold})
    return item(CALL, gold_call, messages=asked(text), slots=slots), gold_call, call(name, "c1", **arguments)


@pytest.mark.parametrize("text, param, gold", [
    ("Fetch the introduction of 2401.00001, 1.2 thousand characters at most.", "max_chars", 1200),
    ("取 2401.00001 的正文,1 200 字以内。", "max_chars", 1200),
    ("在我的笔记里找伍条讲幻觉、打了 baseline 标签的。", "top_k", 5),
    ("Find a couple of my notes about hallucination tagged baseline.", "top_k", 2),
    ("Find a pair of my notes about hallucination tagged baseline.", "top_k", 2),
    ("列出我标了 to-read 的一打笔记。", "limit", 12),
    ("列出我标了 to-read 的半打笔记。", "limit", 6),
    ("List half a dozen of my notes tagged to-read.", "limit", 6),
], ids=["digits-thousand", "spaced-thousands", "financial", "a-couple", "a-pair", "一打", "半打",
        "half-a-dozen"])
def test_s3_does_not_apply_when_the_request_writes_the_count_in_these_ways(text, param, gold):
    it, gold_call, at_default = count_item(text, param, gold)
    default, normalized = scored(it, at_default)
    assert outcome(default) == outcome(normalized) == (False, WRONG_VALUE)
    assert outcome(scored(it, gold_call)[1]) == (True, None)


@pytest.mark.parametrize("text, param, gold", [
    ("List half a dozen of my notes tagged to-read.", "limit", 12),  # half a dozen gives 6, never 12
    ("Find a few of my notes about hallucination tagged baseline.", "top_k", 5),
    ("Find several of my notes about hallucination tagged baseline.", "top_k", 5),
    ("Find some of my notes about hallucination tagged baseline.", "top_k", 5),
    ("找几篇讲幻觉、打了 baseline 标签的笔记。", "top_k", 5),
    ("找几个讲幻觉、打了 baseline 标签的笔记。", "top_k", 5),
    ("找若干讲幻觉、打了 baseline 标签的笔记。", "top_k", 5),
    ("找一些讲幻觉、打了 baseline 标签的笔记。", "top_k", 1),  # the 一 of 一些 is no 1
], ids=["half-a-dozen-is-not-12", "a-few", "several", "some", "几", "几个", "若干", "一些"])
def test_s3_still_applies_when_the_request_gives_no_count_but_a_vague_word(text, param, gold):
    it, gold_call, at_default = count_item(text, param, gold)
    default, normalized = scored(it, at_default)
    assert outcome(default) == (False, WRONG_VALUE) and outcome(normalized) == (True, None)
    assert outcome(scored(it, gold_call)[1]) == (True, None)


# ---- summary ---------------------------------------------------------------------------------------------

def mixed() -> tuple[list[dict], list[dict]]:
    """A: call right; B: call with a wrong date; C: no_call with a call; D: clarify right; E: parallel,
    error."""
    ask = call("ask_clarification", "c5", question="哪个主题?", missing=["topic"])
    items = [item(CALL, SEARCH, id_="A"), item(CALL, SEARCH, id_="B", lang="en"),
             item(NO_CALL, id_="C", slots={}), item(CLARIFY, ask, id_="D", lang="en", slots={}),
             item(PARALLEL, GET_A, GET_B, id_="E", messages=AFTER_SEARCH)]
    wrong = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-08-01")
    results = [ok(SEARCH, id_="A"), ok(wrong, id_="B"), ok(SEARCH, id_="C"), ok(ask, id_="D"),
               {"id": "E", "status": ERROR, "item": {}, "error": "boom", "error_type": "RuntimeError"}]
    return results, items


def test_summarize_reports_every_rate_with_its_n_on_a_mixed_list():
    results, items = mixed()
    summary = summarize(score_items(results, items), items)
    assert summary["n"] == 5 and summary["n_ok"] == 4
    assert summary["success_rate"] == 0.4
    assert summary["by_type"] == {"call": {"n": 2, "success_rate": 0.5},
                                  "next_step": {"n": 0, "success_rate": None},
                                  "no_call": {"n": 1, "success_rate": 0.0},
                                  "clarify": {"n": 1, "success_rate": 1.0},
                                  "recover": {"n": 0, "success_rate": None},
                                  "parallel": {"n": 1, "success_rate": 0.0}}
    assert summary["by_lang"] == {"zh": {"n": 3, "success_rate": 1 / 3}, "en": {"n": 2, "success_rate": 0.5}}
    assert summary["json_valid_rate"] == 1.0  # E's error left no output: not applicable
    assert summary["tool_choice_accuracy"] == 0.6
    assert summary["exact_args_rate"] == 1 / 3
    assert summary["query_strict_rate"] == 0.5
    assert summary["false_call_rate"] == 1.0
    assert summary["think_leak_rate"] == 0.0
    assert summary["clarify_accuracy"] == 1.0
    assert summary["recover_rate"] is None
    # search_arxiv sets query and date_from (max_results, sort_by stay at their defaults and are left out):
    # A agrees on both, B on query only, C's call is unpaired (two false positives);
    # query: tp 2, fp 1 -> 0.8; date_from: tp 1, fp 2, fn 1 -> 0.4;
    # ask_clarification.missing 1.0; get_paper.arxiv_id: E never produced its two -> 0.0
    assert summary["param_f1"] == {"ask_clarification.missing": 1.0, "get_paper.arxiv_id": 0.0,
                                   "search_arxiv.date_from": 0.4, "search_arxiv.query": 0.8}
    assert summary["macro_param_f1"] == pytest.approx(2.2 / 4)
    assert summary["n_per_check"] == {"json_valid": 4, "tool_choice": 5, "arguments": 3, "exact_args": 3,
                                      "query_strict": 2, "ids_grounded": 4, "clarify": 1, "recover": 0,
                                      "no_false_call": 1, "no_think_leak": 4}
    assert summary["error_classes"] == {**dict.fromkeys(ERROR_CLASSES, 0), WRONG_VALUE: 1, UNWANTED_CALL: 1,
                                        ERROR: 1}
    assert list(summary["error_classes"]) == [*ERROR_CLASSES, ERROR]
    assert summary["unclassified_failures"] == 0


def test_summarize_keys_come_in_a_fixed_order():
    results, items = mixed()
    one = summarize(score_items(results, items), items)
    two = summarize(score_items(results[::-1], items[::-1]), items[::-1])
    assert json.dumps(one) == json.dumps(two)


# ---- statistics ------------------------------------------------------------------------------------------

A = [True] * 80 + [False] * 20
B = [True] * 50 + [False] * 50


def test_bootstrap_ci_is_deterministic_for_a_seed_and_brackets_a_known_difference():
    first = bootstrap_ci(A, B, seed=7, n=2000)
    assert first == bootstrap_ci(A, B, seed=7, n=2000)
    assert first["diff"] == pytest.approx(0.3)
    assert first["low"] < 0.3 < first["high"] and first["low"] > 0
    assert first["mean"] == pytest.approx(0.3, abs=0.02)
    assert bootstrap_ci(A, A, seed=1, n=500)["low"] == bootstrap_ci(A, A, seed=1, n=500)["high"] == 0


def test_cluster_bootstrap_resamples_families_and_is_deterministic():
    families = [f"f{k // 10}" for k in range(100)]  # ten blocks of ten neighbours
    first = cluster_bootstrap_ci(A, B, families, seed=3, n=2000)
    assert first == cluster_bootstrap_ci(A, B, families, seed=3, n=2000)
    assert first["clusters"] == 10 and first["diff"] == pytest.approx(0.3)
    assert first["low"] < 0.3 < first["high"]
    with pytest.raises(ValueError):
        cluster_bootstrap_ci(A, B, families[:5], seed=3, n=10)


def test_mcnemar_is_the_exact_binomial_test_on_the_discordant_pairs():
    a = [True] * 5 + [False] * 1 + [True] * 10 + [False] * 4
    b = [False] * 5 + [True] * 1 + [True] * 10 + [False] * 4
    result = mcnemar(a, b)
    assert result["a_only"] == 5 and result["b_only"] == 1 and result["discordant"] == 6
    assert result["p"] == pytest.approx(2 * sum(math.comb(6, k) / 2 ** 6 for k in range(2)))  # 0.21875
    assert mcnemar([True, False], [True, False])["p"] == 1.0
    assert mcnemar([True] * 3 + [False] * 3, [False] * 3 + [True] * 3)["p"] == 1.0
    with pytest.raises(ValueError):
        mcnemar([True], [True, False])


# ---- the command line ------------------------------------------------------------------------------------

class Replay:
    def __init__(self, turns: list) -> None:
        self.turns = list(turns)

    def next_turn(self, messages: list[dict], tools: list[dict]):
        return self.turns.pop(0)


def write_run(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    results, items = mixed()
    items_path, results_path = tmp_path / "dev-draft.jsonl", tmp_path / "run.jsonl"
    write_jsonl(items, items_path)
    write_jsonl(results, results_path)
    return items_path, results_path, items


def test_a_console_that_cannot_print_the_error_classes_does_not_stop_the_registration(tmp_path, monkeypatch):
    items_path, results_path, _ = write_run(tmp_path)  # its error classes have Chinese names
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(io.BytesIO(), encoding="cp1252"))
    assert score_cli.main([str(results_path), "--items", str(items_path), "--out", "results/dev-b0.json",
                           "--numbers"]) == 0
    report = json.loads((tmp_path / "results" / "numbers.json").read_text(encoding="utf-8"))
    assert report["numbers"]["eval.dev-b0.n"] == "5"


def test_a_subset_scores_only_the_drafted_or_only_the_handwritten_items(tmp_path, monkeypatch):
    results, items = mixed()
    for it in items:
        it["source"] = {"handwritten": True} if it["id"] in ("C", "D") else {"request": "r", "step": 1}
    items_path, results_path = tmp_path / "dev.jsonl", tmp_path / "run.jsonl"
    write_jsonl(items, items_path)
    write_jsonl(results, results_path)
    monkeypatch.setattr(score_cli, "ROOT", tmp_path / "repo")
    for subset, ids, rate in (("handwritten", ["C", "D"], 0.5), ("drafted", ["A", "B", "E"], 1 / 3)):
        assert score_cli.main([str(results_path), "--items", str(items_path), "--subset", subset]) == 0
        out = tmp_path / "repo" / "runs" / "eval" / "scores" / f"run-{subset}.json"
        written = json.loads(out.read_text(encoding="utf-8"))
        assert [s["id"] for s in written["scores"]] == ids and written["summary"]["n"] == len(ids)
        assert written["summary"]["success_rate"] == pytest.approx(rate)


def test_the_cli_writes_the_summary_and_the_scores(tmp_path, capsys):
    items_path, results_path, _ = write_run(tmp_path)
    out = tmp_path / "results" / "run.json"
    assert score_cli.main([str(results_path), "--items", str(items_path), "--out", str(out)]) == 0
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["summary"]["success_rate"] == 0.4 and [s["id"] for s in written["scores"]] == list("ABCDE")
    assert "success 40.0%" in capsys.readouterr().out
    before = out.read_bytes()
    score_cli.main([str(results_path), "--items", str(items_path), "--out", str(out)])
    assert out.read_bytes() == before


def test_the_cli_scores_records_the_runner_wrote_for_the_drafts(tmp_path):
    items = [item(CALL, SEARCH, id_="r1-s1"),
             item(PARALLEL, GET_A, GET_B, id_="r1-s2", messages=AFTER_SEARCH),
             item(NO_CALL, id_="r2-s1", slots={})]
    results = run_items(items, Replay([turn(SEARCH), turn(GET_B, GET_A), turn(content="ok")]))
    write_jsonl(items, tmp_path / "items.jsonl")
    write_jsonl(results, tmp_path / "results.jsonl")
    out = tmp_path / "s.json"
    assert score_cli.main([str(tmp_path / "results.jsonl"), "--items", str(tmp_path / "items.jsonl"),
                           "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["summary"]["success_rate"] == 1.0


def test_the_cli_exits_1_on_malformed_input_and_2_when_ids_differ(tmp_path):
    items_path, results_path, items = write_run(tmp_path)
    bad = tmp_path / "bad.jsonl"
    bad.write_text("not json\n", encoding="utf-8")
    assert score_cli.main([str(bad), "--items", str(items_path), "--out", str(tmp_path / "o.json")]) == 1
    write_jsonl([{**items[0], "type": "chat"}, *items[1:]], tmp_path / "typed.jsonl")
    assert score_cli.main([str(results_path), "--items", str(tmp_path / "typed.jsonl"),
                           "--out", str(tmp_path / "o.json")]) == 1
    write_jsonl(items[:4], tmp_path / "four.jsonl")
    assert score_cli.main([str(results_path), "--items", str(tmp_path / "four.jsonl"),
                           "--out", str(tmp_path / "o.json")]) == 2
    assert not (tmp_path / "o.json").exists()


def test_the_cli_refuses_the_test_sets_and_the_handwritten_test_items_with_exit_4(tmp_path, monkeypatch,
                                                                                    capsys):
    """Only the phase-4 test runner scores test material: score.py refuses a test set, and the files of the
    test set's handwritten items with their gold turns, as items before reading them, as run_eval does."""
    _, results_path, items = write_run(tmp_path)
    root = tmp_path / "repo"
    (root / "data" / "eval").mkdir(parents=True)
    monkeypatch.setattr(score_cli, "ROOT", root)
    monkeypatch.chdir(root)
    for name in ("test.jsonl", "e2e-test.jsonl", "handwritten.jsonl", "handwritten-specs.jsonl"):
        write_jsonl(items, root / "data" / "eval" / name)  # well-formed: refused all the same, by location
        argv = [str(results_path), "--items", f"data/eval/{name}", "--out", str(root / "scores" / "s.json")]
        assert score_cli.main(argv) == 4, name
        assert "only the phase-4 test runner" in capsys.readouterr().err
    assert not (root / "scores").exists()


def test_the_cli_refuses_a_copy_of_test_items_by_the_items_it_holds_with_exit_4(tmp_path, monkeypatch,
                                                                                 capsys):
    """A copy of test items whose bytes are not the locked file's (here with CRLF line ends) passes the
    location and digest checks; its items (split "test") refuse it as run_eval refuses it, and nothing is
    written."""
    results, items = mixed()
    copy, results_path = tmp_path / "copy.jsonl", tmp_path / "run.jsonl"
    write_jsonl([{**it, "split": "test"} for it in items], copy)
    copy.write_bytes(copy.read_bytes().replace(b"\n", b"\r\n"))
    write_jsonl(results, results_path)
    monkeypatch.setattr(score_cli, "ROOT", tmp_path / "repo")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "scores" / "s.json"
    assert score_cli.main([str(results_path), "--items", str(copy), "--out", str(out)]) == 4
    assert "holds test material" in capsys.readouterr().err and not out.parent.exists()


def test_numbers_are_rebuilt_from_every_registered_result_and_an_unregistered_file_gives_none(tmp_path,
                                                                                                 monkeypatch):
    """--numbers runs build_report's build (research_agent.eval.report): results/numbers.json is rebuilt from
    every registered result present, the summary just written among them, and numbers written for a file no
    pattern names do not survive. A registered file that cannot be read stops the build (exit 3), and the
    summary is written all the same."""
    items_path, results_path, _ = write_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    bfcl = {"systems": {"b0": {"categories": {"simple_python": {"n": 4, "correct": 3, "accuracy": 0.75}},
                               "aggregate": {"categories": 1, "n": 4, "accuracy": 0.75}}}}
    (tmp_path / "results" / "bfcl-m3.json").write_text(json.dumps(bfcl), encoding="utf-8")
    (tmp_path / "results" / "other.json").write_text("{}\n", encoding="utf-8")
    from research_agent.eval.numbers import write_numbers
    write_numbers({"bfcl.simple": "91.0%"}, [Path("results/other.json")], root=tmp_path,
                  ignored=lambda p: False)
    assert score_cli.main([str(results_path), "--items", str(items_path), "--out", "results/dev-b0.json",
                           "--numbers"]) == 0
    report = json.loads((tmp_path / "results" / "numbers.json").read_text(encoding="utf-8"))
    assert "bfcl.simple" not in report["numbers"] and report["numbers"]["bfcl.m3.b0.aggregate"] == "75.00%"
    assert report["numbers"]["eval.dev-b0.success"] == "40.0%"
    assert report["numbers"]["eval.dev-b0.n"] == "5"
    assert report["numbers"]["eval.dev-b0.success.call"] == "50.0%"
    assert report["numbers"]["eval.dev-b0.param_f1"] == "0.550"
    assert set(report["sources"]) == {"results/bfcl-m3.json", "results/dev-b0.json"}
    before = (tmp_path / "results" / "numbers.json").read_bytes()
    (tmp_path / "results" / "bfcl-m3.json").write_text("{broken", encoding="utf-8")
    assert score_cli.main([str(results_path), "--items", str(items_path), "--out", "results/dev-b1.json",
                           "--numbers"]) == 3
    assert (tmp_path / "results" / "dev-b1.json").is_file()
    assert (tmp_path / "results" / "numbers.json").read_bytes() == before


def test_re_scoring_under_the_same_name_updates_its_numbers_and_only_a_registered_name_is_numbered(
        tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True, capture_output=True)
    items_path, results_path, _ = write_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    args = [str(results_path), "--items", str(items_path), "--out", "results/dev-b0.json", "--numbers"]
    assert score_cli.main(args) == 0
    numbers = tmp_path / "results" / "numbers.json"
    assert json.loads(numbers.read_text(encoding="utf-8"))["numbers"]["eval.dev-b0.success"] == "40.0%"
    results, _ = mixed()
    wrong = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-08-01")
    write_jsonl([ok(wrong, id_="A"), *results[1:]], results_path)  # A now fails as well
    assert score_cli.main(args) == 0
    report = json.loads(numbers.read_text(encoding="utf-8"))
    assert report["numbers"]["eval.dev-b0.success"] == "20.0%"
    assert set(report["sources"]) == {"results/dev-b0.json"}
    for out in ("results/other.json", "results/dev-e2e-b0.json", "results/test-m3-b0.json"):
        with pytest.raises(SystemExit) as exc:  # not the name of a registered dev summary (eval.report)
            score_cli.main([*args[:4], out, "--numbers"])
        assert exc.value.code == 2 and not (tmp_path / out).exists()
    assert json.loads(numbers.read_text(encoding="utf-8"))["numbers"]["eval.dev-b0.success"] == "20.0%"


def test_without_out_the_summary_goes_to_runs_eval_scores_under_the_repository_root(tmp_path, monkeypatch):
    assert (score_cli.ROOT / "pyproject.toml").is_file()
    items_path, results_path, _ = write_run(tmp_path)
    root, elsewhere = tmp_path / "repo", tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.setattr(score_cli, "ROOT", root)
    monkeypatch.chdir(elsewhere)
    assert score_cli.main([str(results_path), "--items", str(items_path)]) == 0
    assert (root / "runs" / "eval" / "scores" / "run.json").is_file()
    assert not (root / "results").exists() and not (elsewhere / "results").exists()


@pytest.mark.parametrize("out", [None, "elsewhere/dev-b0.json"])
def test_numbers_need_an_explicit_out_under_results_and_nothing_is_written_without_one(out, tmp_path,
                                                                                         monkeypatch):
    items_path, results_path, _ = write_run(tmp_path)
    monkeypatch.setattr(score_cli, "ROOT", tmp_path / "repo")
    monkeypatch.chdir(tmp_path)
    argv = [str(results_path), "--items", str(items_path), *(["--out", out] if out else []), "--numbers"]
    with pytest.raises(SystemExit) as exc:
        score_cli.main(argv)
    assert exc.value.code == 2
    assert not any(p.is_dir() for p in (tmp_path / "results", tmp_path / "elsewhere", tmp_path / "repo"))


def test_the_cli_scores_with_the_scorer_given_and_a_normalized_file_says_so(tmp_path, monkeypatch, capsys):
    """--scorer normalized: the file records "scorer": "normalized" first, and the default --out gets
    -normalized after the stem (after the subset's, with --subset); without --scorer, or with --scorer
    default, the file is today's, byte for byte."""
    items = [item(CALL, ON_TODAY, messages=asked("找 9 月以来到今天的论文。"), id_="A"),
             {**item(NO_CALL, id_="B", slots={}, messages=asked("谢谢。")), "source": {"handwritten": True}}]
    results = [ok(SEARCH, id_="A"), ok(content="不客气。", id_="B")]
    items_path, results_path = tmp_path / "dev.jsonl", tmp_path / "run.jsonl"
    write_jsonl(items, items_path)
    write_jsonl(results, results_path)
    monkeypatch.setattr(score_cli, "ROOT", tmp_path / "repo")
    scores = tmp_path / "repo" / "runs" / "eval" / "scores"
    argv = [str(results_path), "--items", str(items_path)]
    assert score_cli.main(argv) == 0
    default = (scores / "run.json").read_bytes()
    assert list(json.loads(default)) == ["summary", "scores"]
    assert json.loads(default)["summary"]["success_rate"] == 0.5
    assert score_cli.main([*argv, "--scorer", "default", "--out", str(tmp_path / "explicit.json")]) == 0
    assert (tmp_path / "explicit.json").read_bytes() == default
    assert "scorer" not in capsys.readouterr().out
    assert score_cli.main([*argv, "--scorer", "normalized"]) == 0
    normalized = json.loads((scores / "run-normalized.json").read_text(encoding="utf-8"))
    assert list(normalized) == ["scorer", "summary", "scores"] and normalized["scorer"] == "normalized"
    assert normalized["summary"]["success_rate"] == 1.0
    assert normalized["scores"] == score_items(results, items, scorer=NORMALIZED_SCORER)
    assert "scorer normalized" in capsys.readouterr().out
    assert score_cli.main([*argv, "--scorer", "normalized", "--subset", "drafted"]) == 0
    drafted = json.loads((scores / "run-drafted-normalized.json").read_text(encoding="utf-8"))
    assert [s["id"] for s in drafted["scores"]] == ["A"] and drafted["summary"]["success_rate"] == 1.0
    with pytest.raises(SystemExit) as exc:
        score_cli.main([*argv, "--scorer", "lenient"])
    assert exc.value.code == 2


def test_numbers_of_a_normalized_summary_come_under_their_own_keys_beside_the_defaults(tmp_path, monkeypatch):
    """A summary the normalized scorer made gives eval.dev-<name>-normalized.* (once, when its name already
    ends so); a default summary under a name of the normalized scorer's is refused before anything is written
    (exit 2), so the numbers stay as they were."""
    items_path, results_path, _ = write_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    argv = [str(results_path), "--items", str(items_path), "--numbers"]
    assert score_cli.main([*argv, "--out", "results/dev-b0.json"]) == 0
    assert score_cli.main([*argv, "--out", "results/dev-b0-normalized.json", "--scorer", "normalized"]) == 0
    assert score_cli.main([*argv, "--out", "results/dev-b1.json", "--scorer", "normalized"]) == 0
    numbers = tmp_path / "results" / "numbers.json"
    report = json.loads(numbers.read_text(encoding="utf-8"))
    assert report["numbers"]["eval.dev-b0.success"] == report["numbers"]["eval.dev-b0-normalized.success"] \
        == "40.0%"
    assert report["numbers"]["eval.dev-b1-normalized.n"] == "5"
    assert not [key for key in report["numbers"]
                if key.startswith(("eval.dev-b1.", "eval.dev-b0-normalized-"))]
    assert set(report["sources"]) == {"results/dev-b0.json", "results/dev-b0-normalized.json",
                                      "results/dev-b1.json"}
    before = numbers.read_bytes()
    assert score_cli.main([*argv, "--out", "results/dev-b2-normalized.json"]) == 2
    assert not (tmp_path / "results" / "dev-b2-normalized.json").exists() and numbers.read_bytes() == before


def test_an_out_named_or_made_for_another_scorer_is_refused_before_anything_is_written(tmp_path, monkeypatch,
                                                                                      capsys):
    """Exit 2, before the items are read and with nothing written: a registered summary whose name is another
    scorer's (a default summary as results/dev-<name>-normalized.json, with or without --numbers), and an
    --out, in results/ or anywhere, that holds a file another scorer made (a file without "scorer" is the
    default scorer's). A file of the same scorer, a file that is no JSON object, and a name of the normalized
    scorer's outside results/ are written as before."""
    items_path, results_path, _ = write_run(tmp_path)
    monkeypatch.chdir(tmp_path)
    argv = [str(results_path), "--items", str(items_path)]
    results = tmp_path / "results"
    for extra in ([], ["--numbers"]):
        assert score_cli.main([*argv, "--out", "results/dev-b2-normalized.json", *extra]) == 2
        assert "the normalized scorer's" in capsys.readouterr().err
    missing = [str(results_path), "--items", str(tmp_path / "missing.jsonl")]  # not read: exit 1 if it were
    assert score_cli.main([*missing, "--out", "results/dev-b2-normalized.json"]) == 2
    assert not results.exists()
    assert score_cli.main([*argv, "--out", "results/dev-b1.json"]) == 0
    default = (results / "dev-b1.json").read_bytes()
    assert score_cli.main([*argv, "--out", "results/dev-b1.json", "--scorer", "normalized"]) == 2
    assert "the default scorer made" in capsys.readouterr().err
    assert score_cli.main([*missing, "--out", "results/dev-b1.json", "--scorer", "normalized"]) == 2
    assert (results / "dev-b1.json").read_bytes() == default
    elsewhere = tmp_path / "elsewhere" / "run.json"
    assert score_cli.main([*argv, "--out", str(elsewhere), "--scorer", "normalized"]) == 0
    normalized = elsewhere.read_bytes()
    assert score_cli.main([*argv, "--out", str(elsewhere)]) == 2
    assert "the normalized scorer made" in capsys.readouterr().err
    assert elsewhere.read_bytes() == normalized
    # the same scorer's file is written over, as before
    assert score_cli.main([*argv, "--out", str(elsewhere), "--scorer", "normalized"]) == 0
    assert score_cli.main([*argv, "--out", "results/dev-b1.json"]) == 0
    assert (results / "dev-b1.json").read_bytes() == default and elsewhere.read_bytes() == normalized
    junk = tmp_path / "elsewhere" / "junk.json"
    junk.write_text("not JSON", encoding="utf-8")
    assert score_cli.main([*argv, "--out", str(junk)]) == 0
    assert json.loads(junk.read_text(encoding="utf-8"))["summary"]["n"] == 5
    unregistered = tmp_path / "elsewhere" / "dev-b2-normalized.json"
    assert score_cli.main([*argv, "--out", str(unregistered)]) == 0
    assert "scorer" not in json.loads(unregistered.read_text(encoding="utf-8"))

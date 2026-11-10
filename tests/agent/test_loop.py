import copy
import dataclasses
import json

import httpx
import pytest
from support.fakes import HITS, PAPER, ScriptedModel, call, table_check, table_decide, table_tools, turn

from research_agent.agent.context import build_context, step_contexts
from research_agent.agent.executor import ToolExecutor
from research_agent.agent.loop import (
    CLARIFYING_HINT,
    REFUSED_STEP_HINT,
    RELAX_HINT,
    REPEAT_HINT,
    STOPPED_HINT,
    TELL_USER_HINT,
    load_transcript,
    run_batch,
    run_turn,
    save_transcript,
)
from research_agent.agent.prompts import PROMPT_HASH, PROMPT_VERSION, system_prompt
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
    StepIssue,
    ToolOutcome,
)
from research_agent.canonical import canonical_json
from research_agent.llm.budget import BudgetExceeded
from research_agent.net.fetch import Fetcher
from research_agent.tools import schemas
from research_agent.tools.adapters import arxiv_tools

TODAY = "2026-10-04"
REQUEST = "帮我找 2026 年 9 月关于 speculative decoding 的论文"
NOTE = {"note_id": "2609.01234-adaptive-draft", "arxiv_id": "2609.01234", "path": "notes/2609.01234.md"}
SCRIPT = {"search_arxiv": [ToolOutcome(HITS)], "get_paper": [ToolOutcome(PAPER)],
          "write_note": [ToolOutcome(NOTE)]}
BAD_ID = ["arxiv_id: expected YYMM.NNNNN"]
QUESTION = "你想看哪个月的论文?"


def rig(script: dict | None = None, check: dict | None = None):
    """The real executor over scripted tools and a table check_call, with the tools' argument log."""
    tools, seen = table_tools(script or SCRIPT)
    return ToolExecutor(tools, table_check(check), today=TODAY), seen


def run(model, executor=None, decide=None, **kwargs):
    return run_turn(REQUEST, model=model, executor=executor or rig()[0], decide=decide or table_decide(),
                    today=TODAY, **kwargs)


def body(message: dict) -> object:
    return json.loads(message["content"])


def skipped(hint: str) -> dict:
    return {"error": schemas.SKIPPED, "hint": hint}


def unfinished_decide(event, times):
    raise NotImplementedError


# answers and ordinary steps


def test_an_answer_without_tool_calls_ends_the_turn():
    model = ScriptedModel([turn(content=QUESTION)])
    t = run(model)
    assert (t.status, t.final_text) == (ANSWERED, QUESTION)
    assert t.messages == [{"role": "system", "content": system_prompt(TODAY)},
                          {"role": "user", "content": REQUEST}, {"role": "assistant", "content": QUESTION}]
    assert model.contexts == [t.messages[:2]]
    assert t.steps == [{"index": 1, "model": "fake", "executed": [], "decisions": [], "attempts": [
        {"model": "fake", "content": QUESTION, "tool_calls": [], "issues": [], "action": None}]}]


def cached(assistant_turn):
    """The turn as the response cache serves it: not paid."""
    return dataclasses.replace(assistant_turn, paid=False, cached=True)


def test_a_two_step_run_keeps_the_history_the_step_records_and_the_accounts():
    search = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-01")
    first = turn(search, usage={"prompt_tokens": 1100, "completion_tokens": 30})
    answer = cached(turn(content="找到两篇: 2609.01234 与 2609.05678。",
                         usage={"prompt_tokens": 1400, "completion_tokens": 20}))
    ex, seen = rig()
    t = run(ScriptedModel([first, answer]), ex, cost_of=lambda tr: 0.002 if tr.paid else 0.0)
    assert (t.status, t.final_text) == (ANSWERED, answer.content)
    assert t.messages[2:] == [first.to_api(),
                              {"role": "tool", "tool_call_id": "c1", "content": canonical_json(HITS)},
                              answer.to_api()]
    assert seen["search_arxiv"] == [{"query": "speculative decoding", "date_from": "2026-09-01"}]
    assert t.steps[0] == {"index": 1, "model": "fake", "decisions": [],
                          "attempts": [{"model": "fake", "content": "", "tool_calls": [search.to_api()],
                                        "issues": [], "action": None}],
                          "executed": [{"call_id": "c1", "name": "search_arxiv", "error": None}]}
    assert [s["index"] for s in t.steps] == [1, 2]
    assert (t.paid_calls, t.cache_hits, t.local_calls) == (1, 1, 0) and t.cost_rmb == pytest.approx(0.002)
    assert t.usage == {"prompt_tokens": 2500, "completion_tokens": 50}
    assert (t.request, t.today) == (REQUEST, TODAY)
    assert (t.schema_hash, t.prompt_version, t.prompt_hash) == (schemas.SCHEMA_HASH, PROMPT_VERSION,
                                                                PROMPT_HASH)


def test_every_model_turn_counts_once_as_paid_cached_or_local():
    paid = turn(call("search_arxiv", "c1", query="speculative decoding"))
    from_cache = cached(turn(call("get_paper", "c2", arxiv_id="2609.01234")))
    rejected = turn(call("get_paper", "c3", arxiv_id="12"), paid=False)  # neither: the local model
    model = ScriptedModel([paid, from_cache, rejected, turn(content="done", paid=False)])
    t = run(model, rig(check={'get_paper:{"arxiv_id":"12"}': BAD_ID})[0],
            table_decide({("invalid_args", 1): "repair"}))
    assert t.status == ANSWERED and len(model.contexts) == 4  # the rejected attempt counts too
    assert (t.paid_calls, t.cache_hits, t.local_calls) == (1, 1, 2)


def test_parallel_calls_run_and_are_answered_in_call_order():
    log = []

    def tool(name: str, outcome: ToolOutcome):
        def run_tool(args: dict) -> ToolOutcome:
            log.append((name, args))
            return outcome

        return run_tool

    tools = {"get_paper": tool("get_paper", ToolOutcome(PAPER)),
             "search_arxiv": tool("search_arxiv", ToolOutcome(HITS))}
    calls = [call("get_paper", "c1", arxiv_id="2609.05678"),
             call("search_arxiv", "c2", query="tree verification"),
             call("get_paper", "c3", arxiv_id="2609.01234")]
    ex = ToolExecutor(tools, table_check(), today=TODAY)
    t = run(ScriptedModel([turn(*calls), turn(content="done")]), ex)
    assert t.status == ANSWERED
    assert log == [("get_paper", {"arxiv_id": "2609.05678"}),
                   ("search_arxiv", {"query": "tree verification"}),
                   ("get_paper", {"arxiv_id": "2609.01234"})]
    assert [m["tool_call_id"] for m in t.messages[3:6]] == ["c1", "c2", "c3"]
    assert [e["call_id"] for e in t.steps[0]["executed"]] == ["c1", "c2", "c3"]


# ask_clarification


def test_ask_clarification_ends_the_turn_with_its_question():
    ask = call("ask_clarification", "c1", question=QUESTION, missing=["date_range"])
    model = ScriptedModel([turn(ask), turn(content="never asked")])
    t = run(model)
    assert (t.status, t.final_text) == (CLARIFICATION, QUESTION)
    assert t.messages[-1] == {"role": "tool", "tool_call_id": "c1",
                              "content": canonical_json({"question": QUESTION, "status": "asked"})}
    assert t.steps[0]["executed"] == [{"call_id": "c1", "name": "ask_clarification", "error": None}]
    assert len(model.turns) == 1


def test_ask_clarification_with_other_calls_runs_none_of_them():
    calls = [call("search_arxiv", "c1", query="speculative decoding"),
             call("ask_clarification", "c2", question=QUESTION),
             call("get_paper", "c3", arxiv_id="2609.01234")]
    ex, seen = rig()
    t = run(ScriptedModel([turn(*calls)]), ex)
    assert (t.status, t.final_text) == (CLARIFICATION, QUESTION)
    assert [(m["tool_call_id"], body(m)) for m in t.messages[3:]] == [
        ("c1", skipped(CLARIFYING_HINT)), ("c2", {"question": QUESTION, "status": "asked"}),
        ("c3", skipped(CLARIFYING_HINT))]
    assert seen["search_arxiv"] == [] and seen["get_paper"] == []
    assert [e["error"] for e in t.steps[0]["executed"]] == ["SKIPPED", None, "SKIPPED"]


# invalid steps: repair, handoff, limits


def test_a_repaired_step_replaces_the_rejected_one_which_never_enters_the_history():
    bad = turn(call("get_paper", "c1", arxiv_id="2609.1234"))
    good = turn(call("get_paper", "c2", arxiv_id="2609.01234"))
    model = ScriptedModel([bad, good, turn(content="done")])
    decide = table_decide({("invalid_args", 1): "repair"})
    ex, seen = rig(check={'get_paper:{"arxiv_id":"2609.1234"}': BAD_ID})
    t = run(model, ex, decide)
    assert t.status == ANSWERED
    # the repair request is the same context plus the rejected step and its error ...
    error = canonical_json({"error": "INVALID_ARGUMENTS", "details": BAD_ID})
    assert model.contexts[1] == model.contexts[0] + [bad.to_api(),
                                                     {"role": "tool", "tool_call_id": "c1", "content": error}]
    # ... but the history holds only the accepted step, and the next request shows no trace of the rejection
    assert t.messages[2] == good.to_api() and "2609.1234" not in canonical_json(t.messages)
    assert model.contexts[2] == build_context(t.messages[:4])
    assert seen["get_paper"] == [{"arxiv_id": "2609.01234"}]
    attempts = t.steps[0]["attempts"]
    assert [a["tool_calls"] for a in attempts] == [[c.to_api() for c in bad.tool_calls],
                                                   [c.to_api() for c in good.tool_calls]]
    assert attempts[0]["issues"] == [{"call_id": "c1", "event": "invalid_args", "messages": BAD_ID}]
    assert [a["action"] for a in attempts] == ["repair", None]
    assert t.steps[0]["decisions"] == [{"call_id": "c1", "event": "invalid_args", "times": 1,
                                        "action": "repair"}]
    assert decide.asked == [("invalid_args", 1)] and t.paid_calls == 3


def test_the_repair_request_answers_every_call_of_the_rejected_step():
    rejected = turn(call("search_papers", "c1", query="x"), call("get_paper", "c2", raw='{"arxiv_id": '),
                    call("get_paper", "c3", arxiv_id="2609.01234"))
    model = ScriptedModel([rejected, turn(content="done")])
    decide = table_decide({("unknown_tool", 1): "repair"})
    t = run(model, decide=decide)
    side = model.contexts[1][len(model.contexts[0]):]
    assert side[0] == rejected.to_api()
    replies = {m["tool_call_id"]: body(m) for m in side[1:]}
    assert list(replies) == ["c1", "c2", "c3"]
    assert replies["c1"]["error"] == "UNKNOWN_TOOL" and "'search_papers'" in replies["c1"]["details"][0]
    assert replies["c2"] == {"error": "INVALID_JSON",
                             "details": ["arguments are not valid JSON: Expecting value at character 13"]}
    assert replies["c3"] == skipped(REFUSED_STEP_HINT)  # valid, but not run: the whole step is redone
    assert decide.asked == [("unknown_tool", 1)]  # the first issue's event is the one decided
    assert (t.status, t.messages[2]) == (ANSWERED, {"role": "assistant", "content": "done"})


def test_a_step_still_invalid_after_two_repairs_ends_as_invalid_output():
    bad = turn(call("get_paper", "c1", arxiv_id="12"))
    model = ScriptedModel([bad, bad, bad, turn(content="never asked")])
    decide = table_decide({("invalid_args", 1): "repair", ("invalid_args", 2): "repair"})
    t = run(model, rig(check={"get_paper": BAD_ID})[0], decide)
    assert (t.status, t.final_text) == (INVALID_OUTPUT, None)
    assert len(t.messages) == 2  # nothing was accepted
    assert len(model.contexts) == 3 and len(model.turns) == 1
    assert decide.asked == [("invalid_args", 1), ("invalid_args", 2), ("invalid_args", 3)]
    assert [a["action"] for a in t.steps[0]["attempts"]] == ["repair", "repair", "stop"]


def test_more_than_three_calls_in_a_step_is_refused_for_every_call():
    four = [call("get_paper", f"c{i}", arxiv_id=f"2609.0000{i}") for i in range(1, 5)]
    model = ScriptedModel([turn(*four), turn(*four[:3]), turn(content="done")])
    ex, seen = rig()
    t = run(model, ex, table_decide({("invalid_args", 1): "repair"}))
    message = "4 tool calls in one step; at most 3 are allowed"
    assert t.steps[0]["attempts"][0]["issues"] == [{"call_id": "", "event": "invalid_args",
                                                    "messages": [message]}]
    side = model.contexts[1][len(model.contexts[0]) + 1:]
    assert [(m["tool_call_id"], body(m)) for m in side] == [
        (f"c{i}", {"error": "INVALID_ARGUMENTS", "details": [message]}) for i in range(1, 5)]
    assert t.status == ANSWERED and seen["get_paper"] == [{"arxiv_id": f"2609.0000{i}"} for i in range(1, 4)]


@pytest.mark.parametrize("kwargs, calls", [({}, 3), ({"max_model_calls_per_step": 2}, 2)])
def test_a_step_gets_at_most_max_model_calls_per_step(kwargs, calls):
    bad = turn(call("get_paper", "c1", arxiv_id="12"))
    model = ScriptedModel([bad] * 4)
    t = run(model, rig(check={"get_paper": BAD_ID})[0], table_decide(default="repair"), **kwargs)
    assert (t.status, len(model.contexts), len(model.turns)) == (INVALID_OUTPUT, calls, 4 - calls)
    assert [a["action"] for a in t.steps[0]["attempts"]] == ["repair"] * calls  # no model call was left


def test_handoff_asks_the_handoff_model_with_the_original_context_for_the_rest_of_the_turn():
    local = ScriptedModel([turn(call("get_paper", "local_0", raw='{"arxiv_id": '), model="local")],
                          name="local")
    accepted = turn(call("get_paper", "c1", arxiv_id="2609.01234"), model="cloud")
    cloud = ScriptedModel([accepted, turn(content="done", model="cloud")], name="cloud")
    t = run(local, decide=table_decide({("bad_json", 1): "handoff"}), handoff_model=cloud)
    assert (t.status, t.final_text) == (ANSWERED, "done")
    assert cloud.contexts[0] == local.contexts[0]  # the rejected step is not shown to the handoff model
    assert len(local.contexts) == 1 and len(cloud.contexts) == 2  # step 2 stays with the handoff model
    assert t.messages[2] == accepted.to_api()
    assert [s["model"] for s in t.steps] == ["cloud", "cloud"]
    assert [(a["model"], a["action"]) for a in t.steps[0]["attempts"]] == [("local", "handoff"),
                                                                           ("cloud", None)]


def test_after_a_repair_the_handoff_model_still_gets_the_original_context():
    bad = turn(call("get_paper", "local_0", raw='{"arxiv_id": '), model="local")
    local = ScriptedModel([bad, bad], name="local")
    cloud = ScriptedModel([turn(call("get_paper", "c1", arxiv_id="2609.01234"), model="cloud"),
                           turn(content="done", model="cloud")], name="cloud")
    decide = table_decide({("bad_json", 1): "repair", ("bad_json", 2): "handoff"})  # the plan's policy
    t = run(local, decide=decide, handoff_model=cloud)
    assert t.status == ANSWERED
    assert len(local.contexts[1]) == len(local.contexts[0]) + 2  # the repair request carried the rejection
    assert cloud.contexts[0] == local.contexts[0]  # the handoff request does not
    assert [a["action"] for a in t.steps[0]["attempts"]] == ["repair", "handoff", None]
    assert decide.asked == [("bad_json", 1), ("bad_json", 2)]


def test_handoff_without_a_handoff_model_is_invalid_output():
    model = ScriptedModel([turn(call("get_paper", "c1", raw="{"))])
    t = run(model, decide=table_decide({("bad_json", 1): "handoff"}))
    assert (t.status, len(model.contexts)) == (INVALID_OUTPUT, 1)


def test_a_handoff_decided_with_no_model_call_left_does_not_switch_models():
    bad = turn(call("get_paper", "c1", raw="{"), model="local")
    local = ScriptedModel([bad, bad, bad], name="local")
    cloud = ScriptedModel([turn(content="never asked", model="cloud")], name="cloud")
    decide = table_decide({("bad_json", 1): "repair", ("bad_json", 2): "repair", ("bad_json", 3): "handoff"})
    t = run(local, decide=decide, handoff_model=cloud)
    assert (t.status, t.steps[0]["model"], len(cloud.contexts)) == (INVALID_OUTPUT, "local", 0)
    assert [a["action"] for a in t.steps[0]["attempts"]] == ["repair", "repair", "handoff"]


def test_times_restarts_at_one_in_every_step_for_the_step_check_events():
    bad_json, bad_args = call("get_paper", "c1", raw='{"arxiv_id": '), call("get_paper", "c2", arxiv_id="12")
    good = [turn(call("get_paper", f"g{i}", arxiv_id=f"2609.0000{i}")) for i in range(1, 4)]
    model = ScriptedModel([turn(bad_json), good[0], turn(bad_args), good[1], turn(bad_json), good[2],
                           turn(content="done")])
    decide = table_decide(default="repair")
    t = run(model, rig(check={'get_paper:{"arxiv_id":"12"}': BAD_ID})[0], decide)
    assert (t.status, len(t.steps)) == (ANSWERED, 4)
    assert decide.asked == [("bad_json", 1), ("invalid_args", 1), ("bad_json", 1)]  # per step, not per turn
    assert [[d["times"] for d in s["decisions"]] for s in t.steps] == [[1], [1], [1], []]


def test_repair_once_then_hand_off_holds_in_every_step():
    bad = [turn(call("get_paper", f"c{i}", arxiv_id="12"), model="local") for i in (1, 3)]
    local = ScriptedModel([bad[0], turn(call("get_paper", "c2", arxiv_id="2609.01234"), model="local"),
                           bad[1], turn(content="done", model="local")], name="local")
    cloud = ScriptedModel([turn(content="never asked", model="cloud")], name="cloud")
    decide = table_decide({("invalid_args", 1): "repair", ("invalid_args", 2): "handoff"})  # the plan's rule
    t = run(local, rig(check={'get_paper:{"arxiv_id":"12"}': BAD_ID})[0], decide, handoff_model=cloud)
    assert (t.status, t.final_text) == (ANSWERED, "done")
    assert decide.asked == [("invalid_args", 1), ("invalid_args", 1)]  # step 2 gets its own repair
    assert cloud.contexts == [] and [s["model"] for s in t.steps] == ["local", "local"]


def test_times_counts_the_rejected_attempts_of_the_step_whichever_event_each_had():
    local = ScriptedModel([turn(call("get_paper", "c1", raw="{"), model="local"),
                           turn(call("get_paper", "c2", arxiv_id="12"), model="local")], name="local")
    cloud = ScriptedModel([turn(content="done", model="cloud")], name="cloud")
    decide = table_decide({("bad_json", 1): "repair", ("invalid_args", 2): "handoff"})
    t = run(local, rig(check={"get_paper": BAD_ID})[0], decide, handoff_model=cloud)
    assert (t.status, t.final_text) == (ANSWERED, "done")
    assert decide.asked == [("bad_json", 1), ("invalid_args", 2)]  # the step's 2nd rejection: no 2nd repair
    assert [(a["model"], a["action"]) for a in t.steps[0]["attempts"]] == [
        ("local", "repair"), ("local", "handoff"), ("cloud", None)]


def test_empty_results_and_repeat_call_keep_counting_over_the_whole_turn():
    ex, _ = rig({"search_arxiv": [ToolOutcome([], schemas.EMPTY_RESULTS)]})
    x = [call("search_arxiv", f"c{i}", query="x") for i in range(1, 5)]
    model = ScriptedModel([turn(x[0]), turn(x[1]), turn(x[2]), turn(x[3]), turn(content="没有找到。")])
    decide = table_decide({("empty_results", 1): "relax", ("empty_results", 2): "tell_user",
                           ("repeat_call", 1): "tell_user", ("repeat_call", 2): "tell_user"})
    t = run(model, ex, decide)
    assert (t.status, len(t.steps)) == (ANSWERED, 5)
    assert decide.asked == [("empty_results", 1), ("empty_results", 2),
                            ("repeat_call", 1), ("repeat_call", 2)]


def test_a_second_handoff_has_nowhere_to_go():
    bad = call("get_paper", "c1", raw="{")
    local = ScriptedModel([turn(bad, model="local")], name="local")
    cloud = ScriptedModel([turn(bad, model="cloud"), turn(content="never asked")], name="cloud")
    decide = table_decide(default="handoff")
    t = run(local, decide=decide, handoff_model=cloud)
    assert (t.status, len(cloud.contexts)) == (INVALID_OUTPUT, 1)
    assert decide.asked == [("bad_json", 1), ("bad_json", 2)]


# repeated calls and empty results


def test_the_third_identical_call_is_a_dead_loop_with_defaults_filled_in():
    x1 = call("search_arxiv", "c1", query="x")
    x2 = call("search_arxiv", "c2", query="x", max_results=5)
    x3 = call("search_arxiv", "c3", raw='{"sort_by": "relevance", "query": "x"}')
    other = call("get_paper", "c4", arxiv_id="2609.01234")
    model = ScriptedModel([turn(x1), turn(x2), turn(x3, other), turn(content="never asked")])
    decide = table_decide({("repeat_call", 1): "stop"})
    ex, seen = rig()
    t = run(model, ex, decide)
    assert (t.status, t.final_text) == (DEAD_LOOP, None)
    assert len(seen["search_arxiv"]) == 2 and seen["get_paper"] == []
    assert [(m["tool_call_id"], body(m)) for m in t.messages[-2:]] == [("c3", skipped(REPEAT_HINT)),
                                                                       ("c4", skipped(STOPPED_HINT))]
    assert t.steps[2]["decisions"] == [{"call_id": "c3", "event": "repeat_call", "times": 1,
                                        "action": "stop"}]
    assert decide.asked == [("repeat_call", 1)]


def test_a_repeated_call_is_answered_with_a_hint_unless_decide_stops():
    x = [call("search_arxiv", f"c{i}", query="x") for i in range(1, 5)]
    other = call("get_paper", "c9", arxiv_id="2609.01234")
    model = ScriptedModel([turn(x[0]), turn(x[1]), turn(x[2], other), turn(x[3])])
    decide = table_decide({("repeat_call", 1): "tell_user"})  # not stop; the default for times 2 is stop
    ex, seen = rig()
    t = run(model, ex, decide)
    assert body(t.messages[7]) == skipped(REPEAT_HINT) and body(t.messages[8]) == PAPER  # the step went on
    assert (t.status, body(t.messages[-1])) == (DEAD_LOOP, skipped(REPEAT_HINT))
    assert decide.asked == [("repeat_call", 1), ("repeat_call", 2)]
    assert len(seen["search_arxiv"]) == 2 and seen["get_paper"] == [{"arxiv_id": "2609.01234"}]


def test_empty_results_carry_the_relax_hint_then_the_tell_user_hint():
    assert RELAX_HINT == "放宽一次检索条件(例如去掉日期或分类限制)再试一次"
    assert TELL_USER_HINT == "告诉用户没有找到,不要编造论文编号"
    ex, _ = rig({"search_arxiv": [ToolOutcome([], schemas.EMPTY_RESULTS)]})
    model = ScriptedModel([turn(call("search_arxiv", "c1", query="x", date_from="2026-09-01")),
                           turn(call("search_arxiv", "c2", query="x")), turn(content="没有找到。")])
    decide = table_decide({("empty_results", 1): "relax", ("empty_results", 2): "tell_user"})
    t = run(model, ex, decide)
    assert (t.status, t.final_text) == (ANSWERED, "没有找到。")
    assert body(t.messages[3]) == {"error": "EMPTY_RESULTS", "hint": RELAX_HINT, "items": []}
    assert body(t.messages[5]) == {"error": "EMPTY_RESULTS", "hint": TELL_USER_HINT, "items": []}
    assert decide.asked == [("empty_results", 1), ("empty_results", 2)]
    assert [e["error"] for s in t.steps for e in s["executed"]] == ["EMPTY_RESULTS", "EMPTY_RESULTS"]


def test_tell_user_on_empty_results_keeps_the_keys_of_dict_content():
    content = {"query": "推测解码", "notes": []}
    ex, _ = rig({"search_notes": [ToolOutcome(content, schemas.EMPTY_RESULTS)]})
    model = ScriptedModel([turn(call("search_notes", "c1", query="推测解码")), turn(content="没有。")])
    t = run(model, ex, table_decide({("empty_results", 1): "tell_user"}))
    assert body(t.messages[3]) == {"error": "EMPTY_RESULTS", "hint": TELL_USER_HINT, **content}
    assert t.status == ANSWERED  # empty results never end the turn


# decide's answers: each allowed one is carried out as given, any other is a harness error


def provoke(event: str) -> tuple[list, ToolExecutor]:
    """Scripted turns and an executor that make event the first question decide is asked."""
    x = [call("search_arxiv", f"c{i}", query="x") for i in range(1, 4)]
    turns = {"bad_json": [turn(call("get_paper", "c1", raw="{"))],
             "unknown_tool": [turn(call("search_papers", "c1", query="x"))],
             "invalid_args": [turn(call("get_paper", "c1", arxiv_id="12"))],
             "empty_results": [turn(x[0])],
             "repeat_call": [turn(c) for c in x]}[event]
    empty = {"search_arxiv": [ToolOutcome([], schemas.EMPTY_RESULTS)]}
    return turns, rig(empty if event == "empty_results" else SCRIPT, check={"get_paper": BAD_ID})[0]


def last_hint(t) -> str | None:
    """The hint in the last tool message of the history, if any."""
    bodies = [body(m) for m in t.messages if m["role"] == "tool"]
    return bodies[-1].get("hint") if bodies and isinstance(bodies[-1], dict) else None


# (status, final text, requests to the model, requests to the handoff model, last hint) per allowed answer
CARRIED_OUT = {
    **{(event, "repair"): (ANSWERED, "done", 2, 0, None) for event in schemas.PER_STEP_EVENTS},
    **{(event, "handoff"): (ANSWERED, "handed off", 1, 1, None) for event in schemas.PER_STEP_EVENTS},
    **{(event, "stop"): (INVALID_OUTPUT, None, 1, 0, None) for event in schemas.PER_STEP_EVENTS},
    ("empty_results", "relax"): (ANSWERED, "done", 2, 0, RELAX_HINT),
    ("empty_results", "tell_user"): (ANSWERED, "done", 2, 0, TELL_USER_HINT),
    ("repeat_call", "stop"): (DEAD_LOOP, None, 3, 0, REPEAT_HINT),
    ("repeat_call", "tell_user"): (ANSWERED, "done", 4, 0, REPEAT_HINT),
}
ALLOWED = [(event, action) for event, actions in schemas.ALLOWED_ACTIONS.items() for action in actions]
FORBIDDEN = [(event, action) for event in schemas.EVENTS for action in schemas.ACTIONS
             if action not in schemas.ALLOWED_ACTIONS[event]]


@pytest.mark.parametrize("event, action", ALLOWED)
def test_every_allowed_answer_is_carried_out_as_given(event, action):
    turns, ex = provoke(event)
    model = ScriptedModel(turns + [turn(content="done")])
    cloud = ScriptedModel([turn(content="handed off", model="cloud")], name="cloud")
    t = run(model, ex, table_decide(default=action), handoff_model=cloud)
    outcome = (t.status, t.final_text, len(model.contexts), len(cloud.contexts), last_hint(t))
    assert outcome == CARRIED_OUT[(event, action)]  # a KeyError: an allowed answer nobody taught the loop


@pytest.mark.parametrize("event, action", FORBIDDEN)
def test_an_answer_outside_the_events_allowed_actions_is_a_harness_error(event, action):
    turns, ex = provoke(event)
    model = ScriptedModel(turns + [turn(content="never asked")])
    cloud = ScriptedModel([turn(content="never asked", model="cloud")], name="cloud")
    t = run(model, ex, table_decide(default=action), handoff_model=cloud)
    assert (t.status, t.final_text) == (HARNESS_ERROR, None)
    message = t.steps[-1]["harness_error"]
    assert f"decide({event!r}, 1)" in message and f"got {action!r}" in message  # event, times and answer
    assert (len(model.turns), cloud.contexts, t.steps[-1]["decisions"]) == (1, [], [])  # nothing carried out
    call_ids = [c["id"] for m in t.messages if m["role"] == "assistant" for c in m.get("tool_calls", [])]
    assert call_ids == [m["tool_call_id"] for m in t.messages if m["role"] == "tool"]  # still well-formed


# limits and stops


def test_the_turn_stops_after_eight_steps():
    model = ScriptedModel([turn(call("get_paper", f"c{i}", arxiv_id=f"2609.{i:05d}")) for i in range(1, 10)])
    t = run(model)
    assert (t.status, t.final_text) == (STEP_LIMIT, None)
    assert [s["index"] for s in t.steps] == list(range(1, 9))
    assert len(model.contexts) == 8 and len(model.turns) == 1
    assert t.messages[-1]["role"] == "tool"  # the 8th step ran; the model was not asked again


def test_max_steps_is_a_parameter():
    model = ScriptedModel([turn(call("get_paper", f"c{i}", arxiv_id=f"2609.{i:05d}")) for i in range(1, 4)])
    t = run(model, max_steps=2)
    assert (t.status, len(t.steps), len(model.turns)) == (STEP_LIMIT, 2, 1)


def test_budget_exceeded_ends_the_turn_as_budget_stop():
    search = turn(call("search_arxiv", "c1", query="speculative decoding"))
    t = run(ScriptedModel([search, BudgetExceeded("stage M1: spent 10.0000 of 10.0 RMB")]))
    assert (t.status, t.final_text) == (BUDGET_STOP, None)
    assert t.messages[2:] == [search.to_api(),
                              {"role": "tool", "tool_call_id": "c1", "content": canonical_json(HITS)}]
    assert len(t.steps) == 2 and t.steps[1]["attempts"] == []
    assert (t.paid_calls, t.cache_hits) == (1, 0)


def test_budget_exceeded_during_a_repair_also_stops_the_turn():
    model = ScriptedModel([turn(call("get_paper", "c1", arxiv_id="12")),
                           BudgetExceeded("all stages: spent 300.0000 of 300.0 RMB")])
    t = run(model, rig(check={"get_paper": BAD_ID})[0], table_decide({("invalid_args", 1): "repair"}))
    assert (t.status, len(t.messages)) == (BUDGET_STOP, 2)
    assert [a["action"] for a in t.steps[0]["attempts"]] == ["repair"]


def test_a_context_over_the_token_budget_ends_the_turn_before_the_model_is_asked():
    limit = schemas.CONTEXT_TOKENS - schemas.REPLY_RESERVE_TOKENS  # 4,096 - 256 = 3,840 prompt tokens
    counted = []

    def count_tokens(messages: list[dict], tools: list[dict]) -> int:
        counted.append((len(messages), tools))
        return limit if len(messages) <= 2 else limit + 1

    model = ScriptedModel([turn(call("search_arxiv", "c1", query="x")), turn(content="never asked")])
    t = run(model, count_tokens=count_tokens)
    assert (t.status, t.final_text) == (CONTEXT_OVERFLOW, None)
    assert counted == [(2, schemas.MODEL_TOOLS), (4, schemas.MODEL_TOOLS)]  # the tool schemas count too
    assert (len(model.contexts), len(t.steps), len(t.messages)) == (1, 2, 4)
    assert t.steps[1]["attempts"] == []  # step 2 began, but its request was never sent


def test_a_repair_request_over_the_token_budget_ends_the_turn_before_it_is_sent():
    limit = schemas.CONTEXT_TOKENS - schemas.REPLY_RESERVE_TOKENS
    counted = []

    def count_tokens(messages: list[dict], tools: list[dict]) -> int:
        counted.append(len(messages))
        return limit + 300 * (len(messages) - 2)  # system prompt and request fit exactly; +300 per message

    model = ScriptedModel([turn(call("get_paper", "c1", arxiv_id="12")), turn(content="never asked")])
    t = run(model, rig(check={"get_paper": BAD_ID})[0], table_decide({("invalid_args", 1): "repair"}),
            count_tokens=count_tokens)
    assert (t.status, t.final_text) == (CONTEXT_OVERFLOW, None)
    assert counted == [2, 4] and len(model.contexts) == 1  # the repair request was measured and never sent
    assert [a["action"] for a in t.steps[0]["attempts"]] == ["repair"] and len(t.messages) == 2


def test_every_request_is_measured_before_it_is_sent_repair_and_handoff_requests_included():
    counted = []

    def count_tokens(messages: list[dict], tools: list[dict]) -> int:
        counted.append(copy.deepcopy(messages))
        return 1000

    bad = turn(call("get_paper", "local_0", raw='{"arxiv_id": '), model="local")
    local = ScriptedModel([bad, bad], name="local")
    cloud = ScriptedModel([turn(call("get_paper", "c1", arxiv_id="2609.01234"), model="cloud"),
                           turn(content="done", model="cloud")], name="cloud")
    decide = table_decide({("bad_json", 1): "repair", ("bad_json", 2): "handoff"})
    t = run(local, decide=decide, handoff_model=cloud, count_tokens=count_tokens)
    assert t.status == ANSWERED
    assert counted == local.contexts + cloud.contexts  # first request, repair, handoff, step 2: as sent


# harness errors


def test_a_harness_error_in_a_tool_ends_the_turn_and_is_never_shown():
    miss = HarnessError("no cassette for https://export.arxiv.org/api/query?id_list=2609.01234")

    def replaying(args: dict) -> ToolOutcome:
        raise miss

    tools, _ = table_tools({"search_arxiv": [ToolOutcome(HITS)]})
    ex = ToolExecutor({**tools, "get_paper": replaying}, table_check(), today=TODAY)
    calls = [call("search_arxiv", "c1", query="x"), call("get_paper", "c2", arxiv_id="2609.01234"),
             call("search_arxiv", "c3", query="y")]
    t = run(ScriptedModel([turn(*calls), turn(content="never asked")]), ex)
    assert (t.status, t.final_text) == (HARNESS_ERROR, None)
    assert t.steps[0]["harness_error"] == str(miss)
    # every call is still answered, so the history stays well-formed
    assert [(m["tool_call_id"], body(m)) for m in t.messages[3:]] == [
        ("c1", HITS), ("c2", skipped(STOPPED_HINT)), ("c3", skipped(STOPPED_HINT))]
    assert "cassette" not in canonical_json(t.messages)


def test_a_failing_check_call_is_a_harness_error_and_nothing_is_accepted():
    def unfinished(tool, args, seen_ids, today):
        raise NotImplementedError

    t = run(ScriptedModel([turn(call("get_paper", "c1", arxiv_id="2609.01234"))]),
            ToolExecutor({}, unfinished, today=TODAY))
    assert (t.status, len(t.messages)) == (HARNESS_ERROR, 2)
    assert "NotImplementedError" in t.steps[0]["harness_error"]
    assert t.steps[0]["attempts"][0]["issues"] is None  # the step was never checked


@pytest.mark.parametrize("decide, message", [(lambda event, times: "retry", "got 'retry'"),
                                             (lambda event, times: None, "got None"),
                                             (unfinished_decide, "NotImplementedError")])
def test_a_broken_decide_is_a_harness_error(decide, message):
    model = ScriptedModel([turn(call("get_paper", "c1", arxiv_id="12"))])
    t = run(model, rig(check={"get_paper": BAD_ID})[0], decide)
    assert t.status == HARNESS_ERROR and message in t.steps[0]["harness_error"]
    assert t.steps[0]["decisions"] == []


def test_a_broken_decide_during_execution_leaves_the_history_well_formed():
    ex, _ = rig({"search_arxiv": [ToolOutcome([], schemas.EMPTY_RESULTS)]})
    calls = [call("search_arxiv", "c1", query="x"), call("get_paper", "c2", arxiv_id="2609.01234")]
    t = run(ScriptedModel([turn(*calls)]), ex, unfinished_decide)
    assert t.status == HARNESS_ERROR
    assert [(m["tool_call_id"], body(m)) for m in t.messages[3:]] == [("c1", skipped(STOPPED_HINT)),
                                                                      ("c2", skipped(STOPPED_HINT))]


# transcripts and batches


def test_a_saved_transcript_loads_back_equal(tmp_path):
    model = ScriptedModel([turn(call("search_arxiv", "c1", query="speculative decoding")),
                           turn(content="找到两篇。")])
    t = run(model, cost_of=lambda tr: 0.0028)
    path = tmp_path / "runs" / "m1" / "turn.json"
    save_transcript(t, path)
    assert load_transcript(path) == t
    raw = path.read_bytes()
    assert raw == (t.to_json() + "\n").encode("utf-8") and b"\r\n" not in raw
    assert REQUEST.encode("utf-8") in raw  # plain UTF-8, no \u escapes
    assert json.loads(raw)["status"] == ANSWERED


def test_run_batch_stops_after_a_budget_stop():
    model = ScriptedModel([turn(content="答一"), BudgetExceeded("stage M1: spent 10.0 of 10.0 RMB"),
                           turn(content="never asked")])
    transcripts = run_batch(["问一", "问二", "问三"], model=model, executor=rig()[0], decide=table_decide(),
                            today=TODAY)
    assert [(t.request, t.status) for t in transcripts] == [("问一", ANSWERED), ("问二", BUDGET_STOP)]
    assert len(model.turns) == 1


def test_the_turn_and_the_executor_must_agree_on_today():
    model = ScriptedModel([turn(content="never asked")])
    with pytest.raises(ValueError, match="2026-10-05"):
        run_turn(REQUEST, model=model, executor=rig()[0], decide=table_decide(), today="2026-10-05")
    assert model.contexts == []


def test_run_batch_runs_every_request_and_each_turn_starts_without_seen_ids():
    check = table_check()
    ex = ToolExecutor(table_tools(SCRIPT)[0], check, today=TODAY)
    model = ScriptedModel([turn(call("search_arxiv", "c1", query="x")), turn(content="a"),
                           turn(call("get_paper", "c2", arxiv_id="2609.01234")), turn(content="b")])
    transcripts = run_batch(["r1", "r2"], model=model, executor=ex, decide=table_decide(), today=TODAY)
    assert [t.status for t in transcripts] == [ANSWERED, ANSWERED]
    assert [seen_ids for _, _, seen_ids, _ in check.calls] == [[], []]  # r1's ids do not carry over to r2


# integration: the real executor and the real context builder


def test_integration_with_the_real_executor_and_context_builder():
    tools, seen = table_tools(SCRIPT)
    check = table_check()
    search = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-01")
    model = ScriptedModel([turn(search), turn(call("get_paper", "c2", arxiv_id="2609.01234")),
                           turn(call("write_note", "c3", arxiv_id="2609.01234")),
                           turn(content="已为 2609.01234 写好笔记。")])
    counted = []

    def count_tokens(messages: list[dict], tools: list[dict]) -> int:
        counted.append(copy.deepcopy(messages))
        return 1000

    ex = ToolExecutor(tools, check, today=TODAY)
    t = run_turn(REQUEST, model=model, executor=ex, decide=table_decide(), today=TODAY,
                 count_tokens=count_tokens)
    assert (t.status, t.final_text) == (ANSWERED, "已为 2609.01234 写好笔记。")
    hits = canonical_json(HITS)
    assert model.contexts[1][3]["content"] == hits  # second request: the search is the latest step, whole
    assert json.loads(model.contexts[2][3]["content"]) == [  # third: a later step exists, so it is compressed
        {"arxiv_id": h["arxiv_id"], "title": h["title"]} for h in HITS]
    assert model.contexts[2][4:] == t.messages[4:6]  # while the latest step (get_paper) stays whole
    assert json.loads(model.contexts[3][5]["content"]) == {"arxiv_id": PAPER["arxiv_id"],
                                                           "title": PAPER["title"]}
    assert t.messages[3]["content"] == hits  # the transcript keeps every output whole
    assert counted == model.contexts  # the token budget is checked on exactly what the model is shown
    assert [context for context, _ in step_contexts(t.messages)] == model.contexts  # training samples match
    assert check.calls[2] == ("write_note", {"arxiv_id": "2609.01234"}, ["2609.01234", "2609.05678"], TODAY)
    assert seen["write_note"] == [{"arxiv_id": "2609.01234"}]


def test_a_sample_of_a_repaired_step_lacks_the_rejected_exchange_the_model_answered():
    bad = turn(call("get_paper", "c1", arxiv_id="2609.1234"))
    good = turn(call("get_paper", "c2", arxiv_id="2609.01234"))
    model = ScriptedModel([bad, good, turn(content="done")])
    ex, _ = rig(check={'get_paper:{"arxiv_id":"2609.1234"}': BAD_ID})
    t = run(model, ex, table_decide({("invalid_args", 1): "repair"}))
    (clean, target), (later, _) = step_contexts(t.messages)
    # the sample pairs the accepted step with the clean context, but the model answered the repair request ...
    assert target == good.to_api() and clean == model.contexts[0] != model.contexts[1]
    assert model.contexts[1][:len(clean)] == clean and len(model.contexts[1]) == len(clean) + 2
    # ... whose extra part, the rejected step and its errors, is kept only in the step record
    assert model.contexts[1][len(clean)]["tool_calls"] == t.steps[0]["attempts"][0]["tool_calls"]
    assert [a["action"] for a in t.steps[0]["attempts"]] == ["repair", None]
    assert "2609.1234" not in canonical_json(t.messages)
    assert later == model.contexts[2]  # a step answered from a plain request matches again


# integration: the real adapters over the real fetcher, recorded and then replayed

CATS = {"categories": ["cs.CL"]}  # parse_atom returns the categories too; no tool shows them
TREE = {**PAPER, **CATS, "arxiv_id": "2609.05678", "title": "Tree Verification at Scale"}
ATOM = {"<feed>search</feed>": [{**PAPER, **CATS}, TREE], "<feed>2609.01234</feed>": [{**PAPER, **CATS}]}
SEARCH_URLS = {"speculative decoding": "https://export.arxiv.org/api/query?search_query=all:speculative"}


def offline_arxiv(request: httpx.Request) -> httpx.Response:
    """The arXiv API, offline: one search feed and one feed per id."""
    arxiv_id = request.url.params.get("id_list")
    return httpx.Response(200, text=f"<feed>{arxiv_id}</feed>" if arxiv_id else "<feed>search</feed>")


def arxiv_executor(fetcher: Fetcher) -> ToolExecutor:
    """The real adapters and executor; the owner's three functions are table stand-ins."""
    tools = arxiv_tools(fetcher, build_search_url=lambda args, today: SEARCH_URLS[args["query"]],
                        parse_atom=lambda xml: copy.deepcopy(ATOM.get(xml, [])),
                        extract_section=lambda html, section, max_chars: None, today=TODAY)
    return ToolExecutor(tools, table_check(), today=TODAY)


def test_integration_a_recorded_turn_replays_to_the_same_transcript(tmp_path):
    script = [turn(call("search_arxiv", "c1", query="speculative decoding")),
              turn(call("get_paper", "c2", arxiv_id="2609.01234")), turn(content="这篇讲自适应草稿长度。")]
    recorder = Fetcher("record", tmp_path, transport=httpx.MockTransport(offline_arxiv), sleep=lambda s: None)
    recorded = run(ScriptedModel(script), arxiv_executor(recorder))
    recorder.close()
    replayed = run(ScriptedModel(script), arxiv_executor(Fetcher("replay", tmp_path)))
    assert recorded.status == ANSWERED and replayed.to_json() == recorded.to_json()
    assert [hit["arxiv_id"] for hit in body(recorded.messages[3])] == ["2609.01234", "2609.05678"]
    # a call that was never recorded is a harness failure, and the model never sees it
    unrecorded = turn(call("get_paper", "c2", arxiv_id="2609.05678"))
    t = run(ScriptedModel([script[0], unrecorded, turn(content="never asked")]),
            arxiv_executor(Fetcher("replay", tmp_path)))
    assert (t.status, t.final_text) == (HARNESS_ERROR, None)
    assert t.steps[1]["harness_error"].startswith("no cassette for https://export.arxiv.org/api/query")
    assert body(t.messages[-1]) == skipped(STOPPED_HINT) and "cassette" not in canonical_json(t.messages)


@pytest.mark.parametrize("ids", [["same", "same"], ["", "c2"]])
def test_empty_or_duplicate_call_ids_from_a_model_adapter_are_a_harness_error(ids):
    calls = [dataclasses.replace(call("get_paper", arxiv_id="2609.01234"), id=ids[0]),
             dataclasses.replace(call("search_arxiv", query="x"), id=ids[1])]
    t = run(ScriptedModel([turn(*calls)]))
    assert t.status == HARNESS_ERROR and "empty or duplicate tool call ids" in t.steps[0]["harness_error"]
    assert len(t.messages) == 2  # nothing from the broken turn entered the history


class _StrayEventExecutor(ToolExecutor):
    """An executor bug: a step issue whose event check_step never reports."""

    def check_step(self, calls):
        return [StepIssue(calls[0].id, "empty_results", ["not a step-check event"])]


def test_a_step_issue_with_an_event_outside_the_step_checks_is_a_harness_error():
    tools, _ = table_tools(SCRIPT)
    executor = _StrayEventExecutor(tools, table_check(), today=TODAY)
    decide = table_decide()
    t = run(ScriptedModel([turn(call("search_arxiv", query="x"))]), executor=executor, decide=decide)
    assert t.status == HARNESS_ERROR and "outside" in t.steps[0]["harness_error"]
    assert decide.asked == []  # the policy is never asked about an event it cannot answer


# ---- M7: the reasoning and the thinking of a local turn in the step record (docs/m7-家族层设计.md §7) ----


def test_an_attempt_records_the_turns_reasoning_and_thinking_only_when_it_has_them():
    thinking = {"mode": "think", "reasoning_tokens": 4, "answer_tokens": 9, "forced": False, "phases": 1,
                "seeds": None}
    search = turn(call("search_arxiv", "c1", query="speculative decoding"))
    search.reasoning, search.thinking = "The user wants papers, so I search.", thinking
    answer = turn(content="找到两篇。")  # a turn without them: its attempt is as it was
    t = run(ScriptedModel([search, answer]))
    assert t.status == ANSWERED
    first, second = (step["attempts"][0] for step in t.steps)
    assert first["reasoning"] == "The user wants papers, so I search." and first["thinking"] == thinking
    assert set(first) == {"model", "content", "tool_calls", "issues", "action", "reasoning", "thinking"}
    assert set(second) == {"model", "content", "tool_calls", "issues", "action"}
    assert "reasoning" not in json.dumps(t.messages)  # the history holds the API form alone
    assert json.loads(t.to_json())["steps"][0]["attempts"][0]["thinking"] == thinking

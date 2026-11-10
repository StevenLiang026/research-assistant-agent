import json

import pytest
from support.fakes import HITS, PAPER, call, table_check, table_tools

from research_agent.agent.executor import ToolExecutor
from research_agent.agent.types import HarnessError, StepIssue, ToolOutcome, ToolResult
from research_agent.tools import schemas

TODAY = "2026-10-04"


def executor(check=None, tools=None) -> ToolExecutor:
    return ToolExecutor(tools or {}, check or table_check(), today=TODAY)


class ReplayMiss(HarnessError):
    """Stands in for the fetcher's replay miss: a HarnessError subclass raised from inside a tool."""


# check_step


def test_three_good_calls_pass_and_each_is_checked_with_seen_ids_and_today():
    check = table_check()
    calls = [call("search_arxiv", "c1", query="speculative decoding"),
             call("get_paper", "c2", arxiv_id="2609.01234"),
             call("ask_clarification", "c3", question="Which month?")]  # validated like any other call
    assert executor(check).check_step(calls) == []
    assert check.calls == [("search_arxiv", {"query": "speculative decoding"}, [], TODAY),
                           ("get_paper", {"arxiv_id": "2609.01234"}, [], TODAY),
                           ("ask_clarification", {"question": "Which month?"}, [], TODAY)]


def test_more_than_three_calls_is_one_issue_for_the_whole_step():
    check = table_check()
    calls = [call("get_paper", f"c{i}", arxiv_id=f"2609.0123{i}") for i in range(4)]
    assert executor(check).check_step(calls) == [
        StepIssue("", "invalid_args", ["4 tool calls in one step; at most 3 are allowed"])]
    assert check.calls == []  # the single calls are not worth checking yet


def test_unknown_tool_and_bad_json_are_refused_without_asking_check_call():
    check = table_check()
    issues = executor(check).check_step([call("search_papers", "c1", query="x"),
                                         call("get_paper", "c2", raw='{"arxiv_id": '),
                                         call("get_paper", "c3", raw='["2609.01234"]')])
    assert [(i.call_id, i.event) for i in issues] == [("c1", "unknown_tool"), ("c2", "bad_json"),
                                                     ("c3", "bad_json")]
    assert "'search_papers'" in issues[0].messages[0] and "search_arxiv, get_paper" in issues[0].messages[0]
    assert issues[1].messages == ["arguments are not valid JSON: Expecting value at character 13"]
    assert issues[2].messages == ["arguments must be a JSON object"]
    assert check.calls == []


def test_check_call_errors_refuse_only_their_call_in_call_order():
    check = table_check({"get_paper": ["arxiv_id: expected YYMM.NNNNN"],
                         'write_note:{"arxiv_id":"2609.01234"}': ["arxiv_id: not seen in this turn"]})
    issues = executor(check).check_step([call("write_note", "c1", arxiv_id="2609.01234"),
                                         call("search_arxiv", "c2", query="x"),
                                         call("get_paper", "c3", arxiv_id="12")])
    assert issues == [StepIssue("c1", "invalid_args", ["arxiv_id: not seen in this turn"]),
                      StepIssue("c3", "invalid_args", ["arxiv_id: expected YYMM.NNNNN"])]
    assert len(check.calls) == 3


def test_issue_events_are_contract_events():
    issues = executor(table_check({"get_paper": ["bad"]})).check_step(
        [call("nope", "c1"), call("get_paper", "c2", raw="{"), call("get_paper", "c3", arxiv_id="1")])
    issues += executor().check_step([call("list_notes", f"c{i}") for i in range(4)])
    assert {i.event for i in issues} == {"unknown_tool", "bad_json", "invalid_args"} <= set(schemas.EVENTS)


def test_check_call_sees_the_sorted_ids_of_earlier_steps():
    tools, _ = table_tools({"search_arxiv": [ToolOutcome(list(reversed(HITS)))]})
    check = table_check()
    ex = executor(check, tools)
    ex.run(call("search_arxiv", "c1", query="speculative decoding"))
    ex.check_step([call("write_note", "c2", arxiv_id="2609.05678")])
    assert check.calls[-1][2] == ["2609.01234", "2609.05678"]


def test_check_call_cannot_change_the_recorded_call_or_the_seen_ids():
    def mutating_check(tool, args, seen_ids, today):
        args["arxiv_id"] = "changed"
        seen_ids.append("2609.99999")
        return []

    c = call("get_paper", "c1", arxiv_id="2609.01234")
    ex = executor(mutating_check)
    assert ex.check_step([c]) == []
    assert c.arguments == {"arxiv_id": "2609.01234"} and ex.seen_ids == []


def test_check_call_failures_are_harness_errors():
    def unfinished(tool, args, seen_ids, today):
        raise NotImplementedError

    with pytest.raises(HarnessError) as info:
        executor(unfinished).check_step([call("get_paper", "c1", arxiv_id="2609.01234")])
    assert isinstance(info.value.__cause__, NotImplementedError)
    with pytest.raises(HarnessError, match="list of strings"):  # a check_call that forgot to return
        executor(lambda tool, args, seen_ids, today: None).check_step([call("get_paper", "c1", arxiv_id="1")])


# run


def test_run_wraps_the_outcome_and_passes_the_arguments():
    tools, seen = table_tools({"search_arxiv": [ToolOutcome(HITS)],
                               "get_paper": [ToolOutcome({}, schemas.NOT_FOUND)]})
    ex = executor(tools=tools)
    assert ex.run(call("search_arxiv", "c1", query="x")) == ToolResult("c1", "search_arxiv", HITS)
    assert ex.run(call("get_paper", "c2", arxiv_id="2609.99999")) == ToolResult("c2", "get_paper", {},
                                                                                 "NOT_FOUND")
    assert seen == {"search_arxiv": [{"query": "x"}], "get_paper": [{"arxiv_id": "2609.99999"}]}
    assert ex.seen_ids == ["2609.01234", "2609.05678"]  # the NOT_FOUND result named no paper


def test_ids_are_collected_at_any_depth_sorted_and_without_duplicates():
    nested = {"arxiv_id": "2609.05678",
              "related": [{"arxiv_id": "2501.00001"}, {"deep": {"arxiv_id": "2609.01234"}}],
              "note": {"arxiv_id": None}, "empty": {"arxiv_id": ""}, "cited": ["2612.00002"], "count": 3}
    tools, _ = table_tools({"list_notes": [ToolOutcome(nested)], "search_arxiv": [ToolOutcome(HITS)]})
    ex = executor(tools=tools)
    ex.run(call("list_notes", "c1"))
    ex.run(call("search_arxiv", "c2", query="x"))
    assert ex.seen_ids == ["2501.00001", "2609.01234", "2609.05678"]  # only strings under an "arxiv_id" key


def test_ids_are_collected_from_the_content_as_the_model_reads_it():
    # a tuple, e.g. from the notes store, and a number key: the model reads them as a list and a string
    rows = ({"arxiv_id": "2609.01234", "title": "Adaptive Draft Length"},)
    ex = executor(tools={"search_notes": lambda args: ToolOutcome({"notes": rows, "per_year": {2026: 1}})})
    result = ex.run(call("search_notes", "c1", query="推测解码"))
    read = {"notes": [{"arxiv_id": "2609.01234", "title": "Adaptive Draft Length"}], "per_year": {"2026": 1}}
    assert result.content == json.loads(result.to_api()["content"]) == read
    assert ex.seen_ids == ["2609.01234"]


def test_a_tool_cannot_change_a_result_it_returned():
    rows = [{"arxiv_id": "2609.01234", "title": "Adaptive Draft Length"}]  # e.g. a list the notes store keeps
    ex = executor(tools={"list_notes": lambda args: ToolOutcome(rows)})
    result = ex.run(call("list_notes", "c1"))
    rows[0]["title"] = "changed"
    rows.append({"arxiv_id": "2609.05678"})
    assert result.content == [{"arxiv_id": "2609.01234", "title": "Adaptive Draft Length"}]


def test_ids_in_a_result_with_an_error_code_count_too():
    # NO_FULLTEXT still names a real paper: its abstract came back in place of the section
    fallback = {"arxiv_id": "2609.01234", "section": "method", "source": "abstract", "text": "Drafts tokens."}
    tools, _ = table_tools({"fetch_fulltext": [ToolOutcome(fallback, schemas.NO_FULLTEXT)]})
    ex = executor(tools=tools)
    assert ex.run(call("fetch_fulltext", "c1", arxiv_id="2609.01234")).error == schemas.NO_FULLTEXT
    assert ex.seen_ids == ["2609.01234"]


def test_seen_ids_is_a_copy_and_reset_starts_a_new_turn():
    tools, _ = table_tools({"get_paper": [ToolOutcome(PAPER)]})
    ex = executor(tools=tools)
    ex.run(call("get_paper", "c1", arxiv_id="2609.01234"))
    ex.seen_ids.append("2609.99999")
    assert ex.seen_ids == ["2609.01234"]
    ex.reset()
    assert ex.seen_ids == []


def test_a_tool_cannot_change_the_recorded_call():
    def mutating_tool(args):
        args["arxiv_id"] = "changed"
        return ToolOutcome(PAPER)

    c = call("get_paper", "c1", arxiv_id="2609.01234")
    executor(tools={"get_paper": mutating_tool}).run(c)
    assert c.arguments == {"arxiv_id": "2609.01234"}


def test_a_tool_without_an_adapter_is_a_harness_error():
    with pytest.raises(HarnessError, match="ask_clarification"):  # the loop answers it; no adapter runs it
        executor().run(call("ask_clarification", "c1", question="Which month?"))


def test_tool_exceptions_become_harness_errors_with_the_cause():
    def broken(args):
        raise KeyError("abstract")

    ex = executor(tools={"get_paper": broken})
    with pytest.raises(HarnessError, match="get_paper") as info:
        ex.run(call("get_paper", "c1", arxiv_id="2609.01234"))
    assert isinstance(info.value.__cause__, KeyError)


def test_harness_errors_from_a_tool_propagate_untouched():
    miss = ReplayMiss("no recording for https://export.arxiv.org/api/query?id_list=2609.01234")

    def replaying(args):
        raise miss

    with pytest.raises(ReplayMiss) as info:
        executor(tools={"get_paper": replaying}).run(call("get_paper", "c1", arxiv_id="2609.01234"))
    assert info.value is miss and info.value.__cause__ is None


@pytest.mark.parametrize("returned", [
    {"arxiv_id": "2609.01234"},  # not a ToolOutcome
    ToolOutcome("2609.01234"),  # content neither a dict nor a list
    ToolOutcome([], "EMPTY"),  # not one of the contract's error codes
    ToolOutcome({"arxiv_id": "2609.01234", "tags": {"llm", "decoding"}}),  # a set has no JSON form
])
def test_malformed_outcomes_are_harness_errors_and_record_nothing(returned):
    ex = executor(tools={"get_paper": lambda args: returned})
    with pytest.raises(HarnessError, match="get_paper"):
        ex.run(call("get_paper", "c1", arxiv_id="2609.01234"))
    assert ex.seen_ids == []


def test_run_refuses_unparsed_arguments():
    tools, seen = table_tools({"get_paper": [ToolOutcome(PAPER)]})
    with pytest.raises(HarnessError, match="unparsed"):
        executor(tools=tools).run(call("get_paper", "c1", raw='{"arxiv_id": '))
    assert seen["get_paper"] == []

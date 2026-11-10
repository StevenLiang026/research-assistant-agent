"""The trajectory validator (research_agent.trajectories), offline, on small hand-made transcripts built the
way the loop builds them and read back through Transcript.to_json()."""

from __future__ import annotations

import copy
import json

import pytest
from support.fakes import HITS, PAPER, call

from research_agent.agent.types import ANSWERED, CLARIFICATION, STEP_LIMIT, ToolCall, ToolResult, Transcript
from research_agent.data import synth
from research_agent.tools import schemas
from research_agent.trajectories import check_trajectory

TODAY = "2026-10-10"
ID1, ID2 = HITS[0]["arxiv_id"], HITS[1]["arxiv_id"]
NOTE_RECORD = {"arxiv_id": ID1, "note_id": f"{ID1}-note", "path": f"notes/{ID1}.md"}


def assistant(*calls: ToolCall, content: str = "") -> dict:
    message: dict = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [c.to_api() for c in calls]
    return message


def tool(c: ToolCall, content: dict | list, error: str | None = None) -> dict:
    return ToolResult(c.id, c.name, content, error).to_api()


def transcript(*exchange: dict, status: str = ANSWERED, final_text: str | None = None) -> dict:
    """A transcript as the generator hands it to the validator: Transcript.to_json() read back. A turn that
    ends answered gets a closing assistant message with final_text."""
    messages = [{"role": "system", "content": "system prompt"}, {"role": "user", "content": "request"},
                *exchange]
    if status == ANSWERED and final_text is not None:
        messages.append(assistant(content=final_text))
    t = Transcript(request="request", today=TODAY, status=status, final_text=final_text, messages=messages,
                   steps=[], schema_hash=schemas.SCHEMA_HASH, prompt_version="p", prompt_hash="h")
    return json.loads(t.to_json())


def request(scenario: str = "normal", **slots) -> dict:
    slots = slots or {"topic": "speculative decoding", "date_from": "2026-09-01"}
    return {"id": "r1", "intent": "find_recent_papers", "scenario": scenario, "lang": "en", "split": "dev",
            "family": "test", "text": "Find papers on speculative decoding since 2026-09-01.", "slots": slots}


SEARCH = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-01")
GET = call("get_paper", "c2", arxiv_id=ID1)
NOTE = call("write_note", "c3", arxiv_id=ID1, mode="scaffold", evidence="Speculative decoding drafts tokens.")
ANSWER = f"Wrote a note on {ID1}."


def normal(search: ToolCall = SEARCH, hits: list = HITS, note: ToolCall = NOTE, answer: str = ANSWER,
           search_error: str | None = None) -> dict:
    """search, get_paper, write_note, answer: an accepted normal trajectory."""
    return transcript(assistant(search), tool(search, hits if search_error is None else {"items": hits},
                                              search_error),
                      assistant(GET), tool(GET, PAPER),
                      assistant(note), tool(note, NOTE_RECORD),
                      final_text=answer)


# ---- accepted -------------------------------------------------------------------------------------------


def test_a_normal_trajectory_is_accepted():
    assert check_trajectory(normal(), request()) == []


def test_a_no_call_answer_and_a_clarification_are_accepted():
    assert check_trajectory(transcript(final_text="Hello."), request("no_call", x=1)) == []
    ask = call("ask_clarification", "c1", question="Which topic?", missing=["topic"])
    clarified = transcript(assistant(ask), tool(ask, {"question": "Which topic?", "status": "asked"}),
                           status=CLARIFICATION, final_text="Which topic?")
    assert check_trajectory(clarified, request("clarify", x=1)) == []


# ---- 1. calls -------------------------------------------------------------------------------------------


def test_a_call_whose_arguments_do_not_parse_is_a_calls_problem():
    broken = call("search_arxiv", "c1", raw='{"query": "speculative')
    t = transcript(assistant(broken), tool(broken, HITS), final_text="none")
    assert check_trajectory(t, request(topic="x")) == [
        "calls step 1: search_arxiv call c1: arguments are not a JSON object"]  # no parsed search: no topic


def test_an_invalid_call_is_a_calls_problem_with_check_calls_message():
    bad = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-01", max_results=9)
    assert check_trajectory(normal(search=bad), request()) == [
        "calls step 1: search_arxiv call c1: max_results: 9 is greater than the maximum of 5"]


def test_an_unknown_tool_is_a_calls_problem():
    odd = call("browse", "c1", url="x")
    t = transcript(assistant(odd), tool(odd, {"ok": True}), final_text="done")
    assert check_trajectory(t, request(x=1)) == ["calls step 1: browse call c1: tool: unknown tool browse"]


def test_a_write_note_on_an_id_no_tool_output_showed_before_its_step_is_a_calls_problem():
    note = call("write_note", "c3", arxiv_id="2609.09999", mode="scaffold", evidence="e")
    t = transcript(assistant(SEARCH), tool(SEARCH, HITS), assistant(note),
                   tool(note, {**NOTE_RECORD, "arxiv_id": "2609.09999"}), final_text="done")
    assert check_trajectory(t, request()) == [
        "calls step 2: write_note call c3: arxiv_id: 2609.09999 did not appear in any tool output this turn"]


def test_seen_ids_count_only_tool_outputs_before_the_step():
    get = call("get_paper", "c2", arxiv_id=ID1)
    note = call("write_note", "c3", arxiv_id=ID1, mode="scaffold", evidence="e")
    t = transcript(assistant(get, note), tool(get, PAPER), tool(note, NOTE_RECORD), final_text="done")
    assert check_trajectory(t, request(x=1)) == [
        f"calls step 1: write_note call c3: arxiv_id: {ID1} did not appear in any tool output this turn"]


# ---- 2. execution ---------------------------------------------------------------------------------------


def test_a_status_other_than_answered_or_clarification_is_an_execution_problem():
    t = transcript(assistant(SEARCH), tool(SEARCH, HITS), status=STEP_LIMIT)
    assert check_trajectory(t, request()) == ["execution: status step_limit is not answered or clarification"]


@pytest.mark.parametrize("code", [schemas.RATE_LIMITED, schemas.NOT_FOUND, schemas.SKIPPED,
                                  schemas.EMPTY_RESULTS])
def test_any_error_code_in_a_normal_trajectory_is_an_execution_problem(code):
    t = normal(search_error=code)
    assert check_trajectory(t, request()) == [f"execution step 1: search_arxiv call c1 returned {code}"]


def test_no_fulltext_is_the_designed_fallback_and_not_a_problem():
    assert check_trajectory(normal(search_error=schemas.NO_FULLTEXT), request()) == []


def test_a_request_that_names_its_ids_needs_no_search_so_the_topic_binds_nothing():
    get = call("get_paper", "c1", arxiv_id=ID1)
    paper = {"arxiv_id": ID1, "title": "Reward models", "authors": [], "published": "2026-09-05",
             "abstract": "On reward models."}
    t = transcript(assistant(get), tool(get, paper), final_text=f"{ID1} is about reward models.")
    assert check_trajectory(t, request(topic="reward model")) == []


@pytest.mark.parametrize("code", synth.RECOVER_CODES)
def test_a_recover_trajectory_with_an_injected_failure_is_accepted(code):
    failed = call("search_arxiv", "c0", query="speculative decoding", date_from="2026-09-01")
    t = transcript(assistant(failed), tool(failed, {"items": []}, code),
                   assistant(SEARCH), tool(SEARCH, HITS), assistant(GET), tool(GET, PAPER),
                   assistant(NOTE), tool(NOTE, NOTE_RECORD), final_text=ANSWER)
    assert check_trajectory(t, request("recover")) == []


def test_a_recover_trajectory_without_an_injected_failure_is_rejected():
    assert check_trajectory(normal(), request("recover")) == [
        ("execution: recover scenario but no tool output carries EMPTY_RESULTS, RATE_LIMITED or "
         "UPSTREAM_ERROR")]


def test_a_recover_trajectory_still_fails_on_codes_that_are_not_injected():
    failed = call("search_arxiv", "c0", query="speculative decoding", date_from="2026-09-01")
    fetch = call("fetch_fulltext", "c4", arxiv_id=ID1)
    t = transcript(assistant(failed), tool(failed, {"items": []}, schemas.RATE_LIMITED),
                   assistant(SEARCH), tool(SEARCH, HITS),
                   assistant(fetch), tool(fetch, {"arxiv_id": ID1, "text": "abstract"}, schemas.NO_FULLTEXT),
                   final_text=ANSWER)
    assert check_trajectory(t, request("recover")) == []  # NO_FULLTEXT is the abstract fall-back, no failure


def test_a_tool_message_whose_content_is_not_json_is_an_execution_problem():
    t = normal()
    t["messages"][5]["content"] = "not json {"  # the get_paper output
    assert check_trajectory(t, request()) == [
        "execution step 2: get_paper call c2 has content that is not JSON"]


# ---- 3. ids ---------------------------------------------------------------------------------------------


def test_an_id_in_the_final_text_that_no_tool_returned_is_an_ids_problem():
    t = normal(answer=f"Wrote a note on {ID1}; see also arXiv:2609.07777v2 and 2026.1001 (not an id).")
    assert check_trajectory(t, request()) == [
        "ids: final text cites 2609.07777, which did not appear in any tool output this turn"]


def test_an_id_the_user_typed_counts_only_once_a_tool_returned_it():
    req = {**request(x=1), "text": "Summarise 2609.07777."}
    unseen = transcript(final_text="2609.07777 is about decoding.")
    assert check_trajectory(unseen, req) == [
        "ids: final text cites 2609.07777, which did not appear in any tool output this turn"]
    get = call("get_paper", "c1", arxiv_id="2609.07777")
    seen = transcript(assistant(get), tool(get, {**PAPER, "arxiv_id": "2609.07777"}),
                      final_text="２６０９.０７７７７ is about decoding.")  # full-width digits normalise
    assert check_trajectory(seen, req) == []


def test_an_answered_turn_without_final_text_is_an_ids_problem():
    t = transcript(assistant(content="x"), final_text=None)
    assert check_trajectory(t, request(x=1)) == ["ids: final_text is missing on an answered turn"]


# ---- 4. constraints -------------------------------------------------------------------------------------


def test_a_search_with_another_date_from_than_the_slots_is_a_constraints_problem():
    late = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-15")
    assert check_trajectory(normal(search=late), request()) == [
        "constraints step 1: search_arxiv call c1 has date_from 2026-09-15, the request asks for 2026-09-01"]


def test_a_search_without_the_slots_date_from_is_a_constraints_problem():
    bare = call("search_arxiv", "c1", query="speculative decoding")
    assert check_trajectory(normal(search=bare), request()) == [
        "constraints step 1: search_arxiv call c1 has date_from none, the request asks for 2026-09-01"]


def test_a_hit_outside_the_date_window_is_a_constraints_problem():
    old = [*HITS, {**HITS[0], "arxiv_id": "2608.04444", "published": "2026-08-20"}]
    assert check_trajectory(normal(hits=old), request()) == [
        ("constraints step 1: search_arxiv call c1 returned 2608.04444 published 2026-08-20, "
         "before date_from 2026-09-01")]
    window = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-01",
                  date_to="2026-09-16")
    slots = {"topic": "speculative decoding", "date_from": "2026-09-01", "date_to": "2026-09-16"}
    assert check_trajectory(normal(search=window), request(**slots)) == [
        (f"constraints step 1: search_arxiv call c1 returned {ID2} published 2026-09-20, after date_to "
         "2026-09-16")]


def test_a_hit_without_a_published_date_is_a_constraints_problem():
    undated = [*HITS, {key: value for key, value in HITS[0].items() if key != "published"}]
    assert check_trajectory(normal(hits=undated), request()) == [
        "constraints step 1: search_arxiv call c1 returned a hit without a YYYY-MM-DD published date"]


def test_categories_compare_as_sets_and_a_missing_topic_is_a_constraints_problem():
    slots = {"topic": "speculative decoding", "categories": ["cs.LG", "cs.CL"]}
    same = call("search_arxiv", "c1", query="speculative decoding", categories=["cs.CL", "cs.LG"])
    assert check_trajectory(normal(search=same), request(**slots)) == []
    other = call("search_arxiv", "c1", query="draft models", categories=["cs.CL"])
    assert check_trajectory(normal(search=other), request(**slots)) == [
        ('constraints step 1: search_arxiv call c1 has categories ["cs.CL"], the request asks for '
         '["cs.CL", "cs.LG"]'),
        "constraints: no search_arxiv query contains the topic speculative decoding"]


# ---- 5. fields ------------------------------------------------------------------------------------------


def test_a_scaffold_note_without_evidence_is_a_fields_problem():
    bare = call("write_note", "c3", arxiv_id=ID1)  # mode defaults to scaffold
    assert check_trajectory(normal(note=bare), request()) == [
        "fields step 3: write_note call c3 in scaffold mode needs a non-empty evidence"]


def test_a_draft_note_needs_the_four_parts():
    draft = call("write_note", "c3", arxiv_id=ID1, mode="draft", problem="p", method="m", evidence=" ")
    assert check_trajectory(normal(note=draft), request()) == [
        "fields step 3: write_note call c3 in draft mode needs a non-empty evidence",
        "fields step 3: write_note call c3 in draft mode needs a non-empty limitations"]


def test_a_write_note_mode_that_is_not_a_string_is_left_to_the_calls_check():
    odd = call("write_note", "c3", arxiv_id=ID1, mode=["x"], evidence="e")
    assert check_trajectory(normal(note=odd), request()) == [  # no "check_trajectory: the checks failed" line
        "calls step 3: write_note call c3: mode: ['x'] is not of type 'string'",
        "calls step 3: write_note call c3: mode: ['x'] is not one of ['scaffold', 'draft']"]


def test_a_no_call_request_with_a_call_is_a_fields_problem():
    assert check_trajectory(normal(), request("no_call")) == [
        "fields: no_call scenario but the turn made 3 tool call(s)"]


def test_a_clarify_request_without_ask_clarification_is_a_fields_problem():
    assert check_trajectory(normal(), request("clarify")) == [
        "fields: clarify scenario but no ask_clarification call",
        "fields: clarify scenario but the turn ended as answered"]


# ---- robustness -----------------------------------------------------------------------------------------


def test_missing_keys_are_reported_not_raised():
    t = normal()
    del t["messages"], t["today"]
    req = request()
    del req["slots"]
    assert check_trajectory(t, req) == ["request: missing key slots", "transcript: missing key messages",
                                        "transcript: missing key today"]
    assert check_trajectory({}, {}) == [
        "request: missing key scenario", "request: missing key slots", "transcript: missing key final_text",
        "transcript: missing key messages", "transcript: missing key status", "transcript: missing key today"]
    t = {**normal(), "messages": "not a list", "final_text": 3}
    assert check_trajectory(t, request()) == ["transcript: final_text is not a string or null",
                                              "transcript: messages is not a list"]
    assert check_trajectory(None, request()) == ["transcript: is not a dict"]  # type: ignore[arg-type]
    assert check_trajectory(normal(), []) == ["request: is not a dict"]  # type: ignore[arg-type]


def test_malformed_messages_are_reported_not_raised():
    t = normal()
    t["messages"][2]["tool_calls"][0] = {"function": "x"}
    assert check_trajectory(t, request()) == [
        "transcript: message 2 is malformed: a tool call without an id and a function name",
        "transcript: message 3 is malformed: a tool message that answers no pending call"]
    t = normal()
    del t["messages"][3]
    assert check_trajectory(t, request()) == [
        "transcript: search_arxiv call c1 of step 1 has no tool message"]


def test_the_function_is_pure():
    t, req = normal(), request()
    before = (copy.deepcopy(t), copy.deepcopy(req))
    first = check_trajectory(t, req)
    assert (t, req) == before
    assert check_trajectory(t, req) == first

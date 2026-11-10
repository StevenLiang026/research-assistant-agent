"""Voting (research_agent.agent.vote; docs/m7-本地提升计划.md WP2): the answer keys of the reproduction mode
(the 10-29 analysis' strict key, voting_analysis.py answer_key) and of the final mode (S1, S2 and the ids flag
on top), rule A, the voted record offline, the failed candidates a vote goes on without and lists under its
errors (one function, online and offline alike), and VotedModel, whose online answer is the offline vote of
its candidates' records by construction."""

from __future__ import annotations

import json

import pytest
from support.fakes import ScriptedModel, table_check, table_decide, table_tools, turn
from support.llama_fakes import FakeLlama, Scripted, fake_loaded_family

from research_agent.agent.context import build_context
from research_agent.agent.executor import ToolExecutor
from research_agent.agent.loop import asking, run_turn
from research_agent.agent.models import LocalModel
from research_agent.agent.prompts import system_prompt
from research_agent.agent.types import HarnessError, ToolCall
from research_agent.agent.vote import (
    FINAL,
    FREE,
    FREE_BAD,
    REPRODUCTION,
    TEXT,
    VotedModel,
    answer_key,
    candidate_error,
    decide,
    plurality,
    summed_usage,
    voted_record,
    voted_settings,
)
from research_agent.canonical import canonical_json
from research_agent.eval.runner import error_fields, run_items
from research_agent.family.core import GREEDY, Decoding, Mode
from research_agent.llm.generation import GenerationSettings, request_seed
from research_agent.tools.schemas import MODEL_TOOLS

TODAY = "2026-10-10"
SYSTEM = {"role": "system", "content": system_prompt(TODAY)}
USER = {"role": "user", "content": "Find mixture-of-experts papers."}
ID1, ID2 = "2609.00001", "2609.00002"
HITS = [{"arxiv_id": ID1, "title": "MoE routing", "snippet": "Extends 2608.05678 to routing."},
        {"arxiv_id": ID2, "title": "Tree Verification at Scale", "snippet": "Tree verification."}]


def api_call(call_id: str, name: str, arguments: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def answer(*calls: dict, content: str = "") -> dict:
    message: dict = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = list(calls)
    return message


FIRST_SEARCH = api_call("c0", "search_arxiv", '{"query":"mixture of experts"}')
HISTORY = [SYSTEM, USER, answer(FIRST_SEARCH),
           {"role": "tool", "tool_call_id": "c0", "content": canonical_json(HITS)}]
USAGE = {"prompt_tokens": 100, "completion_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0},
         "total_tokens": 110}


def ok(*calls: dict, content: str = "", usage: dict | None = None) -> dict:
    """A runner record of an ok turn (eval.runner)."""
    return {"id": "q-1", "status": "ok", "item": {"type": "call"}, "output": answer(*calls, content=content),
            "usage": dict(usage or USAGE), "paid": False, "cached": False, "model": "m:lora=1"}


def failed(message: str = "parse_tool_calls failed", error_type: str = "HarnessError") -> dict:
    """A runner record of a model that raised (eval.runner)."""
    return {"id": "q-1", "status": "error", "item": {"type": "call"}, "error": message,
            "error_type": error_type}


def key_of(*calls: tuple[str, dict]) -> str:
    """The reproduction key of calls given with their arguments as the key normalises them."""
    return canonical_json(sorted([name, canonical_json(args)] for name, args in calls))


# ---- the reproduction key: voting_analysis.py answer_key(record, "strict") ---------------------------------


def test_the_reproduction_key_normalises_the_arguments_as_the_scorer_and_casefolds_the_query():
    search = api_call("local_0", "search_arxiv",
                      '{"query":"Mixture-of-Experts  Routing","date_to":"2026/10/10","max_results":5.0}')
    assert answer_key(ok(search), REPRODUCTION, HISTORY) == key_of(
        ("search_arxiv", {"date_to": TODAY, "max_results": 5, "query": "mixture-of-experts routing",
                          "sort_by": "relevance"}))


def test_the_reproduction_key_replaces_free_text_and_marks_what_the_schema_refuses():
    note = api_call("local_0", "write_note", '{"arxiv_id":"arXiv:2609.00001v2","evidence":"a quote",'
                                             '"new_terms":"not a list","extra":1}')
    ask = api_call("local_1", "ask_clarification", '{"question":"Which paper?","missing":["paper_id"],'
                                                   '"foo":"bar"}')
    assert answer_key(ok(note, ask), REPRODUCTION, HISTORY) == key_of(
        ("write_note", {"arxiv_id": ID1, "evidence": FREE, "extra": 1, "mode": "scaffold",
                        "new_terms": FREE_BAD}),
        ("ask_clarification", {"foo": FREE, "missing": ["set", ['"paper_id"']], "question": FREE}))


def test_the_reproduction_key_of_text_bad_json_and_a_failed_record():
    assert answer_key(ok(content="Here are two papers."), REPRODUCTION, HISTORY) == TEXT == "TEXT"
    broken = api_call("local_0", "get_paper", '{"arxiv_id": \n  "2609')
    assert answer_key(ok(broken), REPRODUCTION, HISTORY) == canonical_json([["get_paper",
                                                                           '!json:{"arxiv_id": "2609']])
    assert answer_key(failed(), REPRODUCTION, HISTORY) == "STATUS:error"


# ---- the final key: the reproduction key with S1, S2 and the ids flag --------------------------------------


def final_of(*calls: tuple[str, dict], ids_in_history: bool = True) -> str:
    return canonical_json({"calls": sorted([name, canonical_json(args)] for name, args in calls),
                           "ids_in_history": ids_in_history})


def test_the_final_key_drops_todays_date_to_and_reads_hyphens_as_spaces():
    search = api_call("local_0", "search_arxiv",
                      '{"query":"Mixture-of-Experts Routing","date_to":"2026-10-10"}')
    assert answer_key(ok(search), FINAL, HISTORY) == final_of(
        ("search_arxiv", {"max_results": 5, "query": "mixture of experts routing", "sort_by": "relevance"}))
    other_day = api_call("local_0", "search_arxiv", '{"query":"moe","date_to":"2026/10/09"}')
    assert answer_key(ok(other_day), FINAL, HISTORY) == final_of(
        ("search_arxiv", {"date_to": "2026-10-09", "max_results": 5, "query": "moe", "sort_by": "relevance"}))
    # without a date in the system message S1 does not apply
    undated = [{"role": "system", "content": "no date here"}, *HISTORY[1:]]
    assert '"date_to' in answer_key(ok(search), FINAL, undated)


def test_the_final_key_flags_an_answer_whose_call_arguments_hold_an_id_the_history_never_showed():
    seen = api_call("local_0", "get_paper", '{"arxiv_id":"2609.00001"}')
    in_text = api_call("local_0", "get_paper", '{"arxiv_id":"2608.05678"}')  # only in a snippet: shown too
    unseen = api_call("local_0", "get_paper", '{"arxiv_id":"2509.00001"}')
    assert answer_key(ok(seen), FINAL, HISTORY) == final_of(("get_paper", {"arxiv_id": ID1}))
    assert json.loads(answer_key(ok(in_text), FINAL, HISTORY))["ids_in_history"] is True
    assert answer_key(ok(unseen), FINAL, HISTORY) == final_of(("get_paper", {"arxiv_id": "2509.00001"}),
                                                              ids_in_history=False)
    # free text the key hides still carries its ids into the flag
    note = api_call("local_0", "write_note", '{"arxiv_id":"2609.00001","evidence":"as 2509.00009 shows"}')
    assert json.loads(answer_key(ok(note), FINAL, HISTORY))["ids_in_history"] is False
    assert answer_key(ok(content="See 2509.00009."), FINAL, HISTORY) == TEXT


def test_a_failed_record_and_one_whose_arguments_do_not_parse_have_no_final_key():
    broken = api_call("local_0", "get_paper", '{"arxiv_id": ')
    fine = api_call("local_1", "get_paper", '{"arxiv_id":"2609.00001"}')
    assert answer_key(failed(), FINAL, HISTORY) is None
    assert answer_key(ok(fine, broken), FINAL, HISTORY) is None


def test_an_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="mode"):
        answer_key(ok(), "coarse", HISTORY)


# ---- rule A ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("keys", "greedy", "chosen", "representative"), [
    (["a", "b", "b", "c", "b"], "a", "b", 1),  # the plurality, its member with the smallest index
    (["a", "a", "b", "b", "c"], "b", "b", 2),  # a tie goes to the class the greedy answer is in
    (["a", "b", "b", "a", "c"], "c", "a", 0),  # else to the tied class with the smallest sample index
    (["b", "a", "a", "b", None], "x", "b", 0),
    (["b", "a", "a", "b", None], None, "b", 0),
    ([None, "a", None, None, None], None, "a", 1),  # a sample without a key does not vote
    ([None] * 5, "a", None, None),  # none voted: the greedy answer stands
])
def test_rule_a(keys, greedy, chosen, representative):
    assert plurality(keys, greedy) == (chosen, representative)


def test_in_reproduction_mode_failed_samples_are_a_class_that_votes_and_can_win():
    search = api_call("local_0", "search_arxiv", '{"query":"moe"}')
    get = api_call("local_0", "get_paper", '{"arxiv_id":"2609.00001"}')
    samples = [failed(), failed("timeout", "TimeoutError"), ok(search), failed(), ok(get)]
    votes = decide(ok(search), samples, HISTORY, REPRODUCTION)
    s_key, g_key = (answer_key(ok(c), REPRODUCTION, HISTORY) for c in (search, get))
    errors = [{"candidate": 0, "error_type": "HarnessError", "message": "parse_tool_calls failed"},
              {"candidate": 1, "error_type": "TimeoutError", "message": "timeout"},
              {"candidate": 3, "error_type": "HarnessError", "message": "parse_tool_calls failed"}]
    assert votes == {"k": 5, "mode": REPRODUCTION, "greedy": s_key,
                     "samples": ["STATUS:error", "STATUS:error", s_key, "STATUS:error", g_key],
                     "counts": {"STATUS:error": 3, s_key: 1, g_key: 1}, "chosen": "STATUS:error",
                     "representative": 0, "errors": errors}
    # the final mode: the failed samples do not vote; the tie of one against one goes to the greedy's class
    final = decide(ok(search), samples, HISTORY, FINAL)
    fs_key, fg_key = (answer_key(ok(c), FINAL, HISTORY) for c in (search, get))
    assert final == {"k": 5, "mode": FINAL, "greedy": fs_key, "samples": [None, None, fs_key, None, fg_key],
                     "counts": {fs_key: 1, fg_key: 1}, "chosen": fs_key, "representative": 2,
                     "errors": errors}
    none = decide(ok(search), [failed()] * 3, HISTORY, FINAL)
    assert (none["chosen"], none["representative"], none["counts"]) == (None, "greedy", {})
    assert [e["candidate"] for e in none["errors"]] == [0, 1, 2]


# ---- the failed candidates: the vote goes on without them and lists them under its errors ----------------


@pytest.mark.parametrize("mode", [FINAL, REPRODUCTION])
def test_the_vote_goes_on_without_the_failed_candidates_and_lists_each_under_errors(mode):
    search = api_call("local_0", "search_arxiv", '{"query":"moe"}')
    samples = [ok(search), failed("read timed out", "ReadTimeout"), ok(search), ok(search),
               failed("parse_tool_calls failed: ValueError('x')")]
    votes = decide(failed("bad call"), samples, HISTORY, mode)
    # the greedy candidate first, then the samples in order, by the sample index the vote's keys use
    assert votes["errors"] == [
        {"candidate": "greedy", "error_type": "HarnessError", "message": "bad call"},
        {"candidate": 1, "error_type": "ReadTimeout", "message": "read timed out"},
        {"candidate": 4, "error_type": "HarnessError", "message": "parse_tool_calls failed: ValueError('x')"}]
    assert (votes["chosen"], votes["representative"]) == (answer_key(ok(search), mode, HISTORY), 0)
    assert voted_record(failed("bad call"), samples, HISTORY, mode)["votes"] == votes
    # no candidate failed: no errors at all, so a vote of answers that all came back is as it was
    assert "errors" not in decide(ok(search), [ok(search), ok(content="x")], HISTORY, mode)


def test_a_record_that_is_not_ok_without_an_error_is_listed_by_its_status():
    # offline only: online every failed candidate raised, and the runner never asks a model whose context
    # does not fit
    search = api_call("local_0", "search_arxiv", '{"query":"moe"}')
    overflow = {"id": "q-1", "status": "context_overflow", "item": {"type": "call"}}
    votes = decide(ok(search), [overflow, ok(search)], HISTORY, FINAL)
    assert votes["errors"] == [{"candidate": 0, "error_type": "context_overflow", "message": None}]
    assert candidate_error("greedy", failed("boom", "KeyError")) == {"candidate": "greedy",
                                                                      "error_type": "KeyError",
                                                                      "message": "boom"}


# ---- the voted record, offline -----------------------------------------------------------------------------


def test_the_voted_record_is_the_representatives_with_every_requests_usage_and_the_vote():
    search = api_call("local_0", "search_arxiv", '{"query":"moe"}')
    other = api_call("local_0", "search_arxiv", '{"query":"mixture-of-experts"}')
    greedy = ok(search, usage={"prompt_tokens": 10, "completion_tokens": 1})
    samples = [ok(other, usage={"prompt_tokens": 20, "completion_tokens": 2,
                                "prompt_tokens_details": {"cached_tokens": 3}}),
               ok(search, usage={"prompt_tokens": 30, "completion_tokens": 3}),
               ok(api_call("x", "search_arxiv", '{"query":"Mixture of Experts"}'),
                  usage={"prompt_tokens": 40, "completion_tokens": 4})]
    # final mode: "mixture-of-experts" and "Mixture of Experts" are one class (S2), which wins 2 to 1
    record = voted_record(greedy, samples, HISTORY, FINAL)
    assert record == {**samples[0], "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                                              "prompt_tokens_details": {"cached_tokens": 3}},
                      "votes": decide(greedy, samples, HISTORY, FINAL)}
    votes = record["votes"]
    assert votes["representative"] == 0 and votes["counts"][votes["chosen"]] == 2
    # reproduction mode: three classes of one; the tie goes to the greedy answer's class, sample 1
    assert voted_record(greedy, samples, HISTORY, REPRODUCTION)["output"] == samples[1]["output"]


def test_a_voted_record_whose_representative_failed_is_that_record_with_the_vote():
    samples = [failed("a"), failed("b"), ok(content="x")]
    record = voted_record(ok(content="y"), samples, HISTORY, REPRODUCTION)
    assert record == {**samples[0], "votes": decide(ok(content="y"), samples, HISTORY, REPRODUCTION)}
    assert voted_record(failed("g"), [failed("a")], HISTORY, FINAL) == {
        **failed("g"), "votes": decide(failed("g"), [failed("a")], HISTORY, FINAL)}


def test_usage_sums_the_counts_and_keeps_the_rest_of_the_first():
    assert summed_usage([{"a": 1, "d": {"x": 1}, "s": "first"}, {"a": 2, "b": 5, "d": {"x": 2, "y": 1}},
                         {"s": "second", "f": 1.5}]) == {"a": 3, "d": {"x": 3, "y": 1}, "s": "first", "b": 5,
                                                         "f": 1.5}
    assert summed_usage([]) == {}


# ---- VotedModel --------------------------------------------------------------------------------------------


def local(text: str = "", *calls: dict, reasoning: str = "R", usage: dict | None = None):
    t = turn(*(ToolCall.from_api(c) for c in calls), content=text, model="m:lora=1", paid=False,
             usage=usage or {"prompt_tokens": 100, "completion_tokens": 10})
    t.signals = {"min_decision_prob": 0.5, "mean_logprob": -0.3, "hard_reason": None}
    t.reasoning, t.thinking = reasoning, {"mode": "think", "forced": False, "seeds": None}
    return t


ITEM = {"id": "q-1", "messages": HISTORY, "expected": {"content": "", "tool_calls": []}, "type": "call",
        "lang": "en", "family": "f/call", "split": "dev", "slots": {}, "source": {"request": "q-1"}}


def candidates(script: list) -> tuple[ScriptedModel, list[ScriptedModel]]:
    """A fresh greedy model and samples, each with its one scripted answer (a turn or an exception)."""
    return ScriptedModel([script[0]], name="m"), [ScriptedModel([s], name="m") for s in script[1:]]


SEARCH = api_call("local_0", "search_arxiv", '{"query":"moe"}')
DASHED = api_call("local_0", "search_arxiv", '{"query":"mixture-of-experts"}')
SPACED = api_call("local_0", "search_arxiv", '{"query":"mixture of experts"}')


def mixed() -> list:
    """The greedy answer and four samples' (an exception raised), made anew for every run."""
    return [local("", SEARCH, reasoning="g"), local("", DASHED, reasoning="s0"),
            local("", SEARCH, reasoning="s1"), local("", SPACED, reasoning="s2"),
            HarnessError("parse_tool_calls failed: boom")]


def one_at_a_time(script: list, **kwargs) -> list[dict]:
    """The record of each candidate's answer, run on its own (the runs the vote is taken over offline)."""
    return [record for answer_ in script for record in run_items([ITEM], ScriptedModel([answer_], name="m"),
                                                                 **kwargs)]


@pytest.mark.parametrize("mode", [FINAL, REPRODUCTION])
def test_the_voted_model_answers_with_the_offline_vote_of_its_candidates_records(mode):
    greedy, samples = candidates(mixed())
    (online,) = run_items([ITEM], VotedModel(greedy, samples, mode=mode), signals=True)
    records = one_at_a_time(mixed(), signals=True)
    assert online == voted_record(records[0], records[1:], HISTORY, mode)
    assert online["votes"]["errors"] == [{"candidate": 3, "error_type": "HarnessError",
                                          "message": "parse_tool_calls failed: boom"}]
    # every candidate was sent the context; the vote read the whole history (eval.runner's asking)
    assert greedy.contexts == [build_context(HISTORY)] and all(s.contexts for s in samples)
    assert online["usage"] == {"prompt_tokens": 400, "completion_tokens": 40}


def test_a_failed_representative_raises_its_own_error_with_the_vote_which_the_runner_records():
    def script() -> list:
        return [local("", SEARCH), HarnessError("bad call"), HarnessError("bad again"), local("", SEARCH)]

    (online,) = run_items([ITEM], VotedModel(*candidates(script()), mode=REPRODUCTION))
    records = one_at_a_time(script())
    assert online == voted_record(records[0], records[1:], HISTORY, REPRODUCTION)
    assert (online["status"], online["error"], online["votes"]["representative"]) == ("error", "bad call", 0)
    assert [e["message"] for e in online["votes"]["errors"]] == ["bad call", "bad again"]


class OwnError(HarnessError):
    """An exception type of its own, with a message of more than one line."""


# every exception a candidate may raise: its entry online (from the exception) and offline (from the runner's
# error record of the candidate's run alone) is one and the same, byte for byte
FAILURES = {"harness": lambda: HarnessError('parse_tool_calls failed: ValueError("引号 \\u2019 \'x\'")'),
            "key": lambda: KeyError("choices"),  # str() quotes a KeyError's key
            "empty": lambda: ValueError(),
            "lines": lambda: OwnError("first line\nsecond line\ttab"),
            "timeout": lambda: TimeoutError("timed out after 600 s")}


@pytest.mark.parametrize("failure", FAILURES)
@pytest.mark.parametrize("failing", [0, 1])  # the greedy candidate, or sample 0
def test_one_function_makes_a_failed_candidates_entry_online_and_offline_byte_for_byte(failure, failing):
    def script() -> list:
        answers: list = [local("", SEARCH), local("", SEARCH), local("", SEARCH)]
        answers[failing] = FAILURES[failure]()
        return answers

    (online,) = run_items([ITEM], VotedModel(*candidates(script()), mode=FINAL))
    records = one_at_a_time(script())
    offline = voted_record(records[0], records[1:], HISTORY, FINAL)
    assert canonical_json(online["votes"]["errors"]).encode() == canonical_json(
        offline["votes"]["errors"]).encode()
    assert online == offline and online["status"] == "ok"
    # the entry is the error record the runner makes of the exception (error_fields), under its own names
    error = FAILURES[failure]()
    assert records[failing]["error"] == error_fields(error)["error"] == str(error)
    assert records[failing]["error_type"] == error_fields(error)["error_type"] == type(error).__name__
    assert online["votes"]["errors"] == [{"candidate": "greedy" if failing == 0 else 0,
                                          "error_type": type(error).__name__, "message": str(error)}]


def test_in_the_loop_a_candidates_harness_error_leaves_the_turn_going_and_stays_under_votes_errors():
    tools, _ = table_tools({})
    executor = ToolExecutor(tools, table_check({}), today=TODAY)
    answered = turn(content="Two papers.", paid=False)
    boom = HarnessError("parse_tool_calls failed: ValueError('boom')")
    # without a vote the harness error ends the turn (agent.loop); a vote goes on over the other candidates
    alone = run_turn("Find moe papers.", model=ScriptedModel([boom]), executor=executor,
                     decide=table_decide(), today=TODAY)
    assert alone.status == "harness_error"
    voted = VotedModel(ScriptedModel([answered]), [ScriptedModel([boom]), ScriptedModel([answered])])
    transcript = run_turn("Find moe papers.", model=voted, executor=executor, decide=table_decide(),
                          today=TODAY)
    assert (transcript.status, transcript.final_text) == ("answered", "Two papers.")
    (attempt,) = transcript.steps[0]["attempts"]
    assert attempt["votes"]["errors"] == [{"candidate": 0, "error_type": "HarnessError",
                                           "message": "parse_tool_calls failed: ValueError('boom')"}]
    assert "harness_error" not in transcript.steps[0]


def test_the_vote_reads_the_whole_history_not_the_context_it_sends():
    # the id 2608.05678 is only in the older search output's snippet, which the context cuts away
    history = [*HISTORY, answer(api_call("c1", "get_paper", '{"arxiv_id":"2609.00002"}')),
               {"role": "tool", "tool_call_id": "c1", "content": canonical_json(HITS[1])}]
    cite = api_call("local_0", "fetch_fulltext", '{"arxiv_id":"2608.05678"}')
    model = VotedModel(*candidates([local("", cite), local("", cite)]), mode=FINAL)
    with asking(history):
        voted = model.next_turn(build_context(history), MODEL_TOOLS)
    assert json.loads(voted.votes["samples"][0])["ids_in_history"] is True
    alone = VotedModel(*candidates([local("", cite), local("", cite)]), mode=FINAL)
    assert json.loads(alone.next_turn(build_context(history), MODEL_TOOLS).votes["samples"][0])[
        "ids_in_history"] is False


def test_the_voted_model_keeps_the_representatives_turn_and_forwards_the_hooks():
    told = []
    greedy, samples = candidates([local("", SEARCH, reasoning="g"), local("", SPACED, reasoning="s0"),
                                  local("", DASHED, reasoning="s1")])
    for model in (greedy, *samples):
        model.begin_task = lambda m=model: told.append(m)
    voted = VotedModel(greedy, samples)
    assert voted.name == "m+votes2" and voted.mode == FINAL and voted.cost(None) == 0.0
    voted.begin_task()
    assert told == [greedy, *samples]
    got = voted.next_turn(HISTORY, MODEL_TOOLS)
    assert got.reasoning == "s0" and got.tool_calls == [ToolCall.from_api(SPACED)] and got.votes["k"] == 2
    with pytest.raises(ValueError, match="sample"):
        VotedModel(greedy, [])
    with pytest.raises(ValueError, match="mode"):
        VotedModel(greedy, samples, mode="coarse")


# ---- the candidates' generation settings, and the requests they send ---------------------------------------

THINK_SAMPLING = Decoding(temperature=0.6, top_p=0.95, top_k=0, min_p=0.0, presence_penalty=0.0)


def test_sample_k_is_the_run_with_base_seed_k_and_the_greedy_candidate_the_greedy_run():
    # the generation records of the FT3@1635 runs the vote is checked against, as their manifests hold them
    # (runs/m7/dev-ft3-1635-greedy.manifest.json; runs/eval/select-ft3/dev/checkpoint-1635.s0 and .s1,
    # runs/m7/voting/dev-ft3-1635-s2 to s4: base_seed k, sample 0)
    settings = GenerationSettings(mode=Mode.THINK, decoding=THINK_SAMPLING, reasoning_budget=1280, votes=5)
    greedy, samples = voted_settings(settings)
    seed_run = {"answer_tokens": 512, "decoding": {"min_p": 0.0, "presence_penalty": 0.0, "temperature": 0.6,
                                                   "top_k": 0, "top_p": 0.95},
                "mode": "think", "reasoning_budget": 1280, "sample": 0}
    assert [s.record() for s in samples] == [{**seed_run, "base_seed": k} for k in range(5)]
    assert greedy.record() == {"answer_tokens": 512, "base_seed": 0, "mode": "think",
                               "reasoning_budget": 1280, "sample": 0,
                               "decoding": {"min_p": 0.0, "presence_penalty": 0.0, "temperature": 0.0,
                                            "top_k": 0, "top_p": 1.0}}
    assert greedy.decoding == GREEDY and not any(s.votes for s in (greedy, *samples))
    with pytest.raises(ValueError, match="vote"):
        voted_settings(GenerationSettings(mode=Mode.THINK, decoding=THINK_SAMPLING))


class SeededLlama(FakeLlama):
    """A llama-server stand-in whose answer is a function of the request's seed: the greedy answer for a
    request without one, else sampled[seed % len(sampled)]; so a request answers alike in every run."""

    def __init__(self, greedy: Scripted, sampled: list[Scripted]) -> None:
        super().__init__([], adapters=[{"id": 0, "path": "ft.gguf", "scale": 1.0}])
        self.greedy, self.sampled = greedy, sampled

    def handle(self, request):
        if request.url.path == "/v1/completions":
            seed = json.loads(request.content).get("seed")
            self.script = [self.greedy if seed is None else self.sampled[seed % len(self.sampled)]]
        return super().handle(request)


TEXTS = ['<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "' + q + '"}}\n</tool_call>'
         for q in ("moe", "mixture of experts", "mixture-of-experts")]


def seeded() -> SeededLlama:
    return SeededLlama(Scripted(("Think.", "\n</think>\n\n", TEXTS[0])),
                       [Scripted(("Sample.", "\n</think>\n\n", text)) for text in TEXTS])


def test_online_the_candidates_send_the_seed_runs_requests_and_the_vote_is_the_offline_one(tmp_path):
    family = fake_loaded_family(tmp_path / "model", "hybrid", decoding={Mode.THINK: THINK_SAMPLING,
                                                                         Mode.NO_THINK: GREEDY})
    settings = GenerationSettings(mode=Mode.THINK, decoding=THINK_SAMPLING, reasoning_budget=16, votes=3)
    fake = seeded()
    greedy_settings, sample_settings = voted_settings(settings)

    def model(client, generation):
        return LocalModel.for_family(client, family, generation, lora_scale=1.0, name="m:lora=1")

    voted = VotedModel(model(fake.client(), greedy_settings),
                       [model(fake.client(), s) for s in sample_settings], mode=FINAL)
    (online,) = run_items([ITEM], voted, signals=True)
    prompt = fake.requests[0]["prompt"]
    seeds = [request_seed(k, prompt, 1, 0) for k in range(3)]
    assert [r.get("seed") for r in fake.requests] == [None, *seeds]
    # the same requests one run at a time: the greedy run and the runs with base seeds 0, 1 and 2
    runs = []
    for generation in (greedy_settings, *sample_settings):
        alone = seeded()
        runs += run_items([ITEM], model(alone.client(), generation), signals=True)
        assert alone.requests == [r for r in fake.requests if r.get("seed") == alone.requests[0].get("seed")]
    assert online == voted_record(runs[0], runs[1:], HISTORY, FINAL)
    assert online["votes"]["k"] == 3 and online["thinking"]["seeds"] is not None

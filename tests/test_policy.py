"""Specification of module 03 (failure policy). Do not edit these tests to make them pass.

Implemented on 2026-10-13 (every test passes; the xfail marker that covered the NotImplementedError stub is
gone). A change to policy.py must keep all of these green.
"""

import pytest

from research_agent.policy import decide
from research_agent.tools import schemas

STEP_EVENTS = ["bad_json", "unknown_tool", "invalid_args"]


# ---- step-check events: repair once, hand off once, then stop --------------------------------------------

@pytest.mark.parametrize("event", STEP_EVENTS)
@pytest.mark.parametrize("times, expected", [
    (1, "repair"), (2, "handoff"), (3, "stop"), (4, "stop"), (10, "stop"), (100, "stop"),
])
def test_step_check_events(event, times, expected):
    assert decide(event, times) == expected


def test_the_three_step_check_events_share_one_policy():
    # times counts their rejected attempts together, so the answer cannot depend on which one it was
    for times in range(1, 7):
        answers = {decide(event, times) for event in schemas.PER_STEP_EVENTS}
        assert len(answers) == 1


# ---- empty_results: relax once, then tell the user -------------------------------------------------------

@pytest.mark.parametrize("times, expected", [
    (1, "relax"), (2, "tell_user"), (3, "tell_user"), (10, "tell_user"),
])
def test_empty_results(times, expected):
    assert decide("empty_results", times) == expected


# ---- repeat_call: a dead loop, every time -----------------------------------------------------------------

@pytest.mark.parametrize("times", [1, 2, 3, 10])
def test_repeat_call_stops(times):
    assert decide("repeat_call", times) == "stop"


# ---- the answer is always one the loop may carry out ------------------------------------------------------

@pytest.mark.parametrize("event", schemas.EVENTS)
@pytest.mark.parametrize("times", range(1, 7))
def test_the_answer_is_allowed_for_the_event(event, times):
    answer = decide(event, times)
    assert isinstance(answer, str)
    assert answer in schemas.ALLOWED_ACTIONS[event]


def test_same_inputs_same_answer():
    for event in schemas.EVENTS:
        assert decide(event, 1) == decide(event, 1)
        assert decide(event, 2) == decide(event, 2)


# ---- programming errors raise ------------------------------------------------------------------------------

@pytest.mark.parametrize("event", ["timeout", "BAD_JSON", "bad json", "", "stop"])
def test_an_unknown_event_raises(event):
    with pytest.raises(ValueError):
        decide(event, 1)


@pytest.mark.parametrize("event", schemas.EVENTS)
@pytest.mark.parametrize("times", [0, -1, -10])
def test_times_below_one_raises(event, times):
    with pytest.raises(ValueError):
        decide(event, times)

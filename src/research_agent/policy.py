"""Failure policy: what the loop does after each failure event, as a function of how often it happened.

Module 03. The loop asks decide(event, times) at every failure point and carries out the answer; an answer
outside schemas.ALLOWED_ACTIONS[event] ends the turn as a harness error. tests/test_policy.py is the
specification; the lecture is docs/modules/03-失败策略判定.md.
"""

from __future__ import annotations

from .tools import schemas


def decide(event: str, times: int) -> str:
    """The action for the times-th occurrence of event, one of schemas.ALLOWED_ACTIONS[event].

    How times counts (schemas.PER_STEP_EVENTS): for bad_json, unknown_tool and invalid_args it is the number
    of rejected attempts of the current step so far, the three events counted together, and it restarts at 1
    in every step; for empty_results and repeat_call it is the number of times that event happened in the
    whole turn.

    - bad_json, unknown_tool, invalid_args: 1 -> "repair" (the errors go back to the same model once),
      2 -> "handoff" (the step goes to the cloud model, which keeps the rest of the turn), 3 or more ->
      "stop" (the turn ends as invalid_output).
    - empty_results: 1 -> "relax" (the model is told to loosen the search once), 2 or more -> "tell_user"
      (the model is told to say nothing was found and to invent no id).
    - repeat_call: always "stop" (the same call for the third time is a dead loop).

    An event outside schemas.EVENTS or times < 1 is a programming error: raise ValueError.
    """
    if event not in schemas.EVENTS or times < 1:
        raise ValueError(f"decide({event!r}, {times!r}): unknown event or times below 1")
    if event in schemas.PER_STEP_EVENTS:
        if times == 1:
            return "repair"
        if times == 2:
            return "handoff"
        return "stop"
    if event == "empty_results":
        return "relax" if times == 1 else "tell_user"
    return "stop"

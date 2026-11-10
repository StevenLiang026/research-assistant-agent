"""Runtime rules (docs/m7-本地提升计划.md WP3): deterministic rewrites of a local model's answer from the
answer and its conversation alone, never the gold, as the 10-29 error analysis defined and measured them
(its errors_rules.py, kept outside the repository: on dev and dev-v2 with FT3's greedy
answers, transient_retry fixes 7 items and id_repair 1, and neither breaks one).

An answer is the assistant message in API form (AssistantTurn.to_api, a runner record's "output"); its
history is the whole conversation before it: the item's messages in the single-step runner, the transcript's
messages end to end, never the context build_context cut from them (agent.loop.current_history). The rules
apply in RULES' order, each to what the one before it left:

- transient_retry. When the history ends with an assistant step's tool results and some of them carry a
  transient error (TRANSIENT_ERRORS: RATE_LIMITED, UPSTREAM_ERROR; failed_calls), the answer becomes exactly
  those failed calls again, in their order, with their names and arguments, new ids and no text: the step's
  successful calls are not repeated. An answer whose calls already are those (the same names and parsed
  arguments in the same order, whatever its ids, spelling and text) is left as it is. The rule does not fire
  when one of the failed calls already ran agent.loop.MAX_RUNS_PER_CALL times in the history's current turn,
  since its last user message: the loop would answer another run SKIPPED. A call ran when its tool result is
  not a SKIPPED one, counted by call_key as the loop counts its runs (call_runs; ask_clarification is never
  run).
- id_repair. The history's ids are the arXiv ids metrics.arxiv_ids finds in every message's text and every
  call's arguments, exactly as the scorer's ids_grounded reads the history (history_ids). A call's arxiv_id
  argument whose canonical id (normalize_arxiv_id) is not among them becomes the one id among them with the
  same sequence part (what follows the dot: "00002" of "2609.00002"), when exactly one has it; otherwise, and
  for an argument that is no arXiv id, it stays.

A rule that changes an answer writes it as the analysis did (errors_rules.make_output), so that its outputs
are the analysis' byte for byte: the text kept (id_repair) or empty (transient_retry), and every call of the
answer with the id pp_<k> (k its position) and its arguments as json.dumps(the parsed arguments,
ensure_ascii=False) writes them. One deviation, which the analysis' data never meets: a call whose arguments
do not parse keeps its text, where the analysis would write "null".

apply_rules returns the answer the rules leave and their changes, one {"rule", "before", "after"} (the API
forms around the change) per rule that fired, [] when none did; ruled_record does the same to a runner record
offline (scripts/vote_offline.py), so a record ruled offline is the record the ruled model gives online.
RuledModel applies them to the wrapped model's turn: the turn keeps its signals, reasoning and thinking
record, which describe the answer the model gave, and carries the changes as AssistantTurn.rules ([] when
nothing changed). It sits inside any routing wrapper and, when voting, inside every candidate
(agent.vote.VotedModel), so the end-to-end loop's check_step validates the rewritten calls. RULES_VERSION
names this rule set in the run manifests.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from collections import Counter
from collections.abc import Callable, Sequence

from ..eval.metrics import _message_text, arxiv_ids
from ..eval.runner import OK
from ..normalize import normalize_arxiv_id
from ..tools.schemas import RATE_LIMITED, SKIPPED, UPSTREAM_ERROR
from .loop import ASK, MAX_RUNS_PER_CALL, Model, begin_task, current_history
from .types import AssistantTurn, ToolCall, call_key

TRANSIENT_RETRY, ID_REPAIR = "transient_retry", "id_repair"
RULES = (TRANSIENT_RETRY, ID_REPAIR)  # the order they apply in
RULES_VERSION = "m7-wp3-1"  # the rule set of this module, as the run manifests record it
TRANSIENT_ERRORS = frozenset({RATE_LIMITED, UPSTREAM_ERROR})
RULE_CALL_ID = "pp_{}"  # a rewritten answer's call ids (errors_rules.make_output's)
ID_PARAM = "arxiv_id"

Rule = Callable[[list[dict], dict], dict | None]  # (history, answer) -> the new answer, None when unchanged


class _Unparsed:
    """The arguments of a call whose text is no JSON: the text itself, kept as it is."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Unparsed) and other.text == self.text

    def __hash__(self) -> int:
        return hash(self.text)


def _parsed(arguments: object) -> object:
    """A call's arguments as the analysis reads them (errors_rules.calls): JSON text parsed, a value given
    as it is; text that is no JSON stays text (_Unparsed)."""
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return _Unparsed(arguments)


def _calls(message: dict | None) -> list[list]:
    """[[name, parsed arguments]] of an assistant message's calls, in order."""
    out = []
    for call in (message or {}).get("tool_calls") or []:
        function = (call.get("function") or {}) if isinstance(call, dict) else {}
        out.append([function.get("name"), _parsed(function.get("arguments"))])
    return out


def _written(calls: list[list], content: str) -> dict:
    """The answer a rule writes (module docstring): errors_rules.make_output's form."""
    return {"role": "assistant", "content": content,
            "tool_calls": [{"id": RULE_CALL_ID.format(k), "type": "function",
                            "function": {"name": name, "arguments": arguments.text
                                         if isinstance(arguments, _Unparsed)
                                         else json.dumps(arguments, ensure_ascii=False)}}
                           for k, (name, arguments) in enumerate(calls)]}


def _results(messages: list[dict], start: int) -> dict[str, object]:
    """The tool messages right after messages[start - 1], by the call id they answer."""
    results = {}
    for message in messages[start:]:
        if message.get("role") != "tool":
            break
        results[message.get("tool_call_id")] = message.get("content")
    return results


def _error(content: object) -> object:
    """The error code of a tool message's content, None without one."""
    try:
        body = json.loads(content) if isinstance(content, str) else None
    except json.JSONDecodeError:
        return None
    return body.get("error") if isinstance(body, dict) else None


def failed_calls(history: list[dict]) -> list[dict]:
    """The calls (API form) of the history's last assistant step whose tool result carries a transient
    error, in call order; [] when the history does not end with that step's tool results or none failed
    (errors_rules.last_step_failures)."""
    last = max((i for i, m in enumerate(history) if m.get("role") == "assistant"), default=None)
    if last is None or not history[last].get("tool_calls") or history[-1].get("role") != "tool":
        return []
    results = {m.get("tool_call_id"): m.get("content") for m in history[last + 1:] if m.get("role") == "tool"}
    return [call for call in history[last]["tool_calls"]
            if _error(results.get(call.get("id"))) in TRANSIENT_ERRORS]


def call_runs(history: list[dict]) -> Counter[str]:
    """How often each call (call_key) ran in the history's current turn, since its last user message, as the
    loop counts its runs: every call whose tool result is not SKIPPED; ask_clarification is never run."""
    start = max((i for i, m in enumerate(history) if m.get("role") == "user"), default=-1) + 1
    runs: Counter[str] = Counter()
    for index in range(start, len(history)):
        message = history[index]
        if message.get("role") != "assistant" or not message.get("tool_calls"):
            continue
        results = _results(history, index + 1)
        for call in message["tool_calls"]:
            parsed = ToolCall.from_api(call)
            if parsed.name == ASK or call.get("id") not in results:
                continue
            if _error(results[call.get("id")]) != SKIPPED:
                runs[call_key(parsed)] += 1
    return runs


def transient_retry(history: list[dict], answer: dict) -> dict | None:
    """The retry of the failed calls (module docstring), None when the rule leaves the answer."""
    failed = failed_calls(history)
    if not failed:
        return None
    runs = call_runs(history)
    if any(runs[call_key(ToolCall.from_api(call))] >= MAX_RUNS_PER_CALL for call in failed):
        return None
    # a call that ran had arguments, read as "{}" when empty (ToolCall.from_api, errors_rules alike)
    again = [[(call.get("function") or {}).get("name"),
              _parsed((call.get("function") or {}).get("arguments") or "{}")] for call in failed]
    if _calls(answer) == again:
        return None
    return _written(again, "")


def history_ids(history: list[dict]) -> set[str]:
    """The canonical arXiv ids of the whole history, read as metrics.score_item's ids_grounded reads it."""
    return arxiv_ids("\n".join(_message_text(m) for m in history if isinstance(m, dict)))


def _sequence(arxiv_id: str) -> str:
    """The sequence part of a canonical arXiv id: what follows the dot (errors_rules.r_id_repair)."""
    return arxiv_id.split(".")[1]


def id_repair(history: list[dict], answer: dict) -> dict | None:
    """The answer with every unseen arxiv_id repaired (module docstring), None when none is."""
    seen = history_ids(history)
    calls = _calls(answer)
    repaired = copy.deepcopy(calls)
    for _name, arguments in repaired:
        if isinstance(arguments, dict) and isinstance(arguments.get(ID_PARAM), str):
            wanted = normalize_arxiv_id(arguments[ID_PARAM])
            if wanted and wanted not in seen:
                same = [shown for shown in seen if _sequence(shown) == _sequence(wanted)]
                if len(same) == 1:
                    arguments[ID_PARAM] = same[0]
    return None if repaired == calls else _written(repaired, answer.get("content") or "")


RULE_FUNCTIONS: dict[str, Rule] = {TRANSIENT_RETRY: transient_retry, ID_REPAIR: id_repair}


def apply_rules(history: list[dict], answer: dict, rules: Sequence[str] = RULES) -> tuple[dict, list[dict]]:
    """(the answer the rules leave, their changes) for an answer to this history (module docstring); rules
    are names of RULE_FUNCTIONS, applied in the order given. The inputs are not changed."""
    changes = []
    for name in rules:
        after = RULE_FUNCTIONS[name](history, answer)
        if after is not None:
            changes.append({"rule": name, "before": answer, "after": after})
            answer = after
    return answer, changes


def ruled_record(record: dict, history: list[dict], rules: Sequence[str] = RULES) -> dict:
    """A runner record (eval.runner) with the rules applied to its output over the item's history, the
    record RuledModel gives online: the output replaced and "rules" added ([] when nothing changed); a record
    that is not ok, which has no output, as it is."""
    if record.get("status") != OK:
        return record
    output, changes = apply_rules(history, record["output"], rules)
    return {**record, "output": output, "rules": changes}


class RuledModel:
    """The wrapped model, its answers rewritten by the runtime rules (module docstring) over the whole
    history (agent.loop.current_history; outside a request, the messages it is given)."""

    def __init__(self, model: Model, rules: Sequence[str] = RULES) -> None:
        unknown = [name for name in rules if name not in RULE_FUNCTIONS]
        if unknown:
            raise ValueError(f"unknown runtime rules {unknown}; the rules are {list(RULE_FUNCTIONS)}")
        self.model, self.rules = model, tuple(rules)

    @property
    def name(self) -> str:
        return f"{self.model.name}+rules"

    def begin_task(self) -> None:
        begin_task(self.model)

    def next_turn(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        turn = self.model.next_turn(messages, tools)
        history = current_history()
        output, changes = apply_rules(messages if history is None else history, turn.to_api(), self.rules)
        if not changes:
            return dataclasses.replace(turn, rules=[])
        return dataclasses.replace(turn, content=output.get("content") or "",
                                   tool_calls=[ToolCall.from_api(c) for c in output.get("tool_calls") or []],
                                   rules=changes)

    def cost(self, turn: AssistantTurn) -> float:
        cost = getattr(self.model, "cost", None)
        return 0.0 if cost is None else cost(turn)

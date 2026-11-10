"""Confidence routing (docs/plan.md, 置信度路由): the local model answers a step unless it is unsure; then the
cloud model answers that step and every later step of the same task.

Signals of one local step (step_signals) are pure functions of the completion data (its text, its tokens,
their log-probabilities and, when llama-server sent them, each token's UTF-8 bytes) together with the step's
tools, the date and the paper ids the context showed, so they can be recomputed offline from saved data. The
completion is the answer's (docs/m7-家族层设计.md §8): a model that thinks first gives its reasoning in a
block of its own, which llm.generation leaves out of the answer Completion it hands LocalModel, so the
signals of a thinking step are those of its answer alone.

- Decision tokens (decision_tokens): the first generated token; for each call of the text, every token that
  overlaps a character of the tool name, and for each argument the first token that overlaps the first
  character of its value (for a string value, the first character after the opening quote). Where the calls
  are is the family's tool format's business (locate, research_agent.family.toolformats.ToolFormat.locate:
  the character spans of the names and values, one CallSpan per call); the default reads Qwen3's <tool_call>
  blocks as research_agent.toolcalls reads them (qwen_json). Tokens are placed in the text through their
  bytes (UTF-8 offsets) when there are bytes for every token, else through the token strings. A character
  split across two tokens (a Chinese character under a byte-level BPE) overlaps both, so the first of them
  is the one taken. Arguments written as JSON text (a string the parser passes on) give no argument tokens:
  their characters sit escaped inside that string.
- min_decision_prob: the minimum of exp(logprob) over the decision tokens; mean_logprob: the mean log-prob
  over all generated tokens; each None for a completion without tokens.
- hard_reason (hard_trigger): the first of these that applies, as "<kind>: <detail>", or None (always None
  for a plain answer without calls):
      parse_failure          a call block the format cannot read as a call (locate: a CallSpan that is not
                             parsed; toolcalls' fallback form for Qwen3), or a parser that raised or
                             returned something other than a list of dicts
      invalid_after_repair   a call whose arguments are still not a JSON object after LocalModel's own
                             repair (text arguments parsed: agent.models._local_call)
      unknown_tool           a call naming a tool the step does not offer
      invalid_args (or the executor's other event)  the step as the executor refuses it before it runs:
                             ToolExecutor.check_step with validation.check_call, the context's date and the
                             ids its tool outputs showed (too many calls, arguments the schema or the meaning
                             rules refuse). A context without a date in its system message (context_facts)
                             skips check_call; the step-level checks still run.

RoutedModel(local, cloud, config) asks the local model; when its turn has a hard reason or the configured
signal is below the threshold (route_reason: a missing value counts as below), the cloud model answers this
step and every later step until begin_task, which the loop's run_turn and the runner's run_items call at the
start of every task. The returned turn is the answering model's, with route = {"answered_by": "local" or
"cloud", both signal values, the hard reason and the reason it was routed}; a step answered in the cloud
because an earlier one was routed has no signal values. Cost, paid and cached are the cloud turns' only: a
local turn is neither paid nor cached and costs nothing.

Stepwise routing (M6, docs/m6-计划.md §3.3 C7; RoutedModel(..., stepwise=True), or its stepwise attribute set
before a task): every step is decided on its own, the local model asked again after a routed step, and a
final text answer, a local turn without any tool call and without a hard reason, is never routed, whatever
its signal and whether or not the rules exempt text (the plan's C1 over C7). A hard reason still routes the
step. The default, sticky routing, is what M4 and M5 ran.

Rules (M6, docs/m6-计划.md §3.3 C1 and C2; RouteRules, the configuration's "rules"). With text_exempt, a
turn without any tool call stays local whatever its signal: a long answer's first token is improbable by
nature, and on the development set every such turn was right. Each param_trigger names a tool, a parameter
and the values that do not trigger (the tool's default), and a call giving the parameter any other value is
routed before the threshold is looked at: the habits FT learnt (search_notes.top_k 5, list_notes.limit 20)
are invisible to a probability and plain to a rule that reads the value. With budget_forced (M7,
docs/m7-家族层设计.md §8), a step whose reasoning hit its budget and was closed by force (llm.generation;
AssistantTurn.thinking["forced"], route_reason's forced) is routed like a hard trigger: before the parameter
triggers, the text exemption and the threshold, and under stepwise routing a final text answer too. It is
off unless the configuration turns it on, so the frozen configurations keep their meaning. route_reason's
order: the hard reason, the budget trigger, a parameter trigger, the text exemption, the threshold. A
configuration without rules routes as before (results/routing-m4.json).
"""

from __future__ import annotations

import bisect
import dataclasses
import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..family.toolformats import CallSpan
from ..family.toolformats.qwen_json import QWEN_JSON
from ..toolcalls import parse_tool_calls as qwen3_tool_calls
from .executor import ToolExecutor, _paper_ids
from .types import AssistantTurn, CheckCall, HarnessError, ToolCall

MIN_DECISION_PROB, MEAN_LOGPROB = "min_decision_prob", "mean_logprob"
SIGNALS = (MIN_DECISION_PROB, MEAN_LOGPROB)  # the preregistered order: a tie goes to the first
HARD_REASON = "hard_reason"
BUDGET_FORCED = "budget_forced"  # the reason's kind when the budget trigger routes a step
LOCAL, CLOUD = "local", "cloud"
STAYED = "routed earlier in this task"

_TODAY = re.compile(r"今天是 (\d{4}-\d{2}-\d{2})")  # agent.prompts' template

ParseToolCalls = Callable[[str], list[dict]]
Locate = Callable[[str], list[CallSpan]]  # a tool format's locate: where the calls of a text are

# ---- where the decisions are in the text ----------------------------------------------------------------


def token_spans(text: str, tokens: Sequence[str],
                token_bytes: Sequence[Sequence[int]] | None = None) -> list[tuple[int, int]]:
    """The character span [start, end) of every token in text: through the bytes (UTF-8 offsets) when there
    are bytes for every token, else through the lengths of the token strings. A token holding part of a
    character spans that whole character."""
    if token_bytes is None or len(token_bytes) != len(tokens):
        spans, start = [], 0
        for token in tokens:
            spans.append((start, start + len(token)))
            start += len(token)
        return spans
    starts, offset = [], 0  # the byte offset at which each character starts
    for char in text:
        starts.append(offset)
        offset += len(char.encode("utf-8", errors="surrogatepass"))
    spans, low = [], 0
    for data in token_bytes:
        high = low + len(data)
        first = max(bisect.bisect_right(starts, low) - 1, 0) if low < offset else len(text)
        last = bisect.bisect_right(starts, high - 1) if high > low else first
        spans.append((first, max(first, min(last, len(text)))))
        low = high
    return spans


def decision_tokens(text: str, tokens: Sequence[str], token_bytes: Sequence[Sequence[int]] | None = None, *,
                    locate: Locate = QWEN_JSON.locate) -> list[int]:
    """The indices of the decision tokens (module docstring), ascending; locate finds the calls."""
    if not tokens:
        return []
    spans = token_spans(text, tokens, token_bytes)
    chosen = {0}
    for call in locate(text):
        if call.name is not None:
            first, last = call.name
            chosen.update(i for i, (a, b) in enumerate(spans) if a < last and first < b)
        for start in call.values:
            hit = next((i for i, (a, b) in enumerate(spans) if a <= start < b), None)
            if hit is not None:
                chosen.add(hit)
    return sorted(chosen)


# ---- hard triggers -------------------------------------------------------------------------------------


def context_facts(messages: list[dict]) -> tuple[str | None, list[str]]:
    """The date of the context's system message (agent.prompts' "今天是 YYYY-MM-DD", None without one) and
    the paper ids its tool messages showed, sorted: what the executor's seen ids were when the step was asked
    (ToolExecutor collects them with the same rule from the outputs it ran; older outputs keep their ids when
    the context compresses them)."""
    today = None
    for message in messages:
        if message.get("role") == "system":
            found = _TODAY.search(message.get("content") or "")
            today = found.group(1) if found else None
            break
    seen: set[str] = set()
    for message in messages:
        if message.get("role") == "tool":
            try:
                seen.update(_paper_ids(json.loads(message.get("content") or "")))
            except (json.JSONDecodeError, TypeError):
                continue
    return today, sorted(seen)


def hard_trigger(text: str, *, tool_names: Sequence[str], today: str | None, seen_ids: Sequence[str],
                 parse: ParseToolCalls = qwen3_tool_calls, check_call: CheckCall | None = None,
                 locate: Locate = QWEN_JSON.locate) -> str | None:
    """The hard reason of a step (module docstring), None when there is none. check_call defaults to the
    owner's validation.check_call, which the executor runs; parse and locate to the Qwen3 JSON format's."""
    for number, call in enumerate(locate(text), 1):
        if not call.parsed:  # the wording of the recorded signals before M7, kept so their bytes stand
            return f"parse_failure: <tool_call> block {number} is not a call the parser can read"
    try:
        parsed = parse(text)
    except Exception as exc:  # noqa: BLE001  the parser's failure is the step's reason
        return f"parse_failure: the parser raised {exc!r}"
    if not isinstance(parsed, list) or not all(isinstance(p, dict) for p in parsed):
        return f"parse_failure: the parser returned {parsed!r:.200}"
    from .models import _local_call  # agent.models imports this module: imported where it is used

    calls = []
    for index, call in enumerate(parsed):
        try:
            calls.append(_local_call(index, call))
        except HarnessError as exc:
            return f"invalid_after_repair: {exc}"
    for call in calls:
        if call.arguments is None:
            return f"invalid_after_repair: the arguments of {call.name!r} are not a JSON object"
    for call in calls:
        if call.name not in tool_names:
            return f"unknown_tool: {call.name!r} is not among the step's tools"
    if not calls:
        return None
    check: CheckCall = _no_check
    if today is not None:
        owner = check_call
        if owner is None:
            from ..validation import check_call as owner  # the owner's module, imported where it is used
        ids = list(seen_ids)

        def check(tool: str, args: dict, _seen: list[str], day: str) -> list[str]:
            return owner(tool, args, list(ids), day)
    executor = ToolExecutor({}, check, today=today or "")
    try:
        issues = executor.check_step(calls)
    except HarnessError as exc:
        return f"invalid_args: the executor's check failed: {exc}"
    if not issues:
        return None
    return f"{issues[0].event}: {'; '.join(issues[0].messages)}"


def _no_check(tool: str, args: dict, seen_ids: list[str], today: str) -> list[str]:
    return []


# ---- signals ----------------------------------------------------------------------------------------------


def step_signals(text: str, tokens: Sequence[str], logprobs: Sequence[float],
                 token_bytes: Sequence[Sequence[int]] | None = None, *, tool_names: Sequence[str],
                 today: str | None, seen_ids: Sequence[str], parse: ParseToolCalls = qwen3_tool_calls,
                 check_call: CheckCall | None = None, locate: Locate = QWEN_JSON.locate) -> dict:
    """{"min_decision_prob", "mean_logprob", "hard_reason"} of one step (module docstring); parse and locate
    read the calls as the family's tool format does."""
    decisions = [i for i in decision_tokens(text, tokens, token_bytes, locate=locate) if i < len(logprobs)]
    return {
        MIN_DECISION_PROB: min(math.exp(logprobs[i]) for i in decisions) if decisions else None,
        MEAN_LOGPROB: sum(logprobs) / len(logprobs) if logprobs else None,
        HARD_REASON: hard_trigger(text, tool_names=tool_names, today=today, seen_ids=seen_ids, parse=parse,
                                  check_call=check_call, locate=locate),
    }


def completion_signals(completion: object, tools: list[dict], messages: list[dict], *,
                       parse: ParseToolCalls = qwen3_tool_calls, locate: Locate = QWEN_JSON.locate) -> dict:
    """step_signals of a llm.local.Completion asked with these tools over this context (API messages): the
    answer completion of a step (module docstring)."""
    names = [t["function"]["name"] for t in tools
             if isinstance(t, dict) and isinstance(t.get("function"), dict)]
    today, seen = context_facts(messages)
    return step_signals(completion.text, completion.tokens, completion.logprobs,
                        getattr(completion, "token_bytes", None), tool_names=names, today=today,
                        seen_ids=seen, parse=parse, locate=locate)


# ---- rules (M6, docs/m6-计划.md §3.3 C1 and C2) --------------------------------------------------------

PARAM_TRIGGER = "param_trigger"
_SCALARS = (str, int, float, bool, type(None))  # the JSON scalars a trigger's allowed values may be


@dataclass(frozen=True)
class ParamTrigger:
    """A tool's parameter whose value, when the call gives one that is not among the allowed values, routes
    the step ({"tool", "param", "allowed"} in the configuration's rules). A call that leaves the parameter
    out never triggers: the tool's default applies."""

    tool: str
    param: str
    allowed: tuple[object, ...]

    def reason(self, call: ToolCall) -> str | None:
        """Why the call triggers, naming the tool, the parameter and the value; None when it does not. A
        call whose arguments are not an object is the hard trigger's business and never triggers here."""
        if call.name != self.tool or call.arguments is None or self.param not in call.arguments:
            return None
        value = call.arguments[self.param]
        if any(_same(value, allowed) for allowed in self.allowed):
            return None
        return (f"{PARAM_TRIGGER}: {self.tool}.{self.param}={json.dumps(value, ensure_ascii=False)} is not "
                f"among {json.dumps(list(self.allowed), ensure_ascii=False)}")


def _require(ok: bool, message: str) -> None:
    """ValueError naming what is malformed unless ok: a configuration's fault, whatever its kind."""
    if not ok:
        raise ValueError(message)


def _same(value: object, allowed: object) -> bool:
    """Equal as JSON values: true is not 1 (Python's bool is an int), 3 and 3.0 are the same number."""
    return isinstance(value, bool) is isinstance(allowed, bool) and value == allowed


@dataclass(frozen=True)
class RouteRules:
    """The rules a configuration adds to the hard triggers and the threshold (module docstring): with
    text_exempt a turn without any tool call stays local whatever its signal; each of param_triggers routes
    a call whose parameter has a value not among its allowed ones; with budget_forced a step whose reasoning
    was closed by force is routed."""

    text_exempt: bool = False
    param_triggers: tuple[ParamTrigger, ...] = ()
    budget_forced: bool = False

    def __post_init__(self) -> None:
        _require(isinstance(self.text_exempt, bool),
                 f"rules: text_exempt must be true or false, got {self.text_exempt!r}")
        _require(isinstance(self.param_triggers, tuple)
                 and all(isinstance(t, ParamTrigger) for t in self.param_triggers),
                 "rules: param_triggers must be a tuple of ParamTrigger")
        _require(isinstance(self.budget_forced, bool),
                 f"rules: budget_forced must be true or false, got {self.budget_forced!r}")

    @classmethod
    def from_json(cls, data: object) -> RouteRules:
        """The rules as the routing configuration writes them ("rules": {"text_exempt": bool,
        "param_triggers": [{"tool", "param", "allowed": [...]}], "budget_forced": bool}, every key optional);
        ValueError naming what is malformed."""
        _require(isinstance(data, dict), f"rules must be a JSON object, got {type(data).__name__}")
        unknown = sorted(set(data) - {"text_exempt", "param_triggers", BUDGET_FORCED})
        _require(not unknown, f"rules: unknown key {', '.join(unknown)}")
        exempt = data.get("text_exempt", False)
        _require(isinstance(exempt, bool), f"rules: text_exempt must be true or false, got {exempt!r}")
        triggers = data.get("param_triggers", [])
        _require(isinstance(triggers, list),
                 f"rules: param_triggers must be a list, got {type(triggers).__name__}")
        forced = data.get(BUDGET_FORCED, False)
        _require(isinstance(forced, bool), f"rules: budget_forced must be true or false, got {forced!r}")
        return cls(text_exempt=exempt, param_triggers=tuple(_trigger(t, i) for i, t in enumerate(triggers)),
                   budget_forced=forced)

    def to_json(self) -> dict:
        """The form from_json reads; budget_forced is written only when it is on, so the configurations
        frozen before it keep their bytes."""
        rules = {"text_exempt": self.text_exempt,
                 "param_triggers": [{"tool": t.tool, "param": t.param, "allowed": list(t.allowed)}
                                    for t in self.param_triggers]}
        if self.budget_forced:
            rules[BUDGET_FORCED] = True
        return rules


def _trigger(data: object, index: int) -> ParamTrigger:
    where = f"rules: param_triggers[{index}]"
    _require(isinstance(data, dict), f"{where} must be a JSON object, got {type(data).__name__}")
    unknown = sorted(set(data) - {"tool", "param", "allowed"})
    _require(not unknown, f"{where}: unknown key {', '.join(unknown)}")
    for key in ("tool", "param"):
        _require(isinstance(data.get(key), str) and bool(data[key]),
                 f"{where}.{key} must be a non-empty string, got {data.get(key)!r}")
    allowed = data.get("allowed")
    _require(isinstance(allowed, list) and all(isinstance(v, _SCALARS) for v in allowed),
             f"{where}.allowed must be a list of JSON scalars, the values that do not trigger, got "
             f"{allowed!r}")
    return ParamTrigger(tool=data["tool"], param=data["param"], allowed=tuple(allowed))


def api_calls(message: dict | None) -> list[ToolCall]:
    """The tool calls of a recorded assistant turn in API form (the runner's "output"), as ToolCall.from_api
    reads them with their arguments parsed as the model's were; none for a plain answer or no message."""
    if not isinstance(message, dict):
        return []
    return [ToolCall.from_api(c) for c in message.get("tool_calls") or [] if isinstance(c, dict)]


# ---- routing ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteConfig:
    """Which signal is compared and below which threshold a step is routed (results/routing-m4.json), and
    the rules when the configuration has them (RouteRules; None routes by the hard triggers and the threshold
    alone, as results/routing-m4.json does)."""

    signal: str
    threshold: float
    rules: RouteRules | None = None

    def __post_init__(self) -> None:
        if self.signal not in SIGNALS:
            raise ValueError(f"unknown signal {self.signal!r}; one of {list(SIGNALS)}")
        if (not isinstance(self.threshold, int | float) or isinstance(self.threshold, bool)
                or not math.isfinite(self.threshold)):
            raise ValueError(f"the threshold must be a finite number, got {self.threshold!r}")
        _require(self.rules is None or isinstance(self.rules, RouteRules),
                 f"rules must be RouteRules or None, got {type(self.rules).__name__}")


def route_reason(signals: dict | None, config: RouteConfig, calls: Sequence[ToolCall] | None = None, *,
                 forced: bool = False) -> str | None:
    """Why a step with these signals is routed, None when the local answer stands, in this order: a step
    without signals is routed ("no signals"); its hard reason; with rules, the budget trigger when it is on
    and the step's reasoning was closed by force (forced: AssistantTurn.thinking["forced"], or a recorded
    turn's "thinking"), the first parameter trigger that fires over the step's calls in their order
    (ParamTrigger.reason), then None for a step without any call when text_exempt is on; else the configured
    signal below the threshold, a missing value counting as below. calls are the step's tool calls
    (AssistantTurn.tool_calls, or api_calls of a recorded turn): a configuration with rules needs them
    (ValueError without), one without rules ignores them."""
    if signals is None:
        return "no signals"
    if signals.get(HARD_REASON):
        return signals[HARD_REASON]
    if config.rules is not None:
        if calls is None:
            raise ValueError("the configuration has rules, which need the step's calls")
        if forced and config.rules.budget_forced:
            return f"{BUDGET_FORCED}: the reasoning hit its budget and was closed by force"
        for call in calls:
            for trigger in config.rules.param_triggers:
                reason = trigger.reason(call)
                if reason is not None:
                    return reason
        if config.rules.text_exempt and not calls:
            return None
    value = signals.get(config.signal)
    if value is None:
        return f"{config.signal} unknown (no tokens)"
    if value < config.threshold:
        return f"{config.signal} {value:.6g} < {config.threshold:.6g}"
    return None


class RoutedModel:
    """The local model, handing a task to the cloud model once a step is routed (module docstring), or
    with stepwise each step on its own (module docstring: Stepwise routing)."""

    def __init__(self, local: object, cloud: object, config: RouteConfig, *, stepwise: bool = False) -> None:
        self.local, self.cloud, self.config = local, cloud, config
        self.stepwise = stepwise
        self.in_cloud = False

    @property
    def name(self) -> str:
        return f"{self.local.name}>{self.cloud.name}"

    def begin_task(self) -> None:
        """A new task starts on the local model."""
        self.in_cloud = False

    def next_turn(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        if self.in_cloud:
            turn = self.cloud.next_turn(messages, tools)
            return dataclasses.replace(turn, route=_route(CLOUD, None, STAYED))
        local = self.local.next_turn(messages, tools)
        reason = route_reason(local.signals, self.config, local.tool_calls,
                              forced=bool((local.thinking or {}).get("forced")))
        if reason is not None and self.stepwise and _final_text(local, reason):
            reason = None
        if reason is None:
            return dataclasses.replace(local, route=_route(LOCAL, local.signals, None))
        if not self.stepwise:
            self.in_cloud = True
        turn = self.cloud.next_turn(messages, tools)
        return dataclasses.replace(turn, route=_route(CLOUD, local.signals, reason))

    def cost(self, turn: AssistantTurn) -> float:
        """The cloud model's cost of a turn it answered; 0.0 for a local one."""
        cost = getattr(self.cloud, "cost", None)
        if (turn.route or {}).get("answered_by") != CLOUD or cost is None:
            return 0.0
        return cost(turn)


def _final_text(turn: AssistantTurn, reason: str) -> bool:
    """A final text answer (module docstring: Stepwise routing): no tool call and no hard trigger, the budget
    trigger included."""
    return (not turn.tool_calls and not (turn.signals or {}).get(HARD_REASON)
            and not reason.startswith(f"{BUDGET_FORCED}:"))


def _route(answered_by: str, signals: dict | None, reason: str | None) -> dict:
    signals = signals or {}
    return {"answered_by": answered_by, MIN_DECISION_PROB: signals.get(MIN_DECISION_PROB),
            MEAN_LOGPROB: signals.get(MEAN_LOGPROB), HARD_REASON: signals.get(HARD_REASON), "reason": reason}

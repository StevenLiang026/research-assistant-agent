"""Data types shared by the model adapters, the executor, the context builder and the loop.

Two message formats exist. API messages (OpenAI format, what DeepSeek and the transcripts use) carry tool-call
arguments as a JSON string; HF template messages (what the Qwen3 chat template renders for training and the
local model) carry them as a dict. api_to_hf converts the first into the second.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

from ..canonical import canonical_json
from ..tools.schemas import DEFAULTS

# Turn outcomes recorded in a Transcript.
ANSWERED = "answered"
CLARIFICATION = "clarification"
DEAD_LOOP = "dead_loop"
STEP_LIMIT = "step_limit"
BUDGET_STOP = "budget_stop"
INVALID_OUTPUT = "invalid_output"
CONTEXT_OVERFLOW = "context_overflow"
HARNESS_ERROR = "harness_error"
STATUSES = [ANSWERED, CLARIFICATION, DEAD_LOOP, STEP_LIMIT, BUDGET_STOP, INVALID_OUTPUT, CONTEXT_OVERFLOW,
            HARNESS_ERROR]


class HarnessError(RuntimeError):
    """A failure of the framework itself (replay miss, bug in a tool adapter). Never shown to the model."""


def parse_arguments(arguments_json: str) -> dict | None:
    """The arguments of an API tool call as a dict, or None when they are not a JSON object.

    Integral floats become ints (3.0 -> 3, normalize_arguments): JSON Schema already counts 3.0 as an integer,
    so without this a schema-valid argument could break slicing in a tool, and call keys for 5 and 5.0 would
    differ.
    """
    try:
        value = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return None
    return normalize_arguments(value) if isinstance(value, dict) else None


def normalize_arguments(value: object) -> object:
    """value with every integral float turned into an int (3.0 -> 3), recursively through dicts and lists;
    everything else, bools included, stays as it is, and the input is not modified.

    The one normalisation of tool arguments: parse_arguments applies it to what the model wrote, call_key to
    the loop's call identity and net/faults.py's search_key to the search's draw key, so 3 and 3.0 are one
    call wherever arguments are keyed.
    """
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: normalize_arguments(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_arguments(v) for v in value]
    return value


_integral_floats_to_int = normalize_arguments  # the former private name, kept so existing callers still work


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments_json: str  # exactly as the model produced it
    arguments: dict | None  # parsed; None when arguments_json is not a JSON object

    @classmethod
    def from_api(cls, call: dict) -> ToolCall:
        fn = call.get("function") or {}
        raw = fn.get("arguments") or "{}"
        if not isinstance(raw, str):  # some servers already return a dict
            raw = json.dumps(raw, ensure_ascii=False)
        return cls(id=call.get("id") or "", name=fn.get("name") or "", arguments_json=raw,
                   arguments=parse_arguments(raw))

    def to_api(self) -> dict:
        return {"id": self.id, "type": "function",
                "function": {"name": self.name, "arguments": self.arguments_json}}


def call_key(call: ToolCall) -> str:
    """Identity of a call for dead-loop detection: name plus canonical arguments, defaults filled in."""
    if call.arguments is None:
        return f"{call.name}:{call.arguments_json}"
    arguments = normalize_arguments({**DEFAULTS.get(call.name, {}), **call.arguments})
    return f"{call.name}:{canonical_json(arguments)}"


@dataclass(frozen=True)
class ToolOutcome:
    """What a tool adapter returns: JSON-serialisable content and an error code (None on success)."""

    content: dict | list
    error: str | None = None


@dataclass(frozen=True)
class ToolResult:
    call_id: str
    name: str
    content: dict | list
    error: str | None = None

    def to_api(self) -> dict:
        body = self.content if self.error is None else {"error": self.error, **_as_dict(self.content)}
        return {"role": "tool", "tool_call_id": self.call_id, "content": canonical_json(body)}


def _as_dict(content: dict | list) -> dict:
    return content if isinstance(content, dict) else {"items": content}


@dataclass(frozen=True)
class StepIssue:
    """A reason the executor rejects a step before running it; event is one of tools.schemas.EVENTS."""

    call_id: str  # "" when the issue concerns the whole step (too many calls)
    event: str
    messages: list[str]


@dataclass
class AssistantTurn:
    content: str
    tool_calls: list[ToolCall]
    model: str
    usage: dict = field(default_factory=dict)
    paid: bool = False  # a paid API call was made for this turn
    cached: bool = False  # served from the response cache; a local-model turn is neither paid nor cached
    # DeepSeek's thinking mode the cloud turn ran in (llm.cloud.THINKING or NON_THINKING); None for every
    # other model. Like the reasoning, it never enters the API form, so histories and transcripts are as
    # they were.
    mode: str | None = None
    # The confidence signals of a local step (agent.route.step_signals: min_decision_prob, mean_logprob,
    # hard_reason), set by LocalModel; None for every other model. route is set by agent.route.RoutedModel:
    # which model answered the step, both signal values and why it was routed. Neither enters the API form
    # nor a turn's equality, so histories, transcripts and the records of the runners stay as they were.
    signals: dict | None = field(default=None, compare=False)
    route: dict | None = field(default=None, compare=False)
    # The reasoning a local model wrote before its answer and how it was generated (llm.generation, set by
    # LocalModel.for_family: {"mode", "reasoning_tokens", "answer_tokens", "forced", "phases", "seeds"});
    # None for a turn generated without either, so the step records and the runners' records of such turns
    # stay as they were. Like the signals, neither enters the API form (earlier reasoning is never shown to
    # the model, docs/m7-计划.md §2 思考) nor a turn's equality.
    reasoning: str | None = field(default=None, compare=False)
    thinking: dict | None = field(default=None, compare=False)
    # What the inference wrappers of M7 did to the turn (docs/m7-本地提升计划.md WP2, WP3): the vote of a
    # voted step (agent.vote.VotedModel: the keys, the counts, the chosen class and its representative) and
    # the changes the runtime rules made ([{"rule", "before", "after"}], agent.rules.RuledModel; [] when they
    # changed nothing). None for a turn of a system without them, so its records and step records stay as they
    # were; like the signals, neither enters the API form nor a turn's equality.
    votes: dict | None = field(default=None, compare=False)
    rules: list | None = field(default=None, compare=False)

    def to_api(self) -> dict:
        message: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = [c.to_api() for c in self.tool_calls]
        return message


@dataclass
class Transcript:
    request: str
    today: str
    status: str
    final_text: str | None
    messages: list[dict]  # the accepted history in API format
    steps: list[dict]  # per step: attempts (rejected ones included), executed results, decisions
    schema_hash: str
    prompt_version: str
    prompt_hash: str
    paid_calls: int = 0
    cache_hits: int = 0
    local_calls: int = 0
    cost_rmb: float = 0.0
    usage: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)


# Signatures of the owner's modules (first assigned to the owner in the plan, implemented by Claude Code;
# tests use scripted stand-ins).
# check_call(tool, args, seen_ids, today) -> errors
CheckCall = Callable[[str, dict, list[str], str], list[str]]
Decide = Callable[[str, int], str]  # decide(event, times) -> action
ToolFn = Callable[[dict], ToolOutcome]  # an adapted tool: args -> outcome


def api_to_hf(messages: list[dict]) -> list[dict]:
    """API messages -> HF template messages: tool-call arguments become dicts, ids are dropped.

    Arguments that are not a JSON object stay as the raw string, which the Qwen3 template prints verbatim,
    so a repair request shows the local model exactly what it wrote.
    """
    out = []
    for m in messages:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                raw = c["function"]["arguments"]
                parsed = parse_arguments(raw) if isinstance(raw, str) else raw
                calls.append({"type": "function", "function": {
                    "name": c["function"]["name"], "arguments": raw if parsed is None else parsed}})
            out.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
        elif m.get("role") == "tool":
            out.append({"role": "tool", "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out

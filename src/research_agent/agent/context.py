"""What the model sees at each step: the latest step in full, older tool outputs reduced to a few keys.

build_context is the one place that decides this. The loop uses it for every model request, and M2 cuts
transcripts into per-step training samples with step_contexts, which is built on it, so the teacher (cloud
model) and the student (local model) see the same context, with one exception: a step accepted after a
repair was answered from a repair request that also showed the model the rejected step and its errors. That
exchange never enters the history, only Transcript.steps[i]["attempts"], so its sample lacks it, and M2 must
drop or mark those samples (see step_contexts). fits() checks the CONTEXT_TOKENS budget: the rendered prompt
with the tool schemas included, plus a reserve for the reply.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path

from ..canonical import canonical_json
from ..family.core import LoadedFamily, Mode, load_family
from ..tools.schemas import CONTEXT_TOKENS, REPLY_RESERVE_TOKENS

KEEP_KEYS = ("arxiv_id", "title", "note_id", "section", "error")
_NON_JSON = {"note": "non-JSON tool output omitted"}

TokenCounter = Callable[[list[dict], list[dict]], int]  # count_tokens(messages, tools) -> tokens


def _keep(value: object) -> object:
    """value with only KEEP_KEYS left in every dict; dicts left empty are dropped from lists."""
    if isinstance(value, dict):
        return {k: _keep(v) for k, v in value.items() if k in KEEP_KEYS}
    if isinstance(value, list):
        kept = (_keep(v) for v in value)
        return [v for v in kept if not (isinstance(v, dict) and not v)]
    return value


def compress_tool_content(content: str) -> str:
    """An older tool output reduced to ids, titles, sections and error codes, as canonical JSON.

    Text that is not JSON becomes a note instead of an empty object, so the model can still tell that the call
    returned something. Compressing an already compressed output changes nothing.
    """
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return canonical_json(_NON_JSON)
    if value == _NON_JSON:  # "note" is not a kept key: filtering the note again would erase it
        return canonical_json(value)
    return canonical_json(_keep(value))


def _latest_step(messages: list[dict]) -> int:
    """Index of the last assistant message with tool calls; -1 when there is none."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant" and messages[i].get("tool_calls"):
            return i
    return -1


def build_context(messages: list[dict]) -> list[dict]:
    """API messages -> the context the model sees.

    The latest step (the last assistant message with tool calls and the tool messages after it, parallel
    results included) stays exactly as it is; every tool message before it is compressed; all other messages
    are copied unchanged. The input is never mutated, changing the result never changes the input, and
    applying build_context to its own result gives the same result.
    """
    start = _latest_step(messages)
    context = []
    for i, message in enumerate(messages):
        copied = copy.deepcopy(message)
        if i < start and copied.get("role") == "tool":
            copied["content"] = compress_tool_content(copied["content"])
        context.append(copied)
    return context


def step_contexts(messages: list[dict]) -> list[tuple[list[dict], dict]]:
    """(context, assistant message) for every assistant message: the per-step samples M2 trains on.

    Each context is build_context of the history before that message. That is what the loop showed the model
    when it produced the message, except for a step accepted after a repair: the model answered a repair
    request, which also carried the rejected step and its errors. Those live only in the step record, never
    in the history, so such a sample's context is shorter than what the model saw, and M2 must drop or mark
    it. The i-th sample belongs to Transcript.steps[i] (only a turn's last step can end without an accepted
    message); it is such a sample when the attempt before the accepted one, the last in
    steps[i]["attempts"], has the action "repair". After a handoff the model saw the original context.
    """
    return [(build_context(messages[:i]), copy.deepcopy(m)) for i, m in enumerate(messages)
            if m.get("role") == "assistant"]


def fits(messages: list[dict], tools: list[dict], count_tokens: TokenCounter, *, limit: int = CONTEXT_TOKENS,
         reserve: int = REPLY_RESERVE_TOKENS) -> bool:
    """True when count_tokens(messages, tools) plus the reply reserve stays within limit."""
    return count_tokens(messages, tools) + reserve <= limit


def family_token_counter(family: LoadedFamily, mode: Mode) -> TokenCounter:
    """count_tokens over a family's prompt in a mode: the ids the local model is sent (family.encode_prompt of
    family.prompt, the BOS rule included), so the context check measures what the model sees."""

    def count_tokens(messages: list[dict], tools: list[dict]) -> int:
        return len(family.encode_prompt(family.prompt(family.to_template_messages(messages), tools, mode)))

    return count_tokens


def qwen_token_counter(model_dir: str | Path) -> TokenCounter:
    """count_tokens with the model folder's own chat template, thinking off: family_token_counter of its
    family (research_agent.family.load_family) in NO_THINK mode, which counts what this counted before M7,
    the prompt as training and inference render it without a special token added."""
    return family_token_counter(load_family(model_dir), Mode.NO_THINK)

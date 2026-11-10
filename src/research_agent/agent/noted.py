"""The glossary note at inference (docs/m7-本地提升计划.md WP4): NotedModel sends the wrapped model each
context with the note of research_agent.glossary on its system message, noted_messages(context, glossary),
the very function the training renderer calls on a step's context (research_agent.train.samples: the
glossary-note option), so a step's prompt is the prompt a sample of the same history trained on. The note
reads the latest user message, which build_context leaves as it is, so the note of a context is that of its
whole history.

Only the request carries the note. The context the wrapper is given, the history the loop and the runner keep
(and so the transcripts and records), and what the runtime rules and the vote read (agent.loop.
current_history: agent.rules.RuledModel and agent.vote.VotedModel sit outside this wrapper) stay without it,
so notes never pile up over the steps of a task. noted_counter measures the context the model is sent: the
note's tokens count toward the context limit, as they count toward the training length.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..glossary import Glossary, noted_messages
from .context import TokenCounter
from .loop import Model, begin_task
from .types import AssistantTurn


def _glossary(glossary: Mapping[str, str]) -> Glossary:
    return glossary if isinstance(glossary, Glossary) else Glossary(glossary)


class NotedModel:
    """The wrapped model, sent every context with the glossary note on its system message (module
    docstring)."""

    def __init__(self, model: Model, glossary: Mapping[str, str]) -> None:
        self.model, self.glossary = model, _glossary(glossary)

    @property
    def name(self) -> str:
        return f"{self.model.name}+glossary"

    def begin_task(self) -> None:
        begin_task(self.model)

    def next_turn(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        noted, _note = noted_messages(messages, self.glossary)
        return self.model.next_turn(noted, tools)

    def cost(self, turn: AssistantTurn) -> float:
        cost = getattr(self.model, "cost", None)
        return 0.0 if cost is None else cost(turn)


def noted_counter(count_tokens: TokenCounter, glossary: Mapping[str, str]) -> TokenCounter:
    """count_tokens over the contexts as a NotedModel sends them."""
    glossary = _glossary(glossary)
    return lambda messages, tools: count_tokens(noted_messages(messages, glossary)[0], tools)

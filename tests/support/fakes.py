"""Scripted stand-ins shared by the M1 tests.

They are lookup tables, not working versions of the owner's modules: check_call and decide answer from a table
given by the test, and tools replay scripted outcomes.
"""

from __future__ import annotations

import copy
import json

from research_agent.agent.types import AssistantTurn, ToolCall, ToolOutcome, parse_arguments
from research_agent.canonical import canonical_json


def call(name: str, call_id: str | None = None, raw: str | None = None, **args) -> ToolCall:
    """A tool call as a model would emit it; raw overrides the JSON text (e.g. to make it invalid)."""
    text = raw if raw is not None else json.dumps(args, ensure_ascii=False)
    return ToolCall(id=call_id or f"call_{name}", name=name, arguments_json=text,
                    arguments=parse_arguments(text))


def turn(*calls: ToolCall, content: str = "", model: str = "fake", paid: bool = True, cached: bool = False,
         usage: dict | None = None, mode: str | None = None) -> AssistantTurn:
    return AssistantTurn(content=content, tool_calls=list(calls), model=model, paid=paid, cached=cached,
                         usage=usage or {"prompt_tokens": 100, "completion_tokens": 10}, mode=mode)


class ScriptedModel:
    """Model adapter stand-in: returns scripted turns in order and records every context it was given."""

    def __init__(self, turns: list, name: str = "fake") -> None:
        self.turns, self.name, self.contexts = list(turns), name, []

    def next_turn(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        self.contexts.append(copy.deepcopy(messages))
        if not self.turns:
            raise AssertionError("ScriptedModel ran out of scripted turns")
        item = self.turns.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def table_check(table: dict[str, list[str]] | None = None):
    """check_call stand-in: errors looked up by "tool:canonical args" or by tool name; [] when absent."""
    table = table or {}
    calls: list[tuple] = []

    def check_call(tool: str, args: dict, seen_ids: list[str], today: str) -> list[str]:
        calls.append((tool, args, list(seen_ids), today))
        return list(table.get(f"{tool}:{canonical_json(args)}", table.get(tool, [])))

    check_call.calls = calls
    return check_call


def table_decide(table: dict[tuple[str, int], str] | None = None, default: str = "stop"):
    """decide stand-in: (event, times) -> action from the table, default otherwise; records every question."""
    asked: list[tuple[str, int]] = []

    def decide(event: str, times: int) -> str:
        asked.append((event, times))
        return (table or {}).get((event, times), default)

    decide.asked = asked
    return decide


def table_tools(script: dict[str, list[ToolOutcome]]):
    """Adapted-tool stand-ins: each tool returns its scripted outcomes in order; the last one repeats."""
    seen: dict[str, list[dict]] = {name: [] for name in script}

    def make(name: str):
        outcomes = list(script[name])

        def run(args: dict) -> ToolOutcome:
            seen[name].append(args)
            return outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]

        return run

    tools = {name: make(name) for name in script}
    return tools, seen


HITS = [
    {"arxiv_id": "2609.01234", "title": "Adaptive Draft Length for Speculative Decoding",
     "published": "2026-09-15",
     "snippet": "We study speculative decoding with a draft model whose length adapts to acceptance rates."},
    {"arxiv_id": "2609.05678", "title": "Tree Verification at Scale", "published": "2026-09-20",
     "snippet": "Tree-structured verification of drafted tokens in large serving systems."},
]
PAPER = {"arxiv_id": "2609.01234", "title": "Adaptive Draft Length for Speculative Decoding",
         "authors": ["A. Author", "B. Author"], "published": "2026-09-15",
         "abstract": "Speculative decoding drafts tokens with a small model and verifies them with the "
                     "target model."}
_SEARCH = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-01")
_GET_FIRST = call("get_paper", "c2", arxiv_id="2609.01234")
_GET_SECOND = call("get_paper", "c3", arxiv_id="2609.05678")
_SECOND_PAPER = {**PAPER, "arxiv_id": "2609.05678", "title": "Tree Verification at Scale"}

# A two-step history in API format: a search, then two parallel get_paper calls.
SAMPLE_MESSAGES = [
    {"role": "system", "content": "system prompt"},
    {"role": "user", "content": "帮我找 2026 年 9 月关于 speculative decoding 的论文"},
    {"role": "assistant", "content": "", "tool_calls": [_SEARCH.to_api()]},
    {"role": "tool", "tool_call_id": "c1", "content": canonical_json(HITS)},
    {"role": "assistant", "content": "", "tool_calls": [_GET_FIRST.to_api(), _GET_SECOND.to_api()]},
    {"role": "tool", "tool_call_id": "c2", "content": canonical_json(PAPER)},
    {"role": "tool", "tool_call_id": "c3", "content": canonical_json(_SECOND_PAPER)},
]

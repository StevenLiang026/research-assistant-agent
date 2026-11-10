"""The glossary note at inference (research_agent.agent.noted; docs/m7-本地提升计划.md WP4): NotedModel sends
the wrapped model glossary.noted_messages of each context, the function the training renderer calls; the
history the loop and the runner keep, and the history the rules and the vote read, stay without the note."""

from __future__ import annotations

from support.fakes import ScriptedModel, call, table_check, table_decide, table_tools, turn

from research_agent.agent.context import build_context
from research_agent.agent.executor import ToolExecutor
from research_agent.agent.loop import current_history, run_turn
from research_agent.agent.noted import NotedModel, noted_counter
from research_agent.agent.prompts import system_prompt
from research_agent.agent.rules import RuledModel
from research_agent.agent.types import ToolOutcome
from research_agent.eval.runner import run_items
from research_agent.glossary import Glossary, noted_messages

TODAY = "2026-10-10"
SYSTEM = {"role": "system", "content": system_prompt(TODAY)}
GLOSSARY = Glossary({"混合专家": "mixture of experts", "幻觉": "hallucination"})
ZH = {"role": "user", "content": "找几篇混合专家路由的论文"}
EN = {"role": "user", "content": "Find mixture-of-experts papers."}
NOTE = "术语对照:混合专家 = mixture of experts"


def test_the_wrapped_model_is_sent_the_context_with_the_note_on_its_system_message():
    inner = ScriptedModel([turn(content="ok")], name="m")
    model = NotedModel(inner, GLOSSARY)
    model.next_turn([SYSTEM, ZH], [])
    (sent,) = inner.contexts
    assert sent == noted_messages([SYSTEM, ZH], GLOSSARY)[0]
    assert sent[0]["content"] == f"{SYSTEM['content']}\n{NOTE}" and sent[1] == ZH
    assert model.name == "m+glossary"


def test_a_context_without_a_glossary_term_is_sent_as_it_is():
    inner = ScriptedModel([turn(content="ok")])
    NotedModel(inner, {"混合专家": "mixture of experts"}).next_turn([SYSTEM, EN], [])
    assert inner.contexts == [[SYSTEM, EN]]


def test_the_loop_keeps_its_history_without_the_note_and_the_rules_read_that_history():
    tools, _ = table_tools({"search_arxiv": [ToolOutcome([{"arxiv_id": "2609.00001", "title": "MoE"}])]})
    executor = ToolExecutor(tools, table_check(), today=TODAY)
    search = turn(call("search_arxiv", "c1", query="mixture of experts"), paid=False)
    histories = []

    class Spy:
        name = "spy"

        def __init__(self) -> None:
            self.inner = ScriptedModel([search, turn(content="Found one.", paid=False)])

        def next_turn(self, messages, tools):
            histories.append([dict(m) for m in current_history()])
            return self.inner.next_turn(messages, tools)

    spy = Spy()
    transcript = run_turn(ZH["content"], model=RuledModel(NotedModel(spy, GLOSSARY)), executor=executor,
                          decide=table_decide(), today=TODAY)
    assert transcript.status == "answered"
    assert transcript.messages[0] == {"role": "system", "content": system_prompt(TODAY)}  # no note stored
    assert all(context[0]["content"].endswith(NOTE) for context in spy.inner.contexts)
    assert [len(h) for h in histories] == [2, 4] and all(NOTE not in str(h) for h in histories)
    assert spy.inner.contexts[1] == noted_messages(build_context(transcript.messages[:4]), GLOSSARY)[0]


def test_the_runner_records_are_those_of_the_model_without_the_note():
    item = {"id": "q-1", "messages": [SYSTEM, ZH], "expected": {}, "type": "call"}
    noted = run_items([item], NotedModel(ScriptedModel([turn(content="ok")]), GLOSSARY))
    plain = run_items([item], ScriptedModel([turn(content="ok")]))
    assert noted == plain


def test_the_noted_counter_counts_the_context_the_model_is_sent():
    counted = []

    def count(messages, tools):
        counted.append(messages)
        return len(messages[0]["content"])

    counter = noted_counter(count, GLOSSARY)
    assert counter([SYSTEM, ZH], []) == len(SYSTEM["content"]) + 1 + len(NOTE)
    assert counter([SYSTEM, EN], []) == len(SYSTEM["content"])
    assert counted[0] == noted_messages([SYSTEM, ZH], GLOSSARY)[0]

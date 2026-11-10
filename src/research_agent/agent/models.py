"""Model adapters: the cloud model and the local model both answer a context with an AssistantTurn.

CloudModel wraps CloudClient (DeepSeek with the response cache and the spending ledger); a turn is either
paid or served from the cache (cached), and records the thinking mode it ran in. CloudModel remembers the
reasoning DeepSeek gave with each of its turns, so a later request whose history holds them can pass it back,
as DeepSeek's thinking mode requires (research_agent.llm.cloud). LocalModel answers on llama-server (or, for
the M3 comparison, on transformers) in one of two ways. Built for a family (LocalModel.for_family,
docs/m7-家族层设计.md §7), it renders and encodes the context with the family (research_agent.family: the
prompt of the generation mode, sent as token ids), generates by the two-phase procedure of
research_agent.llm.generation and reads the answer's calls with the family's tool format; its turns carry the
reasoning and how it was generated (AssistantTurn.reasoning and .thinking) when the generation thought or
sampled, and their signals are the answer's alone. Built the way it was before M7 (the constructor), it
renders the context in HF message format with the injected render, completes the text and reads the tool
calls with parse_tool_calls (research_agent.toolcalls, the default; a stand-in may be injected). Either way
its turns are neither paid nor cached and have no mode; they carry the step's confidence signals
(agent.route), which RoutedModel routes on and the runner records on request. Both adapters give every call
of a turn an id of its own, which the loop needs to answer each call.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Callable
from typing import Protocol

from ..canonical import canonical_json
from ..family.core import LoadedFamily
from ..llm.budget import cost_rmb
from ..llm.cloud import CloudClient
from ..llm.generation import GenerationSettings, generate
from ..llm.local import Completion
from ..toolcalls import parse_tool_calls as qwen3_tool_calls
from .route import completion_signals
from .types import AssistantTurn, HarnessError, ToolCall, api_to_hf, parse_arguments

STOP = ["<|im_end|>"]  # Qwen3's end of turn: generation stops with the assistant message

# Signatures of the injected functions.
Render = Callable[[list[dict], list[dict] | None], str]  # render(hf_messages, tools) -> prompt
ParseToolCalls = Callable[[str], list[dict]]  # the owner's parse_tool_calls(text) -> [{"name", "arguments"}]


class SupportsComplete(Protocol):
    """research_agent.llm.local.LocalClient, research_agent.llm.hf.HFClient, or a test double. The family
    path sends the prompt as ids with the decoding fields of llm.generation, which LocalClient alone takes."""

    def complete(self, prompt: str | list[int], *, lora_scale: float | None = None,
                 stop: list[str] | None = None) -> Completion: ...


class CloudModel:
    """DeepSeek through CloudClient, which brings the response cache, the spending ledger and the retries.

    reasoning maps the API form (canonical JSON of AssistantTurn.to_api) of every turn this instance was
    given, paid or served from the cache, to the non-empty reasoning DeepSeek gave with it; the client passes
    it back with the earlier turns of a later request (reasoning_of). Of two answers with the same form the
    later one's reasoning is kept, so in the loop a turn always goes back with its own. modes counts the turns
    per mode.
    """

    def __init__(self, client: CloudClient, *, temperature: float = 0.0) -> None:
        self.client, self.temperature = client, temperature
        self.reasoning: dict[str, str] = {}
        self.modes: Counter[str] = Counter()

    @property
    def name(self) -> str:
        return self.client.model

    def next_turn(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        """One chat request. BudgetExceeded from the ledger propagates; the loop records budget_stop.

        paid and cached come from the client's counters of paid calls and cache hits, mode from its last_mode.
        """
        paid_before, hits_before = self.client.calls, self.client.cache_hits
        data = self.client.chat(messages, tools=tools, reasoning_of=self.reasoning_of,
                                temperature=self.temperature)
        message = data["choices"][0]["message"]
        calls = _distinct_ids([ToolCall.from_api(c) for c in message.get("tool_calls") or []])
        turn = AssistantTurn(content=message.get("content") or "", tool_calls=calls,
                             model=data.get("model") or self.client.model,
                             usage=dict(data.get("usage") or {}), paid=self.client.calls > paid_before,
                             cached=self.client.cache_hits > hits_before, mode=self.client.last_mode)
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning:  # keyed with the ids as the history will hold them
            self.reasoning[canonical_json(turn.to_api())] = reasoning
        self.modes[turn.mode] += 1
        return turn

    def reasoning_of(self, message: dict) -> str | None:
        """The reasoning given with an assistant message of exactly this API form; None when unknown."""
        return self.reasoning.get(canonical_json(message))

    def cost(self, turn: AssistantTurn) -> float:
        """RMB at the price-list upper bound for a paid turn; 0.0 for a turn served from the cache."""
        if not turn.paid:
            return 0.0
        return cost_rmb(self.client.budget.prices[self.client.model], turn.usage)


class LocalModel:
    """A local model on llama-server (LocalClient) or, for the M3 comparison with the quantized model, in full
    precision on transformers (llm.hf.HFClient); lora_scale 0.0 is the base model and 1.0 the fine-tuned one.
    The constructor builds the model as it was before M7 (module docstring), for_family one over a family.

    Never paid and never cached: the loop counts its turns as local calls.
    """

    def __init__(self, client: SupportsComplete, render: Render,
                 parse_tool_calls: ParseToolCalls = qwen3_tool_calls, *, lora_scale: float,
                 name: str | None = None) -> None:
        self.client, self.render, self.parse_tool_calls = client, render, parse_tool_calls
        self.lora_scale = lora_scale
        self._name = name or f"local:lora={lora_scale:g}"
        self.family: LoadedFamily | None = None  # set by for_family: the family path answers
        self.settings: GenerationSettings | None = None

    @classmethod
    def for_family(cls, client: SupportsComplete, family: LoadedFamily, settings: GenerationSettings, *,
                   lora_scale: float, name: str | None = None) -> LocalModel:
        """The model over a loaded family (research_agent.family.load_family) generating under settings
        (llm.generation.GenerationSettings): the family renders, encodes and reads, the generation runs its
        phases (module docstring)."""
        model = cls(client, _no_render, lora_scale=lora_scale, name=name)
        model.family, model.settings = family, settings
        return model

    @property
    def name(self) -> str:
        return self._name

    def next_turn(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        """Call ids are local_0, local_1, ... within the step; the text is the content only without calls.

        The turn carries the step's confidence signals (agent.route.completion_signals: over the tokens of the
        answer completion, with the step's tools and the context's date and seen ids), outside its API form.
        On the family path it also carries the reasoning and the thinking record of a generation that thought
        or sampled (agent.types.AssistantTurn); a greedy NO_THINK generation leaves both None, so its turn is
        the one the model gave before M7. The record's reasoning_tokens and answer_tokens are the tokens the
        model generated (llm.generation.Generation), not the ones the server listed with log-probabilities,
        which leave out the parts of a character split across tokens: the prompt's ids, the reasoning tokens,
        a forced close's ids and the answer tokens are the step's whole sequence (scripts/run_e2e.py). The
        answer's count is None for a sampled NO_THINK step whose server sent no usage.
        """
        if self.family is None:
            prompt = self.render(api_to_hf(messages), tools)
            completion = self.client.complete(prompt, lora_scale=self.lora_scale, stop=list(STOP))
            calls = [_local_call(i, parsed) for i, parsed in enumerate(self._parse(completion.text))]
            signals = completion_signals(completion, tools, messages, parse=self.parse_tool_calls)
            return AssistantTurn(content="" if calls else completion.text, tool_calls=calls, model=self.name,
                                 usage=dict(completion.raw.get("usage") or {}), paid=False, signals=signals)
        family, settings = self.family, self.settings
        prompt_ids = family.encode_prompt(family.prompt(family.to_template_messages(messages), tools,
                                                        settings.mode))
        gen = generate(self.client, family, prompt_ids, settings, lora_scale=self.lora_scale)

        def parse(text: str) -> list[dict]:
            return family.parse(text, tools)  # the step's tools type the values of a text format

        calls = [_local_call(i, parsed) for i, parsed in enumerate(self._parse(gen.answer_text, parse))]
        signals = completion_signals(gen.answer, tools, messages, parse=parse, locate=family.locate)
        thinking = None
        if gen.reasoning is not None or gen.seeds is not None:  # the answer's tokens as generated, not listed
            thinking = {"mode": str(settings.mode), "reasoning_tokens": gen.reasoning_tokens,
                        "answer_tokens": gen.answer_tokens, "forced": gen.forced, "phases": gen.phases,
                        "seeds": gen.seeds}
        return AssistantTurn(content="" if calls else gen.answer_text, tool_calls=calls, model=self.name,
                             usage=dict(gen.usage), paid=False, signals=signals, reasoning=gen.reasoning,
                             thinking=thinking)

    def cost(self, turn: AssistantTurn) -> float:
        return 0.0

    def _parse(self, text: str, parse: ParseToolCalls | None = None) -> list[dict]:
        """The owner's parse_tool_calls (or parse, the family format's); its failures are HarnessErrors, like
        check_call's in the executor."""
        try:
            parsed = (self.parse_tool_calls if parse is None else parse)(text)
        except HarnessError:
            raise
        except Exception as exc:
            raise HarnessError(f"parse_tool_calls failed: {exc!r}") from exc
        if not isinstance(parsed, list) or not all(isinstance(p, dict) for p in parsed):
            raise HarnessError(f"parse_tool_calls must return a list of dicts, got {parsed!r:.200}")
        return parsed


def _no_render(messages: list[dict], tools: list[dict] | None) -> str:
    """The render of a model built for a family, which never renders text: the family encodes the prompt."""
    raise HarnessError("a LocalModel built for a family sends token ids, not rendered text")


def _distinct_ids(calls: list[ToolCall]) -> list[ToolCall]:
    """calls with every empty or repeated id replaced by call_<index>, index being the call's position.

    The first call with a given id keeps it, as the server sent it. An empty id would also be taken for the
    whole-step sentinel of StepIssue. In the unlikely case that a real id of the turn is already
    call_<index>, underscores are appended until the new id is free.
    """
    taken = {c.id for c in calls if c.id}
    kept: set[str] = set()
    out = []
    for index, c in enumerate(calls):
        if c.id and c.id not in kept:
            kept.add(c.id)
        else:
            new_id = f"call_{index}"
            while new_id in taken:
                new_id += "_"
            taken.add(new_id)
            c = dataclasses.replace(c, id=new_id)
        out.append(c)
    return out


def _local_call(index: int, parsed: dict) -> ToolCall:
    """A parsed call as a ToolCall; arguments_json is canonical JSON whenever the arguments are an object.

    Arguments given as text are parsed here. Text that is not a JSON object is kept as the model wrote it, so
    it reaches the executor as bad_json (and can be repaired or handed off) instead of being lost.
    """
    arguments = parsed.get("arguments")
    if isinstance(arguments, str):
        value = parse_arguments(arguments)
        raw = arguments if value is None else canonical_json(value)
    else:
        try:
            raw = canonical_json({} if arguments is None else arguments)
        except (TypeError, ValueError) as exc:  # a set or another value JSON cannot hold
            raise HarnessError(f"parse_tool_calls returned arguments that JSON cannot hold: {exc!r}") from exc
    return ToolCall(id=f"local_{index}", name=str(parsed.get("name") or ""), arguments_json=raw,
                    arguments=parse_arguments(raw))

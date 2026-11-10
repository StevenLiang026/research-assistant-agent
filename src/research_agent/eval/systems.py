"""The systems an evaluation compares (docs/plan.md, 评测: 对照系统), built the same way for the single-step
runner (scripts/run_eval.py) and the end-to-end runner (scripts/run_e2e.py).

    b0      Qwen3-1.7B base on llama-server: LocalModel with lora scale 0.0
    b1      b0 plus three fixed demonstrations (below)
    b2      deepseek-flash (CloudModel)
    b3      deepseek-v4-pro (CloudModel; the upper bound)
    ft      the fine-tuned adapter on the same server: LocalModel with lora scale 1.0
    b4      the B4 adapter (trained on the public data only), served the same way: LocalModel with lora
            scale 1.0
    ftr     ft routed to a cloud system (agent.route.RoutedModel) under the frozen routing configuration,
            which names it (below): b2 in every frozen configuration, M7's R3 included
    replay  a scripted stand-in that answers each single-step item with its expected turn (ReplayModel), for a
            smoke run of the scoring pipeline that costs nothing; it has no meaning end to end

ft and b4 differ only in the adapter the server loaded, so their records are alike (model
"qwen3-1.7b:lora=1"); the run manifest tells them apart by its system and by the adapters below. A local
record's model is the tokenizer folder's name in lower case and the lora scale (local_name: models/Qwen3-1.7B
gives qwen3-1.7b, models/Qwen3-4B qwen3-4b), LOCAL_NAME when no folder is given (a test's render). The local
systems decode greedily (temperature 0) with thinking off. The cloud systems run in DeepSeek's thinking mode,
which ignores temperature, except on histories with an earlier turn whose reasoning is unknown (below), so b2
and b3, like the M2 teacher, are not greedy decoders.

Local systems. On llama-server a local system runs the family path (M7, docs/m7-家族层设计.md §7): the
tokenizer folder's family (research_agent.family.load_family, or the family given) renders the context the
runner or the loop built with MODEL_TOOLS in the generation mode, encodes it, and the prompt is sent as those
ids (LocalModel.for_family, llm.generation); the calls are read with the family's tool format. The
generation settings (generation, llm.generation.GenerationSettings: the mode, the decoding, the reasoning
budget, the base seed) default to greedy NO_THINK, under which b0, b1, ft, b4 and ftr behave as they did
before M7, the prompt's ids in place of its text: the family's NO_THINK prompt is the training-time
rendering, train.samples.prompt_renderer's very template call. The info of such a system holds the family's
record ("family") and the generation settings ("generation"). With a render injected (the tests, and the
phase-4 runner until it moves) or on the hf backend, the system is the one before M7: the render's text is
completed, the calls read with research_agent.toolcalls.parse_tool_calls (LocalModel's default); a thinking
or sampling generation is refused there (ValueError). The lora scale is sent explicitly whenever the server
has an adapter loaded (spike S1b: a request without it gets the adapter); a server without one serves the
base model, which b0 and b1 ask for (scale 0), and LocalClient refuses the scale 1 of ft and b4 there rather
than measure the base model under their names. b0 and b1 run the same on a server with an adapter loaded,
since scale 0 is sent explicitly. The info of a local system, which the run manifest carries, lists the
adapters the server loaded ("adapters": its /lora-adapters entries as id, the path the server was given and
the SHA-256 of that GGUF, [] for a bare server, None for a stand-in client that cannot list them); an adapter
whose file cannot be hashed here makes the system unavailable, since the run could not say what it
measured. The records carry none of it, so a run on a server with an adapter loaded can be compared byte for
byte with one on a bare server.

Backends. A local system runs on llama-server (backend "llama.cpp", the default: the quantized base model
and the GGUF adapter the server loaded) or, for the M3 comparison of the HF model with the quantized one, in
full precision on transformers (backend "hf", research_agent.llm.hf). There HFClient serves under the same
LocalModel, so the prompt and the reading of the completion are the same; the model folder's weights (the
tokenizer folder) are loaded with the PEFT adapter folder given as adapter (a training run's adapter/ or one
of its trainer checkpoint folders), which scale 1 applies and scale 0 disables. ft and b4 need one; b0 and b1
run the base model with or without one. A GGUF given as the adapter, a folder without the PEFT files, or a
model or adapter that cannot be loaded here (a corrupt adapter_model.safetensors among them) makes the system
unavailable. An hf system's info holds "backend": "hf", provider "transformers", dtype, device and the
adapter as {"path" as given, "safetensors_sha256" of its adapter_model.safetensors} (None without one), where
a llama.cpp system's info holds "backend": "llama.cpp", the server and its adapters. The records are alike on
both backends (ft's model over models/Qwen3-1.7B is "qwen3-1.7b:lora=1" on either), so only the manifest
tells them apart. The cloud systems and replay have no backend.

Cloud systems go through CloudClient over the shared response cache and spending ledger in runs/llm/
(cache.sqlite, budget.sqlite), booked under a stage whose cap is the plan's (STAGE_CAPS); --max-cost lowers
the cap of one invocation to what the stage has spent plus that amount, never above the stage cap. A model
without a price in llm/prices.json is refused before any call (SystemUnavailable).

Thinking mode (research_agent.llm.cloud). DeepSeek thinks by default and then ignores temperature: the M2
teacher and the M2 B2 runs all ran in thinking mode, so the temperature 0.0 they sent never made them greedy.
A thinking request with tools must pass back the reasoning of every earlier assistant turn (enforced since
2026-10-15). A request in the cache under the key the client used before that rule (its plain key) is served
as it was, whatever its history. Otherwise a context whose earlier assistant turns, if any, all came from this
model instance with their reasoning (CloudModel.reasoning) is sent in thinking mode with that reasoning passed
back; a context with any other earlier turn (a handwritten item's scripted steps, a drafted item's teacher
turns, a local model's turn before a handoff, a turn whose reasoning came back empty) is sent with thinking
off, where temperature 0.0 applies. Every ok record of a cloud system carries the mode its turn ran in
("thinking" or "non-thinking"; a context_overflow, error, budget_stop or skipped record has no turn and no
mode), and the run manifest counts the turns per mode.

FT+R (ftr; docs/plan.md, 置信度路由). ft on the configured server (lora scale 1) answers each step unless its
signals route it; then the cloud system whose model the configuration names ("cloud_model", CLOUD_MODELS:
deepseek-flash for b2, as M4 to M6 froze it and as M7's R3 does, docs/m7-计划.md WP-K; deepseek-v4-pro for b3
is accepted too), through the same CloudClient, response cache and ledger as that system, answers that step
and the rest of the task (agent.route; only that step under stepwise routing, which the runners turn on:
scripts/run_e2e.py --routing-config, scripts/run_test.py's e2e_routing). The configuration is frozen in
results/routing-m4.json, which
scripts/route_curve.py writes from the development runs: the signal, the threshold, the SHA-256 of the
development set it was chosen on ("dev_sha256", with the set's path, "dev_set"; a configuration chosen on
several sets jointly, M7's dev and dev-v2, has "dev_sets" instead, one {"path", "sha256", ...} per set in
order), the SHA-256 of the FT adapter it was chosen with, the cloud model and, from M6, the rules of the
chosen policy ("rules", agent.route.RouteRules' JSON form, with the policy's name under "chosen_policy"; a
file without them, results/routing-m4.json, routes by the hard triggers and the threshold alone). ftr is
unavailable (SystemUnavailable, naming the mismatch) without that file, when it has no threshold (the
preregistered rule was not met), names a signal that is not one of agent.route.SIGNALS, malformed rules or a
cloud model that is no cloud system's, when the development sets it was chosen on are not those given, as
they are now (load_routing: as many, each with the SHA-256 the file records for it, in its order; dev_set,
data/eval/dev.jsonl by default, or the sets the file names when none is given), or when the server lists no
adapter with its adapter SHA-256. Its info holds ft's server, tokenizer and adapters, "local" and "cloud"
(ft's and the cloud system's own info), "cloud_model" (the model whose prices cost its paid turns) and
"routing": the configuration's path, its SHA-256, the fields above (the development set's SHA-256 as
"dev_sha256", or "dev_sets": the path and SHA-256 of each), its rules and its policy. ftr runs on llama.cpp
only.

The inference wrappers (M7, docs/m7-本地提升计划.md WP2 to WP4; build_system's generation.votes, vote_mode,
rules and glossary). A local system, and ftr's local side inside its routing, can answer by a vote, with the
runtime rules and with the glossary note, which compose as one: the vote (agent.vote.VotedModel, when the
generation settings have votes, K of them) over a greedy candidate and K sampled ones (agent.vote.
voted_settings: sample k with the base seed base_seed + k), each candidate the local model with the note on
its prompts (agent.noted.NotedModel, glossary: the zh-en glossary file) inside the rules (agent.rules.
RuledModel); so the rules rewrite every candidate's answer before the vote keys it, and the rules and the vote
read the history without the note. A vote samples, so it runs on the family path only; the rules and the
note wrap the model before M7 too. The context counter of a noted system counts the note (agent.noted.
noted_counter). The info records each option only when it is used: the votes in "generation"
(GenerationSettings.record), the key mode as "vote_mode", the rules as {"version": agent.rules.RULES_VERSION,
"names"} under "rules" and the glossary as {"path", "sha256", "terms"} under "glossary"; a voted system's
Built also says how to read llama.cpp's timings of a step, summed over its K + 1 requests (timings). Without
them every system is built, recorded and run as before.

B1's demonstrations. Three steps of the assembled training set data/raw/synth/synth.jsonl, fixed by request
id (DEMONSTRATIONS) and all from the train split, so no dev or test material leaks into the prompt: one tool
call (a dated search), one clarification (ask_clarification for a paper the user did not name) and one answer
without a call (a question about a tool parameter). Each demonstration asks for exactly what its request asks
(the search's arguments are the request's topic, start date, count and ranking, and nothing else; the
clarification lists exactly the missing slot), since every b1 prompt carries them: a search that fetched more
papers than the user wanted would teach over-asking. Each is the record's first user message and the first
assistant turn the teacher gave to it, the call written as the family's tool format writes calls
(demonstration_reply; Qwen3's <tool_call> blocks by default and on the family path of a Qwen3 family, the
bytes of the M2 runs), so the block shows the output format the model must produce. They are rendered once
as a short worked-examples block (demonstrations_block) appended to the system message of every context,
identically for every item and every step (with_demonstrations); its SHA-256 prefix is recorded in every run
manifest. The token counter of a b1 run counts the context with the block, so the context limit is checked
on what the model actually sees.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath

from ..agent.context import TokenCounter, build_context
from ..agent.loop import Model
from ..agent.models import CloudModel, LocalModel, SupportsComplete
from ..agent.noted import NotedModel, noted_counter
from ..agent.route import RouteConfig, RoutedModel, RouteRules
from ..agent.rules import RULES, RULES_VERSION, RuledModel
from ..agent.types import AssistantTurn, HarnessError, ToolCall, parse_arguments
from ..agent.vote import FINAL, VOTE_MODES, VotedModel, voted_settings
from ..canonical import canonical_json
from ..family.core import GREEDY, FamilyError, LoadedFamily, Mode, load_family
from ..family.toolformats.qwen_json import QWEN_JSON
from ..glossary import Glossary, GlossaryError, load_glossary
from ..llm.budget import Budget, load_off_peak, load_prices
from ..llm.cache import ResponseCache
from ..llm.cloud import THINKING_MODES, CloudClient, MissingKeyError
from ..llm.generation import GenerationSettings
from ..llm.hf import ADAPTER_WEIGHTS, AdapterError, check_adapter
from .lock import sha256_of

SYSTEMS = ("b0", "b1", "b2", "b3", "ft", "b4", "ftr", "replay")
LOCAL_SCALES = {"b0": 0.0, "b1": 0.0, "ft": 1.0, "b4": 1.0}  # lora scale per local system
LLAMA_CPP, HF = "llama.cpp", "hf"
BACKENDS = (LLAMA_CPP, HF)  # where a local system runs (module docstring)
CLOUD_MODELS = {"b2": "deepseek-flash", "b3": "deepseek-v4-pro"}
# routed system -> its local system; its cloud system is the one whose model its routing configuration names
ROUTED = {"ftr": "ft"}
ROUTING_CONFIG = Path("results/routing-m4.json")  # ftr's frozen configuration (scripts/route_curve.py)
DEV_SET = Path("data/eval/dev.jsonl")  # the development set the configuration was chosen on
# the second family's development set (docs/m7-计划.md WP-I 5), chosen on jointly with DEV_SET from M7
DEV_SET_V2 = Path("data/eval/v2/dev-v2.jsonl")
DevSets = str | Path | Sequence[str | Path] | None  # one development set, several in order, or the file's own
# RMB per stage: the API caps of docs/plan.md (里程碑)
STAGE_CAPS = {"M0": 3.0, "M1": 10.0, "M2": 90.0, "M3": 30.0, "M4": 20.0, "M5": 15.0, "M6": 30.0,
              "M7": 50.0,  # M7: the owner's budget of about 50 yuan (docs/m7-计划.md)
              "M8": 10.0}  # M8: the owner's cap of 10 yuan (docs/m8-计划.md)
CLOUD_THINKING = THINKING_MODES  # a cloud system's thinking setting: auto (the default) or off
DEFAULT_STAGE = "M3"
LOCAL_NAME = "qwen3-1.7b"  # a local record's model name without a tokenizer folder (local_name)
TRAINING_SET = Path("data/raw/synth/synth.jsonl")
REQUESTS_FILE = Path("data/synth/requests.jsonl")
# (request id, kind): the three b1 demonstrations, in the order they are shown
DEMONSTRATIONS = (("req-train-1572", "call"), ("req-train-0513", "clarify"), ("req-train-0261", "no_call"))
ASK = "ask_clarification"
DEMONSTRATIONS_HEADING = "以下三个示例只演示回复的格式(调用工具、追问、直接回答),与当前请求无关:"


class SystemUnavailable(RuntimeError):
    """A system that cannot be built here: no key, no price, no server, no training set."""


# ---- b1: demonstrations -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Demonstration:
    request: str
    kind: str  # call, clarify or no_call
    user: str
    assistant: dict  # the teacher's first turn, API format


def load_demonstrations(training_set: str | Path = TRAINING_SET, requests: str | Path | None = REQUESTS_FILE,
                        chosen: Iterable[tuple[str, str]] = DEMONSTRATIONS) -> list[Demonstration]:
    """The chosen demonstrations from the assembled training set, in the order chosen.

    ValueError when a chosen id is missing, its record is not from the train run, the requests file (when
    given) does not put it in the train split, or its first assistant turn is not of the chosen kind (exactly
    one call that is not ask_clarification; exactly one ask_clarification call; no call and some text).
    """
    chosen = list(chosen)
    wanted = {rid for rid, _ in chosen}
    records: dict[str, dict] = {}
    for line in Path(training_set).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            if record.get("id") in wanted:
                records[record["id"]] = record
    splits = {}
    if requests is not None:
        for line in Path(requests).read_text(encoding="utf-8").splitlines():
            if line.strip():
                request = json.loads(line)
                if request.get("id") in wanted:
                    splits[request["id"]] = request.get("split")
    out = []
    for rid, kind in chosen:
        record = records.get(rid)
        if record is None:
            raise ValueError(f"demonstration {rid} is not in {training_set}")
        if record.get("run") != "train" or (requests is not None and splits.get(rid) != "train"):
            raise ValueError(f"demonstration {rid} is not from the train split; dev and test material must "
                             "not reach the prompt")
        messages = record["messages"]
        user = next(m for m in messages if m.get("role") == "user")
        assistant = next(m for m in messages if m.get("role") == "assistant")
        if _kind(assistant) != kind:
            raise ValueError(f"demonstration {rid} is a {_kind(assistant)} step, not a {kind} step")
        out.append(Demonstration(rid, kind, user["content"], copy.deepcopy(assistant)))
    return out


def _kind(assistant: dict) -> str | None:
    calls = assistant.get("tool_calls") or []
    if not calls:
        return "no_call" if (assistant.get("content") or "").strip() else None
    if len(calls) != 1:
        return None
    return "clarify" if calls[0]["function"]["name"] == ASK else "call"


def demonstration_reply(assistant: dict, render: Callable[[list[dict]], str] = QWEN_JSON.render) -> str:
    """An assistant turn as the model writes it: its text, then its calls as the tool format's render writes
    them (research_agent.family.toolformats.ToolFormat.render; Qwen3's <tool_call> blocks by default)."""
    parts = [assistant.get("content") or ""] if (assistant.get("content") or "").strip() else []
    calls = []
    for call in assistant.get("tool_calls") or []:
        fn = call["function"]
        arguments = parse_arguments(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
        calls.append({"name": fn["name"], "arguments": arguments})
    if calls:
        parts.append(render(calls))
    return "\n".join(parts)


def demonstrations_block(demonstrations: list[Demonstration],
                         render: Callable[[list[dict]], str] = QWEN_JSON.render) -> str:
    """The worked-examples block appended to the system message (module docstring), the calls written by
    render (demonstration_reply); starts with a blank line, so the system prompt itself stays as it is."""
    lines = ["", "", DEMONSTRATIONS_HEADING]
    for number, demo in enumerate(demonstrations, 1):
        lines += [f"示例 {number}", f"用户:{demo.user}",
                  f"助手:{demonstration_reply(demo.assistant, render)}"]
    return "\n".join(lines)


def block_hash(block: str) -> str:
    return hashlib.sha256(block.encode("utf-8")).hexdigest()[:16]


def with_demonstrations(messages: list[dict], block: str) -> list[dict]:
    """A copy of messages with block appended to the leading system message; HarnessError without one."""
    if not messages or messages[0].get("role") != "system":
        raise HarnessError("the context has no leading system message to append the demonstrations to")
    out = copy.deepcopy(messages)
    out[0]["content"] = (out[0].get("content") or "") + block
    return out


class DemonstratedModel:
    """A model that sees every context with the demonstrations block in its system message."""

    def __init__(self, model: Model, block: str) -> None:
        self.model, self.block = model, block

    @property
    def name(self) -> str:
        return f"{self.model.name}+demos"

    def next_turn(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        return self.model.next_turn(with_demonstrations(messages, self.block), tools)

    def cost(self, turn: AssistantTurn) -> float:
        cost = getattr(self.model, "cost", None)
        return 0.0 if cost is None else cost(turn)


def demonstrated_counter(count_tokens: TokenCounter, block: str) -> TokenCounter:
    """count_tokens over the contexts as a DemonstratedModel sends them."""
    return lambda messages, tools: count_tokens(with_demonstrations(messages, block), tools)


# ---- replay -----------------------------------------------------------------------------------------------


class ReplayModel:
    """Answers each single-step item's context with the item's expected turn (scripts/run_eval.py --system
    replay): a scoring pipeline smoke test, 100% success by construction. Contexts are matched by their
    canonical JSON after build_context, as eval.runner builds them; items with the same context are answered
    in item order. HarnessError for a context no item has."""

    name = "replay"

    def __init__(self, items: Iterable[dict]) -> None:
        self.answers: dict[str, list[dict]] = {}
        for item in items:
            key = canonical_json(build_context(item["messages"]))
            self.answers.setdefault(key, []).append(item["expected"])

    def next_turn(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        queue = self.answers.get(canonical_json(messages))
        if not queue:
            raise HarnessError("replay: no item has this context")
        expected = queue.pop(0)
        calls = []
        for index, call in enumerate(expected.get("tool_calls") or []):
            raw = call.get("arguments_json")
            if not isinstance(raw, str):
                raw = canonical_json(call.get("arguments") or {})
            calls.append(ToolCall(id=f"replay_{index}", name=call["name"], arguments_json=raw,
                                  arguments=parse_arguments(raw)))
        return AssistantTurn(content=expected.get("content") or "", tool_calls=calls, model=self.name)

    def cost(self, turn: AssistantTurn) -> float:
        return 0.0


# ---- building a system ------------------------------------------------------------------------------------


@dataclass
class Built:
    """A system ready to run: the model, the context counter to use (None: no context check) and what the
    run manifest records about it."""

    model: Model
    count_tokens: TokenCounter | None
    info: dict = field(default_factory=dict)
    local_client: object | None = None  # the LocalClient of a local system, for /props
    closers: list[Callable[[], None]] = field(default_factory=list)
    # llama.cpp's timings of the latest step when the local client's last ones are not all of them: a voted
    # system's, summed over its K + 1 requests (agent.vote.VotedModel.last_timings); None for every other
    timings: Callable[[], dict | None] | None = None

    def close(self) -> None:
        for close in self.closers:
            close()


def stage_cap(budget_path: Path, stage: str, max_cost: float | None) -> float:
    """The cap of one invocation: the stage's cap, or what the stage has spent plus max_cost when lower."""
    if stage not in STAGE_CAPS:
        raise ValueError(f"unknown stage {stage!r}; one of {sorted(STAGE_CAPS)}")
    cap = STAGE_CAPS[stage]
    if max_cost is None:
        return cap
    probe = Budget(budget_path, stage=stage, stage_cap_rmb=cap)
    try:
        return min(cap, probe.spent(stage) + max_cost)
    finally:
        probe.close()


def build_system(system: str, *, server: str = "http://127.0.0.1:8080", tokenizer: str | Path | None = None,
                 count_tokens: TokenCounter | None = None, llm_dir: str | Path = "runs/llm",
                 stage: str = DEFAULT_STAGE, max_cost: float | None = None,
                 items: list[dict] | None = None, training_set: str | Path = TRAINING_SET,
                 requests: str | Path | None = REQUESTS_FILE, openai_client: object | None = None,
                 local_client: SupportsComplete | None = None,
                 render: Callable[[list[dict], list[dict] | None], str] | None = None,
                 backend: str = LLAMA_CPP, adapter: str | Path | None = None,
                 routing_config: str | Path = ROUTING_CONFIG, dev_set: DevSets = DEV_SET,
                 cloud_thinking: str = "auto", generation: GenerationSettings | None = None,
                 family: LoadedFamily | None = None, rules: bool = False, glossary: str | Path | None = None,
                 vote_mode: str = FINAL) -> Built:
    """The system by name (SYSTEMS). tokenizer is the model folder whose family renders and encodes local
    prompts (family, one already loaded from it, spares loading it again; render overrides both, for tests)
    and, on the hf backend, whose weights are loaded; count_tokens is the context counter, wrapped for b1.
    generation is how a local system generates (module docstring: Local systems; None: greedy NO_THINK).
    backend is where a local system runs (BACKENDS), adapter the hf backend's PEFT adapter folder (module
    docstring: Backends). openai_client and local_client stand in for the real clients in tests, local_client
    for LocalClient or HFClient. SystemUnavailable when the system cannot be built here (module docstring);
    ValueError for a backend a system does not have, an adapter folder given to llama-server, which loads its
    own, or a thinking or sampling generation where only the family path can run one. routing_config and
    dev_set are ftr's frozen configuration and the development set whose SHA-256 it must hold, or the sets in
    the order the configuration was chosen on them, or None for the sets the configuration names (module
    docstring: FT+R; load_routing); the configuration's cloud model chooses ftr's cloud system.
    cloud_thinking is the thinking setting of a cloud system's client, or of a routed system's cloud side
    (CLOUD_THINKING; research_agent.llm.cloud): auto, or off for every request. generation.votes (a vote over
    that many samples), vote_mode (agent.vote.VOTE_MODES, the vote's keys), rules (the runtime rules) and
    glossary (the zh-en glossary file whose note the prompts carry) are the inference wrappers of a local
    system or of ftr's local side (module docstring: The inference wrappers): ValueError for them with
    another system or for an unknown vote mode, SystemUnavailable for a glossary that cannot be read."""
    if cloud_thinking not in CLOUD_THINKING:
        raise ValueError(f"cloud_thinking must be one of {CLOUD_THINKING}, not {cloud_thinking!r}")
    if system not in SYSTEMS:
        raise ValueError(f"unknown system {system!r}; one of {list(SYSTEMS)}")
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; one of {list(BACKENDS)}")
    if vote_mode not in VOTE_MODES:
        raise ValueError(f"unknown vote mode {vote_mode!r}; one of {list(VOTE_MODES)}")
    voting = generation is not None and generation.votes > 0
    if (rules or glossary is not None or voting) and system not in LOCAL_SCALES and system not in ROUTED:
        raise ValueError(f"votes, the runtime rules and the glossary wrap a local system or ftr's local "
                         f"side, not {system}")
    wrappers = {"rules": rules, "glossary": glossary, "vote_mode": vote_mode}
    if backend == HF and system not in LOCAL_SCALES:
        raise ValueError(f"the hf backend runs the local systems only ({', '.join(LOCAL_SCALES)}), "
                         f"not {system}")
    if adapter is not None and backend != HF:
        raise ValueError("an adapter folder is for the hf backend; llama-server loads its own adapter "
                         "(scripts/serve_local.py --adapter)")
    if system == "replay":
        if items is None:
            raise ValueError("the replay system answers single-step items: pass items")
        return Built(ReplayModel(items), count_tokens, {"system": system, "provider": "replay",
                                                        "model": ReplayModel.name, "lora_scale": None})
    if system in ROUTED:
        return _routed(system, routing_config=Path(routing_config), dev_set=dev_set, server=server,
                       tokenizer=tokenizer, count_tokens=count_tokens, llm_dir=Path(llm_dir), stage=stage,
                       max_cost=max_cost, openai_client=openai_client, local_client=local_client,
                       render=render, cloud_thinking=cloud_thinking, generation=generation, family=family,
                       **wrappers)
    if system in CLOUD_MODELS:
        return _cloud(system, llm_dir=Path(llm_dir), stage=stage, max_cost=max_cost,
                      openai_client=openai_client, count_tokens=count_tokens, thinking=cloud_thinking)
    return _local(system, backend=backend, adapter=adapter, server=server, tokenizer=tokenizer,
                  count_tokens=count_tokens, training_set=training_set, requests=requests,
                  local_client=local_client, render=render, generation=generation, family=family, **wrappers)


def _cloud(system: str, *, llm_dir: Path, stage: str, max_cost: float | None, openai_client: object | None,
           count_tokens: TokenCounter | None, thinking: str = "auto") -> Built:
    name = CLOUD_MODELS[system]
    if name not in load_prices():
        raise SystemUnavailable(f"{system}: no price for {name} in research_agent/llm/prices.json; add the "
                                "official price first (the ledger refuses an unpriced model)")
    cap = stage_cap(llm_dir / "budget.sqlite", stage, max_cost)
    # the ledger books what DeepSeek charges: half the list price off peak (research_agent.llm.budget)
    budget = Budget(llm_dir / "budget.sqlite", stage=stage, stage_cap_rmb=cap, off_peak=load_off_peak())
    cache = ResponseCache(llm_dir / "cache.sqlite")
    try:
        client = CloudClient(name, cache=cache, budget=budget, client=openai_client, thinking=thinking)
    except MissingKeyError as exc:
        budget.close()
        cache.close()
        raise SystemUnavailable(f"{system}: {exc}") from None
    # temperature is what is sent; it applies only to the turns run with thinking off (module docstring)
    info = {"system": system, "provider": "deepseek", "model": name, "lora_scale": None, "temperature": 0.0,
            "thinking": thinking, "stage": stage, "stage_cap_rmb": cap,
            "spent_before_rmb": budget.spent(stage)}
    return Built(CloudModel(client), count_tokens, info, closers=[budget.close, cache.close])


def _local(system: str, *, backend: str, adapter: str | Path | None, server: str,
           tokenizer: str | Path | None, count_tokens: TokenCounter | None, training_set: str | Path,
           requests: str | Path | None, local_client: SupportsComplete | None, render: Callable | None,
           generation: GenerationSettings | None, family: LoadedFamily | None, rules: bool = False,
           glossary: str | Path | None = None, vote_mode: str = FINAL) -> Built:
    scale = LOCAL_SCALES[system]
    # Everything that can fail without a server or a loaded model comes first, so that a failure leaves no
    # client open.
    notes = None if glossary is None else _glossary(system, glossary)
    adapter_record = None
    if backend == HF:
        if adapter is None and scale != 0.0:
            raise SystemUnavailable(f"{system}: lora scale {scale:g} on the hf backend needs the PEFT "
                                    "adapter folder (adapter)")
        try:
            adapter_record = None if adapter is None else hf_adapter(adapter)
        except AdapterError as exc:
            raise SystemUnavailable(f"{system}: {exc}") from None
    settings = None
    if render is None and tokenizer is None and family is None:
        raise SystemUnavailable(f"{system}: the local prompt needs the model folder's chat template "
                                "(tokenizer)")
    if render is None and backend == LLAMA_CPP:  # the family path (module docstring: Local systems)
        if family is None:
            try:
                family = load_family(tokenizer)
            except FamilyError as exc:
                raise SystemUnavailable(f"{system}: {exc}") from None
        settings = generation or GenerationSettings(mode=Mode.NO_THINK, decoding=GREEDY,
                                                    reasoning_budget=family.spec.reasoning_budget)
        if settings.mode not in family.spec.modes:
            raise SystemUnavailable(f"{system}: {family.spec.name} has no {settings.mode} mode (its modes: "
                                    f"{', '.join(family.spec.modes)})")
    else:
        if generation is not None and (generation.mode is not Mode.NO_THINK or generation.samples):
            where = "the hf backend" if backend == HF else "an injected render"
            raise ValueError(f"{system}: {where} completes rendered text greedily without thinking; a "
                             f"{generation.mode} generation with {generation.decoding} needs the family "
                             "path on llama-server")
        family = None
        if render is None:
            from transformers import AutoTokenizer  # heavy import, only for a local system

            from ..train.samples import prompt_renderer

            render = prompt_renderer(AutoTokenizer.from_pretrained(tokenizer))
    demos, block = None, ""
    if system == "b1":
        try:
            demos = load_demonstrations(training_set, requests)
        except FileNotFoundError as exc:
            raise SystemUnavailable(f"b1: the training set is not here ({exc.filename})") from None
        block = demonstrations_block(demos, QWEN_JSON.render if family is None
                                     else family.spec.tool_format.render)
    closers: list[Callable[[], None]] = []
    folder = None if tokenizer is None else str(tokenizer)
    if backend == HF:
        if local_client is None:
            local_client = _load_hf(system, tokenizer, adapter)
            closers.append(local_client.close)
        backend_info = {"provider": "transformers", "tokenizer": folder,
                        "dtype": getattr(local_client, "dtype", None),
                        "device": getattr(local_client, "device", None), "adapter": adapter_record}
    else:
        if local_client is None:
            from ..llm.local import LocalClient

            local_client = LocalClient(server)
            if not local_client.health():
                local_client.close()
                raise SystemUnavailable(f"{system}: no llama-server answers at {server} "
                                        "(scripts/serve_local.py starts one)")
            closers.append(local_client.close)
        try:
            adapters = loaded_adapters(local_client)
        except SystemUnavailable as exc:
            for close in closers:
                close()
            raise SystemUnavailable(f"{system}: {exc}") from None
        backend_info = {"provider": "llama-server", "server": server, "tokenizer": folder,
                        "adapters": adapters}
    name = f"{local_name(tokenizer)}:lora={scale:g}"

    def candidate(generating: GenerationSettings | None) -> Model:
        """The local model generating so, inside the note and the rules when they are on (module docstring:
        The inference wrappers)."""
        model: Model = (LocalModel(local_client, render, lora_scale=scale, name=name) if generating is None
                        else LocalModel.for_family(local_client, family, generating, lora_scale=scale,
                                                   name=name))
        if notes is not None:
            model = NotedModel(model, notes)
        return RuledModel(model) if rules else model

    timings = None
    if settings is not None and settings.votes:
        greedy, samples = voted_settings(settings)
        voted = VotedModel(candidate(greedy), [candidate(sample) for sample in samples], mode=vote_mode,
                           server_timings=lambda: getattr(local_client, "last_timings", None))
        model, timings = voted, (lambda: voted.last_timings)
    else:
        model = candidate(settings)
    if settings is None:
        info = {"system": system, "backend": backend, "model": model.name, "lora_scale": scale,
                "temperature": 0.0, **backend_info}
    else:
        info = {"system": system, "backend": backend, "model": model.name, "lora_scale": scale,
                "temperature": settings.decoding.temperature, **backend_info, "family": family.record(),
                "generation": settings.record()}
    if timings is not None:
        info["vote_mode"] = vote_mode
    if rules:
        info["rules"] = {"version": RULES_VERSION, "names": list(RULES)}
    if notes is not None:
        info["glossary"] = {"path": Path(glossary).as_posix(), "sha256": sha256_of(glossary),
                            "terms": len(notes)}
        if count_tokens is not None:
            count_tokens = noted_counter(count_tokens, notes)
    if demos is not None:
        model = DemonstratedModel(model, block)
        if count_tokens is not None:
            count_tokens = demonstrated_counter(count_tokens, block)
        info["model"] = model.name
        info["demonstrations"] = {"requests": [d.request for d in demos], "kinds": [d.kind for d in demos],
                                  "block_sha256_16": block_hash(block),
                                  "source": Path(training_set).as_posix()}
    return Built(model, count_tokens, info, local_client=local_client, closers=closers, timings=timings)


def _glossary(system: str, path: str | Path) -> Glossary:
    """The glossary file of a system's note (module docstring: The inference wrappers); SystemUnavailable
    when it cannot be read or used."""
    try:
        return load_glossary(path)
    except (OSError, GlossaryError) as exc:
        raise SystemUnavailable(f"{system}: the glossary {Path(path).as_posix()} cannot be used: "
                                f"{exc}") from None


# ---- the wrapper flags of the runners (scripts/run_eval.py, scripts/run_e2e.py) ---------------------------


def votes_count(text: str) -> int:
    """--votes: the number of samples, a whole number from 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a whole number from 1, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a whole number from 1, got {text!r}")
    return value


def add_wrapper_flags(ap: argparse.ArgumentParser) -> None:
    """--votes K, --vote-mode, --rules and --glossary FILE: the inference wrappers of build_system (module
    docstring: The inference wrappers), as both runners take them."""
    ap.add_argument("--votes", type=votes_count, default=None, metavar="K",
                    help="answer by a vote of K samples, the greedy answer breaking ties (sample k: the base "
                         "seed + k)")
    ap.add_argument("--vote-mode", choices=VOTE_MODES, default=None,
                    help=f"the vote's answer keys: {FINAL} (the default) or the 10-29 analysis' reproduction")
    ap.add_argument("--rules", action="store_true", help="apply the runtime rules to every answer")
    ap.add_argument("--glossary", type=Path, default=None, metavar="FILE",
                    help="put this zh-en glossary's note on every prompt (data/glossary/zh-en.json)")


def wrapper_flags_error(args: argparse.Namespace, *, local_side: bool, hf: bool = False) -> str | None:
    """The usage error of the wrapper flags a runner was given, None when there is none: they wrap a local
    system or ftr's local side (local_side), --vote-mode needs --votes, and a vote samples, so neither with
    --greedy nor on the hf backend."""
    if (args.votes is not None or args.rules or args.glossary is not None) and not local_side:
        return (f"--votes, --rules and --glossary wrap the local systems only "
                f"({', '.join((*LOCAL_SCALES, *ROUTED))}), not {args.system}")
    if args.vote_mode is not None and args.votes is None:
        return "--vote-mode chooses the keys of a vote: it needs --votes"
    if args.votes is not None and args.greedy:
        return "--votes with --greedy: a vote's samples sample (the greedy candidate is the vote's own)"
    if args.votes is not None and hf:
        return "--votes on the hf backend: a vote samples on llama-server (--backend llama.cpp)"
    return None


def with_votes(generation: GenerationSettings | None, votes: int | None) -> GenerationSettings | None:
    """The generation settings with the votes asked for, generation itself without; ValueError when they
    cannot vote: no family path, or a decoding that does not sample (GenerationSettings)."""
    if votes is None:
        return generation
    if generation is None:
        raise ValueError("--votes needs the family path on llama-server (the model folder's family)")
    return dataclasses.replace(generation, votes=votes)


def local_name(tokenizer: str | Path | None) -> str:
    """The model name of a local system's records, before the lora scale: the tokenizer folder's name in lower
    case (models/Qwen3-4B: qwen3-4b), LOCAL_NAME without a folder (module docstring). Either separator ends a
    folder name on every platform (PureWindowsPath), so a configuration written on Windows names the same
    model on Linux."""
    return LOCAL_NAME if tokenizer is None else PureWindowsPath(str(tokenizer)).name.lower()


def cloud_system(model: object) -> str | None:
    """The cloud system (CLOUD_MODELS) whose model this is; None for a model of no cloud system."""
    return next((system for system, name in CLOUD_MODELS.items() if name == model), None)


def routing_dev_sets(config: dict) -> list[tuple[object, object]]:
    """(path, SHA-256) of every development set a routing configuration was chosen on, in its order: its
    "dev_sets" (scripts/route_curve.py over several sets), else its one set, "dev_set" and "dev_sha256"
    (either may be missing, as in a hand-made file: None then). ValueError for "dev_sets" that is not a
    non-empty list of objects."""
    sets = config.get("dev_sets")
    if sets is None:
        return [(config.get("dev_set"), config.get("dev_sha256"))]
    if not isinstance(sets, list) or not sets or not all(isinstance(entry, dict) for entry in sets):
        raise ValueError(f"dev_sets must be a non-empty list of objects, got {sets!r:.200}")
    return [(entry.get("path"), entry.get("sha256")) for entry in sets]


def _given_sets(dev_set: DevSets, recorded: list[tuple[object, object]]) -> list[Path]:
    """The development sets to check a configuration against: dev_set as one or several paths, or, for None,
    the paths the configuration records (data/eval/dev.jsonl for a set it does not name)."""
    if dev_set is None:
        return [Path(name) if isinstance(name, str) and name else DEV_SET for name, _ in recorded]
    if isinstance(dev_set, str | Path):
        return [Path(dev_set)]
    return [Path(path) for path in dev_set]


def load_routing(path: Path, dev_set: DevSets = DEV_SET, system: str = "ftr") -> tuple[RouteConfig, dict]:
    """The routing configuration at path, checked against the development sets it was chosen on (dev_set: one
    path, several in the configuration's order, or None for the paths the configuration names), and what the
    manifest records of it (module docstring: FT+R); SystemUnavailable naming what is missing or does not
    match. The adapter's SHA-256 is checked once the server has listed its adapters (_routed); the cloud model
    names the cloud system (cloud_system)."""
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemUnavailable(f"{system}: no routing configuration at {path.as_posix()} "
                                "(scripts/route_curve.py writes it from the development runs)") from None
    except (OSError, ValueError) as exc:
        raise SystemUnavailable(f"{system}: the routing configuration {path.as_posix()} cannot be read: "
                                f"{exc}") from None
    if not isinstance(config, dict):
        raise SystemUnavailable(f"{system}: the routing configuration {path.as_posix()} is not a JSON object")
    cloud = config.get("cloud_model")
    if cloud_system(cloud) is None:
        known = ", ".join(f"{name} ({s})" for s, name in CLOUD_MODELS.items())
        raise SystemUnavailable(f"{system}: the routing configuration routes to {cloud!r}, the model of no "
                                f"cloud system ({known})")
    if config.get("threshold") is None or config.get("rule_met") is False:
        raise SystemUnavailable(f"{system}: the routing configuration has no threshold: the preregistered "
                                "rule was not met on the development set")
    rules = config.get("rules")
    try:
        route = RouteConfig(signal=config.get("signal"), threshold=config["threshold"],
                            rules=None if rules is None else RouteRules.from_json(rules))
    except ValueError as exc:
        raise SystemUnavailable(f"{system}: the routing configuration's {exc}") from None
    try:
        recorded = routing_dev_sets(config)
    except ValueError as exc:
        raise SystemUnavailable(f"{system}: the routing configuration's {exc}") from None
    given = _given_sets(dev_set, recorded)
    if len(given) != len(recorded):
        def sets(count: int) -> str:
            return f"{count} development set{'' if count == 1 else 's'}"

        raise SystemUnavailable(f"{system}: the routing configuration was chosen on {sets(len(recorded))}, "
                                f"not on the {len(given)} given ({', '.join(p.as_posix() for p in given)})")
    digests = []
    for number, ((name, recorded_digest), dev) in enumerate(zip(recorded, given, strict=True), 1):
        try:
            dev_digest = sha256_of(dev)
        except OSError:
            raise SystemUnavailable(f"{system}: the development set {dev.as_posix()} is not here, so the "
                                    "routing configuration cannot be checked against it") from None
        if recorded_digest != dev_digest:
            if "dev_sets" not in config:  # the message of a configuration of one set, as before M7
                raise SystemUnavailable(f"{system}: the dev set SHA-256 in the routing configuration "
                                        f"({recorded_digest}) is not that of {dev.as_posix()} ({dev_digest})")
            raise SystemUnavailable(f"{system}: the routing configuration's development set {number} "
                                    f"({name}) has SHA-256 {recorded_digest}, not that of {dev.as_posix()} "
                                    f"({dev_digest})")
        digests.append(dev_digest)
    adapter = config.get("adapter_sha256")
    if not isinstance(adapter, str) or not adapter:
        raise SystemUnavailable(f"{system}: the routing configuration names no adapter SHA-256")
    dev_record = ({"dev_sets": [{"path": dev.as_posix(), "sha256": digest}
                                for dev, digest in zip(given, digests, strict=True)]}
                  if "dev_sets" in config else {"dev_sha256": digests[0]})
    info = {"path": path.as_posix(), "sha256": sha256_of(path), "signal": route.signal,
            "threshold": route.threshold, **dev_record, "adapter_sha256": adapter,
            "cloud_model": cloud, "rules": None if route.rules is None else route.rules.to_json(),
            "policy": config.get("chosen_policy")}
    return route, info


def _routed(system: str, *, routing_config: Path, dev_set: DevSets, server: str, tokenizer: str | Path | None,
            count_tokens: TokenCounter | None, llm_dir: Path, stage: str, max_cost: float | None,
            openai_client: object | None, local_client: SupportsComplete | None,
            render: Callable | None, cloud_thinking: str = "auto",
            generation: GenerationSettings | None = None, family: LoadedFamily | None = None,
            rules: bool = False, glossary: str | Path | None = None, vote_mode: str = FINAL) -> Built:
    """ftr (module docstring: FT+R): its local system on llama-server, routed to the cloud system its
    configuration names, with the inference wrappers on its local side, inside the routing (module docstring:
    The inference wrappers). Everything that fails without a client comes first; a failure after the local
    client opened closes it."""
    route, routing = load_routing(routing_config, dev_set, system)
    local_system, cloud_name = ROUTED[system], cloud_system(routing["cloud_model"])
    try:
        local = _local(local_system, backend=LLAMA_CPP, adapter=None, server=server, tokenizer=tokenizer,
                       count_tokens=count_tokens, training_set=TRAINING_SET, requests=None,
                       local_client=local_client, render=render, generation=generation, family=family,
                       rules=rules, glossary=glossary, vote_mode=vote_mode)
    except SystemUnavailable as exc:
        raise SystemUnavailable(f"{system}: {exc}") from None
    try:
        digests = [a["sha256"] for a in local.info.get("adapters") or []]
        if not digests:
            raise SystemUnavailable(f"{system}: the server lists no adapter, so the routing configuration's "
                                    f"adapter SHA-256 ({routing['adapter_sha256']}) cannot be checked")
        if routing["adapter_sha256"] not in digests:
            raise SystemUnavailable(f"{system}: the adapter SHA-256 in the routing configuration "
                                    f"({routing['adapter_sha256']}) is not that of the adapter the server "
                                    f"lists ({', '.join(digests)})")
        try:
            cloud = _cloud(cloud_name, llm_dir=llm_dir, stage=stage, max_cost=max_cost,
                           openai_client=openai_client, count_tokens=None, thinking=cloud_thinking)
        except SystemUnavailable as exc:
            raise SystemUnavailable(f"{system}: {exc}") from None
    except SystemUnavailable:
        local.close()
        raise
    model = RoutedModel(local.model, cloud.model, route)
    shared = {k: local.info[k] for k in ("server", "tokenizer", "adapters", "family", "generation",
                                         "vote_mode", "rules", "glossary") if k in local.info}
    info = {"system": system, "provider": "routed", "model": model.name,
            "lora_scale": local.info["lora_scale"], "temperature": local.info["temperature"],
            "backend": LLAMA_CPP, **shared, "cloud_model": cloud.info["model"], "local": local.info,
            "cloud": cloud.info, "routing": routing}
    return Built(model, local.count_tokens, info, local_client=local.local_client,
                 closers=[*local.closers, *cloud.closers], timings=local.timings)


def hf_adapter(path: str | Path) -> dict:
    """An hf system's adapter folder as its manifest records it: the path as given (POSIX) and the SHA-256 of
    its adapter_model.safetensors. AdapterError for anything but a PEFT folder (llm.hf.check_adapter)."""
    folder = check_adapter(path)
    return {"path": Path(path).as_posix(), "safetensors_sha256": sha256_of(folder / ADAPTER_WEIGHTS)}


def _load_errors() -> tuple[type[BaseException], ...]:
    """What loading a model or an adapter here raises when it cannot be done: a missing file or package, no
    GPU, a configuration or an adapter that cannot be used (ValueError, of which AdapterError is one), and the
    error safetensors raises for a weights file it cannot read (a corrupt adapter_model.safetensors), which
    derives from Exception alone. safetensors, absent where the training packages are not installed, is
    imported only here."""
    errors: tuple[type[BaseException], ...] = (OSError, RuntimeError, ImportError, ValueError)
    try:
        from safetensors import SafetensorError
    except ImportError:
        return errors
    return (*errors, SafetensorError)


def _load_hf(system: str, model_dir: str | Path | None, adapter: str | Path | None) -> SupportsComplete:
    """The hf backend's client over the model folder's weights and the adapter folder (llm.hf.load_client);
    SystemUnavailable without a model folder or when the model cannot be loaded here (_load_errors)."""
    if model_dir is None:
        raise SystemUnavailable(f"{system}: the hf backend loads the weights of the model folder (tokenizer)")
    from ..llm.hf import load_client  # imports torch, transformers and peft when it loads

    try:
        return load_client(model_dir, adapter)
    except _load_errors() as exc:
        loaded = Path(model_dir).as_posix() + ("" if adapter is None else f" with {Path(adapter).as_posix()}")
        raise SystemUnavailable(f"{system}: the hf backend could not load {loaded}: {exc}") from None


def loaded_adapters(client: object) -> list[dict] | None:
    """The adapters the server loaded, as a local run's manifest records them (module docstring).

    One {"id", "path", "sha256"} per /lora-adapters entry, path as the server was given it (a relative one is
    taken from the current folder, where the scripts and the server run); None for a client that cannot list
    them (a test stand-in). SystemUnavailable when the listing fails or a listed GGUF is not a file here.
    """
    listing = getattr(client, "adapters", None)
    if listing is None:
        return None
    try:
        entries = listing()
    except Exception as exc:  # noqa: BLE001  e.g. the server went away between /health and the listing
        raise SystemUnavailable(f"the server's adapter listing failed: {exc!r}") from None
    out = []
    for entry in entries:
        path = entry.get("path") if isinstance(entry, dict) else None
        if not isinstance(path, str) or not path or not Path(path).is_file():
            raise SystemUnavailable(f"the server lists the adapter {path!r}, which is not a file here, so "
                                    "the run could not record which GGUF it measured; run from the folder "
                                    "the server runs in")
        out.append({"id": entry.get("id"), "path": path, "sha256": sha256_of(path)})
    return out

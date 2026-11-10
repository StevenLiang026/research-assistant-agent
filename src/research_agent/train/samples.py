"""Per-step training samples: the prompt the model sees at inference and the assistant turn to produce.

A sample is one assistant step of a history: the step's context rendered with the model's own chat template
plus the generation prompt, and the assistant message as the same template renders it, ending at the turn
terminator (the template's trailing newline is cut, or a trainer would append a second terminator). The full
render of context plus message must equal prompt + completion, which is checked for every sample. This is the
format spike S4 settled on after S2 found that TRL's training template renders multi-step histories
differently from the inference template (docs/m0-验证.md).

Two renderers give these samples:

- render_sample / build_step_samples take a tokenizer and give {"prompt", "completion", "tokens"}: the
  prompt is the template call research_agent.llm.local.hf_renderer makes for the local model (thinking off),
  so the local model's prompts are the training prompts (prompt_renderer). scripts/train.py trained FT2 and
  G2 on these strings;
- render_family_sample / build_family_samples take a loaded family (research_agent.family,
  docs/m7-家族层设计.md §5) and give pre-tokenized samples, THINK or NO_THINK per sample (below). For the
  Qwen3 hybrid family in NO_THINK mode their prompt, completion and ids are the first renderer's and TRL's
  tokenization of it, which tests/train/test_regression_anchor.py holds over the whole FT2 data.

Four inputs are accepted:

- an API-format history (a list of messages, as Transcript.messages): one sample per assistant message,
  with build_context applied to the history before it (older tool outputs compressed), as the loop showed
  the model;
- a Transcript: the same, except that a step accepted after a repair is skipped, because the model answered
  a repair request that the history does not hold (agent.context.step_contexts explains);
- a public-data sample {"messages": [...], "tools": [...]}, the TRL conversational format that the
  public-data converter (research_agent.data.public, M2) writes: messages in API or HF format, tools in the
  OpenAI function format (as a list, or as a JSON string). Such a sample is rendered as given, with its own
  tools and without compression, because its tool outputs are not the agent's and the model must still
  read them;
- an agent-history record {"messages": [...], "tools": [...], "history": "agent", ...}, the form
  research_agent.data.assemble writes the synthetic trajectories in: although it carries tools like a public
  sample, it is cut like an API-format history (build_context, older tool outputs compressed), because the
  loop showed the teacher those contexts and will show them to the student. It must carry a "schema_hash"
  and a "prompt_hash" equal to the current ones (the system prompt it holds is the one the model will be
  shown), and its "tools" must be JSON-equal to MODEL_TOOLS (compared in canonical form);
  it is then rendered with MODEL_TOOLS itself, in its own key order, exactly as the agent loop and the eval
  runner render it, since the stored copy has had its keys sorted. A record failing any check is refused.
  It holds no step records, so a step accepted after a repair cannot be recognised here; the assembler
  leaves out every trajectory that has one.

Samples longer than max_length are dropped and counted, never truncated (the plan's rule), so TRL's own
truncation never triggers. Every sample also carries its token count under "tokens".

The family path (docs/m7-家族层设计.md §5). A step's NO_THINK prompt and completion are cut from the
family's renders as above. When a reasoning sidecar (research_agent.data.reasoning.ReasoningIndex) holds the
step's reasoning and the family thinks, the sample is THINK: the prompt is the family's THINK prompt (it ends
with the open thinking block) and the completion is the reasoning without the newlines at either end, the
family's think_close and the NO_THINK completion, joined as the family's template joins them
(LoadedFamily.think_completion: Granite's trims the close of a turn without text); the template's own
render of the target carrying its reasoning must equal that construction when the family declares where the
template reads reasoning from (reasoning_field). Without reasoning the sample is NO_THINK: the empty thinking
block sits in the prompt, outside the loss, so the model never learns to write an empty block. The
completion is constructed, never
cut from a THINK render. family.encode_sample then gives the ids and the completion's first index, with the
token-boundary, BOS and terminator rules asserted. A THINK sample over max_length is rendered NO_THINK
instead and kept (think_fallback); a NO_THINK sample over max_length is dropped (over_length). Earlier steps'
reasoning is never shown (docs/m7-计划.md §2 思考).

Sample keys. A family sample carries "key" = (record name, k): the name is the record's "id", or the data
file's stem and the record's index in it (record_name), and k is the ordinal of the step's assistant message
among the record's assistant messages, from 0, counted before any step is skipped or dropped, the ordinal the
sidecars name a turn by (data.reasoning, scripts/join_reasoning.py). The sample's fields: key, mode ("think"
or "no_think"), think_fallback, prompt, completion, input_ids, completion_start, tokens (len(input_ids)) and
reasoning_tokens (the reasoning's token count for a THINK sample, None otherwise); completion_mask(sample)
gives the mask TRL trains with.

Context turns (docs/m7-本地提升计划.md WP6). A record may say that its first assistant turns are context only:
"context_turns": n on an agent-history record or a public sample, an integer from 0 to below the record's
number of assistant messages (context_turns() refuses anything else, and load_records refuses the key on an
object it reads as a bare history, which would lose it). step_pairs then leaves out the steps with k < n, so
neither renderer gives them a sample and they count as no drop; the steps kept keep their k. scripts/
transform_synth.py --ft4 writes the key on its derived records, cut copies of a trajectory whose copied turns
only show the context of the turn built after them. A record without the key is cut as before.

Refusals. A step the family cannot render or encode raises its FamilyError: a TemplateError (the template
refused the messages), a TokenBoundaryError (the sample does not split at a token boundary) or a BosError
(the BOS rule); the prefix and terminator checks on the NO_THINK render are ValueErrors as in render_sample.
A THINK construction is refused as well when the template's own render disagrees with it (above) or when
the reasoning holds the tag that closes the thinking block: the family's split_reasoning would read the
trained reasoning back shorter at inference (the sidecar validator, data.reasoning.check_row, keeps think
tags out of the sidecars; the renderer holds the rule for every caller). build_family_samples with a refused
list appends {"key", "kind", "error"} to it instead and goes on, the way scripts/train.py's dry run counts
them by kind (docs/m7-计划.md, S2: template, boundary, bos).

Glossary notes (docs/m7-本地提升计划.md, WP4; WP6 注释). With notes (a GlossaryNotes: a glossary and a
dropout), an agent-history record whose id the dropout keeps trains with research_agent.glossary's note on
the system message of every step's context: noted_messages(context, glossary), the function the inference
wrapper calls on the history it sends, so the note reads the latest user message of that context and a step's
prompt is the prompt inference renders for the same history. The note is part of the prompt, so its tokens
count toward max_length like any other text (an over-length sample is dropped, a THINK one falls back to
NO_THINK); the target is untouched, and so is the record: the data files are not rewritten
(scripts/transform_synth.py refuses a system message that is not the prompt). The dropout is keyed on the
record id (NOTE_HASH_RULE, keeps_note): the record keeps its notes when the first 8 bytes of
SHA-256("glossary-note:" + id), read as a big-endian unsigned integer, are at least dropout x 2^64. With
NOTE_DROPOUT (0.5) about half the records train without a note (the plan: the model keeps translating without
one), the same half in every run, whatever the file order. Every sample then carries "note": the note its
prompt holds, or None (a dropped record, a step without a glossary term, a record that is no agent history).
An agent-history record without an id is refused. Without notes (the default) none of this runs, and the
samples are the ones the renderers gave before the option.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..agent.context import step_contexts
from ..agent.prompts import PROMPT_HASH
from ..agent.types import Transcript, api_to_hf
from ..canonical import canonical_json, split_lines
from ..data.reasoning import ReasoningIndex
from ..family.core import BosError, FamilyError, LoadedFamily, Mode, Thinking, TokenBoundaryError
from ..glossary import Glossary, noted_messages
from ..tools.schemas import CONTEXT_TOKENS, MODEL_TOOLS, SCHEMA_HASH

Record = Transcript | list[dict] | dict
HISTORY_KEY, AGENT_HISTORY = "history", "agent"  # the marker of an agent-history record (module docstring)
CONTEXT_TURNS = "context_turns"  # the leading assistant turns that are context only (module docstring)
Pairs = list[tuple[list[dict], dict]]  # (context, assistant message) per step, as step_contexts gives them
Steps = list[tuple[int, list[dict], dict]]  # (k, context, assistant message): the pairs with their ordinal
THINK, NO_THINK = str(Mode.THINK), str(Mode.NO_THINK)  # a sample's "mode"
TEMPLATE, BOUNDARY, BOS = "template", "boundary", "bos"  # the refusal kinds (module docstring)
REFUSAL_KINDS = (TEMPLATE, BOUNDARY, BOS)
BUDGET_FLOOR, BUDGET_STEP, BUDGET_CAP = 1024, 256, 2048  # docs/m7-计划.md §2 (推理): the rule for R
NOTE_DROPOUT = 0.5  # docs/m7-本地提升计划.md WP6 (注释): half the records train without the glossary note
NOTE_SALT = "glossary-note:"
NOTE_HASH_RULE = (f"a record keeps its glossary notes when the first 8 bytes of SHA-256(\"{NOTE_SALT}\" + "
                  "its id, UTF-8), read as a big-endian unsigned integer, are at least dropout x 2^64")


class Tokenizer(Protocol):
    """What is needed from a Hugging Face tokenizer; the tests use a fake with these two methods."""

    eos_token: str

    def apply_chat_template(self, messages: list[dict], *, tools: list[dict] | None = None,
                            tokenize: bool = False, add_generation_prompt: bool = False,
                            enable_thinking: bool = False) -> str: ...

    def __call__(self, text: str, *, add_special_tokens: bool = True) -> dict: ...


@dataclass(frozen=True)
class Dropped:
    """Steps that gave no sample: over max_length (dropped, never truncated) or accepted after a repair."""

    over_length: int = 0
    after_repair: int = 0

    @property
    def total(self) -> int:
        return self.over_length + self.after_repair

    def __add__(self, other: Dropped) -> Dropped:
        return Dropped(self.over_length + other.over_length, self.after_repair + other.after_repair)


@dataclass(frozen=True)
class GlossaryNotes:
    """The glossary-note option of the renderers (module docstring: Glossary notes): the glossary (a plain
    mapping is validated into a research_agent.glossary.Glossary) and the share of records that train without
    their notes, between 0 and 1."""

    glossary: Glossary
    dropout: float = NOTE_DROPOUT

    def __post_init__(self) -> None:
        if not isinstance(self.glossary, Glossary):
            object.__setattr__(self, "glossary", Glossary(self.glossary))
        if not 0 <= self.dropout <= 1:  # also refuses NaN
            raise ValueError(f"the note dropout is a share between 0 and 1, not {self.dropout!r}")


def keeps_note(record_id: str, dropout: float) -> bool:
    """True when the record trains with its glossary notes (NOTE_HASH_RULE): the first 8 bytes of
    SHA-256(NOTE_SALT + id), a big-endian unsigned integer, are at least dropout x 2^64. dropout 0 keeps every
    record, 1 none."""
    digest = hashlib.sha256((NOTE_SALT + record_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >= dropout * 2**64


def build_step_samples(history: Record, tokenizer: Tokenizer, *, tools: list[dict] | None = MODEL_TOOLS,
                       max_length: int = CONTEXT_TOKENS,
                       notes: GlossaryNotes | None = None) -> tuple[list[dict], Dropped]:
    """The samples of one history, transcript or public sample, and the counts of steps that gave none.

    A dict with "steps" is a transcript as save_transcript wrote it. A transcript recorded under another tool
    schema is refused unless tools are passed explicitly: the model would learn calls against a schema it is
    not shown. Public samples use their own tools; the tools argument is their fallback. Agent-history
    records always use MODEL_TOOLS, whatever the tools argument (the module docstring says why). notes puts
    the glossary note on an agent-history record's steps (module docstring: Glossary notes).
    """
    steps, tools, after_repair = step_pairs(history, tools)
    steps, step_notes = noted_steps(history, steps, notes)
    samples, over_length = [], 0
    for position, (_, context, target) in enumerate(steps):
        sample = render_sample(tokenizer, context, target, tools=tools)
        if step_notes is not None:
            sample["note"] = step_notes[position]
        if sample["tokens"] > max_length:
            over_length += 1
        else:
            samples.append(sample)
    return samples, Dropped(over_length=over_length, after_repair=after_repair)


def build_family_samples(history: Record, family: LoadedFamily, *, name: str,
                         reasoning: ReasoningIndex | None = None, tools: list[dict] | None = MODEL_TOOLS,
                         max_length: int = CONTEXT_TOKENS, refused: list[dict] | None = None,
                         notes: GlossaryNotes | None = None) -> tuple[list[dict], Dropped]:
    """The pre-tokenized samples of one record through a family (module docstring: The family path), and
    the counts of steps that gave none. name is the record's name for the sample keys (record_name).

    The reasoning of step k is reasoning.lookup(name, k, target) when an index is given and the family
    thinks (a family that never thinks is not looked up for: it has no THINK mode, so a stale row could not
    change its samples); a stale row is the index's ReasoningError. A step the family refuses raises, or
    goes to refused when that list is given (module docstring: Refusals). notes puts the glossary note on an
    agent-history record's steps (module docstring: Glossary notes).
    """
    steps, tools, after_repair = step_pairs(history, tools)
    steps, step_notes = noted_steps(history, steps, notes)
    thinks = family.spec.thinking is not Thinking.NONE
    samples, over_length = [], 0
    for position, (k, context, target) in enumerate(steps):
        key = (name, k)
        text = reasoning.lookup(name, k, target) if reasoning is not None and thinks else None
        try:
            sample = render_family_sample(family, context, target, tools=tools, reasoning=text,
                                          max_length=max_length)
        except FamilyError as exc:
            if refused is None:
                raise
            refused.append({"key": key, "kind": refusal_kind(exc), "error": str(exc)})
            continue
        if step_notes is not None:
            sample["note"] = step_notes[position]
        if sample["tokens"] > max_length:
            over_length += 1
        else:
            samples.append({"key": key, **sample})
    return samples, Dropped(over_length=over_length, after_repair=after_repair)


def noted_steps(history: Record, steps: Steps,
                notes: GlossaryNotes | None) -> tuple[Steps, list[str | None] | None]:
    """The steps with the glossary note on their contexts' system message, and each step's note (module
    docstring: Glossary notes); the steps themselves and None without notes. Only an agent-history record
    whose id keeps_note keeps is noted; RecordError for an agent-history record without an id."""
    if notes is None:
        return steps, None
    if not is_agent_history(history):
        return steps, [None] * len(steps)
    name = history.get("id")
    if not isinstance(name, str) or not name:
        raise RecordError("an agent-history record without an id cannot take glossary notes: the note "
                          "dropout is keyed on the record id")
    if not keeps_note(name, notes.dropout):
        return steps, [None] * len(steps)
    noted: Steps = []
    step_notes: list[str | None] = []
    for k, context, target in steps:
        shown, note = noted_messages(context, notes.glossary)
        noted.append((k, shown, target))
        step_notes.append(note)
    return noted, step_notes


def refusal_kind(exc: FamilyError) -> str:
    """Which of REFUSAL_KINDS a family's refusal is: a token boundary, the BOS rule, or the template (a
    TemplateError, or any other disagreement of the template's render with the family: the declared suffix,
    the prefix of the full render, the terminator, the THINK construction's render or its split)."""
    if isinstance(exc, TokenBoundaryError):
        return BOUNDARY
    if isinstance(exc, BosError):
        return BOS
    return TEMPLATE


def step_pairs(history: Record, tools: list[dict] | None) -> tuple[Steps, list[dict] | None, int]:
    """The (k, context, target) steps of a record (module docstring: the four inputs), the tools they render
    with, and how many steps a transcript skipped as accepted after a repair. k counts every assistant
    message of the record, skipped ones included (module docstring: Sample keys); the context turns of a
    record that names them are left out (module docstring: Context turns)."""
    skip = context_turns(history)
    if isinstance(history, dict) and "steps" in history:
        history = Transcript(**history)
    after_repair = 0
    if isinstance(history, Transcript):
        steps, after_repair = _transcript_steps(history, tools)
    elif is_agent_history(history):
        pairs, tools = _agent_pairs(history)
        steps = _numbered(pairs)
    elif isinstance(history, dict):
        pairs, tools = _public_pairs(history, tools)
        steps = _numbered(pairs)
    else:
        steps = _numbered(step_contexts(history))
    if skip:
        steps = [step for step in steps if step[0] >= skip]
    return steps, tools, after_repair


def context_turns(record: Record) -> int:
    """How many leading assistant turns of a record are context only (module docstring: Context turns): its
    "context_turns", 0 without one. RecordError, naming the record, when that is no integer from 0 to below
    the record's number of assistant messages."""
    if not isinstance(record, dict) or CONTEXT_TURNS not in record:
        return 0
    value, messages = record[CONTEXT_TURNS], record.get("messages")
    turns = sum(isinstance(m, dict) and m.get("role") == "assistant" for m in messages) \
        if isinstance(messages, list) else 0
    if type(value) is not int or not 0 <= value < turns:
        raise RecordError(f"the record {record.get('id')!r} has {CONTEXT_TURNS} {value!r}: an integer from 0 "
                          f"to below its {turns} assistant messages is needed")
    return value


def _numbered(pairs: Pairs) -> Steps:
    return [(k, context, target) for k, (context, target) in enumerate(pairs)]


def record_name(record: Record, path: str | Path, index: int) -> str:
    """The name a record's samples are keyed by: its "id" when it is an object with one (the synthetic
    files), else the data file's stem and the record's index in the file (from 0, as load_records numbers
    them), "stem:index"."""
    if isinstance(record, dict) and record.get("id") is not None:
        return str(record["id"])
    return f"{Path(path).stem}:{index}"


def prompt_renderer(tokenizer: Tokenizer) -> Callable[[list[dict], list[dict] | None], str]:
    """render(hf_messages, tools) -> the prompt: the template call that renders the prompt half of every
    training sample (render_sample), with the generation prompt appended and thinking off. The local model is
    prompted through this same function (research_agent.eval.systems), so its prompts are the training
    prompts."""

    def render(messages: list[dict], tools: list[dict] | None = None) -> str:
        return tokenizer.apply_chat_template(messages, tools=tools, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)

    return render


def render_sample(tokenizer: Tokenizer, context: list[dict], target: dict, *,
                  tools: list[dict] | None) -> dict:
    """One {"prompt", "completion", "tokens"} sample from an API-format context and assistant message.

    ValueError when the template does not render the context plus the generation prompt as a prefix of
    the full conversation, or when the completion does not end with the turn terminator: in either case
    the training text would not match the inference text.
    """
    hf_context = api_to_hf(context)
    prompt = prompt_renderer(tokenizer)(hf_context, tools)
    full = tokenizer.apply_chat_template(hf_context + api_to_hf([target]), tools=tools, tokenize=False,
                                         enable_thinking=False)
    completion, _ = _cut_completion(prompt, full, tokenizer.eos_token)
    tokens = len(tokenizer(prompt + completion, add_special_tokens=False)["input_ids"])
    return {"prompt": prompt, "completion": completion, "tokens": tokens}


def _cut_completion(prompt: str, full: str, end: str) -> tuple[str, str]:
    """The completion of a full render after its prompt, up to and with the terminator, and what the
    template writes after the terminator (the newline it cuts); ValueError when the render does not begin
    with the prompt or the completion does not end with the terminator (render_sample)."""
    if not full.startswith(prompt):
        raise ValueError("the chat template does not render the context and the generation prompt as a "
                         "prefix of the full conversation; training text would differ from inference text")
    completion, tail = full[len(prompt):], ""
    if completion.endswith(end + "\n"):
        completion, tail = completion[:-1], "\n"  # generation stops at the terminator
    if not completion.endswith(end):
        raise ValueError(f"the completion does not end with the terminator {end!r}: {completion[-40:]!r}")
    return completion, tail


def render_family_sample(family: LoadedFamily, context: list[dict], target: dict, *,
                         tools: list[dict] | None, reasoning: str | None = None,
                         max_length: int = CONTEXT_TOKENS) -> dict:
    """One pre-tokenized sample (module docstring: The family path) from an API-format context and
    assistant message, THINK when reasoning is given and the family thinks and the THINK sample fits
    max_length, else NO_THINK (think_fallback tells when a THINK sample was rendered NO_THINK instead). A
    NO_THINK sample may exceed max_length: the caller drops it. The family's refusals raise (Refusals)."""
    spec = family.spec
    hf = family.to_template_messages(context)
    (message,) = family.to_template_messages([target])
    no_think_prompt = family.prompt(hf, tools, Mode.NO_THINK)
    full = family.full(hf + [message], tools, Mode.NO_THINK)
    answer, tail = _cut_completion(no_think_prompt, full, spec.turn_end)
    fallback = False
    if reasoning is not None and spec.thinking is not Thinking.NONE:
        text = reasoning.strip("\n")
        prompt = family.prompt(hf, tools, Mode.THINK)
        completion = family.think_completion(text, answer, message)
        read, _ = family.split_reasoning(completion)
        if read != text:  # the sample would train a reasoning that inference reads back shorter
            raise FamilyError(f"{spec.name}: the family's split reads {len(read)} of the reasoning's "
                              f"{len(text)} characters back: the reasoning holds the tag that closes the "
                              "thinking block, so inference would read a shorter reasoning than the sample "
                              "trains")
        if spec.reasoning_field is not None:  # the template's own render of the target with its reasoning
            rendered = family.full(hf + [{**message, spec.reasoning_field: reasoning}], tools, Mode.THINK)
            if rendered != prompt + completion + tail:
                raise FamilyError(f"{spec.name}: the template renders the target with its "
                                  f"{spec.reasoning_field} as {rendered[len(prompt) - 20:][:80]!r}..., not "
                                  "as the THINK construction prompt + reasoning + think_close + answer "
                                  f"(joined by {spec.think_join})")
        ids, start = family.encode_sample(prompt, completion)
        if len(ids) <= max_length:
            return {"mode": THINK, "think_fallback": False, "prompt": prompt, "completion": completion,
                    "input_ids": ids, "completion_start": start, "tokens": len(ids),
                    "reasoning_tokens": len(family.encode_text(text))}
        fallback = True
    ids, start = family.encode_sample(no_think_prompt, answer)
    return {"mode": NO_THINK, "think_fallback": fallback, "prompt": no_think_prompt, "completion": answer,
            "input_ids": ids, "completion_start": start, "tokens": len(ids), "reasoning_tokens": None}


def completion_mask(sample: dict) -> list[int]:
    """The mask TRL trains a family sample with: 0 over the prompt's ids, 1 over the completion's."""
    start, total = sample["completion_start"], len(sample["input_ids"])
    return [0] * start + [1] * (total - start)


def reasoning_budget(p99: int) -> int:
    """The reasoning budget R the plan's rule picks from the p99 of the training reasoning lengths (in
    tokens): p99 rounded up to a multiple of 256, at least 1,024 and at most 2,048 (docs/m7-计划.md §2 推理;
    docs/m7-底座对比预登记.md 解码)."""
    rounded = math.ceil(p99 / BUDGET_STEP) * BUDGET_STEP
    return min(BUDGET_CAP, max(BUDGET_FLOOR, rounded))


def accepted_after_repair(step: dict) -> bool:
    """True when the step's accepted attempt answered a repair request.

    The accepted attempt is the last in steps[i]["attempts"]; the model saw the repair exchange when the
    attempt before it was answered with "repair". After a handoff the model saw the original context, so
    such a step stays.
    """
    attempts = step.get("attempts") or []
    return len(attempts) >= 2 and attempts[-2].get("action") == "repair"


class RecordError(ValueError):
    """A data file that is not JSON, or a value in it that is not a record; the message names the place."""


def load_records(path: str | Path) -> list[Record]:
    """The records of one data file, as many as research_agent.train.manifest.count_rows counts for it.

    A record is a transcript (an object with "steps"), a public sample (an object with "messages" and
    "tools"), an object with "messages" only (an API-format history of the agent) or a non-empty list of API
    messages (objects with "role"). A JSONL file holds one record per non-blank line. A JSON file holds one
    record or a list of records, read one by one; the one shape the two functions count differently is a
    JSON file whose whole content is a bare message list (one history here, its messages for count_rows), so
    a history goes on a JSONL line or inside {"messages": [...]}. A line or file that is not JSON, or a value
    that is not a record, is a RecordError naming the file and the line.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix == ".jsonl":
        return [_as_record(_loads(line, path, f"line {number}"), path, f"line {number}")
                for number, line in enumerate(split_lines(text), 1) if line.strip()]
    value = _loads(text, path, "the file")
    if isinstance(value, list):
        not_record = _first_failing(value, _is_record)
        if not_record is None:
            return [_as_record(item, path, f"item {index}") for index, item in enumerate(value)]
        not_message = _first_failing(value, _is_message)
        if not_message is not None:
            raise RecordError(f"{path}: the file is a list that is neither a list of records (item "
                              f"{not_record} has no \"messages\" or \"steps\") nor one history (item "
                              f"{not_message} has no \"role\")")
    return [_as_record(value, path, "the file")]


def _loads(text: str, path: Path, where: str) -> object:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RecordError(f"{path}: {where} is not JSON: {exc}") from exc


def _is_record(value: object) -> bool:
    return isinstance(value, dict) and ("steps" in value or "messages" in value)


def _is_message(value: object) -> bool:
    return isinstance(value, dict) and "role" in value


def _first_failing(items: list, check: Callable[[object], bool]) -> int | None:
    """The index of the first item that fails check; None when every item passes."""
    return next((index for index, item in enumerate(items) if not check(item)), None)


def _as_record(value: object, path: Path, where: str) -> Record:
    if isinstance(value, dict) and "steps" in value:
        return Transcript(**value)
    if isinstance(value, dict) and "messages" in value:
        if CONTEXT_TURNS in value:  # module docstring: Context turns
            if "tools" not in value:
                raise RecordError(f"{path}: {where}: {CONTEXT_TURNS} needs a record that keeps it (one with "
                                  "\"tools\"); this one is read as its bare history, which would train its "
                                  "context turns")
            try:
                context_turns(value)
            except RecordError as exc:
                raise RecordError(f"{path}: {where}: {exc}") from exc
        return value if "tools" in value else value["messages"]
    if isinstance(value, list):
        not_message = _first_failing(value, _is_message)
        if value and not_message is None:
            return value
        detail = "it is empty" if not value else f"item {not_message} has no \"role\""
        raise RecordError(f"{path}: {where} is a list but not a history of API messages: {detail}")
    raise RecordError(f"{path}: {where} is not a record (a transcript, a public sample, an object with "
                      "\"messages\" or a list of API messages)")


def _transcript_steps(transcript: Transcript, tools: list[dict] | None) -> tuple[Steps, int]:
    if tools is MODEL_TOOLS and transcript.schema_hash != SCHEMA_HASH:
        raise ValueError(f"the transcript was recorded under tool schema {transcript.schema_hash}, the "
                         f"current schema is {SCHEMA_HASH}; pass tools= explicitly to train on it anyway")
    pairs = step_contexts(transcript.messages)
    if len(pairs) > len(transcript.steps):
        raise ValueError(f"the transcript has {len(pairs)} assistant messages but only "
                         f"{len(transcript.steps)} steps")
    kept, after_repair = [], 0
    for k, ((context, target), step) in enumerate(zip(pairs, transcript.steps, strict=False)):
        if accepted_after_repair(step):  # only the last step may lack a message
            after_repair += 1
        else:
            kept.append((k, context, target))
    return kept, after_repair


def is_agent_history(record: object) -> bool:
    """True for an agent-history record: an object with "messages" and the marker "history": "agent"."""
    return isinstance(record, dict) and "messages" in record and record.get(HISTORY_KEY) == AGENT_HISTORY


def _agent_pairs(record: dict) -> tuple[Pairs, list[dict]]:
    """The record's steps and MODEL_TOOLS itself, the tools the agent loop and the eval runner send.

    The record's own "tools" are only checked, never rendered: the assembler writes canonical JSON, which
    sorts their keys, and the template renders a tool's keys in dict order, so rendering them would give a
    tool block the model never sees at inference.
    """
    name = record.get("id")
    if "schema_hash" not in record:
        raise ValueError(f"the agent-history record {name!r} has no schema_hash; it cannot be shown to have "
                         f"been recorded under the current tool schema {SCHEMA_HASH}")
    schema_hash = record["schema_hash"]
    if schema_hash != SCHEMA_HASH:
        raise ValueError(f"the record {name!r} was recorded under tool schema {schema_hash}, the "
                         f"current schema is {SCHEMA_HASH}")
    if "prompt_hash" not in record:
        raise RecordError(f"the agent-history record {name!r} has no prompt_hash; it cannot be shown to have "
                          f"been recorded under the current system prompt {PROMPT_HASH}")
    if record["prompt_hash"] != PROMPT_HASH:
        raise RecordError(f"the record {name!r} was recorded under system prompt {record['prompt_hash']}, "
                          f"the current prompt is {PROMPT_HASH}")
    tools = record.get("tools")
    if isinstance(tools, str):
        tools = json.loads(tools)
    if tools is None or canonical_json(tools) != canonical_json(MODEL_TOOLS):
        raise ValueError(f"the agent-history record {name!r} carries tools that are not MODEL_TOOLS; the "
                         "model would be trained on a tool block it is not shown at inference")
    return step_contexts(record["messages"]), MODEL_TOOLS


def _public_pairs(sample: dict, fallback_tools: list[dict] | None) -> tuple[Pairs, list[dict] | None]:
    messages = sample["messages"]
    tools = sample.get("tools", fallback_tools)
    if isinstance(tools, str):
        tools = json.loads(tools)
    pairs = [(messages[:i], m) for i, m in enumerate(messages) if m.get("role") == "assistant"]
    return pairs, tools

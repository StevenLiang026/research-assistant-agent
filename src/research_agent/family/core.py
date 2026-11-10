"""Model families (docs/m7-家族层设计.md §2, §3): what differs between the base models a run can train
and serve, and a family loaded for one model folder, which renders, encodes and reads text for that model.

A Family is a static specification: which chat template it is (pinned SHA-256 hashes), whether it thinks
(never, always, or on a template switch), per mode the template's keyword arguments and the suffix every
prompt ends with, the thinking block's opening and closing strings, the turn terminator, the BOS policy,
the tool-call format, how arguments that are not an object reach the template, where the model's text is
read from, the vendor code it pins, and the training and decoding defaults. load_family identifies the
family of a model folder, loads its tokenizer and checks the specification against what the template and
the tokenizer actually do, so every disagreement is a FamilyError before any sample or request is made from
them.

Template hash: the SHA-256 (lowercase hex) of the UTF-8 bytes of the template text the tokenizer exposes
as tokenizer.chat_template, or, when that is empty, of chat_template.jinja's text read as transformers
reads it (UTF-8 text with universal newlines, so a checkout with CRLF line ends hashes the same). A
tokenizer holding several named templates is refused: a family pins exactly one.

Prompts. A prompt is the template's render of the context with the generation prompt, under the mode's
keyword arguments, and it must end with the mode's declared suffix (generation_prompt), or FamilyError.
Every THINK prompt ends with think_open: a template that leaves the thinking block to the model (Qwen3's
hybrid template with enable_thinking on, and Nanbeige's, end at "<|im_start|>assistant\\n", where the model
writes "<think>\\n" itself) gets think_open appended, so the reasoning starts at the same place in training
and at inference. A family that always thinks (Thinking.ONLY) has a NO_THINK prompt all the same: its THINK
prompt followed by think_close. Its template opens a block in every generation prompt, and its NO_THINK
prompt must be exactly what precedes the answer in a render of a target without reasoning (load_family
checks it), so the model is trained to answer there without learning to write an empty block. Where that
render holds the block closed empty ("<think>\\n" + "\\n</think>\\n\\n": Qwen3-4B-Thinking-2507, Nanbeige),
the family's empty_block is "template"; where the template renders no block at all for a target without
reasoning (LFM2.5, whose render goes straight from "<|im_start|>assistant\\n" to the answer), it is
"constructed": full() in NO_THINK mode then writes think_open + think_close into the render after the
assistant header, the text the NO_THINK prompt ends with, and refuses a template that renders a block
itself there (it would be doubled).

THINK samples. A THINK sample's completion is the reasoning, think_close and the NO_THINK completion of the
target (docs/m7-家族层设计.md §5), joined as the template joins them (think_completion). Under think_join
"concat" it is the three in a row. Under "trim" the template writes the turn's thinking block and its text
as one string with the whitespace at its end trimmed, then the newline before its calls, or the terminator
(Granite's `content | trim`): a target without text of its own (its content empty or blank: a turn of calls
alone, or an empty one) ends the block at the close without its trailing whitespace, "\\n</think>" +
"\\n<tool_call>...", where the three in a row would hold a newline more ("\\n</think>\\n\\n<tool_call>...");
a target with text keeps the whole close before it, a newline the text starts with included. Either way the
text after the reasoning starts with think_close whenever the answer does not end at once, so a forced close
(think_close appended, encode_close) is followed by what the samples hold, and split_reasoning, which skips
every newline after the close, reads the answer back. With reasoning_field set, load_family holds the join
against the template's own render.

Turn terminator. turn_end is the token every assistant turn ends with and the stop string of every request,
whatever the tokenizer's eos_token is (MiniCPM5's eos is "</s>" while its turns end with "<|im_end|>"): the
load-time check that it is one token, encode_sample's rule that a sample's last id is its id, and the
server's stop strings all follow turn_end alone.

BOS (§3). Every encoding passes add_special_tokens=False, so the template's text alone decides the BOS; a
prompt (encode_prompt) and a training sample (encode_sample) hold it exactly once and first under the
"template" policy and never under "none". Under "turn" the BOS token is the one that opens every turn
(Nanbeige's and Spark's templates write it before every message), so a prompt or a sample starts with it
and it is not counted; load_family checks that the probe prompt holds it more than once. A "none" family's
tokenizer must not add one by itself either, or a tool that encodes with special tokens on would give the
model a BOS it was never trained with.

Raw arguments. api_to_hf keeps the arguments of a call that are not a JSON object as the raw text the
model wrote (a call the executor refused, shown back in a repair exchange). Under the "string" policy the
text goes to the template as the arguments, which the Qwen3 and Nanbeige templates print verbatim; under
"inline" (the templates that iterate the arguments or refuse a string: Qwen3.5, Granite, MiniCPM5, Spark,
LFM2.5), to_template_messages writes every call of such a message into its content as the family's tool
format renders them, in order, after the message's own text and one newline when there is text, and drops
the tool_calls, so the template sees plain text where the model wrote its calls. load_family renders a
probe call with raw text and requires the text in the render. tool_call_ids stays declared but
unimplemented: no registered template renders call ids, and a family declaring them is refused.

Text from ids. llama-server writes no text for a control token (its tokenizer.ggml.token_type CONTROL:
an added token the tokenizer calls special, and, by the converter's own rule, any added token shaped
<|...|> or <｜...｜>), so where a family's tool-call tags are such tokens (MiniCPM5's <function; LFM2.5's
<|tool_call_start|>) the answer must be read from the ids the server sends, not from its text
(text_from_ids; research_agent.llm.generation decodes it). load_family checks the flag against the
tokenizer's added tokens: on when a tag of the format (its opening tag, or any in the template's render of
the probe call) or of the thinking block encodes to a control token, off otherwise.

Vendor code. A family with trust_remote_code pins the SHA-256 of every Python file of its folder
(remote_files); load_family checks the files before the tokenizer loads with trust_remote_code=True, and
refuses a changed, missing or unpinned file: the code runs only as reviewed.

Model patches. Vendor code written for another transformers release can hold a class attribute the
installed one reads otherwise (Spark's modeling_spark.py declares _tied_weights_keys as a list, which
transformers 5.18 reads as a mapping from each tied parameter to its source). A family with vendor code may
declare model_patches, class attributes with the values the installed transformers needs, taken from the
model's real parameter names; build_causal_lm, the one way an HF model is built from a family (scripts/
train.py, eval.fidelity's HF backend), sets them on the class the auto class resolves for the folder
(patch_model_class: transformers' own get_class_from_dynamic_module, whose module is the folder's name and
its code's hash, so the auto class builds that very class object) before from_pretrained, and requires the
built model to be of that class. The vendor files are not changed, and their pins are checked first. A
family without patches is built by from_pretrained alone, exactly as before. The family's record lists the
patches of a family that declares any.

Module adapters. A patch mends what transformers reads of a class, not the code it runs. Vendor code that
calls the installed stack in a way it no longer takes (Spark's forward pass calls transformers' mask
functions with 4.57's keywords) or that cannot run under the training stack's quantization (Spark's decoder
layer casts its input to a quantized weight's storage dtype) is adapted at load instead: a family with
vendor code may declare module_adapters, names in its vendor module (a function the module holds, or
Class.attribute) each mapped to an adapter registered in research_agent.family.compat, which replaces it
with what the installed stack runs, after checking that it is what the adapter was written for.
patch_model_class applies them to the module of the class it resolves, before the class patches and before
from_pretrained; the vendor files stay as pinned. The family's record lists the adapters of a family that
declares any. compat's docstring has the two adapters and why each is exact (registry: spark-x2.5-4b).

Attention. A family names the attention implementation its model is built with (attn_implementation):
"sdpa", what scripts/train.py built every model with, unless the model's code has only "eager" (Spark's
vendor class computes its attention with its own eager function whatever the setting and declares no SDPA
support, so transformers refuses "sdpa" when it builds the class). build_causal_lm builds with the family's,
and refuses a caller's other choice; the record names it for a family built otherwise than with "sdpa".

load_family's checks, on a short probe conversation with the model's tools, a call target, a text target
and a target whose arguments are raw text: each mode's prompt ends with its suffix and keeps the BOS rule;
the full render of a target without reasoning begins with the NO_THINK prompt (for an ONLY family, the
empty block above) and its completion ends with the terminator, which must be one token, at a token
boundary; the tool format reads the template's render of the probe call back as the call; when
reasoning_field is set, the template's render of a target carrying its reasoning there is prompt(THINK) +
the reasoning, think_close and the NO_THINK completion as think_completion joins them, the THINK sample
docs/m7-家族层设计.md §5 builds, for the probe's call target and its text target; the BOS
facts; the raw-argument policy; the control tokens; the vendor code; and, for a family that thinks, that the
close has ids of its own after a word, the ids a THINK generation is closed with by force (encode_close).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal

from ..agent.types import ToolCall, api_to_hf
from ..tools.schemas import MODEL_TOOLS
from .toolformats import CallSpan, ToolFormat

THINK_END = "</think>"  # split_reasoning reads the reasoning up to its first occurrence
INT32_MAX = 2**31 - 1  # the largest repeat_last_n llama-server takes (Decoding: The penalty's window)
_CLOSE_ANCHOR = "a"  # the word encode_close encodes the close after: a newline or a tag never merges with it
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PYTHON_FILE = re.compile(r"[^/\\]+\.py")  # a file of the folder itself
_PIPED = re.compile(r"<[|｜][^<>]*[|｜]>")  # what llama.cpp's converter marks as a control token regardless


class FamilyError(ValueError):
    """A family specification, model folder or tokenizer that disagrees with itself or with the others."""


class TemplateError(FamilyError):
    """The chat template raised while rendering (a message shape it does not take, a jinja error): the
    cause is chained. The training path counts these apart from the other kinds (docs/m7-计划.md, S2)."""


class TokenBoundaryError(FamilyError):
    """A sample whose prompt's ids are not the first ids of prompt + completion (encode_sample)."""


class BosError(FamilyError):
    """A prompt or sample that breaks the BOS rule (module docstring: BOS)."""


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise FamilyError(message)


class Mode(StrEnum):
    """THINK: the prompt opens a thinking block for the model to fill; NO_THINK: the model answers at once
    (the prompt holds the block closed empty, or none for a family that never thinks)."""

    THINK = "think"
    NO_THINK = "no_think"


class Thinking(StrEnum):
    """A family never thinks (NONE), always does (ONLY: its template opens a block in every generation
    prompt) or does on a template switch (HYBRID)."""

    NONE = "none"
    ONLY = "only"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class Decoding:
    """Sampling settings of a request. Greedy decoding (temperature 0) sends the temperature alone, so a
    greedy request's bytes are today's (llm.local.LocalClient.complete); sampled decoding sends all five
    fields, so no server default decides any of them, and the repetition penalty after them when it is on
    (not 1.0), under llama-server's name for it, repeat_penalty (tools/server/server-schema.cpp: the
    sampler's penalty_repeat, "1.0 = disabled", the server's default as its /props report it), with its
    window, repeat_last_n, so the request of a decoding without one is the request it was before the two
    fields. A greedy decoding with another field set would silently not send it, and is refused (ValueError),
    as is a value out of range.

    The penalty's window. The penalty is the vendor's repetition_penalty (LFM2.5's generation_config.json),
    which transformers applies, at every step, to every token of the sequence so far, the prompt's included
    (RepetitionPenaltyLogitsProcessor over input_ids), a logit below 0 multiplied by it and one above divided.
    llama-server b11349 applies repeat_penalty the same way, first in its sampler chain, to the tokens among
    the last repeat_last_n it has accepted (src/llama-sampler.cpp, the penalties sampler: a ring buffer of
    repeat_last_n tokens), where it accepts the prompt's tokens before the first step (server-context.cpp,
    init_sampler) and every generated one after it; repeat_last_n is 64 unless the request names a window
    (common.h), and the request field takes 0 to INT32_MAX (server-schema.cpp, hard limits: -1, the command
    line's "the context size", is refused with an error) where 0 turns the penalty off. So the window that
    holds the whole sequence at every step is the request's reach, its prompt's ids and its max_tokens (the
    sequence at the last step is one token shorter): payload(reach=...) sends repeat_last_n = reach. The reach
    is the request's own and not the server's context size or INT32_MAX, because the sampler allocates its
    ring buffer at the window's size (four bytes a token). A decoding with the penalty on sends no request
    without its reach (ValueError), so no request falls back to the 64-token window."""

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0  # 0: off
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0  # 1.0: off, and not sent

    def __post_init__(self) -> None:
        for name in ("temperature", "top_p", "min_p", "presence_penalty", "repetition_penalty"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"decoding: {name} must be a finite number, got {value!r}")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError(f"decoding: top_k must be a whole number from 0 (0 is off), got {self.top_k!r}")
        if (self.temperature < 0 or not 0 < self.top_p <= 1 or not 0 <= self.min_p <= 1
                or self.repetition_penalty <= 0):
            raise ValueError(f"decoding settings out of range (temperature >= 0, 0 < top_p <= 1, "
                             f"0 <= min_p <= 1, repetition_penalty > 0): {self}")
        others = (self.top_p, self.top_k, self.min_p, self.presence_penalty, self.repetition_penalty)
        if self.temperature == 0 and others != (1, 0, 0, 0, 1):
            raise ValueError(f"greedy decoding sends the temperature alone, so the other settings would not "
                             f"be sent: {self}")

    def payload(self, reach: int | None = None) -> dict:
        """The request fields (class docstring); reach, the tokens the request's sequence can reach (its
        prompt's ids and its max_tokens), is the window of a repetition penalty that is on (class docstring:
        The penalty's window). ValueError for a reach that is not a whole number from 1 to INT32_MAX, and
        for a penalty that is on without one."""
        if reach is not None and (isinstance(reach, bool) or not isinstance(reach, int)
                                  or not 1 <= reach <= INT32_MAX):
            raise ValueError(f"decoding: reach must be a whole number of tokens from 1 to {INT32_MAX} (the "
                             f"bounds of llama-server's repeat_last_n), got {reach!r}")
        if self.temperature == 0:
            return {"temperature": 0.0}
        fields = {"temperature": float(self.temperature), "top_p": float(self.top_p), "top_k": self.top_k,
                  "min_p": float(self.min_p), "presence_penalty": float(self.presence_penalty)}
        if self.repetition_penalty != 1:
            if reach is None:
                raise ValueError("decoding: a repetition penalty that is on is sent with its window, "
                                 "repeat_last_n, the tokens the request can reach: pass reach")
            fields["repeat_penalty"] = float(self.repetition_penalty)
            fields["repeat_last_n"] = reach
        return fields


GREEDY = Decoding()


class _Frozen(Mapping):
    """A read-only copy of a mapping: the registry's families are shared, so their mappings must not
    change under them (item assignment is a TypeError)."""

    def __init__(self, items: Mapping) -> None:
        self._items = dict(items)

    def __getitem__(self, key: object) -> object:
        return self._items[key]

    def __iter__(self) -> Iterator:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return repr(self._items)


def _freeze(value: object) -> object:
    """A JSON value with its objects as _Frozen mappings and its arrays as tuples, so a shared family's
    nested values cannot change either."""
    if isinstance(value, Mapping):
        return _Frozen({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: object) -> object:
    """_freeze undone: a fresh JSON value of dicts and lists, the caller's to change."""
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_thaw(v) for v in value]
    return value


def _is_json(value: object) -> bool:
    """Whether value is a JSON value (objects with string keys, no NaN or infinity)."""
    try:
        json.dumps(_thaw(value), allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


@dataclass(frozen=True)
class Family:
    """A model family's static specification (module docstring). Construction checks that the fields
    agree with one another and freezes the mappings (their keys become Modes); load_family checks them
    against a model folder."""

    name: str
    architectures: tuple[str, ...]  # config.json's must all be among them
    template_sha256: frozenset[str]  # pinned (module docstring: Template hash)
    thinking: Thinking
    template_kwargs: Mapping[Mode, Mapping[str, object]] = field(hash=False)  # apply_chat_template's
    generation_prompt: Mapping[Mode, str] = field(hash=False)  # the suffix each mode's prompt ends with
    think_open: str | None  # e.g. "<think>\n": every THINK prompt ends with it
    think_close: str | None  # e.g. "\n</think>\n\n": between reasoning and answer; a forced close appends it
    turn_end: str  # the turn terminator: one token, the last of every training sample
    stop: tuple[str, ...]  # the server's stop strings
    bos: Literal["none", "template", "turn"]  # §3 and the module docstring: BOS
    tool_format: ToolFormat
    raw_arguments: Literal["string", "inline"]  # how arguments that are not an object reach the template
    tool_call_ids: bool  # the template renders call ids (no registered template does: refused at load)
    reasoning_field: str | None  # the message field the template renders reasoning from (checked at load)
    lora_targets: str | tuple[str, ...]  # module names, or one PEFT regex
    empty_block: Literal["template", "constructed"] = "template"  # module docstring: Prompts
    think_join: Literal["concat", "trim"] = "concat"  # module docstring: THINK samples
    text_from_ids: bool = False  # module docstring: Text from ids
    remote_files: Mapping[str, str] = field(default_factory=dict, hash=False)  # file name -> SHA-256
    model_patches: Mapping[str, object] = field(default_factory=dict, hash=False)  # docstring: Model patches
    module_adapters: Mapping[str, str] = field(default_factory=dict, hash=False)  # docstring: Module adapters
    attn_implementation: Literal["sdpa", "eager"] = "sdpa"  # module docstring: Attention
    trust_remote_code: bool = False
    revision: str | None = None
    decoding: Mapping[Mode, Decoding] | None = field(default=None, hash=False)  # None: greedy in each mode
    reasoning_budget: int = 1024  # R: tokens of reasoning before a THINK generation is closed by force
    inference_mode: Mode = Mode.NO_THINK

    def __post_init__(self) -> None:
        _require(isinstance(self.name, str) and bool(self.name), f"a family needs a name, got {self.name!r}")
        where = f"family {self.name}"
        _require(isinstance(self.architectures, tuple) and bool(self.architectures)
                 and all(isinstance(a, str) and a for a in self.architectures),
                 f"{where}: architectures must be a non-empty tuple of names, got {self.architectures!r}")
        _require(isinstance(self.template_sha256, frozenset) and bool(self.template_sha256)
                 and all(isinstance(h, str) and _SHA256.fullmatch(h) for h in self.template_sha256),
                 f"{where}: template_sha256 must be a non-empty frozenset of SHA-256 hex digests, got "
                 f"{self.template_sha256!r}")
        _require(isinstance(self.thinking, Thinking),
                 f"{where}: thinking must be a Thinking, got {self.thinking!r}")
        modes = self.modes
        decoding = {mode: GREEDY for mode in modes} if self.decoding is None else self.decoding
        for name, value in (("template_kwargs", self.template_kwargs),
                            ("generation_prompt", self.generation_prompt), ("decoding", decoding)):
            _require(isinstance(value, Mapping) and set(value) == set(modes),
                     f"{where}: {name} must give exactly the family's modes {[str(m) for m in modes]}, "
                     f"got {value!r}")
        _require(all(isinstance(kwargs, Mapping) and all(isinstance(k, str) for k in kwargs)
                     for kwargs in self.template_kwargs.values()),
                 f"{where}: template_kwargs must map each mode to keyword arguments, got "
                 f"{self.template_kwargs!r}")
        _require(all(isinstance(s, str) and s for s in self.generation_prompt.values()),
                 f"{where}: generation_prompt must map each mode to a non-empty suffix")
        _require(all(isinstance(d, Decoding) for d in decoding.values()),
                 f"{where}: decoding must map each mode to a Decoding, got {decoding!r}")
        self._check_thinking(where)
        _require(isinstance(self.turn_end, str) and bool(self.turn_end),
                 f"{where}: turn_end must be a non-empty string, got {self.turn_end!r}")
        _require(isinstance(self.stop, tuple) and bool(self.stop)
                 and all(isinstance(s, str) and s for s in self.stop),
                 f"{where}: stop must be a non-empty tuple of strings, got {self.stop!r}")
        _require(self.turn_end in self.stop,
                 f"{where}: turn_end {self.turn_end!r} must be among the stop strings {self.stop!r}")
        _require(self.bos in ("none", "template", "turn"),
                 f"{where}: bos must be 'none', 'template' or 'turn', got {self.bos!r}")
        _require(self.empty_block in ("template", "constructed"),
                 f"{where}: empty_block must be 'template' or 'constructed', got {self.empty_block!r}")
        _require(self.empty_block == "template" or self.thinking is Thinking.ONLY,
                 f"{where}: only a family that always thinks constructs its empty thinking block "
                 f"(empty_block 'constructed' with thinking {self.thinking})")
        _require(self.think_join in ("concat", "trim"),
                 f"{where}: think_join must be 'concat' or 'trim', got {self.think_join!r}")
        _require(self.think_join == "concat" or self.thinking is not Thinking.NONE,
                 f"{where}: think_join 'trim' joins a thinking block, which only a family that thinks has")
        _require(isinstance(self.tool_format, ToolFormat),
                 f"{where}: tool_format must be a ToolFormat (name, open_tag, render, parse, locate), got "
                 f"{self.tool_format!r}")
        _require(self.raw_arguments in ("string", "inline"),
                 f"{where}: raw_arguments must be 'string' or 'inline', got {self.raw_arguments!r}")
        _require(isinstance(self.tool_call_ids, bool),
                 f"{where}: tool_call_ids must be true or false, got {self.tool_call_ids!r}")
        _require(isinstance(self.text_from_ids, bool),
                 f"{where}: text_from_ids must be true or false, got {self.text_from_ids!r}")
        _require(self.reasoning_field is None
                 or (isinstance(self.reasoning_field, str) and bool(self.reasoning_field)
                     and self.thinking is not Thinking.NONE),
                 f"{where}: reasoning_field {self.reasoning_field!r} needs a family that thinks")
        targets = self.lora_targets
        _require((isinstance(targets, str) and bool(targets))
                 or (isinstance(targets, tuple) and bool(targets)
                     and all(isinstance(t, str) and t for t in targets)),
                 f"{where}: lora_targets must be a non-empty tuple of module names or one regex, got "
                 f"{targets!r}")
        _require(isinstance(self.trust_remote_code, bool),
                 f"{where}: trust_remote_code must be true or false, got {self.trust_remote_code!r}")
        remote = self.remote_files
        _require(isinstance(remote, Mapping)
                 and all(isinstance(k, str) and _PYTHON_FILE.fullmatch(k) and isinstance(v, str)
                         and _SHA256.fullmatch(v) for k, v in remote.items()),
                 f"{where}: remote_files must map the Python file names of the folder to SHA-256 hex "
                 f"digests, got {remote!r}")
        _require(not remote or self.trust_remote_code,
                 f"{where}: remote_files pins vendor code, but trust_remote_code is off")
        _require(bool(remote) or not self.trust_remote_code,
                 f"{where}: trust_remote_code needs the vendor code pinned in remote_files")
        patches = self.model_patches
        _require(isinstance(patches, Mapping)
                 and all(isinstance(k, str) and k.isidentifier() and _is_json(v) for k, v in patches.items()),
                 f"{where}: model_patches must map class attribute names to JSON values, got {patches!r}")
        _require(not patches or self.trust_remote_code,
                 f"{where}: model_patches set attributes of vendor code, but trust_remote_code is off")
        from .compat import ADAPTERS  # compat imports this module: imported where it is used
        adapters = self.module_adapters
        _require(isinstance(adapters, Mapping)
                 and all(isinstance(k, str) and k.count(".") <= 1
                         and all(part.isidentifier() for part in k.split("."))
                         and isinstance(v, str) and v in ADAPTERS for k, v in adapters.items()),
                 f"{where}: module_adapters must map names of the vendor module (a function, or "
                 f"Class.attribute) to registered adapters ({', '.join(sorted(ADAPTERS))}), got {adapters!r}")
        _require(not adapters or self.trust_remote_code,
                 f"{where}: module_adapters adapt vendor code, but trust_remote_code is off")
        for target, name in adapters.items():
            _require(target in ADAPTERS[name].targets,
                     f"{where}: {name} adapts {', '.join(sorted(ADAPTERS[name].targets))}, not {target}")
        _require(self.attn_implementation in ("sdpa", "eager"),
                 f"{where}: attn_implementation must be 'sdpa' or 'eager', got {self.attn_implementation!r}")
        _require(self.revision is None or (isinstance(self.revision, str) and bool(self.revision)),
                 f"{where}: revision must be None or a non-empty string, got {self.revision!r}")
        _require(isinstance(self.reasoning_budget, int) and not isinstance(self.reasoning_budget, bool)
                 and self.reasoning_budget > 0,
                 f"{where}: reasoning_budget must be a positive number of tokens, got "
                 f"{self.reasoning_budget!r}")
        _require(self.inference_mode in modes,
                 f"{where}: inference_mode {self.inference_mode} is not among its modes "
                 f"{[str(m) for m in modes]}")
        object.__setattr__(self, "template_kwargs",
                           _Frozen({Mode(m): _Frozen(kwargs) for m, kwargs in self.template_kwargs.items()}))
        object.__setattr__(self, "generation_prompt",
                           _Frozen({Mode(m): suffix for m, suffix in self.generation_prompt.items()}))
        object.__setattr__(self, "decoding", _Frozen({Mode(m): d for m, d in decoding.items()}))
        object.__setattr__(self, "remote_files", _Frozen(remote))
        object.__setattr__(self, "model_patches", _Frozen({k: _freeze(v) for k, v in patches.items()}))
        object.__setattr__(self, "module_adapters", _Frozen(adapters))
        object.__setattr__(self, "inference_mode", Mode(self.inference_mode))

    def _check_thinking(self, where: str) -> None:
        """The thinking block's strings against the thinking style and the suffixes."""
        if self.thinking is Thinking.NONE:
            _require(self.think_open is None,
                     f"{where}: a family that never thinks has no think_open, got {self.think_open!r}")
            _require(self.think_close is None or isinstance(self.think_close, str),
                     f"{where}: think_close must be a string or None, got {self.think_close!r}")
            return
        _require(isinstance(self.think_open, str) and bool(self.think_open)
                 and isinstance(self.think_close, str) and bool(self.think_close),
                 f"{where}: a family that thinks needs think_open and think_close, got "
                 f"{self.think_open!r} and {self.think_close!r}")
        think = self.generation_prompt[Mode.THINK]
        _require(think.endswith(self.think_open),
                 f"{where}: the think suffix {think!r} must end with think_open {self.think_open!r}")
        if self.thinking is Thinking.ONLY:
            expected, got = think + self.think_close, self.generation_prompt[Mode.NO_THINK]
            _require(got == expected, f"{where}: a family that always thinks has as no_think suffix its "
                                      f"think suffix followed by think_close, {expected!r}, got {got!r}")

    @property
    def modes(self) -> tuple[Mode, ...]:
        """NO_THINK alone for a family that never thinks, else both."""
        return (Mode.NO_THINK,) if self.thinking is Thinking.NONE else (Mode.THINK, Mode.NO_THINK)


@dataclass(frozen=True, eq=False)
class LoadedFamily:
    """A family with the tokenizer of one model folder, as load_family returns it (module docstring)."""

    spec: Family
    tokenizer: object
    model_dir: Path
    template_sha256: str  # the folder's, one of spec.template_sha256
    tokenizer_json_sha256: str | None  # of the folder's tokenizer.json; None when it has none
    bos_id: int | None  # the tokenizer's BOS token, None when it has none
    turn_end_id: int

    def to_template_messages(self, api: list[dict]) -> list[dict]:
        """API messages as the template's messages: research_agent.agent.types.api_to_hf, which keeps
        arguments that are not a JSON object as the raw string and drops call ids and every other field,
        the reasoning of earlier steps among them (earlier thinking is never shown, docs/m7-计划.md §2 思考);
        then the raw-argument policy (module docstring: Raw arguments). FamilyError for tool_call_ids."""
        spec = self.spec
        _require(not spec.tool_call_ids, f"{spec.name}: tool_call_ids is not implemented: no registered "
                                         "template renders call ids")
        hf = api_to_hf(api)
        if spec.raw_arguments == "inline":
            hf = [self._inline(message) for message in hf]
        return hf

    def prompt(self, hf: list[dict], tools: list[dict] | None, mode: Mode) -> str:
        """The prompt of a context (template messages) in a mode (module docstring: Prompts); FamilyError
        when it does not end with the mode's declared suffix, or when the family has no such mode."""
        spec, mode = self.spec, self._mode(mode)
        if mode is Mode.NO_THINK and spec.thinking is Thinking.ONLY:
            text = self.prompt(hf, tools, Mode.THINK) + spec.think_close
        else:
            text = self._render(hf, tools, mode, generation_prompt=True)
            if mode is Mode.THINK and not text.endswith(spec.think_open):
                text += spec.think_open
        suffix = spec.generation_prompt[mode]
        _require(text.endswith(suffix), f"{spec.name}: the {mode} prompt ends with "
                                        f"{text[-len(suffix) - 20:]!r}, not with the declared suffix "
                                        f"{suffix!r}")
        return text

    def full(self, hf: list[dict], tools: list[dict] | None, mode: Mode) -> str:
        """The template's render of a whole conversation (template messages) in a mode, without a
        generation prompt: a training sample's prompt and completion are cut from it. For a family whose
        empty thinking block is constructed, the NO_THINK render holds the block after the last assistant
        header (module docstring: Prompts)."""
        mode = self._mode(mode)
        text = self._render(hf, tools, mode, generation_prompt=False)
        if mode is Mode.NO_THINK and self.spec.empty_block == "constructed":
            text = self._with_empty_block(hf, tools, text)
        return text

    def encode_prompt(self, text: str) -> list[int]:
        """The ids of a prompt, the BOS rule checked (module docstring: BOS)."""
        ids = self._encode(text)
        self._check_bos(ids, "the prompt")
        return ids

    def encode_sample(self, prompt: str, completion: str) -> tuple[list[int], int]:
        """The ids of prompt + completion and the index of the completion's first id. The text is encoded
        whole, as the trainer's tokenizer did (TRL's prompt-completion path), and the prompt's own ids must
        begin it: a token straddling the boundary would train on a split the model never sees at inference.
        The BOS rule holds and the last id is the terminator's; FamilyError otherwise."""
        name = self.spec.name
        _require(bool(completion), f"{name}: an empty completion has nothing to train on")
        prompt_ids = self._encode(prompt)
        ids = self._encode(prompt + completion)
        start = len(prompt_ids)
        if ids[:start] != prompt_ids:
            raise TokenBoundaryError(f"{name}: the sample does not split at a token boundary: the prompt's "
                                     f"{start} ids are not the first ids of prompt + completion")
        self._check_bos(ids, "the sample")
        _require(ids[-1] == self.turn_end_id, f"{name}: the sample's last id is {ids[-1]}, not the "
                                              f"terminator {self.spec.turn_end!r} ({self.turn_end_id})")
        return ids, start

    def encode_text(self, text: str) -> list[int]:
        """The ids of a piece of text encoded on its own (a reasoning whose length is counted, a tag looked up
        among the control tokens): no BOS rule, the text is not a whole sequence. Inside a sequence the same
        text can have other ids (encode_close)."""
        return self._encode(text)

    def think_completion(self, reasoning: str, answer: str, message: dict) -> str:
        """The completion of a THINK sample: reasoning, think_close and answer, the NO_THINK completion of
        message (the target as a template message), joined as the template joins them (module docstring:
        THINK samples): under think_join "trim", a target without text of its own (its content empty or
        blank) has the close without its trailing whitespace. FamilyError for a family that never thinks."""
        spec = self.spec
        _require(spec.thinking is not Thinking.NONE, f"{spec.name} never thinks: it has no thinking block to "
                                                     "join to an answer")
        close = spec.think_close
        if spec.think_join == "trim" and not str(message.get("content") or "").strip():
            close = close.rstrip()
        return reasoning + close + answer

    def encode_close(self) -> list[int]:
        """The ids think_close has after a word, the usual case where the budget cuts a reasoning: what a
        THINK generation is closed with by force (llm.generation). They are the close's ids after an anchor
        word, the anchor's ids cut: on its own the close would start with the word-start marker a
        SentencePiece tokenizer with a dummy prefix writes at the start of a text (Nanbeige's "▁"), which no
        sample holds after a reasoning. After punctuation or a newline a sample can hold the close with other
        ids, where the vocabulary merges that character with the newline the close starts with (Qwen3's and
        MiniCPM5's after a full stop): a reasoning cut there is closed with a split of the close no sample
        holds. FamilyError for a family that never thinks, or when the close merges with the anchor (it has
        no ids of its own there)."""
        spec = self.spec
        _require(spec.thinking is not Thinking.NONE, f"{spec.name} never thinks: it has no thinking block to "
                                                     "close")
        anchor = self._encode(_CLOSE_ANCHOR)
        ids = self._encode(_CLOSE_ANCHOR + spec.think_close)
        _require(ids[:len(anchor)] == anchor,
                 f"{spec.name}: think_close {spec.think_close!r} merges with the word before it, "
                 f"{_CLOSE_ANCHOR!r}, into one token")
        return ids[len(anchor):]

    def decode(self, ids: list[int]) -> str:
        """The text of ids, special tokens included: the thinking tags and the terminator are text to
        read."""
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def split_reasoning(self, text: str) -> tuple[str | None, int]:
        """A THINK generation's reasoning and where its answer starts: the text before the first "</think>"
        without the newlines at either end (the templates render reasoning so), and the offset just after
        that tag and the newlines that follow it. (None, 0) when there is no "</think>": no reasoning was
        closed."""
        end = text.find(THINK_END)
        if end < 0:
            return None, 0
        answer = end + len(THINK_END)
        while answer < len(text) and text[answer] == "\n":
            answer += 1
        return text[:end].strip("\n"), answer

    def parse(self, answer: str, tools: list[dict] | None) -> list[dict]:
        """The calls of an answer, by the family's tool format."""
        return self.spec.tool_format.parse(answer, tools)

    def locate(self, answer: str) -> list[CallSpan]:
        """Where the calls of an answer name their tools and values, by the family's tool format."""
        return self.spec.tool_format.locate(answer)

    def record(self) -> dict:
        """What a run records about the family it ran: name, template and tokenizer.json hashes, BOS
        policy, thinking, LoRA targets, whether vendor code ran (remote) with the hashes it was pinned at,
        and the pinned revision; for a family that declares model patches or module adapters, those too
        (module docstring: Model patches, Module adapters), and for one built with another attention than
        "sdpa", that attention (module docstring: Attention), so every other family's record is what it was
        before them."""
        spec = self.spec
        targets = spec.lora_targets
        record = {"name": spec.name, "template_sha256": self.template_sha256,
                  "tokenizer_json_sha256": self.tokenizer_json_sha256, "bos": spec.bos,
                  "thinking": str(spec.thinking),
                  "lora_targets": targets if isinstance(targets, str) else list(targets),
                  "remote": spec.trust_remote_code, "remote_files": dict(spec.remote_files),
                  "revision": spec.revision}
        if spec.model_patches:
            record["model_patches"] = _thaw(spec.model_patches)
        if spec.module_adapters:
            record["module_adapters"] = dict(spec.module_adapters)
        if spec.attn_implementation != "sdpa":
            record["attn_implementation"] = spec.attn_implementation
        return record

    def _mode(self, mode: Mode) -> Mode:
        _require(mode in self.spec.modes,
                 f"{self.spec.name} has no {mode} mode (its modes: {', '.join(self.spec.modes)})")
        return Mode(mode)

    def _inline(self, message: dict) -> dict:
        """The template message with its calls written into its content by the tool format when any of
        them carries raw text (module docstring: Raw arguments); the message itself otherwise."""
        calls = message.get("tool_calls") or []
        if not any(isinstance(call["function"]["arguments"], str) for call in calls):
            return message
        text = self.spec.tool_format.render([{"name": call["function"]["name"],
                                              "arguments": call["function"]["arguments"]} for call in calls])
        content = message.get("content") or ""
        return {"role": "assistant", "content": f"{content}\n{text}" if content else text}

    def _with_empty_block(self, hf: list[dict], tools: list[dict] | None, text: str) -> str:
        """text, the template's render of a conversation ending with a target, with the empty thinking
        block written after the target's assistant header (module docstring: Prompts)."""
        spec = self.spec
        _require(bool(hf) and hf[-1].get("role") == "assistant",
                 f"{spec.name}: the empty thinking block is constructed in front of a target, so a "
                 "conversation rendered in no_think mode ends with an assistant message")
        base = self.prompt(hf[:-1], tools, Mode.THINK).removesuffix(spec.think_open)
        _require(text.startswith(base), f"{spec.name}: the template's render does not begin with the "
                                        f"assistant header of its generation prompt, {base[-40:]!r}, so the "
                                        "empty thinking block cannot be constructed")
        rest = text[len(base):]
        _require(not rest.startswith(spec.think_open),
                 f"{spec.name}: the template renders a thinking block itself after the assistant header, "
                 "so empty_block 'constructed' would write a second one; declare 'template'")
        return base + spec.think_open + spec.think_close + rest

    def _render(self, hf: list[dict], tools: list[dict] | None, mode: Mode, *,
                generation_prompt: bool) -> str:
        """The template's text; whatever the template raises is a TemplateError with that cause, so a
        caller can tell a message the template refuses from the other kinds of failure."""
        try:
            return self.tokenizer.apply_chat_template(hf, tools=tools, tokenize=False,
                                                      add_generation_prompt=generation_prompt,
                                                      **self.spec.template_kwargs[mode])
        except Exception as exc:
            raise TemplateError(f"{self.spec.name}: the chat template raised on {len(hf)} messages in "
                                f"{mode} mode: {type(exc).__name__}: {exc}") from exc

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def _check_bos(self, ids: list[int], what: str) -> None:
        spec = self.spec
        if spec.bos == "turn":
            if not ids or ids[0] != self.bos_id:
                raise BosError(f"{spec.name}: {what} does not start with the BOS token (id {self.bos_id}), "
                               "which opens every turn")
            return
        expected = 1 if spec.bos == "template" else 0
        count = 0 if self.bos_id is None else ids.count(self.bos_id)
        if count != expected:
            raise BosError(f"{spec.name}: {what} holds {count} BOS token(s); the BOS policy {spec.bos!r} "
                           f"wants {expected}")
        if expected and ids[0] != self.bos_id:  # the message names where it is, which only a failure has
            raise BosError(f"{spec.name}: {what} holds its BOS token at position {ids.index(self.bos_id)}, "
                           "not first")


def template_hash(text: str) -> str:
    """The template hash of a template's text (module docstring: Template hash)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_family(model_dir: str | Path, *, name: str | None = None, tokenizer: object | None = None,
                families: Sequence[Family] | None = None) -> LoadedFamily:
    """The family of a model folder with its tokenizer, checked (module docstring).

    The family is the one whose pinned hashes hold the template's and whose architectures hold all of
    config.json's (registry.resolve); name, the --family option, must be that family's. Without a
    tokenizer, the template the folder's files hold decides the family first (chat_template.jinja when
    there is one, as transformers takes it over tokenizer_config.json's), the family decides how the
    tokenizer loads (local files only, its trust_remote_code and revision, after its vendor code is
    checked), and the loaded tokenizer must expose that same template. A tokenizer given (one already
    loaded, or the tests' stand-ins) is used as it is; the folder's vendor code is checked all the same.
    families replaces the registry's (registry.FAMILIES) when given.
    """
    from . import registry  # registry imports this module: imported where it is used

    folder = Path(model_dir)
    known = registry.FAMILIES if families is None else tuple(families)
    architectures = _architectures(folder)
    from_files = None
    if tokenizer is None:
        from_files = template_hash(_files_template(folder))
        spec = registry.resolve(architectures, from_files, name=name, families=known)
        _check_remote_files(folder, spec)
        tokenizer = _load_tokenizer(folder, spec)
    digest = template_hash(_exposed_template(tokenizer, folder))
    _require(from_files is None or digest == from_files,
             f"{folder}: the tokenizer exposes a template hashing to {digest}, the folder's files one "
             f"hashing to {from_files}")
    spec = registry.resolve(architectures, digest, name=name, families=known)
    if from_files is None:
        _check_remote_files(folder, spec)
    ends = list(tokenizer(spec.turn_end, add_special_tokens=False)["input_ids"])
    _require(len(ends) == 1, f"{spec.name}: the terminator {spec.turn_end!r} is {len(ends)} tokens, not one")
    tokenizer_json = folder / "tokenizer.json"
    family = LoadedFamily(
        spec=spec, tokenizer=tokenizer, model_dir=folder, template_sha256=digest,
        tokenizer_json_sha256=(hashlib.sha256(tokenizer_json.read_bytes()).hexdigest()
                               if tokenizer_json.is_file() else None),
        bos_id=getattr(tokenizer, "bos_token_id", None), turn_end_id=ends[0])
    _check(family)
    return family


def folder_family(model_dir: str | Path, *, name: str | None = None,
                  families: Sequence[Family] | None = None) -> Family:
    """The family of a model folder from its files alone, as load_family finds it before it loads the
    tokenizer: config.json's architectures and the template the files hold (registry.resolve; name must be
    that family's), the folder's vendor code checked against the family's pins. FamilyError otherwise."""
    from . import registry  # registry imports this module: imported where it is used

    folder = Path(model_dir)
    known = registry.FAMILIES if families is None else tuple(families)
    spec = registry.resolve(_architectures(folder), template_hash(_files_template(folder)), name=name,
                            families=known)
    _check_remote_files(folder, spec)
    return spec


def patch_model_class(spec: Family, model_dir: str | Path) -> type | None:
    """The class transformers' AutoModelForCausalLM builds for the folder, its module adapted and the
    family's model patches set on it (module docstring: Model patches, Module adapters): the vendor files
    checked against the pins first, the class resolved as the auto class resolves remote code (the
    configuration's auto_map, get_class_from_dynamic_module), the module it was defined in adapted
    (research_agent.family.compat.adapt_vendor_module: every target checked before any is replaced, a target
    adapted already left as it is), then each attribute set to a fresh copy of its value, so an adapter's
    refusal (FamilyError) leaves the class and its module as they were. None, with nothing imported, read
    or set, for a family with neither."""
    if not spec.model_patches and not spec.module_adapters:
        return None
    _check_remote_files(Path(model_dir), spec)
    import transformers  # heavy import, only for a family whose vendor code is patched or adapted
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config = transformers.AutoConfig.from_pretrained(model_dir, trust_remote_code=spec.trust_remote_code)
    reference = (getattr(config, "auto_map", None) or {}).get("AutoModelForCausalLM")
    _require(isinstance(reference, str) and bool(reference),
             f"{spec.name}: {model_dir}'s configuration maps no AutoModelForCausalLM class to patch")
    model_class = get_class_from_dynamic_module(reference, model_dir)
    if spec.module_adapters:
        from .compat import adapt_vendor_module  # compat imports this module: imported where it is used

        module = sys.modules.get(model_class.__module__)
        _require(module is not None, f"{spec.name}: the module of {model_class.__name__}, "
                                     f"{model_class.__module__}, is not imported, so it cannot be adapted")
        adapt_vendor_module(module, spec.module_adapters, spec.name)
    for attribute, value in spec.model_patches.items():
        setattr(model_class, attribute, _thaw(value))
    return model_class


def build_causal_lm(spec: Family, model_dir: str | Path, **kwargs) -> object:
    """transformers' AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=spec.
    trust_remote_code, attn_implementation=spec.attn_implementation, **kwargs), the family's module adapters
    and model patches set on the vendor code first (patch_model_class; module docstring: Model patches,
    Module adapters, Attention). FamilyError, before anything is resolved or built, when kwargs name another
    attention than the family's; before anything is built, when an adapter refuses what it finds; and when
    the model built is not of the patched class."""
    attention = kwargs.pop("attn_implementation", spec.attn_implementation)
    _require(attention == spec.attn_implementation,
             f"{spec.name}: its model is built with the family's attention, {spec.attn_implementation!r}, "
             f"not {attention!r}")
    from transformers import AutoModelForCausalLM  # heavy import, only where a model is built

    patched = patch_model_class(spec, model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=spec.trust_remote_code,
                                                 attn_implementation=spec.attn_implementation, **kwargs)
    _require(patched is None or isinstance(model, patched),
             f"{spec.name}: transformers built {type(model).__name__} from {type(model).__module__}, not the "
             f"patched {patched and patched.__module__}.{patched and patched.__name__}")
    return model


# The probe conversation of the checks: a system prompt and a request, answered by a call, by text, or by a
# call whose arguments are raw text (a rejected call as a repair exchange shows it).
_PROBE_CONTEXT = [{"role": "system", "content": "You are a research assistant."},
                  {"role": "user", "content": "Find recent papers on speculative decoding."}]
_PROBE_TARGETS = [
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_probe", "type": "function", "function": {
        "name": "search_arxiv", "arguments": '{"query": "speculative decoding", "max_results": 5}'}}]},
    {"role": "assistant", "content": "Here are the papers I found."},
]
_PROBE_RAW_TEXT = "query=speculative decoding"
_PROBE_RAW = {"role": "assistant", "content": "", "tool_calls": [
    {"id": "call_raw", "type": "function",
     "function": {"name": "search_arxiv", "arguments": _PROBE_RAW_TEXT}}]}
_PROBE_REASONING = "The user wants recent papers on speculative decoding, so I search arXiv for them."


def _check(family: LoadedFamily) -> None:
    """load_family's checks (module docstring); FamilyError at the first disagreement."""
    spec = family.spec
    _require(spec.bos == "none" or family.bos_id is not None,
             f"{spec.name}: the BOS policy is {spec.bos!r} but the tokenizer has no BOS token")
    hf = family.to_template_messages(_PROBE_CONTEXT)  # the message policies
    for mode in spec.modes:
        ids = family.encode_prompt(family.prompt(hf, MODEL_TOOLS, mode))  # the declared suffix, the BOS rule
        if spec.bos == "turn":
            count = ids.count(family.bos_id)
            _require(count >= 2, f"{spec.name}: the BOS policy is 'turn' but the BOS token does not open "
                                 f"every turn: the probe's prompt holds it {count} time(s)")
    completions = []
    for target in _PROBE_TARGETS:
        (message,) = family.to_template_messages([target])
        prompt, completion, tail = _no_think_sample(family, hf, message)
        completions.append(completion)
        family.encode_sample(prompt, completion)  # a token boundary, the BOS rule, the terminator's id
        calls = [{"name": c.name, "arguments": c.arguments}
                 for c in (ToolCall.from_api(c) for c in target.get("tool_calls") or [])]
        read = family.parse(completion.removesuffix(spec.turn_end), MODEL_TOOLS)
        _require(read == calls, f"{spec.name}: the {spec.tool_format.name} format reads the template's "
                                f"render of the probe call back as {read!r}, not as {calls!r}")
        if spec.reasoning_field is not None:
            think = family.prompt(hf, MODEL_TOOLS, Mode.THINK)
            sample = family.think_completion(_PROBE_REASONING, completion, message)
            rendered = family.full(hf + [{**message, spec.reasoning_field: _PROBE_REASONING}], MODEL_TOOLS,
                                   Mode.THINK)
            _require(rendered == think + sample + tail,
                     f"{spec.name}: the template renders a target with its {spec.reasoning_field} as "
                     f"{rendered[len(think) - 20:]!r}, not as the THINK construction {sample + tail!r}")
            family.encode_sample(think, sample)
    (raw,) = family.to_template_messages([_PROBE_RAW])  # the raw-argument policy
    _require(_PROBE_RAW_TEXT in family.full(hf + [raw], MODEL_TOOLS, Mode.NO_THINK),
             f"{spec.name}: the render of a call whose arguments are raw text does not show the text "
             f"under the {spec.raw_arguments!r} policy")
    if spec.bos == "none" and family.bos_id is not None:
        added = family.tokenizer(_PROBE_REASONING, add_special_tokens=True)["input_ids"]
        _require(family.bos_id not in added,
                 f"{spec.name}: the BOS policy is 'none' but the tokenizer adds its BOS token by itself "
                 f"(id {family.bos_id}) when asked to add special tokens")
    _check_text_source(family, completions[0])
    if spec.thinking is not Thinking.NONE:
        family.encode_close()  # the close has ids of its own after a word


def _no_think_sample(family: LoadedFamily, hf: list[dict], message: dict) -> tuple[str, str, str]:
    """The NO_THINK prompt of the context, the completion of the message after it (the full render after
    the prompt, up to and with the terminator) and what the template writes after the terminator."""
    name, end = family.spec.name, family.spec.turn_end
    prompt = family.prompt(hf, MODEL_TOOLS, Mode.NO_THINK)
    full = family.full(hf + [message], MODEL_TOOLS, Mode.NO_THINK)
    _require(full.startswith(prompt), f"{name}: the template's render of the probe conversation does not "
                                      f"begin with the no_think prompt, which ends with {prompt[-40:]!r}; "
                                      "training text would differ from inference text")
    completion = full[len(prompt):]
    tail = ""
    if completion.endswith(end + "\n"):
        completion, tail = completion[:-1], "\n"
    _require(completion.endswith(end), f"{name}: the completion does not end with the terminator {end!r}: "
                                       f"{completion[-40:]!r}")
    return prompt, completion, tail


def _check_text_source(family: LoadedFamily, call: str) -> None:
    """text_from_ids against the tokenizer's control tokens (module docstring: Text from ids): the tags are
    the format's opening tag, the template's render of the probe call (call, its NO_THINK completion, which
    holds every tag a call is written with) without its terminator, and the thinking block's."""
    spec = family.spec
    texts = {spec.tool_format.open_tag, call.removesuffix(spec.turn_end)}
    if spec.think_open is not None:
        texts |= {spec.think_open, spec.think_close}
    control = _control_tokens(family.tokenizer)
    dropped = sorted({control[i] for text in texts for i in family.encode_text(text) if i in control})
    if spec.text_from_ids:
        _require(bool(dropped), f"{spec.name}: no tag of the tool format or the thinking block is a control "
                                "token, so the server's text is whole and text_from_ids must be off")
    else:
        _require(not dropped, f"{spec.name}: {', '.join(dropped)} encode to special tokens, which "
                              "llama-server leaves out of its text: text_from_ids must be on")


def _control_tokens(tokenizer: object) -> dict[int, str]:
    """The added tokens llama-server writes no text for (module docstring: Text from ids), id -> text."""
    decoder = getattr(tokenizer, "added_tokens_decoder", None) or {}
    return {i: token.content for i, token in decoder.items()
            if token.special or _PIPED.fullmatch(token.content)}


def _check_remote_files(folder: Path, spec: Family) -> None:
    """The folder's Python files against the family's pins (module docstring: Vendor code); nothing to
    check for a family without trust_remote_code."""
    if not spec.trust_remote_code:
        return
    for name, digest in spec.remote_files.items():
        path = folder / name
        _require(path.is_file(), f"{spec.name}: the vendor file {name} is missing from {folder}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(actual == digest, f"{spec.name}: the vendor file {name} has sha256 {actual}, not the pinned "
                                   f"{digest}: the code was reviewed at the pinned hash only")
    unpinned = sorted(p.name for p in folder.glob("*.py") if p.name not in spec.remote_files)
    _require(not unpinned, f"{spec.name}: the Python file(s) {', '.join(unpinned)} in {folder} are not "
                           "pinned in remote_files: vendor code runs only when reviewed and pinned")


def _architectures(folder: Path) -> tuple[str, ...]:
    """config.json's architectures, or text_config's for a configuration that keeps them there."""
    config = folder / "config.json"
    _require(config.is_file(), f"{folder} has no config.json")
    data = json.loads(config.read_text(encoding="utf-8"))
    text_config = data.get("text_config")
    names = data.get("architectures") or (text_config.get("architectures") if isinstance(text_config, dict)
                                          else None)
    _require(isinstance(names, list) and bool(names) and all(isinstance(n, str) and n for n in names),
             f"{config} names no architectures")
    return tuple(names)


def _files_template(folder: Path) -> str:
    """The template transformers will expose for the folder, read from its files before loading it (the
    family the template belongs to decides how the tokenizer loads): chat_template.jinja when there is
    one, else tokenizer_config.json's "chat_template"."""
    jinja = folder / "chat_template.jinja"
    if jinja.is_file():
        return jinja.read_text(encoding="utf-8")
    config = folder / "tokenizer_config.json"
    template = (json.loads(config.read_text(encoding="utf-8")).get("chat_template") if config.is_file()
                else None)
    _require(not isinstance(template, list),
             f"{folder}: tokenizer_config.json holds named chat templates; a family pins exactly one")
    _require(isinstance(template, str) and bool(template),
             f"{folder}: no chat template in chat_template.jinja or tokenizer_config.json")
    return template


def _exposed_template(tokenizer: object, folder: Path) -> str:
    """The template text the hash is of (module docstring: Template hash)."""
    template = getattr(tokenizer, "chat_template", None)
    _require(not isinstance(template, dict | list),
             f"{folder}: the tokenizer holds named chat templates; a family pins exactly one")
    if isinstance(template, str) and template:
        return template
    jinja = folder / "chat_template.jinja"
    _require(jinja.is_file(),
             f"{folder}: no chat template: the tokenizer exposes none and there is no chat_template.jinja")
    return jinja.read_text(encoding="utf-8")


def _load_tokenizer(folder: Path, spec: Family) -> object:
    from transformers import AutoTokenizer  # heavy import, only needed for a real model folder

    return AutoTokenizer.from_pretrained(folder, local_files_only=True,
                                         trust_remote_code=spec.trust_remote_code, revision=spec.revision)

"""Stand-ins for the tokenizers the family layer drives (research_agent.family), and the Family that
describes one of them truthfully. Nothing here imports transformers, and nothing here uses the family
layer's own rendering or hashing, so a test cannot pass because the code under test agrees with itself.

FakeFamilyTokenizer renders messages in the shape of the three Qwen3 chat templates, with switches for what
differs between families:

- style "hybrid" (Qwen3-1.7B's template): the generation prompt is "<|im_start|>assistant\\n", followed by
  the empty thinking block "<think>\\n\\n</think>\\n\\n" when enable_thinking is False; "only"
  (Qwen3-4B-Thinking-2507's): the generation prompt always opens a block,
  "<|im_start|>assistant\\n<think>\\n"; "none" (Qwen3-4B-Instruct-2507's): no thinking block anywhere. In
  the hybrid and only styles the last message, when it is an assistant message, starts with its block,
  "<think>\\n" + reasoning + "\\n</think>\\n\\n", the reasoning being its reasoning_content without the
  newlines at either end when renders_reasoning is on, else empty (a template that ignores the field); with
  strips_reasoning off the field is written as given, newlines and all (a template whose render of a
  reasoning with a newline at either end differs from the training path's construction); its content loses
  its leading newlines there, as Qwen3's does. With empty_block off, a last assistant message without
  reasoning gets no block at all (LFM2.5's template), only one with reasoning does.
- style "trim" (Granite 4.2's template): hybrid, the generation prompt "<|im_start|>assistant\\n" followed by
  "<think>\\n", or by the empty block "<think></think>" when enable_thinking is False. The last assistant
  message is written as one text, its block ("<think>\\n" + reasoning + "\\n</think>\\n" when it carries a
  reasoning, "<think></think>" otherwise) followed by its content, with the whitespace at the end trimmed
  (Granite's `content | trim`); then each call after a newline and followed by one, or nothing, and the
  terminator. So a turn of calls alone ends its block at "</think>" whether it is empty or holds a
  reasoning, and the calls' newline follows it.
- bos "none": the tokenizer has no BOS token; "template": the template writes the BOS text "<s>" first;
  "token": the tokenizer has the BOS token but the template never writes it; "turn": the BOS token is
  "<|im_start|>" itself, which opens every turn (Nanbeige's and Spark's templates). With auto_bos the
  tokenizer puts the BOS id first when it is asked to add special tokens (add_special_tokens=True, its
  default).
- eos: the tokenizer's eos_token, "<|im_end|>" unless given; "</s>" makes a tokenizer whose end-of-sequence
  token is not the turn terminator (MiniCPM5's).
- Tools render as one canonical-JSON <tools> block in the system turn. Tool calls render as Qwen3's
  template renders them, written out here independently: '<tool_call>\\n{"name": "' + name +
  '", "arguments": ' + arguments + '}\\n</tool_call>', the arguments as json.dumps writes them with
  ensure_ascii off (the template's tojson) or verbatim when they are a string, "\\n" between calls and after
  non-empty content; with refuses_raw the template raises on a call whose arguments are a string, as the
  templates that iterate the arguments do. Consecutive tool messages share one user turn of <tool_response>
  blocks.
- pieces: more strings the tokenizer reads as one token each, with ids from PIECE_OFFSET in their order,
  such as a newline together with the word after it, which a vocabulary can hold where nothing splits the
  two: a text then merges across a boundary that TOKENS alone keeps apart.
- added_tokens_decoder, as transformers exposes it: every string of TOKENS and the pieces as an added
  token; SPECIAL ones are special, and with special_tags the thinking and tool-call tags are too (MiniCPM5's
  tokenizer), so the family must read its text from the ids.

DummyPrefix is a FakeFamilyTokenizer that writes MARK ("▁", one token when it is among the pieces) first when
the text it encodes starts with plain text, not with one of TOKENS, as a SentencePiece tokenizer with a dummy
prefix does (Nanbeige's): a text encoded on its own starts with the marker where the same text inside a
sequence holds none.

Ids:each string of TOKENS and of the pieces is one token (the longest match first), "\\n\\n" among them;
every other character is one token, CHAR_OFFSET plus its code point. So ids are deterministic, and a "\\n"
ending one text merges with a "\\n" starting the next, as the real byte-level BPE merges them
(encode_sample's prefix check must catch it). The tokenizer keeps every render, encode and decode call with
its keyword arguments, and refuses to tokenize inside apply_chat_template: the family layer renders text
and encodes it itself.
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from research_agent.canonical import canonical_json
from research_agent.family.core import Family, Mode, Thinking
from research_agent.family.toolformats.qwen_json import QWEN_JSON

TOKENS = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2, "<s>": 3, "<think>": 4, "</think>": 5,
          "<tool_call>": 6, "</tool_call>": 7, "\n\n": 8, "</s>": 9}
SPECIAL = frozenset({"<|endoftext|>", "<|im_start|>", "<|im_end|>", "<s>", "</s>"})  # dropped by skip_special
TAGS = ("<think>", "</think>", "<tool_call>", "</tool_call>")  # special too with special_tags
PIECE_OFFSET = 100
CHAR_OFFSET = 1000
BOS, BOS_ID = "<s>", TOKENS["<s>"]
TURN_END = "<|im_end|>"
TURN_END_ID = TOKENS[TURN_END]
ASSISTANT = "<|im_start|>assistant\n"
THINK_OPEN, THINK_CLOSE = "<think>\n", "\n</think>\n\n"
TRIM_CLOSE, TRIM_EMPTY = "\n</think>\n", "<think></think>"  # the "trim" style's close and empty block
ARCHITECTURES = ("FakeForCausalLM",)
STYLES = ("hybrid", "only", "none", "trim")
BOS_MODES = ("none", "template", "token", "turn")


@functools.cache
def _vocabulary(pieces: tuple[str, ...]) -> tuple[dict[str, int], re.Pattern]:
    """The strings read as one token, TOKENS and the pieces, with their ids, and the pattern that finds
    them, the longest first."""
    words = {**TOKENS, **{piece: PIECE_OFFSET + n for n, piece in enumerate(pieces)}}
    return words, re.compile("|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)))


def token_ids(text: str, pieces: tuple[str, ...] = ()) -> list[int]:
    """text as a FakeFamilyTokenizer with these pieces reads it (module docstring)."""
    words, finder = _vocabulary(pieces)
    ids: list[int] = []
    position = 0
    for match in finder.finditer(text):
        ids += [CHAR_OFFSET + ord(c) for c in text[position:match.start()]]
        ids.append(words[match.group()])
        position = match.end()
    return ids + [CHAR_OFFSET + ord(c) for c in text[position:]]


def sha256_text(text: str) -> str:
    """The template hash, computed here without the family layer: SHA-256 of the text's UTF-8 bytes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def call_block(name: str, arguments: object) -> str:
    """One call as the Qwen3 templates write it (module docstring)."""
    text = arguments if isinstance(arguments, str) else json.dumps(arguments, ensure_ascii=False)
    return '<tool_call>\n{"name": "' + name + '", "arguments": ' + text + "}\n</tool_call>"


class FakeFamilyTokenizer:
    """A Qwen3-shaped chat template and tokenizer with the module docstring's switches."""

    def __init__(self, style: str = "hybrid", *, bos: str = "none", eos: str = TURN_END,
                 renders_reasoning: bool = True, strips_reasoning: bool = True, empty_block: bool = True,
                 refuses_raw: bool = False, special_tags: bool = False, auto_bos: bool = False,
                 chat_template: object = None, pieces: tuple[str, ...] = ()) -> None:
        if style not in STYLES or bos not in BOS_MODES or eos not in TOKENS:
            raise ValueError(f"unknown switch: style {style!r}, bos {bos!r}, eos {eos!r}")
        if len(set(pieces)) != len(pieces) or any(p in TOKENS for p in pieces):
            raise ValueError(f"pieces must be distinct and none of TOKENS, got {pieces!r}")
        self.style, self.bos, self.renders_reasoning, self.auto_bos = style, bos, renders_reasoning, auto_bos
        self.strips_reasoning, self.empty_block = strips_reasoning, empty_block
        self.refuses_raw, self.special_tags = refuses_raw, special_tags
        self.pieces = tuple(pieces)
        self.chat_template = (chat_template if chat_template is not None else
                              f"fake template: style={style} bos={bos} renders_reasoning={renders_reasoning} "
                              f"strips_reasoning={strips_reasoning} empty_block={empty_block} "
                              f"refuses_raw={refuses_raw}")
        self.eos_token = eos
        self.eos_token_id = TOKENS[eos]
        self.bos_token = {"none": None, "turn": "<|im_start|>"}.get(bos, BOS)
        self.bos_token_id = None if self.bos_token is None else TOKENS[self.bos_token]
        self.renders: list[dict] = []  # the keyword arguments of every apply_chat_template call
        self.encoded: list[tuple[str, dict]] = []  # (text, keyword arguments) of every encode
        self.decoded: list[tuple[list[int], dict]] = []  # (ids, keyword arguments) of every decode

    @property
    def added_tokens_decoder(self) -> dict[int, SimpleNamespace]:
        special = SPECIAL | (set(TAGS) if self.special_tags else set())
        words = _vocabulary(self.pieces)[0]
        return {i: SimpleNamespace(content=w, special=w in special) for w, i in words.items()}

    def apply_chat_template(self, messages: list[dict], *, tools: list[dict] | None = None,
                            tokenize: bool = True, add_generation_prompt: bool = False, **kwargs) -> str:
        if tokenize:
            raise AssertionError("the family layer must render text (tokenize=False) and encode it itself")
        self.renders.append({"tools": tools, "add_generation_prompt": add_generation_prompt, **kwargs})
        out = BOS if self.bos == "template" else ""
        system = (messages[0].get("content") or "") if messages and messages[0]["role"] == "system" else None
        if tools:
            head = "" if system is None else system + "\n\n"
            out += f"<|im_start|>system\n{head}<tools>{canonical_json(tools)}</tools>{TURN_END}\n"
        elif system is not None:
            out += f"<|im_start|>system\n{system}{TURN_END}\n"
        last = len(messages) - 1
        for index, message in enumerate(messages):
            role, content = message["role"], message.get("content") or ""
            if role == "system" and index == 0:
                continue
            if role in ("user", "system"):
                out += f"<|im_start|>{role}\n{content}{TURN_END}\n"
            elif role == "assistant":
                out += self._assistant(message, last=index == last)
            elif role == "tool":
                if index == 0 or messages[index - 1]["role"] != "tool":
                    out += "<|im_start|>user"
                out += f"\n<tool_response>\n{content}\n</tool_response>"
                if index == last or messages[index + 1]["role"] != "tool":
                    out += f"{TURN_END}\n"
        if add_generation_prompt:
            out += ASSISTANT
            if self.style == "only":
                out += THINK_OPEN
            elif self.style == "hybrid" and kwargs.get("enable_thinking") is False:
                out += THINK_OPEN + THINK_CLOSE
            elif self.style == "trim":
                out += TRIM_EMPTY if kwargs.get("enable_thinking") is False else THINK_OPEN
        return out

    def _assistant(self, message: dict, *, last: bool) -> str:
        content = message.get("content") or ""
        out = ASSISTANT
        if last and self.style == "trim":
            return out + self._trimmed(message, content)
        if last and self.style != "none":
            reasoning = message.get("reasoning_content") if self.renders_reasoning else None
            if not isinstance(reasoning, str):
                reasoning = ""
            elif self.strips_reasoning:
                reasoning = reasoning.strip("\n")
            if reasoning or self.empty_block:
                out += THINK_OPEN + reasoning + THINK_CLOSE
                content = content.lstrip("\n")
        out += content
        for number, call in enumerate(message.get("tool_calls") or []):
            if number or content:
                out += "\n"
            function = call.get("function", call)
            if self.refuses_raw and isinstance(function["arguments"], str):
                raise TypeError("Can only get item pairs from a mapping.")
            out += call_block(function["name"], function["arguments"])
        return out + f"{TURN_END}\n"

    def _trimmed(self, message: dict, content: str) -> str:
        """The last assistant message of the "trim" style after its header (module docstring)."""
        reasoning = message.get("reasoning_content") if self.renders_reasoning else None
        if isinstance(reasoning, str) and reasoning.strip():
            reasoning = reasoning.strip("\n") if self.strips_reasoning else reasoning
            block = THINK_OPEN + reasoning + TRIM_CLOSE
        else:
            block = TRIM_EMPTY
        out = (block + content).strip()
        calls = message.get("tool_calls") or []
        if calls:
            out += "\n"
        for call in calls:
            function = call.get("function", call)
            if self.refuses_raw and isinstance(function["arguments"], str):
                raise TypeError("Can only get item pairs from a mapping.")
            out += call_block(function["name"], function["arguments"]) + "\n"
        return out + f"{TURN_END}\n"

    def __call__(self, text: str, *, add_special_tokens: bool = True, **kwargs) -> dict:
        self.encoded.append((text, {"add_special_tokens": add_special_tokens, **kwargs}))
        ids = token_ids(text, self.pieces)
        if add_special_tokens and self.auto_bos and self.bos_token_id is not None:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}

    def decode(self, ids: list[int], *, skip_special_tokens: bool = False, **kwargs) -> str:
        self.decoded.append((list(ids), {"skip_special_tokens": skip_special_tokens, **kwargs}))
        names = {i: w for w, i in _vocabulary(self.pieces)[0].items()}
        return "".join(names[i] if i in names else chr(i - CHAR_OFFSET) for i in ids
                       if not (skip_special_tokens and names.get(i) in SPECIAL))


MARK = "▁"  # a SentencePiece word-start marker


class DummyPrefix(FakeFamilyTokenizer):
    """A tokenizer with a dummy prefix (module docstring)."""

    def __call__(self, text: str, **kwargs) -> dict:
        ids = super().__call__(text, **kwargs)["input_ids"]
        starts_with_a_token = any(text.startswith(t) for t in TOKENS) or not text
        return {"input_ids": ids if starts_with_a_token else [token_ids(MARK, self.pieces)[0], *ids]}


def fake_family(tokenizer: FakeFamilyTokenizer, **changes) -> Family:
    """The Family that describes tokenizer truthfully (its template's hash, thinking style, suffixes, BOS
    policy, empty block, raw-argument policy, how its template joins the close to an answer and where its
    text comes from); changes replace any field, to describe it wrongly on purpose."""
    thinking = {"hybrid": Thinking.HYBRID, "only": Thinking.ONLY, "none": Thinking.NONE,
                "trim": Thinking.HYBRID}[tokenizer.style]
    thinks = thinking is not Thinking.NONE
    trims = tokenizer.style == "trim"
    if not thinks:
        template_kwargs: dict = {Mode.NO_THINK: {}}
        generation_prompt = {Mode.NO_THINK: ASSISTANT}
    else:
        switch = thinking is Thinking.HYBRID
        template_kwargs = {Mode.THINK: {"enable_thinking": True} if switch else {},
                           Mode.NO_THINK: {"enable_thinking": False} if switch else {}}
        generation_prompt = {Mode.THINK: ASSISTANT + THINK_OPEN,
                             Mode.NO_THINK: ASSISTANT + (TRIM_EMPTY if trims else THINK_OPEN + THINK_CLOSE)}
    fields = {
        "name": f"fake-{tokenizer.style}", "architectures": ARCHITECTURES,
        "template_sha256": frozenset({sha256_text(tokenizer.chat_template)}), "thinking": thinking,
        "template_kwargs": template_kwargs, "generation_prompt": generation_prompt,
        "think_open": THINK_OPEN if thinks else None, "think_close": TRIM_CLOSE if trims else THINK_CLOSE,
        "think_join": "trim" if trims else "concat",
        "turn_end": TURN_END, "stop": (TURN_END,),
        "bos": {"template": "template", "turn": "turn"}.get(tokenizer.bos, "none"),
        "empty_block": "template" if tokenizer.empty_block or not thinks else "constructed",
        "tool_format": QWEN_JSON, "raw_arguments": "inline" if tokenizer.refuses_raw else "string",
        "tool_call_ids": False, "text_from_ids": tokenizer.special_tags,
        "reasoning_field": "reasoning_content" if tokenizer.renders_reasoning and thinks else None,
        "lora_targets": ("q_proj", "v_proj"),
    }
    return Family(**{**fields, **changes})


def model_folder(path: Path, *, architectures: tuple[str, ...] | None = ARCHITECTURES,
                 text_config: tuple[str, ...] | None = None, template: str | None = None,
                 jinja: str | None = None, tokenizer_json: bytes | None = None,
                 python: dict[str, str] | None = None) -> Path:
    """A model folder: config.json naming the architectures (at the top level, or under text_config as
    multimodal configurations keep them), and when given the chat template in tokenizer_config.json, a
    chat_template.jinja, a tokenizer.json and Python files (name -> source), as a folder with vendor code
    holds them."""
    path.mkdir(parents=True, exist_ok=True)
    config: dict = {"model_type": "fake"}
    if architectures is not None:
        config["architectures"] = list(architectures)
    if text_config is not None:
        config["text_config"] = {"architectures": list(text_config)}
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    if template is not None:
        (path / "tokenizer_config.json").write_text(json.dumps({"chat_template": template}), encoding="utf-8")
    if jinja is not None:
        (path / "chat_template.jinja").write_bytes(jinja.encode("utf-8"))
    if tokenizer_json is not None:
        (path / "tokenizer.json").write_bytes(tokenizer_json)
    for name, source in (python or {}).items():
        (path / name).write_bytes(source.encode("utf-8"))  # the bytes a pin hashes, no newline translation
    return path


def fake_transformers(monkeypatch, tokenizer: object) -> list[tuple[tuple, dict]]:
    """A transformers module in sys.modules for this test whose AutoTokenizer.from_pretrained returns
    tokenizer; returns the (args, kwargs) of every from_pretrained call."""
    loads: list[tuple[tuple, dict]] = []

    def from_pretrained(*args, **kwargs) -> object:
        loads.append((args, kwargs))
        return tokenizer

    module = types.ModuleType("transformers")
    module.AutoTokenizer = SimpleNamespace(from_pretrained=from_pretrained)
    monkeypatch.setitem(sys.modules, "transformers", module)
    return loads

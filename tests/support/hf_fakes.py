"""Stand-ins for what research_agent.llm.hf drives: a transformers causal LM, PEFT's wrapper around one and
the Qwen3 tokenizer, plus a PEFT adapter folder on disk. Nothing here imports torch: ids are plain lists where
the real objects hold tensors.

FakeHFTokenizer reads a text one character per token, except its special and added tokens, one token each.
The special ones (<|endoftext|>, <|im_start|>, <|im_end|>) are the ones Qwen3's tokenizer flags special, so
decode(skip_special_tokens=True) leaves them out; <tool_call> and </tool_call> are added tokens that are not
special there, so they stay, as llama-server renders them. FakeCausalLM continues every prompt with its reply
and then its end token, the way a greedy generate does: it stops after an id in eos_token_id or after
max_new_tokens ids, and returns the prompt's ids followed by the new ones. FakePeftModel holds two replies,
the adapter's and the base model's, and gives the base one inside disable_adapter().

fake_stack puts torch, transformers and peft modules into sys.modules for llm.hf.load_client, which imports
them when it loads: every from_pretrained call is kept with its arguments, and a model it returns (Pretrained)
records the eval() and merge_and_unload() calls made on it.
"""

from __future__ import annotations

import contextlib
import re
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

SPECIAL = {"<|endoftext|>": 0, "<|im_start|>": 1, "<|im_end|>": 2}
ADDED = {"<tool_call>": 3, "</tool_call>": 4}
TOKENS = {**SPECIAL, **ADDED}
NAMES = {i: t for t, i in TOKENS.items()}
END_IDS = [2, 0]  # <|im_end|> and <|endoftext|>: Qwen3's generation config eos_token_id, in its order
CHAR_OFFSET = 1000  # a character's id is CHAR_OFFSET plus its code point
_PIECES = re.compile("(" + "|".join(re.escape(t) for t in TOKENS) + ")")


def token_ids(text: str) -> list[int]:
    """text as FakeHFTokenizer reads it."""
    ids: list[int] = []
    for piece in _PIECES.split(text):
        ids += [TOKENS[piece]] if piece in TOKENS else [CHAR_OFFSET + ord(c) for c in piece]
    return ids


class Encoding(dict):
    """A BatchEncoding stand-in: one row of input_ids and of attention_mask; to() records the device."""

    device: str | None = None

    def to(self, device: str) -> Encoding:
        self.device = device
        return self


class FakeHFTokenizer:
    eos_token = "<|im_end|>"

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []  # (text, keyword arguments) per call
        self.decoded: list[tuple[list[int], dict]] = []  # (ids, keyword arguments) per decode
        self.encodings: list[Encoding] = []

    def __call__(self, text: str, **kwargs) -> Encoding:
        self.calls.append((text, kwargs))
        ids = token_ids(text)
        self.encodings.append(Encoding(input_ids=[ids], attention_mask=[[1] * len(ids)]))
        return self.encodings[-1]

    def decode(self, ids: list[int], skip_special_tokens: bool = False, **kwargs) -> str:
        self.decoded.append((list(ids), {"skip_special_tokens": skip_special_tokens, **kwargs}))
        return "".join(NAMES[i] if i in NAMES else chr(i - CHAR_OFFSET) for i in ids
                       if not (skip_special_tokens and NAMES.get(i) in SPECIAL))


class FakeCausalLM:
    """Greedily continues any prompt with reply and then end (module docstring); every generate call's keyword
    arguments are kept in calls."""

    def __init__(self, reply: str, *, end: str = "<|im_end|>") -> None:
        self.reply, self.end = reply, end
        self.generation_config = SimpleNamespace(eos_token_id=list(END_IDS))
        self.calls: list[dict] = []

    def continuation(self) -> str:
        return self.reply

    def generate(self, *, input_ids: list[list[int]], attention_mask: list[list[int]],
                 **kwargs) -> list[list[int]]:
        self.calls.append({"input_ids": input_ids, "attention_mask": attention_mask, **kwargs})
        new: list[int] = []
        for token in token_ids(self.continuation() + self.end):
            if len(new) == kwargs["max_new_tokens"]:
                break
            new.append(token)
            if token in kwargs["eos_token_id"]:
                break
        return [input_ids[0] + new]


class FakePeftModel(FakeCausalLM):
    """PEFT's wrapper over FakeCausalLM: the adapter's reply, or the base model's inside disable_adapter();
    every generate call records whether the adapter was enabled."""

    def __init__(self, adapter_reply: str, base_reply: str, **kwargs) -> None:
        super().__init__(adapter_reply, **kwargs)
        self.base_reply, self.enabled = base_reply, True

    @contextlib.contextmanager
    def disable_adapter(self) -> Iterator[None]:
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = True

    def continuation(self) -> str:
        return self.reply if self.enabled else self.base_reply

    def generate(self, **kwargs) -> list[list[int]]:
        output = super().generate(**kwargs)
        self.calls[-1]["adapter_enabled"] = self.enabled
        return output


def peft_folder(path: Path, weights: bytes = b"safetensors weights") -> Path:
    """A PEFT adapter folder as scripts/train.py saves one (adapter/, or a trainer checkpoint folder)."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text('{"peft_type": "LORA", "r": 16}\n', encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(weights)
    return path


BFLOAT16 = SimpleNamespace(name="torch.bfloat16")  # the fake torch's dtype
FAKE_GPU = "Fake GPU 8 GB"


class Pretrained:
    """A model a fake from_pretrained returned: the arguments it was given and the calls made on it."""

    def __init__(self, args: tuple, kwargs: dict) -> None:
        self.args, self.kwargs = args, kwargs
        self.generation_config = SimpleNamespace(eos_token_id=list(END_IDS))
        self.calls: list[str] = []

    def eval(self) -> Pretrained:
        self.calls.append("eval")
        return self

    def merge_and_unload(self) -> object:
        self.calls.append("merge_and_unload")
        return self.args[0]


def fake_stack(monkeypatch) -> dict[str, list]:
    """torch (a GPU, bfloat16), transformers (AutoTokenizer, AutoModelForCausalLM) and peft (PeftModel) as
    fake modules in sys.modules for this test; returns their from_pretrained calls, kept under "tokenizer"
    as (args, kwargs) and under "model" and "peft" as the Pretrained objects they returned."""
    loads: dict[str, list] = {"tokenizer": [], "model": [], "peft": []}

    def tokenizer(*args, **kwargs) -> FakeHFTokenizer:
        loads["tokenizer"].append((args, kwargs))
        return FakeHFTokenizer()

    def loader(kind: str):
        def from_pretrained(*args, **kwargs) -> Pretrained:
            loads[kind].append(Pretrained(args, kwargs))
            return loads[kind][-1]
        return from_pretrained

    torch = types.ModuleType("torch")
    torch.bfloat16 = BFLOAT16
    torch.cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda index: FAKE_GPU)
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=tokenizer)
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=loader("model"))
    peft = types.ModuleType("peft")
    peft.PeftModel = SimpleNamespace(from_pretrained=loader("peft"))
    for name, module in (("torch", torch), ("transformers", transformers), ("peft", peft)):
        monkeypatch.setitem(sys.modules, name, module)
    return loads

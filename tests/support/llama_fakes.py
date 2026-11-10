"""Stand-ins for what the inference path drives (research_agent.llm.generation, agent.models.LocalModel's
family path and eval.systems' local systems): a llama-server behind httpx.MockTransport, scripted completion
by completion, and a family loaded over tests/support/family_fakes.py's tokenizer. Nothing here comes from the
code under test: the text, tokens, ids, bytes, log-probabilities and finish reason of every completion are
what the script says, and the requests are kept as the client sent them.

FakeLlama answers /health (200), /props (the props given), /lora-adapters (the adapters given) and
/v1/completions with the next Scripted completion of its script, in b11349's format: choices[0].text,
finish_reason and logprobs.content, one item per listed token with its id, token, logprob, UTF-8 bytes and
one top_logprobs entry; usage (prompt_tokens: the number of ids of an id prompt, or of the ids the fake
tokenizer reads a text prompt as, so a text prompt and its ids count alike; completion_tokens: every token
generated, as the server's n_decoded counts them) and timings (prompt_n, predicted_n, predicted_ms). With
omit_ids the items carry no id, as an older server's would. The body of every completion request goes to
requests, decoded from the bytes the client sent, and the bytes themselves to request_bytes, so a test can
hold them against the bytes LocalClient sent before M7. A request past the end of the script is answered
500, which LocalClient raises as httpx.HTTPStatusError.

Which tokens are listed, as b11349 lists them (tools/server/server-context.cpp, process_token, and
server-task.cpp, probs_vector_to_json, read at the tag): a token after which the text generated so far ends
inside a UTF-8 character (validate_utf8: a lead byte among the last four bytes without all its continuation
bytes) is left out of the list, and the next listed token carries the bytes of the tokens left out before it
with its own, its string being those bytes as text. The text is the bytes of every token decoded, a
character cut off at the end, or bytes that are no UTF-8, written as U+FFFD, as the server's JSON writer
replaces them. A completion a stop string ended (stop_type "word") holds the tokens that spelled the string
neither in its text nor in the list (Scripted.trimmed). A token given as str is whole UTF-8, so a script of
str tokens is listed one item per token, as before.

The generated tokens: when a request asks for them ("return_tokens" and "verbose" true, what
LocalClient.complete(return_tokens=True) sends), the response carries "__verbose" with the native response's
"tokens" (the id of every token generated, the ones left out of the list and the stop string's included) and
"stop_type" ("eos", "word" or "limit"), narrowed to "response_fields" when the request names them, as
json_get_nested_values narrows the server's. With return_tokens=False the fake answers as a build without
the field would, without "__verbose".

A Scripted completion is one server answer: its tokens (one per token generated: a str, its text, or bytes,
its UTF-8 bytes, which may hold part of a character), the finish reason, and optionally its ids (ID_BASE plus
the token's index when not given), log-probabilities (LOGPROB each when not given), its stop type (by default
"limit" when cut, "word" when a stop string ended it, "eos" when its last token has no bytes: the end token
the server stopped on) and how many of its last tokens spelled the stop string that ended it (trimmed).

fake_loaded_family loads a fake family over a fake model folder directly (load_family with the tokenizer and
the family given); install_fake_family makes the folder loadable the way the scripts load it, by path alone:
the folder holds the template, a transformers stand-in in sys.modules returns the tokenizer, and the family
is registered in place of the registry's.

FakeToolFormat is a family's tool format that is not Qwen3's: it reads Qwen3's JSON blocks (what the fake
template writes, so load_family's probe round trip holds) but renders calls its own way, "[call NAME
ARGUMENTS]" per call, and locates them as the locate given does (QWEN_JSON's when none); it keeps every text
handed to its parse and its locate (parsed, located). A test can so tell the family's format from the Qwen3
default wherever the code under test must follow the family's, and see which text each reading was given.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from research_agent.canonical import canonical_json
from research_agent.family import registry
from research_agent.family.core import LoadedFamily, load_family
from research_agent.family.toolformats import CallSpan
from research_agent.family.toolformats.qwen_json import QWEN_JSON
from research_agent.llm.local import LocalClient
from support.family_fakes import FakeFamilyTokenizer, fake_family, fake_transformers, model_folder, token_ids

ID_BASE = 50_000  # a scripted token's id when the script gives none: ID_BASE + its index in the completion
LOGPROB = -0.1
PROPS = {"build_info": "fake-b11349", "total_slots": 1, "default_generation_settings": {"n_ctx": 8192}}


STOP_TYPES = ("eos", "word", "limit")


def utf8_cut(data: bytes) -> int:
    """The length of data without a character cut off at its end: b11349's validate_utf8
    (tools/server/server-common.cpp), which looks at the last four bytes for a lead byte without all its
    continuation bytes after it."""
    n = len(data)
    for i in range(1, min(4, n) + 1):
        c = data[n - i]
        if (c & 0xE0 == 0xC0 and i < 2) or (c & 0xF0 == 0xE0 and i < 3) or (c & 0xF8 == 0xF0 and i < 4):
            return n - i
    return n


@dataclass(frozen=True)
class Scripted:
    """One completion as the fake server answers it (module docstring)."""

    tokens: tuple[str | bytes, ...]
    finish_reason: str = "stop"
    ids: tuple[int, ...] | None = None
    logprobs: tuple[float, ...] | None = None
    stop_type: str | None = None
    trimmed: int = 0

    def __post_init__(self) -> None:
        for name, values in (("ids", self.ids), ("logprobs", self.logprobs)):
            if values is not None and len(values) != len(self.tokens):
                raise ValueError(f"a scripted completion's {name} must match its tokens one to one")
        if self.stop_type is not None and self.stop_type not in STOP_TYPES:
            raise ValueError(f"a scripted completion's stop type is one of {STOP_TYPES}, "
                             f"got {self.stop_type!r}")
        if not 0 <= self.trimmed <= len(self.tokens) or (self.trimmed and self.finish_reason != "stop"):
            raise ValueError("a scripted completion trims some of its own tokens, and only when a stop "
                             "string ended it")

    @property
    def pieces(self) -> list[bytes]:
        """Every token's UTF-8 bytes."""
        return [t if isinstance(t, bytes) else t.encode("utf-8") for t in self.tokens]

    @property
    def shown(self) -> int:
        """How many tokens the text and the list hold: all but the ones that spelled a stop string."""
        return len(self.tokens) - self.trimmed

    @property
    def text(self) -> str:
        return b"".join(self.pieces[:self.shown]).decode("utf-8", errors="replace")

    @property
    def token_ids(self) -> list[int]:
        return list(self.ids) if self.ids is not None else [ID_BASE + i for i in range(len(self.tokens))]

    @property
    def token_logprobs(self) -> list[float]:
        return list(self.logprobs) if self.logprobs is not None else [LOGPROB] * len(self.tokens)

    @property
    def kind(self) -> str:
        """The stop type the server reports (module docstring)."""
        if self.stop_type is not None:
            return self.stop_type
        if self.finish_reason == "length":
            return "limit"
        return "word" if self.trimmed or (self.tokens and self.pieces[-1]) else "eos"

    def listed(self) -> list[tuple[int, bytes, float]]:
        """(id, bytes, logprob) of every listed token (module docstring): the tokens after which the text ends
        inside a character are left out, and the next listed one carries their bytes."""
        out, generated, sent = [], b"", 0
        for token_id, piece, logprob in list(zip(self.token_ids, self.pieces, self.token_logprobs,
                                                 strict=True))[:self.shown]:
            generated += piece
            if utf8_cut(generated) < len(generated):
                continue  # left out: the text ends inside a character
            out.append((token_id, generated[sent:], logprob))
            sent = len(generated)
        return out


def scripted(text: str, finish_reason: str = "stop") -> Scripted:
    """A completion of one token holding the whole text (none for an empty text)."""
    return Scripted((text,) if text else (), finish_reason)


class FakeToolFormat:
    """A tool format that is not Qwen3's (module docstring)."""

    name = "fake_format"
    open_tag = "[call "

    def __init__(self, locate: Callable[[str], list[CallSpan]] | None = None) -> None:
        self._locate = locate
        self.parsed: list[str] = []  # every text handed to parse, in order
        self.located: list[str] = []  # every text handed to locate, in order

    def render(self, calls: list[dict]) -> str:
        return "\n".join(f"[call {call['name']} {canonical_json(call['arguments'])}]" for call in calls)

    def parse(self, text: str, tools: list[dict] | None) -> list[dict]:
        self.parsed.append(text)
        return QWEN_JSON.parse(text, tools)

    def locate(self, text: str) -> list[CallSpan]:
        self.located.append(text)
        return QWEN_JSON.locate(text) if self._locate is None else self._locate(text)


class FakeLlama:
    """A llama-server stand-in (module docstring)."""

    def __init__(self, script: list[Scripted], *, adapters: list[dict] | None = None,
                 props: dict | None = None, omit_ids: bool = False, return_tokens: bool = True) -> None:
        self.script = list(script)
        self.adapters = [] if adapters is None else list(adapters)
        self.props = PROPS if props is None else props
        self.omit_ids = omit_ids
        self.return_tokens = return_tokens
        self.requests: list[dict] = []
        self.request_bytes: list[bytes] = []

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def client(self, **kwargs) -> LocalClient:
        return LocalClient(transport=self.transport(), **kwargs)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/props":
            return httpx.Response(200, json=self.props)
        if path == "/lora-adapters":
            return httpx.Response(200, json=self.adapters)
        if path != "/v1/completions":
            return httpx.Response(404, text="not found")
        self.request_bytes.append(request.content)
        body = json.loads(request.content)
        self.requests.append(body)
        if not self.script:
            return httpx.Response(500, json={"error": "the script is over"})
        item = self.script.pop(0)
        prompt = body["prompt"]
        prompt_n = len(prompt) if isinstance(prompt, list) else len(token_ids(prompt))
        content = []
        for tid, data, lp in item.listed():
            token = data.decode("utf-8", errors="replace")
            entry = {"id": tid, "token": token, "logprob": lp, "bytes": list(data),
                     "top_logprobs": [{"id": tid, "token": token, "logprob": lp}]}
            if self.omit_ids:
                del entry["id"]
            content.append(entry)
        n = len(item.tokens)
        answer = {
            "choices": [{"text": item.text, "finish_reason": item.finish_reason,
                         "logprobs": {"content": content}}],
            "usage": {"prompt_tokens": prompt_n, "completion_tokens": n, "total_tokens": prompt_n + n},
            "timings": {"prompt_n": prompt_n, "predicted_n": n, "predicted_ms": 10.0 * n}}
        if self.return_tokens and body.get("return_tokens") and body.get("verbose"):
            native = {"tokens": item.token_ids, "stop_type": item.kind}
            fields = body.get("response_fields")
            answer["__verbose"] = {k: v for k, v in native.items() if not fields or k in fields}
        return httpx.Response(200, json=answer)


def fake_loaded_family(folder: Path, style: str = "hybrid", **changes) -> LoadedFamily:
    """A family of the given template style loaded from a fake model folder made at folder, described by
    fake_family with changes (decoding, inference_mode, reasoning_budget...)."""
    tokenizer = FakeFamilyTokenizer(style)
    return load_family(model_folder(folder), tokenizer=tokenizer,
                       families=[fake_family(tokenizer, **changes)])


def install_fake_family(monkeypatch, folder: Path, style: str = "hybrid", **changes) -> FakeFamilyTokenizer:
    """A fake model folder at folder that load_family(folder) resolves by path alone (module docstring);
    returns its tokenizer."""
    tokenizer = FakeFamilyTokenizer(style)
    model_folder(folder, template=tokenizer.chat_template)
    fake_transformers(monkeypatch, tokenizer)
    monkeypatch.setattr(registry, "FAMILIES", (fake_family(tokenizer, **changes),))
    return tokenizer

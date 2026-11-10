"""llama-server client: completions with a per-request LoRA scale and per-token log-probabilities.

The caller renders prompts with the same Hugging Face chat template used to build the training samples
(hf_renderer), so the text the model sees at inference matches training text exactly, or sends the prompt as
the token ids of that text (the family path, research_agent.llm.generation: the server then adds nothing, no
BOS, and continues a generation from the ids it gave back).

Decoding (docs/m7-家族层设计.md §7). A request names its temperature; the sampling fields (top_p, top_k,
min_p, presence_penalty) and the seed are sent only when given, so a greedy request without them is byte for
byte the request this client sent before M7 (the evaluation records of the Qwen3 hybrid family in NO_THINK
mode depend on it), and a sampled request names every one of them, so no server default decides any. A
repetition penalty (repeat_penalty, llama-server's name: tools/server/server-schema.cpp, 1.0 = disabled) and
its window (repeat_last_n: the last tokens it covers, the prompt's among them; 64 unless given, 0 turns the
penalty off) are sent after them when given, which research_agent.family.core.Decoding does only when the
penalty is on, the window then holding the whole sequence the request can reach (Decoding: The penalty's
window).

The generated tokens (return_tokens). llama-server b11349 leaves out of the logprob list every token after
which the text generated so far ends inside a UTF-8 character (tools/server/server-context.cpp,
process_token: it adds a token to the list only when validate_utf8 finds no character cut at the end of the
text; the next listed token carries the bytes and the text of the ones left out before it), so the ids the
list holds (Completion.token_ids) can miss tokens the model generated: a character split across tokens
leaves out all but its last. /v1/completions holds no other list of the tokens; its native response does
("tokens", every generated id when the request sets return_tokens, server-task.cpp to_json_non_oaicompat),
and the request field "verbose" adds that native response to the OpenAI one under "__verbose", narrowed by
"response_fields" (json_get_nested_values). complete(return_tokens=True) sends the three fields after all
the others, asking for "tokens" and "stop_type" ("eos": the last generated token is the end token the server
stopped on; "word": a stop string ended the completion, erased from the text, while the logprob list lost as
many of its last items as the string has tokens encoded on its own and "tokens" still holds every one;
"limit": the token budget), and the Completion keeps them (generated_ids, stop_type). Sampling reads none of
the three (server-schema.cpp; server-context.cpp reads return_tokens only to keep the ids, server-task.cpp
verbose and response_fields only to write the response), so the tokens are those the same request without
them generates; a request without return_tokens sends none of them.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

# New tokens per completion at most: llama-server's max_tokens here, the hf backend's max_new_tokens (llm.hf).
MAX_TOKENS = 512


@dataclass
class Completion:
    text: str
    tokens: list[str] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    top_logprobs: list[dict[str, float]] = field(default_factory=list)
    finish_reason: str | None = None
    timings: dict | None = None  # llama.cpp server timings, when the server reports them
    wall_ms: float = 0.0
    raw: dict = field(default_factory=dict)
    # The UTF-8 bytes of each token, as llama-server sends them with every logprobs content item; None when
    # it did not send them for every token. They map tokens to text positions (agent.route) where a token
    # holds part of a character, which the token strings cannot show.
    token_bytes: list[list[int]] | None = None
    # The id of each token, as llama-server sends it with every logprobs content item; empty when it did not
    # send one for every token. The list leaves out the tokens after which the text ends inside a character
    # (module docstring: The generated tokens).
    token_ids: list[int] = field(default_factory=list)
    # The id of every token generated, in order, the ones the logprob list leaves out included, and the stop
    # type, as the server sends them when the request asks (return_tokens); None when it did not send them. A
    # generation goes on from these ids (llm.generation).
    generated_ids: list[int] | None = None
    stop_type: str | None = None


def parse_logprobs(lp: dict | None) -> tuple[list[str], list[float], list[dict[str, float]]]:
    """Tokens, their log-probabilities and the top alternatives, from either response format.

    llama-server: {"content": [{"token", "logprob", "top_logprobs": [{"token", "logprob"}, ...]}, ...]}
    legacy OpenAI: {"tokens": [...], "token_logprobs": [...], "top_logprobs": [{token: logprob}, ...]}
    """
    if not lp:
        return [], [], []
    if "content" in lp:
        items = lp["content"] or []
        return ([it["token"] for it in items], [it["logprob"] for it in items],
                [{alt["token"]: alt["logprob"] for alt in it.get("top_logprobs") or []} for it in items])
    return (list(lp.get("tokens") or []), list(lp.get("token_logprobs") or []),
            [dict(d or {}) for d in lp.get("top_logprobs") or []])


def parse_token_bytes(lp: dict | None) -> list[list[int]] | None:
    """The UTF-8 bytes of each token from llama-server's format ({"content": [{"token", "bytes", ...}, ...]});
    None for the legacy format, no logprobs, or a content item without a list of bytes."""
    items = (lp or {}).get("content") if isinstance(lp, dict) else None
    if not items:
        return None
    out = []
    for item in items:
        data = item.get("bytes") if isinstance(item, dict) else None
        if not isinstance(data, list) or not all(isinstance(b, int) and 0 <= b < 256 for b in data):
            return None
        out.append(list(data))
    return out


def parse_token_ids(lp: dict | None) -> list[int]:
    """The id of each token from llama-server's format ({"content": [{"token", "id", ...}, ...]}); empty for
    the legacy format, no logprobs, or a content item without a whole-number id."""
    items = (lp or {}).get("content") if isinstance(lp, dict) else None
    if not items:
        return []
    out = []
    for item in items:
        value = item.get("id") if isinstance(item, dict) else None
        if not isinstance(value, int) or isinstance(value, bool):
            return []
        out.append(value)
    return out


# What a request with return_tokens adds after its other fields (module docstring: The generated tokens).
GENERATED_FIELDS = {"return_tokens": True, "verbose": True, "response_fields": ("tokens", "stop_type")}


def parse_generated(data: dict) -> tuple[list[int] | None, str | None]:
    """The ids of every generated token and the stop type from a response's "__verbose" (module docstring:
    The generated tokens); None for either when it is absent or not a list of whole numbers and a string."""
    native = data.get("__verbose")
    if not isinstance(native, dict):
        return None, None
    tokens, stop_type = native.get("tokens"), native.get("stop_type")
    whole = isinstance(tokens, list) and all(isinstance(t, int) and not isinstance(t, bool) for t in tokens)
    if not whole:
        tokens = None
    return tokens, stop_type if isinstance(stop_type, str) else None


class LocalClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080", *, timeout: float = 120.0,
                 transport: httpx.BaseTransport | None = None) -> None:
        # trust_env=False: requests to the local server never use connection settings from the environment.
        self.http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport, trust_env=False)
        self._adapters: list | None = None
        # llama.cpp's timings of the latest completion, None until one comes back; every complete() clears it
        # first, so a failed request leaves none. The runners' timing sidecars read it (eval.timing).
        self.last_timings: dict | None = None

    def adapters(self) -> list:
        """LoRA adapters loaded in the server (fetched once)."""
        if self._adapters is None:
            response = self.http.get("/lora-adapters")
            self._adapters = response.json() if response.status_code == 200 else []
        return self._adapters

    def complete(self, prompt: str | list[int], *, lora_scale: float | None = None,
                 max_tokens: int = MAX_TOKENS, temperature: float = 0.0, logprobs: int | None = 5,
                 stop: list[str] | None = None, cache_prompt: bool = False, seed: int | None = None,
                 top_p: float | None = None, top_k: int | None = None, min_p: float | None = None,
                 presence_penalty: float | None = None, repeat_penalty: float | None = None,
                 repeat_last_n: int | None = None, return_tokens: bool = False) -> Completion:
        """lora_scale 0.0 gives the base model, 1.0 the fine-tuned one, when the server has an adapter loaded.

        prompt is the text, or its token ids, which the server takes as they are (module docstring). seed,
        top_p, top_k, min_p, presence_penalty, repeat_penalty and repeat_last_n are sent when given and left
        out when None (module docstring: Decoding). return_tokens asks for the ids of every generated token
        and the stop type (module docstring: The generated tokens); the Completion holds them when the server
        sent them.

        cache_prompt False (the default) has the server evaluate every prompt from scratch instead of reusing
        the prefix a slot holds from an earlier request: the reuse changes how a prompt is split into batches,
        and so, now and then, a greedy token (measured 2026-10-15: two runs of the 20 end-to-end dev tasks
        with b0 differed in 4 trajectories with the reuse on). Evaluations need runs that repeat exactly.

        Without the field, llama-server b11349 applies loaded adapters at their global scale (1.0, even
        when it was started with --lora-init-without-apply; measured in spike S1b), so a base-model request
        that forgot the scale would silently use the adapter. That case raises instead. A server without an
        adapter serves the base model only: lora_scale 0.0 is then the base model and no "lora" field is
        sent (there is no adapter id 0 to name), and a scale above 0 raises, so that a run labelled with the
        adapter cannot silently measure the base model.
        """
        self.last_timings = None
        loaded = self.adapters()
        if lora_scale is None and loaded:
            raise ValueError("this server has a LoRA adapter loaded: "
                             "pass lora_scale=0.0 (base) or 1.0 explicitly")
        if lora_scale is not None and lora_scale != 0.0 and not loaded:
            raise ValueError(f"lora_scale {lora_scale:g} needs an adapter, and this server has none loaded "
                             "(scripts/serve_local.py --adapter starts one with it)")
        payload: dict = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature,
                         "cache_prompt": cache_prompt}
        if logprobs:
            payload["logprobs"] = logprobs
        if lora_scale is not None and loaded:
            payload["lora"] = [{"id": 0, "scale": lora_scale}]
        if stop:
            payload["stop"] = stop
        for name, value in (("seed", seed), ("top_p", top_p), ("top_k", top_k), ("min_p", min_p),
                            ("presence_penalty", presence_penalty), ("repeat_penalty", repeat_penalty),
                            ("repeat_last_n", repeat_last_n)):
            if value is not None:
                payload[name] = value
        if return_tokens:
            payload.update(GENERATED_FIELDS)
        t0 = time.perf_counter()
        response = self.http.post("/v1/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        wall_ms = (time.perf_counter() - t0) * 1000
        self.last_timings = data.get("timings")
        choice = data["choices"][0]
        tokens, lps, tops = parse_logprobs(choice.get("logprobs"))
        generated, stop_type = parse_generated(data)
        return Completion(text=choice.get("text", ""), tokens=tokens, logprobs=lps, top_logprobs=tops,
                          finish_reason=choice.get("finish_reason"), timings=data.get("timings"),
                          wall_ms=round(wall_ms, 1), raw=data,
                          token_bytes=parse_token_bytes(choice.get("logprobs")),
                          token_ids=parse_token_ids(choice.get("logprobs")), generated_ids=generated,
                          stop_type=stop_type)

    def props(self) -> dict | None:
        """The server's /props (model path, build, context size, chat template), recorded with every
        evaluation run; None when the server does not answer it with a JSON object."""
        response = self.http.get("/props")
        if response.status_code != 200:
            return None
        try:
            data = response.json()
        except ValueError:
            return None
        return data if isinstance(data, dict) else None

    def health(self) -> bool:
        try:
            return self.http.get("/health").status_code == 200
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        self.http.close()


def hf_renderer(model_dir: str | Path) -> Callable[[list[dict], list[dict] | None], str]:
    """Prompt renderer with the model's own chat template, thinking off, generation prompt appended."""
    from transformers import AutoTokenizer  # heavy import, only needed when rendering locally

    tok = AutoTokenizer.from_pretrained(model_dir)

    def render(messages: list[dict], tools: list[dict] | None = None) -> str:
        return tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)

    return render

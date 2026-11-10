"""DeepSeek through its OpenAI-format API, with the response cache, the spending ledger and retries.

The key is read by name from the process environment and is never logged or stored.

Thinking mode (api-docs.deepseek.com/guides/thinking_mode, read 2026-10-15). DeepSeek's models think by
default, and while thinking they ignore temperature, presence_penalty and frequency_penalty (top_p has a
floor of 0.95). A thinking request that carries tools must pass back the reasoning_content of every earlier
assistant turn of its history, or the API answers HTTP 400 (enforced since the morning of 2026-10-15, when
the dev B2 run met it). chat decides the mode of each request in this order:

1. The plain request (model, messages, tools when there are any, the parameters: what this client sent before
   thinking was handled) is looked up in the cache. Every answer cached under a plain key was given in
   thinking mode, and a hit is served as it is, whatever the history.
2. On a miss, when reasoning_of knows the reasoning of every earlier assistant turn (vacuously when there is
   none), the request is sent in thinking mode with that reasoning attached to the earlier turns of what goes
   on the wire, and only there: the answer is cached and charged under the plain key, and the caller's
   messages are not modified.
3. Otherwise (a handwritten or drafted turn, a turn of another model or run, an empty reasoning) thinking is
   turned off with extra_body (THINKING_OFF), and no reasoning is sent, neither an empty nor an invented one.
   The switch is part of the request and so of its cache key: an answer given without thinking is never
   served for a thinking request, and every answer is served in the mode it was given in.

The caller does not choose the mode of a request: an extra_body with a "thinking" key of its own is a
ValueError, raised before anything is looked up, since its answer would be labelled thinking and cached under
the plain key whatever the switch said. A client made with thinking="off" (M7, docs/m7-计划.md) skips rules 1
and 2 for every request: each is sent as in rule 3, with THINKING_OFF and no reasoning, and only answers given
without thinking are served to it; thinking="auto", the default, is the order above.

A client made with off_peak_only asks the ledger (Budget.check_off_peak) before every paid call and raises its
PeakHours in DeepSeek's peak hours; cache hits are served at any time. The ledger is told when a paid call
started, so a call that starts in the peak hours is booked at the peak price.

The ledger is checked after the lookups and before the SDK is touched. last_mode is the mode of the latest
answer obtained, served or paid (a paid one also when the ledger raises BudgetExceeded after charging it);
None until there is one, and every chat call starts it at None.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

import openai

from .budget import Budget
from .cache import ResponseCache, request_key
from .retry import call_with_retries

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
THINKING = "thinking"
NON_THINKING = "non-thinking"
THINKING_OFF = {"thinking": {"type": "disabled"}}  # the extra_body that turns DeepSeek's thinking off
THINKING_MODES = ("auto", "off")  # the client's thinking setting (module docstring)

ReasoningOf = Callable[[dict], str | None]  # an earlier assistant message -> its reasoning; None when unknown


class MissingKeyError(RuntimeError):
    pass


def is_retryable(exc: BaseException) -> bool:
    """Rate limits, timeouts, dropped connections and 5xx are retried; other client errors are not."""
    if isinstance(exc, (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)):
        return True  # APITimeoutError is a subclass of APIConnectionError
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500


class CloudClient:
    def __init__(self, model: str, *, cache: ResponseCache, budget: Budget, client: object | None = None,
                 base_url: str = DEEPSEEK_BASE_URL, key_env: str = "DEEPSEEK_API_KEY", timeout: float = 60.0,
                 retryable: Callable[[BaseException], bool] = is_retryable,
                 sleep: Callable[[float], None] = time.sleep, thinking: str = "auto",
                 off_peak_only: bool = False) -> None:
        if model not in budget.prices:  # refuse before any paid call rather than after it
            raise KeyError(f"no price for model {model!r}; add it to the price list first")
        if thinking not in THINKING_MODES:
            raise ValueError(f"thinking must be one of {THINKING_MODES}, not {thinking!r}")
        if off_peak_only and budget.off_peak is None:
            raise ValueError("off_peak_only needs a ledger with the peak table (Budget(off_peak=...))")
        if client is None:
            key = os.environ.get(key_env)
            if not key:
                raise MissingKeyError(f"{key_env} is not set in this process environment")
            client = openai.OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)
        self.model, self.cache, self.budget, self.client = model, cache, budget, client
        self.retryable, self.sleep = retryable, sleep
        self.thinking, self.off_peak_only = thinking, off_peak_only
        self.calls = 0  # paid calls made by this client
        self.cache_hits = 0
        # THINKING or NON_THINKING: the mode of the latest answer obtained, served or paid; None until there
        # is one (every chat call starts it at None)
        self.last_mode: str | None = None

    def chat(self, messages: list[dict], tools: list[dict] | None = None, *,
             reasoning_of: ReasoningOf | None = None, **params) -> dict:
        """One chat-completions request, returned as a plain dict; served from the cache when possible.

        reasoning_of gives the reasoning of an earlier assistant message (without it, none is known); the
        module docstring says how the mode is decided. It never enters the request or its cache key.
        ValueError for an extra_body with a "thinking" key: the client decides the mode.
        """
        self.last_mode = None
        if "thinking" in (params.get("extra_body") or {}):
            raise ValueError("extra_body must not carry 'thinking': CloudClient decides the thinking mode "
                             "itself (module docstring), and labels and caches every answer by that decision")
        payload = {"model": self.model, "messages": messages, **({"tools": tools} if tools else {}), **params}
        key = request_key(payload)
        cached = None if self.thinking == "off" else self.cache.get(key)
        if cached is not None:  # a plain key: the answer was given in thinking mode
            return self._served(cached, THINKING)
        earlier = [m for m in messages if m.get("role") == "assistant"]
        known = [None if reasoning_of is None else reasoning_of(m) for m in earlier]
        if self.thinking == "auto" and all(isinstance(r, str) and r for r in known):
            sent, mode = {**payload, "messages": _with_reasoning(messages, known)}, THINKING
        else:
            payload = {**payload, "extra_body": {**(params.get("extra_body") or {}), **THINKING_OFF}}
            key = request_key(payload)
            cached = self.cache.get(key)
            if cached is not None:
                return self._served(cached, NON_THINKING)
            sent, mode = payload, NON_THINKING
        self.budget.check()
        if self.off_peak_only:
            self.budget.check_off_peak()
        started = self.budget.now()
        response = call_with_retries(lambda: self.client.chat.completions.create(**sent),
                                     retryable=self.retryable, sleep=self.sleep)
        data = response.model_dump(exclude_none=True)
        self.calls += 1
        self.last_mode = mode
        # cached before charging, so a budget stop never loses a paid answer
        self.cache.put(key, self.model, data)
        self.budget.charge(self.model, data.get("usage") or {}, request_key=key, started=started)
        return data

    def _served(self, cached: dict, mode: str) -> dict:
        self.cache_hits += 1
        self.last_mode = mode
        return cached


def _with_reasoning(messages: list[dict], reasoning: list[str]) -> list[dict]:
    """messages as sent in thinking mode: a new list in which the i-th assistant message is a copy carrying
    reasoning[i] as its reasoning_content; the other messages are the caller's, unmodified."""
    attached = iter(reasoning)
    return [{**m, "reasoning_content": next(attached)} if m.get("role") == "assistant" else m
            for m in messages]

"""A strict stand-in for DeepSeek's chat completions in thinking mode, for the tests of
research_agent.llm.cloud and its callers: the real openai SDK over an httpx2 mock transport, so every request
is serialised as it is on the wire and every answer is parsed as the SDK parses DeepSeek's.

It behaves as DeepSeek's thinking-mode guide says (api-docs.deepseek.com/guides/thinking_mode, read
2026-10-15) and as the dev B2 run of 2026-10-15 met it: a request thinks unless its body turns thinking off
("thinking": {"type": "disabled"}, which extra_body adds), and a thinking request that carries tools is
refused with HTTP 400 and the message that run recorded unless every earlier assistant message carries a
non-empty reasoning_content. Answers are scripted, one per accepted request, in order; a thinking answer
carries its reasoning_content and reasoning tokens, a non-thinking one neither.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx2
import openai

from research_agent.llm.cloud import DEEPSEEK_BASE_URL

# DeepSeek's message in the dev B2 records (runs/eval/dev-b2.jsonl), with a fixed request id
MISSING_REASONING = ("The `reasoning_content` in the thinking mode must be passed back to the API. "
                     "(request_id: fake)")
USAGE = {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050, "prompt_cache_hit_tokens": 0,
         "prompt_cache_miss_tokens": 1000}


@dataclass(frozen=True)
class Answer:
    """One scripted answer: its text, its calls as (id, name, arguments as a dict or as raw text) and the
    reasoning a thinking request gets with it ("" is the empty reasoning 336 answers of the M2 cache have)."""

    content: str = ""
    calls: tuple[tuple[str, str, dict | str], ...] = ()
    reasoning: str = "先想一想。"


def thinks(body: dict) -> bool:
    """Whether DeepSeek answers this request body in thinking mode, its default."""
    return (body.get("thinking") or {}).get("type") != "disabled"


def sent_reasoning(body: dict) -> list[str | None]:
    """The reasoning_content of every assistant message of a request body, in order; None where absent."""
    return [m.get("reasoning_content") for m in body["messages"] if m.get("role") == "assistant"]


def _arguments(args: dict | str) -> str:
    return args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)


def completion(answer: Answer, thinking: bool, model: str = "deepseek-flash", number: int = 1) -> dict:
    """The response body DeepSeek sends for answer; number makes its id and creation time."""
    message: dict = {"role": "assistant", "content": answer.content}
    if answer.calls:
        message["tool_calls"] = [{"id": call_id, "type": "function",
                                  "function": {"name": name, "arguments": _arguments(args)}}
                                 for call_id, name, args in answer.calls]
    usage = dict(USAGE)
    if thinking:
        message["reasoning_content"] = answer.reasoning
        usage["completion_tokens_details"] = {"reasoning_tokens": 30 if answer.reasoning else 0}
    return {"id": f"chatcmpl-{number}", "object": "chat.completion", "created": 1760520000 + number,
            "model": model, "usage": usage,
            "choices": [{"index": 0, "finish_reason": "tool_calls" if answer.calls else "stop",
                         "message": message}]}


class StrictDeepSeek:
    """The fake endpoint. bodies holds every request body it received, refused ones included."""

    def __init__(self, answers: list[Answer] | None = None, model: str = "deepseek-flash") -> None:
        self.answers, self.model = list(answers or []), model
        self.bodies: list[dict] = []
        self.refused = 0

    def sdk(self) -> openai.OpenAI:
        """openai.OpenAI on this endpoint: a dummy key, no retries, an HTTP client that reads no environment
        settings."""
        http = httpx2.Client(transport=httpx2.MockTransport(self.handle), trust_env=False)
        return openai.OpenAI(api_key="fake", base_url=DEEPSEEK_BASE_URL, http_client=http, max_retries=0)

    def handle(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        thinking = thinks(body)
        if thinking and body.get("tools") and not all(sent_reasoning(body)):
            self.refused += 1
            return httpx2.Response(400, json={"error": {"message": MISSING_REASONING,
                                                        "type": "invalid_request_error", "param": None,
                                                        "code": "invalid_request_error"}})
        if not self.answers:  # a 4xx, which CloudClient does not retry
            return httpx2.Response(422, json={"error": {"message": "the fake endpoint has no answer left"}})
        answer = self.answers.pop(0)
        return httpx2.Response(200, json=completion(answer, thinking, self.model, len(self.bodies)))

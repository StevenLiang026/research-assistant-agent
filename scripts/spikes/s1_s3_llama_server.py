"""Spikes S1 and S3 against a running llama-server.

S1: health; special tokens stay single tokens; chat endpoint returns parsed tool calls with thinking off.
S3: /v1/completions and native /completion return per-token probabilities.

Run (server started separately):
  .venv/Scripts/python.exe scripts/spikes/s1_s3_llama_server.py [base_url]
Writes runs/spikes/s1_s3.json.
"""

from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
MODEL = Path("models/Qwen3-1.7B")
OUT = Path("runs/spikes/s1_s3.json")

TOOLS = [{"type": "function", "function": {
    "name": "search_arxiv",
    "description": "Search arXiv for papers matching a query, optionally within a date range.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "description": "Search terms"},
        "date_from": {"type": "string", "description": "YYYY-MM-DD"},
        "date_to": {"type": "string", "description": "YYYY-MM-DD"},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 20}},
        "required": ["query"]}}}]
MESSAGES = [
    {"role": "system", "content": "你是科研助理。需要检索论文时调用工具。"},
    {"role": "user", "content": "帮我找 2026 年 9 月关于推测解码的 3 篇论文。"},
]
TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def post(path: str, payload: dict, timeout: float = 120.0) -> tuple[dict, float]:
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode("utf-8"))
    return body, (time.perf_counter() - t0) * 1000


def main() -> None:
    report: dict = {"base": BASE}
    with urllib.request.urlopen(BASE + "/health", timeout=30) as r:
        report["health"] = r.status
    print("health:", report["health"])

    tok_resp, _ = post("/tokenize", {"content": "<|im_start|>assistant\n", "with_pieces": True})
    pieces = tok_resp.get("tokens", [])
    report["tokenize_im_start"] = pieces
    print("tokenize '<|im_start|>assistant\\n':", pieces)

    chat, ms = post("/v1/chat/completions", {
        "messages": MESSAGES, "tools": TOOLS, "temperature": 0, "max_tokens": 256,
        "chat_template_kwargs": {"enable_thinking": False}})
    msg = chat["choices"][0]["message"]
    report["chat_tool_calls"] = msg.get("tool_calls")
    report["chat_content"] = msg.get("content")
    report["chat_ms"] = round(ms)
    print("chat tool_calls:", json.dumps(msg.get("tool_calls"), ensure_ascii=False))
    print("chat content:", repr(msg.get("content")), f"({ms:.0f} ms)")

    hf_tok = AutoTokenizer.from_pretrained(MODEL)
    prompt = hf_tok.apply_chat_template(MESSAGES, tools=TOOLS, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
    comp, ms = post("/v1/completions", {"prompt": prompt, "temperature": 0, "max_tokens": 128,
                                        "logprobs": 5})
    choice = comp["choices"][0]
    text = choice.get("text", "")
    lp = choice.get("logprobs")
    report["completions_text"] = text
    report["completions_logprobs_present"] = bool(lp)
    report["completions_logprobs_sample"] = (json.dumps(lp, ensure_ascii=False)[:600] if lp else None)
    report["completions_ms"] = round(ms)
    m = TOOL_CALL.search(text)
    parsed = None
    if m:
        try:
            parsed = json.loads(m.group(1))
        except json.JSONDecodeError as e:
            parsed = f"JSONDecodeError: {e}"
    report["completions_tool_call"] = parsed
    print("completions text:", repr(text[:200]), f"({ms:.0f} ms)")
    print("completions logprobs present:", bool(lp))
    print("parsed tool call:", json.dumps(parsed, ensure_ascii=False))

    native, ms = post("/completion", {"prompt": prompt, "temperature": 0, "n_predict": 32, "n_probs": 5})
    probs = native.get("completion_probabilities")
    report["native_probs_present"] = bool(probs)
    report["native_probs_sample"] = json.dumps(probs[:3], ensure_ascii=False)[:600] if probs else None
    print("native /completion probabilities present:", bool(probs))
    timings = native.get("timings", {})
    report["native_timings"] = timings
    print("native timings:", json.dumps(timings))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

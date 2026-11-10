"""Spike S1 (second half): per-request LoRA switching in llama-server.

Precondition: llama-server started with `--lora <adapter.gguf> --lora-init-without-apply`.
Checks: GET /lora-adapters lists the adapter; the same prompt at scale 0 and scale 1 gives different
first-token probabilities; scale 0 matches the plain base model (a request without the lora field).

Run: <python with transformers> scripts/spikes/s1b_lora_switch.py [base_url]
Writes runs/spikes/s1b_lora_switch.json.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from spike_tools import TOOLS  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
OUT = Path("runs/spikes/s1b_lora_switch.json")
MESSAGES = [
    {"role": "system", "content": "你是科研助理。按需调用工具,参数必须符合 Schema;不确定时追问。"},
    {"role": "user", "content": "帮我找 2026 年 9 月关于 speculative decoding 的 3 篇论文并记笔记。"},
]


def call(path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def first_tokens(prompt: str, lora: list | None) -> dict:
    payload = {"prompt": prompt, "temperature": 0, "n_predict": 48, "n_probs": 3}
    if lora is not None:
        payload["lora"] = lora
    r = call("/completion", payload)
    probs = r.get("completion_probabilities") or []
    first = probs[0] if probs else {}
    return {"text": r.get("content", ""), "first_token": first.get("token"),
            "first_logprob": first.get("logprob"), "top": first.get("top_logprobs")}


def main() -> None:
    tok = AutoTokenizer.from_pretrained("models/Qwen3-1.7B")
    prompt = tok.apply_chat_template(MESSAGES, tools=TOOLS, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    adapters = call("/lora-adapters")
    plain = first_tokens(prompt, None)
    off = first_tokens(prompt, [{"id": 0, "scale": 0.0}])
    on = first_tokens(prompt, [{"id": 0, "scale": 1.0}])
    report = {"adapters": adapters, "plain": plain, "scale0": off, "scale1": on,
              "scale0_equals_plain": off["text"] == plain["text"] and off["first_logprob"] == plain["first_logprob"],
              "scale1_differs": on["text"] != off["text"] or on["first_logprob"] != off["first_logprob"]}
    print(json.dumps({k: report[k] for k in ("adapters", "scale0_equals_plain", "scale1_differs")},
                     ensure_ascii=False, indent=2))
    print("scale 0:", repr(off["text"][:120]), off["first_logprob"])
    print("scale 1:", repr(on["text"][:120]), on["first_logprob"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

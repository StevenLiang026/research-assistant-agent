"""A stand-in for the Qwen3 tokenizer: renders messages deterministically and counts tokens by whitespace.

The render keeps the two properties build_step_samples relies on, which the real template has (the
model-backed test checks them): the context plus the generation prompt is a prefix of the full render, and an
assistant turn ends with the terminator followed by a newline. A different generation prompt breaks the first
property on purpose, to test that it is checked.
"""

from __future__ import annotations

from research_agent.canonical import canonical_json

EOS = "<|im_end|>"
GENERATION_PROMPT = "<|im_start|>assistant\n"


class FakeTokenizer:
    eos_token = EOS

    def __init__(self, generation_prompt: str = GENERATION_PROMPT) -> None:
        self.generation_prompt = generation_prompt

    def apply_chat_template(self, messages: list[dict], *, tools: list[dict] | None = None,
                            tokenize: bool = False, add_generation_prompt: bool = False, **kwargs) -> str:
        parts = []
        if tools:
            parts.append(f"<|im_start|>system\n<tools>{canonical_json(tools)}</tools>{EOS}\n")
        for m in messages:
            body = m.get("content") or ""
            for c in m.get("tool_calls") or []:
                body += f"\n<tool_call>{canonical_json(c['function'])}</tool_call>"
            parts.append(f"<|im_start|>{m['role']}\n{body}{EOS}\n")
        if add_generation_prompt:
            parts.append(self.generation_prompt)
        return "".join(parts)

    def __call__(self, text: str, **kwargs) -> dict:
        return {"input_ids": list(range(len(text.split())))}

    def decode(self, ids: list[int]) -> str:
        return " ".join(str(i) for i in ids)

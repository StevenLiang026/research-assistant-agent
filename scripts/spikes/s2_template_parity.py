"""Spike S2: does TRL's Qwen3 training template render a multi-step tool trajectory the same way
as the inference template, and are Chinese arguments rendered as raw UTF-8?

Run: .venv/Scripts/python.exe scripts/spikes/s2_template_parity.py
Writes runs/spikes/s2_template_parity.json and prints a diff plus the trained (masked-in) text.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path

import trl
from transformers import AutoTokenizer

MODEL = Path("models/Qwen3-1.7B")
OUT = Path("runs/spikes/s2_template_parity.json")

TOOLS = [
    {"type": "function", "function": {
        "name": "search_arxiv",
        "description": "Search arXiv papers.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "date_from": {"type": "string"},
            "date_to": {"type": "string"},
            "max_results": {"type": "integer"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_paper",
        "description": "Get metadata and abstract of one arXiv paper.",
        "parameters": {"type": "object", "properties": {"arxiv_id": {"type": "string"}},
                       "required": ["arxiv_id"]}}},
]

MESSAGES = [
    {"role": "system", "content": "你是科研助理,按需调用工具。"},
    {"role": "user", "content": "帮我找 2026 年 9 月关于推测解码的论文,挑一篇看摘要。"},
    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
        "name": "search_arxiv",
        "arguments": {"query": "推测解码 speculative decoding", "date_from": "2026-09-01",
                      "date_to": "2026-09-30", "max_results": 3}}}]},
    {"role": "tool", "content": json.dumps([{"arxiv_id": "2609.01234", "title": "Example title"}],
                                           ensure_ascii=False)},
    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
        "name": "get_paper", "arguments": {"arxiv_id": "2609.01234"}}}]},
    {"role": "tool", "content": json.dumps({"arxiv_id": "2609.01234", "abstract": "Example abstract."},
                                           ensure_ascii=False)},
    {"role": "assistant", "content": "这篇论文提出了一种推测解码方法。"},
]


def find_training_template(tok) -> str | None:
    """Ask TRL which training template it would swap in for this tokenizer (None = no swap needed)."""
    from trl.chat_template_utils import get_training_chat_template

    return get_training_chat_template(tok)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    infer_full = tok.apply_chat_template(MESSAGES, tools=TOOLS, tokenize=False, enable_thinking=False)
    # What the model sees when generating step 3 at inference time: history up to the last tool output.
    infer_step3 = tok.apply_chat_template(MESSAGES[:-1], tools=TOOLS, tokenize=False,
                                          add_generation_prompt=True, enable_thinking=False)
    tpl = find_training_template(tok)
    root = Path(trl.__file__).parent / "chat_templates"
    name = next((p.name for p in sorted(root.glob("*_training.jinja"))
                 if tpl is not None and p.read_text(encoding="utf-8") == tpl), None)
    report: dict = {"trl_version": trl.__version__, "training_template": name}
    print("TRL", trl.__version__, "training template chosen by TRL:", name)
    print("raw UTF-8 in inference render:", "推测解码" in infer_full,
          "| escaped form present:", "\\u63a8" in infer_full)
    report["inference_raw_utf8"] = "推测解码" in infer_full
    report["inference_escaped"] = "\\u63a8" in infer_full
    print("think blocks in inference full render:", infer_full.count("<think>"))
    print("think blocks in inference step-3 prompt:", infer_step3.count("<think>"))
    report["think_blocks_inference_full"] = infer_full.count("<think>")
    report["think_blocks_inference_step3_prompt"] = infer_step3.count("<think>")
    if tpl is None:
        print("TRL reports the tokenizer's own template needs no swap.")
    else:
        train_full = tok.apply_chat_template(MESSAGES, tools=TOOLS, tokenize=False, chat_template=tpl,
                                             enable_thinking=False)
        print("think blocks in training render:", train_full.count("<think>"))
        print("raw UTF-8 in training render:", "推测解码" in train_full)
        report["think_blocks_training"] = train_full.count("<think>")
        report["training_raw_utf8"] = "推测解码" in train_full
        diff = list(difflib.unified_diff(infer_full.splitlines(), train_full.splitlines(),
                                         "inference", "training", lineterm="", n=1))
        print("\n".join(diff) if diff else "renders are identical")
        report["diff"] = diff
        enc = tok.apply_chat_template(MESSAGES, tools=TOOLS, chat_template=tpl, return_dict=True,
                                      return_assistant_tokens_mask=True, enable_thinking=False)
        trained = tok.decode([t for t, m in zip(enc["input_ids"], enc["assistant_masks"]) if m])
        print("\n--- tokens that would carry loss ---\n" + trained)
        report["trained_text"] = trained
        report["n_tokens"] = len(enc["input_ids"])
        report["n_trained_tokens"] = int(sum(enc["assistant_masks"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

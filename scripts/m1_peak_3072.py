"""M1 decision data: peak VRAM and throughput of the S4 LoRA setup with sequences near 3,072 tokens.

Training settings are those of spike S4 (scripts/spikes/s4_train_50_steps.py): bf16 LoRA r=16 on all
projections, batch 1 with 4-step accumulation, gradient checkpointing, completion-only loss. The samples are
built the way M2 will build per-step samples: an agent history in API format goes through build_context with
the current MODEL_TOOLS and system prompt, and the next assistant turn is the completion. Only samples near
the cap are kept, because the longest sequence sets the peak. No adapter is saved.

Run: .venv/Scripts/python.exe scripts/m1_peak_3072.py [--max-length 3072] [--steps 30] [--extra-searches 4]
    [--dry-run]
A higher cap needs more extra searches so that the kept samples still sit near the cap (--extra-searches 14
for 4,096; the sample report line shows the lengths actually reached).
--dry-run only builds the samples and prints their lengths (tokenizer only, no GPU).
Writes runs/m1/peak_<max_length>.json (gitignored).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from transformers import AutoTokenizer

from research_agent.agent.context import build_context
from research_agent.agent.prompts import system_prompt
from research_agent.agent.types import ToolCall, api_to_hf
from research_agent.canonical import canonical_json
from research_agent.tools.schemas import (
    MAX_ABSTRACT_CHARS,
    MAX_FULLTEXT_CHARS,
    MAX_SNIPPET_CHARS,
    MODEL_TOOLS,
)

MODEL = Path("models/Qwen3-1.7B")
OUT = Path("runs/m1")
TODAY = "2026-10-08"
TOPICS = ["speculative decoding", "mixture of experts routing", "test-time compute", "KV cache compression",
          "retrieval augmented generation", "LoRA fine-tuning", "tool use agents", "长上下文推理"]
WORDS = ["model", "draft", "tokens", "verify", "acceptance", "latency", "throughput", "memory", "cache",
         "attention", "routing", "experts", "benchmark", "accuracy", "training", "inference", "parallel",
         "decoding", "sequence", "layer", "gradient", "budget"]


def text(rng: random.Random, chars: int) -> str:
    out = []
    while sum(len(w) + 1 for w in out) < chars:
        out.append(rng.choice(WORDS))
    return " ".join(out)[:chars]


def assistant(calls: list[ToolCall], content: str = "") -> dict:
    return {"role": "assistant", "content": content, "tool_calls": [c.to_api() for c in calls]}


def call(i: int, name: str, **args) -> ToolCall:
    raw = json.dumps(args, ensure_ascii=False)
    return ToolCall(id=f"c{i}", name=name, arguments_json=raw, arguments=args)


def episode(rng: random.Random, extra_max: int = 4) -> tuple[list[dict], dict]:
    """0-extra_max extra searches, a search, k parallel get_paper calls, 1-3 rounds of k parallel
    fetch_fulltext, then the target write_note. The extra searches lengthen the compressed history toward the
    cap: about 105 tokens each, so a cap of 3,072 is reached with 4 and a cap of 4,096 needs about 14."""
    topic = rng.choice(TOPICS)
    msgs = [{"role": "system", "content": system_prompt(TODAY)},
            {"role": "user", "content": f"帮我找 2026 年 9 月关于 {topic} 的论文,读一读,再写读书笔记。"}]
    n = 0
    for _ in range(rng.randint(0, extra_max)):
        earlier = [{"arxiv_id": f"2608.{rng.randint(10000, 99999)}", "title": text(rng, 60),
                    "published": "2026-08-20", "snippet": text(rng, MAX_SNIPPET_CHARS)} for _ in range(5)]
        msgs += [assistant([call(n, "search_arxiv", query=f"{topic} {text(rng, 12)}")]),
                 {"role": "tool", "tool_call_id": f"c{n}", "content": canonical_json(earlier)}]
        n += 1
    hits = [{"arxiv_id": f"2609.{rng.randint(10000, 99999)}", "title": f"{topic.title()}: {text(rng, 40)}",
             "published": f"2026-09-{rng.randint(10, 29)}", "snippet": text(rng, MAX_SNIPPET_CHARS)}
            for _ in range(rng.randint(3, 5))]
    msgs += [assistant([call(n, "search_arxiv", query=topic, date_from="2026-09-01")]),
             {"role": "tool", "tool_call_id": f"c{n}", "content": canonical_json(hits)}]
    k = rng.randint(1, 3)
    ids = [h["arxiv_id"] for h in hits[:k]]
    calls = [call(n + 1 + j, "get_paper", arxiv_id=aid) for j, aid in enumerate(ids)]
    n += 1 + k
    msgs.append(assistant(calls))
    for c, aid in zip(calls, ids, strict=True):
        paper = {"arxiv_id": aid, "title": text(rng, 60), "authors": ["A. Author", "B. Author", "et al."],
                 "published": "2026-09-15", "abstract": text(rng, MAX_ABSTRACT_CHARS)}
        msgs.append({"role": "tool", "tool_call_id": c.id, "content": canonical_json(paper)})
    for section in rng.sample(["introduction", "method", "experiments"], rng.randint(1, 3)):
        calls = [call(n + j, "fetch_fulltext", arxiv_id=aid, section=section) for j, aid in enumerate(ids)]
        n += k
        msgs.append(assistant(calls))
        for c, aid in zip(calls, ids, strict=True):
            body = {"arxiv_id": aid, "section": section, "source": "html",
                    "text": text(rng, rng.randint(800, MAX_FULLTEXT_CHARS))}
            msgs.append({"role": "tool", "tool_call_id": c.id, "content": canonical_json(body)})
    target = assistant([call(n, "write_note", arxiv_id=ids[0], mode="scaffold")])
    return msgs, target


def sample(tok, msgs: list[dict], target: dict) -> dict:
    context = api_to_hf(build_context(msgs))
    prompt = tok.apply_chat_template(context, tools=MODEL_TOOLS, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    full = tok.apply_chat_template(context + api_to_hf([target]), tools=MODEL_TOOLS, tokenize=False,
                                   enable_thinking=False)
    assert full.startswith(prompt), "full render does not start with the prompt"
    completion = full[len(prompt):]
    if completion.endswith(tok.eos_token + "\n"):
        completion = completion[:-1]
    assert completion.endswith(tok.eos_token)
    return {"prompt": prompt, "completion": completion}


def build(tok, max_length: int, want: int, seed: int = 0, extra_max: int = 4) -> tuple[list[dict], list[int]]:
    rng = random.Random(seed)
    pool = []
    for _ in range(4000):
        s = sample(tok, *episode(rng, extra_max))
        n = len(tok(s["prompt"] + s["completion"])["input_ids"])
        if n <= max_length:
            pool.append((n, s))
    pool.sort(key=lambda p: -p[0])  # the longest sequences set the peak
    chosen = pool[:want]
    return [s for _, s in chosen], [n for n, _ in chosen]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--extra-searches", type=int, default=4,
                    help="the most extra searches an episode draws (4 fills a 3,072 cap, 14 a 4,096 cap)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(MODEL)
    samples, lengths = build(tok, args.max_length, want=args.steps * 4, extra_max=args.extra_searches)
    print(f"samples={len(samples)} tokens: max={max(lengths)} min={min(lengths)} "
          f"mean={sum(lengths) / len(lengths):.0f} (cap {args.max_length})", flush=True)
    if args.dry_run:
        return

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM
    from trl import SFTConfig, SFTTrainer

    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": 0})
    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                          "down_proj"])
    cfg = SFTConfig(output_dir=str(OUT / "peak_trainer"), max_steps=args.steps,
                    per_device_train_batch_size=1, gradient_accumulation_steps=4, learning_rate=1e-4,
                    lr_scheduler_type="cosine",
                    warmup_steps=0.03, bf16=True, gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=10,
                    save_strategy="no", max_length=args.max_length, packing=False, completion_only_loss=True,
                    dataloader_num_workers=0, report_to="none", seed=42, optim="adamw_torch")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=Dataset.from_list(samples),
                         processing_class=tok, peft_config=peft_cfg)
    mask_ok = []
    for i in range(2):
        ex = trainer.train_dataset[i]
        trained = tok.decode([t for t, lab in zip(ex["input_ids"], ex["labels"], strict=True) if lab != -100])
        mask_ok.append(trained.strip() == samples[i]["completion"].strip())
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = trainer.train()
    secs = time.perf_counter() - t0
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    n_seq = args.steps * cfg.gradient_accumulation_steps
    mean_len = sum(lengths) / len(lengths)
    report = {"max_length": args.max_length, "steps": args.steps, "sequences": n_seq,
              "max_tokens": max(lengths), "min_tokens": min(lengths), "mean_tokens": round(mean_len),
              "seconds": round(secs, 1), "tokens_per_s_est": round(n_seq * mean_len / secs),
              "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
              "peak_reserved_gib": round(peak_reserved, 2),
              "peak_reserved_gb": round(peak_reserved * 2**30 / 1e9, 2),
              "within_7_gb": peak_reserved * 2**30 / 1e9 <= 7.0, "final_loss": round(result.training_loss, 4),
              "mask_ok": mask_ok, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    print(json.dumps(report, indent=2))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"peak_{args.max_length}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

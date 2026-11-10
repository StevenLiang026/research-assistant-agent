"""Spike S4: 50 LoRA training steps of Qwen3-1.7B with TRL on Windows (bf16 by default).

Measures peak VRAM and throughput at realistic lengths (seven tool schemas, long tool outputs),
checks that only completion tokens carry loss, and saves the adapter for the llama-server LoRA test.

Samples are per-step prompt-completion pairs: the prompt is the history rendered with the model's own
(inference) chat template plus the generation prompt; the completion is the next assistant turn. The script
asserts that full_render == prompt + completion for every sample, so training text matches inference text.

Run: .venv/Scripts/python.exe scripts/spikes/s4_train_50_steps.py [--qlora] [--max-length 2048]
Writes runs/spikes/s4_train.json and runs/spikes/s4_adapter/.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from spike_tools import TOOLS, TOPICS

MODEL = Path("models/Qwen3-1.7B")
OUT = Path("runs/spikes")


def call(name: str, args: dict) -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
        "name": name, "arguments": args}}]}


def fake_hits(rng: random.Random, n: int) -> str:
    hits = []
    for _ in range(n):
        aid = f"2609.{rng.randint(10000, 99999)}"
        snippet = " ".join(rng.choice(["model", "decoding", "latency", "accuracy", "benchmark", "token",
                                       "draft", "verify", "throughput", "memory"]) for _ in range(45))
        hits.append({"arxiv_id": aid, "title": f"A study of {rng.choice(TOPICS)}", "published": "2026-09-1"
                     f"{rng.randint(0, 9)}", "authors": ["A. Author", "B. Author"], "abstract_snippet": snippet})
    return json.dumps(hits, ensure_ascii=False)


def build_samples(tok, n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    samples = []
    for i in range(n):
        topic = rng.choice(TOPICS)
        user = {"role": "user", "content": f"帮我找 2026 年 9 月关于 {topic} 的 {rng.randint(2, 5)} 篇论文并记笔记。"}
        system = {"role": "system", "content": "你是科研助理。按需调用工具,参数必须符合 Schema;不确定时追问。"}
        hits = fake_hits(rng, rng.randint(5, 8))
        aid = json.loads(hits)[0]["arxiv_id"]
        steps = [
            call("search_arxiv", {"query": topic, "date_from": "2026-09-01", "date_to": "2026-09-30",
                                  "max_results": 5}),
            {"role": "tool", "content": hits},
            call("get_paper", {"arxiv_id": aid}),
            {"role": "tool", "content": json.dumps({"arxiv_id": aid, "abstract": fake_hits(rng, 2)},
                                                   ensure_ascii=False)},
            call("write_note", {"arxiv_id": aid, "mode": "scaffold"}),
        ]
        k = rng.choice([0, 2, 4])  # which assistant step is the target
        history = [system, user] + steps[:k]
        target = steps[k]
        prompt = tok.apply_chat_template(history, tools=TOOLS, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)
        full = tok.apply_chat_template(history + [target], tools=TOOLS, tokenize=False, enable_thinking=False)
        assert full.startswith(prompt), f"sample {i}: full render does not start with the prompt"
        completion = full[len(prompt):]
        # End exactly at the turn terminator (generation stops there); the template's trailing newline
        # would otherwise make TRL append a second terminator.
        if completion.endswith(tok.eos_token + "\n"):
            completion = completion[:-1]
        assert completion.endswith(tok.eos_token), f"sample {i}: completion does not end with {tok.eos_token!r}"
        samples.append({"prompt": prompt, "completion": completion})
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qlora", action="store_true")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    samples = build_samples(tok, 240)
    all_lengths = [len(tok(s["prompt"] + s["completion"])["input_ids"]) for s in samples]
    # Plan rule: samples longer than max_length are dropped and counted, never truncated.
    kept = [(s, n) for s, n in zip(samples, all_lengths) if n <= args.max_length]
    dropped = len(samples) - len(kept)
    samples, lengths = [s for s, _ in kept], [n for _, n in kept]
    print(f"samples kept={len(samples)} dropped(>{args.max_length})={dropped} tokens: min={min(lengths)} "
          f"mean={sum(lengths) / len(lengths):.0f} max={max(lengths)}")
    ds = Dataset.from_list(samples)

    quant = None
    if args.qlora:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
    # device_map streams the weights straight to the GPU instead of staging a full copy in system RAM
    # (the first attempt was stopped by system memory pressure while loading).
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, quantization_config=quant,
                                                 attn_implementation="sdpa", device_map={"": 0})
    peft_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                          "down_proj"])
    cfg = SFTConfig(output_dir=str(OUT / "s4_trainer"), max_steps=args.steps, per_device_train_batch_size=1,
                    gradient_accumulation_steps=4, learning_rate=1e-4, lr_scheduler_type="cosine",
                    warmup_steps=0.03, bf16=True, gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=5,
                    save_strategy="no", max_length=args.max_length, packing=False, completion_only_loss=True,
                    dataloader_num_workers=0, report_to="none", seed=42,
                    optim="paged_adamw_8bit" if args.qlora else "adamw_torch")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=peft_cfg)

    # Mask check: only completion tokens may carry loss.
    mask_ok = []
    for i in range(3):
        ex = trainer.train_dataset[i]
        ids = ex["input_ids"]
        if "completion_mask" in ex:
            cmask = ex["completion_mask"]
        else:  # TRL >= 1.14 builds labels with -100 on masked positions
            cmask = [lab != -100 for lab in ex["labels"]]
        trained = tok.decode([t for t, m in zip(ids, cmask) if m])
        if i == 0:
            print("processed columns:", sorted(ex.keys()), "| eos:", repr(tok.eos_token))
        mask_ok.append(trained.strip() == samples[i]["completion"].strip())
        print(f"mask check {i}: {'ok' if mask_ok[-1] else 'MISMATCH'} -> {trained[:120]!r}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    result = trainer.train()
    secs = time.perf_counter() - t0
    peak_alloc = torch.cuda.max_memory_allocated() / 2**30
    peak_reserved = torch.cuda.max_memory_reserved() / 2**30
    n_seq = args.steps * cfg.gradient_accumulation_steps
    mean_len = sum(lengths) / len(lengths)
    report = {
        "mode": "qlora-4bit" if args.qlora else "bf16-lora", "max_length": args.max_length,
        "steps": args.steps, "sequences": n_seq, "mean_tokens": round(mean_len), "max_tokens": max(lengths),
        "dropped_over_max_length": dropped,
        "seconds": round(secs, 1), "tokens_per_s_est": round(n_seq * mean_len / secs),
        "peak_alloc_gib": round(peak_alloc, 2), "peak_reserved_gib": round(peak_reserved, 2),
        "final_loss": round(result.training_loss, 4), "mask_ok": mask_ok,
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    print(json.dumps(report, indent=2))
    adapter_dir = OUT / ("s4_adapter_qlora" if args.qlora else "s4_adapter")
    trainer.model.save_pretrained(adapter_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"s4_train_{report['mode']}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

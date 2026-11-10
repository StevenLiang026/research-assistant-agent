"""Spike S8: can deepseek-flash produce tool-calling trajectories that fit our schemas, and what does one cost?

Runs N short agent episodes (default 20) against a deterministic stub of the seven tools, validates every tool
call against its JSON Schema, and prices tokens with the upper-bound (peak, cache-miss) price list.
The key is read by name from the environment (DEEPSEEK_API_KEY) and never printed or stored.

Run: .venv/Scripts/python.exe scripts/spikes/s8_deepseek_trajectories.py [n]
Writes runs/spikes/s8_deepseek.json (gitignored).
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

from jsonschema import Draft202012Validator
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from spike_tools import TOOLS, TOPICS  # same seven schemas as the training spike

PRICE_RMB_PER_M = {"input": 2.0, "output": 8.0}  # deepseek-flash, peak, cache miss (upper bound)
MODEL = "deepseek-flash"
OUT = Path("runs/spikes/s8_deepseek.json")
SCHEMAS = {t["function"]["name"]: Draft202012Validator(t["function"]["parameters"]) for t in TOOLS}
REQUESTS = [
    "帮我找 2026 年 9 月关于 {t} 的 3 篇论文,挑最相关的一篇写读书笔记(scaffold 模式)。",
    "Find two recent papers on {t} from the last month and summarise the most cited one.",
    "我之前记过 {t} 的笔记吗?没有的话找一篇新的。",
    "帮我存一篇笔记。",  # missing paper id: should ask for clarification
]


def stub_tool(name: str, args: dict, rng: random.Random) -> str:
    if name == "search_arxiv":
        return json.dumps([{"arxiv_id": f"2609.{rng.randint(10000, 99999)}", "title": f"On {args.get('query', '')}",
                            "published": "2026-09-15"} for _ in range(min(int(args.get("max_results", 3)), 5))],
                          ensure_ascii=False)
    if name == "get_paper":
        return json.dumps({"arxiv_id": args.get("arxiv_id"), "abstract": "We study the problem and report gains."})
    if name == "fetch_fulltext":
        return json.dumps({"arxiv_id": args.get("arxiv_id"), "section": args.get("section", "introduction"),
                           "text": "Introduction. Large language models ..."})
    if name == "search_notes":
        return json.dumps([])
    if name == "write_note":
        return json.dumps({"note_id": "n-001", "path": "notes/n-001.md"})
    if name == "list_notes":
        return json.dumps([])
    return json.dumps({"error": "UNKNOWN_TOOL"})


def main(n: int = 20) -> None:
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DEEPSEEK_API_KEY is not set in this process environment.")
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    rng = random.Random(0)
    episodes, usage_in, usage_out = [], 0, 0
    for i in range(n):
        req = rng.choice(REQUESTS).format(t=rng.choice(TOPICS))
        messages = [{"role": "system", "content": "你是科研助理。按需调用工具,参数必须符合 Schema;缺信息时用 "
                                                  "ask_clarification 追问。"},
                    {"role": "user", "content": req}]
        calls, invalid, t0 = [], [], time.perf_counter()
        for _ in range(4):
            resp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, temperature=0)
            usage_in += resp.usage.prompt_tokens
            usage_out += resp.usage.completion_tokens
            msg = resp.choices[0].message
            if not msg.tool_calls:
                break
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    errs = [e.message for e in SCHEMAS[name].iter_errors(args)] if name in SCHEMAS else ["unknown tool"]
                except json.JSONDecodeError as e:
                    args, errs = {}, [f"bad JSON: {e}"]
                calls.append({"name": name, "args": args, "errors": errs})
                if errs:
                    invalid.append(name)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": stub_tool(name, args, rng)})
            if any(c["name"] == "ask_clarification" for c in calls):
                break
        episodes.append({"request": req, "calls": calls, "invalid": invalid,
                         "seconds": round(time.perf_counter() - t0, 1)})
        print(f"{i + 1:02d} calls={[c['name'] for c in calls]} invalid={invalid}", flush=True)
    n_calls = sum(len(e["calls"]) for e in episodes)
    n_invalid = sum(len(e["invalid"]) for e in episodes)
    cost = usage_in / 1e6 * PRICE_RMB_PER_M["input"] + usage_out / 1e6 * PRICE_RMB_PER_M["output"]
    summary = {"model": MODEL, "episodes": n, "tool_calls": n_calls, "schema_invalid": n_invalid,
               "valid_rate": round(1 - n_invalid / max(n_calls, 1), 3), "input_tokens": usage_in,
               "output_tokens": usage_out, "cost_rmb_upper_bound": round(cost, 4),
               "cost_per_episode_rmb_upper_bound": round(cost / n, 5)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"summary": summary, "episodes": episodes}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)

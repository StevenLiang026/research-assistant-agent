"""M1 smoke run: deepseek-flash drives the real loop, executor, context builder, cache and ledger end to end.

The tools are deterministic STUBS (no network) and the validation and failure policy are PLACEHOLDERS that
live only in this script: the owner's check_call and decide replace them in M1 proper. The run is done twice;
the second run must make 0 paid calls and give transcripts identical to the first apart from the money fields.

Run (key read by name from the environment, never printed):
    .venv/Scripts/python.exe scripts/m1_smoke.py
Writes runs/m1/smoke/run{1,2}/*.json and runs/m1/smoke/summary.json (gitignored).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from pathlib import Path

from research_agent.agent.context import qwen_token_counter
from research_agent.agent.executor import ToolExecutor
from research_agent.agent.loop import run_turn, save_transcript
from research_agent.agent.models import CloudModel
from research_agent.agent.types import ToolOutcome, Transcript
from research_agent.llm.budget import Budget
from research_agent.llm.cache import ResponseCache
from research_agent.llm.cloud import CloudClient
from research_agent.tools import schemas

TODAY = "2026-10-08"
MODEL = "deepseek-flash"
RUN_CAP_RMB = 0.5  # per run, on top of what the M1 stage has already spent
OUT = Path("runs/m1/smoke")
REQUESTS = [
    "帮我找 2026 年 9 月关于 speculative decoding 的 3 篇论文,挑最相关的一篇写读书笔记(scaffold 模式)。",
    "Find two recent papers on mixture-of-experts routing and read the method section of the best one.",
    "帮我存一篇笔记。",  # no paper given: the model should ask
]


def _fake_id(text: str, n: int) -> str:
    digits = str(int(hashlib.sha256(f"{text}|{n}".encode()).hexdigest(), 16))[-5:]
    return f"2609.{digits}"


# ---- STUB tools: deterministic stand-ins for the owner's arXiv and notes tools ---------------------------

def stub_search(args: dict) -> ToolOutcome:
    query = args["query"]
    k = min(int(args.get("max_results", schemas.MAX_SEARCH_RESULTS)), 3)
    hits = [{"arxiv_id": _fake_id(query, i), "title": f"{query.title()}: Study {i + 1}",
             "published": f"2026-09-{10 + i:02d}", "snippet": f"We study {query} from angle {i + 1}."}
            for i in range(k)]
    return ToolOutcome(hits)


def stub_paper(args: dict) -> ToolOutcome:
    aid = args["arxiv_id"]
    return ToolOutcome({"arxiv_id": aid, "title": f"Paper {aid}", "authors": ["A. Author", "B. Author"],
                        "published": "2026-09-12",
                        "abstract": f"Abstract of {aid}: method, evidence, limits."})


def stub_fulltext(args: dict) -> ToolOutcome:
    aid, section = args["arxiv_id"], args.get("section", "introduction")
    return ToolOutcome({"arxiv_id": aid, "section": section, "source": "html",
                        "text": f"[{section} of {aid}] The authors propose a routing rule and evaluate it."})


def stub_search_notes(args: dict) -> ToolOutcome:
    return ToolOutcome([], schemas.EMPTY_RESULTS)


def stub_list_notes(args: dict) -> ToolOutcome:
    return ToolOutcome([])


def stub_write_note(args: dict) -> ToolOutcome:
    aid = args["arxiv_id"]
    return ToolOutcome({"note_id": f"{aid}-note", "arxiv_id": aid, "path": f"notes/{aid}.md"})


STUB_TOOLS = {"search_arxiv": stub_search, "get_paper": stub_paper, "fetch_fulltext": stub_fulltext,
              "search_notes": stub_search_notes, "list_notes": stub_list_notes, "write_note": stub_write_note}


# ---- PLACEHOLDERS for the owner's modules (accept every call; fixed answers) -----------------------------

def placeholder_check_call(tool: str, args: dict, seen_ids: list[str], today: str) -> list[str]:
    return []  # the owner's check_call (JSON Schema + semantic checks) replaces this


def placeholder_decide(event: str, times: int) -> str:
    if event == "empty_results":
        return "relax" if times == 1 else "tell_user"
    return "stop"  # the owner's decide replaces this


def neutral(t: Transcript) -> str:
    """The transcript without the fields that must differ between a paid run and a cached rerun."""
    return dataclasses.replace(t, paid_calls=0, cache_hits=0, local_calls=0, cost_rmb=0.0).to_json()


def run_once(label: str, count_tokens) -> list[Transcript]:
    ledger_path, cache_path = OUT.parent / "budget.sqlite", OUT.parent / "cache.sqlite"
    spent = Budget(ledger_path, stage="M1", stage_cap_rmb=10.0).spent("M1")
    budget = Budget(ledger_path, stage="M1", stage_cap_rmb=min(10.0, spent + RUN_CAP_RMB))
    client = CloudClient(MODEL, cache=ResponseCache(cache_path), budget=budget)
    model = CloudModel(client)
    executor = ToolExecutor(STUB_TOOLS, placeholder_check_call, today=TODAY)
    transcripts = []
    for i, request in enumerate(REQUESTS, 1):
        t = run_turn(request, model=model, executor=executor, decide=placeholder_decide, today=TODAY,
                     count_tokens=count_tokens, cost_of=model.cost)
        save_transcript(t, OUT / label / f"request{i}.json")  # saved at once, in case a later turn fails
        transcripts.append(t)
        calls = [c["function"]["name"] for m in t.messages if m.get("tool_calls") for c in m["tool_calls"]]
        print(f"[{label}] request {i}: {t.status}, steps={len(t.steps)}, calls={calls}, paid={t.paid_calls}, "
              f"cached={t.cache_hits}, cost={t.cost_rmb:.4f} RMB", flush=True)
    return transcripts


def main() -> int:
    count_tokens = qwen_token_counter("models/Qwen3-1.7B")
    first = run_once("run1", count_tokens)
    second = run_once("run2", count_tokens)
    summary = {
        "model": MODEL, "today": TODAY, "schema_hash": schemas.SCHEMA_HASH,
        "statuses": [t.status for t in first],
        "run1_paid_calls": sum(t.paid_calls for t in first),
        "run1_cost_rmb": round(sum(t.cost_rmb for t in first), 5),
        "run2_paid_calls": sum(t.paid_calls for t in second),
        "run2_cache_hits": sum(t.cache_hits for t in second),
        "identical_apart_from_money": [neutral(a) == neutral(b) for a, b in zip(first, second, strict=True)],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    ok = summary["run2_paid_calls"] == 0 and all(summary["identical_apart_from_money"])
    print("smoke criterion (second run free and identical):", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

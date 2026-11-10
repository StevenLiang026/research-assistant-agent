"""Transform the synthetic training file for M6 (docs/m6-计划.md §2.1-2.2, §3.1 A1-A5, §4 WP2).

    uv run --no-sync python scripts/transform_synth.py data/raw/synth/synth.jsonl data/synth/requests.jsonl
        --out data/raw/synth/synth-m6.jsonl --manifest data/synth/train-manifest-m6.json
        --requests-out data/synth/requests-m6.jsonl [--terms data/synth/terms-zh-en.json] [--seed 42]
        [--dev data/eval/dev.jsonl] [--corpus data/corpus/manifest.json] [--force]
        [--r3-total 300] [--r5 100] [--r6 100] [--r7 200]

Why. The teacher wrote habitual non-default values into the trajectories (search_notes.top_k=5,
list_notes.limit=20) that the model's tool view gives no reason for and the scorer counts as wrong, asked for
clarification late and with `missing` lists that disagree with the request, never combined list_notes'
since with a tag, never relaxed a search after EMPTY_RESULTS and never met a Chinese ML term in a request.
The rules below change the data mechanically, without a model or an API call.

Rules (each a pure function below; the manifest counts every rule's hits):
- R1 (A1): in every assistant call, delete search_notes.top_k == 5 and list_notes.limit == 20 unless the
  request's slots.count equals that value (the user asked for it), and fetch_fulltext.max_chars when it
  equals the schema default. Nothing else changes (sort_by stays); the arguments are re-serialised in the
  call's own JSON style, key order kept, and every changed call must pass validation.check_call.
- R2 (A2a): in the clarify-scenario trajectories, every ask_clarification call's `missing` becomes the
  request's own slots.missing (all in schemas.MISSING); a request without slots.missing leaves the call as
  it is and is counted.
- R3 (A2b): single-step records whose only step is ask_clarification(question, missing = slots.missing):
  from the train clarify requests of the requests file, except those whose kept trajectory asks only at
  its second step (one user text must not teach two different first steps; r3_train_requests), then from
  new requests filled from requests.py's own train clarify templates and train topics until --r3-total.
  The question is one of two sentences per missing slot and language (CLARIFY_QUESTIONS), picked by the
  request id's hash.
- R4 (A2c): clarify-scenario trajectories whose first ask_clarification is at step 3 or later are dropped.
- R5 (A3): --r5 new list_notes requests from LIST_NOTES_TEMPLATES (2 zh + 2 en, each combining
  {notes_since} and {tag}; kept here, not in requests.TEMPLATES, so the registered requests stay
  reproducible; their wording is audited against every existing template), one single-step record each:
  list_notes(tag, since).
- R6 (A4): --r6 two-step records from train normal requests with a topic and a date or category slot:
  search_arxiv with the constraint -> the tool message the agent loop really produces for an empty arXiv
  feed (ToolExecutor over tools.adapters.arxiv_tools, the owner's decide, the loop's relax hint) -> the
  same query without the date constraint (or, without dates, without the categories). The recover
  trajectories of the input (RATE_LIMITED, UPSTREAM_ERROR: retry as is) are not touched by this rule.
- R7 (A5): at most --r7 single-step records "<中文术语>方向的论文"-style (TERM_TEMPLATES) ->
  search_arxiv(query=<English term>), from the term list (--terms), which must hold at least MIN_TERMS terms
  and none of the six held-out topics (dev and test of the requests manifest), nothing containing a corpus
  topic phrase or name, and none of the close synonyms in HELD_OUT_EN / HELD_OUT_ZH.

New records are produced by running the real agent loop (agent.loop.run_turn) with a scripted model, so the
system prompt, the message shapes and the tool replies are the loop's own, and cut after the last scripted
step (an ask_clarification keeps the loop's "asked" reply). They carry the current prompt_hash and
schema_hash, history "agent", run "m6-transform" and a family starting with "m6/", which no existing
family does. A record made from an existing train request has the id "<request id>-m6r3" or "-m6r6"; new
requests are numbered req-m6-NNNN (R3, then R5, then R7) and written to --requests-out.

Leakage. Every new request's text is checked with requests.Deduper's own rules (masked character 5-grams):
against the evaluation texts it must stay below TRAIN_VS_EVAL (0.6), and requests filled from templates
(R3, R5) are also refused as exact or near (0.8) duplicates of the train requests and of each other; R7's
template texts are refused only as exact duplicates or evaluation overlaps (two terms in one template are
near duplicates by construction). The evaluation texts are the dev and test rows of the requests file and
the user messages of --dev (the development set, read for this check only and refused if it holds test
material); no test evaluation file is read. Every output record's first user message is also checked
against the evaluation texts; an input record at or over the limit is dropped and counted.

Inputs: the synthetic file (agent-history records), the requests file and its manifest
(<stem>-manifest.json: the split seed and topics), the term list, the dev file and the corpus manifest.
Outputs: --out (canonical JSON lines in id order, UTF-8, LF; staged and read back with
train.samples.load_records before it replaces anything), --requests-out (the new requests, canonical JSON
lines) and --manifest (sorted keys, two-space indent: per-rule counts, records in and out by scenario and
language, the new requests, the term list's and the inputs' SHA-256, the output's SHA-256 and size). The
same inputs and options give byte-identical files.

Records only (M7, docs/m7-计划.md §5 新增训练轨迹: another teacher run goes through M6's transforms).
--records-only applies the rules that change the input records, R1, R2 on the clarify-scenario records and
R4, and the evaluation-overlap drop, exactly as the full run applies them, and nothing else: the scripted
records of R3, R5, R6 and R7 were a one-off for M6's data, so no record is built, no request is drawn,
--requests-out is neither checked nor written and the term list is not read (nor the template audit run).
Its output is the full run's output without the scripted records, byte for byte; the manifest says
"mode": "records-only", lists the rules not run and adds the records in, after R4 and out by scenario and
language. Three options carry over what M6 took from its own inputs, and the manifest records them when
given:
- --derive-habits: R1's habit values are taken from the input records the way M6 found 5 and 20 in
  synth.jsonl (docs/m6-计划.md §2.2): per habit parameter, the value the assistant calls set most often,
  unless that is the schema default (then nothing is deleted for it); ties go to the smaller canonical JSON.
  The manifest's "habits" holds every value's count; R1's "habits" the values applied.
- --eval-requests FILE...: more requests files whose dev and test rows join the evaluation texts (an M7
  requests file built with --only-split train holds none; the registered v1 and v2 requests do).
- --eval-items FILE...: more evaluation item files (development sets) whose user messages join them, each
  refused if it holds test material, like --dev.

FT4 (docs/m7-本地提升计划.md WP6), opt-in with --ft4 and off by default: without it every output is the one
above, byte for byte. The rules run after R1, R2, R4 and the evaluation-overlap drop, over the kept input
records (never over M6's scripted records, which none of them matches); the coverage analysis behind each
rule is the plan's (counts there are over synth-m6 and synth-m7 together).
- R13 first, in place: in a Chinese record, a search_notes query of Chinese characters without a Latin letter
  gets the English phrase of the request's train topic appended ("量化" -> "量化 quantization"); a request
  without a train topic (the placeholder searches of clarify requests) leaves it and is counted (no_term).
  The Chinese + English form follows the development gold (3 of its 4 Chinese search_notes queries are
  mixed). NoteStore requires every term of a query, so the mixed query finds no more than the English term
  alone; the template reasoning (scripts/template_reasoning.py) does not suggest that it widens the search.
- Then the first turn, in order, where one of these fires is relabelled and the record cut right after it
  (an ask_clarification keeps the loop's "asked" reply); the later turns vanish:
  - R11 (turn 0): a clarify-scenario record whose first step only searched the notes and whose first
    ask_clarification, at its second step, misses the topic asks at once instead: CLARIFY_QUESTIONS'
    question (as R3 picks it), the later ask's missing, no text.
  - R8: after a step with a RATE_LIMITED or UPSTREAM_ERROR reply, a target that is not exactly the failed
    calls (the multiset of names and parsed arguments) becomes those calls again, with new ids and no text;
    the calls of the step that returned are not repeated. Not when a failed call already ran
    MAX_RUNS_PER_CALL times (the loop would refuse a third run; counted at_run_limit). No fault injection.
  - R9 S4a: in a note request (NOTE_INTENTS) whose history holds an untagged list_notes reply with at least
    one note (the listing seen, Listing) that is complete, all the user's notes (Listing.complete, below),
    and whose last step met no transient error, a target that only queries the notes again closes instead (an
    S4a point); a listing that hit its limit or had a since sets off nothing. The rule: a request that wants
    one remembered note
    found (PICK_FAMILIES, "I remember writing a note about ...") gets, when two or more listed notes carry its
    topic's tag, ask_clarification(missing=["paper_id"]) whose question names those notes (ids and titles),
    with a declarative text that says how many carry the tag; with one such note, an answer presenting it;
    with none, an answer that says so and shows the listing. Every other note request asks for a list and gets
    a template answer built from the listing (listing_answer: the notes with the request's tag, or its
    topic's tag, else all of them, as a table; with a count in the request, that many, newest first, said
    so). A listing is all the user's notes (Listing.complete) only when its call has no since and no tag and
    listed fewer notes than its limit (the schema default when absent); the texts otherwise speak of "the N
    notes listed" and claim nothing about the rest.
  - R9 S4b: a search_arxiv call after S4B_SEARCHES or more steps that searched arXiv, the last of them the
    step before, becomes an honest report of the queries and of every paper the searches returned.
  - R10: the 16 plain-text mid-turn questions labelled by the analysis (R10_LABELLED, with the missing slot
    read from each question; its 4 borderline ones are left) become ask_clarification when the split is
    clean (split_question): the questions, with their options and the sentences or lines that elaborate them
    (比如, 例如, 或者, 或, e.g., for example, such as, or), move into "question" and are the text's trailing
    block, and the text keeps what comes before them. A label whose questions are followed by other
    declarative content, or whose list (led by a colon) has items that stay, is left as the teacher wrote it
    and counted (skipped_not_trailing, skipped_partial_list). A sentence ends at 。!?！？ and at a full stop
    before a space, but not after a common abbreviation (e.g., i.e., etc., vs., cf., et al., approx.) or a
    single letter.
- Derived records, cut copies of a trajectory (ids <record id>-ft4r9k<k> and -ft4r10k<k>, run FT4_RUN,
  family ft4/r9/... or ft4/r10/...): the turns before turn k are copied as context and turn k is the target.
  Each says so itself: "context_turns": k, the number of its leading assistant turns that are context only,
  which research_agent.train.samples leaves out of training (its step builders skip turns below it).
  - R9 constructs a closing turn (the S4a rule) only where a context is new: at the first S4a point that a
    cut by R8, R10 or R11 removed from a note request. A cut by R9 itself removes no new context (the point
    after an R9 relabel differs from it by one more note query), so none is built there (counted as
    not_built_after_r9_cuts). The plan's 100 are not padded towards.
  - R10 takes the "full listing seen -> ask" turns: at the second step of the records R11 relabelled, where
    the teacher's own ask is kept as the target (changes kind "copy" with target true; its FT3 sidecar row
    follows it), and, as new asks, in the clarify records R4 dropped: one per record, at its first listing
    point (before its first ask) whose listing is complete (Listing.complete). A new ask is an
    ask_clarification (declarative text about the listing, CLARIFY_QUESTIONS' question, the record's own
    missing) and counts as new; of new asks with one target (text and call, _target_key) only the first in the
    hashed order is built (the others: duplicate_targets).
  The new constructions are capped (--r9-new, --r10-new; the teacher's kept turns are not capped) in the
  order of a hash of their ids, and every copy must stay below the evaluation limit of FT4's guard.
- New records (the full run only; ids req-ft4-NNNN in this order, written to --requests-out after M6's,
  run FT4_RUN): R11's relative-date requests (RELATIVE_TEMPLATES, today counted back by relative_range) with
  list_notes(since) or search_arxiv(date_from, date_to[, sort_by]), whose sort_by follows the owner's rule
  (relative_sort_by, RELATIVE_SORT_RULE): the majority effective value of the teacher's search_arxiv calls
  with a date_from in the find_recent_papers records of the input and of --r11-sort-inputs (the M7 input), a
  call without sort_by counted as the default; left out when that majority is the default (the counts and
  the rule are in the manifest); R14's definite references with a topic, filled only from the train-split
  templates that have both (defref_families), whose first step searches the topic; R14's explain_tools
  questions on optional parameters (EXPLAIN_TEMPLATES, written from the tool schemas) answered from
  MODEL_TOOLS (explain_answer); and, given --glossary, R13's R7-style term records (glossary_requests: the
  hook for the glossary another package builds): the glossary read with research_agent.glossary.load_glossary,
  its noted terms only (the blocked ones counted apart), in the order of the SHA-256 of the Chinese term
  (term_order); a term is eligible when its English term is ASCII only, holds every Latin token of the
  Chinese term (Adam算法, C4.5算法) as a whole word, and the term is not one only AITD's AI for Science
  section gives (science_only_terms: the sections read with scripts/build_glossary.py's own parser from the
  AITD files that the provenance.json beside the glossary records, their SHA-256 checked); then R7's dedup
  and check_terms; the query is the English term with every word lowercased unless it has two or more
  capitals or a digit (glossary_query). R11's and R14's calls leave out the arguments at their default
  (sort_by, max_results), as M6's scripted records do: the scorer fills in the defaults. Every new request is
  offered to FT4's guard: the run's evaluation texts plus --ft4-eval-requests and --ft4-eval-items, and the
  train and M6 requests as earlier train texts.
--changes-out gets one line per assistant turn an FT4 rule changed, built or copied, in (id, k) order:
{"id", "k", "rule", "kind": "relabel" | "new" | "copy", "target", "from": [record id, k] | null}; a copied
turn is context ("target": false) or the teacher's turn kept as a derived record's target ("target": true).
The rows with target false are exactly the turns below the records' context_turns (checked before anything is
written). The manifest's "ft4" holds every rule's counts, the turns of the input each rule's situation holds
for, cut or not, as the coverage analysis counted them (targets_in_input: a record is cut at its first
relabel, so later targets of the same record vanish), the samples by step type before and after, the prompts
that repeat with another answer (with the context turns left out, as the trainer leaves them out, and over all
turns), the guard's numbers and the changes file's SHA-256.
--records-only --ft4 runs the in-place rules and the derived records, not the new ones (ft4.not_run).

Exit 0 on success; 1 on malformed or refused input (a record without a request, a record under another
prompt or schema hash, a changed call check_call refuses, a term list that fails its checks, test material
in --dev, --eval-items or --ft4-eval-items, a template audit failure, an R10 label that does not fit its
turn, a glossary load_glossary refuses, an AITD file that is not the one the glossary's provenance records);
2 on usage errors (bad arguments, a missing input (the glossary's provenance.json among them), --out,
--changes-out or, outside --records-only, --requests-out already present without --force, an FT4 option
without --ft4, --ft4 without --changes-out, an option of the new FT4 records with --records-only). Nothing is
written on exit 1 or 2.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

from research_agent import arxiv, policy
from research_agent.agent.context import step_contexts
from research_agent.agent.executor import ToolExecutor
from research_agent.agent.loop import MAX_RUNS_PER_CALL, run_turn
from research_agent.agent.prompts import PROMPT_HASH, system_prompt
from research_agent.agent.types import ANSWERED, AssistantTurn, ToolCall, ToolOutcome, ToolResult, call_key
from research_agent.canonical import canonical_json, split_lines
from research_agent.data import requests as rq
from research_agent.data.reasoning import answer_sha256
from research_agent.data.synth import load_requests, requests_manifest_of
from research_agent.eval.items import item_type
from research_agent.eval.lock import RefusedTestSet, read_jsonl, refuse_test_material
from research_agent.glossary import Glossary, load_glossary
from research_agent.net.fetch import Fetched
from research_agent.normalize import normalize_text
from research_agent.tools import schemas
from research_agent.tools.adapters import arxiv_tools
from research_agent.train.samples import CONTEXT_TURNS, RecordError, is_agent_history, load_records
from research_agent.validation import check_call

REPO = Path(__file__).resolve().parents[1]
OUT = Path("data/raw/synth/synth-m6.jsonl")
MANIFEST = Path("data/synth/train-manifest-m6.json")
REQUESTS_OUT = Path("data/synth/requests-m6.jsonl")
TERMS = Path("data/synth/terms-zh-en.json")
DEV = Path("data/eval/dev.jsonl")
GENERATOR = "transform_synth-m6-2026-10-17"
RUN = "m6-transform"
TARGETS = {"r3_total": 300, "r5": 100, "r6": 100, "r7": 200}
MIN_TERMS = 100
ATTEMPTS_PER_REQUEST = 60  # as requests.MAX_ATTEMPTS_PER_REQUEST

ASK = "ask_clarification"
HABITS = {("search_notes", "top_k"): 5, ("list_notes", "limit"): 20}  # deleted unless slots.count asks for it
AT_DEFAULT = {("fetch_fulltext", "max_chars"): schemas.DEFAULTS["fetch_fulltext"]["max_chars"]}
DATE_KEYS = ("date_from", "date_to")
ABSENT = "absent"  # derive_habits' count of the calls without the parameter
SCRIPTED_RULES = ("R3", "R5", "R6", "R7")  # the rules that build records with a scripted model

# R3: two short questions per missing slot and language.
CLARIFY_QUESTIONS = {
    "topic": {"zh": ("你说的是哪个研究方向?告诉我主题我再去检索。", "想查哪个主题的论文?请给我具体的方向。"),
              "en": ("Which research topic do you mean? Tell me and I'll search for it.",
                     "What topic should I look into? Please name the area.")},
    "paper_id": {"zh": ("你指的是哪篇论文?请给我它的 arXiv 编号。",
                        "我还不知道是哪篇,能发一下论文编号或标题吗?"),
                 "en": ("Which paper do you mean? Please send its arXiv id.",
                        "I don't know which paper that is yet. Could you share its arXiv id or title?")},
    "date_range": {"zh": ("要查哪段时间的论文?请给个起止日期。", "从什么时候开始算?告诉我日期范围。"),
                   "en": ("Which time range should I cover? Please give a start date.",
                          "Since when should I look? Tell me the date range.")},
    "count": {"zh": ("需要几篇?告诉我数量。", "要列出多少条?请给个数字。"),
              "en": ("How many do you want?", "How many results should I return?")},
    "tags": {"zh": ("要看哪个标签下的笔记?", "按哪个标签筛选?请告诉我标签名。"),
             "en": ("Which tag should I filter by?", "Which tag are the notes under?")},
}

# R5: list_notes requests that combine a date and a tag (requests.FILLERS: notes_since -> since, tag -> tags).
LIST_NOTES_TEMPLATES = {
    "zh": ("把我{notes_since}打了 {tag} 标签的笔记都列出来。",
           "{notes_since}归在 {tag} 下面的阅读笔记有哪些?给我列个清单。"),
    "en": ("List every note I filed under {tag} {notes_since}.",
           "Which of my {tag} notes were written {notes_since}? Give me the list."),
}

# R7: Chinese requests naming a term; {term} is the Chinese term.
TERM_TEMPLATES = ("帮我找几篇{term}方向的论文。", "{term}方向有什么值得读的论文?",
                  "想看看{term}相关的论文,帮我搜一下。", "有没有关于{term}的论文?推荐几篇。")
TERM_INTENT = "term_search"
# Close synonyms of the six held-out topics (hallucination, mixture of experts, test-time scaling,
# distillation, looped transformers, tokenization), refused on top of the corpus names and phrases.
HELD_OUT_EN = ("distill", "hallucinat", "expert", "moe", "tokeniz", "byte-pair", "subword", "test-time",
               "inference-time", "looped", "recurren", "chain-of-thought", "self-consistency", "fact-check",
               "fact verification")
HELD_OUT_ZH = ("蒸馏", "幻觉", "专家", "分词", "测试时", "推理时", "循环", "思维链", "词元", "事实")
_HAN = re.compile("[一-鿿]")


class TransformError(ValueError):
    """Input the rules cannot be applied to; the CLI exits 1."""


# ---- arguments ------------------------------------------------------------------------------------------

_STYLES = ({"ensure_ascii": False}, {"ensure_ascii": False, "separators": (",", ":")},
           {"ensure_ascii": True}, {"ensure_ascii": True, "separators": (",", ":")})


def serialize_like(raw: str, args: dict) -> str:
    """args as JSON in the style raw was written in (separators, ASCII escapes); json.dumps' default
    separators without escapes, the style of the input file, when raw matches none of them."""
    parsed = json.loads(raw)
    for style in _STYLES:
        if json.dumps(parsed, **style) == raw:
            return json.dumps(args, **style)
    return json.dumps(args, ensure_ascii=False)


def _checked(name: str, args: dict, where: str, today: str) -> None:
    errors = check_call(name, args, [], today)
    if errors:
        raise TransformError(f"{where}: the changed {name} call is refused by check_call: {errors}")


def _today_of(record: dict) -> str:
    match = re.search(r"\d{4}-\d{2}-\d{2}", record["messages"][0].get("content") or "")
    return match.group(0) if match else ""


def _calls(record: dict):
    for message in record["messages"]:
        if message.get("role") == "assistant":
            yield from message.get("tool_calls") or []


# ---- R1, R2, R4 ------------------------------------------------------------------------------------------


def derive_habits(records: list[dict]) -> tuple[dict, dict]:
    """R1's habit values taken from records as M6 took HABITS from synth.jsonl (the module docstring,
    --derive-habits): per parameter of HABITS, the value its tool's assistant calls set most often, unless
    that is the schema default; ties go to the smaller canonical JSON. The habits ({(tool, param): value},
    without the parameters that have none) and the account per "tool.param": the default, M6's value, the
    derived one (None: no habit), the calls of the tool and the count of every value set (canonical JSON,
    ABSENT for a call without the parameter)."""
    values: dict[tuple[str, str], Counter[str]] = {key: Counter() for key in HABITS}
    for record in records:
        for c in _calls(record):
            fn = c["function"]
            for tool, param in HABITS:
                if fn.get("name") != tool:
                    continue
                args = json.loads(fn.get("arguments") or "{}")
                if isinstance(args, dict):
                    values[tool, param][canonical_json(args[param]) if param in args else ABSENT] += 1
    habits, account = {}, {}
    for (tool, param), counts in values.items():
        default = schemas.DEFAULTS[tool][param]
        top = min(((n, v) for v, n in counts.items() if v != ABSENT), key=lambda nv: (-nv[0], nv[1]),
                  default=None)
        derived = None if top is None or json.loads(top[1]) == default else json.loads(top[1])
        if derived is not None:
            habits[tool, param] = derived
        account[f"{tool}.{param}"] = {"default": default, "m6": HABITS[tool, param], "derived": derived,
                                      "calls": sum(counts.values()), "values": dict(sorted(counts.items()))}
    return habits, account


def rule_r1(record: dict, request: dict, habits: dict | None = None) -> tuple[dict, dict]:
    """R1 on a copy of record, deleting habits (default HABITS, M6's); the counts: deleted and kept_count per
    "tool.param" of HABITS and AT_DEFAULT, calls_changed and records_changed (0 or 1)."""
    habits = HABITS if habits is None else habits
    out = copy.deepcopy(record)
    today = _today_of(record)
    count = (request.get("slots") or {}).get("count")
    deleted: Counter[str] = Counter()
    kept: Counter[str] = Counter()
    changed = 0
    for c in _calls(out):
        fn = c["function"]
        name, raw = fn.get("name"), fn.get("arguments") or "{}"
        args = json.loads(raw)
        if not isinstance(args, dict):
            continue
        new = dict(args)
        for (tool, param), value in habits.items():
            if name == tool and param in new and new[param] == value:
                if count == value:
                    kept[f"{tool}.{param}"] += 1
                else:
                    del new[param]
                    deleted[f"{tool}.{param}"] += 1
        for (tool, param), value in AT_DEFAULT.items():
            if name == tool and param in new and new[param] == value:
                del new[param]
                deleted[f"{tool}.{param}"] += 1
        if new != args:
            _checked(name, new, record.get("id", "?"), today)
            fn["arguments"] = serialize_like(raw, new)
            changed += 1
    keys = [f"{t}.{p}" for t, p in [*HABITS, *AT_DEFAULT]]
    return out, {"deleted": {k: deleted[k] for k in keys},
                 "kept_count": {f"{t}.{p}": kept[f"{t}.{p}"] for t, p in HABITS},
                 "calls_changed": changed, "records_changed": int(changed > 0)}


def rule_r2(record: dict, request: dict) -> tuple[dict, dict]:
    """R2 on a copy of record (any scenario; the CLI applies it to clarify trajectories): every
    ask_clarification call's missing becomes the request's slots.missing."""
    out = copy.deepcopy(record)
    wanted = (request.get("slots") or {}).get("missing")
    if wanted is not None and (not isinstance(wanted, list) or not set(wanted) <= set(schemas.MISSING)):
        raise TransformError(f"{request.get('id')}: slots.missing {wanted!r} is not in {schemas.MISSING}")
    counts = {"calls": 0, "changed": 0, "no_slots_missing": 0}
    for c in _calls(out):
        fn = c["function"]
        if fn.get("name") != ASK:
            continue
        counts["calls"] += 1
        if not wanted:
            counts["no_slots_missing"] += 1
            continue
        raw = fn.get("arguments") or "{}"
        args = json.loads(raw)
        if args.get("missing") == wanted:
            continue
        new = dict(args)
        new["missing"] = list(wanted)  # replaced in place, or appended when absent
        _checked(ASK, new, record.get("id", "?"), _today_of(record))
        fn["arguments"] = serialize_like(raw, new)
        counts["changed"] += 1
    return (record if counts["changed"] == 0 else out), counts


def first_clarification_step(record: dict) -> int | None:
    """The 1-based index of the first assistant message with an ask_clarification call; None without one."""
    step = 0
    for message in record["messages"]:
        if message.get("role") != "assistant":
            continue
        step += 1
        if any((c.get("function") or {}).get("name") == ASK for c in message.get("tool_calls") or []):
            return step
    return None


def r3_train_requests(train: list[dict], kept: list[dict]) -> tuple[list[dict], dict]:
    """R3's requests among the train requests: the clarify ones with slots.missing, except those whose kept
    trajectory (after R4) asks only at its second step, since a first-step record from the same request
    would teach two different first steps for one user text. A request whose trajectory R4 dropped, or which
    asks at step 1 already, stays. The counts: eligible, skipped_conflict, no_slots_missing."""
    first_steps = {r["id"]: first_clarification_step(r) for r in kept if r.get("scenario") == "clarify"}
    eligible, skipped, no_missing = [], 0, 0
    for req in train:
        if req.get("scenario") != "clarify":
            continue
        if not (req.get("slots") or {}).get("missing"):
            no_missing += 1
            continue
        if first_steps.get(req["id"]) not in (None, 1):
            skipped += 1
            continue
        eligible.append(req)
    return eligible, {"eligible": len(eligible), "skipped_conflict": skipped, "no_slots_missing": no_missing}


def rule_r4(records: list[dict]) -> tuple[list[dict], dict]:
    """R4: the records without the clarify-scenario ones whose first ask_clarification is at step 3 or
    later; the counts by first-clarification position and the dropped ids."""
    kept, counts, dropped = [], Counter(), []
    for r in records:
        if r.get("scenario") != "clarify":
            kept.append(r)
            continue
        step = first_clarification_step(r)
        if step is None:
            counts["no_clarification"] += 1
        elif step == 1:
            counts["first_step"] += 1
        elif step == 2:
            counts["second_step"] += 1
        else:
            counts["late_dropped"] += 1
            dropped.append(r["id"])
            continue
        kept.append(r)
    return kept, {"first_step": counts["first_step"], "second_step": counts["second_step"],
                  "late_dropped": counts["late_dropped"], "no_clarification": counts["no_clarification"],
                  "dropped_ids": dropped}


# ---- new records -----------------------------------------------------------------------------------------

_EMPTY_FEED = b'<?xml version="1.0" encoding="UTF-8"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>'


class _EmptyFeed:
    """A fetcher whose every answer is an arXiv feed without entries."""

    def get(self, url: str) -> Fetched:
        return Fetched(url, url, 200, _EMPTY_FEED, "application/atom+xml")


class _Script:
    """A model that answers with the given turns, then with an empty text."""

    name = RUN

    def __init__(self, turns: list[AssistantTurn]) -> None:
        self.turns = list(turns)

    def next_turn(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        return self.turns.pop(0) if self.turns else AssistantTurn("", [], RUN)


def _ordered(tool: str, args: dict) -> dict:
    """args in the order the tool's schema lists its parameters."""
    order = next(t["function"]["parameters"]["properties"] for t in schemas.TOOLS
                 if t["function"]["name"] == tool)
    return {k: args[k] for k in order if k in args}


def _call_id(record_id: str, step: int) -> str:
    return "call_m6_" + hashlib.sha256(f"{record_id}:{step}".encode()).hexdigest()[:24]


def _tools(today: str) -> dict:
    tools = arxiv_tools(_EmptyFeed(), build_search_url=arxiv.build_search_url, parse_atom=arxiv.parse_atom,
                        extract_section=arxiv.extract_section, today=today)
    tools["list_notes"] = lambda args: ToolOutcome([])
    return tools


def scripted_messages(text: str, steps: list[tuple[str, dict]], today: str, record_id: str) -> list[dict]:
    """The history the agent loop records for text when the model makes one call per step, cut after the
    last step (and its tool reply when that step asks for clarification). TransformError when the loop
    did not accept a step as given."""
    turns = []
    for i, (name, args) in enumerate(steps, 1):
        args = _ordered(name, args)
        raw = json.dumps(args, ensure_ascii=False)
        turns.append(AssistantTurn("", [ToolCall(_call_id(record_id, i), name, raw, args)], RUN))
    executor = ToolExecutor(_tools(today), check_call, today=today)
    transcript = run_turn(text, model=_Script(turns), executor=executor, decide=policy.decide, today=today,
                          max_steps=len(steps) + 1)
    messages = transcript.messages
    positions = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
    wanted = [t.to_api() for t in turns]
    if len(positions) < len(steps) or [messages[i] for i in positions[:len(steps)]] != wanted:
        raise TransformError(f"{record_id}: the agent loop did not accept the scripted steps")
    end = positions[len(steps) - 1] + 1
    if steps[-1][0] == ASK:
        end += 1
    return copy.deepcopy(messages[:end])


def new_record(request: dict, messages: list[dict], *, record_id: str, family: str, scenario: str) -> dict:
    return {"family": family, "history": "agent", "id": record_id, "intent": request["intent"],
            "lang": request["lang"], "messages": messages, "prompt_hash": PROMPT_HASH, "run": RUN,
            "scenario": scenario, "schema_hash": schemas.SCHEMA_HASH,
            "tools": copy.deepcopy(schemas.MODEL_TOOLS)}


def _m6_family(rule: str, family: str) -> str:
    return family if str(family).startswith("m6/") else f"m6/{rule}/{family}"


def clarify_question(missing: list[str], lang: str, key: str) -> str:
    """One of the two questions per slot, chosen by key's hash; several slots are joined."""
    pick = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 2
    return ("" if lang == "zh" else " ").join(CLARIFY_QUESTIONS[slot][lang][pick] for slot in missing)


def clarification_record(request: dict, today: str, *, record_id: str | None = None) -> dict:
    """R3: one step, ask_clarification(question, missing = slots.missing)."""
    record_id = record_id or request["id"]
    missing = list(request["slots"]["missing"])
    args = {"question": clarify_question(missing, request["lang"], request["id"]), "missing": missing}
    messages = scripted_messages(request["text"], [(ASK, args)], today, record_id)
    return new_record(request, messages, record_id=record_id, family=_m6_family("r3", request["family"]),
                      scenario="clarify")


def list_notes_record(request: dict, today: str) -> dict:
    """R5: one step, list_notes(tag, since)."""
    slots = request["slots"]
    messages = scripted_messages(request["text"], [("list_notes", {"tag": slots["tags"][0],
                                                                    "since": slots["since"]})],
                                 today, request["id"])
    return new_record(request, messages, record_id=request["id"], family=request["family"], scenario="normal")


def relax_constraint(args: dict) -> tuple[dict, str]:
    """The relaxed arguments and what was dropped: the dates when there are any, else the categories."""
    if any(k in args for k in DATE_KEYS):
        return {k: v for k, v in args.items() if k not in DATE_KEYS}, "date"
    return {k: v for k, v in args.items() if k != "categories"}, "categories"


def relax_record(request: dict, today: str, *, record_id: str) -> dict:
    """R6: search_arxiv with the request's constraint, the loop's EMPTY_RESULTS reply, the relaxed search."""
    slots = request["slots"]
    first = {"query": slots["topic"], **{k: slots[k] for k in ("categories", *DATE_KEYS) if k in slots}}
    second, _ = relax_constraint(first)
    messages = scripted_messages(request["text"], [("search_arxiv", first), ("search_arxiv", second)],
                                 today, record_id)
    reply = json.loads(messages[3]["content"])
    if reply.get("error") != schemas.EMPTY_RESULTS:
        raise TransformError(f"{record_id}: the first search did not come back as {schemas.EMPTY_RESULTS}")
    return new_record(request, messages, record_id=record_id, family=_m6_family("r6", request["family"]),
                      scenario="recover")


def term_record(request: dict, today: str) -> dict:
    """R7: one step, search_arxiv(query = the English term)."""
    messages = scripted_messages(request["text"], [("search_arxiv", {"query": request["slots"]["topic"]})],
                                 today, request["id"])
    return new_record(request, messages, record_id=request["id"], family=request["family"], scenario="normal")


# ---- requests: the guard, the template audit, generation --------------------------------------------------


class EvalGuard:
    """requests.Deduper with the evaluation texts and the existing train texts registered unconditionally
    (offering them could drop an evaluation text that is a near duplicate of another one)."""

    def __init__(self, *, eval_texts: list[str], train_texts: list[str]) -> None:
        self.deduper = rq.Deduper()
        for text in eval_texts:
            self._register(text, "dev")
        for text in train_texts:
            self._register(text, "train")

    def _register(self, text: str, split: str) -> None:
        sh = rq.shingles(rq.mask_text(text))
        self.deduper.texts.add(normalize_text(text))
        self.deduper.all.add(sh)
        if split != "train":
            self.deduper.eval.add(sh)

    def max_eval_jaccard(self, text: str) -> float:
        return self.deduper.eval.max_jaccard(rq.shingles(rq.mask_text(text)))

    def overlaps_eval(self, text: str) -> bool:
        return self.max_eval_jaccard(text) >= self.deduper.train_vs_eval

    def offer(self, text: str) -> str | None:
        """Deduper's rules for a train request (exact, near, train_vs_eval); kept requests are registered."""
        return self.deduper.offer(text, "train")

    def offer_loose(self, text: str) -> str | None:
        """Exact duplicates and evaluation overlaps only; a kept text is registered."""
        if self.overlaps_eval(text):
            return "train_vs_eval"
        if normalize_text(text) in self.deduper.texts:
            return "exact"
        self._register(text, "train")
        return None


def audit_new_templates(templates: dict[str, tuple[str, ...]]) -> dict:
    """template_audit's wording rule for new templates: each against every existing template of its language
    (any intent, kind or split) and the other new ones must stay below TEMPLATE_LIMIT."""
    existing = [(lang, rq.template_wording(t)) for _, _, lang, _, t in rq.template_families()]
    best, pairs, over = 0.0, 0, []
    new = [(lang, t) for lang, ts in templates.items() for t in ts]
    for x, (lang, template) in enumerate(new):
        wording = rq.template_wording(template)
        others = [(w, "requests.TEMPLATES") for l2, w in existing if l2 == lang]
        others += [(rq.template_wording(t), t) for l2, t in new[x + 1:] if l2 == lang]
        for w, name in others:
            pairs += 1
            j = rq.jaccard(wording, w)
            best = max(best, j)
            if j >= rq.TEMPLATE_LIMIT:
                over.append([round(j, 4), template, name])
    return {"max_jaccard": round(best, 4), "pairs": pairs, "limit": rq.TEMPLATE_LIMIT, "over": over,
            "ok": not over}


def _fill_requests(cells: dict[str, list[tuple[str, str, str]]], n: int, *, kind: str, scenario: str,
                   seed: int, purpose: str, corpus: rq.Corpus, topics: tuple, guard: EvalGuard) -> list[dict]:
    """n requests, half per language (the odd one zh), round robin over each language's (intent, family,
    template) cells in a seeded order, each offered to the guard."""
    out = []
    for i, lang in enumerate(("zh", "en")):
        wanted = n // 2 + (n % 2 if i == 0 else 0)
        families = list(cells[lang])
        rng = random.Random(f"{seed}:m6:{purpose}:{lang}")
        rng.shuffle(families)
        got = attempt = 0
        while got < wanted and attempt < ATTEMPTS_PER_REQUEST * max(wanted, 1):
            intent, family, template = families[attempt % len(families)]
            attempt += 1
            text, slots = rq.fill(template, lang, topics, corpus.window, rng, intent=intent, kind=kind)
            if guard.offer(text) is None:
                out.append({"family": family, "intent": intent, "lang": lang, "scenario": scenario,
                            "slots": slots, "split": "train", "text": text})
                got += 1
    return out


def generate_clarify_requests(n: int, *, seed: int, corpus: rq.Corpus, split_seed: int,
                              guard: EvalGuard) -> list[dict]:
    """R3's new requests: requests.py's clarify templates of the train split and the train topics (the
    split of split_seed, the registered requests' seed), filled with a stream of their own."""
    assignment = rq.split_templates(split_seed)
    cells: dict[str, list] = {"zh": [], "en": []}
    for intent, kind, lang, index, template in rq.template_families():
        family = rq.family_name(intent, lang, index)
        if kind == "clarify" and assignment[family] == "train":
            cells[lang].append((intent, family, template))
    topics = rq.split_topics(corpus, split_seed)["train"]
    return _fill_requests(cells, n, kind="clarify", scenario="clarify", seed=seed, purpose="r3",
                          corpus=corpus, topics=topics, guard=guard)


def generate_list_notes_requests(n: int, *, seed: int, corpus: rq.Corpus, split_seed: int,
                                 guard: EvalGuard) -> list[dict]:
    """R5's requests from LIST_NOTES_TEMPLATES, families m6/list_notes/<lang>/<index>."""
    cells = {lang: [("list_notes", f"m6/list_notes/{lang}/{i}", t) for i, t in enumerate(templates)]
             for lang, templates in LIST_NOTES_TEMPLATES.items()}
    topics = rq.split_topics(corpus, split_seed)["train"]
    return _fill_requests(cells, n, kind="normal", scenario="normal", seed=seed, purpose="r5", corpus=corpus,
                          topics=topics, guard=guard)


def check_terms(terms: dict, corpus: rq.Corpus, *, held_out: tuple[str, ...] | None = None,
                minimum: int = MIN_TERMS) -> list[str]:
    """Why the term list cannot be used ([] when it can). Refused: an English phrase containing a corpus
    topic phrase (or a topic id read as words) or a HELD_OUT_EN synonym; a Chinese term equal to a corpus
    topic name, or containing a HELD_OUT_ZH synonym or the name of a held_out topic (topic ids)."""
    problems = []
    if not isinstance(terms, dict):
        return ["the term list must be a JSON object mapping a Chinese term to an English phrase"]
    if len(terms) < minimum:
        problems.append(f"{len(terms)} terms, fewer than {minimum}")
    phrases = {p.casefold() for t in corpus.topics for p in t.phrases}
    phrases |= {t.id.replace("-", " ").casefold() for t in corpus.topics}
    names = {normalize_text(t.name).casefold() for t in corpus.topics}
    held_names = {normalize_text(t.name) for t in corpus.topics if held_out and t.id in held_out}
    seen: set[str] = set()
    for zh, en in terms.items():
        if not isinstance(en, str) or not 1 <= len(en) <= 300 or not _HAN.search(str(zh)):
            problems.append(f"{zh!r}: needs Chinese on the left and an English phrase of 1-300 characters")
            continue
        low = en.casefold()
        if low in seen:
            problems.append(f"{zh}: {en} appears twice")
        seen.add(low)
        hits = [p for p in phrases if re.search(rf"(?<![a-z0-9]){re.escape(p)}(?![a-z0-9])", low)]
        hits += [s for s in HELD_OUT_EN if s in low]  # stems: plain substrings
        if hits:
            problems.append(f"{zh}: {en} contains the corpus phrase or held-out synonym {min(hits)!r}")
        zh_norm = normalize_text(zh)
        if zh_norm.casefold() in names:
            problems.append(f"{zh}: is the name of a corpus topic")
        held = [s for s in (*HELD_OUT_ZH, *sorted(held_names)) if s in zh_norm]
        if held:
            problems.append(f"{zh}: contains the held-out name or synonym {held[0]!r}")
    return problems


def term_requests(terms: dict, limit: int, guard: EvalGuard, *,
                  family_prefix: str = "m6/terms") -> tuple[list[dict], int]:
    """R7's requests: each term with template i % 4, then again with (i + 2) % 4 while under limit; the
    number dropped as evaluation overlaps (exact duplicates are dropped silently). Families
    <family_prefix>/zh/<template> (FT4's glossary hook passes its own prefix)."""
    items = list(terms.items())
    plan = [(i, (i % 4)) for i in range(len(items))] + [(i, (i + 2) % 4) for i in range(len(items))]
    out, overlaps = [], 0
    for i, k in plan:
        if len(out) >= limit:
            break
        zh, en = items[i]
        text = rq.tidy(TERM_TEMPLATES[k].format(term=zh), "zh")
        reason = guard.offer_loose(text)
        if reason == "train_vs_eval":
            overlaps += 1
        if reason is None:
            out.append({"family": f"{family_prefix}/zh/{k}", "intent": TERM_INTENT, "lang": "zh",
                        "scenario": "normal", "slots": {"term_zh": zh, "topic": en}, "split": "train",
                        "text": text})
    return out, overlaps


# ---- FT4 (module docstring; docs/m7-本地提升计划.md WP6) -----------------------------------------------

FT4_GENERATOR = "transform_synth-ft4-2026-10-30"
FT4_RUN = "ft4-transform"  # data.reasoning.FT4_RUN: the records FT4 builds, derived copies and new ones
FT4_TARGETS = {"r9_new": 100, "r10_new": 150, "r11_new": 40, "r14_defref": 250, "r14_explain": 80,
               "r13_terms": 500}
FT4_NEW_OPTIONS = ("r11_new", "r14_defref", "r14_explain", "r13_terms")  # the targets of the new records
FT4_NOT_RUN = ["R11 relative dates", "R14 definite references", "R14 explain_tools", "R13 glossary terms"]
TRANSIENT = (schemas.RATE_LIMITED, schemas.UPSTREAM_ERROR)
NOTE_TOOLS = ("search_notes", "list_notes")
NOTE_INTENTS = ("search_notes", "list_notes")
PICK_FAMILIES = ("search_notes/zh/3", "search_notes/en/3")  # the user remembers one note and wants it found
S4B_SEARCHES = 3
RELATIVE_SORT_INTENTS = ("find_recent_papers",)  # R11's search intent (RELATIVE_INTENTS): relative_sort_by's
RELATIVE_SORT_RULE = ("the majority effective sort_by of the teacher's search_arxiv calls with a date_from "
                      "in the find_recent_papers records of every input, a call without sort_by counted as "
                      "the default (a tie goes to the default); R11's searches leave sort_by out when the "
                      "majority is the default")
# The mid-turn text questions coverage_c3_eos.py labelled C (asks the user to name a paper, tag or topic in
# place of ask_clarification), by (record id, turn), with the missing slot each question asks for. The 4
# borderline ones (B) are left as they are.
R10_LABELLED = {
    ("req-train-0076", 4): ["tags"], ("req-train-0077", 5): ["topic"], ("req-train-0602", 3): ["paper_id"],
    ("req-train-1083", 1): ["paper_id"], ("req-train-1119", 3): ["paper_id"], ("req-train-1385", 2): ["tags"],
    ("req-train-1394", 2): ["tags"], ("req-train-1621", 2): ["tags"], ("req-train-1665", 4): ["tags"],
    ("req-train-1686", 3): ["paper_id"], ("req-train-1844", 3): ["tags"], ("req-train-2181", 2): ["tags"],
    ("req-train-2272", 1): ["paper_id"], ("req-train-2420", 4): ["paper_id"],
    ("req-train-2457", 2): ["paper_id"], ("req-train-2655", 1): ["paper_id"]}
R10_BORDERLINE = ("req-train-0538", "req-train-0729", "req-train-1841", "req-train-2200")
_LATIN = re.compile("[A-Za-z]")
_CATEGORY_TAG = re.compile(r"^[a-z]+\.[A-Za-z]+$")  # cs.LG, stat.ML: an arXiv category, not a topic tag
# R13's glossary terms (glossary_requests): why a noted term gives no record, in the order the reasons are
# tried; the first three make a term ineligible (glossary_skip)
GLOSSARY_SKIPS = ("non_ascii", "latin_token_missing", "science_only", "already_in_r7", "refused")
SCIENCE_SECTION = "section/AI for Science.md"  # the AITD section R13's domain filter leaves out alone
_LATIN_TOKEN = re.compile(r"[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*")  # in a Chinese term: Adam, C4.5, KKT
_WORD = re.compile("[A-Za-z0-9]+")  # glossary_query's word

# R11: relative dates, counted back from today by relative_range.
RELATIVE_PHRASES = {
    "zh": {"yesterday": "昨天", "day_before": "前天", "this_week": "这周", "last_week": "上周",
           "last_month": "上个月"},
    "en": {"yesterday": "yesterday", "day_before": "the day before yesterday", "this_week": "this week",
           "last_week": "last week", "last_month": "last month"}}
RELATIVE_TEMPLATES = {
    "zh": ("把我{rel}写的笔记都列出来。", "{rel}我记的笔记里,打了 {tag} 标签的有哪些?",
           "{rel}有没有新的{topic}论文挂到 arXiv 上?帮我检索一下。",
           "想看{rel}提交到 arXiv 的 {topic} 论文,帮我搜一搜。"),
    "en": ("List the notes I wrote {rel}.", "Which of the notes I took {rel} carry the tag {tag}?",
           "Did any new {topic} papers go up on arXiv {rel}? Please check.",
           "Pull up arXiv papers about {topic} from {rel}.")}
# the intent of each template, by its index
RELATIVE_INTENTS = ("list_notes", "list_notes", "find_recent_papers", "find_recent_papers")

# R14: questions about the optional parameters, written from the tool schemas; {tool} and {param} are filled
# with optional_params().
EXPLAIN_TEMPLATES = {
    "zh": ("调用 {tool} 时,{param} 一定要给吗?不给会怎样?", "{tool} 的参数 {param} 能不能省略?",
           "用 {tool} 的时候可以不写 {param} 吗?"),
    "en": ("When I call {tool}, must I pass {param}? What happens without it?",
           "Can the {param} argument of {tool} be omitted?",
           "Is {param} required for {tool}, or can it be skipped?")}
# A definite reference in a template: the paper, the most relevant one, the top hit, 那篇, 排第一 ...
DEFREF = re.compile(r"\b(?:the|that) (?:[\w-]+ ){0,3}(?:paper|one|article|hit)\b|closest|best[- ]match|"
                    r"那篇|这篇|那一篇|排第一", re.IGNORECASE)


@dataclass
class Ft4Options:
    """What a run with --ft4 needs beyond M6's inputs: the train topics (R13's terms), the targets (over
    FT4_TARGETS), more evaluation texts for FT4's own guard, the glossary (the hook; None: not run) with its
    domain filter (science_only_terms: the terms left out, and the account), and more inputs whose dated
    searches join R11's sort_by count ((name, records); relative_sort_by)."""

    topics: tuple
    targets: dict = field(default_factory=dict)
    eval_texts: list = field(default_factory=list)
    glossary: Glossary | None = None
    science_only: frozenset = frozenset()
    domain: dict | None = None
    sort_inputs: list = field(default_factory=list)


@dataclass
class Relabel:
    """A turn an FT4 rule gives: its rule, the variant (for the counts), the assistant message and, for an
    ask_clarification, the loop's "asked" reply."""

    rule: str
    variant: str
    message: dict
    reply: dict | None = None


@dataclass
class Listing:
    """An untagged list_notes call of a history and the notes its reply listed: what R9 and R10 call the
    listing seen."""

    args: dict
    notes: list[dict]

    @property
    def complete(self) -> bool:
        """Whether the reply is all the user's notes: the call has no since and no tag and listed fewer notes
        than its limit (the schema default when absent). Otherwise a text speaks only of the notes listed."""
        limit = self.args.get("limit")
        if type(limit) is not int:
            limit = schemas.DEFAULTS["list_notes"]["limit"]
        return not self.args.get("since") and not self.args.get("tag") and len(self.notes) < limit


@dataclass
class Ft4Records:
    """apply_ft4's result: the input records (relabelled and cut where a rule fired, in input order), the
    derived records (id order), their changes rows ((id, k) order) and the counts."""

    records: list[dict]
    derived: list[dict]
    changes: list[dict]
    report: dict


def _turn_positions(messages: list[dict]) -> list[int]:
    """The index of every assistant message: turn k is messages[_turn_positions(messages)[k]]."""
    return [i for i, m in enumerate(messages) if m.get("role") == "assistant"]


def _name(c: dict) -> str:
    return (c.get("function") or {}).get("name") or ""


def _args(c: dict) -> dict | None:
    try:
        value = json.loads((c.get("function") or {}).get("arguments") or "{}")
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _signature(c: dict) -> str:
    """A call as the analysis compares calls: its name and its parsed arguments (the text when not JSON)."""
    args = _args(c)
    return canonical_json([_name(c), (c.get("function") or {}).get("arguments") if args is None else args])


def _parsed(message: dict | None) -> object:
    try:
        return json.loads((message or {}).get("content") or "")
    except json.JSONDecodeError:
        return None


def _error(message: dict | None) -> str | None:
    value = _parsed(message)
    return value.get("error") if isinstance(value, dict) and isinstance(value.get("error"), str) else None


def _replies(messages: list[dict], index: int) -> list[dict | None]:
    """The tool message of every call of the assistant message at index, in call order (None when none)."""
    found, j = {}, index + 1
    while j < len(messages) and messages[j].get("role") == "tool":
        found[messages[j].get("tool_call_id")] = messages[j]
        j += 1
    return [found.get(c.get("id")) for c in messages[index].get("tool_calls") or []]


def _rounds(history: list[dict]) -> list[tuple[list[dict], list[dict | None]]]:
    """(calls, their replies) of every assistant message with calls in history, in order."""
    return [(m["tool_calls"], _replies(history, i)) for i, m in enumerate(history)
            if m.get("role") == "assistant" and m.get("tool_calls")]


def _last_round(history: list[dict]) -> tuple[list[dict], list[dict | None]]:
    """The calls and replies of the step whose tool messages end history; ([], []) when none do."""
    j = len(history)
    while j > 0 and history[j - 1].get("role") == "tool":
        j -= 1
    if j == len(history) or j == 0 or not history[j - 1].get("tool_calls"):
        return [], []
    return history[j - 1]["tool_calls"], _replies(history, j - 1)


def _transient_last(history: list[dict]) -> bool:
    return any(_error(r) in TRANSIENT for r in _last_round(history)[1] if r is not None)


def _new_call_id(record_id: str, k: int, i: int) -> str:
    return "call_ft4_" + hashlib.sha256(f"{record_id}:{k}:{i}".encode()).hexdigest()[:24]


def _ask_message(record_id: str, k: int, question: str, missing: list[str], content: str,
                 today: str) -> tuple[dict, dict]:
    """An assistant message with one ask_clarification call (checked by check_call) and the loop's reply."""
    args = {"question": question, "missing": list(missing)}
    _checked(ASK, args, record_id, today)
    c = {"function": {"arguments": json.dumps(args, ensure_ascii=False), "name": ASK},
         "id": _new_call_id(record_id, k, 0), "type": "function"}
    reply = ToolResult(c["id"], ASK, {"question": question, "status": "asked"}).to_api()
    return {"content": content, "role": "assistant", "tool_calls": [c]}, reply


def _first_ask(record: dict) -> tuple[int, dict] | None:
    """The turn of the record's first ask_clarification call and that call's arguments; None without one."""
    for k, i in enumerate(_turn_positions(record["messages"])):
        for c in record["messages"][i].get("tool_calls") or []:
            if _name(c) == ASK:
                return k, _args(c) or {}
    return None


# R8


def _executed(history: list[dict]) -> Counter[str]:
    """How often each call ran in history, by the loop's call_key (defaults filled in): every call whose reply
    is not SKIPPED (a repeat the loop refused, or a call sent with ask_clarification)."""
    runs: Counter[str] = Counter()
    for calls, replies in _rounds(history):
        for c, reply in zip(calls, replies, strict=True):
            if reply is not None and _error(reply) != schemas.SKIPPED:
                runs[call_key(ToolCall.from_api(c))] += 1
    return runs


def rule_r8(history: list[dict], target: dict) -> tuple[str, list[dict]]:
    """R8's verdict on target and the failed calls: "relabel" when the step before it got a RATE_LIMITED or
    UPSTREAM_ERROR reply and target is not exactly the failed calls (names and parsed arguments as a
    multiset, its text aside); "identical" when it is; "at_run_limit" when a failed call already ran
    MAX_RUNS_PER_CALL times, so the loop would refuse it once more; "none" without a transient error."""
    calls, replies = _last_round(history)
    failed = [c for c, r in zip(calls, replies, strict=True) if r is not None and _error(r) in TRANSIENT]
    if not failed:
        return "none", []
    if sorted(map(_signature, target.get("tool_calls") or [])) == sorted(map(_signature, failed)):
        return "identical", failed
    runs = _executed(history)
    if any(runs[call_key(ToolCall.from_api(c))] >= MAX_RUNS_PER_CALL for c in failed):
        return "at_run_limit", failed
    return "relabel", failed


# R9


def listing_of(history: list[dict]) -> Listing | None:
    """The last untagged list_notes call in history whose reply listed at least one note, with those notes;
    None without such a call."""
    found = None
    for calls, replies in _rounds(history):
        for c, reply in zip(calls, replies, strict=True):
            args = _args(c)
            if _name(c) != "list_notes" or args is None or args.get("tag") or reply is None:
                continue
            notes = _parsed(reply)
            notes = [n for n in notes if isinstance(n, dict)] if isinstance(notes, list) else []
            if notes:
                found = Listing(args, notes)
    return found


def s4a(record: dict, history: list[dict]) -> Listing | None:
    """S4a: the listing seen when record is a note request (NOTE_INTENTS), history holds the listing, the
    listing is all the user's notes (Listing.complete: no since, no tag, fewer notes than its limit) and the
    step before met no transient error; None otherwise."""
    if record.get("intent") not in NOTE_INTENTS or _transient_last(history):
        return None
    listing = listing_of(history)
    return listing if listing is not None and listing.complete else None


def s4a_point(record: dict, history: list[dict], target: dict, k: int) -> Listing | None:
    """The listing when turn k (after history) is an S4a point: S4a holds and target only queries the notes
    again (NOTE_TOOLS), the situation R9's relabel takes and the coverage analysis counted; None otherwise."""
    names = [_name(c) for c in target.get("tool_calls") or []]
    if not k or not names or any(n not in NOTE_TOOLS for n in names):
        return None
    return s4a(record, history)


def s4b(history: list[dict]) -> int | None:
    """S4b: the number of search_arxiv calls in history when at least S4B_SEARCHES steps searched arXiv, the
    step before among them and without a transient error; None otherwise."""
    if _transient_last(history) or not any(_name(c) == "search_arxiv" for c in _last_round(history)[0]):
        return None
    rounds = _rounds(history)
    if sum(any(_name(c) == "search_arxiv" for c in calls) for calls, _ in rounds) < S4B_SEARCHES:
        return None
    return sum(_name(c) == "search_arxiv" for calls, _ in rounds for c in calls)


def _topic_tags(notes: list[dict]) -> list[str]:
    """The tags of the notes that are no arXiv category, in order of first appearance."""
    tags: list[str] = []
    for n in notes:
        for t in n.get("tags") or []:
            if isinstance(t, str) and not _CATEGORY_TAG.match(t) and t not in tags:
                tags.append(t)
    return tags


def _note_table(notes: list[dict], lang: str) -> str:
    head = ("| 笔记编号 | 标题 | 标签 | 日期 |" if lang == "zh" else "| Note | Title | Tags | Created |")
    rows = [f"| {n.get('note_id', '')} | {n.get('title', '')} | {', '.join(n.get('tags') or [])} | "
            f"{n.get('created', '')} |" for n in notes]
    return "\n".join([head, "|---|---|---|---|", *rows])


def _newest_first(notes: list[dict]) -> list[dict]:
    return sorted(notes, key=lambda n: str(n.get("created") or ""), reverse=True)  # stable: ties keep order


def _listing_head(listing: Listing, wanted: str | None, hits: int, cut: int | None, lang: str) -> str:
    """listing_answer's sentence before the table: what the table shows (the notes with the wanted tag, or
    the absence of that tag and the notes listed; all the user's notes only for a complete listing) and, when
    cut is not None, that it shows the cut newest of them."""
    n, mine = len(listing.notes), listing.complete
    if lang == "zh":
        if wanted and hits:
            what = (f"打了 {wanted} 标签的笔记有 {hits} 条" if mine
                    else f"列出的笔记里打了 {wanted} 标签的有 {hits} 条")
            return f"{what}:" if cut is None else f"{what},这是其中最新的 {cut} 条(按你要的数量):"
        if wanted and mine:
            lead, whole = f"笔记库里没有打了 {wanted} 标签的笔记。", f"现有的 {n} 条笔记"
        elif wanted:
            lead, whole = f"列出的 {n} 条笔记里没有打了 {wanted} 标签的。", "列出的笔记"
        else:
            lead, whole = "", f"笔记库里现有的 {n} 条笔记" if mine else f"列出的 {n} 条笔记"
        return f"{lead}{whole}如下:" if cut is None else f"{lead}这是{whole}中最新的 {cut} 条(按你要的数量):"
    newest = "is the newest" if cut == 1 else f"are the {cut} newest"
    if wanted and hits:
        what = (f"{hits} of {'your notes' if mine else 'the notes listed'} "
                f"{'carries' if hits == 1 else 'carry'} the tag {wanted}")
        if cut is None:
            return f"{what}:"
        return f"{what}; {'this' if cut == 1 else 'these'} {newest}, as you asked:"
    if wanted and mine:
        lead, whole = f"None of your notes carries the tag {wanted}. ", f"the {n} notes you have"
    elif wanted:
        lead, whole = f"None of the {n} notes listed carries the tag {wanted}. ", "the notes listed"
    else:
        lead, whole = "", f"the {n} notes you have" if mine else f"the {n} notes listed"
    if cut is None:
        return f"{lead}These are {whole}:"
    return f"{lead}{'This' if cut == 1 else 'These'} {newest} of {whole}, as you asked:"


def listing_answer(listing: Listing, slots: dict, lang: str) -> str:
    """R9's closing answer to a list request, from the listing seen: the notes with the request's tag, else
    with its topic's tag (topic ids are note tags), when any; with the absence of that tag said, all notes
    listed otherwise; as a table. It speaks of all the user's notes only when the listing is complete
    (Listing.complete), of the notes listed otherwise. With a count in the request (slots.count), the table
    holds that many, newest first, and the text says so (all of them when there are not more). Declarative: it
    asks nothing back."""
    count = slots.get("count") if type(slots.get("count")) is int and slots["count"] > 0 else None
    wanted = next((w for w in ((slots.get("tags") or [None])[0], slots.get("topic_id")) if w), None)
    hits = [n for n in listing.notes if wanted and wanted in (n.get("tags") or [])]
    shown = hits or listing.notes
    if count is not None:
        shown = _newest_first(shown)
    cut = count if count is not None and count < len(shown) else None
    head = _listing_head(listing, wanted, len(hits), cut, lang)
    return f"{head}\n\n{_note_table(shown[:cut] if cut else shown, lang)}"


def _candidate(note: dict, zh: bool) -> str:
    ident, title = note.get("arxiv_id") or note.get("note_id") or "", note.get("title") or ""
    if not title:
        return ident
    return f"{ident}({title})" if zh else f"{ident} ({title})"


def pick_question(notes: list[dict], lang: str) -> str:
    """The question of R9's ask for a remembered note: which of these notes (ids, titles) the user means."""
    zh = lang == "zh"
    names = [_candidate(n, zh) for n in notes]
    if zh:
        return f"你想找的是哪一篇:{'、'.join(names[:-1])},还是 {names[-1]}?"
    return f"Which one do you mean: {', '.join(names[:-1])} or {names[-1]}?"


def _pick_text(hits: int, tag: str, listing: Listing, lang: str) -> str:
    """The declarative text of R9's ask for a remembered note: how many listed notes carry the topic's tag."""
    if lang == "zh":
        return (f"你的笔记里有 {hits} 条带 {tag} 标签。" if listing.complete
                else f"列出的笔记里有 {hits} 条带 {tag} 标签。")
    return f"{hits} of {'your notes' if listing.complete else 'the notes listed'} carry the tag {tag}."


def search_report(history: list[dict], lang: str) -> str:
    """R9's honest report after the arXiv searches of history: the queries and every paper they returned (ids
    and titles from the tool results only), declarative."""
    queries: list[str] = []
    papers: dict[str, str] = {}
    for calls, replies in _rounds(history):
        for c, reply in zip(calls, replies, strict=True):
            if _name(c) != "search_arxiv":
                continue
            queries.append(str((_args(c) or {}).get("query", "")))
            items = _parsed(reply)
            items = items.get("items") if isinstance(items, dict) else items
            for item in items if isinstance(items, list) else []:
                if isinstance(item, dict) and isinstance(item.get("arxiv_id"), str):
                    papers.setdefault(item["arxiv_id"], str(item.get("title") or ""))
    zh = lang == "zh"
    quoted = "、".join(f"「{q}」" for q in queries) if zh else ", ".join(f'"{q}"' for q in queries)
    head = (f"我已经在 arXiv 上检索了 {len(queries)} 次,检索词依次是 {quoted}。" if zh
            else f"I searched arXiv {len(queries)} times, with the queries {quoted}.")
    if papers:
        lines = "\n".join(f"- {aid} {title}".rstrip() for aid, title in papers.items())
        intro = "检索结果里出现过的论文如下:" if zh else "These are the papers the searches returned:"
        body = f"{intro}\n{lines}"
        tail = ("我没有再继续检索;上面就是这几次检索返回的全部论文。" if zh
                else "I stopped searching here; the list above is everything the searches returned.")
    else:
        body = "这几次检索都没有返回论文。" if zh else "None of these searches returned a paper."
        tail = ("我没有再继续检索,这次没有找到符合要求的论文。" if zh
                else "I stopped searching here and found no paper that fits the request.")
    return f"{head}\n\n{body}\n\n{tail}"


def _closing(record: dict, request: dict, k: int, listing: Listing, key: str, today: str) -> Relabel:
    """R9's closing turn at an S4a point of record (module docstring). A request that wants one remembered
    note found: with two or more listed notes carrying its topic's tag (topic ids are note tags), an ask for
    the paper that names them (variant s4a_pick_ask); else the listing answer for that tag, which presents the
    one note (s4a_pick_one) or says that none carries it and shows the listing (s4a_pick_none). Any other note
    request: the listing answer (s4a_list)."""
    lang = record.get("lang") or "zh"
    slots = request.get("slots") or {}
    if record.get("family") not in PICK_FAMILIES:
        return Relabel("R9", "s4a_list", {"content": listing_answer(listing, slots, lang),
                                          "role": "assistant"})
    tag = slots.get("topic_id") or (slots.get("tags") or [None])[0]
    hits = [n for n in listing.notes if tag and tag in (n.get("tags") or [])]
    if len(hits) >= 2:
        message, reply = _ask_message(key, k, pick_question(hits, lang), ["paper_id"],
                                      _pick_text(len(hits), tag, listing, lang), today)
        return Relabel("R9", "s4a_pick_ask", message, reply)
    answer = listing_answer(listing, {"tags": [tag]} if tag else {}, lang)
    return Relabel("R9", "s4a_pick_one" if hits else "s4a_pick_none",
                   {"content": answer, "role": "assistant"})


# R10

_QUESTION_END = ("?", "？")
_CLOSERS = "\"'”’）)」】*`"
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.、)）])\s*\S")
_LIST_MARKER = re.compile(r"^(\s*)((?:[-*•]|\d+[.、)）])\s*)")
# What a sentence or line that elaborates the question right before it starts with (the owner's list; 或,
# but not 或许, "perhaps"): it moves with that question.
_ELABORATION = re.compile(r"^\s*(?:比如|例如|或者|或(?!许)|e\.g\.|for example\b|such as\b|or\b)",
                          re.IGNORECASE)
# A full stop that ends no sentence: after a common abbreviation (e.g., i.e., etc., vs., cf., et al.,
# approx.) or after a single letter (J. Smith; the "g." of "e.g.").
_ABBREVIATION = re.compile(r"(?:(?<![A-Za-z])(?:e\.g|i\.e|etc|vs|cf|et\s+al|approx)|(?<![A-Za-z])[A-Za-z])"
                           r"\.$", re.IGNORECASE)
NOT_TRAILING, PARTIAL_LIST = "not_trailing", "partial_list"  # why split_question's split is not clean
_Piece = tuple[int, int, bool]  # split_question's pieces: (line, column where the piece starts, moved)


class Split(NamedTuple):
    """split_question's result: the declarative text (before the first question), the question (from it on)
    and why the split is not clean (NOT_TRAILING, PARTIAL_LIST; None when it is)."""

    content: str
    question: str
    skipped: str | None


def _sentence_spans(line: str) -> list[tuple[int, str]]:
    """(start, sentence) of every sentence of line: cut after every 。!?！？ (with the quotes or brackets that
    close it) and after a full stop before a space that does not follow an abbreviation or a single letter
    (_ABBREVIATION); the pieces keep their spacing, so joining them gives line back; blank pieces are left
    out."""
    out, start, i = [], 0, 0
    while i < len(line):
        end = None
        if line[i] in "。！？!?":
            end = i + 1
            while end < len(line) and line[end] in "。！？!?" + _CLOSERS:
                end += 1
        elif line[i] == "." and (i + 1 == len(line) or line[i + 1].isspace()) \
                and not _ABBREVIATION.search(line, 0, i + 1):
            end = i + 1
        if end is None:
            i += 1
            continue
        out.append((start, line[start:end]))
        start = i = end
    if start < len(line):
        out.append((start, line[start:]))
    return [(at, s) for at, s in out if s.strip()]


def _sentences(line: str) -> list[str]:
    """The sentences of line (_sentence_spans)."""
    return [s for _, s in _sentence_spans(line)]


def _is_question(sentence: str) -> bool:
    return sentence.rstrip().rstrip(_CLOSERS).endswith(_QUESTION_END)


def _ends_sentence(item: str) -> bool:
    """Whether a list item ends its sentence (。.!?;, full or half width, before closing quotes or marks; a
    full stop after an abbreviation or a single letter does not); one that does not (", or", a bare name)
    runs on into the next item."""
    end = item.rstrip().rstrip(_CLOSERS)
    if end.endswith(".") and _ABBREVIATION.search(end):
        return False
    return end.endswith(("。", ".", "!", "?", "！", "？", ";", "；"))


def _body(line: str) -> int:
    """Where the text of a list item starts (after its marker); 0 for any other line."""
    marker = _LIST_MARKER.match(line) if _LIST_ITEM.match(line) else None
    return marker.end() if marker else 0


def _line_pieces(line: str, number: int, moved: bool, *, lead: bool = False) -> tuple[list[_Piece], bool]:
    """The pieces of a line (number) and whether its last one moved: a sentence moves when it asks or when it
    elaborates (_ELABORATION) a piece that moved right before it (moved: whether the piece before the line
    did); a list item's first sentence is read without its marker, which goes with it. With lead, the line
    leads a list that asks: its last sentence (the one its colon ends) moves too."""
    body = _body(line)
    spans = _sentence_spans(line[body:])
    pieces = []
    for n, (at, sentence) in enumerate(spans):
        moved = (lead and n == len(spans) - 1) or _is_question(sentence) or (
            moved and _ELABORATION.match(sentence) is not None)
        pieces.append((number, 0 if n == 0 else body + at, moved))
    return pieces, moved


def _run_on(items: list[str], asking: set[int]) -> set[int]:
    """The items of a list that run on (_ends_sentence) into a later item of their sentence that asks: they
    move whole, as part of that question."""
    out, start = set(), 0
    for end in range(len(items)):
        if not _ends_sentence(items[end]) and end + 1 < len(items):
            continue
        last = max((n for n in range(start, end + 1) if n in asking), default=None)
        if last is not None:
            out.update(range(start, last))
        start = end + 1
    return out


def split_question(text: str) -> Split:
    """The declarative text and the question of a text that asks the user something (R10), and whether the
    split is clean. The text is read in pieces: table rows and code blocks stay; a sentence moves when it
    asks, or when it elaborates (_ELABORATION: 比如, 例如, 或者, 或, e.g., for example, such as, or) a piece
    that moved right before it, as a sentence of its line or a line of its own; the list items after a piece
    that moved and ended its line are its options and move; of a list led by a line ending with a colon (one
    that asks: an item holds a question) the sentence the colon ends moves, and so do the items that ask, run
    on into an asking item (_run_on) or elaborate. The content is the text before the first piece that moves,
    the question the text from it on (its lines stripped, blank ones left out). The split is clean (skipped
    None) when everything from that piece on moves, so the questions are the text's trailing block; otherwise
    skipped is PARTIAL_LIST when a list that asks has an item, or part of one, that does not move, else
    NOT_TRAILING. A text that asks nothing gives ("the text", "", None). Blank-line runs closed up."""
    lines = text.split("\n")
    pieces: list[_Piece] = []
    partial = fence = moved = options = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.lstrip().startswith("```"):
            fence = not fence
        if not line.strip():
            i += 1
            continue
        if fence or line.lstrip().startswith(("```", "|")):
            pieces.append((i, 0, False))
            moved = options = False
            i += 1
            continue
        if options and _LIST_ITEM.match(line):  # an option of the piece that moved and ended its line
            pieces.append((i, 0, True))
            i += 1
            continue
        j = i + 1
        while line.rstrip().endswith((":", "：")) and j < len(lines) and _LIST_ITEM.match(lines[j]):
            j += 1
        items = lines[i + 1:j]
        asking = {n for n, item in enumerate(items) if any(map(_is_question, _sentences(item[_body(item):])))}
        if not asking:  # a line of its own, the items after it (if any) read on their own
            got, moved = _line_pieces(line, i, moved)
            pieces += got
            options = moved
            i += 1
            continue
        got, moved = _line_pieces(line, i, moved, lead=True)
        pieces += got
        run_on = _run_on(items, asking)
        for n, item in enumerate(items):
            if n in run_on:
                got, moved = [(i + 1 + n, 0, True)], True
            else:
                got, moved = _line_pieces(item, i + 1 + n, moved)
            pieces += got
            partial |= not all(m for *_, m in got)
        options = moved
        i = j
    first = next((n for n, piece in enumerate(pieces) if piece[2]), None)
    if first is None:
        return Split(re.sub(r"\n{3,}", "\n\n", text).strip(), "", None)
    skipped = PARTIAL_LIST if partial else NOT_TRAILING if not all(m for *_, m in pieces[first:]) else None
    line, column, _ = pieces[first]
    before = [*lines[:line], lines[line][:column].rstrip()]
    content = re.sub(r"\n{3,}", "\n\n", "\n".join(before)).strip()
    after = [lines[line][column:], *lines[line + 1:]]
    return Split(content, "\n".join(part.strip() for part in after if part.strip()), skipped)


# R11 (the first step) and the derived asks


def rule_r11(record: dict) -> list[str] | None:
    """R11's missing slots: a clarify-scenario record whose first step only searched the notes (NOTE_TOOLS)
    and whose first ask_clarification, at its second step, misses the topic; None otherwise."""
    if record.get("scenario") != "clarify":
        return None
    first = _first_ask(record)
    if first is None or first[0] != 1:
        return None
    first_step = record["messages"][_turn_positions(record["messages"])[0]]
    names = [_name(c) for c in first_step.get("tool_calls") or []]
    missing = first[1].get("missing")
    if not names or any(n not in NOTE_TOOLS for n in names) or not isinstance(missing, list) \
            or "topic" not in missing:
        return None
    return list(missing)


_UNKNOWN = {"topic": ("你说的是哪个方向", "which topic you mean"),
            "paper_id": ("你说的是哪一篇", "which paper you mean"),
            "date_range": ("你要的是哪段时间", "which time range you mean"),
            "count": ("你要几篇", "how many you want"), "tags": ("你说的是哪个标签", "which tag you mean")}


def listing_text(listing: Listing, missing: list[str], lang: str) -> str:
    """The declarative text of a derived ask after the listing seen: how many notes, on which topics, and that
    they do not settle the missing slots; all the user's notes only for a complete listing
    (Listing.complete), the notes listed otherwise."""
    notes = listing.notes
    tags = _topic_tags(notes)[:5]
    if lang == "zh":
        about = f",涉及 {'、'.join(tags)} 等方向" if tags else ""
        unknown = ",也看不出".join(_UNKNOWN[m][0] for m in missing)
        seen = f"我看了你的笔记,共 {len(notes)} 条" if listing.complete else f"我列出了 {len(notes)} 条笔记"
        return f"{seen}{about},从中看不出{unknown}。"
    about = f", on {', '.join(tags)}" if tags else ""
    unknown = ", or ".join(_UNKNOWN[m][1] for m in missing)
    seen = (f"I checked your notes: there are {len(notes)}" if listing.complete
            else f"I listed {len(notes)} of your notes")
    return f"{seen}{about}, and none of them tells me {unknown}."


def _listing_ask(record: dict, k: int, listing: Listing, missing: list[str], key: str) -> Relabel:
    lang = record.get("lang") or "zh"
    message, reply = _ask_message(key, k, clarify_question(missing, lang, key), missing,
                                  listing_text(listing, missing, lang), _today_of(record))
    return Relabel("R10", "listing_ask", message, reply)


# R13


def rule_r13(record: dict, request: dict, topics: tuple) -> tuple[dict, dict[int, Counter]]:
    """R13 on a copy of record: in a Chinese record, every search_notes query of Chinese characters without a
    Latin letter gets the English phrase of the request's train topic appended, in the call's own JSON style
    (check_call must accept it); a request without a train topic leaves the query. The record (itself when
    nothing changed) and the counts per turn: pure_chinese, rewritten, no_term."""
    counts: dict[int, Counter] = {}
    if record.get("lang") != "zh":
        return record, counts
    slots = request.get("slots") or {}
    term = slots.get("topic") if slots.get("topic_id") in {t.id for t in topics} else None
    out = copy.deepcopy(record)
    for k, i in enumerate(_turn_positions(out["messages"])):
        for c in out["messages"][i].get("tool_calls") or []:
            args = _args(c)
            query = (args or {}).get("query")
            if _name(c) != "search_notes" or not isinstance(query, str) or not _HAN.search(query) \
                    or _LATIN.search(query):
                continue
            counts.setdefault(k, Counter())["pure_chinese"] += 1
            if not isinstance(term, str):
                counts[k]["no_term"] += 1
                continue
            new = {**args, "query": f"{query} {term}"}  # the key order kept
            _checked("search_notes", new, record.get("id", "?"), _today_of(record))
            c["function"]["arguments"] = serialize_like(c["function"]["arguments"], new)
            counts[k]["rewritten"] += 1
    return (out if any(n["rewritten"] for n in counts.values()) else record), counts


# applying the rules to the records


def _relabel_at(record: dict, request: dict, k: int, r11: list[str] | None, r8: Counter,
                r8_codes: Counter, r10_skipped: dict) -> Relabel | None:
    """The relabel of turn k of record by the first rule that fires there (module docstring), or None; R8's
    verdicts and the error codes of its relabels are counted, and an R10 label whose split is not clean is
    left with its reason in r10_skipped ((id, k) -> split_question's skipped)."""
    messages, rid = record["messages"], record["id"]
    today, lang = _today_of(record), record.get("lang") or "zh"
    i = _turn_positions(messages)[k]
    history, target = messages[:i], messages[i]
    if k == 0 and r11 is not None:
        message, reply = _ask_message(rid, 0, clarify_question(r11, lang, rid), r11, "", today)
        return Relabel("R11", "first_ask", message, reply)
    verdict, failed = rule_r8(history, target)
    r8[verdict] += 1
    if verdict == "relabel":
        calls = [{"function": dict(c["function"]), "id": _new_call_id(rid, k, n), "type": "function"}
                 for n, c in enumerate(failed)]
        r8["mixed_step"] += len(failed) < len(_last_round(history)[0])
        codes = {_error(r) for r in _last_round(history)[1] if _error(r) in TRANSIENT}
        r8_codes["+".join(sorted(codes))] += 1
        return Relabel("R8", "retry", {"content": "", "role": "assistant", "tool_calls": calls})
    names = [_name(c) for c in target.get("tool_calls") or []]
    listing = s4a_point(record, history, target, k)
    if listing is not None:
        return _closing(record, request, k, listing, rid, today)
    if "search_arxiv" in names and s4b(history) is not None:
        return Relabel("R9", "s4b", {"content": search_report(history, lang), "role": "assistant"})
    if (rid, k) in R10_LABELLED:
        split = split_question(target.get("content") or "")
        if target.get("tool_calls") or not split.question:
            raise TransformError(f"{rid} turn {k}: R10_LABELLED names a plain-text question, but this turn "
                                 "has calls or asks nothing")
        if split.skipped is not None:  # the questions are not the text's trailing block: left as it is
            r10_skipped[rid, k] = split.skipped
            return None
        message, reply = _ask_message(rid, k, split.question, R10_LABELLED[rid, k], split.content, today)
        return Relabel("R10", "text_ask", message, reply)
    return None


def _cut(record: dict, k: int, relabel: Relabel | None, **fields) -> dict:
    """A copy of record up to turn k, which relabel gives, with fields set (a derived record's id ...); with
    relabel None, up to turn k as the record has it, with that turn's tool replies."""
    messages = record["messages"]
    i = _turn_positions(messages)[k]
    out = {key: copy.deepcopy(value) for key, value in record.items() if key != "messages"}
    if relabel is None:
        end = i + 1
        while end < len(messages) and messages[end].get("role") == "tool":
            end += 1
        out["messages"] = copy.deepcopy(messages[:end])
    else:
        out["messages"] = copy.deepcopy(messages[:i]) + [relabel.message] + \
            ([relabel.reply] if relabel.reply is not None else [])
    out.update(fields)
    return out


def _change(rid: str, k: int, rule: str, kind: str, source: list | None, target: bool | None = None) -> dict:
    """A changes row; target defaults to kind != "copy" (a copy is context unless it is a kept target)."""
    return {"id": rid, "k": k, "rule": rule, "kind": kind,
            "target": kind != "copy" if target is None else target, "from": source}


def _count_targets(record: dict, positions: list[int], seen: Counter) -> None:
    """Every turn of record (before any cut) R8's or R9's situation holds for, counted into seen: R8's
    verdicts, S4a's note queries, S4b's arXiv searches."""
    for k, i in enumerate(positions):
        history, target = record["messages"][:i], record["messages"][i]
        seen["r8:" + rule_r8(history, target)[0]] += 1
        names = [_name(c) for c in target.get("tool_calls") or []]
        if s4a_point(record, history, target, k) is not None:
            seen["s4a"] += 1
        if "search_arxiv" in names and s4b(history) is not None:
            seen["s4b"] += 1


def _targets_report(seen: Counter) -> dict:
    return {"R8": {"transient_steps": sum(seen[f"r8:{v}"] for v in ("relabel", "identical", "at_run_limit")),
                   "not_identical": seen["r8:relabel"] + seen["r8:at_run_limit"],
                   "identical": seen["r8:identical"], "at_run_limit": seen["r8:at_run_limit"]},
            "R9": {"s4a_note_queries": seen["s4a"], "s4b_arxiv_searches": seen["s4b"]},
            "R10": {"labelled": seen["r10"]}, "R11": {"records": seen["r11"]},
            "R13": {"pure_chinese": seen["pure_chinese"], "with_a_term": seen["with_a_term"]}}


@dataclass
class _Candidate:
    rule: str  # R9 or R10
    record: dict  # the original (after R13) the copy is cut from
    k: int
    listing: Listing
    missing: list[str] | None  # R10's
    source: str  # where it comes from, for the counts
    keep: bool = False  # the original turn k is kept as the target (the teacher already asked there)


def _hashed(candidates: list[_Candidate]) -> list[_Candidate]:
    """The candidates in the order of the SHA-256 of rule:id:k (independent of the input order)."""
    def key(c: _Candidate) -> str:
        return hashlib.sha256(f"{c.rule}:{c.record['id']}:{c.k}".encode()).hexdigest()

    return sorted(candidates, key=key)


def _derived_id(c: _Candidate) -> str:
    return f"{c.record['id']}-ft4{c.rule.lower()}k{c.k}"


def _target_key(message: dict) -> str:
    """A target as two samples' answers compare: its text and its calls (names and parsed arguments), the call
    ids aside."""
    return canonical_json([message.get("content") or "",
                           [[_name(c), _args(c)] for c in message.get("tool_calls") or []]])


def _asks(message: dict) -> bool:
    return any(_name(c) == ASK for c in message.get("tool_calls") or [])


def apply_ft4(records: list[dict], by_id: dict[str, dict], *, topics: tuple, dropped: list[dict] = (),
              targets: dict | None = None, guard: EvalGuard | None = None) -> Ft4Records:
    """FT4's in-place rules (R13, then R11, R8, R9, R10 at the first turn one fires, the record cut after it)
    over records, the kept input records, and the derived records cut from them and from dropped (the clarify
    records R4 dropped, after R1 and R2) (module docstring). guard: a derived record whose first user message
    reaches the evaluation limit is left out; None: no check. TransformError for an R10 label that does not
    fit its turn, or a changed call check_call refuses."""
    targets = {**FT4_TARGETS, **(targets or {})}
    out, changes, candidates = [], [], []
    r8: Counter[str] = Counter()
    r8_codes: Counter[str] = Counter()
    fired: Counter[str] = Counter()
    r13: Counter[str] = Counter()
    labelled = {key: "absent" for key in R10_LABELLED}
    labels_by_id: dict[str, list[int]] = {}
    for rid, k in R10_LABELLED:
        labels_by_id.setdefault(rid, []).append(k)
    removed = after_r9 = 0
    seen: Counter[str] = Counter()  # the targets in the input, cut or not, as the coverage analysis counted
    r10_skipped: dict[tuple[str, int], str] = {}  # the R10 labels whose split is not clean, with the reason
    for record in records:
        request = by_id[record["id"]]
        rewritten, r13_turns = rule_r13(record, request, topics)
        messages = rewritten["messages"]
        positions = _turn_positions(messages)
        r11, cut = rule_r11(rewritten), None
        _count_targets(rewritten, positions, seen)
        seen["r11"] += r11 is not None
        seen["r10"] += sum(k < len(positions) for k in labels_by_id.get(record["id"], []))
        for counts in r13_turns.values():
            seen["pure_chinese"] += counts["pure_chinese"]
            seen["with_a_term"] += counts["rewritten"]
        for k in range(len(positions)):
            relabel = _relabel_at(rewritten, request, k, r11, r8, r8_codes, r10_skipped)
            if relabel is not None:
                cut = (k, relabel)
                break
        cut_k = cut[0] if cut else len(positions)
        for k, counts in sorted(r13_turns.items()):
            if k < cut_k:
                r13.update(counts)
                if counts["rewritten"]:
                    changes.append(_change(record["id"], k, "R13", "relabel", [record["id"], k]))
        for k in labels_by_id.get(record["id"], []):  # a label past the cut was removed with its turn
            if (record["id"], k) in r10_skipped:
                labelled[record["id"], k] = "skipped_" + r10_skipped[record["id"], k]
            else:
                labelled[record["id"], k] = "removed" if cut_k <= k < len(positions) else "present"
        if cut is None:
            out.append(rewritten)
            continue
        k, relabel = cut
        fired[f"{relabel.rule}:{relabel.variant}"] += 1
        if relabel.rule == "R10":
            labelled[record["id"], k] = "relabeled"
        changes.append(_change(record["id"], k, relabel.rule, "relabel", [record["id"], k]))
        out.append(_cut(rewritten, k, relabel))
        removed += len(positions) - k - 1
        # the S4a points the cut removed: a new context only where R8, R10 or R11 cut (module docstring)
        later = [(kk, listing) for kk in range(k + 1, len(positions))
                 if (listing := s4a_point(rewritten, messages[:positions[kk]], messages[positions[kk]], kk))]
        if relabel.rule == "R9":
            after_r9 += len(later)
        elif later:
            candidates.append(_Candidate("R9", rewritten, later[0][0], later[0][1], None, "cut"))
        if relabel.rule == "R11":
            history = messages[:positions[1]]
            listing = listing_of(history)
            if listing and not _transient_last(history):
                candidates.append(_Candidate("R10", rewritten, 1, listing, r11, "from_r11",
                                             keep=_asks(messages[positions[1]])))
    r4_points: Counter[str] = Counter()
    for record in dropped:
        rewritten, _ = rule_r13(record, by_id[record["id"]], topics)
        first = _first_ask(rewritten)
        if first is None or not isinstance(first[1].get("missing"), list) or not first[1]["missing"]:
            continue
        positions = _turn_positions(rewritten["messages"])
        points = [(k, listing) for k in range(1, first[0])  # the listing points before its first ask
                  if (listing := listing_of(rewritten["messages"][:positions[k]]))
                  and not _transient_last(rewritten["messages"][:positions[k]])]
        complete = next(((k, listing) for k, listing in points if listing.complete), None)
        r4_points["with_a_listing_point"] += bool(points)
        if complete is not None:  # one point per record: its first complete listing
            r4_points["with_a_complete_listing"] += 1
            candidates.append(_Candidate("R10", rewritten, complete[0], complete[1],
                                         list(first[1]["missing"]), "from_r4_dropped"))
    misfit = [key for key, state in labelled.items() if state == "present"]
    if misfit:
        raise TransformError(f"R10_LABELLED turns {misfit} are in the records but were not relabelled")
    chosen: list[tuple[_Candidate, Relabel | None]] = []
    for c in _hashed([c for c in candidates if c.rule == "R9"])[:max(targets["r9_new"], 0)]:
        source = c.record
        chosen.append((c, _closing(source, by_id[source["id"]], c.k, c.listing, _derived_id(c),
                                   _today_of(source))))
    built, duplicates = set(), []  # no two new asks with one target: the first in the hashed order is built
    for c in _hashed([c for c in candidates if c.rule == "R10" and not c.keep]):
        if len(built) >= targets["r10_new"]:
            break
        relabel = _listing_ask(c.record, c.k, c.listing, c.missing, _derived_id(c))
        if _target_key(relabel.message) in built:
            duplicates.append(_derived_id(c))
            continue
        built.add(_target_key(relabel.message))
        chosen.append((c, relabel))
    chosen += [(c, None) for c in candidates if c.keep]  # the teacher's kept turns are no new samples
    derived, overlap = [], []
    for c, relabel in chosen:
        source, rule, rid = c.record, c.rule, _derived_id(c)
        new = _cut(source, c.k, relabel, id=rid, run=FT4_RUN, family=f"ft4/{rule.lower()}/{source['family']}",
                   context_turns=c.k)
        if guard is not None and guard.overlaps_eval(_first_user(new)):
            overlap.append(rid)
            continue
        derived.append((new, c))
        changes.extend(_change(rid, j, rule, "copy", [source["id"], j]) for j in range(c.k))
        changes.append(_change(rid, c.k, rule, "copy" if c.keep else "new", [source["id"], c.k], target=True))
    r10 = [c for _, c in derived if c.rule == "R10"]
    changed_ids = {c["id"] for c in changes}
    pick = {v: fired[f"R9:s4a_pick_{v}"] for v in ("ask", "one", "none")}
    report = {
        "rules": {
            "R8": {"situations": sum(r8[v] for v in ("relabel", "identical", "at_run_limit")),
                   "relabeled": fired["R8:retry"], "identical": r8["identical"],
                   "at_run_limit": r8["at_run_limit"], "mixed_step": r8["mixed_step"],
                   "by_code": dict(sorted(r8_codes.items()))},
            "R9": {"s4a": {"relabeled": fired["R9:s4a_list"] + sum(pick.values()),
                           "list": fired["R9:s4a_list"], "pick": pick},
                   "s4b": {"relabeled": fired["R9:s4b"]},
                   "constructed": {"records": sum(c.rule == "R9" for _, c in derived),
                                   "candidates": sum(c.rule == "R9" for c in candidates),
                                   "target": targets["r9_new"], "not_built_after_r9_cuts": after_r9}},
            "R10": {"relabeled": fired["R10:text_ask"], "labelled": len(R10_LABELLED),
                    "labelled_in_input": sum(state != "absent" for state in labelled.values()),
                    "labelled_removed_by_an_earlier_cut": sorted(
                        f"{rid}:{k}" for (rid, k), state in labelled.items() if state == "removed"),
                    **{f"skipped_{reason}": sorted(f"{rid}:{k}" for (rid, k), state in labelled.items()
                                                   if state == f"skipped_{reason}")
                       for reason in (NOT_TRAILING, PARTIAL_LIST)},
                    "borderline_left": list(R10_BORDERLINE),
                    "derived": {"records": len(r10), "new": sum(not c.keep for c in r10),
                                "copy_target": sum(c.keep for c in r10),
                                "candidates_new": sum(c.rule == "R10" and not c.keep for c in candidates),
                                "duplicate_targets": sorted(duplicates),
                                "target": targets["r10_new"],
                                "from_r11": {
                                    "new": sum(c.source == "from_r11" and not c.keep for c in r10),
                                    "copy_target": sum(c.source == "from_r11" and c.keep for c in r10)},
                                "from_r4_dropped": {
                                    "new": sum(c.source == "from_r4_dropped" for c in r10),
                                    "records": len({c.record["id"] for c in r10
                                                    if c.source == "from_r4_dropped"}),
                                    "with_a_listing_point": r4_points["with_a_listing_point"],
                                    "with_a_complete_listing": r4_points["with_a_complete_listing"]}}},
            "R11": {"relabeled": fired["R11:first_ask"]},
            "R13": {key: r13[key] for key in ("pure_chinese", "rewritten", "no_term")},
        },
        "targets_in_input": _targets_report(seen),
        "records": {"in": len(records), "relabeled": sum(r["id"] in changed_ids for r in out),
                    "cut": sum(fired.values()), "dropped_by_r4_seen": len(dropped),
                    "dropped_by_r4_with_a_listing_point": r4_points["with_a_listing_point"],
                    "derived": len(derived)},
        "turns": {"relabel": sum(c["kind"] == "relabel" for c in changes), "removed": removed},
        "eval_overlap": {"derived_dropped": sorted(overlap)},
    }
    changes.sort(key=lambda c: (c["id"], c["k"]))
    return Ft4Records(out, sorted((d for d, _ in derived), key=lambda r: r["id"]), changes, report)


# new records (the full run only)


def relative_range(key: str, today: str) -> tuple[str, str]:
    """The first and last day (ISO) a relative date of RELATIVE_PHRASES means on today: yesterday, the day
    before, this week (Monday to today), last week (Monday to Sunday) or last month."""
    day = date.fromisoformat(today)
    if key == "yesterday":
        start = end = day - timedelta(days=1)
    elif key == "day_before":
        start = end = day - timedelta(days=2)
    elif key == "this_week":
        start, end = day - timedelta(days=day.weekday()), day
    elif key == "last_week":
        start = day - timedelta(days=day.weekday() + 7)
        end = start + timedelta(days=6)
    elif key == "last_month":
        end = day.replace(day=1) - timedelta(days=1)
        start = end.replace(day=1)
    else:
        raise KeyError(key)
    return start.isoformat(), end.isoformat()


def generate_relative_requests(n: int, *, seed: int, corpus: rq.Corpus, split_seed: int, today: str,
                               guard: EvalGuard) -> list[dict]:
    """R11's relative-date requests: half per language (the odd one zh), round robin over (template, relative
    date) in a seeded order, filled with train topics and tags (requests.fill), each offered to the guard.
    The slots record the relative date's key and its range as the call takes it: list_notes' since, or
    search_arxiv's date_from and, for a range that ends before today, date_to."""
    topics = rq.split_topics(corpus, split_seed)["train"]
    out = []
    for i, lang in enumerate(("zh", "en")):
        wanted = n // 2 + (n % 2 if i == 0 else 0)
        cells = [(index, key) for index in range(len(RELATIVE_TEMPLATES[lang]))
                 for key in RELATIVE_PHRASES[lang]]
        rng = random.Random(f"{seed}:ft4:r11:{lang}")
        rng.shuffle(cells)
        got = attempt = 0
        while got < wanted and attempt < ATTEMPTS_PER_REQUEST * max(wanted, 1):
            index, key = cells[attempt % len(cells)]
            attempt += 1
            intent = RELATIVE_INTENTS[index]
            template = RELATIVE_TEMPLATES[lang][index].replace("{rel}", RELATIVE_PHRASES[lang][key])
            text, slots = rq.fill(template, lang, topics, corpus.window, rng, intent=intent, kind="normal")
            start, end = relative_range(key, today)
            slots["relative"] = key
            if intent == "list_notes":
                slots["since"] = start
            else:
                slots["date_from"] = start
                if end < today:
                    slots["date_to"] = end
            if guard.offer(text) is None:
                out.append({"family": f"ft4/r11/{lang}/{index}", "intent": intent, "lang": lang,
                            "scenario": "normal", "slots": slots, "split": "train", "text": text})
                got += 1
    return out


def _ft4_record(request: dict, messages: list[dict], *, family: str, scenario: str) -> dict:
    record = new_record(request, messages, record_id=request["id"], family=family, scenario=scenario)
    record["run"] = FT4_RUN
    return record


def relative_sort_by(inputs: list[tuple[str, list[dict]]]) -> tuple[str | None, dict]:
    """R11's sort_by for its relative-date arXiv searches, the owner's rule (RELATIVE_SORT_RULE): over the
    inputs ((name, records as read: the teacher's)), the search_arxiv calls with a date_from in the
    RELATIVE_SORT_INTENTS records, each counted with its effective value, the schema default for a call
    without sort_by (counted as ABSENT per input); the value most of them take, a tie going to the default
    (then to the smaller canonical JSON). None when that majority is the default: R11's searches leave
    sort_by out, as R14's leave out their defaults. The account the manifest records: the rule, the choice
    and the majority, the counts per input, the effective counts, the intents and the default."""
    default = schemas.DEFAULTS["search_arxiv"]["sort_by"]
    counts: dict[str, dict[str, int]] = {}
    effective: Counter[str] = Counter()
    for name, records in inputs:
        found: Counter[str] = Counter()
        for record in records:
            if record.get("intent") not in RELATIVE_SORT_INTENTS:
                continue
            for c in _calls(record):
                args = _args(c)
                if _name(c) != "search_arxiv" or args is None or "date_from" not in args:
                    continue
                value = args.get("sort_by", ABSENT)
                found[value if isinstance(value, str) else canonical_json(value)] += 1
        counts[name] = dict(sorted(found.items()))
        for value, n in found.items():
            effective[default if value == ABSENT else value] += n
    majority = min(effective, key=lambda v: (-effective[v], v != default, canonical_json(v)), default=default)
    chosen = None if majority == default else majority
    return chosen, {"rule": RELATIVE_SORT_RULE, "chosen": chosen, "majority": majority, "counts": counts,
                    "effective": dict(sorted(effective.items())), "intents": list(RELATIVE_SORT_INTENTS),
                    "default": default}


def relative_date_record(request: dict, today: str, *, sort_by: str | None = None) -> dict:
    """R11: one step, list_notes(tag, since) or search_arxiv(query, date_from, date_to) as the slots say, the
    search with sort_by when one is given (relative_sort_by's choice)."""
    slots = request["slots"]
    if request["intent"] == "list_notes":
        args = {"tag": slots["tags"][0]} if slots.get("tags") else {}
        step = ("list_notes", {**args, "since": slots["since"]})
    else:
        step = ("search_arxiv", {"query": slots["topic"], **{k: slots[k] for k in DATE_KEYS if k in slots},
                                 **({"sort_by": sort_by} if sort_by else {})})
    messages = scripted_messages(request["text"], [step], today, request["id"])
    return _ft4_record(request, messages, family=request["family"], scenario="normal")


def defref_families(split_seed: int) -> list[str]:
    """The train-split normal templates (split_templates of split_seed) that hold a definite reference
    (DEFREF) and name a topic, sorted: R14's only source of wording."""
    assignment = rq.split_templates(split_seed)
    return sorted(rq.family_name(intent, lang, index)
                  for intent, kind, lang, index, template in rq.template_families()
                  if kind == "normal" and assignment[rq.family_name(intent, lang, index)] == "train"
                  and ("{topic}" in template or "{topic_paren}" in template) and DEFREF.search(template))


def generate_defref_requests(n: int, *, seed: int, corpus: rq.Corpus, split_seed: int,
                             guard: EvalGuard) -> list[dict]:
    """R14's definite-reference requests: defref_families' templates filled with train topics as R3 fills
    its own (_fill_requests), each offered to the guard."""
    families = set(defref_families(split_seed))
    cells: dict[str, list] = {"zh": [], "en": []}
    for intent, _, lang, index, template in rq.template_families():
        if rq.family_name(intent, lang, index) in families:
            cells[lang].append((intent, rq.family_name(intent, lang, index), template))
    topics = rq.split_topics(corpus, split_seed)["train"]
    return _fill_requests(cells, n, kind="normal", scenario="normal", seed=seed, purpose="ft4-r14",
                          corpus=corpus, topics=topics, guard=guard)


def defref_record(request: dict, today: str) -> dict:
    """R14: one step, search_arxiv(query = the topic) for a request that names the topic of "the" paper. The
    arguments at their default are left out (sort_by, max_results), as M6's scripted records leave them out:
    the scorer fills in the defaults."""
    messages = scripted_messages(request["text"], [("search_arxiv", {"query": request["slots"]["topic"]})],
                                 today, request["id"])
    return _ft4_record(request, messages, family=f"ft4/r14/{request['family']}", scenario="normal")


def optional_params() -> list[tuple[str, str]]:
    """(tool, parameter) for every parameter MODEL_TOOLS does not list as required."""
    return [(t["function"]["name"], p) for t in schemas.MODEL_TOOLS
            for p in t["function"]["parameters"]["properties"]
            if p not in t["function"]["parameters"].get("required", [])]


def explain_answer(tool: str, param: str, lang: str) -> str:
    """R14's answer to whether an optional parameter can be left out, from what MODEL_TOOLS shows the model
    (its type, range, values or format and the tool's required parameters; no default value, which the
    model's view does not show). Declarative."""
    function = next(t["function"] for t in schemas.MODEL_TOOLS if t["function"]["name"] == tool)
    spec = function["parameters"]["properties"][param]
    required = list(function["parameters"].get("required") or [])
    zh = lang == "zh"
    items_enum = (spec.get("items") or {}).get("enum")
    if spec.get("type") == "integer":
        what = (f"它是整数,取值 {spec['minimum']} 到 {spec['maximum']}。" if zh
                else f"It is an integer from {spec['minimum']} to {spec['maximum']}.")
    elif spec.get("type") == "array":
        what = (f"它是字符串列表,每一项可选 {'、'.join(items_enum)}。" if zh and items_enum
                else f"It is a list of strings, each one of {', '.join(items_enum)}." if items_enum
                else "它是字符串列表。" if zh else "It is a list of strings.")
    elif spec.get("enum"):
        what = (f"它是字符串,可选 {'、'.join(spec['enum'])}。" if zh
                else f"It is a string, one of {', '.join(spec['enum'])}.")
    elif spec.get("description"):
        what = (f"它是字符串,格式为 {spec['description']}。" if zh
                else f"It is a string in the form {spec['description']}.")
    else:
        what = "它是字符串。" if zh else "It is a string."
    if zh:
        must = f"{tool} 只有 {'、'.join(required)} 是必填的。" if required else f"{tool} 没有必填参数。"
        return (f"{param} 是 {tool} 的可选参数,不在必填列表里,可以不写。{what}"
                f"不写时,工具按它自己的默认方式处理。{must}")
    must = f"For {tool}, only {' and '.join(required)} must be given." if required else \
        f"{tool} has no required parameter."
    return (f"{param} is an optional parameter of {tool}: it is not in the required list, so you can leave "
            f"it out. {what} When it is left out, the tool falls back to its own default handling. {must}")


def generate_explain_requests(n: int, *, seed: int, guard: EvalGuard) -> list[dict]:
    """R14's explain_tools questions: half per language (the odd one zh), (template, optional parameter) in a
    seeded order, each offered to the guard; family ft4/explain/<lang>/<template>."""
    pairs = optional_params()
    out = []
    for i, lang in enumerate(("zh", "en")):
        wanted = n // 2 + (n % 2 if i == 0 else 0)
        cells = [(index, tool, param) for index in range(len(EXPLAIN_TEMPLATES[lang]))
                 for tool, param in pairs]
        random.Random(f"{seed}:ft4:explain:{lang}").shuffle(cells)
        got = 0
        for index, tool, param in cells:
            if got >= wanted:
                break
            text = rq.tidy(EXPLAIN_TEMPLATES[lang][index].format(tool=tool, param=param), lang)
            if guard.offer(text) is None:
                out.append({"family": f"ft4/explain/{lang}/{index}", "intent": "explain_tools", "lang": lang,
                            "scenario": "no_call", "slots": {"tool": tool, "param": param}, "split": "train",
                            "text": text})
                got += 1
    return out


def scripted_answer(text: str, answer: str, today: str, record_id: str) -> list[dict]:
    """The history the agent loop records when the model answers text at once, without a call."""
    executor = ToolExecutor(_tools(today), check_call, today=today)
    transcript = run_turn(text, model=_Script([AssistantTurn(answer, [], RUN)]), executor=executor,
                          decide=policy.decide, today=today, max_steps=1)
    if transcript.status != ANSWERED or transcript.messages[2:] != [{"role": "assistant", "content": answer}]:
        raise TransformError(f"{record_id}: the agent loop did not accept the scripted answer")
    return copy.deepcopy(transcript.messages)


def explain_record(request: dict, today: str) -> dict:
    """R14: the answer explain_answer gives, without a call."""
    slots = request["slots"]
    answer = explain_answer(slots["tool"], slots["param"], request["lang"])
    messages = scripted_answer(request["text"], answer, today, request["id"])
    return _ft4_record(request, messages, family=request["family"], scenario="no_call")


def term_order(terms: Iterable[str]) -> list[str]:
    """The Chinese terms in the order of the SHA-256 of each (UTF-8): R13's pick, whatever the file's
    order."""
    return sorted(terms, key=lambda zh: hashlib.sha256(zh.encode("utf-8")).hexdigest())


def glossary_query(en: str) -> str:
    """R13's query for an English glossary term: every word lowercased unless it has two or more capitals or
    a digit (BERT, GPT-2, ResNet stay). A word is a run of ASCII letters and digits, so a hyphenated Title
    Case word is two (Pre-Training -> pre-training)."""
    def word(match: re.Match) -> str:
        w = match.group(0)
        return w if sum(ch.isupper() for ch in w) >= 2 or any(ch.isdigit() for ch in w) else w.lower()

    return _WORD.sub(word, en)


def glossary_skip(zh: str, en: str, science_only: frozenset = frozenset()) -> str | None:
    """Why a noted glossary term cannot be one of R13's term records (GLOSSARY_SKIPS' first three), or None:
    its English term is not ASCII only (non_ascii); a Latin token of the Chinese term (a run of ASCII letters
    and digits, with the dots inside it: Adam算法, C4.5算法) is no whole word of the English term, letter case
    aside (latin_token_missing: Adam算法 = Adaptive Moment Estimation Algorithm); the term is one only AITD's
    AI for Science section gives (science_only: science_only_terms)."""
    if not en.isascii():
        return "non_ascii"
    folded = en.casefold()
    for token in _LATIN_TOKEN.findall(zh):
        if not re.search(rf"(?<![a-z0-9]){re.escape(token.casefold())}(?![a-z0-9])", folded):
            return "latin_token_missing"
    return "science_only" if zh in science_only else None


def glossary_requests(glossary: Mapping[str, str | None], *, existing: dict, corpus: rq.Corpus,
                      held_out: tuple[str, ...], limit: int, guard: EvalGuard,
                      science_only: frozenset = frozenset()) -> tuple[list[dict], dict]:
    """The R13 hook for the glossary another package builds (docs/m7-本地提升计划.md WP4): R7-style
    requests (term_requests, families ft4/terms/zh/<template>), at most limit, from the glossary's noted terms
    (a research_agent.glossary.Glossary, or a mapping it reads; the blocked terms are counted apart) in the
    order of term_order, each eligible (glossary_skip: an ASCII English term holding the Chinese term's Latin
    tokens, not a term of AI for Science only), whose Chinese or English side R7's list (existing) does not
    hold and whose query (glossary_query) check_terms accepts (no corpus phrase or name, no held-out topic or
    synonym). The account: the noted and the blocked terms, the skipped by reason (GLOSSARY_SKIPS), the
    eligible, kept, requests and dropped_eval_overlap, and the order and query rules."""
    glossary = glossary if isinstance(glossary, Glossary) else Glossary(glossary)
    known_zh = {normalize_text(zh) for zh in existing}
    known_en = {str(en).casefold() for en in existing.values()}
    fresh: dict[str, str] = {}
    skipped: Counter[str] = Counter()
    for zh in term_order(glossary):
        en = glossary[zh]
        reason = glossary_skip(zh, en, science_only)
        if reason is None and (normalize_text(zh) in known_zh or en.casefold() in known_en):
            reason = "already_in_r7"
        if reason is None and check_terms({zh: glossary_query(en)}, corpus, held_out=held_out, minimum=1):
            reason = "refused"
        if reason is None:
            fresh[zh] = glossary_query(en)
        else:
            skipped[reason] += 1
    made, overlaps = term_requests(fresh, limit, guard, family_prefix="ft4/terms")
    eligible = len(glossary) - sum(skipped[reason] for reason in GLOSSARY_SKIPS[:3])
    return made, {"terms": len(glossary), "blocked": len(glossary.blocked),
                  "skipped": {reason: skipped[reason] for reason in GLOSSARY_SKIPS}, "eligible": eligible,
                  "kept": len(fresh), "requests": len(made), "dropped_eval_overlap": overlaps,
                  "order": "the SHA-256 of the Chinese term (UTF-8)",
                  "query": "the English term, every word lowercased unless it has two or more capitals or a "
                           "digit"}


def _build_glossary():
    """scripts/build_glossary.py as a module of its own name (scripts/ is no package): its AITD parser."""
    spec = importlib.util.spec_from_file_location("_build_glossary_r13",
                                                  REPO / "scripts" / "build_glossary.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def science_only_terms(provenance: Path) -> tuple[frozenset[str], dict]:
    """R13's domain filter: the Chinese terms every AITD row of which (data/All.md) is a row of
    SCIENCE_SECTION and none a row of another section file, read with scripts/build_glossary.py's own parser
    (parse_aitd, as text) from the AITD files the glossary's provenance records (sources "aitd": the clone's
    location, relative to the repository root, and each file's SHA-256, checked). TransformError for a
    provenance without them, or a file missing or not the one it records. The account: the provenance's path,
    the files read (SHA-256 and rows), the section and the count."""
    try:
        source = next(s for s in json.loads(provenance.read_text(encoding="utf-8"))["sources"]
                      if s.get("name") == "aitd")
        root, recorded = REPO / source["location"], {f["path"]: f["sha256"] for f in source["files"]}
    except (StopIteration, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
        raise TransformError(f"{provenance}: no AITD source with its location and files: {exc!r}") from exc
    names = ["data/All.md", *sorted(p for p in recorded if p.startswith("section/") and p.endswith(".md"))]
    if SCIENCE_SECTION not in names:
        raise TransformError(f"{provenance}: the AITD files hold no {SCIENCE_SECTION}")
    parse_aitd, rows = _build_glossary().parse_aitd, {}
    for name in names:
        path = root / name
        if name not in recorded or not path.is_file() or sha256_file(path) != recorded[name]:
            raise TransformError(f"{path}: not the AITD file {provenance} records (absent, or another "
                                 "SHA-256)")
        rows[name] = parse_aitd(path.read_text(encoding="utf-8"), name)
    sections = {name: {row.index for row in rows[name]} for name in names[1:]}
    elsewhere = set().union(*(found for name, found in sections.items() if name != SCIENCE_SECTION))
    by_term: dict[str, set[str]] = {}
    for row in rows["data/All.md"]:
        for zh in row.zh:
            by_term.setdefault(zh, set()).add(row.index)
    only = frozenset(zh for zh, found in by_term.items()
                     if found <= sections[SCIENCE_SECTION] and not found & elsewhere)
    return only, {"provenance": provenance.as_posix(), "section": SCIENCE_SECTION,
                  "files": {name: {"sha256": recorded[name], "rows": len(rows[name])} for name in names},
                  "science_only_terms": len(only)}


def glossary_record(request: dict, today: str) -> dict:
    """R13's glossary hook: one step, search_arxiv(query = glossary_query of the English term), as R7."""
    messages = scripted_messages(request["text"], [("search_arxiv", {"query": request["slots"]["topic"]})],
                                 today, request["id"])
    return _ft4_record(request, messages, family=request["family"], scenario="normal")


def _dedup_delta(guard: EvalGuard, before: tuple[dict, int]) -> dict:
    dropped, offered = before
    return {"offered": guard.deduper.offered - offered,
            "dedup_dropped": {k: guard.deduper.dropped[k] - dropped[k] for k in rq.DROP_REASONS}}


def ft4_new(*, today: str, corpus: rq.Corpus, split_seed: int, seed: int, guard: EvalGuard, targets: dict,
            glossary: Glossary | None = None, r7_terms: dict | None = None, held_out: tuple[str, ...] = (),
            sort_by: dict | None = None, science_only: frozenset = frozenset(),
            domain: dict | None = None) -> tuple[list[dict], list[dict], list[dict], dict]:
    """The full run's new FT4 requests (ids req-ft4-NNNN: R11's relative dates, R14's definite references
    and explain_tools questions, then the glossary hook's terms), their records, changes rows and counts. The
    new templates must pass the template audit together with R5's. sort_by: relative_sort_by's account of the
    inputs, whose choice R11's relative-date searches take (None: no sort_by). science_only and domain: the
    glossary hook's domain filter (science_only_terms) and its account."""
    targets = {**FT4_TARGETS, **targets}
    templates = {lang: (*LIST_NOTES_TEMPLATES[lang], *RELATIVE_TEMPLATES[lang], *EXPLAIN_TEMPLATES[lang])
                 for lang in ("zh", "en")}
    audit = audit_new_templates(templates)
    if not audit["ok"]:
        raise TransformError(f"FT4's template audit failed: {audit['over']}")
    stats = {}
    batches = []
    makers = {
        "relative_dates": lambda: generate_relative_requests(targets["r11_new"], seed=seed, corpus=corpus,
                                                             split_seed=split_seed, today=today, guard=guard),
        "defref": lambda: generate_defref_requests(targets["r14_defref"], seed=seed, corpus=corpus,
                                                   split_seed=split_seed, guard=guard),
        "explain": lambda: generate_explain_requests(targets["r14_explain"], seed=seed, guard=guard)}
    for key, make in makers.items():
        before = (dict(guard.deduper.dropped), guard.deduper.offered)
        made = make()
        stats[key] = {"requests": len(made), **_dedup_delta(guard, before)}
        batches.append(made)
    glossary_report = None
    if glossary is not None:
        made, glossary_report = glossary_requests(glossary, existing=r7_terms or {}, corpus=corpus,
                                                  held_out=held_out, limit=targets["r13_terms"], guard=guard,
                                                  science_only=science_only)
        glossary_report["domain"] = domain
        batches.append(made)
    else:
        batches.append([])
    requests = [{"id": f"req-ft4-{i:04d}", **r} for i, r in
                enumerate([r for batch in batches for r in batch], 1)]
    chosen = (sort_by or {}).get("chosen")
    builders = {"ft4/r11/": ("R11", lambda req, day: relative_date_record(req, day, sort_by=chosen)),
                "ft4/explain/": ("R14", explain_record), "ft4/terms/": ("R13", glossary_record)}
    records, changes = [], []
    for req in requests:
        rule, build = next((v for prefix, v in builders.items() if req["family"].startswith(prefix)),
                           ("R14", defref_record))
        records.append(build(req, today))
        changes.append(_change(req["id"], 0, rule, "new", None))
    report = {"relative_dates": {**stats["relative_dates"], "records": len(batches[0]),
                                 "target": targets["r11_new"], "templates": RELATIVE_TEMPLATES,
                                 "sort_by": sort_by},
              "defref": {**stats["defref"], "records": len(batches[1]), "target": targets["r14_defref"],
                         "families": defref_families(split_seed)},
              "explain": {**stats["explain"], "records": len(batches[2]), "target": targets["r14_explain"],
                          "templates": EXPLAIN_TEMPLATES},
              "glossary": glossary_report, "template_audit": audit}
    return requests, records, changes, report


def _trained_turns(records: list[dict]):
    """(record id, k, history, message) of every turn the trainer takes from records: all but a derived
    record's context turns (the turns below its context_turns, which train.samples leaves out)."""
    for record in records:
        for k, i in enumerate(_turn_positions(record["messages"])):
            if k >= record.get(CONTEXT_TURNS, 0):
                yield record["id"], k, record["messages"][:i], record["messages"][i]


def _by_type(records: list[dict]) -> dict:
    counts = Counter(item_type(history, message, k + 1) for _, k, history, message in _trained_turns(records))
    return dict(sorted(counts.items()))


def _prompt_conflicts(records: list[dict], changes: list[dict]) -> dict:
    """The turns whose prompt (build_context of the history) another turn shares, with the context turns left
    out as the trainer leaves them out, and under "all_turns" over every turn (were the context trained):
    the turns, the prompts shared, those shared with another answer (answer_sha256), and those of them with
    an FT4 target turn or with a copied turn (changes kind copy: in the first count only the teacher's kept
    targets are trained)."""
    copies = {(c["id"], c["k"]) for c in changes if c["kind"] == "copy"}
    ft4_turns = {(c["id"], c["k"]) for c in changes if c["target"]}
    trained: dict[str, list[tuple[tuple[str, int], str]]] = {}
    every: dict[str, list[tuple[tuple[str, int], str]]] = {}
    for record in records:
        for k, (context, message) in enumerate(step_contexts(record["messages"])):
            key = hashlib.sha256(canonical_json(context).encode("utf-8")).hexdigest()
            turn = ((record["id"], k), answer_sha256(message))
            every.setdefault(key, []).append(turn)
            if k >= record.get(CONTEXT_TURNS, 0):
                trained.setdefault(key, []).append(turn)

    def summary(by_prompt: dict) -> dict:
        shared = [turns for turns in by_prompt.values() if len(turns) > 1]
        conflicts = [turns for turns in shared if len({a for _, a in turns}) > 1]
        return {"turns": sum(map(len, by_prompt.values())), "shared_prompts": len(shared),
                "conflicts": len(conflicts),
                "conflicts_with_ft4": sum(any(key in ft4_turns for key, _ in turns) for turns in conflicts),
                "conflicts_with_copies": sum(any(key in copies for key, _ in turns) for turns in conflicts)}

    return {**summary(trained), "all_turns": summary(every)}


def check_context_turns(records: list[dict], changes: list[dict]) -> None:
    """TransformError unless the changes rows with target false are exactly the records' context turns (the
    turns below each record's context_turns, module docstring), and only FT4's own records carry the
    field."""
    context = {(r["id"], k) for r in records for k in range(r.get(CONTEXT_TURNS, 0))}
    not_target = {(c["id"], c["k"]) for c in changes if not c["target"]}
    if context != not_target:
        raise TransformError(f"the context turns and the changes rows that are no target differ: "
                             f"{sorted(context ^ not_target)[:5]}")
    marked = [r["id"] for r in records if CONTEXT_TURNS in r and r.get("run") != FT4_RUN]
    if marked:
        raise TransformError(f"records FT4 did not build carry {CONTEXT_TURNS}: {marked[:5]}")


def _ft4_report(applied: Ft4Records, new: dict | None, before: list[dict], out: list[dict],
                changes: list[dict], guard: EvalGuard, targets: dict, new_records: list[dict]) -> dict:
    """The manifest's "ft4": the rules' counts, records and turns, the samples by step type before and after
    (with the share of new samples), the prompts and FT4's guard."""
    rules = copy.deepcopy(applied.report["rules"])
    if new is not None:
        rules["R11"]["relative_dates"] = new["relative_dates"]
        rules["R14"] = {"defref": new["defref"], "explain": new["explain"]}
        rules["R13"]["glossary"] = new["glossary"]
    before_types, after_types = _by_type(before), _by_type(out)
    new_turns = sum(c["kind"] == "new" for c in changes)
    ft4_ids = {c["id"] for c in changes if c["kind"] != "relabel"}
    ft4_scores = [guard.max_eval_jaccard(_first_user(r)) for r in out if r["id"] in ft4_ids]
    all_scores = [guard.max_eval_jaccard(_first_user(r)) for r in out]
    limit = rq.TRAIN_VS_EVAL
    return {
        "generator": FT4_GENERATOR, "targets": targets, "rules": rules,
        "targets_in_input": applied.report["targets_in_input"],
        "records": {**applied.report["records"], "new": len(new_records), "out": len(out)},
        "turns": {**applied.report["turns"], "new": new_turns,
                  "context": sum(not c["target"] for c in changes),
                  "copy_target": sum(c["kind"] == "copy" and c["target"] for c in changes)},
        "samples": {"before": before_types, "after": after_types, "new": new_turns,
                    "new_share": round(new_turns / max(sum(after_types.values()), 1), 4)},
        "prompts": _prompt_conflicts(out, changes),
        "eval_overlap": {"limit": limit, "eval_texts": len(guard.deduper.eval.sizes),
                         "ft4_records": {"max_jaccard": round(max(ft4_scores, default=0.0), 4),
                                         "over": sum(s >= limit for s in ft4_scores)},
                         "all_records": {"max_jaccard": round(max(all_scores, default=0.0), 4),
                                         "over": sum(s >= limit for s in all_scores)},
                         "dropped": applied.report["eval_overlap"]["derived_dropped"]},
    }


# ---- the whole transformation ------------------------------------------------------------------------------


@dataclass
class Transformed:
    records: list[dict]
    new_requests: list[dict]
    report: dict
    changes: list[dict] = field(default_factory=list)  # FT4's changes rows (module docstring); [] without it


def _tally(records: list[dict]) -> dict:
    return {"total": len(records),
            "by_scenario": dict(sorted(Counter(r.get("scenario") for r in records).items())),
            "by_lang": dict(sorted(Counter(r.get("lang") for r in records).items()))}


def _first_user(record: dict) -> str:
    return next((m.get("content") or "" for m in record["messages"] if m.get("role") == "user"), "")


def _by_scenario_lang(records: list[dict]) -> dict:
    """{scenario: {lang: records}}, both sorted."""
    counts = Counter((r.get("scenario"), r.get("lang")) for r in records)
    return {s: {lang: counts[s, lang] for lang in sorted({lg for sc, lg in counts if sc == s})}
            for s in sorted({s for s, _ in counts})}


def _check_records(records: list[dict], by_id: dict[str, dict]) -> str:
    """The day of the input records' system prompt; TransformError for a record without a request, under
    another prompt or schema hash, or whose system message is not the current prompt of that one day."""
    today = ""
    for r in records:
        if r.get("id") not in by_id:
            raise TransformError(f"record {r.get('id')!r} has no request in the requests file")
        if r.get("prompt_hash") != PROMPT_HASH or r.get("schema_hash") != schemas.SCHEMA_HASH:
            raise TransformError(f"record {r['id']} carries prompt/schema hash {r.get('prompt_hash')}/"
                                 f"{r.get('schema_hash')}, not the current "
                                 f"{PROMPT_HASH}/{schemas.SCHEMA_HASH}")
        day = _today_of(r)
        if r["messages"][0].get("content") != system_prompt(day) or (today and day != today):
            raise TransformError(f"record {r['id']}: its system message is not the current prompt of one day")
        today = day
    if not today:
        raise TransformError("no input records")
    return today


def _record_rules(records: list[dict], by_id: dict[str, dict],
                  habits: dict | None) -> tuple[list[dict], dict, dict, dict, Counter, list[dict]]:
    """R1 (with habits) and R2 (on the clarify-scenario records) over every input record, then R4: the kept
    records and the counts of R1, R2 and R4, with the calls per tool of the input, and the records R4 dropped
    (after R1 and R2; FT4 cuts copies from them)."""
    r1 = {"deleted": Counter(), "kept_count": Counter(), "calls_changed": 0, "records_changed": 0}
    r2 = {"calls": 0, "changed": 0, "no_slots_missing": 0}
    changed = []
    for r in records:
        req = by_id[r["id"]]
        r, c1 = rule_r1(r, req, habits)
        r1["deleted"].update(c1["deleted"])
        r1["kept_count"].update(c1["kept_count"])
        r1["calls_changed"] += c1["calls_changed"]
        r1["records_changed"] += c1["records_changed"]
        if r.get("scenario") == "clarify":
            r, c2 = rule_r2(r, req)
            for k in r2:
                r2[k] += c2[k]
        changed.append(r)
    kept, r4 = rule_r4(changed)
    calls = Counter((c["function"]["name"]) for r in records for c in _calls(r))
    dropped = set(r4["dropped_ids"])
    return kept, r1, r2, r4, calls, [r for r in changed if r["id"] in dropped]


def _r1_report(r1: dict, calls: Counter) -> dict:
    return {"calls_seen": {k: calls[k] for k in ("search_notes", "list_notes", "fetch_fulltext")},
            "deleted": dict(sorted(r1["deleted"].items())),
            "kept_count": dict(sorted(r1["kept_count"].items())),
            "calls_changed": r1["calls_changed"], "records_changed": r1["records_changed"],
            "scope": "every input record, before R4"}


def _eval_overlap(records: list[dict], guard: EvalGuard) -> tuple[list[dict], list[str]]:
    """The records whose first user message stays below the evaluation limit, and the ids of the others."""
    kept, over = [], []
    for r in records:
        if guard.overlaps_eval(_first_user(r)):
            over.append(r["id"])
        else:
            kept.append(r)
    return kept, over


def transform_records(records: list[dict], requests: list[dict], *, eval_texts: list[str],
                      habits: dict | None = None, ft4: Ft4Options | None = None) -> Transformed:
    """--records-only (module docstring): R1 (with habits; default M6's), R2 on the clarify-scenario records,
    R4 and the evaluation-overlap drop over the input records, as transform applies them; no scripted record
    and no new request. With ft4, FT4's in-place rules and derived records too (not its new records).
    TransformError as in transform."""
    by_id = {r["id"]: r for r in requests}
    today = _check_records(records, by_id)
    kept, r1, r2, r4, calls, r4_dropped = _record_rules(records, by_id, habits)
    guard = EvalGuard(eval_texts=eval_texts, train_texts=[])  # only overlaps_eval is asked
    left, over = _eval_overlap(kept, guard)
    changes, ft4_section = [], None
    if ft4 is not None:
        ft4_guard = EvalGuard(eval_texts=eval_texts + list(ft4.eval_texts), train_texts=[])
        ft4_targets = {**FT4_TARGETS, **ft4.targets}
        before = list(left)
        ft4_applied = apply_ft4(left, by_id, topics=ft4.topics, dropped=r4_dropped, targets=ft4_targets,
                                guard=ft4_guard)
        left, changes = ft4_applied.records + ft4_applied.derived, ft4_applied.changes
        check_context_turns(left, changes)
        ft4_section = _ft4_report(ft4_applied, None, before, sorted(left, key=lambda r: r["id"]), changes,
                                  ft4_guard, ft4_targets, [])
        ft4_section["not_run"] = {"rules": FT4_NOT_RUN,
                                  "why": "they build new records from new requests with a scripted model: "
                                         "the full run's"}
    out = sorted(left, key=lambda r: r["id"])
    ids = [r["id"] for r in out]
    if len(set(ids)) != len(ids):
        raise TransformError("two output records share an id")
    scores = [guard.max_eval_jaccard(_first_user(r)) for r in out]
    applied = HABITS if habits is None else habits
    report = {
        "today": today,
        "mode": "records-only",
        "not_run": {"rules": list(SCRIPTED_RULES),
                    "why": "they build records with a scripted model: a one-off for M6's data, not a "
                           "transform of the input records"},
        "rules": {"R1": {**_r1_report(r1, calls),
                         "habits": {f"{t}.{p}": v for (t, p), v in sorted(applied.items())}},
                  "R2": {**r2, "scope": "clarify-scenario input records, before R4"},
                  "R4": r4},
        "records": {"in": _tally(records), "after_r4": _tally(kept), "out": _tally(out)},
        "by_scenario_lang": {"in": _by_scenario_lang(records), "after_r4": _by_scenario_lang(kept),
                             "out": _by_scenario_lang(out)},
        "eval_overlap": {"limit": rq.TRAIN_VS_EVAL, "max_jaccard": round(max(scores, default=0.0), 4),
                         "over": sum(s >= rq.TRAIN_VS_EVAL for s in scores), "input_records_dropped": over,
                         "eval_texts": len(eval_texts)},
    }
    if ft4_section is not None:
        report["ft4"] = ft4_section
    return Transformed(out, [], report, changes)


def transform(records: list[dict], requests: list[dict], *, eval_texts: list[str], terms: dict,
              corpus: rq.Corpus, split_seed: int, seed: int, held_out: tuple[str, ...] = (),
              targets: dict | None = None, habits: dict | None = None,
              ft4: Ft4Options | None = None) -> Transformed:
    """All rules over the input records (R1 with habits; default M6's), and FT4's with ft4; TransformError on
    input the rules cannot be applied to."""
    targets = {**TARGETS, **(targets or {})}
    by_id = {r["id"]: r for r in requests}
    today = _check_records(records, by_id)
    problems = check_terms(terms, corpus, held_out=held_out)
    if problems:
        raise TransformError("term list refused: " + "; ".join(problems[:10]))
    template_result = audit_new_templates(LIST_NOTES_TEMPLATES)
    if not template_result["ok"]:
        raise TransformError(f"template audit failed: {template_result['over']}")

    # R1, R2 over every input record; then R4.
    kept, r1, r2, r4, r1_calls, r4_dropped = _record_rules(records, by_id, habits)

    train = sorted((r for r in requests if r["split"] == "train"), key=lambda r: r["id"])
    guard = EvalGuard(eval_texts=eval_texts, train_texts=[r["text"] for r in train])

    # R3
    r3_records, r3_overlap = [], 0
    eligible, r3_train = r3_train_requests(train, kept)
    for req in eligible:
        if len(r3_records) >= targets["r3_total"]:
            break
        if guard.overlaps_eval(req["text"]):
            r3_overlap += 1
            continue
        r3_records.append(clarification_record(req, today, record_id=f"{req['id']}-m6r3"))
    from_train = len(r3_records)
    offered = guard.deduper.offered
    dropped_before = dict(guard.deduper.dropped)
    r3_new = generate_clarify_requests(targets["r3_total"] - from_train, seed=seed, corpus=corpus,
                                       split_seed=split_seed, guard=guard)
    r3_dedup = {k: guard.deduper.dropped[k] - dropped_before[k] for k in rq.DROP_REASONS}
    r3_offered = guard.deduper.offered - offered
    # R5
    dropped_before, offered = dict(guard.deduper.dropped), guard.deduper.offered
    r5_new = generate_list_notes_requests(targets["r5"], seed=seed, corpus=corpus, split_seed=split_seed,
                                          guard=guard)
    r5_dedup = {k: guard.deduper.dropped[k] - dropped_before[k] for k in rq.DROP_REASONS}
    r5_offered = guard.deduper.offered - offered
    # R7
    r7_new, r7_overlap = term_requests(terms, targets["r7"], guard)

    new_requests = []
    for i, req in enumerate([*r3_new, *r5_new, *r7_new], 1):
        new_requests.append({"id": f"req-m6-{i:04d}", **req})
    n3, n5 = len(r3_new), len(r5_new)
    r3_records += [clarification_record(req, today) for req in new_requests[:n3]]
    r5_records = [list_notes_record(req, today) for req in new_requests[n3:n3 + n5]]
    r7_records = [term_record(req, today) for req in new_requests[n3 + n5:]]

    # R6
    candidates = [r for r in train if r["scenario"] == "normal" and (r.get("slots") or {}).get("topic")
                  and any(k in r["slots"] for k in ("categories", *DATE_KEYS))]
    random.Random(f"{seed}:m6:r6").shuffle(candidates)
    r6_records, r6_overlap, relaxed = [], 0, Counter()
    for req in candidates:
        if len(r6_records) >= targets["r6"]:
            break
        if guard.overlaps_eval(req["text"]):
            r6_overlap += 1
            continue
        rec = relax_record(req, today, record_id=f"{req['id']}-m6r6")
        r6_records.append(rec)
        relaxed[relax_constraint({k: 1 for k in req["slots"] if k in ("categories", *DATE_KEYS)})[1]] += 1

    # Every output record against the evaluation texts.
    old_kept, old_overlap = _eval_overlap(kept, guard)
    scripted = [*r3_records, *r5_records, *r6_records, *r7_records]
    changes, ft4_section, ft4_records = [], None, []
    if ft4 is not None:  # FT4 after M6's rules, with a guard of its own: M6's requests stay as they were
        ft4_guard = EvalGuard(eval_texts=eval_texts + list(ft4.eval_texts),
                              train_texts=[r["text"] for r in train] + [r["text"] for r in new_requests])
        ft4_targets = {**FT4_TARGETS, **ft4.targets}
        before = [*old_kept, *scripted]
        ft4_applied = apply_ft4(old_kept, by_id, topics=ft4.topics, dropped=r4_dropped, targets=ft4_targets,
                                guard=ft4_guard)
        ft4_requests, ft4_new_records, new_changes, new_report = ft4_new(
            today=today, corpus=corpus, split_seed=split_seed, seed=seed, guard=ft4_guard,
            targets=ft4_targets, glossary=ft4.glossary, r7_terms=terms, held_out=held_out,
            sort_by=relative_sort_by([("input", records), *ft4.sort_inputs])[1],
            science_only=ft4.science_only, domain=ft4.domain)
        old_kept = ft4_applied.records
        ft4_records = [*ft4_applied.derived, *ft4_new_records]
        changes = sorted([*ft4_applied.changes, *new_changes], key=lambda c: (c["id"], c["k"]))
        new_requests = [*new_requests, *ft4_requests]
        ft4_out = sorted([*old_kept, *scripted, *ft4_records], key=lambda r: r["id"])
        check_context_turns(ft4_out, changes)
        ft4_section = _ft4_report(ft4_applied, new_report, before, ft4_out, changes, ft4_guard, ft4_targets,
                                  ft4_new_records)
    out = sorted([*old_kept, *scripted, *ft4_records], key=lambda r: r["id"])
    ids = [r["id"] for r in out]
    if len(set(ids)) != len(ids):
        raise TransformError("two output records share an id")
    scores = [guard.max_eval_jaccard(_first_user(r)) for r in out]
    report = {
        "today": today,
        "rules": {
            "R1": _r1_report(r1, r1_calls),
            "R2": {**r2, "scope": "clarify-scenario input records, before R4"},
            "R3": {"records": len(r3_records), "from_train_requests": from_train, "new_requests": n3,
                   "train_requests": r3_train,
                   "dropped_eval_overlap": r3_overlap, "new_offered": r3_offered,
                   "new_dedup_dropped": r3_dedup,
                   "target": targets["r3_total"]},
            "R4": r4,
            "R5": {"records": len(r5_records), "offered": r5_offered, "dedup_dropped": r5_dedup,
                   "target": targets["r5"], "templates": LIST_NOTES_TEMPLATES,
                   "template_audit": template_result},
            "R6": {"records": len(r6_records), "candidates": len(candidates),
                   "dropped_eval_overlap": r6_overlap,
                   "relaxed": dict(sorted(relaxed.items())), "target": targets["r6"]},
            "R7": {"records": len(r7_records), "terms": len(terms), "dropped_eval_overlap": r7_overlap,
                   "templates": list(TERM_TEMPLATES), "target": targets["r7"]},
        },
        "records": {"in": _tally(records), "after_r4": _tally(kept), "out": _tally(out),
                    "new_by_rule": {"R3": len(r3_records), "R5": len(r5_records), "R6": len(r6_records),
                                    "R7": len(r7_records)}},
        "eval_overlap": {"limit": rq.TRAIN_VS_EVAL, "max_jaccard": round(max(scores, default=0.0), 4),
                         "over": sum(s >= rq.TRAIN_VS_EVAL for s in scores),
                         "input_records_dropped": old_overlap, "eval_texts": len(eval_texts)},
    }
    if ft4_section is not None:
        report["ft4"] = ft4_section
    return Transformed(out, new_requests, report, changes)


# ---- CLI ---------------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Transform the synthetic training file for M6 (R1-R7).")
    parser.add_argument("synth", type=Path, help="the synthetic training file (agent-history records)")
    parser.add_argument("requests", type=Path, help="the requests file the records were generated from")
    parser.add_argument("--out", type=Path, default=OUT, help=f"default: {OUT.as_posix()}")
    parser.add_argument("--manifest", type=Path, default=MANIFEST, help=f"default: {MANIFEST.as_posix()}")
    parser.add_argument("--requests-out", type=Path, default=REQUESTS_OUT,
                        help=f"the new requests (default: {REQUESTS_OUT.as_posix()})")
    parser.add_argument("--terms", type=Path, default=TERMS, help=f"default: {TERMS.as_posix()}")
    parser.add_argument("--dev", type=Path, default=DEV,
                        help="the development set, read for the overlap check only "
                             f"(default: {DEV.as_posix()})")
    parser.add_argument("--corpus", type=Path, default=rq.CORPUS_MANIFEST,
                        help=f"default: {rq.CORPUS_MANIFEST.as_posix()}")
    parser.add_argument("--seed", type=int, default=42, help="the seed of the new requests' draws")
    parser.add_argument("--force", action="store_true", help="replace --out and --requests-out")
    for key, value in TARGETS.items():
        parser.add_argument(f"--{key.replace('_', '-')}", type=int, default=value, dest=key,
                            help=f"default: {value}")
    parser.add_argument("--records-only", action="store_true",
                        help="only the rules that change the input records (R1, R2, R4, the "
                             "evaluation-overlap drop): no scripted record, no new request, no term list "
                             "(M7)")
    parser.add_argument("--derive-habits", action="store_true",
                        help="take R1's habit values from the input records as M6 took 5 and 20 from its own "
                             "(recorded in the manifest)")
    parser.add_argument("--eval-requests", type=Path, nargs="+", default=[], metavar="FILE",
                        help="more requests files whose dev and test rows join the evaluation texts")
    parser.add_argument("--eval-items", type=Path, nargs="+", default=[], metavar="FILE",
                        help="more evaluation item files (development sets) whose user messages join the "
                             "evaluation texts; refused if they hold test material")
    ft4 = parser.add_argument_group("FT4 (docs/m7-本地提升计划.md WP6; module docstring), off without --ft4")
    ft4.add_argument("--ft4", action="store_true", help="apply FT4's rules R8-R11, R13 and R14")
    ft4.add_argument("--changes-out", type=Path, default=None,
                     help="FT4's changes file: one line per turn an FT4 rule changed, built or copied "
                          "(required with --ft4)")
    ft4.add_argument("--ft4-eval-requests", type=Path, nargs="+", default=None, metavar="FILE",
                     help="more requests files whose dev and test rows join FT4's guard only")
    ft4.add_argument("--ft4-eval-items", type=Path, nargs="+", default=None, metavar="FILE",
                     help="more development item files whose user messages join FT4's guard only; refused "
                          "if they hold test material")
    for key, value in FT4_TARGETS.items():
        ft4.add_argument(f"--{key.replace('_', '-')}", type=int, default=None, dest=key,
                         help=f"default: {value}")
    ft4.add_argument("--glossary", type=Path, default=None,
                     help="R13's glossary (data/glossary/zh-en.json, read with research_agent.glossary): "
                          "R7-style records from its eligible terms R7 did not use, its AITD sections read "
                          "from the provenance.json beside it (the full run only)")
    ft4.add_argument("--r11-sort-inputs", type=Path, nargs="+", default=None, metavar="FILE",
                     help="more agent-history files (the M7 input) whose dated find_recent_papers searches "
                          "join R11's sort_by count (the full run only)")
    return parser


def _ft4_usage(args: argparse.Namespace) -> str | None:
    """Why the FT4 options of args cannot run, or None: an FT4 option without --ft4, --ft4 without
    --changes-out, an option of FT4's new records with --records-only."""
    given = [f"--{k.replace('_', '-')}" for k in FT4_TARGETS if getattr(args, k) is not None]
    others = (("--changes-out", args.changes_out), ("--ft4-eval-requests", args.ft4_eval_requests),
              ("--ft4-eval-items", args.ft4_eval_items), ("--glossary", args.glossary),
              ("--r11-sort-inputs", args.r11_sort_inputs))
    given += [name for name, value in others if value is not None]
    if not args.ft4 and given:
        return f"{', '.join(given)} need --ft4"
    if args.ft4 and args.changes_out is None:
        return "--ft4 needs --changes-out"
    new_only = [f"--{k.replace('_', '-')}" for k in FT4_NEW_OPTIONS if getattr(args, k) is not None]
    new_only += [name for name, value in (("--glossary", args.glossary),
                                          ("--r11-sort-inputs", args.r11_sort_inputs)) if value is not None]
    if args.ft4 and args.records_only and new_only:
        return f"{', '.join(new_only)}: --records-only builds no new FT4 record"
    return None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_records(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(split_lines(path.read_text(encoding="utf-8")), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TransformError(f"{path}: line {number} is not JSON: {exc}") from exc
        if not is_agent_history(value):
            raise TransformError(f"{path}: line {number} is not an agent-history record")
        rows.append(value)
    return rows


def _dev_texts(path: Path) -> list[str]:
    items = read_jsonl(path)
    refuse_test_material(items, path)
    return [m["content"] for item in items for m in item.get("messages") or []
            if m.get("role") == "user" and isinstance(m.get("content"), str)]


def _eval_request_texts(path: Path) -> list[str]:
    """The texts of a requests file's dev and test rows."""
    return [r["text"] for r in load_requests(path) if r["split"] != "train"]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    usage = _ft4_usage(args)
    if usage is not None:
        print(usage, file=sys.stderr)
        return 2
    req_manifest = requests_manifest_of(args.requests) if args.requests.is_file() else None
    inputs = {"synth": args.synth, "requests": args.requests, "terms": args.terms, "dev": args.dev,
              "corpus": args.corpus}
    if args.records_only:  # no scripted record: the term list is not read
        del inputs["terms"]
    provenance = None if args.glossary is None else args.glossary.with_name("provenance.json")
    ft4_inputs = [*(args.ft4_eval_requests or []), *(args.ft4_eval_items or []),
                  *([args.glossary, provenance] if args.glossary is not None else []),
                  *(args.r11_sort_inputs or [])]
    missing = [str(p) for p in [*inputs.values(), *args.eval_requests, *args.eval_items, *ft4_inputs]
               if not p.is_file()]
    if req_manifest is None:
        missing.append(f"the manifest next to {args.requests}")
    if missing:
        print(f"missing input: {', '.join(missing)}", file=sys.stderr)
        return 2
    outputs = (args.out,) if args.records_only else (args.out, args.requests_out)
    present = [str(p) for p in (*outputs, *([args.changes_out] if args.ft4 else [])) if p.exists()]
    if present and not args.force:
        print(f"{', '.join(present)} exists; pass --force to replace it", file=sys.stderr)
        return 2
    staged = args.out.with_name(f"{args.out.stem}.staged{args.out.suffix}")
    habit_account = None
    ft4_requests, ft4_items, glossary, ft4_options, sort_inputs = [], [], None, None, []
    try:
        records = _read_records(args.synth)
        requests = load_requests(args.requests)
        manifest_in = json.loads(req_manifest.read_text(encoding="utf-8"))
        corpus = rq.load_corpus(args.corpus)
        split_seed = manifest_in["seed"]
        topics = rq.split_topics(corpus, split_seed)
        if {s: [t.id for t in topics[s]] for s in ("train", "dev", "test")} != \
                {s: manifest_in["topics"][s] for s in ("train", "dev", "test")}:
            raise TransformError("the requests manifest's topics are not the corpus split of its seed")
        eval_texts = [r["text"] for r in requests if r["split"] != "train"] + _dev_texts(args.dev)
        more_requests = [(path, _eval_request_texts(path)) for path in args.eval_requests]
        more_items = [(path, _dev_texts(path)) for path in args.eval_items]
        eval_texts += [text for _, texts in [*more_requests, *more_items] for text in texts]
        habits = None
        if args.derive_habits:
            habits, habit_account = derive_habits(records)
        if args.ft4:
            ft4_requests = [(path, _eval_request_texts(path)) for path in args.ft4_eval_requests or []]
            ft4_items = [(path, _dev_texts(path)) for path in args.ft4_eval_items or []]
            science_only, domain = frozenset(), None
            if args.glossary is not None:  # a GlossaryError is a ValueError: exit 1
                glossary = load_glossary(args.glossary)
                science_only, domain = science_only_terms(provenance)
            sort_inputs = [(path.as_posix(), _read_records(path)) for path in args.r11_sort_inputs or []]
            ft4_options = Ft4Options(
                topics=topics["train"],
                targets={k: getattr(args, k) for k in FT4_TARGETS if getattr(args, k) is not None},
                eval_texts=[text for _, texts in [*ft4_requests, *ft4_items] for text in texts],
                glossary=glossary, science_only=science_only, domain=domain, sort_inputs=sort_inputs)
        if args.records_only:
            result = transform_records(records, requests, eval_texts=eval_texts, habits=habits,
                                       ft4=ft4_options)
        else:
            terms = json.loads(args.terms.read_text(encoding="utf-8"))
            result = transform(records, requests, eval_texts=eval_texts, terms=terms, corpus=corpus,
                               split_seed=split_seed, seed=args.seed,
                               held_out=tuple(manifest_in["topics"]["dev"] + manifest_in["topics"]["test"]),
                               targets={k: getattr(args, k) for k in TARGETS}, habits=habits, ft4=ft4_options)
        staged.parent.mkdir(parents=True, exist_ok=True)
        data = "".join(canonical_json(r) + "\n" for r in result.records).encode("utf-8")
        staged.write_bytes(data)
        loaded = load_records(staged)
        if len(loaded) != len(result.records) or not all(is_agent_history(r) for r in loaded):
            raise TransformError("the written file does not read back as the records written")
    except (TransformError, RecordError, RefusedTestSet, ValueError, KeyError, OSError) as exc:
        staged.unlink(missing_ok=True)
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    os.replace(staged, args.out)
    file_info = {name: {"path": path.as_posix(), "sha256": sha256_file(path)} for name, path in
                 {**inputs, "requests_manifest": req_manifest}.items() if name != "terms"}
    for key, more in (("eval_requests", more_requests), ("eval_items", more_items),
                      ("ft4_eval_requests", ft4_requests), ("ft4_eval_items", ft4_items)):
        if more:
            file_info[key] = [{"path": path.as_posix(), "sha256": sha256_file(path), "texts": len(texts)}
                              for path, texts in more]
    manifest = {
        "generator": GENERATOR, "seed": args.seed, "split_seed": split_seed, "prompt_hash": PROMPT_HASH,
        "schema_hash": schemas.SCHEMA_HASH, **result.report, "inputs": file_info,
        "output": {"path": args.out.as_posix(), "sha256": hashlib.sha256(data).hexdigest(),
                   "bytes": len(data),
                   "records": len(result.records)},
    }
    if habit_account is not None:
        manifest["habits"] = habit_account
    if args.ft4:
        changes_data = "".join(canonical_json(c) + "\n" for c in result.changes).encode("utf-8")
        args.changes_out.parent.mkdir(parents=True, exist_ok=True)
        args.changes_out.write_bytes(changes_data)
        manifest["ft4"]["changes"] = {"path": args.changes_out.as_posix(), "rows": len(result.changes),
                                      "sha256": hashlib.sha256(changes_data).hexdigest()}
        if glossary is not None:
            file_info["glossary"] = {"path": args.glossary.as_posix(), "sha256": sha256_file(args.glossary),
                                     "terms": len(glossary), "blocked": len(glossary.blocked)}
            file_info["glossary_provenance"] = {"path": provenance.as_posix(),
                                                "sha256": sha256_file(provenance)}
        if sort_inputs:
            file_info["r11_sort_inputs"] = [{"path": name, "sha256": sha256_file(Path(name)),
                                             "records": len(records)} for name, records in sort_inputs]
    if not args.records_only:
        req_data = rq.requests_text(result.new_requests).encode("utf-8")
        args.requests_out.parent.mkdir(parents=True, exist_ok=True)
        args.requests_out.write_bytes(req_data)
        manifest["terms"] = {"path": args.terms.as_posix(), "sha256": sha256_file(args.terms),
                             "count": len(terms), "held_out_topics": sorted(manifest_in["topics"]["dev"]
                                                                            + manifest_in["topics"]["test"])}
        manifest["requests_out"] = {
            "path": args.requests_out.as_posix(), "count": len(result.new_requests),
            "sha256": hashlib.sha256(req_data).hexdigest(),
            "by_family_prefix": dict(sorted(Counter(
                "/".join(r["family"].split("/")[:2]) if r["family"].startswith(("m6/", "ft4/"))
                else "requests.TEMPLATES" for r in result.new_requests).items()))}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
    rules = result.report["rules"]
    print(f"R1 deleted {rules['R1']['deleted']}, kept for count {rules['R1']['kept_count']}, "
          f"{rules['R1']['calls_changed']} calls in {rules['R1']['records_changed']} records")
    print(f"R2 {rules['R2']['changed']} of {rules['R2']['calls']} ask_clarification calls changed, "
          f"{rules['R2']['no_slots_missing']} without slots.missing")
    if args.records_only:
        print(f"R3, R5, R6, R7 not run (--records-only); R1 habits {rules['R1']['habits']}")
    else:
        print(f"R3 {rules['R3']['records']} records ({rules['R3']['from_train_requests']} from train "
              f"requests, {rules['R3']['new_requests']} new)")
    print(f"R4 first step {rules['R4']['first_step']}, second {rules['R4']['second_step']}, "
          f"dropped {rules['R4']['late_dropped']}")
    if not args.records_only:
        print(f"R5 {rules['R5']['records']}, R6 {rules['R6']['records']} {rules['R6']['relaxed']}, "
              f"R7 {rules['R7']['records']} records")
    print(f"eval overlap: max {result.report['eval_overlap']['max_jaccard']} "
          f"(limit {rq.TRAIN_VS_EVAL}), {result.report['eval_overlap']['over']} at or over")
    if args.ft4:
        _print_ft4(manifest["ft4"], records_only=args.records_only)
    print(f"records in {result.report['records']['in']['total']}, out {len(result.records)}; "
          f"{args.out.as_posix()} {len(data)} bytes; {len(result.new_requests)} new requests")
    return 0


def _print_ft4(section: dict, *, records_only: bool) -> None:
    """FT4's counts, a few lines (the manifest's "ft4" holds them all)."""
    r = section["rules"]
    s4a, made, pick = r["R9"]["s4a"], r["R9"]["constructed"], r["R9"]["s4a"]["pick"]
    print(f"FT4: R8 relabeled {r['R8']['relabeled']} ({r['R8']['identical']} identical, "
          f"{r['R8']['at_run_limit']} at the run limit, {r['R8']['mixed_step']} mixed steps); R9 S4a "
          f"{s4a['relabeled']} (list {s4a['list']}, pick: ask {pick['ask']}, one {pick['one']}, none "
          f"{pick['none']}), S4b {r['R9']['s4b']['relabeled']}, constructed {made['records']} of "
          f"{made['candidates']} candidates ({made['not_built_after_r9_cuts']} S4a points after R9's cuts "
          "not built)")
    d, r10 = r["R10"]["derived"], r["R10"]
    r4 = d["from_r4_dropped"]
    print(f"FT4: R10 relabeled {r10['relabeled']} (labelled turns here {r10['labelled_in_input']}, removed "
          f"by an earlier cut {len(r10['labelled_removed_by_an_earlier_cut'])}; skipped: not trailing "
          f"{len(r10['skipped_not_trailing'])}, partial list {len(r10['skipped_partial_list'])}), derived "
          f"{d['records']}: new {d['new']} of {d['candidates_new']} candidates (from R11's records "
          f"{d['from_r11']['new']}, from R4's dropped {r4['new']}: one per record at its first complete "
          f"listing, {r4['with_a_complete_listing']} of the {r4['with_a_listing_point']} with a listing "
          f"point; {len(d['duplicate_targets'])} with another's target left out), the teacher's ask kept "
          f"{d['copy_target']}; R11 relabeled {r['R11']['relabeled']}; R13 rewritten "
          f"{r['R13']['rewritten']} of {r['R13']['pure_chinese']} pure-Chinese note queries "
          f"({r['R13']['no_term']} without a train topic)")
    if not records_only:
        glossary, sort_by = r["R13"]["glossary"], r["R11"]["relative_dates"]["sort_by"]
        order = (f"sort_by {sort_by['chosen']}" if sort_by["chosen"] else
                 f"sort_by left out: the default {sort_by['default']} is the majority")
        print(f"FT4 new: R11 relative dates {r['R11']['relative_dates']['records']} ({order} of the "
              f"teacher's dated {'/'.join(sort_by['intents'])} searches {sort_by['effective']}, by input "
              f"{sort_by['counts']}), R14 definite references {r['R14']['defref']['records']}, "
              f"explain_tools {r['R14']['explain']['records']}")
        if glossary is None:
            print("FT4 new: R13 glossary not run")
        else:
            skipped = glossary["skipped"]
            print(f"FT4 new: R13 glossary {glossary['requests']} requests from {glossary['kept']} of "
                  f"{glossary['terms']} noted terms ({glossary['blocked']} blocked apart; skipped: "
                  f"non-ASCII English {skipped['non_ascii']}, Latin token missing "
                  f"{skipped['latin_token_missing']}, AI for Science only {skipped['science_only']}; "
                  f"eligible {glossary['eligible']}; already in R7 {skipped['already_in_r7']}, refused "
                  f"{skipped['refused']}; evaluation overlaps {glossary['dropped_eval_overlap']})")
    t = section["targets_in_input"]
    print(f"FT4 targets in the input (cut or not): R8 {t['R8']['not_identical']} not identical of "
          f"{t['R8']['transient_steps']} after a transient error; R9 S4a {t['R9']['s4a_note_queries']} note "
          f"queries, S4b {t['R9']['s4b_arxiv_searches']} arXiv searches; R10 {t['R10']['labelled']} "
          f"labelled; R11 {t['R11']['records']} records; R13 {t['R13']['pure_chinese']} pure-Chinese "
          f"queries, {t['R13']['with_a_term']} with a train topic")
    samples = section["samples"]
    print(f"FT4 samples by step type, before {samples['before']}, after {samples['after']}")
    e, p, n, turns = section["eval_overlap"], section["prompts"], section["records"], section["turns"]
    print(f"FT4 records: {n['relabeled']} relabeled, {n['derived']} derived, {n['new']} new; turns removed "
          f"{turns['removed']}, context {turns['context']}, the teacher's kept {turns['copy_target']}; new "
          f"samples {samples['new']} ({samples['new_share']:.1%})")
    print(f"FT4 prompts, context turns left out: {p['turns']} turns, {p['shared_prompts']} shared, "
          f"{p['conflicts']} with another answer ({p['conflicts_with_ft4']} with an FT4 turn, "
          f"{p['conflicts_with_copies']} with a copied turn); over all turns: {p['all_turns']['turns']} "
          "turns, "
          f"{p['all_turns']['shared_prompts']} shared, {p['all_turns']['conflicts']} with another answer "
          f"({p['all_turns']['conflicts_with_copies']} with a copied turn)")
    print(f"FT4 guard ({e['eval_texts']} texts): max {e['ft4_records']['max_jaccard']} over FT4's records "
          f"({e['ft4_records']['over']} at or over), {e['all_records']['max_jaccard']} over all "
          f"({e['all_records']['over']} at or over), {len(e['dropped'])} derived dropped; changes "
          f"{section['changes']['rows']} rows")


if __name__ == "__main__":
    sys.exit(main())

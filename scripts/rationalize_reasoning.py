"""Write the missing reasoning of the synthetic training turns with deepseek-flash (docs/m7-家族层设计.md §6;
docs/m7-计划.md §2 思考数据, WP-C).

    .venv/Scripts/python.exe scripts/rationalize_reasoning.py --dry-run
    .venv/Scripts/python.exe scripts/rationalize_reasoning.py [--max-cost 8] [--stage M7]
        [--records data/raw/synth/synth-m6.jsonl] [--teacher data/raw/synth/reasoning-m7-teacher.jsonl]
        [--out data/raw/synth/reasoning-m7-rationalized.jsonl]
        [--report data/raw/synth/reasoning-m7-rationalized.report.json]
        [--dry-run-report data/raw/synth/reasoning-m7-rationalize.dryrun.json] [--tokenizer models/Qwen3-4B]

Targets. Every assistant turn of the records (data.reasoning's turn keys, in (id, k) order) that the teacher
sidecar of scripts/join_reasoning.py does not cover: the turns M6 edited, those whose teacher reasoning was
empty and the turns of the scripted records (about 2,294 in synth-m6). The sidecar is read with
data.reasoning.ReasoningIndex, so a row written for another answer stops the run, and so does a row for a turn
the records do not have; records holding test material are refused (eval.lock.refuse_test_material), since
their text is sent out.

Prompt. A system message, the same in every prompt: the instructions (the requirements below) and three style
examples, the teacher reasoning of three train turns of the sidecar, one per answer kind (a tool call, an
ask_clarification, a text answer). Each is the eligible row with the smallest SHA-256 of "<id>/<k>", so the
choice does not depend on the file's order; a row is eligible when its reasoning has the requested form (no
CJK character, 2-6 sentences, at most 120 words) and passes the checks below. Then a user message: the context
the training sample shows (agent.context.build_context of the record's history before the turn) as text, every
tool output cut to TOOL_OUTPUT_CHARS characters, and the answer, its text and one "call <name> <arguments as
written>" line per call. The tool schemas are not shown: the reasoning has no business with arguments the
answer leaves out.

Requirements (in the prompt): English, 2-6 sentences, at most 120 words; never contradict the answer; name
every called tool exactly; mention no argument the answer leaves out, except to say that the defaults are
fine.

Checks (problems(), on every attempt): not empty; no think tag; at most 400 tokens (--tokenizer,
data.reasoning.token_counter); every tool the answer calls is named; not cut at max_tokens. A request DeepSeek
refuses (HTTP 400) is a failed attempt too. A turn whose first attempt fails is asked once more, the user
message ending with a nonce: a random-looking string derived from the turn's key, so that the retry is another
request while a rerun asks the very same and is served from the cache. A turn that fails twice gets no row and
stays NO_THINK in training; the report counts it by problem.

Client. llm.cloud.CloudClient("deepseek-flash", thinking="off", off_peak_only=True) over the response cache
and the ledger in runs/llm/ that every run shares, booked under --stage (M7) at what DeepSeek charges (half
the list price off peak); the cap of this invocation is what the stage has spent plus --max-cost, never above
the stage's cap (eval.systems.stage_cap). Requests carry no tools, temperature 0.0 and max_tokens MAX_TOKENS.
In DeepSeek's peak hours, or once the cap is reached, the ledger stops the run before the next paid call (exit
3); the report says so and no rows are written. Answers already paid are in the cache, so a rerun goes on at
no cost for them.

Outputs. --out: the sidecar rows of the turns that passed (data.reasoning: source "rationalized", model
deepseek-flash, the cache key of the request whose answer is used, context_equal true, since the reasoning is
written for the training context itself), in (id, k) order; with the cache, a rerun gives the same bytes.
--report: the inputs' SHA-256, the examples by key, the targets by origin, the prompts' SHA-256, the attempts
and failures, paid and cached calls, the ledger before and after, the lengths of the written reasoning
(characters and tokens) and the output's SHA-256.

Dry run. --dry-run builds every first prompt and its retry, counts their tokens with --tokenizer (standing in
for DeepSeek's tokenizer, which is not available here) and writes --dry-run-report: the targets, the examples,
the prompts' SHA-256 and token percentiles, an estimate (one call per target answering in
EXPECTED_COMPLETION_TOKENS) and a ceiling (every target retried, every answer at max_tokens), priced with the
ledger's own arithmetic (llm.budget.cost_rmb over prices.json, times the off-peak factor the ledger books). It
opens neither the API client nor the cache nor the ledger.

Outputs of other records. The defaults name M7's first pass (synth-m6), and a later pass over another records
file (M7b: data/raw/synth/synth-m7.jsonl) must name its own outputs. A run that would replace a file another
records file's pass wrote stops before the ledger, the cache or the client is opened (exit 2) and leaves it as
it is: a dry run whose --dry-run-report, a run whose --report, is not this script's report of the same
records file (inputs.records.sha256), and a run whose --out holds a row of a turn the records do not have. A
rerun over the same records replaces them, as before.

Exit 0 done; 1 refused input (a stale or malformed teacher sidecar, a record that is not an agent history or
whose markers disagree, test material, no eligible example of some kind); 2 usage, a missing input, no API
key or an output of other records; 3 stopped by the ledger (peak hours or the cap).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import openai

from research_agent.agent.context import build_context
from research_agent.canonical import canonical_json
from research_agent.data.reasoning import (
    RATIONALIZED,
    TEACHER,
    ReasoningError,
    ReasoningIndex,
    has_think_tag,
    is_scripted,
    length_stats,
    sidecar_row,
    token_counter,
    turns,
    write_sidecar,
)
from research_agent.eval.lock import RefusedTestSet, refuse_test_material
from research_agent.eval.systems import STAGE_CAPS, stage_cap
from research_agent.llm.budget import Budget, BudgetExceeded, cost_rmb, load_off_peak, load_prices
from research_agent.llm.cache import ResponseCache, request_key
from research_agent.llm.cloud import THINKING_OFF, CloudClient, MissingKeyError
from research_agent.train.samples import RecordError, is_agent_history, load_records

RECORDS = Path("data/raw/synth/synth-m6.jsonl")
TEACHER_SIDECAR = Path("data/raw/synth/reasoning-m7-teacher.jsonl")
OUT = Path("data/raw/synth/reasoning-m7-rationalized.jsonl")
REPORT = Path("data/raw/synth/reasoning-m7-rationalized.report.json")
DRY_RUN_REPORT = Path("data/raw/synth/reasoning-m7-rationalize.dryrun.json")
TOKENIZER = Path("models/Qwen3-4B")
LLM_DIR = Path("runs/llm")
MODEL = "deepseek-flash"
STAGE = "M7"
MAX_COST = 8.0  # RMB this invocation may add to the stage; the plan budgets about 6 (docs/m7-计划.md, 预算)
TEMPERATURE = 0.0
MAX_TOKENS = 512  # bounds what a runaway answer costs, above the 400-token check so that one fails as such
MAX_REASONING_TOKENS = 400
TOOL_OUTPUT_CHARS = 1500
EXPECTED_COMPLETION_TOKENS = 160  # about the 120 words the prompt allows at most
ASK = "ask_clarification"
KINDS = ("call", "clarify", "answer")  # the style examples' answer kinds, in the order the prompt shows them
MAX_WORDS, MIN_SENTENCES, MAX_SENTENCES = 120, 2, 6
_CJK = re.compile(r"[\u3400-\u9fff\uf900-\ufaff]")  # CJK ideographs: extension A, unified, compatibility
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")

# The problems of an attempt (problems(); REFUSED when the API refuses the request).
CUT, EMPTY, THINK_TAG = "cut at max_tokens", "empty", "think tag"
TOO_LONG, NOT_NAMED = f"over {MAX_REASONING_TOKENS} tokens", "a called tool is not named"
REFUSED = "refused by the API"

INSTRUCTIONS = (
    "You are shown what a research-assistant agent saw at one step of a task: its system prompt, the user's "
    "request, and its earlier tool calls with their results (long results are cut short). You are also shown "
    "the reply the agent gave at that step. Write the reasoning the agent had just before it gave that "
    "reply.\n"
    "\nRequirements:\n"
    "- English, 2 to 6 sentences, at most 120 words, plain prose without lists, headings or tags.\n"
    "- Reason towards exactly this reply and never contradict it.\n"
    "- Name every tool the reply calls by its exact name, for example search_arxiv.\n"
    "- Do not mention any argument the reply leaves out, except to say that the defaults are fine.\n"
    "- Write only the reasoning, in the agent's own voice, not the reply itself.\n"
    "\nThree examples of the style, each from another step:"
)
REQUEST = "Write the agent's reasoning before this reply."

CountTokens = Callable[[str], int]


@dataclass(frozen=True)
class Target:
    """A turn to write reasoning for: its key, whether a scripted record holds it, the context the training
    sample shows and the training answer."""

    id: str
    k: int
    scripted: bool
    context: list[dict]
    answer: dict


@dataclass(frozen=True)
class Attempt:
    """One request's answer text, its problems ([] when it passes) and the cache key it is filed under."""

    text: str
    problems: list[str]
    key: str


@dataclass
class Outcome:
    """What a run did: the rows written, the attempts that failed (by problem) and why it stopped."""

    rows: list[dict] = field(default_factory=list)
    first_problems: Counter = field(default_factory=Counter)  # of the first attempts that failed
    failures: Counter = field(default_factory=Counter)  # of the retries that failed as well: NO_THINK turns
    retried: int = 0
    no_think: int = 0
    stopped: str | None = None  # the ledger's message when it stopped the run


def find_targets(records: list[dict], index: ReasoningIndex) -> list[Target]:
    """The turns the sidecar does not cover, in (id, k) order (module docstring); ReasoningError for a stale
    row or a row of a turn the records do not have."""
    found, covered = [], set()
    for record in records:
        scripted = is_scripted(record)
        for k, i, answer in turns(record["messages"]):
            if index.lookup(record["id"], k, answer) is None:
                context = build_context(record["messages"][:i])
                found.append(Target(record["id"], k, scripted, context, answer))
            else:
                covered.add((record["id"], k))
    unknown = sorted(set(index.rows) - covered)
    if unknown:
        raise ReasoningError(f"the teacher sidecar has rows of {len(unknown)} turn(s) the records do not "
                             f"have, the first {unknown[0]!r}; rebuild it with scripts/join_reasoning.py")
    return sorted(found, key=lambda t: (t.id, t.k))


def call_names(answer: dict) -> list[str]:
    return [(c.get("function") or {}).get("name") or "" for c in answer.get("tool_calls") or []]


def answer_kind(answer: dict) -> str:
    """One of KINDS: an ask_clarification step, another tool call, or a text answer."""
    names = call_names(answer)
    if not names:
        return "answer"
    return "clarify" if ASK in names else "call"


def in_requested_form(text: str) -> bool:
    """No CJK character, 2-6 sentences, at most 120 words: the form the prompt asks for."""
    sentences = len(_SENTENCE_END.findall(text.strip()))
    return (not _CJK.search(text) and len(text.split()) <= MAX_WORDS
            and MIN_SENTENCES <= sentences <= MAX_SENTENCES)


def problems(text: str, answer: dict, count_tokens: CountTokens,
             finish_reason: str | None = None) -> list[str]:
    """What is wrong with a reasoning written for answer (module docstring, checks); [] when it passes."""
    found = [CUT] if finish_reason == "length" else []
    if not text.strip():
        return [*found, EMPTY]
    if has_think_tag(text):
        found.append(THINK_TAG)
    if count_tokens(text) > MAX_REASONING_TOKENS:
        found.append(TOO_LONG)
    if any(name not in text for name in call_names(answer)):
        found.append(NOT_NAMED)
    return found


def pick_examples(index: ReasoningIndex, records: list[dict], count_tokens: CountTokens) -> list[dict]:
    """The three style examples (module docstring) as {"id", "k", "kind", "reasoning"}, in KINDS order;
    ReasoningError when no teacher row of some kind is eligible."""
    candidates: dict[str, list[tuple[str, dict]]] = {kind: [] for kind in KINDS}
    for record in records:
        for k, _, answer in turns(record["messages"]):
            row = index.rows.get((record["id"], k))
            if row is None or row["source"] != TEACHER:
                continue
            text = index.lookup(record["id"], k, answer)  # checks the row against the answer
            if in_requested_form(text) and not problems(text, answer, count_tokens):
                rank = hashlib.sha256(f"{record['id']}/{k}".encode()).hexdigest()
                kind = answer_kind(answer)
                example = {"id": record["id"], "k": k, "kind": kind, "reasoning": text}
                candidates[kind].append((rank, example))
    picked = []
    for kind in KINDS:
        if not candidates[kind]:
            raise ReasoningError(f"the teacher sidecar has no {kind} example: no teacher row of that kind "
                                 "has the requested form and passes the checks")
        picked.append(min(candidates[kind], key=lambda candidate: candidate[0])[1])
    return picked


def system_message(examples: list[dict]) -> dict:
    """The system message of every prompt: the instructions and the style examples."""
    shown = "".join(f"\n\nExample {n}:\n{e['reasoning']}" for n, e in enumerate(examples, 1))
    return {"role": "system", "content": INSTRUCTIONS + shown}


def cut(text: str) -> str:
    return text if len(text) <= TOOL_OUTPUT_CHARS else text[:TOOL_OUTPUT_CHARS] + "\n[cut]"


def render_answer(answer: dict) -> str:
    """An assistant message as text: its words, then one "call <name> <arguments as written>" line per
    call."""
    lines = [answer["content"]] if (answer.get("content") or "").strip() else []
    for call in answer.get("tool_calls") or []:
        function = call.get("function") or {}
        raw = function.get("arguments")
        lines.append(f"call {function.get('name')} {raw if isinstance(raw, str) else canonical_json(raw)}")
    return "\n".join(lines)


def render_context(context: list[dict]) -> str:
    """The context as text, one block per message; a tool output is cut (cut()) and named by its call."""
    names = {c.get("id"): (c.get("function") or {}).get("name") for m in context
             for c in m.get("tool_calls") or []}
    blocks = []
    for message in context:
        role = message.get("role")
        if role == "tool":
            name = names.get(message.get("tool_call_id")) or "a tool"
            blocks.append(f"[result of {name}]\n{cut(message.get('content') or '')}")
        elif role == "assistant":
            blocks.append(f"[agent]\n{render_answer(message)}")
        else:
            label = "system prompt" if role == "system" else role
            blocks.append(f"[{label}]\n{message.get('content') or ''}")
    return "\n\n".join(blocks)


def nonce(target: Target) -> str:
    """The retry's nonce: random-looking, yet the turn's own, so a rerun sends the same retry."""
    return hashlib.sha256(f"{target.id}/{target.k}".encode()).hexdigest()[:12]


def build_messages(system: dict, target: Target, nonce: str | None = None) -> list[dict]:
    """The prompt of a target: the system message and the user message; a retry's ends with its nonce."""
    text = (f"What the agent saw:\n\n{render_context(target.context)}\n\n"
            f"The agent's reply:\n{render_answer(target.answer)}\n\n{REQUEST}")
    if nonce is not None:
        text += f"\n\n(nonce {nonce})"
    return [system, {"role": "user", "content": text}]


def request_key_of(messages: list[dict]) -> str:
    """The key CloudClient with thinking off files the answer to messages under (llm.cloud, rule 3: the
    request with extra_body THINKING_OFF): the row's cache_key."""
    return request_key({"model": MODEL, "messages": messages, "temperature": TEMPERATURE,
                        "max_tokens": MAX_TOKENS, "extra_body": THINKING_OFF})


def ask(client: CloudClient, messages: list[dict], answer: dict, count_tokens: CountTokens) -> Attempt:
    """One request and its checks; a request the API refuses (HTTP 400) is an attempt with REFUSED."""
    key = request_key_of(messages)
    try:
        data = client.chat(messages, temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
    except openai.BadRequestError:
        return Attempt("", [REFUSED], key)
    choice = data["choices"][0]
    text = ((choice.get("message") or {}).get("content") or "").strip()
    return Attempt(text, problems(text, answer, count_tokens, choice.get("finish_reason")), key)


def rationalize(targets: list[Target], system: dict, client: CloudClient, count_tokens: CountTokens,
                log: Callable[[str], None] = print) -> Outcome:
    """Ask for the reasoning of every target in order (module docstring: checks, retry); the ledger's
    BudgetExceeded (PeakHours too) ends the run where it is."""
    outcome = Outcome()
    for number, target in enumerate(targets, 1):
        try:
            attempt = ask(client, build_messages(system, target), target.answer, count_tokens)
            if attempt.problems:
                outcome.first_problems.update(attempt.problems)
                outcome.retried += 1
                retry = build_messages(system, target, nonce(target))
                attempt = ask(client, retry, target.answer, count_tokens)
        except BudgetExceeded as exc:
            outcome.stopped = str(exc)
            return outcome
        if attempt.problems:
            outcome.failures.update(attempt.problems)
            outcome.no_think += 1
        else:
            outcome.rows.append(sidecar_row(target.id, target.k, target.answer, source=RATIONALIZED,
                                            reasoning=attempt.text, model=MODEL, cache_key=attempt.key,
                                            context_equal=True))
        if number % 100 == 0:
            log(f"{number}/{len(targets)} turns: {len(outcome.rows)} written, {outcome.no_think} without, "
                f"{client.calls} paid, {client.cache_hits} cached")
    return outcome


def target_counts(targets: list[Target]) -> dict:
    scripted = sum(1 for t in targets if t.scripted)
    return {"total": len(targets), "teacher_trajectory": len(targets) - scripted, "scripted": scripted}


def priced(calls: int, prompt_tokens: int, completion_tokens: int) -> dict:
    """Token totals priced with the ledger's arithmetic: the list price and what an off-peak call is booked
    at."""
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
    list_price = cost_rmb(load_prices()[MODEL], usage)
    return {"calls": calls, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "cost_rmb": list_price * load_off_peak().factor, "cost_rmb_list_price": list_price}


def dry_run(targets: list[Target], system: dict, count_tokens: CountTokens) -> dict:
    """The prompts' account and the cost estimate of the dry run (module docstring); no client, cache or
    ledger."""
    system_tokens = count_tokens(system["content"])
    first, retries = [], []
    digest = hashlib.sha256()
    for target in targets:
        messages = build_messages(system, target)
        digest.update((canonical_json(messages) + "\n").encode("utf-8"))
        first.append(system_tokens + count_tokens(messages[1]["content"]))
        retry = build_messages(system, target, nonce(target))
        retries.append(system_tokens + count_tokens(retry[1]["content"]))
    n = len(targets)
    return {"prompts": {"built": n, "sha256": digest.hexdigest(), "tokens": length_stats(first)},
            "estimate": priced(n, sum(first), n * EXPECTED_COMPLETION_TOKENS),
            "ceiling": priced(2 * n, sum(first) + sum(retries), 2 * n * MAX_TOKENS),
            "price": {"model": MODEL, "off_peak_factor": load_off_peak().factor,
                      **asdict(load_prices()[MODEL])}}


def prompts_sha256(targets: list[Target], system: dict) -> str:
    """The SHA-256 of every first prompt, as the dry run reports it: the two runs can be matched."""
    digest = hashlib.sha256()
    for target in targets:
        digest.update((canonical_json(build_messages(system, target)) + "\n").encode("utf-8"))
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                    newline="\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def report_of_other_records(path: Path, records_sha256: str) -> bool:
    """Whether path exists and is not this script's report (of a run or a dry run) of the records file with
    that SHA-256 (module docstring, outputs of other records)."""
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8"))["inputs"]["records"]["sha256"] != records_sha256
    except (ValueError, KeyError, TypeError):  # not JSON, or no such key: not a report of this script
        return True


def sidecar_of_other_records(path: Path, records: list[dict]) -> bool:
    """Whether path exists and holds a row of a turn the records do not have, or no sidecar at all."""
    if not path.exists():
        return False
    keys = {(record["id"], k) for record in records for k, _, _ in turns(record["messages"])}
    try:
        return not set(ReasoningIndex([path]).rows) <= keys
    except (ReasoningError, OSError, UnicodeDecodeError):
        return True


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Write the missing reasoning of the synthetic training turns.")
    ap.add_argument("--records", type=Path, default=RECORDS, help=f"the training file (default {RECORDS})")
    ap.add_argument("--teacher", type=Path, default=TEACHER_SIDECAR,
                    help=f"the teacher sidecar of scripts/join_reasoning.py (default {TEACHER_SIDECAR})")
    ap.add_argument("--out", type=Path, default=OUT, help=f"the rationalized sidecar (default {OUT})")
    ap.add_argument("--report", type=Path, default=REPORT, help=f"the run's report (default {REPORT})")
    ap.add_argument("--dry-run-report", type=Path, default=DRY_RUN_REPORT,
                    help=f"the dry run's report (default {DRY_RUN_REPORT})")
    ap.add_argument("--dry-run", action="store_true",
                    help="build every prompt and estimate the cost; no API call, no cache, no ledger")
    ap.add_argument("--max-cost", type=float, default=MAX_COST, metavar="RMB",
                    help=f"what this invocation may add to the stage's spending (default {MAX_COST})")
    ap.add_argument("--stage", default=STAGE, choices=sorted(STAGE_CAPS),
                    help=f"the stage the spending is booked under (default {STAGE})")
    ap.add_argument("--tokenizer", type=Path, default=TOKENIZER,
                    help=f"the model folder whose tokenizer counts tokens (default {TOKENIZER})")
    ap.add_argument("--llm-dir", type=Path, default=LLM_DIR, help=argparse.SUPPRESS)
    return ap


def main(argv: list[str] | None = None, *, openai_client: object | None = None,
         clock: Callable[[], datetime] | None = None, count_tokens: CountTokens | None = None) -> int:
    """openai_client stands in for openai.OpenAI in tests (None reads the key from the environment), clock for
    the ledger's clock (the peak hours), count_tokens for the tokenizer (None reads --tokenizer)."""
    args = build_parser().parse_args(argv)
    missing = [str(p) for p in (args.records, args.teacher) if not p.is_file()]
    if count_tokens is None and not (args.tokenizer / "tokenizer.json").is_file():
        missing.append(str(args.tokenizer / "tokenizer.json"))
    if missing:
        print(f"missing input: {', '.join(missing)}", file=sys.stderr)
        return 2
    count = count_tokens or token_counter(args.tokenizer)
    try:
        records = load_records(args.records)
        if not all(is_agent_history(r) for r in records):
            raise RecordError(f"{args.records} holds a record that is not an agent history")
        refuse_test_material(records, args.records)
        index = ReasoningIndex([args.teacher])
        targets = find_targets(records, index)
        examples = pick_examples(index, records, count)
    except (ReasoningError, RecordError, RefusedTestSet) as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    records_sha256 = sha256_file(args.records)
    if args.dry_run:
        checks = [(args.dry_run_report, report_of_other_records(args.dry_run_report, records_sha256))]
    else:
        checks = [(args.out, sidecar_of_other_records(args.out, records)),
                  (args.report, report_of_other_records(args.report, records_sha256))]
    foreign = [path for path, other in checks if other]
    if foreign:
        print(f"{', '.join(str(p) for p in foreign)} was written for another records file than "
              f"{args.records}; name this pass's own outputs (--out, --report, --dry-run-report)",
              file=sys.stderr)
        return 2
    system = system_message(examples)
    common = {"script": "scripts/rationalize_reasoning.py",
              "inputs": {"records": {"path": args.records.as_posix(), "sha256": sha256_file(args.records)},
                         "teacher": {"path": args.teacher.as_posix(), "sha256": sha256_file(args.teacher),
                                     "rows": len(index)},
                         "tokenizer": None if count_tokens else args.tokenizer.as_posix()},
              "examples": [{key: e[key] for key in ("id", "k", "kind")} for e in examples],
              "targets": target_counts(targets), "stage": args.stage, "max_cost_rmb": args.max_cost}
    if args.dry_run:
        report = {**common, "dry_run": True, **dry_run(targets, system, count)}
        write_json(args.dry_run_report, report)
        print(f"dry run: {len(targets)} turns ({report['targets']['scripted']} scripted), "
              f"{report['prompts']['tokens'].get('total', 0)} prompt tokens; estimate "
              f"{report['estimate']['cost_rmb']:.2f} RMB off peak, ceiling "
              f"{report['ceiling']['cost_rmb']:.2f} RMB, --max-cost {args.max_cost}; report -> "
              f"{args.dry_run_report.as_posix()}")
        return 0
    cap = stage_cap(args.llm_dir / "budget.sqlite", args.stage, args.max_cost)
    budget = Budget(args.llm_dir / "budget.sqlite", stage=args.stage, stage_cap_rmb=cap,
                    off_peak=load_off_peak(), clock=clock)
    cache = ResponseCache(args.llm_dir / "cache.sqlite")
    try:
        spent_before = budget.spent(args.stage)
        try:
            client = CloudClient(MODEL, cache=cache, budget=budget, client=openai_client, thinking="off",
                                 off_peak_only=True)
        except MissingKeyError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        print(f"{len(targets)} turns to write, {args.stage} spent {spent_before:.4f} RMB, cap {cap:.4f} RMB",
              flush=True)
        outcome = rationalize(targets, system, client, count, log=lambda line: print(line, flush=True))
        spent_after = budget.spent(args.stage)
    finally:
        budget.close()
        cache.close()
    complete = outcome.stopped is None
    output = None
    if complete:
        sha, size = write_sidecar(outcome.rows, args.out)
        output = {"path": args.out.as_posix(), "rows": len(outcome.rows), "sha256": sha, "bytes": size}
    report = {**common, "dry_run": False, "prompts_sha256": prompts_sha256(targets, system),
              "client": {"model": MODEL, "thinking": "off", "off_peak_only": True, "temperature": TEMPERATURE,
                         "max_tokens": MAX_TOKENS},
              "cap_rmb": cap, "spent_before_rmb": spent_before, "spent_after_rmb": spent_after,
              "calls": {"paid": client.calls, "cached": client.cache_hits}, "retried": outcome.retried,
              "first_attempt_problems": dict(sorted(outcome.first_problems.items())),
              "written": len(outcome.rows), "no_think": outcome.no_think,
              "failures": dict(sorted(outcome.failures.items())), "complete": complete,
              "stopped": outcome.stopped,
              "lengths": {"characters": length_stats([len(r["reasoning"]) for r in outcome.rows]),
                          "tokens": length_stats([count(r["reasoning"]) for r in outcome.rows])},
              "output": output}
    write_json(args.report, report)
    if not complete:
        print(f"stopped: {outcome.stopped}; no rows written, a rerun goes on from the cache "
              f"(report -> {args.report.as_posix()})", file=sys.stderr)
        return 3
    print(f"{len(outcome.rows)} of {len(targets)} turns written -> {args.out.as_posix()}, {outcome.no_think} "
          f"left without ({dict(outcome.failures)}); {client.calls} paid, {client.cache_hits} cached, "
          f"{spent_after - spent_before:.4f} RMB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

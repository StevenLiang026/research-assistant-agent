"""Replay a base model on missing-parameter and irrelevant prompts, keep its answers in the fixed sentence
forms (docs/m6-计划.md §3.4 D2, WP3); for FT3 the bake-off winner, of any registered family, replays itself
in its native mode (docs/m7-计划.md WP-J).

Why. FT's ACEBench special group fell from 19-27% to 3% because FT neither uses the required sentences for
missing parameters and irrelevant requests nor holds back the call, where B0 does both (§2.5). These samples
anchor that slice to B0's own behaviour: the base model (Qwen3-1.7B, lora scale 0, greedy, thinking off)
answers prompts whose system message lists the functions as text and asks for one of two fixed sentences,
and only answers in the right sentence are kept, as training samples. FT3 trains another base, the
bake-off's winner, so its replay is that model's own, in the mode it answers in, and replaces the 1.7B
replay (docs/m7-计划.md WP-J).

Prompts. research_agent.data.public_pool reads glaive-v2 and ToolACE from the local Hugging Face cache and
leaves out the records drawn into data/raw/public/; records of --exclude (the format-mix file by default,
when it exists) are left out too, by (source, source_id).
- incomplete: glaive-v2 conversations whose first assistant turn makes no call and asks the user for
  something (a question mark, or "please provide/tell/specify/share/let me know") and a call follows later,
  so the first request lacked an argument; the prompt is the first user message, the gold function the
  first one called.
- irrelevant: every ToolACE record without a call whose first assistant turn says no listed function
  applies (IRRELEVANT_GOLD, with answers about lacking or missing parameters excluded), then glaive-v2
  conversations that list functions and never call one; the prompt is the first user message.
Half of --limit goes to each kind, a short kind passing its spare to the other; records are drawn with
random.Random(f"{seed}/replay"), the kinds interleaved, and half of the prompts get the Chinese system
message. The system message (own wording, Chinese and English variants) lists each function's JSON schema on
one line and asks for MISSING_TEMPLATE, filled in, when a required parameter is missing and for
IRRELEVANT_SENTENCE when no function fits. Both sentences are English in both variants. IRRELEVANT_SENTENCE
has no placeholder: there is nothing to fill in.

Generation and filter (the text path, M6). Each prompt is rendered with the model's chat template
(prompt_renderer, thinking off, no tools block) and completed by llama-server through LocalClient with
lora_scale 0.0, temperature 0, no prompt cache, stopping at the end of the turn. The server must report one
slot (/props total_slots 1, `-np 1`): with several slots the greedy output depends on the request history. An
answer is kept when, stripped, it is exactly IRRELEVANT_SENTENCE for an irrelevant prompt, or
MISSING_TEMPLATE for an incomplete prompt with a listed function and only parameters that its schema defines,
the names joined however the model joined them (commas, "and", the singular wording for one name:
_MISSING_FORM); the sample then carries the template's own form, the names comma-separated
(canonical_answer; the manifest counts the answers normalised that way). Generation stops at --cap kept
answers, and with --cap-per-kind a kind is skipped once it has that many, so that the kind the model obliges
less often (incomplete, in the run of 2026-10-17) is not crowded out by the other.
A kept sample is the system message, the user message and the answer, in research_agent.data.public's
sample format with "tools": [] (the functions are in the system text) plus "kind" and "instruction_lang".

The family path (M7, docs/m7-家族层设计.md §7). The model folder's family (research_agent.family:
config.json's architectures and the chat template's hash, read from the folder's files; --family must name it
when given; a template no family pins is refused) decides the path. The Qwen3 hybrid family in no_think
mode, M6's, keeps the text path, its requests byte for byte, unless --family-path; every other family, and the
hybrid family in think mode, replays through the family (load_family): a prompt is the family's prompt of its
messages in the mode (to_template_messages, prompt), rendered with the kept sample's own tools, none, so that
it is the prompt the sample trains under, and goes to the server as its ids (encode_prompt;
research_agent.llm.generation.generate through LocalClient, at lora scale 0.0, with the family's stop
strings and no prompt cache). The mode is --mode, or the family's inference mode, its native one: no_think
for a family that never thinks and for the Qwen3 hybrid family, think for every other family that thinks.
A hybrid winner replays in the mode the plan's rule gives it (docs/m7-计划.md §4: think, unless its no_think
greedy beat its think mean by 5 or more), and --mode applies the rule where the inference mode differs. The
Qwen3 hybrid family's no_think is M6's mode and C0's (Qwen3-4B, trained no_think): a Qwen3 hybrid winner
replays no_think only when it is C0 or the rule chose no_think for it, and is otherwise run with --mode
think; a winner of another hybrid family that the rule switched to no_think is run with --mode no_think.
Each mode is decoded with the family's settings for it:
- no_think: one request of at most --max-tokens tokens (greedy for a family that never thinks and for a
  hybrid one, as on the text path; sampled for a family that always thinks);
- think: generate's two phases, at most the reasoning budget R (--reasoning-budget, the family's by
  default) of reasoning, closed by force (the family's think_close) when the model has not closed it by
  then, and at most --max-tokens answer tokens, under the vendor's sampling.
A sampled request carries the seed generation.request_seed derives from --seed (also the draw's seed), the
prompt's ids, the phase and sample 0, so a rerun with the same --seed on a one-slot server repeats every
request and every output byte for byte. The filter is the text path's (check_answer, canonical_answer) on the
answer alone (Generation.answer_text: what follows the close), never on the reasoning: a fixed sentence in
the reasoning keeps nothing. Before the first request the server must report one slot and a context (/props
default_generation_settings.n_ctx) that holds the longest generation a prompt can reach: its ids, in think
mode R and the close's ids, and --max-tokens.

The reasoning sidecar (think mode; research_agent.data.reasoning, what scripts/train.py --reasoning reads). A
kept sample whose generation holds a usable reasoning gets one row, so that train.py trains it as a THINK
sample with that reasoning:
- id: the name train.py gives a record without an "id" (train.samples.record_name), the output's stem and
  the sample's index in it, "<stem>:<index>"; k 0, the sample's one assistant turn. So the rows hold for the
  output under the name they were written with: a copy under another name trains NO_THINK, and a sidecar
  left from another run of the same output name is stale where the answer hash cannot tell, every irrelevant
  sample's answer being the same sentence (the manifest pairs the two files by SHA-256);
- answer_sha256: of the kept sample's answer, its canonical form;
- source "teacher" (the model's own reasoning, written with its answer: the replayed model is its own
  teacher); model, the GGUF the server runs (/props model_path, else the model folder's name);
  context_equal true (the replay's prompt is the sample's);
- cache_key: no response cache holds a local request; request_key of the model, the prompt's ids and the
  generation settings, which name the request and its seeds;
- reasoning: Generation.reasoning. One closed by force ends where the budget cut it, and a character cut
  there, which the server writes as U+FFFD, is left out with the newlines before it (Generation.reasoning
  holds none at either end).
A kept sample gets no row, and trains NO_THINK (its prompt holding the empty block), when its reasoning is
empty, holds a think tag (a sidecar takes neither), or makes a THINK sample the training path refuses
(train.samples.render_family_sample, which train.py renders the sample with: a reasoning that does not split
from the prompt at a token boundary, say); the manifest counts each. Keep the sidecar out of a folder whose
files train.py takes as data: a row is no record.

Written: <out stem>.prompts.jsonl (the prompts, also by --dry-run, which needs no server and no model folder
and writes nothing else); --out (the kept samples); --reasoning-out (the sidecar: required for a think
replay, refused for a no_think one); --manifest: prompt and kept counts per kind and language, rejections by
reason, the server's slots, model file name and build, the output's SHA-256; on the family path also the
family's record, the server's context, the generation (the path, the mode, the decoding, R, the answer
budget, the base seed), the modes the kept samples train in and, in think mode, the forced closes (generated
and kept), the kept samples without a row by reason, the reasoning lengths in tokens
(Generation.reasoning_tokens: the generated ids before the answer, the close's among them, of every
generation and of the kept ones) and the sidecar's path, rows and SHA-256. The manifest names a file (the
exclusion file, the prompts, the output, the sidecar) by its path relative to the current folder, the
repository root, as spelled: lexically, never resolved, so that a file under a junction or a symbolic link
(data/raw/ of a worktree whose data/raw links into the main tree) keeps its data/raw/... path; a file outside
the current folder by its name alone. Existing outputs are overwritten only with --force.

For FT3 keep M6's draw (--limit 3000 --seed 42 and the default exclusion): the prompts file is then
data/raw/public/base-replay.prompts.jsonl byte for byte, the prompts whose records scripts/build_ace_probe.py
left out of the ACE probe. Give a hybrid winner the mode the plan's rule picks (The family path: a Qwen3
hybrid winner other than C0 needs --mode think unless the rule chose no_think).

    uv run --no-sync python scripts/replay_base.py --server http://127.0.0.1:8080 \\
        --out data/raw/public/base-replay.jsonl --manifest data/public/base-replay-manifest.json \\
        [--model models/Qwen3-1.7B] [--family NAME] [--mode think|no_think] [--family-path] \\
        [--reasoning-out FILE] [--reasoning-budget N] [--limit 1000] [--seed 42] [--cap 500] \\
        [--cap-per-kind N] [--force] [--dry-run]

Exit 0 done; 2 usage (a model folder of no registered family, or of another than --family; a mode the family
lacks; a think replay without --reasoning-out, a no_think one with it or with --reasoning-budget; a sidecar
that would overwrite another output), an existing output without --force, no server, a server with more
than one slot or, on the family path, without the context the longest generation needs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from research_agent.agent.models import STOP
from research_agent.agent.types import api_to_hf
from research_agent.data import public_pool
from research_agent.data.public import has_tool_call
from research_agent.data.reasoning import TEACHER, has_think_tag, length_stats, sidecar_row, write_sidecar
from research_agent.family.core import Family, FamilyError, LoadedFamily, Mode, folder_family, load_family
from research_agent.family.registry import QWEN3_HYBRID
from research_agent.llm import generation
from research_agent.llm.cache import request_key
from research_agent.llm.generation import Generation, GenerationSettings
from research_agent.llm.local import LocalClient
from research_agent.train.samples import render_family_sample

SEED = 42
LIMIT = 1000
CAP = 500
MAX_TOKENS = 128
MODEL = "models/Qwen3-1.7B"
EXCLUDE = Path("data/raw/public/format-mix.jsonl")
KINDS = ("incomplete", "irrelevant")
SLOTS = 1
TEXT_FAMILY = QWEN3_HYBRID.name  # the family whose no_think replay keeps the text path (module docstring)
CUT = "�"  # what the server writes for the bytes of a character the budget cut
WITHOUT = ("empty", "think tag", "refused")  # why a kept sample of a think replay has no row
PROGRESS = 50  # the family path reports its counts every this many generated prompts

# The two fixed sentences (own wording, English in both system-message languages).
MISSING_TEMPLATE = "Cannot call {function} yet: the required parameters {params} were not given."
IRRELEVANT_SENTENCE = "No call made: this request lies outside the scope of the listed functions."

_ASKS = re.compile(
    r"\b(?:please|could you|can you|would you)\s+(?:provide|tell|specify|share|let me know)", re.IGNORECASE
)
IRRELEVANT_GOLD = re.compile(
    r"\bnone of the (?:\w+ ){0,2}(?:functions|tools|apis)\b"
    r"|\b(?:given|provided|available|listed|above) (?:functions|tools|apis) (?:do|does|can|could|are) ?not\b"
    r"|\bno (?:suitable|relevant|appropriate|available|applicable) (?:function|tool|api)s?\b",
    re.IGNORECASE,
)
_PARAMETER_TALK = re.compile(
    r"\b(?:lacks?|missing|insufficient|(?:required|necessary) "
    r"(?:argument|parameter)s?)\b",
    re.IGNORECASE,
)

_SYSTEM = {
    "en": (
        "You can use the functions listed below; each line is one function's JSON schema.\n{tools}\n\n"
        "If the user's request fits one of these functions and every required parameter is given, call "
        "the function. If the request fits a function but a required parameter is not given, do not call "
        "it and do not guess the value; reply with exactly this sentence, putting in the names of the "
        "missing parameters (comma-separated) and the function name:\n{missing}\n"
        "If none of the listed functions fits the request, do not call any function; reply with exactly "
        "this sentence:\n{irrelevant}"
    ),
    "zh": (
        "你可以使用下列函数,每行是一个函数的 JSON schema。\n{tools}\n\n"
        "如果用户的请求适合其中某个函数且必需参数齐全,就调用该函数。如果请求适合某个函数但缺少必需参数,"
        "不要调用,也不要猜测参数值,而是原样回复下面这句英文,填入缺少的参数名(用逗号分隔)和函数名:\n"
        "{missing}\n如果列出的函数都不适合该请求,不要调用任何函数,原样回复下面这句英文:\n{irrelevant}"
    ),
}


def system_prompt(tools: list[dict], lang: str) -> str:
    """The system message: the functions as text (one JSON schema per line) and the two sentence rules."""
    listed = "\n".join(json.dumps(t["function"], ensure_ascii=False) for t in tools)
    text = _SYSTEM[lang]
    for key, value in (
        ("{tools}", listed),
        ("{missing}", MISSING_TEMPLATE),
        ("{irrelevant}", IRRELEVANT_SENTENCE),
    ):
        head, _, tail = text.partition(key)
        text = head + value + tail
    return text


# MISSING_TEMPLATE as the base model also writes it: the names joined by commas, "and" or both, and the
# singular wording when one name is missing (seen in the run of 2026-10-17); the sample keeps the template
_MISSING_FORM = re.compile(r"Cannot call (?P<function>\S+) yet: the required parameters? (?P<params>.+?) "
                           r"(?:were|was) not given\.")


def parse_missing(text: str) -> tuple[str, list[str]] | None:
    """(function, parameter names) of a MISSING_TEMPLATE sentence, the names however they were joined
    (_MISSING_FORM); None when text is not one."""
    match = _MISSING_FORM.fullmatch(text.strip())
    if match is None:
        return None
    params = [p for p in re.split(r"\s*,\s*(?:and\s+)?|\s+and\s+", match.group("params")) if p]
    if not params or any(re.search(r"\s", p) for p in params):
        return None
    return match.group("function").strip(), params


def _first_assistant(sample: dict) -> dict | None:
    return next((m for m in sample["messages"] if m["role"] == "assistant"), None)


def asks_for_information(sample: dict) -> bool:
    """A conversation that lists functions, whose first assistant turn asks the user for something without
    calling, and that calls a function later."""
    first = _first_assistant(sample)
    if not sample.get("tools") or first is None or first.get("tool_calls") or not has_tool_call(sample):
        return False
    return "?" in first["content"] or bool(_ASKS.search(first["content"]))


def is_irrelevant_gold(text: str) -> bool:
    """A gold answer saying that no listed function applies (and not that parameters are missing)."""
    return bool(IRRELEVANT_GOLD.search(text)) and not _PARAMETER_TALK.search(text)


def _gold_function(sample: dict) -> str:
    return next(c["function"]["name"] for m in sample["messages"] for c in m.get("tool_calls") or [])


def _no_call_irrelevant(sample: dict, toolace: bool) -> bool:
    if not sample.get("tools") or has_tool_call(sample):
        return False
    first = _first_assistant(sample)
    return first is not None and (not toolace or is_irrelevant_gold(first["content"]))


def build_prompts(
    pools: dict[str, list[dict]], *, limit: int, seed: int, exclude: set[tuple[str, str]]
) -> tuple[list[dict], dict]:
    """The prompts (module docstring) and an account of the candidates and the draw."""

    def kept(samples: list[dict]) -> list[dict]:
        return [s for s in samples if (s["source"], s["source_id"]) not in exclude]

    glaive, toolace = kept(pools.get("glaive_v2", [])), kept(pools.get("toolace", []))
    incomplete = [s for s in glaive if asks_for_information(s)]
    toolace_irrelevant = [s for s in toolace if _no_call_irrelevant(s, toolace=True)]
    glaive_irrelevant = [s for s in glaive if _no_call_irrelevant(s, toolace=False)]
    n_inc = min(len(incomplete), limit // 2)
    n_irr = min(len(toolace_irrelevant) + len(glaive_irrelevant), limit - limit // 2)
    n_inc += min(limit - n_inc - n_irr, len(incomplete) - n_inc)
    n_irr += min(limit - n_inc - n_irr, len(toolace_irrelevant) + len(glaive_irrelevant) - n_irr)
    rng = random.Random(f"{seed}/replay")
    inc = rng.sample(incomplete, n_inc)
    shuffled = rng.sample(toolace_irrelevant, len(toolace_irrelevant))
    irr = shuffled[:n_irr] + rng.sample(glaive_irrelevant, max(0, n_irr - len(shuffled)))
    order: list[tuple[str, dict]] = []
    for i in range(max(len(inc), len(irr))):
        order += [("incomplete", inc[i])] if i < len(inc) else []
        order += [("irrelevant", irr[i])] if i < len(irr) else []
    zh = set(rng.sample(range(len(order)), len(order) // 2))
    prompts = []
    for i, (kind, s) in enumerate(order):
        lang = "zh" if i in zh else "en"
        user = next(m for m in s["messages"] if m["role"] == "user")
        prompts.append(
            {
                "kind": kind,
                "source": s["source"],
                "source_id": s["source_id"],
                "instruction_lang": lang,
                "gold_function": _gold_function(s) if kind == "incomplete" else None,
                "tools": s["tools"],
                "messages": [
                    {"role": "system", "content": system_prompt(s["tools"], lang)},
                    {"role": "user", "content": user["content"]},
                ],
            }
        )
    account = {
        "candidates": {
            "incomplete": len(incomplete),
            "irrelevant": len(toolace_irrelevant) + len(glaive_irrelevant),
        },
        "irrelevant_candidates": {"toolace": len(toolace_irrelevant), "glaive_v2": len(glaive_irrelevant)},
        "prompts": {kind: sum(p["kind"] == kind for p in prompts) for kind in KINDS},
        "irrelevant_prompts": {
            src: sum(p["kind"] == "irrelevant" and p["source"] == src for p in prompts)
            for src in ("toolace", "glaive_v2")
        },
        "languages": {lang: sum(p["instruction_lang"] == lang for p in prompts) for lang in ("en", "zh")},
    }
    return prompts, account


def check_answer(prompt: dict, text: str) -> str | None:
    """None when the answer is kept; else why not ("form: ...", "function not listed: ...",
    "parameter not in the schema: ...")."""
    text = text.strip()
    if prompt["kind"] == "irrelevant":
        return None if text == IRRELEVANT_SENTENCE else "form: not the irrelevant sentence"
    parsed = parse_missing(text)
    if parsed is None:
        return "form: not the missing-parameter sentence"
    function, params = parsed
    tool = next((t["function"] for t in prompt["tools"] if t["function"]["name"] == function), None)
    if tool is None:
        return f"function not listed: {function}"
    properties = (tool.get("parameters") or {}).get("properties") or {}
    unknown = [p for p in params if p not in properties]
    if unknown:
        return f"parameter not in the schema: {', '.join(unknown)} of {function}"
    return None


def canonical_answer(prompt: dict, text: str) -> str | None:
    """The answer as the sample writes it, None when check_answer rejects it: IRRELEVANT_SENTENCE, or
    MISSING_TEMPLATE with the names comma-separated, whatever joining the model used."""
    if check_answer(prompt, text) is not None:
        return None
    if prompt["kind"] == "irrelevant":
        return IRRELEVANT_SENTENCE
    function, params = parse_missing(text)
    return MISSING_TEMPLATE.format(function=function, params=", ".join(params))


def to_sample(prompt: dict, answer: str) -> dict:
    """A kept answer as a sample in research_agent.data.public's format, the functions in the system text."""
    return {
        "messages": [*prompt["messages"], {"role": "assistant", "content": answer.strip()}],
        "tools": [],
        "source": prompt["source"],
        "source_id": prompt["source_id"],
        "kind": prompt["kind"],
        "instruction_lang": prompt["instruction_lang"],
    }


def generate(
    prompts: list[dict], client, render: Callable, *, cap: int, max_tokens: int,
    cap_per_kind: int | None = None,
) -> tuple[list[dict], Counter, int, int]:
    """(kept samples, rejections by reason, prompts completed, answers normalised); stops at cap kept
    samples, and with cap_per_kind skips a kind once it has that many (so that the kind the model obliges
    less often is not crowded out) and stops when both kinds have."""
    kept: list[dict] = []
    rejected: Counter = Counter()
    by_kind: Counter = Counter()
    done = normalised = 0
    for prompt in prompts:
        if len(kept) >= cap:
            break
        if cap_per_kind is not None:
            if all(by_kind[kind] >= cap_per_kind for kind in ("incomplete", "irrelevant")):
                break
            if by_kind[prompt["kind"]] >= cap_per_kind:
                continue
        completion = client.complete(
            render(api_to_hf(prompt["messages"]), None),
            lora_scale=0.0,
            temperature=0.0,
            logprobs=None,
            stop=list(STOP),
            max_tokens=max_tokens,
        )
        done += 1
        reason = check_answer(prompt, completion.text)
        if reason is None:
            answer = canonical_answer(prompt, completion.text)
            normalised += answer != completion.text.strip()
            kept.append(to_sample(prompt, answer))
            by_kind[prompt["kind"]] += 1
        else:
            rejected[f"{prompt['kind']}: {reason.split(':')[0]}"] += 1
    return kept, rejected, done, normalised


# ---- the family path (module docstring) ---------------------------------------------------------------


class ReplayPlan(NamedTuple):
    """How a run generates (module docstring: The family path): the model folder's family, the mode, whether
    it takes M6's text path, and the family path's generation settings (None on the text path)."""

    family: Family
    mode: Mode
    text: bool
    settings: GenerationSettings | None


def replay_plan(args: argparse.Namespace) -> ReplayPlan:
    """The plan of a run that generates, from the model folder's files alone (folder_family) and the options;
    ValueError (a FamilyError among them) naming what is wrong with the folder or the options (module
    docstring: Exit)."""
    spec = folder_family(args.model, name=args.family)
    mode = spec.inference_mode if args.mode is None else Mode(args.mode)
    if mode not in spec.modes:
        raise ValueError(f"{spec.name} has no {mode} mode (its modes: {', '.join(spec.modes)})")
    if mode is Mode.THINK:
        if args.reasoning_out is None:
            raise ValueError(f"a think replay ({spec.name}) writes the reasoning of its kept answers to a "
                             "sidecar: pass --reasoning-out FILE")
        sidecar = Path(args.reasoning_out).resolve()
        for option, path in (("--out", Path(args.out)), ("--manifest", Path(args.manifest)),
                             ("the prompts file", prompts_path(Path(args.out)))):
            if sidecar == path.resolve():
                raise ValueError(f"--reasoning-out {args.reasoning_out} is the same file as {option}: the "
                                 "sidecar would overwrite it")
    else:
        for option, value in (("--reasoning-out", args.reasoning_out),
                              ("--reasoning-budget", args.reasoning_budget)):
            if value is not None:
                raise ValueError(f"{option} is for a think replay; this one is {mode} ({spec.name})")
    if spec.name == TEXT_FAMILY and mode is Mode.NO_THINK and not args.family_path:
        return ReplayPlan(spec, mode, True, None)
    budget = spec.reasoning_budget if args.reasoning_budget is None else args.reasoning_budget
    settings = GenerationSettings(mode=mode, decoding=spec.decoding[mode], reasoning_budget=budget,
                                  answer_tokens=args.max_tokens, base_seed=args.seed)
    return ReplayPlan(spec, mode, False, settings)


@dataclass(frozen=True)
class Replayed:
    """A kept answer of the family path: its sample, its prompt's ids, whether its reasoning was closed by
    force and its length (Generation.reasoning_tokens), and the reasoning its sidecar row holds (None: no row,
    the sample trains NO_THINK; why says why in think mode, one of WITHOUT)."""

    sample: dict
    prompt_ids: list[int]
    forced: bool
    reasoning_tokens: int
    reasoning: str | None
    why: str | None


@dataclass
class FamilyRun:
    """What the family path generated: the kept answers in order, the rejections by reason, the prompts
    generated, how many of them were closed by force and their reasoning lengths in order, and the answers
    normalised (canonical_answer). The generations themselves are not kept: each holds the server's last
    response, every token's log-probabilities among them, which a run of thousands of thinking prompts
    cannot hold in memory."""

    kept: list[Replayed] = field(default_factory=list)
    rejected: Counter = field(default_factory=Counter)
    generated: int = 0
    forced: int = 0
    reasoning_tokens: list[int] = field(default_factory=list)
    normalised: int = 0


def row_reasoning(family: LoadedFamily, sample: dict, gen: Generation) -> tuple[str | None, str | None]:
    """(the reasoning of the kept sample's sidecar row, None), or (None, why it gets no row): "empty",
    "think tag", or "refused" when the training path refuses the THINK sample the row would make
    (render_family_sample, as train.py renders the sample with it). A reasoning closed by force loses a
    character the budget cut (CUT) and the newlines before it. Module docstring: The reasoning sidecar."""
    text = gen.reasoning or ""
    if gen.forced:
        text = text.rstrip(CUT).strip("\n")
    if not text.strip():
        return None, "empty"
    if has_think_tag(text):
        return None, "think tag"
    *context, target = sample["messages"]
    try:
        render_family_sample(family, context, target, tools=sample["tools"], reasoning=text,
                             max_length=sys.maxsize)
    except FamilyError:
        return None, "refused"
    return text, None


def generate_family(prompts: list[dict], prompt_ids: list[list[int]], client, family: LoadedFamily,
                    settings: GenerationSettings, *, cap: int, cap_per_kind: int | None = None,
                    report: Callable[[FamilyRun], None] | None = None) -> FamilyRun:
    """The family path's generation and filter (module docstring): each prompt's ids generated under settings
    at lora scale 0 (generation.generate), the answer alone filtered as generate filters a completion, with
    generate's caps; report, when given, is called with the run every PROGRESS generated prompts."""
    run = FamilyRun()
    by_kind: Counter = Counter()
    for prompt, ids in zip(prompts, prompt_ids, strict=True):
        if len(run.kept) >= cap:
            break
        if cap_per_kind is not None:
            if all(by_kind[kind] >= cap_per_kind for kind in KINDS):
                break
            if by_kind[prompt["kind"]] >= cap_per_kind:
                continue
        gen = generation.generate(client, family, ids, settings, lora_scale=0.0)
        run.generated += 1
        run.forced += gen.forced
        run.reasoning_tokens.append(gen.reasoning_tokens)
        reason = check_answer(prompt, gen.answer_text)
        if reason is None:
            answer = canonical_answer(prompt, gen.answer_text)
            run.normalised += answer != gen.answer_text.strip()
            sample = to_sample(prompt, answer)
            reasoning, why = (row_reasoning(family, sample, gen) if settings.mode is Mode.THINK
                              else (None, None))
            run.kept.append(Replayed(sample, ids, gen.forced, gen.reasoning_tokens, reasoning, why))
            by_kind[prompt["kind"]] += 1
        else:
            run.rejected[f"{prompt['kind']}: {reason.split(':')[0]}"] += 1
        if report is not None and run.generated % PROGRESS == 0:
            report(run)
    return run


def sidecar_rows(kept: list[Replayed], *, stem: str, model: str, settings: GenerationSettings) -> list[dict]:
    """The sidecar rows of the kept answers that have a reasoning, each named by its sample's index in the
    output (module docstring: The reasoning sidecar)."""
    return [sidecar_row(f"{stem}:{index}", 0, item.sample["messages"][-1], source=TEACHER,
                        reasoning=item.reasoning, model=model,
                        cache_key=request_key({"model": model, "prompt_ids": item.prompt_ids,
                                               "generation": settings.record()}),
                        context_equal=True)
            for index, item in enumerate(kept) if item.reasoning is not None]


def thinking_account(run: FamilyRun) -> dict:
    """What a think replay's manifest records of the thinking: forced closes, kept answers without a row by
    reason, and the reasoning lengths in tokens (module docstring: Written)."""
    return {
        "forced": {"generated": run.forced, "kept": sum(item.forced for item in run.kept)},
        "without_reasoning": {why: sum(item.why == why for item in run.kept) for why in WITHOUT},
        "reasoning_tokens": {"generated": length_stats(run.reasoning_tokens),
                             "kept": length_stats([item.reasoning_tokens for item in run.kept])},
    }


def report_progress(run: FamilyRun) -> None:
    print(f"generated {run.generated}, kept {len(run.kept)}, rejected {sum(run.rejected.values())}, "
          f"closed by force {run.forced}", flush=True)


# ---- CLI ----------------------------------------------------------------------------------------------


def load_candidates(manifest: str | Path, raw_dir: str | Path) -> dict[str, list[dict]]:
    entries = public_pool.load_manifest(manifest)["sources"]
    return {
        name: public_pool.load_unused(name, entries[name], raw_dir)[0] for name in ("glaive_v2", "toolace")
    }


def make_renderer(model: str | Path) -> Callable:
    """The training prompt renderer with the model folder's chat template (thinking off)."""
    from transformers import AutoTokenizer  # heavy import, only for a run against a server

    from research_agent.train.samples import prompt_renderer

    return prompt_renderer(AutoTokenizer.from_pretrained(model))


def prompts_path(out: Path) -> Path:
    return out.with_name(f"{out.stem}.prompts.jsonl")


def excluded_ids(path: Path) -> set[tuple[str, str]]:
    if not path.is_file():
        return set()
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {(r["source"], r["source_id"]) for r in rows}


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode("utf-8")


def repo_relative(path: Path) -> str:
    """path relative to the current folder (the repository root) in POSIX form, taken as spelled: lexically
    (os.path.abspath, as research_agent.eval.bakeoff.rel takes it), never resolved, so that a path through a
    junction or a symbolic link (data/raw/ of a worktree whose data/raw links into the main tree) keeps its
    spelling under the current folder; its file name alone when it lies outside the current folder."""
    try:
        return Path(os.path.abspath(path)).relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.name


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=LIMIT, help="prompts to build")
    ap.add_argument("--cap", type=int, default=CAP, help="kept answers at most")
    ap.add_argument("--cap-per-kind", type=int, default=None,
                    help="kept answers at most per kind; a kind that has them is skipped from then on")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="the draw's seed, and the base seed of a sampled generation's request seeds")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="write the prompts only; no server, no model folder")
    ap.add_argument("--model", default=MODEL,
                    help="model folder whose family (its chat template) renders the prompts")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS, help="answer tokens at most")
    ap.add_argument("--exclude", default=str(EXCLUDE), help="a JSONL file whose (source, source_id) to skip")
    ap.add_argument("--source-manifest", default=str(public_pool.MANIFEST))
    ap.add_argument("--raw-dir", default=str(public_pool.RAW_DIR))
    ap.add_argument("--family", default=None,
                    help="the registered family the model folder must resolve to (checked, never assumed)")
    ap.add_argument("--mode", choices=[str(mode) for mode in Mode], default=None,
                    help="the generation mode (default: the family's inference mode, its native one; a "
                         "hybrid winner takes the mode the plan's rule picks: module docstring)")
    ap.add_argument("--family-path", action="store_true",
                    help="the Qwen3 hybrid family's no_think replay through the family, the prompts sent as "
                         "ids, instead of M6's text path")
    ap.add_argument("--reasoning-budget", type=int, default=None, metavar="N",
                    help="think mode: the reasoning tokens before a forced close (default: the family's)")
    ap.add_argument("--reasoning-out", default=None, metavar="FILE",
                    help="think mode: the kept samples' reasoning sidecar (scripts/train.py --reasoning)")
    return ap


class Drawn(NamedTuple):
    """The prompts of a run, their account, the prompts file and its bytes, and the excluded ids."""

    prompts: list[dict]
    account: dict
    path: Path
    data: bytes
    exclude: set[tuple[str, str]]


def server_props(client: LocalClient, server: str) -> dict | None:
    """The server's /props when it answers and reports one slot; None, the reason printed, otherwise."""
    if not client.health():
        print(f"no llama-server answers at {server}", file=sys.stderr)
        return None
    props = client.props() or {}
    if props.get("total_slots") != SLOTS:
        print(
            f"the server reports {props.get('total_slots')} slots (/props); start it with -np 1",
            file=sys.stderr,
        )
        return None
    return props


def model_file(props: dict) -> str | None:
    """The file name of the model the server runs (/props model_path), None when it reports none."""
    model_path = str(props.get("model_path") or "")
    return re.split(r"[\\/]", model_path)[-1] or None


def common_records(args: argparse.Namespace, drawn: Drawn, kept: list[dict], normalised: int) -> dict:
    """The manifest's "excluded" and "prompts" records, and its "kept" counts, as both paths write them."""
    return {
        "excluded": {"file": repo_relative(Path(args.exclude)) if drawn.exclude else None,
                     "records": len(drawn.exclude)},
        "prompts": {
            **drawn.account,
            "path": repo_relative(drawn.path),
            "sha256": hashlib.sha256(drawn.data).hexdigest(),
        },
        "kept": {
            "total": len(kept),
            **{kind: sum(k["kind"] == kind for k in kept) for kind in KINDS},
            "languages": {lang: sum(k["instruction_lang"] == lang for k in kept) for lang in ("en", "zh")},
            "normalised": normalised,
        },
    }


def output_record(out: Path, data: bytes, records: int) -> dict:
    return {"path": repo_relative(out), "records": records, "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data)}


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def text_replay(args: argparse.Namespace, drawn: Drawn) -> int:
    """M6's run (module docstring: the text path): its requests, its outputs and its manifest."""
    out, manifest_path = Path(args.out), Path(args.manifest)
    client = LocalClient(args.server)
    try:
        props = server_props(client, args.server)
        if props is None:
            return 2
        render = make_renderer(args.model)
        kept, rejected, done, normalised = generate(drawn.prompts, client, render, cap=args.cap,
                                                    max_tokens=args.max_tokens,
                                                    cap_per_kind=args.cap_per_kind)
    finally:
        client.close()
    data = jsonl_bytes(kept)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    records = common_records(args, drawn, kept, normalised)
    manifest = {
        "script": "scripts/replay_base.py",
        "seed": args.seed,
        "limit": args.limit,
        "cap": args.cap,
        "cap_per_kind": args.cap_per_kind,
        "templates": {"missing": MISSING_TEMPLATE, "irrelevant": IRRELEVANT_SENTENCE},
        "server": {
            "total_slots": props.get("total_slots"),
            "model": model_file(props),
            "build": props.get("build_info"),
        },
        "generation": {
            "lora_scale": 0.0,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "stop": list(STOP),
            "thinking": "off (chat template with enable_thinking=False)",
            "cache_prompt": False,
        },
        "excluded": records["excluded"],
        "prompts": records["prompts"],
        "generated": done,
        "kept": records["kept"],
        "rejected": dict(sorted(rejected.items())),
        "output": output_record(out, data, len(kept)),
    }
    write_manifest(manifest_path, manifest)
    print(f"{out}: kept {len(kept)} of {done} generated; rejected {dict(rejected)}")
    return 0


def family_replay(args: argparse.Namespace, plan: ReplayPlan, drawn: Drawn) -> int:
    """The family path's run (module docstring): the family loaded, every prompt encoded, the server and its
    context checked, the replay, then the samples, the sidecar of a think replay and the manifest."""
    out, manifest_path = Path(args.out), Path(args.manifest)
    settings = plan.settings
    think = plan.mode is Mode.THINK
    try:
        family = load_family(args.model, name=args.family)
        prompt_ids = [family.encode_prompt(family.prompt(family.to_template_messages(p["messages"]), [],
                                                         plan.mode))
                      for p in drawn.prompts]  # with the kept sample's own tools, none (to_sample)
        close = len(family.encode_close()) if think else 0
    except FamilyError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    longest = max((len(ids) for ids in prompt_ids), default=0)
    reach = longest + (settings.reasoning_budget + close if think else 0) + settings.answer_tokens
    client = LocalClient(args.server)
    try:
        props = server_props(client, args.server)
        if props is None:
            return 2
        n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
        if isinstance(n_ctx, bool) or not isinstance(n_ctx, int) or n_ctx < reach:
            parts = f"the longest prompt's {longest} ids"
            if think:
                parts += f", the reasoning budget {settings.reasoning_budget}, the close's {close} ids"
            print(f"the server's context (/props default_generation_settings.n_ctx: {n_ctx}) does not hold "
                  f"the longest generation, {reach} tokens ({parts} and {settings.answer_tokens} answer "
                  f"tokens); start it with -c {reach} or more", file=sys.stderr)
            return 2
        print(f"{family.spec.name}, {plan.mode}: {len(drawn.prompts)} prompts, the longest {longest} ids, "
              f"server context {n_ctx}", flush=True)
        run = generate_family(drawn.prompts, prompt_ids, client, family, settings, cap=args.cap,
                              cap_per_kind=args.cap_per_kind, report=report_progress)
    finally:
        client.close()
    kept = [item.sample for item in run.kept]
    model = model_file(props)
    rows = (sidecar_rows(run.kept, stem=out.stem, model=model or Path(args.model).name, settings=settings)
            if think else [])  # built, and so checked, before anything is written
    data = jsonl_bytes(kept)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    reasoning = None
    if think:
        sidecar = Path(args.reasoning_out)
        digest, size = write_sidecar(rows, sidecar)
        reasoning = {"path": repo_relative(sidecar), "rows": len(rows), "sha256": digest, "bytes": size}
    records = common_records(args, drawn, kept, run.normalised)
    manifest = {
        "script": "scripts/replay_base.py",
        "seed": args.seed,
        "limit": args.limit,
        "cap": args.cap,
        "cap_per_kind": args.cap_per_kind,
        "templates": {"missing": MISSING_TEMPLATE, "irrelevant": IRRELEVANT_SENTENCE},
        "family": family.record(),
        "server": {
            "total_slots": props.get("total_slots"),
            "model": model,
            "build": props.get("build_info"),
            "n_ctx": n_ctx,
        },
        "generation": {"path": "family", "lora_scale": 0.0, **settings.record(),
                       "stop": list(family.spec.stop), "cache_prompt": False},
        "excluded": records["excluded"],
        "prompts": records["prompts"],
        "generated": run.generated,
        "kept": {**records["kept"], "modes": {"think": len(rows), "no_think": len(kept) - len(rows)}},
        "rejected": dict(sorted(run.rejected.items())),
        "thinking": thinking_account(run) if think else None,
        "reasoning": reasoning,
        "output": output_record(out, data, len(kept)),
    }
    write_manifest(manifest_path, manifest)
    print(f"{out}: kept {len(kept)} of {run.generated} generated ({family.spec.name}, {plan.mode}); "
          f"rejected {dict(run.rejected)}")
    if think:
        thinking = manifest["thinking"]
        print(f"{args.reasoning_out}: {len(rows)} reasoning rows; closed by force {thinking['forced']}; kept "
              f"without a row {thinking['without_reasoning']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out, manifest_path = Path(args.out), Path(args.manifest)
    prompts_file = prompts_path(out)
    sidecar = () if args.reasoning_out is None else (Path(args.reasoning_out),)
    outputs = (prompts_file,) if args.dry_run else (out, manifest_path, *sidecar)
    existing = [p for p in outputs if p.exists()]
    if existing and not args.force:
        print(f"{', '.join(map(str, existing))} exists; pass --force to overwrite", file=sys.stderr)
        return 2
    plan = None
    if not args.dry_run:  # a dry run reads no model folder (module docstring: Written)
        try:
            plan = replay_plan(args)
        except ValueError as exc:  # a FamilyError is one
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2
    exclude = excluded_ids(Path(args.exclude))
    prompts, account = build_prompts(
        load_candidates(args.source_manifest, args.raw_dir), limit=args.limit, seed=args.seed, exclude=exclude
    )
    prompt_data = jsonl_bytes(prompts)
    prompts_file.parent.mkdir(parents=True, exist_ok=True)
    prompts_file.write_bytes(prompt_data)
    print(f"{prompts_file}: {len(prompts)} prompts {account['prompts']} (candidates {account['candidates']})")
    if args.dry_run:
        return 0
    drawn = Drawn(prompts, account, prompts_file, prompt_data, exclude)
    return text_replay(args, drawn) if plan.text else family_replay(args, plan, drawn)


if __name__ == "__main__":
    sys.exit(main())

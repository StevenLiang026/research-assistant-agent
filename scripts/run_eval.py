"""Run one system over a set of single-step evaluation items and write the runner's records
(research_agent.eval.runner.run_items; the systems are research_agent.eval.systems).

    .venv/Scripts/python.exe scripts/run_eval.py <items.jsonl> --system b0|b1|b2|b3|ft|b4|ftr|replay \\
        --out runs/eval/<name>.jsonl [--server http://127.0.0.1:8080] [--stage M3] [--max-cost RMB] \\
        [--limit N] [--tokenizer models/Qwen3-1.7B | --no-tokenizer] [--signals] \\
        [--mode think|no_think] [--seed N] [--reasoning-budget N] [--greedy] \\
        [--votes K [--vote-mode final|reproduction]] [--rules] [--glossary data/glossary/zh-en.json] \\
        [--backend llama.cpp | --backend hf [--adapter runs/train/<run>/adapter]]
    .venv/Scripts/python.exe scripts/score.py runs/eval/<name>.jsonl --items <items.jsonl>

Systems (docs/plan.md, 评测: 对照系统):
    b0      Qwen3-1.7B base on llama-server (lora scale 0.0, sent with every request)
    b1      b0 plus three fixed demonstrations in the system message (below)
    b2      deepseek-flash; b3 deepseek-v4-pro (the upper bound); both through the shared response cache and
            spending ledger in runs/llm/, booked under --stage (default M3) with that stage's cap from the
            plan (M3: 30 RMB); --max-cost lowers this invocation's cap to what the stage has spent plus RMB.
            They run in DeepSeek's thinking mode, or with thinking off when an item's history holds an
            assistant turn whose reasoning is unknown (research_agent.eval.systems); records carry the mode
    ft      the fine-tuned adapter on the same server (lora scale 1.0)
    b4      the B4 adapter (public data only) on a server that loads it (lora scale 1.0); its records are
            ft's, only the manifest's system and adapters tell the two apart
    ftr     ft routed to b2 (FT+R, research_agent.agent.route) under the frozen configuration
            results/routing-m4.json (scripts/route_curve.py): ft answers each item unless its signals route
            it to b2, booked like b2 under --stage. Unavailable (exit 2) without the configuration, or when
            its development set or adapter SHA-256 does not match data/eval/dev.jsonl or the adapter the
            server lists. Every record carries its "route" (which model answered, both signals, the
            reason); the manifest records the configuration ("routing": path, SHA-256, signal, threshold...)
    replay  answers each item with its own expected turn: a smoke test of run_eval plus score.py that costs
            nothing and must score 100%

--signals (local systems only: b0, b1, ft, b4): every ok record gains "signals", the step's confidence
signals (research_agent.agent.route: min_decision_prob, mean_logprob and hard_reason, the hard trigger or
null), which scripts/route_curve.py reads; the manifest records "signals": true. Without it the records are
byte for byte those of a run before M4.
Local systems need a running server (scripts/serve_local.py --adapter ... for ft and b4; b0 and b1 work with
or without an adapter loaded, since the scale is always sent) and the model folder (--tokenizer), whose
family (research_agent.family: identified by its config.json and chat template) renders and encodes the
prompt: the context the runner builds (build_context of the item's messages) with the tool schemas, in the
generation mode, sent as token ids (research_agent.llm.generation, docs/m7-家族层设计.md §7). The family's
NO_THINK prompt is the training-time rendering (research_agent.train.samples.prompt_renderer, the same
template call as every training sample's prompt).

Generation (M7). --mode think or no_think is the generation mode (default: the family's inference mode;
no_think for the Qwen3 hybrid family, so the FT2 runs repeat as they were), decoded with the family's
settings for that mode (greedy, or the vendor's sampling for a mode that thinks). --reasoning-budget N is
the tokens the reasoning may take before it is closed by force (default: the family's, 1024). --seed N is
the base seed of a sampled generation (default 0): every request's seed is derived from it, the prompt's
ids, the phase and the sample number, so a run repeats exactly; a greedy generation sends none. --greedy is
the bake-off's extra greedy thinking run (docs/m7-底座对比预登记.md, 解码: reported, never part of the
primary score): think mode decoded greedily (temperature 0 alone is sent, as in no_think greedy decoding, so
no seed goes out) whatever the family's sampling settings for that mode; it cannot be combined with --mode
no_think. The four are for the local systems on llama-server (b0, b1, ft, b4 and ftr's local side); the hf
backend answers greedily without thinking. The manifest records the settings ("generation") and the family
("family"), and
every ok record of a thinking generation carries the reasoning and how it was generated ("reasoning",
"thinking": the mode, the reasoning and answer token counts, whether the close was forced, the number of
requests and their seeds); a greedy no_think record is byte for byte the record before M7.

The inference wrappers (M7, docs/m7-本地提升计划.md WP2 to WP4; research_agent.eval.systems, The inference
wrappers), for the local systems on llama-server and ftr's local side. --votes K answers every item by a vote
of K samples, the greedy answer breaking ties (research_agent.agent.vote): sample k generates with the base
seed --seed + k and the sample number 0, so with the default seed 0 its answer is the --seed k run's, and the
greedy candidate is the --greedy run's; the samples need a mode whose decoding samples (exit 2 otherwise).
--vote-mode keys the answers as the final system does (final, the default) or as the 10-29 analysis did
(reproduction). --rules applies the runtime rules to every answer, each candidate's when voting
(research_agent.agent.rules), and --glossary FILE puts the glossary's note on every prompt
(research_agent.agent.noted). A voted record carries "votes" (the keys, the counts, the chosen class and its
representative), its usage the sum of the K + 1 requests; a ruled one "rules" (each change, [] for none). The
manifest records each only when used: the votes in "generation", the key mode as "vote_mode", the rules'
version under "rules", the glossary's path and SHA-256 under "glossary"; the timing sidecar's llama.cpp
timings of a voted item are summed over its requests. Without the four flags every output is what it was.

Backends (research_agent.eval.systems). --backend llama.cpp, the default, runs a local system on the server;
--backend hf runs it in full precision on transformers instead (research_agent.llm.hf: the weights of the
--tokenizer folder in bfloat16 on the GPU, no server), for the M3 comparison with the quantized model
(scripts/compare_backends.py): the same prompt and the same reading of the completion, greedy, at most 512 new
tokens. --adapter names the PEFT adapter folder it loads (runs/train/<run>/adapter, or a trainer checkpoint
folder runs/train/<run>/trainer/checkpoint-<N>; never a GGUF, llama-server's form), which scale 1 applies (ft
and b4 need one) and scale 0 disables (b0 and b1 run without one too). Only the local systems have a backend,
and only the hf backend takes --adapter.

Which sets. A locked test set (data/eval/test.jsonl, data/eval/e2e-test.jsonl, however the path is spelled,
or a copy with a test set's digest) is refused before anything is read (exit 4): only the phase-4 test
runner opens one, through LockedSet (research_agent.eval.lock). So are, by location, the files of the test
set's handwritten items with their gold turns, data/eval/handwritten.jsonl and handwritten-specs.jsonl, and,
once read, any items file that holds test material (an item of split "test" or with an id of the test split:
a subset of a test set, or a copy with other line ends; eval.lock.refuse_test_material), before the system is
built. The development set data/eval/dev.jsonl, and any items file with a lock beside it, is opened through
DevSet: its lock must match (exit 5 otherwise).

b1's design. Three demonstrations from the assembled training set data/raw/synth/synth.jsonl, fixed by
request id and all from the train split (checked against data/synth/requests.jsonl, so no dev or test
material leaks): req-train-1572 (a dated search whose arguments are exactly the request's topic, start date,
count and ranking: one search_arxiv call), req-train-0513 (a note for "the paper I just closed": one
ask_clarification call) and req-train-0261 (whether a parameter has a default: an answer without a call).
Each shows the record's user message and the teacher's first turn, calls written as Qwen3
writes them (<tool_call> blocks). They are rendered once as a worked-examples block, headed
"以下三个示例只演示回复的格式(调用工具、追问、直接回答),与当前请求无关", and appended to the
system message of every item, identically; the context check counts the block too. The block's SHA-256
prefix is in the manifest.

The context check: with the tokenizer (default) an item whose context does not fit the context limit is
recorded as context_overflow without asking the model, as the agent loop would; --no-tokenizer skips it.

Output: the records at --out (JSONL, one per item in item order, no timestamps), and next to them the
manifest <out stem>.manifest.json: system, provider, model, lora scale, temperature, prompt version and hash,
schema hash, b1's demonstrations, the items file, its SHA-256 and whether its lock was checked
("items_locked"), the number of items run, the date, the server, its /props and the adapters it loaded
(local systems: each one's path as the server was given it and the GGUF's SHA-256), the backend of a local
system, and on the hf backend the dtype, the device and the adapter folder with the SHA-256 of its
adapter_model.safetensors ("adapter"), with the model folder, GPU and package versions in place of the
server's /props, the stage, its cap, the cost of the run and the turns per thinking mode, "modes" (cloud
systems), the status counts, the git commit and whether the tree was dirty; and the timing sidecar
<out stem>.timing.json (research_agent.eval.timing): per item its wall time and its model calls (wall time,
error, paid or cached, llama.cpp's own timings), so that the records stay free of time.

Exit 0 when every item was run (some may be error or context_overflow records); 1 on malformed input (an
items file that is not JSONL objects, or items without distinct ids); 2 when the system cannot be built here
(no key, no price for the model, no server, no training set for b1, a loaded adapter whose GGUF is not here
to hash; on the hf backend no adapter folder for ft or b4, a GGUF or a folder without the PEFT files as
--adapter, a model or an adapter that cannot be loaded, such as a corrupt adapter_model.safetensors; nothing
is written; for ftr also no routing configuration, or one that does not match the development set or the
adapter the server lists; a thinking run of a family that never thinks) and on a usage error (argparse's
exit, --backend hf with a cloud system or replay, --adapter without --backend hf, --signals, --mode,
--reasoning-budget or --greedy with a system that is not local, and --greedy with --mode no_think or on the
hf backend included; --votes, --rules or --glossary with a system that is not local, --vote-mode without
--votes, --votes with --greedy or on the hf backend, and a vote whose decoding does not sample or a glossary
that cannot be read); 3 when the
spending cap stopped the run (the records and the manifest are
written, the items after the stop marked skipped); 4 when the items are a locked test set, a file of the
handwritten test items or any other test material and 5 when they do not match their lock (nothing is
written).
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from research_agent.agent.context import family_token_counter, qwen_token_counter
from research_agent.agent.prompts import PROMPT_HASH, PROMPT_VERSION
from research_agent.agent.types import BUDGET_STOP
from research_agent.agent.vote import FINAL
from research_agent.canonical import canonical_json
from research_agent.eval.lock import (
    DevSet,
    JsonlError,
    LockError,
    RefusedTestSet,
    lock_required,
    read_jsonl,
    refuse_test_material,
    refuse_test_sets,
    sha256_of,
)
from research_agent.eval.runner import run_items, write_results
from research_agent.eval.systems import (
    BACKENDS,
    CLOUD_THINKING,
    DEFAULT_STAGE,
    DEV_SET,
    HF,
    LLAMA_CPP,
    LOCAL_SCALES,
    ROUTED,
    ROUTING_CONFIG,
    STAGE_CAPS,
    SYSTEMS,
    TRAINING_SET,
    SystemUnavailable,
    add_wrapper_flags,
    build_system,
    with_votes,
    wrapper_flags_error,
)
from research_agent.eval.timing import ItemTimer, TimedModel, write_timing
from research_agent.family.core import GREEDY, Mode, load_family
from research_agent.llm.budget import cost_rmb, load_prices
from research_agent.llm.generation import GenerationSettings
from research_agent.tools.schemas import SCHEMA_HASH
from research_agent.train.manifest import REPO, git_commit, git_dirty

TOKENIZER = "models/Qwen3-1.7B"
LLM_DIR = "runs/llm"
SERVER = "http://127.0.0.1:8080"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run one system over single-step evaluation items.")
    ap.add_argument("items", type=Path, help="the evaluation items (JSONL)")
    ap.add_argument("--system", required=True, choices=SYSTEMS)
    ap.add_argument("--out", type=Path, required=True,
                    help="the records (JSONL), e.g. runs/eval/<name>.jsonl")
    ap.add_argument("--server", default=SERVER, help=f"llama-server for local systems (default {SERVER})")
    ap.add_argument("--cloud-thinking", default="auto", choices=CLOUD_THINKING,
                    help="a cloud system's thinking (or a routed system's cloud side): auto, the default, or "
                         "off for every request (research_agent.llm.cloud)")
    ap.add_argument("--stage", default=DEFAULT_STAGE, choices=sorted(STAGE_CAPS),
                    help=f"the stage cloud spending is booked under (default {DEFAULT_STAGE})")
    ap.add_argument("--max-cost", type=float, default=None, metavar="RMB",
                    help="what this invocation may add to the stage's spending "
                         "(default: up to the stage cap)")
    ap.add_argument("--limit", type=int, default=None, metavar="N", help="run only the first N items")
    ap.add_argument("--tokenizer", default=TOKENIZER,
                    help=f"model folder (a registered family, research_agent.family.registry) for the chat "
                         f"template and the context check, and the weights of --backend hf (default "
                         f"{TOKENIZER})")
    ap.add_argument("--no-tokenizer", action="store_true", help="skip the context check")
    ap.add_argument("--backend", choices=BACKENDS, default=LLAMA_CPP,
                    help=f"where a local system runs: llama-server ({LLAMA_CPP}, the default) or "
                         f"transformers in full precision ({HF})")
    ap.add_argument("--adapter", type=Path, default=None,
                    help="the PEFT adapter folder --backend hf loads (runs/train/<run>/adapter or a trainer "
                         "checkpoint folder; not a GGUF)")
    ap.add_argument("--signals", action="store_true",
                    help="record the confidence signals of every local step (local systems only)")
    ap.add_argument("--mode", choices=[str(mode) for mode in Mode], default=None,
                    help="the generation mode of a local system on llama-server (default: the family's "
                         "inference mode)")
    ap.add_argument("--seed", type=int, default=0, metavar="N",
                    help="the base seed of a sampled generation (default 0): each request's seed is derived "
                         "from it, the prompt, the phase and the sample")
    ap.add_argument("--reasoning-budget", type=int, default=None, metavar="N",
                    help="the tokens the reasoning may take before it is closed by force (default: the "
                         "family's)")
    ap.add_argument("--greedy", action="store_true",
                    help="the extra greedy thinking run of the bake-off: think mode decoded greedily "
                         "(temperature 0, no seed), whatever the family's sampling for that mode")
    add_wrapper_flags(ap)
    ap.add_argument("--llm-dir", default=LLM_DIR, help=argparse.SUPPRESS)
    ap.add_argument("--routing-config", default=str(ROUTING_CONFIG), help=argparse.SUPPRESS)
    ap.add_argument("--dev-set", default=str(DEV_SET), help=argparse.SUPPRESS)
    ap.add_argument("--training-set", default=str(TRAINING_SET), help=argparse.SUPPRESS)
    return ap


def run_cost(records: list[dict], model: str) -> float | None:
    """RMB of the paid turns at the price-list upper bound; None for a model without a price (local)."""
    price = load_prices().get(model)
    if price is None:
        return None
    return sum(cost_rmb(price, r.get("usage") or {}) for r in records if r.get("paid"))


def local_date() -> str:
    """Today's date on this machine's clock, for the manifest (the records themselves carry no time)."""
    return datetime.datetime.now(datetime.UTC).astimezone().date().isoformat()


def manifest_path(out: Path) -> Path:
    return out.with_name(out.stem + ".manifest.json")


def timing_path(out: Path) -> Path:
    return out.with_name(out.stem + ".timing.json")


def main(argv: list[str] | None = None, *, openai_client: object | None = None,
         local_client: object | None = None, render=None,
         clock: Callable[[], float] = time.perf_counter) -> int:
    """openai_client, local_client (LocalClient or, with --backend hf, HFClient) and render stand in for the
    real ones in tests, clock for the timing sidecar's."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.backend == HF and args.system not in LOCAL_SCALES:
        parser.error(f"--backend hf runs the local systems only ({', '.join(LOCAL_SCALES)}), "
                     f"not {args.system}")
    if args.adapter is not None and args.backend != HF:
        parser.error("--adapter is for --backend hf; llama-server loads its own adapter "
                     "(scripts/serve_local.py --adapter)")
    if args.signals and args.system not in LOCAL_SCALES:
        parser.error(f"--signals records the signals of the local systems only ({', '.join(LOCAL_SCALES)}), "
                     f"not {args.system}")
    local_side = args.system in LOCAL_SCALES or args.system in ROUTED
    if (args.mode is not None or args.reasoning_budget is not None or args.greedy) and not local_side:
        parser.error(f"--mode, --reasoning-budget and --greedy set the generation of the local systems only "
                     f"({', '.join((*LOCAL_SCALES, *ROUTED))}), not {args.system}")
    if args.greedy and args.mode == str(Mode.NO_THINK):
        parser.error("--greedy is a thinking run decoded greedily: it cannot be combined with "
                     "--mode no_think")
    if args.backend == HF and (args.mode == str(Mode.THINK) or args.greedy):
        parser.error("--mode think and --greedy need llama-server (--backend llama.cpp): the hf backend "
                     "answers greedily without thinking")
    wrapper_error = wrapper_flags_error(args, local_side=local_side, hf=args.backend == HF)
    if wrapper_error is not None:
        parser.error(wrapper_error)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    roots = (Path.cwd(), REPO)
    try:
        refuse_test_sets(args.items, roots)
        locked = lock_required(args.items, roots)
        items = DevSet(args.items).open() if locked else read_jsonl(args.items)
        refuse_test_material(items, args.items)
    except RefusedTestSet as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 4
    except LockError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 5
    except (JsonlError, OSError, ValueError) as exc:
        print(f"FAIL: malformed items: {exc}", file=sys.stderr)
        return 1
    selected = items if args.limit is None else items[:args.limit]
    family = generation = None
    if local_side and args.backend == LLAMA_CPP and render is None:  # the family path (module docstring)
        try:
            family = load_family(args.tokenizer)
            generation = GenerationSettings.for_family(family, mode=Mode.THINK if args.greedy else args.mode,
                                                       reasoning_budget=args.reasoning_budget,
                                                       base_seed=args.seed)
        except ValueError as exc:  # a FamilyError is one: the folder is no registered family
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2
        if args.greedy:  # the mode's own decoding samples; this run decodes it greedily (module docstring)
            generation = dataclasses.replace(generation, decoding=GREEDY)
    try:
        generation = with_votes(generation, args.votes)
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    count_tokens = None
    if not args.no_tokenizer:  # loads the folder's tokenizer (transformers): only here
        count_tokens = (family_token_counter(family, generation.mode) if family is not None
                        else qwen_token_counter(args.tokenizer))
    try:
        built = build_system(args.system, server=args.server, tokenizer=args.tokenizer,
                             count_tokens=count_tokens, llm_dir=args.llm_dir, stage=args.stage,
                             max_cost=args.max_cost, items=selected, training_set=args.training_set,
                             openai_client=openai_client, local_client=local_client, render=render,
                             backend=args.backend, adapter=args.adapter, routing_config=args.routing_config,
                             dev_set=args.dev_set, cloud_thinking=args.cloud_thinking, generation=generation,
                             family=family, rules=args.rules, glossary=args.glossary,
                             vote_mode=args.vote_mode or FINAL)
    except SystemUnavailable as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    try:
        props = getattr(built.local_client, "props", None)
        server_props = props() if props is not None else None
        last = built.timings or (lambda: getattr(built.local_client, "last_timings", None))  # a vote's sums
        timed = TimedModel(built.model, clock=clock, server_timings=last)
        timer = ItemTimer(timed, clock=clock)
        try:
            records = run_items(selected, timed, count_tokens=built.count_tokens, on_item=timer,
                                signals=args.signals)
        except ValueError as exc:  # ids missing or repeated
            print(f"FAIL: malformed items: {exc}", file=sys.stderr)
            return 1
    finally:
        built.close()
    write_results(records, args.out)
    write_timing({"records": args.out.name, "items": timer.items}, timing_path(args.out))
    statuses = Counter(r["status"] for r in records)
    modes = Counter(r["mode"] for r in records if "mode" in r)  # cloud turns only
    info = built.info
    manifest = {
        **info, "prompt_version": PROMPT_VERSION, "prompt_hash": PROMPT_HASH, "schema_hash": SCHEMA_HASH,
        "items": args.items.as_posix(), "items_sha256": sha256_of(args.items), "items_locked": locked,
        "items_total": len(items), "items_run": len(selected), "limit": args.limit, "date": local_date(),
        "server_props": server_props, "context_check": None if args.no_tokenizer else args.tokenizer,
        "statuses": dict(sorted(statuses.items())),
        "cost_rmb": run_cost(records, info.get("cloud_model") or info["model"]),  # ftr: its cloud turns
        **({"modes": dict(sorted(modes.items()))} if modes else {}),
        **({"signals": True} if args.signals else {}),
        "git_commit": git_commit(REPO), "git_dirty": git_dirty(REPO),
    }
    path = manifest_path(args.out)
    path.write_text(canonical_json(manifest) + "\n", encoding="utf-8", newline="\n")
    counts = ", ".join(f"{k} {v}" for k, v in sorted(statuses.items()))
    where = f", {info['backend']}" if "backend" in info else ""  # local systems only
    print(f"{args.out}: {len(records)} records ({counts}),"
          f" system {args.system} ({info['model']}{where}); manifest {path}; timing {timing_path(args.out)}")
    print(f"score: python scripts/score.py {args.out.as_posix()} --items {args.items.as_posix()}")
    return 3 if statuses.get(BUDGET_STOP) else 0


if __name__ == "__main__":
    sys.exit(main())

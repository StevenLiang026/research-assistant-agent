"""Generate synthetic trajectories: deepseek-flash drives the agent loop over the recorded corpus, one request
record at a time (research_agent.data.synth; docs/modules/接口.md, 数据生成相关的契约).

The owner's functions come from the tree (research_agent.validation.check_call, research_agent.policy.decide,
research_agent.arxiv.parse_atom and extract_section) unless --placeholders swaps check_call, decide and
extract_section for the stand-ins below, which accept every call. parse_atom has no stand-in: the corpus
index is built with it, so without the owner's module --parse-atom <path.py>:<function> names one (any file,
inside or outside the repository); --extract-section does the same for the extractor. Every function is
probed once before anything is paid, so a NotImplementedError stops the run at once.

The spending ledger and the response cache live in runs/llm/ (budget.sqlite, cache.sqlite) and are shared by
every run from now on; the cap for one invocation is what the stage (--stage, M2 by default; its cap is
research_agent.eval.systems.STAGE_CAPS) has spent plus --max-cost, never more than the stage's cap. The ledger
books what DeepSeek charges (half the list price off peak, research_agent.llm.budget). The key is read by name
from the environment, never printed.

M7 (docs/m7-计划.md WP1): --model drives the turns and --rewrite-model (default: --model) rewrites the
requests, each with its own client on the same cache and ledger; --thinking off sends every turn of the
teacher without thinking, --rewrite-thinking off every rewrite (research_agent.llm.cloud; default auto for
both); --off-peak-only makes every paid call wait for DeepSeek's off-peak hours: in the peak hours the run
stops before the request as at a spent cap (budget_stop rows are run again by the next invocation). The
rewrite model and both thinking settings are run settings: another value is another run.

Run:
    .venv/Scripts/python.exe scripts/generate_trajectories.py data/synth/dryrun-requests.jsonl \\
        --placeholders --parse-atom <module.py>:parse_atom [--no-tokenizer] [--no-rewrite] [--stub-notes] \\
        [--no-seed-notes]
Writes runs/m2/synth/<run>/{index.jsonl, manifest.json, transcripts/<id>.json, notes/<id>/} (gitignored):
write_note writes real notes into a store per request unless --stub-notes selects the stand-ins, which
write nothing. Before its turn each request's store holds five scaffold notes on corpus papers its split may
use (the split's topics from <stem>-manifest.json next to the requests file; the whole corpus without one),
unless --no-seed-notes. A second invocation with the same settings resumes the run (research_agent.data.synth
says which rows are redone: a budget_stop row always, the others with --redo). Another seed, date, model,
placeholders flag, rewrite setting, notes or seed-notes setting or requests file is another run and needs
another --run name; --redo does not override that, while --limit, --scenario and --split only narrow a run
and may change.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import re
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from research_agent.agent.models import CloudModel
from research_agent.agent.types import HarnessError
from research_agent.data.synth import (
    OwnerFunctionMissing,
    OwnerFunctions,
    Rewriter,
    corpus_recorded_on,
    generate,
    load_requests,
    load_validator,
    probe,
    requests_manifest_of,
    select_requests,
)
from research_agent.eval.systems import STAGE_CAPS
from research_agent.llm.budget import Budget, load_off_peak
from research_agent.llm.cache import ResponseCache
from research_agent.llm.cloud import THINKING_MODES, CloudClient
from research_agent.net.fetch import Fetcher
from research_agent.tools.corpus_search import MANIFEST, CorpusIndex, corpus_ids, load_papers
from research_agent.train.manifest import sha256_of

DEFAULT_STAGE = "M2"  # the stage the M2 runs were booked under; M7 passes --stage M7
CASSETTES = "data/cassettes"
LLM_DIR = "runs/llm"
OUT = "runs/m2/synth"
TOKENIZER = "models/Qwen3-1.7B"
MODEL = "deepseek-flash"
OWNER_MODULE = "research_agent.arxiv"  # where parse_atom and extract_section live once they are implemented
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_TAG = re.compile(r"<[^>]+>")


# ---- PLACEHOLDERS for the owner's modules (accept every call; fixed answers), as scripts/m1_smoke.py -----


def placeholder_check_call(tool: str, args: dict, seen_ids: list[str], today: str) -> list[str]:
    return []  # the owner's check_call (JSON Schema + semantic checks) replaces this


def placeholder_decide(event: str, times: int) -> str:
    if event == "empty_results":
        return "relax" if times == 1 else "tell_user"
    return "stop"  # the owner's decide replaces this


def placeholder_extract_section(html: str, section: str, max_chars: int) -> str | None:
    """The page's text with the tags stripped, cut to max_chars; None when nothing is left."""
    text = " ".join(_TAG.sub(" ", html).split())
    return text[:max_chars] or None


# ---- owner resolution -----------------------------------------------------------------------------------


def load_function(spec: str, flag: str) -> Callable:
    """The function named by "<path.py>:<name>", imported from that file, which may lie outside the repo;
    flag names the option in the message of a SystemExit."""
    path, sep, name = spec.rpartition(":")  # the last colon: a Windows drive letter has one of its own
    if not sep or not path or not name.isidentifier():
        raise SystemExit(f"{flag} must be <path.py>:<function>, not {spec!r}")
    module_spec = importlib.util.spec_from_file_location(f"owner_module_{Path(path).stem}", path)
    if module_spec is None or module_spec.loader is None:
        raise SystemExit(f"{flag}: cannot import {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    try:
        return getattr(module, name)
    except AttributeError:
        raise SystemExit(f"{flag}: {path} has no function {name}") from None


def owner_function(name: str, flag: str, module: str = OWNER_MODULE) -> Callable:
    """The owner's function from its module in the tree; a SystemExit that points at flag when the module or
    the function is not there yet."""
    try:
        return getattr(importlib.import_module(module), name)
    except ImportError:
        raise SystemExit(f"{module} is not in the tree yet; pass {flag} <path.py>:<function>") from None
    except AttributeError:
        raise SystemExit(f"{module} has no {name}; pass {flag} <path.py>:<function>") from None


def resolve_owner_functions(*, placeholders: bool, parse_atom_spec: str | None,
                            extract_section_spec: str | None) -> OwnerFunctions:
    """The four owner functions: the tree's (or the placeholders), with the two arXiv functions taken from
    the given specs first, then from OWNER_MODULE (extract_section: the placeholder when asked)."""
    if placeholders:
        check_call, decide, extract_section = (placeholder_check_call, placeholder_decide,
                                               placeholder_extract_section)
    else:
        check_call = importlib.import_module("research_agent.validation").check_call
        decide = importlib.import_module("research_agent.policy").decide
        extract_section = None
    if parse_atom_spec:
        parse_atom = load_function(parse_atom_spec, "--parse-atom")
    else:
        parse_atom = owner_function("parse_atom", "--parse-atom")
    if extract_section_spec:
        extract_section = load_function(extract_section_spec, "--extract-section")
    elif extract_section is None:
        extract_section = owner_function("extract_section", "--extract-section")
    return OwnerFunctions(check_call, decide, parse_atom, extract_section)


# ---- command line ---------------------------------------------------------------------------------------


def iso_date(text: str) -> str:
    if not _DATE.fullmatch(text):
        raise argparse.ArgumentTypeError(f"expected YYYY-MM-DD, got {text!r}")
    return text


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate synthetic trajectories over the recorded corpus.")
    ap.add_argument("requests", help="the request records (JSONL)")
    ap.add_argument("--run", default=None, help="run name (default: the requests file's stem)")
    ap.add_argument("--out", default=OUT, help=f"where runs live (default {OUT})")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, metavar="N", help="run only the first N requests")
    ap.add_argument("--scenario", default=None, help="run only this scenario")
    ap.add_argument("--split", default=None, help="run only this split")
    ap.add_argument("--placeholders", action="store_true",
                    help="use the script's stand-ins for check_call, decide and extract_section")
    ap.add_argument("--parse-atom", default=None, metavar="PATH.py:FUNCTION")
    ap.add_argument("--extract-section", default=None, metavar="PATH.py:FUNCTION")
    ap.add_argument("--tokenizer", default=TOKENIZER, help="the model folder the token counter reads")
    ap.add_argument("--no-tokenizer", action="store_true", help="run without the context token check")
    ap.add_argument("--no-rewrite", action="store_true", help="run the request texts as they are")
    ap.add_argument("--stub-notes", action="store_true",
                    help="use the note tools' stand-ins, which write nothing (default: a real note store per "
                         "request under <run>/notes/<id>/)")
    ap.add_argument("--no-seed-notes", action="store_true",
                    help="start each request's note store empty (default: five scaffold notes on corpus "
                         "papers the request's split may use, written before its turn)")
    ap.add_argument("--redo", action="store_true", help="rerun requests already in the index")
    ap.add_argument("--max-cost", type=float, default=5.0, metavar="RMB",
                    help="what this invocation may add to the stage's spending (default 5.0)")
    ap.add_argument("--model", default=MODEL, help=f"the teacher that drives the turns (default {MODEL})")
    ap.add_argument("--rewrite-model", default=None,
                    help="the model that rewrites the requests (default: --model)")
    ap.add_argument("--thinking", default="auto", choices=THINKING_MODES,
                    help="the teacher's thinking: auto (decided per request) or off for every turn")
    ap.add_argument("--rewrite-thinking", default="auto", choices=THINKING_MODES,
                    help="the rewriter's thinking: auto or off for every rewrite")
    ap.add_argument("--stage", default=DEFAULT_STAGE, choices=sorted(STAGE_CAPS),
                    help=f"the stage the spending is booked under (default {DEFAULT_STAGE})")
    ap.add_argument("--off-peak-only", action="store_true",
                    help="pay only outside DeepSeek's peak hours; in them the run stops (module docstring)")
    ap.add_argument("--today", type=iso_date, default=None,
                    help="the loop's date (default: the corpus manifest's recorded_on)")
    ap.add_argument("--cassettes", default=CASSETTES, help=argparse.SUPPRESS)
    ap.add_argument("--corpus-manifest", default=MANIFEST.as_posix(), help=argparse.SUPPRESS)
    ap.add_argument("--llm-dir", default=LLM_DIR, help=argparse.SUPPRESS)
    return ap


def main(argv: list[str] | None = None, *, openai_client: object | None = None,
         clock: Callable[[], datetime] | None = None) -> int:
    """openai_client stands in for the OpenAI client in tests; None reads the key from the environment. clock
    stands in for the ledger's clock (the peak hours) in tests."""
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")  # a cp1252 console must not fail on Chinese text
    today = args.today or corpus_recorded_on(args.corpus_manifest)
    owner = resolve_owner_functions(placeholders=args.placeholders, parse_atom_spec=args.parse_atom,
                                    extract_section_spec=args.extract_section)
    try:
        probe(owner, today)
    except OwnerFunctionMissing as exc:
        raise SystemExit(f"{exc}; pass --placeholders (check_call, decide, extract_section) or "
                         "--parse-atom / --extract-section <path.py>:<function>") from None
    requests = load_requests(args.requests)
    selected = select_requests(requests, scenario=args.scenario, split=args.split, limit=args.limit)
    run = args.run or Path(args.requests).stem

    fetcher = Fetcher("replay", args.cassettes)
    index = CorpusIndex(load_papers(fetcher, corpus_ids(args.corpus_manifest), owner.parse_atom))
    count_tokens = None
    if not args.no_tokenizer:
        from research_agent.agent.context import qwen_token_counter  # imports transformers: only here

        count_tokens = qwen_token_counter(args.tokenizer)
    llm_dir = Path(args.llm_dir)
    stage, stage_cap = args.stage, STAGE_CAPS[args.stage]
    spent = Budget(llm_dir / "budget.sqlite", stage=stage, stage_cap_rmb=stage_cap).spent(stage)
    cap = min(stage_cap, spent + args.max_cost)
    budget = Budget(llm_dir / "budget.sqlite", stage=stage, stage_cap_rmb=cap, off_peak=load_off_peak(),
                    clock=clock)
    cache = ResponseCache(llm_dir / "cache.sqlite")
    client = CloudClient(args.model, cache=cache, budget=budget, client=openai_client, thinking=args.thinking,
                         off_peak_only=args.off_peak_only)
    model = CloudModel(client)
    rewrite_model = None if args.no_rewrite else (args.rewrite_model or args.model)
    rewriter = None
    if rewrite_model is not None:
        rewrite_client = client
        if (rewrite_model, args.rewrite_thinking) != (args.model, args.thinking):
            rewrite_client = CloudClient(rewrite_model, cache=cache, budget=budget, client=openai_client,
                                         thinking=args.rewrite_thinking, off_peak_only=args.off_peak_only)
        rewriter = Rewriter(rewrite_client)
    try:
        validator = load_validator()
    except HarnessError as exc:  # the owner's module is there but check_trajectory is not
        raise SystemExit(str(exc)) from None
    settings = {"requests": str(args.requests), "requests_sha256": sha256_of(args.requests), "run": run,
                "model": args.model, "max_cost_rmb": args.max_cost, "limit": args.limit,
                "scenario": args.scenario, "split": args.split,
                "tokenizer": None if args.no_tokenizer else args.tokenizer, "redo": args.redo,
                "stage": stage, "rewrite_model": rewrite_model, "thinking": args.thinking,
                "rewrite_thinking": None if rewrite_model is None else args.rewrite_thinking,
                "off_peak_only": args.off_peak_only}
    requests_manifest = requests_manifest_of(args.requests)
    if args.no_seed_notes or args.stub_notes:
        seeding = "off"
    else:
        seeding = "per split" if requests_manifest else "whole corpus (no requests manifest)"
    print(f"run {run}: {len(selected)} of {len(requests)} requests, today {today}, index of {len(index)} "
          f"papers, {stage} spent {spent:.4f} RMB, cap {budget.stage_cap:.4f} RMB, "
          f"validator {'on' if validator else 'off'}, rewrite {'off' if args.no_rewrite else 'on'}, "
          f"notes {'stub' if args.stub_notes else 'store'}, seed notes {seeding}",
          flush=True)
    try:
        manifest = generate(selected, out=Path(args.out) / run, model=model, owner=owner, fetcher=fetcher,
                            index=index, today=today, settings=settings, placeholders=args.placeholders,
                            count_tokens=count_tokens, seed=args.seed, rewriter=rewriter,
                            notes="stub" if args.stub_notes else "store", validator=validator, redo=args.redo,
                            corpus_manifest=args.corpus_manifest, seed_notes=not args.no_seed_notes,
                            requests_manifest=requests_manifest,
                            log=lambda line: print(line, flush=True))
    except HarnessError as exc:  # the run folder holds another setting, or an index row that is not one
        raise SystemExit(str(exc)) from None
    cost = manifest["cost_rmb"]
    print(f"done: {manifest['requests']} rows in {Path(args.out) / run}, by status "
          f"{manifest['counts']['by_status']}, realized mix {manifest['realized_mix']['counts']}, "
          f"cost {cost['total']:.4f} RMB (trajectories {cost['trajectories']:.4f}, rewrites "
          f"{cost['rewrites']:.4f}), validated {manifest['validated']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Run one system over a refusal development set of M8 and score it (docs/m8-计划.md WP2; the items, the
scorer and the summary are research_agent.eval.special_dev's).

    .venv/Scripts/python.exe scripts/run_special_dev.py data/eval/special-dev-b.jsonl --system b0|ft \\
        --out runs/m8/special-dev/<name>.jsonl [--tokenizer models/Nanbeige4.1-3B] \\
        [--server http://127.0.0.1:8080] [--reasoning-budget 1280] [--answer-tokens 1200] [--limit N] \\
        [--timeout 600] [--run-copy DIR] [--force]

How an item is answered (docs/m8-计划.md WP2, 运行方式: the family path, thinking, greedy, R 1,280, one
slot). The item's messages are sent as they are, the functions in its system message and no tools field, as
ACEBench sends them: the model folder's family (research_agent.family.load_family over --tokenizer) renders
them with its chat template, tools None, in think mode (LoadedFamily.prompt: the prompt ends with the
thinking block's opening) and encodes them, and the ids go to llama-server through LocalClient by the
two-phase generation of research_agent.llm.generation: the reasoning within --reasoning-budget tokens (1,280,
the R of M7's ACEBench runs, which M8 keeps: docs/m8-计划.md, 方法约束), closed by force at the budget,
then the answer within --answer-tokens (1,200, ACEBench's reply budget: scripts/run_acebench.py
REPLY_TOKENS), decoded greedily whatever the family's sampling for the mode (temperature 0 alone and no
seed, as run_eval's --greedy), stopped at the family's terminator, without a prompt cache. b0 sends lora
scale 0.0 (the base model, with an adapter loaded or without), ft 1.0. An item whose prompt, reasoning
budget and answer budget together do not fit the server's context (/props n_ctx) is recorded as
context_overflow without a request; a request that fails is recorded as an error and the run goes on
(eval.runner's rule).

Which sets. Refused before anything is read (exit 4): a locked test set or a file of the handwritten test
items (eval.lock.refuse_test_sets, as scripts/run_eval.py refuses them), and a held-out set (held_out:
HELD_OUT, data/eval/dev-ext.jsonl and data/eval/v2/dev-v2-ext.jsonl, and anything under HELD_OUT_DIRS,
runs/m7/heldout/, under the repository or its run copy (held_out_roots: the folder --run-copy names, else
the one the environment variable RAA_RUN_COPY names; no default), by resolved path;
dev-ext.jsonl and dev-v2-ext.jsonl by name wherever they are; and by content, a file with the SHA-256 of one
of the two, which is hashed only when its size is the items file's, as scripts/vote_offline.py finds a
copy). Once read, items that are test material (eval.lock.refuse_test_material; exit 4). A set with a lock
beside it, as dev set B has, is opened through eval.lock.DevSet: the lock must match (exit 5). Every item
must pass special_dev.check_item, the ids must be distinct and no fixed-word list may stand for two cases
(special_dev.sentence_families; a family may come in several placeholder styles) (exit 1).

Checks before the first request (exit 2): the model folder is a registered family that thinks, the server
answers /health, its /props reports one slot (-np 1: with several, a greedy answer depends on the requests
before it), every adapter it lists is a file here (its SHA-256 goes to the manifest:
eval.systems.loaded_adapters), and for ft it lists exactly one.

Written, refused when any of the three exists unless --force; the records and the manifest as soon as the
answers are in, before anything is scored, so that no failure of the scorer costs the run:
- --out, the records: JSONL, one per item in item order, canonical JSON, no time; {"id", "status": "ok",
  "answer", "reasoning", "reasoning_chars", "thinking": {mode, reasoning_tokens, answer_tokens, forced,
  phases, seeds}, "finish_reason", "usage", "prompt_tokens", "model"}, {"id", "status": "error", "error",
  "error_type"} or {"id", "status": "context_overflow", "prompt_tokens"};
- <out stem>.manifest.json: the system, its lora scale and model name, the server, its /props (slots, build,
  the base model's path and file name, n_ctx), the adapters it lists with their SHA-256, the model folder
  and its family's record, the generation settings, the request's form, the items file with its SHA-256,
  whether its lock was checked, the items in it and run, the limit, the statuses, the date, the git commit
  and whether the tree was dirty;
- <out stem>.summary.json: the records file's name, the items file and its SHA-256, the scoring rules
  (special_dev.RULES), the summary (under "summary", where scripts/decide_m8.py reads it) and every item's
  score. An item the scorer fails on is scored as failed, its score's status "scoring_error" with the
  error, and counted (summary "scoring_errors"); the others are scored all the same.
Then the summary is printed, and every item that failed with its failure (dev sets are not test material).

Exit 0 written (scoring errors included: the answers are in); 1 malformed items; 2 a usage error (an
existing output without --force, a model folder of no registered family or of a family that never thinks,
no server, not one slot, an adapter that is not a file here, ft without exactly one adapter); 4 test material
or a held-out set; 5 a set that does not match its lock.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from research_agent.agent.types import CONTEXT_OVERFLOW
from research_agent.canonical import canonical_json
from research_agent.eval import special_dev as sd
from research_agent.eval.lock import (
    DevSet,
    JsonlError,
    LockError,
    RefusedTestSet,
    lock_required,
    named_set,
    read_jsonl,
    refuse_test_material,
    refuse_test_sets,
    sha256_of,
    write_jsonl,
)
from research_agent.eval.runner import ERROR, OK, error_fields
from research_agent.eval.systems import SystemUnavailable, loaded_adapters, local_name
from research_agent.family.core import GREEDY, LoadedFamily, Mode, load_family
from research_agent.llm.generation import GenerationSettings, generate
from research_agent.llm.local import LocalClient
from research_agent.train.manifest import REPO, git_commit, git_dirty

SYSTEMS = {"b0": 0.0, "ft": 1.0}  # lora scale per system
SERVER = "http://127.0.0.1:8080"
TOKENIZER = "models/Nanbeige4.1-3B"
REASONING_BUDGET = 1280  # R of M7's ACEBench runs, kept by M8
ANSWER_TOKENS = 1200  # ACEBench's reply budget (scripts/run_acebench.py REPLY_TOKENS)
TIMEOUT = 600.0  # seconds a request may take: a reasoning of R tokens and an answer of 1,200 on one slot
SLOTS = 1
# The held-out sets, never run here (module docstring: Which sets), under the repository and, when one is
# named, its run copy (held_out_roots).
HELD_OUT = (Path("data/eval/dev-ext.jsonl"), Path("data/eval/v2/dev-v2-ext.jsonl"))
HELD_OUT_DIRS = (Path("runs/m7/heldout"),)
RUN_COPY_ENV = "RAA_RUN_COPY"  # names the run copy when --run-copy does not; there is no default run copy
HELD_OUT_ROOTS = (REPO,)


class UsageError(RuntimeError):
    pass


class MalformedItems(ValueError):
    pass


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run one system over a refusal development set of M8.")
    ap.add_argument("items", type=Path, help="the development set (JSONL)")
    ap.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    ap.add_argument("--out", type=Path, required=True, help="the records (JSONL)")
    ap.add_argument("--tokenizer", default=TOKENIZER,
                    help=f"the model folder whose family renders and encodes the prompts (default "
                         f"{TOKENIZER})")
    ap.add_argument("--server", default=SERVER, help=f"llama-server (default {SERVER})")
    ap.add_argument("--reasoning-budget", type=int, default=REASONING_BUDGET, metavar="N",
                    help=f"the tokens the reasoning may take before it is closed by force (default "
                         f"{REASONING_BUDGET})")
    ap.add_argument("--answer-tokens", type=int, default=ANSWER_TOKENS, metavar="N",
                    help=f"the tokens the answer may take (default {ANSWER_TOKENS})")
    ap.add_argument("--limit", type=int, default=None, metavar="N", help="run only the first N items")
    ap.add_argument("--timeout", type=float, default=TIMEOUT,
                    help=f"seconds per request (default {TIMEOUT:g})")
    ap.add_argument("--run-copy", type=Path, default=None, metavar="DIR",
                    help=f"the run copy, a second checkout whose held-out sets are refused as well (default: "
                         f"the folder the environment variable {RUN_COPY_ENV} names; none when it is unset)")
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    return ap


def manifest_path(out: Path) -> Path:
    return out.with_name(out.stem + ".manifest.json")


def summary_path(out: Path) -> Path:
    return out.with_name(out.stem + ".summary.json")


def local_date() -> str:
    """Today's date on this machine's clock, for the manifest (the records carry no time)."""
    return datetime.datetime.now(datetime.UTC).astimezone().date().isoformat()


def _inside(path: Path, folder: Path) -> bool:
    resolved = os.path.normcase(os.path.realpath(path))
    base = os.path.normcase(os.path.realpath(folder))
    return resolved == base or resolved.startswith(base.rstrip(os.sep) + os.sep)


def held_out_roots(run_copy: Path | None = None) -> tuple[Path, ...]:
    """The roots the held-out sets are looked for under: HELD_OUT_ROOTS (the repository) and the run copy, the
    checkout the GPU jobs run from, when one is named: run_copy (--run-copy), else the folder the environment
    variable RAA_RUN_COPY names. Without either, HELD_OUT_ROOTS alone."""
    if run_copy is None:
        named = os.environ.get(RUN_COPY_ENV, "").strip()
        run_copy = Path(named) if named else None
    return HELD_OUT_ROOTS if run_copy is None else (*HELD_OUT_ROOTS, Path(run_copy))


def held_out(path: Path, roots: tuple[Path, ...] | None = None) -> str | None:
    """How path is a held-out set (module docstring: Which sets) under roots (default held_out_roots()):
    "path", "folder", "name" or "content"; None when it is none. A held-out file is read only when its size is
    path's."""
    roots = held_out_roots() if roots is None else roots
    if named_set(path, HELD_OUT, roots) is not None:
        return "path"
    if any(_inside(path, Path(root) / folder) for root in roots for folder in HELD_OUT_DIRS):
        return "folder"
    if path.name in {name.name for name in HELD_OUT}:
        return "name"
    try:
        size = path.stat().st_size
    except OSError:  # no such file: reading the items will say so
        return None
    digest = None
    for root in roots:
        for name in HELD_OUT:
            held = Path(root) / name
            try:
                if not held.is_file() or held.stat().st_size != size:  # another size, another SHA-256
                    continue
            except OSError:
                continue
            digest = digest or sha256_of(path)
            if sha256_of(held) == digest:
                return "content"
    return None


def load_items(path: Path, held_out_in: tuple[Path, ...] | None = None) -> tuple[list[dict], bool]:
    """The items and whether their lock was checked, after the guards and checks of the module docstring
    (RefusedTestSet, LockError, MalformedItems); held_out_in are the roots of the held-out sets (default
    held_out_roots())."""
    roots = (Path.cwd(), REPO)
    refuse_test_sets(path, roots)
    how = held_out(path, held_out_in)
    if how is not None:
        raise RefusedTestSet(f"{path} is a held-out set (found by its {how}); it is never run here")
    locked = lock_required(path, roots)
    try:
        items = DevSet(path).open() if locked else read_jsonl(path)
    except (JsonlError, OSError) as exc:
        raise MalformedItems(str(exc)) from None
    refuse_test_material(items, path)
    ids = [item.get("id") for item in items]
    repeated = sorted({i for i in ids if isinstance(i, str) and ids.count(i) > 1})
    if repeated:
        raise MalformedItems(f"item ids must be distinct; repeated: {repeated}")
    for number, item in enumerate(items, 1):
        errors = sd.check_item(item)
        if errors:
            name = item["id"] if isinstance(item.get("id"), str) else f"#{number}"
            raise MalformedItems(f"{name}: {'; '.join(errors)}")
    try:
        sd.sentence_families(items)
    except ValueError as exc:
        raise MalformedItems(str(exc)) from None
    return items, locked


def server_record(props: dict) -> dict:
    """What the manifest keeps of the server's /props: the slots, the build, the base model's path and file
    name and the context size."""
    model_path = props.get("model_path")
    settings = props.get("default_generation_settings")
    model = re.split(r"[\\/]", model_path)[-1] if isinstance(model_path, str) and model_path else None
    return {"total_slots": props.get("total_slots"), "build_info": props.get("build_info"),
            "model_path": model_path, "model": model,
            "n_ctx": settings.get("n_ctx") if isinstance(settings, dict) else None}


def check_server(client, system: str) -> tuple[dict, list[dict] | None]:
    """The server's /props and its adapters after the checks of the module docstring; UsageError
    otherwise."""
    if not client.health():
        raise UsageError("no llama-server answers /health")
    props = client.props() or {}
    if props.get("total_slots") != SLOTS:
        raise UsageError(f"the server reports {props.get('total_slots')} slots (/props); start it with -np 1")
    try:
        adapters = loaded_adapters(client)
    except SystemUnavailable as exc:
        raise UsageError(str(exc)) from None
    if SYSTEMS[system] and len(adapters or []) != 1:
        raise UsageError(f"{system} runs on the one adapter the server loaded, and it lists "
                         f"{len(adapters or [])}")
    return props, adapters


def generation_settings(family: LoadedFamily, reasoning_budget: int,
                        answer_tokens: int) -> GenerationSettings:
    """The family's think mode, decoded greedily (module docstring); ValueError for a family that never
    thinks."""
    settings = GenerationSettings.for_family(family, mode=Mode.THINK, reasoning_budget=reasoning_budget)
    return dataclasses.replace(settings, decoding=GREEDY, answer_tokens=answer_tokens)


def answer_item(item: dict, client, family: LoadedFamily, settings: GenerationSettings, *,
                lora_scale: float, model: str, n_ctx: int | None) -> dict:
    """The record of one item (module docstring: Written)."""
    try:
        context = family.to_template_messages(item["messages"])
        prompt_ids = family.encode_prompt(family.prompt(context, None, settings.mode))
        if n_ctx and len(prompt_ids) + settings.reasoning_budget + settings.answer_tokens > n_ctx:
            return {"id": item["id"], "status": CONTEXT_OVERFLOW, "prompt_tokens": len(prompt_ids)}
        gen = generate(client, family, prompt_ids, settings, lora_scale=lora_scale)
    except Exception as exc:  # noqa: BLE001  deliberate: one item's failure is recorded, the run goes on
        return {"id": item["id"], "status": ERROR, **error_fields(exc)}
    thinking = {"mode": str(settings.mode), "reasoning_tokens": gen.reasoning_tokens,
                "answer_tokens": gen.answer_tokens, "forced": gen.forced, "phases": gen.phases,
                "seeds": gen.seeds}
    return {"id": item["id"], "status": OK, "answer": gen.answer_text, "reasoning": gen.reasoning,
            "reasoning_chars": len(gen.reasoning or ""), "thinking": thinking,
            "finish_reason": gen.answer.finish_reason, "usage": dict(gen.usage),
            "prompt_tokens": len(prompt_ids), "model": model}


def main(argv: list[str] | None = None, *, local_client=None) -> int:
    """local_client stands in for LocalClient in tests."""
    args = build_parser().parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    paths = (args.out, manifest_path(args.out), summary_path(args.out))
    existing = [path.as_posix() for path in paths if path.exists()]
    if existing and not args.force:
        print(f"FAIL: {', '.join(existing)} exists; pass --force to overwrite", file=sys.stderr)
        return 2
    try:
        items, locked = load_items(args.items, held_out_roots(args.run_copy))
    except RefusedTestSet as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 4
    except LockError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 5
    except MalformedItems as exc:
        print(f"FAIL: malformed items: {exc}", file=sys.stderr)
        return 1
    selected = items if args.limit is None else items[:args.limit]
    try:
        family = load_family(args.tokenizer)
        settings = generation_settings(family, args.reasoning_budget, args.answer_tokens)
    except ValueError as exc:  # a FamilyError is one, and so is a family without a think mode
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    lora_scale = SYSTEMS[args.system]
    model = f"{local_name(args.tokenizer)}:lora={lora_scale:g}"
    client = local_client if local_client is not None else LocalClient(args.server, timeout=args.timeout)
    try:
        try:
            props, adapters = check_server(client, args.system)
        except UsageError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2
        server = server_record(props)
        records = [answer_item(item, client, family, settings, lora_scale=lora_scale, model=model,
                               n_ctx=server["n_ctx"]) for item in selected]
    finally:
        if local_client is None:
            client.close()
    statuses = Counter(record["status"] for record in records)
    manifest = {
        "script": "scripts/run_special_dev.py", "system": args.system, "lora_scale": lora_scale,
        "model": model, "server": args.server, "server_props": server, "adapters": adapters,
        "tokenizer": str(args.tokenizer), "family": family.record(), "generation": settings.record(),
        "request": {"messages": "the item's messages as they are", "tools": None, "cache_prompt": False,
                    "stop": list(family.spec.stop)},
        "items": args.items.as_posix(), "items_sha256": sha256_of(args.items), "items_locked": locked,
        "items_total": len(items), "items_run": len(selected), "limit": args.limit,
        "statuses": dict(sorted(statuses.items())), "date": local_date(),
        "git_commit": git_commit(REPO), "git_dirty": git_dirty(REPO),
    }
    write_jsonl(records, args.out)
    manifest_path(args.out).write_text(canonical_json(manifest) + "\n", encoding="utf-8", newline="\n")
    scores = sd.score_records(records, selected)  # an item the scorer fails on is scored as failed
    summary = sd.summarize(scores, selected)
    report = {"records": args.out.name, "items": manifest["items"], "items_sha256": manifest["items_sha256"],
              "rules": sd.RULES, "summary": summary, "scores": scores}
    summary_path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8", newline="\n")
    counts = ", ".join(f"{status} {count}" for status, count in sorted(statuses.items()))
    print(f"{args.out}: {len(records)} records ({counts}), system {args.system} ({model}); manifest "
          f"{manifest_path(args.out)}; summary {summary_path(args.out)}")
    for line in sd.summary_lines(summary):
        print(line)
    for score in scores:
        if score["failure"] is not None:
            error = f" ({score['error']})" if score["status"] == "scoring_error" else ""
            print(f"{score['id']} {score['kind']} {score['lang']}: {score['failure']}{error}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

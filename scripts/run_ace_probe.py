"""Run the ACEBench-style development probe on one system against a running llama-server and score it
(docs/m6-计划.md §3.4 D3, WP4); research_agent.eval.ace_probe holds the prompt and the scorer.

    uv run --no-sync python scripts/run_ace_probe.py --system b0|ft|b4 [--adapter <gguf>] \\
        [--server http://127.0.0.1:8080] [--out results/ace-probe-<system>.json] \\
        [--probe data/eval/ace-probe.jsonl] [--model models/Qwen3-1.7B] [--force]

How each item is sent. As ACEBench sends its prompts to a local model (scripts/run_acebench.py and the
patched ACEBench client: a chat request of two messages, the system text and the user text, no tools field,
temperature 0, at most 1,200 reply tokens, thinking off through the server's chat-template kwargs): the
item's two messages are rendered with the model folder's chat template (research_agent.llm.local.hf_renderer:
tools None, generation prompt appended, enable_thinking False, so the same text the server's template gives
with thinking off) and completed through LocalClient: temperature 0 (greedy), max_tokens 1,200, stop at the
end of the turn (agent.models.STOP), no logprobs, no prompt cache. b0 is lora_scale 0.0 (the base model,
whether or not the server has an adapter loaded); ft and b4 are lora_scale 1.0 and need the server to list
exactly one adapter, whose file name is the --adapter GGUF's (default: the "chosen" GGUF of
results/checkpoint-selection-<system>.json, whose SHA-256 must still be the recorded one). b0 takes no
--adapter.

Checks before the first request: the probe is not test material (research_agent.eval.lock.refuse_test_sets
on the path, then refuse_test_material on the items read; exit 4), its lock <probe>.sha256 matches (exit 2),
every item passes ace_probe.check_item (exit 2); the server answers /health and its /props reports one slot
(`-np 1`: with several slots a greedy answer depends on the requests before it; exit 2).

Written: --out (default results/ace-probe-<system>.json), JSON with two-space indent, LF, no time, refused
when it exists unless --force: the rules, the manifest (system, lora scale, adapter path and SHA-256 or null,
the server's /props: total_slots, build_info, model file name, n_ctx; the probe's path, SHA-256 and item
count; the generation settings), the rates per kind ("kinds"), per language and kind ("languages"), the
failure shapes per kind and language ("shapes"), and one record per item with its score and answer.

Exit 0 written; 2 usage (an existing output without --force, a bad --adapter, a probe whose lock fails or
that is malformed, no server, a server without one slot or without the system's adapter); 4 test material.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

from research_agent.agent.models import STOP
from research_agent.agent.types import api_to_hf
from research_agent.eval import ace_probe as ap
from research_agent.eval.lock import (
    JsonlError,
    LockError,
    RefusedTestSet,
    check_lock,
    lock_path,
    read_jsonl,
    refuse_test_material,
    refuse_test_sets,
)
from research_agent.llm.local import LocalClient

REPO = Path(__file__).resolve().parents[1]
SYSTEMS = ("b0", "ft", "b4")
ADAPTER_SYSTEMS = ("ft", "b4")
PROBE = Path("data/eval/ace-probe.jsonl")
MODEL = "models/Qwen3-1.7B"
SELECTION = "results/checkpoint-selection-{}.json"
MAX_TOKENS = 1200  # ACEBench's reply budget (run_acebench.GENERATION)
SLOTS = 1


class UsageError(RuntimeError):
    pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--system", required=True, choices=SYSTEMS)
    p.add_argument("--adapter", type=Path, default=None, help="the adapter GGUF (ft, b4)")
    p.add_argument("--server", default="http://127.0.0.1:8080")
    p.add_argument("--out", type=Path, default=None, help="default results/ace-probe-<system>.json")
    p.add_argument("--probe", type=Path, default=PROBE)
    p.add_argument("--model", default=MODEL, help="model folder whose chat template renders the prompts")
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--force", action="store_true")
    return p


def default_out(args: argparse.Namespace) -> Path:
    return args.out if args.out is not None else Path(f"results/ace-probe-{args.system}.json")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_adapter(system: str, explicit: Path | None) -> dict | None:
    """{gguf, sha256, selection} of the system's adapter; None for b0; UsageError when it is wrong."""
    if system not in ADAPTER_SYSTEMS:
        if explicit is not None:
            raise UsageError("b0 is the base model and takes no --adapter")
        return None
    if explicit is not None:
        if not explicit.is_file():
            raise UsageError(f"the adapter {explicit} is not here")
        return {"gguf": explicit.as_posix(), "sha256": sha256_file(explicit), "selection": None}
    selection = Path(SELECTION.format(system))
    try:
        choice = json.loads(selection.read_text(encoding="utf-8"))["chosen"]
        gguf, recorded = Path(choice["gguf"]), choice["sha256"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise UsageError(f"no --adapter and {selection.as_posix()} names no chosen GGUF ({exc})") from None
    if not gguf.is_file():
        raise UsageError(f"{selection.as_posix()} chose {gguf.as_posix()}, which is not here")
    sha = sha256_file(gguf)
    if sha != recorded:
        raise UsageError(f"{gguf.as_posix()} has changed since {selection.as_posix()} chose it")
    return {"gguf": gguf.as_posix(), "sha256": sha, "selection": selection.as_posix()}


def load_probe(path: Path) -> list[dict]:
    """The probe's items after the guards and checks of the module docstring (RefusedTestSet, UsageError)."""
    roots = (Path.cwd(), REPO)
    refuse_test_sets(path, roots)
    if not path.is_file():
        raise UsageError(f"{path} is not here (scripts/build_ace_probe.py writes it)")
    try:
        items = read_jsonl(path)
    except JsonlError as exc:
        raise UsageError(str(exc)) from None
    refuse_test_material(items, path)
    try:
        check_lock(path)
    except LockError as exc:
        raise UsageError(str(exc)) from None
    for item in items:
        errors = ap.check_item(item)
        if errors:
            raise UsageError(f"{path}: {item.get('id')}: {errors}")
    return items


def server_record(props: dict) -> dict:
    model_path = str(props.get("model_path") or "")
    settings = props.get("default_generation_settings") or {}
    return {
        "total_slots": props.get("total_slots"),
        "build_info": props.get("build_info"),
        "model": re.split(r"[\\/]", model_path)[-1] or None,
        "n_ctx": settings.get("n_ctx") if isinstance(settings, dict) else None,
    }


def check_server(client, system: str, adapter: dict | None) -> dict:
    """The server's /props after the checks of the module docstring; UsageError otherwise."""
    if not client.health():
        raise UsageError("no llama-server answers /health")
    props = client.props() or {}
    if props.get("total_slots") != SLOTS:
        raise UsageError(f"the server reports {props.get('total_slots')} slots (/props); start it with -np 1")
    if adapter is not None:
        loaded = client.adapters() or []
        names = [re.split(r"[\\/]", str(a.get("path", "")))[-1] for a in loaded]
        if names != [Path(adapter["gguf"]).name]:
            raise UsageError(f"the server lists adapters {names}, not exactly {Path(adapter['gguf']).name}")
    return props


def make_renderer(model: str | Path) -> Callable:
    """The model folder's chat template, thinking off, no tools block."""
    from research_agent.llm.local import hf_renderer  # heavy import (transformers), only for a real run

    return hf_renderer(model)


def answer_items(items: list[dict], client, render: Callable, *, lora_scale: float, max_tokens: int) -> list:
    answers = []
    for item in items:
        prompt = render(api_to_hf(item["messages"]), None)
        completion = client.complete(
            prompt,
            lora_scale=lora_scale,
            temperature=0.0,
            logprobs=None,
            stop=list(STOP),
            max_tokens=max_tokens,
            cache_prompt=False,
        )
        answers.append((completion.text, getattr(completion, "finish_reason", None)))
    return answers


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = default_out(args)
    if out.exists() and not args.force:
        print(f"FAIL: {out} exists; pass --force to overwrite", file=sys.stderr)
        return 2
    try:
        items = load_probe(args.probe)
        adapter = resolve_adapter(args.system, args.adapter)
    except RefusedTestSet as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 4
    except UsageError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
    lora_scale = 0.0 if adapter is None else 1.0
    client = LocalClient(args.server)
    try:
        try:
            props = check_server(client, args.system, adapter)
        except UsageError as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 2
        render = make_renderer(args.model)
        answers = answer_items(items, client, render, lora_scale=lora_scale, max_tokens=args.max_tokens)
    finally:
        client.close()
    scores = [ap.score_answer(item, text) for item, (text, _) in zip(items, answers, strict=True)]
    summary = ap.summarize(scores)
    result = {
        "probe": "ACEBench-style development probe of M6 (docs/m6-计划.md §3.4 D3)",
        "rules": {
            "scoring": ap.SCORING_RULE,
            "shapes": ap.SHAPE_RULE,
            "format": ap.FORMAT,
            "sentence": ap.SENTENCE_TEMPLATE,
        },
        "manifest": {
            "system": args.system,
            "lora_scale": lora_scale,
            "adapter": adapter,
            "server": server_record(props),
            "probe": {
                "path": args.probe.as_posix(),
                "sha256": sha256_file(args.probe),
                "items": len(items),
                "lock": lock_path(args.probe).as_posix(),
            },
            "generation": {
                "messages": "the item's system and user messages, rendered with the model's chat template",
                "tools_field": None,
                "thinking": "off (chat template with enable_thinking=False)",
                "temperature": 0.0,
                "max_tokens": args.max_tokens,
                "stop": list(STOP),
                "cache_prompt": False,
                "template_model": args.model,
            },
        },
        "kinds": summary["kinds"],
        "languages": summary["languages"],
        "shapes": summary["shapes"],
        "items": [
            {**score, "finish_reason": finish, "answer": text}
            for score, (text, finish) in zip(scores, answers, strict=True)
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    k = summary["kinds"]
    print(
        f"{args.system}: normal parsable {k['normal']['parsable']}/{k['normal']['n']}, exact "
        f"{k['normal']['exact']}; incomplete sentence {k['incomplete']['sentence']}/{k['incomplete']['n']}, "
        f"parameter {k['incomplete']['parameter']}; written {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

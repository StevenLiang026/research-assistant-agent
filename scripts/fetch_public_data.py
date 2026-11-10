"""Fetch the public function-calling datasets, convert them to the one sample format, draw the plan's counts.

The sources and counts are those of docs/plan.md (微调数据): ToolACE 1,500, glaive-function-calling-v2 800,
hermes-function-calling-v1 700, glaive_toolcall_zh 500 and Deepexi/function-calling-small 500. For each
source the Hub id is resolved with huggingface_hub.dataset_info, which also gives the card's license field
and the current commit; the data files are downloaded at that commit with hf_hub_download, into the default
Hugging Face cache (shared with the datasets library); the records are converted with
research_agent.data.public; and the plan's count is drawn with random.Random(seed) over the converted
samples, a fresh generator per source, so one source's draw never depends on another's.

Two rules shape the draw. Length: when --model points at a model directory, every converted sample is
rendered whole with the model's chat template (api_to_hf plus the sample's own tools, thinking off), and a
sample over schemas.CONTEXT_TOKENS is excluded before the draw, because the trainer drops such a sample's
last step instead of truncating it; the manifest counts the exclusions per source. Share of no-call chats:
docs/plan.md gives glaive-v2 as "about 10% no-call", which is its intended anti-forgetting share, while
the converted pools are 44.0% (glaive-v2) and 48.6% (glaive_toolcall_zh) no-call (measured 2026-10-09), so
those two draws are stratified to 10% no-call (80 of 800, 50 of 500) and the other sources are drawn
uniformly. A gated or missing dataset is reported in the manifest and the others go on. Nothing is
deduplicated: that is the owner's module, so the pools are written as they come and the manifest reports
plain counts.

Written:
- data/raw/public/<name>.jsonl: the drawn samples, one JSON object per line, UTF-8 without escapes. They
  live under data/raw/ because that directory is gitignored (the plan keeps raw data out of the repository).
- data/public/manifest.json: per source the Hub id, revision, license as stated on the card, files, rows,
  converted, dropped by reason, the pool's counts (its no-call share among them), the draw (target share,
  candidates, excluded for length, drawn per stratum), the written samples' counts and SHA-256, and token
  statistics of the written samples.
- data/public/LICENSES.md: the attribution each license asks for.

Run: .venv/Scripts/python.exe scripts/fetch_public_data.py [--out data/raw/public] [--seed 42]
     [--model models/Qwen3-1.7B]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import httpx
from huggingface_hub import dataset_info, hf_hub_download, scan_cache_dir
from huggingface_hub.errors import HfHubHTTPError

from research_agent.agent.types import api_to_hf
from research_agent.data.public import CONVERTERS, convert_many, has_tool_call
from research_agent.tools.schemas import CONTEXT_TOKENS

SEED = 42
MODEL = "models/Qwen3-1.7B"
OUT = Path("data/raw/public")
PUBLIC = Path("data/public")
NO_CALL_SHARE = 0.1  # docs/plan.md: glaive-v2 "about 10% no-call" (anti-forgetting); both glaive pools
LICENSE_URLS = {"apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
                "cc-by-4.0": "https://creativecommons.org/licenses/by/4.0/"}
# What can go wrong with one source without being a bug here: Hub errors (missing, gated, revision),
# transport errors, unreadable files. Anything else propagates.
FETCH_ERRORS = (HfHubHTTPError, httpx.HTTPError, OSError, ValueError)


@dataclass(frozen=True)
class Source:
    name: str  # the "source" field of the samples and the JSONL file name
    repo: str  # Hugging Face Hub dataset id
    count: int  # samples to draw (docs/plan.md)
    files: tuple[str, ...]  # data files in the repository, one per config when configs are given
    configs: tuple[str, ...] = ()  # config names of the files, for source ids "config/row"
    excluded: dict[str, str] = field(default_factory=dict)  # configs not used, with the reason
    note: str = ""  # a statement of the card that the attribution repeats
    no_call_share: float | None = None  # share of the draw without a tool call; None draws uniformly


SINGLETURN_NOTE = ("every one of its 1,893 prompts also opens a func_calling conversation (measured "
                   "2026-10-09), so it adds only prefixes")
SOURCES = [
    Source("toolace", "Team-ACE/ToolACE", 1500, ("data.json",),
           note=("The card asks that uses cite: Liu et al., ToolACE: Winning the Points of LLM Function "
                 "Calling, arXiv:2409.00920 (2024).")),
    Source("glaive_v2", "glaiveai/glaive-function-calling-v2", 800, ("glaive-function-calling-v2.json",),
           no_call_share=NO_CALL_SHARE),
    Source("hermes", "NousResearch/hermes-function-calling-v1", 700,
           ("func-calling.json", "glaive-function-calling-5k.json"), ("func_calling", "glaive_func_calling"),
           excluded={"func_calling_singleturn": SINGLETURN_NOTE,
                     "json_mode_agentic": "structured output without tools",
                     "json_mode_singleturn": "structured output without tools"},
           note=("The card describes the data as synthetic, generated by NousResearch and collaborators; the "
                 "glaive_func_calling config is Glaive AI's function-calling data, updated and cleaned.")),
    Source("glaive_zh", "llamafactory/glaive_toolcall_zh", 500, ("glaive_toolcall_zh_1k.json",),
           note=("The card states the data is borrowed from glaiveai/glaive-function-calling-v2 and "
                 "translated by GPT-3.5."), no_call_share=NO_CALL_SHARE),
    Source("deepexi", "Deepexi/function-calling-small", 500,
           ("function-calling-small_aliyun_openapi_V2.csv",),
           note="The card describes the data as information on 700+ Alibaba Cloud OpenAPIs, in Chinese."),
]


def card(source: Source) -> dict:
    """What the Hub says about the dataset: commit, license field, gating and the files it holds."""
    info = dataset_info(source.repo, files_metadata=True)
    data = info.card_data.to_dict() if info.card_data else {}
    files = {s.rfilename: s.size for s in info.siblings or []}
    missing = [f for f in source.files if f not in files]
    if missing:
        raise FileNotFoundError(f"{source.repo}: {missing} not in the repository; it holds {sorted(files)}")
    return {"repo": source.repo, "revision": info.sha, "license": data.get("license"), "gated": info.gated,
            "last_modified": info.last_modified.isoformat() if info.last_modified else None,
            "files": {f: files[f] for f in source.files},
            "notice_files": sorted(f for f in files if f.upper().startswith(("NOTICE", "LICENSE")))}


def download(source: Source, revision: str) -> list[Path]:
    return [Path(hf_hub_download(source.repo, filename, repo_type="dataset", revision=revision))
            for filename in source.files]


def load_records(path: Path) -> list[dict]:
    """The records of a data file: a JSON list, or the rows of a CSV with a header."""
    if path.suffix == ".csv":
        with path.open(encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise TypeError(f"{path.name}: expected a JSON list, got {type(records).__name__}")
    return records


def convert(source: Source, paths: list[Path]) -> tuple[list[dict], list[dict]]:
    """All converted samples of the source in file order, and a per-file account."""
    samples: list[dict] = []
    files = []
    converter = CONVERTERS[source.name]
    for index, path in enumerate(paths):
        config = source.configs[index] if source.configs else None
        records = load_records(path)
        ids = [f"{config}/{i}" for i in range(len(records))] if config else None
        converted, dropped = convert_many(records, converter, ids=ids)
        files.append({"file": source.files[index], "config": config, "rows": len(records),
                      "converted": len(converted), "dropped": dict(dropped.most_common())})
        samples += converted
    return samples, files


def render(sample: dict, tokenizer) -> str:
    """The whole conversation as the chat template renders it: the sample's own tools, thinking off."""
    return tokenizer.apply_chat_template(api_to_hf(sample["messages"]), tools=sample["tools"] or None,
                                         tokenize=False, enable_thinking=False)


def token_lengths(samples: list[dict], tokenizer, chunk: int = 1000) -> list[int]:
    """The token count of every sample's render, tokenized chunk texts at a time (one call per text is
    four times slower with the Hugging Face tokenizer; measured 2026-10-09 on the glaive-v2 pool)."""
    lengths: list[int] = []
    for start in range(0, len(samples), chunk):
        texts = [render(s, tokenizer) for s in samples[start:start + chunk]]
        lengths += [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
    return lengths


def stratum_sizes(count: int, no_call_share: float) -> tuple[int, int]:
    """(samples without a call, samples with one) that make up count at the given no-call share."""
    no_call = round(count * no_call_share)
    return no_call, count - no_call


def draw(samples: list[dict], count: int, seed: int, *, no_call_share: float | None = None,
         lengths: list[int] | None = None) -> tuple[list[int], dict]:
    """The indices of count samples drawn without replacement with random.Random(seed), in pool order, and
    an account of the draw.

    With lengths (one per sample), a sample over CONTEXT_TOKENS is not a candidate. Without no_call_share
    the draw is uniform over the candidates; with it, stratum_sizes gives how many candidates without a tool
    call and how many with one are drawn, a stratum that is too small being taken whole and the other one
    filling the count. Fewer candidates than count are all taken.
    """
    fit = [i for i in range(len(samples)) if lengths is None or lengths[i] <= CONTEXT_TOKENS]
    rng = random.Random(seed)
    if no_call_share is None:
        picked = list(fit) if count >= len(fit) else rng.sample(fit, count)
    else:
        without = [i for i in fit if not has_tool_call(samples[i])]
        with_call = [i for i in fit if has_tool_call(samples[i])]
        n_without, n_with = stratum_sizes(count, no_call_share)
        n_without = min(n_without, len(without))
        n_with = min(count - n_without, len(with_call))
        n_without = min(count - n_with, len(without))
        picked = rng.sample(without, n_without) + rng.sample(with_call, n_with)
    picked.sort()
    no_call = sum(1 for i in picked if not has_tool_call(samples[i]))
    return picked, {"no_call_share": no_call_share, "candidates": len(samples),
                    "excluded_for_length": None if lengths is None else len(samples) - len(fit),
                    "fit": len(fit), "no_call": no_call, "with_call": len(picked) - no_call}


def repo_relative(path: Path) -> str:
    """path relative to the working directory, or its bare name when it lies outside (no user paths)."""
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def write_jsonl(path: Path, samples: list[dict]) -> str:
    """samples as JSON lines, UTF-8 without escapes, LF line ends; returns the file's SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counts(samples: list[dict]) -> dict:
    """Plain counts: samples, those without a tool call (and their share), parallel steps, multi-step
    samples. No duplicate statistics: dedup is the owner's module and gets no ready-made key from here."""
    steps = [[m for m in s["messages"] if m.get("tool_calls")] for s in samples]
    no_call = sum(1 for s in samples if not has_tool_call(s))
    return {"samples": len(samples), "no_call": no_call,
            "no_call_share": round(no_call / len(samples), 4) if samples else None,
            "parallel_steps": sum(1 for calls in steps for m in calls if len(m["tool_calls"]) > 1),
            "multi_step": sum(1 for calls in steps if len(calls) > 1)}


def token_stats(lengths: list[int]) -> dict:
    """Statistics of whole-conversation token counts (as token_lengths gives them).

    The last training step of a sample is its whole conversation, so a sample over CONTEXT_TOKENS loses at
    least that step (research_agent.train.samples drops over-length steps, never truncates).
    """
    lengths = sorted(lengths)
    n = len(lengths)
    return {"samples": n, "min": lengths[0], "median": lengths[n // 2], "mean": round(sum(lengths) / n),
            "p95": lengths[min(n - 1, int(0.95 * (n - 1)))], "max": lengths[-1],
            "over_context": sum(1 for length in lengths if length > CONTEXT_TOKENS),
            "context_tokens": CONTEXT_TOKENS}


def attribution(source: Source, entry: dict, written: int) -> str:
    """The LICENSES.md section of one source, from the card's license field at the fetched revision."""
    license_id = entry.get("license") or "not stated on the card"
    url = LICENSE_URLS.get(license_id)
    lines = [f"## {source.repo}", "",
             f"- Source: https://huggingface.co/datasets/{source.repo}, revision `{entry['revision']}`.",
             f"- License as stated on the dataset card: `{license_id}`" + (f" ({url})." if url else "."),
             (f"- Use in this project: {written} of its {entry['rows']} records (files: "
              f"{', '.join(entry['files'])}) were converted to the messages/tools format by "
              f"`src/research_agent/data/public.py` and drawn into `data/raw/public/{source.name}.jsonl`, "
              "which is not part of the repository; the fine-tuned adapter is trained on them.")]
    if license_id == "cc-by-4.0":
        creator = source.repo.split("/")[0]
        lines.append(f"- Attribution (CC BY 4.0): created by {creator} and published on the Hugging Face Hub "
                     "under the Creative Commons Attribution 4.0 International license. Changes made: the "
                     "dataset's own notation (systemPrompt, userPrompt, assistantResponse) was converted to "
                     "messages with tool calls and a tools field, the system prompt was replaced by that "
                     "field, and quotes around the prompts were removed. The licensor does not endorse this "
                     "project. This notice accompanies the converted data and the documentation of any model "
                     "trained on it.")
    elif license_id == "apache-2.0":
        notice = entry.get("notice_files") or []
        held = (f"the repository holds the notice files {notice}, whose contents must travel with it."
                if notice else "the repository holds no NOTICE or LICENSE file at this revision, so the "
                               "license text at the link above is the whole notice.")
        lines.append("- Apache License 2.0: redistributions of the data must keep this license notice; "
                     + held)
    if source.note:
        lines.append(f"- {source.note}")
    if source.excluded:
        unused = "; ".join(f"{k} ({v})" for k, v in source.excluded.items())
        lines.append(f"- Configs not used: {unused}.")
    return "\n".join(lines) + "\n"


def licenses_md(manifest: dict) -> str:
    head = ["# Licenses and attribution of the public training data", "",
            ("Generated by `scripts/fetch_public_data.py` from the dataset cards fetched on "
             f"{manifest['generated'][:10]}; the license field of each card is quoted as found at the "
             "recorded revision. The converted pools are not redistributed with this repository; this file "
             "is the attribution for their use in training and for any release of the trained adapter."), ""]
    sections = []
    for source in SOURCES:
        entry = manifest["sources"][source.name]
        if "error" in entry:
            sections.append(f"## {source.repo}\n\n- Not fetched: {entry['error']}\n")
        else:
            sections.append(attribution(source, entry, entry["written"]["samples"]))
    return "\n".join(head + sections)


def cache_report() -> dict:
    """Size of the Hugging Face cache on disk, in total and per dataset repository (sizes only, no paths)."""
    cache = scan_cache_dir()
    datasets = {r.repo_id: r.size_on_disk for r in cache.repos if r.repo_type == "dataset"}
    return {"size_bytes": cache.size_on_disk, "size_gb": round(cache.size_on_disk / 1e9, 3),
            "datasets_bytes": dict(sorted(datasets.items()))}


def fetch_one(source: Source, out: Path, seed: int, tokenizer) -> dict:
    t0 = time.perf_counter()
    entry = card(source)
    paths = download(source, entry["revision"])
    samples, files = convert(source, paths)
    lengths = token_lengths(samples, tokenizer) if tokenizer is not None else None
    picked, account = draw(samples, source.count, seed, no_call_share=source.no_call_share, lengths=lengths)
    chosen = [samples[i] for i in picked]
    path = out / f"{source.name}.jsonl"
    dropped = sum((Counter(f["dropped"]) for f in files), Counter())
    entry.update({"plan_count": source.count, "seed": seed, "configs": list(source.configs),
                  "excluded_configs": source.excluded, "per_file": files,
                  "rows": sum(f["rows"] for f in files), "converted": len(samples),
                  "dropped": dict(dropped.most_common()), "pool": counts(samples), "draw": account,
                  "written": counts(chosen), "path": repo_relative(path),
                  "sha256": write_jsonl(path, chosen)})
    if lengths is not None and picked:
        entry["tokens"] = token_stats([lengths[i] for i in picked])
    entry["seconds"] = round(time.perf_counter() - t0, 1)
    return entry


def summary_line(name: str, entry: dict) -> str:
    pool, drawn, written, tokens = entry["pool"], entry["draw"], entry["written"], entry.get("tokens")
    line = (f"{name}: {entry['repo']}@{entry['revision'][:8]} license={entry['license']} "
            f"rows={entry['rows']} converted={entry['converted']} dropped={sum(entry['dropped'].values())} "
            f"excluded_for_length={drawn['excluded_for_length']} written={written['samples']} "
            f"(no-call {written['no_call']}, share {written['no_call_share']}; pool share "
            f"{pool['no_call_share']})")
    if tokens:
        line += f" tokens max={tokens['max']} p95={tokens['p95']} over={tokens['over_context']}"
    return line + f" [{entry['seconds']}s]"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(OUT), help="directory of the JSONL pools (default: data/raw/public)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--model", default=MODEL,
                    help="model directory for the length filter and token statistics; skipped when absent")
    args = ap.parse_args(argv)
    tokenizer = None
    if Path(args.model).is_dir():
        from transformers import AutoTokenizer  # heavy import, only for the length filter and statistics

        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        print(f"no model directory at {args.model}: no length filter, no token statistics", flush=True)
    manifest: dict = {"generated": datetime.now(UTC).isoformat(timespec="seconds"), "seed": args.seed,
                      "context_tokens": CONTEXT_TOKENS, "model": args.model if tokenizer else None,
                      "sources": {}}
    failed = 0
    for source in SOURCES:
        try:
            entry = fetch_one(source, Path(args.out), args.seed, tokenizer)
        except FETCH_ERRORS as exc:
            failed += 1
            entry = {"repo": source.repo, "plan_count": source.count, "error": f"{type(exc).__name__}: {exc}"}
            print(f"{source.name}: FAILED {entry['error']}", flush=True)
        else:
            print(summary_line(source.name, entry), flush=True)
        manifest["sources"][source.name] = entry
    manifest["hf_cache"] = cache_report()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8", newline="\n")
    (PUBLIC / "LICENSES.md").write_text(licenses_md(manifest), encoding="utf-8", newline="\n")
    print(f"manifest: {PUBLIC / 'manifest.json'}; HF cache {manifest['hf_cache']['size_gb']} GB; "
          f"{failed} source(s) failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

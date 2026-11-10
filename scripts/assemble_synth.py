"""Assemble the synthetic training file from generator runs (research_agent.data.assemble explains the rules).

    uv run --no-sync python scripts/assemble_synth.py runs/m2/synth/train runs/m2/synth/dev runs/m2/synth/test
        [--out data/raw/synth/synth.jsonl] [--manifest data/synth/train-manifest.json]
        [--tokenizer models/Qwen3-1.7B]

Selects the index rows whose split is train, whose verdict accepted them and whose turn ended answered or as
a clarification, turns each transcript into one agent-history record the trainer reads
(research_agent.train.samples.load_records), drops trajectories with a step accepted after a repair or
a think tag in an assistant text (the Qwen3 template would render it as reasoning), repairs scaffold note
evidence that is not quoted from the tool outputs (research_agent.data.evidence_repair; a trajectory whose
note step had no visible text of its paper is dropped, a final answer restating a detail only the old
evidence held is removed), drops duplicates after normalize_text and train records whose family appears among
the dev or test rows of the given runs (pass the dev and test runs for that check; their rows are never
selected), and writes the records in request-id order as canonical JSON lines. The manifest (sorted
keys, two-space indent) records the counts
(rows seen, selected, dropped per reason, kept by scenario and language), what the evidence repair did
(evidence_repair: calls and records repaired, sentences per call, the text fields quoted, the records whose
final answer was removed and those dropped without visible text), the source runs (name, path, the
run manifest's SHA-256, prompt and schema hashes, requests SHA-256), the output file's SHA-256 and size, the
per-scenario mix next to the planned one and, with --tokenizer, the token statistics of the written file
taken the way scripts/train.py takes them (read back with load_records, cut with build_step_samples): the
per-step samples, how many exceed CONTEXT_TOKENS (the trainer drops those) and the statistics of the rest.
The file is staged next to --out and moved into place only after that read-back succeeds, so a record the
trainer refuses leaves nothing behind. Prints one line of counts per group.

Exit 0 on success; 1 on malformed input (a run folder without manifest.json or index.jsonl, an index line
that is not a JSON object or lacks a key, a request id repeated across runs, a missing or malformed
transcript, a record the trainer would refuse); 2 on usage errors (bad arguments, a run directory that does
not exist). Nothing is written on exit 1 or 2.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from research_agent.agent.prompts import PROMPT_HASH, PROMPT_VERSION
from research_agent.data.assemble import (
    DROP_REASONS,
    SELECTED_STATUSES,
    AssembleError,
    assemble,
    token_counts,
    write_records,
)
from research_agent.tools.schemas import CONTEXT_TOKENS, SCHEMA_HASH

OUT = Path("data/raw/synth/synth.jsonl")
MANIFEST = Path("data/synth/train-manifest.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble the synthetic training file from generator runs.")
    parser.add_argument("run_dirs", nargs="+", type=Path,
                        help="generator run folders; dev and test runs are read for the family check only")
    parser.add_argument("--out", type=Path, default=OUT,
                        help=f"the training file (default: {OUT.as_posix()})")
    parser.add_argument("--manifest", type=Path, default=MANIFEST,
                        help=f"the manifest (default: {MANIFEST.as_posix()})")
    parser.add_argument("--tokenizer", default=None,
                        help="a Hugging Face model directory: count the per-step samples and those over "
                             "CONTEXT_TOKENS")
    return parser


def load_tokenizer(path: str):
    from transformers import AutoTokenizer  # heavy import, only with --tokenizer

    return AutoTokenizer.from_pretrained(path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    missing = [str(path) for path in args.run_dirs if not path.is_dir()]
    if missing:
        print(f"not a run directory: {', '.join(missing)}", file=sys.stderr)
        return 2
    # Becomes --out only once the trainer's loader accepts it; the suffix stays, load_records reads by it.
    staged = args.out.with_name(f"{args.out.stem}.staged{args.out.suffix}")
    try:
        result = assemble(args.run_dirs)
        sha256, size = write_records(result.records, staged)
        tokens = None
        if args.tokenizer:
            tokens = token_counts(staged, load_tokenizer(args.tokenizer), CONTEXT_TOKENS)
    except (AssembleError, ValueError) as exc:
        staged.unlink(missing_ok=True)
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    os.replace(staged, args.out)
    counts = result.counts()
    manifest = {
        "generated": datetime.now(UTC).isoformat(timespec="seconds"),
        "schema_hash": SCHEMA_HASH, "prompt_hash": PROMPT_HASH, "prompt_version": PROMPT_VERSION,
        "context_tokens": CONTEXT_TOKENS, "tokenizer": args.tokenizer,
        "selection": {"split": "train", "verdict": "accepted", "statuses": list(SELECTED_STATUSES),
                      "after_repair": "trajectories with a step accepted after a repair are dropped: the "
                                      "record carries no step records, so the trainer could not skip the "
                                      "step",
                      "think_tag": "trajectories whose assistant text holds <think> or </think> are "
                                   "dropped: the Qwen3 template renders the text before </think> as "
                                   "reasoning",
                      "dedup": "user and assistant turns equal after normalize_text (tool calls in "
                               "canonical JSON); the smaller request id is kept",
                      "family_check": "a train family also among the dev or test rows of the given runs",
                      "evidence_repair": "scaffold write_note evidence not quoted from the earlier tool "
                                         "outputs becomes up to 4 sentences quoted from the paper's text "
                                         "in the step before the note; a trajectory without such text is "
                                         "dropped; a final answer sharing a number or a 6-token run with the "
                                         "old evidence that the repaired context lacks is removed"},
        "sources": [run.describe() for run in result.runs],
        "counts": counts, "mix": result.mix(), "evidence_repair": result.evidence_repair(),
        "output": {"path": args.out.as_posix(), "sha256": sha256, "bytes": size,
                   "records": len(result.records)},
        "tokens": tokens,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8", newline="\n")
    dropped = ", ".join(f"{reason} {counts['dropped'][reason]}" for reason in DROP_REASONS)
    print(f"rows seen {counts['rows_seen']}, selected {counts['selected']}, kept {counts['kept']}; "
          f"dropped: {dropped}")
    repaired = result.evidence_repair()
    print(f"evidence repair: {repaired['calls_repaired']} of {repaired['scaffold_calls']} scaffold notes in "
          f"{repaired['records_repaired']} records repaired, {len(repaired['final_answers_removed'])} final "
          f"answers removed, {len(repaired['no_visible_text'])} records dropped without visible text")
    for run in result.runs:
        d = run.describe()
        print(f"source {d['name']} ({d['path']}): {d['rows']} rows {d['by_split']}")
    print("kept by scenario: " + ", ".join(f"{s} {n}" for s, n in counts["by_scenario"].items()))
    print("kept by language: " + ", ".join(f"{lang} {n}" for lang, n in counts["by_lang"].items()))
    print("mix: " + ", ".join(f"{s} {share:.3f} (planned {result.mix()['planned'].get(s, 0):.2f})"
                              for s, share in result.mix()["shares"].items()))
    if tokens:
        print(f"tokens: {tokens['samples']} per-step samples from {tokens['records']} records, "
               f"{tokens['over_context']} over {CONTEXT_TOKENS} (dropped by the trainer); "
              f"min {tokens['min']}, "
              f"median {tokens['median']}, mean {tokens['mean']}, p95 {tokens['p95']}, max {tokens['max']}")
    print(f"{args.out.as_posix()}: {len(result.records)} records, {size} bytes, sha256 {sha256}; "
          f"manifest {args.manifest.as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Build the request records for the synthetic generator (research_agent.data.requests explains the templates,
slot pools, split, dedup and audit, and the incremental builds).

    .venv/Scripts/python.exe scripts/make_requests.py [--out data/synth/requests.jsonl]
        [--manifest <out without .jsonl>-manifest.json] [--seed 42] [--train 1600] [--dev 400] [--test 400]
        [--corpus data/corpus/manifest.json] [--draw-seed N] [--only SPLIT[,SPLIT]]
        [--id-start SPLIT=N ...] [--preload FILE ...] [--mix SCENARIO=SHARE,...]

Reads the corpus manifest (topics, their paper ids, the recording window), splits templates 4/1/1 and topics
with their paper ids 70/15/15 with the seed, fills every split's cells to the planned mix (55/15/10/15/5,
half Chinese and half English), drops exact and near duplicates and train requests too close to a dev or test
request (near and train-vs-eval compare texts with ids, dates and numbers masked), audits the splits for
shared families, topics and paper ids and for templates of different splits that share their wording, and only
then writes --out as canonical JSON lines and --manifest (seed, sizes, counts per split x scenario x language,
dedup counts, both audit results, the corpus manifest's SHA-256). The same inputs give byte-identical files.
It prints the counts per split, scenario and language, the dedup counts, the audit line and the template audit
line. --manifest defaults to the name of --out with -manifest.json for .jsonl (data/synth/requests.jsonl ->
data/synth/requests-manifest.json).

Incremental builds (docs/m7-计划.md §5 and WP-I 1) add requests to registered ones; --seed still fixes the
split of templates, topics and paper ids.
- --draw-seed N: the seed of the draws within a split (templates, topics, slots, the order of the records);
  default --seed.
- --only SPLIT[,SPLIT] (also --only-split): build only these splits; the sizes of the others become 0.
- --id-start SPLIT=N, repeatable: the first id number of a split, so new ids continue after existing ones
  (req-test-0401, ...).
- --preload FILE, repeatable: an existing request file whose requests seed the deduper before generation. The
  exact and the near rule run against them whatever their split, the train-vs-eval rule from every new train
  request to every preloaded dev and test request (a new dev or test request in that band of a preloaded
  train request is kept). They are not written, an id they hold is refused, and the leakage audit and the
  template audit run over the preloaded and the new requests together. --out and --manifest must not name a
  preloaded file.
- --mix SCENARIO=SHARE,...: the mix instead of the planned one (known scenarios, shares of at least 0 that
  sum to 1, a scenario left out gets 0).
With all five at their defaults the files are those of a build without them, byte for byte; otherwise the
manifest also records the draw seed, the splits built, the id starts, each preloaded file (path, SHA-256,
requests per split), the drops against preloaded requests and, as planned_mix, the mix used. With --preload it
also prints how many requests were preloaded and how many drops a preloaded request caused.

Exit 0 when both files are written. Exit 1 when either audit fails (nothing is written) or a cell cannot be
filled after dedup. Exit 2 on usage errors: a negative size, all sizes of the splits built zero, a corpus
manifest that is missing or malformed, a malformed --only, --id-start or --mix, a preloaded file that is
missing or malformed or repeats another's ids, an id a preloaded file already holds, --out or --manifest
naming a preloaded file or each other.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from research_agent.data import requests as req
from research_agent.data.synth import LANGS, SCENARIOS, SPLITS


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build the synthetic request records and their manifest.")
    ap.add_argument("--out", default="data/synth/requests.jsonl", help="the request records (JSONL)")
    ap.add_argument("--manifest", default=None,
                    help="the build manifest (default: --out with -manifest.json for .jsonl)")
    ap.add_argument("--seed", type=int, default=42, help="fixes the split of templates, topics and paper ids")
    for split in SPLITS:
        ap.add_argument(f"--{split}", type=int, default=req.DEFAULT_SIZES[split], metavar="N",
                        help=f"requests in {split} (default {req.DEFAULT_SIZES[split]})")
    ap.add_argument("--corpus", default=req.CORPUS_MANIFEST.as_posix(), help="the corpus manifest")
    ap.add_argument("--draw-seed", type=int, default=None, metavar="N",
                    help="the seed of the draws within a split (default: --seed)")
    ap.add_argument("--only", "--only-split", dest="only", default=None, metavar="SPLIT[,SPLIT]",
                    help="build only these splits (the sizes of the others become 0)")
    ap.add_argument("--id-start", action="append", default=[], metavar="SPLIT=N",
                    help="the first id number of a split (repeatable; default 1)")
    ap.add_argument("--preload", action="append", default=[], metavar="FILE",
                    help="an existing request file the dedup and the audits run against (repeatable)")
    ap.add_argument("--mix", default=None, metavar="SCENARIO=SHARE,...",
                    help="the scenario mix instead of the planned one (shares sum to 1)")
    return ap


def default_manifest(out: str) -> str:
    """<out without .jsonl>-manifest.json beside --out."""
    path = Path(out)
    return str(path.with_name(f"{path.name.removesuffix('.jsonl')}-manifest.json"))


def parse_only(text: str) -> list[str]:
    """--only: split names separated by commas (requests.check_only validates them)."""
    names = [name.strip() for name in text.split(",")]
    if not all(names):
        raise ValueError(f"--only {text!r}: an empty split name")
    return names


def parse_id_starts(values: list[str]) -> dict[str, int]:
    """--id-start SPLIT=N, each split at most once (requests.check_id_start validates the values)."""
    starts: dict[str, int] = {}
    for value in values:
        split, sep, number = value.partition("=")
        split = split.strip()
        if not sep or not split:
            raise ValueError(f"--id-start {value!r}: expected SPLIT=N")
        try:
            start = int(number)
        except ValueError:
            raise ValueError(f"--id-start {value!r}: {number!r} is not an integer") from None
        if split in starts:
            raise ValueError(f"--id-start names {split} twice")
        starts[split] = start
    return starts


def parse_mix(text: str) -> dict[str, float]:
    """--mix SCENARIO=SHARE,...: each scenario at most once (requests.check_mix validates the names and the
    shares)."""
    mix: dict[str, float] = {}
    for item in text.split(","):
        name, sep, share = item.partition("=")
        name = name.strip()
        if not sep or not name:
            raise ValueError(f"--mix {text!r}: expected SCENARIO=SHARE items, got {item!r}")
        if name in mix:
            raise ValueError(f"--mix names {name} twice")
        try:
            mix[name] = float(share)
        except ValueError:
            raise ValueError(f"--mix {text!r}: {share!r} is not a number") from None
    return mix


def same_file(a: str | Path, b: str | Path) -> bool:
    """Whether two paths name one file: os.path.samefile when both exist, else the same absolute path."""
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def print_counts(records: list[dict], splits: tuple[str, ...]) -> None:
    counts = req.counts_by_cell(records)
    for split in splits:
        total = sum(counts[split][s][lang] for s in SCENARIOS for lang in LANGS)
        cells = ", ".join(f"{s} {counts[split][s]['zh']}+{counts[split][s]['en']}" for s in SCENARIOS)
        print(f"{split}: {total} requests (zh+en): {cells}")


def usage(message: str) -> int:
    print(f"usage error: {message}", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:  # argparse exits 2 on a usage error, 0 on --help
        return int(exc.code or 0)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    given = {split: getattr(args, split) for split in SPLITS}
    try:
        only = req.check_only(None if args.only is None else parse_only(args.only))
    except ValueError as exc:
        return usage(str(exc))
    sizes = {split: given[split] if split in only else 0 for split in SPLITS}
    if any(n < 0 for n in given.values()) or not any(sizes.values()):
        return usage(f"sizes must be non-negative and not all zero, got {sizes}")
    try:
        mix = None if args.mix is None else req.check_mix(parse_mix(args.mix))
        id_start = req.check_id_start(parse_id_starts(args.id_start), sizes)
    except ValueError as exc:
        return usage(str(exc))
    try:
        corpus = req.load_corpus(args.corpus)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return usage(f"cannot read the corpus manifest {args.corpus}: {exc}")
    try:
        preloaded = [req.load_preload(path) for path in args.preload]
    except (OSError, ValueError) as exc:
        return usage(f"cannot preload a request file: {exc}")
    manifest_path = args.manifest if args.manifest is not None else default_manifest(args.out)
    if same_file(args.out, manifest_path):
        return usage(f"--out and --manifest name one file: {Path(args.out).as_posix()}")
    for target in (args.out, manifest_path):
        for path in args.preload:
            if same_file(target, path):
                return usage(f"{Path(target).as_posix()} is a preloaded file; write the build elsewhere")
    try:
        req.check_ids(sizes, id_start, preloaded)
    except ValueError as exc:
        return usage(str(exc))
    try:
        build = req.generate(args.seed, sizes, corpus, draw_seed=args.draw_seed, only=only, id_start=id_start,
                             preload=preloaded, mix=mix)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    together = build.preloaded + build.records
    print_counts(build.records, only)
    dedup = build.report["dedup"]
    print(f"dedup: offered {dedup['offered']}, dropped exact {dedup['exact']}, near {dedup['near']}, "
          f"train vs eval {dedup['train_vs_eval']}")
    if preloaded:
        against = build.report["dedup_against_preloaded"]
        files = f"{len(preloaded)} file{'s' if len(preloaded) != 1 else ''}"
        print(f"preloaded {len(build.preloaded)} requests from {files}; of the drops, against a preloaded "
              f"request: exact {against['exact']}, near {against['near']}, "
              f"train vs eval {against['train_vs_eval']}")
    print(req.audit_line(req.audit(together)))
    print(req.template_audit_line(req.template_audit(req.split_templates(args.seed),
                                                     together if preloaded else None)))
    try:
        manifest = req.write_requests(build.records, args.out, manifest_path, build.report,
                                      preloaded=build.preloaded)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"wrote {manifest['requests']} requests to {Path(args.out).as_posix()} "
          f"(sha256 {manifest['requests_sha256'][:12]}), manifest {Path(manifest_path).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

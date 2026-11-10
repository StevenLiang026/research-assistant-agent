"""Build the ACEBench-style development probe (docs/m6-计划.md §3.4 D3, WP4): 150 items from public
records that no training file uses.

Why. ACEBench takes about 50 minutes per candidate and is the held-out benchmark; the round needs a stand-in
of a few minutes that predicts it on every checkpoint without using it. research_agent.eval.ace_probe holds
the item format, the prompt (own wording, one fixed JSON-list format and one fixed sentence) and the scorer.

Records. research_agent.data.public_pool reads the five public sources back from the local Hugging Face cache
at the revisions of data/public/manifest.json, converts them with research_agent.data.public and leaves out
every source id drawn into data/raw/public/<name>.jsonl. Left out too, by (source, source_id): the records of
WP3's files, read through their manifests when present (--exclude-manifest, default
data/public/format-mix-manifest.json and data/public/base-replay-manifest.json: the files named by their
"output" and "prompts" entries, whose SHA-256 must still be the recorded one, and any "source_ids" they
list), and, unless --no-default-excludes, the files data/raw/public/format-mix.jsonl, base-replay.jsonl and
base-replay.prompts.jsonl when they exist. The probe's own source ids are written into its manifest, so that
WP3 can leave them out in turn; the draw uses its own seed (SEED), not WP3's.

Eligibility (a record that fails gets a reason, counted per source). The record opens with a user message
answered by an assistant message that holds tool calls and no text (FIRST_STEP); every number in the calls
is finite (NOT_FINITE); the user text holds none of ACEBench's notations or sentences (ACEBENCH_TEXT); the
longer of the two system texts plus the user text is at most MAX_CHARS characters (TOO_LONG: the
longest prompt then stays under 6,000 tokens, which with ACEBench's 1,200 reply tokens fits the 8,192-token
context run_acebench.py serves); every string and number in the calls occurs in the user text (strings after
normalize_text and casefold, numbers as whole numbers; booleans and nulls are not looked for), so that the
gold calls can be read off the request (UNGROUNDED); and neither the user text nor either system text
shares a 13-gram with a BFCL or ACEBench question (OVERLAP, research_agent.data.overlap, the grams of
scripts/check_benchmark_overlap.py).

Incomplete items are made from eligible records whose first step is one call: among the required parameters
of the called function (its schema's "required", all present in the call), the ones whose value is a string
of at least two characters or an integer and occurs exactly once in the user text (case-insensitive; an ASCII
letter or digit at either end of the value must not continue into the text) are candidates, and one is drawn
with random.Random(f"{seed}/ace-probe/remove/{source}/{source_id}"). Its occurrence is cut from the user text
(with quotes around it), doubled spaces and spaces before punctuation are removed, and the record is used
only when the rest is at least 10 characters, no longer holds the value, still holds the other required
values, and shares no 13-gram with the benchmarks. Otherwise the record is skipped. expected is the function
and the cut parameter.

Draw. The records of zh items come from glaive_zh and deepexi (Chinese dialogues), those of en items from
toolace, glaive_v2 and hermes; when the Chinese sources run short for a kind (deepexi's schemas declare no
required parameters, so only glaive_zh gives incomplete items), the rest of the zh items of that kind come
from the English sources, with the Chinese system text (lang is the language of the system text, as in
format-mix), counted in the manifest as <kind>_from_en_sources; each kind is split evenly over the
languages (zh gets the smaller half of an odd count). Each source's eligible records are shuffled with
random.Random(f"{seed}/ace-probe/{source}"); the incomplete items are drawn first, round robin over the
language's sources, then the normal items, round robin over the records not yet used. Items are written
by kind, then language, ids ace-probe-<kind>-<lang>-<nnn>.

Written: --out, one JSON object per line (json.dumps, UTF-8 without escapes, LF), the same bytes for the same
seed, cache and exclusions; its lock <out>.sha256 (research_agent.eval.lock.write_lock: a hash check, as
scripts/run_ace_probe.py makes before reading it; the probe is a development file, not a test set: no access
log, no opening limit, not in eval.lock's TEST_SETS); --manifest, the counts per kind and language, the source
ids per source, the eligibility and exclusion accounts, the benchmark files checked and the output's SHA-256.
An existing output is overwritten only with --force.

    uv run --no-sync python scripts/build_ace_probe.py [--out data/eval/ace-probe.jsonl] \\
        [--manifest data/eval/ace-probe-manifest.json] [--normal 100] [--incomplete 50] [--seed 1017] \\
        [--force] [--bfcl-dir <bfcl_eval/data>] [--acebench-dir runs/third_party/ACEBench/data_all]

Exit 0 written; 2 usage (an output exists without --force, the benchmark folders are missing); 3 the data do
not meet the rules (too few records, a WP3 file changed since its manifest, an item that fails its check).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

from research_agent.data import public_pool
from research_agent.data.overlap import BenchmarkIndex, BenchmarkMissing, text_ngrams
from research_agent.data.public import SOURCES
from research_agent.eval import ace_probe as ap
from research_agent.eval.lock import lock_path, write_lock
from research_agent.normalize import normalize_text

SEED = 1017  # the probe's own seed (WP3 uses 42)
NORMAL, INCOMPLETE = 100, 50
MAX_CHARS = 6000
MIN_REST = 10  # characters an incomplete user text keeps at least
EN_SOURCES = ("toolace", "glaive_v2", "hermes")
ZH_SOURCES = ("glaive_zh", "deepexi")
OUT = Path("data/eval/ace-probe.jsonl")
MANIFEST = Path("data/eval/ace-probe-manifest.json")
WP3_MANIFESTS = (Path("data/public/format-mix-manifest.json"), Path("data/public/base-replay-manifest.json"))
WP3_FILES = (
    Path("data/raw/public/format-mix.jsonl"),
    Path("data/raw/public/base-replay.jsonl"),
    Path("data/raw/public/base-replay.prompts.jsonl"),
)
BFCL_VENV = Path(".venv-bfcl")
BFCL_LAYOUTS = ("Lib/site-packages/bfcl_eval/data", "lib/python3*/site-packages/bfcl_eval/data")
ACEBENCH_DIR = Path("runs/third_party/ACEBench/data_all")

# Reasons a record is not eligible.
FIRST_STEP = "the first step is not a user request answered by tool calls alone"
NOT_FINITE = "a call holds a non-finite number"
ACEBENCH_TEXT = "the user text holds an ACEBench notation or sentence"
TOO_LONG = f"system and user text over {MAX_CHARS} characters"
UNGROUNDED = "a call value does not occur in the user text"
OVERLAP = "shares a 13-gram with a BFCL or ACEBench question"

Call = tuple[str, dict]
_ASCII_WORD = re.compile(r"[A-Za-z0-9_]")
_QUOTE_PAIRS = {'"': '"', "'": "'", "“": "”", "‘": "’", "「": "」", "《": "》", "`": "`"}


def lang_of(source: str) -> str:
    return "zh" if source in ZH_SOURCES else "en"


def first_step(sample: dict) -> tuple[str, list[Call]] | None:
    """(user text, calls) of a record that opens with a user request answered by tool calls alone."""
    messages = sample["messages"]
    if len(messages) < 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return None
    answer = messages[1]
    if not answer.get("tool_calls") or answer["content"].strip():
        return None
    calls = [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in answer["tool_calls"]]
    return messages[0]["content"], calls


def _norm(text: str) -> str:
    return normalize_text(text).casefold()


def _number_text(value: float) -> str:
    return str(int(value)) if isinstance(value, int) or float(value).is_integer() else repr(value)


def _occurrences(value: str, text: str) -> list[re.Match]:
    """The case-insensitive occurrences of value in text; an ASCII word character at an end of value must not
    continue into the text."""
    pattern = re.escape(value)
    if _ASCII_WORD.match(value[:1]):
        pattern = r"(?<![A-Za-z0-9_])" + pattern
    if _ASCII_WORD.match(value[-1:]):
        pattern += r"(?![A-Za-z0-9_])"
    return list(re.finditer(pattern, text, re.IGNORECASE))


def _leaves(value: object) -> list[object]:
    if isinstance(value, dict):
        return [leaf for v in value.values() for leaf in _leaves(v)]
    if isinstance(value, list):
        return [leaf for v in value for leaf in _leaves(v)]
    return [value]


def _leaf_in(leaf: object, text: str) -> bool:
    if leaf is None or isinstance(leaf, bool):
        return True
    if isinstance(leaf, (int, float)):
        return bool(_occurrences(_number_text(leaf), text))
    if isinstance(leaf, str):
        return _norm(leaf) in _norm(text)
    return False


def grounded(user: str, calls: list[Call]) -> bool:
    """Every string and number of the calls occurs in the user text (module docstring)."""
    return all(_leaf_in(leaf, user) for _, arguments in calls for leaf in _leaves(arguments))


def _overlaps(texts: list[str], index) -> bool:
    if index is None:
        return False
    return any(gram in index.grams for text in texts for gram in text_ngrams(text))


def _finite(calls: list[Call]) -> bool:
    return all(ap._finite(arguments) for _, arguments in calls)


def normal_ineligible(sample: dict, index=None) -> str | None:
    """Why a record cannot give a normal item (module docstring); None when it can."""
    step = first_step(sample)
    if step is None:
        return FIRST_STEP
    user, calls = step
    if not _finite(calls):
        return NOT_FINITE
    if ap.acebench_markers(user):
        return ACEBENCH_TEXT
    systems = [ap.system_prompt(sample["tools"], lang) for lang in ap.LANGS]  # either may be used
    if max(len(system) for system in systems) + len(user) > MAX_CHARS:
        return TOO_LONG
    if not grounded(user, calls):
        return UNGROUNDED
    if _overlaps([user, *systems], index):
        return OVERLAP
    return None


def _schema(sample: dict, name: str) -> dict:
    for tool in sample["tools"]:
        if tool["function"]["name"] == name:
            return tool["function"].get("parameters") or {}
    return {}


def _cut(text: str, match: re.Match) -> str:
    start, end = match.span()
    if start > 0 and end < len(text) and _QUOTE_PAIRS.get(text[start - 1]) == text[end]:
        start, end = start - 1, end + 1
    rest = text[:start] + text[end:]
    rest = re.sub(r"[ \t]{2,}", " ", rest)
    rest = re.sub(r"[ \t]+([,.?!;:，。？！；：])", r"\1", rest)
    return rest.strip()


def make_incomplete(sample: dict, seed: int, index=None) -> tuple[str, str, str] | None:
    """(user text without one required value, function, the cut parameter), or None when the record cannot
    give an incomplete item (module docstring)."""
    step = first_step(sample)
    if step is None or len(step[1]) != 1:
        return None
    user, [(function, arguments)] = step
    required = list(_schema(sample, function).get("required") or [])
    if not required or any(k not in arguments for k in required):
        return None
    if not grounded(user, [(function, {k: arguments[k] for k in required})]):
        return None
    candidates = {}
    for k in required:
        value = arguments[k]
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            continue
        text = value.strip() if isinstance(value, str) else str(value)
        if isinstance(value, str) and (len(text) < 2 or text != value):
            continue
        found = _occurrences(text, user)
        if len(found) == 1:
            candidates[k] = (text, found[0])
    if not candidates:
        return None
    rng = random.Random(f"{seed}/ace-probe/remove/{sample['source']}/{sample['source_id']}")
    k = rng.choice(sorted(candidates))
    text, match = candidates[k]
    rest = _cut(user, match)
    others = {o: arguments[o] for o in required if o != k}
    if (
        len(rest) < MIN_REST
        or _norm(text) in _norm(rest)
        or not grounded(rest, [(function, others)])
        or _overlaps([rest], index)
    ):
        return None
    return rest, function, k


def _item(sample: dict, kind: str, lang: str, user: str, expected: dict) -> dict:
    system = ap.system_prompt(sample["tools"], lang)
    return {
        "id": "",
        "kind": kind,
        "lang": lang,
        "source": sample["source"],
        "source_id": sample["source_id"],
        "system": system,
        "user": user,
        "tools": sample["tools"],
        "expected": expected,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }


def _round_robin(queues: dict[str, list[dict]], names: tuple[str, ...], n: int, used: set, accept) -> list:
    """Up to n (record, accepted value) pairs, one per source in turn, skipping used records."""
    pointers = dict.fromkeys(names, 0)
    picks: list = []
    while len(picks) < n:
        progress = False
        for name in names:
            queue = queues.get(name, [])
            while pointers[name] < len(queue):
                s = queue[pointers[name]]
                pointers[name] += 1
                key = (s["source"], s["source_id"])
                if key in used:
                    continue
                value = accept(s)
                if value is not None:
                    picks.append((s, value))
                    used.add(key)
                    progress = True
                    break
            if len(picks) >= n:
                break
        if not progress:
            break
    return picks


def build(
    pools: dict[str, list[dict]],
    *,
    seed: int,
    normal: int,
    incomplete: int,
    exclude: set[tuple[str, str]],
    index=None,
) -> tuple[list[dict], dict]:
    """The probe's items (module docstring) and an account per source and language."""
    queues: dict[str, list[dict]] = {}
    per_source: dict[str, dict] = {}
    for name in [n for n in SOURCES if n in pools]:
        kept = [s for s in pools[name] if (s["source"], s["source_id"]) not in exclude]
        reasons: dict[str, int] = {}
        eligible = []
        for s in kept:
            reason = normal_ineligible(s, index)
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
            else:
                eligible.append(s)
        random.Random(f"{seed}/ace-probe/{name}").shuffle(eligible)
        queues[name] = eligible
        per_source[name] = {
            "candidates": len(pools[name]),
            "excluded": len(pools[name]) - len(kept),
            "ineligible": dict(sorted(reasons.items())),
            "eligible": len(eligible),
        }
    used: set[tuple[str, str]] = set()
    chosen: dict[tuple[str, str], list[dict]] = {}
    fallback: dict[tuple[str, str], int] = {}
    for kind, count in (("incomplete", incomplete), ("normal", normal)):
        if kind == "incomplete":
            accept = lambda s: make_incomplete(s, seed, index)
        else:
            accept = first_step
        for lang, names in (("zh", ZH_SOURCES), ("en", EN_SOURCES)):
            n = count // 2 if lang == "zh" else count - count // 2
            picks = _round_robin(queues, names, n, used, accept)
            fallback[kind, lang] = 0
            if lang == "zh" and len(picks) < n:  # English-source records with the Chinese system text
                more = _round_robin(queues, EN_SOURCES, n - len(picks), used, accept)
                fallback[kind, lang] = len(more)
                picks += more
            if kind == "incomplete":
                chosen[kind, lang] = [
                    _item(s, kind, lang, user, {"function": function, "missing": [param]})
                    for s, (user, function, param) in picks
                ]
            else:
                chosen[kind, lang] = [
                    _item(s, kind, lang, user, {"calls": [{"name": c, "arguments": a} for c, a in calls]})
                    for s, (user, calls) in picks
                ]
    items = []
    for kind in ap.KINDS:
        for lang in ap.LANGS:
            for number, item in enumerate(chosen[kind, lang], 1):
                item["id"] = f"ace-probe-{kind}-{lang}-{number:03d}"
                items.append(item)
    for name, account in per_source.items():
        account.update(
            {kind: sum(i["source"] == name and i["kind"] == kind for i in items) for kind in ap.KINDS}
        )
    per_language = {
        lang: {
            **{kind: len(chosen[kind, lang]) for kind in ap.KINDS},
            **{f"{kind}_from_en_sources": fallback[kind, lang] for kind in ap.KINDS},
        }
        for lang in ap.LANGS
    }
    return items, {
        "per_source": per_source,
        "per_language": per_language,
        "shortfall": normal + incomplete - len(items),
    }


# ---- exclusions -------------------------------------------------------------------------------------------


def _ids_of(path: Path) -> set[tuple[str, str]]:
    rows = (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return {(r["source"], str(r["source_id"])) for r in rows}


def _listed(value: object) -> set[tuple[str, str]]:
    """source ids a manifest lists: {source: [ids]}, [{source, source_id}] or ["source/id"]."""
    out: set[tuple[str, str]] = set()
    if isinstance(value, dict):
        for source, ids in value.items():
            out |= {(source, str(i)) for i in ids or []}
    elif isinstance(value, list):
        for entry in value:
            if isinstance(entry, dict):
                out.add((entry["source"], str(entry["source_id"])))
            elif isinstance(entry, str) and "/" in entry:
                source, _, sid = entry.partition("/")
                out.add((source, sid))
    return out


def wp3_excluded(manifests: list[Path], extra_files: list[Path]) -> tuple[set[tuple[str, str]], dict]:
    """The (source, source_id) of WP3's records (module docstring) and an account; ValueError when a file has
    another SHA-256 than its manifest records."""
    ids: set[tuple[str, str]] = set()
    account: dict = {"manifests": [], "files": []}
    for manifest in manifests:
        entry: dict = {"path": manifest.as_posix(), "present": manifest.is_file(), "files": [], "listed": 0}
        account["manifests"].append(entry)
        if not manifest.is_file():
            continue
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for key in ("output", "prompts"):
            named = data.get(key)
            if not isinstance(named, dict) or not named.get("path"):
                continue
            path = Path(named["path"])
            if not path.is_file():
                entry["files"].append({"path": path.as_posix(), "records": None})
                continue
            recorded = named.get("sha256")
            if recorded and hashlib.sha256(path.read_bytes()).hexdigest() != recorded:
                raise ValueError(f"{path.as_posix()}: SHA-256 differs from {manifest.as_posix()}")
            found = _ids_of(path)
            ids |= found
            entry["files"].append({"path": path.as_posix(), "records": len(found)})
        listed = _listed(data.get("source_ids"))
        entry["listed"] = len(listed)
        ids |= listed
    for path in extra_files:
        if path.is_file():
            found = _ids_of(path)
            ids |= found
            account["files"].append({"path": path.as_posix(), "records": len(found)})
    account["total"] = len(ids)
    return ids, account


# ---- CLI --------------------------------------------------------------------------------------------------


def load_pools(manifest: str | Path, raw_dir: str | Path) -> tuple[dict[str, list[dict]], dict[str, dict]]:
    """The unused records of every source from the Hugging Face cache, and their accounts (with revision)."""
    entries = public_pool.load_manifest(manifest)["sources"]
    pools, accounts = {}, {}
    for name in SOURCES:
        pools[name], accounts[name] = public_pool.load_unused(name, entries[name], raw_dir)
        accounts[name]["revision"] = entries[name]["revision"]
    return pools, accounts


def load_index(bfcl_dir: Path | None, acebench_dir: Path) -> BenchmarkIndex:
    """The BFCL and ACEBench grams (BenchmarkMissing when a folder is absent or ambiguous)."""
    if bfcl_dir is None:
        found = sorted({p for pattern in BFCL_LAYOUTS for p in BFCL_VENV.glob(pattern) if p.is_dir()})
        if len(found) > 1:
            raise BenchmarkMissing(f"several bfcl_eval data folders under {BFCL_VENV}; pass --bfcl-dir")
        bfcl_dir = found[0] if found else BFCL_VENV / BFCL_LAYOUTS[0]
    return BenchmarkIndex.build(bfcl_dir, acebench_dir)


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def build_manifest(args, items, account, accounts, wp3, index, data: bytes, out: Path) -> dict:
    counts = {
        kind: {
            **{lang: sum(i["kind"] == kind and i["lang"] == lang for i in items) for lang in ap.LANGS},
            "total": sum(i["kind"] == kind for i in items),
        }
        for kind in ap.KINDS
    }
    source_ids = {name: sorted(i["source_id"] for i in items if i["source"] == name) for name in SOURCES}
    per_source = {name: {**accounts.get(name, {}), **entry} for name, entry in account["per_source"].items()}
    normal_calls = [len(i["expected"]["calls"]) for i in items if i["kind"] == "normal"]
    return {
        "script": "scripts/build_ace_probe.py",
        "purpose": "development probe of M6 (docs/m6-计划.md §3.4 D3); not a test set: no access log, no "
        "opening limit; the lock is a hash check",
        "test_material": False,
        "seed": args.seed,
        "requested": {"normal": args.normal, "incomplete": args.incomplete},
        "counts": {**counts, "total": len(items)},
        "format": ap.FORMAT,
        "sentence": ap.SENTENCE_TEMPLATE,
        "system_text": "each function's OpenAI function object as JSON on one line, then the format and "
        "sentence rules (research_agent.eval.ace_probe.system_prompt), zh or en",
        "scoring_rule": ap.SCORING_RULE,
        "shape_rule": ap.SHAPE_RULE,
        "normal_calls": {
            "items_with_parallel_calls": sum(n > 1 for n in normal_calls),
            "calls": sum(normal_calls),
        },
        "per_language": account["per_language"],
        "source_manifest": repo_relative(Path(args.source_manifest)),
        "source_ids": source_ids,
        "per_source": per_source,
        "excluded": {
            "drawn": {name: accounts.get(name, {}).get("drawn", 0) for name in SOURCES},
            "wp3": wp3,
        },
        "overlap": {
            "method": "research_agent.data.overlap.text_ngrams of user and system text against the "
            "user turns of the BFCL and ACEBench questions; overlapping records are not used",
            "benchmark_files": getattr(index, "files", []),
            "benchmark_grams": len(index.grams),
        },
        "output": {
            "path": repo_relative(out),
            "records": len(items),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "lock": repo_relative(lock_path(out)),
        },
    }


def main(argv: list[str] | None = None) -> int:
    ap_ = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap_.add_argument("--out", default=str(OUT))
    ap_.add_argument("--manifest", default=str(MANIFEST))
    ap_.add_argument("--normal", type=int, default=NORMAL)
    ap_.add_argument("--incomplete", type=int, default=INCOMPLETE)
    ap_.add_argument("--seed", type=int, default=SEED)
    ap_.add_argument("--force", action="store_true", help="overwrite existing outputs")
    ap_.add_argument("--source-manifest", default=str(public_pool.MANIFEST))
    ap_.add_argument("--raw-dir", default=str(public_pool.RAW_DIR), help="the drawn files to exclude")
    ap_.add_argument(
        "--exclude-manifest",
        action="append",
        type=Path,
        default=None,
        help="a WP3 manifest whose files' records to leave out (repeatable; default "
        + ", ".join(p.as_posix() for p in WP3_MANIFESTS)
        + ")",
    )
    ap_.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="do not read " + ", ".join(p.as_posix() for p in WP3_FILES) + " directly",
    )
    ap_.add_argument("--bfcl-dir", type=Path, default=None)
    ap_.add_argument("--acebench-dir", type=Path, default=ACEBENCH_DIR)
    args = ap_.parse_args(argv)
    out, manifest_path = Path(args.out), Path(args.manifest)
    existing = [p for p in (out, manifest_path, lock_path(out)) if p.exists()]
    if existing and not args.force:
        print(f"{', '.join(map(str, existing))} exists; pass --force to overwrite", file=sys.stderr)
        return 2
    try:
        index = load_index(args.bfcl_dir, args.acebench_dir)
    except BenchmarkMissing as exc:
        print(f"benchmarks missing: {exc}", file=sys.stderr)
        return 2
    manifests = args.exclude_manifest if args.exclude_manifest is not None else list(WP3_MANIFESTS)
    try:
        exclude, wp3 = wp3_excluded(manifests, [] if args.no_default_excludes else list(WP3_FILES))
    except ValueError as exc:
        print(f"WP3 exclusion: {exc}", file=sys.stderr)
        return 3
    pools, accounts = load_pools(args.source_manifest, args.raw_dir)
    items, account = build(
        pools, seed=args.seed, normal=args.normal, incomplete=args.incomplete, exclude=exclude, index=index
    )
    if account["shortfall"]:
        print(
            f"only {len(items)} items for {args.normal + args.incomplete}: {account['per_language']}",
            file=sys.stderr,
        )
        return 3
    for item in items:
        errors = ap.check_item(item)
        if errors:
            print(f"{item['id']}: {errors}", file=sys.stderr)
            return 3
    data = "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items).encode("utf-8")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    write_lock(out)
    manifest = build_manifest(args, items, account, accounts, wp3, index, data, out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(
        f"{out}: {len(items)} items {manifest['counts']}; excluded WP3 records {wp3['total']}; "
        f"per source { ({k: (v['normal'], v['incomplete']) for k, v in account['per_source'].items()}) }"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

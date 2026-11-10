"""FT4's reasoning sidecars: the FT3 sidecars filtered and re-keyed for the FT4 records, and template
reasoning for every turn an FT4 rule changed or built (docs/m7-本地提升计划.md WP6, 思考侧车).

    uv run --no-sync python scripts/template_reasoning.py --manifest runs/train/ft3/manifest.json
        --records data/raw/synth/synth-m6-ft4.jsonl data/raw/synth/synth-m7-ft4.jsonl
        --originals data/raw/synth/synth-m6.jsonl data/raw/synth/synth-m7.jsonl
        --changes data/synth/ft4-changes-m6.jsonl data/synth/ft4-changes-m7.jsonl
        [--out-dir data/raw/synth] [--report data/raw/synth/reasoning-ft4.report.json] [--force]

Why. A sidecar row names a turn by (record id, k) and carries the hash of the answer it was written for
(research_agent.data.reasoning). scripts/transform_synth.py --ft4 relabels turns, cuts records after them,
copies trajectory prefixes into derived records and builds new records, and writes a changes file that
names every such turn. The FT3 sidecars still name the original records' turns: used as they are, a
relabelled turn's row would be stale and the derived records' turns would have none. This script rewrites
them, and writes the reasoning of the changed and new turns from templates, without a model or an API call.

What it writes into --out-dir:
- <sidecar stem>-ft4.jsonl for every sidecar the training manifest lists (config.data.reasoning.files)
  whose rows name turns of the original records (--originals). Its rows:
  - a row whose turn an FT4 rule relabelled (a changes row of kind relabel) is dropped (dropped_changed);
  - a row whose turn the FT4 records no longer hold (cut away, or its record left out) is dropped
    (dropped_vanished);
  - every other row is kept after its binding is checked: its answer_sha256 must be that of the FT4 turn
    (exit 1 otherwise: the records changed a turn no changes row names). Its context_equal turns false
    when the FT4 record's history before the turn differs from the original's (an earlier query R13
    rewrote), since the reasoning was not written for the context the sample now shows (context_changed);
  - a copied turn (a changes row of kind copy, from its source turn) gets its source turn's row, re-keyed
    to the copy's (id, k), when that row's answer is the copy's (rekeyed); its context_equal as above. That
    holds for the context turns of a derived record (target false: the trainer leaves them out) and for the
    teacher's turn a derived record keeps as its target (target true: R10's "full listing seen -> ask" where
    the teacher already asked), which so trains with the teacher's own reasoning.
  A listed sidecar none of whose rows names a turn of the original records (base-replay-c5's) is left as
  it is and reported as untouched; one that mixes both is refused. A derived record cut from a record R4
  dropped names a source outside the originals (sources_outside_the_originals): no row exists for it.
- reasoning-template-ft4.jsonl: one row of source "template" per turn an FT4 rule relabelled or built (a
  changes row of kind relabel or new, which must be a target): English reasoning of 2 to 4 sentences, per rule
  and variant (template_reasoning), that reasons towards exactly that answer and names every tool it calls.
  Each must pass problems() (English, MIN_SENTENCES to MAX_SENTENCES sentences, at most MAX_WORDS words, every
  called tool named, no think tag) and data.reasoning.check_row; model MODEL; cache_key the SHA-256 of the
  template's identity (MODEL, rule, id, k), which traces the row to its template as a cache key traces a
  model's; context_equal true. A copied turn whose source turn's row is a template row (a query R13
  rewrote, copied into a derived record) gets that row re-keyed here.
- --report (JSON, sorted keys): per sidecar the rows in, kept, dropped (changed, vanished), re-keyed, out,
  with context changed, and the output's SHA-256; the untouched sidecars; the template rows by rule; the
  verification: research_agent.data.reasoning.ReasoningIndex loads the FT4 sidecars and the untouched ones
  together (no turn in two rows) and every turn of the FT4 records looks up without a binding error
  (turns, context turns, the teacher's kept targets and how many of them have reasoning, turns with
  reasoning, rows indexed). It runs on staged copies before anything is written.

The changes rows with target false must be exactly the context turns the records name (the turns below a
derived record's context_turns, research_agent.train.samples), so that the trainer and the changes files
agree on what is trained.

Exit 0 on success; 1 on input it cannot use (a malformed records, changes or sidecar file, a changes row
for a turn the records lack, a relabel or new row that is no target, context turns that are not the rows
with target false, a kept row bound to another answer, a sidecar that mixes original and other records, a
template that fails its checks, a failed verification); 2 on usage errors (a missing input, an output
present without --force). Nothing is written on exit 1 or 2.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

from research_agent.canonical import canonical_json, split_lines
from research_agent.data.reasoning import (
    TEMPLATE,
    ReasoningError,
    ReasoningIndex,
    answer_sha256,
    check_row,
    has_think_tag,
    sidecar_row,
    turns,
    write_sidecar,
)
from research_agent.train.samples import context_turns

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = Path("data/raw/synth")
REPORT = OUT_DIR / "reasoning-ft4.report.json"
TEMPLATE_FILE = "reasoning-template-ft4.jsonl"
MODEL = "template_reasoning-ft4-2026-10-30"
MAX_WORDS, MIN_SENTENCES, MAX_SENTENCES = 120, 2, 4
NOT_ENGLISH, SENTENCES = "not English", f"not {MIN_SENTENCES} to {MAX_SENTENCES} sentences"
TOO_LONG, NOT_NAMED, THINK_TAG = f"over {MAX_WORDS} words", "a called tool is not named", "think tag"
CHANGE_KEYS = ("id", "k", "rule", "kind", "target", "from")
KINDS = ("relabel", "new", "copy")
_CJK = re.compile(r"[㐀-鿿豈-﫿]")  # as scripts/rationalize_reasoning.py's
_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")
_LAST_CJK = re.compile(r"[㐀-鿿豈-﫿](?=[^㐀-鿿豈-﫿]*$)")
_SLOT_WORDS = {"topic": "topic", "paper_id": "paper", "tags": "tag", "date_range": "time range",
               "count": "number of results"}


def _load_transform():
    """scripts/transform_synth.py as a module of its own name (scripts/ is no package): its FT4 helpers."""
    spec = importlib.util.spec_from_file_location("_transform_synth_ft4",
                                                  REPO / "scripts" / "transform_synth.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ts = _load_transform()


class TemplateError(ValueError):
    """Input this script cannot use; the CLI exits 1."""


# ---- the templates -------------------------------------------------------------------------------------


def _call_names(answer: dict) -> list[str]:
    names: list[str] = []
    for c in answer.get("tool_calls") or []:
        name = (c.get("function") or {}).get("name") or ""
        if name not in names:
            names.append(name)
    return names


def _joined(words: list[str]) -> str:
    return " and ".join(words)


def _slots(missing: list[str]) -> str:
    return _joined([_SLOT_WORDS.get(m, m) for m in missing])


def _first_args(answer: dict, name: str) -> dict:
    return next((ts._args(c) or {} for c in answer.get("tool_calls") or [] if ts._name(c) == name), {})


def _relative_key(text: str, lang: str) -> str:
    """The key of the relative date in a request (the longest phrase first: the day before yesterday)."""
    phrases = ts.RELATIVE_PHRASES["zh" if lang == "zh" else "en"]
    for key, phrase in sorted(phrases.items(), key=lambda kv: -len(kv[1])):
        if phrase in text:
            return key
    raise TemplateError(f"no relative date of RELATIVE_PHRASES in the request {text!r}")


def _optional_param(text: str) -> tuple[str, str]:
    """The (tool, optional parameter) a question of R14's EXPLAIN_TEMPLATES asks about."""
    found = [(tool, param) for tool, param in ts.optional_params()
             if re.search(rf"(?<![\w]){re.escape(tool)}(?![\w])", text)
             and re.search(rf"(?<![\w]){re.escape(param)}(?![\w])", text)]
    if len(found) != 1:
        raise TemplateError(f"the question {text!r} names {len(found)} optional parameters, not one")
    return found[0]


def _r8(history: list[dict], names: list[str]) -> str:
    calls, replies = ts._last_round(history)
    failed = [ts._name(c) for c, r in zip(calls, replies, strict=True) if ts._error(r) in ts.TRANSIENT]
    codes = sorted({ts._error(r) for r in replies if ts._error(r) in ts.TRANSIENT})
    plural = "s" if len(failed) > 1 else ""
    text = (f"The previous {_joined(names)} call{plural} returned {_joined(codes)}, a temporary error on "
            "the tool's side rather than a problem with the request.")
    if len(failed) < len(calls):
        text += " The other call of that step returned its results, so it does not need to run again."
    return text + f" I will repeat exactly the same {_joined(names)} call{plural} with the same arguments."


def _r9(history: list[dict], names: list[str]) -> str:
    if ts.ASK in names:  # a remembered note, two or more listed notes with its topic's tag
        return ("Several of the notes listed in the tool results carry the topic of the request, and nothing "
                "in it says which of them the user remembers. I will call ask_clarification and name those "
                "notes so that the user can pick one.")
    searches = ts.s4b(history)
    if searches is not None:
        return (f"I have already searched arXiv {searches} times for this request. Another search is "
                "unlikely to turn up anything new, so I will stop and report honestly what the searches "
                "returned.")
    return ("The user's notes are already listed in the tool results, so another note query is not needed "
            "for this request. I will answer from the notes listed instead of calling another tool.")


def _relative(record: dict, answer: dict, names: list[str]) -> str:
    today = ts._today_of(record)
    key = _relative_key(ts._first_user(record), record.get("lang") or "zh")
    start, end = ts.relative_range(key, today)
    span = start if start == end else f"{start} to {end}"
    text = f"Today is {today}, so {ts.RELATIVE_PHRASES['en'][key]} means {span}."
    if "list_notes" in names:
        args = _first_args(answer, "list_notes")
        tag = f" and tag={args['tag']}" if args.get("tag") else ""
        keep = f" and keep only the notes up to {end} in the answer" if end < today else ""
        return text + f" I will call list_notes with since={args.get('since')}{tag}{keep}."
    args = _first_args(answer, "search_arxiv")
    until = f" and date_to={args['date_to']}" if args.get("date_to") else ""
    order = f", sort_by={args['sort_by']}" if args.get("sort_by") else ""
    return text + (f" I will call search_arxiv for {args.get('query')} with "
                   f"date_from={args.get('date_from')}{until}{order}.")


def template_reasoning(record: dict, k: int, rule: str) -> str:
    """The template reasoning of turn k of an FT4 record whose turn rule changed or built (module
    docstring): the variant is read from the answer and its history, as transform_synth built them."""
    messages = record["messages"]
    i = ts._turn_positions(messages)[k]
    history, answer = messages[:i], messages[i]
    names = _call_names(answer)
    if rule == "R8":
        return _r8(history, names)
    if rule == "R9":
        return _r9(history, names)
    if rule == "R10":
        missing = _first_args(answer, ts.ASK).get("missing") or []
        return (f"The results so far do not tell me which {_slots(missing)} the user means. I will say "
                f"what I found and call ask_clarification for the missing {_slots(missing)}.")
    if rule == "R11" and ts.ASK in names:
        missing = _first_args(answer, ts.ASK).get("missing") or []
        return (f"The request refers to a {_slots(missing)} it never names, and searching my notes cannot "
                "tell me which one the user means. I will call ask_clarification right away instead of "
                "searching first.")
    if rule == "R11":
        return _relative(record, answer, names)
    if rule == "R13" and "search_notes" not in names:  # the glossary hook's R7-style record
        query = _first_args(answer, "search_arxiv").get("query")
        return ("The user names a technical term in Chinese, and arXiv is searched in English. I will call "
                f"search_arxiv with the English term {query}.")
    if rule == "R13":  # the term R13 appended to the (first) query with Chinese characters
        queries = [str((ts._args(c) or {}).get("query") or "") for c in answer.get("tool_calls") or []
                   if ts._name(c) == "search_notes"]
        query = next((q for q in queries if _CJK.search(q)), queries[0] if queries else "")
        match = _LAST_CJK.search(query)
        term = query[match.end():].strip() if match else query
        return (f"The user names the topic in Chinese, and its standard English term is {term}. I will call "
                f"{_joined(names)} with a query that keeps the Chinese name and adds {term}.")
    if rule == "R14" and "search_arxiv" in names:
        query = _first_args(answer, "search_arxiv").get("query")
        return (f"The user asks about the top paper on {query}, and the topic is named, so nothing is "
                "missing and there is no need to ask which paper is meant. I will call search_arxiv with "
                f"the query {query} to find it.")
    if rule == "R14":
        tool, param = _optional_param(ts._first_user(record))
        return (f"The user asks whether {param} of {tool} can be left out. The tool description lists it "
                "as optional, so I can answer from the schema without calling any tool.")
    raise TemplateError(f"{record.get('id')} turn {k}: no template for rule {rule!r}")


def problems(text: str, answer: dict) -> list[str]:
    """What is wrong with a template reasoning for answer (module docstring); [] when it passes."""
    found = []
    if _CJK.search(text):
        found.append(NOT_ENGLISH)
    if not MIN_SENTENCES <= len(_SENTENCE_END.findall(text.strip())) <= MAX_SENTENCES:
        found.append(SENTENCES)
    if len(text.split()) > MAX_WORDS:
        found.append(TOO_LONG)
    if any(name not in text for name in _call_names(answer)):
        found.append(NOT_NAMED)
    if has_think_tag(text):
        found.append(THINK_TAG)
    return found


# ---- inputs --------------------------------------------------------------------------------------------


def _jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(split_lines(path.read_text(encoding="utf-8")), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise TemplateError(f"{path}: line {number} is not JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise TemplateError(f"{path}: line {number} is not a JSON object")
        rows.append(value)
    return rows


def _records(paths: list[Path]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in paths:
        for record in _jsonl(path):
            if not isinstance(record.get("messages"), list) or record.get("id") in out:
                raise TemplateError(f"{path}: record {record.get('id')!r} has no messages or comes twice")
            out[str(record["id"])] = record
    return out


def _turn_count(record: dict) -> int:
    return len(turns(record["messages"]))


def _source_fits(row: dict, originals: dict[str, dict]) -> bool | None:
    """Whether a changes row's source fits it: a relabel's is its own turn of an original record; a new
    record's (req-ft4-NNNN) is none; a derived record's (<source id>-ft4...) is a turn of that record, which
    is an original or a record outside them (one R4 dropped: None)."""
    source = row["from"]
    if row["kind"] == "relabel":
        return source == [row["id"], row["k"]] and row["id"] in originals
    if source is None:
        return row["kind"] == "new" and str(row["id"]).startswith("req-ft4-")
    if not (isinstance(source, list) and len(source) == 2 and str(row["id"]).startswith(f"{source[0]}-ft4")):
        return False
    if source[0] not in originals:
        return None
    return isinstance(source[1], int) and 0 <= source[1] < _turn_count(originals[source[0]])


def _changes(paths: list[Path], records: dict[str, dict],
             originals: dict[str, dict]) -> tuple[list[dict], int]:
    """The changes rows, each checked (its keys, its kind, its turn among the FT4 records' turns, its source:
    _source_fits), and how many sources lie outside the originals."""
    rows, outside = [], 0
    for path in paths:
        for row in _jsonl(path):
            where = f"{path}: {row.get('id')} turn {row.get('k')}"
            if set(row) != set(CHANGE_KEYS) or row["kind"] not in KINDS or type(row["target"]) is not bool \
                    or (row["kind"] != "copy" and not row["target"]):
                raise TemplateError(f"{where}: not a changes row ({', '.join(CHANGE_KEYS)}; kinds {KINDS}; "
                                    "a relabel or new turn is a target)")
            record = records.get(row["id"])
            if record is None or not isinstance(row["k"], int) or not 0 <= row["k"] < _turn_count(record):
                raise TemplateError(f"{where}: the FT4 records have no such turn")
            fits = _source_fits(row, originals)
            if fits is False:
                raise TemplateError(f"{where}: its source {row['from']} does not fit a {row['kind']} turn")
            outside += fits is None
            rows.append(row)
    keys = [(r["id"], r["k"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise TemplateError("a turn has two changes rows")
    context = {(rid, k) for rid, record in records.items() for k in range(context_turns(record))}
    not_target = {(r["id"], r["k"]) for r in rows if not r["target"]}
    if context != not_target:
        raise TemplateError(f"the records' context turns (context_turns) are not the changes rows with "
                            f"target false: {sorted(context ^ not_target)[:5]}")
    return rows, outside


def _sidecar_rows(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(split_lines(path.read_text(encoding="utf-8")), 1):
        if line.strip():
            try:
                rows.append(check_row(json.loads(line), f"{path}: line {number}"))
            except json.JSONDecodeError as exc:
                raise TemplateError(f"{path}: line {number} is not JSON: {exc}") from exc
    return rows


# ---- the rewriting -------------------------------------------------------------------------------------


def _turn(record: dict, k: int) -> tuple[list[dict], dict] | None:
    positions = [i for _, i, _ in turns(record["messages"])]
    if not 0 <= k < len(positions):
        return None
    return record["messages"][:positions[k]], record["messages"][positions[k]]


def _same_context(record: dict, k: int, original: dict, ok: int) -> bool:
    mine, theirs = _turn(record, k), _turn(original, ok)
    return mine is not None and theirs is not None and mine[0] == theirs[0]


def _filtered(path: Path, rows: list[dict], records: dict[str, dict], originals: dict[str, dict],
              relabelled: set, copies: dict, served: set) -> dict:
    """One processed sidecar: its rows kept and re-keyed (module docstring) and the counts."""
    counts, out = Counter(rows_in=len(rows)), []
    for row in rows:
        key = (row["id"], row["k"])
        if key in relabelled:
            counts["dropped_changed"] += 1
            continue
        turn = _turn(records[row["id"]], row["k"]) if row["id"] in records else None
        if turn is None:
            counts["dropped_vanished"] += 1
            continue
        if row["answer_sha256"] != answer_sha256(turn[1]):
            raise TemplateError(f"{path}: the row of {key} was written for another answer than the FT4 "
                                "record's turn, and no changes row names that turn")
        same = _same_context(records[row["id"]], row["k"], originals[row["id"]], row["k"])
        counts["context_changed"] += row["context_equal"] and not same
        out.append({**row, "context_equal": row["context_equal"] and same})
        counts["kept"] += 1
    by_key = {(row["id"], row["k"]): row for row in rows}
    for (cid, ck), source in sorted(copies.items()):
        row = by_key.get(source)
        if row is None or row["answer_sha256"] != answer_sha256(_turn(records[cid], ck)[1]):
            continue
        same = _same_context(records[cid], ck, originals[source[0]], source[1])
        out.append({**row, "id": cid, "k": ck, "context_equal": row["context_equal"] and same})
        served.add((cid, ck))
        counts["rekeyed"] += 1
    return {"path": path, "rows": out, "counts": counts}


def rewrite(sidecars: list[tuple[Path, list[dict]]], records: dict[str, dict], originals: dict[str, dict],
            changes: list[dict]) -> tuple[list[dict], list[dict], list[dict], dict]:
    """The FT4 sidecars: per processed sidecar (path, rows, counts), the untouched ones (path, rows), the
    template rows and their count by rule (module docstring)."""
    relabelled = {(c["id"], c["k"]) for c in changes if c["kind"] == "relabel"}
    copies = {(c["id"], c["k"]): tuple(c["from"]) for c in changes if c["kind"] == "copy"}
    processed, untouched, served = [], [], set()
    for path, rows in sidecars:
        known = sum(row["id"] in originals for row in rows)
        if known == 0:
            untouched.append({"path": path.as_posix(), "rows": len(rows)})
        elif known != len(rows):
            raise TemplateError(f"{path}: {known} of its {len(rows)} rows name turns of the original "
                                "records; a sidecar must belong to them or to other records")
        else:
            processed.append(_filtered(path, rows, records, originals, relabelled, copies, served))
    template, by_rule = [], Counter()
    for change in sorted((c for c in changes if c["kind"] != "copy"), key=lambda c: (c["id"], c["k"])):
        record = records[change["id"]]
        answer = _turn(record, change["k"])[1]
        text = template_reasoning(record, change["k"], change["rule"])
        found = problems(text, answer)
        if found:
            raise TemplateError(f"{change['id']} turn {change['k']}: the {change['rule']} template fails "
                                f"its checks ({'; '.join(found)}): {text!r}")
        identity = {"model": MODEL, "rule": change["rule"], "id": change["id"], "k": change["k"]}
        key = hashlib.sha256(canonical_json(identity).encode("utf-8")).hexdigest()
        template.append(sidecar_row(change["id"], change["k"], answer, source=TEMPLATE, reasoning=text,
                                    model=MODEL, cache_key=key, context_equal=True))
        by_rule[change["rule"]] += 1
    by_template = {(row["id"], row["k"]): row for row in template}
    for (cid, ck), source in sorted(copies.items()):  # a copy of a turn whose row is now a template row
        row = by_template.get(source)
        if (cid, ck) not in served and row is not None and \
                row["answer_sha256"] == answer_sha256(_turn(records[cid], ck)[1]):
            template.append({**row, "id": cid, "k": ck})
    return processed, untouched, template, dict(sorted(by_rule.items()))


def verify(paths: list[Path], records: dict[str, dict], kept: set) -> dict:
    """ReasoningIndex over the sidecar files (no turn in two rows) and a lookup of every turn of the
    records (no binding error): the turns, the context turns, the teacher's kept targets (kept: their (id,
    k)) and how many of them have reasoning, the turns with reasoning and the rows indexed."""
    index = ReasoningIndex(paths)
    total = found = context = kept_found = 0
    for rid, record in records.items():
        skip = context_turns(record)
        for k, _, answer in turns(record["messages"]):
            total += 1
            context += k < skip
            text = index.lookup(rid, k, answer)
            found += text is not None
            kept_found += (rid, k) in kept and text is not None
    return {"turns": total, "context_turns": context, "copy_targets": len(kept),
            "copy_targets_with_reasoning": kept_found, "with_reasoning": found, "index_rows": len(index)}


# ---- CLI -----------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write FT4's reasoning sidecars (the FT3 ones rewritten "
                                                 "and template reasoning for FT4's turns).")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="the training manifest whose config.data.reasoning lists the sidecars")
    parser.add_argument("--records", type=Path, nargs="+", required=True, help="the FT4 record files")
    parser.add_argument("--originals", type=Path, nargs="+", required=True,
                        help="the record files the FT4 records were made from (the sidecars' records)")
    parser.add_argument("--changes", type=Path, nargs="+", required=True, help="FT4's changes files")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help=f"default: {OUT_DIR.as_posix()}")
    parser.add_argument("--report", type=Path, default=REPORT, help=f"default: {REPORT.as_posix()}")
    parser.add_argument("--force", action="store_true", help="replace the outputs")
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _staged(processed: list[dict], template: list[dict], untouched: list[dict], records: dict[str, dict],
            kept: set) -> tuple[dict[str, bytes], dict]:
    """The output files' bytes by name, written and verified in a staging folder first."""
    with tempfile.TemporaryDirectory() as staging:
        paths = []
        for entry in processed:
            paths.append(Path(staging) / f"{entry['path'].stem}-ft4.jsonl")
            write_sidecar(entry["rows"], paths[-1])
        paths.append(Path(staging) / TEMPLATE_FILE)
        write_sidecar(template, paths[-1])
        verification = verify([*paths, *(Path(u["path"]) for u in untouched)], records, kept)
        return {path.name: path.read_bytes() for path in paths}, verification


def _print(report: dict) -> None:
    for s in report["sidecars"]:
        print(f"{s['path']}: {s['rows_in']} rows, kept {s['kept']}, dropped {s['dropped_changed']} changed "
              f"and {s['dropped_vanished']} vanished, re-keyed {s['rekeyed']}, out {s['rows_out']} (context "
              f"changed {s['context_changed']}) -> {s['out']}")
    for u in report["untouched"]:
        print(f"{u['path']}: untouched ({u['rows']} rows of other records)")
    t, v = report["template"], report["verification"]
    print(f"template: {t['rows']} rows {t['by_rule']} -> {t['path']}")
    print(f"verified: {v['turns']} turns ({v['context_turns']} context, {v['copy_targets']} the teacher's "
          f"kept targets, {v['copy_targets_with_reasoning']} of them with reasoning), {v['with_reasoning']} "
          f"with reasoning, {v['index_rows']} rows indexed")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = [args.manifest, *args.records, *args.originals, *args.changes]
    missing = [str(p) for p in inputs if not p.is_file()]
    if missing:
        print(f"missing input: {', '.join(missing)}", file=sys.stderr)
        return 2
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        listed = [Path(f["path"]) for f in manifest["config"]["data"]["reasoning"]["files"]]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"FAIL: {args.manifest} lists no reasoning sidecars (config.data.reasoning.files): {exc!r}",
              file=sys.stderr)
        return 1
    missing = [str(p) for p in listed if not p.is_file()]
    if missing:
        print(f"missing sidecar: {', '.join(missing)}", file=sys.stderr)
        return 2
    outputs = [args.out_dir / f"{p.stem}-ft4.jsonl" for p in listed]
    present = [str(p) for p in (*outputs, args.out_dir / TEMPLATE_FILE, args.report) if p.exists()]
    if present and not args.force:
        print(f"{', '.join(present)} exists; pass --force to replace it", file=sys.stderr)
        return 2
    try:
        records, originals = _records(args.records), _records(args.originals)
        changes, outside = _changes(args.changes, records, originals)
        sidecars = [(path, _sidecar_rows(path)) for path in listed]
        processed, untouched, template, by_rule = rewrite(sidecars, records, originals, changes)
        kept = {(c["id"], c["k"]) for c in changes if c["kind"] == "copy" and c["target"]}
        data, verification = _staged(processed, template, untouched, records, kept)
    except (TemplateError, ReasoningError, ValueError, KeyError, OSError) as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_sidecars = []
    for entry in processed:
        out = args.out_dir / f"{entry['path'].stem}-ft4.jsonl"
        out.write_bytes(data[out.name])
        c = entry["counts"]
        report_sidecars.append({"path": entry["path"].as_posix(), "out": out.as_posix(),
                                "rows_in": c["rows_in"], "kept": c["kept"],
                                "dropped_changed": c["dropped_changed"],
                                "dropped_vanished": c["dropped_vanished"], "rekeyed": c["rekeyed"],
                                "rows_out": len(entry["rows"]), "context_changed": c["context_changed"],
                                "sha256": hashlib.sha256(data[out.name]).hexdigest()})
    template_path = args.out_dir / TEMPLATE_FILE
    template_path.write_bytes(data[TEMPLATE_FILE])
    report = {
        "script": "scripts/template_reasoning.py", "model": MODEL,
        "inputs": {name: [{"path": p.as_posix(), "sha256": _sha256(p)} for p in paths] for name, paths in
                   (("manifest", [args.manifest]), ("records", args.records), ("originals", args.originals),
                    ("changes", args.changes))},
        "changes": {"rows": len(changes), "sources_outside_the_originals": outside},
        "sidecars": report_sidecars, "untouched": untouched,
        "template": {"path": template_path.as_posix(), "rows": len(template), "by_rule": by_rule,
                     "sha256": hashlib.sha256(data[TEMPLATE_FILE]).hexdigest()},
        "verification": verification,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8", newline="\n")
    _print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())

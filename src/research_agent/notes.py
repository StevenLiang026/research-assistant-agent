"""The notes library behind search_notes, write_note and list_notes (docs/plan.md, 架构).

Contract.

render_note(paper, args, evidence, today) renders one entry of the reading-notes template (the block that goes
under the "记录" heading of the notes file), line for line:

    ### <today> <title>
    <blank>
    - 来源:https://arxiv.org/abs/<id>;<published>;论文
    - 与本周所学的关系:<args[RELATION_FIELD] when the write_note schema has that field, else empty>
    - Summary (about 300 words)
      - Problem: <draft: args["problem"]; scaffold: empty>
      - Method: ...
      - Evidence: ...
      - Limitations: ...
    - 证据句(原文)              <- scaffold only, followed by one "  - <sentence>" line per evidence sentence
    - 新术语:<args["new_terms"], comma-separated>
    - 待解术语:<args["open_terms"], comma-separated>

An empty field ends at its colon (no trailing space); one-line values have their whitespace collapsed, so no
value can break the entry's line structure; the entry ends with exactly one newline and has no blank line
other than the one after the heading. import_entries parses such entries (and the owner's hand-written ones,
empty fields allowed) back into the same fields.

pick_evidence(text, k, keywords) chooses up to k evidence sentences of text: sentences are split at English
(. ! ? followed by whitespace, except after a few abbreviations) and Chinese (。!?;) sentence ends; keyword
sentences (any keyword occurring case-insensitively after normalize_text) are preferred, then the earliest;
the chosen ones come back in text order. A sentence is verbatim: when its span in text has no line break it is
returned exactly as it stands (a substring of text); when it has, its whitespace runs are collapsed to single
spaces, and it is then found by " ".join(text.split()).find(sentence) — locate_evidence applies exactly these
two lookups.

NoteStore(root) keeps one Markdown file per note under root, named "<arxiv_id>-<n>.md" with n counting the
notes of that paper from 1, and root/"notes.sqlite" with a notes table (note_id, arxiv_id, title, path,
created, tags) plus an FTS5 table over title and body (searchable_body: the entry's content without the
template's labels, which every note shares) for BM25 search. CJK characters are indexed one per token, so a
Chinese term matches as a phrase of characters; Latin words match as words. A query is the AND of its
content terms (STOPWORDS dropped, CJK runs split at CJK_PARTICLES); scores are the negated bm25 rank,
unrounded, higher better. The arxiv_id must be canonical (normalize_arxiv_id's fixed point), else ValueError
before anything is written; the database rows are committed before the file appears. load_entries seeds a
store from import_entries' entries (the owner's existing notes). Nothing depends on the clock: note ids,
file names and the created date come from the writes and the today passed in, so the same sequence of writes
gives the same store. The constructor raises RuntimeError when sqlite3 lacks FTS5.

note_tools(store, papers, today) adapts the store to the three tools, keeping the output shapes of the
stand-ins in data/synth.default_note_tools: write_note answers {"note_id", "arxiv_id", "path"} (NOT_FOUND
with {} when papers(arxiv_id) is None, INVALID_ARGUMENTS with {} for a call without a paper or a mode), and
search_notes and list_notes answer a list, an empty one without an error code, so an empty answer never uses
up the turn's single relax.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

from .agent.types import ToolFn, ToolOutcome
from .canonical import canonical_json
from .normalize import normalize_arxiv_id, normalize_text
from .tools import schemas
from .tools.adapters import capped

ABS_URL = "https://arxiv.org/abs/"
PAPER_TYPE = "论文"
SUMMARY_HEADING = "Summary (about 300 words)"
EVIDENCE_HEADING = "证据句(原文)"
PARTS = [("problem", "Problem"), ("method", "Method"), ("evidence", "Evidence"),
         ("limitations", "Limitations")]
EVIDENCE_SENTENCES = 4  # one per Summary part
DATABASE = "notes.sqlite"
TITLE_WEIGHT = 2.0  # BM25 column weights: title, body
BODY_WEIGHT = 1.0
LIST_LIMIT_MAX = 20  # list_notes' schema maximum

# Optional write_note fields: used only while the schema defines them (it defines neither on 2026-10-13).
RELATION_FIELD = "relation"  # 与本周所学的关系
EXCERPT_FIELD = "excerpt"  # a full-text excerpt to pick evidence from, besides the abstract


def _write_note_properties() -> dict:
    tool = next(t for t in schemas.TOOLS if t["function"]["name"] == "write_note")
    return tool["function"]["parameters"]["properties"]


# ---- rendering ---------------------------------------------------------------------------------------------


def _one_line(value: object) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _terms(value: object) -> str:
    if isinstance(value, list):
        return ", ".join(term for term in (_one_line(v) for v in value) if term)
    return _one_line(value)


def _field(label: str, value: str, space: bool = False) -> str:
    """"label:" with the value after the colon (one space before it when space), nothing when empty."""
    if not value:
        return f"{label}:"
    return f"{label}: {value}" if space else f"{label}:{value}"


def render_note(paper: dict, args: dict, evidence: list[str], today: str) -> str:
    """The owner's template entry for paper (arxiv_id, title, published) on the day today; see the module
    docstring for the line layout. args carries mode ("scaffold" by default), the four parts (used in draft),
    new_terms and open_terms; evidence is listed in scaffold mode only, each sentence's whitespace
    collapsed."""
    mode = args.get("mode", schemas.DEFAULTS["write_note"]["mode"])
    relation = args.get(RELATION_FIELD) if RELATION_FIELD in _write_note_properties() else ""
    entry = {"today": today, "title": paper.get("title", ""), "link": f"{ABS_URL}{paper['arxiv_id']}",
             "published": paper.get("published", ""), "type": PAPER_TYPE, "relation": relation,
             **{key: args.get(key) if mode == "draft" else "" for key, _ in PARTS},
             "evidence_sentences": evidence if mode != "draft" else [],
             "new_terms": args.get("new_terms"), "open_terms": args.get("open_terms")}
    return render_entry(entry, evidence_block=mode != "draft")


def render_entry(entry: dict, evidence_block: bool | None = None) -> str:
    """One template entry from the fields import_entries reads (today, title, link, published, type,
    relation, the four parts, evidence_sentences, new_terms, open_terms; missing ones empty). Every value has
    its whitespace collapsed. The 证据句 block is written when evidence_block is True; None writes it when
    there are evidence sentences or no part is filled in (an entry as scaffold mode leaves it), so
    import_entries(render_entry(e)) gives e back for an entry import_entries read from a rendered note."""
    sentences = [s for s in (_one_line(v) for v in entry.get("evidence_sentences") or []) if s]
    parts = {key: _one_line(entry.get(key)) for key, _ in PARTS}
    if evidence_block is None:
        evidence_block = bool(sentences) or not any(parts.values())
    source_parts = [_one_line(entry.get(key)) for key in ("link", "published", "type")]
    source = ";".join(source_parts) if any(source_parts) else ""
    lines = [f"### {_one_line(entry.get('today'))} {_one_line(entry.get('title'))}", "",
             _field("- 来源", source), _field("- 与本周所学的关系", _one_line(entry.get("relation"))),
             f"- {SUMMARY_HEADING}"]
    lines.extend(_field(f"  - {label}", parts[key], space=True) for key, label in PARTS)
    if evidence_block:
        lines.append(f"- {EVIDENCE_HEADING}")
        lines.extend(f"  - {sentence}" for sentence in sentences)
    lines.append(_field("- 新术语", _terms(entry.get("new_terms"))))
    lines.append(_field("- 待解术语", _terms(entry.get("open_terms"))))
    return "\n".join(lines) + "\n"


# ---- evidence ----------------------------------------------------------------------------------------------

_SENTENCE_END = re.compile(r"[.!?]+[\"')\]]*(?=\s)|[。!?;]+[”’」』)]*")
_ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "etc.", "vs.", "fig.", "eq.", "cf.", "approx.")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) spans of text's sentences without surrounding whitespace; empty ones are dropped."""
    spans, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        end = match.end()
        if text[start:end].casefold().endswith(_ABBREVIATIONS) and match.group().startswith("."):
            continue
        spans.append((start, end))
        start = end
    spans.append((start, len(text)))
    out = []
    for begin, end in spans:
        piece = text[begin:end]
        stripped = piece.strip()
        if stripped:
            begin += len(piece) - len(piece.lstrip())
            out.append((begin, begin + len(stripped)))
    return out


def _as_returned(sentence: str) -> str:
    return " ".join(sentence.split()) if ("\n" in sentence or "\r" in sentence) else sentence


def locate_evidence(text: str, sentence: str) -> int:
    """Where pick_evidence's sentence stands: text.find(sentence), else " ".join(text.split()).find(sentence)
    (the position in the collapsed text); -1 when it is in neither."""
    position = text.find(sentence)
    return position if position >= 0 else " ".join(text.split()).find(sentence)


def pick_evidence(text: str, k: int, keywords: list[str]) -> list[str]:
    """Up to k sentences of text, keyword sentences first, then the earliest, returned in text order; see the
    module docstring for the verbatim rule. Deterministic; k <= 0 gives []."""
    if k <= 0 or not text:
        return []
    wanted = [w for w in (normalize_text(kw).casefold() for kw in keywords) if w]
    sentences = [_as_returned(text[b:e]) for b, e in split_sentences(text)]
    sentences = [s for s in sentences if any(ch.isalnum() for ch in s)]

    def has_keyword(sentence: str) -> bool:
        folded = normalize_text(sentence).casefold()
        return any(w in folded for w in wanted)

    ranked = sorted(range(len(sentences)), key=lambda i: (not has_keyword(sentences[i]), i))
    return [sentences[i] for i in sorted(ranked[:k])]


# ---- importing the owner's entries -------------------------------------------------------------------------

_HEADING = re.compile(r"^### (\d{4}-\d{2}-\d{2})[ \t]*(.*)$")
_TERM_SPLIT = re.compile(r"[,,、;;]")
_LABELS = {"来源": "source", "与本周所学的关系": "relation", "新术语": "new_terms", "待解术语": "open_terms"}
_PART_KEYS = {label: key for key, label in PARTS}


def _split_terms(value: str) -> list[str]:
    return [term.strip() for term in _TERM_SPLIT.split(value) if term.strip()]


def _label_value(item: str) -> tuple[str, str]:
    """("label", "value") of "label:value" (ASCII or full-width colon); ("item", "") without a colon."""
    parts = re.split(r"[::]", item, maxsplit=1)
    return parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""


def import_entries(markdown: str) -> list[dict]:
    """The "### YYYY-MM-DD title" entries of the owner's notes file, in file order, each as a dict with
    today, title, arxiv_id (None when the source has no arXiv link), link, published, type, relation, the four
    parts (problem, method, evidence, limitations), evidence_sentences, new_terms and open_terms. Missing or
    empty fields are "" (lists: []); an entry ends at the next "#" heading or the end of the text."""
    entries: list[dict] = []
    current: dict | None = None
    block = ""
    for line in markdown.replace("\r\n", "\n").split("\n"):
        heading = _HEADING.match(line)
        if heading:
            current = {"today": heading.group(1), "title": heading.group(2).strip(), "arxiv_id": None,
                       "link": "", "published": "", "type": "", "relation": "",
                       **{key: "" for key, _ in PARTS}, "evidence_sentences": [], "new_terms": [],
                       "open_terms": []}
            entries.append(current)
            block = ""
            continue
        if current is None:
            continue
        if line.startswith("#"):
            current = None
            continue
        if line.startswith("- "):
            item = line[2:].strip()
            block = ""
            if item.startswith("Summary"):
                block = "summary"
                continue
            if item.startswith(EVIDENCE_HEADING.split("(")[0]):
                block = "evidence"
                continue
            label, value = _label_value(item)
            field = _LABELS.get(label)
            if field == "source":
                parts = [p.strip() for p in re.split(r"[;;]", value)]
                current["link"] = parts[0] if parts else ""
                current["published"] = parts[1] if len(parts) > 1 else ""
                current["type"] = parts[2] if len(parts) > 2 else ""
                current["arxiv_id"] = normalize_arxiv_id(current["link"]) if current["link"] else None
            elif field in ("new_terms", "open_terms"):
                current[field] = _split_terms(value)
            elif field == "relation":
                current["relation"] = value
        elif line.startswith(("  - ", "\t- ")):
            item = line.strip()[2:].strip()
            if block == "summary":
                label, value = _label_value(item)
                if label in _PART_KEYS:
                    current[_PART_KEYS[label]] = value
            elif block == "evidence" and item:
                current["evidence_sentences"].append(item)
    return entries


# ---- the store ---------------------------------------------------------------------------------------------

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿豈-﫿가-힯]")
_CJK_RUN = re.compile(_CJK.pattern + "+")
_WORD = re.compile(r"[^\W_]+")


def _index_text(text: str) -> str:
    """text as indexed: NFKC, casefolded, every CJK character a token of its own."""
    return _CJK.sub(lambda m: f" {m.group()} ", normalize_text(text).casefold())


# Query words that carry no content: required, they would match nearly every note or none.
_STOPWORD_TEXT = """
    a an the of in on at to for from by with without into onto about as and or nor but not no is are was
    were be been being am do does did has have had it its this that these those there their them they we
    our you your i me my he she his her what which who whom whose how when where why than then so such any
    all some each every can could may might will would shall should must also just only very more most
    other per via vs if else up out over under between among across through during before after again once
    both either neither
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())
# Chinese particles: they split a CJK run into terms (each still a phrase of characters) and are dropped. A
# particle inside a word splits it too ("现在" leaves "现"), which only widens that term's matches.
CJK_PARTICLES = "的了和与在是及或把被"
_PARTICLE = re.compile(f"[{CJK_PARTICLES}]+")


def _query_terms(query: str) -> list[str]:
    """The query's content terms in order, deduplicated: Latin words outside STOPWORDS, and the pieces of CJK
    runs between CJK_PARTICLES (each a phrase of characters)."""
    text = normalize_text(query).casefold()
    terms: list[str] = []
    for piece in re.split(f"({_CJK_RUN.pattern})", text):
        if _CJK_RUN.fullmatch(piece):
            found = [run for run in _PARTICLE.split(piece) if run]
        else:
            found = [word for word in _WORD.findall(piece) if word not in STOPWORDS]
        terms.extend(t for t in found if t not in terms)
    return terms


def _fts_query(terms: list[str]) -> str:
    """An FTS5 MATCH expression requiring every term: quoted terms joined by AND (a CJK term as a phrase of
    its characters, which the index holds one per token)."""
    return " AND ".join('"' + " ".join(t) + '"' if _CJK_RUN.fullmatch(t) else f'"{t}"' for t in terms)


def fts5_available() -> bool:
    """Whether this sqlite3 build can create an FTS5 table (probed in a throwaway in-memory database)."""
    try:
        with closing(sqlite3.connect(":memory:")) as db:
            db.execute("CREATE VIRTUAL TABLE probe USING fts5(body)")
        return True
    except sqlite3.OperationalError:
        return False


_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    seq INTEGER PRIMARY KEY, note_id TEXT UNIQUE NOT NULL, arxiv_id TEXT NOT NULL, title TEXT NOT NULL,
    path TEXT NOT NULL, created TEXT NOT NULL, tags TEXT NOT NULL, body TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS note_tags (note_id TEXT NOT NULL, tag_key TEXT NOT NULL,
    PRIMARY KEY (note_id, tag_key));
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(title, body, tokenize = 'unicode61');
"""


def searchable_body(entry: dict) -> str:
    """What the FTS index holds besides the title: an entry's content (relation, the four parts, the evidence
    sentences and both term lists, one per line) without the template's labels, which every note shares and
    which would otherwise match every note."""
    values = [entry["relation"], *(entry[key] for key, _ in PARTS), *entry["evidence_sentences"],
              ", ".join(entry["new_terms"]), ", ".join(entry["open_terms"])]
    return "\n".join(value for value in values if value)


def _tag_key(tag: str) -> str:
    return normalize_text(tag).casefold()


def _checked_id(arxiv_id: object) -> str:
    """arxiv_id when it is a canonical new-style id (it names files), else a ValueError."""
    if not (isinstance(arxiv_id, str) and normalize_arxiv_id(arxiv_id) == arxiv_id):
        raise ValueError(f"arxiv_id {arxiv_id!r:.80} is not a canonical new-style arXiv id (YYMM.NNNNN); "
                         "it would name the note's file")
    return arxiv_id


class NoteStore:
    """Markdown notes under root plus root/notes.sqlite (see the module docstring)."""

    def __init__(self, root: Path | str) -> None:
        if not fts5_available():
            raise RuntimeError("this Python's sqlite3 module was built without FTS5, which NoteStore needs "
                               "for BM25 search; use a Python whose SQLite has the FTS5 extension")
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / DATABASE
        with closing(self._connect()) as db, db:
            db.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database)

    def write(self, paper: dict, args: dict, evidence: list[str], today: str,
              tags: list[str] | None = None) -> dict:
        """Render the note, index it and write it to root/<arxiv_id>-<n>.md; {"note_id", "arxiv_id",
        "path"} with path relative to root in forward slashes. A paper whose arxiv_id is not a canonical
        new-style id (normalize_arxiv_id's fixed point) is a ValueError before anything is written."""
        arxiv_id = _checked_id(paper.get("arxiv_id"))
        return self._add(arxiv_id, paper.get("title", ""), render_note(paper, args, evidence, today), today,
                         tags)

    def load_entries(self, entries: list[dict], tags: list[str] | None = None) -> list[dict]:
        """Seed the store with import_entries' entries (the owner's notes file): each entry with an arXiv id
        becomes a note titled and dated (created) as the entry, its file the entry rendered by render_entry,
        its tags the entry's "tags" list (when it has one) plus tags. Entries without an arXiv id (a report,
        a blog post) are skipped. The records of the notes written, in entry order; a non-canonical
        arxiv_id is a ValueError raised before any entry is written."""
        chosen = [entry for entry in entries if entry.get("arxiv_id")]
        for entry in chosen:
            _checked_id(entry["arxiv_id"])
        return [self._add(entry["arxiv_id"], entry.get("title", ""), render_entry(entry),
                          _one_line(entry.get("today")), [*(entry.get("tags") or []), *(tags or [])])
                for entry in chosen]

    def _add(self, arxiv_id: str, title: str, markdown: str, created: str, tags: list[str] | None) -> dict:
        """Index the rendered entry markdown and write it out. The rows go in first, in a transaction; the
        file is written (to a temporary name, then renamed) only once the transaction has committed, so a
        failing insert leaves no file behind (and no orphan file shifts a later note's number)."""
        title = _one_line(title)
        tags = [t for t in dict.fromkeys(_one_line(t) for t in (tags or [])) if t]
        body = searchable_body(import_entries(markdown)[0])
        with closing(self._connect()) as db, db:  # the transaction commits before the connection closes
            (count,) = db.execute("SELECT COUNT(*) FROM notes WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
            n = count + 1
            while (db.execute("SELECT 1 FROM notes WHERE note_id = ?", (f"{arxiv_id}-{n}",)).fetchone()
                   or (self.root / f"{arxiv_id}-{n}.md").exists()):
                n += 1
            note_id = f"{arxiv_id}-{n}"
            path = f"{note_id}.md"
            cursor = db.execute(
                "INSERT INTO notes (note_id, arxiv_id, title, path, created, tags, body) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (note_id, arxiv_id, title, path, created, canonical_json(tags), body))
            db.execute("INSERT INTO notes_fts (rowid, title, body) VALUES (?, ?, ?)",
                       (cursor.lastrowid, _index_text(title), _index_text(body)))
            db.executemany("INSERT OR IGNORE INTO note_tags (note_id, tag_key) VALUES (?, ?)",
                           [(note_id, _tag_key(t)) for t in tags])
        # committed: the row exists, so the file may appear (whole, through a temporary name)
        staged = self.root / f"{path}.tmp"
        staged.write_text(markdown, encoding="utf-8", newline="\n")
        os.replace(staged, self.root / path)
        return {"note_id": note_id, "arxiv_id": arxiv_id, "path": path}

    def search(self, query: str, tags: list[str] | None = None, top_k: int = 3) -> list[dict]:
        """BM25 hits for query among the notes carrying every tag in tags (case-insensitive), best first,
        ties by note_id; at most top_k. Every content term of the query is required (an AND; STOPWORDS and
        CJK_PARTICLES are dropped). score is the negated FTS5 bm25 rank, unrounded: higher is better (FTS5
        gives a term found in most notes a tiny idf, so close scores are real and must not be flattened).
        An empty query, a query without content terms or no match gives []."""
        terms = _query_terms(query or "")
        if not terms or top_k <= 0:
            return []
        sql = ("SELECT n.note_id, n.arxiv_id, n.title, n.body, bm25(notes_fts, ?, ?) AS rank "
               "FROM notes_fts JOIN notes n ON n.seq = notes_fts.rowid WHERE notes_fts MATCH ?")
        params: list = [TITLE_WEIGHT, BODY_WEIGHT, _fts_query(terms)]
        keys = sorted({_tag_key(t) for t in tags or [] if _tag_key(t)})
        if keys:
            sql += (f" AND n.note_id IN (SELECT note_id FROM note_tags WHERE tag_key IN "
                    f"({', '.join('?' * len(keys))}) GROUP BY note_id HAVING COUNT(*) = ?)")
            params += [*keys, len(keys)]
        sql += " ORDER BY rank, n.note_id LIMIT ?"
        params.append(top_k)
        with closing(self._connect()) as db:
            rows = db.execute(sql, params).fetchall()
        return [{"note_id": note_id, "arxiv_id": arxiv_id, "title": title,
                 "snippet": _snippet(body or title, terms), "score": -rank}
                for note_id, arxiv_id, title, body, rank in rows]

    def list(self, tag: str | None = None, since: str | None = None, limit: int = 10) -> list[dict]:
        """Notes newest first (created, then the order of writing), optionally only those carrying tag and
        created on or after since (YYYY-MM-DD); at most limit."""
        sql, params = "SELECT note_id, arxiv_id, title, created, tags FROM notes WHERE 1 = 1", []
        if tag and _tag_key(tag):
            sql += " AND note_id IN (SELECT note_id FROM note_tags WHERE tag_key = ?)"
            params.append(_tag_key(tag))
        if since:
            sql += " AND created >= ?"
            params.append(since)
        sql += " ORDER BY created DESC, seq DESC LIMIT ?"
        params.append(max(0, limit))
        with closing(self._connect()) as db:
            rows = db.execute(sql, params).fetchall()
        return [{"note_id": note_id, "arxiv_id": arxiv_id, "title": title, "created": created,
                 "tags": json.loads(tags)} for note_id, arxiv_id, title, created, tags in rows]


SNIPPET_LEAD = 40  # at most this many characters of lead-in before the first match


def _snippet(body: str, terms: list[str]) -> str:
    """At most MAX_SNIPPET_CHARS of the body, whitespace collapsed, starting at a word boundary at most
    SNIPPET_LEAD characters before the first occurrence of any query term (at the match itself when no word
    starts in that lead-in; the start of the body when no term occurs literally)."""
    text = " ".join(body.split())
    folded = text.casefold()  # same length as text for the scripts notes use
    positions = [p for p in (folded.find(t) for t in terms) if p >= 0]
    start = 0
    if positions and len(folded) == len(text):
        match = min(positions)
        start = max(0, match - SNIPPET_LEAD)
        while start < match and _inside_word(text, start):
            start += 1
        start += len(text[start:match]) - len(text[start:match].lstrip())
    return text[start:start + schemas.MAX_SNIPPET_CHARS]


def _inside_word(text: str, i: int) -> bool:
    """Whether position i falls between two characters of one Latin word (CJK characters are words of their
    own, so a cut between two of them is a boundary)."""
    return (i > 0 and text[i - 1].isalnum() and text[i].isalnum()
            and not _CJK.match(text[i - 1]) and not _CJK.match(text[i]))


# ---- the tools ---------------------------------------------------------------------------------------------


def note_tools(store: NoteStore, papers: Callable[[str], dict | None], today: str) -> dict[str, ToolFn]:
    """search_notes, list_notes and write_note over store. papers(arxiv_id) is the metadata the executor has
    seen (title, published, abstract; categories, when present, become the note's tags), None for an unknown
    paper. Output shapes as in data/synth.default_note_tools; see the module docstring."""
    properties = _write_note_properties()

    def write_note(args: dict) -> ToolOutcome:
        args = {**schemas.DEFAULTS["write_note"], **args}
        arxiv_id, mode = args.get("arxiv_id"), args.get("mode")
        if not (isinstance(arxiv_id, str) and arxiv_id and isinstance(mode, str)
                and mode in properties["mode"]["enum"]):
            return ToolOutcome({}, schemas.INVALID_ARGUMENTS)
        paper = papers(arxiv_id)
        if paper is None:
            return ToolOutcome({}, schemas.NOT_FOUND)
        paper = {**paper, "arxiv_id": paper.get("arxiv_id") or arxiv_id}
        keywords = [t for key in ("new_terms", "open_terms") if isinstance(args.get(key), list)
                    for t in args[key] if isinstance(t, str)]
        evidence = pick_evidence(paper.get("abstract", ""), EVIDENCE_SENTENCES, keywords)
        if EXCERPT_FIELD in properties and isinstance(args.get(EXCERPT_FIELD), str):
            extra = pick_evidence(args[EXCERPT_FIELD], EVIDENCE_SENTENCES, keywords)
            evidence += [s for s in extra if s not in evidence][:max(0, EVIDENCE_SENTENCES - len(evidence))]
        tags = [t for t in paper.get("categories", []) if isinstance(t, str)]
        return ToolOutcome(store.write(paper, args, evidence, today, tags))

    def search_notes(args: dict) -> ToolOutcome:
        args = {**schemas.DEFAULTS["search_notes"], **args}
        query, tags = args.get("query", ""), args.get("tags") or []
        tags_ok = isinstance(tags, list) and all(isinstance(t, str) for t in tags)
        if not isinstance(query, str) or not tags_ok:
            return ToolOutcome([], schemas.INVALID_ARGUMENTS)
        return ToolOutcome(store.search(query, tags, capped(args, "top_k", schemas.MAX_NOTE_HITS)))

    def list_notes(args: dict) -> ToolOutcome:
        args = {**schemas.DEFAULTS["list_notes"], **args}
        tag, since = args.get("tag"), args.get("since")
        if not all(value is None or isinstance(value, str) for value in (tag, since)):
            return ToolOutcome([], schemas.INVALID_ARGUMENTS)
        return ToolOutcome(store.list(tag, since, capped(args, "limit", LIST_LIMIT_MAX)))

    return {"search_notes": search_notes, "list_notes": list_notes, "write_note": write_note}

"""The evidence repair of the synthetic training records (M3; the owner's decision of 2026-10-15).

Scaffold notes are meant to carry evidence quoted verbatim from the paper (docs/plan.md; eval.e2e checks
it). About a third of the teacher's scaffold write_note calls paraphrase, translate or join sentences
instead, and a student trained on them would learn to do the same. repair_evidence rewrites, in one
trajectory's message history, the evidence argument of every scaffold write_note call that fails the e2e
evidence check (eval.e2e.unsupported_fragments against the strings of every earlier tool output about the
note's paper) with sentences quoted verbatim from the text the model could see when it made the call.

What the model could see. agent.context.build_context keeps only the latest step whole: the tool outputs of
the assistant message with tool calls just before the call; older tool outputs are reduced to ids and
titles. The sentences are therefore taken only from the prose fields (TEXT_FIELDS: a fetch_fulltext
section's "text", a get_paper "abstract", a search hit's "snippet") of that previous step's outputs about the
note's paper. A call whose previous step holds no such text, or no sentence that passes the guards, is
unrepairable: the record is to be dropped (the caller decides), since quoting text the model cannot see
would teach it to recite what it never read.

Which sentences. notes.split_sentences cuts each text; a sentence qualifies when it ends with a sentence
mark (so the tail of a text cut at its character cap never does), holds no quotation mark (QUOTE_MARKS, which
would break e2e's quotation spans), no "-" before a line break (e2e joins such words in the source) and no
special-token text (SPECIAL), and has at least e2e.MIN_QUOTE_CHARS token characters. The candidates, without
repeats, are ranked by how many distinct tokens they share with the teacher's original argument (the sentences
it was paraphrasing come first), then by how many of the call's new_terms and open_terms they contain, then by
their order in the step's outputs; the first K are written in that output order as
"S1" / "S2" / ... (SEPARATOR between double-quoted sentences: the teacher's most frequent form). Only the
"evidence" value changes: the arguments are parsed with json.loads, which keeps the keys in order, and
written back with json.dumps(ensure_ascii=False). Each repaired argument is checked against the same sources
as e2e's check before it is kept (RepairError otherwise: that would be a bug, not data).

The final answer. Once the evidence is repaired, a final answer may restate details that only the old
argument held: the earlier tool outputs are compressed by then, so the argument was the model's only full
copy. A detail of a text is one of its number tokens (e2e.evidence_tokens; a token holding a digit) or one of
its runs of NGRAM tokens. When the trajectory's last message is an answer without calls and it shares a detail
with an original argument of a repaired call that is a detail of no string of the repaired context of that
answer (build_context of the history before it: message texts, call arguments, tool outputs as they are
left), the answer is removed from the history: the trainer cuts one sample per assistant message, so the
record keeps every earlier step and that one answer leaves the loss. Numbers count alone because a restated
figure is the costliest thing to learn to say without seeing it.

Calls left alone: write_note calls in draft mode, calls whose evidence is not a non-empty string (the
owner's check_trajectory refuses those in accepted trajectories), and calls that already pass. The input is
never mutated; equal input gives equal output.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field

from ..agent.context import build_context
from ..eval.e2e import (
    MIN_QUOTE_CHARS,
    SCAFFOLD,
    WRITE_NOTE,
    _calls,
    _paper_strings,
    evidence_check,
    evidence_tokens,
    unsupported_fragments,
)
from ..normalize import normalize_arxiv_id, normalize_text
from ..notes import split_sentences
from ..tools.schemas import DEFAULTS

K = 4  # sentences per repaired argument, as many as the note tool picks (notes.EVIDENCE_SENTENCES)
SEPARATOR = " / "
TEXT_FIELDS = ("text", "abstract", "snippet")
QUOTE_MARKS = ('"', "“", "”", "「", "」", "『", "』")
SPECIAL = ("<|", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>", "<tools>", "</tools>",
           "<think>", "</think>")
NGRAM = 6  # tokens in a run that counts as a restated detail (a number counts alone)
_SENTENCE_END = re.compile(r"[.!?。！？][)\]]*$")


class RepairError(ValueError):
    """A repaired argument failed the check it was built to pass (a bug, not data)."""


@dataclass
class Repair:
    """The repaired history and what was done: one entry per repaired call ({"call_id", "step", "sentences",
    "sources": {field: count}}), the ids of the calls that could not be repaired, whether the final answer
    was removed, and how many scaffold write_note calls there were."""

    messages: list[dict]
    calls: list[dict] = field(default_factory=list)
    unrepairable: list[str] = field(default_factory=list)
    final_answer_dropped: bool = False
    scaffold_calls: int = 0  # scaffold write_note calls looked at, repaired or not

    @property
    def changed(self) -> bool:
        return bool(self.calls) or self.final_answer_dropped


@dataclass(frozen=True)
class _Candidate:
    sentence: str
    field: str
    order: int  # position among the previous step's sentences


def _parse(text: object) -> object:
    if not isinstance(text, str):
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _strings(node: object) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def _paper_nodes(node: object, arxiv_id: str) -> Iterator[dict]:
    """Every dict at any depth whose arxiv_id is arxiv_id (version and prefix aside)."""
    if isinstance(node, dict):
        own = node.get("arxiv_id")
        if isinstance(own, str) and normalize_arxiv_id(own) == arxiv_id:
            yield node
        else:
            for value in node.values():
                yield from _paper_nodes(value, arxiv_id)
    elif isinstance(node, list):
        for value in node:
            yield from _paper_nodes(value, arxiv_id)


def _previous_step_texts(messages: list[dict], index: int, arxiv_id: str) -> list[tuple[str, str]]:
    """(field, text) of the prose fields about the paper in the previous step's tool outputs, in order."""
    start = next((i for i in range(index - 1, -1, -1)
                  if messages[i].get("role") == "assistant" and messages[i].get("tool_calls")), -1)
    if start < 0:
        return []
    texts = []
    for message in messages[start + 1:index]:
        if message.get("role") != "tool":
            continue
        for node in _paper_nodes(_parse(message.get("content")), arxiv_id):
            texts += [(name, node[name]) for name in TEXT_FIELDS if isinstance(node.get(name), str)]
    return texts


def _qualifies(raw: str) -> bool:
    if "-\n" in raw or any(mark in raw for mark in QUOTE_MARKS) or any(tag in raw for tag in SPECIAL):
        return False
    if not _SENTENCE_END.search(raw):
        return False
    return sum(map(len, evidence_tokens(raw))) >= MIN_QUOTE_CHARS


def _candidates(texts: list[tuple[str, str]]) -> list[_Candidate]:
    out, seen = [], set()
    for name, text in texts:
        for begin, end in split_sentences(text):
            raw = text[begin:end]
            if not _qualifies(raw):
                continue
            sentence = " ".join(raw.split())
            key = tuple(evidence_tokens(sentence))
            if key in seen:
                continue
            seen.add(key)
            out.append(_Candidate(sentence, name, len(out)))
    return out


def _keywords(args: dict) -> list[str]:
    words = [t for key in ("new_terms", "open_terms") if isinstance(args.get(key), list)
             for t in args[key] if isinstance(t, str)]
    return [w for w in (normalize_text(w).casefold() for w in words) if w]


def _choose(candidates: list[_Candidate], original: str, keywords: list[str]) -> list[_Candidate]:
    wanted = set(evidence_tokens(original))

    def rank(c: _Candidate) -> tuple[int, int, int]:
        shared = len(set(evidence_tokens(c.sentence)) & wanted)
        hits = sum(1 for w in keywords if w in normalize_text(c.sentence).casefold())
        return -shared, -hits, c.order

    return sorted(sorted(candidates, key=rank)[:K], key=lambda c: c.order)


def _details(text: str) -> set[tuple[str, ...]]:
    """The number tokens of text (as one-tuples) and its runs of NGRAM tokens."""
    tokens = evidence_tokens(text)
    numbers = {(t,) for t in tokens if any(ch.isdigit() for ch in t)}
    return numbers | {tuple(tokens[i:i + NGRAM]) for i in range(len(tokens) - NGRAM + 1)}


def _context_details(messages: list[dict]) -> set[tuple[str, ...]]:
    grams: set[tuple[str, ...]] = set()
    for message in build_context(messages):
        texts = list(_strings(_parse(message.get("content")) if message.get("role") == "tool"
                              else message.get("content") or ""))
        for c in message.get("tool_calls") or []:
            texts += list(_strings(_parse((c.get("function") or {}).get("arguments"))))
        for text in texts:
            grams |= _details(text)
    return grams


def repair_evidence(messages: list[dict]) -> Repair:
    """The repaired copy of a trajectory's messages (the module docstring has the rules)."""
    out = copy.deepcopy(messages)
    repair = Repair(out)
    originals: list[str] = []
    # e2e's own reading of the history (bodies by tool_call_id, step numbers), aligned with the raw calls and
    # the index of the message that made each; tool messages never change here, so it stays valid
    raw = [(i, c) for i, m in enumerate(out) if m.get("role") == "assistant"
           for c in m.get("tool_calls") or []]
    calls = _calls(out)
    left_alone: set[str] = set()  # scaffold calls e2e checks but the repair does not touch
    for call, (index, raw_call) in zip(calls, raw, strict=True):
        if call.name != WRITE_NOTE or call.args is None:
            continue
        if call.args.get("mode", DEFAULTS[WRITE_NOTE]["mode"]) != SCAFFOLD:
            continue
        repair.scaffold_calls += 1
        evidence, raw_id = call.args.get("evidence"), call.args.get("arxiv_id")
        arxiv_id = normalize_arxiv_id(raw_id) if isinstance(raw_id, str) else None
        if not (isinstance(evidence, str) and evidence.strip()) or arxiv_id is None:
            left_alone.add(call.id)
            continue
        sources = [s for earlier in calls if earlier.step < call.step
                   for s in _paper_strings(earlier.body, arxiv_id)]
        if not unsupported_fragments(evidence, sources):
            continue
        args = json.loads(raw_call["function"]["arguments"])  # _calls parsed the same text: it is a dict
        visible = _candidates(_previous_step_texts(out, index, arxiv_id))
        chosen = _choose(visible, evidence, _keywords(args))
        if not chosen:
            repair.unrepairable.append(call.id)
            continue
        new = SEPARATOR.join(f'"{c.sentence}"' for c in chosen)
        left = unsupported_fragments(new, sources)
        if left:
            raise RepairError(f"call {call.id}: the repaired evidence still has unsupported {left[:2]}")
        args["evidence"] = new
        raw_call["function"]["arguments"] = json.dumps(args, ensure_ascii=False)
        originals.append(evidence)
        repair.calls.append({"call_id": call.id, "step": call.step, "sentences": len(chosen),
                             "sources": dict(Counter(c.field for c in chosen))})
    failing = {entry.split(": ", 1)[0] for entry in evidence_check(out)["not_retrieved"]} - left_alone
    if failing != set(repair.unrepairable):
        raise RepairError(f"after the repair e2e still fails {sorted(failing)}, unrepairable "
                          f"{sorted(repair.unrepairable)}")
    last = out[-1] if out else {}
    if originals and last.get("role") == "assistant" and not last.get("tool_calls"):
        old = set().union(*(_details(text) for text in originals))
        if (_details(last.get("content") or "") & old) - _context_details(out[:-1]):
            out.pop()
            repair.final_answer_dropped = True
    return repair

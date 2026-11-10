"""The external zh-en glossary and the term note (docs/m7-本地提升计划.md, WP4).

The glossary maps a Chinese term to one English term, or to None: a blocked term, which is matched like any
term but never noted (Matching). data/glossary/zh-en.json holds it as one canonical JSON object, a blocked
term with the value null, built by scripts/build_glossary.py from the sources its provenance.json records
(where the terms come from, the rule that picks one English term per Chinese term, the filters that drop or
block a term and their measurement on the training data). load_glossary reads and validates it; the blocked
terms travel in the same file, so the file's SHA-256 (what a training run records and an inference
configuration names) covers them, and training and inference read the same spans.

The note. note_for(messages, glossary) is the line the model is shown for a history: the glossary's noted
terms in the latest user message (the last message with role "user"; no other message is read), matched
leftmost-longest without overlap (Matching), each term once, at most MAX_TERMS of them in order of appearance,
as one line

    术语对照:{zh} = {en};{zh} = {en}

or None when no noted term occurs. apply_note(system_text, note) appends the note to a system message's text
on a line of its own; noted_messages does both for a history and returns a copy with the note on its first
system message. The training renderer (research_agent.train.samples, the glossary-note option) and the
inference wrapper call these same functions, so a training sample's system message is the one inference
sends.

Matching. The message and the terms are compared in match form: term_form (NFKC, whitespace runs collapsed
to one space and trimmed, whitespace next to a Han character removed) folded with casefold(). So the
message "深度 Q 学习" matches the term "深度Q学习", "ＫＶ缓存" (full-width letters) and "kv 缓存" match
"KV缓存", and the note shows the glossary's own term ("KV缓存 = KV Cache"). At every position the longest
term that starts there is taken, noted or blocked, and the scan goes on after it; where no term starts, the
scan moves one character on. A blocked term's span is taken like any other, so no shorter term inside it
matches there (with 预训练 blocked, 训练数据 is not noted in "预训练数据"), and it takes no place among the
MAX_TERMS. A term is not taken where it would start or end inside a Latin word: when its first character is
an ASCII letter or digit and so is the character before it, or its last character is one and so is the
character after it (E步 is not matched in "TYPE步骤"); a shorter term at the same position may still be. A
content that is not a string has no term.

Validation (Glossary, load_glossary): a JSON object whose values are strings or null; every Chinese term,
noted or blocked, holds a Han character, is its own term form and holds no separator of the note ("=", ";")
and no line break; every English term is non-empty, one line of single-spaced text
(" ".join(text.split()) == text), without a separator or a Han character; no two Chinese terms share a match
form (the note could show either, or a span would be both noted and blocked). A failure is a GlossaryError
naming the term.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterator, Mapping
from pathlib import Path

MAX_TERMS = 5  # terms per note, the plan's cap
NOTE_PREFIX = "术语对照:"
PAIR_JOIN, PAIR_SEPARATOR = " = ", ";"  # "{zh} = {en}", pairs joined by ";"
SEPARATORS = ("=", ";")  # characters a term may not hold: the note could not be read back
_HAN = "㐀-䶿一-鿿豈-﫿\U00020000-\U0003134f"  # CJK unified ideographs and extensions
HAN = re.compile(f"[{_HAN}]")
_SPACE_BY_HAN = re.compile(f"\\s+(?=[{_HAN}])|(?<=[{_HAN}])\\s+")


class GlossaryError(ValueError):
    """A glossary that cannot be used (module docstring: Validation), or a note that is not one line, or a
    note due on a history without a system message; the message names the place."""


def term_form(text: str) -> str:
    """NFKC, whitespace runs collapsed to one space and trimmed, then whitespace next to a Han character
    removed: "深度 Q 学习" -> "深度Q学习", "偏差 - 方差分解" -> "偏差-方差分解". The glossary's Chinese
    terms are stored in this form."""
    collapsed = " ".join(unicodedata.normalize("NFKC", text).split())
    return _SPACE_BY_HAN.sub("", collapsed)


def match_form(text: str) -> str:
    """The form a message and a term are compared in: term_form folded with casefold()."""
    return term_form(text).casefold()


def has_han(text: str) -> bool:
    return HAN.search(text) is not None


def is_latin(character: str) -> bool:
    """An ASCII letter or digit: the characters a Latin word is made of (module docstring: Matching)."""
    return character.isascii() and character.isalnum()


def within_latin_word(text: str, start: int, end: int) -> bool:
    """True when the span text[start:end] starts or ends inside a Latin word: its first character and the one
    before it are both ASCII letters or digits, or its last character and the one after it are."""
    return (start > 0 and is_latin(text[start]) and is_latin(text[start - 1])) or (
        end < len(text) and is_latin(text[end - 1]) and is_latin(text[end]))


class Glossary(Mapping[str, str]):
    """A validated zh -> en glossary (module docstring: Validation) with its match index. The mapping holds
    the noted terms; blocked holds the blocked ones and entries() both, as the glossary file does."""

    def __init__(self, terms: object, *, where: str = "the glossary") -> None:
        if not isinstance(terms, Mapping):
            raise GlossaryError(f"{where} is not a JSON object of Chinese terms and English terms")
        index: dict[str, str] = {}
        for zh, en in terms.items():
            _check_term(zh, en, where)
            folded = match_form(zh)
            if folded in index:
                raise GlossaryError(f"{where}: {index[folded]!r} and {zh!r} have the same match form "
                                    f"{folded!r}; a note could show either")
            index[folded] = zh
        self._entries: dict[str, str | None] = dict(terms)
        self._terms = {zh: en for zh, en in self._entries.items() if en is not None}
        self._blocked = tuple(sorted(zh for zh, en in self._entries.items() if en is None))
        self._index = index
        self._sizes = sorted({len(folded) for folded in index}, reverse=True)

    def __getitem__(self, zh: str) -> str:
        return self._terms[zh]

    def __iter__(self) -> Iterator[str]:
        return iter(self._terms)

    def __len__(self) -> int:
        return len(self._terms)

    def __eq__(self, other: object) -> bool:
        """Two glossaries are equal when they note and block the same terms; a plain mapping compares with the
        noted terms."""
        if isinstance(other, Glossary):
            return self._entries == other._entries
        return super().__eq__(other)

    @property
    def blocked(self) -> tuple[str, ...]:
        """The blocked terms, sorted (module docstring: Matching)."""
        return self._blocked

    def entries(self) -> dict[str, str | None]:
        """Every term with its English term, None for a blocked term: what the glossary file holds."""
        return dict(self._entries)

    def terms_in(self, text: str) -> list[str]:
        """The glossary's noted terms in text, matched leftmost-longest without overlap in match form, a
        blocked term's span taken without a note and no term taken inside a Latin word, each once, at most
        MAX_TERMS, in order of appearance (module docstring: Matching)."""
        folded = match_form(text)
        found: list[str] = []
        position = 0
        while position < len(folded) and len(found) < MAX_TERMS:
            for size in self._sizes:
                end = position + size
                if end > len(folded):
                    continue
                term = self._index.get(folded[position:end])
                if term is None or within_latin_word(folded, position, end):
                    continue
                if term in self._terms and term not in found:
                    found.append(term)
                position = end
                break
            else:
                position += 1
        return found


def _check_term(zh: object, en: object, where: str) -> None:
    if not isinstance(zh, str) or not (en is None or isinstance(en, str)):
        raise GlossaryError(f"{where}: {zh!r} -> {en!r}: a term is not a string (an English term is a "
                            "string, or null for a blocked term)")
    if not has_han(zh):
        raise GlossaryError(f"{where}: {zh!r} holds no Chinese character")
    if term_form(zh) != zh:
        raise GlossaryError(f"{where}: {zh!r} is not in term form ({term_form(zh)!r})")
    if en is None:  # a blocked term: the Chinese term only
        if any(separator in zh for separator in SEPARATORS):
            raise GlossaryError(f"{where}: {zh!r} holds a separator of the note ({' '.join(SEPARATORS)})")
        return
    if not en:
        raise GlossaryError(f"{where}: the English term of {zh!r} is empty")
    if " ".join(en.split()) != en:
        raise GlossaryError(f"{where}: the English term {en!r} of {zh!r} is not one line of single-spaced "
                            "text")
    for text in (zh, en):
        if any(separator in text for separator in SEPARATORS):
            raise GlossaryError(f"{where}: {text!r} holds a separator of the note ({' '.join(SEPARATORS)})")
    if has_han(en):
        raise GlossaryError(f"{where}: the English term {en!r} of {zh!r} holds a Chinese character")


def load_glossary(path: str | Path) -> Glossary:
    """The glossary in a JSON file (a zh -> en object, null for a blocked term), validated; GlossaryError
    naming the file."""
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GlossaryError(f"{path}: not JSON: {exc}") from exc
    return Glossary(value, where=str(path))


def latest_user_text(messages: list[dict]) -> str:
    """The content of the last message with role "user"; "" when there is none or it is not a string."""
    for message in reversed(messages):
        if message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def _as_glossary(glossary: Mapping[str, str | None]) -> Glossary:
    return glossary if isinstance(glossary, Glossary) else Glossary(glossary)


def note_terms(messages: list[dict], glossary: Mapping[str, str | None]) -> list[tuple[str, str]]:
    """The (zh, en) pairs the note of this history shows (module docstring: The note); [] when none."""
    glossary = _as_glossary(glossary)
    return [(zh, glossary[zh]) for zh in glossary.terms_in(latest_user_text(messages))]


def format_note(pairs: list[tuple[str, str]]) -> str:
    return NOTE_PREFIX + PAIR_SEPARATOR.join(f"{zh}{PAIR_JOIN}{en}" for zh, en in pairs)


def note_for(messages: list[dict], glossary: Mapping[str, str | None]) -> str | None:
    """The note line of this history, or None when its latest user message holds no noted glossary term."""
    pairs = note_terms(messages, glossary)
    return format_note(pairs) if pairs else None


def apply_note(system_text: str, note: str | None) -> str:
    """system_text with the note appended on a line of its own; system_text itself when note is None."""
    if note is None:
        return system_text
    if "\n" in note or "\r" in note:
        raise GlossaryError(f"a note is one line: {note!r}")
    return f"{system_text}\n{note}"


def noted_messages(messages: list[dict],
                   glossary: Mapping[str, str | None]) -> tuple[list[dict], str | None]:
    """(a copy of messages with the note applied to its first system message, the note); the messages
    themselves and None when there is no note. The input is not changed: the system message is replaced by a
    copy, the other messages are the same objects. GlossaryError when a note is due and no message has the
    role "system"."""
    note = note_for(messages, glossary)
    if note is None:
        return messages, None
    for position, message in enumerate(messages):
        if message.get("role") == "system":
            content = message.get("content")
            if not isinstance(content, str):
                raise GlossaryError(f"the system message's content is not text: {content!r}")
            noted = {**message, "content": apply_note(content, note)}
            return [*messages[:position], noted, *messages[position + 1:]], note
    raise GlossaryError(f"the history has no system message to carry the note {note!r}")

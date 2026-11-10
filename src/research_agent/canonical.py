"""The one canonical JSON form used for cache keys, call keys and hashes."""

from __future__ import annotations

import json


def canonical_json(value: object) -> str:
    """Sorted keys, no whitespace, UTF-8 without escapes: equal data always gives the same string."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def split_lines(text: str) -> list[str]:
    """The lines of a JSON Lines (or any line-oriented) text: split on "\\n" only.

    str.splitlines() is wrong here: it also breaks on U+2028, U+2029, U+0085, \\x1c-\\x1e, \\x0b and \\x0c.
    canonical_json writes with ensure_ascii=False, so such a character inside a JSON string is stored raw;
    the row is one physical line on disk, but splitlines() would cut it in two. One trailing "\\r" per line
    is stripped (CRLF files), and the empty element after a final newline is dropped; blank lines in the
    middle are kept, so callers count line numbers the way an editor does.
    """
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return [line.removesuffix("\r") for line in lines]

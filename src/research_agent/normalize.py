"""Text normalisation shared by tool-argument checks and evaluation metrics.

Module 01. Each function returns the canonical form of its input, or None when the input is not valid, and
is idempotent: f(f(x)) == f(x). tests/test_normalize.py is the specification; the lecture is
docs/modules/01-文本归一.md.
"""

from __future__ import annotations

import unicodedata


def normalize_text(s: str) -> str:
    """NFKC-normalise s, then collapse every run of whitespace into one space and trim both ends.

    "  Ｓｐｅｃｕｌａｔｉｖｅ　decoding\n" -> "Speculative decoding"
    """
    return " ".join(unicodedata.normalize("NFKC", s).split())


# The hyphens fold_hyphens reads as spaces: hyphen-minus, hyphen, non-breaking hyphen, figure dash, en dash.
HYPHENS = "\N{HYPHEN-MINUS}\N{HYPHEN}\N{NON-BREAKING HYPHEN}\N{FIGURE DASH}\N{EN DASH}"
_HYPHENS_AS_SPACES = str.maketrans(dict.fromkeys(HYPHENS, " "))


def fold_hyphens(s: str) -> str:
    """normalize_text with every hyphen read as a space: after NFKC, each character of HYPHENS (U+002D,
    U+2010, U+2011, U+2012, U+2013) becomes a space, then the whitespace collapses. NFKC first, so a
    full-width or small hyphen-minus folds too; the em dash and the minus sign are no hyphens here.

    The normalized scorer's S2 (eval.metrics): "mixture-of-experts" -> "mixture of experts".
    """
    return " ".join(unicodedata.normalize("NFKC", s).translate(_HYPHENS_AS_SPACES).split())


def normalize_arxiv_id(s: str) -> str | None:
    """Canonical new-style arXiv identifier "YYMM.NNNNN" ("YYMM.NNNN" before 2015), or None.

    Accepts surrounding whitespace, full-width characters, an "arXiv:" prefix in any letter case, arxiv.org
    abs/pdf/html links (with or without "https://", "http://", "www." and a trailing ".pdf") and a version
    suffix such as "v2", which is dropped. The month must be 01-12, the identifier must not predate this
    scheme (April 2007, "0704"), and the sequence has 4 digits before "1501" and 5 digits from "1501" on.
    Old-style identifiers such as "cs/0112017" are not accepted.
    """
    t = normalize_text(s).lower()
    for prefix in ("https://", "http://"):
        t = t.removeprefix(prefix)
    t = t.removeprefix("www.")
    for prefix in ("arxiv.org/abs/", "arxiv.org/pdf/", "arxiv.org/html/", "arxiv:"):
        if t.startswith(prefix):
            t = t.removeprefix(prefix)
            break
    t = t.removesuffix(".pdf")
    head, sep, version = t.rpartition("v")
    if sep:
        if not (version.isascii() and version.isdigit()):
            return None
        t = head
    yymm, dot, seq = t.partition(".")
    if dot != "." or len(yymm) != 4:
        return None
    if not (yymm.isascii() and yymm.isdigit() and seq.isascii() and seq.isdigit()):
        return None
    if not 1 <= int(yymm[2:]) <= 12 or yymm < "0704":
        return None
    if len(seq) != (4 if yymm < "1501" else 5):
        return None
    return f"{yymm}.{seq}"


def is_leap_year(year: int) -> bool:
    """Gregorian rule: divisible by 4, except century years, except century years divisible by 400."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def days_in_month(year: int, month: int) -> int:
    """Number of days in month 1-12 of the given year."""
    if month == 2:
        return 29 if is_leap_year(year) else 28
    if month in (4, 6, 9, 11):
        return 30
    return 31


def normalize_date(s: str) -> str | None:
    """ISO 8601 calendar date "YYYY-MM-DD", or None when s is not a real date.

    Accepts "2026-09-01", "2026-9-1", "2026/09/01", "2026.9.1", "2026年9月1日", "20260901", their full-width
    forms and surrounding whitespace. Rejects two-digit years, times of day and impossible dates such as
    2026-02-29 or 2026-13-01.
    """
    t = normalize_text(s)
    for sep in ("/", ".", "年", "月"):
        t = t.replace(sep, "-")
    t = t.removesuffix("日")
    if len(t) == 8 and t.isascii() and t.isdigit():
        t = f"{t[:4]}-{t[4:6]}-{t[6:]}"
    parts = t.split("-")
    if len(parts) != 3 or not all(p.isascii() and p.isdigit() for p in parts):
        return None
    ys, ms, ds = parts
    if len(ys) != 4 or not 1 <= len(ms) <= 2 or not 1 <= len(ds) <= 2:
        return None
    y, m, d = int(ys), int(ms), int(ds)
    if not 1 <= m <= 12 or not 1 <= d <= days_in_month(y, m):
        return None
    return f"{y:04d}-{m:02d}-{d:02d}"


def query_contains_all(query: str, required: list[str]) -> bool:
    """True when every required term occurs in the query, both compared after normalize_text and casefold().

    Grades free-text search queries: "Speculative Decoding 2026" contains ["speculative", "decoding"].
    Matching is by substring, so "code" counts as contained in "decoder". An empty list is satisfied.
    """
    q = normalize_text(query).casefold()
    return all(normalize_text(term).casefold() in q for term in required)

"""README result numbers and the report they come from (docs/plan.md, 数字对账).

Every result number in README.md sits in a marker block, <!-- num:KEY -->VALUE<!-- /num -->, and
results/numbers.json, which scripts/build_report.py writes (M3) through write_numbers, holds the values:

    {"numbers": {KEY: VALUE, ...}, "sources": {PATH: SHA256, ...}}

VALUE is the text exactly as README.md shows it: the report does the formatting, so the check compares
strings, never floats rounded twice. sources are the result files the report read, as paths relative to the
repository root, each with its SHA-256 at that time; write_numbers refuses a source outside the repository or
one that git ignores, since a clean checkout would not have it. results/ is byte-exact in .gitattributes, so
the hashes hold on every checkout. check() returns one line per problem:

- a marker whose value is not the report's, or whose key the report does not have;
- a malformed marker block (not closed, or a key that is not KEY-shaped);
- markers in README.md but no report, or a report that is not valid JSON;
- a source file of the report that is missing or no longer has the SHA-256 the report read: the report is
  stale and has to be built again before README.md can quote it;
- with tracked (the command line passes git's index): a report, source or access log that git does not track,
  so that a clean checkout would fail where this working tree passes;
- test_set.opens, which comes from the test set's access log rather than from the report: once the test set
  has been opened, README.md must state the number of entries in data/eval/test-access.jsonl in that marker
  (docs/plan.md: 测试集访问日志与 README 写的次数一致);
- test_set_v2.opens, the same for the second family of test sets (M7, docs/m7-计划.md WP-I 7), counted from
  its single-step set's access log, data/eval/v2/test-v2-access.jsonl (eval.lock.TEST_ACCESS_LOG_V2), as
  test_set.opens is from the first family's. Each count stands for its own family only.

Markers inside code (fenced blocks and code spans) only show the syntax and are not read. Keys of the report
that README.md does not quote are no problem, and whitespace just inside a marker block is ignored.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Iterable
from pathlib import Path

from .lock import TEST_ACCESS_LOG, TEST_ACCESS_LOG_V2, JsonlError, read_jsonl, sha256_of

README = Path("README.md")
NUMBERS = Path("results/numbers.json")
TEST_SET_OPENS = "test_set.opens"  # the keys whose values come from an access log, not from the report:
TEST_SET_V2_OPENS = "test_set_v2.opens"  # the first family's opens and the second's (module docstring)
KEY = r"[A-Za-z0-9][A-Za-z0-9_.\-]*"
# The value may not contain "<!--": an unclosed block must not run on into the next one and swallow it.
_MARKER = re.compile(rf"<!--\s*num:({KEY})\s*-->((?:(?!<!--).)*?)<!--\s*/num\s*-->", re.DOTALL)
_OPENING = re.compile(r"<!--\s*num:")
_FENCED = re.compile(r"^(```|~~~).*?^\1", re.DOTALL | re.MULTILINE)
_CODE_SPAN = re.compile(r"(`+)[^\n]*?\1")

Tracked = Callable[[list[str]], set[str]]  # tracked(paths) -> the paths among them that git tracks


def _without_code(text: str) -> str:
    """text with fenced blocks and code spans blanked out; newlines stay, so line numbers still hold."""

    def blank(match: re.Match) -> str:
        return re.sub(r"[^\n]", " ", match.group(0))

    return _CODE_SPAN.sub(blank, _FENCED.sub(blank, text))


def markers(text: str) -> list[tuple[str, str, int]]:
    """(key, value, line number) of every closed marker block outside code; the value without outer
    whitespace."""
    text = _without_code(text)
    found = _MARKER.finditer(text)
    return [(m.group(1), m.group(2).strip(), text.count("\n", 0, m.start()) + 1) for m in found]


def unclosed(text: str) -> list[int]:
    """Line numbers of marker openings outside code that start no well-formed marker block."""
    text = _without_code(text)
    closed = {m.start() for m in _MARKER.finditer(text)}
    return [text.count("\n", 0, m.start()) + 1 for m in _OPENING.finditer(text) if m.start() not in closed]


def git_ignored(path: str, root: str | Path = ".") -> bool:
    """True when git ignores path; False outside a repository or without git."""
    try:
        done = subprocess.run(["git", "-C", str(root), "check-ignore", "-q", "--", path], capture_output=True,
                              check=False)  # exit 1: not ignored; 128: no repository
    except OSError:
        return False
    return done.returncode == 0


def git_tracked(paths: list[str], root: str | Path = ".") -> set[str]:
    """The paths among paths that git's index holds (staged counts as tracked)."""
    done = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--cached", "--", *paths],
                          capture_output=True, check=True)
    return {path for path in done.stdout.decode("utf-8").split("\0") if path}


def write_numbers(numbers: dict[str, str], sources: Iterable[str | Path], path: str | Path = NUMBERS, *,
                  root: str | Path = ".", ignored: Callable[[str], bool] | None = None) -> dict:
    """Write the report's numbers with the SHA-256 of each source file; returns what was written.

    Values must already be formatted strings (README.md quotes them verbatim), keys must match KEY, and
    sources are paths inside root, stored relative to it with forward slashes. A source outside root, or one
    that ignored(path) says git ignores (default: git_ignored), is refused.
    """
    for key, value in numbers.items():
        if not re.fullmatch(KEY, key):
            raise ValueError(f"key {key!r} does not match {KEY}")
        if not isinstance(value, str):
            raise TypeError(f"{key}: values are formatted strings, not {type(value).__name__} ({value!r})")
    root = Path(root)
    is_ignored = ignored or (lambda source: git_ignored(source, root))
    hashed = {}
    for source in sources:
        full = (root / source).resolve()
        try:
            relative = full.relative_to(root.resolve()).as_posix()
        except ValueError:
            raise ValueError(f"source {source} is outside the repository") from None
        if is_ignored(relative):
            raise ValueError(f"source {relative} is ignored by git: a clean checkout would not have it")
        hashed[relative] = sha256_of(full)
    report = {"numbers": dict(sorted(numbers.items())), "sources": dict(sorted(hashed.items()))}
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return report


def logged_opens(access_log: str | Path) -> int:
    """Entries in the test set's access log; 0 while it does not exist."""
    path = Path(access_log)
    return len(read_jsonl(path)) if path.exists() else 0


def _opens(found: list[tuple[str, str, int]], key: str, access_log: Path, what: str, readme: Path,
           root: Path) -> tuple[list[str], list[str]]:
    """The problems of one family's count (key, from access_log; what names its test set) and, once that set
    was opened, its access log as a path git must track."""
    shown_readme, shown_log = readme.as_posix(), access_log.as_posix()  # the same on every platform
    problems, must_track = [], []
    try:
        opens = logged_opens(root / access_log)
    except (JsonlError, OSError) as exc:
        return [f"{shown_log}: unreadable access log ({exc})"], []
    stated = [(value, line) for name, value, line in found if name == key]
    if opens:
        must_track.append(shown_log)
        if not stated:
            problems.append(f"{shown_readme}: {what} was opened {opens} time(s) (access log {shown_log}); "
                            f"state it as <!-- num:{key} -->{opens}<!-- /num -->")
    for value, line in stated:
        if value != str(opens):
            problems.append(f"{shown_readme}:{line}: {key} is {value!r}, the access log {shown_log} has "
                            f"{opens}")
    return problems, must_track


def check(readme: str | Path = README, numbers: str | Path = NUMBERS,
          access_log: str | Path = TEST_ACCESS_LOG, *, access_log_v2: str | Path = TEST_ACCESS_LOG_V2,
          root: str | Path = ".", tracked: Tracked | None = None) -> list[str]:
    """Problems between README.md, the report and the access logs (access_log the first family's, whose
    count is test_set.opens, access_log_v2 the second's, test_set_v2.opens; see the module docstring); []
    when none."""
    root = Path(root)
    text = (root / readme).read_text(encoding="utf-8")
    readme, numbers = Path(readme), Path(numbers)
    shown = {path: path.as_posix() for path in (readme, numbers)}  # the same on every platform
    found = markers(text)
    problems = [f"{shown[readme]}:{line}: malformed marker block (<!-- num:KEY -->VALUE<!-- /num -->, KEY of "
                f"letters, digits, '.', '_' and '-')"
                for line in unclosed(text)]
    must_track = []

    for key, log, what in ((TEST_SET_OPENS, access_log, "the test set"),
                           (TEST_SET_V2_OPENS, access_log_v2, "the v2 test set")):
        found_problems, logs = _opens(found, key, Path(log), what, readme, root)
        problems += found_problems
        must_track += logs

    quoted = [(key, value, line) for key, value, line in found
              if key not in (TEST_SET_OPENS, TEST_SET_V2_OPENS)]
    if quoted:
        report_path = root / numbers
        if not report_path.exists():
            problems.append(f"{shown[readme]}: {len(quoted)} marker(s) but no report at {shown[numbers]}")
        else:
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except ValueError as exc:
                report = None
                problems.append(f"{shown[numbers]}: not valid JSON ({exc})")
            if report is not None:
                values = report.get("numbers", {})
                must_track.append(shown[numbers])
                for source, digest in report.get("sources", {}).items():
                    must_track.append(source)
                    path = root / source
                    if not path.exists():
                        problems.append(f"{shown[numbers]}: source {source} is missing")
                    elif sha256_of(path) != digest:
                        problems.append(f"{shown[numbers]}: source {source} changed since the report was "
                                        f"built; build it again")
                for key, value, line in quoted:
                    if key not in values:
                        problems.append(f"{shown[readme]}:{line}: {key} is not in {shown[numbers]}")
                    elif value != values[key]:
                        problems.append(f"{shown[readme]}:{line}: {key} is {value!r}, the report says "
                                        f"{values[key]!r}")

    if tracked is not None and must_track:
        known = tracked(must_track)
        problems += [f"{path} is not tracked by git: add it, or a clean checkout fails this check"
                     for path in must_track if path not in known]
    return problems

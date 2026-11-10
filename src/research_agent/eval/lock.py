"""Hash locks for the evaluation sets, and the access log of the test set.

A set is a JSONL file, one item per line. Its lock is <path>.sha256, one line in the sha256sum format: the hex
digest, two spaces, the file name (write_lock and scripts/lock_set.py write exactly that). read_lock accepts
the variants sha256sum itself writes and Windows tools add on the way: "*" before the name (binary mode), hex
in either case, a CRLF line end, surrounding whitespace and a UTF-8 byte-order mark. It still refuses a lock
that names another file, a digest that is not 64 hex digits, and a lock with more than one line. check_lock
refuses a set that has no lock or whose bytes no longer give the digest in the lock. The test set is opened
through LockedSet: the lock is checked, the open is refused once the access log holds max_opens entries
(three over the whole project, docs/plan.md "测试集锁"), and every successful open appends one line to the
log and returns the items as an OpenedTestSet (who opens what, below). The log is committed, so apart from
its timestamps it is deterministic, and a refused or failed open leaves it as it is. The development set is
opened through DevSet: the same lock check, no limit, no log.

The locked sets live in data/eval, relative to the repository root, and are committed together with their
locks and the access log, so that a checkout holds exactly what the reported numbers were measured on
(.gitattributes keeps data/eval/*.jsonl and *.sha256 byte-exact on every platform):

    data/eval/dev.jsonl           the development set, opened through DevSet
    data/eval/dev.jsonl.sha256    its lock (DevSet makes the same lock check)
    data/eval/test.jsonl          the test set, opened through LockedSet
    data/eval/test.jsonl.sha256   its lock
    data/eval/test-access.jsonl   the access log: one line per open, at most MAX_TEST_SET_OPENS lines
    data/eval/e2e-dev.jsonl       the end-to-end development tasks, lock-checked like the development set
    data/eval/e2e-test.jsonl      the end-to-end test tasks, with their own lock and their own access log,
                                  data/eval/e2e-test-access.jsonl (the M3 plan, default 4)
    data/eval/e2e-test-retired-reads.jsonl  the reads of the retired end-to-end test tasks (RetiredSet, below)
    data/eval/test-retired-reads.jsonl  the reads of the retired single-step test items
                                  (RetiredSet.single_step, below)
    data/eval/handwritten-specs-retired-reads.jsonl  the reads of v1's hand-written specs
                                  (RetiredSet.handwritten_specs, below)
    data/eval/test-completions.jsonl  the completion markers of the opens: one line per open of either log,
                                  written when the phase-4 test runner's session reached its end, or when the
                                  owner abandoned an open whose session did not (scripts/run_test.py)

Beside them, without a lock, two inputs of the set assembly (scripts/finalize_eval_sets.py) that hold the
test set's handwritten items with their gold turns: data/eval/handwritten.jsonl, the handwritten items of
both splits, and data/eval/handwritten-specs.jsonl, the specs they were built from.

The second family, v2 (M7, docs/m7-计划.md: the first family's three opens were used up by M3, M5 and M6),
lives in data/eval/v2 with the same layout and its own access logs and completion markers, so it has its own
three opens: data/eval/v2/test-v2.jsonl and data/eval/v2/e2e-test-v2.jsonl with their locks, their logs
test-v2-access.jsonl and e2e-test-v2-access.jsonl, the markers test-completions-v2.jsonl, and the handwritten
files handwritten-v2.jsonl and handwritten-specs-v2.jsonl. TEST_FAMILIES names both families' sets, logs and
marker files for the phase-4 test runner (its configuration's "test_sets"); the guards below cover the sets
and handwritten files of both.

Retired sets. A test set whose access log holds max_opens entries can never be opened again; v1's end-to-end
tasks, data/eval/e2e-test.jsonl, are one since M6. RetiredSet reads such a set for one purpose only: how close
new texts come to its task texts, so that a new test set (the v2 end-to-end tasks, docs/m7-计划.md WP-I 6;
scripts/check_e2e_tasks.py) is kept apart from it. RetiredSet.similarities(texts) returns, for each text in
order, the largest masked 5-gram Jaccard against the set's task texts (data.requests: jaccard over the
shingles of mask_text, the measure of the leak rule and of the contract tests), as a float, and nothing else
of the set: no id, no text, no field. It refuses, before anything is read, texts that are not a list of
strings; then a set that is not retired (fewer opens in its access log than max_opens) and a set that does
not match its lock; and every read it answers appends one line to its reads log,
data/eval/e2e-test-retired-reads.jsonl for v1's end-to-end tasks: {"lock_ok", "sha256", "texts", "ts"} in
canonical key order (the lock was checked and matched, the set's digest, the number of texts, the time),
never a text or a field of a task. A retired read is no open: the access log is read to count the opens and
never written, so it keeps its line count. The guards below are not changed by it: refuse_test_sets still
refuses the retired set to every runner, and a test run still cannot open it outside a test marked
test_material (tests/conftest.py), RetiredSet's reads included.

v1's single-step test set is retired as well (its access log, data/eval/test-access.jsonl, holds three
opens since M6), and with it the hand-written specs its handwritten items were built from. Two named
constructors read them the same way, so that the v2 hand-written items are kept apart from them
(docs/m7-计划.md WP-I 4; scripts/build_handwritten_items.py --family v2). RetiredSet.single_step reads
data/eval/test.jsonl: an item's text is its user's request, the first user message of its history (not
the system prompt, a tool output or the expected turn); the reads go to data/eval/test-retired-reads.jsonl.
RetiredSet.handwritten_specs reads data/eval/handwritten-specs.jsonl: a spec's text is its request, of the
dev specs and the test specs alike (not its rationale or expected turn); the reads go to
data/eval/handwritten-specs-retired-reads.jsonl. Both count the opens in the single-step test set's access
log. The specs have no lock of their own, so their bytes are checked against the SHA-256 the assembly
recorded for them in data/eval/sets-manifest.json (inputs.handwritten_specs; check_recorded), and
"lock_ok" in their reads log means that check. Everything else is as above: one float per text, one
log line per read, no access log written, and a refusal names an entry by its position alone (item 2,
spec 7, as task 3 for an end-to-end task).

The paths are the EVAL_DIR, DEV_SET, TEST_SET, TEST_ACCESS_LOG, E2E_* and HANDWRITTEN_* constants below; the
M2 builders write the sets and scripts/lock_set.py writes the locks.

Who opens what. Only the phase-4 test runner (scripts/run_test.py) opens a test set, through LockedSet, and
only it evaluates test material. The scripts that evaluate a file of items or tasks (the single-step and
end-to-end runners scripts/run_eval.py and scripts/run_e2e.py, scripts/select_checkpoint.py,
scripts/compare_backends.py and scripts/score.py) guard their input twice, with the same exit code and the
same kind of message. Before reading it, refuse_test_sets refuses a test set by location, however the path is
spelled (named_set: resolved names, then os.path.samefile, so a link to it is refused too; only file metadata
is consulted), or by its bytes, when the file's SHA-256 is the digest in a test set's lock (a copy under
another name; only the locks are read, never a test set), and the two files of handwritten items
(HANDWRITTEN_SETS) by location in the same way. Once read, refuse_test_material refuses the input by its
content: any item or task whose split is "test", or whose id or source.request is an id of the test split
(TEST_ID: a test request req-test-<nnnn> and the items drafted from it, <request id>-s<step>; a handwritten
item hw-test-<nnn>; an end-to-end task e2e-test-<nn>). So a subset of a test set, a copy with CRLF line ends
or without the final newline, and a copy of the handwritten items are refused too; the check reads the input
only, never a test set. The list-level library functions that evaluate (eval.runner.run_items,
eval.e2e.run_tasks and eval.metrics.score_items) make the same content check on what they are given; the
per-item functions they are built on (eval.metrics.score_item, eval.e2e.score_task, data.synth.run_request)
have no guard, so code calling those directly is outside this protection. What passes is an OpenedTestSet, the
list LockedSet.open returns, which carries the log entry of that open: the phase-4 runner passes it to those
functions as it is. A slice of it, or a list() of it, is a plain list again and refused, while copy.copy and
copy.deepcopy keep the type: the guards stop accidents, they are no barrier against code written to get past
them. A list the runner derives on purpose (validated copies of the end-to-end tasks, say) it wraps again as
OpenedTestSet(derived, opened.entry), and nothing else constructs one. Apart from the runner, only the
contract tests of tests/eval/test_e2e.py read a test file (data/eval/e2e-test.jsonl: its size, splits, dates
and topics; data/eval/v2/e2e-test-v2.jsonl likewise; no system runs on them), and only in a pytest run with
--test-material, and RetiredSet reads a retired one for similarities alone (above): other runs are refused any
open of test material by path, links included (tests/conftest.py; an open relative to a directory descriptor
is not recognized). The tools that build the sets from the handwritten files
(scripts/build_handwritten_items.py, scripts/finalize_eval_sets.py, the review scripts in tools/eval-review)
read them and call neither guard; eval.handwritten checks every item it builds with score_item, the per-item
scorer, which has no guard. The two runners, scripts/select_checkpoint.py and scripts/compare_backends.py, not
scripts/score.py, also lock-check a development set (named_set over DEV_SETS) and any other input that has a
lock beside it before reading it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from ..canonical import canonical_json, split_lines
from ..data.requests import jaccard, mask_text, shingles
from ..net.fetch import utc_now

LOCK_SUFFIX = ".sha256"
MAX_TEST_SET_OPENS = 3  # the plan's limit; README states how many opens were used
EVAL_DIR = Path("data/eval")  # relative to the repository root, where the scripts run
DEV_SET = EVAL_DIR / "dev.jsonl"
TEST_SET = EVAL_DIR / "test.jsonl"
TEST_ACCESS_LOG = EVAL_DIR / "test-access.jsonl"
E2E_DEV_SET = EVAL_DIR / "e2e-dev.jsonl"
E2E_TEST_SET = EVAL_DIR / "e2e-test.jsonl"
E2E_TEST_ACCESS_LOG = EVAL_DIR / "e2e-test-access.jsonl"  # written by the phase-4 test runner
E2E_TEST_RETIRED_READS = EVAL_DIR / "e2e-test-retired-reads.jsonl"  # RetiredSet's reads of those tasks
TEST_COMPLETIONS = EVAL_DIR / "test-completions.jsonl"  # the opens' completion markers (scripts/run_test.py)
# The handwritten items with their gold turns, test items among them, and the specs they were built from:
# inputs of scripts/finalize_eval_sets.py, which refuse_test_sets refuses by location like the test sets.
HANDWRITTEN_ITEMS = EVAL_DIR / "handwritten.jsonl"
HANDWRITTEN_SPECS = EVAL_DIR / "handwritten-specs.jsonl"
# The reads of v1's two other retired files (RetiredSet.single_step, RetiredSet.handwritten_specs), and the
# assembly's manifest (scripts/finalize_eval_sets.py), whose SHA-256 of the specs stands in for a lock.
TEST_RETIRED_READS = EVAL_DIR / "test-retired-reads.jsonl"
HANDWRITTEN_SPECS_RETIRED_READS = EVAL_DIR / "handwritten-specs-retired-reads.jsonl"
SETS_MANIFEST = EVAL_DIR / "sets-manifest.json"
# The second family of test sets (M7), with its own logs, markers and handwritten files (module docstring).
V2_DIR = EVAL_DIR / "v2"
TEST_SET_V2 = V2_DIR / "test-v2.jsonl"
TEST_ACCESS_LOG_V2 = V2_DIR / "test-v2-access.jsonl"
E2E_TEST_SET_V2 = V2_DIR / "e2e-test-v2.jsonl"
E2E_TEST_ACCESS_LOG_V2 = V2_DIR / "e2e-test-v2-access.jsonl"
TEST_COMPLETIONS_V2 = V2_DIR / "test-completions-v2.jsonl"
HANDWRITTEN_ITEMS_V2 = V2_DIR / "handwritten-v2.jsonl"
HANDWRITTEN_SPECS_V2 = V2_DIR / "handwritten-specs-v2.jsonl"
DEV_SETS = (DEV_SET, E2E_DEV_SET)
TEST_SETS = (TEST_SET, E2E_TEST_SET, TEST_SET_V2, E2E_TEST_SET_V2)
HANDWRITTEN_SETS = (HANDWRITTEN_ITEMS, HANDWRITTEN_SPECS, HANDWRITTEN_ITEMS_V2, HANDWRITTEN_SPECS_V2)
TEST_SPLIT = "test"
# The ids of test material (the module docstring: who opens what), matched at the start of an item's or task's
# id and of a drafted item's source.request: a test request req-test-<nnnn> (data.requests numbers them) and
# the items drafted from it, <request id>-s<step> (eval.items); a handwritten item hw-test-<nnn>
# (eval.handwritten.ID_PATTERN); an end-to-end task e2e-test-<nn>.
TEST_ID = re.compile(r"(?:req|hw|e2e)-test-\d")
# One sha256sum entry: the digest, whitespace, "*" (binary mode) or nothing, the file name; surrounding
# whitespace is ignored.
_ENTRY = re.compile(r"\s*([0-9A-Fa-f]{64})\s+\*?(\S.*?)\s*")

Now = Callable[[], str]  # now() -> UTC time in ISO 8601; net.fetch.utc_now unless a test injects a clock


@dataclass(frozen=True)
class TestFamily:
    """A family of test sets the phase-4 test runner opens together: (set, access log) for the single-step
    and the end-to-end set, and the file of the opens' completion markers."""

    __test__ = False  # not a pytest test class, whatever its name

    sets: tuple[tuple[Path, Path], tuple[Path, Path]]
    completions: Path


TEST_FAMILIES = {
    "v1": TestFamily(((TEST_SET, TEST_ACCESS_LOG), (E2E_TEST_SET, E2E_TEST_ACCESS_LOG)), TEST_COMPLETIONS),
    "v2": TestFamily(((TEST_SET_V2, TEST_ACCESS_LOG_V2), (E2E_TEST_SET_V2, E2E_TEST_ACCESS_LOG_V2)),
                     TEST_COMPLETIONS_V2),
}


class LockError(RuntimeError):
    """A set refused: no lock, a lock that does not match, or no test-set open left."""


class RefusedTestSet(LockError):
    """A locked test set given to a runner that may not open it (refuse_test_sets)."""


class JsonlError(ValueError):
    """A JSONL file with a line that is not a JSON object."""


def sha256_of(path: str | Path) -> str:
    """Hex SHA-256 of the file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lock_path(path: str | Path) -> Path:
    """<path>.sha256, next to the file."""
    path = Path(path)
    return path.with_name(path.name + LOCK_SUFFIX)


def write_lock(path: str | Path) -> str:
    """Write <path>.sha256 ("<hex>  <file name>", one line, LF) and return the digest."""
    path = Path(path)
    digest = sha256_of(path)
    lock_path(path).write_text(f"{digest}  {path.name}\n", encoding="utf-8", newline="\n")
    return digest


def read_lock(path: str | Path) -> str:
    """The digest in <path>.sha256 as lower-case hex.

    LockError when the lock is missing, malformed (not exactly one sha256sum line) or names another file.
    """
    path = Path(path)
    lock = lock_path(path)
    if not lock.is_file():
        raise LockError(f"{path.name} has no lock: {lock.name} is missing")
    lines = [line for line in split_lines(lock.read_text(encoding="utf-8-sig")) if line.strip()]
    if len(lines) != 1:
        raise LockError(f"{lock.name} is malformed: {len(lines)} lines where one '<sha256 hex>  {path.name}' "
                        "was expected")
    entry = _ENTRY.fullmatch(lines[0])
    if entry is None:
        raise LockError(f"{lock.name} is malformed; expected '<sha256 hex>  {path.name}', got {lines[0]!r}")
    digest, name = entry.group(1).lower(), entry.group(2)
    if name != path.name:
        raise LockError(f"{lock.name} names {name}, not {path.name}")
    return digest


def check_lock(path: str | Path) -> str:
    """The file's digest when it matches the lock; LockError when the lock is missing or does not match."""
    path = Path(path)
    expected = read_lock(path)
    actual = sha256_of(path)
    if actual != expected:
        raise LockError(f"{path.name} does not match its lock: sha256 {actual}, lock says {expected}")
    return actual


def check_recorded(path: str | Path, *, manifest: str | Path, key: str) -> str:
    """The file's digest when it matches the SHA-256 manifest records for it under inputs[key] ({"path",
    "sha256"}, as scripts/finalize_eval_sets.py writes data/eval/sets-manifest.json): the check of a file
    that has no lock of its own. LockError when the manifest cannot be read or records no such SHA-256,
    when the record names another file, or when the bytes no longer give the recorded digest."""
    path, manifest = Path(path), Path(manifest)
    missing = f"{manifest.name} records no SHA-256 of {path.name} (inputs.{key})"
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))["inputs"][key]
        name, expected = Path(record["path"]).name, record["sha256"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise LockError(missing) from exc
    if not isinstance(expected, str) or re.fullmatch(r"[0-9A-Fa-f]{64}", expected) is None:
        raise LockError(missing)
    if name != path.name:
        raise LockError(f"{manifest.name}: inputs.{key} names {name}, not {path.name}")
    actual = sha256_of(path)
    if actual != expected.lower():
        raise LockError(f"{path.name} does not match the SHA-256 {manifest.name} records for it: sha256 "
                        f"{actual}, recorded {expected.lower()}")
    return actual


def _same_file(path: Path, other: Path) -> bool:
    if os.path.normcase(os.path.realpath(path)) == os.path.normcase(os.path.realpath(other)):
        return True
    try:
        return os.path.samefile(path, other)  # another name of the same file, e.g. a hard link
    except OSError:  # one of the two does not exist
        return False


def named_set(path: str | Path, sets: Iterable[str | Path], roots: Iterable[str | Path]) -> Path | None:
    """root / set for the first root and set that path names (the module docstring: who opens what), or None.

    Compared resolved (os.path.realpath, case folded where the file system folds it), then, when both files
    exist, with os.path.samefile; no file is read.
    """
    sets = [Path(s) for s in sets]
    for root in roots:
        for name in sets:
            if _same_file(Path(path), Path(root) / name):
                return Path(root) / name
    return None


def refuse_test_sets(path: str | Path, roots: Iterable[str | Path]) -> None:
    """RefusedTestSet when path is a locked test set (TEST_SETS under any of roots), by location or by its
    bytes (a test set's digest), or a file of the handwritten test items (HANDWRITTEN_SETS), by location (the
    module docstring: who opens what); only the phase-4 test runner opens a test set, through LockedSet. What
    the file holds is refuse_test_material's to check, once it is read."""
    roots = list(roots)
    hit = named_set(path, TEST_SETS, roots)
    if hit is not None:
        raise RefusedTestSet(f"{path} is the locked test set {hit.as_posix()}; only the phase-4 test runner "
                             "(scripts/run_test.py) opens it, through LockedSet and its access log")
    hit = named_set(path, HANDWRITTEN_SETS, roots)
    if hit is not None:
        raise RefusedTestSet(f"{path} is {hit.as_posix()}, which holds the test set's handwritten items with "
                             "their gold turns; only the phase-4 test runner (scripts/run_test.py) evaluates "
                             "them, in the test set it opens through LockedSet and its access log")
    digests: dict[str, Path] = {}
    for root in roots:
        for name in TEST_SETS:
            try:
                digests.setdefault(read_lock(Path(root) / name), name)
            except (LockError, ValueError):  # no readable lock under that root (a test's folder)
                continue
    if digests and Path(path).is_file():
        name = digests.get(sha256_of(path))
        if name is not None:
            raise RefusedTestSet(f"{path} has the SHA-256 of the locked test set {name.as_posix()} (a copy "
                                 "of it); only the phase-4 test runner (scripts/run_test.py) opens it, "
                                 "through LockedSet and its access log")


def find_test_material(items: Iterable[object]) -> list[str]:
    """The items or tasks of items that are test material (the module docstring: who opens what): split
    "test", or an id or a source.request of the test split (TEST_ID). Each is named by its id, or by its
    position (#1 for the first) when it has none; an entry that is not an object is passed over."""
    found = []
    for number, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        source = item.get("source")
        ids = (item.get("id"), source.get("request") if isinstance(source, dict) else None)
        if item.get("split") == TEST_SPLIT or any(isinstance(i, str) and TEST_ID.match(i) for i in ids):
            found.append(item["id"] if isinstance(item.get("id"), str) else f"#{number}")
    return found


class OpenedTestSet(list):
    """The items of one successful LockedSet.open, in file order: the form in which the phase-4 test runner
    evaluates test material (refuse_test_material lets it through; the module docstring: who opens what).
    entry is that open's line of the access log. A slice of it, or another list made from it, is a plain list
    again."""

    def __init__(self, items: Iterable[dict], entry: dict) -> None:
        super().__init__(items)
        self.entry = entry


def refuse_test_material(items: Iterable[object], source: str | Path) -> None:
    """RefusedTestSet when items (read from source: a path, or what a function was given) hold test material
    (find_test_material), unless they are an OpenedTestSet; the module docstring: who opens what."""
    if isinstance(items, OpenedTestSet):
        return
    found = find_test_material(items)
    if found:
        raise RefusedTestSet(f"{source} holds test material ({len(found)} item(s) with split "
                             f"{TEST_SPLIT!r} or an id of the test split, the first {found[0]}); only the "
                             "phase-4 test runner (scripts/run_test.py) evaluates it, as the items "
                             "LockedSet.open returns through its access log")


def lock_required(path: str | Path, roots: Iterable[str | Path]) -> bool:
    """Whether a runner must lock-check path before reading it: a development set (named_set over DEV_SETS,
    whose lock must then exist) or any other file with a lock beside it."""
    return lock_path(path).is_file() or named_set(path, DEV_SETS, roots) is not None


def read_jsonl(path: str | Path) -> list[dict]:
    """One JSON object per line; blank lines are skipped, any other line is a JsonlError."""
    path = Path(path)
    items = []
    with path.open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JsonlError(f"{path.name}: line {number} is not JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise JsonlError(f"{path.name}: line {number} is not a JSON object")
            items.append(value)
    return items


def write_jsonl(items: list[dict], path: str | Path) -> None:
    """One canonical JSON line per item (sorted keys, UTF-8, LF line ends); parent folders are created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(item) + "\n" for item in items), encoding="utf-8", newline="\n")


class LockedSet:
    """The test set: opened only with a matching lock, at most max_opens times, every open logged.

    The access log at log_path is JSONL, one {"milestone", "purpose", "sha256", "ts"} per successful open in
    canonical key order, so two logs made with the same clock are byte-identical. Entries written by earlier
    sessions count toward the limit, whatever digest they carry. open returns the items as an OpenedTestSet,
    the only items that pass refuse_test_material.
    """

    def __init__(self, path: str | Path, log_path: str | Path, *, max_opens: int = MAX_TEST_SET_OPENS,
                 now: Now = utc_now) -> None:
        self.path, self.log_path, self.max_opens, self.now = Path(path), Path(log_path), max_opens, now
        self.last_open: dict | None = None  # the log entry of this object's latest successful open

    def log_entries(self) -> list[dict]:
        """The access log so far; empty when it does not exist yet. A corrupt log is a LockError."""
        if not self.log_path.exists():
            return []
        try:
            return read_jsonl(self.log_path)
        except JsonlError as exc:
            raise LockError(f"access log {self.log_path.name} is not valid JSONL: {exc}") from exc

    def opens_left(self) -> int:
        return max(self.max_opens - len(self.log_entries()), 0)

    def open(self, milestone: str, purpose: str) -> OpenedTestSet:
        """The items, when the lock matches and an open is left, as an OpenedTestSet carrying the log entry;
        the open goes to the log.

        milestone and purpose must be given: the log has to say what every open was for. The lock, the limit
        and the items are all checked before anything is written, so a failed open leaves the log untouched.
        """
        if not milestone.strip() or not purpose.strip():
            raise LockError("a test-set open must state its milestone and purpose")
        digest = check_lock(self.path)
        used = len(self.log_entries())
        if used >= self.max_opens:
            raise LockError(f"{self.path.name} has been opened {used} times; the limit is {self.max_opens}")
        items = read_jsonl(self.path)
        entry = {"ts": self.now(), "milestone": milestone, "purpose": purpose, "sha256": digest}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(canonical_json(entry) + "\n")
        self.last_open = entry
        return OpenedTestSet(items, entry)


class DevSet:
    """The development set: the same lock check, opened as often as needed, nothing written anywhere."""

    def __init__(self, path: str | Path, *, now: Now = utc_now) -> None:
        self.path, self.now = Path(path), now
        self.last_open: dict | None = None  # {"ts", "sha256"} of the latest open, for a run's own record

    def open(self) -> list[dict]:
        """The items when the lock matches; LockError otherwise."""
        digest = check_lock(self.path)
        items = read_jsonl(self.path)
        self.last_open = {"ts": self.now(), "sha256": digest}
        return items


class RetiredSet:
    """A test set whose opens are used up, read for one purpose only: the similarity of new texts to its task
    texts (the module docstring: retired sets). The defaults are v1's end-to-end tasks, their access log and
    their reads log, relative to the repository root.

    The keyword arguments read another retired file the same way: text_of(entry) gives an entry's text
    (None: its "text", as an end-to-end task holds it), entry is what a refusal calls an entry, and
    digest_of(path) checks the file's bytes and gives its digest (None: check_lock, the lock beside it).
    single_step and handwritten_specs build the readers of v1's two other retired files."""

    def __init__(self, path: str | Path = E2E_TEST_SET, access_log: str | Path = E2E_TEST_ACCESS_LOG,
                 reads_log: str | Path = E2E_TEST_RETIRED_READS, *, max_opens: int = MAX_TEST_SET_OPENS,
                 now: Now = utc_now, text_of: Callable[[dict], object] | None = None, entry: str = "task",
                 digest_of: Callable[[Path], str] | None = None) -> None:
        self.path, self.access_log, self.reads_log = Path(path), Path(access_log), Path(reads_log)
        self.max_opens, self.now = max_opens, now
        self.text_of, self.entry, self.digest_of = text_of, entry, digest_of

    @classmethod
    def single_step(cls, root: str | Path = ".", **options) -> RetiredSet:
        """v1's single-step test items, data/eval/test.jsonl under root (the module docstring: retired sets):
        an item's text is its user's request; its lock and its access log; the reads logged in
        data/eval/test-retired-reads.jsonl under root. options: max_opens, now."""
        root = Path(root)
        return cls(root / TEST_SET, root / TEST_ACCESS_LOG, root / TEST_RETIRED_READS, text_of=_item_request,
                   entry="item", **options)

    @classmethod
    def handwritten_specs(cls, root: str | Path = ".", **options) -> RetiredSet:
        """v1's hand-written specs, data/eval/handwritten-specs.jsonl under root (the module docstring:
        retired sets): a spec's text is its request, dev and test specs alike; retired with the single-step
        test set, whose access log counts the opens; checked against the SHA-256 data/eval/sets-manifest.json
        records for them (check_recorded); the reads logged in data/eval/handwritten-specs-retired-reads.jsonl
        under root. options: max_opens, now."""
        root = Path(root)
        return cls(root / HANDWRITTEN_SPECS, root / TEST_ACCESS_LOG, root / HANDWRITTEN_SPECS_RETIRED_READS,
                   text_of=_spec_request, entry="spec",
                   digest_of=partial(check_recorded, manifest=root / SETS_MANIFEST, key="handwritten_specs"),
                   **options)

    def similarities(self, texts: Sequence[str]) -> list[float]:
        """For each text, in order, the largest masked 5-gram Jaccard against the set's task texts (0.0 for a
        set without tasks), and nothing else of the set; the read goes to the reads log.

        TypeError, before anything is read, when texts is not a sequence of strings (a bare string is not
        one); LockError when the set is not retired (its access log holds fewer than max_opens opens), has no
        lock or does not match it, or holds a task without a text (named by its position, never by a field):
        nothing is logged then. The access log is read, never written. With text_of, entry and digest_of
        (the class docstring) the texts, the name of an entry and the check of the bytes are theirs.
        """
        if isinstance(texts, str | bytes) or not isinstance(texts, Sequence):
            raise TypeError("texts must be a list of strings, not a single string or another object")
        if not all(isinstance(text, str) for text in texts):
            raise TypeError("texts must be a list of strings")
        used = len(LockedSet(self.path, self.access_log, max_opens=self.max_opens).log_entries())
        if used < self.max_opens:
            raise LockError(f"{self.path.name} is not retired: its access log {self.access_log.name} holds "
                            f"{used} of its {self.max_opens} opens, so it is read only by opening it")
        digest = check_lock(self.path) if self.digest_of is None else self.digest_of(self.path)
        retired = [shingles(mask_text(text)) for text in _task_texts(self.path, self.text_of, self.entry)]
        scores = [max((jaccard(shingles(mask_text(text)), other) for other in retired), default=0.0)
                  for text in texts]
        entry = {"ts": self.now(), "texts": len(texts), "sha256": digest, "lock_ok": True}
        self.reads_log.parent.mkdir(parents=True, exist_ok=True)
        with self.reads_log.open("a", encoding="utf-8", newline="\n") as f:
            f.write(canonical_json(entry) + "\n")
        return [float(score) for score in scores]


def _task_texts(path: Path, text_of: Callable[[dict], object] | None = None,
                entry: str = "task") -> list[str]:
    """The text of every task of the set at path, in file order (text_of(task), its "text" when None); a
    LockError names a task without one by its position alone, as an entry of that name."""
    texts = []
    for number, task in enumerate(read_jsonl(path), 1):
        text = task.get("text") if text_of is None else text_of(task)
        if not isinstance(text, str) or not text.strip():
            raise LockError(f"{path.name}: {entry} {number} has no text")
        texts.append(text)
    return texts


def _item_request(item: dict) -> object:
    """A single-step item's text for RetiredSet.single_step: its user's request, the content of the first user
    message of its history; None without one."""
    messages = item.get("messages")
    for message in messages if isinstance(messages, list) else []:
        if isinstance(message, dict) and message.get("role") == "user":
            return message.get("content")
    return None


def _spec_request(spec: dict) -> object:
    """A hand-written spec's text for RetiredSet.handwritten_specs: its request."""
    return spec.get("request")

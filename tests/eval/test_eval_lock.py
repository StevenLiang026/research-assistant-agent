import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from research_agent.canonical import canonical_json
from research_agent.data.requests import TRAIN_VS_EVAL, jaccard, mask_text, shingles
from research_agent.eval import lock as lock_module
from research_agent.eval.lock import (
    DEV_SETS,
    TEST_SETS,
    DevSet,
    LockedSet,
    LockError,
    OpenedTestSet,
    RefusedTestSet,
    check_lock,
    find_test_material,
    lock_path,
    lock_required,
    named_set,
    read_jsonl,
    read_lock,
    refuse_test_material,
    refuse_test_sets,
    sha256_of,
    write_jsonl,
    write_lock,
)

ITEMS = [{"id": "t001", "messages": [{"role": "user", "content": "找推测解码的论文"}],
          "expected": {"tool": "search_arxiv"}},
         {"id": "t002", "messages": [{"role": "user", "content": "hi"}], "expected": {"tool": None}}]
STAMPS = ["2026-10-20T10:00:00+00:00", "2026-11-01T09:30:00+00:00", "2026-11-15T20:00:00+00:00"]
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "lock_set.py"


def clock(*stamps: str):
    """now() stand-in: the given timestamps in order."""
    it = iter(stamps)
    return lambda: next(it)


@pytest.fixture
def locked(tmp_path):
    """A locked two-item set at tmp_path/test.jsonl."""
    path = tmp_path / "test.jsonl"
    write_jsonl(ITEMS, path)
    write_lock(path)
    return path


def edit(path: Path) -> None:
    """Change the file by one byte."""
    path.write_bytes(path.read_bytes() + b"\n")


# the lock


def test_lock_round_trip(tmp_path):
    path = tmp_path / "test.jsonl"
    write_jsonl(ITEMS, path)
    digest = write_lock(path)
    assert digest == sha256_of(path) == hashlib.sha256(path.read_bytes()).hexdigest()
    assert lock_path(path) == tmp_path / "test.jsonl.sha256"
    assert lock_path(path).read_bytes() == f"{digest}  test.jsonl\n".encode()
    assert check_lock(path) == digest
    assert read_jsonl(path) == ITEMS  # UTF-8 and key order survive the round trip


def test_an_edited_set_is_refused(locked):
    edit(locked)
    with pytest.raises(LockError, match="does not match its lock"):
        check_lock(locked)


def test_a_set_without_a_lock_is_refused(tmp_path):
    path = tmp_path / "test.jsonl"
    write_jsonl(ITEMS, path)
    with pytest.raises(LockError, match="has no lock"):
        check_lock(path)
    lock_path(path).write_text("not a digest\n", encoding="utf-8")
    with pytest.raises(LockError, match="malformed"):
        check_lock(path)


def test_a_lock_that_names_another_file_is_refused(locked):
    other = locked.with_name("other.jsonl")
    other.write_bytes(locked.read_bytes())
    lock_path(other).write_bytes(lock_path(locked).read_bytes())  # right digest, wrong name
    with pytest.raises(LockError, match="names test.jsonl, not other.jsonl"):
        check_lock(other)


def test_read_lock_accepts_what_sha256sum_writes(locked):
    digest = sha256_of(locked)
    lock = lock_path(locked)
    variants = {
        "binary mode": f"{digest} *test.jsonl\n",
        "upper-case hex": f"{digest.upper()}  test.jsonl\n",
        "CRLF": f"{digest}  test.jsonl\r\n",
        "surrounding whitespace": f"\n  {digest}  test.jsonl   \n\n",
        "UTF-8 byte-order mark": f"﻿{digest}  test.jsonl\n",
        "no final newline": f"{digest}  test.jsonl",
    }
    for label, text in variants.items():
        lock.write_bytes(text.encode("utf-8"))
        assert read_lock(locked) == digest, label
        assert check_lock(locked) == digest, label


def test_read_lock_still_refuses_another_name_a_bad_digest_or_a_second_entry(locked):
    digest = sha256_of(locked)
    lock = lock_path(locked)
    refused = {
        "another file": (f"{digest} *other.jsonl\n", "names other.jsonl, not test.jsonl"),
        "a path": (f"{digest}  data/eval/test.jsonl\n", "names data/eval/test.jsonl, not test.jsonl"),
        "63 hex digits": (f"{digest[:-1]}  test.jsonl\n", "malformed"),
        "a non-hex digest": (f"{'g' * 64}  test.jsonl\n", "malformed"),
        "no separator": (f"{digest}test.jsonl\n", "malformed"),
        "a second entry": (f"{digest}  test.jsonl\n{digest}  dev.jsonl\n", "malformed: 2 lines"),
        "an empty file": ("", "malformed"),
    }
    for text, message in refused.values():
        lock.write_bytes(text.encode("utf-8"))
        with pytest.raises(LockError, match=message):
            read_lock(locked)
    lock.write_bytes(f"{digest} *test.jsonl\n".encode())
    edit(locked)
    with pytest.raises(LockError, match="does not match its lock"):
        check_lock(locked)


def test_the_locked_sets_live_in_data_eval():
    eval_dir = lock_module.EVAL_DIR
    assert eval_dir == Path("data/eval")
    assert (lock_module.DEV_SET, lock_module.TEST_SET) == (eval_dir / "dev.jsonl", eval_dir / "test.jsonl")
    assert lock_path(lock_module.TEST_SET) == eval_dir / "test.jsonl.sha256"
    assert lock_path(lock_module.DEV_SET) == eval_dir / "dev.jsonl.sha256"
    assert lock_module.TEST_ACCESS_LOG == eval_dir / "test-access.jsonl"
    for name in ("data/eval/dev.jsonl", "data/eval/test.jsonl", "test.jsonl.sha256", "test-access.jsonl"):
        assert name in lock_module.__doc__, name


def test_jsonl_lines_must_be_json_objects(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "a"}\n[1, 2]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2 is not a JSON object"):
        read_jsonl(path)
    path.write_text('{"id": "a"}\n\n{"id": "b"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 3 is not JSON"):
        read_jsonl(path)
    path.write_text('{"id": "a"}\n\n', encoding="utf-8")
    assert read_jsonl(path) == [{"id": "a"}]  # blank lines are skipped


# the test set: three opens, every one logged


def test_three_opens_are_logged_and_the_fourth_is_refused(locked, tmp_path):
    log = tmp_path / "logs" / "access.jsonl"
    test_set = LockedSet(locked, log, now=clock(*STAMPS))
    digest = sha256_of(locked)
    assert test_set.opens_left() == 3 and test_set.log_entries() == [] and test_set.last_open is None
    purposes = ["FT vs B0/B1/B4", "FT+R vs B2", "second test"]
    for n, (milestone, purpose) in enumerate(zip(["M3", "M4", "M5"], purposes, strict=True), 1):
        assert test_set.open(milestone, purpose) == ITEMS
        assert test_set.opens_left() == 3 - n
    entries = [{"ts": ts, "milestone": m, "purpose": p, "sha256": digest}
               for ts, m, p in zip(STAMPS, ["M3", "M4", "M5"], purposes, strict=True)]
    assert test_set.log_entries() == entries and test_set.last_open == entries[-1]
    assert log.read_bytes() == "".join(canonical_json(e) + "\n" for e in entries).encode("utf-8")
    before = log.read_bytes()
    with pytest.raises(LockError, match="opened 3 times; the limit is 3"):
        test_set.open("M5", "one more")
    assert log.read_bytes() == before and test_set.opens_left() == 0


def test_log_lines_are_canonical_and_identical_for_the_same_clock(locked, tmp_path):
    logs = [tmp_path / "a" / "access.jsonl", tmp_path / "b" / "access.jsonl"]
    for log in logs:
        test_set = LockedSet(locked, log, now=clock(*STAMPS))
        test_set.open("M3", "主比较")
        test_set.open("M5", "second test")
    assert logs[0].read_bytes() == logs[1].read_bytes()
    first = logs[0].read_bytes().split(b"\n")[0].decode("utf-8")
    assert first == ('{"milestone":"M3","purpose":"主比较","sha256":"' + sha256_of(locked)
                     + '","ts":"2026-10-20T10:00:00+00:00"}')
    assert b"\r" not in logs[0].read_bytes()


def test_failed_opens_leave_the_log_untouched(tmp_path):
    path = tmp_path / "test.jsonl"
    write_jsonl(ITEMS, path)
    log = tmp_path / "access.jsonl"
    test_set = LockedSet(path, log, now=clock(*STAMPS))
    with pytest.raises(LockError, match="has no lock"):
        test_set.open("M3", "run")
    write_lock(path)
    original = path.read_bytes()
    edit(path)
    with pytest.raises(LockError, match="does not match"):
        test_set.open("M3", "run")
    path.write_bytes(original)
    with pytest.raises(LockError, match="milestone and purpose"):
        test_set.open("", "run")
    with pytest.raises(LockError, match="milestone and purpose"):
        test_set.open("M3", "  ")
    assert not log.exists() and test_set.last_open is None
    assert test_set.open("M3", "run") == ITEMS
    assert [e["ts"] for e in test_set.log_entries()] == [STAMPS[0]]  # the clock was only read for this one


def test_a_malformed_set_is_not_logged(tmp_path):
    path = tmp_path / "test.jsonl"
    path.write_text('{"id": "a"}\nnot json\n', encoding="utf-8")
    write_lock(path)
    log = tmp_path / "access.jsonl"
    with pytest.raises(ValueError, match="line 2"):
        LockedSet(path, log, now=clock(*STAMPS)).open("M3", "run")
    assert not log.exists()


def test_entries_from_earlier_sessions_count_toward_the_limit(locked, tmp_path):
    log = tmp_path / "access.jsonl"
    earlier = LockedSet(locked, log, now=clock(*STAMPS[:2]))
    earlier.open("M3", "a")
    earlier.open("M3", "b")
    later = LockedSet(locked, log, now=clock(STAMPS[2]))
    assert later.opens_left() == 1
    assert later.open("M5", "c") == ITEMS
    with pytest.raises(LockError, match="opened 3 times"):
        later.open("M5", "d")
    assert len(later.log_entries()) == 3
    assert LockedSet(locked, log, max_opens=4, now=clock("2026-12-01T00:00:00+00:00")).opens_left() == 1


def test_a_corrupt_log_refuses_the_open(locked, tmp_path):
    log = tmp_path / "access.jsonl"
    log.write_text("not json\n", encoding="utf-8")
    test_set = LockedSet(locked, log, now=clock(*STAMPS))
    with pytest.raises(LockError, match="access log access.jsonl is not valid JSONL"):
        test_set.open("M3", "run")
    assert log.read_text(encoding="utf-8") == "not json\n"


# the development set


def test_dev_set_is_unlimited_and_writes_nothing(locked, tmp_path):
    dev = DevSet(locked, now=clock(*STAMPS, *STAMPS))
    for _ in range(5):
        assert dev.open() == ITEMS
    assert dev.last_open == {"ts": STAMPS[1], "sha256": sha256_of(locked)}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["test.jsonl", "test.jsonl.sha256"]
    edit(locked)
    with pytest.raises(LockError, match="does not match"):
        dev.open()


# who may open which set: the runners refuse the test sets and lock-check the development sets


def test_the_end_to_end_sets_live_beside_the_single_step_sets():
    eval_dir = lock_module.EVAL_DIR
    assert (lock_module.E2E_DEV_SET, lock_module.E2E_TEST_SET) == (eval_dir / "e2e-dev.jsonl",
                                                                   eval_dir / "e2e-test.jsonl")
    assert lock_module.E2E_TEST_ACCESS_LOG == eval_dir / "e2e-test-access.jsonl"
    assert TEST_SETS[:2] == (lock_module.TEST_SET, lock_module.E2E_TEST_SET)
    assert DEV_SETS == (lock_module.DEV_SET, lock_module.E2E_DEV_SET)
    for name in ("data/eval/e2e-dev.jsonl", "data/eval/e2e-test.jsonl", "e2e-test-access.jsonl"):
        assert name in lock_module.__doc__, name


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A stand-in repository root holding fake sets (never the real ones), as the current folder."""
    root = tmp_path / "repo"
    (root / "data" / "eval").mkdir(parents=True)
    for name in (*TEST_SETS, *DEV_SETS):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("not json: a refused set is never parsed\n", encoding="utf-8")
    monkeypatch.chdir(root)
    return root


def test_a_test_set_is_refused_by_its_location_however_it_is_spelled(repo):
    spellings = ["data/eval/test.jsonl", "./data/eval/../eval/test.jsonl", "data/eval/e2e-test.jsonl",
                 repo / "data" / "eval" / "test.jsonl"]
    if os.name == "nt":  # the file system folds case and takes either separator
        spellings += ["DATA\\EVAL\\TEST.JSONL"]
    for spelling in spellings:
        with pytest.raises(RefusedTestSet, match="locked test set"):
            refuse_test_sets(spelling, roots=[repo])
    os.link(repo / "data" / "eval" / "test.jsonl", repo / "linked.jsonl")  # the same file under another name
    with pytest.raises(RefusedTestSet, match="locked test set"):
        refuse_test_sets("linked.jsonl", roots=[repo])
    for allowed in ("data/eval/dev.jsonl", "data/eval/e2e-dev.jsonl", "elsewhere/test.jsonl"):
        refuse_test_sets(allowed, roots=[repo])  # returns without a word


def test_the_handwritten_items_and_their_specs_are_refused_by_their_location(repo):
    """data/eval/handwritten.jsonl and handwritten-specs.jsonl hold the test set's handwritten items with
    their gold turns. They have no lock, so their location is what tells them, however the path is spelled."""
    for name in ("handwritten.jsonl", "handwritten-specs.jsonl"):
        write_jsonl(ITEMS, repo / "data" / "eval" / name)  # well-formed: refused all the same, never read
    spellings = ["data/eval/handwritten.jsonl", "./data/eval/../eval/handwritten-specs.jsonl",
                 repo / "data" / "eval" / "handwritten.jsonl"]
    if os.name == "nt":  # the file system folds case and takes either separator
        spellings += ["DATA\\EVAL\\HANDWRITTEN-SPECS.JSONL"]
    for spelling in spellings:
        with pytest.raises(RefusedTestSet, match="handwritten items with their gold turns"):
            refuse_test_sets(spelling, roots=[repo])
    os.link(repo / "data" / "eval" / "handwritten.jsonl", repo / "linked.jsonl")
    with pytest.raises(RefusedTestSet, match="phase-4 test runner"):
        refuse_test_sets("linked.jsonl", roots=[repo])
    for allowed in ("elsewhere/handwritten.jsonl", "data/eval/review/handwritten-verdicts.jsonl"):
        refuse_test_sets(allowed, roots=[repo])  # returns without a word
    assert lock_module.HANDWRITTEN_SETS[:2] == (lock_module.EVAL_DIR / "handwritten.jsonl",
                                                lock_module.EVAL_DIR / "handwritten-specs.jsonl")
    for name in ("data/eval/handwritten.jsonl", "data/eval/handwritten-specs.jsonl"):
        assert name in lock_module.__doc__, name


def test_a_copy_of_a_test_set_is_refused_by_the_digest_in_its_lock(repo):
    test_set = repo / "data" / "eval" / "test.jsonl"
    write_jsonl(ITEMS, test_set)
    write_lock(test_set)
    copy = repo / "copy.jsonl"
    copy.write_bytes(test_set.read_bytes())
    with pytest.raises(RefusedTestSet, match="SHA-256 of the locked test set"):
        refuse_test_sets(copy, roots=[repo])
    edit(copy)
    refuse_test_sets(copy, roots=[repo])  # no longer the test set's bytes
    refuse_test_sets(copy, roots=[repo / "elsewhere"])  # no lock to compare with


def test_named_set_finds_a_development_set_under_any_root(repo, tmp_path):
    dev = repo / "data" / "eval" / "dev.jsonl"
    assert named_set("data/eval/dev.jsonl", DEV_SETS, [tmp_path, repo]) == dev
    assert named_set(repo / "data/eval/e2e-dev.jsonl", DEV_SETS, [repo]) == repo / "data/eval/e2e-dev.jsonl"
    assert named_set("dev.jsonl", DEV_SETS, [repo]) is None
    assert issubclass(RefusedTestSet, LockError)


def test_a_development_set_or_a_file_with_a_lock_beside_it_must_be_lock_checked(repo):
    assert lock_required("data/eval/dev.jsonl", [repo])  # by name, even with its lock missing
    assert lock_required("data/eval/e2e-dev.jsonl", [repo])
    other = repo / "other.jsonl"
    write_jsonl(ITEMS, other)
    assert not lock_required(other, [repo])
    write_lock(other)
    assert lock_required(other, [repo])


# test material by content: what a copy of a test set carries when its bytes are not the locked ones

# a drafted item (and the test request it was drafted from), a handwritten item and an end-to-end task
TEST_MATERIAL = [
    {"id": "req-test-0007-s2", "messages": [{"role": "user", "content": "找 MoE 路由的论文"}],
     "split": "test", "source": {"request": "req-test-0007", "step": 2}},
    {"id": "hw-test-003", "messages": [{"role": "user", "content": "hi"}], "split": "test",
     "source": {"handwritten": True}},
    {"id": "e2e-test-05", "intent": "lookup_paper", "split": "test", "text": "What is 2609.01234 about?"},
]


def test_test_material_is_an_item_or_task_of_split_test_or_with_an_id_of_the_test_split():
    ids = ["req-test-0007-s2", "hw-test-003", "e2e-test-05"]
    assert find_test_material(TEST_MATERIAL) == ids
    assert find_test_material([{**item, "split": "dev"} for item in TEST_MATERIAL]) == ids  # the ids tell
    relabelled = {"id": "x-1", "split": "dev", "source": {"request": "req-test-0007", "step": 2}}
    assert find_test_material([relabelled, {"split": "test"}, "not an item"]) == ["x-1", "#2"]
    development = [{"id": "req-dev-0007-s2", "split": "dev",
                    "source": {"request": "req-dev-0007", "step": 2}},
                   {"id": "hw-dev-003", "split": "dev", "source": {"handwritten": True}},
                   {"id": "e2e-dev-05", "split": "dev"}, {"id": "req-train-0513", "split": "train"},
                   {"id": "latest-test-run"}, *ITEMS]
    assert find_test_material(development) == []


def test_a_subset_or_a_re_encoded_copy_of_a_test_set_is_refused_by_the_items_it_holds(repo):
    """refuse_test_sets knows a test set by its location and its exact bytes only: a subset, a copy with CRLF
    line ends and one without the final newline pass it. refuse_test_material refuses them by their items,
    reading the copy, never the test set."""
    test_set = repo / "data" / "eval" / "test.jsonl"
    write_jsonl(TEST_MATERIAL, test_set)
    write_lock(test_set)
    raw = test_set.read_bytes()
    copies = {"subset.jsonl": raw.split(b"\n", 1)[1], "crlf.jsonl": raw.replace(b"\n", b"\r\n"),
              "unterminated.jsonl": raw.rstrip(b"\n")}
    for name, data in copies.items():
        copy = repo / name
        copy.write_bytes(data)
        refuse_test_sets(copy, roots=[repo])  # neither the location nor the bytes of a test set
        with pytest.raises(RefusedTestSet, match=f"{name} holds test material .* the phase-4 test runner"):
            refuse_test_material(read_jsonl(copy), copy)
    refuse_test_material(ITEMS, "items.jsonl")  # no test material: returns without a word


def test_only_the_items_lockedset_open_returns_pass_the_content_guard(tmp_path):
    path = tmp_path / "test.jsonl"
    write_jsonl(TEST_MATERIAL, path)
    write_lock(path)
    test_set = LockedSet(path, tmp_path / "access.jsonl", now=clock(*STAMPS))
    opened = test_set.open("M3", "FT vs B0/B1/B4")
    assert isinstance(opened, OpenedTestSet) and opened == TEST_MATERIAL
    assert opened.entry == test_set.last_open == test_set.log_entries()[0]
    refuse_test_material(opened, path)  # the phase-4 runner's items: let through
    for derived in (opened[:2], list(opened), [dict(item) for item in opened], read_jsonl(path)):
        with pytest.raises(RefusedTestSet, match="only the phase-4 test runner"):
            refuse_test_material(derived, path)
    assert OpenedTestSet(opened[:2], opened.entry).entry == opened.entry  # a list derived on purpose


# the CLI


def test_lock_set_cli_writes_checks_and_keeps_a_stale_lock_unless_forced(tmp_path):
    path = tmp_path / "test.jsonl"
    write_jsonl(ITEMS, path)

    def cli(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(SCRIPT), str(path), *args], capture_output=True,
                              text=True, encoding="utf-8", check=False)

    first = sha256_of(path)
    done = cli()
    assert (done.returncode, done.stdout.strip()) == (0, f"{first}  test.jsonl")
    assert check_lock(path) == first
    assert cli().returncode == 0  # the same lock again is fine
    assert cli("--check").returncode == 0
    edit(path)
    checked = cli("--check")
    assert checked.returncode == 1 and "does not match" in checked.stderr
    refused = cli()
    assert refused.returncode == 1 and "--force" in refused.stderr
    assert lock_path(path).read_text(encoding="utf-8").startswith(first)  # the stale lock is still there
    forced = cli("--force")
    assert forced.returncode == 0 and forced.stdout.strip() == f"{sha256_of(path)}  test.jsonl"
    assert check_lock(path) == sha256_of(path) != first
    helped = cli("--help")
    assert helped.returncode == 0 and "data/eval/test.jsonl" in helped.stdout  # where the locked sets live


def test_a_set_and_lock_whose_file_name_holds_a_line_separator_round_trip(tmp_path):
    path = tmp_path / "test set.jsonl"
    items = [{"id": "t001", "text": "line one line two"}]
    write_jsonl(items, path)
    digest = write_lock(path)
    assert read_lock(path) == digest == check_lock(path)
    assert read_jsonl(path) == items


# ---- M7: the second family of test sets, v2 (docs/m7-计划.md WP4) ---------------------------------------


def test_the_v2_family_has_its_own_sets_access_logs_and_completion_markers():
    v2 = lock_module.EVAL_DIR / "v2"
    assert (lock_module.TEST_SET_V2, lock_module.E2E_TEST_SET_V2) == (v2 / "test-v2.jsonl",
                                                                       v2 / "e2e-test-v2.jsonl")
    assert (lock_module.TEST_ACCESS_LOG_V2, lock_module.E2E_TEST_ACCESS_LOG_V2) == (
        v2 / "test-v2-access.jsonl", v2 / "e2e-test-v2-access.jsonl")
    assert lock_module.TEST_COMPLETIONS_V2 == v2 / "test-completions-v2.jsonl"
    assert TEST_SETS[2:] == (lock_module.TEST_SET_V2, lock_module.E2E_TEST_SET_V2)
    assert lock_module.HANDWRITTEN_SETS[2:] == (v2 / "handwritten-v2.jsonl",
                                                v2 / "handwritten-specs-v2.jsonl")
    families = lock_module.TEST_FAMILIES
    assert set(families) == {"v1", "v2"}
    assert families["v1"].sets == ((lock_module.TEST_SET, lock_module.TEST_ACCESS_LOG),
                                   (lock_module.E2E_TEST_SET, lock_module.E2E_TEST_ACCESS_LOG))
    assert families["v1"].completions == lock_module.TEST_COMPLETIONS
    assert families["v2"].sets == ((lock_module.TEST_SET_V2, lock_module.TEST_ACCESS_LOG_V2),
                                   (lock_module.E2E_TEST_SET_V2, lock_module.E2E_TEST_ACCESS_LOG_V2))
    assert families["v2"].completions == lock_module.TEST_COMPLETIONS_V2
    for name in ("data/eval/v2/test-v2.jsonl", "data/eval/v2/e2e-test-v2.jsonl", "test-v2-access.jsonl",
                 "test-completions-v2.jsonl", "handwritten-v2.jsonl"):
        assert name in lock_module.__doc__, name


def test_a_v2_set_is_refused_like_v1_and_opens_three_times_through_its_own_log(repo):
    v2 = repo / "data" / "eval" / "v2"
    test_set = v2 / "test-v2.jsonl"
    write_jsonl(ITEMS, test_set)
    write_lock(test_set)
    with pytest.raises(RefusedTestSet, match="locked test set"):
        refuse_test_sets("data/eval/v2/test-v2.jsonl", roots=[repo])
    (repo / "copy.jsonl").write_bytes(test_set.read_bytes())
    with pytest.raises(RefusedTestSet, match="SHA-256"):
        refuse_test_sets("copy.jsonl", roots=[repo])
    write_jsonl(ITEMS, v2 / "handwritten-v2.jsonl")
    with pytest.raises(RefusedTestSet, match="handwritten items with their gold turns"):
        refuse_test_sets("data/eval/v2/handwritten-v2.jsonl", roots=[repo])
    # the v1 log is spent; the v2 set has its own log and its own three opens
    v1_log = repo / "data" / "eval" / "test-access.jsonl"
    entries = [{"milestone": m, "purpose": "p", "sha256": "0" * 64, "ts": "2026-10-17T00:00:00Z"}
               for m in ("M3", "M5", "M6")]
    v1_log.write_text("".join(canonical_json(e) + "\n" for e in entries), encoding="utf-8")
    locked = LockedSet(test_set, v2 / "test-v2-access.jsonl")
    assert locked.opens_left() == 3
    opened = locked.open("M7", "the first v2 open")
    assert len(opened) == len(ITEMS) and locked.opens_left() == 2
    assert len(v1_log.read_text(encoding="utf-8").splitlines()) == 3


# ---- M7: the retired read of v1's end-to-end test set (docs/m7-计划.md WP-I 6) -------------------------

REPO = Path(__file__).resolve().parents[2]
RETIRED_TASKS = [
    {"id": "e2e-test-01", "intent": "find_recent_papers", "scenario": "normal", "lang": "en", "split": "test",
     "family": "e2e/find_recent_papers",
     "slots": {"count": 3, "date_from": "2026-08-01", "topic": "tokenizer", "topic_id": "tokenization"},
     "text": "Find three tokenizer papers posted since 2026-08-01 and rank them by relevance for me."},
    {"id": "e2e-test-02", "intent": "read_section", "scenario": "normal", "lang": "zh", "split": "test",
     "family": "e2e/read_section", "slots": {"arxiv_id": "2609.05022", "section": "experiments"},
     "text": "帮我看看 2609.05022 的实验部分,作者到底用了哪些数据集?"},
    {"id": "e2e-test-03", "intent": "chit_chat", "scenario": "no_call", "lang": "en", "split": "test",
     "family": "e2e/chit_chat", "slots": {}, "text": "That is all for today, thanks a lot and see you soon!"},
]
SPENT = [{"milestone": m, "purpose": "p", "sha256": "0" * 64, "ts": "2026-10-17T00:00:00+00:00"}
         for m in ("M3", "M5", "M6")]
READ_STAMPS = ["2026-10-21T08:00:00+00:00", "2026-10-21T09:00:00+00:00", "2026-10-21T10:00:00+00:00"]


@pytest.fixture
def retired(tmp_path):
    """A retired end-to-end set at tmp_path/e2e-test.jsonl: locked, its access log holding three opens."""
    path = tmp_path / "e2e-test.jsonl"
    write_jsonl(RETIRED_TASKS, path)
    write_lock(path)
    write_jsonl(SPENT, tmp_path / "e2e-test-access.jsonl")
    return path


def retired_set(path: Path, **options) -> "lock_module.RetiredSet":
    return lock_module.RetiredSet(path, path.with_name("e2e-test-access.jsonl"),
                                  path.with_name("e2e-test-retired-reads.jsonl"),
                                  now=options.pop("now", None) or clock(*READ_STAMPS), **options)


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


def test_a_retired_read_returns_one_float_per_text_and_nothing_else(retired):
    texts = ["Find three tokenizer papers posted since 2026-08-01 and rank them by relevance for me.",
             "完全无关的一句话,只是问问天气怎么样。", ""]
    result = retired_set(retired).similarities(texts)
    assert type(result) is list and len(result) == len(texts)
    assert all(type(value) is float and 0.0 <= value <= 1.0 for value in result)
    assert result[0] == 1.0  # the same text as a retired task


def test_a_near_duplicate_of_a_retired_task_reaches_the_train_vs_eval_threshold(retired):
    near = ["Find four tokenizer papers posted since 2026-09-15 and rank them by relevance for me!",
            "帮我看看 2608.07713 的实验部分,作者到底用了哪些数据集呢?"]
    far = ["Who wrote 2609.33395, and when was it published?"]
    scores = retired_set(retired).similarities([*near, *far])
    assert all(score >= TRAIN_VS_EVAL for score in scores[:2]), scores
    assert scores[2] < 0.3, scores


def test_a_retired_score_is_the_contract_tests_masked_five_gram_jaccard_at_its_largest(retired):
    """Exactly data.requests' measure (jaccard over the 5-gram shingles of mask_text), the largest over the
    retired texts: another shingle size, the texts unmasked or some of the retired texts left out would
    give other values."""
    texts = ["Find four tokenizer papers posted since 2026-09-15 and rank them for me.",
             "帮我看看 2608.07713 的方法部分,作者用了什么数据?",
             "That is all for this week, thanks, and see you on Monday!"]
    retired_grams = [shingles(mask_text(task["text"])) for task in RETIRED_TASKS]
    expected = [max(jaccard(shingles(mask_text(text)), grams) for grams in retired_grams) for text in texts]
    assert retired_set(retired).similarities(texts) == expected
    assert all(0.1 < value < 1.0 for value in expected), expected  # each one decided by a real overlap


def test_every_retired_read_logs_one_line_with_no_text_and_no_task_field(retired):
    reads = retired.with_name("e2e-test-retired-reads.jsonl")
    retired_set_ = retired_set(retired)
    asked = ["Find tokenizer papers since 2026-08-01, three of them.", "See you on Monday!"]
    retired_set_.similarities(asked)
    retired_set_.similarities(asked[:1])
    entries = read_jsonl(reads)
    digest = sha256_of(retired)
    assert entries == [{"lock_ok": True, "sha256": digest, "texts": 2, "ts": READ_STAMPS[0]},
                       {"lock_ok": True, "sha256": digest, "texts": 1, "ts": READ_STAMPS[1]}]
    raw = reads.read_text(encoding="utf-8")
    assert raw == "".join(canonical_json(entry) + "\n" for entry in entries)
    fields = {str(value) for task in RETIRED_TASKS for key, value in task.items() if key != "slots"}
    fields |= {str(value) for task in RETIRED_TASKS for value in task["slots"].values()}
    # every value of five or more characters (a digit or two may occur in a digest or a time by chance)
    assert not [field for field in fields | set(asked) if len(field) >= 5 and field in raw]


@pytest.mark.needs_data("data/eval")  # the real access log, which the public copy leaves out
def test_a_retired_read_is_no_open_of_the_set(retired):
    access = retired.with_name("e2e-test-access.jsonl")
    real_access = REPO / lock_module.E2E_TEST_ACCESS_LOG
    before, real_before = access.read_bytes(), line_count(real_access)
    for _ in range(4):  # more reads than a test set has opens
        retired_set(retired, now=clock(*READ_STAMPS, *READ_STAMPS)).similarities(["any text at all"])
    assert access.read_bytes() == before and line_count(access) == len(SPENT)
    assert line_count(real_access) == real_before  # the real access log keeps its line count too


def test_a_set_that_is_not_retired_or_not_locked_is_refused_and_nothing_is_logged(retired):
    reads = retired.with_name("e2e-test-retired-reads.jsonl")
    access = retired.with_name("e2e-test-access.jsonl")
    write_jsonl(SPENT[:2], access)
    with pytest.raises(LockError, match="not retired"):
        retired_set(retired).similarities(["a text"])
    write_jsonl(SPENT, access)
    original = retired.read_bytes()
    edit(retired)
    with pytest.raises(LockError, match="does not match its lock"):
        retired_set(retired).similarities(["a text"])
    retired.write_bytes(original)
    lock_path(retired).unlink()
    with pytest.raises(LockError, match="has no lock"):
        retired_set(retired).similarities(["a text"])
    assert not reads.exists()


def test_the_texts_must_be_strings_and_a_refused_call_reads_nothing(retired):
    reads = retired.with_name("e2e-test-retired-reads.jsonl")
    for bad in ("one bare string", ["fine", 3], [None]):
        with pytest.raises(TypeError, match="texts"):
            retired_set(retired).similarities(bad)
    assert not reads.exists()
    assert retired_set(retired).similarities([]) == []
    assert len(read_jsonl(reads)) == 1  # an empty read is logged too


def test_the_retired_reads_log_lives_beside_the_end_to_end_access_log():
    eval_dir = lock_module.EVAL_DIR
    assert lock_module.E2E_TEST_RETIRED_READS == eval_dir / "e2e-test-retired-reads.jsonl"
    default = lock_module.RetiredSet()
    assert (default.path, default.access_log, default.reads_log) == (
        lock_module.E2E_TEST_SET, lock_module.E2E_TEST_ACCESS_LOG, lock_module.E2E_TEST_RETIRED_READS)
    assert lock_module.E2E_TEST_SET in TEST_SETS  # the guards still refuse every other read of it
    for name in ("data/eval/e2e-test-retired-reads.jsonl", "RetiredSet"):
        assert name in lock_module.__doc__, name

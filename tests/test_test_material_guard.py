"""The test-material guard of tests/conftest.py: a test run cannot open the locked test sets or the
handwritten files that hold test items, by any spelling or link, unless the test is marked test_material and
the run has --test-material; a refusal the code swallows is still on the guard's record."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import conftest
import pytest

REPO = Path(__file__).resolve().parents[1]
MATERIAL = ("test.jsonl", "e2e-test.jsonl", "handwritten.jsonl", "handwritten-specs.jsonl")


@pytest.fixture
def forgive():
    """Take the refusals this test provokes on purpose off the guard's record, so the guard's check after the
    test (and at the end of the session) passes."""
    before = len(conftest.REFUSED)
    yield
    del conftest.REFUSED[before:]


@pytest.mark.parametrize("name", MATERIAL)
def test_a_test_cannot_open_a_locked_test_set_or_a_handwritten_file(name, forgive):
    path = REPO / "data" / "eval" / name
    with pytest.raises(conftest.MaterialOpenRefused, match=name):
        open(path, "rb").close()
    with pytest.raises(conftest.MaterialOpenRefused):
        path.read_bytes()


def test_the_guard_sees_every_spelling_of_the_path_and_os_open(monkeypatch, forgive):
    monkeypatch.chdir(REPO)
    spellings = ["data/eval/test.jsonl", "./data/eval/../eval/e2e-test.jsonl",
                 Path("data/eval/handwritten.jsonl"),
                 os.fsencode(REPO / "data" / "eval" / "handwritten-specs.jsonl")]
    if os.name == "nt":
        spellings.append("DATA\\EVAL\\TEST.JSONL")
    for spelling in spellings:
        with pytest.raises(conftest.MaterialOpenRefused):
            open(spelling, "rb").close()
    with pytest.raises(conftest.MaterialOpenRefused):
        os.close(os.open("data/eval/e2e-test.jsonl", os.O_RDONLY))


def test_a_hard_link_to_test_material_is_refused(tmp_path, forgive):
    link = tmp_path / "copy.jsonl"
    try:
        os.link(REPO / "data" / "eval" / "e2e-test.jsonl", link)  # a second name of the same file
    except OSError as exc:  # another volume, or a file system without hard links
        pytest.skip(f"no hard link here: {exc}")
    try:
        with pytest.raises(conftest.MaterialOpenRefused):
            open(link, "rb").close()
    finally:
        link.unlink()


@pytest.mark.needs_data("data/eval")  # the link's target, a test set: left out of the public copy
def test_a_symbolic_link_without_the_jsonl_suffix_is_refused(tmp_path, forgive):
    link = tmp_path / "plain"
    try:
        link.symlink_to(REPO / "data" / "eval" / "e2e-test.jsonl")
    except OSError as exc:  # Windows without the right to create symbolic links
        pytest.skip(f"no symbolic link here: {exc}")
    with pytest.raises(conftest.MaterialOpenRefused):
        open(link, "rb").close()


def test_a_refusal_the_code_swallows_is_still_on_the_record(forgive):
    before = len(conftest.REFUSED)
    with contextlib.suppress(conftest.MaterialOpenRefused):  # what careless code under test might do
        open(REPO / "data" / "eval" / "test.jsonl", "rb").close()
    assert conftest.REFUSED[before:] == [str(REPO / "data" / "eval" / "test.jsonl")]


@pytest.mark.needs_data("data/eval/dev.jsonl", "data/eval/e2e-dev.jsonl")  # left out of the public copy
def test_the_dev_sets_and_a_same_named_file_elsewhere_open(tmp_path, monkeypatch):
    for name in ("dev.jsonl", "e2e-dev.jsonl"):
        assert (REPO / "data" / "eval" / name).read_bytes()
    elsewhere = tmp_path / "data" / "eval" / "test.jsonl"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text("{}\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert Path("data/eval/test.jsonl").read_text(encoding="utf-8") == "{}\n"


@pytest.mark.test_material
def test_a_marked_test_runs_only_with_the_option_and_may_then_open_test_material(request):
    assert request.config.getoption("--test-material")
    open(REPO / "data" / "eval" / "e2e-test.jsonl", "rb").close()

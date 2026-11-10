"""The needs_data marker of tests/conftest.py: a test that reads data the public copy leaves out names the
paths, runs where they all exist, is skipped with the missing ones as the reason where one does not, and runs
anyway with --run-needs-data; a marker without a path or naming test material is a usage error."""

from __future__ import annotations

import conftest
import pytest

PRESENT = "pyproject.toml"
ABSENT = "data/no-such-folder/never-there.jsonl"


class Config:
    def __init__(self, **options: bool) -> None:
        self.options = {"--test-material": False, "--run-needs-data": False, **options}

    def getoption(self, name: str) -> bool:
        return self.options[name]


class Item:
    """What pytest_collection_modifyitems uses of a test item: its node id, its markers and add_marker."""

    def __init__(self, *marks: pytest.MarkDecorator) -> None:
        self.nodeid = "tests/test_x.py::test_y"
        self.marks = [mark.mark for mark in marks]
        self.added: list = []

    def iter_markers(self, name: str):
        return (mark for mark in self.marks if mark.name == name)

    def get_closest_marker(self, name: str):
        return next(self.iter_markers(name), None)

    def add_marker(self, marker: pytest.MarkDecorator) -> None:
        self.added.append(marker.mark)


def skip_reasons(item: Item) -> list[str]:
    return [mark.kwargs["reason"] for mark in item.added if mark.name == "skip"]


def test_missing_data_names_the_paths_that_do_not_exist():
    assert conftest.missing_data((PRESENT, ABSENT)) == [ABSENT]
    assert conftest.missing_data((PRESENT,)) == []


def test_a_marked_test_runs_where_its_data_is_and_is_skipped_with_the_missing_paths_where_it_is_not():
    here, gone = Item(pytest.mark.needs_data(PRESENT)), Item(pytest.mark.needs_data(PRESENT, ABSENT))
    unmarked = Item()
    conftest.pytest_collection_modifyitems(Config(), [here, gone, unmarked])
    assert skip_reasons(here) == [] and skip_reasons(unmarked) == []
    assert skip_reasons(gone) == [f"needs data the public copy leaves out: {ABSENT}"]


def test_the_paths_of_every_needs_data_marker_count():
    item = Item(pytest.mark.needs_data(PRESENT), pytest.mark.needs_data(ABSENT, PRESENT))
    assert conftest.needed_data(item) == (PRESENT, ABSENT)
    conftest.pytest_collection_modifyitems(Config(), [item])
    assert skip_reasons(item) == [f"needs data the public copy leaves out: {ABSENT}"]


def test_with_run_needs_data_a_marked_test_runs_even_without_its_data():
    item = Item(pytest.mark.needs_data(ABSENT))
    conftest.pytest_collection_modifyitems(Config(**{"--run-needs-data": True}), [item])
    assert skip_reasons(item) == []


def test_a_marker_without_a_path_or_naming_test_material_is_a_usage_error():
    with pytest.raises(pytest.UsageError, match="names no path"):
        conftest.needed_data(Item(pytest.mark.needs_data()))
    with pytest.raises(pytest.UsageError, match="names test material"):  # the path is never opened
        conftest.needed_data(Item(pytest.mark.needs_data(PRESENT, conftest.TEST_SETS[0].as_posix())))


def test_a_test_marked_test_material_too_is_skipped_for_its_test_material_alone():
    for paths in ((PRESENT,), (ABSENT,)):
        item = Item(pytest.mark.test_material, pytest.mark.needs_data(*paths))
        conftest.pytest_collection_modifyitems(Config(), [item])
        assert skip_reasons(item) == ["reads test material: run with --test-material"]
    item = Item(pytest.mark.test_material, pytest.mark.needs_data(ABSENT))  # with --test-material: the data
    conftest.pytest_collection_modifyitems(Config(**{"--test-material": True}), [item])
    assert skip_reasons(item) == [f"needs data the public copy leaves out: {ABSENT}"]

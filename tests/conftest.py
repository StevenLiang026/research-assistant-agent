"""Shared test setup.

Git environment. Tests that run git in temporary folders must not inherit what git exports to the hooks of the
repository that runs them. GIT_DIR and GIT_INDEX_FILE (in a linked worktree the pre-commit pytest run sees
GIT_DIR) make git answer for that repository instead of the folder the test made; the committer's identity
and -c options (GIT_AUTHOR_NAME, GIT_AUTHOR_EMAIL, GIT_AUTHOR_DATE, GIT_CONFIG_PARAMETERS) override a test's
own -c user.name and change the commits it makes, and so the hashes a test pins. Every one of GIT_ENVIRONMENT
is removed from the environment when pytest imports this file, before any fixture of any scope (a module
fixture that makes commits runs before every function fixture); tests/test_git_environment.py checks it
covers what git exports to a pre-commit hook.

Test material. A test run cannot open the locked test sets (data/eval/test.jsonl, e2e-test.jsonl) or the
handwritten files that hold test items (data/eval/handwritten.jsonl, handwritten-specs.jsonl): an audit hook
refuses every open of them by path, before the file is opened, whatever the spelling and through a symbolic or
a hard link too (it compares the identity of the file the path names, st_dev and st_ino, and the path itself,
which alone decides for a file that does not exist), and keeps a record of it, so a refusal the code under
test swallows still fails the test, and one outside any test (collection, a module fixture) fails the session.
The one exception is a test marked test_material in a run with --test-material; without the option such tests
are skipped, so neither the pre-commit run nor the cloud check reads test material. The hook does not
recognize an open relative to a directory descriptor (os.open(name, flags, dir_fd=fd)): the audit event
carries the name without the directory, and the hook resolves it against the current folder. The marked tests
are the contract checks of the end-to-end test file (tests/eval/test_e2e.py); the file is locked by the hash
of its bytes, which the phase-4 runner checks before it opens the set (scripts/run_test.py).

Data the public copy leaves out. This repository is an export of a private one without the evaluation sets,
the cassettes, the glossary data, the synthetic data, the per-item results and the progress records. A test
that reads such a file is marked needs_data with the paths it reads, relative to the repository root, and
for a file that is test material the folder that holds it, never the file itself:
@pytest.mark.needs_data("data/eval/dev.jsonl", "data/synth", ...). It runs whenever every named path exists
(privately), and is skipped with the missing paths as the reason when one does not. With --run-needs-data the
marked tests run even then: the export runs them that way in the public copy to check that each one fails
without its data, so no test is marked that does not need it.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from research_agent.eval.lock import HANDWRITTEN_SETS, TEST_SETS

GIT_LOCATION = ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR", "GIT_OBJECT_DIRECTORY",
                "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_PREFIX")
GIT_IDENTITY = ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_AUTHOR_DATE", "GIT_COMMITTER_NAME",
                "GIT_COMMITTER_EMAIL", "GIT_COMMITTER_DATE", "GIT_CONFIG_PARAMETERS", "GIT_CONFIG_COUNT")
GIT_ENVIRONMENT = GIT_LOCATION + GIT_IDENTITY
for _name in GIT_ENVIRONMENT:  # module docstring: Git environment
    os.environ.pop(_name, None)
REPO = Path(__file__).resolve().parents[1]
MATERIAL = frozenset(os.path.normcase(spelling) for path in (*TEST_SETS, *HANDWRITTEN_SETS)
                     for spelling in (os.path.abspath(REPO / path), os.path.realpath(REPO / path)))


def _identities() -> frozenset[tuple[int, int]]:
    """The (st_dev, st_ino) of every test-material file that exists, which its symbolic and hard links share;
    none from a file system without inode numbers (st_ino 0), where MATERIAL's paths are the only check."""
    found = set()
    for path in (*TEST_SETS, *HANDWRITTEN_SETS):
        try:
            info = os.stat(REPO / path)
        except OSError:
            continue
        if info.st_ino:
            found.add((info.st_dev, info.st_ino))
    return frozenset(found)


MATERIAL_IDS = _identities()
REFUSED: list[str] = []  # every refused open of the session, in order, as the code spelled the path
_guard = {"allowed": False, "installed": False}


class MaterialOpenRefused(RuntimeError):
    """An open of test material by a test not marked test_material, or in a run without --test-material."""


def _refuse_test_material(event: str, args: tuple) -> None:
    """The audit hook: refuse an open of test material (module docstring) unless the running test may open
    it."""
    if event != "open" or _guard["allowed"] or not args or args[0] is None or isinstance(args[0], int):
        return
    try:
        name = os.fsdecode(os.fspath(args[0]))
    except TypeError:
        return
    try:
        info = os.stat(name)
    except (OSError, ValueError):  # no such file (yet), or a name the system refuses
        material = os.path.normcase(os.path.abspath(name)) in MATERIAL
    else:
        material = ((info.st_dev, info.st_ino) in MATERIAL_IDS
                    or os.path.normcase(os.path.abspath(name)) in MATERIAL)
    if material:
        REFUSED.append(name)
        raise MaterialOpenRefused(f"{name} is test material: only a test marked test_material, in a run with "
                                  f"--test-material, may open it")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--test-material", action="store_true", default=False,
                     help="also run the tests marked test_material, which read a locked test set or a "
                          "handwritten file that holds test items")
    parser.addoption("--run-needs-data", action="store_true", default=False,
                     help="run the tests marked needs_data even when a path they name is missing (the "
                          "export's check that each one fails without its data)")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "test_material: reads test material; skipped unless the run has "
                                       "--test-material")
    config.addinivalue_line("markers", "needs_data(*paths): reads repository data that the public copy "
                                       "leaves out; skipped when one of the paths (from the repository "
                                       "root) is missing")
    if not _guard["installed"]:  # an audit hook cannot be removed: one per process
        sys.addaudithook(_refuse_test_material)
        _guard["installed"] = True


def missing_data(paths: tuple[str, ...]) -> list[str]:
    """The paths among paths (from the repository root) that do not exist."""
    return [path for path in paths if not (REPO / path).exists()]


def needed_data(item: pytest.Item) -> tuple[str, ...]:
    """The paths every needs_data marker of item names (a test may get one from its class or module too), in
    order; a marker without a path, or a path that is test material, is a usage error."""
    paths: list[str] = []
    for marker in item.iter_markers("needs_data"):
        if not marker.args:
            raise pytest.UsageError(f"{item.nodeid}: needs_data names no path")
        for path in marker.args:
            if os.path.normcase(os.path.abspath(REPO / path)) in MATERIAL:
                raise pytest.UsageError(f"{item.nodeid}: needs_data names test material; name its folder")
            if path not in paths:
                paths.append(path)
    return tuple(paths)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    material = not config.getoption("--test-material")  # skip the tests marked test_material
    skip = pytest.mark.skip(reason="reads test material: run with --test-material")
    for item in items:
        paths = needed_data(item)  # module docstring: Data the public copy leaves out
        if material and item.get_closest_marker("test_material"):
            item.add_marker(skip)  # the test-material skip comes first: its reason is the one reported
            continue
        missing = missing_data(paths) if paths and not config.getoption("--run-needs-data") else []
        if missing:
            reason = f"needs data the public copy leaves out: {', '.join(missing)}"
            item.add_marker(pytest.mark.skip(reason=reason))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    if REFUSED and exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: pytest.TerminalReporter) -> None:
    if REFUSED:
        terminalreporter.section("test material", red=True)
        for name in REFUSED:
            terminalreporter.write_line(f"refused an open of {name}: the run fails")


@pytest.fixture(autouse=True)
def no_run_copy_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test sees a run copy named in the environment (RAA_RUN_COPY, which scripts/run_special_dev.py and
    scripts/vote_offline.py read when --run-copy is not given): the held-out roots a test sees are its own."""
    monkeypatch.delenv("RAA_RUN_COPY", raising=False)


@pytest.fixture(autouse=True)
def guard_test_material(request: pytest.FixtureRequest) -> Iterator[None]:
    """Let a marked test open test material in a --test-material run, and fail any test that was refused."""
    before = len(REFUSED)
    _guard["allowed"] = bool(request.node.get_closest_marker("test_material")
                             and request.config.getoption("--test-material"))
    try:
        yield
    finally:
        _guard["allowed"] = False
    if len(REFUSED) > before:
        pytest.fail(f"opened test material: {REFUSED[before:]}", pytrace=False)

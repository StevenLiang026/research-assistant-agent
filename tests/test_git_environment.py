"""The shared setup's git isolation (tests/conftest.py, Git environment): every variable git exports to a
pre-commit hook that would change what a test's own git does is removed before each test, so a test run from
the pre-commit hook makes the same commits, with the same hashes, as a run from the shell."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import conftest
import pytest

# what git exports to its hooks that only locates programs, and changes no repository, identity or setting
PROGRAMS = {"GIT_EXEC_PATH", "GIT_EDITOR", "GIT_ASKPASS"}
HOOK = '#!/bin/sh\nenv | sed -n "s/^\\(GIT_[A-Za-z0-9_]*\\)=.*/\\1/p" | sort > hook-env.txt\n'


def test_the_isolation_covers_every_variable_git_exports_to_a_pre_commit_hook(tmp_path: Path):
    repo = tmp_path / "repo"
    env = dict(os.environ)  # this test's own environment, already without the isolated variables

    def git(*args: str) -> None:
        done = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=env,
                              check=False)
        assert done.returncode == 0, done.stderr

    repo.mkdir()
    git("init", "-q")
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_bytes(HOOK.encode("utf-8"))
    hook.chmod(0o755)
    (repo / "a.txt").write_bytes(b"a\n")
    git("add", "a.txt")
    # an identity and a setting given as -c, as a developer's commit may carry them
    git("-c", "user.name=someone", "-c", "user.email=someone@example.com", "-c", "core.autocrlf=false",
        "commit", "-q", "-m", "a")
    in_hook = set((repo / "hook-env.txt").read_text(encoding="utf-8").split())
    # what git itself exports, beyond the environment it ran in (a machine's own GIT_ variables, such as a
    # cloud runner's GIT_SSL_CAINFO, reach the hook too, but no test's git differs for them)
    exported = in_hook - {name for name in env if name.startswith("GIT_")}
    assert {"GIT_INDEX_FILE", "GIT_AUTHOR_NAME", "GIT_CONFIG_PARAMETERS"} <= exported  # the hook ran
    assert exported - PROGRAMS <= set(conftest.GIT_ENVIRONMENT)


@pytest.fixture(scope="module")
def module_environment() -> set[str]:
    """The isolated variables a module fixture sees: one that makes commits runs before every function
    fixture, so the isolation must be in place before it."""
    return {name for name in conftest.GIT_ENVIRONMENT if name in os.environ}


def test_no_isolated_variable_reaches_a_module_fixture_or_a_test(module_environment):
    assert module_environment == set()
    assert not [name for name in conftest.GIT_ENVIRONMENT if name in os.environ]

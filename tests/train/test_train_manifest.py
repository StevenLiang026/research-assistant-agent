import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime

import pytest

from research_agent.train.manifest import (
    ABSENT,
    REPO,
    TRACKED_PACKAGES,
    UNKNOWN,
    count_rows,
    frozen_packages,
    git_commit,
    git_dirty,
    package_version,
    run_manifest,
    sha256_of,
    write_manifest,
)

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)


def data_files(tmp_path):
    jsonl = tmp_path / "a.jsonl"
    jsonl.write_text('{"messages": []}\n\n{"messages": [1]}\n{"messages": [2]}\n', encoding="utf-8")
    single = tmp_path / "b.json"
    single.write_text('{"messages": []}', encoding="utf-8")
    listed = tmp_path / "c.json"
    listed.write_text("[1, 2, 3, 4]", encoding="utf-8")
    return [jsonl, single, listed]


def test_manifest_records_config_data_hashes_rows_versions_gpu_and_the_clock(tmp_path):
    files = data_files(tmp_path)
    manifest = run_manifest({"lr": 1e-4}, files, now=lambda: NOW, git_dir=None, gpu_name=lambda: "fake GPU")
    assert manifest["timestamp"] == "2026-10-09T12:00:00+00:00"
    assert manifest["config"] == {"lr": 1e-4}
    assert [d["rows"] for d in manifest["data_files"]] == [3, 1, 4]
    for entry, path in zip(manifest["data_files"], files, strict=True):
        assert entry["path"] == str(path) and entry["bytes"] == path.stat().st_size
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest() == sha256_of(path)
    assert manifest["git_commit"] == UNKNOWN and manifest["git_dirty"] is None
    assert manifest["versions"]["python"] == platform.python_version()
    assert set(manifest["versions"]) == {"python", *TRACKED_PACKAGES}
    assert manifest["gpu"] == "fake GPU"
    assert manifest["packages"] == frozen_packages()


def test_count_rows_follows_the_file_type(tmp_path):
    jsonl, single, listed = data_files(tmp_path)
    assert (count_rows(jsonl), count_rows(single), count_rows(listed)) == (3, 1, 4)


def test_versions_are_read_from_package_metadata_and_absent_when_not_installed():
    assert package_version("pytest") == importlib.metadata.version("pytest")
    assert package_version("no-such-package-for-this-test") == ABSENT
    packages = frozen_packages()
    assert packages == sorted(packages, key=str.lower) and all("==" in p for p in packages)
    assert f"pytest=={importlib.metadata.version('pytest')}" in packages


def test_git_commit_is_unknown_without_git_dir_or_outside_a_repository(tmp_path):
    assert git_commit(None) == UNKNOWN and git_dirty(None) is None
    assert git_commit(tmp_path) == UNKNOWN and git_dirty(tmp_path) is None


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_commit_and_dirty_flag_of_a_temporary_repository(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.txt").write_text("a\n", encoding="utf-8")
    identity = ["-c", "user.name=test", "-c", "user.email=test@example.com"]
    for cmd in (["init", "-q"], ["add", "a.txt"], [*identity, "commit", "-q", "-m", "init"]):
        done = subprocess.run(["git", "-C", str(repo), *cmd], capture_output=True, text=True, check=False)
        if done.returncode != 0:
            pytest.skip(f"git cannot commit in a temporary repository here: {done.stderr.strip()}")
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()
    assert git_commit(repo) == head and len(head) == 40 and git_dirty(repo) is False
    (repo / "b.txt").write_text("b\n", encoding="utf-8")
    assert git_dirty(repo) is True
    assert REPO.joinpath("pyproject.toml").is_file()  # the default git_dir is the project root


def test_write_manifest_creates_parents_and_round_trips(tmp_path):
    manifest = run_manifest({"seed": 42, "note": "中文"}, [], now=lambda: NOW, git_dir=None,
                            gpu_name=lambda: None)
    path = write_manifest(manifest, tmp_path / "runs" / "x" / "manifest.json")
    raw = path.read_bytes()
    assert raw.endswith(b"\n") and b"\r\n" not in raw and "中文".encode() in raw
    assert json.loads(raw.decode("utf-8")) == manifest and manifest["gpu"] is None


def test_count_rows_counts_a_row_holding_a_line_separator_as_one_row(tmp_path):
    jsonl = tmp_path / "a.jsonl"
    jsonl.write_text(json.dumps({"text": "line one line two"}, ensure_ascii=False) + "\n\n"
                     + json.dumps({"text": "next"}) + "\n", encoding="utf-8")
    assert count_rows(jsonl) == 2

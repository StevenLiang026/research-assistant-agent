"""What every training run records before it starts: config, data hashes, code version and environment.

The plan asks each run to record the data manifest (SHA-256), the git commit, the full configuration and a
package freeze next to the curves, so that a result can be traced back to exactly what produced it. Nothing
here needs a GPU: torch is imported only to read the device name, and only when it is installed.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from ..canonical import split_lines

REPO = Path(__file__).resolve().parents[3]
TRACKED_PACKAGES = ["torch", "transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes"]
ABSENT = "absent"
UNKNOWN = "unknown"


def _utc_now() -> datetime:
    return datetime.now(UTC)


def sha256_of(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_rows(path: str | Path) -> int:
    """Records in a data file: non-blank lines of JSONL; 1 for a JSON object, the length of a JSON list."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        value = json.loads(text)
        return len(value) if isinstance(value, list) else 1
    return sum(1 for line in split_lines(text) if line.strip())


def _git(git_dir: str | Path | None, *args: str) -> str | None:
    """stdout of a git command in git_dir, or None when git or the repository is unavailable."""
    if git_dir is None:
        return None
    try:
        done = subprocess.run(["git", "-C", str(git_dir), *args], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def git_commit(git_dir: str | Path | None) -> str:
    """The HEAD commit of git_dir, or "unknown"."""
    return _git(git_dir, "rev-parse", "HEAD") or UNKNOWN


def git_dirty(git_dir: str | Path | None) -> bool | None:
    """True when the working tree has uncommitted changes; None when that cannot be told."""
    status = _git(git_dir, "status", "--porcelain")
    return None if status is None else bool(status.strip())


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ABSENT


def frozen_packages() -> list[str]:
    """Every installed distribution as name==version, sorted (the pip freeze the plan asks for)."""
    entries = set()
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"] if dist.metadata else None
        if name:
            entries.add(f"{name}=={dist.version}")
    return sorted(entries, key=str.lower)


def cuda_device_name() -> str | None:
    """The first CUDA device's name, or None when torch is absent or sees no CUDA device."""
    try:
        import torch
    except ImportError:
        return None
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def run_manifest(config: dict, data_files: list[str | Path], *, now: Callable[[], datetime] = _utc_now,
                 git_dir: str | Path | None = REPO,
                 gpu_name: Callable[[], str | None] = cuda_device_name) -> dict:
    """The run record: config, data files with SHA-256 and row counts, git commit, versions, GPU, freeze."""
    return {
        "timestamp": now().isoformat(),
        "config": config,
        "data_files": [{"path": str(p), "sha256": sha256_of(p), "rows": count_rows(p),
                        "bytes": Path(p).stat().st_size} for p in data_files],
        "git_commit": git_commit(git_dir),
        "git_dirty": git_dirty(git_dir),
        "versions": {"python": platform.python_version(),
                     **{name: package_version(name) for name in TRACKED_PACKAGES}},
        "gpu": gpu_name(),
        "packages": frozen_packages(),
    }


def write_manifest(manifest: dict, path: str | Path) -> Path:
    """manifest as UTF-8 JSON with LF line ends and a final newline; parent folders are created."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8",
                    newline="\n")
    return path

"""Pre-commit hook: refuse files that contain API-key-like strings or a user's home directory path.

Usage: python scripts/check_no_secrets.py <file> [<file> ...]  (pre-commit passes the staged files)
       python scripts/check_no_secrets.py --history [<folder>]  (every blob any ref of the repository reaches)
Exit 1 and print file:line for every hit. Exit 2 when an argument is not an existing file (each one is named
on stderr; the other files are still scanned): a misspelt path, or one git printed in escaped form, must not
pass unscanned. pre-commit only passes staged files, which exist. --history scans what a push of every ref
would publish, earlier commits included (scan_history); exit 1 on a hit, 2 when the folder is not a git
repository.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

from research_agent.canonical import split_lines

PATTERNS = {
    # prefixed keys use a separator (sk-..., tvly-..., ghp_...); AWS access key ids are AKIA + 16 upper-case;
    # Hugging Face tokens hf_ + 30 or more letters and digits; GitHub server, user and refresh tokens
    # ghs_/ghu_/ghr_ + 30 or more; Slack tokens xox?- (bot, user, app, refresh, ...); Google API keys
    # AIza + 35
    "api key": re.compile(r"\b(sk|tvly|ghp|gho|github_pat|xoxb|xoxp)[-_][A-Za-z0-9_\-]{16,}"
                          r"|\bAKIA[0-9A-Z]{16}\b"
                          r"|\bhf_[A-Za-z0-9]{30,}"
                          r"|\bgh[sur]_[A-Za-z0-9]{30,}"
                          r"|\bxox[a-z]-[A-Za-z0-9\-]{10,}"
                          r"|\bAIza[0-9A-Za-z_\-]{35}"),
    # the armour line of a PEM or PGP private key (RSA, EC, DSA, OPENSSH, ENCRYPTED, PGP ... BLOCK)
    "private key": re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----"),
    # a URL whose authority holds a user, a colon and a literal password before the @ (no placeholder braces,
    # no $, no angle brackets)
    "credentials in a URL": re.compile(r"\b[A-Za-z][A-Za-z0-9+.\-]*://[^\s/:@'\"<>{}$]+:[^\s/@'\"<>{}$]+@"),
    # an Authorization bearer value, or a JSON web token (two base64url JSON parts and a signature)
    "bearer token": re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]{20,}=*"
                               r"|\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"),
    # the slashes are escaped so this source line does not match its own pattern when the file is scanned
    "home directory path": re.compile(r"[A-Za-z]:\\Users\\[^\\/\s\"']+"
                                      r"|\/home\/[^/\s\"']+\/|\/Users\/[^/\s\"']+\/"),
}
SKIP_SUFFIXES = {".png", ".jpg", ".gif", ".pdf", ".gguf", ".safetensors", ".sqlite", ".bin"}


def scan(path: Path) -> list[str]:
    if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    hits = []
    for number, line in enumerate(split_lines(text), 1):
        for kind, pattern in PATTERNS.items():
            if pattern.search(line):
                hits.append(f"{path}:{number}: {kind}")
    return hits


HISTORY_EXCLUDE = {"uv.lock"}  # what the no-secrets hook of .pre-commit-config.yaml leaves out


class NotARepository(RuntimeError):
    """The folder --history was given is not in a git repository, or git could not list its objects."""


def _git(repo: Path, *args: str, stdin: bytes | None = None) -> bytes:
    done = subprocess.run(["git", "-C", str(repo), "-c", "core.quotepath=off", *args], input=stdin,
                          capture_output=True, check=False)
    if done.returncode:
        error = done.stderr.decode("utf-8", "replace").strip()
        raise NotARepository(f"{repo}: git {args[0]} failed: {error}")
    return done.stdout


def history_blobs(repo: Path) -> list[tuple[str, str]]:
    """Every blob reachable from any ref of repo, once each, with the first path git rev-list --objects --all
    names it by, sorted by path."""
    named: dict[str, str] = {}
    for line in _git(repo, "rev-list", "--objects", "--all").decode("utf-8").splitlines():
        sha, _, path = line.partition(" ")
        if path and sha not in named:
            named[sha] = path
    if not named:
        return []
    listed = _git(repo, "cat-file", "--batch-check=%(objectname) %(objecttype)",
                  stdin="".join(f"{sha}\n" for sha in named).encode("ascii")).decode("ascii")
    blobs = {sha for sha, _, kind in (line.partition(" ") for line in listed.splitlines()) if kind == "blob"}
    return sorted(((sha, path) for sha, path in named.items() if sha in blobs), key=lambda pair: pair[::-1])


def scan_history(repo: Path) -> list[str]:
    """scan's hits over every blob reachable from any ref of repo (history_blobs), named "<path> (blob
    <12 hex>):<line>: <kind>": a key or path that a later commit removed is still in the history a push
    publishes. Binaries (SKIP_SUFFIXES), uv.lock (HISTORY_EXCLUDE) and blobs that are not UTF-8 are skipped,
    as the hook skips them; NotARepository when git cannot list the objects."""
    blobs = [(sha, path) for sha, path in history_blobs(repo)
             if PurePosixPath(path).suffix.lower() not in SKIP_SUFFIXES and path not in HISTORY_EXCLUDE]
    hits: list[str] = []
    if not blobs:
        return hits
    with subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE) as git:
        for sha, path in blobs:  # one request, one answer: the pipes never fill up
            git.stdin.write(f"{sha}\n".encode("ascii"))
            git.stdin.flush()
            header = git.stdout.readline().split()
            if len(header) != 3:  # "<sha> missing": not in this clone
                continue
            data = git.stdout.read(int(header[2]))
            git.stdout.read(1)  # the newline after the content
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError:
                continue
            for number, line in enumerate(split_lines(text), 1):
                for kind, pattern in PATTERNS.items():
                    if pattern.search(line):
                        hits.append(f"{path} (blob {sha[:12]}):{number}: {kind}")
        git.stdin.close()
    return hits


def history_main(argv: list[str]) -> int:
    """--history [<folder>]: scan_history of the repository at folder (default: the current one)."""
    if len(argv) > 1:
        print("usage: check_no_secrets.py --history [<folder>]", file=sys.stderr)
        return 2
    try:
        hits = scan_history(Path(argv[0]) if argv else Path("."))
    except NotARepository as exc:
        print(exc, file=sys.stderr)
        return 2
    for hit in hits:
        print(hit)
    return 1 if hits else 0


def main(argv: list[str]) -> int:
    if argv[:1] == ["--history"]:
        return history_main(argv[1:])
    absent = [name for name in argv if not Path(name).is_file()]
    for name in absent:
        print(f"{name}: not an existing file, nothing scanned", file=sys.stderr)
    hits = [hit for name in argv if name not in absent for hit in scan(Path(name))]
    for hit in hits:
        print(hit)
    if absent:
        return 2
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

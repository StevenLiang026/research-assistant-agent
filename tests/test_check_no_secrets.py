"""The pre-commit secret scanner must catch key-like strings and home paths and let clean files through.

The patterns are built at runtime so this file itself never contains anything the hook would refuse.
"""

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_no_secrets.py"
spec = importlib.util.spec_from_file_location("check_no_secrets", SCRIPT)
check_no_secrets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_no_secrets)

FAKE_KEY = "sk" + "-" + "0123456789abcdef" * 2
AWS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"  # the documented example id: no separator after the prefix
HOME_PATH = "C:" + "\\" + "Users" + "\\" + "someone" + "\\" + "file.txt"


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_key_like_strings_are_reported_with_line_numbers(tmp_path):
    path = write(tmp_path, "config.py", f"x = 1\nKEY = '{FAKE_KEY}'\n")
    assert check_no_secrets.scan(path) == [f"{path}:2: api key"]


def test_aws_access_key_ids_are_reported(tmp_path):
    path = write(tmp_path, "settings.ini", f"id = {AWS_KEY}\n")
    assert check_no_secrets.scan(path) == [f"{path}:1: api key"]


def test_home_directory_paths_are_reported(tmp_path):
    path = write(tmp_path, "notes.md", f"see {HOME_PATH}\n")
    assert check_no_secrets.scan(path) == [f"{path}:1: home directory path"]


def test_clean_files_pass_and_binaries_are_skipped(tmp_path):
    clean = write(tmp_path, "clean.py", "DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')\n")
    binary = tmp_path / "model.gguf"
    binary.write_bytes(FAKE_KEY.encode())
    assert check_no_secrets.scan(clean) == [] and check_no_secrets.scan(binary) == []


@pytest.mark.parametrize("text, code", [("fine\n", 0), (FAKE_KEY + "\n", 1)])
def test_exit_code(tmp_path, text, code, capsys):
    path = write(tmp_path, "f.txt", text)
    assert check_no_secrets.main([str(path)]) == code
    assert (str(path) in capsys.readouterr().out) == (code == 1)


def test_a_path_that_is_not_an_existing_file_exits_2_and_is_named_while_the_others_are_scanned(tmp_path,
                                                                                               capsys):
    clean = write(tmp_path, "clean.txt", "fine\n")
    keyed = write(tmp_path, "keyed.txt", FAKE_KEY + "\n")
    missing = tmp_path / "missing.txt"
    assert check_no_secrets.main([str(clean), str(missing), str(keyed)]) == 2
    out, err = capsys.readouterr()
    assert str(missing) in err and f"{keyed}:1: api key" in out
    assert check_no_secrets.main([str(tmp_path)]) == 2


def test_a_hit_after_a_line_holding_a_line_separator_gets_the_right_line_number(tmp_path):
    path = write(tmp_path, "data.jsonl", f'{{"text": "a b"}}\nKEY = "{FAKE_KEY}"\n')
    assert check_no_secrets.scan(path) == [f"{path}:2: api key"]


# ---- more credential shapes: one case each, every value made up and assembled at runtime -------------------

ALNUM36 = "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
CREDENTIALS = {
    "hugging face token": ("hf" + "_" + ALNUM36[:34], "api key"),
    "github server token": ("ghs" + "_" + ALNUM36, "api key"),
    "github user token": ("ghu" + "_" + ALNUM36, "api key"),
    "github refresh token": ("ghr" + "_" + ALNUM36, "api key"),
    "slack app token": ("xoxa" + "-" + "2-1234567890-" + ALNUM36[:12], "api key"),
    "slack refresh token": ("xoxe" + "-" + "1-" + ALNUM36, "api key"),
    "google api key": ("AIza" + "Sy" + ALNUM36[:33], "api key"),
    "rsa private key block": ("-----BEGIN " + "RSA PRIVATE KEY" + "-----", "private key"),
    "openssh private key block": ("-----BEGIN " + "OPENSSH PRIVATE KEY" + "-----", "private key"),
    "pgp private key block": ("-----BEGIN " + "PGP PRIVATE KEY BLOCK" + "-----", "private key"),
    "password in a url": ("https://" + "deploy" + ":" + "s3cret-" + ALNUM36[:8] + "@"
                          + "example.com/repo.git", "credentials in a URL"),
    "bearer token": ("Authorization: " + "Bearer" + " " + ALNUM36, "bearer token"),
    "json web token": ("eyJ" + "hbGciOiJIUzI1NiJ9" + "." + "eyJ" + "zdWIiOiIxMjM0NTY3ODkwIn0" + "." + ALNUM36,
                       "bearer token"),
}


@pytest.mark.parametrize("name", sorted(CREDENTIALS))
def test_each_further_credential_shape_is_reported(tmp_path, name):
    value, kind = CREDENTIALS[name]
    path = write(tmp_path, "settings.txt", f"fine\nvalue = {value}\n")
    assert check_no_secrets.scan(path) == [f"{path}:2: {kind}"]


@pytest.mark.parametrize("text", [
    'headers = {"Authorization": f"Bearer {token}"}',  # a template, not a token
    "LOCAL_OPENAI_BASE_URL = http://127.0.0.1:8080/v1",  # a port, no credentials
    "git clone git@github.com:owner/repo.git",  # a user name without a password
    "ssh://git@example.com/repo.git",
    "https://user@example.com/path",
    "postgres://{user}:{password}@{host}/db",  # placeholders
    "-----BEGIN " + "PUBLIC KEY" + "-----",
    "hf_" + "short",
    "the AIza prefix alone",
])
def test_lookalikes_of_the_further_shapes_pass(tmp_path, text):
    assert check_no_secrets.scan(write(tmp_path, "clean.txt", text + "\n")) == []


# ---- --history: every blob reachable from any ref ---------------------------------------------------------


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
                           "-c", "commit.gpgsign=false", *args],
                          check=True, capture_output=True, text=True, encoding="utf-8").stdout


def commit(repo: Path, files: dict[str, str | bytes | None], message: str) -> None:
    """One commit writing (a str or bytes) or deleting (None) each file."""
    for name, content in files.items():
        path = repo / name
        if content is None:
            git(repo, "rm", "-q", name)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
        git(repo, "add", name)
    git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    folder = tmp_path / "repo"
    folder.mkdir()
    git(folder, "init", "-q")
    return folder


def blob_of(repo: Path, rev: str, name: str) -> str:
    return git(repo, "rev-parse", f"{rev}:{name}").strip()


def test_history_reports_a_key_an_earlier_commit_held_though_the_tree_no_longer_does(repo):
    commit(repo, {"config.py": "x = 1\n"}, "one")
    commit(repo, {"config.py": f"x = 1\nKEY = '{FAKE_KEY}'\n", "docs/notes.md": f"see {HOME_PATH}\n"}, "two")
    leaked, noted = blob_of(repo, "HEAD", "config.py"), blob_of(repo, "HEAD", "docs/notes.md")
    commit(repo, {"config.py": "x = 1\n", "docs/notes.md": None}, "three")
    assert check_no_secrets.scan(repo / "config.py") == []
    assert check_no_secrets.scan_history(repo) == [
        f"config.py (blob {leaked[:12]}):2: api key",
        f"docs/notes.md (blob {noted[:12]}):1: home directory path"]


def test_history_scans_the_blobs_of_every_branch_once_each(repo):
    commit(repo, {"a.txt": "fine\n"}, "one")
    git(repo, "checkout", "-q", "-b", "side")
    commit(repo, {"b.txt": FAKE_KEY + "\n"}, "side")
    git(repo, "checkout", "-q", "-")
    commit(repo, {"c.txt": FAKE_KEY + "\n"}, "main")  # the same bytes as b.txt: one blob, named once
    hits = check_no_secrets.scan_history(repo)
    blob = blob_of(repo, "side", "b.txt")[:12]
    assert hits in ([f"b.txt (blob {blob}):1: api key"], [f"c.txt (blob {blob}):1: api key"])


def test_history_skips_binaries_and_the_lock_file_as_the_hook_does(repo):
    commit(repo, {"model.gguf": FAKE_KEY.encode(), "uv.lock": FAKE_KEY + "\n",
                  "image.png": b"\x89PNG" + bytes(8),
                  "clean.py": "DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')\n"}, "one")
    assert check_no_secrets.scan_history(repo) == []


def test_history_names_files_with_non_ascii_names_as_they_are(repo):
    commit(repo, {"docs/进度.md": f"KEY = {FAKE_KEY}\n"}, "one")
    assert check_no_secrets.scan_history(repo) == [
        f"docs/进度.md (blob {blob_of(repo, 'HEAD', 'docs/进度.md')[:12]}):1: api key"]


def test_history_exit_codes(repo, tmp_path, capsys):
    commit(repo, {"a.txt": "fine\n"}, "one")
    assert check_no_secrets.main(["--history", str(repo)]) == 0
    commit(repo, {"a.txt": FAKE_KEY + "\n"}, "two")
    commit(repo, {"a.txt": "fine\n"}, "three")
    assert check_no_secrets.main(["--history", str(repo)]) == 1
    assert "a.txt (blob " in capsys.readouterr().out
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    assert check_no_secrets.main(["--history", str(not_a_repo)]) == 2
    assert str(not_a_repo) in capsys.readouterr().err

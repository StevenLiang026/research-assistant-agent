"""README result numbers against the report: research_agent.eval.numbers and its command line."""

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from research_agent.eval import numbers
from research_agent.eval.lock import sha256_of

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_readme_numbers.py"
spec = importlib.util.spec_from_file_location("check_readme_numbers", SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

SOURCE = "results/test_ft.json"
VALUES = {"test.ft.success": "61.3%", "test.b0.success": "42.0%", "test.diff.ci95": "[12.4, 26.1]"}
README = (
    "# Agent\n"
    "\n"
    "单题成功率由 <!-- num:test.b0.success -->42.0%<!-- /num --> 升到 "
    "<!-- num:test.ft.success --> 61.3% <!-- /num -->,\n"
    "差值的 95% 置信区间 <!-- num:test.diff.ci95 -->[12.4, 26.1]<!-- /num -->。\n"
)
NEXT_LINE = 5  # the line where text appended to README starts
OPENS = "测试集共打开 <!-- num:test_set.opens -->{n}<!-- /num --> 次。\n"
NOT_IGNORED = {"ignored": lambda path: False}  # tmp_path is no repository: say so instead of asking git


def make_repo(root: Path, readme: str, *, report: dict | None = VALUES, opens: int | None = None) -> Path:
    """README.md; a result file and the report built from it unless report is None; opens log entries."""
    (root / "README.md").write_text(readme, encoding="utf-8")
    if report is not None:
        (root / "results").mkdir(exist_ok=True)
        (root / SOURCE).write_text('{"success": 0.613}\n', encoding="utf-8")
        numbers.write_numbers(report, [SOURCE], root=root, **NOT_IGNORED)
    if opens is not None:
        log = root / "data" / "eval" / "test-access.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("".join(json.dumps({"milestone": f"M{i + 3}"}) + "\n" for i in range(opens)),
                       encoding="utf-8")
    return root


def problems(root: Path, **kwargs) -> list[str]:
    return numbers.check(root=root, **kwargs)


def test_a_readme_without_markers_passes_with_or_without_a_report(tmp_path):
    assert problems(make_repo(tmp_path, "# Agent\n\n成功率 61.3%。\n", report=None)) == []
    assert problems(make_repo(tmp_path, "# Agent\n", report=VALUES)) == []


def test_quoted_numbers_that_are_the_reports_pass(tmp_path):
    again = "再引一次:<!-- num:test.ft.success -->61.3%<!-- /num -->\n"
    assert problems(make_repo(tmp_path, README + again)) == []
    found = numbers.markers(README)
    assert [key for key, _, _ in found] == ["test.b0.success", "test.ft.success", "test.diff.ci95"]
    assert found[1] == ("test.ft.success", "61.3%", 3)  # whitespace inside the block dropped, line counted


def test_a_number_that_is_not_the_reports_is_reported_with_line_key_and_both_values(tmp_path):
    found = problems(make_repo(tmp_path, README.replace(" 61.3% ", "61.4%")))
    assert found == ["README.md:3: test.ft.success is '61.4%', the report says '61.3%'"]


def test_a_key_the_report_does_not_have_is_reported(tmp_path):
    found = problems(make_repo(tmp_path, README + "<!-- num:test.b1.success -->50.0%<!-- /num -->\n"))
    assert found == [f"README.md:{NEXT_LINE}: test.b1.success is not in results/numbers.json"]


def test_markers_without_a_report_fail(tmp_path):
    found = problems(make_repo(tmp_path, README, report=None))
    assert found == ["README.md: 3 marker(s) but no report at results/numbers.json"]


@pytest.mark.parametrize("broken", ["<!-- num:test.ft.success -->61.3%\n",
                                    "<!-- num:bad key -->1<!-- /num -->\n"])
def test_a_malformed_marker_block_is_reported(tmp_path, broken):
    found = problems(make_repo(tmp_path, README + broken))
    assert len(found) == 1 and found[0].startswith(f"README.md:{NEXT_LINE}: malformed marker block")


def test_an_unclosed_block_does_not_swallow_the_next_one(tmp_path):
    readme = "<!-- num:test.b0.success -->42.0%\n<!-- num:test.ft.success -->61.3%<!-- /num -->\n"
    assert numbers.markers(readme) == [("test.ft.success", "61.3%", 2)] and numbers.unclosed(readme) == [1]
    found = problems(make_repo(tmp_path, readme))
    assert len(found) == 1 and found[0].startswith("README.md:1: malformed marker block")


def test_markers_shown_inside_code_are_not_read(tmp_path):
    readme = ("结果数字写成 `<!-- num:KEY -->VALUE<!-- /num -->`,例如:\n\n"
              "```markdown\n<!-- num:test.ft.success -->99%<!-- /num -->\n<!-- num:open\n```\n"
              "成功率 <!-- num:test.ft.success -->61.3%<!-- /num -->\n")
    assert numbers.markers(readme) == [("test.ft.success", "61.3%", 7)] and numbers.unclosed(readme) == []
    assert problems(make_repo(tmp_path, readme)) == []


def test_a_changed_or_missing_source_makes_the_report_stale(tmp_path):
    root = make_repo(tmp_path, README)
    (root / SOURCE).write_text('{"success": 0.700}\n', encoding="utf-8")
    stale = f"results/numbers.json: source {SOURCE} changed since the report was built; build it again"
    assert problems(root) == [stale]
    (root / SOURCE).unlink()
    assert problems(root) == [f"results/numbers.json: source {SOURCE} is missing"]


def test_a_report_that_is_not_json_is_a_problem_line_not_a_crash(tmp_path):
    root = make_repo(tmp_path, README)
    (root / numbers.NUMBERS).write_text("{not json", encoding="utf-8")
    found = problems(root)
    assert len(found) == 1 and found[0].startswith("results/numbers.json: not valid JSON")


def test_test_set_opens_must_be_the_number_of_access_log_entries(tmp_path):
    assert problems(make_repo(tmp_path, OPENS.format(n=2), report=None, opens=2)) == []  # needs no report
    found = problems(make_repo(tmp_path, OPENS.format(n=2), report=None, opens=1))
    assert found == ["README.md:1: test_set.opens is '2', the access log data/eval/test-access.jsonl has 1"]
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    assert problems(make_repo(fresh, OPENS.format(n=0), report=None)) == []  # no log yet: no opens


def test_once_the_test_set_was_opened_readme_must_state_how_often(tmp_path):
    found = problems(make_repo(tmp_path, "# Agent\n\n测试集只打开过一次。\n", report=None, opens=2))
    expected = ("README.md: the test set was opened 2 time(s) (access log data/eval/test-access.jsonl); "
                "state it as <!-- num:test_set.opens -->2<!-- /num -->")
    assert found == [expected]


def test_an_unreadable_access_log_is_a_problem_line_not_a_crash(tmp_path):
    root = make_repo(tmp_path, OPENS.format(n=1), report=None, opens=1)
    (root / "data" / "eval" / "test-access.jsonl").write_text("not json\n", encoding="utf-8")
    found = problems(root)
    assert len(found) == 1 and found[0].startswith("data/eval/test-access.jsonl: unreadable access log")


def test_with_tracked_the_report_its_sources_and_the_log_must_be_in_git(tmp_path):
    root = make_repo(tmp_path, README + OPENS.format(n=1), opens=1)
    everything = {"results/numbers.json", SOURCE, "data/eval/test-access.jsonl"}
    assert problems(root, tracked=lambda paths: set(paths) & everything) == []
    found = problems(root, tracked=lambda paths: set(paths) & {"results/numbers.json"})
    untracked = "is not tracked by git: add it, or a clean checkout fails this check"
    assert found == [f"data/eval/test-access.jsonl {untracked}", f"{SOURCE} {untracked}"]
    plain = tmp_path / "plain"
    plain.mkdir()
    asked = []
    assert problems(make_repo(plain, "# Agent\n"), tracked=lambda paths: asked.append(paths) or set()) == []
    assert asked == []  # nothing quoted, nothing logged: git is not asked


def test_write_numbers_stores_formatted_strings_sorted_with_source_hashes(tmp_path):
    root = make_repo(tmp_path, "# Agent\n")
    report = json.loads((root / numbers.NUMBERS).read_text(encoding="utf-8"))
    assert report == {"numbers": dict(sorted(VALUES.items())), "sources": {SOURCE: sha256_of(root / SOURCE)}}
    assert list(report["numbers"]) == sorted(VALUES)
    assert (root / numbers.NUMBERS).read_bytes().endswith(b"}\n")
    with pytest.raises(TypeError, match="formatted strings"):
        numbers.write_numbers({"test.ft.success": 0.613}, [SOURCE], root=root, **NOT_IGNORED)
    with pytest.raises(ValueError, match="does not match"):
        numbers.write_numbers({"bad key": "1"}, [SOURCE], root=root, **NOT_IGNORED)


def test_write_numbers_refuses_sources_outside_the_repository_or_ignored_by_git(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    make_repo(root, "# Agent\n")
    (tmp_path / "elsewhere.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the repository"):
        numbers.write_numbers(VALUES, ["../elsewhere.json"], root=root, **NOT_IGNORED)
    with pytest.raises(ValueError, match="ignored by git"):
        numbers.write_numbers(VALUES, [SOURCE], root=root, ignored=lambda path: path == SOURCE)


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_the_command_line_asks_git_and_prints_each_problem(tmp_path, monkeypatch, capsys):
    root = make_repo(tmp_path, README.replace(" 61.3% ", "61.4%"))
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    monkeypatch.chdir(root)
    assert cli.main([]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "README.md:3: test.ft.success is '61.4%', the report says '61.3%'"
    assert out[1:] == [f"{path} is not tracked by git: add it, or a clean checkout fails this check"
                       for path in ("results/numbers.json", SOURCE)]
    (root / "README.md").write_text(README, encoding="utf-8")
    subprocess.run(["git", "-C", str(root), "add", "results"], check=True)
    assert cli.main([]) == 0 and capsys.readouterr().out == ""


# the access logs (data/eval) and the per-item results among the report's sources (results/dev-*.json and
# others): data the public copy leaves out (tests/conftest.py), which has a check of its own
@pytest.mark.needs_data("data/eval", "results/dev-ft.json")
def test_the_repositorys_own_readme_passes():
    assert numbers.check(root=REPO, tracked=lambda paths: numbers.git_tracked(paths, REPO)) == []


# ---- M7: the second family of test sets has its own count, test_set_v2.opens (docs/m7-计划.md WP-I 7) ------

V2_OPENS = "v2 测试集共打开 <!-- num:test_set_v2.opens -->{n}<!-- /num --> 次。\n"
V2_LOG = "data/eval/v2/test-v2-access.jsonl"


def v2_opens(root: Path, opens: int) -> Path:
    """The v2 single-step set's access log with opens entries."""
    log = root / V2_LOG
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("".join(json.dumps({"milestone": "M7", "n": i}) + "\n" for i in range(opens)),
                   encoding="utf-8")
    return log


def test_the_v2_count_is_read_from_the_v2_access_log_of_eval_lock():
    from research_agent.eval.lock import TEST_ACCESS_LOG_V2

    assert numbers.TEST_SET_V2_OPENS == "test_set_v2.opens" and TEST_ACCESS_LOG_V2.as_posix() == V2_LOG
    assert numbers.check.__kwdefaults__["access_log_v2"] == TEST_ACCESS_LOG_V2


def test_test_set_v2_opens_must_be_the_number_of_v2_access_log_entries_and_needs_no_report(tmp_path):
    root = make_repo(tmp_path, README + V2_OPENS.format(n=1))  # a report without the key: not looked up there
    v2_opens(root, 1)
    assert problems(root) == []
    v2_opens(root, 2)
    stale = f"README.md:{NEXT_LINE}: test_set_v2.opens is '1', the access log {V2_LOG} has 2"
    assert problems(root) == [stale]
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    assert problems(make_repo(fresh, V2_OPENS.format(n=0), report=None)) == []  # no log yet: no opens


def test_once_the_v2_test_set_was_opened_readme_must_state_how_often(tmp_path):
    root = make_repo(tmp_path, OPENS.format(n=3), report=None, opens=3)
    v2_opens(root, 1)
    expected = (f"README.md: the v2 test set was opened 1 time(s) (access log {V2_LOG}); state it as "
                "<!-- num:test_set_v2.opens -->1<!-- /num -->")
    assert problems(root) == [expected]


def test_the_two_families_opens_are_counted_apart(tmp_path):
    root = make_repo(tmp_path, OPENS.format(n=3) + V2_OPENS.format(n=0), report=None, opens=3)
    assert problems(root) == []
    v2_opens(root, 3)  # the v1 count does not stand for the v2 one, nor the other way round
    assert problems(root) == [f"README.md:2: test_set_v2.opens is '0', the access log {V2_LOG} has 3"]
    swapped = tmp_path / "swapped"
    swapped.mkdir()
    root = make_repo(swapped, OPENS.format(n=1) + V2_OPENS.format(n=3), report=None, opens=3)
    v2_opens(root, 1)
    assert problems(root) == [
        "README.md:1: test_set.opens is '1', the access log data/eval/test-access.jsonl has 3",
        f"README.md:2: test_set_v2.opens is '3', the access log {V2_LOG} has 1"]


def test_the_v2_access_log_must_be_tracked_once_opened_and_an_unreadable_one_is_a_problem_line(tmp_path):
    root = make_repo(tmp_path, OPENS.format(n=1) + V2_OPENS.format(n=1), report=None, opens=1)
    v2_opens(root, 1)
    logs = {"data/eval/test-access.jsonl", V2_LOG}
    assert problems(root, tracked=lambda paths: set(paths) & logs) == []
    untracked = "is not tracked by git: add it, or a clean checkout fails this check"
    assert problems(root, tracked=lambda paths: set(paths) - {V2_LOG}) == [f"{V2_LOG} {untracked}"]
    (root / V2_LOG).write_text("not json\n", encoding="utf-8")
    found = problems(root)
    assert len(found) == 1 and found[0].startswith(f"{V2_LOG}: unreadable access log")

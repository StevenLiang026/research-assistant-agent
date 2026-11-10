"""scripts/check_public_numbers.py: the README number check of the public copy (README markers against
results/numbers.json, the opening counts against docs/provenance.md, the present sources' hashes)."""

import importlib.util
import json
from pathlib import Path

from research_agent.eval.lock import sha256_of

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_public_numbers.py"
spec = importlib.util.spec_from_file_location("check_public_numbers", SCRIPT)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)

README = ("# Results\n\nFT reaches <!-- num:test.ft.success --> 61.3% <!-- /num --> on the test set, opened "
          "<!-- num:test_set.opens -->3<!-- /num --> times.\n")
PROVENANCE = "Opened <!-- num:test_set.opens -->3<!-- /num --> times.\n"


def make(tmp_path: Path, readme: str = README, provenance: str | None = PROVENANCE,
         sources: dict[str, bytes] | None = None, absent: tuple[str, ...] = ()) -> Path:
    root = tmp_path / "copy"
    (root / "results").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "README.md").write_text(readme, encoding="utf-8")
    if provenance is not None:
        (root / "docs" / "provenance.md").write_text(provenance, encoding="utf-8")
    hashed = {}
    for name, data in (sources or {"results/a.json": b"{}\n"}).items():
        (root / name).write_bytes(data)
        hashed[name] = sha256_of(root / name)
    hashed.update({name: "0" * 64 for name in absent})
    report = {"numbers": {"test.ft.success": "61.3%"}, "sources": hashed}
    (root / "results" / "numbers.json").write_text(json.dumps(report), encoding="utf-8")
    return root


def test_numbers_that_are_the_reports_and_counts_that_are_recorded_pass(tmp_path):
    problems, counts = cli.check(root=make(tmp_path, absent=("results/test-m3-ft.json",)))
    assert problems == []
    assert counts == {"markers": 1, "opens": 1, "sources_present": 1, "sources_absent": 1}


def test_a_number_that_is_not_the_reports_is_reported(tmp_path):
    problems, _ = cli.check(root=make(tmp_path, README.replace("61.3%", "61.4%")))
    assert problems == ["README.md:3: test.ft.success is '61.4%', the report says '61.3%'"]


def test_a_key_the_report_does_not_have_and_a_malformed_block_are_reported(tmp_path):
    readme = README + "Also <!-- num:other.key -->1<!-- /num --> and <!-- num:bad key -->2<!-- /num -->.\n"
    problems, _ = cli.check(root=make(tmp_path, readme))
    assert problems == ["README.md:4: malformed marker block",
                        "README.md:4: other.key is not in results/numbers.json"]


def test_an_opening_count_must_be_the_one_the_provenance_records(tmp_path):
    problems, _ = cli.check(root=make(tmp_path, provenance=PROVENANCE.replace(">3<", ">2<")))
    assert problems == ["README.md:3: test_set.opens is '3', docs/provenance.md records '2'"]
    problems, _ = cli.check(root=make(tmp_path / "none", provenance=None))
    assert problems == ["README.md:3: test_set.opens is not recorded in docs/provenance.md"]


def test_a_present_source_must_keep_its_hash_and_an_absent_one_is_only_counted(tmp_path):
    root = make(tmp_path, absent=("results/kept-private.json",))
    (root / "results" / "a.json").write_bytes(b"{\"changed\": true}\n")
    problems, counts = cli.check(root=root)
    assert problems == ["results/numbers.json: source results/a.json changed since the report was built"]
    assert counts["sources_absent"] == 1


def test_the_command_line_prints_each_problem_and_a_summary(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(make(tmp_path, README.replace("61.3%", "60.0%")))
    assert cli.main([]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("README.md:3: test.ft.success is '60.0%'") and out[-1].endswith("1 problem(s)")


def test_the_repositorys_own_readme_passes(monkeypatch, capsys):
    monkeypatch.chdir(REPO)
    assert cli.main([]) == 0, capsys.readouterr().out

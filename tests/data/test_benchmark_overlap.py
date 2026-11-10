"""The 13-gram overlap check against BFCL and ACEBench (research_agent.data.overlap,
scripts/check_benchmark_overlap.py) and the trainer skipping the records it names, on fixture benchmark
files."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
from train.tokenizer_fake import FakeTokenizer

from research_agent.data import overlap

REPO = Path(__file__).resolve().parents[2]
EN_QUESTION = ("Find the area of a triangle with a base of ten units and a height of five units "
               "please and round it")
ZH_QUESTION = "我需要计划一个拍卖活动。我已经有了活动编号和地点在纽约大会堂。"
ZH_EMAIL = "john.doe@example.com"
TOOL = {"type": "function", "function": {"name": "f", "description": "d", "parameters": {"type": "object",
                                                                                           "properties": {}}}}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load_script("check_benchmark_overlap")
train = load_script("train")


def write_lines(path: Path, rows: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


@pytest.fixture()
def benchmarks(tmp_path) -> tuple[Path, Path]:
    bfcl = tmp_path / "bfcl"
    write_lines(bfcl / "BFCL_v4_simple_python.json", [
        {"id": "simple_python_0", "question": [[{"role": "user", "content": EN_QUESTION}]], "function": []},
        {"id": "simple_python_1", "question": [[{"role": "system", "content": "be brief " * 10},
                                                {"role": "user", "content": "short"}]]}])
    (bfcl / "BFCL_v4_format_sensitivity.json").write_text('{\n  "simple_python": ["simple_python_0"]\n}\n',
                                                          encoding="utf-8")  # no question entries
    ace = tmp_path / "ace"
    write_lines(ace / "data_zh" / "data_special_incomplete.json", [
        {"id": "special_incomplete_0", "question": f"user: {ZH_QUESTION}\nsystem: 请提供时间。\nuser: 明天",
         "function": "[]", "time": ""}])
    write_lines(ace / "data_zh" / "data_normal_atom_bool.json", [
        {"id": "normal_atom_bool_2", "question": f"user: 请把报告发到{ZH_EMAIL}，谢谢。", "function": "[]",
         "time": ""}])
    write_lines(ace / "data_en" / "data_normal_single_turn_single_function.json", [
        {"id": "normal_single_turn_single_function_0",
         "question": "user: I would like to know\nwhen the next total solar eclipse happens over the city of "
                     "Lisbon in Portugal", "function": "[]", "time": ""}])
    write_lines(ace / "data_en" / "data_agent_multi_step.json", [
        {"id": "agent_multi_step_0", "question": "user: " + "agent only words here " * 5}])  # not read
    return bfcl, ace


def sample(user: str) -> dict:
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": "ok"}],
            "tools": [TOOL]}


def test_text_ngrams_are_word_grams_lower_cased_after_normalize_text_or_character_grams_for_cjk():
    words = " ".join(f"W{i}" for i in range(14))
    assert overlap.text_ngrams(words) == {" ".join(f"w{i}" for i in range(13)),
                                          " ".join(f"w{i}" for i in range(1, 14))}
    assert overlap.text_ngrams("one two three") == set()
    assert overlap.text_ngrams("Ｗ0  w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12") == {
        " ".join(f"w{i}" for i in range(13))}
    zh = "我需要 计划一个拍卖活动编号十二"  # 15 non-space characters: three character 13-grams
    assert overlap.text_ngrams(zh) == {"我需要计划一个拍卖活动编号", "需要计划一个拍卖活动编号十",
                                       "要计划一个拍卖活动编号十二"}


def test_text_with_cjk_gives_character_grams_only_for_mostly_cjk_windows_and_word_grams_too():
    # 23 non-space characters: no window of 13 holds 7 CJK characters; the e-mail address is one word, too
    # short for a word gram
    assert overlap.text_ngrams(f"请发到 {ZH_EMAIL}") == set()
    words = " ".join(f"W{i}" for i in range(13))
    mixed = overlap.text_ngrams(f"查一下{words}谢谢")  # CJK characters separate the words like spaces
    assert " ".join(f"w{i}" for i in range(13)) in mixed
    assert all(overlap._CJK.search(g) for g in mixed if " " not in g)


def test_a_character_window_counts_only_when_most_of_its_characters_are_cjk():
    # the public-data false positives of the first rule: one CJK character before a placeholder phone number,
    # five before a placeholder name
    assert overlap.text_ngrams("电话是123-456-7890") == set()
    assert overlap.text_ngrams("我的名字是john doe,") == set()
    assert overlap.text_ngrams("一二三四五六七abcdef") == {"一二三四五六七abcdef"}  # 7 of 13: a majority
    assert overlap.text_ngrams("一二三四五六abcdefg") == set()  # 6 of 13


def test_a_latin_fragment_in_chinese_text_does_not_match_but_genuine_overlaps_in_either_script_do(benchmarks,
                                                                                                  tmp_path):
    words = EN_QUESTION.split()
    path = write_lines(tmp_path / "train.jsonl", [
        sample(f"我的邮箱是{ZH_EMAIL}。"),  # shares only Latin-only windows with ACEBench
        sample("帮我看看，" + ZH_QUESTION[2:15] + "。"),  # 13 consecutive characters of the ACEBench question
        sample("帮我算一下" + " ".join(words[3:16]) + "谢谢"),  # 13 BFCL words inside Chinese text
        sample("帮我算一下" + " ".join(words[3:15]) + "谢谢"),  # 12 words only
        sample("帮" + ZH_QUESTION[2:14] + "吧"),  # 12 characters only
    ])
    report = overlap.check_file(path, overlap.BenchmarkIndex.build(*benchmarks))
    hits = {r["index"]: r["matches"][0] for r in report["records_overlapping"]}
    assert sorted(hits) == [1, 2]
    assert hits[1]["entry"] == "special_incomplete_0" and hits[1]["gram"] == ZH_QUESTION[2:15]
    assert hits[2]["entry"] == "simple_python_0" and hits[2]["gram"] == " ".join(words[3:16]).lower()


def test_the_benchmark_index_reads_bfcl_user_turns_and_acebench_normal_and_special_user_turns(benchmarks):
    index = overlap.BenchmarkIndex.build(*benchmarks)
    files = {Path(f["path"]).name: f for f in index.files}
    assert files["BFCL_v4_simple_python.json"]["entries"] == 2
    assert files["BFCL_v4_format_sensitivity.json"]["entries"] == 0
    assert "data_agent_multi_step.json" not in files
    zh = files["data_special_incomplete.json"]
    assert zh["benchmark"] == "acebench" and zh["questions"] == 2  # two user turns, the system turn left out
    assert zh["sha256"] == hashlib.sha256(Path(zh["path"]).read_bytes()).hexdigest()
    assert overlap.acebench_user_turns("user: a\nb\nsystem: c\nuser: d") == ["a\nb", "d"]
    assert overlap.acebench_user_turns("no prefix at all") == ["no prefix at all"]


def test_an_english_13_gram_and_a_chinese_character_13_gram_hit_and_a_12_gram_near_miss_does_not(benchmarks,
                                                                                                  tmp_path):
    words = EN_QUESTION.split()
    near_miss = " ".join(words[:12]) + " XX " + " ".join(words[12:13])  # 12 shared words, then a break
    path = write_lines(tmp_path / "train.jsonl", [
        sample("Please: " + " ".join(words[3:16]) + " thanks"),  # 13 consecutive words of the BFCL question
        sample("帮我看看，" + ZH_QUESTION[2:15] + "。"),  # 13 consecutive characters of the ACEBench question
        sample(near_miss),
        {**sample("unrelated"), "id": "req-x"},
    ])
    index = overlap.BenchmarkIndex.build(*benchmarks)
    report = overlap.check_file(path, index)
    assert report["records"] == 4 and report["overlapping"] == 2
    first, second = report["records_overlapping"]
    assert first["index"] == 0 and first["matches"][0]["benchmark"] == "bfcl"
    assert first["matches"][0]["entry"] == "simple_python_0"
    assert second["index"] == 1 and second["matches"][0]["entry"] == "special_incomplete_0"
    assert second["matches"][0]["benchmark"] == "acebench"
    assert all(r["index"] != 2 for r in report["records_overlapping"])


def test_the_cli_writes_the_report_per_file_with_counts_and_the_benchmark_files_with_their_hashes(
        benchmarks, tmp_path, capsys):
    hit = write_lines(tmp_path / "a.jsonl", [sample("x"), {**sample(EN_QUESTION), "id": "req-7"}])
    clean = write_lines(tmp_path / "b.jsonl", [sample("nothing to see")])
    out = tmp_path / "overlap.json"
    argv = [str(hit), str(clean), "--out", str(out), "--bfcl-dir", str(benchmarks[0]),
            "--acebench-dir", str(benchmarks[1])]
    assert script.main(argv) == 0
    printed = capsys.readouterr().out
    assert "a.jsonl: 1 of 2 records overlap" in printed and "b.jsonl: 0 of 1 records overlap" in printed
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["n"] == 13 and report["total"] == {"records": 3, "overlapping": 1}
    assert [f["path"] for f in report["files"]] == [hit.as_posix(), clean.as_posix()]
    entry = report["files"][0]
    assert entry["sha256"] == hashlib.sha256(hit.read_bytes()).hexdigest()
    assert entry["records_overlapping"][0]["index"] == 1 and entry["records_overlapping"][0]["id"] == "req-7"
    assert {Path(b["path"]).name for b in report["benchmarks"]} >= {"BFCL_v4_simple_python.json",
                                                                    "data_special_incomplete.json"}
    assert all(len(b["sha256"]) == 64 for b in report["benchmarks"])


def test_the_cli_exits_2_without_the_benchmark_folders_and_1_on_a_malformed_training_file(benchmarks,
                                                                                           tmp_path, capsys):
    data = write_lines(tmp_path / "a.jsonl", [sample("x")])
    out = ["--out", str(tmp_path / "o.json")]
    assert script.main([str(data), *out, "--bfcl-dir", str(tmp_path / "none"),
                        "--acebench-dir", str(benchmarks[1])]) == 2
    err = capsys.readouterr().err
    assert "none" in err and "BFCL" in err and not (tmp_path / "o.json").exists()
    assert script.main([str(data), *out, "--bfcl-dir", str(benchmarks[0]),
                        "--acebench-dir", str(tmp_path / "none")]) == 2
    assert "ACEBench" in capsys.readouterr().err
    (tmp_path / "bad.jsonl").write_text("{not json\n", encoding="utf-8")
    assert script.main([str(tmp_path / "bad.jsonl"), *out, "--bfcl-dir", str(benchmarks[0]),
                        "--acebench-dir", str(benchmarks[1])]) == 1


@pytest.mark.parametrize("layout", ["Lib/site-packages", "lib/python3.12/site-packages"])
def test_without_bfcl_dir_the_data_folder_is_found_in_either_virtualenv_layout(layout, benchmarks, tmp_path,
                                                                             monkeypatch, capsys):
    bfcl, ace = benchmarks
    work = tmp_path / "work"
    shutil.copytree(bfcl, work / ".venv-bfcl" / layout / "bfcl_eval" / "data")
    data = write_lines(work / "a.jsonl", [sample("x")])
    monkeypatch.chdir(work)
    assert script.find_bfcl_dirs() == [Path(".venv-bfcl", *layout.split("/"), "bfcl_eval", "data")]
    assert script.main([str(data), "--out", "o.json", "--acebench-dir", str(ace)]) == 0
    report = json.loads((work / "o.json").read_text(encoding="utf-8"))
    assert any(Path(b["path"]).name == "BFCL_v4_simple_python.json" for b in report["benchmarks"])
    other = "lib/python3.12/site-packages" if layout.startswith("Lib") else "Lib/site-packages"
    shutil.copytree(bfcl, work / ".venv-bfcl" / other / "bfcl_eval" / "data")
    assert script.main([str(data), "--out", "o2.json", "--acebench-dir", str(ace)]) == 2
    assert "pass --bfcl-dir" in capsys.readouterr().err and not (work / "o2.json").exists()


def test_the_trainer_skips_the_records_the_report_names_and_records_how_many(benchmarks, tmp_path):
    data = write_lines(tmp_path / "a.jsonl", [sample("one"), sample(EN_QUESTION), sample("three")])
    report_path = tmp_path / "overlap.json"
    assert script.main([str(data), "--out", str(report_path), "--bfcl-dir", str(benchmarks[0]),
                        "--acebench-dir", str(benchmarks[1])]) == 0
    skips, info = overlap.load_skips(report_path, [data])
    assert skips == {data.resolve(): {1}} and info["skipped_records"] == 1
    samples, dropped, per_file = train.assemble([data], FakeTokenizer(), 3072, skips=skips)
    assert per_file[0]["records"] == 3 and per_file[0]["skipped_overlap"] == 1 and len(samples) == 2
    assert all("triangle" not in s["prompt"] for s in samples)
    summary = train.data_summary(samples, dropped, per_file, info)
    assert summary["benchmark_overlap"]["skipped"] == 1
    report_sha256 = hashlib.sha256(report_path.read_bytes()).hexdigest()
    assert summary["benchmark_overlap"]["report_sha256"] == report_sha256
    assert train.build_parser().parse_args(["d"]).overlap_report == "data/synth/benchmark-overlap.json"
    assert overlap.load_skips(tmp_path / "absent.json", [data]) == ({}, None)
    data.write_text(data.read_text(encoding="utf-8") + json.dumps(sample("four")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stale"):
        overlap.load_skips(report_path, [data])  # the file changed since the check: indices may be wrong

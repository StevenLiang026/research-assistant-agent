"""The glossary-note option of the training renderers (research_agent.train.samples, docs/m7-本地提升计划.md
WP4 and WP6 注释) and scripts/train.py's --glossary and --glossary-dropout. Offline, over the fake tokenizer
(tokenizer_fake) and the fake family (family_stub)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from family_stub import family_hook, loaded_fake_family
from support.family_fakes import model_folder
from test_train_cli import load_script
from test_train_family_samples import REASONING, sidecar
from test_train_samples import AGENT_RECORD, LONG, PUBLIC_SAMPLE, TOK, accepted, prompt_of, step, transcript

from research_agent.agent.context import step_contexts
from research_agent.canonical import canonical_json
from research_agent.family.core import Mode
from research_agent.glossary import Glossary, noted_messages
from research_agent.tools.schemas import MODEL_TOOLS
from research_agent.train.samples import (
    NOTE_DROPOUT,
    Dropped,
    GlossaryNotes,
    RecordError,
    build_family_samples,
    build_step_samples,
    keeps_note,
)

train = load_script("train")
GLOSSARY = Glossary({"推测解码": "Speculative Decoding", "知识蒸馏": "Knowledge Distillation"})
NOTE = "术语对照:推测解码 = Speculative Decoding"
ZH_LONG = [LONG[0], {**LONG[1], "content": "帮我找 2026 年 9 月关于推测解码的论文"}, *LONG[2:]]
RECORD = {**AGENT_RECORD, "messages": ZH_LONG}
ALWAYS = GlossaryNotes(GLOSSARY, dropout=0.0)  # every record keeps its notes


def ids_with(kept: bool, count: int = 1) -> list[str]:
    """Record ids whose notes the 0.5 dropout keeps (kept) or drops."""
    found = [f"req-train-{n:04d}" for n in range(1000) if keeps_note(f"req-train-{n:04d}", 0.5) is kept]
    return found[:count]


# research_agent.train.samples


def test_without_the_option_the_samples_are_the_default_ones():
    plain, dropped = build_step_samples(RECORD, TOK)
    assert build_step_samples(RECORD, TOK, notes=None) == (plain, dropped)
    assert all(set(s) == {"prompt", "completion", "tokens"} for s in plain)  # no "note" key
    assert NOTE not in "".join(s["prompt"] for s in plain)


def test_the_option_changes_nothing_but_the_note_where_no_note_is_due():
    plain, dropped = build_step_samples(RECORD, TOK)
    for notes in (GlossaryNotes(GLOSSARY, dropout=1.0),  # every record drops its notes
                  GlossaryNotes(Glossary({"知识蒸馏": "Knowledge Distillation"}), dropout=0.0)):  # no term
        samples, same = build_step_samples(RECORD, TOK, notes=notes)
        assert same == dropped and samples == [{**s, "note": None} for s in plain]


def test_a_kept_record_carries_the_note_on_the_system_message_of_every_step():
    plain, _ = build_step_samples(RECORD, TOK)
    samples, dropped = build_step_samples(RECORD, TOK, notes=ALWAYS)
    assert dropped == Dropped() and len(samples) == len(plain) == 4
    for (context, _), sample, before in zip(step_contexts(ZH_LONG), samples, plain, strict=True):
        noted = [{**context[0], "content": context[0]["content"] + "\n" + NOTE}, *context[1:]]
        assert sample["prompt"] == prompt_of(noted) and sample["note"] == NOTE
        assert sample["completion"] == before["completion"]  # the target is untouched
    assert RECORD["messages"][0] == {"role": "system", "content": "system prompt"}  # the record is unchanged


def test_the_note_of_a_step_reads_the_latest_user_message_of_its_context():
    question = {"name": "ask_clarification", "arguments": "{\"question\": \"哪个时间段?\"}"}
    clarify = {"role": "assistant", "content": "",
               "tool_calls": [{"id": "q1", "type": "function", "function": question}]}
    messages = [ZH_LONG[0], ZH_LONG[1], clarify, {"role": "tool", "tool_call_id": "q1", "content": "{}"},
                {"role": "user", "content": "最近一周的"}, *LONG[2:5]]
    samples, _ = build_step_samples({**RECORD, "messages": messages}, TOK, notes=ALWAYS)
    assert [s["note"] for s in samples] == [NOTE, None, None]
    assert NOTE in samples[0]["prompt"] and NOTE not in samples[1]["prompt"]


def test_the_note_dropout_is_a_hash_of_the_record_id():
    assert NOTE_DROPOUT == 0.5
    for rid in ("req-train-0001", "req-m6-0042", "其他"):
        value = int.from_bytes(hashlib.sha256(("glossary-note:" + rid).encode("utf-8")).digest()[:8], "big")
        assert keeps_note(rid, 0.5) is (value >= 2**63) and keeps_note(rid, 0.5) is keeps_note(rid, 0.5)
        assert keeps_note(rid, 0.0) and not keeps_note(rid, 1.0)
    share = sum(keeps_note(f"req-train-{n:04d}", 0.5) for n in range(2000)) / 2000
    assert 0.45 <= share <= 0.55
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            GlossaryNotes(GLOSSARY, dropout=bad)


def test_the_dropout_decides_per_record_and_a_dropped_record_renders_as_without_the_option():
    halved = GlossaryNotes(GLOSSARY)
    assert halved.dropout == 0.5
    (kept,), (dropped,) = ids_with(True), ids_with(False)
    noted, _ = build_step_samples({**RECORD, "id": kept}, TOK, notes=halved)
    assert [s["note"] for s in noted] == [NOTE] * 4
    plain, _ = build_step_samples({**RECORD, "id": dropped}, TOK)
    samples, _ = build_step_samples({**RECORD, "id": dropped}, TOK, notes=halved)
    assert samples == [{**s, "note": None} for s in plain]


def test_only_agent_history_records_take_the_note():
    t = transcript([step(1, accepted(ZH_LONG[2])), step(2, accepted(ZH_LONG[4]))], messages=ZH_LONG[:7])
    public = {**PUBLIC_SAMPLE, "messages": [{"role": "system", "content": "s"},
                                            {"role": "user", "content": "推测解码"},
                                            *PUBLIC_SAMPLE["messages"][1:]]}
    for record in (ZH_LONG, t, public, {"messages": ZH_LONG}):
        plain, dropped = build_step_samples(record, TOK)
        samples, same = build_step_samples(record, TOK, notes=ALWAYS)
        assert same == dropped and samples == [{**s, "note": None} for s in plain]


def test_an_agent_record_without_an_id_cannot_take_notes():
    record = {k: v for k, v in RECORD.items() if k != "id"}
    assert build_step_samples(record, TOK)[0]  # without the option nothing needs the id
    with pytest.raises(RecordError, match="id"):
        build_step_samples(record, TOK, notes=ALWAYS)


def test_the_notes_tokens_count_toward_max_length():
    plain, _ = build_step_samples(RECORD, TOK)
    noted, _ = build_step_samples(RECORD, TOK, notes=ALWAYS)
    assert all(n["tokens"] > p["tokens"] for n, p in zip(noted, plain, strict=True))
    limit = plain[0]["tokens"]  # the first step fits without its note, not with it
    assert noted[0]["tokens"] > limit
    kept, dropped = build_step_samples(RECORD, TOK, notes=ALWAYS, max_length=limit)
    assert kept == [] and dropped == Dropped(over_length=4)
    assert build_step_samples(RECORD, TOK, max_length=limit)[1] == Dropped(over_length=3)


def test_a_blocked_term_gives_no_note_in_training_as_at_inference(tmp_path):
    # 推测解码 is blocked: its span is taken and no note is due, so 解码 inside it is not noted either
    path = tmp_path / "zh-en.json"
    path.write_text(canonical_json({"推测解码": None, "解码": "Decoding"}) + "\n", encoding="utf-8")
    glossary = train.load_glossary(path)
    assert glossary.blocked == ("推测解码",)
    plain, dropped = build_step_samples(RECORD, TOK)
    samples, same = build_step_samples(RECORD, TOK, notes=GlossaryNotes(glossary, dropout=0.0))
    assert same == dropped and samples == [{**s, "note": None} for s in plain]
    for context, _ in step_contexts(ZH_LONG):
        assert noted_messages(context, glossary) == (context, None)  # what the inference wrapper sends
    deleted = GlossaryNotes(Glossary({"解码": "Decoding"}), dropout=0.0)  # without the blocked span
    exposed, _ = build_step_samples(RECORD, TOK, notes=deleted)
    assert [s["note"] for s in exposed] == ["术语对照:解码 = Decoding"] * 4


def test_family_samples_carry_the_note_in_the_prompt_inference_renders_for_the_same_history(tmp_path):
    family = loaded_fake_family(tmp_path / "model")
    index = sidecar(tmp_path / "teacher.jsonl", [(RECORD["id"], 1, ZH_LONG[4], REASONING)])
    plain, _ = build_family_samples(RECORD, family, name=RECORD["id"], reasoning=index)
    samples, dropped = build_family_samples(RECORD, family, name=RECORD["id"], reasoning=index, notes=ALWAYS)
    assert dropped == Dropped()
    assert [s["mode"] for s in samples] == ["no_think", "think", "no_think", "no_think"]
    for (context, _), sample, before in zip(step_contexts(ZH_LONG), samples, plain, strict=True):
        shown, note = noted_messages(context, GLOSSARY)  # what the inference wrapper sends
        mode = Mode.THINK if sample["mode"] == "think" else Mode.NO_THINK
        assert note == NOTE and sample["note"] == NOTE
        assert sample["prompt"] == family.prompt(family.to_template_messages(shown), MODEL_TOOLS, mode)
        assert sample["completion"] == before["completion"] and sample["key"] == before["key"]
        assert sample["tokens"] > before["tokens"]
    unnoted, _ = build_family_samples(RECORD, family, name=RECORD["id"], reasoning=index,
                                      notes=GlossaryNotes(GLOSSARY, dropout=1.0))
    assert unnoted == [{**s, "note": None} for s in plain]


# scripts/train.py


def agent_file(tmp_path: Path, ids: list[str]) -> tuple[Path, Path]:
    """A data file of noted-language agent records with these ids, and an overlap report listing it."""
    data = tmp_path / "agent.jsonl"
    data.write_text("".join(canonical_json({**RECORD, "id": rid}) + "\n" for rid in ids), encoding="utf-8")
    report = tmp_path / "overlap.json"
    report.write_text(json.dumps({"files": [{"path": str(data), "sha256": hashlib.sha256(
        data.read_bytes()).hexdigest(), "records": len(ids), "overlapping": 0, "records_overlapping": []}]}),
        encoding="utf-8")
    return data, report


def glossary_file(tmp_path: Path, terms: dict[str, str] | None = None) -> Path:
    path = tmp_path / "zh-en.json"
    path.write_text(canonical_json(terms if terms is not None else dict(GLOSSARY)) + "\n", encoding="utf-8")
    return path


def test_train_glossary_flags_default_off_and_check_the_dropout():
    args = train.build_parser().parse_args(["a.jsonl", "--out", "o"])
    assert args.glossary is None and args.glossary_dropout == NOTE_DROPOUT == 0.5
    args = train.build_parser().parse_args(["a.jsonl", "--glossary", "g.json", "--glossary-dropout", "0.25"])
    assert args.glossary == "g.json" and args.glossary_dropout == 0.25
    for bad in ("1.5", "-0.1", "x", "nan"):
        with pytest.raises(SystemExit):
            train.build_parser().parse_args(["a.jsonl", "--glossary-dropout", bad])


@pytest.mark.parametrize("flags", [
    ["--glossary-dropout", "0.2"],
    ["--glossary-dropout=0"],
    ["--glossary-dropout", "0.5"],  # the default value, given: still a dropout without a glossary
    ["--glossary-drop", "0.3"],  # argparse's abbreviation of the flag
])
def test_train_refuses_a_glossary_dropout_without_a_glossary(flags, capsys):
    # refused before anything is read: the data file, the model folder and the overlap report do not exist
    assert train.main(["missing.jsonl", "--out", "o", *flags]) == 2
    assert "--glossary-dropout needs --glossary" in capsys.readouterr().err


def test_train_refuses_an_empty_glossary_path(capsys):
    for flags in (["--glossary", ""], ["--glossary=", "--glossary-dropout", "0.2"]):
        assert train.main(["missing.jsonl", "--out", "o", *flags]) == 2
        assert "--glossary names no file" in capsys.readouterr().err


def test_train_records_the_glossary_flags_only_when_a_glossary_is_used():
    plain = train.build_parser().parse_args(["a.jsonl", "--out", "o"])
    recorded = train.recorded_args(plain)
    assert "glossary" not in recorded and "glossary_dropout" not in recorded
    assert recorded == {k: v for k, v in vars(plain).items() if k not in ("glossary", "glossary_dropout")}
    assert list(recorded) == [k for k in vars(plain) if k in recorded]  # the order the manifest writes
    used = train.build_parser().parse_args(["a.jsonl", "--out", "o", "--glossary", "g.json"])
    assert train.recorded_args(used) == vars(used)
    assert "glossary" not in train.data_summary([], Dropped(), [], None)


def test_train_dry_run_with_a_glossary_counts_the_noted_samples_and_records_the_file(tmp_path, capsys):
    kept = ids_with(True, 2)
    data, report = agent_file(tmp_path, [*kept, *ids_with(False, 1)])
    model, out = model_folder(tmp_path / "model"), tmp_path / "run"
    argv = [str(data), "--out", str(out), "--overlap-report", str(report), "--model", str(model), "--dry-run",
            "--max-length", "1000000"]
    assert train.main(argv, load_family=family_hook()) == 0
    printed = capsys.readouterr().out
    plain = json.loads(printed.splitlines()[-1].removeprefix("summary: "))
    assert "glossary" not in plain and "glossary:" not in printed
    path = glossary_file(tmp_path)
    assert train.main([*argv, "--glossary", str(path)], load_family=family_hook()) == 0
    printed = capsys.readouterr().out
    summary = json.loads(printed.splitlines()[-1].removeprefix("summary: "))
    assert summary["glossary"] == {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "terms": 2,
        "dropout": 0.5, "hash_rule": train.NOTE_HASH_RULE, "records_noted": 2, "samples_noted": 8}
    assert "glossary: 2 terms" in printed and "8 samples of 2 records carry a note" in printed
    assert summary["samples"] == plain["samples"] == 12
    assert summary["tokens"]["total"] > plain["tokens"]["total"]  # the notes' tokens
    assert {k: v for k, v in summary.items() if k not in ("glossary", "tokens")} == {
        k: v for k, v in plain.items() if k != "tokens"}
    assert train.main([*argv, "--glossary", str(path), "--glossary-dropout", "0"],
                      load_family=family_hook()) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[-1].removeprefix("summary: "))
    assert (summary["glossary"]["records_noted"], summary["glossary"]["samples_noted"]) == (3, 12)
    (tmp_path / "bad.json").write_text(json.dumps({"pruning": "Pruning"}), encoding="utf-8")
    assert train.main([*argv, "--glossary", str(tmp_path / "bad.json")], load_family=family_hook()) == 2
    assert "pruning" in capsys.readouterr().err
    assert train.main([*argv, "--glossary", str(tmp_path / "none.json")], load_family=family_hook()) == 2


def test_train_resume_compares_the_glossary_and_its_dropout(tmp_path, capsys):
    data, report = agent_file(tmp_path, ids_with(True, 1))
    model, out = model_folder(tmp_path / "model"), tmp_path / "run"
    path = glossary_file(tmp_path)
    base = [str(data), "--out", str(out), "--overlap-report", str(report), "--model", str(model)]
    argv = [*base, "--glossary", str(path)]
    args = train.build_parser().parse_args(argv)
    files = train.expand_data_files(args.data)
    family = loaded_fake_family(args.model)
    _, gate = train.overlap_gate(args.overlap_report, files, check=True)
    benchmark = {key: gate[key] for key in ("report", "report_sha256", "checked", "unchecked")}
    noted = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "dropout": 0.5}
    config = {**train.settings(args, family.spec), "family": family.record(),
              "args": train.recorded_args(args),
              "data": {"benchmark_overlap": benchmark, "reasoning": {"files": [], "ignored": False},
                       "sample_keys": None, "glossary": noted}}
    out.mkdir()
    manifest = train.run_manifest(config, files, git_dir=None, gpu_name=lambda: None)
    (out / "manifest.json").write_text(json.dumps(manifest, default=str), encoding="utf-8")
    assert train.resume_differences(train.build_parser().parse_args([*argv, "--resume"]), files, family) == []
    hook = family_hook()
    assert train.main([*argv, "--resume", "--glossary-dropout", "0.25"], load_family=hook) == 5
    assert "glossary.dropout: 0.5 -> 0.25" in capsys.readouterr().err
    assert train.main([*base, "--resume"], load_family=hook) == 5  # the glossary left out
    assert f"glossary.sha256: '{noted['sha256']}' -> None" in capsys.readouterr().err
    (tmp_path / "other").mkdir()
    other = glossary_file(tmp_path / "other", {"推测解码": "Speculative Decoding"})
    assert train.main([*base, "--resume", "--glossary", str(other)], load_family=hook) == 5
    assert "glossary.sha256" in capsys.readouterr().err
    # a run without a glossary is compared as before: a manifest without the key and no --glossary agree
    config["data"] = {k: v for k, v in config["data"].items() if k != "glossary"}
    manifest = train.run_manifest(config, files, git_dir=None, gpu_name=lambda: None)
    (out / "manifest.json").write_text(json.dumps(manifest, default=str), encoding="utf-8")
    assert train.resume_differences(train.build_parser().parse_args([*base, "--resume"]), files, family) == []
    assert train.main([*argv, "--resume"], load_family=hook) == 5
    assert "glossary.sha256: None ->" in capsys.readouterr().err

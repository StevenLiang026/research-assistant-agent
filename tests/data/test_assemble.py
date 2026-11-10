"""The synthetic training file (research_agent.data.assemble) and scripts/assemble_synth.py, over run folders
that the generator writes with the scripted world of tests/data/test_synth.py."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from support.fakes import ScriptedModel, call, table_check, turn
from test_synth import (
    GET,
    ID1,
    ID2,
    SEARCH,
    corpus_manifest,
    decide,
    load_script,
    normal_turns,
    owner,
    request,
    run,
)
from train.tokenizer_fake import FakeTokenizer

from research_agent.agent.prompts import PROMPT_HASH
from research_agent.canonical import canonical_json
from research_agent.data import assemble as asm
from research_agent.data.synth import TrajectoryValidator
from research_agent.eval.e2e import evidence_check
from research_agent.tools import schemas
from research_agent.train.samples import build_step_samples, load_records

script = load_script("assemble_synth")
train_script = load_script("train")
TOK = FakeTokenizer()


class OrderedTokenizer(FakeTokenizer):
    """Renders the tools in their own key order, as the Qwen3 template does, and merges the token pair that
    exists only in MODEL_TOOLS's order ('{"type": "function", ...'), as a BPE merge would: a tool block whose
    keys were sorted renders differently and counts more tokens."""

    MERGE = '{"type": "function", '

    def apply_chat_template(self, messages: list[dict], *, tools: list[dict] | None = None, **kwargs) -> str:
        text = super().apply_chat_template(messages, tools=None, **kwargs)
        if tools:
            block = json.dumps(tools, ensure_ascii=False)
            text = f"<|im_start|>system\n<tools>{block}</tools><|im_end|>\n" + text
        return text

    def __call__(self, text: str, **kwargs) -> dict:
        return {"input_ids": list(range(len(text.replace(self.MERGE, '{"type":"function",').split())))}


ORDERED = OrderedTokenizer()
ASK = call("ask_clarification", "c5", question="Which month?", missing=["date_range"])
BAD_GET = call("get_paper", "c9", arxiv_id="bad")
REJECTED = {"t-02"}


def repair_decide(event: str, times: int) -> str:
    return "repair" if event == "invalid_args" else decide(event, times)


def validator() -> TrajectoryValidator:
    return TrajectoryValidator(lambda transcript, req: ["made up"] if req["id"] in REJECTED else [])


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> tuple[Path, Path]:
    """A train run (accepted, rejected, clarification, duplicate, family leak, repaired) and a dev run."""
    root = tmp_path_factory.mktemp("synth")
    manifest = corpus_manifest(root)
    train_requests = [
        request("t-01", split="train", family="fam-a"),
        request("t-02", split="train", family="fam-b"),
        request("t-03", "clarify", "en", split="train", family="fam-c"),
        request("t-04", split="train", family="fam-d"),  # the same turns as t-01: a duplicate
        request("t-05", split="train", family="fam-dev"),  # its family is also a dev family
        request("t-06", split="train", family="fam-e"),  # its second step is accepted after a repair
        request("t-07", split="train", family="fam-f"),  # its answer quotes a think tag
    ]
    turns = (normal_turns() + normal_turns("没找到合适的论文。") + [turn(ASK)] + normal_turns()
             + normal_turns("已写好 2609.00001 的笔记。")
             + [turn(SEARCH), turn(BAD_GET), turn(GET), turn(content="只读了 2609.00001 的摘要。")]
             + normal_turns("论文的输出格式是 `<think>β</think><answer>α</answer>`。"))
    check = table_check({f"get_paper:{canonical_json({'arxiv_id': 'bad'})}": ["arxiv_id: bad"]})
    train_dir = root / "train"
    run(train_requests, ScriptedModel(turns), train_dir, manifest, validator=validator(),
        owner=owner(check=check, decide_fn=repair_decide), settings={"requests": "r.jsonl", "run": "train"})
    dev_dir = root / "dev"
    run([request("d-01", family="fam-dev")], ScriptedModel(normal_turns()), dev_dir, manifest,
        validator=validator(), settings={"requests": "r.jsonl", "run": "dev"})
    return train_dir, dev_dir


def test_only_accepted_train_rows_that_ended_answered_or_clarified_are_kept_and_every_drop_is_counted(runs):
    result = asm.assemble(list(runs))
    assert [r["id"] for r in result.records] == ["t-01", "t-03"]
    assert result.rows_seen == 8 and result.selected == 6
    assert result.dropped == {asm.OTHER_SPLIT: 1, asm.NOT_ACCEPTED: 1, asm.OTHER_STATUS: 0,
                              asm.AFTER_REPAIR: 1, asm.THINK_TAG: 1, asm.FAMILY_LEAK: 1,
                              asm.NO_VISIBLE_TEXT: 0, asm.DUPLICATE: 1}
    counts = result.counts()
    assert counts["kept"] == 2 and counts["by_scenario"] == {"normal": 1, "clarify": 1}
    assert counts["by_lang"] == {"en": 1, "zh": 1}
    assert counts["by_scenario_lang"] == {"normal": {"en": 0, "zh": 1}, "clarify": {"en": 1, "zh": 0}}


def test_a_think_tag_in_assistant_text_is_found_in_either_form():
    assert asm.has_think_tag([{"role": "assistant", "content": "x <think>y"}])
    assert asm.has_think_tag([{"role": "assistant", "content": "y</think> x"}])
    assert not asm.has_think_tag([{"role": "user", "content": "</think>"},
                                  {"role": "assistant", "content": "ok"}])
    assert not asm.has_think_tag([{"role": "assistant", "content": None}])


def test_drop_reason_checks_split_then_verdict_then_status():
    row = {"split": "train", "verdict": {"accepted": True, "problems": []}, "status": "answered"}
    assert asm.drop_reason(row) is None
    assert asm.drop_reason({**row, "status": "clarification"}) is None
    assert asm.drop_reason({**row, "split": "dev"}) == asm.OTHER_SPLIT
    assert asm.drop_reason({**row, "verdict": None}) == asm.NOT_ACCEPTED
    assert asm.drop_reason({**row, "verdict": {"accepted": False, "problems": ["x"]}}) == asm.NOT_ACCEPTED
    assert asm.drop_reason({**row, "status": "step_limit"}) == asm.OTHER_STATUS


def test_a_record_holds_the_transcript_history_the_model_tools_and_the_provenance(runs):
    record = asm.assemble(list(runs)).records[0]
    transcript = json.loads((runs[0] / "transcripts" / "t-01.json").read_text(encoding="utf-8"))
    assert record["messages"] == transcript["messages"] and record["tools"] == schemas.MODEL_TOOLS
    assert record["history"] == "agent"
    assert {k: record[k] for k in asm.PROVENANCE} == {
        "id": "t-01", "intent": "find_recent_papers", "family": "fam-a", "scenario": "normal", "lang": "zh",
        "run": "train", "prompt_hash": PROMPT_HASH, "schema_hash": schemas.SCHEMA_HASH}


def test_the_written_file_round_trips_through_the_trainers_loader_and_is_cut_like_the_transcript(runs,
                                                                                                  tmp_path):
    result = asm.assemble(list(runs))
    path = tmp_path / "synth.jsonl"
    asm.write_records(result.records, path)
    loaded = load_records(path)
    assert loaded == result.records
    transcript = json.loads((runs[0] / "transcripts" / "t-01.json").read_text(encoding="utf-8"))
    assert build_step_samples(loaded[0], TOK) == build_step_samples(transcript, TOK)


def test_samples_from_the_written_file_equal_the_transcripts_byte_for_byte_under_an_order_keeping_template(
        runs, tmp_path):
    result = asm.assemble(list(runs))
    path = tmp_path / "synth.jsonl"
    asm.write_records(result.records, path)
    for record in load_records(path):
        transcript_path = runs[0] / "transcripts" / f"{record['id']}.json"
        transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        from_file = build_step_samples(record, ORDERED)
        from_transcript = build_step_samples(transcript, ORDERED)
        assert [s["prompt"].encode("utf-8") for s in from_file[0]] == [
            s["prompt"].encode("utf-8") for s in from_transcript[0]]
        assert from_file == from_transcript


def test_the_manifest_token_statistics_equal_the_trainers_path_on_the_written_file(runs, tmp_path,
                                                                                    monkeypatch):
    monkeypatch.setattr(script, "load_tokenizer", lambda path: ORDERED)
    out, manifest_path = tmp_path / "synth.jsonl", tmp_path / "train-manifest.json"
    probe = build_step_samples(asm.assemble(list(runs)).records[0], ORDERED, max_length=10**9)[0]
    context = sorted(s["tokens"] for s in probe)[len(probe) // 2]  # some samples over it, some within
    monkeypatch.setattr(script, "CONTEXT_TOKENS", context)
    assert script.main([str(runs[0]), str(runs[1]), "--out", str(out), "--manifest", str(manifest_path),
                        "--tokenizer", "models/x"]) == 0
    tokens = json.loads(manifest_path.read_text(encoding="utf-8"))["tokens"]
    trained, dropped, per_file = train_script.assemble([out], ORDERED, context)
    every = [s["tokens"] for r in load_records(out)
             for s in build_step_samples(r, ORDERED, max_length=10**9)[0]]
    assert (tokens["within_context"], tokens["over_context"]) == (len(trained), dropped.over_length)
    assert dropped.over_length > 0 and tokens["context_tokens"] == context
    assert (tokens["min"], tokens["max"], tokens["samples"]) == (min(every), max(every), len(every))
    assert tokens["mean"] == round(sum(every) / len(every))
    assert tokens["trained"] == train_script.data_summary(trained, dropped, per_file, None)["tokens"]


def test_the_cli_writes_nothing_when_the_trainer_refuses_a_record(runs, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(script, "load_tokenizer", lambda path: ORDERED)
    monkeypatch.setattr(asm, "MODEL_TOOLS", schemas.MODEL_TOOLS[:1])  # tools the trainer refuses
    out, manifest_path = tmp_path / "synth.jsonl", tmp_path / "train-manifest.json"
    assert script.main([str(runs[0]), str(runs[1]), "--out", str(out), "--manifest", str(manifest_path),
                        "--tokenizer", "models/x"]) == 1
    assert "t-01" in capsys.readouterr().err
    assert not out.exists() and not manifest_path.exists() and list(tmp_path.iterdir()) == []


def test_dedup_compares_user_and_assistant_turns_after_normalize_text_and_tells_calls_apart():
    first = [{"role": "system", "content": "a"}, {"role": "user", "content": "Find  papers\n"},
             {"role": "assistant", "content": "", "tool_calls": [SEARCH.to_api()]},
             {"role": "tool", "tool_call_id": "c1", "content": "[1]"}]
    spaced = [{"role": "system", "content": "b"}, {"role": "user", "content": " Ｆｉｎｄ papers"},
              {"role": "assistant", "content": "", "tool_calls": [{**SEARCH.to_api(), "id": "other"}]},
              {"role": "tool", "tool_call_id": "other", "content": "[2]"}]
    assert asm.content_key(first) == asm.content_key(spaced)  # system prompt, ids and tool outputs ignored
    other_call = [*first[:2], {"role": "assistant", "content": "", "tool_calls": [GET.to_api()]}]
    assert asm.content_key(first) != asm.content_key(other_call)


def test_assembly_is_deterministic_whatever_the_order_of_the_runs(runs, tmp_path):
    a = asm.write_records(asm.assemble(list(runs)).records, tmp_path / "a.jsonl")
    b = asm.write_records(asm.assemble(list(reversed(runs))).records, tmp_path / "b.jsonl")
    assert a == b and (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert b"\r" not in (tmp_path / "a.jsonl").read_bytes()


def test_the_cli_writes_the_file_and_a_manifest_of_the_counts_sources_output_and_tokens(runs, tmp_path,
                                                                                         monkeypatch, capsys):
    monkeypatch.setattr(script, "load_tokenizer", lambda path: TOK)
    out, manifest_path = tmp_path / "raw" / "synth.jsonl", tmp_path / "synth" / "train-manifest.json"
    argv = [str(runs[0]), str(runs[1]), "--out", str(out), "--manifest", str(manifest_path),
            "--tokenizer", "models/x"]
    assert script.main(argv) == 0
    printed = capsys.readouterr().out
    assert "rows seen 8, selected 6, kept 2" in printed and "duplicate 1" in printed
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = out.read_bytes()
    assert manifest["output"] == {"path": out.as_posix(), "sha256": hashlib.sha256(data).hexdigest(),
                                  "bytes": len(data), "records": 2}
    assert manifest["counts"]["dropped"][asm.AFTER_REPAIR] == 1 and manifest["counts"]["kept"] == 2
    assert manifest["mix"]["counts"] == {"normal": 1, "clarify": 1}
    assert manifest["mix"]["shares"]["normal"] == 0.5
    train_source = manifest["sources"][0]
    assert train_source["name"] == "train" and train_source["schema_hash"] == schemas.SCHEMA_HASH
    assert train_source["prompt_hash"] == PROMPT_HASH and train_source["rows"] == 7
    run_manifest = (runs[0] / "manifest.json").read_bytes()
    assert train_source["manifest_sha256"] == hashlib.sha256(run_manifest).hexdigest()
    assert len(train_source["requests_sha256"] or "") in (0, 64)
    assert (manifest["schema_hash"], manifest["prompt_hash"]) == (schemas.SCHEMA_HASH, PROMPT_HASH)
    records = load_records(out)
    expected = sum(len(build_step_samples(r, TOK, max_length=10**9)[0]) for r in records)
    assert manifest["tokens"]["samples"] == expected and manifest["tokens"]["over_context"] == 0
    assert manifest["tokens"]["context_tokens"] == schemas.CONTEXT_TOKENS
    assert script.main(argv) == 0
    assert out.read_bytes() == data  # deterministic
    assert script.main(argv[:-2]) == 0
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["tokens"] is None  # without --tokenizer


def test_the_cli_exits_2_on_usage_errors_and_1_on_malformed_input(runs, tmp_path, capsys):
    out = ["--out", str(tmp_path / "o.jsonl"), "--manifest", str(tmp_path / "m.json")]
    assert script.main([str(tmp_path / "nowhere"), *out]) == 2
    assert "nowhere" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        script.main([])
    assert exc.value.code == 2
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "manifest.json").write_text("{}\n", encoding="utf-8")
    (broken / "index.jsonl").write_text("[1]\n", encoding="utf-8")
    assert script.main([str(broken), *out]) == 1
    assert "FAIL" in capsys.readouterr().err and not (tmp_path / "o.jsonl").exists()
    index_text = (runs[0] / "index.jsonl").read_text(encoding="utf-8")
    (broken / "index.jsonl").write_text(index_text, encoding="utf-8")
    assert script.main([str(broken), *out]) == 1  # the transcripts it names are missing
    assert "missing" in capsys.readouterr().err
    assert script.main([str(runs[0]), str(runs[0]), *out]) == 1  # every request id twice
    assert "appears in" in capsys.readouterr().err


def test_the_assemble_script_carries_a_usage_docstring_and_no_machine_path():
    source = (Path(script.__file__)).read_text(encoding="utf-8")
    assert script.__doc__ and "Exit" in script.__doc__ and "assemble_synth.py" in script.__doc__
    assert ":\\" not in source


QUOTED = '"Speculative decoding drafts tokens with a small model and verifies them."'  # ID1's abstract
PARAPHRASE = call("write_note", "c3", arxiv_id=ID1, mode="scaffold",
                  evidence="Drafts get checked by a larger model.")
FIGURE = call("write_note", "c3", arxiv_id=ID1, mode="scaffold",
              evidence="It drafts 2.8 tokens per step on average.")
GET_OTHER = call("get_paper", "c6", arxiv_id=ID2)


@pytest.fixture(scope="module")
def repair_run(tmp_path_factory) -> Path:
    """A train run whose notes paraphrase: one repaired, one whose step before the note shows only another
    paper (its trajectory is dropped), one whose final answer restates a figure only the old evidence held."""
    root = tmp_path_factory.mktemp("repair")
    manifest = corpus_manifest(root)
    requests = [request(f"r-0{n}", split="train", family=f"fam-r{n}") for n in (1, 2, 3)]
    turns = ([turn(SEARCH), turn(GET), turn(PARAPHRASE), turn(content="已为 2609.00001 写好笔记。")]
             + [turn(SEARCH), turn(GET), turn(GET_OTHER), turn(PARAPHRASE), turn(content="已写好笔记。")]
             + [turn(SEARCH), turn(GET), turn(FIGURE), turn(content="笔记写好了:平均每步起草 2.8 个词元。")])
    out = root / "train"
    run(requests, ScriptedModel(turns), out, manifest, validator=validator(),
        settings={"requests": "r.jsonl", "run": "train"})
    return out


def _note_evidence(record: dict) -> str:
    note = next(c for m in record["messages"] for c in m.get("tool_calls") or []
                if c["function"]["name"] == "write_note")
    return json.loads(note["function"]["arguments"])["evidence"]


def test_paraphrased_notes_are_repaired_a_note_without_visible_text_drops_its_trajectory_and_a_restated_figure_loses_the_answer(  # noqa: E501
        repair_run):
    result = asm.assemble([repair_run])
    assert [r["id"] for r in result.records] == ["r-01", "r-03"]
    assert result.dropped[asm.NO_VISIBLE_TEXT] == 1 and result.no_visible_text == ["r-02"]
    for record in result.records:
        assert _note_evidence(record) == QUOTED and evidence_check(record["messages"])["ok"]
    first, third = result.records
    assert first["messages"][-1] == {"role": "assistant", "content": "已为 2609.00001 写好笔记。"}
    assert third["messages"][-1]["role"] == "tool"
    assert result.evidence_repair() == {
        "scaffold_calls": 2, "calls_repaired": 2, "records_repaired": 2, "sentences": {"1": 2},
        "sources": {"abstract": 2}, "final_answers_removed": ["r-03"], "repaired": ["r-01", "r-03"],
        "no_visible_text": ["r-02"]}


def test_the_cli_writes_the_repaired_records_and_the_repair_block_and_the_trainer_cuts_them(
        repair_run, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(script, "load_tokenizer", lambda path: TOK)
    out, manifest_path = tmp_path / "synth.jsonl", tmp_path / "train-manifest.json"
    argv = [str(repair_run), "--out", str(out), "--manifest", str(manifest_path), "--tokenizer", "models/x"]
    assert script.main(argv) == 0
    assert ("evidence repair: 2 of 2 scaffold notes in 2 records repaired, 1 final answers removed, "
            "1 records dropped without visible text") in capsys.readouterr().out
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["evidence_repair"] == asm.assemble([repair_run]).evidence_repair()
    assert manifest["counts"]["dropped"][asm.NO_VISIBLE_TEXT] == 1 and manifest["counts"]["kept"] == 2
    assert "evidence_repair" in manifest["selection"]
    records = load_records(out)
    assert [len(build_step_samples(r, TOK, max_length=10**9)[0]) for r in records] == [4, 3]
    assert [_note_evidence(r) for r in records] == [QUOTED, QUOTED]

import dataclasses
import json
from pathlib import Path

import pytest
from support.fakes import HITS, PAPER, SAMPLE_MESSAGES, call
from tokenizer_fake import EOS, FakeTokenizer

from research_agent.agent.context import build_context, step_contexts
from research_agent.agent.prompts import PROMPT_HASH, PROMPT_VERSION
from research_agent.agent.types import ANSWERED, Transcript, api_to_hf
from research_agent.canonical import canonical_json
from research_agent.tools import schemas
from research_agent.train.manifest import count_rows
from research_agent.train.samples import (
    Dropped,
    RecordError,
    accepted_after_repair,
    build_family_samples,
    build_step_samples,
    load_records,
    render_sample,
    step_pairs,
)

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "Qwen3-1.7B"
TOK = FakeTokenizer()
NOTE = {"note_id": "2609.01234-adaptive-draft", "arxiv_id": "2609.01234", "path": "notes/2609.01234.md"}
# SAMPLE_MESSAGES plus a third step (write_note) and the closing answer: four assistant steps.
LONG = SAMPLE_MESSAGES + [
    {"role": "assistant", "content": "",
     "tool_calls": [call("write_note", "c4", arxiv_id="2609.01234").to_api()]},
    {"role": "tool", "tool_call_id": "c4", "content": canonical_json(NOTE)},
    {"role": "assistant", "content": "已为 2609.01234 写好笔记。"},
]
WEATHER_TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Weather of a city.",
                                                 "parameters": {"type": "object", "properties": {
                                                     "city": {"type": "string"}}, "required": ["city"]}}}
PUBLIC_SAMPLE = {
    "messages": [
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
            "name": "get_weather", "arguments": {"city": "Paris"}}}]},
        {"role": "tool", "content": '{"temperature_c": 18, "sky": "overcast"}'},
        {"role": "assistant", "content": "It is 18 degrees and overcast in Paris."},
    ],
    "tools": [WEATHER_TOOL],
}


def prompt_of(context: list[dict], tools=schemas.MODEL_TOOLS) -> str:
    return TOK.apply_chat_template(api_to_hf(context), tools=tools, add_generation_prompt=True)


def completion_of(message: dict) -> str:
    rendered = TOK.apply_chat_template(api_to_hf([message]))  # "<|im_start|>assistant\n...<|im_end|>\n"
    return rendered.removeprefix("<|im_start|>assistant\n").removesuffix("\n")


def accepted(message: dict) -> dict:
    return {"model": "fake", "content": message.get("content") or "",
            "tool_calls": list(message.get("tool_calls") or []), "issues": [], "action": None}


def rejected(action: str) -> dict:
    return {"model": "fake", "content": "", "tool_calls": [call("get_paper", "c9", arxiv_id="bad").to_api()],
            "issues": [{"call_id": "c9", "event": "invalid_args", "messages": ["arxiv_id: bad"]}],
            "action": action}


def step(index: int, *attempts: dict) -> dict:
    return {"index": index, "model": "fake", "attempts": list(attempts), "executed": [], "decisions": []}


def transcript(steps: list[dict], messages: list[dict] = SAMPLE_MESSAGES) -> Transcript:
    return Transcript(request=messages[1]["content"], today="2026-10-04", status=ANSWERED, final_text=None,
                      messages=messages, steps=steps, schema_hash=schemas.SCHEMA_HASH,
                      prompt_version=PROMPT_VERSION, prompt_hash=PROMPT_HASH)


def test_one_sample_per_assistant_step_with_the_context_the_model_was_shown():
    samples, dropped = build_step_samples(LONG, TOK)
    assert dropped == Dropped() and len(samples) == 4
    targets = [m for m in LONG if m["role"] == "assistant"]
    for (context, target), sample, expected in zip(step_contexts(LONG), samples, targets, strict=True):
        assert sample["prompt"] == prompt_of(context)
        assert sample["completion"] == completion_of(expected)
        assert sample["completion"].endswith(EOS) and not sample["completion"].endswith("\n")
        full = TOK.apply_chat_template(api_to_hf(context + [target]), tools=schemas.MODEL_TOOLS)
        assert full == sample["prompt"] + sample["completion"] + "\n"
    assert samples[0]["prompt"] == prompt_of(LONG[:2])  # no step yet: system prompt and request only
    assert samples[-1]["completion"] == "已为 2609.01234 写好笔记。" + EOS


def test_older_tool_outputs_are_compressed_in_later_steps_only():
    samples, _ = build_step_samples(LONG, TOK)
    snippet = HITS[0]["snippet"]
    assert snippet in samples[1]["prompt"]  # the search is the latest step there: kept whole
    assert snippet not in samples[2]["prompt"] and HITS[0]["title"] in samples[2]["prompt"]
    assert PAPER["abstract"] in samples[2]["prompt"]  # the get_paper step is the latest step there
    assert PAPER["abstract"] not in samples[3]["prompt"]
    assert samples[2]["prompt"] == prompt_of(build_context(SAMPLE_MESSAGES))


def test_tokens_are_counted_and_over_length_samples_are_dropped_not_truncated():
    samples, _ = build_step_samples(SAMPLE_MESSAGES, TOK)
    first, second = samples
    assert first["tokens"] == len((first["prompt"] + first["completion"]).split()) < second["tokens"]
    kept, dropped = build_step_samples(SAMPLE_MESSAGES, TOK, max_length=first["tokens"])
    assert kept == [first] and dropped == Dropped(over_length=1, after_repair=0)
    none, dropped = build_step_samples(SAMPLE_MESSAGES, TOK, max_length=first["tokens"] - 1)
    assert none == [] and dropped == Dropped(over_length=2)


def test_a_step_accepted_after_a_repair_is_skipped_but_a_handoff_is_kept():
    plain, _ = build_step_samples(SAMPLE_MESSAGES, TOK)
    repaired = transcript([step(1, rejected("repair"), accepted(SAMPLE_MESSAGES[2])),
                           step(2, accepted(SAMPLE_MESSAGES[4]))])
    samples, dropped = build_step_samples(repaired, TOK)
    assert samples == [plain[1]] and dropped == Dropped(over_length=0, after_repair=1)
    handed_off = transcript([step(1, rejected("handoff"), accepted(SAMPLE_MESSAGES[2])),
                             step(2, accepted(SAMPLE_MESSAGES[4]))])
    assert build_step_samples(handed_off, TOK) == (plain, Dropped())
    # repair, then handoff, then accepted: the handoff model saw the original context
    both = transcript([step(1, rejected("repair"), rejected("handoff"), accepted(SAMPLE_MESSAGES[2])),
                       step(2, accepted(SAMPLE_MESSAGES[4]))])
    assert build_step_samples(both, TOK) == (plain, Dropped())
    assert accepted_after_repair(step(1, rejected("repair"), accepted(SAMPLE_MESSAGES[2])))
    assert not accepted_after_repair(step(1, accepted(SAMPLE_MESSAGES[2])))
    assert not accepted_after_repair({"attempts": []})


def test_a_last_step_without_an_accepted_message_is_fine_but_more_messages_than_steps_is_not():
    plain, _ = build_step_samples(SAMPLE_MESSAGES, TOK)
    three_steps = transcript([step(1, accepted(SAMPLE_MESSAGES[2])), step(2, accepted(SAMPLE_MESSAGES[4])),
                              step(3, rejected("stop"))])
    assert build_step_samples(three_steps, TOK) == (plain, Dropped())
    with pytest.raises(ValueError, match="assistant messages"):
        build_step_samples(transcript([step(1, accepted(SAMPLE_MESSAGES[2]))]), TOK)


def test_a_transcript_dict_is_accepted_and_another_schema_is_refused_unless_tools_are_given():
    plain, _ = build_step_samples(SAMPLE_MESSAGES, TOK)
    t = transcript([step(1, accepted(SAMPLE_MESSAGES[2])), step(2, accepted(SAMPLE_MESSAGES[4]))])
    assert build_step_samples(json.loads(t.to_json()), TOK) == (plain, Dropped())
    old = dataclasses.replace(t, schema_hash="0" * 16)
    with pytest.raises(ValueError, match="tool schema"):
        build_step_samples(old, TOK)
    samples, _ = build_step_samples(old, TOK, tools=list(schemas.MODEL_TOOLS))
    assert samples == plain


def test_public_samples_keep_their_own_tools_and_their_whole_history():
    samples, dropped = build_step_samples(PUBLIC_SAMPLE, TOK)
    assert dropped == Dropped() and len(samples) == 2
    assert "get_weather" in samples[0]["prompt"] and "search_arxiv" not in samples[0]["prompt"]
    assert samples[0]["completion"] == completion_of(PUBLIC_SAMPLE["messages"][1])
    assert '{"temperature_c": 18, "sky": "overcast"}' in samples[1]["prompt"]  # not compressed
    assert samples[1]["completion"] == "It is 18 degrees and overcast in Paris." + EOS
    as_string = {**PUBLIC_SAMPLE, "tools": json.dumps([WEATHER_TOOL])}
    assert build_step_samples(as_string, TOK) == (samples, Dropped())
    without_tools = {"messages": PUBLIC_SAMPLE["messages"]}
    fallback, _ = build_step_samples(without_tools, TOK, tools=[WEATHER_TOOL])
    assert fallback == samples


def test_a_template_that_breaks_the_prefix_property_is_rejected():
    thinking = FakeTokenizer(generation_prompt="<|im_start|>assistant\n<think>\n\n</think>\n\n")
    with pytest.raises(ValueError, match="prefix"):
        build_step_samples(SAMPLE_MESSAGES, thinking)
    with pytest.raises(ValueError, match="prefix"):
        render_sample(thinking, SAMPLE_MESSAGES[:2], SAMPLE_MESSAGES[2], tools=None)


def test_dropped_counts_add_up():
    total = Dropped(1, 2) + Dropped(3, 4)
    assert total == Dropped(over_length=4, after_repair=6) and total.total == 10
    assert dataclasses.asdict(Dropped()) == {"over_length": 0, "after_repair": 0}


def test_load_records_reads_jsonl_lines_and_single_json_files(tmp_path):
    t = transcript([step(1, accepted(SAMPLE_MESSAGES[2])), step(2, accepted(SAMPLE_MESSAGES[4]))])
    lines = [t.to_json().replace("\n", ""), json.dumps(PUBLIC_SAMPLE),
             json.dumps({"messages": SAMPLE_MESSAGES}), json.dumps(SAMPLE_MESSAGES), ""]
    jsonl = tmp_path / "mixed.jsonl"
    jsonl.write_text("\n".join(lines), encoding="utf-8")
    records = load_records(jsonl)
    assert isinstance(records[0], Transcript) and records[0].messages == SAMPLE_MESSAGES
    assert records[1] == PUBLIC_SAMPLE and records[2] == SAMPLE_MESSAGES and records[3] == SAMPLE_MESSAGES
    assert len(records) == 4
    single = tmp_path / "one.json"
    single.write_text(t.to_json(), encoding="utf-8")
    assert load_records(single)[0] == t
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"text": "not a record"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="record"):
        load_records(bad)


def test_load_records_counts_like_the_manifest_and_reads_json_lists_of_records(tmp_path):
    t = transcript([step(1, accepted(SAMPLE_MESSAGES[2])), step(2, accepted(SAMPLE_MESSAGES[4]))])
    lines = [t.to_json().replace("\n", ""), json.dumps(PUBLIC_SAMPLE),
             json.dumps({"messages": SAMPLE_MESSAGES}), json.dumps(SAMPLE_MESSAGES)]
    jsonl = tmp_path / "mixed.jsonl"
    jsonl.write_bytes(("\r\n".join(lines) + "\r\n\r\n").encode("utf-8"))  # CRLF line ends and a blank line
    listed = tmp_path / "list.json"
    listed.write_text(json.dumps([PUBLIC_SAMPLE, {"messages": SAMPLE_MESSAGES}, json.loads(t.to_json())]),
                      encoding="utf-8")
    single = tmp_path / "one.json"
    single.write_text(json.dumps(PUBLIC_SAMPLE), encoding="utf-8")
    none = tmp_path / "none.json"
    none.write_text("[]", encoding="utf-8")
    for path, rows in [(jsonl, 4), (listed, 3), (single, 1), (none, 0)]:
        assert len(load_records(path)) == count_rows(path) == rows, path.name
    records = load_records(listed)
    assert records[0] == PUBLIC_SAMPLE and records[1] == SAMPLE_MESSAGES and records[2] == t
    assert [len(build_step_samples(r, TOK)[0]) for r in records] == [2, 2, 2]
    history = tmp_path / "history.json"  # a bare history as a whole .json file: one record, as on a line
    history.write_text(json.dumps(SAMPLE_MESSAGES), encoding="utf-8")
    assert load_records(history) == [SAMPLE_MESSAGES]  # count_rows (manifest.py) counts its messages instead


def test_load_records_names_the_file_and_the_line_of_a_malformed_record(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"messages": SAMPLE_MESSAGES}) + "\n{oops\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl: line 2 is not JSON"):
        load_records(bad)
    bad.write_text(json.dumps({"messages": SAMPLE_MESSAGES}) + "\n" + json.dumps([PUBLIC_SAMPLE]) + "\n",
                   encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl: line 2 .*item 0"):  # one record per line, not a list
        load_records(bad)
    bad.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_records(bad)
    bad.write_text(json.dumps({"tools": []}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):  # a public sample needs its messages
        load_records(bad)
    mixed = tmp_path / "mixed.json"
    mixed.write_text(json.dumps([PUBLIC_SAMPLE, {"role": "user", "content": "x"}]), encoding="utf-8")
    with pytest.raises(ValueError, match=r"mixed\.json: .*item 1 .*item 0"):
        load_records(mixed)
    broken = tmp_path / "broken.json"
    broken.write_text("{oops", encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken\.json: .*not JSON"):
        load_records(broken)


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="local model files are not present")
def test_real_tokenizer_round_trip_matches_the_inference_renderer():
    from transformers import AutoTokenizer

    from research_agent.llm.local import hf_renderer

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    samples, dropped = build_step_samples(SAMPLE_MESSAGES, tok)
    assert dropped == Dropped() and len(samples) == 2
    render = hf_renderer(MODEL_DIR)
    for (context, target), sample in zip(step_contexts(SAMPLE_MESSAGES), samples, strict=True):
        assert sample["prompt"] == render(api_to_hf(context), schemas.MODEL_TOOLS)  # training == inference
        assert sample["prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert sample["completion"].startswith("<tool_call>") and sample["completion"].endswith("<|im_end|>")
        full = tok.apply_chat_template(api_to_hf(context + [target]), tools=schemas.MODEL_TOOLS,
                                       tokenize=False, enable_thinking=False)
        assert full == sample["prompt"] + sample["completion"] + "\n"
        assert 900 < sample["tokens"] <= schemas.CONTEXT_TOKENS  # the tool schemas alone are about 900
    assert samples[0]["completion"] == (
        '<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "speculative decoding", '
        '"date_from": "2026-09-01"}}\n</tool_call><|im_end|>')
    assert samples[1]["completion"].count("<tool_call>") == 2  # the two parallel get_paper calls


def test_load_records_keeps_a_jsonl_record_holding_a_line_separator_whole(tmp_path):
    record = {"messages": [{"role": "user", "content": "line one line two"}], "tools": []}
    jsonl = tmp_path / "sep.jsonl"
    jsonl.write_text(canonical_json(record) + "\n\n" + canonical_json(PUBLIC_SAMPLE) + "\n", encoding="utf-8")
    assert load_records(jsonl) == [record, PUBLIC_SAMPLE] and count_rows(jsonl) == 2


def test_an_agent_history_record_with_tools_is_cut_like_an_agent_history_not_like_a_public_sample(tmp_path):
    record = {"messages": LONG, "tools": schemas.MODEL_TOOLS, "history": "agent", "id": "req-train-0001",
              "schema_hash": schemas.SCHEMA_HASH, "prompt_hash": PROMPT_HASH}
    samples, dropped = build_step_samples(record, TOK)
    assert (samples, dropped) == build_step_samples(LONG, TOK)  # older tool outputs compressed
    assert PAPER["abstract"] not in samples[3]["prompt"]
    path = tmp_path / "synth.jsonl"
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
    assert build_step_samples(load_records(path)[0], TOK) == (samples, dropped)
    with pytest.raises(ValueError, match="tool schema"):
        build_step_samples({**record, "schema_hash": "0" * 16}, TOK)


AGENT_RECORD = {"messages": LONG, "tools": schemas.MODEL_TOOLS, "history": "agent", "id": "req-train-0001",
                "schema_hash": schemas.SCHEMA_HASH, "prompt_hash": PROMPT_HASH}


def test_an_agent_history_record_without_a_schema_hash_is_refused_naming_the_record():
    record = {k: v for k, v in AGENT_RECORD.items() if k != "schema_hash"}
    with pytest.raises(ValueError, match="req-train-0001.*schema_hash"):
        build_step_samples(record, TOK)


def test_an_agent_history_record_whose_tools_differ_from_the_model_tools_is_refused_naming_the_record():
    altered = [dict(tool) for tool in schemas.MODEL_TOOLS]
    altered[0] = {**altered[0], "function": {**altered[0]["function"], "description": "Something else."}}
    for tools in (altered, schemas.MODEL_TOOLS[1:], [], json.dumps(altered)):
        with pytest.raises(ValueError, match="req-train-0001.*MODEL_TOOLS"):
            build_step_samples({**AGENT_RECORD, "tools": tools}, TOK)
    with pytest.raises(ValueError, match="req-train-0001"):
        build_step_samples({k: v for k, v in AGENT_RECORD.items() if k != "tools"}, TOK)


def test_an_agent_history_record_is_rendered_with_the_model_tools_themselves_whatever_its_key_order():
    seen: list = []

    class Spy(FakeTokenizer):
        def apply_chat_template(self, messages, *, tools=None, **kwargs):
            seen.append(tools)
            return super().apply_chat_template(messages, tools=tools, **kwargs)

    reordered = json.loads(canonical_json(schemas.MODEL_TOOLS))  # sorted keys, as the assembler writes them
    build_step_samples({**AGENT_RECORD, "tools": reordered}, Spy(), tools=None)
    assert seen and all(tools is schemas.MODEL_TOOLS for tools in seen)


def test_an_agent_history_record_without_a_prompt_hash_is_refused_naming_the_record():
    record = {k: v for k, v in AGENT_RECORD.items() if k != "prompt_hash"}
    with pytest.raises(RecordError, match="req-train-0001.*prompt_hash"):
        build_step_samples(record, TOK)


def test_an_agent_history_record_under_another_prompt_is_refused_naming_the_record():
    with pytest.raises(RecordError, match=f"req-train-0001.*{'0' * 16}.*{PROMPT_HASH}"):
        build_step_samples({**AGENT_RECORD, "prompt_hash": "0" * 16}, TOK)
    assert build_step_samples(AGENT_RECORD, TOK)[0]  # the current prompt hash passes


# ---- context turns (docs/m7-本地提升计划.md WP6): the leading turns of a cut copy are context only ------


def test_a_record_with_context_turns_trains_only_the_turns_from_there_on_and_keeps_their_k(tmp_path):
    from family_stub import loaded_fake_family

    full, _ = build_step_samples(AGENT_RECORD, TOK)
    record = {**AGENT_RECORD, "context_turns": 2}
    samples, dropped = build_step_samples(record, TOK)
    assert dropped == Dropped() and len(full) == 4 and samples == full[2:]
    steps, _, _ = step_pairs(record, schemas.MODEL_TOOLS)
    assert [k for k, _, _ in steps] == [2, 3]  # k still counts every assistant message (the sidecars' key)
    family = loaded_fake_family(tmp_path / "model")
    keyed, _ = build_family_samples(record, family, name="req-train-0001")
    assert [s["key"] for s in keyed] == [("req-train-0001", 2), ("req-train-0001", 3)]
    assert build_step_samples({**AGENT_RECORD, "context_turns": 0}, TOK) == (full, Dropped())
    public, _ = build_step_samples({**PUBLIC_SAMPLE, "context_turns": 0}, TOK)
    assert public == build_step_samples(PUBLIC_SAMPLE, TOK)[0]  # a public sample with the field is cut alike


def test_load_records_reads_context_turns_below_the_number_of_assistant_turns(tmp_path):
    path = tmp_path / "ft4.jsonl"
    path.write_text(json.dumps(AGENT_RECORD) + "\n" + json.dumps({**AGENT_RECORD, "context_turns": 3}) + "\n",
                    encoding="utf-8")
    plain, cut = load_records(path)
    assert "context_turns" not in plain and cut["context_turns"] == 3
    assert len(build_step_samples(plain, TOK)[0]) == 4 and len(build_step_samples(cut, TOK)[0]) == 1


@pytest.mark.parametrize("value", [4, 9, -1, True, False, "1", 1.0, None])
def test_a_context_turns_that_is_no_count_below_the_assistant_turns_is_refused(tmp_path, value):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(AGENT_RECORD) + "\n" + json.dumps({**AGENT_RECORD, "context_turns": value})
                    + "\n", encoding="utf-8")
    with pytest.raises(RecordError, match=r"bad\.jsonl: line 2: .*context_turns"):
        load_records(path)
    with pytest.raises(RecordError, match="req-train-0001.*context_turns"):
        build_step_samples({**AGENT_RECORD, "context_turns": value}, TOK)


def test_context_turns_is_refused_on_a_record_that_would_drop_it(tmp_path):
    """An object with messages but no tools is read as its bare history: the field would be lost and the
    context turns trained, so it is refused instead."""
    path = tmp_path / "bare.jsonl"
    path.write_text(json.dumps({"messages": LONG, "context_turns": 1}) + "\n", encoding="utf-8")
    with pytest.raises(RecordError, match=r"bare\.jsonl: line 1: .*context_turns"):
        load_records(path)

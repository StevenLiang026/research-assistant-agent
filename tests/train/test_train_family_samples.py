"""The family path of research_agent.train.samples (docs/m7-家族层设计.md §5): pre-tokenized samples rendered
through a LoadedFamily, THINK or NO_THINK per sample from a reasoning sidecar, the fallback and the drop
rules, the sample keys and the refusal kinds the dry run counts. Offline, over tests/support/family_fakes'
tokenizer (tests/train/family_stub.py); the real templates are in test_regression_anchor.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from family_stub import loaded_fake_family
from support.fakes import SAMPLE_MESSAGES
from support.family_fakes import THINK_CLOSE, THINK_OPEN, TURN_END, FakeFamilyTokenizer, token_ids
from test_train_samples import AGENT_RECORD, LONG, PUBLIC_SAMPLE, accepted, rejected, step, transcript

from research_agent.agent.context import step_contexts
from research_agent.data.reasoning import ReasoningError, ReasoningIndex, sidecar_row, write_sidecar
from research_agent.family.core import FamilyError, Mode
from research_agent.tools.schemas import MODEL_TOOLS
from research_agent.train.samples import (
    Dropped,
    build_family_samples,
    build_step_samples,
    reasoning_budget,
    record_name,
    refusal_kind,
    render_family_sample,
)

REASONING = "\nThe search found the paper, so I fetch both abstracts.\n"


def sidecar(path: Path, rows: list[tuple[str, int, dict, str]]) -> ReasoningIndex:
    """A sidecar file of teacher rows (record id, k, the training answer, the reasoning) and its index."""
    write_sidecar([sidecar_row(rid, k, answer, source="teacher", reasoning=text, model="deepseek-flash",
                               cache_key="a" * 64, context_equal=True)
                   for rid, k, answer, text in rows], path)
    return ReasoningIndex([path])


def test_family_samples_are_the_render_samples_pre_tokenized_with_the_completion_start(tmp_path):
    family = loaded_fake_family(tmp_path / "model")
    samples, dropped = build_family_samples(LONG, family, name="h")
    assert dropped == Dropped() and len(samples) == 4
    old, _ = build_step_samples(LONG, family.tokenizer)  # today's path over the same tokenizer
    for k, (sample, before) in enumerate(zip(samples, old, strict=True)):
        assert (sample["prompt"], sample["completion"], sample["tokens"]) == (
            before["prompt"], before["completion"], before["tokens"])
        assert sample["input_ids"] == token_ids(before["prompt"] + before["completion"])
        assert sample["completion_start"] == len(token_ids(before["prompt"]))
        assert sample["tokens"] == len(sample["input_ids"]) and sample["input_ids"][-1] == family.turn_end_id
        assert (sample["key"], sample["mode"], sample["think_fallback"]) == (("h", k), "no_think", False)
        assert sample["reasoning_tokens"] is None
        assert sample["prompt"].endswith(THINK_OPEN + THINK_CLOSE)  # the empty block sits in the prompt


def test_a_turn_with_reasoning_is_a_think_sample_and_the_others_keep_the_empty_block_in_the_prompt(tmp_path):
    family = loaded_fake_family(tmp_path / "model")
    index = sidecar(tmp_path / "teacher.jsonl", [("h", 1, LONG[4], REASONING)])
    samples, dropped = build_family_samples(LONG, family, name="h", reasoning=index)
    assert dropped == Dropped()
    assert [s["mode"] for s in samples] == ["no_think", "think", "no_think", "no_think"]
    plain, _ = build_family_samples(LONG, family, name="h")
    think, context = samples[1], step_contexts(LONG)[1][0]
    hf = family.to_template_messages(context)
    prompt = family.prompt(hf, MODEL_TOOLS, Mode.THINK)
    assert think["prompt"] == prompt and prompt.endswith(THINK_OPEN) and not prompt.endswith(THINK_CLOSE)
    assert think["completion"] == REASONING.strip("\n") + THINK_CLOSE + plain[1]["completion"]
    assert think["input_ids"] == token_ids(prompt + think["completion"])
    assert think["completion_start"] == len(token_ids(prompt))
    assert think["tokens"] == len(think["input_ids"])
    assert think["reasoning_tokens"] == len(token_ids(REASONING.strip("\n")))
    assert (think["key"], think["think_fallback"]) == (("h", 1), False)
    # the construction is the template's own render of the target carrying its reasoning
    (target,) = family.to_template_messages([LONG[4]])
    full = family.full(hf + [{**target, "reasoning_content": REASONING}], MODEL_TOOLS, Mode.THINK)
    assert full == think["prompt"] + think["completion"] + "\n"
    assert [s for i, s in enumerate(samples) if i != 1] == [s for i, s in enumerate(plain) if i != 1]


def test_a_think_sample_over_max_length_falls_back_to_no_think_and_a_no_think_one_over_it_is_dropped(
        tmp_path):
    family = loaded_fake_family(tmp_path / "model")
    history = SAMPLE_MESSAGES  # two steps, the second the longer one, which gets the reasoning
    index = sidecar(tmp_path / "teacher.jsonl", [("h", 1, history[4], REASONING)])
    plain, _ = build_family_samples(history, family, name="h")
    think, _ = build_family_samples(history, family, name="h", reasoning=index)
    assert think[1]["tokens"] > plain[1]["tokens"] > plain[0]["tokens"]
    limit = think[1]["tokens"] - 1  # the THINK sample does not fit, its NO_THINK rendering does
    samples, dropped = build_family_samples(history, family, name="h", reasoning=index, max_length=limit)
    assert dropped == Dropped() and len(samples) == 2
    assert samples[1]["mode"] == "no_think" and samples[1]["think_fallback"] is True
    assert samples[1]["reasoning_tokens"] is None
    assert {**samples[1], "think_fallback": False} == plain[1]  # the NO_THINK sample, kept in its place
    samples, dropped = build_family_samples(history, family, name="h", reasoning=index,
                                            max_length=plain[1]["tokens"] - 1)
    assert dropped == Dropped(over_length=1) and [s["key"] for s in samples] == [("h", 0)]


def test_reasoning_is_not_consulted_for_a_family_that_never_thinks(tmp_path):
    family = loaded_fake_family(tmp_path / "model", "none")
    stale = sidecar(tmp_path / "teacher.jsonl", [("h", 1, LONG[2], REASONING)])  # written for another answer
    samples, dropped = build_family_samples(LONG, family, name="h", reasoning=stale)
    assert dropped == Dropped() and {s["mode"] for s in samples} == {"no_think"}
    assert all(not s["think_fallback"] and s["prompt"].endswith("<|im_start|>assistant\n") for s in samples)


def test_a_stale_sidecar_row_stops_the_build(tmp_path):
    family = loaded_fake_family(tmp_path / "model")
    stale = sidecar(tmp_path / "teacher.jsonl", [("h", 1, LONG[2], REASONING)])
    with pytest.raises(ReasoningError, match="stale.*h.*turn 1"):
        build_family_samples(LONG, family, name="h", reasoning=stale)


def test_the_turn_ordinal_counts_every_assistant_message_so_a_skipped_step_keeps_its_k(tmp_path):
    family = loaded_fake_family(tmp_path / "model")
    repaired = transcript([step(1, rejected("repair"), accepted(SAMPLE_MESSAGES[2])),
                           step(2, accepted(SAMPLE_MESSAGES[4]))])
    index = sidecar(tmp_path / "teacher.jsonl", [("t", 1, SAMPLE_MESSAGES[4], REASONING)])
    samples, dropped = build_family_samples(repaired, family, name="t", reasoning=index)
    assert dropped == Dropped(after_repair=1)
    assert [(s["key"], s["mode"]) for s in samples] == [(("t", 1), "think")]
    public, _ = build_family_samples(PUBLIC_SAMPLE, family, name="glaive_v2:3")
    assert [s["key"] for s in public] == [("glaive_v2:3", 0), ("glaive_v2:3", 1)]
    assert "get_weather" in public[0]["prompt"]  # a public sample keeps its own tools


def test_the_record_name_is_its_id_or_the_file_stem_and_index():
    assert record_name(AGENT_RECORD, Path("data/raw/synth/synth-m6.jsonl"), 7) == "req-train-0001"
    assert record_name(PUBLIC_SAMPLE, Path("data/raw/public/glaive_v2.jsonl"), 17) == "glaive_v2:17"
    assert record_name(SAMPLE_MESSAGES, Path("a/b/histories.json"), 0) == "histories:0"
    t = transcript([step(1, accepted(SAMPLE_MESSAGES[2])), step(2, accepted(SAMPLE_MESSAGES[4]))])
    assert record_name(t, Path("runs/x/transcripts.jsonl"), 3) == "transcripts:3"


class RefusingTokenizer(FakeFamilyTokenizer):
    """Its template refuses a message whose content holds "[refuse]", as a template refuses a shape."""

    def apply_chat_template(self, messages, **kwargs):
        if any("[refuse]" in (m.get("content") or "") for m in messages):
            raise TypeError("the template cannot render this message")
        return super().apply_chat_template(messages, **kwargs)


def test_refused_samples_are_raised_or_counted_by_kind_when_a_sink_is_given(tmp_path):
    family = loaded_fake_family(tmp_path / "model", tokenizer=RefusingTokenizer("hybrid", bos="token"))
    rows = [("h", 0, LONG[2], "<s> a BOS in the reasoning"), ("h", 1, LONG[4], REASONING)]
    index = sidecar(tmp_path / "teacher.jsonl", rows)
    # the marked tool output is shown whole to step 2 only; step 3 sees it compressed (build_context)
    history = [*LONG[:5], {**LONG[5], "content": LONG[5]["content"] + "[refuse]"}, *LONG[6:]]
    with pytest.raises(ValueError, match="BOS"):
        build_family_samples(history, family, name="h", reasoning=index)
    refused: list[dict] = []
    samples, dropped = build_family_samples(history, family, name="h", reasoning=index, refused=refused)
    assert dropped == Dropped() and [s["key"] for s in samples] == [("h", 1), ("h", 3)]
    assert [(r["key"], r["kind"]) for r in refused] == [(("h", 0), "bos"), (("h", 2), "template")]
    assert "BOS" in refused[0]["error"] and "cannot render" in refused[1]["error"]
    # a token boundary the sample does not split at: the prompt's "\n" merges with the reasoning's first word
    merging = loaded_fake_family(tmp_path / "merge", reasoning_field=None,
                                 tokenizer=FakeFamilyTokenizer("hybrid", pieces=("\nThe",)))
    refused = []
    samples, _ = build_family_samples(LONG, merging, name="h", reasoning=index, refused=refused)
    assert [(r["key"], r["kind"]) for r in refused] == [(("h", 1), "boundary")]
    assert [s["key"] for s in samples] == [("h", 0), ("h", 2), ("h", 3)]


def test_a_think_construction_the_template_renders_otherwise_is_refused_as_a_template_disagreement(tmp_path):
    # a template that writes the reasoning field as given: the load-time probe, whose reasoning has no
    # newline at either end, renders alike, so only a sidecar row with one shows the disagreement
    verbatim = FakeFamilyTokenizer("hybrid", strips_reasoning=False)
    family = loaded_fake_family(tmp_path / "model", tokenizer=verbatim)
    index = sidecar(tmp_path / "teacher.jsonl", [("h", 1, LONG[4], REASONING), ("h", 2, LONG[7], "Plain.")])
    with pytest.raises(FamilyError, match="renders the target with its reasoning_content") as info:
        build_family_samples(LONG, family, name="h", reasoning=index)
    assert refusal_kind(info.value) == "template"
    refused: list[dict] = []
    samples, dropped = build_family_samples(LONG, family, name="h", reasoning=index, refused=refused)
    assert dropped == Dropped() and [(r["key"], r["kind"]) for r in refused] == [(("h", 1), "template")]
    assert "reasoning_content" in refused[0]["error"]
    assert [(s["key"], s["mode"]) for s in samples] == [(("h", 0), "no_think"), (("h", 2), "think"),
                                                        (("h", 3), "no_think")]
    # a family that does not declare where its template reads reasoning from is not cross-checked
    unchecked = loaded_fake_family(tmp_path / "unchecked", tokenizer=verbatim, reasoning_field=None)
    samples, _ = build_family_samples(LONG, unchecked, name="h", reasoning=index)
    assert [s["mode"] for s in samples] == ["no_think", "think", "think", "no_think"]


def test_the_think_samples_of_a_template_that_trims_the_turn_join_the_close_as_it_renders_it(tmp_path):
    # Granite's kind of template (family_fakes' "trim" style, core: THINK samples): every step's THINK
    # sample, calls or text, is the template's own render of the target carrying its reasoning, so the
    # cross-check passes; a turn of calls alone holds one newline after the close, not two
    family = loaded_fake_family(tmp_path / "model", "trim")
    targets = [m for m in LONG if m["role"] == "assistant"]
    index = sidecar(tmp_path / "teacher.jsonl", [("h", k, m, REASONING) for k, m in enumerate(targets)])
    samples, dropped = build_family_samples(LONG, family, name="h", reasoning=index)
    assert dropped == Dropped() and [s["mode"] for s in samples] == ["think"] * len(targets) == ["think"] * 4
    text = REASONING.strip("\n")
    for (context, target), sample in zip(step_contexts(LONG), samples, strict=True):
        hf = family.to_template_messages(context)
        (message,) = family.to_template_messages([target])
        rendered = family.full(hf + [{**message, "reasoning_content": REASONING}], MODEL_TOOLS, Mode.THINK)
        assert rendered == sample["prompt"] + sample["completion"] + "\n"
        answer = render_family_sample(family, context, target, tools=MODEL_TOOLS)["completion"]
        if target.get("tool_calls"):  # its NO_THINK completion starts with the calls' newline
            assert sample["completion"] == text + "\n</think>" + answer and answer.startswith("\n<tool_call>")
        else:
            assert sample["completion"] == text + "\n</think>\n" + answer
        assert sample["completion"].startswith(text + "\n</think>\n")  # what a forced close appends


def test_a_reasoning_holding_the_closing_tag_is_refused_whoever_renders_it(tmp_path):
    # the sidecar validator keeps think tags out of the sidecars; the renderer holds the rule for a direct
    # caller too: what the sample trains must be what the family's split reads back at inference
    family = loaded_fake_family(tmp_path / "model")
    context, target = step_contexts(LONG)[1]
    sample = render_family_sample(family, context, target, tools=MODEL_TOOLS, reasoning=REASONING)
    text = REASONING.strip("\n")
    assert family.split_reasoning(sample["completion"]) == (text, len(text + THINK_CLOSE))
    with pytest.raises(FamilyError, match="closes the thinking block") as info:
        render_family_sample(family, context, target, tools=MODEL_TOOLS, reasoning="I think.\n</think>\nmore")
    assert refusal_kind(info.value) == "template"


def test_a_single_sample_is_rendered_from_a_context_and_a_target(tmp_path):
    family = loaded_fake_family(tmp_path / "model")
    context, target = step_contexts(LONG)[1]
    sample = render_family_sample(family, context, target, tools=MODEL_TOOLS, reasoning=REASONING)
    built, _ = build_family_samples(LONG, family, name="h",
                                    reasoning=sidecar(tmp_path / "t.jsonl", [("h", 1, LONG[4], REASONING)]))
    assert sample == {k: v for k, v in built[1].items() if k != "key"}
    plain = render_family_sample(family, context, target, tools=MODEL_TOOLS)
    assert plain["mode"] == "no_think" and plain["completion"].endswith(TURN_END)
    assert plain == {k: v for k, v in render_family_sample(family, context, target, tools=MODEL_TOOLS,
                                                            reasoning=REASONING, max_length=1).items()
                     if k != "think_fallback"} | {"think_fallback": False}  # the fallback, over any length


@pytest.mark.parametrize(("p99", "budget"), [
    (0, 1024), (100, 1024), (1024, 1024), (1025, 1280), (1653, 1792), (2000, 2048), (3000, 2048),
])
def test_the_reasoning_budget_is_the_p99_rounded_up_to_256_within_the_plans_floor_and_cap(p99, budget):
    assert reasoning_budget(p99) == budget  # docs/m7-计划.md §2 (推理): R = max(1024, p99 up to 256) <= 2048

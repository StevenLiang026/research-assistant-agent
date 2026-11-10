"""The family core and the registry (research_agent.family.core and .registry, docs/m7-家族层设计.md §2, §3),
offline: fake tokenizers (tests/support/family_fakes.py) stand in for the three Qwen3 template styles. The
real templates are checked in test_golden.py, which needs the model folders."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import types
from types import SimpleNamespace

import pytest
from support.family_fakes import (
    ARCHITECTURES,
    ASSISTANT,
    BOS_ID,
    MARK,
    THINK_CLOSE,
    THINK_OPEN,
    TOKENS,
    TURN_END,
    TURN_END_ID,
    DummyPrefix,
    FakeFamilyTokenizer,
    call_block,
    fake_family,
    fake_transformers,
    model_folder,
    sha256_text,
    token_ids,
)

from research_agent.agent.models import STOP
from research_agent.agent.types import api_to_hf
from research_agent.family import core, registry
from research_agent.family.core import (
    GREEDY,
    Decoding,
    FamilyError,
    LoadedFamily,
    Mode,
    Thinking,
    load_family,
    template_hash,
)
from research_agent.family.toolformats.qwen_json import QWEN_JSON
from research_agent.tools.schemas import MODEL_TOOLS

CONTEXT = [{"role": "system", "content": "system prompt"}, {"role": "user", "content": "找 KV cache 的论文"}]
CALL = {"id": "call_0", "type": "function",
        "function": {"name": "search_arxiv", "arguments": '{"query": "KV cache"}'}}
TARGET = {"role": "assistant", "content": "", "tool_calls": [CALL]}
ANSWER = call_block("search_arxiv", {"query": "KV cache"})
REASONING = "The user wants papers on KV cache, so I search arXiv."


def loaded(tmp_path, tokenizer: FakeFamilyTokenizer, **changes) -> LoadedFamily:
    """tokenizer loaded from a fake model folder as the family that describes it, with changes."""
    return load_family(model_folder(tmp_path / "model"), tokenizer=tokenizer,
                       families=[fake_family(tokenizer, **changes)])


def no_think_sample(family: LoadedFamily, context: list[dict], target: dict) -> tuple[str, str, str]:
    """(prompt, completion, full render) of docs/m7-家族层设计.md §5's NO_THINK sample, as WP-B builds it."""
    hf = family.to_template_messages(context)
    prompt = family.prompt(hf, MODEL_TOOLS, Mode.NO_THINK)
    full = family.full(hf + family.to_template_messages([target]), MODEL_TOOLS, Mode.NO_THINK)
    assert full.startswith(prompt)
    completion = full[len(prompt):]
    if completion.endswith(family.spec.turn_end + "\n"):
        completion = completion[:-1]
    assert completion.endswith(family.spec.turn_end)
    return prompt, completion, full


def with_reasoning(target: dict, reasoning: str) -> dict:
    """The target as a template message carrying its reasoning where the Qwen3 templates read it."""
    (message,) = api_to_hf([target])
    return {**message, "reasoning_content": reasoning}


# ---- prompts per thinking style -------------------------------------------------------------------------


def test_a_hybrid_familys_think_prompt_opens_the_block_its_template_leaves_to_the_model(tmp_path):
    tokenizer = FakeFamilyTokenizer("hybrid")
    family = loaded(tmp_path, tokenizer)
    hf = family.to_template_messages(CONTEXT)
    think = family.prompt(hf, MODEL_TOOLS, Mode.THINK)
    no_think = family.prompt(hf, MODEL_TOOLS, Mode.NO_THINK)
    render = tokenizer.apply_chat_template
    assert think == render(hf, tools=MODEL_TOOLS, tokenize=False, add_generation_prompt=True,
                           enable_thinking=True) + THINK_OPEN
    assert no_think == render(hf, tools=MODEL_TOOLS, tokenize=False, add_generation_prompt=True,
                              enable_thinking=False)
    assert think.endswith(ASSISTANT + THINK_OPEN) and no_think.endswith(ASSISTANT + THINK_OPEN + THINK_CLOSE)


def test_an_only_familys_no_think_prompt_is_its_think_prompt_closed_as_its_template_closes_no_reasoning(
        tmp_path):
    tokenizer = FakeFamilyTokenizer("only")
    family = loaded(tmp_path, tokenizer)
    hf = family.to_template_messages(CONTEXT)
    think = family.prompt(hf, MODEL_TOOLS, Mode.THINK)
    assert think == tokenizer.apply_chat_template(hf, tools=MODEL_TOOLS, tokenize=False,
                                                  add_generation_prompt=True)
    no_think = family.prompt(hf, MODEL_TOOLS, Mode.NO_THINK)
    assert no_think == think + "\n</think>\n\n"
    # the template renders a target without reasoning, or with an empty one, with exactly that empty block
    full = family.full(hf + family.to_template_messages([TARGET]), MODEL_TOOLS, Mode.NO_THINK)
    assert full == no_think + ANSWER + TURN_END + "\n"
    assert family.full(hf + [with_reasoning(TARGET, "")], MODEL_TOOLS, Mode.THINK) == full


def test_a_family_that_never_thinks_has_the_no_think_mode_only(tmp_path):
    family = loaded(tmp_path, FakeFamilyTokenizer("none"))
    hf = family.to_template_messages(CONTEXT)
    assert family.spec.modes == (Mode.NO_THINK,)
    assert family.prompt(hf, MODEL_TOOLS, Mode.NO_THINK).endswith(ASSISTANT)
    with pytest.raises(FamilyError, match="has no think mode"):
        family.prompt(hf, MODEL_TOOLS, Mode.THINK)
    with pytest.raises(FamilyError, match="has no think mode"):
        family.full(hf, MODEL_TOOLS, Mode.THINK)


@pytest.mark.parametrize("style", ["hybrid", "only", "none"])
def test_the_no_think_sample_is_the_full_render_after_the_prompt_ending_at_the_terminator(tmp_path, style):
    family = loaded(tmp_path, FakeFamilyTokenizer(style))
    prompt, completion, full = no_think_sample(family, CONTEXT, TARGET)
    assert completion == ANSWER + TURN_END and full == prompt + completion + "\n"
    ids, start = family.encode_sample(prompt, completion)
    assert ids == token_ids(prompt + completion) and start == len(token_ids(prompt))
    assert ids[-1] == TURN_END_ID == family.turn_end_id


@pytest.mark.parametrize("style", ["hybrid", "only"])
def test_the_think_construction_is_the_templates_own_render_of_a_target_with_its_reasoning(tmp_path, style):
    family = loaded(tmp_path, FakeFamilyTokenizer(style))
    hf = family.to_template_messages(CONTEXT)
    _, answer, _ = no_think_sample(family, CONTEXT, TARGET)
    think = family.prompt(hf, MODEL_TOOLS, Mode.THINK)
    completion = REASONING + family.spec.think_close + answer
    thought = with_reasoning(TARGET, "\n" + REASONING + "\n")  # the template strips its newlines
    assert family.full(hf + [thought], MODEL_TOOLS, Mode.THINK) == think + completion + "\n"
    ids, start = family.encode_sample(think, completion)
    assert ids[:start] == token_ids(think) and ids[-1] == TURN_END_ID
    assert family.split_reasoning(completion) == (REASONING, len(completion) - len(answer))
    assert family.think_completion(REASONING, answer, api_to_hf([TARGET])[0]) == completion


# A template that writes the turn's thinking block and text as one string with its end trimmed (the "trim"
# style, Granite's): a turn without text of its own ends the block at "</think>", and the newline before
# its calls follows. Each target: its expected THINK completion after the reasoning.
TRIMMED_TARGETS = {
    "calls": (TARGET, "\n</think>\n" + ANSWER + "\n" + TURN_END),
    "blank and calls": ({**TARGET, "content": "  "}, "\n</think>\n" + ANSWER + "\n" + TURN_END),
    "text": ({"role": "assistant", "content": "Two papers."}, "\n</think>\nTwo papers." + TURN_END),
    "text from a newline": ({"role": "assistant", "content": "\nTwo papers."},
                            "\n</think>\n\nTwo papers." + TURN_END),
    "text and calls": ({**TARGET, "content": "I search."},
                       "\n</think>\nI search.\n" + ANSWER + "\n" + TURN_END),
    "nothing": ({"role": "assistant", "content": ""}, "\n</think>" + TURN_END),
}


@pytest.mark.parametrize("case", list(TRIMMED_TARGETS))
def test_a_template_that_trims_the_turn_has_its_close_joined_to_the_answer_as_it_renders_it(tmp_path, case):
    family = loaded(tmp_path, FakeFamilyTokenizer("trim"))  # the load-time cross-check holds the rule
    assert (family.spec.think_join, family.spec.reasoning_field) == ("trim", "reasoning_content")
    target, expected = TRIMMED_TARGETS[case]
    hf = family.to_template_messages(CONTEXT)
    (message,) = family.to_template_messages([target])
    _, answer, _ = no_think_sample(family, CONTEXT, target)
    completion = family.think_completion(REASONING, answer, message)
    assert completion == REASONING + expected
    think = family.prompt(hf, MODEL_TOOLS, Mode.THINK)
    assert family.full(hf + [{**message, "reasoning_content": REASONING}], MODEL_TOOLS,
                       Mode.THINK) == think + completion + "\n"
    # the split reads the reasoning and the answer back, the answer without the newlines after the close
    read, offset = family.split_reasoning(completion)
    assert (read, completion[offset:]) == (REASONING, answer.lstrip("\n"))


def test_joining_by_plain_concatenation_where_the_template_trims_is_refused_at_load(tmp_path):
    with pytest.raises(FamilyError, match="not as the THINK construction"):
        loaded(tmp_path, FakeFamilyTokenizer("trim"), think_join="concat")
    # the call target is where they differ: its NO_THINK completion starts with the calls' newline
    family = loaded(tmp_path, FakeFamilyTokenizer("trim"), reasoning_field=None, think_join="concat")
    (message,) = family.to_template_messages([TARGET])
    _, answer, _ = no_think_sample(family, CONTEXT, TARGET)
    assert family.think_completion(REASONING, answer, message) == REASONING + "\n</think>\n\n" + answer[1:]



@pytest.mark.parametrize(("style", "suffixes"), [
    ("hybrid", {Mode.THINK: ASSISTANT + THINK_OPEN, Mode.NO_THINK: ASSISTANT}),  # it renders the empty block
    ("none", {Mode.NO_THINK: ASSISTANT + THINK_OPEN + THINK_CLOSE}),  # it renders none
])
def test_a_declared_suffix_the_template_does_not_render_is_refused_at_load(tmp_path, style, suffixes):
    with pytest.raises(FamilyError, match="no_think prompt ends with .* not with the declared suffix"):
        loaded(tmp_path, FakeFamilyTokenizer(style), generation_prompt=suffixes)


def test_an_only_family_whose_template_closes_an_empty_reasoning_otherwise_is_refused_at_load(tmp_path):
    close = "\n</think>\n\n\n"  # one newline more than the template writes after an empty reasoning
    suffixes = {Mode.THINK: ASSISTANT + THINK_OPEN, Mode.NO_THINK: ASSISTANT + THINK_OPEN + close}
    with pytest.raises(FamilyError, match="does not begin with the no_think prompt"):
        loaded(tmp_path, FakeFamilyTokenizer("only"), think_close=close, generation_prompt=suffixes)


@pytest.mark.parametrize("style", ["hybrid", "only"])
def test_a_declared_reasoning_field_the_template_ignores_is_refused_at_load(tmp_path, style):
    tokenizer = FakeFamilyTokenizer(style, renders_reasoning=False)
    with pytest.raises(FamilyError, match="not as the THINK construction"):
        loaded(tmp_path, tokenizer, reasoning_field="reasoning_content")
    assert loaded(tmp_path, tokenizer).spec.reasoning_field is None  # undeclared: nothing to cross-check


@pytest.mark.parametrize("style", ["hybrid", "only"])
def test_a_think_sample_that_does_not_split_at_a_token_boundary_is_refused_at_load(tmp_path, style):
    # a vocabulary holding a newline and the word after it as one token: the THINK prompt ends with
    # think_open's "\n" and the probe's reasoning starts with that word, so the THINK sample is the
    # template's own render, yet its ids do not begin with the prompt's
    word = core._PROBE_REASONING.split()[0]
    tokenizer = FakeFamilyTokenizer(style, pieces=("\n" + word,))
    with pytest.raises(FamilyError, match="does not split at a token boundary"):
        loaded(tmp_path, tokenizer)
    # without a reasoning field there is no THINK sample to check, and the same tokenizer loads: its
    # NO_THINK samples split cleanly
    assert loaded(tmp_path, tokenizer, reasoning_field=None).spec.reasoning_field is None


def test_a_terminator_that_is_not_one_token_is_refused_at_load(tmp_path):
    with pytest.raises(FamilyError, match="is 2 tokens, not one"):
        loaded(tmp_path, FakeFamilyTokenizer("none"), turn_end="\n" + TURN_END, stop=("\n" + TURN_END,))


def test_call_ids_in_the_template_are_refused_at_load_since_no_registered_template_renders_them(tmp_path):
    with pytest.raises(FamilyError, match="tool_call_ids .*no registered template"):
        loaded(tmp_path, FakeFamilyTokenizer("hybrid"), tool_call_ids=True)


class OtherFormat:
    """A tool format that writes calls its own way: the template's Qwen3 blocks are not its calls."""

    name, open_tag = "other", "<call>"

    def render(self, calls: list[dict]) -> str:
        return "".join(f"<call>{c['name']}</call>" for c in calls)

    def parse(self, text: str, tools: list[dict] | None) -> list[dict]:
        return []

    def locate(self, text: str) -> list:
        return []


def test_a_tool_format_that_cannot_read_the_templates_calls_is_refused_at_load(tmp_path):
    with pytest.raises(FamilyError, match="other format reads the template's render of the probe call"):
        loaded(tmp_path, FakeFamilyTokenizer("hybrid"), tool_format=OtherFormat())


# ---- BOS (docs/m7-家族层设计.md §3) --------------------------------------------------------------------


def test_a_template_bos_is_counted_once_and_first_in_every_prompt_and_sample(tmp_path):
    family = loaded(tmp_path, FakeFamilyTokenizer("hybrid", bos="template"))
    prompt, completion, _ = no_think_sample(family, CONTEXT, TARGET)
    assert prompt.startswith("<s>") and family.bos_id == BOS_ID
    ids = family.encode_prompt(prompt)
    assert ids[0] == BOS_ID and ids.count(BOS_ID) == 1
    sample, _ = family.encode_sample(prompt, completion)
    assert sample[0] == BOS_ID and sample.count(BOS_ID) == 1
    with pytest.raises(FamilyError, match="holds 0 BOS"):
        family.encode_prompt(prompt.removeprefix("<s>"))
    with pytest.raises(FamilyError, match="holds 2 BOS"):
        family.encode_prompt(prompt + "<s>")
    with pytest.raises(FamilyError, match="BOS .*not first"):
        family.encode_prompt("x" + prompt)
    with pytest.raises(FamilyError, match="holds 2 BOS"):
        family.encode_sample(prompt, "<s>" + completion)


def test_a_family_without_bos_refuses_one_in_any_prompt_or_sample(tmp_path):
    tokenizer = FakeFamilyTokenizer("hybrid", bos="token")  # it has a BOS token; the template never writes it
    family = loaded(tmp_path, tokenizer)
    prompt, completion, _ = no_think_sample(family, CONTEXT, TARGET)
    assert BOS_ID not in family.encode_prompt(prompt) and family.bos_id == BOS_ID
    with pytest.raises(FamilyError, match="holds 1 BOS"):
        family.encode_prompt("<s>" + prompt)
    with pytest.raises(FamilyError, match="holds 1 BOS"):
        family.encode_sample(prompt, "<s>" + completion)


@pytest.mark.parametrize(("tokenizer", "declared", "message"), [
    (FakeFamilyTokenizer("hybrid", bos="template"), "none", "holds 1 BOS"),  # the template writes one
    (FakeFamilyTokenizer("hybrid", bos="token"), "template", "holds 0 BOS"),  # the template never writes it
    (FakeFamilyTokenizer("hybrid", bos="none"), "template", "has no BOS token"),
    (FakeFamilyTokenizer("hybrid", bos="token", auto_bos=True), "none", "adds its BOS token by itself"),
], ids=["template-writes-one", "template-writes-none", "no-bos-token", "tokenizer-adds-one"])
def test_a_bos_policy_the_template_or_the_tokenizer_contradicts_is_refused_at_load(tmp_path, tokenizer,
                                                                                  declared, message):
    with pytest.raises(FamilyError, match=message):
        loaded(tmp_path, tokenizer, bos=declared)


def test_every_encoding_leaves_special_tokens_to_the_text(tmp_path):
    tokenizer = FakeFamilyTokenizer("hybrid", bos="template", auto_bos=True)  # adds a second BOS if asked
    family = loaded(tmp_path, tokenizer)
    prompt, completion, _ = no_think_sample(family, CONTEXT, TARGET)
    tokenizer.encoded.clear()
    assert family.encode_prompt(prompt).count(BOS_ID) == 1
    assert family.encode_sample(prompt, completion)[0].count(BOS_ID) == 1
    assert family.encode_text(THINK_CLOSE) == token_ids(THINK_CLOSE)
    assert tokenizer.encoded
    assert all(kwargs == {"add_special_tokens": False} for _, kwargs in tokenizer.encoded)


# ---- encoding and decoding ------------------------------------------------------------------------------


def test_a_sample_splits_at_a_token_boundary_ends_with_the_terminator_and_is_not_empty(tmp_path):
    family = loaded(tmp_path, FakeFamilyTokenizer("none"))
    prompt, completion, _ = no_think_sample(family, CONTEXT, TARGET)
    assert prompt.endswith("\n")
    # a completion starting with a newline merges with the prompt's last one into the "\n\n" token
    with pytest.raises(FamilyError, match="token boundary"):
        family.encode_sample(prompt, "\n" + completion)
    with pytest.raises(FamilyError, match="last id .* not the terminator"):
        family.encode_sample(prompt, completion.removesuffix(TURN_END))
    with pytest.raises(FamilyError, match="empty completion"):
        family.encode_sample(prompt + completion, "")


def test_the_close_a_generation_is_forced_with_has_the_ids_a_sample_holds_after_the_reasoning(tmp_path):
    family = loaded(tmp_path, DummyPrefix("hybrid", pieces=(MARK,)))
    mark = token_ids(MARK, (MARK,))
    assert family.encode_text(THINK_CLOSE)[:1] == mark  # alone, the close starts with the marker
    close = family.encode_close()
    assert close == token_ids(THINK_CLOSE, (MARK,))  # after the reasoning it has none
    prompt = family.prompt(family.to_template_messages(CONTEXT), MODEL_TOOLS, Mode.THINK)
    ids, _ = family.encode_sample(prompt, REASONING + THINK_CLOSE + ANSWER + TURN_END)
    before = family.encode_prompt(prompt + REASONING)
    assert ids[len(before):len(before) + len(close)] == close
    # a tokenizer without the marker: the close's own ids (every Qwen3 family's forced close is unchanged)
    plain = loaded(tmp_path / "plain", FakeFamilyTokenizer("only"))
    assert plain.encode_close() == plain.encode_text(THINK_CLOSE) == token_ids(THINK_CLOSE)
    with pytest.raises(FamilyError, match="has no thinking block"):
        loaded(tmp_path / "none", FakeFamilyTokenizer("none")).encode_close()
    # a close with no ids of its own after a word is refused at load, before any generation needs them
    with pytest.raises(FamilyError, match="merges with"):
        loaded(tmp_path / "merging", FakeFamilyTokenizer("hybrid", pieces=("a\n",)))


def test_decoding_keeps_the_special_tokens(tmp_path):
    tokenizer = FakeFamilyTokenizer("only")
    family = loaded(tmp_path, tokenizer)
    text = "<think>\nabc\n</think>\n\n" + ANSWER + TURN_END
    assert family.decode(token_ids(text)) == text
    assert tokenizer.decoded[-1][1] == {"skip_special_tokens": False}


@pytest.mark.parametrize(("text", "reasoning", "answer"), [
    ("a plan\n</think>\n\nthe answer", "a plan", "the answer"),
    ("\n\nfirst\n\n</think>\n\n\nanswer</think>more", "first", "answer</think>more"),  # the first tag counts
    ("</think>answer", "", "answer"),
    ("</think>", "", ""),
])
def test_split_reasoning_cuts_at_the_first_closing_tag_and_skips_the_newlines_after_it(tmp_path, text,
                                                                                       reasoning, answer):
    family = loaded(tmp_path, FakeFamilyTokenizer("only"))
    found, offset = family.split_reasoning(text)
    assert (found, text[offset:]) == (reasoning, answer)


def test_split_reasoning_finds_no_reasoning_without_a_closing_tag(tmp_path):
    family = loaded(tmp_path, FakeFamilyTokenizer("only"))
    assert family.split_reasoning("still thinking, cut off by the budget") == (None, 0)
    assert family.split_reasoning("") == (None, 0)


class RecordingFormat:
    """qwen_json, recording what the family hands it."""

    name, open_tag = "recording", "<tool_call>"

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def render(self, calls: list[dict]) -> str:
        return QWEN_JSON.render(calls)

    def parse(self, text: str, tools: list[dict] | None) -> list[dict]:
        self.calls.append(("parse", text, tools))
        return QWEN_JSON.parse(text, tools)

    def locate(self, text: str) -> list:
        self.calls.append(("locate", text))
        return QWEN_JSON.locate(text)


def test_parse_and_locate_are_the_tool_formats(tmp_path):
    form = RecordingFormat()
    family = loaded(tmp_path, FakeFamilyTokenizer("hybrid"), tool_format=form)
    form.calls.clear()
    assert family.parse(ANSWER, MODEL_TOOLS) == [{"name": "search_arxiv", "arguments": {"query": "KV cache"}}]
    assert family.locate(ANSWER) == QWEN_JSON.locate(ANSWER)
    assert form.calls == [("parse", ANSWER, MODEL_TOOLS), ("locate", ANSWER)]


def test_template_messages_are_api_to_hf_under_the_string_policy(tmp_path):
    family = loaded(tmp_path, FakeFamilyTokenizer("hybrid"))
    rejected = {"role": "assistant", "content": "", "reasoning_content": "an earlier thought",
                "tool_calls": [{"id": "call_7", "type": "function",
                                "function": {"name": "search_arxiv", "arguments": "query=KV cache"}}]}
    error = {"role": "tool", "tool_call_id": "call_7", "content": '{"error": "INVALID_JSON"}'}
    api = CONTEXT + [rejected, error]
    hf = family.to_template_messages(api)
    assert hf == api_to_hf(api)  # ids and the earlier step's reasoning are dropped
    # the raw arguments reach the template as the string the model wrote, which it prints verbatim
    assert hf[2]["tool_calls"][0]["function"]["arguments"] == "query=KV cache"
    assert '{"name": "search_arxiv", "arguments": query=KV cache}' in family.prompt(hf, None, Mode.NO_THINK)


# ---- identification -------------------------------------------------------------------------------------


def test_a_folder_is_identified_by_its_architectures_and_template_hash(tmp_path):
    hybrid, only = FakeFamilyTokenizer("hybrid"), FakeFamilyTokenizer("only")
    families = [fake_family(hybrid), fake_family(only)]
    folder = model_folder(tmp_path / "model")
    family = load_family(folder, tokenizer=only, families=families)
    assert family.spec is families[1] and family.tokenizer is only and family.model_dir == folder
    assert family.template_sha256 == sha256_text(only.chat_template) == template_hash(only.chat_template)
    assert load_family(folder, tokenizer=only, name="fake-only", families=families).spec is families[1]
    with pytest.raises(FamilyError, match="--family 'fake-hybrid' does not agree.*fake-only"):
        load_family(folder, tokenizer=only, name="fake-hybrid", families=families)
    with pytest.raises(FamilyError, match="unknown family 'qwen9'"):
        load_family(folder, tokenizer=only, name="qwen9", families=families)
    stranger = FakeFamilyTokenizer("none")
    with pytest.raises(FamilyError, match=f"unknown chat template {sha256_text(stranger.chat_template)}"):
        load_family(folder, tokenizer=stranger, families=families)


def test_a_known_template_on_other_architectures_is_not_the_family(tmp_path):
    tokenizer = FakeFamilyTokenizer("hybrid")
    folder = model_folder(tmp_path / "model", architectures=("LlamaForCausalLM",))
    with pytest.raises(FamilyError, match="LlamaForCausalLM.*not among.*FakeForCausalLM"):
        load_family(folder, tokenizer=tokenizer, families=[fake_family(tokenizer)])


def test_the_architectures_may_sit_under_text_config(tmp_path):
    tokenizer = FakeFamilyTokenizer("hybrid")
    folder = model_folder(tmp_path / "model", architectures=None, text_config=ARCHITECTURES)
    family = load_family(folder, tokenizer=tokenizer, families=[fake_family(tokenizer)])
    assert family.spec.name == "fake-hybrid"


def test_a_folder_without_a_configuration_naming_its_architectures_is_refused(tmp_path):
    tokenizer = FakeFamilyTokenizer("hybrid")
    families = [fake_family(tokenizer)]
    with pytest.raises(FamilyError, match="has no config.json"):
        load_family(tmp_path / "empty", tokenizer=tokenizer, families=families)
    bare = model_folder(tmp_path / "bare", architectures=None)
    with pytest.raises(FamilyError, match="names no architectures"):
        load_family(bare, tokenizer=tokenizer, families=families)


def test_two_families_claiming_one_folder_are_an_error(tmp_path):
    tokenizer = FakeFamilyTokenizer("hybrid")
    twins = [fake_family(tokenizer), fake_family(tokenizer, name="fake-twin")]
    with pytest.raises(FamilyError, match="fake-hybrid, fake-twin all claim"):
        load_family(model_folder(tmp_path / "model"), tokenizer=tokenizer, families=twins)


def test_the_template_hashed_is_the_exposed_one_or_chat_template_jinja_when_that_is_empty(tmp_path):
    jinja = "{{ messages }}\n"
    tokenizer = FakeFamilyTokenizer("hybrid", chat_template="")
    folder = model_folder(tmp_path / "model", jinja=jinja)
    family = load_family(folder, tokenizer=tokenizer,
                         families=[fake_family(tokenizer, template_sha256=frozenset({sha256_text(jinja)}))])
    assert family.template_sha256 == sha256_text(jinja)
    with pytest.raises(FamilyError, match="no chat template"):
        load_family(model_folder(tmp_path / "none"), tokenizer=tokenizer, families=[fake_family(tokenizer)])
    named = FakeFamilyTokenizer("hybrid", chat_template={"default": "a", "tool_use": "b"})
    with pytest.raises(FamilyError, match="named chat templates"):
        load_family(folder, tokenizer=named, families=[fake_family(tokenizer)])


def test_without_a_tokenizer_the_family_is_found_from_the_files_and_loaded_from_local_files_only(
        tmp_path, monkeypatch):
    tokenizer = FakeFamilyTokenizer("hybrid")
    loads = fake_transformers(monkeypatch, tokenizer)
    family = fake_family(tokenizer, revision="0123abc")
    folder = model_folder(tmp_path / "model", template=tokenizer.chat_template)
    assert load_family(folder, families=[family]).spec is family
    assert loads == [((folder,), {"local_files_only": True, "trust_remote_code": False,
                                  "revision": "0123abc"})]
    # chat_template.jinja is what transformers exposes when there is one; its text decides the family
    shadowed = model_folder(tmp_path / "shadowed", template="another template", jinja=tokenizer.chat_template)
    assert load_family(shadowed, families=[family]).spec is family
    with pytest.raises(FamilyError, match="unknown chat template"):
        load_family(model_folder(tmp_path / "other", template="another template"), families=[family])
    # the tokenizer must expose the template the files hold
    claims_the_other = fake_family(tokenizer, name="claims-the-other",
                                   template_sha256=frozenset({sha256_text("another template")}))
    differs = model_folder(tmp_path / "differs", template="another template")
    with pytest.raises(FamilyError, match="exposes a template hashing to"):
        load_family(differs, families=[family, claims_the_other])


def test_the_record_names_what_a_run_depends_on(tmp_path):
    tokenizer = FakeFamilyTokenizer("only")
    folder = model_folder(tmp_path / "model", tokenizer_json=b'{"model": "fake"}')
    family = load_family(folder, tokenizer=tokenizer, families=[fake_family(tokenizer)])
    record = family.record()
    assert record == {"name": "fake-only", "template_sha256": sha256_text(tokenizer.chat_template),
                      "tokenizer_json_sha256": hashlib.sha256(b'{"model": "fake"}').hexdigest(),
                      "bos": "none", "thinking": "only", "lora_targets": ["q_proj", "v_proj"],
                      "remote": False, "remote_files": {}, "revision": None}
    assert json.loads(json.dumps(record)) == record
    assert loaded(tmp_path / "bare", tokenizer).record()["tokenizer_json_sha256"] is None


# ---- decoding settings ----------------------------------------------------------------------------------


def test_greedy_decoding_sends_the_temperature_alone_as_today_and_sampling_sends_all_five_fields():
    assert GREEDY == Decoding() and GREEDY.payload() == {"temperature": 0.0}
    assert json.dumps(Decoding(temperature=0).payload()) == '{"temperature": 0.0}'  # today's request bytes
    assert Decoding(0.6, 0.95, 20, 0.0).payload() == {"temperature": 0.6, "top_p": 0.95, "top_k": 20,
                                                      "min_p": 0.0, "presence_penalty": 0.0}
    assert Decoding(temperature=1, presence_penalty=1.5).payload() == {
        "temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "presence_penalty": 1.5}


def test_a_repetition_penalty_is_sent_as_llama_servers_repeat_penalty_only_when_it_is_on():
    # 1.0 is off (server-schema.cpp: "1.0 = disabled", the server's default): not sent, so every payload
    # of a decoding without one is the payload it was, whatever the request's reach
    lfm = Decoding(temperature=0.1, top_k=50, repetition_penalty=1.1)
    assert lfm.payload(reach=4608) == {"temperature": 0.1, "top_p": 1.0, "top_k": 50, "min_p": 0.0,
                                       "presence_penalty": 0.0, "repeat_penalty": 1.1, "repeat_last_n": 4608}
    assert list(lfm.payload(reach=7))[-2:] == ["repeat_penalty", "repeat_last_n"]  # after the five fields
    off = Decoding(temperature=0.1, top_k=50, repetition_penalty=1.0)
    assert off.payload() == off.payload(reach=4608) == {
        "temperature": 0.1, "top_p": 1.0, "top_k": 50, "min_p": 0.0, "presence_penalty": 0.0}
    assert Decoding(temperature=1, repetition_penalty=1).payload()["temperature"] == 1.0
    assert Decoding().repetition_penalty == 1.0 and GREEDY.payload() == GREEDY.payload(reach=9) == {
        "temperature": 0.0}
    assert isinstance(Decoding(temperature=1, repetition_penalty=2).payload(reach=9)["repeat_penalty"], float)
    # any penalty but 1.0 is on and sent: one below 1.0 favours repeated tokens, and llama-server reads it
    assert Decoding(temperature=0.1, top_k=50, repetition_penalty=0.9).payload(reach=9) == {
        "temperature": 0.1, "top_p": 1.0, "top_k": 50, "min_p": 0.0, "presence_penalty": 0.0,
        "repeat_penalty": 0.9, "repeat_last_n": 9}


def test_a_repetition_penalty_goes_with_the_window_of_the_whole_sequence_the_request_can_reach():
    # transformers penalises every token of the sequence, prompt included; llama-server the last
    # repeat_last_n tokens (64 unless the request names a window), and -1 is outside the field's bounds
    # (server-schema.cpp: 0 to INT32_MAX), so the request names its whole reach and never the default
    lfm = Decoding(temperature=0.1, top_k=50, repetition_penalty=1.1)
    with pytest.raises(ValueError, match="repeat_last_n"):
        lfm.payload()
    for reach in (0, -1, 2**31, 2.0, True, "4096"):
        with pytest.raises(ValueError, match="reach"):
            lfm.payload(reach=reach)
    assert lfm.payload(reach=1)["repeat_last_n"] == 1
    assert lfm.payload(reach=2**31 - 1)["repeat_last_n"] == 2**31 - 1


@pytest.mark.parametrize("settings", [
    {"top_k": 20}, {"top_p": 0.95}, {"min_p": 0.1}, {"presence_penalty": 1.0},  # greedy would not send them
    {"repetition_penalty": 1.1},
    {"temperature": -0.1}, {"temperature": float("nan")}, {"temperature": 0.7, "top_p": 0.0},
    {"temperature": 0.7, "top_p": 1.5}, {"temperature": 0.7, "min_p": 1.5}, {"temperature": 0.7, "top_k": -1},
    {"temperature": 0.7, "top_k": 2.5}, {"temperature": 0.7, "top_k": True}, {"temperature": True},
    {"temperature": 0.7, "repetition_penalty": 0.0}, {"temperature": 0.7, "repetition_penalty": -1.1},
    {"temperature": 0.7, "repetition_penalty": float("inf")},
    {"temperature": 0.7, "repetition_penalty": True},
])
def test_decoding_settings_that_would_not_be_sent_or_are_out_of_range_are_refused(settings):
    with pytest.raises(ValueError):
        Decoding(**settings)


# ---- the specification ----------------------------------------------------------------------------------

HYBRID = FakeFamilyTokenizer("hybrid")
ONLY = FakeFamilyTokenizer("only")
NONE = FakeFamilyTokenizer("none")


@pytest.mark.parametrize(("tokenizer", "changes", "message"), [
    (NONE, {"template_kwargs": {Mode.THINK: {}, Mode.NO_THINK: {}}}, "template_kwargs must give exactly"),
    (HYBRID, {"generation_prompt": {Mode.THINK: ASSISTANT + THINK_OPEN}},
     "generation_prompt must give exactly"),
    (HYBRID, {"decoding": {Mode.NO_THINK: GREEDY}}, "decoding must give exactly"),
    (HYBRID, {"generation_prompt": {Mode.THINK: ASSISTANT, Mode.NO_THINK: ASSISTANT}},
     "must end with think_open"),
    (ONLY, {"generation_prompt": {Mode.THINK: ASSISTANT + THINK_OPEN, Mode.NO_THINK: ASSISTANT}},
     "think suffix followed by think_close"),
    (HYBRID, {"think_open": None}, "needs think_open and think_close"),
    (HYBRID, {"think_close": ""}, "needs think_open and think_close"),
    (NONE, {"think_open": THINK_OPEN}, "never thinks has no think_open"),
    (NONE, {"reasoning_field": "reasoning_content"}, "reasoning_field .* needs a family that thinks"),
    (NONE, {"inference_mode": Mode.THINK}, "inference_mode think is not among its modes"),
    (HYBRID, {"stop": ("</s>",)}, "turn_end .* must be among the stop strings"),
    (HYBRID, {"bos": "auto"}, "bos must be"),
    (HYBRID, {"empty_block": "none"}, "empty_block must be"),
    (HYBRID, {"think_join": "trimmed"}, "think_join must be"),
    (NONE, {"think_join": "trim"}, "think_join 'trim' .*a family that thinks"),
    (HYBRID, {"empty_block": "constructed"}, "only a family that always thinks constructs"),
    (NONE, {"empty_block": "constructed"}, "only a family that always thinks constructs"),
    (HYBRID, {"raw_arguments": "json"}, "raw_arguments must be"),
    (HYBRID, {"tool_call_ids": "no"}, "tool_call_ids must be true or false"),
    (HYBRID, {"text_from_ids": 1}, "text_from_ids must be true or false"),
    (HYBRID, {"remote_files": {"modeling.py": "0" * 64}}, "remote_files .*trust_remote_code"),
    (HYBRID, {"trust_remote_code": True}, "trust_remote_code .*remote_files"),
    (HYBRID, {"trust_remote_code": True, "remote_files": {"modeling.py": "abc"}}, "remote_files must map"),
    (HYBRID, {"trust_remote_code": True, "remote_files": {"modeling.txt": "0" * 64}},
     "remote_files must map"),
    (HYBRID, {"trust_remote_code": True, "remote_files": {"sub/modeling.py": "0" * 64}},
     "remote_files must map"),
    (HYBRID, {"architectures": ()}, "architectures must be"),
    (HYBRID, {"template_sha256": frozenset({"abc"})}, "template_sha256 must be"),
    (HYBRID, {"template_sha256": {"0" * 64}}, "template_sha256 must be"),
    (HYBRID, {"thinking": "hybrid"}, "thinking must be a Thinking"),
    (HYBRID, {"tool_format": object()}, "tool_format must be a ToolFormat"),
    (HYBRID, {"lora_targets": ()}, "lora_targets must be"),
    (HYBRID, {"lora_targets": ""}, "lora_targets must be"),
    (HYBRID, {"reasoning_budget": 0}, "reasoning_budget must be"),
    (HYBRID, {"trust_remote_code": 1}, "trust_remote_code must be true or false"),
    (HYBRID, {"revision": ""}, "revision must be"),
    (HYBRID, {"name": ""}, "a family needs a name"),
    (HYBRID, {"attn_implementation": "flash_attention_2"}, "attn_implementation must be"),
    (HYBRID, {"attn_implementation": None}, "attn_implementation must be"),
])
def test_a_specification_that_contradicts_itself_is_refused(tokenizer, changes, message):
    with pytest.raises(FamilyError, match=message):
        fake_family(tokenizer, **changes)


def test_a_specification_has_defaults_and_cannot_be_changed():
    kwargs = {Mode.THINK: {"enable_thinking": True}, Mode.NO_THINK: {"enable_thinking": False}}
    family = fake_family(HYBRID, template_kwargs=kwargs)
    assert (family.trust_remote_code, family.revision, family.reasoning_budget, family.inference_mode) == (
        False, None, 1024, Mode.NO_THINK)
    assert family.attn_implementation == "sdpa"  # what scripts/train.py built every model with before
    assert (family.empty_block, family.text_from_ids, dict(family.remote_files)) == ("template", False, {})
    assert dict(family.decoding) == {Mode.THINK: GREEDY, Mode.NO_THINK: GREEDY}
    assert family.modes == (Mode.THINK, Mode.NO_THINK)
    kwargs[Mode.NO_THINK]["enable_thinking"] = True  # the caller's dict is copied, not kept
    assert family.template_kwargs[Mode.NO_THINK] == {"enable_thinking": False}
    with pytest.raises(dataclasses.FrozenInstanceError):
        family.name = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        family.template_kwargs[Mode.NO_THINK]["enable_thinking"] = True  # type: ignore[index]
    with pytest.raises(TypeError):
        family.generation_prompt[Mode.THINK] = ""  # type: ignore[index]
    with pytest.raises(TypeError):
        family.remote_files["x.py"] = "0" * 64  # type: ignore[index]
    assert family.template_kwargs["no_think"] == {"enable_thinking": False}  # modes are strings too
    assert hash(family) == hash(fake_family(HYBRID))  # usable as a key; the mappings stay out of the hash


# ---- the registry (docs/m7-家族层设计.md §2, 登记表) ----------------------------------------------------

# SHA-256 of each local model folder's chat template, as tokenizer.chat_template exposes it (2026-10-19):
# the templates of Qwen3-1.7B and Qwen3-4B are byte-identical, so the hybrid family pins one hash.
HYBRID_SHA = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
INSTRUCT_SHA = "64f85b198065d0fba2a81f37e10ed68161ce2c19a754c7100e67e0ca2ee9c326"
THINKING_SHA = "3802169b2a02b81e6adb7ab4f64f91ff02db753c8c3a64a01c35192d3a61d8d7"
QWEN3_ARCHITECTURES = ("Qwen3ForCausalLM",)
SEVEN_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
EMPTY_BLOCK = "<think>\n\n</think>\n\n"


def test_the_registry_holds_the_three_qwen3_families_as_the_design_lists_them():
    hybrid, instruct, thinking = registry.FAMILIES[:3]
    assert [f.name for f in registry.FAMILIES[:3]] == ["qwen3-hybrid", "qwen3-instruct-2507",
                                                       "qwen3-thinking-2507"]
    assert [f.template_sha256 for f in registry.FAMILIES[:3]] == [
        frozenset({HYBRID_SHA}), frozenset({INSTRUCT_SHA}), frozenset({THINKING_SHA})]
    assert [f.thinking for f in registry.FAMILIES[:3]] == [Thinking.HYBRID, Thinking.NONE, Thinking.ONLY]
    assert dict(hybrid.generation_prompt) == {Mode.THINK: "<|im_start|>assistant\n<think>\n",
                                              Mode.NO_THINK: "<|im_start|>assistant\n" + EMPTY_BLOCK}
    assert dict(instruct.generation_prompt) == {Mode.NO_THINK: "<|im_start|>assistant\n"}
    assert dict(thinking.generation_prompt) == {Mode.THINK: "<|im_start|>assistant\n<think>\n",
                                                Mode.NO_THINK: "<|im_start|>assistant\n" + EMPTY_BLOCK}
    assert {m: dict(k) for m, k in hybrid.template_kwargs.items()} == {
        Mode.THINK: {"enable_thinking": True}, Mode.NO_THINK: {"enable_thinking": False}}
    assert {m: dict(k) for m, k in instruct.template_kwargs.items()} == {Mode.NO_THINK: {}}
    assert {m: dict(k) for m, k in thinking.template_kwargs.items()} == {Mode.THINK: {}, Mode.NO_THINK: {}}
    # docs/m7-计划.md §2 (解码): thinking samples with the vendor's settings; qwen3-hybrid answers at once
    # greedily, as G2 did, with today's request bytes
    sampled = Decoding(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
    assert [dict(f.decoding) for f in registry.FAMILIES[:3]] == [
        {Mode.THINK: sampled, Mode.NO_THINK: GREEDY}, {Mode.NO_THINK: GREEDY},
        {Mode.THINK: sampled, Mode.NO_THINK: sampled}]
    assert hybrid.decoding[Mode.NO_THINK].payload() == {"temperature": 0.0}
    assert [f.inference_mode for f in registry.FAMILIES[:3]] == [Mode.NO_THINK, Mode.NO_THINK, Mode.THINK]
    assert [f.think_open for f in registry.FAMILIES[:3]] == ["<think>\n", None, "<think>\n"]
    assert [f.reasoning_field for f in registry.FAMILIES[:3]] == ["reasoning_content", None,
                                                                  "reasoning_content"]
    for family in registry.FAMILIES[:3]:
        assert family.architectures == QWEN3_ARCHITECTURES
        assert family.think_close == "\n</think>\n\n"
        assert (family.turn_end, family.stop) == ("<|im_end|>", tuple(STOP))  # agent.models' stop string
        assert family.bos == "none" and family.tool_format is QWEN_JSON
        assert (family.raw_arguments, family.tool_call_ids) == ("string", False)
        assert (family.empty_block, family.text_from_ids) == ("template", False)
        assert family.lora_targets == SEVEN_PROJECTIONS  # scripts/train.py's TARGET_MODULES
        assert (family.trust_remote_code, family.revision, family.reasoning_budget) == (False, None, 1024)


# The seven families of the third phase (docs/m7-计划.md, 候选 C3-C9), as probed from the local folders on
# 2026-10-19 with transformers 5.18: the template hash each pins, and what the template does.
QWEN35_SHA = "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
GRANITE_SHA = "f0ba43f79b3cabca5e5a7584c77aec92f49feae45cdf7779c87d9fc54cd90258"
NANBEIGE_THINKING_SHA = "7db1a76279e4bdabda27ca6cf0c523ad0576b3d61b647e2bd49b5f065c9eb117"
NANBEIGE41_SHA = "757f108242fb797d23b79f72c088cb5b4bc37fb3744e29dc8771c1dc2b1cb335"
MINICPM5_SHA = "bc1f78ef8a86b6bdf64a056cf89dd0ac506db4f198ab363d85168e2500ab44dd"
SPARK_SHA = "705345721970da40e6aa56db548a46ecd1d8d94a5418808fa180ac19ee77a9ba"
LFM25_SHA = "ea663864491de7ade391839479860ca95541f892f72665c73251fbd4643b1bef"
SPARK_CODE = {"modeling_spark.py": "9cf0d1ad2b54b9f7088792779ddd8b5cb4d4fe63bfea054da7f2dd16362bcaf4",
              "configuration_spark.py": "02218597240490f490659b184052db940b08163ff9c3af7b7e59323f6f963722"}
IM_END = "<|im_end|>"
SPARK_END = "<｜end▁of▁sentence｜>"
LLAMA_TARGETS = SEVEN_PROJECTIONS
QWEN_PROMPTS = ("<|im_start|>assistant\n<think>\n", "<|im_start|>assistant\n<think>\n\n</think>\n\n")
QWEN_TAGS = ("<think>\n", "\n</think>\n\n")
QWEN35_TARGETS = (r"model\.(language_model\.)?layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\."
                  r"(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)")
SPARK_TARGETS = ("q_k_v_proj", "out_proj", "gate_proj", "up_proj", "down_proj")
LFM2_TARGETS = ("q_proj", "k_proj", "v_proj", "out_proj", "in_proj", "w1", "w2", "w3")
THIRD_PHASE = {  # name: (architecture, hash, thinking, bos, format, (think prompt, no_think prompt),
    #                     (open, close), turn end, raw policy, text from ids, empty block, LoRA targets,
    #                     remote, inference)
    "qwen3.5-4b": ("Qwen3_5ForConditionalGeneration", QWEN35_SHA, Thinking.HYBRID, "none", "qwen_xml",
                   QWEN_PROMPTS, QWEN_TAGS, IM_END, "inline", False, "template", QWEN35_TARGETS, False,
                   Mode.THINK),
    "granite-4.2-3b": ("GraniteForCausalLM", GRANITE_SHA, Thinking.HYBRID, "none", "qwen_xml",
                       ("<|im_start|>assistant\n<think>\n", "<|im_start|>assistant\n<think></think>"),
                       ("<think>\n", "\n</think>\n"), IM_END, "inline", False, "template", LLAMA_TARGETS,
                       False, Mode.THINK),
    "nanbeige4-thinking-2511": ("LlamaForCausalLM", NANBEIGE_THINKING_SHA, Thinking.ONLY, "turn", "qwen_json",
                                QWEN_PROMPTS, QWEN_TAGS, IM_END, "string", False, "template", LLAMA_TARGETS,
                                False, Mode.THINK),
    "nanbeige4.1-3b": ("LlamaForCausalLM", NANBEIGE41_SHA, Thinking.ONLY, "turn", "qwen_json", QWEN_PROMPTS,
                       QWEN_TAGS, IM_END, "string", False, "template", LLAMA_TARGETS, False, Mode.THINK),
    "minicpm5-2b": ("LlamaForCausalLM", MINICPM5_SHA, Thinking.HYBRID, "template", "minicpm_xml",
                    QWEN_PROMPTS, QWEN_TAGS, IM_END, "inline", True, "template", LLAMA_TARGETS, False,
                    Mode.THINK),
    "spark-x2.5-4b": ("Spark2_5ForCausalLM", SPARK_SHA, Thinking.HYBRID, "turn", "glm_kv",
                      ("<|Bot|><think>", "<|Bot|></think>"), ("<think>", "</think>"), SPARK_END, "inline",
                      False, "template", SPARK_TARGETS, True, Mode.THINK),
    "lfm2.5-2.6b": ("Lfm2ForCausalLM", LFM25_SHA, Thinking.ONLY, "template", "pythonic",
                    ("<|im_start|>assistant\n<think>", "<|im_start|>assistant\n<think></think>"),
                    ("<think>", "</think>"), IM_END, "inline", True, "constructed", LFM2_TARGETS, False,
                    Mode.THINK),
}
THIRD_PHASE_DECODING = {  # the vendor's sampling for thinking (generation_config.json, or the model card)
    "qwen3.5-4b": Decoding(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5),
    "granite-4.2-3b": Decoding(temperature=1.0, top_p=0.95),
    "nanbeige4-thinking-2511": Decoding(temperature=0.6, top_p=0.95),
    "nanbeige4.1-3b": Decoding(temperature=0.6, top_p=0.95),
    "minicpm5-2b": Decoding(temperature=1.0, top_p=0.95),
    "spark-x2.5-4b": Decoding(temperature=1.0, top_p=0.95),
    "lfm2.5-2.6b": Decoding(temperature=0.1, top_k=50, repetition_penalty=1.1),
}


def test_the_registry_holds_the_seven_families_of_the_third_phase_as_probed():
    families = {f.name: f for f in registry.FAMILIES[3:]}
    assert list(families) == list(THIRD_PHASE)
    for name, facts in THIRD_PHASE.items():
        (architecture, sha, thinking, bos, form, (think, no_think), (open_, close), end, raw, from_ids, block,
         targets, remote, inference) = facts
        family = families[name]
        assert family.architectures == (architecture,) and family.template_sha256 == frozenset({sha})
        assert (family.thinking, family.bos, family.tool_format.name) == (thinking, bos, form)
        assert dict(family.generation_prompt) == {Mode.THINK: think, Mode.NO_THINK: no_think}
        assert (family.think_open, family.think_close, family.turn_end) == (open_, close, end)
        assert family.stop == (end,) and family.tool_call_ids is False
        assert (family.raw_arguments, family.text_from_ids, family.empty_block) == (raw, from_ids, block)
        assert (family.lora_targets, family.trust_remote_code, family.inference_mode) == (targets, remote,
                                                                                         inference)
        sampled = THIRD_PHASE_DECODING[name]
        no_think_decoding = sampled if thinking is Thinking.ONLY else GREEDY  # a hybrid answers greedily
        assert dict(family.decoding) == {Mode.THINK: sampled, Mode.NO_THINK: no_think_decoding}
        assert family.reasoning_budget == 1024
        # LFM2.5's repetition penalty is the only one on, so its requests alone carry repeat_penalty, with
        # the window of the whole sequence
        assert ("repeat_penalty" in sampled.payload(reach=4608)) is (name == "lfm2.5-2.6b")
        assert ("repeat_last_n" in sampled.payload(reach=4608)) is (name == "lfm2.5-2.6b")
    assert families["spark-x2.5-4b"].revision == "0bcb35678590218655dff3765b9e61c83b35e9c4"
    assert dict(families["spark-x2.5-4b"].remote_files) == SPARK_CODE
    assert all(f.revision is None and not f.remote_files for n, f in families.items() if n != "spark-x2.5-4b")
    # Spark's vendor class declares its tied weights as a list; transformers 5.18 reads a mapping from the
    # tied parameter to its source, the embedding of Spark2_5Model (core: Model patches)
    assert {k: dict(v) for k, v in families["spark-x2.5-4b"].model_patches.items()} == {
        "_tied_weights_keys": {"lm_head.weight": "model.embedding.weight"}}
    assert all(not f.model_patches for f in registry.FAMILIES if f.name != "spark-x2.5-4b")
    # Spark's vendor attention is eager only (core: Attention); every other model is built with sdpa
    assert {f.name: f.attn_implementation for f in registry.FAMILIES} == {
        f.name: ("eager" if f.name == "spark-x2.5-4b" else "sdpa") for f in registry.FAMILIES}
    # the template switches: Granite, MiniCPM5 and Spark answer with enable_thinking False; Qwen3.5 thinks
    # by default and answers with it False; the always-thinking templates take nothing
    assert {m: dict(k) for m, k in families["qwen3.5-4b"].template_kwargs.items()} == {
        Mode.THINK: {}, Mode.NO_THINK: {"enable_thinking": False}}
    for name in ("granite-4.2-3b", "minicpm5-2b", "spark-x2.5-4b"):
        assert {m: dict(k) for m, k in families[name].template_kwargs.items()} == {
            Mode.THINK: {"enable_thinking": True}, Mode.NO_THINK: {"enable_thinking": False}}
    for name in ("nanbeige4-thinking-2511", "nanbeige4.1-3b", "lfm2.5-2.6b"):
        assert {m: dict(k) for m, k in families[name].template_kwargs.items()} == {Mode.THINK: {},
                                                                                   Mode.NO_THINK: {}}
    # the reasoning cross-check: every template renders reasoning_content, and Granite's trims a turn
    # without text after the close, which its join declares (core: THINK samples)
    assert {n: f.reasoning_field for n, f in families.items()} == dict.fromkeys(families, "reasoning_content")
    assert {f.name: f.think_join for f in registry.FAMILIES} == {
        f.name: ("trim" if f.name == "granite-4.2-3b" else "concat") for f in registry.FAMILIES}


def test_the_registry_resolves_by_architectures_and_template_hash_and_a_given_name_must_agree():
    hybrid, instruct, thinking = registry.FAMILIES[:3]
    assert registry.resolve(QWEN3_ARCHITECTURES, HYBRID_SHA) is hybrid
    assert registry.resolve(list(QWEN3_ARCHITECTURES), INSTRUCT_SHA, name="qwen3-instruct-2507") is instruct
    assert registry.resolve(QWEN3_ARCHITECTURES, THINKING_SHA) is thinking
    with pytest.raises(FamilyError, match="--family 'qwen3-hybrid' does not agree.*qwen3-thinking-2507"):
        registry.resolve(QWEN3_ARCHITECTURES, THINKING_SHA, name="qwen3-hybrid")
    with pytest.raises(FamilyError, match="unknown chat template"):
        registry.resolve(QWEN3_ARCHITECTURES, "0" * 64)
    with pytest.raises(FamilyError, match="not among"):
        registry.resolve(("LlamaForCausalLM",), HYBRID_SHA)


def test_registered_families_have_distinct_names_and_templates():
    names = [f.name for f in registry.FAMILIES]
    hashes = [h for f in registry.FAMILIES for h in f.template_sha256]
    assert len(names) == 10 and len(set(names)) == len(names) and len(set(hashes)) == len(hashes)
    for family in registry.FAMILIES:
        assert registry.resolve(family.architectures, next(iter(family.template_sha256))) is family


# ---- the error kinds the training path tells apart (docs/m7-计划.md, S2: template, boundary, BOS) ----------


class RefusingTemplate(FakeFamilyTokenizer):
    """A template that raises on a message it cannot render, as jinja raises on a shape it does not take."""

    def apply_chat_template(self, messages, **kwargs):
        if any("[refuse]" in (m.get("content") or "") for m in messages):
            raise TypeError("unsupported message shape")
        return super().apply_chat_template(messages, **kwargs)


def test_a_template_that_raises_is_a_template_error_naming_the_cause(tmp_path):
    family = loaded(tmp_path, RefusingTemplate("hybrid"))
    hf = family.to_template_messages(CONTEXT + [{"role": "user", "content": "[refuse] this"}])
    with pytest.raises(core.TemplateError, match="fake-hybrid: the chat template raised .*TypeError: "
                                                 "unsupported message shape") as caught:
        family.prompt(hf, MODEL_TOOLS, Mode.NO_THINK)
    assert isinstance(caught.value, FamilyError) and isinstance(caught.value.__cause__, TypeError)
    with pytest.raises(core.TemplateError):
        family.full(hf, MODEL_TOOLS, Mode.NO_THINK)


def test_a_token_boundary_failure_and_a_bos_failure_are_their_own_kinds_of_family_error(tmp_path):
    family = loaded(tmp_path, FakeFamilyTokenizer("none"))
    prompt, completion, _ = no_think_sample(family, CONTEXT, TARGET)
    with pytest.raises(core.TokenBoundaryError, match="token boundary") as boundary:
        family.encode_sample(prompt, "\n" + completion)
    assert isinstance(boundary.value, FamilyError)
    with_bos = loaded(tmp_path / "bos", FakeFamilyTokenizer("hybrid", bos="token"))
    prompt, completion, _ = no_think_sample(with_bos, CONTEXT, TARGET)
    with pytest.raises(core.BosError, match="holds 1 BOS") as bos:
        with_bos.encode_sample(prompt, "<s>" + completion)
    assert isinstance(bos.value, FamilyError)
    with pytest.raises(core.BosError):
        with_bos.encode_prompt("<s>" + prompt)
    assert not issubclass(core.TokenBoundaryError, core.BosError)
    assert not issubclass(core.TemplateError, core.TokenBoundaryError | core.BosError)


# ---- the third phase's policies: a BOS that opens every turn, a terminator that is not the eos, the ------
# ---- constructed empty block, inline raw arguments, text from ids and pinned vendor code -----------------


def test_a_bos_that_opens_every_turn_must_come_first_and_is_not_counted(tmp_path):
    tokenizer = FakeFamilyTokenizer("hybrid", bos="turn")  # "<|im_start|>" is the BOS and opens every turn
    family = loaded(tmp_path, tokenizer)
    assert family.spec.bos == "turn" and family.bos_id == TOKENS["<|im_start|>"]
    prompt, completion, _ = no_think_sample(family, CONTEXT, TARGET)
    ids = family.encode_prompt(prompt)
    assert ids[0] == family.bos_id and ids.count(family.bos_id) == 3  # system, user, assistant
    sample, _ = family.encode_sample(prompt, completion)
    assert sample[0] == family.bos_id and sample.count(family.bos_id) == 3
    with pytest.raises(core.BosError, match="does not start with the BOS"):
        family.encode_prompt("x" + prompt)
    with pytest.raises(core.BosError, match="does not start with the BOS"):
        family.encode_sample("x" + prompt, completion)
    assert family.encode_text("<|im_start|>x") == token_ids("<|im_start|>x")  # no rule inside a sequence


@pytest.mark.parametrize(("declared", "message"), [
    ("none", "holds 3 BOS"), ("template", "holds 3 BOS"),
])
def test_a_turn_opening_bos_contradicts_the_other_policies_at_load(tmp_path, declared, message):
    with pytest.raises(FamilyError, match=message):
        loaded(tmp_path, FakeFamilyTokenizer("hybrid", bos="turn"), bos=declared)


@pytest.mark.parametrize(("bos", "message"), [
    ("none", "has no BOS token"), ("token", "does not start with the BOS"),
    ("template", "does not open every turn"),  # written once: the probe's prompt holds it once, not per turn
])
def test_turn_declared_for_a_template_that_does_not_open_its_turns_with_the_bos_is_refused(tmp_path, bos,
                                                                                            message):
    with pytest.raises(FamilyError, match=message):
        loaded(tmp_path, FakeFamilyTokenizer("hybrid", bos=bos), bos="turn")


def test_the_terminator_is_the_turn_end_whatever_the_tokenizers_eos_is(tmp_path):
    tokenizer = FakeFamilyTokenizer("hybrid", eos="</s>")  # MiniCPM5: eos </s>, turns end with <|im_end|>
    family = loaded(tmp_path, tokenizer)
    assert tokenizer.eos_token_id == TOKENS["</s>"] != family.turn_end_id == TURN_END_ID
    prompt, completion, _ = no_think_sample(family, CONTEXT, TARGET)
    assert completion.endswith(TURN_END)
    ids, _ = family.encode_sample(prompt, completion)
    assert ids[-1] == TURN_END_ID
    with pytest.raises(FamilyError, match="last id .* not the terminator"):
        family.encode_sample(prompt, completion.removesuffix(TURN_END) + "</s>")
    assert family.spec.stop == (TURN_END,)


def test_an_only_family_whose_template_renders_no_empty_block_has_it_constructed(tmp_path):
    tokenizer = FakeFamilyTokenizer("only", empty_block=False)  # LFM2.5: a target without reasoning, bare
    family = loaded(tmp_path, tokenizer)
    assert family.spec.empty_block == "constructed"
    hf = family.to_template_messages(CONTEXT)
    think = family.prompt(hf, MODEL_TOOLS, Mode.THINK)
    no_think = family.prompt(hf, MODEL_TOOLS, Mode.NO_THINK)
    assert no_think == think + THINK_CLOSE
    # the template writes no block for the target; the family's full render holds the empty block where the
    # NO_THINK prompt has it, so the completion is the answer alone
    bare = tokenizer.apply_chat_template(hf + api_to_hf([TARGET]), tools=MODEL_TOOLS, tokenize=False)
    assert "<think>" not in bare[len(think) - len(ASSISTANT + THINK_OPEN):]
    prompt, completion, full = no_think_sample(family, CONTEXT, TARGET)
    assert prompt == no_think and completion == ANSWER + TURN_END and full == no_think + completion + "\n"
    assert full == bare.replace(ASSISTANT + ANSWER, ASSISTANT + THINK_OPEN + THINK_CLOSE + ANSWER)
    # THINK: the template's own render with reasoning is the construction
    thought = family.full(hf + [with_reasoning(TARGET, REASONING)], MODEL_TOOLS, Mode.THINK)
    assert thought == think + REASONING + THINK_CLOSE + ANSWER + TURN_END + "\n"
    ids, start = family.encode_sample(prompt, completion)
    assert ids[:start] == token_ids(prompt) and ids[-1] == TURN_END_ID


def test_the_constructed_block_needs_a_target_and_a_template_that_renders_none(tmp_path):
    bare = FakeFamilyTokenizer("only", empty_block=False)
    with pytest.raises(FamilyError, match="does not begin with the no_think prompt"):
        loaded(tmp_path, bare, empty_block="template")  # the template's render holds no block
    with pytest.raises(FamilyError, match="renders a thinking block itself"):
        loaded(tmp_path, FakeFamilyTokenizer("only"), empty_block="constructed")  # it would be doubled
    family = loaded(tmp_path, bare)
    hf = family.to_template_messages(CONTEXT)
    with pytest.raises(FamilyError, match="ends with an assistant message"):
        family.full(hf, MODEL_TOOLS, Mode.NO_THINK)  # no target to put the block in front of


def test_raw_arguments_reach_a_refusing_template_inline_as_the_formats_text(tmp_path):
    refusing = FakeFamilyTokenizer("hybrid", refuses_raw=True)
    family = loaded(tmp_path, refusing)
    assert family.spec.raw_arguments == "inline"
    rejected = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_7", "type": "function",
         "function": {"name": "search_arxiv", "arguments": "query=KV cache"}},
        {"id": "call_8", "type": "function",
         "function": {"name": "get_paper", "arguments": '{"arxiv_id": "2609.01234"}'}}]}
    error = {"role": "tool", "tool_call_id": "call_7", "content": '{"error": "INVALID_JSON"}'}
    hf = family.to_template_messages(CONTEXT + [rejected, error])
    # the message's calls, all of them and in order, are the format's text in its content: the template
    # (which refuses a string) sees no tool_calls at all
    inline = (call_block("search_arxiv", "query=KV cache") + "\n"
              + call_block("get_paper", {"arxiv_id": "2609.01234"}))
    assert hf[2] == {"role": "assistant", "content": inline}
    assert hf[3] == {"role": "tool", "content": '{"error": "INVALID_JSON"}'}
    assert inline in family.prompt(hf, MODEL_TOOLS, Mode.NO_THINK)
    # a message with text before its calls keeps the text, one newline before the calls
    noted = {**rejected, "content": "Let me look."}
    assert family.to_template_messages([noted])[0]["content"] == "Let me look.\n" + inline
    # an object-argument message is untouched: the template renders its calls
    fine = {**rejected, "tool_calls": rejected["tool_calls"][1:]}
    assert family.to_template_messages([fine]) == api_to_hf([fine])
    # the string policy on such a template is refused at load: the probe's raw call raises in the template
    with pytest.raises(FamilyError, match="the chat template raised"):
        loaded(tmp_path / "string", refusing, raw_arguments="string")


def test_the_string_policy_must_show_the_raw_text_in_the_render(tmp_path):
    family = loaded(tmp_path, FakeFamilyTokenizer("hybrid"))
    assert family.spec.raw_arguments == "string"
    raw = {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_7", "type": "function", "function": {"name": "search_arxiv", "arguments": "query=x"}}]}
    assert family.to_template_messages([raw]) == api_to_hf([raw])  # the string goes to the template as it is
    assert "query=x" in family.full(family.to_template_messages(CONTEXT + [raw]), MODEL_TOOLS, Mode.NO_THINK)
    # the inline policy is accepted on a template that takes strings too: it is a choice, not a fact of the
    # template, and the render holds the text either way
    inline = loaded(tmp_path / "inline", FakeFamilyTokenizer("hybrid"), raw_arguments="inline")
    assert inline.spec.raw_arguments == "inline"


def test_text_from_ids_follows_whether_the_tags_are_control_tokens(tmp_path):
    special = FakeFamilyTokenizer("hybrid", special_tags=True)  # MiniCPM5: the tags are special tokens
    assert loaded(tmp_path, special).spec.text_from_ids is True
    with pytest.raises(FamilyError, match="special tokens.*text_from_ids"):
        loaded(tmp_path, special, text_from_ids=False)  # the server's text would lose them
    plain = FakeFamilyTokenizer("hybrid")
    assert loaded(tmp_path, plain).spec.text_from_ids is False
    with pytest.raises(FamilyError, match="no tag .*text_from_ids"):
        loaded(tmp_path, plain, text_from_ids=True)  # nothing to decode for: the server's text is whole


class ClosingTagSpecial(FakeFamilyTokenizer):
    """A tokenizer whose closing call tag alone is a special token: the opening tag is text."""

    @property
    def added_tokens_decoder(self) -> dict:
        return {**super().added_tokens_decoder,
                TOKENS["</tool_call>"]: SimpleNamespace(content="</tool_call>", special=True)}


def test_every_tag_of_the_templates_call_counts_for_text_from_ids_not_only_the_opening_one(tmp_path):
    tokenizer = ClosingTagSpecial("hybrid")
    with pytest.raises(FamilyError, match="</tool_call> encode to special tokens.*text_from_ids"):
        loaded(tmp_path, tokenizer)  # the server's text would lose the closing tag of every call
    assert loaded(tmp_path, tokenizer, text_from_ids=True).spec.text_from_ids


class PipedFormat:
    """qwen_json's reading behind a "<|...|>" opening tag, the shape llama.cpp's converter marks as a control
    token whether or not the tokenizer calls it special (LFM2.5's <|tool_call_start|>)."""

    name, open_tag = "piped", "<|tool_call_start|>"

    def render(self, calls: list[dict]) -> str:
        return QWEN_JSON.render(calls)

    def parse(self, text: str, tools: list[dict] | None) -> list[dict]:
        return QWEN_JSON.parse(text, tools)

    def locate(self, text: str) -> list:
        return QWEN_JSON.locate(text)


def test_a_piped_tag_counts_as_a_control_token_whether_or_not_the_tokenizer_calls_it_special(tmp_path):
    tokenizer = FakeFamilyTokenizer("hybrid", pieces=("<|tool_call_start|>",))  # an added, non-special token
    family = loaded(tmp_path, tokenizer, tool_format=PipedFormat(), text_from_ids=True)
    assert family.spec.text_from_ids
    with pytest.raises(FamilyError, match="special tokens.*text_from_ids"):
        loaded(tmp_path, tokenizer, tool_format=PipedFormat(), text_from_ids=False)


SOURCE = "class Model: pass\n"
OTHER = "class Config: pass\n"


def pinned(*sources: str) -> dict[str, str]:
    names = ("modeling.py", "configuration.py")
    return {name: hashlib.sha256(source.encode("utf-8")).hexdigest() for name, source in zip(names, sources)}


def test_pinned_vendor_code_is_checked_before_the_tokenizer_loads_with_trust_remote_code(tmp_path,
                                                                                         monkeypatch):
    tokenizer = FakeFamilyTokenizer("hybrid")
    loads = fake_transformers(monkeypatch, tokenizer)
    family = fake_family(tokenizer, trust_remote_code=True, remote_files=pinned(SOURCE, OTHER),
                         revision="0bcb356")
    folder = model_folder(tmp_path / "model", template=tokenizer.chat_template,
                          python={"modeling.py": SOURCE, "configuration.py": OTHER})
    loaded_family = load_family(folder, families=[family])
    assert loaded_family.spec is family and loads[-1][1]["trust_remote_code"] is True
    record = loaded_family.record()
    assert record["remote"] and record["remote_files"] == pinned(SOURCE, OTHER)
    # a changed file, a missing file and an unpinned Python file are each refused before anything loads
    loads.clear()
    model_folder(tmp_path / "changed", template=tokenizer.chat_template,
                 python={"modeling.py": SOURCE + "# changed\n", "configuration.py": OTHER})
    with pytest.raises(FamilyError, match="modeling.py .*sha256"):
        load_family(tmp_path / "changed", families=[family])
    model_folder(tmp_path / "missing", template=tokenizer.chat_template, python={"modeling.py": SOURCE})
    with pytest.raises(FamilyError, match="configuration.py .*missing"):
        load_family(tmp_path / "missing", families=[family])
    model_folder(tmp_path / "extra", template=tokenizer.chat_template,
                 python={"modeling.py": SOURCE, "configuration.py": OTHER, "helper.py": "x = 1\n"})
    with pytest.raises(FamilyError, match="helper.py .*not pinned"):
        load_family(tmp_path / "extra", families=[family])
    assert loads == []
    # a tokenizer given skips the loading, not the check (the folder's code is what training will run)
    with pytest.raises(FamilyError, match="not pinned"):
        load_family(tmp_path / "extra", tokenizer=tokenizer, families=[family])


def test_a_family_without_vendor_code_ignores_python_files_in_the_folder(tmp_path):
    tokenizer = FakeFamilyTokenizer("hybrid")
    folder = model_folder(tmp_path / "model", python={"granite_thinking_parser.py": "x = 1\n"})
    family = load_family(folder, tokenizer=tokenizer, families=[fake_family(tokenizer)])
    assert family.record()["remote"] is False


# ---- model patches: vendor code built by the installed transformers (module docstring: Model patches) ------

TIED = {"lm_head.weight": "model.embedding.weight"}  # the mapping transformers 5.18 reads


class VendorModel:
    """A model class as a vendor's module defines it: its tied weights declared as a list."""

    _tied_weights_keys = ["lm_head.weight"]  # noqa: RUF012  as the vendor wrote it


class OtherModel:
    """Another class than the one the family patches."""


def fake_hf(monkeypatch, vendor: type, *, built: type | None = None) -> list[tuple]:
    """transformers in sys.modules for this test: AutoConfig gives a configuration whose auto_map names the
    vendor class, dynamic_module_utils resolves it, and AutoModelForCausalLM.from_pretrained builds it (or
    built) and keeps (what, path, keyword arguments, the class's _tied_weights_keys when it is built)."""
    calls: list[tuple] = []

    def config(path, **kwargs):
        calls.append(("config", str(path), dict(kwargs)))
        return SimpleNamespace(auto_map={"AutoModelForCausalLM": "modeling.VendorModel"})

    def resolve(reference, path, **kwargs):
        calls.append(("class", reference, str(path)))
        return vendor

    def from_pretrained(path, **kwargs):
        calls.append(("model", str(path), dict(kwargs), vendor._tied_weights_keys))
        return (built or vendor)()

    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = SimpleNamespace(from_pretrained=config)
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=from_pretrained)
    dynamic = types.ModuleType("transformers.dynamic_module_utils")
    dynamic.get_class_from_dynamic_module = resolve
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.dynamic_module_utils", dynamic)
    monkeypatch.setattr(vendor, "_tied_weights_keys", vendor._tied_weights_keys)  # restored afterwards
    return calls


def patched_family(tokenizer: FakeFamilyTokenizer, **changes) -> core.Family:
    fields = {"trust_remote_code": True, "remote_files": pinned(SOURCE, OTHER),
              "model_patches": {"_tied_weights_keys": TIED}}
    return fake_family(tokenizer, **{**fields, **changes})


def vendor_folder(path, tokenizer: FakeFamilyTokenizer, modeling: str = SOURCE):
    return model_folder(path, template=tokenizer.chat_template,
                        python={"modeling.py": modeling, "configuration.py": OTHER})


def test_a_family_sets_its_model_patches_on_the_vendor_class_before_the_model_is_built(tmp_path, monkeypatch):
    tokenizer = FakeFamilyTokenizer("hybrid")
    family = patched_family(tokenizer)
    folder = vendor_folder(tmp_path / "model", tokenizer)
    calls = fake_hf(monkeypatch, VendorModel)
    model = core.build_causal_lm(family, folder, dtype="bf16", device_map={"": 0})
    assert isinstance(model, VendorModel) and VendorModel._tied_weights_keys == TIED
    config, resolved, built = calls
    assert config == ("config", str(folder), {"trust_remote_code": True})
    assert resolved == ("class", "modeling.VendorModel", str(folder))
    # the auto class builds the patched class, with the call scripts/train.py made before the patch
    assert built == ("model", str(folder), {"dtype": "bf16", "device_map": {"": 0}, "trust_remote_code": True,
                                            "attn_implementation": "sdpa"}, TIED)
    # a copy each time: what transformers does with the mapping cannot change the family's
    VendorModel._tied_weights_keys["x.weight"] = "y.weight"
    assert dict(family.model_patches["_tied_weights_keys"]) == TIED
    assert core.patch_model_class(family, folder) is VendorModel and VendorModel._tied_weights_keys == TIED
    # the vendor files are as pinned: the family still loads from the folder
    assert load_family(folder, tokenizer=tokenizer, families=[family]).record()["model_patches"] == {
        "_tied_weights_keys": TIED}


def test_a_family_without_model_patches_builds_the_model_as_before_and_resolves_nothing(tmp_path,
                                                                                       monkeypatch):
    tokenizer = FakeFamilyTokenizer("hybrid")
    family = fake_family(tokenizer)
    calls = fake_hf(monkeypatch, VendorModel)
    core.build_causal_lm(family, tmp_path / "model", dtype="bf16")
    assert calls == [("model", str(tmp_path / "model"),
                      {"dtype": "bf16", "trust_remote_code": False, "attn_implementation": "sdpa"},
                      ["lm_head.weight"])]
    assert core.patch_model_class(family, tmp_path / "model") is None and len(calls) == 1
    record = load_family(model_folder(tmp_path / "model"), tokenizer=tokenizer, families=[family]).record()
    assert "model_patches" not in record and "attn_implementation" not in record


def test_the_model_is_built_with_the_familys_attention_which_a_caller_cannot_change(tmp_path, monkeypatch):
    # core: Attention. Spark's vendor class runs its own eager attention and declares no SDPA support, so
    # transformers refuses "sdpa" when it builds it: the family names "eager", and build_causal_lm, not the
    # caller, decides
    tokenizer = FakeFamilyTokenizer("hybrid")
    eager = patched_family(tokenizer, attn_implementation="eager")
    folder = vendor_folder(tmp_path / "model", tokenizer)
    calls = fake_hf(monkeypatch, VendorModel)
    core.build_causal_lm(eager, folder, dtype="bf16")
    core.build_causal_lm(fake_family(tokenizer), tmp_path / "plain", dtype="bf16")
    built = [call[2] for call in calls if call[0] == "model"]
    assert [kwargs["attn_implementation"] for kwargs in built] == ["eager", "sdpa"]
    calls.clear()
    with pytest.raises(FamilyError, match="the family's attention, 'eager', not 'sdpa'"):
        core.build_causal_lm(eager, folder, attn_implementation="sdpa")
    assert calls == []  # refused before anything was resolved or built
    assert core.build_causal_lm(eager, folder, attn_implementation="eager") is not None  # the same: fine
    # the record names the attention of a family built otherwise than with sdpa
    record = load_family(folder, tokenizer=tokenizer, families=[eager]).record()
    assert record["attn_implementation"] == "eager"


def test_patching_refuses_changed_vendor_code_and_a_model_built_from_another_class(tmp_path, monkeypatch):
    tokenizer = FakeFamilyTokenizer("hybrid")
    family = patched_family(tokenizer)
    calls = fake_hf(monkeypatch, VendorModel)
    changed = vendor_folder(tmp_path / "changed", tokenizer, modeling=SOURCE + "# changed\n")
    with pytest.raises(FamilyError, match="modeling.py .*sha256"):
        core.build_causal_lm(family, changed)
    assert calls == [] and VendorModel._tied_weights_keys == ["lm_head.weight"]  # nothing ran, nothing set
    calls = fake_hf(monkeypatch, VendorModel, built=OtherModel)
    with pytest.raises(FamilyError, match="not the patched"):
        core.build_causal_lm(family, vendor_folder(tmp_path / "model", tokenizer))


@pytest.mark.parametrize(("changes", "message"), [
    ({"trust_remote_code": False, "remote_files": {}}, "model_patches .*vendor code"),
    ({"model_patches": {"not a name": 1}}, "model_patches must map"),
    ({"model_patches": {"_tied": {1, 2}}}, "model_patches must map"),
    ({"model_patches": {"_tied": float("nan")}}, "model_patches must map"),
])
def test_model_patches_need_vendor_code_names_and_json_values(changes, message):
    with pytest.raises(FamilyError, match=message):
        patched_family(FakeFamilyTokenizer("hybrid"), **changes)


def test_model_patches_are_frozen_copies():
    value = {"lm_head.weight": "model.embedding.weight"}
    family = fake_family(HYBRID, trust_remote_code=True, remote_files=pinned(SOURCE, OTHER),
                         model_patches={"_tied_weights_keys": value})
    value["other.weight"] = "x"
    assert dict(family.model_patches["_tied_weights_keys"]) == TIED
    with pytest.raises(TypeError):
        family.model_patches["_x"] = 1  # type: ignore[index]
    assert dict(fake_family(HYBRID).model_patches) == {}


# ---- module adapters: vendor code adapted to the installed stack at load (module docstring: Module adapters)

MASKS = ("create_causal_mask", "create_sliding_window_causal_mask")
MASK_ADAPTERS = {name: "masking_4_57" for name in MASKS}


def mask_function(name: str):
    """A mask function with transformers 5.18's keywords, returning its name."""

    def function(config, inputs_embeds, attention_mask, past_key_values, position_ids=None):
        return name

    function.__name__ = name
    return function


def adapted_vendor(monkeypatch) -> tuple[type, types.ModuleType, types.ModuleType]:
    """A vendor module registered for this test, defining a model class and holding the installed mask
    functions as Spark's module imports them (transformers.masking_utils, a stand-in with 5.18's keywords);
    (the class, the module, the stand-in)."""
    masking = types.ModuleType("transformers.masking_utils")
    for name in MASKS:
        setattr(masking, name, mask_function(name))
    module = types.ModuleType("transformers_modules.fake.modeling")
    exec("class AdaptedModel:\n    _tied_weights_keys = ['lm_head.weight']\n", module.__dict__)  # noqa: S102
    for name in MASKS:
        setattr(module, name, getattr(masking, name))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.AdaptedModel, module, masking


def test_build_causal_lm_adapts_the_vendor_module_before_the_model_is_built(tmp_path, monkeypatch):
    tokenizer = FakeFamilyTokenizer("hybrid")
    model_class, module, masking = adapted_vendor(monkeypatch)
    family = patched_family(tokenizer, module_adapters=MASK_ADAPTERS)
    folder = vendor_folder(tmp_path / "model", tokenizer)
    fake_hf(monkeypatch, model_class)
    monkeypatch.setitem(sys.modules, "transformers.masking_utils", masking)
    seen = []

    def from_pretrained(path, **kwargs):  # what the module holds when the model is built
        seen.append({name: getattr(getattr(module, name), "__family_adapter__", None) for name in MASKS})
        return model_class()

    sys.modules["transformers"].AutoModelForCausalLM = SimpleNamespace(from_pretrained=from_pretrained)
    model = core.build_causal_lm(family, folder, dtype="bf16")
    assert isinstance(model, model_class) and model_class._tied_weights_keys == TIED
    assert seen == [MASK_ADAPTERS]  # adapted before from_pretrained
    assert [getattr(module, name).__wrapped__ for name in MASKS] == [getattr(masking, name) for name in MASKS]
    # a family with module adapters and no class patch resolves its class all the same
    adapters_only = patched_family(tokenizer, model_patches={}, module_adapters=MASK_ADAPTERS)
    assert core.patch_model_class(adapters_only, folder) is model_class
    # an adapter that refuses what it finds stops the build before anything is built
    seen.clear()
    module.create_causal_mask = mask_function("the vendor's own")
    with pytest.raises(FamilyError, match="not transformers.masking_utils.create_causal_mask"):
        core.build_causal_lm(family, folder, dtype="bf16")
    assert seen == []


def test_a_refused_adapter_leaves_the_class_unpatched_and_an_unimported_module_is_refused(tmp_path,
                                                                                        monkeypatch):
    # the module is adapted before the class is patched: a refusal sets nothing on either
    tokenizer = FakeFamilyTokenizer("hybrid")
    model_class, module, masking = adapted_vendor(monkeypatch)
    family = patched_family(tokenizer, module_adapters=MASK_ADAPTERS)
    folder = vendor_folder(tmp_path / "model", tokenizer)
    calls = fake_hf(monkeypatch, model_class)
    monkeypatch.setitem(sys.modules, "transformers.masking_utils", masking)
    module.create_sliding_window_causal_mask = mask_function("the vendor's own")
    with pytest.raises(FamilyError, match="not transformers.masking_utils.create_sliding_window_causal_mask"):
        core.build_causal_lm(family, folder, dtype="bf16")
    assert model_class._tied_weights_keys == ["lm_head.weight"] and not [c for c in calls if c[0] == "model"]
    assert module.create_causal_mask is masking.create_causal_mask  # the first target was not replaced
    # a class whose module is not imported cannot be adapted
    monkeypatch.delitem(sys.modules, module.__name__)
    with pytest.raises(FamilyError, match="is not imported"):
        core.patch_model_class(family, folder)


@pytest.mark.parametrize(("changes", "message"), [
    ({"trust_remote_code": False, "remote_files": {}, "model_patches": {}, "module_adapters": MASK_ADAPTERS},
     "module_adapters .*vendor code"),
    ({"module_adapters": {"create_causal_mask": "no_such_adapter"}}, "module_adapters must map"),
    ({"module_adapters": {"not a name": "masking_4_57"}}, "module_adapters must map"),
    ({"module_adapters": {"a.b.c": "masking_4_57"}}, "module_adapters must map"),
    ({"module_adapters": {"create_causal_mask": 1}}, "module_adapters must map"),
    ({"module_adapters": ["create_causal_mask"]}, "module_adapters must map"),
    ({"module_adapters": {"Spark2_5DecoderLayer.forward": "masking_4_57"}},
     "masking_4_57 adapts .*not Spark2_5DecoderLayer.forward"),
    ({"module_adapters": {"create_causal_mask": "spark_projection_dtype"}}, "spark_projection_dtype adapts"),
])
def test_module_adapters_need_vendor_code_and_registered_adapters_of_their_targets(changes, message):
    with pytest.raises(FamilyError, match=message):
        patched_family(FakeFamilyTokenizer("hybrid"), **changes)


def test_module_adapters_are_frozen_copies_recorded_for_a_family_that_declares_them(tmp_path):
    tokenizer = FakeFamilyTokenizer("hybrid")
    adapters = dict(MASK_ADAPTERS)
    family = patched_family(tokenizer, module_adapters=adapters)
    adapters["create_causal_mask"] = "spark_projection_dtype"  # the caller's dict is copied, not kept
    assert dict(family.module_adapters) == MASK_ADAPTERS
    with pytest.raises(TypeError):
        family.module_adapters["x"] = "masking_4_57"  # type: ignore[index]
    assert dict(fake_family(HYBRID).module_adapters) == {}
    record = load_family(vendor_folder(tmp_path / "model", tokenizer), tokenizer=tokenizer,
                         families=[family]).record()
    assert record["module_adapters"] == MASK_ADAPTERS and json.loads(json.dumps(record)) == record
    plain = load_family(model_folder(tmp_path / "plain"), tokenizer=tokenizer,
                        families=[fake_family(tokenizer)])
    assert "module_adapters" not in plain.record()


def test_a_folders_family_is_found_from_its_files_alone_its_vendor_code_checked(tmp_path, monkeypatch):
    tokenizer = FakeFamilyTokenizer("hybrid")
    family = patched_family(tokenizer)
    loads = fake_transformers(monkeypatch, tokenizer)
    assert core.folder_family(vendor_folder(tmp_path / "model", tokenizer), families=[family]) is family
    assert loads == []  # no tokenizer is loaded
    with pytest.raises(FamilyError, match="modeling.py .*sha256"):
        core.folder_family(vendor_folder(tmp_path / "changed", tokenizer, modeling="x = 2\n"),
                           families=[family])
    with pytest.raises(FamilyError, match="does not agree"):
        core.folder_family(vendor_folder(tmp_path / "model", tokenizer), name="other", families=[
            family, dataclasses.replace(family, name="other", template_sha256=frozenset({"0" * 64}))])

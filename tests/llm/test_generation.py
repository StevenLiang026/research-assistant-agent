"""Two-phase generation (research_agent.llm.generation; docs/m7-家族层设计.md §7), offline: a fake family
(tests/support/family_fakes.py) over a scripted llama-server (tests/support/llama_fakes.py). The four cases of
the procedure, the request seeds, the answer slice at the boundary, the usage over the phases, the one
request a NO_THINK generation makes, byte for byte the request LocalClient sent before M7, and the text of a
family whose tags are control tokens, read from the ids the server sends (text_from_ids)."""

from __future__ import annotations

import dataclasses
import itertools
import json
from pathlib import Path

import pytest
from support.family_fakes import (
    CHAR_OFFSET,
    MARK,
    PIECE_OFFSET,
    THINK_CLOSE,
    TOKENS,
    TURN_END_ID,
    DummyPrefix,
    FakeFamilyTokenizer,
    fake_family,
    model_folder,
    token_ids,
)
from support.llama_fakes import ID_BASE, FakeLlama, Scripted, fake_loaded_family, scripted, utf8_cut

from research_agent.agent.route import token_spans
from research_agent.family.core import GREEDY, Decoding, Mode, load_family
from research_agent.llm.generation import (
    Generation,
    GenerationError,
    GenerationSettings,
    generate,
    request_seed,
)
from research_agent.llm.local import MAX_TOKENS, Completion
from research_agent.tools.schemas import MODEL_TOOLS
from research_agent.train.samples import render_family_sample

CONTEXT = [{"role": "system", "content": "system prompt"}, {"role": "user", "content": "找 KV cache 的论文"}]
SAMPLING = Decoding(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
ADAPTER = {"id": 0, "path": "runs/spikes/s4_adapter-f16.gguf", "scale": 1.0}
CALL = '<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "KV cache"}}\n</tool_call>'
# The body LocalClient sent before M7 for a greedy request with the adapter at scale 1 (captured on
# 2026-10-19 from the client as it was), the prompt text replaced by the prompt's ids.
TODAY_BODY = (b'{"prompt":%s,"max_tokens":512,"temperature":0.0,"cache_prompt":false,"logprobs":5,'
              b'"lora":[{"id":0,"scale":1.0}],"stop":["<|im_end|>"]}')
EOS_ID = TOKENS["</s>"]  # a control token that ends a sequence, not a turn (MiniCPM5's eos)


@pytest.fixture
def family(tmp_path):
    return fake_loaded_family(tmp_path / "model", "hybrid")


def prompt_ids(family, mode: Mode = Mode.THINK) -> list[int]:
    return family.encode_prompt(family.prompt(family.to_template_messages(CONTEXT), MODEL_TOOLS, mode))


def think(reasoning_budget: int = 8, decoding: Decoding = SAMPLING, **changes) -> GenerationSettings:
    return GenerationSettings(mode=Mode.THINK, decoding=decoding, reasoning_budget=reasoning_budget,
                              **changes)


def run(family, script: list[Scripted], settings: GenerationSettings, *, lora_scale: float = 0.0,
        **options) -> tuple[Generation, FakeLlama]:
    fake = FakeLlama(script, adapters=[ADAPTER] if lora_scale else [], **options)
    return generate(fake.client(), family, prompt_ids(family, settings.mode), settings,
                    lora_scale=lora_scale), fake


# ---- settings and seeds ---------------------------------------------------------------------------------


def test_settings_default_to_the_spec_and_refuse_nonsense():
    settings = GenerationSettings(mode=Mode.THINK, decoding=SAMPLING)
    assert (settings.reasoning_budget, settings.answer_tokens, settings.base_seed, settings.sample) == (
        1024, MAX_TOKENS, 0, 0)
    assert settings.samples and not GenerationSettings(mode=Mode.NO_THINK, decoding=GREEDY).samples
    for bad in ({"reasoning_budget": 0}, {"answer_tokens": 0}, {"base_seed": -1}, {"sample": -1},
                {"reasoning_budget": 2.5}, {"base_seed": True}):
        with pytest.raises(ValueError):
            GenerationSettings(mode=Mode.THINK, decoding=SAMPLING, **bad)
    with pytest.raises(ValueError, match="mode"):
        GenerationSettings(mode="think", decoding=SAMPLING)
    with pytest.raises(ValueError, match="decoding"):
        GenerationSettings(mode=Mode.THINK, decoding={"temperature": 0.6})


def test_settings_record_what_a_manifest_keeps():
    settings = GenerationSettings(mode=Mode.THINK, decoding=SAMPLING, reasoning_budget=256, base_seed=2,
                                  sample=1)
    assert settings.record() == {
        "mode": "think", "decoding": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                                      "presence_penalty": 0.0},
        "reasoning_budget": 256, "answer_tokens": 512, "base_seed": 2, "sample": 1}
    assert json.loads(json.dumps(settings.record())) == settings.record()
    # a repetition penalty that is on is recorded too, with its window, every request's whole sequence;
    # one that is off leaves the record as it was
    penalised = GenerationSettings(mode=Mode.THINK, decoding=PENALISED)
    assert penalised.record()["decoding"] == {"temperature": 0.1, "top_p": 1.0, "top_k": 50, "min_p": 0.0,
                                              "presence_penalty": 0.0, "repetition_penalty": 1.1,
                                              "repetition_window": "sequence"}
    # so is one below 1.0: on as well, and sent (core.Decoding.payload)
    below = GenerationSettings(mode=Mode.THINK, decoding=Decoding(temperature=0.1, repetition_penalty=0.9))
    assert below.record()["decoding"]["repetition_penalty"] == 0.9
    assert below.record()["decoding"]["repetition_window"] == "sequence"
    assert "repetition_window" not in settings.record()["decoding"]


def test_a_voted_system_records_its_votes_and_every_other_settings_record_stays_as_it_was():
    # docs/m7-本地提升计划.md WP2: "votes" only when voting, so the manifests before it keep their bytes
    plain = GenerationSettings(mode=Mode.THINK, decoding=SAMPLING, reasoning_budget=256, base_seed=2,
                               sample=1)
    assert plain.votes == 0 and "votes" not in plain.record()
    voted = GenerationSettings(mode=Mode.THINK, decoding=SAMPLING, votes=5)
    assert voted.record() == {**GenerationSettings(mode=Mode.THINK, decoding=SAMPLING).record(), "votes": 5}
    for bad in (-1, 2.0, True):
        with pytest.raises(ValueError, match="votes"):
            GenerationSettings(mode=Mode.THINK, decoding=SAMPLING, votes=bad)
    with pytest.raises(ValueError, match="votes"):  # the samples of a vote must sample
        GenerationSettings(mode=Mode.THINK, decoding=GREEDY, votes=5)


PENALISED = Decoding(temperature=0.1, top_k=50, repetition_penalty=1.1)  # LFM2.5's vendor sampling


def test_a_repetition_penalty_goes_with_every_request_of_a_generation_and_with_no_other(family):
    gen, fake = run(family, [Scripted(("thinking", " on"), "length"), Scripted(("Answer",))],
                    think(reasoning_budget=2, decoding=PENALISED))
    assert [r["repeat_penalty"] for r in fake.requests] == [1.1, 1.1] and gen.forced
    _, fake = run(family, [Scripted(("thinking", " on"), "length"), Scripted(("Answer",))],
                  think(reasoning_budget=2))
    assert all(not {"repeat_penalty", "repeat_last_n"} & set(r) for r in fake.requests)
    _, fake = run(family, [Scripted(("thinking", " on"), "length"), Scripted(("Answer",))],
                  think(reasoning_budget=2, decoding=Decoding(temperature=0.1, repetition_penalty=0.9)))
    assert [r["repeat_penalty"] for r in fake.requests] == [0.9, 0.9]


def window(request: dict) -> int:
    """The tokens a request's sequence can reach: its prompt's ids and its max_tokens."""
    return len(request["prompt"]) + request["max_tokens"]


def test_the_penalty_window_of_every_request_is_the_whole_sequence_it_can_reach(family):
    # transformers penalises every token of the sequence, the prompt's included; llama-server the last
    # repeat_last_n, which every request therefore sets to its prompt's ids and its max_tokens: the window
    # holds the whole sequence at every step, the first request's prompt, the reasoning and the close in
    # the second's prompt included
    gen, fake = run(family, [Scripted(("thinking", " on"), "length"), Scripted(("Answer",))],
                    think(reasoning_budget=2, decoding=PENALISED))
    first, second = fake.requests
    assert first["repeat_last_n"] == len(prompt_ids(family)) + 2
    assert second["repeat_last_n"] == len(prompt_ids(family)) + 2 + len(token_ids(THINK_CLOSE)) + MAX_TOKENS
    assert [r["repeat_last_n"] for r in fake.requests] == [window(r) for r in fake.requests] and gen.forced
    assert list(first)[-5:] == ["repeat_penalty", "repeat_last_n", *ASKED]  # after the penalty
    # a closed reasoning whose answer goes on: the continuation's window is its own prompt and budget
    gen, fake = run(family, [Scripted(("T", "\n</think>\n\n", "A"), "length"), Scripted(("B",))],
                    think(reasoning_budget=3, answer_tokens=50, decoding=PENALISED))
    assert gen.phases == 2
    assert [r["repeat_last_n"] for r in fake.requests] == [window(r) for r in fake.requests]
    assert fake.requests[1]["repeat_last_n"] == len(prompt_ids(family)) + 3 + 49
    # a NO_THINK generation's one request
    settings = GenerationSettings(mode=Mode.NO_THINK, decoding=PENALISED, answer_tokens=64)
    _, fake = run(family, [scripted("A")], settings)
    (request,) = fake.requests
    assert request["repeat_last_n"] == len(prompt_ids(family, Mode.NO_THINK)) + 64 == window(request)


def test_settings_for_a_family_take_its_inference_mode_decoding_and_budget(tmp_path):
    thinking = fake_loaded_family(tmp_path / "only", "only", inference_mode=Mode.THINK, reasoning_budget=768,
                                  decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY})
    settings = GenerationSettings.for_family(thinking)
    assert (settings.mode, settings.decoding, settings.reasoning_budget) == (Mode.THINK, SAMPLING, 768)
    assert (settings.base_seed, settings.sample) == (0, 0)
    chosen = GenerationSettings.for_family(thinking, mode="no_think", reasoning_budget=32, base_seed=2)
    assert (chosen.mode, chosen.decoding, chosen.reasoning_budget, chosen.base_seed) == (
        Mode.NO_THINK, GREEDY, 32, 2)
    hybrid = fake_loaded_family(tmp_path / "hybrid", "hybrid")  # greedy in every mode, answers at once
    assert GenerationSettings.for_family(hybrid) == GenerationSettings(mode=Mode.NO_THINK, decoding=GREEDY)
    never = fake_loaded_family(tmp_path / "none", "none")
    with pytest.raises(ValueError, match="has no think mode"):
        GenerationSettings.for_family(never, mode=Mode.THINK)


def test_request_seeds_are_deterministic_separate_by_every_input_and_fit_a_signed_32_bit_seed():
    ids = [5, 6, 7]
    seed = request_seed(0, ids, 1, 0)
    assert seed == request_seed(0, list(ids), 1, 0)
    others = [request_seed(0, ids, 2, 0), request_seed(0, ids, 1, 1), request_seed(1, ids, 1, 0),
              request_seed(0, [*ids, 8], 1, 0)]
    assert len({seed, *others}) == 5
    assert all(0 <= s < 2 ** 31 for s in (seed, *others))
    assert seed == 565281805  # pinned: the encoding must not drift, a run's seeds are its reproducibility


# ---- the four cases of a THINK generation ----------------------------------------------------------------


def test_a_reasoning_closed_within_the_budget_is_one_request_split_at_the_close(family):
    gen, fake = run(family, [Scripted(("Let me", " think", "\n</think>", "\n\n", "Hello", " world"))],
                    think(reasoning_budget=8, base_seed=3))
    ids = prompt_ids(family)
    (request,) = fake.requests
    assert request["prompt"] == ids and request["max_tokens"] == 8 and request["stop"] == ["<|im_end|>"]
    assert request["seed"] == request_seed(3, ids, 1, 0) and "lora" not in request
    assert {k: request[k] for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty")} == {
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
    assert gen.reasoning == "Let me think" and gen.answer_text == "Hello world"
    assert (gen.forced, gen.phases, gen.seeds, gen.reasoning_tokens) == (False, 1, [request["seed"]], 4)
    assert gen.answer.tokens == ["Hello", " world"] and gen.answer.text == "Hello world"
    assert gen.answer.token_ids == [ID_BASE + 4, ID_BASE + 5] and gen.answer.finish_reason == "stop"
    assert gen.answer.logprobs == [-0.1, -0.1] and gen.answer.token_bytes == [list(b"Hello"), list(b" world")]
    assert gen.usage == {"prompt_tokens": len(ids), "completion_tokens": 6, "total_tokens": len(ids) + 6}


def test_a_reasoning_cut_by_the_budget_is_closed_by_force_and_the_answer_asked_in_a_second_request(family):
    gen, fake = run(family, [Scripted(("I keep", " thinking"), "length"), Scripted(("Answer", "."))],
                    think(reasoning_budget=2, base_seed=1))
    ids = prompt_ids(family)
    first, second = fake.requests
    assert first["max_tokens"] == 2 and first["seed"] == request_seed(1, ids, 1, 0)
    assert second["prompt"] == ids + [ID_BASE, ID_BASE + 1] + token_ids(THINK_CLOSE)
    assert second["max_tokens"] == MAX_TOKENS and second["seed"] == request_seed(1, ids, 2, 0)
    assert second["temperature"] == 0.6 and second["stop"] == ["<|im_end|>"]
    assert gen.reasoning == "I keep thinking" and gen.answer_text == "Answer."
    assert (gen.forced, gen.phases, gen.reasoning_tokens) == (True, 2, 2)
    assert gen.seeds == [first["seed"], second["seed"]]
    assert gen.answer.tokens == ["Answer", "."] and gen.answer.text == "Answer."
    assert gen.answer.token_ids == [ID_BASE, ID_BASE + 1] and gen.answer.finish_reason == "stop"
    prompts = len(ids) + len(second["prompt"])
    assert gen.usage == {"prompt_tokens": prompts, "completion_tokens": 4, "total_tokens": prompts + 4}


def test_a_closed_reasoning_whose_answer_was_cut_goes_on_in_a_second_request_within_the_answer_budget(family):
    gen, fake = run(family, [Scripted(("Think", "\n</think>\n\n", "Hel"), "length"), Scripted(("lo",))],
                    think(reasoning_budget=3, answer_tokens=100))
    ids = prompt_ids(family)
    first, second = fake.requests
    assert first["max_tokens"] == 3
    assert second["prompt"] == ids + [ID_BASE, ID_BASE + 1, ID_BASE + 2]  # the reasoning and the close too
    assert second["max_tokens"] == 100 - 1  # one answer token came with the first request
    assert gen.reasoning == "Think" and gen.answer_text == "Hello"
    assert (gen.forced, gen.phases, gen.reasoning_tokens) == (False, 2, 2)
    assert gen.answer.tokens == ["Hel", "lo"] and gen.answer.text == "Hello"
    assert gen.answer.token_ids == [ID_BASE + 2, ID_BASE] and gen.answer.finish_reason == "stop"
    assert gen.answer.token_bytes == [list(b"Hel"), list(b"lo")] and gen.answer.logprobs == [-0.1, -0.1]
    assert gen.answer.raw["usage"] == {  # raw and timings are the last request's
        "prompt_tokens": len(ids) + 3, "completion_tokens": 1, "total_tokens": len(ids) + 4}
    prompts = 2 * len(ids) + 3
    assert gen.usage == {"prompt_tokens": prompts, "completion_tokens": 4, "total_tokens": prompts + 4}


def test_a_reasoning_cut_right_after_its_close_asks_the_whole_answer_in_a_second_request(family):
    # the budget ran out on the close itself, so no answer token came with the first request: the second
    # goes on from every id of the first and may use the whole answer budget
    gen, fake = run(family, [Scripted(("Think", "\n</think>\n\n"), "length"), Scripted(("Answer",))],
                    think(reasoning_budget=2, decoding=GREEDY))
    ids = prompt_ids(family)
    first, second = fake.requests
    assert first["max_tokens"] == 2 and second["prompt"] == ids + [ID_BASE, ID_BASE + 1]
    assert second["max_tokens"] == MAX_TOKENS
    assert gen.reasoning == "Think" and gen.answer_text == "Answer" and not gen.forced
    assert (gen.phases, gen.reasoning_tokens) == (2, 2)
    assert gen.answer.tokens == ["Answer"] and gen.answer.text == "Answer"
    assert gen.answer.token_ids == [ID_BASE] and gen.answer.logprobs == [-0.1]


def test_an_answer_that_already_fills_its_budget_is_not_continued(family):
    gen, fake = run(family, [Scripted(("T", "\n</think>\n\n", "a", "b", "c"), "length")],
                    think(reasoning_budget=5, answer_tokens=3))
    assert len(fake.requests) == 1 and gen.phases == 1 and gen.answer.finish_reason == "length"
    assert gen.answer_text == "abc" and gen.answer.tokens == ["a", "b", "c"]


def test_a_turn_ended_inside_the_reasoning_is_closed_by_force_after_the_terminator_is_dropped(family):
    # llama-server lists the end token it stopped on among the tokens (its text is empty): the model's own
    # terminator cannot precede the forced close, or the answer would be asked after the end of the turn
    ended = Scripted(("\nThinking\n", ""), "stop", ids=(ID_BASE, TURN_END_ID))
    gen, fake = run(family, [ended, Scripted(("Answer",))], think(reasoning_budget=8))
    ids = prompt_ids(family)
    first, second = fake.requests
    assert first["max_tokens"] == 8
    assert second["prompt"] == ids + [ID_BASE] + token_ids(THINK_CLOSE)
    assert gen.forced and gen.phases == 2 and gen.reasoning == "Thinking" and gen.answer_text == "Answer"
    # the reasoning's tokens are the ids the forced close follows: the dropped terminator is not among them
    # (the newlines the recorded reasoning loses are), so with the prompt's ids, the close's and the answer's
    # tokens they make up the step's whole sequence
    assert gen.reasoning_tokens == 1


@pytest.mark.parametrize("special_tags", [False, True], ids=["text", "text_from_ids"])
def test_a_turn_ended_inside_the_reasoning_on_another_end_token_is_closed_without_that_token(tmp_path,
                                                                                         special_tags):
    # llama-server stops on every end-of-generation token of the model, not only on the family's terminator:
    # MiniCPM5's generation config ends a sequence on its eos "</s>" too, Qwen3's on <|endoftext|>. The server
    # lists the one it stopped on last, without text or bytes, and the forced close cannot follow it either
    tokenizer = FakeFamilyTokenizer("hybrid", eos="</s>", special_tags=special_tags)
    family = load_family(model_folder(tmp_path / "eos"), tokenizer=tokenizer,
                         families=[fake_family(tokenizer)])
    assert family.spec.text_from_ids is special_tags and EOS_ID != family.turn_end_id
    ended = Scripted(("\nThinking\n", ""), "stop", ids=(ID_BASE, EOS_ID))
    gen, fake = run(family, [ended, Scripted(("Answer",))], think(reasoning_budget=8))
    _, second = fake.requests
    assert second["prompt"] == prompt_ids(family) + [ID_BASE] + token_ids(THINK_CLOSE)
    assert gen.forced and gen.reasoning == "Thinking" and gen.answer_text == "Answer"
    assert gen.reasoning_tokens == 1  # the end token is none of the reasoning's ids


def test_a_reasoning_keeps_every_id_it_did_not_stop_on_before_a_forced_close(family):
    # a stopped completion whose last token has text did not stop on an end token (a stop string ended it),
    # and a completion the budget cut ends where it was cut, on whatever token that is, a control token too
    gen, fake = run(family, [Scripted(("I keep", " thinking"), "stop"), Scripted(("Answer",))], think())
    assert fake.requests[1]["prompt"] == prompt_ids(family) + [ID_BASE, ID_BASE + 1] + token_ids(THINK_CLOSE)
    assert gen.forced and gen.reasoning_tokens == 2
    control = TOKENS["<|im_start|>"]
    cut = Scripted(("I keep", ""), "length", ids=(ID_BASE, control))
    gen, fake = run(family, [cut, Scripted(("Answer",))], think(reasoning_budget=2))
    assert fake.requests[1]["prompt"] == prompt_ids(family) + [ID_BASE, control] + token_ids(THINK_CLOSE)
    assert gen.forced and gen.reasoning_tokens == 2


def test_a_forced_close_is_the_ids_the_close_has_after_a_word_not_the_ids_it_has_alone(tmp_path):
    # a SentencePiece tokenizer with a dummy prefix (Nanbeige's) starts the close encoded on its own with the
    # word-start marker, which no sample holds after a reasoning, so the close goes as it follows a word
    # (LoadedFamily.encode_close): the model is not asked to answer after a token it never saw there
    tokenizer = DummyPrefix("hybrid", pieces=(MARK,))
    family = load_family(model_folder(tmp_path / "prefix"), tokenizer=tokenizer,
                         families=[fake_family(tokenizer)])
    close = token_ids(THINK_CLOSE, (MARK,))
    assert family.encode_text(THINK_CLOSE) == token_ids(MARK, (MARK,)) + close  # the two differ here
    gen, fake = run(family, [Scripted(("I keep", " thinking"), "length"), Scripted(("Answer",))],
                    think(reasoning_budget=2))
    _, second = fake.requests
    assert second["prompt"] == prompt_ids(family) + [ID_BASE, ID_BASE + 1] + close
    assert gen.forced and gen.reasoning_tokens == 2 and gen.answer_text == "Answer"


def test_a_token_straddling_the_boundary_is_the_answers_first_token(family):
    gen, _ = run(family, [Scripted(("Think", "\n</think>", "\n\nHello", " world"))], think())
    assert gen.reasoning == "Think" and gen.answer_text == "Hello world"  # what follows the close
    # the slice is token-aligned, so the signals place its tokens in its text (agent.route.token_spans)
    assert gen.answer.tokens == ["\n\nHello", " world"] and gen.answer.text == "\n\nHello world"
    assert gen.answer.token_bytes == [list(b"\n\nHello"), list(b" world")]
    assert gen.answer.token_ids == [ID_BASE + 2, ID_BASE + 3] and gen.reasoning_tokens == 2


def test_an_empty_answer_keeps_the_end_token_as_its_own(family):
    ended = Scripted(("Think", "\n</think>\n\n", ""), "stop", ids=(ID_BASE, ID_BASE + 1, TURN_END_ID))
    gen, _ = run(family, [ended], think())
    assert gen.answer_text == "" and gen.answer.tokens == [""] and gen.answer.token_ids == [TURN_END_ID]
    assert gen.reasoning_tokens == 2 and gen.phases == 1


def test_greedy_thinking_sends_no_seed_and_no_sampling_field(family):
    gen, fake = run(family, [Scripted(("T", "\n</think>\n\n", "A"), "length"), Scripted(("B",))],
                    think(reasoning_budget=3, decoding=GREEDY))
    for request in fake.requests:
        assert request["temperature"] == 0.0
        assert not {"seed", "top_p", "top_k", "min_p", "presence_penalty"} & set(request)
    assert gen.seeds is None and gen.phases == 2 and gen.answer_text == "AB"


def test_a_think_generation_needs_the_ids_of_the_tokens_it_generated_and_of_the_ones_listed(family):
    # its reasoning tokens are counted, and a generation goes on, from the server's list of every token it
    # generated (return_tokens) placed against its logprob items by their ids: without either list nothing
    # can be counted exactly, so a reasoning closed within one request is refused too
    for options, message in (({"omit_ids": True}, "token id"), ({"return_tokens": False}, "generated")):
        with pytest.raises(GenerationError, match=message):
            run(family, [Scripted(("cut",), "length"), Scripted(("never",))], think(reasoning_budget=1),
                **options)
        with pytest.raises(GenerationError, match=message):
            run(family, [Scripted(("T", "\n</think>\n\n", "A"))], think(), **options)
    # a NO_THINK generation needs neither: it never goes on and counts no reasoning
    settings = GenerationSettings(mode=Mode.NO_THINK, decoding=GREEDY)
    for options in ({"omit_ids": True}, {"return_tokens": False}):
        gen, _ = run(family, [scripted("A")], settings, **options)
        assert gen.answer_text == "A"


ASKED = ("return_tokens", "verbose", "response_fields")  # what the first request of a THINK generation adds


def test_the_first_request_of_a_think_generation_asks_for_its_tokens_and_no_other_request_does(family):
    gen, fake = run(family, [Scripted(("T", "\n</think>\n\n", "A"), "length"), Scripted(("B",))],
                    think(reasoning_budget=3))
    first, second = fake.requests
    assert {k: first[k] for k in ASKED} == {"return_tokens": True, "verbose": True,
                                            "response_fields": ["tokens", "stop_type"]}
    assert list(first)[-3:] == list(ASKED)  # after every field the request had before
    assert not set(ASKED) & set(second) and gen.answer_text == "AB"
    _, fake = run(family, [Scripted(("cut",), "length"), Scripted(("B",))], think(reasoning_budget=1))
    assert set(ASKED) <= set(fake.requests[0]) and not set(ASKED) & set(fake.requests[1])
    _, fake = run(family, [scripted("A")], GenerationSettings(mode=Mode.NO_THINK, decoding=SAMPLING))
    assert not set(ASKED) & set(fake.requests[0])


# ---- tokens llama-server leaves out of its list (module docstring of llm.generation: Token ids) ------------

ZHONG = "中".encode()  # three bytes; the tokens below split them two and one


def test_the_server_lists_no_token_after_which_the_text_ends_inside_a_character():
    # b11349's process_token: the first part of 中 leaves the text inside the character and is not listed;
    # the second part is, with the bytes and the text of both; every id is in the token list it returns
    fake = FakeLlama([Scripted(("我", ZHONG[:2], ZHONG[2:], "文"), "length")])
    completion = fake.client().complete([1, 2], lora_scale=0.0, return_tokens=True)
    assert completion.text == "我中文" and completion.tokens == ["我", "中", "文"]
    assert completion.token_ids == [ID_BASE, ID_BASE + 2, ID_BASE + 3]
    assert completion.token_bytes == [list("我".encode()), list(ZHONG), list("文".encode())]
    assert completion.generated_ids == [ID_BASE, ID_BASE + 1, ID_BASE + 2, ID_BASE + 3]
    assert completion.stop_type == "limit" and completion.raw["usage"]["completion_tokens"] == 4


def test_a_forced_close_goes_on_from_every_token_generated_the_ones_left_out_of_the_list_too(family):
    gen, fake = run(family, [Scripted(("我想", ZHONG[:2], ZHONG[2:], "文"), "length"), Scripted(("Answer",))],
                    think(reasoning_budget=4))
    _, second = fake.requests
    assert second["prompt"] == prompt_ids(family) + [ID_BASE + i for i in range(4)] + token_ids(THINK_CLOSE)
    assert gen.forced and gen.reasoning == "我想中文" and gen.answer_text == "Answer"
    assert gen.reasoning_tokens == 4  # the part of 中 the server did not list among them


def test_the_reasoning_tokens_of_a_closed_reasoning_count_the_tokens_left_out_of_the_list(family):
    gen, fake = run(family, [Scripted(("想", ZHONG[:2], ZHONG[2:], "\n</think>", "\n\n", "好", ""), "stop",
                                      ids=(*range(ID_BASE, ID_BASE + 6), TURN_END_ID))], think())
    assert len(fake.requests) == 1 and gen.phases == 1 and not gen.forced
    assert gen.reasoning == "想中" and gen.answer_text == "好"
    assert gen.reasoning_tokens == 5  # 想, the two parts of 中, the close and the newlines after it
    assert gen.answer.tokens == ["好", ""] and gen.answer.token_ids == [ID_BASE + 5, TURN_END_ID]


def test_listed_tokens_with_one_id_stand_each_at_its_own_place_among_the_generated_ones(family):
    # the two newlines after the close are one token id twice: the second is placed after the first
    gen, _ = run(family, [Scripted(("想", ZHONG[:2], ZHONG[2:], "\n</think>", "\n", "\n", "好"), "stop",
                                   ids=(10, 11, 12, 13, 14, 14, 15))], think())
    assert gen.reasoning == "想中" and gen.answer_text == "好" and gen.reasoning_tokens == 6


def test_an_answer_the_budget_cut_goes_on_from_every_token_within_the_answer_budget(family):
    cut = Scripted(("T", "\n</think>\n\n", "答", ZHONG[:2], ZHONG[2:], "a"), "length")
    gen, fake = run(family, [cut, Scripted(("b",))], think(reasoning_budget=6, answer_tokens=10))
    _, second = fake.requests
    assert second["prompt"] == prompt_ids(family) + [ID_BASE + i for i in range(6)]
    assert second["max_tokens"] == 10 - 4  # 答, the two parts of 中 and a came with the first request
    assert gen.reasoning == "T" and gen.answer_text == "答中ab" and gen.reasoning_tokens == 2
    assert gen.answer.tokens == ["答", "中", "a", "b"] and gen.answer.text == "答中ab"


def test_an_answer_the_budget_cut_inside_a_character_goes_on_from_its_ids_its_text_marked_where_it_was_cut(
        family):
    # the ids go on exactly; the text cannot join whole there (module docstring, 2): the server writes the
    # cut character's first bytes, left out of the list, as U+FFFD, and the second request's last byte of it
    # alone as U+FFFD too
    cut = Scripted(("T", "\n</think>\n\n", "答", ZHONG[:2]), "length")
    gen, fake = run(family, [cut, Scripted((ZHONG[2:], "文"))], think(reasoning_budget=4, answer_tokens=10))
    assert fake.requests[1]["prompt"] == prompt_ids(family) + [ID_BASE + i for i in range(4)]
    assert fake.requests[1]["max_tokens"] == 10 - 2 and gen.reasoning_tokens == 2
    assert gen.answer_text == "答��文" and gen.answer.token_ids == [ID_BASE + 2, ID_BASE, ID_BASE + 1]


def test_an_end_token_after_a_cut_character_is_dropped_before_a_forced_close_though_it_is_not_listed(family):
    # the end token leaves the text inside 中 too, so the server lists neither it nor the part before it, and
    # reports the stop as "eos": the close follows every token generated but the end token
    ended = Scripted(("思", ZHONG[:2], ""), "stop", ids=(ID_BASE, ID_BASE + 1, TURN_END_ID))
    gen, fake = run(family, [ended, Scripted(("Answer",))], think())
    assert fake.requests[1]["prompt"] == prompt_ids(family) + [ID_BASE, ID_BASE + 1] + token_ids(THINK_CLOSE)
    assert gen.forced and gen.reasoning_tokens == 2 and gen.answer_text == "Answer"


def test_a_stop_string_the_model_spelled_inside_its_reasoning_is_dropped_before_a_forced_close(family):
    # the server stops on the string (stop type "word"), erases it from the text and trims its tokens from
    # the list: the close follows the tokens up to the last one listed
    spelled = Scripted(("I keep", " thinking", "<|im_end|>"), "stop", trimmed=1)
    gen, fake = run(family, [spelled, Scripted(("Answer",))], think())
    assert fake.requests[1]["prompt"] == prompt_ids(family) + [ID_BASE, ID_BASE + 1] + token_ids(THINK_CLOSE)
    assert gen.forced and gen.reasoning == "I keep thinking" and gen.reasoning_tokens == 2


def test_a_stop_string_the_model_spelled_first_leaves_the_close_after_no_generated_token(family):
    # the string came before any other token: the server trims every token it listed, and the close
    # follows none of the generated ones, the string's own dropped with the rest of the list
    spelled = Scripted(("<|im_end|>",), "stop", trimmed=1)
    gen, fake = run(family, [spelled, Scripted(("Answer",))], think())
    assert fake.requests[0]["return_tokens"] and fake.requests[1]["prompt"] == prompt_ids(family) + token_ids(
        THINK_CLOSE)
    assert gen.forced and gen.reasoning == "" and gen.reasoning_tokens == 0 and gen.answer_text == "Answer"


HAN3 = tuple(bytes([b]) for b in "中".encode())  # 中 split in three, one byte per token
SMILE4 = tuple(bytes([b]) for b in "😀".encode())  # a four-byte character split in four


def test_a_character_split_in_three_or_four_tokens_is_listed_once_and_every_part_is_counted(family):
    # the server lists the last part alone, with every byte of the character; the parts before it are left
    # out, and the reasoning that holds them counts them all
    fake = FakeLlama([Scripted(("想", *HAN3, *SMILE4, "了"), "length")])
    completion = fake.client().complete([1], lora_scale=0.0, return_tokens=True)
    assert completion.tokens == ["想", "中", "😀", "了"] and completion.text == "想中😀了"
    assert completion.token_ids == [ID_BASE, ID_BASE + 3, ID_BASE + 7, ID_BASE + 8]
    assert completion.generated_ids == [ID_BASE + i for i in range(9)]
    closed = Scripted(("想", *HAN3, *SMILE4, "\n</think>\n\n", "好"), "length")
    gen, fake = run(family, [closed, Scripted(("!",))], think(reasoning_budget=10, answer_tokens=5))
    assert gen.reasoning == "想中😀" and gen.reasoning_tokens == 9 and gen.answer_text == "好!"
    assert fake.requests[1]["prompt"] == prompt_ids(family) + [ID_BASE + i for i in range(10)]
    assert fake.requests[1]["max_tokens"] == 5 - 1
    forced = Scripted(("想", *HAN3, *SMILE4), "length")
    gen, fake = run(family, [forced, Scripted(("Answer",))], think(reasoning_budget=8))
    assert fake.requests[1]["prompt"] == (prompt_ids(family) + [ID_BASE + i for i in range(8)]
                                          + token_ids(THINK_CLOSE))
    assert gen.forced and gen.reasoning == "想中😀" and gen.reasoning_tokens == 8


def test_an_answer_that_starts_with_a_split_character_counts_its_first_part_as_the_answers(family):
    # the part of 中 left out of the list belongs to the answer's first listed token, which carries its bytes
    ended = Scripted(("想", "\n</think>\n\n", ZHONG[:2], ZHONG[2:], "文", ""), "stop",
                     ids=(*range(ID_BASE, ID_BASE + 5), TURN_END_ID))
    gen, _ = run(family, [ended], think(reasoning_budget=6))
    assert gen.reasoning == "想" and gen.answer_text == "中文" and gen.reasoning_tokens == 2
    assert gen.answer.tokens == ["中", "文", ""]
    assert gen.answer.token_ids == [ID_BASE + 3, ID_BASE + 4, TURN_END_ID]


def test_a_token_that_straddles_the_close_and_the_answer_is_the_answers_first_after_a_split_character(family):
    # 中's last part and the close in one token, then the close's end and the answer's start in one token:
    # the reasoning is every token before the straddling one, the part left out of the list among them
    tokens = ("想", ZHONG[:2], ZHONG[2:] + b"\n</th", "ink>\n\n好", "的", "")
    gen, _ = run(family, [Scripted(tokens, "stop", ids=(*range(ID_BASE, ID_BASE + 5), TURN_END_ID))],
                 think(reasoning_budget=6))
    assert gen.reasoning == "想中" and gen.answer_text == "好的" and gen.reasoning_tokens == 3
    assert gen.answer.tokens == ["ink>\n\n好", "的", ""] and gen.answer.text == "ink>\n\n好的"


def sequence_identity(family, gen, fake, mode: Mode = Mode.THINK) -> None:
    """The step's whole sequence, as Generation's docstring has it (the prompt's ids, the reasoning tokens, a
    forced close's ids and the answer tokens), is the last request's prompt and every token it generated."""
    close = len(family.encode_close()) if gen.forced else 0
    last = fake.requests[-1]
    assert gen.answer_tokens <= gen.usage["completion_tokens"]  # the usage sums every request's tokens
    last_generated = gen.answer.raw["usage"]["completion_tokens"]  # the last request's raw response
    assert len(prompt_ids(family, mode)) + gen.reasoning_tokens + close + gen.answer_tokens == (
        len(last["prompt"]) + last_generated)


def test_the_answer_tokens_count_every_token_generated_after_the_reasoning_the_unlisted_ones_too(family):
    # b11349 lists a character split across tokens once, as its last part (Token ids): the answer's listed
    # tokens can be fewer than the tokens it generated, which a step's length counts (scripts/run_e2e.py)
    closed = Scripted(("想", "\n</think>\n\n", ZHONG[:2], ZHONG[2:], "文", ""), "stop",
                      ids=(*range(ID_BASE, ID_BASE + 5), TURN_END_ID))
    gen, fake = run(family, [closed], think())
    assert gen.answer_text == "中文" and gen.reasoning_tokens == 2
    assert len(gen.answer.tokens) == 3 and gen.answer_tokens == 4  # the first part of 中 is not listed
    sequence_identity(family, gen, fake)
    # closed by force: the answer is the second request's, every token it generated
    gen, fake = run(family, [Scripted(("cut",), "length"), Scripted((ZHONG[:2], ZHONG[2:], "文"))],
                    think(reasoning_budget=1))
    assert gen.forced and len(gen.answer.tokens) == 2 and gen.answer_tokens == 3
    sequence_identity(family, gen, fake)
    # an answer that goes on: the first request's tokens after the reasoning and the second's
    gen, fake = run(family, [Scripted(("T", "\n</think>\n\n", "A"), "length"),
                             Scripted((ZHONG[:2], ZHONG[2:]))],
                    think(reasoning_budget=3, answer_tokens=10))
    assert gen.phases == 2 and gen.answer_text == "A中" and len(gen.answer.tokens) == 2
    assert gen.answer_tokens == 3
    sequence_identity(family, gen, fake)
    # one that does not: the tokens after the reasoning, the end token among them
    gen, fake = run(family, [Scripted(("T", "\n</think>\n\n", "a", "b"), "length")],
                    think(reasoning_budget=4, answer_tokens=2))
    assert gen.phases == 1 and gen.answer_tokens == 2 and gen.reasoning_tokens == 2
    # a NO_THINK generation: every token its one request generated
    settings = GenerationSettings(mode=Mode.NO_THINK, decoding=GREEDY)
    gen, fake = run(family, [Scripted((ZHONG[:2], ZHONG[2:], "文", ""), "stop",
                                      ids=(ID_BASE, ID_BASE + 1, ID_BASE + 2, TURN_END_ID))], settings)
    assert gen.answer_text == "中文" and len(gen.answer.tokens) == 3 and gen.answer_tokens == 4
    sequence_identity(family, gen, fake, Mode.NO_THINK)


def test_an_answer_whose_tokens_the_server_did_not_count_is_none_without_thinking_and_refused_with_it(family):
    # the count of a request whose generated ids were not asked comes from its usage: without it a NO_THINK
    # generation, which records no thinking, leaves the count unknown, and a THINK generation, whose step
    # length needs it, is refused
    uncounted = dataclasses.replace(served(["好", "的"], None, [1, 2], "好的"), raw={})
    gen = generate(ServedAsIs([uncounted]), family, prompt_ids(family, Mode.NO_THINK),
                   GenerationSettings(mode=Mode.NO_THINK, decoding=GREEDY), lora_scale=0.0)
    assert gen.answer_text == "好的" and gen.answer_tokens is None
    cut = served(["cut"], None, [7], "cut", generated=[7], stop_type="limit")
    with pytest.raises(GenerationError, match="completion tokens"):
        generate(ServedAsIs([cut, uncounted]), family, prompt_ids(family), think(), lora_scale=0.0)
    closed = served(["T", "\n</think>\n\n", "A"], None, [7, 8, 9], "T\n</think>\n\nA", generated=[7, 8, 9],
                    stop_type="limit")
    closed = dataclasses.replace(closed, finish_reason="length")
    with pytest.raises(GenerationError, match="completion tokens"):
        generate(ServedAsIs([closed, uncounted]), family, prompt_ids(family), think(reasoning_budget=3),
                 lora_scale=0.0)


def test_a_budget_that_cuts_a_character_of_the_reasoning_closes_it_after_every_part_generated(family):
    # the split at the phase boundary: the first request ends inside 中, its last part left out of the list
    # (the text shows the cut character as U+FFFD, as the server's JSON writer does), and the forced close
    # follows the part all the same
    gen, fake = run(family, [Scripted(("我想", ZHONG[:2]), "length"), Scripted(("Answer",))],
                    think(reasoning_budget=2))
    assert fake.requests[1]["prompt"] == (prompt_ids(family) + [ID_BASE, ID_BASE + 1]
                                          + token_ids(THINK_CLOSE))
    assert gen.forced and gen.reasoning_tokens == 2 and gen.reasoning == "我想�"


def test_the_second_request_after_a_forced_close_is_the_request_it_was_before_its_ids_aside(family):
    # the bytes of a continuation are the bytes HEAD sent: the prompt's ids, max_tokens, the decoding, the
    # stop strings and the seed, nothing asked for the generated tokens
    gen, fake = run(family, [Scripted(("我想", ZHONG[:2], ZHONG[2:]), "length"), Scripted(("Answer",))],
                    think(reasoning_budget=3))
    seed = request_seed(0, prompt_ids(family), 2, 0)
    body = {"prompt": prompt_ids(family) + [ID_BASE + i for i in range(3)] + token_ids(THINK_CLOSE),
            "max_tokens": 512, "temperature": 0.6, "cache_prompt": False, "logprobs": 5,
            "stop": ["<|im_end|>"], "seed": seed, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
            "presence_penalty": 0.0}
    assert fake.request_bytes[1] == json.dumps(body, separators=(",", ":")).encode()
    assert gen.seeds == [request_seed(0, prompt_ids(family), 1, 0), seed]


CALL_TARGET = {"role": "assistant", "content": "", "tool_calls": [{
    "id": "call_0", "type": "function",
    "function": {"name": "search_arxiv", "arguments": '{"query": "KV cache"}'}}]}


def test_a_trimming_familys_forced_close_and_split_agree_with_its_training_samples(tmp_path):
    # Granite's kind of template (family_fakes' "trim" style): the THINK sample of a call target holds the
    # reasoning, the close and the call with one newline between them. A forced close appends the close's ids
    # to the reasoning's, which is the sample's own beginning (two newlines there would be one token, "\n\n",
    # the close's last newline another), and a generation that writes the sample's completion is read back as
    # its reasoning and the answer the NO_THINK sample holds, without the newline after the close
    family = fake_loaded_family(tmp_path / "model", "trim")
    reasoning = "I search arXiv"
    sample = render_family_sample(family, CONTEXT, CALL_TARGET, tools=MODEL_TOOLS, reasoning=reasoning)
    answer = render_family_sample(family, CONTEXT, CALL_TARGET, tools=MODEL_TOOLS)["completion"]
    assert sample["mode"] == "think" and sample["completion"] == reasoning + "\n</think>" + answer
    ids, start = sample["input_ids"], sample["completion_start"]
    _, fake = run(family, [Scripted(tuple(reasoning), "length", ids=tuple(token_ids(reasoning))),
                           Scripted(("x",))], think(reasoning_budget=len(reasoning)))
    continued = fake.requests[1]["prompt"]
    assert continued == ids[:len(continued)] and continued[-3:] == family.encode_close()
    generated = ids[start:]
    pieces = tuple("" if i == TURN_END_ID else family.decode([i]) for i in generated)
    gen, _ = run(family, [Scripted(pieces, "stop", ids=tuple(generated))], think(reasoning_budget=200))
    assert gen.reasoning == reasoning and not gen.forced and gen.phases == 1
    assert gen.answer_text == answer.lstrip("\n").removesuffix("<|im_end|>")
    assert family.parse(gen.answer_text, MODEL_TOOLS) == [{"name": "search_arxiv",
                                                          "arguments": {"query": "KV cache"}}]
    assert gen.reasoning_tokens == len(token_ids(reasoning + "\n</think>\n"))


def test_a_server_whose_lists_disagree_is_refused(family):
    # the logprob list must be a subsequence of the generated tokens, and these as many as the usage counts
    text = "T\n</think>\n\nA"
    for generated, count, message in (([7, 9, 8], None, "not among"), ([7, 8, 9], 4, "completion tokens")):
        client = ServedAsIs([served(["T", "\n</think>\n\n", "A"], None, [7, 8, 9], text, generated=generated,
                                    stop_type="eos", completion_tokens=count)])
        with pytest.raises(GenerationError, match=message):
            generate(client, family, prompt_ids(family), think(), lora_scale=0.0)


@pytest.mark.parametrize("stop_type", [None, "none", "length", "EOS"])
def test_a_think_generation_needs_a_stop_type_the_server_writes_with_the_ids_it_generated(family, stop_type):
    # the stop type tells a turn the model ended (its end token, the last generated, is dropped before a
    # forced close) from a budget cut (every token kept): without one of "eos", "word" and "limit" neither the
    # close nor the count could be placed, so the first request is refused before a second is sent, whether
    # the reasoning was closed or not (b11349 writes "none" for a stop it has no type for)
    closed = served(["T", "\n</think>\n\n", "A"], None, [7, 8, 9], "T\n</think>\n\nA", generated=[7, 8, 9],
                    stop_type=stop_type)
    ended = served(["I think", ""], [b"I think", b""], [7, TURN_END_ID], "I think",
                   generated=[7, TURN_END_ID], stop_type=stop_type)
    for first in (closed, ended):
        client = ServedAsIs([first, served(["Answer"], None, [10], "Answer")])
        with pytest.raises(GenerationError, match="stop type"):
            generate(client, family, prompt_ids(family), think(), lora_scale=0.0)
        assert len(client.requests) == 1


def test_a_mode_the_family_does_not_have_is_refused_before_any_request(tmp_path):
    never = fake_loaded_family(tmp_path / "none", "none")
    fake = FakeLlama([scripted("x")])
    with pytest.raises(ValueError, match="has no think mode"):
        generate(fake.client(), never, [1, 2, 3], think(), lora_scale=0.0)
    assert fake.requests == []


# ---- NO_THINK and a family that never thinks: one request, today's bytes ---------------------------------


def test_a_greedy_no_think_generation_sends_todays_request_body_with_the_prompt_as_ids(family):
    settings = GenerationSettings(mode=Mode.NO_THINK, decoding=GREEDY)
    gen, fake = run(family, [scripted(CALL)], settings, lora_scale=1.0)
    ids = prompt_ids(family, Mode.NO_THINK)
    assert fake.request_bytes == [TODAY_BODY % json.dumps(ids, separators=(",", ":")).encode()]
    assert (gen.reasoning, gen.seeds, gen.forced) == (None, None, False)
    assert (gen.phases, gen.reasoning_tokens) == (1, 0)
    assert gen.answer_text == CALL and isinstance(gen.answer, Completion)
    assert (gen.answer.text, gen.answer.tokens, gen.answer.token_ids) == (CALL, [CALL], [ID_BASE])
    assert gen.usage == gen.answer.raw["usage"] == {"prompt_tokens": len(ids), "completion_tokens": 1,
                                                    "total_tokens": len(ids) + 1}


def test_a_sampled_no_think_generation_sends_its_seed_and_every_sampling_field(family):
    settings = GenerationSettings(mode=Mode.NO_THINK, decoding=SAMPLING, base_seed=7, sample=2)
    gen, fake = run(family, [scripted("好的。")], settings)
    (request,) = fake.requests
    ids = prompt_ids(family, Mode.NO_THINK)
    assert request["seed"] == request_seed(7, ids, 1, 2) and gen.seeds == [request["seed"]]
    assert request["top_p"] == 0.95 and request["top_k"] == 20 and request["presence_penalty"] == 0.0
    assert request["max_tokens"] == MAX_TOKENS and gen.reasoning is None


def test_a_family_that_never_thinks_makes_one_request(tmp_path):
    never = fake_loaded_family(tmp_path / "none", "none")
    fake = FakeLlama([scripted("好的。")])
    settings = GenerationSettings(mode=Mode.NO_THINK, decoding=GREEDY, answer_tokens=64)
    gen = generate(fake.client(), never, [1, 2, 3], settings, lora_scale=0.0)
    (request,) = fake.requests
    assert request["prompt"] == [1, 2, 3] and request["max_tokens"] == 64 and "seed" not in request
    assert gen.answer_text == "好的。" and gen.reasoning is None and gen.phases == 1


# ---- text from ids: a family whose tags are control tokens (MiniCPM5, LFM2.5) ----------------------------

CALL_JSON = '{"name": "search_arxiv", "arguments": {"query": "KV cache"}}'
PIECES = ("Let me think", CALL_JSON)  # read as one token each by the tokenizer below
NEWLINE = CHAR_OFFSET + ord("\n")
CLOSE_ID, OPEN_ID, END_ID = (TOKENS[t] for t in ("</think>", "<tool_call>", "</tool_call>"))
# the call as the server sends it when the tags are control tokens: no text and no bytes for them, nor for
# the end token
SERVED_CALL = (("", "\n", CALL_JSON, "\n", "", ""),
               (OPEN_ID, NEWLINE, PIECE_OFFSET + 1, NEWLINE, END_ID, TURN_END_ID))
CALL_TEXT = "<tool_call>\n" + CALL_JSON + "\n</tool_call>"


def special_family(tmp_path):
    """A family whose thinking and tool-call tags are special tokens, so it reads its text from the ids."""
    tokenizer = FakeFamilyTokenizer("hybrid", special_tags=True, pieces=PIECES)
    family = load_family(model_folder(tmp_path / "special"), tokenizer=tokenizer,
                         families=[fake_family(tokenizer)])
    assert family.spec.text_from_ids
    return family


def test_the_control_tokens_of_such_a_family_are_read_from_the_ids_the_server_sends(tmp_path):
    family = special_family(tmp_path)
    served = Scripted(("Let me think", "\n", "", "\n\n", *SERVED_CALL[0]), "stop",
                      ids=(PIECE_OFFSET, NEWLINE, CLOSE_ID, TOKENS["\n\n"], *SERVED_CALL[1]))
    gen, fake = run(family, [served], think(reasoning_budget=16, decoding=GREEDY))
    assert len(fake.requests) == 1 and gen.phases == 1 and not gen.forced
    assert gen.reasoning == "Let me think" and gen.answer_text == CALL_TEXT
    # the answer's tokens hold the tags' text and bytes; the end token stays empty, as on the text path
    assert gen.answer.tokens == ["<tool_call>", "\n", CALL_JSON, "\n", "</tool_call>", ""]
    assert gen.answer.text == CALL_TEXT and gen.answer.token_ids == list(SERVED_CALL[1])
    assert gen.answer.token_bytes == [list(t.encode("utf-8")) for t in gen.answer.tokens]
    assert gen.answer.logprobs == [-0.1] * 6 and gen.reasoning_tokens == 4
    assert family.parse(gen.answer_text, MODEL_TOOLS) == [{"name": "search_arxiv",
                                                            "arguments": {"query": "KV cache"}}]
    # one decode per control token, the end token left alone
    assert [ids for ids, _ in family.tokenizer.decoded] == [[CLOSE_ID], [OPEN_ID], [END_ID]]
    assert all(kwargs == {"skip_special_tokens": False} for _, kwargs in family.tokenizer.decoded)


def test_a_forced_close_of_such_a_family_reads_the_second_request_from_its_ids_too(tmp_path):
    family = special_family(tmp_path)
    cut = Scripted(("I keep", " thinking"), "length")
    gen, fake = run(family, [cut, Scripted(SERVED_CALL[0], "stop", ids=SERVED_CALL[1])],
                    think(reasoning_budget=2, decoding=GREEDY))
    ids = prompt_ids(family)
    first, second = fake.requests
    assert first["max_tokens"] == 2
    assert second["prompt"] == ids + [ID_BASE, ID_BASE + 1] + token_ids(THINK_CLOSE)
    assert gen.forced and gen.reasoning == "I keep thinking" and gen.answer_text == CALL_TEXT
    assert gen.answer.tokens[0] == "<tool_call>" and gen.answer.tokens[-1] == ""


def test_a_no_think_generation_of_such_a_family_reads_its_one_request_from_the_ids(tmp_path):
    family = special_family(tmp_path)
    settings = GenerationSettings(mode=Mode.NO_THINK, decoding=GREEDY)
    gen, fake = run(family, [Scripted(SERVED_CALL[0], "stop", ids=SERVED_CALL[1])], settings)
    (request,) = fake.requests
    assert request["prompt"] == prompt_ids(family, Mode.NO_THINK) and "seed" not in request
    assert gen.answer_text == CALL_TEXT and gen.reasoning is None and gen.phases == 1
    assert gen.answer.text == CALL_TEXT and gen.answer.tokens[0] == "<tool_call>"


def test_such_a_family_needs_an_id_per_token_before_any_text_is_read(tmp_path):
    family = special_family(tmp_path)
    settings = GenerationSettings(mode=Mode.NO_THINK, decoding=GREEDY)
    with pytest.raises(GenerationError, match="text cannot be read"):
        run(family, [Scripted(SERVED_CALL[0], "stop")], settings, omit_ids=True)


class ServedAsIs:
    """A client whose every complete() answers the next Completion as given, the fields set by hand where
    FakeLlama derives them from the scripted tokens: a list that disagrees with itself, tokens without
    bytes."""

    def __init__(self, completions: list[Completion]) -> None:
        self.completions = list(completions)
        self.requests: list[dict] = []

    def complete(self, prompt: list[int], **request) -> Completion:
        self.requests.append({"prompt": prompt, **request})
        return self.completions.pop(0)


def served(tokens: list[str], token_bytes: list[bytes] | None, ids: list[int], text: str, *,
           generated: list[int] | None = None, stop_type: str | None = None,
           completion_tokens: int | None = None) -> Completion:
    """A completion as the server sends it: text is its own text, which holds every character whole; the
    ids of the tokens it generated and its stop type when given (a THINK generation's first request asks for
    them), its usage counting them (or completion_tokens)."""
    n = len(tokens)
    count = completion_tokens if completion_tokens is not None else len(generated if generated else tokens)
    return Completion(text=text, tokens=tokens, logprobs=[-0.1] * n, top_logprobs=[{}] * n,
                      finish_reason="stop", raw={"usage": {"completion_tokens": count}},
                      token_bytes=None if token_bytes is None else [list(b) for b in token_bytes],
                      token_ids=ids, generated_ids=generated, stop_type=stop_type)


HAN = "中".encode()  # three bytes; the tokens below split them two and one
QUERY = '{"name": "search_arxiv", "arguments": {"query": "KV '


def test_the_text_of_such_a_family_is_its_tokens_bytes_so_a_character_split_between_tokens_stays_whole(
        tmp_path):
    # the server lists the second part of 中 alone, with the bytes of both parts (the first left the text
    # inside the character: FakeLlama's module docstring); the control tokens around it have no bytes
    family = special_family(tmp_path)
    split = Scripted(("", "\n", QUERY, HAN[:2], HAN[2:], '"}}', "\n", "", ""), "stop",
                     ids=(OPEN_ID, NEWLINE, 60_001, 60_002, 60_003, 60_004, NEWLINE, END_ID, TURN_END_ID))
    settings = GenerationSettings(mode=Mode.NO_THINK, decoding=GREEDY)
    gen, _ = run(family, [split], settings)
    call = '<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "KV 中"}}\n</tool_call>'
    assert gen.answer_text == gen.answer.text == call
    assert family.parse(gen.answer_text, MODEL_TOOLS) == [{"name": "search_arxiv",
                                                            "arguments": {"query": "KV 中"}}]
    # the control tokens take their text and bytes; the listed part of the character keeps what the server
    # sent, every byte of the character
    assert gen.answer.tokens == ["<tool_call>", "\n", QUERY, "中", '"}}', "\n", "</tool_call>", ""]
    assert gen.answer.token_ids == [OPEN_ID, NEWLINE, 60_001, 60_003, 60_004, NEWLINE, END_ID, TURN_END_ID]
    assert bytes(b for data in gen.answer.token_bytes for b in data).decode("utf-8") == call
    spans = token_spans(gen.answer.text, gen.answer.tokens, gen.answer.token_bytes)
    han = call.index("中")
    assert spans[2][1] == han and spans[3] == (han, han + 1)
    assert spans[0] == (0, len("<tool_call>")) and spans[-1] == (len(call), len(call))


def test_such_a_family_needs_the_bytes_of_every_token_to_place_the_text_of_its_control_tokens(tmp_path):
    family = special_family(tmp_path)
    client = ServedAsIs([served(["", "\n", CALL_JSON, "\n", "", ""], None, list(SERVED_CALL[1]),
                                "\n" + CALL_JSON + "\n")])
    settings = GenerationSettings(mode=Mode.NO_THINK, decoding=GREEDY)
    with pytest.raises(GenerationError, match="bytes"):
        generate(client, family, prompt_ids(family, Mode.NO_THINK), settings, lora_scale=0.0)


def test_the_end_token_a_stopped_completion_ends_on_keeps_its_empty_text_whichever_it_is(tmp_path):
    # MiniCPM5: its turns end with <|im_end|> but its eos is </s>, and llama-server stops on either; the one
    # it stopped on comes last, without text, as on the text path, whereas a control token a completion cut
    # short ends with is the model's text
    tokenizer = FakeFamilyTokenizer("hybrid", eos="</s>", special_tags=True, pieces=PIECES)
    family = load_family(model_folder(tmp_path / "eos"), tokenizer=tokenizer,
                         families=[fake_family(tokenizer)])
    assert family.spec.text_from_ids and tokenizer.eos_token_id == EOS_ID != family.turn_end_id
    settings = GenerationSettings(mode=Mode.NO_THINK, decoding=GREEDY)
    ids = (*SERVED_CALL[1][:-1], EOS_ID)
    gen, _ = run(family, [Scripted(SERVED_CALL[0], "stop", ids=ids)], settings)
    assert gen.answer_text == gen.answer.text == CALL_TEXT
    assert gen.answer.tokens[-1] == "" and gen.answer.token_bytes[-1] == []
    assert EOS_ID not in [i for decoded, _ in tokenizer.decoded for i in decoded]
    cut = Scripted(("", "\n", CALL_JSON, "\n", ""), "length",
                   ids=(OPEN_ID, NEWLINE, PIECE_OFFSET + 1, NEWLINE, END_ID))
    gen, _ = run(family, [cut], settings)
    assert gen.answer_text == CALL_TEXT and gen.answer.tokens[-1] == "</tool_call>"


def test_the_end_token_a_forced_close_drops_is_the_one_the_server_says_it_stopped_on(family):
    # the stop type decides, not the list: a completion that stopped on an end token ("eos") loses its last
    # generated token, bytes or none; one the budget cut ("limit") keeps every token, an empty one too
    client = ServedAsIs([served(["\nThinking\n", ""], None, [ID_BASE, EOS_ID], "\nThinking\n",
                                generated=[ID_BASE, EOS_ID], stop_type="eos"),
                         served(["Answer"], None, [ID_BASE], "Answer")])
    gen = generate(client, family, prompt_ids(family), think(), lora_scale=0.0)
    assert client.requests[1]["prompt"] == prompt_ids(family) + [ID_BASE] + token_ids(THINK_CLOSE)
    assert gen.forced and gen.reasoning_tokens == 1 and gen.answer_text == "Answer"
    client = ServedAsIs([served(["\nThinking\n", ""], None, [ID_BASE, EOS_ID], "\nThinking\n",
                                generated=[ID_BASE, EOS_ID], stop_type="limit"),
                         served(["Answer"], None, [ID_BASE], "Answer")])
    gen = generate(client, family, prompt_ids(family), think(), lora_scale=0.0)
    assert client.requests[1]["prompt"] == prompt_ids(family) + [ID_BASE, EOS_ID] + token_ids(THINK_CLOSE)
    assert gen.forced and gen.reasoning_tokens == 2


def test_a_family_whose_tags_are_text_never_decodes_and_keeps_the_servers_completion(family):
    ended = Scripted(("Think", "\n</think>\n\n", "A", ""), "stop", ids=(ID_BASE, ID_BASE + 1, ID_BASE + 2,
                                                                        TURN_END_ID))
    gen, _ = run(family, [ended], think())
    assert not family.spec.text_from_ids and family.tokenizer.decoded == []
    assert gen.answer_text == "A" and gen.answer.tokens == ["A", ""] and gen.answer.text == "A"
    assert gen.answer.token_bytes == [list(b"A"), []]


# ---- a real tokenizer: Qwen3-4B-Thinking-2507 reasoning in Chinese --------------------------------------

THINKING_2507 = Path(__file__).resolve().parents[2] / "models" / "Qwen3-4B-Thinking-2507"
ZH_REASONING = "用户想找关于知识蒸馏的论文，我先用 search_arxiv 检索，再挑最相关的一篇看摘要。"
ZH_ANSWER = "我用 search_arxiv 检索到两篇关于知识蒸馏的论文。"
ZH_CUT = "我用 search_arxiv 检索"  # where the budget cuts the answer below, after a character split in three


def byte_of() -> dict[str, int]:
    """The byte each character of a byte-level BPE token stands for: GPT-2's bytes_to_unicode, reversed."""
    printable = [*range(ord("!"), ord("~") + 1), *range(ord("¡"), ord("¬") + 1),
                 *range(ord("®"), ord("ÿ") + 1)]
    others = [b for b in range(256) if b not in printable]
    return {**{chr(b): b for b in printable}, **{chr(256 + n): b for n, b in enumerate(others)}}


def server_pieces(tokenizer, ids: list[int]) -> list[bytes]:
    """The bytes llama-server writes for each id of a byte-level BPE vocabulary (llama-vocab.cpp,
    token_to_piece without special tokens): none for a special added token (a control token in the GGUF), an
    added token's own text, else the bytes the token's characters stand for."""
    added, table = tokenizer.added_tokens_decoder, byte_of()
    return [(b"" if added[i].special else added[i].content.encode("utf-8")) if i in added
            else bytes(table[c] for c in token)
            for i, token in zip(ids, tokenizer.convert_ids_to_tokens(ids), strict=True)]


@pytest.mark.skipif(not (THINKING_2507 / "tokenizer.json").is_file(),
                    reason="local model files are not present")
def test_a_real_chinese_reasoning_is_counted_and_continued_from_every_token_the_server_generated():
    family = load_family(THINKING_2507)
    assert family.spec.name == "qwen3-thinking-2507"
    ids = prompt_ids(family)
    reasoning = family.encode_text(ZH_REASONING)
    pieces = server_pieces(family.tokenizer, reasoning)
    assert b"".join(pieces) == ZH_REASONING.encode()
    listed = Scripted(tuple(pieces), "length", ids=tuple(reasoning)).listed()
    assert len(listed) < len(reasoning)  # some are left out of the list

    # cut by the budget and closed by force: the close follows every token generated
    gen, fake = run(family, [Scripted(tuple(pieces), "length", ids=tuple(reasoning)), Scripted(("好",))],
                    think(reasoning_budget=len(reasoning)))
    assert fake.requests[1]["prompt"] == ids + reasoning + family.encode_close()
    assert gen.forced and gen.reasoning == ZH_REASONING and gen.reasoning_tokens == len(reasoning)

    # closed by the model: its reasoning tokens are the tokens before the first one reaching past the close
    text = ZH_REASONING + family.spec.think_close + ZH_ANSWER
    whole = family.encode_text(text) + [family.turn_end_id]
    pieces = server_pieces(family.tokenizer, whole)
    assert b"".join(pieces) == text.encode()
    ends = list(itertools.accumulate(len(p) for p in pieces))
    answer_starts = len((ZH_REASONING + family.spec.think_close).encode())
    first = next(i for i, end in enumerate(ends) if end > answer_starts)
    gen, fake = run(family, [Scripted(tuple(pieces), "stop", ids=tuple(whole))],
                    think(reasoning_budget=len(whole)))
    assert gen.reasoning == ZH_REASONING and gen.answer_text == ZH_ANSWER and gen.phases == 1
    assert gen.reasoning_tokens == first

    # the answer cut by the budget goes on from every token, within the answer budget
    cut = ends.index(len((ZH_REASONING + family.spec.think_close + ZH_CUT).encode())) + 1
    left_out = [i for i in range(cut) if utf8_cut(b"".join(pieces[:i + 1])) < ends[i]]
    assert any(i >= first for i in left_out)  # the server leaves tokens of the answer out of its list too
    head = Scripted(tuple(pieces[:cut]), "length", ids=tuple(whole[:cut]))
    gen, fake = run(family, [head, Scripted(("到两篇。",))],
                    think(reasoning_budget=cut, answer_tokens=100))
    _, second = fake.requests
    assert second["prompt"] == ids + whole[:cut] and second["max_tokens"] == 100 - (cut - first)
    assert gen.reasoning_tokens == first and gen.answer_text == ZH_CUT + "到两篇。"

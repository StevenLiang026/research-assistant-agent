import copy
import json
from pathlib import Path

import pytest
from support.fakes import HITS, PAPER, SAMPLE_MESSAGES, ScriptedModel, call, turn
from support.family_fakes import token_ids
from support.llama_fakes import install_fake_family

from research_agent.agent.context import (
    KEEP_KEYS,
    build_context,
    compress_tool_content,
    family_token_counter,
    fits,
    qwen_token_counter,
    step_contexts,
)
from research_agent.agent.prompts import system_prompt
from research_agent.agent.types import ToolResult, api_to_hf
from research_agent.canonical import canonical_json
from research_agent.family.core import Mode, load_family
from research_agent.tools import schemas

MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "Qwen3-1.7B"
NON_JSON_NOTE = canonical_json({"note": "non-JSON tool output omitted"})

EVIDENCE = "Speculative decoding drafts tokens with a small model and verifies them with the target model."
NOTE = {"note_id": "2609.01234-adaptive-draft", "arxiv_id": "2609.01234", "path": "notes/2609.01234.md",
        "evidence": [EVIDENCE]}
# SAMPLE_MESSAGES plus a third step (write_note) and the closing answer.
LONG = SAMPLE_MESSAGES + [
    {"role": "assistant", "content": "",
     "tool_calls": [call("write_note", "c4", arxiv_id="2609.01234").to_api()]},
    {"role": "tool", "tool_call_id": "c4", "content": canonical_json(NOTE)},
    {"role": "assistant", "content": "已为 2609.01234 写好笔记。"},
]
# The search step answered with plain text instead of JSON.
TEXT_FIRST = [*SAMPLE_MESSAGES[:3],
              {"role": "tool", "tool_call_id": "c1", "content": "upstream sent an error page"},
              *SAMPLE_MESSAGES[4:]]


def test_latest_step_stays_whole_and_the_older_search_keeps_only_ids_and_titles():
    context = build_context(SAMPLE_MESSAGES)
    assert context[4:] == SAMPLE_MESSAGES[4:]  # both parallel get_paper results, untouched
    assert context[:3] == SAMPLE_MESSAGES[:3]
    assert context[3] == {"role": "tool", "tool_call_id": "c1", "content": canonical_json(
        [{"arxiv_id": h["arxiv_id"], "title": h["title"]} for h in HITS])}


def test_every_tool_output_before_the_latest_step_is_compressed():
    context = build_context(LONG)
    assert json.loads(context[5]["content"]) == {"arxiv_id": "2609.01234", "title": PAPER["title"]}
    assert json.loads(context[6]["content"]) == {"arxiv_id": "2609.05678",
                                                 "title": "Tree Verification at Scale"}
    assert context[7:] == LONG[7:]  # the write_note step and the closing answer
    assert [m["role"] for m in context] == [m["role"] for m in LONG]


def test_history_without_a_step_is_copied_unchanged():
    assert build_context(SAMPLE_MESSAGES[:2]) == SAMPLE_MESSAGES[:2]
    assert build_context([]) == []


def test_applying_it_twice_changes_nothing():
    for history in (SAMPLE_MESSAGES, LONG, TEXT_FIRST):
        once = build_context(history)
        assert build_context(once) == once


def test_input_is_not_mutated_and_the_result_shares_nothing_with_it():
    history = copy.deepcopy(LONG)
    before = copy.deepcopy(history)
    context = build_context(history)
    context[2]["tool_calls"][0]["function"]["arguments"] = "{}"
    context[8]["content"] = "changed"  # a message of the latest step
    context.append({"role": "user", "content": "more"})
    pairs = step_contexts(history)
    pairs[0][1]["content"] = "changed"
    pairs[1][0][3]["content"] = "changed"
    assert history == before


def test_error_envelopes_keep_the_error_code():
    empty = ToolResult("c1", "search_arxiv", [], error=schemas.EMPTY_RESULTS).to_api()["content"]
    assert compress_tool_content(empty) == canonical_json({"error": "EMPTY_RESULTS"})
    fallback = ToolResult("c5", "fetch_fulltext", {"arxiv_id": "2609.01234", "section": "introduction",
                                                    "text": PAPER["abstract"]}, error=schemas.NO_FULLTEXT)
    assert json.loads(compress_tool_content(fallback.to_api()["content"])) == {
        "arxiv_id": "2609.01234", "error": "NO_FULLTEXT", "section": "introduction"}


def test_non_json_output_becomes_a_note_not_an_empty_object():
    for text in ("upstream sent an error page", ""):
        assert compress_tool_content(text) == NON_JSON_NOTE
    assert compress_tool_content(NON_JSON_NOTE) == NON_JSON_NOTE  # stable when compressed again
    context = build_context(TEXT_FIRST)
    assert context[3]["content"] == NON_JSON_NOTE and context[4:] == TEXT_FIRST[4:]


def test_only_kept_keys_survive_at_every_depth():
    assert KEEP_KEYS == ("arxiv_id", "title", "note_id", "section", "error")
    raw = [{"arxiv_id": "2609.01234", "authors": ["A. Author"],
            "abstract": "Speculative decoding drafts tokens."},
           {"snippet": "no kept key, so the dict is dropped from the list"},
           [{"note_id": "n1", "tags": ["decoding"]}, {}],
           # a key outside KEEP_KEYS goes with everything under it, ids included
           {"error": "UPSTREAM_ERROR", "items": [{"arxiv_id": "2609.05678"}]}]
    assert json.loads(compress_tool_content(json.dumps(raw))) == [
        {"arxiv_id": "2609.01234"}, [{"note_id": "n1"}], {"error": "UPSTREAM_ERROR"}]
    assert compress_tool_content('{"title": "T", "arxiv_id": "A"}') == '{"arxiv_id":"A","title":"T"}'
    assert compress_tool_content('{"text": "a long section"}') == "{}"


def test_step_contexts_pair_each_assistant_message_with_the_context_before_it():
    pairs = step_contexts(LONG)
    assert [target for _, target in pairs] == [LONG[2], LONG[4], LONG[7], LONG[9]]
    assert [context for context, _ in pairs] == [build_context(LONG[:i]) for i in (2, 4, 7, 9)]
    assert pairs[0][0] == LONG[:2]  # no step yet: system prompt and request only
    assert pairs[1][0] == LONG[:4]  # the search is the latest step here, so it is kept whole
    assert pairs[2][0] == build_context(SAMPLE_MESSAGES)  # ... and compressed one step later


def test_training_samples_match_what_the_model_was_shown():
    # A minimal loop: each step the model gets build_context(history), then the tool results are appended.
    results = {"c1": canonical_json(HITS), "c2": canonical_json(PAPER), "c4": canonical_json(NOTE)}
    model = ScriptedModel([turn(call("search_arxiv", "c1", query="speculative decoding")),
                           turn(call("get_paper", "c2", arxiv_id="2609.01234")),
                           turn(call("write_note", "c4", arxiv_id="2609.01234")),
                           turn(content="已为 2609.01234 写好笔记。")])
    history = copy.deepcopy(SAMPLE_MESSAGES[:2])
    for _ in range(4):
        answer = model.next_turn(build_context(history), schemas.MODEL_TOOLS)
        history.append(answer.to_api())
        history += [{"role": "tool", "tool_call_id": c.id, "content": results[c.id]}
                    for c in answer.tool_calls]
    pairs = step_contexts(history)
    assert [context for context, _ in pairs] == model.contexts
    assert [target for _, target in pairs] == [m for m in history if m["role"] == "assistant"]


def test_fits_counts_the_reply_reserve_against_the_limit():
    seen = []

    def counter(n: int):
        def count_tokens(messages: list[dict], tools: list[dict]) -> int:
            seen.append((messages, tools))
            return n

        return count_tokens

    prompt_budget = schemas.CONTEXT_TOKENS - schemas.REPLY_RESERVE_TOKENS  # 4,096 - 256 = 3,840
    assert fits(SAMPLE_MESSAGES, schemas.MODEL_TOOLS, counter(prompt_budget))  # exactly at the limit
    assert not fits(SAMPLE_MESSAGES, schemas.MODEL_TOOLS, counter(prompt_budget + 1))  # one over
    assert seen[0] == (SAMPLE_MESSAGES, schemas.MODEL_TOOLS)
    assert fits([], [], counter(90), limit=100, reserve=10)
    assert not fits([], [], counter(91), limit=100, reserve=10)


@pytest.mark.skipif(not MODEL_DIR.exists(), reason="local model files are not present")
def test_qwen_counter_measures_the_rendered_prompt_with_the_tools():
    count = qwen_token_counter(MODEL_DIR)
    base = [{"role": "system", "content": system_prompt("2026-10-04")}, SAMPLE_MESSAGES[1]]
    n = count(base, schemas.MODEL_TOOLS)
    assert 1000 < n < 1100  # 1,048 measured with a one-line request (tools/schemas.py)
    assert count(base, []) + 900 < n  # the tool schemas are counted too
    history = base + SAMPLE_MESSAGES[2:]
    full, compressed = count(history, schemas.MODEL_TOOLS), count(build_context(history), schemas.MODEL_TOOLS)
    assert full > compressed > n
    assert fits(build_context(history), schemas.MODEL_TOOLS, count)


# ---- M7: the counter over a family (docs/m7-家族层设计.md §7) ---------------------------------------------


def test_the_family_counter_counts_the_familys_prompt_and_the_qwen_counter_delegates_to_it(tmp_path,
                                                                                          monkeypatch):
    tokenizer = install_fake_family(monkeypatch, tmp_path / "model")
    family = load_family(tmp_path / "model")
    count = family_token_counter(family, Mode.NO_THINK)
    hf = api_to_hf(SAMPLE_MESSAGES)
    # the count before M7: the template with thinking off, the generation prompt appended, no special token
    text = tokenizer.apply_chat_template(hf, tools=schemas.MODEL_TOOLS, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    before = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    assert count(SAMPLE_MESSAGES, schemas.MODEL_TOOLS) == before == len(token_ids(text))
    assert qwen_token_counter(tmp_path / "model")(SAMPLE_MESSAGES, schemas.MODEL_TOOLS) == before
    think = family_token_counter(family, Mode.THINK)
    opened = tokenizer.apply_chat_template(hf, tools=schemas.MODEL_TOOLS, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True) + "<think>\n"
    assert think(SAMPLE_MESSAGES, schemas.MODEL_TOOLS) == len(token_ids(opened)) < before

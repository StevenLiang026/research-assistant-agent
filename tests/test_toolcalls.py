"""parse_tool_calls on Qwen3 completions (docs/modules/接口.md, 本地模型接入时), and LocalModel's default
parser.

The completions are written the way Qwen3-1.7B writes them in non-thinking mode through llama-server's
/v1/completions: each call a <tool_call> block on lines of its own holding {"name", "arguments"}, the turn
ending at the stop string <|im_end|>, which the server does not return.
"""

from __future__ import annotations

import json

from research_agent.agent.models import LocalModel
from research_agent.llm.local import Completion
from research_agent.toolcalls import parse_tool_calls


def fixed_render(messages: list[dict], tools: list[dict] | None = None) -> str:
    return "PROMPT"


ONE = ('<tool_call>\n{"name": "search_arxiv", "arguments": {"query": "speculative decoding", '
       '"date_from": "2026-09-01", "max_results": 5}}\n</tool_call>')
TWO = ('<tool_call>\n{"name": "get_paper", "arguments": {"arxiv_id": "2609.33791"}}\n</tool_call>\n'
       '<tool_call>\n{"name": "get_paper", "arguments": {"arxiv_id": "2610.09639"}}\n</tool_call>')


def test_a_single_call_becomes_its_name_and_argument_object():
    assert parse_tool_calls(ONE) == [{"name": "search_arxiv", "arguments": {
        "query": "speculative decoding", "date_from": "2026-09-01", "max_results": 5}}]


def test_two_calls_come_back_in_the_order_written():
    assert parse_tool_calls(TWO) == [{"name": "get_paper", "arguments": {"arxiv_id": "2609.33791"}},
                                     {"name": "get_paper", "arguments": {"arxiv_id": "2610.09639"}}]


def test_prose_before_and_after_the_block_is_ignored():
    text = "我先检索一下最近的论文。\n\n" + ONE + "\n检索完成后再挑一篇。"
    assert [c["name"] for c in parse_tool_calls(text)] == ["search_arxiv"]


def test_broken_json_keeps_the_block_text_and_the_name_it_can_read():
    broken = '{"name": "search_arxiv", "arguments": {"query": "kv cache", "max_results": 5}'
    assert parse_tool_calls(f"<tool_call>\n{broken}\n</tool_call>") == [
        {"name": "search_arxiv", "arguments": broken}]


def test_a_block_without_a_string_name_gets_an_empty_name_and_its_text():
    body = '{"arguments": {"query": "tokenizer"}}'
    assert parse_tool_calls(f"<tool_call>\n{body}\n</tool_call>") == [{"name": "", "arguments": body}]
    numeric = '{"name": 7, "arguments": {}}'
    assert parse_tool_calls(f"<tool_call>{numeric}</tool_call>") == [{"name": "", "arguments": numeric}]


def test_a_name_inside_the_arguments_is_not_taken_as_the_calls_name():
    body = '{"arguments": {"name": "inner", "query": "x"}}'
    assert parse_tool_calls(f"<tool_call>{body}</tool_call>") == [{"name": "", "arguments": body}]


def test_a_top_level_name_after_a_nested_one_is_found_in_broken_json():
    body = '{"arguments": {"name": "inner"}, "name": "search_arxiv",}'
    assert parse_tool_calls(f"<tool_call>{body}</tool_call>") == [{"name": "search_arxiv", "arguments": body}]
    escaped = '{"note": "a \\"name\\": \\"fake\\"", "name": "get_paper", "arguments": {"arxiv_id": }}'
    assert parse_tool_calls(f"<tool_call>{escaped}</tool_call>")[0]["name"] == "get_paper"


def test_braces_and_closing_tags_inside_strings_do_not_end_the_call():
    args = {"arxiv_id": "2609.33791", "mode": "scaffold",
            "evidence": "The loss is {KL(p||q)} and the tag </tool_call> is just text."}
    text = f"<tool_call>\n{json.dumps({'name': 'write_note', 'arguments': args})}\n</tool_call>"
    assert parse_tool_calls(text) == [{"name": "write_note", "arguments": args}]


def test_an_unterminated_last_block_is_parsed_when_its_json_is_complete():
    cut = '<tool_call>\n{"name": "list_notes", "arguments": {"limit": 5}}\n'
    assert parse_tool_calls(ONE + "\n" + cut) == [parse_tool_calls(ONE)[0],
                                                 {"name": "list_notes", "arguments": {"limit": 5}}]


def test_an_unterminated_block_with_incomplete_json_keeps_its_text():
    assert parse_tool_calls('<tool_call>\n{"name": "get_paper", "arguments": {"arxiv_id": "26') == [
        {"name": "get_paper", "arguments": '{"name": "get_paper", "arguments": {"arxiv_id": "26'}]


def test_text_without_a_block_has_no_calls():
    assert parse_tool_calls("这两篇论文的方法不同:前者蒸馏 logits,后者只用采样序列。") == []
    assert parse_tool_calls("") == []


def test_a_call_without_arguments_gets_an_empty_object_and_a_string_argument_is_passed_on():
    assert parse_tool_calls('<tool_call>{"name": "list_notes"}</tool_call>') == [
        {"name": "list_notes", "arguments": {}}]
    assert parse_tool_calls('<tool_call>{"name": "get_paper", "arguments": "{\\"arxiv_id\\": '
                            '\\"2609.33791\\"}"}</tool_call>') == [
        {"name": "get_paper", "arguments": '{"arxiv_id": "2609.33791"}'}]


def test_arguments_that_are_neither_an_object_nor_text_keep_the_block_text():
    body = '{"name": "get_paper", "arguments": ["2609.33791"]}'
    assert parse_tool_calls(f"<tool_call>{body}</tool_call>") == [{"name": "get_paper", "arguments": body}]


class _Client:
    def __init__(self, text: str) -> None:
        self.text = text

    def complete(self, prompt: str, *, lora_scale: float | None = None,
                 stop: list[str] | None = None) -> Completion:
        return Completion(text=self.text, raw={})


def test_local_model_reads_qwen3_calls_with_parse_tool_calls_by_default():
    model = LocalModel(_Client(TWO), fixed_render, lora_scale=1.0)
    turn = model.next_turn([{"role": "user", "content": "compare"}], [])
    assert [(c.name, c.arguments) for c in turn.tool_calls] == [
        ("get_paper", {"arxiv_id": "2609.33791"}), ("get_paper", {"arxiv_id": "2610.09639"})]


def test_local_model_hands_broken_json_to_the_framework_as_bad_arguments():
    broken = '<tool_call>\n{"name": "get_paper", "arguments": {"arxiv_id": }\n</tool_call>'
    model = LocalModel(_Client(broken), fixed_render, lora_scale=0.0)
    (call,) = model.next_turn([{"role": "user", "content": "x"}], []).tool_calls
    assert call.name == "get_paper" and call.arguments is None

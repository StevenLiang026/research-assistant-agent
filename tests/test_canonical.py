"""canonical_json and split_lines: the JSON form written to disk and the way to split it back into lines."""

from __future__ import annotations

import json

import pytest

from research_agent.canonical import canonical_json, split_lines

# every character str.splitlines() treats as a line break besides "\n" and "\r"
SEPARATORS = {"U+2028": " ", "U+2029": " ", "U+0085": "\x85", "x1c": "\x1c", "x1d": "\x1d",
              "x1e": "\x1e", "x0b": "\x0b", "x0c": "\x0c"}


@pytest.mark.parametrize("name", SEPARATORS)
def test_split_lines_keeps_a_row_whose_string_holds_a_unicode_or_control_separator_whole(name):
    row = {"text": f"line one{SEPARATORS[name]}line two"}
    text = canonical_json(row) + "\n" + canonical_json({"text": "next"}) + "\n"
    lines = split_lines(text)
    assert len(lines) == 2
    assert json.loads(lines[0]) == row


def test_split_lines_on_u2028_alone_gives_one_line():
    assert split_lines("a b") == ["a b"]


def test_split_lines_on_u2029_alone_gives_one_line():
    assert split_lines("a b") == ["a b"]


def test_split_lines_on_u0085_alone_gives_one_line():
    assert split_lines("a\x85b") == ["a\x85b"]


def test_split_lines_on_x1c_alone_gives_one_line():
    assert split_lines("a\x1cb") == ["a\x1cb"]


def test_split_lines_on_x0b_alone_gives_one_line():
    assert split_lines("a\x0bb") == ["a\x0bb"]


def test_split_lines_drops_only_the_empty_element_after_a_trailing_newline():
    assert split_lines("a\nb\n") == ["a", "b"]
    assert split_lines("a\n\nb\n\n") == ["a", "", "b", ""]
    assert split_lines("") == []
    assert split_lines("a") == ["a"]


def test_split_lines_strips_one_carriage_return_per_line_of_a_crlf_file():
    assert split_lines("a\r\nb\r\n") == ["a", "b"]
    assert split_lines("a\r\r\n") == ["a\r"]


def test_canonical_json_sorts_keys_and_keeps_non_ascii_raw():
    assert canonical_json({"b": 1, "a": "论文 "}) == '{"a":"论文 ","b":1}'

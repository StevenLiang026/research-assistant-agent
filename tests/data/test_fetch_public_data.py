"""scripts/fetch_public_data.py offline: the draw (uniform or stratified, length-filtered), the counts and the
manifest entry of one source built from a local data file; the Hub lookups are stubbed, nothing is fetched.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from train.tokenizer_fake import FakeTokenizer

from research_agent.data.public import EMPTY_TURN
from research_agent.tools.schemas import CONTEXT_TOKENS

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # the script's dataclass resolves its postponed annotations through it
    spec.loader.exec_module(module)
    return module


fetch = load_script("fetch_public_data")

CALL = {"id": "call_0", "type": "function", "function": {"name": "f", "arguments": "{}"}}
TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}


class BatchTokenizer(FakeTokenizer):
    """FakeTokenizer that also tokenizes a list of texts at once, as the Hugging Face tokenizers do."""

    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            return {"input_ids": [list(range(len(t.split()))) for t in text]}
        return super().__call__(text, **kwargs)


def sample(i: int, *, call: bool, words: int = 3) -> dict:
    messages = [{"role": "user", "content": " ".join(f"w{i}" for _ in range(words))}]
    if call:
        messages.append({"role": "assistant", "content": "", "tool_calls": [CALL]})
        messages.append({"role": "tool", "tool_call_id": "call_0", "content": "{}"})
        messages.append({"role": "assistant", "content": "done"})
    else:
        messages.append({"role": "assistant", "content": "no tool fits"})
    return {"messages": messages, "tools": [TOOL], "source": "t", "source_id": str(i)}


def pool(count: int, no_call: int, *, over_length: int = 0) -> list[dict]:
    """count samples, the first no_call of them without a call, the last over_length of them too long."""
    samples = [sample(i, call=i >= no_call) for i in range(count)]
    for s in samples[count - over_length:]:
        s["messages"][0]["content"] = " ".join("w" for _ in range(CONTEXT_TOKENS + 1))
    return samples


def no_call_count(samples: list[dict], picked: list[int]) -> int:
    return sum(1 for i in picked if not fetch.has_tool_call(samples[i]))


# draw


def test_uniform_draw_is_deterministic_and_keeps_the_pool_order():
    samples = pool(100, 40)
    picked, account = fetch.draw(samples, 10, 42)
    assert len(picked) == 10 and len(set(picked)) == 10 and picked == sorted(picked)
    assert fetch.draw(samples, 10, 42)[0] == picked
    assert fetch.draw(samples, 10, 7)[0] != picked
    no_call = no_call_count(samples, picked)
    assert account == {"no_call_share": None, "candidates": 100, "excluded_for_length": None, "fit": 100,
                       "no_call": no_call, "with_call": 10 - no_call}
    assert fetch.draw(samples, 100, 42)[0] == list(range(100))
    assert fetch.draw(samples, 500, 42)[0] == list(range(100))


def test_stratified_draw_gives_the_no_call_share_exactly():
    samples = pool(2000, 880)  # the glaive-v2 pool is 44% no-call; the plan wants 10%
    picked, account = fetch.draw(samples, 800, 42, no_call_share=0.1)
    assert len(picked) == 800 and len(set(picked)) == 800 and picked == sorted(picked)
    assert no_call_count(samples, picked) == 80
    assert fetch.draw(samples, 800, 42, no_call_share=0.1)[0] == picked
    assert account["no_call_share"] == 0.1 and (account["no_call"], account["with_call"]) == (80, 720)
    picked, _ = fetch.draw(samples, 500, 42, no_call_share=0.1)
    assert len(picked) == 500 and no_call_count(samples, picked) == 50
    # a stratum that is too small is taken whole and the other one fills the count
    samples = pool(200, 5)
    picked, account = fetch.draw(samples, 100, 42, no_call_share=0.1)
    assert len(picked) == 100 and no_call_count(samples, picked) == 5 and account["no_call"] == 5
    samples = pool(50, 45)
    picked, account = fetch.draw(samples, 48, 42, no_call_share=0.1)
    assert len(picked) == 48 and no_call_count(samples, picked) == 43 and account["with_call"] == 5


def test_over_length_candidates_are_excluded_before_the_draw():
    samples = pool(60, 20, over_length=10)
    lengths = fetch.token_lengths(samples, BatchTokenizer())
    picked, account = fetch.draw(samples, 45, 42, lengths=lengths)
    assert len(picked) == 45 and all(i < 50 for i in picked)
    assert account["candidates"] == 60 and account["excluded_for_length"] == 10 and account["fit"] == 50
    picked, account = fetch.draw(samples, 55, 42, lengths=lengths)
    assert picked == list(range(50)) and account["excluded_for_length"] == 10  # every fit candidate
    picked, account = fetch.draw(samples, 30, 42, no_call_share=0.1, lengths=lengths)
    assert len(picked) == 30 and no_call_count(samples, picked) == 3 and all(i < 50 for i in picked)
    # without lengths (no model directory) nothing is excluded and the account says so
    picked, account = fetch.draw(samples, 55, 42)
    assert len(picked) == 55 and account["excluded_for_length"] is None and account["fit"] == 60


def test_token_lengths_match_the_whole_conversation_render_per_sample():
    tokenizer = BatchTokenizer()
    samples = pool(7, 3, over_length=2)
    lengths = fetch.token_lengths(samples, tokenizer, chunk=3)
    expected = [len(tokenizer(fetch.render(s, tokenizer), add_special_tokens=False)["input_ids"])
                for s in samples]
    assert lengths == expected and sum(1 for n in lengths if n > CONTEXT_TOKENS) == 2
    assert fetch.token_lengths([], tokenizer) == []


# counts and token statistics


def test_counts_report_plain_counts_and_the_no_call_share_only():
    samples = pool(10, 4)
    counts = fetch.counts(samples)
    assert counts == {"samples": 10, "no_call": 4, "no_call_share": 0.4, "parallel_steps": 0, "multi_step": 0}
    assert "distinct" not in counts and "distinct_messages" not in counts  # dedup is the owner's module
    assert fetch.counts([]) == {"samples": 0, "no_call": 0, "no_call_share": None, "parallel_steps": 0,
                                "multi_step": 0}


def test_token_stats_from_the_lengths():
    stats = fetch.token_stats([5, 1, CONTEXT_TOKENS + 1, 3])  # sorted 1, 3, 5, 3073: the upper median
    assert stats == {"samples": 4, "min": 1, "median": 5, "mean": round((5 + 1 + CONTEXT_TOKENS + 1 + 3) / 4),
                     "p95": 5, "max": CONTEXT_TOKENS + 1, "over_context": 1, "context_tokens": CONTEXT_TOKENS}


# one source end to end, from a local file


def test_fetch_one_builds_the_manifest_entry_from_a_local_file(tmp_path, monkeypatch):
    zh = {"name": "roll", "description": "A die", "parameters": {"type": "object", "properties": {}}}
    tools = json.dumps([zh])
    records = []
    for i in range(12):
        if i % 2:
            records.append({"conversations": [
                {"from": "human", "value": f"roll {i}"},
                {"from": "function_call", "value": json.dumps({"name": "roll", "arguments": {}})},
                {"from": "observation", "value": "{\"n\": 4}"}, {"from": "gpt", "value": "4"}],
                "tools": tools})
        else:
            records.append({"conversations": [{"from": "human", "value": f"hi {i}"},
                                              {"from": "gpt", "value": "hello"}], "tools": "[]"})
    records.append({"conversations": [{"from": "human", "value": "x"}, {"from": "gpt", "value": ""}],
                    "tools": "[]"})  # an empty assistant turn: refused and counted
    data = tmp_path / "glaive_toolcall_zh_1k.json"
    data.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    source = fetch.Source("glaive_zh", "llamafactory/glaive_toolcall_zh", 4, ("glaive_toolcall_zh_1k.json",),
                          no_call_share=0.5)
    stub = {"repo": source.repo, "revision": "abc123", "license": "apache-2.0", "gated": False,
            "last_modified": None, "files": {source.files[0]: data.stat().st_size}, "notice_files": []}
    monkeypatch.setattr(fetch, "card", lambda s: dict(stub))
    monkeypatch.setattr(fetch, "download", lambda s, revision: [data])
    entry = fetch.fetch_one(source, tmp_path / "out", 42, BatchTokenizer())
    assert entry["rows"] == 13 and entry["converted"] == 12
    assert entry["dropped"] == {EMPTY_TURN: 1}
    assert entry["pool"] == {"samples": 12, "no_call": 6, "no_call_share": 0.5, "parallel_steps": 0,
                             "multi_step": 0}
    assert entry["draw"] == {"no_call_share": 0.5, "candidates": 12, "excluded_for_length": 0, "fit": 12,
                             "no_call": 2, "with_call": 2}
    assert entry["written"]["samples"] == 4 and entry["written"]["no_call"] == 2
    assert entry["tokens"]["samples"] == 4 and entry["tokens"]["over_context"] == 0
    lines = (tmp_path / "out" / "glaive_zh.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4 and all(json.loads(line)["source"] == "glaive_zh" for line in lines)
    for key in ("distinct", "distinct_messages"):
        assert key not in entry["pool"] and key not in entry["written"]
    line = fetch.summary_line("glaive_zh", entry)
    assert "distinct" not in line and "no-call 2" in line and "excluded_for_length=0" in line


def test_the_plan_sources_stratify_only_the_two_glaive_pools():
    shares = {s.name: s.no_call_share for s in fetch.SOURCES}
    assert shares == {"toolace": None, "glaive_v2": 0.1, "hermes": None, "glaive_zh": 0.1, "deepexi": None}
    assert [s.count for s in fetch.SOURCES] == [1500, 800, 700, 500, 500]


@pytest.mark.parametrize("count, share, no_call", [(800, 0.1, 80), (500, 0.1, 50), (10, 0.3, 3)])
def test_stratum_sizes(count, share, no_call):
    assert fetch.stratum_sizes(count, share) == (no_call, count - no_call)

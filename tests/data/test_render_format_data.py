"""scripts/render_format_data.py (M6 WP3, D1) and research_agent.data.public_pool, offline.

The serialisers and parsers of the three preregistered formats are tested by round trips on nested and
parallel calls; the renderer on inline samples, which must load with samples.load_records and cut into one
training step per assistant message; the selection and the CLI on small pools stubbed in place of the
Hugging Face cache (nothing is read from the cache or the network here).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from test_synth import load_script
from train.tokenizer_fake import FakeTokenizer

from research_agent.data import public_pool
from research_agent.train.samples import build_step_samples, load_records

rf = load_script("render_format_data")
TOK = FakeTokenizer()
ACEBENCH_CALL_LIST = re.compile(r"\[\w+\(")  # the notation [ApiName(key='value')] the data must never use

NESTED = [
    (
        "book_hotel",
        {
            "city": "Paris",
            "dates": {"from": "2026-10-01", "to": "2026-10-03"},
            "guests": [{"name": 'Ann "A" O\'Neil', "age": 30}, {"name": "李雷", "age": 7}],
            "smoking": False,
            "budget": 120.5,
            "notes": None,
            "rooms": 2,
            "text": "line one\nline two\ttab \\ back",
        },
    )
]
PARALLEL = [("get_weather", {"city": "Tokyo"}), ("get_weather", {"city": "Kyoto", "unit": "c"}), ("ping", {})]
EDGE = [
    (
        "svc.op_v2",
        {
            "empty_obj": {},
            "empty_list": [],
            "deep": {"a": {"b": {"c": [1, [2, 3], {"d": True}]}}},
            "yes": "no",
            "num_text": "007",
            "big": 1e20,
            "neg": -3,
            "colon": "a: b",
            "dash": "- x",
            "true": "null",
            "hash": "# not a comment",
            "zh": "中文:冒号",
        },
    )
]
CASES = [NESTED, PARALLEL, EDGE]


def tool(name: str, properties: dict | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} does a thing",
            "parameters": {"type": "object", "properties": properties or {}},
        },
    }


def call(i: int, name: str, args: dict) -> dict:
    return {
        "id": f"call_{i}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def sample(
    source: str = "toolace",
    sid: str = "1",
    *,
    obj: bool = False,
    parallel: bool = False,
    user: str = "Find hotels please",
) -> dict:
    args = {"city": "Paris", "dates": {"from": "a", "to": "b"}} if obj else {"city": "Paris"}
    calls = [call(0, "find_hotel", args)] + ([call(1, "get_weather", {"city": "Paris"})] if parallel else [])
    messages = [{"role": "user", "content": user}, {"role": "assistant", "content": "", "tool_calls": calls}]
    messages += [
        {"role": "tool", "tool_call_id": c["id"], "content": f'{{"result": {i}}}'}
        for i, c in enumerate(calls)
    ]
    messages.append({"role": "assistant", "content": "Here is what I found."})
    return {
        "messages": messages,
        "tools": [
            tool("find_hotel", {"city": {"type": "string"}}),
            tool("get_weather", {"city": {"type": "string"}}),
        ],
        "source": source,
        "source_id": sid,
    }


# ---- serialisation ------------------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", rf.FORMATS)
@pytest.mark.parametrize("calls", CASES, ids=["nested", "parallel", "edge"])
def test_round_trip(fmt, calls):
    text = rf.serialize_calls(calls, fmt)
    assert rf.parse_calls(text, fmt) == calls
    assert not ACEBENCH_CALL_LIST.match(text.lstrip())


def test_three_formats_preregistered():
    assert rf.FORMATS == ("python", "json", "yaml")


def test_python_format_shape():
    text = rf.serialize_calls(PARALLEL, "python")
    assert text.splitlines() == ['get_weather(city="Tokyo")', 'get_weather(city="Kyoto", unit="c")', "ping()"]
    nested = rf.serialize_calls(NESTED, "python")
    assert nested.startswith('book_hotel(city="Paris", dates={"from": "2026-10-01", "to": "2026-10-03"}, ')
    assert "smoking=False" in nested and "notes=None" in nested
    assert not re.search(r"\{\s*\w+\s*=", nested)  # nested objects are literals, never kwargs {key=value}
    assert not nested.startswith("[")


def test_json_format_shape():
    text = rf.serialize_calls(PARALLEL, "json")
    assert json.loads(text) == [{"name": n, "arguments": a} for n, a in PARALLEL]
    assert text.startswith('[{"name": "get_weather", "arguments": {"city": "Tokyo"}}')


def test_yaml_format_shape():
    text = rf.serialize_calls([("f", {"k": "v", "o": {"num": 3, "n": 1}})], "yaml")
    assert (
        text == '- name: f\n  arguments:\n    k: "v"\n    o:\n      num: 3\n      "n": 1'
    )  # n: YAML 1.1 bool
    assert rf.serialize_calls([("ping", {})], "yaml") == "- name: ping\n  arguments: {}"


@pytest.mark.parametrize("calls", CASES, ids=["nested", "parallel", "edge"])
def test_yaml_agrees_with_pyyaml(calls):
    yaml = pytest.importorskip("yaml")
    loaded = yaml.safe_load(rf.serialize_calls(calls, "yaml"))
    assert [(c["name"], c["arguments"]) for c in loaded] == calls


@pytest.mark.parametrize("fmt", rf.FORMATS)
@pytest.mark.parametrize("bad", ["", "hello there", "[get_weather(city='x')]"])
def test_parse_refuses_other_text(fmt, bad):
    with pytest.raises(ValueError):
        rf.parse_calls(bad, fmt)


def test_python_parse_refuses_positional_and_expressions():
    for bad in ['f("x")', "f(a=1+2)", "f(**{'a': 1})", "x = f(a=1)"]:
        with pytest.raises(ValueError):
            rf.parse_calls(bad, "python")


# ---- eligibility and object arguments -----------------------------------------------------------------


def test_has_object_argument():
    assert rf.has_object_argument(sample(obj=True))
    assert not rf.has_object_argument(sample())
    listed = sample()
    listed["messages"][1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"rows": [{"a": 1}]})
    assert rf.has_object_argument(listed)


def test_eligibility_reasons():
    assert rf.ineligible(sample()) is None
    no_call = {
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        "tools": [],
        "source": "glaive_v2",
        "source_id": "3",
    }
    assert rf.ineligible(no_call) == rf.NO_CALL
    talky = sample()
    talky["messages"][1]["content"] = "Sure, let me check."
    assert rf.ineligible(talky) == rf.TEXT_WITH_CALL
    spaced = sample()
    spaced["messages"][1]["tool_calls"][0]["function"]["name"] = "Find Hotel"
    assert rf.ineligible(spaced) == rf.NOT_PYTHON_NAME
    keyword = sample()
    keyword["messages"][1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"from": "x"})
    assert rf.ineligible(keyword) == rf.NOT_PYTHON_NAME


# ---- rendering ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", rf.FORMATS)
@pytest.mark.parametrize("lang", ["zh", "en"])
def test_render_sample(fmt, lang):
    original = sample(obj=True, parallel=True)
    out = rf.render_sample(original, fmt, lang, variant=1)
    assert out["tools"] == [] and out["format"] == fmt and out["instruction_lang"] == lang
    assert (out["source"], out["source_id"]) == ("toolace", "1")
    system, user, assistant, tool1, tool2, final = out["messages"]
    assert system["role"] == "system"
    for t in original["tools"]:
        assert json.dumps(t["function"], ensure_ascii=False) in system["content"]
    assert "<tools>" not in system["content"] and "<tool_call>" not in system["content"]
    assert not ACEBENCH_CALL_LIST.search(system["content"])
    assert user == original["messages"][0]
    assert "tool_calls" not in assistant
    expected = [
        ("find_hotel", {"city": "Paris", "dates": {"from": "a", "to": "b"}}),
        ("get_weather", {"city": "Paris"}),
    ]
    assert assistant["content"] == rf.serialize_calls(expected, fmt)
    assert rf.parse_calls(assistant["content"], fmt) == expected
    assert tool1 == {"role": "tool", "content": '{"result": 0}'}
    assert tool2 == {"role": "tool", "content": '{"result": 1}'}
    assert final == {"role": "assistant", "content": "Here is what I found."}
    assert rf.check_rendered(out) == []
    assert original["messages"][1]["tool_calls"]  # the input is not modified


def test_instruction_languages_differ():
    tools = sample()["tools"]
    zh, en = (rf.instruction(tools, "python", lang, 0) for lang in ("zh", "en"))
    assert re.search(r"[\u4e00-\u9fff]", zh) and not re.search(r"[\u4e00-\u9fff]", en)
    assert rf.instruction(tools, "python", "en", 0) != rf.instruction(tools, "python", "en", 1)


def test_rendered_samples_load_and_cut_into_steps(tmp_path):
    rendered = [rf.render_sample(sample(obj=True, parallel=True), fmt, "en", 0) for fmt in rf.FORMATS]
    path = tmp_path / "mix.jsonl"
    path.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in rendered), encoding="utf-8")
    records = load_records(path)
    assert len(records) == 3 and all(isinstance(r, dict) for r in records)
    for record, original in zip(records, rendered, strict=True):
        steps, dropped = build_step_samples(record, TOK)
        assert dropped.total == 0
        assert len(steps) == 2  # one per assistant message: the calls, then the answer
        assert "<tools>" not in steps[0]["prompt"]
        assert steps[0]["completion"].startswith(original["messages"][2]["content"])
        assert original["messages"][0]["content"] in steps[0]["prompt"]


def test_check_rendered_finds_problems():
    good = rf.render_sample(sample(), "json", "en", 0)
    assert rf.check_rendered(good) == []
    bad = json.loads(json.dumps(good))
    bad["tools"] = [tool("x")]
    bad["messages"][2]["tool_calls"] = []
    bad["messages"][0]["role"] = "user"
    assert len(rf.check_rendered(bad)) >= 3


# ---- selection and assignment -------------------------------------------------------------------------


def pool(source: str, n: int, n_obj: int, *, start: int = 0) -> list[dict]:
    return [
        sample(source, str(start + i), obj=i < n_obj, parallel=i % 3 == 0, user=f"q{source}{i}")
        for i in range(n)
    ]


def test_select_meets_quotas_and_object_share():
    pools = {"toolace": pool("toolace", 40, 30), "hermes": pool("hermes", 40, 2)}
    picks, account = rf.select(pools, {"toolace": 10, "hermes": 10}, seed=1, object_share=0.6)
    assert [s["source"] for s in picks].count("toolace") == 10
    by_source = account["per_source"]
    assert by_source["toolace"]["object"] == 6 and by_source["hermes"]["object"] == 2
    assert picks == rf.select(pools, {"toolace": 10, "hermes": 10}, seed=1, object_share=0.6)[0]
    assert picks != rf.select(pools, {"toolace": 10, "hermes": 10}, seed=2, object_share=0.6)[0]


def test_select_respects_fits_and_moves_shortfall():
    pools = {"toolace": pool("toolace", 20, 10), "glaive_zh": pool("glaive_zh", 3, 0)}
    too_long = {"0", "1", "2"}
    picks, account = rf.select(
        pools,
        {"toolace": 5, "glaive_zh": 5},
        seed=3,
        object_share=0.6,
        fits=lambda s: not (s["source"] == "toolace" and s["source_id"] in too_long),
    )
    assert len(picks) == 10
    assert not any(s["source"] == "toolace" and s["source_id"] in too_long for s in picks)
    assert account["per_source"]["glaive_zh"]["picked"] == 3
    assert account["per_source"]["toolace"]["picked"] == 7


def test_assign_formats_and_languages():
    pools = {"toolace": pool("toolace", 60, 40), "deepexi": pool("deepexi", 30, 15)}
    picks, _ = rf.select(pools, {"toolace": 40, "deepexi": 20}, seed=5, object_share=0.6)
    rendered = rf.assign(picks, seed=5)
    formats = [r["format"] for r in rendered]
    assert {f: formats.count(f) for f in rf.FORMATS} == {"python": 20, "json": 20, "yaml": 20}
    langs = [r["instruction_lang"] for r in rendered]
    assert langs.count("zh") == 30
    assert all(r["instruction_lang"] == "zh" for r in rendered if r["source"] == "deepexi")
    for r in rendered:
        calls = next(m["content"] for m in r["messages"] if m["role"] == "assistant")
        assert not ACEBENCH_CALL_LIST.match(calls.lstrip())
        assert rf.check_rendered(r) == []


def test_quotas_sum_to_count():
    for count in (1500, 300, 7):
        assert sum(rf.quotas(count).values()) == count
    assert rf.quotas(1500)["glaive_zh"] == 62


# ---- public_pool: the cache, never the network --------------------------------------------------------


def write_jsonl(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8", newline="\n"
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def glaive_zh_record(i: int) -> dict:
    tools = [{"name": "f", "description": "", "parameters": {"type": "object", "properties": {"a": {}}}}]
    return {
        "conversations": [
            {"from": "human", "value": f"问题 {i}"},
            {"from": "function_call", "value": json.dumps({"name": "f", "arguments": {"a": i}})},
            {"from": "observation", "value": "ok"},
            {"from": "gpt", "value": "好的"},
        ],
        "tools": json.dumps(tools),
    }


def test_load_unused_excludes_drawn_ids(tmp_path, monkeypatch):
    data = tmp_path / "cache" / "glaive_toolcall_zh_1k.json"
    data.parent.mkdir()
    data.write_text(json.dumps([glaive_zh_record(i) for i in range(5)]), encoding="utf-8")
    seen = []

    def fake_cache(repo_id, filename, *, revision, repo_type):
        seen.append((repo_id, filename, revision, repo_type))
        return str(data)

    monkeypatch.setattr(public_pool, "try_to_load_from_cache", fake_cache)
    raw = tmp_path / "raw"
    sha = write_jsonl(raw / "glaive_zh.jsonl", [{"source_id": "1"}, {"source_id": "3"}])
    entry = {
        "repo": "llamafactory/glaive_toolcall_zh",
        "revision": "abc",
        "sha256": sha,
        "per_file": [{"file": "glaive_toolcall_zh_1k.json", "config": None}],
    }
    samples, account = public_pool.load_unused("glaive_zh", entry, raw)
    assert [s["source_id"] for s in samples] == ["0", "2", "4"]
    assert account == {"rows": 5, "converted": 5, "dropped": {}, "drawn": 2, "unused": 3}
    assert seen == [("llamafactory/glaive_toolcall_zh", "glaive_toolcall_zh_1k.json", "abc", "dataset")]


def test_load_unused_refuses_changed_draw_and_cache_miss(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    write_jsonl(raw / "glaive_zh.jsonl", [{"source_id": "1"}])
    entry = {
        "repo": "r/x",
        "revision": "abc",
        "sha256": "0" * 64,
        "per_file": [{"file": "x.json", "config": None}],
    }
    monkeypatch.setattr(public_pool, "try_to_load_from_cache", lambda *a, **k: str(tmp_path / "none"))
    with pytest.raises(ValueError, match="SHA-256"):
        public_pool.load_unused("glaive_zh", entry, raw)
    entry["sha256"] = None
    monkeypatch.setattr(public_pool, "try_to_load_from_cache", lambda *a, **k: None)
    with pytest.raises(FileNotFoundError, match="not in the Hugging Face cache"):
        public_pool.load_unused("glaive_zh", entry, raw)


def test_hermes_ids_carry_the_config(tmp_path, monkeypatch):
    files = {}
    for name in ("a.json", "b.json"):
        files[name] = tmp_path / name
        files[name].write_text("[]", encoding="utf-8")
    monkeypatch.setattr(public_pool, "try_to_load_from_cache", lambda repo, fn, **k: str(files[fn]))
    entry = {
        "repo": "r/h",
        "revision": "abc",
        "sha256": None,
        "per_file": [{"file": "a.json", "config": "func_calling"}, {"file": "b.json", "config": "glaive"}],
    }
    assert public_pool.source_ids(entry, 0, 2) == ["func_calling/0", "func_calling/1"]
    write_jsonl(tmp_path / "raw" / "hermes.jsonl", [])
    assert public_pool.load_unused("hermes", entry, tmp_path / "raw")[1]["rows"] == 0


# ---- CLI ----------------------------------------------------------------------------------------------


@pytest.fixture
def stub_pools(monkeypatch):
    pools = {
        "toolace": pool("toolace", 60, 40),
        "glaive_v2": pool("glaive_v2", 40, 10),
        "hermes": pool("hermes", 30, 5),
        "deepexi": pool("deepexi", 40, 20),
        "glaive_zh": pool("glaive_zh", 8, 0),
    }
    accounts = {
        name: {"rows": len(p) + 5, "converted": len(p) + 2, "dropped": {}, "drawn": 2, "unused": len(p)}
        for name, p in pools.items()
    }

    def fake_load(manifest, raw_dir):
        return {k: json.loads(json.dumps(v)) for k, v in pools.items()}, accounts

    monkeypatch.setattr(rf, "load_pools", fake_load)
    return pools


def run_cli(tmp_path: Path, name: str, *extra: str) -> tuple[int, Path, Path]:
    out, manifest = tmp_path / f"{name}.jsonl", tmp_path / f"{name}-manifest.json"
    code = rf.main(
        [
            "--out",
            str(out),
            "--manifest",
            str(manifest),
            "--count",
            "60",
            "--seed",
            "7",
            "--model",
            str(tmp_path / "no-model"),
            *extra,
        ]
    )
    return code, out, manifest


def test_cli_writes_deterministic_file_and_manifest(tmp_path, stub_pools):
    code, out, manifest_path = run_cli(tmp_path, "a")
    assert code == 0
    code2, out2, _ = run_cli(tmp_path, "b")
    assert code2 == 0 and out.read_bytes() == out2.read_bytes()
    data = out.read_bytes()
    assert b"\r\n" not in data and data.endswith(b"\n")
    rows = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["output"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert manifest["output"]["bytes"] == len(data) and manifest["output"]["records"] == len(rows) == 60
    assert manifest["formats"] == {
        f: {"description": rf.FORMAT_DESCRIPTIONS[f], "records": sum(r["format"] == f for r in rows)}
        for f in rf.FORMATS
    }
    assert manifest["languages"] == {
        "en": sum(r["instruction_lang"] == "en" for r in rows),
        "zh": sum(r["instruction_lang"] == "zh" for r in rows),
    }
    assert manifest["languages"]["zh"] == 30
    per_source = {s: sum(r["source"] == s for r in rows) for s in stub_pools}
    assert {s: v["picked"] for s, v in manifest["per_source"].items()} == per_source
    objects = sum(rf.has_object_argument(rf.unrender(r)) for r in rows)
    assert manifest["object_argument"]["records"] == objects and objects / 60 >= 0.5
    assert manifest["excluded_source_ids"]["total"] == 10
    assert all(rf.check_rendered(r) == [] for r in rows)
    assert not any(
        ACEBENCH_CALL_LIST.search(m["content"])
        for r in rows
        for m in r["messages"]
        if m["role"] in ("system", "assistant")
    )


def test_cli_refuses_to_overwrite_without_force(tmp_path, stub_pools):
    assert run_cli(tmp_path, "a")[0] == 0
    assert run_cli(tmp_path, "a")[0] == 2
    assert run_cli(tmp_path, "a", "--force")[0] == 0


def test_unrender_inverts_render():
    original = sample(obj=True, parallel=True)
    for fmt in rf.FORMATS:
        back = rf.unrender(rf.render_sample(original, fmt, "zh", 0))
        calls = [
            (c["function"]["name"], json.loads(c["function"]["arguments"]))
            for m in back["messages"]
            for c in m.get("tool_calls") or []
        ]
        assert calls == [
            ("find_hotel", {"city": "Paris", "dates": {"from": "a", "to": "b"}}),
            ("get_weather", {"city": "Paris"}),
        ]

"""scripts/build_ace_probe.py (M6 WP4, D3), offline: eligibility, incomplete items, the draw and the CLI.

The pools are stubbed in place of the Hugging Face cache and the benchmark index in place of BFCL and
ACEBench; nothing is read from the cache or the network here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_synth import load_script

from research_agent.data.overlap import text_ngrams
from research_agent.eval import ace_probe as ap
from research_agent.eval.lock import check_lock, find_test_material
from research_agent.train.samples import load_records

bp = load_script("build_ace_probe")


def tool(name: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"{name} does a thing",
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


FLIGHT = tool(
    "book_flight",
    {"origin": {"type": "string"}, "destination": {"type": "string"}, "seats": {"type": "integer"}},
    ["origin", "destination"],
)
WEATHER = tool("get_weather", {"city": {"type": "string"}, "unit": {"type": "string"}}, ["city"])


def sample(
    source: str, sid: str, user: str, calls: list[tuple[str, dict]], tools: list[dict], text: str = ""
) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "content": text,
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "type": "function",
                        "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)},
                    }
                    for i, (n, a) in enumerate(calls)
                ],
            },
            {"role": "tool", "tool_call_id": "call_0", "content": "{}"},
            {"role": "assistant", "content": "Done."},
        ],
        "tools": tools,
        "source": source,
        "source_id": sid,
    }


def flight(source: str, sid: str, origin="Boston", destination="Denver") -> dict:
    user = (
        f"Please book a flight from {origin} to {destination} for me next week, "
        f"this is record {sid} of my plan."
    )
    return sample(
        source,
        sid,
        user,
        [("book_flight", {"origin": origin, "destination": destination})],
        [FLIGHT, WEATHER],
    )


# ---- eligibility ------------------------------------------------------------------------------------------


def test_normal_eligibility_reasons():
    assert bp.normal_ineligible(flight("toolace", "1")) is None
    no_call = flight("toolace", "2")
    no_call["messages"][1] = {"role": "assistant", "content": "Which city?"}
    assert bp.normal_ineligible(no_call) == bp.FIRST_STEP
    with_text = sample(
        "toolace",
        "3",
        "Weather in Oslo?",
        [("get_weather", {"city": "Oslo"})],
        [WEATHER],
        text="Sure, checking.",
    )
    assert bp.normal_ineligible(with_text) == bp.FIRST_STEP
    ungrounded = sample(
        "toolace", "4", "Weather in the capital of Norway?", [("get_weather", {"city": "Oslo"})], [WEATHER]
    )
    assert bp.normal_ineligible(ungrounded) == bp.UNGROUNDED
    acebench = sample(
        "toolace",
        "5",
        "Run [get_weather(city='Oslo')] for Oslo",
        [("get_weather", {"city": "Oslo"})],
        [WEATHER],
    )
    assert bp.normal_ineligible(acebench) == bp.ACEBENCH_TEXT
    long = sample(
        "toolace",
        "6",
        "Weather in Oslo? " + "x" * bp.MAX_CHARS,
        [("get_weather", {"city": "Oslo"})],
        [WEATHER],
    )
    assert bp.normal_ineligible(long) == bp.TOO_LONG


def test_grounding_reads_numbers_case_and_whitespace():
    calls = [("book_flight", {"origin": "new york", "destination": "Denver", "seats": 3})]
    assert bp.grounded("Fly from New  York to denver, 3 seats.", calls)
    assert not bp.grounded("Fly from New York to Denver, 13 seats.", calls)  # 3 only inside 13
    assert bp.grounded("x", [("ping", {"flag": True, "nothing": None})])


# ---- incomplete items -------------------------------------------------------------------------------------


def test_make_incomplete_removes_one_required_value_that_occurs_once():
    made = bp.make_incomplete(flight("toolace", "1"), seed=1)
    assert made is not None
    user, function, missing = made
    assert function == "book_flight" and missing in ("origin", "destination")
    removed = {"origin": "Boston", "destination": "Denver"}[missing]
    assert removed not in user and "  " not in user and " ," not in user
    other = {"origin": "Boston", "destination": "Denver"}
    other.pop(missing)
    assert next(iter(other.values())) in user
    assert bp.make_incomplete(flight("toolace", "1"), seed=1) == made  # deterministic


def test_make_incomplete_skips_records_it_cannot_cut_cleanly():
    twice = sample(
        "toolace", "1", "Weather in Oslo, Oslo please", [("get_weather", {"city": "Oslo"})], [WEATHER]
    )
    assert bp.make_incomplete(twice, seed=1) is None
    parallel = sample(
        "toolace",
        "2",
        "Weather in Oslo and Rome",
        [("get_weather", {"city": "Oslo"}), ("get_weather", {"city": "Rome"})],
        [WEATHER],
    )
    assert bp.make_incomplete(parallel, seed=1) is None
    only_value = sample("toolace", "3", "Oslo", [("get_weather", {"city": "Oslo"})], [WEATHER])
    assert bp.make_incomplete(only_value, seed=1) is None
    no_required = sample(
        "toolace",
        "4",
        "Weather in Oslo",
        [("get_weather", {"city": "Oslo"})],
        [tool("get_weather", {"city": {"type": "string"}}, [])],
    )
    assert bp.make_incomplete(no_required, seed=1) is None
    paraphrased = sample(
        "toolace",
        "5",
        "Book a flight from Boston to the Mile High City",
        [("book_flight", {"origin": "Boston", "destination": "Denver"})],
        [FLIGHT],
    )
    assert bp.make_incomplete(paraphrased, seed=1) is None  # another required value is not in the text


def test_make_incomplete_removes_quotes_around_the_value():
    s = sample(
        "toolace",
        "1",
        'Please look up the weather for "Oslo" today.',
        [("get_weather", {"city": "Oslo"})],
        [WEATHER],
    )
    user, _, missing = bp.make_incomplete(s, seed=3)
    assert missing == "city" and '""' not in user and "Oslo" not in user


# ---- the draw ---------------------------------------------------------------------------------------------


def pools(n_en: int = 40, n_zh: int = 30) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {name: [] for name in bp.SOURCES}
    cities = ["Boston", "Denver", "Austin", "Seattle", "Chicago", "Miami", "Dallas", "Phoenix"]
    for name in bp.EN_SOURCES:
        for i in range(n_en):
            out[name].append(flight(name, str(i), cities[i % 8], f"Town{name}{i}"))
    for name in bp.ZH_SOURCES:
        for i in range(n_zh):
            user = f"帮我订一张从北京到上海{name}{i}号的机票。"
            out[name].append(
                sample(
                    name,
                    str(i),
                    user,
                    [("book_flight", {"origin": "北京", "destination": f"上海{name}{i}号"})],
                    [FLIGHT],
                )
            )
    return out


def test_build_counts_languages_and_disjointness():
    items, account = bp.build(pools(), seed=5, normal=8, incomplete=4, exclude=set())
    assert len(items) == 12
    assert sum(i["kind"] == "normal" for i in items) == 8
    assert {
        (k, lang): sum(i["kind"] == k and i["lang"] == lang for i in items)
        for k in ap.KINDS
        for lang in ap.LANGS
    } == {("normal", "zh"): 4, ("normal", "en"): 4, ("incomplete", "zh"): 2, ("incomplete", "en"): 2}
    keys = [(i["source"], i["source_id"]) for i in items]
    assert len(set(keys)) == len(keys)
    assert {i["source"] for i in items if i["lang"] == "zh"} <= set(bp.ZH_SOURCES)
    assert {i["source"] for i in items if i["lang"] == "en"} <= set(bp.EN_SOURCES)
    assert len({i["id"] for i in items}) == 12
    for item in items:
        assert ap.check_item(item) == []
        assert item["messages"] == [
            {"role": "system", "content": item["system"]},
            {"role": "user", "content": item["user"]},
        ]
        assert item["system"] == ap.system_prompt(item["tools"], item["lang"])
    assert find_test_material(items) == []
    assert account["per_language"]["en"]["normal"] == 4


def test_build_is_deterministic_for_a_seed_and_differs_across_seeds():
    a, _ = bp.build(pools(), seed=5, normal=8, incomplete=4, exclude=set())
    b, _ = bp.build(pools(), seed=5, normal=8, incomplete=4, exclude=set())
    c, _ = bp.build(pools(), seed=6, normal=8, incomplete=4, exclude=set())
    assert a == b
    assert [(i["source"], i["source_id"]) for i in a] != [(i["source"], i["source_id"]) for i in c]


def test_build_leaves_out_excluded_records_and_overlapping_ones():
    p = pools(n_en=3, n_zh=3)
    exclude = {(name, "0") for name in bp.SOURCES}
    overlapping = p["toolace"][1]["messages"][0]["content"]
    unique = [g for g in text_ngrams(overlapping) if "towntoolace1 " in g]
    index = SimpleNamespace(grams={g: [("acebench", "f.json", "x")] for g in unique})
    assert index.grams  # the text is long enough to give grams
    items, account = bp.build(p, seed=5, normal=4, incomplete=2, exclude=exclude, index=index)
    keys = {(i["source"], i["source_id"]) for i in items}
    assert not keys & exclude
    assert ("toolace", "1") not in keys
    assert account["per_source"]["toolace"]["excluded"] == 1
    assert account["per_source"]["toolace"]["ineligible"][bp.OVERLAP] == 1


def test_build_reports_a_shortfall():
    items, account = bp.build(pools(n_en=1, n_zh=1), seed=5, normal=10, incomplete=4, exclude=set())
    assert len(items) < 14 and account["shortfall"] > 0


# ---- exclusion of WP3's records ---------------------------------------------------------------------------


def write_jsonl(path: Path, rows: list[dict]) -> bytes:
    data = "".join(json.dumps(r) + "\n" for r in rows).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return data


def test_wp3_exclusion_reads_manifest_outputs_prompts_and_listed_ids(tmp_path):
    out = tmp_path / "format-mix.jsonl"
    data = write_jsonl(out, [{"source": "toolace", "source_id": "7"}])
    prompts = tmp_path / "replay.prompts.jsonl"
    write_jsonl(prompts, [{"source": "glaive_v2", "source_id": "9"}])
    m1 = tmp_path / "fm.json"
    m1.write_text(
        json.dumps({"output": {"path": out.as_posix(), "sha256": hashlib.sha256(data).hexdigest()}})
    )
    m2 = tmp_path / "br.json"
    m2.write_text(json.dumps({"prompts": {"path": prompts.as_posix()}, "source_ids": {"hermes": ["3"]}}))
    ids, account = bp.wp3_excluded([m1, m2, tmp_path / "absent.json"], extra_files=[])
    assert ids == {("toolace", "7"), ("glaive_v2", "9"), ("hermes", "3")}
    assert [a["present"] for a in account["manifests"]] == [True, True, False]


def test_wp3_exclusion_refuses_a_changed_output(tmp_path):
    out = tmp_path / "format-mix.jsonl"
    write_jsonl(out, [{"source": "toolace", "source_id": "7"}])
    m1 = tmp_path / "fm.json"
    m1.write_text(json.dumps({"output": {"path": out.as_posix(), "sha256": "0" * 64}}))
    with pytest.raises(ValueError, match="SHA-256"):
        bp.wp3_excluded([m1], extra_files=[])


def test_wp3_exclusion_reads_existing_default_files_without_a_manifest(tmp_path):
    f = tmp_path / "base-replay.prompts.jsonl"
    write_jsonl(f, [{"source": "toolace", "source_id": "1"}])
    ids, account = bp.wp3_excluded([], extra_files=[f, tmp_path / "missing.jsonl"])
    assert ids == {("toolace", "1")}
    assert [a["records"] for a in account["files"]] == [1]


# ---- CLI --------------------------------------------------------------------------------------------------


@pytest.fixture
def stub(monkeypatch):
    def fake_pools(manifest, raw_dir):
        p = pools()
        return p, {
            name: {"rows": len(v), "drawn": 0, "unused": len(v), "revision": "r"} for name, v in p.items()
        }

    monkeypatch.setattr(bp, "load_pools", fake_pools)
    monkeypatch.setattr(bp, "load_index", lambda bfcl, ace: SimpleNamespace(grams={}, files=[{"x": 1}]))


def run_cli(tmp_path: Path, *extra: str) -> tuple[int, Path, Path]:
    out, manifest = tmp_path / "probe.jsonl", tmp_path / "probe-manifest.json"
    code = bp.main(
        [
            "--out",
            str(out),
            "--manifest",
            str(manifest),
            "--normal",
            "8",
            "--incomplete",
            "4",
            "--exclude-manifest",
            str(tmp_path / "none.json"),
            "--no-default-excludes",
            *extra,
        ]
    )
    return code, out, manifest


def test_cli_writes_items_manifest_and_lock(tmp_path, stub):
    code, out, manifest = run_cli(tmp_path)
    assert code == 0
    data = out.read_bytes()
    items = [json.loads(line) for line in data.decode().splitlines()]
    assert len(items) == 12 and data.endswith(b"\n") and b"\r" not in data
    m = json.loads(manifest.read_text(encoding="utf-8"))
    assert m["output"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert m["output"]["records"] == 12
    assert m["counts"]["normal"] == {"zh": 4, "en": 4, "total": 8}
    assert m["counts"]["incomplete"] == {"zh": 2, "en": 2, "total": 4}
    listed = {(s, i) for s, ids in m["source_ids"].items() for i in ids}
    assert listed == {(i["source"], i["source_id"]) for i in items}
    assert m["seed"] == bp.SEED and m["format"] == ap.FORMAT and m["sentence"] == ap.SENTENCE_TEMPLATE
    assert m["test_material"] is False
    assert m["per_language"]["zh"]["incomplete_from_en_sources"] == 0
    assert check_lock(out) == hashlib.sha256(data).hexdigest()
    assert len(load_records(out)) == 12  # the overlap check reads the probe as public samples


def test_cli_is_deterministic_and_refuses_to_overwrite(tmp_path, stub):
    _, out, _ = run_cli(tmp_path)
    first = out.read_bytes()
    assert run_cli(tmp_path)[0] == 2
    assert run_cli(tmp_path, "--force")[0] == 0
    assert out.read_bytes() == first


def test_cli_fails_on_a_shortfall(tmp_path, stub):
    out, manifest = tmp_path / "probe.jsonl", tmp_path / "m.json"
    code = bp.main(
        [
            "--out",
            str(out),
            "--manifest",
            str(manifest),
            "--normal",
            "800",
            "--incomplete",
            "4",
            "--exclude-manifest",
            str(tmp_path / "none.json"),
            "--no-default-excludes",
        ]
    )
    assert code == 3 and not out.exists()


def test_cli_needs_the_benchmarks(tmp_path, monkeypatch, stub):
    from research_agent.data.overlap import BenchmarkMissing

    def missing(bfcl, ace):
        raise BenchmarkMissing("no ACEBench")

    monkeypatch.setattr(bp, "load_index", missing)
    assert run_cli(tmp_path)[0] == 2


def test_zh_items_fall_back_to_english_records_with_the_chinese_system_text():
    p = pools()
    for s in p["glaive_zh"] + p["deepexi"]:  # deepexi-like: schemas without required parameters
        s["tools"] = [
            {
                **t,
                "function": {**t["function"], "parameters": {**t["function"]["parameters"], "required": []}},
            }
            for t in s["tools"]
        ]
    items, account = bp.build(p, seed=5, normal=8, incomplete=4, exclude=set())
    zh_incomplete = [i for i in items if i["kind"] == "incomplete" and i["lang"] == "zh"]
    assert len(zh_incomplete) == 2
    assert {i["source"] for i in zh_incomplete} <= set(bp.EN_SOURCES)
    for i in zh_incomplete:
        assert i["system"] == ap.system_prompt(i["tools"], "zh") and ap.check_item(i) == []
    assert {i["source"] for i in items if i["kind"] == "normal" and i["lang"] == "zh"} <= set(bp.ZH_SOURCES)
    assert account["per_language"]["zh"]["incomplete_from_en_sources"] == 2
    assert account["shortfall"] == 0

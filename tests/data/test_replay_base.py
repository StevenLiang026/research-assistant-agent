"""scripts/replay_base.py (M6 WP3, D2), offline: prompt builders, the answer filter and the CLI.

The base model is never run: the CLI tests replace LocalClient with a fake that returns scripted texts and
the chat template with a plain renderer; the pools are stubbed in place of the Hugging Face cache. The test of
the manifest's paths makes a directory junction (a symbolic link where junctions do not exist) inside its
temporary folder, and is skipped only where neither can be made.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import ClassVar

import pytest
from test_synth import load_script

from research_agent.agent.models import STOP
from research_agent.data.public import check_sample
from research_agent.family.registry import QWEN3_HYBRID

rb = load_script("replay_base")

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
            "required": ["city"],
        },
    },
}
PASSWORD = {
    "type": "function",
    "function": {
        "name": "generate_password",
        "description": "A random password",
        "parameters": {
            "type": "object",
            "properties": {"length": {"type": "integer"}, "include_symbols": {"type": "boolean"}},
        },
    },
}


def glaive_asking(sid: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"I need a new password ({sid})."},
            {"role": "assistant", "content": "Sure. How long should it be?"},
            {"role": "user", "content": "12"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_0",
                        "type": "function",
                        "function": {"name": "generate_password", "arguments": '{"length": 12}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_0", "content": '{"password": "x"}'},
            {"role": "assistant", "content": "Here it is: x"},
        ],
        "tools": [PASSWORD],
        "source": "glaive_v2",
        "source_id": sid,
    }


def glaive_direct(sid: str) -> dict:
    s = glaive_asking(sid)
    s["messages"] = [s["messages"][0]] + s["messages"][3:]
    return s


def glaive_no_call(sid: str, tools: list | None = None) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"Can you order a pizza? ({sid})"},
            {"role": "assistant", "content": "I'm sorry, I can only generate passwords."},
        ],
        "tools": [PASSWORD] if tools is None else tools,
        "source": "glaive_v2",
        "source_id": sid,
    }


def toolace_no_call(sid: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"What is the tallest tree? ({sid})"},
            {"role": "assistant", "content": answer},
        ],
        "tools": [WEATHER],
        "source": "toolace",
        "source_id": sid,
    }


NONE_FIT = "None of the given functions can be used to answer this question."
LACKS = "The given question lacks the parameters required by the function."


# ---- templates and detectors --------------------------------------------------------------------------


def test_sentence_templates():
    missing = rb.MISSING_TEMPLATE.format(function="get_weather", params="city, unit")
    assert rb.parse_missing(missing) == ("get_weather", ["city", "unit"])
    assert rb.parse_missing(rb.IRRELEVANT_SENTENCE) is None
    for sentence in (rb.MISSING_TEMPLATE, rb.IRRELEVANT_SENTENCE):
        assert not re.search(r"[\u4e00-\u9fff]", sentence)
        low = sentence.lower()
        assert "missing necessary parameters" not in low and "for the api" not in low
        assert "limitations of the function" not in low


def test_asks_for_information():
    assert rb.asks_for_information(glaive_asking("1"))
    assert not rb.asks_for_information(glaive_direct("1"))
    assert not rb.asks_for_information(glaive_no_call("1"))  # a question with no call after it


def test_irrelevant_gold():
    assert rb.is_irrelevant_gold(NONE_FIT)
    assert rb.is_irrelevant_gold("The available functions do not support retrieving tree heights.")
    assert rb.is_irrelevant_gold("The given functions do not provide the capability to measure trees.")
    assert not rb.is_irrelevant_gold(LACKS)
    assert not rb.is_irrelevant_gold("The query does not provide the required arguments for get_weather.")


# ---- prompts ------------------------------------------------------------------------------------------


def pools():
    glaive = (
        [glaive_asking(str(i)) for i in range(10)]
        + [glaive_direct(str(i)) for i in range(10, 13)]
        + [glaive_no_call(str(i)) for i in range(20, 30)]
        + [glaive_no_call("40", tools=[])]
    )
    toolace = [toolace_no_call(str(i), NONE_FIT) for i in range(5)] + [
        toolace_no_call(str(i), LACKS) for i in range(5, 8)
    ]
    return {"glaive_v2": glaive, "toolace": toolace}


@pytest.mark.parametrize("lang", ["zh", "en"])
def test_system_prompt(lang):
    text = rb.system_prompt([WEATHER, PASSWORD], lang)
    for t in (WEATHER, PASSWORD):
        assert json.dumps(t["function"], ensure_ascii=False) in text
    assert rb.MISSING_TEMPLATE in text and rb.IRRELEVANT_SENTENCE in text
    assert bool(re.search(r"[\u4e00-\u9fff]", text)) == (lang == "zh")
    assert not re.search(r"\[\w+\(", text)


def test_build_prompts():
    prompts, account = rb.build_prompts(pools(), limit=12, seed=1, exclude={("glaive_v2", "0")})
    kinds = [p["kind"] for p in prompts]
    assert kinds.count("incomplete") == 6 and kinds.count("irrelevant") == 6
    ids = {(p["source"], p["source_id"]) for p in prompts}
    assert ("glaive_v2", "0") not in ids and ("glaive_v2", "40") not in ids
    assert not any(p["source"] == "toolace" and int(p["source_id"]) >= 5 for p in prompts)
    toolace = [p for p in prompts if p["source"] == "toolace"]
    assert len(toolace) == 5  # every ToolACE no-fit record is taken before glaive's no-call chats
    for p in prompts:
        system, user = p["messages"]
        assert system["role"] == "system" and user["role"] == "user"
        assert system["content"] == rb.system_prompt(p["tools"], p["instruction_lang"])
        if p["kind"] == "incomplete":
            assert p["gold_function"] == "generate_password"
            assert user["content"].startswith("I need a new password")
    assert account["candidates"] == {"incomplete": 9, "irrelevant": 15}
    assert prompts == rb.build_prompts(pools(), limit=12, seed=1, exclude={("glaive_v2", "0")})[0]
    langs = [p["instruction_lang"] for p in prompts]
    assert langs.count("zh") == 6


# ---- filter -------------------------------------------------------------------------------------------


def prompt(kind: str) -> dict:
    return {"kind": kind, "tools": [WEATHER, PASSWORD]}


@pytest.mark.parametrize(
    "text",
    [
        rb.MISSING_TEMPLATE.format(function="get_weather", params="city"),
        rb.MISSING_TEMPLATE.format(function="get_weather", params="city, unit"),
        "  " + rb.MISSING_TEMPLATE.format(function="generate_password", params="length") + "\n",
        # the base model's own ways of listing the names (seen in the 2026-10-17 run): "and", an Oxford
        # comma, and the singular when one name is missing
        "Cannot call get_weather yet: the required parameters city and unit were not given.",
        "Cannot call get_weather yet: the required parameters city, and unit were not given.",
        "Cannot call get_weather yet: the required parameter city was not given.",
    ],
)
def test_filter_accepts_incomplete(text):
    assert rb.check_answer(prompt("incomplete"), text) is None


@pytest.mark.parametrize(
    "text,canonical",
    [
        ("Cannot call get_weather yet: the required parameters city and unit were not given.",
         rb.MISSING_TEMPLATE.format(function="get_weather", params="city, unit")),
        ("Cannot call get_weather yet: the required parameter city was not given.\n",
         rb.MISSING_TEMPLATE.format(function="get_weather", params="city")),
        (rb.MISSING_TEMPLATE.format(function="get_weather", params="city, unit"),
         rb.MISSING_TEMPLATE.format(function="get_weather", params="city, unit")),
    ],
)
def test_a_kept_incomplete_answer_is_written_in_the_canonical_form(text, canonical):
    # the sample teaches one sentence form: the names comma-separated, the plural wording
    assert rb.canonical_answer(prompt("incomplete"), text) == canonical


def test_canonical_answer_is_none_for_a_rejected_answer_and_the_sentence_for_an_irrelevant_one():
    assert rb.canonical_answer(prompt("incomplete"), "Sure! Please provide the city.") is None
    padded = " " + rb.IRRELEVANT_SENTENCE + "\n"
    assert rb.canonical_answer(prompt("irrelevant"), padded) == rb.IRRELEVANT_SENTENCE
    assert rb.canonical_answer(prompt("irrelevant"), "Cannot call get_weather yet: the required parameters "
                               "city were not given.") is None


@pytest.mark.parametrize(
    "text,reason",
    [
        (rb.MISSING_TEMPLATE.format(function="get_weather", params="country"), "parameter"),
        (rb.MISSING_TEMPLATE.format(function="get_time", params="city"), "function"),
        (rb.MISSING_TEMPLATE.format(function="get_weather", params=""), "form"),
        (rb.IRRELEVANT_SENTENCE, "form"),
        ("Sure! " + rb.MISSING_TEMPLATE.format(function="get_weather", params="city"), "form"),
        ('get_weather(city="Paris")', "form"),
    ],
)
def test_filter_rejects_incomplete(text, reason):
    assert reason in rb.check_answer(prompt("incomplete"), text)


def test_filter_irrelevant():
    assert rb.check_answer(prompt("irrelevant"), rb.IRRELEVANT_SENTENCE) is None
    assert rb.check_answer(prompt("irrelevant"), rb.IRRELEVANT_SENTENCE + " Sorry.") is not None
    assert (
        rb.check_answer(
            prompt("irrelevant"), rb.MISSING_TEMPLATE.format(function="get_weather", params="city")
        )
        is not None
    )


def test_kept_record_is_a_public_sample():
    p = rb.build_prompts(pools(), limit=4, seed=1, exclude=set())[0][0]
    record = rb.to_sample(p, "  answer text \n")
    assert check_sample(record) == []
    assert record["tools"] == [] and record["messages"][-1] == {"role": "assistant", "content": "answer text"}
    assert record["messages"][:2] == p["messages"]
    assert (record["source"], record["source_id"], record["kind"]) == (p["source"], p["source_id"], p["kind"])


# ---- CLI ----------------------------------------------------------------------------------------------


class FakeClient:
    instances: ClassVar[list[FakeClient]] = []

    def __init__(self, server, *, slots=1, texts=None):
        self.server, self.slots, self.calls = server, slots, []
        self.texts = texts
        FakeClient.instances.append(self)

    def health(self):
        return True

    def props(self):
        return {
            "total_slots": self.slots,
            "model_path": "C:/somewhere/Qwen3-1.7B-Q8_0.gguf",
            "build_info": "b1",
        }

    def complete(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return type("C", (), {"text": self.texts(prompt), "finish_reason": "stop"})()

    def close(self):
        pass


def scripted(prompt_text: str) -> str:
    """A right answer for every other prompt, a wrong one for the rest (by the user text's id)."""
    number = int(re.search(r"\((\d+)\)", prompt_text).group(1))
    if number % 2:
        return "I think you should call a function."
    if "pizza" in prompt_text or "tallest" in prompt_text:
        return rb.IRRELEVANT_SENTENCE
    return rb.MISSING_TEMPLATE.format(function="generate_password", params="length")


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    FakeClient.instances = []
    monkeypatch.setattr(rb, "load_candidates", lambda manifest, raw_dir: pools())
    monkeypatch.setattr(rb, "make_renderer", lambda model: lambda messages, tools: json.dumps(messages))
    # the default model folder, models/Qwen3-1.7B, is the Qwen3 hybrid family's (M6's text path), here or not
    monkeypatch.setattr(rb, "folder_family", lambda model, name=None: QWEN3_HYBRID)
    return tmp_path


def run(tmp_path: Path, *extra: str) -> tuple[int, Path, Path]:
    out, manifest = tmp_path / "replay.jsonl", tmp_path / "replay-manifest.json"
    code = rb.main(
        [
            "--server",
            "http://127.0.0.1:9",
            "--out",
            str(out),
            "--manifest",
            str(manifest),
            "--limit",
            "20",
            "--seed",
            "3",
            "--exclude",
            str(tmp_path / "none.jsonl"),
            *extra,
        ]
    )
    return code, out, manifest


def test_dry_run_needs_no_server(stubbed, monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("a dry run must not contact a server")

    monkeypatch.setattr(rb, "LocalClient", refuse)
    code, out, manifest = run(stubbed, "--dry-run")
    assert code == 0 and not out.exists() and not manifest.exists()
    prompts = [json.loads(line) for line in rb.prompts_path(out).read_text(encoding="utf-8").splitlines()]
    assert len(prompts) == 20
    assert {p["kind"] for p in prompts} == {"incomplete", "irrelevant"}


def test_cli_generates_filters_and_caps(stubbed, monkeypatch):
    monkeypatch.setattr(rb, "LocalClient", lambda server: FakeClient(server, texts=scripted))
    code, out, manifest_path = run(stubbed, "--cap", "4")
    assert code == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4
    assert all(check_sample(r) == [] and r["tools"] == [] for r in rows)
    for r in rows:
        answer = r["messages"][-1]["content"]
        assert answer in (
            rb.IRRELEVANT_SENTENCE,
            rb.MISSING_TEMPLATE.format(function="generate_password", params="length"),
        )
    client = FakeClient.instances[0]
    for _, kwargs in client.calls:
        assert kwargs["lora_scale"] == 0.0 and kwargs["temperature"] == 0.0 and kwargs["stop"] == list(STOP)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["kept"]["total"] == 4 and manifest["generated"] == len(client.calls)
    assert manifest["server"] == {"total_slots": 1, "model": "Qwen3-1.7B-Q8_0.gguf", "build": "b1"}
    assert sum(manifest["rejected"].values()) == manifest["generated"] - 4


def test_cli_caps_each_kind_on_its_own_when_asked(stubbed, monkeypatch):
    monkeypatch.setattr(rb, "LocalClient", lambda server: FakeClient(server, texts=scripted))
    code, out, manifest_path = run(stubbed, "--cap", "100", "--cap-per-kind", "1")
    assert code == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["kept"]["incomplete"] == 1 and manifest["kept"]["irrelevant"] == 1
    assert manifest["kept"]["total"] == 2 and manifest["cap_per_kind"] == 1
    assert manifest["kept"]["normalised"] == 0  # the scripted answers are already in the template's form
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert sorted(r["kind"] for r in rows) == ["incomplete", "irrelevant"]


def test_cli_refuses_a_multi_slot_server(stubbed, monkeypatch):
    monkeypatch.setattr(rb, "LocalClient", lambda server: FakeClient(server, slots=4, texts=scripted))
    code, out, _ = run(stubbed)
    assert code == 2 and not out.exists()
    assert FakeClient.instances[0].calls == []


def test_cli_refuses_to_overwrite(stubbed, monkeypatch):
    monkeypatch.setattr(rb, "LocalClient", lambda server: FakeClient(server, texts=scripted))
    assert run(stubbed)[0] == 0
    assert run(stubbed)[0] == 2
    assert run(stubbed, "--force")[0] == 0


def test_cli_excludes_format_mix_ids(stubbed, monkeypatch):
    mix = stubbed / "mix.jsonl"
    mix.write_text(json.dumps({"source": "glaive_v2", "source_id": "2"}) + "\n", encoding="utf-8")
    out = stubbed / "r.jsonl"
    assert (
        rb.main(
            [
                "--out",
                str(out),
                "--manifest",
                str(stubbed / "m.json"),
                "--limit",
                "100",
                "--exclude",
                str(mix),
                "--dry-run",
            ]
        )
        == 0
    )
    prompts = [json.loads(line) for line in rb.prompts_path(out).read_text(encoding="utf-8").splitlines()]
    assert ("glaive_v2", "2") not in {(p["source"], p["source_id"]) for p in prompts}
    assert len(prompts) == 10 - 1 + 15  # ten asking conversations less the excluded one, fifteen irrelevant


def linked_folder(link: Path, target: Path) -> None:
    """link made a link to the folder target: a directory junction on Windows (cmd /c mklink /J, as the run
    worktree's data/raw is one), else a symbolic link; the test is skipped only when neither can be made."""
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        done = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], capture_output=True,
                              text=True, check=False)
        if done.returncode == 0 and link.is_dir():
            return
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"neither a junction nor a symbolic link can be made here: {exc}")


def test_the_manifest_keeps_a_path_through_a_linked_folder_as_spelled_under_the_current_folder(
        stubbed, monkeypatch):
    # the run worktree's data/raw is a junction into the main tree: a resolved path leaves the worktree, and
    # the manifest fell back to the bare file names; the paths are taken lexically, as spelled under the
    # current folder
    worktree, main_tree = stubbed / "worktree", stubbed / "main"
    linked_folder(worktree / "data" / "raw", main_tree / "data" / "raw")
    monkeypatch.chdir(worktree)
    Path("data/raw/format-mix.jsonl").write_text(json.dumps({"source": "glaive_v2", "source_id": "2"}) + "\n",
                                                 encoding="utf-8", newline="")
    monkeypatch.setattr(rb, "LocalClient", lambda server: FakeClient(server, texts=scripted))
    code = rb.main(["--server", "http://127.0.0.1:9", "--out", "data/raw/base-replay-c2.jsonl", "--manifest",
                    "data/public/base-replay-c2-manifest.json", "--limit", "20", "--seed", "3", "--exclude",
                    "data/raw/format-mix.jsonl"])
    assert code == 0
    assert (main_tree / "data" / "raw" / "base-replay-c2.jsonl").is_file()  # written through the link
    manifest = json.loads(Path("data/public/base-replay-c2-manifest.json").read_text(encoding="utf-8"))
    assert manifest["output"]["path"] == "data/raw/base-replay-c2.jsonl"
    assert manifest["prompts"]["path"] == "data/raw/base-replay-c2.prompts.jsonl"
    assert manifest["excluded"] == {"file": "data/raw/format-mix.jsonl", "records": 1}
    # an absolute spelling under the current folder and one with '..' are kept the same way; a path outside
    # the current folder, the link's target among them, keeps the fallback: its file name alone
    assert rb.repo_relative(worktree / "data" / "raw" / "x.jsonl") == "data/raw/x.jsonl"
    assert rb.repo_relative(Path("data/raw/sub/../x.jsonl")) == "data/raw/x.jsonl"
    assert rb.repo_relative(main_tree / "data" / "raw" / "x.jsonl") == "x.jsonl"

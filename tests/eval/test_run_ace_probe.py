"""scripts/run_ace_probe.py (M6 WP4, D3), offline: a fake LocalClient returns scripted answers covering every
scoring branch, and a plain renderer stands in for the chat template; no server or model is run."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import ClassVar

import pytest

from research_agent.agent.models import STOP
from research_agent.eval import ace_probe as ap
from research_agent.eval.lock import write_lock

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rp = load_script("run_ace_probe")

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
GOOD = '[{"name": "get_weather", "arguments": {"city": "Oslo"}}]'

# user text -> (kind, lang, scripted answer); every scoring branch and failure shape once
SCRIPT = {
    "Weather in Oslo, exact?": ("normal", "en", GOOD),
    "Weather in Oslo, wrong value?": (
        "normal",
        "en",
        '[{"name": "get_weather", "arguments": {"city": "Rome"}}]',
    ),
    "Weather in Oslo, kwargs?": ("normal", "zh", '[{"name": "get_weather", "arguments": {city="Oslo"}}]'),
    "Weather in Oslo, tool call?": ("normal", "zh", "<tool_call>\n" + GOOD + "\n</tool_call>"),
    "Weather in Oslo, placeholder?": ("normal", "en", '[{"name": "example_function", "arguments": {}}]'),
    "Weather in Oslo, runaway?": ("normal", "zh", "get_weather(" * 300),
    "Weather in, hit?": ("incomplete", "en", ap.SENTENCE.format(function="get_weather", params="city")),
    "Weather in, wrong param?": (
        "incomplete",
        "zh",
        ap.SENTENCE.format(function="get_weather", params="unit"),
    ),
    "Weather in, called?": ("incomplete", "en", GOOD),
    "Weather in, prose?": ("incomplete", "zh", "Which city do you mean?"),
}


def make_items() -> list[dict]:
    items = []
    for n, (user, (kind, lang, _)) in enumerate(SCRIPT.items()):
        system = ap.system_prompt([WEATHER], lang)
        expected = (
            {"calls": [{"name": "get_weather", "arguments": {"city": "Oslo"}}]}
            if kind == "normal"
            else {"function": "get_weather", "missing": ["city"]}
        )
        items.append(
            {
                "id": f"ace-probe-{kind}-{lang}-{n:03d}",
                "kind": kind,
                "lang": lang,
                "system": system,
                "user": user,
                "tools": [WEATHER],
                "expected": expected,
                "source": "toolace",
                "source_id": str(n),
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            }
        )
    return items


@pytest.fixture
def probe(tmp_path) -> Path:
    path = tmp_path / "ace-probe.jsonl"
    path.write_bytes("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in make_items()).encode())
    write_lock(path)
    return path


class FakeClient:
    instances: ClassVar[list] = []

    def __init__(self, base_url="http://127.0.0.1:8080", *, slots=1, adapters=(), healthy=True):
        self.base_url, self.slots, self._adapters, self.healthy = base_url, slots, list(adapters), healthy
        self.calls: list[dict] = []
        self.closed = False
        FakeClient.instances.append(self)

    def health(self):
        return self.healthy

    def props(self):
        return {
            "total_slots": self.slots,
            "build_info": "b11349-test",
            "model_path": "C:/m/Qwen3-1.7B-Q8_0.gguf",
            "default_generation_settings": {"n_ctx": 8192},
            "chat_template": "{{ long }}",
        }

    def adapters(self):
        return self._adapters

    def complete(self, prompt, **kw):
        self.calls.append({"prompt": prompt, **kw})
        messages = json.loads(prompt)
        assert [m["role"] for m in messages] == ["system", "user"]
        return type("C", (), {"text": SCRIPT[messages[1]["content"]][2], "finish_reason": "stop"})()

    def close(self):
        self.closed = True


def render(messages, tools):
    assert tools is None  # no tools field: the functions are in the system text, as ACEBench sends them
    return json.dumps(messages, ensure_ascii=False)


@pytest.fixture
def fake(monkeypatch):
    FakeClient.instances = []
    state = {"slots": 1, "adapters": [], "healthy": True}
    monkeypatch.setattr(
        rp,
        "LocalClient",
        lambda url: FakeClient(
            url, slots=state["slots"], adapters=state["adapters"], healthy=state["healthy"]
        ),
    )
    monkeypatch.setattr(rp, "make_renderer", lambda model: render)
    return state


def run(tmp_path, probe, *args):
    out = tmp_path / "result.json"
    code = rp.main(["--probe", str(probe), "--out", str(out), *args])
    return code, out


def test_b0_run_scores_every_branch_and_writes_the_result(tmp_path, probe, fake):
    code, out = run(tmp_path, probe, "--system", "b0")
    assert code == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    client = FakeClient.instances[-1]
    assert client.closed and len(client.calls) == len(SCRIPT)
    for call in client.calls:
        assert call["lora_scale"] == 0.0 and call["temperature"] == 0.0 and call["max_tokens"] == 1200
        assert call["stop"] == list(STOP) and call.get("logprobs") is None
        assert call.get("cache_prompt", False) is False
    k = r["kinds"]
    assert k["normal"]["n"] == 6 and k["normal"]["parsable"] == 3 and k["normal"]["exact"] == 1
    assert k["incomplete"] == {
        "n": 4,
        "sentence": 2,
        "sentence_rate": 0.5,
        "parameter": 1,
        "parameter_rate": 0.25,
        "called": 1,
        "called_rate": 0.25,
    }
    shapes = r["shapes"]["normal"]["all"]
    assert (shapes["kwargs"], shapes["tool_call"], shapes["placeholder"], shapes["runaway"]) == (1, 1, 1, 1)
    assert r["languages"]["en"]["normal"]["parsable_rate"] == 1.0
    m = r["manifest"]
    assert m["system"] == "b0" and m["adapter"] is None and m["lora_scale"] == 0.0
    assert m["probe"]["sha256"] == hashlib.sha256(probe.read_bytes()).hexdigest()
    assert m["probe"]["items"] == len(SCRIPT)
    assert m["server"]["total_slots"] == 1 and m["server"]["build_info"] == "b11349-test"
    assert m["server"]["model"] == "Qwen3-1.7B-Q8_0.gguf" and m["server"]["n_ctx"] == 8192
    assert "chat_template" not in m["server"]
    assert m["generation"]["tools_field"] is None and m["generation"]["thinking"].startswith("off")
    assert [i["id"] for i in r["items"]] == [i["id"] for i in make_items()]
    assert all("answer" in i for i in r["items"])
    assert out.read_bytes().endswith(b"\n") and b"\r\n" not in out.read_bytes()


def test_ft_run_needs_its_adapter_loaded_and_records_its_sha(tmp_path, probe, fake):
    gguf = tmp_path / "checkpoint-1-f16.gguf"
    gguf.write_bytes(b"adapter")
    fake["adapters"] = [{"id": 0, "path": f"C:/x/{gguf.name}", "scale": 0.0}]
    code, out = run(tmp_path, probe, "--system", "ft", "--adapter", str(gguf))
    assert code == 0
    r = json.loads(out.read_text(encoding="utf-8"))
    assert r["manifest"]["adapter"]["sha256"] == hashlib.sha256(b"adapter").hexdigest()
    assert r["manifest"]["lora_scale"] == 1.0
    assert all(c["lora_scale"] == 1.0 for c in FakeClient.instances[-1].calls)


@pytest.mark.parametrize(
    "setup,args",
    [
        ({"adapters": []}, ["--system", "ft", "--adapter", "GGUF"]),  # no adapter loaded
        ({"adapters": [{"id": 0, "path": "other.gguf"}]}, ["--system", "b4", "--adapter", "GGUF"]),
        ({"slots": 4}, ["--system", "b0"]),
        ({"healthy": False}, ["--system", "b0"]),
        ({}, ["--system", "b0", "--adapter", "GGUF"]),  # b0 takes no adapter
        ({}, ["--system", "ft", "--adapter", "missing.gguf"]),
    ],
)
def test_usage_errors_exit_2_and_write_nothing(tmp_path, probe, fake, setup, args):
    gguf = tmp_path / "a.gguf"
    gguf.write_bytes(b"x")
    fake.update(setup)
    args = [str(gguf) if a == "GGUF" else a for a in args]
    if fake.get("adapters"):
        fake["adapters"] = [dict(a, path=a["path"]) for a in fake["adapters"]]
    code, out = run(tmp_path, probe, *args)
    assert code == 2 and not out.exists()


def test_refuses_to_overwrite_without_force(tmp_path, probe, fake):
    assert run(tmp_path, probe, "--system", "b0")[0] == 0
    assert run(tmp_path, probe, "--system", "b0")[0] == 2
    assert run(tmp_path, probe, "--system", "b0", "--force")[0] == 0


def test_refuses_a_probe_whose_lock_does_not_match(tmp_path, probe, fake):
    probe.write_bytes(probe.read_bytes() + b"\n")
    assert run(tmp_path, probe, "--system", "b0")[0] == 2


def test_refuses_test_material(tmp_path, fake):
    path = tmp_path / "items.jsonl"
    item = {**make_items()[0], "id": "req-test-0001-s1"}
    path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    assert rp.main(["--probe", str(path), "--out", str(tmp_path / "r.json"), "--system", "b0"]) == 4
    assert not FakeClient.instances or not FakeClient.instances[-1].calls


def test_default_out_is_named_after_the_system():
    args = rp.build_parser().parse_args(["--system", "b4"])
    assert rp.default_out(args) == Path("results/ace-probe-b4.json")

"""research_agent.live.frozen: the code's hashes pinned to the evaluated ones, the frozen record and the
session configuration checked field by field (over a fake final system, research_agent's own family layer
loading a fake family), a run's identity held against them, and, where the data is here, the real record."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from support.live_fakes import final_server, pin_record, write_final

from research_agent.agent.context import family_token_counter
from research_agent.agent.prompts import PROMPT_HASH, PROMPT_VERSION
from research_agent.eval.lock import sha256_of
from research_agent.eval.systems import build_system
from research_agent.live import frozen as frozen_module
from research_agent.live.frozen import (
    EVALUATED,
    EVALUATED_SERVER,
    SLOTS,
    TRAINING_LENGTH,
    FrozenMismatch,
    code_hashes,
    identity_problems,
    load_frozen,
)
from research_agent.llm.local import LocalClient
from research_agent.tools.schemas import SCHEMA_HASH

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---- the pins ---------------------------------------------------------------------------------------------


def test_the_code_is_the_evaluated_code_and_the_pins_are_the_evaluated_values():
    assert EVALUATED == {"prompt_version": "m2-2026-10-13", "prompt_hash": "c827097301c986b1",
                         "schema_hash": "7d831712685e42d0", "rules_version": "m7-wp3-1"}
    assert code_hashes() == EVALUATED
    assert (PROMPT_VERSION, PROMPT_HASH, SCHEMA_HASH) == ("m2-2026-10-13", "c827097301c986b1",
                                                          "7d831712685e42d0")
    assert EVALUATED_SERVER == {"port": 8080, "ctx_size": 8192, "timeout_s": 120,
                                "llama_cpp_build": "b11349-fb4b2737a"}
    assert TRAINING_LENGTH == 5120


def test_the_live_server_has_the_slots_of_the_test_session():
    assert load_script("run_test").SLOTS == SLOTS == 1


# ---- load_frozen over a fake final system -----------------------------------------------------------------


def rewrite(paths: dict, name: str, change, monkeypatch=None) -> None:
    """change the JSON file paths[name] in place; a changed final.json is named again by the configuration
    (its SHA-256) and, given monkeypatch, pinned again as the record M7 froze, so only the change itself
    disagrees."""
    path = Path(paths[name])
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    if name == "final":
        rewrite(paths, "config",
                lambda config: config["systems"]["final"]["frozen"].update(sha256=sha256_of(path)))
        if monkeypatch is not None:
            pin_record(monkeypatch, path)


def mismatch(paths: dict) -> list[str]:
    with pytest.raises(FrozenMismatch) as caught:
        load_frozen(paths["final"], paths["config"])
    assert str(caught.value) == "; ".join(caught.value.problems)
    return caught.value.problems


def test_a_record_that_describes_the_system_loads_with_what_a_run_needs(tmp_path, monkeypatch):
    paths = write_final(tmp_path / "final", monkeypatch, votes=3)
    frozen = load_frozen(paths["final"], paths["config"])
    assert frozen.settings.votes == 3 and frozen.settings.reasoning_budget == 32
    assert frozen.vote_mode == "final" and frozen.rules and frozen.tokenizer == paths["model"]
    assert frozen.family.spec.name == "fake-only"
    assert (frozen.base_path, frozen.adapter_path) == (paths["base"], paths["adapter"])
    assert (frozen.server.port, frozen.server.ctx_size, frozen.server.idle_max_mib) == (8080, 8192, 1500)
    assert frozen.frozen_sha256 == sha256_of(paths["final"])
    assert frozen.config_sha256 == sha256_of(paths["config"])


def test_code_that_is_not_the_evaluated_code_is_refused_before_anything_is_read(tmp_path, monkeypatch):
    monkeypatch.setitem(frozen_module.EVALUATED, "prompt_hash", "0" * 16)
    problems = mismatch({"final": str(tmp_path / "absent.json"), "config": str(tmp_path / "absent.json")})
    assert problems == [f"prompt_hash: this code has {PROMPT_HASH!r}, the evaluated system {'0' * 16!r}"]


@pytest.mark.parametrize(("name", "change", "named"), [
    ("final", lambda r: r["family"].update(name="another"), "family:"),
    ("final", lambda r: r["generation"].update(answer_tokens=1024), "generation:"),
    ("final", lambda r: r.update(votes=4), "generation:"),
    ("final", lambda r: r["rules"].update(version="m7-wp3-0"), "rules:"),
    ("final", lambda r: r.update(vote_mode="majority"), "vote_mode:"),
    ("final", lambda r: r.update(glossary={"path": "data/glossary/zh-en.json"}), "glossary:"),
    ("final", lambda r: r["adapter"].update(sha256="0" * 64), "adapter:"),
    ("final", lambda r: r["base"].update(path="missing.gguf"), "base:"),
    ("config", lambda c: c["systems"]["final"]["frozen"].update(sha256="0" * 64),
     "systems.final.frozen.sha256"),
    ("config", lambda c: c["systems"]["final"].update(rules=False), "systems.final.rules"),
    ("config", lambda c: c["adapters"]["final"].update(gguf="other.gguf"), "adapters.final.gguf"),
    ("config", lambda c: c["adapters"]["final"]["generation"].update(reasoning_budget=1280),
     "adapters.final.generation.reasoning_budget"),
    ("config", lambda c: c["server"].update(ctx_size=4096), "server.ctx_size"),
    ("config", lambda c: c["server"].update(llama_cpp_build="b9999"), "server.llama_cpp_build"),
])
def test_every_field_that_disagrees_is_named(tmp_path, monkeypatch, name, change, named):
    paths = write_final(tmp_path / "final", monkeypatch)
    rewrite(paths, name, change, monkeypatch)
    problems = mismatch(paths)
    assert any(problem.startswith(named) for problem in problems), problems
    assert not any(problem.startswith("frozen record:") for problem in problems)


def test_several_disagreements_are_named_together(tmp_path, monkeypatch):
    paths = write_final(tmp_path / "final", monkeypatch)
    Path(paths["adapter"]).write_bytes(b"another adapter")
    rewrite(paths, "config", lambda c: c["server"].update(port=8081))
    problems = mismatch(paths)
    assert [problem.split(":")[0] for problem in problems] == ["adapter", "server.port"]


def test_a_record_frozen_again_with_a_configuration_naming_it_is_not_the_evaluated_record(tmp_path,
                                                                                         monkeypatch):
    paths = write_final(tmp_path / "final", monkeypatch)
    rewrite(paths, "final", lambda r: r.update(vote_mode="reproduction"))  # not pinned again
    rewrite(paths, "config", lambda c: c["systems"]["final"].update(vote_mode="reproduction"))
    problems = mismatch(paths)
    assert [problem.split(":")[0] for problem in problems] == ["frozen record", "vote_mode"]
    assert problems[0].endswith(f"the record M7 froze {frozen_module.FROZEN_SHA256}")
    assert problems[1] == ("vote_mode: the frozen record holds 'reproduction', the final system votes in "
                           "'final' mode")


def test_files_and_fields_that_are_missing_are_named(tmp_path, monkeypatch):
    paths = write_final(tmp_path / "final", monkeypatch)
    assert "cannot be read" in mismatch({**paths, "config": str(tmp_path / "absent.json")})[0]
    rewrite(paths, "final", lambda r: r.pop("generation"), monkeypatch)
    assert mismatch(paths) == [f"{Path(paths['final']).as_posix()} has no generation"]
    rewrite(paths, "final", lambda r: r.update(tokenizer=str(tmp_path / "no-model")), monkeypatch)
    assert mismatch(paths)[0].startswith(f"family: the family of {tmp_path / 'no-model'} cannot be loaded")


# ---- identity_problems ------------------------------------------------------------------------------------


def built_final(tmp_path, monkeypatch):
    """The fake final system loaded and built as scripts/ask.py builds it: (frozen, info, props)."""
    paths = write_final(tmp_path / "final", monkeypatch)
    frozen = load_frozen(paths["final"], paths["config"])
    fake = final_server(paths, [])
    built = build_system("ft", tokenizer=frozen.tokenizer, family=frozen.family, generation=frozen.settings,
                         count_tokens=family_token_counter(frozen.family, frozen.settings.mode), rules=True,
                         vote_mode=frozen.vote_mode, local_client=LocalClient(transport=fake.transport()))
    return frozen, built.info, fake.props


def test_the_system_built_from_the_record_on_the_evaluated_server_has_its_identity(tmp_path, monkeypatch):
    frozen, info, props = built_final(tmp_path, monkeypatch)
    assert identity_problems(info, frozen, props) == []
    manifest = {**info, **{key: EVALUATED[key] for key in ("prompt_version", "prompt_hash", "schema_hash")}}
    assert identity_problems(manifest, frozen, props) == []  # a run_e2e manifest's extra fields agree
    assert identity_problems(info, frozen, {k: v for k, v in props.items()
                                            if k != "default_generation_settings"}) == []


@pytest.mark.parametrize(("info_changes", "props_changes", "named"), [
    ({"family": {"name": "another"}}, {}, "family:"),
    ({"generation": {"mode": "think"}}, {}, "generation:"),
    ({"vote_mode": "reproduction"}, {}, "vote_mode:"),
    ({"rules": None}, {}, "rules:"),
    ({"glossary": {"path": "g.json"}}, {}, "glossary:"),
    ({"backend": "hf"}, {}, "backend:"),
    ({"lora_scale": 0.0}, {}, "lora_scale:"),
    ({"tokenizer": "models/Qwen3-1.7B"}, {}, "tokenizer:"),
    ({"adapters": []}, {}, "adapters: the run records [], not exactly one adapter"),
    ({"adapters": [{"id": 0, "path": "other.gguf", "sha256": "0" * 64}]}, {}, "adapters: the server loaded"),
    ({"prompt_hash": "0" * 16}, {}, "prompt_hash:"),
    ({}, {"total_slots": 4}, "server: 4 slots"),
    ({}, {"build_info": "b9999"}, "server: build 'b9999'"),
    ({}, {"model_path": "models/Qwen3-1.7B-Q8_0.gguf"}, "server: model"),
    ({}, {"default_generation_settings": {"n_ctx": 4096}}, "server: context size 4096"),
])
def test_each_disagreement_of_a_run_is_named(tmp_path, monkeypatch, info_changes, props_changes, named):
    frozen, info, props = built_final(tmp_path, monkeypatch)
    problems = identity_problems({**info, **info_changes}, frozen, {**props, **props_changes})
    assert problems and problems[0].startswith(named), problems
    assert {problem.split(":")[0] for problem in problems} == {named.split(":")[0]}  # nothing else


def test_a_run_without_server_props_cannot_be_checked(tmp_path, monkeypatch):
    frozen, info, _ = built_final(tmp_path, monkeypatch)
    assert identity_problems(info, frozen, None) == [
        "server: no /props recorded, so its slots, build and model cannot be checked"]
    second = {**info, "adapters": [*info["adapters"], *info["adapters"]]}
    assert identity_problems(second, frozen, None)[0].startswith("adapters:")


# ---- the real record --------------------------------------------------------------------------------------


@pytest.mark.needs_data("results/final-m7.json", "results/test-config-m7.json", "models/Nanbeige4.1-3B",
                        "models/Nanbeige4.1-3B-Q8_0.gguf", "runs/train/ft3/gguf/checkpoint-1635-f16.gguf")
def test_the_frozen_record_holds_here(monkeypatch):
    monkeypatch.chdir(REPO)  # the record's paths are the repository's
    frozen = load_frozen()
    assert frozen.frozen_sha256 == frozen_module.FROZEN_SHA256  # the pin is the record here
    assert frozen.settings.votes == 5 and frozen.settings.reasoning_budget == 1280
    assert frozen.vote_mode == "final" and frozen.rules and frozen.family.spec.name == "nanbeige4.1-3b"
    assert frozen.server.ctx_size == 8192 and frozen.server.idle_max_mib == 1500


@pytest.mark.needs_data("runs/train/ft3/manifest.json")
def test_the_training_length_is_ft3s():
    manifest = json.loads((REPO / "runs/train/ft3/manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["sft"]["max_length"] == TRAINING_LENGTH

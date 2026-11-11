"""scripts/check_final_identity.py offline: a session.json of scripts/ask.py and a report.json of
scripts/run_e2e.py held against the fake final system (tests/support/live_fakes.py); exit 0 when the run was
it, on a clean working tree, 1 otherwise."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from support.live_fakes import final_server, write_final

from research_agent.agent.context import family_token_counter
from research_agent.eval.systems import build_system
from research_agent.live.frozen import EVALUATED, TRAINING_LENGTH, FrozenMismatch, load_frozen
from research_agent.llm.local import LocalClient

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = load_script("check_final_identity")


@pytest.fixture
def final(tmp_path, monkeypatch):
    """The fake final system and what a run of it records: (frozen, info, props)."""
    paths = write_final(tmp_path / "final", monkeypatch)
    frozen = load_frozen(paths["final"], paths["config"])
    fake = final_server(paths, [])
    built = build_system("ft", tokenizer=frozen.tokenizer, family=frozen.family, generation=frozen.settings,
                         count_tokens=family_token_counter(frozen.family, frozen.settings.mode), rules=True,
                         vote_mode=frozen.vote_mode, local_client=LocalClient(transport=fake.transport()))
    return frozen, built.info, fake.props


def write(tmp_path, data: dict) -> str:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def report(info: dict, props: dict, **changes) -> dict:
    """A report.json of scripts/run_e2e.py for final's run, on a clean working tree: its manifest."""
    manifest = {**info, **{key: EVALUATED[key] for key in ("prompt_version", "prompt_hash", "schema_hash")},
                "context_check": info["tokenizer"], "training_length": TRAINING_LENGTH,
                "steps_over_training_length": 0, "server_props": props, "git_commit": "0" * 40,
                "git_dirty": False, **changes}
    return {"manifest": manifest, "summary": {}, "tasks": []}


def session(info: dict, props: dict, **changes) -> dict:
    """A session.json of scripts/ask.py run on a clean working tree."""
    return {"kind": "batch", "info": info, "props": props, "git_commit": "0" * 40, "git_dirty": False,
            **changes}


def test_an_ask_session_and_an_end_to_end_report_of_the_final_system_pass(tmp_path, final, capsys):
    frozen, info, props = final
    assert check.main([write(tmp_path, session(info, props))], load=lambda: frozen) == 0
    assert capsys.readouterr().out.startswith("OK: ")
    assert check.main([write(tmp_path, report(info, props))], load=lambda: frozen) == 0


@pytest.mark.parametrize(("changes", "named"), [
    ({"training_length": 4096}, "training_length: the report measured against 4096"),
    ({"context_check": None, "steps_over_training_length": None}, "steps_over_training_length:"),
    ({"server_props": None}, "server: no /props recorded"),
    ({"vote_mode": "reproduction"}, "vote_mode:"),
    ({"git_dirty": True}, "git_dirty: the working tree had uncommitted changes"),
    ({"git_dirty": None}, "git_dirty: the run recorded None, not whether its working tree was clean"),
])
def test_a_report_that_was_not_the_final_system_fails_and_names_why(tmp_path, final, capsys, changes, named):
    frozen, info, props = final
    assert check.main([write(tmp_path, report(info, props, **changes))], load=lambda: frozen) == 1
    out = capsys.readouterr().out
    assert out.startswith("FAIL: ") and f"  {named}" in out


@pytest.mark.parametrize("changes", [{"git_dirty": True}, {"git_dirty": None}, {"git_dirty": "no"}])
def test_an_ask_session_on_a_working_tree_not_known_to_be_clean_fails(tmp_path, final, capsys, changes):
    frozen, info, props = final
    data = session(info, props, **changes)
    assert check.main([write(tmp_path, data)], load=lambda: frozen) == 1
    assert "  git_dirty: " in capsys.readouterr().out
    del data["git_dirty"]  # a session that recorded nothing of its tree
    assert check.main([write(tmp_path, data)], load=lambda: frozen) == 1


def test_files_that_are_neither_kind_and_a_frozen_system_that_does_not_hold_fail(tmp_path, final, capsys):
    frozen, info, props = final
    assert check.main([write(tmp_path, {"tasks": []})], load=lambda: frozen) == 1
    assert check.main([str(tmp_path / "absent.json")], load=lambda: frozen) == 1

    def mismatched():
        raise FrozenMismatch(["adapter: another file"])

    assert check.main([write(tmp_path, session(info, props))], load=mismatched) == 1
    out = capsys.readouterr().out
    assert "neither a report" in out and "the frozen system does not hold: adapter: another file" in out

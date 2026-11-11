"""Stand-ins for the tests of research_agent.live and scripts/ask.py: a frozen final system over a fake
family, its record, session configuration and GGUF files written the way scripts/freeze_final.py and the M7
session configuration have them, the server that serves it (FakeLlama, its /props the evaluated server's),
and a fake clock. Nothing is loaded and nothing goes on the network.

write_final writes, under a folder: a fake model folder that load_family(path) resolves (llama_fakes.
install_fake_family: the "only" thinking style, think mode sampled with SAMPLING), adapter.gguf and base.gguf
(a few bytes each), final.json (the frozen record: the adapter and the base with their SHA-256, the tokenizer
folder, the family's record, the think generation with the votes, the vote mode, the rules of this code, no
glossary) and config.json (systems.final naming final.json and its SHA-256, adapters.final, the evaluated
server and the GPU's idle threshold), and pins final.json, for the test, as the record M7 froze (pin_record:
live.frozen.FROZEN_SHA256). load_frozen(paths["final"], paths["config"]) then holds.
"""

from __future__ import annotations

import json
from pathlib import Path

from research_agent.agent.rules import RULES, RULES_VERSION
from research_agent.eval.lock import sha256_of
from research_agent.eval.systems import with_votes
from research_agent.family.core import GREEDY, Decoding, Mode, load_family
from research_agent.live import frozen
from research_agent.live.frozen import EVALUATED_SERVER, SAMPLED
from research_agent.llm.generation import GenerationSettings
from support.llama_fakes import FakeLlama, Scripted, install_fake_family

SAMPLING = Decoding(temperature=0.6, top_p=0.95, top_k=0, min_p=0.0)
BUDGET = 32  # the reasoning budget of the fake final system
IDLE_MIB = 1500
TEXT_ANSWER = ("Thinking it over.", "\n</think>\n\n", "Here is the answer.")  # a think turn without calls


def write_final(folder: Path, monkeypatch, *, votes: int = 2) -> dict[str, str]:
    """The fake final system's files under folder (module docstring); their paths by name: model, adapter,
    base, final, config."""
    folder.mkdir(parents=True, exist_ok=True)
    model = folder / "model"
    install_fake_family(monkeypatch, model, "only", inference_mode=Mode.THINK,
                        decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY}, reasoning_budget=BUDGET)
    family = load_family(model)
    adapter, base = folder / "adapter.gguf", folder / "base.gguf"
    adapter.write_bytes(b"GGUF adapter")
    base.write_bytes(b"GGUF base model")
    settings = with_votes(GenerationSettings.for_family(family, mode="think", reasoning_budget=BUDGET,
                                                        base_seed=0), votes)
    final = folder / "final.json"
    record = {"adapter": {"path": str(adapter), "sha256": sha256_of(adapter)},
              "base": {"path": str(base), "sha256": sha256_of(base)}, "tokenizer": str(model),
              "family": family.record(), "generation": settings.record(), "votes": votes,
              "vote_mode": "final", "rules": {"version": RULES_VERSION, "names": list(RULES)},
              "glossary": None}
    final.write_text(json.dumps(record, indent=2), encoding="utf-8")
    config = folder / "config.json"
    config.write_text(json.dumps({
        "systems": {"final": {"votes": votes, "vote_mode": "final", "rules": True, "glossary": None,
                              "frozen": {"path": str(final), "sha256": sha256_of(final)}}},
        "adapters": {"final": {"gguf": str(adapter), "sha256": sha256_of(adapter), "model": str(base),
                               "tokenizer": str(model),
                               "generation": {"mode": "think", "decoding": SAMPLED,
                                              "reasoning_budget": BUDGET, "base_seed": 0}}},
        "server": dict(EVALUATED_SERVER), "gpu": {"idle_max_mib": IDLE_MIB, "sample_ms": 200}}, indent=2),
        encoding="utf-8")
    pin_record(monkeypatch, final)
    return {"model": str(model), "adapter": str(adapter), "base": str(base), "final": str(final),
            "config": str(config)}


def pin_record(monkeypatch, path: str | Path) -> None:
    """Take the record at path, as it is now, for the record M7 froze (live.frozen.FROZEN_SHA256), for the
    test that calls it only."""
    monkeypatch.setattr(frozen, "FROZEN_SHA256", sha256_of(path))


def server_props(base: str, **changes) -> dict:
    """The /props of the evaluated server over the base model base, with changes."""
    return {"build_info": EVALUATED_SERVER["llama_cpp_build"], "total_slots": 1, "model_path": base,
            "default_generation_settings": {"n_ctx": EVALUATED_SERVER["ctx_size"]}, **changes}


def final_server(paths: dict[str, str], script: list[Scripted], **props) -> FakeLlama:
    """FakeLlama serving the fake final system: its adapter loaded, the evaluated server's /props."""
    return FakeLlama(script, adapters=[{"id": 0, "path": paths["adapter"], "scale": 1.0}],
                     props=server_props(paths["base"], **props))


def text_turns(candidates: int) -> list[Scripted]:
    """One think completion without calls for each candidate of a voted step."""
    return [Scripted(TEXT_ANSWER) for _ in range(candidates)]


class FakeClock:
    """A clock that moves only when something sleeps on it; every sleep is kept."""

    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now, self.sleeps = start, []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

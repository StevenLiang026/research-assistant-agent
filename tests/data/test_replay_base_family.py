"""scripts/replay_base.py's family path (M7 WP-J, docs/m7-计划.md): the bake-off winner, any registered
family, replays itself in its native mode for FT3; the Qwen3 hybrid family in no_think mode keeps M6's text
path.

Offline: the model folder is a fake family's (tests/support/llama_fakes.install_fake_family: the folder holds
the fake template, a transformers stand-in returns the fake tokenizer and the registry holds that family
alone), the server a scripted llama-server behind httpx (FakeLlama) that answers each prompt with the
completions planned for the number its user message carries, and the pools are stubbed in place of the Hugging
Face cache. scripts/train.py's dry run then reads the replay's samples with its sidecar, and a request's
prompt is held against the prompt half of its sample as the training path renders it (render_family_sample),
not against a copy of the replay's own calls. Three tests need this machine's own files (the Hugging Face
cache and data/raw/public; models/) and are skipped without them: the M6 dry run byte for byte, and the plans
of two real model folders.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest
from support.family_fakes import TURN_END, FakeFamilyTokenizer, fake_family, fake_transformers, model_folder
from support.llama_fakes import ID_BASE, PROPS, FakeLlama, Scripted, install_fake_family
from test_replay_base import glaive_asking, glaive_no_call
from test_synth import load_script

from research_agent.agent.types import api_to_hf
from research_agent.data import public_pool
from research_agent.data.reasoning import ROW_KEYS, ReasoningIndex, answer_sha256
from research_agent.family import registry
from research_agent.family.core import (
    GREEDY,
    Decoding,
    FamilyError,
    LoadedFamily,
    Mode,
    TokenBoundaryError,
    load_family,
)
from research_agent.family.registry import QWEN3_HYBRID
from research_agent.llm.cache import request_key
from research_agent.llm.generation import GenerationSettings, request_seed
from research_agent.train.samples import render_family_sample

rb = load_script("replay_base")
train = load_script("train")

REPO = Path(__file__).resolve().parents[2]
MISSING = rb.MISSING_TEMPLATE.format(function="generate_password", params="length")
SINGULAR = "Cannot call generate_password yet: the required parameter length was not given."
IRRELEVANT = rb.IRRELEVANT_SENTENCE
SAMPLING = Decoding(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)
BUDGET = 64  # the fake thinking family's reasoning budget R
SERVED = {**PROPS, "model_path": "C:/models/Fake-4B-Q8_0.gguf"}
ADAPTER = {"id": 0, "path": "runs/train/x/adapter.gguf", "scale": 1.0}
GENERATED = [50_000, 50_001, 50_002]  # FakeLlama's ids for a three-token completion (ID_BASE + index)


def small_pools() -> dict[str, list[dict]]:
    """Four conversations that ask for a missing value (1-4) and four that call nothing (11-14): --limit 8
    draws all of them."""
    return {"glaive_v2": [glaive_asking(str(n)) for n in (1, 2, 3, 4)]
            + [glaive_no_call(str(n)) for n in (11, 12, 13, 14)], "toolace": []}


def user_number(content: str) -> int:
    """The number a user message of small_pools carries."""
    return int(re.search(r"\((\d+)\)", content).group(1))


def number(prompt: str | list[int], decode=None) -> int:
    """The number the user message of a prompt carries (the prompt as text, or as ids, which decode reads:
    the fake tokenizer's by default)."""
    text = prompt if isinstance(prompt, str) else (decode or FakeFamilyTokenizer().decode)(prompt)
    return user_number(text.split("<|im_start|>user\n", 1)[1])


def closed(reasoning: str, answer: str) -> Scripted:
    """A completion after a THINK prompt that closes its reasoning, then answers."""
    return Scripted(((reasoning,) if reasoning else ()) + ("\n</think>", "\n\n", answer))


class ReplayServer(FakeLlama):
    """FakeLlama answering each completion request with the next completion planned for its prompt (by the
    number of the prompt's user message, its ids read with decode); users lists that number for every
    completion request, in order."""

    def __init__(self, plans: dict[int, list[Scripted]], *, decode=None, **kwargs) -> None:
        super().__init__([], **kwargs)
        self.plans = {n: list(planned) for n, planned in plans.items()}
        self.decode = decode
        self.users: list[int] = []

    def handle(self, request):
        if request.url.path == "/v1/completions":
            n = number(json.loads(request.content)["prompt"], self.decode)
            self.users.append(n)
            self.script = [self.plans[n].pop(0)]
        return super().handle(request)


# What the thinking family writes for each prompt: every case the filter and the sidecar tell apart.
THINK_PLANS = {
    1: [closed("The user wants a password but gave no length.", MISSING)],  # kept, a row
    2: [closed("Only the length is missing.", SINGULAR)],  # kept in the canonical form, a row
    3: [Scripted(("I must", " check", " the schema"), "length"), Scripted((MISSING,))],  # closed by force
    4: [closed(f"I could reply: {MISSING}", "Sure! How long should it be?")],  # the sentence in the reasoning
    11: [closed("Ordering pizza is out of scope.", IRRELEVANT)],  # kept, a row
    12: [closed("", IRRELEVANT)],  # kept, its reasoning empty: no row
    13: [Scripted(("<think>", "nested", "\n</think>", "\n\n", IRRELEVANT))],  # kept, a think tag: no row
    14: [closed(IRRELEVANT, "I can only generate passwords.")],  # the sentence in the reasoning only
}
REASONING = {1: "The user wants a password but gave no length.", 2: "Only the length is missing.",
             3: "I must check the schema", 11: "Ordering pizza is out of scope."}  # the rows' reasoning
NO_THINK_PLANS = {1: [Scripted((MISSING,))], 2: [Scripted((SINGULAR,))], 3: [Scripted(("Sure, how long?",))],
                  4: [Scripted((MISSING,))], 11: [Scripted((IRRELEVANT,))], 12: [Scripted(("No.",))],
                  13: [Scripted((IRRELEVANT,))], 14: [Scripted((IRRELEVANT,))]}


def refuse(*args, **kwargs):
    raise AssertionError("the family path never renders text with the old renderer")


def thinking_folder(monkeypatch, tmp_path: Path, **changes) -> Path:
    """A model folder of a fake family that always thinks and samples, as the bake-off's thinking winners."""
    folder = tmp_path / "model"
    install_fake_family(monkeypatch, folder, "only", inference_mode=Mode.THINK, reasoning_budget=BUDGET,
                        decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: SAMPLING}, **changes)
    return folder


def run(tmp_path: Path, monkeypatch, server: FakeLlama, model: Path, *extra: str,
        renderer=refuse) -> tuple[int, Path, Path]:
    """main() over small_pools against server, with the replay's outputs in tmp_path."""
    monkeypatch.setattr(rb, "LocalClient", lambda url: server.client())
    monkeypatch.setattr(rb, "load_candidates", lambda manifest, raw_dir: small_pools())
    monkeypatch.setattr(rb, "make_renderer", renderer)
    out, manifest = tmp_path / "self-replay.jsonl", tmp_path / "self-replay-manifest.json"
    code = rb.main(["--server", "http://127.0.0.1:9", "--out", str(out), "--manifest", str(manifest),
                    "--model", str(model), "--limit", "8", "--seed", "3", "--exclude",
                    str(tmp_path / "none.jsonl"), *extra])
    return code, out, manifest


def rows_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def trained_prompt(family: LoadedFamily, sample: dict, reasoning: str | None) -> list[int]:
    """The ids of the prompt half of a sample as scripts/train.py renders it (train.samples.
    render_family_sample, with the sample's own tools): THINK with a reasoning, NO_THINK without one."""
    *context, target = sample["messages"]
    rendered = render_family_sample(family, context, target, tools=sample["tools"], reasoning=reasoning)
    assert rendered["mode"] == str(Mode.THINK if reasoning is not None else Mode.NO_THINK)
    return rendered["input_ids"][:rendered["completion_start"]]


def prompt_ids(model: Path, mode: Mode, prompts: Path | None = None) -> dict[int, list[int]]:
    """The ids of every drawn prompt (of prompts, the replay's prompts file by default) in mode, by the number
    of its user message: the prompt half of the sample the prompt makes (rb.to_sample, a placeholder answer)
    as the training path renders it (trained_prompt; THINK with a placeholder reasoning in think mode). So a
    request is held against the prompt its sample trains under, not against a copy of the replay's own
    calls."""
    family = load_family(model)
    reasoning = "A placeholder reasoning." if mode is Mode.THINK else None
    return {user_number(p["messages"][1]["content"]): trained_prompt(family, rb.to_sample(p, IRRELEVANT),
                                                                     reasoning)
            for p in rows_of(prompts or model.parent / "self-replay.prompts.jsonl")}


def kept_numbers(out: Path) -> list[int]:
    """The number of each kept sample's user message, in the output's order."""
    return [user_number(k["messages"][1]["content"]) for k in rows_of(out)]


def train_dry_run(tmp_path: Path, model: Path, data: Path, sidecar: Path, capsys) -> dict:
    """scripts/train.py's dry run over data with the sidecar and the family of model: its summary line."""
    capsys.readouterr()
    assert train.main([str(data), "--out", str(tmp_path / "run"), "--model", str(model), "--dry-run",
                       "--reasoning", str(sidecar), "--no-overlap-check", "--overlap-report",
                       str(tmp_path / "no-overlap-report.json"), "--max-length", str(10**6)],
                      load_family=lambda folder, name: load_family(folder, name=name)) == 0
    return json.loads(capsys.readouterr().out.splitlines()[-1].removeprefix("summary: "))


# ---- a family that thinks: THINK, its sampling, its budget, a seed per request, the sidecar ----------------


def test_a_thinking_family_replays_itself_in_think_mode_with_a_sidecar_for_its_kept_answers(
        tmp_path, monkeypatch):
    model = thinking_folder(monkeypatch, tmp_path)
    server = ReplayServer(THINK_PLANS, props=SERVED, adapters=[ADAPTER])
    sidecar = tmp_path / "reasoning" / "self-replay-reasoning.jsonl"
    code, out, manifest_path = run(tmp_path, monkeypatch, server, model, "--reasoning-out", str(sidecar))
    assert code == 0
    ids = prompt_ids(model, Mode.THINK)
    family = load_family(model)
    # the requests: the THINK prompt's ids, R tokens, the family's sampling, a seed per request and phase,
    # the base model (lora scale 0), and the forced close of prompt 3 after its three reasoning tokens
    assert sorted(server.users) == [1, 2, 3, 3, 4, 11, 12, 13, 14]
    seen: set[int] = set()
    for n, body in zip(server.users, server.requests, strict=True):
        phase = 2 if n in seen else 1
        seen.add(n)
        if phase == 1:
            assert body["prompt"] == ids[n] and body["max_tokens"] == BUDGET
        else:
            assert body["prompt"] == ids[n] + GENERATED + family.encode_close() and body["max_tokens"] == 128
        assert body["seed"] == request_seed(3, ids[n], phase, 0)
        assert {k: body[k] for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty")} == {
            "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 0.0}
        assert body["lora"] == [{"id": 0, "scale": 0.0}] and body["stop"] == ["<|im_end|>"]
        assert body["cache_prompt"] is False
        assert body.get("return_tokens", False) is (phase == 1)  # the first asks for every generated id
    # the kept samples: the public sample format, the canonical answers
    kept = rows_of(out)
    numbers = kept_numbers(out)
    assert sorted(numbers) == [1, 2, 3, 11, 12, 13]
    for sample, n in zip(kept, numbers, strict=True):
        assert sample["tools"] == [] and set(sample) == {"messages", "tools", "source", "source_id", "kind",
                                                         "instruction_lang"}
        assert sample["messages"][-1] == {"role": "assistant", "content": MISSING if n < 10 else IRRELEVANT}
    # the sidecar: one row per kept sample with a usable reasoning, keyed as train.py names the sample
    rows = rows_of(sidecar)
    settings = GenerationSettings(mode=Mode.THINK, decoding=SAMPLING, reasoning_budget=BUDGET,
                                  answer_tokens=128, base_seed=3)
    expected = []
    for index, (sample, n) in enumerate(zip(kept, numbers, strict=True)):
        if n in REASONING:
            expected.append({"id": f"self-replay:{index}", "k": 0,
                             "answer_sha256": answer_sha256(sample["messages"][-1]), "source": "teacher",
                             "reasoning": REASONING[n], "model": "Fake-4B-Q8_0.gguf",
                             "cache_key": request_key({"model": "Fake-4B-Q8_0.gguf", "prompt_ids": ids[n],
                                                       "generation": settings.record()}),
                             "context_equal": True})
    assert sorted(rows, key=lambda r: r["id"]) == sorted(expected, key=lambda r: r["id"])
    assert all(list(row) == sorted(ROW_KEYS) for row in rows)  # canonical JSON lines
    assert len(ReasoningIndex([sidecar])) == 4
    # the manifest: the family, the mode, the decoding, the seed, R, forced closes, reasoning lengths, the
    # sidecar's SHA-256
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["family"] == family.record()
    assert manifest["generation"] == {"path": "family", "lora_scale": 0.0, **settings.record(),
                                      "stop": ["<|im_end|>"], "cache_prompt": False}
    assert manifest["server"] == {"total_slots": 1, "model": "Fake-4B-Q8_0.gguf", "build": "fake-b11349",
                                  "n_ctx": 8192}
    assert manifest["seed"] == 3 and manifest["generated"] == 8
    assert manifest["kept"] == {"total": 6, "incomplete": 3, "irrelevant": 3,
                                "languages": manifest["kept"]["languages"], "normalised": 1,
                                "modes": {"think": 4, "no_think": 2}}
    assert manifest["rejected"] == {"incomplete: form": 1, "irrelevant: form": 1}
    thinking = manifest["thinking"]
    assert thinking["forced"] == {"generated": 1, "kept": 1}
    assert thinking["without_reasoning"] == {"empty": 1, "think tag": 1, "refused": 0}
    lengths = thinking["reasoning_tokens"]  # Generation.reasoning_tokens: 3 per prompt, 2 for 12, 4 for 13
    assert (lengths["generated"]["n"], lengths["generated"]["total"]) == (8, 24)
    assert (lengths["kept"]["n"], lengths["kept"]["total"]) == (6, 18)
    data = sidecar.read_bytes()
    assert manifest["reasoning"] == {"path": "self-replay-reasoning.jsonl", "rows": 4,
                                     "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    assert manifest["output"]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()


def test_the_filter_reads_the_answer_span_only(tmp_path, monkeypatch):
    # a fixed sentence in the reasoning keeps nothing (4, 14); junk in the reasoning rejects nothing (2's
    # reasoning is no sentence, its answer is): the answer span alone decides
    model = thinking_folder(monkeypatch, tmp_path)
    plans = {**THINK_PLANS, 2: [closed("Sure! Please provide the length. ???", SINGULAR)],
             4: [closed(MISSING, "")], 14: [closed(IRRELEVANT, IRRELEVANT + " Sorry.")]}
    server = ReplayServer(plans, props=SERVED)
    code, out, manifest_path = run(tmp_path, monkeypatch, server, model, "--reasoning-out",
                                   str(tmp_path / "r" / "x.jsonl"))
    assert code == 0
    assert set(kept_numbers(out)) == {1, 2, 3, 11, 12, 13}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["rejected"] == {"incomplete: form": 1, "irrelevant: form": 1}


def test_a_rerun_with_the_same_seed_is_identical_and_every_seed_derives_from_it(tmp_path, monkeypatch):
    model = thinking_folder(monkeypatch, tmp_path)
    outputs = []
    for name in ("first", "second"):
        folder = tmp_path / name
        folder.mkdir()
        server = ReplayServer(THINK_PLANS, props=SERVED)
        monkeypatch.setattr(rb, "LocalClient", lambda url, server=server: server.client())
        monkeypatch.setattr(rb, "load_candidates", lambda manifest, raw_dir: small_pools())
        monkeypatch.setattr(rb, "make_renderer", refuse)
        out, manifest, sidecar = folder / "replay.jsonl", folder / "m.json", folder / "side" / "r.jsonl"
        assert rb.main(["--out", str(out), "--manifest", str(manifest), "--model", str(model), "--limit", "8",
                        "--seed", "5", "--exclude", str(tmp_path / "none.jsonl"),
                        "--reasoning-out", str(sidecar)]) == 0
        outputs.append(([out.read_bytes(), sidecar.read_bytes(), manifest.read_bytes()],
                        [body["seed"] for body in server.requests]))
    assert outputs[0] == outputs[1]
    assert len(set(outputs[0][1])) == len(outputs[0][1])  # one seed per prompt and phase


@pytest.mark.parametrize("tokens", [("I must", " check", b"\xe6\x80"),
                                    ("I must", " check", "\n", b"\xe6\x80")])
def test_a_reasoning_the_budget_cut_inside_a_character_is_written_without_that_character(tmp_path,
                                                                                         monkeypatch, tokens):
    # the budget cut prompt 3's reasoning after two bytes of a three-byte character, which the server writes
    # as U+FFFD; the generation goes on from every generated id, and the row leaves out the unfinished
    # character and, in the second case, the newline before it (Generation.reasoning holds none at either end)
    model = thinking_folder(monkeypatch, tmp_path)
    plans = {**THINK_PLANS, 3: [Scripted(tokens, "length"), Scripted((MISSING,))]}
    server = ReplayServer(plans, props=SERVED)
    sidecar = tmp_path / "side" / "r.jsonl"
    code, out, manifest_path = run(tmp_path, monkeypatch, server, model, "--reasoning-out", str(sidecar))
    assert code == 0
    ids = prompt_ids(model, Mode.THINK)
    second = [body for n, body in zip(server.users, server.requests, strict=True) if n == 3][1]
    generated = [ID_BASE + index for index in range(len(tokens))]
    assert second["prompt"] == ids[3] + generated + load_family(model).encode_close()
    index = kept_numbers(out).index(3)
    rows = {row["id"]: row for row in rows_of(sidecar)}
    assert rows[f"self-replay:{index}"]["reasoning"] == "I must check"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["thinking"]["forced"] == {"generated": 1, "kept": 1}


def test_the_reasoning_budget_option_replaces_the_familys(tmp_path, monkeypatch):
    model = thinking_folder(monkeypatch, tmp_path)
    server = ReplayServer(THINK_PLANS, props=SERVED)
    code, _, manifest_path = run(tmp_path, monkeypatch, server, model, "--reasoning-out",
                                 str(tmp_path / "s" / "r.jsonl"), "--reasoning-budget", "48")
    assert code == 0
    firsts = {}
    for n, body in zip(server.users, server.requests, strict=True):
        firsts.setdefault(n, body)
    assert {body["max_tokens"] for body in firsts.values()} == {48}
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["generation"]["reasoning_budget"] == 48


# ---- the caps: --cap kept answers in all, --cap-per-kind kept answers per kind (FT3's 500) ----------------


SEED_3_DRAW = [3, 11, 2, 12, 4, 14, 1, 13]  # small_pools' draw under --seed 3, by user number


def drawn(tmp_path: Path) -> list[int]:
    """The user number of every prompt of the replay's prompts file, in the drawn order."""
    return [user_number(p["messages"][1]["content"]) for p in rows_of(tmp_path / "self-replay.prompts.jsonl")]


def test_the_family_path_stops_at_cap_kept_answers_not_prompts(tmp_path, monkeypatch):
    # in the draw's order, 4 and 14 are rejected (THINK_PLANS), so the fifth kept answer is 1's, the seventh
    # prompt generated: the run stops there, and 13, the last one drawn, is never sent
    model = thinking_folder(monkeypatch, tmp_path)
    server = ReplayServer(THINK_PLANS, props=SERVED)
    sidecar = tmp_path / "side" / "r.jsonl"
    code, out, manifest_path = run(tmp_path, monkeypatch, server, model, "--reasoning-out", str(sidecar),
                                   "--cap", "5")
    assert code == 0 and drawn(tmp_path) == SEED_3_DRAW
    assert server.users == [3, 3, 11, 2, 12, 4, 14, 1]  # 3's forced close takes a second request
    numbers = kept_numbers(out)
    assert numbers == [3, 11, 2, 12, 1]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (manifest["cap"], manifest["generated"], manifest["kept"]["total"]) == (5, 7, 5)
    assert manifest["rejected"] == {"incomplete: form": 1, "irrelevant: form": 1}
    assert manifest["thinking"]["reasoning_tokens"]["generated"]["n"] == 7
    # the sidecar holds rows of those five alone (12's reasoning is empty: no row), by their index
    rows = {row["id"]: row["reasoning"] for row in rows_of(sidecar)}
    assert rows == {f"self-replay:{index}": REASONING[n] for index, n in enumerate(numbers) if n in REASONING}
    assert manifest["reasoning"]["rows"] == len(rows) == 4


def test_the_family_path_fills_each_kind_to_cap_per_kind_and_stops_when_both_are_full(tmp_path, monkeypatch):
    # 11 is rejected here and 12 thinks: 3 fills the incomplete kind, so 2 is skipped, never sent, while the
    # irrelevant kind waits for 12; with both kinds full the run stops, and 4, 14, 1 and 13 are never sent
    model = thinking_folder(monkeypatch, tmp_path)
    plans = {**THINK_PLANS, 11: [closed("Ordering pizza is out of scope.", "I cannot order pizza.")],
             12: [closed("No listed function orders food.", IRRELEVANT)]}
    server = ReplayServer(plans, props=SERVED)
    sidecar = tmp_path / "side" / "r.jsonl"
    code, out, manifest_path = run(tmp_path, monkeypatch, server, model, "--reasoning-out", str(sidecar),
                                   "--cap-per-kind", "1")
    assert code == 0 and drawn(tmp_path) == SEED_3_DRAW
    assert server.users == [3, 3, 11, 12]
    assert kept_numbers(out) == [3, 12]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (manifest["cap_per_kind"], manifest["generated"]) == (1, 3)
    assert {key: manifest["kept"][key] for key in ("total", "incomplete", "irrelevant")} == {
        "total": 2, "incomplete": 1, "irrelevant": 1}
    assert manifest["rejected"] == {"irrelevant: form": 1}
    assert {row["id"]: row["reasoning"] for row in rows_of(sidecar)} == {
        "self-replay:0": REASONING[3], "self-replay:1": "No listed function orders food."}


# ---- the sidecar trains the replay's samples as THINK samples (scripts/train.py --dry-run) ----------------


def test_trains_dry_run_reads_the_replay_with_its_sidecar_as_think_samples(tmp_path, monkeypatch, capsys):
    model = thinking_folder(monkeypatch, tmp_path)
    sidecar = tmp_path / "reasoning" / "self-replay-reasoning.jsonl"
    code, out, _ = run(tmp_path, monkeypatch, ReplayServer(THINK_PLANS, props=SERVED), model,
                       "--reasoning-out", str(sidecar))
    assert code == 0
    summary = train_dry_run(tmp_path, model, out, sidecar, capsys)
    assert summary["modes"] == {"think": 4, "no_think": 2, "think_fallback": 0}
    assert summary["refused"] == {"template": 0, "boundary": 0, "bos": 0}
    assert summary["reasoning"]["rows"] == 4 and summary["reasoning"]["tokens"]["n"] == 4
    # the rows name the samples by the replay file's stem: a copy under another name trains NO_THINK
    copy = tmp_path / "renamed.jsonl"
    copy.write_bytes(out.read_bytes())
    assert train_dry_run(tmp_path, model, copy, sidecar, capsys)["modes"] == {"think": 0, "no_think": 6,
                                                                              "think_fallback": 0}


def install_tokenizer(monkeypatch, folder: Path, tokenizer: FakeFamilyTokenizer, **changes) -> None:
    """install_fake_family's model folder for a tokenizer made here: the folder holds its template, the
    transformers stand-in returns it and the registry holds the family that describes it, with changes."""
    model_folder(folder, template=tokenizer.chat_template)
    fake_transformers(monkeypatch, tokenizer)
    monkeypatch.setattr(registry, "FAMILIES", (fake_family(tokenizer, **changes),))


def install_with_pieces(monkeypatch, folder: Path, pieces: tuple[str, ...], **changes) -> FakeFamilyTokenizer:
    """install_fake_family's model folder, of an "only" family whose tokenizer reads each of pieces as one
    token."""
    tokenizer = FakeFamilyTokenizer("only", pieces=pieces)
    install_tokenizer(monkeypatch, folder, tokenizer, **changes)
    return tokenizer


def test_a_reasoning_whose_think_sample_the_family_refuses_gets_no_row(tmp_path, monkeypatch, capsys):
    # the tokenizer reads "\nI" as one token, so prompt 3's reasoning ("I must check the schema") merges with
    # the newline that ends the THINK prompt: its THINK sample does not split at a token boundary, which the
    # training path refuses; the replay writes no row for it (the sample trains NO_THINK), and train.py's dry
    # run refuses nothing
    model = tmp_path / "model"
    tokenizer = install_with_pieces(monkeypatch, model, ("\nI",), inference_mode=Mode.THINK,
                                    reasoning_budget=BUDGET,
                                    decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: SAMPLING})
    sidecar = tmp_path / "side" / "r.jsonl"
    server = ReplayServer(THINK_PLANS, props=SERVED, decode=tokenizer.decode)
    code, out, manifest_path = run(tmp_path, monkeypatch, server, model, "--reasoning-out", str(sidecar))
    assert code == 0
    numbers = kept_numbers(out)
    assert sorted(numbers) == [1, 2, 3, 11, 12, 13]
    *context, target = rows_of(out)[numbers.index(3)]["messages"]
    with pytest.raises(TokenBoundaryError):  # what a row would have made of prompt 3's sample
        render_family_sample(load_family(model), context, target, tools=[], reasoning=REASONING[3])
    rows = {row["id"]: row["reasoning"] for row in rows_of(sidecar)}
    assert rows == {f"self-replay:{numbers.index(n)}": REASONING[n] for n in (1, 2, 11)}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["thinking"]["without_reasoning"] == {"empty": 1, "think tag": 1, "refused": 1}
    assert manifest["kept"]["modes"] == {"think": 3, "no_think": 3}
    summary = train_dry_run(tmp_path, model, out, sidecar, capsys)
    assert summary["modes"] == {"think": 3, "no_think": 3, "think_fallback": 0}
    assert summary["refused"] == {"template": 0, "boundary": 0, "bos": 0}


class EmptyToolsBlock(FakeFamilyTokenizer):
    """The fake template of a family whose template writes its tools block for an empty list of tools too, as
    a template that tests `tools is not none` does, and none for no tools (None): a prompt rendered with a
    sample's own tools, [], is not the prompt rendered with None. Every prompt here has a system message,
    which the block joins."""

    def __init__(self, style: str) -> None:
        super().__init__(style, chat_template=f"fake template: style={style}, a tools block for []")

    def apply_chat_template(self, messages: list[dict], *, tools: list[dict] | None = None, **kwargs) -> str:
        out = super().apply_chat_template(messages, tools=tools, **kwargs)
        if tools is None or tools:
            return out
        head = f"<|im_start|>system\n{messages[0]['content']}"
        assert messages[0]["role"] == "system" and out.startswith(head + TURN_END)
        return head + "\n\n<tools>[]</tools>" + out[len(head):]


@pytest.mark.parametrize("style,mode", [("only", Mode.THINK), ("none", Mode.NO_THINK)])
def test_every_request_prompts_with_the_prompt_its_kept_sample_trains_under(tmp_path, monkeypatch, style,
                                                                            mode):
    # held against the training path itself (trained_prompt: render_family_sample with the kept sample's own
    # tools and its sidecar row's reasoning), on a template that tells the sample's tools, [], from none: a
    # replay prompt rendered without the sample's tools would not be the prompt the sample trains under
    model = tmp_path / "model"
    changes = ({"inference_mode": Mode.THINK, "reasoning_budget": BUDGET,
                "decoding": {Mode.THINK: SAMPLING, Mode.NO_THINK: SAMPLING}} if mode is Mode.THINK else {})
    install_tokenizer(monkeypatch, model, EmptyToolsBlock(style), **changes)
    server = ReplayServer(THINK_PLANS if mode is Mode.THINK else NO_THINK_PLANS, props=SERVED)
    sidecar = tmp_path / "side" / "r.jsonl"
    extra = ("--reasoning-out", str(sidecar)) if mode is Mode.THINK else ()
    code, out, _ = run(tmp_path, monkeypatch, server, model, *extra)
    assert code == 0
    family = load_family(model)
    kept = rows_of(out)
    *context, _ = kept[0]["messages"]
    hf = family.to_template_messages(context)
    assert family.prompt(hf, [], mode) != family.prompt(hf, None, mode)  # the template tells them apart
    firsts: dict[int, list[int]] = {}
    for n, body in zip(server.users, server.requests, strict=True):
        firsts.setdefault(n, body["prompt"])
    rows = {row["id"]: row["reasoning"] for row in rows_of(sidecar)} if mode is Mode.THINK else {}
    trained = 0
    for index, (sample, n) in enumerate(zip(kept, kept_numbers(out), strict=True)):
        reasoning = rows.get(f"self-replay:{index}")
        if mode is Mode.THINK and reasoning is None:
            continue  # no row: the sample trains NO_THINK, under the empty block its THINK request lacks
        assert sample["tools"] == [] and firsts[n] == trained_prompt(family, sample, reasoning)
        trained += 1
    assert trained == (4 if mode is Mode.THINK else 6)


# ---- a family that never thinks, or one asked not to: NO_THINK through the family, as today ---------------


def test_a_family_that_never_thinks_replays_no_think_greedily_through_the_family(tmp_path, monkeypatch):
    model = tmp_path / "model"
    install_fake_family(monkeypatch, model, "none")
    server = ReplayServer(NO_THINK_PLANS, props=SERVED)
    code, out, manifest_path = run(tmp_path, monkeypatch, server, model)
    assert code == 0
    ids = prompt_ids(model, Mode.NO_THINK)
    assert sorted(server.users) == [1, 2, 3, 4, 11, 12, 13, 14]  # one request per prompt
    for n, sent in zip(server.users, server.request_bytes, strict=True):
        expected = {"prompt": ids[n], "max_tokens": 128, "temperature": 0.0, "cache_prompt": False,
                    "logprobs": 5, "stop": ["<|im_end|>"]}
        assert sent == json.dumps(expected, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    kept = rows_of(out)
    assert len(kept) == 6 and all(k["tools"] == [] for k in kept)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["generation"]["mode"] == "no_think" and manifest["generation"]["path"] == "family"
    assert manifest["generation"]["decoding"] == GenerationSettings(mode=Mode.NO_THINK,
                                                                    decoding=GREEDY).record()["decoding"]
    assert manifest["family"]["name"] == "fake-none" and manifest["family"]["thinking"] == "none"
    assert manifest["thinking"] is None and manifest["reasoning"] is None
    assert manifest["kept"]["modes"] == {"think": 0, "no_think": 6} and manifest["kept"]["normalised"] == 1
    assert manifest["rejected"] == {"incomplete: form": 1, "irrelevant: form": 1}


def test_a_family_that_thinks_replays_no_think_on_request_without_a_sidecar(tmp_path, monkeypatch):
    model = tmp_path / "model"
    install_fake_family(monkeypatch, model, "hybrid", inference_mode=Mode.THINK,
                        decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: GREEDY})
    server = ReplayServer(NO_THINK_PLANS, props=SERVED)
    code, out, manifest_path = run(tmp_path, monkeypatch, server, model, "--mode", "no_think")
    assert code == 0
    ids = prompt_ids(model, Mode.NO_THINK)
    for n, body in zip(server.users, server.requests, strict=True):  # the empty block in the prompt, greedy
        assert body["prompt"] == ids[n] and body["temperature"] == 0.0 and body["max_tokens"] == 128
        assert "seed" not in body and "return_tokens" not in body
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert (manifest["family"]["name"], manifest["generation"]["mode"]) == ("fake-hybrid", "no_think")
    assert manifest["thinking"] is None and manifest["reasoning"] is None and len(rows_of(out)) == 6


# ---- the Qwen3 hybrid family in no_think mode: M6's text path stays the default --------------------------


def hybrid_folder(monkeypatch, tmp_path: Path) -> tuple[Path, FakeFamilyTokenizer]:
    """A model folder of a fake family named as M6's, qwen3-hybrid (the hybrid template, greedy)."""
    folder = tmp_path / "model"
    tokenizer = install_fake_family(monkeypatch, folder, "hybrid", name=QWEN3_HYBRID.name)
    return folder, tokenizer


def text_prompt(tokenizer: FakeFamilyTokenizer, prompt: dict) -> str:
    """The prompt M6's text path renders (prompt_renderer: thinking off, no tools block)."""
    return tokenizer.apply_chat_template(api_to_hf(prompt["messages"]), tools=None, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def test_the_qwen3_hybrid_family_keeps_m6s_text_path_and_requests_by_default(tmp_path, monkeypatch):
    model, tokenizer = hybrid_folder(monkeypatch, tmp_path)
    monkeypatch.setattr(rb, "load_family", refuse)  # the text path loads no family
    server = ReplayServer(NO_THINK_PLANS, props=SERVED)
    code, out, manifest_path = run(tmp_path, monkeypatch, server, model, renderer=rb.make_renderer)
    assert code == 0
    prompts = rows_of(model.parent / "self-replay.prompts.jsonl")
    texts = {user_number(p["messages"][1]["content"]): text_prompt(tokenizer, p) for p in prompts}
    for n, sent in zip(server.users, server.request_bytes, strict=True):  # the M6 request, byte for byte
        expected = {"prompt": texts[n], "max_tokens": 128, "temperature": 0.0, "cache_prompt": False,
                    "stop": ["<|im_end|>"]}
        assert sent == json.dumps(expected, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert list(manifest) == ["script", "seed", "limit", "cap", "cap_per_kind", "templates", "server",
                              "generation", "excluded", "prompts", "generated", "kept", "rejected", "output"]
    assert manifest["server"] == {"total_slots": 1, "model": "Fake-4B-Q8_0.gguf", "build": "fake-b11349"}
    assert manifest["generation"] == {"lora_scale": 0.0, "temperature": 0.0, "max_tokens": 128,
                                      "stop": ["<|im_end|>"],
                                      "thinking": "off (chat template with enable_thinking=False)",
                                      "cache_prompt": False}
    assert list(manifest["kept"]) == ["total", "incomplete", "irrelevant", "languages", "normalised"]
    assert len(rows_of(out)) == 6


def test_the_qwen3_hybrid_family_replays_through_the_family_on_request_or_in_think_mode(tmp_path,
                                                                                        monkeypatch):
    model, tokenizer = hybrid_folder(monkeypatch, tmp_path)
    server = ReplayServer(NO_THINK_PLANS, props=SERVED)
    code, _, manifest_path = run(tmp_path, monkeypatch, server, model, "--family-path")
    assert code == 0
    family = load_family(model)
    prompts = {user_number(p["messages"][1]["content"]): p
               for p in rows_of(model.parent / "self-replay.prompts.jsonl")}
    for body in server.requests:  # the same prompt as the text path renders, as its ids
        text = text_prompt(tokenizer, prompts[number(body["prompt"])])
        assert body["prompt"] == family.encode_prompt(text) and body["logprobs"] == 5
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["family"]["name"] == "qwen3-hybrid"
    # --mode think: the family path, THINK
    think = ReplayServer({n: [closed("Because.", MISSING if n < 10 else IRRELEVANT)] for n in NO_THINK_PLANS},
                         props=SERVED)
    side = tmp_path / "side" / "r.jsonl"
    code, _, manifest_path = run(tmp_path, monkeypatch, think, model, "--mode", "think", "--reasoning-out",
                                 str(side), "--force")
    assert code == 0 and all(FakeFamilyTokenizer().decode(body["prompt"]).endswith(
        "<|im_start|>assistant\n<think>\n") for body in think.requests)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["generation"]["mode"] == "think" and manifest["reasoning"]["rows"] == 8


# ---- usage: the mode, the sidecar options, the outputs, the server's context ------------------------------


@pytest.mark.parametrize("style,extra,message", [
    ("only", (), "--reasoning-out"),  # a THINK replay writes its sidecar
    ("none", ("--reasoning-out", "SIDE"), "--reasoning-out"),  # a NO_THINK replay writes none
    ("none", ("--reasoning-budget", "32"), "--reasoning-budget"),
    ("none", ("--mode", "think"), "no think mode"),
    ("only", ("--family", "qwen3-hybrid", "--reasoning-out", "SIDE"), "qwen3-hybrid"),
    ("only", ("--reasoning-out", "OUT"), "same file"),  # the sidecar would overwrite the samples
    ("only", ("--reasoning-out", "MANIFEST"), "same file"),
    ("only", ("--reasoning-out", "SIDE", "--reasoning-budget", "0"), "reasoning_budget"),
])
def test_usage_errors_stop_before_anything_is_written_or_sent(tmp_path, monkeypatch, capsys, style, extra,
                                                               message):
    model = tmp_path / "model"
    install_fake_family(monkeypatch, model, style, inference_mode=Mode.NO_THINK if style == "none"
                        else Mode.THINK)
    server = ReplayServer(THINK_PLANS, props=SERVED)
    places = {"SIDE": str(tmp_path / "side" / "r.jsonl"), "OUT": str(tmp_path / "self-replay.jsonl"),
              "MANIFEST": str(tmp_path / "self-replay-manifest.json")}
    code, out, manifest = run(tmp_path, monkeypatch, server, model, *(places.get(a, a) for a in extra))
    assert code == 2 and message in capsys.readouterr().err
    assert server.requests == [] and not out.exists() and not manifest.exists()
    assert not (tmp_path / "self-replay.prompts.jsonl").exists() and not (tmp_path / "side").exists()


def test_a_folder_of_no_registered_family_is_refused(tmp_path, monkeypatch, capsys):
    model = tmp_path / "model"
    install_fake_family(monkeypatch, model, "none")
    (model / "tokenizer_config.json").write_text(json.dumps({"chat_template": "another template"}),
                                                 encoding="utf-8")
    server = ReplayServer(NO_THINK_PLANS, props=SERVED)
    code, _, _ = run(tmp_path, monkeypatch, server, model)
    assert code == 2 and "unknown chat template" in capsys.readouterr().err and server.requests == []


def test_an_existing_sidecar_is_overwritten_only_with_force(tmp_path, monkeypatch, capsys):
    model = thinking_folder(monkeypatch, tmp_path)
    sidecar = tmp_path / "side" / "r.jsonl"
    sidecar.parent.mkdir()
    sidecar.write_text("old\n", encoding="utf-8")
    code, out, _ = run(tmp_path, monkeypatch, ReplayServer(THINK_PLANS, props=SERVED), model,
                       "--reasoning-out", str(sidecar))
    assert code == 2 and str(sidecar) in capsys.readouterr().err and not out.exists()
    code, _, _ = run(tmp_path, monkeypatch, ReplayServer(THINK_PLANS, props=SERVED), model,
                     "--reasoning-out", str(sidecar), "--force")
    assert code == 0 and len(rows_of(sidecar)) == 4


def thinking_prompt_lengths(tmp_path: Path, monkeypatch, model: Path) -> list[list[int]]:
    """The THINK prompts' ids of a dry run over small_pools (prompt_ids)."""
    monkeypatch.setattr(rb, "load_candidates", lambda manifest, raw_dir: small_pools())
    out = tmp_path / "dry" / "self-replay.jsonl"
    assert rb.main(["--out", str(out), "--manifest", str(tmp_path / "dry" / "m.json"), "--limit", "8",
                    "--seed", "3", "--exclude", str(tmp_path / "none.jsonl"), "--dry-run"]) == 0
    return list(prompt_ids(model, Mode.THINK, rb.prompts_path(out)).values())


def test_a_server_context_too_small_for_the_longest_generation_is_refused(tmp_path, monkeypatch, capsys):
    model = thinking_folder(monkeypatch, tmp_path)
    longest = max(len(ids) for ids in thinking_prompt_lengths(tmp_path, monkeypatch, model))
    reach = longest + BUDGET + len(load_family(model).encode_close()) + 128  # forced close, full answer
    for n_ctx, code_expected in ((reach - 1, 2), (reach, 0)):
        props = {**SERVED, "default_generation_settings": {"n_ctx": n_ctx}}
        server = ReplayServer(THINK_PLANS, props=props)
        code, _, _ = run(tmp_path, monkeypatch, server, model, "--reasoning-out",
                         str(tmp_path / "s" / "r.jsonl"), "--force")
        assert code == code_expected
        if code == 2:
            assert "n_ctx" in capsys.readouterr().err and server.requests == []


def test_a_server_that_reports_no_context_is_refused_on_the_family_path(tmp_path, monkeypatch, capsys):
    model = tmp_path / "model"
    install_fake_family(monkeypatch, model, "none")
    server = ReplayServer(NO_THINK_PLANS, props={"total_slots": 1, "build_info": "fake-b11349"})
    code, out, _ = run(tmp_path, monkeypatch, server, model)
    assert code == 2 and "n_ctx" in capsys.readouterr().err and server.requests == [] and not out.exists()


def test_a_dry_run_reads_no_model_folder(tmp_path, monkeypatch):
    # the M6 contract: a dry run writes the prompts and nothing else, needs no server and no model folder
    monkeypatch.setattr(rb, "load_candidates", lambda manifest, raw_dir: small_pools())
    monkeypatch.setattr(rb, "folder_family", refuse)
    monkeypatch.setattr(rb, "load_family", refuse)
    monkeypatch.setattr(rb, "LocalClient", refuse)
    out = tmp_path / "self-replay.jsonl"
    assert rb.main(["--out", str(out), "--manifest", str(tmp_path / "m.json"), "--model",
                    str(tmp_path / "no-such-folder"), "--limit", "8", "--dry-run"]) == 0
    assert len(rows_of(rb.prompts_path(out))) == 8 and not out.exists()


# ---- this machine's own files: the M6 dry run byte for byte, the real folders' plans ----------------------


M6_PROMPTS = REPO / "data" / "raw" / "public" / "base-replay.prompts.jsonl"
# the prompts file's SHA-256 as data/public/base-replay-manifest.json records it
M6_SHA256 = "a7e5c2a5baa18346f34d0c1597c23cb27dd41b2487d7d5960b29ddb16c0ab053"


def pools_cached() -> bool:
    """The Hugging Face cache holds the glaive-v2 and ToolACE files the pools are read from."""
    manifest = REPO / "data" / "public" / "manifest.json"
    if not manifest.is_file():
        return False
    sources = json.loads(manifest.read_text(encoding="utf-8"))["sources"]
    try:
        for name in ("glaive_v2", "toolace"):
            for item in sources[name]["per_file"]:
                public_pool.cached_file(sources[name]["repo"], item["file"], sources[name]["revision"])
    except (FileNotFoundError, KeyError):
        return False
    return True


@pytest.mark.skipif(not (M6_PROMPTS.is_file() and pools_cached()),
                    reason="the M6 prompts file or the Hugging Face cache is not here")
def test_a_dry_run_with_m6s_options_writes_m6s_prompts_file_byte_for_byte(tmp_path):
    out = tmp_path / "base-replay.jsonl"
    assert rb.main(["--out", str(out), "--manifest", str(tmp_path / "m.json"), "--limit", "3000",
                    "--seed", "42", "--model", str(REPO / "models" / "Qwen3-1.7B"),
                    "--exclude", str(REPO / "data" / "raw" / "public" / "format-mix.jsonl"),
                    "--source-manifest", str(REPO / "data" / "public" / "manifest.json"),
                    "--raw-dir", str(REPO / "data" / "raw" / "public"), "--dry-run"]) == 0
    data = rb.prompts_path(out).read_bytes()
    assert hashlib.sha256(data).hexdigest() == M6_SHA256 and data == M6_PROMPTS.read_bytes()


def real_folder(name: str) -> Path:
    return REPO / "models" / name


def present(name: str) -> bool:
    return all((real_folder(name) / f).is_file() for f in ("config.json", "tokenizer_config.json"))


@pytest.mark.skipif(not present("Qwen3-1.7B"), reason="models/Qwen3-1.7B is not here")
def test_the_qwen3_1_7b_folder_keeps_the_text_path_by_default():
    base = ["--out", "x.jsonl", "--manifest", "m.json", "--model", str(real_folder("Qwen3-1.7B"))]
    plan = rb.replay_plan(rb.build_parser().parse_args(base))
    assert (plan.family, plan.mode, plan.text, plan.settings) == (QWEN3_HYBRID, Mode.NO_THINK, True, None)
    plan = rb.replay_plan(rb.build_parser().parse_args([*base, "--family-path"]))
    assert (plan.mode, plan.text, plan.settings.decoding) == (Mode.NO_THINK, False, GREEDY)


@pytest.mark.skipif(not present("Qwen3-4B-Thinking-2507"), reason="models/Qwen3-4B-Thinking-2507 is not here")
def test_a_thinking_winners_folder_replays_think_through_the_family():
    base = ["--out", "x.jsonl", "--manifest", "m.json", "--model", str(real_folder("Qwen3-4B-Thinking-2507"))]
    with pytest.raises(ValueError, match="--reasoning-out"):
        rb.replay_plan(rb.build_parser().parse_args(base))
    plan = rb.replay_plan(rb.build_parser().parse_args([*base, "--reasoning-out", "r.jsonl"]))
    assert (plan.family.name, plan.mode, plan.text) == ("qwen3-thinking-2507", Mode.THINK, False)
    assert (plan.settings.decoding, plan.settings.reasoning_budget, plan.settings.base_seed) == (
        registry.QWEN3_THINKING_2507.decoding[Mode.THINK], 1024, 42)
    with pytest.raises(FamilyError):
        rb.replay_plan(rb.build_parser().parse_args([*base, "--family", "qwen3-hybrid"]))

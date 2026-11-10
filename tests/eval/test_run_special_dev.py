"""scripts/run_special_dev.py (M8 WP2), offline: a fake llama-server (tests/support/llama_fakes.py) answers
with scripted completions over a fake family loaded from a fake model folder, so the requests can be held
against the run's settings (the family path, think mode, greedy, the reasoning budget, one slot, no tools
field); nothing is started and no request leaves the process. The held-out roots are moved to an empty folder
for every test, so no held-out set of the repository is ever looked at."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest
from support.family_fakes import THINK_OPEN
from support.llama_fakes import PROPS, FakeLlama, Scripted, install_fake_family

from research_agent.data import special_mix as sm
from research_agent.eval.lock import write_lock
from research_agent.family.core import Decoding, Mode

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rsd = load_script("run_special_dev")
SERVER = "http://fake-llama.invalid"
SAMPLING = Decoding(temperature=0.6, top_p=0.95)  # the family's thinking decoding, which the run overrides
BOOK_ROOM = {"name": "book_room", "description": "Book a meeting room.", "parameters": {
    "type": "object", "properties": {"room": {"type": "string"}, "people": {"type": "integer", "minimum": 1},
                                     "date": {"type": "string"}}, "required": ["room", "people", "date"]}}
PLAN_TRIP = {"name": "plan_trip", "description": "Plan a trip.", "parameters": {
    "type": "object", "properties": {"traveller": {"type": "object"}, "destination": {"type": "string"}},
    "required": ["traveller", "destination"]}}
SYSTEM = "\n".join([
    "You may call the functions below; each line is one function as JSON.",
    json.dumps(BOOK_ROOM), json.dumps(PLAN_TRIP), "", "How to reply:",
    "- When the request can be served, reply only with the calls, one per line, as Python calls.",
    ("- When a value given breaks its definition, reply only with: Not sent: the value <value> of <param> "
     "breaks the definition."),
    "- When a required parameter has no value, reply only with: Not sent: <api> still needs <params>",
    "- When no listed function fits, reply only with: Not sent: no listed function can do this.",
])
FIXED = {"incorrect": ["Not sent: the value", "of", "breaks the definition."],
         "missing": ["Not sent:", "still needs"]}
CALL = 'book_room(room="会议室A", people=4, date="2026-11-05")'


def item(number: int, kind: str, user: str, expected: dict) -> dict:
    return {"id": f"sdev-b-{number:04d}", "split": "b", "lang": "en", "kind": kind,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "call_format": "python", "family": "fx-run", "expected": expected,
            "source": {"dataset": "fixture", "record": str(number)}}


ITEMS = [
    item(1, "incorrect", "Book room A-12 for -3 people on 2026-11-05.",
         {"type": "refusal", "case": "incorrect", "fixed": FIXED["incorrect"], "values": ["-3"],
          "params": ["people"], "api": ""}),
    item(2, "missing", "Plan a trip to Lisbon.",
         {"type": "refusal", "case": "missing", "fixed": FIXED["missing"], "values": [],
          "params": ["traveller"], "api": "plan_trip"}),
    item(3, "valid_verbatim", "Book room 会议室A for 4 people on 2026-11-05.",
         {"type": "call", "calls": [{"name": "book_room", "arguments": {"room": "会议室A", "people": 4,
                                                                         "date": "2026-11-05"}}],
          "verbatim": ["会议室A"]}),
]
INCORRECT_ANSWER = "Not sent: the value <-3> of <people> breaks the definition."
MISSING_ANSWER = "Not sent: plan_trip still needs traveller"


def script() -> list[Scripted]:
    """The completions of the three items: a reasoning closed by the model, one closed by force at the budget
    (two requests), and a call."""
    return [Scripted(("The value -3 is below 1.", "\n</think>\n\n", INCORRECT_ANSWER)),
            Scripted(("No traveller is given, so",), finish_reason="length"),
            Scripted((MISSING_ANSWER,)),
            Scripted(("All values are there.", "\n</think>\n\n", CALL))]


@pytest.fixture(autouse=True)
def no_held_out_sets(tmp_path, monkeypatch):
    """The held-out roots of every test: a folder of its own, empty unless the test fills it."""
    monkeypatch.setattr(rsd, "HELD_OUT_ROOTS", (tmp_path / "held-root",))


@pytest.fixture
def model(tmp_path, monkeypatch):
    """A model folder of a family that always thinks and samples its thinking, as Nanbeige's does."""
    folder = tmp_path / "model"
    tokenizer = install_fake_family(monkeypatch, folder, "only", inference_mode=Mode.THINK,
                                    decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: SAMPLING})
    return folder, tokenizer


def write_items(path: Path, items: list[dict], *, lock: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    if lock:
        write_lock(path)
    return path


def run(tmp_path, model, fake: FakeLlama, *args: str, items: Path | None = None, system: str = "b0") -> int:
    items = items or write_items(tmp_path / "special-dev-b.jsonl", ITEMS)
    return rsd.main([str(items), "--system", system, "--tokenizer", str(model[0]), "--server", SERVER,
                     "--out", str(tmp_path / "out" / "b0.jsonl"), *args],
                    local_client=fake.client(base_url=SERVER))


def outputs(tmp_path) -> tuple[list[dict], dict, dict]:
    out = tmp_path / "out"
    records = [json.loads(line) for line in (out / "b0.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((out / "b0.manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "b0.summary.json").read_text(encoding="utf-8"))
    return records, manifest, summary


PROPS_WITH_MODEL = {**PROPS, "model_path": "C:/models/Nanbeige4.1-3B-Q8_0.gguf"}


def test_b0_answers_every_item_on_the_family_path_thinking_greedily_without_tools(tmp_path, model, capsys):
    fake = FakeLlama(script(), props=PROPS_WITH_MODEL)
    assert run(tmp_path, model, fake) == 0
    records, manifest, summary = outputs(tmp_path)
    assert [r["id"] for r in records] == [i["id"] for i in ITEMS] and {r["status"] for r in records} == {"ok"}
    first, forced, call = records
    assert first["answer"] == INCORRECT_ANSWER and first["reasoning"] == "The value -3 is below 1."
    assert first["reasoning_chars"] == len("The value -3 is below 1.") and first["finish_reason"] == "stop"
    assert first["thinking"] == {"mode": "think", "reasoning_tokens": 2, "answer_tokens": 1, "forced": False,
                                 "phases": 1, "seeds": None}
    assert forced["answer"] == MISSING_ANSWER and forced["thinking"]["forced"] is True
    assert forced["thinking"]["phases"] == 2 and forced["reasoning"] == "No traveller is given, so"
    assert call["answer"] == CALL and call["model"] == "model:lora=0" and call["prompt_tokens"] > 0
    # the requests: the item's messages as the family renders them in think mode, no tools, greedy, R = 1,280
    requests = fake.requests
    assert len(requests) == 4
    tokenizer = model[1]
    prompt = tokenizer.decode(requests[0]["prompt"])
    assert prompt.startswith("<|im_start|>system\n" + SYSTEM) and ITEMS[0]["messages"][1]["content"] in prompt
    assert prompt.endswith("<|im_start|>assistant\n" + THINK_OPEN) and "<tools>" not in prompt
    assert all(r["tools"] is None for r in tokenizer.renders[-3:])
    assert all(r["temperature"] == 0.0 and "seed" not in r and "top_p" not in r for r in requests)
    assert all("lora" not in r and r["cache_prompt"] is False for r in requests)  # a bare server, scale 0
    # the reasoning's budget, then the answer's after the close
    assert requests[0]["max_tokens"] == 1280 and requests[2]["max_tokens"] == 1200
    assert requests[2]["prompt"][:len(requests[1]["prompt"])] == requests[1]["prompt"]
    assert manifest["system"] == "b0" and manifest["lora_scale"] == 0.0 and manifest["adapters"] == []
    assert manifest["server"] == SERVER and manifest["server_props"] == {
        "total_slots": 1, "build_info": "fake-b11349", "model_path": "C:/models/Nanbeige4.1-3B-Q8_0.gguf",
        "model": "Nanbeige4.1-3B-Q8_0.gguf", "n_ctx": 8192}
    generation = manifest["generation"]
    assert generation["mode"] == "think" and generation["decoding"]["temperature"] == 0.0
    assert generation["reasoning_budget"] == 1280 and generation["answer_tokens"] == 1200
    assert manifest["family"]["name"] == "fake-only" and manifest["tokenizer"] == str(model[0])
    assert manifest["request"]["tools"] is None and manifest["request"]["cache_prompt"] is False
    items_path = tmp_path / "special-dev-b.jsonl"
    assert manifest["items_sha256"] == hashlib.sha256(items_path.read_bytes()).hexdigest()
    assert manifest["items_locked"] is True and manifest["items_run"] == manifest["items_total"] == 3
    assert manifest["statuses"] == {"ok": 3} and "date" in manifest and "git_commit" in manifest
    assert summary["summary"]["special"]["strict"] == 2 and summary["summary"]["valid"]["exact"] == 1
    assert summary["items"] == items_path.as_posix() and summary["items_sha256"] == manifest["items_sha256"]
    assert [s["id"] for s in summary["scores"]] == [i["id"] for i in ITEMS] and "strict" in summary["rules"]
    printed = capsys.readouterr().out
    assert "special: strict 2/2 (100.0%)" in printed and "valid: exact 1/1 (100.0%)" in printed


def test_ft_runs_at_scale_1_on_the_one_adapter_the_server_lists_and_records_its_sha(tmp_path, model):
    gguf = tmp_path / "checkpoint-1635-f16.gguf"
    gguf.write_bytes(b"GGUF of FT3")
    fake = FakeLlama(script(), adapters=[{"id": 0, "path": str(gguf), "scale": 0.0}])
    assert run(tmp_path, model, fake, system="ft") == 0
    _, manifest, _ = outputs(tmp_path)
    assert manifest["system"] == "ft" and manifest["lora_scale"] == 1.0
    assert manifest["adapters"] == [{"id": 0, "path": str(gguf),
                                     "sha256": hashlib.sha256(b"GGUF of FT3").hexdigest()}]
    assert all(r["lora"] == [{"id": 0, "scale": 1.0}] for r in fake.requests)


def test_b0_sends_scale_0_on_a_server_with_an_adapter(tmp_path, model):
    gguf = tmp_path / "adapter.gguf"
    gguf.write_bytes(b"GGUF")
    fake = FakeLlama(script(), adapters=[{"id": 0, "path": str(gguf), "scale": 0.0}])
    assert run(tmp_path, model, fake) == 0
    assert all(r["lora"] == [{"id": 0, "scale": 0.0}] for r in fake.requests)


class Unhealthy(FakeLlama):
    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(503, json={"status": "loading"})
        return super().handle(request)


@pytest.mark.parametrize(
    "fake,system",
    [
        (lambda gguf: FakeLlama(script()), "ft"),  # no adapter loaded
        (lambda gguf: FakeLlama(script(), adapters=[{"id": n, "path": str(gguf)} for n in (0, 1)]), "ft"),
        (lambda gguf: FakeLlama(script(), props={**PROPS, "total_slots": 4}), "b0"),
        (lambda gguf: Unhealthy(script()), "b0"),
        (lambda gguf: FakeLlama(script(), adapters=[{"id": 0, "path": "not-here.gguf"}]), "b0"),
    ],
)
def test_a_server_that_cannot_run_the_system_exits_2_and_writes_nothing(tmp_path, model, fake, system):
    gguf = tmp_path / "a.gguf"
    gguf.write_bytes(b"GGUF")
    server = fake(gguf)
    assert run(tmp_path, model, server, system=system) == 2
    assert not (tmp_path / "out").exists() and server.requests == []


def test_a_family_that_does_not_think_or_no_registered_family_exits_2(tmp_path, monkeypatch):
    folder = tmp_path / "model"
    install_fake_family(monkeypatch, folder, "none")
    assert run(tmp_path, (folder, None), FakeLlama(script())) == 2
    assert run(tmp_path, (tmp_path / "nowhere", None), FakeLlama(script())) == 2
    assert not (tmp_path / "out").exists()


def test_existing_outputs_are_kept_unless_forced(tmp_path, model):
    assert run(tmp_path, model, FakeLlama(script())) == 0
    before = (tmp_path / "out" / "b0.jsonl").read_bytes()
    fake = FakeLlama(script())
    assert run(tmp_path, model, fake) == 2 and fake.requests == []
    assert run(tmp_path, model, FakeLlama(script()), "--force") == 0
    assert (tmp_path / "out" / "b0.jsonl").read_bytes() == before  # greedy on one slot: the same records


def test_a_failed_request_is_an_error_record_and_the_run_goes_on(tmp_path, model):
    fake = FakeLlama(script()[:1])  # the second item's request is past the script's end: 500
    assert run(tmp_path, model, fake) == 0
    records, manifest, summary = outputs(tmp_path)
    assert [r["status"] for r in records] == ["ok", "error", "error"]
    assert records[1]["error_type"] == "HTTPStatusError" and manifest["statuses"] == {"error": 2, "ok": 1}
    assert summary["summary"]["special"]["failures"]["no_answer"] == 1
    assert summary["summary"]["valid"]["failures"]["no_answer"] == 1


def test_an_item_that_does_not_fit_the_context_is_not_asked(tmp_path, model):
    fake = FakeLlama(script(), props={**PROPS, "default_generation_settings": {"n_ctx": 2000}})
    assert run(tmp_path, model, fake) == 0
    records, manifest, _ = outputs(tmp_path)
    assert {r["status"] for r in records} == {"context_overflow"} and fake.requests == []
    assert all(r["prompt_tokens"] > 0 for r in records) and manifest["statuses"] == {"context_overflow": 3}


def test_limit_runs_the_first_items(tmp_path, model):
    assert run(tmp_path, model, FakeLlama(script()), "--limit", "1") == 0
    records, manifest, summary = outputs(tmp_path)
    assert [r["id"] for r in records] == [ITEMS[0]["id"]] and manifest["items_run"] == 1
    assert manifest["limit"] == 1 and summary["summary"]["n"] == 1


@pytest.mark.parametrize("change", [{"id": "req-test-0001-s1"}, {"split": "test"}])
def test_test_material_is_refused_before_any_request(tmp_path, model, change):
    items = write_items(tmp_path / "items.jsonl", [{**ITEMS[0], **change}, *ITEMS[1:]], lock=False)
    fake = FakeLlama(script())
    assert run(tmp_path, model, fake, items=items) == 4
    assert fake.requests == [] and not (tmp_path / "out").exists()


def test_held_out_sets_are_refused_by_path_name_folder_and_content(tmp_path, model, monkeypatch):
    root = tmp_path / "held-root"
    held = write_items(root / "data" / "eval" / "v2" / "dev-v2-ext.jsonl", ITEMS, lock=False)
    by_name = write_items(tmp_path / "elsewhere" / "dev-ext.jsonl", ITEMS, lock=False)
    by_folder = write_items(root / "runs" / "m7" / "heldout" / "sets" / "items.jsonl", ITEMS[:1], lock=False)
    copy = tmp_path / "copy.jsonl"
    copy.write_bytes(held.read_bytes())
    for path in (held, by_name, by_folder, copy):
        fake = FakeLlama(script())
        assert run(tmp_path, model, fake, items=path) == 4, path
        assert fake.requests == []
    assert rsd.held_out(write_items(tmp_path / "fine.jsonl", ITEMS[:2], lock=False)) is None


# ---- the run copy: named by --run-copy or the environment, never built in ----------------------------------


def test_without_a_named_run_copy_the_held_out_sets_are_looked_for_under_the_repository_alone():
    fresh = load_script("run_special_dev")  # this module's roots are the fixture's; a fresh load has its own
    assert fresh.HELD_OUT_ROOTS == (fresh.REPO,)
    assert fresh.held_out_roots() == (fresh.REPO,)


def test_the_run_copy_comes_from_the_option_or_else_the_environment(tmp_path, monkeypatch):
    root, option, environment = tmp_path / "held-root", tmp_path / "option", tmp_path / "environment"
    assert rsd.held_out_roots() == (root,)
    monkeypatch.setenv(rsd.RUN_COPY_ENV, str(environment))
    assert rsd.held_out_roots() == (root, environment)
    assert rsd.held_out_roots(option) == (root, option)  # the option wins over the environment
    monkeypatch.setenv(rsd.RUN_COPY_ENV, "")
    assert rsd.held_out_roots() == (root,)


def test_a_set_in_the_run_copys_held_out_folder_is_refused_once_the_run_copy_is_named(tmp_path, model,
                                                                                    monkeypatch, capsys):
    copy = tmp_path / "run-copy"
    items = write_items(copy / "runs" / "m7" / "heldout" / "items.jsonl", ITEMS, lock=False)
    assert rsd.held_out(items) is None  # no run copy named: not one of the held-out folders
    assert rsd.held_out(items, rsd.held_out_roots(copy)) == "folder"
    fake = FakeLlama(script())
    assert run(tmp_path, model, fake, "--run-copy", str(copy), items=items) == 4
    assert fake.requests == [] and "found by its folder" in capsys.readouterr().err
    monkeypatch.setenv(rsd.RUN_COPY_ENV, str(copy))
    fake = FakeLlama(script())
    assert run(tmp_path, model, fake, items=items) == 4 and fake.requests == []


def test_a_set_that_does_not_match_its_lock_exits_5(tmp_path, model):
    items = write_items(tmp_path / "special-dev-b.jsonl", ITEMS)
    items.write_bytes(items.read_bytes() + b"\n")
    fake = FakeLlama(script())
    assert run(tmp_path, model, fake, items=items) == 5 and fake.requests == []


def refixed(item: dict, fixed: list[str], **changes) -> str:
    """The item as a JSON line with other fixed words in its expected sentence."""
    return json.dumps({**item, **changes, "expected": {**item["expected"], "fixed": fixed}})


# a sentence whose two slots can hold an incorrect case's fillers and a missing case's alike
EITHER = "- When unsure, reply only with: Not sent: <function> or <value> fails"


def either(item: dict, family: str) -> str:
    """The item, its system message holding EITHER too, answered with EITHER's fixed words, of the family."""
    messages = [{"role": "system", "content": SYSTEM + "\n" + EITHER}, *item["messages"][1:]]
    return refixed(item, ["Not sent:", "or", "fails"], messages=messages, family=family)


@pytest.mark.parametrize(
    "lines",
    [
        ["not json"],
        [json.dumps(ITEMS[0]), json.dumps(ITEMS[0])],  # one id twice
        [refixed(ITEMS[0], ["Not sent: the value", "is odd"])],  # not in the system message
        [refixed(ITEMS[0], ["Not sent: the value", "breaks the definition."])],  # one slot for two
        [either(ITEMS[0], "fx-one"), either(ITEMS[1], "fx-two")],  # one sentence for two cases
    ],
)
def test_malformed_items_exit_1(tmp_path, model, lines):
    items = tmp_path / "items.jsonl"
    items.write_text("\n".join(lines) + "\n", encoding="utf-8")
    fake = FakeLlama(script())
    assert run(tmp_path, model, fake, items=items) == 1 and fake.requests == []


def test_either_sentence_is_well_formed_for_each_case_alone():
    for line in (either(ITEMS[0], "fx-one"), either(ITEMS[1], "fx-two")):
        assert rsd.sd.check_item(json.loads(line)) == []


# ---- one family in two placeholder styles, as the data package may write a set ---------------------------

STYLE_WORDS = {"angle": {"value": "value", "param": "parameter", "params": "parameters", "api": "function"},
               "caps": {"value": "VALUE_GIVEN", "param": "PARAM_NAME", "params": "MISSING_PARAMS",
                        "api": "FUNCTION_NAME"}}


def mix_family(case: str) -> sm.Family:
    """The first English development family of a case (research_agent.data.special_mix, read only)."""
    return next(family for fid, family in sorted(sm.FAMILIES.items()) if fid.startswith(f"{case}-en-b"))


def mixed_system(style: str) -> str:
    sentences = [sm.fill_sentence(mix_family(case), style, "en", **STYLE_WORDS[style])
                 for case in ("incorrect", "missing", "irrelevant")]
    return "\n".join([
        "You may call the functions below; each line is one function as JSON.",
        json.dumps(BOOK_ROOM), json.dumps(PLAN_TRIP), "", "How to reply:",
        "- When the request can be served, reply only with the calls, one per line, as Python calls.",
        f"- When a value given breaks its definition, reply only with: {sentences[0]}",
        f"- When a required parameter has no value, reply only with: {sentences[1]}",
        f"- When no listed function fits, reply only with: {sentences[2]}"])


def mixed_items() -> list[tuple[dict, str]]:
    """(item, its ideal answer) of a set whose incorrect and missing families each come in two styles."""
    out = []
    cases = [("incorrect", "angle", "Book room A-12 for -3 people on 2026-11-05.",
              {"values": ["-3"], "params": ["people"]}),
             ("incorrect", "caps", "Book room A-12 for 0 people on 2026-11-05.",
              {"values": ["0"], "params": ["people"]}),
             ("missing", "angle", "Plan a trip to Lisbon.", {"params": ["traveller"], "api": "plan_trip"}),
             ("missing", "caps", "Book room A-12 for 2 people.", {"params": ["date"], "api": "book_room"})]
    for number, (case, style, user, names) in enumerate(cases, 1):
        family = mix_family(case)
        expected = {"type": "refusal", "case": case, "fixed": sm.fixed_parts(family, style, "en"),
                    "values": names.get("values", []), "params": names["params"], "api": names.get("api", "")}
        mixed = {**item(number, case, user, expected), "family": family.id}
        mixed["messages"] = [{"role": "system", "content": mixed_system(style)}, *mixed["messages"][1:]]
        answer = sm.fill_sentence(family, style, "en", value=", ".join(expected["values"]) or None,
                                  param=", ".join(names["params"]), params=", ".join(names["params"]),
                                  api=expected["api"] or None)
        out.append((mixed, answer))
    valid = {**ITEMS[2], "id": "sdev-b-0005"}
    valid["messages"] = [{"role": "system", "content": mixed_system("caps")}, *valid["messages"][1:]]
    return [*out, (valid, CALL)]


def test_a_set_that_writes_one_family_in_two_styles_is_run_and_scored(tmp_path, model):
    pairs = mixed_items()
    path = write_items(tmp_path / "mixed" / "special-dev-b.jsonl", [mixed for mixed, _ in pairs])
    items, locked = rsd.load_items(path)
    assert locked and len(items) == 5
    assert {len(fixed) for fixed in rsd.sd.sentence_families(items).values()} == {2}
    fake = FakeLlama([Scripted(("Read the rules.", "\n</think>\n\n", answer)) for _, answer in pairs])
    assert run(tmp_path, model, fake, items=path) == 0
    _, _, summary = outputs(tmp_path)
    special, valid = summary["summary"]["special"], summary["summary"]["valid"]
    assert (special["n"], special["strict"], special["lenient"]) == (4, 4, 4)
    assert (valid["n"], valid["exact"], valid["false_refusals"]) == (1, 1, 0)


# ---- what scoring may not cost: the records ---------------------------------------------------------------


def test_the_records_and_the_manifest_are_written_before_scoring(tmp_path, model, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("the summary broke")

    monkeypatch.setattr(rsd.sd, "summarize", broken)
    with pytest.raises(RuntimeError, match="the summary broke"):
        run(tmp_path, model, FakeLlama(script()))
    out = tmp_path / "out"
    records = [json.loads(line) for line in (out / "b0.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["status"] for r in records] == ["ok", "ok", "ok"]
    assert json.loads((out / "b0.manifest.json").read_text(encoding="utf-8"))["items_run"] == 3
    assert not (out / "b0.summary.json").exists()


def test_a_scoring_error_is_kept_to_its_item_counted_and_the_run_exits_0(tmp_path, model, monkeypatch,
                                                                         capsys):
    real = rsd.sd.score_item

    def flaky(answer, item, **kwargs):
        if item["id"] == ITEMS[1]["id"]:
            raise ValueError("cannot score this")
        return real(answer, item, **kwargs)

    monkeypatch.setattr(rsd.sd, "score_item", flaky)
    assert run(tmp_path, model, FakeLlama(script())) == 0
    records, manifest, summary = outputs(tmp_path)
    assert [r["status"] for r in records] == ["ok", "ok", "ok"] and manifest["statuses"] == {"ok": 3}
    assert summary["summary"]["scoring_errors"] == 1 and summary["summary"]["special"]["strict"] == 1
    failed = summary["scores"][1]
    assert (failed["status"], failed["failure"]) == ("scoring_error", "scoring_error")
    assert "ValueError: cannot score this" in failed["error"]
    printed = capsys.readouterr().out
    assert f"{ITEMS[1]['id']} missing en: scoring_error (ValueError: cannot score this)" in printed


def test_the_files_keep_the_keys_decide_m8_reads(tmp_path, model):
    assert run(tmp_path, model, FakeLlama(script())) == 0
    _, manifest, summary = outputs(tmp_path)
    assert summary["records"] == "b0.jsonl" and summary["items_sha256"] == manifest["items_sha256"]
    assert summary["items"] == (tmp_path / "special-dev-b.jsonl").as_posix()
    assert all(isinstance(summary["summary"]["special"][key], int) for key in ("n", "strict", "lenient"))
    assert all(isinstance(summary["summary"]["valid"][key], int)
               for key in ("n", "exact", "false_refusals", "zh_altered"))
    assert {"system", "lora_scale", "adapters", "generation", "items_sha256", "items_run",
            "items_total"} <= set(manifest)

"""scripts/check_m8_revision.py (docs/m8-计划.md WP3, the data revision), offline, on fixture files in a
temporary folder: data/eval/dev.jsonl (locked) with the selection runs of FT5's last checkpoint (seed 0) and
of FT3's steps 1308 and 1635 (seed 0's run over dev written, seed 1 and dev-v2 listed alone: the check reads
neither), their run manifests beside them as scripts/run_eval.py writes them, and FT5's training result; and a
locked dev set A of research_agent.eval.special_dev's items (both languages; the special kinds, priority and
valid items) with FT3's and FT5's runs over it as scripts/run_special_dev.py writes them (records, run
manifest and summary, the summary the scorer's of the records). Each trigger at its edges (R1 at 3/5 and one
item below, the priority items left out; R2 at +2 and +3; R3 at -7 and -8, counted with the runtime rules
applied offline) and with FT5 better than FT3 (R2 at -3, R3 at -8), the file written with no trigger fired
(exit 0) and with one (exit 3), the refusal to check twice, and the refusal of every input that is not the
one the check needs: a step the selection does not hold, a checkpoint that is not FT5's last as FT5 is now (a
selection of another run, a step before the last, FT3's GGUF, a GGUF trained again or gone), a dev run's
manifest of another adapter, set or SHA-256 or measured otherwise than FT3's, a dev set A run of another
adapter, set, lock or generation, records not all ok, a summary counting otherwise than its records scored
again, a set changed since its lock, test material and a held-out set. The last test runs the real
scripts/run_special_dev.py (on tests/support's FakeLlama) for the dev set A runs. The guards look under the
test's folder alone (the repository root, the check's default --out and the held-out roots are moved there)
and the runner's git calls are stubbed, so no set, run or result of the repository, no test set's lock and no
held-out set is looked at, nothing is written in the repository, and git is not run."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
from support.llama_fakes import PROPS, FakeLlama, Scripted, install_fake_family

from research_agent.agent.prompts import system_prompt
from research_agent.eval import special_dev
from research_agent.eval.lock import sha256_of, write_jsonl, write_lock
from research_agent.family.core import Decoding, Mode

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = load_script("check_m8_revision")
SCRIPT_OUT = check.OUT  # the default --out, before the repo fixture moves it under the test's folder

DEV, DEV_V2 = "data/eval/dev.jsonl", "data/eval/v2/dev-v2.jsonl"
DEV_A = "data/eval/special-dev-a.jsonl"
FT5_SELECTION = "runs/m8/queue/selection-ft5-last-s0.json"
FT3_SELECTION = "results/checkpoint-selection-ft3.json"  # the default --baseline
OUT = "results/revision-check-m8.json"  # given as --out: under the test's folder
FT5_RUN, FT3_RUN = "runs/train/ft5", "runs/train/ft3"  # FT5_RUN: the default --run
FT5_RESULT = f"{FT5_RUN}/result.json"  # scripts/train.py's result of FT5's run: the steps trained
STEP, FT3_STEP, FT3_OTHER = 448, 1635, 1308
EARLIER = 336  # FT5's 3/4 checkpoint, before its last (STEP)
TRIGGERS = ("R1", "R2", "R3")
SYSTEM = {"role": "system", "content": system_prompt("2026-10-10")}
ARGS = {"query": "hallucination", "max_results": 5}
FIND = {"content": "", "tool_calls": [{"name": "search_arxiv", "arguments": ARGS,
                                       "arguments_json": json.dumps(ARGS)}]}
BYE = {"content": "See you!", "tool_calls": []}


def call(arguments: dict = ARGS, id_: str = "call_0") -> dict:
    return {"id": id_, "type": "function",
            "function": {"name": "search_arxiv", "arguments": json.dumps(arguments)}}


def user(text: str) -> list[dict]:
    return [SYSTEM, {"role": "user", "content": text}]


FAILED = [*user("Find hallucination papers again."),
          {"role": "assistant", "content": "", "tool_calls": [call(id_="c1")]},
          {"role": "tool", "tool_call_id": "c1", "content": json.dumps({"error": "RATE_LIMITED"})}]


def item(id_: str, messages: list[dict], expected: dict, type_: str) -> dict:
    return {"id": id_, "messages": messages, "expected": expected, "type": type_, "lang": "en",
            "family": "find/en", "split": "dev", "slots": {"topic": "hallucination"},
            "source": {"request": id_.rsplit("-s", 1)[0], "step": 1}}


# data/eval/dev.jsonl: eight searches, one search after a rate limit (the runtime rules retry it), a goodbye
DEV_ITEMS = [*(item(f"req-dev-{k:04d}-s1", user(f"Find hallucination papers ({k})."), FIND, "call")
               for k in range(1, 9)),
             item("req-dev-0009-s1", FAILED, FIND, "recover"),
             item("req-dev-0010-s1", user("Thanks, that's all."), BYE, "no_call")]
RIGHT, CATS = ("", (call(),)), ("", (call({"query": "cats", "max_results": 5}),))
BUSY, GOODBYE = ("The service is busy, try again later.", ()), ("See you!", ())
GENERATION = {"answer_tokens": 512, "base_seed": 0, "mode": "think", "reasoning_budget": 1280, "sample": 0,
              "decoding": {"min_p": 0.0, "presence_penalty": 0.0, "temperature": 0.6, "top_k": 0,
                           "top_p": 0.95}}
FAMILY = {"name": "nanbeige4.1-3b", "thinking": "only", "template_sha256": "7" * 64}
BASE_GGUF = "models/Base-Q8_0.gguf"
# what else a run manifest of scripts/run_eval.py records that makes a count FT3's metric (FT3-1635's values)
MEASURED = {"prompt_hash": "c827097301c986b1", "prompt_version": "m2-2026-10-13",
            "schema_hash": "7d831712685e42d0", "lora_scale": 1.0,
            "server_props": {"build_info": "b11349-fb4b2737a", "model_path": BASE_GGUF, "total_slots": 1}}
DEV_V2_SHA256 = "a" * 64  # dev-v2 is listed in the selections' item_sets, and no file of it is here


def answers_of(wrong: int, busy: bool = False) -> dict[str, tuple]:
    """A seed-0 run's answers: the first `wrong` searches answered with another query, the search after the
    rate limit answered with a busy notice (which the rules turn into the retry) when busy, every other one
    right: 10 - wrong successes with the rules, one fewer without them when busy."""
    searches = [row["id"] for row in DEV_ITEMS if row["type"] == "call"]
    out = {}
    for row in DEV_ITEMS:
        if row["type"] == "no_call":
            out[row["id"]] = GOODBYE
        elif row["type"] == "recover":
            out[row["id"]] = BUSY if busy else RIGHT
        else:
            out[row["id"]] = CATS if searches.index(row["id"]) < wrong else RIGHT
    return out


def record(row: dict, answer: tuple) -> dict:
    content, calls = answer
    output = {"role": "assistant", "content": content, **({"tool_calls": list(calls)} if calls else {})}
    return {"id": row["id"], "status": "ok",
            "item": {k: v for k, v in row.items() if k not in ("id", "messages")}, "output": output,
            "usage": {"prompt_tokens": 900, "completion_tokens": 20}, "paid": False, "cached": False,
            "model": "nanbeige4.1-3b:lora=1"}


def dev_records(run: str, step: int, seed: str = "s0", path: str = DEV) -> str:
    """Where a checkpoint's run of a seed over a set is (scripts/select_checkpoint.py with several sets)."""
    return f"runs/eval/select-{Path(run).name}/{Path(path).stem}/checkpoint-{step}.{seed}.jsonl"


def dev_manifest(run: str, step: int) -> Path:
    records = Path(dev_records(run, step))
    return records.with_name(records.stem + ".manifest.json")


def gguf_path(run: str, step: int) -> Path:
    """A checkpoint's GGUF, as scripts/export_adapter.py names it in its run."""
    return Path(f"{run}/gguf/checkpoint-{step}-f16.gguf")


def write_checkpoint(run: str, step: int, dev: tuple[int, bool], seeds: tuple[str, ...]) -> dict:
    """A checkpoint's GGUF, its seed-0 run over dev with the run manifest beside it (as scripts/run_eval.py
    writes them; dev: the wrong searches and the busy notice of answers_of), and its row of a selection file
    (as scripts/select_checkpoint.py writes it with --variant and two --items): every seed of seeds over both
    sets listed, the run of seed 0 over dev alone written."""
    gguf = gguf_path(run, step)
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(f"GGUF {run} {step}".encode())
    variants = {seed: {"sets": {path: {"n": len(DEV_ITEMS) if path == DEV else 5,
                                       "records": dev_records(run, step, seed, path)}
                                for path in (DEV, DEV_V2)}} for seed in seeds}
    answers = answers_of(*dev)
    write_jsonl([record(row, answers[row["id"]]) for row in DEV_ITEMS], dev_records(run, step))
    n = len(DEV_ITEMS)
    manifest = {"adapters": [{"id": 0, "path": gguf.as_posix(), "sha256": sha256_of(gguf)}],
                "family": FAMILY, "generation": GENERATION, "items": DEV, "items_sha256": sha256_of(DEV),
                "items_run": n, "items_total": n, "statuses": {"ok": n}, "system": "ft",
                **json.loads(json.dumps(MEASURED))}
    dev_manifest(run, step).write_text(json.dumps(manifest), encoding="utf-8")
    return {"step": step, "gguf": gguf.as_posix(), "sha256": sha256_of(gguf), "variants": variants}


def write_selection(path: str, run: str, rows: list[dict], seeds: tuple[str, ...]) -> None:
    selection = {"run": run, "system": "ft", "lora_scale": 1.0,
                 "item_sets": [{"path": DEV, "sha256": sha256_of(DEV)},
                               {"path": DEV_V2, "sha256": DEV_V2_SHA256}],
                 "variants": [{"name": seed, "args": ["--mode", "think", "--seed", seed[1:],
                                                      "--reasoning-budget", "1280", "--signals"]}
                              for seed in seeds],
                 "selection": {"metric": "mean", "variants": list(seeds)}, "checkpoints": rows,
                 "chosen": rows[-1]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(selection, indent=2), encoding="utf-8")


# Dev set A in miniature, research_agent.eval.special_dev's items as the tests of scripts/decide_m8.py write
# dev set B's: two functions and the three refusal sentences in the system message, the languages alternating
BOOK_ROOM = {"name": "book_room", "description": "Book a meeting room.", "parameters": {
    "type": "object", "properties": {"room": {"type": "string"}, "people": {"type": "integer", "minimum": 1},
                                     "date": {"type": "string"}}, "required": ["room", "people", "date"]}}
PLAN_TRIP = {"name": "plan_trip", "description": "Plan a trip.", "parameters": {
    "type": "object", "properties": {"traveller": {"type": "object"}, "destination": {"type": "string"}},
    "required": ["traveller", "destination"]}}
DEV_A_SYSTEM = "\n".join([
    "You may call the functions below; each line is one function as JSON.",
    json.dumps(BOOK_ROOM), json.dumps(PLAN_TRIP), "", "How to reply:",
    "- When the request can be served, reply only with the calls, one per line, as Python calls.",
    ("- When a value given breaks its definition, reply only with: Not sent: the value <value> of <param> "
     "breaks the definition."),
    "- When a required parameter has no value, reply only with: Not sent: <api> still needs <params>",
    "- When no listed function fits, reply only with: Not sent: no listed function can do this.",
])
FIXED = {"incorrect": ["Not sent: the value", "of", "breaks the definition."],
         "missing": ["Not sent:", "still needs"],
         "irrelevant": ["Not sent: no listed function can do this."]}
NO_FUNCTION = "Not sent: no listed function can do this."
# dev set A's generation as scripts/run_special_dev.py records it: think, greedy, R 1280
DEV_A_GENERATION = {"answer_tokens": 1200, "base_seed": 0, "mode": "think", "reasoning_budget": 1280,
                    "sample": 0, "decoding": {"min_p": 0.0, "presence_penalty": 0.0, "temperature": 0.0,
                                              "top_k": 0, "top_p": 1.0}}


def dev_a_item(number: int, kind: str, text: str, expected: dict) -> dict:
    return {"id": f"sdev-a-{number:04d}", "split": "a", "lang": ("zh", "en")[number % 2], "kind": kind,
            "messages": [{"role": "system", "content": DEV_A_SYSTEM}, {"role": "user", "content": text}],
            "call_format": "python", "family": "fx-revise", "expected": expected,
            "source": {"dataset": "fixture", "record": str(number)}}


def refusal(case: str, values: tuple = (), params: tuple = (), api: str = "") -> dict:
    return {"type": "refusal", "case": case, "fixed": FIXED[case], "values": list(values),
            "params": list(params), "api": api}


def booking(room: str, people: int = 4) -> str:
    return f'book_room(room="{room}", people={people}, date="2026-11-05")'


def dev_a_set() -> list[tuple[dict, str]]:
    """Dev set A's items, each with its right answer: five of each special kind (15 special items, both
    languages), five priority items and ten valid ones."""
    rows = [("incorrect", f"Book room A-12 for -{k} people on 2026-11-05.",
             refusal("incorrect", (f"-{k}",), ("people",)),
             f"Not sent: the value -{k} of people breaks the definition.") for k in range(1, 6)]
    rows += [("missing", f"Plan trip {k} to Lisbon.",
              refusal("missing", params=("traveller",), api="plan_trip"),
              "Not sent: plan_trip still needs traveller") for k in range(1, 6)]
    rows += [("irrelevant", f"What is the weather in Oslo on day {k}?", refusal("irrelevant"), NO_FUNCTION)
             for k in range(1, 6)]
    rows += [("priority", f"Book room A-12 for -{k} people.", refusal("incorrect", (f"-{k}",), ("people",)),
              f"Not sent: the value -{k} of people breaks the definition.") for k in range(6, 11)]
    for k in range(1, 11):
        room = f"会议室{k}"
        rows.append(("valid_verbatim", f"Book room {room} for 4 people on 2026-11-05.",
                     {"type": "call", "calls": [{"name": "book_room", "arguments": {
                         "room": room, "people": 4, "date": "2026-11-05"}}], "verbatim": [room]},
                     booking(room)))
    return [(dev_a_item(number, kind, text, expected), right)
            for number, (kind, text, expected, right) in enumerate(rows, 1)]


DEV_A_SET = dev_a_set()
DEV_A_ITEMS = len(DEV_A_SET)  # 30: 15 special, 5 priority and 10 valid


LENIENT_ONLY = 3  # special items answered lenient and not strict after the strict ones (R1 reads strict)


def records_a(strict: int, refused: int) -> list[dict]:
    """A run's records over dev set A: the first `strict` special items answered with their sentence alone,
    the next LENIENT_ONLY with their sentence and another one after it (lenient, not strict), the others
    "OK."; every priority item with its sentence; the first `refused` valid items with the irrelevant case's
    sentence (false refusals), the others with their call."""
    special = valid = 0
    records = []
    for row, right in DEV_A_SET:
        if row["kind"] in special_dev.SPECIAL_KINDS:
            answer = (right if special < strict else f"{right} Please send it again."
                      if special < strict + LENIENT_ONLY else "OK.")
            special += 1
        elif row["kind"] == special_dev.PRIORITY:
            answer = right
        else:
            answer = NO_FUNCTION if valid < refused else right
            valid += 1
        records.append({"id": row["id"], "status": "ok", "answer": answer})
    return records


_SCORED: dict[tuple[int, int], tuple[list[dict], list[dict], dict]] = {}


def scored_a(strict: int, refused: int) -> tuple[list[dict], list[dict], dict]:
    """records_a's records with their scores and summary by special_dev, checked to count what they are made
    to count; computed once per pair (not to be changed in place)."""
    if (strict, refused) not in _SCORED:
        records, items = records_a(strict, refused), [row for row, _ in DEV_A_SET]
        scores = special_dev.score_records(records, items)
        value = special_dev.summarize(scores, items)
        got = (value["special"]["n"], value["special"]["strict"], value["special"]["lenient"],
               value["valid"]["n"], value["valid"]["false_refusals"], value["priority"]["strict"])
        want = (15, strict, min(strict + LENIENT_ONLY, 15), 10, refused, 5)
        assert got == want and not value["scoring_errors"], f"the answers count {got}, not {want}"
        _SCORED[(strict, refused)] = (records, scores, value)
    return _SCORED[(strict, refused)]


def dev_a_files(label: str) -> tuple[Path, Path, Path]:
    """A dev set A run's records, run manifest and summary at the default places of --dev-a-ft3 and
    --dev-a-ft5, named as scripts/run_special_dev.py names them."""
    records = Path("runs/m8/special-dev/" + ("a-ft3.jsonl" if label == "ft3" else f"a-ft5-{STEP}.jsonl"))
    return (records, records.with_name(records.stem + ".manifest.json"),
            records.with_name(records.stem + ".summary.json"))


def adapter_of(row: dict) -> dict:
    """The server's adapter listing of a checkpoint, as a run manifest records it (eval.systems)."""
    return {"id": 0, "path": row["gguf"], "sha256": row["sha256"]}


def write_dev_a(label: str, counts: tuple[int, int], *, adapter: dict, lock: str) -> None:
    """A run over dev set A as scripts/run_special_dev.py writes it: its records (records_a(*counts)), its
    manifest (ft: the adapter at scale 1, every record ok) and its summary (the records' name, the set, its
    SHA-256, the scorer's rules, its summary under "summary" and every item's score)."""
    records, manifest, summary_file = dev_a_files(label)
    rows, scores, value = scored_a(*counts)
    write_jsonl(rows, records)
    manifest.write_text(json.dumps({
        "script": "scripts/run_special_dev.py", "system": "ft", "lora_scale": 1.0, "adapters": [adapter],
        "family": FAMILY, "generation": DEV_A_GENERATION, "items": DEV_A, "items_sha256": lock,
        "items_locked": True, "items_total": DEV_A_ITEMS, "items_run": DEV_A_ITEMS, "limit": None,
        "statuses": {"ok": DEV_A_ITEMS},
        "server_props": {"total_slots": 1, "build_info": "b11349-fb4b2737a", "model_path": BASE_GGUF}}),
        encoding="utf-8")
    summary_file.write_text(json.dumps({"records": records.name, "items": DEV_A, "items_sha256": lock,
                                        "rules": special_dev.RULES, "summary": value, "scores": scores},
                                       ensure_ascii=False), encoding="utf-8")


def world(*, ft3_dev: tuple[int, bool] = (0, False), ft5_dev: tuple[int, bool] = (7, False),
          ft3_a: tuple[int, int] = (12, 1), ft5_a: tuple[int, int] = (9, 3)) -> list[str]:
    """Everything the check reads, written in the current folder; its arguments (--baseline, --run and the
    dev set A runs at their defaults). *_dev: a seed-0 run's wrong searches and busy notice over dev; *_a: a
    dev set A run's strict special items and false refusals. By default every trigger stands at its edge,
    unfired: R1 FT5's 9/15 = 3/5, R2 3 - 1 = 2, R3 10 - 3 = 7 (FT3's step 1308, 7 successes, is not the
    baseline)."""
    write_jsonl(DEV_ITEMS, DEV)
    write_lock(DEV)
    write_jsonl([row for row, _ in DEV_A_SET], DEV_A)
    lock = write_lock(DEV_A)
    ft5 = write_checkpoint(FT5_RUN, STEP, ft5_dev, ("s0",))
    write_selection(FT5_SELECTION, FT5_RUN, [ft5], ("s0",))
    # FT5 trained to its end: scripts/train.py's result (much else beside the steps), the last checkpoint STEP
    Path(FT5_RESULT).write_text(json.dumps({"mode": "think", "steps": STEP, "stopped_at_step": None}),
                                encoding="utf-8")
    ft3 = [write_checkpoint(FT3_RUN, FT3_OTHER, (3, False), ("s0", "s1")),
           write_checkpoint(FT3_RUN, FT3_STEP, ft3_dev, ("s0", "s1"))]
    write_selection(FT3_SELECTION, FT3_RUN, ft3, ("s0", "s1"))
    write_dev_a("ft3", ft3_a, adapter=adapter_of(ft3[1]), lock=lock)
    write_dev_a("ft5", ft5_a, adapter=adapter_of(ft5), lock=lock)
    return ["--step", str(STEP), "--selection", FT5_SELECTION, "--out", OUT]


def edit(path: str | Path, change) -> None:
    """Rewrite a JSON file after change(value) has changed it in place."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    change(value)
    Path(path).write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def drop_first_record(path: str | Path) -> None:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    Path(path).write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")


def result() -> dict:
    return json.loads(Path(OUT).read_text(encoding="utf-8"))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # the guards look here alone: the repository root they look under (with the current folder) and the
    # held-out roots of the dev set A runner (the repository and its run copy); and a check made without
    # --out goes here, not to the repository's results/
    monkeypatch.setattr(check, "REPO", tmp_path)
    monkeypatch.setattr(check, "OUT", tmp_path / "results" / "revision-check-m8.json")
    monkeypatch.setattr(check.runner, "REPO", tmp_path)
    monkeypatch.setattr(check.runner, "HELD_OUT_ROOTS", (tmp_path / "held-root",))
    return tmp_path


def expected_inputs() -> dict:
    """The inputs the check records, in its order: the two selection files, FT5's training result, each dev
    run's records and manifest, each dev set A run's records, manifest and summary."""
    files = {"selection": FT5_SELECTION, "baseline": FT3_SELECTION, "run_result": FT5_RESULT,
             "dev_ft3_records": dev_records(FT3_RUN, FT3_STEP),
             "dev_ft3_manifest": dev_manifest(FT3_RUN, FT3_STEP).as_posix(),
             "dev_ft5_records": dev_records(FT5_RUN, STEP),
             "dev_ft5_manifest": dev_manifest(FT5_RUN, STEP).as_posix(),
             **{f"dev_a_{label}_{kind}": path.as_posix() for label in ("ft3", "ft5")
                for kind, path in zip(("records", "manifest", "summary"), dev_a_files(label), strict=True)}}
    return {name: {"path": path, "sha256": sha256_of(path)} for name, path in files.items()}


def fired(out: dict) -> dict[str, bool]:
    return {name: trigger["fired"] for name, trigger in out["triggers"].items()}


def checkpoint_of(label: str) -> dict:
    """A system's checkpoint as the check records it: its GGUF, by the path and SHA-256 of its selection
    file's row (FT3's step 1635 in the baseline, FT5's last step in the selection)."""
    path = gguf_path(FT3_RUN, FT3_STEP) if label == "ft3" else gguf_path(FT5_RUN, STEP)
    return {"gguf": path.as_posix(), "gguf_sha256": sha256_of(path)}


def test_no_trigger_at_its_edge_writes_the_check_and_exits_0(repo, capsys):
    assert check.main(world()) == 0
    printed = capsys.readouterr().out
    out = result()
    assert list(out) == ["about", "rules", "step", "baseline_step", "inputs", "dev_set_a", "ft3", "ft5",
                         "triggers", "triggered"]
    assert out["step"] == STEP and out["baseline_step"] == FT3_STEP
    assert out["triggered"] is False and list(out["rules"]) == list(TRIGGERS)
    # FT3 is step 1635 of the baseline (10 successes; step 1308 has 7), FT5 the selection's step 448
    assert out["ft3"] == {**checkpoint_of("ft3"), "dev_s0": 10, "special_n": 15, "special_strict": 12,
                          "valid_n": 10, "false_refusals": 1}
    assert out["ft5"] == {**checkpoint_of("ft5"), "dev_s0": 3, "special_n": 15, "special_strict": 9,
                          "valid_n": 10, "false_refusals": 3}
    assert checkpoint_of("ft5")["gguf_sha256"] != checkpoint_of("ft3")["gguf_sha256"]
    assert out["triggers"] == {
        "R1": {"value": "3/5", "value_float": 0.6, "threshold": "3/5", "threshold_float": 0.6,
               "fired": False},
        "R2": {"value": 2, "threshold": 3, "fired": False},
        "R3": {"value": 7, "threshold": 8, "fired": False}}
    assert out["dev_set_a"] == {"path": DEV_A, "sha256": sha256_of(DEV_A)}
    assert out["inputs"] == expected_inputs() and list(out["inputs"]) == list(expected_inputs())
    raw = Path(OUT).read_bytes()
    assert raw.endswith(b"}\n") and b"\r" not in raw
    assert raw.decode("utf-8") == json.dumps(out, ensure_ascii=False, indent=2) + "\n"  # no escapes, 2 spaces
    lines = printed.splitlines()
    assert any(line.startswith("R1 ") and "9/15" in line and "not fired" in line for line in lines), printed
    assert any(line.startswith("R2 ") and "3 - FT3 1 = 2" in line for line in lines), printed
    assert any(line.startswith("R3 ") and "10 - FT5 3 = 7" in line for line in lines), printed
    assert "triggered: no" in printed and OUT in printed


@pytest.mark.parametrize(("change", "trigger", "value"), [
    # one special item below 3/5, strict; lenient it is 11/15, and with the 5 priority items 13/20, >= 3/5
    ({"ft5_a": (8, 3)}, "R1", "8/15"),
    ({"ft5_a": (9, 4)}, "R2", 3),  # FT3's false refusals + 3
    ({"ft5_dev": (8, False)}, "R3", 8),  # FT3's successes - 8
], ids=["R1-one-item-below-3-of-5", "R2-plus-3", "R3-minus-8"])
def test_one_trigger_fired_writes_the_check_and_exits_3(repo, capsys, change, trigger, value):
    assert check.main(world(**change)) == 3
    out = result()
    assert fired(out) == {name: name == trigger for name in TRIGGERS}
    assert out["triggers"][trigger]["value"] == value and out["triggered"] is True
    assert out["ft5"]["special_n"] == 15  # the special kinds of both languages, the priority items left out
    printed = capsys.readouterr().out
    assert "triggered: yes" in printed and trigger in printed.split("triggered: yes")[-1]


def test_every_trigger_fired_is_recorded(repo, capsys):
    assert check.main(world(ft5_a=(0, 10), ft5_dev=(8, False))) == 3  # every search wrong: 2 successes
    out = result()
    assert fired(out) == dict.fromkeys(TRIGGERS, True)
    assert (out["triggers"]["R1"]["value"], out["triggers"]["R1"]["value_float"]) == ("0/1", 0.0)
    assert (out["triggers"]["R2"]["value"], out["triggers"]["R3"]["value"]) == (9, 8)
    assert "triggered: yes (R1, R2, R3)" in capsys.readouterr().out


@pytest.mark.parametrize(("change", "trigger", "value"), [
    ({"ft3_a": (12, 4), "ft5_a": (12, 1)}, "R2", -3),  # FT5 with 3 false refusals fewer than FT3
    ({"ft3_dev": (8, False), "ft5_dev": (0, False)}, "R3", -8),  # FT5 with 8 dev successes more: 2 and 10
], ids=["R2-ft5-3-fewer", "R3-ft5-8-more"])
def test_ft5_better_than_ft3_by_a_triggers_size_fires_nothing(repo, capsys, change, trigger, value):
    # R2 and R3 are differences, FT5's worse direction counted: never their size
    assert check.main(world(**change)) == 0
    out = result()
    assert out["triggers"][trigger]["value"] == value and fired(out) == dict.fromkeys(TRIGGERS, False)
    assert out["triggered"] is False and "triggered: no" in capsys.readouterr().out


def test_r3_counts_the_successes_with_the_runtime_rules_applied_offline(repo, capsys):
    # FT5's busy notice after the rate limit becomes the retry once the rules apply: 3 successes with the
    # rules, 2 without them; FT3's 10 less 3 is 7, below R3's 8
    argv = world(ft5_dev=(7, True))
    items = [json.loads(line) for line in Path(DEV).read_text(encoding="utf-8").splitlines()]
    assert check.decide_m8.successes(dev_records(FT5_RUN, STEP), items) == (3, 2)
    assert check.main(argv) == 0
    out = result()
    assert out["ft5"]["dev_s0"] == 3 and out["triggers"]["R3"] == {"value": 7, "threshold": 8, "fired": False}


def test_an_existing_check_is_never_overwritten_and_nothing_is_read_then(repo, capsys):
    Path(OUT).parent.mkdir(parents=True)
    Path(OUT).write_text("{}\n", encoding="utf-8")
    assert check.main(["--step", str(STEP), "--selection", "runs/none.json", "--out", OUT]) == 2
    printed = capsys.readouterr().out
    assert f"{OUT} exists" in printed and "nothing was read" in printed
    assert Path(OUT).read_text(encoding="utf-8") == "{}\n"
    argv = world()
    assert check.main(argv) == 2  # the inputs there now: still refused, still not overwritten
    assert Path(OUT).read_text(encoding="utf-8") == "{}\n"


def test_a_check_written_while_this_one_ran_is_never_overwritten(repo, capsys, monkeypatch):
    argv = world()
    read_inputs = check.read_inputs

    def meanwhile(args):  # another check writes --out after this one found none there, while it reads
        inputs = read_inputs(args)
        Path(OUT).parent.mkdir(parents=True, exist_ok=True)
        Path(OUT).write_text("{}\n", encoding="utf-8")
        return inputs

    monkeypatch.setattr(check, "read_inputs", meanwhile)
    assert check.main(argv) == 2
    assert f"{OUT} was written while the check ran" in capsys.readouterr().out
    assert Path(OUT).read_text(encoding="utf-8") == "{}\n"


def test_the_defaults_and_another_place_of_ft5s_dev_set_a_run(repo, capsys):
    args = check.build_parser().parse_args(["--step", "7", "--selection", "s.json"])
    assert (args.baseline, args.baseline_step) == (Path(FT3_SELECTION), FT3_STEP)
    assert args.run == Path(FT5_RUN)
    assert args.dev_a_ft3 == Path("runs/m8/special-dev/a-ft3.jsonl") and args.dev_a_ft5 is None
    # the default --out is results/ of the script's repository; in these tests the fixture moves it here
    script_repo = Path(check.__file__).resolve().parents[1]
    assert SCRIPT_OUT == script_repo / "results" / "revision-check-m8.json"
    assert args.out == check.OUT == repo / "results" / "revision-check-m8.json"
    with pytest.raises(SystemExit) as stopped:  # no --step: a usage error
        check.main(["--selection", FT5_SELECTION, "--out", OUT])
    assert stopped.value.code == 2
    argv = world()
    # FT5's dev set A run made under another name: --dev-a-ft5 names it, and the default is not read; --run
    # spelled otherwise (absolute, the system's separators) is still the selection's run
    records, manifest, summary = dev_a_files("ft5")
    other = records.with_name("a-ft5-again.jsonl")
    records.replace(other)
    manifest.replace(other.with_name("a-ft5-again.manifest.json"))
    edit(summary, lambda s: s.update(records=other.name))
    summary.replace(other.with_name("a-ft5-again.summary.json"))
    assert check.main([*argv, "--dev-a-ft5", other.as_posix(), "--run", str(repo / Path(FT5_RUN))]) == 0
    assert result()["inputs"]["dev_a_ft5_records"] == {"path": other.as_posix(), "sha256": sha256_of(other)}
    Path(OUT).unlink()
    capsys.readouterr()
    assert check.main(argv) == 2  # by default runs/m8/special-dev/a-ft5-448.jsonl, which is not there now
    assert f"a-ft5-{STEP}" in capsys.readouterr().out and not Path(OUT).exists()


def test_a_step_the_selection_or_the_baseline_does_not_hold_is_refused(repo, capsys):
    argv = world()
    assert check.main([*argv, "--step", "999"]) == 2
    printed = capsys.readouterr().out
    assert "holds no step 999" in printed and FT5_SELECTION in printed and not Path(OUT).exists()
    assert check.main([*argv, "--baseline-step", "1962"]) == 2
    printed = capsys.readouterr().out
    assert "holds no step 1962" in printed and FT3_SELECTION in printed and not Path(OUT).exists()


# A checkpoint that is not FT5's last as FT5 is now, each world otherwise one the check would judge (exit 0)
def _another_run(argv: list[str]) -> list[str]:  # a selection of another FT5 attempt than --run's
    edit(FT5_SELECTION, lambda s: s.update(run="runs/train/ft5-first"))
    return argv


def _ft3s_selection_as_ft5s(argv: list[str]) -> list[str]:  # FT3's selection, step and dev set A run as FT5's
    return ["--step", str(FT3_STEP), "--selection", FT3_SELECTION, "--out", OUT,
            "--dev-a-ft5", dev_a_files("ft3")[0].as_posix()]


def _the_baselines_run(argv: list[str]) -> list[str]:  # --run names FT3's run, and the selection says it too
    edit(FT5_SELECTION, lambda s: s.update(run=FT3_RUN))
    return [*argv, "--run", FT3_RUN]


def _an_earlier_checkpoint(argv: list[str]) -> list[str]:
    """FT5's 3/4 checkpoint asked for, every run of it made: the selection holds it beside the last one, and
    FT5's dev set A run is of it."""
    earlier = write_checkpoint(FT5_RUN, EARLIER, (0, False), ("s0",))
    rows = json.loads(Path(FT5_SELECTION).read_text(encoding="utf-8"))["checkpoints"]
    write_selection(FT5_SELECTION, FT5_RUN, [earlier, *rows], ("s0",))
    edit(dev_a_files("ft5")[1], lambda m: m.update(adapters=[adapter_of(earlier)]))
    return [*argv, "--step", str(EARLIER), "--dev-a-ft5", dev_a_files("ft5")[0].as_posix()]


def _not_trained_to_its_end(argv: list[str]) -> list[str]:  # FT5's run without its training result
    Path(FT5_RESULT).unlink()
    return argv


def _ft3s_bytes(argv: list[str]) -> list[str]:
    """FT5's last GGUF holds FT3 step 1635's bytes, the selection records their SHA-256, and every FT5 run
    applied it."""
    ft3 = gguf_path(FT3_RUN, FT3_STEP)
    digest = sha256_of(ft3)
    gguf_path(FT5_RUN, STEP).write_bytes(ft3.read_bytes())
    edit(FT5_SELECTION, lambda s: s["checkpoints"][0].update(sha256=digest))
    for manifest in (dev_manifest(FT5_RUN, STEP), dev_a_files("ft5")[1]):
        edit(manifest, lambda m: m["adapters"][0].update(sha256=digest))
    return argv


def _trained_again(argv: list[str]) -> list[str]:  # FT5 trained and exported again since its runs were made
    gguf_path(FT5_RUN, STEP).write_bytes(b"GGUF runs/train/ft5 448, trained again")
    return argv


def _gguf_gone(argv: list[str]) -> list[str]:
    gguf_path(FT5_RUN, STEP).unlink()
    return argv


# per refusal: how the arguments or the files are changed, and what the refusal must name
NOT_THE_LAST = {
    "a-selection-of-another-run": (_another_run, ("runs/train/ft5-first", f"{FT5_RUN} (--run)")),
    "ft3s-selection-and-runs-as-ft5s": (_ft3s_selection_as_ft5s, (f"'{FT3_RUN}'", f"{FT5_RUN} (--run)")),
    "the-baselines-run-as-ft5s": (_the_baselines_run, (f"--run {FT3_RUN}", FT3_SELECTION)),
    "an-earlier-checkpoint": (_an_earlier_checkpoint, (f"--step {EARLIER}", FT5_RESULT, f"{STEP} steps")),
    "not-trained-to-its-end": (_not_trained_to_its_end, (FT5_RESULT,)),
    "ft5s-gguf-of-ft3s-bytes": (_ft3s_bytes, (f"FT3 step {FT3_STEP}", FT5_SELECTION)),
    "ft5-trained-again-since-its-runs": (_trained_again, (gguf_path(FT5_RUN, STEP).as_posix(), "SHA-256")),
    "ft5s-gguf-gone": (_gguf_gone, (gguf_path(FT5_RUN, STEP).as_posix(),)),
}


@pytest.mark.parametrize("case", list(NOT_THE_LAST))
def test_a_checkpoint_that_is_not_ft5s_last_as_ft5_is_now_is_refused(repo, capsys, case):
    change, named = NOT_THE_LAST[case]
    assert check.main(change(world())) == 2
    printed = capsys.readouterr().out
    assert all(name in printed for name in named), printed
    assert not Path(OUT).exists()


@pytest.mark.parametrize(("run", "step", "change", "named"), [
    (FT5_RUN, STEP, lambda m: m["adapters"][0].update(sha256="0" * 64), "adapters"),  # another GGUF
    (FT5_RUN, STEP, lambda m: m.update(items=DEV_V2), "items"),  # the run over dev-v2
    (FT5_RUN, STEP, lambda m: m.update(items_sha256="0" * 64), "items"),  # over another dev
    (FT5_RUN, STEP, lambda m: m.update(items_run=9), "items_run"),  # not every item run
    (FT3_RUN, FT3_STEP, lambda m: m["adapters"][0].update(sha256="0" * 64), "adapters"),  # not step 1635's
    (FT3_RUN, FT3_STEP, lambda m: m["adapters"].append({"id": 1, "path": "x.gguf", "sha256": "1" * 64}),
     "adapters"),  # two adapters
], ids=["ft5-another-gguf", "ft5-another-set", "ft5-another-sha256", "ft5-not-every-item", "ft3-another-gguf",
        "ft3-two-adapters"])
def test_a_dev_run_manifest_not_of_its_checkpoint_and_the_set_is_refused(repo, capsys, run, step, change,
                                                                          named):
    argv = world()
    target = dev_manifest(run, step)
    edit(target, change)
    assert check.main(argv) == 2
    printed = capsys.readouterr().out
    assert named in printed and target.as_posix() in printed and not Path(OUT).exists()


@pytest.mark.parametrize(("field", "change"), [
    ("generation", lambda m: m["generation"].update(reasoning_budget=1024)),
    ("generation", lambda m: m["generation"].update(base_seed=1)),  # seed 1's run
    ("family", lambda m: m["family"].update(template_sha256="0" * 64)),
    ("family", lambda m: m["family"].update(name="other-family")),
    ("prompt_hash", lambda m: m.update(prompt_hash="0" * 16)),  # another agent prompt
    ("prompt_version", lambda m: m.update(prompt_version="m8-2026-11-02")),
    ("schema_hash", lambda m: m.update(schema_hash="0" * 16)),  # other tool schemas
    ("lora_scale", lambda m: m.update(lora_scale=0.5)),
    ("server_props.model_path", lambda m: m["server_props"].update(model_path="models/Other-Q8_0.gguf")),
], ids=["budget", "seed", "template", "family-name", "prompt-hash", "prompt-version", "schema-hash",
        "lora-scale", "model-path"])
def test_an_ft5_dev_run_measured_otherwise_than_ft3s_is_refused_by_its_field(repo, capsys, field, change):
    argv = world()
    target = dev_manifest(FT5_RUN, STEP)
    edit(target, change)
    assert check.main(argv) == 2
    printed = capsys.readouterr().out
    assert field in printed and target.as_posix() in printed and "FT3's" in printed and not Path(OUT).exists()


def _relocked_dev() -> None:  # dev rebuilt since the runs, without its last item, and locked again
    rows = [json.loads(line) for line in Path(DEV).read_text(encoding="utf-8").splitlines()]
    write_jsonl(rows[:-1], DEV)
    write_lock(DEV)


def _changed_dev() -> None:  # dev changed since the runs, without its last item, its lock as it was
    rows = [json.loads(line) for line in Path(DEV).read_text(encoding="utf-8").splitlines()]
    write_jsonl(rows[:-1], DEV)


@pytest.mark.parametrize(("change", "code", "named"), [
    (_relocked_dev, 2, (FT5_SELECTION, DEV, "changed")),
    (_changed_dev, 5, ("dev.jsonl", "does not match its lock")),  # as dev set A in that state
    (lambda: edit(FT3_SELECTION, lambda s: s["item_sets"][0].update(sha256="0" * 64)), 2,
     (FT3_SELECTION, DEV)),
    (lambda: edit(FT5_SELECTION, lambda s: s.update(item_sets=s["item_sets"][1:])), 2,
     (FT5_SELECTION, "item_sets")),
    (lambda: Path(DEV).unlink(), 2, (DEV,)),
    (lambda: Path(DEV + ".sha256").unlink(), 5, ("dev.jsonl", "no lock")),
    (lambda: Path(DEV + ".sha256").write_text("0" * 64 + "  dev.jsonl\n", encoding="utf-8"), 5,
     ("dev.jsonl", "does not match")),
], ids=["dev-rebuilt-since-the-runs", "dev-changed-since-its-lock", "baseline-of-another-dev",
        "selection-without-dev", "dev-gone", "dev-lock-gone", "dev-off-its-lock"])
def test_dev_must_be_the_locked_set_both_selections_ran_on(repo, capsys, change, code, named):
    argv = world()
    change()
    assert check.main(argv) == code
    printed = capsys.readouterr().out
    assert all(name in printed for name in named), printed
    assert not Path(OUT).exists()


def _swap_adapter(label: str, other: str):
    def swap() -> None:
        adapters = json.loads(dev_a_files(other)[1].read_text(encoding="utf-8"))["adapters"]
        edit(dev_a_files(label)[1], lambda m: m.update(adapters=adapters))
    return swap


def _manifest(label: str, change):
    return lambda: edit(dev_a_files(label)[1], change)


def _summary(label: str, change):
    return lambda: edit(dev_a_files(label)[2], change)


def _both(change):
    return lambda: [edit(dev_a_files(label)[1], change) for label in ("ft3", "ft5")]


DEV_SET_B = "data/eval/special-dev-b.jsonl"
# per refusal: what is changed after the world is written, and what the refusal must name
DEV_A_RUNS = {
    "ft3-on-ft5s-gguf": (_swap_adapter("ft3", "ft5"), ("ft3", "adapters")),
    "ft5-on-ft3s-gguf": (_swap_adapter("ft5", "ft3"), ("ft5", "adapters")),
    "ft5-without-an-adapter": (_manifest("ft5", lambda m: m.update(adapters=[], lora_scale=0.0)),
                               ("ft5", "adapters", "lora_scale")),
    "ft3-not-applied": (_manifest("ft3", lambda m: m.update(lora_scale=0.0)), ("ft3", "lora_scale")),
    "summary-of-another-lock": (_summary("ft5", lambda s: s.update(items_sha256="c" * 64)),
                                ("ft5", "summary items_sha256")),
    "manifest-of-another-lock": (_manifest("ft3", lambda m: m.update(items_sha256="c" * 64)),
                                 ("ft3", "manifest items_sha256")),
    "manifest-of-dev-set-b": (_manifest("ft5", lambda m: m.update(items=DEV_SET_B)),
                              ("ft5", "special-dev-b")),
    "summary-of-dev-set-b": (_summary("ft3", lambda s: s.update(items=DEV_SET_B)), ("ft3", "special-dev-b")),
    "not-every-item-run": (_manifest("ft5", lambda m: m.update(items_run=20)), ("ft5", "items_run")),
    "error-records": (_manifest("ft5", lambda m: m.update(statuses={"error": 1, "ok": DEV_A_ITEMS - 1})),
                      ("ft5", "statuses")),
    "a-context-overflow": (_manifest("ft3", lambda m: m.update(
        statuses={"context_overflow": 1, "ok": DEV_A_ITEMS - 1})), ("ft3", "statuses")),
    "summary-with-scoring-errors": (_summary("ft5", lambda s: s["summary"].update(scoring_errors=2)),
                                    ("ft5", "scoring_errors")),
    "four-slots": (_manifest("ft5", lambda m: m["server_props"].update(total_slots=4)),
                   ("ft5", "server_props.total_slots")),
    "not-think": (_both(lambda m: m["generation"].update(mode="no_think")), ("generation.mode",)),
    "not-greedy": (_both(lambda m: m["generation"]["decoding"].update(temperature=0.6, top_p=0.95)),
                   ("generation.decoding",)),
    "another-budget": (_both(lambda m: m["generation"].update(reasoning_budget=1024)),
                       ("generation.reasoning_budget",)),
    "another-generation": (_manifest("ft5", lambda m: m["generation"].update(answer_tokens=1000)),
                           ("generation", "ft5")),
    "another-base-model": (_manifest("ft5", lambda m: m["server_props"].update(
        model_path="models/Other-Q8_0.gguf")), ("server_props.model_path",)),
    "another-build": (_manifest("ft3", lambda m: m["server_props"].update(build_info="b9999-0000000")),
                      ("server_props.build_info",)),
    "another-template": (_manifest("ft5", lambda m: m["family"].update(template_sha256="0" * 64)),
                         ("family.template_sha256",)),
    "another-family": (_manifest("ft5", lambda m: m["family"].update(name="other-family")), ("family.name",)),
    "base-not-recorded": (_both(lambda m: m["server_props"].update(model_path=None)),
                          ("server_props.model_path",)),
    "summary-of-other-records": (_summary("ft5", lambda s: s.update(records="a-other.jsonl")),
                                 ("ft5", "a-other.jsonl")),
    "no-manifest": (lambda: dev_a_files("ft5")[1].unlink(), (f"a-ft5-{STEP}.manifest.json",)),
    "no-summary": (lambda: dev_a_files("ft3")[2].unlink(), ("a-ft3.summary.json",)),
    "no-records": (lambda: dev_a_files("ft5")[0].unlink(), (f"a-ft5-{STEP}.jsonl",)),
}


@pytest.mark.parametrize("case", list(DEV_A_RUNS))
def test_a_dev_set_a_run_that_is_not_the_one_its_label_needs_is_refused_by_its_field(repo, capsys, case):
    argv = world()
    change, named = DEV_A_RUNS[case]
    change()
    assert check.main(argv) == 2
    printed = capsys.readouterr().out
    assert all(name in printed for name in named), printed
    assert not Path(OUT).exists()


def test_dev_set_a_must_still_be_the_locked_set_the_runs_were_made_on(repo, capsys):
    argv = world()
    with Path(DEV_A).open("a", encoding="utf-8", newline="\n") as f:  # one more item, well formed
        f.write(json.dumps({**DEV_A_SET[-1][0], "id": "sdev-a-9999"}, ensure_ascii=False) + "\n")
    assert check.main(argv) == 5  # changed since its lock
    assert "special-dev-a.jsonl" in capsys.readouterr().out and not Path(OUT).exists()
    write_lock(DEV_A)  # rebuilt and locked again since the runs: their lock's SHA-256 is another
    assert check.main(argv) == 2
    printed = capsys.readouterr().out
    assert "items_sha256" in printed and "lock" in printed and not Path(OUT).exists()


@pytest.mark.parametrize(("change", "code", "named"), [
    (lambda: Path(DEV_A).unlink(), 2, (DEV_A,)),
    (lambda: Path(DEV_A + ".sha256").unlink(), 5, ("special-dev-a.jsonl", "no lock", "as it was locked")),
], ids=["dev-set-a-gone", "dev-set-a-lock-gone"])
def test_dev_set_a_must_be_there_with_its_lock(repo, capsys, change, code, named):
    argv = world()
    change()
    assert check.main(argv) == code
    printed = capsys.readouterr().out
    assert all(name in printed for name in named), printed
    assert not Path(OUT).exists()


def _record_failed(label: str) -> None:  # the server failed on the first item: an error record in its place
    records = dev_a_files(label)[0]
    rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
    rows[0] = {"id": rows[0]["id"], "status": "error", "error": "the server failed",
               "error_type": "HTTPError"}
    write_jsonl(rows, records)


def test_dev_set_a_records_not_all_ok_are_refused(repo, capsys):
    argv = world()
    _record_failed("ft5")  # the manifest still says every one ok
    assert check.main(argv) == 2
    printed = capsys.readouterr().out
    assert "ft5" in printed and "'error': 1" in printed and "every one ok" in printed
    assert not Path(OUT).exists()


@pytest.mark.parametrize(("label", "name", "value"), [
    ("ft5", "special.strict", 12),  # R1 would read 12/15
    ("ft3", "valid.false_refusals", 0),  # R2's reference
    ("ft5", "special.n", 20),
    ("ft3", "special.lenient", 0),
    ("ft5", "valid.exact", 10),
], ids=["ft5-strict", "ft3-false-refusals", "ft5-special-n", "ft3-lenient", "ft5-exact"])
def test_a_summary_whose_counts_are_not_its_records_scored_again_is_refused(repo, capsys, label, name, value):
    argv = world()
    group, key = name.split(".")
    edit(dev_a_files(label)[2], lambda s: s["summary"][group].update({key: value}))
    assert check.main(argv) == 2
    printed = capsys.readouterr().out
    assert name in printed and label in printed and "scored again" in printed, printed
    assert not Path(OUT).exists()


def test_runs_the_scorer_counts_otherwise_now_than_their_summaries_are_refused(repo, capsys, monkeypatch):
    # the scorer revised since the runs: it now judges a refusal to a missing item otherwise, so both
    # summaries are of another scorer than the one the triggers would read
    argv = world()
    real = special_dev.score_item

    def revised(answer, item, **kwargs):
        row = real(answer, item, **kwargs)
        return {**row, "lenient": False, "strict": False} if item["kind"] == "missing" else row

    monkeypatch.setattr(special_dev, "score_item", revised)
    assert check.main(argv) == 2
    printed = capsys.readouterr().out
    assert "ft3" in printed and "special.strict" in printed and "scored again" in printed
    assert not Path(OUT).exists()


def test_records_the_scorer_fails_on_now_are_refused(repo, capsys, monkeypatch):
    argv = world()
    real = special_dev.score_item

    def failing(answer, item, **kwargs):  # the scorer as it is now fails on the first item
        if item["id"] == "sdev-a-0001":
            raise ValueError("cannot score this")
        return real(answer, item, **kwargs)

    monkeypatch.setattr(special_dev, "score_item", failing)
    assert check.main(argv) == 2
    printed = capsys.readouterr().out
    assert "scoring_errors 1" in printed and "ft3" in printed and not Path(OUT).exists()


def _malformed_dev_a_item() -> None:
    rows = [row for row, _ in DEV_A_SET]
    write_jsonl([{k: v for k, v in rows[0].items() if k != "family"}, *rows[1:]], DEV_A)
    write_lock(DEV_A)


def _dev_with_a_repeated_id() -> None:  # dev holding its first item twice, locked, both selections ran on it
    write_jsonl([*DEV_ITEMS, DEV_ITEMS[0]], DEV)
    write_lock(DEV)
    for path in (FT5_SELECTION, FT3_SELECTION):
        edit(path, lambda s: s["item_sets"][0].update(sha256=sha256_of(DEV)))


# per malformed input: what is changed after the world is written, and what the refusal must name
MALFORMED = {
    "selection-not-json": (lambda: Path(FT5_SELECTION).write_text("{", encoding="utf-8"), FT5_SELECTION),
    "baseline-without-checkpoints": (lambda: edit(FT3_SELECTION, lambda s: s.update(checkpoints=[])),
                                     FT3_SELECTION),
    "row-without-s0-over-dev": (lambda: edit(FT5_SELECTION, lambda s: s["checkpoints"][0]["variants"]["s0"][
        "sets"].pop(DEV)), FT5_SELECTION),
    "row-step-not-a-number": (lambda: edit(FT3_SELECTION, lambda s: s["checkpoints"][0].update(step="1308")),
                              FT3_SELECTION),
    "selection-listing-a-step-twice": (lambda: edit(FT5_SELECTION, lambda s: s.update(
        checkpoints=s["checkpoints"] * 2)), f"{FT5_SELECTION} lists step {STEP} twice"),
    "training-result-without-its-steps": (lambda: edit(FT5_RESULT, lambda r: r.pop("steps")), FT5_RESULT),
    "dev-with-a-repeated-id": (_dev_with_a_repeated_id, f"the first {DEV_ITEMS[0]['id']!r}"),
    "dev-run-without-a-record-of-every-item": (lambda: drop_first_record(dev_records(FT5_RUN, STEP)),
                                               dev_records(FT5_RUN, STEP)),
    "dev-set-a-run-without-a-record-of-every-item": (lambda: drop_first_record(dev_a_files("ft3")[0]),
                                                     "a-ft3.jsonl"),
    "summary-without-its-counts": (lambda: edit(dev_a_files("ft5")[2], lambda s: s.update(
        summary={"special": {"n": 15}})), f"a-ft5-{STEP}.summary.json"),
    "a-malformed-item-of-dev-set-a": (_malformed_dev_a_item, "family"),
}


@pytest.mark.parametrize("case", list(MALFORMED))
def test_a_malformed_input_is_refused_as_malformed(repo, capsys, case):
    argv = world()
    change, named = MALFORMED[case]
    change()
    assert check.main(argv) == 1
    printed = capsys.readouterr().out
    assert named in printed, printed
    assert not Path(OUT).exists()


def _with(argv: list[str], option: str, value: str) -> list[str]:
    """argv with option given value (after the one there, which argparse lets it override)."""
    return [*argv, option, value]


def _held_out_dev_run(argv: list[str]) -> list[str]:  # the selection names records in the held-out folder
    edit(FT5_SELECTION, lambda s: s["checkpoints"][0]["variants"]["s0"]["sets"][DEV].update(
        records=f"held-root/runs/m7/heldout/checkpoint-{STEP}.s0.jsonl"))
    return argv


def _dev_set_a_of_the_test_split(argv: list[str]) -> list[str]:
    rows = [row for row, _ in DEV_A_SET]
    write_jsonl([{**rows[0], "split": "test"}, *rows[1:]], DEV_A)
    write_lock(DEV_A)
    return argv


def _dev_set_a_a_held_out_copy(argv: list[str]) -> list[str]:  # dev set A's bytes are a held-out set's
    held = Path("held-root/data/eval/dev-ext.jsonl")
    held.parent.mkdir(parents=True)
    held.write_bytes(Path(DEV_A).read_bytes())
    return argv


def _dev_of_the_test_split(argv: list[str]) -> list[str]:  # both selections ran on it as it is now
    rows = [json.loads(line) for line in Path(DEV).read_text(encoding="utf-8").splitlines()]
    write_jsonl([{**rows[0], "split": "test"}, *rows[1:]], DEV)
    write_lock(DEV)
    for path in (FT5_SELECTION, FT3_SELECTION):
        edit(path, lambda s: s["item_sets"][0].update(sha256=sha256_of(DEV)))
    return argv


def _dev_the_bytes_of_a_locked_test_set(argv: list[str]) -> list[str]:
    """A locked test set (the test's folder's data/eval/test.jsonl and its lock) holding dev's very bytes:
    items no content check refuses (split dev, ids of the dev split), so dev is refused by its bytes alone,
    and before it is compared with any run (the selection's item_sets, stale here, are not reached)."""
    shutil.copyfile(DEV, "data/eval/test.jsonl")
    write_lock("data/eval/test.jsonl")
    edit(FT5_SELECTION, lambda s: s["item_sets"][0].update(sha256="0" * 64))
    return argv


def _run_in_the_held_out_folder(argv: list[str]) -> list[str]:  # FT5's run given, and selected, there
    run = "held-root/runs/m7/heldout/ft5"
    edit(FT5_SELECTION, lambda s: s.update(run=run))
    return _with(argv, "--run", run)


def _gguf_at_a_test_set(argv: list[str]) -> list[str]:  # the selection names a test set as FT5's GGUF
    edit(FT5_SELECTION, lambda s: s["checkpoints"][0].update(gguf="data/eval/test.jsonl"))
    return argv


# per refusal: how the arguments or the files are changed, and what the refusal must name
NOT_READ = {
    "selection-at-a-test-set": (lambda argv: _with(argv, "--selection", "data/eval/test.jsonl"),
                                "locked test set"),
    "baseline-at-the-handwritten-items": (lambda argv: _with(argv, "--baseline",
                                                             "data/eval/handwritten.jsonl"), "handwritten"),
    "dev-set-a-run-in-the-held-out-folder": (lambda argv: _with(
        argv, "--dev-a-ft5", "held-root/runs/m7/heldout/a-ft5.jsonl"), "held-out set (found by its folder)"),
    "dev-run-in-the-held-out-folder": (_held_out_dev_run, "held-out set (found by its folder)"),
    "dev-set-a-of-the-test-split": (_dev_set_a_of_the_test_split, "test material"),
    "dev-set-a-a-copy-of-a-held-out-set": (_dev_set_a_a_held_out_copy, "held-out set (found by its content)"),
    "dev-of-the-test-split": (_dev_of_the_test_split, "test material"),
    "dev-the-bytes-of-a-locked-test-set": (_dev_the_bytes_of_a_locked_test_set,
                                           f"{DEV} has the SHA-256 of the locked test set"),
    "training-result-in-the-held-out-folder": (_run_in_the_held_out_folder,
                                               "result.json is a held-out set (found by its folder)"),
    "ft5s-gguf-at-a-test-set": (_gguf_at_a_test_set, "ft5_gguf: data/eval/test.jsonl is the locked test set"),
}


@pytest.mark.parametrize("case", list(NOT_READ))
def test_test_material_and_held_out_sets_are_refused(repo, capsys, case):
    change, named = NOT_READ[case]
    assert check.main(change(world())) == 4
    printed = capsys.readouterr().out
    assert named in printed, printed
    assert not Path(OUT).exists()


# ---- with the real runner ---------------------------------------------------------------------------------
# Dev set A of three items, run for FT3 and FT5 by scripts/run_special_dev.py on tests/support's FakeLlama (a
# fake family's model folder): the check reads what the runner writes.

SERVER = "http://fake-llama.invalid"
SAMPLING = Decoding(temperature=0.6, top_p=0.95)  # the family's thinking decoding, which the runner overrides
CONTRACT_ITEMS = [
    dev_a_item(1, "incorrect", "Book room A-12 for -3 people on 2026-11-05.",
               refusal("incorrect", ("-3",), ("people",))),
    dev_a_item(2, "missing", "Plan a trip to Lisbon.",
               refusal("missing", params=("traveller",), api="plan_trip")),
    dev_a_item(3, "valid_verbatim", "Book room 会议室A for 4 people on 2026-11-05.",
               {"type": "call", "calls": [{"name": "book_room", "arguments": {
                   "room": "会议室A", "people": 4, "date": "2026-11-05"}}], "verbatim": ["会议室A"]}),
]
ANSWERED = ("Not sent: the value -3 of people breaks the definition.",
            "Not sent: plan_trip still needs traveller", booking("会议室A"))
CALLED = (booking("A-12", people=-3), 'plan_trip(destination="Lisbon")', booking("会议室A"))


def test_checks_what_the_real_runner_writes(repo, capsys, monkeypatch):
    rsd = load_script("run_special_dev")
    monkeypatch.setattr(rsd, "REPO", repo)  # its guards' root, as the check's
    monkeypatch.setattr(rsd, "HELD_OUT_ROOTS", (repo / "held-root",))
    monkeypatch.setattr(rsd, "git_commit", lambda _: "0" * 40)  # no git run on any folder
    monkeypatch.setattr(rsd, "git_dirty", lambda _: False)
    folder = repo / "model"
    install_fake_family(monkeypatch, folder, "only", inference_mode=Mode.THINK,
                        decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: SAMPLING})
    argv = world()  # the dev runs; dev set A and its runs are replaced below
    write_jsonl(CONTRACT_ITEMS, DEV_A)
    write_lock(DEV_A)
    ggufs = {"ft3": f"{FT3_RUN}/gguf/checkpoint-{FT3_STEP}-f16.gguf",
             "ft5": f"{FT5_RUN}/gguf/checkpoint-{STEP}-f16.gguf"}
    for label, answers in (("ft3", ANSWERED), ("ft5", CALLED)):  # FT5 calls where it should refuse
        fake = FakeLlama([Scripted(("Read the rules.", "\n</think>\n\n", answer)) for answer in answers],
                         adapters=[{"id": 0, "path": ggufs[label], "scale": 0.0}],
                         props={**PROPS, "model_path": BASE_GGUF})
        runner_argv = [DEV_A, "--system", "ft", "--tokenizer", str(folder), "--server", SERVER, "--out",
                       dev_a_files(label)[0].as_posix(), "--force"]
        assert rsd.main(runner_argv, local_client=fake.client(base_url=SERVER)) == 0
    capsys.readouterr()
    assert check.main(argv) == 3
    out = result()
    assert out["ft3"] == {**checkpoint_of("ft3"), "dev_s0": 10, "special_n": 2, "special_strict": 2,
                          "valid_n": 1, "false_refusals": 0}
    assert out["ft5"] == {**checkpoint_of("ft5"), "dev_s0": 3, "special_n": 2, "special_strict": 0,
                          "valid_n": 1, "false_refusals": 0}
    assert fired(out) == {"R1": True, "R2": False, "R3": False}
    assert out["dev_set_a"] == {"path": DEV_A, "sha256": sha256_of(DEV_A)}
    assert out["inputs"]["dev_a_ft5_summary"]["path"] == dev_a_files("ft5")[2].as_posix()
    assert "triggered: yes" in capsys.readouterr().out

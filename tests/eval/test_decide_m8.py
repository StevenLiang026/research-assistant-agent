"""scripts/decide_m8.py (docs/m8-计划.md WP4), offline, on fixture sets standing in for data/eval/dev.jsonl
and data/eval/v2/dev-v2.jsonl (in a temporary folder) with the selection runs of FT5's four checkpoints and
of FT3's step 1635 (seeds 0 and 1 over both sets, their run manifests beside them), a locked dev set B of
research_agent.eval.special_dev's items and its runs as scripts/run_special_dev.py writes them (records, run
manifest and summary, the summary the scorer's of the records) of B0, FT3 and the candidates, and the
calibration file naming B0's and FT3's runs with their summaries: the runtime rules applied to every record
offline, the normalized scorer, the ranking by the two-seed sum with a tie to the later step, the walk to the
first checkpoint passing PG1 to PG4 (dev set B needed only that far, exit 3 otherwise), MG against FT3's sum,
the PG1 fallback read from the calibration file, where every dev set B run comes from, every run's records
scored again (the counts the gates read), the ranking alone (--rank-only), and the refusal to decide twice.
The last tests run the real scripts/run_special_dev.py (on tests/support's FakeLlama) and
scripts/calibrate_special_dev.py before deciding. The held-out roots are moved to an empty folder, so no
held-out set is looked at; nothing here reads a real run."""

from __future__ import annotations

import importlib.util
import json
import sys
from fractions import Fraction
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


decide_m8 = load_script("decide_m8")

DEV, DEV_V2 = "data/eval/dev.jsonl", "data/eval/v2/dev-v2.jsonl"
DEV_B = "data/eval/special-dev-b.jsonl"
FT5_SELECTION = "results/checkpoint-selection-ft5.json"
FT3_SELECTION = "results/checkpoint-selection-ft3.json"
CALIBRATION = "runs/m8/special-dev-calibration.json"
OUT = "results/decision-m8.json"
STEPS = (112, 224, 336, 448)
FT3_STEP = 1635
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


# dev: eight searches, one search after a rate limit (the runtime rules retry it), one goodbye; dev-v2: four
# searches and a goodbye. 15 items a seed.
ITEMS = {DEV: [*(item(f"req-dev-{k:04d}-s1", user(f"Find hallucination papers ({k})."), FIND, "call")
                 for k in range(1, 9)),
               item("req-dev-0009-s1", FAILED, FIND, "recover"),
               item("req-dev-0010-s1", user("Thanks, that's all."), BYE, "no_call")],
         DEV_V2: [*(item(f"req-dv2-{k:04d}-s1", user(f"Search hallucination papers ({k})."), FIND, "call")
                    for k in range(1, 5)),
                  item("req-dv2-0005-s1", user("That's all for now, thanks."), BYE, "no_call")]}
RIGHT, CATS = ("", (call(),)), ("", (call({"query": "cats", "max_results": 5}),))
BUSY, GOODBYE = ("The service is busy, try again later.", ()), ("See you!", ())
GENERATION = {"answer_tokens": 512, "base_seed": 0, "mode": "think", "reasoning_budget": 1280, "sample": 0,
              "decoding": {"min_p": 0.0, "presence_penalty": 0.0, "temperature": 0.6, "top_k": 0,
                           "top_p": 0.95}}
FAMILY = {"name": "nanbeige4.1-3b", "thinking": "only", "template_sha256": "7" * 64}
BASE_GGUF = "models/Base-Q8_0.gguf"
# what else a run manifest of scripts/run_eval.py records that makes a sum FT3's metric (FT3-1635's values)
MEASURED = {"prompt_hash": "c827097301c986b1", "prompt_version": "m2-2026-10-13",
            "schema_hash": "7d831712685e42d0", "lora_scale": 1.0,
            "server_props": {"build_info": "b11349-fb4b2737a", "model_path": BASE_GGUF, "total_slots": 1}}
# dev set B's generation as scripts/run_special_dev.py records it: think, greedy, R 1280
DEV_B_GENERATION = {"answer_tokens": 1200, "base_seed": 0, "mode": "think", "reasoning_budget": 1280,
                    "sample": 0, "decoding": {"min_p": 0.0, "presence_penalty": 0.0, "temperature": 0.0,
                                              "top_k": 0, "top_p": 1.0}}
# Dev set B in miniature, research_agent.eval.special_dev's items as the tests of
# scripts/calibrate_special_dev.py write them: two functions and the three refusal sentences in the system
# message; 50 items of each special kind, 20 priority items and 90 valid ones
BOOK_ROOM = {"name": "book_room", "description": "Book a meeting room.", "parameters": {
    "type": "object", "properties": {"room": {"type": "string"}, "people": {"type": "integer", "minimum": 1},
                                     "date": {"type": "string"}}, "required": ["room", "people", "date"]}}
PLAN_TRIP = {"name": "plan_trip", "description": "Plan a trip.", "parameters": {
    "type": "object", "properties": {"traveller": {"type": "object"}, "destination": {"type": "string"}},
    "required": ["traveller", "destination"]}}
DEV_B_SYSTEM = "\n".join([
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


def dev_b_item(number: int, kind: str, user: str, expected: dict) -> dict:
    return {"id": f"sdev-b-{number:04d}", "split": "b", "lang": "en", "kind": kind,
            "messages": [{"role": "system", "content": DEV_B_SYSTEM}, {"role": "user", "content": user}],
            "call_format": "python", "family": "fx-decide", "expected": expected,
            "source": {"dataset": "fixture", "record": str(number)}}


def refusal(case: str, values: tuple = (), params: tuple = (), api: str = "") -> dict:
    return {"type": "refusal", "case": case, "fixed": FIXED[case], "values": list(values),
            "params": list(params), "api": api}


def booking(room: str, people: int = 4) -> str:
    return f'book_room(room="{room}", people={people}, date="2026-11-05")'


def dev_b_set() -> list[tuple[dict, str]]:
    """Dev set B's items, each with its right answer."""
    rows = []
    for k in range(1, 51):
        rows.append(("incorrect", f"Book room A-12 for -{k} people on 2026-11-05.",
                     refusal("incorrect", (f"-{k}",), ("people",)),
                     f"Not sent: the value -{k} of people breaks the definition."))
    rows += [("missing", f"Plan trip {k} to Lisbon.",
              refusal("missing", params=("traveller",), api="plan_trip"),
              "Not sent: plan_trip still needs traveller") for k in range(1, 51)]
    rows += [("irrelevant", f"What is the weather in Oslo on day {k}?", refusal("irrelevant"), NO_FUNCTION)
             for k in range(1, 51)]
    rows += [("priority", f"Book room A-12 for -{k} people.", refusal("incorrect", (f"-{k}",), ("people",)),
              f"Not sent: the value -{k} of people breaks the definition.") for k in range(1, 21)]
    for k in range(1, 91):
        room = f"会议室{k}"
        rows.append(("valid_verbatim", f"Book room {room} for 4 people on 2026-11-05.",
                     {"type": "call", "calls": [{"name": "book_room", "arguments": {
                         "room": room, "people": 4, "date": "2026-11-05"}}], "verbatim": [room]},
                     booking(room)))
    return [(dev_b_item(number, kind, user, expected), right)
            for number, (kind, user, expected, right) in enumerate(rows, 1)]


DEV_B_SET = dev_b_set()
DEV_B_ITEMS = len(DEV_B_SET)  # 260: 150 special, 20 priority and 90 valid


def answers_of(wrong: int, busy: bool = False) -> dict[str, tuple]:
    """One seed's answers: the first `wrong` searches answered with another query, the search after the rate
    limit answered with a busy notice (which the rules turn into the retry) when busy, every other one
    right."""
    out = {}
    searches = [row["id"] for rows in ITEMS.values() for row in rows if row["type"] == "call"]
    for rows in ITEMS.values():
        for row in rows:
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


def stem(path: str) -> str:
    return Path(path).stem


def write_runs(run: str, step: int, seeds: tuple[dict, dict], *, generation: dict | None = None) -> dict:
    """A checkpoint's GGUF, its two seeds' runs over both sets with their run manifests (as run_eval writes
    them), and its row of a selection file (as select_checkpoint writes it)."""
    gguf = Path(f"{run}/gguf/checkpoint-{step}-f16.gguf")
    gguf.parent.mkdir(parents=True, exist_ok=True)
    gguf.write_bytes(f"GGUF {run} {step}".encode())
    folder = "runs/eval/select-" + Path(run).name
    variants = {}
    for seed, answers in enumerate(seeds):
        name, sets = f"s{seed}", {}
        for path, rows in ITEMS.items():
            records = f"{folder}/{stem(path)}/checkpoint-{step}.{name}.jsonl"
            write_jsonl([record(row, answers[row["id"]]) for row in rows], records)
            manifest = {"adapters": [{"id": 0, "path": gguf.as_posix(), "sha256": sha256_of(gguf)}],
                        "family": FAMILY, "generation": {**(generation or GENERATION), "base_seed": seed},
                        "items": path, "items_sha256": sha256_of(path), "items_run": len(rows),
                        "items_total": len(rows), "statuses": {"ok": len(rows)}, "system": "ft",
                        **json.loads(json.dumps(MEASURED))}
            Path(records).with_name(Path(records).stem + ".manifest.json").write_text(json.dumps(manifest),
                                                                                    encoding="utf-8")
            sets[path] = {"n": len(rows), "records": records}
        variants[name] = {"sets": sets}
    return {"step": step, "gguf": gguf.as_posix(), "sha256": sha256_of(gguf), "variants": variants}


def write_selection(path: str, run: str, rows: list[dict]) -> None:
    selection = {"run": run, "system": "ft",
                 "item_sets": [{"path": p, "sha256": sha256_of(p)} for p in ITEMS],
                 "variants": [{"name": "s0", "args": ["--mode", "think", "--seed", "0"]},
                              {"name": "s1", "args": ["--mode", "think", "--seed", "1"]}],
                 "selection": {"metric": "mean", "variants": ["s0", "s1"]}, "checkpoints": rows,
                 "chosen": rows[0]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(selection, indent=2), encoding="utf-8")


# A dev set B run by what its scores count, as the gates read them: of the 150 special items, strict and
# lenient; of the 90 valid ones, exact calls, false refusals and zh alterations.
# B0: lenient 120/150, so PG1 asks for 0.85; FT3: exact 72/90 (PG4: at least 0.75), 3 false refusals (PG3:
# at most 5), 4 zh alterations (PG4: at most 4); K1 holds (B0's lenient 6.7 points above FT3's)
B0_B, FT3_B = (90, 120, 80, 1, 2), (100, 110, 72, 3, 4)
FT3_K1_FAILS = (100, 115, 72, 3, 4)  # lenient 115/150, 3.3 points below B0's: K1 fails
CANDIDATES_B = {112: (100, 140, 80, 2, 1),  # strict 0.667: PG2 fails
                224: (110, 135, 70, 4, 3),  # passes all four
                336: (115, 120, 75, 2, 2),  # lenient 0.80: PG1 fails, unless K1 failed (strict 0.767)
                448: (120, 140, 80, 2, 1)}  # passes all four, ranked last


def records_of(counts: tuple) -> list[dict]:
    """A run's records over dev set B whose scores count counts (strict, lenient, exact, false refusals, zh
    altered): of the special items the first `strict` answered with the sentence alone, the next up to
    `lenient` with the sentence and another one after it (lenient, not strict), the rest "OK."; the priority
    items with their sentence; of the valid items the first `exact` with the call, then `false refusals` with
    a refusal sentence, `zh altered` with the room's name altered, and the rest with another number of
    people."""
    strict, lenient, exact, refused, altered = counts
    special = valid = 0
    records = []
    for row, right in DEV_B_SET:
        if row["kind"] in special_dev.SPECIAL_KINDS:
            answer = (right if special < strict else f"{right} Please send it again." if special < lenient
                      else "OK.")
            special += 1
        elif row["kind"] == special_dev.PRIORITY:
            answer = right
        else:
            room = row["expected"]["calls"][0]["arguments"]["room"]
            answer = (right if valid < exact else NO_FUNCTION if valid < exact + refused
                      else booking(f"Room {room[3:]}") if valid < exact + refused + altered
                      else booking(room, people=5))
            valid += 1
        records.append({"id": row["id"], "status": "ok", "answer": answer})
    return records


_SCORED: dict[tuple, tuple[list[dict], list[dict], dict]] = {}


def scored_run(counts: tuple) -> tuple[list[dict], list[dict], dict]:
    """counts' records with their scores and summary by special_dev (score_records, summarize), checked to
    count counts; computed once per counts, as the fixtures write the same runs again and again (not to be
    changed in place)."""
    if counts not in _SCORED:
        records, items = records_of(counts), [row for row, _ in DEV_B_SET]
        scores = special_dev.score_records(records, items)
        value = special_dev.summarize(scores, items)
        got = (value["special"]["strict"], value["special"]["lenient"], value["valid"]["exact"],
               value["valid"]["false_refusals"], value["valid"]["zh_altered"])
        assert got == counts and not value["scoring_errors"], f"the answers count {got}, not {counts}"
        _SCORED[counts] = (records, scores, value)
    return _SCORED[counts]


def counted(counts: tuple) -> dict:
    """counts as the decision records a dev set B run's counts."""
    strict, lenient, exact, refused, altered = counts
    return {"special_n": 150, "strict": strict, "lenient": lenient, "valid_n": 90, "exact": exact,
            "false_refusals": refused, "zh_altered": altered}
# per seed the searches answered wrong, and whether the busy notice comes: 336 and 224 tie at 29 (224's busy
# notices fixed by the rules), 112 has 27, 448 24; FT3 28
SEEDS = {112: ((1, False), (2, False)), 224: ((0, True), (1, True)), 336: ((1, False), (0, False)),
         448: ((3, False), (3, False))}
FT3_SEEDS = ((1, False), (1, False))


def special_files(label: str) -> tuple[Path, Path, Path]:
    """A dev set B run's records, run manifest and summary, named as scripts/run_special_dev.py names them
    (--out runs/m8/special-dev/b-<label>.jsonl)."""
    records = Path(f"runs/m8/special-dev/b-{label}.jsonl")
    return (records, records.with_name(records.stem + ".manifest.json"),
            records.with_name(records.stem + ".summary.json"))


def write_dev_b() -> str:
    """Dev set B and its lock; the locked SHA-256."""
    write_jsonl([row for row, _ in DEV_B_SET], DEV_B)
    return write_lock(DEV_B)


def write_special(label: str, counts: tuple, *, adapter: dict | None, lock: str) -> str:
    """A run over dev set B as scripts/run_special_dev.py writes it: its records (records_of(counts)), its
    manifest (B0 at lora scale 0 on a server without an adapter, an adapter at scale 1; every record ok) and
    its summary (the records' name, the set, its SHA-256, the scorer's rules, its summary under "summary" and
    every item's score); the summary's path."""
    records, manifest, summary_file = special_files(label)
    rows, scores, value = scored_run(counts)
    write_jsonl(rows, records)
    manifest.write_text(json.dumps({
        "script": "scripts/run_special_dev.py", "system": "b0" if adapter is None else "ft",
        "lora_scale": 0.0 if adapter is None else 1.0, "adapters": [] if adapter is None else [adapter],
        "family": FAMILY, "generation": DEV_B_GENERATION, "items": DEV_B, "items_sha256": lock,
        "items_locked": True, "items_total": DEV_B_ITEMS, "items_run": DEV_B_ITEMS, "limit": None,
        "statuses": {"ok": DEV_B_ITEMS},
        "server_props": {"total_slots": 1, "build_info": "b11349-fb4b2737a", "model_path": BASE_GGUF}}),
        encoding="utf-8")
    summary_file.write_text(json.dumps({"records": records.name, "items": DEV_B, "items_sha256": lock,
                                        "rules": special_dev.RULES, "summary": value, "scores": scores},
                                       ensure_ascii=False), encoding="utf-8")
    return summary_file.as_posix()


def adapter_of(row: dict) -> dict:
    """The server's adapter listing of a checkpoint, as a run manifest records it (eval.systems)."""
    return {"id": 0, "path": row["gguf"], "sha256": row["sha256"]}


def write_calibration(b0_b: tuple, ft3_b: tuple, lock: str, verdict: dict | None) -> None:
    """scripts/calibrate_special_dev.py's file: the set and the two runs it calibrated on, their summaries,
    and K1's verdict (verdict: by default K1 of the summaries, B0's lenient special rate at least 5 points
    above FT3's)."""
    runs = {system: {"path": special_files(label)[0].as_posix(), "sha256": sha256_of(special_files(label)[0])}
            for system, label in (("b0", "b0"), ("ft", "ft3"))}
    summaries = {"b0": scored_run(b0_b)[2], "ft": scored_run(ft3_b)[2]}
    holds = (Fraction(b0_b[1], 150) - Fraction(ft3_b[1], 150)) >= Fraction(5, 100)
    identity = {"items": {"path": DEV_B, "sha256": lock, "locked": True, "n": DEV_B_ITEMS}, "runs": runs,
                "summaries": summaries}
    verdict = {"K1": {"holds": holds}, "pg1_fallback": not holds} if verdict is None else verdict
    Path(CALIBRATION).parent.mkdir(parents=True, exist_ok=True)
    Path(CALIBRATION).write_text(json.dumps({**identity, **verdict}, ensure_ascii=False), encoding="utf-8")


def world(*, seeds: dict = SEEDS, ft3_seeds: tuple = FT3_SEEDS, b0_b: tuple = B0_B, ft3_b: tuple = FT3_B,
          candidates_b: dict = CANDIDATES_B, calibration: dict | None = None) -> list[str]:
    """Everything decide_m8 reads, written in the current folder; its arguments."""
    for path, rows in ITEMS.items():
        write_jsonl(rows, path)
        write_lock(path)
    lock = write_dev_b()
    ft5 = [write_runs("runs/train/ft5", step, tuple(answers_of(*s) for s in seeds[step])) for step in STEPS]
    write_selection(FT5_SELECTION, "runs/train/ft5", ft5)
    ft3 = write_runs("runs/train/ft3", FT3_STEP, tuple(answers_of(*s) for s in ft3_seeds))
    write_selection(FT3_SELECTION, "runs/train/ft3", [ft3])
    adapters = {"b0": None, "ft3": adapter_of(ft3), **{str(row["step"]): adapter_of(row) for row in ft5}}
    specials = ["--special", f"b0={write_special('b0', b0_b, adapter=None, lock=lock)}",
                "--special", f"ft3={write_special('ft3', ft3_b, adapter=adapters['ft3'], lock=lock)}"]
    for step, counts in candidates_b.items():
        path = write_special(str(step), counts, adapter=adapters[str(step)], lock=lock)
        specials += ["--special", f"{step}={path}"]
    write_calibration(b0_b, ft3_b, lock, calibration)
    return ["--selection", FT5_SELECTION, "--baseline", FT3_SELECTION, "--baseline-step", str(FT3_STEP),
            *specials, "--calibration", CALIBRATION]


def edit(path: str | Path, change) -> None:
    """Rewrite a JSON file after change(value) has changed it in place."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    change(value)
    Path(path).write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(decide_m8, "OUT", tmp_path / OUT)  # the decision's one place, here
    # dev set B is opened as scripts/run_special_dev.py opens it; its held-out guard looks here, at nothing
    monkeypatch.setattr(decide_m8.runner, "HELD_OUT_ROOTS", (tmp_path / "held-root",))
    return tmp_path


def decision() -> dict:
    return json.loads(Path(OUT).read_text(encoding="utf-8"))


def test_go_walks_the_ranking_to_the_first_checkpoint_passing_every_gate(repo, capsys):
    assert decide_m8.main(world()) == 0
    printed = capsys.readouterr().out
    out = decision()
    # the two-seed sums under the normalized scorer with the rules applied offline; 336 and 224 tie, the later
    # step first
    sums = {c["step"]: (c["s0"], c["s1"], c["sum"], c["sum_without_rules"]) for c in out["candidates"]}
    assert sums == {112: (14, 13, 27, 27), 224: (15, 14, 29, 27), 336: (14, 15, 29, 29),
                    448: (12, 12, 24, 24)}
    assert out["ranking"] == [336, 224, 112, 448] and out["walk"] == [336, 224]
    gates = {c["step"]: {name: g["passed"] for name, g in c["gates"].items()} for c in out["candidates"]}
    assert gates[336] == {"PG1": False, "PG2": True, "PG3": True, "PG4": True}
    assert gates[224] == {"PG1": True, "PG2": True, "PG3": True, "PG4": True}
    assert gates[112]["PG2"] is False and all(gates[448].values())
    assert out["chosen"]["step"] == 224
    assert out["chosen"]["gguf"] == "runs/train/ft5/gguf/checkpoint-224-f16.gguf"
    assert out["chosen"]["sha256"] == sha256_of("runs/train/ft5/gguf/checkpoint-224-f16.gguf")
    assert out["baseline"]["step"] == FT3_STEP and out["baseline"]["sum"] == 28
    assert out["mg"] == {"sum": 29, "ft3_sum": 28, "threshold": 18, "passed": True}
    assert out["decision"] == "GO" and out["pg1_fallback"] is False
    pg1 = next(c for c in out["candidates"] if c["step"] == 224)["gates"]["PG1"]
    assert (pg1["measure"], pg1["value"], pg1["threshold"]) == ("lenient special", 0.9, 0.85)
    pg4 = next(c for c in out["candidates"] if c["step"] == 224)["gates"]["PG4"]
    assert (pg4["exact"], pg4["exact_threshold"]) == (70 / 90, 0.75)
    assert (pg4["zh_altered"], pg4["zh_threshold"]) == (3, 4)
    # where every dev set B number comes from: the set at its lock, each run's records and manifest
    assert out["dev_set_b"] == {"path": DEV_B, "sha256": sha256_of(DEV_B)}
    runs = out["special_dev"]
    assert set(runs) == {"b0", "ft3", "112", "224", "336", "448"}
    records, manifest, summary_file = special_files("224")
    assert runs["224"]["records"] == {"path": records.as_posix(), "sha256": sha256_of(records)}
    assert runs["224"]["manifest"] == {"path": manifest.as_posix(), "sha256": sha256_of(manifest)}
    assert runs["224"]["path"] == summary_file.as_posix() and runs["224"]["lora_scale"] == 1.0
    assert runs["224"]["adapters"] == [out["chosen"]["sha256"]] and runs["b0"]["adapters"] == []
    assert out["special_generation"] == DEV_B_GENERATION
    assert out["calibration"]["runs"] == {name: sha256_of(special_files(label)[0])
                                          for name, label in (("b0", "b0"), ("ft", "ft3"))}
    # the counts the gates read: every run's records scored again by special_dev, every record ok
    assert {label: run["counts"] for label, run in runs.items()} == {
        "b0": counted(B0_B), "ft3": counted(FT3_B), **{str(s): counted(c) for s, c in CANDIDATES_B.items()}}
    assert all(run["statuses"] == {"ok": DEV_B_ITEMS} and run["scoring_errors"] == 0 for run in runs.values())
    assert out["b0"]["special_dev"] == counted(B0_B) and out["baseline"]["special_dev"] == counted(FT3_B)
    assert next(c for c in out["candidates"] if c["step"] == 224)["special_dev"] == counted(CANDIDATES_B[224])
    scorer = Path(special_dev.__file__)
    assert out["special_scorer"] == {"module": "research_agent.eval.special_dev",
                                     "path": "src/research_agent/eval/special_dev.py",
                                     "sha256": sha256_of(scorer)}
    assert out["special_base"] == {
        "server_props.model_path": BASE_GGUF, "server_props.build_info": "b11349-fb4b2737a",
        "family.name": FAMILY["name"], "family.template_sha256": FAMILY["template_sha256"]}
    assert "decision: GO" in printed and "checkpoint 224" in printed
    assert "336" in printed and "PG1" in printed and "MG" in printed
    assert Path(OUT).read_text(encoding="utf-8").endswith("}\n")


def test_with_k1_failed_pg1_reads_the_strict_special_rate_against_0_75(repo, capsys):
    k1_failed = {"K1": {"holds": False}, "pg1_fallback": True}
    assert decide_m8.main(world(ft3_b=FT3_K1_FAILS, calibration=k1_failed)) == 0
    out = decision()
    assert out["pg1_fallback"] is True and out["chosen"]["step"] == 336  # first in the ranking, now passing
    pg1 = out["candidates"][[c["step"] for c in out["candidates"]].index(336)]["gates"]["PG1"]
    assert (pg1["measure"], pg1["value"], pg1["threshold"], pg1["passed"]) == (
        "strict special (K1 failed)", 115 / 150, 0.75, True)
    # the flag alone, or K1's verdict alone, says the same
    Path(OUT).unlink()
    capsys.readouterr()
    assert decide_m8.main(world(ft3_b=FT3_K1_FAILS, calibration={"K1": {"holds": False}})) == 0
    assert decision()["pg1_fallback"] is True
    Path(OUT).unlink()
    assert decide_m8.main(world(calibration={"pg1_fallback": False})) == 0
    assert decision()["pg1_fallback"] is False


@pytest.mark.parametrize("calibration", [{"K1": {"holds": True}, "pg1_fallback": True}, {},
                                         {"pg1_fallback": "no"}],
                         ids=["disagreeing", "silent", "not-a-flag"])
def test_a_calibration_file_that_cannot_tell_the_fallback_is_refused(repo, capsys, calibration):
    assert decide_m8.main(world(calibration=calibration)) == 2
    assert "pg1_fallback" in capsys.readouterr().out and not Path(OUT).exists()


def test_no_go_when_no_checkpoint_passes_the_gates(repo, capsys):
    failing = {step: (*counts[:4], 5) for step, counts in CANDIDATES_B.items()}  # 5 zh alterations, FT3 4
    assert decide_m8.main(world(candidates_b=failing)) == 0
    out = decision()
    assert out["decision"] == "NO-GO" and out["chosen"] is None and out["mg"] is None
    assert out["walk"] == [336, 224, 112, 448]
    assert "no checkpoint passes PG1-PG4" in out["reason"]
    assert "decision: NO-GO" in capsys.readouterr().out


def test_no_go_when_the_chosen_checkpoint_falls_more_than_10_below_ft3_on_dev(repo, capsys):
    weak = {step: ((7, False), (7, False)) for step in STEPS}  # 8 + 8 = 16 of 30
    assert decide_m8.main(world(seeds=weak, ft3_seeds=((0, False), (0, False)))) == 0  # FT3: 30
    out = decision()
    assert out["chosen"]["step"] == 448  # all four tie: the latest step first, and it passes every gate
    assert out["mg"] == {"sum": 16, "ft3_sum": 30, "threshold": 20, "passed": False}
    assert out["decision"] == "NO-GO" and "MG" in out["reason"]


def test_mg_ends_the_walk_at_the_first_checkpoint_passing_whatever_ranks_after_it(repo, capsys):
    # 448 is first in the ranking and passes PG1-PG4; MG fails for it, so NO-GO: the checkpoints ranked after
    # it have no higher sum, and need no dev set B run
    weak = {step: ((7, False), (7, False)) for step in STEPS}
    argv = world(seeds=weak, ft3_seeds=((0, False), (0, False)), candidates_b={448: CANDIDATES_B[448]})
    assert decide_m8.main(argv) == 0
    out = decision()
    assert out["walk"] == [448] and out["chosen"]["step"] == 448
    assert out["mg"]["passed"] is False and out["decision"] == "NO-GO"


def test_dev_set_b_is_needed_only_down_the_ranking_to_the_first_checkpoint_passing(repo, capsys):
    # 336 fails PG1 and 224 passes: 112 and 448, ranked after them, need no dev set B run
    assert decide_m8.main(world(candidates_b={336: CANDIDATES_B[336], 224: CANDIDATES_B[224]})) == 0
    out = decision()
    assert out["decision"] == "GO" and out["chosen"]["step"] == 224 and out["walk"] == [336, 224]
    rows = {c["step"]: c for c in out["candidates"]}
    assert rows[112]["special_dev"] is None and rows[112]["gates"] is None and rows[448]["passes"] is None
    assert set(out["special_dev"]) == {"b0", "ft3", "336", "224"}


def test_a_checkpoint_the_walk_reaches_without_dev_set_b_stops_it_with_exit_3(repo, capsys):
    assert decide_m8.main(world(candidates_b={336: CANDIDATES_B[336]})) == 3  # 336 fails PG1: 224 is next
    printed = capsys.readouterr().out
    assert "NEED dev set B for checkpoint 224" in printed and not Path(OUT).exists()
    assert "runs/train/ft5/gguf/checkpoint-224-f16.gguf" in printed
    # none run yet: the top of the ranking is asked for
    assert decide_m8.main(world(candidates_b={})) == 3
    assert "NEED dev set B for checkpoint 336" in capsys.readouterr().out and not Path(OUT).exists()
    # a run of a checkpoint ranked lower does not stand in for it
    assert decide_m8.main(world(candidates_b={448: CANDIDATES_B[448]})) == 3
    assert "NEED dev set B for checkpoint 336" in capsys.readouterr().out and not Path(OUT).exists()


def test_rank_only_prints_the_ranking_and_ft3s_mg_baseline_and_writes_nothing(repo, capsys):
    world()
    Path(CALIBRATION).unlink()  # neither dev set B nor the calibration is read
    ranking = ["--selection", FT5_SELECTION, "--baseline", FT3_SELECTION, "--rank-only"]
    assert decide_m8.main(ranking) == 0
    printed = capsys.readouterr().out
    lines = [line.strip() for line in printed.splitlines()]
    ranked = [line.split(":")[0] for line in lines if line[:1].isdigit()]
    assert ranked == ["1. step 336", "2. step 224", "3. step 112", "4. step 448"]
    assert "FT3 step 1635: s0 14 + s1 14 = 28" in printed and "MG" in printed and "28 - 10 = 18" in printed
    assert "NEED" not in printed and "336" in printed.split("next")[-1]
    assert not Path(OUT).exists()
    # dev set B's summaries are not read then: refused, not passed over
    assert decide_m8.main([*ranking, "--special", "b0=x"]) == 2
    assert "--rank-only" in capsys.readouterr().out
    # a decision already written does not stop it, and stays as it is
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT).write_text("{}\n", encoding="utf-8")
    assert decide_m8.main(ranking) == 0
    assert Path(OUT).read_text(encoding="utf-8") == "{}\n"


def test_an_existing_decision_is_never_overwritten_and_nothing_is_read_then(repo, capsys):
    Path(OUT).parent.mkdir(parents=True)
    Path(OUT).write_text("{}\n", encoding="utf-8")
    nowhere = ["--selection", "results/none.json", "--special", "b0=x", "--calibration", "y"]
    assert decide_m8.main(nowhere) == 2
    printed = capsys.readouterr().out
    assert f"{OUT} exists" in printed and "decided once" in printed
    assert Path(OUT).read_text(encoding="utf-8") == "{}\n"
    with pytest.raises(SystemExit):  # no other place can be asked for
        decide_m8.main(["--selection", FT5_SELECTION, "--calibration", CALIBRATION, "--out",
                        "elsewhere.json"])


def test_b0_and_ft3_always_need_a_summary_and_every_summary_one_set(repo, capsys):
    argv = world()
    for label in ("b0", "ft3"):
        at = argv.index(next(a for a in argv if a.startswith(f"{label}=")))
        assert decide_m8.main([*argv[:at - 1], *argv[at + 1:]]) == 2
        assert f"no dev set B summary of {label}" in capsys.readouterr().out and not Path(OUT).exists()
    at = argv.index(next(a for a in argv if a.startswith("448=")))
    without = [*argv[:at - 1], *argv[at + 1:]]
    gguf = "runs/train/ft5/gguf/checkpoint-448-f16.gguf"
    adapter = adapter_of({"gguf": gguf, "sha256": sha256_of(gguf)})
    other = write_special("other", CANDIDATES_B[448], adapter=adapter, lock=sha256_of(DEV_B))
    edit(other, lambda s: s["summary"]["valid"].update(n=36))  # another set's counts
    assert decide_m8.main([*without, "--special", f"448={other}"]) == 2
    assert "valid" in capsys.readouterr().out and not Path(OUT).exists()
    assert decide_m8.main([*argv, "--special", f"999={other}"]) == 2  # a step that is no candidate
    assert "999" in capsys.readouterr().out
    assert decide_m8.main([*argv, "--special", f"448={other}"]) == 2  # one twice
    assert "448" in capsys.readouterr().out and not Path(OUT).exists()


@pytest.mark.parametrize(("change", "named"), [
    ({"items_sha256": "c" * 64}, "items_sha256"),  # another set's run, whatever its counts
    ({"items": "data/eval/special-dev-a.jsonl"}, "special-dev-a"),  # dev set A, the data's development set
], ids=["other-sha256", "dev-set-a"])
def test_the_summaries_must_all_be_of_dev_set_b(repo, capsys, change, named):
    argv = world()
    edit(special_files("336")[2], lambda value: value.update(change))
    assert decide_m8.main(argv) == 2
    printed = capsys.readouterr().out
    assert named in printed and "336" in printed and not Path(OUT).exists()


def _adapters(label: str) -> list[dict]:
    return json.loads(special_files(label)[1].read_text(encoding="utf-8"))["adapters"]


def _swap_adapter(label: str, other: str):
    return lambda: edit(special_files(label)[1], lambda m: m.update(adapters=_adapters(other)))


def _every_run(change):
    return lambda: [edit(special_files(label)[1], change) for label in ("b0", "ft3", *map(str, STEPS))]


def _rerun_ft3() -> None:  # FT3's run made again after the calibration: other records
    write_jsonl([{"id": "sdev-b-0000", "status": "ok", "answer": "again"}], special_files("ft3")[0])


# per refusal: what is changed after the world is written, and what the refusal must name
PROVENANCE = {
    "b0-with-an-adapter": (lambda: edit(special_files("b0")[1], lambda m: m.update(
        lora_scale=1.0, adapters=_adapters("ft3"))), ("b0", "lora_scale")),
    "ft3-on-another-gguf": (_swap_adapter("ft3", "448"), ("ft3", "adapters")),
    "candidate-on-another-gguf": (_swap_adapter("224", "336"), ("224", "adapters")),  # labels swapped
    "candidate-not-applied": (lambda: edit(special_files("224")[1], lambda m: m.update(lora_scale=0.0)),
                              ("224", "lora_scale")),
    "another-generation": (lambda: edit(special_files("336")[1], lambda m: m["generation"].update(
        answer_tokens=1000)), ("generation", "336")),
    "not-think": (_every_run(lambda m: m["generation"].update(mode="no_think")), ("generation.mode",)),
    "not-greedy": (_every_run(lambda m: m["generation"]["decoding"].update(temperature=0.6, top_p=0.95)),
                   ("generation.decoding",)),
    "another-budget": (_every_run(lambda m: m["generation"].update(reasoning_budget=1024)),
                       ("generation.reasoning_budget",)),
    "manifest-of-another-set": (lambda: edit(special_files("112")[1], lambda m: m.update(
        items_sha256="0" * 64)), ("112", "items_sha256")),
    "manifest-of-dev-set-a": (lambda: edit(special_files("112")[1], lambda m: m.update(
        items="data/eval/special-dev-a.jsonl")), ("112", "special-dev-a")),
    "a-limited-run": (lambda: edit(special_files("448")[1], lambda m: m.update(items_run=100)),
                      ("448", "items_run")),
    "no-manifest": (lambda: special_files("448")[1].unlink(), ("b-448.manifest.json",)),
    "summary-naming-no-records": (lambda: edit(special_files("448")[2], lambda s: s.pop("records")),
                                  ("448", "records")),
    "calibrated-on-another-set": (lambda: edit(CALIBRATION, lambda c: c["items"].update(sha256="0" * 64)),
                                  ("items.sha256",)),
    "calibrated-on-another-b0-run": (lambda: edit(CALIBRATION, lambda c: c["runs"]["b0"].update(
        sha256="0" * 64)), ("runs.b0.sha256",)),
    "calibrated-on-another-ft3-run": (_rerun_ft3, ("runs.ft.sha256",)),
}


@pytest.mark.parametrize("case", list(PROVENANCE))
def test_a_dev_set_b_run_that_is_not_the_one_its_label_needs_is_refused_by_its_field(repo, capsys, case):
    argv = world()
    change, named = PROVENANCE[case]
    change()
    assert decide_m8.main(argv) == 2
    printed = capsys.readouterr().out
    assert all(name in printed for name in named), printed
    assert not Path(OUT).exists()


def test_b0_on_a_server_with_an_adapter_at_lora_scale_0_is_b0(repo, capsys):
    argv = world()
    adapters = json.loads(special_files("ft3")[1].read_text(encoding="utf-8"))["adapters"]
    edit(special_files("b0")[1], lambda m: m.update(adapters=adapters))  # loaded, scale 0: not applied
    assert decide_m8.main(argv) == 0 and decision()["special_dev"]["b0"]["lora_scale"] == 0.0


def test_dev_set_b_must_still_be_the_locked_set_the_runs_were_made_on(repo, capsys):
    argv = world()
    with Path(DEV_B).open("a", encoding="utf-8", newline="\n") as f:  # one more item, well formed
        f.write(json.dumps({**DEV_B_SET[-1][0], "id": "sdev-b-9999"}, ensure_ascii=False) + "\n")
    assert decide_m8.main(argv) == 5  # changed since its lock
    assert "special-dev-b.jsonl" in capsys.readouterr().out and not Path(OUT).exists()
    write_lock(DEV_B)  # rebuilt and locked again since the runs: they are of the set before
    assert decide_m8.main(argv) == 2
    assert "items_sha256" in capsys.readouterr().out and not Path(OUT).exists()


@pytest.mark.parametrize(("change", "code", "named"), [
    (lambda: special_files("336")[0].unlink(), 2, ("records", "336", "b-336.jsonl")),
    (lambda: Path(DEV_B).unlink(), 2, (DEV_B,)),
    (lambda: Path(DEV_B + ".sha256").unlink(), 5, ("special-dev-b.jsonl", "no lock")),
], ids=["records-gone", "dev-set-b-gone", "lock-gone"])
def test_dev_set_b_and_every_runs_records_must_be_there_and_dev_set_b_locked(repo, capsys, change, code,
                                                                              named):
    argv = world()
    change()
    assert decide_m8.main(argv) == code
    printed = capsys.readouterr().out
    assert all(name in printed for name in named), printed
    assert not Path(OUT).exists()


@pytest.mark.parametrize(("labels", "change", "named"), [
    (("b0",), lambda m: m["server_props"].update(model_path="models/Other-F16.gguf"),
     "server_props.model_path"),  # B0 on another base model
    (("ft3",), lambda m: m["server_props"].update(build_info="b9999-0000000"),
     "server_props.build_info"),  # another llama.cpp build
    (("224",), lambda m: m["family"].update(template_sha256="0" * 64), "family.template_sha256"),
    (("448",), lambda m: m["family"].update(name="other-family"), "family.name"),
    (("336",), lambda m: m["server_props"].update(total_slots=4), "server_props.total_slots"),
    (("b0", "ft3", *map(str, STEPS)), lambda m: m["server_props"].update(model_path=None),
     "server_props.model_path"),  # not recorded anywhere: nothing shows the runs share their base
], ids=["b0-other-base", "ft3-other-build", "224-other-template", "448-other-family", "336-four-slots",
        "base-not-recorded"])
def test_dev_set_b_runs_on_another_base_build_or_family_are_refused_by_the_field(repo, capsys, labels, change,
                                                                                named):
    argv = world()
    for label in labels:
        edit(special_files(label)[1], change)
    assert decide_m8.main(argv) == 2
    printed = capsys.readouterr().out
    assert named in printed and labels[0] in printed, printed
    assert not Path(OUT).exists()


def _record_failed(label: str) -> None:  # the server failed on the first item: an error record in its place
    records = special_files(label)[0]
    rows = [json.loads(line) for line in records.read_text(encoding="utf-8").splitlines()]
    rows[0] = {"id": rows[0]["id"], "status": "error", "error": "the server failed",
               "error_type": "HTTPError"}
    write_jsonl(rows, records)


# per refusal: what is changed after the world is written, and what the refusal must name
FAILED = {
    "b0-with-error-records": (lambda: edit(special_files("b0")[1], lambda m: m.update(
        statuses={"error": 2, "ok": DEV_B_ITEMS - 2})), ("b0", "statuses")),
    "ft3-with-a-context-overflow": (lambda: edit(special_files("ft3")[1], lambda m: m.update(
        statuses={"context_overflow": 1, "ok": DEV_B_ITEMS - 1})), ("ft3", "statuses")),
    "an-error-record-the-manifest-does-not-count": (lambda: _record_failed("336"), ("336", "error")),
    "a-summary-with-scoring-errors": (lambda: edit(special_files("224")[2], lambda s: s["summary"].update(
        scoring_errors=40)), ("224", "scoring_errors")),
}


@pytest.mark.parametrize("case", list(FAILED))
def test_a_dev_set_b_run_whose_requests_failed_or_could_not_be_scored_is_refused(repo, capsys, case):
    argv = world()
    change, named = FAILED[case]
    change()
    assert decide_m8.main(argv) == 2
    printed = capsys.readouterr().out
    assert all(name in printed for name in named), printed
    assert not Path(OUT).exists()


def test_records_the_scorer_fails_on_now_are_refused(repo, capsys, monkeypatch):
    argv = world()
    real = special_dev.score_item

    def failing(answer, item, **kwargs):  # the scorer as it is now fails on the first item
        if item["id"] == "sdev-b-0001":
            raise ValueError("cannot score this")
        return real(answer, item, **kwargs)

    monkeypatch.setattr(special_dev, "score_item", failing)
    assert decide_m8.main(argv) == 2
    printed = capsys.readouterr().out
    assert "scoring_errors 1" in printed and "b0" in printed and not Path(OUT).exists()


@pytest.mark.parametrize(("label", "name", "value"), [
    ("b0", "special.lenient", 100),  # B0's summary of an older scorer: PG1's bar would move
    ("ft3", "valid.false_refusals", 9),  # PG3's
    ("336", "special.strict", 140),
    ("224", "valid.zh_altered", 0),
    ("448", "valid.exact", 90),
], ids=["b0-lenient", "ft3-false-refusals", "336-strict", "224-zh-altered", "448-exact"])
def test_a_summary_whose_counts_are_not_its_records_scored_again_is_refused(repo, capsys, label, name, value):
    argv = world()
    group, key = name.split(".")
    edit(special_files(label)[2], lambda s: s["summary"][group].update({key: value}))
    assert decide_m8.main(argv) == 2
    printed = capsys.readouterr().out
    assert name in printed and label in printed and "scored again" in printed, printed
    assert not Path(OUT).exists()


def test_runs_the_scorer_counts_otherwise_now_than_their_summaries_are_refused(repo, capsys, monkeypatch):
    # the scorer revised since the runs (and their summaries) were made: it now judges a refusal to a missing
    # item otherwise, so every summary is of another scorer than the one that would decide
    argv = world()
    real = special_dev.score_item

    def revised(answer, item, **kwargs):
        row = real(answer, item, **kwargs)
        return {**row, "lenient": False, "strict": False} if item["kind"] == "missing" else row

    monkeypatch.setattr(special_dev, "score_item", revised)
    assert decide_m8.main(argv) == 2
    printed = capsys.readouterr().out
    assert "b0" in printed and "special.lenient" in printed and "scored again" in printed
    assert not Path(OUT).exists()


@pytest.mark.parametrize(("change", "named"), [
    (lambda c: c["summaries"]["b0"]["special"].update(lenient=100), "summaries.b0 special.lenient"),
    (lambda c: c["summaries"]["ft"]["valid"].update(exact=60), "summaries.ft valid.exact"),
    (lambda c: c.pop("summaries"), "summaries.b0"),
], ids=["b0-lenient", "ft-exact", "no-summaries"])
def test_a_calibration_whose_summaries_are_not_the_runs_scored_again_is_refused(repo, capsys, change, named):
    argv = world()
    edit(CALIBRATION, change)
    assert decide_m8.main(argv) == 2
    printed = capsys.readouterr().out
    assert named in printed and CALIBRATION in printed and not Path(OUT).exists()


def test_a_calibration_whose_k1_is_not_that_of_the_runs_scored_again_is_refused(repo, capsys):
    # B0's lenient 120/150 and FT3's 110/150 are 6.7 points apart: K1 holds, whatever the file says
    assert decide_m8.main(world(calibration={"K1": {"holds": False}, "pg1_fallback": True})) == 2
    printed = capsys.readouterr().out
    assert "K1 holds" in printed and CALIBRATION in printed and not Path(OUT).exists()
    # FT3's 115/150 is 3.3 points below B0's: K1 fails, and a file saying it holds is refused
    assert decide_m8.main(world(ft3_b=FT3_K1_FAILS, calibration={"K1": {"holds": True},
                                                                 "pg1_fallback": False})) == 2
    assert "K1 fails" in capsys.readouterr().out and not Path(OUT).exists()


def test_the_calibration_file_is_the_calibration_scripts_by_default(repo, capsys, monkeypatch):
    argv = world()
    at = argv.index("--calibration")
    Path(decide_m8.CALIBRATION).parent.mkdir(parents=True, exist_ok=True)
    Path(CALIBRATION).replace(decide_m8.CALIBRATION)
    assert decide_m8.CALIBRATION == Path("results/special-dev-calibration-m8.json")
    assert decide_m8.main([*argv[:at], *argv[at + 2:]]) == 0
    assert decision()["calibration"]["path"] == "results/special-dev-calibration-m8.json"


def test_a_malformed_summary_is_refused_as_malformed(repo, capsys):
    argv = world()
    edit(special_files("112")[2], lambda s: s.update(summary={"special": {"n": 150}}))
    assert decide_m8.main(argv) == 1
    assert "112" in capsys.readouterr().out and not Path(OUT).exists()


@pytest.mark.parametrize("change", ["seed", "adapter", "items-sha", "items-run", "generation"])
def test_a_run_manifest_that_does_not_fit_the_candidate_is_refused(repo, capsys, change):
    argv = world()
    target = Path("runs/eval/select-ft5/dev-v2/checkpoint-336.s1.manifest.json")
    manifest = json.loads(target.read_text(encoding="utf-8"))
    if change == "seed":
        manifest["generation"]["base_seed"] = 0  # seed 1's run made with seed 0
    elif change == "adapter":
        manifest["adapters"][0]["sha256"] = "0" * 64
    elif change == "items-sha":
        manifest["items_sha256"] = "0" * 64
    elif change == "items-run":
        manifest["items_run"] = 3
    else:
        manifest["generation"]["reasoning_budget"] = 1024  # not FT3's: the sums would not be comparable
    target.write_text(json.dumps(manifest), encoding="utf-8")
    assert decide_m8.main(argv) == 2
    assert target.as_posix() in capsys.readouterr().out and not Path(OUT).exists()


@pytest.mark.parametrize(("field", "change"), [
    ("prompt_hash", lambda m: m.update(prompt_hash="0" * 16)),  # another agent prompt
    ("prompt_version", lambda m: m.update(prompt_version="m8-2026-11-02")),
    ("schema_hash", lambda m: m.update(schema_hash="0" * 16)),  # other tool schemas
    ("lora_scale", lambda m: m.update(lora_scale=0.5)),
    ("server_props.model_path", lambda m: m["server_props"].update(model_path="models/Other-Q8_0.gguf")),
], ids=["prompt-hash", "prompt-version", "schema-hash", "lora-scale", "model-path"])
def test_a_selection_run_measured_otherwise_than_ft3s_is_refused_by_its_field(repo, capsys, field, change):
    argv = world()
    target = Path("runs/eval/select-ft5/dev/checkpoint-224.s0.manifest.json")
    edit(target, change)
    assert decide_m8.main(argv) == 2
    printed = capsys.readouterr().out
    assert field in printed and target.as_posix() in printed and not Path(OUT).exists()
    assert decide_m8.main([*argv[:6], "--rank-only"]) == 2  # the ranking compares the same metric too


def test_a_set_changed_since_the_runs_or_another_baseline_set_is_refused(repo, capsys):
    argv = world()
    rows = [json.loads(line) for line in Path(DEV_V2).read_text(encoding="utf-8").splitlines()]
    write_jsonl(rows[:-1], DEV_V2)
    write_lock(DEV_V2)  # a consistent lock: the set itself changed since the selection
    assert decide_m8.main(argv) == 2
    assert DEV_V2 in capsys.readouterr().out and not Path(OUT).exists()
    argv = world()
    selection = json.loads(Path(FT3_SELECTION).read_text(encoding="utf-8"))
    selection["item_sets"] = selection["item_sets"][:1]
    Path(FT3_SELECTION).write_text(json.dumps(selection), encoding="utf-8")
    assert decide_m8.main(argv) == 2
    assert "item_sets" in capsys.readouterr().out


def test_a_run_without_a_record_of_every_item_is_malformed(repo, capsys):
    argv = world()
    records = Path("runs/eval/select-ft5/dev/checkpoint-112.s0.jsonl")
    lines = records.read_text(encoding="utf-8").splitlines()
    records.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    assert decide_m8.main(argv) == 1
    assert records.as_posix() in capsys.readouterr().out and not Path(OUT).exists()


def test_dev_set_b_records_without_one_of_every_item_are_malformed(repo, capsys):
    argv = world()
    records = special_files("448")[0]
    lines = records.read_text(encoding="utf-8").splitlines()
    records.write_text("\n".join(lines[1:]) + "\n", encoding="utf-8")
    assert decide_m8.main(argv) == 1
    printed = capsys.readouterr().out
    assert records.as_posix() in printed and "448" in printed and not Path(OUT).exists()


# ---- with the real runner and calibration ----------------------------------------------------------------
# Dev set B of three items, run by scripts/run_special_dev.py on tests/support's FakeLlama (a fake family's
# model folder) and calibrated by scripts/calibrate_special_dev.py: decide_m8 reads what the two write.

SERVER = "http://fake-llama.invalid"
SAMPLING = Decoding(temperature=0.6, top_p=0.95)  # the family's thinking decoding, which the runner overrides
CONTRACT_ITEMS = [
    dev_b_item(1, "incorrect", "Book room A-12 for -3 people on 2026-11-05.",
               refusal("incorrect", ("-3",), ("people",))),
    dev_b_item(2, "missing", "Plan a trip to Lisbon.",
               refusal("missing", params=("traveller",), api="plan_trip")),
    dev_b_item(3, "valid_verbatim", "Book room 会议室A for 4 people on 2026-11-05.",
               {"type": "call", "calls": [{"name": "book_room", "arguments": {
                   "room": "会议室A", "people": 4, "date": "2026-11-05"}}], "verbatim": ["会议室A"]}),
]
ANSWERED = ("Not sent: the value -3 of people breaks the definition.",
            "Not sent: plan_trip still needs traveller", booking("会议室A"))
CALLED = (booking("A-12", people=-3), 'plan_trip(destination="Lisbon")', booking("会议室A"))


@pytest.fixture
def real_runs(repo, monkeypatch):
    """The world's selection files with CONTRACT_ITEMS for dev set B (locked), a fake family's model folder,
    and the real runner and calibration, their held-out guards looking at nothing: (run, calibrate, the
    selection arguments). run(label, answers) runs label over dev set B (b0 on a server without an adapter,
    ft3 and a step on their GGUF) and gives its --special; calibrate() calibrates on b0's and ft3's runs and
    gives the file."""
    rsd, cal = load_script("run_special_dev"), load_script("calibrate_special_dev")
    for module in (rsd, cal.runner):
        monkeypatch.setattr(module, "HELD_OUT_ROOTS", (repo / "held-root",))
    folder = repo / "model"
    install_fake_family(monkeypatch, folder, "only", inference_mode=Mode.THINK,
                        decoding={Mode.THINK: SAMPLING, Mode.NO_THINK: SAMPLING})
    base = world()[:6]  # the dev sets, both selection files and the GGUFs
    write_jsonl(CONTRACT_ITEMS, DEV_B)
    write_lock(DEV_B)
    ggufs = {"ft3": f"runs/train/ft3/gguf/checkpoint-{FT3_STEP}-f16.gguf",
             **{str(step): f"runs/train/ft5/gguf/checkpoint-{step}-f16.gguf" for step in STEPS}}

    def run(label: str, answers: tuple[str, ...]) -> list[str]:
        name = label if label in ("b0", "ft3") else f"ft5-{label}"  # M8's paths
        adapters = [] if label == "b0" else [{"id": 0, "path": ggufs[label], "scale": 0.0}]
        fake = FakeLlama([Scripted(("Read the rules.", "\n</think>\n\n", answer)) for answer in answers],
                         adapters=adapters, props={**PROPS, "model_path": BASE_GGUF})
        argv = [DEV_B, "--system", "b0" if label == "b0" else "ft", "--tokenizer", str(folder), "--server",
                SERVER, "--out", f"runs/m8/special-dev/b-{name}.jsonl", "--force"]
        assert rsd.main(argv, local_client=fake.client(base_url=SERVER)) == 0
        return ["--special", f"{label}=runs/m8/special-dev/b-{name}.summary.json"]

    def calibrate() -> dict:
        argv = ["--b0", "runs/m8/special-dev/b-b0.jsonl", "--ft", "runs/m8/special-dev/b-ft3.jsonl",
                "--items", DEV_B, "--out", CALIBRATION, "--force"]
        assert cal.main(argv) == 0
        return json.loads(Path(CALIBRATION).read_text(encoding="utf-8"))

    return run, calibrate, [*base, "--calibration", CALIBRATION]


def test_decides_on_what_the_real_runner_and_calibration_write(real_runs, capsys):
    run, calibrate, base = real_runs
    specials = run("b0", CALLED) + run("ft3", ANSWERED)  # B0 below FT3 on dev set B: K1 fails
    calibration = calibrate()
    specials += run("336", ANSWERED)
    assert calibration["K1"]["holds"] is False and calibration["pg1_fallback"] is True
    assert decide_m8.main([*base, *specials]) == 0
    out = decision()
    assert out["decision"] == "GO" and out["chosen"]["step"] == 336 and out["walk"] == [336]
    assert out["pg1_fallback"] is True
    assert out["special_dev"]["336"]["counts"] == {"special_n": 2, "strict": 2, "lenient": 2, "valid_n": 1,
                                                   "exact": 1, "false_refusals": 0, "zh_altered": 0}
    assert out["special_dev"]["b0"]["statuses"] == {"ok": 3} and out["special_dev"]["b0"]["adapters"] == []
    generation = out["special_generation"]
    assert (generation["mode"], generation["reasoning_budget"], generation["answer_tokens"]) == (
        "think", 1280, 1200)
    assert out["special_base"]["server_props.model_path"] == BASE_GGUF


def test_summaries_written_by_a_scorer_revised_since_are_refused(real_runs, capsys, monkeypatch):
    # B0 and FT3 run under a scorer that judged the missing item's refusal neither strict nor lenient; the
    # calibration and checkpoint 336 under today's
    run, calibrate, base = real_runs
    real = special_dev.score_item

    def older(answer, item, **kwargs):
        row = real(answer, item, **kwargs)
        return {**row, "lenient": False, "strict": False} if item["id"] == "sdev-b-0002" else row

    monkeypatch.setattr(special_dev, "score_item", older)
    specials = run("b0", CALLED) + run("ft3", ANSWERED)
    monkeypatch.setattr(special_dev, "score_item", real)
    calibrate()
    specials += run("336", ANSWERED)
    assert decide_m8.main([*base, *specials]) == 2
    printed = capsys.readouterr().out
    assert "ft3" in printed and "special.lenient 1" in printed and "scored again give 2" in printed, printed
    assert not Path(OUT).exists()


def test_a_b0_run_whose_server_failed_mid_run_is_refused(real_runs, capsys):
    run, calibrate, base = real_runs
    specials = run("b0", CALLED[:1]) + run("ft3", ANSWERED)  # past its one answer the server fails
    calibrate()
    specials += run("336", ANSWERED)
    manifest = json.loads(special_files("b0")[1].read_text(encoding="utf-8"))
    assert manifest["statuses"] == {"error": 2, "ok": 1} and manifest["items_run"] == manifest["items_total"]
    assert decide_m8.main([*base, *specials]) == 2
    printed = capsys.readouterr().out
    assert "b0" in printed and "statuses" in printed and not Path(OUT).exists()

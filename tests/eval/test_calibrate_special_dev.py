"""scripts/calibrate_special_dev.py (M8 WP2, 校准) on fixture runs: K1 to K3 on B0's and FT's records over a
small dev set B, the PG1 basis that K1 decides, and the inputs it refuses. The held-out roots are moved to
an empty folder for every test, so no held-out set of the repository is ever looked at."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from research_agent.data import special_mix as sm
from research_agent.eval import special_dev as sd
from research_agent.eval.lock import write_lock

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cal = load_script("calibrate_special_dev")
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
         "missing": ["Not sent:", "still needs"],
         "irrelevant": ["Not sent: no listed function can do this."]}


def item(number: int, kind: str, user: str, expected: dict) -> dict:
    return {"id": f"sdev-b-{number:04d}", "split": "b", "lang": "en", "kind": kind,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            "call_format": "python", "family": "fx-cal", "expected": expected,
            "source": {"dataset": "fixture", "record": str(number)}}


def refusal(case: str, values=(), params=(), api="") -> dict:
    return {"type": "refusal", "case": case, "fixed": FIXED[case], "values": list(values),
            "params": list(params), "api": api}


def booking(room: str, people: int, date: str) -> dict:
    return {"type": "call", "calls": [{"name": "book_room",
                                       "arguments": {"room": room, "people": people, "date": date}}],
            "verbatim": [room]}


ITEMS = [
    item(1, "incorrect", "Book room A-12 for -3 people on 2026-11-05.",
         refusal("incorrect", ["-3"], ["people"])),
    item(2, "incorrect", "Book room A-12 for ten-ish people on 2026-11-05.",
         refusal("incorrect", ["ten-ish"], ["people"])),
    item(3, "missing", "Plan a trip to Lisbon.", refusal("missing", params=["traveller"], api="plan_trip")),
    item(4, "irrelevant", "What is the weather in Oslo?", refusal("irrelevant")),
    item(5, "valid_verbatim", "Book room 会议室A for 4 people on 2026-11-05.",
         booking("会议室A", 4, "2026-11-05")),
    item(6, "valid_verbatim", "Book room 会议室B for 2 people on 2026-11-06.",
         booking("会议室B", 2, "2026-11-06")),
]
SENTENCES = {
    "I1": "Not sent: the value -3 of people breaks the definition.",
    "I2": "Not sent: the value ten-ish of people breaks the definition.",
    "M1": "Not sent: plan_trip still needs traveller",
    "R1": "Not sent: no listed function can do this.",
    "V1": 'book_room(room="会议室A", people=4, date="2026-11-05")',
    "V2": 'book_room(room="会议室B", people=2, date="2026-11-06")',
}
B0 = dict(zip([i["id"] for i in ITEMS], SENTENCES.values(), strict=True))
FT = {**B0,
      "sdev-b-0001": 'book_room(room="A-12", people=-3, date="2026-11-05")',  # called with the invalid value
      "sdev-b-0002": "Not sent: book_room still needs people",  # the missing-parameter sentence
      "sdev-b-0005": 'book_room(room="Meeting room A", people=4, date="2026-11-05")'}  # a zh value altered
GREEDY = {"temperature": 0.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "presence_penalty": 0.0}
GENERATION = {"mode": "think", "decoding": GREEDY, "reasoning_budget": 1280, "answer_tokens": 1200,
              "base_seed": 0, "sample": 0}


@pytest.fixture(autouse=True)
def no_held_out_sets(tmp_path, monkeypatch):
    monkeypatch.setattr(cal.runner, "HELD_OUT_ROOTS", (tmp_path / "held-root",))


def write_items(path: Path, items: list[dict] = ITEMS, *, lock: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8")
    if lock:
        write_lock(path)
    return path


def write_run(folder: Path, name: str, answers: dict[str, str], items: Path,
              changes: dict | None = None) -> Path:
    """name's records at folder/<name>.jsonl and its manifest beside them, changed by changes."""
    records = folder / f"{name}.jsonl"
    folder.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"id": rid, "status": "ok", "answer": answer}, ensure_ascii=False)
             for rid, answer in answers.items()]
    records.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    base = {"system": name, "items_sha256": hashlib.sha256(items.read_bytes()).hexdigest(),
            "generation": GENERATION, "family": {"name": "nanbeige4.1-3b"}, "adapters": []}
    manifest = folder / f"{name}.manifest.json"
    manifest.write_text(json.dumps({**base, **(changes or {})}), encoding="utf-8")
    return records


def calibrate(tmp_path, b0=B0, ft=FT, *args, b0_manifest=None, ft_manifest=None,
              items=None) -> tuple[int, Path]:
    items = items or write_items(tmp_path / "special-dev-b.jsonl")
    out = tmp_path / "results" / "calibration.json"
    code = cal.main(["--b0", str(write_run(tmp_path / "runs", "b0", b0, items, b0_manifest)),
                     "--ft", str(write_run(tmp_path / "runs", "ft", ft, items, ft_manifest)),
                     "--items", str(items), "--out", str(out), *args])
    return code, out


def test_k1_to_k3_hold_on_the_failure_pattern_of_acebench(tmp_path, capsys):
    code, out = calibrate(tmp_path)
    assert code == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    k1, k2, k3 = (result[k] for k in ("K1", "K2", "K3"))  # at the top, where scripts/decide_m8.py reads K1
    assert (k1["b0"], k1["ft"], k1["difference"], k1["holds"]) == (1.0, 0.5, 0.5, True)
    assert k2["b0"] == {"n": 2, "count": 0, "share": 0.0, "called_with_invalid": 0, "missing_sentence": 0,
                        "broad": {"count": 0, "share": 0.0, "called": 0, "wrong_case": 0}}
    assert k2["ft"] == {"n": 2, "count": 2, "share": 1.0, "called_with_invalid": 1, "missing_sentence": 1,
                        "broad": {"count": 2, "share": 1.0, "called": 1, "wrong_case": 1}}
    assert k2["holds"] is True
    assert (k3["b0"], k3["ft"], k3["holds"]) == (0, 1, True)
    assert result["pg1_fallback"] is False
    assert result["pg1"]["basis"] == "lenient" and result["pg1"]["b0_lenient_special"] == 1.0
    summaries = result["summaries"]
    assert summaries["ft"]["special"]["lenient"] == 2 and summaries["b0"]["valid"]["exact"] == 2
    assert result["items"]["n"] == 6 and result["items"]["locked"] is True
    assert result["runs"]["ft"]["family"] == "nanbeige4.1-3b" and set(result["rules"]) >= {"K1", "K2", "K3"}
    assert result["runs"]["b0"]["scoring_errors"] == 0
    printed = capsys.readouterr().out
    assert "K1" in printed and "K2" in printed and "K3" in printed and printed.count("holds") == 3
    assert "PG1: lenient" in printed


def scores_of(answers: dict[str, str]) -> list[dict]:
    """The scores of answers to the fixture set, B0's answer where answers gives none."""
    return sd.score_items([answers.get(i["id"], B0[i["id"]]) for i in ITEMS], ITEMS)


@pytest.mark.parametrize(
    "answer,counted",
    [
        ('book_room(room="A-12", people=-3, date="2026-11-05")', True),  # called with the invalid value
        ('book_room("A-12", -3, "2026-11-05")', True),  # the same call, loosely written
        ("Not sent: book_room still needs people", True),  # the missing-parameter sentence in its place
        ('book_room(room="A-12", people=3, date="2026-11-05")', False),  # called with a corrected value
        ("Not sent: no listed function can do this.", False),  # the irrelevant sentence
        ("Not sent: the value -3 of people breaks the definition.\nNot sent: book_room still needs people",
         False),  # its own sentence stands too
    ],
)
def test_k2_counts_a_call_with_the_invalid_value_or_the_missing_sentence_in_place_of_its_own(answer, counted):
    result = cal.k2(scores_of({}), scores_of({"sdev-b-0001": answer}))
    assert (result["ft"]["count"], result["holds"]) == (int(counted), counted)


def test_k2_counts_the_missing_sentence_written_without_its_final_mark():
    # a missing sentence that ends with a full stop ("... still needs <params>."), answered without it, stands
    # all the same (special_dev reads an answer's end as the sentence's end), in place of the item's own
    system = SYSTEM.replace("still needs <params>", "still needs <params>.")
    items = copy.deepcopy(ITEMS)
    for one in items:
        one["messages"][0]["content"] = system
        if one["expected"].get("case") == "missing":
            one["expected"]["fixed"] = ["Not sent:", "still needs", "."]
    assert all(sd.check_item(i) == [] for i in items)

    def scores(answers: dict[str, str]) -> list[dict]:
        return sd.score_items([answers.get(i["id"], B0[i["id"]]) for i in items], items)

    result = cal.k2(scores({}), scores({"sdev-b-0001": "Not sent: book_room still needs people"}))
    assert (result["ft"]["missing_sentence"], result["ft"]["count"], result["holds"]) == (1, 1, True)


def test_k2_keeps_the_broad_count_as_a_side_figure():
    ft = scores_of({"sdev-b-0001": 'book_room(room="A-12", people=3, date="2026-11-05")',
                    "sdev-b-0002": "Not sent: no listed function can do this."})
    result = cal.k2(scores_of({}), ft)
    assert result["ft"]["count"] == 0
    assert result["ft"]["broad"] == {"count": 2, "share": 1.0, "called": 1, "wrong_case": 1}
    assert result["holds"] is False


def test_k2_reads_the_incorrect_kind_alone():
    def row(kind: str, invalid: bool) -> dict:
        return {"kind": kind, "call_notation": "python" if invalid else None, "called_with_invalid": invalid,
                "sentences": [], "sentence": False, "failure": "called" if invalid else "no_sentence"}

    b0 = [row("incorrect", False), row("priority", False)]
    ft = [row("incorrect", False), row("priority", True)]  # a priority item called with its invalid value
    result = cal.k2(b0, ft)
    assert (result["ft"]["n"], result["ft"]["count"], result["holds"]) == (1, 0, False)


def test_k2_needs_a_larger_share_than_b0s():
    invalid = scores_of({"sdev-b-0001": 'book_room(room="A-12", people=-3, date="2026-11-05")'})
    assert cal.k2(invalid, invalid)["holds"] is False
    assert cal.k2(scores_of({}), invalid)["holds"] is True
    assert cal.k2(invalid, scores_of({}))["holds"] is False
    with pytest.raises(cal.InputError):
        cal.k2([], [])


def test_k1_failing_moves_pg1_to_the_strict_rate(tmp_path, capsys):
    like_b0 = {**FT, "sdev-b-0001": B0["sdev-b-0001"], "sdev-b-0002": B0["sdev-b-0002"]}
    code, out = calibrate(tmp_path, ft=like_b0)
    assert code == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["K1"]["holds"] is False and result["K2"]["holds"] is False and result["K3"]["holds"] is True
    assert result["pg1_fallback"] is True
    assert result["pg1"]["basis"] == "strict" and "0.75" in result["pg1"]["rule"]
    printed = capsys.readouterr().out
    assert "does not hold" in printed and "PG1: strict" in printed


def test_k3_needs_more_alterations_than_b0():
    def summary(altered: int) -> dict:
        return {"valid": {"zh_altered": altered}}

    assert cal.k3(summary(1), summary(2))["holds"] is True
    assert cal.k3(summary(2), summary(2))["holds"] is False


def test_k1_holds_at_exactly_five_points():
    def special(lenient: int, n: int = 20) -> dict:
        return {"special": {"n": n, "lenient": lenient}}

    assert cal.k1(special(11), special(10))["holds"] is True  # 55% - 50% = 5 points exactly
    assert cal.k1(special(10), special(10))["holds"] is False
    assert cal.k1(special(3, 4), special(2, 4))["difference"] == 0.25


STYLE_WORDS = {"angle": {"value": "value", "param": "parameter", "params": "parameters", "api": "function"},
               "caps": {"value": "VALUE_GIVEN", "param": "PARAM_NAME", "params": "MISSING_PARAMS",
                        "api": "FUNCTION_NAME"}}


def mix_family(case: str) -> sm.Family:
    """The first English development family of a case (research_agent.data.special_mix, read only)."""
    return next(family for fid, family in sorted(sm.FAMILIES.items()) if fid.startswith(f"{case}-en-b"))


def mixed_set() -> tuple[list[dict], dict[str, str]]:
    """Dev set B whose incorrect family comes in two placeholder styles, and the ideal answers."""
    items, answers = [], {}
    for number, style, value in ((1, "angle", "-3"), (2, "caps", "0")):
        family = mix_family("incorrect")
        sentences = [sm.fill_sentence(mix_family(case), style, "en", **STYLE_WORDS[style])
                     for case in ("incorrect", "missing", "irrelevant")]
        system = SYSTEM.replace("Not sent: the value <value> of <param> breaks the definition.", sentences[0])
        system = system.replace("Not sent: <api> still needs <params>", sentences[1])
        system = system.replace("Not sent: no listed function can do this.", sentences[2])
        expected = {"type": "refusal", "case": "incorrect", "fixed": sm.fixed_parts(family, style, "en"),
                    "values": [value], "params": ["people"], "api": ""}
        mixed = {**item(number, "incorrect", f"Book room A-12 for {value} people on 2026-11-05.", expected),
                 "family": family.id}
        mixed["messages"] = [{"role": "system", "content": system}, *mixed["messages"][1:]]
        items.append(mixed)
        answers[mixed["id"]] = sm.fill_sentence(family, style, "en", value=value, param="people")
    items += [copy.deepcopy(i) for i in ITEMS[2:]]
    answers.update({i["id"]: B0[i["id"]] for i in ITEMS[2:]})
    return items, answers


def test_a_set_that_writes_one_family_in_two_styles_is_calibrated(tmp_path):
    items, b0 = mixed_set()
    assert all(sd.check_item(i) == [] for i in items)
    ft = {**b0, "sdev-b-0002": "Not sent: book_room still needs people"}
    code, out = calibrate(tmp_path, b0, ft, items=write_items(tmp_path / "special-dev-b.jsonl", items))
    assert code == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["summaries"]["b0"]["special"]["strict"] == 4 and result["K2"]["ft"]["count"] == 1


def test_error_records_are_counted_as_no_answers(tmp_path, capsys):
    items = write_items(tmp_path / "special-dev-b.jsonl")
    b0 = write_run(tmp_path / "runs", "b0", B0, items)
    lines = b0.read_text(encoding="utf-8").splitlines()
    lines[0] = json.dumps({"id": "sdev-b-0001", "status": "error", "error": "boom",
                           "error_type": "ReadTimeout"})
    b0.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = tmp_path / "calibration.json"
    code = cal.main(["--b0", str(b0), "--ft", str(write_run(tmp_path / "runs", "ft", FT, items)),
                     "--items", str(items), "--out", str(out)])
    assert code == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["runs"]["b0"]["errors"] == 1 and result["summaries"]["b0"]["special"]["lenient"] == 3
    assert "1 error record" in capsys.readouterr().out


@pytest.mark.parametrize(
    "change",
    [
        {"b0": {k: v for k, v in B0.items() if k != "sdev-b-0003"}},  # an item without a record
        {"b0_manifest": {"system": "ft"}},
        {"ft_manifest": {"items_sha256": "0" * 64}},
        {"ft_manifest": {"generation": {**GENERATION, "reasoning_budget": 1024}}},
        {"b0_manifest": {"generation": {**GENERATION, "decoding": {**GREEDY, "temperature": 0.6}}}},
        {"b0_manifest": {"generation": {**GENERATION, "mode": "no_think"}}},
        {"ft_manifest": {"family": {"name": "qwen3-hybrid"}}},
    ],
)
def test_runs_that_do_not_belong_together_exit_2_and_write_nothing(tmp_path, change):
    code, out = calibrate(tmp_path, **change)
    assert code == 2 and not out.exists()


def test_a_manifest_that_is_missing_exits_2(tmp_path):
    items = write_items(tmp_path / "special-dev-b.jsonl")
    b0 = write_run(tmp_path / "runs", "b0", B0, items)
    (tmp_path / "runs" / "b0.manifest.json").unlink()
    code = cal.main(["--b0", str(b0), "--ft", str(write_run(tmp_path / "runs", "ft", FT, items)),
                     "--items", str(items), "--out", str(tmp_path / "c.json")])
    assert code == 2 and not (tmp_path / "c.json").exists()


def test_the_set_must_be_dev_set_b_locked(tmp_path):
    unlocked = write_items(tmp_path / "a" / "special-dev-b.jsonl", lock=False)
    assert calibrate(tmp_path, items=unlocked)[0] == 5
    changed = write_items(tmp_path / "b" / "special-dev-b.jsonl")
    changed.write_bytes(changed.read_bytes() + b"\n")
    assert calibrate(tmp_path, items=changed)[0] == 5
    set_a = write_items(tmp_path / "c" / "special-dev-a.jsonl", [{**i, "split": "a"} for i in ITEMS])
    assert calibrate(tmp_path, items=set_a)[0] == 2
    test = write_items(tmp_path / "d" / "items.jsonl", [{**ITEMS[0], "id": "req-test-0001-s1"}, *ITEMS[1:]])
    assert calibrate(tmp_path, items=test)[0] == 4
    held = write_items(tmp_path / "e" / "dev-v2-ext.jsonl")
    assert calibrate(tmp_path, items=held)[0] == 4


def test_an_existing_result_is_kept_unless_forced(tmp_path):
    code, out = calibrate(tmp_path)
    assert code == 0
    before = out.read_bytes()
    assert calibrate(tmp_path, B0, B0)[0] == 2 and out.read_bytes() == before
    assert calibrate(tmp_path, B0, B0, "--force")[0] == 0 and out.read_bytes() != before


def test_the_fixture_is_well_formed():
    assert all(sd.check_item(i) == [] for i in ITEMS)
    assert copy.deepcopy(ITEMS) == ITEMS

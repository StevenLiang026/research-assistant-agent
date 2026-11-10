"""scripts/check_e2e_tasks.py: the checks a v2 end-to-end task file must pass (docs/m7-计划.md WP-I 6), on
fixture tasks (ids e2e-test-901 and up, slot values no real slot record has) over the recorded corpus, the
real training requests and dev tasks, and a fixture retired set in place of v1's tasks."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from research_agent.canonical import canonical_json
from research_agent.eval import lock as lock_module
from research_agent.eval.e2e import SCENARIO_OF
from research_agent.eval.lock import read_jsonl, write_jsonl, write_lock

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "data" / "eval" / "e2e-dev.jsonl"
REAL_SLOTS = REPO / "data" / "eval" / "v2" / "e2e-slots-v2.jsonl"
TRAIN_REQUESTS = REPO / "data" / "synth" / "requests.jsonl"
# What the public copy leaves out (tests/conftest.py): the checks' default inputs (the training requests and
# their manifest under data/synth, the dev tasks, the recorded corpus), and the v2 slot records and writing
# instructions under data/eval/v2
NEEDS_INPUTS = pytest.mark.needs_data("data/synth", "data/eval/e2e-dev.jsonl", "data/cassettes")
NEEDS_V2 = pytest.mark.needs_data("data/eval/v2")


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_e2e_tasks = load_script("check_e2e_tasks")
draw_e2e_slots = load_script("draw_e2e_slots")


def record(number: int, intent: str, lang: str, slots: dict) -> dict:
    return {"id": f"e2e-test-{number}", "split": "test", "family": f"e2e/{intent}", "intent": intent,
            "lang": lang, "scenario": SCENARIO_OF.get(intent, "normal"), "slots": slots,
            "expect": draw_e2e_slots.expect_for(intent, slots)}


SLOTS = [
    record(901, "find_recent_papers", "zh", {"count": 3, "date_from": "2026-08-02", "topic": "tokenizer",
                                             "topic_id": "tokenization"}),
    record(902, "lookup_paper", "en", {"arxiv_id": "2609.38360"}),
    record(903, "write_note", "en", {"arxiv_id": "2609.21619", "mode": "scaffold"}),
    record(904, "list_notes", "zh", {"since": "2026-09-05"}),
    record(905, "read_section", "en", {"arxiv_id": "2609.10321", "section": "experiments"}),
    record(906, "compare_papers", "zh", {"arxiv_ids": ["2607.17689", "2608.18062"]}),
    record(907, "find_recent_papers", "en", {"categories": ["cs.CL"], "count": 2, "date_from": "2026-07-10",
                                             "date_to": "2026-09-20", "topic": "tokenization",
                                             "topic_id": "tokenization"}),
    record(908, "chit_chat", "en", {}),
    record(909, "needs_clarification", "zh", {}),
]
TEXTS = {
    "e2e-test-901": "最近在关注 tokenizer 的进展,2026-08-02 之后出的论文里给我挑出最有代表性的三篇。",
    "e2e-test-902": "Before I cite 2609.38360, could you check who the authors are and what date it went up?",
    "e2e-test-903": "Please draft a scaffold note for 2609.21619 so I can bring it to Thursday's lab "
                    "meeting.",
    "e2e-test-904": "9 月 5 日以后我都新记了哪些笔记?帮我列一下。",
    "e2e-test-905": "Skim the experiments of 2609.10321 for me and summarize which baselines it compares "
                    "against.",
    "e2e-test-906": "2607.17689 和 2608.18062 放在一起看,两篇的核心做法差在哪里?",
    "e2e-test-907": "Restricting to cs.CL, which two tokenization papers from July 10 to September 20, 2026 "
                    "stand out the most?",
    "e2e-test-908": "Thanks for the help today, I'm logging off now. Talk soon!",
    "e2e-test-909": "上周组会讨论的那篇,能不能帮我整理一下要点?",
}
RETIRED = [{"id": "e2e-test-01", "split": "test", "intent": "chit_chat", "slots": {},
            "text": "OK, that wraps up my reading list for the week, see you after the holiday!"},
           {"id": "e2e-test-02", "split": "test", "intent": "lookup_paper",
            "slots": {"arxiv_id": "2609.05022"}, "text": "这篇 2609.05022 是谁写的?什么时候挂出来的?"}]
SPENT = [{"milestone": m, "purpose": "p", "sha256": "0" * 64, "ts": "2026-10-17T00:00:00+00:00"}
         for m in ("M3", "M5", "M6")]


def tasks() -> list[dict]:
    return [{**copy.deepcopy(r), "text": TEXTS[r["id"]]} for r in SLOTS]


def by_id(items: list[dict], task_id: str) -> dict:
    return next(t for t in items if t["id"] == task_id)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    """The fixture slot records and a retired set (locked, three opens logged) in a folder of their own."""
    root = tmp_path_factory.mktemp("check")
    write_jsonl(SLOTS, root / "e2e-slots-v2.jsonl")
    retired = root / "e2e-test.jsonl"
    write_jsonl(RETIRED, retired)
    write_lock(retired)
    write_jsonl(SPENT, root / "e2e-test-access.jsonl")
    return root


def retired_set(root: Path) -> lock_module.RetiredSet:
    return lock_module.RetiredSet(root / "e2e-test.jsonl", root / "e2e-test-access.jsonl",
                                  root / "e2e-test-retired-reads.jsonl")


@pytest.fixture(scope="module")
def inputs(world):
    return check_e2e_tasks.load_inputs(slots=world / "e2e-slots-v2.jsonl", retired=retired_set(world))


def problems(items: list[dict], inputs, task_id: str, **options) -> list[str]:
    """The reason codes of task_id in a partial check of items; every code of the report is one of REASONS,
    the vocabulary data/eval/v2/e2e-writing.md explains."""
    report = check_e2e_tasks.check(items, inputs, partial=True, **options)
    assert codes([c for found in report.failures.values() for c in found]) <= set(check_e2e_tasks.REASONS)
    return report.failures.get(task_id, [])


def codes(found: list[str]) -> set[str]:
    """The reason codes without their details (near_task:<id>:<similarity> -> near_task)."""
    return {code.split(":")[0] for code in found}


@NEEDS_V2
def test_the_fixture_slots_are_no_real_slot_records():
    real = {canonical_json(r["slots"]) for r in read_jsonl(REAL_SLOTS)}
    assert not [r["id"] for r in SLOTS if r["slots"] and canonical_json(r["slots"]) in real]


# ---- a good batch, and the text checks on the dev tasks ----------------------------------------------


@NEEDS_INPUTS
def test_a_good_batch_passes_every_check(inputs):
    report = check_e2e_tasks.check(tasks(), inputs, partial=True)
    assert report.failures == {} and report.set_problems == [] and report.ok


@NEEDS_INPUTS
def test_the_text_checks_accept_every_dev_task(inputs):
    """The dev tasks are the style the writers follow: every text check passes on them."""
    for task in read_jsonl(DEV):
        assert check_e2e_tasks.text_problems(task, inputs) == [], task["id"]


# ---- the record checks -------------------------------------------------------------------------------


def mutated(task_id: str, change) -> list[dict]:
    items = tasks()
    change(by_id(items, task_id))
    return items


def set_slot(key, value, text=None):
    def change(task):
        task["slots"][key] = value
        if text is not None:
            task["text"] = text
    return change


RECORD_CASES = [  # (the code, the task changed, the change, the id the code is reported under)
    ("malformed", "e2e-test-902", lambda t: t.pop("expect")),
    ("malformed", "e2e-test-908", lambda t: t.update(scenario="normal")),
    ("no_slot_record", "e2e-test-902", lambda t: t.update(id="e2e-test-999"), "e2e-test-999"),
    ("differs_from_slots", "e2e-test-901",
     set_slot("count", 4, TEXTS["e2e-test-901"].replace("三篇", "四篇"))),
    ("differs_from_slots", "e2e-test-902",
     lambda t: t["expect"].update(expected_tools_any=["fetch_fulltext"])),
    ("not_test_split", "e2e-test-902", lambda t: t.update(split="dev")),
    ("topic_outside_split", "e2e-test-901",
     lambda t: (set_slot("topic_id", "hallucination")(t), set_slot("topic", "hallucination")(t),
                t.update(text=t["text"].replace("tokenizer", "hallucination")))),
    ("id_outside_split", "e2e-test-902",
     set_slot("arxiv_id", "2609.27374", TEXTS["e2e-test-902"].replace("2609.38360", "2609.27374"))),
    ("id_not_in_metadata", "e2e-test-902",
     set_slot("arxiv_id", "2609.99999", TEXTS["e2e-test-902"].replace("2609.38360", "2609.99999"))),
    ("date_outside_window", "e2e-test-901",
     set_slot("date_from", "2026-06-20", TEXTS["e2e-test-901"].replace("2026-08-02", "2026-06-20"))),
    ("date_order", "e2e-test-907", set_slot("date_to", "2026-07-05",
                                            TEXTS["e2e-test-907"].replace("September 20", "July 5"))),
    ("too_few_hits", "e2e-test-901",
     set_slot("date_from", "2026-10-05", TEXTS["e2e-test-901"].replace("2026-08-02", "2026-10-05"))),
    ("duplicate_id", "e2e-test-903", lambda t: t.update(id="e2e-test-902"), "e2e-test-902"),
    # a slot of a type the checks cannot read is reported, not a crash
    ("malformed", "e2e-test-902", set_slot("arxiv_id", {"id": "2609.38360"})),
    ("differs_from_slots", "e2e-test-901", set_slot("count", [3])),
    # an id that is no file-name-safe id is never printed: the task is named by its line
    ("malformed", "e2e-test-902", lambda t: t.update(id="find me tokenizer papers, please"), "#2"),
]


@NEEDS_INPUTS
@pytest.mark.parametrize("case", RECORD_CASES, ids=[f"{c[0]}-{c[1]}" for c in RECORD_CASES])
def test_each_record_check_names_its_reason(inputs, case):
    code, task_id, change, *reported = case
    found = problems(mutated(task_id, change), inputs, reported[0] if reported else task_id)
    assert code in codes(found), found


# ---- the text checks ---------------------------------------------------------------------------------


def texted(task_id: str, text: str) -> list[dict]:
    items = tasks()
    by_id(items, task_id)["text"] = text
    return items


TEXT_CASES = [
    ("id_not_in_slots", "e2e-test-902", TEXTS["e2e-test-902"] + " It may build on 2609.33395."),
    ("slot_id_missing", "e2e-test-902", "Before I cite it, could you check who the authors are?"),
    ("slot_id_missing", "e2e-test-906", "2607.17689 和另外那篇放在一起看,两篇的核心做法差在哪里?"),
    ("topic_missing", "e2e-test-901", TEXTS["e2e-test-901"].replace("tokenizer", "分词器")),
    ("other_topic_in_text", "e2e-test-901", TEXTS["e2e-test-901"] + "顺便看看 knowledge distillation。"),
    ("other_topic_in_text", "e2e-test-907",
     TEXTS["e2e-test-907"].replace("tokenization", "tokenization and tokenizer")),
    ("topic_in_text", "e2e-test-908", "Thanks, the tokenizer papers were great. Talk soon!"),
    ("topic_in_text", "e2e-test-904", "9 月 5 日以后我都新记了哪些蒸馏方面的笔记?帮我列一下。"),
    ("category_missing", "e2e-test-907", TEXTS["e2e-test-907"].replace("Restricting to cs.CL, w", "W")),
    ("category_not_in_slots", "e2e-test-901", TEXTS["e2e-test-901"].replace("。", ",只看 cs.LG。")),
    ("date_missing", "e2e-test-901", TEXTS["e2e-test-901"].replace("2026-08-02 之后出的", "近期")),
    ("date_missing", "e2e-test-907",
     TEXTS["e2e-test-907"].replace(" to September 20, 2026", ", 2026, onwards")),
    ("date_not_in_slots", "e2e-test-904",
     TEXTS["e2e-test-904"].replace("9 月 5 日", "9 月 5 日到 9 月 30 日")),
    ("date_not_in_slots", "e2e-test-908", "Thanks for the help today; see you on October 12!"),
    ("date_outside_window", "e2e-test-907", TEXTS["e2e-test-907"].replace("2026", "2025")),
    ("count_missing", "e2e-test-901", TEXTS["e2e-test-901"].replace("三篇", "几篇")),
    ("count_missing", "e2e-test-907", TEXTS["e2e-test-907"].replace("which two", "which")),
    ("section_missing", "e2e-test-905", TEXTS["e2e-test-905"].replace("the experiments of", "the paper")),
    ("note_missing", "e2e-test-903", TEXTS["e2e-test-903"].replace("scaffold note", "scaffold summary")),
    ("scaffold_missing", "e2e-test-903", TEXTS["e2e-test-903"].replace("scaffold note", "short note")),
    ("note_in_text", "e2e-test-902", TEXTS["e2e-test-902"].replace("?", " and add a note on it?")),
    ("note_in_text", "e2e-test-906", TEXTS["e2e-test-906"] + "顺便记个笔记。"),
    ("note_in_text", "e2e-test-908", "Thanks for the help today, I'll tidy up my notes and log off."),
    ("vague_date", "e2e-test-907",
     TEXTS["e2e-test-907"].replace("from July 10 to September 20, 2026", "from early July to mid-September")),
    ("vague_date", "e2e-test-904", TEXTS["e2e-test-904"].replace("9 月 5 日", "9 月初")),
    ("line_break", "e2e-test-908", "Thanks for the help today.\nTalk soon!"),
    ("too_long", "e2e-test-908", " ".join([TEXTS["e2e-test-908"]] * 5)),
    ("wrong_language", "e2e-test-904", "Which notes did I add on or after September 5?"),
    ("wrong_language", "e2e-test-908", "Thanks for the help today, 再见!"),
]


@NEEDS_INPUTS
@pytest.mark.parametrize(("code", "task_id", "text"), TEXT_CASES)
def test_each_text_check_names_its_reason(inputs, code, task_id, text):
    found = problems(texted(task_id, text), inputs, task_id)
    assert code in codes(found), found


@NEEDS_INPUTS
def test_a_whole_month_or_a_natural_date_expresses_a_slot_date(inputs):
    """Dates may be written as ISO, as a natural date in either language, or as a whole month for the first
    day of a from date and the last day of a to date (the dev tasks do all three)."""
    natural = ("Restricting to cs.CL, which two tokenization papers from 10 July to 20 Sept 2026 stand "
               "out?")
    iso = ("Restricting to cs.CL, which two tokenization papers between 2026-07-10 and 2026-09-20 stand "
           "out?")
    for text in (natural, iso):
        assert problems(texted("e2e-test-907", text), inputs, "e2e-test-907") == [], text
    month = copy.deepcopy(by_id(tasks(), "e2e-test-907"))
    month["slots"].update(date_from="2026-08-01", date_to="2026-09-30")
    english = ("Restricting to cs.CL, which two tokenization papers from August to September 2026 stand "
               "out?")
    for text in (english, "只看 cs.CL,2026 年 8 月到 9 月之间的 tokenization 论文里哪两篇最值得读?"):
        assert check_e2e_tasks.text_problems({**month, "text": text, "lang": "zh" if "只" in text else "en"},
                                             inputs) == [], text
    assert "date_missing:date_to" in check_e2e_tasks.text_problems(
        {**month, "text": "Which two tokenization papers in cs.CL since August 2026 stand out?"}, inputs)


@NEEDS_INPUTS
def test_a_part_of_a_month_is_a_vague_date_even_where_the_whole_month_would_do(inputs):
    month = copy.deepcopy(by_id(tasks(), "e2e-test-907"))
    month["slots"].update(date_from="2026-08-01", date_to="2026-09-30")
    texts = {"en": "Restricting to cs.CL, which two tokenization papers from early August to the end of "
                   "September 2026 stand out?",
             "zh": "只看 cs.CL,2026 年 8 月初到 9 月底之间的 tokenization 论文里哪两篇最值得读?"}
    for lang, text in texts.items():
        assert check_e2e_tasks.text_problems({**month, "text": text, "lang": lang}, inputs) == ["vague_date"]


FIND = {"count": 3, "topic": "tokenizer", "topic_id": "tokenization"}
DATE_FORMS = [  # the date forms data/eval/v2/e2e-writing.md allows, and the ones it forbids, with their code
    ("en", {"date_from": "2026-09-15"}, "Find three tokenizer papers posted after Sept 15, 2026.", None),
    ("en", {"date_from": "2026-09-15"}, "Find three tokenizer papers posted after 15 September.", None),
    ("zh", {"date_from": "2026-09-15"}, "找三篇 2026 年 9 月 15 号以后的 tokenizer 论文。", None),
    ("en", {"date_from": "2026-07-23", "date_to": "2026-08-15"},
     "Three tokenizer papers from July 23 to August 15, 2026, please.", None),
    ("zh", {"date_from": "2026-08-01"}, "2026 年 8 月起的 tokenizer 论文挑 3 篇。", None),
    ("en", {"date_from": "2026-07-23", "date_to": "2026-08-31"},
     "Three tokenizer papers from 2026-07-23 through August.", None),
    ("zh", {"date_from": "2026-07-23", "date_to": "2026-09-30"},
     "从 2026-07-23 到 9 月的 tokenizer 论文挑 3 篇。", None),
    ("zh", {"date_from": "2026-07-01", "date_to": "2026-07-31"}, "7 月份的 tokenizer 论文挑 3 篇。", None),
    ("en", {"date_from": "2026-09-15"}, "Find three tokenizer papers from last week.", "date_missing"),
    ("zh", {"date_from": "2026-09-01"}, "上个月以来的 tokenizer 论文挑 3 篇。", "date_missing"),
    ("zh", {"date_from": "2026-09-15"}, "九月十五日以来的 tokenizer 论文挑 3 篇。", "date_missing"),
    ("en", {"date_from": "2026-08-01"}, "Three tokenizer papers since mid-August.", "vague_date"),
    ("zh", {"date_from": "2026-08-01"}, "8 月中旬以来的 tokenizer 论文挑 3 篇。", "vague_date"),
    ("en", {"date_from": "2026-09-15"}, "Three tokenizer papers since September 15, 2025.",
     "date_outside_window"),
]


@NEEDS_INPUTS
@pytest.mark.parametrize(("lang", "dates", "text", "code"), DATE_FORMS)
def test_the_date_forms_of_the_writing_instructions(inputs, lang, dates, text, code):
    task = {"id": "e2e-test-999", "intent": "find_recent_papers", "lang": lang, "slots": {**FIND, **dates},
            "text": text}
    found = check_e2e_tasks.text_problems(task, inputs)
    assert (found == [] if code is None else code in codes(found)), found


@NEEDS_INPUTS
def test_the_topic_is_its_slot_phrase_not_a_longer_phrase_of_the_topic(inputs):
    """The recorded hits were counted for the slot phrase: a longer phrase of the topic that contains it
    (another search) is reported; a plural of the phrase itself is not."""
    task = copy.deepcopy(by_id(tasks(), "e2e-test-907"))
    for topic, topic_id, longer in (("distillation", "distillation", "knowledge distillation"),
                                    ("distillation", "distillation", "on-policy distillation"),
                                    ("looped transformer", "looped-transformers", "looped transformers")):
        task["slots"].update(topic=topic, topic_id=topic_id)
        good = (f"Restricting to cs.CL, which two {topic} papers from July 10 to September 20, 2026 stand "
                "out?")
        assert check_e2e_tasks.text_problems({**task, "text": good}, inputs) == [], good
        found = check_e2e_tasks.text_problems({**task, "text": good.replace(topic, longer)}, inputs)
        assert found == ["topic_in_longer_phrase"], (longer, found)
    task["slots"].update(topic="tokenizer", topic_id="tokenization")
    plural = "Restricting to cs.CL, which two tokenizers from July 10 to September 20, 2026 stand out?"
    assert check_e2e_tasks.text_problems({**task, "text": plural}, inputs) == []


@NEEDS_INPUTS
def test_a_section_is_named_as_a_word_not_found_inside_another(inputs):
    task = copy.deepcopy(by_id(tasks(), "e2e-test-905"))
    task["slots"]["section"] = "introduction"
    named = "Skim the intro of 2609.10321 for me and summarize the problem it sets out to solve."
    assert check_e2e_tasks.text_problems({**task, "text": named}, inputs) == []
    inside = "Tell me what 2609.10321 introduces and which problem it sets out to solve."
    assert check_e2e_tasks.text_problems({**task, "text": inside}, inputs) == ["section_missing"]
    task["slots"]["section"] = "method"
    assert check_e2e_tasks.text_problems({**task, "text": "Walk me through the methodology of 2609.10321."},
                                         inputs) == []


# ---- the similarity checks ---------------------------------------------------------------------------


@NEEDS_INPUTS
def test_a_task_near_a_training_request_another_task_a_dev_task_or_a_retired_task_is_named(inputs):
    train = next(r for r in map(json.loads, TRAIN_REQUESTS.read_text(encoding="utf-8").splitlines())
                 if r["split"] == "train")
    found = problems(texted("e2e-test-908", train["text"]), inputs, "e2e-test-908")
    assert f"near_train:{train['id']}" in {":".join(c.split(":")[:2]) for c in found}, found
    found = problems(texted("e2e-test-908", TEXTS["e2e-test-902"]), inputs, "e2e-test-908")
    assert "near_task:e2e-test-902" in {":".join(c.split(":")[:2]) for c in found}, found
    dev = next(t for t in read_jsonl(DEV) if t["intent"] == "chit_chat" and t["lang"] == "en")
    found = problems(texted("e2e-test-908", dev["text"]), inputs, "e2e-test-908")
    assert f"near_task:{dev['id']}" in {":".join(c.split(":")[:2]) for c in found}, found
    found = problems(texted("e2e-test-908", RETIRED[0]["text"].replace("holiday", "holidays")), inputs,
                     "e2e-test-908")
    near = [c for c in found if c.startswith("near_retired:")]
    assert near and float(near[0].split(":")[1]) >= 0.6, found
    assert not any(RETIRED[0]["text"][:20] in c for c in found)  # a similarity, never the retired text


# ---- the set and the batch ---------------------------------------------------------------------------


@NEEDS_INPUTS
def test_the_whole_file_is_counted_against_the_slot_records(inputs):
    report = check_e2e_tasks.check(tasks(), inputs)
    assert report.ok and report.set_problems == []
    short = tasks()[:-2]
    report = check_e2e_tasks.check(short, inputs)
    assert not report.ok
    assert report.failures == {"e2e-test-908": ["missing_task"], "e2e-test-909": ["missing_task"]}
    assert set(report.set_problems) == {"size:7/9", "intent:chit_chat:0/1", "intent:needs_clarification:0/1",
                                        "lang:zh:3/4", "lang:en:4/5"}
    assert check_e2e_tasks.check(short, inputs, partial=True).ok  # a batch is not the whole set


@NEEDS_INPUTS
def test_a_batch_is_compared_with_the_other_drafts_and_must_cover_its_slots(world, tmp_path, capsys):
    drafts = tmp_path / "e2e-drafts"
    drafts.mkdir()
    items = tasks()
    write_jsonl(SLOTS[:5], drafts / "slots-01.jsonl")
    write_jsonl(items[:4], drafts / "tasks-01.jsonl")  # e2e-test-905 left out
    other = {**items[5], "id": "e2e-test-905", "text": TEXTS["e2e-test-902"]}
    write_jsonl([other, items[6]], drafts / "tasks-02.jsonl")
    (drafts / "tasks-03.jsonl").write_text("not json\n", encoding="utf-8")
    code = check_e2e_tasks.main([str(drafts / "tasks-01.jsonl"), "--partial", "--slots",
                                 str(world / "e2e-slots-v2.jsonl")], retired=retired_set(world))
    out, err = capsys.readouterr()
    assert code == 1
    assert "e2e-test-905: missing_task" in out
    assert "e2e-test-902: near_task:e2e-test-905:" in out  # the same text in another draft
    assert "tasks-03.jsonl" in err  # an unreadable draft is named, not compared
    write_jsonl(items[:5], drafts / "tasks-01.jsonl")
    (drafts / "tasks-02.jsonl").unlink()
    assert check_e2e_tasks.main([str(drafts / "tasks-01.jsonl"), "--partial", "--slots",
                                 str(world / "e2e-slots-v2.jsonl")], retired=retired_set(world)) == 0
    capsys.readouterr()
    write_jsonl([{**items[6], "id": "e2e-test-902"}], drafts / "tasks-02.jsonl")
    assert check_e2e_tasks.main([str(drafts / "tasks-01.jsonl"), "--partial", "--slots",
                                 str(world / "e2e-slots-v2.jsonl")], retired=retired_set(world)) == 1
    assert "e2e-test-902: duplicate_id" in capsys.readouterr().out


# ---- the script --------------------------------------------------------------------------------------


@NEEDS_INPUTS
def test_the_cli_prints_ids_and_reason_codes_only_and_exits_by_the_result(world, tmp_path, capsys):
    path = tmp_path / "tasks.jsonl"
    items = tasks()
    by_id(items, "e2e-test-901")["text"] = TEXTS["e2e-test-901"].replace("三篇", "几篇")
    write_jsonl(items, path)
    reads = world / "e2e-test-retired-reads.jsonl"
    before = len(read_jsonl(reads)) if reads.exists() else 0
    argv = [str(path), "--slots", str(world / "e2e-slots-v2.jsonl")]
    assert check_e2e_tasks.main(argv, retired=retired_set(world)) == 1
    out = capsys.readouterr().out
    assert "e2e-test-901: count_missing" in out and "FAIL" in out
    assert not [text for text in TEXTS.values() if text[:12] in out]
    assert len(read_jsonl(reads)) == before + 1  # one retired read per run, logged
    assert check_e2e_tasks.main([*argv, "--json"], retired=retired_set(world)) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False and report["failures"] == {"e2e-test-901": ["count_missing"]}
    assert report["tasks"] == 9 and report["set"] == []
    write_jsonl(tasks(), path)
    assert check_e2e_tasks.main(argv, retired=retired_set(world)) == 0
    assert "OK" in capsys.readouterr().out
    path.write_text("not json\n", encoding="utf-8")
    assert check_e2e_tasks.main(argv, retired=retired_set(world)) == 2
    assert check_e2e_tasks.main([str(tmp_path / "missing.jsonl")], retired=retired_set(world)) == 2
    not_retired = lock_module.RetiredSet(world / "e2e-test.jsonl", tmp_path / "access.jsonl",
                                         tmp_path / "reads.jsonl")
    write_jsonl(tasks(), path)
    assert check_e2e_tasks.main(argv, retired=not_retired) == 2  # the retired read is refused: no check
    assert "not retired" in capsys.readouterr().err


def test_other_test_material_is_refused_before_it_is_read(world, tmp_path, capsys):
    """v1's test sets and the handwritten files are refused by location, before any read (the conftest
    guard would fail this test on an open); the v2 end-to-end test set itself is what the checker checks."""
    refuse = check_e2e_tasks.refuse_other_test_material
    for name in ("data/eval/e2e-test.jsonl", "data/eval/test.jsonl", "data/eval/handwritten.jsonl",
                 "data/eval/v2/test-v2.jsonl", "data/eval/v2/handwritten-v2.jsonl"):
        with pytest.raises(lock_module.RefusedTestSet):
            refuse(REPO / name)
    assert refuse(REPO / "data" / "eval" / "v2" / "e2e-test-v2.jsonl") is None
    assert refuse(tmp_path / "tasks-01.jsonl") is None
    for argv in ([str(REPO / "data" / "eval" / "e2e-test.jsonl")],
                 [str(tmp_path / "tasks.jsonl"), "--slots", str(REPO / "data" / "eval" / "test.jsonl")]):
        assert check_e2e_tasks.main(argv, retired=retired_set(world)) == 2
        assert "locked test set" in capsys.readouterr().err


@NEEDS_V2
def test_the_writing_instructions_explain_every_reason_code_and_give_the_command():
    writing = (REPO / check_e2e_tasks.WRITING).read_text(encoding="utf-8")
    missing = [code for code in (*check_e2e_tasks.REASONS, *check_e2e_tasks.SET_REASONS)
               if f"`{code}`" not in writing]
    assert missing == []
    assert "scripts\\check_e2e_tasks.py" in writing and "--partial" in writing
    assert all(f"slots-{n:02d}.jsonl" in writing for n in range(1, 11))


def test_the_default_inputs_are_the_v2_slots_the_three_training_files_and_v1s_retired_tasks():
    assert check_e2e_tasks.SLOTS == Path("data/eval/v2/e2e-slots-v2.jsonl")
    assert check_e2e_tasks.TRAIN_FILES == (Path("data/synth/requests.jsonl"),
                                           Path("data/synth/requests-m6.jsonl"),
                                           Path("data/synth/requests-m7-train.jsonl"))
    retired = check_e2e_tasks.default_retired()
    assert (retired.path, retired.access_log, retired.reads_log) == (
        REPO / lock_module.E2E_TEST_SET, REPO / lock_module.E2E_TEST_ACCESS_LOG,
        REPO / lock_module.E2E_TEST_RETIRED_READS)

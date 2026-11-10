"""scripts/build_special_mix.py (M8 WP1, WP2), offline: the exclusions, the record screen, the draw (counts,
disjoint records, held-out families and function names, the share of English records under Chinese prompts)
and the CLI (outputs, locks, sidecar binding, determinism, refusals).

The pools are stubbed in place of the Hugging Face cache and the benchmarks in place of BFCL and ACEBench.
The last test reads the real outputs when they exist (they are development files, not test material).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

import pytest
from test_synth import load_script

from research_agent.data import special_mix as sm
from research_agent.data.reasoning import ReasoningIndex, turns
from research_agent.eval.lock import check_lock, find_test_material

bs = load_script("build_special_mix")
REPO = Path(__file__).resolve().parents[2]
CLONE = REPO / "runs" / "third_party" / "ACEBench"


def tool(name: str, properties: dict, required: list[str] | None = None, description: str = "") -> dict:
    parameters: dict = {"type": "object", "properties": properties}
    if required is not None:
        parameters["required"] = required
    return {"type": "function", "function": {"name": name, "description": description or f"{name} helps",
                                             "parameters": parameters}}


def sample(source: str, sid: str, user: str, calls: list[tuple[str, dict]], tools: list[dict],
           text: str = "") -> dict:
    messages: list[dict] = [{"role": "user", "content": user}]
    if calls:
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": n, "arguments": json.dumps(a, ensure_ascii=False)}}
            for i, (n, a) in enumerate(calls)]})
    else:
        messages.append({"role": "assistant", "content": text})
    return {"messages": messages, "tools": tools, "source": source, "source_id": sid}


# Per booking record, a sentence of five words of one of twelve occasions: the bookings' requests fall into
# twelve templates (special_mix.same_template), as real requests do, rather than one, which the development
# sets would hold only twice (bs.DEV_TEMPLATE_LIMIT)
OCCASIONS = (("anniversary", "celebration", "garden", "breakfast", "quiet"),
             ("conference", "keynote", "projector", "badge", "lanyard"),
             ("wedding", "bridesmaid", "bouquet", "photographer", "ceremony"),
             ("marathon", "runner", "medal", "stretching", "hydration"),
             ("orchestra", "violin", "rehearsal", "conductor", "concert"),
             ("hackathon", "laptop", "prototype", "pitch", "mentor"),
             ("graduation", "diploma", "parents", "gown", "speech"),
             ("pilgrimage", "cathedral", "candle", "choir", "chapel"),
             ("safari", "binoculars", "ranger", "savanna", "jeep"),
             ("festival", "lantern", "parade", "drummer", "costume"),
             ("retreat", "meditation", "yoga", "silence", "cushion"),
             ("tournament", "chess", "opening", "endgame", "trophy"))


def booking(source: str, i: int) -> dict:
    name = f"book_room_{source}_{i}"
    room = ("single", "double", "suite")[i % 3]
    nights, floor, day = 2 + i % 5, 10 + i % 5, 15 + i % 5
    words = OCCASIONS[i % len(OCCASIONS)]
    user = (f"Book a {room} room for guest Guest{i}X on 2024-05-{day} for {nights} nights, floor {floor} "
            f"with sea{i} view, mail g{i}@example.org. It is for the {words[0]} {words[1]}; we want "
            f"{words[2]}, {words[3]} and {words[4]}.")
    options = {"type": "object", "description": "Room options",
               "properties": {"floor": {"type": "integer"}, "view": {"type": "string"}}}
    tools = [tool(name, {"guest": {"type": "string", "description": "Guest name"},
                         "date": {"type": "string", "format": "date", "description": "Arrival day"},
                         "nights": {"type": "integer", "minimum": 1, "maximum": 30},
                         "room": {"type": "string", "enum": ["single", "double", "suite"]},
                         "options": options,
                         "email": {"type": "string", "description": "email address of the guest"}},
                  ["guest", "date", "nights", "options"]),
             tool(f"lookup_forecast_{source}_{i}", {"city": {"type": "string"}}, ["city"])]
    args = {"guest": f"Guest{i}X", "date": f"2024-05-{day}", "nights": nights, "room": room,
            "options": {"floor": floor, "view": f"sea{i}"}, "email": f"g{i}@example.org"}
    return sample(source, str(i), user, [(name, args)], tools)


def lookup(source: str, i: int) -> dict:
    """An English record without a top-level object, of one of the twelve occasions: a request a development
    irrelevant item may pair (the records able to lose an object stay for training)."""
    words = OCCASIONS[i % len(OCCASIONS)]
    name, day = f"lookup_schedule_{source}_{i}", f"2024-07-{10 + i % 9}"
    find = tool(name, {"city": {"type": "string", "description": "City"},
                       "date": {"type": "string", "format": "date", "description": "Day"}}, ["city", "date"])
    user = f"Look up the {words[0]} {words[1]} schedule for Town{i}W on {day}."
    return sample(source, f"s{i}", user, [(name, {"city": f"Town{i}W", "date": day})], [find])


OWNERS = ("张伟", "王芳", "李娜", "刘洋", "陈静", "杨磊", "赵敏", "黄强", "周杰", "吴琳")
# a project word per record, so that the templated requests are no near-duplicates of each other
PROJECTS = ("星河", "远航", "磐石", "青松", "明月", "长风", "飞鸿", "云帆", "金桥", "银杏")
# a purpose per pair of deepexi records (and per folder record), sharing no character pair with another, so
# that the Chinese requests fall into twelve templates per kind of record (OCCASIONS)
PURPOSES = ("季度核算专项", "年度审计复盘", "春季促销筹备", "客户回访整理", "仓库盘点核对",
            "新员工入职培训", "产品发布预热", "供应商资质审查", "门店巡检记录", "跨部门协作周会",
            "设备维护排期", "市场调研分析")


def deepexi(i: int) -> dict:
    if i % 2:
        name = f"DescribeTaskQueue{i}"
        tools = [tool(name, {"ProjectId": {"type": "integer", "description": "工作空间ID。"},
                             "Status": {"type": "string",
                                        "description": "状态,取值:RUNNING、STOPPED、FAILED。"},
                             "PageSize": {"type": "integer", "description": "每页条数,最大值为100。"},
                             "StartDate": {"type": "string", "description": "开始日期,格式为yyyy-MM-dd。"},
                             "Owner": {"type": "string", "description": "负责人"}}),
                 tool(f"CreateTaskFolder{i}", {"Name": {"type": "string", "description": "目录名"}})]
        owner = OWNERS[i % 10] + OWNERS[(i // 10) % 10][1]
        project = PROJECTS[i % 10] + PROJECTS[(i // 10) % 10]
        user = (f"查询{project}项目工作空间{4000 + i}里负责人为{owner}且状态为RUNNING的任务,"
                f"每页{20 + i % 5}条,开始日期2024-03-{10 + i % 9}。")
        args = {"ProjectId": 4000 + i, "Owner": owner, "Status": "RUNNING", "PageSize": 20 + i % 5,
                "StartDate": f"2024-03-{10 + i % 9}"}
    else:
        name = f"SendInvoiceMail{i}"
        tools = [tool(name, {"InvoiceNo": {"type": "integer", "description": "发票号。"},
                             "Title": {"type": "string", "description": "发票抬头"},
                             "Copies": {"type": "integer", "description": "份数,取值范围:1~10。"}}),
                 tool(f"ListInvoiceBox{i}", {"Size": {"type": "integer"}})]
        title = OWNERS[i % 10] + "工作室" + OWNERS[(i // 10) % 10][1]
        user = f"把发票{70000 + i}寄出,抬头写{title},要{1 + i % 9}份。"
        args = {"InvoiceNo": 70000 + i, "Title": title, "Copies": 1 + i % 9}
    user += f"这是{PURPOSES[(i // 2) % len(PURPOSES)]}的需要。"
    return sample("deepexi", str(i), user, [(name, args)], tools)


def no_call(source: str, i: int) -> dict:
    if source == "toolace":
        return sample(source, f"n{i}", f"Book me a flight to Rome number {i}, please.", [],
                      [tool(f"currency_rate_{i}", {"base": {"type": "string"}}, ["base"])],
                      text="None of the provided functions can be used to book a flight.")
    if source == "glaive_zh":
        return sample(source, f"n{i}", f"你能帮我订一张去纽约的机票吗?编号{i}", [],
                      [tool(f"calculate_gpa_{i}", {"grades": {"type": "array", "items": {"type": "number"}}},
                            ["grades"])],
                      text="很抱歉,我无法帮您预订机票。我的功能仅限于计算平均绩点。")
    return sample(source, f"n{i}", f"Can you order pizza number {i} for me?", [],
                  [tool(f"calculate_tip_{source}_{i}", {"bill": {"type": "number"}}, ["bill"])],
                  text="I'm sorry, but I can't order pizza. My capabilities are limited to tips.")


def pools(n: int = 120) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {name: [] for name in bs.SOURCES}
    for source in ("toolace", "glaive_v2", "hermes"):
        out[source] = [booking(source, i) for i in range(n)] + [lookup(source, i) for i in range(n // 3)]
    out["toolace"] += [no_call("toolace", i) for i in range(30)]
    out["glaive_v2"] += [no_call("glaive_v2", i) for i in range(30)]
    out["deepexi"] = [deepexi(i) for i in range(n + 60)]
    out["glaive_zh"] = [no_call("glaive_zh", i) for i in range(30)]
    return out


TEMPLATES = ("Please answer the user in one short sentence",)


def benchmarks(names: set[str] = frozenset(), grams: set[str] = frozenset(),
               templates: tuple[str, ...] = TEMPLATES) -> bs.Benchmarks:
    return bs.Benchmarks(index=bs.SimpleIndex(set(grams), [{"benchmark": "stub", "path": "x"}]),
                         names=frozenset(n.casefold() for n in names), grams=frozenset(),
                         windows=sm.TemplateWindows.from_texts(templates), files=[])


# ---- exclusions ---------------------------------------------------------------------------------------


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_used_records_reads_ft3_files_the_exclusion_files_and_the_probe(tmp_path):
    public = tmp_path / "format-mix.jsonl"
    write_jsonl(public, [{"source": "toolace", "source_id": "7", "messages": []}])
    synth = tmp_path / "synth.jsonl"
    write_jsonl(synth, [{"id": "req-1", "messages": []}])
    ft3 = tmp_path / "ft3.json"
    ft3.write_text(json.dumps({"config": {"args": {"data": [str(public), str(synth)]}}}), encoding="utf-8")
    prompts = tmp_path / "replay.prompts.jsonl"
    write_jsonl(prompts, [{"source": "glaive_v2", "source_id": "9"}])
    probe = tmp_path / "probe.json"
    probe.write_text(json.dumps({"source_ids": {"hermes": ["3"], "deepexi": ["5"]}}), encoding="utf-8")
    ids, account = bs.used_records(ft3, [prompts, tmp_path / "absent.jsonl"], probe)
    assert ids == {("toolace", "7"), ("glaive_v2", "9"), ("hermes", "3"), ("deepexi", "5")}
    assert [f["records"] for f in account["ft3_files"]] == [1, 0]
    assert account["files"][1]["present"] is False and account["probe"]["ids"] == 2
    assert account["total"] == 4


def test_load_pools_records_every_source_file_and_drawn_file_with_its_sha256(tmp_path, monkeypatch):
    cache, raw = tmp_path / "cache", tmp_path / "raw"
    cache.mkdir()
    raw.mkdir()
    for name in bs.SOURCES:
        (cache / f"{name}.json").write_text(f'["{name}"]', encoding="utf-8")
        (raw / f"{name}.jsonl").write_text("", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"sources": {name: {"repo": f"hub/{name}", "revision": "abc123",
                                                       "per_file": [{"file": f"{name}.json"}]}
                                                for name in bs.SOURCES}}), encoding="utf-8")
    monkeypatch.setattr(bs.public_pool, "load_unused",
                        lambda name, entry, raw_dir: ([], {"rows": 0, "drawn": 0, "unused": 0}))
    monkeypatch.setattr(bs.public_pool, "cached_file",
                        lambda repo, filename, revision: cache / filename)
    _, accounts = bs.load_pools(manifest, raw)
    for name in bs.SOURCES:
        assert accounts[name]["revision"] == "abc123"
        assert accounts[name]["files"] == [{"file": f"{name}.json",
                                            "sha256": hashlib.sha256(f'["{name}"]'.encode()).hexdigest()}]
        assert accounts[name]["drawn_file"] == {"path": (raw / f"{name}.jsonl").as_posix(),
                                                "sha256": hashlib.sha256(b"").hexdigest()}


def test_screen_leaves_out_used_named_marked_windowed_and_overlapping_records():
    clean = booking("hermes", 1)
    used = booking("hermes", 2)
    named = booking("hermes", 3)
    marked = booking("hermes", 4)
    marked["messages"][0]["content"] += " Run [lookup(x=1)] too."
    windowed = booking("hermes", 5)
    windowed["tools"][0]["function"]["description"] = "Please answer the user in one short sentence."
    overlapping = booking("hermes", 6)
    gram = min(sm_grams(overlapping["messages"][0]["content"]))
    described = booking("hermes", 7)
    described["tools"][1]["function"]["description"] = "Call it as [lookup(city='x')]."
    parameter = booking("hermes", 8)
    parameter["tools"][1]["function"]["parameters"]["properties"]["ApiName"] = {"type": "string"}
    keyword = booking("hermes", 9)  # ApiName inside an identifier (D6)
    keyword["tools"][1]["function"]["parameters"]["properties"]["ApiNameKeyword"] = {"type": "string"}
    called = booking("hermes", 10)  # in the gold call's arguments
    arguments = json.loads(called["messages"][1]["tool_calls"][0]["function"]["arguments"])
    called["messages"][1]["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {**arguments, "guest": "apinameX"})
    vocabulary = booking("hermes", 11)
    bench = benchmarks(names={"book_room_hermes_3"}, grams={gram})
    records = [clean, used, named, marked, windowed, overlapping, described, parameter, keyword, called,
               vocabulary]
    kept, account = bs.screen({"hermes": records}, {("hermes", "2")}, bench,
                              vocabulary=frozenset({("hermes", "11")}))
    assert [s["source_id"] for s in kept["hermes"]] == ["1"]
    assert account["hermes"] == {"candidates": 11, "used": 1, "vocabulary": 1, "apiname": 3,
                                 "acebench_name": 1, "acebench_text": 2, "template_window": 1, "overlap": 1,
                                 "kept": 1}


def test_screen_leaves_out_a_gold_call_whose_name_holds_apiname():
    # D6: the gold call's name counts as well as its arguments, even when no listed function carries it
    record = booking("hermes", 12)
    record["messages"][1]["tool_calls"][0]["function"]["name"] = "lookup_apinames"
    kept, account = bs.screen({"hermes": [record]}, set(), benchmarks())
    assert kept["hermes"] == [] and account["hermes"]["apiname"] == 1


def sm_grams(text: str) -> set[str]:
    from research_agent.data.overlap import text_ngrams
    return text_ngrams(text)


def test_mentions_finds_identifier_like_names_as_whole_words():
    found = bs.mentions({"ListAlgorithms", "get_weather", "translate", "Get Weather", "查询工具"})
    assert found.search("您可以调用ListAlgorithms获取算法列表。")
    assert found.search("Use get_weather first.") and not found.search("Use get_weather_v2 first.")
    assert not found.search("Please translate this.")  # a plain word is no identifier: not looked for
    assert not found.search("Get weather for Oslo.")
    assert found.search("请用查询工具。")
    assert bs.mentions(set()) is None
    dotted = bs.mentions({"mealprep.get_recipes"})
    assert dotted.search("Try mealprep.get_recipes now.")
    assert not dotted.search("Try mealprep.get_recipes_v2.")


def test_a_training_request_naming_a_dev_function_is_dropped_before_its_partners():
    request = deepexi(1)
    request["messages"][0]["content"] += " 参考ListJobs。"
    partners = [deepexi(2 * i) for i in range(1, 6)]
    drawer = bs.Drawer({"deepexi": [*partners, request]}, 5, benchmarks(), {})
    drawer.dev_names = {"ListJobs"}
    assert drawer.attempt("train", "irrelevant", "zh", {}, request, random.Random(1)) is None
    assert drawer.rejected == {"dev_name_mentioned": 1}


def test_partners_skip_records_that_cannot_be_partners():
    front = [no_call("glaive_v2", i) for i in range(60)]  # what earlier kinds leave at a queue's front
    good = [booking("glaive_v2", i) for i in range(5)]
    drawer = bs.Drawer({"glaive_v2": front + good}, 5, benchmarks(), {})
    offered = list(drawer.partners("train", "glaive_v2", good[0], "switch",
                                   lambda s: sm.first_step(s) is not None))
    assert [s["source_id"] for s in offered] == ["1", "2", "3", "4"]
    # the unsuitable front is passed for good, by this split
    assert drawer.pointers[("train", "glaive_v2", "partner", "switch")] == 60


def test_records_a_development_set_passes_over_stay_open_to_training():
    queue = [routed(i) for i in range(5)] + [booking("glaive_v2", i) for i in range(5)]
    drawer = bs.Drawer({"glaive_v2": queue}, 5, benchmarks(), {}, closed=frozenset({"plan_route"}))
    dev = drawer.slot("b", "valid_verbatim", "en", {}, ("glaive_v2",), 0)
    assert dev is not None and dev.core.api != "plan_route"  # B passes over the frequent function
    train = drawer.slot("train", "valid_verbatim", "en", {}, ("glaive_v2",), 0)
    assert train is not None and train.core.api == "plan_route" and train.core.records == ("r0",)


def test_partners_a_development_set_passes_over_stay_open_to_training():
    queue = [routed(i) for i in range(3)] + [booking("glaive_v2", i) for i in range(3)]
    drawer = bs.Drawer({"glaive_v2": queue}, 5, benchmarks(), {}, closed=frozenset({"plan_route"}))
    other = booking("glaive_v2", 9)

    def offered(split: str) -> list[str]:
        def suitable(s: dict) -> bool:
            return sm.single_call(s, drawer.avoid(split)) is not None

        return [s["source_id"] for s in drawer.partners(split, "glaive_v2", other, "switch", suitable)]

    assert offered("b") == ["0", "1", "2"]
    assert offered("train")[:3] == ["r0", "r1", "r2"]


def test_a_partner_whose_item_fails_its_checks_is_skipped():
    def sms(sid: str, description: str) -> dict:
        return sample("deepexi", sid, "列出全部。", [(f"SendSmsBatch{sid}", {})],
                      [tool(f"SendSmsBatch{sid}", {"Size": {"type": "integer"}}, description=description)])

    poisoned, clean = sms("p1", "调用ListJobs获取短信模板。"), sms("p2", "发送短信。")
    request = deepexi(1)
    drawer = bs.Drawer({"deepexi": [poisoned, clean, request]}, 5, benchmarks(), {})
    drawer.dev_names = {"ListJobs"}  # as if a development item had shown it
    item = drawer.attempt("train", "irrelevant", "zh", {}, request, random.Random(1))
    assert item is not None and item.core.records == ("1", "p2")
    assert drawer.rejected["dev_name_mentioned"] == 1


def test_switch_items_are_drawn_for_training(drawn):
    splits, account = drawn
    modes = [i.core.variant["mode"] for i in splits["train"] if i.core.kind == "valid_multiturn"]
    assert "switch" in modes and "change" in modes
    assert not any("switch" in fell and "->change" in fell for fell in json_values(account["fallbacks"]))


def json_values(nested: dict) -> list[str]:
    return [label for where in nested.values() for label in where]


# ---- the draw -----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def drawn():
    p = pools()
    return bs.build(p, seed=5, scale=0.05, bench=benchmarks(), banks=bs.value_banks(p))


def test_draw_meets_the_scaled_plan(drawn):
    splits, account = drawn
    for split, plan in (("train", sm.PLAN), ("a", sm.DEV_PLAN["a"]), ("b", sm.DEV_PLAN["b"])):
        for kind, counts in plan.items():
            for lang, n in counts.items():
                got = sum(i.core.kind == kind and i.lang == lang and not bs.extra_item(i)
                          for i in splits[split])
                assert got == bs.scaled(n, 0.05), (split, kind, lang)
    assert account["short"] == {}
    assert not any(bs.extra_item(i) for split in ("a", "b") for i in splits[split])  # training's only


# The development schedules as the locked build drew them (computed from the code of f023317 before revision
# 1): every development kind and language at the full and a twentieth scale, with the next draw of each
# generator after it.
LOCKED_SCHEDULES = "d2944523313ff410e66d6d73b7451defc03ad96f183a4ba1ec674c8f2757f5a1"


def test_development_schedules_are_those_of_the_locked_build():
    rows = []
    for split, plan in (("a", sm.DEV_PLAN["a"]), ("b", sm.DEV_PLAN["b"])):
        for kind in bs.DRAW_ORDER:
            if kind not in plan:
                continue
            for lang in sm.LANGS:
                for scale in (1.0, 0.05):
                    n = bs.scaled(plan[kind][lang], scale)
                    generator = random.Random(f"5/special-mix/schedule/{split}/{kind}/{lang}")
                    rows.append([split, kind, lang, scale, bs.schedule(kind, n, generator, split=split),
                                 generator.random()])
    assert hashlib.sha256(json.dumps(rows, sort_keys=True).encode("utf-8")).hexdigest() == LOCKED_SCHEDULES


def test_the_training_schedule_of_revision_1():
    # docs/m8-预登记.md, 修订 1: two values 20 %, a wrong value on an optional parameter 20 %, half of the
    # format items a format the description alone implies; irrelevant items 70 % pairings
    generator = random.Random(1)
    incorrect = bs.schedule("incorrect", 500, generator, split="train")
    assert Counter(v["category"] for v in incorrect) == {"obvious": 250, "format": 125, "vague": 125}
    # two values and an optional parameter: each scheduled a little above the analysis's 20 %, since a few
    # such slots find no record and fall back (fallbacks: a second value goes before the category)
    assert sum(v["two"] for v in incorrect) == round(500 * sm.TRAIN_TWO_VALUES) == 105
    assert sum(v.get("optional", False) for v in incorrect) == round(500 * sm.OPTIONAL_SHARE) == 110
    assert sum(v.get("described", False) for v in incorrect) == round(125 * sm.DESCRIBED_SHARE)
    assert all(v["category"] == "format" for v in incorrect if v.get("described"))
    irrelevant = bs.schedule("irrelevant", 110, generator, split="train")
    assert sum(v.get("pairing", False) for v in irrelevant) == round(110 * bs.PAIRING_SHARE)
    missing = bs.schedule("missing", 220, generator, split="train")
    assert Counter(v["subkind"] for v in missing) == {"single": 88, "two": 55, "object": 44, "optional": 33}


def test_a_training_incorrect_slot_falls_back_keeping_its_category_before_its_second_value():
    # revision 1: the format share (at least 25 %) is a floor, so a training slot no record gives drops the
    # revision's own choices first, then its second value, then its category; a development slot falls back
    # as the locked build's did (straight to the plain construction), so the sets' draws stay as they were
    rich = {"category": "format", "two": True, "optional": True, "described": True}
    assert bs.fallbacks("incorrect", rich, split="train") == [
        {"category": "format", "two": True}, {"category": "format", "two": False},
        {"category": "obvious", "two": True}, {"category": "obvious", "two": False}]
    assert bs.fallbacks("incorrect", {"category": "vague", "two": True}, split="train") == [
        {"category": "vague", "two": False}, {"category": "obvious", "two": True},
        {"category": "obvious", "two": False}]
    optional = {"category": "format", "two": False, "optional": True}
    assert bs.fallbacks("incorrect", optional, split="train") == [
        {"category": "format", "two": False}, {"category": "obvious", "two": False}]
    assert bs.fallbacks("incorrect", {"category": "obvious", "two": False}, split="train") == []
    for split in ("a", "b"):
        assert bs.fallbacks("incorrect", {"category": "vague", "two": True}, split=split) == [
            {"category": "obvious", "two": False}]
        assert bs.fallbacks("incorrect", {"category": "obvious", "two": False}, split=split) == []
        assert bs.fallbacks("missing", {"subkind": "object"}, split=split) == [{"subkind": "single"}]


def forgotten(i: int) -> dict:
    """An English record whose gold call leaves out its function's required title (the user cannot recall it;
    record 448 of the revision's first build)."""
    name = f"search_movie_{i}"
    movies = tool(name, {"title": {"type": "string", "description": "Title of the movie"},
                         "year": {"type": "integer", "minimum": 1900, "maximum": 2030},
                         "genre": {"type": "string", "description": "Genre"}}, ["title"])
    user = f"I can't remember the title of a movie from {1990 + i}; it is a drama number {i}."
    return sample("glaive_v2", f"m{i}", user, [(name, {"year": 1990 + i, "genre": "drama"})], [movies])


def test_a_training_incorrect_slot_takes_no_gold_call_without_a_required_value():
    # the review of revision 1: such a record gives a wrong and a missing value at once, a priority case the
    # prompt's order decides (record 448 was answered with the wrong-value sentence under a missing-first
    # prompt); a training slot passes it over, a development slot draws as the locked build did
    obvious = {"category": "obvious", "two": False}
    drawer = bs.Drawer({"glaive_v2": [forgotten(1), booking("glaive_v2", 2)]}, 5, benchmarks(), {})
    item, _ = bs.draw_slot(drawer, "train", "incorrect", "en", obvious, ("glaive_v2",), 0)
    assert item.core.records == ("2",)
    drawer = bs.Drawer({"glaive_v2": [forgotten(1), booking("glaive_v2", 2)]}, 5, benchmarks(), {})
    item, _ = bs.draw_slot(drawer, "a", "incorrect", "en", obvious, ("glaive_v2",), 0)
    assert item.core.records == ("m1",) and item.core.required == ["title"]


def test_the_followers_of_a_training_slot():
    # revision 1: twins (a vague item's valid twin always; a missing twin for some of the other incorrect
    # items, TWIN_SHARE of them in all; an irrelevant twin for some missing items), and a tenth of the refusal
    # items in context
    generator = random.Random(2)
    incorrect = bs.schedule("incorrect", 500, generator, split="train")
    extras = bs.extras("incorrect", incorrect, generator)
    assert len(extras) == 500
    assert all(e["twin"] == "valid_verbatim" for v, e in zip(incorrect, extras, strict=True)
               if v["category"] == "vague")
    # scheduled above the analysis's 40 %, since about a quarter of the twins other than the vague items'
    # find no record or fail a check (the first revised build: 174 of its 500 incorrect items had one)
    assert sum(e["twin"] is not None for e in extras) == round(500 * bs.TWIN_SHARE) == 240
    assert {e["twin"] for e in extras} == {"valid_verbatim", "missing", None}
    assert sum(e["context"] for e in extras) == round(500 * bs.CONTEXT_SHARE)
    missing = bs.schedule("missing", 400, generator, split="train")
    extras = bs.extras("missing", missing, generator)
    assert {e["twin"] for e in extras} == {"irrelevant", None}
    assert all(e["twin"] is None for v, e in zip(missing, extras, strict=True) if v["subkind"] == "object")
    assert sum(e["twin"] == "irrelevant" for e in extras) == \
        round(sum(v["subkind"] != "object" for v in missing) * bs.IRRELEVANT_TWIN_SHARE)
    assert bs.extras("valid_verbatim", [{}] * 5, generator) == [{"twin": None, "context": False}] * 5


def test_twins_come_from_the_record_of_the_item_before_them(drawn):
    splits, account = drawn
    train = sorted(splits["train"], key=lambda i: i.core.drawn)
    twins = [i for i in train if i.core.variant.get("twin_of")]
    assert {"valid_verbatim", "missing"} <= {i.core.kind for i in twins} <= {"valid_verbatim", "missing",
                                                                                 "irrelevant"}
    for twin in twins:
        primary = train[twin.core.drawn - 1]
        assert primary.core.records[0] == twin.core.records[0] and primary.core.source == twin.core.source
        assert twin.core.variant["twin_of"] == primary.core.kind
        if twin.core.kind == "valid_verbatim":
            assert primary.core.kind == "incorrect"
            [(name, arguments)] = twin.core.calls
            assert name == primary.core.api
            assert all(sm.value_text(arguments[b.param]) != b.value for b in primary.core.invalid)
        elif twin.core.kind == "missing":
            assert primary.core.kind == "incorrect" and twin.core.missing == [primary.core.invalid[0].param]
            assert twin.core.variant["cut"] == "frame"  # a missing twin only on a fluent cut
        else:
            assert primary.core.kind == "missing" and twin.core.variant["pairing"]
            assert twin.core.history[-1] == primary.core.history[-1]
            assert not {t["function"]["name"] for t in twin.tools} & {primary.core.api}
    vague = [i for i in train if i.core.kind == "incorrect" and i.core.variant["category"] == "vague"]
    assert any(train[i.core.drawn + 1].core.variant.get("twin_of") for i in vague)
    assert account["followers"]["train"]["twins"]


def test_a_missing_item_takes_an_irrelevant_twin_from_its_own_record():
    # the same cut request once with the function it needs and once with another record's unrelated
    # functions only; the partners are tried near-domain first, each still unrelated to the request
    queue = [booking("toolace", 1), lookup("toolace", 3), no_call("toolace", 2)]
    drawer = bs.Drawer({"toolace": queue}, 5, benchmarks(), {})
    item, _ = bs.draw_slot(drawer, "train", "missing", "en", {"subkind": "single"}, ("toolace",), 0)
    [twin] = drawer.followers("train", item, {"twin": "irrelevant", "context": False})
    assert twin.core.kind == "irrelevant" and twin.core.variant == {"pairing": True, "twin_of": "missing"}
    assert twin.core.records == (item.core.records[0], "n2") and twin.core.history == item.core.history
    assert twin.core.drawn == item.core.drawn + 1 and drawer.revision["twins"] == {"irrelevant": 1}
    assert twin.expected["case"] == "irrelevant" and not sm.shares_content_word(twin.messages[-1]["content"],
                                                                                 twin.tools)


def test_a_follower_that_fails_is_counted_with_what_turned_it_away():
    # the manifest says why the followers the analysis asked for fall short: a twin with no record to give it
    # (no_core), no partner to pair (no_partner), or the check that refused it (a near-duplicate cluster full)
    drawer = bs.Drawer({"toolace": [booking("toolace", 1)]}, 5, benchmarks(), {})
    item, _ = bs.draw_slot(drawer, "train", "missing", "en", {"subkind": "single"}, ("toolace",), 0)
    assert drawer.followers("train", item, {"twin": "irrelevant", "context": False}) == []
    assert drawer.revision["failed"] == {"twin/irrelevant/no_partner": 1}
    # two records of one request: an incorrect item and a second item fill the cluster, so the first one's
    # twin is turned away; its speed cut leaves "... to Stop0B at /h.", which does not read fluently, so the
    # twin asked for as a missing one is the valid one (the same request with the original value)
    drawer = bs.Drawer({"glaive_v2": [same_route(0), same_route(1)]}, 5, benchmarks(), {})
    obvious = {"category": "obvious", "two": False}
    item, _ = bs.draw_slot(drawer, "train", "incorrect", "en", obvious, ("glaive_v2",), 0)
    other, _ = bs.draw_slot(drawer, "train", "incorrect", "en", obvious, ("glaive_v2",), 1)
    assert item.core.records == ("s0",) and other.core.records == ("s1",)
    assert drawer.followers("train", item, {"twin": "missing", "context": False}) == []
    assert drawer.revision["failed"] == {"missing_twin/bare->valid_verbatim": 1,
                                         "twin/valid_verbatim/train_cap": 1}
    assert drawer.revision["twins"] == {}


def taxes(i: int) -> dict:
    """An English record whose income, cut, leaves a request that does not read fluently ("My income is and
    the tax rate is 20%.", as the review of revision 1 found among the missing twins); its income is the only
    value with a definition to break."""
    name = f"calculate_tax_{i}"
    calc = tool(name, {"income": {"type": "number", "minimum": 0, "description": "Annual income"},
                       "tax_rate": {"type": "string", "description": "Tax rate in percent"}},
                ["income", "tax_rate"])
    user = f"Hi, I need help with calculating my tax. My income is {50000 + i} and the tax rate is 20%."
    return sample("glaive_v2", f"x{i}", user, [(name, {"income": 50000 + i, "tax_rate": "20%"})], [calc])


def test_a_missing_twin_takes_only_a_fluent_cut_and_else_the_valid_twin():
    # the review of revision 1: a twin has one choice of value to cut, and build_missing gives its bare cut
    # when that choice has no fluent one; 38 of the 67 missing twins were bare (9 % of the plain missing
    # items). The revision says a valid twin comes when the cut does not read fluently: it does now, counted
    drawer = bs.Drawer({"glaive_v2": [taxes(1)]}, 5, benchmarks(), {})
    obvious = {"category": "obvious", "two": False}
    item, _ = bs.draw_slot(drawer, "train", "incorrect", "en", obvious, ("glaive_v2",), 0)
    assert item.core.invalid[0].param == "income"
    bare = sm.build_missing(taxes(1), "en", random.Random(1), subkind="single", keys=["income"])
    assert bare is not None and bare.variant["cut"] == "bare"  # what the twin was
    [twin] = drawer.followers("train", item, {"twin": "missing", "context": False})
    assert twin.core.kind == "valid_verbatim" and twin.core.variant["twin_of"] == "incorrect"
    assert twin.core.records == item.core.records and twin.core.drawn == item.core.drawn + 1
    assert drawer.revision["failed"] == {"missing_twin/bare->valid_verbatim": 1}
    assert drawer.revision["twins"] == {"valid_verbatim": 1}


def test_object_and_priority_cores_are_rendered_twice(drawn):
    splits, _ = drawn
    train = sorted(splits["train"], key=lambda i: i.core.drawn)
    second = [i for i in train if i.core.variant.get("render") == 2]
    assert {i.core.kind for i in second} == {"missing", "priority"}
    for item in second:
        [first] = [i for i in train[:item.core.drawn] if i.core.records == item.core.records
                   and i.core.kind == item.core.kind and not i.core.variant.get("render")]
        assert (first.core.kind, first.core.records, first.core.history) == \
            (item.core.kind, item.core.records, item.core.history)
        if item.core.kind == "priority":
            assert first.core.drawn == item.core.drawn - 1  # right after its core
        assert item.system != first.system
        if item.core.kind == "priority":
            assert {first.rules.order, item.rules.order} == set(sm.ORDERS)
            assert {first.expected["case"], item.expected["case"]} == {"incorrect", "missing"}
        else:
            assert item.core.nested and item.core.variant["subkind"] == "object"
    objects = [i for i in train if i.core.kind == "missing" and i.core.variant.get("subkind") == "object"]
    assert objects and all(i.core.variant.get("render") in (None, 2) for i in objects)


def test_a_tenth_of_the_training_refusal_items_answer_in_context(drawn):
    splits, account = drawn
    framed = [i for i in splits["train"] if i.core.variant.get("context")]
    assert framed
    for item in framed:
        assert item.core.kind in ("incorrect", "missing", "irrelevant", "priority")
        assert not (item.core.kind == "irrelevant" and len(item.core.records) > 2)
        assert [role for role, _ in item.core.history] == ["user", "assistant", "user"]
        if item.core.packed:
            assert len(item.messages) == 2 and item.context_turns == 0
        else:
            assert [m["role"] for m in item.messages] == ["system", "user", "assistant", "user"]
    assert sum(account["followers"]["train"]["contexts"].values()) == len(framed)


def test_a_refusal_in_context_tries_its_partners_until_a_framed_item_passes():
    # a partner that cannot frame the core (a parameter name in common) or whose framed item fails a check
    # is passed over for the next; the first PAIRING_SCAN suitable partners are tried shortest first, since a
    # framed prompt must stay within MAX_PROMPT_TOKENS
    slot = tool("find_slot_g1", {"guest": {"type": "string", "description": "Guest"}}, ["guest"])
    shared = sample("toolace", "c1", "Find Ana's slot.", [("find_slot_g1", {"guest": "Ana"})], [slot])
    zone = {"type": "string", "description": "The time zone to read the clock in"}
    clock = tool("get_time_c2", {"zone": zone}, ["zone"])
    good = sample("toolace", "c2", "What time is it now in the time zone called CET, please?",
                  [("get_time_c2", {"zone": "CET"})], [clock])
    drawer = bs.Drawer({"toolace": [booking("toolace", 1), good, shared]}, 5, benchmarks(), {})
    core = sm.build_missing(booking("toolace", 1), "en", random.Random(1), subkind="single")
    drawer.used.add(("toolace", "1"))
    item = drawer.framed("train", core, booking("toolace", 1), random.Random(1))
    assert item is not None and item.core.records == ("1", "c2")
    assert item.core.variant["earlier"] == "get_time_c2"
    assert drawer.revision["failed"]["context/unframed"] == 1  # the shorter partner shares guest


def test_training_irrelevant_items_are_mostly_pairings_and_few_canned(drawn):
    splits, account = drawn
    for lang in sm.LANGS:
        items = [i for i in splits["train"] if i.core.kind == "irrelevant" and i.lang == lang]
        pairings = [i for i in items if i.core.variant.get("pairing")]
        assert len(pairings) >= 0.65 * len(items), (lang, len(pairings), len(items))
        own = [i for i in items if not i.core.variant.get("pairing")]
        canned = [i for i in own if sm.canned_request(i.core.originals[0])]
        assert len(canned) <= bs.canned_cap(bs.scaled(sm.PLAN["irrelevant"][lang], 0.05))
    assert account["irrelevant"]["train"]["zh"]["pairings"] >= 1


def test_training_refusal_answers_take_the_cell_deck_in_order(drawn):
    splits, _ = drawn
    for lang in sm.LANGS:
        deck = bs.deck(5, lang)
        answered = [i for i in sorted(splits["train"], key=lambda i: i.core.drawn)
                    if i.lang == lang and i.core.kind in bs.CELL_KINDS]
        assert answered
        for item, cell in zip(answered, deck, strict=False):
            family = item.rules.families[item.expected["case"]]
            assert (item.rules.style, sm.family_shape(family)) == (cell.style, (cell.order, cell.position))


def test_item_problems_check_the_reasoning_of_a_training_item_only():
    item = sm.dress(sm.build_incorrect(booking("toolace", 1), "en", random.Random(1)), split="train",
                    rng=random.Random(1))
    assert bs.item_problems(item, benchmarks()) == []
    broken = dataclasses.replace(item, reasoning=item.reasoning.replace(sm.REASONING_OPENING, "Tools:"))
    assert bs.item_problems(broken, benchmarks()) == ["reasoning_form"]
    dev = sm.dress(sm.build_incorrect(booking("toolace", 1), "en", random.Random(1)), split="b",
                   rng=random.Random(1))
    assert dev.reasoning == "" and bs.item_problems(dev, benchmarks()) == []


def test_draw_keeps_splits_apart(drawn):
    splits, _ = drawn
    keys = {split: {(i.core.source, r) for i in items for r in i.core.records}
            for split, items in splits.items()}
    assert not keys["train"] & keys["a"] and not keys["train"] & keys["b"] and not keys["a"] & keys["b"]
    names = {split: {t["function"]["name"].casefold() for i in items for t in i.tools}
             for split, items in splits.items()}
    assert not names["train"] & names["b"] and not names["train"] & names["a"]
    dev_names = {t["function"]["name"] for split in ("a", "b") for i in splits[split] for t in i.tools}
    found = bs.mentions(dev_names)
    for item in splits["train"]:
        texts = [m["content"] for m in item.messages] + [item.answer, item.reasoning]
        assert not any(found.search(text) for text in texts)
    for split, items in splits.items():
        for item in items:
            if item.rules is not None:
                assert {f.split for f in item.rules.families.values()} == {split}


def test_zh_special_items_take_a_share_of_english_records(drawn):
    splits, _ = drawn
    # the slots' items: a follower (revision 1) comes from its item's record
    slots = [i for i in splits["train"] if not bs.extra_item(i)]
    zh = [i for i in slots if i.lang == "zh" and i.core.kind in sm.SPECIAL_KINDS]
    from_en = sum(i.core.source in bs.EN_SOURCES for i in zh)
    assert 0.15 * len(zh) <= from_en <= 0.45 * len(zh)
    assert all(i.core.source in bs.ZH_SOURCES for i in slots
               if i.lang == "zh" and i.core.kind in sm.VALID_KINDS)
    assert all(i.core.source in bs.EN_SOURCES for i in splits["train"] if i.lang == "en")


def test_zh_items_the_chinese_sources_cannot_give_fall_back_to_english_records():
    p = pools()
    plain = tool("CreateNoteFolder", {"Name": {"type": "string", "description": "目录名"},
                                      "Memo": {"type": "string", "description": "备注"}})
    def folder(i: int) -> dict:
        name, memo = f"财务{OWNERS[i % 10]}{i}", f"季度{i}汇总"
        purpose = PURPOSES[i % len(PURPOSES)]
        return sample("deepexi", str(i), f"新建一个名为{name}的目录,备注写{memo}。这是{purpose}的需要。",
                      [(f"CreateNoteFolder{i}", {"Name": name, "Memo": memo})],
                      [{**plain, "function": {**plain["function"], "name": f"CreateNoteFolder{i}"}}])

    p["deepexi"] = [folder(i) for i in range(300)]  # free strings only: no near miss can be made of them
    splits, account = bs.build(p, seed=5, scale=0.05, bench=benchmarks(), banks=bs.value_banks(p))
    nearmiss = [i for i in splits["train"] if i.core.kind == "valid_nearmiss" and i.lang == "zh"]
    assert len(nearmiss) == bs.scaled(90, 0.05)
    assert all(i.core.source in bs.EN_SOURCES for i in nearmiss)
    assert account["zh_from_en_sources"]["train/valid_nearmiss/zh"] == len(nearmiss)
    assert "train/valid_nearmiss/zh" not in account["short"]


def test_item_problems_drop_a_filler_that_reads_as_a_placeholder():
    level = {"type": "integer", "minimum": 1, "maximum": 9, "description": "Level"}
    gauge = tool("set_gauge", {"value": level}, ["value"])
    s = sample("toolace", "g1", "Set the gauge to level 5 now.", [("set_gauge", {"value": 5})], [gauge])
    item = sm.dress(sm.build_incorrect(s, "en", random.Random(1), category="obvious"), split="train",
                    rng=random.Random(2))
    assert item.expected["params"] == ["value"]

    def styled(style: str) -> sm.Item:  # the item in another style, its reasoning written for it
        restyled = dataclasses.replace(item, rules=dataclasses.replace(item.rules, style=style))
        return dataclasses.replace(restyled, reasoning=sm.reasoning_text(restyled))

    assert bs.item_problems(styled("angle"), benchmarks()) == ["placeholder_lookalike"]  # <value>
    assert bs.item_problems(styled("square"), benchmarks()) == ["placeholder_lookalike"]  # [VALUE]
    assert bs.item_problems(styled("paren"), benchmarks()) == []  # (given value)


def test_the_chinese_source_shortfall_is_reported_per_kind():
    p = pools()
    name = {"type": "string", "description": "目录名"}
    # free strings only: no wrong value and no near miss can be made of these
    p["deepexi"] = [sample("deepexi", str(i), f"新建一个名为财务{OWNERS[i % 10]}{i}的目录。"
                                              f"这是{PURPOSES[i % len(PURPOSES)]}的需要。",
                           [(f"CreateNoteFolder{i}", {"Name": f"财务{OWNERS[i % 10]}{i}"})],
                           [tool(f"CreateNoteFolder{i}", {"Name": name})])
                    for i in range(300)]
    _, account = bs.build(p, seed=5, scale=0.05, bench=benchmarks(), banks=bs.value_banks(p))
    n = bs.scaled(90, 0.05)
    assert account["zh_sources"]["train/valid_nearmiss/zh"] == {"planned": n, "got": 0, "short": n}
    m = bs.scaled(300, 0.05)
    incorrect = account["zh_sources"]["train/incorrect/zh"]
    assert incorrect == {"planned": m - round(m * bs.EN_SHARE), "got": 0, "short": m - round(m * bs.EN_SHARE)}
    verbatim = account["zh_sources"]["train/valid_verbatim/zh"]
    assert verbatim["short"] == 0 and verbatim["got"] == verbatim["planned"] == bs.scaled(250, 0.05)


def test_value_banks_leave_out_values_that_name_their_parameter():
    folder = tool("RenameFolder", {"FolderName": {"type": "string", "description": "文件夹名称"}})
    named = sample("deepexi", "n1", "文件夹名称是什么?", [("RenameFolder", {"FolderName": "文件夹名称"})],
                   [folder])
    given = sample("deepexi", "n2", '把文件夹改名为"新文件夹"。',
                   [("RenameFolder", {"FolderName": "新文件夹"})], [folder])
    banks = bs.value_banks({"deepexi": [named, given]})
    assert banks["deepexi"] == {"renamefolder": {"FolderName": ["新文件夹"]}}


def test_a_value_change_takes_its_new_value_from_records_of_its_own_function():
    # hand check, round 1: a signature's name changed to "Eiffel Tower", a job search's location to "My House"
    def signed(sid: str, name: str) -> dict:
        signature = tool("make_signature", {"name": {"type": "string", "description": "Person"}}, ["name"])
        return sample("glaive_v2", sid, f"Make an email signature for {name}.",
                      [("make_signature", {"name": name})], [signature])

    def landmark(sid: str, name: str) -> dict:
        place = tool("find_landmark", {"name": {"type": "string", "description": "Landmark"}}, ["name"])
        return sample("glaive_v2", sid, f"Tell me about {name}.", [("find_landmark", {"name": name})],
                      [place])

    records = [signed("s1", "Ana Lima"), signed("s2", "Bo Chen"), landmark("l1", "Eiffel Tower"),
               landmark("l2", "Big Ben"), landmark("l3", "Old Bridge")]
    banks = bs.value_banks({"glaive_v2": records})
    assert banks["glaive_v2"] == {"find_landmark": {"name": ["Big Ben", "Eiffel Tower", "Old Bridge"]},
                                  "make_signature": {"name": ["Ana Lima", "Bo Chen"]}}
    for seed in range(10):
        drawer = bs.Drawer({"glaive_v2": [records[0]]}, 5, benchmarks(), banks)
        change = {"mode": "change", "packed": False}
        item = drawer.attempt("train", "valid_multiturn", "en", change, records[0], random.Random(seed))
        assert item is not None and item.core.variant["new"] == "Bo Chen"


def test_no_two_development_items_make_the_same_call():
    # hand check, round 1: set B held UpdateFolder(FolderId="123456", FolderName="新文件夹") twice, under two
    # wordings no near-duplicate test joins
    rename = tool("RenameFolder", {"FolderId": {"type": "string", "description": "文件夹ID"},
                                   "FolderName": {"type": "string", "description": "新的名称"}})
    call = [("RenameFolder", {"FolderId": "f-100", "FolderName": "季度报表"})]
    first = sample("deepexi", "d1", "把文件夹f-100改名为季度报表。", call, [rename])
    reworded = "请将ID是f-100的那个文件夹的名字修改成季度报表吧，谢谢。"
    second = sample("deepexi", "d2", reworded, call, [rename])
    catalog = tool("CreateNoteFolder", {"Name": {"type": "string", "description": "目录名"},
                                        "Memo": {"type": "string", "description": "备注"}})
    other = sample("deepexi", "d3", "新建一个名为销售数据的目录,备注写季度汇总。",
                   [("CreateNoteFolder", {"Name": "销售数据", "Memo": "季度汇总"})], [catalog])
    assert not sm.near_duplicate(first["messages"][0]["content"], second["messages"][0]["content"])
    drawer = bs.Drawer({"deepexi": [first, second, other]}, 5, benchmarks(), {})
    one = drawer.slot("b", "valid_verbatim", "zh", {}, ("deepexi",), 0)
    two = drawer.slot("b", "valid_verbatim", "zh", {}, ("deepexi",), 1)
    assert one.core.records == ("d1",) and two.core.records == ("d3",)
    assert drawer.near[("dev", "call", "valid_verbatim")] == 1
    assert bs.near_problems({"b": [one, two], "a": [], "train": []}) == []
    twin = dataclasses.replace(two, expected=one.expected)
    assert bs.near_problems({"b": [one, twin], "a": [], "train": []})  # the final check sees it too


def test_frequent_names_are_the_names_more_than_the_limit_of_records_show():
    p = {"glaive_v2": [booking("glaive_v2", i) for i in range(3)], "hermes": [booking("hermes", 9)]}
    for s in p["glaive_v2"]:
        s["tools"].append(tool("Shared_Lookup", {"q": {"type": "string"}}))
    assert bs.frequent_names(p, 2) == frozenset({"shared_lookup"})
    assert bs.frequent_names(p, 3) == frozenset()


def test_closed_deepexi_names_leave_a_seeded_share_open():
    p = {"deepexi": [deepexi(i) for i in range(200)], "toolace": [booking("toolace", 1)]}
    names = {t["function"]["name"].casefold() for s in p["deepexi"] for t in s["tools"]}
    closed = bs.closed_deepexi_names(p, 5, 25)
    assert closed <= names and "book_room_toolace_1" not in closed
    assert 0.15 * len(names) <= len(names - closed) <= 0.35 * len(names)
    assert bs.closed_deepexi_names(p, 5, 25) == closed
    assert bs.closed_deepexi_names(p, 6, 25) != closed  # another seed opens another share
    assert bs.closed_deepexi_names(p, 5, 100) == frozenset() and bs.closed_deepexi_names(p, 5, 0) == names


def test_development_sets_show_only_the_open_deepexi_names(drawn):
    splits, account = drawn
    p = pools()
    closed = bs.closed_deepexi_names(p, 5, bs.DEV_DEEPEXI_SHARE)
    assert account["dev_deepexi_share"] == bs.DEV_DEEPEXI_SHARE
    assert account["dev_deepexi_closed"] == len(closed)
    for split in ("a", "b"):
        deepexi_items = [i for i in splits[split] if i.core.source == "deepexi"]
        assert deepexi_items
        assert not {t["function"]["name"].casefold() for i in deepexi_items for t in i.tools} & closed
    trained = {t["function"]["name"].casefold() for i in splits["train"] if i.core.source == "deepexi"
               for t in i.tools}
    assert trained & closed  # training is not held to the share


def test_development_sets_avoid_the_frequent_names_and_training_does_not():
    drawer = bs.Drawer({}, 5, benchmarks(), {}, closed=frozenset({"shared_lookup"}))
    drawer.reserved["b"] = {"only_in_b"}
    assert "shared_lookup" in drawer.avoid("b") and "shared_lookup" in drawer.avoid("a")
    assert "only_in_b" in drawer.avoid("a") and "only_in_b" in drawer.avoid("train")
    assert "shared_lookup" not in drawer.avoid("train")


def routed(i: int) -> dict:
    """A record of plan_route, the one function many records share, able to lose its top-level object."""
    stops = {"type": "object", "description": "First and last stop",
             "properties": {"first": {"type": "string"}, "last": {"type": "string"}}}
    route = tool("plan_route", {"origin": {"type": "string", "description": "Start city"}, "stops": stops,
                                "speed": {"type": "integer", "minimum": 10, "maximum": 130}},
                 ["origin", "stops", "speed"])
    speed = 40 + i % 50
    user = f"Plan a route from Town{i}Q via Stop{i}A to Stop{i}B at {speed} km/h."
    args = {"origin": f"Town{i}Q", "stops": {"first": f"Stop{i}A", "last": f"Stop{i}B"}, "speed": speed}
    return sample("glaive_v2", f"r{i}", user, [("plan_route", args)], [route])


def test_the_draw_leaves_frequent_functions_to_training(monkeypatch):
    p = pools()
    p["glaive_v2"] += [routed(i) for i in range(80)]
    monkeypatch.setattr(bs, "DEV_NAME_LIMIT", 40)
    splits, account = bs.build(p, seed=5, scale=0.05, bench=benchmarks(), banks=bs.value_banks(p))
    assert account["dev_name_limit"] == 40 and account["dev_avoided_names"] == 1
    for split in ("a", "b"):
        assert not [i for i in splits[split] if any(t["function"]["name"] == "plan_route" for t in i.tools)]
    assert [i for i in splits["train"] if i.core.api == "plan_route"]


def hotel(i: int) -> dict:
    """An English record with scalar values only: it gives every kind but a top-level-object cut."""
    find = tool(f"find_hotel_{i}", {"city": {"type": "string", "description": "City"},
                                    "nights": {"type": "integer", "minimum": 1, "maximum": 30}},
                ["city", "nights"])
    return sample("toolace", f"h{i}", f"Find a hotel in City{i}Z for {2 + i % 5} nights.",
                  [(f"find_hotel_{i}", {"city": f"City{i}Z", "nights": 2 + i % 5})], [find])


def same_route(i: int) -> dict:
    """routed(0) under another id: six records whose one request is one near-duplicate cluster."""
    record = routed(0)
    record["source_id"] = f"s{i}"
    return record


@pytest.mark.parametrize("first", [True, False])
def test_the_scarce_object_construction_is_drawn_before_the_kinds_competing_for_its_records(first,
                                                                                             monkeypatch):
    p = pools()
    p["toolace"] = [hotel(i) for i in range(200)] + [no_call("toolace", i) for i in range(30)]
    p["glaive_v2"] = [same_route(i) for i in range(6)] + [no_call("glaive_v2", i) for i in range(30)]
    p["hermes"] = [hotel(1000 + i) for i in range(100)]
    monkeypatch.setattr(bs, "DEV_NAME_LIMIT", 5)  # plan_route stays out of the development sets
    if not first:
        monkeypatch.setattr(bs, "FIRST", {})
    _, account = bs.build(p, seed=5, scale=0.05, bench=benchmarks(), banks=bs.value_banks(p))
    objects = account["objects"]["train/missing/en"]
    planned = account["planned_constructions"]["train/missing/en"]["object"]
    if first:
        # the cluster's two training items are two object cuts: the second renderings (revision 1) come after
        # every object slot, so they take only the room the cores leave, which here is none
        assert objects["items"] == planned == 2 and objects["cores"] == 2 and objects["renders"] == 0
        # each core's second rendering was tried and refused by the cluster's limit, which the manifest says
        assert account["followers"]["train"]["failed"].get("render/missing/train_cap") == 2
    else:
        assert objects["items"] < planned  # the kinds drawn earlier filled the cluster


def test_an_object_core_alone_in_its_cluster_is_rendered_again_after_the_object_slots(monkeypatch):
    p = pools()
    p["toolace"] = [hotel(i) for i in range(200)] + [no_call("toolace", i) for i in range(30)]
    # six routes, six near-duplicate clusters
    p["glaive_v2"] = [routed(i) for i in range(6)] + [no_call("glaive_v2", i) for i in range(30)]
    p["hermes"] = [hotel(1000 + i) for i in range(100)]
    monkeypatch.setattr(bs, "DEV_NAME_LIMIT", 5)  # plan_route stays out of the development sets
    splits, account = bs.build(p, seed=5, scale=0.05, bench=benchmarks(), banks=bs.value_banks(p))
    objects = account["objects"]["train/missing/en"]
    assert objects["cores"] == 2 and objects["renders"] == 2 and objects["items"] == 4
    train = sorted(splits["train"], key=lambda i: i.core.drawn)
    objects = [i for i in train if i.core.variant.get("subkind") == "object"]
    cores = [i for i in objects if not i.core.variant.get("render")]
    renders = [i for i in objects if i.core.variant.get("render")]
    assert max(i.core.drawn for i in cores) < min(i.core.drawn for i in renders)  # every core first
    assert [i.core.records for i in renders] == [i.core.records for i in cores]


def test_an_object_construction_short_of_its_schedule_fails_the_build():
    account = {"short": {}, "planned_constructions": {"train/missing/en": {"single": 4, "object": 2},
                                                      "a/missing/en": {"single": 1}},
               "variants": {"train/missing/en": {"single": 5, "object": 1}, "a/missing/en": {"single": 1}}}
    assert bs.shortfalls(account) == ["train/missing/en: 1 of 2 object items"]
    account["variants"]["train/missing/en"] = {"single": 4, "object": 2}
    assert bs.shortfalls(account) == []


def test_draw_varies_the_incorrect_and_missing_constructions(drawn):
    splits, _ = drawn
    incorrect = [i.core.variant for i in splits["train"] if i.core.kind == "incorrect"]
    assert {v["category"] for v in incorrect} == {"obvious", "format", "vague"}
    assert any(v["values"] == 2 for v in incorrect)
    missing = [i.core for i in splits["train"] if i.core.kind == "missing"]
    assert {c.variant["subkind"] for c in missing} == set(sm.SUBKINDS)
    assert any(c.nested for c in missing)


# ---- near-duplicate requests (D1), English irrelevant pairing (D2), cuts (D8), gold calls (D3) ----------


def pizza(i: int) -> dict:
    """glaive's most repeated no-call request, under another id each time."""
    return sample("glaive_v2", f"p{i}", "Can you order a pizza for me?", [],
                  [tool(f"calculate_tip_p{i}", {"bill": {"type": "number"}}, ["bill"])],
                  text="I'm sorry, but I can't order pizza. My capabilities are limited to tips.")


def duplicated_pools() -> dict[str, list[dict]]:
    """pools() whose only English no-call requests are one pizza order under 40 ids, and whose hermes records
    repeat three booking requests under 30 more ids."""
    p = pools()
    p["toolace"] = [booking("toolace", i) for i in range(120)]
    p["glaive_v2"] = [pizza(i) for i in range(40)] + [booking("glaive_v2", i) for i in range(120)]
    twins = []
    for i in range(30):  # booking records whose requests repeat under new ids
        twin = booking("hermes", i % 3)
        twin["source_id"] = f"t{i}"
        twins.append(twin)
    p["hermes"] = twins + p["hermes"]
    return p


@pytest.fixture(scope="module")
def drawn_duplicates():
    p = duplicated_pools()
    return bs.build(p, seed=5, scale=0.05, bench=benchmarks(), banks=bs.value_banks(p))


def test_no_development_item_repeats_another_request(drawn_duplicates):
    splits, account = drawn_duplicates
    dev = [(split, item) for split in ("b", "a") for item in splits[split]]
    for n, (split, item) in enumerate(dev):
        for other_split, other in dev[n + 1:]:
            assert not any(sm.near_duplicate(x, y) for x in sm.request_texts(item)
                           for y in sm.request_texts(other)), (split, other_split)
        for trained in splits["train"]:
            assert not any(sm.near_duplicate(x, y) for x in sm.request_texts(item)
                           for y in sm.request_texts(trained))
    rejected = account["near_duplicates"]
    assert rejected["rejected"] and sum(rejected["by_rule"].values()) == sum(rejected["by_test"].values())
    assert set(rejected["by_rule"]) <= {"dev", "train_cap", "dev_template", "train_template"}


def test_training_keeps_at_most_two_near_duplicates_of_a_request(drawn_duplicates):
    splits, _ = drawn_duplicates
    train = sorted(splits["train"], key=lambda item: item.core.drawn)  # in the order of the draw
    assert [i.core.drawn for i in train] == list(range(len(train)))
    for n, item in enumerate(train):
        earlier = sum(any(sm.near_duplicate(x, y) for x in sm.request_texts(item)
                          for y in sm.request_texts(other)) for other in train[:n])
        assert earlier < 2, item.messages[-1]["content"]
    pizzas = [i for i in train if "pizza" in i.messages[-1]["content"]]
    assert len(pizzas) <= 2
    assert bs.near_problems(splits) == []


def test_near_problems_follow_the_order_of_the_draw():
    # three requests, the first near-duplicating each of the others, which are no near-duplicates of each
    # other
    hub = sm.dress(sm.build_verbatim(booking("toolace", 1), "en", random.Random(1)), split="train",
                   rng=random.Random(1))
    left = dataclasses.replace(hub, core=dataclasses.replace(hub.core, originals=()))
    texts = {"hub": "Book room 7 for Ana and Bo on Monday with breakfast",
             "left": "Book room 7 for Ana and Bo on Monday with breakfast please now",
             "right": "Book room 7 for Ana and Bo on Monday with breakfast at noon"}
    assert sm.near_duplicate(texts["hub"], texts["left"]) and sm.near_duplicate(texts["hub"], texts["right"])
    assert not sm.near_duplicate(texts["left"], texts["right"])

    def item(name: str, drawn: int) -> sm.Item:
        messages = [left.messages[0], {"role": "user", "content": texts[name]}]
        return dataclasses.replace(left, messages=messages, core=dataclasses.replace(left.core, drawn=drawn))

    listed = [item("left", 1), item("right", 2), item("hub", 0)]  # the hub drawn first, listed last
    assert bs.near_problems({"b": [], "a": [], "train": listed}) == []
    listed = [item("left", 0), item("right", 1), item("hub", 2)]  # the hub drawn last: two before it
    assert bs.near_problems({"b": [], "a": [], "train": listed})


def test_english_irrelevant_items_pair_requests_with_unrelated_functions(drawn_duplicates):
    splits, account = drawn_duplicates
    paired = [i for split in splits.values() for i in split
              if i.core.kind == "irrelevant" and i.lang == "en" and i.core.variant.get("pairing")]
    assert paired  # the duplicated pizzas cannot fill the English irrelevant items
    for item in paired:
        assert len(item.core.records) == 2 and not sm.shares_content_word(item.messages[-1]["content"],
                                                                           item.tools)
    counted = account["irrelevant_pairing"]
    assert sum(counted[split]["en"] for split in ("train", "a", "b")) == len(paired)


def test_pairing_needs_requests_and_functions_without_a_shared_content_word():
    order = tool("order_food", {"dish": {"type": "string"}}, ["dish"], description="Order a dish.")
    tip = tool("calculate_tip", {"bill": {"type": "number"}}, ["bill"], description="Work out a tip.")
    assert sm.shares_content_word("Can you order a pizza for me?", [order])
    assert not sm.shares_content_word("Can you order a pizza for me?", [tip])
    described = tool("lookup_rate", {"base": {"type": "string", "description": "pizza size"}}, ["base"])
    assert sm.shares_content_word("Can you order a pizza for me?", [described])  # parameter text counts
    request = booking("glaive_v2", 1)
    other = sample("glaive_v2", "o1", "What's the tip on 20?", [("calculate_tip", {"bill": 20})], [tip])
    core = sm.build_irrelevant(request, "en", random.Random(1), tools_from=other)
    assert core is not None and core.variant["pairing"] and core.records == ("1", "o1")
    related = sample("glaive_v2", "o2", "Order soup.", [("order_food", {"dish": "soup"})],
                     [tool("room_service", {"floor": {"type": "integer"}}, ["floor"],
                           description="Bring food to a guest's room.")])
    assert sm.build_irrelevant(request, "en", random.Random(1), tools_from=related) is None  # room, guest


def anagram(i: int) -> dict:
    """A request whose required values can only be cut bare (the cut leaves 'if and')."""
    check = tool(f"check_anagram_{i}", {"word1": {"type": "string"}, "word2": {"type": "string"}},
                 ["word1", "word2"])
    first, second = f"racecar{i}x", f"level{i}y"
    return sample("toolace", f"g{i}", f"Check if {first} and {second} are anagrams.",
                  [(f"check_anagram_{i}", {"word1": first, "word2": second})], [check])


def test_can_give_turns_away_only_records_the_builders_refuse():
    drawer = bs.Drawer({}, 5, benchmarks(), {})
    p = duplicated_pools()
    p["hermes"].append(trip_with_budget())
    builders = {
        ("incorrect", "obvious"): lambda s, lang: sm.build_incorrect(s, lang, random.Random(1)),
        ("missing", "single"): lambda s, lang: sm.build_missing(s, lang, random.Random(1)),
        ("missing", "object"): lambda s, lang: sm.build_missing(s, lang, random.Random(1), subkind="object"),
        ("priority", ""): lambda s, lang: sm.build_priority(s, lang, random.Random(1)),
        ("valid_verbatim", ""): lambda s, lang: sm.build_verbatim(s, lang, random.Random(1),
                                                                  require_cjk=lang == "zh"),
        ("valid_nearmiss", ""): lambda s, lang: sm.build_nearmiss(s, lang, random.Random(1)),
        ("irrelevant", ""): lambda s, lang: sm.build_irrelevant(s, lang, random.Random(1)),
    }
    refused = 0
    for records in p.values():
        for record in records[:80]:
            for (kind, construction), build in builders.items():
                variant = {"subkind": construction} if kind == "missing" else {}
                for lang in sm.LANGS:
                    if not drawer.can_give(kind, lang, variant, record):
                        refused += 1
                        if kind == "irrelevant" and record["source"] == "deepexi":
                            continue  # deepexi's irrelevant items are pairings, built with a partner
                        assert build(record, lang) is None, (kind, construction, record["source_id"])
    assert refused > 100


def test_the_draw_is_the_same_without_can_give(monkeypatch, drawn_duplicates):
    p = duplicated_pools()
    monkeypatch.setattr(bs.Drawer, "can_give", lambda self, kind, lang, variant, sample: True)
    splits, _ = bs.build(p, seed=5, scale=0.05, bench=benchmarks(), banks=bs.value_banks(p))
    filtered, _ = drawn_duplicates
    for split in splits:
        assert [(i.core.records, i.answer) for i in splits[split]] == \
            [(i.core.records, i.answer) for i in filtered[split]]


def trip_with_budget() -> dict:
    budget = {"type": "object", "description": "Budget", "properties": {"amount": {"type": "integer"},
                                                                         "currency": {"type": "string"}}}
    route = tool("book_tour", {"city": {"type": "string"}, "budget": budget}, ["city", "budget"])
    return sample("hermes", "tb1", "Book a tour of Lisbon with a budget of 1500 EUR.",
                  [("book_tour", {"city": "Lisbon", "budget": {"amount": 1500, "currency": "EUR"}})], [route])


def test_the_draw_takes_fluent_cuts_first_and_bare_ones_to_fill_the_quota():
    queue = [anagram(0), booking("toolace", 1), anagram(2)]
    drawer = bs.Drawer({"toolace": queue}, 5, benchmarks(), {})
    single = {"subkind": "single"}
    first, _ = bs.draw_slot(drawer, "train", "missing", "en", single, ("toolace",), 0)
    assert first.core.records == ("1",) and first.core.variant["cut"] == "frame"
    second, _ = bs.draw_slot(drawer, "train", "missing", "en", single, ("toolace",), 1)
    assert second.core.records == ("g0",) and second.core.variant["cut"] == "bare"
    assert drawer.rejected["bare_deferred"] == 2  # both anagrams were passed over by the fluent pass


def test_the_manifest_reports_the_fluent_share_and_the_dev_items_their_cut(tmp_path, stub):
    code, paths = run_cli(tmp_path)
    assert code == 0
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    for split in ("train", "a", "b"):
        for kind in ("missing", "priority"):
            cuts = manifest["cuts"][split][kind]
            assert cuts["frame"] + cuts["bare"] == manifest["counts"][split][kind]["total"]
            assert cuts["fluent_share"] == round(cuts["frame"] / max(1, cuts["frame"] + cuts["bare"]), 4)
    labels: dict[str, Counter] = {split: Counter() for split in ("train", "a", "b")}
    for split in ("a", "b"):
        for line in paths[f"dev_{split}"].read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            assert ("cut" in item["source"]) is (item["kind"] in ("missing", "priority"))
            if "cut" in item["source"]:
                labels[split][item["kind"], item["source"]["cut"]] += 1
    for line in paths["out"].read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["kind"] in bs.CUT_KINDS:
            labels["train"][record["kind"], record["variant"]["cut"]] += 1
    for split, kinds in manifest["cuts"].items():  # the counts are the items' own labels
        for kind, cuts in kinds.items():
            own = labels[split]
            assert (cuts["frame"], cuts["bare"]) == (own[kind, "frame"], own[kind, "bare"])
    assert sum(count for (_, cut), count in labels["train"].items() if cut == "frame")


def test_cut_account_counts_each_items_own_cut():
    item = sm.dress(sm.build_missing(booking("toolace", 1), "en", random.Random(1)), split="train",
                    rng=random.Random(1))

    def cut(label: str) -> sm.Item:
        core = dataclasses.replace(item.core, variant={**item.core.variant, "cut": label})
        return dataclasses.replace(item, core=core)

    account = bs.cut_account({"train": [cut("frame"), cut("bare"), cut("bare")], "a": [], "b": [cut("bare")]})
    assert account["train"]["missing"] == {"frame": 1, "bare": 2, "fluent_share": 0.3333}
    assert account["b"]["missing"] == {"frame": 0, "bare": 1, "fluent_share": 0.0}
    assert account["a"]["missing"] == {"frame": 0, "bare": 0, "fluent_share": 0.0}


def test_the_manifest_counts_the_records_whose_gold_call_is_not_fully_grounded(tmp_path, monkeypatch, stub):
    tags = tool("TagResource", {"TagId": {"type": "string"}, "TagValue": {"type": "string"}})
    copied = [sample("deepexi", f"c{i}", f"给资源{i}打上IT部门这个标签。",
                     [("TagResource", {"TagId": "IT部门", "TagValue": "IT部门"})], [tags]) for i in range(3)]

    def with_copies(manifest, raw_dir):
        p = pools()
        p["deepexi"] = copied + p["deepexi"]
        return p, {name: {"rows": len(v), "drawn": 0, "unused": len(v)} for name, v in p.items()}

    monkeypatch.setattr(bs, "load_pools", with_copies)
    code, paths = run_cli(tmp_path)
    assert code == 0
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["gold_calls"]["deepexi"]["copied_value"] == 3
    records = {(json.loads(line)["source"], json.loads(line)["source_id"])
               for line in paths["out"].read_text(encoding="utf-8").splitlines()}
    assert not records & {("deepexi", f"c{i}") for i in range(3)}


# ---- the fix round's review: templates (F1), values naming their own parameter (F5) ---------------------


def similar(x: sm.Item, y: sm.Item, *, whole: bool = False) -> str | None:
    """special_mix.same_template over the two items' records' requests (Core.originals), or with whole over
    all their texts (special_mix.request_texts)."""
    def texts(item: sm.Item) -> tuple[str, ...]:
        return sm.request_texts(item) if whole else item.core.originals

    return next((found for a in texts(x) for b in texts(y)
                 if (found := sm.same_template(sm.template_tokens(a), sm.template_tokens(b)))), None)


def dev_in_draw_order(splits: dict) -> list[sm.Item]:
    return [item for split in ("b", "a") for item in sorted(splits[split], key=lambda i: i.core.drawn)]


def test_development_irrelevant_items_are_pairings(drawn):
    # the records' own refusals are a handful of templates (glaive_v2's 13,614 refusal records of the
    # seed-1102 pools ask 195 distinct requests, nearly all to book a flight or to order a pizza), so a
    # development irrelevant item pairs a request answered by a call with another record's unrelated functions
    splits, account = drawn
    for split in ("a", "b"):
        irrelevant = [i for i in splits[split] if i.core.kind == "irrelevant"]
        assert irrelevant and all(i.core.variant.get("pairing") and len(i.core.records) == 2
                                  for i in irrelevant)
        for lang in sm.LANGS:
            assert account["planned_constructions"][f"{split}/irrelevant/{lang}"] == {
                "irrelevant+pairing": bs.scaled(sm.DEV_PLAN[split]["irrelevant"][lang], 0.05)}
    assert [i for i in splits["train"] if i.core.kind == "irrelevant" and not i.core.variant.get("pairing")]


def test_development_items_hold_at_most_two_of_a_template(drawn):
    splits, _ = drawn
    dev = dev_in_draw_order(splits)
    for n, item in enumerate(dev):
        assert sum(bool(similar(item, other)) for other in dev[:n]) < bs.DEV_TEMPLATE_LIMIT, item.core.records


def test_training_holds_no_request_of_a_development_items_template_and_kind(drawn):
    splits, _ = drawn
    dev = [i for split in ("a", "b") for i in splits[split]]
    for item in splits["train"]:
        for other in dev:
            if bs.template_kind(other.core.kind) == bs.template_kind(item.core.kind):
                assert not similar(item, other), (item.core.records, other.core.records)


def test_a_development_pairing_leaves_records_able_to_lose_an_object_to_training():
    # a development item keeps its request's near-duplicates out of training (D1), and the records able to
    # lose a top-level object are the scarcest (FIRST): no development pairing takes their requests
    tip = no_call("glaive_v2", 1)
    find = tool("find_hotel", {"city": {"type": "string"}, "nights": {"type": "integer"}}, ["city", "nights"])
    plain = sample("glaive_v2", "h1", "Find a hotel in Porto for 3 nights.",
                   [("find_hotel", {"city": "Porto", "nights": 3})], [find])
    drawer = bs.Drawer({"glaive_v2": [routed(0), plain, tip]}, 5, benchmarks(), {})
    item = drawer.slot("b", "irrelevant", "en", {"pairing": True}, ("glaive_v2",), 0)
    assert item is not None and item.core.records[0] == "h1" and item.core.variant["pairing"]
    assert drawer.rejected["object_kept_for_training"] == 1
    train = bs.Drawer({"glaive_v2": [routed(0), tip]}, 5, benchmarks(), {})
    paired = train.slot("train", "irrelevant", "en", {"pairing": True}, ("glaive_v2",), 0)
    assert paired.core.records[0] == "r0"


def test_a_third_development_item_of_a_template_is_turned_away():
    queue = [booking("toolace", 12 * k) for k in range(4)] + [booking("toolace", 1)]  # one occasion, another
    drawer = bs.Drawer({"toolace": queue}, 5, benchmarks(), {})
    items = [drawer.slot("b", "valid_verbatim", "en", {}, ("toolace",), n) for n in range(3)]
    assert [i.core.records for i in items] == [("0",), ("12",), ("1",)]
    assert drawer.near[("dev_template", "jaccard", "valid_verbatim")] == 2  # 24 and 36 turned away


def test_training_turns_away_the_template_of_a_development_item_of_its_kind_only():
    queue = [booking("toolace", 0), booking("toolace", 12), booking("toolace", 1)]
    drawer = bs.Drawer({"toolace": queue}, 5, benchmarks(), {})
    dev = drawer.slot("b", "valid_verbatim", "en", {}, ("toolace",), 0)
    train = drawer.slot("train", "valid_verbatim", "en", {}, ("toolace",), 0)
    assert dev.core.records == ("0",) and train.core.records == ("1",)
    assert drawer.near[("train_template", "jaccard", "valid_verbatim")] == 1
    other = bs.Drawer({"toolace": queue}, 5, benchmarks(), {})
    other.slot("b", "valid_verbatim", "en", {}, ("toolace",), 0)
    missing = other.slot("train", "missing", "en", {"subkind": "single"}, ("toolace",), 0)
    assert missing.core.records == ("12",)  # a training item of another kind may share the template
    assert bs.template_kind("ask") == "missing" and bs.template_kind("priority") == "priority"


def test_near_problems_see_the_template_rules():
    def dressed(i: int, split: str, kind: str = "valid_verbatim") -> sm.Item:
        record = booking("toolace", i)
        core = (sm.build_verbatim(record, "en", random.Random(1)) if kind == "valid_verbatim"
                else sm.build_missing(record, "en", random.Random(1)))
        item = sm.dress(core, split=split, rng=random.Random(1))
        item.core.drawn = i
        return item

    dev = [dressed(12 * k, "b") for k in range(3)]  # one template
    assert bs.near_problems({"b": dev[:2], "a": [], "train": []}) == []
    assert bs.near_problems({"b": dev, "a": [], "train": []})  # a third of the template
    assert bs.near_problems({"b": dev[:1], "a": [], "train": [dressed(48, "train")]})  # its template and kind
    assert bs.near_problems({"b": dev[:1], "a": [], "train": [dressed(48, "train", "missing")]}) == []


def test_templates_compare_the_records_requests_not_the_text_this_module_adds():
    # a packed history, a follow-up and a serialised call are this module's own text, which every multi-turn
    # item shares whatever its user asked: templates are read from the records' requests (Core.originals)
    def change(i: int, split: str) -> sm.Item:
        core = sm.build_change(booking("toolace", i), "en", random.Random(3),
                               bank={"guest": ["Ana Lima", "Bo Chen"]}, packed=True)
        item = sm.dress(core, split=split, rng=random.Random(3))
        item.core.drawn = 0
        return item

    dev, train = change(0, "b"), change(1, "train")
    assert similar(dev, train, whole=True)  # their whole texts: the wrapper, the call notation, the follow-up
    assert not sm.same_template(*(sm.template_tokens(x.core.originals[0]) for x in (dev, train)))
    assert bs.near_problems({"b": [dev], "a": [], "train": [train]}) == []
    drawer = bs.Drawer({}, 5, benchmarks(), {})
    drawer.templates.add(("b", 0), "valid_multiturn", dev.core.originals)
    assert drawer.near_conflict("train", list(sm.request_texts(train)), "valid_multiturn",
                                list(train.core.originals)) is None


def test_the_manifest_counts_the_template_rejections(tmp_path, stub):
    code, paths = run_cli(tmp_path)
    assert code == 0
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    near = manifest["near_duplicates"]
    assert set(near["by_rule"]) <= {"dev", "train_cap", "dev_template", "train_template"}
    assert near["by_rule"].get("dev_template") and near["by_rule"].get("train_template")
    assert sum(near["by_rule"].values()) == sum(near["by_test"].values()) == sum(near["by_kind"].values())
    assert set(near["by_test"]) <= {"exact", "signature", "jaccard", "contains", "call"}


def test_value_banks_leave_out_values_equal_to_their_parameters_name():
    plan = tool("UpdatePlan", {"planName": {"type": "string", "description": "计划名称"}})
    named = sample("deepexi", "q1", '更新计划,planName为"planName"。',
                   [("UpdatePlan", {"planName": "planName"})], [plan])
    given = sample("deepexi", "q2", '更新计划,planName为"春季计划"。',
                   [("UpdatePlan", {"planName": "春季计划"})], [plan])
    banks = bs.value_banks({"deepexi": [named, given]})
    assert banks["deepexi"] == {"updateplan": {"planName": ["春季计划"]}}


# ---- the CLI --------------------------------------------------------------------------------------------


@pytest.fixture
def stub(monkeypatch):
    def fake_pools(manifest, raw_dir):
        p = pools()
        return p, {name: {"rows": len(v), "drawn": 0, "unused": len(v), "revision": "r"}
                   for name, v in p.items()}

    monkeypatch.setattr(bs, "load_pools", fake_pools)
    monkeypatch.setattr(bs, "load_benchmarks", lambda acebench, bfcl: benchmarks())


def exclusion_file(path: Path, records: list[list[str]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"description": "records excluded by the private exclusion list",
                                "records": records}), encoding="utf-8")
    return path


def run_cli(tmp_path: Path, *extra: str,
            excluded: list[list[str]] | None = None) -> tuple[int, dict[str, Path]]:
    paths = {"out": tmp_path / "ft5" / "special-mix.jsonl",
             "reasoning": tmp_path / "ft5" / "reasoning-special-mix.jsonl",
             "manifest": tmp_path / "ft5" / "special-mix-manifest.json",
             "dev_a": tmp_path / "eval" / "special-dev-a.jsonl",
             "dev_b": tmp_path / "eval" / "special-dev-b.jsonl"}
    vocabulary = exclusion_file(tmp_path / "vocabulary-excluded.json", excluded or [])
    code = bs.main(["--out", str(paths["out"]), "--reasoning-out", str(paths["reasoning"]),
                    "--manifest", str(paths["manifest"]), "--dev-a", str(paths["dev_a"]),
                    "--dev-b", str(paths["dev_b"]), "--scale", "0.05", "--seed", "5",
                    "--ft3-manifest", str(tmp_path / "none.json"),
                    "--probe-manifest", str(tmp_path / "none.json"), "--no-default-excludes",
                    "--vocabulary-exclusions", str(vocabulary), *extra])
    return code, paths


def test_cli_honours_the_vocabulary_exclusion_file(tmp_path, stub):
    excluded = [["hermes", "3"], ["deepexi", "5"], ["glaive_v2", "n1"], ["toolace", "0"]]
    code, paths = run_cli(tmp_path, excluded=excluded)
    assert code == 0
    used = set()
    for key in ("out", "dev_a", "dev_b"):
        for line in paths[key].read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if key == "out":
                used |= {(row["source"], i) for i in row.get("source_ids", [row["source_id"]])}
            else:
                used |= {(row["source"]["dataset"], i) for i in row["source"]["record"].split("+")}
    assert not used & {tuple(pair) for pair in excluded}
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    vocabulary = manifest["vocabulary_exclusions"]
    file = tmp_path / "vocabulary-excluded.json"
    assert vocabulary["sha256"] == hashlib.sha256(file.read_bytes()).hexdigest()
    assert vocabulary["records"] == 4
    screened = manifest["exclusions"]["screen"]
    assert screened["hermes"]["vocabulary"] == 1 and screened["deepexi"]["vocabulary"] == 1
    assert sum(source["vocabulary"] for source in screened.values()) == 4


def test_stale_outputs_tells_a_build_of_this_code_from_an_older_one(tmp_path, stub):
    # the fix round's review, F2: round 1's outputs stayed in data/ with locks that still match; a run checks
    # the manifest before it trusts the locks
    code, paths = run_cli(tmp_path)
    assert code == 0
    vocabulary = tmp_path / "vocabulary-excluded.json"
    assert bs.stale_outputs(paths["manifest"], vocabulary) is None
    exclusion_file(vocabulary, [["hermes", "3"]])  # the file changed after the build
    assert "vocabulary-excluded.json" in bs.stale_outputs(paths["manifest"], vocabulary)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    del manifest["vocabulary_exclusions"]  # a build before the file, as round 1's
    paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
    assert bs.stale_outputs(paths["manifest"], vocabulary) is not None
    assert "absent" in bs.stale_outputs(tmp_path / "none.json", vocabulary)


def test_cli_needs_a_well_formed_vocabulary_exclusion_file(tmp_path, stub):
    code, _ = run_cli(tmp_path, "--vocabulary-exclusions", str(tmp_path / "absent.json"))
    assert code == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"description": "x", "records": [["hermes", "3"]], "words": ["y"]}),
                   encoding="utf-8")
    assert run_cli(tmp_path, "--vocabulary-exclusions", str(bad))[0] == 2  # nothing but the pairs
    bad.write_text(json.dumps({"description": "x", "records": [["hermes"]]}), encoding="utf-8")
    assert run_cli(tmp_path, "--vocabulary-exclusions", str(bad))[0] == 2


def test_cli_writes_the_training_set_the_sidecar_the_locked_dev_sets_and_the_manifest(tmp_path, stub):
    code, paths = run_cli(tmp_path)
    assert code == 0
    data = paths["out"].read_bytes()
    assert data.endswith(b"\n") and b"\r" not in data
    records = [json.loads(line) for line in data.decode("utf-8").splitlines()]
    assert all(sm.check_training_record(r) == [] for r in records)
    index = ReasoningIndex([paths["reasoning"]])
    assert len(index) == len(records)
    for number, record in enumerate(records):
        k, _, answer = turns(record["messages"])[-1]
        assert index.lookup(f"{paths['out'].stem}:{number}", k, answer)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    for key, path in paths.items():
        if key != "manifest":
            assert manifest["outputs"][key]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), key
    for split in ("a", "b"):
        path = paths[f"dev_{split}"]
        assert check_lock(path) == hashlib.sha256(path.read_bytes()).hexdigest()
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        assert all(sm.check_dev_item(i) == [] for i in items)
        assert [i["id"] for i in items] == [f"sdev-{split}-{n:04d}" for n in range(1, len(items) + 1)]
        assert find_test_material(items) == []
    assert manifest["counts"]["train"]["incorrect"]["zh"] == bs.scaled(300, 0.05)
    assert manifest["test_material"] is False
    closed = len(bs.closed_deepexi_names(pools(), 5, bs.DEV_DEEPEXI_SHARE))
    assert manifest["dev_names"] == {"limit": bs.DEV_NAME_LIMIT, "avoided": 0,
                                     "deepexi_share": bs.DEV_DEEPEXI_SHARE, "deepexi_closed": closed}
    plans = (("train", sm.PLAN), ("a", sm.DEV_PLAN["a"]), ("b", sm.DEV_PLAN["b"]))
    assert set(manifest["zh_sources"]) - {"note"} == {f"{split}/{kind}/zh" for split, plan in plans
                                                      for kind in plan}
    assert manifest["planned_constructions"]["train/missing/zh"]["object"] >= 1
    for split in ("a", "b"):
        items = [json.loads(line) for line in paths[f"dev_{split}"].read_text(encoding="utf-8").splitlines()]
        assert sm.dev_set_problems(items) == []


def test_cli_is_deterministic_and_refuses_to_overwrite(tmp_path, stub):
    _, paths = run_cli(tmp_path)
    locks = [Path(f"{paths[key]}.sha256") for key in ("dev_a", "dev_b")]
    first = {str(p): p.read_bytes() for p in [*paths.values(), *locks]}
    assert run_cli(tmp_path)[0] == 2
    assert run_cli(tmp_path, "--force")[0] == 0
    assert {str(p): p.read_bytes() for p in [*paths.values(), *locks]} == first  # the manifest too


def test_cli_fails_when_the_supply_is_short(tmp_path, monkeypatch, stub):
    small = {name: records[:3] for name, records in pools().items()}
    monkeypatch.setattr(bs, "load_pools", lambda manifest, raw_dir: (small, {}))
    code, paths = run_cli(tmp_path)
    assert code == 3 and not paths["out"].exists() and not paths["dev_b"].exists()


def test_cli_writes_a_short_supply_only_when_exactly_its_shortfalls_are_accepted(tmp_path, monkeypatch, stub):
    # the M8 preregistration accepts the shortfalls of the final build one by one: any other shortfall, or an
    # accepted one that is not there, still fails the build
    short = ["train/valid_verbatim/zh: 1 of 2", "a/valid_verbatim/zh: 0 of 1"]
    monkeypatch.setattr(bs, "shortfalls", lambda account: list(short))
    code, paths = run_cli(tmp_path, "--accept-shortfall", short[0])
    assert code == 3 and not paths["out"].exists()
    code, paths = run_cli(tmp_path, "--accept-shortfall", short[0], "--accept-shortfall", short[1],
                          "--accept-shortfall", "train/missing/en: 1 of 2 object items")
    assert code == 3 and not paths["out"].exists()
    code, paths = run_cli(tmp_path, "--accept-shortfall", short[1], "--accept-shortfall", short[0])
    assert code == 0 and paths["out"].exists()
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["shortfalls_accepted"] == sorted(short)
    monkeypatch.setattr(bs, "shortfalls", lambda account: [])
    assert run_cli(tmp_path, "--force", "--accept-shortfall", short[0])[0] == 3  # accepted, not short
    assert run_cli(tmp_path, "--force")[0] == 0
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    assert manifest["shortfalls_accepted"] == []


def test_cli_writes_only_when_the_drawn_development_sets_match_the_locks_given(tmp_path, stub):
    # revision 1 changes the training items alone: the build that writes them must draw sets A and B again
    # byte for byte, which --locked-dev-a and --locked-dev-b check against the locks of the sets in use
    code, first = run_cli(tmp_path / "first")
    assert code == 0
    locks = {split: Path(f"{first[f'dev_{split}']}.sha256") for split in ("a", "b")}
    code, again = run_cli(tmp_path / "again", "--locked-dev-a", str(locks["a"]), "--locked-dev-b",
                          str(locks["b"]))
    assert code == 0
    manifest = json.loads(again["manifest"].read_text(encoding="utf-8"))
    for split in ("a", "b"):
        digest = hashlib.sha256(again[f"dev_{split}"].read_bytes()).hexdigest()
        assert manifest["dev_locks"][split] == {"lock": bs.repo_relative(locks[split]), "sha256": digest,
                                                "matches": True}
    other = tmp_path / "other" / "special-dev-b.jsonl.sha256"
    other.parent.mkdir()
    other.write_text("0" * 64 + "  special-dev-b.jsonl\n", encoding="utf-8")
    code, third = run_cli(tmp_path / "third", "--locked-dev-b", str(other))
    assert code == 3 and not third["out"].exists() and not third["dev_b"].exists()
    named = tmp_path / "named.sha256"
    other_name = locks["a"].read_text(encoding="utf-8").replace("special-dev-a", "other")
    named.write_text(other_name, encoding="utf-8")
    assert run_cli(tmp_path / "fourth", "--locked-dev-a", str(named))[0] == 2  # a lock of another file


def test_the_manifest_reports_the_revision(tmp_path, stub):
    code, paths = run_cli(tmp_path)
    assert code == 0
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    revision = manifest["revision"]
    assert revision["preregistration"].startswith("docs/m8-预登记.md")
    assert set(revision["followers"]) >= {"twins", "renders", "contexts", "fills"}
    assert revision["incorrect_mix"]["train"]["two_values"] >= 1
    assert set(revision["cells"]) == set(sm.LANGS)
    for lang in sm.LANGS:
        cells = revision["cells"][lang]
        assert cells["answers"] == sum(cells["by_cell"].values())
    records = [json.loads(line) for line in paths["out"].read_text(encoding="utf-8").splitlines()]
    twins = sum(bool(r["variant"].get("twin_of")) for r in records)
    assert twins == sum(revision["followers"]["twins"].values())
    assert manifest["dev_locks"] == {}


def test_cli_needs_the_benchmarks(tmp_path, monkeypatch, stub):
    from research_agent.data.overlap import BenchmarkMissing

    def missing(acebench, bfcl):
        raise BenchmarkMissing("no ACEBench")

    monkeypatch.setattr(bs, "load_benchmarks", missing)
    assert run_cli(tmp_path)[0] == 2


# ---- the real outputs -------------------------------------------------------------------------------------

def _real_outputs() -> dict[str, Path]:
    """The build's outputs: the repository's, or those of the folder SPECIAL_MIX_DIR names (a build written
    elsewhere under the default file names)."""
    folder = os.environ.get("SPECIAL_MIX_DIR")
    if folder:
        base = Path(folder)
        return {"train": base / "special-mix.jsonl", "reasoning": base / "reasoning-special-mix.jsonl",
                "manifest": base / "special-mix-manifest.json", "a": base / "special-dev-a.jsonl",
                "b": base / "special-dev-b.jsonl"}
    return {"train": REPO / "data" / "raw" / "ft5" / "special-mix.jsonl",
            "reasoning": REPO / "data" / "raw" / "ft5" / "reasoning-special-mix.jsonl",
            "manifest": REPO / "data" / "raw" / "ft5" / "special-mix-manifest.json",
            "a": REPO / "data" / "eval" / "special-dev-a.jsonl",
            "b": REPO / "data" / "eval" / "special-dev-b.jsonl"}


REAL = _real_outputs()
_NAMES = (re.compile(r'"name": "([^"]+)"'), re.compile(r"'name': '([^']+)'"),
          re.compile(r"^### (.+)$", re.MULTILINE))


def _locked() -> bool:
    """Whether the outputs are all present and each development set passes its lock (else they are being
    rebuilt)."""
    if not all(p.is_file() for p in REAL.values()):
        return False
    try:
        return all(check_lock(REAL[key]) for key in ("a", "b"))
    except Exception:  # noqa: BLE001 - a missing or failing lock: the sets are being rebuilt
        return False


def _current() -> bool:
    """Whether the outputs come from a build of this script's code: present and locked (_locked), and their
    manifest records the vocabulary exclusion file as it is now (build_special_mix.stale_outputs)."""
    return _locked() and bs.stale_outputs(REAL["manifest"], REPO / bs.VOCABULARY) is None


@pytest.mark.skipif(not _locked(), reason="the special-mix outputs are absent or fail their locks (rebuilt)")
def test_locked_outputs_are_a_build_of_this_code():
    # the fix round's review, F2: round 1's outputs stayed in data/ with locks that still match, so a run
    # would take them, and they hold records the private exclusion list excludes. Present and locked, the
    # outputs must be this code's build with the current exclusion file; a stale build is rebuilt, or moved
    # out of data/ with the owner's consent (never deleted unasked).
    problem = bs.stale_outputs(REAL["manifest"], REPO / bs.VOCABULARY)
    assert problem is None, f"{problem}: no run may use these outputs and their locks"


@pytest.mark.skipif(not _current() or not CLONE.is_dir(),
                    reason="the special-mix outputs of this code (or the ACEBench clone) are not present")
def test_real_outputs_hold_no_near_duplicate_request_and_label_their_cuts():
    rows = {key: [json.loads(line) for line in REAL[key].read_text(encoding="utf-8").splitlines()]
            for key in ("train", "a", "b")}

    def request(row: dict) -> str:
        return "\n".join(m["content"] for m in row["messages"] if m["role"] == "user")

    index = sm.RequestIndex()
    for key in ("a", "b"):
        for row in rows[key]:
            found = index.matches([request(row)])
            assert not found, (row["id"], sorted(found))
            index.add(row["id"], [request(row)])
            assert ("cut" in row["source"]) is (row["kind"] in ("missing", "priority"))
            assert row["source"].get("cut", "frame") in ("frame", "bare")
    for row in rows["train"]:
        assert not index.matches([request(row)]), request(row)[:80]
        texts = [m["content"] for m in row["messages"]]
        assert not any(sm.holds_apiname(text) for text in texts)


@pytest.mark.skipif(not _current() or not CLONE.is_dir(),
                    reason="the special-mix outputs of this code (or the ACEBench clone) are not present")
def test_real_outputs_hold_no_acebench_text_window_or_function_name():
    windows = sm.TemplateWindows.from_files([CLONE / "model_inference" / "prompt_en.py",
                                             CLONE / "model_inference" / "prompt_zh.py"])
    names = {n.casefold() for n in sm.acebench_function_names(CLONE / "data_all")}
    texts: list[str] = []
    systems: list[str] = []
    for key in ("train", "a", "b"):
        for line in REAL[key].read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            texts += [m["content"] for m in row["messages"]]
            systems.append(row["messages"][0]["content"])
    rows = REAL["reasoning"].read_text(encoding="utf-8").splitlines()
    texts += [json.loads(line)["reasoning"] for line in rows]
    assert len(texts) > 4000
    for text in texts:
        assert sm.banned_markers(text) == [], text[:200]
        assert windows.violations(text) == [], (text[:200], windows.violations(text))
    shown = {n.casefold() for system in systems for pattern in _NAMES for n in pattern.findall(system)}
    assert len(shown) > 500
    assert not shown & names
    for key in ("a", "b"):
        items = [json.loads(line) for line in REAL[key].read_text(encoding="utf-8").splitlines()]
        assert all(sm.check_dev_item(i) == [] for i in items) and sm.dev_set_problems(items) == []
    try:
        bench = bs.load_benchmarks(CLONE, None)
    except FileNotFoundError:  # no BFCL environment: the overlap report covers the training file
        return
    users = [m["content"] for key in ("train", "a", "b")
             for line in REAL[key].read_text(encoding="utf-8").splitlines()
             for m in json.loads(line)["messages"] if m["role"] == "user"]
    assert not [u[:80] for u in users if bs.overlaps(u, bench)]

"""scripts/transform_synth.py: the M6 rules over the synthetic training file (docs/m6-计划.md §3.1, WP2)
and FT4's opt-in rules (docs/m7-本地提升计划.md WP6).

Every rule is tested on inline records; the CLI on small files written to tmp_path, with the repository's
corpus manifest and term list. No test reads a test-split evaluation file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from test_synth import load_script
from train.tokenizer_fake import FakeTokenizer

from research_agent.agent.loop import RELAX_HINT
from research_agent.agent.prompts import PROMPT_HASH, system_prompt
from research_agent.canonical import canonical_json
from research_agent.data import reasoning
from research_agent.data import requests as rq
from research_agent.tools import schemas
from research_agent.train.samples import build_step_samples, load_records
from research_agent.validation import check_call

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "data" / "corpus" / "manifest.json"
TERMS = REPO / "data" / "synth" / "terms-zh-en.json"
NEEDS_TERMS = pytest.mark.needs_data("data/synth/terms-zh-en.json")  # left out of the public copy (conftest)
TODAY = "2026-10-10"
TOK = FakeTokenizer()
ts = load_script("transform_synth")


# ---- inline records -------------------------------------------------------------------------------------


def call(name: str, args: dict, cid: str = "c1", compact: bool = False) -> dict:
    raw = json.dumps(args, ensure_ascii=False, separators=(",", ":") if compact else None)
    return {"function": {"arguments": raw, "name": name}, "id": cid, "type": "function"}


def record(rid: str, steps: list[list[dict]], *, scenario: str = "normal", user: str = "帮我找论文",
           tool_content: str = "[]") -> dict:
    messages = [{"content": system_prompt(TODAY), "role": "system"}, {"content": user, "role": "user"}]
    for calls in steps:
        messages.append({"content": "", "role": "assistant", "tool_calls": calls})
        for c in calls:
            messages.append({"content": tool_content, "role": "tool", "tool_call_id": c["id"]})
    messages.append({"content": "好的。", "role": "assistant"})
    return {"family": "x/zh/1", "history": "agent", "id": rid, "intent": "search_notes", "lang": "zh",
            "messages": messages, "prompt_hash": PROMPT_HASH, "run": "train", "scenario": scenario,
            "schema_hash": schemas.SCHEMA_HASH, "tools": schemas.MODEL_TOOLS}


def calls_of(rec: dict) -> list[dict]:
    return [c for m in rec["messages"] if m["role"] == "assistant" for c in m.get("tool_calls") or []]


def args_of(c: dict) -> dict:
    return json.loads(c["function"]["arguments"])


def request(rid: str, *, scenario: str = "normal", slots: dict | None = None, lang: str = "zh",
            split: str = "train", intent: str = "search_notes", text: str | None = None,
            family: str = "x/zh/1") -> dict:
    return {"family": family, "id": rid, "intent": intent, "lang": lang, "scenario": scenario,
            "slots": slots or {}, "split": split, "text": text or f"请求 {rid}"}


def steps_of(rec: dict) -> int:
    samples, dropped = build_step_samples(rec, TOK)
    assert dropped.total == 0
    return len(samples)


# ---- R1 parameter normalisation -------------------------------------------------------------------------


def test_r1_deletes_only_the_habit_values_and_keeps_key_order():
    rec = record("req-train-0001", [
        [call("search_notes", {"query": "lora", "top_k": 5, "tags": ["survey"]}, "a")],
        [call("list_notes", {"limit": 20, "since": "2026-09-01"}, "b")],
        [call("fetch_fulltext", {"arxiv_id": "2610.00814", "section": "method", "max_chars": 1500}, "c")],
        [call("search_arxiv", {"query": "q", "max_results": 5, "sort_by": "submittedDate"}, "d")],
        [call("search_notes", {"query": "rag", "top_k": 3}, "e"), call("list_notes", {"limit": 10}, "f")],
        [call("fetch_fulltext", {"arxiv_id": "2610.00814", "max_chars": 800}, "g")],
    ])
    out, counts = ts.rule_r1(rec, request("req-train-0001", slots={"count": 3}))
    got = [c["function"]["arguments"] for c in calls_of(out)]
    assert got == ['{"query": "lora", "tags": ["survey"]}', '{"since": "2026-09-01"}',
                   '{"arxiv_id": "2610.00814", "section": "method"}',
                   '{"query": "q", "max_results": 5, "sort_by": "submittedDate"}',
                   '{"query": "rag", "top_k": 3}', '{"limit": 10}',
                   '{"arxiv_id": "2610.00814", "max_chars": 800}']
    assert counts["deleted"] == {"search_notes.top_k": 1, "list_notes.limit": 1,
                                 "fetch_fulltext.max_chars": 1}
    assert counts["calls_changed"] == 3 and counts["records_changed"] == 1
    for c in calls_of(out):
        assert check_call(c["function"]["name"], args_of(c), [], TODAY) == []
    assert calls_of(rec)[0]["function"]["arguments"].count("top_k") == 1  # the input is not changed
    assert [m for m in out["messages"] if m["role"] != "assistant"] == \
        [m for m in rec["messages"] if m["role"] != "assistant"]


def test_r1_keeps_the_value_the_user_asked_for():
    rec = record("req-train-0002", [[call("search_notes", {"query": "x", "top_k": 5}, "a")],
                                    [call("list_notes", {"limit": 20}, "b")]])
    out, counts = ts.rule_r1(rec, request("req-train-0002", slots={"count": 5}))
    assert [args_of(c) for c in calls_of(out)] == [{"query": "x", "top_k": 5}, {}]
    assert counts["kept_count"] == {"search_notes.top_k": 1, "list_notes.limit": 0}
    out, counts = ts.rule_r1(rec, request("req-train-0002", slots={"count": 20}))
    assert args_of(calls_of(out)[0]) == {"query": "x"} and args_of(calls_of(out)[1]) == {"limit": 20}
    assert counts["kept_count"] == {"search_notes.top_k": 0, "list_notes.limit": 1}


def test_r1_keeps_the_input_serialisation_style():
    rec = record("req-train-0003", [[call("search_notes", {"query": "长上下文", "top_k": 5, "tags": ["a"]},
                                          "a", compact=True)]])
    out, _ = ts.rule_r1(rec, request("req-train-0003"))
    assert calls_of(out)[0]["function"]["arguments"] == '{"query":"长上下文","tags":["a"]}'


# ---- R2 and R4 clarification ---------------------------------------------------------------------------


def ask(missing: list[str] | None, cid: str = "q") -> dict:
    args = {"question": "哪个方向?"}
    if missing is not None:
        args["missing"] = missing
    return call("ask_clarification", args, cid)


def test_r2_sets_missing_to_the_request_slots():
    rec = record("req-train-0004", [[call("list_notes", {}, "a")], [ask(["topic", "date_range"])]],
                 scenario="clarify")
    out, counts = ts.rule_r2(rec, request("req-train-0004", scenario="clarify",
                                          slots={"missing": ["topic"]}))
    asked = args_of(calls_of(out)[-1])
    assert asked == {"question": "哪个方向?", "missing": ["topic"]}
    assert list(asked) == ["question", "missing"]
    assert set(asked["missing"]) <= set(schemas.MISSING)
    assert counts == {"calls": 1, "changed": 1, "no_slots_missing": 0}
    assert check_call("ask_clarification", asked, [], TODAY) == []
    out, counts = ts.rule_r2(record("r", [[ask(None)]], scenario="clarify"),
                             request("r", scenario="clarify", slots={"missing": ["paper_id"]}))
    assert args_of(calls_of(out)[0])["missing"] == ["paper_id"] and counts["changed"] == 1


def test_r2_leaves_a_call_without_request_slots_and_counts_it():
    rec = record("r", [[ask(["topic"])]], scenario="clarify")
    out, counts = ts.rule_r2(rec, request("r", scenario="clarify"))
    assert out == rec and counts == {"calls": 1, "changed": 0, "no_slots_missing": 1}


def test_r2_refuses_a_slot_outside_missing_vocabulary():
    with pytest.raises(ValueError):
        ts.rule_r2(record("r", [[ask(["topic"])]], scenario="clarify"),
                   request("r", scenario="clarify", slots={"missing": ["venue"]}))


def test_r4_drops_late_clarifications_and_counts_each_position():
    first = record("req-train-0010", [[ask(["topic"])]], scenario="clarify")
    second = record("req-train-0011", [[call("list_notes", {}, "a")], [ask(["topic"])]], scenario="clarify")
    third = record("req-train-0012", [[call("list_notes", {}, "a")],
                                      [call("search_notes", {"query": "x"}, "b")],
                                      [ask(["topic"])]], scenario="clarify")
    normal = record("req-train-0013", [[call("list_notes", {}, "a")], [call("list_notes", {}, "b")],
                                       [ask(["topic"])]])
    assert [ts.first_clarification_step(r) for r in (first, second, third)] == [1, 2, 3]
    kept, counts = ts.rule_r4([first, second, third, normal])
    assert [r["id"] for r in kept] == ["req-train-0010", "req-train-0011", "req-train-0013"]
    assert counts == {"first_step": 1, "second_step": 1, "late_dropped": 1, "no_clarification": 0,
                      "dropped_ids": ["req-train-0012"]}


def test_r3_skips_a_train_request_whose_kept_trajectory_asks_only_at_its_second_step():
    # R4 keeps the trajectories that search first and ask at step 2; a first-step record from the same request
    # would teach two different first steps for one user text. A request whose trajectory R4 dropped, or which
    # asks at step 1 already, still gives a record
    first = record("req-train-0010", [[ask(["topic"])]], scenario="clarify")
    second = record("req-train-0011", [[call("list_notes", {}, "a")], [ask(["topic"])]], scenario="clarify")
    reqs = [request("req-train-0010", scenario="clarify", slots={"missing": ["topic"]}),
            request("req-train-0011", scenario="clarify", slots={"missing": ["topic"]}),
            request("req-train-0012", scenario="clarify", slots={"missing": ["topic"]}),  # dropped by R4
            request("req-train-0013", scenario="clarify", slots={}),
            request("req-train-0014", scenario="normal", slots={"missing": ["topic"]})]
    eligible, counts = ts.r3_train_requests(reqs, kept=[first, second])
    assert [r["id"] for r in eligible] == ["req-train-0010", "req-train-0012"]
    assert counts == {"eligible": 2, "skipped_conflict": 1, "no_slots_missing": 1}


# ---- new records: R3, R5, R6, R7 ------------------------------------------------------------------------


def check_new(rec: dict, steps: int) -> None:
    assert rec["history"] == "agent" and rec["prompt_hash"] == PROMPT_HASH
    assert rec["schema_hash"] == schemas.SCHEMA_HASH and rec["family"].startswith("m6/")
    assert rec["messages"][0] == {"role": "system", "content": system_prompt(TODAY)}
    assert steps_of(rec) == steps


def test_r3_first_step_clarification_record(tmp_path):
    req = request("req-train-0020", scenario="clarify", intent="summarize_paper", lang="en",
                  slots={"missing": ["paper_id"]}, text="Summarise the paper I told you about yesterday.")
    rec = ts.clarification_record(req, TODAY, record_id="req-train-0020-m6r3")
    check_new(rec, 1)
    assistant = rec["messages"][2]
    (c,) = assistant["tool_calls"]
    asked = args_of(c)
    assert c["function"]["name"] == "ask_clarification" and asked["missing"] == ["paper_id"]
    assert asked["question"] in ts.CLARIFY_QUESTIONS["paper_id"]["en"]
    assert rec["messages"][3]["role"] == "tool"
    assert json.loads(rec["messages"][3]["content"])["status"] == "asked"
    for slot in schemas.MISSING:
        for lang in ("zh", "en"):
            assert len(ts.CLARIFY_QUESTIONS[slot][lang]) == 2
    path = tmp_path / "r3.jsonl"
    path.write_text(canonical_json(rec) + "\n", encoding="utf-8")
    (loaded,) = load_records(path)
    assert steps_of(loaded) == 1


def test_r5_templates_pass_the_template_audit_and_combine_since_and_tag():
    result = ts.audit_new_templates(ts.LIST_NOTES_TEMPLATES)
    assert result["ok"] and result["max_jaccard"] < rq.TEMPLATE_LIMIT
    for lang in ("zh", "en"):
        assert len(ts.LIST_NOTES_TEMPLATES[lang]) == 2
        for template in ts.LIST_NOTES_TEMPLATES[lang]:
            assert {"notes_since", "tag"} <= set(rq.placeholders(template))
    copied = {"zh": ("看看我标了 {tag} 的笔记,{notes_since}的。",), "en": ()}
    assert not ts.audit_new_templates(copied)["ok"]


def test_r5_list_notes_record():
    req = request("req-m6-0001", intent="list_notes", slots={"since": "2026-09-01", "tags": ["survey"]},
                  family="m6/list_notes/zh/0")
    rec = ts.list_notes_record(req, TODAY)
    check_new(rec, 1)
    (c,) = rec["messages"][2]["tool_calls"]
    assert c["function"]["name"] == "list_notes"
    assert c["function"]["arguments"] == '{"tag": "survey", "since": "2026-09-01"}'


def test_r6_empty_results_then_relax():
    req = request("req-train-0030", intent="find_recent_papers",
                  slots={"topic": "KV cache", "date_from": "2026-09-01", "categories": ["cs.CL"]})
    rec = ts.relax_record(req, TODAY, record_id="req-train-0030-m6r6")
    check_new(rec, 2)
    assert rec["scenario"] == "recover"
    first, tool, second = rec["messages"][2:5]
    assert args_of(first["tool_calls"][0]) == {"query": "KV cache", "categories": ["cs.CL"],
                                              "date_from": "2026-09-01"}
    assert json.loads(tool["content"]) == {"error": schemas.EMPTY_RESULTS, "hint": RELAX_HINT, "items": []}
    assert args_of(second["tool_calls"][0]) == {"query": "KV cache", "categories": ["cs.CL"]}
    assert len(rec["messages"]) == 5
    req = request("r", slots={"topic": "rag", "categories": ["cs.IR", "cs.CL"]})
    rec = ts.relax_record(req, TODAY, record_id="r-m6r6")
    assert args_of(rec["messages"][4]["tool_calls"][0]) == {"query": "rag"}


def test_transport_recovery_still_retries_as_is():
    failing = call("search_notes", {"query": "x", "top_k": 5}, "a")
    retry = call("search_notes", {"query": "x", "top_k": 5}, "b")
    rec = record("req-train-0040", [[failing], [retry]], scenario="recover",
                 tool_content='{"error":"RATE_LIMITED","items":[]}')
    out, _ = ts.rule_r1(rec, request("req-train-0040", scenario="recover"))
    a, b = calls_of(out)
    assert a["function"] == {**b["function"]} and args_of(a) == {"query": "x"}
    assert [m["content"] for m in out["messages"] if m["role"] == "tool"] == \
        [m["content"] for m in rec["messages"] if m["role"] == "tool"]


@NEEDS_TERMS
def test_r7_term_list_excludes_the_held_out_topics():
    terms = json.loads(TERMS.read_text(encoding="utf-8"))
    corpus = rq.load_corpus(CORPUS)
    assert len(terms) >= 100
    assert ts.check_terms(terms, corpus) == []
    assert ts.check_terms({"混合专家": "mixture of experts"}, corpus, minimum=1)
    assert ts.check_terms({"知识蒸馏": "knowledge distillation"}, corpus, minimum=1)
    assert ts.check_terms({"缩放定律研究": "scaling law"}, corpus, minimum=1)
    assert ts.check_terms({"注意力机制": "attention mechanism"}, corpus, minimum=1) == []
    assert ts.check_terms({"注意力机制": "attention mechanism"}, corpus)  # fewer than 100


def test_r7_term_record():
    req = request("req-m6-0101", intent=ts.TERM_INTENT, text="帮我找几篇注意力机制方向的论文。",
                  slots={"topic": "attention mechanism", "term_zh": "注意力机制"}, family="m6/terms/zh/0")
    rec = ts.term_record(req, TODAY)
    check_new(rec, 1)
    assert args_of(rec["messages"][2]["tool_calls"][0]) == {"query": "attention mechanism"}


# ---- requests: the eval guard and generation -------------------------------------------------------------


def test_guard_rejects_a_request_too_similar_to_an_evaluation_request():
    dev = "帮我找 2026-09-01 以来关于 KV cache 的论文,挑三篇最相关的。"
    guard = ts.EvalGuard(eval_texts=[dev], train_texts=["列出我写的笔记。"])
    assert guard.offer("帮我找 2026-08-11 以来关于 KV cache 的论文,挑两篇最相关的。") == "train_vs_eval"
    assert guard.offer("列出我写的笔记。") == "exact"
    assert guard.offer("把 survey 标签下最新的三条笔记列给我。") is None
    assert guard.max_eval_jaccard(dev) == 1.0
    assert guard.overlaps_eval(dev.replace("三篇", "四篇"))


def test_generated_requests_use_train_templates_and_topics_only():
    corpus = rq.load_corpus(CORPUS)
    guard = ts.EvalGuard(eval_texts=[], train_texts=[])
    assignment = rq.split_templates(42)
    made = ts.generate_clarify_requests(10, seed=42, corpus=corpus, split_seed=42, guard=guard)
    assert len(made) == 10 and sum(r["lang"] == "zh" for r in made) == 5
    for r in made:
        assert assignment[r["family"]] == "train" and r["scenario"] == "clarify" and r["split"] == "train"
        assert r["slots"]["missing"] and set(r["slots"]["missing"]) <= set(schemas.MISSING)
    listed = ts.generate_list_notes_requests(8, seed=42, corpus=corpus, split_seed=42, guard=guard)
    assert len(listed) == 8
    for r in listed:
        assert r["intent"] == "list_notes" and r["family"].startswith("m6/list_notes/")
        assert set(r["slots"]) == {"since", "tags"}
    assert made == ts.generate_clarify_requests(10, seed=42, corpus=corpus, split_seed=42,
                                                guard=ts.EvalGuard(eval_texts=[], train_texts=[]))


# ---- CLI ------------------------------------------------------------------------------------------------


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(canonical_json(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return path


@pytest.fixture
def inputs(tmp_path):
    corpus = rq.load_corpus(CORPUS)
    topics = rq.split_topics(corpus, 42)
    reqs = [
        request("req-train-0001", slots={"count": 3}, text="搜一下我的笔记"),
        request("req-train-0002", scenario="clarify", intent="summarize_paper",
                slots={"missing": ["paper_id"]}, text="导师早上转我的那篇讲了什么?一段话就行。"),
        request("req-train-0003", scenario="clarify", intent="find_recent_papers", lang="en",
                slots={"missing": ["topic"]}, text="Same field as before, please: two papers since May."),
        request("req-train-0004", intent="find_recent_papers",
                slots={"topic": "KV cache", "topic_id": "kv-cache", "date_from": "2026-09-01"},
                text="帮我找 9 月以来关于 KV cache 的论文。"),
        request("req-train-0005", scenario="recover", slots={"count": 5}, text="找五条笔记"),
        request("req-dev-0001", split="dev", text="dev 请求"),
        request("req-test-0001", split="test", text="test 请求"),
    ]
    synth = [
        record("req-train-0001", [[call("search_notes", {"query": "x", "top_k": 5}, "a")]]),
        record("req-train-0002", [[call("list_notes", {"limit": 20}, "a")], [ask(["paper_id", "topic"])]],
               scenario="clarify"),
        record("req-train-0003", [[call("list_notes", {}, "a")], [call("list_notes", {}, "b")],
                                  [ask(["topic"])]], scenario="clarify"),
        record("req-train-0005", [[call("search_notes", {"query": "x", "top_k": 5}, "a")]],
               scenario="recover"),
    ]
    requests_path = write_jsonl(tmp_path / "requests.jsonl", reqs)
    (tmp_path / "requests-manifest.json").write_text(json.dumps(
        {"seed": 42, "topics": {s: [t.id for t in topics[s]] for s in ("train", "dev", "test")}}),
        encoding="utf-8")
    dev = [{"id": "hw-dev-001", "split": "dev", "messages": [
        {"role": "user", "content": "帮我找 2026-09-01 以来关于 KV cache 的论文,挑三篇最相关的。"}]}]
    return {"synth": write_jsonl(tmp_path / "synth.jsonl", synth), "requests": requests_path,
            "dev": write_jsonl(tmp_path / "dev.jsonl", dev), "dir": tmp_path}


def argv(inputs: dict, *extra: str) -> list[str]:
    d = inputs["dir"]
    return [str(inputs["synth"]), str(inputs["requests"]), "--out", str(d / "out" / "synth-m6.jsonl"),
            "--manifest", str(d / "out" / "manifest.json"), "--requests-out", str(d / "out" / "req-m6.jsonl"),
            "--terms", str(TERMS), "--dev", str(inputs["dev"]), "--corpus", str(CORPUS),
            "--r3-total", "6", "--r5", "4", "--r6", "1", "--r7", "5", *extra]


@NEEDS_TERMS
def test_cli_writes_records_requests_and_manifest(inputs, capsys):
    assert ts.main(argv(inputs)) == 0
    out = inputs["dir"] / "out"
    data = (out / "synth-m6.jsonl").read_bytes()
    assert b"\r\n" not in data
    records = load_records(out / "synth-m6.jsonl")
    ids = [r["id"] for r in records]
    assert ids == sorted(ids) and "req-train-0003" not in ids  # a late clarification
    rules = {}
    for r in records:
        rule = r["family"].split("/")[1] if r["family"].startswith("m6/") else "old"
        rules[rule] = rules.get(rule, 0) + 1
        steps = steps_of(r)
        if rule in ("r3", "list_notes", "terms"):
            assert steps == 1
        if rule == "r6":
            assert steps == 2
    assert rules == {"old": 3, "r3": 6, "list_notes": 4, "r6": 1, "terms": 5}
    lines = (out / "req-m6.jsonl").read_text(encoding="utf-8").splitlines()
    new_requests = [json.loads(line) for line in lines]
    assert [r["id"] for r in new_requests] == [f"req-m6-{i:04d}" for i in range(1, len(new_requests) + 1)]
    # 5 new clarify requests: of the file's 2, req-train-0003's trajectory was dropped (so it gives a record)
    # and req-train-0002's kept trajectory asks at step 2 (so it gives none); 4 list, 5 terms
    assert len(new_requests) == 5 + 4 + 5
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rules"]["R1"]["deleted"]["search_notes.top_k"] == 1
    assert manifest["rules"]["R1"]["kept_count"]["search_notes.top_k"] == 1
    assert manifest["rules"]["R2"]["changed"] == 1
    assert manifest["rules"]["R4"]["late_dropped"] == 1
    assert manifest["rules"]["R3"]["from_train_requests"] == 1 and manifest["rules"]["R3"]["records"] == 6
    assert manifest["rules"]["R3"]["train_requests"] == {"eligible": 1, "skipped_conflict": 1,
                                                         "no_slots_missing": 0}
    assert "req-train-0003-m6r3" in ids and "req-train-0002-m6r3" not in ids
    assert manifest["records"]["in"]["total"] == 4 and manifest["records"]["out"]["total"] == len(records)
    assert manifest["requests_out"]["count"] == len(new_requests)
    assert manifest["output"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert manifest["output"]["bytes"] == len(data)
    assert manifest["inputs"]["synth"]["sha256"] == hashlib.sha256(inputs["synth"].read_bytes()).hexdigest()
    assert manifest["terms"]["sha256"] == hashlib.sha256(TERMS.read_bytes()).hexdigest()
    assert manifest["eval_overlap"]["max_jaccard"] < rq.TRAIN_VS_EVAL
    assert manifest["eval_overlap"]["over"] == 0
    text = (out / "manifest.json").read_text(encoding="utf-8")
    assert text == json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


@NEEDS_TERMS
def test_cli_reruns_byte_identical_and_keeps_out_without_force(inputs):
    out = inputs["dir"] / "out"
    assert ts.main(argv(inputs)) == 0
    first = {p.name: p.read_bytes() for p in out.iterdir()}
    assert ts.main(argv(inputs)) == 2
    assert {p.name: p.read_bytes() for p in out.iterdir()} == first
    assert ts.main(argv(inputs, "--force")) == 0
    assert {p.name: p.read_bytes() for p in out.iterdir()} == first


@NEEDS_TERMS
def test_cli_exit_codes(inputs):
    missing = argv(inputs)
    missing[0] = str(inputs["dir"] / "nope.jsonl")
    assert ts.main(missing) == 2
    rows = inputs["synth"].read_text(encoding="utf-8").splitlines()
    orphan = json.loads(rows[0])
    orphan["id"] = "req-train-9999"
    inputs["synth"].write_text("\n".join([*rows, canonical_json(orphan)]) + "\n", encoding="utf-8")
    assert ts.main(argv(inputs)) == 1
    assert not (inputs["dir"] / "out" / "synth-m6.jsonl").exists()


@NEEDS_TERMS
def test_cli_refuses_test_material_as_the_dev_file(inputs):
    write_jsonl(inputs["dev"], [{"id": "hw-test-001", "split": "test",
                                 "messages": [{"role": "user", "content": "x"}]}])
    assert ts.main(argv(inputs)) == 1


@NEEDS_TERMS
def test_cli_drops_new_requests_that_match_the_dev_file(inputs):
    """The dev items enter the guard: a term request equal to a dev text is dropped, not written."""
    terms = json.loads(TERMS.read_text(encoding="utf-8"))
    zh = next(iter(terms))
    text = rq.tidy(ts.TERM_TEMPLATES[0].format(term=zh), "zh")
    write_jsonl(inputs["dev"], [{"id": "hw-dev-001", "split": "dev",
                                 "messages": [{"role": "user", "content": text}]}])
    assert ts.main(argv(inputs)) == 0
    out = inputs["dir"] / "out"
    lines = (out / "req-m6.jsonl").read_text(encoding="utf-8").splitlines()
    texts = [json.loads(line)["text"] for line in lines]
    assert text not in texts
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rules"]["R7"]["dropped_eval_overlap"] >= 1


@NEEDS_TERMS
def test_the_full_runs_manifest_keeps_m6s_keys(inputs):
    """The options added for M7 leave a run without them as M6 made it: no key of theirs in the manifest."""
    assert ts.main(argv(inputs)) == 0
    manifest = json.loads((inputs["dir"] / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {"generator", "seed", "split_seed", "prompt_hash", "schema_hash", "today",
                             "rules", "records", "eval_overlap", "inputs", "terms", "requests_out", "output"}
    assert set(manifest["inputs"]) == {"synth", "requests", "dev", "corpus", "requests_manifest"}
    assert set(manifest["rules"]["R1"]) == {"calls_seen", "deleted", "kept_count", "calls_changed",
                                            "records_changed", "scope"}


# ---- R1's habits, derived from the input as M6 derived them ---------------------------------------------


def test_r1_deletes_the_habits_it_is_given():
    rec = record("req-train-0001", [[call("search_notes", {"query": "x", "top_k": 10}, "a")],
                                    [call("search_notes", {"query": "y", "top_k": 5}, "b")],
                                    [call("list_notes", {"limit": 20}, "c")]])
    out, counts = ts.rule_r1(rec, request("req-train-0001"), habits={("search_notes", "top_k"): 10})
    assert [args_of(c) for c in calls_of(out)] == [{"query": "x"}, {"query": "y", "top_k": 5}, {"limit": 20}]
    assert counts["deleted"] == {"search_notes.top_k": 1, "list_notes.limit": 0,
                                 "fetch_fulltext.max_chars": 0}
    assert counts["kept_count"] == {"search_notes.top_k": 0, "list_notes.limit": 0}
    out, counts = ts.rule_r1(rec, request("req-train-0001", slots={"count": 10}),
                             habits={("search_notes", "top_k"): 10})
    assert args_of(calls_of(out)[0]) == {"query": "x", "top_k": 10}  # the user asked for ten
    assert counts["kept_count"]["search_notes.top_k"] == 1
    assert ts.rule_r1(rec, request("req-train-0001"))[0] == ts.rule_r1(rec, request("req-train-0001"),
                                                                      habits=dict(ts.HABITS))[0]


def test_derived_habits_are_the_values_set_most_often_unless_that_is_the_default():
    recs = [record("req-train-0001", [[call("search_notes", {"query": "x", "top_k": 10}, "a")],
                                      [call("search_notes", {"query": "y", "top_k": 10}, "b")],
                                      [call("search_notes", {"query": "z", "top_k": 5}, "c")],
                                      [call("search_notes", {"query": "w"}, "d")]]),
            record("req-train-0002", [[call("list_notes", {"limit": 10}, "e")],
                                      [call("list_notes", {"limit": 10}, "f")],
                                      [call("list_notes", {"limit": 20}, "g")]])]
    habits, account = ts.derive_habits(recs)
    assert habits == {("search_notes", "top_k"): 10}  # list_notes: the default is the value set most often
    assert account == {
        "search_notes.top_k": {"default": 3, "m6": 5, "derived": 10, "calls": 4,
                               "values": {"10": 2, "5": 1, "absent": 1}},
        "list_notes.limit": {"default": 10, "m6": 20, "derived": None, "calls": 3,
                             "values": {"10": 2, "20": 1}}}
    tie = [record("req-train-0003", [[call("search_notes", {"query": "x", "top_k": 7}, "a")],
                                     [call("search_notes", {"query": "y", "top_k": 5}, "b")]])]
    assert ts.derive_habits(tie)[0] == {("search_notes", "top_k"): 5}  # a tie: the smaller canonical JSON
    assert ts.derive_habits([])[0] == {}


# ---- records only: the per-record rules over another run's records (M7) ---------------------------------


def records_only_argv(inputs: dict, *extra: str) -> list[str]:
    """--records-only, with a term list that does not exist: the mode must not read one."""
    d = inputs["dir"]
    return [str(inputs["synth"]), str(inputs["requests"]), "--records-only",
            "--out", str(d / "only" / "synth-m7.jsonl"), "--manifest", str(d / "only" / "manifest.json"),
            "--requests-out", str(d / "only" / "requests.jsonl"), "--terms", str(d / "no-terms.json"),
            "--dev", str(inputs["dev"]), "--corpus", str(CORPUS), *extra]


def only_manifest(inputs: dict) -> dict:
    return json.loads((inputs["dir"] / "only" / "manifest.json").read_text(encoding="utf-8"))


def test_records_only_applies_the_record_rules_and_makes_no_scripted_record(inputs, capsys):
    assert ts.main(records_only_argv(inputs)) == 0
    only = inputs["dir"] / "only"
    assert sorted(p.name for p in only.iterdir()) == ["manifest.json", "synth-m7.jsonl"]  # no requests file
    records = load_records(only / "synth-m7.jsonl")
    assert [r["id"] for r in records] == ["req-train-0001", "req-train-0002", "req-train-0005"]
    by_id = {r["id"]: r for r in records}
    assert [args_of(c) for c in calls_of(by_id["req-train-0001"])] == [{"query": "x"}]  # R1, count 3
    assert [args_of(c) for c in calls_of(by_id["req-train-0005"])] == [{"query": "x", "top_k": 5}]  # count 5
    assert [args_of(c) for c in calls_of(by_id["req-train-0002"])] == [
        {}, {"question": "哪个方向?", "missing": ["paper_id"]}]  # R1 on limit, R2 on missing
    assert all(r["run"] == "train" and not r["family"].startswith("m6/") for r in records)
    manifest = only_manifest(inputs)
    assert manifest["mode"] == "records-only"
    assert set(manifest["rules"]) == {"R1", "R2", "R4"}
    assert manifest["not_run"]["rules"] == ["R3", "R5", "R6", "R7"]
    assert manifest["rules"]["R1"]["deleted"] == {"search_notes.top_k": 1, "list_notes.limit": 1,
                                                  "fetch_fulltext.max_chars": 0}
    assert manifest["rules"]["R1"]["kept_count"]["search_notes.top_k"] == 1
    assert manifest["rules"]["R1"]["habits"] == {"search_notes.top_k": 5, "list_notes.limit": 20}
    assert manifest["rules"]["R2"]["changed"] == 1 and manifest["rules"]["R4"]["late_dropped"] == 1
    assert "terms" not in manifest and "requests_out" not in manifest and "habits" not in manifest
    assert manifest["records"]["in"]["total"] == 4 and manifest["records"]["out"]["total"] == 3
    assert manifest["by_scenario_lang"] == {
        "in": {"clarify": {"zh": 2}, "normal": {"zh": 1}, "recover": {"zh": 1}},
        "after_r4": {"clarify": {"zh": 1}, "normal": {"zh": 1}, "recover": {"zh": 1}},
        "out": {"clarify": {"zh": 1}, "normal": {"zh": 1}, "recover": {"zh": 1}}}
    data = (only / "synth-m7.jsonl").read_bytes()
    assert manifest["output"] == {"path": (only / "synth-m7.jsonl").as_posix(), "records": 3,
                                  "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    text = (only / "manifest.json").read_text(encoding="utf-8")
    assert text == json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    assert "records in 4, out 3" in capsys.readouterr().out


@NEEDS_TERMS
def test_records_only_gives_the_full_runs_records_of_the_input(inputs):
    """The rules over the input records are those of the full run: its output without the scripted records,
    byte for byte."""
    assert ts.main(argv(inputs)) == 0
    assert ts.main(records_only_argv(inputs)) == 0
    full = (inputs["dir"] / "out" / "synth-m6.jsonl").read_text(encoding="utf-8").splitlines()
    only = (inputs["dir"] / "only" / "synth-m7.jsonl").read_text(encoding="utf-8").splitlines()
    assert only == [line for line in full if json.loads(line)["run"] != ts.RUN]


def test_records_only_reruns_byte_identical_and_keeps_out_without_force(inputs):
    only = inputs["dir"] / "only"
    assert ts.main(records_only_argv(inputs)) == 0
    first = {p.name: p.read_bytes() for p in only.iterdir()}
    assert ts.main(records_only_argv(inputs)) == 2
    assert ts.main(records_only_argv(inputs, "--force")) == 0
    assert {p.name: p.read_bytes() for p in only.iterdir()} == first


def test_derive_habits_takes_r1s_values_from_the_input_and_records_how(inputs):
    assert ts.main(records_only_argv(inputs)) == 0
    plain = (inputs["dir"] / "only" / "synth-m7.jsonl").read_bytes()
    assert ts.main(records_only_argv(inputs, "--derive-habits", "--force")) == 0
    assert (inputs["dir"] / "only" / "synth-m7.jsonl").read_bytes() == plain  # the input's habits are M6's
    manifest = only_manifest(inputs)
    assert manifest["habits"] == {
        "search_notes.top_k": {"default": 3, "m6": 5, "derived": 5, "calls": 2, "values": {"5": 2}},
        "list_notes.limit": {"default": 10, "m6": 20, "derived": 20, "calls": 3,
                             "values": {"20": 1, "absent": 2}}}
    assert manifest["rules"]["R1"]["habits"] == {"search_notes.top_k": 5, "list_notes.limit": 20}
    synth = [json.loads(line) for line in inputs["synth"].read_text(encoding="utf-8").splitlines()]
    for rec in synth:  # the teacher of another run writes top_k=4 instead
        for c in calls_of(rec):
            if c["function"]["name"] == "search_notes":
                c["function"]["arguments"] = json.dumps({"query": "x", "top_k": 4}, ensure_ascii=False)
    write_jsonl(inputs["synth"], synth)
    assert ts.main(records_only_argv(inputs, "--derive-habits", "--force")) == 0
    records = {r["id"]: r for r in load_records(inputs["dir"] / "only" / "synth-m7.jsonl")}
    assert [args_of(c) for c in calls_of(records["req-train-0001"])] == [{"query": "x"}]
    assert only_manifest(inputs)["rules"]["R1"]["habits"] == {"search_notes.top_k": 4, "list_notes.limit": 20}


def test_eval_requests_add_the_dev_and_test_texts_of_more_requests_files(inputs):
    """M7's requests file holds train requests only: the dev and test rows of the files --eval-requests names
    join the evaluation texts; their train rows do not."""
    d = inputs["dir"]
    synth = [json.loads(line) for line in inputs["synth"].read_text(encoding="utf-8").splitlines()]
    synth[0]["messages"][1]["content"] = "帮我把 LoRA 方向 9 月以后的论文都找出来,按时间排一下。"
    write_jsonl(inputs["synth"], synth)
    more = write_jsonl(d / "requests-v2.jsonl", [
        request("req-test-0401", split="test", text="帮我把 LoRA 方向 9 月以后的论文都找出来,按时间排一下。"),
        request("req-train-1601", text="帮我找论文")])  # equal to the other records' text, but a train row
    assert ts.main(records_only_argv(inputs)) == 0
    assert only_manifest(inputs)["eval_overlap"]["input_records_dropped"] == []
    assert ts.main(records_only_argv(inputs, "--eval-requests", str(more), "--force")) == 0
    manifest = only_manifest(inputs)
    assert manifest["eval_overlap"]["input_records_dropped"] == ["req-train-0001"]
    assert manifest["records"]["out"]["total"] == 2
    assert manifest["inputs"]["eval_requests"] == [
        {"path": more.as_posix(), "sha256": hashlib.sha256(more.read_bytes()).hexdigest(), "texts": 1}]
    assert manifest["eval_overlap"]["eval_texts"] == 2 + 1 + 1  # the requests file's 2, the dev file's 1


def test_eval_items_add_their_user_messages_and_refuse_test_material(inputs):
    d = inputs["dir"]
    synth = [json.loads(line) for line in inputs["synth"].read_text(encoding="utf-8").splitlines()]
    synth[1]["messages"][1]["content"] = "上周组会提到的那篇论文讲了什么?用一段话概括。"
    write_jsonl(inputs["synth"], synth)
    ext = write_jsonl(d / "dev-ext.jsonl", [{"id": "req-dev-0002-s1", "split": "dev", "messages": [
        {"role": "user", "content": "上周组会提到的那篇论文讲了什么?用一段话概括。"}]}])
    assert ts.main(records_only_argv(inputs, "--eval-items", str(ext))) == 0
    manifest = only_manifest(inputs)
    assert manifest["eval_overlap"]["input_records_dropped"] == ["req-train-0002"]
    assert manifest["inputs"]["eval_items"] == [
        {"path": ext.as_posix(), "sha256": hashlib.sha256(ext.read_bytes()).hexdigest(), "texts": 1}]
    write_jsonl(ext, [{"id": "hw-test-001", "split": "test", "messages": [{"role": "user", "content": "x"}]}])
    assert ts.main(records_only_argv(inputs, "--eval-items", str(ext), "--force")) == 1


def test_records_only_needs_no_term_list_but_the_full_run_does(inputs):
    full = argv(inputs)
    full[full.index("--terms") + 1] = str(inputs["dir"] / "no-terms.json")
    assert ts.main(full) == 2
    assert ts.main(records_only_argv(inputs)) == 0


# ---- FT4 (docs/m7-本地提升计划.md WP6): the opt-in rules R8-R11, R13, R14 ----------------------------------

TRAIN_TOPICS = rq.split_topics(rq.load_corpus(CORPUS), 42)["train"]
NOTES = [{"arxiv_id": "2610.00814", "created": "2026-10-09", "note_id": "2610.00814-1",
          "tags": ["cs.LG", "pretraining-data"],
          "title": "Training-Aware Target Coverage for Data Selection"},
         {"arxiv_id": "2609.32831", "created": "2026-09-30", "note_id": "2609.32831-1",
          "tags": ["cs.LG", "kv-cache"], "title": "UniCache: Type-Aware KV Cache Compression"}]
NOTE_HITS = [{"arxiv_id": "2609.32831", "note_id": "2609.32831-1", "score": 1.2, "snippet": "KV cache memory",
              "title": "UniCache: Type-Aware KV Cache Compression"}]
HITS = [{"arxiv_id": "2609.01234", "published": "2026-09-02", "snippet": "Draft lengths",
         "title": "Adaptive Draft Length for Speculative Decoding"}]
HITS2 = [{"arxiv_id": "2609.05678", "published": "2026-09-20", "snippet": "Tree drafts",
          "title": "Tree Drafting at Scale"}]
PAPER = {"abstract": "We study draft lengths.", "arxiv_id": "2609.01234", "authors": ["A. Author"],
         "published": "2026-09-02", "title": "Adaptive Draft Length for Speculative Decoding"}
RATE_LIMITED = {"error": "RATE_LIMITED", "items": []}
UPSTREAM = {"error": "UPSTREAM_ERROR", "items": []}
OK = "好的。"


def step_msgs(*pairs: tuple[dict, object], text: str = "") -> list[dict]:
    """An assistant step with its calls, then one tool message per call with the given result."""
    message = {"content": text, "role": "assistant", "tool_calls": [c for c, _ in pairs]}
    return [message, *({"content": r if isinstance(r, str) else canonical_json(r), "role": "tool",
                        "tool_call_id": c["id"]} for c, r in pairs)]


def one(name: str, args: dict, result: object, cid: str = "a", **kwargs) -> list[dict]:
    """A step with one call and its result."""
    return step_msgs((call(name, args, cid, **kwargs), result))


def text_msg(text: str) -> list[dict]:
    return [{"content": text, "role": "assistant"}]


def ask_msgs(question: str, missing: list[str], cid: str = "q") -> list[dict]:
    return one("ask_clarification", {"question": question, "missing": missing},
               {"question": question, "status": "asked"}, cid)


def history_record(rid: str, *steps: list[dict], scenario: str = "normal", intent: str = "search_notes",
                   lang: str = "zh", user: str = "帮我找论文", family: str = "x/zh/1") -> dict:
    messages = [{"content": system_prompt(TODAY), "role": "system"}, {"content": user, "role": "user"}]
    for s in steps:
        messages.extend(s)
    return {"family": family, "history": "agent", "id": rid, "intent": intent, "lang": lang,
            "messages": messages, "prompt_hash": PROMPT_HASH, "run": "train", "scenario": scenario,
            "schema_hash": schemas.SCHEMA_HASH, "tools": schemas.MODEL_TOOLS}


def turns_of(rec: dict) -> list[dict]:
    return [m for m in rec["messages"] if m["role"] == "assistant"]


def request_of(rec: dict, **slots) -> dict:
    return request(rec["id"], scenario=rec["scenario"], intent=rec["intent"], lang=rec["lang"],
                   family=rec["family"], slots=slots, text=rec["messages"][1]["content"])


def ft4(records: list[dict], reqs: list[dict] | None = None, **kwargs):
    """apply_ft4 over records (the requests default to ones without slots) with the train topics."""
    if reqs is None:
        reqs = [request_of(r) for r in [*records, *kwargs.get("dropped", [])]]
    return ts.apply_ft4(records, {r["id"]: r for r in reqs}, topics=TRAIN_TOPICS, **kwargs)


def relabel(rid: str, k: int, rule: str) -> dict:
    return {"id": rid, "k": k, "rule": rule, "kind": "relabel", "target": True, "from": [rid, k]}


def s4a_record(rid: str, later: list[list[dict]], *, intent: str = "list_notes",
               family: str = "list_notes/zh/2") -> dict:
    """A note request that listed the notes by a tag (nothing) and then without one (the full listing)."""
    return history_record(rid, one("list_notes", {"tag": "survey"}, [], "a"),
                          one("list_notes", {}, NOTES, "b"), *later, intent=intent, family=family)


def late_clarification(rid: str, notes: list[dict] = NOTES) -> dict:
    """A clarify record that R4 drops (its first ask is at step 4) and that saw the full listing (notes) at
    step 2."""
    return history_record(rid, one("search_notes", {"query": "advisor paper"}, [], "a"),
                          one("list_notes", {}, notes, "b"),
                          one("search_notes", {"query": "recommended"}, [], "c"),
                          ask_msgs("Which paper do you mean?", ["paper_id"]), scenario="clarify",
                          intent="summarize_paper", lang="en", family="summarize_paper/en/8",
                          user="What's the core idea of the paper our lab recommended?")


# R8: after a transient error, exactly the failed calls again


def test_r8_relabels_a_rewritten_retry_as_the_failed_call_and_cuts_after_it():
    failed = call("search_arxiv", {"query": "speculative decoding", "max_results": 5}, "a")
    rewritten = call("search_arxiv", {"query": "speculative decoding LLM", "max_results": 5}, "b")
    rec = history_record("req-train-0101", step_msgs((failed, RATE_LIMITED)),
                         step_msgs((rewritten, HITS), text="Let me try another query."),
                         one("get_paper", {"arxiv_id": "2609.01234"}, PAPER, "c"),
                         text_msg("这篇讲自适应草稿长度。"), intent="find_recent_papers")
    out = ft4([rec])
    (got,) = out.records
    assert len(got["messages"]) == 5 and got["messages"][:4] == rec["messages"][:4]
    retry = got["messages"][4]
    assert retry["role"] == "assistant" and retry["content"] == ""
    assert [c["function"] for c in retry["tool_calls"]] == [failed["function"]]
    assert retry["tool_calls"][0]["id"] not in ("a", "b")
    assert check_call("search_arxiv", args_of(retry["tool_calls"][0]), [], TODAY) == []
    assert out.changes == [relabel("req-train-0101", 1, "R8")] and out.derived == []
    assert out.report["rules"]["R8"]["relabeled"] == 1
    assert rec["messages"][4]["tool_calls"] == [rewritten]  # the input is not changed


def test_r8_repeats_only_the_failed_call_of_a_mixed_step():
    failed = call("search_arxiv", {"query": "KV cache"}, "a")
    fine = call("search_notes", {"query": "KV cache"}, "b")
    rec = history_record("req-train-0102", step_msgs((failed, UPSTREAM), (fine, NOTE_HITS)),
                         one("get_paper", {"arxiv_id": "2609.32831"}, PAPER, "c"), text_msg(OK))
    out = ft4([rec])
    retry = turns_of(out.records[0])[1]
    assert [c["function"] for c in retry["tool_calls"]] == [failed["function"]]
    assert out.report["rules"]["R8"]["mixed_step"] == 1


def test_r8_leaves_an_identical_retry_and_a_retry_after_other_errors_but_drops_added_calls():
    failed = call("search_arxiv", {"query": "KV cache", "sort_by": "relevance"}, "a")
    same = call("search_arxiv", {"sort_by": "relevance", "query": "KV cache"}, "b")  # key order aside
    plain = history_record("req-train-0103", step_msgs((failed, RATE_LIMITED)),
                           step_msgs((same, HITS), text="Let me retry."), text_msg(OK))
    extra = history_record("req-train-0104", step_msgs((failed, RATE_LIMITED)),
                           step_msgs((same, HITS), (call("search_notes", {"query": "KV cache"}, "c"), [])),
                           text_msg(OK))
    empty = history_record("req-train-0105",
                           step_msgs((failed, {"error": "EMPTY_RESULTS", "hint": RELAX_HINT, "items": []})),
                           one("search_arxiv", {"query": "KV"}, HITS, "b"), text_msg(OK))
    out = ft4([plain, extra, empty])
    assert out.records[0] == plain and out.records[2] == empty
    assert [c["function"] for c in turns_of(out.records[1])[1]["tool_calls"]] == [failed["function"]]
    assert out.changes == [relabel("req-train-0104", 1, "R8")]
    assert out.report["rules"]["R8"]["identical"] == 1


def test_r8_stays_out_once_the_failed_call_has_run_twice():
    """The loop refuses a third run of one call (agent.loop.MAX_RUNS_PER_CALL): no retry target there."""
    rec = history_record("req-train-0106", one("search_arxiv", {"query": "KV cache"}, RATE_LIMITED, "a"),
                         one("search_arxiv", {"query": "KV cache"}, RATE_LIMITED, "b"),
                         one("search_arxiv", {"query": "KV cache compression"}, HITS, "c"), text_msg(OK))
    out = ft4([rec])
    assert out.records == [rec] and out.changes == []
    assert out.report["rules"]["R8"]["at_run_limit"] == 1 and out.report["rules"]["R8"]["identical"] == 1


# R9: close a note request once the full listing is in view; report after the third arXiv search


def test_r9_closes_a_list_request_after_the_full_listing_with_an_answer_built_from_it():
    rec = s4a_record("req-train-0201", [one("search_notes", {"query": "survey"}, [], "c"),
                                        text_msg("没有找到。")])
    out = ft4([rec], [request_of(rec, tags=["survey"], since="2026-09-01")])
    (got,) = out.records
    turns = turns_of(got)
    assert len(turns) == 3 and "tool_calls" not in turns[2] and got["messages"][-1] == turns[2]
    answer = turns[2]["content"]
    assert "survey" in answer and all(n["note_id"] in answer and n["title"] in answer for n in NOTES)
    assert "?" not in answer and "？" not in answer
    assert out.changes[0] == relabel("req-train-0201", 2, "R9")
    assert out.report["rules"]["R9"]["s4a"] == {"relabeled": 1, "list": 1,
                                                "pick": {"ask": 0, "one": 0, "none": 0}}


KV_NOTE = {"arxiv_id": "2607.00760", "created": "2026-09-07", "note_id": "2607.00760-1",
           "tags": ["cs.LG", "kv-cache"], "title": "MosaicKV: Dynamic Two-D KV Cache Compression"}


def pick_record(rid: str, listing: list[dict], lang: str = "en") -> dict:
    """A "remembered note" request (PICK_FAMILIES) that listed the notes and then searched them again."""
    user = ("I remember writing a note about KV cache. Can you find it?" if lang == "en"
            else "我记得写过一篇关于 KV cache 的笔记,帮我找出来。")
    return history_record(rid, one("search_notes", {"query": "KV cache"}, [], "a"),
                          one("list_notes", {}, listing, "b"),
                          one("search_notes", {"query": "cache compression"}, [], "c"),
                          text_msg("I could not find it."), lang=lang, family=f"search_notes/{lang}/3",
                          user=user)


def test_r9_asks_which_paper_when_two_listed_notes_carry_the_topic_of_a_remembered_note():
    for lang in ("en", "zh"):
        rec = pick_record("req-train-0202", [*NOTES, KV_NOTE], lang)
        out = ft4([rec], [request_of(rec, topic="KV cache", topic_id="kv-cache")])
        got = out.records[0]
        ask, reply = got["messages"][-2:]
        (c,) = ask["tool_calls"]
        args = args_of(c)
        assert c["function"]["name"] == "ask_clarification" and args["missing"] == ["paper_id"]
        for note in (NOTES[1], KV_NOTE):  # the two candidates, by id and title; not the third note
            assert note["arxiv_id"] in args["question"] and note["title"] in args["question"]
        assert NOTES[0]["arxiv_id"] not in args["question"]
        content = ask["content"]
        assert "2" in content and "kv-cache" in content and "?" not in content and "？" not in content
        assert not any(claim in content for claim in ("tag would", "标签就能", "all ", "全部"))
        assert json.loads(reply["content"]) == {"question": args["question"], "status": "asked"}
        assert reply["tool_call_id"] == c["id"] and len(turns_of(got)) == 3
        assert check_call("ask_clarification", args, [], TODAY) == []
        assert out.report["rules"]["R9"]["s4a"]["pick"] == {"ask": 1, "one": 0, "none": 0}


def test_r9_presents_the_one_listed_note_with_the_topic_or_says_that_none_carries_it():
    rec = pick_record("req-train-0203", NOTES)  # one note carries kv-cache
    out = ft4([rec], [request_of(rec, topic="KV cache", topic_id="kv-cache")])
    answer = turns_of(out.records[0])[2]
    assert "tool_calls" not in answer and "?" not in answer["content"]
    assert NOTES[1]["note_id"] in answer["content"] and NOTES[1]["title"] in answer["content"]
    assert NOTES[0]["note_id"] not in answer["content"]
    assert out.report["rules"]["R9"]["s4a"]["pick"] == {"ask": 0, "one": 1, "none": 0}
    rec = pick_record("req-train-0204", NOTES)  # no listed note carries quantization
    out = ft4([rec], [request_of(rec, topic="quantization", topic_id="quantization")])
    answer = turns_of(out.records[0])[2]["content"]
    assert "None of your notes carries the tag quantization" in answer
    assert all(n["note_id"] in answer and n["title"] in answer for n in NOTES)  # the listing is shown
    assert out.report["rules"]["R9"]["s4a"]["pick"] == {"ask": 0, "one": 0, "none": 1}


NOTES5 = [{"arxiv_id": f"2609.0{i}000", "created": f"2026-09-{10 + i}", "note_id": f"2609.0{i}000-1",
           "tags": ["cs.LG", "kv-cache" if i % 2 else "rag"], "title": f"Paper number {i}"}
          for i in range(5, 0, -1)]  # newest first, as list_notes gives them


def table_rows(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("| ") and "---" not in line][1:]


def test_a_listing_is_complete_only_without_since_or_tag_and_under_its_limit():
    assert ts.Listing({}, NOTES).complete and ts.Listing({"limit": 3}, NOTES).complete
    for args, notes in (({"limit": 5}, NOTES5), ({}, NOTES5 * 2), ({"since": "2026-09-01"}, NOTES),
                        ({"limit": 2}, NOTES)):
        assert not ts.Listing(args, notes).complete
    complete = ts.listing_answer(ts.Listing({}, NOTES), {}, "zh")
    assert complete.startswith("笔记库里现有的 2 条笔记如下") and len(table_rows(complete)) == 2
    for lang, said, claim in (("zh", "列出的 5 条笔记", "现有"), ("en", "the 5 notes listed", "you have")):
        text = ts.listing_answer(ts.Listing({"limit": 5}, NOTES5), {}, lang)
        assert said in text and claim not in text and len(table_rows(text)) == 5
        text = ts.listing_answer(ts.Listing({"limit": 5}, NOTES5), {"tags": ["survey"]}, lang)
        assert said in text and claim not in text and "survey" in text


def test_a_listing_answer_shows_the_count_asked_for_newest_first_and_says_so():
    text = ts.listing_answer(ts.Listing({}, NOTES5[::-1][:3]), {"count": 1}, "zh")  # oldest first given
    assert "最新的 1 条" in text and table_rows(text) == [table_rows(ts._note_table([NOTES5[2]], "zh"))[0]]
    text = ts.listing_answer(ts.Listing({"limit": 5}, NOTES5), {"count": 2}, "en")
    assert "the 2 newest" in text and "the 5 notes listed" in text and "you have" not in text
    assert [row.split(" | ")[0] for row in table_rows(text)] == ["| 2609.05000-1", "| 2609.04000-1"]
    text = ts.listing_answer(ts.Listing({"limit": 5}, NOTES5), {"tags": ["kv-cache"], "count": 1}, "en")
    assert "3 of the notes listed carry the tag kv-cache; this is the newest, as you asked:" in text
    assert len(table_rows(text)) == 1 and "2609.05000-1" in text
    text = ts.listing_answer(ts.Listing({"limit": 5}, NOTES5), {"tags": ["survey"], "count": 1}, "en")
    assert "None of the 5 notes listed carries the tag survey. This is the newest of the notes listed" in text
    text = ts.listing_answer(ts.Listing({}, NOTES), {"count": 3}, "en")  # fewer notes than asked: all of them
    assert len(table_rows(text)) == 2 and "These are the 2 notes you have" in text


def test_r9_closes_only_after_a_complete_listing():
    """S4a needs the listing seen to be complete (Listing.complete: no since, no tag, fewer notes than its
    limit), the rule of the listing texts: after a listing that hit its limit or had a since, the note query
    stays (round 2 closed there: req-train-0122, 0394, 2045)."""
    def note_request(rid: str, listing: dict, notes: list[dict]) -> dict:
        return history_record(rid, one("list_notes", listing, notes, "a"),
                              one("search_notes", {"query": "survey"}, [], "b"), text_msg("没有。"),
                              intent="list_notes", family="list_notes/zh/1")

    capped = note_request("req-train-0217", {"limit": 5}, NOTES5)
    since = note_request("req-train-0218", {"since": "2026-09-01"}, NOTES)
    full = note_request("req-train-0219", {}, NOTES)
    out = ft4([capped, since, full], [request_of(r, count=2) for r in (capped, since, full)])
    assert out.records[:2] == [capped, since] and out.changes == [relabel("req-train-0219", 1, "R9")]
    assert "现有的 2 条笔记" in turns_of(out.records[2])[1]["content"]
    assert out.report["rules"]["R9"]["s4a"]["relabeled"] == 1
    assert out.report["targets_in_input"]["R9"]["s4a_note_queries"] == 1


def test_r9_leaves_closing_steps_other_tools_other_intents_and_partial_listings():
    closing = s4a_record("req-train-0203", [text_msg("笔记如下。")])
    reading = s4a_record("req-train-0204", [one("get_paper", {"arxiv_id": "2610.00814"}, PAPER, "c"),
                                            text_msg(OK)])
    whats_new = s4a_record("req-train-0205", [one("search_notes", {"query": "x"}, [], "c"), text_msg(OK)],
                           intent="whats_new", family="whats_new/zh/1")
    tagged = history_record("req-train-0206", one("list_notes", {"tag": "kv-cache"}, NOTES, "a"),
                            one("search_notes", {"query": "x"}, [], "c"), text_msg(OK), intent="list_notes")
    empty = history_record("req-train-0207", one("list_notes", {}, [], "a"),
                           one("search_notes", {"query": "x"}, [], "c"), text_msg(OK), intent="list_notes")
    records = [closing, reading, whats_new, tagged, empty]
    out = ft4(records)
    assert out.records == records and out.changes == [] and out.derived == []


def arxiv_steps(*results: list) -> list[list[dict]]:
    return [one("search_arxiv", {"query": f"query {i}"}, r, f"s{i}") for i, r in enumerate(results)]


def test_r9_turns_a_fourth_arxiv_search_into_an_honest_report():
    rec = history_record("req-train-0211", *arxiv_steps(HITS, [], HITS2),
                         one("search_arxiv", {"query": "query 3"}, HITS, "s3"), text_msg(OK),
                         intent="find_recent_papers")
    out = ft4([rec])
    turns = turns_of(out.records[0])
    assert len(turns) == 4 and "tool_calls" not in turns[3]
    report = turns[3]["content"]
    assert all(s in report for s in ("2609.01234", "2609.05678", "query 0", "query 1", "query 2"))
    assert "?" not in report and "？" not in report
    assert out.changes == [relabel("req-train-0211", 3, "R9")]
    assert out.report["rules"]["R9"]["s4b"]["relabeled"] == 1


def test_r9_s4b_needs_three_searches_before_a_fourth():
    third = history_record("req-train-0212", *arxiv_steps(HITS, []),
                           one("search_arxiv", {"query": "q"}, HITS2, "s9"), text_msg(OK))
    reading = history_record("req-train-0213", *arxiv_steps(HITS, [], HITS2),
                             one("get_paper", {"arxiv_id": "2609.01234"}, PAPER, "g"), text_msg(OK))
    out = ft4([third, reading])
    assert out.records == [third, reading] and out.changes == []


def test_the_targets_in_the_input_are_counted_as_the_analysis_counted_them_cut_or_not():
    """Every turn a rule's situation holds for, in the records as they came (before any cut): the counts
    the coverage analysis gave per rule; a cut leaves at most one relabel per record."""
    rec = s4a_record("req-train-0215", [one("search_notes", {"query": "survey"}, [], "c"),
                                        one("list_notes", {"tag": "surveys"}, [], "d"),
                                        one("search_arxiv", {"query": "q"}, UPSTREAM, "e"),
                                        one("search_arxiv", {"query": "q2"}, HITS, "f"), text_msg(OK)])
    out = ft4([rec], [request_of(rec, tags=["survey"])])
    assert out.report["rules"]["R9"]["s4a"]["relabeled"] == 1
    assert out.report["targets_in_input"] == {
        "R8": {"transient_steps": 1, "not_identical": 1, "identical": 0, "at_run_limit": 0},
        "R9": {"s4a_note_queries": 2, "s4b_arxiv_searches": 0}, "R10": {"labelled": 0},
        "R11": {"records": 0}, "R13": {"pure_chinese": 0, "with_a_term": 0}}


def r8_then_s4a(rid: str) -> dict:
    """A list request that R8 cuts at turn 1 (a rewritten retry) and whose cut removes a later S4a point: the
    full listing at turn 2, then a note query again at turn 3."""
    return history_record(rid, one("search_arxiv", {"query": "survey"}, RATE_LIMITED, "a"),
                          one("search_arxiv", {"query": "survey papers"}, HITS, "b"),
                          one("list_notes", {}, NOTES, "c"),
                          one("search_notes", {"query": "survey"}, [], "d"),
                          text_msg("没有找到。"), intent="list_notes", family="list_notes/zh/2",
                          user="把我的笔记都列出来,看看有没有综述。")


def test_r9_constructs_closings_only_at_s4a_points_a_cut_by_r8_r10_or_r11_removed():
    after_r9 = s4a_record("req-train-0214", [one("search_notes", {"query": "survey"}, [], "c"),
                                             one("list_notes", {"tag": "surveys"}, [], "d"),
                                             one("search_notes", {"query": "surveys"}, [], "e"),
                                             text_msg("没有找到。")])
    after_r8 = r8_then_s4a("req-train-0216")
    reqs = [request_of(after_r9, tags=["survey"]), request_of(after_r8, tags=["survey"])]
    out = ft4([after_r9, after_r8], reqs)
    assert out.changes[0] == relabel("req-train-0214", 2, "R9")  # its later S4a points stay unbuilt
    assert relabel("req-train-0216", 1, "R8") in out.changes
    (derived,) = out.derived
    assert derived["id"] == "req-train-0216-ft4r9k3" and derived["messages"][:8] == after_r8["messages"][:8]
    assert derived["run"] == ts.FT4_RUN and derived["family"] == "ft4/r9/list_notes/zh/2"
    assert derived["context_turns"] == 3 and "context_turns" not in out.records[1]
    closing = turns_of(derived)[3]
    assert "tool_calls" not in closing and "survey" in closing["content"]
    rows = [(c["k"], c["kind"], c["target"], c["from"]) for c in out.changes if c["id"] == derived["id"]]
    assert rows == [(0, "copy", False, ["req-train-0216", 0]), (1, "copy", False, ["req-train-0216", 1]),
                    (2, "copy", False, ["req-train-0216", 2]), (3, "new", True, ["req-train-0216", 3])]
    samples, _ = build_step_samples(derived, TOK)
    whole, _ = build_step_samples({k: v for k, v in derived.items() if k != "context_turns"}, TOK)
    assert len(whole) == 4 and samples == whole[-1:]  # the trainer takes the closing only
    assert out.report["rules"]["R9"]["constructed"] == {
        "records": 1, "candidates": 1, "target": 100, "not_built_after_r9_cuts": 2}


# R10: the labelled text questions become ask_clarification; asks derived where the full listing was seen

R10_TEXT = ("`reading-group` 标签下目前没有任何笔记。\n\n| # | 标题 |\n|---|---|\n| 1 | Doc-REFRAG |\n\n"
            "看起来并不存在 `reading-group` 这个标签。请问你想用哪个标签来筛选?"
            "比如 `rag`、`kv-cache` 等,或者按时间直接列出最新的 5 条笔记。告诉我后我再帮你精确列出。")
R10_CLEAN = R10_TEXT.removesuffix("告诉我后我再帮你精确列出。")  # the questions are the trailing block
# The end of synth-m7 req-train-1665 turn 4 (R10_LABELLED). Its last line holds "e.g." in mid-sentence;
# round 2 cut the sentence there and left "Or did you mean a different tag — e.g." in the text.
LINE_1665 = ("- Or did you mean a different tag — e.g. did you intend `position-extrapolation`, which is "
             "where your rotary material actually lives?")
ASK_1665 = ("Would you like me to:\n"
            "- **Read out the MeRoPE note** (`2609.01252`) in full, and/or the `2608.10296` note?\n"
            "- **Search arXiv** for recent rotary-position papers to add to your library?\n" + LINE_1665)
TEXT_1665 = ("I stopped here rather than pulling full text, since your request was about your own notes "
             "rather than new papers.\n\n" + ASK_1665)
# The end of synth-m7 req-train-2420 turn 4: a list led by a colon whose first item does not ask.
TEXT_2420 = ("笔记已保存为 `2609.40316-1.md`（scaffold 模式，仅证据句）。下面有两处需要你确认或补充：\n"
             "1. `ScaLoRA` 只在摘要片段里出现过，我没能核实其含义，已标为 open term；\n"
             "2. 你 9 月 19 日读的那篇具体是哪一篇？把它告诉我，我就能做真正的“增量对比”。\n\n"
             "如果需要，我可以继续为 2609.36263（配对多模态）或 2609.27588（能力流形）补充笔记；"
             "也可以把范围扩大到 9 月全月的缩放定律综述性工作，做一份对比表。")


def test_a_full_stop_after_an_abbreviation_or_a_single_letter_ends_no_sentence():
    body = LINE_1665.removeprefix("- ")
    assert ts._sentences(body) == [body]  # one question: "e.g." ends no sentence
    probe = "Which one, e.g. the RoPE paper or the ALiBi one?"
    assert ts._sentences(probe) == [probe] and ts.split_question(probe) == ("", probe, None)
    for sentence in ("Use RoPE, e.g. the original one.", "Use the base model, i.e. the one before tuning.",
                     "It covers RoPE, ALiBi, etc. in one table.", "Compare RoPE vs. ALiBi first.",
                     "See the appendix, cf. the main text.", "Vaswani et al. introduced it.",
                     "It takes approx. two hours.", "Ask J. Smith about it.", "E.g. the RoPE one is fine.",
                     "Cf. the main text.", "ETC. is fine."):
        assert ts._sentences(f"{sentence} Done.") == [sentence, " Done."], sentence
    assert ts._sentences("I chose the plan. Which one?") == ["I chose the plan.", " Which one?"]
    assert not ts._ends_sentence("- papers on RoPE, ALiBi, etc.") and ts._ends_sentence("- papers on RoPE.")


def test_split_question_moves_only_a_trailing_block_of_questions():
    """The questions move out of the text only as its trailing block: after the first question come only
    more questions, the options of a question that ends its line and the sentences or lines that elaborate a
    question right before them (比如, 例如, 或者, 或, e.g., for example, such as, or). Anything else after the
    first question leaves the split unclean (not_trailing), and so does a list led by a colon whose items do
    not all move (partial_list)."""
    # the review's three broken cases: synth-m7 req-train-1665 turn 4 is clean now; req-train-2420 turn 4 and
    # synth-m6 req-train-1385 turn 2 are not
    assert ts.split_question(TEXT_1665) == (TEXT_1665.split("\n\n")[0], ASK_1665, None)
    assert ts.split_question(TEXT_2420).skipped == "partial_list"
    assert ts.split_question(R10_TEXT).skipped == "not_trailing"  # 告诉我后我再帮你精确列出。
    kept = ("`reading-group` 标签下目前没有任何笔记。\n\n| # | 标题 |\n|---|---|\n| 1 | Doc-REFRAG |\n\n"
            "看起来并不存在 `reading-group` 这个标签。")
    asked = "请问你想用哪个标签来筛选?比如 `rag`、`kv-cache` 等,或者按时间直接列出最新的 5 条笔记。"
    assert ts.split_question(R10_CLEAN) == (kept, asked, None)
    asks = "Which tag? For example rag or kv-cache. Or by date?"
    assert ts.split_question(asks) == ("", asks, None)
    assert ts.split_question(asks + " Tell me.").skipped == "not_trailing"
    table = "I found two notes.\n\n| Title |\n|---|\n| Why Not Both? |"
    ask = "Would you like me to:\n1. Read the first one, or\n2. Search arXiv instead?"
    assert ts.split_question(f"{table}\n\n{ask}") == (table, ask, None)
    assert ts.split_question(f"{table}\n\n{ask}\n\nLet me know which direction and I'll continue.").skipped \
        == "not_trailing"
    listed = "你想找的是哪一篇?\n\n- 如果是 9 月 1 日那篇;\n- 如果是 9 月 28 日那篇;"
    options = "你想找的是哪一篇?\n- 如果是 9 月 1 日那篇;\n- 如果是 9 月 28 日那篇;"
    assert ts.split_question(listed) == ("", options, None)  # the options of a question that ends its line
    assert ts.split_question(listed + "\n\n告诉我是哪一篇。").skipped == "not_trailing"
    for text in ("Which one did you have in mind? If you tell me, I can open it.",
                 "I found two.\n\n  Which one? If you tell me, I open it.",
                 "Which one?\n\n| A | B |",
                 "你想看哪一篇？或许两篇都要。"):  # 或许 (perhaps) elaborates nothing
        assert ts.split_question(text).skipped == "not_trailing", text
    assert ts.split_question("I found two.\n\n  Which one? Or the newest, if you like.") == (
        "I found two.", "Which one? Or the newest, if you like.", None)
    assert ts.split_question("你说的是哪个标签?\n\n例如 `rag` 或 `kv-cache`。") == (
        "", "你说的是哪个标签?\n例如 `rag` 或 `kv-cache`。", None)  # a line that elaborates the one before
    assert ts.split_question("Which area? Such as RoPE or ALiBi.") == (
        "", "Which area? Such as RoPE or ALiBi.", None)
    offer = ("Which one were you thinking of? Or if you meant another angle, tell me and I can search "
             "further — I can also pull up the full note contents.")
    assert ts.split_question(offer) == ("", offer, None)
    lead = "Saved the note. Two things to confirm:\n"
    assert ts.split_question(lead + "1. A is unclear;\n2. Which paper did you read?").skipped \
        == "partial_list"
    assert ts.split_question(lead + "1. Is A right?\n2. Which paper did you read? Tell me.").skipped \
        == "partial_list"
    assert ts.split_question(lead + "1. Is A right?\n2. Which paper did you read?") == (
        "Saved the note.", "Two things to confirm:\n1. Is A right?\n2. Which paper did you read?", None)
    stems = ("Would you like me to:\n1. Pull the details, or\n2. Search arXiv instead, or\n"
             "3. Drop the tag filter?")  # items that run on into one question move together
    assert ts.split_question(stems) == ("", stems, None)
    assert ts.split_question(stems + "\n\nLet me know.").skipped == "not_trailing"
    assert ts.split_question("I found two notes. Both are about RoPE.") == (
        "I found two notes. Both are about RoPE.", "", None)  # it asks nothing


def test_r10_turns_a_labelled_text_question_into_ask_clarification():
    rid = "req-train-1385"
    assert ts.R10_LABELLED[rid, 2] == ["tags"] and len(ts.R10_LABELLED) == 16
    rec = history_record(rid, one("list_notes", {"tag": "reading-group"}, [], "a"),
                         one("list_notes", {}, NOTES, "b"), text_msg(R10_CLEAN), intent="list_notes")
    out = ft4([rec])
    ask, reply = out.records[0]["messages"][-2:]
    content, question, _ = ts.split_question(R10_CLEAN)
    assert args_of(ask["tool_calls"][0]) == {"question": question, "missing": ["tags"]}
    assert ask["content"] == content and json.loads(reply["content"])["status"] == "asked"
    assert out.changes == [relabel(rid, 2, "R10")] and out.report["rules"]["R10"]["relabeled"] == 1


def test_r10_relabels_a_labelled_text_question_only_when_its_split_is_clean():
    """The review's three broken R10 targets: synth-m7 req-train-1665 turn 4 is relabelled now; req-train-2420
    turn 4 (a list whose first item stays) and synth-m6 req-train-1385 turn 2 (a sentence after the question)
    stay as the teacher wrote them, counted by reason."""
    looked = [one("search_notes", {"query": f"note {n}"}, NOTE_HITS, f"s{n}") for n in range(4)]
    clean = history_record("req-train-1665", *looked, text_msg(TEXT_1665), lang="en")
    partial = history_record("req-train-2420", *looked, text_msg(TEXT_2420))
    trailing = history_record("req-train-1385", *looked[:2], text_msg(R10_TEXT), intent="list_notes")
    out = ft4([clean, partial, trailing])
    assert out.records[1:] == [partial, trailing] and out.changes == [relabel("req-train-1665", 4, "R10")]
    ask = turns_of(out.records[0])[4]
    assert args_of(ask["tool_calls"][0]) == {"question": ASK_1665, "missing": ["tags"]}
    assert ask["content"] == TEXT_1665.split("\n\n")[0]
    r10 = out.report["rules"]["R10"]
    assert r10["relabeled"] == 1 and r10["labelled_in_input"] == 3
    assert r10["skipped_not_trailing"] == ["req-train-1385:2"]
    assert r10["skipped_partial_list"] == ["req-train-2420:4"]


def test_r10_leaves_borderline_and_unlabelled_questions_and_refuses_a_label_that_does_not_fit():
    borderline = history_record("req-train-0538", one("get_paper", {"arxiv_id": "2609.01234"}, PAPER),
                                text_msg("你手头另有该文的编号吗?"), intent="lookup_paper")
    plain = history_record("req-train-0301", one("search_notes", {"query": "KV cache"}, NOTE_HITS),
                           text_msg("你想看哪一篇?"))
    out = ft4([borderline, plain])
    assert out.records == [borderline, plain] and out.changes == []
    misfit = history_record("req-train-1385", one("list_notes", {"tag": "x"}, [], "a"),
                            one("list_notes", {}, NOTES, "b"),
                            one("get_paper", {"arxiv_id": "2609.32831"}, PAPER, "c"), text_msg(OK),
                            intent="list_notes")
    with pytest.raises(ts.TransformError, match="req-train-1385"):
        ft4([misfit])


def test_r10_derives_an_ask_at_the_listing_from_a_record_r4_dropped():
    dropped = late_clarification("req-train-0305")
    out = ft4([], [request_of(dropped, missing=["paper_id"])], dropped=[dropped])
    (derived,) = out.derived
    assert derived["id"] == "req-train-0305-ft4r10k2" and derived["messages"][:6] == dropped["messages"][:6]
    ask = turns_of(derived)[2]
    args = args_of(ask["tool_calls"][0])
    assert args["missing"] == ["paper_id"] and args["question"] in ts.CLARIFY_QUESTIONS["paper_id"]["en"]
    assert ask["content"] and "?" not in ask["content"]
    assert [c["kind"] for c in out.changes] == ["copy", "copy", "new"]
    assert out.report["rules"]["R10"]["derived"]["from_r4_dropped"] == {
        "new": 1, "records": 1, "with_a_listing_point": 1, "with_a_complete_listing": 1}


def test_r10_derives_one_ask_per_record_r4_dropped_at_its_first_complete_listing():
    """The new asks follow a full listing only (Listing.complete), one listing point per source record: its
    first complete one."""
    rec = history_record("req-train-0306", one("search_notes", {"query": "advisor paper"}, [], "a"),
                         one("list_notes", {"since": "2026-09-01"}, NOTES, "b"),  # a partial listing first
                         one("list_notes", {}, NOTES, "c"),
                         one("search_notes", {"query": "recommended"}, [], "d"),
                         one("search_notes", {"query": "lab"}, [], "e"),
                         ask_msgs("Which paper do you mean?", ["paper_id"]), scenario="clarify",
                         intent="summarize_paper", lang="en", family="summarize_paper/en/8",
                         user="What's the core idea of the paper our lab recommended?")
    capped = history_record("req-train-0307", one("search_notes", {"query": "advisor paper"}, [], "a"),
                            one("list_notes", {"limit": 5}, NOTES5, "b"),  # a listing that hit its limit
                            one("search_notes", {"query": "recommended"}, [], "c"),
                            ask_msgs("Which paper do you mean?", ["paper_id"]), scenario="clarify",
                            intent="summarize_paper", lang="en", family="summarize_paper/en/8",
                            user="Summarise the paper my advisor sent this morning.")
    reqs = [request_of(r, missing=["paper_id"]) for r in (rec, capped)]
    out = ft4([], reqs, dropped=[rec, capped])
    assert [(d["id"], d["context_turns"], len(turns_of(d))) for d in out.derived] == [
        ("req-train-0306-ft4r10k3", 3, 4)]
    (derived,) = out.derived
    assert args_of(turns_of(derived)[-1]["tool_calls"][0])["missing"] == ["paper_id"]
    assert {c["k"] for c in out.changes if not c["target"]} == {0, 1, 2}
    assert "there are 2" in turns_of(derived)[-1]["content"]  # all the user's notes: the listing is complete
    assert out.report["rules"]["R10"]["derived"]["from_r4_dropped"] == {
        "new": 1, "records": 1, "with_a_listing_point": 2, "with_a_complete_listing": 1}


def test_r10_new_asks_with_one_target_are_built_once():
    """No two new samples share a target (text and call): the first in the hashed order is built, the others
    are counted."""
    question = {}
    for i in range(20, 80):  # two records whose asks pick the same question (by the hash of the derived id)
        rid = f"req-train-03{i:02d}"
        question.setdefault(ts.clarify_question(["paper_id"], "en", f"{rid}-ft4r10k2"), []).append(rid)
    twins = next(ids for ids in question.values() if len(ids) >= 2)[:2]
    dropped = [late_clarification(rid) for rid in twins]
    out = ft4([], [request_of(r, missing=["paper_id"]) for r in dropped], dropped=dropped)
    (derived,) = out.derived
    derived_r10 = out.report["rules"]["R10"]["derived"]
    assert derived_r10["new"] == 1 and derived_r10["candidates_new"] == 2
    assert derived_r10["duplicate_targets"] == [f"{rid}-ft4r10k2" for rid in twins
                                                if f"{rid}-ft4r10k2" != derived["id"]]


def test_the_derived_samples_are_capped_in_a_hashed_order():
    listings = [[NOTES[0]], [NOTES[1]], NOTES, NOTES5[:2], NOTES5[:3], NOTES5[:4]]  # six different targets
    dropped = [late_clarification(f"req-train-03{i:02d}", notes) for i, notes in zip(range(10, 16), listings)]
    reqs = [request_of(r, missing=["paper_id"]) for r in dropped]
    full = ft4([], reqs, dropped=dropped)
    capped = ft4([], reqs, dropped=dropped, targets={"r10_new": 2})
    assert len(full.derived) == 6 and len(capped.derived) == 2
    assert {d["id"] for d in capped.derived} <= {d["id"] for d in full.derived}
    assert capped.report["rules"]["R10"]["derived"]["candidates_new"] == 6
    assert ft4([], reqs, dropped=dropped[::-1], targets={"r10_new": 2}).derived == capped.derived
    looked = step_msgs((call("list_notes", {}, "a"), NOTES))
    kept = history_record("req-train-0407", looked, ask_msgs("哪个方向?", ["topic"]), scenario="clarify",
                          intent="find_recent_papers")  # R11: the teacher's ask after the listing is kept
    none = ft4([kept], [*reqs, request_of(kept, missing=["topic"])], dropped=dropped, targets={"r10_new": 0})
    assert [d["id"] for d in none.derived] == ["req-train-0407-ft4r10k1"]  # the cap counts new asks only


# R11: ask at once when the topic is missing; relative dates


def test_r11_asks_at_once_where_a_missing_topic_record_listed_notes_and_keeps_the_teachers_later_ask():
    """The teacher already asked after the listing (R11's records ask at their second step): the derived
    record keeps that turn as its target, a copy with target true, and it is no new decision."""
    looked = step_msgs((call("list_notes", {}, "a"), NOTES), text="我先看笔记。")
    rec = history_record("req-train-0401", looked,
                         ask_msgs("你说的是哪个方向?", ["topic"]), scenario="clarify",
                         intent="find_recent_papers",
                         user="帮我找找 9 月以来我们组在做的那个方向的论文,两篇就行。")
    out = ft4([rec], [request_of(rec, missing=["topic"])])
    got = out.records[0]
    assert [m["role"] for m in got["messages"]] == ["system", "user", "assistant", "tool"]
    ask = got["messages"][2]
    question = ts.clarify_question(["topic"], "zh", "req-train-0401")
    assert ask["content"] == ""
    assert args_of(ask["tool_calls"][0]) == {"question": question, "missing": ["topic"]}
    (derived,) = out.derived
    assert derived["id"] == "req-train-0401-ft4r10k1" and derived["family"].startswith("ft4/r10/")
    assert derived["messages"] == rec["messages"] and derived["context_turns"] == 1  # the teacher's ask kept
    assert reasoning.is_scripted(derived) and not reasoning.is_scripted(got)
    copied = {"id": "req-train-0401-ft4r10k1", "rule": "R10", "kind": "copy"}
    assert out.changes == [relabel("req-train-0401", 0, "R11"),
                           {**copied, "k": 0, "target": False, "from": ["req-train-0401", 0]},
                           {**copied, "k": 1, "target": True, "from": ["req-train-0401", 1]}]
    assert len(build_step_samples(derived, TOK)[0]) == 1
    assert out.report["rules"]["R11"]["relabeled"] == 1
    assert out.report["rules"]["R10"]["derived"] == {
        "records": 1, "new": 0, "copy_target": 1, "candidates_new": 0, "duplicate_targets": [], "target": 150,
        "from_r11": {"new": 0, "copy_target": 1},
        "from_r4_dropped": {"new": 0, "records": 0, "with_a_listing_point": 0, "with_a_complete_listing": 0}}


def test_r10_derives_a_new_ask_where_the_teacher_queried_the_notes_again_after_the_listing():
    rec = history_record("req-train-0406", one("list_notes", {}, NOTES, "a"),
                         one("search_notes", {"query": "方向"}, [], "b"),
                         ask_msgs("你说的是哪个方向?", ["topic"]), scenario="clarify",
                         intent="find_recent_papers", user="那个方向最近有什么新论文?")
    out = ft4([], [request_of(rec, missing=["topic"])], dropped=[rec])  # R4 drops it: an ask at step 3
    (derived,) = out.derived
    assert derived["id"] == "req-train-0406-ft4r10k1" and derived["context_turns"] == 1
    later = turns_of(derived)[1]
    args = args_of(later["tool_calls"][0])
    assert args["missing"] == ["topic"] and args["question"] in ts.CLARIFY_QUESTIONS["topic"]["zh"]
    assert later["content"] and "?" not in later["content"] and str(len(NOTES)) in later["content"]
    assert json.loads(derived["messages"][-1]["content"])["status"] == "asked"
    assert [(c["k"], c["kind"], c["target"]) for c in out.changes] == [(0, "copy", False), (1, "new", True)]
    assert out.report["rules"]["R10"]["derived"]["from_r4_dropped"] == {
        "new": 1, "records": 1, "with_a_listing_point": 1, "with_a_complete_listing": 1}


def test_r11_leaves_paper_requests_first_step_asks_and_first_steps_off_the_notes():
    paper = history_record("req-train-0402", one("list_notes", {}, NOTES), ask_msgs("哪一篇?", ["paper_id"]),
                           scenario="clarify", intent="summarize_paper")
    first = history_record("req-train-0403", ask_msgs("哪个方向?", ["topic"]), scenario="clarify",
                           intent="find_recent_papers")
    arxiv = history_record("req-train-0404", one("search_arxiv", {"query": "new papers"}, HITS),
                           ask_msgs("哪个方向?", ["topic"]), scenario="clarify", intent="find_recent_papers")
    out = ft4([paper, first, arxiv])
    assert out.records == [paper, first, arxiv] and out.changes == [] and out.derived == []


def test_relative_ranges_count_back_from_today():
    assert ts.relative_range("yesterday", TODAY) == ("2026-10-09", "2026-10-09")
    assert ts.relative_range("day_before", TODAY) == ("2026-10-08", "2026-10-08")
    assert ts.relative_range("this_week", TODAY) == ("2026-10-05", "2026-10-10")  # TODAY is a Saturday
    assert ts.relative_range("last_week", TODAY) == ("2026-09-28", "2026-10-04")
    assert ts.relative_range("last_month", TODAY) == ("2026-09-01", "2026-09-30")
    assert ts.relative_range("this_week", "2026-10-05") == ("2026-10-05", "2026-10-05")
    assert ts.relative_range("last_month", "2026-01-15") == ("2025-12-01", "2025-12-31")


def test_the_new_templates_pass_the_template_audit():
    for templates in (ts.RELATIVE_TEMPLATES, ts.EXPLAIN_TEMPLATES):
        result = ts.audit_new_templates(templates)
        assert result["ok"] and result["max_jaccard"] < rq.TEMPLATE_LIMIT
    assert len(ts.RELATIVE_TEMPLATES["zh"]) == len(ts.RELATIVE_TEMPLATES["en"]) == len(ts.RELATIVE_INTENTS)


def test_relative_date_requests_and_records_use_the_range_counted_from_today():
    corpus = rq.load_corpus(CORPUS)

    def made():
        return ts.generate_relative_requests(12, seed=42, corpus=corpus, split_seed=42, today=TODAY,
                                             guard=ts.EvalGuard(eval_texts=[], train_texts=[]))

    requests = made()
    assert len(requests) == 12 and sum(r["lang"] == "zh" for r in requests) == 6
    train_ids = {t.id for t in TRAIN_TOPICS}
    for r in requests:
        start, end = ts.relative_range(r["slots"]["relative"], TODAY)
        assert r["family"].startswith("ft4/r11/") and r["split"] == "train" and r["scenario"] == "normal"
        rec = ts.relative_date_record({"id": "req-ft4-0001", **r}, TODAY)
        (c,) = turns_of(rec)[0]["tool_calls"]
        args = args_of(c)
        if r["intent"] == "list_notes":
            assert c["function"]["name"] == "list_notes" and args["since"] == start == r["slots"]["since"]
        else:
            assert c["function"]["name"] == "search_arxiv" and args["date_from"] == start
            assert args.get("date_to") == (end if end < TODAY else None)
            assert args["query"] == r["slots"]["topic"] and r["slots"]["topic_id"] in train_ids
        assert rec["run"] == ts.FT4_RUN and rec["family"] == r["family"] and steps_of(rec) == 1
    assert made() == requests


def dated(rid: str, intent: str, *sorts: str | None, date: bool = True) -> dict:
    """A record whose steps search arXiv with (date) or without a date_from, sort_by as given (None:
    absent)."""
    steps = []
    for i, sort in enumerate(sorts):
        args = {"query": "KV cache", **({"date_from": "2026-09-01"} if date else {}),
                **({"sort_by": sort} if sort else {})}
        steps.append(one("search_arxiv", args, HITS, f"s{i}"))
    return history_record(rid, *steps, text_msg(OK), intent=intent)


def test_the_relative_date_searches_take_the_teachers_majority_sort_and_leave_out_the_default():
    """The owner's rule: the effective sort_by of the teacher's search_arxiv calls with a date_from in the
    find_recent_papers records (R11's search intent) of every input, a call without sort_by counted as the
    default relevance; the majority value, left out when it is the default (as R14 leaves defaults out)."""
    m6 = [dated("req-train-0601", "find_recent_papers", "submittedDate", "submittedDate", "submittedDate"),
          dated("req-train-0602", "whats_new", "submittedDate", "submittedDate"),  # another intent
          dated("req-train-0603", "find_recent_papers", "relevance", "relevance", date=False)]  # no date
    chosen, account = ts.relative_sort_by([("synth.jsonl", m6)])
    assert chosen == "submittedDate"
    assert account == {"chosen": "submittedDate", "majority": "submittedDate", "default": "relevance",
                       "intents": ["find_recent_papers"], "rule": ts.RELATIVE_SORT_RULE,
                       "counts": {"synth.jsonl": {"submittedDate": 3}}, "effective": {"submittedDate": 3}}
    m7 = [dated("req-train-1601", "find_recent_papers", "relevance", "relevance", None, None, None)]
    chosen, account = ts.relative_sort_by([("synth.jsonl", m6), ("synth-m7.jsonl", m7)])
    assert chosen is None and account["majority"] == "relevance"  # 2 + 3 without sort_by against 3
    assert account["counts"] == {"synth.jsonl": {"submittedDate": 3},
                                 "synth-m7.jsonl": {"absent": 3, "relevance": 2}}
    assert account["effective"] == {"relevance": 5, "submittedDate": 3} and account["chosen"] is None
    tie = [dated("req-train-0605", "find_recent_papers", "relevance", "submittedDate")]
    assert ts.relative_sort_by([("x", tie)])[0] is None  # a tie goes to the default
    absent = [dated("req-train-0606", "find_recent_papers", None, None, "submittedDate")]
    chosen, account = ts.relative_sort_by([("x", absent)])
    assert chosen is None and account["effective"] == {"relevance": 2, "submittedDate": 1}
    chosen, account = ts.relative_sort_by([("x", [])])
    assert chosen is None and account["majority"] == "relevance" and account["effective"] == {}
    request = {"id": "req-ft4-0001", "family": "ft4/r11/en/3", "intent": "find_recent_papers", "lang": "en",
               "scenario": "normal", "split": "train",
               "text": "Pull up arXiv papers about KV cache from last week.",
               "slots": {"topic": "KV cache", "topic_id": "kv-cache", "relative": "last_week",
                         "date_from": "2026-09-28", "date_to": "2026-10-04"}}
    rec = ts.relative_date_record(request, TODAY, sort_by="submittedDate")
    assert args_of(turns_of(rec)[0]["tool_calls"][0]) == {"query": "KV cache", "date_from": "2026-09-28",
                                                          "date_to": "2026-10-04", "sort_by": "submittedDate"}
    listing = {**request, "intent": "list_notes", "slots": {"relative": "last_week", "since": "2026-09-28"}}
    rec = ts.relative_date_record(listing, TODAY, sort_by="submittedDate")  # list_notes has no sort_by
    assert args_of(turns_of(rec)[0]["tool_calls"][0]) == {"since": "2026-09-28"}


# R13: the English term next to a pure-Chinese note query; the glossary hook


def test_r13_adds_the_english_term_to_a_pure_chinese_note_query():
    rec = history_record("req-train-0501", one("search_notes", {"query": "量化", "tags": ["idea"]}, [],
                                               compact=True), text_msg("没有找到。"))
    out = ft4([rec], [request_of(rec, topic="quantization", topic_id="quantization", tags=["idea"])])
    first = turns_of(out.records[0])[0]["tool_calls"][0]
    assert first["function"]["arguments"] == '{"query":"量化 quantization","tags":["idea"]}'
    assert out.changes == [relabel("req-train-0501", 0, "R13")]
    assert out.report["rules"]["R13"] == {"pure_chinese": 1, "rewritten": 1, "no_term": 0}


def test_r13_leaves_english_mixed_and_topicless_queries_and_arxiv_searches():
    en = history_record("req-train-0502", one("search_notes", {"query": "量化"}, []), text_msg("Nothing."),
                        lang="en")
    mixed = history_record("req-train-0503", one("search_notes", {"query": "量化 quantization"}, []),
                           text_msg("没有。"))
    topicless = history_record("req-train-0504", one("search_notes", {"query": "导师推荐"}, []),
                               ask_msgs("哪一篇?", ["paper_id"]), scenario="clarify",
                               intent="summarize_paper")
    arxiv = history_record("req-train-0505", one("search_arxiv", {"query": "量化"}, HITS), text_msg(OK))
    topic = {"topic": "quantization", "topic_id": "quantization"}
    reqs = [request_of(en, **topic), request_of(mixed, **topic), request_of(topicless, missing=["paper_id"]),
            request_of(arxiv, **topic)]
    out = ft4([en, mixed, topicless, arxiv], reqs)
    assert out.records == [en, mixed, topicless, arxiv] and out.changes == []
    assert out.report["rules"]["R13"] == {"pure_chinese": 1, "rewritten": 0, "no_term": 1}


def test_the_r8_retry_repeats_the_failed_note_search_with_its_r13_query():
    rec = history_record("req-train-0506", one("search_notes", {"query": "量化"}, RATE_LIMITED, "a"),
                         one("search_notes", {"query": "量化 低比特"}, NOTE_HITS, "b"), text_msg(OK))
    out = ft4([rec], [request_of(rec, topic="quantization", topic_id="quantization")])
    first, retry = turns_of(out.records[0])
    assert args_of(first["tool_calls"][0]) == {"query": "量化 quantization"}
    assert retry["tool_calls"][0]["function"] == first["tool_calls"][0]["function"]
    assert out.changes == [relabel("req-train-0506", 0, "R13"), relabel("req-train-0506", 1, "R8")]


@NEEDS_TERMS
def test_the_glossary_hook_makes_r7_style_records_from_terms_r7_did_not_use():
    corpus = rq.load_corpus(CORPUS)
    r7 = json.loads(TERMS.read_text(encoding="utf-8"))
    first = next(iter(r7))
    glossary = {first: r7[first], "注意力汇聚": "attention sink", "混合专家路由": "expert routing",
                "梯度裁剪": "gradient clipping", "假设": None}  # a blocked term: matched, never noted
    made, report = ts.glossary_requests(glossary, existing=r7, corpus=corpus,
                                        held_out=("mixture-of-experts",),
                                        limit=10, guard=ts.EvalGuard(eval_texts=[], train_texts=[]))
    assert {r["slots"]["term_zh"] for r in made} == {"注意力汇聚", "梯度裁剪"}
    assert all(r["family"].startswith("ft4/terms/zh/") and r["intent"] == ts.TERM_INTENT for r in made)
    assert report["terms"] == 4 and report["blocked"] == 1  # the blocked term is counted apart
    assert report["skipped"] == {"non_ascii": 0, "latin_token_missing": 0, "science_only": 0,
                                 "already_in_r7": 1, "refused": 1}
    assert report["eligible"] == 4 and report["kept"] == 2 and report["requests"] == 4  # two templates each
    rec = ts.glossary_record({"id": "req-ft4-0004", **made[0]}, TODAY)
    assert args_of(turns_of(rec)[0]["tool_calls"][0]) == {"query": made[0]["slots"]["topic"]}
    assert rec["run"] == ts.FT4_RUN


def test_the_glossary_query_lowercases_every_word_but_those_with_two_capitals_or_a_digit():
    """R13's query: a word is a run of ASCII letters and digits, so a hyphenated Title Case word is two."""
    for english, query in (("Gradient Clipping", "gradient clipping"), ("BERT", "BERT"), ("GPT-2", "GPT-2"),
                           ("ResNet Block", "ResNet block"), ("Pre-Training", "pre-training"),
                           ("Density-Based Spatial Clustering Of Applications With Noise",
                            "density-based spatial clustering of applications with noise"),
                           ("C4.5 Algorithm", "C4.5 algorithm"), ("L1 Regularization", "L1 regularization"),
                           ("Q-Learning", "q-learning"), ("t-SNE", "t-SNE"),
                           ("Word2Vec Model", "Word2Vec model")):
        assert ts.glossary_query(english) == query, english


def test_glossary_terms_are_eligible_by_their_english_and_taken_in_the_order_of_their_hash():
    """Eligible: an ASCII English term that holds every Latin token of the Chinese term as a whole word
    (letter case aside), and a term that is not only in AITD's AI for Science section; then R7's dedup and
    check_terms. The terms are taken in the order of the SHA-256 of the Chinese term, not the file's."""
    glossary = {"Adam算法": "Adaptive Moment Estimation Algorithm", "Adam优化器": "Adam Optimizer",
                "C4.5决策树": "C4.5 Decision Tree", "E步": "Expectation Step", "P-R曲线": "PR Curve",
                "交换关联": "Exchange–Correlation", "偶联反应": "Cross-Coupling Reaction",
                "梯度裁剪": "Gradient Clipping", "预训练模型": "Pre-Trained Model", "残差网络": "ResNet",
                "注意力汇聚": "Attention Sink", "分类": None}
    assert [ts.glossary_skip(zh, en, frozenset({"偶联反应"})) for zh, en in glossary.items() if en] == [
        "latin_token_missing", None, None, "latin_token_missing", "latin_token_missing", "non_ascii",
        "science_only", None, None, None, None]
    corpus = rq.load_corpus(CORPUS)
    made, report = ts.glossary_requests(glossary, existing={}, corpus=corpus, held_out=(), limit=20,
                                        guard=ts.EvalGuard(eval_texts=[], train_texts=[]),
                                        science_only=frozenset({"偶联反应"}))
    kept = ["Adam优化器", "C4.5决策树", "梯度裁剪", "预训练模型", "残差网络", "注意力汇聚"]
    assert [r["slots"]["term_zh"] for r in made][:len(kept)] == ts.term_order(kept)
    assert ts.term_order(kept) == sorted(kept, key=lambda zh: hashlib.sha256(zh.encode()).hexdigest())
    assert {r["slots"]["term_zh"]: r["slots"]["topic"] for r in made} == {
        "Adam优化器": "adam optimizer", "C4.5决策树": "C4.5 decision tree", "梯度裁剪": "gradient clipping",
        "预训练模型": "pre-trained model", "残差网络": "ResNet", "注意力汇聚": "attention sink"}
    assert report["terms"] == 11 and report["blocked"] == 1 and report["eligible"] == 6
    assert report["skipped"] == {"non_ascii": 1, "latin_token_missing": 3, "science_only": 1,
                                 "already_in_r7": 0, "refused": 0}
    assert report["kept"] == 6 and report["requests"] == len(made) == 12  # twice each, two templates
    # the file's order makes no difference
    again, _ = ts.glossary_requests(dict(reversed(glossary.items())), existing={}, corpus=corpus, held_out=(),
                                    limit=20, guard=ts.EvalGuard(eval_texts=[], train_texts=[]),
                                    science_only=frozenset({"偶联反应"}))
    assert again == made


def aitd_clone(root: Path, rows: list[tuple[str, str, str]], science: list[str], learning: list[str]) -> Path:
    """A fake AITD clone holding data/All.md with the rows (index, English, Chinese) and the two section files
    with the rows of the indices given, as AITD's markdown tables (scripts/build_glossary.py parses them)."""
    head = "|索引编号|英文术语|中文翻译|常用缩写|来源&扩展|备注|\n|---|---|---|---|---|---|\n"
    for name, chosen in (("data/All.md", [r[0] for r in rows]), ("section/AI for Science.md", science),
                         ("section/Machine Learning.md", learning)):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(head + "".join(f"{i}|{en}|{zh}||||\n" for i, en, zh in rows if i in chosen),
                        encoding="utf-8", newline="\n")
    return root


def provenance_of(path: Path, clone: Path) -> Path:
    """A glossary provenance (scripts/build_glossary.py's) recording the clone's location and files."""
    files = [{"path": p.relative_to(clone).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in sorted(clone.rglob("*.md"))]
    path.write_text(json.dumps({"sources": [{"name": "aitd", "location": clone.as_posix(), "files": files}]}),
                    encoding="utf-8")
    return path


AITD_ROWS = [("AITD-00001", "Cross-Coupling Reaction", "偶联反应"),
             ("AITD-00002", "Gradient Clipping", "梯度裁剪"), ("AITD-00003", "Attention Sink", "注意力汇聚"),
             ("AITD-00004", "Density Functional", "密度泛函"),
             ("AITD-00005", "Density Functional Theory", "密度泛函")]


def test_the_science_only_terms_are_read_from_the_aitd_sections_the_provenance_records(tmp_path):
    """A term is AI for Science only when every AITD row that gives it is in that section and none is in
    another: 偶联反应 (only there); not 梯度裁剪 (in both sections), 注意力汇聚 (in no section) or 密度泛函
    (one row there, one in no section)."""
    clone = aitd_clone(tmp_path / "aitd", AITD_ROWS, science=["AITD-00001", "AITD-00002", "AITD-00004"],
                       learning=["AITD-00002"])
    provenance = provenance_of(tmp_path / "provenance.json", clone)
    only, account = ts.science_only_terms(provenance)
    assert only == frozenset({"偶联反应"})
    assert account["science_only_terms"] == 1 and set(account["files"]) == {
        "data/All.md", "section/AI for Science.md", "section/Machine Learning.md"}
    assert account["files"]["data/All.md"]["rows"] == 5
    (clone / "section" / "AI for Science.md").write_text("changed", encoding="utf-8")
    with pytest.raises(ts.TransformError, match="AI for Science.md"):
        ts.science_only_terms(provenance)


# R14: definite references with a topic; explain_tools on optional parameters


def test_defref_requests_come_only_from_train_templates_with_a_definite_reference_and_a_topic():
    families = ts.defref_families(42)
    assert families == ["read_section/en/0", "read_section/en/3", "read_section/zh/3", "read_section/zh/5"]
    assignment = rq.split_templates(42)
    assert all(assignment[f] == "train" for f in families)
    corpus = rq.load_corpus(CORPUS)
    made = ts.generate_defref_requests(8, seed=42, corpus=corpus, split_seed=42,
                                       guard=ts.EvalGuard(eval_texts=[], train_texts=[]))
    assert len(made) == 8 and {r["family"] for r in made} <= set(families)
    for r in made:
        assert r["slots"]["topic_id"] in {t.id for t in TRAIN_TOPICS}
        rec = ts.defref_record({"id": "req-ft4-0002", **r}, TODAY)
        assert [(c["function"]["name"], args_of(c)) for c in turns_of(rec)[0]["tool_calls"]] == [
            ("search_arxiv", {"query": r["slots"]["topic"]})]
        assert rec["family"] == f"ft4/r14/{r['family']}" and steps_of(rec) == 1


def test_explain_answers_come_from_model_tools_and_ask_nothing_back():
    en = ts.explain_answer("search_notes", "top_k", "en")
    assert all(s in en for s in ("top_k", "search_notes", "optional", "1", "5", "query"))
    zh = ts.explain_answer("list_notes", "since", "zh")
    assert all(s in zh for s in ("list_notes", "since", "YYYY-MM-DD", "没有必填参数"))
    for text in (en, zh):
        assert "?" not in text and "？" not in text
    pairs = ts.optional_params()
    assert ("search_arxiv", "sort_by") in pairs and ("ask_clarification", "missing") in pairs
    assert ("get_paper", "arxiv_id") not in pairs and ("search_arxiv", "query") not in pairs
    made = ts.generate_explain_requests(6, seed=42, guard=ts.EvalGuard(eval_texts=[], train_texts=[]))
    assert len(made) == 6 and all(r["intent"] == "explain_tools" and r["scenario"] == "no_call" for r in made)
    rec = ts.explain_record({"id": "req-ft4-0003", **made[0]}, TODAY)
    assert [m["role"] for m in rec["messages"]] == ["system", "user", "assistant"]
    assert "tool_calls" not in rec["messages"][2] and steps_of(rec) == 1
    slots = made[0]["slots"]
    assert rec["messages"][2]["content"] == ts.explain_answer(slots["tool"], slots["param"], made[0]["lang"])


# the CLI


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def lines_by_id(path: Path) -> dict[str, str]:
    return {json.loads(line)["id"]: line for line in path.read_text(encoding="utf-8").splitlines()}


@pytest.fixture
def ft4_inputs(inputs):
    """inputs plus one record per FT4 situation, each with its request."""
    reqs = read_jsonl(inputs["requests"])
    synth = read_jsonl(inputs["synth"])
    rewritten = {"query": "KV cache inference", "date_from": "2026-09-01", "sort_by": "submittedDate"}
    more = [
        (history_record("req-train-0006",
                        one("search_arxiv", {"query": "KV cache", "date_from": "2026-09-01",
                                             "sort_by": "submittedDate"}, RATE_LIMITED),
                        one("search_arxiv", rewritten, HITS, "b"), text_msg("找到一篇。"),
                        intent="find_recent_papers", family="find_recent_papers/zh/1",
                        user="9 月以来 KV cache 方向有哪些新论文?给我一篇就够了。"),
         {"topic": "KV cache", "topic_id": "kv-cache", "date_from": "2026-09-01", "count": 1}),
        (s4a_record("req-train-0007", [one("search_notes", {"query": "survey"}, [], "c"),
                                       text_msg("没有。")]),
         {"tags": ["survey"]}),
        (history_record("req-train-0008", one("list_notes", {}, NOTES), ask_msgs("哪个方向?", ["topic"]),
                        scenario="clarify", intent="find_recent_papers",
                        user="上次聊的那个主题,9 月以来又出了哪些论文?挑两篇。"), {"missing": ["topic"]}),
        (history_record("req-train-0009", one("search_notes", {"query": "量化"}, NOTE_HITS),
                        text_msg("找到一条。"), user="我以前读过量化相关的东西吗?查一下我的笔记。"),
         {"topic": "quantization", "topic_id": "quantization"}),
        (late_clarification("req-train-0010"), {"missing": ["paper_id"]}),
        (r8_then_s4a("req-train-0011"), {"tags": ["survey"]}),
        (history_record("req-train-1385", one("list_notes", {"tag": "reading-group"}, [], "a"),
                        one("list_notes", {}, NOTES, "b"), text_msg(R10_TEXT), intent="list_notes",
                        user="把 reading-group 标签下的笔记都列出来。"), {}),  # R10: not clean, left as it is
    ]
    for rec, slots in more:
        synth.append(rec)
        reqs.append(request_of(rec, **slots))
    write_jsonl(inputs["requests"], reqs)
    write_jsonl(inputs["synth"], synth)
    return inputs


def ft4_argv(inputs: dict, *extra: str) -> list[str]:
    changes = str(inputs["dir"] / "out" / "changes.jsonl")
    return argv(inputs, "--ft4", "--changes-out", changes, "--r11-new", "4", "--r14-defref", "4",
                "--r14-explain", "4", *extra)


@NEEDS_TERMS
def test_cli_ft4_writes_the_records_the_changes_and_the_manifest(ft4_inputs, capsys):
    assert ts.main(ft4_argv(ft4_inputs)) == 0
    out = ft4_inputs["dir"] / "out"
    records = load_records(out / "synth-m6.jsonl")
    by_id = {r["id"]: r for r in records}
    assert list(by_id) == sorted(by_id)
    for r in records:
        steps_of(r)  # every record cuts into samples the trainer accepts
        assert reasoning.is_scripted(r) == (r["run"] in (ts.RUN, ts.FT4_RUN))
    assert len(turns_of(by_id["req-train-0006"])) == 2  # R8: the retry ends the record
    derived = {"req-train-0011-ft4r9k3", "req-train-0008-ft4r10k1", "req-train-0010-ft4r10k2"}
    assert {r["id"] for r in records if "context_turns" in r} == derived  # 0007's R9 cut builds none
    new = [r for r in records if r["id"].startswith("req-ft4-")]
    assert len(new) == 12 and all(r["run"] == ts.FT4_RUN for r in new)
    new_requests = [r["id"] for r in read_jsonl(out / "req-m6.jsonl") if r["id"].startswith("req-ft4-")]
    assert new_requests == [f"req-ft4-{i:04d}" for i in range(1, 13)] == [r["id"] for r in new]
    changes = read_jsonl(out / "changes.jsonl")
    assert changes == sorted(changes, key=lambda c: (c["id"], c["k"]))
    for c in changes:
        assert set(c) == {"id", "k", "rule", "kind", "target", "from"}
        assert 0 <= c["k"] < len(turns_of(by_id[c["id"]]))
        assert c["kind"] in ("relabel", "new", "copy") and (c["target"] or c["kind"] == "copy")
    assert {(c["id"], c["k"]) for c in changes if not c["target"]} == {
        (r["id"], k) for r in records for k in range(r.get("context_turns", 0))}  # the context turns, exactly
    assert [(c["id"], c["k"]) for c in changes if c["kind"] == "copy" and c["target"]] == [
        ("req-train-0008-ft4r10k1", 1)]  # the teacher's own ask after the listing
    assert {(r["id"], 0) for r in new} <= {(c["id"], c["k"]) for c in changes}
    assert {c["rule"] for c in changes} == {"R8", "R9", "R10", "R11", "R13", "R14"}
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    data = (out / "changes.jsonl").read_bytes()
    section = manifest["ft4"]
    assert section["changes"] == {"path": (out / "changes.jsonl").as_posix(), "rows": len(changes),
                                  "sha256": hashlib.sha256(data).hexdigest()}
    rules = section["rules"]
    assert rules["R8"]["relabeled"] == 2 and rules["R11"]["relabeled"] == rules["R13"]["rewritten"] == 1
    assert rules["R9"]["s4a"]["relabeled"] == rules["R9"]["constructed"]["records"] == 1
    assert rules["R10"]["derived"]["records"] == 2 and rules["R10"]["derived"]["new"] == 1
    assert rules["R10"]["derived"]["copy_target"] == 1
    assert rules["R10"]["relabeled"] == 0 and rules["R10"]["skipped_not_trailing"] == ["req-train-1385:2"]
    assert rules["R10"]["skipped_partial_list"] == []
    assert rules["R11"]["relative_dates"]["sort_by"] == {
        "chosen": "submittedDate", "majority": "submittedDate", "default": "relevance",
        "intents": ["find_recent_papers"], "rule": ts.RELATIVE_SORT_RULE,
        "counts": {"input": {"submittedDate": 2}}, "effective": {"submittedDate": 2}}
    for r in new:
        for c in turns_of(r)[0].get("tool_calls") or []:
            if r["family"].startswith("ft4/r11/") and c["function"]["name"] == "search_arxiv":
                assert args_of(c)["sort_by"] == "submittedDate"
    assert section["eval_overlap"]["ft4_records"]["over"] == 0
    prompts = section["prompts"]
    assert prompts["conflicts_with_ft4"] == 0 and prompts["conflicts_with_copies"] == 0
    # were the context trained: 0008's R11 ask and 0011's R8 retry, and 0010's first search against the first
    # step of R3's record of the same request (req-train-0010-m6r3)
    assert prompts["all_turns"]["conflicts_with_copies"] == 3
    assert section["records"]["new"] == 12 and section["turns"]["new"] == 14  # 12 new records, 2 derived
    assert section["turns"]["copy_target"] == 1
    assert manifest["output"]["records"] == len(records)
    printed = capsys.readouterr().out
    assert "R8 relabeled 2" in printed and "skipped: not trailing 1, partial list 0" in printed
    assert by_id["req-train-1385"] == next(r for r in read_jsonl(ft4_inputs["synth"])
                                           if r["id"] == "req-train-1385")  # the teacher's text kept


REAL_FT4 = [(REPO / "data" / "raw" / "synth" / f"synth-{part}-ft4.jsonl",
             REPO / "data" / "synth" / f"ft4-changes-{part}.jsonl") for part in ("m6", "m7")]


@pytest.mark.skipif(not all(p.is_file() for pair in REAL_FT4 for p in pair),
                    reason="the FT4 outputs are absent")
def test_the_ft4_outputs_mark_as_context_exactly_the_turns_their_changes_rows_leave_out():
    """The repository's FT4 files (WP6): a derived record says itself which turns are context
    (context_turns), and those are exactly the turns its changes rows do not train."""
    for records_path, changes_path in REAL_FT4:
        records = load_records(records_path)
        not_target = {(c["id"], c["k"]) for c in read_jsonl(changes_path) if not c["target"]}
        context = {(r["id"], k) for r in records for k in range(r.get("context_turns", 0))}
        assert context and context == not_target, records_path.name


@NEEDS_TERMS
def test_cli_ft4_counts_r11s_sort_over_more_inputs_and_leaves_out_the_default(ft4_inputs, capsys):
    """--r11-sort-inputs: more agent-history files (the M7 input) whose dated find_recent_papers searches join
    R11's count; here they make the default the majority, so R11's searches leave sort_by out."""
    d = ft4_inputs["dir"]
    more = write_jsonl(d / "synth-m7.jsonl", [dated(f"req-train-17{i:02d}", "find_recent_papers", "relevance",
                                                    None) for i in range(2)])
    assert ts.main(ft4_argv(ft4_inputs, "--r11-new", "8", "--r11-sort-inputs", str(more))) == 0
    manifest = json.loads((d / "out" / "manifest.json").read_text(encoding="utf-8"))
    account = manifest["ft4"]["rules"]["R11"]["relative_dates"]["sort_by"]
    assert account["counts"] == {"input": {"submittedDate": 2},
                                 more.as_posix(): {"absent": 2, "relevance": 2}}
    assert account["effective"] == {"relevance": 4, "submittedDate": 2} and account["chosen"] is None
    assert manifest["inputs"]["r11_sort_inputs"] == [
        {"path": more.as_posix(), "sha256": hashlib.sha256(more.read_bytes()).hexdigest(), "records": 2}]
    searches = [c for r in load_records(d / "out" / "synth-m6.jsonl") if r["family"].startswith("ft4/r11/")
                for c in turns_of(r)[0]["tool_calls"] if c["function"]["name"] == "search_arxiv"]
    assert searches and all("sort_by" not in args_of(c) and "date_from" in args_of(c) for c in searches)
    assert "sort_by left out" in capsys.readouterr().out
    assert ts.main(argv(ft4_inputs, "--r11-sort-inputs", str(more))) == 2  # an FT4 option
    changes = str(d / "only" / "changes.jsonl")
    assert ts.main(records_only_argv(ft4_inputs, "--ft4", "--changes-out", changes, "--r11-sort-inputs",
                                     str(more))) == 2  # it decides new records only


@NEEDS_TERMS
def test_cli_ft4_glossary_terms_are_eligible_in_hash_order_and_their_counts_printed(ft4_inputs, capsys):
    """--glossary: the glossary read with research_agent.glossary.load_glossary, its blocked terms counted
    apart, its AITD sections read from the provenance.json beside it (R13's domain filter)."""
    d = ft4_inputs["dir"]
    folder = d / "glossary"
    folder.mkdir()
    glossary = {"偶联反应": "Cross-Coupling Reaction", "梯度裁剪": "Gradient Clipping",
                "注意力汇聚": "Attention Sink", "Adam算法": "Adaptive Moment Estimation Algorithm",
                "分类": None}
    (folder / "zh-en.json").write_text(json.dumps(glossary, ensure_ascii=False), encoding="utf-8")
    clone = aitd_clone(d / "aitd", AITD_ROWS[:3], science=["AITD-00001"], learning=["AITD-00002"])
    provenance = provenance_of(folder / "provenance.json", clone)
    argv_glossary = ft4_argv(ft4_inputs, "--glossary", str(folder / "zh-en.json"), "--r13-terms", "4")
    assert ts.main(argv_glossary) == 0
    printed = capsys.readouterr().out
    records = [r for r in load_records(d / "out" / "synth-m6.jsonl") if r["family"].startswith("ft4/terms/")]
    queries = [args_of(turns_of(r)[0]["tool_calls"][0])["query"] for r in records]
    order = {"梯度裁剪": "gradient clipping", "注意力汇聚": "attention sink"}
    assert queries == [order[zh] for zh in ts.term_order(order)] * 2  # in the order of the hash, twice
    manifest = json.loads((d / "out" / "manifest.json").read_text(encoding="utf-8"))
    report = manifest["ft4"]["rules"]["R13"]["glossary"]
    assert report["terms"] == 4 and report["blocked"] == 1 and report["eligible"] == 2
    assert report["skipped"] == {"non_ascii": 0, "latin_token_missing": 1, "science_only": 1,
                                 "already_in_r7": 0, "refused": 0}
    assert report["domain"]["science_only_terms"] == 1
    assert manifest["inputs"]["glossary"] == {
        "path": (folder / "zh-en.json").as_posix(), "terms": 4, "blocked": 1,
        "sha256": hashlib.sha256((folder / "zh-en.json").read_bytes()).hexdigest()}
    assert manifest["inputs"]["glossary_provenance"] == {
        "path": provenance.as_posix(), "sha256": hashlib.sha256(provenance.read_bytes()).hexdigest()}
    assert "glossary 4 requests from 2 of 4 noted terms (1 blocked apart" in printed
    assert "AI for Science only 1" in printed
    provenance.unlink()  # the domain filter cannot run: a missing input
    assert ts.main([*argv_glossary, "--force"]) == 2


@NEEDS_TERMS
def test_cli_ft4_reruns_byte_identical_and_keeps_its_outputs_without_force(ft4_inputs):
    out = ft4_inputs["dir"] / "out"
    assert ts.main(ft4_argv(ft4_inputs)) == 0
    first = {p.name: p.read_bytes() for p in out.iterdir()}
    assert ts.main(ft4_argv(ft4_inputs)) == 2
    assert ts.main(ft4_argv(ft4_inputs, "--force")) == 0
    assert {p.name: p.read_bytes() for p in out.iterdir()} == first


@NEEDS_TERMS
def test_ft4_leaves_the_records_it_does_not_change_and_m6s_new_requests_as_they_were(ft4_inputs):
    out = ft4_inputs["dir"] / "out"
    assert ts.main(argv(ft4_inputs)) == 0
    plain = lines_by_id(out / "synth-m6.jsonl")
    plain_requests = (out / "req-m6.jsonl").read_text(encoding="utf-8").splitlines()
    assert ts.main(ft4_argv(ft4_inputs, "--force")) == 0
    lines = lines_by_id(out / "synth-m6.jsonl")
    changed = {c["id"] for c in read_jsonl(out / "changes.jsonl")}
    assert all(lines[rid] == line for rid, line in plain.items() if rid not in changed)
    assert changed - set(plain) == {rid for rid in lines if rid not in plain}  # FT4 adds only what it lists
    requests = (out / "req-m6.jsonl").read_text(encoding="utf-8").splitlines()
    assert requests[:len(plain_requests)] == plain_requests


def test_the_ft4_options_need_ft4_and_ft4_needs_a_changes_file(ft4_inputs):
    assert ts.main(argv(ft4_inputs, "--r11-new", "4")) == 2
    assert ts.main(argv(ft4_inputs, "--changes-out", str(ft4_inputs["dir"] / "c.jsonl"))) == 2
    assert ts.main(argv(ft4_inputs, "--ft4")) == 2
    assert not (ft4_inputs["dir"] / "out").exists()


@NEEDS_TERMS
def test_records_only_ft4_gives_the_full_runs_ft4_records_of_the_input(ft4_inputs):
    d = ft4_inputs["dir"]
    assert ts.main(ft4_argv(ft4_inputs)) == 0
    changes = str(d / "only" / "changes.jsonl")
    assert ts.main(records_only_argv(ft4_inputs, "--ft4", "--changes-out", changes)) == 0
    assert ts.main(records_only_argv(ft4_inputs, "--ft4", "--changes-out", changes, "--force",
                                     "--r11-new", "4")) == 2  # no new record without the full run
    full = (d / "out" / "synth-m6.jsonl").read_text(encoding="utf-8").splitlines()
    only = (d / "only" / "synth-m7.jsonl").read_text(encoding="utf-8").splitlines()
    assert only == [line for line in full if json.loads(line)["run"] != ts.RUN
                    and not json.loads(line)["id"].startswith("req-ft4-")]
    full_changes = read_jsonl(d / "out" / "changes.jsonl")
    only_changes = read_jsonl(d / "only" / "changes.jsonl")
    assert only_changes == [c for c in full_changes if not c["id"].startswith("req-ft4-")]
    section = only_manifest(ft4_inputs)["ft4"]
    assert section["not_run"]["rules"] == ["R11 relative dates", "R14 definite references",
                                           "R14 explain_tools", "R13 glossary terms"]


@NEEDS_TERMS
def test_ft4_drops_a_derived_record_too_close_to_an_extra_evaluation_text(ft4_inputs):
    d = ft4_inputs["dir"]
    extra = write_jsonl(d / "dev-v2.jsonl", [{"id": "req-dev-0401-s1", "split": "dev", "messages": [
        {"role": "user", "content": "What's the core idea of the paper our lab recommended?"}]}])
    assert ts.main(ft4_argv(ft4_inputs, "--ft4-eval-items", str(extra))) == 0
    ids = {r["id"] for r in load_records(d / "out" / "synth-m6.jsonl")}
    assert "req-train-0010-ft4r10k2" not in ids and "req-train-0008-ft4r10k1" in ids
    manifest = json.loads((d / "out" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ft4"]["eval_overlap"]["dropped"] == ["req-train-0010-ft4r10k2"]
    assert manifest["inputs"]["ft4_eval_items"] == [
        {"path": extra.as_posix(), "sha256": hashlib.sha256(extra.read_bytes()).hexdigest(), "texts": 1}]
    test_item = {"id": "hw-test-001", "split": "test", "messages": [{"role": "user", "content": "x"}]}
    write_jsonl(extra, [test_item])
    assert ts.main(ft4_argv(ft4_inputs, "--ft4-eval-items", str(extra), "--force")) == 1

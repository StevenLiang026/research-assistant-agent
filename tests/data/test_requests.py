"""research_agent.data.requests and scripts/make_requests.py offline: the template catalogue, the seeded
4/1/1 template split and 70/15/15 topic split, the record contract (docs/modules/接口.md, 请求记录), the
planned mix per split, dedup, the leakage audit and the byte-identical writer, and the CLI's exit codes; and
the incremental builds (docs/m7-计划.md §5, WP-I 1): the draw seed, the splits built, the id start, the
preloaded request files every dedup rule runs against, a mix of one's own, what such a build's manifest
records, and the audits over preloaded and new requests together. tests/data/test_make_requests.py covers the
new command-line flags."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import re
import subprocess
from collections import Counter
from pathlib import Path

import pytest

from research_agent.data import requests as req
from research_agent.data.synth import LANGS, PLANNED_MIX, SCENARIOS, SPLITS
from research_agent.normalize import normalize_arxiv_id, normalize_text
from research_agent.tools import schemas

REPO = Path(__file__).resolve().parents[2]
CORPUS = REPO / "data" / "corpus" / "manifest.json"
ID_CHARSET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RECORD_KEYS = {"id", "intent", "scenario", "lang", "split", "family", "text", "slots"}


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_requests = load_script("make_requests")


@pytest.fixture(scope="module")
def corpus() -> req.Corpus:
    return req.load_corpus(CORPUS)


@pytest.fixture(scope="module")
def build(corpus) -> req.Build:
    return req.generate(42, None, corpus)


def test_there_are_twelve_documented_intents_each_with_six_templates_per_language():
    assert len(req.INTENTS) == 12 and set(req.TEMPLATES) == set(req.INTENTS)
    for intent, kinds in req.TEMPLATES.items():
        assert req.INTENTS[intent].strip()
        for lang in LANGS:
            templates = [t for kind in kinds.values() for t in kind.get(lang, ())]
            assert len(templates) >= 6, (intent, lang)
            assert len(set(templates)) == len(templates), (intent, lang)
        for kind, pools in kinds.items():
            assert kind in req.KINDS
            for lang in LANGS:
                assert len(pools[lang]) >= 6, (intent, kind, lang)


def test_the_no_call_kind_belongs_to_exactly_the_two_no_call_intents():
    no_call = {intent for intent, kinds in req.TEMPLATES.items() if "no_call" in kinds}
    assert no_call == {"explain_tools", "chit_chat"}
    assert all(set(req.TEMPLATES[i]) == {"no_call"} for i in no_call)
    assert set(req.CLARIFY_MISSING) == {i for i, k in req.TEMPLATES.items() if "clarify" in k}
    assert all(set(m) <= set(schemas.MISSING) for m in req.CLARIFY_MISSING.values())


def test_every_placeholder_has_a_filler_and_every_template_fills_completely(corpus):
    topics = req.split_topics(corpus, 42)["train"]
    for intent, kind, lang, _, template in req.template_families():
        assert set(req.placeholders(template)) <= set(req.FILLERS), template
        for draw in range(5):
            text, slots = req.fill(template, lang, topics, corpus.window, random.Random(draw), intent=intent,
                                   kind=kind)
            assert "{" not in text and "}" not in text, text
            assert text == text.strip() and "  " not in text
            if "{topic}" in template:
                assert slots["topic"] and slots["topic_id"]
            if kind == "clarify":
                assert slots["missing"] == req.CLARIFY_MISSING[intent]


def test_the_template_split_is_four_one_one_per_pool_and_seeded():
    a, b, c = req.split_templates(42), req.split_templates(42), req.split_templates(7)
    assert a == b and a != c
    assert set(a) == {req.family_name(i, lang, n) for i, _, lang, n, _ in req.template_families()}
    pools: dict[tuple, Counter] = {}
    for intent, kind, lang, index, _ in req.template_families():
        pools.setdefault((intent, kind, lang), Counter())[a[req.family_name(intent, lang, index)]] += 1
    for key, counts in pools.items():
        size = sum(counts.values())
        assert counts["test"] == 1 and counts["dev"] == 1 and counts["train"] == size - 2 >= 4, key


def test_topics_and_paper_ids_split_seventy_fifteen_fifteen_disjointly_and_seeded(corpus):
    a, b, c = req.split_topics(corpus, 42), req.split_topics(corpus, 42), req.split_topics(corpus, 7)
    assert a == b and a != c
    assert [len(a[s]) for s in SPLITS] == [14, 3, 3]
    ids = {s: {p for t in a[s] for p in t.papers} for s in SPLITS}
    assert [len(ids[s]) for s in SPLITS] == [210, 45, 45]
    topic_ids = {s: {t.id for t in a[s]} for s in SPLITS}
    for i, x in enumerate(SPLITS):
        for y in SPLITS[i + 1:]:
            assert not ids[x] & ids[y] and not topic_ids[x] & topic_ids[y]


def test_records_match_the_request_contract(build, corpus):
    phrases = {t.id: set(t.phrases) for t in corpus.topics}
    first_phrase = {t.id: t.phrases[0] for t in corpus.topics}
    start, end = corpus.window
    assert len({r["id"] for r in build.records}) == len(build.records)
    for r in build.records:
        assert set(r) == RECORD_KEYS
        assert ID_CHARSET.match(r["id"]) and re.fullmatch(rf"req-{r['split']}-\d{{4}}", r["id"])
        assert r["intent"] in req.INTENTS and r["scenario"] in SCENARIOS
        assert r["lang"] in LANGS and r["split"] in SPLITS
        intent, lang, index = r["family"].split("/")
        assert intent == r["intent"] and lang == r["lang"] and index.isdigit()
        slots = r["slots"]
        for key in ("date_from", "date_to", "since"):
            if key in slots:
                assert ISO.match(slots[key]) and start <= slots[key] <= end, (key, r)
        if "date_to" in slots:
            assert slots["date_from"] <= slots["date_to"]
        for aid in ([slots["arxiv_id"]] if "arxiv_id" in slots else []) + slots.get("arxiv_ids", []):
            assert normalize_arxiv_id(aid) == aid and aid in r["text"]
        if "arxiv_ids" in slots:
            assert len(set(slots["arxiv_ids"])) == len(slots["arxiv_ids"]) >= 2
        if "categories" in slots:
            assert slots["categories"] == sorted(slots["categories"])
            assert set(slots["categories"]) <= set(schemas.CATEGORIES)
        if "tags" in slots:
            assert isinstance(slots["tags"], list) and set(slots["tags"]) <= set(req.TAGS)
        if "topic" in slots:
            assert slots["topic"] in phrases[slots["topic_id"]]
            if r["lang"] == "en":
                assert slots["topic"] in r["text"]
            elif slots["topic"] not in r["text"]:  # the Chinese name alone records the first phrase
                assert slots["topic"] == first_phrase[slots["topic_id"]], r
        assert slots.get("count", 1) in req.COUNTS
        assert slots.get("section", "all") in schemas.SECTIONS
        assert slots.get("mode", "draft") in req.MODES


def test_scenarios_use_the_templates_of_their_kind(build):
    kind_of = {req.family_name(i, lang, n): k for i, k, lang, n, _ in req.template_families()}
    for r in build.records:
        assert kind_of[r["family"]] == req.SCENARIO_KIND[r["scenario"]], r


def test_the_mix_and_the_languages_hold_within_each_split(build):
    counts = req.counts_by_cell(build.records)
    for split, size in req.DEFAULT_SIZES.items():
        assert sum(sum(c.values()) for c in counts[split].values()) == size
        for scenario in SCENARIOS:
            assert abs(sum(counts[split][scenario].values()) - PLANNED_MIX[scenario] * size) <= 2
        by_lang = {lang: sum(counts[split][s][lang] for s in SCENARIOS) for lang in LANGS}
        assert abs(by_lang["zh"] - by_lang["en"]) <= 1


def test_the_build_leaks_nothing_and_survives_its_own_dedup(build):
    assert req.audit_summary(req.audit(build.records))["ok"]
    kept, dropped = req.dedup(build.records)
    assert kept == build.records and not any(dropped.values())


@pytest.mark.needs_data("data/synth/dryrun-requests.jsonl")
def test_no_record_repeats_a_dry_run_sentence(build):
    lines = (REPO / "data/synth/dryrun-requests.jsonl").read_text("utf-8").split("\n")
    dry = [req.shingles(req.mask_text(json.loads(line)["text"])) for line in lines if line.strip()]
    repeats = [(r["text"], round(j, 3)) for r in build.records
               for j in [max(req.jaccard(req.shingles(req.mask_text(r["text"])), d) for d in dry)]
               if j >= req.TRAIN_VS_EVAL]
    assert not repeats


# ---- masking, the template audit and surface quality ---------------------------------------------------

ID_RE = re.compile(r"(?i)(?:arxiv:)?\d{4}\.\d{4,5}")
CJK_CHAR = "一-鿿"
# the repository writes Chinese with half-width , ? ! : ; ( ) and full-width 。 、, attached without spaces
ZH_PUNCT = ",?!:;()　-〿＀-￯"
SPACING_FAULTS = {
    "no space between a Chinese character and a Latin letter or digit":
        re.compile(f"[{CJK_CHAR}][A-Za-z0-9]|[A-Za-z0-9][{CJK_CHAR}]"),
    "a space beside Chinese punctuation": re.compile(f" [{ZH_PUNCT}]|[{ZH_PUNCT}] "),
    "a space between two Chinese characters": re.compile(f"[{CJK_CHAR}] [{CJK_CHAR}]"),
    "a double space": re.compile("  "),
}


def test_mask_text_replaces_ids_dates_and_numbers_with_fixed_tokens():
    assert req.mask_text(" Compare arXiv:2609.11716v2 and 2607.00012 since 2026-09-03,  top 3 ") == \
        "Compare <id> and <id> since <date>, top <n>"
    assert req.mask_text("９ 月 3 日以来") == "<n> 月 <n> 日以来"


def test_dedup_compares_texts_after_masking_ids_dates_and_numbers():
    a = "What paper is 2609.11716? Who wrote it, and is it from 2026-09-03?"
    b = "What paper is arXiv:2607.00012? Who wrote it, and is it from 2026-08-17?"
    assert req.jaccard(req.shingles(a), req.shingles(b)) < req.NEAR_DUP  # raw texts look distinct
    kept, dropped = req.dedup([record("a", "train", a), record("b", "train", b)])
    assert [r["id"] for r in kept] == ["a"] and dropped["near"] == 1
    c = "Give me 3 papers on reward models since 2026-08-01, with ids like 2608.01234 and 2608.04321."
    d = ("Give me 5 papers on reward models since 2026-07-15, ids like 2610.00007 and 2609.12345, "
         "newest first.")
    assert req.jaccard(req.shingles(c), req.shingles(d)) < req.TRAIN_VS_EVAL
    masked = req.jaccard(req.shingles(req.mask_text(c)), req.shingles(req.mask_text(d)))
    assert req.TRAIN_VS_EVAL <= masked < req.NEAR_DUP
    kept, dropped = req.dedup([record("d", "train", d), record("c", "dev", c)])
    assert [r["id"] for r in kept] == ["c"] and dropped["train_vs_eval"] == 1


def wording(template: str) -> frozenset[str]:
    """Independent of the module: placeholders removed, whitespace collapsed, character 5-grams."""
    return req.shingles(" ".join(re.sub(r"\{[a-z_]+\}", " ", template).split()))


def test_no_two_templates_of_different_splits_share_their_wording():
    assignment = req.split_templates(42)
    rows = [(req.family_name(i, lang, n), lang, wording(t)) for i, _, lang, n, t in req.template_families()]
    worst = max((req.jaccard(a[2], b[2]), a[0], b[0]) for x, a in enumerate(rows) for b in rows[x + 1:]
                if a[1] == b[1] and assignment[a[0]] != assignment[b[0]])
    assert worst[0] < req.TEMPLATE_LIMIT, worst
    result = req.template_audit(assignment)
    assert result["ok"] and not result["over"]
    assert result["max_jaccard"] == pytest.approx(worst[0])


def test_the_writer_refuses_templates_of_different_splits_that_share_wording(build, tmp_path, monkeypatch):
    catalogue = copy.deepcopy(req.TEMPLATES)
    pool = catalogue["compare_papers"]["clarify"]["zh"]
    catalogue["compare_papers"]["clarify"]["zh"] = ("那两篇哪篇更适合做我的{use}?", *pool[1:])
    pool = catalogue["compare_papers"]["normal"]["zh"]
    catalogue["compare_papers"]["normal"]["zh"] = ("{id_pair} 这两篇哪篇更适合做我的{use}?", *pool[1:])
    monkeypatch.setattr(req, "TEMPLATES", catalogue)
    clarify = req.family_name("compare_papers", "zh", len(pool))
    normal = req.family_name("compare_papers", "zh", 0)
    assignment = {**req.split_templates(42), normal: "test", clarify: "train"}
    monkeypatch.setattr(req, "split_templates", lambda seed: assignment)
    result = req.template_audit(assignment)
    assert not result["ok"] and result["max_jaccard"] >= req.TEMPLATE_LIMIT
    assert "FAIL" in req.template_audit_line(result)
    out, manifest = tmp_path / "r.jsonl", tmp_path / "m.json"
    with pytest.raises(ValueError, match="template audit"):
        req.write_requests(build.records, out, manifest, build.report)
    assert not out.exists() and not manifest.exists()


# templates that deliberately name the topic by its English phrase only (none so far)
ENGLISH_TOPIC_TEMPLATES: frozenset[str] = frozenset()


def test_chinese_requests_name_the_topic_by_its_chinese_name(build, corpus):
    names = {t.id: t.name for t in corpus.topics}
    zh = [r for r in build.records if r["lang"] == "zh" and "topic_id" in r["slots"]]
    lacking = [r["text"] for r in zh
               if names[r["slots"]["topic_id"]] not in r["text"]
               and r["family"] not in ENGLISH_TOPIC_TEMPLATES]
    assert zh and not lacking, (len(lacking), len(zh), lacking[:5])
    with_topic = [t for _, _, lang, _, t in req.template_families()
                  if lang == "zh" and ("{topic}" in t or "{topic_paren}" in t)]
    with_paren = [t for t in with_topic if "{topic_paren}" in t]
    assert with_paren and 3 * len(with_paren) <= len(with_topic)
    assert not any("{topic_paren}" in t for _, _, lang, _, t in req.template_families() if lang == "en")


def test_a_count_of_one_reads_naturally_in_both_languages(build):
    ones = [r for r in build.records if r["slots"].get("count") == 1]
    assert {r["lang"] for r in ones} == set(LANGS)
    for r in ones:
        text = r["text"]
        if r["lang"] == "en":
            assert re.search(r"\bone (paper|note)\b", text) and not re.search(r"\b1 (paper|note)", text), text
            assert not re.search(r"\b(at most|max|latest one|recent one|newest one)\b", text), text
        else:
            assert ("一篇" in text or "一条笔记" in text) and not re.search(r"1 (篇|条)", text), text
            assert "以内" not in text, text


def test_chinese_spacing_follows_the_repository_style(build):
    zh = [r["text"] for r in build.records if r["lang"] == "zh"]
    faults = {name: [t for t in zh if rx.search(t)][:3] for name, rx in SPACING_FAULTS.items()}
    assert not any(faults.values()), faults
    raw = "关于 tool use (工具调用) 的 3篇 , 9月以来 , write_note 的 arXiv:2609.00001 ?"
    assert req.tidy(raw, "zh") == "关于 tool use(工具调用)的 3 篇,9 月以来,write_note 的 arXiv:2609.00001?"
    assert req.tidy("Compare  A and B: how do they differ ?", "en") == "Compare A and B: how do they differ ?"


ZH_REFERENT = re.compile(r"那|这两篇|我们组|我关注|之前|上次|刚才|刚读完|昨天|上周|导师|组里|老板|"
                         r"我说的|提到|聊")
EN_REFERENT = re.compile(r"\b(that|those|the (paper|one|two|topic|subject|area|field|direction|project|"
                         r"problem)|I mentioned|I told you|we (discussed|talked)|my (area|advisor|boss)|"
                         r"our (group|lab)|brought up|yesterday|last week|the lab)\b", re.IGNORECASE)


def test_clarify_requests_refer_to_a_topic_or_paper_they_do_not_name(build, corpus):
    terms = [t.name for t in corpus.topics] + [p for t in corpus.topics for p in t.phrases]
    for intent, kind, lang, _, template in req.template_families():
        if kind != "clarify":
            continue
        named = {"topic", "topic_paren", "id", "id_pair", "ids"} & set(req.placeholders(template))
        assert not named, template
        assert (ZH_REFERENT if lang == "zh" else EN_REFERENT).search(template), template
    for r in build.records:
        if r["scenario"] == "clarify":
            assert not ID_RE.search(r["text"]) and not any(t in r["text"] for t in terms), r["text"]
            assert not {"topic", "topic_id", "arxiv_id", "arxiv_ids"} & set(r["slots"]), r


def record(rid: str, split: str, text: str) -> dict:
    return {"id": rid, "intent": "find_recent_papers", "scenario": "normal", "lang": "en", "split": split,
            "family": f"find_recent_papers/en/{rid}", "text": text, "slots": {}}


BASE = "Find three papers on speculative decoding published since September 2026, most relevant first."


def test_dedup_drops_an_exact_and_a_near_duplicate_keeping_the_first():
    near = BASE[:-1] + "!!"
    assert req.jaccard(req.shingles(BASE), req.shingles(near)) >= req.NEAR_DUP
    records = [record("a", "train", BASE), record("b", "train", "  " + BASE.replace(" ", "  ") + " "),
               record("c", "train", near), record("d", "train", "What is 2601.00001 about?")]
    kept, dropped = req.dedup(records)
    assert [r["id"] for r in kept] == ["a", "d"]
    assert dropped == {"exact": 1, "near": 1, "train_vs_eval": 0}


def test_dedup_drops_a_train_request_too_close_to_a_test_request_even_when_offered_first():
    close = ("Find three papers on speculative decoding published since September 2026, "
             "newest ones first, thanks.")
    similarity = req.jaccard(req.shingles(BASE), req.shingles(close))
    assert req.TRAIN_VS_EVAL <= similarity < req.NEAR_DUP
    kept, dropped = req.dedup([record("t", "train", close), record("x", "test", BASE)])
    assert [r["id"] for r in kept] == ["x"] and dropped["train_vs_eval"] == 1
    kept, _ = req.dedup([record("t", "train", close), record("u", "train", BASE)])
    assert [r["id"] for r in kept] == ["t", "u"]  # within train only the near threshold applies


def test_jaccard_is_exact_over_shingle_sets():
    index = req._ShingleIndex()
    sets = [req.shingles(t) for t in (BASE, "Summarise 2609.11716 in three sentences.", "hi")]
    for s in sets:
        index.add(s)
    probe = req.shingles(BASE.replace("three", "four"))
    assert index.max_jaccard(probe) == pytest.approx(max(req.jaccard(probe, s) for s in sets))
    assert req.shingles("hi") == frozenset({"hi"})


@pytest.mark.parametrize("field", ["family", "topic", "paper"])
def test_the_audit_and_the_writer_refuse_an_injected_overlap(build, tmp_path, field):
    records = copy.deepcopy(build.records)
    kind = {"family": "families", "topic": "topics", "paper": "paper_ids"}[field]
    slot = "topic_id" if field == "topic" else "arxiv_id"
    test_rec = next(r for r in records if r["split"] == "test" and slot in r["slots"])
    train_rec = next(r for r in records if r["split"] == "train" and slot in r["slots"])
    if field == "family":
        train_rec["family"] = test_rec["family"]
    elif field == "topic":
        train_rec["slots"]["topic_id"] = test_rec["slots"]["topic_id"]
    else:
        train_rec["slots"]["arxiv_id"] = "arXiv:" + test_rec["slots"]["arxiv_id"]
    overlaps = req.audit(records)
    assert overlaps[kind]["train/test"] and not req.audit_summary(overlaps)["ok"]
    assert "FAIL" in req.audit_line(overlaps)
    out, manifest = tmp_path / "r.jsonl", tmp_path / "m.json"
    with pytest.raises(ValueError, match=kind):
        req.write_requests(records, out, manifest, build.report)
    assert not out.exists() and not manifest.exists()


def test_the_writer_is_byte_identical_across_two_runs_and_records_the_manifest(corpus, tmp_path):
    files = []
    for run in ("one", "two"):
        build = req.generate(42, None, corpus)
        out, manifest = tmp_path / run / "requests.jsonl", tmp_path / run / "manifest.json"
        req.write_requests(build.records, out, manifest, build.report)
        files.append((out.read_bytes(), manifest.read_bytes()))
    assert files[0] == files[1]
    data = json.loads(files[0][1])
    assert data["seed"] == 42 and data["audit"]["ok"] and data["requests"] == 2400
    assert set(data["dedup"]) == {"offered", *req.DROP_REASONS}
    assert data["counts"]["test"]["parallel"] == {"zh": 10, "en": 10}
    lines = files[0][0].decode("utf-8").split("\n")
    assert lines[-1] == "" and len(lines) == 2401 and "\r" not in files[0][0].decode("utf-8")


def test_another_seed_gives_other_requests(corpus):
    a = req.generate(42, {"train": 40, "dev": 20, "test": 20}, corpus).records
    b = req.generate(43, {"train": 40, "dev": 20, "test": 20}, corpus).records
    texts_a, texts_b = Counter(r["text"] for r in a), Counter(r["text"] for r in b)
    assert texts_a != texts_b and sum((texts_a & texts_b).values()) < len(a) // 2


def test_the_manifest_pins_the_corpus_manifest_by_sha256(corpus, tmp_path):
    build = req.generate(42, {"train": 40, "dev": 20, "test": 20}, corpus)
    digest = hashlib.sha256(CORPUS.read_bytes()).hexdigest()
    assert corpus.sha256 == digest and build.report["corpus_manifest_sha256"] == digest
    manifest = req.write_requests(build.records, tmp_path / "r.jsonl", tmp_path / "m.json", build.report)
    assert manifest["corpus_manifest_sha256"] == digest
    assert json.loads((tmp_path / "m.json").read_text("utf-8"))["corpus_manifest_sha256"] == digest
    assert manifest["template_audit"]["ok"] and manifest["template_audit"]["limit"] == req.TEMPLATE_LIMIT


def test_two_cli_runs_in_separate_processes_write_identical_bytes(tmp_path):
    outputs = []
    for run in ("one", "two"):
        out, manifest = tmp_path / run / "requests.jsonl", tmp_path / run / "manifest.json"
        argv = ["uv", "run", "--no-sync", "python", "scripts/make_requests.py", "--out", str(out),
                "--manifest", str(manifest), "--train", "200", "--dev", "60", "--test", "60"]
        done = subprocess.run(argv, cwd=REPO, capture_output=True, text=True, encoding="utf-8", timeout=300,
                              check=False)
        assert done.returncode == 0, done.stderr
        outputs.append((out.read_bytes(), manifest.read_bytes()))
    assert outputs[0] == outputs[1]


def test_the_cli_writes_both_files_and_exits_zero(tmp_path, capsys):
    out, manifest = tmp_path / "r.jsonl", tmp_path / "m.json"
    code = make_requests.main(["--out", str(out), "--manifest", str(manifest), "--train", "60", "--dev", "20",
                               "--test", "20", "--corpus", str(CORPUS)])
    printed = capsys.readouterr().out
    assert code == 0 and out.exists() and manifest.exists()
    assert "dedup: offered" in printed and "audit: shared families 0, topics 0, paper ids 0" in printed
    assert "test: 20 requests" in printed
    assert "template audit: max cross-split wording Jaccard" in printed and "(limit 0.6): ok" in printed


@pytest.mark.parametrize("argv", [["--train", "-1"], ["--train", "0", "--dev", "0", "--test", "0"],
                                  ["--seed", "x"], ["--bogus"], ["--corpus", "no/such/manifest.json"]])
def test_the_cli_exits_two_on_usage_errors(tmp_path, argv):
    base = ["--out", str(tmp_path / "r.jsonl"), "--manifest", str(tmp_path / "m.json")]
    assert make_requests.main(base + argv) == 2
    assert not (tmp_path / "r.jsonl").exists()


def test_the_cli_exits_one_when_the_audit_fails(tmp_path, monkeypatch, corpus):
    real = req.generate

    def leaky(seed, sizes, corpus_arg, **options):
        build = real(seed, sizes, corpus_arg, **options)
        test_family = next(r["family"] for r in build.records if r["split"] == "test")
        next(r for r in build.records if r["split"] == "train")["family"] = test_family
        return build

    monkeypatch.setattr(make_requests.req, "generate", leaky)
    out = tmp_path / "r.jsonl"
    code = make_requests.main(["--out", str(out), "--manifest", str(tmp_path / "m.json"), "--train", "40",
                               "--dev", "20", "--test", "20", "--corpus", str(CORPUS)])
    assert code == 1 and not out.exists()


def test_the_cli_exits_one_when_a_cell_cannot_be_filled(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(make_requests.req, "MAX_ATTEMPTS_PER_REQUEST", 0)
    out, manifest = tmp_path / "r.jsonl", tmp_path / "m.json"
    code = make_requests.main(["--out", str(out), "--manifest", str(manifest), "--train", "40", "--dev", "20",
                               "--test", "20", "--corpus", str(CORPUS)])
    assert code == 1 and not out.exists() and not manifest.exists()
    assert "could not fill every cell" in capsys.readouterr().err


# ---- incremental builds: the draw seed, the splits built, the id start, preloaded files, the mix ---------

SMALL = {"train": 40, "dev": 20, "test": 20}
V2_MIX = {"normal": 0.40, "no_call": 0.15, "clarify": 0.15, "recover": 0.20, "parallel": 0.10}
M7_MIX = {"normal": 0.50, "no_call": 0.10, "clarify": 0.20, "recover": 0.12, "parallel": 0.08}
REGISTERED = (REPO / "data" / "synth" / "requests.jsonl", REPO / "data" / "synth" / "requests-m6.jsonl")
NO_DROPS = dict.fromkeys(req.DROP_REASONS, 0)
CLOSE = ("Find three papers on speculative decoding published since September 2026, "
         "newest ones first, thanks.")  # within the train-vs-eval band of BASE, not a near duplicate
SUFFIX = ("and", "please", "also", "add", "one", "short", "line", "on", "why", "each", "of", "them",
          "matters", "for", "my", "work", "this", "week")  # words in_band_variant appends one at a time


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text(req.requests_text(records), encoding="utf-8", newline="\n")
    return path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def masked(text: str) -> frozenset[str]:
    return req.shingles(req.mask_text(text))


def in_band_variant(text: str) -> str:
    """text with words appended until its masked Jaccard to text falls below NEAR_DUP while it stays at or
    above TRAIN_VS_EVAL: close enough for the train-vs-eval rule, not a near duplicate."""
    base, variant = masked(text), text
    for word in SUFFIX:
        variant = f"{variant} {word}"
        similarity = req.jaccard(base, masked(variant))
        if similarity < req.NEAR_DUP:
            assert similarity >= req.TRAIN_VS_EVAL, (text, variant, similarity)
            return variant
    raise AssertionError(f"no variant between the two thresholds for {text!r}")


def family_of(split: str, lang: str = "en") -> str:
    """A find_recent_papers family that the seed-42 template split puts in this split."""
    return next(f for f, s in sorted(req.split_templates(42).items())
                if s == split and f.startswith(f"find_recent_papers/{lang}/"))


def made_request(rid: str, split: str, text: str) -> dict:
    return {"id": rid, "intent": "find_recent_papers", "scenario": "normal", "lang": "en", "split": split,
            "family": family_of(split), "text": text, "slots": {}}


def without_ids(records: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "id"} for r in records]


def by_scenario(plan: dict[str, dict[str, int]]) -> dict[str, int]:
    return {s: sum(plan[s].values()) for s in SCENARIOS}


def test_the_draw_seed_defaults_to_the_seed_and_moves_only_the_draws_within_a_split(corpus):
    default = req.generate(42, SMALL, corpus)
    named = req.generate(42, SMALL, corpus, draw_seed=42)
    assert named.records == default.records and named.report == default.report
    assert "draw_seed" not in default.report
    other = req.generate(42, SMALL, corpus, draw_seed=7)
    assert other.report["draw_seed"] == 7
    texts = Counter(r["text"] for r in default.records)
    assert sum((texts & Counter(r["text"] for r in other.records)).values()) < len(default.records) // 2
    for key in ("seed", "sizes", "topics", "paper_ids", "families"):
        assert other.report[key] == default.report[key], key
    assignment = req.split_templates(42)
    topic_split = {t.id: split for split, topics in req.split_topics(corpus, 42).items() for t in topics}
    for r in other.records:
        assert assignment[r["family"]] == r["split"], r
        if "topic_id" in r["slots"]:
            assert topic_split[r["slots"]["topic_id"]] == r["split"], r
    assert req.counts_by_cell(other.records) == req.counts_by_cell(default.records)
    # every draw seed fills a split's cells in one order and to the same counts, so the sequence of cells
    # along a split changes only when the records are shuffled with the draw seed, not with the seed
    for split in SPLITS:
        cells = [[(r["scenario"], r["lang"]) for r in records if r["split"] == split]
                 for records in (default.records, other.records)]
        assert cells[0] != cells[1], split


def test_only_builds_the_named_splits_and_makes_them_as_a_full_build_does(corpus):
    full = req.generate(42, SMALL, corpus)
    both = req.generate(42, SMALL, corpus, only=["test", "dev"])
    assert both.records == [r for r in full.records if r["split"] != "train"]  # train is filled last
    assert both.report["sizes"] == {"train": 0, "dev": 20, "test": 20}
    assert both.report["only"] == ["dev", "test"]
    assert req.generate(42, SMALL, corpus, only=["train", "dev", "test"]).report == full.report
    for bad in (["validation"], [], ["test", "test"]):
        with pytest.raises(ValueError, match="split"):
            req.generate(42, SMALL, corpus, only=bad)


def test_id_start_continues_the_numbering_with_one_width_per_split(corpus):
    base = req.generate(42, {"test": 20}, corpus)
    moved = req.generate(42, {"test": 20}, corpus, id_start={"test": 401})
    assert [r["id"] for r in base.records] == [f"req-test-{i:04d}" for i in range(1, 21)]
    assert [r["id"] for r in moved.records] == [f"req-test-{i:04d}" for i in range(401, 421)]
    assert without_ids(moved.records) == without_ids(base.records)
    assert moved.report["id_start"] == {"test": 401}
    wide = req.generate(42, {"test": 20}, corpus, id_start={"test": 9990})
    assert [wide.records[0]["id"], wide.records[-1]["id"]] == ["req-test-09990", "req-test-10009"]
    for bad, match in (({"test": 0}, "positive"), ({"dev": 401}, "does not make"), ({"bogus": 1}, "split")):
        with pytest.raises(ValueError, match=match):
            req.generate(42, {"test": 20}, corpus, id_start=bad)


def test_an_id_a_preloaded_file_already_holds_is_refused_whatever_its_width(corpus, tmp_path):
    first = req.generate(42, {"test": 20}, corpus)
    path = write_jsonl(tmp_path / "first.jsonl", first.records)
    with pytest.raises(ValueError, match="req-test-0020"):
        req.generate(42, {"test": 20}, corpus, draw_seed=5, id_start={"test": 20}, preload=[path])
    with pytest.raises(ValueError, match="req-test-0001"):  # ids start at 1 unless told otherwise
        req.generate(42, {"test": 20}, corpus, draw_seed=5, preload=[path])
    after = req.generate(42, {"test": 20}, corpus, draw_seed=5, id_start={"test": 21}, preload=[path])
    assert after.records[0]["id"] == "req-test-0021" and after.preloaded == first.records
    unpadded = write_jsonl(tmp_path / "unpadded.jsonl", [made_request("req-test-21", "test", "Hello there.")])
    with pytest.raises(ValueError, match="req-test-0021"):
        req.generate(42, {"test": 20}, corpus, id_start={"test": 21}, preload=[unpadded])
    with pytest.raises(ValueError, match="repeat"):  # the same file twice
        req.generate(42, {"test": 20}, corpus, id_start={"test": 21}, preload=[path, path])


def test_the_writer_refuses_a_record_whose_id_a_preloaded_file_holds(corpus, tmp_path):
    held = made_request("req-dev-0001", "dev", "Thanks, that is all for today.")
    path = write_jsonl(tmp_path / "pre.jsonl", [held])
    build = req.generate(42, {"dev": 20}, corpus, draw_seed=9, id_start={"dev": 2}, preload=[path])
    records = copy.deepcopy(build.records)
    records[0]["id"] = "req-dev-1"  # req-dev-0001 but for the zero padding, an id generate itself refuses
    out, manifest = tmp_path / "r.jsonl", tmp_path / "m.json"
    with pytest.raises(ValueError, match="ids already in a preloaded file, nothing written: req-dev-1$"):
        req.write_requests(records, out, manifest, build.report, preloaded=build.preloaded)
    assert not out.exists() and not manifest.exists()
    req.write_requests(build.records, out, manifest, build.report, preloaded=build.preloaded)
    assert read_jsonl(out) == build.records


@pytest.mark.parametrize("line, match", [("not json", "JSON"), ('{"id": "req-train-9002"}', "missing"),
                                         ("same id", "repeats line 1"), ("bad split", "split")])
def test_a_preloaded_file_must_hold_request_records(tmp_path, line, match):
    good = made_request("req-train-9001", "train", "Thanks, that is all for today.")
    if line == "same id":
        line = json.dumps({**good, "text": "Another text."})
    elif line == "bad split":
        line = json.dumps({**good, "id": "req-x-1", "split": "validation"})
    path = tmp_path / "pre.jsonl"
    path.write_text(json.dumps(good) + "\n" + line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=f"line 2: .*{match}"):
        req.load_preload(path)
    loaded = req.load_preload(write_jsonl(tmp_path / "ok.jsonl", [good]))
    assert loaded.records == (good,) and loaded.sha256 == sha(tmp_path / "ok.jsonl")
    assert loaded.summary() == {"path": (tmp_path / "ok.jsonl").as_posix(), "sha256": loaded.sha256,
                                "requests": 1, "splits": {"train": 1, "dev": 0, "test": 0}}


def test_preloaded_requests_seed_every_dedup_rule_each_in_its_own_direction():
    summary = "Summarise 2609.11716 in three sentences, please."
    near = BASE[:-1] + "!!"
    assert req.jaccard(masked(BASE), masked(near)) >= req.NEAR_DUP
    assert req.TRAIN_VS_EVAL <= req.jaccard(masked(BASE), masked(CLOSE)) < req.NEAR_DUP
    deduper = req.Deduper()
    deduper.register(BASE, "train")
    deduper.register(summary, "test")
    assert deduper.preloaded == 2 and deduper.offered == 0
    assert deduper.offer("  " + BASE.replace(" ", "  "), "test") == "exact"  # against a request of any split
    assert deduper.offer(near, "dev") == "near"  # an evaluation request near a preloaded train request
    assert deduper.offer(CLOSE, "dev") is None  # train vs eval never runs from evaluation to preloaded train
    assert deduper.offer(in_band_variant(summary), "train") == "train_vs_eval"  # train near a preloaded test
    assert deduper.dropped == {"exact": 1, "near": 1, "train_vs_eval": 1} == deduper.dropped_preloaded
    # drops against new requests count, but not as drops against preloaded ones: between is in the band of
    # the new dev request CLOSE and of the preloaded train request BASE, and only CLOSE can make it a drop
    between = "Find three papers on speculative decoding published since September 2026, newest first."
    for other in (CLOSE, BASE):
        assert req.TRAIN_VS_EVAL <= req.jaccard(masked(between), masked(other)) < req.NEAR_DUP, other
    assert req.jaccard(masked(between), masked(summary)) < req.TRAIN_VS_EVAL
    assert deduper.offer(between, "train") == "train_vs_eval"  # against the new dev request CLOSE
    assert deduper.offer(CLOSE, "test") == "exact"
    assert deduper.dropped == {"exact": 2, "near": 1, "train_vs_eval": 2}
    assert deduper.dropped_preloaded == {"exact": 1, "near": 1, "train_vs_eval": 1}
    assert deduper.offered == 6 and deduper.preloaded == 2
    within_train = req.Deduper()
    within_train.register(BASE, "train")
    assert within_train.offer(CLOSE, "train") is None  # within train only the near rule applies


def test_a_build_with_its_predecessor_preloaded_repeats_none_of_it(corpus, tmp_path):
    first = req.generate(42, SMALL, corpus)
    path = write_jsonl(tmp_path / "first.jsonl", first.records)
    second = req.generate(42, SMALL, corpus, preload=[path], id_start={"train": 41, "dev": 21, "test": 21})
    assert second.preloaded == first.records
    # the same draw seed offers every request of the first build again: each is dropped against its file
    assert second.report["dedup_against_preloaded"]["exact"] >= len(first.records)
    assert second.report["preload"] == [{"path": path.as_posix(), "sha256": sha(path), "requests": 80,
                                         "splits": {"train": 40, "dev": 20, "test": 20}}]
    old_texts = {normalize_text(r["text"]) for r in first.records}
    old = [(r["split"], masked(r["text"])) for r in first.records]
    for r in second.records:
        assert normalize_text(r["text"]) not in old_texts, r
        new = masked(r["text"])
        for split, shingle_set in old:
            similarity = req.jaccard(new, shingle_set)
            assert similarity < req.NEAR_DUP, r
            if r["split"] == "train" and split != "train":
                assert similarity < req.TRAIN_VS_EVAL, r


def test_a_new_train_request_close_to_a_preloaded_evaluation_request_is_dropped(corpus, tmp_path):
    plain = req.generate(42, {"train": 40}, corpus, draw_seed=3)
    target = next(r for r in plain.records if r["lang"] == "en")
    variant = in_band_variant(target["text"])
    path = write_jsonl(tmp_path / "eval.jsonl", [made_request("req-dev-9001", "dev", variant)])
    again = req.generate(42, {"train": 40}, corpus, draw_seed=3, preload=[path])
    assert target["text"] not in {r["text"] for r in again.records}
    assert again.report["dedup_against_preloaded"]["train_vs_eval"] >= 1
    assert all(req.jaccard(masked(r["text"]), masked(variant)) < req.TRAIN_VS_EVAL for r in again.records)


def test_a_new_evaluation_request_close_to_a_preloaded_train_request_is_kept(corpus, tmp_path):
    plain = req.generate(42, {"test": 20}, corpus, draw_seed=3)
    target = next(r for r in plain.records if r["lang"] == "en")
    variant = in_band_variant(target["text"])
    path = write_jsonl(tmp_path / "train.jsonl", [made_request("req-train-9001", "train", variant)])
    again = req.generate(42, {"test": 20}, corpus, draw_seed=3, preload=[path])
    assert again.records == plain.records and target in again.records
    assert again.report["dedup_against_preloaded"]["train_vs_eval"] == 0


def test_a_new_train_request_close_to_a_preloaded_train_request_is_kept(corpus, tmp_path):
    plain = req.generate(42, {"train": 40}, corpus, draw_seed=3)
    target = next(r for r in plain.records if r["lang"] == "en")
    variant = in_band_variant(target["text"])
    path = write_jsonl(tmp_path / "train.jsonl", [made_request("req-train-9001", "train", variant)])
    again = req.generate(42, {"train": 40}, corpus, draw_seed=3, preload=[path])
    assert again.records == plain.records and target in again.records  # within train only the near rule
    assert again.report["dedup_against_preloaded"]["train_vs_eval"] == 0


def test_a_mix_names_known_scenarios_with_shares_that_sum_to_one():
    assert req.check_mix(PLANNED_MIX) == PLANNED_MIX and req.check_mix(V2_MIX) == V2_MIX
    assert req.check_mix({"normal": 0.5, "clarify": 0.5}) == {"normal": 0.5, "no_call": 0.0, "clarify": 0.5,
                                                              "recover": 0.0, "parallel": 0.0}
    for bad, match in (({"normal": 0.5}, "sum"), ({}, "sum"), ({"normal": 0.9, "bogus": 0.1}, "scenario"),
                       ({"normal": 1.2, "clarify": -0.2}, "share"), ({"normal": float("nan")}, "share"),
                       ({"normal": float("inf")}, "share")):
        with pytest.raises(ValueError, match=match):
            req.check_mix(bad)


def test_plan_counts_applies_a_mix_by_largest_remainder():
    assert req.plan_counts(400) == req.plan_counts(400, PLANNED_MIX)
    assert by_scenario(req.plan_counts(260, V2_MIX)) == {"normal": 104, "no_call": 39, "clarify": 39,
                                                         "recover": 52, "parallel": 26}
    assert by_scenario(req.plan_counts(220, V2_MIX)) == {"normal": 88, "no_call": 33, "clarify": 33,
                                                         "recover": 44, "parallel": 22}
    assert by_scenario(req.plan_counts(1300, M7_MIX)) == {"normal": 650, "no_call": 130, "clarify": 260,
                                                          "recover": 156, "parallel": 104}
    tie = req.plan_counts(7, {"normal": 0.5, "clarify": 0.5})  # 3.5 and 3.5: the tie goes in SCENARIOS order
    assert tie["normal"] == {"zh": 2, "en": 2} and tie["clarify"] == {"zh": 2, "en": 1}
    assert by_scenario(tie) == {"normal": 4, "no_call": 0, "clarify": 3, "recover": 0, "parallel": 0}


def test_a_build_with_its_own_mix_fills_that_mix_and_records_it(corpus, tmp_path):
    build = req.generate(42, {"test": 60}, corpus, mix=V2_MIX)
    assert req.counts_by_cell(build.records)["test"] == req.plan_counts(60, V2_MIX)
    assert build.report["planned_mix"] == V2_MIX
    manifest = req.write_requests(build.records, tmp_path / "r.jsonl", tmp_path / "m.json", build.report)
    assert manifest["planned_mix"] == V2_MIX
    with pytest.raises(ValueError, match="sum"):
        req.generate(42, {"test": 60}, corpus, mix={"normal": 0.5})


@pytest.mark.parametrize("option", ["draw_seed", "only", "id_start", "preload", "mix"])
def test_a_build_off_the_defaults_in_any_one_parameter_records_all_of_them(corpus, tmp_path, option):
    held = made_request("req-train-9001", "train", "Thanks, that is all for today.")
    path = write_jsonl(tmp_path / "pre.jsonl", [held])
    changed = {"draw_seed": 7, "only": ["test", "dev"], "id_start": {"test": 21}, "preload": [path],
               "mix": V2_MIX}
    build = req.generate(42, SMALL, corpus, **{option: changed[option]})
    manifest = req.write_requests(build.records, tmp_path / "r.jsonl", tmp_path / "m.json", build.report,
                                  preloaded=build.preloaded)
    recorded = {"draw_seed": 42, "only": ["train", "dev", "test"],  # the defaults, as a manifest records them
                "id_start": {"train": 1, "dev": 1, "test": 1}, "preload": [],
                "dedup_against_preloaded": NO_DROPS}
    recorded.update({
        "draw_seed": {"draw_seed": 7},
        "only": {"only": ["dev", "test"], "id_start": {"dev": 1, "test": 1}},  # only the splits built
        "id_start": {"id_start": {"train": 1, "dev": 1, "test": 21}},
        "preload": {"preload": [req.load_preload(path).summary()]},
        "mix": {},
    }[option])
    assert {key: build.report.get(key) for key in recorded} == recorded
    assert {key: manifest.get(key) for key in recorded} == recorded
    assert manifest["planned_mix"] == (V2_MIX if option == "mix" else PLANNED_MIX)


def test_the_leakage_audit_runs_over_preloaded_and_new_requests_together(corpus, tmp_path):
    leaky = copy.deepcopy(req.generate(42, SMALL, corpus).records)
    with_topic = [r for r in leaky if "topic_id" in r["slots"]]
    test_topic = next(r["slots"]["topic_id"] for r in with_topic if r["split"] == "test")
    next(r for r in with_topic if r["split"] == "train")["slots"]["topic_id"] = test_topic
    path = write_jsonl(tmp_path / "leaky.jsonl", leaky)
    build = req.generate(42, {"dev": 20}, corpus, draw_seed=9, id_start={"dev": 21}, preload=[path])
    assert req.audit_summary(req.audit(build.records))["ok"]  # the new requests alone are clean
    out, manifest = tmp_path / "r.jsonl", tmp_path / "m.json"
    with pytest.raises(ValueError, match="topics train/test"):
        req.write_requests(build.records, out, manifest, build.report, preloaded=build.preloaded)
    with pytest.raises(ValueError, match="preloaded"):  # the report names a preloaded file: pass its records
        req.write_requests(build.records, out, manifest, build.report)
    assert not out.exists() and not manifest.exists()


def test_the_template_audit_refuses_a_preloaded_request_outside_its_template_split(corpus, tmp_path):
    misplaced = {**made_request("req-train-9001", "train", "Thanks, that is all for today."),
                 "family": family_of("test")}
    path = write_jsonl(tmp_path / "misplaced.jsonl", [misplaced])
    build = req.generate(42, {"dev": 20}, corpus, draw_seed=9, preload=[path])
    assert req.audit_summary(req.audit(build.preloaded + build.records))["ok"]
    result = req.template_audit(req.split_templates(42), build.preloaded + build.records)
    assert not result["ok"] and result["misplaced"] == [[misplaced["family"], "train", "test"]]
    assert result["over"] == [] and "FAIL" in req.template_audit_line(result)
    assert req.template_audit(req.split_templates(42), build.records)["ok"]
    out, manifest = tmp_path / "r.jsonl", tmp_path / "m.json"
    with pytest.raises(ValueError, match="template audit"):
        req.write_requests(build.records, out, manifest, build.report, preloaded=build.preloaded)
    assert not out.exists() and not manifest.exists()


@pytest.mark.needs_data("data/synth")  # the registered request files and their manifest
def test_a_build_on_the_registered_files_passes_both_audits_and_records_its_parameters(corpus, tmp_path):
    build = req.generate(42, {"test": 26, "dev": 22}, corpus, only=["test", "dev"], draw_seed=20261020,
                         id_start={"test": 401, "dev": 401}, preload=REGISTERED, mix=V2_MIX)
    out, manifest_path = tmp_path / "v2.jsonl", tmp_path / "v2-manifest.json"
    manifest = req.write_requests(build.records, out, manifest_path, build.report, preloaded=build.preloaded)
    assert json.loads(manifest_path.read_text("utf-8")) == manifest and read_jsonl(out) == build.records
    assert manifest["requests"] == 48 and manifest["totals"] == {"train": 0, "dev": 22, "test": 26}
    assert manifest["audit"] == {"families": 0, "topics": 0, "paper_ids": 0, "ok": True}
    m6_families = sorted({r["family"] for r in read_jsonl(REGISTERED[1])} - set(req.split_templates(42)))
    assert m6_families and manifest["template_audit"]["uncatalogued"] == m6_families
    assert manifest["template_audit"]["ok"] and manifest["template_audit"]["misplaced"] == 0
    assert manifest["draw_seed"] == 20261020 and manifest["only"] == ["dev", "test"]
    assert manifest["id_start"] == {"dev": 401, "test": 401} and manifest["planned_mix"] == V2_MIX
    assert manifest["preload"] == [
        {"path": REGISTERED[0].as_posix(), "sha256": sha(REGISTERED[0]), "requests": 2400,
         "splits": {"train": 1600, "dev": 400, "test": 400}},
        {"path": REGISTERED[1].as_posix(), "sha256": sha(REGISTERED[1]), "requests": 510,
         "splits": {"train": 510, "dev": 0, "test": 0}}]
    assert set(manifest["dedup_against_preloaded"]) == set(req.DROP_REASONS)
    registered = json.loads((REPO / "data" / "synth" / "requests-manifest.json").read_text("utf-8"))
    assert set(registered) <= set(manifest)
    for key in ("seed", "topics", "paper_ids", "families", "window", "corpus_manifest_sha256"):
        assert manifest[key] == registered[key], key  # the split is the registered build's
    with_topic = [r for r in build.records if "topic_id" in r["slots"]]
    for split in ("dev", "test"):
        used = {r["slots"]["topic_id"] for r in with_topic if r["split"] == split}
        assert used and used <= set(registered["topics"][split]), split

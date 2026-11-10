"""The second family's hand-written specs (v2, M7: docs/m7-计划.md WP-I 4) in research_agent.eval.handwritten
and scripts/build_handwritten_items.py: check 10 against every request file and, through eval.lock.RetiredSet,
against v1's hand-written requests and single-step test items; ids from hw-test-165; the writers' plan, batch
check and replay. And v1's behaviour, byte for byte what the script wrote before v2 (a fixture: v1's own specs
file holds test specs and a test run may not open it)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from data.test_synth import ID1, ID2
from eval.test_handwritten import REQUEST, TOPIC, call, load_script, search, spec, world
from eval.test_retired_reads import SYSTEM, write_v1

from research_agent.data.requests import jaccard, mask_text, shingles
from research_agent.data.synth import SeedPool, load_seed_pool
from research_agent.eval.handwritten import (
    BATCH_SIZE,
    FIRST_ID_V2,
    HANDWRITTEN_LIMIT,
    TEMPLATE_LIMIT,
    Retired,
    SpecError,
    World,
    batch_problems,
    build_items,
    check_shape,
    numbering_problems,
    peer_problems,
    request_texts,
    retired_v1,
    stratum_problems,
    writing_plan,
)
from research_agent.eval.lock import (
    HANDWRITTEN_SPECS,
    HANDWRITTEN_SPECS_RETIRED_READS,
    TEST_ACCESS_LOG,
    TEST_RETIRED_READS,
    TEST_SET,
    LockError,
    RetiredSet,
    read_jsonl,
    write_jsonl,
)

REPO = Path(__file__).resolve().parents[2]
NEEDED_V2 = REPO / "data/eval/v2/review/handwritten-needed.json"
PLAN = REPO / "data/eval/v2/handwritten-specs/plan.json"
WRITING = REPO / "data/eval/v2/handwritten-writing.md"
NEEDS_V2 = pytest.mark.needs_data("data/eval/v2")  # v2's review and specs: left out of the public copy
INSTRUCTIONS = REPO / "data/eval/v2/review/instructions.md"
REQUEST_FILES = ("requests.jsonl", "requests-m6.jsonl", "requests-v2.jsonl", "requests-v2b.jsonl",
                 "requests-m7-train.jsonl")


def sha(data: bytes | str) -> str:
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def write_specs(path: Path, specs: list[dict]) -> Path:
    path.write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in specs), encoding="utf-8")
    return path


def largest(text: str, others: list[str]) -> float:
    mine = shingles(mask_text(text))
    return max(jaccard(mine, shingles(mask_text(other))) for other in others)


# ---- v1, byte-identical ----------------------------------------------------------------------------------

# Every type, both languages and both splits, over the v1 tests' world (tests/eval/test_handwritten.py).
V1_SPECS = [
    spec(),
    spec("next_step", request="Search speculative decoding and open the abstract of the top result.",
         prior=[[call("search_arxiv", query="speculative decoding")]],
         expected={"tool_calls": [call("get_paper", arxiv_id=ID1)]}, sid="hw-dev-002"),
    spec("no_call", request="Thanks, that was all for today!", slots={},
         expected={"content": "You're welcome."}, sid="hw-dev-003"),
    spec("clarify", request="Summarize the paper for me.", slots={"missing": ["paper_id"]},
         expected={"tool_calls": [call("ask_clarification", question="Which paper?", missing=["paper_id"])]},
         sid="hw-dev-004"),
    spec("recover", request=f"Get {ID1} for me, it is on speculative decoding.",
         slots={**TOPIC, "arxiv_id": ID1},
         prior=[[{**call("get_paper", arxiv_id=ID1), "inject": "RATE_LIMITED"}]],
         expected={"tool_calls": [call("get_paper", arxiv_id=ID1)]}, sid="hw-dev-005"),
    spec("parallel", request=f"Get the abstracts of {ID1} and {ID2} at once.",
         slots={"arxiv_ids": [ID1, ID2]},
         expected={"tool_calls": [call("get_paper", arxiv_id=ID1), call("get_paper", arxiv_id=ID2)]},
         sid="hw-dev-006"),
    spec("recover", request="Speculative decoding since 2026-10-01; if nothing, drop the date.",
         slots={**TOPIC, "date_from": "2026-10-01"},
         prior=[[{**search(date_from="2026-10-01"), "inject": "EMPTY_RESULTS"}]], sid="hw-dev-007"),
    spec("no_call", lang="zh", request="谢谢,今天先到这里。", slots={}, expected={"content": "不客气。"},
         sid="hw-dev-008"),
    spec("no_call", split="test", request="Thank you so much, goodbye for now!", slots={},
         expected={"content": "Goodbye!"}, sid="hw-test-001"),
]
V1_BAD = [
    spec("call", request="Thanks, I am done for the week.", slots={}, expected={"content": "x"},
         sid="hw-dev-009"),
    spec(sid="hw-dev-010", request=REQUEST.replace(".", "!")),
    spec("no_call", request="Bye for now, and thanks.", slots={}, expected={"content": "Bye."},
         sid="hw-dev-003"),
]
V1_NEEDED = {"strata": {"dev/call/en": {"needed": 2}, "dev/no_call/en": {"needed": 0},
                        "dev/clarify/en": {"needed": 4}, "test/no_call/en": {"needed": 0}}}
# Every exit of v1's script: the specs (None: no specs file) and the arguments, <needed> standing for a needed
# file of V1_NEEDED and <none> for a needed file that does not exist.
V1_CASES = {"ok": (V1_SPECS, ()), "test_split": (V1_SPECS, ("--split", "test")),
            "bad": (V1_SPECS + V1_BAD, ()), "needed": (V1_SPECS, ("--needed", "<needed>")),
            "dev_split_needed": (V1_SPECS, ("--split", "dev", "--needed", "<needed>")),
            "not_object": ([V1_SPECS[0], [1, 2]], ()), "no_specs": (None, ()),
            "no_needed": (V1_SPECS, ("--needed", "<none>"))}
# Computed with the script and module of commit 92bac51 (before v2) on the fixture above: the items file's
# SHA-256, and stdout and stderr with the temporary paths replaced by <out>, <specs>, <needed> and <none>.
V1_GOLDEN = {
    "ok": (0, "ea47888ec2c5732076cdb41456a79d7067e28b7d2157e262cb4b6269f8060629",
           "9f3dd01dc3b6adb0f53d1a9da505579e29770a0db376641ab8552b6dca1ff5ec", ""),
    "test_split": (0, "9c1d7bc8d7d4df2d8d16ba1d43bf0ba0e8b3349cf3e2d5b03b9156b3bd7146c4",
                   "a3ab4aa1fc10fb86d02b7b3887a52f2330c9722c178476031a051335a66e119e", ""),
    "bad": (1, None, "3992d1eb687301db81b443effde79237bf6ef6d2c9b23be6e7f2be34ec97c9fa",
            "3dbb39ff775814f4f5aa6bc1eaa8e5969b6b4f42107a754fae357800d812e4f1"),
    "needed": (1, None, "22a8f7b763116a31d1e36a38694fbc96da6ba4dccba09ec4c34c86c5ff5e4ebf",
               "3dbb39ff775814f4f5aa6bc1eaa8e5969b6b4f42107a754fae357800d812e4f1"),
    "dev_split_needed": (1, None, "22a8f7b763116a31d1e36a38694fbc96da6ba4dccba09ec4c34c86c5ff5e4ebf",
                         "3dbb39ff775814f4f5aa6bc1eaa8e5969b6b4f42107a754fae357800d812e4f1"),
    "not_object": (1, None, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                   "f6e2c9dd2e8bb0e9f3272a72311708cd97a94cb8b57ed8103ab7491f47e20d3d"),
    "no_specs": (2, None, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                 "3b60be989584d5694a3d2f61d7bf45bd76edefe8e773f7cc5d67cdf7aca9a23a"),
    "no_needed": (2, None, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                  "c160f465f4c6258b03d50fa190a88fc1219c526d4ec5c7dbdff86bc1c20af574"),
}


def run_v1(tmp_path: Path, monkeypatch, capsys, specs: list | None, *args: str) -> tuple:
    script = load_script()
    monkeypatch.setattr(script, "load_world", lambda **kwargs: world())
    path, out = tmp_path / "specs.jsonl", tmp_path / "items.jsonl"
    for stale in (path, out):
        if stale.exists():
            stale.unlink()
    if specs is not None:
        write_specs(path, specs)
    shown = {str(out): "<out>", str(path): "<specs>", str(tmp_path / "needed.json"): "<needed>",
             str(tmp_path / "none.json"): "<none>"}
    real = {placeholder: name for name, placeholder in shown.items()}
    code = script.main(["--specs", str(path), "--out", str(out), *[real.get(arg, arg) for arg in args]])
    captured = capsys.readouterr()

    def norm(text: str) -> str:
        for name, placeholder in shown.items():
            text = text.replace(name, placeholder)
        return text

    items = sha(out.read_bytes()) if out.exists() else None
    return code, items, sha(norm(captured.out)), sha(norm(captured.err)) if captured.err else ""


@pytest.mark.parametrize("case", list(V1_CASES))
def test_v1_builds_byte_for_byte_what_it_built_before_v2(tmp_path, monkeypatch, capsys, case):
    (tmp_path / "needed.json").write_text(json.dumps(V1_NEEDED), encoding="utf-8")
    specs, args = V1_CASES[case]
    assert run_v1(tmp_path, monkeypatch, capsys, specs, *args) == V1_GOLDEN[case]


def test_v1_reads_its_one_request_file_and_nothing_retired(tmp_path, monkeypatch, capsys):
    script = load_script()
    seen = {}

    def load(**kwargs):
        seen.update(kwargs)
        return world()

    monkeypatch.setattr(script, "load_world", load)
    monkeypatch.setattr(script, "retired_v1", lambda *a, **k: pytest.fail("v1 read a retired file"))
    path = write_specs(tmp_path / "specs.jsonl", V1_SPECS[:1])
    assert script.main(["--specs", str(path), "--out", str(tmp_path / "items.jsonl")]) == 0
    assert seen == {"corpus_manifest": "data/corpus/manifest.json", "cassettes": "data/cassettes",
                    "requests": "data/synth/requests.jsonl",
                    "requests_manifest": "data/synth/requests-manifest.json"}


# ---- v2: the world and check 10 -------------------------------------------------------------------------


def v2_world(references: tuple[str, ...] = (), retired: tuple = ()) -> World:
    """The v1 tests' world with speculative decoding as a test topic, so test specs can be written on it."""
    base = world()
    pool = SeedPool(topics=base.pool.topics, names=base.pool.names,
                    splits={"train": ("kv-cache",), "dev": (), "test": ("speculative-decoding",)})
    return World(index=base.index, fetcher=base.fetcher, owner=base.owner, pool=pool, window=base.window,
                 references=tuple(references), retired=tuple(retired))


def v2(type_: str = "call", n: int = FIRST_ID_V2, **changes) -> dict:
    return spec(type_, split="test", sid=f"hw-test-{n:03d}", **changes)


THANKS = {"request": "Thanks, that was all for today!", "slots": {},
          "expected": {"content": "You're welcome."}}


class Scores:
    """A stand-in for RetiredSet: the given similarities, and a record of every call."""

    def __init__(self, *scores: float) -> None:
        self.scores, self.calls = list(scores), []

    def similarities(self, texts):
        self.calls.append(list(texts))
        return self.scores[:len(texts)]


class Refusing:
    def similarities(self, texts):
        raise LockError("test.jsonl is not retired: its access log test-access.jsonl holds 2 of its 3 opens")


def test_the_request_texts_of_several_files_come_in_file_order(tmp_path):
    files = []
    for k, texts in enumerate((["first a", "first b"], ["second a"]), 1):
        path = tmp_path / f"requests-{k}.jsonl"
        write_jsonl([{"id": f"req-train-{k}{n:03d}", "intent": "chit_chat", "text": t, "lang": "en",
                      "split": "train", "scenario": "no_call", "family": 1} for n, t in enumerate(texts)],
                    path)
        files.append(path)
    assert request_texts(files) == ["first a", "first b", "second a"]
    assert request_texts(files[1]) == ["second a"]  # one file, as v1 gives it


def test_check_10_compares_with_the_requests_of_every_file():
    later = "Search arXiv for some speculative decoding work, if you would."
    _, problems = build_items([v2()], v2_world(references=("an unrelated text about the weather", later,
                                                           REQUEST.replace("please", "thanks"))))
    score = largest(REQUEST, [REQUEST.replace("please", "thanks")])
    want = f"hw-test-{FIRST_ID_V2}: request too close to a generated request (Jaccard {score:.2f})"
    assert problems == [want]


def test_the_v1_check_holds_each_request_below_each_limit_and_reads_each_file_once():
    specs = [v2(), v2("no_call", n=166, **THANKS)]
    hand, items_ = Scores(0.79, 0.8), Scores(0.6, 0.59)
    retired = (Retired("v1's hand-written requests", hand, HANDWRITTEN_LIMIT),
               Retired("v1's single-step test items", items_, TEMPLATE_LIMIT))
    _, problems = build_items(specs, v2_world(retired=retired))
    assert problems == ["hw-test-166: request too close to v1's hand-written requests (Jaccard 0.80)",
                        "hw-test-165: request too close to v1's single-step test items (Jaccard 0.60)"]
    assert hand.calls == items_.calls == [[s["request"] for s in specs]]  # one read each, every request
    assert (HANDWRITTEN_LIMIT, TEMPLATE_LIMIT) == (0.8, 0.6)


def test_the_v1_check_reads_v1s_files_through_the_retired_interface_alone(tmp_path):
    v1_spec = "Find speculative decoding papers on arXiv and rank them by relevance for me."
    v1_item = "Please dig up recent speculative decoding work on arXiv, I want to read it tonight."
    item = {"id": "req-test-0009-s1", "split": "test", "type": "call", "lang": "en",
            "messages": [SYSTEM, {"role": "user", "content": v1_item}], "expected": {}}
    root = write_v1(tmp_path, items=[item], specs=[{"id": "hw-test-077", "request": v1_spec}])
    mine = {165: "Find speculative decoding papers on arXiv and rank them by relevance for me!",
            166: "Please dig up some speculative decoding work on arXiv, I would like to read it tonight.",
            167: "Could you find speculative decoding papers on arXiv and rank them by relevance?",
            168: "Which arXiv preprints cover speculative decoding? A short list would help my reading "
                 "group."}
    assert largest(mine[165], [v1_spec]) >= 0.8 and 0.6 <= largest(mine[166], [v1_item]) < 0.8
    assert 0.6 <= largest(mine[167], [v1_spec]) < 0.8  # near, but a hand-written request is held to 0.8
    access = (root / TEST_ACCESS_LOG).read_bytes()
    specs = [v2(n=n, request=text) for n, text in mine.items()]
    items, problems = build_items(specs, v2_world(retired=retired_v1(root)))
    assert [p.split(" (Jaccard")[0] for p in problems] == [
        "hw-test-165: request too close to v1's hand-written requests",
        "hw-test-166: request too close to v1's single-step test items"]
    assert len(items) == 4 and not [p for p in problems if v1_spec[:20] in p or v1_item[:20] in p]
    assert len(read_jsonl(root / HANDWRITTEN_SPECS_RETIRED_READS)) == 1
    assert [e["texts"] for e in read_jsonl(root / TEST_RETIRED_READS)] == [4]
    assert (root / TEST_ACCESS_LOG).read_bytes() == access


def test_v1s_retired_files_are_its_hand_written_specs_and_single_step_test_set_under_the_root():
    hand, single = retired_v1(REPO)  # nothing is read until a check asks for similarities
    assert (hand.name, hand.limit, single.name, single.limit) == (
        "v1's hand-written requests", HANDWRITTEN_LIMIT, "v1's single-step test items", TEMPLATE_LIMIT)
    assert isinstance(hand.reader, RetiredSet) and isinstance(single.reader, RetiredSet)
    assert (hand.reader.path, hand.reader.access_log) == (REPO / HANDWRITTEN_SPECS, REPO / TEST_ACCESS_LOG)
    assert (single.reader.path, single.reader.access_log) == (REPO / TEST_SET, REPO / TEST_ACCESS_LOG)


# ---- v2: ids, strata and the plan --------------------------------------------------------------------------


def test_v2_ids_are_test_ids_from_hw_test_165():
    assert FIRST_ID_V2 == 165
    specs = [v2(n=165), v2(n=301), {**v2(n=164)}, {**spec(sid="hw-dev-170")}, {"id": "hw-test-9"}]
    assert numbering_problems(specs) == [
        "hw-test-164: a v2 id is hw-test-<nnn> from hw-test-165 on (v1 used hw-test-001 to hw-test-164)",
        "hw-dev-170: a v2 id is hw-test-<nnn> from hw-test-165 on (v1 used hw-test-001 to hw-test-164)"]


def test_a_stratum_without_a_needed_count_takes_no_specs():
    needed = {"strata": {"dev/call/en": {"kept": 3, "dev-v2": {"quota": 4, "taken": 3, "short": 1}},
                         "test/call/en": {"needed": 1}, "test/no_call/en": {"needed": 5}}}
    test_specs = [v2(), v2(n=166), *[v2("no_call", n=n) for n in range(167, 174)]]
    assert stratum_problems(test_specs, needed) == []
    assert stratum_problems([*test_specs, spec()], needed) == [
        "stratum dev/call/en: 1 specs, none wanted (the stratum takes no hand-written items)"]


@NEEDS_V2
def test_the_plan_of_the_v2_needed_file():
    needed = json.loads(NEEDED_V2.read_text(encoding="utf-8"))
    plan = writing_plan(needed, folder="data/eval/v2/handwritten-specs")
    want = {key: row["needed"] for key, row in needed["strata"].items() if "needed" in row}
    assert list(plan["strata"]) == list(want) and all(k.startswith("test/") for k in want)
    numbers = []
    for batch in plan["batches"]:
        first, last = (int(batch[k].rsplit("-", 1)[1]) for k in ("first_id", "last_id"))
        assert last - first + 1 == batch["count"] <= BATCH_SIZE == 25
        assert batch["out"] == f"data/eval/v2/handwritten-specs/{batch['batch']}.jsonl"
        numbers += range(first, last + 1)
    assert numbers == list(range(165, 302))  # hw-test-165 to hw-test-301, in stratum order, no gap
    for key, n in want.items():
        row = plan["strata"][key]
        sizes = [b["count"] for b in plan["batches"] if b["stratum"] == key]
        assert row["specs"] == n + 1 + n // 5 == sum(sizes) and row["spares"] == 1 + n // 5
        assert len(sizes) == -(-row["specs"] // 25) and max(sizes) - min(sizes) <= 1
        assert row["batches"] == [b["batch"] for b in plan["batches"] if b["stratum"] == key]
    assert plan["totals"] == {"needed": 108, "spares": 29, "specs": 137, "batches": 13}
    assert [b["count"] for b in plan["batches"] if b["stratum"] == "test/call/zh"] == [16, 15]
    assert [b["batch"] for b in plan["batches"]] == [f"batch-{k:02d}" for k in range(1, 14)]
    assert (plan["first_id"], plan["last_id"], plan["split"], plan["batch_size"]) == (
        "hw-test-165", "hw-test-301", "test", 25)


def test_a_large_stratum_is_cut_into_near_equal_batches_and_ids_run_on():
    needed = {"strata": {"test/call/en": {"needed": 51}, "dev/call/en": {"needed": 4},
                         "test/no_call/en": {"needed": 0}}}
    plan = writing_plan(needed, folder="f", first=7, batch_size=25)
    assert [(b["stratum"], b["count"], b["first_id"], b["last_id"]) for b in plan["batches"]] == [
        ("test/call/en", 21, "hw-test-007", "hw-test-027"),
        ("test/call/en", 21, "hw-test-028", "hw-test-048"),
        ("test/call/en", 20, "hw-test-049", "hw-test-068"),
        ("test/no_call/en", 1, "hw-test-069", "hw-test-069")]
    with pytest.raises(ValueError, match="999"):
        writing_plan(needed, folder="f", first=950)


def test_a_batch_holds_exactly_its_planned_ids_of_its_stratum():
    batch = {"batch": "batch-02", "stratum": "test/no_call/en", "count": 3, "first_id": "hw-test-168",
             "last_id": "hw-test-170", "out": "f/batch-02.jsonl"}
    good = [v2("no_call", n=n, **THANKS) for n in (168, 169, 170)]
    assert batch_problems(good, batch) == []
    bad = [good[0], v2("call", n=169), v2("no_call", n=171, **THANKS)]
    assert batch_problems(bad, batch) == [
        "hw-test-169: stratum test/call/en, but batch-02 is test/no_call/en",
        "hw-test-171: not an id of batch-02 (hw-test-168 to hw-test-170)",
        "hw-test-170: missing from batch-02"]


def test_a_batch_is_compared_with_the_other_batches_present():
    mine = [v2(n=165), v2("no_call", n=166, **THANKS)]
    peers = [("batch-01.jsonl", v2(n=170, request=REQUEST.replace(",", ";"))),
             ("batch-03.jsonl", {"id": "hw-test-166", "request": "A different request about the weather."}),
             ("batch-03.jsonl", {"id": 7, "request": None})]
    score = largest(REQUEST, [REQUEST.replace(",", ";")])
    assert peer_problems(mine, peers) == [
        f"hw-test-165: request too close to hw-test-170 in batch-01.jsonl (Jaccard {score:.2f})",
        "hw-test-166: id also used in batch-03.jsonl"]
    assert peer_problems(mine, []) == []


def test_a_peer_is_held_to_the_hand_written_limit_not_the_template_limit():
    near = "Look on arXiv for work about speculative decoding, please, I need it for Friday."  # [0.6, 0.8)
    nearer = "Look on arXiv for some work about speculative decoding, please."  # 0.8 or more
    assert TEMPLATE_LIMIT <= largest(near, [REQUEST]) < HANDWRITTEN_LIMIT <= largest(nearer, [REQUEST])
    assert peer_problems([v2(n=165)], [("batch-02.jsonl", v2(n=181, request=near))]) == []
    found = peer_problems([v2(n=165)], [("batch-02.jsonl", v2(n=181, request=nearer))])
    assert [p.split(" (Jaccard")[0] for p in found] == [
        "hw-test-165: request too close to hw-test-181 in batch-02.jsonl"]


# ---- v2: the script ----------------------------------------------------------------------------------------


def run_v2(monkeypatch, capsys, argv: list[str], *, retired=(), references: tuple[str, ...] = ()) -> tuple:
    script = load_script()
    seen = {}

    def load(**kwargs):
        seen.update(kwargs)
        return v2_world(references=references, retired=kwargs.get("retired", ()))

    monkeypatch.setattr(script, "load_world", load)
    code = script.main(argv, retired=retired)
    captured = capsys.readouterr()
    return code, captured.out, captured.err, seen, script


V2_SPECS = [v2(), v2("no_call", n=166, **THANKS),
            v2("next_step", n=167,
               request="Search speculative decoding and open the abstract of the top result.",
               prior=[[call("search_arxiv", query="speculative decoding")]],
               expected={"tool_calls": [call("get_paper", arxiv_id=ID1)]})]


def test_the_v2_family_builds_over_every_request_file_with_the_v1_check(tmp_path, monkeypatch, capsys):
    path, out = write_specs(tmp_path / "specs.jsonl", V2_SPECS), tmp_path / "items.jsonl"
    hand = Scores(0.1, 0.1, 0.1)
    retired = (Retired("v1's hand-written requests", hand, HANDWRITTEN_LIMIT),)
    code, stdout, _, seen, script = run_v2(monkeypatch, capsys, ["--family", "v2", "--specs", str(path),
                                                                 "--out", str(out)], retired=retired)
    assert code == 0 and stdout.startswith(f"wrote 3 items to {out}")
    assert [Path(p) for p in seen["requests"]] == [REPO / "data/synth" / name for name in REQUEST_FILES]
    assert Path(seen["corpus_manifest"]) == REPO / "data/corpus/manifest.json"
    assert Path(seen["cassettes"]) == REPO / "data/cassettes"
    assert Path(seen["requests_manifest"]) == REPO / "data/synth/requests-manifest.json"
    assert seen["retired"] == retired and len(hand.calls) == 1
    assert [json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines()] == [
        "hw-test-165", "hw-test-166", "hw-test-167"]
    defaults = script.family_defaults("v2")
    assert (defaults["specs"], defaults["out"]) == (REPO / "data/eval/v2/handwritten-specs-v2.jsonl",
                                                    REPO / "data/eval/v2/handwritten-v2.jsonl")


def test_v2_refuses_an_id_below_165_and_names_it(tmp_path, monkeypatch, capsys):
    path, out = write_specs(tmp_path / "specs.jsonl", [v2(n=164)]), tmp_path / "items.jsonl"
    argv = ["--family", "v2", "--specs", str(path), "--out", str(out)]
    code, stdout, _, _, _ = run_v2(monkeypatch, capsys, argv)
    assert code == 1 and not out.exists()
    assert stdout.startswith("hw-test-164: a v2 id is hw-test-<nnn> from hw-test-165 on")


def test_v2_exits_2_when_v1s_retired_files_refuse_the_read(tmp_path, monkeypatch, capsys):
    path, out = write_specs(tmp_path / "specs.jsonl", V2_SPECS[:1]), tmp_path / "items.jsonl"
    retired = (Retired("v1's single-step test items", Refusing(), TEMPLATE_LIMIT),)
    code, _, stderr, _, _ = run_v2(monkeypatch, capsys, ["--family", "v2", "--specs", str(path), "--out",
                                                         str(out)], retired=retired)
    assert code == 2 and not out.exists()
    assert stderr.startswith("FAIL: the retired read was refused: test.jsonl is not retired")


def test_v2_with_needed_checks_the_test_strata_and_takes_several_specs_files(tmp_path, monkeypatch, capsys):
    needed = tmp_path / "needed.json"
    needed.write_text(json.dumps({"strata": {"dev/call/en": {"kept": 3}, "test/call/en": {"needed": 1},
                                             "test/no_call/en": {"needed": 0}}}), encoding="utf-8")
    other = v2(n=168, request="Find speculative decoding work on arXiv for my group.")
    first = write_specs(tmp_path / "a.jsonl", [V2_SPECS[0], other])
    second = write_specs(tmp_path / "b.jsonl", [V2_SPECS[1]])
    out = tmp_path / "items.jsonl"
    argv = ["--family", "v2", "--specs", str(first), str(second), "--out", str(out), "--needed", str(needed)]
    code, stdout, _, _, _ = run_v2(monkeypatch, capsys, argv)
    assert code == 0 and stdout.startswith(f"wrote 3 items to {out}")


def make_batches(tmp_path: Path) -> tuple[Path, dict]:
    """A plan of two batches beside its needed file: test/call/en, 2 needed (3 specs: hw-test-165 to 167), and
    test/no_call/en, 1 needed (2 specs: 168 and 169)."""
    needed = tmp_path / "needed.json"
    needed.write_text(json.dumps({"strata": {"dev/call/en": {"kept": 3}, "test/call/en": {"needed": 2},
                                             "test/no_call/en": {"needed": 1}}}), encoding="utf-8")
    folder = tmp_path / "handwritten-specs"
    folder.mkdir()
    script = load_script()
    plan = script.make_plan(needed, folder / "plan.json")
    (folder / "plan.json").write_text(script.render_plan(plan), encoding="utf-8")
    return folder, plan


CALLS = [v2(n=165),
         v2(n=166, request="Find speculative decoding work on arXiv for my group."),
         v2(n=167, request="Pull arXiv papers about speculative decoding for tomorrow's seminar.")]


def test_a_writers_batch_passes_and_nothing_is_written(tmp_path, monkeypatch, capsys):
    folder, plan = make_batches(tmp_path)
    assert [(b["batch"], b["stratum"], b["first_id"], b["last_id"]) for b in plan["batches"]] == [
        ("batch-01", "test/call/en", "hw-test-165", "hw-test-167"),
        ("batch-02", "test/no_call/en", "hw-test-168", "hw-test-169")]
    batch = write_specs(folder / "batch-01.jsonl", CALLS)
    before = sorted(p.name for p in tmp_path.rglob("*"))
    code, stdout, stderr, _, _ = run_v2(monkeypatch, capsys, ["--family", "v2", "--batch", str(batch)])
    assert (code, stderr) == (0, "")
    assert stdout == "OK: batch-01 (test/call/en): 3 specs pass every check; nothing written\n"
    assert sorted(p.name for p in tmp_path.rglob("*")) == before


def test_a_writers_batch_is_held_to_its_plan_and_to_the_other_batches(tmp_path, monkeypatch, capsys):
    folder, _ = make_batches(tmp_path)
    write_specs(folder / "batch-01.jsonl", CALLS)
    near = REQUEST.replace("please", "please!")
    bye = v2("no_call", n=165, request="Bye now, see you.", slots={}, expected={"content": "Bye!"})
    batch = write_specs(folder / "batch-02.jsonl",
                        [v2("no_call", n=168, **THANKS), bye, v2(n=169, request=near)])
    code, stdout, stderr, _, _ = run_v2(monkeypatch, capsys, ["--family", "v2", "--batch", str(batch)])
    assert code == 1
    lines = stdout.splitlines()
    assert "hw-test-165: a v2 id" not in stdout  # 165 is a v2 id, only not this batch's
    assert "hw-test-165: not an id of batch-02 (hw-test-168 to hw-test-169)" in lines
    assert "hw-test-169: stratum test/call/en, but batch-02 is test/no_call/en" in lines
    assert "hw-test-165: id also used in batch-01.jsonl" in lines
    assert [line for line in lines if line.startswith("hw-test-169: request too close to hw-test-165 in "
                                                      "batch-01.jsonl")]
    assert stderr.endswith("problem(s) in batch-02; nothing written\n")


def test_a_peer_that_cannot_be_read_is_named_and_passed_over(tmp_path, monkeypatch, capsys):
    folder, _ = make_batches(tmp_path)
    batch = write_specs(folder / "batch-01.jsonl", CALLS)
    (folder / "batch-02.jsonl").write_text('{"id": "hw-test-168", "request": "half writ', encoding="utf-8")
    code, stdout, stderr, _, _ = run_v2(monkeypatch, capsys, ["--family", "v2", "--batch", str(batch)])
    assert code == 0 and stdout.startswith("OK: batch-01")
    assert stderr.startswith("warning: batch-02.jsonl cannot be read") and "passed over" in stderr


def test_a_batch_without_its_plan_or_outside_it_exits_2(tmp_path, monkeypatch, capsys):
    stray = write_specs(tmp_path / "batch-01.jsonl", CALLS)
    code, _, stderr, _, _ = run_v2(monkeypatch, capsys, ["--family", "v2", "--batch", str(stray)])
    assert code == 2 and stderr.startswith("no plan at")
    folder, _ = make_batches(tmp_path)
    other = write_specs(folder / "batch-09.jsonl", CALLS)
    code, _, stderr, _, _ = run_v2(monkeypatch, capsys, ["--family", "v2", "--batch", str(other)])
    assert code == 2 and "batch-09.jsonl is not a batch of" in stderr
    with pytest.raises(SystemExit):
        load_script().main(["--batch", str(other)])  # v2 only
    capsys.readouterr()


@NEEDS_V2
def test_write_plan_writes_the_plan_and_the_committed_plan_is_its_output(tmp_path, monkeypatch, capsys):
    needed = tmp_path / "needed.json"
    needed.write_bytes(NEEDED_V2.read_bytes())
    plan_path = tmp_path / "out" / "plan.json"
    code, stdout, _, _, script = run_v2(monkeypatch, capsys, ["--family", "v2", "--needed", str(needed),
                                                              "--write-plan", str(plan_path)])
    assert code == 0 and stdout.startswith(f"wrote the plan of 137 specs in 13 batches to {plan_path}")
    written = json.loads(plan_path.read_text(encoding="utf-8"))
    committed = json.loads(PLAN.read_text(encoding="utf-8"))
    assert written["batches"][0]["out"].endswith("/out/batch-01.jsonl")
    assert PLAN.read_bytes() == script.render_plan(script.make_plan(NEEDED_V2, PLAN)).encode("utf-8")
    assert committed["needed"] == {"path": "data/eval/v2/review/handwritten-needed.json",
                                   "sha256": sha(NEEDED_V2.read_bytes())}
    assert committed["checker"] == script.CHECKER
    assert "{file}" in script.CHECKER and "--family v2 --batch" in script.CHECKER
    assert {k: v for k, v in written.items() if k not in ("needed", "batches")} == {
        k: v for k, v in committed.items() if k not in ("needed", "batches")}


def test_replay_shows_the_history_each_spec_leads_to(tmp_path, monkeypatch, capsys):
    path = write_specs(tmp_path / "specs.jsonl", [V2_SPECS[2], {**V2_SPECS[0], "id": "bad id"}])
    code, stdout, _, _, _ = run_v2(monkeypatch, capsys, ["--family", "v2", "--specs", str(path), "--replay"])
    assert code == 1
    first, second = (json.loads(line) for line in stdout.splitlines())
    assert first["id"] == "hw-test-167" and first["seen_ids"] == sorted(first["seen_ids"])
    assert [m["role"] for m in first["messages"]] == ["user", "assistant", "tool"]
    assert ID1 in first["seen_ids"] and first["messages"][0]["content"] == V2_SPECS[2]["request"]
    assert second["id"] == "bad id" and "id must match" in second["error"]


def test_replay_prints_ascii_lines_and_reads_no_retired_file(tmp_path, monkeypatch, capsys):
    """Redirected on Windows, stdout is cp1252: a line holding a Chinese request or an accented author would
    reach the reader as bytes no JSON reader decodes, so every non-ASCII character is escaped."""
    zh = v2("next_step", n=167, lang="zh", request="先搜投机解码(speculative decoding),再打开第一篇。",
            prior=[[call("search_arxiv", query="speculative decoding")]],
            expected={"tool_calls": [call("get_paper", arxiv_id=ID1)]})
    path = write_specs(tmp_path / "specs.jsonl", [zh, {**V2_SPECS[0], "id": "编号-165"}])
    hand = Scores(0.1, 0.1)
    retired = (Retired("v1's hand-written requests", hand, HANDWRITTEN_LIMIT),)
    argv = ["--family", "v2", "--specs", str(path), "--replay"]
    code, stdout, _, seen, _ = run_v2(monkeypatch, capsys, argv, retired=retired)
    assert code == 1 and stdout.isascii()
    first, second = (json.loads(line) for line in stdout.splitlines())
    assert first["messages"][0] == {"role": "user", "content": zh["request"]}
    assert second["id"] == "编号-165" and "id must match" in second["error"]
    assert "retired" not in seen and hand.calls == []


# ---- the writers' instructions ----------------------------------------------------------------------------


def rubric_section(text: str) -> str:
    """instructions.md from its "## Rubric" heading to the next heading of that level: the rubric with the
    tool list it is sent with."""
    start = text.index("## Rubric (v1, verbatim)")
    return text[start:text.index("\n## ", start) + 1]


@NEEDS_V2
def test_the_writers_instructions_quote_the_rubric_verbatim_and_state_v1s_lesson():
    writing = WRITING.read_text(encoding="utf-8")
    section = rubric_section(INSTRUCTIONS.read_text(encoding="utf-8"))
    assert section.count("\n") > 250 and section in writing
    from research_agent.agent.prompts import system_prompt

    assert system_prompt("2026-10-10") in writing  # the system prompt the model sees, today included
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["checker"] in writing and plan["writing"] == "data/eval/v2/handwritten-writing.md"
    for words in ("single most appropriate next action", "10 were direct answers", "default",
                  "hw-test-165", "plan.json", "--replay", "data/eval/v2/handwritten-specs/",
                  "seeded from the item's id and its slots' topic", "The lines are ASCII"):
        assert words in writing, words
    for k in range(1, 11):
        assert f"\n{k}. **" in writing, k  # the ten checks, numbered as in the module docstring
    assert "\r" not in writing and writing.endswith("\n")


@NEEDS_V2
def test_the_writers_instructions_list_the_plans_batches_and_strata():
    writing, plan = WRITING.read_text(encoding="utf-8"), json.loads(PLAN.read_text(encoding="utf-8"))
    rows = set(writing.splitlines())
    for b in plan["batches"]:
        ids = f"`{b['first_id']}` to `{b['last_id']}`"
        line = f"| `{b['batch']}` | `{b['stratum']}` | {b['count']} | {ids} | `{b['out']}` |"
        assert line in rows, b["batch"]
    for key, row in plan["strata"].items():
        names = ", ".join(f"`{name}`" for name in row["batches"])
        assert f"| `{key}` | {row['needed']} | {row['spares']} | {row['specs']} | {names} |" in rows, key
    totals = plan["totals"]
    assert (f"| all {len(plan['strata'])} | {totals['needed']} | {totals['spares']} | {totals['specs']} | "
            f"{totals['batches']} |") in rows
    assert f"{totals['specs']} specs,\nids `{plan['first_id']}` to `{plan['last_id']}`" in writing


@NEEDS_V2
def test_the_writers_instructions_quote_the_checkers_own_messages():
    writing = WRITING.read_text(encoding="utf-8")
    for change in ({"request": "找一些 tokenizer 论文"}, {"lang": "zh"}):
        with pytest.raises(SpecError) as refused:
            check_shape({**v2(), **change})
        message = str(refused.value).split(": ", 1)[1]
        assert f"`{message}`" in writing, message


@NEEDS_V2
@pytest.mark.needs_data("data/synth")  # the requests' manifest
def test_the_writers_instructions_give_the_world_of_the_test_split():
    """The test topics with their names, phrases and papers, every word of the other splits' topics (check 9
    refuses them in a request) and the window, as the corpus and requests manifests have them."""
    writing = WRITING.read_text(encoding="utf-8")
    corpus, requests = REPO / "data/corpus/manifest.json", REPO / "data/synth/requests-manifest.json"
    pool = load_seed_pool(corpus, requests)
    test = set(pool.allowed("test"))
    rows = [line for line in writing.splitlines() if line.startswith("| ")]
    for topic in json.loads(corpus.read_text(encoding="utf-8"))["topics"]:
        if topic["id"] in test:
            assert len(pool.topics[topic["id"]]) == 15
            assert f"| `{topic['id']}` | {topic['name']} | {', '.join(topic['phrases'])} |" in rows
            continue
        row = [line for line in rows if f"`{topic['id']}`," in line]
        assert len(row) == 1, topic["id"]
        assert all(f"`{word}`" in row[0] for word in (topic["id"], topic["name"], *topic["phrases"]))
    assert len(test) == 3 and "45 papers of these three topics" in writing
    first, last = json.loads(requests.read_text(encoding="utf-8"))["window"]
    assert f"The recording window runs from {first} to {last}, and today is {last}." in writing

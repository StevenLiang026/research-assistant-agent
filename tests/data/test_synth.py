"""The synthetic trajectory generator (research_agent.data.synth) and scripts/generate_trajectories.py,
offline: scripted models, a hand-made corpus, a fetcher that answers from a table, a stub client for the
rewriter and a fake API for the script."""

import copy
import importlib.util
import json
import sys
import time
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import httpx
import pytest
from support.fakes import HITS, ScriptedModel, call, table_check, turn
from support.strict_deepseek import Answer, StrictDeepSeek, sent_reasoning

from research_agent import notes
from research_agent.agent.prompts import PROMPT_HASH, PROMPT_VERSION
from research_agent.agent.types import (
    ANSWERED,
    BUDGET_STOP,
    CLARIFICATION,
    HARNESS_ERROR,
    HarnessError,
    ToolOutcome,
    Transcript,
)
from research_agent.canonical import canonical_json
from research_agent.data import synth
from research_agent.data.synth import (
    FETCH_FAULTS,
    PLANNED_MIX,
    PROMPT,
    RECOVER_CODES,
    REWRITE_PROMPT_HASH,
    REWRITE_PROMPT_VERSION,
    SCENARIOS,
    SPLITS,
    FaultSetting,
    FaultTrace,
    OwnerFunctionMissing,
    OwnerFunctions,
    Rewrite,
    Rewriter,
    TrajectoryValidator,
    check_requests,
    default_note_tools,
    generate,
    load_requests,
    load_validator,
    make_tools,
    no_search_url,
    preserved,
    probe,
    realized_mix,
    realized_scenario,
    run_request,
    select_requests,
)
from research_agent.llm.budget import Budget, BudgetExceeded, Price, cost_rmb
from research_agent.llm.cloud import THINKING_OFF
from research_agent.net.faults import (
    FAULTS,
    SEARCH_FAULTS,
    FaultInjector,
    SearchFaults,
    draw_key,
    search_key,
)
from research_agent.net.fetch import Fetched, Fetcher, ReplayMiss, canonical_url
from research_agent.notes import pick_evidence, render_note
from research_agent.tools import schemas
from research_agent.tools.adapters import fulltext_url, paper_url
from research_agent.tools.corpus_search import CorpusIndex

REPO = Path(__file__).resolve().parents[2]
TODAY = "2026-10-10"
WINDOW = ["2026-07-01", "2026-10-10"]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = load_script("generate_trajectories")


# ---- the world: a corpus of three papers, a fetcher that answers from a table, scripted owner functions ---


def paper(arxiv_id: str, title: str, abstract: str, published: str) -> dict:
    return {"arxiv_id": arxiv_id, "title": title, "authors": ["A. Author"], "published": published,
            "abstract": abstract, "categories": ["cs.CL"]}


CORPUS = [
    paper("2609.00001", "Adaptive Draft Length for Speculative Decoding",
          "Speculative decoding drafts tokens with a small model and verifies them.", "2026-09-15"),
    paper("2609.00002", "Tree Verification at Scale",
          "Tree-structured verification of drafted tokens for speculative decoding.", "2026-09-20"),
    paper("2608.00003", "KV Cache Compression for Long Context",
          "The key-value cache of long prompts is compressed by evicting entries.", "2026-08-05"),
]
ID1, ID2, ID3 = (p["arxiv_id"] for p in CORPUS)
PAGE = "<html><body><section>Method text.</section></body></html>"


def feed_text(papers: list[dict]) -> str:
    """What the scripted parse_atom reads: the papers as JSON."""
    return json.dumps(papers)


def parse_feed(xml: str) -> list[dict] | None:
    """The scripted parse_atom: the JSON feed's papers, None for a feed it cannot read (the probe's)."""
    try:
        return json.loads(xml)
    except json.JSONDecodeError:
        return None


class Pages:
    """The replay fetcher: answers from a table of URL -> (status, text), ReplayMiss for any other URL."""

    def __init__(self, pages: dict | None = None) -> None:
        if pages is None:
            pages = {paper_url(p["arxiv_id"]): (200, feed_text([p])) for p in CORPUS}
            pages |= {fulltext_url(p["arxiv_id"]): (200, PAGE) for p in CORPUS}
        self.pages, self.requests = pages, []

    def get(self, url: str) -> Fetched:
        self.requests.append(url)
        if url not in self.pages:
            raise ReplayMiss(f"no cassette for {url}")
        status, text = self.pages[url]
        return Fetched(url=url, final_url=url, status=status, body=text.encode("utf-8"), content_type="")


def decide(event: str, times: int) -> str:
    """scripts/m1_smoke.py's placeholder rule."""
    if event == "empty_results":
        return "relax" if times == 1 else "tell_user"
    return "stop"


def owner(check=None, decide_fn=decide, parse_atom=parse_feed,
          extract_section=lambda html, section, max_chars: "Method text.") -> OwnerFunctions:
    return OwnerFunctions(check or table_check(), decide_fn, parse_atom, extract_section)


class PricedModel(ScriptedModel):
    """A scripted model with CloudModel's cost method: 0.002 RMB per paid turn."""

    temperature = 0.0

    def cost(self, assistant_turn) -> float:
        return 0.002 if assistant_turn.paid else 0.0


def request(rid: str, scenario: str = "normal", lang: str = "zh", split: str = "dev", text: str | None = None,
            **extra) -> dict:
    texts = {"zh": "帮我找 2026-09-01 以后关于 speculative decoding 的论文,写一份笔记。",
             "en": "Find papers on speculative decoding since 2026-09-01 and write a note."}
    return {"id": rid, "intent": "find_recent_papers", "scenario": scenario, "lang": lang, "split": split,
            "family": f"test/{scenario}/{lang}", "text": text or texts[lang],
            "slots": {"topic": "speculative decoding", "date_from": "2026-09-01"}, **extra}


SEARCH = call("search_arxiv", "c1", query="speculative decoding", date_from="2026-09-01")
GET = call("get_paper", "c2", arxiv_id=ID1)
NOTE = call("write_note", "c3", arxiv_id=ID1, mode="scaffold")
ROW_KEYS = {"id", "intent", "scenario", "lang", "split", "family", "text", "rewritten", "rewrite_fallback",
            "rewrite_cost_rmb", "status", "error", "steps", "paid_calls", "cache_hits", "cost_rmb", "faults",
            "verdict", "placeholders", "transcript", "slots", "extra"}


def normal_turns(answer: str = "已为 2609.00001 写好笔记。") -> list:
    return [turn(SEARCH), turn(GET), turn(NOTE), turn(content=answer)]


@pytest.fixture(scope="module")
def index() -> CorpusIndex:
    return CorpusIndex(CORPUS)


def write_jsonl(path: Path, rows: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [row if isinstance(row, str) else json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8", newline="\n")
    return path


def corpus_manifest(tmp_path: Path, papers: list[dict] = CORPUS) -> Path:
    data = {"recorded_on": TODAY, "window": WINDOW, "topics": [
        {"id": "speculative-decoding", "name": "投机解码", "phrases": ["speculative decoding"],
         "papers": [p["arxiv_id"] for p in papers]}]}
    path = tmp_path / "corpus" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def run(requests: list[dict], model, out: Path, manifest: Path, **options) -> dict:
    """generate over the hand-made world, without git, logging to options["log"] when given; settings and
    today may be overridden through options. Seed notes are off unless options turn them on: these tests
    are about everything else, and the seeding tests below build a world of several topics for it."""
    settings = options.pop("settings", {"requests": "requests.jsonl", "run": "t"})
    return generate(requests, out=out, model=model, owner=options.pop("owner", None) or owner(),
                    fetcher=Pages(), index=CorpusIndex(CORPUS), today=options.pop("today", TODAY),
                    settings=settings, corpus_manifest=manifest, git_dir=None,
                    **{"log": lambda line: None, "seed_notes": False, **options})


def read_index(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "index.jsonl").read_text(encoding="utf-8").splitlines()]


def files_of(out: Path) -> dict[str, bytes]:
    """Every file under a run folder with its bytes, by its path inside the folder."""
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()}


def final_text(out: Path, rid: str) -> str | None:
    return json.loads((out / "transcripts" / f"{rid}.json").read_text(encoding="utf-8"))["final_text"]


# ---- request records --------------------------------------------------------------------------------------


def test_load_requests_reads_the_records_in_order_and_fills_in_empty_slots(tmp_path):
    rows = [request("b-2", "no_call", "en"), {**request("a-1"), "note": "pass-through"}]
    del rows[0]["slots"]
    loaded = load_requests(write_jsonl(tmp_path / "r.jsonl", [rows[0], "", rows[1]]))
    assert [r["id"] for r in loaded] == ["b-2", "a-1"]  # file order, blank lines skipped
    assert loaded[0]["slots"] == {} and loaded[1]["slots"] == {"topic": "speculative decoding",
                                                                "date_from": "2026-09-01"}
    assert loaded[1]["note"] == "pass-through"
    loaded[1]["slots"]["topic"] = "changed"
    assert rows[1]["slots"]["topic"] == "speculative decoding"  # copies, not the caller's dicts


@pytest.mark.parametrize("record, message", [
    ({**request("x"), "id": ""}, "id must be a non-empty string"),
    ({**request("x"), "id": "a/b"}, "names the transcript file"),
    ({**request("x"), "intent": 3}, "intent must be a non-empty string"),
    ({**request("x"), "scenario": "weird"}, "scenario must be one of"),
    ({**request("x"), "lang": "fr"}, "lang must be one of"),
    ({**request("x"), "split": "eval"}, "split must be one of"),
    ({**request("x"), "family": ""}, "family must be a non-empty string or an integer"),
    ({**request("x"), "family": True}, "family must be a non-empty string or an integer"),  # a bool is not
    ({**request("x"), "family": 2.5}, "family must be a non-empty string or an integer"),
    ({**request("x"), "text": "  "}, "text must be a non-empty string"),
    ({**request("x"), "slots": ["topic"]}, "slots must be an object"),
    ("[1, 2]", "line 2 is not a JSON object"),
    ("{not json", "line 2 is not JSON"),
])
def test_load_requests_refuses_a_bad_record_naming_its_line(tmp_path, record, message):
    path = write_jsonl(tmp_path / "r.jsonl", [request("ok"), record])
    with pytest.raises(ValueError, match="line 2") as caught:
        load_requests(path)
    assert message in str(caught.value)


def test_load_requests_rejects_a_repeated_id(tmp_path):
    path = write_jsonl(tmp_path / "r.jsonl", [request("same"), request("other"), request("same", "recover")])
    with pytest.raises(ValueError, match="line 3: id 'same' already used on line 1"):
        load_requests(path)


def test_load_requests_takes_an_integer_family_as_well_as_a_string(tmp_path):
    rows = [{**request("a"), "family": 3}, {**request("b"), "family": "f/1"}]
    assert [r["family"] for r in load_requests(write_jsonl(tmp_path / "r.jsonl", rows))] == [3, "f/1"]


def test_check_requests_makes_load_requests_checks_on_records_already_read():
    """The end-to-end test tasks a LockedSet.open returned stay in memory (scripts/run_test.py): the same
    checks as load_requests, each record named by its position in what was given."""
    rows = [request("b-2", "no_call", "en"), {**request("a-1"), "note": "pass-through"}]
    del rows[0]["slots"]
    checked = check_requests(rows, "opened")
    assert [r["id"] for r in checked] == ["b-2", "a-1"] and checked[0]["slots"] == {}
    assert checked[1]["note"] == "pass-through"
    checked[1]["slots"]["topic"] = "changed"
    assert rows[1]["slots"]["topic"] == "speculative decoding"  # copies, not the caller's dicts
    for bad, message in (({**request("x"), "lang": "fr"}, "opened: record 2: lang must be one of"),
                         ({**request("x"), "id": "a/b"}, "opened: record 2: id 'a/b' must be letters"),
                         ([1, 2], "opened: record 2 is not a JSON object")):
        with pytest.raises(ValueError, match=message):
            check_requests([request("ok"), bad], "opened")
    with pytest.raises(ValueError, match="opened: record 3: id 'same' already used by record 1"):
        check_requests([request("same"), request("other"), request("same", "recover")], "opened")


def test_select_requests_filters_by_scenario_and_split_and_limits():
    rows = [request("a", "normal", split="dev"), request("b", "recover", split="test"),
            request("c", "normal", split="test"), request("d", "normal", split="test")]
    assert [r["id"] for r in select_requests(rows)] == ["a", "b", "c", "d"]
    assert [r["id"] for r in select_requests(rows, scenario="normal")] == ["a", "c", "d"]
    assert [r["id"] for r in select_requests(rows, split="test")] == ["b", "c", "d"]
    assert [r["id"] for r in select_requests(rows, scenario="normal", split="test", limit=1)] == ["c"]
    assert select_requests(rows, scenario="clarify") == []


@pytest.mark.needs_data("data/synth/dryrun-requests.jsonl")
def test_the_dryrun_requests_cover_every_scenario_in_both_languages_with_real_paper_ids():
    rows = load_requests(REPO / "data" / "synth" / "dryrun-requests.jsonl")
    assert [r["id"] for r in rows] == [f"dry-{n:03d}" for n in range(1, 11)]
    assert {s: sum(r["scenario"] == s for r in rows) for s in SCENARIOS} == dict.fromkeys(SCENARIOS, 2)
    assert sum(r["lang"] == "zh" for r in rows) == 5 and sum(r["lang"] == "en" for r in rows) == 5
    assert sum(r["split"] == "dev" for r in rows) == 6 and sum(r["split"] == "test" for r in rows) == 4
    assert all(r["family"] == f"dryrun/{r['scenario']}/{r['lang']}" for r in rows)
    manifest = json.loads((REPO / "data" / "corpus" / "manifest.json").read_text(encoding="utf-8"))
    recorded = {arxiv_id for topic in manifest["topics"] for arxiv_id in topic["papers"]}
    for r in rows:
        assert r["intent"] and isinstance(r["slots"], dict)
        for arxiv_id in r["slots"].get("arxiv_ids", []):
            assert arxiv_id in recorded and arxiv_id in r["text"]
        if "date_from" in r["slots"]:
            assert manifest["window"][0] <= r["slots"]["date_from"] <= manifest["window"][1]
        if r["scenario"] == "parallel":
            assert len(r["slots"]["arxiv_ids"]) >= 2
        if r["scenario"] in ("clarify", "no_call"):
            assert "arxiv_ids" not in r["slots"] and "topic" not in r["slots"]


# ---- the rewriter ----------------------------------------------------------------------------------------


class StubClient:
    """CloudClient's surface for the rewriter: scripted answers, the counters, the price list."""

    model = "flash"

    def __init__(self, answers: list, cached: bool = False) -> None:
        self.answers, self.cached, self.payloads = list(answers), cached, []
        self.calls = self.cache_hits = 0
        self.budget = Budget(":memory:", stage="M2", stage_cap_rmb=1.0, prices={"flash": Price(2.0, 8.0)})

    def chat(self, messages: list[dict], tools: list[dict] | None = None, **params) -> dict:
        self.payloads.append({"messages": copy.deepcopy(messages), "tools": tools, **params})
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        if self.cached:
            self.cache_hits += 1
        else:
            self.calls += 1
        return {"choices": [{"message": {"role": "assistant", "content": answer}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 40}}


ORIGINAL = "帮我找 2026-09-01 以后关于 speculative decoding 的论文,顺便看看 2609.00001 的摘要。"
GOOD = "请从 2026-09-01 起找 speculative decoding 方面的论文,并读一下 2609.00001 的摘要。"


def test_a_rewrite_keeps_the_ids_and_dates_and_is_priced_like_a_model_turn():
    client = StubClient(["  " + GOOD + "\n"])
    rewrite = Rewriter(client).rewrite(ORIGINAL, "zh")
    usage = {"prompt_tokens": 120, "completion_tokens": 40}
    assert rewrite == Rewrite(GOOD, cost_rmb(Price(2.0, 8.0), usage), False, False)
    assert rewrite.cost_rmb == pytest.approx((120 * 2.0 + 40 * 8.0) / 1e6)
    assert client.payloads == [{"messages": [{"role": "system", "content": PROMPT["zh"]},
                                             {"role": "user", "content": ORIGINAL}],
                                "tools": None, "temperature": 0.7}]  # no tool schemas, the default heat
    assert Rewriter(StubClient([GOOD]), temperature=0.3).rewrite(ORIGINAL, "zh").text == GOOD
    client = StubClient(["Find papers on KV cache since 2026-09-01."])
    english = Rewriter(client, temperature=0.3).rewrite("Papers on KV cache from 2026-09-01.", "en")
    assert english.fallback is False and client.payloads[0]["temperature"] == 0.3
    assert client.payloads[0]["messages"][0]["content"] == PROMPT["en"]


@pytest.mark.parametrize("answer", [
    "请从 2026-09-01 起找 speculative decoding 方面的论文,并读一下那篇论文的摘要。",  # the id is gone
    "请找 speculative decoding 方面的论文,并读一下 2609.00001 的摘要。",  # the date is gone
    "请从 2026-09-01 起找论文,并读一下 2609.0001 的摘要。",  # the id is altered
    "请从 2026-09-01 起找论文,并读一下 2609.00001 和 2609.00002 的摘要。",  # an id is added (hallucinated)
    "请从 2026-09-01 到 2026-10-01 找论文,并读一下 2609.00001 的摘要。",  # a date is added
    "",
    "   ",
    " 帮我找  2026-09-01 以后关于 speculative decoding 的论文,顺便看看 2609.00001 的摘要。 ",  # spaces
])
def test_a_rewrite_that_loses_or_adds_a_literal_or_changes_nothing_falls_back_to_the_original(answer):
    client = StubClient([answer])
    rewrite = Rewriter(client).rewrite(ORIGINAL, "zh")
    assert rewrite.text == ORIGINAL and rewrite.fallback is True and rewrite.cached is False
    assert rewrite.cost_rmb > 0  # the call was still paid
    assert preserved(ORIGINAL, answer) is False and preserved(ORIGINAL, GOOD) is True


def test_a_cached_rewrite_costs_nothing():
    rewrite = Rewriter(StubClient([GOOD], cached=True)).rewrite(ORIGINAL, "zh")
    assert rewrite == Rewrite(GOOD, 0.0, True, False)


def test_budget_exceeded_from_the_rewriter_propagates_and_an_unknown_lang_is_refused():
    with pytest.raises(BudgetExceeded):
        Rewriter(StubClient([BudgetExceeded("stage M2: spent 90.0000 of 90.0 RMB")])).rewrite(ORIGINAL, "zh")
    with pytest.raises(ValueError, match="lang"):
        Rewriter(StubClient([GOOD])).rewrite(ORIGINAL, "fr")


def test_the_rewrite_prompts_are_fixed_versioned_and_hashed():
    import hashlib

    assert REWRITE_PROMPT_VERSION == "synth-2026-10-12"
    expected = hashlib.sha256((REWRITE_PROMPT_VERSION + PROMPT["zh"] + PROMPT["en"]).encode()).hexdigest()
    assert REWRITE_PROMPT_HASH == expected[:16] and set(PROMPT) == {"zh", "en"}
    assert "arXiv" in PROMPT["zh"] and "arXiv" in PROMPT["en"]


# ---- the owner's functions and the probe ------------------------------------------------------------------


def test_probe_names_the_owner_function_that_is_not_implemented():
    from research_agent.policy import decide as owner_decide
    from research_agent.validation import check_call as owner_check_call

    probe(owner(), TODAY)  # scripted stand-ins pass
    probe(owner(check=owner_check_call, decide_fn=owner_decide), TODAY)  # the implemented modules pass too

    def unfinished_check(tool, args, seen, today):
        raise NotImplementedError

    def unfinished_decide(event, times):
        raise NotImplementedError

    with pytest.raises(OwnerFunctionMissing, match="check_call") as caught:
        probe(owner(check=unfinished_check), TODAY)
    assert caught.value.name == "check_call"
    with pytest.raises(OwnerFunctionMissing) as caught:
        probe(owner(decide_fn=unfinished_decide), TODAY)
    assert caught.value.name == "decide"

    def unfinished_parse(xml):
        raise NotImplementedError

    def unfinished_extract(html, section, max_chars):
        raise NotImplementedError

    assert pytest.raises(OwnerFunctionMissing, probe, owner(parse_atom=unfinished_parse), TODAY).value.name \
        == "parse_atom"
    assert pytest.raises(OwnerFunctionMissing, probe, owner(extract_section=unfinished_extract),
                         TODAY).value.name == "extract_section"


def test_probe_refuses_an_answer_outside_the_contract_and_lets_other_failures_through():
    with pytest.raises(HarnessError, match="decide"):
        probe(owner(decide_fn=lambda event, times: "retry"), TODAY)
    with pytest.raises(HarnessError, match="check_call"):
        probe(owner(check=lambda tool, args, seen, today: None), TODAY)
    with pytest.raises(HarnessError, match="parse_atom"):
        probe(owner(parse_atom=lambda xml: {"papers": []}), TODAY)
    with pytest.raises(KeyError):
        probe(owner(parse_atom=lambda xml: {}["x"]), TODAY)
    assert probe(owner(parse_atom=lambda xml: None, extract_section=lambda h, s, m: None), TODAY) is None


def test_build_search_url_is_never_needed_and_refuses_to_be_called():
    with pytest.raises(HarnessError, match="corpus index"):
        no_search_url({"query": "x"}, TODAY)


# ---- tools per request ------------------------------------------------------------------------------------


def tools_for(scenario: str, task: str = "r1", seed: int = 42, fetcher=None, **options):
    return make_tools(fetcher=fetcher or Pages(), index=CorpusIndex(CORPUS), owner=owner(), today=TODAY,
                      scenario=scenario, seed=seed, task=task, **options)


def test_make_tools_serves_the_search_from_the_index_and_the_papers_from_the_fetcher():
    fetcher = Pages()
    tools, injectors = tools_for("normal", fetcher=fetcher)
    assert set(tools) == set(schemas.TOOL_NAMES) - {"ask_clarification"}
    assert tools["search_arxiv"] is injectors["search"] and isinstance(injectors["search"], SearchFaults)
    assert isinstance(injectors["fetch"], FaultInjector) and injectors["fetch"].inner is fetcher
    assert injectors["fetch"].faults == FETCH_FAULTS == injectors["search"].faults  # transport faults only
    assert injectors["fetch"].budget is injectors["search"].budget
    assert (injectors["fetch"].task, injectors["fetch"].seed) == ("r1", 42) == (injectors["search"].task,
                                                                               injectors["search"].seed)
    found = tools["search_arxiv"]({"query": "speculative decoding", "date_from": "2026-09-01"})
    assert [hit["arxiv_id"] for hit in found.content] == [ID1, ID2] and found.error is None
    assert fetcher.requests == []  # the search never reaches the fetcher
    looked_up = tools["get_paper"]({"arxiv_id": ID3})
    assert looked_up.error is None and looked_up.content["title"] == CORPUS[2]["title"]
    section = tools["fetch_fulltext"]({"arxiv_id": ID1, "section": "method"})
    assert section.content == {"arxiv_id": ID1, "section": "method", "source": "html", "text": "Method text."}
    assert fetcher.requests == [paper_url(ID3), fulltext_url(ID1)]
    assert tools["search_notes"]({"query": "x"}) == ToolOutcome([])


def requests_of(tools: dict, order: str) -> list[ToolOutcome]:
    steps = [("search_arxiv", {"query": "speculative decoding"}), ("get_paper", {"arxiv_id": ID1}),
             ("fetch_fulltext", {"arxiv_id": ID1}), ("search_arxiv", {"query": "kv cache"}),
             ("get_paper", {"arxiv_id": ID2}), ("search_arxiv", {"query": "speculative decoding"})]
    if order == "fetch first":
        steps = steps[1:] + steps[:1]
    return [tools[name](dict(args)) for name, args in steps]


@pytest.mark.parametrize("order", ["search first", "fetch first"])
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_a_recover_request_meets_exactly_one_fault_across_the_search_and_the_fetch(order, seed):
    tools, injectors = tools_for("recover", seed=seed)
    answers = requests_of(tools, order)
    plain = requests_of(tools_for("normal", seed=seed)[0], order)
    injected = {name: {k: v for k, v in injector.counts().items() if k != "none"}
                for name, injector in injectors.items()}
    assert sum(sum(c.values()) for c in injected.values()) == 1
    assert answers[0] != plain[0] and answers[1:] == plain[1:]  # the first request meets the fault, no other
    assert answers[0].content in ([], {})  # a fault shows no paper and no id
    if order == "search first":
        assert answers[0].error in RECOVER_CODES and injected["fetch"] == {}
        assert list(injected["search"]) == [injectors["search"].log[0]["fault"]]
    else:
        assert answers[0].error in (schemas.UPSTREAM_ERROR, schemas.RATE_LIMITED) and injected["search"] == {}
        assert list(injected["fetch"]) == [injectors["fetch"].log[0]["fault"]]
    assert all(e.get("budget_exhausted") for e in (injectors["fetch"].log + injectors["search"].log)[1:]
               if e["fault"] is None)
    assert injectors["fetch"].budget.injected == 1


@pytest.mark.parametrize("scenario", ["normal", "no_call", "clarify", "parallel"])
def test_every_other_scenario_meets_no_fault(scenario):
    tools, injectors = tools_for(scenario, seed=7)
    answers = requests_of(tools, "search first")
    assert answers == requests_of(tools_for("normal", seed=99)[0], "search first")
    assert injectors["fetch"].counts() == {"none": 3} and injectors["search"].counts() == {"none": 3}
    assert injectors["fetch"].rate == 0.0 and injectors["search"].rate == 0.0


def test_a_recover_request_never_meets_an_empty_search_only_a_transport_fault():
    assert set(FETCH_FAULTS) == set(SEARCH_FAULTS) - {"empty_results"}
    met = set()
    for seed in range(20):
        tools, injectors = tools_for("recover", seed=seed)
        tools["search_arxiv"]({"query": "speculative decoding"})
        met.add(injectors["search"].log[0]["fault"])
    assert met == set(FETCH_FAULTS)  # never empty_results; every transport fault over a few seeds


def drawn_search_fault(seed: int, task: str) -> str:
    """The fault SearchFaults draws for the first search of a recover request, from the hash itself (as
    tests/net/test_faults.py computes it): at rate 1.0 every draw injects, and k picks the fault."""
    _, k = draw_key(seed, task, search_key({"query": "speculative decoding"}), 0)
    return FETCH_FAULTS[k % len(FETCH_FAULTS)]


def met_search_fault(seed: int, task: str) -> str:
    """The fault the search injector of a recover request logged for its first search."""
    tools, injectors = tools_for("recover", task=task, seed=seed)
    requests_of(tools, "search first")
    return injectors["search"].log[0]["fault"]


def test_the_same_recover_request_meets_the_same_fault_in_every_run_drawn_per_task_and_seed():
    first = requests_of(tools_for("recover", task="dry-007", seed=42)[0], "search first")
    again = requests_of(tools_for("recover", task="dry-007", seed=42)[0], "search first")
    assert first == again
    assert met_search_fault(42, "dry-007") == drawn_search_fault(42, "dry-007")
    # a seed and a task whose hash draws another fault than (42, dry-007): the injectors must differ too
    other_seed = next(s for s in range(43, 10_000)
                      if drawn_search_fault(s, "dry-007") != drawn_search_fault(42, "dry-007"))
    other_task = next(t for t in (f"dry-{n:03d}" for n in range(8, 10_000))
                      if drawn_search_fault(42, t) != drawn_search_fault(42, "dry-007"))
    assert met_search_fault(other_seed, "dry-007") != met_search_fault(42, "dry-007")
    assert met_search_fault(42, other_task) != met_search_fault(42, "dry-007")


def test_the_note_stubs_answer_without_error_codes_except_a_note_without_a_paper():
    tools = default_note_tools()
    assert set(tools) == {"search_notes", "list_notes", "write_note"}
    assert tools["search_notes"]({"query": "推测解码", "top_k": 3}) == ToolOutcome([])  # no EMPTY_RESULTS
    assert tools["list_notes"]({}) == ToolOutcome([])
    assert tools["write_note"]({"arxiv_id": ID1}) == ToolOutcome({"note_id": f"{ID1}-note", "arxiv_id": ID1,
                                                                 "path": f"notes/{ID1}.md"})
    assert tools["write_note"]({"arxiv_id": ID1, "mode": "draft", "problem": "p"}).error is None
    for args in ({}, {"mode": "scaffold"}, {"arxiv_id": ""}, {"arxiv_id": ID1, "mode": ""},
                 {"arxiv_id": 7, "mode": "draft"}):
        assert tools["write_note"](args) == ToolOutcome({}, schemas.INVALID_ARGUMENTS)
    custom = {"write_note": lambda args: ToolOutcome({"note_id": "custom"})}
    tools, _ = tools_for("normal", note_tools=custom)
    assert tools["write_note"]({}) == ToolOutcome({"note_id": "custom"}) and "list_notes" not in tools


SEARCH_NOTES = call("search_notes", "c4", query="speculative decoding")


def test_a_write_note_in_run_request_writes_a_real_note_that_search_notes_then_finds(tmp_path):
    notes_dir = tmp_path / "run" / "notes" / "req-n"
    model = PricedModel([turn(SEARCH), turn(GET), turn(NOTE), turn(SEARCH_NOTES), turn(content="写好了。")])
    outcome = run_request(request("req-n"), model=model, owner=owner(), fetcher=Pages(),
                          index=CorpusIndex(CORPUS), today=TODAY, notes_dir=notes_dir)
    t = outcome.transcript
    record = {"note_id": f"{ID1}-1", "arxiv_id": ID1, "path": f"{ID1}-1.md"}
    assert json.loads(t.messages[7]["content"]) == record  # the stand-in's keys, a real record
    assert set(record) == set(default_note_tools()["write_note"]({"arxiv_id": ID1}).content)
    expected = render_note(CORPUS[0], {"arxiv_id": ID1, "mode": "scaffold"},
                           pick_evidence(CORPUS[0]["abstract"], notes.EVIDENCE_SENTENCES, []), TODAY)
    assert (notes_dir / record["path"]).read_text(encoding="utf-8") == expected
    assert "  - Speculative decoding drafts tokens with a small model and verifies them.\n" in expected
    hits = json.loads(t.messages[9]["content"])
    assert [h["note_id"] for h in hits] == [record["note_id"]] and hits[0]["title"] == CORPUS[0]["title"]
    assert t.status == ANSWERED
    # a paper outside the corpus is NOT_FOUND, and no file is written for it
    missing = PricedModel([turn(call("write_note", "c1", arxiv_id="2609.09999", mode="scaffold")),
                           turn(content="没有这篇。")])
    other = tmp_path / "other"
    t = run_request(request("req-m"), model=missing, owner=owner(), fetcher=Pages(),
                    index=CorpusIndex(CORPUS), today=TODAY, notes_dir=other).transcript
    assert json.loads(t.messages[3]["content"])["error"] == schemas.NOT_FOUND
    assert list(other.glob("*.md")) == []


def test_note_tools_that_are_never_called_create_no_store(tmp_path):
    notes_dir = tmp_path / "notes" / "req-q"
    run_request(request("req-q", "no_call"), model=PricedModel([turn(content="好的。")]), owner=owner(),
                fetcher=Pages(), index=CorpusIndex(CORPUS), today=TODAY, notes_dir=notes_dir)
    assert not notes_dir.exists()


# ---- one request ------------------------------------------------------------------------------------------


def test_run_request_gives_a_transcript_and_a_row_with_every_key():
    req = {**request("req-1", extra_key={"source": "hand"}), "slots": {"topic": "speculative decoding"}}
    model = PricedModel(normal_turns())
    outcome = run_request(req, model=model, owner=owner(), fetcher=Pages(), index=CorpusIndex(CORPUS),
                          today=TODAY)
    t = outcome.transcript
    assert (t.status, t.request, t.today) == (ANSWERED, req["text"], TODAY)
    assert [m["role"] for m in t.messages] == ["system", "user", "assistant", "tool", "assistant", "tool",
                                               "assistant", "tool", "assistant"]
    assert json.loads(t.messages[3]["content"])[0]["arxiv_id"] == ID1  # the search answered from the index
    assert json.loads(t.messages[5]["content"])["title"] == CORPUS[0]["title"]  # get_paper from the fetcher
    assert json.loads(t.messages[7]["content"]) == {"note_id": f"{ID1}-note", "arxiv_id": ID1,
                                                     "path": f"notes/{ID1}.md"}
    assert t.cost_rmb == pytest.approx(0.008) and t.paid_calls == 4
    assert set(outcome.row) == ROW_KEYS
    assert outcome.row == {
        "id": "req-1", "intent": "find_recent_papers", "scenario": "normal", "lang": "zh", "split": "dev",
        "family": "test/normal/zh", "text": req["text"], "rewritten": None, "rewrite_fallback": False,
        "rewrite_cost_rmb": 0.0, "status": ANSWERED, "error": None, "steps": 4, "paid_calls": 4,
        "cache_hits": 0, "cost_rmb": t.cost_rmb, "faults": {"fetch": {"none": 1}, "search": {"none": 1}},
        "verdict": None, "placeholders": False, "transcript": "transcripts/req-1.json",
        "slots": {"topic": "speculative decoding"}, "extra": {"extra_key": {"source": "hand"}}}
    assert outcome.faults == outcome.row["faults"]
    outcome.row["slots"]["topic"] = "changed"
    assert req["slots"]["topic"] == "speculative decoding"


class StubRewriter:
    """A rewriter that answers from a list: a string, or an exception to raise."""

    temperature = 0.7

    def __init__(self, answers: list) -> None:
        self.answers, self.asked = list(answers), []

    def rewrite(self, text: str, lang: str) -> Rewrite:
        self.asked.append((text, lang))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def test_run_request_runs_the_rewritten_text_and_records_the_rewrite():
    req = request("req-2", lang="en")
    rewriter = StubRewriter([Rewrite("Please find speculative decoding papers since 2026-09-01.", 0.0012,
                                     False, False)])
    model = PricedModel(normal_turns("done"))
    outcome = run_request(req, model=model, owner=owner(), fetcher=Pages(), index=CorpusIndex(CORPUS),
                          today=TODAY, rewriter=rewriter, placeholders=True)
    assert rewriter.asked == [(req["text"], "en")]
    assert outcome.transcript.request == "Please find speculative decoding papers since 2026-09-01."
    assert outcome.transcript.messages[1]["content"] == outcome.transcript.request
    assert model.contexts[0][1]["content"] == outcome.transcript.request  # the model saw the rewrite
    row = outcome.row
    assert (row["text"], row["rewritten"]) == (req["text"], outcome.transcript.request)
    assert (row["rewrite_fallback"], row["rewrite_cost_rmb"], row["placeholders"]) == (False, 0.0012, True)
    fallen = run_request(req, model=PricedModel(normal_turns("done")), owner=owner(), fetcher=Pages(),
                         index=CorpusIndex(CORPUS), today=TODAY,
                         rewriter=StubRewriter([Rewrite(req["text"], 0.001, True, True)]))
    assert fallen.row["rewritten"] == req["text"] and fallen.row["rewrite_fallback"] is True


def test_run_request_records_the_validators_verdict_and_what_it_was_shown():
    shown = []

    def check(transcript: dict, req: dict) -> list[str]:
        shown.append((transcript, req))
        return [] if transcript["status"] == ANSWERED else ["not answered"]

    req = request("req-3")
    validator = TrajectoryValidator(check)
    outcome = run_request(req, model=PricedModel(normal_turns()), owner=owner(), fetcher=Pages(),
                          index=CorpusIndex(CORPUS), today=TODAY, validator=validator)
    assert outcome.row["verdict"] == {"accepted": True, "problems": []}
    assert shown[0] == (json.loads(outcome.transcript.to_json()), req)  # the saved shape, the request record
    clarifying = ScriptedModel([turn(call("ask_clarification", "c1", question="哪篇?"))])
    outcome = run_request(req, model=clarifying, owner=owner(), fetcher=Pages(), index=CorpusIndex(CORPUS),
                          today=TODAY, validator=validator)
    assert outcome.row["verdict"] == {"accepted": False, "problems": ["not answered"]}
    assert outcome.row["status"] == CLARIFICATION and outcome.row["cost_rmb"] == 0.0  # no cost method: 0


def test_a_validator_that_raises_not_implemented_counts_as_absent_for_the_rest_of_the_run():
    def unfinished(transcript, req):
        raise NotImplementedError

    validator = TrajectoryValidator(unfinished)
    outcome = run_request(request("req-4"), model=PricedModel(normal_turns()), owner=owner(), fetcher=Pages(),
                          index=CorpusIndex(CORPUS), today=TODAY, validator=validator)
    assert outcome.row["verdict"] is None and validator.available is False
    broken = TrajectoryValidator(lambda transcript, req: "fine")
    with pytest.raises(HarnessError, match="list of strings"):
        run_request(request("req-5"), model=PricedModel(normal_turns()), owner=owner(), fetcher=Pages(),
                    index=CorpusIndex(CORPUS), today=TODAY, validator=broken)


def test_load_validator_finds_the_owners_module_or_gives_none(monkeypatch):
    assert load_validator("research_agent.no_such_module") is None
    stub = types.ModuleType("owner_trajectories_stub")
    stub.check_trajectory = lambda transcript, req: (_ for _ in ()).throw(NotImplementedError())
    monkeypatch.setitem(sys.modules, "owner_trajectories_stub", stub)
    assert load_validator("owner_trajectories_stub") is None
    seen = []
    real = types.ModuleType("owner_trajectories_real")
    real.check_trajectory = lambda transcript, req: seen.append((transcript["today"], req["id"])) or []
    monkeypatch.setitem(sys.modules, "owner_trajectories_real", real)
    validator = load_validator("owner_trajectories_real", today="2026-10-10")
    assert isinstance(validator, TrajectoryValidator) and validator.available
    assert seen == [("2026-10-10", "probe")]  # probed once on a one-step transcript
    bare = types.ModuleType("owner_trajectories_bare")  # the module is there but the function is not
    monkeypatch.setitem(sys.modules, "owner_trajectories_bare", bare)
    with pytest.raises(HarnessError, match=r"owner_trajectories_bare has no check_trajectory\(transcript, "
                                           r"request\)"):
        load_validator("owner_trajectories_bare")


# ---- the realized mix -------------------------------------------------------------------------------------


def transcript(status: str, messages: list[dict]) -> Transcript:
    return Transcript(request="r", today=TODAY, status=status, final_text=None, messages=messages, steps=[],
                      schema_hash=schemas.SCHEMA_HASH, prompt_version=PROMPT_VERSION, prompt_hash=PROMPT_HASH)


def tool_message(call_id: str, body: object) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "content": canonical_json(body)}


SYSTEM = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
ONE_CALL = [turn(SEARCH).to_api(), tool_message("c1", HITS)]
EMPTY = [turn(SEARCH).to_api(), tool_message("c1", {"error": schemas.EMPTY_RESULTS, "items": [],
                                                    "hint": "h"})]
TWO_CALLS = [turn(GET, call("get_paper", "c3", arxiv_id=ID2)).to_api(), tool_message("c2", {}),
             tool_message("c3", {})]
ASKED = [turn(call("ask_clarification", "c9", question="q")).to_api(),
         tool_message("c9", {"question": "q", "status": "asked"})]
ANSWER = [{"role": "assistant", "content": "done"}]


@pytest.mark.parametrize("status, messages, scenario", [
    (ANSWERED, SYSTEM + ANSWER, "no_call"),
    (ANSWERED, SYSTEM + ONE_CALL + ANSWER, "normal"),
    (ANSWERED, SYSTEM + EMPTY + ONE_CALL + ANSWER, "recover"),
    (ANSWERED, SYSTEM + ONE_CALL + [turn(SEARCH).to_api(), tool_message("c1", {"error": schemas.RATE_LIMITED,
                                                                               "items": []})] + ANSWER,
     "recover"),
    (ANSWERED, SYSTEM + ONE_CALL + [turn(GET).to_api(), tool_message("c2", {"error": schemas.UPSTREAM_ERROR})]
     + ANSWER, "recover"),
    (ANSWERED, SYSTEM + ONE_CALL + [turn(GET).to_api(), tool_message("c2", {"error": schemas.NOT_FOUND})]
     + ANSWER, "normal"),  # NOT_FOUND, NO_FULLTEXT and SKIPPED are not recover codes
    (ANSWERED, SYSTEM + ONE_CALL + TWO_CALLS + ANSWER, "parallel"),
    (CLARIFICATION, SYSTEM + ASKED, "clarify"),
    (CLARIFICATION, SYSTEM + EMPTY + ASKED, "clarify"),  # clarify before recover
    (CLARIFICATION, SYSTEM + TWO_CALLS + ASKED, "parallel"),  # parallel before clarify
    (BUDGET_STOP, SYSTEM + EMPTY, "recover"),
    (BUDGET_STOP, SYSTEM, "no_call"),
    ("step_limit", SYSTEM + ONE_CALL * 8, "normal"),
    (HARNESS_ERROR, SYSTEM + TWO_CALLS, None),
])
def test_the_realized_scenario_is_parallel_clarify_no_call_recover_then_normal_in_that_order(status, messages,
                                                                                            scenario):
    assert realized_scenario(transcript(status, messages)) == scenario


def test_the_realized_mix_counts_and_shares_leaving_harness_errors_out():
    mix = realized_mix(["normal", "normal", None, "recover", "parallel"])
    assert mix == {"trajectories": 4,
                   "counts": {"normal": 2, "no_call": 0, "clarify": 0, "recover": 1, "parallel": 1},
                   "shares": {"normal": 0.5, "no_call": 0.0, "clarify": 0.0, "recover": 0.25,
                              "parallel": 0.25}}
    assert realized_mix([]) == {"trajectories": 0, "counts": dict.fromkeys(SCENARIOS, 0),
                                "shares": dict.fromkeys(SCENARIOS, 0.0)}
    assert sum(PLANNED_MIX.values()) == pytest.approx(1.0) and set(PLANNED_MIX) == set(SCENARIOS)


# ---- the run ----------------------------------------------------------------------------------------------


THREE = [request("req-b", "no_call", "en", "test", text="Thanks, that is all."),
         request("req-a", "normal", "zh", "dev", source="hand"),
         request("req-c", "parallel", "en", "train", text=f"Compare {ID1} and {ID2}.")]


def three_turns() -> list:
    return [turn(content="You're welcome."), *normal_turns(),
            turn(GET, call("get_paper", "c3", arxiv_id=ID2)), turn(content="compared")]


def test_generate_writes_the_index_sorted_the_transcripts_and_the_manifest_byte_stable_across_runs(tmp_path):
    manifest_path = corpus_manifest(tmp_path)
    outs = [tmp_path / "run1", tmp_path / "run2"]
    lines = []
    manifests = [run(THREE, PricedModel(three_turns()), out, manifest_path, log=lines.append) for out in outs]
    assert (outs[0] / "index.jsonl").read_bytes() == (outs[1] / "index.jsonl").read_bytes()
    assert (outs[0] / "manifest.json").read_bytes() == (outs[1] / "manifest.json").read_bytes()
    rows = read_index(outs[0])
    assert [r["id"] for r in rows] == ["req-a", "req-b", "req-c"]  # sorted by id, not request order
    assert [r["status"] for r in rows] == [ANSWERED] * 3
    raw = (outs[0] / "index.jsonl").read_bytes()
    assert raw.split(b"\n")[0] == canonical_json(rows[0]).encode("utf-8") and b"\r" not in raw
    for row in rows:
        saved = outs[0] / row["transcript"]
        assert saved.is_file() and json.loads(saved.read_text(encoding="utf-8"))["request"] == row["text"]
        assert set(row) == ROW_KEYS
    assert rows[0]["extra"] == {"source": "hand"} and rows[1]["extra"] == {}
    m = manifests[0]
    assert m == json.loads((outs[0] / "manifest.json").read_text(encoding="utf-8"))
    assert m["config"] == {"requests": "requests.jsonl", "run": "t", "seed": 42, "placeholders": False,
                           "rewrite": False, "temperature": None, "model_temperature": 0.0, "notes": "store",
                           "seed_notes": False, "seed_scope": None}
    assert (m["today"], m["schema_hash"], m["prompt_version"], m["prompt_hash"]) == (
        TODAY, schemas.SCHEMA_HASH, PROMPT_VERSION, PROMPT_HASH)
    assert (m["rewrite_prompt_version"], m["rewrite_prompt_hash"]) == (REWRITE_PROMPT_VERSION,
                                                                       REWRITE_PROMPT_HASH)
    assert m["corpus_manifest_sha256"] == synth.sha256_of(manifest_path)
    assert (m["git_commit"], m["git_dirty"], m["validated"], m["requests"]) == ("unknown", None, False, 3)
    assert m["counts"] == {"by_scenario": {"no_call": {ANSWERED: 1}, "normal": {ANSWERED: 1},
                                           "parallel": {ANSWERED: 1}},
                           "by_split": {"dev": {ANSWERED: 1}, "test": {ANSWERED: 1}, "train": {ANSWERED: 1}},
                           "by_status": {ANSWERED: 3}}
    assert m["planned_mix"] == PLANNED_MIX
    assert m["realized_mix"]["counts"] == {"normal": 1, "no_call": 1, "clarify": 0, "recover": 0,
                                           "parallel": 1}
    assert m["cost_rmb"] == {"trajectories": pytest.approx(0.002 * 7), "rewrites": 0.0,
                             "total": pytest.approx(0.002 * 7)}
    assert m["calls"] == {"paid": 7, "cached": 0}
    assert [line.split(":")[0] for line in lines[:3]] == ["req-b (no_call/en/test)", "req-a (normal/zh/dev)",
                                                          "req-c (parallel/en/train)"]
    assert "calls=['get_paper', 'get_paper']" in lines[2] and "rewrite=off" in lines[2]


def test_generate_skips_rows_already_in_the_index_unless_redo(tmp_path):
    manifest_path = corpus_manifest(tmp_path)
    out = tmp_path / "run"
    run(THREE, PricedModel(three_turns()), out, manifest_path)
    before = (out / "index.jsonl").read_bytes()
    lines = []
    run(THREE, PricedModel([]), out, manifest_path, log=lines.append)  # a model asked anything would raise
    assert (out / "index.jsonl").read_bytes() == before
    assert lines == [f"{rid}: already in the index, skipped" for rid in ("req-b", "req-a", "req-c")]
    # redo replaces the rows, and the file stays sorted
    run(THREE[:2], PricedModel([turn(content="再见"), *normal_turns("重做")]), out, manifest_path, redo=True)
    rows = read_index(out)
    assert [r["id"] for r in rows] == ["req-a", "req-b", "req-c"]
    assert final_text(out, "req-b") == "再见" and final_text(out, "req-a") == "重做"
    # a folder whose manifest is gone is resumed by its index alone; a row made with the other placeholders
    # flag is then run again without redo (with the manifest in place that flag cannot change, see below)
    (out / "manifest.json").unlink()
    run(THREE[:1], PricedModel([turn(content="placeholder run")]), out, manifest_path, placeholders=True)
    rows = read_index(out)
    assert [(r["id"], r["placeholders"]) for r in rows] == [("req-a", False), ("req-b", True),
                                                            ("req-c", False)]
    assert final_text(out, "req-b") == "placeholder run"
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["config"]["placeholders"] is True


def test_generate_refuses_to_resume_a_run_made_with_other_settings_and_names_every_one(tmp_path):
    manifest_path = corpus_manifest(tmp_path)
    out = tmp_path / "run"
    run(THREE[:2], PricedModel([turn(content="You're welcome."), *normal_turns()]), out, manifest_path,
        seed=1)
    before = files_of(out)
    with pytest.raises(HarnessError, match="use another --run name") as caught:
        run(THREE, PricedModel([]), out, manifest_path, seed=2)  # a model asked anything would raise
    assert str(caught.value) == (f"{out / 'manifest.json'}: the run there was generated with other settings "
                                 "(seed 1 -> 2); a different setting is a different run: use another --run "
                                 "name (--redo does not override this)")
    assert files_of(out) == before  # nothing was rewritten
    # every differing setting is named, in a fixed order, and redo does not override the refusal
    other = {"requests": "requests.jsonl", "run": "t", "model": "deepseek-pro", "requests_sha256": "f" * 64}
    with pytest.raises(HarnessError) as caught:
        run(THREE, PricedModel([]), out, manifest_path, seed=2, placeholders=True, redo=True,
            rewriter=StubRewriter([]), today="2026-10-11", settings=other)
    assert "(seed 1 -> 2, today '2026-10-10' -> '2026-10-11', model None -> 'deepseek-pro', " \
           "placeholders False -> True, rewrite False -> True, temperature None -> 0.7, " \
           f"requests_sha256 None -> '{'f' * 64}', rewrite_model None -> 'deepseek-pro', " \
           "rewrite_thinking None -> 'auto')" in str(caught.value)
    assert files_of(out) == before
    # the same settings resume normally, and the filters may change: they only narrow a run
    lines = []
    run(THREE, PricedModel([turn(GET, call("get_paper", "c3", arxiv_id=ID2)), turn(content="compared")]), out,
        manifest_path, seed=1, log=lines.append)
    assert lines[:2] == [f"{rid}: already in the index, skipped" for rid in ("req-b", "req-a")]
    assert [r["id"] for r in read_index(out)] == ["req-a", "req-b", "req-c"]
    narrowed = {"requests": "requests.jsonl", "run": "t", "limit": 1, "scenario": "normal", "split": "dev"}
    run(THREE[1:2], PricedModel([]), out, manifest_path, seed=1, settings=narrowed)
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["config"]["limit"] == 1
    assert set(synth.RUN_SETTINGS) == {"seed", "today", "model", "model_temperature", "placeholders",
                                       "rewrite", "temperature", "rewrite_prompt_hash", "schema_hash",
                                       "prompt_hash", "requests_sha256", "notes", "seed_notes",
                                       "seed_scope", "rewrite_model", "thinking", "rewrite_thinking"}


class WarmModel(PricedModel):
    """PricedModel sampled at another temperature than PricedModel's 0.0: another run."""

    temperature = 0.7


def test_generate_refuses_to_resume_a_run_whose_model_sampled_at_another_temperature(tmp_path):
    manifest_path = corpus_manifest(tmp_path)
    out = tmp_path / "run"
    run(THREE[:1], PricedModel([turn(content="You're welcome.")]), out, manifest_path)
    recorded = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["config"]
    assert recorded["model_temperature"] == 0.0 and recorded["temperature"] is None
    before = files_of(out)
    with pytest.raises(HarnessError, match="use another --run name") as caught:
        run(THREE, WarmModel([]), out, manifest_path, redo=True)  # a model asked anything would raise
    assert "(model_temperature 0.0 -> 0.7)" in str(caught.value)
    assert files_of(out) == before  # nothing was rewritten, and redo did not override the refusal
    # a stand-in without a temperature of its own is recorded as None, which is another setting too
    with pytest.raises(HarnessError, match="model_temperature 0.0 -> None"):
        run(THREE, ScriptedModel([]), out, manifest_path)
    assert files_of(out) == before
    # the same temperature resumes the folder
    lines = []
    run(THREE[:1], PricedModel([]), out, manifest_path, log=lines.append)
    assert lines == ["req-b: already in the index, skipped"]


def test_an_exception_out_of_run_request_leaves_a_harness_error_row_without_a_transcript_and_the_run_goes_on(
        tmp_path):
    manifest_path = corpus_manifest(tmp_path)
    out = tmp_path / "run"
    model = PricedModel([turn(content="You're welcome."), RuntimeError("the API answered 502 three times"),
                         turn(GET, call("get_paper", "c3", arxiv_id=ID2)), turn(content="compared")])
    lines = []
    m = run(THREE, model, out, manifest_path, log=lines.append)
    rows = {r["id"]: r for r in read_index(out)}
    assert [(r["id"], r["status"]) for r in read_index(out)] == [("req-a", HARNESS_ERROR),
                                                                 ("req-b", ANSWERED), ("req-c", ANSWERED)]
    failed = rows["req-a"]
    assert set(failed) == ROW_KEYS == set(synth.INDEX_KEYS) and failed["transcript"] is None
    assert failed["error"] == "the API answered 502 three times"
    assert (failed["steps"], failed["paid_calls"], failed["cache_hits"], failed["cost_rmb"]) == (0, 0, 0, 0.0)
    assert (failed["rewritten"], failed["rewrite_fallback"], failed["rewrite_cost_rmb"]) == (None, False, 0.0)
    assert (failed["faults"], failed["verdict"], failed["placeholders"]) == ({}, None, False)
    assert (failed["family"], failed["extra"], failed["slots"]) == (
        "test/normal/zh", {"source": "hand"}, {"topic": "speculative decoding", "date_from": "2026-09-01"})
    assert not (out / "transcripts" / "req-a.json").exists() and model.turns == []  # the run went on
    assert lines[1].startswith("req-a: ") and "RuntimeError: the API answered 502 three times" in lines[1]
    assert m["counts"]["by_status"] == {ANSWERED: 2, HARNESS_ERROR: 1}
    assert m["counts"]["by_scenario"]["normal"] == {HARNESS_ERROR: 1}
    assert m["realized_mix"]["trajectories"] == 2 and m["cost_rmb"]["total"] == pytest.approx(0.006)
    assert m == json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    # the next run resumes past the row like any other harness_error row; redo runs it again
    lines = []
    run(THREE, PricedModel([]), out, manifest_path, log=lines.append)
    assert lines == [f"{rid}: already in the index, skipped" for rid in ("req-b", "req-a", "req-c")]
    run(THREE[1:2], PricedModel(normal_turns()), out, manifest_path, redo=True)
    assert [(r["id"], r["status"]) for r in read_index(out)] == [("req-a", ANSWERED), ("req-b", ANSWERED),
                                                                 ("req-c", ANSWERED)]


def test_a_redo_whose_rerun_raises_discards_the_stale_transcript_and_only_that_file(tmp_path):
    manifest_path = corpus_manifest(tmp_path)
    out = tmp_path / "run"
    run(THREE[:2], PricedModel([turn(content="You're welcome."), *normal_turns()]), out, manifest_path)
    stale, kept = out / "transcripts" / "req-a.json", out / "transcripts" / "req-b.json"
    assert stale.is_file() and kept.is_file()
    untouched = kept.read_bytes()
    failing = PricedModel([RuntimeError("the API answered 502 three times")])
    lines = []
    m = run(THREE[1:2], failing, out, manifest_path, redo=True, log=lines.append)
    rows = {r["id"]: r for r in read_index(out)}
    assert (rows["req-a"]["status"], rows["req-a"]["transcript"]) == (HARNESS_ERROR, None)
    assert rows["req-a"]["error"] == "the API answered 502 three times"
    assert not stale.exists()  # the earlier run's transcript would misdescribe the error row
    assert kept.read_bytes() == untouched and rows["req-b"]["status"] == ANSWERED  # only that file went
    assert set(files_of(out)) == {"index.jsonl", "manifest.json", "transcripts/req-b.json"}
    assert m["counts"]["by_status"] == {ANSWERED: 1, HARNESS_ERROR: 1}
    assert m["realized_mix"]["trajectories"] == 1 and "RuntimeError" in lines[0]
    # the folder resumes like any other with a harness_error row, and a redo that answers writes it anew
    lines = []
    run(THREE[:2], PricedModel([]), out, manifest_path, log=lines.append)
    assert lines == [f"{rid}: already in the index, skipped" for rid in ("req-b", "req-a")]
    run(THREE[1:2], PricedModel(normal_turns("重做")), out, manifest_path, redo=True)
    assert final_text(out, "req-a") == "重做"
    assert read_index(out)[0]["transcript"] == "transcripts/req-a.json"


@pytest.mark.parametrize("damage, message", [
    (lambda row: canonical_json(row)[:-20], "line 2 is not JSON"),  # cut while it was being written
    (lambda row: canonical_json({k: v for k, v in row.items() if k != "status"}), "line 2 lacks status"),
    (lambda row: canonical_json({k: v for k, v in row.items() if k != "cost_rmb"}), "line 2 lacks cost_rmb"),
    (lambda row: canonical_json({**row, "placeholders": "no"}), "line 2: placeholders must be true or false"),
    (lambda row: canonical_json({**row, "status": None}), "line 2: status must be a string"),
    (lambda row: canonical_json({**row, "transcript": 3}), "line 2: transcript must be a string or null"),
    (lambda row: canonical_json({**row, "id": 7}), "line 2: id must be a non-empty string"),
    (lambda row: "[1, 2]", "line 2 is not an index row"),
])
def test_generate_refuses_a_resumed_index_row_that_is_not_one_naming_its_line(tmp_path, damage, message):
    manifest_path = corpus_manifest(tmp_path)
    out = tmp_path / "run"
    run(THREE[:2], PricedModel([turn(content="You're welcome."), *normal_turns()]), out, manifest_path)
    rows = read_index(out)
    write_jsonl(out / "index.jsonl", [canonical_json(rows[0]), damage(rows[1])])
    before = files_of(out)
    with pytest.raises(HarnessError) as caught:
        run(THREE[:2], PricedModel([]), out, manifest_path)
    assert str(caught.value).startswith(f"{out / 'index.jsonl'}: {message}")
    assert files_of(out) == before


def test_a_budget_stop_inside_the_loop_writes_its_row_and_stops_the_run_which_a_later_run_redoes(tmp_path):
    manifest_path = corpus_manifest(tmp_path)
    out = tmp_path / "run"
    lines = []
    model = PricedModel([turn(SEARCH), BudgetExceeded("stage M2: spent 90.0000 of 90.0 RMB"),
                         turn(content="never asked")])
    m = run([THREE[1], THREE[0]], model, out, manifest_path, log=lines.append)
    rows = read_index(out)
    assert [(r["id"], r["status"]) for r in rows] == [("req-a", BUDGET_STOP)]
    assert len(model.turns) == 1 and "the run stops" in lines[-1]
    assert m["counts"]["by_status"] == {BUDGET_STOP: 1} and m["realized_mix"]["counts"]["normal"] == 1
    # the next run redoes the cut turn without --redo, then goes on
    run([THREE[1], THREE[0]], PricedModel([*normal_turns(), turn(content="ok")]), out, manifest_path)
    assert [(r["id"], r["status"]) for r in read_index(out)] == [("req-a", ANSWERED), ("req-b", ANSWERED)]


def test_a_budget_exceeded_from_the_rewriter_stops_before_the_request_and_leaves_nothing_for_it(tmp_path):
    manifest_path = corpus_manifest(tmp_path)
    out = tmp_path / "run"
    rewriter = StubRewriter([Rewrite("Thanks a lot, that is everything.", 0.001, False, False),
                             BudgetExceeded("stage M2: spent 90.0000 of 90.0 RMB")])
    model = PricedModel([turn(content="Bye"), turn(content="never asked")])
    lines = []
    m = run(THREE[:2], model, out, manifest_path, rewriter=rewriter, log=lines.append)
    rows = read_index(out)
    assert [r["id"] for r in rows] == ["req-b"]
    assert rows[0]["rewritten"] == "Thanks a lot, that is everything."
    assert rows[0]["rewrite_cost_rmb"] == 0.001 and len(model.turns) == 1
    assert not (out / "transcripts" / "req-a.json").exists() and "stops before this request" in lines[-1]
    assert m["cost_rmb"] == {"trajectories": pytest.approx(0.002), "rewrites": 0.001,
                             "total": pytest.approx(0.003)}
    assert m["config"]["rewrite"] is True and m["config"]["temperature"] == 0.7
    assert "rewrite=ok" in lines[0]


def test_a_harness_error_row_carries_the_error_text_and_stays_out_of_the_realized_mix(tmp_path):
    manifest_path = corpus_manifest(tmp_path)
    out = tmp_path / "run"
    missing = call("get_paper", "c2", arxiv_id="2609.09999")  # no cassette: a replay miss
    model = PricedModel([turn(content="fine"), turn(SEARCH), turn(missing), turn(content="never asked")])
    m = run(THREE[:2], model, out, manifest_path)
    rows = {r["id"]: r for r in read_index(out)}
    assert rows["req-a"]["status"] == HARNESS_ERROR and rows["req-a"]["error"].startswith("no cassette for ")
    assert rows["req-b"]["error"] is None and rows["req-a"]["steps"] == 2
    assert m["counts"]["by_status"] == {ANSWERED: 1, HARNESS_ERROR: 1}
    assert m["realized_mix"] == {"trajectories": 1, "counts": {"normal": 0, "no_call": 1, "clarify": 0,
                                                               "recover": 0, "parallel": 0},
                                 "shares": {"normal": 0.0, "no_call": 1.0, "clarify": 0.0, "recover": 0.0,
                                            "parallel": 0.0}}
    assert "cassette" not in json.dumps(json.loads((out / "transcripts" / "req-a.json").read_text(
        encoding="utf-8"))["messages"])


def test_generate_records_the_validator_and_a_recover_request_in_the_manifest(tmp_path):
    manifest_path = corpus_manifest(tmp_path)
    out = tmp_path / "run"
    validator = TrajectoryValidator(lambda transcript, req: [] if req["scenario"] == "recover" else ["x"])
    model = PricedModel([turn(SEARCH), turn(call("search_arxiv", "c2", query="speculative decoding")),
                         turn(GET), turn(content="done")])
    m = run([request("rec-1", "recover")], model, out, manifest_path, validator=validator)
    row = read_index(out)[0]
    assert row["verdict"] == {"accepted": True, "problems": []} and m["validated"] is True
    injected = {k: v for counts in row["faults"].values() for k, v in counts.items() if k != "none"}
    assert sum(injected.values()) == 1 and m["realized_mix"]["counts"]["recover"] == 1
    first = json.loads(json.loads((out / "transcripts" / "rec-1.json").read_text(encoding="utf-8"))
                       ["messages"][3]["content"])
    assert first["error"] in RECOVER_CODES  # the first tool request met the fault


def test_generate_keeps_each_requests_notes_under_the_run_folder_and_a_rerun_regenerates_them(tmp_path):
    manifest_path = corpus_manifest(tmp_path)
    outs = [tmp_path / "run1", tmp_path / "run2"]
    for out in outs:
        m = run(THREE, PricedModel(three_turns()), out, manifest_path)
        assert m["config"]["notes"] == "store"
    note = f"notes/req-a/{ID1}-1.md"
    written = sorted(p for p in files_of(outs[0]) if p.startswith("notes/"))
    assert written == [note, "notes/req-a/notes.sqlite"]
    assert files_of(outs[0])[note] == files_of(outs[1])[note]  # the same note id, path and bytes
    saved = json.loads((outs[0] / "transcripts" / "req-a.json").read_text(encoding="utf-8"))
    assert json.loads(saved["messages"][7]["content"]) == {"note_id": f"{ID1}-1", "arxiv_id": ID1,
                                                           "path": f"{ID1}-1.md"}
    # a redo starts the request's store afresh: the note is -1 again, not -2
    run(THREE[1:2], PricedModel(normal_turns("重做")), outs[0], manifest_path, redo=True)
    assert sorted(p.name for p in (outs[0] / "notes" / "req-a").glob("*.md")) == [f"{ID1}-1.md"]
    # an error row takes its request's notes away with its transcript
    run(THREE[1:2], PricedModel([RuntimeError("the API answered 502 three times")]), outs[0], manifest_path,
        redo=True)
    assert not (outs[0] / "notes" / "req-a").exists()


def test_generate_with_stub_notes_writes_no_notes_and_the_setting_is_fixed_per_run(tmp_path):
    manifest_path = corpus_manifest(tmp_path)
    out = tmp_path / "run"
    m = run(THREE[1:2], PricedModel(normal_turns()), out, manifest_path, notes="stub")
    assert m["config"]["notes"] == "stub" and not (out / "notes").exists()
    saved = json.loads((out / "transcripts" / "req-a.json").read_text(encoding="utf-8"))
    assert json.loads(saved["messages"][7]["content"]) == {"note_id": f"{ID1}-note", "arxiv_id": ID1,
                                                           "path": f"notes/{ID1}.md"}
    before = files_of(out)
    with pytest.raises(HarnessError, match=r"\(notes 'stub' -> 'store'\)"):
        run(THREE, PricedModel([]), out, manifest_path, redo=True)
    assert files_of(out) == before
    with pytest.raises(ValueError, match="notes must be one of"):
        run(THREE, PricedModel([]), tmp_path / "other", manifest_path, notes="memory")


def test_generate_with_no_requests_still_leaves_a_manifest_and_an_empty_index(tmp_path):
    manifest_path = corpus_manifest(tmp_path)
    out = tmp_path / "run"
    m = run([], PricedModel([]), out, manifest_path)
    assert (out / "index.jsonl").read_bytes() == b"" and m["requests"] == 0
    assert m["realized_mix"]["trajectories"] == 0 and m["cost_rmb"]["total"] == 0.0


# ---- seed notes -------------------------------------------------------------------------------------------

# topic id -> (phrase, split, number of papers): four dev topics, two test topics, three train topics of which
# one has a single paper
SEED_TOPICS = {"speculative-decoding": ("speculative decoding", "dev", 3),
               "kv-cache": ("KV cache", "dev", 3), "quantization": ("quantization", "dev", 3),
               "rag": ("retrieval-augmented generation", "dev", 3),
               "reward-models": ("reward model", "test", 3),
               "distillation": ("knowledge distillation", "test", 3),
               "scaling-laws": ("scaling law", "train", 3), "tokenization": ("tokenization", "train", 3),
               "lonely": ("lonely topic", "train", 1)}


def seed_papers() -> dict[str, list[dict]]:
    """topic id -> its papers: titles and abstracts name the topic's phrase and no other topic's."""
    out = {}
    for i, (tid, (phrase, _, n)) in enumerate(SEED_TOPICS.items()):
        out[tid] = [{"arxiv_id": f"2609.{10 * i + j + 1:05d}", "title": f"A study of {phrase} number {j}",
                     "authors": ["A. Author"], "published": f"2026-0{8 + j % 2}-1{j}",
                     "abstract": f"We study {phrase} at scale. Results improve on benchmark {j}.",
                     "categories": ["cs.CL", "cs.LG"] if j % 2 else ["cs.CL"]} for j in range(n)]
    return out


@dataclass
class SeedWorld:
    corpus: Path
    requests: Path
    requests_manifest: Path
    index: CorpusIndex
    papers: dict[str, list[dict]]

    def split_papers(self, split: str) -> set[str]:
        return {p["arxiv_id"] for tid, (_, s, _) in SEED_TOPICS.items() if s == split
                for p in self.papers[tid]}

    def topic_of(self, arxiv_id: str) -> str:
        return next(tid for tid, papers in self.papers.items() if arxiv_id in {p["arxiv_id"] for p in papers})

    def paper(self, arxiv_id: str) -> dict:
        return next(p for papers in self.papers.values() for p in papers if p["arxiv_id"] == arxiv_id)


@pytest.fixture
def seed_world(tmp_path) -> SeedWorld:
    papers = seed_papers()
    corpus = tmp_path / "corpus" / "manifest.json"
    corpus.parent.mkdir(parents=True)
    corpus.write_text(json.dumps({"recorded_on": TODAY, "window": WINDOW, "topics": [
        {"id": tid, "name": tid, "phrases": [phrase], "papers": [p["arxiv_id"] for p in papers[tid]]}
        for tid, (phrase, _, _) in SEED_TOPICS.items()]}, indent=2) + "\n", encoding="utf-8")
    requests_path = write_jsonl(tmp_path / "synth" / "seed-requests.jsonl", [])
    manifest = requests_path.with_name("seed-requests-manifest.json")
    topics = {split: sorted(tid for tid, (_, s, _) in SEED_TOPICS.items() if s == split) for split in SPLITS}
    manifest.write_text(json.dumps({"topics": topics, "corpus_manifest_sha256": synth.sha256_of(corpus)}),
                        encoding="utf-8")
    return SeedWorld(corpus, requests_path, manifest,
                     CorpusIndex([p for group in papers.values() for p in group]), papers)


def seed_run(world: SeedWorld, requests: list[dict], model, out: Path, **options) -> dict:
    """generate over the seeding world, seed notes on (the default), the requests manifest next to the
    requests file unless options say otherwise."""
    return generate(requests, out=out, model=model, owner=options.pop("owner", None) or owner(),
                    fetcher=Pages(), index=world.index,
                    today=TODAY, settings={"requests": "seed-requests.jsonl", "run": "s"},
                    corpus_manifest=world.corpus, git_dir=None,
                    **{"log": lambda line: None, "requests_manifest": world.requests_manifest, **options})


def seeded(out: Path, rid: str) -> list[dict]:
    """The notes in request rid's store, oldest first by creation date, then by note id."""
    listed = notes.NoteStore(out / "notes" / rid).list(limit=50)
    return sorted(listed, key=lambda n: (n["created"], n["note_id"]))


def store_snapshot(out: Path, rid: str) -> tuple[list[dict], dict[str, bytes]]:
    """What a store holds, without the database file's bytes: its notes and its Markdown files."""
    folder = out / "notes" / rid
    return seeded(out, rid), {p.name: p.read_bytes() for p in sorted(folder.glob("*.md"))}


def bye() -> list:
    return [turn(content="好的。")]


def test_a_request_with_a_topic_gets_two_seeded_notes_of_its_topic_and_three_of_other_topics_from_its_split(
        seed_world, tmp_path):
    asks = [request("seed-a", split="dev"),  # the topic named by its phrase, as the dry-run file does
            request("seed-b", split="dev", slots={"topic": "KV cache", "topic_id": "kv-cache"})]  # as made
    out = tmp_path / "run1"
    seed_run(seed_world, asks, PricedModel(bye() * 2), out)
    for rid, topic in (("seed-a", "speculative-decoding"), ("seed-b", "kv-cache")):
        listed = seeded(out, rid)
        assert len(listed) == 5 and len({n["arxiv_id"] for n in listed}) == 5
        topics = [seed_world.topic_of(n["arxiv_id"]) for n in listed]
        assert topics.count(topic) == 2 and len(set(topics) - {topic}) == 3  # one per other topic
        assert {n["arxiv_id"] for n in listed} <= seed_world.split_papers("dev")
        for note in listed:
            paper = seed_world.paper(note["arxiv_id"])
            assert note["tags"] == [*paper["categories"], seed_world.topic_of(note["arxiv_id"])]
            assert paper["published"] <= note["created"] < TODAY
            assert note["note_id"] == f"{paper['arxiv_id']}-1" and note["title"] == paper["title"]
            expected = render_note(paper, {"mode": "scaffold"},
                                   pick_evidence(paper["abstract"], notes.EVIDENCE_SENTENCES, []),
                                   note["created"])
            assert (out / "notes" / rid / f"{note['note_id']}.md").read_text(encoding="utf-8") == expected
    # the same in a second run, and after a redo, which empties the store and seeds it again
    first = [store_snapshot(out, rid) for rid in ("seed-a", "seed-b")]
    seed_run(seed_world, asks, PricedModel(bye() * 2), tmp_path / "run2")
    assert [store_snapshot(tmp_path / "run2", rid) for rid in ("seed-a", "seed-b")] == first
    seed_run(seed_world, asks, PricedModel(bye() * 2), out, redo=True)
    assert [store_snapshot(out, rid) for rid in ("seed-a", "seed-b")] == first
    # the draw depends on the request id: another id draws other notes
    seed_run(seed_world, [request("seed-z", split="dev")], PricedModel(bye()), out)
    assert store_snapshot(out, "seed-z") != first[0]


def test_a_request_without_a_topic_gets_five_notes_from_its_split_and_a_thin_topic_takes_what_exists(
        seed_world, tmp_path):
    no_topic = {**request("seed-n", "no_call", "en", split="test"), "slots": {}}
    lonely = {**request("seed-l", split="train"), "slots": {"topic": "lonely topic", "topic_id": "lonely"}}
    out = tmp_path / "run"
    m = seed_run(seed_world, [no_topic, lonely], PricedModel(bye() * 2), out)
    listed = seeded(out, "seed-n")
    assert len(listed) == 5 and {n["arxiv_id"] for n in listed} <= seed_world.split_papers("test")
    assert {seed_world.topic_of(n["arxiv_id"]) for n in listed} == {"reward-models", "distillation"}
    thin = seeded(out, "seed-l")
    topics = [seed_world.topic_of(n["arxiv_id"]) for n in thin]
    assert len(thin) == 4 and topics.count("lonely") == 1  # one paper of its topic, three of the others
    assert {n["arxiv_id"] for n in thin} <= seed_world.split_papers("train")
    assert m["seeded_notes"]["per_request"] == {"seed-l": 4, "seed-n": 5}


def test_a_requests_file_without_a_manifest_draws_seed_notes_from_the_whole_corpus(seed_world, tmp_path):
    asks = [request(f"dry-{i:03d}", split="dev") for i in range(10)]
    out = tmp_path / "run"
    m = seed_run(seed_world, asks, PricedModel(bye() * 10), out, requests_manifest=None)
    drawn = {n["arxiv_id"] for i in range(10) for n in seeded(out, f"dry-{i:03d}")}
    assert drawn - seed_world.split_papers("dev")  # papers of other splits' topics as well
    for i in range(10):
        topics = [seed_world.topic_of(n["arxiv_id"]) for n in seeded(out, f"dry-{i:03d}")]
        assert len(topics) == 5 and topics.count("speculative-decoding") == 2
    assert (m["config"]["seed_notes"], m["config"]["seed_scope"]) == (True, "corpus")
    assert m["seeded_notes"]["split_restricted"] is False
    assert synth.requests_manifest_of(seed_world.requests) == seed_world.requests_manifest
    assert synth.requests_manifest_of(tmp_path / "dryrun-requests.jsonl") is None


def test_the_run_manifest_records_the_seed_setting_and_the_counts_and_the_setting_is_fixed_per_run(
        seed_world, tmp_path):
    out = tmp_path / "run"
    m = seed_run(seed_world, [request("seed-a", split="dev")], PricedModel(bye()), out)
    assert (m["config"]["seed_notes"], m["config"]["seed_scope"]) == (True, "split")
    assert m["seeded_notes"] == {"split_restricted": True, "per_request": {"seed-a": 5}}
    before = files_of(out)
    with pytest.raises(HarnessError, match=r"seed_notes True -> False, seed_scope 'split' -> None"):
        seed_run(seed_world, [request("seed-a", split="dev")], PricedModel([]), out, seed_notes=False)
    with pytest.raises(HarnessError, match=r"\(seed_scope 'split' -> 'corpus'\)"):
        seed_run(seed_world, [request("seed-a", split="dev")], PricedModel([]), out, requests_manifest=None)
    assert files_of(out) == before
    # a resumed run counts the rows it keeps as well; a topic outside the request's split has no papers
    # there, so that request gets only the three notes of other topics
    m = seed_run(seed_world, [request("seed-a", split="dev"), request("seed-b", split="test")],
                 PricedModel(bye()), out)
    assert m["seeded_notes"]["per_request"] == {"seed-a": 5, "seed-b": 3}
    assert {seed_world.topic_of(n["arxiv_id"]) for n in seeded(out, "seed-b")} <= {"reward-models",
                                                                                  "distillation"}


def test_without_seed_notes_the_store_is_empty_before_the_turn(seed_world, tmp_path):
    out = tmp_path / "run"
    model = PricedModel([turn(call("list_notes", "c1")), turn(content="还没有笔记。")])
    m = seed_run(seed_world, [request("seed-a", split="dev")], model, out, seed_notes=False)
    saved = json.loads((out / "transcripts" / "seed-a.json").read_text(encoding="utf-8"))
    assert json.loads(saved["messages"][3]["content"]) == []
    assert (m["config"]["seed_notes"], m["config"]["seed_scope"]) == (False, None)
    assert m["seeded_notes"] == {"split_restricted": None, "per_request": {"seed-a": 0}}
    # a turn that calls no note tool leaves no store at all
    seed_run(seed_world, [request("seed-b", split="dev")], PricedModel(bye()), out, seed_notes=False)
    assert not (out / "notes" / "seed-b").exists()


def test_search_notes_with_the_topics_term_finds_the_seeded_notes_in_a_scripted_turn(seed_world, tmp_path):
    out = tmp_path / "run"
    model = PricedModel([turn(call("search_notes", "c1", query="speculative decoding")),
                         turn(call("list_notes", "c2")), turn(content="找到两篇笔记。")])
    seed_run(seed_world, [request("seed-a", split="dev")], model, out)
    saved = json.loads((out / "transcripts" / "seed-a.json").read_text(encoding="utf-8"))
    hits = json.loads(saved["messages"][3]["content"])
    mine = {n["note_id"] for n in seeded(out, "seed-a")
            if seed_world.topic_of(n["arxiv_id"]) == "speculative-decoding"}
    assert {h["note_id"] for h in hits} == mine and len(mine) == 2
    assert all("speculative decoding" in h["title"] for h in hits)
    listed = json.loads(saved["messages"][5]["content"])
    assert len(listed) == 5 and [n["created"] for n in listed] == sorted((n["created"] for n in listed),
                                                                         reverse=True)


def test_a_write_note_on_a_paper_a_search_notes_hit_named_counts_as_seen_by_the_validator(seed_world,
                                                                                            tmp_path):
    from research_agent.trajectories import check_trajectory
    from research_agent.validation import check_call

    out = tmp_path / "run"
    hit = sorted(n["arxiv_id"] for n in seed_world.papers["speculative-decoding"])
    probe_model = PricedModel([turn(call("search_notes", "c1", query="speculative decoding")),
                               turn(content="ok")])
    seed_run(seed_world, [request("seed-a", split="dev")], probe_model, out)
    found = [h["arxiv_id"] for h in json.loads(json.loads(
        (out / "transcripts" / "seed-a.json").read_text(encoding="utf-8"))["messages"][3]["content"])]
    assert found and set(found) <= set(hit)
    model = PricedModel([turn(call("search_notes", "c1", query="speculative decoding")),
                         turn(call("write_note", "c2", arxiv_id=found[0], mode="scaffold")),
                         turn(content=f"已为 {found[0]} 补了一份笔记。")])
    seed_run(seed_world, [request("seed-a", split="dev")], model, out, redo=True,
             owner=OwnerFunctions(check_call, decide, parse_feed, lambda html, section, n: "x"))
    saved = json.loads((out / "transcripts" / "seed-a.json").read_text(encoding="utf-8"))
    assert json.loads(saved["messages"][5]["content"])["note_id"] == f"{found[0]}-2"  # the seeded one is -1
    problems = check_trajectory(saved, request("seed-a", split="dev"))
    assert not [p for p in problems if p.startswith(("calls", "ids", "execution"))]


def test_an_error_row_deletes_the_seeded_notes_of_its_request(seed_world, tmp_path):
    out = tmp_path / "run"
    seed_run(seed_world, [request("seed-a", split="dev")], PricedModel(bye()), out)
    assert len(seeded(out, "seed-a")) == 5
    m = seed_run(seed_world, [request("seed-a", split="dev")],
                 PricedModel([RuntimeError("the API answered 502 three times")]), out, redo=True)
    assert not (out / "notes" / "seed-a").exists()
    assert m["seeded_notes"]["per_request"] == {"seed-a": 0}


def test_a_requests_manifest_made_against_another_corpus_or_naming_an_unknown_topic_is_refused(seed_world,
                                                                                               tmp_path):
    data = json.loads(seed_world.requests_manifest.read_text(encoding="utf-8"))
    seed_world.requests_manifest.write_text(json.dumps({**data, "corpus_manifest_sha256": "0" * 64}),
                                            encoding="utf-8")
    with pytest.raises(HarnessError, match="another corpus manifest"):
        seed_run(seed_world, [request("seed-a", split="dev")], PricedModel([]), tmp_path / "run")
    seed_world.requests_manifest.write_text(json.dumps({"topics": {**data["topics"], "dev": ["nope"]}}),
                                            encoding="utf-8")
    with pytest.raises(HarnessError, match="unknown topic 'nope'"):
        seed_run(seed_world, [request("seed-a", split="dev")], PricedModel([]), tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.needs_data("data/synth")  # the registered requests' manifest
def test_the_repositorys_requests_manifest_gives_every_split_its_topics_and_their_papers():
    pool = synth.load_seed_pool(REPO / "data" / "corpus" / "manifest.json",
                                REPO / "data" / "synth" / "requests-manifest.json")
    recorded = json.loads((REPO / "data" / "synth" / "requests-manifest.json").read_text(encoding="utf-8"))
    assert pool.scope == "split"
    for split in SPLITS:
        assert list(pool.splits[split]) == recorded["topics"][split]
        assert sum(len(pool.topics[t]) for t in pool.splits[split]) == recorded["paper_ids"][split]
    every = synth.load_seed_pool(REPO / "data" / "corpus" / "manifest.json")
    assert every.scope == "corpus" and every.splits is None and len(every.topics) == 20
    assert synth.request_topic({"slots": {"topic": "KV cache"}}, every) == "kv-cache"
    assert synth.request_topic({"slots": {"topic": "llm-as-a-judge"}}, every) == "llm-as-judge"
    assert synth.request_topic({"slots": {"topic": "nothing like it"}}, every) is None


# ---- the script ------------------------------------------------------------------------------------------


def test_the_script_parser_defaults():
    args = script.build_parser().parse_args(["data/synth/dryrun-requests.jsonl"])
    assert (args.run, args.out, args.seed) == (None, "runs/m2/synth", 42)
    assert (args.limit, args.scenario, args.split) == (None, None, None)
    assert (args.placeholders, args.parse_atom, args.extract_section) == (False, None, None)
    assert (args.tokenizer, args.no_tokenizer, args.no_rewrite, args.redo) == ("models/Qwen3-1.7B", False,
                                                                               False, False)
    assert (args.max_cost, args.model, args.today, args.stub_notes) == (5.0, "deepseek-flash", None, False)
    assert args.no_seed_notes is False
    assert script.build_parser().parse_args(["r.jsonl", "--no-seed-notes"]).no_seed_notes is True
    assert (args.cassettes, args.llm_dir) == ("data/cassettes", "runs/llm")
    assert args.corpus_manifest == "data/corpus/manifest.json"
    assert script.build_parser().parse_args(["r.jsonl", "--today", "2026-10-11"]).today == "2026-10-11"
    with pytest.raises(SystemExit):
        script.build_parser().parse_args(["r.jsonl", "--today", "yesterday"])
    assert (args.stage, args.rewrite_model, args.thinking) == ("M2", None, "auto")
    assert (args.rewrite_thinking, args.off_peak_only) == ("auto", False)
    m7 = script.build_parser().parse_args(["r.jsonl", "--stage", "M7", "--rewrite-model", "deepseek-flash",
                                           "--thinking", "off", "--off-peak-only"])
    assert (m7.stage, m7.rewrite_model, m7.thinking) == ("M7", "deepseek-flash", "off")
    assert m7.off_peak_only is True
    for wrong in (["--stage", "M9"], ["--thinking", "sometimes"]):
        with pytest.raises(SystemExit):
            script.build_parser().parse_args(["r.jsonl", *wrong])


def test_the_placeholders_accept_every_call_decide_like_m1_smoke_and_strip_tags():
    assert script.placeholder_check_call("write_note", {}, [], TODAY) == []
    assert [script.placeholder_decide("empty_results", n) for n in (1, 2, 3)] == ["relax", "tell_user",
                                                                                  "tell_user"]
    assert all(script.placeholder_decide(e, 1) == "stop" for e in ("bad_json", "invalid_args", "repeat_call"))
    assert script.placeholder_extract_section("<p>Hello <b>wide</b>\n world</p>", "introduction", 1500) == \
        "Hello wide world"
    assert script.placeholder_extract_section("<p>Hello world</p>", "method", 5) == "Hello"
    assert script.placeholder_extract_section("<p></p>", "method", 100) is None


def test_resolve_owner_functions_takes_the_placeholders_or_the_tree_and_the_specs(tmp_path, monkeypatch):
    module = tmp_path / "owner_arxiv.py"
    module.write_text("import json\n\ndef parse_atom(xml):\n    return json.loads(xml) if xml[0] == '[' "
                      "else None\n\ndef extract_section(html, section, max_chars):\n    return 'S'\n",
                      encoding="utf-8")
    spec = f"{module}:parse_atom"
    stand_ins = script.resolve_owner_functions(placeholders=True, parse_atom_spec=spec,
                                               extract_section_spec=None)
    assert (stand_ins.check_call, stand_ins.decide, stand_ins.extract_section) == (
        script.placeholder_check_call, script.placeholder_decide, script.placeholder_extract_section)
    assert stand_ins.parse_atom("[]") == []
    real = script.resolve_owner_functions(placeholders=False, parse_atom_spec=spec,
                                          extract_section_spec=f"{module}:extract_section")
    from research_agent.policy import decide as owner_decide
    from research_agent.validation import check_call as owner_check_call

    assert (real.check_call, real.decide) == (owner_check_call, owner_decide)
    assert real.extract_section("<html/>", "method", 10) == "S"
    probe(real, TODAY)  # the tree's check_call and decide are implemented (2026-10-13), so the probe passes
    from research_agent import arxiv as tree_arxiv

    defaulted = script.resolve_owner_functions(placeholders=True, parse_atom_spec=None,
                                               extract_section_spec=None)
    assert defaulted.parse_atom is tree_arxiv.parse_atom  # the tree's arxiv module is the default since 10-13
    monkeypatch.delattr(tree_arxiv, "parse_atom")
    with pytest.raises(SystemExit, match="has no parse_atom; pass --parse-atom"):
        script.resolve_owner_functions(placeholders=True, parse_atom_spec=None, extract_section_spec=None)


def test_owner_function_and_load_function_point_at_the_flag_when_something_is_missing(tmp_path):
    with pytest.raises(SystemExit, match="no_such_module is not in the tree yet; pass --parse-atom"):
        script.owner_function("parse_atom", "--parse-atom", module="research_agent.no_such_module")
    with pytest.raises(SystemExit, match="has no extract_section; pass --extract-section"):
        script.owner_function("extract_section", "--extract-section", module="research_agent.canonical")
    assert script.owner_function("canonical_json", "--x", module="research_agent.canonical") is canonical_json
    module = tmp_path / "owner_parse.py"
    module.write_text("def parse_atom(xml):\n    return []\n", encoding="utf-8")
    assert script.load_function(f"{module}:parse_atom", "--parse-atom")("<feed/>") == []
    with pytest.raises(SystemExit, match="--parse-atom: .* has no function other"):
        script.load_function(f"{module}:other", "--parse-atom")
    with pytest.raises(SystemExit, match="--extract-section must be <path.py>:<function>"):
        script.load_function(str(module), "--extract-section")


class FakeResponse:
    def __init__(self, data: dict) -> None:
        self.data = data

    def model_dump(self, **_) -> dict:
        return self.data


class FakeOpenAI:
    """Stands in for openai.OpenAI: scripted chat completions, every payload recorded."""

    def __init__(self, script_: list) -> None:
        self.script, self.payloads = list(script_), []
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **payload) -> FakeResponse:
        self.payloads.append(payload)
        if not self.script:
            raise AssertionError("the fake API ran out of scripted answers")
        return FakeResponse(self.script.pop(0))


def completion(content: str = "", calls: list[tuple[str, dict]] = ()) -> dict:
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [{"id": f"call_{i}", "type": "function",
                                  "function": {"name": name, "arguments": json.dumps(args)}}
                                 for i, (name, args) in enumerate(calls)]
    return {"model": "deepseek-flash", "choices": [{"message": message}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 50}}


PEAK = datetime.fromisoformat("2026-10-19T06:43:00+00:00")  # 14:43 in Beijing: charged at the list price
OFF_PEAK = datetime.fromisoformat("2026-10-19T13:00:00+00:00")  # 21:00 in Beijing: half the price


def script_world(tmp_path: Path) -> list[str]:
    """Cassettes recorded through the real fetcher, a corpus manifest, a parse_atom module, two requests:
    the script's arguments with every output under tmp_path."""
    feeds = {paper_url(p["arxiv_id"]): feed_text([p]) for p in CORPUS}

    def arxiv(request_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=feeds[str(request_.url)])

    recorder = Fetcher("record", tmp_path / "cassettes", transport=httpx.MockTransport(arxiv),
                       sleep=lambda s: None)
    for url in feeds:
        recorder.get(url)
    recorder.close()
    manifest = corpus_manifest(tmp_path)
    module = tmp_path / "owner_arxiv.py"
    module.write_text("import json\n\ndef parse_atom(xml):\n    try:\n        return json.loads(xml)\n"
                      "    except json.JSONDecodeError:\n        return None\n\n"
                      "def extract_section(html, section, max_chars):\n    return html[:max_chars] or None\n",
                      encoding="utf-8")
    requests = write_jsonl(tmp_path / "e2e-requests.jsonl", [
        request("dry-a", "no_call", "en", text="Thanks, that is all."),
        request("dry-b", "normal", "zh", text="找 2026-09-01 之后 speculative decoding 的论文。")])
    return [str(requests), "--placeholders", "--parse-atom", f"{module}:parse_atom", "--no-tokenizer",
            "--out", str(tmp_path / "out"), "--llm-dir", str(tmp_path / "llm"), "--cassettes",
            str(tmp_path / "cassettes"), "--corpus-manifest", str(manifest), "--run", "e2e"]


def test_the_script_runs_end_to_end_offline_and_a_rerun_is_free_and_identical(tmp_path, capsys):
    argv = script_world(tmp_path)
    api = FakeOpenAI([
        completion("Thank you, that will be all."),  # the rewrite of dry-a
        completion("You're welcome."),
        completion("请找 2026-09-01 以后关于 speculative decoding 的论文。"),  # the rewrite of dry-b
        completion(calls=[("search_arxiv", {"query": "speculative decoding", "date_from": "2026-09-01"})]),
        completion(calls=[("get_paper", {"arxiv_id": ID1})]),
        completion(f"找到 {ID1}。"),
    ])
    assert script.main(argv, openai_client=api, clock=lambda: PEAK) == 0
    out = capsys.readouterr().out
    assert "run e2e: 2 of 2 requests, today 2026-10-10, index of 3 papers, M2 spent 0.0000 RMB" in out
    assert "validator on, rewrite on" in out and "done: 2 rows" in out  # trajectories.py exists since 10-13
    run_dir = tmp_path / "out" / "e2e"
    rows = read_index(run_dir)
    assert [(r["id"], r["status"], r["rewrite_fallback"]) for r in rows] == [("dry-a", ANSWERED, False),
                                                                             ("dry-b", ANSWERED, False)]
    assert rows[1]["rewritten"] == "请找 2026-09-01 以后关于 speculative decoding 的论文。"
    one_call = (1000 * 2.0 + 50 * 8.0) / 1e6  # the price list's upper bound for the fake usage
    assert rows[1]["paid_calls"] == 3 and rows[1]["cost_rmb"] == pytest.approx(3 * one_call)
    assert rows[1]["rewrite_cost_rmb"] == pytest.approx(one_call)
    assert all(r["placeholders"] for r in rows) and len(api.payloads) == 6
    assert "tools" not in api.payloads[0] and api.payloads[0]["temperature"] == 0.7  # the rewrite request
    assert api.payloads[1]["tools"] == schemas.MODEL_TOOLS and api.payloads[1]["temperature"] == 0.0
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["requests_sha256"] == synth.sha256_of(argv[0])
    assert manifest["config"]["run"] == "e2e" and manifest["config"]["tokenizer"] is None
    assert manifest["config"]["placeholders"] is True and manifest["validated"] is True
    assert all(isinstance(r["verdict"], dict) and set(r["verdict"]) == {"accepted", "problems"} for r in rows)
    assert manifest["realized_mix"]["counts"] == {"normal": 1, "no_call": 1, "clarify": 0, "recover": 0,
                                                  "parallel": 0}
    ledger = Budget(tmp_path / "llm" / "budget.sqlite", stage="M2", stage_cap_rmb=1.0)
    assert ledger.spent("M2") == pytest.approx(6 * (1000 * 2.0 + 50 * 8.0) / 1e6)
    assert (tmp_path / "llm" / "cache.sqlite").is_file()
    # a rerun skips both rows, so the fake API is never asked, and the files do not change
    before = (run_dir / "index.jsonl").read_bytes()
    assert script.main(argv, openai_client=FakeOpenAI([])) == 0
    assert (run_dir / "index.jsonl").read_bytes() == before
    # --redo answers from the cache: no paid call, the same transcripts apart from the money fields
    assert script.main([*argv, "--redo"], openai_client=FakeOpenAI([])) == 0
    redone = read_index(run_dir)
    assert [(r["paid_calls"], r["cache_hits"], r["cost_rmb"]) for r in redone] == [(0, 1, 0.0), (0, 3, 0.0)]
    assert [r["rewrite_cost_rmb"] for r in redone] == [0.0, 0.0]
    for row in redone:
        saved = json.loads((run_dir / row["transcript"]).read_text(encoding="utf-8"))
        assert saved["messages"] == json.loads(json.dumps(saved["messages"]))  # well-formed
    assert [r["rewritten"] for r in redone] == [r["rewritten"] for r in rows]
    assert f"M2 spent {6 * one_call:.4f} RMB" in capsys.readouterr().out  # the ledger carries over


def test_the_script_passes_each_steps_reasoning_back_to_an_api_that_insists_and_a_redo_is_free(tmp_path):
    """DeepSeek in thinking mode refuses a tools request that does not pass back the reasoning of every
    earlier step (support.strict_deepseek); the generator's earlier steps are its own, so it always can."""
    argv = script_world(tmp_path)
    search = ("call_00_s", "search_arxiv", {"query": "speculative decoding", "date_from": "2026-09-01"})
    server = StrictDeepSeek([
        Answer("Thank you, that will be all."),  # the rewrite of dry-a
        Answer("You're welcome."),
        Answer("请找 2026-09-01 以后关于 speculative decoding 的论文。"),  # the rewrite of dry-b
        Answer(calls=(search,), reasoning="先检索。"),
        Answer(calls=(("call_00_g", "get_paper", {"arxiv_id": ID1}),), reasoning="读第一篇。"),
        Answer(f"找到 {ID1}。", reasoning="可以回答了。"),
    ])
    assert script.main(argv, openai_client=server.sdk()) == 0
    run_dir = tmp_path / "out" / "e2e"
    rows = read_index(run_dir)
    assert [(r["id"], r["status"], r["paid_calls"]) for r in rows] == [("dry-a", ANSWERED, 1),
                                                                      ("dry-b", ANSWERED, 3)]
    assert server.refused == 0 and not any("thinking" in body for body in server.bodies)
    sent = [sent_reasoning(body) for body in server.bodies]
    assert sent == [[], [], [], [], ["先检索。"], ["先检索。", "读第一篇。"]]
    for row in rows:  # the reasoning went out on the wire only
        assert "reasoning_content" not in (run_dir / row["transcript"]).read_text(encoding="utf-8")
    # --redo answers every request from the cache, the steps after the first under their plain keys too
    again = StrictDeepSeek()
    assert script.main([*argv, "--redo"], openai_client=again.sdk()) == 0
    assert again.bodies == []
    redone = read_index(run_dir)
    assert [(r["paid_calls"], r["cache_hits"], r["status"]) for r in redone] == [(0, 1, ANSWERED),
                                                                                (0, 3, ANSWERED)]


def test_the_script_stops_before_anything_is_paid_when_an_owner_function_is_missing(tmp_path, capsys,
                                                                                     monkeypatch):
    argv = [a for a in script_world(tmp_path) if a != "--placeholders"]
    monkeypatch.delattr("research_agent.arxiv.extract_section", raising=False)  # as the tree was before 10-13
    with pytest.raises(SystemExit, match="has no extract_section; pass --extract-section"):
        script.main(argv, openai_client=FakeOpenAI([]))  # a tree without extract_section stops at once
    spec = argv[argv.index("--parse-atom") + 1].replace("parse_atom", "extract_section")
    argv += ["--extract-section", spec]

    def unfinished_check(tool, args, seen, today):
        raise NotImplementedError

    monkeypatch.setattr("research_agent.validation.check_call", unfinished_check)  # the stub of before 10-13
    with pytest.raises(SystemExit, match="check_call is not implemented yet.*--placeholders"):
        script.main(argv, openai_client=FakeOpenAI([]))  # the stubbed check_call is probed before any payment
    assert not (tmp_path / "out").exists() and not (tmp_path / "llm").exists()


def test_the_script_refuses_another_setting_on_the_same_run_name_but_not_another_filter(tmp_path):
    argv = script_world(tmp_path)
    api = FakeOpenAI([completion("Thank you, that will be all."), completion("You're welcome.")])
    assert script.main([*argv, "--limit", "1"], openai_client=api) == 0
    run_dir = tmp_path / "out" / "e2e"
    before = files_of(run_dir)
    with pytest.raises(SystemExit, match=r"seed 42 -> 7\); a different setting is a different run: use "
                                         r"another --run name"):
        script.main([*argv, "--limit", "1", "--seed", "7"], openai_client=FakeOpenAI([]))
    with pytest.raises(SystemExit, match="rewrite True -> False, temperature 0.7 -> None"):
        script.main([*argv, "--limit", "1", "--no-rewrite", "--redo"], openai_client=FakeOpenAI([]))
    with pytest.raises(SystemExit, match="today '2026-10-10' -> '2026-10-11'"):
        script.main([*argv, "--limit", "1", "--today", "2026-10-11"], openai_client=FakeOpenAI([]))
    assert files_of(run_dir) == before  # a refused call writes nothing
    # the filters only narrow a run: the same run name goes on with another limit
    api = FakeOpenAI([completion("请找 2026-09-01 以后关于 speculative decoding 的论文。"),
                      completion("没有找到。")])  # the rewrite of dry-b, then an answer without a call
    assert script.main([*argv, "--limit", "2"], openai_client=api) == 0
    assert [(r["id"], r["status"]) for r in read_index(run_dir)] == [("dry-a", ANSWERED), ("dry-b", ANSWERED)]


NOTE_RUN = [completion("请找 2026-09-01 以后关于 speculative decoding 的论文。"),  # the rewrite of dry-b
            completion(calls=[("write_note", {"arxiv_id": ID1, "mode": "scaffold"})]),
            completion("笔记已写好。")]


def test_the_script_writes_real_notes_by_default_and_stub_notes_keeps_the_stand_ins(tmp_path):
    argv = [*script_world(tmp_path), "--scenario", "normal"]
    assert script.main(argv, openai_client=FakeOpenAI(NOTE_RUN)) == 0
    run_dir = tmp_path / "out" / "e2e"
    saved = json.loads((run_dir / "transcripts" / "dry-b.json").read_text(encoding="utf-8"))
    written = json.loads(saved["messages"][3]["content"])
    note = run_dir / "notes" / "dry-b" / written["path"]
    assert note.read_text(encoding="utf-8").startswith(f"### {TODAY} {CORPUS[0]['title']}\n")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["notes"] == "store"
    # seed notes by default: the file has no manifest next to it, so they come from the whole corpus (two
    # papers of the request's topic, which is the corpus's only topic, and no other topic to draw from)
    assert (manifest["config"]["seed_notes"], manifest["config"]["seed_scope"]) == (True, "corpus")
    assert manifest["seeded_notes"] == {"split_restricted": False, "per_request": {"dry-b": 2}}
    assert len(list((run_dir / "notes" / "dry-b").glob("*.md"))) == 3
    bare_argv = [*argv, "--run", "e2e-bare", "--no-seed-notes"]
    assert script.main(bare_argv, openai_client=FakeOpenAI([completion("笔记已写好。")])) == 0
    bare = json.loads((tmp_path / "out" / "e2e-bare" / "manifest.json").read_text(encoding="utf-8"))
    assert (bare["config"]["seed_notes"], bare["config"]["seed_scope"]) == (False, None)
    bare_notes = tmp_path / "out" / "e2e-bare" / "notes" / "dry-b"
    assert [p.name for p in bare_notes.glob("*.md")] == [f"{ID1}-1.md"]
    stub_argv = [*argv, "--run", "e2e-stub", "--stub-notes"]
    # the rewrite and the first turn come from the cache; the turn after write_note sees another record
    api = FakeOpenAI([completion("笔记已写好。")])
    assert script.main(stub_argv, openai_client=api) == 0 and api.script == []
    stub_dir = tmp_path / "out" / "e2e-stub"
    assert [r["status"] for r in read_index(stub_dir)] == [ANSWERED]
    assert not (stub_dir / "notes").exists()
    saved = json.loads((stub_dir / "transcripts" / "dry-b.json").read_text(encoding="utf-8"))
    assert json.loads(saved["messages"][3]["content"])["path"] == f"notes/{ID1}.md"
    stub_config = json.loads((stub_dir / "manifest.json").read_text(encoding="utf-8"))["config"]
    assert (stub_config["notes"], stub_config["seed_notes"]) == ("stub", False)  # stand-ins hold no notes


def test_the_script_refuses_a_resume_whose_notes_setting_differs(tmp_path):
    argv = [*script_world(tmp_path), "--limit", "1"]
    api = FakeOpenAI([completion("Thank you, that will be all."), completion("You're welcome.")])
    assert script.main([*argv, "--stub-notes"], openai_client=api) == 0
    run_dir = tmp_path / "out" / "e2e"
    before = files_of(run_dir)
    with pytest.raises(SystemExit, match=r"\(notes 'stub' -> 'store', seed_notes False -> True, seed_scope "
                                         r"None -> 'corpus'\); a different setting is a different run"):
        script.main(argv, openai_client=FakeOpenAI([]))
    assert files_of(run_dir) == before


def test_the_script_turns_a_validator_module_without_check_trajectory_into_a_message(tmp_path, monkeypatch):
    def bare_module() -> None:
        raise HarnessError("research_agent.trajectories has no check_trajectory(transcript, request)")

    monkeypatch.setattr(script, "load_validator", bare_module)
    with pytest.raises(SystemExit, match="has no check_trajectory"):
        script.main(script_world(tmp_path), openai_client=FakeOpenAI([]))
    assert not (tmp_path / "out").exists()  # nothing was paid or written


def test_generate_trajectories_carries_no_machine_path():
    source = (REPO / "scripts" / "generate_trajectories.py").read_text(encoding="utf-8")
    assert "Users" not in source and "sk-" not in source


def test_read_index_keeps_a_row_whose_text_holds_a_line_separator_on_one_line(tmp_path):
    row = {key: None for key in synth.INDEX_KEYS}
    row.update(id="r1", placeholders=False, status="ok", text="line one line two")
    path = tmp_path / "index.jsonl"
    path.write_text(canonical_json(row) + "\n", encoding="utf-8")
    assert synth._read_index(path) == [row]


# ---- fault injection for the evaluation (M4): a FaultSetting and a FaultTrace -----------------------------


def test_a_fault_setting_gives_both_injectors_all_their_faults_no_budget_and_its_sleep():
    """scripts/run_e2e.py --fault-rate: whatever the scenario, the fetch injector may inject all seven faults
    and the search all four of its own, neither under a budget, persistent as asked, sleeping for real by
    default (DELAY_S) and with the setting's sleep when one is given."""
    slept: list[float] = []
    for scenario in ("normal", "recover"):
        tools, injectors = tools_for(scenario, seed=3, faults=FaultSetting(rate=0.5, persistent=True,
                                                                          sleep=slept.append))
        fetch, search = injectors["fetch"], injectors["search"]
        assert fetch.faults == FAULTS and search.faults == SEARCH_FAULTS
        assert fetch.budget is None and search.budget is None
        assert fetch.rate == search.rate == 0.5 and fetch.persistent and search.persistent
        assert (fetch.seed, fetch.task, search.seed, search.task) == (3, "r1", 3, "r1")
        assert fetch.sleep == slept.append
        assert tools["search_arxiv"] is search  # no trace asked for: the injector itself is the tool
    default = FaultSetting(rate=0.2)
    assert (default.persistent, default.faults, default.sleep) == (False, FAULTS, time.sleep)
    with pytest.raises(ValueError, match="rate"):
        tools_for("normal", faults=FaultSetting(rate=1.5))
    narrow = tools_for("normal", faults=FaultSetting(rate=0.5, faults=("delay", "timeout")))[1]
    assert narrow["fetch"].faults == ("delay", "timeout") and narrow["search"].faults == ("timeout",)


def test_a_fault_trace_records_every_arxiv_tool_call_its_wall_time_and_the_faults_it_met():
    now = [0.0]

    def clock() -> float:
        now[0] += 0.25
        return now[0]

    trace = FaultTrace(clock=clock)
    tools, injectors = tools_for("normal", seed=3, faults=FaultSetting(rate=1.0, sleep=lambda s: None),
                                 fault_trace=trace)
    tools["search_arxiv"]({"query": "speculative decoding"})
    tools["get_paper"]({"arxiv_id": ID1})
    tools["fetch_fulltext"]({"arxiv_id": ID2})
    tools["search_notes"]({"query": "x"})  # not an arXiv tool: not traced
    assert [c["tool"] for c in trace.calls] == ["search_arxiv", "get_paper", "fetch_fulltext"]
    assert [c["wall_ms"] for c in trace.calls] == [250.0, 250.0, 250.0]
    fetch, search = injectors["fetch"].log, injectors["search"].log
    assert trace.calls[0]["faults"] == [{"key": search_key({"query": "speculative decoding"}),
                                         "kind": search[0]["fault"], "ordinal": 0}]
    assert trace.calls[1]["faults"] == [{"key": canonical_url(paper_url(ID1)), "kind": fetch[0]["fault"],
                                         "ordinal": 0}]
    # an html_404 sends fetch_fulltext on to the abstract: both of its requests are the call's
    assert trace.calls[2]["faults"] == [{"key": e["url"], "kind": e["fault"], "ordinal": e["ordinal"]}
                                        for e in fetch[1:]]
    assert all(c["faults"] for c in trace.calls)  # rate 1: every request met a fault


def test_a_fault_trace_keeps_a_call_whose_tool_raised_and_leaves_requests_without_a_fault_out():
    trace = FaultTrace()
    tools, _ = tools_for("normal", faults=FaultSetting(rate=0.0), fault_trace=trace)
    assert tools["get_paper"]({"arxiv_id": ID1}).error is None
    with pytest.raises(ReplayMiss):
        tools["get_paper"]({"arxiv_id": "2601.99999"})
    assert [(c["tool"], c["faults"]) for c in trace.calls] == [("get_paper", []), ("get_paper", [])]


def test_run_request_passes_a_fault_setting_and_a_trace_to_the_tools(tmp_path):
    trace = FaultTrace()
    model = PricedModel([turn(GET), turn(content="2609.00001 讲的是投机解码。")])
    outcome = run_request(request("req-f"), model=model, owner=owner(), fetcher=Pages(),
                          index=CorpusIndex(CORPUS), today=TODAY, seed=3,
                          faults=FaultSetting(rate=1.0, sleep=lambda s: None), fault_trace=trace)
    [traced] = trace.calls
    assert traced["tool"] == "get_paper" and traced["faults"]
    assert sum(n for fault, n in outcome.faults["fetch"].items() if fault != "none") == len(traced["faults"])
    plain = run_request(request("req-f"), model=PricedModel([turn(GET), turn(content="好的。")]),
                        owner=owner(), fetcher=Pages(), index=CorpusIndex(CORPUS), today=TODAY, seed=3)
    assert plain.faults == {"fetch": {"none": 1}, "search": {}}  # a normal request without a setting


# ---- M7: the stage, the rewriter's own model, thinking off, off-peak only (docs/m7-计划.md WP1) ----------


def test_an_m7_run_books_the_teacher_and_the_rewriter_apart_without_thinking_at_off_peak_prices(tmp_path):
    argv = [*script_world(tmp_path), "--stage", "M7", "--model", "deepseek-v4-pro", "--rewrite-model",
            "deepseek-flash", "--thinking", "off", "--off-peak-only"]
    api = FakeOpenAI([
        completion("Thank you, that will be all."),  # the rewrite of dry-a, by flash
        completion("You're welcome."),
        completion("请找 2026-09-01 以后关于 speculative decoding 的论文。"),  # the rewrite of dry-b
        completion(calls=[("search_arxiv", {"query": "speculative decoding", "date_from": "2026-09-01"})]),
        completion(calls=[("get_paper", {"arxiv_id": ID1})]),
        completion(f"找到 {ID1}。"),
    ])
    assert script.main(argv, openai_client=api, clock=lambda: OFF_PEAK) == 0
    assert [p["model"] for p in api.payloads] == ["deepseek-flash", "deepseek-v4-pro", "deepseek-flash",
                                                  "deepseek-v4-pro", "deepseek-v4-pro", "deepseek-v4-pro"]
    assert all(p.get("extra_body") == THINKING_OFF for p in api.payloads if p["model"] == "deepseek-v4-pro")
    assert all("extra_body" not in p for p in api.payloads if p["model"] == "deepseek-flash")
    manifest = json.loads((tmp_path / "out" / "e2e" / "manifest.json").read_text(encoding="utf-8"))
    config = manifest["config"]
    assert (config["stage"], config["rewrite_model"], config["thinking"]) == ("M7", "deepseek-flash", "off")
    ledger = Budget(tmp_path / "llm" / "budget.sqlite", stage="M7", stage_cap_rmb=1.0)
    flash_call, pro_call = (1000 * 2.0 + 50 * 8.0) / 1e6, (1000 * 9.0 + 50 * 27.0) / 1e6
    assert ledger.spent("M7") == pytest.approx(0.5 * (2 * flash_call + 4 * pro_call))
    assert ledger.spent("M2") == 0.0
    # another rewrite model or thinking setting is another run
    with pytest.raises(SystemExit, match="other settings"):
        script.main([*argv[:-4], "deepseek-v4-pro", "--thinking", "off", "--off-peak-only"],
                    openai_client=FakeOpenAI([]), clock=lambda: OFF_PEAK)


def test_off_peak_only_stops_a_run_in_the_peak_before_any_paid_call(tmp_path, capsys):
    argv = [*script_world(tmp_path), "--off-peak-only"]
    api = FakeOpenAI([])
    assert script.main(argv, openai_client=api, clock=lambda: PEAK) == 0
    assert api.payloads == [] and read_index(tmp_path / "out" / "e2e") == []
    assert "peak" in capsys.readouterr().out

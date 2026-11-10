"""The replay search over the recorded corpus (research_agent.tools.corpus_search) and its check script,
offline: a hand-made corpus of eight papers, a fake fetcher that only reads, and temporary manifests."""

import copy
import dataclasses
import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

from research_agent.agent.types import HarnessError, ToolOutcome
from research_agent.net.fetch import Fetched, Fetcher, canonical_url
from research_agent.tools import schemas
from research_agent.tools.adapters import hit, paper_url
from research_agent.tools.corpus_search import (
    MANIFEST,
    CorpusIndex,
    Query,
    corpus_ids,
    load_papers,
    parse_query,
    phrase_tokens,
    tokenize,
)

REPO = Path(__file__).resolve().parents[2]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = load_script("check_corpus_search")

WINDOW = ["2026-07-01", "2026-10-10"]
NOTHING = Query((), (), (), (), ())  # a text that yields no term: no alternative either


def one_alternative(terms: tuple = (), phrases: tuple = (), excluded: tuple = (),
                    categories: tuple = ()) -> Query:
    """The Query of a text without OR that yields a term: one alternative holding all its phrases."""
    return Query(terms, phrases, excluded, categories, (phrases,))


def paper(arxiv_id: str, title: str, abstract: str, published: str, categories: list[str]) -> dict:
    return {"arxiv_id": arxiv_id, "title": title, "authors": ["A. Author", "B. Author"],
            "published": published, "abstract": abstract, "categories": categories}


CORPUS = [
    paper("2609.00001", "Adaptive Draft Length for Speculative Decoding",
          "Speculative decoding drafts tokens with a small model; the target model accepts them after "
          "verification. We adapt the draft length to the acceptance rate.", "2026-09-15",
          ["cs.CL", "cs.LG"]),
    paper("2609.00002", "Tree Verification at Scale",
          "Tree-structured checking of drafted tokens in large serving systems, with speculative decoding "
          "as the baseline.", "2026-09-20", ["cs.LG"]),
    paper("2608.00003", "Mixture-of-Experts Routing with Load Balance",
          "A router sends tokens to experts; we balance the load of a mixture-of-experts model without an "
          "auxiliary loss.", "2026-08-05", ["cs.LG", "cs.AI"]),
    paper("2607.00004", "Quantization of Large Language Models to 4-bit",
          "Post-training quantization keeps accuracy at four bits per weight; outlier channels are kept in "
          "higher precision.", "2026-07-10", ["cs.CL"]),
    paper("2609.00005", "KV Cache Compression for Long Context",
          "The key-value cache of long prompts is compressed by evicting low-attention entries.",
          "2026-09-01", ["cs.CL"]),
    paper("2608.00006", "A Survey of Speculative Decoding Methods",
          "This survey reviews speculative decoding methods, their draft models and their verification "
          "schemes.", "2026-08-20", ["cs.CL"]),
    paper("2610.00007", "Rotary Position Embeddings Beyond Training Length",
          "Length extrapolation of rotary position embeddings to contexts longer than those seen in "
          "training.", "2026-10-02", ["cs.CL", "cs.LG"]),
    paper("2607.00008", "Reward Models for Preference Optimization",
          "Reward models trained on preference pairs guide direct preference optimization of a policy.",
          "2026-07-25", ["cs.LG", "stat.ML"]),
]
BY_ID = {p["arxiv_id"]: p for p in CORPUS}
SPECULATIVE = {"2609.00001", "2609.00002", "2608.00006"}  # the papers that mention speculative decoding
EVERY_PAPER = {"query": "decoding verification experts quantization cache rotary reward"}  # one term each


@pytest.fixture(scope="module")
def index() -> CorpusIndex:
    return CorpusIndex(CORPUS)


def ids_of(papers: list[dict]) -> list[str]:
    return [p["arxiv_id"] for p in papers]


def test_the_index_looks_a_paper_up_by_id_handing_out_a_copy(index):
    found = index.paper("2608.00003")
    assert found == BY_ID["2608.00003"] and set(found) == set(schemas.PAPER_KEYS)
    found["categories"].append("changed")
    assert index.paper("2608.00003")["categories"] == ["cs.LG", "cs.AI"]  # a copy, not the index's own
    assert index.paper("2699.99999") is None and index.paper("2608.00003v2") is None


def id_set(papers: list[dict]) -> set[str]:
    return set(ids_of(papers))


# tokenization


@pytest.mark.parametrize("text, tokens", [
    ("Speculative Decoding", ["speculative", "decoding"]),
    ("Mixture-of-Experts routing", ["mixture-of-experts", "mixture", "of", "experts", "routing"]),
    ("state-of-the-art", ["state-of-the-art", "state", "of", "the", "art"]),
    ("don't stop", ["don't", "don", "t", "stop"]),
    ("4-bit (LLM)!", ["4-bit", "4", "bit", "llm"]),
    ("ＢＥＲＴ ﬁne-tuning", ["bert", "fine-tuning", "fine", "tuning"]),  # NFKC: full width, ligature
    ("--leading - trailing-", ["leading", "trailing"]),
    ("投机解码 混合专家", []),
    ("", []),
])
def test_tokens_are_lower_case_latin_words_with_hyphenated_ones_also_split_into_parts(text, tokens):
    assert tokenize(text) == tokens


@pytest.mark.parametrize("text, tokens", [
    ("Speculative Decoding", ("speculative", "decoding")),
    ("Mixture-of-Experts routing", ("mixture", "of", "experts", "routing")),
    ("scaling test-time compute", ("scaling", "test", "time", "compute")),
    ("don't stop", ("don", "t", "stop")),
    ("投机解码", ()),
])
def test_phrase_tokens_are_the_parts_only_so_both_writings_of_a_hyphenated_word_meet(text, tokens):
    assert phrase_tokens(text) == tokens


# parse_query


SPEC, TREE, DRAFT = ("speculative", "decoding"), ("tree", "verification"), ("draft", "length")


@pytest.mark.parametrize("text, query", [
    ("speculative decoding", one_alternative(("speculative", "decoding"))),
    ('"speculative decoding"', one_alternative(("speculative", "decoding"), (SPEC,))),
    ('ti:"KV cache" abs:compression all:long au:Smith',
     one_alternative(("kv", "cache", "compression", "long", "smith"), (("kv", "cache"),))),
    # a phrase is its parts only: the compound token still counts as a term
    ('"mixture-of-experts" routing', one_alternative(("mixture-of-experts", "mixture", "of", "experts",
                                                      "routing"), (("mixture", "of", "experts"),))),
    ('"mixture of experts"', one_alternative(("mixture", "of", "experts"), (("mixture", "of", "experts"),))),
    ("cat:cs.CL cat:cs.LG routing", one_alternative(("routing",), (), (), ("cs.CL", "cs.LG"))),
    ("(cat:cs.CL OR cat:cs.LG) AND routing", one_alternative(("routing",), (), (), ("cs.CL", "cs.LG"))),
    ("cat:CS.cl cat:Stat.ml routing", one_alternative(("routing",), (), (), ("cs.CL", "stat.ML"))),  # mapped
    ("cat:math.OC routing", one_alternative(("routing",), (), (), ("math.OC",))),  # unknown: kept as written
    ("cat:cs.CL", Query((), (), (), ("cs.CL",), ())),  # a filter alone is no alternative: it matches nothing
    # OR splits the query into alternatives, each holding its own phrases; phrases is their union
    ('"speculative decoding" OR "tree verification"',
     Query(("speculative", "decoding", "tree", "verification"), (SPEC, TREE), (), (), ((SPEC,), (TREE,)))),
    ('"draft length" OR draft', Query(("draft", "length"), (DRAFT,), (), (), ((DRAFT,), ()))),
    ('"draft length" "speculative decoding" OR "tree verification" AND "speculative decoding"',
     Query(("draft", "length", "speculative", "decoding", "tree", "verification"), (DRAFT, SPEC, TREE),
           (), (), ((DRAFT, SPEC), (TREE, SPEC)))),
    # parentheses are dropped before the split: ("A" OR "B") AND "C" reads as "A" OR ("B" AND "C")
    ('("draft length" OR "tree verification") AND "speculative decoding"',
     Query(("draft", "length", "tree", "verification", "speculative", "decoding"), (DRAFT, TREE, SPEC),
           (), (), ((DRAFT,), (TREE, SPEC)))),
    # NOT and cat: apply to the whole query, whichever alternative holds them
    ('"draft length" ANDNOT adaptive OR "tree verification" cat:cs.LG',
     Query(("draft", "length", "tree", "verification"), (DRAFT, TREE), ("adaptive",), ("cs.LG",),
           ((DRAFT,), (TREE,)))),
    # an alternative without a term cannot match a paper on its own and is dropped from the alternation
    ('"draft length" OR', one_alternative(("draft", "length"), (DRAFT,))),
    ('OR "draft length" OR OR', one_alternative(("draft", "length"), (DRAFT,))),
    ('"draft length" OR 投机解码', one_alternative(("draft", "length"), (DRAFT,))),
    ('"draft length" OR NOT adaptive', one_alternative(("draft", "length"), (DRAFT,), ("adaptive",))),
    ('"draft length" OR cat:cs.CL', one_alternative(("draft", "length"), (DRAFT,), (), ("cs.CL",))),
    ('ti:"draft length" OR abs:"draft length"', one_alternative(("draft", "length"), (DRAFT,))),  # distinct
    ("draft AND decoding OR tree", one_alternative(("draft", "decoding", "tree"))),
    # operators are upper-case only, as on arXiv: lower- and title-case ones are ordinary terms
    ("draft and decoding or tree", one_alternative(("draft", "and", "decoding", "or", "tree"))),
    ("Not All Tokens Are Equal", one_alternative(("not", "all", "tokens", "are", "equal"))),
    ("decoding not survey", one_alternative(("decoding", "not", "survey"))),
    ("decoding Andnot survey", one_alternative(("decoding", "andnot", "survey"))),
    ("and or not", one_alternative(("and", "or", "not"))),
    ("decoding ANDNOT survey", one_alternative(("decoding",), (), ("survey",))),
    ("decoding NOT survey", one_alternative(("decoding",), (), ("survey",))),
    ("decoding AND NOT survey", one_alternative(("decoding",), (), ("survey",))),
    ('decoding ANDNOT "tree verification"', one_alternative(("decoding",), (), ("tree", "verification"))),
    ("decoding NOT mixture-of-experts", one_alternative(("decoding",), (), ("mixture-of-experts",))),
    ("decoding NOT ti:survey", one_alternative(("decoding",), (), ("survey",))),
    # a cat: token after NOT or ANDNOT is dropped, not collected as a filter
    ("decoding NOT cat:cs.CL survey", one_alternative(("decoding", "survey"))),
    ("decoding ANDNOT cat:cs.CL", one_alternative(("decoding",))),
    ("decoding submittedDate:[202607010000 TO 202610102359]", one_alternative(("decoding",))),
    ("decoding AND submittedDate:[2026-07-01 TO 2026-10-10] cat:cs.CL",
     one_alternative(("decoding",), (), (), ("cs.CL",))),
    ("decoding submittedDate:2026-07-01 tree", one_alternative(("decoding", "tree"))),
    # an unclosed range drops only its bracketed token (TO stays a term); a bare prefix drops only itself
    ("submittedDate:[2026-07-01 TO decoding tree", one_alternative(("to", "decoding", "tree"))),
    ("submittedDate: decoding tree", one_alternative(("decoding", "tree"))),
    ("(decoding)", one_alternative(("decoding",))),
    ('"speculative decoding" speculative', one_alternative(("speculative", "decoding"), (SPEC,))),
    ('"投机解码"', NOTHING),
    ("投机解码 路由", NOTHING),
    ("", NOTHING),
    ("AND OR NOT", NOTHING),
])
def test_parse_query_reads_the_documented_subset_of_arxiv_syntax(text, query):
    assert parse_query(text) == query


def test_a_query_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        parse_query("x").terms = ()


# ranking


def test_a_title_match_outranks_an_abstract_only_match(index):
    ranked = ids_of(index.search({"query": "verification"}))  # title of 2609.00002; abstracts of two others
    assert ranked[0] == "2609.00002" and set(ranked) == {"2609.00001", "2609.00002", "2608.00006"}


def test_a_rarer_term_weighs_more_than_a_common_one():
    # "alpha" is in one abstract, "beta" in three; the texts are otherwise alike
    corpus = [paper("2609.00001", "Paper one", "we study beta here", "2026-09-01", ["cs.CL"]),
              paper("2609.00002", "Paper two", "we study alpha here", "2026-09-01", ["cs.CL"]),
              paper("2609.00003", "Paper three", "we study beta there", "2026-09-01", ["cs.CL"]),
              paper("2609.00004", "Paper four", "we study beta again", "2026-09-01", ["cs.CL"])]
    assert ids_of(CorpusIndex(corpus).search({"query": "alpha beta"}))[0] == "2609.00002"


def test_a_quoted_phrase_must_occur_contiguously_in_the_title_or_the_abstract():
    corpus = [paper("2609.00001", "Paper one", "decoding that is speculative", "2026-09-01", ["cs.CL"]),
              paper("2609.00002", "Paper two", "speculative decoding that is", "2026-09-01", ["cs.CL"]),
              paper("2609.00003", "Speculative Decoding", "a title match", "2026-09-01", ["cs.CL"])]
    unquoted = CorpusIndex(corpus).search({"query": "speculative decoding"})
    assert id_set(unquoted) == {"2609.00001", "2609.00002", "2609.00003"}
    quoted = CorpusIndex(corpus).search({"query": '"speculative decoding"'})
    assert ids_of(quoted) == ["2609.00003", "2609.00002"]  # the words out of order are no hit


def test_phrases_side_by_side_or_joined_by_and_are_all_required(index):
    assert ids_of(index.search({"query": '"speculative decoding" "draft length"'})) == ["2609.00001"]
    assert ids_of(index.search({"query": '"speculative decoding" AND "draft length"'})) == ["2609.00001"]
    assert index.search({"query": '"draft length" "tree verification"'}) == []  # no paper holds both


def test_an_or_of_phrases_keeps_a_paper_holding_either_and_drops_one_holding_neither(index):
    # 2609.00001 holds "draft length" only, 2609.00002 "tree verification" only; the survey 2608.00006 has
    # the terms draft and verification but neither phrase, so it is no hit
    assert id_set(index.search({"query": '"draft length" OR "tree verification"'})) == {"2609.00001",
                                                                                          "2609.00002"}
    assert id_set(index.search({"query": '"speculative decoding" OR "tree verification"'})) == SPECULATIVE
    either = index.tool()({"query": '"draft length" OR "tree verification"'})
    assert either.error is None and id_set(either.content) == {"2609.00001", "2609.00002"}


def test_an_alternative_without_a_phrase_requires_nothing(index):
    # every candidate (a paper with draft, length or verification) passes, phrase or not
    assert id_set(index.search({"query": '"draft length" OR verification'})) == {"2609.00001", "2609.00002",
                                                                                 "2608.00006", "2610.00007"}


def test_a_not_in_either_alternative_excludes_from_the_whole_query(index):
    either = {"2609.00001", "2609.00002"}
    assert ids_of(index.search({"query": '"draft length" OR "tree verification" ANDNOT scale'})) == [
        "2609.00001"]  # 2609.00002 is "Tree Verification at Scale"
    assert ids_of(index.search({"query": '"draft length" NOT adaptive OR "tree verification"'})) == [
        "2609.00002"]  # 2609.00001 is "Adaptive Draft Length ..."
    assert index.search({"query": '"draft length" OR "tree verification" ANDNOT adaptive NOT scale'}) == []
    assert id_set(index.search({"query": '"draft length" OR "tree verification" ANDNOT survey'})) == either


def test_parentheses_are_dropped_so_an_or_group_is_not_nested(index):
    # ("A" OR "B") AND "C" reads as "A" OR ("B" AND "C"): 2609.00001 holds "draft length", so it is a hit
    # although it has no "survey"; nested grouping would answer nothing, as no paper holds "survey" with
    # either phrase
    assert ids_of(index.search({"query": '("draft length" OR "tree verification") AND "survey"'})) == [
        "2609.00001"]
    assert ids_of(index.search({"query": '("draft length" OR "tree verification") AND "speculative decoding"'
                                })) == ["2609.00001", "2609.00002"]  # 2608.00006 holds the third phrase only


def test_each_phrase_a_paper_holds_adds_its_bonus_once():
    # both titles hold the same four tokens once, so BM25 ties; only 2609.00002 holds "gamma delta"
    corpus = [paper("2609.00001", "alpha beta delta gamma", "x", "2026-09-01", ["cs.CL"]),
              paper("2609.00002", "alpha beta gamma delta", "x", "2026-09-01", ["cs.CL"])]
    built = CorpusIndex(corpus)
    assert ids_of(built.search({"query": '"alpha beta"'})) == ["2609.00001", "2609.00002"]  # tie: by id
    assert ids_of(built.search({"query": '"alpha beta" OR "gamma delta"'})) == ["2609.00002", "2609.00001"]
    assert ids_of(built.search({"query": '"gamma delta" OR "alpha beta"'})) == ["2609.00002", "2609.00001"]


def test_a_quoted_phrase_meets_either_writing_of_a_hyphenated_word_wherever_it_stands(index):
    assert ids_of(index.search({"query": '"mixture of experts"'})) == ["2608.00003"]
    assert ids_of(index.search({"query": '"mixture-of-experts"'})) == ["2608.00003"]
    corpus = [paper("2609.00001", "Scaling Test-Time Compute", "x", "2026-09-01", ["cs.CL"]),
              paper("2609.00002", "Scaling test time compute", "x", "2026-09-01", ["cs.CL"]),
              paper("2609.00003", "Compute scaling at test time", "x", "2026-09-01", ["cs.CL"])]  # no phrase
    for query in ('"scaling test-time compute"', '"scaling test time compute"', '"test-time compute"'):
        assert id_set(CorpusIndex(corpus).search({"query": query})) == {"2609.00001", "2609.00002"}


def test_a_title_with_a_natural_language_not_retrieves_itself_first(index):
    titled = paper("2609.00010", "Not All Tokens Are Equal", "Tokens are weighted by their loss in "
                   "pretraining; verification of drafted tokens is not the topic.", "2026-09-05", ["cs.CL"])
    with_title = CorpusIndex([*CORPUS, titled])
    assert ids_of(with_title.search({"query": titled["title"]}))[0] == "2609.00010"
    assert check.self_retrieval(with_title, [*CORPUS, titled])["missed"] == []
    assert "2609.00010" in id_set(with_title.search({"query": "tokens not weighted"}))  # "not" is a term
    assert "2609.00010" not in id_set(with_title.search({"query": "tokens NOT weighted"}))


def test_an_excluded_term_drops_a_paper(index):
    with_survey = ids_of(index.search({"query": "speculative decoding"}))
    assert "2608.00006" in with_survey
    without = ids_of(index.search({"query": "speculative decoding ANDNOT survey"}))
    assert "2608.00006" not in without and set(without) == set(with_survey) - {"2608.00006"}
    assert ids_of(index.search({"query": 'speculative decoding NOT "draft length"'})) == ["2609.00002"]


def test_only_papers_with_at_least_one_term_are_candidates(index):
    assert id_set(index.search({"query": "decoding"})) == SPECULATIVE
    assert id_set(index.search(EVERY_PAPER)) == set(BY_ID)
    assert index.search({"query": "cat:cs.CL"}) == []  # a category alone matches nothing


# filters


def test_the_categories_argument_keeps_papers_sharing_a_category(index):
    assert ids_of(index.search({**EVERY_PAPER, "categories": ["stat.ML"]})) == ["2607.00008"]
    assert id_set(index.search({**EVERY_PAPER, "categories": ["cs.AI", "stat.ML"]})) == {"2607.00008",
                                                                                         "2608.00003"}
    assert id_set(index.search({**EVERY_PAPER, "categories": []})) == set(BY_ID)


def test_a_cat_term_in_the_query_filters_and_is_intersected_with_the_argument(index):
    query = EVERY_PAPER["query"]
    assert ids_of(index.search({"query": f"{query} cat:stat.ML"})) == ["2607.00008"]
    assert ids_of(index.search({"query": f"{query} cat:stat.ML cat:cs.AI", "categories": ["cs.AI"]})) == [
        "2608.00003"]
    assert index.search({"query": f"{query} cat:stat.ML", "categories": ["cs.AI"]}) == []


def test_date_bounds_are_inclusive_and_absent_bounds_do_not_cut(index):
    args = {"query": "speculative decoding"}
    assert id_set(index.search({**args, "date_from": "2026-09-15"})) == {"2609.00001", "2609.00002"}
    assert id_set(index.search({**args, "date_to": "2026-08-20"})) == {"2608.00006"}
    assert id_set(index.search({**args, "date_from": "2026-08-20", "date_to": "2026-09-15"})) == {
        "2609.00001", "2608.00006"}
    assert index.search({**args, "date_from": "2026-10-01"}) == []
    assert id_set(index.search({**args, "date_from": "", "date_to": None})) == SPECULATIVE


# sorting


def test_relevance_ranks_both_fields_above_the_abstract_alone_and_date_order_is_newest_first(index):
    # the two papers with the phrase in the title too lead; BM25 orders them, the abstract-only one is last
    by_relevance = ids_of(index.search({"query": "speculative decoding"}))
    assert set(by_relevance[:2]) == {"2609.00001", "2608.00006"} and by_relevance[2] == "2609.00002"
    by_date = index.search({"query": "speculative decoding", "sort_by": "submittedDate"})
    assert ids_of(by_date) == ["2609.00002", "2609.00001", "2608.00006"]
    assert [p["published"] for p in by_date] == ["2026-09-20", "2026-09-15", "2026-08-20"]
    with pytest.raises(ValueError, match="sort_by"):
        index.search({"query": "decoding", "sort_by": "lastUpdatedDate"})


def test_a_null_sort_by_categories_or_date_is_its_default_but_a_null_max_results_is_refused(index):
    args = {"query": "speculative decoding"}
    assert index.search({**args, "sort_by": None}) == index.search(args)
    assert index.tool()({**args, "sort_by": None}) == index.tool()(args)
    nulls = {**args, "sort_by": None, "categories": None, "date_from": None, "date_to": None}
    assert index.tool()(nulls) == index.tool()(args)
    with pytest.raises(TypeError, match="max_results must be an integer, got None"):
        index.tool()({**nulls, "max_results": None})


def test_equal_scores_are_ordered_by_id():
    corpus = [paper("2609.00003", "Alpha", "x", "2026-09-01", ["cs.CL"]),
              paper("2609.00001", "Alpha", "x", "2026-09-01", ["cs.CL"]),
              paper("2609.00002", "Alpha", "x", "2026-09-01", ["cs.CL"])]
    ranked = CorpusIndex(corpus).search({"query": "alpha"})
    assert ids_of(ranked) == ["2609.00001", "2609.00002", "2609.00003"]


def test_the_same_date_sorts_by_id_descending():
    corpus = [paper("2609.00001", "Alpha one", "x", "2026-09-01", ["cs.CL"]),
              paper("2609.00003", "Alpha three", "x", "2026-09-01", ["cs.CL"]),
              paper("2609.00002", "Alpha two", "x", "2026-09-01", ["cs.CL"])]
    ranked = CorpusIndex(corpus).search({"query": "alpha", "sort_by": "submittedDate"})
    assert ids_of(ranked) == ["2609.00003", "2609.00002", "2609.00001"]


# the tool


def test_the_tool_answers_with_hits_as_the_adapters_shape_them(index):
    args = {"query": "speculative decoding"}
    outcome = index.tool()(args)
    assert outcome == ToolOutcome([hit(paper) for paper in index.search(args)])
    assert id_set(outcome.content) == SPECULATIVE
    assert list(outcome.content[0]) == ["arxiv_id", "title", "published", "snippet"]
    assert all(len(h["snippet"]) <= schemas.MAX_SNIPPET_CHARS for h in outcome.content)


@pytest.mark.parametrize("max_results, shown", [(2, 2), (5, 5), (9, 5), (-1, 1), (3.0, 3), (None, 5)])
def test_the_tool_shows_at_most_max_results_and_never_more_than_the_cap(index, max_results, shown):
    args = dict(EVERY_PAPER)
    if max_results is not None:
        args["max_results"] = max_results
    ranked = index.search(args)
    assert len(ranked) == 8
    assert index.tool()(args).content == [hit(paper) for paper in ranked[:shown]]


@pytest.mark.parametrize("value", [2.5, "3", True, None])
def test_a_count_that_is_not_an_integer_is_refused(index, value):
    with pytest.raises(TypeError, match="max_results must be an integer"):
        index.tool()({"query": "decoding", "max_results": value})


@pytest.mark.parametrize("query", ["nothing here", "投机解码", "混合专家 路由", '"投机解码"', "", "AND OR"])
def test_no_match_and_a_chinese_query_are_empty_results(index, query):
    assert index.tool()({"query": query}) == ToolOutcome([], schemas.EMPTY_RESULTS)


@pytest.mark.parametrize("query", [
    '"speculative decoding" ANDNOT survey',
    '"draft length" OR "tree verification" OR "speculative decoding" NOT scale',
])
def test_the_tool_is_deterministic(index, query):
    args = {"query": query, "categories": ["cs.CL"], "max_results": 2}
    assert index.tool()(args) == index.tool()(args) == CorpusIndex(CORPUS).tool()(args)


# the index


def test_results_do_not_depend_on_the_input_order(index):
    queries = ["speculative decoding", '"mixture of experts"', "models cache", "rotary",
               '"draft length" OR "tree verification" OR "speculative decoding"']
    expected = [index.search({"query": q}) for q in queries]
    for seed in range(5):
        shuffled = list(CORPUS)
        random.Random(seed).shuffle(shuffled)
        other = CorpusIndex(shuffled)
        assert other.ids() == sorted(BY_ID) and len(other) == 8
        assert [other.search({"query": q}) for q in queries] == expected


def test_a_repeated_id_is_refused():
    with pytest.raises(ValueError, match=r"paper ids repeat: \['2609.00001'\]"):
        CorpusIndex([CORPUS[0], {**CORPUS[0], "title": "Another"}, CORPUS[1]])


def test_the_index_keeps_copies_of_the_six_paper_keys_only():
    source = [{**copy.deepcopy(p), "updated": "2026-09-30"} for p in CORPUS]
    built = CorpusIndex(source)
    source[0]["title"] = "Changed"
    source[0]["categories"].append("cs.CV")
    results = built.search({"query": "adaptive draft"})
    assert results[0] == CORPUS[0] and list(results[0]) == schemas.PAPER_KEYS
    results[0]["categories"].append("stat.ML")
    assert built.search({"query": "adaptive draft"})[0]["categories"] == ["cs.CL", "cs.LG"]


def test_an_empty_index_matches_nothing():
    assert len(CorpusIndex([])) == 0 and CorpusIndex([]).ids() == []
    assert CorpusIndex([]).tool()({"query": "decoding"}) == ToolOutcome([], schemas.EMPTY_RESULTS)


# load_papers


def feed_text(papers: list[dict]) -> str:
    """A cassette body the test's parse_atom reads: the papers as JSON (the real one reads Atom)."""
    return json.dumps(papers)


def parse_json_feed(xml: str) -> list[dict] | None:
    return None if xml == "broken" else json.loads(xml)


class Cassettes:
    """A fetcher double that only reads: url -> (status, body); nothing for a URL without a cassette."""

    def __init__(self, pages: dict) -> None:
        self.pages, self.asked = pages, []

    def recorded(self, url: str) -> Fetched | None:
        self.asked.append(url)
        if url not in self.pages:
            return None
        status, body = self.pages[url]
        return Fetched(url=url, final_url=url, status=status, body=body.encode("utf-8"),
                       content_type="application/atom+xml")

    def get(self, url: str) -> Fetched:
        raise AssertionError(f"load_papers must not fetch, but asked for {url}")


def test_load_papers_reads_each_paper_from_its_cassette_in_id_order():
    ids = ["2609.00002", "2609.00001"]
    fetcher = Cassettes({paper_url(i): (200, feed_text([BY_ID[i]])) for i in ids})
    assert load_papers(fetcher, ids, parse_json_feed) == [BY_ID["2609.00002"], BY_ID["2609.00001"]]
    assert fetcher.asked == [paper_url("2609.00002"), paper_url("2609.00001")]


@pytest.mark.parametrize("pages, problem", [
    ({}, "no cassette for https://export.arxiv.org/api/query\\?id_list=2609.00001"),
    ({paper_url("2609.00001"): (404, "")}, "its cassette holds status 404, not 200"),
    ({paper_url("2609.00001"): (200, "broken")}, "parse_atom could not parse its recorded feed"),
    ({paper_url("2609.00001"): (200, feed_text([]))}, "its feed holds 0 papers, not 1"),
    ({paper_url("2609.00001"): (200, feed_text([CORPUS[0], CORPUS[1]]))}, "its feed holds 2 papers, not 1"),
    ({paper_url("2609.00001"): (200, feed_text([CORPUS[1]]))}, "its feed holds paper 2609.00002"),
])
def test_load_papers_refuses_a_cassette_that_does_not_hold_exactly_the_paper(pages, problem):
    with pytest.raises(HarnessError, match=f"^corpus paper 2609.00001: {problem}"):
        load_papers(Cassettes(pages), ["2609.00001"], parse_json_feed)


def test_load_papers_holds_parse_atom_to_its_contract_under_the_index_name():
    without_categories = {k: v for k, v in CORPUS[0].items() if k != "categories"}
    fetcher = Cassettes({paper_url("2609.00001"): (200, feed_text([without_categories]))})
    with pytest.raises(HarnessError, match="^tool 'corpus_index': parse_atom returned a paper without categ"):
        load_papers(fetcher, ["2609.00001"], parse_json_feed)


def test_load_papers_through_the_real_fetcher_reads_cassettes_without_requests(tmp_path):
    fetcher = Fetcher("replay", tmp_path)
    write_cassette(tmp_path, "2609.00001", feed_text([CORPUS[0]]))
    assert load_papers(fetcher, ["2609.00001"], parse_json_feed) == [CORPUS[0]]
    with pytest.raises(HarnessError, match="corpus paper 2609.00002: no cassette"):
        load_papers(fetcher, ["2609.00001", "2609.00002"], parse_json_feed)


def write_cassette(cassette_dir: Path, arxiv_id: str, body: str, status: int = 200) -> None:
    """A cassette as net.fetch records one, for the paper's id_list URL."""
    url = canonical_url(paper_url(arxiv_id))
    path = Fetcher("replay", cassette_dir).cassette_path(url)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"url": url, "final_url": url, "status": status, "content_type":
                                "application/atom+xml", "fetched_at": "2026-10-10T08:00:00+00:00",
                                "body_base64": False, "body": body}), encoding="utf-8")


# corpus_ids


def write_manifest(path: Path, topics: list[dict], window: list | None = WINDOW) -> Path:
    data = {"recorded_on": "2026-10-10", "topics": topics}
    if window is not None:
        data["window"] = window
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_corpus_ids_is_the_sorted_union_of_the_topics_papers(tmp_path):
    manifest = write_manifest(tmp_path / "manifest.json", [
        {"id": "a", "phrases": ["x"], "papers": ["2609.00002", "2609.00001"]},
        {"id": "b", "phrases": ["y"], "papers": ["2608.00003", "2609.00001"]}])
    assert corpus_ids(manifest) == ["2608.00003", "2609.00001", "2609.00002"]
    assert corpus_ids(str(manifest)) == corpus_ids(manifest)


@pytest.mark.parametrize("data, problem", [
    ({"recorded_on": "2026-10-10"}, "no topics list"),
    ({"topics": {"a": []}}, "no topics list"),
    ({"topics": [{"id": "a", "phrases": ["x"]}]}, "topic 'a' has no papers list"),
    ({"topics": [{"id": "a", "papers": "2609.00001"}]}, "topic 'a' has no papers list"),
    ({"topics": [{"id": "a", "papers": [2609]}]}, "topic 'a' has no papers list"),
    ({"topics": [{"id": "a", "papers": []}]}, "its topics name no papers"),
    ({"topics": []}, "its topics name no papers"),
])
def test_corpus_ids_refuses_a_manifest_without_topics_or_papers(tmp_path, data, problem):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(HarnessError, match=f"^corpus manifest .*manifest.json: {problem}$"):
        corpus_ids(manifest)


@pytest.mark.needs_data("data/cassettes/api")  # the recorded metadata: left out of the public copy
def test_the_committed_manifest_names_300_papers_with_a_recorded_200_cassette_each():
    ids = corpus_ids(REPO / MANIFEST)  # anchored to the repository, whatever the working directory
    assert len(ids) == 300 and ids == sorted(ids)
    fetcher = Fetcher("replay", REPO / "data" / "cassettes")
    missing = [i for i in ids if (page := fetcher.recorded(paper_url(i))) is None or page.status != 200]
    assert missing == []


# the check script


def test_load_function_imports_a_function_from_a_file_anywhere(tmp_path):
    module = tmp_path / "owner_parse.py"
    module.write_text("def parse_atom(xml):\n    return []\n", encoding="utf-8")
    assert check.load_function(f"{module}:parse_atom")("<feed/>") == []
    with pytest.raises(SystemExit, match="has no function other"):
        check.load_function(f"{module}:other")
    with pytest.raises(SystemExit, match="must be <path.py>:<function>"):
        check.load_function(str(module))


@pytest.mark.parametrize("phrase, literal, loose", [
    ("speculative decoding", True, True),
    ("Speculative  Decoding", True, True),
    ("draft length", True, True),
    ("mixture of experts", False, False),
    ("adaptive draft", True, True),
])
def test_phrase_in_matches_case_and_whitespace_insensitively(phrase, literal, loose):
    assert check.phrase_in(CORPUS[0], phrase) is literal
    assert check.phrase_in(CORPUS[0], phrase, hyphens=True) is loose


def test_phrase_in_with_hyphens_equates_the_hyphenated_writing():
    assert check.phrase_in(BY_ID["2608.00003"], "mixture of experts") is False
    assert check.phrase_in(BY_ID["2608.00003"], "mixture of experts", hyphens=True) is True
    assert check.phrase_in(BY_ID["2608.00003"], "mixture-of-experts") is True


def test_precision_at_counts_relevant_ids_among_the_first_k_over_k():
    assert check.precision_at(["a", "b", "c", "d", "e", "f"], {"a", "c", "f"}) == 0.4
    assert check.precision_at(["a"], {"a"}) == 0.2  # fewer hits than k count as misses
    assert check.precision_at([], {"a"}) == 0.0
    assert check.precision_at(["a", "b"], {"a", "b"}, k=2) == 1.0


def test_phrase_relevance_scores_each_phrase_with_its_ceiling(index):
    topics = [{"id": "spec", "phrases": ["speculative decoding", "draft length"], "papers": []},
              {"id": "moe", "phrases": ["mixture of experts"], "papers": []}]
    report = check.phrase_relevance(index, BY_ID, topics, WINDOW)
    spec = report["topics"]["spec"]["phrases"]
    top = ids_of(index.search({"query": '"speculative decoding"', "date_from": WINDOW[0],
                               "date_to": WINDOW[1]}))
    assert set(top) == SPECULATIVE
    assert spec["speculative decoding"] == {"p_at_5": 0.6, "literal_matches": 3, "ceiling": 0.6,
                                            "p_at_5_hyphen_insensitive": 0.6, "top": top}
    assert spec["draft length"]["p_at_5"] == 0.2 and spec["draft length"]["literal_matches"] == 1
    moe = report["topics"]["moe"]["phrases"]["mixture of experts"]  # the phrase is required: one hit, the
    assert moe["p_at_5"] == 0.0 and moe["p_at_5_hyphen_insensitive"] == 0.2  # hyphenated writing
    assert moe["top"] == ["2608.00003"]
    assert report["phrases"] == 3 and report["mean"] == pytest.approx((0.6 + 0.2 + 0.0) / 3)
    assert report["ceiling"] == pytest.approx((0.6 + 0.2 + 0.0) / 3)  # the ceiling is literal too
    assert report["mean_hyphen_insensitive"] == pytest.approx((0.6 + 0.2 + 0.2) / 3)
    assert report["topics"]["spec"]["mean"] == pytest.approx(0.4)


def test_self_retrieval_reports_the_papers_that_do_not_rank_first(index):
    assert check.self_retrieval(index, CORPUS) == {"fraction": 1.0, "papers": 8, "missed": []}
    twin = paper("2609.00009", "Adaptive Draft Length for Speculative Decoding", "A twin title.",
                 "2026-09-16", ["cs.CL"])
    report = check.self_retrieval(CorpusIndex([*CORPUS, twin]), [*CORPUS, twin])
    assert report["fraction"] == pytest.approx(8 / 9)
    assert report["missed"] == [{"arxiv_id": "2609.00009", "first": "2609.00001"}]


def test_topic_or_scores_the_or_of_the_phrases_against_the_topics_ids(index):
    topics = [{"id": "spec", "phrases": ["draft length", "tree verification"],
               "papers": ["2609.00001", "2609.00002", "2608.00006"]},
              {"id": "none", "phrases": [], "papers": ["2609.00001"]}]
    # OR reads as alternatives: 2609.00001 and 2609.00002 carry one phrase each, the survey neither
    assert check.topic_or(index, topics, WINDOW) == {"topics": {"spec": 0.4}, "mean": 0.4}
    single = [{"id": "spec", "phrases": ["speculative decoding"], "papers": sorted(SPECULATIVE)}]
    assert check.topic_or(index, single, WINDOW) == {"topics": {"spec": 0.6}, "mean": 0.6}


def test_chinese_queries_must_all_be_empty_results(index):
    report = check.chinese_queries(index)
    assert report == {"codes": dict.fromkeys(check.CHINESE_QUERIES, schemas.EMPTY_RESULTS), "all_empty": True}
    assert check.chinese_queries(index, ["decoding"]) == {"codes": {"decoding": None}, "all_empty": False}


def script_world(tmp_path: Path, topics: list[dict], corpus: list[dict] = CORPUS) -> list[str]:
    """Cassettes for the corpus, a manifest with topics, a parse_atom module: the script's arguments."""
    for p in corpus:
        write_cassette(tmp_path / "cassettes", p["arxiv_id"], feed_text([p]))
    manifest = write_manifest(tmp_path / "manifest.json", topics)
    module = tmp_path / "owner_parse.py"
    module.write_text("import json\n\ndef parse_atom(xml):\n    return json.loads(xml)\n", encoding="utf-8")
    return ["--parse-atom", f"{module}:parse_atom", "--cassettes", str(tmp_path / "cassettes"),
            "--manifest", str(manifest), "--json", str(tmp_path / "out" / "report.json")]


# six papers that all write "speculative decoding", so P@5 can reach 1.0 and every title is its own
SPECULATIVE_CORPUS = [
    paper(f"2609.0001{n}", f"{title} for Speculative Decoding",
          f"We study speculative decoding from the angle of {title.lower()}.", f"2026-09-{10 + n:02d}",
          ["cs.CL"])
    for n, title in enumerate(["Draft Trees", "Token Budgets", "Early Exits", "Layer Skipping",
                               "Cache Reuse", "Batch Scheduling"])]


def test_the_script_passes_and_writes_its_report_when_the_thresholds_hold(tmp_path, capsys):
    topics = [{"id": "spec", "phrases": ["speculative decoding"], "papers": ids_of(SPECULATIVE_CORPUS)}]
    argv = script_world(tmp_path, topics, SPECULATIVE_CORPUS)
    assert check.main(argv) == 0
    out = capsys.readouterr().out
    assert "index: 6 papers" in out and "PASS: phrase relevance 1.000 >= 0.9, self-retrieval 1.000" in out
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "PASS" and report["papers"] == 6 and report["build_seconds"] >= 0
    assert report["phrase_relevance"]["mean"] == 1.0 and report["self_retrieval"]["fraction"] == 1.0
    assert report["topic_or"]["topics"] == {"spec": 1.0} and report["chinese"]["all_empty"] is True
    assert report["schema_hash"] == schemas.SCHEMA_HASH


def test_the_script_fails_when_a_phrase_is_found_in_too_few_papers(tmp_path, capsys):
    topics = [{"id": "spec", "phrases": ["speculative decoding"], "papers": ids_of(CORPUS)}]
    assert check.main(script_world(tmp_path, topics)) == 1
    out = capsys.readouterr().out
    assert "FAIL: phrase relevance 0.600 >= 0.9" in out and '"speculative decoding" 0.60 (ceiling 0.60' in out
    assert json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))["verdict"] == "FAIL"


def atom_feed(p: dict) -> str:
    """One paper as an arXiv API Atom feed, which research_agent.arxiv.parse_atom reads."""
    authors = "".join(f"<author><name>{name}</name></author>" for name in p["authors"])
    categories = "".join(f'<category term="{term}"/>' for term in p["categories"])
    return (f'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/{p["arxiv_id"]}v1</id>'
            f'<title>{p["title"]}</title>{authors}<published>{p["published"]}T00:00:00Z</published>'
            f'<summary>{p["abstract"]}</summary>{categories}</entry></feed>')


def test_the_script_without_parse_atom_builds_the_index_with_the_arxiv_modules_parser(tmp_path, capsys):
    for p in SPECULATIVE_CORPUS:
        write_cassette(tmp_path / "cassettes", p["arxiv_id"], atom_feed(p))
    topics = [{"id": "spec", "phrases": ["speculative decoding"], "papers": ids_of(SPECULATIVE_CORPUS)}]
    manifest = write_manifest(tmp_path / "manifest.json", topics)
    assert check.main(["--cassettes", str(tmp_path / "cassettes"), "--manifest", str(manifest)]) == 0
    out = capsys.readouterr().out
    assert "index: 6 papers" in out and "parser research_agent.arxiv.parse_atom" in out
    assert "PASS: phrase relevance 1.000 >= 0.9, self-retrieval 1.000" in out


def test_the_script_names_the_parse_atom_it_was_given(tmp_path, capsys):
    topics = [{"id": "spec", "phrases": ["speculative decoding"], "papers": ids_of(SPECULATIVE_CORPUS)}]
    argv = script_world(tmp_path, topics, SPECULATIVE_CORPUS)
    assert check.main(argv) == 0
    assert f"parser {argv[1]}" in capsys.readouterr().out


def test_read_manifest_checks_the_manifest_as_corpus_ids_does_and_carries_its_ids(tmp_path):
    topics = [{"id": "spec", "phrases": ["speculative decoding"], "papers": ["2609.00002", "2609.00001"]}]
    manifest = write_manifest(tmp_path / "manifest.json", topics)
    assert check.read_manifest(manifest) == {"ids": ["2609.00001", "2609.00002"], "topics": topics,
                                             "window": WINDOW}
    assert check.read_manifest(write_manifest(manifest, topics, window=None))["window"] == [None, None]


def test_the_script_refuses_a_manifest_without_papers_with_the_corpus_ids_message(tmp_path):
    argv = script_world(tmp_path, [{"id": "spec", "phrases": ["speculative decoding"]}])
    with pytest.raises(HarnessError, match="^corpus manifest .*: topic 'spec' has no papers list$"):
        check.main(argv)
    with pytest.raises(HarnessError, match="topic 'spec' has no papers list"):
        check.read_manifest(tmp_path / "manifest.json")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("topics, position", [
    ([{"phrases": ["speculative decoding"], "papers": ["2609.00001"]}], 0),
    ([{"id": "spec", "papers": ["2609.00001"]}, {"id": 3, "papers": ["2609.00002"]}], 1),
    ([{"id": "spec", "papers": ["2609.00001"]}, {"id": None, "papers": ["2609.00002"]}], 1),
])
def test_read_manifest_refuses_a_topic_without_an_id_string_although_corpus_ids_accepts_it(tmp_path, topics,
                                                                                            position):
    argv = script_world(tmp_path, topics)
    manifest = tmp_path / "manifest.json"
    assert corpus_ids(manifest)  # the ids are fine: only the script needs the topic's name
    problem = rf"topics\[{position}\] has no id string$"
    with pytest.raises(HarnessError, match=rf"^corpus manifest .*manifest.json: {problem}"):
        check.read_manifest(manifest)
    with pytest.raises(HarnessError, match=rf"topics\[{position}\] has no id string$"):
        check.main(argv)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("window", [["2026-07-01"], "2026-07-01", ["2026-07-01", None],
                                    ["2026-07-01", "2026-10-10", "2026-10-11"], [], {}, [20260701, 20261010]])
def test_read_manifest_refuses_a_window_that_is_not_two_strings(tmp_path, window):
    topics = [{"id": "spec", "phrases": ["speculative decoding"], "papers": ["2609.00001"]}]
    argv = script_world(tmp_path, topics)  # writes the manifest with a good window: overwritten next
    manifest = write_manifest(tmp_path / "manifest.json", topics, window=window)
    assert corpus_ids(manifest) == ["2609.00001"]  # corpus_ids does not read the window
    with pytest.raises(HarnessError, match="^corpus manifest .*manifest.json: window is not a list of two "
                                           "strings: "):
        check.read_manifest(manifest)
    with pytest.raises(HarnessError, match="window is not a list of two strings"):
        check.main(argv)
    assert not (tmp_path / "out").exists()

"""Tests for research_agent.notes: rendering, evidence picking, the store, the tools and the importer."""

from __future__ import annotations

import sqlite3

import pytest

from research_agent import notes
from research_agent.agent.types import ToolOutcome
from research_agent.data.synth import default_note_tools
from research_agent.notes import (
    NoteStore,
    import_entries,
    locate_evidence,
    note_tools,
    pick_evidence,
    render_note,
)
from research_agent.tools import schemas

TODAY = "2026-10-13"
PAPER = {"arxiv_id": "2609.01234", "title": "Speculative  Decoding\n with Draft Trees",
         "published": "2026-09-02", "abstract": "", "categories": ["cs.CL", "cs.LG"]}
EVIDENCE = ["Decoding is slow.", "We propose draft trees."]

SCAFFOLD = """\
### 2026-10-13 Speculative Decoding with Draft Trees

- 来源:https://arxiv.org/abs/2609.01234;2026-09-02;论文
- 与本周所学的关系:
- Summary (about 300 words)
  - Problem:
  - Method:
  - Evidence:
  - Limitations:
- 证据句(原文)
  - Decoding is slow.
  - We propose draft trees.
- 新术语:draft tree, acceptance rate
- 待解术语:
"""

DRAFT = """\
### 2026-10-13 Speculative Decoding with Draft Trees

- 来源:https://arxiv.org/abs/2609.01234;2026-09-02;论文
- 与本周所学的关系:
- Summary (about 300 words)
  - Problem: Autoregressive decoding is slow.
  - Method: A small model drafts a tree of tokens.
  - Evidence: 2.1x speed-up on two benchmarks.
  - Limitations:
- 新术语:
- 待解术语:KV cache
"""


# ---- render_note -------------------------------------------------------------------------------------------


def test_render_note_in_scaffold_mode_matches_the_template_line_for_line():
    args = {"arxiv_id": PAPER["arxiv_id"], "problem": "ignored in scaffold",
            "new_terms": ["draft tree", "acceptance rate"]}
    assert render_note(PAPER, args, EVIDENCE, TODAY).split("\n") == SCAFFOLD.split("\n")


def test_render_note_in_draft_mode_carries_the_four_parts_and_no_evidence_bullet():
    args = {"arxiv_id": PAPER["arxiv_id"], "mode": "draft", "problem": "Autoregressive decoding\nis slow.",
            "method": "A small model drafts a tree of tokens.",
            "evidence": "2.1x speed-up on two benchmarks.",
            "open_terms": ["KV cache"]}
    assert render_note(PAPER, args, EVIDENCE, TODAY).split("\n") == DRAFT.split("\n")


def test_render_note_ends_with_one_newline_and_has_only_the_blank_line_after_the_heading():
    text = render_note(PAPER, {"arxiv_id": PAPER["arxiv_id"]}, [], TODAY)
    assert text.endswith("- 待解术语:\n") and not text.endswith("\n\n")
    assert text.split("\n")[:-1].count("") == 1 and text.split("\n")[1] == ""


def test_render_note_collapses_a_line_break_inside_an_evidence_sentence_so_the_entry_keeps_its_lines():
    text = render_note(PAPER, {"arxiv_id": PAPER["arxiv_id"]}, ["x\ny"], TODAY)
    assert len(text.splitlines()) == 13 and "\n  - x y\n" in text
    (entry,) = import_entries(text)
    assert entry["evidence_sentences"] == ["x y"]


def test_the_relation_field_stays_empty_while_the_schema_has_no_such_field():
    assert notes.RELATION_FIELD not in notes._write_note_properties()
    text = render_note(PAPER, {"arxiv_id": PAPER["arxiv_id"], notes.RELATION_FIELD: "related"}, [], TODAY)
    assert "- 与本周所学的关系:\n" in text


# ---- pick_evidence -----------------------------------------------------------------------------------------

ABSTRACT = ("Large language models decode one token at a time. This makes inference slow, e.g. on long\n"
            "outputs. We propose   draft trees, which verify many tokens per step! Is it exact? Yes: the\n"
            "output distribution is unchanged. Speed-ups reach 2.1x on two benchmarks.")


def test_pick_evidence_returns_sentences_found_verbatim_or_after_the_same_collapse():
    picked = pick_evidence(ABSTRACT, 10, [])
    collapsed = " ".join(ABSTRACT.split())
    assert picked[0] == "Large language models decode one token at a time."
    assert picked[1] == "This makes inference slow, e.g. on long outputs."  # collapsed; e.g. kept
    assert "We propose   draft trees, which verify many tokens per step!" in picked  # no line break: as is
    for sentence in picked:
        assert locate_evidence(ABSTRACT, sentence) >= 0
        assert sentence in ABSTRACT or collapsed.find(sentence) >= 0
        if "\n" not in sentence and sentence in ABSTRACT:
            assert ABSTRACT.find(sentence) >= 0
    assert collapsed.find(picked[1]) >= 0 and picked[1] not in ABSTRACT


def test_pick_evidence_prefers_keyword_sentences_then_the_earliest_and_respects_k():
    assert pick_evidence(ABSTRACT, 2, ["SPEED-UPS", "distribution"]) == [
        "Yes: the output distribution is unchanged.", "Speed-ups reach 2.1x on two benchmarks."]
    assert pick_evidence(ABSTRACT, 2, ["no such word"]) == pick_evidence(ABSTRACT, 2, [])
    assert pick_evidence(ABSTRACT, 1, ["ｄｒａｆｔ"]) == [
        "We propose   draft trees, which verify many tokens per step!"]
    assert len(pick_evidence(ABSTRACT, 3, [])) == 3 and pick_evidence(ABSTRACT, 0, []) == []
    assert pick_evidence("", 3, ["x"]) == []


def test_pick_evidence_is_deterministic():
    expected = ["Large language models decode one token at a time.",
                "This makes inference slow, e.g. on long outputs.",
                "We propose   draft trees, which verify many tokens per step!"]
    assert pick_evidence(ABSTRACT, 3, ["tokens"]) == expected  # the keyword sentence, then the two earliest
    assert pick_evidence(ABSTRACT, 3, ["tokens"]) == expected


def test_pick_evidence_splits_chinese_and_mixed_punctuation():
    text = "推测解码用小模型起草。大模型一次验证多个词元!它是无损的吗?是的;分布不变。Speed-ups are 2x."
    assert pick_evidence(text, 10, []) == ["推测解码用小模型起草。", "大模型一次验证多个词元!",
                                           "它是无损的吗?", "是的;", "分布不变。", "Speed-ups are 2x."]
    assert pick_evidence(text, 1, ["验证"]) == ["大模型一次验证多个词元!"]
    for sentence in pick_evidence(text, 10, []):
        assert sentence in text


# ---- NoteStore ---------------------------------------------------------------------------------------------


def paper(arxiv_id: str, title: str, abstract: str = "", published: str = "2026-09-01") -> dict:
    return {"arxiv_id": arxiv_id, "title": title, "published": published, "abstract": abstract}


def test_the_store_writes_files_and_answers_search_and_list_with_tags_and_since(tmp_path):
    store = NoteStore(tmp_path)
    first = store.write(paper("2609.00001", "Speculative decoding"), {"mode": "scaffold"},
                        ["Drafts are verified."], "2026-10-01", ["cs.CL", "推测解码"])
    second = store.write(paper("2609.00002", "Mixture of experts routing"), {}, [], "2026-10-05", ["cs.LG"])
    assert first == {"note_id": "2609.00001-1", "arxiv_id": "2609.00001", "path": "2609.00001-1.md"}
    assert (tmp_path / first["path"]).read_text(encoding="utf-8").startswith(
        "### 2026-10-01 Speculative decoding\n")
    assert (tmp_path / "notes.sqlite").is_file()
    hits = store.search("decoding", None, 5)
    assert [h["note_id"] for h in hits] == ["2609.00001-1"]
    assert set(hits[0]) == {"note_id", "arxiv_id", "title", "snippet", "score"}
    assert len(hits[0]["snippet"]) <= schemas.MAX_SNIPPET_CHARS and hits[0]["score"] > 0
    assert store.search("verified", ["CS.cl"], 5)[0]["note_id"] == "2609.00001-1"
    assert store.search("verified", ["cs.LG"], 5) == []
    assert store.search("verified", ["cs.CL", "cs.LG"], 5) == []  # every tag is required
    assert [r["note_id"] for r in store.list()] == ["2609.00002-1", "2609.00001-1"]
    assert store.list(tag="推测解码") == [{"note_id": "2609.00001-1", "arxiv_id": "2609.00001",
                                           "title": "Speculative decoding", "created": "2026-10-01",
                                           "tags": ["cs.CL", "推测解码"]}]
    assert [r["note_id"] for r in store.list(since="2026-10-02")] == ["2609.00002-1"]
    assert [r["note_id"] for r in store.list(limit=1)] == ["2609.00002-1"]
    assert second["path"] == "2609.00002-1.md"


def test_two_notes_of_one_paper_get_distinct_file_names(tmp_path):
    store = NoteStore(tmp_path)
    one = store.write(paper("2609.00001", "A"), {}, [], TODAY)
    two = store.write(paper("2609.00001", "A"), {"mode": "draft", "problem": "p"}, [], TODAY)
    assert (one["note_id"], two["note_id"]) == ("2609.00001-1", "2609.00001-2")
    assert sorted(p.name for p in tmp_path.glob("*.md")) == ["2609.00001-1.md", "2609.00001-2.md"]
    assert [r["note_id"] for r in store.list()] == ["2609.00001-2", "2609.00001-1"]  # same day: last first


def test_bm25_orders_a_three_note_store_and_ties_break_by_note_id(tmp_path):
    store = NoteStore(tmp_path)
    store.write(paper("2609.00003", "Quantization of large models"), {"mode": "draft",
                "method": "Low-bit weights."}, [], TODAY)
    store.write(paper("2609.00001", "Speculative decoding"), {"mode": "draft",
                "method": "Speculative drafts; speculative verification of speculative tokens."}, [], TODAY)
    store.write(paper("2609.00002", "Tree attention"), {"mode": "draft",
                "method": "A speculative variant is discussed."}, [], TODAY)
    hits = store.search("speculative", None, 5)
    assert [h["arxiv_id"] for h in hits] == ["2609.00001", "2609.00002"]
    assert hits[0]["score"] > hits[1]["score"]
    assert store.search("speculative quantization", None, 2) == []  # no note holds both: an AND
    assert [h["arxiv_id"] for h in store.search("speculative tokens", None, 2)] == ["2609.00001"]
    tied = NoteStore(tmp_path / "tied")
    for arxiv_id in ("2609.00009", "2609.00005"):
        tied.write(paper(arxiv_id, "Same title"), {}, [], TODAY)
    assert [h["arxiv_id"] for h in tied.search("same", None, 5)] == ["2609.00005", "2609.00009"]


def test_search_without_a_match_or_without_terms_returns_an_empty_list(tmp_path):
    store = NoteStore(tmp_path)
    store.write(paper("2609.00001", "Speculative decoding"), {}, ["我们提出推测解码。"], TODAY)
    assert store.search("diffusion", None, 3) == []
    assert store.search("", None, 3) == [] and store.search("  ?! ", None, 3) == []
    assert store.search("推测解码", None, 3)[0]["note_id"] == "2609.00001-1"  # Chinese as a character phrase
    assert store.search("解码推测", None, 3) == []
    for label in ("summary", "words", "Problem", "来源", "论文", "arxiv"):  # template text is not indexed
        assert store.search(label, None, 3) == []


def test_the_store_is_deterministic_for_the_same_writes(tmp_path):
    def build(root):
        store = NoteStore(root)
        store.write(paper("2609.00001", "A"), {}, ["x."], TODAY, ["t"])
        store.write(paper("2609.00001", "A"), {}, ["y."], TODAY, ["t"])
        return store.list(), {p.name: p.read_bytes() for p in root.glob("*.md")}

    listed, files = build(tmp_path / "a")
    assert [r["note_id"] for r in listed] == ["2609.00001-2", "2609.00001-1"]
    assert sorted(files) == ["2609.00001-1.md", "2609.00001-2.md"] and all(files.values())
    assert build(tmp_path / "b") == (listed, files)


THREE_TITLES = [("2609.00001", "Speculative decoding of draft trees"),
                ("2609.00002", "Mixture of experts routing"),
                ("2609.00003", "Quantization and pruning of large models")]


def test_a_title_query_matches_only_the_note_holding_every_content_word(tmp_path):
    store = NoteStore(tmp_path)
    for arxiv_id, title in THREE_TITLES:
        store.write(paper(arxiv_id, title), {}, [], TODAY)
    assert [h["arxiv_id"] for h in store.search("Mixture of experts routing", None, 5)] == ["2609.00002"]
    assert [h["arxiv_id"] for h in store.search("the routing of experts in a mixture", None, 5)] == [
        "2609.00002"]
    assert store.search("mixture quantization", None, 5) == []  # every content word is required
    assert store.search("of and to the in", None, 5) == []  # stopwords alone are no query
    assert store.search("的 了 和", None, 5) == []


def test_chinese_particles_split_a_chinese_query_into_terms_that_are_all_required(tmp_path):
    store = NoteStore(tmp_path)
    store.write(paper("2609.00001", "推测解码综述"), {}, ["我们提出推测解码。"], TODAY)
    store.write(paper("2609.00002", "混合专家"), {}, ["路由与负载均衡。"], TODAY)
    assert [h["arxiv_id"] for h in store.search("推测解码的综述", None, 5)] == ["2609.00001"]
    assert [h["arxiv_id"] for h in store.search("路由和负载均衡", None, 5)] == ["2609.00002"]


def test_scores_are_unrounded_so_a_stronger_hit_scores_higher_and_comes_first(tmp_path):
    store = NoteStore(tmp_path)  # two notes that both hold the term: FTS5's idf is tiny, the ranks close
    store.write(paper("2609.00001", "Alpha"), {"mode": "draft", "method": "routing note"}, [], TODAY)
    store.write(paper("2609.00002", "Beta"), {"mode": "draft", "method": "routing routing routing note"},
                [], TODAY)
    hits = store.search("routing", None, 5)
    assert [h["arxiv_id"] for h in hits] == ["2609.00002", "2609.00001"]
    assert hits[0]["score"] > hits[1]["score"] > 0


def test_a_snippet_starts_at_a_word_boundary(tmp_path):
    store = NoteStore(tmp_path)
    method = ("Alpha beta gamma delta epsilon zeta eta theta iota kappa we provide a flexible router "
              "for every token.")
    store.write(paper("2609.00001", "Routing"), {"mode": "draft", "method": method}, [], TODAY)
    snippet = store.search("router", None, 1)[0]["snippet"]
    start = method.find(snippet)
    assert start >= 0 and "router" in snippet
    assert start == 0 or method[start - 1] == " "  # no word cut in half ("ovide a flexible")


def test_the_store_refuses_a_non_canonical_arxiv_id_before_writing_anything(tmp_path):
    store = NoteStore(tmp_path)
    for bad in ("hep-th/9901001", "2609.00001v2", "../2609.00001", "2609.1"):
        with pytest.raises(ValueError, match="arxiv_id"):
            store.write(paper(bad, "Old style"), {}, [], TODAY)
    assert list(tmp_path.glob("*.md")) == [] and store.list() == []


def test_a_failing_insert_leaves_no_markdown_file(tmp_path):
    store = NoteStore(tmp_path)
    with sqlite3.connect(tmp_path / "notes.sqlite") as db:
        db.execute("CREATE TRIGGER refuse BEFORE INSERT ON notes BEGIN SELECT RAISE(ABORT, 'refused'); END")
    with pytest.raises(sqlite3.DatabaseError, match="refused"):
        store.write(paper("2609.00001", "A"), {}, [], TODAY)
    assert list(tmp_path.glob("*.md*")) == []
    with sqlite3.connect(tmp_path / "notes.sqlite") as db:
        db.execute("DROP TRIGGER refuse")
    assert store.write(paper("2609.00001", "A"), {}, [], TODAY)["note_id"] == "2609.00001-1"


def test_imported_entries_seed_a_store_and_are_found_by_search(tmp_path):
    rendered = render_note(PAPER, {"arxiv_id": PAPER["arxiv_id"], "new_terms": ["draft tree"]}, EVIDENCE,
                           "2026-10-01")
    hand_written = "### 2026-10-06 AI Index Report 2026\n\n- 来源:\n- 新术语:指数\n"
    entries = import_entries(rendered + "\n" + hand_written)
    store = NoteStore(tmp_path)
    loaded = store.load_entries(entries, tags=["imported"])
    assert loaded == [{"note_id": "2609.01234-1", "arxiv_id": "2609.01234", "path": "2609.01234-1.md"}]
    assert (tmp_path / "2609.01234-1.md").read_text(encoding="utf-8") == rendered
    assert import_entries((tmp_path / "2609.01234-1.md").read_text(encoding="utf-8")) == entries[:1]
    hits = store.search("draft trees", ["imported"], 3)
    assert [h["note_id"] for h in hits] == ["2609.01234-1"]
    assert store.list() == [{"note_id": "2609.01234-1", "arxiv_id": "2609.01234",
                             "title": "Speculative Decoding with Draft Trees", "created": "2026-10-01",
                             "tags": ["imported"]}]


def test_a_store_opened_again_on_its_folder_finds_its_notes_and_numbers_the_next_one_on(tmp_path):
    first = NoteStore(tmp_path)
    first.write(paper("2609.00001", "Speculative decoding"), {}, ["Drafts are verified."], "2026-10-01",
                ["cs.CL"])
    again = NoteStore(tmp_path)  # another process on the same folder: nothing is created anew or lost
    assert [h["note_id"] for h in again.search("verified", ["cs.CL"], 3)] == ["2609.00001-1"]
    assert again.list() == first.list() and [r["note_id"] for r in again.list()] == ["2609.00001-1"]
    written = again.write(paper("2609.00001", "Speculative decoding"), {}, [], TODAY)
    assert written["note_id"] == "2609.00001-2"
    assert sorted(p.name for p in tmp_path.glob("*.md")) == ["2609.00001-1.md", "2609.00001-2.md"]
    assert [r["note_id"] for r in NoteStore(tmp_path).list()] == ["2609.00001-2", "2609.00001-1"]


def test_skip_existing_imports_a_file_once_and_the_default_imports_every_entry_again(tmp_path):
    other = paper("2609.00002", "Mixture of experts")
    owner_file = (render_note(PAPER, {"arxiv_id": PAPER["arxiv_id"]}, EVIDENCE, "2026-10-01") + "\n"
                  + render_note(other, {"arxiv_id": "2609.00002"}, [], "2026-10-02"))
    entries = import_entries(owner_file)
    store = NoteStore(tmp_path)
    loaded = store.load_entries(entries + entries[:1], tags=["imported"], skip_existing=True)
    assert [r["note_id"] for r in loaded] == ["2609.01234-1", "2609.00002-1"]  # a repeat in one call too
    assert store.load_entries(entries, tags=["imported"], skip_existing=True) == []  # imported again: nothing
    respaced = {**entries[0], "title": " Speculative  Decoding\twith Draft Trees "}  # the title, collapsed
    assert store.load_entries([respaced], skip_existing=True) == []
    # the same paper on another day or under another title is another entry
    later, renamed = {**entries[0], "today": "2026-10-08"}, {**entries[0], "title": "Draft trees, revisited"}
    assert [r["note_id"] for r in store.load_entries([later, renamed], skip_existing=True)] == [
        "2609.01234-2", "2609.01234-3"]
    # a note the write_note tool wrote counts as held when its paper, day and title are the entry's
    store.write(paper("2609.00003", "Tree attention"), {}, [], "2026-10-09")
    held = {**entries[1], "arxiv_id": "2609.00003", "title": "Tree attention", "today": "2026-10-09"}
    assert store.load_entries([held], skip_existing=True) == []
    assert sorted(p.name for p in tmp_path.glob("*.md")) == [
        "2609.00002-1.md", "2609.00003-1.md", "2609.01234-1.md", "2609.01234-2.md", "2609.01234-3.md"]
    # the default is unchanged: every entry is written, again
    assert [r["note_id"] for r in store.load_entries(entries)] == ["2609.01234-4", "2609.00002-2"]


def test_the_store_refuses_a_sqlite_without_fts5(tmp_path, monkeypatch):
    assert notes.fts5_available() is True
    monkeypatch.setattr(notes, "fts5_available", lambda: False)
    with pytest.raises(RuntimeError, match="FTS5"):
        NoteStore(tmp_path)


def test_the_fts5_probe_reports_false_when_the_virtual_table_cannot_be_created(monkeypatch):
    class NoFts:
        def execute(self, sql):
            raise sqlite3.OperationalError("no such module: fts5")

        def close(self):
            pass

    monkeypatch.setattr(notes.sqlite3, "connect", lambda *_: NoFts())
    assert notes.fts5_available() is False


# ---- note_tools --------------------------------------------------------------------------------------------

KNOWN = {"2609.01234": {"arxiv_id": "2609.01234", "title": "Draft trees", "published": "2026-09-02",
                        "abstract": "Decoding is slow. We propose draft trees. They are exact.",
                        "categories": ["cs.CL"]}}


def tools_over(tmp_path):
    return note_tools(NoteStore(tmp_path), KNOWN.get, TODAY)


def test_the_note_tools_keep_the_stand_ins_output_shapes(tmp_path):
    tools, stubs = tools_over(tmp_path), default_note_tools()
    assert set(tools) == set(stubs)
    args = {"arxiv_id": "2609.01234"}
    real, stub = tools["write_note"](args), stubs["write_note"](args)
    assert real.error is stub.error is None and set(real.content) == set(stub.content)
    assert real.content == {"note_id": "2609.01234-1", "arxiv_id": "2609.01234", "path": "2609.01234-1.md"}
    for name, call in (("search_notes", {"query": "diffusion"}), ("list_notes", {"tag": "cs.AI"})):
        assert tools[name](call) == stubs[name](call) == ToolOutcome([])  # empty: no error code
    found = tools["search_notes"]({"query": "draft", "top_k": 9}).content
    assert [h["note_id"] for h in found] == ["2609.01234-1"]
    assert tools["list_notes"]({}).content[0]["tags"] == ["cs.CL"]


def test_write_note_puts_abstract_evidence_into_the_scaffold(tmp_path):
    tools = tools_over(tmp_path)
    path = tools["write_note"]({"arxiv_id": "2609.01234", "new_terms": ["exact"]}).content["path"]
    text = (tmp_path / path).read_text(encoding="utf-8")
    assert "- 证据句(原文)\n  - Decoding is slow.\n  - We propose draft trees.\n  - They are exact.\n" in text


def test_write_note_answers_not_found_for_an_unknown_paper_and_invalid_arguments_like_the_stand_in(tmp_path):
    tools = tools_over(tmp_path)
    assert tools["write_note"]({"arxiv_id": "2609.99999"}) == ToolOutcome({}, schemas.NOT_FOUND)
    for args in ({}, {"arxiv_id": ""}, {"arxiv_id": "2609.01234", "mode": "other"}, {"arxiv_id": 7}):
        assert tools["write_note"](args) == ToolOutcome({}, schemas.INVALID_ARGUMENTS)
    assert list(tmp_path.glob("*.md")) == []


# ---- import_entries ----------------------------------------------------------------------------------------


def test_import_entries_round_trips_rendered_entries():
    owner_file = ("## 记录\n\n(从 2026-10-05 这周开始)\n\n" + SCAFFOLD + "\n" + DRAFT
                  + "\n## 别的\n- 新术语:x\n")
    first, second = import_entries(owner_file)
    assert first == {"today": TODAY, "title": "Speculative Decoding with Draft Trees",
                     "arxiv_id": "2609.01234",
                     "link": "https://arxiv.org/abs/2609.01234", "published": "2026-09-02", "type": "论文",
                     "relation": "", "problem": "", "method": "", "evidence": "", "limitations": "",
                     "evidence_sentences": EVIDENCE, "new_terms": ["draft tree", "acceptance rate"],
                     "open_terms": []}
    assert second["problem"] == "Autoregressive decoding is slow." and second["open_terms"] == ["KV cache"]
    assert second["evidence_sentences"] == [] and second["limitations"] == ""


def test_import_entries_tolerates_hand_written_entries_with_empty_fields():
    text = "### 2026-10-06 AI Index Report 2026\n\n- 来源:\n- 新术语:A、B,C\n  - Problem: stray\n"
    (entry,) = import_entries(text)
    assert entry["title"] == "AI Index Report 2026" and entry["arxiv_id"] is None and entry["link"] == ""
    assert entry["new_terms"] == ["A", "B", "C"] and entry["problem"] == ""

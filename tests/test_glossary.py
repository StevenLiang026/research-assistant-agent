"""The external zh-en glossary (docs/m7-本地提升计划.md, WP4): research_agent.glossary's loading, validation,
blocked terms, matching and note, and scripts/build_glossary.py's parsing, merging, filters, training-data
measurement and build. Offline: the sources are small text fixtures in the formats the downloads have (an AITD
markdown table, a tvly extract of the Google glossary); the real build is recorded in
data/glossary/provenance.json."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from research_agent.canonical import canonical_json
from research_agent.glossary import (
    MAX_TERMS,
    NOTE_PREFIX,
    Glossary,
    GlossaryError,
    apply_note,
    load_glossary,
    match_form,
    note_for,
    note_terms,
    noted_messages,
    term_form,
)

REPO = Path(__file__).resolve().parents[1]
TERMS = {"混合专家": "Mixture of Experts", "混合专家模型": "Mixture-of-Experts Model", "专家": "Expert",
         "推测解码": "Speculative Decoding", "解码器": "Decoder", "深度Q学习": "Deep Q-Learning",
         "KV缓存": "KV Cache", "知识蒸馏": "Knowledge Distillation", "剪枝": "Pruning",
         "量化": "Quantization", "注意力": "Attention", "强化学习": "Reinforcement Learning"}
SYSTEM = {"role": "system", "content": "你是科研助理,今天是 2026-10-10。"}


def history(*users: str) -> list[dict]:
    """A system message, then each user message followed by an assistant call and its tool result."""
    messages = [dict(SYSTEM)]
    for k, text in enumerate(users):
        messages += [{"role": "user", "content": text},
                     {"role": "assistant", "content": "", "tool_calls": [
                         {"id": f"c{k}", "type": "function",
                          "function": {"name": "search_arxiv", "arguments": "{\"query\": \"x\"}"}}]},
                     {"role": "tool", "tool_call_id": f"c{k}", "content": "{\"hits\": []}"}]
    return messages


GLOSSARY = Glossary(TERMS)


# research_agent.glossary: matching


def test_the_longest_term_starting_at_a_position_wins():
    longest = note_terms(history("有没有混合专家模型的新论文?"), GLOSSARY)
    assert longest == [("混合专家模型", "Mixture-of-Experts Model")]
    assert note_terms(history("混合专家的路由"), GLOSSARY) == [("混合专家", "Mixture of Experts")]


def test_matches_do_not_overlap_and_the_leftmost_match_is_taken_first():
    # 推测解码 starts first and is taken whole; 解码器 overlaps it and is not matched
    assert note_terms(history("推测解码器"), GLOSSARY) == [("推测解码", "Speculative Decoding")]
    assert note_terms(history("解码器与推测解码"), GLOSSARY) == [("解码器", "Decoder"),
                                                         ("推测解码", "Speculative Decoding")]


def test_at_most_five_terms_in_order_of_appearance_each_once():
    text = "剪枝、量化、剪枝、知识蒸馏、注意力、强化学习、推测解码"
    terms = note_terms(history(text), GLOSSARY)
    assert MAX_TERMS == 5
    # 剪枝 once, and 推测解码 after the fifth term is cut
    assert [zh for zh, _ in terms] == ["剪枝", "量化", "知识蒸馏", "注意力", "强化学习"]


def test_only_the_latest_user_message_is_matched():
    messages = history("混合专家的论文", "再找推测解码的")
    assert note_terms(messages, GLOSSARY) == [("推测解码", "Speculative Decoding")]
    assert note_terms(messages[:4], GLOSSARY) == [("混合专家", "Mixture of Experts")]  # before the second
    system_with_term = [{"role": "system", "content": "量化"}, {"role": "user", "content": "你好"}]
    assert note_terms(system_with_term, GLOSSARY) == []  # the system message is never matched
    after_tool = history("剪枝") + [{"role": "assistant", "content": "量化和知识蒸馏也相关。"}]
    assert note_terms(after_tool, GLOSSARY) == [("剪枝", "Pruning")]  # an assistant message is not matched
    assert note_terms([dict(SYSTEM)], GLOSSARY) == [] and note_for([dict(SYSTEM)], GLOSSARY) is None


def test_the_message_is_compared_in_match_form_and_the_note_shows_the_glossarys_own_term():
    assert note_terms(history("深度 Q 学习"), GLOSSARY) == [("深度Q学习", "Deep Q-Learning")]
    assert note_terms(history("kv 缓存怎么压缩"), GLOSSARY) == [("KV缓存", "KV Cache")]
    assert note_terms(history("ＫＶ缓存"), GLOSSARY) == [("KV缓存", "KV Cache")]  # full-width letters
    assert note_terms(history("Find papers on speculative decoding"), GLOSSARY) == []
    assert term_form(" 深度 Q 学习 ") == "深度Q学习" and term_form("偏差 - 方差分解") == "偏差-方差分解"
    assert term_form("Gram 矩阵") == "Gram矩阵" and term_form("a  b 矩阵") == "a b矩阵"
    assert match_form("KV 缓存") == "kv缓存"


def test_the_note_is_one_line_in_the_plans_format():
    note = note_for(history("推测解码和 KV 缓存"), GLOSSARY)
    assert note == "术语对照:推测解码 = Speculative Decoding;KV缓存 = KV Cache"
    assert note.startswith(NOTE_PREFIX) and "\n" not in note
    assert note_for(history("今天天气不错"), GLOSSARY) is None


def test_apply_note_appends_the_note_on_a_line_of_its_own():
    note = "术语对照:剪枝 = Pruning"
    assert apply_note("你是科研助理。", note) == "你是科研助理。\n术语对照:剪枝 = Pruning"
    assert apply_note("你是科研助理。", None) == "你是科研助理。"
    with pytest.raises(GlossaryError, match="one line"):
        apply_note("你是科研助理。", "术语对照:剪枝 = Pruning\n更多")


def test_noted_messages_puts_the_note_on_the_system_message_of_a_copy():
    messages = history("找剪枝的论文")
    before = copy.deepcopy(messages)
    noted, note = noted_messages(messages, GLOSSARY)
    assert note == "术语对照:剪枝 = Pruning" and messages == before  # the input is not changed
    assert noted[0] == {"role": "system", "content": SYSTEM["content"] + "\n" + note}
    assert noted[1:] == messages[1:]
    unchanged, none = noted_messages(history("你好"), GLOSSARY)
    assert none is None and unchanged == history("你好")
    with pytest.raises(GlossaryError, match="system message"):
        noted_messages([{"role": "user", "content": "剪枝"}], GLOSSARY)


# research_agent.glossary: the Latin boundary


def test_a_match_does_not_start_or_end_inside_a_latin_word():
    glossary = Glossary({"E步": "Expectation Step", "M步": "Maximization Step", "步骤": "Step",
                         "KV缓存": "KV Cache", "3D卷积": "3D Convolution", "图G": "Graph G"})
    # E步 would start inside TYPE; the shorter 步骤 after it still matches
    assert glossary.terms_in("TYPE步骤") == ["步骤"]
    assert glossary.terms_in("E步和M步") == ["E步", "M步"]
    assert glossary.terms_in("用 E 步更新") == ["E步"]  # the term form drops the spaces by a Han character
    assert glossary.terms_in("QKV缓存") == [] and glossary.terms_in("Q-KV缓存") == ["KV缓存"]
    assert glossary.terms_in("13D卷积") == [] and glossary.terms_in("1 3D卷积") == ["3D卷积"]
    assert glossary.terms_in("图GPU") == [] and glossary.terms_in("图G上的游走") == ["图G"]  # the end, too
    assert glossary.terms_in("ＴＹＰＥ步骤") == ["步骤"]  # in match form: full-width letters are ASCII ones


def test_a_term_rejected_at_the_latin_boundary_lets_a_shorter_one_at_the_same_position_match():
    glossary = Glossary({"图G": "Graph G", "图": "Graph"})
    assert glossary.terms_in("图GPU") == ["图"]  # 图G ends inside GPU; 图 starts at the same position
    assert Glossary({"E步": None, "步骤": "Step"}).terms_in("TYPE步骤") == ["步骤"]  # a blocked term too


# research_agent.glossary: blocked terms (a term whose English term is null)


BLOCKING = {"训练数据": "Training Data", "学习": "Learning", "剪枝": "Pruning", "预训练": None,
            "持续学习": None}


def test_a_blocked_term_takes_its_span_and_gives_no_note():
    blocking = Glossary(BLOCKING)
    # 预训练 starts first and is taken whole, so 训练数据 inside 预训练数据 cannot match there
    assert blocking.terms_in("预训练数据的综述") == []
    assert blocking.terms_in("持续学习里的剪枝") == ["剪枝"]
    assert blocking.terms_in("训练数据与学习") == ["训练数据", "学习"]  # outside a blocked span they match
    assert note_for(history("预训练数据的综述"), blocking) is None
    deleted = Glossary({"训练数据": "Training Data", "学习": "Learning"})  # the blocked terms left out
    assert deleted.terms_in("预训练数据的综述") == ["训练数据"] and deleted.terms_in("持续学习") == ["学习"]


def test_a_glossary_maps_its_noted_terms_and_lists_its_blocked_terms_apart():
    blocking = Glossary(BLOCKING)
    noted = {"训练数据": "Training Data", "学习": "Learning", "剪枝": "Pruning"}
    assert dict(blocking) == noted and len(blocking) == 3 and "预训练" not in blocking
    assert blocking.blocked == ("持续学习", "预训练")  # sorted
    assert blocking.entries() == BLOCKING  # what its file holds
    assert Glossary(TERMS).blocked == () and Glossary(TERMS).entries() == TERMS
    # two glossaries are equal when they note and block the same terms
    assert Glossary(BLOCKING) == blocking != Glossary(noted)


def test_blocked_spans_take_no_place_among_the_five_terms():
    glossary = Glossary({**TERMS, "预训练": None, "持续学习": None})
    text = "预训练、剪枝、持续学习、量化、知识蒸馏、注意力、强化学习"
    assert [zh for zh, _ in note_terms(history(text), glossary)] == ["剪枝", "量化", "知识蒸馏", "注意力",
                                                                     "强化学习"]


# research_agent.glossary: validation and loading


@pytest.mark.parametrize(("terms", "error"), [
    (["剪枝", "Pruning"], "object"),
    ({"pruning": "Pruning"}, "Chinese"),
    ({" 剪枝": "Pruning"}, "term form"),
    ({"剪枝": ""}, "empty"),
    ({"剪枝": "Pruning;Cutting"}, "separator"),
    ({"剪=枝": "Pruning"}, "separator"),
    ({"剪枝": "Pruning\nCutting"}, "one line"),
    ({"剪枝": "  Pruning"}, "one line"),
    ({"剪枝": "剪枝法"}, "Chinese"),
    ({"剪枝": 3}, "string"),
    ({"剪枝": ["Pruning"]}, "string"),
    ({"KV缓存": "KV Cache", "kv缓存": "kv cache"}, "same match form"),
    ({"pretraining": None}, "Chinese"),  # a blocked term is a Chinese term like any other
    ({" 预训练": None}, "term form"),
    ({"预=训练": None}, "separator"),
    ({"KV缓存": "KV Cache", "kv缓存": None}, "same match form"),  # a blocked term and a noted one
])
def test_a_malformed_glossary_is_refused_naming_the_term(terms, error):
    with pytest.raises(GlossaryError, match=error):
        Glossary(terms)


def test_load_glossary_reads_and_validates_a_json_object(tmp_path):
    path = tmp_path / "zh-en.json"
    path.write_text(canonical_json(TERMS) + "\n", encoding="utf-8")
    glossary = load_glossary(path)
    assert dict(glossary) == TERMS and len(glossary) == len(TERMS) and glossary["剪枝"] == "Pruning"
    path.write_text("{oops", encoding="utf-8")
    with pytest.raises(GlossaryError, match=r"zh-en\.json.*not JSON"):
        load_glossary(path)
    path.write_text(json.dumps({"pruning": "Pruning"}), encoding="utf-8")
    with pytest.raises(GlossaryError, match=r"zh-en\.json.*pruning"):
        load_glossary(path)


def test_load_glossary_reads_a_null_english_term_as_a_blocked_term(tmp_path):
    path = tmp_path / "zh-en.json"
    path.write_text(canonical_json({"训练数据": "Training Data", "剪枝": "Pruning", "预训练": None}) + "\n",
                    encoding="utf-8")
    glossary = load_glossary(path)
    assert glossary.blocked == ("预训练",)
    assert dict(glossary) == {"训练数据": "Training Data", "剪枝": "Pruning"}
    assert glossary.terms_in("预训练数据和剪枝") == ["剪枝"]
    path.write_text(json.dumps({"预训练": None, "pretraining": None}), encoding="utf-8")
    with pytest.raises(GlossaryError, match=r"zh-en\.json.*pretraining"):
        load_glossary(path)


def test_a_plain_mapping_is_matched_like_a_glossary():
    assert note_for(history("剪枝"), {"剪枝": "Pruning"}) == "术语对照:剪枝 = Pruning"
    assert note_for(history("预训练数据"), {"训练数据": "Training Data", "预训练": None}) is None


# scripts/build_glossary.py


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"script_{name}", REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load_script("build_glossary")

# Third-party text: the AITD rows of AITD_TABLE, SEPARATOR_TABLE, SPLIT_TABLE and BUILD_TABLE are taken from,
# or modelled on, the Artificial Intelligence Terminology Database by 机器之心 (Synced) and its contributors,
# https://github.com/jiqizhixin/Artificial-Intelligence-Terminology-Database (commit d4ec80fd), licensed
# CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/); rows were changed or made up to test
# the parser. These lines are under that licence, not under the repository's MIT licence.
AITD_TABLE = """# Letter All
[*Return*](https://example.org/README.md)
索引编号|英文术语|中文翻译|常用缩写|来源&扩展|备注
---|---|---|---|---|---
AITD-00003|Accept-Reject Sampling Method|接受-拒绝抽样法/接受-拒绝采样法||[1]||
AITD-00007|Accuracy|准确率||[1]||
AITD-00021|Adaptive Boosting|AdaBoost||[1]||
AITD-00147|Bias|偏差/偏置||[1]||
AITD-00148|Bias In Affine Function|偏置||[1]||
AITD-00149|Bias In Statistics|偏差 ||[1]||
AITD-00611|Encoder-Decoder|编码器-解码器（模型）||[1]||
AITD-01482|Precision|准确率||[1]||
AITD-02000|Deep Q-Learning|深度 Q 学习||[1]||
AITD-02442|K-Edge (O-K Edge)|K-边缘（O-K 边缘）||[1]||
"""


def test_aitd_rows_are_parsed_with_their_variants_split_and_cleaned():
    rows = build.parse_aitd(AITD_TABLE, "All.md")
    assert [r.index for r in rows][:3] == ["AITD-00003", "AITD-00007", "AITD-00021"]
    first = rows[0]
    assert (first.en, first.zh) == ("Accept-Reject Sampling Method", ("接受-拒绝抽样法", "接受-拒绝采样法"))
    by_index = {r.index: r for r in rows}
    assert by_index["AITD-00021"].zh == ()  # "AdaBoost": no Chinese character, no term
    assert by_index["AITD-00149"].zh == ("偏差",)  # trailing space trimmed
    assert by_index["AITD-00611"].zh == ("编码器-解码器",)  # the parenthesised aside is dropped
    assert by_index["AITD-02000"].zh == ("深度Q学习",)  # term form
    assert by_index["AITD-02442"].en == "K-Edge" and by_index["AITD-02442"].zh == ("K-边缘",)
    with pytest.raises(ValueError, match=r"All\.md: line 5"):
        build.parse_aitd(AITD_TABLE.replace("AITD-00003|", "AITD-00003|extra|"), "All.md")


SEPARATOR_TABLE = """索引编号|英文术语|中文翻译|常用缩写|来源&扩展|备注
---|---|---|---|---|---
AITD-02248|Findable, Accessible, Interoperable, Reusable|可查找、可访问、可互操作、可重用||[1]||
AITD-02257|Garbage In, Garbage Out|无用数据入、无用数据出||[1]||
AITD-02360|Mechanical Sympathy|机械同感，软硬件协同编程||[1]||
AITD-02361|Separators|甲乙、丙丁;戊己；庚辛／壬癸,子丑/甲乙（别名，旧称）||[1]||
AITD-02400|Named Entity Recognition，NER|命名实体识别||[1]||
"""


def test_a_chinese_cell_is_split_at_every_variant_separator_unless_its_english_cell_is_a_comma_list():
    by_index = {r.index: r for r in build.parse_aitd(SEPARATOR_TABLE, "All.md")}
    assert by_index["AITD-02360"].zh == ("机械同感", "软硬件协同编程")  # the full-width comma, after NFKC
    # / , 、 ; and their full-width forms; the comma inside the aside goes with it; 甲乙 once
    assert by_index["AITD-02361"].zh == ("甲乙", "丙丁", "戊己", "庚辛", "壬癸", "子丑")
    # an English comma list: the Chinese cell lists the same items and stays one term
    assert by_index["AITD-02248"].zh == ("可查找、可访问、可互操作、可重用",)
    assert by_index["AITD-02257"].zh == ("无用数据入、无用数据出",)
    assert by_index["AITD-02400"].en == "Named Entity Recognition,NER"
    assert build.comma_list_rows(list(by_index.values())) == [
        {"index": "AITD-02248", "en": "Findable, Accessible, Interoperable, Reusable",
         "zh": ["可查找、可访问、可互操作、可重用"]},
        {"index": "AITD-02257", "en": "Garbage In, Garbage Out", "zh": ["无用数据入、无用数据出"]},
        {"index": "AITD-02400", "en": "Named Entity Recognition,NER", "zh": ["命名实体识别"]}]


def test_a_term_with_several_english_terms_in_one_source_takes_the_lowest_index():
    rows = build.parse_aitd(AITD_TABLE, "All.md")
    terms, conflicts = build.source_terms(rows)
    assert terms["准确率"] == "Accuracy"  # AITD-00007 before AITD-01482 (Precision)
    assert terms["偏置"] == "Bias" and terms["偏差"] == "Bias"  # AITD-00147 before 00148 and 00149
    assert conflicts == 3  # 准确率, 偏差 and 偏置 had more than one English term
    assert "AdaBoost" not in terms.values()


# Third-party text: a few headings and one definition of Google's Machine Learning Glossary,
# https://developers.google.com/machine-learning/glossary, content licensed CC BY 4.0
# (https://creativecommons.org/licenses/by/4.0/), shortened and changed to test the parser.
GOOGLE_EXTRACT = {"results": [{"url": "https://developers.google.com/machine-learning/glossary?hl=zh-cn",
                               "raw_content": "# 机器学习术语表\n\n## A\n\n## 准确率 (accuracy)\n\n"
                                              "正确预测的比例。\n\n## 消融 (ablation)\n\n## ablation\n\n"
                                              "## 偏差（伦理/公平性） (bias (ethics/fairness))\n\n"
                                              "## 剪枝 (pruning)\n"}],
                  "failed_results": []}


def test_the_google_glossary_headings_give_zh_en_pairs_and_english_only_headings_none():
    pairs, headings = build.parse_google(json.dumps(GOOGLE_EXTRACT, ensure_ascii=False))
    assert pairs == [("准确率", "accuracy"), ("消融", "ablation"), ("偏差", "bias"), ("剪枝", "pruning")]
    assert headings == 6  # the level-2 headings: A, the four terms and the English "ablation"
    english = {"results": [{"raw_content": "## A\n\n## ablation\n\n## ablation\n\n## accuracy\n"}]}
    assert build.parse_google(json.dumps(english)) == ([], 4)
    with pytest.raises(ValueError, match="results"):
        build.parse_google(json.dumps({"results": [], "failed_results": [{"url": "x", "error": "403"}]}))
    split = {"results": [{"raw_content": "## 正则化、规范化 (regularization)\n"}]}
    assert build.parse_google(json.dumps(split, ensure_ascii=False)) == (
        [("正则化", "regularization"), ("规范化", "regularization")], 1)  # the AITD cells' separators


def test_the_union_of_the_sources_lets_the_first_source_win_on_conflict():
    merged, report = build.merge([("aitd", {"准确率": "Accuracy", "剪枝": "Pruning"}),
                                  ("google", {"准确率": "precision", "消融": "ablation", "剪枝": "pruning"})])
    assert merged == {"准确率": "Accuracy", "剪枝": "Pruning", "消融": "ablation"}
    # 准确率 conflicts (the first source's Accuracy stays); 剪枝 is the same term in another letter case
    assert report == [{"source": "aitd", "terms": 2, "added": 2, "conflicts": 0, "same": 0},
                      {"source": "google", "terms": 3, "added": 1, "conflicts": 1, "same": 1}]


def training_record(rid: str, user: str, queries: list[str], lang: str = "zh") -> dict:
    """An agent-history record with one request and a search_arxiv call per query."""
    messages = [dict(SYSTEM), {"role": "user", "content": user}]
    for k, query in enumerate(queries):
        arguments = json.dumps({"query": query})
        function = {"name": "search_arxiv", "arguments": arguments}
        messages += [{"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{k}", "type": "function", "function": function}]},
            {"role": "tool", "tool_call_id": f"c{k}", "content": "{\"hits\": []}"}]
    messages.append({"role": "assistant", "content": "done"})
    return {"id": rid, "lang": lang, "history": "agent", "messages": messages}


RECORDS = [
    training_record("r1", "找推测解码的新论文", ["speculative decoding"]),
    training_record("r2", "推测解码 加速", ["speculative-decoding speedup"]),  # hyphen as space: agrees
    training_record("r3", "模型剪枝", ["model pruning"]),
    training_record("r4", "剪枝方法", ["network sparsification"]),  # disagrees
    training_record("r5", "剪枝", ["sparsity"]),  # disagrees
    training_record("r6", "剪枝的综述", ["pruning survey"]),
    training_record("r7", "帮我写剪枝的笔记", []),  # no query: not judged
    training_record("r8", "Find speculative decoding papers", ["speculative decoding"], lang="en"),
    training_record("r9", "今天的论文", ["new papers"]),  # no term: no note
]


def test_the_measurement_compares_each_noted_english_term_with_the_teachers_queries():
    glossary = Glossary({"推测解码": "Speculative Decoding", "剪枝": "Pruning"})
    m = build.measure(glossary, RECORDS)
    assert (m["records"], m["records_noted"], m["pairs"], m["pairs_judged"], m["pairs_agreeing"]) == (
        8, 7, 7, 6, 4)  # r8 is English; r9 has no term; r7 has no query
    assert m["agreement"] == round(4 / 6, 4)
    assert m["terms"]["剪枝"] == {"noted": 5, "records": 5, "judged": 4, "agreeing": 2}
    assert m["terms"]["推测解码"] == {"noted": 2, "records": 2, "judged": 2, "agreeing": 2}


def test_a_terms_records_count_each_record_once_however_many_of_its_requests_note_it():
    record = training_record("t1", "证据", [])
    record["messages"] += [{"role": "user", "content": "再找证据"}, {"role": "assistant", "content": "好"}]
    m = build.measure(Glossary({"证据": "Evidence"}), [record])
    assert (m["units"], m["units_noted"], m["records_noted"]) == (2, 2, 1)
    assert m["terms"]["证据"] == {"noted": 2, "records": 1, "judged": 0, "agreeing": 0}


def test_a_noted_english_term_agrees_with_a_query_holding_it_as_whole_words():
    assert build.agrees("Speculative Decoding", ["speculative-decoding speedup"])  # the S2 hyphen as space
    assert build.agrees("Speculative Decoding", ["survey: speculative decoding, 2025"])  # punctuation bounds
    assert build.agrees("Mixture of Experts", ["混合专家mixture of experts"])  # so does a Han character
    assert build.agrees("Ｋ-Means", ["K means clustering"])  # NFKC and casefold, both sides
    # equal once spaces and hyphens are removed from both, still as whole words of the query
    assert build.agrees("Pre-Training", ["pretraining"])
    assert build.agrees("Pre-Training", ["pretraining data"])
    assert build.agrees("Pretraining", ["pre-training of LLMs"])
    assert build.agrees("Fine-Tuning", ["fine tuning of LLMs"])
    assert build.agrees("Fine-Tuning", ["finetuning"])
    # inside a longer word it does not agree
    assert not build.agrees("Training Data", ["pretraining data"])
    assert not build.agrees("RAG", ["fragment retrieval"]) and build.agrees("RAG", ["rag pipelines"])
    assert not build.agrees("GPT", ["gpt4 reasoning"]) and build.agrees("GPT", ["gpt-4 reasoning"])
    assert not build.agrees("Data", ["metadata"]) and not build.agrees("Pre-Training", ["pretrained models"])
    # a whole word of the query: 学习 = Learning agrees with "continual learning", so the span of
    # 持续学习 must be blocked for 学习 not to be noted there (and counted as agreeing)
    assert build.agrees("Learning", ["continual learning"])
    assert not build.agrees("Pruning", [])


def test_a_noted_english_term_agrees_with_the_regular_plural_of_its_last_word():
    assert build.agrees("Word Embedding", ["word embeddings"])  # + s
    assert build.agrees("Adversarial Example", ["adversarial examples"])
    assert build.agrees("Adversarial Example", ["adversarial examples for LLMs"])
    assert build.agrees("Query", ["queries"])  # a consonant and y: y becomes ies
    assert build.agrees("Batch", ["batches"])  # + es after s, x, z, sh or ch
    assert build.agrees("Bias", ["social biases"]) and build.agrees("Box", ["bounding boxes"])
    assert build.agrees("Epoch", ["training epochs"])  # after ch + s too: ch may be read as k
    assert build.agrees("Key", ["keys"]) and not build.agrees("Key", ["kies"])  # a vowel and y: + s
    assert build.agrees("Mixture-of-Experts Model", ["mixture of experts models"])
    assert build.agrees("Query", ["query"]) and build.agrees("Batch", ["mini-batch"])  # the singular still
    assert build.agrees("Word Embedding", ["word-embeddings"])  # hyphens and spaces aside, as before
    # still on word boundaries
    assert not build.agrees("Training Data", ["pretraining data"])
    assert not build.agrees("Batch", ["batchesx"]) and not build.agrees("Query", ["queriesx"])
    assert not build.agrees("Data", ["datasets"]) and not build.agrees("Word Embedding", ["word embeddingss"])
    # only the last word takes a plural, and only a regular one: + es only after s, x, z, sh or ch
    assert not build.agrees("Word Embedding", ["words embedding"])
    assert not build.agrees("Plan", ["planes"]) and not build.agrees("Bias", ["biass"])
    assert not build.agrees("Query", ["querys"])


def test_the_filters_drop_short_terms_and_block_the_terms_training_data_shows_disagreeing():
    terms = {"推测解码": "Speculative Decoding", "剪枝": "Pruning", "图": "Graph"}
    kept, dropped = build.length_filter(terms, min_chars=2)
    assert kept == {"推测解码": "Speculative Decoding", "剪枝": "Pruning"} and dropped == ["图"]
    blocked, glossary = build.blocked_terms(Glossary(kept), RECORDS, min_support=3, max_agreement=0.5,
                                            generic_records=3)
    # judged 4 times, agreeing 2: not more than half
    assert blocked == {"剪枝": {"noted": 5, "records": 5, "judged": 4, "agreeing": 2,
                                "reasons": ["disagreement"]}}
    assert dict(glossary) == {"推测解码": "Speculative Decoding"} and glossary.blocked == ("剪枝",)
    none, same = build.blocked_terms(Glossary(kept), RECORDS, min_support=5, max_agreement=0.5,
                                     generic_records=3)
    assert none == {} and same == Glossary(kept)  # too little evidence to block anything


def test_a_blocked_term_keeps_its_span_so_no_shorter_term_inside_it_is_noted():
    terms = {"持续学习": "Continuous Learning", "学习": "Learning"}
    records = [training_record(f"c{k}", "持续学习的新论文", ["continual learning"]) for k in range(3)]
    blocked, glossary = build.blocked_terms(Glossary(terms), records, min_support=2, max_agreement=0.5,
                                            generic_records=3)
    assert list(blocked) == ["持续学习"] and blocked["持续学习"]["reasons"] == ["disagreement", "generic"]
    assert glossary.entries() == {"持续学习": None, "学习": "Learning"}
    # 学习 is not noted inside the blocked span: had 持续学习 been deleted, 学习 = Learning would be noted in
    # all three records and counted as agreeing with "continual learning"
    m = build.measure(glossary, records)
    assert (m["records_noted"], m["pairs"], m["terms"]) == (0, 0, {})


def test_blocking_is_repeated_until_no_noted_term_meets_a_rule():
    # the five-term cap: once the first five terms are blocked their spans take no place among the five,
    # so the sixth term is noted, judged and blocked in its turn
    text = "剪枝、量化、知识蒸馏、注意力、强化学习、推测解码"
    records = [training_record(f"s{k}", text, ["sparsity"]) for k in range(3)]
    blocked, glossary = build.blocked_terms(GLOSSARY, records, min_support=2, max_agreement=0.5,
                                            generic_records=3)
    order = list(blocked)
    assert set(order[:5]) == {"剪枝", "量化", "知识蒸馏", "注意力", "强化学习"} and order[5:] == ["推测解码"]
    assert set(glossary.blocked) == set(order) and build.measure(glossary, records)["pairs"] == 0


GENERIC = [
    training_record("g1", "帮我整理证据", []),  # 证据, not judged
    training_record("g2", "证据的笔记", []),
    training_record("g3", "找证据", ["proof"]),  # 证据, judged once, disagrees
    training_record("g4", "假设检验", []),  # 假设, in two records only
    training_record("g5", "假设的笔记", []),
    training_record("g6", "剪枝的笔记", []),  # 剪枝, agreeing once below
    training_record("g7", "剪枝", []),
    training_record("g8", "剪枝的综述", ["pruning survey"]),
    training_record("g9", "按类别找", ["category"]),  # 类别, judged three times, never agreeing
    training_record("g10", "类别", ["categories"]),
    training_record("g11", "类别不平衡", ["imbalanced categories"]),
]
GENERIC_TERMS = {"证据": "Evidence", "假设": "Hypothesis", "剪枝": "Pruning", "类别": "Class"}


def test_a_term_noted_in_three_records_that_agrees_in_none_is_blocked_as_generic():
    blocked, glossary = build.blocked_terms(Glossary(GENERIC_TERMS), GENERIC, min_support=2,
                                            max_agreement=0.5, generic_records=3)
    assert blocked == {
        "类别": {"noted": 3, "records": 3, "judged": 3, "agreeing": 0,
                 "reasons": ["disagreement", "generic"]},
        "证据": {"noted": 3, "records": 3, "judged": 1, "agreeing": 0, "reasons": ["generic"]}}
    assert dict(glossary) == {"假设": "Hypothesis", "剪枝": "Pruning"}  # two records; agreeing once
    assert glossary.blocked == ("类别", "证据")
    # the generic rule is not the stop rule: it applies without one (min_support None), and its threshold
    # counts records
    alone, _ = build.blocked_terms(Glossary(GENERIC_TERMS), GENERIC, min_support=None, max_agreement=0.5,
                                   generic_records=3)
    assert {zh: c["reasons"] for zh, c in alone.items()} == {"类别": ["generic"], "证据": ["generic"]}
    two, _ = build.blocked_terms(Glossary(GENERIC_TERMS), GENERIC, min_support=None, max_agreement=0.5,
                                 generic_records=2)
    assert set(two) == {"类别", "证据", "假设"}


def test_the_filters_apply_the_length_filter_then_block_by_both_rules():
    terms = {**GENERIC_TERMS, "图": "Graph"}
    glossary, short, blocked = build.filtered(terms, GENERIC, min_chars=2, min_support=2, max_agreement=0.5,
                                              generic_records=3)
    assert short == ["图"] and list(blocked) == ["类别", "证据"]
    assert glossary.entries() == {"证据": None, "假设": "Hypothesis", "剪枝": "Pruning", "类别": None}
    # per rule, and per combination of rules (four rules: "both" would not say which two)
    assert build.blocked_by_reason(blocked) == {
        "disagreement": 1, "generic": 2, "adjectival": 0, "split_short": 0,
        "combinations": {"disagreement+generic": 1, "generic": 1}}
    unblocked, _, none = build.filtered(terms, GENERIC, min_chars=2, min_support=None, max_agreement=0.5,
                                        generic_records=None)
    assert none == {} and unblocked.blocked == ()  # no stop rule and no generic rule


ADJECTIVAL_TERMS = {"线性的": "Linear", "基于梯度的": "Gradient-Based", "可微的": "Differentiable",
                    "梯度": "Gradient", "剪枝": "Pruning"}
ADJECTIVAL = [
    training_record("d1", "基于梯度的剪枝", ["gradient-based pruning"]),  # 基于梯度的 agrees here
    training_record("d2", "基于梯度的剪枝方法", ["gradient based pruning methods"]),
    training_record("d3", "这个变换是线性的吗", []),  # 线性的, noted once and not judged
]


def test_a_chinese_term_ending_in_de_is_blocked_as_adjectival_whatever_the_training_records_show():
    blocked, glossary = build.blocked_terms(Glossary(ADJECTIVAL_TERMS), ADJECTIVAL, min_support=2,
                                            max_agreement=0.5, generic_records=3, adjectival=True)
    # 基于梯度的 agrees in both records and is blocked all the same; 可微的 is never noted and blocked too
    assert blocked == {
        "可微的": {"noted": 0, "records": 0, "judged": 0, "agreeing": 0, "reasons": ["adjectival"]},
        "基于梯度的": {"noted": 2, "records": 2, "judged": 2, "agreeing": 2, "reasons": ["adjectival"]},
        "线性的": {"noted": 1, "records": 1, "judged": 0, "agreeing": 0, "reasons": ["adjectival"]}}
    assert glossary.entries() == {"线性的": None, "基于梯度的": None, "可微的": None, "梯度": "Gradient",
                                  "剪枝": "Pruning"}
    assert glossary.terms_in("基于梯度的剪枝") == ["剪枝"]  # a blocked span: 梯度 inside it is not noted
    assert build.is_adjectival("基于梯度的") and not build.is_adjectival("梯度")
    # the rule is off unless asked for, and the length filter goes first
    none, _ = build.blocked_terms(Glossary(ADJECTIVAL_TERMS), ADJECTIVAL, min_support=2, max_agreement=0.5,
                                  generic_records=3)
    assert none == {}
    _, short, blocked = build.filtered(ADJECTIVAL_TERMS, ADJECTIVAL, min_chars=4, min_support=2,
                                       max_agreement=0.5, generic_records=3, adjectival=True)
    assert short == ["剪枝", "可微的", "梯度", "线性的"] and list(blocked) == ["基于梯度的"]


SPLIT_TABLE = """索引编号|英文术语|中文翻译|常用缩写|来源&扩展|备注
---|---|---|---|---|---
AITD-00147|Bias|偏差/偏置||[1]||
AITD-00148|Bias In Affine Function|偏置||[1]||
AITD-00565|Estimator|估计/估计量||[1]||
AITD-00954|Label|标签/标记||[1]||
AITD-01226|Moral Graph|端正图/道德图||[1]||
AITD-02130|Bit Collisions|字节碰撞/冲突||[1]||
AITD-02131|Hash Conflict|哈希碰撞、冲突||[1]||
AITD-02200|Scale|尺度/缩放/规模||[1]||
"""


def test_split_short_candidates_are_two_character_later_variants_of_multi_variant_cells_alone():
    rows = build.parse_aitd(SPLIT_TABLE, "All.md")
    # 冲突: the second variant of two cells; 缩放 and 规模: later variants of one cell; 标记: the second
    # variant of AITD-00954. Not: 偏置 (AITD-00148 gives it alone), 估计, 标签, 偏差 and 尺度 (first
    # variants), 道德图 (three characters)
    assert build.split_short_candidates(rows) == {"冲突", "标记", "缩放", "规模"}
    assert build.split_short_candidates(rows, other_terms={"规模"}) == {"冲突", "标记", "缩放"}  # Google's


SPLIT_RECORDS = [
    training_record("p1", "给标记的数据找论文", ["labeled data"]),  # 标记 = Label: does not agree
    training_record("p2", "标记噪声", ["label noise"]),  # agrees
    training_record("p3", "缩放定律", ["scaling laws"]),  # 缩放 = Scale: does not agree
    training_record("p4", "估计", ["estimation"]),  # 估计 = Estimator: does not agree, a first variant
]


def test_a_split_short_term_is_blocked_unless_a_training_record_agrees_with_it():
    rows = build.parse_aitd(SPLIT_TABLE, "All.md")
    terms, _ = build.source_terms(rows)
    candidates = build.split_short_candidates(rows)
    blocked, glossary = build.blocked_terms(Glossary(terms), SPLIT_RECORDS, min_support=3, max_agreement=0.5,
                                            generic_records=3, split_short=candidates)
    # 冲突 and 规模 are never noted, 缩放 is noted and never agrees; 标记 agrees once and stays noted
    never = {"noted": 0, "records": 0, "judged": 0, "agreeing": 0}
    assert blocked == {"冲突": {**never, "reasons": ["split_short"]},
                       "缩放": {"noted": 1, "records": 1, "judged": 1, "agreeing": 0,
                                "reasons": ["split_short"]},
                       "规模": {**never, "reasons": ["split_short"]}}
    # a first variant is not a candidate, whatever the training records show
    assert glossary["标记"] == "Label" and glossary["估计"] == "Estimator"
    assert glossary.blocked == ("冲突", "缩放", "规模")
    # the rule is off without candidates
    assert build.blocked_terms(Glossary(terms), SPLIT_RECORDS, min_support=3, max_agreement=0.5,
                               generic_records=3)[0] == {}
    # the variants with their cells and outcomes, for provenance
    _, short, blocked = build.filtered(terms, SPLIT_RECORDS, min_chars=2, min_support=3, max_agreement=0.5,
                                       generic_records=3, split_short=candidates)
    listed = build.split_variants(rows, candidates, glossary, short, blocked)
    # in the order of their first cell (index order), then of their place in it
    assert [entry["zh"] for entry in listed] == ["偏差", "偏置", "估计", "标签", "标记", "冲突", "尺度",
                                                 "缩放", "规模"]
    by_term = {entry["zh"]: entry for entry in listed}
    assert by_term["冲突"] == {
        "zh": "冲突", "en": "Bit Collisions", "cells": [
            {"index": "AITD-02130", "en": "Bit Collisions", "zh": ["字节碰撞", "冲突"], "variant": 2},
            {"index": "AITD-02131", "en": "Hash Conflict", "zh": ["哈希碰撞", "冲突"], "variant": 2}],
        "alone_in": [], "candidate": True, "outcome": "blocked", "reasons": ["split_short"]}
    assert by_term["偏置"] == {
        "zh": "偏置", "en": "Bias", "cells": [{"index": "AITD-00147", "en": "Bias", "zh": ["偏差", "偏置"],
                                              "variant": 2}],
        "alone_in": ["AITD-00148"], "candidate": False, "outcome": "noted"}
    assert (by_term["标记"]["candidate"], by_term["标记"]["outcome"]) == (True, "noted")
    assert (by_term["估计"]["candidate"], by_term["估计"]["outcome"]) == (False, "noted")
    dropped = build.split_variants(rows, candidates, Glossary({}), ["冲突"], {})
    assert {entry["zh"]: entry["outcome"] for entry in dropped}["冲突"] == "dropped"


def test_the_sweep_lists_the_settings_tied_on_net_agreement_beside_the_tie_break():
    def line(min_chars: int, min_support: int | None, net: int, terms: int) -> dict:
        return {"min_chars": min_chars, "min_support": min_support, "terms": terms,
                "cross_fitted": {"net_agreement": net}}

    lines = [line(1, 2, 400, 30), line(2, 2, 414, 20), line(2, 3, 414, 22), line(2, 5, 414, 22),
             line(2, None, 412, 25)]
    # the tie-break: more terms, then the earlier line
    assert build.best_setting(lines) == {"min_chars": 2, "min_support": 3}
    assert build.tied_best(lines) == [{"min_chars": 2, "min_support": 2}, {"min_chars": 2, "min_support": 3},
                                      {"min_chars": 2, "min_support": 5}]


BUILD_TABLE = """# Letter All
索引编号|英文术语|中文翻译|常用缩写|来源&扩展|备注
---|---|---|---|---|---
AITD-00001|Speculative Decoding|推测解码||[1]||
AITD-00002|Pruning|剪枝||[1]||
AITD-00003|Label|标签/标记||[1]||
AITD-00004|Bit Collisions|字节碰撞/冲突||[1]||
AITD-00005|Linear|线性的||[1]||
AITD-00006|Word Embedding|词嵌入||[1]||
AITD-00007|Graph|图||[1]||
"""
BUILD_TRAINING = {
    "m6.jsonl": [training_record("a1", "找推测解码的新论文", ["speculative decoding"]),
                 training_record("a2", "词嵌入的综述", ["word embeddings survey"]),
                 training_record("a3", "标签噪声", ["label noise"]),
                 training_record("a4", "线性的模型", ["linear models"])],
    "m7.jsonl": [training_record("b1", "剪枝方法", ["network pruning"]),
                 training_record("b2", "两篇论文结论冲突", ["conflicting conclusions"]),
                 training_record("b3", "词嵌入", ["word embeddings"])],
}


def write_build_inputs(root: Path) -> list[str]:
    """An AITD clone, a Google extract and two training files under root, in the forms the build reads; the
    build's arguments, relative to root."""
    aitd, google = root / "aitd", root / "google"
    (aitd / "data").mkdir(parents=True)
    (aitd / ".git").mkdir()
    (aitd / "data" / "All.md").write_text(BUILD_TABLE, encoding="utf-8")
    (aitd / "README.md").write_text("# AITD\n\n## 许可证\n\n本作品采用知识共享署名-非商业性使用-相同方式共享 "
                                    "4.0 国际许可协议进行许可。\n", encoding="utf-8")
    (aitd / "LICENSE").write_text("# Attribution-NonCommercial-ShareAlike 4.0 International\n",
                                  encoding="utf-8")
    (aitd / ".git" / "HEAD").write_text("d" * 40 + "\n", encoding="utf-8")
    google.mkdir()
    content = ("# 机器学习术语表\n\n## 剪枝 (pruning)\n\n"
               "本页面中的内容已根据知识共享署名 4.0 许可获得了许可。\n\n最后更新时间 (UTC)：2026-04-10。\n")
    extract = json.dumps({"results": [{"raw_content": content}]})
    (google / "extract.json").write_text(extract, encoding="utf-8")
    for name, records in BUILD_TRAINING.items():
        (root / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                                 encoding="utf-8")
    return ["--aitd", "aitd", "--google", "google", "--fetched", "2026-10-30", "--train", *BUILD_TRAINING,
            "--out", "out"]


def test_the_build_blocks_by_its_four_rules_and_lists_the_split_variants_and_check_and_sweep_agree(
        tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    args = write_build_inputs(tmp_path)
    assert build.main(args) == 0
    printed = capsys.readouterr().out
    glossary = load_glossary(tmp_path / "out" / "zh-en.json")
    filters = json.loads((tmp_path / "out" / "provenance.json").read_text(encoding="utf-8"))["filters"]
    # 线性的 agrees in a4 and is blocked all the same; 冲突 (judged once, disagreeing) and 标记 (never noted)
    # come only from a split cell, as its second variant; 词嵌入 agrees with "word embeddings" twice
    assert {zh: counts["reasons"] for zh, counts in filters["blocked"].items()} == {
        "冲突": ["split_short"], "标记": ["split_short"], "线性的": ["adjectival"]}
    assert glossary.blocked == ("冲突", "标记", "线性的") and filters["dropped_short"] == ["图"]
    assert dict(glossary) == {"推测解码": "Speculative Decoding", "剪枝": "Pruning", "标签": "Label",
                              "字节碰撞": "Bit Collisions", "词嵌入": "Word Embedding"}
    assert filters["blocked_by_reason"] == {
        "disagreement": 0, "generic": 0, "adjectival": 1, "split_short": 2,
        "combinations": {"adjectival": 1, "split_short": 2}}
    assert filters["blocked_terms_by_reason"] == {"disagreement": [], "generic": [], "adjectival": ["线性的"],
                                                  "split_short": ["冲突", "标记"]}
    assert filters["split_short_rule"]["candidates"] == ["冲突", "标记"]
    # every two-character variant of a cell of several variants, with its cells and its outcome
    variants = filters["split_variants"]
    assert [(v["zh"], v["candidate"], v["outcome"], v["cells"][0]["index"]) for v in variants] == [
        ("标签", False, "noted", "AITD-00003"), ("标记", True, "blocked", "AITD-00003"),
        ("冲突", True, "blocked", "AITD-00004")]
    assert "adjectival 1, split_short 2" in printed
    # --check builds again and compares; --sweep prints the build's own sweep, the same rules applied
    assert build.main([*args, "--check"]) == 0
    assert capsys.readouterr().out == "zh-en.json: equal; provenance.json: equal\n"
    assert build.main([*args, "--sweep"]) == 0
    assert [json.loads(line) for line in capsys.readouterr().out.splitlines()] == filters["sweep"]


PROVENANCE = REPO / "data" / "glossary" / "provenance.json"


@pytest.mark.needs_data("data/glossary/provenance.json")  # the glossary data: left out of the public copy
def test_the_built_setting_is_the_sweeps_best_in_the_code_its_docstring_and_the_provenance():
    filters = json.loads(PROVENANCE.read_text(encoding="utf-8"))["filters"]
    setting = {"min_chars": build.MIN_CHARS, "min_support": build.MIN_SUPPORT}
    # the documented selection rule, applied to the recorded sweep, names the code's setting
    assert build.best_setting(filters["sweep"]) == filters["best"] == setting
    assert {"min_chars": filters["min_chars"], "min_support": filters["stop_rule"]["min_support"]} == setting
    assert filters["setting_is_best"] is True
    assert f"MIN_CHARS = {build.MIN_CHARS}" in build.__doc__
    assert f"MIN_SUPPORT = {build.MIN_SUPPORT}" in build.__doc__


@pytest.mark.needs_data("data/glossary/provenance.json", "data/glossary/zh-en.json")
def test_the_provenance_describes_the_glossary_file_beside_it():
    path = REPO / "data" / "glossary" / "zh-en.json"
    provenance, glossary = json.loads(PROVENANCE.read_text(encoding="utf-8")), load_glossary(path)
    record, filters = provenance["glossary"], provenance["filters"]
    assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (record["terms"], record["blocked"]) == (len(glossary), len(glossary.blocked))
    assert sorted(filters["blocked"]) == sorted(glossary.blocked)
    assert filters["blocked_terms_by_reason"] == {
        reason: sorted(zh for zh, counts in filters["blocked"].items() if reason in counts["reasons"])
        for reason in build.REASONS}
    # every two-character split variant, its outcome as the glossary file has it
    for variant in filters["split_variants"]:
        zh = variant["zh"]
        assert (variant["outcome"] == "noted") == (zh in glossary)
        assert (variant["outcome"] == "blocked") == (zh in glossary.blocked)
        assert variant["candidate"] == (zh in filters["split_short_rule"]["candidates"])
    assert filters["blocked_terms_by_reason"]["split_short"] == sorted(
        v["zh"] for v in filters["split_variants"] if "split_short" in v.get("reasons", []))


@pytest.mark.needs_data("data/glossary/provenance.json", "data/glossary/NOTICE.md")
def test_the_notice_names_the_sources_and_licences_the_glossary_was_built_from():
    sources = {s["name"]: s for s in json.loads(PROVENANCE.read_text(encoding="utf-8"))["sources"]}
    notice = (REPO / "data" / "glossary" / "NOTICE.md").read_text(encoding="utf-8")
    aitd, google = sources["aitd"], sources["google"]
    assert aitd["url"] in notice and aitd["commit"] in notice and aitd["licence"]["name"] in notice
    assert "https://creativecommons.org/licenses/by-nc-sa/4.0/" in notice
    # the Google glossary contributed no term, which the notice says
    assert google["url"] in notice and google["added"] == 0 and "CC BY 4.0" in notice
    assert "scripts/build_glossary.py" in notice  # the code that builds them, the repository's own


def test_the_glossary_file_is_canonical_json_with_a_final_newline(tmp_path):
    path = tmp_path / "zh-en.json"
    digest = build.write_glossary({"剪枝": "Pruning", "推测解码": "Speculative Decoding"}, path)
    data = path.read_bytes()
    assert data == (canonical_json({"推测解码": "Speculative Decoding", "剪枝": "Pruning"}) + "\n").encode()
    assert digest == hashlib.sha256(data).hexdigest() and b"\r" not in data
    assert dict(load_glossary(path)) == {"剪枝": "Pruning", "推测解码": "Speculative Decoding"}


def test_the_glossary_file_holds_a_blocked_term_with_a_null_english_term(tmp_path):
    glossary = Glossary({"剪枝": "Pruning", "持续学习": None})
    path = tmp_path / "zh-en.json"
    build.write_glossary(glossary.entries(), path)
    assert path.read_bytes() == '{"剪枝":"Pruning","持续学习":null}\n'.encode()  # sorted keys: 剪 before 持
    assert load_glossary(path) == glossary and load_glossary(path).blocked == ("持续学习",)


def test_raw_files_are_listed_with_their_sha256_relative_to_the_download_folder(tmp_path):
    folder = tmp_path / "jiqizhixin"
    (folder / "data").mkdir(parents=True)
    (folder / "data" / "All.md").write_text(AITD_TABLE, encoding="utf-8")
    (folder / "LICENSE").write_text("licence", encoding="utf-8")
    (folder / ".git").mkdir()
    (folder / ".git" / "HEAD").write_text("ref: refs/heads/master\n", encoding="utf-8")
    files = build.raw_files(folder)
    assert [f["path"] for f in files] == ["LICENSE", "data/All.md"]  # sorted, the clone's .git left out
    assert files[1]["sha256"] == hashlib.sha256((folder / "data" / "All.md").read_bytes()).hexdigest()
    assert files[1]["bytes"] == len((folder / "data" / "All.md").read_bytes())


def test_the_clone_commit_is_read_from_the_git_folder_without_running_git(tmp_path):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/master\n", encoding="utf-8")
    (git / "refs" / "heads" / "master").write_text("a" * 40 + "\n", encoding="utf-8")
    assert build.clone_commit(tmp_path) == "a" * 40
    (git / "refs" / "heads" / "master").unlink()
    (git / "packed-refs").write_text("# pack-refs\n" + "b" * 40 + " refs/heads/master\n", encoding="utf-8")
    assert build.clone_commit(tmp_path) == "b" * 40
    (git / "HEAD").write_text("c" * 40 + "\n", encoding="utf-8")  # a detached head
    assert build.clone_commit(tmp_path) == "c" * 40

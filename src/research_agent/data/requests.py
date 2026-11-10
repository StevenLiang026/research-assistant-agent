"""Request records for the synthetic generator (docs/plan.md, 微调数据 第 2-3 条 and 去重与防泄漏;
docs/modules/接口.md, 请求记录): twelve intents with Chinese and English templates, slot pools drawn from the
corpus manifest, the split fixed before anything is generated, dedup, the leakage audit and the writer. Every
step is deterministic from one seed, so the same seed, sizes and corpus manifest give byte-identical files.

Templates. TEMPLATES[intent][kind][lang] is a tuple of format strings with {placeholder} fields. The kinds are
"normal" (the ten tool-calling intents: a complete request), "clarify" (five of them: the same ask with a
required slot left out on purpose, no topic or no paper id, which slots["missing"] names in the
schemas.MISSING vocabulary), "parallel" (three of them: the request names two or three ids, so several
calls can go out in one step) and "no_call" (the two intents that need no tool). Every pool holds at least
six templates per language. family = "<intent>/<lang>/<index>", the index counting over the intent's
templates in that language in KINDS order, so a family names one template.

Scenarios (synth.SCENARIOS) map onto kinds (SCENARIO_KIND): normal and recover use the normal templates
(recover is the same request; the generator injects the fault), clarify the clarify templates, parallel the
parallel templates, no_call the no-call intents. The planned mix synth.PLANNED_MIX (55/15/10/15/5), or the
mix a build names instead, holds per split by construction (plan_counts: largest remainder, then half and
half by language), unless a cell cannot be filled, which is an error.

Clarify templates must be unanswerable without the slot they leave out: they refer to a specific topic or
paper the request does not name (那个方向, 我说的那篇, the paper I mentioned, the topic our group works
on), so a broad search cannot answer them, and they contain no topic, id or id list placeholder.

Chinese topics. Chinese templates name a topic by its Chinese name from the corpus manifest: {topic} renders
the name alone, {topic_paren} the name followed by one English phrase in parentheses (混合专家(mixture of
experts)); {topic_paren} is used in at most a third of the Chinese templates that name a topic and never in
English ones. The slot keeps an English phrase as the query's required words: the phrase shown, or the
topic's first phrase when only the Chinese name is shown. Counts of one are spelled out (一篇, one paper) and
templates avoid "at most" constructions. tidy() applies the repository's Chinese spacing (docs/ and the dry
run write Chinese with half-width , ? ! : ; ( ) and full-width 。 、): one space between a Chinese character
and a Latin letter or digit (9 月以来, scaffold 模式), none between Chinese characters and none on either side
of a punctuation mark (tool use(工具调用)的, 以来,write_note).

Split before generation (split_templates, split_topics):
- each template pool (intent, kind, lang) of six or more templates goes 4/1/1 to train/dev/test, drawn with
  the seed; the templates beyond six go to train. This departs on purpose from the brief's wording (4/1/1 per
  intent and language as a whole): a whole-intent draw could put all of an intent's clarify or parallel
  templates in train, leaving dev or test without a clarify or parallel template for that intent, while the
  per-kind draw keeps a clarify and a parallel template of every intent that has them in every split, and
  still gives each intent and language 4/1/1 per kind (so at least 1/1 overall in dev and test);
- the 20 topics go 14/3/3 (round(0.15 n) to dev and to test, the rest to train), drawn with the seed, and
  every paper id follows its topic: the manifest lists 15 distinct ids per topic, so the ids split 210/45/45
  and a request that names a topic and a paper names one of that topic's papers. A request only combines a
  template, a topic and ids of its own split, so a test request's wording, topic and ids never occur in train.

Slots (canonical values; the text renders them in one of several surface forms): topic (an English phrase of
the topic, the required words of a search query: query_contains_all(query, [slots["topic"]])) and
topic_id; date_from, date_to (ISO dates inside the recording window; natural phrases such as 9 月以来 or since
September record the first day of the month); since (list_notes' date); count (1-5); categories (a sorted
list from schemas.CATEGORIES); section (schemas.SECTIONS); arxiv_id or arxiv_ids (canonical ids; the text may
carry an "arXiv:" prefix); mode (scaffold or draft); tags (a list of one tag from TAGS); missing (clarify);
tool, param, tools (the no-call questions about the tools). Surface fillers (SURFACE: tone, occasion, day)
vary the wording and are not recorded.

Dedup (Deduper, dedup): an exact duplicate of a kept request after normalize_text is dropped. The near rule
(Jaccard >= NEAR_DUP over the sets of character 5-grams, SHINGLE, against any kept request) and the train rule
(a train request with Jaccard >= TRAIN_VS_EVAL against any dev or test request) compare masked texts
(mask_text): after normalize_text, arXiv ids with an optional "arXiv:" prefix and version become <id>, ISO
dates <date> and every other run of digits <n>. Without the mask two fillings of one template differ only
in their ids and dates, and those distinct shingles hide a repeated wording. Jaccard is exact over the shingle
sets (the plan's MinHash approximates it): an inverted index from shingle to kept requests counts the shared
shingles, so one offer costs O(s + P), s the candidate's shingle count and P the summed length of the postings
it touches, at worst O(N s) against N kept requests and O(N^2 s) for a whole build, which for a few thousand
requests of under 200 characters is a few seconds. generate() offers candidates as it makes them (test, then
dev, then train, so the train rule sees every evaluation request) and draws again until each cell is full.

Leakage audit (audit): families, topics (topic_id) and paper ids (arxiv_id, arxiv_ids) shared by any two
splits. Template audit (template_audit): any two templates of the same language (any intent, any kind) whose
families sit in different splits must have wording Jaccard < TEMPLATE_LIMIT over 5-gram shingles with the
placeholders removed (template_wording), since family ids alone do not stop a test template and a train
template of another intent or kind from sharing nearly all their words. The catalogue is worded so that every
same-language pair stays below the limit, so the audit holds for any seed; no sibling rule is needed.
write_requests raises a ValueError naming every overlap and writes nothing unless both audits pass. The
manifest records both results and corpus_manifest_sha256, the SHA-256 of the corpus manifest the slot pools
came from.

Incremental builds (docs/m7-计划.md §5 and WP-I 1) add requests to registered ones and keep their split: the
seed still fixes the split of templates, topics and paper ids, and five parameters of generate() change the
rest.
- draw_seed (default: the seed) seeds the draws within a split instead of the seed: the round robin over a
  cell's templates, the topic and the slots each request is filled with, and the order of the records.
- only names the splits to build; the sizes of the others become 0 (check_only).
- id_start gives a split's first id number, so new ids continue after existing ones (req-test-0401, ...);
  numbered_ids pads every id of a split to four digits or to its last number's width (check_id_start).
- preload names existing request files (load_preload, Preloaded). Their requests are registered with the
  Deduper before the first candidate, unchecked, and every rule then runs against them as against kept
  requests: the exact and the near rule whatever their split, the train rule from a new train request to
  every preloaded and new dev or test request. A new dev or test request within the train-vs-eval band of a
  preloaded train request is kept (only the near rule applies to that pair, as in a whole build, where the
  evaluation splits are filled first). The drops a preloaded request alone would have caused are counted
  apart (dedup_against_preloaded). Preloaded requests are not written, a new id that a preloaded file
  already holds is refused (check_ids: req-test-21 and req-test-0021 are one id), and the leakage audit and
  the template audit run over the preloaded and the new requests together; given the records, the template
  audit also requires every record's catalogue family to sit in the split the template split gives it and
  lists the families outside the catalogue (another tool's, such as scripts/transform_synth.py's m6/ ones),
  whose wording that tool audits.
- mix replaces synth.PLANNED_MIX (check_mix: known scenarios, shares of at least 0 that sum to 1, a scenario
  left out gets 0).
A build that leaves all five at their defaults writes the files it wrote before, byte for byte, and its
manifest has the same fields; any other build's manifest also records draw_seed, only, id_start, preload (per
file: path, SHA-256, requests and their splits), dedup_against_preloaded and, in planned_mix, the mix used.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import math
import random
import re
import string
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..canonical import canonical_json
from ..normalize import normalize_arxiv_id, normalize_text
from ..tools import schemas
from .synth import LANGS, PLANNED_MIX, REQUEST_KEYS, SCENARIOS, SPLITS

GENERATOR_VERSION = "requests-2026-10-13.2"
CORPUS_MANIFEST = Path("data/corpus/manifest.json")  # relative to the repository root, where scripts run
DEFAULT_SIZES = {"train": 1600, "dev": 400, "test": 400}
SHINGLE = 5
NEAR_DUP = 0.8
TRAIN_VS_EVAL = 0.6
TEMPLATE_LIMIT = 0.6  # wording Jaccard two templates of different splits must stay below
TEMPLATE_SPLIT = {"test": 1, "dev": 1}  # per template pool; everything else goes to train
SPLIT_SHARE = {"test": 0.15, "dev": 0.15}  # of the topics (and with them the paper ids)
MAX_ATTEMPTS_PER_REQUEST = 60  # draws per wanted request before a cell counts as unfillable
MIX_TOLERANCE = 1e-9  # how far the shares of a mix may sum from 1
YEAR = 2026

INTENTS = {
    "find_recent_papers": "find papers on a topic published since a date, a given number of them",
    "find_by_category": "find papers on a topic restricted to given arXiv categories",
    "read_section": "read one section of the most relevant paper on a topic or of a given id",
    "summarize_paper": "summarise a given paper",
    "compare_papers": "compare two (in parallel requests two or three) given papers",
    "write_note": "write a reading note for a given paper in scaffold or draft mode",
    "search_notes": "search the owner's notes for a topic, optionally by tag",
    "list_notes": "list recent notes, optionally by tag, date or number",
    "lookup_paper": "look up the metadata of a given id (two or three ids in parallel requests)",
    "whats_new": "follow up on what is new on a topic since a date",
    "explain_tools": "a question about the note template or the tools, answered without a call (no_call)",
    "chit_chat": "small talk or thanks (no_call)",
}
KINDS = ("normal", "clarify", "parallel", "no_call")
SCENARIO_KIND = {"normal": "normal", "recover": "normal", "clarify": "clarify", "parallel": "parallel",
                 "no_call": "no_call"}
TAGS = ("to-read", "survey", "baseline", "reproduce", "reading-group", "idea")
MODES = ("scaffold", "draft")
COUNTS = (1, 2, 3, 4, 5)
SURFACE = ("tone", "focus", "use", "aspect", "occasion", "day", "why", "fields")

TEMPLATES: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    "find_recent_papers": {
        "normal": {
            "zh": (
                "帮我找{since}关于 {topic} 的论文,挑 {n_papers}最相关的。",
                "{since}{topic}方向有哪些新论文?给我 {n_papers}就够了。",
                "我想跟进一下 {topic_paren},把{since}发表的论文找出来,{n_papers}就行。",
                "检索{since}讨论 {topic} 的 arXiv 论文,按相关度挑 {n_papers}给我。",
                "{day}的{occasion}要讲 {topic},帮我找 {n_papers}{since}的相关工作,附上编号。",
                "{topic}最近有什么值得读的?只要{since}的,{n_papers}就够。",
                "搜一下{between}关于 {topic_paren} 的论文,给我 {n_papers}。",
            ),
            "en": (
                "Find {n_papers} on {topic} published {since}.",
                "Give me {n_papers} on {topic} from arXiv {since}, chosen by relevance.",
                "I'm catching up on {topic}; pull {n_papers} submitted {since}, each with its arXiv id.",
                "Search arXiv for {topic} work {since} and send me {n_papers} worth reading.",
                "For {occasion} {day} I'm covering {topic}. Can you find {n_papers} posted {since}?",
                "Any new {topic} papers {since}? I only need {n_papers}.",
                "Look for papers about {topic} {between} and return {n_papers}.",
            ),
        },
        "clarify": {
            "zh": (
                "帮我找找{since}我们组在做的那个方向的论文,{n_papers}就行。",
                "上次聊的那个主题,{since}又出了哪些论文?挑 {n_papers}。",
                "{day}{occasion}要用,把我关注的那个课题{since}的论文找 {n_papers}出来。",
                "老师让我跟的那块内容,{since}有新论文吗?给我 {n_papers}。",
                "按我之前说的方向检索一下,{since}发表的,{n_papers}足够。",
                "就是我刚才提的那个话题,帮我搜 {n_papers}{since}的论文。",
            ),
            "en": (
                "Find {n_papers} on the topic our group works on, published {since}.",
                "That subject we talked about last time: what has come out {since}? Pick {n_papers}.",
                "For {occasion} {day}, dig up {n_papers} {since} in the area I follow.",
                "Any new papers {since} on the project my advisor gave me? I need {n_papers}.",
                "Search the direction I mentioned earlier for work posted {since}; send {n_papers}.",
                "Same field as before, please: {n_papers} that appeared {since}.",
            ),
        },
    },
    "find_by_category": {
        "normal": {
            "zh": (
                "只在 {categories} 里搜 {topic} 的论文,给我 {n_papers}。",
                "帮我找 {categories} 分类下关于 {topic_paren} 的工作,{n_papers}就够。",
                "限定 {categories},检索与{topic}相关的论文,{since}的优先。",
                "{topic}的论文我只看 {categories} 的,帮我筛 {n_papers}。",
                "在 arXiv 的 {categories} 类别里查一下 {topic},只要{since}的。",
                "我只关心 {categories} 方向,讲 {topic} 的论文给我找 {n_papers}。",
            ),
            "en": (
                "Search only {categories} for papers on {topic}; {n_papers} is enough.",
                "Find {n_papers} on {topic}, restricted to {categories}.",
                "I only follow {categories}. Which {topic} papers are there? Give me {n_papers}.",
                "Limit the search to {categories} and look for work on {topic}, newest first.",
                "Within {categories}, find {topic} papers posted {since}.",
                "Pull {n_papers} about {topic}, but only from {categories}.",
            ),
        },
    },
    "read_section": {
        "normal": {
            "zh": (
                "找一篇最相关的 {topic} 论文,读一下它的{section}。",
                "帮我看看 {id} 的{section},讲了什么?这篇是{why}的。",
                "{id} 这篇的{section}里写了什么?帮我取出来概括一下,{tone}。",
                "检索 {topic_paren},挑最相关的那篇,把{section}的要点整理出来。",
                "我想知道 {id} 的{section}具体怎么写的,{day}{occasion}前帮我读一下。",
                "{topic}方向排第一的论文,{section}讲了什么?",
                "打开 {id},看它的{section},告诉我关键结论。",
            ),
            "en": (
                "Find the most relevant paper on {topic} and read {section}.",
                "What does {section} of {id} say? {why}",
                "Open {id} and give me the gist of {section} {tone}.",
                "Search for {topic}, take the top hit and walk me through {section}.",
                "I need the details from {section} of {id} before {occasion} {day}. Can you read it?",
                "For the best-matching {topic} paper, what's in {section}?",
            ),
        },
        "parallel": {
            "zh": (
                "同时读 {ids} 的{section},比较一下。",
                "把 {ids} 各自的{section}取出来,{day}要用。",
                "{ids} 的{section}分别写了什么?一起看。",
                "并行打开 {ids},看它们的{section},{tone}。",
                "帮我一次性读完 {ids} 的{section},列出要点。",
                "对比一下 {ids} 在{section}里的说法,{occasion}上要讲。",
            ),
            "en": (
                "Read {section} of {ids} at the same time and compare.",
                "Pull {section} from each of {ids}; I need them {day}.",
                "What does {section} say in each of {ids}? Check them together.",
                "Open {ids} in parallel and go through {section} {tone}.",
                "Read {section} of {ids} in one pass and list the key points.",
                "Contrast what {section} of each of {ids} says, for {occasion}.",
            ),
        },
    },
    "summarize_paper": {
        "normal": {
            "zh": (
                "帮我总结一下 {id},{tone}。",
                "{id} 这篇讲了什么?{tone}。",
                "我没时间读 {id},你帮我概括下它的{focus}。",
                "给 {id} 写个简短摘要,{tone},{day}要看。",
                "{id} 的{focus}是什么?这篇是{why}的。",
                "帮我把 {id} 读一遍,告诉我{focus},{tone}。",
            ),
            "en": (
                "Summarise {id} for me {tone}.",
                "What is {id} about? Answer {tone}.",
                "I don't have time to read {id}. What's its {focus}?",
                "Can you give me a short summary of {id} {tone}? I'll read it {day}.",
                "Tell me the {focus} of {id}. {why}",
                "Quick one: summarise paper {id} {tone}.",
            ),
        },
        "clarify": {
            "zh": (
                "我昨天跟你提过的那篇论文,帮我总结一下,{tone}。",
                "导师早上转我的那篇讲了什么?{tone}。",
                "组里推荐的那篇,{focus}是什么?{day}{occasion}要讲。",
                "刚才群里发的那篇帮我写个摘要,{tone}。",
                "你还记得我上周读的那篇吗?说说它的{focus},{tone}。",
                "参考文献里有一篇我没读过,就是我说的那篇,要点讲一下,{tone}。",
            ),
            "en": (
                "Summarise the paper I told you about yesterday, {tone}.",
                "My advisor forwarded me a paper this morning. What's it about? Answer {tone}.",
                "What's the {focus} of the paper our lab recommended? I present it at {occasion} {day}.",
                "Write me a summary of that paper from the group chat, {tone}.",
                "Remember the one I read last week? Tell me its {focus}, {tone}.",
                "There's a paper in my reference list I mentioned but never read; give me the gist {tone}.",
            ),
        },
    },
    "compare_papers": {
        "normal": {
            "zh": (
                "比较一下 {id_pair} 在{aspect}上的区别,{tone}。",
                "{id_pair} 这两篇,哪篇更适合当我的{use}?",
                "帮我对比 {id_pair} 的实验设置,看看结论是否一致,{tone}。",
                "{id_pair} 都在做类似的事吗?说说它们在{aspect}上的异同。",
                "取一下 {id_pair} 的摘要,{day}{occasion}前告诉我它们的{aspect}有什么不同。",
                "我在 {id_pair} 之间犹豫,从{aspect}上比较一下。",
            ),
            "en": (
                "Compare {id_pair}: how do their {aspect} differ? Answer {tone}.",
                "Which of {id_pair} would make the better {use}?",
                "Contrast the experimental setups of {id_pair}. Do their findings agree? Answer {tone}.",
                "Are {id_pair} solving the same problem? Lay out how their {aspect} compare.",
                "Fetch the abstracts of {id_pair}; before {occasion} {day}, say how their {aspect} differ.",
                "I'm torn between {id_pair}. Compare them on {aspect}.",
            ),
        },
        "clarify": {
            "zh": (
                "我前面提到的那两篇,在{aspect}上有什么区别?",
                "组里推荐的两篇论文,当{use}的话选哪篇好?",
                "导师发来的两篇论文,帮我对照一下{aspect},{tone}。",
                "上周读的两篇我拿不定主意,从{aspect}上帮我比一比。",
                "我说的那两篇结论一致吗?{tone}。",
                "把昨天那两篇的异同列一下,{day}{occasion}要用。",
            ),
            "en": (
                "How do the two papers I mentioned earlier differ in their {aspect}?",
                "Of the two papers the lab recommended, which should I pick as a {use}?",
                "Contrast the {aspect} of the two papers my advisor sent, {tone}.",
                "I can't decide between the two papers from last week. Help me weigh their {aspect}.",
                "Do the two papers I brought up reach the same conclusion? Answer {tone}.",
                "List how yesterday's two papers differ; I need it for {occasion} {day}.",
            ),
        },
        "parallel": {
            "zh": (
                "同时拿一下 {ids} 的摘要,告诉我哪篇和{topic}最贴近。",
                "一起取 {ids} 的元数据,按发表日期排个序,{day}要用。",
                "把 {ids} 的摘要都调出来,比较它们的{aspect}。",
                "并行查一下 {ids},看看它们的{aspect}有什么不同,{tone}。",
                "{ids} 这几篇一起看,{day}{occasion}前告诉我谁的方法最简单。",
                "批量取 {ids} 的信息,告诉我哪篇最适合当{use}。",
            ),
            "en": (
                "Grab {ids} in one go and say which of them fits {topic} best.",
                "Fetch the metadata of {ids} at once and sort them by publication date; I need it {day}.",
                "Get the abstracts for {ids} and compare their {aspect}.",
                "Look up {ids} in parallel and tell me how their {aspect} differ, {tone}.",
                "Read {ids} side by side: whose method is the simplest? Tell me before {occasion} {day}.",
                "Fetch {ids} together and tell me which would be the best {use}.",
            ),
        },
    },
    "write_note": {
        "normal": {
            "zh": (
                "给 {id} 写一份笔记,用{mode},{day}要看。",
                "把 {id} 记到笔记里,{mode}就行,这篇是{why}的。",
                "{id} 读完了,{day}前帮我生成{mode}的阅读笔记。",
                "帮我存一条 {id} 的笔记,{mode},{occasion}上要用。",
                "{id} 这篇是{why}的,值得记下来,按{mode}写。",
                "请为 {id} 写阅读笔记,{mode},{day}{occasion}要用。",
            ),
            "en": (
                "Write a note for {id} in {mode}; I'll read it {day}.",
                "Save a reading note on {id} in {mode}. {why}",
                "I've finished {id}; write it up as a note in {mode} before {occasion}.",
                "Create a note for paper {id} using {mode}. {why}",
                "Log {id} in my notes using {mode}, so I have it {day}.",
                "For {occasion} {day} I need a note on {id}; use {mode}.",
            ),
        },
        "clarify": {
            "zh": (
                "给我昨天读完的那篇写份笔记,用{mode},{day}要用。",
                "导师推荐的那篇存一条阅读笔记,{mode}就行。",
                "按{mode}把上周组里讨论的那篇记下来,{occasion}上还要讲。",
                "刚读完的那篇建个笔记吧,{mode}。",
                "我说的那篇论文,笔记帮我写一下,{mode},{day}{occasion}要用。",
                "{why}的那篇,用{mode}记一篇笔记。",
            ),
            "en": (
                "Write up a note in {mode} for the paper I finished yesterday; I need it {day}.",
                "Save a reading note in {mode} for the one my advisor recommended.",
                "Log the paper we discussed at {occasion} in my notes using {mode}.",
                "That paper I just closed: create a note in {mode} before {occasion} {day}.",
                "Can you make a note in {mode} for the paper I mentioned? I want it {day}.",
                "Please write a note in {mode} for the paper from the group chat.",
            ),
        },
    },
    "search_notes": {
        "normal": {
            "zh": (
                "我以前读过{topic}相关的东西吗?查一下我的笔记。",
                "在我的笔记里搜 {topic},给我 {n_notes}。",
                "翻一下我的笔记,有没有打了 {tag} 标签、讲 {topic} 的?",
                "我记得写过一篇关于 {topic_paren} 的笔记,帮我找出来。",
                "查我的读书笔记:{topic},只看标签 {tag}。",
                "笔记库里关于 {topic} 的记录有哪些?{n_notes}就够。",
            ),
            "en": (
                "Have I read anything on {topic} before? Check my notes.",
                "Search my notes for {topic} and show me {n_notes}.",
                "Look through my notes tagged {tag} for anything on {topic}.",
                "I remember writing a note about {topic}. Can you find it?",
                "What do my reading notes say about {topic}? Show {n_notes}.",
                "Find my notes on {topic} with the tag {tag}.",
            ),
        },
    },
    "list_notes": {
        "normal": {
            "zh": (
                "列出我{notes_since}写的笔记,{day}要复盘。",
                "我最近写的 {n_notes}是什么?帮我调出来。",
                "看看我标了 {tag} 的笔记,{notes_since}的。",
                "把 {tag} 标签下最新的 {n_notes}列给我。",
                "我{notes_since}都记了哪些论文?列一下。",
                "给我看最近的笔记,只要 {tag} 的,{n_notes}就行。",
            ),
            "en": (
                "List the notes I've written {notes_since}.",
                "Pull up the {n_notes} I wrote most recently.",
                "Show my notes tagged {tag} {notes_since}.",
                "Pick {n_notes} from my notes tagged {tag}; I want the newest.",
                "Which papers have I taken notes on {notes_since}? I'm reviewing them {day}.",
                "Bring up my recent notes, just the {tag} ones; {n_notes} will do.",
            ),
        },
    },
    "lookup_paper": {
        "normal": {
            "zh": (
                "{id} 是哪篇论文?想知道{fields}。",
                "这篇是{why}的,帮我查一下 {id} 的{fields}。",
                "帮我看看 {id} 属于哪个分类、什么时候提交的,{day}要用。",
                "{id} 的作者都有谁?这篇是{why}的。",
                "我手上只有编号 {id},帮我查下这篇的{fields}。",
                "给我 {id} 的元数据,重点看{fields}。",
            ),
            "en": (
                "What paper is {id}? {why}",
                "Look up the {fields} of {id}.",
                "Which category is {id} in, and when was it submitted? {why}",
                "Who are the authors of {id}? I need them for {occasion} {day}.",
                "All I have is the id {id}. Can you tell me its {fields}?",
                "{why} Please fetch the {fields} for {id}.",
            ),
        },
        "parallel": {
            "zh": (
                "帮我同时查 {ids} 的{fields}。",
                "{ids} 分别是哪几篇?一起查一下,{day}要用。",
                "一次性给我 {ids} 的{fields},{occasion}上要引用。",
                "这几个编号 {ids} 对应什么论文?它们是{why}的。",
                "批量查一下 {ids} 这几篇的{fields}。",
                "把 {ids} 的基本信息都列出来,{fields}都要。",
            ),
            "en": (
                "Do a parallel lookup of {ids} and report each one's {fields}.",
                "Which papers are {ids}? Check them all together; I need them {day}.",
                "Give me the {fields} of {ids} in one go, for {occasion}.",
                "What are the papers behind the ids {ids}? {why}",
                "Batch-fetch the metadata for {ids}, mainly the {fields}.",
                "List the basics for {ids}: I want the {fields}.",
            ),
        },
    },
    "whats_new": {
        "normal": {
            "zh": (
                "我上次看{topic}还是 {on_date},之后出了什么新东西?",
                "{since}{topic}有新进展吗?{tone}。",
                "从 {on_date} 到现在,{topic}方向有哪些新论文?",
                "跟进一下 {topic_paren}:{since}有没有值得注意的新工作?",
                "{on_date} 之后{topic}又冒出了哪些论文?帮我盘一下。",
                "帮我补一下{topic}的进度,我只读到 {on_date}。",
            ),
            "en": (
                "I last checked {topic} around {on_date}. What's new since then?",
                "Anything new on {topic} {since}? Answer {tone}.",
                "What {topic} papers have appeared between {on_date} and now?",
                "Catch me up on {topic}: what's been posted {since}?",
                "Since {on_date}, which new papers on {topic} are worth a look?",
                "My {topic} reading stops at {on_date}. Fill me in on what came after.",
            ),
        },
        "clarify": {
            "zh": (
                "我那个方向上次是 {on_date} 看的,之后有什么新论文?",
                "导师让我跟的那个题目,{since}有什么新进展?{tone}。",
                "我们组关注的那块,从 {on_date} 到现在出了哪些新工作?{n_papers}就行。",
                "之前跟你说的那个主题,帮我补一下进度,我只读到 {on_date}。",
                "老板关心的那个领域,{on_date} 之后又冒出了哪些论文?挑 {n_papers}。",
                "上次聊过的那个问题,{since}有没有值得注意的新论文?",
            ),
            "en": (
                "I last checked my area around {on_date}. What new papers are there since?",
                "Any progress {since} on the problem my advisor assigned me? Answer {tone}.",
                "What has our group's topic produced between {on_date} and now? I need {n_papers}.",
                "Fill me in on the subject I told you about; my reading stops at {on_date}.",
                "Since {on_date}, what has appeared in the area my boss cares about? Pick {n_papers}.",
                "Follow-up on the field we discussed: anything notable posted {since}?",
            ),
        },
    },
    "explain_tools": {
        "no_call": {
            "zh": (
                "先别搜,{tool_param}是做什么用的?",
                "{tool_param}有默认值吗?直接告诉我就行。",
                "解释一下{tool_param},不用调用工具。",
                "{tool_pair}有什么区别?不用实际调用,讲讲就好。",
                "如果{tool_param}不填,会按什么处理?",
                "笔记用{mode}的话要写哪几部分?另外{tool_param}是什么意思?",
                "{tool_pair}分别适合什么场景?简单说说。",
            ),
            "en": (
                "No need to search: what does {tool_param} do?",
                "Does {tool_param} have a default? Just tell me.",
                "Explain {tool_param} without calling any tool.",
                "What's the difference between {tool_pair}? Just explain, don't run anything.",
                "If I leave out {tool_param}, what happens?",
                "In {mode}, which parts does a note need? And what is {tool_param} for?",
                "When would you use {tool_pair}? A short answer is fine.",
            ),
        },
    },
    "chit_chat": {
        "no_call": {
            "zh": (
                "谢谢,{topic}那几篇正好用在{occasion}上。",
                "今天先这样,{day}{occasion}前我再来找你。",
                "好的,辛苦了!{topic}的事{day}再聊。",
                "你刚才的解释很清楚,我{day}{occasion}就照这个讲。",
                "哈哈,今天读{topic}读得头都大了,先休息一下,{day}继续。",
                "感谢帮忙,{occasion}的材料差不多齐了,{day}见。",
            ),
            "en": (
                "Thanks, those {topic} papers are exactly what I need for {occasion}.",
                "That's it for today. I'll come back before {occasion} {day}.",
                "Great, thanks a lot! Let's pick up {topic} again {day}.",
                "Your explanation was really clear; I'll present it that way at {occasion} {day}.",
                "Wow, my head is spinning after all this {topic} reading. Taking a break until {day}.",
                "Appreciate the help, my slides for {occasion} are almost done. See you {day}.",
            ),
        },
    },
}

# surface pools: wording only, never recorded in slots. Chinese values are clauses that templates end with
# 。 or join with 的 ({why}: 这篇是{why}的); English {why} values are whole sentences.
_SURFACE_POOLS = {
    "tone": {"zh": ("三句话以内", "一段话就行", "讲清楚问题和方法就好", "重点说贡献", "给个 TL;DR",
                    "用大白话讲", "列几个要点"),
             "en": ("in three sentences", "in one short paragraph", "in two or three lines", "in plain words",
                    "in a few bullet points", "in under a hundred words")},
    "focus": {"zh": ("核心思路", "主要贡献", "实验结论", "局限", "要解决的问题", "关键假设"),
              "en": ("main contribution", "core idea", "key result", "biggest limitation", "central claim",
                     "key assumption")},
    "use": {"zh": ("基线", "参考实现", "综述的切入点", "组会讲解的论文", "复现对象"),
            "en": ("baseline", "starting point for a survey", "reference implementation",
                   "reading-group pick", "reproduction target")},
    "aspect": {"zh": ("方法", "数据", "评测", "假设", "计算开销", "实验规模"),
               "en": ("methods", "datasets", "evaluations", "assumptions", "compute costs",
                      "experiment scales")},
    "occasion": {"zh": ("组会", "读书会", "开题答辩", "组内分享", "导师面谈", "论文研讨会"),
                 "en": ("the group meeting", "the reading group", "my proposal defence", "the lab seminar",
                        "my advisor meeting", "journal club")},
    "day": {"zh": ("周一", "周二", "周三", "周四", "周五", "周末", "明天", "后天", "下周"),
            "en": ("on Monday", "on Tuesday", "on Wednesday", "on Thursday", "on Friday", "this weekend",
                   "tomorrow", "the day after tomorrow", "next week")},
    "why": {"zh": ("我打算引用", "审稿人提到过", "导师转给我", "参考文献里看到", "组里有人推荐",
                   "在网上看到有人讨论"),
            "en": ("I want to cite it.", "A reviewer brought it up.", "My advisor forwarded it to me.",
                   "It came up in a reference list.", "Someone in the lab recommended it.",
                   "I saw it discussed online.")},
    "fields": {"zh": ("标题和作者", "作者和发表日期", "所属分类和提交日期", "标题、作者和摘要", "提交日期",
                      "完整的元数据"),
               "en": ("title and authors", "authors and submission date", "categories and submission date",
                      "title, authors and abstract", "submission date", "full metadata")},
}
_SECTION_TEXT = {
    "zh": {"introduction": ("引言部分", "引言"), "method": ("方法部分", "方法章节"),
           "experiments": ("实验部分", "实验章节"), "conclusion": ("结论部分", "结论"),
           "all": ("全文", "完整正文")},
    "en": {"introduction": ("the introduction", "the intro section"),
           "method": ("the method section", "the methods part"),
           "experiments": ("the experiments section", "the experimental results"),
           "conclusion": ("the conclusion", "the concluding section"),
           "all": ("the full text", "the whole paper")},
}
_MODE_TEXT = {"zh": {"scaffold": ("scaffold 模式", "scaffold(只摘证据句)"),
                     "draft": ("draft 模式", "draft(完整草稿)")},
              "en": {"scaffold": ("scaffold mode", "the scaffold mode that quotes evidence sentences"),
                     "draft": ("draft mode", "the draft mode that writes a full draft")}}
_ZH_NUMBER = {1: "一", 2: "两", 3: "三", 4: "四", 5: "五"}
_EN_NUMBER = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
_EN_MONTH = {7: "July", 8: "August", 9: "September", 10: "October"}
_HAN = "一-鿿"
_CJK_PUNCT = "　-〿＀-￯"
_HAN_GAP = re.compile(f"(?<=[{_HAN}]) +(?=[{_HAN}])")
# Chinese text punctuation: half-width , ? ! : ; ( ) as the repository docs write it, and the CJK blocks
_ZH_PUNCT = ",?!:;()" + _CJK_PUNCT
_PUNCT_GAP = re.compile(f" *([{_ZH_PUNCT}]) *")
_HAN_LATIN = re.compile(f"(?<=[{_HAN}])(?=[A-Za-z0-9])|(?<=[A-Za-z0-9])(?=[{_HAN}])")
_SPACES = re.compile(r"\s+")
_PLACEHOLDER = re.compile(r"\{[a-z_]+\}")
_MASKS = ((re.compile(r"(?i)(?:arxiv:\s*)?\d{4}\.\d{4,5}(?:v\d+)?"), "<id>"),
          (re.compile(r"\d{4}-\d{2}-\d{2}"), "<date>"),
          (re.compile(r"\d+"), "<n>"))
_REQUEST_ID = re.compile(r"req-([a-z0-9]+)-(\d+)")  # req-<split>-<number>; req-m6-<number> for M6's own


# ---- corpus pools and the split -------------------------------------------------------------------------


@dataclass(frozen=True)
class Topic:
    id: str
    name: str  # Chinese name
    phrases: tuple[str, ...]  # English phrases; the one a request names is its topic slot
    papers: tuple[str, ...]  # canonical arXiv ids


@dataclass(frozen=True)
class Corpus:
    topics: tuple[Topic, ...]
    window: tuple[str, str]  # ISO dates, the recording window
    sha256: str | None = None  # of the manifest file's bytes; None for a corpus built in memory


def load_corpus(path: str | Path = CORPUS_MANIFEST) -> Corpus:
    """The topics, the recording window and the SHA-256 of the corpus manifest; a ValueError when an id is not
    canonical or a paper is listed under two topics (the split would then not be disjoint)."""
    data = Path(path).read_bytes()
    manifest = json.loads(data.decode("utf-8"))
    topics, owner = [], {}
    for entry in manifest["topics"]:
        papers = []
        for raw in entry["papers"]:
            aid = normalize_arxiv_id(raw)
            if aid is None:
                raise ValueError(f"topic {entry['id']}: {raw!r} is not an arXiv id")
            if aid in owner:
                raise ValueError(f"paper {aid} is listed under both {owner[aid]} and {entry['id']}")
            owner[aid] = entry["id"]
            papers.append(aid)
        topics.append(Topic(entry["id"], entry["name"], tuple(entry["phrases"]), tuple(papers)))
    start, end = manifest["window"]
    return Corpus(tuple(topics), (start, end), hashlib.sha256(data).hexdigest())


def _rng(seed: int, *purpose: object) -> random.Random:
    """A generator per purpose, so adding a draw in one place never shifts another (str seeds are hashed
    with SHA-512 by random, independent of PYTHONHASHSEED)."""
    return random.Random(":".join([str(seed), *map(str, purpose)]))


def template_families() -> list[tuple[str, str, str, int, str]]:
    """(intent, kind, lang, index, template) for every template; index counts per intent and language over
    the kinds in KINDS order, so f"{intent}/{lang}/{index}" names one template."""
    rows = []
    for intent, kinds in TEMPLATES.items():
        for lang in LANGS:
            index = 0
            for kind in KINDS:
                for template in kinds.get(kind, {}).get(lang, ()):
                    rows.append((intent, kind, lang, index, template))
                    index += 1
    return rows


def family_name(intent: str, lang: str, index: int) -> str:
    return f"{intent}/{lang}/{index}"


def split_templates(seed: int) -> dict[str, str]:
    """family -> split: each (intent, kind, lang) pool, shuffled with the seed, gives its first template to
    test, the next to dev and the rest (four, or more when the pool has more than six) to train."""
    pools: dict[tuple[str, str, str], list[str]] = {}
    for intent, kind, lang, index, _ in template_families():
        pools.setdefault((intent, kind, lang), []).append(family_name(intent, lang, index))
    assignment = {}
    for key, families in pools.items():
        if len(families) < 6:
            raise ValueError(f"template pool {'/'.join(key)} has {len(families)} templates, fewer than six")
        order = sorted(families)
        _rng(seed, "templates", *key).shuffle(order)
        cut = 0
        for split in ("test", "dev"):
            for family in order[cut:cut + TEMPLATE_SPLIT[split]]:
                assignment[family] = split
            cut += TEMPLATE_SPLIT[split]
        for family in order[cut:]:
            assignment[family] = "train"
    return assignment


def split_topics(corpus: Corpus, seed: int) -> dict[str, tuple[Topic, ...]]:
    """split -> topics: round(0.15 n) topics each to test and dev, drawn with the seed, the rest to train.
    Paper ids follow their topic (Corpus guarantees each id belongs to one topic)."""
    order = sorted(corpus.topics, key=lambda t: t.id)
    _rng(seed, "topics").shuffle(order)
    out, cut = {}, 0
    for split in ("test", "dev"):
        n = round(SPLIT_SHARE[split] * len(order))
        out[split] = tuple(sorted(order[cut:cut + n], key=lambda t: t.id))
        cut += n
    out["train"] = tuple(sorted(order[cut:], key=lambda t: t.id))
    return {split: out[split] for split in SPLITS}


# ---- filling a template ---------------------------------------------------------------------------------


def placeholders(template: str) -> list[str]:
    """The field names of a template, in order of appearance."""
    return [name for _, name, _, _ in string.Formatter().parse(template) if name]


@dataclass
class _Context:
    lang: str
    topics: tuple[Topic, ...]
    window: tuple[str, str]
    rng: random.Random
    topic: Topic | None = None


def _pick(ctx: _Context, values: Iterable):
    values = list(values)
    return values[ctx.rng.randrange(len(values))]


def _request_topic(ctx: _Context) -> Topic:
    if ctx.topic is None:
        ctx.topic = _pick(ctx, ctx.topics)
    return ctx.topic


def _topic_slots(ctx: _Context) -> tuple[str, dict]:
    """English: one of the topic's phrases, which the slot records. Chinese: the Chinese name only, and the
    slot records the topic's first phrase (the query's required words stay English)."""
    topic = _request_topic(ctx)
    if ctx.lang == "zh":
        return topic.name, {"topic": topic.phrases[0], "topic_id": topic.id}
    phrase = _pick(ctx, topic.phrases)
    return phrase, {"topic": phrase, "topic_id": topic.id}


def _topic_paren_slots(ctx: _Context) -> tuple[str, dict]:
    """Chinese only: the Chinese name followed by one English phrase in parentheses (half-width, as the docs),
    混合专家(mixture of experts), or the name alone when it already is the phrase (KV cache)."""
    topic = _request_topic(ctx)
    phrase = _pick(ctx, topic.phrases)
    if normalize_text(topic.name).casefold() == phrase.casefold():
        return topic.name, {"topic": phrase, "topic_id": topic.id}
    return f"{topic.name}({phrase})", {"topic": phrase, "topic_id": topic.id}


def _papers(ctx: _Context, k: int) -> list[str]:
    """k distinct ids: from the request's topic when it names one, else from any topic of the split."""
    pool = ctx.topic.papers if ctx.topic is not None else [p for t in ctx.topics for p in t.papers]
    return ctx.rng.sample(sorted(pool), k)


def _id_text(ctx: _Context, aid: str) -> str:
    return aid if ctx.rng.random() < 0.7 else f"arXiv:{aid}"


def _join(ctx: _Context, items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    if ctx.lang == "zh":
        head = "、".join(items[:-1])
        return f"{head} {_pick(ctx, ('和', '与'))} {items[-1]}" if len(items) > 2 or ctx.rng.random() < 0.7 \
            else "、".join(items)
    return f"{', '.join(items[:-1])} and {items[-1]}"


def _fill_id(ctx: _Context) -> tuple[str, dict]:
    aid = _papers(ctx, 1)[0]
    return _id_text(ctx, aid), {"arxiv_id": aid}


def _fill_id_pair(ctx: _Context) -> tuple[str, dict]:
    ids = _papers(ctx, 2)
    return _join(ctx, [_id_text(ctx, a) for a in ids]), {"arxiv_ids": ids}


def _fill_ids(ctx: _Context) -> tuple[str, dict]:
    ids = _papers(ctx, _pick(ctx, (2, 3)))
    return _join(ctx, [_id_text(ctx, a) for a in ids]), {"arxiv_ids": ids}


def _iso(month: int, day: int) -> str:
    return f"{YEAR:04d}-{month:02d}-{day:02d}"


def _days(ctx: _Context, *, whole_window: bool = False) -> list[tuple[int, int]]:
    """(month, day) inside the window; without whole_window the window's last week is left out, so papers
    can follow a 'since' date."""
    start, end = ctx.window
    days = [(month, day) for month in _months(ctx) for day in range(1, _month_len(month) + 1)
            if start <= _iso(month, day) <= end]
    return days if whole_window else days[:-7]


def _month_len(month: int) -> int:
    return calendar.monthrange(YEAR, month)[1]


def _months(ctx: _Context) -> list[int]:
    start, end = ctx.window
    return list(range(int(start[5:7]), int(end[5:7]) + 1))


def _since_phrase(ctx: _Context) -> tuple[str, str]:
    """A 'since <date>' phrase and the ISO date it records."""
    if ctx.rng.random() < 0.4:  # a whole month
        month = _pick(ctx, _months(ctx))
        iso = _iso(month, 1)
        forms = {"zh": (f"{month} 月以来", f"{YEAR} 年 {month} 月以来", f"从 {month} 月起"),
                 "en": (f"since {_EN_MONTH[month]}", f"since {_EN_MONTH[month]} {YEAR}",
                        f"from {_EN_MONTH[month]} {YEAR} on")}
        return _pick(ctx, forms[ctx.lang]), iso
    month, day = _pick(ctx, _days(ctx))
    iso = _iso(month, day)
    forms = {"zh": (f"{iso} 以来", f"从 {iso} 起", f"{month} 月 {day} 日以来",
                    f"自 {YEAR} 年 {month} 月 {day} 日起"),
             "en": (f"since {iso}", f"on or after {iso}", f"since {_EN_MONTH[month]} {day}, {YEAR}",
                    f"since {day} {_EN_MONTH[month]} {YEAR}")}
    return _pick(ctx, forms[ctx.lang]), iso


def _fill_since(ctx: _Context) -> tuple[str, dict]:
    text, iso = _since_phrase(ctx)
    return text, {"date_from": iso}


def _fill_notes_since(ctx: _Context) -> tuple[str, dict]:
    text, iso = _since_phrase(ctx)
    return text, {"since": iso}


def _fill_on_date(ctx: _Context) -> tuple[str, dict]:
    if ctx.rng.random() < 0.3:
        month = _pick(ctx, _months(ctx)[:-1])
        text = f"{month} 月初" if ctx.lang == "zh" else f"the start of {_EN_MONTH[month]}"
        return text, {"date_from": _iso(month, 1)}
    month, day = _pick(ctx, _days(ctx))
    iso = _iso(month, day)
    forms = {"zh": (iso, f"{month} 月 {day} 日"), "en": (iso, f"{_EN_MONTH[month]} {day}, {YEAR}")}
    return _pick(ctx, forms[ctx.lang]), {"date_from": iso}


def _fill_between(ctx: _Context) -> tuple[str, dict]:
    months = _months(ctx)[:-1]  # whole months must end inside the window
    if ctx.rng.random() < 0.4 and len(months) >= 2:
        first, last = sorted(ctx.rng.sample(months, 2))
        start, end = _iso(first, 1), _iso(last, _month_len(last))
        forms = {"zh": (f"{first} 月到 {last} 月之间",),
                 "en": (f"between {_EN_MONTH[first]} and {_EN_MONTH[last]} {YEAR}",)}
        return _pick(ctx, forms[ctx.lang]), {"date_from": start, "date_to": end}
    days = _days(ctx, whole_window=True)
    i, j = sorted(ctx.rng.sample(range(len(days)), 2))
    if j - i < 7:
        j = min(len(days) - 1, i + 7 + ctx.rng.randrange(40))
    (m1, d1), (m2, d2) = days[i], days[j]
    start, end = _iso(m1, d1), _iso(m2, d2)
    forms = {"zh": (f"{start} 到 {end} 之间", f"{m1} 月 {d1} 日至 {m2} 月 {d2} 日"),
             "en": (f"between {start} and {end}",
                    f"from {_EN_MONTH[m1]} {d1} to {_EN_MONTH[m2]} {d2}, {YEAR}")}
    return _pick(ctx, forms[ctx.lang]), {"date_from": start, "date_to": end}


def _count_text(ctx: _Context, n: int, zh_unit: str, en_noun: str) -> str:
    """A count in words or digits; a count of one is always spelled out (一篇, one paper)."""
    if ctx.lang == "zh":
        return f"一{zh_unit}" if n == 1 else _pick(ctx, (f"{n} {zh_unit}", f"{_ZH_NUMBER[n]}{zh_unit}"))
    if n == 1:
        return f"one {en_noun}"
    return _pick(ctx, (f"{n} {en_noun}s", f"{_EN_NUMBER[n]} {en_noun}s"))


def _fill_n_papers(ctx: _Context) -> tuple[str, dict]:
    n = _pick(ctx, COUNTS)
    return _count_text(ctx, n, "篇", "paper"), {"count": n}


def _fill_n_notes(ctx: _Context) -> tuple[str, dict]:
    n = _pick(ctx, COUNTS)
    return _count_text(ctx, n, "条笔记", "note"), {"count": n}


def _fill_categories(ctx: _Context) -> tuple[str, dict]:
    chosen = ctx.rng.sample(schemas.CATEGORIES, _pick(ctx, (1, 1, 2)))
    if len(chosen) == 1:
        text = chosen[0]
    elif ctx.lang == "zh":
        text = _pick(ctx, (f"{chosen[0]} 和 {chosen[1]}", f"{chosen[0]}、{chosen[1]}"))
    else:
        text = _pick(ctx, (f"{chosen[0]} and {chosen[1]}", f"{chosen[0]} or {chosen[1]}"))
    return text, {"categories": sorted(chosen)}


def _fill_section(ctx: _Context) -> tuple[str, dict]:
    section = _pick(ctx, schemas.SECTIONS)
    return _pick(ctx, _SECTION_TEXT[ctx.lang][section]), {"section": section}


def _fill_mode(ctx: _Context) -> tuple[str, dict]:
    mode = _pick(ctx, MODES)
    return _pick(ctx, _MODE_TEXT[ctx.lang][mode]), {"mode": mode}


def _fill_tag(ctx: _Context) -> tuple[str, dict]:
    tag = _pick(ctx, TAGS)
    return tag, {"tags": [tag]}


def _tool_params() -> list[tuple[str, str]]:
    return [(t["function"]["name"], p)
            for t in schemas.TOOLS for p in t["function"]["parameters"]["properties"]]


def _fill_tool_param(ctx: _Context) -> tuple[str, dict]:
    tool, param = _pick(ctx, _tool_params())
    text = f"{tool} 的 {param} 参数" if ctx.lang == "zh" else f"the {param} parameter of {tool}"
    return text, {"tool": tool, "param": param}


def _fill_tool_pair(ctx: _Context) -> tuple[str, dict]:
    tools = ctx.rng.sample(schemas.TOOL_NAMES, 2)
    text = f"{tools[0]} 和 {tools[1]}" if ctx.lang == "zh" else f"{tools[0]} and {tools[1]}"
    return text, {"tools": tools}


def _surface(name: str) -> Callable[[_Context], tuple[str, dict]]:
    return lambda ctx: (_pick(ctx, _SURFACE_POOLS[name][ctx.lang]), {})


FILLERS: dict[str, Callable[[_Context], tuple[str, dict]]] = {
    "topic": _topic_slots, "topic_paren": _topic_paren_slots, "id": _fill_id, "id_pair": _fill_id_pair,
    "ids": _fill_ids, "since": _fill_since,
    "notes_since": _fill_notes_since, "on_date": _fill_on_date, "between": _fill_between,
    "n_papers": _fill_n_papers, "n_notes": _fill_n_notes, "categories": _fill_categories,
    "section": _fill_section, "mode": _fill_mode, "tag": _fill_tag, "tool_param": _fill_tool_param,
    "tool_pair": _fill_tool_pair, **{name: _surface(name) for name in SURFACE},
}
# what a clarify template leaves out on purpose, in the schemas.MISSING vocabulary
CLARIFY_MISSING = {"find_recent_papers": ["topic"], "whats_new": ["topic"], "summarize_paper": ["paper_id"],
                   "write_note": ["paper_id"], "compare_papers": ["paper_id"]}


def tidy(text: str, lang: str) -> str:
    """Whitespace collapsed; Chinese text also gets the repository's spacing, whatever spaces the
    placeholders left: one space between a Chinese character and a Latin letter or digit (9 月以来,
    scaffold 模式), no space between two Chinese characters and none on either side of a punctuation mark
    (half-width , ? ! : ; ( ) as the docs write them, and 。 、), so tool use(工具调用)的 and
    以来,write_note. English text keeps its spaces."""
    text = _SPACES.sub(" ", text).strip()
    if lang != "zh":
        return text
    text = _HAN_GAP.sub("", _HAN_LATIN.sub(" ", text))
    return _PUNCT_GAP.sub(r"\1", text).strip()


def fill(template: str, lang: str, topics: tuple[Topic, ...], window: tuple[str, str],
         rng: random.Random, *, intent: str | None = None, kind: str = "normal") -> tuple[str, dict]:
    """The text and slots of one request from a template, its language and the split's topics. The topic is
    filled first, so ids drawn later in the same template come from that topic's papers."""
    ctx = _Context(lang=lang, topics=topics, window=window, rng=rng)
    names = placeholders(template)
    order = sorted(dict.fromkeys(names), key=lambda n: n != "topic")
    values, slots = {}, {}
    for name in order:
        if name not in FILLERS:
            raise KeyError(f"template {template!r} has a placeholder {{{name}}} without a filler")
        values[name], extra = FILLERS[name](ctx)
        slots.update(extra)
    if kind == "clarify" and intent is not None:
        slots["missing"] = list(CLARIFY_MISSING[intent])
    return tidy(template.format(**values), lang), slots


def mask_text(text: str) -> str:
    """normalize_text(text) with arXiv ids (an "arXiv:" prefix and a version included) replaced by <id>,
    then ISO dates by <date>, then every remaining run of digits by <n>, so two requests that differ only in
    their ids, dates or numbers mask to the same text."""
    text = normalize_text(text)
    for pattern, token in _MASKS:
        text = pattern.sub(token, text)
    return text


# ---- dedup ----------------------------------------------------------------------------------------------


def shingles(text: str, k: int = SHINGLE) -> frozenset[str]:
    """The character k-grams of normalize_text(text); a text shorter than k is its own single shingle."""
    t = normalize_text(text)
    if len(t) < k:
        return frozenset([t])
    return frozenset(t[i:i + k] for i in range(len(t) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class _ShingleIndex:
    """Kept shingle sets with an inverted index: max_jaccard counts shared shingles per kept set through the
    postings the candidate touches, then |A & B| / (|A| + |B| - |A & B|) is exact."""

    def __init__(self) -> None:
        self.sizes: list[int] = []
        self.postings: dict[str, list[int]] = {}

    def add(self, shingle_set: frozenset[str]) -> None:
        n = len(self.sizes)
        self.sizes.append(len(shingle_set))
        for s in shingle_set:
            self.postings.setdefault(s, []).append(n)

    def max_jaccard(self, shingle_set: frozenset[str]) -> float:
        shared: Counter[int] = Counter()
        for s in shingle_set:
            shared.update(self.postings.get(s, ()))
        best = 0.0
        for i, inter in shared.items():
            best = max(best, inter / (len(shingle_set) + self.sizes[i] - inter))
        return best


DROP_REASONS = ("exact", "near", "train_vs_eval")


class Deduper:
    """Accepts requests one at a time (offer); counts what it drops by reason. A train request offered before
    the dev and test requests it is too close to is not caught, so offer the evaluation splits first.
    Requests that already exist (a preloaded file) are registered unchecked (register); every rule then runs
    against them as against kept requests, and dropped_preloaded counts the drops that a preloaded request
    alone would have caused (such a drop may also have a new request at its threshold)."""

    def __init__(self, *, near: float = NEAR_DUP, train_vs_eval: float = TRAIN_VS_EVAL) -> None:
        self.near, self.train_vs_eval = near, train_vs_eval
        self.texts: set[str] = set()
        self.all = _ShingleIndex()
        self.eval = _ShingleIndex()
        self.dropped = dict.fromkeys(DROP_REASONS, 0)
        self.offered = 0
        self.preloaded = 0
        self.dropped_preloaded = dict.fromkeys(DROP_REASONS, 0)
        self._preloaded_texts: set[str] = set()
        self._preloaded_all = _ShingleIndex()
        self._preloaded_eval = _ShingleIndex()

    def register(self, text: str, split: str) -> None:
        """Keep a request that already exists without checking it: offering it could drop it as a near
        duplicate of another existing request, which is not this build's to decide."""
        key, sh = normalize_text(text), shingles(mask_text(text))
        self.preloaded += 1
        for texts, every, evaluation in ((self.texts, self.all, self.eval),
                                         (self._preloaded_texts, self._preloaded_all, self._preloaded_eval)):
            texts.add(key)
            every.add(sh)
            if split != "train":
                evaluation.add(sh)

    def _against_preloaded(self, text: str, reason: str) -> bool:
        """Whether a preloaded request alone meets the rule the text was dropped by."""
        if reason == "exact":
            return normalize_text(text) in self._preloaded_texts
        sh = shingles(mask_text(text))
        if reason == "near":
            return self._preloaded_all.max_jaccard(sh) >= self.near
        return self._preloaded_eval.max_jaccard(sh) >= self.train_vs_eval

    def check(self, text: str, split: str) -> str | None:
        """The reason the request would be dropped, or None when it would be kept (nothing is recorded)."""
        key = normalize_text(text)
        if key in self.texts:
            return "exact"
        sh = shingles(mask_text(text))
        if self.all.max_jaccard(sh) >= self.near:
            return "near"
        if split == "train" and self.eval.max_jaccard(sh) >= self.train_vs_eval:
            return "train_vs_eval"
        return None

    def offer(self, text: str, split: str) -> str | None:
        """Keep the request (None) or count and return the reason it is dropped."""
        self.offered += 1
        reason = self.check(text, split)
        if reason is not None:
            self.dropped[reason] += 1
            if self.preloaded and self._against_preloaded(text, reason):
                self.dropped_preloaded[reason] += 1
            return reason
        sh = shingles(mask_text(text))
        self.texts.add(normalize_text(text))
        self.all.add(sh)
        if split != "train":
            self.eval.add(sh)
        return None


def dedup(records: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """The records that survive dedup, in their original order, and the drop counts by reason. The dev and
    test records are offered first (in order), then the train records, so the first of two near duplicates
    within the evaluation splits or within train is kept, and a train record never displaces an evaluation
    record."""
    deduper = Deduper()
    kept_ids = set()
    ordered = [i for i, r in enumerate(records) if r["split"] != "train"] + \
              [i for i, r in enumerate(records) if r["split"] == "train"]
    for i in ordered:
        if deduper.offer(records[i]["text"], records[i]["split"]) is None:
            kept_ids.add(i)
    return [r for i, r in enumerate(records) if i in kept_ids], dict(deduper.dropped)


# ---- the build ------------------------------------------------------------------------------------------


def check_mix(mix: Mapping[str, float]) -> dict[str, float]:
    """scenario -> share for every scenario of synth.SCENARIOS (one the mix leaves out gets 0). A ValueError
    for a scenario SCENARIOS does not know, a share that is not a finite number of at least 0, or shares that
    do not sum to 1 (within MIX_TOLERANCE)."""
    unknown = sorted(set(mix) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"unknown scenario {', '.join(map(repr, unknown))} in the mix; the scenarios are "
                         f"{', '.join(SCENARIOS)}")
    shares = {}
    for scenario in SCENARIOS:
        share = mix.get(scenario, 0.0)
        if isinstance(share, bool) or not isinstance(share, int | float) or not math.isfinite(share) \
                or share < 0:
            raise ValueError(f"the share of {scenario} must be a finite number of at least 0, got {share!r}")
        shares[scenario] = float(share)
    total = math.fsum(shares.values())
    if abs(total - 1) > MIX_TOLERANCE:
        raise ValueError(f"the shares of the mix sum to {total:g}, not 1")
    return shares


def check_only(only: Iterable[str] | None) -> tuple[str, ...]:
    """The splits a build makes, in SPLITS order: every split for None. A ValueError for an unknown split, a
    split named twice or no split at all."""
    if only is None:
        return SPLITS
    named = list(only)
    if not named:
        raise ValueError("only names no split")
    unknown = [split for split in named if split not in SPLITS]
    if unknown:
        raise ValueError(f"unknown split {', '.join(map(repr, unknown))}; the splits are {', '.join(SPLITS)}")
    repeated = sorted({split for split in named if named.count(split) > 1})
    if repeated:
        raise ValueError(f"split {', '.join(repeated)} named twice")
    return tuple(split for split in SPLITS if split in named)


def check_id_start(id_start: Mapping[str, int] | None, sizes: Mapping[str, int]) -> dict[str, int]:
    """split -> the first id number for every split the build makes (a size above 0): 1 unless id_start
    names the split. A ValueError for an unknown split, a split the build does not make or a start that is
    not a positive integer."""
    given = dict(id_start or {})
    for split, start in given.items():
        if split not in SPLITS:
            raise ValueError(f"id start for an unknown split {split!r}; the splits are {', '.join(SPLITS)}")
        if sizes.get(split, 0) <= 0:
            raise ValueError(f"id start for {split}, a split this build does not make")
        if isinstance(start, bool) or not isinstance(start, int) or start < 1:
            raise ValueError(f"the id start of {split} must be a positive integer, got {start!r}")
    return {split: given.get(split, 1) for split in SPLITS if sizes.get(split, 0) > 0}


def numbered_ids(split: str, start: int, n: int) -> list[str]:
    """req-<split>-<number> for n numbers from start, padded with zeros to one width: four digits, or the
    digits of the last number when it has more."""
    width = max(4, len(str(start + n - 1)))
    return [f"req-{split}-{i:0{width}d}" for i in range(start, start + n)]


def _id_key(rid: str) -> tuple[str, int] | str:
    """req-<name>-<digits> as (name, number), so req-test-21 and req-test-0021 are one id; any other id as
    it is."""
    match = _REQUEST_ID.fullmatch(rid)
    return (match.group(1), int(match.group(2))) if match else rid


def _listing(values: list[str], limit: int = 10) -> str:
    more = len(values) - limit
    return ", ".join(values[:limit]) + (f" and {more} more" if more > 0 else "")


@dataclass(frozen=True)
class Preloaded:
    """A request file that already exists, read for an incremental build: its requests seed the dedup and
    join both audits, and the build does not write them again."""

    path: str  # as given, with forward slashes
    sha256: str  # of the file's bytes
    records: tuple[dict, ...]

    def summary(self) -> dict:
        """The manifest's entry: the path, the SHA-256, the number of requests and of requests per split."""
        return {"path": self.path, "sha256": self.sha256, "requests": len(self.records),
                "splits": {split: sum(1 for r in self.records if r["split"] == split) for split in SPLITS}}


def _request_problem(record: object) -> str | None:
    """What keeps a parsed line from being a request record, or None."""
    if not isinstance(record, dict):
        return "not a JSON object"
    missing = [key for key in REQUEST_KEYS if key not in record]
    if missing:
        return f"not a request record, missing {', '.join(missing)}"
    if record["split"] not in SPLITS:
        return f"split {record['split']!r} is not one of {', '.join(SPLITS)}"
    for key in ("id", "family", "text"):
        if not isinstance(record[key], str) or not record[key]:
            return f"{key} is not a non-empty string"
    if not isinstance(record["slots"], dict):
        return "slots is not an object"
    return None


def load_preload(path: str | Path) -> Preloaded:
    """A request file: JSON lines, each a request record with every key of synth.REQUEST_KEYS (blank lines
    are skipped). A ValueError names the first line that is not a JSON object, lacks a key, has a split
    outside SPLITS, an id, family or text that is not a non-empty string or slots that are not an object, or
    repeats the id of an earlier line; an OSError when the file cannot be read."""
    data = Path(path).read_bytes()
    name = Path(path).as_posix()
    records, lines = [], {}
    for n, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            record = None
        problem = _request_problem(record)
        if problem is None and record["id"] in lines:
            problem = f"id {record['id']} repeats line {lines[record['id']]}"
        if problem is not None:
            raise ValueError(f"{name}: line {n}: {problem}")
        lines[record["id"]] = n
        records.append(record)
    return Preloaded(name, hashlib.sha256(data).hexdigest(), tuple(records))


def check_ids(sizes: Mapping[str, int], id_start: Mapping[str, int], preloaded: Iterable[Preloaded]) -> None:
    """A ValueError when the preloaded files repeat an id among them (the same file given twice, say), or
    when an id the build would give (numbered_ids from id_start, for the sizes) is already in a preloaded
    file, zero padding aside."""
    held: dict = {}
    repeated = []
    for pre in preloaded:
        for r in pre.records:
            key = _id_key(r["id"])
            if key in held:
                repeated.append(r["id"])
            held.setdefault(key, r["id"])
    if repeated:
        raise ValueError(f"the preloaded files repeat ids: {_listing(repeated)}")
    taken = [rid for split, start in id_start.items()
             for rid in numbered_ids(split, start, sizes.get(split, 0)) if _id_key(rid) in held]
    if taken:
        raise ValueError(f"ids already in a preloaded file: {_listing(taken)}; number the new requests after "
                         "them (id start)")


def plan_counts(size: int, mix: Mapping[str, float] | None = None) -> dict[str, dict[str, int]]:
    """scenario -> lang -> count for a split of the given size: the mix (PLANNED_MIX unless given; check_mix
    validates one) by largest remainder (ties in SCENARIOS order), each scenario's count halved between zh
    and en, the odd one going alternately to zh and en so the languages differ by at most one over the
    split."""
    shares = PLANNED_MIX if mix is None else mix
    raw = {s: shares.get(s, 0.0) * size for s in SCENARIOS}
    counts = {s: int(raw[s]) for s in SCENARIOS}
    rest = size - sum(counts.values())
    for s in sorted(SCENARIOS, key=lambda s: (-(raw[s] - counts[s]), SCENARIOS.index(s)))[:rest]:
        counts[s] += 1
    out, extra_to = {}, 0
    for s in SCENARIOS:
        half, odd = divmod(counts[s], 2)
        out[s] = {"zh": half, "en": half}
        if odd:
            out[s][LANGS[extra_to]] += 1
            extra_to = 1 - extra_to
    return out


@dataclass
class Build:
    records: list[dict]
    report: dict
    preloaded: list[dict] = field(default_factory=list)  # the preloaded files' requests, in file order


def _families_by_cell(assignment: dict[str, str]) -> dict[tuple[str, str, str], list[tuple]]:
    """(split, kind, lang) -> [(intent, family, template)], in family order."""
    cells: dict[tuple[str, str, str], list[tuple]] = {}
    for intent, kind, lang, index, template in template_families():
        family = family_name(intent, lang, index)
        cells.setdefault((assignment[family], kind, lang), []).append((intent, family, template))
    return cells


def _incremental(seed: int, draw: int, built: tuple[str, ...], starts: dict[str, int],
                 preloaded: list[Preloaded], shares: dict[str, float]) -> bool:
    """Whether any of the incremental parameters differs from its default (module docstring)."""
    return (draw != seed or built != SPLITS or any(start != 1 for start in starts.values()) or bool(preloaded)
            or shares != dict(PLANNED_MIX))


def generate(seed: int = 42, sizes: Mapping[str, int] | None = None,
             corpus: Corpus | str | Path = CORPUS_MANIFEST, *, draw_seed: int | None = None,
             only: Iterable[str] | None = None, id_start: Mapping[str, int] | None = None,
             preload: Iterable[Preloaded | str | Path] = (), mix: Mapping[str, float] | None = None) -> Build:
    """The request records for the given split sizes and the report the manifest is made of. The seed fixes
    the split (split_templates, split_topics); the draw seed (the seed unless given) every draw within a
    split, so the same arguments give the same records. Cells are filled test first, then dev, then train;
    each draw takes the next family of the cell in a seeded round robin and fills it with a fresh draw of
    slots, and the Deduper decides, with the preloaded files' requests registered before the first draw.
    A ValueError names every cell that is still short after MAX_ATTEMPTS_PER_REQUEST draws per wanted
    request; before any draw, a ValueError refuses a bad only, id_start or mix (check_only, check_id_start,
    check_mix) and an id a preloaded file holds (check_ids); an OSError or a ValueError comes from a preloaded
    file that cannot be read (load_preload). Within a split the records are shuffled with the draw seed and
    numbered req-<split>-<nnnn> from the split's id start (numbered_ids). The module docstring lists what the
    report of an incremental build adds; Build.preloaded holds the preloaded requests, which write_requests
    audits with the new ones."""
    built = check_only(only)
    given = dict(DEFAULT_SIZES if sizes is None else sizes)
    sizes = {split: given.get(split, 0) if split in built else 0 for split in SPLITS}
    shares = check_mix(PLANNED_MIX if mix is None else mix)
    starts = check_id_start(id_start, sizes)
    preloaded = [p if isinstance(p, Preloaded) else load_preload(p) for p in preload]
    check_ids(sizes, starts, preloaded)
    draw = seed if draw_seed is None else draw_seed
    corpus = corpus if isinstance(corpus, Corpus) else load_corpus(corpus)
    assignment = split_templates(seed)
    topics = split_topics(corpus, seed)
    cells = _families_by_cell(assignment)
    deduper = Deduper()
    preloaded_records = [r for p in preloaded for r in p.records]
    for r in preloaded_records:
        deduper.register(r["text"], r["split"])
    by_split: dict[str, list[dict]] = {split: [] for split in SPLITS}
    short = []
    for split in ("test", "dev", "train"):
        plan = plan_counts(sizes[split], shares)
        for scenario in SCENARIOS:
            kind = SCENARIO_KIND[scenario]
            for lang in LANGS:
                wanted = plan[scenario][lang]
                if wanted == 0:
                    continue
                families = list(cells.get((split, kind, lang), []))
                rng = _rng(draw, "fill", split, scenario, lang)
                rng.shuffle(families)
                got, attempt = 0, 0
                while got < wanted and attempt < MAX_ATTEMPTS_PER_REQUEST * wanted:
                    intent, family, template = families[attempt % len(families)]
                    attempt += 1
                    text, slots = fill(template, lang, topics[split], corpus.window, rng, intent=intent,
                                       kind=kind)
                    if deduper.offer(text, split) is None:
                        by_split[split].append({"intent": intent, "scenario": scenario, "lang": lang,
                                                "split": split, "family": family, "text": text,
                                                "slots": slots})
                        got += 1
                if got < wanted:
                    short.append(f"{split}/{scenario}/{lang}: {got} of {wanted}")
    if short:
        raise ValueError("could not fill every cell after dedup: " + "; ".join(short))
    records = []
    for split in SPLITS:
        rows = by_split[split]
        _rng(draw, "order", split).shuffle(rows)
        ids = numbered_ids(split, starts.get(split, 1), len(rows))
        records += [{"id": rid, **row} for rid, row in zip(ids, rows, strict=True)]
    report = {
        "seed": seed,
        "sizes": dict(sizes),
        "dedup": {"offered": deduper.offered, **deduper.dropped},
        "topics": {split: [t.id for t in topics[split]] for split in SPLITS},
        "paper_ids": {split: sum(len(t.papers) for t in topics[split]) for split in SPLITS},
        "families": {split: sum(1 for s in assignment.values() if s == split) for split in SPLITS},
        "window": list(corpus.window),
        "corpus_manifest_sha256": corpus.sha256,
        "planned_mix": shares,
    }
    if _incremental(seed, draw, built, starts, preloaded, shares):
        report.update({"draw_seed": draw, "only": list(built), "id_start": starts,
                       "preload": [p.summary() for p in preloaded],
                       "dedup_against_preloaded": dict(deduper.dropped_preloaded)})
    return Build(records, report, preloaded_records)


# ---- audit and writer -----------------------------------------------------------------------------------


def _paper_ids(slots: dict) -> list[str]:
    ids = list(slots.get("arxiv_ids") or [])
    if slots.get("arxiv_id"):
        ids.append(slots["arxiv_id"])
    return ids


def audit(records: list[dict]) -> dict[str, dict[str, list]]:
    """kind -> "<split>/<split>" -> the sorted values two splits share, for families, topics (topic_id) and
    paper ids (arxiv_id and arxiv_ids, canonicalised). Empty inner dicts mean no leakage."""
    seen: dict[str, dict[str, set]] = {k: {s: set() for s in SPLITS}
                                       for k in ("families", "topics", "paper_ids")}
    for r in records:
        split, slots = r["split"], r.get("slots") or {}
        seen["families"][split].add(str(r["family"]))
        if slots.get("topic_id"):
            seen["topics"][split].add(slots["topic_id"])
        for aid in _paper_ids(slots):
            seen["paper_ids"][split].add(normalize_arxiv_id(aid) or aid)
    out: dict[str, dict[str, list]] = {}
    for kind, per_split in seen.items():
        out[kind] = {}
        for i, a in enumerate(SPLITS):
            for b in SPLITS[i + 1:]:
                shared = per_split[a] & per_split[b]
                if shared:
                    out[kind][f"{a}/{b}"] = sorted(shared)
    return out


def audit_summary(overlaps: dict[str, dict[str, list]]) -> dict:
    """{"families": n, "topics": n, "paper_ids": n, "ok": bool}, n counting shared values over split pairs."""
    counts = {kind: sum(len(v) for v in pairs.values()) for kind, pairs in overlaps.items()}
    return {**counts, "ok": not any(counts.values())}


def audit_line(overlaps: dict[str, dict[str, list]]) -> str:
    s = audit_summary(overlaps)
    return (f"audit: shared families {s['families']}, topics {s['topics']}, paper ids {s['paper_ids']} "
            f"across splits: {'ok' if s['ok'] else 'FAIL'}")


def template_wording(template: str) -> frozenset[str]:
    """The shingles of a template's own wording: each placeholder replaced by a space, then shingles()."""
    return shingles(" ".join(_PLACEHOLDER.sub(" ", template).split()))


def template_audit(assignment: dict[str, str], records: Iterable[dict] | None = None) -> dict:
    """Wording overlap between templates of different splits: every pair of templates in the same language
    (any intent, any kind) whose families the assignment puts in different splits is compared by
    template_wording Jaccard. {"max_jaccard": the largest such value, "pairs": how many pairs were compared,
    "limit": TEMPLATE_LIMIT, "over": [[jaccard, family, family], ...] at or above the limit, "ok": bool}.
    Given records (a build's preloaded and new requests), it also checks where they use their families:
    "misplaced": [[family, the split a record uses it in, the split the assignment gives it], ...] for
    catalogue families used outside their split (each fails the audit, since the wording comparison assumes
    the assignment), and "uncatalogued": the sorted families outside the catalogue, whose wording this audit
    cannot see (the tool that made them audits it, as scripts/transform_synth.py does for its m6/ ones)."""
    rows = [(family_name(i, lang, n), lang, template_wording(t)) for i, _, lang, n, t in template_families()]
    best, pairs, over = 0.0, 0, []
    for x, (fa, lang_a, wa) in enumerate(rows):
        for fb, lang_b, wb in rows[x + 1:]:
            if lang_a != lang_b or assignment[fa] == assignment[fb]:
                continue
            pairs += 1
            j = jaccard(wa, wb)
            best = max(best, j)
            if j >= TEMPLATE_LIMIT:
                over.append([round(j, 4), fa, fb])
    result = {"max_jaccard": best, "pairs": pairs, "limit": TEMPLATE_LIMIT, "over": over, "ok": not over}
    if records is not None:
        misplaced, uncatalogued = set(), set()
        for r in records:
            family = str(r["family"])
            if family not in assignment:
                uncatalogued.add(family)
            elif assignment[family] != r["split"]:
                misplaced.add((family, r["split"], assignment[family]))
        result["misplaced"] = [list(m) for m in sorted(misplaced)]
        result["uncatalogued"] = sorted(uncatalogued)
        result["ok"] = not over and not misplaced
    return result


def template_audit_line(result: dict) -> str:
    families = ""
    if "misplaced" in result:
        families = (f", families used outside their split {len(result['misplaced'])}, outside the catalogue "
                    f"{len(result['uncatalogued'])} (their wording is not audited here)")
    return (f"template audit: max cross-split wording Jaccard {result['max_jaccard']:.4f} over "
            f"{result['pairs']} template pairs (limit {result['limit']}){families}: "
            f"{'ok' if result['ok'] else 'FAIL'}")


def counts_by_cell(records: list[dict]) -> dict[str, dict[str, dict[str, int]]]:
    """split -> scenario -> lang -> count."""
    out = {split: {s: dict.fromkeys(LANGS, 0) for s in SCENARIOS} for split in SPLITS}
    for r in records:
        out[r["split"]][r["scenario"]][r["lang"]] += 1
    return out


def requests_text(records: list[dict]) -> str:
    """The JSONL file content: one canonical JSON line per record, LF line ends."""
    return "".join(canonical_json(r) + "\n" for r in records)


def write_requests(records: list[dict], out: str | Path, manifest_path: str | Path, report: dict, *,
                   preloaded: Iterable[dict] = ()) -> dict:
    """Audit, then write the records (canonical JSON lines) and the manifest (sorted keys, indent 2), and
    return the manifest. Both audits run over the preloaded requests (Build.preloaded) and the records
    together; only the records are written. A ValueError naming every overlap is raised, and nothing is
    written, when any family, topic or paper id is shared by two splits, or when the template audit (on
    split_templates(report["seed"]), with the records' families when there are preloaded requests) finds two
    templates of different splits at or above TEMPLATE_LIMIT or a family used outside its split; also when
    the report's preload entries count other requests than the preloaded ones passed, or a record's id is a
    preloaded one (zero padding aside, as in check_ids)."""
    preloaded = list(preloaded)
    listed = sum(entry["requests"] for entry in report.get("preload", ()))
    if listed != len(preloaded):
        raise ValueError(f"the report lists {listed} preloaded requests but {len(preloaded)} were passed; "
                         "pass the build's preloaded requests (Build.preloaded), nothing written")
    held = {_id_key(r["id"]) for r in preloaded}
    taken = [r["id"] for r in records if _id_key(r["id"]) in held]
    if taken:
        raise ValueError(f"ids already in a preloaded file, nothing written: {_listing(taken)}")
    together = preloaded + list(records)
    overlaps = audit(together)
    summary = audit_summary(overlaps)
    if not summary["ok"]:
        named = "; ".join(f"{kind} {pair}: {', '.join(values[:10])}{' ...' if len(values) > 10 else ''}"
                          for kind, pairs in overlaps.items() for pair, values in pairs.items())
        raise ValueError(f"leakage audit failed, nothing written: {named}")
    wording = template_audit(split_templates(report["seed"]), together if preloaded else None)
    if not wording["ok"]:
        named = "; ".join([f"{a} / {b} {j}" for j, a, b in wording["over"]]
                          + [f"{family} used in {used}, a {assigned} template"
                             for family, used, assigned in wording.get("misplaced", [])])
        raise ValueError(f"template audit failed, nothing written: {named}")
    text = requests_text(records)
    data = text.encode("utf-8")
    manifest = {
        **report,
        "generator": GENERATOR_VERSION,
        "requests": len(records),
        "requests_sha256": hashlib.sha256(data).hexdigest(),
        "counts": counts_by_cell(records),
        "totals": {split: sum(1 for r in records if r["split"] == split) for split in SPLITS},
        "audit": summary,
        "template_audit": {"max_jaccard": round(wording["max_jaccard"], 4), "pairs": wording["pairs"],
                           "limit": wording["limit"], "ok": wording["ok"]},
        "planned_mix": dict(report.get("planned_mix", PLANNED_MIX)),
        "intents": dict(INTENTS),
    }
    if preloaded:  # a misplaced family stops the write above, so the count written is always 0
        manifest["template_audit"].update({"misplaced": len(wording["misplaced"]),
                                           "uncatalogued": wording["uncatalogued"]})
    out, manifest_path = Path(out), Path(manifest_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                             encoding="utf-8", newline="\n")
    return manifest

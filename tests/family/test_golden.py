"""The registered families against their real chat templates (docs/m7-家族层设计.md §9, the local tests).

For every folder of models/ that is present, the family it is identified as renders six cases (a step with
parallel calls after a first step, a plain answer after two steps, a clarification, a repair exchange whose
rejected call carries raw text arguments, a public-style sample with nested objects, and a THINK case), and
the prompt, the full render and the completion of each must equal the golden files under
tests/family/golden/<family>/. Beside the files: the full render is prompt + completion + what the template
writes after the terminator (TAIL: "\\n", or nothing for Spark); the sample splits at a token boundary, ends
with the terminator (encode_sample) and holds the BOS as the family's policy says (BOS); parse(answer) gives
the target's calls back, which the family's format renders as the template does (CALL_WRAP: Granite's
completion holds the blocks between the newline after its empty thinking block and the one after each block);
and the THINK construction, prompt(THINK) + reasoning, think_close and answer as the family joins them
(core.LoadedFamily.think_completion: Granite's template trims the close of a turn of calls alone), is the
template's own render of the target carrying reasoning_content, for the THINK case's call target and for the
no_call case's text target, and every family that thinks declares reasoning_field. Decoding
(docs/m7-计划.md §2, 解码): a family that thinks samples its thinking with the settings of the folder's
generation_config.json, the vendor's own, and where that file is absent or lacks a setting, with the model
card's (MODEL_CARD pins what each folder's README recommends), the repetition penalty among them (LFM2.5's
1.1, sent as llama-server's repeat_penalty); a family that never thinks is greedy. Every other field of the
file is one no sampling depends on, or a sampling setting no request carries (core.Decoding and
llm.local.LocalClient have no field for it) at the value that changes nothing; UNCARRIED would list a known
gap, and lists none. Three more checks
against the real templates and tokenizers: a call holding every kind of value (KINDS: strings with each
character that calls for CDATA alone and together and ones that need escapes, numbers, booleans, null,
arrays and objects, empty ones) is written by the family's format exactly as the template writes it and read
back typed; the ids a THINK generation is closed with by force (encode_close) are the ids the close has in a
sample after a reasoning that ends with a word; and a sample's completion ids decode to its text, except
under the legacy tokenizer of LEGACY, whose decode the training run's mask check refuses.

Every test of one folder is selected by the folder's name, its parameter id: pytest -k Qwen3.5-4B (the
bake-off's smoke S1). Qwen3-4B's name is part of two other folders' names, so its own tests are
-k "Qwen3-4B and not Instruct and not Thinking".

The regression: for the first 300 training samples of data/raw/synth/synth-m6.jsonl and the first 300 of
data/raw/public/glaive_v2*.jsonl, qwen3-hybrid's NO_THINK prompt and completion are those
research_agent.train.samples.render_sample gives today.

models/ and data/raw/ are not in the repository: a test skips when its folder or data is absent, the way
the suite's other model-backed tests do (tests/train/test_train_samples.py).

Golden files are named <case>.<mode>.<kind>.txt and hold the exact UTF-8 text. To write them anew (a
template or a case changed on purpose), run these tests with FAMILY_GOLDEN_WRITE=1: each test then writes
its files and skips, so a writing run never passes; review the diff, then run again without the variable.
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from support.fakes import HITS, PAPER, call

from research_agent.agent.context import step_contexts
from research_agent.agent.types import ToolCall, ToolResult
from research_agent.canonical import canonical_json
from research_agent.family.core import (
    GREEDY,
    Decoding,
    FamilyError,
    LoadedFamily,
    Mode,
    Thinking,
    load_family,
)
from research_agent.family.registry import FAMILIES
from research_agent.tools.schemas import CONTEXT_TOKENS, INVALID_JSON, MODEL_TOOLS
from research_agent.train import samples

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models"
DOWNLOADS = REPO / "runs" / "bakeoff" / "m7" / "downloads"  # the HF revisions the folders were fetched at
GOLDEN = Path(__file__).resolve().parent / "golden"
WRITE = os.environ.get("FAMILY_GOLDEN_WRITE") == "1"
FOLDERS = {
    "Qwen3-1.7B": "qwen3-hybrid", "Qwen3-4B": "qwen3-hybrid",
    "Qwen3-4B-Instruct-2507": "qwen3-instruct-2507", "Qwen3-4B-Thinking-2507": "qwen3-thinking-2507",
    "Qwen3.5-4B": "qwen3.5-4b", "granite-4.2-3b": "granite-4.2-3b",
    "Nanbeige4-3B-Thinking-2511": "nanbeige4-thinking-2511", "Nanbeige4.1-3B": "nanbeige4.1-3b",
    "MiniCPM5-2B": "minicpm5-2b", "Spark-X2.5-4B": "spark-x2.5-4b", "LFM2.5-2.6B": "lfm2.5-2.6b",
}
# The BOS token of each family's tokenizer (None: it has none) and the family's policy: "none" with a token is
# a template that never writes it (Granite's "<s>"), "template" one that writes it once at the start,
# "turn" one whose BOS opens every turn.
BOS = {
    "qwen3-hybrid": (None, "none"), "qwen3-instruct-2507": (None, "none"),
    "qwen3-thinking-2507": (None, "none"), "qwen3.5-4b": (None, "none"), "granite-4.2-3b": ("<s>", "none"),
    "nanbeige4-thinking-2511": ("<|im_start|>", "turn"), "nanbeige4.1-3b": ("<|im_start|>", "turn"),
    "minicpm5-2b": ("<s>", "template"), "spark-x2.5-4b": ("<｜start▁of▁sentence｜>", "turn"),
    "lfm2.5-2.6b": ("<|startoftext|>", "template"),
}
TAIL = {"spark-x2.5-4b": ""}  # what the template writes after the terminator; "\n" for every other family
CALL_WRAP = {"granite-4.2-3b": ("\n", "\n")}  # around the format's render of the calls in a completion
MODEL_CARD = {  # the sampling settings a folder's generation_config.json does not hold, from its README.md
    # (read on 2026-10-20): Qwen3.5-4B has no generation_config.json ("Thinking mode for general tasks"); the
    # Nanbeige files hold no sampling field ("Temperature: 0.6, Top-p: 0.95")
    "Qwen3.5-4B": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5},
    "Nanbeige4-3B-Thinking-2511": {"temperature": 0.6, "top_p": 0.95},
    "Nanbeige4.1-3B": {"temperature": 0.6, "top_p": 0.95},
}
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")
NEUTRAL = {"frequency_penalty": 0.0}  # not carried; this value changes nothing
UNCARRIED: dict[str, dict] = {}  # the vendor's settings no request carries: none (LFM2.5's repetition
# penalty, the one there was, is carried since core.Decoding has the field)
NOT_SAMPLING = {"do_sample", "bos_token_id", "eos_token_id", "pad_token_id", "max_tokens", "use_cache",
                "output_attentions", "output_hidden_states", "_from_model_config", "transformers_version"}
SYNTH = REPO / "data" / "raw" / "synth" / "synth-m6.jsonl"
GLAIVE = sorted((REPO / "data" / "raw" / "public").glob("glaive_v2*.jsonl"))
FIRST = 300  # training samples compared per source


def present(folder: str) -> bool:
    """The folder holds what a family needs: its configuration and its tokenizer files (not the weights)."""
    return all((MODELS / folder / name).is_file()
               for name in ("config.json", "tokenizer_config.json", "tokenizer.json"))


FOLDER_PARAMS = [pytest.param(folder, marks=pytest.mark.skipif(not present(folder),
                                                               reason="local model files are not present"))
                 for folder in FOLDERS]


@functools.cache
def family_of(folder: str) -> LoadedFamily:
    return load_family(MODELS / folder)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---- the six cases --------------------------------------------------------------------------------------

# A fixed system message, not agent.prompts': the golden files follow the templates, not the prompt's wording.
SYSTEM = {"role": "system", "content": "你是科研助理,帮用户查找和阅读 arXiv 论文。今天是 2026-10-19。"}
SECOND_PAPER = {**PAPER, "arxiv_id": "2609.05678", "title": "Tree Verification at Scale"}
HISTORY = [
    SYSTEM,
    {"role": "user", "content": "帮我找 2026 年 9 月关于 speculative decoding 的论文,对比前两篇"},
    {"role": "assistant", "content": "", "tool_calls": [
        call("search_arxiv", "call_1", query="speculative decoding", date_from="2026-09-01").to_api()]},
    {"role": "tool", "tool_call_id": "call_1", "content": canonical_json(HITS)},
    {"role": "assistant", "content": "", "tool_calls": [
        call("get_paper", "call_2", arxiv_id="2609.01234").to_api(),
        call("get_paper", "call_3", arxiv_id="2609.05678").to_api()]},
    {"role": "tool", "tool_call_id": "call_2", "content": canonical_json(PAPER)},
    {"role": "tool", "tool_call_id": "call_3", "content": canonical_json(SECOND_PAPER)},
    {"role": "assistant", "content": "两篇都发表于 2026 年 9 月:前者按接受率调整草稿长度,"
                                     "后者研究大规模服务中的树形验证。"},
]
STEPS = step_contexts(HISTORY)  # (context, target) per assistant message, the context as the loop showed it

REJECTED = ToolCall(id="call_9", name="search_arxiv", arguments_json="query=KV cache compression",
                    arguments=None)
REPAIR_CONTEXT = [  # agent.loop's repair request: the context, the rejected step and its errors
    SYSTEM, {"role": "user", "content": "找最近关于 KV cache 压缩的论文"},
    {"role": "assistant", "content": "", "tool_calls": [REJECTED.to_api()]},
    ToolResult("call_9", "search_arxiv", {"details": ["the arguments are not a JSON object"]},
               INVALID_JSON).to_api(),
]
REPAIR_TARGET = {"role": "assistant", "content": "", "tool_calls": [
    call("search_arxiv", "call_10", query="KV cache compression").to_api()]}

CLARIFY_CONTEXT = [SYSTEM, {"role": "user", "content": "帮我总结一下那篇论文的方法部分"}]
CLARIFY_TARGET = {"role": "assistant", "content": "", "tool_calls": [call(
    "ask_clarification", "call_4", question="你指的是哪一篇论文?请给出 arXiv 编号或标题。",
    missing=["paper_id"]).to_api()]}

CHILD = {"type": "object", "properties": {"age": {"type": "integer"}}}
PARTY = {"adults": {"type": "integer"}, "children": {"type": "array", "items": CHILD}}
BOOK_TABLE = {"type": "function", "function": {
    "name": "book_table", "description": "Book a table at a restaurant.",
    "parameters": {"type": "object", "properties": {
        "restaurant": {"type": "object", "properties": {"name": {"type": "string"},
                                                        "city": {"type": "string"}}},
        "party": {"type": "object", "properties": PARTY},
        "time": {"type": "string"},
        "requests": {"type": "array", "items": {"type": "string"}}},
        "required": ["restaurant", "party", "time"]}}}
BOOKING = {"restaurant": {"name": "Café de Flore", "city": "Paris"},
           "party": {"adults": 2, "children": [{"age": 5}]},
           "time": "19:30", "requests": ["window seat", "high chair"]}
PUBLIC = [  # the TRL conversational form the public-data converter writes, with the sample's own tool
    {"role": "user", "content": "Book a table at Café de Flore in Paris for two adults and a five-year-old "
                                "at 19:30, by the window, with a high chair."},
    {"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function", "function": {
        "name": "book_table", "arguments": json.dumps(BOOKING, ensure_ascii=False)}}]},
]

KV_HITS = [
    {"arxiv_id": "2609.04321", "title": "Quantized KV Cache Compression for Long Contexts",
     "published": "2026-09-12", "snippet": "We compress the KV cache to 2 bits per value with little loss."},
    {"arxiv_id": "2609.07777", "title": "Eviction Policies for Streaming Attention",
     "published": "2026-09-25", "snippet": "Which cached tokens to evict when the context window is full."},
]
THINK_HISTORY = [
    SYSTEM,
    {"role": "user", "content": "找 2026 年 9 月以来 cs.CL 里关于 KV cache 压缩的论文,挑最相关的一篇看摘要"},
    {"role": "assistant", "content": "", "tool_calls": [call(
        "search_arxiv", "call_5", query="KV cache compression", categories=["cs.CL"],
        date_from="2026-09-01").to_api()]},
    {"role": "tool", "tool_call_id": "call_5", "content": canonical_json(KV_HITS)},
    {"role": "assistant", "content": "", "tool_calls": [
        call("get_paper", "call_6", arxiv_id="2609.04321").to_api()]},
]
REASONING = {  # the reasoning of the THINK constructions: the think case's call target, no_call's text target
    "think": ("The search returned two cs.CL papers. The first one, on quantized KV cache compression, "
              "matches the request best, so I fetch its abstract."),
    "no_call": "Both abstracts are in hand, so I compare the two papers for the user in one short answer.",
}


@dataclass(frozen=True)
class Case:
    context: list[dict]  # API messages, as the loop or the public data hold them
    target: dict  # the assistant message to produce
    tools: list[dict]


CASES = {
    "parallel": Case(*STEPS[1], tools=MODEL_TOOLS),
    "no_call": Case(*STEPS[2], tools=MODEL_TOOLS),
    "clarify": Case(CLARIFY_CONTEXT, CLARIFY_TARGET, tools=MODEL_TOOLS),
    "repair": Case(REPAIR_CONTEXT, REPAIR_TARGET, tools=MODEL_TOOLS),
    "public_nested": Case(PUBLIC[:1], PUBLIC[1], tools=[BOOK_TABLE]),
    "think": Case(*step_contexts(THINK_HISTORY)[1], tools=MODEL_TOOLS),
}


def calls_of(target: dict) -> list[dict]:
    """The target's calls in the parser's contract ({"name", "arguments"}), arguments read as the agent
    reads them."""
    return [{"name": c.name, "arguments": c.arguments}
            for c in (ToolCall.from_api(c) for c in target.get("tool_calls") or [])]


def no_think(family: LoadedFamily, context: list[dict], target: dict,
             tools: list[dict]) -> tuple[str, str, str]:
    """docs/m7-家族层设计.md §5's NO_THINK prompt and completion, built through the family, and the full
    render."""
    hf = family.to_template_messages(context)
    prompt = family.prompt(hf, tools, Mode.NO_THINK)
    full = family.full(hf + family.to_template_messages([target]), tools, Mode.NO_THINK)
    assert full.startswith(prompt)
    completion = full[len(prompt):]
    if completion.endswith(family.spec.turn_end + "\n"):
        completion = completion[:-1]
    assert completion.endswith(family.spec.turn_end)
    return prompt, completion, full


def check_bos(family: LoadedFamily, ids: list[int]) -> None:
    """The ids hold the BOS as the family's policy says (BOS)."""
    token, policy = BOS[family.spec.name]
    assert family.spec.bos == policy and family.tokenizer.bos_token == token
    if token is None:
        assert family.bos_id is None
        return
    count = ids.count(family.bos_id)
    if policy == "none":
        assert count == 0
    elif policy == "template":
        assert count == 1 and ids[0] == family.bos_id
    else:
        assert count >= 2 and ids[0] == family.bos_id  # every turn opens with it


class Golden:
    """Compares texts with the golden files, or writes them (module docstring)."""

    def __init__(self) -> None:
        self.written: list[Path] = []

    def check(self, family: str, name: str, text: str) -> None:
        path = GOLDEN / family / name
        if WRITE:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))
            self.written.append(path)
            return
        assert path.is_file(), f"no golden file {path}: write it with FAMILY_GOLDEN_WRITE=1 (docstring)"
        assert path.read_bytes().decode("utf-8") == text, f"{path} differs from the render"

    def done(self) -> None:
        if self.written:
            pytest.skip(f"wrote {len(self.written)} golden files under {GOLDEN}")


# ---- the tests ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("folder", FOLDER_PARAMS)
def test_a_model_folder_is_identified_as_its_registered_family(folder):
    family = family_of(folder)
    spec, folder_path = family.spec, MODELS / folder
    assert spec.name == FOLDERS[folder] and family.template_sha256 in spec.template_sha256
    token, policy = BOS[spec.name]
    assert (family.tokenizer.bos_token, spec.bos, family.bos_id is None) == (token, policy, token is None)
    assert family.decode([family.turn_end_id]) == spec.turn_end
    record = family.record()
    assert record["tokenizer_json_sha256"] == sha256_of(folder_path / "tokenizer.json")
    # vendor code runs for Spark alone, and only as pinned: the folder's Python files hash as the pins say
    assert spec.trust_remote_code is (spec.name == "spark-x2.5-4b")
    assert record["remote_files"] == {name: sha256_of(folder_path / name) for name in spec.remote_files}
    if spec.trust_remote_code:  # every Python file of the folder is pinned, and the revision is the folder's
        assert {p.name for p in folder_path.glob("*.py")} == set(spec.remote_files)
        downloads = DOWNLOADS / f"{folder}.json"
        if downloads.is_file():
            assert json.loads(downloads.read_text(encoding="utf-8"))["revision"] == spec.revision
    else:
        assert not spec.remote_files and spec.revision is None
    assert load_family(folder_path, name=FOLDERS[folder], tokenizer=family.tokenizer).spec is spec
    other = next(f.name for f in FAMILIES if f.name != FOLDERS[folder])
    with pytest.raises(FamilyError, match="does not agree"):
        load_family(folder_path, name=other, tokenizer=family.tokenizer)


@pytest.mark.parametrize("folder", FOLDER_PARAMS)
def test_a_family_thinks_with_the_sampling_of_its_folders_generation_config_or_model_card(folder):
    spec = family_of(folder).spec
    if spec.thinking is Thinking.NONE:
        assert dict(spec.decoding) == {Mode.NO_THINK: GREEDY}
        return
    path = MODELS / folder / "generation_config.json"
    vendor = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    card = MODEL_CARD.get(folder, {})
    assert not (set(vendor) & set(card))  # the card fills in only what the file lacks
    assert vendor.get("do_sample", True) is True
    settings = {}
    for field in SAMPLING_FIELDS:
        value = vendor.get(field, card.get(field))
        if value is not None:
            settings[field] = 0 if field == "top_k" and value == -1 else value  # top_k -1 is off, as 0 is
    assert settings, f"{folder}: neither the generation config nor the model card names a sampling setting"
    assert spec.decoding[Mode.THINK] == Decoding(**settings)
    # every other field is no sampling setting, or one no request carries at a value that changes nothing,
    # or a known gap
    assert set(vendor) <= NOT_SAMPLING | set(SAMPLING_FIELDS) | set(NEUTRAL), sorted(vendor)
    uncarried = {field: value for field, neutral in NEUTRAL.items()
                 if (value := vendor.get(field, neutral)) != neutral}
    assert uncarried == UNCARRIED.get(folder, {})
    assert not set(NEUTRAL) & set(Decoding.__dataclass_fields__)  # the gap is real: no request sends them
    if spec.thinking is Thinking.ONLY:  # it samples when answering at once too; a hybrid answers greedily
        assert spec.decoding[Mode.NO_THINK] == spec.decoding[Mode.THINK]
    else:
        assert spec.decoding[Mode.NO_THINK] == GREEDY


@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("folder", FOLDER_PARAMS)
def test_each_case_renders_as_its_golden_files(folder, case):
    family, golden = family_of(folder), Golden()
    spec, item = family.spec, CASES[case]
    prompt, completion, full = no_think(family, item.context, item.target, item.tools)
    golden.check(spec.name, f"{case}.no_think.prompt.txt", prompt)
    golden.check(spec.name, f"{case}.no_think.full.txt", full)
    golden.check(spec.name, f"{case}.no_think.completion.txt", completion)
    golden.done()
    assert full == prompt + completion + TAIL.get(spec.name, "\n")
    ids, start = family.encode_sample(prompt, completion)  # the token prefix, the BOS rule, the terminator
    assert ids[:start] == family.encode_prompt(prompt) and ids[-1] == family.turn_end_id
    check_bos(family, ids)
    answer = completion.removesuffix(spec.turn_end)
    calls = calls_of(item.target)
    assert family.parse(answer, item.tools) == calls
    if calls:
        before, after = CALL_WRAP.get(spec.name, ("", ""))
        assert answer == before + spec.tool_format.render(calls) + after  # the format writes as the template
        spans = family.locate(answer)
        assert len(spans) == len(calls) and all(span.parsed for span in spans)
        assert [answer[a:b] for a, b in (span.name for span in spans)] == [c["name"] for c in calls]
        assert all(len(span.values) == len(c["arguments"]) for span, c in zip(spans, calls, strict=True))
    else:
        assert answer == item.target["content"]


# Every kind of value an argument can hold, beyond the six cases' MODEL_TOOLS-shaped ones (public samples and
# ACEBench carry them all): each format must write it as its template does and read it back, typed by the
# schema (toolformats: coerce). The strings hold each character MiniCPM5's template writes a CDATA section for
# ("<", "&", a newline) alone and all together, and ">", which it writes none for.
KINDS_TOOL = {"type": "function", "function": {
    "name": "set_values", "description": "Store one value of every kind.",
    "parameters": {"type": "object", "properties": {
        "text": {"type": "string"}, "markup": {"type": "string"}, "ampersand": {"type": "string"},
        "less": {"type": "string"}, "newline": {"type": "string"}, "greater": {"type": "string"},
        "quoted": {"type": "string"}, "empty": {"type": "string"}, "count": {"type": "integer"},
        "negative": {"type": "integer"}, "ratio": {"type": "number"}, "yes": {"type": "boolean"},
        "no": {"type": "boolean"}, "nothing": {"type": "null"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "numbers": {"type": "array", "items": {"type": "number"}}, "none_yet": {"type": "array"},
        "nested": {"type": "object"}, "blank": {"type": "object"}}}}}
KINDS = {"text": "幻觉 hallucination", "markup": "a <b> & c\nline two", "ampersand": "R&D only",
         "less": "x < y", "newline": "one\ntwo", "greater": "a > b", "quoted": "it's \"q\" \\ path",
         "empty": "", "count": 5, "negative": -3, "ratio": 1.5, "yes": True, "no": False, "nothing": None,
         "tags": ["cs.CL", "cs.LG"], "numbers": [1, 2.5], "none_yet": [],
         "nested": {"a": {"b": [True, None]}, "c": "d"}, "blank": {}}


@pytest.mark.parametrize("folder", FOLDER_PARAMS)
def test_every_kind_of_value_is_written_as_the_template_writes_it_and_read_back(folder):
    family = family_of(folder)
    spec = family.spec
    call_0 = {"id": "call_0", "type": "function",
              "function": {"name": "set_values", "arguments": json.dumps(KINDS, ensure_ascii=False)}}
    target = {"role": "assistant", "content": "", "tool_calls": [call_0]}
    context = [{"role": "user", "content": "Store one value of every kind."}]
    _, completion, _ = no_think(family, context, target, [KINDS_TOOL])
    answer = completion.removesuffix(spec.turn_end)
    calls = calls_of(target)
    assert calls == [{"name": "set_values", "arguments": KINDS}]
    before, after = CALL_WRAP.get(spec.name, ("", ""))
    assert answer == before + spec.tool_format.render(calls) + after
    assert family.parse(answer, [KINDS_TOOL]) == calls
    (span,) = family.locate(answer)
    assert span.parsed and answer[span.name[0]:span.name[1]] == "set_values"
    assert len(span.values) == len(KINDS) and list(span.values) == sorted(span.values)


# The family whose tokenizer is legacy SentencePiece (its tokenizer_config.json: "legacy": true): it encodes a
# "▁" after every added token, so a sample's completion ids decode to its text with a space after each added
# token in it. scripts/train.py's mask check holds the decode against the completion's text and so refuses a
# training run of this family (exit 3), whereas Nanbeige4.1's tokenizer (the same tokenizer.json, "legacy":
# false) decodes the text back.
LEGACY = {"nanbeige4-thinking-2511"}


@pytest.mark.parametrize("folder", FOLDER_PARAMS)
def test_a_samples_completion_ids_decode_to_its_text_except_under_a_legacy_tokenizer(folder):
    family = family_of(folder)
    added = {token.content for token in family.tokenizer.added_tokens_decoder.values()}
    differs = []
    for item in CASES.values():
        prompt, completion, _ = no_think(family, item.context, item.target, item.tools)
        ids, start = family.encode_sample(prompt, completion)
        decoded = family.decode(ids[start:])
        if family.spec.name in LEGACY:  # the text with a space after each added token
            restored = decoded
            for token in added:
                restored = restored.replace(token + " ", token)
            assert restored == completion
        else:
            assert decoded == completion
        differs.append(decoded != completion)
    assert any(differs) is (family.spec.name in LEGACY)


# The families whose tokenizer writes a word-start marker at the start of a text (SentencePiece's dummy
# prefix), so think_close encoded on its own starts with "▁" where a sample holds none (encode_close).
DUMMY_PREFIX = {"nanbeige4-thinking-2511", "nanbeige4.1-3b"}


@pytest.mark.parametrize("case", ["no_call", "think"])
@pytest.mark.parametrize("folder", FOLDER_PARAMS)
def test_a_forced_close_has_the_ids_the_close_has_in_a_sample_after_a_reasoning_ending_with_a_word(folder,
                                                                                                    case):
    # a forced close continues the generation as the THINK samples go on after their reasoning, a text answer
    # (no_call) and a call (think) alike: Granite's call samples, joined as its template trims them, hold the
    # close and then the call, where the three in a row held a second newline the close's ids do not have
    family = family_of(folder)
    spec = family.spec
    if spec.thinking is Thinking.NONE:
        with pytest.raises(FamilyError, match="never thinks"):
            family.encode_close()
        return
    item = CASES[case]
    prompt = family.prompt(family.to_template_messages(item.context), item.tools, Mode.THINK)
    _, answer, _ = no_think(family, item.context, item.target, item.tools)
    (target,) = family.to_template_messages([item.target])
    reasoning = "So I compare the two papers"  # it ends with a word, which nothing merges with
    ids, _ = family.encode_sample(prompt, family.think_completion(reasoning, answer, target))
    before = family.encode_prompt(prompt + reasoning)
    close = family.encode_close()
    assert ids[:len(before)] == before and ids[len(before):len(before) + len(close)] == close
    # what the model writes after the forced close is the answer the split reads after a close
    rest = family.decode(ids[len(before) + len(close):])
    assert rest == answer.lstrip("\n") or spec.name in LEGACY
    alone = family.encode_text(spec.think_close)
    if spec.name in DUMMY_PREFIX:
        assert family.tokenizer.convert_ids_to_tokens(alone[:1]) == ["▁"] and alone[1:] == close
    else:  # every other family's forced close is the close's own ids, as before encode_close
        assert alone == close


@pytest.mark.parametrize("case", list(REASONING))
@pytest.mark.parametrize("folder", FOLDER_PARAMS)
def test_the_think_construction_is_the_templates_own_render_beside_the_no_think_sample(folder, case):
    family, golden = family_of(folder), Golden()
    spec, item, reasoning = family.spec, CASES[case], REASONING[case]
    hf = family.to_template_messages(item.context)
    (target,) = family.to_template_messages([item.target])
    no_think_prompt, answer, _ = no_think(family, item.context, item.target, item.tools)
    if spec.thinking is Thinking.NONE:
        with pytest.raises(FamilyError, match="has no think mode"):
            family.prompt(hf, item.tools, Mode.THINK)
        return
    prompt = family.prompt(hf, item.tools, Mode.THINK)
    completion = family.think_completion(reasoning, answer, target)
    # every registered template reads reasoning_content, and every family that thinks declares it for the
    # load-time cross-check
    rendered = family.full(hf + [{**target, "reasoning_content": reasoning}], item.tools, Mode.THINK)
    golden.check(spec.name, f"{case}.think.prompt.txt", prompt)
    golden.check(spec.name, f"{case}.think.full.txt", rendered)
    golden.check(spec.name, f"{case}.think.completion.txt", completion)
    golden.done()
    calls = calls_of(item.target)
    assert rendered == prompt + completion + TAIL.get(spec.name, "\n")
    assert spec.reasoning_field == "reasoning_content"
    # the join is the three in a row, except where the template trims a turn of calls alone (Granite's)
    trims = spec.think_join == "trim" and calls and not (target.get("content") or "").strip()
    close = spec.think_close.rstrip() if trims else spec.think_close
    assert completion == reasoning + close + answer and (spec.name == "granite-4.2-3b") is (
        spec.think_join == "trim")
    assert prompt.endswith(spec.generation_prompt[Mode.THINK]) and prompt.endswith(spec.think_open)
    if spec.thinking is Thinking.ONLY:  # the template opens the block; NO_THINK closes it empty, as it does
        assert no_think_prompt == prompt + spec.think_close
    else:  # a hybrid template opens the block itself, or leaves it to the model and the family opens it
        template = family.tokenizer.apply_chat_template(hf, tools=item.tools, tokenize=False,
                                                        add_generation_prompt=True,
                                                        **dict(spec.template_kwargs[Mode.THINK]))
        assert prompt == (template if template.endswith(spec.think_open) else template + spec.think_open)
    ids, start = family.encode_sample(prompt, completion)
    assert ids[:start] == family.encode_prompt(prompt) and ids[-1] == family.turn_end_id
    check_bos(family, ids)
    read, offset = family.split_reasoning(completion)
    # the split skips the newlines after the close, so the newline a Granite call completion starts with
    # (CALL_WRAP) is read as part of the close at inference, where the answer is parsed without it
    assert (read, completion[offset:]) == (reasoning, answer.lstrip("\n"))
    assert family.parse(completion[offset:].removesuffix(spec.turn_end), item.tools) == calls


def training_pairs(record: object) -> tuple[list[tuple[list[dict], dict]], list[dict] | None]:
    """The (context, target) pairs build_step_samples cuts a record of these files into and the tools it
    renders them with, through train.samples' own helpers (scripts/train.py passes the default tools,
    MODEL_TOOLS)."""
    if samples.is_agent_history(record):
        return samples._agent_pairs(record)
    return samples._public_pairs(record, MODEL_TOOLS)


def training_samples(paths: list[Path],
                     family: LoadedFamily) -> Iterator[tuple[list[dict], dict, list, dict]]:
    """(context, target, tools, render_sample's sample) of the files' training samples, in order: a sample
    over CONTEXT_TOKENS, which build_step_samples drops, is not one."""
    for path in paths:
        for record in samples.load_records(path):
            pairs, tools = training_pairs(record)
            for context, target in pairs:
                sample = samples.render_sample(family.tokenizer, context, target, tools=tools)
                if sample["tokens"] <= CONTEXT_TOKENS:
                    yield context, target, tools, sample


@pytest.mark.skipif(not (present("Qwen3-1.7B") and SYNTH.is_file() and GLAIVE),
                    reason="local model files or training data are not present")
@pytest.mark.parametrize("source", ["synth", "glaive"])
def test_the_hybrid_familys_no_think_samples_are_render_samples_on_the_first_training_samples(source):
    family = family_of("Qwen3-1.7B")
    compared = 0
    for context, target, tools, sample in itertools.islice(
            training_samples([SYNTH] if source == "synth" else GLAIVE, family), FIRST):
        prompt, completion, _ = no_think(family, context, target, tools)
        assert (prompt, completion) == (sample["prompt"], sample["completion"])
        ids, _ = family.encode_sample(prompt, completion)
        assert len(ids) == sample["tokens"] and ids[-1] == family.turn_end_id
        compared += 1
    assert compared == FIRST

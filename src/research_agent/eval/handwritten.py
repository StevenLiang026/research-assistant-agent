"""Hand-written evaluation items (docs/plan.md, 评测: at least 30 % of the items written from scratch).

The drafter (research_agent.eval.items) cuts items out of teacher transcripts; the items here start from a
hand-written spec, one JSON object per line of data/eval/handwritten-specs.jsonl, and are built so that
they are indistinguishable in form from drafted ones. A spec has exactly SPEC_KEYS:

- id: "hw-<split>-<nnn>", unique;
- split ("dev" or "test"), lang ("zh" or "en"), type (one of items.TYPES);
- request: the user's message, written for this item, in the item's language;
- slots: the request's constraints as request records carry them (topic, topic_id, date_from, date_to,
  since, count, categories, tags, tag, arxiv_id, arxiv_ids, section, mode, sort_by, missing), plus
  required_words, the words a correct query must contain (metrics.required_words);
- prior: the steps taken before the decision, a list of steps, each a non-empty list of calls {"name",
  "arguments"}; in a recover item the one call of the last step carries "inject": one of INJECTABLE, and is
  answered with that error instead of its real output;
- expected: {"content": str} for an answer without a call, or {"tool_calls": [{"name", "arguments"}]};
- rationale: one sentence on why the expected turn is the single most appropriate next action.

build_item turns a spec into an item exactly like a drafted one. The prior steps are played through the
agent loop itself (agent.loop.run_turn) by a scripted model, over the tools the generator builds for a request
(data.synth.make_tools at fault rate 0: the corpus index for search_arxiv, the replay fetcher and the owner's
parse_atom and extract_section for get_paper and fetch_fulltext, and a note store in a scratch folder seeded
by seed_plan and seed_notes for the item id, as the generator seeds the store of a request with that id), with
the owner's check_call and decide. So messages are [system_prompt(today), the request, then per prior step
the assistant message with its tool calls (ids c1, c2, ... over the whole history) and one tool message per
call], byte for byte what the loop records; an injected call runs the real tool first (a replay miss still
surfaces) and is then answered as the fault injectors answer: ([], code) for search_arxiv (net.faults.
SearchFaults), ({}, code) for get_paper and fetch_fulltext (the adapters' answer to a failed fetch), and an
EMPTY_RESULTS search gets the loop's relax hint from decide. The item is {"id", "messages", "expected":
{"content", "tool_calls": [{"name", "arguments", "arguments_json"}]}, "type", "lang", "split", "family":
"handwritten/<type>/<lang>", "slots", "source": {"handwritten": true}}, arguments_json being
json.dumps(arguments, ensure_ascii=False).

Checks (every failure is a SpecError naming the item id; check_items collects them all):

1. Shape: the keys above with their types; an id matching ID_PATTERN whose split part is the split; a zh
   request holding Chinese characters and an en request none; INJECTABLE codes only on search_arxiv,
   get_paper or fetch_fulltext, EMPTY_RESULTS only on a search with a date filter, in the last prior step,
   which then has that one call; an inject in a recover item and nowhere else.
2. Replay: every prior step accepted by the executor (check_call) at the first attempt, no harness error, no
   ask_clarification before the decision.
3. Type: items.item_type of the expected turn after that history equals the spec's type; beyond that a call
   or next_step item has at most one call, a parallel item exactly two different calls, a clarify item only
   ask_clarification with a non-empty "missing" equal as a set to slots["missing"], a no_call item a
   non-empty answer.
4. Validity: every expected call passes validation.check_call with the ids seen in the history.
5. Last message: the history ends with a user or tool message (runner.LAST_ROLES).
6. Scoring: the expected turn, played back as the model's output, is a success under metrics.score_item;
   this also holds every arXiv id of the answer to the history.
7. Recovery: after RATE_LIMITED or UPSTREAM_ERROR the expected turn is the injected call again, alone
   (call_key, defaults filled in); after EMPTY_RESULTS it is the same search without date_from and date_to.
8. Requested arguments only: a date argument appears verbatim in the request; a category, tag or note mode
   appears in the request; a count argument (max_results, top_k, limit) equals slots["count"]; section equals
   slots["section"], sort_by slots["sort_by"]; an argument set to its schema default is one the request asks
   for in these terms; the four write_note content fields are not used; a query has required words
   (metrics.required_words non-empty), so the scoring never falls back to literal equality.
9. Split discipline: the topics of the slots are topics of the item's split, the request names no topic of
   another split (its id, Chinese name or phrases, casefolded), every arXiv id of the request, the slots,
   the prior calls and the expected turn is a paper of the split's topics, and every ISO date of the request,
   the slots and the calls lies in the recording window, today included.
10. Wording: the masked request (data.requests.mask_text) has a 5-gram shingle Jaccard below TEMPLATE_LIMIT
   (0.6) against every generated request text (all splits, so no template wording returns) and below
   HANDWRITTEN_LIMIT (0.8) against every other hand-written request; no two specs share an id. With
   World.retired (the second family, below), each retired file's limit against its texts as well, every
   file read once for all the specs.

The second family (v2, M7: docs/m7-计划.md WP-I 4; scripts/build_handwritten_items.py --family v2) has
hand-written items in the test split only, numbered from hw-test-165 (FIRST_ID_V2; v1 used hw-test-001 to
hw-test-164; numbering_problems). Its items pass the same ten checks over a wider world. Check 10 compares
with the generated requests of every request file (request_texts: v1's of every split, M6's and M7's training
requests, and v2's development and test requests) and with v1's two retired files, read only through
eval.lock.RetiredSet, which returns the similarities alone and logs every read (retired_v1): v1's
hand-written requests, dev and test (data/eval/handwritten-specs.jsonl), stay below HANDWRITTEN_LIMIT, and
the user requests of v1's single-step test items (data/eval/test.jsonl, its hand-written items among them)
below TEMPLATE_LIMIT. The specs are written in batches by a plan
(writing_plan: per stratum of the review's needed counts, needed + spares(needed) specs, ids in stratum order,
at most BATCH_SIZE specs and one stratum per batch); batch_problems holds a writer's batch to its entry in the
plan, peer_problems to the other batches present, and replayed shows a writer the history a spec leads to.
"""

from __future__ import annotations

import copy
import json
import re
import shutil
import tempfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..agent.executor import ToolExecutor
from ..agent.loop import ASK, run_turn
from ..agent.types import ANSWERED, AssistantTurn, ToolCall, ToolFn, ToolOutcome, call_key, parse_arguments
from ..data.requests import NEAR_DUP, TRAIN_VS_EVAL, jaccard, mask_text, shingles
from ..data.synth import (
    OwnerFunctions,
    SeedPool,
    make_tools,
    request_topic,
    seed_notes,
    seed_plan,
    store_note_tools,
)
from ..normalize import normalize_text
from ..tools.corpus_search import CorpusIndex
from ..tools.schemas import DEFAULTS, EMPTY_RESULTS, MISSING, RATE_LIMITED, UPSTREAM_ERROR
from .items import CALL, CLARIFY, LANGS, NEXT_STEP, NO_CALL, PARALLEL, RECOVER, SPLITS, TYPES, item_type
from .lock import RetiredSet
from .metrics import required_words, score_item
from .runner import LAST_ROLES, OK

SPEC_KEYS = ("id", "split", "lang", "type", "request", "slots", "prior", "expected", "rationale")
ID_PATTERN = re.compile(r"hw-(dev|test)-(\d{3})")
INJECTABLE = (RATE_LIMITED, UPSTREAM_ERROR, EMPTY_RESULTS)
INJECT_TOOLS = ("search_arxiv", "get_paper", "fetch_fulltext")
DATE_ARGS = ("date_from", "date_to")
TEMPLATE_LIMIT = TRAIN_VS_EVAL  # 0.6: against every generated request text
HANDWRITTEN_LIMIT = NEAR_DUP  # 0.8: against every other hand-written request
COUNT_ARGS = ("max_results", "top_k", "limit")
CONTENT_FIELDS = ("problem", "method", "evidence", "limitations", "new_terms", "open_terms")
MODEL_NAME = "handwritten"
END_TEXT = "(end of the scripted prior steps)"
FIRST_ID_V2 = 165  # the second family's first id, hw-test-165: v1 used hw-test-001 to hw-test-164
BATCH_SIZE = 25  # the most specs a writer's batch holds (writing_plan)
_LAST_NUMBER = 999  # ID_PATTERN numbers an id with three digits

_HAN = re.compile(r"[一-鿿]")
_ARXIV_ID = re.compile(r"(?<![\d.])\d{4}\.\d{4,5}(?!\d)")
_ISO_DATE = re.compile(r"(?<!\d)\d{4}-\d{2}-\d{2}(?!\d)")


class SpecError(ValueError):
    """A spec that cannot become an item; the message starts with the item id."""


@dataclass(frozen=True)
class Retired:
    """A retired file the requests must stay apart from (check 10): what a problem calls it, its reader (an
    eval.lock.RetiredSet, or anything else whose similarities(texts) gives one float per text) and the limit
    a similarity must stay below. A problem names the spec and the similarity, never a text of the file."""

    name: str
    reader: object
    limit: float


@dataclass(frozen=True)
class World:
    """What items are built over: the corpus index, the fetcher behind get_paper and fetch_fulltext, the
    owner's functions, the seed pool (topics per split, the words naming them), the recording window
    (first day, last day = today), the generated request texts the wording check compares against, and the
    retired files it compares with too (none for v1)."""

    index: CorpusIndex
    fetcher: object
    owner: OwnerFunctions
    pool: SeedPool
    window: tuple[str, str]
    references: Sequence[str] = ()
    retired: Sequence[Retired] = ()
    _reference_shingles: list[frozenset[str]] = field(default_factory=list, repr=False, compare=False)

    @property
    def today(self) -> str:
        return self.window[1]

    def split_papers(self, split: str) -> set[str]:
        """The paper ids of the topics split may use."""
        return {a for t in self.pool.allowed(split) for a in self.pool.topics[t]}

    def reference_shingles(self) -> list[frozenset[str]]:
        if len(self._reference_shingles) != len(self.references):
            self._reference_shingles[:] = [shingles(mask_text(t)) for t in self.references]
        return self._reference_shingles


# ---- reading specs -----------------------------------------------------------------------------------------


def read_specs(path: str | Path) -> list[dict]:
    """The specs of a JSONL file in file order; a SpecError names the line that is not a JSON object."""
    specs = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").split("\n"), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SpecError(f"line {number}: not JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise SpecError(f"line {number}: not a JSON object")
        specs.append(value)
    return specs


def check_shape(spec: dict) -> None:
    """Check 1 of the module docstring."""
    sid = spec.get("id")
    where = f"{sid!r}" if isinstance(sid, str) else f"spec {sid!r}"
    if set(spec) != set(SPEC_KEYS):
        raise SpecError(f"{where}: keys must be {list(SPEC_KEYS)}, not {sorted(spec)}")
    match = ID_PATTERN.fullmatch(sid) if isinstance(sid, str) else None
    if match is None:
        raise SpecError(f"{where}: id must match {ID_PATTERN.pattern}")
    if spec["split"] not in SPLITS or match.group(1) != spec["split"]:
        raise SpecError(f"{sid}: split {spec['split']!r} must be one of {list(SPLITS)} and the id's")
    if spec["lang"] not in LANGS:
        raise SpecError(f"{sid}: lang must be one of {list(LANGS)}")
    if spec["type"] not in TYPES:
        raise SpecError(f"{sid}: type must be one of {list(TYPES)}")
    for key in ("request", "rationale"):
        if not isinstance(spec[key], str) or not spec[key].strip():
            raise SpecError(f"{sid}: {key} must be a non-empty string")
    has_han = bool(_HAN.search(spec["request"]))
    if has_han != (spec["lang"] == "zh"):
        raise SpecError(f"{sid}: a {spec['lang']} request must {'' if spec['lang'] == 'zh' else 'not '}"
                        "hold Chinese characters")
    if not isinstance(spec["slots"], dict):
        raise SpecError(f"{sid}: slots must be an object")
    _check_prior_shape(spec)
    _check_expected_shape(spec)


def _is_call(value: object) -> bool:
    return isinstance(value, dict) and isinstance(value.get("name"), str) and \
        isinstance(value.get("arguments"), dict)


def _check_prior_shape(spec: dict) -> None:
    sid, prior = spec["id"], spec["prior"]
    if not isinstance(prior, list) or not all(isinstance(s, list) and s for s in prior):
        raise SpecError(f"{sid}: prior must be a list of non-empty lists of calls")
    injected = []
    for k, step in enumerate(prior):
        for call in step:
            if not _is_call(call) or not set(call) <= {"name", "arguments", "inject"}:
                raise SpecError(f"{sid}: prior step {k + 1}: a call is {{name, arguments[, inject]}}: "
                                f"{call!r}")
            if call["name"] == ASK:
                raise SpecError(f"{sid}: prior step {k + 1} calls {ASK}, which ends the turn")
            if "inject" in call:
                injected.append((k, call))
    if spec["type"] != RECOVER:
        if injected:
            raise SpecError(f"{sid}: only a recover item injects a fault")
        return
    if len(injected) != 1 or injected[0][0] != len(prior) - 1 or len(prior[-1]) != 1:
        raise SpecError(f"{sid}: a recover item injects one fault, into the single call of its last prior "
                        "step")
    call = injected[0][1]
    if call["inject"] not in INJECTABLE or call["name"] not in INJECT_TOOLS:
        raise SpecError(f"{sid}: inject must be one of {list(INJECTABLE)} on one of {list(INJECT_TOOLS)}")
    if call["inject"] == EMPTY_RESULTS and (call["name"] != "search_arxiv"
                                            or not any(k in call["arguments"] for k in DATE_ARGS)):
        raise SpecError(f"{sid}: EMPTY_RESULTS is injected only into a search with a date filter")


def _check_expected_shape(spec: dict) -> None:
    sid, expected = spec["id"], spec["expected"]
    if not isinstance(expected, dict) or len(expected) != 1:
        raise SpecError(f"{sid}: expected must be {{content}} or {{tool_calls}}")
    if "content" in expected:
        if not isinstance(expected["content"], str) or not expected["content"].strip():
            raise SpecError(f"{sid}: expected content must be a non-empty string")
    elif "tool_calls" in expected:
        calls = expected["tool_calls"]
        if not isinstance(calls, list) or not calls or \
                not all(_is_call(c) and set(c) == {"name", "arguments"} for c in calls):
            raise SpecError(f"{sid}: expected tool_calls must be a non-empty list of {{name, arguments}}")
    else:
        raise SpecError(f"{sid}: expected must be {{content}} or {{tool_calls}}")


# ---- replaying the prior steps ----------------------------------------------------------------------


def _api_call(call_id: str, name: str, arguments: dict) -> dict:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}}


class _ScriptedPrior:
    """Plays the prior steps (call ids c1, c2, ... across steps), then answers END_TEXT, which ends the
    turn."""

    name = MODEL_NAME

    def __init__(self, prior: list[list[dict]]) -> None:
        self.steps, self.asked = prior, 0
        self.next_id = 1

    def next_turn(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        self.asked += 1
        if self.asked > len(self.steps):
            return AssistantTurn(END_TEXT, [], MODEL_NAME)
        calls = []
        for call in self.steps[self.asked - 1]:
            api = _api_call(f"c{self.next_id}", call["name"], call["arguments"])
            self.next_id += 1
            calls.append(ToolCall.from_api(api))
        return AssistantTurn("", calls, MODEL_NAME)


def _injecting(tools: dict[str, ToolFn], position: int | None, code: str | None) -> dict[str, ToolFn]:
    """tools where the position-th executed call (0-based, over the whole turn) runs and is then answered
    with code, in the injectors' shape: ([], code) for search_arxiv, ({}, code) for the fetching tools."""
    count = [0]

    def wrap(name: str, tool: ToolFn) -> ToolFn:
        def run(args: dict) -> ToolOutcome:
            outcome = tool(args)  # first, so a replay miss surfaces as it would
            here = count[0]
            count[0] += 1
            if here == position:
                return ToolOutcome([] if name == "search_arxiv" else {}, code)
            return outcome

        return run

    return {name: wrap(name, tool) for name, tool in tools.items()}


def replay(spec: dict, world: World, scratch: Path) -> tuple[list[dict], list[str]]:
    """The history before the decision and the ids seen in it, from playing spec's prior steps through the
    loop (module docstring). scratch is an empty folder for the item's note store."""
    sid, prior = spec["id"], spec["prior"]
    record = {"id": sid, "split": spec["split"], "slots": spec["slots"]}
    seed_notes(scratch, seed_plan(record, world.pool, world.index, world.today), world.index)
    notes = store_note_tools(scratch, world.index, world.today)
    tools, _ = make_tools(fetcher=world.fetcher, index=world.index, owner=world.owner, today=world.today,
                          scenario="normal", seed=0, task=sid, note_tools=notes)
    flat = [c for step in prior for c in step]
    position = next((i for i, c in enumerate(flat) if "inject" in c), None)
    code = None if position is None else flat[position]["inject"]
    executor = ToolExecutor(_injecting(tools, position, code), world.owner.check_call, today=world.today)
    model = _ScriptedPrior(prior)
    transcript = run_turn(spec["request"], model=model, executor=executor, decide=world.owner.decide,
                          today=world.today, max_steps=len(prior) + 1, max_model_calls_per_step=1)
    for k, step in enumerate(transcript.steps, 1):
        if "harness_error" in step:
            raise SpecError(f"{sid}: prior step {k}: harness error: {step['harness_error']}")
        issues = [i for a in step["attempts"] for i in a.get("issues") or []]
        if issues or len(step["attempts"]) != 1:
            raise SpecError(f"{sid}: prior step {k} was refused by the executor: {issues}")
    if transcript.status != ANSWERED or transcript.final_text != END_TEXT:
        raise SpecError(f"{sid}: the prior steps did not replay cleanly (status {transcript.status})")
    history = transcript.messages[:-1]
    if position is not None:
        last = [m for m in history if m.get("role") == "tool"][position]
        if json.loads(last["content"]).get("error") != code:
            raise SpecError(f"{sid}: the injected {code} did not reach the tool message")
    return history, executor.seen_ids


# ---- building and checking --------------------------------------------------------------------------


def _expected(spec: dict) -> dict:
    calls = []
    for call in spec["expected"].get("tool_calls", []):
        arguments_json = json.dumps(call["arguments"], ensure_ascii=False)
        calls.append({"name": call["name"], "arguments": parse_arguments(arguments_json),
                      "arguments_json": arguments_json})
    return {"content": spec["expected"].get("content", ""), "tool_calls": calls}


def _output(expected: dict, first_id: int) -> dict:
    """The expected turn as the model's API output, call ids continuing the history's."""
    calls = [{"id": f"c{first_id + i}", "type": "function",
              "function": {"name": c["name"], "arguments": c["arguments_json"]}}
             for i, c in enumerate(expected["tool_calls"])]
    output: dict = {"role": "assistant", "content": expected["content"]}
    if calls:
        output["tool_calls"] = calls
    return output


def build_item(spec: dict, world: World, scratch: Path | None = None) -> dict:
    """The item of one spec after checks 1 to 9 of the module docstring (check 10 needs every spec:
    check_items). scratch: an empty folder for the note store; a temporary one when None."""
    check_shape(spec)
    if scratch is None:
        folder = Path(tempfile.mkdtemp(prefix="hw-notes-"))
        try:
            return build_item(spec, world, folder / "store")
        finally:
            shutil.rmtree(folder, ignore_errors=True)
    sid = spec["id"]
    history, seen = replay(spec, world, Path(scratch))
    expected = _expected(spec)
    item = {"id": sid, "messages": history, "expected": expected, "type": spec["type"], "lang": spec["lang"],
            "family": f"handwritten/{spec['type']}/{spec['lang']}", "split": spec["split"],
            "slots": copy.deepcopy(spec["slots"]), "source": {"handwritten": True}}
    n_calls = sum(len(m.get("tool_calls") or []) for m in history)
    output = _output(expected, n_calls + 1)
    problems = [*_type_problems(spec, history, output), *_validity_problems(expected, seen, world.today)]
    if history[-1].get("role") not in LAST_ROLES:
        problems.append(f"the history ends with a {history[-1].get('role')!r} message")
    score = score_item({"id": sid, "status": OK, "output": output}, item)
    if not score["success"]:
        failed = [c for c in ("json_valid", "tool_choice", "arguments", "exact_args", "ids_grounded",
                              "no_false_call", "clarify", "recover") if score.get(c) is False]
        problems.append(f"the expected turn does not score as a success (failed: {failed})")
    problems += _recover_problems(spec, expected)
    problems += _argument_problems(spec, expected, seen)
    problems += split_problems(spec, world)
    if problems:
        raise SpecError(f"{sid}: " + "; ".join(problems))
    return item


def _type_problems(spec: dict, history: list[dict], output: dict) -> list[str]:
    got = item_type(history, output, len(spec["prior"]) + 1)
    if got != spec["type"]:
        return [f"type {spec['type']!r}, but the type rules give {got!r} for this expected turn"]
    calls = spec["expected"].get("tool_calls", [])
    if spec["type"] in (CALL, NEXT_STEP) and len(calls) > 1:
        return [f"a {spec['type']} item expects at most one call"]
    if spec["type"] == PARALLEL:
        keys = {call_key(ToolCall.from_api(_api_call("x", c["name"], c["arguments"]))) for c in calls}
        if len(calls) != 2 or len(keys) != 2:
            return ["a parallel item expects exactly two different calls"]
    if spec["type"] == CLARIFY:
        if [c["name"] for c in calls] != [ASK]:
            return [f"a clarify item expects {ASK} alone"]
        missing = calls[0]["arguments"].get("missing")
        if not isinstance(missing, list) or not missing or not set(missing) <= set(MISSING):
            return [f"missing must name slots of {MISSING}"]
        if set(missing) != set(spec["slots"].get("missing") or []):
            return ["missing must equal slots['missing']"]
    if spec["type"] == NO_CALL and not spec["expected"].get("content", "").strip():
        return ["a no_call item expects an answer"]
    return []


def _validity_problems(expected: dict, seen: list[str], today: str) -> list[str]:
    from ..validation import check_call  # the owner's module, imported where it is used

    problems = []
    for call in expected["tool_calls"]:
        errors = check_call(call["name"], copy.deepcopy(call["arguments"]), seen, today)
        if errors:
            problems.append(f"{call['name']} fails check_call: {errors}")
    return problems


def _recover_problems(spec: dict, expected: dict) -> list[str]:
    if spec["type"] != RECOVER:
        return []
    failed = spec["prior"][-1][0]
    calls = expected["tool_calls"]
    if len(calls) != 1:
        return ["a recover item expects exactly one call"]
    got = call_key(ToolCall.from_api(_api_call("x", calls[0]["name"], calls[0]["arguments"])))
    if failed["inject"] == EMPTY_RESULTS:
        relaxed = {k: v for k, v in failed["arguments"].items() if k not in DATE_ARGS}
        want = call_key(ToolCall.from_api(_api_call("x", "search_arxiv", relaxed)))
        return [] if got == want else [("after EMPTY_RESULTS the expected call is the same search without "
                                        "its date filter")]
    want = call_key(ToolCall.from_api(_api_call("x", failed["name"], failed["arguments"])))
    return [] if got == want else [f"after {failed['inject']} the expected call retries the failed one"]


def _mentioned(value: object, text: str) -> bool:
    return normalize_text(str(value)).casefold() in normalize_text(text).casefold()


def _argument_problems(spec: dict, expected: dict, seen: list[str]) -> list[str]:
    """Check 8 of the module docstring."""
    text, slots, problems = spec["request"], spec["slots"], []
    for call in expected["tool_calls"]:
        name, args = call["name"], call["arguments"]
        if name == ASK:
            continue
        for key, value in args.items():
            default = DEFAULTS.get(name, {}).get(key)
            where = f"{name}.{key}={value!r}"
            if key in ("date_from", "date_to", "since"):
                ok = isinstance(value, str) and value in text
            elif key == "categories":
                ok = isinstance(value, list) and all(_mentioned(v, text) for v in value)
            elif key in ("tags", "tag"):
                ok = all(_mentioned(v, text) for v in (value if isinstance(value, list) else [value]))
            elif key in COUNT_ARGS:
                ok = slots.get("count") == value
            elif key in ("section", "sort_by", "mode"):
                ok = slots.get(key) == value or (key == "mode" and _mentioned(value, text))
            elif key == "arxiv_id":
                ok = value in text or value in seen
            elif key == "query":
                ok = bool(required_words(slots, value))
                where += " has no required words in the slots"
            else:
                ok = key not in CONTENT_FIELDS
            if not ok:
                problems.append(f"{where} is not asked for by the request")
            elif default is not None and value == default and key not in ("section", "sort_by", "mode") \
                    and not _mentioned(value, text) and slots.get("count") != value:
                problems.append(f"{where} is the schema default and the request does not ask for it")
    return problems


def _topic_words(world: World, topic_ids: Iterable[str]) -> list[str]:
    ids = set(topic_ids)
    return sorted(word for word, tid in world.pool.names.items() if tid in ids)


def _names(word: str, text: str) -> bool:
    """Whether text names word (casefolded, whitespace collapsed): a Latin word only as a whole word, so
    "rag" is not found in "storage"; a Chinese name anywhere."""
    text = " ".join(normalize_text(text).split()).casefold()
    pattern = re.escape(word)
    if word[:1].isascii() and word[:1].isalnum():
        pattern = r"(?<![a-z0-9])" + pattern
    if word[-1:].isascii() and word[-1:].isalnum():
        pattern += r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def _slot_topics(slots: dict, world: World, split: str) -> list[str | None]:
    topic = slots.get("topic")
    topics = topic if isinstance(topic, list) else ([topic] if topic is not None else [])
    found = [request_topic({"slots": {"topic": t}}, world.pool) for t in topics]
    if "topic_id" in slots:
        found.append(slots["topic_id"] if slots["topic_id"] in world.pool.topics else None)
    return found


def split_problems(spec: dict, world: World) -> list[str]:
    """Check 9 of the module docstring."""
    split, problems = spec["split"], []
    allowed = set(world.pool.allowed(split))
    for tid in _slot_topics(spec["slots"], world, split):
        if tid not in allowed:
            problems.append(f"a slot topic is not a {split} topic ({tid})")
    others = [t for t in world.pool.topics if t not in allowed]
    named = [w for w in _topic_words(world, others) if _names(w, spec["request"])]
    if named:
        problems.append(f"the request names topics of another split: {named}")
    calls = [c for step in spec["prior"] for c in step] + spec["expected"].get("tool_calls", [])
    blob = json.dumps([spec["request"], spec["slots"], calls, spec["expected"].get("content", "")],
                      ensure_ascii=False)
    papers = world.split_papers(split)
    outside = sorted({a for a in _ARXIV_ID.findall(blob) if a not in papers})
    if outside:
        problems.append(f"ids outside the {split} split's papers: {outside}")
    first, last = world.window
    dates = sorted({d for d in _ISO_DATE.findall(blob) if not first <= d <= last})
    if dates:
        problems.append(f"dates outside the window {first}..{last}: {dates}")
    return problems


def wording_problems(specs: Sequence[dict], world: World) -> list[str]:
    """Check 10 of the module docstring, one message per offending spec."""
    problems = []
    references = world.reference_shingles()
    own = [shingles(mask_text(s["request"])) for s in specs]
    texts = {}
    for i, spec in enumerate(specs):
        worst = max((jaccard(own[i], r) for r in references), default=0.0)
        if worst >= TEMPLATE_LIMIT:
            problems.append(f"{spec['id']}: request too close to a generated request (Jaccard {worst:.2f})")
        key = normalize_text(spec["request"])
        for j in range(i):
            score = jaccard(own[i], own[j])
            if score >= HANDWRITTEN_LIMIT or key == texts.get(j):
                problems.append(f"{spec['id']}: request too close to {specs[j]['id']} (Jaccard {score:.2f})")
        texts[i] = key
    if specs:
        for retired in world.retired:
            scores = retired.reader.similarities([spec["request"] for spec in specs])
            for spec, score in zip(specs, scores, strict=True):
                if score >= retired.limit:
                    problems.append(f"{spec['id']}: request too close to {retired.name} "
                                    f"(Jaccard {score:.2f})")
    return problems


def build_items(specs: Sequence[dict], world: World,
                progress: Callable[[str], None] | None = None) -> tuple[list[dict], list[str]]:
    """(the items of the specs that pass, every problem found). Ids must be unique; the wording check runs
    over the specs whose shape is fine. The items keep the specs' order."""
    problems, items, fine = [], [], []
    ids = [s.get("id") for s in specs]
    for sid in sorted({i for i in ids if ids.count(i) > 1}, key=str):
        problems.append(f"{sid}: id used {ids.count(sid)} times")
    for spec in specs:
        try:
            item = build_item(spec, world)
        except SpecError as exc:
            problems.append(str(exc))
            try:
                check_shape(spec)
                fine.append(spec)
            except SpecError:
                pass
            continue
        fine.append(spec)
        items.append(item)
        if progress is not None:
            progress(spec["id"])
    problems += wording_problems(fine, world)
    return items, problems


def spares(needed: int) -> int:
    """The spare items a stratum gets on top of its needed count."""
    return 1 + needed // 5


def stratum_problems(specs: Sequence[dict], needed: dict) -> list[str]:
    """For each split the specs cover: every stratum "<split>/<type>/<lang>" of needed["strata"] must hold
    needed + spares(needed) specs, and no spec may fall outside the strata. A stratum without a needed count
    (the second family's dev strata, which get no hand-written items) must hold none."""
    counts: dict[str, int] = {}
    for spec in specs:
        key = f"{spec.get('split')}/{spec.get('type')}/{spec.get('lang')}"
        counts[key] = counts.get(key, 0) + 1
    splits = {key.split("/")[0] for key in counts}
    problems = []
    for key, row in needed["strata"].items():
        if key.split("/")[0] in splits and "needed" not in row:
            if counts.get(key, 0):
                problems.append(f"stratum {key}: {counts[key]} specs, none wanted (the stratum takes no "
                                "hand-written items)")
        elif key.split("/")[0] in splits:
            want = row["needed"] + spares(row["needed"])
            if counts.get(key, 0) != want:
                problems.append(f"stratum {key}: {counts.get(key, 0)} specs, {want} wanted "
                                f"({row['needed']} needed + {spares(row['needed'])} spares)")
    unknown = sorted(set(counts) - set(needed["strata"]))
    if unknown:
        problems.append(f"specs outside the strata: {unknown}")
    return problems


# ---- the second family: numbering, the writers' plan and their batches ------------------------------------


def numbering_problems(specs: Sequence[dict], first: int = FIRST_ID_V2) -> list[str]:
    """The second family's ids: every one a test id numbered from first on, so that no id of v1 returns. An
    id that does not match ID_PATTERN is check 1's to name."""
    problems = []
    for spec in specs:
        sid = spec.get("id")
        match = ID_PATTERN.fullmatch(sid) if isinstance(sid, str) else None
        if match is not None and (match.group(1) != "test" or int(match.group(2)) < first):
            problems.append(f"{sid}: a v2 id is hw-test-<nnn> from hw-test-{first:03d} on (v1 used "
                            f"hw-test-001 to hw-test-{first - 1:03d})")
    return problems


def _hw_id(split: str, number: int) -> str:
    return f"hw-{split}-{number:03d}"


def writing_plan(needed: dict, *, folder: str, first: int = FIRST_ID_V2, split: str = "test",
                 batch_size: int = BATCH_SIZE) -> dict:
    """The writers' plan for the hand-written specs of one split. Every stratum of needed["strata"] of that
    split with a needed count n gets n + spares(n) specs (the spares replace items a review drops), in the
    file's order, numbered from hw-<split>-<first> on in that order; a stratum is cut into the fewest batches
    of at most batch_size specs, of near-equal size (the larger first), one stratum per batch. The batches are
    named batch-01, batch-02, ... in order, each written to <folder>/<name>.jsonl. Returns {"split",
    "first_id", "last_id", "batch_size", "totals", "strata": {stratum: {"needed", "spares", "specs",
    "first_id", "last_id", "batches"}}, "batches": [{"batch", "stratum", "count", "first_id", "last_id",
    "out"}]}; ValueError when the ids would run past hw-<split>-999."""
    strata: dict[str, dict] = {}
    batches: list[dict] = []
    number = first
    for key, row in needed["strata"].items():
        if key.split("/")[0] != split or "needed" not in row:
            continue
        n, start = row["needed"], number
        count = n + spares(n)
        parts = -(-count // batch_size)
        base, extra = divmod(count, parts)
        names = []
        for k in range(parts):
            size = base + (1 if k < extra else 0)
            name = f"batch-{len(batches) + 1:02d}"
            batches.append({"batch": name, "stratum": key, "count": size, "first_id": _hw_id(split, number),
                            "last_id": _hw_id(split, number + size - 1), "out": f"{folder}/{name}.jsonl"})
            names.append(name)
            number += size
        strata[key] = {"needed": n, "spares": spares(n), "specs": count, "first_id": _hw_id(split, start),
                       "last_id": _hw_id(split, number - 1), "batches": names}
    if number - 1 > _LAST_NUMBER:
        raise ValueError(f"the plan's ids would run to hw-{split}-{number - 1}, past "
                         f"hw-{split}-{_LAST_NUMBER}")
    totals = {"needed": sum(row["needed"] for row in strata.values()),
              "spares": sum(row["spares"] for row in strata.values()), "specs": number - first,
              "batches": len(batches)}
    return {"split": split, "first_id": _hw_id(split, first), "last_id": _hw_id(split, number - 1),
            "batch_size": batch_size, "totals": totals, "strata": strata, "batches": batches}


def batch_problems(specs: Sequence[dict], batch: dict) -> list[str]:
    """A writer's batch against its entry in the plan (writing_plan): exactly the ids first_id to last_id,
    every spec of the batch's stratum. An id used twice is build_items' to name."""
    first, last = (ID_PATTERN.fullmatch(batch[key]) for key in ("first_id", "last_id"))
    planned = [_hw_id(first.group(1), n) for n in range(int(first.group(2)), int(last.group(2)) + 1)]
    name, problems = batch["batch"], []
    for spec in specs:
        sid = spec.get("id")
        if sid not in planned:
            problems.append(f"{sid}: not an id of {name} ({batch['first_id']} to {batch['last_id']})")
        stratum = f"{spec.get('split')}/{spec.get('type')}/{spec.get('lang')}"
        if stratum != batch["stratum"]:
            problems.append(f"{sid}: stratum {stratum}, but {name} is {batch['stratum']}")
    have = {spec.get("id") for spec in specs if isinstance(spec.get("id"), str)}
    return problems + [f"{sid}: missing from {name}" for sid in planned if sid not in have]


def peer_problems(specs: Sequence[dict], peers: Sequence[tuple[str, dict]]) -> list[str]:
    """A writer's batch against the other batches present, given as (file name, spec) pairs: no id of the
    batch is used there, and no request comes within HANDWRITTEN_LIMIT of one of theirs (check 10 across
    batches; the closest is named, the first on a tie). A peer without a string id is passed over, one without
    a string request for the wording."""
    used: dict[str, str] = {}
    theirs = []
    for name, peer in peers:
        pid = peer.get("id") if isinstance(peer, dict) else None
        if not isinstance(pid, str):
            continue
        used.setdefault(pid, name)
        if isinstance(peer.get("request"), str):
            theirs.append((name, pid, shingles(mask_text(peer["request"]))))
    problems = []
    for spec in specs:
        sid, request = spec.get("id"), spec.get("request")
        if isinstance(sid, str) and sid in used:
            problems.append(f"{sid}: id also used in {used[sid]}")
        if not isinstance(request, str) or not theirs:
            continue
        mine = shingles(mask_text(request))
        score, other, name = max(((jaccard(mine, grams), pid, name) for name, pid, grams in theirs),
                                 key=lambda found: found[0])
        if score >= HANDWRITTEN_LIMIT:
            problems.append(f"{sid}: request too close to {other} in {name} (Jaccard {score:.2f})")
    return problems


def replayed(spec: dict, world: World) -> tuple[list[dict], list[str]]:
    """What the model will see before the decision, for a writer to look at: the history and the ids seen in
    it (replay), after check 1, over a note store in a temporary folder; a SpecError as build_item raises
    it."""
    check_shape(spec)
    folder = Path(tempfile.mkdtemp(prefix="hw-notes-"))
    try:
        return replay(spec, world, folder / "store")
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def retired_v1(root: str | Path = ".", **options) -> tuple[Retired, Retired]:
    """v1's two retired files a second-family request stays apart from (check 10), under root, each read only
    through eval.lock.RetiredSet: v1's hand-written requests, dev and test (RetiredSet.handwritten_specs),
    below HANDWRITTEN_LIMIT, and the user requests of v1's single-step test items (RetiredSet.single_step),
    below TEMPLATE_LIMIT. options (max_opens, now) go to both readers; nothing is read before a check asks."""
    return (Retired("v1's hand-written requests", RetiredSet.handwritten_specs(root, **options),
                    HANDWRITTEN_LIMIT),
            Retired("v1's single-step test items", RetiredSet.single_step(root, **options), TEMPLATE_LIMIT))


# ---- the recorded world -----------------------------------------------------------------------------


def request_texts(requests: str | Path | Sequence[str | Path]) -> list[str]:
    """The texts of the generated requests of a request file, or of several in their order (v1 compares with
    one, the second family with every one): what check 10 compares with."""
    from ..data.synth import load_requests

    paths = [requests] if isinstance(requests, str | Path) else list(requests)
    return [r["text"] for path in paths for r in load_requests(path)]


def load_world(*, corpus_manifest: str | Path, cassettes: str | Path,
               requests: str | Path | Sequence[str | Path], requests_manifest: str | Path,
               retired: Sequence[Retired] = ()) -> World:
    """The world the generator runs in: the corpus index over the recorded cassettes (the owner's parse_atom),
    the replay fetcher, the owner's check_call, decide, parse_atom and extract_section, the seed pool
    restricted per split by the requests manifest, its window, the generated request texts of the request file
    or files (request_texts), and the retired files check 10 compares with too (none for v1)."""
    from .. import arxiv, policy, validation
    from ..data.synth import load_seed_pool
    from ..net.fetch import Fetcher
    from ..tools.corpus_search import corpus_ids, load_papers

    owner = OwnerFunctions(validation.check_call, policy.decide, arxiv.parse_atom, arxiv.extract_section)
    fetcher = Fetcher("replay", str(cassettes))
    index = CorpusIndex(load_papers(fetcher, corpus_ids(corpus_manifest), owner.parse_atom))
    pool = load_seed_pool(corpus_manifest, requests_manifest, index.paper)
    window = json.loads(Path(requests_manifest).read_text(encoding="utf-8"))["window"]
    texts = request_texts(requests)
    return World(index=index, fetcher=fetcher, owner=owner, pool=pool, window=(window[0], window[1]),
                 references=texts, retired=tuple(retired))

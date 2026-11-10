"""End-to-end tasks (docs/plan.md, 评测: 端到端): a researcher's whole request, run by the agent loop over the
frozen recordings and scored with deterministic checks.

A task is a request record (data.synth.load_requests: id, intent, scenario, lang, split, family, text, slots)
plus "expect", what a correct run must show:

    status                 "answered" or "clarification": the turn's final status
    must_write_note        true when a successful write_note call is required
    note_about             with must_write_note: {"arxiv_id": id} (the note must be on that paper) or
                           {"topic_id": topic} (on a paper of that corpus topic, or one whose title or
                           abstract contains one of the topic's phrases: the replay search covers every
                           topic, so the most relevant hit may lie outside the topic's own list); else null
    must_cite_ids_within   "tool_outputs": every arXiv id in the final text appeared in a tool output of
                           this turn
    expected_tools_any     tools of which at least one must be called; [] means no tool call at all

The intents (E2E_INTENTS) are what the owner does weekly: find recent papers on a topic with a date and a
count, read a named section, write a scaffold note, compare two named papers, look up an id, search or list
his notes, a request that needs clarification, and small talk. The scenario follows from the intent
(needs_clarification -> clarify, chit_chat -> no_call, everything else normal), so check_trajectory applies
its clarify and no_call rules. Tasks of the dev split use only the dev topics and paper ids of
data/synth/requests-manifest.json, test tasks only the test ones.

Running (run_tasks). Each task goes through data.synth.run_request in replay: no rewriting, no fault
injection unless a FaultSetting is given (a normal scenario never meets one; below), the owner's functions
from the tree, the request's note store
in <out>/notes/<id>/ emptied and seeded with the five scaffold notes generation gives it (seed_plan, from the
topics the task's split may use), today the corpus recording date. The transcript goes to
<out>/transcripts/<id>.json. A run stops after a task that ended in budget_stop. Test tasks run only as the
OpenedTestSet of a LockedSet.open (research_agent.eval.lock, who opens what).

Scoring (score_task), all deterministic:

    trajectory  trajectories.check_trajectory's five checks on the transcript and the task: calls parse and
                validate, execution without error codes, no cited id outside the tool outputs, the date and
                category constraints of the slots, the note fields; plus its clarify and no_call rules
    status      the transcript's status equals expect.status
    tools       some call of expected_tools_any was made ([]: no call was made)
    note        with must_write_note, a successful write_note on a paper note_about allows
    cited_ids   every arXiv id in the final text (metrics.arxiv_ids) appeared in a tool output this turn
    evidence    the plan's "scaffold 证据句是正文原样子串", on what the model wrote: every write_note call in
                scaffold mode (the default) quotes retrieved text of its own paper. Each fragment of the
                call's evidence argument (evidence_fragments) must occur, compared by evidence_tokens, in a
                string of a tool output of an earlier step of this turn that belongs to the note's paper (an
                object whose arxiv_id is the call's: a search hit, get_paper, fetch_fulltext); an argument
                with no fragment fails. Draft calls are not checked (their evidence is a written part, not a
                quotation). The note the tool stored picks its own sentences from the abstract, so its
                evidence is reported under "stored" for information only.

Evidence fragments (evidence_fragments). When the argument holds quoted spans ("...", curly quotes, 「」 or
『』) of at least MIN_QUOTE_CHARS token characters, those spans are the quotations and the words around them
(labels, list markers, section notes) are ignored; otherwise every sentence of the argument
(notes.split_sentences, line by line, list markers dropped) is a quotation. Each quotation is cut at elisions
("...", "…", or a bracketed insertion "[...]") into fragments; fragments without a word or a number are
dropped. A fragment counts as retrieved when its evidence_tokens occur as a contiguous run of a source's:
tokens are words, numbers with their decimal points and signs, single CJK characters, and the symbols a
claim can turn on (< > = % ± ¬ ∈ ∉ arrows ≤ ≥ ≠ ≈, each in one form whatever its ASCII, LaTeX or Unicode
spelling), casefolded after NFKC, with other LaTeX commands removed or read as the letter they print, and
thousands separators and line-break hyphens dropped. So typography (quote marks, apostrophe forms, dashes,
x or ×, $...$, \\%, spacing, a full stop added at the end of a cut sentence) is no difference, while a
changed, added or dropped word, digit, decimal point, sign or relation is, and a run must start and end on
token boundaries. Known residuals: an elision can hide a dropped word (a negation included), since each
fragment is checked on its own; notation changes such as 1e-5 against 1×10^-5, superscript exponents, and a
sign standing apart from its number ("+ 3") are not reconciled. Measured on the teacher's 824 scaffold calls
of the assembled training set (2026-10-15): 568 pass; the others paraphrase, translate, join or annotate
inside the quotation marks.

Fault injection (M4; docs/plan.md, 故障注入). Given a FaultSetting, every task gets its own pair of
injectors (net.faults: a FaultInjector over the fetcher, SearchFaults over the search), keyed by the run's
seed and the task id, with the setting's rate, faults and persistence and no fault budget (data.synth.
FaultSetting), so two systems running a task with the same seed meet the same fault on their k-th request
for the same URL or search. Every fault the task meets goes into its record as "faults", in call order, each
{"tool", "key" (the canonical URL, or the search key "search:" plus the canonical arguments), "kind" (the
fault), "ordinal" (the request's ordinal among the task's requests for that key), "step" (the transcript
step whose call met it), "call" (the call's position among the task's calls of that tool in the accepted
history, from 0: the order tool_calls counts), "call_id"}, and into the transcript, under "faults" in that
step. A fetch_fulltext call whose full text met html_404 goes on to the abstract, so one call may meet two
faults. The calls are matched to the transcript by their order: the n-th run of a tool is its n-th call in
the steps' executed list that was not answered SKIPPED (the loop answers SKIPPED without running the tool).
A task that ended in a harness error keeps the faults met until then, with step, call and call_id null. A
record made without a setting has no "faults" key at all, so a run without injection writes the bytes it
wrote before M4; a task that met none under a setting has "faults": []. The time of every arXiv tool call
(the delay fault's sleep included) goes to run_tasks' on_tools hook, the timing sidecar's, never a record.

A task succeeds when every check passes. Each record also carries the steps, the tool calls by name, the tool
errors by code, the tokens, the cost and the call counts. No record and no summary carries a time: the wall
times go to the timing sidecar (research_agent.eval.timing, through run_tasks' on_task hook), so two runs over
the same tasks with the same answers make the same records.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from ..agent.context import TokenCounter
from ..agent.loop import Model, save_transcript
from ..agent.types import ANSWERED, BUDGET_STOP, CLARIFICATION, HARNESS_ERROR, Transcript
from ..data.synth import (
    FaultSetting,
    FaultTrace,
    OwnerFunctions,
    SeedPool,
    check_requests,
    load_requests,
    load_seed_pool,
    run_request,
)
from ..normalize import normalize_arxiv_id, normalize_text
from ..notes import import_entries, locate_evidence, split_sentences
from ..tools.schemas import DEFAULTS, MODEL_TOOLS, SKIPPED
from ..trajectories import check_trajectory
from .lock import refuse_test_material
from .metrics import arxiv_ids

E2E_INTENTS = ("find_recent_papers", "read_section", "write_note", "compare_papers", "lookup_paper",
               "search_notes", "list_notes", "needs_clarification", "chit_chat")
SCENARIO_OF = {"needs_clarification": "clarify", "chit_chat": "no_call"}  # every other intent: normal
E2E_SPLITS = ("dev", "test")
EXPECT_KEYS = ("status", "must_write_note", "note_about", "must_cite_ids_within", "expected_tools_any")
FINAL_STATUSES = (ANSWERED, CLARIFICATION)
CITE_SCOPE = "tool_outputs"
TOOL_NAMES = tuple(t["function"]["name"] for t in MODEL_TOOLS)
CHECKS = ("trajectory", "status", "tools", "note", "cited_ids", "evidence")
WRITE_NOTE = "write_note"
SCAFFOLD = "scaffold"
REPORT_FILE = "report.json"
ARXIV_TOOLS = ("search_arxiv", "get_paper", "fetch_fulltext")  # the tools behind the fault injectors
MIN_QUOTE_CHARS = 20  # letters or digits a quoted span needs to count as a quotation rather than a term


# ---- tasks ------------------------------------------------------------------------------------------------


def load_tasks(path: str | Path) -> list[dict]:
    """The tasks of a JSONL file in file order; ValueError naming the first task that is not a request
    record (load_requests' checks), has an intent outside E2E_INTENTS, a scenario other than its intent's, a
    split outside dev and test, or an expect that check_expect rejects."""
    return _checked_tasks(load_requests(path), Path(path).name)


def check_tasks(tasks: Iterable[object], source: str) -> list[dict]:
    """load_tasks' checks on tasks already read, such as the end-to-end test tasks a LockedSet.open returned,
    which stay in memory (scripts/run_test.py): checked copies in the order given (data.synth.check_requests
    names a record that is no request by its position, the other checks name the task by its id, after
    source)."""
    return _checked_tasks(check_requests(tasks, source), source)


def _checked_tasks(tasks: list[dict], source: str) -> list[dict]:
    for task in tasks:
        where = f"{source}: task {task['id']}"
        if task["intent"] not in E2E_INTENTS:
            raise ValueError(f"{where}: intent {task['intent']!r} is not one of {list(E2E_INTENTS)}")
        scenario = SCENARIO_OF.get(task["intent"], "normal")
        if task["scenario"] != scenario:
            raise ValueError(f"{where}: a {task['intent']} task has scenario {scenario}, "
                             f"not {task['scenario']}")
        if task["split"] not in E2E_SPLITS:
            raise ValueError(f"{where}: split must be dev or test, not {task['split']!r}")
        problems = check_expect(task.get("expect"))
        if problems:
            raise ValueError(f"{where}: {'; '.join(problems)}")
    return tasks


def check_expect(expect: object) -> list[str]:
    """What is wrong with a task's expect (the module docstring); [] when it is well formed."""
    if not isinstance(expect, dict):
        return ["expect must be an object"]
    problems = [f"expect lacks {key}" for key in EXPECT_KEYS if key not in expect]
    problems += [f"expect has unknown key {key}" for key in expect if key not in EXPECT_KEYS]
    if problems:
        return problems
    if expect["status"] not in FINAL_STATUSES:
        problems.append(f"expect.status must be one of {list(FINAL_STATUSES)}")
    if expect["must_cite_ids_within"] != CITE_SCOPE:
        problems.append(f"expect.must_cite_ids_within must be {CITE_SCOPE!r}")
    tools = expect["expected_tools_any"]
    if not isinstance(tools, list) or not all(t in TOOL_NAMES for t in tools):
        problems.append(f"expect.expected_tools_any must be a list of tool names from {list(TOOL_NAMES)}")
    must, about = expect["must_write_note"], expect["note_about"]
    if not isinstance(must, bool):
        problems.append("expect.must_write_note must be true or false")
    elif must and not (isinstance(about, dict) and len(about) == 1
                       and isinstance(about.get("arxiv_id", about.get("topic_id")), str)):
        problems.append("expect.note_about must be {\"arxiv_id\": id} or {\"topic_id\": topic} when a "
                        "note is required")
    elif not must and about is not None:
        problems.append("expect.note_about must be null when no note is required")
    if expect["status"] == CLARIFICATION and "ask_clarification" not in (tools or []):
        problems.append("a clarification task must expect ask_clarification")
    return problems


# ---- the world --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Topic:
    papers: frozenset[str]
    phrases: tuple[str, ...]


def corpus_topics(corpus_manifest: str | Path) -> dict[str, Topic]:
    """topic id -> its papers and phrases (the id with hyphens as spaces, and the manifest's phrases)."""
    data = json.loads(Path(corpus_manifest).read_text(encoding="utf-8"))
    return {t["id"]: Topic(frozenset(t.get("papers", [])),
                           tuple(dict.fromkeys([t["id"].replace("-", " "), *t.get("phrases", [])])))
            for t in data.get("topics") or []}


@dataclass
class World:
    """What every task runs against: the owner's functions, the replay fetcher, the corpus index, the seed
    pool for the note stores and the corpus topics for the note check."""

    owner: OwnerFunctions
    fetcher: object
    index: object  # tools.corpus_search.CorpusIndex
    seed_pool: SeedPool | None
    topics: dict[str, Topic] = field(default_factory=dict)


def build_world(*, cassettes: str | Path, corpus_manifest: str | Path,
                requests_manifest: str | Path | None) -> World:
    """The replay world of the recorded corpus with the owner's functions from the tree."""
    from ..arxiv import extract_section, parse_atom
    from ..net.fetch import Fetcher
    from ..policy import decide
    from ..tools.corpus_search import CorpusIndex, corpus_ids, load_papers
    from ..validation import check_call

    owner = OwnerFunctions(check_call, decide, parse_atom, extract_section)
    fetcher = Fetcher("replay", cassettes)
    index = CorpusIndex(load_papers(fetcher, corpus_ids(corpus_manifest), parse_atom))
    pool = load_seed_pool(corpus_manifest, requests_manifest, index.paper)
    return World(owner, fetcher, index, pool, corpus_topics(corpus_manifest))


# ---- evidence ---------------------------------------------------------------------------------------------

# Quoted spans: straight double quotes, curly double quotes (U+201C/U+201D) and the CJK corner brackets
# (U+300C/U+300D, U+300E/U+300F); an elision is three full stops, the ellipsis U+2026 or a bracketed
# insertion; a list marker is -, *, the bullet U+2022, or a number with . or ).
_QUOTED = re.compile(r'"([^"]+)"|“([^”]+)”|「([^」]+)」|『([^』]+)』')
_ELISION = re.compile(r"\.\.\.|…|\[[^\]]*\]")
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_LATEX_COMMAND = re.compile(r"\\([A-Za-z]+)")
_GREEK = ("alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa", "lambda",
          "mu", "nu", "xi", "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega")
_GREEK_VARIANTS = {"varepsilon": "epsilon", "vartheta": "theta", "varphi": "phi", "varpi": "pi",
                   "varrho": "rho", "varsigma": "sigma"}


def _greek(name: str) -> str:
    return unicodedata.lookup(f"GREEK SMALL LETTER {name.upper()}".replace("LAMBDA", "LAMDA"))


# The symbols a claim can turn on (relations, membership, arrows, percent, plus-minus, negation): each is a
# token of its own, in the form written here whatever its source spelling (ASCII digraph, LaTeX command,
# Unicode character)
_SYMBOLS = "<>=%±¬∈∉→←↑↓≤≥≠≈"
_SYMBOL_COMMANDS = {"le": "≤", "leq": "≤", "ge": "≥", "geq": "≥", "ne": "≠", "neq": "≠", "approx": "≈",
                    "pm": "±", "neg": "¬", "lnot": "¬", "in": "∈", "notin": "∉", "to": "→", "rightarrow": "→",
                    "leftarrow": "←", "uparrow": "↑", "downarrow": "↓"}
# "<-" is left alone: in "x<-1" it is a comparison with a negative number, not an arrow
_DIGRAPHS = (("<=", "≤"), (">=", "≥"), ("!=", "≠"), ("->", "→"))
# LaTeX commands that print a letter or one of the symbols (a capital Greek name gives the small letter:
# tokens are casefolded); every other command prints nothing a quotation would keep
_LATEX_TEXT = {"times": "x", "ell": "l", **{name: _greek(name) for name in _GREEK},
               **{name.capitalize(): _greek(name) for name in _GREEK},
               **{name: _greek(base) for name, base in _GREEK_VARIANTS.items()}, **_SYMBOL_COMMANDS}
_CJK = f"{chr(0x3400)}-{chr(0x4DBF)}{chr(0x4E00)}-{chr(0x9FFF)}"
# a token: one CJK character; one of the symbols; a number with its decimal points, and with its sign when
# the sign follows a space, an opening bracket, $, ^ or one of the symbols (so "-3" in "a -3% error", in
# "10^-3" and in "r=-3" keeps it, the hyphen of "Qwen3-14B" does not); or a run of other letters
_TOKEN = re.compile(rf"[{_CJK}]|[{_SYMBOLS}]|(?<![^\s(\[{{$^{_SYMBOLS}])[-+]?\d+(?:\.\d+)*|\d+(?:\.\d+)*"
                    rf"|[^\W\d_{_CJK}]+")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
# digit-group separators other than the comma (no-break, thin and narrow no-break spaces, apostrophes),
# removed before NFKC turns the no-break ones into plain spaces
_GROUP_SPACES = re.compile(rf"(?<=\d)[{chr(0xA0)}{chr(0x2009)}{chr(0x202F)}'{chr(0x2019)}](?=\d{{3}}(?!\d))")
_EN_DASH_MINUS = re.compile(rf"(?:^|(?<=[\s(\[{{$^])){chr(0x2013)}(?=\d)")  # an en dash before a digit
_LINE_BREAK_HYPHEN = re.compile(r"(?<=[A-Za-z])-\r?\n(?=[a-z])")
_MINUS, _TIMES = chr(0x2212), chr(0xD7)
_SEPARATOR = chr(0x1F)  # joins tokens, so that a run of tokens is found only on token boundaries


def evidence_tokens(text: str) -> list[str]:
    """text as the words, numbers and symbols a quotation must reproduce (module docstring): casefolded
    after NFKC; the minus sign and an en dash before a digit read as "-", the multiplication sign as x;
    thousands separators and line-break hyphens dropped; LaTeX commands read as the letter or symbol they
    print (\\times as x, Greek names in any case or var form as the small letter, \\leq as ≤ and so on) or
    removed; <=, >= and != read as ≤, ≥ and ≠."""
    text = unicodedata.normalize("NFKC", _GROUP_SPACES.sub("", text))
    text = _LINE_BREAK_HYPHEN.sub("", text.replace(_MINUS, "-").replace(_TIMES, " x "))
    text = _LATEX_COMMAND.sub(lambda m: f" {_LATEX_TEXT.get(m.group(1), '')} ", _EN_DASH_MINUS.sub("-", text))
    for digraph, symbol in _DIGRAPHS:
        text = text.replace(digraph, symbol)
    return _TOKEN.findall(_THOUSANDS.sub("", text).casefold())


def _joined(tokens: list[str]) -> str:
    return _SEPARATOR + _SEPARATOR.join(tokens) + _SEPARATOR


def evidence_fragments(evidence: str) -> list[str]:
    """The fragments of a write_note evidence argument that must be retrieved text (module docstring):
    the quoted spans of at least MIN_QUOTE_CHARS token characters, else the sentences, cut at elisions."""
    quoted = [next(group for group in match.groups() if group is not None)
              for match in _QUOTED.finditer(evidence)]
    quotations = [span for span in quoted if sum(map(len, evidence_tokens(span))) >= MIN_QUOTE_CHARS]
    if not quotations:
        for line in evidence.splitlines():
            line = _LIST_MARKER.sub("", line)
            quotations += [line[begin:end] for begin, end in split_sentences(line)]
    return [" ".join(fragment.split()) for quotation in quotations for fragment in _ELISION.split(quotation)
            if evidence_tokens(fragment)]


def unsupported_fragments(evidence: object, sources: list[str]) -> list[str]:
    """The fragments of evidence whose tokens occur as a contiguous run of no source's tokens, in order;
    ["(nothing quoted)"] when evidence is not a string or holds no fragment."""
    fragments = evidence_fragments(evidence) if isinstance(evidence, str) else []
    if not fragments:
        return ["(nothing quoted)"]
    joined = [_joined(evidence_tokens(source)) for source in sources]
    return [f for f in fragments if not any(_joined(evidence_tokens(f)) in text for text in joined)]


# ---- scoring ----------------------------------------------------------------------------------------------


@dataclass
class _Call:
    name: str
    args: dict | None
    body: object = None
    answered: bool = False
    step: int = 0  # the number of the assistant message that made the call, from 1
    id: str = ""


def _calls(messages: list[dict]) -> list[_Call]:
    """Every call of the accepted history with its tool message's parsed body."""
    calls: list[_Call] = []
    by_id: dict[str, _Call] = {}
    step = 0
    for message in messages:
        if message.get("role") == "assistant":
            step += 1
            for raw in message.get("tool_calls") or []:
                fn = raw.get("function") or {}
                arguments = fn.get("arguments")
                try:
                    args = json.loads(arguments) if isinstance(arguments, str) else arguments
                except json.JSONDecodeError:
                    args = None
                call = _Call(fn.get("name") or "", args if isinstance(args, dict) else None, step=step,
                             id=str(raw.get("id") or ""))
                calls.append(call)
                by_id[raw.get("id")] = call
        elif message.get("role") == "tool" and message.get("tool_call_id") in by_id:
            call = by_id[message["tool_call_id"]]
            call.answered = True
            try:
                call.body = json.loads(message.get("content") or "")
            except json.JSONDecodeError:
                call.body = message.get("content")
    return calls


def _strings(node: object) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, list):
        for value in node:
            yield from _strings(value)


def _ids(node: object) -> Iterator[str]:
    """Every non-empty string under an "arxiv_id" key, at any depth (what the executor counts as seen)."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "arxiv_id" and isinstance(value, str) and value:
                yield value
            else:
                yield from _ids(value)
    elif isinstance(node, list):
        for value in node:
            yield from _ids(value)


def _error(body: object) -> str | None:
    return body.get("error") if isinstance(body, dict) else None


def _written_notes(calls: list[_Call]) -> list[dict]:
    """The write_note answers {"note_id", "arxiv_id", "path"} of this turn's successful write_note calls."""
    return [c.body for c in calls if c.name == WRITE_NOTE and isinstance(c.body, dict)
            and _error(c.body) is None and isinstance(c.body.get("path"), str)]


def _about(arxiv_id: str, about: dict, papers: Callable[[str], dict | None],
           topics: dict[str, Topic]) -> bool:
    if "arxiv_id" in about:
        return arxiv_id == about["arxiv_id"]
    topic = topics.get(about["topic_id"])
    if topic is None:
        return False
    if arxiv_id in topic.papers:
        return True
    paper = papers(arxiv_id) or {}
    text = normalize_text(f"{paper.get('title', '')} {paper.get('abstract', '')}").casefold()
    return any(normalize_text(phrase).casefold() in text for phrase in topic.phrases)


def score_task(task: dict, transcript: dict, *, notes_dir: Path, papers: Callable[[str], dict | None],
               topics: dict[str, Topic]) -> dict:
    """{"success": bool, "checks": {check: {"ok": bool, ...details}}} for one task's transcript (the dict
    Transcript.to_json() reads back); the module docstring defines the checks."""
    expect = task["expect"]
    messages = transcript.get("messages") or []
    calls = _calls(messages)
    called = [c.name for c in calls]
    seen = sorted({i for c in calls for i in _ids(c.body)})
    checks: dict[str, dict] = {}

    problems = check_trajectory(transcript, task)
    checks["trajectory"] = {"ok": not problems, "problems": problems}
    checks["status"] = {"ok": transcript.get("status") == expect["status"], "got": transcript.get("status"),
                        "want": expect["status"]}
    wanted = expect["expected_tools_any"]
    tools_ok = not called if not wanted else any(name in wanted for name in called)
    checks["tools"] = {"ok": tools_ok, "called": called, "want_any": wanted}

    written = _written_notes(calls)
    note_ok = True
    if expect["must_write_note"]:
        note_ok = any(_about(n.get("arxiv_id", ""), expect["note_about"], papers, topics) for n in written)
    checks["note"] = {"ok": note_ok, "written": [n.get("arxiv_id") for n in written],
                      "about": expect["note_about"]}

    cited = sorted(arxiv_ids(transcript.get("final_text") or ""))
    unseen = [i for i in cited if i not in seen]
    checks["cited_ids"] = {"ok": not unseen, "cited": cited, "unseen": unseen}

    checks["evidence"] = _evidence(calls)
    checks["evidence"]["stored"] = _stored_evidence(calls, written, Path(notes_dir), papers, seen)
    return {"success": all(check["ok"] for check in checks.values()), "checks": checks}


def _paper_strings(node: object, arxiv_id: str) -> Iterator[str]:
    """The strings of every object (at any depth) whose arxiv_id is arxiv_id (version and prefix aside):
    what a tool output says about that paper."""
    if isinstance(node, dict):
        own = node.get("arxiv_id")
        if isinstance(own, str) and normalize_arxiv_id(own) == arxiv_id:
            yield from _strings(node)
        else:
            for value in node.values():
                yield from _paper_strings(value, arxiv_id)
    elif isinstance(node, list):
        for value in node:
            yield from _paper_strings(value, arxiv_id)


def evidence_check(messages: list[dict]) -> dict:
    """The evidence check of score_task on a message history alone: {"ok", "calls", "fragments",
    "not_retrieved"} (data.evidence_repair verifies its output against it)."""
    return _evidence(_calls(messages))


def _evidence(calls: list[_Call]) -> dict:
    """The evidence check (module docstring): every scaffold write_note call's evidence argument against
    the strings about its paper in the tool outputs of earlier steps."""
    checked, fragments, bad = 0, 0, []
    for call in calls:
        if call.name != WRITE_NOTE or call.args is None:
            continue
        if call.args.get("mode", DEFAULTS[WRITE_NOTE]["mode"]) != SCAFFOLD:
            continue
        raw_id = call.args.get("arxiv_id")
        arxiv_id = normalize_arxiv_id(raw_id) if isinstance(raw_id, str) else None
        sources = [] if arxiv_id is None else [s for earlier in calls if earlier.step < call.step
                                               for s in _paper_strings(earlier.body, arxiv_id)]
        evidence = call.args.get("evidence")
        checked += 1
        fragments += len(evidence_fragments(evidence)) if isinstance(evidence, str) else 0
        bad += [f"{call.id}: {fragment[:120]}" for fragment in unsupported_fragments(evidence, sources)]
    return {"ok": not bad, "calls": checked, "fragments": fragments, "not_retrieved": bad}


def _stored_evidence(calls: list[_Call], written: list[dict], notes_dir: Path,
                     papers: Callable[[str], dict | None], seen: list[str]) -> dict:
    """For information: the evidence sentences of the notes this turn wrote (read back from the store),
    each looked up verbatim (notes.locate_evidence) in this turn's tool outputs and the recorded abstracts of
    the papers seen; the note tool picks them from the abstract itself."""
    sources = [s for c in calls for s in _strings(c.body)]
    sources += [p["abstract"] for p in (papers(i) for i in seen) if p and isinstance(p.get("abstract"), str)]
    checked, missing = 0, []
    for note in written:
        path = notes_dir / note["path"]
        if not path.is_file():
            missing.append(f"{note['path']}: the note file is missing")
            continue
        entries = import_entries(path.read_text(encoding="utf-8"))
        for sentence in entries[0]["evidence_sentences"] if entries else []:
            checked += 1
            if not any(locate_evidence(source, sentence) >= 0 for source in sources):
                missing.append(f"{note['path']}: {sentence[:120]}")
    return {"sentences": checked, "not_in_source": missing}


# ---- running ----------------------------------------------------------------------------------------------


def run_tasks(tasks: list[dict], *, model: Model, world: World, today: str, out: str | Path,
              count_tokens: TokenCounter | None = None, seed: int = 42,
              log: Callable[[str], None] = print,
              on_task: Callable[[dict, Transcript | None], None] | None = None,
              faults: FaultSetting | None = None, clock: Callable[[], float] | None = None,
              on_tools: Callable[[dict, list[dict]], None] | None = None) -> list[dict]:
    """Run and score the tasks in order (module docstring); one record per task run. Transcripts go to
    out/transcripts/<id>.json and note stores to out/notes/<id>/. A run_request exception becomes a
    harness_error record and the run goes on; a budget_stop ends the run after that task. on_task, when
    given, is called with every record and its transcript (None when run_request raised) as soon as the
    record is made: the timing sidecar's hook (research_agent.eval.timing), which keeps the wall times that
    the records do not carry. faults, when given, injects faults into every task (module docstring, Fault
    injection); on_tools then hears, after on_task, every record with its arXiv tool calls, each {"tool",
    "step", "call", "wall_ms", "faults": the kinds it met}, timed on clock (time.perf_counter unless given).
    Test tasks are refused (RefusedTestSet) before any runs, unless they are the
    OpenedTestSet of a LockedSet.open (research_agent.eval.lock, who opens what)."""
    refuse_test_material(tasks, "run_tasks' input")
    out = Path(out)
    records = []
    for task in tasks:
        notes_dir = out / "notes" / task["id"]
        trace = None if faults is None else FaultTrace() if clock is None else FaultTrace(clock=clock)
        try:
            outcome = run_request(task, model=model, owner=world.owner, fetcher=world.fetcher,
                                  index=world.index, today=today, count_tokens=count_tokens, seed=seed,
                                  notes_dir=notes_dir, seed_pool=world.seed_pool, faults=faults,
                                  fault_trace=trace)
        except Exception as exc:  # noqa: BLE001  deliberate: the failure is recorded, the run goes on
            record = error_record(task, f"{type(exc).__name__}: {exc}")
            if trace is not None:
                record["faults"] = fault_events(linked_calls(None, trace.calls))
            records.append(record)
            if on_task is not None:
                on_task(record, None)
            if trace is not None and on_tools is not None:
                on_tools(record, tool_times(linked_calls(None, trace.calls)))
            log(f"{task['id']}: {HARNESS_ERROR} {record['error']}")
            continue
        transcript = outcome.transcript
        linked = None if trace is None else linked_calls(transcript, trace.calls)
        if linked is not None:
            mark_faults(transcript, fault_events(linked))
        name = f"transcripts/{task['id']}.json"
        save_transcript(transcript, out / name)
        scored = score_task(task, json.loads(transcript.to_json()), notes_dir=notes_dir,
                            papers=world.index.paper, topics=world.topics)
        records.append(task_record(task, transcript, scored, name))
        if linked is not None:
            records[-1]["faults"] = fault_events(linked)
        if on_task is not None:
            on_task(records[-1], transcript)
        if linked is not None and on_tools is not None:
            on_tools(records[-1], tool_times(linked))
        failed = [k for k, v in scored["checks"].items() if not v["ok"]]
        log(f"{task['id']} ({task['intent']}/{task['lang']}): {transcript.status}, "
            f"{'ok' if scored['success'] else 'FAIL ' + ','.join(failed)}, steps {len(transcript.steps)}")
        if transcript.status == BUDGET_STOP:
            log(f"{task['id']}: the spending cap was reached; the run stops")
            break
    return records


def linked_calls(transcript: Transcript | None, traced: list[dict]) -> list[dict]:
    """The traced arXiv tool calls (data.synth.FaultTrace.calls) with the transcript step, the position among
    the task's calls of that tool and the call id of each (module docstring, Fault injection); all three None
    for a call the transcript does not show as run (no transcript, or a tool that raised)."""
    runs: dict[str, list[tuple[int, int, str]]] = {name: [] for name in ARXIV_TOOLS}
    positions: Counter[str] = Counter()
    for step in [] if transcript is None else transcript.steps:
        for executed in step.get("executed") or []:
            name = executed.get("name")
            if name not in runs:
                continue
            if executed.get("error") != SKIPPED:
                runs[name].append((step["index"], positions[name], executed.get("call_id")))
            positions[name] += 1
    used: Counter[str] = Counter()
    linked = []
    for traced_call in traced:
        tool = traced_call["tool"]
        mine = runs.get(tool, [])
        step, position, call_id = mine[used[tool]] if used[tool] < len(mine) else (None, None, None)
        used[tool] += 1
        linked.append({**traced_call, "step": step, "call": position, "call_id": call_id})
    return linked


def fault_events(linked: list[dict]) -> list[dict]:
    """The faults of the linked calls, in call order: the record's "faults" (module docstring)."""
    return [{"tool": c["tool"], "key": f["key"], "kind": f["kind"], "ordinal": f["ordinal"],
             "step": c["step"], "call": c["call"], "call_id": c["call_id"]}
            for c in linked for f in c["faults"]]


def tool_times(linked: list[dict]) -> list[dict]:
    """What on_tools hears of the linked calls: {"tool", "step", "call", "wall_ms", "faults": kinds}."""
    return [{"tool": c["tool"], "step": c["step"], "call": c["call"], "wall_ms": c["wall_ms"],
             "faults": [f["kind"] for f in c["faults"]]} for c in linked]


def mark_faults(transcript: Transcript, events: list[dict]) -> None:
    """Every fault event into its transcript step, under "faults"; a step that met none gets no key."""
    for step in transcript.steps:
        mine = [e for e in events if e["step"] == step["index"]]
        if mine:
            step["faults"] = mine


def task_record(task: dict, transcript: Transcript, scored: dict, name: str) -> dict:
    calls = _calls(transcript.messages)
    errors = Counter(code for c in calls if (code := _error(c.body)) is not None)
    return {"id": task["id"], "intent": task["intent"], "lang": task["lang"], "split": task["split"],
            "status": transcript.status, "success": scored["success"], "checks": scored["checks"],
            "steps": len(transcript.steps), "tool_calls": dict(Counter(c.name for c in calls)),
            "tool_errors": dict(sorted(errors.items())), "tokens": dict(transcript.usage),
            "cost_rmb": transcript.cost_rmb, "paid_calls": transcript.paid_calls,
            "cache_hits": transcript.cache_hits, "local_calls": transcript.local_calls, "transcript": name,
            "error": None}


def error_record(task: dict, error: str) -> dict:
    return {"id": task["id"], "intent": task["intent"], "lang": task["lang"], "split": task["split"],
            "status": HARNESS_ERROR, "success": False, "checks": {}, "steps": 0, "tool_calls": {},
            "tool_errors": {}, "tokens": {}, "cost_rmb": 0.0, "paid_calls": 0, "cache_hits": 0,
            "local_calls": 0, "transcript": None, "error": error}


def _rate(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(records: list[dict]) -> dict:
    """Success overall, per check (over the records that were scored), per intent and per language, the
    status counts and the totals of steps, tool errors, tokens and cost (the wall time is the timing
    sidecar's)."""
    scored = [r for r in records if r["checks"]]
    tokens: Counter[str] = Counter()
    for r in records:
        tokens.update({k: int(v) for k, v in r["tokens"].items()})

    def group(key: str) -> dict:
        out: dict[str, dict] = {}
        for value in sorted({r[key] for r in records}):
            chosen = [r["success"] for r in records if r[key] == value]
            out[value] = {"n": len(chosen), "success_rate": _rate(chosen)}
        return out

    return {
        "n": len(records), "success_rate": _rate([r["success"] for r in records]),
        "by_check": {check: _rate([r["checks"][check]["ok"] for r in scored]) for check in CHECKS},
        "by_intent": group("intent"), "by_lang": group("lang"),
        "statuses": dict(sorted(Counter(r["status"] for r in records).items())),
        "steps_mean": _rate([r["steps"] for r in records]) if records else None,
        "tool_errors": sum(sum(r["tool_errors"].values()) for r in records),
        "tokens": dict(sorted(tokens.items())), "cost_rmb": sum(r["cost_rmb"] for r in records),
    }

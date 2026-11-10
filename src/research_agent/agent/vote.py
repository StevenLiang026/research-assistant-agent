"""Self-consistency voting (docs/m7-本地提升计划.md WP2): a voted system asks a greedy candidate and K sampled
candidates for every step, keys their answers and answers with the representative of the class rule A
chooses.

Candidates. VotedModel(greedy, samples) wraps one greedy model and K sampled ones. The builder (eval.systems)
makes them from the system's generation settings with voted_settings: sample k (0-based) generates with the
base seed base_seed + k and the sample number 0, so that for the same context on the same server its answer
is byte for byte the answer of a run with --seed base_seed + k (llm.generation: every request's seed is
request_seed of the base seed, the prompt's ids, the phase and the sample number); the greedy candidate
decodes greedily and sends no seed, as a --greedy run does. With the default base seed 0, sample k is the
--seed k run: FT3@1635's seed runs s0 to s4 and its greedy run are the five samples and the greedy candidate
of a vote with K = 5.

Answer keys, two modes (answer_key):
- REPRODUCTION, exactly the 10-29 analysis' strict key (its voting_analysis.py, kept outside the repository,
  answer_key(record, "strict")): "STATUS:<status>" for a record that is not ok, "TEXT" for
  an answer without calls, else the sorted list of [tool name, canonical JSON of its arguments], the
  arguments normalised as the scorer compares them (metrics.normalize_call_arguments: the tool's defaults
  filled in, NFKC and whitespace, canonical arXiv ids and dates, categories, tags and missing as sets), the
  free text the scorer never compares replaced by FREE, or by FREE_BAD where the tool's schema refuses the
  value (write_note's six free parameters, metrics.SHAPE_CHECKED; every parameter of ask_clarification but
  missing, unchecked), a parameter write_note does not have kept (the scorer's 多参), every query
  casefolded; a call whose arguments are not a JSON object is [name, "!json:" + its text with whitespace runs
  collapsed].
- FINAL, the reproduction key with three changes: S1, a date_to of a tool in metrics.DATE_TO_TOOLS on the
  history's today (metrics.today_of: the system message's "今天是 ...", compared after normalize_date) is
  dropped before the defaults are filled in; S2, every query is folded with normalize.fold_hyphens (hyphens
  read as spaces) before it is casefolded; and the flag ids_in_history, whether every arXiv id in the
  answer's call arguments (metrics.arxiv_ids over their text) occurs in the history (agent.rules.history_ids,
  the scorer's reading of the whole history). An answer with calls has the canonical JSON of {"calls": the
  sorted parts, "ids_in_history": the flag} as its key, one without calls "TEXT"; a record that is not ok,
  or one with a call whose arguments are not a JSON object, has none (None): it does not vote.

Rule A (plurality). The K samples vote, each with its key; a sample without a key does not vote. The key held
by the most samples wins; a tie goes to the class of the greedy answer's key when it is among the tied, else
to the tied class with the smallest sample index; the representative is the member of the chosen class with
the smallest sample index. When no sample votes (all K failed, FINAL), the greedy answer is the output. In
REPRODUCTION mode every sample has a key, a failed one its STATUS key, so failed samples are a class like any
other, as in the analysis, whose plurality() over the samples with the tie-break order greedy, sample 0,
sample 1, ... is this rule (its last resort, the smallest tied key, is never reached: every tied key is a
sample's). When the runtime rules are on, every candidate is a RuledModel (agent.rules) and its answer is
keyed as the rules left it.

The vote (decide; AssistantTurn.votes, a record's "votes"): {"k": K, "mode", "greedy": the greedy answer's
key, "samples": the samples' keys in order (None where a sample has none), "counts": {key: the samples
holding it}, "chosen": the winning key (None when no sample voted), "representative": the index of the chosen
sample, or "greedy"}, and, only when a candidate failed, "errors": one entry per failed candidate, the greedy
one first, then the samples in order, each {"candidate": "greedy" or the sample index, "error_type",
"message"} (candidate_error).

Failed candidates. A candidate fails when its record is not ok: online, when it raised, a HarnessError
included (the model's answer could not be read, say); offline, when its run holds an error record, or any
other record that is not ok, for the item. The vote goes on without it (Rule A: in FINAL mode it has no key,
in REPRODUCTION mode its STATUS key votes as the analysis' did) and lists it under "errors". candidate_error
makes every entry, from the candidate's runner record: the error record's error_type and error, which are
eval.runner.error_fields of the exception online (VotedModel) and offline (the runner wrote the run's record
with the same function), so an entry reads byte for byte alike either way; a record that is not ok without
them (offline only) gives its status as the type and no message. So an end-to-end step goes on where the
failed candidate alone would have ended the task as harness_error (agent.loop): the attempt keeps the entry
under its votes, and scripts/run_e2e.py counts the failed candidates in its report. Only a vote whose chosen
answer is a failed one (the failed samples' class winning in REPRODUCTION mode, or every candidate failed)
still ends the step: its representative's exception is raised again (below).

Offline and online are one function. decide and voted_record take runner records (eval.runner: "status" and
an ok record's "output"); voted_record gives the representative's record with the usage of every candidate
that answered summed (summed_usage: whole numbers added, nested objects merged alike, any other value the
first one met) and the vote, a failed representative's record as it is but for the vote. VotedModel makes
such a record of every candidate's turn (status ok, its API form) or exception (status error, with
eval.runner.error_fields of it), takes decide's vote over them and answers with the representative's turn,
its usage the sum and its votes the vote. So its runner record is voted_record of the records the candidates
give one run at a time (scripts/vote_offline.py) whenever the candidates answer alike. A failed
representative's exception is raised again with the vote as its votes attribute, which the runner records.
The keys read the whole history (agent.loop.current_history; outside a request, the messages given). The
latency of a voted step is that of its K + 1 requests: the step's wall time spans them, and server_timings,
when given, is read after every candidate and last_timings sums those timings' counts and times
(TIMING_SUMS) for the timing sidecar.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Callable, Sequence

from ..canonical import canonical_json
from ..eval.metrics import (
    COMPARED,
    DATE_TO_TOOLS,
    QUERY_PARAMS,
    SHAPE_CHECKED,
    _message_text,
    arxiv_ids,
    normalize_call_arguments,
    today_of,
)
from ..eval.runner import ERROR, OK, error_fields
from ..family.core import GREEDY
from ..llm.budget import BudgetExceeded
from ..llm.generation import GenerationSettings
from ..normalize import fold_hyphens, normalize_date
from ..tools.schemas import TOOLS
from .loop import Model, begin_task, current_history
from .rules import history_ids
from .types import AssistantTurn, ToolCall

REPRODUCTION, FINAL = "reproduction", "final"
VOTE_MODES = (REPRODUCTION, FINAL)
TEXT = "TEXT"
FREE, FREE_BAD = "<free>", "<free:invalid>"
GREEDY_CANDIDATE = "greedy"  # the representative when no sample voted; the greedy candidate in "errors"
IDS_FLAG = "ids_in_history"
# Free text: the parameters of the tools in metrics.COMPARED that the scorer never compares.
FREE_TEXT = {t["function"]["name"]: frozenset(t["function"]["parameters"]["properties"])
             - COMPARED[t["function"]["name"]] for t in TOOLS if t["function"]["name"] in COMPARED}
TIMING_SUMS = ("prompt_n", "prompt_ms", "predicted_n", "predicted_ms", "cache_n")  # llama.cpp's, summed

Answer = AssistantTurn | Exception  # what a candidate gave: its turn, or what it raised


def _check_mode(mode: object) -> None:
    if mode not in VOTE_MODES:
        raise ValueError(f"unknown vote mode {mode!r}; one of {list(VOTE_MODES)}")


def _whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


# ---- answer keys ------------------------------------------------------------------------------------------


def _part(call: ToolCall, *, today: str | None, fold: bool) -> list[str] | None:
    """[name, canonical JSON of its normalised arguments] of a call (module docstring: Answer keys); None
    when its arguments are not a JSON object. today drops a date_to on that day (S1), fold folds the queries'
    hyphens (S2)."""
    if call.arguments is None:
        return None
    arguments = call.arguments
    date_to = arguments.get("date_to")
    if (today is not None and call.name in DATE_TO_TOOLS and isinstance(date_to, str)
            and normalize_date(date_to) == today):
        arguments = {k: v for k, v in arguments.items() if k != "date_to"}
    args = normalize_call_arguments(call.name, arguments)
    gate = SHAPE_CHECKED.get(call.name) or {}
    if call.name in COMPARED and call.name not in SHAPE_CHECKED:
        free = frozenset(args) - COMPARED[call.name]  # ask_clarification: only "missing" is compared
    else:
        free = FREE_TEXT.get(call.name, frozenset())  # write_note: an unknown parameter stays (多参)
    for key in list(args):
        if key in free:
            args[key] = FREE if key not in gate or gate[key].is_valid(call.arguments.get(key)) else FREE_BAD
    for key in QUERY_PARAMS:
        if isinstance(args.get(key), str):
            args[key] = (fold_hyphens(args[key]) if fold else args[key]).casefold()
    return [call.name, canonical_json(args)]


def _call_ids(output: dict) -> set[str]:
    """The arXiv ids in an answer's call arguments, read as the scorer reads them (metrics._message_text)."""
    return arxiv_ids(_message_text({"tool_calls": output.get("tool_calls") or []}))


def _key(record: dict, mode: str, today: str | None, ids: set[str]) -> str | None:
    """answer_key with the history read once: its today and its ids (FINAL only)."""
    final = mode == FINAL
    if record.get("status") != OK:
        return None if final else f"STATUS:{record.get('status')}"
    output = record.get("output") or {}
    calls = [ToolCall.from_api(c) for c in output.get("tool_calls") or []]
    if not calls:
        return TEXT
    parts = []
    for call in calls:
        part = _part(call, today=today if final else None, fold=final)
        if part is None:
            if final:
                return None
            part = [call.name, "!json:" + " ".join(call.arguments_json.split())]
        parts.append(part)
    parts.sort()
    if not final:
        return canonical_json(parts)
    return canonical_json({"calls": parts, IDS_FLAG: _call_ids(output) <= ids})


def _reading(history: list[dict], mode: str) -> tuple[str | None, set[str]]:
    """What the final key reads of the history: its today and its ids; nothing for the reproduction key."""
    if mode != FINAL:
        return None, set()
    return today_of({"messages": history}), history_ids(history)


def answer_key(record: dict, mode: str, history: list[dict]) -> str | None:
    """The key of a runner record's answer in a mode (module docstring: Answer keys) over the whole history
    before it; None for an answer that does not vote (FINAL). ValueError for an unknown mode."""
    _check_mode(mode)
    return _key(record, mode, *_reading(history, mode))


# ---- rule A and the vote ----------------------------------------------------------------------------------


def plurality(keys: Sequence[str | None], greedy_key: str | None) -> tuple[str | None, int | None]:
    """(the chosen key, its representative's sample index) by rule A over the samples' keys (None: a sample
    that does not vote); (None, None) when no sample votes."""
    voting = [(index, key) for index, key in enumerate(keys) if key is not None]
    if not voting:
        return None, None
    counts = Counter(key for _, key in voting)
    top = max(counts.values())
    tied = [key for _, key in voting if counts[key] == top]  # in sample order: the first, the smallest index
    chosen = greedy_key if greedy_key in tied else tied[0]
    return chosen, next(index for index, key in voting if key == chosen)


def candidate_error(candidate: int | str, record: dict) -> dict:
    """The entry of a failed candidate (module docstring: Failed candidates), candidate being its sample index
    or "greedy", read from its runner record: {"candidate", "error_type", "message"}, an error record's
    error_type and error, or the status and None of a record that is not ok without them."""
    return {"candidate": candidate, "error_type": record.get("error_type", record.get("status")),
            "message": record.get("error")}


def decide(greedy: dict, samples: Sequence[dict], history: list[dict], mode: str) -> dict:
    """The vote over the greedy candidate's record and the samples' (runner records) in a mode, over the whole
    history before them (module docstring: The vote), with "errors" when a candidate failed. ValueError for an
    unknown mode."""
    _check_mode(mode)
    reading = _reading(history, mode)
    greedy_key = _key(greedy, mode, *reading)
    keys = [_key(sample, mode, *reading) for sample in samples]
    chosen, representative = plurality(keys, greedy_key)
    votes = {"k": len(samples), "mode": mode, "greedy": greedy_key, "samples": keys,
             "counts": dict(Counter(key for key in keys if key is not None)), "chosen": chosen,
             "representative": GREEDY_CANDIDATE if representative is None else representative}
    errors = [candidate_error(candidate, record)
              for candidate, record in ((GREEDY_CANDIDATE, greedy), *enumerate(samples))
              if record.get("status") != OK]
    if errors:  # only then: a vote whose candidates all answered is as it was
        votes["errors"] = errors
    return votes


def summed_usage(usages: Sequence[dict]) -> dict:
    """The usages added up (module docstring: Offline and online are one function)."""
    total: dict = {}
    for usage in usages:
        _add(total, usage)
    return total


def _add(total: dict, usage: dict) -> None:
    for key, value in usage.items():
        if _whole(value) and _whole(total.get(key, 0)):
            total[key] = total.get(key, 0) + value
        elif isinstance(value, dict) and isinstance(total.get(key, {}), dict):
            _add(total.setdefault(key, {}), value)
        else:
            total.setdefault(key, value)


def voted_record(greedy: dict, samples: Sequence[dict], history: list[dict], mode: str) -> dict:
    """The record a voted system gives for these candidates' records (module docstring): the
    representative's, its usage the sum of every ok candidate's (an ok representative's only), and the
    vote."""
    votes = decide(greedy, samples, history, mode)
    index = votes["representative"]
    chosen = greedy if index == GREEDY_CANDIDATE else samples[index]
    record = dict(chosen)
    if chosen.get("status") == OK:
        record["usage"] = summed_usage([r.get("usage") or {} for r in (greedy, *samples)
                                        if r.get("status") == OK])
    record["votes"] = votes
    return record


def voted_settings(settings: GenerationSettings) -> tuple[GenerationSettings, list[GenerationSettings]]:
    """(the greedy candidate's settings, every sample's) of settings that vote (module docstring:
    Candidates): the greedy one decodes greedily, sample k samples with the base seed base_seed + k and the
    sample number 0; none of them votes. ValueError for settings without votes."""
    if not settings.votes:
        raise ValueError("these generation settings do not vote (votes 0)")
    greedy = dataclasses.replace(settings, decoding=GREEDY, votes=0)
    samples = [dataclasses.replace(settings, base_seed=settings.base_seed + k, sample=0, votes=0)
               for k in range(settings.votes)]
    return greedy, samples


def summed_timings(timings: Sequence[dict | None]) -> dict | None:
    """llama.cpp's timings of a voted step: the sum of each of TIMING_SUMS over the candidates' timings that
    have it (milliseconds rounded to the microsecond), None when no candidate has timings."""
    present = [t for t in timings if isinstance(t, dict)]
    if not present:
        return None
    total = {}
    for key in TIMING_SUMS:
        values = [t[key] for t in present
                  if isinstance(t.get(key), int | float) and not isinstance(t[key], bool)]
        if values:
            total[key] = round(sum(values), 3) if key.endswith("_ms") else sum(values)
    return total


def _as_record(answer: Answer) -> dict:
    """What decide reads of a candidate's answer: the runner record's status and output, or its error fields
    (eval.runner.error_fields, as the runner records the exception)."""
    if isinstance(answer, AssistantTurn):
        return {"status": OK, "output": answer.to_api()}
    return {"status": ERROR, **error_fields(answer)}


class VotedModel:
    """One greedy candidate and K sampled ones, answering by their vote in a mode (module docstring)."""

    def __init__(self, greedy: Model, samples: Sequence[Model], *, mode: str = FINAL,
                 server_timings: Callable[[], dict | None] | None = None) -> None:
        if not samples:
            raise ValueError("a vote needs at least one sample")
        _check_mode(mode)
        self.greedy, self.samples, self.mode = greedy, list(samples), mode
        self.server_timings = server_timings
        self.last_timings: dict | None = None  # the latest step's, summed over its candidates

    @property
    def name(self) -> str:
        return f"{self.greedy.name}+votes{len(self.samples)}"

    def begin_task(self) -> None:
        """A new task starts for every candidate that has the hook (agent.loop.begin_task)."""
        begin_task(self.greedy, *self.samples)

    def next_turn(self, messages: list[dict], tools: list[dict]) -> AssistantTurn:
        """The representative's turn, its usage that of every candidate that answered and its votes the vote;
        a failed representative's exception raised again with the vote (module docstring)."""
        answers: list[Answer] = []
        timings: list[dict | None] = []
        for model in (self.greedy, *self.samples):
            answers.append(self._ask(model, messages, tools))
            if self.server_timings is not None:
                timings.append(self.server_timings())
        self.last_timings = summed_timings(timings) if self.server_timings is not None else None
        history = current_history()
        records = [_as_record(answer) for answer in answers]
        votes = decide(records[0], records[1:], messages if history is None else history, self.mode)
        index = votes["representative"]
        chosen = answers[0] if index == GREEDY_CANDIDATE else answers[1 + index]
        if isinstance(chosen, Exception):
            chosen.votes = votes
            raise chosen
        usage = summed_usage([answer.usage for answer in answers if isinstance(answer, AssistantTurn)])
        return dataclasses.replace(chosen, usage=usage, votes=votes)

    @staticmethod
    def _ask(model: Model, messages: list[dict], tools: list[dict]) -> Answer:
        """A candidate's turn, or the exception it raised, a HarnessError included: a failed candidate, which
        the vote goes on without and lists (module docstring: Failed candidates). BudgetExceeded, the
        spending cap, ends the step as it would without a vote."""
        try:
            return model.next_turn(messages, tools)
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001  deliberate: a candidate's failure is its answer in the vote
            return exc

    def cost(self, turn: AssistantTurn) -> float:
        cost = getattr(self.greedy, "cost", None)
        return 0.0 if cost is None else cost(turn)

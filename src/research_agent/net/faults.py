"""Fault injection for the tool evaluation (docs/plan.md, 故障注入; M4): the adapters' fetcher and the search.

A FaultInjector answers like the fetcher it wraps, except that a fraction `rate` of the requests meets one of
seven faults. Use one injector per evaluation task. Whether a request meets a fault, and which, follows from a
SHA-256 of (seed, task, canonical URL, the request's ordinal among this task's requests for that URL). So the
faults of a task do not depend on what other tasks fetched or in which order the tasks ran, a fault table can
be reproduced, and two systems (FT, B0, B2) running the same task meet the same fault on their k-th request
for the same URL: the per-item pairing of the plan's statistics holds. With persistent=True the ordinal is
left out and every request of the task for a URL meets the same fault, an interface that stays down. Retry
storms against it are counted from the agent loop's calls, not from this log: the loop already refuses a call
that ran twice in the turn (agent/loop.py, MAX_RUNS_PER_CALL). The plan's rates are 0, 0.2 and 0.5 (RATES).

The wrapped fetcher is always asked first, so a URL without a cassette still raises ReplayMiss, a harness
failure, instead of hiding behind an injected fault. Every fault reaches the adapters in one of the two forms
they already handle, a FetchFailed or a page:

    fault           applies to            what the adapters receive               error code the model sees
    timeout         API and HTML          FetchFailed without a status            UPSTREAM_ERROR
    rate_limited    API and HTML          FetchFailed 429, Retry-After in reason  RATE_LIMITED
    server_error    API and HTML          FetchFailed 500                         UPSTREAM_ERROR
    empty_results   API answers with 200  the feed without its entries            EMPTY_RESULTS / NOT_FOUND
    truncated_atom  API answers with 200  the first half of the feed              UPSTREAM_ERROR (*)
    html_404        HTML answers with 200 status 404                              NO_FULLTEXT, abstract
    delay           API and HTML          the answer, DELAY_S seconds later       none: latency only

"API and HTML" are the fetches of get_paper and fetch_fulltext: the search sends none through a fetcher
(below), so delay and truncated_atom, like html_404, apply to fetches only, and a search can meet only
timeout, rate_limited, server_error and empty_results, through SearchFaults. In M4 the latency row of the
fault table therefore covers the two fetch tools.

The codes of the five status faults follow from the adapters alone (tests/net/test_faults.py runs them through
the real adapters). The codes of the two feed faults also depend on the owner's parse_atom: [] for a feed
without entries, and (*) None for a feed it cannot parse, as its contract asks (docs/modules/接口.md); the
tests script those answers instead of calling the module. truncated_atom is the plan's 截断的 Atom, a 200
whose feed breaks off; a live download that breaks off would more often surface as a transport error, the
timeout row.
A 429 or 500 here stands for one that outlasted the fetcher's retries: replay retries nothing, so an injected
fault is what a live run shows the model once the 3 s, 6 s and 12 s back-offs are spent. A timeout costs no
waiting time here. The two feed faults edit the recorded feed as text (entries cut out, totalResults set to 0;
or the second half cut off); they read no field of it, which stays the owner's parse_atom. Every decision is
logged (task, canonical URL, ordinal, fault or None) for the fault table.

From M2 on, search_arxiv answers from the recorded corpus (tools/corpus_search.py) and none of its requests
reaches a fetcher, so the search has a fault hook of its own. SearchFaults wraps the tool function itself
(args -> ToolOutcome) and draws from the same hash, with "search:" plus the canonical arguments, defaults
filled in and integral floats made ints as call keys are (agent/types.py, normalize_arguments), in place of
the canonical URL (draw_key takes any key verbatim; draw is draw_key over the canonical URL), with the ordinal
left out when persistent, as for a FaultInjector. Its four faults
are what the adapter makes of a fetcher fault, already in the tool's form: timeout and server_error give
([], UPSTREAM_ERROR), rate_limited ([], RATE_LIMITED) and empty_results ([], EMPTY_RESULTS), on which the
plan's "空结果后的编号编造" metric depends. A faulted outcome never carries an arxiv_id, so an injected fault
adds nothing to the ids the model has seen; and the wrapped tool is called first, as the fetcher is, so a
harness error in the replay search surfaces as it is.

A FaultBudget caps how many faults a run injects in all. An injector asks budget.take() only once it has
drawn a fault that applies, and a refused fault is logged with fault None and "budget_exhausted": True, so a
run with a budget draws exactly what the run without one draws and the fault table still shows where the
budget bit. One budget may be shared by several injectors, a FaultInjector under get_paper and fetch_fulltext
and a SearchFaults over search_arxiv, so a trajectory meets exactly one fault overall, whichever tool it comes
through; with persistent=True a budget of k is an interface that is down for the first k requests and then
up (see FaultInjector).
"""

from __future__ import annotations

import hashlib
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Protocol

from ..agent.types import ToolFn, ToolOutcome, normalize_arguments
from ..canonical import canonical_json
from ..tools.schemas import DEFAULTS, EMPTY_RESULTS, RATE_LIMITED, UPSTREAM_ERROR
from .fetch import API_DIR, Fetched, FetchFailed, canonical_url, cassette_folder

FAULTS = ("timeout", "rate_limited", "server_error", "empty_results", "truncated_atom", "html_404", "delay")
ANY_ANSWER = ("timeout", "rate_limited", "server_error", "delay")  # replace or slow down any request
API_200 = ("empty_results", "truncated_atom")  # edit a feed the API answered with 200
HTML_200 = ("html_404",)  # replace a full text the HTML server answered with 200
SEARCH_FAULTS = ("timeout", "rate_limited", "server_error", "empty_results")  # what SearchFaults can inject
RATES = (0.0, 0.2, 0.5)
DELAY_S = 5.0
RETRY_AFTER_S = 30
_ENTRY = re.compile(r"<entry\b.*?</entry>\s*", re.DOTALL)
_TOTAL = re.compile(r"(<opensearch:totalResults\b[^>]*>)\s*\d+\s*(<)")
# The error code of each search fault: the code the adapter gives search_arxiv for the fetcher fault of the
# same name (the table above), the content [] as there.
_SEARCH_CODES = {"timeout": UPSTREAM_ERROR, "rate_limited": RATE_LIMITED, "server_error": UPSTREAM_ERROR,
                 "empty_results": EMPTY_RESULTS}


class SupportsGet(Protocol):
    def get(self, url: str) -> Fetched: ...


def applicable(url: str, status: int, faults: Sequence[str] = FAULTS) -> list[str]:
    """The faults that can meet a request for url whose real answer has this status, in FAULTS order."""
    content = API_200 if cassette_folder(canonical_url(url)) == API_DIR else HTML_200
    allowed = ANY_ANSWER + (content if status == 200 else ())
    return [fault for fault in FAULTS if fault in allowed and fault in faults]


def draw_key(seed: int, task: str, key: str, ordinal: int | None) -> tuple[float, int]:
    """(u, k): u, uniform in [0, 1), decides whether a fault is injected and k picks which one. Both come from
    a SHA-256 of seed, task, key and ordinal (None for a persistent fault): nothing else matters. The key is
    hashed verbatim; a caller that wants two spellings of one request to draw alike canonicalises first."""
    text = f"{seed}\x1f{task}\x1f{key}\x1f{'*' if ordinal is None else ordinal}"
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64, int.from_bytes(digest[8:16], "big")


def draw(seed: int, task: str, url: str, ordinal: int | None) -> tuple[float, int]:
    """draw_key over the canonical URL: equal URLs written differently draw alike."""
    return draw_key(seed, task, canonical_url(url), ordinal)


def search_key(args: dict) -> str:
    """The key a search draws by: "search:" plus the canonical arguments with the schema defaults filled in
    and integral floats made ints, as the loop's call key has them, so a search asked with and without its
    defaults, or with max_results 3.0 in place of 3, is one search, as it is one call key for the loop."""
    return "search:" + canonical_json(normalize_arguments({**DEFAULTS["search_arxiv"], **args}))


def empty_feed(feed: str) -> str:
    """feed with every entry cut out and totalResults set to 0: an answer without results."""
    return _TOTAL.sub(r"\g<1>0\g<2>", _ENTRY.sub("", feed))


def truncated(text: str) -> str:
    """The first half of text, as a download that broke off would leave it."""
    return text[: len(text) // 2]


class FaultBudget:
    """How many faults a run may inject in all; None for no cap. Injectors given the same budget share its
    count, so a trajectory can be held to exactly one fault whichever tool it comes through."""

    def __init__(self, max_faults: int | None) -> None:
        if max_faults is not None and (isinstance(max_faults, bool) or not isinstance(max_faults, int)
                                       or max_faults < 0):
            raise ValueError(f"max_faults must be None or an integer >= 0, not {max_faults!r}")
        self.max_faults = max_faults
        self._injected = 0

    def take(self) -> bool:
        """Whether one more fault may be injected; when so, it is counted."""
        if self.max_faults is not None and self._injected >= self.max_faults:
            return False
        self._injected += 1
        return True

    @property
    def injected(self) -> int:
        """Faults granted so far."""
        return self._injected

    @property
    def remaining(self) -> int | None:
        """Faults still allowed; None when there is no cap."""
        return None if self.max_faults is None else self.max_faults - self._injected


class FaultInjector:
    """A fetcher that turns a deterministic fraction of the wrapped fetcher's answers into faults, for one
    task: its ordinals start at 0 and its draws carry the task's name. A budget caps the faults it injects
    (together with every other injector given the same budget): a request whose drawn fault the budget
    refuses is answered as recorded and logged with "budget_exhausted". persistent=True with a budget of k
    is an interface that is down for the first k requests and then up: every request for a URL draws the
    same fault, the first k requests that draw one meet it, and all later ones are answered as recorded."""

    def __init__(self, inner: SupportsGet, *, rate: float, seed: int, task: str,
                 faults: Sequence[str] = FAULTS, persistent: bool = False,
                 sleep: Callable[[float], None] = time.sleep, budget: FaultBudget | None = None) -> None:
        self.faults = _checked(rate, faults, FAULTS)
        self.inner, self.rate, self.seed, self.task, self.budget = inner, rate, seed, task, budget
        self.persistent, self.sleep = persistent, sleep
        self.log: list[dict] = []
        self._ordinals: Counter[str] = Counter()

    def get(self, url: str) -> Fetched:
        page = self.inner.get(url)  # first: a ReplayMiss or a real failure surfaces as it is
        canonical = canonical_url(url)
        ordinal = self._ordinals[canonical]
        self._ordinals[canonical] += 1
        u, k = draw(self.seed, self.task, canonical, None if self.persistent else ordinal)
        choices = applicable(url, page.status, self.faults)
        fault = choices[k % len(choices)] if u < self.rate and choices else None
        entry = {"task": self.task, "url": canonical, "ordinal": ordinal, "fault": fault}
        fault = _budgeted(self.budget, entry)
        self.log.append(entry)
        return self._apply(fault, url, page)

    def counts(self) -> dict[str, int]:
        """Requests per fault, with "none" for the requests that met none; for the fault table."""
        return _tally(self.log, FAULTS)

    def _apply(self, fault: str | None, url: str, page: Fetched) -> Fetched:
        if fault is None:
            return page
        if fault == "timeout":
            raise FetchFailed(url, None, "injected fault: timed out")
        if fault == "rate_limited":
            raise FetchFailed(url, 429, f"injected fault: status 429, Retry-After: {RETRY_AFTER_S}")
        if fault == "server_error":
            raise FetchFailed(url, 500, "injected fault: status 500")
        if fault == "delay":
            self.sleep(DELAY_S)
            return page
        if fault == "html_404":
            return Fetched(url=page.url, final_url=page.final_url, status=404, body=b"",
                           content_type=page.content_type)
        text = empty_feed(page.text) if fault == "empty_results" else truncated(page.text)
        return Fetched(url=page.url, final_url=page.final_url, status=page.status, body=text.encode("utf-8"),
                       content_type=page.content_type)


class SearchFaults:
    """search_arxiv with a deterministic fraction of its answers turned into the outcomes the adapter gives
    for a failed or empty fetch; a tool function itself (args -> ToolOutcome), so the executor calls it like
    the tool it wraps. One per task, like a FaultInjector: a draw is keyed by search_key(args), so a search
    meets the same fault on its k-th repetition in every system, and the ordinal counts per key. With
    persistent=True the ordinal is left out of the draw, as a persistent FaultInjector leaves it out: every
    repetition of a search meets the same fault (the ordinal is still logged)."""

    def __init__(self, inner: ToolFn, *, rate: float, seed: int, task: str,
                 faults: Sequence[str] = SEARCH_FAULTS, persistent: bool = False,
                 budget: FaultBudget | None = None) -> None:
        self.faults = _checked(rate, faults, SEARCH_FAULTS)
        self.inner, self.rate, self.seed, self.task, self.budget = inner, rate, seed, task, budget
        self.persistent = persistent
        self.log: list[dict] = []
        self._ordinals: Counter[str] = Counter()

    def __call__(self, args: dict) -> ToolOutcome:
        outcome = self.inner(args)  # first: a harness error in the replay search surfaces as it is
        key = search_key(args)
        ordinal = self._ordinals[key]
        self._ordinals[key] += 1
        u, k = draw_key(self.seed, self.task, key, None if self.persistent else ordinal)
        fault = self.faults[k % len(self.faults)] if u < self.rate and self.faults else None
        entry = {"task": self.task, "key": key, "ordinal": ordinal, "fault": fault}
        fault = _budgeted(self.budget, entry)
        self.log.append(entry)
        return outcome if fault is None else ToolOutcome([], _SEARCH_CODES[fault])

    def counts(self) -> dict[str, int]:
        """Searches per fault, with "none" for the searches that met none; for the fault table."""
        return _tally(self.log, SEARCH_FAULTS)


def _checked(rate: float, faults: Sequence[str], known: Sequence[str]) -> tuple[str, ...]:
    """faults as a tuple once rate and faults are checked; a ValueError names what is wrong."""
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"rate must be in [0, 1], not {rate!r}")
    unknown = sorted(set(faults) - set(known))
    if unknown:
        raise ValueError(f"unknown faults {unknown}; known: {list(known)}")
    return tuple(faults)


def _budgeted(budget: FaultBudget | None, entry: dict) -> str | None:
    """The fault of a log entry once the budget has been asked: the drawn one when there is no budget or it
    grants one more, else None with the entry marked "budget_exhausted". Only a drawn fault asks, so a
    request that drew none costs the budget nothing."""
    if entry["fault"] is not None and budget is not None and not budget.take():
        entry["fault"], entry["budget_exhausted"] = None, True
    return entry["fault"]


def _tally(log: list[dict], names: Sequence[str]) -> dict[str, int]:
    """Log entries per fault, "none" first and then the names in their given order; absent ones left out."""
    tally = Counter(entry["fault"] or "none" for entry in log)
    return {name: tally[name] for name in ("none", *names) if tally[name]}

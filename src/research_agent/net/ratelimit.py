"""Per-host rate limits for the tools' web requests.

arXiv asks for at most one API request every 3 s (export.arxiv.org), and arxiv.org/robots.txt sets a 15 s
crawl delay for the HTML full texts. The fetcher waits before every request it sends, keyed by that request's
own host: retries wait too, because the first 3 s backoff is shorter than the 15 s delay, and so do redirect
hops, which may lead to another host.

Limits live in one HostLimits object: share it (or the Fetcher holding it) within a process, since separate
instances do not know about each other's requests.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from urllib.parse import urlsplit

DEFAULT_INTERVALS = {"export.arxiv.org": 3.0, "arxiv.org": 15.0}


class RateLimiter:
    """Keeps calls at least min_interval seconds apart; the first call never waits."""

    def __init__(self, min_interval: float, *, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.min_interval, self.clock, self.sleep = min_interval, clock, sleep
        self._last: float | None = None  # when the previous wait() returned

    def wait(self) -> float:
        """Sleep until min_interval has passed since the previous call returned; returns the seconds slept."""
        delay = 0.0 if self._last is None else max(0.0, self._last + self.min_interval - self.clock())
        if delay > 0:
            self.sleep(delay)
        self._last = self.clock()
        return delay


class HostLimits:
    """One RateLimiter per host. Hosts without an interval of their own use the default.

    A host is matched lower-cased and without the root dot of a fully qualified name ("arxiv.org." is
    arxiv.org), so no spelling of a host escapes its limit.
    """

    def __init__(self, intervals: dict[str, float] | None = None, default: float = 1.0, *,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        source = DEFAULT_INTERVALS if intervals is None else intervals
        self.intervals = {_host_key(host): seconds for host, seconds in source.items()}
        self.default, self.clock, self.sleep = default, clock, sleep
        self._limiters: dict[str, RateLimiter] = {}

    def wait(self, url: str) -> float:
        """Wait as long as the URL's host requires; returns the seconds slept."""
        host = _host_key(urlsplit(url).hostname or "")
        if host not in self._limiters:
            self._limiters[host] = RateLimiter(self.intervals.get(host, self.default), clock=self.clock,
                                               sleep=self.sleep)
        return self._limiters[host].wait()


def _host_key(host: str) -> str:
    return host.lower().rstrip(".")

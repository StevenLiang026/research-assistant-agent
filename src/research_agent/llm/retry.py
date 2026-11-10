"""Fixed-backoff retries (3 s, 6 s, 12 s) for rate limits, timeouts and server errors."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

DELAYS = (3.0, 6.0, 12.0)


def retry_after_seconds(exc: BaseException) -> float | None:
    """Delay asked for by a Retry-After header (seconds form) on the exception's HTTP response, if any."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    value = (headers.get("retry-after") or headers.get("Retry-After")) if headers else None
    try:
        return float(value) if value is not None else None
    except ValueError:  # the HTTP-date form is not used by the APIs here
        return None


def call_with_retries[T](fn: Callable[[], T], *, retryable: Callable[[BaseException], bool],
                         delays: Sequence[float] = DELAYS, max_wait: float = 60.0,
                         sleep: Callable[[float], None] = time.sleep) -> T:
    """Call fn; after a retryable error wait delays[i] (a longer Retry-After wins, capped at max_wait)."""
    for attempt in range(len(delays) + 1):
        try:
            return fn()
        except Exception as exc:
            if attempt == len(delays) or not retryable(exc):
                raise
            hinted = retry_after_seconds(exc)
            sleep(min(max(delays[attempt], hinted or 0.0), max_wait))
    raise AssertionError("unreachable")

import pytest

from research_agent.llm.retry import call_with_retries, retry_after_seconds


class Flaky(Exception):
    pass


class RateLimited(Exception):
    def __init__(self, retry_after: str):
        super().__init__("rate limited")
        self.response = type("Response", (), {"headers": {"retry-after": retry_after}})()


def failing(n: int, exc: type[Exception] = Flaky):
    state = {"calls": 0}

    def fn():
        state["calls"] += 1
        if state["calls"] <= n:
            raise exc("boom")
        return "ok"

    return fn, state


def test_success_needs_no_wait():
    sleeps = []
    fn, state = failing(0)
    assert call_with_retries(fn, retryable=lambda e: True, sleep=sleeps.append) == "ok"
    assert state["calls"] == 1 and sleeps == []


def test_backoff_is_3_6_12_seconds():
    sleeps = []
    fn, state = failing(3)
    assert call_with_retries(fn, retryable=lambda e: isinstance(e, Flaky), sleep=sleeps.append) == "ok"
    assert sleeps == [3.0, 6.0, 12.0] and state["calls"] == 4


def test_gives_up_after_three_retries():
    sleeps = []
    fn, state = failing(10)
    with pytest.raises(Flaky):
        call_with_retries(fn, retryable=lambda e: True, sleep=sleeps.append)
    assert state["calls"] == 4 and sleeps == [3.0, 6.0, 12.0]


def test_non_retryable_error_is_raised_at_once():
    sleeps = []
    fn, state = failing(1, exc=ValueError)
    with pytest.raises(ValueError):
        call_with_retries(fn, retryable=lambda e: isinstance(e, Flaky), sleep=sleeps.append)
    assert state["calls"] == 1 and sleeps == []


def test_retry_after_parsing():
    assert retry_after_seconds(RateLimited("20")) == 20.0
    assert retry_after_seconds(RateLimited("Wed, 21 Oct 2026 07:28:00 GMT")) is None
    assert retry_after_seconds(Flaky("no response")) is None


def test_retry_after_lengthens_the_wait_up_to_the_cap():
    sleeps = []
    errors = iter([RateLimited("20"), RateLimited("1"), RateLimited("999")])

    def fn():
        exc = next(errors, None)
        if exc is not None:
            raise exc
        return "ok"

    assert call_with_retries(fn, retryable=lambda e: True, sleep=sleeps.append, max_wait=60.0) == "ok"
    assert sleeps == [20.0, 6.0, 60.0]

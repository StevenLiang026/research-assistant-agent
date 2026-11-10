from datetime import datetime

import httpx2
import openai
import pytest

from research_agent.llm.budget import Budget, BudgetExceeded, OffPeak, PeakHours, Price
from research_agent.llm.cache import ResponseCache
from research_agent.llm.cloud import NON_THINKING, THINKING_OFF, CloudClient, MissingKeyError, is_retryable

MESSAGES = [{"role": "user", "content": "找三篇推测解码的论文"}]
USAGE = {"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100}
COST = (1000 * 2.0 + 100 * 8.0) / 1e6


class FakeResponse:
    def __init__(self, data: dict):
        self.data = data

    def model_dump(self, **_):
        return self.data


class FakeOpenAI:
    """Stands in for openai.OpenAI: scripted responses or exceptions, every payload recorded."""

    def __init__(self, script: list):
        self.script, self.payloads = list(script), []
        self.chat = type("Chat", (), {})()
        self.chat.completions = self

    def create(self, **payload):
        self.payloads.append(payload)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)


def answer(text: str = "ok") -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}], "usage": USAGE}


def budget(tmp_path, cap: float = 1.0, name: str = "b.sqlite") -> Budget:
    return Budget(tmp_path / name, stage="M1", stage_cap_rmb=cap, prices={"flash": Price(2.0, 8.0)})


@pytest.fixture
def cache(tmp_path):
    c = ResponseCache(tmp_path / "cache.sqlite")
    yield c
    c.close()


def test_identical_request_is_served_from_cache_and_charged_once(tmp_path, cache):
    fake, ledger = FakeOpenAI([answer("first")]), budget(tmp_path)
    client = CloudClient("flash", cache=cache, budget=ledger, client=fake)
    first, second = client.chat(MESSAGES, temperature=0), client.chat(MESSAGES, temperature=0)
    assert first == second and first["choices"][0]["message"]["content"] == "first"
    assert len(fake.payloads) == 1 and client.calls == 1 and client.cache_hits == 1
    assert ledger.spent("M1") == pytest.approx(COST)


def test_payload_is_model_messages_tools_and_params(tmp_path, cache):
    fake = FakeOpenAI([answer()])
    tools = [{"type": "function", "function": {"name": "search_arxiv", "parameters": {"type": "object"}}}]
    CloudClient("flash", cache=cache, budget=budget(tmp_path), client=fake).chat(MESSAGES, tools=tools,
                                                                                temperature=0)
    assert fake.payloads[0] == {"model": "flash", "messages": MESSAGES, "tools": tools, "temperature": 0}


def test_cap_stops_the_run_but_the_paid_answer_is_kept(tmp_path, cache):
    fake = FakeOpenAI([answer("paid")])
    client = CloudClient("flash", cache=cache, budget=budget(tmp_path, cap=0.001), client=fake)
    with pytest.raises(BudgetExceeded):
        client.chat(MESSAGES)  # costs 0.0028 > 0.001: recorded, then the run stops
    assert client.chat(MESSAGES)["choices"][0]["message"]["content"] == "paid"  # a rerun is free
    with pytest.raises(BudgetExceeded):
        client.chat([{"role": "user", "content": "another request"}])  # refused before any call
    assert len(fake.payloads) == 1


def test_retryable_error_then_success(tmp_path, cache):
    sleeps = []
    fake = FakeOpenAI([TimeoutError("slow"), answer("after retry")])
    client = CloudClient("flash", cache=cache, budget=budget(tmp_path), client=fake, sleep=sleeps.append,
                         retryable=lambda e: isinstance(e, TimeoutError))
    assert client.chat(MESSAGES)["choices"][0]["message"]["content"] == "after retry"
    assert sleeps == [3.0] and client.calls == 1


def test_missing_key_is_reported_by_name(tmp_path, cache, monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(MissingKeyError, match="DEEPSEEK_API_KEY"):
        CloudClient("flash", cache=cache, budget=budget(tmp_path))


def test_model_without_price_is_refused_before_any_call(tmp_path, cache):
    with pytest.raises(KeyError):
        CloudClient("unpriced", cache=cache, budget=budget(tmp_path), client=FakeOpenAI([]))


def test_sdk_errors_are_classified():
    request = httpx2.Request("POST", "https://api.deepseek.com/chat/completions")

    def status(cls, code):
        return cls("x", response=httpx2.Response(code, request=request), body=None)

    assert is_retryable(openai.APITimeoutError(request=request))
    assert is_retryable(openai.APIConnectionError(request=request))
    assert is_retryable(status(openai.RateLimitError, 429))
    assert is_retryable(status(openai.InternalServerError, 500))
    assert is_retryable(status(openai.APIStatusError, 503))
    assert not is_retryable(status(openai.BadRequestError, 400))
    assert not is_retryable(status(openai.AuthenticationError, 401))


# ---- M7: thinking off on request, paid calls only off peak (docs/m7-计划.md WP1) --------------------------

TABLE = OffPeak(factor=0.5, utc_offset_hours=8, weekdays=(0, 1, 2, 3, 4), windows=((540, 720), (840, 1080)))


def test_thinking_off_sends_the_switch_and_never_serves_a_thinking_answer(tmp_path, cache):
    fake, ledger = FakeOpenAI([answer("thinking"), answer("plain")]), budget(tmp_path)
    CloudClient("flash", cache=cache, budget=ledger, client=fake).chat(MESSAGES)  # cached under the plain key
    off = CloudClient("flash", cache=cache, budget=ledger, client=fake, thinking="off")
    data = off.chat(MESSAGES)
    assert data["choices"][0]["message"]["content"] == "plain" and off.last_mode == NON_THINKING
    assert fake.payloads[1]["extra_body"] == THINKING_OFF
    assert off.chat(MESSAGES) == data and len(fake.payloads) == 2  # its own answer comes from the cache
    # a known reasoning is not attached either: the request stays a non-thinking one
    fake.script.append(answer("again"))
    history = [*MESSAGES, {"role": "assistant", "content": "earlier"}, {"role": "user", "content": "go on"}]
    off.chat(history, reasoning_of=lambda message: "known")
    assert fake.payloads[2]["extra_body"] == THINKING_OFF
    assert all("reasoning_content" not in m for m in fake.payloads[2]["messages"])
    with pytest.raises(ValueError, match="thinking"):
        CloudClient("flash", cache=cache, budget=ledger, client=fake, thinking="sometimes")


def test_off_peak_only_refuses_a_paid_call_in_the_peak_and_still_serves_the_cache(tmp_path, cache):
    fake = FakeOpenAI([answer("before")])
    CloudClient("flash", cache=cache, budget=budget(tmp_path), client=fake).chat(MESSAGES)
    peak = Budget(tmp_path / "peak.sqlite", stage="M1", stage_cap_rmb=1.0, prices={"flash": Price(2.0, 8.0)},
                  off_peak=TABLE, clock=lambda: datetime.fromisoformat("2026-10-19T06:43:00+00:00"))
    guarded = CloudClient("flash", cache=cache, budget=peak, client=fake, off_peak_only=True)
    assert guarded.chat(MESSAGES)["choices"][0]["message"]["content"] == "before"  # free: from the cache
    with pytest.raises(PeakHours):
        guarded.chat([{"role": "user", "content": "another request"}])
    assert len(fake.payloads) == 1 and peak.spent() == 0.0
    with pytest.raises(ValueError, match="peak table"):
        CloudClient("flash", cache=cache, budget=budget(tmp_path, name="c.sqlite"), client=fake,
                    off_peak_only=True)


def test_a_paid_call_that_starts_in_the_peak_is_charged_in_full(tmp_path, cache):
    times = iter([datetime.fromisoformat("2026-10-19T09:59:59+00:00"),  # 17:59:59 in Beijing: sent
                  datetime.fromisoformat("2026-10-19T10:00:01+00:00")])  # 18:00:01: answered
    ledger = Budget(tmp_path / "t.sqlite", stage="M1", stage_cap_rmb=1.0, prices={"flash": Price(2.0, 8.0)},
                    off_peak=TABLE, clock=lambda: next(times))
    CloudClient("flash", cache=cache, budget=ledger, client=FakeOpenAI([answer()])).chat(MESSAGES)
    assert ledger.spent("M1") == pytest.approx(COST)

from datetime import datetime

import pytest

from research_agent.llm.budget import (
    Budget,
    BudgetExceeded,
    OffPeak,
    PeakHours,
    Price,
    cost_rmb,
    is_peak,
    load_off_peak,
    load_prices,
)

FLASH = Price(input_cache_miss=2.0, output=8.0)


def make(tmp_path, stage="M1", cap=1.0, total=300.0):
    return Budget(tmp_path / "b.sqlite", stage=stage, stage_cap_rmb=cap, total_cap_rmb=total,
                  prices={"flash": FLASH})


def test_without_cache_split_all_input_is_charged_at_the_miss_price():
    assert cost_rmb(FLASH, {"prompt_tokens": 1_000_000, "completion_tokens": 0}) == pytest.approx(2.0)
    assert cost_rmb(FLASH, {"prompt_tokens": 0, "completion_tokens": 500_000}) == pytest.approx(4.0)


def test_deepseek_cache_split_is_priced_per_part():
    price = Price(input_cache_miss=2.0, output=8.0, input_cache_hit=0.5)
    usage = {"prompt_tokens": 1000, "prompt_cache_hit_tokens": 600, "prompt_cache_miss_tokens": 400,
             "completion_tokens": 100}
    assert cost_rmb(price, usage) == pytest.approx((600 * 0.5 + 400 * 2.0 + 100 * 8.0) / 1e6)


def test_unknown_hit_price_is_charged_at_the_miss_price():
    usage = {"prompt_tokens": 1000, "prompt_cache_hit_tokens": 1000, "prompt_cache_miss_tokens": 0,
             "completion_tokens": 0}
    assert cost_rmb(FLASH, usage) == pytest.approx(1000 * 2.0 / 1e6)


def test_charges_accumulate_and_the_stage_cap_stops_the_run(tmp_path):
    budget = make(tmp_path, cap=1.0)
    assert budget.charge("flash", {"prompt_tokens": 250_000, "completion_tokens": 0}) == pytest.approx(0.5)
    budget.check()
    with pytest.raises(BudgetExceeded, match="stage M1"):
        budget.charge("flash", {"prompt_tokens": 250_000, "completion_tokens": 0})
    with pytest.raises(BudgetExceeded):
        budget.check()
    assert budget.spent("M1") == pytest.approx(1.0)  # the call that reached the cap is still on the ledger


def test_total_cap_spans_stages(tmp_path):
    make(tmp_path, stage="M1", cap=10.0, total=1.0).charge("flash", {"prompt_tokens": 400_000})
    m2 = make(tmp_path, stage="M2", cap=10.0, total=1.0)
    m2.check()
    with pytest.raises(BudgetExceeded, match="all stages"):
        m2.charge("flash", {"prompt_tokens": 100_000})
    assert m2.spent("M2") == pytest.approx(0.2) and m2.spent() == pytest.approx(1.0)


def test_model_without_price_is_refused(tmp_path):
    with pytest.raises(KeyError):
        make(tmp_path).charge("unpriced", {"prompt_tokens": 1})


def test_ledger_persists(tmp_path):
    make(tmp_path).charge("flash", {"prompt_tokens": 100_000, "completion_tokens": 0})
    assert make(tmp_path).spent("M1") == pytest.approx(0.2)


def test_shipped_price_list_prices_flash():
    flash = load_prices()["deepseek-flash"]
    assert flash.input_cache_miss > 0 and flash.output > 0


# ---- M7: off-peak pricing (docs/m7-计划.md WP1) ----------------------------------------------------------

# DeepSeek's price page (read 2026-10-19): Beijing weekdays 09:00-12:00 and 14:00-18:00 are peak hours,
# everything else is charged at half the price
TABLE = OffPeak(factor=0.5, utc_offset_hours=8, weekdays=(0, 1, 2, 3, 4), windows=((540, 720), (840, 1080)))
QUARTER = {"prompt_tokens": 250_000, "completion_tokens": 0}  # 0.5 RMB at the list price


def at(text: str) -> datetime:
    return datetime.fromisoformat(text)


def test_the_peak_windows_are_beijing_weekday_mornings_and_afternoons():
    # 2026-10-19 is a Monday; Beijing is UTC+8 all year
    assert is_peak(at("2026-10-19T01:00:00+00:00"), TABLE)  # 09:00 in Beijing: a window starts
    assert is_peak(at("2026-10-19T06:43:00+00:00"), TABLE)  # 14:43
    assert not is_peak(at("2026-10-19T04:00:00+00:00"), TABLE)  # 12:00: the morning window has ended
    assert not is_peak(at("2026-10-19T10:00:00+00:00"), TABLE)  # 18:00
    assert not is_peak(at("2026-10-24T02:00:00+00:00"), TABLE)  # a Saturday, 10:00
    # an aware time counts by its instant whatever its offset: Monday 09:30 given in Beijing's own offset
    assert is_peak(at("2026-10-19T09:30:00+08:00"), TABLE)
    with pytest.raises(ValueError, match="time zone"):
        is_peak(datetime(2026, 10, 19, 10, 0), TABLE)  # noqa: DTZ001  deliberately naive


def test_the_ledger_charges_an_off_peak_call_at_the_factor_and_a_peak_call_in_full(tmp_path):
    times = iter([at("2026-10-19T13:00:00+00:00"), at("2026-10-19T06:00:00+00:00")])
    budget = Budget(tmp_path / "b.sqlite", stage="M7", stage_cap_rmb=10.0, prices={"flash": FLASH},
                    off_peak=TABLE, clock=lambda: next(times))
    assert budget.charge("flash", QUARTER) == pytest.approx(0.25)
    assert budget.charge("flash", QUARTER) == pytest.approx(0.5)
    assert budget.spent("M7") == pytest.approx(0.75)


def test_a_call_that_started_in_the_peak_is_charged_in_full(tmp_path):
    budget = Budget(tmp_path / "b.sqlite", stage="M7", stage_cap_rmb=10.0, prices={"flash": FLASH},
                    off_peak=TABLE, clock=lambda: at("2026-10-19T10:00:01+00:00"))  # 18:00:01: off-peak
    assert budget.charge("flash", QUARTER, started=at("2026-10-19T09:59:59+00:00")) == pytest.approx(0.5)
    assert budget.charge("flash", QUARTER, started=at("2026-10-19T10:00:00+00:00")) == pytest.approx(0.25)


def test_without_a_peak_table_every_call_is_charged_at_the_list_price(tmp_path):
    budget = Budget(tmp_path / "b.sqlite", stage="M7", stage_cap_rmb=10.0, prices={"flash": FLASH},
                    clock=lambda: at("2026-10-24T02:00:00+00:00"))  # a Saturday
    assert budget.charge("flash", QUARTER) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="peak table"):
        budget.check_off_peak()


def test_check_off_peak_raises_peak_hours_which_stops_a_run_like_a_spent_cap(tmp_path):
    times = iter([at("2026-10-19T06:43:00+00:00"), at("2026-10-19T13:00:00+00:00")])
    budget = Budget(tmp_path / "b.sqlite", stage="M7", stage_cap_rmb=10.0, prices={"flash": FLASH},
                    off_peak=TABLE, clock=lambda: next(times))
    with pytest.raises(PeakHours, match="peak"):
        budget.check_off_peak()
    budget.check_off_peak()  # 21:00 in Beijing
    assert issubclass(PeakHours, BudgetExceeded)


def test_shipped_price_list_prices_v4_pro_and_its_peak_table():
    prices = load_prices()
    assert prices["deepseek-v4-pro"] == Price(input_cache_miss=9.0, output=27.0, input_cache_hit=0.3)
    # flash keeps the list price M4 to M6 reported against: its cache hits at the miss price
    assert prices["deepseek-flash"] == Price(input_cache_miss=2.0, output=8.0)
    assert load_off_peak() == TABLE

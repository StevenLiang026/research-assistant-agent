"""Spending ledger: prices token usage in RMB with an upper-bound price list and stops at the caps.

There is one cap per stage (milestone) and one over all stages. check() runs before every paid call;
charge() records a call and then checks again, so a run stops as soon as a cap is reached and overshoots
by at most one call.

cost_rmb is the list price: DeepSeek's peak-hour price, the one every report since M3 states costs at.
DeepSeek charges half of it outside its peak hours (the price page, read 2026-10-19: Beijing weekdays
09:00-12:00 and 14:00-18:00 are peak; Chinese public holidays are off-peak there but count as peak here, an
upper bound). A ledger given that table (off_peak, load_off_peak from the price list) books what DeepSeek
charges: the list price times the factor for a call that neither started nor ended in the peak hours, the
list price otherwise; one without it books the list price, as every ledger did before M7. check_off_peak
raises PeakHours, a BudgetExceeded, in the peak hours, for runs that pay only off peak.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

PRICES_FILE = Path(__file__).with_name("prices.json")


class BudgetExceeded(RuntimeError):
    pass


class PeakHours(BudgetExceeded):
    """A paid call refused in DeepSeek's peak hours by a run that pays only off peak: it stops the run as a
    spent cap does, and the run is resumed later."""


@dataclass(frozen=True)
class Price:
    """RMB per 1M tokens. A missing cache-hit price is charged at the cache-miss price."""

    input_cache_miss: float
    output: float
    input_cache_hit: float | None = None


@dataclass(frozen=True)
class OffPeak:
    """DeepSeek's time-of-day prices: on the weekdays (Monday is 0) of the clock at utc_offset_hours, inside
    the windows (minutes after midnight, start included, end excluded) the list price; at any other time the
    list price times factor."""

    factor: float
    utc_offset_hours: int
    weekdays: tuple[int, ...]
    windows: tuple[tuple[int, int], ...]


def load_prices(path: str | Path = PRICES_FILE) -> dict[str, Price]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {model: Price(**p) for model, p in data["models"].items()}


def _minutes(text: str) -> int:
    hours, minutes = text.split(":")
    return int(hours) * 60 + int(minutes)


def load_off_peak(path: str | Path = PRICES_FILE) -> OffPeak:
    """The price list's peak table ("off_peak")."""
    table = json.loads(Path(path).read_text(encoding="utf-8"))["off_peak"]
    return OffPeak(factor=float(table["factor"]), utc_offset_hours=int(table["utc_offset_hours"]),
                   weekdays=tuple(int(day) for day in table["peak_weekdays"]),
                   windows=tuple((_minutes(start), _minutes(end)) for start, end in table["peak_windows"]))


def is_peak(when: datetime, table: OffPeak) -> bool:
    """Whether when (a time with its time zone) falls in the table's peak hours."""
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError(f"{when!r} has no time zone: the peak hours are another time zone's")
    local = when.astimezone(timezone(timedelta(hours=table.utc_offset_hours)))
    minute = local.hour * 60 + local.minute
    return local.weekday() in table.weekdays and any(start <= minute < end for start, end in table.windows)


def cost_rmb(price: Price, usage: dict) -> float:
    """Usage as the API returns it; DeepSeek also splits prompt tokens into cache hits and misses."""
    hit, miss = usage.get("prompt_cache_hit_tokens"), usage.get("prompt_cache_miss_tokens")
    if hit is None or miss is None:
        hit, miss = 0, usage.get("prompt_tokens", 0)
    hit_price = price.input_cache_miss if price.input_cache_hit is None else price.input_cache_hit
    output = usage.get("completion_tokens", 0)
    return (hit * hit_price + miss * price.input_cache_miss + output * price.output) / 1e6


class Budget:
    def __init__(self, path: str | Path, *, stage: str, stage_cap_rmb: float, total_cap_rmb: float = 300.0,
                 prices: dict[str, Price] | None = None, off_peak: OffPeak | None = None,
                 clock: Callable[[], datetime] | None = None) -> None:
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS spend (ts TEXT NOT NULL, stage TEXT NOT NULL, "
                        "model TEXT NOT NULL, request_key TEXT, prompt_tokens INTEGER, "
                        "completion_tokens INTEGER, cost_rmb REAL NOT NULL)")
        self.db.commit()
        self.stage, self.stage_cap, self.total_cap = stage, stage_cap_rmb, total_cap_rmb
        self.prices = load_prices() if prices is None else prices
        self.off_peak = off_peak  # None: every call at the list price (module docstring)
        self.clock = clock or (lambda: datetime.now(UTC))

    def now(self) -> datetime:
        return self.clock()

    def spent(self, stage: str | None = None) -> float:
        if stage is None:
            row = self.db.execute("SELECT COALESCE(SUM(cost_rmb), 0) FROM spend").fetchone()
        else:
            row = self.db.execute("SELECT COALESCE(SUM(cost_rmb), 0) FROM spend WHERE stage = ?",
                                  (stage,)).fetchone()
        return float(row[0])

    def check(self) -> None:
        stage_spent, total_spent = self.spent(self.stage), self.spent()
        if stage_spent >= self.stage_cap:
            raise BudgetExceeded(f"stage {self.stage}: spent {stage_spent:.4f} of {self.stage_cap} RMB")
        if total_spent >= self.total_cap:
            raise BudgetExceeded(f"all stages: spent {total_spent:.4f} of {self.total_cap} RMB")

    def check_off_peak(self) -> None:
        """PeakHours when now is in the peak hours; a ValueError for a ledger without the peak table."""
        if self.off_peak is None:
            raise ValueError("paying only off peak needs the ledger's peak table (off_peak, load_off_peak)")
        when = self.now()
        if is_peak(when, self.off_peak):
            raise PeakHours(f"{when.isoformat()} is in DeepSeek's peak hours; this run pays only off peak")

    def charge(self, model: str, usage: dict, request_key: str | None = None,
               started: datetime | None = None) -> float:
        """Book a paid call: the list price, times the off-peak factor when the ledger has the peak table and
        the call neither started (started, when given) nor ended (now) in the peak hours."""
        if model not in self.prices:
            raise KeyError(f"no price for model {model!r}; add it to the price list first")
        cost = cost_rmb(self.prices[model], usage)
        ended = self.now()
        if self.off_peak is not None and not any(is_peak(t, self.off_peak) for t in (started, ended)
                                                 if t is not None):
            cost *= self.off_peak.factor
        self.db.execute("INSERT INTO spend VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (ended.astimezone(UTC).isoformat(), self.stage, model, request_key,
                         usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), cost))
        self.db.commit()
        self.check()
        return cost

    def close(self) -> None:
        self.db.close()

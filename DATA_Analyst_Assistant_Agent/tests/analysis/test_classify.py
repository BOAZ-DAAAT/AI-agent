from __future__ import annotations

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.analysis.nodes.classify import resolve_time_grain


def _frame(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"order_date": dates, "amount": range(len(dates))})


def test_one_month_span_resolves_to_daily() -> None:
    # "last month" style question: ~30 days -> daily, never forced to monthly.
    dates = [f"2026-06-{day:02d}" for day in range(1, 31)]
    grain, span = resolve_time_grain(_frame(dates), "order_date")
    assert grain == "D"
    assert span == 29


def test_two_year_span_resolves_to_monthly() -> None:
    dates = ["2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01", "2025-12-31"]
    grain, span = resolve_time_grain(_frame(dates), "order_date")
    assert grain == "M"
    assert span >= 700


def test_half_year_span_resolves_to_weekly() -> None:
    dates = ["2026-01-01", "2026-03-01", "2026-06-01"]
    grain, _ = resolve_time_grain(_frame(dates), "order_date")
    assert grain == "W"


def test_many_year_span_resolves_to_coarse_grain() -> None:
    dates = ["2018-01-01", "2026-01-01"]
    grain, _ = resolve_time_grain(_frame(dates), "order_date")
    assert grain == "Y"


def test_missing_or_unparseable_time_column_returns_none() -> None:
    assert resolve_time_grain(_frame(["2026-06-01"]), None) == (None, None)
    assert resolve_time_grain(_frame(["2026-06-01"]), "nope") == (None, None)
    bad = pd.DataFrame({"order_date": ["n/a", "unknown"], "amount": [1, 2]})
    assert resolve_time_grain(bad, "order_date") == (None, None)

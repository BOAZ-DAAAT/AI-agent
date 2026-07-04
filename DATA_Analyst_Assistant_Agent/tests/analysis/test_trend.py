from __future__ import annotations

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.analysis.tools.trend import analyze_time_series


def test_analyze_time_series_daily_avoids_resample_for_daily_grain(monkeypatch) -> None:
    original_resample = pd.Series.resample

    def fail_resample(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("daily frequency should not call pandas resample")

    monkeypatch.setattr(pd.Series, "resample", fail_resample)
    try:
        records = [
            {"sales_date": "2017-06-01", "daily_revenue": 10.0},
            {"sales_date": "2017-06-02", "daily_revenue": 12.0},
            {"sales_date": "2017-06-03", "daily_revenue": 14.0},
            {"sales_date": "2017-06-04", "daily_revenue": 16.0},
        ]
        result = analyze_time_series.invoke(
            {
                "records": records,
                "metric": "daily_revenue",
                "time_column": "sales_date",
                "frequency": "D",
                "aggregation": "sum",
            }
        )
    finally:
        monkeypatch.setattr(pd.Series, "resample", original_resample)

    assert result["period_count"] == 4
    assert result["frequency"] == "D"
    assert [item["value"] for item in result["series"]] == [10.0, 12.0, 14.0, 16.0]

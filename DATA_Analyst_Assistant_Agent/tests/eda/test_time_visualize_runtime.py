from __future__ import annotations

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda.lib.time_skill import run_time_skill


def test_run_time_skill_handles_string_month_buckets_without_runtime_crash() -> None:
    months = pd.date_range("2017-01-01", periods=23, freq="MS").strftime("%Y-%m-%d").tolist()
    df = pd.DataFrame(
        {
            "seller_id": [f"s{i % 20}" for i in range(1000)],
            "purchase_month": [months[i % len(months)] for i in range(1000)],
            "delivery_days": [(i % 40) + 1 for i in range(1000)],
            "review_score_at_order": [float((i % 5) + 1) for i in range(1000)],
        }
    )

    result = run_time_skill(
        df,
        measure_cols=["delivery_days", "review_score_at_order"],
        time_cols=["purchase_month"],
        key_col="seller_id",
    )

    assert len(result["timeseries"]["chart_paths"]) == 2
    assert len(result["seasonality"]["chart_paths"]) == 4
    assert len(result["multiline"]["chart_paths"]) == 2

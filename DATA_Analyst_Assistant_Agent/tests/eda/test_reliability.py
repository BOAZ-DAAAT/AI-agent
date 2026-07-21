from __future__ import annotations

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda.lib.reliability import assess_sample_reliability


def test_assess_sample_reliability_ignores_non_numeric_count_col() -> None:
    # count_col이 datetime 컬럼으로 잘못 지정돼도 크래시 없이 판정을 보류해야 한다
    # (Invalid comparison between dtype=datetime64[ns] and int 회귀 방지).
    df = pd.DataFrame(
        {
            "seller_id": ["s1", "s2", "s1", "s2"],
            "review_creation_date": pd.to_datetime(
                ["2018-01-05", "2018-01-06", "2018-01-07", "2018-01-08"]
            ),
        }
    )

    result = assess_sample_reliability(df, key_col="seller_id", count_col="review_creation_date")

    assert result["basis"] == ""
    assert result["low_n_groups"] == []
    assert result["total_groups"] == 0


def test_assess_sample_reliability_uses_numeric_count_col_per_row() -> None:
    df = pd.DataFrame(
        {
            "seller_id": ["s1", "s2"],
            "order_count": [5, 40],
        }
    )

    result = assess_sample_reliability(df, count_col="order_count", min_n=30)

    assert result["basis"] == "order_count(행별)"
    assert result["low_n_count"] == 1

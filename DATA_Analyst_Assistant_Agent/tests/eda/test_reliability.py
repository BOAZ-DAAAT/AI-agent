import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda.lib.reliability import assess_sample_reliability


def test_assess_sample_reliability_falls_back_to_group_size_for_string_count_col() -> None:
    df = pd.DataFrame(
        {
            "purchase_date": ["2017-06-01", "2017-06-01", "2017-06-02"],
            "order_id": ["o1", "o2", "o3"],
        }
    )

    result = assess_sample_reliability(
        df,
        key_col="purchase_date",
        count_col="order_id",
        data_level="raw",
        min_n=2,
    )

    assert result["basis"] == "그룹 행수"
    assert result["low_n_count"] == 1
    assert result["low_n_groups"][0]["group"] == "2017-06-02"
    assert result["low_n_groups"][0]["n"] == 1


def test_assess_sample_reliability_uses_numeric_count_col_when_available() -> None:
    df = pd.DataFrame(
        {
            "purchase_date": ["2017-06-01", "2017-06-01", "2017-06-02"],
            "item_count": [2, 3, 1],
        }
    )

    result = assess_sample_reliability(
        df,
        key_col="purchase_date",
        count_col="item_count",
        data_level="raw",
        min_n=2,
    )

    assert result["basis"] == "item_count 합(그룹별)"
    assert result["low_n_count"] == 1
    assert result["low_n_groups"][0]["group"] == "2017-06-02"
    assert result["low_n_groups"][0]["n"] == 1

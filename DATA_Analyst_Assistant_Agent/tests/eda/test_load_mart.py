from __future__ import annotations

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda._runtime import EdaContext, reset_context, set_context
from DATA_Analyst_Assistant_Agent.agents.eda.nodes.load import load_mart_node


def teardown_function() -> None:
    reset_context()


def test_load_mart_uses_deterministic_column_classification_for_rfm_feature_mart() -> None:
    df = pd.DataFrame(
        {
            "customer_unique_id": ["a" * 32, "b" * 32, "c" * 32],
            "analysis_ref_ts": ["2018-10-17 17:30:18"] * 3,
            "window_start_ts": ["2017-10-17 17:30:18"] * 3,
            "last_purchase_ts_12m": [
                "2018-08-02 14:26:38",
                "2018-07-23 17:21:43",
                "2018-08-04 14:40:31",
            ],
            "recency_days": [76, 86, 74],
            "frequency_12m_orders": [2, 2, 2],
            "monetary_12m_value": [256.2, 255.47, 254.27],
            "avg_review_score_12m": [5.0, 5.0, 4.5],
            "segment": ["HighRFM_NotLowReview"] * 3,
        }
    )
    set_context(EdaContext(df=df, question_type="mart"))

    result = load_mart_node({"question_type": "mart", "mart_design": {}})

    assert result["time_columns"] == []
    assert result["has_time_column"] is False
    assert result["time_detection_status"] == "no_usable_time_columns"
    assert result["count_column"] == "frequency_12m_orders"


def test_load_mart_recognizes_timestamp_named_columns() -> None:
    df = pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(8)],
            "order_purchase_timestamp": [
                "2018-01-01 10:00:00",
                "2018-01-01 11:00:00",
                "2018-01-02 10:00:00",
                "2018-01-02 11:00:00",
                "2018-01-03 10:00:00",
                "2018-01-03 11:00:00",
                "2018-01-04 10:00:00",
                "2018-01-04 11:00:00",
            ],
            "sales_amount": [10.0, 12.0, 9.0, 11.0, 15.0, 14.0, 13.0, 16.0],
        }
    )
    set_context(EdaContext(df=df, question_type="time"))

    result = load_mart_node({"question_type": "time", "mart_design": {}})

    assert result["time_columns"] == ["order_purchase_timestamp"]
    assert result["has_time_column"] is True
    assert result["time_detection_status"] == "ok"


def test_load_mart_rejects_timestamp_columns_with_too_few_time_buckets() -> None:
    df = pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(6)],
            "order_purchase_timestamp": [
                "2018-01-01 10:00:00",
                "2018-01-01 11:00:00",
                "2018-01-01 12:00:00",
                "2018-01-02 10:00:00",
                "2018-01-02 11:00:00",
                "2018-01-02 12:00:00",
            ],
            "sales_amount": [10.0, 12.0, 9.0, 11.0, 15.0, 14.0],
        }
    )
    set_context(EdaContext(df=df, question_type="time"))

    result = load_mart_node({"question_type": "time", "mart_design": {}})

    assert result["time_columns"] == []
    assert result["has_time_column"] is False
    assert result["time_detection_status"] == "no_usable_time_columns"


def test_load_mart_never_selects_datetime_column_as_count_column() -> None:
    # review_creation_date는 "n_" 마커를 우연히 포함하는 datetime 컬럼 — 실제 라이브 크래시 재현
    # (Invalid comparison between dtype=datetime64[ns] and int).
    df = pd.DataFrame(
        {
            "seller_id": ["s1", "s2", "s1", "s2"],
            "order_purchase_timestamp": pd.to_datetime(
                ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"]
            ),
            "review_creation_date": pd.to_datetime(
                ["2018-01-05", "2018-01-06", "2018-01-07", "2018-01-08"]
            ),
            "delivery_days": [3, 5, 2, 8],
            "review_score": [4, 5, 3, 2],
        }
    )
    set_context(EdaContext(df=df, question_type="mart"))

    result = load_mart_node({"question_type": "mart", "mart_design": {}})

    assert result["count_column"] != "review_creation_date"


def test_load_mart_continues_without_time_columns_when_detection_errors(monkeypatch) -> None:
    df = pd.DataFrame(
        {
            "order_id": ["o1", "o2", "o3"],
            "created_at": ["2018-01-01", "2018-01-02", "2018-01-03"],
            "sales_amount": [10.0, 12.0, 9.0],
        }
    )
    set_context(EdaContext(df=df, question_type="time"))

    def boom(df: pd.DataFrame, candidate_cols: list[str]) -> list[str]:
        raise ValueError("bad datetime parse")

    monkeypatch.setattr(
        "DATA_Analyst_Assistant_Agent.agents.eda.nodes.load.usable_time_columns",
        boom,
    )

    result = load_mart_node({"question_type": "time", "mart_design": {}})

    assert result["time_columns"] == []
    assert result["has_time_column"] is False
    assert result["time_detection_status"] == "skipped_due_error"
    assert result["time_skip_reason"] == "bad datetime parse"

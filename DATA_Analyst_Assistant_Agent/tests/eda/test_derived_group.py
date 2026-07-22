from __future__ import annotations

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda.lib.derived_group import (
    build_derived_group_frame,
    compute_derived_group_comparison,
)


def test_derived_group_comparison_runs_only_for_branch_questions() -> None:
    df = pd.DataFrame({
        "seller_id": ["s1", "s2"],
        "order_id": ["o1", "o2"],
        "delivery_days": [20, 5],
        "review_score": [2, 5],
    })

    result = compute_derived_group_comparison(
        df,
        "seller_id 기준으로 배송 소요일이 긴 판매자군을 비교해줘",
    )

    assert result is None


def test_derived_group_comparison_builds_entity_level_summary() -> None:
    rows = []
    for idx in range(10):
        rows.append({
            "seller_id": "slow",
            "order_id": f"slow-{idx}",
            "delivery_days": 20,
            "review_score": 2,
        })
        rows.append({
            "seller_id": "fast",
            "order_id": f"fast-{idx}",
            "delivery_days": 5,
            "review_score": 5,
        })
    for idx in range(5):
        rows.append({
            "seller_id": "small",
            "order_id": f"small-{idx}",
            "delivery_days": 30,
            "review_score": 1,
        })
    df = pd.DataFrame(rows)

    result = compute_derived_group_comparison(
        df,
        "[추가 지시사항] seller_id별 주문 수 10건 이상 판매자만 남긴 뒤 "
        "평균 배송 소요일이 긴 판매자군에서 리뷰 점수가 더 낮게 나타나는지 탐색해줘",
    )

    assert result is not None
    assert result["entity_col"] == "seller_id"
    assert result["observation_col"] == "order_id"
    assert result["metric_col"] == "delivery_days"
    assert result["target_col"] == "review_score"
    assert result["min_count"] == 10
    assert result["eligible_entities"] == 2
    assert result["excluded_entities"] == 1
    assert result["relationships"]["eligible_entities"]["spearman"] == -1.0
    assert result["high_metric_group"]["target_mean_gap_high_minus_rest"] == -3.0
    assert result["findings"]


def test_derived_group_frame_can_be_used_as_eda_graph_input() -> None:
    rows = []
    for idx in range(10):
        rows.append({
            "seller_id": "slow",
            "order_id": f"slow-{idx}",
            "delivery_days": 20,
            "review_score": 2,
        })
        rows.append({
            "seller_id": "fast",
            "order_id": f"fast-{idx}",
            "delivery_days": 5,
            "review_score": 5,
        })
    df = pd.DataFrame(rows)

    prepared = build_derived_group_frame(
        df,
        "[추가 지시사항] seller_id별 주문 수 10건 이상 판매자만 남긴 뒤 "
        "평균 배송 소요일이 긴 판매자군에서 리뷰 점수가 더 낮게 나타나는지 탐색해줘",
    )

    assert prepared is not None
    graph_df = prepared["dataframe"]
    assert list(graph_df.columns) == [
        "seller_id",
        "observation_count",
        "avg_delivery_days",
        "avg_review_score",
        "delivery_days_group",
    ]
    assert prepared["mart_design"]["dimension_columns"] == ["delivery_days_group"]
    assert prepared["measure_cols"] == [
        "observation_count",
        "avg_delivery_days",
        "avg_review_score",
    ]
    assert prepared["target_col"] == "avg_review_score"

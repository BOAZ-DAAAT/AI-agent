from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from DATA_Analyst_Assistant_Agent.agents.eda.lib.derived_group import (
    build_derived_group_frame,
    compute_derived_group_comparison,
)


class _FakeLLM:
    """단일 고정 응답만 흉내낸다(호출마다 같은 content를 반환)."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        return SimpleNamespace(content=self._content)


class _QueueLLM:
    """호출 순서대로 다른 응답을 돌려준다(분류 호출 → 표현식 호출 2단계를 흉내낼 때 필요, 2026-07-24).

    분기 지시사항 처리가 (1) entity_comparison/row_filter/unrelated 분류 호출 →
    (2, entity_comparison일 때만) 집계 컬럼 기준 필터 표현식 호출, 2단계로 나뉘면서
    _FakeLLM 하나로는 두 호출에 서로 다른 응답을 줄 수 없어 도입했다.
    """

    def __init__(self, contents: list[str]) -> None:
        self._contents = list(contents)
        self.calls = 0

    def invoke(self, prompt: str):
        self.calls += 1
        content = self._contents[min(self.calls, len(self._contents)) - 1]
        return SimpleNamespace(content=content)


_NOT_A_FILTER_LLM = _QueueLLM(['{"request_type": "unrelated", "filter_expression": null}'])


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

    llm = _QueueLLM([
        '{"request_type": "entity_comparison", "filter_expression": null}',
        '{"is_filter_request": true, "filter_expression": "df[\\"observation_count\\"] >= 10"}',
    ])
    result = compute_derived_group_comparison(
        df,
        "[추가 지시사항] seller_id별 주문 수 10건 이상 판매자만 남긴 뒤 "
        "평균 배송 소요일이 긴 판매자군에서 리뷰 점수가 더 낮게 나타나는지 탐색해줘",
        llm=llm,
    )

    assert result is not None
    assert result["entity_col"] == "seller_id"
    assert result["observation_col"] == "order_id"
    assert result["metric_col"] == "delivery_days"
    assert result["target_col"] == "review_score"
    assert result["filter_expression"] == 'df["observation_count"] >= 10'
    assert result["eligible_entities"] == 2
    assert result["excluded_entities"] == 1
    assert result["relationships"]["eligible_entities"]["spearman"] == -1.0
    assert result["findings"]


def test_derived_group_comparison_returns_none_when_llm_says_not_a_filter() -> None:
    """차트 요청처럼 조건 필터링과 무관한 지시사항이면 이 메커니즘 자체가 개입하지 않는다."""
    df = pd.DataFrame({
        "seller_id": ["s1", "s2"],
        "order_id": ["o1", "o2"],
        "delivery_days": [20, 5],
        "review_score": [2, 5],
    })

    result = compute_derived_group_comparison(
        df,
        "[추가 지시사항] 나 다른 차트 생성해서 분석 이어가고 싶어",
        llm=_NOT_A_FILTER_LLM,
    )

    assert result is None


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

    llm = _QueueLLM([
        '{"request_type": "entity_comparison", "filter_expression": null}',
        '{"is_filter_request": true, "filter_expression": "df[\\"observation_count\\"] >= 10"}',
    ])
    prepared = build_derived_group_frame(
        df,
        "[추가 지시사항] seller_id별 주문 수 10건 이상 판매자만 남긴 뒤 "
        "평균 배송 소요일이 긴 판매자군에서 리뷰 점수가 더 낮게 나타나는지 탐색해줘",
        llm=llm,
    )

    assert prepared is not None
    graph_df = prepared["dataframe"]
    assert list(graph_df.columns) == [
        "seller_id",
        "observation_count",
        "avg_delivery_days",
        "avg_review_score",
    ]
    assert prepared["mart_design"]["dimension_columns"] == ["seller_id"]
    assert prepared["measure_cols"] == [
        "observation_count",
        "avg_delivery_days",
        "avg_review_score",
    ]
    assert prepared["target_col"] == "avg_review_score"


def test_row_filter_keeps_original_grain_without_entity_column() -> None:
    """실사용 버그 재현(2026-07-24): 마트에 order_id 외 엔티티 ID 컬럼이 아예 없어도
    "이상치 제거" 같은 단순 행 필터 요청은 개체 집계를 거치지 않고 원본 그레인 그대로
    처리돼야 한다(run_ea521dbf5eaa48df8f29018362bffed2에서 entity_col을 못 찾아
    필터 자체가 조용히 스킵됐던 문제)."""
    df = pd.DataFrame({
        "order_id": [f"o{i}" for i in range(12)],
        "delivery_delay_days": [1, 2, 3, 2, 1, 3, 2, 1, 2, 3, 100, -80],
        "review_score": [5, 4, 3, 4, 5, 3, 4, 5, 4, 3, 1, 1],
    })

    llm = _QueueLLM([
        (
            '{"request_type": "row_filter", '
            '"filter_expression": "df[\\"delivery_delay_days\\"].between(-10, 10)"}'
        ),
    ])
    prepared = build_derived_group_frame(
        df,
        "[추가 지시사항] 이상치를 제거한 상태에서도 관계가 유지되는지 보고 싶어",
        llm=llm,
    )

    assert prepared is not None
    graph_df = prepared["dataframe"]
    # 원본 스키마·그레인 그대로 유지 — 집계 컬럼(avg_*, observation_count)이 생기지 않는다.
    assert list(graph_df.columns) == ["order_id", "delivery_delay_days", "review_score"]
    assert len(graph_df) == 10  # 100, -80 두 행만 제외
    assert prepared["metadata"]["kind"] == "row_filter"
    assert prepared["metadata"]["total_rows"] == 12
    assert prepared["metadata"]["eligible_rows"] == 10
    assert prepared["key_col"] is None
    assert prepared["measure_cols"] is None


def test_row_filter_returns_none_when_expression_fails_gate() -> None:
    df = pd.DataFrame({"order_id": ["o1", "o2"], "amount": [10, 20]})
    llm = _QueueLLM([
        '{"request_type": "row_filter", "filter_expression": "df[\\"not_a_real_column\\"] > 0"}',
    ])

    result = build_derived_group_frame(
        df,
        "[추가 지시사항] 이상한 컬럼으로 필터링해줘",
        llm=llm,
    )

    assert result is None

from __future__ import annotations

from DATA_Analyst_Assistant_Agent.agents.sql.validation_contract import validate_datamart_reusability


def _plan() -> dict:
    return {
        "validation_contract": {
            "expected_result_shape": "datamart_creation",
            "mart_policy": "prefer_row_preserving",
        }
    }


def test_groupby_datamart_without_justification_is_flagged() -> None:
    findings = validate_datamart_reusability(
        _plan(),
        {
            "grain": "월별 1행",
            "base_grain": "월별 1행",
            "aggregation_policy": "prefer_row_preserving",
            "aggregation_rationale": None,
            "design_reasoning": "월별 매출을 저장합니다.",
        },
        {
            "sql": "CREATE TABLE analytics.monthly_sales AS SELECT month, SUM(revenue) AS revenue FROM orders GROUP BY month;",
            "reasoning": "월별 요약 결과를 저장합니다.",
        },
    )

    assert any(item["category"] == "mart_summary_bias" for item in findings)


def test_groupby_datamart_with_explicit_justification_is_allowed() -> None:
    findings = validate_datamart_reusability(
        _plan(),
        {
            "grain": "고객별 latest 상태 1행",
            "base_grain": "원본 이벤트 행 수준 유지가 불가능한 최신 상태 스냅샷",
            "aggregation_policy": "aggregate_if_justified",
            "aggregation_rationale": "원본 행 수준을 유지하면 중복 상태가 남아 고객 최신 상태 마트 목적에 맞지 않습니다.",
            "design_reasoning": "원본 행 수준 대신 고객 최신 상태 스냅샷이 필요하므로 예외적으로 집계합니다.",
        },
        {
            "sql": "CREATE TABLE analytics.customer_latest AS SELECT customer_id, MAX(updated_at) AS latest_updated_at FROM customer_events GROUP BY customer_id;",
            "reasoning": "원본 행 수준을 유지할 수 없는 예외적인 스냅샷 마트입니다.",
        },
    )

    assert findings == []


def test_row_preserving_datamart_is_allowed() -> None:
    findings = validate_datamart_reusability(
        _plan(),
        {
            "grain": "주문 1건당 1행",
            "base_grain": "주문 1건당 1행",
            "aggregation_policy": "prefer_row_preserving",
            "aggregation_rationale": None,
            "design_reasoning": "주문 상세를 조인/정제한 재사용 가능한 기반 테이블입니다.",
        },
        {
            "sql": (
                "CREATE TABLE analytics.order_enriched AS "
                "SELECT o.order_id, o.customer_id, o.order_date, oi.product_id "
                "FROM orders o JOIN order_items oi ON o.order_id = oi.order_id;"
            ),
            "reasoning": "원본 행 수준을 최대한 유지했습니다.",
        },
    )

    assert findings == []

from __future__ import annotations

import json

from DATA_Analyst_Assistant_Agent.agents.sql.validation_contract import (
    validate_datamart_reusability,
    validate_sql_identifiers,
    validate_sql_intent,
)


def _schema_text(**tables: list[str]) -> str:
    return json.dumps({"tables": {name: {"columns": cols} for name, cols in tables.items()}}, ensure_ascii=False)


# ── TRIM/EXTRACT/SUBSTRING의 FROM은 테이블 참조가 아니다 (오탐 재현: TRIM(BOTH x FROM col)) ──


def test_trim_from_is_not_mistaken_for_a_table_reference() -> None:
    plan = {"route_kind": "simple"}
    sql_draft = {
        "sql": "SELECT TRIM(BOTH '\\r' FROM t.product_category_name_english) AS name FROM translation t",
        "source_tables": ["translation"],
    }
    schema_text = _schema_text(translation=["product_category_name_english"])

    findings = validate_sql_identifiers(plan, sql_draft, schema_text)

    assert findings == []


def test_extract_from_is_not_mistaken_for_a_table_reference() -> None:
    plan = {"route_kind": "simple"}
    sql_draft = {
        "sql": "SELECT EXTRACT(YEAR FROM o.order_purchase_timestamp) AS yr FROM orders o",
        "source_tables": ["orders"],
    }
    schema_text = _schema_text(orders=["order_purchase_timestamp"])

    findings = validate_sql_identifiers(plan, sql_draft, schema_text)

    assert findings == []


def test_genuine_missing_table_reference_is_still_caught() -> None:
    plan = {"route_kind": "simple"}
    sql_draft = {
        "sql": "SELECT * FROM orders o JOIN not_a_real_table x ON o.id = x.id",
        "source_tables": ["orders"],
    }
    schema_text = _schema_text(orders=["id"])

    findings = validate_sql_identifiers(plan, sql_draft, schema_text)

    assert any(f["category"] == "missing_table" and "not_a_real_table" in f["detail"] for f in findings)


# ── 서브쿼리 사전집계(fan-out 방지 JOIN)를 마트 전체 요약으로 오인하지 않는다 ──


def test_subquery_preaggregation_for_join_is_not_flagged_as_mart_summary() -> None:
    plan = {"route_kind": "comprehensive"}
    mart_design = {"aggregation_policy": "preserve_common_grain", "grain": "order_id"}
    sql_draft = {
        "sql": (
            "CREATE TABLE analytics.mart AS "
            "SELECT o.order_id, oi.total_amount FROM orders o "
            "LEFT JOIN (SELECT order_id, SUM(price) AS total_amount FROM order_items GROUP BY order_id) oi "
            "ON o.order_id = oi.order_id"
        ),
    }

    findings = validate_datamart_reusability(plan, mart_design, sql_draft)

    assert findings == []


def test_top_level_group_by_still_flagged_when_policy_preserves_grain() -> None:
    plan = {"route_kind": "comprehensive"}
    mart_design = {"aggregation_policy": "preserve_common_grain", "grain": "customer_unique_id"}
    sql_draft = {
        "sql": "CREATE TABLE analytics.mart AS SELECT customer_unique_id, SUM(payment_value) FROM orders GROUP BY customer_unique_id",
    }

    findings = validate_datamart_reusability(plan, mart_design, sql_draft)

    assert any(f["category"] == "mart_policy_mismatch" for f in findings)


# ── required_aggregations 계약이 comprehensive(datamart_creation)에도 적용된다 (RFM류 파생값 누락) ──


def test_missing_required_aggregation_is_caught_for_comprehensive_route() -> None:
    plan = {
        "route_kind": "comprehensive",
        "required_aggregations": ["SUM", "COUNT", "MAX"],
    }
    sql_draft = {
        # 고객 단위로 grain을 접지 않고 주문 단위만 그대로 내보내는 SQL — RFM 실패 재현
        "sql": "CREATE TABLE analytics.mart AS SELECT order_id, customer_unique_id, payment_value FROM orders",
        "source_tables": [],
    }

    findings = validate_sql_intent(plan, sql_draft)

    categories = {f["category"] for f in findings}
    assert "intent_mismatch" in categories


def test_present_required_aggregations_pass_for_comprehensive_route() -> None:
    plan = {
        "route_kind": "comprehensive",
        "required_aggregations": ["SUM", "COUNT", "MAX"],
    }
    sql_draft = {
        "sql": (
            "CREATE TABLE analytics.mart AS SELECT customer_unique_id, "
            "MAX(order_purchase_timestamp) AS last_order_at, "
            "COUNT(order_id) AS frequency, "
            "SUM(payment_value) AS monetary "
            "FROM orders GROUP BY customer_unique_id"
        ),
        "source_tables": [],
    }

    findings = validate_sql_intent(plan, sql_draft)

    assert not any(f["category"] == "intent_mismatch" for f in findings)

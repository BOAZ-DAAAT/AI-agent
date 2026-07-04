from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import default_validation_contract
from DATA_Analyst_Assistant_Agent.agents.sql.validation_contract import validate_sql_intent


def test_simple_trend_query_defaults_to_table_preview_not_grouped_aggregate() -> None:
    contract = default_validation_contract(
        question="2017년 6월의 일별 매출 추이를 분석해줘",
        route_kind="simple",
        target_metric="매출",
        dimensions=["일"],
        selected_tables=["orders", "order_items"],
    )

    assert contract["expected_result_shape"] == "table_preview"
    assert contract["required_aggregations"] == []


def test_simple_query_with_unnecessary_group_by_is_invalid() -> None:
    plan = {
        "route_kind": "simple",
        "task_type": "query_answer",
        "dimensions": ["일"],
        "selected_join_tables": ["orders", "order_items"],
        "relevant_tables": ["orders", "order_items"],
        "validation_contract": {
            "expected_result_shape": "table_preview",
            "required_aggregations": [],
            "required_tables": ["orders", "order_items"],
            "dimensions": ["일"],
        },
    }
    sql_draft = {
        "sql": (
            "SELECT DATE(o.order_purchase_timestamp) AS sales_date, "
            "SUM(oi.price) AS daily_revenue "
            "FROM orders o JOIN order_items oi ON o.order_id = oi.order_id "
            "GROUP BY sales_date"
        ),
        "source_tables": ["orders", "order_items"],
        "columns_used": ["order_purchase_timestamp", "price"],
    }

    findings = validate_sql_intent(plan, sql_draft)

    assert any(item["category"] == "simple_summary_bias" for item in findings)

"""Olist 템플릿 매칭과 결정론적 SQL 계약 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from DATA_Analyst_Assistant_Agent.agents.sql.olist_templates import (
    OlistTemplateId,
    build_olist_sql_draft,
    build_olist_validation_plan,
    match_olist_template,
)
from DATA_Analyst_Assistant_Agent.agents.sql.validation_contract import (
    validate_sql_dialect_and_route,
    validate_sql_identifiers,
    validate_sql_intent,
)
from DATA_Analyst_Assistant_Agent.agents.sql.graph import build_app, route_after_validation
from DATA_Analyst_Assistant_Agent.agents.sql.nodes.prevalidate import route_after_prevalidation
from DATA_Analyst_Assistant_Agent.shared.contracts import AnalysisPlan


@pytest.fixture(scope="module")
def olist_catalog() -> dict:
    path = Path(__file__).parents[2] / "agents" / "sql" / "data" / "db_schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("template_id", "question"),
    [
        (OlistTemplateId.monthly_sales_orders, "월별 매출과 주문 수를 보여줘"),
        (OlistTemplateId.monthly_sales_orders, "월간 상품 판매액 및 주문 건수 추이"),
        (OlistTemplateId.monthly_sales_orders, "month 기준 sales와 order count"),
        (OlistTemplateId.order_status_distribution, "주문 상태별 주문 분포를 알려줘"),
        (OlistTemplateId.order_status_distribution, "주문 상태 구성 비율"),
        (OlistTemplateId.order_status_distribution, "order status distribution"),
        (OlistTemplateId.category_sales, "카테고리별 매출을 보여줘"),
        (OlistTemplateId.category_sales, "상품군별 판매 현황"),
        (OlistTemplateId.category_sales, "category sales summary"),
        (OlistTemplateId.review_score_distribution, "리뷰 평점 분포를 보여줘"),
        (OlistTemplateId.review_score_distribution, "별점별 리뷰 건수와 비율"),
        (OlistTemplateId.review_score_distribution, "review score distribution"),
        (OlistTemplateId.payment_method_summary, "결제 수단별 요약"),
        (OlistTemplateId.payment_method_summary, "결제 방식별 금액과 주문 수"),
        (OlistTemplateId.payment_method_summary, "payment type amount summary"),
    ],
)
def test_match_olist_template_variants(
    olist_catalog: dict,
    template_id: OlistTemplateId,
    question: str,
) -> None:
    match = match_olist_template(question, olist_catalog)

    assert match.supported is True
    assert match.template_id == template_id


@pytest.mark.parametrize(
    "question",
    [
        "2024년 월별 매출과 주문 수를 보여줘",
        "최근 3개월 주문 상태별 분포",
        "카테고리 매출 상위 10개",
        "카테고리별 매출과 주문 상태 분포를 함께 보여줘",
    ],
)
def test_unsupported_or_ambiguous_question_uses_semantic_fallback(
    olist_catalog: dict,
    question: str,
) -> None:
    assert match_olist_template(question, olist_catalog).supported is False


def test_template_is_disabled_when_required_column_is_missing(olist_catalog: dict) -> None:
    catalog = json.loads(json.dumps(olist_catalog))
    catalog["orders"]["columns"] = [
        column
        for column in catalog["orders"]["columns"]
        if column["name"] != "order_purchase_timestamp"
    ]

    match = match_olist_template("월별 매출과 주문 수", catalog)

    assert match.supported is False
    assert "orders.order_purchase_timestamp" in match.reason


@pytest.mark.parametrize("legacy_source", ["llm", "repair"])
def test_analysis_plan_reads_legacy_generation_sources(legacy_source: str) -> None:
    plan = AnalysisPlan(goal="이전 계획", sql_generation_source=legacy_source)  # type: ignore[arg-type]

    assert plan.sql_generation_source == "semantic_llm"


@pytest.mark.parametrize("template_id", list(OlistTemplateId))
def test_olist_sql_drafts_pass_existing_prevalidation_contracts(
    olist_catalog: dict,
    template_id: OlistTemplateId,
) -> None:
    draft = build_olist_sql_draft(template_id, olist_catalog).model_dump()
    plan = build_olist_validation_plan(template_id)
    schema_text = json.dumps(olist_catalog, ensure_ascii=False)

    findings = [
        *validate_sql_dialect_and_route(plan, draft),
        *validate_sql_intent(plan, draft),
        *validate_sql_identifiers(plan, draft, schema_text),
    ]

    assert findings == []
    assert draft["sql_type"] == "select"
    assert draft["sql"].lstrip().upper().startswith(("SELECT", "WITH"))


def test_review_distribution_keeps_group_by_in_top_level_select(olist_catalog: dict) -> None:
    template_id = OlistTemplateId.review_score_distribution
    draft = build_olist_sql_draft(template_id, olist_catalog).model_dump()
    plan = build_olist_validation_plan(template_id)

    findings = validate_sql_intent(plan, draft)

    assert not any("GROUP BY가 없습니다" in item.get("detail", "") for item in findings)
    assert "GROUP BY s.review_score" in draft["sql"]


def test_template_graph_skips_semantic_planning_and_sql_generation_llm(
    monkeypatch: pytest.MonkeyPatch,
    olist_catalog: dict,
) -> None:
    import DATA_Analyst_Assistant_Agent.agents.sql.nodes as nodes

    def forbidden_node(_state: dict) -> dict:
        raise AssertionError("템플릿 경로에서 semantic planning/generation을 호출하면 안 됩니다.")

    def fake_execute(_state: dict) -> dict:
        rows = [{"purchase_month": "2017-01-01", "product_sales": 10.0, "order_count": 1}]
        return {
            "sql_result": rows,
            "statement_results": [{"index": 0, "rows": rows, "row_count": 1}],
            "row_count": 1,
            "error": "",
            "execution_error_info": {},
        }

    monkeypatch.setattr(nodes, "plan_question", forbidden_node)
    monkeypatch.setattr(nodes, "generate_sql", forbidden_node)
    monkeypatch.setattr(nodes, "execute_sql", fake_execute)
    payload = {
        "user_question": "월별 매출과 주문 수를 보여줘",
        "required_db_schema": json.dumps(olist_catalog, ensure_ascii=False),
        "schema_text": "",
        "integrity_text": "",
        "integrity_dataset_name": "olist",
        "integrity_preplan": {},
        "integrity_refresh": {},
        "schema_refresh": {},
        "question_plan": {},
        "final_table_plan": {},
        "planning_stages": {},
        "plan": {},
        "mart_design": {},
        "sql_draft": {},
        "previous_sql_draft": {},
        "sql_result": None,
        "statement_results": [],
        "row_count": 0,
        "precheck_result": None,
        "postcheck_result": None,
        "mart_quality_result": {},
        "validation": {},
        "validation_findings": [],
        "retry_hint": {},
        "validation_summary": {},
        "retry_count": 0,
        "max_retries": 1,
        "feedback": "",
        "error": "",
        "generation_source": "olist_template",
        "sql_generation_source": "olist_template",
        "sql_template_id": "monthly_sales_orders",
        "generation_failure_reason": "",
        "generation_context_diagnostics": [],
        "failed_statement_index": None,
        "failed_statement_sql": "",
        "failed_sql_component": None,
        "execution_error_info": {},
        "classification": "none",
        "repair_strategy": "none",
        "repair_attempted": False,
        "repair_validation_result": {},
        "final_answer": "",
    }

    result = build_app().invoke(payload)

    assert result["validation"]["result"] == "valid"
    assert result["sql_generation_source"] == "olist_template"
    assert result["sql_template_id"] == "monthly_sales_orders"
    assert result["retry_count"] == 0


def test_semantic_prevalidation_failure_skips_execution_and_retries_only_once() -> None:
    state = {
        "validation": {"result": "invalid"},
        "retry_hint": {"retryable": True, "reason_code": "mysql_dialect_error"},
        "retry_count": 0,
        "max_retries": 1,
        "sql_template_id": None,
    }

    assert route_after_prevalidation(state) == "validate"  # type: ignore[arg-type]
    assert route_after_validation(state) == "retry"  # type: ignore[arg-type]
    assert route_after_validation({**state, "retry_count": 1}) == "finalize"  # type: ignore[arg-type]

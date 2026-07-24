"""Olist 25개 하이브리드 템플릿의 레지스트리·매칭·파라미터 계약 테스트."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from DATA_Analyst_Assistant_Agent.agents.sql.olist_templates import (
    OLIST_TEMPLATE_REGISTRY,
    OlistTemplateId,
    build_olist_sql_draft,
    match_olist_template,
)
from DATA_Analyst_Assistant_Agent.shared.contracts import OlistTemplateKind


@pytest.fixture(scope="module")
def olist_catalog() -> dict:
    path = Path(__file__).parents[2] / "agents" / "sql" / "data" / "db_schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


TEMPLATE_QUESTIONS = [
    (OlistTemplateId.monthly_sales_orders, "월별 매출과 주문 수", "monthly sales and order count"),
    (OlistTemplateId.daily_sales_orders, "일별 매출과 주문 수", "daily sales and order count"),
    (OlistTemplateId.order_status_distribution, "주문 상태별 분포", "order status distribution"),
    (OlistTemplateId.category_sales, "카테고리별 매출", "category sales"),
    (OlistTemplateId.review_score_distribution, "리뷰 평점 분포", "review rating distribution"),
    (OlistTemplateId.payment_method_summary, "결제 수단별 요약", "payment method summary"),
    (OlistTemplateId.customer_state_sales, "고객 주별 매출", "customer state sales"),
    (OlistTemplateId.seller_state_sales, "판매자 주별 매출", "seller state sales"),
    (OlistTemplateId.seller_performance, "판매자별 성과", "seller performance"),
    (OlistTemplateId.delivery_delay_summary, "배송 지연 및 소요일", "delivery delay lead time"),
    (OlistTemplateId.category_review_summary, "카테고리별 리뷰 요약", "category review summary"),
    (OlistTemplateId.payment_installment_summary, "할부 개월별 요약", "installment summary"),
    (OlistTemplateId.freight_cost_summary, "카테고리 배송비 요약", "freight cost summary"),
    (OlistTemplateId.basket_size_summary, "장바구니 크기 요약", "basket size summary"),
    (OlistTemplateId.repeat_customer_summary, "신규 재구매 고객 요약", "repeat customer summary"),
    (OlistTemplateId.customer_rfm, "RFM 데이터마트", "RFM data mart"),
    (OlistTemplateId.monthly_customer_cohort, "월별 코호트 데이터마트", "monthly cohort data mart"),
    (OlistTemplateId.customer_repeat_behavior, "재구매 행동 데이터마트", "repeat behavior data mart"),
    (OlistTemplateId.order_delivery_performance, "주문별 배송 성과 데이터마트", "order delivery performance data mart"),
    (OlistTemplateId.monthly_category_performance, "월별 카테고리 성과 데이터마트", "monthly category performance data mart"),
    (OlistTemplateId.monthly_seller_performance, "월별 판매자 성과 데이터마트", "monthly seller performance data mart"),
    (OlistTemplateId.customer_seller_geo, "고객 지역과 판매자 지역 간 geo 데이터마트", "customer region seller region geo data mart"),
    (OlistTemplateId.category_review_delivery, "카테고리 리뷰 배송 데이터마트", "category review delivery data mart"),
    (OlistTemplateId.payment_behavior, "결제 행동 데이터마트", "payment behavior data mart"),
    (OlistTemplateId.product_logistics, "상품 물류 데이터마트", "product logistics data mart"),
]


def test_registry_has_exactly_fifteen_queries_and_ten_marts() -> None:
    definitions = list(OLIST_TEMPLATE_REGISTRY.values())

    assert len(definitions) == 25
    assert len({item.template_id for item in definitions}) == 25
    assert sum(item.template_kind == OlistTemplateKind.query for item in definitions) == 15
    assert sum(item.template_kind == OlistTemplateKind.mart for item in definitions) == 10


@pytest.mark.parametrize(("template_id", "korean", "english"), TEMPLATE_QUESTIONS)
def test_all_templates_match_korean_and_english_variants(
    olist_catalog: dict,
    template_id: OlistTemplateId,
    korean: str,
    english: str,
) -> None:
    for question in (korean, english):
        match = match_olist_template(question, olist_catalog)
        assert match.supported is True, (question, match.reason)
        assert match.template_id == template_id
        assert match.template_kind == OLIST_TEMPLATE_REGISTRY[template_id].template_kind


@pytest.mark.parametrize(
    "question",
    [
        "월별 주문 목록", "일별 주문 목록", "주문 상태 목록", "카테고리 목록", "리뷰 요약",
        "결제 수단 목록", "고객 지역 현황", "판매자 지역 현황", "판매자 목록", "배송 현황",
        "카테고리 리뷰", "할부 방식", "배송비 내역", "장바구니 분석", "재구매 고객 분석",
        "RFM 분석", "월별 코호트 분석", "재구매 행동 분석", "주문별 배송 성과 분석",
        "월별 카테고리 성과 분석", "월별 판매자 성과 분석", "고객 판매자 지역 분석",
        "카테고리 리뷰 배송 분석", "결제 행동 분석", "상품 물류 분석",
    ],
)
def test_partial_intent_does_not_match_a_template(olist_catalog: dict, question: str) -> None:
    assert match_olist_template(question, olist_catalog).supported is False


@pytest.mark.parametrize(("template_id", "question"), [(row[0], row[1]) for row in TEMPLATE_QUESTIONS])
def test_each_template_falls_back_when_required_schema_is_missing(
    olist_catalog: dict,
    template_id: OlistTemplateId,
    question: str,
) -> None:
    catalog = json.loads(json.dumps(olist_catalog))
    missing_table = next(iter(OLIST_TEMPLATE_REGISTRY[template_id].required_schema))
    catalog.pop(missing_table)

    match = match_olist_template(question, catalog)

    assert match.supported is False
    assert missing_table in match.reason


@pytest.mark.parametrize(("template_id", "question"), [(row[0], row[1]) for row in TEMPLATE_QUESTIONS])
def test_each_template_falls_back_when_required_column_is_missing(
    olist_catalog: dict,
    template_id: OlistTemplateId,
    question: str,
) -> None:
    catalog = json.loads(json.dumps(olist_catalog))
    required_schema = OLIST_TEMPLATE_REGISTRY[template_id].required_schema
    table_name = next(name for name, columns in required_schema.items() if columns)
    column_name = next(iter(required_schema[table_name]))
    catalog[table_name]["columns"] = [
        column
        for column in catalog[table_name]["columns"]
        if column["name"] != column_name
    ]

    match = match_olist_template(question, catalog)

    assert match.supported is False
    assert f"{table_name}.{column_name}" in match.reason


def test_allowlisted_parameters_are_typed_and_rendered(olist_catalog: dict) -> None:
    match = match_olist_template(
        "2024년 delivered 카테고리 매출 상위 10개",
        olist_catalog,
    )

    assert match.supported is True
    assert match.parameters.start_date == "2024-01-01"
    assert match.parameters.end_date == "2024-12-31"
    assert match.parameters.order_statuses == ["delivered"]
    assert match.parameters.top_n == 10
    sql = build_olist_sql_draft(
        match.template_id,
        olist_catalog,
        parameters=match.parameters,
    ).sql
    assert "o.order_purchase_timestamp >= '2024-01-01'" in sql
    assert "o.order_status IN ('delivered')" in sql
    assert "LIMIT 10" in sql


def test_region_parameter_is_normalized_before_rendering(olist_catalog: dict) -> None:
    match = match_olist_template("판매자 주별 매출, seller state sp", olist_catalog)

    assert match.supported is True
    assert match.parameters.seller_states == ["SP"]
    sql = build_olist_sql_draft(match.template_id, olist_catalog, parameters=match.parameters).sql
    assert "s.seller_state IN ('SP')" in sql


@pytest.mark.parametrize(
    "question",
    [
        "최근 6개월 월별 매출과 주문 수",
        "RFM 데이터마트 상위 10개",
        "delivered 주문 상태별 분포",
        "월별 매출이 1000 이상인 기간",
    ],
)
def test_unsupported_parameters_use_semantic_fallback(olist_catalog: dict, question: str) -> None:
    assert match_olist_template(question, olist_catalog).supported is False


def test_unrecognized_structured_filter_uses_semantic_fallback(olist_catalog: dict) -> None:
    match = match_olist_template(
        "월별 매출과 주문 수",
        olist_catalog,
        filters=["브라질 남동부"],
    )

    assert match.supported is False
    assert "지원하지 않는 필터" in match.reason


@pytest.mark.parametrize("template_id", [item[0] for item in TEMPLATE_QUESTIONS[15:]])
def test_mart_contract_has_single_ctas_and_declared_output_columns(
    olist_catalog: dict,
    template_id: OlistTemplateId,
) -> None:
    definition = OLIST_TEMPLATE_REGISTRY[template_id]
    draft = build_olist_sql_draft(template_id, olist_catalog)

    assert definition.route_kind == "comprehensive"
    assert draft.sql_type == "create_table_as"
    assert draft.sql.upper().count("CREATE TABLE") == 1
    assert draft.target_table == f"analytics.olist_{template_id.value}"
    assert draft.business_grain == definition.business_grain
    assert draft.output_columns == definition.output_columns

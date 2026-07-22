"""Olist 고정 스키마용 결정론적 SQL 템플릿 레지스트리."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel

from DATA_Analyst_Assistant_Agent.agents.sql.state import SQLDraft
from DATA_Analyst_Assistant_Agent.shared.contracts import OlistTemplateId


class OlistTemplateMatch(BaseModel):
    template_id: OlistTemplateId | None = None
    reason: str
    supported: bool = False


_REQUIRED_SCHEMA: dict[OlistTemplateId, dict[str, set[str]]] = {
    OlistTemplateId.monthly_sales_orders: {
        "orders": {"order_id", "order_purchase_timestamp"},
        "order_items": {"order_id", "price"},
    },
    OlistTemplateId.order_status_distribution: {
        "orders": {"order_id", "order_status"},
    },
    OlistTemplateId.category_sales: {
        "order_items": {"order_id", "product_id", "price"},
        "products": {"product_id", "product_category_name"},
        "product_category_name_translation": {
            "product_category_name",
            "product_category_name_english",
        },
    },
    OlistTemplateId.review_score_distribution: {
        "order_reviews": {"review_id", "review_score"},
    },
    OlistTemplateId.payment_method_summary: {
        "order_payments": {"order_id", "payment_type", "payment_value"},
    },
}

_UNSUPPORTED_CONSTRAINTS = (
    re.compile(r"\b(?:19|20)\d{2}(?:년|[-/.])?"),
    re.compile(r"\b\d{1,2}\s*(?:월|일|분기)\b"),
    re.compile(r"(?:최근|지난|이번|작년|올해|기간|부터|까지|사이|between|after|before|last\s+\d+)", re.I),
    re.compile(r"(?:상위|하위|top|bottom)(?:\s*\d+|\s*(?:몇|n|개))?", re.I),
    re.compile(r"(?:주문\s*상태|order[_ ]?status)\s*(?:가|는|=|in|중)", re.I),
    re.compile(r"(?:delivered|canceled|cancelled|shipped|processing|invoiced|unavailable|배송\s*완료|취소(?:된)?|배송\s*중|처리\s*중)", re.I),
)

_SIGNALS: dict[OlistTemplateId, tuple[tuple[str, ...], ...]] = {
    OlistTemplateId.monthly_sales_orders: (
        ("월별", "월간", "month"),
        ("매출", "판매액", "sales", "revenue"),
    ),
    OlistTemplateId.order_status_distribution: (
        ("주문 상태", "상태별 주문", "order status"),
        ("분포", "비율", "구성", "현황", "주문 수", "distribution", "share"),
    ),
    OlistTemplateId.category_sales: (
        ("카테고리", "상품군", "category"),
        ("매출", "판매", "sales", "revenue"),
    ),
    OlistTemplateId.review_score_distribution: (
        ("리뷰", "평점", "별점", "review score", "rating"),
        ("분포", "비율", "건수", "개수", "리뷰 수", "distribution", "share", "count"),
    ),
    OlistTemplateId.payment_method_summary: (
        ("결제 수단", "결제 방식", "payment method", "payment type"),
        ("요약", "현황", "주문 수", "결제금액", "금액", "summary", "amount"),
    ),
}


def match_olist_template(
    question: str,
    catalog: dict[str, Any] | str | None,
    *,
    metric: str | None = None,
    dimension: str | None = None,
    filters: list[str] | None = None,
    query_rules: dict[str, Any] | None = None,
) -> OlistTemplateMatch:
    """질문을 지원 범위가 좁은 Olist 유형에 고정밀도로 매칭한다."""
    normalized = " ".join(str(question or "").casefold().split())
    if not normalized:
        return OlistTemplateMatch(reason="질문이 비어 있습니다.")

    hint_text = " ".join(
        part
        for part in (
            str(metric or "").casefold(),
            str(dimension or "").casefold(),
            _flatten_text(query_rules or {}).casefold(),
        )
        if part
    )
    combined = f"{normalized} {hint_text}".strip()
    if filters or any(pattern.search(normalized) for pattern in _UNSUPPORTED_CONSTRAINTS):
        return OlistTemplateMatch(reason="v1이 지원하지 않는 기간·상태·Top N 조건이 있습니다.")

    candidates = [
        template_id
        for template_id, signal_groups in _SIGNALS.items()
        if all(any(signal in combined for signal in group) for group in signal_groups)
    ]
    if len(candidates) != 1:
        reason = "둘 이상의 템플릿 의미가 겹칩니다." if candidates else "고정 템플릿 의미와 정확히 일치하지 않습니다."
        return OlistTemplateMatch(reason=reason)

    template_id = candidates[0]
    missing = missing_catalog_requirements(catalog, template_id)
    if missing:
        return OlistTemplateMatch(
            template_id=template_id,
            reason=f"Olist 필수 스키마가 부족합니다: {', '.join(missing)}",
            supported=False,
        )
    return OlistTemplateMatch(
        template_id=template_id,
        reason=f"질문과 catalog가 {template_id.value}의 고정 의미를 충족합니다.",
        supported=True,
    )


def missing_catalog_requirements(
    catalog: dict[str, Any] | str | None,
    template_id: OlistTemplateId,
) -> list[str]:
    tables = _catalog_tables(catalog)
    missing: list[str] = []
    for table_name, required_columns in _REQUIRED_SCHEMA[template_id].items():
        table = tables.get(table_name)
        if not isinstance(table, dict):
            missing.append(table_name)
            continue
        available_columns = _catalog_columns(table)
        for column_name in sorted(required_columns - available_columns):
            missing.append(f"{table_name}.{column_name}")
    return missing


def build_olist_sql_draft(
    template_id: OlistTemplateId | str,
    catalog: dict[str, Any] | str | None = None,
) -> SQLDraft:
    """선택된 Olist 템플릿을 기존 SQLDraft 계약으로 반환한다."""
    selected = OlistTemplateId(template_id)
    if catalog is not None:
        missing = missing_catalog_requirements(catalog, selected)
        if missing:
            raise ValueError(f"Olist 템플릿 필수 스키마가 없습니다: {', '.join(missing)}")
    return SQLDraft(**_DRAFTS[selected])


def build_olist_validation_plan(template_id: OlistTemplateId | str) -> dict[str, Any]:
    """기존 intent 검증기가 템플릿 의미를 검사할 수 있는 최소 계획을 만든다."""
    selected = OlistTemplateId(template_id)
    drafts = {
        OlistTemplateId.monthly_sales_orders: {
            "question_type": "trend",
            "target_metrics": ["상품 매출", "서로 다른 주문 수"],
            "dimensions": ["purchase_month"],
            "required_aggregations": ["SUM", "COUNT_DISTINCT"],
        },
        OlistTemplateId.order_status_distribution: {
            "question_type": "distribution",
            "target_metrics": ["주문 수", "전체 주문 대비 비율"],
            "dimensions": ["order_status"],
            "required_aggregations": ["COUNT_DISTINCT"],
        },
        OlistTemplateId.category_sales: {
            "question_type": "aggregation",
            "target_metrics": ["상품 매출", "상품행 수", "서로 다른 주문 수"],
            "dimensions": ["category_name"],
            "required_aggregations": ["SUM", "COUNT", "COUNT_DISTINCT"],
        },
        OlistTemplateId.review_score_distribution: {
            "question_type": "distribution",
            "target_metrics": ["서로 다른 리뷰 수", "전체 리뷰 대비 비율"],
            "dimensions": ["review_score"],
            "required_aggregations": ["COUNT_DISTINCT"],
        },
        OlistTemplateId.payment_method_summary: {
            "question_type": "aggregation",
            "target_metrics": ["서로 다른 주문 수", "총 결제금액"],
            "dimensions": ["payment_type"],
            "required_aggregations": ["COUNT_DISTINCT", "SUM"],
        },
    }
    payload = drafts[selected]
    return {
        "route_kind": "simple",
        **payload,
        "filters": [],
        "selected_join_tables": list(_REQUIRED_SCHEMA[selected]),
        "required_columns": [],
        "validation_contract": {
            "expected_result_shape": "grouped_aggregate",
            "required_tables": list(_REQUIRED_SCHEMA[selected]),
            "dimensions": payload["dimensions"],
            "target_metrics": payload["target_metrics"],
            "required_aggregations": payload["required_aggregations"],
        },
    }


def _catalog_tables(catalog: dict[str, Any] | str | None) -> dict[str, Any]:
    if isinstance(catalog, str):
        try:
            catalog = json.loads(catalog)
        except (TypeError, json.JSONDecodeError):
            return {}
    if not isinstance(catalog, dict):
        return {}
    tables = catalog.get("tables")
    if isinstance(tables, dict):
        return {str(name).split(".")[-1].strip("`"): value for name, value in tables.items()}
    return {
        str(name).split(".")[-1].strip("`"): value
        for name, value in catalog.items()
        if isinstance(value, dict) and "columns" in value
    }


def _catalog_columns(table: dict[str, Any]) -> set[str]:
    raw_columns = table.get("columns") or []
    if isinstance(raw_columns, dict):
        return {str(name) for name in raw_columns}
    return {
        str(column.get("name"))
        for column in raw_columns
        if isinstance(column, dict) and column.get("name")
    }


def _flatten_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(f"{key} {_flatten_text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_text(item) for item in value)
    return str(value or "")


_DRAFTS: dict[OlistTemplateId, dict[str, Any]] = {
    OlistTemplateId.monthly_sales_orders: {
        "sql": """SELECT
    DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m-01') AS purchase_month,
    SUM(oi.price) AS product_sales,
    COUNT(DISTINCT o.order_id) AS order_count
FROM orders AS o
LEFT JOIN order_items AS oi ON oi.order_id = o.order_id
GROUP BY DATE_FORMAT(o.order_purchase_timestamp, '%Y-%m-01')
ORDER BY purchase_month;""",
        "sql_type": "select",
        "source_tables": ["orders", "order_items"],
        "source_column_refs": ["orders.order_id", "orders.order_purchase_timestamp", "order_items.order_id", "order_items.price"],
        "derived_columns": ["purchase_month", "product_sales", "order_count"],
        "output_columns": ["purchase_month", "product_sales", "order_count"],
        "business_grain": "구매 월",
        "reasoning": "상품행이 없는 주문도 제외하지 않고 모든 주문 상태를 포함해 구매 월별 상품 가격 합계와 서로 다른 주문 수를 계산합니다.",
    },
    OlistTemplateId.order_status_distribution: {
        "sql": """SELECT
    o.order_status,
    COUNT(DISTINCT o.order_id) AS order_count,
    ROUND(100.0 * COUNT(DISTINCT o.order_id) / SUM(COUNT(DISTINCT o.order_id)) OVER (), 2) AS order_percentage
FROM orders AS o
GROUP BY o.order_status
ORDER BY order_count DESC, o.order_status;""",
        "sql_type": "select",
        "source_tables": ["orders"],
        "source_column_refs": ["orders.order_id", "orders.order_status"],
        "derived_columns": ["order_count", "order_percentage"],
        "output_columns": ["order_status", "order_count", "order_percentage"],
        "business_grain": "주문 상태",
        "reasoning": "주문 상태별 서로 다른 주문 수와 전체 주문 대비 비율을 계산합니다.",
    },
    OlistTemplateId.category_sales: {
        "sql": """SELECT
    COALESCE(NULLIF(t.product_category_name_english, ''), NULLIF(p.product_category_name, ''), 'unknown') AS category_name,
    SUM(oi.price) AS product_sales,
    COUNT(*) AS item_row_count,
    COUNT(DISTINCT oi.order_id) AS order_count
FROM order_items AS oi
JOIN products AS p ON p.product_id = oi.product_id
LEFT JOIN product_category_name_translation AS t
    ON t.product_category_name = p.product_category_name
GROUP BY COALESCE(NULLIF(t.product_category_name_english, ''), NULLIF(p.product_category_name, ''), 'unknown')
ORDER BY product_sales DESC, category_name;""",
        "sql_type": "select",
        "source_tables": ["order_items", "products", "product_category_name_translation"],
        "source_column_refs": ["order_items.order_id", "order_items.product_id", "order_items.price", "products.product_id", "products.product_category_name", "product_category_name_translation.product_category_name", "product_category_name_translation.product_category_name_english"],
        "derived_columns": ["category_name", "product_sales", "item_row_count", "order_count"],
        "output_columns": ["category_name", "product_sales", "item_row_count", "order_count"],
        "business_grain": "상품 카테고리",
        "reasoning": "검증된 상품 조인으로 영문 카테고리를 우선 사용하고 상품 매출·상품행·주문 수를 구분합니다.",
    },
    OlistTemplateId.review_score_distribution: {
        "sql": """WITH review_scores AS (
    SELECT 1 AS review_score
    UNION ALL SELECT 2
    UNION ALL SELECT 3
    UNION ALL SELECT 4
    UNION ALL SELECT 5
)
SELECT
    s.review_score,
    COUNT(DISTINCT r.review_id) AS review_count,
    ROUND(100.0 * COUNT(DISTINCT r.review_id) / NULLIF((SELECT COUNT(DISTINCT review_id) FROM order_reviews), 0), 2) AS review_percentage
FROM review_scores AS s
LEFT JOIN order_reviews AS r ON r.review_score = s.review_score
GROUP BY s.review_score
ORDER BY s.review_score;""",
        "sql_type": "select",
        "source_tables": ["order_reviews"],
        "source_column_refs": ["order_reviews.review_id", "order_reviews.review_score"],
        "derived_columns": ["review_count", "review_percentage"],
        "output_columns": ["review_score", "review_count", "review_percentage"],
        "business_grain": "리뷰 점수(1~5점)",
        "reasoning": "1~5점 축을 보존해 서로 다른 리뷰 수와 전체 리뷰 대비 비율을 계산합니다.",
    },
    OlistTemplateId.payment_method_summary: {
        "sql": """SELECT
    p.payment_type,
    COUNT(DISTINCT p.order_id) AS order_count,
    SUM(p.payment_value) AS total_payment_value
FROM order_payments AS p
GROUP BY p.payment_type
ORDER BY total_payment_value DESC, p.payment_type;""",
        "sql_type": "select",
        "source_tables": ["order_payments"],
        "source_column_refs": ["order_payments.order_id", "order_payments.payment_type", "order_payments.payment_value"],
        "derived_columns": ["order_count", "total_payment_value"],
        "output_columns": ["payment_type", "order_count", "total_payment_value"],
        "business_grain": "결제 수단",
        "reasoning": "결제 레코드 수가 아니라 서로 다른 주문 수와 결제금액 합계를 계산합니다.",
    },
}


__all__ = [
    "OlistTemplateId",
    "OlistTemplateMatch",
    "build_olist_sql_draft",
    "build_olist_validation_plan",
    "match_olist_template",
    "missing_catalog_requirements",
]

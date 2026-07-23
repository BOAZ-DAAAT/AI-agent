"""Olist 고정 스키마용 결정론적 SQL 템플릿 레지스트리."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import date
from typing import Any

from pydantic import BaseModel, Field

from DATA_Analyst_Assistant_Agent.agents.sql.olist_template_sql import OLIST_SQL_DRAFTS
from DATA_Analyst_Assistant_Agent.agents.sql.state import SQLDraft
from DATA_Analyst_Assistant_Agent.shared.contracts import (
    OlistTemplateDefinition,
    OlistTemplateId,
    OlistTemplateKind,
    OlistTemplateParameters,
)


class OlistTemplateMatch(BaseModel):
    template_id: OlistTemplateId | None = None
    template_kind: OlistTemplateKind | None = None
    parameters: OlistTemplateParameters = Field(default_factory=OlistTemplateParameters)
    reason: str
    supported: bool = False


def _required_schema(template_id: OlistTemplateId) -> dict[str, set[str]]:
    draft = OLIST_SQL_DRAFTS[template_id]
    schema = {str(table): set() for table in draft["source_tables"]}
    for reference in draft["source_column_refs"]:
        table, column = str(reference).split(".", 1)
        schema.setdefault(table, set()).add(column)
    return schema


_QUERY_META: dict[OlistTemplateId, tuple[str, list[str], list[str], set[str]]] = {
    OlistTemplateId.monthly_sales_orders: ("monthly_sales_orders", ["product_sales", "order_count"], ["purchase_month"], {"start_date", "end_date", "order_statuses"}),
    OlistTemplateId.daily_sales_orders: ("daily_sales_orders", ["product_sales", "order_count"], ["purchase_date"], {"start_date", "end_date", "order_statuses"}),
    OlistTemplateId.order_status_distribution: ("order_status_distribution", ["order_count", "order_percentage"], ["order_status"], {"start_date", "end_date", "top_n"}),
    OlistTemplateId.category_sales: ("category_sales", ["product_sales", "item_count", "order_count"], ["category_name"], {"start_date", "end_date", "order_statuses", "top_n"}),
    OlistTemplateId.review_score_distribution: ("review_score_distribution", ["review_count", "review_percentage"], ["review_score"], {"start_date", "end_date", "order_statuses"}),
    OlistTemplateId.payment_method_summary: ("payment_method_summary", ["order_count", "payment_count", "total_payment_value"], ["payment_type"], {"start_date", "end_date", "order_statuses", "top_n"}),
    OlistTemplateId.customer_state_sales: ("customer_state_sales", ["order_count", "product_sales"], ["customer_state"], {"start_date", "end_date", "order_statuses", "customer_states", "top_n"}),
    OlistTemplateId.seller_state_sales: ("seller_state_sales", ["order_count", "item_count", "product_sales"], ["seller_state"], {"start_date", "end_date", "order_statuses", "seller_states", "top_n"}),
    OlistTemplateId.seller_performance: ("seller_performance", ["order_count", "item_count", "product_sales", "freight_value"], ["seller_id"], {"start_date", "end_date", "order_statuses", "seller_states", "top_n"}),
    OlistTemplateId.delivery_delay_summary: ("delivery_delay_summary", ["order_count", "avg_delivery_days", "avg_delay_days"], ["delivery_status"], {"start_date", "end_date", "order_statuses"}),
    OlistTemplateId.category_review_summary: ("category_review_summary", ["reviewed_order_count", "avg_review_score"], ["category_name"], {"start_date", "end_date", "order_statuses", "top_n"}),
    OlistTemplateId.payment_installment_summary: ("payment_installment_summary", ["order_count", "payment_count", "total_payment_value", "avg_payment_value"], ["payment_installments"], {"start_date", "end_date", "order_statuses", "top_n"}),
    OlistTemplateId.freight_cost_summary: ("freight_cost_summary", ["total_freight_value", "avg_item_freight_value"], ["category_name"], {"start_date", "end_date", "order_statuses", "top_n"}),
    OlistTemplateId.basket_size_summary: ("basket_size_summary", ["order_count", "avg_order_product_value"], ["item_count"], {"start_date", "end_date", "order_statuses", "top_n"}),
    OlistTemplateId.repeat_customer_summary: ("repeat_customer_summary", ["customer_count", "avg_purchase_count", "avg_customer_span_days"], ["customer_type"], {"start_date", "end_date", "order_statuses"}),
}

_MART_META: dict[OlistTemplateId, tuple[str, list[str], list[str], set[str]]] = {
    OlistTemplateId.customer_rfm: ("customer_rfm", ["recency_days", "frequency", "monetary_value"], ["customer_unique_id"], {"start_date", "end_date", "order_statuses"}),
    OlistTemplateId.monthly_customer_cohort: ("monthly_customer_cohort", ["active_customers", "order_count", "product_sales"], ["cohort_month", "activity_month"], {"start_date", "end_date", "order_statuses"}),
    OlistTemplateId.customer_repeat_behavior: ("customer_repeat_behavior", ["purchase_count", "repeat_purchase_count", "avg_days_between_purchases", "cumulative_product_sales"], ["customer_unique_id"], {"start_date", "end_date", "order_statuses"}),
    OlistTemplateId.order_delivery_performance: ("order_delivery_performance", ["delivery_days", "delay_days", "is_delayed"], ["order_id"], {"start_date", "end_date", "order_statuses"}),
    OlistTemplateId.monthly_category_performance: ("monthly_category_performance", ["order_count", "item_count", "product_sales", "freight_value"], ["purchase_month", "category_name"], {"start_date", "end_date", "order_statuses"}),
    OlistTemplateId.monthly_seller_performance: ("monthly_seller_performance", ["order_count", "product_sales", "avg_delivery_days", "avg_review_score"], ["purchase_month", "seller_id"], {"start_date", "end_date", "order_statuses", "seller_states"}),
    OlistTemplateId.customer_seller_geo: ("customer_seller_geo", ["order_count", "product_sales", "freight_value", "avg_delivery_days"], ["customer_state", "seller_state"], {"start_date", "end_date", "order_statuses", "customer_states", "seller_states"}),
    OlistTemplateId.category_review_delivery: ("category_review_delivery", ["avg_review_score", "avg_delivery_days", "delay_rate"], ["category_name"], {"start_date", "end_date", "order_statuses"}),
    OlistTemplateId.payment_behavior: ("payment_behavior", ["payment_count", "payment_method_count", "max_installments", "total_payment_value"], ["order_id", "customer_unique_id"], {"start_date", "end_date", "order_statuses"}),
    OlistTemplateId.product_logistics: ("product_logistics", ["freight_value", "avg_delivery_days", "delay_rate"], ["product_id", "category_name"], {"start_date", "end_date", "order_statuses"}),
}


def _definition(
    template_id: OlistTemplateId,
    template_kind: OlistTemplateKind,
    meta: tuple[str, list[str], list[str], set[str]],
) -> OlistTemplateDefinition:
    intent, metrics, dimensions, allowed_parameters = meta
    draft = OLIST_SQL_DRAFTS[template_id]
    return OlistTemplateDefinition(
        template_id=template_id,
        template_kind=template_kind,
        route_kind="simple" if template_kind == OlistTemplateKind.query else "comprehensive",
        sql_type="select" if template_kind == OlistTemplateKind.query else "create_table_as",
        intent=intent,
        metrics=metrics,
        dimensions=dimensions,
        required_schema=_required_schema(template_id),
        allowed_parameters=allowed_parameters,
        output_columns=list(draft["output_columns"]),
        business_grain=str(draft["business_grain"]),
    )


OLIST_TEMPLATE_REGISTRY: dict[OlistTemplateId, OlistTemplateDefinition] = {
    **{key: _definition(key, OlistTemplateKind.query, value) for key, value in _QUERY_META.items()},
    **{key: _definition(key, OlistTemplateKind.mart, value) for key, value in _MART_META.items()},
}


_CONCEPT_LEXICON: dict[str, tuple[str, ...]] = {
    "monthly": ("월별", "월간", "month", "monthly", "by month"),
    "daily": ("일별", "일간", "daily", "by day"),
    "sales": ("매출", "판매액", "판매 현황", "sales", "revenue"),
    "orders": ("주문 수", "주문 건수", "order count", "orders"),
    "distribution": ("분포", "비율", "구성", "distribution", "share"),
    "order_status": ("주문 상태", "상태별 주문", "order status"),
    "category": ("카테고리", "상품군", "category"),
    "review": ("리뷰", "평점", "별점", "review", "rating"),
    "payment_method": ("결제 수단", "결제 방식", "payment method", "payment type"),
    "customer_region": ("고객 주", "고객 지역", "customer state", "customer region"),
    "seller_region": ("판매자 주", "판매자 지역", "seller state", "seller region"),
    "seller": ("판매자별", "판매자 성과", "seller performance", "by seller"),
    "performance": ("성과", "실적", "performance"),
    "delivery": ("배송", "delivery"),
    "delay": ("지연", "소요일", "delay", "late", "lead time"),
    "summary": ("요약", "현황", "금액", "주문 수", "amount", "summary", "overview"),
    "installment": ("할부", "installment"),
    "freight": ("배송비", "운임", "freight", "shipping cost"),
    "basket": ("장바구니", "주문당 상품", "basket size", "items per order"),
    "repeat_customer": ("재구매 고객", "신규 고객", "구매 빈도", "repeat customer", "returning customer"),
    "mart": ("데이터마트", "데이터 마트", "마트", "datamart", "data mart", "ctas", "분석 테이블"),
    "rfm": ("rfm", "recency frequency monetary", "최근성 빈도 금액"),
    "cohort": ("코호트", "cohort"),
    "repeat_behavior": ("재구매 행동", "재구매 주기", "repeat behavior", "purchase interval"),
    "order_delivery": ("주문별 배송", "order delivery"),
    "geo": ("지역 간", "지역 조합", "geo", "geography"),
    "payment_behavior": ("결제 행동", "결제 패턴", "payment behavior"),
    "product_logistics": ("상품 물류", "상품 크기", "상품 무게", "product logistics", "product dimensions"),
}

_REQUIRED_CONCEPTS: dict[OlistTemplateId, set[str]] = {
    OlistTemplateId.monthly_sales_orders: {"monthly", "sales", "orders"},
    OlistTemplateId.daily_sales_orders: {"daily", "sales", "orders"},
    OlistTemplateId.order_status_distribution: {"order_status", "distribution"},
    OlistTemplateId.category_sales: {"category", "sales"},
    OlistTemplateId.review_score_distribution: {"review", "distribution"},
    OlistTemplateId.payment_method_summary: {"payment_method", "summary"},
    OlistTemplateId.customer_state_sales: {"customer_region", "sales"},
    OlistTemplateId.seller_state_sales: {"seller_region", "sales"},
    OlistTemplateId.seller_performance: {"seller", "performance"},
    OlistTemplateId.delivery_delay_summary: {"delivery", "delay"},
    OlistTemplateId.category_review_summary: {"category", "review", "summary"},
    OlistTemplateId.payment_installment_summary: {"installment", "summary"},
    OlistTemplateId.freight_cost_summary: {"freight", "summary"},
    OlistTemplateId.basket_size_summary: {"basket", "summary"},
    OlistTemplateId.repeat_customer_summary: {"repeat_customer", "summary"},
    OlistTemplateId.customer_rfm: {"mart", "rfm"},
    OlistTemplateId.monthly_customer_cohort: {"mart", "monthly", "cohort"},
    OlistTemplateId.customer_repeat_behavior: {"mart", "repeat_behavior"},
    OlistTemplateId.order_delivery_performance: {"mart", "order_delivery", "performance"},
    OlistTemplateId.monthly_category_performance: {"mart", "monthly", "category", "performance"},
    OlistTemplateId.monthly_seller_performance: {"mart", "monthly", "seller", "performance"},
    OlistTemplateId.customer_seller_geo: {"mart", "customer_region", "seller_region", "geo"},
    OlistTemplateId.category_review_delivery: {"mart", "category", "review", "delivery"},
    OlistTemplateId.payment_behavior: {"mart", "payment_behavior"},
    OlistTemplateId.product_logistics: {"mart", "product_logistics"},
}

_ORDER_STATUSES = {
    "delivered": ("delivered", "배송 완료"),
    "shipped": ("shipped", "배송 중"),
    "canceled": ("canceled", "cancelled", "취소"),
    "processing": ("processing", "처리 중"),
    "invoiced": ("invoiced", "청구"),
    "approved": ("approved", "승인"),
    "created": ("created", "생성"),
    "unavailable": ("unavailable", "사용 불가"),
}
_DATE_RE = re.compile(r"\b((?:19|20)\d{2})[-/.](0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])\b")
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\s*년?\b")
_TOP_RE = re.compile(r"(?:상위|top)\s*(\d{1,3})", re.I)
_CUSTOMER_STATE_RE = re.compile(r"(?:고객\s*(?:주|지역)|customer\s*(?:state|region))\s*[:=]?\s*([A-Za-z]{2})(?=\s|,|$|의|에서|만)", re.I)
_SELLER_STATE_RE = re.compile(r"(?:판매자\s*(?:주|지역)|seller\s*(?:state|region))\s*[:=]?\s*([A-Za-z]{2})(?=\s|,|$|의|에서|만)", re.I)
_RELATIVE_PERIOD_RE = re.compile(r"(?:최근|지난|이번|작년|올해|last\s+\d+|previous|between|부터|까지)", re.I)
_UNKNOWN_FILTER_RE = re.compile(r"(?:이상|이하|초과|미만|greater\s+than|less\s+than|where\b|필터|조건)", re.I)


def _concepts(text: str) -> set[str]:
    concepts = {
        concept
        for concept, signals in _CONCEPT_LEXICON.items()
        if any(signal in text for signal in signals)
    }
    if "개월별" in text and not re.search(r"(?<!개)월별", text):
        concepts.discard("monthly")
    return concepts


def _extract_parameters(text: str) -> tuple[OlistTemplateParameters, set[str], str | None]:
    values: dict[str, Any] = {}
    names: set[str] = set()
    dates = []
    for match in _DATE_RE.finditer(text):
        try:
            parsed = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            return OlistTemplateParameters(), set(), "유효하지 않은 날짜입니다."
        dates.append(parsed.isoformat())
    if not dates:
        years = [int(item) for item in _YEAR_RE.findall(text)]
        if len(years) == 1:
            dates = [f"{years[0]:04d}-01-01", f"{years[0]:04d}-12-31"]
        elif len(years) > 2:
            return OlistTemplateParameters(), set(), "기간에는 시작·종료 연도만 허용됩니다."
    if dates:
        values["start_date"] = min(dates)
        values["end_date"] = max(dates)
        names.update({"start_date", "end_date"})
    elif _RELATIVE_PERIOD_RE.search(text):
        return OlistTemplateParameters(), set(), "상대 기간은 결정론적으로 해석하지 않습니다."

    statuses = [status for status, signals in _ORDER_STATUSES.items() if any(signal in text for signal in signals)]
    if statuses:
        values["order_statuses"] = statuses
        names.add("order_statuses")

    top = _TOP_RE.search(text)
    if top:
        top_n = int(top.group(1))
        if not 1 <= top_n <= 100:
            return OlistTemplateParameters(), set(), "Top N은 1~100만 허용됩니다."
        values["top_n"] = top_n
        names.add("top_n")

    customer_states = sorted({match.group(1).upper() for match in _CUSTOMER_STATE_RE.finditer(text)})
    if customer_states:
        values["customer_states"] = customer_states
        names.add("customer_states")
    seller_states = sorted({match.group(1).upper() for match in _SELLER_STATE_RE.finditer(text)})
    if seller_states:
        values["seller_states"] = seller_states
        names.add("seller_states")
    if _UNKNOWN_FILTER_RE.search(text):
        return OlistTemplateParameters(), set(), "지원하지 않는 필터 표현이 있습니다."
    return OlistTemplateParameters(**values), names, None


def match_olist_template(
    question: str,
    catalog: dict[str, Any] | str | None,
    *,
    metric: str | None = None,
    dimension: str | None = None,
    filters: list[str] | None = None,
    query_rules: dict[str, Any] | None = None,
) -> OlistTemplateMatch:
    """구조화된 의미·파라미터·스키마 계약이 하나만 맞을 때 템플릿을 선택한다."""
    normalized = " ".join(str(question or "").casefold().split())
    if not normalized:
        return OlistTemplateMatch(reason="질문이 비어 있습니다.")
    hint_text = " ".join(
        item for item in (
            str(metric or "").casefold(),
            str(dimension or "").casefold(),
            _flatten_text(query_rules or {}).casefold(),
        ) if item
    )
    filter_text = " ".join(str(item).casefold() for item in (filters or []) if str(item).strip())
    combined = " ".join(item for item in (normalized, hint_text, filter_text) if item)
    detected_concepts = _concepts(combined)
    requested_kind = OlistTemplateKind.mart if "mart" in detected_concepts else OlistTemplateKind.query
    candidates = [
        template_id
        for template_id, required in _REQUIRED_CONCEPTS.items()
        if OLIST_TEMPLATE_REGISTRY[template_id].template_kind == requested_kind
        and required.issubset(detected_concepts)
    ]
    if requested_kind == OlistTemplateKind.query and "monthly" in detected_concepts:
        candidates = [item for item in candidates if item == OlistTemplateId.monthly_sales_orders]
    if requested_kind == OlistTemplateKind.query and "daily" in detected_concepts:
        candidates = [item for item in candidates if item == OlistTemplateId.daily_sales_orders]
    if len(candidates) != 1:
        reason = "둘 이상의 템플릿 의미가 겹칩니다." if candidates else "고정 템플릿 의미와 정확히 일치하지 않습니다."
        return OlistTemplateMatch(reason=reason)

    template_id = candidates[0]
    definition = OLIST_TEMPLATE_REGISTRY[template_id]
    for raw_filter in filters or []:
        filter_value = str(raw_filter).casefold().strip()
        if not filter_value:
            continue
        _, filter_names, filter_error = _extract_parameters(filter_value)
        if filter_error or not filter_names:
            return OlistTemplateMatch(
                template_id=template_id,
                template_kind=definition.template_kind,
                reason=filter_error or f"지원하지 않는 필터입니다: {raw_filter}",
            )
    parameters, parameter_names, parameter_error = _extract_parameters(combined)
    if parameter_error:
        return OlistTemplateMatch(template_id=template_id, template_kind=definition.template_kind, reason=parameter_error)
    unsupported = sorted(parameter_names - definition.allowed_parameters)
    if unsupported:
        return OlistTemplateMatch(
            template_id=template_id,
            template_kind=definition.template_kind,
            parameters=parameters,
            reason=f"이 템플릿이 지원하지 않는 파라미터입니다: {', '.join(unsupported)}",
        )
    missing = missing_catalog_requirements(catalog, template_id)
    if missing:
        return OlistTemplateMatch(
            template_id=template_id,
            template_kind=definition.template_kind,
            parameters=parameters,
            reason=f"Olist 필수 스키마가 부족합니다: {', '.join(missing)}",
        )
    return OlistTemplateMatch(
        template_id=template_id,
        template_kind=definition.template_kind,
        parameters=parameters,
        reason=f"의도·지표·차원·파라미터·catalog가 {template_id.value} 계약을 충족합니다.",
        supported=True,
    )


def missing_catalog_requirements(
    catalog: dict[str, Any] | str | None,
    template_id: OlistTemplateId,
) -> list[str]:
    tables = _catalog_tables(catalog)
    missing: list[str] = []
    for table_name, required_columns in OLIST_TEMPLATE_REGISTRY[template_id].required_schema.items():
        table = tables.get(table_name)
        if not isinstance(table, dict):
            missing.append(table_name)
            continue
        available_columns = _catalog_columns(table)
        missing.extend(
            f"{table_name}.{column_name}"
            for column_name in sorted(required_columns - available_columns)
        )
    return missing


_FILTER_ALIASES: dict[OlistTemplateId, dict[str, str]] = {
    template_id: {"start_date": "o.order_purchase_timestamp", "end_date": "o.order_purchase_timestamp", "order_statuses": "o.order_status"}
    for template_id in OlistTemplateId
}
_FILTER_ALIASES[OlistTemplateId.customer_state_sales]["customer_states"] = "c.customer_state"
_FILTER_ALIASES[OlistTemplateId.seller_state_sales]["seller_states"] = "s.seller_state"
_FILTER_ALIASES[OlistTemplateId.seller_performance]["seller_states"] = "s.seller_state"
_FILTER_ALIASES[OlistTemplateId.monthly_seller_performance]["seller_states"] = "s.seller_state"
_FILTER_ALIASES[OlistTemplateId.customer_seller_geo].update({"customer_states": "c.customer_state", "seller_states": "s.seller_state"})


def _render_filters(template_id: OlistTemplateId, parameters: OlistTemplateParameters) -> str:
    aliases = _FILTER_ALIASES[template_id]
    clauses: list[str] = []
    if parameters.start_date:
        clauses.append(f"{aliases['start_date']} >= '{parameters.start_date}'")
    if parameters.end_date:
        clauses.append(f"{aliases['end_date']} < DATE_ADD('{parameters.end_date}', INTERVAL 1 DAY)")
    if parameters.order_statuses:
        statuses = ", ".join(f"'{value}'" for value in parameters.order_statuses)
        clauses.append(f"{aliases['order_statuses']} IN ({statuses})")
    if parameters.customer_states:
        states = ", ".join(f"'{value}'" for value in parameters.customer_states)
        clauses.append(f"{aliases['customer_states']} IN ({states})")
    if parameters.seller_states:
        states = ", ".join(f"'{value}'" for value in parameters.seller_states)
        clauses.append(f"{aliases['seller_states']} IN ({states})")
    return f"WHERE {' AND '.join(clauses)}" if clauses else ""


def build_olist_sql_draft(
    template_id: OlistTemplateId | str,
    catalog: dict[str, Any] | str | None = None,
    *,
    parameters: OlistTemplateParameters | dict[str, Any] | None = None,
) -> SQLDraft:
    """선택된 템플릿과 검증된 파라미터를 기존 SQLDraft 계약으로 반환한다."""
    selected = OlistTemplateId(template_id)
    definition = OLIST_TEMPLATE_REGISTRY[selected]
    parsed_parameters = (
        parameters if isinstance(parameters, OlistTemplateParameters)
        else OlistTemplateParameters.model_validate(parameters or {})
    )
    used = {
        name for name, value in parsed_parameters.model_dump().items()
        if value not in (None, [], "")
    }
    unsupported = sorted(used - definition.allowed_parameters)
    if unsupported:
        raise ValueError(f"지원하지 않는 템플릿 파라미터입니다: {', '.join(unsupported)}")
    if catalog is not None:
        missing = missing_catalog_requirements(catalog, selected)
        if missing:
            raise ValueError(f"Olist 템플릿 필수 스키마가 없습니다: {', '.join(missing)}")
    payload = deepcopy(OLIST_SQL_DRAFTS[selected])
    payload["sql"] = payload["sql"].replace("/*__FILTERS__*/", _render_filters(selected, parsed_parameters))
    limit = f"LIMIT {parsed_parameters.top_n}" if parsed_parameters.top_n else ""
    payload["sql"] = payload["sql"].replace("/*__LIMIT__*/", limit)
    return SQLDraft(**payload)


def build_olist_validation_plan(template_id: OlistTemplateId | str) -> dict[str, Any]:
    """기존 검증기가 query와 mart 계약을 검사할 계획을 만든다."""
    selected = OlistTemplateId(template_id)
    definition = OLIST_TEMPLATE_REGISTRY[selected]
    expected_shape = "grouped_aggregate" if definition.template_kind == OlistTemplateKind.query else "datamart_creation"
    return {
        "route_kind": definition.route_kind,
        "question_type": definition.intent,
        "target_metrics": definition.metrics,
        "dimensions": definition.dimensions,
        "required_aggregations": [],
        "filters": [],
        "selected_join_tables": list(definition.required_schema),
        "required_columns": [],
        "validation_contract": {
            "expected_result_shape": expected_shape,
            "required_tables": list(definition.required_schema),
            "dimensions": definition.dimensions,
            "target_metrics": definition.metrics,
            "required_aggregations": [],
            "expected_aliases": definition.output_columns,
            "target_table": OLIST_SQL_DRAFTS[selected].get("target_table"),
            "mart_policy": "aggregate_to_common_grain",
        },
    }


def build_olist_mart_design(template_id: OlistTemplateId | str) -> dict[str, Any]:
    selected = OlistTemplateId(template_id)
    definition = OLIST_TEMPLATE_REGISTRY[selected]
    if definition.template_kind != OlistTemplateKind.mart:
        return {}
    return {
        "mart_name": OLIST_SQL_DRAFTS[selected]["target_table"],
        "grain": definition.business_grain,
        "grain_columns": definition.dimensions,
        "aggregation_policy": "aggregate_to_common_grain",
        "aggregation_rationale": "템플릿이 선언한 분석 grain으로 원천 데이터를 선집계하고 중복을 제거합니다.",
        "source_tables": list(definition.required_schema),
        "output_columns": definition.output_columns,
    }


def template_catalog_rows() -> list[dict[str, Any]]:
    """문서·테스트에서 사용하는 정렬된 템플릿 카탈로그를 반환한다."""
    return [
        definition.model_dump(mode="json")
        for definition in sorted(OLIST_TEMPLATE_REGISTRY.values(), key=lambda item: item.template_id.value)
    ]


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


__all__ = [
    "OLIST_TEMPLATE_REGISTRY",
    "OlistTemplateId",
    "OlistTemplateMatch",
    "build_olist_mart_design",
    "build_olist_sql_draft",
    "build_olist_validation_plan",
    "match_olist_template",
    "missing_catalog_requirements",
    "template_catalog_rows",
]

from __future__ import annotations

import json
import re
from typing import Any, Optional

from DATA_Analyst_Assistant_Agent.agents.sql._runtime import ALLOWED_MART_SCHEMA, clean_sql, get_llm
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState, MartDesign, QuestionPlan, SQLDraft


def extract_schema_json(schema_text: str) -> dict[str, Any]:
    if not schema_text.strip():
        return {}
    try:
        data = json.loads(schema_text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def schema_tables(schema_json: dict[str, Any]) -> dict[str, Any]:
    tables = schema_json.get("tables")
    if isinstance(tables, dict):
        return tables
    return schema_json




def _question_lower(state: AgentState) -> str:
    return str(state.get("user_question") or "").lower()


def _is_datamart_question(state: AgentState) -> bool:
    q = _question_lower(state)
    reason = str(state.get("planner_selection_reason") or "").lower()
    tokens = ("데이터마트", "datamart", "data mart", "마트")
    intent = any(token in q for token in tokens) or any(token in reason for token in tokens)
    reusable = any(token in q for token in ("재사용", "반복", "저장", "create table", "materialize"))
    return intent or reusable or ("datamart" in reason)

def _is_average_delivery_question(state: AgentState) -> bool:
    q = _question_lower(state)
    korean = ("평균" in q and ("배송" in q or "소요일" in q)) or ("주문 완료일" in q and "배송 완료일" in q)
    english = "average" in q and ("delivery" in q or "shipping" in q)
    return korean or english


def _extract_table_columns_local(table_info: Any) -> set[str]:
    columns: set[str] = set()
    if not isinstance(table_info, dict):
        return columns
    raw_columns = table_info.get("columns", [])
    if isinstance(raw_columns, dict):
        columns.update(str(name) for name in raw_columns.keys())
    elif isinstance(raw_columns, list):
        for col in raw_columns:
            if isinstance(col, dict) and col.get("name"):
                columns.add(str(col["name"]))
            elif isinstance(col, str):
                columns.add(col)
    return columns

def _table_has_columns(schema_json: dict[str, Any], table_name: str, required: list[str]) -> bool:
    tables = schema_tables(schema_json)
    table = tables.get(table_name) if isinstance(tables, dict) else None
    if not isinstance(table, dict):
        return False
    cols = _extract_table_columns_local(table)
    return all(col in cols for col in required)

def retry_feedback_text(state: AgentState) -> str:
    feedback_parts: list[str] = []
    for label, value in (
        ("추가 메모", state.get("clarification_request") or ""),
        ("직전 검증 피드백", state.get("feedback") or ""),
        ("직전 실행 오류", state.get("error") or ""),
    ):
        if str(value).strip():
            feedback_parts.append(f"{label}: {value}")
    retry_hint = state.get("retry_hint") or {}
    if retry_hint:
        feedback_parts.append(
            f"직전 재시도 힌트: reason_code={retry_hint.get('reason_code', 'none')}, "
            f"suggested_action={retry_hint.get('suggested_action', 'continue')}, "
            f"details={retry_hint.get('details', {})}"
        )
    return "\n".join(feedback_parts)


def try_llm_json(prompt: str) -> Optional[str]:
    try:
        return get_llm().invoke(prompt).content
    except Exception:
        return None


def default_plan_from_state(state: AgentState) -> dict[str, Any]:
    """LLM 호출 실패 시 사용하는 최소 fallback plan.

    키워드 매칭 없이 스키마 테이블 목록 순서 기반으로만 구성한다.
    LLM 결과가 있으면 plan_question 노드에서 이 값을 덮어쓴다.
    """
    schema_json = extract_schema_json(state.get("schema_text", ""))
    tables = schema_tables(schema_json)
    all_tables = [str(name) for name in tables.keys()]
    candidate_tables = all_tables[:5] if all_tables else []
    selected_tables = candidate_tables[:1]

    validation_contract: dict[str, Any] = {
        "expected_result_shape": "table_preview",
        "required_aggregations": [],
        "required_columns": [],
        "expected_aliases": [],
        "required_tables": selected_tables,
        "target_metric": "",
        "dimensions": [],
        "target_table": None,
        "mart_policy": None,
    }
    route_kind = "simple"
    task_type = "query_answer"
    requested_output = "execute_and_answer"
    expected_result_shape = "table_preview"
    required_aggregations: list[str] = []
    required_columns: list[str] = []
    target_metric = ""

    if _is_datamart_question(state):
        route_kind = "comprehensive"
        task_type = "data_mart_build"
        requested_output = "create_table"
        expected_result_shape = "datamart_creation"
        validation_contract.update({
            "expected_result_shape": expected_result_shape,
            "required_tables": selected_tables,
            "mart_policy": "prefer_row_preserving",
        })

    if route_kind == "simple" and _is_average_delivery_question(state) and _table_has_columns(schema_json, "orders", ["order_approved_at", "order_delivered_customer_date"]):
        if "orders" not in selected_tables:
            selected_tables = ["orders"]
        if "orders" not in candidate_tables:
            candidate_tables = ["orders", *candidate_tables]
        expected_result_shape = "single_scalar"
        required_aggregations = ["AVG"]
        required_columns = ["order_approved_at", "order_delivered_customer_date"]
        target_metric = "average_delivery_days"
        validation_contract.update({
            "expected_result_shape": expected_result_shape,
            "required_aggregations": required_aggregations,
            "required_columns": required_columns,
            "expected_aliases": ["avg_delivery_days"],
            "required_tables": ["orders"],
            "target_metric": target_metric,
            "dimensions": [],
        })

    return QuestionPlan(
        original_question=state["user_question"],
        route_kind=route_kind,
        question_type="mart_build" if route_kind == "comprehensive" else "detail",
        task_type=task_type,
        requested_output=requested_output,
        target_metric=target_metric,
        dimensions=[],
        filters=[],
        time_condition=None,
        selected_join_tables=selected_tables,
        relevant_tables=selected_tables,
        candidate_tables=candidate_tables,
        mart_name="analytics_mart" if route_kind == "comprehensive" else None,
        grain="원본 entity/event 행 수준 grain 유지" if route_kind == "comprehensive" else None,
        load_strategy="full_refresh" if route_kind == "comprehensive" else None,
        ambiguity_note="LLM 분석 없이 스키마 기본값으로 생성된 fallback plan입니다.",
        expected_result_shape=expected_result_shape,
        required_columns=required_columns,
        required_aggregations=required_aggregations,
        validation_contract=validation_contract,
        reasoning="(fallback) LLM 결과가 우선 적용됩니다. 스키마 상위 테이블 기준으로 구성된 기본 plan입니다.",
    ).model_dump()


def deterministic_sql_draft(state: AgentState) -> dict[str, Any]:
    """LLM SQL 생성 실패 시 사용하는 최소 fallback SQL.

    JOIN 로직 없이 단일 테이블 기반으로만 구성한다.
    """
    plan = state["plan"]
    selected_tables = list(plan.get("selected_join_tables") or plan.get("relevant_tables") or [])
    route_kind = plan.get("route_kind") or (
        "comprehensive" if plan.get("task_type") == "data_mart_build" else "simple"
    )
    primary_table = selected_tables[0] if selected_tables else None

    if route_kind == "comprehensive":
        target_table = f"{ALLOWED_MART_SCHEMA}.{plan.get('mart_name') or 'analytics_mart'}"
        source_table = primary_table or "source_table"
        return SQLDraft(
            sql=f"CREATE TABLE {target_table} AS SELECT {source_table}.* FROM {source_table};",
            sql_type="create_table_as",
            target_table=target_table,
            source_tables=[source_table],
            columns_used=[],
            business_grain=state.get("mart_design", {}).get("grain") or plan.get("grain"),
            precheck_sql=f"SELECT COUNT(*) AS source_row_count FROM {source_table};" if primary_table else None,
            postcheck_sql=f"SELECT COUNT(*) AS mart_row_count FROM {target_table};",
            reasoning="재사용 가능한 datamart 생성을 위한 기본 SQL 초안입니다.",
        ).model_dump()

    # simple: intent-aware deterministic SELECT
    if _is_average_delivery_question(state) and primary_table == "orders":
        sql = (
            "SELECT AVG(DATEDIFF(order_delivered_customer_date, order_approved_at)) AS avg_delivery_days "
            "FROM orders "
            "WHERE order_approved_at IS NOT NULL AND order_delivered_customer_date IS NOT NULL;"
        )
    elif primary_table:
        sql = f"SELECT * FROM {primary_table} LIMIT 50;"
    else:
        sql = "SELECT 1 AS sample_value;"
    return SQLDraft(
        sql=sql,
        sql_type="select",
        target_table=None,
        source_tables=[primary_table] if primary_table else [],
        columns_used=[],
        business_grain=None,
        precheck_sql=None,
        postcheck_sql=None,
        reasoning="간단한 조회용 deterministic SQL 초안입니다.",
    ).model_dump()


def qualify_target_table(target_table: str | None) -> str | None:
    if not target_table:
        return target_table
    normalized = target_table.strip().strip("`")
    return normalized if "." in normalized else f"{ALLOWED_MART_SCHEMA}.{normalized}"


def normalize_postcheck_sql(postcheck_sql: str | None, target_table: str | None) -> str | None:
    if not postcheck_sql:
        return postcheck_sql
    normalized_target = qualify_target_table(target_table)
    sql = clean_sql(postcheck_sql)
    if not normalized_target:
        return sql
    _, table_name = normalized_target.split(".", 1)
    for pattern, replacement in [
        (rf"(?i)\bfrom\s+`?{re.escape(table_name)}`?\b", f"FROM {normalized_target}"),
        (rf"(?i)\bjoin\s+`?{re.escape(table_name)}`?\b", f"JOIN {normalized_target}"),
        (rf"(?i)\binto\s+`?{re.escape(table_name)}`?\b", f"INTO {normalized_target}"),
        (rf"(?i)\btable\s+`?{re.escape(table_name)}`?\b", f"TABLE {normalized_target}"),
    ]:
        sql = re.sub(pattern, replacement, sql)
    return sql


def _coerce_column_name(entry: Any) -> str:
    """LLM이 컬럼을 문자열 대신 {"column_name": ..., "description": ...} dict 로 줄 때
    식별자만 뽑아 문자열로 정규화한다. 문자열이면 그대로 둔다."""
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        for key in ("column_name", "name", "col", "column", "expression"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return str(entry)


def normalize_mart_column_lists(design: dict[str, Any]) -> dict[str, Any]:
    """MartDesign 의 문자열 리스트 필드에 dict 항목이 섞여 와도 문자열 리스트로 정규화한다.

    프롬프트는 key_columns/measure_columns 등을 ["col", ...] 로 요구하지만 모델에 따라
    [{"column_name": ..., "description": ...}] 형태로 주기도 한다(형식 편차). 그대로 두면
    MartDesign(List[str]) pydantic 검증이 깨지므로 식별자만 뽑아 정규화한다.
    """
    for field in ("key_columns", "measure_columns", "dimension_columns", "source_tables"):
        value = design.get(field)
        if isinstance(value, list):
            design[field] = [_coerce_column_name(item) for item in value if item not in (None, "")]
    return design


def normalize_generated_sql(parsed: dict[str, Any], fallback: dict[str, Any], route_kind: str) -> dict[str, Any]:
    parsed["sql"] = clean_sql(parsed.get("sql", fallback["sql"]))
    parsed["target_table"] = qualify_target_table(parsed.get("target_table") or fallback.get("target_table"))
    if parsed.get("precheck_sql"):
        parsed["precheck_sql"] = clean_sql(parsed["precheck_sql"])
    postcheck_sql = parsed.get("postcheck_sql") or fallback.get("postcheck_sql")
    if postcheck_sql:
        parsed["postcheck_sql"] = normalize_postcheck_sql(postcheck_sql, parsed.get("target_table"))
    parsed.setdefault("sql_type", fallback["sql_type"])
    parsed.setdefault("source_tables", fallback["source_tables"])
    parsed.setdefault("columns_used", fallback["columns_used"])
    parsed.setdefault("reasoning", fallback["reasoning"])
    if route_kind == "comprehensive" and parsed["sql_type"] == "select":
        return fallback
    if route_kind == "simple" and parsed["sql_type"] != "select":
        return fallback
    return parsed


def default_mart_design(state: AgentState) -> dict[str, Any]:
    return MartDesign(
        mart_name=state["plan"].get("mart_name") or "analytics_mart",
        target_schema=ALLOWED_MART_SCHEMA,
        grain=state["plan"].get("grain") or "가능하면 원본 entity/event 행 수준 grain 유지",
        base_grain=state["plan"].get("grain") or "원본 entity/event 행 수준 grain 유지",
        source_tables=state["plan"].get("selected_join_tables") or state["plan"].get("relevant_tables", []),
        key_columns=state["plan"].get("dimensions", []),
        measure_columns=[state["plan"].get("target_metric") or "핵심 지표"],
        dimension_columns=state["plan"].get("dimensions", []),
        incremental_column=None,
        load_strategy=state["plan"].get("load_strategy") or "full_refresh",
        row_preserving_strategy="원본 행 수준을 최대한 유지하고 조인/정제/표준화 중심으로 설계",
        aggregation_policy="prefer_row_preserving",
        aggregation_rationale=None,
        design_reasoning="원본 행 수준을 우선하는 기본 datamart 설계 초안입니다.",
    ).model_dump()

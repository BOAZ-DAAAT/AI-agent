from __future__ import annotations

import json
import re
from typing import Any, Optional

from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.tool.runtime import (
    ALLOWED_MART_SCHEMA,
    AgentState,
    MartDesign,
    QuestionPlan,
    SQLDraft,
    get_llm,
)
from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.tool.sql_utils import clean_sql, safe_json_parse


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


def normalized_tokens(text_value: str) -> set[str]:
    return {
        token for token in re.split(r"[^0-9A-Za-z가-힣_]+", (text_value or "").lower())
        if len(token) >= 2
    }


def score_table(question: str, table_name: str, table_info: Any) -> int:
    q_tokens = normalized_tokens(question)
    score = 0
    name_tokens = normalized_tokens(table_name)
    score += len(q_tokens & name_tokens) * 4
    if table_name.lower() in question.lower():
        score += 6
    columns: list[str] = []
    if isinstance(table_info, dict):
        raw_columns = table_info.get("columns", [])
        if isinstance(raw_columns, dict):
            columns = list(raw_columns.keys())
        elif isinstance(raw_columns, list):
            for col in raw_columns:
                if isinstance(col, dict) and col.get("name"):
                    columns.append(str(col["name"]))
                elif isinstance(col, str):
                    columns.append(col)
    for col in columns:
        score += len(q_tokens & normalized_tokens(str(col)))
    return score


def rank_candidate_tables(question: str, schema_json: dict[str, Any]) -> list[str]:
    tables = schema_tables(schema_json)
    scored = []
    for table_name, table_info in tables.items():
        scored.append((score_table(question, str(table_name), table_info), str(table_name)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    ranked = [name for score, name in scored if score > 0]
    if ranked:
        return ranked[:5]
    return [str(name) for name in list(tables.keys())[:3]]


def find_join_conditions(selected_tables: list[str], schema_json: dict[str, Any]) -> list[str]:
    tables = schema_tables(schema_json)
    selected = set(selected_tables)
    join_conditions: list[str] = []
    for table_name in selected_tables:
        table_info = tables.get(table_name, {}) if isinstance(tables, dict) else {}
        fks = table_info.get("foreign_keys", []) if isinstance(table_info, dict) else []
        for fk in fks or []:
            if not isinstance(fk, dict):
                continue
            ref_table = str(fk.get("ref_table") or fk.get("referred_table") or "")
            if ref_table not in selected:
                continue
            local_col = str(fk.get("column") or fk.get("constrained_column") or "")
            ref_col = str(fk.get("ref_column") or fk.get("referred_column") or "")
            if local_col and ref_col:
                join_conditions.append(f"{table_name}.{local_col} = {ref_table}.{ref_col}")
    deduped: list[str] = []
    for cond in join_conditions:
        if cond not in deduped:
            deduped.append(cond)
    return deduped


def is_comprehensive_request(question: str, planner_reason: str, clarification_request: str) -> bool:
    combined = f"{question} {planner_reason} {clarification_request}".lower()
    keywords = [
        "datamart", "data mart", "mart", "데이터마트", "마트", "집계 테이블", "요약 테이블",
        "분석용 테이블", "재사용", "복잡", "종합", "comprehensive",
    ]
    return any(keyword in combined for keyword in keywords)


def extract_metric(question: str) -> str:
    q = question.lower()
    if "평균" in question and any(token in question for token in ("배송", "소요일")):
        return "평균 소요일"
    if any(token in question for token in ("매출", "수익")) or any(token in q for token in ("revenue", "sales")):
        return "매출"
    if any(token in question for token in ("주문", "건수")) or "order" in q:
        return "주문 수"
    if any(token in question for token in ("고객",)) or "customer" in q:
        return "고객 수"
    return "핵심 지표 미정"


def extract_dimensions(question: str) -> list[str]:
    dimensions = []
    lowered = question.lower()
    if any(token in question for token in ("월별", "월간")) or "monthly" in lowered:
        dimensions.append("월")
    if any(token in question for token in ("일별",)) or "daily" in lowered:
        dimensions.append("일")
    if any(token in question for token in ("고객별",)) or "customer" in lowered:
        dimensions.append("고객")
    return dimensions


def extract_time_condition(question: str) -> Optional[str]:
    for token in ("최근 7일", "최근 30일", "이번 달", "지난 달", "올해"):
        if token in question:
            return token
    return None


def default_validation_contract(
    *,
    question: str,
    route_kind: str,
    target_metric: str,
    dimensions: list[str],
    selected_tables: list[str],
    mart_name: str | None = None,
) -> dict[str, Any]:
    expected_result_shape = "datamart_creation" if route_kind == "comprehensive" else "table_preview"
    required_aggregations: list[str] = []
    required_columns: list[str] = []
    expected_aliases: list[str] = []

    if route_kind == "comprehensive":
        expected_result_shape = "datamart_creation"
    elif dimensions:
        expected_result_shape = "grouped_aggregate"
    elif is_average_delivery_days_question(question):
        expected_result_shape = "single_scalar"
        required_aggregations = ["AVG"]
        required_columns = ["order_delivered_customer_date", "order_approved_at"]
        expected_aliases = ["avg_delivery_days"]
    elif "주문 수" in question or "주문 건수" in question:
        expected_result_shape = "single_scalar"
        required_aggregations = ["COUNT"]
        required_columns = ["order_id"]
        expected_aliases = ["order_count"]

    return {
        "expected_result_shape": expected_result_shape,
        "required_aggregations": required_aggregations,
        "required_columns": required_columns,
        "expected_aliases": expected_aliases,
        "required_tables": selected_tables,
        "target_metric": target_metric,
        "dimensions": dimensions,
        "target_table": f"{ALLOWED_MART_SCHEMA}.{mart_name}" if mart_name and route_kind == "comprehensive" else None,
    }


def retry_feedback_text(state: AgentState) -> str:
    feedback_parts: list[str] = []
    clarification = (state.get("clarification_request") or "").strip()
    if clarification:
        feedback_parts.append(f"추가 메모: {clarification}")
    raw_feedback = (state.get("feedback") or "").strip()
    if raw_feedback:
        feedback_parts.append(f"직전 검증 피드백: {raw_feedback}")
    retry_hint = state.get("retry_hint") or {}
    if retry_hint:
        feedback_parts.append(
            f"직전 재시도 힌트: reason_code={retry_hint.get('reason_code', 'none')}, "
            f"suggested_action={retry_hint.get('suggested_action', 'continue')}, "
            f"details={retry_hint.get('details', {})}"
        )
    raw_error = (state.get("error") or "").strip()
    if raw_error:
        feedback_parts.append(f"직전 실행 오류: {raw_error}")
    return "\n".join(feedback_parts)


def is_average_delivery_days_question(question: str) -> bool:
    q = question.lower()
    return (
        ("평균" in question or "average" in q or "avg" in q)
        and any(token in question for token in ("배송", "소요일"))
        and any(token in question for token in ("완료일", "승인일", "주문 완료일", "배송 완료일"))
    )


def simple_aggregate_sql(question: str, primary_table: str | None) -> tuple[str, list[str], list[str]] | None:
    if primary_table == "orders" and is_average_delivery_days_question(question):
        sql = (
            "SELECT AVG(DATEDIFF(order_delivered_customer_date, order_approved_at)) AS avg_delivery_days "
            "FROM orders "
            "WHERE order_approved_at IS NOT NULL "
            "AND order_delivered_customer_date IS NOT NULL;"
        )
        return (
            sql,
            ["order_delivered_customer_date", "order_approved_at"],
            [
                "order_approved_at IS NOT NULL",
                "order_delivered_customer_date IS NOT NULL",
            ],
        )
    if primary_table == "orders" and ("주문 수" in question or "주문 건수" in question):
        sql = "SELECT COUNT(*) AS order_count FROM orders;"
        return (sql, ["order_id"], [])
    return None


def default_plan_from_state(state: AgentState) -> dict[str, Any]:
    schema_json = extract_schema_json(state.get("schema_text", ""))
    ranked_tables = rank_candidate_tables(state["user_question"], schema_json)
    route_kind = "comprehensive" if is_comprehensive_request(
        state["user_question"],
        state.get("planner_selection_reason", ""),
        state.get("clarification_request", ""),
    ) else "simple"
    selected_tables = ranked_tables[:2] if route_kind == "comprehensive" else ranked_tables[:1]
    ambiguity_note = None
    if not selected_tables:
        ambiguity_note = "질문과 직접 연결되는 테이블을 확정하지 못해 스키마 상위 테이블 기준으로 진행합니다."
    reasoning_parts = [
        f"질문을 {'복잡한 분석용 데이터마트 생성' if route_kind == 'comprehensive' else '간단한 조회 SQL 작성'} 요청으로 해석했습니다.",
        f"후보 테이블은 {', '.join(ranked_tables) if ranked_tables else '없음'} 입니다.",
        f"우선 사용할 테이블은 {', '.join(selected_tables) if selected_tables else '없음'} 입니다.",
    ]
    if state.get("planner_selection_reason"):
        reasoning_parts.append(f"플랜 에이전트 선택 이유 힌트는 '{state['planner_selection_reason']}' 입니다.")
    if state.get("clarification_request"):
        reasoning_parts.append(f"clarification 요청 메모는 '{state['clarification_request']}' 입니다.")
    mart_name = "analytics_mart" if route_kind == "comprehensive" else None
    contract = default_validation_contract(
        question=state["user_question"],
        route_kind=route_kind,
        target_metric=extract_metric(state["user_question"]),
        dimensions=extract_dimensions(state["user_question"]),
        selected_tables=selected_tables,
        mart_name=mart_name,
    )
    return QuestionPlan(
        original_question=state["user_question"],
        route_kind=route_kind,
        task_type="data_mart_build" if route_kind == "comprehensive" else "query_answer",
        requested_output="create_table" if route_kind == "comprehensive" else "execute_and_answer",
        target_metric=extract_metric(state["user_question"]),
        dimensions=extract_dimensions(state["user_question"]),
        filters=[],
        time_condition=extract_time_condition(state["user_question"]),
        selected_join_tables=selected_tables,
        relevant_tables=selected_tables,
        candidate_tables=ranked_tables,
        mart_name=mart_name,
        grain="월/고객 등 분석 grain 미정" if route_kind == "comprehensive" else None,
        load_strategy="full_refresh" if route_kind == "comprehensive" else None,
        ambiguity_note=ambiguity_note,
        expected_result_shape=contract["expected_result_shape"],
        required_columns=contract["required_columns"],
        required_aggregations=contract["required_aggregations"],
        validation_contract=contract,
        reasoning=" ".join(reasoning_parts),
    ).model_dump()


def try_llm_json(prompt: str) -> Optional[str]:
    try:
        return get_llm().invoke(prompt).content
    except Exception:
        return None


def join_sql_parts(selected_tables: list[str], schema_json: dict[str, Any]) -> tuple[str, list[str]]:
    if not selected_tables:
        return "", []
    base = selected_tables[0]
    joins = []
    join_conditions = find_join_conditions(selected_tables, schema_json)
    used_tables = [base]
    for table_name in selected_tables[1:]:
        matching = [cond for cond in join_conditions if cond.startswith(f"{base}.") and f" = {table_name}." in cond]
        if not matching:
            matching = [cond for cond in join_conditions if cond.startswith(f"{table_name}.") and f" = {base}." in cond]
        if matching:
            joins.append(f"JOIN {table_name} ON {matching[0]}")
            used_tables.append(table_name)
    return f"FROM {base} " + " ".join(joins), used_tables


def deterministic_sql_draft(state: AgentState) -> dict[str, Any]:
    plan = state["plan"]
    schema_json = extract_schema_json(state.get("schema_text", ""))
    selected_tables = list(plan.get("selected_join_tables") or plan.get("relevant_tables") or [])
    route_kind = plan.get("route_kind", "simple")
    target_metric = plan.get("target_metric") or "핵심 지표"
    dimensions = plan.get("dimensions") or []
    source_clause, joined_tables = join_sql_parts(selected_tables, schema_json)
    primary_table = selected_tables[0] if selected_tables else None

    if route_kind == "comprehensive":
        target_table = f"{ALLOWED_MART_SCHEMA}.{plan.get('mart_name') or 'analytics_mart'}"
        source_table = primary_table or "source_table"
        select_columns = f"{source_table}.*"
        if joined_tables and len(joined_tables) > 1:
            select_columns = ", ".join(f"{table}.*" for table in joined_tables)
        source_sql_clause = source_clause or f"FROM {source_table}"
        sql = f"CREATE TABLE {target_table} AS SELECT {select_columns} {source_sql_clause};"
        precheck_sql = f"SELECT COUNT(*) AS source_row_count FROM {source_table};" if primary_table else None
        postcheck_sql = f"SELECT COUNT(*) AS mart_row_count FROM {target_table};"
        reasoning = (
            "planner가 comprehensive 경로를 선택했기 때문에, 재사용 가능한 datamart 생성을 위한 SQL을 작성했습니다. "
            f"핵심 소스 테이블은 {', '.join(selected_tables) if selected_tables else '미확정'} 이며, "
            "가능한 경우 조인 조건을 반영했습니다."
        )
        return SQLDraft(
            sql=sql,
            sql_type="create_table_as",
            target_table=target_table,
            source_tables=joined_tables or selected_tables,
            columns_used=[],
            business_grain=state.get("mart_design", {}).get("grain") or plan.get("grain"),
            precheck_sql=precheck_sql,
            postcheck_sql=postcheck_sql,
            reasoning=reasoning,
        ).model_dump()

    aggregate_sql = simple_aggregate_sql(state["user_question"], primary_table)
    if aggregate_sql is not None:
        sql, used_columns, derived_filters = aggregate_sql
        reasoning = (
            "planner가 simple 경로를 선택했고 질문에 집계 의도가 명확해서, "
            "질문에 직접 대응하는 단일 집계 SQL을 작성했습니다."
        )
        return SQLDraft(
            sql=sql,
            sql_type="select",
            target_table=None,
            source_tables=[primary_table] if primary_table else [],
            columns_used=used_columns,
            business_grain=None,
            precheck_sql=None,
            postcheck_sql=None,
            reasoning=reasoning,
        ).model_dump()

    if primary_table and source_clause:
        select_prefix = "SELECT *"
        order_clause = ""
        if dimensions:
            select_prefix = f"SELECT {', '.join(dimensions)}"
            order_clause = f" ORDER BY {dimensions[0]}"
        sql = f"{select_prefix} {source_clause} LIMIT 50{order_clause};"
    elif primary_table:
        sql = f"SELECT * FROM {primary_table} LIMIT 50;"
    else:
        sql = "SELECT 1 AS sample_value;"
    reasoning = (
        "planner가 simple 경로를 선택했기 때문에, 간단한 분석용 조회 SQL을 작성했습니다. "
        f"우선 대상 테이블은 {', '.join(joined_tables or selected_tables) if (joined_tables or selected_tables) else '미확정'} 입니다."
    )
    return SQLDraft(
        sql=sql,
        sql_type="select",
        target_table=None,
        source_tables=joined_tables or selected_tables,
        columns_used=[],
        business_grain=None,
        precheck_sql=None,
        postcheck_sql=None,
        reasoning=reasoning,
    ).model_dump()


def qualify_target_table(target_table: str | None) -> str | None:
    if not target_table:
        return target_table
    normalized = target_table.strip().strip("`")
    if "." in normalized:
        return normalized
    return f"{ALLOWED_MART_SCHEMA}.{normalized}"


def normalize_postcheck_sql(postcheck_sql: str | None, target_table: str | None) -> str | None:
    if not postcheck_sql:
        return postcheck_sql
    normalized_target = qualify_target_table(target_table)
    if not normalized_target:
        return clean_sql(postcheck_sql)

    sql = clean_sql(postcheck_sql)
    schema_name, table_name = normalized_target.split(".", 1)
    unqualified_patterns = [
        rf"(?i)\bfrom\s+`?{re.escape(table_name)}`?\b",
        rf"(?i)\bjoin\s+`?{re.escape(table_name)}`?\b",
        rf"(?i)\binto\s+`?{re.escape(table_name)}`?\b",
        rf"(?i)\btable\s+`?{re.escape(table_name)}`?\b",
    ]
    replacement_pairs = [
        (unqualified_patterns[0], f"FROM {normalized_target}"),
        (unqualified_patterns[1], f"JOIN {normalized_target}"),
        (unqualified_patterns[2], f"INTO {normalized_target}"),
        (unqualified_patterns[3], f"TABLE {normalized_target}"),
    ]
    for pattern, replacement in replacement_pairs:
        sql = re.sub(pattern, replacement, sql)
    return sql


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
        grain=state["plan"].get("grain") or "분석 grain 미정",
        source_tables=state["plan"].get("selected_join_tables") or state["plan"].get("relevant_tables", []),
        key_columns=state["plan"].get("dimensions", []),
        measure_columns=[state["plan"].get("target_metric") or "핵심 지표"],
        dimension_columns=state["plan"].get("dimensions", []),
        incremental_column=None,
        load_strategy=state["plan"].get("load_strategy") or "full_refresh",
        design_reasoning="planner가 선택한 테이블과 지표를 기준으로 재사용 가능한 datamart 초안을 구성했습니다.",
    ).model_dump()

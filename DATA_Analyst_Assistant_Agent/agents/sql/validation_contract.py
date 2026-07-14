from __future__ import annotations

import re
from typing import Any

from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import (
    extract_schema_json,
    normalize_sql_draft_columns,
    schema_tables,
)
from DATA_Analyst_Assistant_Agent.agents.sql.self_check import mysql_dialect_error
from DATA_Analyst_Assistant_Agent.agents.sql.sql_text import split_sql_statements


def _normalized_sql(sql: str) -> str:
    return (sql or "").replace("```sql", "").replace("```", "").strip()


def _normalized_upper_sql(sql: str) -> str:
    return _normalized_sql(sql).upper()


def build_intent_contract(plan: dict[str, Any]) -> dict[str, Any]:
    contract = dict(plan.get("validation_contract") or {})
    contract.setdefault("expected_result_shape", plan.get("expected_result_shape") or "table_preview")
    contract.setdefault("required_columns", list(plan.get("required_columns") or []))
    contract.setdefault("required_aggregations", list(plan.get("required_aggregations") or []))
    contract.setdefault("required_tables", list(plan.get("selected_join_tables") or plan.get("relevant_tables") or []))
    contract.setdefault("dimensions", list(plan.get("dimensions") or []))
    contract.setdefault("target_metric", plan.get("target_metric") or "")
    contract.setdefault("expected_aliases", [])
    contract.setdefault("target_table", None)
    return contract


def validate_sql_dialect_and_route(plan: dict[str, Any], sql_draft: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    sql = sql_draft.get("sql") or ""
    statements = split_sql_statements(sql)
    route_kind = plan.get("route_kind") or ("comprehensive" if plan.get("task_type") == "data_mart_build" else "simple")
    sql_type = sql_draft.get("sql_type", "select")
    for index, statement in enumerate(statements):
        dialect_issue = mysql_dialect_error(statement)
        if dialect_issue:
            findings.append({
                "category": "mysql_dialect_error",
                "severity": "error",
                "retryable": True,
                "detail": f"{index + 1}번 statement: {dialect_issue}",
            })
    if route_kind == "simple" and sql_type != "select":
        findings.append({"category": "route_kind_mismatch", "severity": "error", "retryable": True, "detail": "simple 경로에서는 조회 SQL만 허용됩니다."})
    if route_kind == "simple":
        for index, statement in enumerate(statements):
            first_word = _normalized_sql(statement).split()[0].upper() if _normalized_sql(statement) else ""
            if first_word not in {"SELECT", "WITH"}:
                findings.append({
                    "category": "route_kind_mismatch",
                    "severity": "error",
                    "retryable": True,
                    "detail": f"simple 경로의 {index + 1}번 statement는 SELECT/WITH로 시작해야 합니다.",
                })
    if route_kind == "comprehensive" and sql_type == "select":
        findings.append({"category": "route_kind_mismatch", "severity": "error", "retryable": True, "detail": "comprehensive 경로에서는 datamart 생성 SQL이 필요합니다."})
    if route_kind == "comprehensive" and len(statements) > 1:
        findings.append({
            "category": "route_kind_mismatch",
            "severity": "error",
            "retryable": True,
            "detail": "comprehensive 경로에서는 단일 datamart 생성 statement만 허용됩니다.",
        })
    return findings


def validate_sql_intent(plan: dict[str, Any], sql_draft: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    sql = _normalized_upper_sql(sql_draft.get("sql") or "")
    contract = build_intent_contract(plan)
    if contract.get("expected_result_shape") == "datamart_creation":
        return findings
    for agg in contract.get("required_aggregations", []):
        if agg.upper() not in sql:
            findings.append({"category": "intent_mismatch", "severity": "error", "retryable": True, "detail": f"질문 의도상 필요한 집계 함수 {agg} 가 SQL에 없습니다."})
    dimensions = [str(d) for d in contract.get("dimensions", []) if d]
    if contract.get("expected_result_shape") == "grouped_aggregate" and dimensions and "GROUP BY" not in sql:
        findings.append({"category": "result_shape_mismatch", "severity": "error", "retryable": True, "detail": "그룹 집계 질문인데 GROUP BY가 없습니다."})
    required_tables = [str(t) for t in contract.get("required_tables", []) if t]
    source_tables = [str(t) for t in sql_draft.get("source_tables", []) if t]
    sql_lower = _normalized_sql(sql_draft.get("sql") or "").lower()
    for table_name in required_tables:
        if table_name not in source_tables and table_name.lower() not in sql_lower:
            findings.append({"category": "invalid_join_plan", "severity": "warning", "retryable": True, "detail": f"planner가 선택한 핵심 테이블 {table_name} 이 SQL에 반영되지 않았습니다."})
    return findings


def validate_datamart_reusability(plan: dict[str, Any], mart_design: dict[str, Any], sql_draft: dict[str, Any]) -> list[dict[str, Any]]:
    contract = build_intent_contract(plan)
    if contract.get("expected_result_shape") != "datamart_creation":
        return []
    findings: list[dict[str, Any]] = []
    sql = _normalized_upper_sql(sql_draft.get("sql") or "")
    has_aggregate_summary = any(token in sql for token in ("GROUP BY", "HAVING", "COUNT(", "SUM(", "AVG(", "MIN(", "MAX("))
    if not has_aggregate_summary:
        return findings
    policy = str((mart_design or {}).get("aggregation_policy") or contract.get("mart_policy") or "prefer_row_preserving")
    rationale = " ".join(str(value or "") for value in ((mart_design or {}).get("aggregation_rationale"), (mart_design or {}).get("design_reasoning"), sql_draft.get("reasoning")))
    has_justification = any(token in rationale for token in ("원본 행", "행 수준", "row-level", "row level", "불가피", "정당화", "예외"))
    base_grain = str((mart_design or {}).get("base_grain") or (mart_design or {}).get("grain") or "")
    if policy == "prefer_row_preserving" and not has_justification:
        findings.append({"category": "mart_summary_bias", "severity": "error", "retryable": True, "detail": "datamart가 재사용 가능한 기반 테이블보다 질문 전용 요약 결과에 가깝습니다. 원본 행 수준 유지 전략 또는 집계 정당화가 필요합니다."})
    elif not base_grain.strip():
        findings.append({"category": "mart_grain_missing", "severity": "warning", "retryable": True, "detail": "datamart 설계에 base grain 설명이 없습니다."})
    return findings


def validate_sql_identifiers(plan: dict[str, Any], sql_draft: dict[str, Any], schema_text: str) -> list[dict[str, Any]]:
    sql_draft = normalize_sql_draft_columns(sql_draft)
    schema_json = extract_schema_json(schema_text)
    tables = schema_tables(schema_json) if schema_json else {}
    findings: list[dict[str, Any]] = []
    if not isinstance(tables, dict) or not tables:
        return findings
    available_tables = set(str(name) for name in tables.keys())
    available_columns = {str(table_name): _extract_table_columns(table_info) for table_name, table_info in tables.items()}
    sql = _normalized_sql(sql_draft.get("sql") or "")
    cte_names = _extract_cte_names(sql)
    source_tables = [str(t) for t in sql_draft.get("source_tables", []) if t]
    required_tables = [str(t) for t in build_intent_contract(plan).get("required_tables", []) if t]
    candidate_tables = list(dict.fromkeys(source_tables + required_tables))
    for table_name in candidate_tables:
        bare_name = table_name.split(".")[-1]
        if bare_name not in available_tables and table_name not in available_tables:
            findings.append({"category": "missing_table", "severity": "error", "retryable": False, "detail": f"테이블 {table_name} 이(가) 제공된 스키마에 없습니다."})
    for column_name in [str(c) for c in sql_draft.get("source_column_refs", []) if c]:
        bare_column_name = column_name.split(".")[-1]
        if not any(bare_column_name in cols for cols in available_columns.values()):
            findings.append({"category": "missing_column", "severity": "error", "retryable": False, "detail": f"컬럼 {column_name} 이(가) 제공된 스키마에 없습니다."})
    sql_lower = sql.lower()
    for ref in re.findall(r"(?:from|join|into|table)\s+([a-zA-Z_][a-zA-Z0-9_\\.]*)", sql_lower):
        bare_name = ref.split(".")[-1]
        if bare_name in cte_names:
            continue
        if bare_name not in available_tables and not ref.startswith("analytics."):
            findings.append({"category": "missing_table", "severity": "error", "retryable": False, "detail": f"SQL이 참조한 테이블 {ref} 이(가) 제공된 스키마에 없습니다."})
    return _dedupe_findings(findings)


def validate_result_shape(plan: dict[str, Any], sql_draft: dict[str, Any], sql_result: Any, row_count: int, postcheck_result: Any = None) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    contract = build_intent_contract(plan)
    expected = contract.get("expected_result_shape")
    rows = list(sql_result or [])
    first_row = rows[0] if rows else None
    aliases = [str(alias) for alias in contract.get("expected_aliases", []) if alias]
    first_row_mapping = dict(first_row._mapping) if hasattr(first_row, "_mapping") else {}
    if expected == "single_scalar":
        if row_count != 1:
            findings.append({"category": "result_shape_mismatch", "severity": "error", "retryable": True, "detail": f"단일 집계 결과는 1행이어야 하는데 {row_count}행입니다."})
        if aliases and first_row_mapping and not any(alias in first_row_mapping for alias in aliases):
            findings.append({"category": "result_shape_mismatch", "severity": "warning", "retryable": True, "detail": f"기대 alias {aliases} 가 결과 컬럼에 없습니다."})
        if first_row is None:
            findings.append({"category": "empty_result", "severity": "error", "retryable": True, "detail": "단일 집계 결과가 비어 있습니다."})
    if expected == "grouped_aggregate" and row_count <= 0:
        findings.append({"category": "empty_result", "severity": "error", "retryable": True, "detail": "그룹 집계 결과가 비어 있습니다."})
    if expected == "datamart_creation":
        target_table = sql_draft.get("target_table")
        if not target_table:
            findings.append({"category": "postcheck_failed", "severity": "error", "retryable": True, "detail": "datamart 생성인데 target_table 이 없습니다."})
        if postcheck_result is None or postcheck_result == []:
            findings.append({"category": "postcheck_failed", "severity": "warning", "retryable": True, "detail": "datamart postcheck 결과가 없습니다."})
    return findings


def summarize_validation(findings: list[dict[str, Any]]) -> dict[str, Any]:
    has_error = any(item.get("severity") == "error" for item in findings)
    return {
        "result": "invalid" if has_error else "valid",
        "reason": findings[0].get("detail", "validation failed") if has_error else "validation passed",
        "feedback": retry_feedback_from_findings(findings),
        "findings": findings,
        "retry_hint": make_retry_hint(findings),
    }


def make_retry_hint(findings: list[dict[str, Any]]) -> dict[str, Any]:
    if not findings:
        return {"retryable": False, "suggested_action": "continue", "reason_code": "none", "details": {}}
    priority = {"sql_generation_failed": 0, "mysql_dialect_error": 1, "missing_table": 2, "missing_column": 3, "intent_mismatch": 4, "result_shape_mismatch": 5, "invalid_join_plan": 6, "postcheck_failed": 7, "mart_summary_bias": 8, "mart_grain_missing": 9, "execution_error": 10}
    ranked_findings = sorted(findings, key=lambda item: priority.get(str(item.get("category")), 99))
    primary = ranked_findings[0]
    category = primary.get("category", "validation_failed")
    suggested_action = primary.get("suggested_action") or {
        "sql_generation_failed": "regenerate_sql",
        "mysql_dialect_error": "rewrite_mysql_dialect",
        "missing_table": "reselect_table",
        "missing_column": "reselect_column",
        "intent_mismatch": "rewrite_for_metric",
        "result_shape_mismatch": "rewrite_result_shape",
        "invalid_join_plan": "rebuild_join_plan",
        "postcheck_failed": "repair_postcheck",
        "mart_summary_bias": "rewrite_row_preserving_mart",
        "mart_grain_missing": "clarify_mart_grain",
    }.get(category, "fix_sql")
    return {
        "retryable": bool(primary.get("retryable", False)),
        "suggested_action": suggested_action,
        "reason_code": category,
        "details": {
            "categories": [item.get("category") for item in ranked_findings],
            "messages": [item.get("detail") for item in ranked_findings],
            "primary_code": primary.get("code") or category,
            "generation_reason_code": (primary.get("code") or category) if category == "sql_generation_failed" else None,
            "primary_details": dict(primary.get("details") or {}),
        },
    }


def retry_feedback_from_findings(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return ""
    return "검증 실패 유형을 반영해 SQL을 다시 작성하세요. " + " / ".join(str(item.get("detail", "")) for item in findings[:3])


def _extract_cte_names(sql: str) -> set[str]:
    cte_names: set[str] = set()
    sql_lower = _normalized_sql(sql).lower()
    for match in re.finditer(r"(?:with|,)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+as\s*\(", sql_lower):
        cte_names.add(match.group(1))
    return cte_names


def _extract_table_columns(table_info: Any) -> set[str]:
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


def _dedupe_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in findings:
        key = (str(item.get("category", "")), str(item.get("detail", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped

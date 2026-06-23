from __future__ import annotations

import re
from typing import Any

from DATA_Analyst_Assistant_Agent.agents.sql.self_check import mysql_dialect_error
from DATA_Analyst_Assistant_Agent.agents.sql.sql_agent.tool.planner_support import extract_schema_json, schema_tables


def _normalized_sql(sql: str) -> str:
    return (sql or "").replace("```sql", "").replace("```", "").strip()


def _normalized_upper_sql(sql: str) -> str:
    return _normalized_sql(sql).upper()


def build_intent_contract(plan: dict[str, Any]) -> dict[str, Any]:
    contract = dict(plan.get("validation_contract") or {})
    if "expected_result_shape" not in contract:
        contract["expected_result_shape"] = plan.get("expected_result_shape") or "table_preview"
    contract.setdefault("required_columns", list(plan.get("required_columns") or []))
    contract.setdefault("required_aggregations", list(plan.get("required_aggregations") or []))
    contract.setdefault("required_tables", list(plan.get("selected_join_tables") or []))
    contract.setdefault("dimensions", list(plan.get("dimensions") or []))
    contract.setdefault("target_metric", plan.get("target_metric") or "")
    contract.setdefault("expected_aliases", [])
    contract.setdefault("target_table", None)
    return contract


def validate_sql_dialect_and_route(plan: dict[str, Any], sql_draft: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    sql = sql_draft.get("sql") or ""
    route_kind = plan.get("route_kind", "simple")
    sql_type = sql_draft.get("sql_type", "select")

    dialect_issue = mysql_dialect_error(sql)
    if dialect_issue:
        findings.append(
            {
                "category": "mysql_dialect_error",
                "severity": "error",
                "retryable": True,
                "detail": dialect_issue,
            }
        )

    if route_kind == "simple" and sql_type != "select":
        findings.append(
            {
                "category": "route_kind_mismatch",
                "severity": "error",
                "retryable": True,
                "detail": "simple 경로에서는 조회 SQL만 허용됩니다.",
            }
        )
    if route_kind == "comprehensive" and sql_type == "select":
        findings.append(
            {
                "category": "route_kind_mismatch",
                "severity": "error",
                "retryable": True,
                "detail": "comprehensive 경로에서는 datamart 생성 SQL이 필요합니다.",
            }
        )
    return findings


def validate_sql_intent(plan: dict[str, Any], sql_draft: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    sql = _normalized_upper_sql(sql_draft.get("sql") or "")
    contract = build_intent_contract(plan)
    if contract.get("expected_result_shape") == "datamart_creation":
        return findings

    for agg in contract.get("required_aggregations", []):
        if agg.upper() not in sql:
            findings.append(
                {
                    "category": "intent_mismatch",
                    "severity": "error",
                    "retryable": True,
                    "detail": f"질문 의도상 필요한 집계 함수 {agg} 가 SQL에 없습니다.",
                }
            )

    dimensions = [str(d) for d in contract.get("dimensions", []) if d]
    if contract.get("expected_result_shape") == "grouped_aggregate" and dimensions and "GROUP BY" not in sql:
        findings.append(
            {
                "category": "result_shape_mismatch",
                "severity": "error",
                "retryable": True,
                "detail": "그룹 집계 질문인데 GROUP BY가 없습니다.",
            }
        )

    required_tables = [str(t) for t in contract.get("required_tables", []) if t]
    source_tables = [str(t) for t in sql_draft.get("source_tables", []) if t]
    sql_lower = _normalized_sql(sql_draft.get("sql") or "").lower()
    for table_name in required_tables:
        if table_name not in source_tables and table_name.lower() not in sql_lower:
            findings.append(
                {
                    "category": "invalid_join_plan",
                    "severity": "error",
                    "retryable": True,
                    "detail": f"planner가 선택한 핵심 테이블 {table_name} 이 SQL에 반영되지 않았습니다.",
                }
            )

    return findings


def validate_sql_identifiers(plan: dict[str, Any], sql_draft: dict[str, Any], schema_text: str) -> list[dict[str, Any]]:
    schema_json = extract_schema_json(schema_text)
    tables = schema_tables(schema_json) if schema_json else {}
    findings: list[dict[str, Any]] = []
    if not isinstance(tables, dict) or not tables:
        return findings

    available_tables = set(str(name) for name in tables.keys())
    available_columns = {
        str(table_name): _extract_table_columns(table_info)
        for table_name, table_info in tables.items()
    }
    sql = _normalized_sql(sql_draft.get("sql") or "")
    source_tables = [str(t) for t in sql_draft.get("source_tables", []) if t]
    required_tables = [str(t) for t in build_intent_contract(plan).get("required_tables", []) if t]
    candidate_tables = list(dict.fromkeys(source_tables + required_tables))

    for table_name in candidate_tables:
        bare_name = table_name.split(".")[-1]
        if bare_name not in available_tables and table_name not in available_tables:
            findings.append(
                {
                    "category": "missing_table",
                    "severity": "error",
                    "retryable": True,
                    "detail": f"테이블 {table_name} 이(가) 제공된 스키마에 없습니다.",
                }
            )

    for column_name in [str(c) for c in sql_draft.get("columns_used", []) if c]:
        bare_column_name = column_name.split(".")[-1]
        if not any(bare_column_name in cols for cols in available_columns.values()):
            findings.append(
                {
                    "category": "missing_column",
                    "severity": "error",
                    "retryable": True,
                    "detail": f"컬럼 {column_name} 이(가) 제공된 스키마에 없습니다.",
                }
            )

    route_kind = plan.get("route_kind")
    target_table = sql_draft.get("target_table")
    if route_kind == "comprehensive" and target_table and "." not in target_table:
        findings.append(
            {
                "category": "missing_table",
                "severity": "error",
                "retryable": True,
                "detail": "comprehensive 경로의 target_table 은 schema-qualified 형식이어야 합니다.",
            }
        )

    sql_lower = sql.lower()
    suspicious_refs = re.findall(r"(?:from|join|into|table)\s+([a-zA-Z_][a-zA-Z0-9_\.]*)", sql_lower)
    for ref in suspicious_refs:
        bare_name = ref.split(".")[-1]
        if bare_name not in available_tables and not ref.startswith("analytics."):
            findings.append(
                {
                    "category": "missing_table",
                    "severity": "error",
                    "retryable": True,
                    "detail": f"SQL이 참조한 테이블 {ref} 이(가) 제공된 스키마에 없습니다.",
                }
            )
    return _dedupe_findings(findings)


def validate_result_shape(
    plan: dict[str, Any],
    sql_draft: dict[str, Any],
    sql_result: Any,
    row_count: int,
    postcheck_result: Any = None,
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    contract = build_intent_contract(plan)
    expected = contract.get("expected_result_shape")
    rows = list(sql_result or [])
    first_row = rows[0] if rows else None
    aliases = [str(alias) for alias in contract.get("expected_aliases", []) if alias]
    first_row_mapping = dict(first_row._mapping) if hasattr(first_row, "_mapping") else {}

    if expected == "single_scalar":
        if row_count != 1:
            findings.append(
                {
                    "category": "result_shape_mismatch",
                    "severity": "error",
                    "retryable": True,
                    "detail": f"단일 집계 결과는 1행이어야 하는데 {row_count}행입니다.",
                }
            )
        if aliases and first_row_mapping and not any(alias in first_row_mapping for alias in aliases):
            findings.append(
                {
                    "category": "result_shape_mismatch",
                    "severity": "warning",
                    "retryable": True,
                    "detail": f"기대 alias {aliases} 가 결과 컬럼에 없습니다.",
                }
            )
        if first_row is None:
            findings.append(
                {
                    "category": "empty_result",
                    "severity": "error",
                    "retryable": True,
                    "detail": "단일 집계 결과가 비어 있습니다.",
                }
            )
        elif isinstance(first_row, (tuple, list)) and all(value is None for value in first_row):
            findings.append(
                {
                    "category": "null_only_metric",
                    "severity": "warning",
                    "retryable": True,
                    "detail": "집계 결과가 모두 NULL 입니다.",
                }
            )

    if expected == "grouped_aggregate" and row_count <= 0:
        findings.append(
            {
                "category": "empty_result",
                "severity": "error",
                "retryable": True,
                "detail": "그룹 집계 결과가 비어 있습니다.",
            }
        )

    if expected == "datamart_creation":
        target_table = sql_draft.get("target_table")
        if not target_table:
            findings.append(
                {
                    "category": "postcheck_failed",
                    "severity": "error",
                    "retryable": True,
                    "detail": "datamart 생성인데 target_table 이 없습니다.",
                }
            )
        if postcheck_result is None or postcheck_result == []:
            findings.append(
                {
                    "category": "postcheck_failed",
                    "severity": "warning",
                    "retryable": True,
                    "detail": "datamart postcheck 결과가 없습니다.",
                }
            )
    return findings


def summarize_validation(findings: list[dict[str, Any]]) -> dict[str, Any]:
    has_error = any(item.get("severity") == "error" for item in findings)
    if has_error:
        result = "invalid"
        reason = findings[0].get("detail", "validation failed")
    else:
        result = "valid"
        reason = "validation passed"
    retry_hint = make_retry_hint(findings)
    feedback = retry_feedback_from_findings(findings)
    return {
        "result": result,
        "reason": reason,
        "feedback": feedback,
        "findings": findings,
        "retry_hint": retry_hint,
    }


def make_retry_hint(findings: list[dict[str, Any]]) -> dict[str, Any]:
    if not findings:
        return {"retryable": False, "suggested_action": "continue", "reason_code": "none", "details": {}}
    priority = {
        "mysql_dialect_error": 0,
        "missing_table": 1,
        "missing_column": 2,
        "intent_mismatch": 3,
        "result_shape_mismatch": 4,
        "invalid_join_plan": 5,
        "postcheck_failed": 6,
        "execution_error": 7,
    }
    ranked_findings = sorted(findings, key=lambda item: priority.get(str(item.get("category")), 99))
    first = ranked_findings[0]
    category = first.get("category", "validation_failed")
    suggested_action = {
        "mysql_dialect_error": "rewrite_mysql_dialect",
        "missing_table": "reselect_table",
        "missing_column": "reselect_column",
        "intent_mismatch": "rewrite_for_metric",
        "result_shape_mismatch": "rewrite_result_shape",
        "invalid_join_plan": "rebuild_join_plan",
        "postcheck_failed": "repair_postcheck",
    }.get(category, "fix_sql")
    return {
        "retryable": any(item.get("retryable", False) for item in findings),
        "suggested_action": suggested_action,
        "reason_code": category,
        "details": {
            "categories": [item.get("category") for item in ranked_findings],
            "messages": [item.get("detail") for item in ranked_findings],
        },
    }


def retry_feedback_from_findings(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return ""
    details = " / ".join(str(item.get("detail", "")) for item in findings[:3])
    return f"검증 실패 유형을 반영해 SQL을 다시 작성하세요. {details}"


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

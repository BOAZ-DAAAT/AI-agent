from __future__ import annotations

import re
from typing import Any

from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import (
    extract_schema_json,
    normalize_sql_draft_columns,
    require_route_kind,
    schema_tables,
)
from DATA_Analyst_Assistant_Agent.agents.sql.self_check import mysql_dialect_error
from DATA_Analyst_Assistant_Agent.agents.sql.sql_features import (
    SQLFeatures,
    bare_identifier,
    canonical_aggregation,
    canonical_identifier,
    extract_sql_features,
)
from DATA_Analyst_Assistant_Agent.agents.sql.sql_text import split_sql_statements


def _normalized_sql(sql: str) -> str:
    return (sql or "").replace("```sql", "").replace("```", "").strip()


def _normalized_upper_sql(sql: str) -> str:
    return _normalized_sql(sql).upper()


def _normalized_mart_policy(plan: dict[str, Any], mart_design: dict[str, Any]) -> str:
    raw_policy = (
        (mart_design or {}).get("aggregation_policy")
        or build_intent_contract(plan).get("mart_policy")
        or ""
    )
    policy = str(raw_policy).strip().lower()
    return {
        "prefer_row_preserving": "prefer_row_preserving",
        "row_preserving": "prefer_row_preserving",
        "preserve_common_grain": "preserve_common_grain",
        "aggregate_if_justified": "aggregate_if_justified",
        "aggregate_to_common_grain": "aggregate_to_common_grain",
    }.get(policy, policy)


def _has_aggregation_justification(mart_design: dict[str, Any], sql_draft: dict[str, Any]) -> bool:
    text = " ".join(
        str(value or "")
        for value in (
            (mart_design or {}).get("aggregation_rationale"),
            (mart_design or {}).get("design_reasoning"),
            (sql_draft or {}).get("reasoning"),
        )
    ).strip()
    if not text:
        return False
    lowered = text.lower()
    markers = (
        "justify",
        "because",
        "exception",
        "aggregate",
        "deduplicate",
        "latest",
        "snapshot",
        "예외",
        "집계",
        "중복",
        "최신",
        "스냅샷",
        "원본",
    )
    return any(marker in lowered for marker in markers)


def _parse_sql_features(sql: str) -> tuple[SQLFeatures | None, list[dict[str, Any]]]:
    try:
        return extract_sql_features(sql), []
    except Exception as exc:
        return None, [{
            "category": "sql_parse_error",
            "severity": "error",
            "retryable": True,
            "detail": f"SQL 구조를 파싱하지 못했습니다: {exc}",
        }]


def _has_required_aggregation(features: SQLFeatures, aggregation: Any) -> bool:
    required = canonical_aggregation(aggregation)
    if not required:
        return True
    return required in features.aggregations or required in features.window_functions


# TRIM(BOTH x FROM y) / EXTRACT(YEAR FROM col) / SUBSTRING(str FROM pos FOR len) 등은
# ANSI SQL 함수 인자 문법으로 FROM을 쓴다 — 테이블 참조가 아니다. 이 함수 호출 구간을
# 통째로 마스킹해 뒤따르는 테이블-참조 정규식이 인자 안의 FROM/컬럼명을 오인하지 않게 한다.
_FROM_ARG_FUNCTION_START = re.compile(r"\b(?:trim|extract|substring|position|overlay)\s*\(")


def _mask_from_arg_functions(sql_lower: str) -> str:
    chars = list(sql_lower)
    for match in _FROM_ARG_FUNCTION_START.finditer(sql_lower):
        depth = 1
        idx = match.end()
        while idx < len(chars) and depth > 0:
            if chars[idx] == "(":
                depth += 1
            elif chars[idx] == ")":
                depth -= 1
            idx += 1
        for i in range(match.start(), min(idx, len(chars))):
            chars[i] = " "
    return "".join(chars)


def build_intent_contract(plan: dict[str, Any]) -> dict[str, Any]:
    route_kind = require_route_kind(plan)
    contract = dict(plan.get("validation_contract") or {})
    contract["expected_result_shape"] = (
        "datamart_creation" if route_kind == "comprehensive" else "table_preview"
    )
    contract.setdefault("required_columns", list(plan.get("required_columns") or []))
    contract.setdefault("required_aggregations", list(plan.get("required_aggregations") or []))
    contract.setdefault("required_tables", list(plan.get("selected_join_tables") or []))
    contract.setdefault("dimensions", list(plan.get("dimensions") or []))
    contract.setdefault("target_metrics", list(plan.get("target_metrics") or []))
    contract.setdefault("expected_aliases", [])
    contract.setdefault("target_table", None)
    return contract


def validate_sql_dialect_and_route(plan: dict[str, Any], sql_draft: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    sql = sql_draft.get("sql") or ""
    statements = split_sql_statements(sql)
    route_kind = require_route_kind(plan)
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
    if route_kind == "comprehensive":
        if sql_type != "create_table_as":
            findings.append({"category": "route_kind_mismatch", "severity": "error", "retryable": True, "detail": "comprehensive 경로에서는 create_table_as SQL만 허용됩니다."})
        if len(statements) != 1:
            findings.append({
                "category": "route_kind_mismatch",
                "severity": "error",
                "retryable": True,
                "detail": "comprehensive 경로에서는 단일 datamart 생성 statement만 허용됩니다.",
            })
        first_statement = _normalized_upper_sql(statements[0]) if statements else ""
        if not re.match(r"^CREATE\s+TABLE\s+.+?\s+AS\s+(?:WITH\b|SELECT\b)", first_statement, re.DOTALL):
            findings.append({
                "category": "route_kind_mismatch",
                "severity": "error",
                "retryable": True,
                "detail": "comprehensive 경로의 SQL은 CREATE TABLE ... AS SELECT 형식이어야 합니다.",
            })
    return findings


def validate_sql_intent(plan: dict[str, Any], sql_draft: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    sql = sql_draft.get("sql") or ""
    features, parse_findings = _parse_sql_features(sql)
    if parse_findings:
        return parse_findings
    assert features is not None
    contract = build_intent_contract(plan)
    # required_aggregations/required_tables는 datamart_creation(comprehensive)에도 그대로
    # 적용한다 — RFM처럼 "이 파생 집계 컬럼이 마트에 있어야 한다"는 요구를 semantic LLM
    # 판정에만 맡기지 않고 결정론적으로 먼저 걸러낸다. grouped_aggregate 전용 GROUP BY
    # 체크만 datamart_creation에서 자연히 스킵된다(아래 조건이 애초에 안 걸림).
    for agg in contract.get("required_aggregations", []):
        if not _has_required_aggregation(features, agg):
            findings.append({"category": "intent_mismatch", "severity": "warning", "retryable": True, "detail": f"Required aggregation hint {agg} was not explicitly detected in SQL."})
    dimensions = [str(d) for d in contract.get("dimensions", []) if d]
    if contract.get("expected_result_shape") == "grouped_aggregate" and dimensions and not features.top_level_group_by:
        findings.append({"category": "result_shape_mismatch", "severity": "error", "retryable": True, "detail": "그룹 집계 질문인데 GROUP BY가 없습니다."})
    required_tables = [str(t) for t in contract.get("required_tables", []) if t]
    source_tables = {
        canonical_identifier(t) for t in sql_draft.get("source_tables", []) if t
    } | features.source_tables
    for table_name in required_tables:
        table = canonical_identifier(table_name)
        if table not in source_tables and bare_identifier(table) not in {bare_identifier(t) for t in source_tables}:
            findings.append({"category": "invalid_join_plan", "severity": "warning", "retryable": True, "detail": f"planner가 선택한 핵심 테이블 {table_name} 이 SQL에 반영되지 않았습니다."})
    return findings


def _has_top_level_token(sql: str, tokens: tuple[str, ...]) -> bool:
    """subquery/서브쿼리(괄호 안)에 갇힌 집계는 최종 마트 grain을 안 건드린다 — fan-out 방지용
    자식테이블 사전집계(JOIN (SELECT ... GROUP BY ...))가 그 표준 패턴이다. 괄호 깊이 0(가장
    바깥 SELECT)에 등장하는 토큰만 "마트 전체가 요약본"이라는 증거로 센다."""
    depth = 0
    depths: list[int] = []
    for ch in sql:
        depths.append(depth)
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
    for token in tokens:
        start = 0
        while True:
            idx = sql.find(token, start)
            if idx == -1:
                break
            if depths[idx] == 0:
                return True
            start = idx + 1
    return False


def validate_datamart_reusability(plan: dict[str, Any], mart_design: dict[str, Any], sql_draft: dict[str, Any]) -> list[dict[str, Any]]:
    contract = build_intent_contract(plan)
    if contract.get("expected_result_shape") != "datamart_creation":
        return []
    findings: list[dict[str, Any]] = []
    features, parse_findings = _parse_sql_features(sql_draft.get("sql") or "")
    if parse_findings:
        return parse_findings
    assert features is not None
    has_aggregate_summary = features.top_level_group_by or bool(features.top_level_aggregations)
    policy = _normalized_mart_policy(plan, mart_design)
    grain = str((mart_design or {}).get("grain") or "")
    if not grain.strip():
        findings.append({"category": "mart_grain_missing", "severity": "error", "retryable": True, "detail": "datamart 설계에 공통 분석 grain 설명이 없습니다."})
    if policy == "preserve_common_grain" and has_aggregate_summary:
        findings.append({"category": "mart_policy_mismatch", "severity": "error", "retryable": True, "detail": "preserve_common_grain 설계인데 SQL에 집계가 포함되었습니다. 설계 계약에 맞게 SQL을 다시 생성해야 합니다."})
    if policy == "prefer_row_preserving" and has_aggregate_summary and not _has_aggregation_justification(mart_design, sql_draft):
        findings.append({"category": "mart_summary_bias", "severity": "error", "retryable": True, "detail": "row-preserving datamart가 우선인데 SQL이 정당화 없이 요약 집계를 생성했습니다."})
    return findings


def validate_sql_identifiers(plan: dict[str, Any], sql_draft: dict[str, Any], schema_text: str) -> list[dict[str, Any]]:
    sql_draft = normalize_sql_draft_columns(sql_draft)
    schema_json = extract_schema_json(schema_text)
    tables = schema_tables(schema_json) if schema_json else {}
    findings: list[dict[str, Any]] = []
    if not isinstance(tables, dict) or not tables:
        return findings
    available_tables = {canonical_identifier(name) for name in tables.keys()}
    available_bare_tables = {bare_identifier(name) for name in available_tables}
    available_columns = {str(table_name): _extract_table_columns(table_info) for table_name, table_info in tables.items()}
    sql = _normalized_sql(sql_draft.get("sql") or "")
    features, parse_findings = _parse_sql_features(sql)
    if parse_findings:
        return parse_findings
    assert features is not None
    source_tables = [str(t) for t in sql_draft.get("source_tables", []) if t]
    required_tables = [str(t) for t in build_intent_contract(plan).get("required_tables", []) if t]
    candidate_tables = list(dict.fromkeys(source_tables + required_tables))
    for table_name in candidate_tables:
        table = canonical_identifier(table_name)
        bare_name = bare_identifier(table)
        if bare_name not in available_bare_tables and table not in available_tables:
            findings.append({"category": "missing_table", "severity": "error", "retryable": False, "detail": f"테이블 {table_name} 이(가) 제공된 스키마에 없습니다."})
    for column_name in [str(c) for c in sql_draft.get("source_column_refs", []) if c]:
        bare_column_name = column_name.split(".")[-1]
        if not any(bare_column_name in cols for cols in available_columns.values()):
            findings.append({"category": "missing_column", "severity": "error", "retryable": False, "detail": f"컬럼 {column_name} 이(가) 제공된 스키마에 없습니다."})
    for ref in sorted(features.source_tables):
        bare_name = bare_identifier(ref)
        if bare_name not in available_bare_tables and ref not in available_tables:
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
            findings.append({"category": "empty_result", "severity": "error", "retryable": False, "detail": "단일 집계 결과가 비어 있습니다."})
    if expected == "grouped_aggregate" and row_count <= 0:
        findings.append({"category": "empty_result", "severity": "error", "retryable": False, "detail": "그룹 집계 결과가 비어 있습니다."})
    if expected == "datamart_creation":
        target_table = sql_draft.get("target_table")
        if not target_table:
            findings.append({"category": "postcheck_failed", "severity": "error", "retryable": False, "detail": "datamart 생성인데 target_table 이 없습니다."})
        if postcheck_result is None or postcheck_result == []:
            findings.append({"category": "postcheck_failed", "severity": "warning", "retryable": False, "detail": "datamart postcheck 결과가 없습니다."})
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
    priority = {"sql_generation_failed": 0, "sql_parse_error": 1, "mysql_dialect_error": 2, "missing_table": 3, "missing_column": 4, "intent_mismatch": 5, "result_shape_mismatch": 6, "invalid_join_plan": 7, "postcheck_failed": 8, "mart_summary_bias": 9, "mart_policy_mismatch": 10, "mart_grain_missing": 11, "execution_error": 12, "empty_result": 13}
    ranked_findings = sorted(
        findings,
        key=lambda item: (
            0 if item.get("severity") == "error" else 1,
            priority.get(str(item.get("category")), 99),
        ),
    )
    primary = ranked_findings[0]
    category = primary.get("category", "validation_failed")
    suggested_action = primary.get("suggested_action") or {
        "sql_generation_failed": "regenerate_sql",
        "sql_parse_error": "rewrite_parseable_sql",
        "mysql_dialect_error": "rewrite_mysql_dialect",
        "route_kind_mismatch": "repair_sql_type",
        "missing_table": "reselect_table",
        "missing_column": "reselect_column",
        "intent_mismatch": "rewrite_for_metric",
        "result_shape_mismatch": "rewrite_result_shape",
        "invalid_join_plan": "rebuild_join_plan",
        "postcheck_failed": "repair_postcheck",
        "mart_summary_bias": "rewrite_row_preserving_mart",
        "mart_policy_mismatch": "rewrite_for_mart_policy",
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
    error_findings = [item for item in findings if item.get("severity") == "error"]
    if not error_findings:
        return ""
    return "Rewrite SQL to address validation errors. " + " / ".join(str(item.get("detail", "")) for item in error_findings[:3])


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

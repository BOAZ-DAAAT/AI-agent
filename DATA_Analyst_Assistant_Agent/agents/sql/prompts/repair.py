"""유효한 SQLDraft의 국소 오류만 수정하는 최소 repair 프롬프트."""

from __future__ import annotations

import json
import re
from typing import Any

from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import extract_schema_json, schema_tables
from DATA_Analyst_Assistant_Agent.agents.sql.validation_contract import build_intent_contract


def _referenced_tables(state: dict[str, Any], draft: dict[str, Any]) -> list[str]:
    candidates = [str(item) for item in draft.get("source_tables", []) if item]
    sql_parts = [draft.get("sql"), draft.get("precheck_sql"), draft.get("postcheck_sql"), state.get("failed_statement_sql")]
    for sql in sql_parts:
        for match in re.findall(
            r"(?:from|join|into|table)\s+`?([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?)`?",
            str(sql or ""),
            flags=re.IGNORECASE,
        ):
            candidates.append(match)
    result: list[str] = []
    for candidate in candidates:
        bare_name = candidate.strip("`").split(".")[-1]
        if bare_name and bare_name not in result:
            result.append(bare_name)
    return result


def _compact_related_schema(state: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    tables = schema_tables(extract_schema_json(str(state.get("schema_text") or "")))
    if not isinstance(tables, dict):
        return {"tables": {}}
    compact: dict[str, Any] = {}
    for table_name in _referenced_tables(state, draft):
        table = tables.get(table_name)
        if not isinstance(table, dict):
            continue
        columns: list[dict[str, str]] = []
        raw_columns = table.get("columns") or []
        if isinstance(raw_columns, dict):
            raw_columns = [{"name": name, **(value if isinstance(value, dict) else {})} for name, value in raw_columns.items()]
        for column in raw_columns:
            if isinstance(column, str):
                columns.append({"name": column})
            elif isinstance(column, dict) and column.get("name"):
                item = {"name": str(column["name"])}
                if column.get("type"):
                    item["type"] = str(column["type"])
                columns.append(item)
        compact[table_name] = {"columns": columns}
    return {"tables": compact}


def _repair_contract(state: dict[str, Any], route_kind: str, previous_sql_draft: dict[str, Any]) -> dict[str, Any]:
    plan = state.get("plan") or {}
    if route_kind == "simple":
        intent = build_intent_contract(plan)
        return {
            "sql_type": "select",
            "required_tables": intent.get("required_tables", []),
            "required_columns": intent.get("required_columns", []),
            "required_aggregations": intent.get("required_aggregations", []),
            "dimensions": intent.get("dimensions", []),
        }
    design = state.get("mart_design") or {}
    return {
        "sql_type": "create_table_as",
        "grain": design.get("grain"),
        "grain_columns": design.get("grain_columns", []),
        "output_column_plan": design.get("column_plan", []),
        "aggregation_policy": design.get("aggregation_policy"),
        "target_table": previous_sql_draft.get("target_table"),
    }


def repair_sql_prompt(state: dict[str, Any], previous_sql_draft: dict[str, Any]) -> str:
    """전체 계획/정합성 컨텍스트를 제외한 SQL repair 전용 프롬프트를 만든다."""
    route_kind = str((state.get("plan") or {}).get("route_kind") or "").strip().lower()
    statement_results = [
        {"index": item.get("index"), "sql": item.get("sql"), "row_count": item.get("row_count"), "columns": item.get("columns", [])}
        for item in list(state.get("statement_results") or [])
    ]
    failure_context = {
        "validation_findings": state.get("validation_findings") or [],
        "error": state.get("error") or "",
        "failed_component": state.get("failed_sql_component"),
        "failed_statement_index": state.get("failed_statement_index"),
        "failed_statement_sql": state.get("failed_statement_sql") or "",
        "execution_error_info": state.get("execution_error_info") or {},
        "successful_statements": statement_results,
        "retry_hint": state.get("retry_hint") or {},
    }
    return f"""
너는 MySQL SQL repair 전담 작성기다. 아래의 유효한 이전 SQLDraft에서 보고된 국소 오류만 수정한다.

이전 SQLDraft:
{json.dumps(previous_sql_draft, ensure_ascii=False, indent=2)}

정확한 실패 컨텍스트:
{json.dumps(failure_context, ensure_ascii=False, indent=2)}

관련 테이블 스키마:
{json.dumps(_compact_related_schema(state, previous_sql_draft), ensure_ascii=False, indent=2)}

유지해야 할 최소 계약:
{json.dumps(_repair_contract(state, route_kind, previous_sql_draft), ensure_ascii=False, indent=2)}

repair 범위:
- 사용자 질문에 없던 필터, 조건, 지표, 테이블을 임의로 추가하지 말 것
- SQL 유형({previous_sql_draft.get('sql_type')})을 변경하지 말 것
- 실패 component가 precheck 또는 postcheck이면 main sql을 변경하지 말 것
- simple 다중 statement에서는 성공한 statement를 유지하고 실패 statement만 우선 수정할 것
- 전체 SQLDraft를 반환하되 실패와 무관한 필드는 가능한 한 유지할 것
- MySQL 문법만 사용하고 반드시 JSON object만 출력할 것

출력 형식은 이전 SQLDraft와 동일하다.
""".strip()

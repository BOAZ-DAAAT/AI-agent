from __future__ import annotations

import json
import re
from typing import Any, Optional

from DATA_Analyst_Assistant_Agent.agents.sql._runtime import ALLOWED_MART_SCHEMA, clean_sql, get_llm
from DATA_Analyst_Assistant_Agent.agents.sql.sql_text import extract_sql_aliases
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState, SQLDraft


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
    if state.get("failed_statement_index") is not None:
        feedback_parts.append(f"직전 실패 statement 번호: {int(state['failed_statement_index']) + 1}")
    if str(state.get("failed_statement_sql") or "").strip():
        feedback_parts.append(f"직전 실패 statement SQL: {state['failed_statement_sql']}")
    statement_results = list(state.get("statement_results") or [])
    if statement_results:
        successful = [
            {
                "index": item.get("index"),
                "row_count": item.get("row_count"),
                "sql": item.get("sql"),
            }
            for item in statement_results[:3]
        ]
        feedback_parts.append(f"직전 성공/부분성공 statement 요약: {successful}")
    return "\n".join(feedback_parts)


def try_llm_json(prompt: str) -> Optional[str]:
    try:
        return get_llm().invoke(prompt).content
    except Exception:
        return None


def empty_sql_draft(state: AgentState, *, reasoning: str) -> dict[str, Any]:
    plan = state.get("plan") or {}
    route_kind = plan.get("route_kind") or (
        "comprehensive" if plan.get("task_type") == "data_mart_build" else "simple"
    )
    target_table = None
    if route_kind == "comprehensive":
        mart_name = (state.get("mart_design") or {}).get("mart_name") or plan.get("mart_name")
        if mart_name:
            target_table = qualify_target_table(str(mart_name))
    return SQLDraft(
        sql="",
        sql_type="create_table_as" if route_kind == "comprehensive" else "select",
        target_table=target_table,
        source_tables=[],
        source_column_refs=[],
        derived_columns=[],
        output_columns=[],
        business_grain=(state.get("mart_design") or {}).get("grain") or plan.get("grain"),
        precheck_sql=None,
        postcheck_sql=None,
        reasoning=reasoning,
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
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, dict):
        for key in ("column_name", "name", "col", "column", "expression"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return str(entry)


def normalize_mart_column_lists(design: dict[str, Any]) -> dict[str, Any]:
    for field in ("key_columns", "measure_columns", "dimension_columns", "source_tables"):
        value = design.get(field)
        if isinstance(value, list):
            design[field] = [_coerce_column_name(item) for item in value if item not in (None, "")]
    return design


def normalize_generated_sql(
    parsed: dict[str, Any],
    route_kind: str,
    *,
    default_business_grain: str | None = None,
) -> dict[str, Any]:
    normalized = normalize_sql_draft_columns(parsed)
    normalized["sql"] = clean_sql(normalized.get("sql", ""))
    normalized["target_table"] = qualify_target_table(normalized.get("target_table"))
    if normalized.get("precheck_sql"):
        normalized["precheck_sql"] = clean_sql(normalized["precheck_sql"])
    postcheck_sql = normalized.get("postcheck_sql")
    if postcheck_sql:
        normalized["postcheck_sql"] = normalize_postcheck_sql(postcheck_sql, normalized.get("target_table"))
    normalized.setdefault("sql_type", "create_table_as" if route_kind == "comprehensive" else "select")
    normalized.setdefault("source_tables", [])
    normalized.setdefault("source_column_refs", [])
    normalized.setdefault("derived_columns", [])
    normalized.setdefault("output_columns", [])
    normalized.setdefault("business_grain", default_business_grain)
    normalized.setdefault("reasoning", "")
    if route_kind == "comprehensive" and normalized["sql_type"] == "select":
        raise ValueError("comprehensive route requires datamart SQL")
    if route_kind == "simple" and normalized["sql_type"] != "select":
        raise ValueError("simple route requires select SQL")
    return normalized


def normalize_sql_draft_columns(parsed: dict[str, Any]) -> dict[str, Any]:
    """신·구 SQLDraft 컬럼 계약을 입력을 변경하지 않고 정규화한다."""
    normalized = dict(parsed)
    legacy_columns_used = "source_column_refs" not in normalized and "columns_used" in normalized
    if legacy_columns_used:
        aliases = extract_sql_aliases(str(normalized.get("sql") or ""))
        normalized["source_column_refs"] = _normalize_legacy_source_column_refs(
            normalized.get("columns_used"),
            aliases,
        )
        normalized["derived_columns"] = []
        normalized["output_columns"] = []
    normalized.pop("columns_used", None)
    return normalized


def _normalize_legacy_source_column_refs(value: Any, aliases: set[str]) -> list[str]:
    if not isinstance(value, list):
        return []
    refs: list[str] = []
    seen: set[str] = set()
    for entry in value:
        reference = str(entry or "").strip().replace("`", "")
        if not reference:
            continue
        is_bare_identifier = "." not in reference and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", reference)
        if is_bare_identifier and reference.casefold() in aliases:
            continue
        key = reference.casefold()
        if key not in seen:
            seen.add(key)
            refs.append(reference)
    return refs

"""Context nodes: schema/integrity metadata load and scoped integrity refresh."""

from __future__ import annotations

import json
from typing import Any

from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState
from DATA_Analyst_Assistant_Agent.agents.sql.validator.integrity_loader import (
    compact_integrity_summary_text,
    load_integrity_json,
    load_schema_catalog_text,
    load_scoped_schema_text,
    load_scoped_integrity_text,
    preload_backend_integrity_summary,
    refresh_backend_integrity_summary,
)


def load_context(state: AgentState):
    return {
        "schema_text": load_schema_catalog_text(),
        "integrity_text": compact_integrity_summary_text(load_integrity_json()),
    }


def preplan_integrity_gate(state: AgentState):
    dataset_name = state.get("integrity_dataset_name") or state.get("datasource_id") or "default"
    preload = preload_backend_integrity_summary(str(dataset_name))
    update = {"integrity_preplan": preload}
    if preload.get("status") == "prefetched" and preload.get("integrity_text"):
        update["integrity_text"] = preload["integrity_text"]
    return update


def _as_table_list(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        candidates = [value]
    elif isinstance(value, (list, tuple, set)):
        candidates = list(value)
    else:
        return []

    tables: list[str] = []
    for candidate in candidates:
        normalized = str(candidate or "").strip().strip("`")
        if not normalized:
            continue
        normalized = normalized.split(".")[-1]
        if normalized not in tables:
            tables.append(normalized)
    return tables


def _planned_tables(plan: dict[str, Any]) -> list[str]:
    tables: list[str] = []
    for key in ("selected_join_tables", "relevant_tables", "candidate_tables"):
        for table in _as_table_list(plan.get(key)):
            if table not in tables:
                tables.append(table)
    return tables


def _first_selected_table_list(plan: dict[str, Any]) -> list[str]:
    for key in ("selected_join_tables", "relevant_tables", "candidate_tables"):
        tables = _as_table_list(plan.get(key))
        if tables:
            return tables
    return []


def _schema_table_names(schema_text: str) -> list[str]:
    try:
        schema = json.loads(schema_text)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(schema, dict):
        return []
    tables = schema.get("tables") if isinstance(schema.get("tables"), dict) else schema
    return list(tables) if isinstance(tables, dict) else []


def refresh_schema_context(state: AgentState):
    """계획에서 선택한 테이블의 상세 스키마로 프롬프트 컨텍스트를 교체한다."""
    tables = _first_selected_table_list(state.get("plan") or {})
    if not tables:
        return {
            "schema_text": state.get("schema_text", ""),
            "schema_refresh": {
                "status": "skipped",
                "candidate_tables": [],
                "applied_tables": [],
                "reason": "no_tables",
            }
        }

    scoped_schema_text = load_scoped_schema_text(tables)
    if not scoped_schema_text:
        return {
            "schema_text": state.get("schema_text", ""),
            "schema_refresh": {
                "status": "skipped",
                "candidate_tables": tables,
                "applied_tables": [],
                "reason": "no_valid_tables",
            }
        }

    applied_tables = _schema_table_names(scoped_schema_text)
    return {
        "schema_text": scoped_schema_text,
        "schema_refresh": {
            "status": "refreshed",
            "candidate_tables": tables,
            "applied_tables": applied_tables,
            "reason": "",
        },
    }


def refresh_integrity_context(state: AgentState):
    """Refresh prompt-facing integrity context for tables selected by planning.

    The backend integrity service is optional. If absent or failing, this node is
    a no-op for prompt context and records only small refresh metadata.
    """
    plan = state.get("plan") or {}
    tables = _planned_tables(plan)
    if not tables:
        return {"integrity_refresh": {"status": "skipped", "reason": "no_tables", "tables": []}}

    dataset_name = (
        state.get("integrity_dataset_name")
        or state.get("datasource_id")
        or "default"
    )
    result = refresh_backend_integrity_summary(
        str(dataset_name),
        tables,
        wait_timeout_s=0.0,
        metadata={"source": "sql_agent"},
    )
    refresh_meta = {key: value for key, value in result.items() if key != "integrity_text"}
    update: dict[str, Any] = {"integrity_refresh": refresh_meta}
    if result.get("status") == "refreshed" and result.get("ready"):
        update["integrity_text"] = result.get("integrity_text", "")
    else:
        # 백엔드 정합성 서비스가 없을 때(로컬/오프라인) → 미리 생성해둔 정적
        # db_integrity_result.json 을 planned tables 로 스코핑해 읽는다.
        # 실패 검사만(fail_only) + 캡 없음(max_lines=None) → 관련 테이블의 문제만 최소로.
        update["integrity_text"] = load_scoped_integrity_text(tables)
        update["integrity_refresh"] = {**refresh_meta, "local_snapshot_used": True, "scoped_tables": tables}
    return update

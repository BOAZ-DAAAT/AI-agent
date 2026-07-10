"""Context nodes: schema/integrity metadata load and scoped integrity refresh."""

from __future__ import annotations

from typing import Any

from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState
from DATA_Analyst_Assistant_Agent.agents.sql.validator.integrity_loader import (
    load_all_metadata,
    preload_backend_integrity_summary,
    refresh_backend_integrity_summary,
)


def load_context(state: AgentState):
    metadata = load_all_metadata()
    return {
        "schema_text": metadata["schema_text"],
        "integrity_text": metadata["integrity_text"],
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
    return update

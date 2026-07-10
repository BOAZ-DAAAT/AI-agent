"""design_mart 노드: 데이터마트 설계(task_type=data_mart_build 일 때만)."""

from __future__ import annotations

from typing import Any

from DATA_Analyst_Assistant_Agent.agents.sql import prompts
from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import (
    default_mart_design,
    normalize_mart_column_lists,
    try_llm_json,
)
from DATA_Analyst_Assistant_Agent.agents.sql._runtime import safe_json_parse
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState, MartDesign


def _normalize_column_list(values: Any) -> list[str]:
    if not values:
        return []
    if not isinstance(values, list):
        values = [values]

    normalized: list[str] = []
    for item in values:
        if isinstance(item, str):
            candidate = item.strip()
        elif isinstance(item, dict):
            candidate = str(
                item.get("column_name")
                or item.get("name")
                or item.get("column")
                or item.get("field")
                or ""
            ).strip()
        else:
            candidate = str(item).strip()
        if candidate and candidate not in normalized:
            normalized.append(candidate)
    return normalized


def _normalize_mart_design_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    for key in ("key_columns", "measure_columns", "dimension_columns", "source_tables"):
        normalized[key] = _normalize_column_list(normalized.get(key))
    return normalized


def design_mart(state: AgentState):
    if state["plan"].get("task_type") != "data_mart_build":
        return {"mart_design": {}}

    fallback = default_mart_design(state)
    response = try_llm_json(prompts.mart_design_prompt(state))
    parsed = safe_json_parse(response, fallback) if response else fallback
    merged = dict(fallback)
    merged.update({k: v for k, v in parsed.items() if v not in (None, "", [], {})})
    # LLM이 컬럼을 문자열 대신 dict({"column_name":...}) 로 주는 형식 편차를 흡수한다.
    merged = normalize_mart_column_lists(merged)
    return {"mart_design": MartDesign(**merged).model_dump()}

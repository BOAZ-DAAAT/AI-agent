"""design_mart 노드: comprehensive 경로의 데이터마트 설계."""

from __future__ import annotations

import json
from typing import Any

from DATA_Analyst_Assistant_Agent.agents.sql import prompts
from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import (
    normalize_mart_column_lists,
    require_route_kind,
    try_llm_json,
)
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
    for key in ("source_tables", "grain_columns", "deduplication_keys"):
        normalized[key] = _normalize_column_list(normalized.get(key))
    source_grains = normalized.get("source_grains")
    if isinstance(source_grains, dict):
        normalized["source_grains"] = {
            str(table).strip(): _normalize_column_list(columns)
            for table, columns in source_grains.items()
            if str(table).strip()
        }
    return normalized


def _validate_target_metric_support(design: MartDesign, plan: dict[str, Any]) -> None:
    target_metrics = list(dict.fromkeys(str(metric).strip() for metric in plan.get("target_metrics", []) if str(metric).strip()))
    supported_metrics = [item.metric_name for item in design.metric_support]
    if supported_metrics != target_metrics:
        raise ValueError(
            "metric_support는 고유 target_metrics를 같은 순서로 정확히 한 번 포함해야 합니다: "
            f"expected={target_metrics}, actual={supported_metrics}"
        )


def validate_mart_design_state(payload: dict[str, Any], plan: dict[str, Any]) -> MartDesign:
    """저장되었거나 새로 생성된 마트 설계가 현재 계약과 계획을 만족하는지 확인한다."""
    design = MartDesign(**_normalize_mart_design_payload(payload))
    _validate_target_metric_support(design, plan)
    return design


def _mart_design_failure(*, reason_code: str, detail: str, retryable: bool) -> dict[str, Any]:
    suggested_action = "redesign_mart" if retryable else "stop_and_surface_error"
    finding = {
        "category": "sql_mart_design_failed",
        "severity": "error",
        "retryable": retryable,
        "detail": detail,
        "message": detail,
        "source": "sql_mart_designer",
        "code": reason_code,
        "suggested_action": suggested_action,
        "details": {"reason_code": reason_code},
    }
    retry_hint = {
        "retryable": retryable,
        "suggested_action": suggested_action,
        "reason_code": "sql_mart_design_failed",
        "details": {
            "mart_design_reason_code": reason_code,
            "messages": [detail],
        },
    }
    return {
        "mart_design": {},
        "validation": {
            "result": "invalid",
            "reason": detail,
            "feedback": detail,
            "findings": [finding],
            "retry_hint": retry_hint,
        },
        "validation_findings": [finding],
        "retry_hint": retry_hint,
        "feedback": detail,
        "error": detail,
    }


def design_mart(state: AgentState):
    if require_route_kind(state["plan"]) != "comprehensive":
        return {"mart_design": {}}

    response = try_llm_json(prompts.mart_design_prompt(state))
    if not response:
        return _mart_design_failure(
            reason_code="llm_empty_response",
            detail="LLM이 mart 설계 초안을 반환하지 못했습니다.",
            retryable=True,
        )

    cleaned = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return _mart_design_failure(
            reason_code="llm_json_parse_failed",
            detail="LLM mart 설계 응답을 JSON으로 파싱하지 못했습니다.",
            retryable=True,
        )

    if not isinstance(parsed, dict):
        return _mart_design_failure(
            reason_code="llm_json_not_object",
            detail="LLM mart 설계 응답이 JSON object가 아닙니다.",
            retryable=True,
        )

    normalized = normalize_mart_column_lists(_normalize_mart_design_payload(parsed))
    try:
        design = validate_mart_design_state(normalized, state.get("plan") or {})
    except Exception as exc:
        return _mart_design_failure(
            reason_code="invalid_mart_design_payload",
            detail=f"LLM mart 설계 응답이 필수 구조를 만족하지 못했습니다: {exc}",
            retryable=True,
        )

    return {"mart_design": design.model_dump()}

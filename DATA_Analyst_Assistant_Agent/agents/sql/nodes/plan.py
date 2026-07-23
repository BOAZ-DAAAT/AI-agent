"""plan_question 노드: 사용자 질문 분석."""

from __future__ import annotations

import json
from typing import Any

from DATA_Analyst_Assistant_Agent.agents.sql import prompts
from DATA_Analyst_Assistant_Agent.agents.sql.planner_support import try_llm_json
from DATA_Analyst_Assistant_Agent.agents.sql.state import AgentState, QuestionPlan


def _list_of_str(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    raise ValueError("plan field must be a list")


def _plan_failure(*, reason_code: str, detail: str, retryable: bool) -> dict[str, Any]:
    suggested_action = "replan_question" if retryable else "stop_and_surface_error"
    finding = {
        "category": "sql_plan_failed",
        "severity": "error",
        "retryable": retryable,
        "detail": detail,
        "message": detail,
        "source": "sql_planner",
        "code": reason_code,
        "suggested_action": suggested_action,
        "details": {"reason_code": reason_code},
    }
    retry_hint = {
        "retryable": retryable,
        "suggested_action": suggested_action,
        "reason_code": "sql_plan_failed",
        "details": {
            "plan_reason_code": reason_code,
            "messages": [detail],
        },
    }
    return {
        "plan": {},
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


def _normalize_question_plan(state: AgentState, parsed: dict[str, Any]) -> dict[str, Any]:
    required_fields = {
        "route_kind",
        "question_type",
        "target_metrics",
        "analysis_entities",
        "dimensions",
        "filters",
        "candidate_tables",
        "required_aggregations",
        "reasoning",
    }
    missing_fields = sorted(required_fields.difference(parsed))
    if missing_fields:
        raise ValueError(f"question plan missing required fields: {', '.join(missing_fields)}")
    route_kind_raw = str(parsed.get("route_kind") or "").strip().lower()
    route_kind = route_kind_raw
    if route_kind not in {"simple", "comprehensive"}:
        raise ValueError(f"unsupported route_kind: {route_kind or 'empty'}")
    if state.get("required_derivations"):
        route_kind = "comprehensive"

    target_metrics = _list_of_str(parsed.get("target_metrics"))
    if route_kind == "comprehensive" and not target_metrics:
        target_metrics = ["mart_metric"]
    normalized = {
        "route_kind": route_kind,
        "question_type": str(parsed.get("question_type") or "").strip(),
        "target_metrics": target_metrics,
        "analysis_entities": _list_of_str(parsed.get("analysis_entities")),
        "dimensions": _list_of_str(parsed.get("dimensions")),
        "filters": _list_of_str(parsed.get("filters")),
        "candidate_tables": _list_of_str(parsed.get("candidate_tables")),
        "required_aggregations": _list_of_str(parsed.get("required_aggregations")),
        "reasoning": str(parsed.get("reasoning") or "").strip(),
    }
    try:
        return QuestionPlan(**normalized).model_dump()
    except Exception as exc:
        raise ValueError(f"invalid question plan payload: {exc}") from exc


def plan_question(state: AgentState):
    response = try_llm_json(prompts.plan_prompt(state))
    if not response:
        return _plan_failure(
            reason_code="llm_empty_response",
            detail="LLM이 질문 분석 계획을 반환하지 않았습니다.",
            retryable=True,
        )

    cleaned = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(cleaned)
    except Exception:
        return _plan_failure(
            reason_code="llm_json_parse_failed",
            detail="LLM planner 응답을 JSON으로 파싱하지 못했습니다.",
            retryable=True,
        )

    if not isinstance(parsed, dict):
        return _plan_failure(
            reason_code="llm_json_not_object",
            detail="LLM planner 응답이 JSON object가 아닙니다.",
            retryable=True,
        )

    try:
        normalized = _normalize_question_plan(state, parsed)
    except ValueError as exc:
        return _plan_failure(
            reason_code="invalid_question_plan",
            detail=str(exc),
            retryable=True,
        )

    return {
        "question_plan": normalized,
        "final_table_plan": {},
        "planning_stages": {"question_plan": normalized, "final_table_plan": {}},
        "plan": {},
        "mart_design": {},
        "sql_draft": {},
    }
